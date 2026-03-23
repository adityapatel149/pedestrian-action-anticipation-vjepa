import logging
import os
import random
from dataclasses import dataclass
from itertools import islice
from multiprocessing import Value
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards_list):
    num_shards = len(shards_list)
    total_size = num_shards
    return total_size, num_shards


def log_and_continue(exn):
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


class split_by_node(wds.PipelineStage):
    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size

    def run(self, src):
        if self.world_size > 1:
            yield from islice(src, self.rank, None, self.world_size)
        else:
            yield from src


class ResampledShards(IterableDataset):
    """An iterable dataset yielding a list of urls (video file paths)."""

    def __init__(self, urls, epoch, training):
        super().__init__()
        self.epoch = epoch
        self.training = training
        self.urls = np.array(urls)
        logging.info("Done initializing ResampledShards")

    def __iter__(self):
        if self.training:
            epoch = self.epoch.get_value()
            gen = torch.Generator()
            gen.manual_seed(epoch)
            yield from self.urls[torch.randperm(len(self.urls), generator=gen)]
        else:
            yield from self.urls[torch.arange(len(self.urls))]


# ----------------------------------------
# PIE-specific decode stage
# ----------------------------------------

class decode_videos_to_clips(wds.PipelineStage):
    """
    API-compatible with original pie.py decode_videos_to_clips, but:
      - Uses start_frame/stop_frame from CSV directly
      - Loads EXACT consecutive frames in [start_frame, stop_frame]
      - Ignores anticipation sampling
      - Ignores fps/fpc resampling (frames_per_clip/fps kept only for signature compatibility)

    NOTE:
      - Your CSV should already define fixed-length windows (e.g., 16 frames).
      - We still yield `anticipation_time` to keep downstream tuple spec unchanged.
    """

    def __init__(
        self,
        annotations,
        frames_per_clip=16,                  
        fps=5,                               
        transform=None,
        anticipation_time_sec=(1.0, 1.0),    # kept for compatibility; ignored
        anticipation_point=(0.25, 0.75),     # kept for compatibility; ignored
        label_keys=("cross",),
        framewise_bboxes=None,
    ):
        self.annotations = annotations
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.transform = transform
        self.anticipation_time = anticipation_time_sec
        self.anticipation_point = anticipation_point
        self.label_keys = label_keys
        self.framewise_bboxes = framewise_bboxes

    def _get_bboxes(self, video_id: str, participant_id: str, indices: np.ndarray, buffer: np.ndarray) -> np.ndarray:
        if (
            self.framewise_bboxes is None
            or video_id not in self.framewise_bboxes
            or participant_id is None
            or participant_id not in self.framewise_bboxes[video_id]
        ):
            return np.zeros((len(indices), 4), dtype=np.float32)

        fw_boxes = self.framewise_bboxes[video_id][participant_id]
        if fw_boxes is None or len(fw_boxes) == 0:
            return np.zeros((len(indices), 4), dtype=np.float32)

        fw_keys = np.array(sorted(fw_boxes.keys()), dtype=np.int64)
        bbox_seq = []
        for f in indices.astype(int):
            if int(f) in fw_boxes:
                bbox_seq.append(fw_boxes[int(f)])
            else:
                pos = np.searchsorted(fw_keys, f)
                if pos == 0:
                    nnk = int(fw_keys[0])
                elif pos == len(fw_keys):
                    nnk = int(fw_keys[-1])
                else:
                    left = int(fw_keys[pos - 1])
                    right = int(fw_keys[pos])
                    nnk = left if (f - left) <= (right - f) else right
                bbox_seq.append(fw_boxes[nnk])

        bboxes = np.asarray(bbox_seq, dtype=np.float32)

        H, W = (buffer.shape[1], buffer.shape[2])  # (T,H,W,C)
        if W > 0 and H > 0:
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0.0, float(W))
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0.0, float(H))
        else:
            bboxes = np.zeros((len(indices), 4), dtype=np.float32)

        return bboxes

    
    def _sample_indices_timebased(self, sf: int, ef: int, native_fps: float, target_fps: float, fpc: int) -> np.ndarray:
        """
        Sample fpc frames starting at sf, spaced by 1/target_fps seconds, mapped to native frame indices.
        If the CSV window is too short, we clamp/pad to ef.
        """
        sf = int(sf)
        ef = int(ef)
        fpc = int(fpc)
    
        if fpc <= 0:
            return np.zeros((0,), dtype=np.int64)
    
        # Guard rails
        if native_fps is None or native_fps <= 0:
            native_fps = 30.0  # fallback
        if target_fps is None or target_fps <= 0:
            # If fps is invalid, fall back to dense consecutive frames (best-effort)
            idx = np.arange(sf, sf + fpc, dtype=np.int64)
            return np.clip(idx, sf, ef)
    
        # Convert start frame -> start time (sec)
        start_t = sf / native_fps
    
        # Target times for each frame in the clip
        # (0, 1/fps, 2/fps, ..., (fpc-1)/fps)
        t = start_t + (np.arange(fpc, dtype=np.float32) / float(target_fps))
    
        # Map times back to native frame indices
        idx = np.rint(t * native_fps).astype(np.int64)
    
        # Clamp to the CSV window; this effectively pads with ef if we run past it
        idx = np.clip(idx, sf, ef)
        return idx

    def run(self, src):
        # Ensure `src` is an iterable of paths, even if it’s just one
        if isinstance(src, str):
            src = [src]

        for path in src:
            # Recover video_id in same form as CSV, e.g. "set01/video_0001"
            dir_name = os.path.basename(os.path.dirname(path))
            base_name = os.path.splitext(os.path.basename(path))[0]
            video_id = f"{dir_name}/{base_name}"

            if video_id not in self.annotations:
                logging.warning(
                    f"Video ID '{video_id}' not found in annotations; skipping path '{path}'"
                )
                continue

            ano = self.annotations[video_id]

            # Validate required columns
            if "start_frame" not in ano.columns or "stop_frame" not in ano.columns:
                raise KeyError("CSV must contain 'start_frame' and 'stop_frame' columns.")

            label_cols = {}
            for k in self.label_keys:
                col = f"{k}_class"
                if col not in ano.columns:
                    raise KeyError(f"Expected column '{col}' in annotations for PIE but not found.")
                label_cols[k] = ano[col].values

            # Load video
            try:
                vr = VideoReader(path, num_threads=-1, ctx=cpu(0))
                vr.seek(0)
            except Exception as e:
                logging.info(f"Encountered exception loading video {e=}")
                continue
            
            try:
                native_fps = float(vr.get_avg_fps())
            except Exception:
                native_fps = 30.0

            start_frames = ano["start_frame"].values
            stop_frames = ano["stop_frame"].values

            tte_secs = ano["time_to_event_sec"].to_numpy(dtype=float)

            for i, (sf, ef) in enumerate(zip(start_frames, stop_frames)):
                sf = int(sf)
                ef = int(ef)
                 # Per-row anticipation time in seconds
                at = float(tte_secs[i])

                # EXACT consecutive indices from CSV
                # indices = np.arange(sf, ef + 1, dtype=np.int64)
                indices = self._sample_indices_timebased(
                    sf=sf,
                    ef=ef,
                    native_fps=native_fps,
                    target_fps=float(self.fps),
                    fpc=int(self.frames_per_clip),
                )
                
                # Labels read exactly from CSV
                labels = {k: int(label_cols[k][i]) for k in self.label_keys}

                # Ped id (for bbox lookup)
                participant_id = str(ano.iloc[i].get("participant_id", None))

                try:
                    buffer = vr.get_batch(indices).asnumpy()
                except Exception as e:
                    logging.info(f"Encountered exception getting indices {e=}")
                    continue

                bboxes = self._get_bboxes(video_id, participant_id, indices, buffer)


                if self.transform is not None:
                    # Support bbox-aware transforms: transform(buffer, bboxes) -> (video, bboxes)
                    try:
                        out = self.transform(buffer, bboxes)
                    except TypeError:
                        out = self.transform(buffer)

                    if isinstance(out, tuple) and len(out) == 2:
                        buffer, bboxes = out
                    else:
                        buffer = out

                yield dict(video=buffer, bboxes=bboxes, **labels, anticipation_time=at)


# ----------------------------------------
# Annotation filtering / path building
# ----------------------------------------

def _build_paths_and_annotations(
    df: pd.DataFrame,
    base_path: str,
    exts=(".mp4", ".MP4", ".avi", ".mov", ".MOV", ".mkv", ".MKV"),
):
    """
    PIE layout:
      base_path/setXX/video_XXXX.ext

    `video_id` in CSV already includes subfolder, e.g.:
      set01/video_0001
    """
    video_paths = []
    annotations = {}

    unique_videos = list(dict.fromkeys(df["video_id"].values))

    for vid in unique_videos:
        fpath = None
        for ext in exts:
            candidate = os.path.join(base_path, vid + ext)
            if os.path.exists(candidate):
                fpath = candidate
                break

        if fpath is None:
            logging.warning(
                f"[PIE] file path not found for video_id={vid} under base_path={base_path}"
            )
            continue

        video_paths.append(fpath)
        annotations[vid] = (
            df[df["video_id"] == vid]
            .reset_index(drop=True)
        )

    return video_paths, annotations


def filter_annotations(
    base_path: str,
    train_annotations_path: str,
    val_annotations_path: str,
    label_keys=("cross",),
):
    """
    Requires columns: video_id, start_frame, stop_frame, and {label}_class per label key.
    """
    tdf = pd.read_csv(train_annotations_path)
    vdf = pd.read_csv(val_annotations_path)

    def action_tuple_rows(df):
        return list(zip(*[df[f"{k}_class"].values for k in label_keys]))

    # encoders (stable ordering)
    per_label_values = {k: sorted(set(tdf[f"{k}_class"].astype(int).tolist())) for k in label_keys}
    label_encoders = {k: {orig: i for i, orig in enumerate(per_label_values[k])} for k in label_keys}

    val_label_sets = {
        k: set(label_encoders[k][int(x)] for x in vdf[f"{k}_class"].values) for k in label_keys
    }

    train_annotations = _build_paths_and_annotations(tdf, base_path)
    val_annotations = _build_paths_and_annotations(vdf, base_path)

    return dict(
        label_encoders=label_encoders,
        val_label_sets=val_label_sets,
        train=train_annotations,   # (paths, {video_id: df})
        val=val_annotations,
        label_keys=tuple(label_keys),
    )


def nested_defaultdict():
    return defaultdict(dict)


def load_framewise_bboxes(csv_path):
    df = pd.read_csv(
        csv_path,
        dtype={
            "video_id": "string",
            "participant_id": "string",
            "frame": "int32",
            "x1": "float32",
            "y1": "float32",
            "x2": "float32",
            "y2": "float32",
        }
    )

    required_cols = {"video_id", "frame", "participant_id", "x1", "y1", "x2", "y2"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    framewise_bboxes = defaultdict(nested_defaultdict)
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        pid = str(row["participant_id"])
        frame = int(row["frame"])
        bbox = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]
        framewise_bboxes[vid][pid][frame] = bbox

    return framewise_bboxes


# ----------------------------------------
# WebDataset-style pipeline assembly
# ----------------------------------------

def get_video_wds_dataset(
    batch_size,
    input_shards,
    video_decoder,
    training,
    epoch=0,
    world_size=1,
    rank=0,
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    label_keys = ("cross",),
):
    assert input_shards is not None
    _, num_shards = get_dataset_size(input_shards)
    logging.info(f"Total number of shards across all data is num_shards={num_shards}")
    
    tuple_fields = ["video", "bboxes", *label_keys, "anticipation_time"]

    epoch = SharedEpoch(epoch=epoch)
    pipeline = [
        ResampledShards(input_shards, epoch=epoch, training=training),
        split_by_node(rank=rank, world_size=world_size),
        wds.split_by_worker,
        video_decoder,
        wds.to_tuple(*tuple_fields),
        wds.batched(batch_size, partial=True, collation_fn=torch.utils.data.default_collate),
    ]
    dataset = wds.DataPipeline(*pipeline)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
        pin_memory=pin_memory,
    )

    return dataset, DataInfo(dataloader=dataloader, shared_epoch=epoch)


def make_webvid(
    base_path,
    annotations_path,
    batch_size,
    transform,
    frames_per_clip=16,                 
    fps=5,                                
    num_workers=8,
    world_size=1,
    rank=0,
    anticipation_time_sec=(0.0, 0.0),    
    persistent_workers=True,
    pin_memory=True,
    training=True,
    anticipation_point=(0.25, 0.75),      # accepted but ignored
    label_keys=("cross", ),
    framewise_bboxes_csv: str | None = None,
    **kwargs,
):
    paths, annotations = annotations_path
    num_clips = sum([len(a) for a in annotations.values()])

    framewise_bboxes = None
    if framewise_bboxes_csv is not None and len(framewise_bboxes_csv) > 0:
        logging.info(f"Loading framewise bboxes from: {framewise_bboxes_csv}")
        framewise_bboxes = load_framewise_bboxes(framewise_bboxes_csv)

    video_decoder = decode_videos_to_clips(
        annotations=annotations,
        frames_per_clip=frames_per_clip,
        fps=fps,
        transform=transform,
        anticipation_time_sec=anticipation_time_sec,
        anticipation_point=anticipation_point,
        label_keys=label_keys,
        framewise_bboxes=framewise_bboxes,
    )

    dataset, datainfo = get_video_wds_dataset(
        batch_size=batch_size,
        input_shards=paths,
        epoch=0,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        video_decoder=video_decoder,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        training=training,
        label_keys=label_keys,
    )

    datainfo.dataloader.num_batches = num_clips // max(1, (world_size * batch_size))
    datainfo.dataloader.num_samples = num_clips

    return dataset, datainfo.dataloader, datainfo
