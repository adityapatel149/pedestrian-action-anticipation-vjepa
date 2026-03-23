import argparse
import csv
import os
import queue
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from evals.action_anticipation_frozen.models import init_classifier, init_module
from src.utils.checkpoint_loader import robust_checkpoint_loader

torch.backends.cudnn.benchmark = True


@dataclass
class Detection:
    track_id: int
    bbox_xyxy_norm: Tuple[float, float, float, float]
    score: float = 1.0


@dataclass
class OverlayPrediction:
    track_id: int
    cross_prob: float


class FramewiseBBoxCSV:
    def __init__(
        self,
        csv_path: str,
        video_id: str,
        frame_width: int,
        frame_height: int,
        max_boxes: int = 10,
    ):
        self.max_boxes = max_boxes
        self.video_id = str(video_id)
        self.frame_width = float(frame_width)
        self.frame_height = float(frame_height)
        self.by_frame: Dict[int, List[Detection]] = {}

        self.pid_to_track_id: Dict[str, int] = {}
        self.next_track_id = 0

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            required = {"video_id", "frame", "participant_id", "x1", "y1", "x2", "y2"}
            missing = required - fields
            if missing:
                raise ValueError(f"CSV missing columns: {sorted(missing)}")

            for row in reader:
                if str(row["video_id"]).strip() != self.video_id:
                    continue

                frame_idx = int(row["frame"])
                pid = str(row["participant_id"]).strip()

                if pid not in self.pid_to_track_id:
                    self.pid_to_track_id[pid] = self.next_track_id
                    self.next_track_id += 1
                track_id = self.pid_to_track_id[pid]

                x1 = float(row["x1"]) / self.frame_width
                y1 = float(row["y1"]) / self.frame_height
                x2 = float(row["x2"]) / self.frame_width
                y2 = float(row["y2"]) / self.frame_height

                det = Detection(
                    track_id=track_id,
                    bbox_xyxy_norm=(x1, y1, x2, y2),
                    score=1.0,
                )
                self.by_frame.setdefault(frame_idx, []).append(det)

        for k, v in self.by_frame.items():
            self.by_frame[k] = v[: self.max_boxes]

        print(
            f"[FramewiseBBoxCSV] video_id={self.video_id}, "
            f"frames_with_boxes={len(self.by_frame)}, "
            f"total_boxes={sum(len(v) for v in self.by_frame.values())}, "
            f"unique_ids={len(self.pid_to_track_id)}"
        )

    def get(self, frame_idx: int) -> List[Detection]:
        return self.by_frame.get(frame_idx, [])


class YOLOPedTracker:
    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        max_boxes: int = 10,
        device: str = "cuda:0",
        imgsz: int = 416,
        tracker_cfg: str = "bytetrack.yaml",
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_name, task="detect")
        self.conf = conf
        self.iou = iou
        self.max_boxes = max_boxes
        self.device = device
        self.imgsz = imgsz
        self.tracker_cfg = tracker_cfg

        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        try:
            _ = self.model.track(
                source=dummy,
                persist=True,
                tracker=self.tracker_cfg,
                verbose=False,
                conf=self.conf,
                iou=self.iou,
                classes=[0],
                device=self.device,
                max_det=self.max_boxes,
                imgsz=self.imgsz,
                stream=False,
            )
            print(f"[YOLOTracker] warmup complete with {self.tracker_cfg}")
        except Exception as e:
            print(f"[YOLOTracker] warmup skipped: {e}")

    def get(self, frame_bgr: np.ndarray) -> List[Detection]:
        h, w = frame_bgr.shape[:2]

        results = self.model.track(
            source=frame_bgr,
            persist=True,
            tracker=self.tracker_cfg,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            classes=[0],
            device=self.device,
            max_det=self.max_boxes,
            imgsz=self.imgsz,
            stream=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = (
            boxes.conf.detach().cpu().numpy()
            if boxes.conf is not None
            else np.ones((len(xyxy),), dtype=np.float32)
        )

        if boxes.id is not None:
            ids = boxes.id.detach().cpu().numpy().astype(np.int64).tolist()
        else:
            ids = list(range(len(xyxy)))

        for track_id, b, s in zip(ids, xyxy, confs):
            x1, y1, x2, y2 = b.tolist()
            detections.append(
                Detection(
                    track_id=int(track_id),
                    bbox_xyxy_norm=(x1 / w, y1 / h, x2 / w, y2 / h),
                    score=float(s),
                )
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[: self.max_boxes]


class ModelRunner:
    def __init__(
        self,
        config_path: str,
        classifier_ckpt: str,
        sweep_idx: int = 0,
        device: str = "cuda:0",
        use_fp16: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device.type == "cuda"

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        args_pretrain = cfg["model_kwargs"]
        args_model = args_pretrain["pretrain_kwargs"]
        args_wrapper = args_pretrain["wrapper_kwargs"]
        args_exp = cfg["experiment"]
        args_classifier = args_exp["classifier"]
        args_data = args_exp["data"]
        args_opt = args_exp["optimization"]

        self.data_base_path = args_data.get("base_path")
        self.frames_per_clip = int(args_data["frames_per_clip"])
        self.frames_per_second = float(args_data["frames_per_second"])
        self.resolution = int(args_data.get("resolution", 224))
        self.num_probe_blocks = int(args_classifier.get("num_probe_blocks", 1))
        self.num_heads = int(args_classifier["num_heads"])
        self.num_classifiers = len(args_opt["multihead_kwargs"])

        cross_classes = {0: 0, 1: 1}
        action_classes = {0: 0, 1: 1}
        intersection_classes = {i: i for i in range(5)}
        signalized_classes = {i: i for i in range(4)}

        self.encoder = init_module(
            module_name=args_pretrain["module_name"],
            frames_per_clip=self.frames_per_clip,
            frames_per_second=self.frames_per_second,
            resolution=self.resolution,
            checkpoint=args_pretrain["checkpoint"],
            model_kwargs=args_model,
            wrapper_kwargs=args_wrapper,
            device=self.device,
        )
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        classifiers = init_classifier(
            embed_dim=self.encoder.embed_dim,
            num_heads=self.num_heads,
            cross_classes=cross_classes,
            action_classes=action_classes,
            intersection_classes=intersection_classes,
            signalized_classes=signalized_classes,
            num_blocks=self.num_probe_blocks,
            device=self.device,
            num_classifiers=self.num_classifiers,
        )

        checkpoint = robust_checkpoint_loader(classifier_ckpt, map_location=torch.device("cpu"))
        if "classifiers" not in checkpoint:
            raise KeyError("Expected classifier checkpoint with key 'classifiers'.")
        if sweep_idx >= len(checkpoint["classifiers"]):
            raise IndexError(
                f"sweep_idx={sweep_idx} out of range for {len(checkpoint['classifiers'])} sweeps"
            )

        self.classifier = classifiers[sweep_idx]
        state_dict = checkpoint["classifiers"][sweep_idx]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        msg = self.classifier.load_state_dict(state_dict, strict=True)
        print(f"Loaded sweep {sweep_idx}: {msg}")

        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)

        del checkpoint
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        return np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0

    def build_bbox_tensor(self, detections: List[Detection]) -> torch.Tensor:
        if len(detections) == 0:
            return torch.zeros((0, self.frames_per_clip, 4), dtype=torch.float32)

        arr = np.zeros((len(detections), self.frames_per_clip, 4), dtype=np.float32)
        for i, det in enumerate(detections):
            arr[i, :, :] = np.asarray(det.track_sequence, dtype=np.float32)  # type: ignore[attr-defined]
        return torch.from_numpy(arr)

    @torch.inference_mode()
    def infer_preprocessed(
        self,
        clip_cthw: np.ndarray,
        detections: List[Detection],
        anticipation_time_sec: float = 1.0,
    ) -> List[OverlayPrediction]:
        if len(detections) == 0:
            return []

        clip = torch.from_numpy(clip_cthw).unsqueeze(0).to(self.device, non_blocking=True)
        bboxes = self.build_bbox_tensor(detections).to(self.device, non_blocking=True)
        anticipation_times = torch.tensor([anticipation_time_sec], dtype=torch.float32, device=self.device)

        with torch.cuda.amp.autocast(enabled=self.use_fp16, dtype=torch.float16):
            features = self.encoder(clip, anticipation_times)
            features = features.expand(len(detections), -1, -1)
            out = self.classifier(features, bboxes=bboxes)
            cross_probs = F.softmax(out["cross"], dim=-1)[..., 1]

        probs = cross_probs.detach().float().cpu().numpy().tolist()
        return [
            OverlayPrediction(track_id=det.track_id, cross_prob=float(prob))
            for det, prob in zip(detections, probs)
        ]


class AsyncVideoProcessor:
    def __init__(
        self,
        runner: ModelRunner,
        video_path: Optional[str],
        output_path: str,
        bbox_csv: Optional[str] = None,
        detector_name: str = "yolo26n.pt",
        detector_conf: float = 0.25,
        max_boxes: int = 10,
        display: bool = True,
        stride_overlap: float = 0.30,
        render_scale: float = 1.0,
        detector_imgsz: int = 416,
        anticipation_time: float = 1.0,
        tracker_cfg: str = "bytetrack.yaml",
    ):
        self.runner = runner
        self.video_path = video_path
        self.output_path = output_path
        self.display = display
        self.max_boxes = max_boxes
        self.render_scale = float(render_scale)
        self.anticipation_time = anticipation_time
        self.detector = None
        self.bbox_source = None

        if video_path is None or str(video_path).strip() == "":
            raise ValueError("A valid video path is required")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.src_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out_width = max(1, int(round(self.width * self.render_scale)))
        self.out_height = max(1, int(round(self.height * self.render_scale)))

        if bbox_csv is not None:
            if not self.runner.data_base_path:
                raise ValueError("runner.data_base_path is required when using bbox_csv")

            video_id = os.path.splitext(
                os.path.relpath(video_path, self.runner.data_base_path)
            )[0].replace("\\", "/")

            print(f"Using CSV video_id: {video_id}")
            self.bbox_source = FramewiseBBoxCSV(
                csv_path=bbox_csv,
                video_id=video_id,
                frame_width=self.width,
                frame_height=self.height,
                max_boxes=max_boxes,
            )
        else:
            self.detector = YOLOPedTracker(
                model_name=detector_name,
                conf=detector_conf,
                max_boxes=max_boxes,
                device=str(runner.device),
                imgsz=detector_imgsz,
                tracker_cfg=tracker_cfg,
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, self.src_fps, (self.out_width, self.out_height))

        self.window_size = max(1, int(round(0.5 * self.src_fps)))
        self.encoder_frames = runner.frames_per_clip
        self.infer_stride = max(1, int(round(self.window_size * (1.0 - stride_overlap))))
        self.window_sample_idx = np.linspace(0, self.window_size - 1, self.encoder_frames).round().astype(np.int64)

        self.frame_buffer: Deque[Tuple[int, np.ndarray]] = deque(maxlen=self.window_size)
        self.preproc_buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)
        self.detection_history: Deque[Tuple[int, List[Detection]]] = deque(maxlen=self.window_size)

        self.current_predictions: Dict[int, OverlayPrediction] = {}

        self.read_queue: "queue.Queue[Optional[Tuple[int, np.ndarray]]]" = queue.Queue(maxsize=128)
        self.infer_queue: "queue.Queue[Optional[Tuple[int, np.ndarray, List[Detection]]]]" = queue.Queue(maxsize=16)
        self.pred_queue: "queue.Queue[Optional[List[OverlayPrediction]]]" = queue.Queue(maxsize=32)

        self.stop_event = threading.Event()

    def read_loop(self):
        frame_idx = 0
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break
            self.read_queue.put((frame_idx, frame))
            frame_idx += 1
        self.read_queue.put(None)

    def inference_loop(self):
        while not self.stop_event.is_set():
            item = self.infer_queue.get()
            if item is None:
                break

            _, clip_cthw, detections = item
            preds = self.runner.infer_preprocessed(
                clip_cthw,
                detections,
                anticipation_time_sec=self.anticipation_time,
            )
            self.pred_queue.put(preds)

        self.pred_queue.put(None)

    def get_detections(self, frame_idx: int, frame: np.ndarray) -> List[Detection]:
        if self.bbox_source is not None:
            return self.bbox_source.get(frame_idx)[: self.max_boxes]
        return self.detector.get(frame)

    def build_tracked_detections_for_window(self) -> List[Detection]:
        if len(self.detection_history) < self.window_size:
            return []

        window = list(self.detection_history)
        _, current_dets = window[-1]
        if len(current_dets) == 0:
            return []

        current_track_ids = [det.track_id for det in current_dets]
        track_to_boxes: Dict[int, List[Optional[Tuple[float, float, float, float]]]] = {
            track_id: [None] * self.window_size for track_id in current_track_ids
        }

        for t, (_, dets) in enumerate(window):
            for det in dets:
                if det.track_id in track_to_boxes:
                    track_to_boxes[det.track_id][t] = det.bbox_xyxy_norm

        valid_tracks: List[Detection] = []
        for det in current_dets:
            seq = track_to_boxes[det.track_id]
            if all(b is None for b in seq):
                continue

            first = next(i for i, b in enumerate(seq) if b is not None)
            last = max(i for i, b in enumerate(seq) if b is not None)

            for i in range(first - 1, -1, -1):
                seq[i] = seq[i + 1]
            for i in range(last + 1, self.window_size):
                seq[i] = seq[i - 1]
            for i in range(first + 1, last):
                if seq[i] is None:
                    seq[i] = seq[i - 1]

            sampled_seq = np.asarray(seq, dtype=np.float32)[self.window_sample_idx]

            out_det = Detection(
                track_id=det.track_id,
                bbox_xyxy_norm=det.bbox_xyxy_norm,
                score=det.score,
            )
            out_det.track_sequence = sampled_seq  # type: ignore[attr-defined]
            valid_tracks.append(out_det)

        valid_tracks.sort(key=lambda d: d.score, reverse=True)
        return valid_tracks[: self.max_boxes]

    def build_clip_from_preprocessed(self) -> np.ndarray:
        preproc_list = list(self.preproc_buffer)
        sampled = [preproc_list[i] for i in self.window_sample_idx]
        return np.stack(sampled, axis=1)

    @staticmethod
    def prob_to_color(prob: float) -> Tuple[int, int, int]:
        prob = float(np.clip(prob, 0.0, 1.0))
        if prob < 0.5:
            alpha = prob / 0.5
            r, g = 255, int(255 * alpha)
        else:
            alpha = (prob - 0.5) / 0.5
            r, g = int(255 * (1.0 - alpha)), 255
        return (0, g, r)

    def draw_predictions(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        canvas = frame
        h, w = canvas.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy_norm
            px1 = int(np.clip(x1, 0, 1) * w)
            py1 = int(np.clip(y1, 0, 1) * h)
            px2 = int(np.clip(x2, 0, 1) * w)
            py2 = int(np.clip(y2, 0, 1) * h)

            pred = self.current_predictions.get(det.track_id)

            if pred is None:
                label = "no-pred"
                color = (255, 255, 255)
            else:
                prob = pred.cross_prob
                label = f"cross={prob:.3f}" if prob >= 0.5 else f"not-cross={1.0 - prob:.3f}"
                color = self.prob_to_color(prob)

            cv2.rectangle(canvas, (px1, py1), (px2, py2), color, 2)
            cv2.rectangle(canvas, (px1, max(0, py1 - 24)), (min(w - 1, px1 + 180), py1), color, -1)
            cv2.putText(
                canvas,
                label,
                (px1 + 4, max(14, py1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def run(self):
        t_reader = threading.Thread(target=self.read_loop, daemon=True)
        t_reader.start()

        t_infer = threading.Thread(target=self.inference_loop, daemon=True)
        t_infer.start()

        next_infer_frame_idx = self.window_size - 1
        pred_done = False
        frame_done = False

        while True:
            while not pred_done:
                try:
                    pred_item = self.pred_queue.get_nowait()
                except queue.Empty:
                    break

                if pred_item is None:
                    pred_done = True
                    break

                self.current_predictions = {p.track_id: p for p in pred_item}

            if frame_done:
                break

            item = self.read_queue.get()
            if item is None:
                frame_done = True
                self.infer_queue.put(None)
                continue

            frame_idx, frame = item

            self.frame_buffer.append((frame_idx, frame))
            self.preproc_buffer.append(self.runner.preprocess_frame(frame))

            detections = self.get_detections(frame_idx, frame)
            self.detection_history.append((frame_idx, detections))

            if len(self.frame_buffer) == self.window_size and frame_idx >= next_infer_frame_idx:
                tracked_detections = self.build_tracked_detections_for_window()

                if len(tracked_detections) > 0:
                    clip_cthw = self.build_clip_from_preprocessed()
                    try:
                        self.infer_queue.put_nowait((frame_idx, clip_cthw, tracked_detections))
                    except queue.Full:
                        pass
                else:
                    self.current_predictions = {}

                next_infer_frame_idx = frame_idx + self.infer_stride

            rendered = self.draw_predictions(frame.copy(), detections)
            if (self.out_width, self.out_height) != (self.width, self.height):
                rendered = cv2.resize(
                    rendered,
                    (self.out_width, self.out_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            self.writer.write(rendered)

            if self.display:
                cv2.imshow("crossing-demo", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    self.stop_event.set()
                    break

        t_reader.join(timeout=1.0)
        t_infer.join(timeout=1.0)
        self.cap.release()
        self.writer.release()
        if self.display:
            cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Async raw bbox track + raw inference overlay")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--classifier-ckpt", type=str, required=True)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--bbox-csv", type=str, default=None, help="optional framewise bbox CSV; if omitted, use YOLO tracking")
    p.add_argument("--detector", type=str, default="yolo26n.pt")
    p.add_argument("--tracker", type=str, default="bytetrack.yaml")
    p.add_argument("--detector-conf", type=float, default=0.25)
    p.add_argument("--max-boxes", type=int, default=10)
    p.add_argument("--sweep-idx", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--display", action="store_true")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--stride-overlap", type=float, default=0.30)
    p.add_argument("--render-scale", type=float, default=1.0)
    p.add_argument("--detector-imgsz", type=int, default=640)
    p.add_argument("--anticipation-time", type=float, default=1.0)
    return p.parse_args()


def resolve_video_path_from_csv(csv_path: str, config_path: str) -> str:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_path = cfg["experiment"]["data"].get("base_path")
    if not base_path:
        raise ValueError("Config experiment.data.base_path is required to resolve --video from CSV")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        first = next(reader, None)
        if first is None:
            raise ValueError("CSV is empty")
        if "video_id" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain a video_id column")

        video_id = str(first["video_id"]).strip()
        if not video_id:
            raise ValueError("CSV video_id is empty")

    return os.path.join(base_path, f"{video_id}.mp4")


def main():
    args = parse_args()

    if (args.video is None or str(args.video).strip() == "") and args.bbox_csv is not None:
        args.video = resolve_video_path_from_csv(args.bbox_csv, args.config)
        print(f"Resolved video path from CSV/config: {args.video}")

    if args.video is None or str(args.video).strip() == "":
        raise ValueError("A valid --video path is required")

    runner = ModelRunner(
        config_path=args.config,
        classifier_ckpt=args.classifier_ckpt,
        sweep_idx=args.sweep_idx,
        device=args.device,
        use_fp16=not args.no_fp16,
    )

    processor = AsyncVideoProcessor(
        runner=runner,
        video_path=args.video,
        output_path=args.output,
        bbox_csv=args.bbox_csv,
        detector_name=args.detector,
        detector_conf=args.detector_conf,
        max_boxes=args.max_boxes,
        display=args.display,
        stride_overlap=args.stride_overlap,
        render_scale=args.render_scale,
        detector_imgsz=args.detector_imgsz,
        anticipation_time=args.anticipation_time,
        tracker_cfg=args.tracker,
    )
    processor.run()


if __name__ == "__main__":
    main()
