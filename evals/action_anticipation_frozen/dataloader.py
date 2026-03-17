from logging import getLogger
from typing import Iterable, Tuple

import torch
import torchvision.transforms as transforms

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from evals.action_anticipation_frozen.jaad import filter_annotations as jaad_filter_annotations
from evals.action_anticipation_frozen.jaad import make_webvid as jaad_make_webvid
from evals.action_anticipation_frozen.pie import (
    filter_annotations as pie_filter_annotations,
    make_webvid as pie_make_webvid,
)

from src.datasets.utils.video.randerase import RandomErasing
import numpy as np

_GLOBAL_SEED = 0
logger = getLogger()


def _normalize_anticipation_time(anticipation_time_sec) -> Tuple[float, float]:
    """Ensure anticipation time is a 2-tuple (lo, hi). Accepts float or iterable."""
    if anticipation_time_sec is None:
        return (0.0, 0.0)
    if isinstance(anticipation_time_sec, (int, float)):
        return (float(anticipation_time_sec), float(anticipation_time_sec))
    if isinstance(anticipation_time_sec, Iterable):
        vals = list(anticipation_time_sec)
        if len(vals) == 1:
            return (float(vals[0]), float(vals[0]))
        return (float(vals[0]), float(vals[1]))
    return (0.0, 0.0)


def init_data(
    base_path,
    annotations_path,
    batch_size,
    dataset,
    frames_per_clip=16,
    fps=5,
    crop_size=224,
    rank=0,
    world_size=1,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    training=True,
    decode_video=True,
    anticipation_time_sec=0.0,
    decode_one_clip=False,
    random_resize_scale=(0.9, 1.0),
    reprob=0,
    auto_augment=False,
    motion_shift=False,
    anticipation_point=[0.1, 0.1],
    framewise_bboxes_csv: str | None = None,
):
    transform = make_transforms(
        training=training,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3 / 4, 4 / 3),
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    make_webvid = None
    dataset_lower = dataset.lower()

    if "jaad" in dataset_lower:
        make_webvid = jaad_make_webvid
    elif "pie" in dataset_lower:
        make_webvid = pie_make_webvid
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    anticipation_time_range = _normalize_anticipation_time(anticipation_time_sec)

    dataset_obj, data_loader, data_info = make_webvid(
        training=training,
        decode_one_clip=decode_one_clip,
        world_size=world_size,
        rank=rank,
        base_path=base_path,
        annotations_path=annotations_path,
        batch_size=batch_size,
        transform=transform,
        frames_per_clip=frames_per_clip,
        num_workers=num_workers,
        fps=fps,
        decode_video=decode_video,
        anticipation_time_sec=anticipation_time_range,
        persistent_workers=persistent_workers,
        pin_memory=pin_mem,
        anticipation_point=anticipation_point,
        framewise_bboxes_csv=framewise_bboxes_csv,
    )

    if training and rank == 0:
        try:
            sample = next(iter(data_loader))
            videos, bboxes = sample[0], sample[1]
            logger.info(f"[DEBUG] Sample batch video shape: {videos.shape}")
            logger.info(f"[DEBUG] Sample batch bbox shape: {bboxes.shape}")
            logger.info(
                f"[DEBUG] BBox value range: min={bboxes.min().item():.4f}, max={bboxes.max().item():.4f}"
            )
        except Exception as e:
            logger.warning(f"[DEBUG] Failed bbox sanity check: {e}")

    return dataset_obj, data_loader, data_info


def filter_annotations(
    dataset,
    base_path,
    train_annotations_path,
    val_annotations_path,
    **kwargs,
):
    _filter = None
    dataset_lower = dataset.lower()

    if "jaad" in dataset_lower:
        _filter = jaad_filter_annotations
    elif "pie" in dataset_lower:
        _filter = pie_filter_annotations

    return _filter(
        base_path=base_path,
        train_annotations_path=train_annotations_path,
        val_annotations_path=val_annotations_path,
        **kwargs,
    )


def make_transforms(
    training=True,
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(3 / 4, 4 / 3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=224,
    normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
):
    return VideoTransform(
        training=training,
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=crop_size,
        normalize=normalize,
    )


class VideoTransform(object):
    def __init__(
        self,
        training=True,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3 / 4, 4 / 3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=224,
        normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ):
        self.training = training

        short_side_size = int(crop_size * 256 / 224)
        self.eval_transform = video_transforms.Compose(
            [
                video_transforms.Resize(short_side_size, interpolation="bilinear"),
                video_transforms.CenterCrop(size=(crop_size, crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=normalize[0], std=normalize[1]),
            ]
        )

        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.motion_shift = motion_shift
        self.crop_size = crop_size
        self.normalize = normalize

        self.autoaug_transform = video_transforms.create_random_augment(
            input_size=(crop_size, crop_size),
            auto_augment="rand-m7-n4-mstd0.5-inc1",
            interpolation="bicubic",
        )

        
        # bbox-safe (photometric-only) via weighted choice set w1        
        self.autoaug_transform_bbox = video_transforms.create_random_augment(
            input_size=(crop_size, crop_size),
            auto_augment="rand-m6-n4-w1-mstd0.5-inc1",
            interpolation="bicubic",
        )


        self.spatial_transform = (
            video_transforms.random_resized_crop_with_shift
            if motion_shift
            else video_transforms.random_resized_crop
        )

        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob,
            mode="pixel",
            max_count=1,
            num_splits=1,
            device="cpu",
        )


    def __call__(self, buffer, bboxes=None):
        """Apply video augmentations.

        Args:
            buffer: video frames as numpy array/list in (T, H, W, C) uint8.
            bboxes: optional pixel-space boxes in xyxy format. Expected shape:
                - (T, 4) for frame-wise single-object tracks (PIE/JAAD style), or
                - (N, 4) for per-clip boxes.
        Returns:
            If bboxes is None: video tensor (C, T, H, W)
            Else: (video, boxes) where boxes are normalized xyxy in [0,1]
            in the final (crop_size x crop_size) space.
        """
        boxes = bboxes
        # -------------------------
        # EVAL
        # -------------------------
        if not self.training:
            # Keep original eval pipeline when there are no boxes.
            if bboxes is None:
                return self.eval_transform(buffer)

            # BBox-aware eval: short-side resize then center crop (uniform_crop idx=1).
            video = volume_transforms.ClipToTensor()(buffer)  # C T H W
            video = video.permute(1, 0, 2, 3)  # T C H W

            short_side = int(self.crop_size * 256 / 224)
            video, boxes = video_transforms.random_short_side_scale_jitter(
                video, short_side, short_side, boxes=boxes
            )
            video, boxes = video_transforms.uniform_crop(
                video, self.crop_size, spatial_idx=1, boxes=boxes
            )

            # Normalize video (expects C T H W)
            video = video.permute(1, 0, 2, 3)  # C T H W
            video = video_transforms.Normalize(mean=self.normalize[0], std=self.normalize[1])(video)

            #  Keep numpy for clip_boxes_to_image too (it also uses .copy()).
            boxes = video_transforms.clip_boxes_to_image(boxes, self.crop_size, self.crop_size)

            #  Convert boxes to torch ONLY at the very end.
            boxes = torch.from_numpy(np.asarray(boxes, dtype=np.float32))
            boxes[:, [0, 2]] /= float(self.crop_size)
            boxes[:, [1, 3]] /= float(self.crop_size)
            return video, boxes

        # -------------------------
        # TRAIN
        # -------------------------
        if bboxes is not None:
            video = volume_transforms.ClipToTensor()(buffer)  # C T H W
            video = video.permute(1, 0, 2, 3)  # T C H W

            video, boxes = video_transforms.random_resized_crop_with_boxes(
                video,
                target_height=self.crop_size,
                target_width=self.crop_size,
                boxes=boxes,
                scale=self.random_resize_scale,
                ratio=self.random_resize_aspect_ratio,
            )

            if self.random_horizontal_flip:
                video, boxes = video_transforms.horizontal_flip(0.5, video, boxes=boxes)
            
            if self.auto_augment:
                # video is T C H W (float) here
                frames_pil = [transforms.ToPILImage()(video[t]) for t in range(video.shape[0])]
                frames_pil = self.autoaug_transform_bbox(frames_pil)
                video = torch.stack([transforms.ToTensor()(im) for im in frames_pil])  # T C H W

            if self.reprob > 0:
                video = self.erase_transform(video)

            # Normalize video (expects C T H W)
            video = video.permute(1, 0, 2, 3)  # C T H W
            video = video_transforms.Normalize(mean=self.normalize[0], std=self.normalize[1])(video)

            #  Keep numpy for clip_boxes_to_image too (it also uses .copy()).
            boxes = video_transforms.clip_boxes_to_image(boxes, self.crop_size, self.crop_size)

            #  Convert boxes to torch ONLY at the very end.
            boxes = torch.from_numpy(np.asarray(boxes, dtype=np.float32))
            boxes[:, [0, 2]] /= float(self.crop_size)
            boxes[:, [1, 3]] /= float(self.crop_size)
            return video, boxes

        # -------------------------
        # TRAIN (no boxes): original path
        # -------------------------
        buffer_pil = [transforms.ToPILImage()(frame) for frame in buffer]
        if self.auto_augment:
            buffer_pil = self.autoaug_transform(buffer_pil)

        buffer_tensor = [transforms.ToTensor()(img) for img in buffer_pil]
        video = torch.stack(buffer_tensor)  # T C H W

        # Normalize video (expects C T H W)
        video = video.permute(1, 0, 2, 3)  # C T H W
        video = video_transforms.Normalize(mean=self.normalize[0], std=self.normalize[1])(video)
        video = video.permute(1, 0, 2, 3)  # T C H W

        # Spatial aug (no boxes)
        video =  self.spatial_transform(
            images=video,
            target_height=self.crop_size,
            target_width=self.crop_size,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )

        if self.random_horizontal_flip:
            video = video_transforms.horizontal_flip(0.5, video)

        if self.reprob > 0:
            video = self.erase_transform(video)

        video = video.permute(1, 0, 2, 3)  # C T H W
        return video
