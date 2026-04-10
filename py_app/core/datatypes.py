from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

@dataclass
class Detection:
    track_id: int
    bbox_xyxy_norm: Tuple[float, float, float, float]
    score: float = 1.0
    track_sequence: Optional[np.ndarray] = None


@dataclass
class OverlayPrediction:
    track_id: int
    cross_prob: float
    risk_score: float = 0.0


@dataclass(frozen=True)
class CameraConfig:
    K: np.ndarray
    D: np.ndarray
    cam_height_mm: float
    cam_pitch_deg: float


@dataclass(frozen=True)
class BevConfig:
    bev_size: int
    max_range_m: float
    bev_half_width_m: float
    depth_sample_step: int
    depth_max_points: int