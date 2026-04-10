import cv2
import numpy as np
from typing import Optional, Tuple

def project_pixel_to_ground_undistorted(
    u: float,
    v: float,
    K: np.ndarray,
    D: np.ndarray,
    cam_height_m: float,
    cam_pitch_deg: float,
) -> Optional[Tuple[float, float]]:

    pt = np.array([[[u, v]]], dtype=np.float32)

    undist = cv2.undistortPoints(pt, K, D)
    x_n = float(undist[0, 0, 0])
    y_n = float(undist[0, 0, 1])

    pitch = np.deg2rad(float(cam_pitch_deg))
    beta = np.arctan(y_n)

    denom = np.tan(pitch + beta)
    if abs(denom) < 1e-6:
        return None

    z_m = cam_height_m / denom
    if z_m <= 0:
        return None

    x_m = x_n * z_m
    return float(x_m), float(z_m)


def project_world_to_bev(
    x_m: float,
    z_m: float,
    bev_size: int,
    half_width_m: float,
    max_range_m: float,
) -> Tuple[int, int]:

    x_norm = (x_m + half_width_m) / (2.0 * half_width_m)
    y_norm = 1.0 - (z_m / max_range_m)

    px = int(np.clip(x_norm * (bev_size - 1), 12, bev_size - 12))
    py = int(np.clip(y_norm * (bev_size - 1), 0, bev_size - 1))
    return px, py

