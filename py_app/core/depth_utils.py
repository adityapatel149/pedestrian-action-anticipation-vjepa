import numpy as np
import cv2
from typing import Optional, Tuple

def sample_depth_at_pixel(depth: np.ndarray, x: int, y: int) -> float:
    # Ensure coordinates are within bounds
    if x < 0 or x >= depth.shape[1] or y < 0 or y >= depth.shape[0]:
        return np.nan  # Return NaN if out of bounds
    return depth[y, x]

def sample_depth_median(depth: np.ndarray, x: int, y: int, k: int = 5) -> float:
    h, w = depth.shape
    x1 = max(0, x - k)
    y1 = max(0, y - k)
    x2 = min(w, x + k + 1)
    y2 = min(h, y + k + 1)
    patch = depth[y1:y2, x1:x2]
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        return float("nan")
    return float(np.median(patch))

def compute_bev_points_from_depth(
    depth: np.ndarray,
    K: np.ndarray,
    half_width_m: float,
    max_range_m: float,
    scale_m_per_unit: Optional[float],
    sample_step: int = 12,
    max_points: int = 12000,
    min_row_frac: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray]:

    h, w = depth.shape

    # ---- sampling grid (no float conversion yet) ----
    ys = np.arange(0, h, sample_step, dtype=np.int32)
    xs = np.arange(0, w, sample_step, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    u = grid_x.ravel()
    v = grid_y.ravel()

    # ---- direct indexing (no repeated astype) ----
    d = depth[v, u]

    # ---- validity mask ----
    min_row = int(min_row_frac * h)
    valid = np.isfinite(d)
    valid &= d > 0.0
    valid &= v > min_row

    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    u = u[valid].astype(np.float32, copy=False)
    d = d[valid].astype(np.float32, copy=False)

    # ---- depth scaling ----
    if scale_m_per_unit is not None:
        z_m = d * scale_m_per_unit
    else:
        #  percentile is expensive → reduce cost
        if d.size > 2000:
            sample = d[:: max(1, d.size // 2000)]
        else:
            sample = d

        lo = np.percentile(sample, 5.0)
        hi = np.percentile(sample, 95.0)
        if hi <= lo:
            hi = lo + 1e-6

        inv_range = 1.0 / (hi - lo)
        d_norm = (d - lo) * inv_range

        # faster clip
        d_norm = np.minimum(np.maximum(d_norm, 0.0), 1.0)

        z_m = (1.0 - d_norm) * max_range_m

    # ---- clamp z ----
    z_m = np.minimum(np.maximum(z_m, 0.5), max_range_m)

    # ---- projection ----
    fx = K[0, 0]
    cx = K[0, 2]

    x_m = ((u - cx) / fx) * z_m

    # ---- spatial filtering ----
    keep = (np.abs(x_m) <= half_width_m) & (z_m <= max_range_m)

    if not np.any(keep):
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    x_m = x_m[keep]
    z_m = z_m[keep]

    # ---- subsample ----
    if x_m.size > max_points:
        step = x_m.size // max_points
        x_m = x_m[::step]
        z_m = z_m[::step]

    # ---- values ----
    vals = 1.0 - (z_m / max_range_m)
    vals = np.minimum(np.maximum(vals, 0.0), 1.0)

    pts_world = np.empty((x_m.size, 2), dtype=np.float32)
    pts_world[:, 0] = x_m
    pts_world[:, 1] = z_m

    return pts_world, vals.astype(np.float32, copy=False)


def undistort_depth(
    depth: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
) -> np.ndarray:
    return cv2.remap(
        depth,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )