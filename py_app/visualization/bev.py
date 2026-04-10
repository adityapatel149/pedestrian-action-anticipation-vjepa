import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

from py_app.core.datatypes import Detection, Prediction, CameraConfig, BevConfig
from py_app.core.geometry import project_pixel_to_ground_undistorted, project_world_to_bev
from py_app.core.depth_utils import compute_bev_points_from_depth
from py_app.core.risk import risk_to_color


def build_bev_background(bev_config: BevConfig) -> np.ndarray:
    bev = np.zeros((bev_config.bev_size, bev_config.bev_size, 3), dtype=np.uint8)
    bev[:] = (22, 22, 22)

    cx = bev_config.bev_size // 2
    y_base = bev_config.bev_size - 20
    bonnet_w = int(0.34 * bev_config.bev_size)
    bonnet_h = int(0.08 * bev_config.bev_size)

    overlay = bev.copy()
    cv2.ellipse(overlay, (cx, y_base), (bonnet_w // 2, bonnet_h), 0, 180, 360, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.ellipse(overlay, (cx, y_base), (bonnet_w // 2, bonnet_h), 0, 180, 360, (55, 55, 55), 2, cv2.LINE_AA)

    inner_w = int(bonnet_w * 0.72)
    inner_h = int(bonnet_h * 0.58)
    cv2.ellipse(overlay, (cx, y_base + 1), (inner_w // 2, inner_h), 0, 180, 360, (115, 115, 115), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.92, bev, 0.08, 0, bev)
    return bev


def compute_ground_distance(
    frame_bgr: np.ndarray,
    det: Detection,
    camera_config: CameraConfig,
    bev_config: BevConfig,
) -> Optional[Tuple[float, float, float, float, float]]:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = det.bbox_xyxy_norm

    foot_x = float(np.clip((x1 + x2) * 0.5, 0.0, 1.0) * (w - 1))
    box_h = max(0.0, y2 - y1)
    foot_y_norm = min(1.0, y2 + 0.08 * box_h)
    foot_y = float(foot_y_norm * (h - 1))

    ground_pt = project_pixel_to_ground_undistorted(
        u=foot_x,
        v=foot_y,
        K=camera_config.K,
        D=camera_config.D,
        cam_height_m=camera_config.cam_height_mm / 1000.0,
        cam_pitch_deg=camera_config.cam_pitch_deg,
    )
    if ground_pt is None:
        return None

    x_m, z_m = ground_pt

    if abs(x_m) > bev_config.bev_half_width_m or z_m < 0.0 or z_m > bev_config.max_range_m:
        return None

    z_m -= 3
    dist_m = float(np.sqrt(x_m * x_m + z_m * z_m))
    return foot_x, foot_y, x_m, z_m, dist_m


def draw_depth_on_bev(
    bev: np.ndarray,
    depth: np.ndarray,
    camera_config,
    bev_config,
    depth_scale_m_per_unit: Optional[float],
) -> np.ndarray:

    pts_world, vals = compute_bev_points_from_depth(
        depth=depth,
        K=camera_config.K,
        half_width_m=bev_config.bev_half_width_m,
        max_range_m=bev_config.max_range_m,
        scale_m_per_unit=depth_scale_m_per_unit,
        sample_step=bev_config.depth_sample_step,
        max_points=bev_config.depth_max_points,
        min_row_frac=0.45,
    )

    if pts_world.shape[0] == 0:
        return bev

    x_m = pts_world[:, 0]
    z_m = pts_world[:, 1]
    z_m = np.clip(z_m, 0.0, bev_config.max_range_m)

    x_norm = (x_m + bev_config.bev_half_width_m) / (2.0 * bev_config.bev_half_width_m)
    y_norm = 1.0 - (z_m / bev_config.max_range_m)

    px = np.clip((x_norm * (bev_config.bev_size - 1)).astype(np.int32), 12, bev_config.bev_size - 12)
    py = np.clip((y_norm * (bev_config.bev_size - 1)).astype(np.int32), 0, bev_config.bev_size - 1)

    occupancy = np.zeros((bev_config.bev_size, bev_config.bev_size), dtype=np.float32)
    np.maximum.at(occupancy, (py, px), vals)

    mask = occupancy > 0
    intensity = (45 + 110 * occupancy[mask]).astype(np.uint8)

    bev[mask] = np.stack([intensity] * 3, axis=1)

    return bev


def render_bev(
    frame_bgr: np.ndarray,
    detections: List[Detection],
    predictions: Dict[int, Prediction],
    camera_config: CameraConfig,
    bev_config: BevConfig,
    depth: Optional[np.ndarray] = None,
    depth_scale_m_per_unit: Optional[float] = None,
    alpha: float = 0.1,
) -> np.ndarray:
    bev = build_bev_background(bev_config)

    if not hasattr(render_bev, "previous_positions"):
        render_bev.previous_positions = {}

    if depth is not None:
        bev = draw_depth_on_bev(
            bev=bev,
            depth=depth,
            camera_config=camera_config,
            bev_config=bev_config,
            depth_scale_m_per_unit=depth_scale_m_per_unit,
        )

        if depth_scale_m_per_unit is not None:
            cv2.putText(
                bev,
                f"depth scale={depth_scale_m_per_unit:.3f} m/unit",
                (14, bev_config.bev_size - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

    for det in detections:
        info = compute_ground_distance(frame_bgr, det, camera_config, bev_config)
        if info is None:
            continue

        _foot_x, _foot_y, x_m, z_m, dist_m = info
        bev_x, bev_y = project_world_to_bev(
            x_m,
            z_m,
            bev_config.bev_size,
            bev_config.bev_half_width_m,
            bev_config.max_range_m,
        )

        if det.track_id in render_bev.previous_positions:
            prev_bev_x, prev_bev_y = render_bev.previous_positions[det.track_id]
            bev_x = alpha * bev_x + (1 - alpha) * prev_bev_x
            bev_y = alpha * bev_y + (1 - alpha) * prev_bev_y

        render_bev.previous_positions[det.track_id] = (bev_x, bev_y)

        bev_y -= int((3.0 / bev_config.max_range_m) * (bev_config.bev_size - 1))

        pred = predictions.get(det.track_id)

        if pred is None:
            color = (255, 255, 255)
            label = f"{dist_m:.1f}m"
        else:
            if pred.distance_m is None:
                pred.distance_m = dist_m
            color = risk_to_color(pred.risk_score)
            label = f"{pred.distance_m:.1f}m r={pred.risk_score:.2f}"

        cv2.circle(bev, (int(bev_x), int(bev_y)), 4, color, 2, cv2.LINE_AA)
        tx = min(bev_config.bev_size - 150, int(bev_x) + 12)
        ty = max(18, int(bev_y) - 8)
        cv2.putText(bev, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return bev