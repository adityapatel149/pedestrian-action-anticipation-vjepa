from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

from py_app.runners.depth_runner import DepthAnythingRunner
from py_app.core.datatypes import Detection, Prediction, CameraConfig, BevConfig
from py_app.core.depth_utils import undistort_depth, sample_depth_median
from py_app.visualization.overlay import draw_predictions
from py_app.visualization.bev import compute_ground_distance, render_bev


class VisualizationNode:
    def __init__(
        self,
        *,
        frame_width: int,
        frame_height: int,
        src_fps: float,
        use_depth: bool = False,
        depth_model: Optional[str] = None,
        depth_every_n: int = 3,
        depth_calib_interval_sec: float = 3.0,
        depth_scale_alpha: float = 0.15,
        depth_min_calib_points: int = 2,
        depth_sample_step: int = 12,
        depth_max_points: int = 12000,
        depth_smooth_alpha: float = 0.2,
        bev_size=None,
        enable_overlay_render: bool = True,
        enable_bev_render: bool = True,
        camera_config: Optional[CameraConfig] = None,
        bev_config: Optional[BevConfig] = None,
        device: str = "cuda:0",
    ):
        self.width = int(frame_width)
        self.height = int(frame_height)
        self.src_fps = float(src_fps or 30.0)
        self.enable_overlay_render = bool(enable_overlay_render)
        self.enable_bev_render = bool(enable_bev_render)

        self.use_depth = bool(use_depth)
        self.depth_model = depth_model
        self.depth_every_n = max(1, int(depth_every_n))
        self.depth_runner: Optional[DepthAnythingRunner] = None
        self.last_depth: Optional[np.ndarray] = None
        self.last_bev: Optional[np.ndarray] = None
        self.last_overlay: Optional[np.ndarray] = None

        self.depth_scale_m_per_unit: Optional[float] = None
        self.depth_smooth_alpha = float(depth_smooth_alpha)
        self.depth_scale_alpha = float(depth_scale_alpha)
        self.depth_calib_interval_sec = float(depth_calib_interval_sec)
        self.depth_min_calib_points = int(depth_min_calib_points)
        self.last_depth_calib_frame = -10**9
        self.depth_sample_step = int(depth_sample_step)
        self.depth_max_points = int(depth_max_points)
        self.depth_calib_interval_frames = max(
            1, int(round(self.depth_calib_interval_sec * self.src_fps))
        )

        if camera_config is None:
            camera_config = CameraConfig(
                K=np.array(
                    [
                        [1004.8374471951423, 0.0, 960.1025514993675],
                        [0.0, 1004.3912782107128, 573.5538287373604],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                D=np.array(
                    [
                        [-0.02748054291929438],
                        [-0.007055051080370751],
                        [-0.039625194298025156],
                        [0.019310795479533783],
                    ],
                    dtype=np.float32,
                ),
                cam_height_mm=1270.0,
                cam_pitch_deg=-10.0,
            )
        self.camera_config = camera_config

        if bev_config is None:
            bev_config = BevConfig(
                bev_size=700 if bev_size is None else bev_size,
                max_range_m=30.0,
                bev_half_width_m=12.0,
                depth_sample_step=self.depth_sample_step,
                depth_max_points=self.depth_max_points,
            )
        self.bev_config = bev_config

        self.depth_undistort_map1, self.depth_undistort_map2 = cv2.initUndistortRectifyMap(
            self.camera_config.K,
            self.camera_config.D,
            None,
            self.camera_config.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        if self.use_depth:
            if not self.depth_model:
                raise ValueError("--depth-model is required when --use-depth is enabled")
            self.depth_runner = DepthAnythingRunner(
                model_path=self.depth_model,
                device=device,
            )

    def update_depth_and_bev(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        detections: List[Detection],
        predictions: Dict[int, Prediction],
    ):
        if not self.enable_bev_render:
            return

        depth = self.last_depth
        need_new_depth = (
            self.depth_runner is not None
            and ((frame_idx % self.depth_every_n) == 0 or self.last_depth is None)
        )

        if need_new_depth:
            undistorted_frame = cv2.remap(
                frame_bgr,
                self.depth_undistort_map1,
                self.depth_undistort_map2,
                cv2.INTER_LINEAR,
            )

            try:
                new_depth = self.depth_runner.predict_from_undistorted_bgr(
                    undistorted_frame
                ).astype(np.float32)
                new_depth = cv2.GaussianBlur(new_depth, (5, 5), 0)

                if self.last_depth is None:
                    depth = new_depth
                else:
                    a = self.depth_smooth_alpha
                    depth = (1.0 - a) * self.last_depth + a * new_depth

                self.update_depth_scale(
                    frame_idx=frame_idx,
                    frame=frame_bgr,
                    depth=depth,
                    detections=detections,
                )
                self.last_depth = depth
            except Exception as e:
                print(f"[depth] ERROR: {e}")
                depth = self.last_depth

        self.last_bev = render_bev(
            frame_bgr=frame_bgr,
            detections=detections,
            predictions=predictions,
            camera_config=self.camera_config,
            bev_config=self.bev_config,
            depth=depth,
            depth_scale_m_per_unit=self.depth_scale_m_per_unit,
        )

    def undistort_depth(self, depth: np.ndarray) -> np.ndarray:
        return undistort_depth(depth, self.depth_undistort_map1, self.depth_undistort_map2)

    def update_depth_scale(
        self,
        frame_idx: int,
        frame: np.ndarray,
        depth: np.ndarray,
        detections: List[Detection],
    ):
        if (frame_idx - self.last_depth_calib_frame) < self.depth_calib_interval_frames:
            return

        candidates: List[Tuple[float, float]] = []
        for det in detections:
            info = compute_ground_distance(frame, det, self.camera_config, self.bev_config)
            if info is None:
                continue

            foot_x, foot_y, _x_m, z_m, dist_m = info
            rel_d = sample_depth_median(depth, int(foot_x), int(foot_y), k=4)
            if not np.isfinite(rel_d):
                continue
            if rel_d <= 1e-6:
                continue
            if dist_m < 1.5 or dist_m > self.bev_config.max_range_m:
                continue

            candidates.append((dist_m, z_m / float(rel_d)))

        if len(candidates) < self.depth_min_calib_points:
            return

        candidates.sort(key=lambda x: x[0])
        selected = candidates[:3] if len(candidates) >= 3 else candidates
        ratios = [ratio for _, ratio in selected]

        new_scale = float(np.median(np.asarray(ratios, dtype=np.float32)))
        if not np.isfinite(new_scale) or new_scale <= 0.0:
            return

        if self.depth_scale_m_per_unit is None:
            self.depth_scale_m_per_unit = new_scale
        else:
            a = self.depth_scale_alpha
            self.depth_scale_m_per_unit = (1.0 - a) * self.depth_scale_m_per_unit + a * new_scale

        self.last_depth_calib_frame = frame_idx

    def render_overlay(
        self,
        frame_bgr: np.ndarray,
        detections: List[Detection],
        predictions: Dict[int, Prediction],
    ) -> Optional[np.ndarray]:
        if not self.enable_overlay_render:
            return None

        self.last_overlay = draw_predictions(
            frame=frame_bgr.copy(),
            detections=detections,
            predictions=predictions,
            camera_config=self.camera_config,
            bev_config=self.bev_config,
        )
        return self.last_overlay