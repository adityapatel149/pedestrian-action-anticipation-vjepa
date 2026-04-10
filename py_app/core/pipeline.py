from typing import List, Tuple, Dict, Optional, Deque
from collections import deque
import numpy as np
import cv2
import os
import queue
import threading

from py_app.runners.depth_runner import DepthAnythingRunner
from py_app.tracking.bbox_csv import FramewiseBBoxCSV
from py_app.tracking.yolo_tracker import YOLOPedTracker
from py_app.core.datatypes import Detection, OverlayPrediction, CameraConfig, BevConfig
from py_app.core.depth_utils import undistort_depth, sample_depth_at_pixel, sample_depth_median
from py_app.visualization.overlay import draw_predictions
from py_app.visualization.bev import compute_ground_distance, render_bev



class AsyncVideoProcessor:
    def __init__(
        self,
        runner,
        video_path: Optional[str],
        output_path: Optional[str] = None,
        save_bev_video: bool = False,
        bev_size=None,
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
        use_depth: bool = False,
        depth_model: Optional[str] = None,
        depth_every_n: int = 3,
        depth_calib_interval_sec: float = 3.0,
        depth_scale_alpha: float = 0.15,
        depth_min_calib_points: int = 2,
        depth_sample_step: int = 12,
        depth_max_points: int = 12000,
        depth_smooth_alpha: float = 0.2,
    ):
        self.runner = runner
        self.video_path = video_path
        self.display = display
        self.max_boxes = max_boxes
        self.render_scale = float(render_scale)
        self.anticipation_time = anticipation_time
        self.detector = None
        self.bbox_source = None

        self.use_depth = bool(use_depth)
        self.depth_model = depth_model
        self.depth_every_n = max(1, int(depth_every_n))
        self.depth_runner: Optional[DepthAnythingRunner] = None
        self.last_depth: Optional[np.ndarray] = None
        self.last_bev: Optional[np.ndarray] = None

        # Fixed depth scale. Do not smooth this.
        self.depth_scale_m_per_unit: Optional[float] = None

        # Smooth the depth map itself.
        self.depth_smooth_alpha = float(depth_smooth_alpha)

        self.depth_scale_alpha = float(depth_scale_alpha)
        self.depth_calib_interval_sec = float(depth_calib_interval_sec)
        self.depth_min_calib_points = int(depth_min_calib_points)
        self.last_depth_calib_frame = -10**9
        self.depth_sample_step = int(depth_sample_step)
        self.depth_max_points = int(depth_max_points)

        self.cam_height_mm = 1270.0
        self.cam_pitch_deg = -10.0

        self.K = np.array([
            [1004.8374471951423, 0.0, 960.1025514993675],
            [0.0, 1004.3912782107128, 573.5538287373604],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        self.D = np.array([
            [-0.02748054291929438],
            [-0.007055051080370751],
            [-0.039625194298025156],
            [0.019310795479533783],
        ], dtype=np.float32)

        self.max_range_m = 30.0
        self.bev_half_width_m = 12.0
        self.camera_config = CameraConfig(
            K=self.K,
            D=self.D,
            cam_height_mm=self.cam_height_mm,
            cam_pitch_deg=self.cam_pitch_deg,
        )

        self.bev_config = BevConfig(
            bev_size=700 if bev_size is None else bev_size,
            max_range_m=self.max_range_m,
            bev_half_width_m=self.bev_half_width_m,
            depth_sample_step=self.depth_sample_step,
            depth_max_points=self.depth_max_points,
        )

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
        self.output_path = output_path
        self.save_bev_video = bool(save_bev_video)

        if self.save_bev_video and self.output_path is None:
            raise ValueError("output_path is required when save_bev_video=True")

        self.enable_overlay_render = self.display or (self.output_path is not None)
        self.enable_bev_render = self.display or self.save_bev_video

        self.overlay_writer = None
        self.bev_writer = None
        self.bev_output_path: Optional[str] = None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if self.output_path is not None:
            out_dir = os.path.dirname(self.output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            self.overlay_writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.src_fps,
                (self.out_width, self.out_height),
            )
            if not self.overlay_writer.isOpened():
                raise RuntimeError(f"Failed to open overlay writer: {self.output_path}")

        if self.save_bev_video:
            base, ext = os.path.splitext(self.output_path)
            if ext == "":
                ext = ".mp4"
            self.bev_output_path = f"{base}_bev{ext}"

            bev_dir = os.path.dirname(self.bev_output_path)
            if bev_dir:
                os.makedirs(bev_dir, exist_ok=True)

            bev_size_px = self.bev_config.bev_size
            self.bev_writer = cv2.VideoWriter(
                self.bev_output_path,
                fourcc,
                self.src_fps,
                (bev_size_px, bev_size_px),
            )
            if not self.bev_writer.isOpened():
                raise RuntimeError(f"Failed to open BEV writer: {self.bev_output_path}")


        self.depth_calib_interval_frames = max(1, int(round(self.depth_calib_interval_sec * self.src_fps)))

        self.depth_undistort_map1, self.depth_undistort_map2 = cv2.initUndistortRectifyMap(
            self.K,
            self.D,
            None,
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

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

        if self.use_depth:
            if not self.depth_model:
                raise ValueError("--depth-model is required when --use-depth is enabled")
            self.depth_runner = DepthAnythingRunner(
                model_path=self.depth_model,
                device=str(runner.device),
            )

        self.window_size = max(1, int(round(0.5 * self.src_fps)))
        self.encoder_frames = runner.frames_per_clip
        self.infer_stride = max(1, int(round(self.window_size * (1.0 - stride_overlap))))
        self.window_sample_idx = np.linspace(
            0, self.window_size - 1, self.encoder_frames
        ).round().astype(np.int64)

        self.frame_buffer: Deque[Tuple[int, np.ndarray]] = deque(maxlen=self.window_size)
        self.preproc_buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)
        self.detection_history: Deque[Tuple[int, List[Detection]]] = deque(maxlen=self.window_size)

        self.current_predictions: Dict[int, OverlayPrediction] = {}

        self.read_queue: "queue.Queue[Optional[Tuple[int, np.ndarray]]]" = queue.Queue(maxsize=128)
        self.infer_queue: "queue.Queue[Optional[Tuple[int, np.ndarray, List[Detection]]]]" = queue.Queue(maxsize=16)
        self.pred_queue: "queue.Queue[Optional[List[OverlayPrediction]]]" = queue.Queue(maxsize=32)

        self.stop_event = threading.Event()


    # ------------------------ loops ------------------------
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
            try:
                preds = self.runner.predict(
                    clip_cthw,
                    detections,
                    anticipation_time_sec=self.anticipation_time,
                )
                self.pred_queue.put(preds)
            except Exception as e:
                print(f"[inference_loop] ERROR: {e}")
                self.pred_queue.put([])
                break

        self.pred_queue.put(None)

    # ------------------------ detections ------------------------
    def get_detections(self, frame_idx: int, frame: np.ndarray) -> List[Detection]:
        if self.bbox_source is not None:
            return self.bbox_source.get(frame_idx)[: self.max_boxes]
        if self.detector is None:
            return []
        return self.detector.get(frame)

    def build_track_sequences(self) -> List[Detection]:
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
            valid_tracks.append(
                Detection(
                    track_id=det.track_id,
                    bbox_xyxy_norm=det.bbox_xyxy_norm,
                    score=det.score,
                    track_sequence=sampled_seq,
                )
            )

        valid_tracks.sort(key=lambda d: d.score, reverse=True)
        return valid_tracks[: self.max_boxes]

    def build_clip(self) -> np.ndarray:
        if not hasattr(self, "_clip_buf"):
            self._clip_buf = np.empty(
                (3, self.encoder_frames, self.runner.resolution, self.runner.resolution),
                dtype=np.float32,
            )

        preproc_list = list(self.preproc_buffer)
        for j, i in enumerate(self.window_sample_idx):
            self._clip_buf[:, j, :, :] = preproc_list[i]
        return self._clip_buf

    # ------------------------ depth / bev ------------------------
    def update_depth_and_bev(self, frame_idx: int, frame: np.ndarray, detections: List[Detection]):
        if not self.enable_bev_render:
            return

        depth = self.last_depth

        need_new_depth = (
            self.depth_runner is not None
            and ((frame_idx % self.depth_every_n) == 0 or self.last_depth is None)
        )

        if need_new_depth:
            undistorted_frame = cv2.remap(
                frame,
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
                    frame=frame,
                    depth=depth,
                    detections=detections,
                )

                self.last_depth = depth

            except Exception as e:
                print(f"[depth] ERROR: {e}")
                depth = self.last_depth

        self.last_bev = render_bev(
            frame_bgr=frame,
            detections=detections,
            predictions=self.current_predictions,
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


    # ------------------------ main run ------------------------
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

            clip_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip_rgb_resized = cv2.resize(
                clip_rgb,
                (self.runner.resolution, self.runner.resolution),
                interpolation=cv2.INTER_LINEAR,
            )
            self.preproc_buffer.append(self.runner.preprocess_rgb_resized(clip_rgb_resized))

            detections = self.get_detections(frame_idx, frame)
            self.detection_history.append((frame_idx, detections))

            if len(self.frame_buffer) == self.window_size and frame_idx >= next_infer_frame_idx:
                tracked_detections = self.build_track_sequences()
                if len(tracked_detections) > 0:
                    clip_cthw = self.build_clip()
                    try:
                        self.infer_queue.put_nowait((frame_idx, clip_cthw, tracked_detections))
                    except queue.Full:
                        pass
                else:
                    self.current_predictions = {}

                next_infer_frame_idx = frame_idx + self.infer_stride

            if self.enable_bev_render:
                self.update_depth_and_bev(frame_idx, frame, detections)

                if self.bev_writer is not None and self.last_bev is not None:
                    self.bev_writer.write(self.last_bev)

            rendered = None
            if self.enable_overlay_render:
                rendered = draw_predictions(
                    frame=frame.copy(),
                    detections=detections,
                    predictions=self.current_predictions,
                    camera_config=self.camera_config,
                    bev_config=self.bev_config,
                )

                if (self.out_width, self.out_height) != (self.width, self.height):
                    rendered = cv2.resize(
                        rendered,
                        (self.out_width, self.out_height),
                        interpolation=cv2.INTER_LINEAR,
                    )

                if self.overlay_writer is not None:
                    self.overlay_writer.write(rendered)

            if self.display:
                if rendered is not None:
                    cv2.imshow("crossing-demo", rendered)
                if self.last_bev is not None:
                    cv2.imshow("bev-demo", self.last_bev)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    self.stop_event.set()
                    break

        t_reader.join(timeout=1.0)
        t_infer.join(timeout=1.0)
        self.cap.release()

        if self.overlay_writer is not None:
            self.overlay_writer.release()

        if self.bev_writer is not None:
            self.bev_writer.release()

        if self.display:
            cv2.destroyAllWindows()