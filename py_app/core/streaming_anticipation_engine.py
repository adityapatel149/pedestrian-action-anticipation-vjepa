from typing import List, Tuple, Dict, Optional, Deque
from collections import deque

import cv2
import numpy as np

from py_app.core.datatypes import Detection, Prediction, EngineOutput
from py_app.core.risk import estimate_risk
from py_app.visualization.bev import compute_ground_distance


class StreamingAnticipationEngine:
    """
    Anticipation-only streaming engine.

    Detection is handled outside this class.
    Visualization/debug rendering is handled outside this class.
    This class owns temporal buffering + anticipation inference + structured outputs.
    """

    def __init__(
        self,
        runner,
        *,
        src_fps: float,
        frame_width: int,
        frame_height: int,
        max_boxes: int = 10,
        stride_overlap: float = 0.30,
        anticipation_time: float = 1.0,
        camera_config=None,
        bev_config=None,
    ):
        self.runner = runner
        self.src_fps = float(src_fps or 30.0)
        self.width = int(frame_width)
        self.height = int(frame_height)
        self.max_boxes = int(max_boxes)
        self.anticipation_time = float(anticipation_time)

        self.camera_config = camera_config
        self.bev_config = bev_config

        self.window_size = max(1, int(round(0.5 * self.src_fps)))
        self.encoder_frames = runner.frames_per_clip
        self.infer_stride = max(1, int(round(self.window_size * (1.0 - stride_overlap))))
        self.window_sample_idx = np.linspace(
            0, self.window_size - 1, self.encoder_frames
        ).round().astype(np.int64)

        self.frame_buffer: Deque[Tuple[int, np.ndarray]] = deque(maxlen=self.window_size)
        self.preproc_buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)
        self.detection_history: Deque[Tuple[int, List[Detection]]] = deque(maxlen=self.window_size)

        self.current_predictions: Dict[int, Prediction] = {}
        self.next_infer_frame_idx = self.window_size - 1
        self._clip_buf: Optional[np.ndarray] = None

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
        if self._clip_buf is None:
            self._clip_buf = np.empty(
                (3, self.encoder_frames, self.runner.resolution, self.runner.resolution),
                dtype=np.float32,
            )

        preproc_list = list(self.preproc_buffer)
        for j, i in enumerate(self.window_sample_idx):
            self._clip_buf[:, j, :, :] = preproc_list[i]
        return self._clip_buf

    def infer(self, tracked_detections: List[Detection]):
        if len(tracked_detections) == 0:
            self.current_predictions = {}
            return

        clip_cthw = self.build_clip()
        preds = self.runner.predict(
            clip_cthw,
            tracked_detections,
            anticipation_time_sec=self.anticipation_time,
        )
        self.current_predictions = {p.track_id: p for p in preds}

    def run_inference(self, frame_idx: int):
        if len(self.frame_buffer) < self.window_size:
            return
        if frame_idx < self.next_infer_frame_idx:
            return

        tracked_detections = self.build_track_sequences()
        self.infer(tracked_detections)
        self.next_infer_frame_idx = frame_idx + self.infer_stride

    def _update_prediction_metrics(
        self,
        frame_bgr: np.ndarray,
        detections: List[Detection],
    ) -> None:
        for det in detections:
            pred = self.current_predictions.get(det.track_id)
            if pred is None:
                continue

            pred.distance_m = None

            if self.camera_config is None or self.bev_config is None:
                continue

            geom = compute_ground_distance(
                frame_bgr,
                det,
                self.camera_config,
                self.bev_config,
            )
            if geom is None:
                continue

            _foot_x, _foot_y, x_m, _z_m, dist_m = geom
            if dist_m is None or not np.isfinite(dist_m):
                continue

            pred.distance_m = float(dist_m)
            pred.risk_score = float(
                estimate_risk(
                    cross_prob=pred.cross_prob,
                    dist_m=dist_m,
                    max_range_m=self.bev_config.max_range_m,
                    x_m=x_m,
                )
            )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        detections: List[Detection],
    ) -> EngineOutput:
        self.frame_buffer.append((frame_idx, frame_bgr))

        clip_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        clip_rgb_resized = cv2.resize(
            clip_rgb,
            (self.runner.resolution, self.runner.resolution),
            interpolation=cv2.INTER_LINEAR,
        )
        self.preproc_buffer.append(self.runner.preprocess_rgb_resized(clip_rgb_resized))

        self.detection_history.append((frame_idx, detections))
        self.run_inference(frame_idx)
        self._update_prediction_metrics(frame_bgr, detections)

        return EngineOutput(
            detections=detections,
            predictions=list(self.current_predictions.values()),
            overlay=None,
            bev=None,
        )