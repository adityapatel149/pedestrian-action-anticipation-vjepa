from typing import List, Optional
import os
import numpy as np

from py_app.tracking.bbox_csv import FramewiseBBoxCSV
from py_app.tracking.yolo_tracker import YOLOPedTracker
from py_app.core.datatypes import Detection


class TrackingNode:
    def __init__(
        self,
        runner,
        *,
        frame_width: int,
        frame_height: int,
        bbox_csv: Optional[str] = None,
        video_path: Optional[str] = None,
        detector_name: str = "yolo26n.pt",
        detector_conf: float = 0.25,
        max_boxes: int = 10,
        detector_imgsz: int = 416,
        tracker_cfg: str = "bytetrack.yaml",
    ):
        self.runner = runner
        self.width = int(frame_width)
        self.height = int(frame_height)
        self.max_boxes = int(max_boxes)

        self.detector = None
        self.bbox_source = None

        if bbox_csv is not None:
            if not self.runner.data_base_path:
                raise ValueError("runner.data_base_path is required when using bbox_csv")
            if not video_path:
                raise ValueError("video_path is required when using bbox_csv")

            video_id = os.path.splitext(
                os.path.relpath(video_path, self.runner.data_base_path)
            )[0].replace("\\", "/")

            self.bbox_source = FramewiseBBoxCSV(
                csv_path=bbox_csv,
                video_id=video_id,
                frame_width=self.width,
                frame_height=self.height,
                max_boxes=self.max_boxes,
            )
        else:
            self.detector = YOLOPedTracker(
                model_name=detector_name,
                conf=detector_conf,
                max_boxes=self.max_boxes,
                device=str(runner.device),
                imgsz=detector_imgsz,
                tracker_cfg=tracker_cfg,
            )

    def get_detections(self, frame_idx: int, frame_bgr: np.ndarray) -> List[Detection]:
        if self.bbox_source is not None:
            return self.bbox_source.get(frame_idx)[: self.max_boxes]
        if self.detector is None:
            return []
        return self.detector.get(frame_bgr)