import numpy as np
from typing import List

from py_app.core.datatypes import Detection


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
        inv_w = 1.0 / max(w, 1)
        inv_h = 1.0 / max(h, 1)

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

        if not results:
            return []

        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy_t = boxes.xyxy
        conf_t = boxes.conf if boxes.conf is not None else None
        id_t = boxes.id if boxes.id is not None else None

        xyxy = xyxy_t.detach().cpu().numpy()
        confs = (
            conf_t.detach().cpu().numpy()
            if conf_t is not None
            else np.ones((xyxy.shape[0],), dtype=np.float32)
        )
        ids = (
            id_t.detach().cpu().numpy().astype(np.int64, copy=False)
            if id_t is not None
            else np.arange(xyxy.shape[0], dtype=np.int64)
        )

        n = min(xyxy.shape[0], self.max_boxes)
        detections: List[Detection] = []

        for i in range(n):
            b = xyxy[i]
            detections.append(
                Detection(
                    track_id=int(ids[i]),
                    bbox_xyxy_norm=(
                        float(b[0] * inv_w),
                        float(b[1] * inv_h),
                        float(b[2] * inv_w),
                        float(b[3] * inv_h),
                    ),
                    score=float(confs[i]),
                )
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        return detections[: self.max_boxes]