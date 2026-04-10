import numpy as np
import cv2
from typing import Dict, List

from py_app.core.datatypes import Detection, Prediction, CameraConfig, BevConfig
from py_app.visualization.bev import compute_ground_distance
from py_app.core.risk import risk_to_color


def draw_predictions(
    frame: np.ndarray,
    detections: List[Detection],
    predictions: Dict[int, Prediction],
    camera_config: CameraConfig,
    bev_config: BevConfig,
) -> np.ndarray:
    canvas = frame
    h, w = canvas.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy_norm
        px1 = int(np.clip(x1, 0, 1) * w)
        py1 = int(np.clip(y1, 0, 1) * h)
        px2 = int(np.clip(x2, 0, 1) * w)
        py2 = int(np.clip(y2, 0, 1) * h)

        pred = predictions.get(det.track_id)

        dist_m = None
        if pred is not None and pred.distance_m is not None:
            dist_m = pred.distance_m
        else:
            geom = compute_ground_distance(frame, det, camera_config, bev_config)
            if geom is not None:
                _foot_x, _foot_y, _x_m, _z_m, dist_m = geom

        dist_text = "dist=?" if dist_m is None else f"{dist_m:.1f}m"

        if pred is None:
            label = f"no-pred {dist_text}"
            color = (255, 255, 255)
        else:
            color = risk_to_color(pred.risk_score)
            label = f"P={pred.cross_prob:.2f} Risk={pred.risk_score:.2f} {dist_text}"

        cv2.rectangle(canvas, (px1, py1), (px2, py2), color, 2)
        cv2.rectangle(
            canvas,
            (px1, max(0, py1 - 24)),
            (min(w - 1, px1 + 260), py1),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (px1 + 5, max(14, py1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas