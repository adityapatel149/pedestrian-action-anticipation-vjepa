import numpy as np
import cv2
from typing import Dict, List

from py_app.core.datatypes import Detection, OverlayPrediction, CameraConfig, BevConfig
from py_app.visualization.bev import compute_ground_distance
from py_app.core.risk import estimate_risk, risk_to_color


def draw_predictions(
    frame: np.ndarray,
    detections: List[Detection],
    predictions: Dict[int, OverlayPrediction],
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
        geom = compute_ground_distance(frame, det, camera_config, bev_config)

        dist_text = "dist=?"
        risk_score = 0.0

        if geom is not None:
            _foot_x, _foot_y, x_m, _z_m, dist_m = geom
            dist_text = f"{dist_m:.1f}m"
        else:
            x_m, dist_m = None, None

        if pred is None:
            label = f"no-pred {dist_text}"
            color = (255, 255, 255)
        else:
            risk_score = 0.0 if dist_m is None else estimate_risk(
                cross_prob=pred.cross_prob,
                dist_m=dist_m,
                max_range_m=bev_config.max_range_m,
                x_m=x_m,
            )
            pred.risk_score = risk_score
            label = f"P={pred.cross_prob:.2f} Risk={risk_score:.2f} {dist_text}"
            color = risk_to_color(risk_score)

        cv2.rectangle(canvas, (px1, py1), (px2, py2), color, 2)
        cv2.rectangle(canvas, (px1, max(0, py1 - 24)), (min(w - 1, px1 + 260), py1), color, -1)
        cv2.putText(canvas, label, (px1 + 5, max(14, py1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas