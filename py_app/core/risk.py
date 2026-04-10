from typing import Optional, Tuple
import numpy as np


def estimate_risk(
    cross_prob: float,
    dist_m: float,
    max_range_m: float,
    x_m: Optional[float] = None,
) -> float:
    distance_factor = 1.0 - min(float(dist_m) / float(max_range_m), 1.0)

    if x_m is not None:
        lane_half_width = 4.0
        center_factor = 1.0 - min(abs(float(x_m)) / lane_half_width, 1.0)
    else:
        center_factor = 0.5

    risk = (
        0.25 * distance_factor +
        0.25 * center_factor +
        0.50 * cross_prob
    )
    return float(np.clip(risk, 0.0, 1.0))


def risk_to_color(risk: float) -> Tuple[int, int, int]:
    risk = float(np.clip(risk, 0.0, 1.0))
    midpoint = 0.6  # <-- new yellow point

    if risk < midpoint:
        alpha = risk / midpoint
        g, r = 255, int(255 * alpha)
    else:
        alpha = (risk - midpoint) / (1.0 - midpoint)
        g, r = int(255 * (1.0 - alpha)), 255

    return (0, g, r)