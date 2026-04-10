import numpy as np
import torch
from typing import List

from py_app.core.datatypes import Detection


class BaseRunner:
    def preprocess_rgb_resized(self, rgb_resized: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_preproc_chw"):
            self._preproc_chw = np.empty(
                (3, self.resolution, self.resolution),
                dtype=np.float32,
            )
            self._inv255 = np.float32(1.0 / 255.0)

        np.multiply(
            np.transpose(rgb_resized, (2, 0, 1)),
            self._inv255,
            out=self._preproc_chw,
            casting="unsafe",
        )
        return self._preproc_chw

    def build_bbox_array(self, detections: List[Detection]) -> np.ndarray:
        n = len(detections)
        if n == 0:
            return np.zeros((0, self.frames_per_clip, 4), dtype=np.float32)

        need_shape = (n, self.frames_per_clip, 4)
        if not hasattr(self, "_bbox_buf") or self._bbox_buf.shape[0] < n:
            self._bbox_buf = np.empty(need_shape, dtype=np.float32)

        arr = self._bbox_buf[:n]
        for i, det in enumerate(detections):
            if det.track_sequence is None:
                raise ValueError(f"Detection track_id={det.track_id} is missing track_sequence")
            arr[i] = det.track_sequence
        return arr

    def build_bbox_tensor(self, detections: List[Detection]) -> torch.Tensor:
        return torch.from_numpy(self.build_bbox_array(detections))