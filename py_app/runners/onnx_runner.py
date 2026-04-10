import torch
import onnxruntime as ort
import numpy as np
from typing import List, Dict

from py_app.runners.base_runner import BaseRunner
from py_app.core.config import load_runtime_config, compute_softmax
from py_app.core.datatypes import Detection, OverlayPrediction

class ONNXRunner(BaseRunner):
    def __init__(
        self,
        config_path: str,
        encoder_model: str,
        classifier_model: str,
        device: str = "cuda:0",
    ):
        runtime_cfg = load_runtime_config(config_path)

        self.data_base_path = runtime_cfg["data_base_path"]
        self.frames_per_clip = runtime_cfg["frames_per_clip"]
        self.frames_per_second = runtime_cfg["frames_per_second"]
        self.resolution = runtime_cfg["resolution"]

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        use_cuda = self.device.type == "cuda"

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        self.encoder_sess = ort.InferenceSession(encoder_model, providers=providers)
        self.classifier_sess = ort.InferenceSession(classifier_model, providers=providers)

        print(f"[ONNXRunner] Providers: {self.encoder_sess.get_providers()}")

    def predict(
        self,
        clip_cthw: np.ndarray,
        detections: List[Detection],
        anticipation_time_sec: float = 1.0,
    ) -> List[OverlayPrediction]:
        if len(detections) == 0:
            return []

        clip = np.expand_dims(clip_cthw.astype(np.float32, copy=False), axis=0)
        ant = np.array([anticipation_time_sec], dtype=np.float32)

        features = self.encoder_sess.run(
            ["features"],
            {"clip": clip, "anticipation_times": ant},
        )[0]

        features = np.repeat(features, repeats=len(detections), axis=0)
        bboxes = self.build_bbox_array(detections)

        cross = self.classifier_sess.run(
            ["cross"],
            {"features": features.astype(np.float32, copy=False), "bboxes": bboxes},
        )[0]

        cross_probs = compute_softmax(cross, axis=-1)[..., 1].tolist()
        return [
            OverlayPrediction(track_id=det.track_id, cross_prob=float(prob))
            for det, prob in zip(detections, cross_probs)
        ]