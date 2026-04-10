import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict

from evals.action_anticipation_frozen.models import init_classifier, init_module
from src.utils.checkpoint_loader import robust_checkpoint_loader

from py_app.runners.base_runner import BaseRunner
from py_app.core.config import load_runtime_config
from py_app.core.datatypes import Detection, OverlayPrediction

class PyTorchRunner(BaseRunner):
    def __init__(
        self,
        config_path: str,
        encoder_model: str,
        classifier_model: str,
        sweep_idx: int = 0,
        device: str = "cuda:0",
        use_fp16: bool = True,
    ):
        runtime_cfg = load_runtime_config(config_path)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device.type == "cuda"

        self.data_base_path = runtime_cfg["data_base_path"]
        self.frames_per_clip = runtime_cfg["frames_per_clip"]
        self.frames_per_second = runtime_cfg["frames_per_second"]
        self.resolution = runtime_cfg["resolution"]
        self.num_probe_blocks = runtime_cfg["num_probe_blocks"]
        self.num_heads = runtime_cfg["num_heads"]
        self.num_classifiers = runtime_cfg["num_classifiers"]

        args_pretrain = runtime_cfg["args_pretrain"]
        args_model = runtime_cfg["args_model"]
        args_wrapper = runtime_cfg["args_wrapper"]

        cross_classes = {0: 0, 1: 1}
        action_classes = {0: 0, 1: 1}
        intersection_classes = {i: i for i in range(5)}
        signalized_classes = {i: i for i in range(4)}

        self.encoder = init_module(
            module_name=args_pretrain["module_name"],
            frames_per_clip=self.frames_per_clip,
            frames_per_second=self.frames_per_second,
            resolution=self.resolution,
            checkpoint=encoder_model,
            model_kwargs=args_model,
            wrapper_kwargs=args_wrapper,
            device=self.device,
        )
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        classifiers = init_classifier(
            embed_dim=self.encoder.embed_dim,
            num_heads=self.num_heads,
            cross_classes=cross_classes,
            action_classes=action_classes,
            intersection_classes=intersection_classes,
            signalized_classes=signalized_classes,
            num_blocks=self.num_probe_blocks,
            device=self.device,
            num_classifiers=self.num_classifiers,
        )

        checkpoint = robust_checkpoint_loader(classifier_model, map_location=torch.device("cpu"))
        if "classifiers" not in checkpoint:
            raise KeyError("Expected classifier checkpoint with key 'classifiers'.")
        if sweep_idx >= len(checkpoint["classifiers"]):
            raise IndexError(
                f"sweep_idx={sweep_idx} out of range for {len(checkpoint['classifiers'])} sweeps"
            )

        self.classifier = classifiers[sweep_idx]
        state_dict = checkpoint["classifiers"][sweep_idx]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        msg = self.classifier.load_state_dict(state_dict, strict=True)
        print(f"[PyTorchRunner] Loaded sweep {sweep_idx}: {msg}")

        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)

        if hasattr(self.classifier, "pooler") and hasattr(self.classifier.pooler, "use_activation_checkpointing"):
            self.classifier.pooler.use_activation_checkpointing = False

        del checkpoint

    @torch.inference_mode()
    def predict(
        self,
        clip_cthw: np.ndarray,
        detections: List[Detection],
        anticipation_time_sec: float = 1.0,
    ) -> List[OverlayPrediction]:
        if len(detections) == 0:
            return []

        clip = torch.from_numpy(clip_cthw).unsqueeze(0).to(self.device, non_blocking=True)
        bboxes = self.build_bbox_tensor(detections).to(self.device, non_blocking=True)
        anticipation_times = torch.tensor([anticipation_time_sec], dtype=torch.float32, device=self.device)

        with torch.cuda.amp.autocast(enabled=self.use_fp16, dtype=torch.float16):
            features = self.encoder(clip, anticipation_times)
            features = features.expand(len(detections), -1, -1)
            out = self.classifier(features, bboxes=bboxes)
            cross_probs = F.softmax(out["cross"], dim=-1)[..., 1]

        probs = cross_probs.detach().float().cpu().numpy().tolist()
        return [
            OverlayPrediction(track_id=det.track_id, cross_prob=float(prob))
            for det, prob in zip(detections, probs)
        ]