import yaml
import numpy as np
import os
from typing import Dict 

def compute_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def get_model_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        return "pt"
    if ext == ".onnx":
        return "onnx"
    if ext == ".engine":
        return "engine"
    raise ValueError(f"Unsupported model extension: {ext} for {path}")

def load_runtime_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    args_pretrain = cfg["model_kwargs"]
    args_model = args_pretrain["pretrain_kwargs"]
    args_wrapper = args_pretrain["wrapper_kwargs"]
    args_exp = cfg["experiment"]
    args_classifier = args_exp["classifier"]
    args_data = args_exp["data"]
    args_opt = args_exp["optimization"]

    return {
        "cfg": cfg,
        "args_pretrain": args_pretrain,
        "args_model": args_model,
        "args_wrapper": args_wrapper,
        "args_exp": args_exp,
        "args_classifier": args_classifier,
        "args_data": args_data,
        "args_opt": args_opt,
        "data_base_path": args_data.get("base_path"),
        "frames_per_clip": int(args_data["frames_per_clip"]),
        "frames_per_second": float(args_data["frames_per_second"]),
        "resolution": int(args_data.get("resolution", 224)),
        "num_probe_blocks": int(args_classifier.get("num_probe_blocks", 1)),
        "num_heads": int(args_classifier["num_heads"]),
        "num_classifiers": len(args_opt["multihead_kwargs"]),
    }
