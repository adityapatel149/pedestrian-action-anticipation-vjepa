from py_app.runners.pytorch_runner import PyTorchRunner
from py_app.runners.onnx_runner import ONNXRunner
from py_app.runners.tensorrt_runner import TensorRTRunner
from py_app.core.config import get_model_format

def build_runner(
    config_path: str,
    encoder_model: str,
    classifier_model: str,
    sweep_idx: int = 0,
    device: str = "cuda:0",
    use_fp16: bool = True,
):
    enc_fmt = get_model_format(encoder_model)
    cls_fmt = get_model_format(classifier_model)

    if enc_fmt != cls_fmt:
        raise ValueError(
            f"Mixed backends are not supported. Got encoder={enc_fmt}, classifier={cls_fmt}"
        )

    if enc_fmt == "pt":
        return PyTorchRunner(
            config_path=config_path,
            encoder_model=encoder_model,
            classifier_model=classifier_model,
            sweep_idx=sweep_idx,
            device=device,
            use_fp16=use_fp16,
        )

    if enc_fmt == "onnx":
        return ONNXRunner(
            config_path=config_path,
            encoder_model=encoder_model,
            classifier_model=classifier_model,
            device=device,
        )

    if enc_fmt == "engine":
        return TensorRTRunner(
            config_path=config_path,
            encoder_model=encoder_model,
            classifier_model=classifier_model,
            device=device,
        )

    raise ValueError(f"Unsupported backend: {enc_fmt}")
