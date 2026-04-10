import torch
import numpy as np
from typing import List, Dict, Set, Optional

from py_app.runners.base_runner import BaseRunner
from py_app.core.config import load_runtime_config, compute_softmax
from py_app.core.datatypes import Detection, Prediction


class TensorRTRunner(BaseRunner):
    def __init__(
        self,
        config_path: str,
        encoder_model: str,
        classifier_model: str,
        device: str = "cuda:0",
    ):
        import tensorrt as trt

        runtime_cfg = load_runtime_config(config_path)

        self.data_base_path = runtime_cfg["data_base_path"]
        self.frames_per_clip = runtime_cfg["frames_per_clip"]
        self.frames_per_second = runtime_cfg["frames_per_second"]
        self.resolution = runtime_cfg["resolution"]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("TensorRTRunner requires CUDA.")

        self._tensor_cache = {}
        self.stream = torch.cuda.current_stream(device=self.device)

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)

        self.encoder_engine = self._load_engine(encoder_model)
        self.classifier_engine = self._load_engine(classifier_model)

        self.encoder_context = self.encoder_engine.create_execution_context()
        self.classifier_context = self.classifier_engine.create_execution_context()

        if self.encoder_context is None:
            raise RuntimeError(f"Failed to create execution context for {encoder_model}")
        if self.classifier_context is None:
            raise RuntimeError(f"Failed to create execution context for {classifier_model}")

        print("[TensorRTRunner] Loaded TensorRT engines successfully")

    def _load_engine(self, engine_path: str):
        with open(engine_path, "rb") as f, self.trt.Runtime(self.logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
        return engine

    def _get_torch_dtype(self, dtype):
        import tensorrt as trt

        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT8: torch.int8,
            trt.DataType.INT32: torch.int32,
            trt.DataType.BOOL: torch.bool,
            getattr(trt.DataType, "INT64", None): torch.int64,
        }
        if dtype not in mapping or mapping[dtype] is None:
            raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
        return mapping[dtype]

    def _ensure_cuda_tensor(self, x, dtype):
        if isinstance(x, np.ndarray):
            if not x.flags["C_CONTIGUOUS"]:
                x = np.ascontiguousarray(x)
            t = torch.from_numpy(x)
        else:
            t = x

        if t.dtype != dtype:
            t = t.to(dtype=dtype)

        if t.device != self.device:
            t = t.to(device=self.device, non_blocking=False)

        return t.contiguous()

    def _run_engine(self, engine, context, inputs, output_names=None):
        bindings = {}
        requested_outputs = set(output_names) if output_names is not None else None

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)

            if mode == self.trt.TensorIOMode.INPUT:
                x = inputs[name]
                trt_dtype = engine.get_tensor_dtype(name)
                torch_dtype = self._get_torch_dtype(trt_dtype)
                x = self._ensure_cuda_tensor(x, torch_dtype)
                context.set_input_shape(name, tuple(x.shape))
                bindings[name] = x

        unresolved = context.infer_shapes()
        if unresolved:
            raise RuntimeError(f"TensorRT shape inference unresolved tensors: {unresolved}")

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)

            if mode == self.trt.TensorIOMode.OUTPUT:
                shape = tuple(context.get_tensor_shape(name))
                trt_dtype = engine.get_tensor_dtype(name)
                torch_dtype = self._get_torch_dtype(trt_dtype)
                key = (id(engine), name, shape, torch_dtype)
                out = self._tensor_cache.get(key)
                if out is None:
                    out = torch.empty(shape, dtype=torch_dtype, device=self.device)
                    self._tensor_cache[key] = out
                bindings[name] = out

        for name, tensor in bindings.items():
            context.set_tensor_address(name, int(tensor.data_ptr()))

        ok = context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed.")

        return {
            name: tensor
            for name, tensor in bindings.items()
            if engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT
            and (requested_outputs is None or name in requested_outputs)
        }

    @torch.inference_mode()
    def predict(self, clip_cthw, detections, anticipation_time_sec=1.0):
        if len(detections) == 0:
            return []

        clip = np.expand_dims(clip_cthw.astype(np.float32, copy=False), axis=0)
        ant = np.array([anticipation_time_sec], dtype=np.float32)

        encoder_out = self._run_engine(
            self.encoder_engine,
            self.encoder_context,
            {"clip": clip, "anticipation_times": ant},
            output_names={"features"},
        )

        features = encoder_out["features"]
        features = features.expand(len(detections), *features.shape[1:]).contiguous()

        bboxes = self.build_bbox_tensor(detections).to(self.device, non_blocking=False)

        classifier_out = self._run_engine(
            self.classifier_engine,
            self.classifier_context,
            {"features": features, "bboxes": bboxes},
            output_names={"cross"},
        )

        cross_probs = torch.softmax(classifier_out["cross"].float(), dim=-1)[..., 1]
        probs = cross_probs.detach().cpu().numpy()

        return [
            Prediction(
                track_id=det.track_id,
                cross_prob=float(prob),
            )
            for det, prob in zip(detections, probs)
        ]