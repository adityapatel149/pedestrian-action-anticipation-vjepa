import numpy as np
import cv2
import torch

from py_app.core.config import get_model_format


class DepthAnythingRunner:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.backend = get_model_format(model_path)

        self.input_name = "image"
        self.output_name = "depth"
        self.input_h = 280
        self.input_w = 504

        self._input_bnchw = None
        self._inv255 = np.float32(1.0 / 255.0)

        if self.backend == "onnx":
            self._init_onnx()
        elif self.backend == "engine":
            self._init_tensorrt()
        else:
            raise ValueError(f"Unsupported depth backend: {self.backend}")

        self._input_bnchw = np.empty((1, 1, 3, self.input_h, self.input_w), dtype=np.float32)

        print(
            f"[DepthAnythingRunner] loaded {self.backend.upper()}: {model_path} "
            f"input_name={self.input_name} output_name={self.output_name} "
            f"expected_input=(1, 1, 3, {self.input_h}, {self.input_w})"
        )

    # -------------------- init backends --------------------

    def _init_onnx(self):
        import onnxruntime as ort

        use_cuda = self.device.type == "cuda"
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=providers)

        try:
            inp = self.session.get_inputs()[0]
            out = self.session.get_outputs()[0]
            self.input_name = inp.name
            self.output_name = out.name

            shape = inp.shape
            if len(shape) == 5 and isinstance(shape[-2], int) and isinstance(shape[-1], int):
                self.input_h = int(shape[-2])
                self.input_w = int(shape[-1])
        except Exception:
            pass

        print(f"[DepthAnythingRunner] ONNX providers={self.session.get_providers()}")

    def _init_tensorrt(self):
        import tensorrt as trt

        if self.device.type != "cuda":
            raise RuntimeError("TensorRT depth runner requires CUDA.")

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(self.model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {self.model_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {self.model_path}")

        self.stream = torch.cuda.current_stream(device=self.device)
        self._tensor_cache = {}

        input_names = []
        output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)

        if not input_names:
            raise RuntimeError("TensorRT depth engine has no input tensor.")
        if not output_names:
            raise RuntimeError("TensorRT depth engine has no output tensor.")

        self.input_name = input_names[0]
        self.output_name = output_names[0]

        # Try to infer H/W from static engine shape
        shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if len(shape) == 5 and shape[-2] > 0 and shape[-1] > 0:
            self.input_h = int(shape[-2])
            self.input_w = int(shape[-1])

    # -------------------- preprocessing --------------------

    def preprocess_bgr_resized(self, frame_bgr_resized: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)
        np.multiply(
            np.transpose(rgb, (2, 0, 1)),
            self._inv255,
            out=self._input_bnchw[0, 0],
            casting="unsafe",
        )
        return self._input_bnchw

    # -------------------- TensorRT helpers --------------------

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

    def _run_tensorrt(self, inputs):
        bindings = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == self.trt.TensorIOMode.INPUT:
                if name not in inputs:
                    raise KeyError(f"Missing TensorRT input tensor: {name}")

                trt_dtype = self.engine.get_tensor_dtype(name)
                torch_dtype = self._get_torch_dtype(trt_dtype)

                x = self._ensure_cuda_tensor(inputs[name], torch_dtype)
                self.context.set_input_shape(name, tuple(x.shape))
                bindings[name] = x

        unresolved = self.context.infer_shapes()
        if unresolved:
            raise RuntimeError(f"TensorRT shape inference unresolved tensors: {unresolved}")

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == self.trt.TensorIOMode.OUTPUT:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(dim < 0 for dim in shape):
                    raise RuntimeError(f"Invalid TensorRT output shape for '{name}': {shape}")

                trt_dtype = self.engine.get_tensor_dtype(name)
                torch_dtype = self._get_torch_dtype(trt_dtype)

                key = (name, shape, torch_dtype)
                out = self._tensor_cache.get(key)
                if out is None:
                    out = torch.empty(shape, dtype=torch_dtype, device=self.device)
                    self._tensor_cache[key] = out
                bindings[name] = out

        for name, tensor in bindings.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        ok = self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT depth execution failed.")

        return bindings[self.output_name]

    # -------------------- inference --------------------

    def _predict_onnx(self, inp: np.ndarray) -> np.ndarray:
        depth = self.session.run(
            [self.output_name],
            {self.input_name: inp},
        )[0]

        if not isinstance(depth, np.ndarray) or depth.dtype != np.float32:
            depth = np.asarray(depth, dtype=np.float32)
        return depth

    def _predict_tensorrt(self, inp: np.ndarray) -> np.ndarray:
        depth_t = self._run_tensorrt({self.input_name: inp})
        depth = depth_t.float().detach().cpu().numpy()
        return depth

    def predict_from_undistorted_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        inp_h, inp_w = self.input_h, self.input_w
        h, w = frame_bgr.shape[:2]

        if (h, w) != (inp_h, inp_w):
            frame_bgr_resized = cv2.resize(
                frame_bgr,
                (inp_w, inp_h),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            frame_bgr_resized = frame_bgr

        inp = self.preprocess_bgr_resized(frame_bgr_resized)

        if self.backend == "onnx":
            depth = self._predict_onnx(inp)
        else:
            depth = self._predict_tensorrt(inp)

        if depth.ndim == 4:
            depth = depth[0, 0]
        elif depth.ndim == 3:
            depth = depth[0]
        else:
            raise ValueError(f"Unexpected depth output shape: {depth.shape}")

        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth