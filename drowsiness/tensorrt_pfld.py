from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np


class TensorRTPFLDLandmarkDetector:
    """Run a PFLD TensorRT engine with the same interface as the OpenCV detector."""

    def __init__(
        self,
        engine_path: str | Path,
        input_size: tuple[int, int] = (112, 112),
        face_margin: float = 0.15,
        device_id: int = 0,
    ) -> None:
        path = Path(engine_path)
        if not path.exists():
            raise FileNotFoundError(f"PFLD TensorRT engine not found: {path}")
        if face_margin < 0:
            raise ValueError("face_margin must not be negative.")

        try:
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError(
                "TensorRT PFLD requires the tensorrt and pycuda packages."
            ) from error

        self._cuda = cuda
        self._trt = trt
        self._input_size = input_size
        self._face_margin = float(face_margin)
        self._lock = Lock()
        self._closed = False

        self._cuda_context: Any = None
        self._runtime: Any = None
        self._engine: Any = None
        self._execution_context: Any = None
        self._stream: Any = None
        self._input_device: Any = None
        self._output_device: Any = None

        cuda.init()
        if device_id < 0 or device_id >= cuda.Device.count():
            raise ValueError(
                f"CUDA device {device_id} does not exist "
                f"(available devices: {cuda.Device.count()})."
            )

        self._cuda_context = cuda.Device(device_id).retain_primary_context()
        try:
            self._cuda_context.push()
            try:
                self._initialize_engine(path)
            except Exception:
                self._release_cuda_resources()
                raise
            finally:
                cuda.Context.pop()
        except Exception:
            self._cuda_context.detach()
            self._cuda_context = None
            raise

    def _initialize_engine(self, path: Path) -> None:
        trt = self._trt
        cuda = self._cuda

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        self._runtime = trt.Runtime(logger)
        self._engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {path}")

        self._execution_context = self._engine.create_execution_context()
        if self._execution_context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        input_indices = [
            index
            for index in range(self._engine.num_bindings)
            if self._engine.binding_is_input(index)
        ]
        output_indices = [
            index
            for index in range(self._engine.num_bindings)
            if not self._engine.binding_is_input(index)
        ]
        if len(input_indices) != 1 or len(output_indices) != 1:
            raise RuntimeError(
                "PFLD engine must have exactly one input and one output; "
                f"got {len(input_indices)} inputs and {len(output_indices)} outputs."
            )

        self._input_index = input_indices[0]
        self._output_index = output_indices[0]
        input_width, input_height = self._input_size
        requested_input_shape = (1, 3, input_height, input_width)
        engine_input_shape = tuple(
            self._engine.get_binding_shape(self._input_index)
        )

        if any(dimension < 0 for dimension in engine_input_shape):
            if not self._execution_context.set_binding_shape(
                self._input_index,
                requested_input_shape,
            ):
                raise RuntimeError(
                    "TensorRT rejected PFLD input shape "
                    f"{requested_input_shape}."
                )
        elif engine_input_shape != requested_input_shape:
            raise RuntimeError(
                f"PFLD engine input shape is {engine_input_shape}, "
                f"but {requested_input_shape} was requested."
            )

        input_shape = tuple(
            self._execution_context.get_binding_shape(self._input_index)
        )
        output_shape = tuple(
            self._execution_context.get_binding_shape(self._output_index)
        )
        if any(dimension < 0 for dimension in input_shape + output_shape):
            raise RuntimeError(
                "TensorRT binding shapes are unresolved: "
                f"input={input_shape}, output={output_shape}."
            )

        input_count = int(trt.volume(input_shape))
        output_count = int(trt.volume(output_shape))
        if output_count != 136:
            raise RuntimeError(
                "Expected 136 PFLD output values for 68 landmarks, "
                f"got {output_count}."
            )

        input_dtype = self._numpy_dtype(
            self._engine.get_binding_dtype(self._input_index)
        )
        output_dtype = self._numpy_dtype(
            self._engine.get_binding_dtype(self._output_index)
        )
        self._input_host = cuda.pagelocked_empty(input_count, input_dtype)
        self._output_host = cuda.pagelocked_empty(output_count, output_dtype)
        self._input_device = cuda.mem_alloc(self._input_host.nbytes)
        self._output_device = cuda.mem_alloc(self._output_host.nbytes)
        self._bindings = [0] * self._engine.num_bindings
        self._bindings[self._input_index] = int(self._input_device)
        self._bindings[self._output_index] = int(self._output_device)
        self._stream = cuda.Stream()

    def _numpy_dtype(self, tensor_dtype: Any) -> np.dtype[Any]:
        """Map TensorRT 8 data types without using its NumPy 1.24-incompatible helper."""
        trt = self._trt
        dtype_map = {
            trt.DataType.FLOAT: np.dtype(np.float32),
            trt.DataType.HALF: np.dtype(np.float16),
            trt.DataType.INT8: np.dtype(np.int8),
            trt.DataType.INT32: np.dtype(np.int32),
            trt.DataType.BOOL: np.dtype(np.bool_),
        }
        try:
            return dtype_map[tensor_dtype]
        except KeyError as error:
            raise RuntimeError(
                f"Unsupported TensorRT binding data type: {tensor_dtype}."
            ) from error

    def detect(
        self,
        frame: np.ndarray,
        face_box: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        if self._closed:
            raise RuntimeError("TensorRT PFLD detector is already closed.")

        crop_box = self._square_crop_box(frame.shape, face_box)
        x1, y1, x2, y2 = crop_box
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            raise RuntimeError("PFLD face crop is empty.")

        blob = cv2.dnn.blobFromImage(
            face_crop,
            scalefactor=1.0 / 255.0,
            size=self._input_size,
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        np.copyto(
            self._input_host,
            np.asarray(blob, dtype=self._input_host.dtype).reshape(-1),
        )

        with self._lock:
            self._cuda_context.push()
            try:
                self._cuda.memcpy_htod_async(
                    self._input_device,
                    self._input_host,
                    self._stream,
                )
                succeeded = self._execution_context.execute_async_v2(
                    bindings=self._bindings,
                    stream_handle=self._stream.handle,
                )
                if not succeeded:
                    raise RuntimeError("TensorRT PFLD inference failed.")
                self._cuda.memcpy_dtoh_async(
                    self._output_host,
                    self._output_device,
                    self._stream,
                )
                self._stream.synchronize()
            finally:
                self._cuda.Context.pop()

        output = np.asarray(self._output_host, dtype=np.float32)
        normalized = output.reshape(68, 2)
        crop_width = x2 - x1
        crop_height = y2 - y1
        landmarks = normalized.copy()
        landmarks[:, 0] = x1 + normalized[:, 0] * crop_width
        landmarks[:, 1] = y1 + normalized[:, 1] * crop_height
        return landmarks, crop_box

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._cuda_context is None:
            return

        with self._lock:
            self._cuda_context.push()
            try:
                self._release_cuda_resources()
            finally:
                self._cuda.Context.pop()
                self._cuda_context.detach()
                self._cuda_context = None

    def _release_cuda_resources(self) -> None:
        if self._input_device is not None:
            self._input_device.free()
            self._input_device = None
        if self._output_device is not None:
            self._output_device.free()
            self._output_device = None
        self._stream = None
        self._execution_context = None
        self._engine = None
        self._runtime = None

    def __enter__(self) -> TensorRTPFLDLandmarkDetector:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _square_crop_box(
        self,
        frame_shape: tuple[int, ...],
        face_box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        frame_height, frame_width = frame_shape[:2]
        x, y, width, height = face_box
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        size = max(width, height) * (1.0 + 2.0 * self._face_margin)

        x1 = max(0, int(round(center_x - size / 2.0)))
        y1 = max(0, int(round(center_y - size / 2.0)))
        x2 = min(frame_width, int(round(center_x + size / 2.0)))
        y2 = min(frame_height, int(round(center_y + size / 2.0)))
        return x1, y1, x2, y2
