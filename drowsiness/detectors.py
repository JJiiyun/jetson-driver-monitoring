from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _opencv_dnn_backend(device: str) -> tuple[int, int]:
    if device == "cpu":
        return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
    if device == "cuda":
        return cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA
    raise ValueError(f"Unsupported OpenCV DNN device: {device}")


@dataclass(frozen=True)
class FaceDetection:
    box: tuple[int, int, int, int]
    score: float


class YuNetFaceDetector:
    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int],
        score_threshold: float = 0.8,
        device: str = "cpu",
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"YuNet model not found: {path}")
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("This OpenCV build has no FaceDetectorYN.")

        backend_id, target_id = _opencv_dnn_backend(device)
        self._detector = cv2.FaceDetectorYN.create(
            str(path),
            "",
            input_size,
            score_threshold,
            0.3,
            5000,
            backend_id,
            target_id,
        )

    def detect_largest(self, frame: np.ndarray) -> FaceDetection | None:
        height, width = frame.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None

        face = max(faces, key=lambda item: float(item[2] * item[3]))
        x, y, w, h = face[:4].astype(int)
        x = max(0, x)
        y = max(0, y)
        w = min(width - x, max(1, w))
        h = min(height - y, max(1, h))
        return FaceDetection(
            box=(x, y, w, h),
            score=float(face[-1]),
        )


class PFLDLandmarkDetector:
    def __init__(
        self,
        model_path: str | Path,
        input_size: tuple[int, int] = (112, 112),
        face_margin: float = 0.15,
        device: str = "cpu",
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"PFLD model not found: {path}")
        if face_margin < 0:
            raise ValueError("face_margin must not be negative.")

        self._net = cv2.dnn.readNetFromONNX(str(path))
        backend_id, target_id = _opencv_dnn_backend(device)
        self._net.setPreferableBackend(backend_id)
        self._net.setPreferableTarget(target_id)
        self._input_size = input_size
        self._face_margin = float(face_margin)

    def detect(
        self,
        frame: np.ndarray,
        face_box: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
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
        self._net.setInput(blob)
        output = np.asarray(self._net.forward(), dtype=np.float32).reshape(-1)
        if output.size != 136:
            raise RuntimeError(
                "Expected 136 PFLD output values for 68 landmarks, "
                f"got {output.size}."
            )

        normalized = output.reshape(68, 2)
        crop_width = x2 - x1
        crop_height = y2 - y1
        landmarks = normalized.copy()
        landmarks[:, 0] = x1 + normalized[:, 0] * crop_width
        landmarks[:, 1] = y1 + normalized[:, 1] * crop_height
        return landmarks, crop_box

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


class PFLDTensorRTDetector:
    """Run the PFLD landmark model with a TensorRT engine and PyCUDA."""

    def __init__(
        self,
        engine_path: str | Path,
        input_size: tuple[int, int] = (112, 112),
        face_margin: float = 0.15,
    ) -> None:
        path = Path(engine_path)
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"PFLD TensorRT engine not found: {path}")
        if face_margin < 0:
            raise ValueError("face_margin must not be negative.")

        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt
        except Exception as error:
            raise RuntimeError(
                "Could not initialize TensorRT/PyCUDA. Activate zzmvenv, "
                "verify pycuda==2022.1, and check CUDA GPU access."
            ) from error

        self._cuda = cuda
        self._trt = trt
        self._input_size = input_size
        self._face_margin = float(face_margin)
        self._logger = trt.Logger(trt.Logger.WARNING)

        runtime = trt.Runtime(self._logger)
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {path}")
        if engine.num_bindings != 2:
            raise RuntimeError(
                "Expected one PFLD input and one output binding, "
                f"got {engine.num_bindings}."
            )

        self._runtime = runtime
        self._engine = engine
        self._context = engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Could not create TensorRT execution context.")

        self._input_index = next(
            (index for index in range(engine.num_bindings)
             if engine.binding_is_input(index)),
            -1,
        )
        self._output_index = next(
            (index for index in range(engine.num_bindings)
             if not engine.binding_is_input(index)),
            -1,
        )
        if self._input_index < 0 or self._output_index < 0:
            raise RuntimeError("TensorRT engine bindings are invalid.")

        expected_shape = (1, 3, input_size[1], input_size[0])
        if -1 in tuple(engine.get_binding_shape(self._input_index)):
            if not self._context.set_binding_shape(
                self._input_index, expected_shape
            ):
                raise RuntimeError(
                    f"Could not set TensorRT input shape to {expected_shape}."
                )
        input_shape = tuple(self._context.get_binding_shape(self._input_index))
        output_shape = tuple(
            self._context.get_binding_shape(self._output_index)
        )
        if input_shape != expected_shape:
            raise RuntimeError(
                f"Expected TensorRT input shape {expected_shape}, "
                f"got {input_shape}."
            )

        input_dtype = trt.nptype(engine.get_binding_dtype(self._input_index))
        output_dtype = trt.nptype(engine.get_binding_dtype(self._output_index))
        input_count = int(np.prod(input_shape))
        output_count = int(np.prod(output_shape))
        if output_count != 136:
            raise RuntimeError(
                "Expected 136 PFLD output values for 68 landmarks, "
                f"got shape {output_shape}."
            )

        self._host_input = cuda.pagelocked_empty(input_count, input_dtype)
        self._host_output = cuda.pagelocked_empty(output_count, output_dtype)
        self._device_input = cuda.mem_alloc(self._host_input.nbytes)
        self._device_output = cuda.mem_alloc(self._host_output.nbytes)
        self._bindings = [0] * engine.num_bindings
        self._bindings[self._input_index] = int(self._device_input)
        self._bindings[self._output_index] = int(self._device_output)
        self._stream = cuda.Stream()

    def detect(
        self,
        frame: np.ndarray,
        face_box: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        crop_box = PFLDLandmarkDetector._square_crop_box(
            self, frame.shape, face_box
        )
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
        np.copyto(self._host_input, blob.reshape(-1), casting="no")
        self._cuda.memcpy_htod_async(
            self._device_input, self._host_input, self._stream
        )
        if not self._context.execute_async_v2(
            bindings=self._bindings,
            stream_handle=self._stream.handle,
        ):
            raise RuntimeError("TensorRT PFLD inference failed.")
        self._cuda.memcpy_dtoh_async(
            self._host_output, self._device_output, self._stream
        )
        self._stream.synchronize()

        normalized = np.asarray(
            self._host_output, dtype=np.float32
        ).reshape(68, 2)
        if not np.isfinite(normalized).all():
            raise RuntimeError("TensorRT PFLD returned non-finite values.")
        landmarks = normalized.copy()
        landmarks[:, 0] = x1 + normalized[:, 0] * (x2 - x1)
        landmarks[:, 1] = y1 + normalized[:, 1] * (y2 - y1)
        return landmarks, crop_box
