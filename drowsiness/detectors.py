from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"YuNet model not found: {path}")
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("This OpenCV build has no FaceDetectorYN.")

        self._detector = cv2.FaceDetectorYN.create(
            str(path),
            "",
            input_size,
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5000,
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
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"PFLD model not found: {path}")
        if face_margin < 0:
            raise ValueError("face_margin must not be negative.")

        self._net = cv2.dnn.readNetFromONNX(str(path))
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
