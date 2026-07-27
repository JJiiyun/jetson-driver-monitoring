from __future__ import annotations

from collections.abc import Sequence

import numpy as np


RIGHT_EYE_INDICES = (36, 37, 38, 39, 40, 41)
LEFT_EYE_INDICES = (42, 43, 44, 45, 46, 47)
MOUTH_DISPLAY_INDICES = (62, 66)


def _as_landmarks(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 68:
        raise ValueError(
            "Expected at least 68 landmarks with shape (N, 2), "
            f"got {points.shape}."
        )
    return points


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(first - second))


def eye_aspect_ratio(
    landmarks: np.ndarray,
    indices: Sequence[int],
) -> float:
    """Calculate standard six-point EAR for one eye."""
    points = _as_landmarks(landmarks)
    if len(indices) != 6:
        raise ValueError("EAR requires exactly six eye landmark indices.")

    p1, p2, p3, p4, p5, p6 = points[list(indices)]
    vertical = _distance(p2, p6) + _distance(p3, p5)
    horizontal = 2.0 * _distance(p1, p4)
    return vertical / max(horizontal, 1e-6)


def mean_eye_aspect_ratio(landmarks: np.ndarray) -> tuple[float, float, float]:
    """Return mean, right-eye, and left-eye EAR."""
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
    return (right_ear + left_ear) / 2.0, right_ear, left_ear


def eye_display_points(
    landmarks: np.ndarray,
    indices: Sequence[int],
) -> np.ndarray:
    """Return only left, top, right, and bottom display points for one eye."""
    points = _as_landmarks(landmarks)
    if len(indices) != 6:
        raise ValueError("Eye display points require six eye indices.")

    p1, p2, p3, p4, p5, p6 = points[list(indices)]
    top = (p2 + p3) / 2.0
    bottom = (p5 + p6) / 2.0
    return np.asarray((p1, top, p4, bottom), dtype=np.float32)


def mouth_display_points(landmarks: np.ndarray) -> np.ndarray:
    """Return the inner upper-lip and lower-lip points only."""
    points = _as_landmarks(landmarks)
    return points[list(MOUTH_DISPLAY_INDICES)].copy()
