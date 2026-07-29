from __future__ import annotations

import cv2
import numpy as np

from .calibration import EyeState
from .perclos_monitor import PerclosState


WHITE = (255, 255, 255)


def state_color(status: str) -> tuple[int, int, int]:
    if status == "DANGER":
        return 0, 0, 255
    if status == "EYES CLOSED":
        return 0, 165, 255
    if status == "CALIBRATING":
        return 0, 255, 255
    if status == "NO FACE":
        return 160, 160, 160
    return 0, 255, 0


def format_optional(value: float | None, precision: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{precision}f}"


def draw_text(
    frame: np.ndarray,
    text: str,
    row: int,
    color: tuple[int, int, int] = WHITE,
    origin: tuple[int, int] = (8, 8),
) -> None:
    origin_x, origin_y = origin
    cv2.putText(
        frame,
        text,
        (origin_x + 8, origin_y + 20 + row * 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        color,
        1,
        cv2.LINE_AA,
    )


def _intersection_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    width = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
    height = max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
    return width * height


def overlay_origin(
    frame_shape: tuple[int, ...],
    panel_size: tuple[int, int],
    face_box: tuple[int, int, int, int] | None,
    margin: int = 8,
) -> tuple[int, int]:
    """Return the top corner that overlaps the detected face the least."""
    _frame_height, frame_width = frame_shape[:2]
    panel_width, panel_height = panel_size
    left = (margin, margin)
    right = (max(margin, frame_width - panel_width - margin), margin)
    if face_box is None:
        return left

    face_x, face_y, face_width, face_height = face_box
    face_rect = (
        face_x,
        face_y,
        face_x + face_width,
        face_y + face_height,
    )

    def overlap(origin: tuple[int, int]) -> int:
        panel_x, panel_y = origin
        panel_rect = (
            panel_x,
            panel_y,
            panel_x + panel_width,
            panel_y + panel_height,
        )
        return _intersection_area(panel_rect, face_rect)

    return right if overlap(right) < overlap(left) else left


def draw_status_overlay(
    frame: np.ndarray,
    eye_state: EyeState,
    perclos_state: PerclosState,
    *,
    right_ear: float | None,
    left_ear: float | None,
    detection_score: float | None,
    fps: float,
    face_box: tuple[int, int, int, int] | None = None,
) -> None:
    """Draw a compact status panel away from the detected face."""
    margin = 8
    panel_width = min(max(1, frame.shape[1] - 2 * margin), 280)
    panel_height = min(max(1, frame.shape[0] - 2 * margin), 174)
    origin = overlay_origin(
        frame.shape,
        (panel_width, panel_height),
        face_box,
        margin,
    )
    panel_x, panel_y = origin
    panel = frame[
        panel_y : panel_y + panel_height,
        panel_x : panel_x + panel_width,
    ]
    shade = np.zeros_like(panel)
    cv2.addWeighted(shade, 0.25, panel, 0.75, 0.0, panel)

    color = state_color(eye_state.status)
    draw_text(frame, f"STATE: {eye_state.status}", 0, color, origin)

    ear_text = f"EAR: {format_optional(eye_state.ear)}"
    if right_ear is not None and left_ear is not None:
        ear_text += f" R:{right_ear:.3f} L:{left_ear:.3f}"
    draw_text(frame, ear_text, 1, origin=origin)

    if not eye_state.calibrated:
        percent = int(eye_state.calibration_progress * 100)
        draw_text(
            frame,
            f"CAL: {percent}% KEEP EYES OPEN",
            2,
            color,
            origin,
        )
    else:
        draw_text(
            frame,
            "BASE:"
            f"{format_optional(eye_state.baseline_ear)} "
            "REL:"
            f"{format_optional(eye_state.relative_ear, 2)}",
            2,
            origin=origin,
        )
        draw_text(
            frame,
            "C/O: "
            f"{format_optional(eye_state.closed_threshold)}/"
            f"{format_optional(eye_state.reopen_threshold)}",
            3,
            origin=origin,
        )

    if perclos_state.is_warning:
        perclos_color = (0, 0, 255)
    elif perclos_state.is_caution:
        perclos_color = (0, 255, 255)
    else:
        perclos_color = color

    draw_text(
        frame,
        "CLOSED:"
        f"{eye_state.closed_seconds:.2f}s "
        f"PERCLOS:{perclos_state.perclos:.2f}",
        4,
        perclos_color,
        origin,
    )
    draw_text(
        frame,
        "FACE:"
        f"{format_optional(detection_score, 2)} "
        f"FPS:{fps:.1f}",
        5,
        origin=origin,
    )
    draw_text(frame, "q:quit  r:recalibrate", 6, origin=origin)
