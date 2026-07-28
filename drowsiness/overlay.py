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
) -> None:
    cv2.putText(
        frame,
        text,
        (20, 30 + row * 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_status_overlay(
    frame: np.ndarray,
    eye_state: EyeState,
    perclos_state: PerclosState,
    *,
    right_ear: float | None,
    left_ear: float | None,
    detection_score: float | None,
    fps: float,
) -> None:
    """Draw the state-machine output and its supporting metrics."""
    panel_width = min(frame.shape[1], 620)
    panel_height = min(frame.shape[0], 270)
    panel = frame[:panel_height, :panel_width]
    shade = np.zeros_like(panel)
    cv2.addWeighted(shade, 0.55, panel, 0.45, 0.0, panel)

    color = state_color(eye_state.status)
    draw_text(frame, f"STATE: {eye_state.status}", 0, color)

    ear_text = f"EAR: {format_optional(eye_state.ear)}"
    if right_ear is not None and left_ear is not None:
        ear_text += f"  R: {right_ear:.3f}  L: {left_ear:.3f}"
    draw_text(frame, ear_text, 1)

    if not eye_state.calibrated:
        percent = int(eye_state.calibration_progress * 100)
        draw_text(
            frame,
            f"CALIBRATION: {percent}% - KEEP EYES OPEN",
            2,
            color,
        )
    else:
        draw_text(
            frame,
            "BASE: "
            f"{format_optional(eye_state.baseline_ear)}  "
            "REL: "
            f"{format_optional(eye_state.relative_ear, 2)}",
            2,
        )
        draw_text(
            frame,
            "THRESH CLOSE/OPEN: "
            f"{format_optional(eye_state.closed_threshold)} / "
            f"{format_optional(eye_state.reopen_threshold)}",
            3,
        )
        draw_text(
            frame,
            f"CLOSED: {eye_state.closed_seconds:.2f}s",
            4,
            color,
        )

    if perclos_state.is_warning:
        perclos_color = (0, 0, 255)
    elif perclos_state.is_caution:
        perclos_color = (0, 255, 255)
    else:
        perclos_color = WHITE

    draw_text(
        frame,
        f"PERCLOS: {perclos_state.perclos:.2f}",
        5,
        perclos_color,
    )
    draw_text(
        frame,
        f"FACE SCORE: {format_optional(detection_score, 2)}",
        6,
    )
    draw_text(frame, f"FPS: {fps:.1f}", 7)
    draw_text(frame, "q: quit  r: recalibrate", 8)
