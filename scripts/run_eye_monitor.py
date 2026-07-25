#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drowsiness import (
    EyeClosureMonitor,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    eye_display_points,
    mean_eye_aspect_ratio,
    mouth_display_points,
)
from drowsiness.detectors import PFLDLandmarkDetector, YuNetFaceDetector


DEFAULT_YUNET_PATH = PROJECT_ROOT / "models/face_detector/yunet.onnx"
DEFAULT_PFLD_PATH = PROJECT_ROOT / "models/landmark/pfld_sim.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time EAR calibration and eye-closure monitor."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--yunet", type=Path, default=DEFAULT_YUNET_PATH)
    parser.add_argument("--pfld", type=Path, default=DEFAULT_PFLD_PATH)
    parser.add_argument("--calibration-seconds", type=float, default=3.0)
    parser.add_argument("--closed-ratio", type=float, default=0.70)
    parser.add_argument("--danger-seconds", type=float, default=2.0)
    return parser.parse_args()


def draw_points(
    frame: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 4,
) -> None:
    for point_x, point_y in np.rint(points).astype(int):
        cv2.circle(frame, (point_x, point_y), radius, color, -1)


def draw_text(
    frame: np.ndarray,
    text: str,
    row: int,
    color: tuple[int, int, int] = (255, 255, 255),
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


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(args.camera)

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    return cap


def main() -> int:
    args = parse_args()

    try:
        face_detector = YuNetFaceDetector(
            args.yunet,
            input_size=(args.width, args.height),
        )
        landmark_detector = PFLDLandmarkDetector(args.pfld)
    except (FileNotFoundError, RuntimeError, cv2.error) as error:
        print(f"[ERROR] {error}")
        return 1

    monitor = EyeClosureMonitor(
        calibration_seconds=args.calibration_seconds,
        closed_ratio=args.closed_ratio,
        danger_seconds=args.danger_seconds,
    )
    cap = open_camera(args)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}.")
        return 1

    print("Keep both eyes naturally open during the first 3 seconds.")
    print("Press r to recalibrate. Press q to quit.")
    previous_frame_at = time.monotonic()
    smoothed_fps = 0.0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[ERROR] Failed to read camera frame.")
                break

            now = time.monotonic()
            frame_seconds = max(now - previous_frame_at, 1e-6)
            previous_frame_at = now
            current_fps = 1.0 / frame_seconds
            smoothed_fps = (
                current_fps
                if smoothed_fps == 0.0
                else 0.9 * smoothed_fps + 0.1 * current_fps
            )

            detection = face_detector.detect_largest(frame)
            mean_ear = None
            right_ear = None
            left_ear = None

            if detection is not None:
                try:
                    landmarks, crop_box = landmark_detector.detect(
                        frame,
                        detection.box,
                    )
                except (RuntimeError, cv2.error) as error:
                    print(f"[ERROR] Landmark inference failed: {error}")
                    break

                mean_ear, right_ear, left_ear = (
                    mean_eye_aspect_ratio(landmarks)
                )

                right_eye_points = eye_display_points(
                    landmarks,
                    RIGHT_EYE_INDICES,
                )
                left_eye_points = eye_display_points(
                    landmarks,
                    LEFT_EYE_INDICES,
                )
                lip_points = mouth_display_points(landmarks)

                # Exactly four visible points per eye and two for the mouth.
                draw_points(frame, right_eye_points, (0, 255, 0))
                draw_points(frame, left_eye_points, (0, 255, 0))
                draw_points(frame, lip_points, (255, 0, 0))

                x, y, width, height = detection.box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2,
                )

            eye_state = monitor.update(mean_ear, timestamp=now)
            color = state_color(eye_state.status)

            if eye_state.ear is None:
                draw_text(frame, "EAR: --", 0)
            else:
                draw_text(frame, f"EAR: {eye_state.ear:.3f}", 0)
                draw_text(
                    frame,
                    f"R: {right_ear:.3f}  L: {left_ear:.3f}",
                    1,
                )

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
                    f"BASE: {eye_state.baseline_ear:.3f}  "
                    f"REL: {eye_state.relative_ear:.2f}",
                    2,
                )
                draw_text(
                    frame,
                    f"CLOSED: {eye_state.closed_seconds:.2f}s",
                    3,
                    color,
                )

            draw_text(frame, f"STATE: {eye_state.status}", 4, color)
            draw_text(frame, f"FPS: {smoothed_fps:.1f}", 5)
            draw_text(frame, "q: quit  r: recalibrate", 6)

            cv2.imshow("ZZM EAR Eye Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                monitor.reset()
                print("EAR calibration reset.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
