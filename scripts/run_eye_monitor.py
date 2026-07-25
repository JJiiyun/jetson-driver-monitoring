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
from benchmark import PerformanceLogger


DEFAULT_YUNET_PATH = PROJECT_ROOT / "models/face_detector/yunet.onnx"
DEFAULT_PFLD_PATH = PROJECT_ROOT / "models/landmark/pfld_sim.onnx"
RESULTS_DIR = PROJECT_ROOT / "benchmark/results"
VIDEO_DIR = PROJECT_ROOT / "outputs/videos"
EYE_FRAME_FIELDS = [
    "ear",
    "right_ear",
    "left_ear",
    "baseline_ear",
    "relative_ear",
    "closed_threshold",
    "is_eye_closed",
    "closed_seconds",
    "eye_state",
]


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
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--video-codec", default="MJPG")
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


def format_optional(value: float | None, precision: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{precision}f}"


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
    if len(args.video_codec) != 4:
        print("[ERROR] --video-codec must contain exactly four characters.")
        return 1

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

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    video_fps = reported_fps if reported_fps > 1.0 else float(args.fps)

    logger = PerformanceLogger(
        backend="opencv_yunet_pfld_fp32",
        output_dir=RESULTS_DIR,
        warmup_frames=args.warmup_frames,
        input_source=f"camera:{args.camera}",
        width=actual_width,
        height=actual_height,
        target_fps=args.fps,
        extra_frame_fields=EYE_FRAME_FIELDS,
    )

    args.video_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.video_dir / f"{logger.run_id}.avi"
    video_writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*args.video_codec),
        video_fps,
        (actual_width, actual_height),
    )
    if not video_writer.isOpened():
        cap.release()
        print(f"[ERROR] Cannot create video file: {video_path}")
        return 1

    print("Keep both eyes naturally open during the first 3 seconds.")
    print("Press r to recalibrate. Press q to quit.")
    print(f"Video: {video_path}")

    try:
        while True:
            frame_started_at = time.perf_counter()
            capture_started_at = time.perf_counter()
            success, frame = cap.read()
            capture_ms = (
                time.perf_counter() - capture_started_at
            ) * 1000.0
            if not success:
                print("[ERROR] Failed to read camera frame.")
                break

            now = time.monotonic()
            inference_started_at = time.perf_counter()
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

            inference_ms = (
                time.perf_counter() - inference_started_at
            ) * 1000.0
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
                    "BASE: "
                    f"{format_optional(eye_state.baseline_ear)}  "
                    "REL: "
                    f"{format_optional(eye_state.relative_ear, 2)}",
                    2,
                )
                draw_text(
                    frame,
                    f"CLOSED: {eye_state.closed_seconds:.2f}s",
                    3,
                    color,
                )

            draw_text(frame, f"STATE: {eye_state.status}", 4, color)
            draw_text(frame, f"FPS: {logger.current_fps:.1f}", 5)
            draw_text(frame, "q: quit  r: recalibrate", 6)

            video_writer.write(frame)
            cv2.imshow("ZZM EAR Eye Monitor", frame)
            key = cv2.waitKey(1) & 0xFF

            logger.record_frame(
                frame_started_at=frame_started_at,
                frame_finished_at=time.perf_counter(),
                capture_ms=capture_ms,
                inference_ms=inference_ms,
                face_count=0 if detection is None else 1,
                extra_metrics={
                    "ear": eye_state.ear,
                    "right_ear": right_ear,
                    "left_ear": left_ear,
                    "baseline_ear": eye_state.baseline_ear,
                    "relative_ear": eye_state.relative_ear,
                    "closed_threshold": eye_state.closed_threshold,
                    "is_eye_closed": eye_state.is_closed,
                    "closed_seconds": eye_state.closed_seconds,
                    "eye_state": eye_state.status,
                },
            )

            if key == ord("q"):
                break
            if key == ord("r"):
                monitor.reset()
                print("EAR calibration reset.")
    except KeyboardInterrupt:
        print("\nStopped by keyboard interrupt.")
    finally:
        video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

    summary = logger.write_csv()
    logger.print_summary(summary)
    print(f"Video: {video_path}")
    print(f"Frame CSV: {logger.frame_csv_path}")
    print(f"Summary CSV: {logger.summary_csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
