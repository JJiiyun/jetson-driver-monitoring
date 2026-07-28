#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
    mean_eye_aspect_ratio,
)
from drowsiness.detectors import PFLDLandmarkDetector, YuNetFaceDetector
from drowsiness.overlay import draw_status_overlay
from drowsiness.perclos_monitor import PerclosMonitor  # [PERCLOS] 추가
from drowsiness.tensorrt_pfld import TensorRTPFLDLandmarkDetector
from drowsiness.yawn_monitor import YawnMonitor
from benchmark import PerformanceLogger


DEFAULT_YUNET_PATH = PROJECT_ROOT / "models/face_detector/yunet.onnx"
DEFAULT_PFLD_PATH = PROJECT_ROOT / "models/landmark/pfld_sim.onnx"
DEFAULT_PFLD_FP16_ENGINE_PATH = (
    PROJECT_ROOT / "models/engine/pfld_sim_fp16.engine"
)
DEFAULT_PFLD_FP32_ENGINE_PATH = (
    PROJECT_ROOT / "models/engine/pfld_sim_fp32.engine"
)
RESULTS_DIR = PROJECT_ROOT / "benchmark/results"
VIDEO_DIR = PROJECT_ROOT / "outputs/videos"
# 입 상하좌우 4점 (68점 규약): 48=좌끝, 54=우끝, 62=안쪽 윗입술, 66=안쪽 아랫입술
MOUTH_LRTB_INDICES = (48, 54, 62, 66)
EYE_FRAME_FIELDS = [
    "detection_score",
    "ear",
    "right_ear",
    "left_ear",
    "baseline_ear",
    "relative_ear",
    "closed_threshold",
    "reopen_threshold",
    "is_eye_closed",
    "closed_seconds",
    "eye_state",
    "perclos",          # [PERCLOS] 추가
    "perclos_caution",  # [PERCLOS] 추가
    "perclos_warning",  # [PERCLOS] 추가
    "pfld_inference_ms",
    "mar",
    "is_yawning",
    "yawn_seconds",
    "yawn_state",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            "must be a finite number greater than zero"
        )
    return parsed


def open_unit_interval(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError(
            "must be a finite number between 0 and 1 (exclusive)"
        )
    return parsed


def closed_unit_interval(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(
            "must be a finite number between 0 and 1 (inclusive)"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time EAR calibration and eye-closure monitor."
    )
    parser.add_argument("--camera", type=nonnegative_int, default=0)
    parser.add_argument("--width", type=positive_int, default=640)
    parser.add_argument("--height", type=positive_int, default=480)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--yunet", type=Path, default=DEFAULT_YUNET_PATH)
    parser.add_argument("--pfld", type=Path, default=DEFAULT_PFLD_PATH)
    parser.add_argument(
        "--landmark-backend",
        choices=("opencv-fp32", "tensorrt-fp32", "tensorrt-fp16"),
        default="opencv-fp32",
        help="PFLD execution backend (default: opencv-fp32).",
    )
    parser.add_argument(
        "--pfld-engine",
        type=Path,
        default=None,
        help="Override the TensorRT engine selected for the backend.",
    )
    parser.add_argument(
        "--trt-warmup-iterations",
        type=nonnegative_int,
        default=5,
        help="Untimed TensorRT calls on the first detected face.",
    )
    parser.add_argument(
        "--calibration-seconds", type=positive_float, default=3.0
    )
    parser.add_argument(
        "--closed-ratio", type=open_unit_interval, default=0.72
    )
    parser.add_argument(
        "--reopen-ratio", type=closed_unit_interval, default=0.85
    )
    parser.add_argument(
        "--danger-seconds", type=positive_float, default=1.7
    )
    parser.add_argument(
        "--yawn-open-ratio",
        type=positive_float,
        default=0.18,
    )
    parser.add_argument(
        "--yawn-close-ratio",
        type=positive_float,
        default=0.14,
    )
    parser.add_argument(
        "--yawn-seconds",
        type=positive_float,
        default=0.3,
    )
    # [PERCLOS] 추가 옵션
    parser.add_argument(
        "--perclos-window", type=positive_float, default=30.0
    )
    parser.add_argument(
        "--perclos-caution", type=closed_unit_interval, default=0.15
    )
    parser.add_argument(
        "--perclos-warning", type=closed_unit_interval, default=0.30
    )
    parser.add_argument("--warmup-frames", type=nonnegative_int, default=30)
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    parser.add_argument("--video-codec", default="MJPG")
    args = parser.parse_args()
    if args.perclos_caution > args.perclos_warning:
        parser.error(
            "--perclos-caution must be less than or equal to "
            "--perclos-warning"
        )
    if args.closed_ratio >= args.reopen_ratio:
        parser.error(
            "--reopen-ratio must be greater than --closed-ratio"
        )
    if args.yawn_close_ratio >= args.yawn_open_ratio:
        parser.error(
            "--yawn-close-ratio must be less than --yawn-open-ratio"
        )
    return args


def draw_points(
    frame: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 1,
) -> None:
    for point_x, point_y in np.rint(points).astype(int):
        cv2.circle(frame, (point_x, point_y), radius, color, -1)


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
        if args.landmark_backend.startswith("tensorrt-"):
            engine_path = args.pfld_engine
            if engine_path is None:
                engine_path = (
                    DEFAULT_PFLD_FP32_ENGINE_PATH
                    if args.landmark_backend == "tensorrt-fp32"
                    else DEFAULT_PFLD_FP16_ENGINE_PATH
                )
            landmark_detector = TensorRTPFLDLandmarkDetector(engine_path)
        else:
            landmark_detector = PFLDLandmarkDetector(args.pfld)
    except (FileNotFoundError, ValueError, RuntimeError, cv2.error) as error:
        print(f"[ERROR] {error}")
        return 1

    monitor = EyeClosureMonitor(
        calibration_seconds=args.calibration_seconds,
        closed_ratio=args.closed_ratio,
        reopen_ratio=args.reopen_ratio,
        danger_seconds=args.danger_seconds,
    )
    # [PERCLOS] 모니터 생성
    perclos_monitor = PerclosMonitor(
        window_seconds=args.perclos_window,
        caution_perclos=args.perclos_caution,
        warning_perclos=args.perclos_warning,
    )
    yawn_monitor = YawnMonitor(
        open_ratio=args.yawn_open_ratio,
        close_ratio=args.yawn_close_ratio,
        yawn_seconds=args.yawn_seconds,
    )
    cap = open_camera(args)
    if not cap.isOpened():
        close_detector = getattr(landmark_detector, "close", None)
        if close_detector is not None:
            close_detector()
        print(f"[ERROR] Cannot open camera index {args.camera}.")
        return 1

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    video_fps = reported_fps if reported_fps > 1.0 else float(args.fps)

    logger = PerformanceLogger(
        backend=f"{args.landmark_backend.replace('-', '_')}_camera_fsm",
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
        close_detector = getattr(landmark_detector, "close", None)
        if close_detector is not None:
            close_detector()
        print(f"[ERROR] Cannot create video file: {video_path}")
        return 1

    print("Keep both eyes naturally open during the first 3 seconds.")
    print("Press r to recalibrate. Press q to quit.")
    print(f"Landmark backend: {args.landmark_backend}")
    print(f"Video: {video_path}")

    tensorrt_warmed_up = False
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

            # AI 모델 추론: YuNet + 얼굴이 검출된 경우 PFLD
            detection = face_detector.detect_largest(frame)

            landmarks = None
            pfld_inference_ms = None
            if detection is not None:
                try:
                    if (
                        args.landmark_backend.startswith("tensorrt-")
                        and not tensorrt_warmed_up
                    ):
                        for _ in range(args.trt_warmup_iterations):
                            landmark_detector.detect(frame, detection.box)
                        tensorrt_warmed_up = True
                    pfld_started_at = time.perf_counter()
                    landmarks, _crop_box = landmark_detector.detect(
                        frame,
                        detection.box,
                    )
                    pfld_inference_ms = (
                        time.perf_counter() - pfld_started_at
                    ) * 1000.0
                except (RuntimeError, cv2.error) as error:
                    print(f"[ERROR] Landmark inference failed: {error}")
                    break

            # 얼굴이 없어도 YuNet 추론 시간은 기록한다.
            inference_ms = (
                time.perf_counter() - inference_started_at
            ) * 1000.0

            # 추론 시간에 포함하지 않는 후처리
            detection_score = (
                None if detection is None else detection.score
            )
            mean_ear = None
            right_ear = None
            left_ear = None
            mar_value = None
            yawn_state = None

            if landmarks is not None:
                mean_ear, right_ear, left_ear = mean_eye_aspect_ratio(
                    landmarks
                )
                mouth_left = landmarks[MOUTH_LRTB_INDICES[0]]
                mouth_right = landmarks[MOUTH_LRTB_INDICES[1]]
                mouth_top = landmarks[MOUTH_LRTB_INDICES[2]]
                mouth_bottom = landmarks[MOUTH_LRTB_INDICES[3]]
                mouth_width = float(
                    np.linalg.norm(mouth_left - mouth_right)
                )
                mouth_height = float(
                    np.linalg.norm(mouth_top - mouth_bottom)
                )
                mar_value = (
                    mouth_height / mouth_width
                    if mouth_width > 1e-6
                    else 0.0
                )
                yawn_state = yawn_monitor.update(
                    mar_value,
                    timestamp=now,
                )

                draw_points(
                    frame, landmarks[list(RIGHT_EYE_INDICES)], (0, 255, 0)
                )
                draw_points(
                    frame, landmarks[list(LEFT_EYE_INDICES)], (0, 255, 0)
                )
                draw_points(
                    frame, landmarks[list(MOUTH_LRTB_INDICES)], (0, 255, 0)
                )

                x, y, width, height = detection.box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2,
                )
                if yawn_state.is_yawning:
                    cv2.putText(
                        frame,
                        "YAWNING",
                        (x, max(24, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                yawn_monitor.reset()

            eye_state = monitor.update(mean_ear, timestamp=now)
            # [PERCLOS] 눈 감김 판정을 PERCLOS에 연결
            perclos_state = perclos_monitor.update(
                is_closed=eye_state.is_closed,
                valid_face=eye_state.valid_face,
                timestamp=now,
            )
            draw_status_overlay(
                frame,
                eye_state,
                perclos_state,
                right_ear=right_ear,
                left_ear=left_ear,
                detection_score=detection_score,
                fps=logger.current_fps,
                face_box=(
                    None if detection is None else detection.box
                ),
            )

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
                    "detection_score": detection_score,
                    "ear": eye_state.ear,
                    "right_ear": right_ear,
                    "left_ear": left_ear,
                    "baseline_ear": eye_state.baseline_ear,
                    "relative_ear": eye_state.relative_ear,
                    "closed_threshold": eye_state.closed_threshold,
                    "reopen_threshold": eye_state.reopen_threshold,
                    "is_eye_closed": eye_state.is_closed,
                    "closed_seconds": eye_state.closed_seconds,
                    "eye_state": eye_state.status,
                    "perclos": perclos_state.perclos,           # [PERCLOS]
                    "perclos_caution": perclos_state.is_caution,  # [PERCLOS]
                    "perclos_warning": perclos_state.is_warning,  # [PERCLOS]
                    "pfld_inference_ms": pfld_inference_ms,
                    "mar": mar_value,
                    "is_yawning": (
                        False
                        if yawn_state is None
                        else yawn_state.is_yawning
                    ),
                    "yawn_seconds": (
                        0.0
                        if yawn_state is None
                        else yawn_state.open_seconds
                    ),
                    "yawn_state": (
                        "NO FACE"
                        if yawn_state is None
                        else yawn_state.status
                    ),
                },
            )

            if key == ord("q"):
                break
            if key == ord("r"):
                monitor.reset()
                perclos_monitor.reset()  # [PERCLOS] 재캘리브레이션 시 함께 리셋
                yawn_monitor.reset()
                print("EAR calibration reset.")
    except KeyboardInterrupt:
        print("\nStopped by keyboard interrupt.")
    finally:
        close_detector = getattr(landmark_detector, "close", None)
        if close_detector is not None:
            close_detector()
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
