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
    DrowsinessRiskController,
    EyeClosureMonitor,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    mean_eye_aspect_ratio,
)
from drowsiness.actions import (
    BuzzerPatternController,
    JetsonGPIOOutput,
    RiskEventPublisher,
)
from drowsiness.detectors import (
    PFLDLandmarkDetector,
    PFLDTensorRTDetector,
    YuNetFaceDetector,
)
from drowsiness.overlay import draw_status_overlay
from drowsiness.perclos_monitor import PerclosMonitor  # [PERCLOS] 추가
from drowsiness.yawn_monitor import YawnMonitor
from benchmark import PerformanceLogger


DEFAULT_YUNET_PATH = PROJECT_ROOT / "models/face_detector/yunet.onnx"
DEFAULT_PFLD_PATH = PROJECT_ROOT / "models/landmark/pfld_sim.onnx"
DEFAULT_PFLD_ENGINE_PATH = (
    PROJECT_ROOT / "models/engines/fp16/pfld_fp16.engine"
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
    "mar",
    "is_yawning",
    "yawn_seconds",
    "risk_level",
    "risk_reasons",
    "recent_yawn_count",
    "buzzer_mode",
    "hazard_light",
    "stop_request",
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
        "--opencv-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="OpenCV DNN device for YuNet and ONNX PFLD (default: cpu)",
    )
    parser.add_argument(
        "--landmark-backend",
        choices=("opencv", "tensorrt"),
        default="opencv",
        help="PFLD inference backend (default: opencv)",
    )
    parser.add_argument(
        "--pfld-engine", type=Path, default=DEFAULT_PFLD_ENGINE_PATH
    )
    parser.add_argument(
        "--calibration-seconds", type=positive_float, default=3.0
    )
    parser.add_argument(
        "--closed-ratio", type=open_unit_interval, default=0.70
    )
    parser.add_argument(
        "--reopen-ratio", type=closed_unit_interval, default=0.80
    )
    parser.add_argument(
        "--danger-seconds", type=positive_float, default=2.0
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
    parser.add_argument("--yawn-open-ratio", type=positive_float, default=0.18)
    parser.add_argument("--yawn-close-ratio", type=positive_float, default=0.14)
    parser.add_argument("--yawn-seconds", type=positive_float, default=0.3)
    parser.add_argument(
        "--buzzer-pin", type=positive_int, default=None,
        help="Physical buzzer GPIO pin; omitted means no GPIO access",
    )
    parser.add_argument(
        "--gpio-numbering", choices=("BOARD", "BCM"), default="BOARD"
    )
    parser.add_argument(
        "--buzzer-active-low", action="store_true",
        help="Use a buzzer module that turns on with a LOW signal",
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
            "--yawn-open-ratio must be greater than --yawn-close-ratio"
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
    buzzer_actuator = None
    if len(args.video_codec) != 4:
        print("[ERROR] --video-codec must contain exactly four characters.")
        return 1

    try:
        face_detector = YuNetFaceDetector(
            args.yunet,
            input_size=(args.width, args.height),
            device=args.opencv_device,
        )
        landmark_detector = (
            PFLDTensorRTDetector(args.pfld_engine)
            if args.landmark_backend == "tensorrt"
            else PFLDLandmarkDetector(
                args.pfld, device=args.opencv_device
            )
        )
    except (FileNotFoundError, RuntimeError, cv2.error) as error:
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
    risk_controller = DrowsinessRiskController()
    risk_publisher = RiskEventPublisher()
    cap = open_camera(args)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}.")
        return 1

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    video_fps = reported_fps if reported_fps > 1.0 else float(args.fps)

    logger = PerformanceLogger(
        backend=(
            f"opencv_{args.opencv_device}_yunet_pfld_tensorrt_fp16"
            if args.landmark_backend == "tensorrt"
            else f"opencv_{args.opencv_device}_yunet_pfld_fp32"
        ),
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

    if args.buzzer_pin is not None:
        try:
            buzzer_output = JetsonGPIOOutput(
                args.buzzer_pin,
                numbering=args.gpio_numbering,
                active_high=not args.buzzer_active_low,
            )
            buzzer_actuator = BuzzerPatternController(buzzer_output)
            risk_publisher.subscribe(buzzer_actuator.publish)
        except (ValueError, RuntimeError) as error:
            video_writer.release()
            cap.release()
            print(f"[ERROR] Cannot initialize buzzer: {error}")
            return 1

    print("Keep both eyes naturally open during the first 3 seconds.")
    print("Press r to recalibrate, a to acknowledge, q to quit.")
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

            # AI 모델 추론: YuNet + 얼굴이 검출된 경우 PFLD
            detection = face_detector.detect_largest(frame)

            landmarks = None
            if detection is not None:
                try:
                    landmarks, _crop_box = landmark_detector.detect(
                        frame,
                        detection.box,
                    )
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
            mar_value = 0.0

            if landmarks is not None:
                mean_ear, right_ear, left_ear = mean_eye_aspect_ratio(
                    landmarks
                )
                left_mouth = landmarks[MOUTH_LRTB_INDICES[0]]
                right_mouth = landmarks[MOUTH_LRTB_INDICES[1]]
                top_mouth = landmarks[MOUTH_LRTB_INDICES[2]]
                bottom_mouth = landmarks[MOUTH_LRTB_INDICES[3]]
                mouth_width = float(np.linalg.norm(left_mouth - right_mouth))
                mouth_height = float(np.linalg.norm(top_mouth - bottom_mouth))
                mar_value = (
                    mouth_height / mouth_width if mouth_width > 1e-6 else 0.0
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

            eye_state = monitor.update(mean_ear, timestamp=now)
            yawn_state = yawn_monitor.update(mar_value, timestamp=now)
            # [PERCLOS] 눈 감김 판정을 PERCLOS에 연결
            perclos_state = perclos_monitor.update(
                is_closed=eye_state.is_closed,
                valid_face=eye_state.valid_face,
                timestamp=now,
            )
            risk_decision = risk_controller.update(
                timestamp=now,
                eye_danger=eye_state.is_danger,
                perclos_caution=perclos_state.is_caution,
                perclos_warning=perclos_state.is_warning,
                is_yawning=yawn_state.is_yawning,
                valid_face=eye_state.valid_face,
            )
            risk_publisher.publish(risk_decision)
            draw_status_overlay(
                frame,
                eye_state,
                perclos_state,
                right_ear=right_ear,
                left_ear=left_ear,
                detection_score=detection_score,
                fps=logger.current_fps,
            )
            cv2.putText(
                frame,
                f"RISK: {risk_decision.level.value}",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255) if risk_decision.stop_request else (0, 200, 255),
                2,
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
                    "mar": yawn_state.mar,
                    "is_yawning": yawn_state.is_yawning,
                    "yawn_seconds": yawn_state.open_seconds,
                    "risk_level": risk_decision.level.value,
                    "risk_reasons": "|".join(risk_decision.reasons),
                    "recent_yawn_count": risk_decision.recent_yawn_count,
                    "buzzer_mode": risk_decision.buzzer_mode.value,
                    "hazard_light": risk_decision.hazard_light,
                    "stop_request": risk_decision.stop_request,
                },
            )

            if key == ord("q"):
                break
            if key == ord("r"):
                monitor.reset()
                perclos_monitor.reset()  # [PERCLOS] 재캘리브레이션 시 함께 리셋
                yawn_monitor.reset()
                risk_controller.reset()
                print("EAR calibration reset.")
            if key == ord("a"):
                risk_controller.acknowledge()
                print("Drowsiness warning acknowledged; verifying recovery.")
    except KeyboardInterrupt:
        print("\nStopped by keyboard interrupt.")
    finally:
        if buzzer_actuator is not None:
            buzzer_actuator.close()
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
