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

from benchmark import PerformanceLogger
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
    TensorRTYuNetFaceDetector,
    YuNetFaceDetector,
)
from drowsiness.overlay import draw_status_overlay
from drowsiness.perclos_monitor import PerclosMonitor
from drowsiness.qt_dashboard import create_qt_application, create_risk_dashboard
# === [하품 추가 1] YawnMonitor import ===
# 하품 검출 모듈. drowsiness/yawn_monitor.py 에 위치.
from drowsiness.yawn_monitor import YawnMonitor


DEFAULT_YUNET_PATH = PROJECT_ROOT / "models/face_detector/yunet.onnx"
DEFAULT_YUNET_FP16_ENGINE_PATH = PROJECT_ROOT / "models/engines/fp16/yunet_fp16.engine"
DEFAULT_PFLD_PATH = PROJECT_ROOT / "models/landmark/pfld_sim.onnx"
DEFAULT_PFLD_FP16_ENGINE_PATH = (
    PROJECT_ROOT / "models/engines/fp16/pfld_fp16.engine"
)
DEFAULT_PFLD_FP32_ENGINE_PATH = (
    PROJECT_ROOT / "models/engines/fp32/pfld_fp32.engine"
)
RESULTS_DIR = PROJECT_ROOT / "benchmark/results"
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "outputs/video_inference"

# 입 랜드마크 4점 (68점 dlib 표준): 좌=48, 우=54, 상=62, 하=66
# MAR = 세로거리(62-66) / 가로거리(48-54)
MOUTH_LRTB_INDICES = (48, 54, 62, 66)

FRAME_FIELDS = [
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
    "perclos",
    "perclos_caution",
    "perclos_warning",
    # === [하품 추가 2] CSV 컬럼 3개 ===
    # 이 필드들이 있어야 evaluate.py가 하품을 채점할 수 있다.
    "mar",            # 입 종횡비 (Mouth Aspect Ratio)
    "is_yawning",     # 하품 판정 여부 (True/False)
    "yawn_seconds",   # 연속 입벌림 시간
    "risk_level",
    "risk_reasons",
    "recent_yawn_count",
    "buzzer_mode",
    "hazard_light",
    "stop_request",
]


LANDMARK_BACKEND_ALIASES = {
    "opencv": "opencv-fp32",
    "tensorrt": "tensorrt-fp16",
}


def landmark_backend(value: str) -> str:
    normalized = LANDMARK_BACKEND_ALIASES.get(value, value)
    valid = ("opencv-fp32", "tensorrt-fp32", "tensorrt-fp16")
    if normalized not in valid:
        raise argparse.ArgumentTypeError(
            "must be opencv-fp32, tensorrt-fp32, or tensorrt-fp16"
        )
    return normalized


def selected_engine_path(args: argparse.Namespace) -> Path | None:
    if not args.landmark_backend.startswith("tensorrt-"):
        return None
    if args.pfld_engine is not None:
        return args.pfld_engine
    if args.landmark_backend == "tensorrt-fp32":
        return DEFAULT_PFLD_FP32_ENGINE_PATH
    return DEFAULT_PFLD_FP16_ENGINE_PATH


def parse_args(use_fsm: bool = False) -> argparse.Namespace:
    mode = "FSM with hysteresis" if use_fsm else "single EAR threshold"
    parser = argparse.ArgumentParser(
        description=(
            f"Run FP32 video inference using {mode} and save a separate "
            "annotated video."
        )
    )
    parser.add_argument("input", type=Path, help="오버레이가 없는 원본 영상")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="결과 MP4 경로(기본: outputs/video_inference)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="처리 중 결과 화면 표시",
    )
    parser.add_argument("--yunet", type=Path, default=DEFAULT_YUNET_PATH)
    parser.add_argument(
        "--face-backend", choices=("opencv-fp32", "tensorrt-fp16"),
        default="opencv-fp32",
    )
    parser.add_argument(
        "--yunet-engine", type=Path, default=DEFAULT_YUNET_FP16_ENGINE_PATH
    )
    parser.add_argument("--pfld", type=Path, default=DEFAULT_PFLD_PATH)
    parser.add_argument(
        "--opencv-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="OpenCV DNN device for YuNet and ONNX PFLD (default: cpu)",
    )
    parser.add_argument(
        "--landmark-backend",
        type=landmark_backend,
        default="opencv-fp32",
        metavar="{opencv-fp32,tensorrt-fp32,tensorrt-fp16}",
        help="PFLD inference backend and precision (default: opencv-fp32)",
    )
    parser.add_argument(
        "--pfld-engine",
        type=Path,
        default=None,
        help="Override the FP32/FP16 TensorRT engine selected by the backend",
    )
    parser.add_argument("--calibration-seconds", type=float, default=3.0)
    parser.add_argument("--closed-ratio", type=float, default=0.70)
    if use_fsm:
        parser.add_argument("--reopen-ratio", type=float, default=0.80)
    else:
        parser.set_defaults(reopen_ratio=None)
    parser.add_argument("--danger-seconds", type=float, default=2.0)
    parser.add_argument("--perclos-window", type=float, default=30.0)
    parser.add_argument("--perclos-caution", type=float, default=0.15)
    parser.add_argument("--perclos-warning", type=float, default=0.30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    # === [하품 추가 3] 하품 임계값 옵션 ===
    # 명령줄에서 조정 가능하게. 실제 하품 영상 보고 튜닝하면 된다.
    parser.add_argument(
        "--yawn-open-ratio", type=float, default=0.18,
        help="MAR이 이 값 이상이면 입 벌어짐 (기본 0.18)",
    )
    parser.add_argument(
        "--yawn-close-ratio", type=float, default=0.14,
        help="MAR이 이 값 아래로 내려가야 닫힘 (히스테리시스, 기본 0.14)",
    )
    parser.add_argument(
        "--yawn-seconds", type=float, default=0.3,
        help="이 시간 이상 벌어져 있으면 하품 (기본 0.3초)",
    )
    parser.add_argument(
        "--buzzer-pin",
        type=int,
        default=None,
        help="물리 부저 GPIO 핀. 생략하면 GPIO를 사용하지 않음",
    )
    parser.add_argument(
        "--gpio-numbering",
        choices=("BOARD", "BCM"),
        default="BOARD",
    )
    parser.add_argument(
        "--buzzer-active-low",
        action="store_true",
        help="LOW 신호에서 켜지는 부저 모듈 사용",
    )
    parser.add_argument(
        "--qt-dashboard",
        action="store_true",
        help="추론 중 Qt 위험 대시보드 표시",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input video does not exist: {args.input}")
    if args.calibration_seconds <= 0.0:
        parser.error("--calibration-seconds must be greater than zero")
    if use_fsm:
        if not 0.0 < args.closed_ratio < args.reopen_ratio <= 1.0:
            parser.error(
                "ratios must satisfy 0 < --closed-ratio < "
                "--reopen-ratio <= 1"
            )
    elif not 0.0 < args.closed_ratio < 1.0:
        parser.error("--closed-ratio must be between 0 and 1")
    if args.danger_seconds <= 0.0:
        parser.error("--danger-seconds must be greater than zero")
    if args.perclos_window <= 0.0:
        parser.error("--perclos-window must be greater than zero")
    if not 0.0 <= args.perclos_caution <= args.perclos_warning <= 1.0:
        parser.error(
            "PERCLOS thresholds must satisfy 0 <= caution <= warning <= 1"
        )
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be zero or greater")
    # === [하품 추가 3-1] 하품 임계값 검증 ===
    if not 0.0 < args.yawn_close_ratio < args.yawn_open_ratio:
        parser.error(
            "--yawn-close-ratio must be > 0 and < --yawn-open-ratio"
        )
    if args.yawn_seconds <= 0.0:
        parser.error("--yawn-seconds must be greater than zero")
    if (
        args.output is not None
        and args.output.resolve() == args.input.resolve()
    ):
        parser.error("--output must be different from the input video")
    return args


def draw_points(
    frame: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    for point_x, point_y in np.rint(points).astype(int):
        cv2.circle(frame, (point_x, point_y), 1, color, -1)


def default_output_path(
    input_path: Path,
    run_id: str,
    use_fsm: bool,
) -> Path:
    mode = "FSM" if use_fsm else "basic"
    return (
        VIDEO_OUTPUT_DIR
        / f"{input_path.stem}_{mode}_{run_id}_annotated.mp4"
    )


def main(use_fsm: bool = False) -> int:
    args = parse_args(use_fsm)
    buzzer_actuator = None
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        print(f"[ERROR] Cannot open input video: {args.input}")
        return 1

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        print("[ERROR] Input video has an invalid frame size.")
        return 1
    if not np.isfinite(fps) or fps <= 1.0:
        capture.release()
        print("[ERROR] Input video has invalid or missing FPS metadata.")
        return 1

    try:
        face_detector = (
            TensorRTYuNetFaceDetector(args.yunet_engine)
            if args.face_backend == "tensorrt-fp16"
            else YuNetFaceDetector(
                args.yunet, input_size=(width, height), device=args.opencv_device
            )
        )
        engine_path = selected_engine_path(args)
        landmark_detector = (
            PFLDTensorRTDetector(engine_path)
            if engine_path is not None
            else PFLDLandmarkDetector(args.pfld, device=args.opencv_device)
        )
        eye_monitor = EyeClosureMonitor(
            calibration_seconds=args.calibration_seconds,
            closed_ratio=args.closed_ratio,
            reopen_ratio=(
                args.reopen_ratio
                if use_fsm
                else args.closed_ratio
            ),
            use_hysteresis=use_fsm,
            danger_seconds=args.danger_seconds,
        )
        perclos_monitor = PerclosMonitor(
            window_seconds=args.perclos_window,
            caution_perclos=args.perclos_caution,
            warning_perclos=args.perclos_warning,
        )
        # === [하품 추가 4] YawnMonitor 초기화 ===
        # 눈 감김 EyeClosureMonitor와 같은 구조(히스테리시스 + 지속시간).
        yawn_monitor = YawnMonitor(
            open_ratio=args.yawn_open_ratio,
            close_ratio=args.yawn_close_ratio,
            yawn_seconds=args.yawn_seconds,
        )
        risk_controller = DrowsinessRiskController()
        risk_publisher = RiskEventPublisher()
        qt_app = None
        dashboard = None
        if args.qt_dashboard:
            qt_app = create_qt_application(sys.argv)
            dashboard = create_risk_dashboard(
                risk_controller, risk_publisher
            )
            dashboard.show()
    except (FileNotFoundError, ValueError, RuntimeError, cv2.error) as error:
        capture.release()
        print(f"[ERROR] {error}")
        return 1

    landmark_backend_name = (
        "fp32"
        if args.landmark_backend == "opencv-fp32"
        else args.landmark_backend.replace("-", "_")
    )
    logger = PerformanceLogger(
        backend=(
            f"{args.face_backend.replace('-', '_')}_yunet_pfld_{landmark_backend_name}"
            f"_video_{'fsm' if use_fsm else 'basic'}"
        ),
        output_dir=RESULTS_DIR,
        warmup_frames=args.warmup_frames,
        input_source=f"video:{args.input.name}",
        width=width,
        height=height,
        target_fps=fps,
        extra_frame_fields=FRAME_FIELDS,
    )
    output_path = (
        args.output
        if args.output is not None
        else default_output_path(args.input, logger.run_id, use_fsm)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        print(f"[ERROR] Cannot create output video: {output_path}")
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
            writer.release()
            capture.release()
            print(f"[ERROR] Cannot initialize buzzer: {error}")
            return 1

    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    print(f"Landmark backend: {args.landmark_backend}")
    print(f"Face backend: {args.face_backend}")
    if engine_path is not None:
        print(f"TensorRT engine: {engine_path}")
    print(
        "Mode: "
        + (
            "FSM + hysteresis"
            if use_fsm
            else "basic single-threshold"
        )
    )
    print(f"Frames: {total_frames if total_frames > 0 else 'unknown'}")
    print("The first 3 seconds with a valid face are used for calibration.")

    frame_index = 0
    try:
        while True:
            frame_started_at = time.perf_counter()
            capture_started_at = time.perf_counter()
            success, frame = capture.read()
            capture_ms = (
                time.perf_counter() - capture_started_at
            ) * 1000.0
            if not success:
                break

            timestamp = frame_index / fps
            inference_started_at = time.perf_counter()
            detection = face_detector.detect_largest(frame)
            landmarks = None
            if detection is not None:
                try:
                    landmarks, _crop_box = landmark_detector.detect(
                        frame,
                        detection.box,
                    )
                except (RuntimeError, cv2.error) as error:
                    print(
                        f"[ERROR] Landmark inference failed at frame "
                        f"{frame_index}: {error}"
                    )
                    return 1
            inference_ms = (
                time.perf_counter() - inference_started_at
            ) * 1000.0

            detection_score = (
                None if detection is None else detection.score
            )
            mean_ear = None
            right_ear = None
            left_ear = None
            # === [하품 추가 5] MAR / 하품 상태 변수 초기화 ===
            mar_value = None
            yawn_state = None
            if landmarks is not None:
                mean_ear, right_ear, left_ear = mean_eye_aspect_ratio(
                    landmarks
                )
                # === [하품 추가 6] MAR 계산 (입 4점) ===
                # 좌48-우54 = 가로, 상62-하66 = 세로
                # 입이 벌어질수록 세로가 커져 MAR이 커진다.
                left_pt = landmarks[MOUTH_LRTB_INDICES[0]]    # 48
                right_pt = landmarks[MOUTH_LRTB_INDICES[1]]   # 54
                top_pt = landmarks[MOUTH_LRTB_INDICES[2]]     # 62
                bottom_pt = landmarks[MOUTH_LRTB_INDICES[3]]  # 66
                h_dist = float(np.linalg.norm(left_pt - right_pt))
                v_dist = float(np.linalg.norm(top_pt - bottom_pt))
                mar_value = v_dist / h_dist if h_dist > 1e-6 else 0.0
                # 하품 상태 갱신 (timestamp는 frame_index/fps = 정확한 시간축)
                yawn_state = yawn_monitor.update(
                    mar_value, timestamp=timestamp
                )

                draw_points(
                    frame,
                    landmarks[list(RIGHT_EYE_INDICES)],
                    (0, 255, 0),
                )
                draw_points(
                    frame,
                    landmarks[list(LEFT_EYE_INDICES)],
                    (0, 255, 0),
                )
                draw_points(
                    frame,
                    landmarks[list(MOUTH_LRTB_INDICES)],
                    (255, 0, 0),
                )
                x, y, box_width, box_height = detection.box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + box_width, y + box_height),
                    (0, 255, 0),
                    2,
                )
                # === [하품 추가 7] 화면에 하품 표시 (선택) ===
                if yawn_state is not None and yawn_state.is_yawning:
                    cv2.putText(
                        frame, "YAWNING", (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2,
                    )

            eye_state = eye_monitor.update(mean_ear, timestamp=timestamp)
            perclos_state = perclos_monitor.update(
                is_closed=eye_state.is_closed,
                valid_face=eye_state.valid_face,
                timestamp=timestamp,
            )
            is_yawning = (
                yawn_state.is_yawning
                if yawn_state is not None
                else False
            )
            risk_decision = risk_controller.update(
                timestamp=timestamp,
                eye_danger=eye_state.is_danger,
                perclos_caution=perclos_state.is_caution,
                perclos_warning=perclos_state.is_warning,
                is_yawning=is_yawning,
                valid_face=eye_state.valid_face,
            )
            risk_publisher.publish(risk_decision)
            if qt_app is not None:
                qt_app.processEvents()
                if dashboard is not None and not dashboard.isVisible():
                    break
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
            cv2.putText(
                frame,
                f"RISK: {risk_decision.level.value}",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255) if risk_decision.stop_request else (0, 200, 255),
                2,
            )
            writer.write(frame)

            if args.show:
                cv2.imshow("ZZM Video Inference", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

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
                    "perclos": perclos_state.perclos,
                    "perclos_caution": perclos_state.is_caution,
                    "perclos_warning": perclos_state.is_warning,
                    # === [하품 추가 8] CSV에 하품 데이터 기록 ===
                    # landmarks 없으면 mar=None, 하품=False로 기록
                    "mar": mar_value,
                    "is_yawning": (
                        yawn_state.is_yawning
                        if yawn_state is not None
                        else False
                    ),
                    "yawn_seconds": (
                        yawn_state.open_seconds
                        if yawn_state is not None
                        else 0.0
                    ),
                    "risk_level": risk_decision.level.value,
                    "risk_reasons": "|".join(risk_decision.reasons),
                    "recent_yawn_count": risk_decision.recent_yawn_count,
                    "buzzer_mode": risk_decision.buzzer_mode.value,
                    "hazard_light": risk_decision.hazard_light,
                    "stop_request": risk_decision.stop_request,
                },
            )
            frame_index += 1
            if frame_index % 100 == 0:
                if total_frames > 0:
                    progress = min(100.0, frame_index / total_frames * 100.0)
                    print(
                        f"Processed {frame_index}/{total_frames} "
                        f"frames ({progress:.1f}%)"
                    )
                else:
                    print(f"Processed {frame_index} frames")
    except KeyboardInterrupt:
        print("\nStopped by keyboard interrupt.")
    finally:
        if dashboard is not None:
            dashboard.close()
        if buzzer_actuator is not None:
            buzzer_actuator.close()
        writer.release()
        capture.release()
        cv2.destroyAllWindows()

    summary = logger.write_csv()
    logger.print_summary(summary)
    print(f"Annotated video: {output_path}")
    print(f"Frame CSV: {logger.frame_csv_path}")
    print(f"Summary CSV: {logger.summary_csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
