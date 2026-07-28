#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    EyeClosureMonitor,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    mean_eye_aspect_ratio,
)
from drowsiness.detectors import PFLDLandmarkDetector, YuNetFaceDetector
from drowsiness.tensorrt_pfld import TensorRTPFLDLandmarkDetector
from drowsiness.overlay import draw_status_overlay
from drowsiness.perclos_monitor import PerclosMonitor


DEFAULT_YUNET_PATH = PROJECT_ROOT / "models/face_detector/yunet.onnx"
DEFAULT_PFLD_PATH = PROJECT_ROOT / "models/landmark/pfld_sim.onnx"
DEFAULT_PFLD_FP16_ENGINE_PATH = (
    PROJECT_ROOT / "models/engine/pfld_sim_fp16.engine"
)
DEFAULT_PFLD_FP32_ENGINE_PATH = (
    PROJECT_ROOT / "models/engine/pfld_sim_fp32.engine"
)
RESULTS_DIR = PROJECT_ROOT / "benchmark/results"
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "outputs/video_inference"
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
    "pfld_inference_ms",
    "face_box",
    "crop_box",
    "landmarks_json",
]


def parse_args(use_fsm: bool = False) -> argparse.Namespace:
    mode = "FSM with hysteresis" if use_fsm else "single EAR threshold"
    parser = argparse.ArgumentParser(
        description=(
            f"Run video inference using {mode} and save a separate "
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
        help=(
            "Override the default TensorRT PFLD engine selected for the "
            "requested backend."
        ),
    )
    parser.add_argument(
        "--trt-warmup-iterations",
        type=int,
        default=5,
        help="Untimed TensorRT calls on the first detected face (default: 5).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many decoded video frames.",
    )
    parser.add_argument("--calibration-seconds", type=float, default=3.0)
    parser.add_argument("--closed-ratio", type=float, default=0.72)
    if use_fsm:
        parser.add_argument("--reopen-ratio", type=float, default=0.85)
    else:
        parser.set_defaults(reopen_ratio=None)
    parser.add_argument("--danger-seconds", type=float, default=1.7)
    parser.add_argument("--perclos-window", type=float, default=30.0)
    parser.add_argument("--perclos-caution", type=float, default=0.15)
    parser.add_argument("--perclos-warning", type=float, default=0.30)
    parser.add_argument("--warmup-frames", type=int, default=30)
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
    if args.trt_warmup_iterations < 0:
        parser.error("--trt-warmup-iterations must be zero or greater")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")
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
        face_detector = YuNetFaceDetector(
            args.yunet,
            input_size=(width, height),
        )
        if args.landmark_backend.startswith("tensorrt-"):
            engine_path = args.pfld_engine
            if engine_path is None:
                engine_path = (
                    DEFAULT_PFLD_FP32_ENGINE_PATH
                    if args.landmark_backend == "tensorrt-fp32"
                    else DEFAULT_PFLD_FP16_ENGINE_PATH
                )
            landmark_detector = TensorRTPFLDLandmarkDetector(
                engine_path
            )
        else:
            landmark_detector = PFLDLandmarkDetector(args.pfld)
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
    except (FileNotFoundError, ValueError, RuntimeError, cv2.error) as error:
        capture.release()
        print(f"[ERROR] {error}")
        return 1

    logger = PerformanceLogger(
        backend=(
            f"{args.landmark_backend.replace('-', '_')}_video_"
            f"{'fsm' if use_fsm else 'basic'}"
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

    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    print(
        "Mode: "
        + (
            "FSM + hysteresis"
            if use_fsm
            else "basic single-threshold"
        )
    )
    print(f"Landmark backend: {args.landmark_backend}")
    print(f"Frames: {total_frames if total_frames > 0 else 'unknown'}")
    print("The first 3 seconds with a valid face are used for calibration.")

    frame_index = 0
    tensorrt_warmed_up = False
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
            crop_box = None
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
                    landmarks, crop_box = landmark_detector.detect(
                        frame,
                        detection.box,
                    )
                    pfld_inference_ms = (
                        time.perf_counter() - pfld_started_at
                    ) * 1000.0
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
            if landmarks is not None:
                mean_ear, right_ear, left_ear = mean_eye_aspect_ratio(
                    landmarks
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

            eye_state = eye_monitor.update(mean_ear, timestamp=timestamp)
            perclos_state = perclos_monitor.update(
                is_closed=eye_state.is_closed,
                valid_face=eye_state.valid_face,
                timestamp=timestamp,
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
                    "pfld_inference_ms": pfld_inference_ms,
                    "face_box": (
                        None
                        if detection is None
                        else json.dumps(
                            [int(value) for value in detection.box]
                        )
                    ),
                    "crop_box": (
                        None
                        if crop_box is None
                        else json.dumps([int(value) for value in crop_box])
                    ),
                    "landmarks_json": (
                        None
                        if landmarks is None
                        else json.dumps(landmarks.tolist())
                    ),
                },
            )
            frame_index += 1
            if (
                args.max_frames is not None
                and frame_index >= args.max_frames
            ):
                break
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
        close_detector = getattr(landmark_detector, "close", None)
        if close_detector is not None:
            close_detector()
        writer.release()
        capture.release()
        cv2.destroyAllWindows()

    measured_pfld_ms = [
        float(extras["pfld_inference_ms"])
        for frame_metrics, extras in zip(
            logger.frames, logger.frame_extras
        )
        if (
            not frame_metrics.is_warmup
            and extras.get("pfld_inference_ms") is not None
        )
    ]
    if measured_pfld_ms:
        pfld_summary = {
            "pfld_measured_frames": len(measured_pfld_ms),
            "pfld_inference_mean_ms": float(np.mean(measured_pfld_ms)),
            "pfld_inference_median_ms": float(
                np.median(measured_pfld_ms)
            ),
            "pfld_inference_p95_ms": float(
                np.percentile(measured_pfld_ms, 95)
            ),
        }
    else:
        pfld_summary = {
            "pfld_measured_frames": 0,
            "pfld_inference_mean_ms": None,
            "pfld_inference_median_ms": None,
            "pfld_inference_p95_ms": None,
        }

    summary = logger.write_csv(extra_summary_metrics=pfld_summary)
    logger.print_summary(summary)
    print(
        "PFLD latency: "
        f"mean {pfld_summary['pfld_inference_mean_ms']:.2f} ms, "
        f"median {pfld_summary['pfld_inference_median_ms']:.2f} ms, "
        f"P95 {pfld_summary['pfld_inference_p95_ms']:.2f} ms"
        if measured_pfld_ms
        else "PFLD latency: N/A"
    )
    print(f"Annotated video: {output_path}")
    print(f"Frame CSV: {logger.frame_csv_path}")
    print(f"Summary CSV: {logger.summary_csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
