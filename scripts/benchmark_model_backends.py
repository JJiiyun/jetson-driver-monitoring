#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drowsiness.detectors import (
    FaceDetection,
    PFLDLandmarkDetector,
    PFLDTensorRTDetector,
    TensorRTYuNetFaceDetector,
    YuNetFaceDetector,
)


MODELS = PROJECT_ROOT / "models"
RESULTS = PROJECT_ROOT / "benchmark/results"
YUNET_ONNX = MODELS / "face_detector/yunet.onnx"
PFLD_ONNX = MODELS / "landmark/pfld_sim.onnx"
YUNET_ENGINES = {
    "tensorrt-fp32": MODELS / "engines/fp32/yunet_fp32.engine",
    "tensorrt-fp16": MODELS / "engines/fp16/yunet_fp16.engine",
}
PFLD_ENGINES = {
    "tensorrt-fp32": MODELS / "engines/fp32/pfld_fp32.engine",
    "tensorrt-fp16": MODELS / "engines/fp16/pfld_fp16.engine",
}
INPUT_SIZE = (640, 640)


class OpenCVLetterboxYuNet:
    """Run OpenCV CUDA YuNet with the same 640x640 letterbox as TensorRT."""

    def __init__(self) -> None:
        self._detector = YuNetFaceDetector(
            YUNET_ONNX,
            input_size=INPUT_SIZE,
            score_threshold=0.8,
            device="cuda",
        )

    def detect_largest(self, frame: np.ndarray) -> FaceDetection | None:
        frame_height, frame_width = frame.shape[:2]
        input_width, input_height = INPUT_SIZE
        scale = min(input_width / frame_width, input_height / frame_height)
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))
        resized = cv2.resize(frame, (resized_width, resized_height))
        canvas = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        canvas[:resized_height, :resized_width] = resized
        detection = self._detector.detect_largest(canvas)
        if detection is None:
            return None
        x, y, width, height = detection.box
        x1 = max(0, min(frame_width - 1, int(round(x / scale))))
        y1 = max(0, min(frame_height - 1, int(round(y / scale))))
        x2 = max(x1 + 1, min(frame_width, int(round((x + width) / scale))))
        y2 = max(y1 + 1, min(frame_height, int(round((y + height) / scale))))
        return FaceDetection((x1, y1, x2 - x1, y2 - y1), detection.score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark only YuNet and PFLD with matched model inputs."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--backend",
        required=True,
        choices=("opencv-cuda-fp32", "tensorrt-fp32", "tensorrt-fp16"),
    )
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input video does not exist: {args.input}")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be zero or greater")
    return args


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def stats(prefix: str, values: list[float]) -> dict[str, float]:
    return {
        f"{prefix}_mean_ms": statistics.fmean(values),
        f"{prefix}_median_ms": statistics.median(values),
        f"{prefix}_p95_ms": percentile(values, 95),
        f"{prefix}_min_ms": min(values),
        f"{prefix}_max_ms": max(values),
    }


def create_detectors(backend: str):
    if backend == "opencv-cuda-fp32":
        return (
            OpenCVLetterboxYuNet(),
            PFLDLandmarkDetector(PFLD_ONNX, device="cuda"),
        )
    return (
        TensorRTYuNetFaceDetector(YUNET_ENGINES[backend]),
        PFLDTensorRTDetector(PFLD_ENGINES[backend]),
    )


def main() -> int:
    args = parse_args()
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        print(f"[ERROR] Cannot open input video: {args.input}")
        return 1
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    face_detector, landmark_detector = create_detectors(args.backend)
    rows: list[dict[str, object]] = []
    frame_index = 0

    print(f"Backend: {args.backend}")
    print(f"Input: {args.input}")
    print(f"YuNet input: {INPUT_SIZE[0]}x{INPUT_SIZE[1]} letterbox")
    print("PFLD input: 112x112")
    print("Excluded: decode, FSM, EAR/MAR, overlay, display, video encoding")
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            face_started = time.perf_counter_ns()
            detection = face_detector.detect_largest(frame)
            face_ms = (time.perf_counter_ns() - face_started) / 1_000_000.0

            landmark_ms: float | None = None
            if detection is not None:
                landmark_started = time.perf_counter_ns()
                landmark_detector.detect(frame, detection.box)
                landmark_ms = (
                    time.perf_counter_ns() - landmark_started
                ) / 1_000_000.0
            combined_ms = face_ms + (landmark_ms or 0.0)
            rows.append({
                "frame_index": frame_index,
                "is_warmup": frame_index < args.warmup_frames,
                "face_detected": detection is not None,
                "face_score": "" if detection is None else detection.score,
                "yunet_ms": face_ms,
                "pfld_ms": "" if landmark_ms is None else landmark_ms,
                "combined_ms": combined_ms,
            })
            frame_index += 1
            if frame_index % 500 == 0:
                print(f"Processed {frame_index}/{total_frames}")
    finally:
        capture.release()

    measured = rows[args.warmup_frames:]
    if not measured:
        print("[ERROR] No measured frames after warmup.")
        return 1
    yunet = [float(row["yunet_ms"]) for row in measured]
    pfld = [float(row["pfld_ms"]) for row in measured if row["pfld_ms"] != ""]
    combined = [float(row["combined_ms"]) for row in measured]
    detected = sum(bool(row["face_detected"]) for row in measured)
    summary: dict[str, object] = {
        "backend": args.backend,
        "input_source": args.input.name,
        "yunet_input": "640x640_letterbox",
        "pfld_input": "112x112",
        "total_frames": len(rows),
        "warmup_frames": min(args.warmup_frames, len(rows)),
        "measured_frames": len(measured),
        "face_detected_frames": detected,
        "face_detection_rate": detected / len(measured),
    }
    summary.update(stats("yunet", yunet))
    summary.update(stats("pfld", pfld))
    summary.update(stats("combined", combined))
    summary["model_fps"] = 1000.0 / float(summary["combined_mean_ms"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pure_backend_{args.backend.replace('-', '_')}_{timestamp}"
    frames_path = args.output_dir / f"{stem}_frames.csv"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    with frames_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print("\n=== Pure Backend Summary ===")
    print(f"Measured frames: {summary['measured_frames']}")
    print(f"Face detection rate: {summary['face_detection_rate']:.4f}")
    print(
        f"YuNet: mean {summary['yunet_mean_ms']:.3f} ms, "
        f"P95 {summary['yunet_p95_ms']:.3f} ms"
    )
    print(
        f"PFLD: mean {summary['pfld_mean_ms']:.3f} ms, "
        f"P95 {summary['pfld_p95_ms']:.3f} ms"
    )
    print(
        f"Combined: mean {summary['combined_mean_ms']:.3f} ms, "
        f"P95 {summary['combined_p95_ms']:.3f} ms"
    )
    print(f"Model FPS: {summary['model_fps']:.2f}")
    print(f"Frames CSV: {frames_path}")
    print(f"Summary CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    if exit_code == 0:
        # TensorRT and PyCUDA can segfault during interpreter teardown when
        # multiple contexts are destroyed in an unsafe module-finalization
        # order. All benchmark outputs are already closed and flushed here,
        # so let the OS release process-owned CUDA resources.
        os._exit(0)
    raise SystemExit(exit_code)
