"""Compare OpenCV FP32 and TensorRT FP16 PFLD outputs on identical frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drowsiness.detectors import PFLDLandmarkDetector, YuNetFaceDetector
from drowsiness.tensorrt_pfld import TensorRTPFLDLandmarkDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same video frames and YuNet face boxes through OpenCV FP32 "
            "and TensorRT FP16 PFLD, then report landmark-coordinate errors."
        )
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=PROJECT_ROOT / "data/converted/final_test.mp4",
    )
    parser.add_argument(
        "--yunet",
        type=Path,
        default=PROJECT_ROOT / "models/face_detector/yunet.onnx",
    )
    parser.add_argument(
        "--pfld",
        type=Path,
        default=PROJECT_ROOT / "models/landmark/pfld_sim.onnx",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=PROJECT_ROOT / "models/engine/pfld_sim_fp16.engine",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of face-containing frames to compare (default: 10).",
    )
    parser.add_argument(
        "--max-scan-frames",
        type=int,
        default=300,
        help="Maximum video frames to scan while finding faces (default: 300).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="TensorRT warm-up iterations on the first detected face (default: 5).",
    )
    return parser.parse_args()


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero, got {value}.")


def main() -> int:
    args = parse_args()
    require_positive("--frames", args.frames)
    require_positive("--max-scan-frames", args.max_scan_frames)
    if args.warmup < 0:
        raise ValueError(f"--warmup must not be negative, got {args.warmup}.")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    face_detector = YuNetFaceDetector(args.yunet, input_size=(640, 360))
    opencv_detector = PFLDLandmarkDetector(args.pfld)
    tensorrt_detector = TensorRTPFLDLandmarkDetector(args.engine)
    frame_metrics: list[tuple[int, float, float, float]] = []
    warmed_up = False

    try:
        for frame_index in range(args.max_scan_frames):
            ok, frame = capture.read()
            if not ok:
                break

            detection = face_detector.detect_largest(frame)
            if detection is None:
                continue

            if not warmed_up:
                for _ in range(args.warmup):
                    tensorrt_detector.detect(frame, detection.box)
                warmed_up = True

            opencv_landmarks, opencv_crop = opencv_detector.detect(
                frame, detection.box
            )
            tensorrt_landmarks, tensorrt_crop = tensorrt_detector.detect(
                frame, detection.box
            )
            if opencv_crop != tensorrt_crop:
                raise RuntimeError(
                    "Detector crop boxes differ: "
                    f"OpenCV={opencv_crop}, TensorRT={tensorrt_crop}."
                )
            if not (
                np.isfinite(opencv_landmarks).all()
                and np.isfinite(tensorrt_landmarks).all()
            ):
                raise RuntimeError(f"Non-finite landmark at video frame {frame_index}.")

            point_errors = np.linalg.norm(
                opencv_landmarks - tensorrt_landmarks, axis=1
            )
            x1, y1, x2, y2 = opencv_crop
            crop_diagonal = float(np.hypot(x2 - x1, y2 - y1))
            mean_error = float(point_errors.mean())
            max_error = float(point_errors.max())
            normalized_mean_error = mean_error / crop_diagonal
            frame_metrics.append(
                (frame_index, mean_error, max_error, normalized_mean_error)
            )
            print(
                f"frame={frame_index:4d} "
                f"mean={mean_error:.6f}px "
                f"max={max_error:.6f}px "
                f"NME={normalized_mean_error:.8f}"
            )

            if len(frame_metrics) == args.frames:
                break
    finally:
        tensorrt_detector.close()
        capture.release()

    if len(frame_metrics) != args.frames:
        raise RuntimeError(
            f"Compared only {len(frame_metrics)}/{args.frames} face-containing "
            f"frames while scanning at most {args.max_scan_frames} frames."
        )

    means = np.asarray([item[1] for item in frame_metrics])
    maxima = np.asarray([item[2] for item in frame_metrics])
    normalized = np.asarray([item[3] for item in frame_metrics])
    print("\n=== OpenCV FP32 vs TensorRT FP16 PFLD ===")
    print(f"compared frames : {len(frame_metrics)}")
    print(f"mean point error: {means.mean():.6f} px")
    print(f"worst mean error: {means.max():.6f} px")
    print(f"max point error : {maxima.max():.6f} px")
    print(f"mean NME        : {normalized.mean():.8f}")
    print(f"[PASS] {len(frame_metrics)}-frame PFLD output comparison completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
