#!/usr/bin/env python3
"""Compare frame logs from TensorRT FP32 and FP16 PFLD pipelines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align FP32 and FP16 frame CSVs by frame_index and compare "
            "landmarks, EAR, FSM states, and latency."
        )
    )
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--fp16", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("benchmark/results/tensorrt_fp32_vs_fp16"),
        help="Output path without the _frames.csv/_summary.csv suffix.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Frame CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {
            "frame_index",
            "input_source",
            "width",
            "height",
            "target_fps",
            "is_warmup",
            "ear",
            "mar",
            "eye_state",
            "is_eye_closed",
            "is_yawning",
            "pfld_inference_ms",
            "inference_ms",
            "frame_time_ms",
            "landmarks_json",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )
        rows: dict[int, dict[str, str]] = {}
        for row in reader:
            frame_index = int(row["frame_index"])
            if frame_index in rows:
                raise ValueError(
                    f"{path} contains duplicate frame_index {frame_index}."
                )
            rows[frame_index] = row
    if not rows:
        raise ValueError(f"{path} contains no frame rows.")
    return rows


def validate_alignment(
    fp32_rows: dict[int, dict[str, str]],
    fp16_rows: dict[int, dict[str, str]],
) -> None:
    if set(fp32_rows) != set(fp16_rows):
        only_fp32 = sorted(set(fp32_rows) - set(fp16_rows))
        only_fp16 = sorted(set(fp16_rows) - set(fp32_rows))
        raise ValueError(
            "Frame sets differ: "
            f"only FP32={only_fp32[:5]}, only FP16={only_fp16[:5]}."
        )

    first_index = min(fp32_rows)
    fp32_first = fp32_rows[first_index]
    fp16_first = fp16_rows[first_index]
    for field in ("input_source", "width", "height", "target_fps"):
        if fp32_first[field] != fp16_first[field]:
            raise ValueError(
                f"Input metadata differs for {field}: "
                f"FP32={fp32_first[field]!r}, FP16={fp16_first[field]!r}."
            )


def optional_float(row: dict[str, str], field: str) -> float | None:
    raw = row.get(field, "").strip()
    if not raw:
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def optional_landmarks(row: dict[str, str]) -> np.ndarray | None:
    raw = row.get("landmarks_json", "").strip()
    if not raw:
        return None
    landmarks = np.asarray(json.loads(raw), dtype=np.float32)
    if landmarks.shape != (68, 2) or not np.isfinite(landmarks).all():
        raise ValueError(
            "landmarks_json must contain finite landmarks with shape (68, 2)."
        )
    return landmarks


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def percentile95(values: list[float]) -> float | None:
    return float(np.percentile(values, 95)) if values else None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def compare(
    fp32_rows: dict[int, dict[str, str]],
    fp16_rows: dict[int, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_alignment(fp32_rows, fp16_rows)
    comparisons: list[dict[str, Any]] = []

    for frame_index in sorted(fp32_rows):
        fp32 = fp32_rows[frame_index]
        fp16 = fp16_rows[frame_index]
        fp32_landmarks = optional_landmarks(fp32)
        fp16_landmarks = optional_landmarks(fp16)
        mean_landmark_error = None
        max_landmark_error = None
        if fp32_landmarks is not None and fp16_landmarks is not None:
            point_errors = np.linalg.norm(
                fp32_landmarks - fp16_landmarks, axis=1
            )
            mean_landmark_error = float(point_errors.mean())
            max_landmark_error = float(point_errors.max())

        fp32_ear = optional_float(fp32, "ear")
        fp16_ear = optional_float(fp16, "ear")
        ear_absolute_error = (
            abs(fp32_ear - fp16_ear)
            if fp32_ear is not None and fp16_ear is not None
            else None
        )
        fp32_mar = optional_float(fp32, "mar")
        fp16_mar = optional_float(fp16, "mar")
        mar_absolute_error = (
            abs(fp32_mar - fp16_mar)
            if fp32_mar is not None and fp16_mar is not None
            else None
        )
        comparisons.append(
            {
                "frame_index": frame_index,
                "is_warmup": (
                    truthy(fp32["is_warmup"])
                    or truthy(fp16["is_warmup"])
                ),
                "landmark_mean_error_px": mean_landmark_error,
                "landmark_max_error_px": max_landmark_error,
                "fp32_ear": fp32_ear,
                "fp16_ear": fp16_ear,
                "ear_absolute_error": ear_absolute_error,
                "fp32_mar": fp32_mar,
                "fp16_mar": fp16_mar,
                "mar_absolute_error": mar_absolute_error,
                "fp32_eye_state": fp32["eye_state"],
                "fp16_eye_state": fp16["eye_state"],
                "fsm_state_match": fp32["eye_state"] == fp16["eye_state"],
                "eye_closed_match": (
                    truthy(fp32["is_eye_closed"])
                    == truthy(fp16["is_eye_closed"])
                ),
                "yawn_state_match": (
                    truthy(fp32["is_yawning"])
                    == truthy(fp16["is_yawning"])
                ),
                "fp32_pfld_ms": optional_float(fp32, "pfld_inference_ms"),
                "fp16_pfld_ms": optional_float(fp16, "pfld_inference_ms"),
                "fp32_inference_ms": optional_float(fp32, "inference_ms"),
                "fp16_inference_ms": optional_float(fp16, "inference_ms"),
                "fp32_frame_time_ms": optional_float(fp32, "frame_time_ms"),
                "fp16_frame_time_ms": optional_float(fp16, "frame_time_ms"),
            }
        )

    measured = [row for row in comparisons if not row["is_warmup"]]
    landmark_means = [
        row["landmark_mean_error_px"]
        for row in comparisons
        if row["landmark_mean_error_px"] is not None
    ]
    landmark_maxima = [
        row["landmark_max_error_px"]
        for row in comparisons
        if row["landmark_max_error_px"] is not None
    ]
    ear_errors = [
        row["ear_absolute_error"]
        for row in comparisons
        if row["ear_absolute_error"] is not None
    ]
    mar_errors = [
        row["mar_absolute_error"]
        for row in comparisons
        if row["mar_absolute_error"] is not None
    ]

    def measured_values(field: str) -> list[float]:
        return [
            float(row[field])
            for row in measured
            if row[field] is not None
        ]

    fp32_pfld = measured_values("fp32_pfld_ms")
    fp16_pfld = measured_values("fp16_pfld_ms")
    fp32_inference = measured_values("fp32_inference_ms")
    fp16_inference = measured_values("fp16_inference_ms")
    fp32_frame_time = measured_values("fp32_frame_time_ms")
    fp16_frame_time = measured_values("fp16_frame_time_ms")
    fp32_frame_mean = mean(fp32_frame_time)
    fp16_frame_mean = mean(fp16_frame_time)

    summary = {
        "total_frames": len(comparisons),
        "measured_frames": len(measured),
        "landmark_compared_frames": len(landmark_means),
        "landmark_mean_error_px": mean(landmark_means),
        "landmark_max_error_px": (
            max(landmark_maxima) if landmark_maxima else None
        ),
        "ear_compared_frames": len(ear_errors),
        "ear_mean_absolute_error": mean(ear_errors),
        "ear_max_absolute_error": max(ear_errors) if ear_errors else None,
        "mar_compared_frames": len(mar_errors),
        "mar_mean_absolute_error": mean(mar_errors),
        "mar_max_absolute_error": max(mar_errors) if mar_errors else None,
        "fsm_state_match_rate": mean(
            [float(row["fsm_state_match"]) for row in comparisons]
        ),
        "eye_closed_match_rate": mean(
            [float(row["eye_closed_match"]) for row in comparisons]
        ),
        "yawn_state_match_rate": mean(
            [float(row["yawn_state_match"]) for row in comparisons]
        ),
        "fp32_pfld_mean_ms": mean(fp32_pfld),
        "fp16_pfld_mean_ms": mean(fp16_pfld),
        "pfld_fp16_speedup": ratio(mean(fp32_pfld), mean(fp16_pfld)),
        "fp32_inference_mean_ms": mean(fp32_inference),
        "fp16_inference_mean_ms": mean(fp16_inference),
        "inference_fp16_speedup": ratio(
            mean(fp32_inference), mean(fp16_inference)
        ),
        "fp32_frame_mean_ms": fp32_frame_mean,
        "fp16_frame_mean_ms": fp16_frame_mean,
        "fp32_end_to_end_fps": (
            1000.0 / fp32_frame_mean
            if fp32_frame_mean is not None and fp32_frame_mean > 0.0
            else None
        ),
        "fp16_end_to_end_fps": (
            1000.0 / fp16_frame_mean
            if fp16_frame_mean is not None and fp16_frame_mean > 0.0
            else None
        ),
        "end_to_end_fp16_speedup": ratio(
            fp32_frame_mean, fp16_frame_mean
        ),
        "fp32_pfld_median_ms": median(fp32_pfld),
        "fp16_pfld_median_ms": median(fp16_pfld),
        "fp32_pfld_p95_ms": percentile95(fp32_pfld),
        "fp16_pfld_p95_ms": percentile95(fp16_pfld),
    }
    return comparisons, summary


def write_results(
    output_prefix: Path,
    comparisons: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frames_path = output_prefix.with_name(
        f"{output_prefix.name}_frames.csv"
    )
    summary_path = output_prefix.with_name(
        f"{output_prefix.name}_summary.csv"
    )
    with frames_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return frames_path, summary_path


def format_metric(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    args = parse_args()
    fp32_rows = load_rows(args.fp32)
    fp16_rows = load_rows(args.fp16)
    comparisons, summary = compare(fp32_rows, fp16_rows)
    frames_path, summary_path = write_results(
        args.output_prefix, comparisons, summary
    )

    print("=== TensorRT FP32 vs FP16 ===")
    print(f"Frames: {summary['total_frames']}")
    print(
        "Landmark error: mean "
        f"{format_metric(summary['landmark_mean_error_px'])} px, max "
        f"{format_metric(summary['landmark_max_error_px'])} px"
    )
    print(
        "EAR error: mean "
        f"{format_metric(summary['ear_mean_absolute_error'])}, max "
        f"{format_metric(summary['ear_max_absolute_error'])}"
    )
    print(
        "FSM state match: "
        f"{format_metric(summary['fsm_state_match_rate'] * 100, 2)}%"
    )
    print(
        "MAR error: mean "
        f"{format_metric(summary['mar_mean_absolute_error'])}, max "
        f"{format_metric(summary['mar_max_absolute_error'])}"
    )
    print(
        "Yawn state match: "
        f"{format_metric(summary['yawn_state_match_rate'] * 100, 2)}%"
    )
    print(
        "PFLD FP16 speedup: "
        f"{format_metric(summary['pfld_fp16_speedup'], 3)}x"
    )
    print(
        "End-to-end FP16 speedup: "
        f"{format_metric(summary['end_to_end_fp16_speedup'], 3)}x"
    )
    print(f"Frame comparison CSV: {frames_path}")
    print(f"Summary CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
