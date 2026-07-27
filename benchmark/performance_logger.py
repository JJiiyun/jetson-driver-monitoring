"""Reusable FPS and latency logger for real-time video pipelines."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union


PathLike = Union[str, Path]


@dataclass(frozen=True)
class FrameMetrics:
    """Timing values collected for one processed frame."""

    frame_index: int
    timestamp: str
    is_warmup: bool
    capture_ms: float
    inference_ms: float
    processing_ms: float
    frame_time_ms: float
    instant_fps: float
    face_count: int


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    """Return a linearly interpolated percentile without extra dependencies."""

    if not values:
        return None

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    weight = position - lower_index
    return (
        ordered[lower_index] * (1.0 - weight)
        + ordered[upper_index] * weight
    )


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "benchmark"


class PerformanceLogger:
    """Collect frame timings and write frame/summary CSV files.

    All duration inputs use milliseconds. Warm-up frames are retained in the
    frame CSV for traceability but excluded from the summary statistics.
    """

    FRAME_FIELDS = [
        "run_id",
        "backend",
        "input_source",
        "width",
        "height",
        "target_fps",
        "frame_index",
        "timestamp",
        "is_warmup",
        "capture_ms",
        "inference_ms",
        "processing_ms",
        "frame_time_ms",
        "instant_fps",
        "face_count",
    ]

    SUMMARY_FIELDS = [
        "run_id",
        "backend",
        "input_source",
        "width",
        "height",
        "target_fps",
        "warmup_frames",
        "total_frames",
        "measured_frames",
        "measured_duration_s",
        "end_to_end_fps",
        "capture_mean_ms",
        "capture_p95_ms",
        "inference_mean_ms",
        "inference_p95_ms",
        "processing_mean_ms",
        "processing_p95_ms",
        "frame_time_mean_ms",
        "frame_time_p95_ms",
        "average_face_count",
        "started_at",
        "finished_at",
    ]

    def __init__(
        self,
        backend: str,
        output_dir: PathLike = "benchmark/results",
        run_name: Optional[str] = None,
        warmup_frames: int = 30,
        input_source: str = "camera:0",
        width: int = 0,
        height: int = 0,
        target_fps: float = 0.0,
        extra_frame_fields: Optional[Sequence[str]] = None,
    ) -> None:
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be zero or greater")

        extra_fields = list(extra_frame_fields or [])
        duplicate_fields = set(extra_fields) & set(self.FRAME_FIELDS)
        if duplicate_fields:
            raise ValueError(
                "Extra frame fields duplicate built-in fields: "
                f"{sorted(duplicate_fields)}"
            )
        if len(extra_fields) != len(set(extra_fields)):
            raise ValueError("Extra frame field names must be unique")

        now = datetime.now().astimezone()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        base_name = _safe_name(run_name or backend)

        self.backend = backend
        self.output_dir = Path(output_dir)
        self.run_id = f"{base_name}_{timestamp}"
        self.warmup_frames = warmup_frames
        self.input_source = input_source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.extra_frame_fields = extra_fields
        self.started_at = now.isoformat(timespec="milliseconds")
        self.frames: List[FrameMetrics] = []
        self.frame_extras: List[Dict[str, object]] = []

        self.frame_csv_path = self.output_dir / f"{self.run_id}_frames.csv"
        self.summary_csv_path = self.output_dir / f"{self.run_id}_summary.csv"

    def record_frame(
        self,
        frame_started_at: float,
        frame_finished_at: float,
        capture_ms: float,
        inference_ms: float = 0.0,
        face_count: int = 0,
        extra_metrics: Optional[Mapping[str, object]] = None,
    ) -> FrameMetrics:
        """Record one frame using perf_counter timestamps and stage timings."""

        extras = dict(extra_metrics or {})
        unknown_fields = set(extras) - set(self.extra_frame_fields)
        if unknown_fields:
            raise ValueError(
                "Unknown extra frame fields: "
                f"{sorted(unknown_fields)}"
            )

        frame_time_ms = max(
            0.0,
            (frame_finished_at - frame_started_at) * 1000.0,
        )
        processing_ms = max(
            0.0,
            frame_time_ms - capture_ms - inference_ms,
        )
        frame_index = len(self.frames) + 1

        metrics = FrameMetrics(
            frame_index=frame_index,
            timestamp=datetime.now()
            .astimezone()
            .isoformat(timespec="milliseconds"),
            is_warmup=frame_index <= self.warmup_frames,
            capture_ms=max(0.0, capture_ms),
            inference_ms=max(0.0, inference_ms),
            processing_ms=processing_ms,
            frame_time_ms=frame_time_ms,
            instant_fps=(
                1000.0 / frame_time_ms if frame_time_ms > 0.0 else 0.0
            ),
            face_count=max(0, int(face_count)),
        )
        self.frames.append(metrics)
        self.frame_extras.append(extras)
        return metrics

    @property
    def measured_frames(self) -> List[FrameMetrics]:
        return [frame for frame in self.frames if not frame.is_warmup]

    @property
    def current_fps(self) -> float:
        """Return live FPS over up to the latest 30 recorded frames."""

        recent = self.frames[-30:]
        duration_ms = sum(frame.frame_time_ms for frame in recent)
        if not recent or duration_ms <= 0.0:
            return 0.0
        return len(recent) * 1000.0 / duration_ms

    def summary(self) -> Dict[str, object]:
        measured = self.measured_frames
        duration_s = sum(frame.frame_time_ms for frame in measured) / 1000.0

        capture = [frame.capture_ms for frame in measured]
        inference = [frame.inference_ms for frame in measured]
        processing = [frame.processing_ms for frame in measured]
        frame_time = [frame.frame_time_ms for frame in measured]
        face_counts = [float(frame.face_count) for frame in measured]

        return {
            "run_id": self.run_id,
            "backend": self.backend,
            "input_source": self.input_source,
            "width": self.width,
            "height": self.height,
            "target_fps": self.target_fps,
            "warmup_frames": self.warmup_frames,
            "total_frames": len(self.frames),
            "measured_frames": len(measured),
            "measured_duration_s": duration_s,
            "end_to_end_fps": (
                len(measured) / duration_s if duration_s > 0.0 else None
            ),
            "capture_mean_ms": _mean(capture),
            "capture_p95_ms": _percentile(capture, 0.95),
            "inference_mean_ms": _mean(inference),
            "inference_p95_ms": _percentile(inference, 0.95),
            "processing_mean_ms": _mean(processing),
            "processing_p95_ms": _percentile(processing, 0.95),
            "frame_time_mean_ms": _mean(frame_time),
            "frame_time_p95_ms": _percentile(frame_time, 0.95),
            "average_face_count": _mean(face_counts),
            "started_at": self.started_at,
            "finished_at": datetime.now()
            .astimezone()
            .isoformat(timespec="milliseconds"),
        }

    def write_csv(self) -> Dict[str, object]:
        """Write frame-level and run-summary CSV files, then return summary."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        common = {
            "run_id": self.run_id,
            "backend": self.backend,
            "input_source": self.input_source,
            "width": self.width,
            "height": self.height,
            "target_fps": self.target_fps,
        }

        with self.frame_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            frame_fields = self.FRAME_FIELDS + self.extra_frame_fields
            writer = csv.DictWriter(csv_file, fieldnames=frame_fields)
            writer.writeheader()
            for frame, extras in zip(self.frames, self.frame_extras):
                writer.writerow({**common, **asdict(frame), **extras})

        summary = self.summary()
        with self.summary_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerow(summary)

        return summary

    @staticmethod
    def print_summary(summary: Dict[str, object]) -> None:
        def display(key: str, suffix: str = "") -> str:
            value = summary.get(key)
            if value is None:
                return "N/A"
            if isinstance(value, float):
                return f"{value:.2f}{suffix}"
            return f"{value}{suffix}"

        print("\n=== Performance Summary ===")
        print(f"Run ID: {summary['run_id']}")
        print(f"Measured frames: {summary['measured_frames']}")
        print(f"End-to-end FPS: {display('end_to_end_fps')}")
        print(
            "Inference latency: "
            f"mean {display('inference_mean_ms', ' ms')}, "
            f"P95 {display('inference_p95_ms', ' ms')}"
        )
        print(
            "Frame latency: "
            f"mean {display('frame_time_mean_ms', ' ms')}, "
            f"P95 {display('frame_time_p95_ms', ' ms')}"
        )
