from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np


class EyeStatus(str, Enum):
    CALIBRATING = "CALIBRATING"
    NO_FACE = "NO FACE"
    NORMAL = "NORMAL"
    EYES_CLOSED = "EYES CLOSED"
    DANGER = "DANGER"


@dataclass(frozen=True)
class EyeState:
    status: str
    calibrated: bool
    valid_face: bool
    ear: float | None
    baseline_ear: float | None
    relative_ear: float | None
    closed_threshold: float | None
    reopen_threshold: float | None
    is_closed: bool
    closed_seconds: float
    calibration_progress: float
    is_danger: bool


class EyeClosureMonitor:
    """Calibrate an EAR baseline and measure continuous eye-closure time."""

    def __init__(
        self,
        calibration_seconds: float = 3.0,
        closed_ratio: float = 0.70,
        reopen_ratio: float = 0.80,
        danger_seconds: float = 2.0,
        min_calibration_samples: int = 15,
    ) -> None:
        if calibration_seconds <= 0:
            raise ValueError("calibration_seconds must be positive.")
        if not 0.0 < closed_ratio < 1.0:
            raise ValueError("closed_ratio must be between 0 and 1.")
        if not closed_ratio < reopen_ratio <= 1.0:
            raise ValueError(
                "reopen_ratio must be greater than closed_ratio "
                "and no greater than 1."
            )
        if danger_seconds <= 0:
            raise ValueError("danger_seconds must be positive.")

        self.calibration_seconds = float(calibration_seconds)
        self.closed_ratio = float(closed_ratio)
        self.reopen_ratio = float(reopen_ratio)
        self.danger_seconds = float(danger_seconds)
        self.min_calibration_samples = int(min_calibration_samples)
        self.reset()

    def reset(self) -> None:
        self._calibration_started_at: float | None = None
        self._calibration_samples: list[float] = []
        self._baseline_ear: float | None = None
        self._closed_started_at: float | None = None
        self._is_closed = False
        self._status = EyeStatus.CALIBRATING

    @property
    def baseline_ear(self) -> float | None:
        return self._baseline_ear

    @property
    def status(self) -> str:
        return self._status.value

    def update(
        self,
        ear: float | None,
        timestamp: float | None = None,
    ) -> EyeState:
        now = time.monotonic() if timestamp is None else float(timestamp)

        if ear is None or not np.isfinite(ear) or ear <= 0:
            self._closed_started_at = None
            self._is_closed = False
            return self._make_state(
                status=EyeStatus.NO_FACE,
                ear=None,
                now=now,
                valid_face=False,
            )

        current_ear = float(ear)
        if self._baseline_ear is None:
            self._update_calibration(current_ear, now)
            if self._baseline_ear is None:
                return self._make_state(
                    status=EyeStatus.CALIBRATING,
                    ear=current_ear,
                    now=now,
                    valid_face=True,
                )

        relative_ear = current_ear / max(self._baseline_ear, 1e-6)
        if self._is_closed:
            self._is_closed = relative_ear < self.reopen_ratio
        else:
            self._is_closed = relative_ear < self.closed_ratio

        if self._is_closed:
            if self._closed_started_at is None:
                self._closed_started_at = now
            closed_seconds = max(0.0, now - self._closed_started_at)
        else:
            self._closed_started_at = None
            closed_seconds = 0.0

        is_danger = closed_seconds >= self.danger_seconds
        if is_danger:
            status = EyeStatus.DANGER
        elif self._is_closed:
            status = EyeStatus.EYES_CLOSED
        else:
            status = EyeStatus.NORMAL
        self._status = status

        return EyeState(
            status=status.value,
            calibrated=True,
            valid_face=True,
            ear=current_ear,
            baseline_ear=self._baseline_ear,
            relative_ear=relative_ear,
            closed_threshold=self._baseline_ear * self.closed_ratio,
            reopen_threshold=self._baseline_ear * self.reopen_ratio,
            is_closed=self._is_closed,
            closed_seconds=closed_seconds,
            calibration_progress=1.0,
            is_danger=is_danger,
        )

    def _update_calibration(self, ear: float, now: float) -> None:
        if self._calibration_started_at is None:
            self._calibration_started_at = now

        self._calibration_samples.append(ear)
        elapsed = now - self._calibration_started_at
        enough_time = elapsed >= self.calibration_seconds
        enough_samples = (
            len(self._calibration_samples) >= self.min_calibration_samples
        )

        if enough_time and enough_samples:
            self._baseline_ear = float(
                np.median(self._calibration_samples)
            )
            self._closed_started_at = None
            self._is_closed = False
            self._status = EyeStatus.NORMAL

    def _calibration_progress(self, now: float) -> float:
        if self._baseline_ear is not None:
            return 1.0
        if self._calibration_started_at is None:
            return 0.0
        elapsed = max(0.0, now - self._calibration_started_at)
        return min(1.0, elapsed / self.calibration_seconds)

    def _make_state(
        self,
        status: EyeStatus,
        ear: float | None,
        now: float,
        valid_face: bool,
    ) -> EyeState:
        self._status = status
        baseline = self._baseline_ear
        relative = None
        threshold = None
        reopen_threshold = None
        if baseline is not None:
            threshold = baseline * self.closed_ratio
            reopen_threshold = baseline * self.reopen_ratio
            if ear is not None:
                relative = ear / max(baseline, 1e-6)

        return EyeState(
            status=status.value,
            calibrated=baseline is not None,
            valid_face=valid_face,
            ear=ear,
            baseline_ear=baseline,
            relative_ear=relative,
            closed_threshold=threshold,
            reopen_threshold=reopen_threshold,
            is_closed=False,
            closed_seconds=0.0,
            calibration_progress=self._calibration_progress(now),
            is_danger=False,
        )
