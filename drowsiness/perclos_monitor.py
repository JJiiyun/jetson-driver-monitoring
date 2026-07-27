from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PerclosState:
    perclos: float
    window_seconds: float
    valid_seconds: float
    closed_seconds: float
    is_warning: bool
    is_caution: bool


class PerclosMonitor:
    """Accumulate eye-closure over a sliding time window and report PERCLOS.

    PERCLOS = closed time / valid-face time within the most recent
    ``window_seconds``. Time is accumulated in seconds (not frame counts),
    so the value is stable across different processing frame rates
    (e.g. FP32 vs FP16 backends).

    This is distinct from ``EyeClosureMonitor.closed_seconds``, which tracks
    a single continuous closure. PERCLOS is a windowed ratio.
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        caution_perclos: float = 0.15,
        warning_perclos: float = 0.30,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        if not 0.0 <= caution_perclos <= 1.0:
            raise ValueError("caution_perclos must be between 0 and 1.")
        if not 0.0 <= warning_perclos <= 1.0:
            raise ValueError("warning_perclos must be between 0 and 1.")
        if warning_perclos < caution_perclos:
            raise ValueError(
                "warning_perclos must be greater than or equal to caution_perclos."
            )

        self.window_seconds = float(window_seconds)
        self.caution_perclos = float(caution_perclos)
        self.warning_perclos = float(warning_perclos)
        self.reset()

    def reset(self) -> None:
        # Each item: (timestamp, dt, valid_face, counted_closed)
        self._events: deque[tuple[float, float, bool, bool]] = deque()
        self._last_timestamp: float | None = None

    def update(
        self,
        is_closed: bool,
        valid_face: bool,
        timestamp: float | None = None,
    ) -> PerclosState:
        """Process one frame and return the current PERCLOS state.

        is_closed:   whether the eyes are judged closed this frame.
        valid_face:  whether a face was validly detected this frame.
        timestamp:   frame time in seconds. For video files use
                     ``frame_index / fps`` so the time axis is independent
                     of processing speed. Defaults to ``time.monotonic()``.
        """
        now = time.monotonic() if timestamp is None else float(timestamp)

        if self._last_timestamp is None:
            dt = 0.0
        else:
            dt = now - self._last_timestamp
            if dt < 0.0:
                dt = 0.0
        self._last_timestamp = now

        # Closure only counts while a face is validly detected.
        counted_closed = bool(valid_face and is_closed)
        self._events.append((now, dt, bool(valid_face), counted_closed))

        self._evict_old(now)

        valid_seconds = 0.0
        closed_seconds = 0.0
        for _ts, ev_dt, ev_valid, ev_closed in self._events:
            if ev_valid:
                valid_seconds += ev_dt
            if ev_closed:
                closed_seconds += ev_dt

        perclos = closed_seconds / valid_seconds if valid_seconds > 0 else 0.0

        return PerclosState(
            perclos=perclos,
            window_seconds=self.window_seconds,
            valid_seconds=valid_seconds,
            closed_seconds=closed_seconds,
            is_warning=perclos >= self.warning_perclos,
            is_caution=perclos >= self.caution_perclos,
        )

    def _evict_old(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
