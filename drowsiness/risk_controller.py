from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    NORMAL = "NORMAL"
    PRE_DROWSY = "PRE_DROWSY"
    DROWSY = "DROWSY"


class BuzzerMode(str, Enum):
    OFF = "OFF"
    ALERT = "ALERT"
    EMERGENCY = "EMERGENCY"


@dataclass(frozen=True)
class RiskDecision:
    timestamp: float
    level: RiskLevel
    previous_level: RiskLevel
    reasons: tuple[str, ...]
    recent_yawn_count: int
    buzzer_mode: BuzzerMode
    hazard_light: bool
    stop_request: bool
    changed: bool
    acknowledged: bool


class DrowsinessRiskController:
    """Combine eye FSM, PERCLOS, and yawns into actionable risk states.

    DROWSY is latched by default. It only clears after ``acknowledge`` is
    called and all danger inputs remain clear for ``recovery_seconds``.
    The controller produces requests; vehicle hardware decides how to carry
    out a controlled stop.
    """

    def __init__(
        self,
        yawn_window_seconds: float = 60.0,
        pre_drowsy_yawn_count: int = 2,
        recovery_seconds: float = 5.0,
        latch_drowsy: bool = True,
    ) -> None:
        if yawn_window_seconds <= 0:
            raise ValueError("yawn_window_seconds must be positive.")
        if pre_drowsy_yawn_count <= 0:
            raise ValueError("pre_drowsy_yawn_count must be positive.")
        if recovery_seconds < 0:
            raise ValueError("recovery_seconds must not be negative.")
        self.yawn_window_seconds = float(yawn_window_seconds)
        self.pre_drowsy_yawn_count = int(pre_drowsy_yawn_count)
        self.recovery_seconds = float(recovery_seconds)
        self.latch_drowsy = bool(latch_drowsy)
        self.reset()

    def reset(self) -> None:
        self._level = RiskLevel.NORMAL
        self._yawn_events: deque[float] = deque()
        self._was_yawning = False
        self._acknowledged = False
        self._safe_started_at: float | None = None

    @property
    def level(self) -> RiskLevel:
        return self._level

    def acknowledge(self) -> None:
        """Record explicit driver acknowledgement for a latched DROWSY state."""
        self._acknowledged = True
        self._safe_started_at = None

    def update(
        self,
        *,
        timestamp: float,
        eye_danger: bool,
        perclos_caution: bool,
        perclos_warning: bool,
        is_yawning: bool,
        valid_face: bool,
    ) -> RiskDecision:
        now = float(timestamp)
        if is_yawning and not self._was_yawning:
            self._yawn_events.append(now)
        self._was_yawning = bool(is_yawning)
        self._evict_old_yawns(now)

        recent_yawns = len(self._yawn_events)
        danger_reasons: list[str] = []
        if eye_danger:
            danger_reasons.append("CONTINUOUS_EYE_CLOSURE")
        if perclos_warning and recent_yawns >= 1:
            danger_reasons.append("PERCLOS_WARNING_WITH_YAWN")

        pre_reasons: list[str] = []
        if recent_yawns >= self.pre_drowsy_yawn_count:
            pre_reasons.append("REPEATED_YAWN")
        if perclos_caution:
            pre_reasons.append("PERCLOS_CAUTION")

        previous = self._level
        if danger_reasons:
            level = RiskLevel.DROWSY
            reasons = danger_reasons
            self._acknowledged = False
            self._safe_started_at = None
        elif self._level is RiskLevel.DROWSY and self.latch_drowsy:
            level, reasons = self._update_latched_state(
                now, valid_face, pre_reasons
            )
        elif pre_reasons:
            level = RiskLevel.PRE_DROWSY
            reasons = pre_reasons
        else:
            level = RiskLevel.NORMAL
            reasons = [] if valid_face else ["NO_FACE"]

        self._level = level
        return self._make_decision(
            now, previous, reasons, recent_yawns
        )

    def _update_latched_state(
        self,
        now: float,
        valid_face: bool,
        pre_reasons: list[str],
    ) -> tuple[RiskLevel, list[str]]:
        if not self._acknowledged:
            return RiskLevel.DROWSY, ["AWAITING_ACKNOWLEDGEMENT"]
        if not valid_face or pre_reasons:
            self._safe_started_at = None
            return RiskLevel.DROWSY, ["RECOVERY_NOT_SAFE"]
        if self._safe_started_at is None:
            self._safe_started_at = now
        if now - self._safe_started_at < self.recovery_seconds:
            return RiskLevel.DROWSY, ["RECOVERY_CONFIRMING"]
        self._acknowledged = False
        self._safe_started_at = None
        return RiskLevel.NORMAL, []

    def _evict_old_yawns(self, now: float) -> None:
        cutoff = now - self.yawn_window_seconds
        while self._yawn_events and self._yawn_events[0] < cutoff:
            self._yawn_events.popleft()

    def _make_decision(
        self,
        now: float,
        previous: RiskLevel,
        reasons: list[str],
        recent_yawns: int,
    ) -> RiskDecision:
        if self._level is RiskLevel.DROWSY:
            buzzer = BuzzerMode.EMERGENCY
            hazard = True
            stop = True
        elif self._level is RiskLevel.PRE_DROWSY:
            buzzer = BuzzerMode.ALERT
            hazard = False
            stop = False
        else:
            buzzer = BuzzerMode.OFF
            hazard = False
            stop = False
        return RiskDecision(
            timestamp=now,
            level=self._level,
            previous_level=previous,
            reasons=tuple(reasons),
            recent_yawn_count=recent_yawns,
            buzzer_mode=buzzer,
            hazard_light=hazard,
            stop_request=stop,
            changed=self._level is not previous,
            acknowledged=self._acknowledged,
        )
