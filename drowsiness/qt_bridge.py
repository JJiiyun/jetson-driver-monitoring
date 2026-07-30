from __future__ import annotations

from typing import Any


def create_qt_risk_bridge(parent: Any = None) -> Any:
    """Return a QObject bridge without forcing a Qt dependency on the core.

    The returned object is callable and can be subscribed directly to
    ``RiskEventPublisher``. PySide6 is preferred, with PyQt5 as fallback.
    """
    try:
        from PySide6.QtCore import QObject, Signal
    except ImportError:
        try:
            from PyQt5.QtCore import QObject, pyqtSignal as Signal
        except ImportError as error:
            raise RuntimeError("Install PySide6 or PyQt5 to use the Qt bridge.") from error

    class QtRiskBridge(QObject):
        decision_changed = Signal(object)
        risk_level_changed = Signal(str)
        actions_changed = Signal(str, bool, bool)

        def __call__(self, decision: Any) -> None:
            self.decision_changed.emit(decision)
            if decision.changed:
                self.risk_level_changed.emit(decision.level.value)
            self.actions_changed.emit(
                decision.buzzer_mode.value,
                decision.hazard_light,
                decision.stop_request,
            )

    return QtRiskBridge(parent)
