#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drowsiness.actions import RiskEventPublisher
from drowsiness.qt_dashboard import create_risk_dashboard
from drowsiness.risk_controller import DrowsinessRiskController


def main() -> int:
    try:
        try:
            from PySide6.QtCore import QTimer
            from PySide6.QtWidgets import QApplication
        except ImportError:
            from PyQt5.QtCore import QTimer
            from PyQt5.QtWidgets import QApplication
    except ImportError:
        print("[ERROR] Install PySide6 or PyQt5 to run the Qt dashboard.")
        return 1

    app = QApplication(sys.argv)
    controller = DrowsinessRiskController(recovery_seconds=5.0)
    publisher = RiskEventPublisher()
    dashboard = create_risk_dashboard(controller, publisher)
    dashboard.show()

    # Preview sequence: normal → two yawns → eye danger. Production code calls
    # the same publisher with decisions from each inference frame.
    def publish_at(second: float, **overrides: bool) -> None:
        values = {
            "eye_danger": False,
            "perclos_caution": False,
            "perclos_warning": False,
            "is_yawning": False,
            "valid_face": True,
        }
        values.update(overrides)
        decision = controller.update(timestamp=second, **values)
        publisher.publish(decision)

    events = [
        (100, lambda: publish_at(0.1)),
        (1800, lambda: publish_at(1.8, is_yawning=True)),
        (2200, lambda: publish_at(2.2)),
        (4000, lambda: publish_at(4.0, is_yawning=True)),
        (4400, lambda: publish_at(4.4)),
        (6500, lambda: publish_at(6.5, eye_danger=True)),
        (7000, lambda: publish_at(7.0)),
    ]
    for delay_ms, callback in events:
        QTimer.singleShot(delay_ms, callback)

    exec_app = getattr(app, "exec", None) or app.exec_
    return int(exec_app())


if __name__ == "__main__":
    sys.exit(main())
