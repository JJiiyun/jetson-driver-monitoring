from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .actions import RiskEventPublisher
from .qt_bridge import create_qt_risk_bridge
from .risk_controller import DrowsinessRiskController, RiskDecision, RiskLevel


LEVEL_PRESENTATION = {
    RiskLevel.NORMAL: ("정상", "#1f9d68", "현재 뚜렷한 졸음 위험이 없습니다."),
    RiskLevel.PRE_DROWSY: (
        "졸음 직전",
        "#e7a326",
        "주의가 필요합니다. 자세를 바로잡고 휴식을 준비하세요.",
    ),
    RiskLevel.DROWSY: (
        "졸음 위험",
        "#d64545",
        "긴급 경고 중입니다. 차량 제어기에 안전 정차를 요청합니다.",
    ),
}

REASON_LABELS = {
    "CONTINUOUS_EYE_CLOSURE": "연속 눈 감김 감지",
    "PERCLOS_WARNING_WITH_YAWN": "높은 PERCLOS와 하품 동시 감지",
    "REPEATED_YAWN": "최근 60초 내 반복 하품",
    "PERCLOS_CAUTION": "PERCLOS 주의 수준",
    "AWAITING_ACKNOWLEDGEMENT": "운전자 확인 대기",
    "RECOVERY_NOT_SAFE": "아직 안전 복귀 조건 미충족",
    "RECOVERY_CONFIRMING": "안전 상태 5초 확인 중",
    "NO_FACE": "운전자 얼굴을 찾을 수 없음",
}


def create_qt_application(argv: list[str]) -> Any:
    """Return the existing QApplication or create one lazily."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        try:
            from PyQt5.QtWidgets import QApplication
        except ImportError as error:
            raise RuntimeError(
                "Install PySide6 or PyQt5 to use the risk dashboard."
            ) from error
    return QApplication.instance() or QApplication(argv)


def reason_text(reasons: tuple[str, ...]) -> str:
    if not reasons:
        return "감지된 위험 요인 없음"
    return " · ".join(REASON_LABELS.get(reason, reason) for reason in reasons)


def create_risk_dashboard(
    controller: DrowsinessRiskController,
    publisher: RiskEventPublisher,
    *,
    parent: Any = None,
    on_close: Callable[[], None] | None = None,
) -> Any:
    """Create a Qt dashboard subscribed to the risk event publisher.

    Qt is imported lazily so headless inference does not require a Qt package.
    PySide6 is preferred and PyQt5 is used as a fallback.
    """
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        try:
            from PyQt5.QtCore import Qt
            from PyQt5.QtWidgets import (
                QFrame,
                QGridLayout,
                QHBoxLayout,
                QLabel,
                QMainWindow,
                QPushButton,
                QVBoxLayout,
                QWidget,
            )
        except ImportError as error:
            raise RuntimeError(
                "Install PySide6 or PyQt5 to use the risk dashboard."
            ) from error

    align_center = (
        Qt.AlignCenter
        if hasattr(Qt, "AlignCenter")
        else Qt.AlignmentFlag.AlignCenter
    )

    class IndicatorCard(QFrame):
        def __init__(self, title: str) -> None:
            super().__init__()
            self.setObjectName("indicatorCard")
            layout = QVBoxLayout(self)
            title_label = QLabel(title)
            title_label.setObjectName("cardTitle")
            self.value_label = QLabel("-")
            self.value_label.setObjectName("cardValue")
            self.value_label.setAlignment(align_center)
            layout.addWidget(title_label)
            layout.addWidget(self.value_label, 1)

        def set_value(self, value: str, active: bool = False) -> None:
            self.value_label.setText(value)
            self.setProperty("active", active)
            self.style().unpolish(self)
            self.style().polish(self)

    class RiskDashboard(QMainWindow):
        def __init__(self) -> None:
            super().__init__(parent)
            self._last_decision: RiskDecision | None = None
            self._bridge = create_qt_risk_bridge(self)
            self._bridge.decision_changed.connect(self.apply_decision)
            publisher.subscribe(self._bridge)

            self.setWindowTitle("ZZM 운전자 졸음 안전 모니터")
            self.setMinimumSize(900, 620)
            root = QWidget()
            self.setCentralWidget(root)
            outer = QVBoxLayout(root)
            outer.setContentsMargins(28, 24, 28, 24)
            outer.setSpacing(18)

            title_row = QHBoxLayout()
            title = QLabel("운전자 졸음 안전 모니터")
            title.setObjectName("title")
            self.connection_label = QLabel("● 실시간 판단 연결")
            self.connection_label.setObjectName("connection")
            title_row.addWidget(title)
            title_row.addStretch(1)
            title_row.addWidget(self.connection_label)
            outer.addLayout(title_row)

            self.status_panel = QFrame()
            self.status_panel.setObjectName("statusPanel")
            status_layout = QVBoxLayout(self.status_panel)
            self.level_label = QLabel("정상")
            self.level_label.setObjectName("riskLevel")
            self.level_label.setAlignment(align_center)
            self.guide_label = QLabel(LEVEL_PRESENTATION[RiskLevel.NORMAL][2])
            self.guide_label.setObjectName("guide")
            self.guide_label.setAlignment(align_center)
            self.guide_label.setWordWrap(True)
            status_layout.addWidget(self.level_label)
            status_layout.addWidget(self.guide_label)
            outer.addWidget(self.status_panel)

            grid = QGridLayout()
            grid.setSpacing(14)
            self.yawn_card = IndicatorCard("최근 하품")
            self.buzzer_card = IndicatorCard("부저")
            self.hazard_card = IndicatorCard("비상등 요청")
            self.stop_card = IndicatorCard("차량 정차 요청")
            grid.addWidget(self.yawn_card, 0, 0)
            grid.addWidget(self.buzzer_card, 0, 1)
            grid.addWidget(self.hazard_card, 0, 2)
            grid.addWidget(self.stop_card, 0, 3)
            outer.addLayout(grid)

            reason_frame = QFrame()
            reason_frame.setObjectName("reasonFrame")
            reason_layout = QVBoxLayout(reason_frame)
            reason_title = QLabel("판단 근거")
            reason_title.setObjectName("sectionTitle")
            self.reason_label = QLabel("감지된 위험 요인 없음")
            self.reason_label.setObjectName("reason")
            self.reason_label.setWordWrap(True)
            reason_layout.addWidget(reason_title)
            reason_layout.addWidget(self.reason_label)
            outer.addWidget(reason_frame)

            bottom = QHBoxLayout()
            self.time_label = QLabel("마지막 판단: 대기 중")
            self.time_label.setObjectName("muted")
            self.ack_button = QPushButton("경고 확인")
            self.ack_button.setObjectName("ackButton")
            self.ack_button.setEnabled(False)
            self.ack_button.clicked.connect(self._acknowledge)
            bottom.addWidget(self.time_label)
            bottom.addStretch(1)
            bottom.addWidget(self.ack_button)
            outer.addLayout(bottom)

            self.setStyleSheet(_style_sheet())
            self._set_level(RiskLevel.NORMAL)
            self.yawn_card.set_value("0회")
            self.buzzer_card.set_value("꺼짐")
            self.hazard_card.set_value("OFF")
            self.stop_card.set_value("없음")

        def apply_decision(self, decision: RiskDecision) -> None:
            self._last_decision = decision
            self._set_level(decision.level)
            self.reason_label.setText(reason_text(decision.reasons))
            self.yawn_card.set_value(f"{decision.recent_yawn_count}회")
            self.buzzer_card.set_value(
                {"OFF": "꺼짐", "ALERT": "주의 경고", "EMERGENCY": "긴급 경고"}[
                    decision.buzzer_mode.value
                ],
                decision.buzzer_mode.value != "OFF",
            )
            self.hazard_card.set_value(
                "ON" if decision.hazard_light else "OFF",
                decision.hazard_light,
            )
            self.stop_card.set_value(
                "요청 중" if decision.stop_request else "없음",
                decision.stop_request,
            )
            self.ack_button.setEnabled(
                decision.level is RiskLevel.DROWSY and not decision.acknowledged
            )
            self.ack_button.setText(
                "안전 상태 확인 중"
                if decision.acknowledged
                else "경고 확인"
            )
            self.time_label.setText(f"판단 시각: {decision.timestamp:.1f}초")

        def _set_level(self, level: RiskLevel) -> None:
            label, color, guide = LEVEL_PRESENTATION[level]
            self.level_label.setText(label)
            self.guide_label.setText(guide)
            self.status_panel.setStyleSheet(
                f"QFrame#statusPanel {{ border: 3px solid {color}; "
                f"background: {color}; border-radius: 18px; }}"
            )

        def _acknowledge(self) -> None:
            controller.acknowledge()
            self.ack_button.setEnabled(False)
            self.ack_button.setText("안전 상태 확인 중")

        def closeEvent(self, event: Any) -> None:
            publisher.unsubscribe(self._bridge)
            if on_close is not None:
                on_close()
            super().closeEvent(event)

    return RiskDashboard()


def _style_sheet() -> str:
    return """
        QWidget { background: #111827; color: #f8fafc; font-size: 16px; }
        QLabel#title { font-size: 26px; font-weight: 700; }
        QLabel#connection { color: #66d9a6; font-size: 14px; }
        QLabel#riskLevel { font-size: 52px; font-weight: 800; background: transparent; }
        QLabel#guide { font-size: 17px; background: transparent; }
        QFrame#indicatorCard { background: #1f2937; border-radius: 12px; }
        QFrame#indicatorCard[active="true"] { border: 2px solid #f59e0b; }
        QLabel#cardTitle { color: #aeb9ca; font-size: 14px; }
        QLabel#cardValue { font-size: 22px; font-weight: 700; }
        QFrame#reasonFrame { background: #1f2937; border-radius: 12px; }
        QLabel#sectionTitle { color: #aeb9ca; font-size: 14px; }
        QLabel#reason { font-size: 18px; }
        QLabel#muted { color: #94a3b8; font-size: 14px; }
        QPushButton#ackButton { background: #f8fafc; color: #111827; border: 0;
            border-radius: 10px; padding: 12px 24px; font-weight: 700; }
        QPushButton#ackButton:disabled { background: #475569; color: #94a3b8; }
    """
