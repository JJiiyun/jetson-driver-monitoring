from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from .actions import RiskEventPublisher
from .qt_bridge import create_qt_risk_bridge
from .risk_controller import DrowsinessRiskController, RiskDecision, RiskLevel


LEVEL_PRESENTATION: Final = {
    RiskLevel.NORMAL: {
        "label": "정상",
        "accent": "#2dd4bf",
        "panel_top": "#123238",
        "panel_bottom": "#0e242a",
        "guide": "졸음 위험이 감지되지 않았습니다. 안전 운전 중입니다.",
    },
    RiskLevel.PRE_DROWSY: {
        "label": "졸음 직전",
        "accent": "#fbbf24",
        "panel_top": "#3a2c12",
        "panel_bottom": "#241c0e",
        "guide": "주의가 필요합니다. 자세를 바로잡고 휴식을 준비하세요.",
    },
    RiskLevel.DROWSY: {
        "label": "졸음 위험",
        "accent": "#fb7185",
        "panel_top": "#401b28",
        "panel_bottom": "#281019",
        "guide": "긴급 경고 중입니다. 안전한 곳에 즉시 정차하세요.",
    },
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


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _mix(color_a: str, color_b: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    ar, ag, ab = _hex_to_rgb(color_a)
    br, bg, bb = _hex_to_rgb(color_b)
    red = round(ar + (br - ar) * ratio)
    green = round(ag + (bg - ag) * ratio)
    blue = round(ab + (bb - ab) * ratio)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _with_alpha(hex_color: str, alpha: float) -> str:
    red, green, blue = _hex_to_rgb(hex_color)
    alpha_value = max(0, min(255, round(alpha * 255)))
    return f"rgba({red}, {green}, {blue}, {alpha_value})"


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
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import (
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QGraphicsDropShadowEffect,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        try:
            from PyQt5.QtCore import Qt
            from PyQt5.QtGui import QColor
            from PyQt5.QtWidgets import (
                QFrame,
                QGridLayout,
                QHBoxLayout,
                QLabel,
                QMainWindow,
                QPushButton,
                QGraphicsDropShadowEffect,
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

    def create_shadow(
        *,
        blur_radius: int = 28,
        y_offset: int = 8,
        color: str = "#000000",
        alpha: int = 110,
    ) -> Any:
        shadow_color = QColor(color)
        shadow_color.setAlpha(max(0, min(255, alpha)))
        effect = QGraphicsDropShadowEffect()
        effect.setBlurRadius(blur_radius)
        effect.setOffset(0, y_offset)
        effect.setColor(shadow_color)
        return effect

    class IndicatorCard(QFrame):
        def __init__(self, title: str) -> None:
            super().__init__()
            self.setObjectName("indicatorCard")
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 14, 16, 16)
            top_row = QHBoxLayout()
            title_label = QLabel(title)
            title_label.setObjectName("cardTitle")
            self.status_label = QLabel("대기")
            self.status_label.setObjectName("cardStatus")
            self.value_label = QLabel("-")
            self.value_label.setObjectName("cardValue")
            self.value_label.setAlignment(align_center)
            top_row.addWidget(title_label)
            top_row.addStretch(1)
            top_row.addWidget(self.status_label)
            layout.addLayout(top_row)
            layout.addWidget(self.value_label, 1)

        def set_value(
            self,
            value: str,
            active: bool = False,
            danger: bool = False,
        ) -> None:
            self.value_label.setText(value)
            self.setProperty("active", active)
            self.setProperty("danger", danger)
            self.status_label.setText(
                "긴급" if danger else ("작동" if active else "대기")
            )
            self.status_label.setProperty("active", active)
            self.status_label.setProperty("danger", danger)
            self.style().unpolish(self)
            self.style().polish(self)
            self.status_label.style().unpolish(self.status_label)
            self.status_label.style().polish(self.status_label)
            self.update()

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
            root.setObjectName("centralWidget")
            self.setCentralWidget(root)
            outer = QVBoxLayout(root)
            outer.setContentsMargins(28, 24, 28, 24)
            outer.setSpacing(18)

            title_row = QHBoxLayout()
            title_column = QVBoxLayout()
            title = QLabel("운전자 졸음 안전 모니터")
            title.setObjectName("title")
            subtitle = QLabel("REAL-TIME DRIVER SAFETY SYSTEM")
            subtitle.setObjectName("subtitle")
            title_column.addWidget(title)
            title_column.addWidget(subtitle)
            self.connection_label = QLabel("● 실시간 판단 연결")
            self.connection_label.setObjectName("connection")
            title_row.addLayout(title_column)
            title_row.addStretch(1)
            title_row.addWidget(self.connection_label)
            outer.addLayout(title_row)

            self.status_panel = QFrame()
            self.status_panel.setObjectName("statusPanel")
            status_layout = QHBoxLayout(self.status_panel)
            status_text_layout = QVBoxLayout()
            status_caption = QLabel("CURRENT DRIVER STATUS")
            status_caption.setObjectName("statusCaption")
            self.level_label = QLabel("정상")
            self.level_label.setObjectName("riskLevel")
            self.level_label.setAlignment(align_center)
            self.guide_label = QLabel(
                LEVEL_PRESENTATION[RiskLevel.NORMAL]["guide"]
            )
            self.guide_label.setObjectName("guide")
            self.guide_label.setAlignment(align_center)
            self.guide_label.setWordWrap(True)
            self.status_icon = QLabel("✓")
            self.status_icon.setObjectName("statusIcon")
            self.status_icon.setAlignment(align_center)
            status_text_layout.addWidget(status_caption)
            status_text_layout.addWidget(self.level_label)
            status_text_layout.addWidget(self.guide_label)
            status_layout.addLayout(status_text_layout, 1)
            status_layout.addWidget(self.status_icon)
            outer.addWidget(self.status_panel)

            grid = QGridLayout()
            grid.setSpacing(14)
            self.yawn_card = IndicatorCard("최근 하품")
            self.buzzer_card = IndicatorCard("부저")
            self.hazard_card = IndicatorCard("비상등 요청")
            self.stop_card = IndicatorCard("차량 정차 요청")
            self.indicator_cards = (
                self.yawn_card,
                self.buzzer_card,
                self.hazard_card,
                self.stop_card,
            )
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
            self._status_shadow = create_shadow(
                blur_radius=38,
                y_offset=8,
                color="#2dd4bf",
                alpha=100,
            )
            self.status_panel.setGraphicsEffect(self._status_shadow)
            for card in self.indicator_cards:
                card.setGraphicsEffect(
                    create_shadow(
                        blur_radius=20,
                        y_offset=5,
                        color="#000000",
                        alpha=85,
                    )
                )
            self._set_level(RiskLevel.NORMAL)
            self.yawn_card.set_value("0회")
            self.buzzer_card.set_value("꺼짐")
            self.hazard_card.set_value("OFF")
            self.stop_card.set_value("없음")

        def apply_decision(self, decision: RiskDecision) -> None:
            self._last_decision = decision
            self._set_level(decision.level)
            self.reason_label.setText(reason_text(decision.reasons))
            self.yawn_card.set_value(
                f"{decision.recent_yawn_count}회",
                decision.recent_yawn_count >= 2,
            )
            self.buzzer_card.set_value(
                {"OFF": "꺼짐", "ALERT": "주의 경고", "EMERGENCY": "긴급 경고"}[
                    decision.buzzer_mode.value
                ],
                decision.buzzer_mode.value != "OFF",
                decision.buzzer_mode.value == "EMERGENCY",
            )
            self.hazard_card.set_value(
                "ON" if decision.hazard_light else "OFF",
                decision.hazard_light,
                decision.hazard_light,
            )
            self.stop_card.set_value(
                "요청 중" if decision.stop_request else "없음",
                decision.stop_request,
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
            presentation = LEVEL_PRESENTATION[level]
            label = presentation["label"]
            accent = presentation["accent"]
            panel_top = presentation["panel_top"]
            panel_bottom = presentation["panel_bottom"]
            guide = presentation["guide"]

            self.level_label.setText(label)
            self.guide_label.setText(guide)
            self.status_icon.setText(
                {
                    RiskLevel.NORMAL: "✓",
                    RiskLevel.PRE_DROWSY: "!",
                    RiskLevel.DROWSY: "!!",
                }[level]
            )
            self.status_icon.setStyleSheet(
                "QLabel#statusIcon { "
                f"background: {_mix(panel_bottom, accent, 0.14)}; "
                f"border: 1px solid {_mix(panel_bottom, accent, 0.52)}; "
                f"border-radius: 28px; color: {accent}; font-size: 24px; "
                "font-weight: 900; min-width: 56px; min-height: 56px; "
                "max-width: 56px; max-height: 56px; }"
            )
            self.status_panel.setStyleSheet(
                f"QFrame#statusPanel {{ border: 2px solid {accent}; "
                "border-radius: 22px; "
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
                f"stop:0 {panel_top}, stop:0.70 {panel_bottom}, "
                f"stop:1 {_mix(panel_bottom, accent, 0.13)}); padding: 20px; }}"
            )
            self.level_label.setStyleSheet(
                "QLabel#riskLevel { background: transparent; "
                f"color: {accent}; font-size: 56px; font-weight: 900; }}"
            )
            is_danger = level is RiskLevel.DROWSY
            self.ack_button.setProperty("danger", is_danger)
            self.ack_button.style().unpolish(self.ack_button)
            self.ack_button.style().polish(self.ack_button)
            self.ack_button.update()
            if hasattr(self, "_status_shadow"):
                shadow_color = QColor(accent)
                shadow_color.setAlpha(115)
                self._status_shadow.setColor(shadow_color)

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
        QWidget {
            background-color: #08111f; color: #e6edf7;
            font-family: "Pretendard", "Noto Sans KR", "Segoe UI", sans-serif;
            font-size: 15px;
        }
        QWidget#centralWidget {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #0b1628, stop:0.55 #091321, stop:1 #07101c);
        }
        QLabel#title {
            background: transparent; color: #f8fafc;
            font-size: 25px; font-weight: 800;
        }
        QLabel#subtitle {
            background: transparent; color: #8292aa;
            font-size: 13px; font-weight: 500;
        }
        QLabel#connection {
            background: rgba(45, 212, 191, 26); color: #5eead4;
            border: 1px solid rgba(45, 212, 191, 82);
            border-radius: 10px; padding: 5px 10px;
            font-size: 12px; font-weight: 700;
        }
        QLabel#connection[connected="false"] {
            background: rgba(251, 113, 133, 26); color: #fda4af;
            border: 1px solid rgba(251, 113, 133, 89);
        }
        QFrame#statusPanel { border-radius: 22px; }
        QLabel#statusCaption {
            background: transparent; color: rgba(255, 255, 255, 173);
            font-size: 12px; font-weight: 700;
        }
        QLabel#riskLevel {
            background: transparent; color: #ffffff;
            font-size: 56px; font-weight: 900;
        }
        QLabel#guide {
            background: transparent; color: rgba(255, 255, 255, 224);
            font-size: 16px; font-weight: 500;
        }
        QLabel#statusIcon {
            background: rgba(255, 255, 255, 20);
            border: 1px solid rgba(255, 255, 255, 31);
            border-radius: 28px; color: #ffffff;
            font-size: 27px; font-weight: 800;
            min-width: 56px; min-height: 56px;
            max-width: 56px; max-height: 56px;
        }
        QFrame#indicatorCard {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #14213a, stop:1 #101b30);
            border: 1px solid #233552; border-radius: 16px;
        }
        QFrame#indicatorCard:hover { border: 1px solid #365078; background: #162641; }
        QFrame#indicatorCard[active="true"] {
            border: 2px solid #fbbf24;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #2f2918, stop:1 #1d1c18);
        }
        QFrame#indicatorCard[danger="true"] {
            border: 2px solid #fb7185;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #321c29, stop:1 #21151d);
        }
        QLabel#cardTitle {
            background: transparent; color: #8fa2bf;
            font-size: 12px; font-weight: 700;
        }
        QLabel#cardValue {
            background: transparent; color: #f8fafc;
            font-size: 27px; font-weight: 800;
        }
        QLabel#cardStatus {
            background: rgba(148, 163, 184, 26); color: #a9b7cb;
            border-radius: 8px; padding: 3px 7px;
            font-size: 11px; font-weight: 700;
        }
        QLabel#cardStatus[active="true"] {
            background: rgba(251, 191, 36, 38); color: #fcd34d;
        }
        QLabel#cardStatus[danger="true"] {
            background: rgba(251, 113, 133, 38); color: #fda4af;
        }
        QFrame#reasonFrame {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #111f36, stop:1 #0e192c);
            border: 1px solid #22344f; border-radius: 16px;
        }
        QLabel#sectionTitle {
            background: transparent; color: #8093b0;
            font-size: 12px; font-weight: 800;
        }
        QLabel#reason {
            background: transparent; color: #e8eef7;
            font-size: 17px; font-weight: 600;
        }
        QLabel#muted { background: transparent; color: #60718b; font-size: 12px; }
        QPushButton#ackButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #f8fafc, stop:1 #dfe7f1);
            color: #0a1322; border: 1px solid #ffffff;
            border-radius: 13px; padding: 13px 28px;
            font-size: 15px; font-weight: 800;
        }
        QPushButton#ackButton:hover { background: #ffffff; border: 1px solid #dbeafe; }
        QPushButton#ackButton:pressed {
            background: #dbe4ee; padding-top: 15px; padding-bottom: 11px;
        }
        QPushButton#ackButton:disabled {
            background: #1b2941; color: #586981; border: 1px solid #273750;
        }
        QPushButton#ackButton[danger="true"] {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #fb7185, stop:1 #e11d48);
            color: #ffffff; border: 1px solid #fda4af;
        }
        QPushButton#ackButton[danger="true"]:hover { background: #fb4664; }
        QToolTip {
            background: #16233a; color: #f1f5f9;
            border: 1px solid #344867; border-radius: 6px;
            padding: 6px; font-size: 12px;
        }
    """
