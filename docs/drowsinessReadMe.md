# drowsiness 디렉터리

현재 제품 파이프라인의 핵심 모듈입니다.

| 파일 | 책임 |
|---|---|
| `detectors.py` | OpenCV/TensorRT YuNet 및 PFLD 추론 |
| `metrics.py` | 68 landmarks에서 EAR와 표시 좌표 계산 |
| `calibration.py` | 개인별 열린 눈 보정 및 눈 감김 FSM |
| `yawn_monitor.py` | MAR 기반 하품 FSM |
| `perclos_monitor.py` | 시간 창 내 눈 감김 비율 계산 |
| `risk_controller.py` | 세 신호를 행동 단계로 합치는 종합 위험 FSM |
| `actions.py` | 이벤트 배포, GPIO, 수동 피에조, 부저 패턴 |
| `qt_dashboard.py`, `qt_bridge.py` | Qt 표시와 스레드 안전 이벤트 연결 |
| `overlay.py` | OpenCV 영상 상태 오버레이 |

## 눈 감김 FSM

초기 보정 시간 동안 열린 눈 EAR 중앙값을 baseline으로 잡습니다.

```text
relative EAR = current EAR / baseline EAR
relative EAR <= closed_ratio  -> CLOSED
relative EAR >= reopen_ratio  -> OPEN
CLOSED가 danger_seconds 이상 -> DANGER
```

두 임계값 사이에서는 직전 상태를 유지하는 hysteresis를 적용해 경계 잡음을
억제합니다. 현재 채택 실험값은 `0.72 / 0.85 / 1.7초`입니다.

## 하품 FSM

입의 세로 길이를 가로 길이로 나눈 MAR를 사용합니다. `open_ratio` 이상이
`yawn_seconds` 동안 지속되면 하품으로 확정하고, `close_ratio` 아래로 내려가야
다음 하품을 셉니다. 기본 실험값은 `0.18 / 0.14 / 0.3초`입니다.

## 종합 위험 FSM

```text
NORMAL
  | 반복 하품(60초 내 2회) 또는 PERCLOS caution
  v
PRE_DROWSY --------------------> buzzer ALERT
  | 연속 눈 감김 DANGER
  | 또는 PERCLOS warning + 최근 하품
  v
DROWSY ------------------------> buzzer EMERGENCY
                                  hazard_light=True
                                  stop_request=True
```

`DROWSY`는 latch됩니다. 위험 신호가 사라지는 것만으로는 풀리지 않습니다.
사용자가 `a`로 acknowledge하고, 얼굴이 유효하며 PRE_DROWSY 조건도 없는 상태가
연속 5초 유지되어야 `NORMAL`로 돌아갑니다. 따라서 실제 시연 대기 시간은
“눈을 뜬 뒤 5초”가 아니라 “acknowledge 이후 모든 안전 조건이 만족된 시점부터
연속 5초”입니다.

## 확장 방식

`RiskEventPublisher`에 새 sink의 `publish` 메서드를 subscribe하면 GPIO 부저
외에도 CAN, MQTT, 로깅, 차량 제어기 어댑터를 추가할 수 있습니다. 종합 FSM은
요청만 생성하고 실제 정차 정책은 별도 제어 계층에서 구현합니다.
