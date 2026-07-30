# 졸음 위험 판단과 출력 장치 연동

## 구조

```text
YuNet + PFLD
  ├─ EAR 눈 감김 FSM
  ├─ PERCLOS
  └─ MAR 하품 감지
          │
          ▼
DrowsinessRiskController
  ├─ NORMAL      → 출력 없음
  ├─ PRE_DROWSY  → ALERT 부저 요청
  └─ DROWSY      → EMERGENCY 부저 + 비상등 + 정차 요청
          │
          ▼
RiskEventPublisher
  ├─ Qt signal bridge
  ├─ Jetson GPIO buzzer
  └─ 향후 비상등/차량 제어 어댑터
```

판단 로직은 Qt와 GPIO를 직접 알지 못한다. 따라서 GUI 프레임워크나 실제
출력 장치가 바뀌어도 `DrowsinessRiskController`를 재사용할 수 있다.

현재 기본 정책은 다음과 같다.

- 60초 이내 하품 2회 또는 PERCLOS 주의: `PRE_DROWSY`
- 눈 감김 FSM의 위험 상태: `DROWSY`
- PERCLOS 경고와 최근 하품이 함께 발생: `DROWSY`
- `PRE_DROWSY`: 경고 부저만 요청
- `DROWSY`: 긴급 부저, 비상등 ON, 제어된 정차를 요청

`DROWSY`는 순간적으로 얼굴이 사라지거나 눈을 떴다고 바로 해제되지 않는다.
운전자 승인 후 위험 신호가 없는 상태가 5초 유지되어야 해제된다. 실시간
모니터에서는 `a` 키가 승인 입력이며, Qt에서는 승인 버튼을
`risk_controller.acknowledge()`에 연결하면 된다.

`stop_request`는 물리 버튼이 아니라 차량 제어 계층에 전달할 소프트웨어
요청 값이다. 현재 코드는 모터나 브레이크를 직접 제어하지 않는다.

## Qt 연동

PySide6를 우선 사용하고, 없으면 PyQt5를 사용하는 선택적 브리지를 제공한다.
Qt를 사용하지 않는 실행에는 두 패키지가 필요하지 않다.

```python
from drowsiness.actions import RiskEventPublisher
from drowsiness.qt_bridge import create_qt_risk_bridge

publisher = RiskEventPublisher()
bridge = create_qt_risk_bridge(parent=window)
publisher.subscribe(bridge)

bridge.risk_level_changed.connect(window.update_risk_label)
bridge.actions_changed.connect(window.update_action_indicators)

# 승인 버튼은 판단 컨트롤러에 연결
window.ack_button.clicked.connect(risk_controller.acknowledge)
```

`actions_changed`는 `(buzzer_mode, hazard_light, stop_request)`를 내보낸다.
실제 차량 제어는 별도의 검증된 어댑터가 이 신호를 받아 처리해야 한다.

완성된 대시보드는 `drowsiness/qt_dashboard.py`의
`create_risk_dashboard()`로 생성한다. 화면에는 최종 위험 단계, 판단 근거,
최근 하품 횟수, 부저, 비상등 요청, 정차 요청과 운전자 확인 버튼이 표시된다.

Qt가 설치된 환경에서는 다음 미리보기로 세 단계 전환을 확인할 수 있다.

```bash
python scripts/run_risk_dashboard_demo.py
```

실제 추론 코드에서는 컨트롤러와 publisher를 생성한 뒤 화면에 같은 객체를
전달한다.

```python
controller = DrowsinessRiskController()
publisher = RiskEventPublisher()
dashboard = create_risk_dashboard(controller, publisher)
dashboard.show()

# 추론 루프에서
decision = controller.update(...)
publisher.publish(decision)
```

현재 실행 스크립트에서는 옵션 하나로 연결된다.

```bash
python scripts/run_eye_monitor.py --qt-dashboard

python scripts/run_video_inference_FSM.py \
  data/final_test_640x360.mp4 --qt-dashboard
```

## Jetson GPIO 부저

GPIO를 사용하지 않을 때는 기존 명령 그대로 실행한다. 핀을 명시한 경우에만
`Jetson.GPIO`를 불러오고 해당 핀을 출력으로 설정한다.

```bash
python scripts/run_video_inference.py \
  --input data/final_test_640x360.mp4 \
  --buzzer-pin 12 \
  --gpio-numbering BOARD
```

LOW 신호로 켜지는 부저 모듈이면 `--buzzer-active-low`도 추가한다. 위의 12번
핀은 사용 예시일 뿐이다. Jetson 모델별 핀맵과 다른 기능이 할당되어 있지
않은지 확인한 뒤 실제 핀을 선택해야 한다.

`run_experiment.sh`에서는 환경변수로 같은 옵션을 전달할 수 있다.

```bash
BUZZER_PIN=12 GPIO_NUMBERING=BOARD \
  bash scripts/run_experiment.sh data/final_test_640x360.mp4 0.72 0.85 1.7
```

active-low 모듈은 `BUZZER_ACTIVE_LOW=1`을 함께 지정한다.

GPIO 핀으로 소비 전류가 큰 부저를 직접 구동하지 않는다. 일반적인 연결은
Jetson GPIO → 저항 → 트랜지스터/MOSFET 제어단, 부저 전원 → 적절한 외부
전원, Jetson GND ↔ 외부 전원 GND 공통 구성이다. 코일형 부저라면 역기전력
보호 다이오드도 필요하다. 정확한 소자와 저항값은 부저의 정격 전압·전류에
맞춰 정해야 한다.

이 코드는 프로토타입 판단 및 요청 계층이다. 실제 차량의 비상등, 구동 모터,
브레이크에 연결하기 전에는 독립적인 안전 인터록, 수동 우선 제어, 통신 실패
시 안전 동작, 정차 가능 조건을 차량 제어 계층에서 반드시 검증해야 한다.
