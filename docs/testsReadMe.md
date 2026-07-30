# tests 디렉터리

| 파일 | 검증 내용 |
|---|---|
| `test_eye_monitor.py` | EAR와 눈 감김 보정/FSM |
| `test_risk_controller.py` | 종합 위험 전이, latch, 부저 패턴 |
| `test_performance_logger.py` | frames/summary CSV와 지표 계산 |
| `test_camera.py` | Jetson 카메라 열기 |
| `test_yunet.py` | YuNet 모델 로딩과 얼굴 검출 |

하드웨어가 필요 없는 단위 테스트:

```bash
source zzmvenv/bin/activate
python3 -m unittest \
  tests.test_eye_monitor \
  tests.test_risk_controller \
  tests.test_performance_logger
```

카메라·YuNet 테스트는 모델 파일과 GUI/카메라가 연결된 Jetson에서 별도로
실행합니다. TensorRT 변경 후에는 단위 테스트 외에도 5초 샘플, 전체 검증 영상,
실시간 카메라 순서로 확인합니다.
