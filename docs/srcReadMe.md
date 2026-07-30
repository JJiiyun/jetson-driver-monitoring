# src 디렉터리

`src/`는 프로젝트 초기 프로토타입과 기능별 실험 코드가 남아 있는 영역입니다.
`drowsiness_pipeline.py`, `ear.py`, `perclos.py`에서 초기 OpenCV DNN 기반 흐름을
확인할 수 있습니다.

현재 실시간 시연과 TensorRT 백엔드는 `scripts/run_eye_monitor.py`와
`drowsiness/` 패키지가 기준입니다. 기능을 수정할 때 `src/`만 바꾸면 현재
파이프라인에는 반영되지 않을 수 있으므로 새 코드는 가능한 `drowsiness/`의 독립
모듈로 구현하고 실행 스크립트에서 연결합니다.
