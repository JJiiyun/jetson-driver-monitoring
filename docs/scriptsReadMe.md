# scripts 디렉터리

`scripts/`는 모델 엔진 생성부터 실시간 시연, 영상 평가, 순수 모델 및 전력
측정까지의 실행 진입점을 모아 둔 디렉터리입니다.

## 주요 실행 파일

| 파일 | 역할 |
|---|---|
| `run_eye_monitor.py` | 카메라 실시간 전체 시스템, Qt·부저 포함 |
| `run_video_inference_FSM.py` | 저장 영상으로 눈 FSM과 종합 위험 FSM 실행 |
| `run_video_inference.py` | 공통 영상 추론 구현 |
| `run_experiment.sh` | 영상 추론 → 라벨 채점 → metrics 생성 자동화 |
| `benchmark_model_backends.py` | YuNet+PFLD 순수 모델 성능 비교 |
| `run_power_backend_benchmark.sh` | 모델 성능과 tegrastats/INA3221 동시 수집 |
| `collect_tegrastats.py` | GPU·CPU·RAM·온도·전력 CSV 수집/요약 |
| `build_*_engine.sh` | YuNet/PFLD TensorRT FP32/FP16 엔진 생성 |
| `resize_video_640x360.sh` | 비교용 640×360, 30 FPS 영상 생성 |
| `use_opencv_cuda.sh` | 프로젝트 로컬 OpenCV CUDA 경로 설정 |
| `check_system.sh` | JetPack/CUDA/cuDNN/TensorRT/OpenCV 확인 |

## 파라미터 실험

```bash
bash scripts/run_experiment.sh data/final_test_0727_14_30.mp4 0.72 0.85 1.7
```

인자는 차례로 `closed_ratio`, `reopen_ratio`, `danger_seconds`입니다. 스크립트는
가상환경과 OpenCV CUDA 빌드를 로드하고 최신 frames/summary CSV를 찾아
`benchmark/evaluate.py`와 `benchmark/compute_metrics.py`로 평가합니다.

## 순수 백엔드 비교

```bash
python3 scripts/benchmark_model_backends.py data/final_test_640x360.mp4 \
  --backend opencv-cuda-fp32
python3 scripts/benchmark_model_backends.py data/final_test_640x360.mp4 \
  --backend tensorrt-fp32
python3 scripts/benchmark_model_backends.py data/final_test_640x360.mp4 \
  --backend tensorrt-fp16
```

이 측정에는 영상 decode, FSM, EAR/MAR, overlay, display, video encoding이 포함되지
않습니다. YuNet 입력은 세 방식 모두 640×640 letterbox, PFLD는 112×112로
고정됩니다.

## 전력 비교

각 백엔드를 3회 실행하고, 실행 사이에는 GPU 온도가 시작 조건으로 돌아올 때까지
충분히 쉽니다.

```bash
bash scripts/run_power_opencv_cuda_fp32.sh data/final_test_640x360.mp4
bash scripts/run_power_tensorrt_fp32.sh data/final_test_640x360.mp4
bash scripts/run_power_tensorrt_fp16.sh data/final_test_640x360.mp4
```

산출물은 `/srv/samba/pure_backend_<backend>_runN/`에 저장됩니다. 각 폴더에는
프레임 성능 CSV, 모델 요약 CSV, tegrastats CSV, 전력 요약 CSV, 실행 로그가
있습니다. 중단·exit 139 실행은 전체 프레임과 model summary 존재 여부를 확인한
후 유효 반복에서 제외합니다.

## TensorRT 엔진 생성

```bash
bash scripts/build_yunet_fp32_engine.sh
bash scripts/build_yunet_fp16_engine.sh
bash scripts/build_pfld_fp16_engine.sh
```

PFLD FP32 엔진은 `trtexec`로 `1x3x112x112` 고정 shape를 지정해 생성해야 합니다.
엔진은 대상 Jetson에서 생성하고 다른 JetPack 장치의 엔진을 그대로 복사하지
않는 것이 안전합니다.

## 실시간 옵션 핵심

- `--face-backend`: `opencv-fp32`, `tensorrt-fp32`, `tensorrt-fp16`
- `--landmark-backend`: 위와 동일
- `--opencv-device`: OpenCV backend일 때 `cpu` 또는 `cuda`
- `--qt-dashboard`: Qt 위험 화면 표시
- `--buzzer-pin`, `--gpio-numbering`: 물리 부저 출력
- `--passive-buzzer-frequency`: 수동 피에조 소프트웨어 톤 주파수
- `a`: DROWSY 확인(acknowledge), 이후 안전 조건이 연속 5초 유지되어야 복귀
- `r`: 보정/FSM 초기화, `q`: 종료
