# ZZM Driver Monitoring

Jetson Nano에서 YuNet 얼굴 검출, PFLD 68개 랜드마크, 눈 감김·하품·
PERCLOS를 결합해 졸음 위험을 판단하고 Qt·부저·차량 제어 요청으로 전달하는
실시간 운전자 모니터링 프로젝트입니다. OpenCV CUDA FP32와 TensorRT
FP32/FP16의 순수 모델 성능 및 전력도 같은 조건에서 비교합니다.

전체 구조, 실험 결과 표와 그래프, 재현 명령은
[`docs/ProjectReadMe.md`](docs/ProjectReadMe.md)에서 시작하세요.

## Environment

- NVIDIA Jetson Nano
- L4T R32.6.1
- Ubuntu 20.04.6 LTS
- Python 3.8.10
- CUDA 10.2
- cuDNN 8.2.1
- TensorRT 8.0.1.6
- OpenCV 5.0.0
- NumPy 1.24.4
- PyCUDA 2022.1

## Project Structure

| 경로 | 역할 | 상세 문서 |
|---|---|---|
| `drowsiness/` | 검출기, 눈·하품·PERCLOS·종합 위험 FSM, Qt, 부저 | [`docs/drowsinessReadMe.md`](docs/drowsinessReadMe.md) |
| `scripts/` | 실시간 실행, 영상 실험, 엔진 생성, 성능·전력 측정 | [`docs/scriptsReadMe.md`](docs/scriptsReadMe.md) |
| `benchmark/` | 프레임 로그, 라벨 채점, 성능 지표 계산 | [`docs/benchmarkReadMe.md`](docs/benchmarkReadMe.md) |
| `models/` | YuNet/PFLD ONNX와 Jetson 전용 TensorRT 엔진 | [`docs/modelsReadMe.md`](docs/modelsReadMe.md) |
| `data/` | 원본·압축 실험 영상과 정답 라벨 | [`docs/dataReadMe.md`](docs/dataReadMe.md) |
| `outputs/`, `logs/` | 오버레이 영상, 화면 캡처, 런타임 로그 | [`docs/outputsReadMe.md`](docs/outputsReadMe.md) |
| `src/` | 초기 프로토타입 파이프라인 | [`docs/srcReadMe.md`](docs/srcReadMe.md) |
| `tests/` | FSM·성능 로거·카메라 테스트 | [`docs/testsReadMe.md`](docs/testsReadMe.md) |

## Setup

```bash
sudo apt install -y libboost-python-dev libboost-thread-dev
bash scripts/setup_env.sh
source zzmvenv/bin/activate
```

## PFLD inference backends

Generate the device-specific engine on the Jetson Nano:

```bash
bash scripts/build_pfld_fp16_engine.sh
```

Choose OpenCV FP32, TensorRT FP32, or TensorRT FP16 explicitly:

```bash
python3 scripts/run_eye_monitor.py --landmark-backend opencv-fp32
python3 scripts/run_eye_monitor.py --landmark-backend tensorrt-fp32
python3 scripts/run_eye_monitor.py --landmark-backend tensorrt-fp16
```

The same choices work for video inference and the Qt dashboard:

```bash
python3 scripts/run_video_inference_FSM.py INPUT.mp4 \
  --landmark-backend tensorrt-fp16 \
  --qt-dashboard
```

OpenCV FP32 remains the default. TensorRT FP32 automatically uses
`models/engines/fp32/pfld_fp32.engine`, while TensorRT FP16 uses
`models/engines/fp16/pfld_fp16.engine`. `--pfld-engine` overrides the
automatically selected path. The old `opencv` and `tensorrt` names remain
compatible aliases for `opencv-fp32` and `tensorrt-fp16`.

## OpenCV CUDA experiment

`run_experiment.sh` activates the project-local OpenCV 4.8 CUDA build and
runs both YuNet and ONNX PFLD with the OpenCV CUDA FP32 target.

```bash
bash scripts/run_experiment.sh data/final_test_640x360.mp4
```

Select another backend or enable the Qt dashboard with environment variables:

```bash
LANDMARK_BACKEND=tensorrt-fp32 QT_DASHBOARD=1 \
  bash scripts/run_experiment.sh data/final_test_640x360.mp4
```

## Camera Test

```bash
python3 tests/test_camera.py
```
