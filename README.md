# ZZM

Jetson Nano-based real-time driver drowsiness monitoring and TensorRT benchmarking project.

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

## Camer Test
