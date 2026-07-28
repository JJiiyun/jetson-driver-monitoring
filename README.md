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

## PFLD TensorRT FP16

Generate the device-specific engine on the Jetson Nano:

```bash
bash scripts/build_pfld_fp16_engine.sh
```

Run camera inference with the TensorRT PFLD backend:

```bash
python3 scripts/run_eye_monitor.py --landmark-backend tensorrt
```

Run video inference with the same backend:

```bash
python3 scripts/run_video_inference.py INPUT.mp4 \
  --landmark-backend tensorrt
```

OpenCV remains the default backend. TensorRT uses
`models/engines/fp16/pfld_fp16.engine` unless `--pfld-engine` is supplied.

## OpenCV CUDA experiment

`run_experiment.sh` activates the project-local OpenCV 4.8 CUDA build and
runs both YuNet and ONNX PFLD with the OpenCV CUDA FP32 target.

```bash
bash scripts/run_experiment.sh data/final_test_640x360.mp4
```

## Camer Test
