# models 디렉터리

## 필요한 파일 배치

```text
models/
├── face_detector/yunet.onnx
├── landmark/pfld_sim.onnx
└── engines/
    ├── fp32/yunet_fp32.engine
    ├── fp32/pfld_fp32.engine
    ├── fp16/yunet_fp16.engine
    └── fp16/pfld_fp16.engine
```

YuNet은 얼굴 box를 찾고 PFLD는 잘라낸 얼굴에서 68개 landmark를 예측합니다.
눈 landmark로 EAR, 입 landmark로 MAR를 계산합니다.

## 백엔드 선택

| 설정 | YuNet | PFLD |
|---|---|---|
| `opencv-fp32 --opencv-device cuda` | ONNX/OpenCV CUDA FP32 | ONNX/OpenCV CUDA FP32 |
| `tensorrt-fp32` | TensorRT FP32 engine | TensorRT FP32 engine |
| `tensorrt-fp16` | TensorRT FP16 engine | TensorRT FP16 engine |

FP16은 16비트 부동소수점으로 메모리 대역폭과 TensorRT 최적화 기회를 늘립니다.
다만 실제 향상은 모델 구조, 전처리, 메모리 복사에 따라 달라지므로 이 프로젝트는
정확도와 순수 모델 지연을 함께 검증합니다.

## Git과 호환성

ONNX, engine, weight 파일과 `models/` 내용은 `.gitignore` 대상입니다. 새 장치에는
별도로 모델을 전달해야 합니다. TensorRT engine은 GPU, TensorRT, CUDA 버전에
종속될 수 있으므로 ONNX를 전달하고 대상 Jetson에서 엔진을 다시 만드는 것이
가장 재현성이 높습니다. `models/source/`와 `models/opencv_zoo/`는 참고 소스이며
실행에 필요한 핵심 파일은 위 네 engine과 두 ONNX입니다.
