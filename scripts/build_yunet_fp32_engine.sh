#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_PATH="${1:-$PROJECT_ROOT/models/face_detector/yunet.onnx}"
ENGINE_PATH="${2:-$PROJECT_ROOT/models/engines/fp32/yunet_fp32.engine}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

if [ ! -s "$ONNX_PATH" ]; then
    echo "[ERROR] YuNet ONNX model is missing: $ONNX_PATH" >&2
    exit 1
fi
if [ ! -x "$TRTEXEC" ]; then
    echo "[ERROR] trtexec is not executable: $TRTEXEC" >&2
    exit 1
fi

mkdir -p "$(dirname "$ENGINE_PATH")"
"$TRTEXEC" \
    --onnx="$ONNX_PATH" \
    --saveEngine="$ENGINE_PATH" \
    --workspace=64 \
    --minTiming=1 \
    --avgTiming=1 \
    --buildOnly

echo "[OK] TensorRT YuNet FP32 engine: $ENGINE_PATH"
