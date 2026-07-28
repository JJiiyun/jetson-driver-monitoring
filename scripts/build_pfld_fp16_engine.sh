#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONNX_PATH="${1:-${PROJECT_ROOT}/models/landmark/pfld_sim.onnx}"
ENGINE_PATH="${2:-${PROJECT_ROOT}/models/engines/fp16/pfld_fp16.engine}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

if [[ ! -s "${ONNX_PATH}" ]]; then
    echo "[ERROR] ONNX model is missing or empty: ${ONNX_PATH}" >&2
    exit 1
fi
if [[ ! -x "${TRTEXEC}" ]]; then
    echo "[ERROR] trtexec is not executable: ${TRTEXEC}" >&2
    exit 1
fi

mkdir -p "$(dirname "${ENGINE_PATH}")"

"${TRTEXEC}" \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${ENGINE_PATH}" \
    --explicitBatch \
    --minShapes=input:1x3x112x112 \
    --optShapes=input:1x3x112x112 \
    --maxShapes=input:1x3x112x112 \
    --shapes=input:1x3x112x112 \
    --workspace=512 \
    --fp16

echo "[OK] TensorRT FP16 engine: ${ENGINE_PATH}"
