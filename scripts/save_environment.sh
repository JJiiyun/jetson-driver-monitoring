#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_FILE="${PROJECT_DIR}/docs/environment.txt"

mkdir -p "${PROJECT_DIR}/docs"

{
    echo "===== ZZM Jetson Nano Environment ====="
    echo
    echo "Python: $(python3 --version 2>&1)"

    python3 - <<'PY'
import cv2
import tensorrt as trt
import numpy as np

print("OpenCV:", cv2.__version__)
print("TensorRT:", trt.__version__)
print("NumPy:", np.__version__)
PY

    echo
    nvcc --version | grep "release" || true

    echo
    cat /etc/nv_tegra_release

    echo
    grep PRETTY_NAME /etc/os-release
} | tee "${OUTPUT_FILE}"

echo
echo "Saved environment information to:"
echo "${OUTPUT_FILE}"
