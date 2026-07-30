#!/usr/bin/env bash

# Source this file to use the project-local OpenCV 4.8 CUDA build.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENCV_ROOT="${PROJECT_ROOT}/third_party/opencv-cuda-4.8.0"
OPENCV_PYTHON="${OPENCV_ROOT}/python"

if [[ ! -f "${OPENCV_PYTHON}/cv2/python-3.8/cv2.cpython-38-aarch64-linux-gnu.so" ]]; then
    echo "[ERROR] OpenCV CUDA build is not installed: ${OPENCV_ROOT}" >&2
    return 1 2>/dev/null || exit 1
fi

export PYTHONPATH="${OPENCV_PYTHON}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${OPENCV_ROOT}/lib:${LD_LIBRARY_PATH:-}"
