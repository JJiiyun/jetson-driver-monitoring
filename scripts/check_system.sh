#!/usr/bin/env bash

set -u

echo "========================================"
echo " ZZM Jetson Nano system check"
echo "========================================"

echo
echo "===== Board / L4T ====="
cat /etc/nv_tegra_release 2>/dev/null \
    || echo "L4T information not found"

echo
echo "===== Operating System ====="
grep -E "PRETTY_NAME|VERSION_ID" /etc/os-release 2>/dev/null \
    || echo "OS information not found"

echo
echo "===== Python ====="
python3 --version

echo
echo "===== CUDA ====="
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "nvcc not found"
fi

echo
echo "===== cuDNN ====="
dpkg -l | grep -E "libcudnn" \
    || echo "cuDNN package not found"

echo
echo "===== TensorRT packages ====="
dpkg -l | grep -E "tensorrt|nvinfer" \
    || echo "TensorRT package not found"

echo
echo "===== Python modules ====="
python3 - <<'PY'
modules = [
    ("OpenCV", "cv2"),
    ("TensorRT", "tensorrt"),
    ("NumPy", "numpy"),
]

for display_name, module_name in modules:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"{display_name}: {version}")
    except Exception as exc:
        print(f"{display_name}: import failed ({exc})")
PY

echo
echo "===== trtexec ====="
TRTEXEC_PATH="$(find /usr -type f -name trtexec 2>/dev/null | head -n 1)"

if [ -n "${TRTEXEC_PATH}" ]; then
    echo "${TRTEXEC_PATH}"
else
    echo "trtexec not found"
fi

echo
echo "===== tegrastats ====="
if command -v tegrastats >/dev/null 2>&1; then
    command -v tegrastats
else
    echo "tegrastats not found"
fi

echo
echo "===== Camera devices ====="
if compgen -G "/dev/video*" >/dev/null; then
    ls -l /dev/video*
else
    echo "No video device found"
fi
