#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${HOME}/zzmenv"

echo "========================================"
echo " ZZM virtual environment setup"
echo "========================================"

echo
echo "[1/5] Checking Python"
python3 --version

echo
echo "[2/5] Checking venv module"
if ! python3 -c "import venv" 2>/dev/null; then
    echo "python3-venv is not installed."
    echo "Run: sudo apt install -y python3-venv"
    exit 1
fi

echo
echo "[3/5] Creating virtual environment"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv --system-site-packages "${VENV_DIR}"
    echo "Created: ${VENV_DIR}"
else
    echo "Already exists: ${VENV_DIR}"
fi

echo
echo "[4/5] Installing project packages"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade "pip<24" setuptools wheel
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo
echo "[5/5] Verifying required modules"
python - <<'PY'
import sys

print("Python executable:", sys.executable)

modules = [
    ("OpenCV", "cv2"),
    ("TensorRT", "tensorrt"),
    ("NumPy", "numpy"),
]

failed = False

for display_name, module_name in modules:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        path = getattr(module, "__file__", "built-in")

        print(f"[OK] {display_name}: {version}")
        print(f"     path: {path}")
    except Exception as exc:
        failed = True
        print(f"[FAIL] {display_name}: {exc}")

if failed:
    raise SystemExit(
        "Some Jetson system packages are unavailable. "
        "Check whether the environment was created with "
        "--system-site-packages."
    )
PY

echo
echo "Environment setup complete."
echo "Activate with:"
echo "source ${VENV_DIR}/bin/activate"
