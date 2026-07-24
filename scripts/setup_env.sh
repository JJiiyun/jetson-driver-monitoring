#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
REQUIREMENTS_FILE="${PROJECT_DIR}/requirements.txt"

echo "========================================"
echo " ZZM environment setup"
echo "========================================"
echo "Project: ${PROJECT_DIR}"
echo "Virtual environment: ${VENV_DIR}"

echo
echo "[1/6] Checking Python"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 is not installed."
    exit 1
fi

python3 --version

echo
echo "[2/6] Checking venv support"

if ! python3 -c "import venv" >/dev/null 2>&1; then
    echo "[ERROR] Python venv module is not installed."
    echo "Run:"
    echo "sudo apt update"
    echo "sudo apt install -y python3-venv"
    exit 1
fi

echo
echo "[3/6] Creating virtual environment"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    rm -rf "${VENV_DIR}"
    python3 -m venv --system-site-packages "${VENV_DIR}"
    echo "[OK] Created ${VENV_DIR}"
else
    echo "[OK] Existing environment found"
fi

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "[ERROR] Virtual environment creation failed."
    exit 1
fi

echo
echo "[4/6] Activating virtual environment"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "Python executable: $(which python)"
python --version

echo
echo "[5/6] Installing project packages"

python -m pip install --upgrade "pip<24" setuptools wheel

if [ -f "${REQUIREMENTS_FILE}" ]; then
    python -m pip install -r "${REQUIREMENTS_FILE}"
else
    echo "[WARNING] requirements.txt not found."
fi

echo
echo "[6/6] Verifying required modules"

python - <<'PY'
import sys

print("Python executable:", sys.executable)

required_modules = [
    ("NumPy", "numpy"),
    ("OpenCV", "cv2"),
    ("TensorRT", "tensorrt"),
]

failed = False

for display_name, module_name in required_modules:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        path = getattr(module, "__file__", "built-in")

        print(f"[OK] {display_name}: {version}")
        print(f"     {path}")
    except Exception as exc:
        failed = True
        print(f"[FAIL] {display_name}: {exc}")

if failed:
    raise SystemExit(
        "\nRequired Jetson system libraries are missing.\n"
        "Check CUDA, TensorRT, OpenCV and JetPack/L4T installation."
    )
PY

echo
echo "========================================"
echo " Setup complete"
echo "========================================"
echo "Activate the environment with:"
echo "source ${VENV_DIR}/bin/activate"