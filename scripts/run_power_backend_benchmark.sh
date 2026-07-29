#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "사용법: $0 <backend> [video]"
    exit 1
fi

BACKEND="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO="${2:-$PROJECT_ROOT/data/converted/final_test.mp4}"
PYTHON="$PROJECT_ROOT/zzmvenv/bin/python"
BENCHMARK="$PROJECT_ROOT/scripts/benchmark_model_backends.py"
COLLECTOR="$PROJECT_ROOT/scripts/collect_tegrastats.py"
SAMBA_ROOT="/srv/samba"
BACKEND_TAG="${BACKEND//-/_}"
RUN_NUMBER=1
while [ -e "$SAMBA_ROOT/pure_backend_${BACKEND_TAG}_run${RUN_NUMBER}" ]; do
    RUN_NUMBER=$((RUN_NUMBER + 1))
done
OUTPUT_DIR="$SAMBA_ROOT/pure_backend_${BACKEND_TAG}_run${RUN_NUMBER}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STEM="power_${BACKEND_TAG}_run${RUN_NUMBER}_${TIMESTAMP}"
TEGRA_CSV="$OUTPUT_DIR/${STEM}_tegrastats.csv"
TEGRA_SUMMARY="$OUTPUT_DIR/${STEM}_tegrastats_summary.csv"
RUN_LOG="$OUTPUT_DIR/${STEM}_run.log"
COLLECTOR_PID=""
COLLECTOR_WRAPPER_PID=""
COLLECTOR_PID_FILE="$OUTPUT_DIR/.collector.pid"

case "$BACKEND" in
    opencv-cuda-fp32|tensorrt-fp32|tensorrt-fp16)
        ;;
    *)
        echo "[ERROR] 지원하지 않는 backend: $BACKEND"
        exit 1
        ;;
esac

if [ ! -x "$PYTHON" ]; then
    echo "[ERROR] Python 가상환경이 없습니다: $PYTHON"
    exit 1
fi
if [ ! -f "$VIDEO" ]; then
    echo "[ERROR] 입력 영상이 없습니다: $VIDEO"
    exit 1
fi

cleanup() {
    if [ -n "$COLLECTOR_PID" ] && sudo kill -0 "$COLLECTOR_PID" 2>/dev/null; then
        sudo kill -TERM "$COLLECTOR_PID" 2>/dev/null || true
        for _ in 1 2 3 4 5 6; do
            if ! sudo kill -0 "$COLLECTOR_PID" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        if sudo kill -0 "$COLLECTOR_PID" 2>/dev/null; then
            sudo kill -KILL "$COLLECTOR_PID" 2>/dev/null || true
        fi
    fi
    if [ -n "$COLLECTOR_WRAPPER_PID" ]; then
        wait "$COLLECTOR_WRAPPER_PID" 2>/dev/null || true
    fi
    sudo rm -f "$COLLECTOR_PID_FILE" 2>/dev/null || true
    if [ -f "$TEGRA_CSV" ]; then
        sudo chown "$(id -u):$(id -g)" "$TEGRA_CSV" 2>/dev/null || true
    fi
    if [ -f "$TEGRA_SUMMARY" ]; then
        sudo chown "$(id -u):$(id -g)" "$TEGRA_SUMMARY" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "=========================================="
echo "Pure model power benchmark"
echo "  backend: $BACKEND"
echo "  video: $VIDEO"
echo "  warmup: 30 frames"
echo "  tegrastats interval: 500 ms"
echo "  run number: $RUN_NUMBER"
echo "  output: $OUTPUT_DIR"
echo "=========================================="

echo "[1/3] Jetson MAXN 및 클럭 고정"
sudo -v
sudo nvpmodel -m 0
sudo jetson_clocks
sudo mkdir -p "$OUTPUT_DIR"
sudo chown "$(id -u):$(id -g)" "$OUTPUT_DIR"

echo "[2/3] tegrastats + INA3221 수집 시작"
sudo -E sh -c '
    echo "$$" > "$1"
    exec "$2" "$3" \
        --interval 500 \
        --out "$4" \
        --summary-out "$5"
' sh "$COLLECTOR_PID_FILE" "$PYTHON" "$COLLECTOR" \
    "$TEGRA_CSV" "$TEGRA_SUMMARY" &
COLLECTOR_WRAPPER_PID=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if [ -s "$COLLECTOR_PID_FILE" ]; then
        break
    fi
    sleep 0.2
done
if [ ! -s "$COLLECTOR_PID_FILE" ]; then
    echo "[ERROR] 전력 수집기 PID를 확인할 수 없습니다."
    exit 1
fi
COLLECTOR_PID="$(cat "$COLLECTOR_PID_FILE")"
sleep 2

echo "[3/3] 모델 벤치마크 시작"
set +e
"$PYTHON" -u "$BENCHMARK" "$VIDEO" \
    --backend "$BACKEND" \
    --warmup-frames 30 \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$RUN_LOG"
BENCHMARK_STATUS=${PIPESTATUS[0]}
set -e

cleanup
COLLECTOR_PID=""
COLLECTOR_WRAPPER_PID=""
trap - EXIT INT TERM

if [ "$BENCHMARK_STATUS" -ne 0 ]; then
    echo "[ERROR] 벤치마크 실패: exit $BENCHMARK_STATUS"
    exit "$BENCHMARK_STATUS"
fi

echo "=========================================="
echo "완료"
echo "  실행 번호: $RUN_NUMBER"
echo "  전체 산출물 폴더: $OUTPUT_DIR"
echo "  실행 로그: $RUN_LOG"
echo "  전력/온도 프레임 CSV: $TEGRA_CSV"
echo "  전력/온도 요약 CSV: $TEGRA_SUMMARY"
echo "=========================================="
