#!/bin/bash
# 실험 자동화: 추론 -> 채점 -> 지표 리포트까지 한 번에
#
# 사용법:
#   bash run_experiment.sh <영상경로> [backend] \
#       [closed] [reopen] [danger] [yawn_open] [yawn_close] [yawn_seconds]
#
# backend:
#   opencv-fp32 | tensorrt-fp32 | tensorrt-fp16
#
# 예시:
#   bash run_experiment.sh ../data/converted/final_test.mp4 tensorrt-fp32
#   bash run_experiment.sh ../data/converted/final_test.mp4 tensorrt-fp16 \
#       0.72 0.85 1.7 0.18 0.14 0.3
#
# 파라미터를 생략하면 다음 기본값을 사용한다.
#   눈 감김: closed=0.72, reopen=0.85, danger=1.7초
#   하품:    open=0.18, close=0.14, duration=0.3초

set -o pipefail

VIDEO="${1:-}"
BACKEND="${2:-tensorrt-fp16}"
CLOSED="${3:-0.72}"
REOPEN="${4:-0.85}"
DANGER="${5:-1.7}"
YAWN_OPEN="${6:-0.18}"
YAWN_CLOSE="${7:-0.14}"
YAWN_SECONDS="${8:-0.3}"

if [ -z "$VIDEO" ] || [ ! -f "$VIDEO" ]; then
    echo "사용법: bash run_experiment.sh <영상경로> [backend] \\"
    echo "  [closed] [reopen] [danger] [yawn_open] [yawn_close] [yawn_seconds]"
    echo "  영상 파일이 존재해야 합니다."
    exit 1
fi
VIDEO="$(realpath "$VIDEO")"

case "$BACKEND" in
    opencv-fp32|tensorrt-fp32|tensorrt-fp16)
        ;;
    *)
        echo "[ERROR] 지원하지 않는 backend: $BACKEND"
        echo "  사용 가능: opencv-fp32, tensorrt-fp32, tensorrt-fp16"
        exit 1
        ;;
esac

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$PROJECT/scripts"
BENCH="$PROJECT/benchmark"
RESULTS="$BENCH/results"
LABELS="$RESULTS/labels/labels.csv"

if [ ! -f "$LABELS" ]; then
    echo "[ERROR] 라벨 파일이 없습니다: $LABELS"
    exit 1
fi

RES=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
      -of csv=p=0:s=x "$VIDEO" 2>/dev/null)
if [ -z "$RES" ]; then
    echo "[ERROR] 영상 해상도를 확인할 수 없습니다: $VIDEO"
    exit 1
fi

BACKEND_TAG="${BACKEND//-/_}_video_fsm"
TAG="${RES}_${BACKEND}_c${CLOSED}_r${REOPEN}_d${DANGER}"
TAG="${TAG}_yo${YAWN_OPEN}_yc${YAWN_CLOSE}_ys${YAWN_SECONDS}"

echo "=========================================="
echo "실험 시작: $(date)"
echo "  영상: $VIDEO (해상도 $RES)"
echo "  backend: $BACKEND"
echo "  눈 감김: closed=$CLOSED reopen=$REOPEN danger=$DANGER"
echo "  하품: open=$YAWN_OPEN close=$YAWN_CLOSE duration=$YAWN_SECONDS"
echo "=========================================="

echo ""
echo "[1/3] 추론 실행 중..."
cd "$SCRIPTS" || exit 1
python3 run_video_inference_FSM.py "$VIDEO" \
    --landmark-backend "$BACKEND" \
    --closed-ratio "$CLOSED" \
    --reopen-ratio "$REOPEN" \
    --danger-seconds "$DANGER" \
    --yawn-open-ratio "$YAWN_OPEN" \
    --yawn-close-ratio "$YAWN_CLOSE" \
    --yawn-seconds "$YAWN_SECONDS"
if [ $? -ne 0 ]; then
    echo "[ERROR] 추론 실패"
    exit 1
fi

FRAMES=$(ls -t "$RESULTS"/"${BACKEND_TAG}"_*_frames.csv 2>/dev/null | head -1)
if [ -z "$FRAMES" ] || [ ! -f "$FRAMES" ]; then
    echo "[ERROR] 생성된 frame CSV를 찾을 수 없습니다."
    exit 1
fi
SUMMARY="${FRAMES%_frames.csv}_summary.csv"
if [ ! -f "$SUMMARY" ]; then
    echo "[ERROR] frame CSV와 짝이 맞는 summary CSV가 없습니다: $SUMMARY"
    exit 1
fi

echo ""
echo "[2/3] 채점 중... (frames: $(basename "$FRAMES"))"
python3 "$BENCH/evaluate.py" \
    --labels "$LABELS" \
    --log "$FRAMES" \
    --out "$RESULTS/labels/score_${TAG}.csv"
if [ $? -ne 0 ]; then
    echo "[ERROR] 채점 실패"
    exit 1
fi

echo ""
echo "[3/3] 지표 리포트 생성 중..."
python3 "$BENCH/compute_metrics.py" \
    --frames "$FRAMES" \
    --labels "$LABELS" \
    --summary "$SUMMARY" \
    --out "$RESULTS/labels/metrics_${TAG}.csv"
if [ $? -ne 0 ]; then
    echo "[ERROR] 지표 리포트 생성 실패"
    exit 1
fi

echo ""
echo "=========================================="
echo "실험 완료: $(date)"
echo "  backend: $BACKEND"
echo "  frame CSV: $FRAMES"
echo "  summary CSV: $SUMMARY"
echo "  채점 결과: $RESULTS/labels/score_${TAG}.csv"
echo "  지표 리포트: $RESULTS/labels/metrics_${TAG}.csv"
echo "=========================================="
