#!/bin/bash
# 실험 자동화: 추론 -> 채점 -> 지표 리포트까지 한 번에
#
# 사용법:
#   bash run_experiment.sh <영상경로> [closed_ratio] [reopen_ratio] [danger_sec]
#
# 예시:
#   bash run_experiment.sh ../data/final_test_640x360.mp4
#   bash run_experiment.sh ../data/final_test_640x360.mp4 0.72 0.85 1.7
#
# 파라미터 안 주면 기본값(0.72 / 0.85 / 1.7) 사용.

VIDEO="$1"
CLOSED="${2:-0.72}"
REOPEN="${3:-0.85}"
DANGER="${4:-1.7}"

if [ -z "$VIDEO" ] || [ ! -f "$VIDEO" ]; then
    echo "사용법: bash run_experiment.sh <영상경로> [closed] [reopen] [danger]"
    echo "  영상 파일이 존재해야 합니다."
    exit 1
fi

PROJECT=~/jetson-driver-monitoring
SCRIPTS="$PROJECT/scripts"
BENCH="$PROJECT/benchmark"
RESULTS="$BENCH/results"
LABELS="$RESULTS/labels/labels.csv"

# 영상 해상도 (파일명 태그용)
RES=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
      -of csv=p=0:s=x "$VIDEO" 2>/dev/null)
TAG="${RES}_c${CLOSED}"

echo "=========================================="
echo "실험 시작: $(date)"
echo "  영상: $VIDEO (해상도 $RES)"
echo "  파라미터: closed=$CLOSED reopen=$REOPEN danger=$DANGER"
echo "=========================================="

# 1) 추론
echo ""
echo "[1/3] 추론 실행 중..."
cd "$SCRIPTS"
python3 run_video_inference_FSM.py "$VIDEO" \
    --closed-ratio "$CLOSED" \
    --reopen-ratio "$REOPEN" \
    --danger-seconds "$DANGER"
if [ $? -ne 0 ]; then
    echo "[ERROR] 추론 실패"
    exit 1
fi

# 2) 방금 생성된 최신 frames/summary 자동 탐지
FRAMES=$(ls -t "$RESULTS"/*_frames.csv | head -1)
SUMMARY=$(ls -t "$RESULTS"/*_summary.csv | head -1)
echo ""
echo "[2/3] 채점 중... (frames: $(basename $FRAMES))"
python3 "$BENCH/evaluate.py" \
    --labels "$LABELS" \
    --log "$FRAMES" \
    --out "$RESULTS/labels/score_${TAG}.csv"

# 3) 지표 리포트
echo ""
echo "[3/3] 지표 리포트 생성 중..."
python3 "$BENCH/compute_metrics.py" \
    --frames "$FRAMES" \
    --labels "$LABELS" \
    --summary "$SUMMARY" \
    --out "$RESULTS/labels/metrics_${TAG}.csv"

echo ""
echo "=========================================="
echo "실험 완료: $(date)"
echo "  채점 결과: $RESULTS/labels/score_${TAG}.csv"
echo "  지표 리포트: $RESULTS/labels/metrics_${TAG}.csv"
echo "=========================================="
