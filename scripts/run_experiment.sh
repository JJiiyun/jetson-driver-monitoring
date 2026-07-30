#!/usr/bin/env bash
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

set -euo pipefail

VIDEO="${1:-}"
CLOSED="${2:-0.72}"
REOPEN="${3:-0.85}"
DANGER="${4:-1.7}"
YAWN_OPEN="0.18"
YAWN_CLOSE="0.14"
YAWN_SECONDS="0.3"
LANDMARK_BACKEND="${LANDMARK_BACKEND:-opencv-fp32}"
FACE_BACKEND="${FACE_BACKEND:-opencv-fp32}"
QT_DASHBOARD="${QT_DASHBOARD:-0}"
BUZZER_PIN="${BUZZER_PIN:-}"
GPIO_NUMBERING="${GPIO_NUMBERING:-BOARD}"
BUZZER_ACTIVE_LOW="${BUZZER_ACTIVE_LOW:-0}"

BUZZER_ARGS=()
DISPLAY_ARGS=()
if [ -n "$BUZZER_PIN" ]; then
    BUZZER_ARGS+=(--buzzer-pin "$BUZZER_PIN" --gpio-numbering "$GPIO_NUMBERING")
    if [ "$BUZZER_ACTIVE_LOW" = "1" ]; then
        BUZZER_ARGS+=(--buzzer-active-low)
    fi
fi
if [ "$QT_DASHBOARD" = "1" ]; then
    DISPLAY_ARGS+=(--qt-dashboard)
fi

if [ -z "$VIDEO" ] || [ ! -f "$VIDEO" ]; then
    echo "사용법: bash run_experiment.sh <영상경로> [closed] [reopen] [danger]"
    echo "  영상 파일이 존재해야 합니다."
    exit 1
fi

VIDEO="$(realpath "$VIDEO")"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$PROJECT/scripts"
BENCH="$PROJECT/benchmark"
RESULTS="$BENCH/results"
LABELS="$RESULTS/labels/labels.csv"

source "$PROJECT/zzmvenv/bin/activate"
source "$SCRIPTS/use_opencv_cuda.sh"

python3 - <<'PY'
import cv2

print("OpenCV:", cv2.__version__)
print("CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
if cv2.cuda.getCudaEnabledDeviceCount() < 1:
    raise SystemExit("[ERROR] OpenCV cannot see a CUDA device.")
PY

# 영상 해상도 (파일명 태그용)
RES=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
      -of csv=p=0:s=x "$VIDEO" 2>/dev/null)
TAG="${RES}_c${CLOSED}"

echo "=========================================="
echo "실험 시작: $(date)"
echo "  영상: $VIDEO (해상도 $RES)"
echo "  파라미터: closed=$CLOSED reopen=$REOPEN danger=$DANGER"
echo "  하품 파라미터: open=$YAWN_OPEN close=$YAWN_CLOSE duration=$YAWN_SECONDS"
echo "  랜드마크 백엔드: $LANDMARK_BACKEND"
echo "  얼굴 백엔드: $FACE_BACKEND"
echo "=========================================="

# 1) 추론
echo ""
echo "[1/3] 추론 실행 중..."
cd "$SCRIPTS"
python3 run_video_inference_FSM.py "$VIDEO" \
    --landmark-backend "$LANDMARK_BACKEND" \
    --face-backend "$FACE_BACKEND" \
    --opencv-device cuda \
    --closed-ratio "$CLOSED" \
    --reopen-ratio "$REOPEN" \
    --danger-seconds "$DANGER" \
    --yawn-open-ratio "$YAWN_OPEN" \
    --yawn-close-ratio "$YAWN_CLOSE" \
    --yawn-seconds "$YAWN_SECONDS" \
    "${BUZZER_ARGS[@]}" \
    "${DISPLAY_ARGS[@]}"

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
