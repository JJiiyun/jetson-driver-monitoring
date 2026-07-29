#!/bin/bash
# 같은 영상 + 같은 튜닝값으로 3번 순차 실행 (밤새 돌리기용)
#
# 사용법:
#   bash run_3times.sh <영상경로>
#   또는 백그라운드로:  nohup bash run_3times.sh <영상경로> > ~/run3.log 2>&1 &
#
# 튜닝값: closed=0.75(Recall↑,DANGER유도), reopen=0.85(오탐억제), danger=1.7

VIDEO="$1"
if [ -z "$VIDEO" ] || [ ! -f "$VIDEO" ]; then
    echo "사용법: bash run_3times.sh <영상경로>  (파일이 존재해야 함)"
    exit 1
fi

SCRIPTS=~/jetson-driver-monitoring/scripts
cd "$SCRIPTS"

echo "=========================================="
echo "3회 순차 실행 시작: $(date)"
echo "영상: $VIDEO"
echo "튜닝값: closed-ratio=0.72, reopen-ratio=0.85, danger-seconds=1.7"
echo "=========================================="

for i in 1 2 3; do
    echo ""
    echo "----- [$i/3] 실행 시작: $(date) -----"
    python3 run_video_inference_FSM.py "$VIDEO" \
        --closed-ratio 0.72 \
        --reopen-ratio 0.85 \
        --danger-seconds 1.7
    echo "----- [$i/3] 실행 완료: $(date) -----"
done

echo ""
echo "=========================================="
echo "3회 전부 완료: $(date)"
echo "결과 CSV는 benchmark/results/ 에 3세트 생성됨"
echo "(각 실행마다 _frames.csv, _summary.csv 한 쌍씩)"
echo "=========================================="
