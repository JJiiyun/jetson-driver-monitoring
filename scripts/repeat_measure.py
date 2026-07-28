#!/usr/bin/env python3
"""
반복 측정 자동화 스크립트.

같은 입력 영상으로 추론을 N회 반복 실행하고, 각 실행의 summary CSV를 모아
평균/표준편차를 계산해 하나의 집계 CSV로 저장한다.

목적:
  - 단발 측정은 그때그때 시스템 상태(백그라운드 부하 등)에 흔들린다.
  - 같은 영상을 여러 번 재서 평균 내면 신뢰할 수 있는 기준선이 나온다.
  - Day 5의 "FP32/FP16 3회 반복 측정 비교"의 FP32 쪽을 이걸로 처리.

전제:
  - run_video_inference.py (또는 FSM 래퍼)가 실행될 때마다
    benchmark/results/ 에 {run_id}_summary.csv 를 생성한다.
  - 이 스크립트는 그 스크립트를 subprocess로 N번 부르고,
    새로 생긴 summary CSV들을 읽어 집계한다.

사용법:
    # FSM 버전으로 같은 영상 3회 측정
    python3 repeat_measure.py --video ../data/검증영상.mp4 --runs 3 --fsm \
        --out ../benchmark/results/fp32_fsm_agg.csv

    # 기본(비FSM) 버전
    python3 repeat_measure.py --video ../data/검증영상.mp4 --runs 3 \
        --out ../benchmark/results/fp32_basic_agg.csv
"""

import argparse
import csv
import glob
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "benchmark" / "results"

# 집계할 수치 컬럼 (summary CSV에 있는 항목)
METRICS = [
    "end_to_end_fps",
    "capture_mean_ms", "capture_p95_ms",
    "inference_mean_ms", "inference_p95_ms",
    "processing_mean_ms", "processing_p95_ms",
    "frame_time_mean_ms", "frame_time_p95_ms",
    "average_face_count",
]


def latest_summary(before_set):
    """실행 후 새로 생긴 summary CSV 경로를 찾는다."""
    after = set(glob.glob(str(RESULTS_DIR / "*_summary.csv")))
    new = after - before_set
    if not new:
        return None
    # 새 파일이 여럿이면 가장 최근 것
    return max(new, key=os.path.getmtime)


def run_once(video, use_fsm, extra_args):
    """추론 스크립트를 1회 실행하고, 생성된 summary CSV 경로를 반환."""
    script = "run_video_inference_FSM.py" if use_fsm else "run_video_inference.py"
    script_path = SCRIPT_DIR / script

    before = set(glob.glob(str(RESULTS_DIR / "*_summary.csv")))

    cmd = [sys.executable, str(script_path), str(video)] + extra_args
    print(f"  실행: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, universal_newlines=True)
    if result.returncode != 0:
        print(f"  [ERROR] 실행 실패 (returncode {result.returncode})")
        print(result.stdout[-500:])
        print(result.stderr[-500:])
        return None

    summary = latest_summary(before)
    if summary is None:
        print("  [ERROR] 새 summary CSV를 찾지 못함")
        return None
    return summary


def read_summary(path):
    with open(path) as f:
        row = next(csv.DictReader(f))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="입력 영상 경로")
    ap.add_argument("--runs", type=int, default=3, help="반복 횟수 (기본 3)")
    ap.add_argument("--fsm", action="store_true", help="FSM 버전 사용")
    ap.add_argument("--out", default=None, help="집계 결과 CSV 경로")
    # 추론 스크립트에 그대로 넘길 추가 옵션 (예: --closed-ratio 0.73)
    ap.add_argument("passthrough", nargs="*",
                    help="추론 스크립트에 전달할 추가 인자 (--옵션 값)")
    args = ap.parse_args()

    if not Path(args.video).is_file():
        print(f"[ERROR] 영상 파일이 없음: {args.video}")
        return 1

    mode = "FSM" if args.fsm else "basic"
    print(f"반복 측정: {args.video}")
    print(f"  모드={mode}, 반복={args.runs}회")

    summaries = []
    for i in range(args.runs):
        print(f"\n[{i+1}/{args.runs}] 측정 중...")
        t0 = time.time()
        path = run_once(args.video, args.fsm, args.passthrough)
        if path is None:
            print("  이 회차 실패, 건너뜀")
            continue
        row = read_summary(path)
        summaries.append(row)
        print(f"  완료 ({time.time()-t0:.0f}s) "
              f"FPS={row.get('end_to_end_fps','?')}, "
              f"inference_mean={row.get('inference_mean_ms','?')}ms")

    if not summaries:
        print("\n[ERROR] 성공한 측정이 없음")
        return 1

    # 집계
    print(f"\n=== {len(summaries)}회 집계 ===")
    agg_rows = []
    for metric in METRICS:
        vals = []
        for s in summaries:
            v = s.get(metric, "")
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass
        if not vals:
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg_rows.append({
            "metric": metric,
            "mean": round(mean, 3),
            "std": round(std, 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
            "n": len(vals),
        })
        print(f"  {metric}: 평균 {mean:.2f} (±{std:.2f}), "
              f"범위 {min(vals):.2f}~{max(vals):.2f}")

    # 저장
    out = args.out
    if out is None:
        out = str(RESULTS_DIR / f"agg_{mode}_{int(time.time())}.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric", "mean", "std", "min", "max", "n"])
        writer.writeheader()
        for r in agg_rows:
            writer.writerow(r)
    print(f"\n집계 결과 저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
