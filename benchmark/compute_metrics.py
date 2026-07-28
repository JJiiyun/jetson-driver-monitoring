#!/usr/bin/env python3
"""
6.2 평가 지표 종합 리포트.

여러 소스를 읽어 계획서 6.2 표의 지표를 한 번에 계산한다.
  - frames.csv   : 프레임별 판정(is_eye_closed, eye_state, frame_index, target_fps)
  - labels.csv   : 눈 감김/하품 정답 구간
  - summary.csv  : FPS, 추론/처리 지연 (performance_logger 출력)
  - tegra.csv    : CPU/GPU/메모리(/전력) (collect_tegrastats 출력, 선택)

계산 지표:
  End-to-end FPS, 평균 추론 지연, P95 지연시간   (summary.csv)
  눈 감김 Event Recall                          (frames + labels)
  False Alarms per Minute                       (frames + labels)
  위험 경고 지연 (눈감김 시작→DANGER)            (frames + labels)
  메모리 사용량, 전력 사용량                      (tegra.csv)
  경고 등급 일치율은 FP16이 있어야 하므로 여기선 생략(별도)

사용법:
  python3 compute_metrics.py \
      --frames run_xxx_frames.csv \
      --labels labels.csv \
      --summary run_xxx_summary.csv \
      --tegra tegra_fp32.csv \
      --out metrics_report.csv
"""

import argparse
import csv
from datetime import datetime


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def load_frames(path):
    """frames.csv -> [{t, eye_closed, state}], frame_index/fps로 시간 계산."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        has_fi = "frame_index" in fields and "target_fps" in fields
        rows = []
        for r in reader:
            if has_fi:
                t = (float(r["frame_index"]) - 1) / float(r["target_fps"])
            else:
                # fallback: timestamp
                try:
                    t = float(r.get("timestamp", 0))
                except ValueError:
                    t = 0.0
            rows.append({
                "t": t,
                "eye_closed": _truthy(r.get("is_eye_closed", "")),
                "state": (r.get("eye_state", "") or "").strip().upper(),
            })
    if rows:
        t0 = rows[0]["t"]
        if t0 != 0:
            for r in rows:
                r["t"] -= t0
    return rows


def load_labels(path):
    labels = {"eye_closed": [], "yawn": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                et = row["event_type"].strip()
                s = float(row["start_sec"]); e = float(row["end_sec"])
            except (KeyError, ValueError):
                continue
            if e >= s:
                labels.setdefault(et, []).append((s, e))
    for et in labels:
        labels[et].sort()
    return labels


def load_summary(path):
    with open(path, newline="") as f:
        return next(csv.DictReader(f))


def load_tegra(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def detected_intervals(frames, key="eye_closed"):
    intervals, start, prev = [], None, None
    for fr in frames:
        if fr[key]:
            if start is None:
                start = fr["t"]
        else:
            if start is not None:
                intervals.append((start, prev)); start = None
        prev = fr["t"]
    if start is not None:
        intervals.append((start, prev))
    return intervals


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def event_recall(frames, gt):
    det = detected_intervals(frames, "eye_closed")
    matched = sum(1 for g in gt if any(overlaps(g, d) for d in det))
    recall = matched / len(gt) if gt else None
    return recall, matched, len(gt)


def false_alarms_per_min(frames, gt):
    """정답 구간에 안 겹치는 감지 구간 수 / 전체 분."""
    det = detected_intervals(frames, "eye_closed")
    fa = sum(1 for d in det if not any(overlaps(g, d) for g in gt))
    total_min = (frames[-1]["t"] - frames[0]["t"]) / 60.0 if frames else 0
    per_min = fa / total_min if total_min > 0 else None
    return per_min, fa, round(total_min, 2)


def danger_latency(frames, gt):
    """
    각 눈 감김 라벨의 시작 시각부터, 그 근처에서 처음 DANGER 상태가 뜬 시각까지의 지연.
    라벨 시작 이후 ~5초 안에 DANGER가 뜬 경우만 집계(2초 이상 감은 이벤트 대상).
    """
    danger_times = [fr["t"] for fr in frames if fr["state"] == "DANGER"]
    latencies = []
    for (s, e) in gt:
        # 라벨 시작 s 이후 처음 나온 DANGER
        after = [dt for dt in danger_times if s <= dt <= s + 5.0]
        if after:
            latencies.append(after[0] - s)
    if not latencies:
        return None, 0
    return sum(latencies) / len(latencies), len(latencies)


def tegra_stats(rows, col):
    vals = []
    for r in rows:
        v = r.get(col, "")
        if v not in ("", None):
            try:
                vals.append(float(str(v).replace("%", "")))
            except ValueError:
                pass
    if not vals:
        return None, None
    return sum(vals) / len(vals), max(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--summary")
    ap.add_argument("--tegra")
    ap.add_argument("--out", default="metrics_report.csv")
    args = ap.parse_args()

    frames = load_frames(args.frames)
    labels = load_labels(args.labels)
    gt_closed = labels.get("eye_closed", [])

    report = []  # (지표, 값, 목표, 판정)

    # --- summary 기반 ---
    if args.summary:
        s = load_summary(args.summary)
        fps = float(s.get("end_to_end_fps", 0) or 0)
        report.append(("End-to-end FPS", f"{fps:.2f}", "15 이상",
                       "PASS" if fps >= 15 else "FAIL"))
        report.append(("평균 추론 지연(ms)",
                       s.get("inference_mean_ms", "-"), "구성별 비교", "-"))
        report.append(("P95 추론 지연(ms)",
                       s.get("inference_p95_ms", "-"), "구성별 비교", "-"))
        report.append(("평균 프레임 지연(ms)",
                       s.get("frame_time_mean_ms", "-"), "구성별 비교", "-"))
        report.append(("P95 프레임 지연(ms)",
                       s.get("frame_time_p95_ms", "-"), "구성별 비교", "-"))

    # --- 정확도 (frames + labels) ---
    recall, matched, total = event_recall(frames, gt_closed)
    if recall is not None:
        report.append(("눈 감김 Event Recall",
                       f"{recall*100:.1f}% ({matched}/{total})", "90% 이상",
                       "PASS" if recall >= 0.90 else "FAIL"))

    fa_pm, fa_cnt, mins = false_alarms_per_min(frames, gt_closed)
    if fa_pm is not None:
        report.append(("False Alarms/min",
                       f"{fa_pm:.2f} ({fa_cnt}건/{mins}분)", "0.33 이하",
                       "PASS" if fa_pm <= 0.33 else "FAIL"))

    lat, n = danger_latency(frames, gt_closed)
    if lat is not None:
        report.append(("위험 경고 지연(초)",
                       f"{lat:.2f} ({n}건 평균)", "2.5 이하",
                       "PASS" if lat <= 2.5 else "FAIL"))
    else:
        report.append(("위험 경고 지연(초)", "DANGER 미발생", "2.5 이하", "-"))

    # --- tegrastats 기반 ---
    if args.tegra:
        trows = load_tegra(args.tegra)
        ram_mean, ram_max = tegra_stats(trows, "ram_used_mb")
        if ram_mean is not None:
            report.append(("메모리 사용량(MB)",
                           f"평균 {ram_mean:.0f} / 최대 {ram_max:.0f}",
                           "구성별 비교", "-"))
        # 전력: tegrastats에 POM_5V_IN 등이 있으면 (없을 수 있음)
        pow_mean, _ = tegra_stats(trows, "power_mw")
        if pow_mean is not None:
            report.append(("전력 사용량(mW)", f"평균 {pow_mean:.0f}",
                           "구성별 비교", "-"))
        gpu_mean, gpu_max = tegra_stats(trows, "gr3d_pct")
        if gpu_mean is not None:
            report.append(("GPU 사용률(%)",
                           f"평균 {gpu_mean:.1f} / 최대 {gpu_max:.1f}",
                           "참고", "-"))
        cpu_mean, cpu_max = tegra_stats(trows, "cpu_pct_avg")
        if cpu_mean is not None:
            report.append(("CPU 사용률(%)",
                           f"평균 {cpu_mean:.1f} / 최대 {cpu_max:.1f}",
                           "참고", "-"))

    # --- 출력 ---
    print("\n" + "=" * 60)
    print("6.2 평가 지표 리포트")
    print("=" * 60)
    w = max(len(r[0]) for r in report)
    for name, val, target, verdict in report:
        mark = {"PASS": "[O]", "FAIL": "[X]", "-": "   "}[verdict]
        print(f"{mark} {name:<{w}} : {val}  (목표: {target})")
    print("=" * 60)
    print("경고 등급 일치율은 FP16 결과가 있어야 계산 가능(별도)")

    # --- CSV 저장 ---
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "target", "verdict"])
        for row in report:
            writer.writerow(row)
    print(f"\n리포트 저장: {args.out}")


if __name__ == "__main__":
    main()
