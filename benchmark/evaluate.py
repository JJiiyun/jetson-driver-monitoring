"""
이벤트 평가 코드 (frame_index 기반 시간축).

라벨링 정답지와 파이프라인 로그(*_frames.csv)를 비교해 채점한다.

시간축: timestamp(벽시계)는 처리 속도에 따라 왜곡되므로 쓰지 않는다.
        frame_index / target_fps 로 영상 경과초를 계산한다.
        (첫 프레임 index=1 을 0초로 맞춤)

사용법:
    python3 evaluate.py --labels labels.csv --log run_xxx_frames.csv --out score.csv
"""

import argparse
import csv
from datetime import datetime

EYE_ALIASES = ["is_eye_closed", "eye_closed", "is_closed", "closed", "eye_close"]
YAWN_ALIASES = ["is_yawning", "yawning", "yawn"]
TIME_ALIASES = ["video_time", "frame_time", "timestamp", "time", "t", "elapsed_sec"]


def _pick_column(fieldnames, aliases):
    lower = {f.lower(): f for f in fieldnames}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def _parse_time(raw):
    raw = str(raw).strip()
    try:
        return float(raw)
    except ValueError:
        pass
    return datetime.fromisoformat(raw).timestamp()


def load_labels(path):
    labels = {"eye_closed": [], "yawn": []}
    skipped = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                et = row["event_type"].strip()
                s = float(row["start_sec"])
                e = float(row["end_sec"])
            except (KeyError, ValueError, AttributeError):
                skipped += 1
                continue
            if e < s:
                skipped += 1
                continue
            labels.setdefault(et, []).append((s, e))
    for et in labels:
        labels[et].sort()
    if skipped:
        print(f"  (경고: 형식 오류로 건너뛴 라벨 {skipped}개)")
    return labels


def load_log(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        eye_col = _pick_column(fields, EYE_ALIASES)
        yawn_col = _pick_column(fields, YAWN_ALIASES)

        has_frame_idx = "frame_index" in fields and "target_fps" in fields
        t_col = _pick_column(fields, TIME_ALIASES)

        if not has_frame_idx and t_col is None:
            raise ValueError(
                f"시간 정보를 못 찾음(frame_index+target_fps 또는 시간 컬럼 필요).\n"
                f"  실제 컬럼: {fields}"
            )

        frames = []
        for row in reader:
            if has_frame_idx:
                fps = float(row["target_fps"])
                t_val = (float(row["frame_index"]) - 1) / fps
            else:
                t_val = _parse_time(row[t_col])
            frames.append({
                "t": t_val,
                "eye_closed": _truthy(row[eye_col]) if eye_col else False,
                "yawning": _truthy(row[yawn_col]) if yawn_col else False,
            })

    # 첫 프레임이 0이 아니면 0으로 맞춤(안전장치)
    if frames:
        t0 = frames[0]["t"]
        if t0 != 0:
            for fr in frames:
                fr["t"] -= t0

    time_src = "frame_index/target_fps" if has_frame_idx else t_col
    info = {"time": time_src, "eye_closed": eye_col, "yawning": yawn_col}
    return frames, info


def extract_detected_intervals(frames, key):
    intervals = []
    start = None
    prev_t = None
    for fr in frames:
        if fr[key]:
            if start is None:
                start = fr["t"]
        else:
            if start is not None:
                intervals.append((start, prev_t))
                start = None
        prev_t = fr["t"]
    if start is not None:
        intervals.append((start, prev_t))
    return intervals


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def event_level_score(gt_intervals, det_intervals):
    matched_gt = 0
    matched_det_idx = set()
    for gt in gt_intervals:
        found = False
        for i, det in enumerate(det_intervals):
            if overlaps(gt, det):
                found = True
                matched_det_idx.add(i)
        if found:
            matched_gt += 1
    false_alarms = len(det_intervals) - len(matched_det_idx)
    recall = matched_gt / len(gt_intervals) if gt_intervals else None
    return {"gt_total": len(gt_intervals), "gt_matched": matched_gt,
            "recall": recall, "det_total": len(det_intervals),
            "false_alarms": false_alarms}


def frame_level_score(frames, gt_intervals, key):
    def in_gt(t):
        return any(s <= t <= e for s, e in gt_intervals)
    tp = fp = fn = tn = 0
    for fr in frames:
        pred = fr[key]
        truth = in_gt(fr["t"])
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision and recall and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else ("-" if x is None else str(x))


def score_one(name, log_key, frames, gt_intervals):
    det = extract_detected_intervals(frames, log_key)
    ev = event_level_score(gt_intervals, det)
    fl = frame_level_score(frames, gt_intervals, log_key)
    print(f"\n===== {name} =====")
    print("[이벤트 단위]")
    print(f"  정답 이벤트: {ev['gt_total']}개")
    print(f"  잡은 이벤트: {ev['gt_matched']}개")
    print(f"  검출율(Recall): {fmt(ev['recall'])}")
    print(f"  감지 구간 수: {ev['det_total']}개")
    print(f"  오탐(False Alarm) 구간: {ev['false_alarms']}개")
    print("[프레임 단위]")
    print(f"  TP={fl['tp']} FP={fl['fp']} FN={fl['fn']} TN={fl['tn']}")
    print(f"  Precision: {fmt(fl['precision'])}")
    print(f"  Recall:    {fmt(fl['recall'])}")
    print(f"  F1:        {fmt(fl['f1'])}")
    return {"target": name, "event_gt_total": ev["gt_total"],
            "event_gt_matched": ev["gt_matched"], "event_recall": fmt(ev["recall"]),
            "det_total": ev["det_total"], "false_alarms": ev["false_alarms"],
            "frame_tp": fl["tp"], "frame_fp": fl["fp"], "frame_fn": fl["fn"],
            "frame_tn": fl["tn"], "frame_precision": fmt(fl["precision"]),
            "frame_recall": fmt(fl["recall"]), "frame_f1": fmt(fl["f1"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    frames, info = load_log(args.log)

    print(f"정답지: {args.labels}")
    print(f"로그: {args.log} ({len(frames)} 프레임)")
    if frames:
        print(f"로그 구간: 0.0 ~ {frames[-1]['t']:.1f}초")
    print(f"시간 출처: {info['time']}, 눈감김='{info['eye_closed']}', "
          f"하품='{info['yawning']}'")
    if info["yawning"] is None:
        print("  (하품 컬럼 없음 -> 하품 채점 생략)")

    results = [score_one("eye_closed", "eye_closed",
                         frames, labels.get("eye_closed", []))]
    if info["yawning"] is not None:
        results.append(score_one("yawn", "yawning",
                                 frames, labels.get("yawn", [])))

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"\n채점 결과 저장: {args.out}")


if __name__ == "__main__":
    main()