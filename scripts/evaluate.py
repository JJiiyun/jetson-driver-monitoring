"""
이벤트 평가 코드.

라벨링 정답지(labels.csv)와 파이프라인 로그(*_frames.csv)를 비교해
눈 감김 / 하품 감지 성능을 채점한다.

두 관점으로 채점한다:
  1. 이벤트 단위 (Event-level): 라벨된 이벤트 구간을 잡았는가?
     - Recall     = 잡은 정답 이벤트 / 전체 정답 이벤트
     - False Alarm = 정답에 없는데 감지가 발생한 구간 수
  2. 프레임 단위 (Frame-level): 매 프레임 판정이 정답과 맞는가?
     - Precision / Recall / F1

로그 CSV의 컬럼 이름은 파이프라인마다 다를 수 있어, 흔한 이름 후보를
자동으로 찾아 매칭한다(아래 *_ALIASES). 못 찾으면 명확히 알려준다.

정답지 형식 (labels.csv):
    event_type,start_sec,end_sec
    eye_closed,5.2,6.1
    yawn,25.0,27.3

사용법:
    python3 evaluate.py --labels labels.csv --log run_xxx_frames.csv
    python3 evaluate.py --labels labels.csv --log run_xxx_frames.csv --out score.csv
"""

import argparse
import csv

# --- 로그 CSV 컬럼 이름 후보 (앞에서부터 먼저 발견되는 것을 사용) ---
TIME_ALIASES = ["video_time", "frame_time", "timestamp", "time", "t", "elapsed_sec"]
EYE_ALIASES = ["eye_closed", "is_closed", "closed", "eye_close"]
YAWN_ALIASES = ["yawning", "is_yawning", "yawn"]


def _pick_column(fieldnames, aliases):
    """fieldnames에서 aliases 중 먼저 발견되는 실제 컬럼명을 반환. 없으면 None."""
    lower = {f.lower(): f for f in fieldnames}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def _truthy(v):
    """'1', 'true', 'True', 1 등을 bool로."""
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def load_labels(path):
    """정답지 로드. {event_type: [(start, end), ...]} 반환. 잘못된 구간은 걸러냄."""
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
            if e < s:  # 시간이 거꾸로면 스킵
                skipped += 1
                continue
            labels.setdefault(et, []).append((s, e))
    for et in labels:
        labels[et].sort()
    if skipped:
        print(f"  (경고: 형식 오류로 건너뛴 라벨 {skipped}개)")
    return labels


def load_log(path):
    """
    파이프라인 로그 로드. 컬럼 이름을 자동 매칭.
    반환: (frames, info)
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        t_col = _pick_column(fields, TIME_ALIASES)
        eye_col = _pick_column(fields, EYE_ALIASES)
        yawn_col = _pick_column(fields, YAWN_ALIASES)

        if t_col is None:
            raise ValueError(
                f"로그에서 시간 컬럼을 찾지 못함. 후보: {TIME_ALIASES}\n"
                f"  실제 컬럼: {fields}"
            )

        frames = []
        for row in reader:
            frames.append({
                "t": float(row[t_col]),
                "eye_closed": _truthy(row[eye_col]) if eye_col else False,
                "yawning": _truthy(row[yawn_col]) if yawn_col else False,
            })

    info = {"time": t_col, "eye_closed": eye_col, "yawning": yawn_col}
    return frames, info


def extract_detected_intervals(frames, key):
    """로그에서 key가 True인 연속 구간을 이벤트로 묶는다. [(start,end), ...]."""
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
    return {
        "gt_total": len(gt_intervals),
        "gt_matched": matched_gt,
        "recall": recall,
        "det_total": len(det_intervals),
        "false_alarms": false_alarms,
    }


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

    return {
        "target": name,
        "event_gt_total": ev["gt_total"],
        "event_gt_matched": ev["gt_matched"],
        "event_recall": fmt(ev["recall"]),
        "det_total": ev["det_total"],
        "false_alarms": ev["false_alarms"],
        "frame_tp": fl["tp"], "frame_fp": fl["fp"],
        "frame_fn": fl["fn"], "frame_tn": fl["tn"],
        "frame_precision": fmt(fl["precision"]),
        "frame_recall": fmt(fl["recall"]),
        "frame_f1": fmt(fl["f1"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="정답지 CSV")
    ap.add_argument("--log", required=True, help="파이프라인 로그 CSV")
    ap.add_argument("--out", help="채점 결과를 저장할 CSV 경로 (선택)")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    frames, info = load_log(args.log)

    print(f"정답지: {args.labels}")
    print(f"로그: {args.log} ({len(frames)} 프레임)")
    print(f"매칭된 컬럼: 시간='{info['time']}', "
          f"눈감김='{info['eye_closed']}', 하품='{info['yawning']}'")
    if info["eye_closed"] is None:
        print("  (경고: 눈감김 컬럼을 못 찾음 -> 눈감김 채점은 전부 미검출 처리됨)")
    if info["yawning"] is None:
        print("  (경고: 하품 컬럼을 못 찾음 -> 하품 채점 생략)")

    results = []
    results.append(
        score_one("eye_closed", "eye_closed",
                  frames, labels.get("eye_closed", []))
    )
    if info["yawning"] is not None:
        results.append(
            score_one("yawn", "yawning",
                      frames, labels.get("yawn", []))
        )

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\n채점 결과 저장: {args.out}")


if __name__ == "__main__":
    main()
