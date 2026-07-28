#!/usr/bin/env python3
"""
tegrastats 수집 스크립트.

Jetson Nano의 tegrastats 출력을 백그라운드로 받아 CSV로 저장한다.
추론 실행과 동시에 돌리면, CPU/GPU/메모리 사용률을 시간에 따라 기록해
성능 병목(CPU로 도는지 GPU로 도는지, processing 구간이 왜 느린지)을 진단할 수 있다.

사용법:
    # 1) 별도 터미널에서 수집 시작 (측정할 시간만큼)
    python3 collect_tegrastats.py --duration 120 --out tegra_fp32.csv

    # 2) 수집 도는 동안 다른 터미널에서 추론 실행
    #    python3 run_video_inference_FSM.py 영상.mp4

    # duration을 안 주면 Ctrl+C 로 멈출 때까지 계속 수집한다.

파싱 항목(있을 때만):
    ram_used_mb, ram_total_mb  : 메모리 사용/전체 (MB)
    cpu_pct_avg                : CPU 코어 평균 사용률 (%)
    cpu_cores                  : 코어별 사용률 (예: "12%,3%,0%,5%")
    gr3d_pct                   : GPU(GR3D) 사용률 (%)  ← 0이면 GPU 미사용
    temp_cpu_c, temp_gpu_c     : CPU/GPU 온도 (있으면)
"""

import argparse
import csv
import re
import signal
import subprocess
import sys
import time


# tegrastats 출력 예:
# RAM 2450/3956MB ... CPU [12%@1479,3%@1479,0%@1479,5%@1479] ... GR3D_FREQ 0%@76 ...
RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
CPU_RE = re.compile(r"CPU \[([^\]]+)\]")
GR3D_RE = re.compile(r"GR3D_FREQ (\d+)%")
CPU_TEMP_RE = re.compile(r"CPU@([\d.]+)C")
GPU_TEMP_RE = re.compile(r"GPU@([\d.]+)C")


def parse_line(line):
    row = {
        "ram_used_mb": "", "ram_total_mb": "",
        "cpu_pct_avg": "", "cpu_cores": "",
        "gr3d_pct": "", "temp_cpu_c": "", "temp_gpu_c": "",
    }

    m = RAM_RE.search(line)
    if m:
        row["ram_used_mb"] = m.group(1)
        row["ram_total_mb"] = m.group(2)

    m = CPU_RE.search(line)
    if m:
        cores = m.group(1)  # "12%@1479,3%@1479,..."
        pcts = re.findall(r"(\d+)%", cores)
        if pcts:
            vals = [int(p) for p in pcts]
            row["cpu_pct_avg"] = round(sum(vals) / len(vals), 1)
            row["cpu_cores"] = ",".join(f"{v}%" for v in vals)

    m = GR3D_RE.search(line)
    if m:
        row["gr3d_pct"] = m.group(1)

    m = CPU_TEMP_RE.search(line)
    if m:
        row["temp_cpu_c"] = m.group(1)
    m = GPU_TEMP_RE.search(line)
    if m:
        row["temp_gpu_c"] = m.group(1)

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tegrastats.csv", help="저장할 CSV 경로")
    ap.add_argument("--interval", type=int, default=1000,
                    help="tegrastats 샘플 간격(ms), 기본 1000")
    ap.add_argument("--duration", type=float, default=None,
                    help="수집 시간(초). 안 주면 Ctrl+C 까지 계속")
    args = ap.parse_args()

    fields = ["elapsed_s", "ram_used_mb", "ram_total_mb",
              "cpu_pct_avg", "cpu_cores", "gr3d_pct",
              "temp_cpu_c", "temp_gpu_c"]

    # tegrastats 실행 (sudo 필요할 수 있음)
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", str(args.interval)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1,
        )
    except FileNotFoundError:
        print("[ERROR] tegrastats 명령을 찾을 수 없습니다. Jetson 환경인지 확인하세요.")
        return 1

    print(f"수집 시작 -> {args.out}")
    if args.duration:
        print(f"  {args.duration}초 동안 수집합니다.")
    else:
        print("  Ctrl+C 로 멈출 때까지 수집합니다.")

    start = time.time()
    count = 0
    stopping = {"flag": False}

    def handle_sigint(sig, frame):
        stopping["flag"] = True
    signal.signal(signal.SIGINT, handle_sigint)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        try:
            for line in proc.stdout:
                if stopping["flag"]:
                    break
                elapsed = time.time() - start
                if args.duration and elapsed > args.duration:
                    break
                parsed = parse_line(line)
                parsed["elapsed_s"] = round(elapsed, 1)
                writer.writerow(parsed)
                f.flush()
                count += 1
                # 진행 표시
                if count % 10 == 0:
                    gpu = parsed.get("gr3d_pct", "?")
                    cpu = parsed.get("cpu_pct_avg", "?")
                    print(f"  {elapsed:.0f}s  CPU평균 {cpu}%  GPU {gpu}%")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    print(f"\n수집 완료: {count}개 샘플 -> {args.out}")

    # 요약
    _summarize(args.out)
    return 0


def _summarize(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    def col(name):
        vals = []
        for r in rows:
            v = r.get(name, "")
            if v not in ("", None):
                try:
                    vals.append(float(str(v).replace("%", "")))
                except ValueError:
                    pass
        return vals

    def stat(name, unit=""):
        vals = col(name)
        if not vals:
            return f"  {name}: 데이터 없음"
        return (f"  {name}: 평균 {sum(vals)/len(vals):.1f}{unit}, "
                f"최대 {max(vals):.1f}{unit}")

    print("\n=== tegrastats 요약 ===")
    print(stat("cpu_pct_avg", "%"))
    print(stat("gr3d_pct", "%"))
    print(stat("ram_used_mb", "MB"))
    gpu_vals = col("gr3d_pct")
    if gpu_vals:
        gpu_active = sum(1 for v in gpu_vals if v > 0) / len(gpu_vals) * 100
        print(f"  GPU 사용된 샘플 비율: {gpu_active:.0f}% "
              f"({'GPU 활용중' if gpu_active > 10 else 'GPU 거의 미사용=CPU로 추론'})")


if __name__ == "__main__":
    sys.exit(main())
