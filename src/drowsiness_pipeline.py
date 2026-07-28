"""
Jetson Nano 졸음 감지 파이프라인 (길 B: OpenCV로 ONNX 추론).

흐름:
  영상 프레임
    -> YuNet(cv2.FaceDetectorYN)으로 얼굴 박스
    -> 얼굴 crop (정사각형 x1.1, 112x112)
    -> PFLD ONNX (cv2.dnn) -> 68점 랜드마크
    -> EAR 계산 -> 상대 EAR 캘리브레이션 -> eye_closed
    -> PERCLOS 모듈 -> 졸음 상태 판정
    -> 화면 오버레이 + 결과 영상 저장 + CSV 로그

사용법:
  python3 drowsiness_pipeline.py --video 입력영상.mp4 --out 결과영상.mp4 --csv 로그.csv

  웹캠으로 바로 테스트하려면:
  python3 drowsiness_pipeline.py --camera 0

주의: timestamp는 영상 기준 경과초(frame_idx / fps)를 쓴다.
      실제 처리 속도(FP16/FP32)와 무관하게 동일 시간축으로 비교하기 위함.
"""

import argparse
import csv
import time

import cv2
import numpy as np

from perclos import PerclosCalculator
from ear import compute_ear, EyeStateJudge


# ---------- 경로 기본값 (프로젝트 구조 기준) ----------
YUNET_PATH = "models/face_detector/yunet.onnx"
PFLD_ONNX_PATH = "models/face_landmark/pfld_sim.onnx"

PFLD_INPUT_SIZE = 112   # PFLD 입력 크기
CROP_SCALE = 1.1        # 얼굴 박스 확대 비율 (여백 10%)


# ---------- PFLD 랜드마크 추론 ----------
class PFLDLandmark:
    def __init__(self, onnx_path):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)

    def detect(self, frame, box):
        """
        frame: 원본 BGR 이미지
        box: (x, y, w, h) 얼굴 박스 (YuNet 결과)
        반환: (68,2) 랜드마크 (원본 이미지 좌표), 실패 시 None
        """
        h_img, w_img = frame.shape[:2]
        x, y, w, h = box

        # 정사각형으로 확장 (긴 변 기준 x1.1)
        size = int(max(w, h) * CROP_SCALE)
        cx = x + w // 2
        cy = y + h // 2
        x1 = cx - size // 2
        y1 = cy - size // 2
        x2 = x1 + size
        y2 = y1 + size

        # 이미지 밖으로 나간 만큼 패딩 계산
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(w_img, x2)
        y2c = min(h_img, y2)

        cropped = frame[y1c:y2c, x1c:x2c]
        if cropped.size == 0:
            return None

        edx = max(0, x2 - w_img)
        edy = max(0, y2 - h_img)
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            cropped = cv2.copyMakeBorder(
                cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, value=0
            )

        # 112x112 RGB, 0~1 정규화 (test_camera_pfld_onnx.py 규약과 동일)
        face = cv2.resize(cropped, (PFLD_INPUT_SIZE, PFLD_INPUT_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        blob = face.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))       # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)         # -> (1,3,112,112)

        self.net.setInput(blob)
        out = self.net.forward()                    # (1,136)
        lm = out.reshape(-1, 2)                      # (68,2), 0~1 정규화 좌표

        # crop 좌표계(0~1) -> 원본 이미지 좌표
        lm[:, 0] = lm[:, 0] * size + x1
        lm[:, 1] = lm[:, 1] * size + y1
        return lm


# ---------- 졸음 상태 판정 (계획서 4.3) ----------
def decide_state(perclos, continuous_closed,
                 caution_perclos, warning_perclos, danger_sec):
    """히스테리시스 없는 기본 판정. 임계값은 인자로 받아 튜닝 가능."""
    if continuous_closed >= danger_sec:
        return "DANGER", (0, 0, 255)       # 빨강
    if perclos >= warning_perclos:
        return "WARNING", (0, 128, 255)    # 주황
    if perclos >= caution_perclos:
        return "CAUTION", (0, 255, 255)    # 노랑
    return "NORMAL", (0, 255, 0)           # 초록


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="입력 영상 파일 경로")
    src.add_argument("--camera", type=int, help="웹캠 인덱스 (예: 0)")
    ap.add_argument("--out", default="outputs/result.mp4", help="결과 영상 저장 경로")
    ap.add_argument("--csv", default="logs/perclos_log.csv", help="CSV 로그 경로")
    ap.add_argument("--yunet", default=YUNET_PATH)
    ap.add_argument("--pfld", default=PFLD_ONNX_PATH)
    ap.add_argument("--show", action="store_true", help="실시간 화면 표시")

    # --- 튜닝용 임계값 (검증하며 조정) ---
    ap.add_argument("--close-ratio", type=float, default=0.75,
                    help="상대 EAR 감김 판정선. 현재EAR/기준EAR 이 값 미만이면 눈 감김 (기본 0.75)")
    ap.add_argument("--calib-sec", type=float, default=3.0,
                    help="시작 후 기준 EAR 캘리브레이션 시간(초) (기본 3.0)")
    ap.add_argument("--window-sec", type=float, default=30.0,
                    help="PERCLOS 슬라이딩 윈도우 길이(초) (기본 30.0)")
    ap.add_argument("--caution-perclos", type=float, default=0.15,
                    help="주의 상태 PERCLOS 임계값 (기본 0.15)")
    ap.add_argument("--warning-perclos", type=float, default=0.30,
                    help="경고 상태 PERCLOS 임계값 (기본 0.30)")
    ap.add_argument("--danger-sec", type=float, default=2.0,
                    help="위험 판정 연속 눈 감김 시간(초) (기본 2.0)")
    args = ap.parse_args()

    # 입력 소스 열기
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("입력 소스를 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0  # 웹캠 등 fps 정보 없을 때 기본값
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    print(f"입력: {w}x{h} @ {fps:.1f}fps")

    # YuNet 얼굴 검출기 (OpenCV 4.5.4+ 기능)
    detector = cv2.FaceDetectorYN.create(
        args.yunet, "", (w, h),
        score_threshold=0.7, nms_threshold=0.3, top_k=5000
    )
    detector.setInputSize((w, h))

    pfld = PFLDLandmark(args.pfld)
    judge = EyeStateJudge(calib_sec=args.calib_sec, close_ratio=args.close_ratio)
    perclos_calc = PerclosCalculator(window_sec=args.window_sec)
    print(f"임계값: close_ratio={args.close_ratio}, calib={args.calib_sec}s, "
          f"window={args.window_sec}s, caution={args.caution_perclos}, "
          f"warning={args.warning_perclos}, danger={args.danger_sec}s")

    # 결과 영상 저장 준비
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    # CSV 로그 준비
    csv_file = open(args.csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["frame", "video_time", "face_valid", "ear", "relative_ear",
         "eye_closed", "continuous_closed", "perclos", "state"]
    )

    frame_idx = 0
    infer_times = []  # 추론 지연 측정 (랜드마크 부분)
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_time = frame_idx / fps  # 영상 기준 경과초 (핵심)

        # 얼굴 검출
        _, faces = detector.detect(frame)
        face_valid = faces is not None and len(faces) > 0

        ear_val = None
        rel_ear = None
        eye_closed = False
        landmarks = None

        if face_valid:
            # 가장 큰 얼굴 하나 선택
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            fb = faces[0]
            box = (int(fb[0]), int(fb[1]), int(fb[2]), int(fb[3]))

            t0 = time.time()
            landmarks = pfld.detect(frame, box)
            infer_times.append(time.time() - t0)

            if landmarks is not None:
                ear_val = compute_ear(landmarks)
                jr = judge.update(video_time, ear_val)
                rel_ear = jr["relative_ear"]
                eye_closed = jr["eye_closed"]

        # PERCLOS 갱신
        pr = perclos_calc.update(video_time, face_valid, eye_closed)
        state, color = decide_state(
            pr["perclos"], pr["continuous_closed"],
            args.caution_perclos, args.warning_perclos, args.danger_sec,
        )

        # 캘리브레이션 중이면 상태 대신 안내
        if not judge.is_calibrated():
            state = "CALIBRATING"
            color = (200, 200, 200)

        # ---- 오버레이 ----
        if landmarks is not None:
            for (px, py) in landmarks.astype(int):
                cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)
        if face_valid:
            x, y, bw, bh = box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

        cv2.putText(frame, f"State: {state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"PERCLOS: {pr['perclos']:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Closed: {pr['continuous_closed']:.1f}s", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if ear_val is not None:
            rel_txt = f"{rel_ear:.2f}" if rel_ear is not None else "-"
            cv2.putText(frame, f"EAR: {ear_val:.2f} (rel {rel_txt})", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame)

        csv_writer.writerow([
            frame_idx, f"{video_time:.3f}", int(face_valid),
            f"{ear_val:.4f}" if ear_val is not None else "",
            f"{rel_ear:.4f}" if rel_ear is not None else "",
            int(eye_closed), f"{pr['continuous_closed']:.3f}",
            f"{pr['perclos']:.4f}", state,
        ])

        if args.show:
            cv2.imshow("drowsiness", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    elapsed = time.time() - t_start
    cap.release()
    writer.release()
    csv_file.close()
    if args.show:
        cv2.destroyAllWindows()

    # 성능 요약
    proc_fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"\n처리 완료: {frame_idx} 프레임, {elapsed:.1f}초")
    print(f"전체 처리 FPS (end-to-end): {proc_fps:.1f}")
    if infer_times:
        arr = np.array(infer_times) * 1000  # ms
        print(f"랜드마크 추론 지연: 평균 {arr.mean():.1f}ms, "
              f"P95 {np.percentile(arr, 95):.1f}ms")
    print(f"결과 영상: {args.out}")
    print(f"CSV 로그: {args.csv}")


if __name__ == "__main__":
    main()
