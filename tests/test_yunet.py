import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import PerformanceLogger


MODEL_PATH = PROJECT_ROOT / "models/face_detector/yunet.onnx"
RESULTS_DIR = PROJECT_ROOT / "benchmark/results"
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
WARMUP_FRAMES = 30


def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return 1

    if not hasattr(cv2, "FaceDetectorYN"):
        print("[ERROR] FaceDetectorYN is unavailable.")
        print("OpenCV version:", cv2.__version__)
        return 1

    detector = cv2.FaceDetectorYN.create(
        str(MODEL_PATH),
        "",
        (FRAME_WIDTH, FRAME_HEIGHT),
        score_threshold=0.8,
        nms_threshold=0.3,
        top_k=5000,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("[ERROR] Cannot open USB camera.")
        return 1

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger = PerformanceLogger(
        backend="opencv_yunet_fp32",
        output_dir=RESULTS_DIR,
        warmup_frames=WARMUP_FRAMES,
        input_source=f"camera:{CAMERA_INDEX}",
        width=actual_width,
        height=actual_height,
        target_fps=TARGET_FPS,
    )

    while True:
        frame_started_at = time.perf_counter()

        capture_started_at = time.perf_counter()
        success, frame = cap.read()
        capture_ms = (
            time.perf_counter() - capture_started_at
        ) * 1000.0

        if not success:
            print("[ERROR] Failed to read camera frame.")
            break

        height, width = frame.shape[:2]
        detector.setInputSize((width, height))

        inference_started_at = time.perf_counter()
        _, faces = detector.detect(frame)
        inference_ms = (
            time.perf_counter() - inference_started_at
        ) * 1000.0
        face_count = 0 if faces is None else len(faces)

        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4].astype(int)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

                landmarks = face[4:14].reshape(5, 2).astype(int)

                for point_x, point_y in landmarks:
                    cv2.circle(
                        frame,
                        (point_x, point_y),
                        3,
                        (0, 0, 255),
                        -1,
                    )

                score = float(face[-1])

                cv2.putText(
                    frame,
                    f"Face: {score:.2f}",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        cv2.putText(
            frame,
            f"FPS: {logger.current_fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("ZZM YuNet Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        logger.record_frame(
            frame_started_at=frame_started_at,
            frame_finished_at=time.perf_counter(),
            capture_ms=capture_ms,
            inference_ms=inference_ms,
            face_count=face_count,
        )

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    summary = logger.write_csv()
    logger.print_summary(summary)
    print(f"Frame CSV: {logger.frame_csv_path}")
    print(f"Summary CSV: {logger.summary_csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
