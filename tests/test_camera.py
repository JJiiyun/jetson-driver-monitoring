import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import PerformanceLogger


RESULTS_DIR = PROJECT_ROOT / "benchmark/results"
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
WARMUP_FRAMES = 30


def main() -> int:
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera index: {CAMERA_INDEX}")
    print(f"Resolution: {actual_width}x{actual_height}")
    print(f"Requested FPS: {TARGET_FPS}")
    print(f"Reported FPS: {actual_fps:.1f}")
    print("Press q to quit.")

    logger = PerformanceLogger(
        backend="opencv_camera_only",
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
            print("[ERROR] Failed to read frame.")
            break

        cv2.putText(
            frame,
            f"FPS: {logger.current_fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.imshow("ZZM USB Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF

        logger.record_frame(
            frame_started_at=frame_started_at,
            frame_finished_at=time.perf_counter(),
            capture_ms=capture_ms,
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
