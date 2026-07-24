import sys
import time

import cv2


CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30


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

    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    while True:
        success, frame = cap.read()

        if not success:
            print("[ERROR] Failed to read frame.")
            break

        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        previous_time = current_time

        if elapsed > 0:
            current_fps = 1.0 / elapsed

            if smoothed_fps == 0:
                smoothed_fps = current_fps
            else:
                smoothed_fps = (
                    0.9 * smoothed_fps
                    + 0.1 * current_fps
                )

        cv2.putText(
            frame,
            f"FPS: {smoothed_fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.imshow("ZZM USB Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
