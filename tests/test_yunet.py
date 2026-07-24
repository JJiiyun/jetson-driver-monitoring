import sys
import time
from pathlib import Path

import cv2


MODEL_PATH = Path("models/face_detector/yunet.onnx")
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


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
    cap.set(cv2.CAP_PROP_FPS, 30)

    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    while True:
        success, frame = cap.read()

        if not success:
            print("[ERROR] Failed to read camera frame.")
            break

        height, width = frame.shape[:2]
        detector.setInputSize((width, height))

        _, faces = detector.detect(frame)

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

        current_time = time.perf_counter()
        elapsed = current_time - previous_time
        previous_time = current_time

        if elapsed > 0:
            current_fps = 1.0 / elapsed
            smoothed_fps = (
                current_fps
                if smoothed_fps == 0
                else 0.9 * smoothed_fps + 0.1 * current_fps
            )

        cv2.putText(
            frame,
            f"FPS: {smoothed_fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("ZZM YuNet Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
