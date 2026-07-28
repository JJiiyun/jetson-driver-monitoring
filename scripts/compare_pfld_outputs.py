import time
import cv2
import numpy as np

from drowsiness.detectors import YuNetFaceDetector
from drowsiness.tensorrt_pfld import TensorRTPFLDLandmarkDetector

video_path = "data/converted/final_test.mp4"

capture = cv2.VideoCapture(video_path)
if not capture.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

face_detector = YuNetFaceDetector(
    "models/face_detector/yunet.onnx",
    input_size=(640, 360),
)
landmark_detector = TensorRTPFLDLandmarkDetector(
    "models/engine/pfld_sim_fp16.engine"
)

try:
    for frame_index in range(300):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Failed to read video frame.")

        detection = face_detector.detect_largest(frame)
        if detection is None:
            continue

        # 워밍업
        for _ in range(5):
            landmark_detector.detect(frame, detection.box)

        started = time.perf_counter()
        landmarks, crop_box = landmark_detector.detect(
            frame,
            detection.box,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        print("Frame:", frame_index)
        print("Face score:", detection.score)
        print("Face box:", detection.box)
        print("Crop box:", crop_box)
        print("Landmark shape:", landmarks.shape)
        print("Finite output:", np.isfinite(landmarks).all())
        print(f"TensorRT PFLD time: {elapsed_ms:.3f} ms")
        print("[PASS] MP4 TensorRT test completed.")
        break
    else:
        raise RuntimeError("No face found in the first 300 frames.")
finally:
    landmark_detector.close()
    capture.release()
PY