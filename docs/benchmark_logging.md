# FPS and latency logging

`benchmark/performance_logger.py` records reusable performance measurements
for camera, OpenCV FP32, and future TensorRT FP16 pipelines.

## Run benchmarks

Run from the project root:

```bash
# Camera-only baseline (inference latency is zero)
python3 tests/test_camera.py

# Camera plus YuNet FP32 inference
python3 tests/test_yunet.py
```

Press `q` to stop. Two files are then written under `benchmark/results/`:

- `*_frames.csv`: one row per frame, including the warm-up frames
- `*_summary.csv`: one row per run, excluding the warm-up frames

The first 30 frames are warm-up frames by default. They have
`is_warmup=True` in the frame CSV and are not used for summary statistics.

## Measurement definitions

| Field | Definition |
|---|---|
| `capture_ms` | Time spent in `cap.read()` |
| `inference_ms` | Time spent in the model call |
| `processing_ms` | Remaining frame work, including drawing and display |
| `frame_time_ms` | Camera input through display and key handling |
| `instant_fps` | `1000 / frame_time_ms` for the frame |
| `end_to_end_fps` | Measured frames divided by measured duration |

The FPS shown in the window uses up to the latest 30 recorded frames. It is
only a live indicator and can include warm-up frames. Use `end_to_end_fps` in
the summary CSV for FP32/FP16 comparisons.

## Fair comparison checklist

- Use the same input video or camera conditions.
- Use the same resolution, target FPS, power mode, and GUI setting.
- Exclude the same number of warm-up frames.
- Run each configuration at least three times.
- Keep `backend` names distinct, such as `opencv_yunet_fp32` and
  `tensorrt_fp16`.
