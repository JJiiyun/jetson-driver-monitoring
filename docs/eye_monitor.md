# EAR eye monitor

## Required local model files

Model binaries are ignored by Git and must be copied to the Jetson manually.

```text
jetson-driver-monitoring/
├── models/
│   ├── face_detector/
│   │   └── yunet.onnx
│   └── landmark/
│       └── pfld_sim.onnx
├── drowsiness/
│   ├── __init__.py
│   ├── calibration.py
│   ├── detectors.py
│   └── metrics.py
├── scripts/
│   └── run_eye_monitor.py
└── tests/
    └── test_eye_monitor.py
```

## Run

Keep both eyes naturally open during the first three seconds.

```bash
cd ~/jetson-driver-monitoring
source ~/zzmenv/bin/activate
python3 scripts/run_eye_monitor.py
```

`run_eye_monitor.py` is for live camera inference. To process a clean,
unannotated source video with the original single-threshold behavior:

```bash
python3 scripts/run_video_inference.py data/minjin_test_1_raw.mp4
```

To process the same video with the FSM and EAR hysteresis:

```bash
python3 scripts/run_video_inference_FSM.py data/minjin_test_1_raw.mp4
```

Do not use a video previously produced by `run_eye_monitor.py` as the input:
its text, bounding boxes, and landmarks are already burned into the pixels.
Offline results are written separately under `outputs/video_inference/`.

- `q`: quit
- `r`: restart the three-second EAR calibration

The annotated video is saved from the first captured frame through the frame
where `q` is pressed:

```text
outputs/videos/<run_id>.avi
```

The same `run_id` links the video to its performance and EAR data:

```text
benchmark/results/<run_id>_frames.csv
benchmark/results/<run_id>_summary.csv
```

The frame CSV contains the performance columns plus `ear`, `right_ear`,
`left_ear`, `baseline_ear`, `relative_ear`, `closed_threshold`,
`reopen_threshold`, `is_eye_closed`, `closed_seconds`, and `eye_state`.

Optional thresholds:

```bash
python3 scripts/run_eye_monitor.py \
  --calibration-seconds 3 \
  --closed-ratio 0.70 \
  --reopen-ratio 0.80 \
  --danger-seconds 2
```

The state machine transitions through `CALIBRATING`, `NORMAL`,
`EYES CLOSED`, `DANGER`, and `NO FACE`. Hysteresis uses separate thresholds:
the eyes close below `closed-ratio` and reopen at or above `reopen-ratio`.
Values between the thresholds preserve the previous open/closed state.

The on-screen status panel shows the current state, EAR, close/open
thresholds, continuous closure time, PERCLOS, face score, and processing FPS.

EAR is calculated from all six standard landmarks for each eye. The display
shows only four derived positions per eye: left, top, right, and bottom. The
mouth display shows only inner upper-lip landmark 62 and lower-lip landmark 66.
