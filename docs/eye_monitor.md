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
