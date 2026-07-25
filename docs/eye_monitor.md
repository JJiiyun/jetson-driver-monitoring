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

Optional thresholds:

```bash
python3 scripts/run_eye_monitor.py \
  --calibration-seconds 3 \
  --closed-ratio 0.70 \
  --danger-seconds 2
```

EAR is calculated from all six standard landmarks for each eye. The display
shows only four derived positions per eye: left, top, right, and bottom. The
mouth display shows only inner upper-lip landmark 62 and lower-lip landmark 66.
