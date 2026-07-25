import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drowsiness import (
    EyeClosureMonitor,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    eye_display_points,
    mean_eye_aspect_ratio,
    mouth_display_points,
)


class EyeMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.landmarks = np.zeros((68, 2), dtype=np.float32)
        eye = np.asarray(
            (
                (0.0, 0.0),
                (1.0, -1.0),
                (3.0, -1.0),
                (4.0, 0.0),
                (3.0, 1.0),
                (1.0, 1.0),
            ),
            dtype=np.float32,
        )
        self.landmarks[list(RIGHT_EYE_INDICES)] = eye
        self.landmarks[list(LEFT_EYE_INDICES)] = eye + (10.0, 0.0)
        self.landmarks[62] = (7.0, 4.0)
        self.landmarks[66] = (7.0, 8.0)

    def test_mean_ear_uses_six_hidden_eye_points(self) -> None:
        mean_ear, right_ear, left_ear = mean_eye_aspect_ratio(
            self.landmarks
        )
        self.assertAlmostEqual(right_ear, 0.5)
        self.assertAlmostEqual(left_ear, 0.5)
        self.assertAlmostEqual(mean_ear, 0.5)

    def test_only_four_eye_points_and_two_mouth_points_are_displayed(
        self,
    ) -> None:
        right_points = eye_display_points(
            self.landmarks,
            RIGHT_EYE_INDICES,
        )
        left_points = eye_display_points(
            self.landmarks,
            LEFT_EYE_INDICES,
        )
        lip_points = mouth_display_points(self.landmarks)
        self.assertEqual(right_points.shape, (4, 2))
        self.assertEqual(left_points.shape, (4, 2))
        self.assertEqual(lip_points.shape, (2, 2))


class EyeClosureMonitorTest(unittest.TestCase):
    def test_calibration_and_continuous_closure(self) -> None:
        monitor = EyeClosureMonitor(
            calibration_seconds=3.0,
            closed_ratio=0.7,
            danger_seconds=2.0,
            min_calibration_samples=4,
        )

        monitor.update(0.30, timestamp=0.0)
        monitor.update(0.31, timestamp=1.0)
        monitor.update(0.29, timestamp=2.0)
        calibrated = monitor.update(0.30, timestamp=3.0)

        self.assertTrue(calibrated.calibrated)
        self.assertAlmostEqual(calibrated.baseline_ear, 0.30)
        self.assertEqual(calibrated.status, "NORMAL")

        just_closed = monitor.update(0.15, timestamp=4.0)
        self.assertEqual(just_closed.status, "EYES CLOSED")
        self.assertAlmostEqual(just_closed.closed_seconds, 0.0)

        danger = monitor.update(0.15, timestamp=6.1)
        self.assertEqual(danger.status, "DANGER")
        self.assertTrue(danger.is_danger)
        self.assertAlmostEqual(danger.closed_seconds, 2.1)

        reopened = monitor.update(0.30, timestamp=6.2)
        self.assertEqual(reopened.status, "NORMAL")
        self.assertEqual(reopened.closed_seconds, 0.0)

        missing_face = monitor.update(None, timestamp=6.3)
        self.assertEqual(missing_face.status, "NO FACE")
        self.assertTrue(missing_face.calibrated)
        self.assertAlmostEqual(missing_face.baseline_ear, 0.30)
        self.assertIsNone(missing_face.relative_ear)
        self.assertEqual(missing_face.closed_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
