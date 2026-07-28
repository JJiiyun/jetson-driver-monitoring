from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drowsiness.yawn_monitor import YawnMonitor


class YawnMonitorTest(unittest.TestCase):
    def test_duration_and_hysteresis(self) -> None:
        monitor = YawnMonitor(
            open_ratio=0.18,
            close_ratio=0.14,
            yawn_seconds=0.3,
        )

        opened = monitor.update(0.20, timestamp=1.0)
        self.assertFalse(opened.is_yawning)
        self.assertEqual(opened.open_seconds, 0.0)

        yawning = monitor.update(0.20, timestamp=1.3)
        self.assertTrue(yawning.is_yawning)
        self.assertEqual(yawning.status, "YAWNING")

        between_thresholds = monitor.update(0.16, timestamp=1.4)
        self.assertTrue(between_thresholds.is_yawning)

        closed = monitor.update(0.13, timestamp=1.5)
        self.assertFalse(closed.is_yawning)
        self.assertEqual(closed.open_seconds, 0.0)
        self.assertEqual(closed.status, "NORMAL")

    def test_reset_clears_open_duration(self) -> None:
        monitor = YawnMonitor(
            open_ratio=0.18,
            close_ratio=0.14,
            yawn_seconds=0.3,
        )
        monitor.update(0.20, timestamp=1.0)
        monitor.reset()

        restarted = monitor.update(0.20, timestamp=3.0)
        self.assertFalse(restarted.is_yawning)
        self.assertEqual(restarted.open_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
