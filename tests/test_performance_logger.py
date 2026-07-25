import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.performance_logger import PerformanceLogger, _percentile


class PerformanceLoggerTest(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(_percentile([], 0.95), None)
        self.assertEqual(_percentile([10.0], 0.95), 10.0)
        self.assertAlmostEqual(
            _percentile([10.0, 20.0, 30.0, 40.0], 0.95),
            38.5,
        )

    def test_summary_excludes_warmup_frames(self):
        logger = PerformanceLogger(
            backend="test_backend",
            warmup_frames=1,
        )

        logger.record_frame(1.000, 1.100, 20.0, 30.0, 1)
        logger.record_frame(2.000, 2.050, 10.0, 20.0, 2)
        logger.record_frame(3.000, 3.100, 20.0, 40.0, 0)

        summary = logger.summary()

        self.assertEqual(summary["total_frames"], 3)
        self.assertEqual(summary["measured_frames"], 2)
        self.assertAlmostEqual(summary["measured_duration_s"], 0.15)
        self.assertAlmostEqual(summary["end_to_end_fps"], 2 / 0.15)
        self.assertAlmostEqual(summary["inference_mean_ms"], 30.0)
        self.assertAlmostEqual(summary["frame_time_p95_ms"], 97.5)
        self.assertAlmostEqual(summary["average_face_count"], 1.0)

    def test_write_csv_creates_frame_and_summary_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            logger = PerformanceLogger(
                backend="test backend",
                output_dir=temporary_directory,
                run_name="unit test",
                warmup_frames=0,
                width=640,
                height=480,
                target_fps=30,
            )
            logger.record_frame(1.000, 1.040, 10.0, 20.0, 1)

            summary = logger.write_csv()

            self.assertTrue(logger.frame_csv_path.exists())
            self.assertTrue(logger.summary_csv_path.exists())
            self.assertEqual(summary["measured_frames"], 1)

            with logger.frame_csv_path.open(
                newline="",
                encoding="utf-8",
            ) as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["backend"], "test backend")
            self.assertEqual(rows[0]["face_count"], "1")
            self.assertEqual(rows[0]["is_warmup"], "False")


if __name__ == "__main__":
    unittest.main()
