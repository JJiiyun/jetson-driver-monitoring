from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.compare_backends import compare, validate_alignment


def make_row(
    frame_index: int,
    offset: float,
    state: str = "NORMAL",
) -> dict[str, str]:
    landmarks = np.zeros((68, 2), dtype=np.float32) + offset
    return {
        "frame_index": str(frame_index),
        "input_source": "video:test.mp4",
        "width": "640",
        "height": "360",
        "target_fps": "30.0",
        "is_warmup": "False",
        "ear": str(0.25 + offset),
        "mar": str(0.10 + offset),
        "eye_state": state,
        "is_eye_closed": "False",
        "is_yawning": "False",
        "pfld_inference_ms": str(20.0 + offset),
        "inference_ms": str(80.0 + offset),
        "frame_time_ms": str(100.0 + offset),
        "landmarks_json": json.dumps(landmarks.tolist()),
    }


class CompareBackendsTest(unittest.TestCase):
    def test_compare_calculates_accuracy_and_speedup(self):
        fp32 = {1: make_row(1, 0.0), 2: make_row(2, 0.0)}
        fp16 = {1: make_row(1, 0.5), 2: make_row(2, 0.5)}
        for row in fp16.values():
            row["pfld_inference_ms"] = "10.0"

        frames, summary = compare(fp32, fp16)

        self.assertEqual(len(frames), 2)
        self.assertAlmostEqual(
            summary["landmark_mean_error_px"], 2**0.5 / 2
        )
        self.assertAlmostEqual(summary["ear_mean_absolute_error"], 0.5)
        self.assertAlmostEqual(summary["mar_mean_absolute_error"], 0.5)
        self.assertEqual(summary["fsm_state_match_rate"], 1.0)
        self.assertEqual(summary["yawn_state_match_rate"], 1.0)
        self.assertEqual(summary["pfld_fp16_speedup"], 2.0)

    def test_alignment_rejects_different_frame_sets(self):
        with self.assertRaisesRegex(ValueError, "Frame sets differ"):
            validate_alignment(
                {1: make_row(1, 0.0)},
                {2: make_row(2, 0.0)},
            )


if __name__ == "__main__":
    unittest.main()
