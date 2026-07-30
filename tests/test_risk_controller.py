import sys
import time
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drowsiness.actions import BuzzerPatternController, RiskEventPublisher
from drowsiness.risk_controller import (
    BuzzerMode,
    DrowsinessRiskController,
    RiskLevel,
)


def update(controller, timestamp, **overrides):
    values = {
        "eye_danger": False,
        "perclos_caution": False,
        "perclos_warning": False,
        "is_yawning": False,
        "valid_face": True,
    }
    values.update(overrides)
    return controller.update(timestamp=timestamp, **values)


class DrowsinessRiskControllerTest(unittest.TestCase):
    def test_repeated_yawns_enter_pre_drowsy(self):
        controller = DrowsinessRiskController()
        update(controller, 1.0, is_yawning=True)
        update(controller, 2.0, is_yawning=False)
        decision = update(controller, 10.0, is_yawning=True)

        self.assertEqual(decision.level, RiskLevel.PRE_DROWSY)
        self.assertEqual(decision.buzzer_mode, BuzzerMode.ALERT)
        self.assertFalse(decision.hazard_light)
        self.assertFalse(decision.stop_request)
        self.assertIn("REPEATED_YAWN", decision.reasons)

    def test_eye_danger_requests_emergency_actions(self):
        controller = DrowsinessRiskController()
        decision = update(controller, 1.0, eye_danger=True)

        self.assertEqual(decision.level, RiskLevel.DROWSY)
        self.assertEqual(decision.buzzer_mode, BuzzerMode.EMERGENCY)
        self.assertTrue(decision.hazard_light)
        self.assertTrue(decision.stop_request)

    def test_drowsy_requires_acknowledgement_and_safe_recovery(self):
        controller = DrowsinessRiskController(recovery_seconds=2.0)
        update(controller, 1.0, eye_danger=True)
        self.assertEqual(update(controller, 5.0).level, RiskLevel.DROWSY)

        controller.acknowledge()
        self.assertEqual(update(controller, 6.0).level, RiskLevel.DROWSY)
        decision = update(controller, 8.1)
        self.assertEqual(decision.level, RiskLevel.NORMAL)
        self.assertFalse(decision.stop_request)

    def test_publisher_fans_out_decisions(self):
        controller = DrowsinessRiskController()
        publisher = RiskEventPublisher()
        received = []
        publisher.subscribe(received.append)
        decision = update(controller, 1.0, perclos_caution=True)
        publisher.publish(decision)
        self.assertEqual(received, [decision])


class FakeDigitalOutput:
    def __init__(self):
        self.values = []
        self.closed = False

    def set_active(self, active):
        self.values.append(bool(active))

    def close(self):
        self.closed = True


class BuzzerPatternControllerTest(unittest.TestCase):
    def test_close_turns_output_off_and_releases_driver(self):
        output = FakeDigitalOutput()
        buzzer = BuzzerPatternController(output)
        controller = DrowsinessRiskController()
        buzzer.publish(update(controller, 1.0, eye_danger=True))
        time.sleep(0.02)
        buzzer.close()

        self.assertIn(True, output.values)
        self.assertFalse(output.values[-1])
        self.assertTrue(output.closed)


if __name__ == "__main__":
    unittest.main()
