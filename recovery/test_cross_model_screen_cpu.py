"""CPU-only negative guards for the source-pinned model-screen controller."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_cross_model_screen as controller


class CrossModelScreenTest(unittest.TestCase):
    def test_explicit_guards_survive_optimized_python(self):
        with self.assertRaisesRegex(ValueError, "negative guard"):
            controller.need(False, "negative guard")
        for name in ("run_cross_model_screen.py", "prepare_cross_model_screen.py"):
            tree = ast.parse(Path(__file__).with_name(name).read_text())
            self.assertFalse(any(isinstance(n, ast.Assert) for n in ast.walk(tree)))

    def test_order_is_exact_existing_six_run_protocol(self):
        self.assertEqual(controller.ORDER[:3], ("correctness-A", "correctness-B", "compare"))
        self.assertEqual(controller.ORDER[3:], ("timing-A1", "timing-B1", "timing-B2", "timing-A2", "timing-A3", "timing-B3"))

    def test_timing_has_no_capture_environment(self):
        plan = dict(base_environment={"ASCEND_RT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15"}, output="/example")
        self.assertEqual(controller.stage_environment(plan, "timing-A1"), plan["base_environment"])
        env = controller.stage_environment(plan, "correctness-B")
        self.assertEqual(env["LINGBOT_CAPTURE_DIR"], "/example/correctness-B/captures")
        self.assertEqual(env["LINGBOT_CAPTURE_RANK"], "-1")
        self.assertNotIn("LINGBOT_CAPTURE_DIR", plan["base_environment"])

    def test_selected_environment_never_copies_unrelated_secret(self):
        data = controller.selected_environment({"SECRET_TOKEN": "private", "TASK_QUEUE_ENABLE": "2", "LINGBOT_CAPTURE_DIR": "/case"})
        self.assertEqual(data, {"TASK_QUEUE_ENABLE": "2", "LINGBOT_CAPTURE_DIR": "/case"})

    def test_unexpected_request_command_is_refused_before_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            controller.write(run / "smoke-command.json", ["wrong command"])
            with self.assertRaisesRegex(ValueError, "Request command"):
                controller.validate_request(dict(port=7816, model="model", image={"path": "image"}), run, 80)

    def test_completed_stage_does_not_inherit_changed_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "plan.json"
            controller.write(plan_path, {"frozen": True})
            plan = dict(output=tmp, plan_path=str(plan_path))
            controller.write(Path(tmp) / "correctness-A/stage.json", dict(completed=True, stage="correctness-A", plan={"sha256": "old"}, evidence=[]))
            with self.assertRaisesRegex(ValueError, "another plan"):
                controller.verify_stage(plan, "correctness-A")


if __name__ == "__main__":
    unittest.main()
