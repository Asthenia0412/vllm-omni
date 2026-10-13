"""CPU-only negative checks for cross timing receipt bindings."""
import copy
import json
from pathlib import Path
import shlex
import tempfile
import unittest

import summarize_cross_real_matrix as summary


class SerialLaunchBinding(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.timing, self.correctness = self.root / "timing1", self.root / "correctness"
        self.matrix = {"manifest": {"path": str(self.root / "build.json")},
                       "capture_plan": {"path": str(self.root / "plan.json")},
                       "capture_audit": {"path": str(self.root / "audit.json")}}
        self.options = {"--capture-plan": self.matrix["capture_plan"]["path"],
                        "--matrix-audit": self.matrix["capture_audit"]["path"],
                        "--mode": "timing", "--round": "1", "--correctness-dir": str(self.correctness),
                        "--rank": "$replay_rank", "--device": "$replay_rank",
                        "--output": str(self.timing / "rank$replay_rank.json")}

    def tearDown(self):
        self.temp.cleanup()

    def write(self, mutate_result=None, duplicate="", loop=None):
        tokens = ["python", "recovery/check_cross_real_matrix.py"]
        for key, value in self.options.items():
            tokens.extend((key, value))
        script = ("set -e\n" + (loop or "for replay_rank in 0 1 2 3 4 5 6 7; do") + "\n"
                  + shlex.join(tokens) + duplicate + "\ndone")
        launch = dict(nonce="fresh", leader=dict(pid=1, start=2), command=["bash", "-c", script],
                      mode="timing", manifest=self.matrix["manifest"]["path"], fatal=None)
        result = dict(copy.deepcopy(launch), exit_code=0, remaining_owned_processes=[])
        if mutate_result:
            mutate_result(result)
        (self.root / "launch.json").write_text(json.dumps(launch))
        (self.root / "result.json").write_text(json.dumps(result))

    def validate(self):
        return summary.serial_launch(self.root / "result.json", self.timing, 1, self.matrix, self.correctness)

    def test_exact_serial_launcher_passes(self):
        self.write()
        self.assertEqual(self.validate()[0]["exit_code"], 0)

    def test_round_prefix_is_rejected(self):
        self.options["--round"] = "10"
        self.write()
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_output_prefix_is_rejected(self):
        self.options["--output"] = str(self.root / "timing1-other" / "rank$replay_rank.json")
        self.write()
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_changed_capture_is_rejected(self):
        self.options["--capture-plan"] = str(self.root / "other-plan.json")
        self.write()
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_launch_result_identity_is_bound(self):
        self.write(lambda result: result["leader"].update(pid=2))
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_duplicate_option_is_rejected(self):
        self.write(duplicate=" --round 1")
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_background_operation_is_rejected(self):
        self.write(duplicate=" &")
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_incomplete_rank_loop_is_rejected(self):
        self.write(loop="for replay_rank in 0 1 2 3 4 5 6; do")
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_nonzero_exit_is_rejected(self):
        self.write(lambda result: result.update(exit_code=1))
        with self.assertRaises(RuntimeError):
            self.validate()

    def test_remaining_process_is_rejected(self):
        self.write(lambda result: result.update(remaining_owned_processes=[{"pid": 12}]))
        with self.assertRaises(RuntimeError):
            self.validate()


if __name__ == "__main__":
    unittest.main()
