"""CPU contract tests; unit metadata is not real-device matrix evidence."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

import check_qk_real_matrix as checker


class RealMatrixContractTest(unittest.TestCase):
    def test_ordinary_guards_survive_optimized_python(self):
        with self.assertRaisesRegex(RuntimeError, "guard deliberately false"):
            checker.need(False, "guard deliberately false")
        tree = ast.parse(Path(checker.__file__).read_text())
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))

    def test_case_order_refuses_missing_duplicate_and_layer_fabrication(self):
        cases = [dict(kv_len=length, probe=probe, layer=0) for length, probe in checker.EXPECTED]
        checker.validate_order(cases)  # Cardinality helper only; no admission.
        for bad in (cases[:-1], [*cases[:-1], cases[0]], list(reversed(cases))):
            with self.assertRaisesRegex(RuntimeError, "canonical history/probe order"):
                checker.validate_order(bad)
        bad = copy.deepcopy(cases)
        bad[0]["layer"] = 1
        with self.assertRaisesRegex(RuntimeError, "only captured layer0"):
            checker.validate_order(bad)

    def test_timing_refuses_stale_or_incomplete_correctness(self):
        # Negative receipt tests only. These dummy pins never enter a device path.
        matrix = {"sha256": "matrix", "ranks": []}
        manifest, sources = {"sha256": "candidate"}, {"harness": "source"}
        report = dict(schema=checker.SCHEMA, mode="correctness", passed=True, rank=0,
                      matrix_sha256="stale", manifest=manifest, sources=sources,
                      original_hccl_included=True, all40_payloads=False, real_payload_layers=[0], cases=[])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0.json"
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "fixture/build/source pins differ"):
                checker.verify_correctness(directory, matrix, manifest, sources)
            report["matrix_sha256"] = "matrix"
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "canonical history/probe order"):
                checker.verify_correctness(directory, matrix, manifest, sources)
            report["passed"] = False
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "successful same-rank correctness receipt"):
                checker.verify_correctness(directory, matrix, manifest, sources)

    def test_phase_is_barrier_then_sync_and_never_sample_loop(self):
        events = []
        class Distributed:
            @staticmethod
            def barrier():
                events.append("barrier")
        class NPU:
            @staticmethod
            def synchronize():
                events.append("sync")
        class Torch:
            npu = NPU()
        result = checker.phase(Distributed(), Torch(), "unit_boundary")
        self.assertEqual(events, ["barrier", "sync"])
        self.assertGreaterEqual(result["ready_ns"], result["enter_ns"])
        tree = ast.parse(Path(checker.__file__).read_text())
        loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                 and ast.unparse(node.iter) == "range(30)"]
        self.assertEqual(len(loops), 2)
        for loop in loops:
            self.assertFalse(any(isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Name) and node.func.id == "phase"
                or isinstance(node.func, ast.Attribute) and node.func.attr == "barrier")
                for node in ast.walk(loop)))


if __name__ == "__main__":
    unittest.main()
