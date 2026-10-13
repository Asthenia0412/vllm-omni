"""Small stdlib-only integration contracts; no torch/device imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import diagnose_qk_collective_boundaries as wrapper


class IntegrationTests(unittest.TestCase):
    def test_frozen_bodies_private_constructor_and_defaults(self):
        for name in ("run_arm", "instrumented_correctness"):
            original, adapted = getattr(wrapper.frozen, name), getattr(wrapper, name)
            self.assertIs(original.__code__, adapted.__code__)
            self.assertIsNot(original.__globals__, adapted.__globals__)
            self.assertEqual(original.__defaults__, adapted.__defaults__)
            self.assertEqual(original.__kwdefaults__ or {}, adapted.__kwdefaults__)
            different = {key for key in original.__globals__
                         if original.__globals__[key] is not adapted.__globals__[key]}
            self.assertEqual(different, {"InstrumentedCall"})
            self.assertIs(original.__globals__["InstrumentedCall"], wrapper.frozen.InstrumentedCall)
        self.assertIs(wrapper.InstrumentedCall.__call__, wrapper.frozen.InstrumentedCall.__call__)

    def test_row_mapping_does_not_overwrite_correctness_cases(self):
        trace = wrapper.BoundaryIntegerTrace("correctness")
        rows = []
        for case in range(24):
            trace.case_index = case
            for call in range(4):
                trace.row = call
                rows.append(trace.boundary_row())
        self.assertEqual(rows, list(range(96)))
        profile = wrapper.BoundaryIntegerTrace("profile")
        profile.row = 5759
        self.assertEqual(profile.boundary_row(), 5759)
        self.assertEqual(profile.rows, 5760)

    def test_compiled_code_pin_matches_imported_code_without_execution_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.py"
            path.write_text("def all_to_all_5D(x):\n    return x\n")
            namespace = {}
            exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), namespace)
            pin = wrapper.expected_collective_code(wrapper.replay_api.pin(path))
            actual = wrapper.observer_api.fingerprint(namespace["all_to_all_5D"])
            self.assertEqual({key: pin[key] for key in actual}, actual)

    def test_boundary_validator_rejects_incomplete_or_nonmonotonic_receipt(self):
        api = wrapper.observer_api
        pins = {"source_path": "/synthetic", "source_sha256": "0" * 64, "code_sha256": "1" * 64,
                "code_fingerprint_method": api.CODE_FINGERPRINT_METHOD}
        data = dict(schema=api.SCHEMA, diagnostic_only=True, performance_gate=False,
            row_count=1, marks=list(api.MARKS), columns=list(api.COLUMNS),
            intervals=[list(item) for item in api.INTERVALS], status_bits=api.STATUS,
            terminal_failure=False, all_rows_complete=True, pins=pins,
            status=[0], visited=[1], restored_original_bindings=[1], complete_rows=[True],
            data=[value for point in range(6) for value in (100 + point, 10 + point, point, point)])
        wrapper.validate_boundaries(data, rows=1, code_pin=pins)
        for key, value in (("restored_original_bindings", [0]), ("terminal_failure", True),
                           ("complete_rows", [False]), ("schema", wrapper.frozen.SCHEMA)):
            bad = copy.deepcopy(data)
            bad[key] = value
            with self.assertRaises(RuntimeError):
                wrapper.validate_boundaries(bad, rows=1, code_pin=pins)
        bad = copy.deepcopy(data)
        bad["data"][4] = 0
        with self.assertRaises(RuntimeError):
            wrapper.validate_boundaries(bad, rows=1, code_pin=pins)

    def test_old_diagnostic_receipt_cannot_authorize_new_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "rank0.json").write_text(json.dumps({
                "schema": wrapper.frozen.SCHEMA, "method": wrapper.frozen.METHOD,
                "mode": "correctness", "rank": 0, "passed": True}))
            with self.assertRaisesRegex(RuntimeError, "new diagnostic correctness binding differs"):
                wrapper.verify_diagnostic_correctness(directory, {"sha256": "synthetic"}, {}, {}, {}, {})


if __name__ == "__main__":
    unittest.main()
