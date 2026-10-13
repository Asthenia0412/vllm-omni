"""CPU-only admission/dispatch tests; no torch import and no device execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import check_cross_real_matrix as replay


class CrossReplayAdmission(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.sources = {"harness": {"sha256": "source"}}
        self.matrix = {"sha256": "matrix", "manifest": {"sha256": "build"}, "ranks": []}
        for rank in range(8):
            entries = [dict(identity=dict(rank=rank, request_id="actual", chunk=index // 4,
                                          kv_len=length, probe=probe, layer=0),
                            fixture={"path": f"case-{rank}-{length}-{probe}", "sha256": "case"})
                       for index, (length, probe) in enumerate(replay.EXPECTED)]
            self.matrix["ranks"].append(dict(cases=entries))
            receipt = dict(schema=replay.SCHEMA, mode="correctness", passed=True, rank=rank,
                           matrix_sha256="matrix", manifest=self.matrix["manifest"], sources=self.sources,
                           full_attention_forward=True, constructor_context="isolated_from_captured_config",
                           all40_payloads=False,
                           cases=[dict(**entry, passed=True, checks=list(replay.CHECKS),
                                       full_owner_sha256_verified=True) for entry in entries])
            self.write(rank, receipt)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, rank, value):
        (self.directory / f"rank{rank}.json").write_text(json.dumps(value))

    def change(self, rank, mutate):
        value = json.loads((self.directory / f"rank{rank}.json").read_text())
        mutate(value)
        self.write(rank, value)

    def verify(self):
        return replay.verify_correctness(self.directory, self.matrix, self.sources)

    def test_requires_every_actual_case(self):
        self.assertEqual(len(self.verify()), 8)
        self.change(7, lambda receipt: receipt["cases"].pop())
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_rejects_missing_rank(self):
        (self.directory / "rank7.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_rejects_wrong_matrix(self):
        self.change(3, lambda receipt: receipt.update(matrix_sha256="other"))
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_rejects_wrong_sources(self):
        self.change(3, lambda receipt: receipt.update(sources={}))
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_rejects_wrong_fixture(self):
        self.change(2, lambda receipt: receipt["cases"][2].update(fixture={"sha256": "other"}))
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_rejects_duplicate_history_probe(self):
        self.change(1, lambda receipt: receipt["cases"].__setitem__(1, receipt["cases"][0]))
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_rejects_bare_kernel(self):
        self.change(0, lambda receipt: receipt.update(full_attention_forward=False))
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_rejects_unverified_owner(self):
        self.change(6, lambda receipt: receipt["cases"][9].update(full_owner_sha256_verified=False))
        with self.assertRaises(RuntimeError):
            self.verify()

    def test_rejects_missing_finite_gate(self):
        self.change(5, lambda receipt: receipt["cases"][0]["checks"].remove("finite_outputs"))
        with self.assertRaises(RuntimeError):
            self.verify()


class Provider:
    def __init__(self, result):
        self.result = result

    def call(self, name, *args):
        return self.result


class FullDispatchChecks(unittest.TestCase):
    def setUp(self):
        self.output, self.operands = object(), (object(), object(), object())
        self.provider = Provider(self.output)

    def test_plain_and_combined_capture_and_restore(self):
        for arm in replay.ARMS:
            tail = (True,) if arm == "combined" else ()
            module = lambda *args: self.provider.call(replay.OPERATIONS[arm], *args, 128 ** -.5, *tail)
            output = replay.dispatch_checked(self.provider, module, arm, self.operands)
            self.assertIs(output, self.output)
            self.assertNotIn("call", self.provider.__dict__)

    def test_exception_restores_spy(self):
        def module(*args):
            raise ValueError("queued operation failed")
        with self.assertRaises(ValueError):
            replay.dispatch_checked(self.provider, module, "plain", self.operands)
        self.assertNotIn("call", self.provider.__dict__)

    def test_fallback_is_not_accepted(self):
        with self.assertRaises(RuntimeError):
            replay.dispatch_checked(self.provider, lambda *args: self.output, "plain", self.operands)
        self.assertNotIn("call", self.provider.__dict__)

    def test_wrong_operation_is_not_accepted(self):
        module = lambda *args: self.provider.call("other", *args, 128 ** -.5)
        with self.assertRaises(RuntimeError):
            replay.dispatch_checked(self.provider, module, "plain", self.operands)

    def test_caller_copy_is_not_accepted(self):
        def module(*args):
            self.provider.call("cross", *args, 128 ** -.5)
            return object()
        with self.assertRaises(RuntimeError):
            replay.dispatch_checked(self.provider, module, "plain", self.operands)

    def test_changed_operand_is_not_accepted(self):
        module = lambda *args: self.provider.call("cross", object(), args[1], args[2], 128 ** -.5)
        with self.assertRaises(RuntimeError):
            replay.dispatch_checked(self.provider, module, "plain", self.operands)

    def test_cpu_import_has_no_torch(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("torch_npu", sys.modules)


if __name__ == "__main__":
    unittest.main()
