"""Stdlib regression checks for dependency drift before device/service launch.

The fixture contains small fake files, never a loadable library or NPU tensor.
These checks establish preflight refusal behavior, not device correctness.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_stageB_cpp_operator as controller


class Preflight(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.compiled = {
            "schema": "lingbot_stageB_guard_cpu_build_v2", "namespace": "lingbot_stageB_guard_v2",
            "registration": True, "npu_initialized": False, "device_admitted": False,
            "model_admitted": False, "performance_admitted": False,
            "v1_qualification_inherited": False, "guard_timing": "per_layer_before_consumption",
            "torch": "2.10.0+cpu", "torch_npu": "2.10.0.post4", "cxx11_abi": 1,
            "sdk": "/fake/cann-9.1.0", "compiler_command": ["fake-compiler", "-shared"],
            "cpu_rejection_cases": [
                {"name": "Frame rejects CPU before NPU StorageImpl access", "passed": True},
                {"name": "Entry rejects CPU before NPU StorageImpl access", "passed": True}],
        }
        for key in ("source", "library", "builder", "provenance", "source_contract_tests"):
            self.compiled[key] = self.file(key, "fixture " + key)
        self.compiled["abi_headers"] = [self.file("header", "fixture header")]
        self.compiled["abi_libraries"] = [self.file("abi-so", "fixture abi library")]
        self.terminal = dict(schema=self.compiled["schema"], compiler_command=self.compiled["compiler_command"],
                             compilation_succeeded=True, exit_code=0, error=None, registration_attempted=False)
        self.compiled["compile_result"] = self.json_file("terminal.json", self.terminal)
        self.native = {"runtime_sources": [self.file("runtime.py", "# fake runtime source\n")]}
        self.runtime = {key: copy.deepcopy(self.compiled[key]) for key in
                        ("source", "library", "abi_headers", "abi_libraries", "cxx11_abi", "torch", "torch_npu", "sdk")}
        self.runtime.update(schema="lingbot_stageB_guard_runtime_v2", namespace="lingbot_stageB_guard_v2",
                            runtime_sources=copy.deepcopy(self.native["runtime_sources"]),
                            device_admitted=False, model_admitted=False, historical_admission_inherited=False,
                            v1_qualification_inherited=False, guard_timing="per_layer_before_consumption")
        self.plan = dict(schema="lingbot_actual_stageB_cpp_operator_plan_v2", worktree=str(Path.cwd()))
        self.freeze()

    def file(self, name, text):
        path = self.root / name
        path.write_text(text)
        return controller.pin(path)

    def json_file(self, name, value):
        return self.file(name, json.dumps(value, sort_keys=True))

    def freeze(self):
        self.plan["manifest"] = self.json_file("native.json", self.native)
        self.plan["cpp_compilation"] = self.json_file("build.json", self.compiled)
        self.runtime["compilation_ancestor"] = copy.deepcopy(self.plan["cpp_compilation"])
        self.plan["cpp_manifest"] = self.json_file("runtime.json", self.runtime)

    def verify(self):
        controller.verify_cpp_binding(self.plan)

    def test_complete_chain_passes_without_importing_torch(self):
        self.verify()
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("torch_npu", sys.modules)

    def test_file_drift_rejected_despite_unchanged_receipts(self):
        pins = [self.compiled[key] for key in
                ("source", "library", "builder", "provenance", "source_contract_tests", "compile_result")]
        pins += self.compiled["abi_headers"] + self.compiled["abi_libraries"] + self.native["runtime_sources"]
        for item in pins:
            with self.subTest(path=item["path"]):
                path = Path(item["path"])
                before = path.read_bytes()
                try:
                    path.write_bytes(before + b"changed")
                    with self.assertRaisesRegex(ValueError, "dependency changed before launch"):
                        self.verify()
                finally:
                    path.write_bytes(before)

    def test_missing_dependency_rejected(self):
        Path(self.compiled["abi_libraries"][0]["path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_runtime_cannot_select_another_validly_pinned_library(self):
        self.runtime["library"] = self.file("other-library", "different library")
        self.freeze()
        with self.assertRaisesRegex(ValueError, "identity differs: library"):
            self.verify()

    def test_runtime_fixed_field_drift_rejected(self):
        for key, value in (("torch", "other"), ("torch_npu", "other"),
                           ("cxx11_abi", 0), ("sdk", "/different/sdk")):
            with self.subTest(key=key):
                original = self.runtime[key]
                self.runtime[key] = value
                self.freeze()
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    self.verify()
                self.runtime[key] = original

    def test_ancestor_cannot_be_replaced_even_with_identical_build_contents(self):
        self.runtime["compilation_ancestor"] = self.json_file("other-build.json", self.compiled)
        self.plan["cpp_manifest"] = self.json_file("runtime.json", self.runtime)
        with self.assertRaisesRegex(ValueError, "exact compilation ancestor"):
            self.verify()

    def test_receipt_tampering_rejected(self):
        for key in ("cpp_manifest", "cpp_compilation", "manifest"):
            with self.subTest(key=key):
                path = Path(self.plan[key]["path"])
                before = path.read_bytes()
                try:
                    path.write_bytes(before + b" ")
                    with self.assertRaisesRegex(ValueError, "binding receipt changed"):
                        self.verify()
                finally:
                    path.write_bytes(before)

    def test_incomplete_runtime_coverage_rejected(self):
        self.runtime["runtime_sources"] = []
        self.freeze()
        with self.assertRaisesRegex(ValueError, "coverage differs or is empty"):
            self.verify()

    def test_empty_abi_coverage_rejected_in_both_receipts(self):
        for key in ("abi_headers", "abi_libraries"):
            with self.subTest(key=key):
                previous = self.compiled[key]
                self.compiled[key] = self.runtime[key] = []
                self.freeze()
                with self.assertRaisesRegex(ValueError, "coverage differs or is empty"):
                    self.verify()
                self.compiled[key] = self.runtime[key] = previous

    def test_no_inherited_admission_or_npu_initialization(self):
        cases = [(self.runtime, key, True) for key in
                 ("device_admitted", "model_admitted", "historical_admission_inherited", "v1_qualification_inherited")]
        cases += [(self.compiled, key, True) for key in
                  ("device_admitted", "model_admitted", "performance_admitted", "v1_qualification_inherited", "npu_initialized")]
        cases += [(self.compiled, "registration", False), (self.runtime, "guard_timing", "once_per_forward")]
        for record, key, value in cases:
            with self.subTest(key=key):
                previous = record[key]
                record[key] = value
                self.freeze()
                with self.assertRaises(ValueError):
                    self.verify()
                record[key] = previous

    def test_compiler_terminal_and_cpu_rejections_required(self):
        self.terminal["exit_code"] = 1
        self.compiled["compile_result"] = self.json_file("terminal.json", self.terminal)
        self.freeze()
        with self.assertRaisesRegex(ValueError, "compilation terminal differs"):
            self.verify()
        self.terminal["exit_code"] = 0
        self.compiled["compile_result"] = self.json_file("terminal.json", self.terminal)
        self.compiled["cpu_rejection_cases"][1]["passed"] = False
        self.freeze()
        with self.assertRaisesRegex(ValueError, "CPU refusal evidence incomplete"):
            self.verify()

    def test_run_rejects_library_drift_before_any_subprocess_or_launch_intent(self):
        Path(self.compiled["library"]["path"]).write_text("changed after preparation")
        plan_file = self.root / "plan.json"
        plan_file.write_text(json.dumps(self.plan))
        with patch.object(controller, "source_check"), \
             patch.object(controller.subprocess, "run") as subprocess_run, \
             patch.object(controller.subprocess, "Popen") as popen, \
             patch.object(controller, "write") as write:
            with self.assertRaisesRegex(ValueError, "dependency changed before launch"):
                controller.run(plan_file, 1)
            subprocess_run.assert_not_called()
            popen.assert_not_called()
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
