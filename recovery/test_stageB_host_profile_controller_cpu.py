"""Stdlib-only host-profile receipt/preflight checks using temporary fake files.

No framework import, library load, subprocess, device or model work is allowed
by these fixtures. They test diagnostic classification and refusal boundaries.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_stageB_host_profile as controller
import run_stageB_cpp_operator as frozen_controller


class HostProfileController(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.compiled = dict(
            schema="lingbot_stageB_guard_cpu_build_v2", namespace="lingbot_stageB_guard_v2",
            registration=True, npu_initialized=False, device_admitted=False, model_admitted=False,
            performance_admitted=False, v1_qualification_inherited=False,
            guard_timing="per_layer_before_consumption", torch="2.10.0+cpu", torch_npu="2.10.0.post4",
            cxx11_abi=1, sdk="/fake/cann-9.1.0", compiler_command=["fake-compiler", "-shared"],
            cpu_rejection_cases=[
                dict(name="Frame rejects CPU before NPU StorageImpl access", passed=True),
                dict(name="Entry rejects CPU before NPU StorageImpl access", passed=True)])
        for key in ("source", "library", "builder", "provenance", "source_contract_tests"):
            self.compiled[key] = self.file("cpp-" + key, "fixture " + key)
        self.compiled["abi_headers"] = [self.file("cpp-header", "fixture header")]
        self.compiled["abi_libraries"] = [self.file("cpp-abi-so", "fixture ABI library")]
        self.compiled["compile_result"] = self.json_file("compile-result.json", dict(
            schema=self.compiled["schema"], compiler_command=self.compiled["compiler_command"],
            compilation_succeeded=True, exit_code=0, error=None, registration_attempted=False))
        self.profile_pin = self.file("host-profile.py", "# fake host profile\n")
        self.screen_pin = self.file("screen.py", "# fake original screen\n")
        self.core_pin = self.file("lingbot_fixed_modulation_stageB_cpp.py", "# fake CPP core\n")
        self.native = dict(runtime_sources=[self.profile_pin, self.screen_pin, self.core_pin],
                           abi_headers=[self.file("native-header", "fixture native header")],
                           kernel_sources=[self.file("native-kernel.cpp", "fixture native kernel")])
        for key in ("source", "library", "kernels"):
            self.native[key] = self.file("native-" + key, "fixture native " + key)
        self.runtime = {key: copy.deepcopy(self.compiled[key]) for key in (
            "source", "library", "abi_headers", "abi_libraries", "cxx11_abi", "torch", "torch_npu", "sdk")}
        self.runtime.update(schema="lingbot_stageB_guard_runtime_v2", namespace="lingbot_stageB_guard_v2",
                            runtime_sources=copy.deepcopy(self.native["runtime_sources"]),
                            device_admitted=False, model_admitted=False, historical_admission_inherited=False,
                            v1_qualification_inherited=False, guard_timing="per_layer_before_consumption")
        self.stage = self.root / "profile1"
        self.stage.mkdir()
        self.plan = dict(
            schema="lingbot_actual_stageB_host_profile_plan_v1", worktree=str(Path.cwd()), output=str(self.root),
            histories=list(controller.HISTORIES), probes=4, ranks=8, actual_layers=40, profile_iterations=20,
            diagnostic_only=True, performance_admitted=False, model_admitted=False,
            speed_gate_eligible=False, model_timing_eligible=False, final_acceptance=False,
            profile_source=self.profile_pin, screen_source=self.screen_pin,
            base_environment={"LINGBOT_STAGEB_HOST_PROFILE": "1", "VLLM_LINGBOT_NPU_ASYNC_METADATA": "0"},
            configs={"B": self.file("config-B.yaml", "fake CPPv2 config")}, harness=[], benchmark=[], protected=[],
            launch_scripts=[self.file("profile1/launch.sh", "# fake launch\n")])
        for key in ("parent_plan", "parent_baseline_config", "env_script", "integrity", "image"):
            self.plan[key] = self.file(key, "fixture " + key)
        self.freeze()
        stats = dict(function=dict(filename=self.screen_pin["path"], line=7, name="invoke"),
                     primitive_calls=20, total_calls=20, tottime_s=0.002, cumtime_s=0.003)
        method = dict(stats, callers=[dict(stats, function=dict(filename="driver.py", line=8, name="run"))])
        arm = dict(enqueue_calls=20, actual_layers_per_call=40,
                   cprofile_total_tottime_s=0.004, methods=[method])
        self.report = dict(
            schema="lingbot_actual_stageB_host_profile_v1", rank=0, nonce="fresh-owned-nonce", passed=True,
            actual_layers=40, actual_histories=list(controller.HISTORIES), actual_probes=4, profile_iterations=20,
            diagnostic_only=True, performance_admitted=False, model_admitted=False,
            speed_gate_eligible=False, model_timing_eligible=False,
            source=self.profile_pin["path"], source_sha256=self.profile_pin["sha256"], screen_source=self.screen_pin,
            cases=[dict(history=length, probe=probe, correctness=True,
                        arms={"native": copy.deepcopy(arm), "stageB": copy.deepcopy(arm)})
                   for length in controller.HISTORIES for probe in range(4)])
        for case in self.report["cases"]:
            for name, calls in (("prepare_forward", 20), ("get", 800)):
                case["arms"]["stageB"]["methods"].append(dict(
                    function=dict(filename=self.core_pin["path"], line=123, name=name),
                    primitive_calls=calls, total_calls=calls, tottime_s=0.001, cumtime_s=0.002, callers=[]))

    def file(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return controller.pin(path)

    def json_file(self, name, value):
        return self.file(name, json.dumps(value, sort_keys=True))

    def freeze(self):
        self.plan["manifest"] = self.json_file("native.json", self.native)
        self.plan["cpp_compilation"] = self.json_file("build.json", self.compiled)
        self.runtime["compilation_ancestor"] = self.plan["cpp_compilation"]
        self.plan["cpp_manifest"] = self.json_file("runtime.json", self.runtime)

    def validate(self, report=None):
        controller.validate_rank(self.plan, self.report if report is None else report, 0, "fresh-owned-nonce")

    def test_diagnostic_receipt_and_binding_pass_without_framework_import(self):
        controller.verify_inputs(self.plan)
        self.validate()
        self.assertIs(controller.verify_cpp_binding, frozen_controller.verify_cpp_binding)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("torch_npu", sys.modules)

    def test_formal_operator_receipt_cannot_be_relabelled(self):
        formal = copy.deepcopy(self.report)
        formal["schema"] = "lingbot_actual_stageB_full_region_screen_v1"
        with self.assertRaisesRegex(ValueError, "host-profile evidence"):
            self.validate(formal)
        for name, value in (("warmups", 20), ("wall_measurements", 30), ("event_measurements", 30)):
            with self.subTest(name=name):
                changed = copy.deepcopy(self.report)
                changed[name] = value
                with self.assertRaisesRegex(ValueError, "Formal wall/event/speed"):
                    self.validate(changed)
        for name in ("wall_ns", "event_ms", "speed_gate"):
            with self.subTest(name=name):
                changed = copy.deepcopy(self.report)
                changed["cases"][0]["arms"]["stageB"][name] = [1] * 30
                with self.assertRaisesRegex(ValueError, "Formal wall/event/speed"):
                    self.validate(changed)

    def test_every_no_admission_flag_is_mandatory(self):
        for key in controller.INELIGIBLE:
            for replacement in (True, None, 0):
                with self.subTest(key=key, value=replacement):
                    changed = copy.deepcopy(self.report)
                    changed[key] = replacement
                    with self.assertRaisesRegex(ValueError, "cannot admit"):
                        self.validate(changed)
        self.report["diagnostic_only"] = False
        with self.assertRaisesRegex(ValueError, "cannot admit"):
            self.validate()

    def test_wrong_nonce_rank_and_incomplete_run_rejected(self):
        for key, value in (("nonce", "another-service"), ("rank", 1), ("rank", False), ("passed", False)):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.report)
                changed[key] = value
                with self.assertRaisesRegex(ValueError, "host-profile evidence"):
                    self.validate(changed)

    def test_missing_or_reordered_actual_cases_rejected(self):
        for cases in (self.report["cases"][:-1], list(reversed(self.report["cases"])),
                      self.report["cases"][:-1] + [self.report["cases"][0]]):
            with self.subTest(size=len(cases)):
                changed = dict(self.report, cases=cases)
                with self.assertRaisesRegex(ValueError, "history/probe matrix"):
                    self.validate(changed)

    def test_profile_scope_and_correctness_rejected(self):
        for key, value in (("actual_layers", 39), ("actual_probes", 3), ("profile_iterations", 30),
                           ("actual_layers", 40.0), ("actual_histories", [4680])):
            with self.subTest(key=key):
                changed = dict(self.report, **{key: value})
                with self.assertRaisesRegex(ValueError, "corpus/protocol"):
                    self.validate(changed)
        self.report["cases"][0]["correctness"] = False
        with self.assertRaisesRegex(ValueError, "arm/correctness"):
            self.validate()

    def test_arm_call_count_and_layers_are_exact(self):
        for key, value in (("enqueue_calls", 19), ("enqueue_calls", 20.0), ("actual_layers_per_call", 39)):
            changed = copy.deepcopy(self.report)
            changed["cases"][0]["arms"]["stageB"][key] = value
            with self.assertRaisesRegex(ValueError, "complete40-layer"):
                self.validate(changed)

    def test_helper_and_screen_identity_required(self):
        for key, value in (("source", "/other/helper.py"), ("source_sha256", "0" * 64),
                           ("screen_source", self.profile_pin)):
            with self.subTest(key=key):
                changed = dict(self.report, **{key: value})
                with self.assertRaisesRegex(ValueError, "source identity"):
                    self.validate(changed)

    def test_method_and_caller_counts_reject_boolean_negative_or_reversed(self):
        for is_caller in (False, True):
            for key, value in (("primitive_calls", True), ("primitive_calls", -1),
                               ("primitive_calls", 21), ("total_calls", 1.0)):
                with self.subTest(caller=is_caller, key=key):
                    changed = copy.deepcopy(self.report)
                    target = changed["cases"][0]["arms"]["native"]["methods"][0]
                    if is_caller:
                        target = target["callers"][0]
                    target[key] = value
                    with self.assertRaisesRegex(ValueError, "call counts"):
                        self.validate(changed)

    def test_nonfinite_negative_or_boolean_times_rejected(self):
        for is_caller in (False, True):
            for field in ("tottime_s", "cumtime_s"):
                for bad in (-1, float("nan"), float("inf"), True):
                    with self.subTest(caller=is_caller, field=field, value=bad):
                        changed = copy.deepcopy(self.report)
                        target = changed["cases"][0]["arms"]["native"]["methods"][0]
                        if is_caller:
                            target = target["callers"][0]
                        target[field] = bad
                        with self.assertRaisesRegex(ValueError, "cProfile times"):
                            self.validate(changed)
        for bad in (-1, float("nan"), float("inf"), True):
            changed = copy.deepcopy(self.report)
            changed["cases"][0]["arms"]["native"]["cprofile_total_tottime_s"] = bad
            with self.assertRaisesRegex(ValueError, "total time"):
                self.validate(changed)

    def test_duplicate_or_empty_method_records_rejected(self):
        changed = copy.deepcopy(self.report)
        changed["cases"][0]["arms"]["native"]["methods"] = []
        with self.assertRaisesRegex(ValueError, "Nonempty"):
            self.validate(changed)
        for callers in (False, True):
            changed = copy.deepcopy(self.report)
            methods = changed["cases"][0]["arms"]["native"]["methods"]
            target = methods[0]["callers"] if callers else methods
            target.append(copy.deepcopy(target[0]))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                self.validate(changed)

    def test_actual_invoke_prepare_get_counts_are_required(self):
        for name, count in (("invoke", 19), ("prepare_forward", 19), ("get", 799)):
            for mode in ("missing", "short", "other_file"):
                with self.subTest(name=name, mode=mode):
                    changed = copy.deepcopy(self.report)
                    methods = changed["cases"][0]["arms"]["stageB"]["methods"]
                    row = next(method for method in methods if method["function"]["name"] == name)
                    if mode == "missing":
                        methods.remove(row)
                    elif mode == "short":
                        row["primitive_calls"] = row["total_calls"] = count
                    else:
                        row["function"]["filename"] = "/unbound/source.py"
                    with self.assertRaisesRegex(ValueError, "Actual profile scope count"):
                        self.validate(changed)

    def test_scope_function_rows_cannot_be_split_by_line(self):
        changed = copy.deepcopy(self.report)
        methods = changed["cases"][0]["arms"]["stageB"]["methods"]
        row = copy.deepcopy(methods[-1])
        row["function"]["line"] += 1
        methods.append(row)
        with self.assertRaisesRegex(ValueError, "Actual profile scope count"):
            self.validate(changed)

    def test_native_cannot_invoke_the_candidate_core(self):
        for name in ("prepare_forward", "get"):
            with self.subTest(name=name):
                changed = copy.deepcopy(self.report)
                case = changed["cases"][0]
                row = copy.deepcopy(next(method for method in case["arms"]["stageB"]["methods"]
                                         if method["function"]["name"] == name))
                case["arms"]["native"]["methods"].append(row)
                with self.assertRaisesRegex(ValueError, "must not call CPP core"):
                    self.validate(changed)

    def test_descriptive_line_numbers_can_change_with_pinned_source(self):
        for case in self.report["cases"]:
            for arm in case["arms"].values():
                for method in arm["methods"]:
                    method["function"]["line"] += 200
        self.validate()

    def test_cpp_or_native_file_drift_prevents_all_device_commands(self):
        items = [self.compiled["library"], *self.compiled["abi_headers"], *self.compiled["abi_libraries"],
                 self.compiled["compile_result"], self.native["library"], self.profile_pin, self.screen_pin]
        plan_file = self.root / "plan.json"
        plan_file.write_text(json.dumps(self.plan))
        for item in items:
            with self.subTest(path=item["path"]):
                path = Path(item["path"])
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"changed")
                    with patch.object(controller.subprocess, "run") as run, \
                         patch.object(controller.subprocess, "Popen") as popen, \
                         patch.object(controller, "write") as write:
                        with self.assertRaises(ValueError):
                            controller.run(plan_file)
                        run.assert_not_called()
                        popen.assert_not_called()
                        write.assert_not_called()
                finally:
                    path.write_bytes(original)

    def test_diagnostic_sources_must_be_in_runtime_binding(self):
        self.native["runtime_sources"] = self.runtime["runtime_sources"] = [self.screen_pin]
        self.freeze()
        with self.assertRaisesRegex(ValueError, "Both diagnostic sources"):
            controller.verify_inputs(self.plan)

    def test_formal_plan_protocol_and_wrong_modes_refused(self):
        mutations = [{"rounds": 3}, {"speed_gate": {}}, {"schema": "lingbot_actual_stageB_cpp_operator_plan_v2"},
                     {"performance_admitted": True}, {"profile_iterations": 30},
                     {"configs": {"A": self.plan["configs"]["B"], **self.plan["configs"]}},
                     {"base_environment": {"LINGBOT_STAGEB_HOST_PROFILE": "0", "VLLM_LINGBOT_NPU_ASYNC_METADATA": "0"}}]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    controller.verify_inputs(dict(self.plan, **mutation))

    def test_prior_launch_never_restarted(self):
        self.file("profile1/launch-intent.json", "{}")
        plan_file = self.root / "plan.json"
        plan_file.write_text(json.dumps(self.plan))
        with patch.object(controller.subprocess, "run") as run, \
             patch.object(controller.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "Never restart"):
                controller.run(plan_file)
            run.assert_not_called()
            popen.assert_not_called()

    def mocked_service(self, drift=False, fatal=None):
        """Exercise run/finally with an in-memory process and fake /proc bytes."""
        self.plan.update(model="/fake/model", port=7816)
        plan_file = self.root / "plan.json"
        plan_file.write_text(json.dumps(self.plan))
        run_dir = self.stage / "services/owned"
        leader = dict(pid=987654321, uid=os.getuid(), pgid=987654321, start=12345)
        launch = dict(leader=leader, nonce="fresh-owned-nonce", mode="profile",
                      manifest=self.plan["manifest"]["path"],
                      command=["vllm", "serve", self.plan["model"], "--omni", "--deploy-config",
                               self.plan["configs"]["B"]["path"], "--port", "7816", "--init-timeout", "1800",
                               "--stage-init-timeout", "1800"])
        environment = dict(self.plan["base_environment"],
                           LINGBOT_STAGEB_OPERATOR_SCREEN_DIR=str(self.stage / "checks"),
                           ROOT_LINGBOT_SP8_FATAL_DIR=str(run_dir),
                           ROOT_LINGBOT_SP8_FATAL_NONCE=launch["nonce"],
                           ROOT_LINGBOT_SP8_EXPERIMENT_MODE="profile")
        environment_bytes = b"\0".join((key + "=" + value).encode() for key, value in environment.items())
        original_read_bytes = Path.read_bytes

        def fake_read_bytes(path):
            if str(path) == f"/proc/{leader['pid']}/environ":
                return environment_bytes
            return original_read_bytes(path)

        class Process:
            returncode = None

            def poll(self):
                return None

            def wait(self, timeout):
                self.returncode = 247
                result = dict(launch, exit_code=-9, fatal=fatal, remaining_owned_processes=[])
                (run_dir / "result.json").write_text(json.dumps(result))
                return self.returncode

        def popen(*args, **kwargs):
            run_dir.mkdir(parents=True)
            (run_dir / "launch.json").write_text(json.dumps(launch))
            return Process()

        commands = []

        def subprocess_run(command, **kwargs):
            commands.append(command)
            if command[0] == "npu-smi":
                return
            if command[1] != "recovery/run_smoke.py":
                raise AssertionError("Unexpected simulated subprocess")
            for rank in range(8):
                self.json_file(f"profile1/checks/rank{rank}.json", dict(self.report, rank=rank))
            if drift:
                Path(self.compiled["abi_libraries"][0]["path"]).write_text("changed during service")

        with patch.object(controller.subprocess, "run", side_effect=subprocess_run), \
             patch.object(controller.subprocess, "Popen", side_effect=popen) as start, \
             patch.object(controller.Path, "read_bytes", fake_read_bytes), \
             patch.object(controller, "identity", return_value=leader), \
             patch.object(controller, "retire") as retire, \
             patch.object(controller, "verify_cpp_binding", wraps=controller.verify_cpp_binding) as verify:
            if drift or fatal is not None:
                with self.assertRaises(ValueError):
                    controller.run(plan_file)
            else:
                controller.run(plan_file)
            self.assertEqual(verify.call_count, 2)
            start.assert_called_once()
            retire.assert_called_once_with(leader)
            self.assertEqual(len(commands), 2)
        return controller.read(self.stage / "controller-result.json")

    def test_mocked_service_checks_again_and_retires_without_any_admission(self):
        result = self.mocked_service()
        self.assertTrue(result["completed"])
        self.assertTrue(result["diagnostic_only"])
        self.assertTrue(all(result[key] is False for key in controller.INELIGIBLE))
        self.assertEqual(result["remaining_owned_processes"], [])
        checks = controller.read(self.stage / "checks-completed.json")
        self.assertEqual(len(checks["checks"]), 8)

    def test_post_profile_abi_drift_refuses_success_but_retires_same_service(self):
        result = self.mocked_service(drift=True)
        self.assertFalse(result["completed"])
        self.assertFalse((self.stage / "checks-completed.json").exists())

    def test_fatal_service_never_becomes_completed_or_restarted(self):
        fatal = {"origin": "test-host-profile", "nonce": "fresh-owned-nonce"}
        result = self.mocked_service(fatal=fatal)
        self.assertFalse(result["completed"])
        self.assertEqual(result["fatal"], fatal)


if __name__ == "__main__":
    unittest.main()
