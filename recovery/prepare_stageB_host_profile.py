"""Freeze one diagnostic-only StageB host profile; never a performance screen."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shlex

from run_cross_model_screen import need, pin, read, source_check, write
from run_stageB_cpp_operator import verify_cpp_binding

FLAG = "lingbot_npu_fixed_modulation_cpp_v2"
PROFILE_SOURCE = "vllm_omni/diffusion/layers/lingbot_fixed_modulation_host_profile.py"
SCREEN_SOURCE = "vllm_omni/diffusion/layers/lingbot_fixed_modulation_screen.py"


def prepare(parent_plan, output, port, cpp_build):
    import yaml
    root = Path(__file__).resolve().parents[1]
    parent_plan = Path(parent_plan).resolve(strict=True)
    parent = read(parent_plan)
    output = Path(output).resolve()
    need(not output.exists() or not any(output.iterdir()), "Fresh empty diagnostic directory required")
    source_check(parent["manifest"])
    need(pin(parent["configs"]["A"]["path"]) == parent["configs"]["A"], "Parent baseline config changed")
    cpp_build = Path(cpp_build).resolve(strict=True)
    compiled = read(cpp_build)
    need(compiled.get("schema") == "lingbot_stageB_guard_cpu_build_v2"
         and compiled.get("namespace") == "lingbot_stageB_guard_v2"
         and compiled.get("registration") is True and compiled.get("npu_initialized") is False
         and compiled.get("v1_qualification_inherited") is False,
         "Fresh independently registered CPU guard required")
    for item in (compiled["source"], compiled["library"], compiled["builder"], compiled["provenance"],
                 compiled["source_contract_tests"], *compiled["abi_headers"], *compiled["abi_libraries"]):
        need(pin(item["path"]) == item, "CPP compilation source/ABI changed")
    output.mkdir(parents=True, exist_ok=True)
    ancestry = read(parent["manifest"]["path"])
    manifest = copy.deepcopy(ancestry)
    old_root = Path(parent["worktree"])
    # Preserve the qualified native libraries and their exact source bytes.
    manifest["compilation_ancestor"] = parent["manifest"]
    manifest.update(runtime_binding_only=True, recompiled=False,
                    device_validated=False, model_validated=False,
                    historical_admission_inherited=False)
    for key in ("source",):
        relative = Path(manifest[key]["path"]).relative_to(old_root)
        new = pin(root / relative)
        need((new["bytes"], new["sha256"]) == (manifest[key]["bytes"], manifest[key]["sha256"]),
             "Unchanged native source required")
        manifest[key] = new
    paths = [Path(item["path"]).relative_to(old_root) for item in ancestry["runtime_sources"]]
    paths += [Path(p) for p in (
        "vllm_omni/diffusion/models/lingbot_world/pipeline.py",
        "vllm_omni/diffusion/models/lingbot_world/dmd_block.py",
        "vllm_omni/diffusion/layers/lingbot_fixed_modulation_stageB.py",
        SCREEN_SOURCE,
        PROFILE_SOURCE,
        "vllm_omni/diffusion/layers/lingbot_fixed_modulation_stageB_cpp.py",
        "vllm_omni/diffusion/layers/lingbot_fixed_modulation_cpp_provider.py",
        "vllm_omni/diffusion/layers/lingbot_fixed_modulation_dispatch.py",
        "vllm_omni/diffusion/layers/lingbot_fixed_timestep_bank_stageA_v1.py",
        "vllm_omni/diffusion/hooks/base.py",
        "vllm_omni/diffusion/hooks/sequence_parallel.py",
        "vllm_omni/diffusion/distributed/sp_plan.py",
        "vllm_omni/diffusion/layers/lingbot_conditioning_ops.py",
        "vllm_omni/diffusion/layers/lingbot_frame_conditioning.py")]
    manifest["runtime_sources"] = [pin(root / p) for p in sorted(set(paths))]
    manifest_path = output / "host_manifest.json"
    write(manifest_path, manifest)
    source_check(pin(manifest_path))
    cpp_runtime = {key: compiled[key] for key in ("source", "library", "abi_headers", "abi_libraries",
                                                "cxx11_abi", "torch", "torch_npu", "sdk")}
    cpp_runtime.update(schema="lingbot_stageB_guard_runtime_v2", namespace="lingbot_stageB_guard_v2",
                       compilation_ancestor=pin(cpp_build), runtime_sources=manifest["runtime_sources"],
                       device_admitted=False, model_admitted=False, historical_admission_inherited=False,
                       v1_qualification_inherited=False, guard_timing="per_layer_before_consumption")
    cpp_runtime_path = output / "cpp_runtime_manifest.json"
    write(cpp_runtime_path, cpp_runtime)
    parity = read(root / "recovery/validation/historical-infrastructure-source-parity-20261013.json")
    need(parity["passed"] and len(parity["files"]) == 228, "Historical file list must cover228")
    protected = []
    for item in parity["files"]:
        actual = pin(root / item["relative_path"])
        need(actual["sha256"] == item["historical_sha256"], "Protected Python changed: " + item["relative_path"])
        protected.append(actual)
    configs = {}
    for arm in ("B",):
        config = yaml.safe_load(Path(parent["configs"]["A"]["path"]).read_text())
        config["stages"][0]["model_config"]["lingbot_npu_fixed_modulation_stageB"] = False
        config["stages"][0]["model_config"][FLAG] = True
        path = output / f"config-{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        configs[arm] = pin(path)
    env = dict(parent["base_environment"])
    env["PYTHONPATH"] = env["PYTHONPATH"].replace(str(old_root), str(root), 1)
    env.update(LINGBOT_RECOVERY_MANIFEST=str(manifest_path),
               LINGBOT_RECOVERY_MANIFEST_SHA256=pin(manifest_path)["sha256"])
    env.update(LINGBOT_STAGEB_CPP_MANIFEST=str(cpp_runtime_path),
               LINGBOT_STAGEB_CPP_MANIFEST_SHA256=pin(cpp_runtime_path)["sha256"])
    env["LINGBOT_STAGEB_HOST_PROFILE"] = "1"
    need(env["VLLM_LINGBOT_NPU_ASYNC_METADATA"] == "0", "Metadata must remain OFF")
    need(not any(env.get(key) for key in ("LINGBOT_CAPTURE_DIR", "LINGBOT_PROFILE_CONTROL",
                                         "LINGBOT_STAGEB_OPERATOR_SCREEN_DIR")),
         "Parent environment must not carry another capture/profile directory")
    harness = ["prepare_stageB_host_profile.py", "run_stageB_host_profile.py", "run_stageB_cpp_operator.py",
               "test_stageB_host_profile_controller_cpu.py", "run_smoke.py", "supervise.py",
               "run_cross_model_screen.py", "summarize_timing.py"]
    plan = dict(schema="lingbot_actual_stageB_host_profile_plan_v1", worktree=str(root), output=str(output),
                parent_plan=pin(parent_plan), parent_baseline_config=parent["configs"]["A"],
                manifest=pin(manifest_path), cpp_manifest=pin(cpp_runtime_path),
                cpp_compilation=pin(cpp_build), protected=protected, configs=configs,
                profile_source=pin(root / PROFILE_SOURCE), screen_source=pin(root / SCREEN_SOURCE),
                base_environment=env, port=port, model=parent["model"], image=parent["image"],
                integrity=parent["integrity"], model_revision=parent["model_revision"],
                harness=[pin(root / "recovery" / name) for name in harness],
                benchmark=[pin(root / "benchmarks/lingbot_world" / name) for name in (
                    "benchmark_lingbot_world_realtime.py", "workload.py", "interaction_telemetry.py")],
                env_script=pin(root / "recovery/env.sh"),
                histories=[4680, 9360, 14040, 18720, 23400, 28080], probes=4, ranks=8,
                actual_layers=40, profile_iterations=20, diagnostic_only=True,
                performance_admitted=False, model_admitted=False, speed_gate_eligible=False,
                model_timing_eligible=False, final_acceptance=False)
    for label in ("profile1",):
        stage = output / label
        stage.mkdir(exist_ok=True)
        selected = dict(env, LINGBOT_STAGEB_OPERATOR_SCREEN_DIR=str(stage / "checks"))
        shell = "#!/usr/bin/env bash\nset -eo pipefail\ncd " + shlex.quote(str(root)) + "\nsource recovery/env.sh\n"
        shell += 'while IFS= read -r key; do\n case "$key" in LINGBOT_*|VLLM_LINGBOT_*|ROOT_LINGBOT_*|OMP_NUM_THREADS|MKL_NUM_THREADS|ASCEND_RT_VISIBLE_DEVICES|TASK_QUEUE_ENABLE|PYTORCH_NPU_ALLOC_CONF|ASCEND_CUSTOM_OPP_PATH|PYTHONPATH|VLLM_TARGET_DEVICE|VLLM_WORKER_MULTIPROC_METHOD) unset "$key";; esac\ndone < <(compgen -e)\n'
        for key, value in selected.items():
            shell += "export " + key + "=" + shlex.quote(value) + "\n"
        shell += "exec " + shlex.join(["python", "recovery/supervise.py", "--runs", str(stage / "services"),
            "--mode", "profile", "--recv", "--manifest", str(manifest_path), "--", "vllm", "serve", plan["model"],
            "--omni", "--deploy-config", configs["B"]["path"], "--port", str(port), "--init-timeout", "1800",
            "--stage-init-timeout", "1800"]) + "\n"
        (stage / "launch.sh").write_text(shell)
    plan["launch_scripts"] = [pin(output / "profile1/launch.sh")]
    # This remains file-only validation of the previously built guard. The
    # independent runtime binding never recompiles or inherits device admission.
    verify_cpp_binding(plan)
    write(output / "plan.json", plan)
    return output / "plan.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=7816)
    parser.add_argument("--cpp-build", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.parent_plan, args.output, args.port, args.cpp_build))
