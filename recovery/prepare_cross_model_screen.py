"""CPU-only seal of immutable cross-only configs and per-stage launch commands."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shlex

from run_cross_model_screen import ORDER, DYNAMIC, need, pin, read, source_check, write


def prepare(freeze_path, port=7816):
    import yaml
    freeze_path = Path(freeze_path).resolve(strict=True)
    freeze = read(freeze_path)
    output, worktree = freeze_path.parent, Path(__file__).resolve().parents[1]
    plan_path = output / "plan.json"
    need(not plan_path.exists(), "Never overwrite a frozen model-screen plan")
    source_check(freeze["manifest"])
    need(pin(freeze["source_config"]["path"]) == freeze["source_config"], "Actual full-fixture source config changed")
    source = yaml.safe_load(Path(freeze["source_config"]["path"]).read_text())
    for arm in ("A", "B"):
        item = freeze["configs"][arm]
        need(pin(item["path"]) == item, "Prepared config changed")
        actual = yaml.safe_load(Path(item["path"]).read_text())
        wanted = copy.deepcopy(source)
        wanted["stages"][0]["model_config"].update(lingbot_npu_recovery_qk=True,
            lingbot_npu_recovery_qk_combined=False, lingbot_npu_recovery_cross=arm == "A",
            lingbot_npu_recovery_cross_combined=arm == "B")
        need(actual == wanted, "A/B differs beyond the two cross flags")
    model = source["stages"][0]["model_config"]
    need((model["ar_diffusion_width"], model["ar_diffusion_height"], source["stages"][0]["default_sampling_params"]["num_inference_steps"]) == (832, 480, 4), "Frozen workload changed")
    # A was already launched manually by root; preserve its live snapshot as the
    # authority for ordinary launch policy rather than inventing new environment.
    snapshot = read(output / "correctness-A/service-environment.json")
    common = {k: v for k, v in snapshot["environment"].items() if k not in DYNAMIC
              and not any(word in k for word in ("CAPTURE", "PROFILE", "DIAGNOSTIC"))}
    need(common.get("ASCEND_RT_VISIBLE_DEVICES") == "8,9,10,11,12,13,14,15"
         and common.get("VLLM_LINGBOT_NPU_ASYNC_METADATA") == "0"
         and common.get("LINGBOT_RECOVERY_RECV") == "1", "Original devices/metadata/recv changed")
    for k in ("LINGBOT_QK_COMBINED_MANIFEST", "LINGBOT_QK_COMBINED_MANIFEST_SHA256", "LINGBOT_SP8_OBSERVE_MODE"):
        need(k not in common, "Unwanted experiment environment present")
    paths = ("recovery/run_cross_model_screen.py", "recovery/prepare_cross_model_screen.py", "recovery/run_smoke.py",
             "recovery/run_timing.py", "recovery/supervise.py", "recovery/summarize_timing.py", "recovery/compare_captures.py",
             "recovery/env.sh", "benchmarks/lingbot_world/benchmark_lingbot_world_realtime.py", "benchmarks/lingbot_world/workload.py",
             "benchmarks/lingbot_world/interaction_telemetry.py", "vllm_omni/diffusion/layers/lingbot_capture.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("screen_workload", worktree / "benchmarks/lingbot_world/workload.py")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    integrity = read(freeze["integrity"]["path"])
    need(integrity["complete"] is True and integrity["passed"] is True, "Existing model hash validation failed")
    plan = dict(schema="lingbot_cross_only_model_screen_v1", plan_path=str(plan_path), output=str(output), worktree=str(worktree),
        order=list(ORDER), config_freeze=pin(freeze_path), configs=freeze["configs"], manifest=freeze["manifest"],
        image=freeze["image"], integrity=freeze["integrity"], model=freeze["model"], port=port,
        base_environment=common, environment_authority=pin(output / "correctness-A/service-environment.json"),
        environment_scope="selected LingBot policy, device, thread and allocator variables; not the entire process environment or inherited loader PATH/LD_LIBRARY_PATH",
        harness_pins=[pin(worktree / p) for p in paths], model_revision=integrity["revision"],
        workload=dict(width=832, height=480, mux_fps=12, target_fps=12, seed=module.DEFAULT_SEED,
                      prompt=module.DEFAULT_PROMPT, flow_shift=module.DEFAULT_FLOW_SHIFT, camera_pattern="forward",
                      denoise_steps=4, correctness_chunks=80, warmup_chunks=40, measured_chunks=161),
        model_sha_source="pinned previously passing full-model integrity receipt; not a new weight rehash",
        source_pinned=True, prepared_only=True, final_acceptance=False)
    for stage in ORDER:
        directory = output / stage
        directory.mkdir(exist_ok=True)
        if stage == "compare":
            continue
        mode, arm = stage.split("-")
        arm = arm[0]
        env = dict(common)
        if mode == "correctness":
            env.update(LINGBOT_CAPTURE_DIR=str(directory / "captures"), LINGBOT_CAPTURE_RANK="-1", LINGBOT_CAPTURE_ACTIVATIONS="0")
        shell = "#!/usr/bin/env bash\nset -eo pipefail\ncd " + shlex.quote(str(worktree)) + "\nsource recovery/env.sh\n"
        shell += "python recovery/run_cross_model_screen.py --plan " + shlex.quote(str(plan_path)) + " --stage " + stage + "\n"
        # Clear every selected inherited value, including absent TASK_QUEUE_ENABLE.
        shell += 'while IFS= read -r key; do\n case "$key" in LINGBOT_*|VLLM_LINGBOT_*|ROOT_LINGBOT_*|OMP_NUM_THREADS|MKL_NUM_THREADS|ASCEND_RT_VISIBLE_DEVICES|TASK_QUEUE_ENABLE|PYTORCH_NPU_ALLOC_CONF|ASCEND_CUSTOM_OPP_PATH|PYTHONPATH|VLLM_TARGET_DEVICE|VLLM_WORKER_MULTIPROC_METHOD) unset "$key";; esac\ndone < <(compgen -e)\n'
        for key, value in env.items():
            shell += "export " + key + "=" + shlex.quote(value) + "\n"
        command = ["python", "recovery/supervise.py", "--runs", str(directory / "services"), "--mode", mode, "--recv", "--manifest", plan["manifest"]["path"],
                   "--", "vllm", "serve", plan["model"], "--omni", "--deploy-config", plan["configs"][arm]["path"], "--port", str(port), "--init-timeout", "1800", "--stage-init-timeout", "1800"]
        shell += "exec " + shlex.join(command) + "\n"
        launch = directory / "launch.sh"
        need(not launch.exists(), "Launch script already exists")
        launch.write_text(shell)
    plan["launch_scripts"] = {stage: pin(output / stage / "launch.sh") for stage in ORDER if stage != "compare"}
    write(plan_path, plan)
    return plan_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-freeze", type=Path, required=True)
    parser.add_argument("--port", type=int, default=7816)
    args = parser.parse_args()
    print(prepare(args.config_freeze, args.port))
