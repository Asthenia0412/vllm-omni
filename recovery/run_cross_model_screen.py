"""Cross-only model screen controller for an already-owned service.

No service is launched here. Generated launch.sh uses existing supervise.py;
this controller reuses run_smoke/run_timing/compare_captures and identity/retire.
An explicit --execute is required to submit requests, compare, or retire workers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from supervise import identity, retire

ORDER = ("correctness-A", "correctness-B", "compare", "timing-A1", "timing-B1", "timing-B2", "timing-A2", "timing-A3", "timing-B3")
ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "ASCEND_RT_VISIBLE_DEVICES", "TASK_QUEUE_ENABLE",
            "PYTORCH_NPU_ALLOC_CONF", "ASCEND_CUSTOM_OPP_PATH", "PYTHONPATH", "VLLM_TARGET_DEVICE", "VLLM_WORKER_MULTIPROC_METHOD")
DYNAMIC = ("ROOT_LINGBOT_SP8_FATAL_BASE", "ROOT_LINGBOT_SP8_FATAL_DIR", "ROOT_LINGBOT_SP8_FATAL_NONCE", "ROOT_LINGBOT_SP8_EXPERIMENT_MODE")


def need(ok, text):
    if not ok:
        raise ValueError(text)


def pin(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def selected_environment(env):
    return {key: value for key, value in env.items() if key.startswith(("LINGBOT_", "VLLM_LINGBOT_", "ROOT_LINGBOT_")) or key in ENV_KEYS}


def source_check(manifest):
    value = read(manifest["path"])
    need(pin(manifest["path"]) == manifest, "Build manifest changed")
    for item in (value["source"], value["library"], value["kernels"], *value["kernel_sources"], *value["runtime_sources"], *value["abi_headers"]):
        need(pin(item["path"]) == item, "Build/runtime/header changed: " + item["path"])


def verify_stage(plan, name):
    item = read(Path(plan["output"]) / name / "stage.json")
    need(item["completed"] is True and item["stage"] == name and item["plan"] == pin(plan["plan_path"]), "Prior stage incomplete or from another plan")
    for pinned in item["evidence"]:
        need(pin(pinned["path"]) == pinned, "Prior stage evidence changed")
    return item


def preflight(plan, stage):
    need(plan["schema"] == "lingbot_cross_only_model_screen_v1" and tuple(plan["order"]) == ORDER, "Plan/order differs")
    need(Path.cwd().resolve() == Path(plan["worktree"]), "Use MAIN worktree cwd")
    source_check(plan["manifest"])
    for item in (plan["config_freeze"], plan["image"], plan["integrity"], *plan["configs"].values(), *plan["harness_pins"], *plan["launch_scripts"].values()):
        need(pin(item["path"]) == item, "Pinned input/config/controller changed: " + item["path"])
    integrity = read(plan["integrity"]["path"])
    need(integrity["complete"] and integrity["passed"], "Model integrity receipt is not passing")
    for prior in ORDER[:ORDER.index(stage)]:
        verify_stage(plan, prior)
    if stage.startswith("timing"):
        paired = read(Path(plan["output"]) / "compare" / "paired80.json")
        need(paired["passed"] is True and [(r["rank"], r["latent_chunks"], r["rgb_frames"]) for r in paired["ranks"]]
             == [(rank, 80, 957 if rank == 0 else 0) for rank in range(8)], "Fresh paired80 correctness is required")


def stage_environment(plan, stage):
    value = dict(plan["base_environment"])
    if stage.startswith("correctness"):
        value.update(LINGBOT_CAPTURE_DIR=str(Path(plan["output"]) / stage / "captures"),
                     LINGBOT_CAPTURE_RANK="-1", LINGBOT_CAPTURE_ACTIVATIONS="0")
    return value


def validate_launch(plan, stage, run):
    launch = read(run / "launch.json")
    arm = stage.split("-")[1][0]
    mode = stage.split("-")[0]
    wanted = ["vllm", "serve", plan["model"], "--omni", "--deploy-config", plan["configs"][arm]["path"],
              "--port", str(plan["port"]), "--init-timeout", "1800", "--stage-init-timeout", "1800"]
    need(launch["command"] == wanted and launch["mode"] == mode and launch["manifest"] == plan["manifest"]["path"], "Owned launch config/model/mode differs")
    need(launch["leader"]["uid"] == os.getuid() and run.parent.resolve() == Path(plan["output"]) / stage / "services", "Wrong stage's owned service")
    if stage.startswith("timing-") and stage != "timing-A1":
        prior = verify_stage(plan, ORDER[ORDER.index(stage) - 1])
        proof = read(prior["retirement"]["path"])
        need(prior["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
             and launch["leader"]["start"] * 1_000_000_000 // os.sysconf("SC_CLK_TCK") >= proof["end_boot_ns"],
             "Timing service started before prior owned service completed retirement")
    return launch


def snapshot(plan, stage, run, launch):
    target = Path(plan["output"]) / stage / "service-environment.json"
    leader = launch["leader"]
    if identity(leader["pid"]) == leader:
        raw = (Path("/proc") / str(leader["pid"]) / "environ").read_bytes()
        env = selected_environment(dict(x.decode().split("=", 1) for x in raw.split(b"\0") if b"=" in x))
        need(identity(leader["pid"]) == leader, "Owned leader changed while reading environment")
        value = dict(schema="cross_only_service_environment_v1", run=str(run), launch=launch, environment=env,
                     captured_while_leader_alive=True, observed_pid_start=str(leader["start"]))
        if target.exists():
            previous = read(target)
            need(previous["environment"] == env and previous["launch"] == launch, "Existing live environment differs")
        else:
            write(target, value)
    need(target.exists(), "Completed adoption requires a previously captured live environment")
    value = read(target)
    need(value["run"] == str(run) and value["launch"] == launch and value["captured_while_leader_alive"] is True
         and int(value["observed_pid_start"]) == leader["start"], "Environment snapshot belongs to another launch")
    env = value["environment"]
    need({k: v for k, v in env.items() if k not in DYNAMIC} == stage_environment(plan, stage), "Service environment differs from pinned cross-only workload")
    need(env.get("ROOT_LINGBOT_SP8_FATAL_DIR") == str(run) and env.get("ROOT_LINGBOT_SP8_FATAL_NONCE") == launch["nonce"]
         and env.get("ROOT_LINGBOT_SP8_EXPERIMENT_MODE") == launch["mode"], "Fatal supervision identity differs")
    return pin(target)


def retirement(run, launch, completed_request):
    receipt = run.parent.parent / "retirement.json"
    if identity(launch["leader"]["pid"]) == launch["leader"]:
        write(receipt, dict(kind="controller_identity_retirement", launch=pin(run / "launch.json"),
            leader=launch["leader"], completed_request_before_retirement=completed_request,
            request_exit=pin(run / "smoke-exit.json") if (run / "smoke-exit.json").exists() else None,
            begin_boot_ns=time.clock_gettime_ns(time.CLOCK_BOOTTIME)))
        retire(launch["leader"])
    for _ in range(80):
        if (run / "result.json").exists():
            break
        time.sleep(.5)
    result = read(run / "result.json")
    for key in ("nonce", "leader", "command", "mode", "manifest"):
        need(result[key] == launch[key], "Retirement belongs to another launch")
    need(result["fatal"] is None and result["remaining_owned_processes"] == [], "Fatal or surviving owned workers")
    # Persistent model services may need SIGTERM/SIGKILL after a successful request.
    need(completed_request and result["exit_code"] in (0, -15, -9), "Unexpected persistent service termination")
    if not receipt.exists() and (run / "intentional-retirement.json").exists():
        # Root's contemporaneous record was written BEFORE its manual retire.
        original = read(run / "intentional-retirement.json")
        need(original["leader"] == launch["leader"] and "correctness capture complete" in original["reason"]
             and (run / "smoke-exit.json").stat().st_mtime_ns <= original["observed_time_ns"] <= (run / "result.json").stat().st_mtime_ns,
             "Root retirement intent does not follow the completed request and precede the result")
        write(receipt, dict(kind="root_manual_retirement", launch=pin(run / "launch.json"), leader=launch["leader"],
             request_exit=pin(run / "smoke-exit.json"), completed_request_before_retirement=True,
             result=pin(run / "result.json"), contemporaneous_intent=pin(run / "intentional-retirement.json"),
             observed_root_wall_ns=original["observed_time_ns"], post_hoc_binding_only=True))
    need(receipt.exists(), "Post-hoc retirement requires independent root retirement evidence")
    proof = read(receipt)
    need(proof["kind"] in ("controller_identity_retirement", "root_manual_retirement")
         and proof["launch"] == pin(run / "launch.json")
         and proof["request_exit"] == pin(run / "smoke-exit.json")
         and proof["completed_request_before_retirement"] is True, "Retirement not tied to completed owned request")
    if proof["kind"] == "controller_identity_retirement":
        proof.update(result=pin(run / "result.json"), end_boot_ns=time.clock_gettime_ns(time.CLOCK_BOOTTIME))
        write(receipt, proof)
    else:
        need(proof["result"] == pin(run / "result.json"), "Root retirement result changed")
    return pin(run / "result.json"), pin(receipt)


def validate_request(plan, run, count, label=""):
    command = read(run / (label + "smoke-command.json"))
    wanted = [sys.executable, "benchmarks/lingbot_world/benchmark_lingbot_world_realtime.py",
              "--port", str(plan["port"]), "--model", plan["model"], "--image", plan["image"]["path"],
              "--num-chunks", str(count), "--width", "832", "--height", "480", "--fps", "12",
              "--output-json", str(run / f"rollout{count}.json"), "--print-chunks"]
    need(command == wanted and read(run / (label + "smoke-exit.json"))["exit_code"] == 0, "Request command/exit differs")
    value = read(run / f"rollout{count}.json")
    workload = value["workload"]
    need(all(workload[key] == plan["workload"][key] for key in ("width", "height", "seed", "flow_shift"))
         and workload["fps"] == 12 and workload["num_chunks"] == count and workload["prompt_updates"] == []
         and value["config"]["camera_pattern"] == "forward" and value["model"] == plan["model"]
         and value["config"]["sessions"] == 1 and value["config"]["warmup_sessions"] == 0,
         "Actual client workload differs")
    from summarize_timing import summarize
    return value, summarize(value, count)


def measure(plan, stage, run, adopt):
    directory = Path(plan["output"]) / stage
    launch = validate_launch(plan, stage, run)
    env_pin = snapshot(plan, stage, run, launch)
    target = directory / "stage.json"
    if target.exists() and read(target).get("completed"):
        return verify_stage(plan, stage)
    if (directory / "in-progress.json").exists():
        previous = read(directory / "in-progress.json")
        need(previous["run"] == str(run) and adopt, "Resume only the same completed request with --adopt-completed")
        previous_owner = previous.get("controller")
        need(previous_owner is None or identity(previous_owner["pid"]) != previous_owner, "Previous controller still active")
    need(adopt or identity(launch["leader"]["pid"]) == launch["leader"], "Fresh owned service required")
    mode = launch["mode"]
    if not adopt:
        need(not list(run.glob("rollout*.json")) and not (run / "timing-protocol.json").exists()
             and not (run / "smoke-command.json").exists(), "Do not resend to a used service")
    write(directory / "in-progress.json", dict(stage=stage, run=str(run), controller=identity(os.getpid()),
                                             plan=pin(plan["plan_path"]), adopted_completed=adopt))
    completed = False
    evidence = [pin(run / "launch.json"), env_pin]
    try:
        if not adopt:
            script = "run_smoke.py" if mode == "correctness" else "run_timing.py"
            command = [sys.executable, "recovery/" + script, "--run", str(run), "--model", plan["model"],
                       "--integrity", plan["integrity"]["path"], "--image", plan["image"]["path"],
                       "--port", str(plan["port"]), "--chunks", "80" if mode == "correctness" else "161"]
            subprocess.run(command, check=True)
        count = 80 if mode == "correctness" else 161
        need(read(run / "smoke-exit.json")["exit_code"] == 0, "Request did not complete successfully")
        rollout, summary = validate_request(plan, run, count)
        if mode == "timing":
            timing = read(run / "timing-protocol.json")
            need(timing["completed"] is True and timing["warmup_chunks"] == 40 and timing["measurement_chunks"] == 161
                 and timing["launch"] == launch and timing["config_sha256"] == plan["configs"][stage.split('-')[1][0]]["sha256"]
                 and timing["manifest_sha256"] == plan["manifest"]["sha256"], "Independent warm40+161 receipt differs")
            warm, _ = validate_request(plan, run, 40, "warm40-")
            need(warm["sessions"][0]["request_id"] != rollout["sessions"][0]["request_id"], "Warm and measured requests must differ")
            evidence += [pin(run / "timing-protocol.json"), pin(run / "main-window.json"), pin(run / "rollout40.json"),
                         pin(run / "warm40-smoke-command.json"), pin(run / "warm40-smoke-exit.json")]
        else:
            for rank in range(8):
                paths = list((directory / "captures" / f"rank{rank}").glob("*/capture.json"))
                need(len(paths) == 1, "Exactly one fresh capture session per rank required")
                data = read(paths[0])
                need(data["rank"] == rank and data["timing_eligible"] is False
                     and [c["chunk"] for c in data["chunks"]] == list(range(80)), "Missing/duplicate correctness chunks")
                need(data["session_id"].startswith(rollout["sessions"][0]["request_id"] + "-")
                     and Path(data["payload_directory"]).resolve() == paths[0].parent.resolve(), "Capture not from this request/fresh directory")
                for chunk in data["chunks"]:
                    for key in ("latent_path", "rgb_path"):
                        if key in chunk:
                            need(Path(chunk[key]).name == chunk[key] and (paths[0].parent / chunk[key]).is_file(), "Capture payload path escapes directory")
                evidence.append(pin(paths[0]))
        evidence += [pin(run / f"rollout{count}.json"), pin(run / "smoke-command.json"), pin(run / "smoke-exit.json")]
        preflight(plan, stage)  # Recheck source/policy pins before recording success.
        completed = True
    finally:
        # Only this exact uid/pgid/start-time launch is retired, never device/name sweeping.
        retired, retirement_proof = retirement(run, launch, completed)
    evidence.extend((retired, retirement_proof))
    result = dict(schema="lingbot_cross_model_stage_v1", stage=stage, completed=True, plan=pin(plan["plan_path"]),
                  run=str(run), config=plan["configs"][stage.split('-')[1][0]], manifest=plan["manifest"],
                  environment=env_pin, evidence=evidence, summary=summary,
                  adopted_completed=adopt, retired_monotonic_ns=time.perf_counter_ns(),
                  retirement=retirement_proof, boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                  clock_ticks_per_second=os.sysconf("SC_CLK_TCK"), leader_start_ticks=launch["leader"]["start"],
                  timing_eligible=mode == "timing",
                  persistent_service_exit_code=read(retired["path"])["exit_code"], final_acceptance=False)
    write(target, result)
    return result


def compare(plan):
    output = Path(plan["output"])
    directory = output / "compare"
    directory.mkdir(exist_ok=True)
    result_path = directory / "paired80.json"
    need(not result_path.exists(), "Use existing completed compare receipt or a fresh plan; do not overwrite")
    command = [sys.executable, "recovery/compare_captures.py", "--reference", str(output / "correctness-A/captures"),
               "--candidate", str(output / "correctness-B/captures"), "--chunks", "80", "--output", str(result_path)]
    subprocess.run(command, check=True)
    need(read(result_path)["passed"] is True, "Fresh paired80 comparison failed")
    result = dict(schema="lingbot_cross_model_stage_v1", stage="compare", completed=True, plan=pin(plan["plan_path"]),
                  evidence=[pin(result_path), pin(output / "correctness-A/stage.json"), pin(output / "correctness-B/stage.json")],
                  manifest=plan["manifest"], final_acceptance=False)
    write(directory / "stage.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=ORDER, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--adopt-completed", action="store_true", help="Never submits another request; requires completed request files")
    args = parser.parse_args()
    plan = read(args.plan)
    need(str(args.plan.resolve()) == plan["plan_path"], "Plan path differs")
    try:
        preflight(plan, args.stage)
    except BaseException as error:
        if args.execute:
            write(Path(plan["output"]) / args.stage / "failure.json", dict(error=repr(error),
                run=str(args.run) if args.run else None, completed=False,
                needs_manual_retirement=args.run is not None, reason="preflight failed before trusted stage execution"))
        raise
    if not args.execute:
        print("PREFLIGHT PASSED; no request, comparison or retirement executed")
        return
    target = Path(plan["output"]) / args.stage / "stage.json"
    if target.exists() and read(target).get("completed"):
        verify_stage(plan, args.stage)
        print("ALREADY COMPLETE " + str(target))
        return
    try:
        if args.stage == "compare":
            result = compare(plan)
        else:
            need(args.run is not None, "Existing owned supervisor --run required")
            result = measure(plan, args.stage, args.run.resolve(strict=True), args.adopt_completed)
    except BaseException as error:
        write(Path(plan["output"]) / args.stage / "failure.json", dict(error=repr(error),
            run=str(args.run) if args.run else None, completed=False,
            needs_manual_retirement=args.run is not None and not (args.run / "result.json").exists(),
            reason="no fallback or automatic repeat; inspect owned supervisor before resuming"))
        raise
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
