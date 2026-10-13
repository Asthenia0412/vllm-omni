"""One fresh identity-owned StageB host profile; no performance/model admission.

All device and service work requires --execute. Import and receipt validation
use the standard library only and reuse the frozen CPP ancestry verifier.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run_cross_model_screen import need, pin, read, source_check, write, selected_environment, DYNAMIC
from supervise import identity, retire


from run_stageB_cpp_operator import verify_cpp_binding

HISTORIES = [4680, 9360, 14040, 18720, 23400, 28080]
INELIGIBLE = ("performance_admitted", "model_admitted", "speed_gate_eligible", "model_timing_eligible")
FORMAL_FIELDS = frozenset(("round", "rounds", "warmups", "wall_samples", "event_samples",
                           "wall_measurements", "event_measurements", "wall_ns", "event_ms", "speed_gate"))


def reject_formal_fields(value):
    """A diagnostic must not acquire formal timing evidence under another label."""
    if isinstance(value, dict):
        need(not FORMAL_FIELDS.intersection(value), "Formal wall/event/speed evidence is forbidden in a host profile")
        for item in value.values():
            reject_formal_fields(item)
    elif isinstance(value, list):
        for item in value:
            reject_formal_fields(item)


def function_key(value):
    need(type(value) is dict and set(value) == {"filename", "line", "name"}, "Invalid cProfile function identity")
    need(type(value["filename"]) is str and bool(value["filename"])
         and type(value["name"]) is str and bool(value["name"])
         and type(value["line"]) is int and value["line"] >= 0,
         "Invalid cProfile filename/line/name")
    return value["filename"], value["line"], value["name"]


def validate_stats(value):
    key = function_key(value["function"])
    cc, nc = value["primitive_calls"], value["total_calls"]
    need(type(cc) is int and type(nc) is int and 0 <= cc <= nc,
         "Invalid primitive/total call counts")
    need(all(type(value[name]) in (int, float) and math.isfinite(value[name]) and value[name] >= 0
             for name in ("tottime_s", "cumtime_s")), "Invalid finite nonnegative cProfile times")
    return key


def validate_rank(plan, result, rank, nonce):
    reject_formal_fields(result)
    need(result.get("schema") == "lingbot_actual_stageB_host_profile_v1"
         and result.get("passed") is True and type(result.get("rank")) is int and result["rank"] == rank
         and result.get("nonce") == nonce, "Missing/incomplete owned host-profile evidence")
    need(result.get("diagnostic_only") is True and all(result.get(key) is False for key in INELIGIBLE),
         "Host diagnostic cannot admit performance or model timing")
    need(all(type(result.get(key)) is int and result[key] == expected
             for key, expected in (("actual_layers", 40), ("actual_probes", 4), ("profile_iterations", 20)))
         and result.get("actual_histories") == HISTORIES
         and all(type(length) is int for length in result["actual_histories"]),
         "Actual host-profile corpus/protocol differs")
    need(result.get("source") == plan["profile_source"]["path"]
         and result.get("source_sha256") == plan["profile_source"]["sha256"]
         and result.get("screen_source") == plan["screen_source"],
         "Profile helper or original screen source identity differs")
    runtime = read(plan["cpp_manifest"]["path"])
    core_paths = [item["path"] for item in runtime["runtime_sources"]
                  if Path(item["path"]).name == "lingbot_fixed_modulation_stageB_cpp.py"]
    need(len(core_paths) == 1, "Exactly one actual CPP core source must be bound")
    core_path = core_paths[0]
    cases = result.get("cases")
    need(type(cases) is list and [(case.get("history"), case.get("probe")) for case in cases]
         == [(length, probe) for length in HISTORIES for probe in range(4)],
         "Actual history/probe matrix differs")
    for case in cases:
        need(type(case["history"]) is int and type(case["probe"]) is int
             and case.get("correctness") is True and set(case.get("arms", {})) == {"native", "stageB"},
             "Actual arm/correctness differs")
        for arm_name, arm in case["arms"].items():
            need(type(arm.get("enqueue_calls")) is int and arm["enqueue_calls"] == 20
                 and type(arm.get("actual_layers_per_call")) is int and arm["actual_layers_per_call"] == 40,
                 "Exactly20 complete40-layer enqueue calls required")
            total = arm.get("cprofile_total_tottime_s")
            need(type(total) in (int, float) and math.isfinite(total) and total >= 0,
                 "Invalid cProfile total time")
            methods = arm.get("methods")
            need(type(methods) is list and bool(methods), "Nonempty cProfile method list required")
            seen = set()
            for method in methods:
                key = validate_stats(method)
                need(key not in seen, "Duplicate cProfile method identity")
                seen.add(key)
                callers = method.get("callers")
                need(type(callers) is list, "cProfile caller list required")
                caller_keys = [validate_stats(caller) for caller in callers]
                need(len(caller_keys) == len(set(caller_keys)), "Duplicate cProfile caller identity")
            # Counters are evidence about the actual producer/consumer scope,
            # not merely claims supplied by enqueue_calls/actual_layers_per_call.
            # Line numbers remain descriptive: source pins already bind bytes.
            for filename, name, calls in ((plan["screen_source"]["path"], "invoke", 20),
                                          (core_path, "prepare_forward", 20 if arm_name == "stageB" else 0),
                                          (core_path, "get", 800 if arm_name == "stageB" else 0)):
                rows = [method for method in methods if method["function"]["filename"] == filename
                        and method["function"]["name"] == name]
                if calls:
                    need(len(rows) == 1 and rows[0]["primitive_calls"] == rows[0]["total_calls"] == calls,
                         "Actual profile scope count differs: " + name)
                else:
                    need(len(rows) <= 1 and all(row["primitive_calls"] == row["total_calls"] == 0 for row in rows),
                         "Native host profile must not call CPP core: " + name)


def verify_inputs(plan):
    need(plan.get("schema") == "lingbot_actual_stageB_host_profile_plan_v1"
         and Path.cwd().resolve() == Path(plan["worktree"]), "Wrong plan/worktree")
    need(not FORMAL_FIELDS.intersection(plan), "Host diagnostic plan cannot carry a formal speed protocol")
    need(plan.get("diagnostic_only") is True and all(plan.get(key) is False for key in INELIGIBLE)
         and plan.get("final_acceptance") is False, "Host diagnostic plan cannot admit performance/model")
    need((plan.get("histories"), plan.get("probes"), plan.get("ranks"), plan.get("actual_layers"),
          plan.get("profile_iterations")) == (HISTORIES, 4, 8, 40, 20), "Host profile plan protocol differs")
    need(plan["base_environment"].get("LINGBOT_STAGEB_HOST_PROFILE") == "1"
         and plan["base_environment"].get("VLLM_LINGBOT_NPU_ASYNC_METADATA") == "0",
         "Explicit host diagnostic and metadata OFF required")
    need(set(plan["configs"]) == {"B"}, "Only CPPv2 candidate service may launch")
    source_check(plan["manifest"])
    verify_cpp_binding(plan)
    for item in (plan["cpp_manifest"], plan["cpp_compilation"], plan["parent_plan"], plan["parent_baseline_config"],
                 plan["env_script"], plan["integrity"], plan["image"], plan["profile_source"], plan["screen_source"],
                 *plan["harness"], *plan["benchmark"], *plan["protected"],
                 *plan["configs"].values(), *plan["launch_scripts"]):
        need(pin(item["path"]) == item, "Frozen input/source changed: " + item["path"])
    runtime = read(plan["cpp_manifest"]["path"])
    need(plan["profile_source"] in runtime["runtime_sources"] and plan["screen_source"] in runtime["runtime_sources"],
         "Both diagnostic sources must belong to the runtime binding")
    need(len(plan["launch_scripts"]) == 1 and plan["launch_scripts"][0]["path"]
         == str(Path(plan["output"]) / "profile1/launch.sh"), "Exactly one fresh profile1 service required")


def run(plan_path):
    plan_path = Path(plan_path).resolve(strict=True)
    plan = read(plan_path)
    plan_pin = pin(plan_path)
    verify_inputs(plan)
    stage = Path(plan["output"]) / "profile1"
    need(not (stage / "launch-intent.json").exists(), "Never restart an observed experiment implicitly")
    need(not list((stage / "services").glob("*/launch.json"))
         and not list((stage / "checks").glob("rank*.json")), "Diagnostic service/evidence must be fresh")
    # Runtime/supervisor checks below remain authoritative; this is a fresh
    # hardware occupancy snapshot, not a global reservation or exclusivity proof.
    with (stage / "devices-before.txt").open("w") as stream:
        subprocess.run(["npu-smi", "info"], stdout=stream, stderr=subprocess.STDOUT, check=True)
    write(stage / "launch-intent.json", dict(plan=plan_pin, diagnostic_only=True,
          begin_boot_ns=time.clock_gettime_ns(time.CLOCK_BOOTTIME)))
    process = None
    launch = run_dir = None
    completed = False
    with (stage / "supervisor.log").open("w") as log:
        process = subprocess.Popen(["bash", str(stage / "launch.sh")], stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(400):
                launches = list((stage / "services").glob("*/launch.json"))
                if launches:
                    need(len(launches) == 1, "Ambiguous owned service")
                    run_dir, launch = launches[0].parent, read(launches[0])
                    break
                need(process.poll() is None, "Supervisor exited before launch receipt")
                time.sleep(.25)
            need(launch is not None and identity(launch["leader"]["pid"]) == launch["leader"], "No live owned launch")
            wanted = ["vllm", "serve", plan["model"], "--omni", "--deploy-config", plan["configs"]["B"]["path"],
                      "--port", str(plan["port"]), "--init-timeout", "1800", "--stage-init-timeout", "1800"]
            need(launch["command"] == wanted and launch["mode"] == "profile"
                 and launch["manifest"] == plan["manifest"]["path"] and launch["leader"]["uid"] == os.getuid()
                 and run_dir.parent.resolve() == (stage / "services").resolve(), "Wrong owned launch/configuration")
            actual = (Path("/proc") / str(launch["leader"]["pid"]) / "environ").read_bytes()
            env = selected_environment(dict(x.decode().split("=", 1) for x in actual.split(b"\0") if b"=" in x))
            need(identity(launch["leader"]["pid"]) == launch["leader"], "Owned leader changed during environment snapshot")
            expected = dict(plan["base_environment"], LINGBOT_STAGEB_OPERATOR_SCREEN_DIR=str(stage / "checks"))
            need({k:v for k,v in env.items() if k not in DYNAMIC} == expected, "Actual selected environment differs")
            need(env.get("ROOT_LINGBOT_SP8_FATAL_DIR") == str(run_dir)
                 and env.get("ROOT_LINGBOT_SP8_FATAL_NONCE") == launch["nonce"]
                 and env.get("ROOT_LINGBOT_SP8_EXPERIMENT_MODE") == "profile", "Fatal ownership identity differs")
            write(stage / "service-environment.json", dict(launch=launch, environment=env, observed_alive=True))
            subprocess.run([sys.executable, "recovery/run_smoke.py", "--run", str(run_dir), "--model", plan["model"],
                "--integrity", plan["integrity"]["path"], "--image", plan["image"]["path"], "--port", str(plan["port"]),
                "--chunks", "8"], check=True)
            checks = []
            for rank in range(8):
                path = stage / f"checks/rank{rank}.json"
                result = read(path)
                validate_rank(plan, result, rank, launch["nonce"])
                checks.append(pin(path))
            need(pin(plan_path) == plan_pin, "Host profile plan changed during execution")
            verify_inputs(plan)
            write(stage / "checks-completed.json", dict(plan=plan_pin, launch=launch, checks=checks,
                  diagnostic_only=True, performance_admitted=False, model_admitted=False,
                  speed_gate_eligible=False, model_timing_eligible=False))
            completed = True
        finally:
            if launch is not None:
                write(stage / "retirement-intent.json", dict(launch=launch, request_completed=completed,
                      begin_boot_ns=time.clock_gettime_ns(time.CLOCK_BOOTTIME)))
                if identity(launch["leader"]["pid"]) == launch["leader"]:
                    retire(launch["leader"])
            if process is not None:
                process.wait(timeout=60)
            if run_dir is not None and (run_dir / "result.json").exists():
                result = read(run_dir / "result.json")
                need(all(result[key] == launch[key] for key in ("leader", "nonce", "command", "mode", "manifest"))
                     and result["remaining_owned_processes"] == [], "Retirement ownership changed or owned workers remain")
                write(stage / "controller-result.json", dict(completed=completed and result["fatal"] is None
                      and result["exit_code"] in (0, -15, -9),
                    exit_code=result["exit_code"], intentional_retirement=True, fatal=result["fatal"],
                    supervisor_return_code=process.returncode, remaining_owned_processes=result["remaining_owned_processes"],
                    launch=pin(run_dir / "launch.json"), result=pin(run_dir / "result.json"),
                    diagnostic_only=True, performance_admitted=False, model_admitted=False,
                    speed_gate_eligible=False, model_timing_eligible=False,
                    end_boot_ns=time.clock_gettime_ns(time.CLOCK_BOOTTIME)))
    final = stage / "controller-result.json"
    need(final.exists() and read(final)["completed"] is True
         and read(final)["fatal"] is None and read(final)["remaining_owned_processes"] == [],
         "Host profile incomplete/fatal; inspect this same launch, never retry implicitly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required for service/device work")
    run(args.plan)
