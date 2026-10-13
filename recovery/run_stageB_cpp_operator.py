"""One fresh, identity-owned service for the StageB actual-region diagnostic."""
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


def verify_cpp_binding(plan):
    """Recheck the complete CPU build ancestry before any device/service call.

    Preparation and the worker loader also validate this chain. Neither replaces
    this check: an unchanged receipt can still refer to a changed library/header.
    This is file validation only; do not import torch or load the guard here.
    """
    for item in (plan["cpp_manifest"], plan["cpp_compilation"], plan["manifest"]):
        need(pin(item["path"]) == item, "CPP binding receipt changed: " + item["path"])
    runtime = read(plan["cpp_manifest"]["path"])
    compiled = read(plan["cpp_compilation"]["path"])
    native = read(plan["manifest"]["path"])
    need(runtime.get("schema") == "lingbot_stageB_guard_runtime_v2"
         and compiled.get("schema") == "lingbot_stageB_guard_cpu_build_v2"
         and runtime.get("namespace") == compiled.get("namespace") == "lingbot_stageB_guard_v2"
         and runtime.get("compilation_ancestor") == plan["cpp_compilation"],
         "CPP runtime must bind this exact compilation ancestor")
    need(compiled.get("registration") is True and compiled.get("npu_initialized") is False
         and not compiled.get("error"), "CPP CPU registration must succeed without initializing NPU")
    for record in (runtime, compiled):
        need(all(record.get(key) is False for key in
                 ("device_admitted", "model_admitted", "v1_qualification_inherited"))
             and record.get("guard_timing") == "per_layer_before_consumption",
             "CPP independent qualification/check timing changed")
    need(runtime.get("historical_admission_inherited") is False
         and compiled.get("performance_admitted") is False,
         "CPP prior device/model/performance admission cannot be inherited")
    for key in ("source", "library", "abi_headers", "abi_libraries", "cxx11_abi", "torch", "torch_npu", "sdk"):
        need(runtime[key] == compiled[key], "CPP runtime/compilation identity differs: " + key)
    need(runtime["runtime_sources"] == native["runtime_sources"]
         and bool(runtime["runtime_sources"]) and bool(compiled["abi_headers"])
         and bool(compiled["abi_libraries"]), "CPP runtime/ABI source coverage differs or is empty")
    for item in (compiled["source"], compiled["library"], compiled["builder"], compiled["provenance"],
                 compiled["source_contract_tests"], compiled["compile_result"],
                 *compiled["abi_headers"], *compiled["abi_libraries"], *runtime["runtime_sources"]):
        need(pin(item["path"]) == item, "CPP dependency changed before launch: " + item["path"])
    terminal = read(compiled["compile_result"]["path"])
    need(terminal.get("schema") == compiled["schema"]
         and terminal.get("compiler_command") == compiled["compiler_command"]
         and terminal.get("compilation_succeeded") is True and terminal.get("exit_code") == 0
         and terminal.get("error") is None and terminal.get("registration_attempted") is False,
         "CPP durable successful compilation terminal differs")
    expected_rejections = ["Frame rejects CPU before NPU StorageImpl access",
                           "Entry rejects CPU before NPU StorageImpl access"]
    rejections = compiled.get("cpu_rejection_cases", [])
    need([case.get("name") for case in rejections] == expected_rejections
         and all(case.get("passed") is True for case in rejections),
         "CPP registration CPU refusal evidence incomplete")


def run(plan_path, number):
    plan_path = Path(plan_path).resolve(strict=True)
    plan = read(plan_path)
    need(plan["schema"] == "lingbot_actual_stageB_cpp_operator_plan_v2"
         and Path.cwd().resolve() == Path(plan["worktree"]), "Wrong plan/worktree")
    source_check(plan["manifest"])
    verify_cpp_binding(plan)
    for item in (plan["cpp_manifest"], plan["cpp_compilation"], plan["parent_plan"], plan["parent_baseline_config"], plan["env_script"],
                 plan["integrity"], plan["image"], *plan["harness"], *plan["benchmark"],
                 *plan["protected"], *plan["configs"].values(), *plan["launch_scripts"]):
        need(pin(item["path"]) == item, "Frozen input/source changed: " + item["path"])
    stage = Path(plan["output"]) / f"operator{number}"
    need(not (stage / "launch-intent.json").exists(), "Never restart an observed experiment implicitly")
    if number > 1:
        previous = read(Path(plan["output"]) / f"operator{number-1}/controller-result.json")
        need(previous["completed"] and previous["remaining_owned_processes"] == [], "Prior service not closed")
    # Runtime/supervisor checks below remain authoritative; this is a fresh
    # hardware occupancy snapshot, not a global reservation or exclusivity proof.
    with (stage / "devices-before.txt").open("w") as stream:
        subprocess.run(["npu-smi", "info"], stdout=stream, stderr=subprocess.STDOUT, check=True)
    write(stage / "launch-intent.json", dict(plan=pin(plan_path), round=number,
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
            expected = dict(plan["base_environment"], LINGBOT_STAGEB_OPERATOR_SCREEN_DIR=str(stage / "checks"))
            need({k:v for k,v in env.items() if k not in DYNAMIC} == expected, "Actual selected environment differs")
            write(stage / "service-environment.json", dict(launch=launch, environment=env, observed_alive=True))
            subprocess.run([sys.executable, "recovery/run_smoke.py", "--run", str(run_dir), "--model", plan["model"],
                "--integrity", plan["integrity"]["path"], "--image", plan["image"]["path"], "--port", str(plan["port"]),
                "--chunks", "8"], check=True)
            checks = []
            for rank in range(8):
                path = stage / f"checks/rank{rank}.json"
                result = read(path)
                need(result["schema"] == "lingbot_actual_stageB_full_region_screen_v1"
                     and result["passed"] and result["rank"] == rank and result["nonce"] == launch["nonce"]
                     and len(result["cases"]) == 24 and result["actual_layers"] == 40,
                     "Missing/incomplete actual operator evidence")
                expected_cases = [(length, probe) for length in plan["histories"] for probe in range(4)]
                need([(c["history"], c["probe"]) for c in result["cases"]] == expected_cases,
                     "Actual history/probe matrix differs")
                need((result["warmups"], result["wall_measurements"], result["event_measurements"]) == (20,30,30),
                     "Actual measurement protocol differs")
                for case in result["cases"]:
                    need(case["correctness"] is True and set(case["arms"]) == {"native", "stageB"}, "Actual arm/correctness differs")
                    for arm in case["arms"].values():
                        for values in (arm["wall_ns"], arm["event_ms"]):
                            need(len(values) == 30 and all(type(x) in (int,float) and math.isfinite(x) and x > 0 for x in values),
                                 "Finite positive30-sample arm required")
                checks.append(pin(path))
            source_check(plan["manifest"])
            verify_cpp_binding(plan)
            write(stage / "checks-completed.json", dict(plan=pin(plan_path), launch=launch, checks=checks))
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
                need(result["leader"] == launch["leader"] and result["nonce"] == launch["nonce"]
                     and result["remaining_owned_processes"] == [], "Owned workers remain")
                write(stage / "controller-result.json", dict(completed=completed and result["fatal"] is None,
                    exit_code=result["exit_code"], intentional_retirement=True, fatal=result["fatal"],
                    supervisor_return_code=process.returncode, remaining_owned_processes=result["remaining_owned_processes"],
                    launch=pin(run_dir / "launch.json"), result=pin(run_dir / "result.json"),
                    end_boot_ns=time.clock_gettime_ns(time.CLOCK_BOOTTIME)))
    final = stage / "controller-result.json"
    need(final.exists() and read(final)["completed"] is True
         and read(final)["fatal"] is None and read(final)["remaining_owned_processes"] == [],
         "Operator screen incomplete/fatal; inspect this same launch before continuing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--round", type=int, choices=(1,2,3), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required for service/device work")
    run(args.plan, args.round)
