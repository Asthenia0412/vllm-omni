"""Six fresh cross-only model runs; client-arrival screening, never final acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from summarize_timing import summarize as summarize_window

ORDER = ("A1", "B1", "B2", "A2", "A3", "B3")


def need(value, message):
    if not value:
        raise ValueError(message)


def read_pinned(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def check_pin(pin):
    value, actual = read_pinned(pin["path"])
    need(actual == pin, "file pin differs: " + str(pin["path"]))
    return value


def screen_gate(runs):
    """Three label-matched pairs, raw MAD of baseline means, no pooling/outlier cuts."""
    need([row["label"] for row in runs] == list(ORDER), "expected fresh A1 B1 B2 A2 A3 B3 order")
    indexed = {row["label"]: row for row in runs}
    for row in runs:
        for key in ("mean_ms", "p99_ms"):
            value = row[key]
            need(type(value) in (int, float) and math.isfinite(value) and value > 0, "invalid timing scalar")
    baseline = [indexed[f"A{i}"]["mean_ms"] for i in (1, 2, 3)]
    median = statistics.median(baseline)
    mad = statistics.median(abs(value - median) for value in baseline)
    threshold = max(5.0, 3.0 * mad)
    pairs = []
    for number in (1, 2, 3):
        a, b = indexed[f"A{number}"], indexed[f"B{number}"]
        pairs.append(dict(pair=number, reference_label=a["label"], candidate_label=b["label"],
                          reference_mean_ms=a["mean_ms"], candidate_mean_ms=b["mean_ms"],
                          improvement_ms=a["mean_ms"] - b["mean_ms"],
                          mean_improved=b["mean_ms"] < a["mean_ms"],
                          reference_p99_ms=a["p99_ms"], candidate_p99_ms=b["p99_ms"],
                          p99_limit_ms=a["p99_ms"] + 5.0,
                          p99_passed=b["p99_ms"] <= a["p99_ms"] + 5.0))
    improvement = statistics.mean(pair["improvement_ms"] for pair in pairs)
    gate = all(pair["mean_improved"] and pair["p99_passed"] for pair in pairs) and improvement >= threshold
    return dict(screen_passed=gate, pairs=pairs, baseline_mean_median_ms=median,
                baseline_mean_raw_mad_ms=mad, required_average_improvement_ms=threshold,
                average_improvement_ms=improvement, average_improvement_passed=improvement >= threshold,
                final_acceptance=False,
                metric="client media inter-arrival; 149 samples, chunks11..159; linear-interpolated P99",
                reference="plain public BSND FIA with historical reconstruction QK",
                candidate="combined C++ cross admission with the same BSND FIA and unchanged QK",
                caveats=["This is six-run candidate screening, not three independent warm40+600 final acceptance.",
                         "This comparison is plain BSND versus combined BSND, not the prior 944 ms SDPA configuration.",
                         "Failed/unadmitted QK combined and async metadata are disabled in both arms.",
                         "The mean/MAD convention is a screening rule, not a statistical significance claim.",
                         "Runs and pairs are kept separate; no trimming, outlier removal or pooling."])


def validate_rollout(plan, run, count, *, command_name="smoke-command.json", exit_name="smoke-exit.json"):
    """Recompute from original arrivals and pin the actual benchmark invocation."""
    from urllib.parse import urlparse
    path = Path(run) / f"rollout{count}.json"
    rollout, rollout_pin = read_pinned(path)
    command, command_pin = read_pinned(Path(run) / command_name)
    status, status_pin = read_pinned(Path(run) / exit_name)
    need(status["exit_code"] == 0, "benchmark process failed")
    expected = ["benchmarks/lingbot_world/benchmark_lingbot_world_realtime.py", "--port", str(plan["port"]),
                "--model", plan["model"], "--image", plan["image"]["path"], "--num-chunks", str(count),
                "--width", "832", "--height", "480", "--fps", "12", "--output-json", str(path), "--print-chunks"]
    need(isinstance(command, list) and command[1:] == expected, "actual client workload command differs")
    need(Path(command[0]).resolve() == (Path(plan["base_environment"]["LINGBOT_VENV"]) / "bin/python").resolve(),
         "client Python executable differs")
    need(rollout["benchmark"] == "lingbot_world_realtime" and rollout["model"] == plan["model"], "wrong benchmark/model")
    url = urlparse(rollout["endpoint"])
    need(url.scheme == "ws" and url.hostname == "127.0.0.1" and url.port == plan["port"]
         and url.path == "/v1/realtime/video" and not url.query and not url.fragment, "wrong endpoint")
    workload, wanted = rollout["workload"], plan["workload"]
    exact_workload = dict(flow_shift=wanted["flow_shift"], fps=12, height=480, width=832,
                          num_chunks=count, num_frames=count * 12 - 3, prompt_updates=[], seed=wanted["seed"],
                          video_seconds=(count * 12 - 3) / 12)
    need(workload == exact_workload, "serialized workload/seed differs")
    config = rollout["config"]
    need(config == dict(camera_pattern="forward", chunk_slo_ms=None, mux_fps=12, playback_buffer_chunks=1,
                        sessions=1, target_fps=12.0, warmup_chunks=6, warmup_sessions=0, workload_file=None),
         "benchmark workload/timing settings differ")
    summary = summarize_window(rollout, count)
    request = rollout["sessions"][0]["request_id"]
    need(isinstance(request, str) and bool(request), "missing request identity")
    return dict(summary=summary, request_id=request, timestamp=rollout["timestamp"],
                rollout=rollout_pin, command=command_pin, exit=status_pin, document=rollout)


def validate_config_pair(plan):
    """Pinned A is fullcapture; B only swaps the two mutually exclusive cross flags."""
    freeze = check_pin(plan["config_freeze"])
    need(freeze["configs"] == plan["configs"] and freeze["manifest"] == plan["manifest"]
         and freeze["image"] == plan["image"] and freeze["integrity"] == plan["integrity"]
         and freeze["model"] == plan["model"], "plan differs from config freeze")
    def pinned_bytes(pin):
        raw = Path(pin["path"]).read_bytes()
        need(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"], "config/source bytes changed")
        return raw
    a, b = pinned_bytes(plan["configs"]["A"]), pinned_bytes(plan["configs"]["B"])
    need(a == pinned_bytes(freeze["source_config"]), "reference differs from frozen fullcapture config")
    for text in (b"lingbot_npu_recovery_cross: true", b"lingbot_npu_recovery_cross_combined: false",
                 b"lingbot_npu_recovery_qk: true", b"lingbot_npu_recovery_qk_combined: false"):
        need(a.count(text) == 1, "reference route flags missing/duplicate")
    expected = a.replace(b"lingbot_npu_recovery_cross: true", b"lingbot_npu_recovery_cross: false").replace(
        b"lingbot_npu_recovery_cross_combined: false", b"lingbot_npu_recovery_cross_combined: true")
    need(b == expected, "candidate changed more than cross routing")
    return freeze


STAGES = ("correctness-A", "correctness-B", "compare") + tuple("timing-" + label for label in ORDER)
DYNAMIC = ("ROOT_LINGBOT_SP8_FATAL_BASE", "ROOT_LINGBOT_SP8_FATAL_DIR", "ROOT_LINGBOT_SP8_FATAL_NONCE", "ROOT_LINGBOT_SP8_EXPERIMENT_MODE")


def verify_bytes(pin):
    raw = Path(pin["path"]).read_bytes()
    need(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"], "source/input pin changed: " + pin["path"])


def stage_record(plan, plan_pin, name):
    data, pin = read_pinned(Path(plan["output"]) / name / "stage.json")
    need(data.get("schema") == "lingbot_cross_model_stage_v1" and data.get("completed") is True
         and data.get("stage") == name and data.get("plan") == plan_pin and data.get("manifest") == plan["manifest"]
         and data.get("final_acceptance") is False, "wrong/incomplete stage or old plan")
    evidence = {}
    for item in data["evidence"]:
        verify_bytes(item)
        need(item["path"] not in evidence, "duplicate evidence path")
        evidence[item["path"]] = item
    return data, pin, evidence


def need_evidence(evidence, path):
    path = str(Path(path).resolve())
    need(path in evidence, "required evidence not bound: " + path)
    return evidence[path]


def owned_stage(plan, data, evidence, name):
    run = Path(data["run"])
    mode, suffix = name.split("-")
    arm = suffix[0]
    need(run.is_absolute() and run.parent == Path(plan["output"]) / name / "services", "run outside this stage")
    launch = check_pin(need_evidence(evidence, run / "launch.json"))
    result = check_pin(need_evidence(evidence, run / "result.json"))
    wanted = ["vllm", "serve", plan["model"], "--omni", "--deploy-config", plan["configs"][arm]["path"],
              "--port", str(plan["port"]), "--init-timeout", "1800", "--stage-init-timeout", "1800"]
    need(launch["command"] == wanted and launch["mode"] == mode and launch["manifest"] == plan["manifest"]["path"]
         and launch["nonce"] == run.name and launch["fatal"] is None, "wrong launch configuration/identity")
    need(data["config"] == plan["configs"][arm] and data["timing_eligible"] is (mode == "timing"), "wrong stage config/scope")
    for key in ("nonce", "leader", "command", "mode", "manifest"):
        need(launch[key] == result[key], "retirement belongs to another service")
    need(result["fatal"] is None and result["remaining_owned_processes"] == []
         and result["exit_code"] in (0, -15, -9) and data["persistent_service_exit_code"] == result["exit_code"], "unclean retirement")
    snapshot = check_pin(data["environment"])
    need(need_evidence(evidence, data["environment"]["path"]) == data["environment"], "environment not in evidence")
    need(snapshot["run"] == str(run) and snapshot["launch"] == launch and snapshot["captured_while_leader_alive"] is True
         and int(snapshot["observed_pid_start"]) == launch["leader"]["start"], "environment belongs to another leader")
    env = snapshot["environment"]
    expected = dict(plan["base_environment"])
    if mode == "correctness":
        expected.update(LINGBOT_CAPTURE_DIR=str(Path(plan["output"]) / name / "captures"),
                        LINGBOT_CAPTURE_RANK="-1", LINGBOT_CAPTURE_ACTIVATIONS="0")
    need({key: value for key, value in env.items() if key not in DYNAMIC} == expected, "pinned selected environment differs")
    need(env["ROOT_LINGBOT_SP8_FATAL_DIR"] == str(run) and env["ROOT_LINGBOT_SP8_FATAL_BASE"] == str(run.parent)
         and env["ROOT_LINGBOT_SP8_FATAL_NONCE"] == launch["nonce"]
         and env["ROOT_LINGBOT_SP8_EXPERIMENT_MODE"] == mode, "supervisor environment differs")
    proof = check_pin(data["retirement"])
    need(need_evidence(evidence, data["retirement"]["path"]) == data["retirement"], "retirement proof not bound")
    need(proof["launch"] == need_evidence(evidence, run / "launch.json") and proof["leader"] == launch["leader"]
         and proof["result"] == need_evidence(evidence, run / "result.json")
         and proof["request_exit"] == need_evidence(evidence, run / "smoke-exit.json")
         and proof["completed_request_before_retirement"] is True, "retirement lacks completed-request binding")
    if mode == "timing":
        need(proof["kind"] == "controller_identity_retirement", "timing requires contemporaneous controller retirement")
        need(type(proof["begin_boot_ns"]) is int and type(proof["end_boot_ns"]) is int
             and 0 < proof["begin_boot_ns"] <= proof["end_boot_ns"], "invalid retirement clock")
    else:
        need(proof["kind"] in ("controller_identity_retirement", "root_manual_retirement"), "missing retirement intent")
        if proof["kind"] == "root_manual_retirement":
            intent = check_pin(proof["contemporaneous_intent"])
            need(intent["leader"] == launch["leader"] and "correctness capture complete" in intent["reason"]
                 and intent["observed_time_ns"] == proof["observed_root_wall_ns"]
                 and proof["post_hoc_binding_only"] is True, "manual adoption lacks root intent")
    need(type(data["clock_ticks_per_second"]) is int and data["clock_ticks_per_second"] > 0
         and data["leader_start_ticks"] == launch["leader"]["start"] and bool(data["boot_id"]), "process clock evidence missing")
    return run, launch, env, proof


def quality_stage(plan, data, evidence, name):
    run, launch, env, proof = owned_stage(plan, data, evidence, name)
    actual = validate_rollout(plan, run, 80)
    for item in (actual["rollout"], actual["command"], actual["exit"]):
        need(need_evidence(evidence, item["path"]) == item, "quality request evidence differs")
    need(data["summary"] == actual["summary"], "quality summary differs from raw arrivals")
    captures, session_ids = [], set()
    for rank in range(8):
        paths = list((Path(plan["output"]) / name / "captures" / f"rank{rank}").glob("*/capture.json"))
        need(len(paths) == 1, "quality capture not fresh/singular")
        capture = check_pin(need_evidence(evidence, paths[0]))
        need(capture["rank"] == rank and capture["timing_eligible"] is False
             and capture["session_id"].startswith(actual["request_id"] + "-")
             and Path(capture["payload_directory"]).resolve() == paths[0].parent.resolve(), "capture identity/owner differs")
        need([row["chunk"] for row in capture["chunks"]] == list(range(80)), "quality chunk coverage incomplete")
        for row in capture["chunks"]:
            need(row["latent_shape"] == [1, 16, 3, 60, 104] and row["latent_dtype"] == "torch.float32", "latent descriptor differs")
            for key in ("latent_path", "rgb_path"):
                if key in row:
                    need(Path(row[key]).name == row[key], "capture path escaped its owner")
            if rank == 0:
                need(row["rgb_shape"] == [9 if row["chunk"] == 0 else 12, 480, 832, 3], "RGB descriptor differs")
            else:
                need("rgb_path" not in row, "unexpected nonprimary RGB")
        captures.append(need_evidence(evidence, paths[0]))
        session_ids.add(capture["session_id"])
    need(len(session_ids) == 1, "ranks captured different requests")
    return dict(stage=name, run=str(run), request_id=actual["request_id"], capture_manifests=captures,
                rollout=actual["rollout"], retirement=data["retirement"])


def timing_stage(plan, data, evidence, name):
    from datetime import datetime
    run, launch, env, proof = owned_stage(plan, data, evidence, name)
    need(set(p.name for p in run.glob("rollout*.json")) == {"rollout40.json", "rollout161.json"}, "service contains extra/old requests")
    warm = validate_rollout(plan, run, 40, command_name="warm40-smoke-command.json", exit_name="warm40-smoke-exit.json")
    measured = validate_rollout(plan, run, 161)
    need(warm["request_id"] != measured["request_id"], "warm and measurement request reused")
    need(datetime.fromisoformat(warm["timestamp"]) < datetime.fromisoformat(measured["timestamp"]), "warm40 did not precede measurement")
    for actual in (warm, measured):
        for key in ("rollout", "command", "exit"):
            need(need_evidence(evidence, actual[key]["path"]) == actual[key], "request evidence not pinned")
    protocol = check_pin(need_evidence(evidence, run / "timing-protocol.json"))
    need(protocol["completed"] is True and protocol["final_acceptance"] is False
         and protocol["warmup_chunks"] == 40 and protocol["measurement_chunks"] == 161
         and protocol["launch"] == launch and protocol["config_sha256"] == data["config"]["sha256"]
         and protocol["manifest_sha256"] == plan["manifest"]["sha256"], "timing protocol differs")
    selected = {k: v for k, v in env.items() if k.startswith(("LINGBOT_", "VLLM_LINGBOT_", "ROOT_LINGBOT_"))
                or k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "ASCEND_RT_VISIBLE_DEVICES")}
    need(protocol["environment"] == selected, "timing protocol/live environment differs")
    summary = measured["summary"]
    need(summary["window_inclusive"] == [11, 159] and summary["sample_count"] == 149, "main window differs")
    saved = check_pin(need_evidence(evidence, run / "main-window.json"))
    need(saved == dict(summary, input_sha256=measured["rollout"]["sha256"]) and data["summary"] == summary,
         "saved summary differs from raw untrimmed window")
    return dict(label=name.removeprefix("timing-"), run=str(run), nonce=launch["nonce"], leader=launch["leader"],
                boot_id=data["boot_id"], clock_ticks_per_second=data["clock_ticks_per_second"],
                start_boot_ns=data["leader_start_ticks"] * 1_000_000_000 // data["clock_ticks_per_second"],
                retirement_begin_boot_ns=proof["begin_boot_ns"], retirement_end_boot_ns=proof["end_boot_ns"],
                source_commit=protocol["source_commit"], worktree_status=protocol["worktree_status"],
                warm_request_id=warm["request_id"], measured_request_id=measured["request_id"],
                warm_rollout=warm["rollout"], measured_rollout=measured["rollout"], **summary)


def summarize_screen(plan_path, expected_source_commit=None):
    plan, plan_pin = read_pinned(plan_path)
    need(plan["schema"] == "lingbot_cross_only_model_screen_v1" and tuple(plan["order"]) == STAGES
         and plan["plan_path"] == str(Path(plan_path).resolve()) and plan["final_acceptance"] is False, "wrong screening plan")
    validate_config_pair(plan)
    base = plan["base_environment"]
    need(base["ASCEND_RT_VISIBLE_DEVICES"] == "8,9,10,11,12,13,14,15" and base["VLLM_LINGBOT_NPU_ASYNC_METADATA"] == "0"
         and base["VLLM_LINGBOT_NPU_ATTENTION_BACKEND"] == "fia_bsnd" and base["LINGBOT_RECOVERY_RECV"] == "1"
         and base["LINGBOT_RECOVERY_MANIFEST"] == plan["manifest"]["path"]
         and base["LINGBOT_RECOVERY_MANIFEST_SHA256"] == plan["manifest"]["sha256"], "wrong baseline/backend/environment")
    need(not any(k.startswith("LINGBOT_QK_COMBINED_") or "CAPTURE" in k or "PROFILE" in k or "DIAGNOSTIC" in k
                 for k in base), "experimental or timing-contaminating environment")
    need("LINGBOT_SP8_OBSERVE_MODE" not in base, "observer enabled")
    for item in (*plan["harness_pins"], *plan["launch_scripts"].values(), plan["image"], plan["integrity"]):
        verify_bytes(item)
    integrity = check_pin(plan["integrity"])
    need(integrity["complete"] is True and integrity["passed"] is True and integrity["revision"] == plan["model_revision"], "model identity failed")
    manifest = check_pin(plan["manifest"])
    for item in (manifest["source"], manifest["library"], manifest["kernels"], *manifest["kernel_sources"], *manifest["runtime_sources"], *manifest["abi_headers"]):
        verify_bytes(item)
    stage_pins, records, evidence_by_stage = [], {}, {}
    for name in STAGES:
        records[name], pin, evidence_by_stage[name] = stage_record(plan, plan_pin, name)
        stage_pins.append(pin)
    quality = [quality_stage(plan, records[name], evidence_by_stage[name], name) for name in STAGES[:2]]
    need(quality[0]["request_id"] != quality[1]["request_id"], "paired model requests reused")
    compare_evidence = evidence_by_stage["compare"]
    paired = check_pin(need_evidence(compare_evidence, Path(plan["output"]) / "compare/paired80.json"))
    for pin in stage_pins[:2]:
        need(need_evidence(compare_evidence, pin["path"]) == pin, "comparison not bound to these correctness runs")
    need(paired["passed"] is True and paired["reference"] == str(Path(plan["output"]) / "correctness-A/captures")
         and paired["candidate"] == str(Path(plan["output"]) / "correctness-B/captures")
         and [(r["rank"], r["latent_chunks"], r["rgb_frames"]) for r in paired["ranks"]]
         == [(rank, 80, 957 if rank == 0 else 0) for rank in range(8)], "fresh paired80 prerequisite failed")
    runs = [timing_stage(plan, records[name], evidence_by_stage[name], name) for name in STAGES[3:]]
    need(len({r["nonce"] for r in runs}) == 6 and len({(r["leader"]["pid"], r["leader"]["start"]) for r in runs}) == 6,
         "timing did not use six fresh processes")
    commits = {r["source_commit"] for r in runs}
    need(len(commits) == 1 and all(len(c) == 40 and all(x in "0123456789abcdef" for x in c) for c in commits),
         "source commit changed between six fresh services")
    if expected_source_commit is not None:
        need(8 <= len(expected_source_commit) <= 40 and next(iter(commits)).startswith(expected_source_commit),
             "six-run source commit differs from predeclared commit")
    need(len({r["boot_id"] for r in runs}) == 1 and len({r["clock_ticks_per_second"] for r in runs}) == 1, "incomparable process clocks")
    for before, after in zip(runs, runs[1:]):
        need(after["start_boot_ns"] >= before["retirement_end_boot_ns"], "timing services overlap or order differs")
    request_ids = [x for row in runs for x in (row["warm_request_id"], row["measured_request_id"])] + [q["request_id"] for q in quality]
    need(len(set(request_ids)) == len(request_ids), "request identity reused across fresh runs")
    result = screen_gate(runs)
    source = Path(__file__).resolve().read_bytes()
    return dict(schema="lingbot_cross_only_six_model_screen_v1", plan=plan_pin, stage_pins=stage_pins,
                summary_source=dict(path=str(Path(__file__).resolve()), bytes=len(source), sha256=hashlib.sha256(source).hexdigest()),
                frozen_source_commit=next(iter(commits)), expected_source_commit=expected_source_commit,
                quality80_passed=True, paired80=need_evidence(compare_evidence, Path(plan["output"]) / "compare/paired80.json"),
                quality80_runs=quality, runs=runs, **result,
                environment_scope="Pinned selected experiment/runtime keys; not a claim that every process environment variable was captured.",
                source_scope="Runtime/host/kernel/header/benchmark/controller pins checked; original historical checkpoint identity is not thereby established.",
                quality_scope="Fresh eight-rank80-chunk latent plus957-rootRGB bit parity; no browser/lifecycle/full-video-quality/final600 acceptance.",
                prerequisite_order_scope="The A1-after-paired80 prerequisite is enforced by the pinned launch/controller preflight. The compare stage has no completion clock, so this is not an independent timestamp proof of that first transition. Six timing services have separate non-overlap checks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-commit", help="Predeclared full SHA or prefix of at least eight hexadecimal digits")
    args = parser.parse_args()
    need(not args.output.exists(), "refuse to overwrite a six-run screen")
    result = summarize_screen(args.plan, args.expected_source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("screen_passed", "pairs", "baseline_mean_raw_mad_ms",
          "required_average_improvement_ms", "average_improvement_ms", "final_acceptance")}, indent=2))


if __name__ == "__main__":
    main()
