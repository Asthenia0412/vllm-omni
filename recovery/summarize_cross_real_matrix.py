"""CPU-only aggregation of three actual192 complete cross-call timing rounds."""
import argparse
import json
import math
from pathlib import Path
import shlex
import statistics

import check_cross_real_matrix as replay


def samples(values, name):
    replay.need(len(values) == 30 and all(type(v) in (int, float) and math.isfinite(v) and v > 0
                                        for v in values), f"invalid30 {name} samples")
    return values


def serial_launch(path, directory, number, matrix, correctness_dir):
    """Admit the exact owned serial shell plan; never interpret/execute its text."""
    path = Path(path)
    result = json.loads(path.read_text())
    launch_path = path.parent / "launch.json"
    launch = json.loads(launch_path.read_text())
    replay.need(all(result[key] == launch[key] for key in
                    ("nonce", "leader", "command", "mode", "manifest", "fatal")),
                "supervisor launch/result identities differ")
    replay.need(result["mode"] == "timing" and result["exit_code"] == 0
                and not result["remaining_owned_processes"] and result["fatal"] is None
                and result["manifest"] == matrix["manifest"]["path"],
                "timing supervisor did not finish the expected build cleanly")
    command = result["command"]
    replay.need(len(command) == 3 and command[:2] == ["bash", "-c"], "exact serial bash launcher required")
    lines = command[2].strip().splitlines()
    replay.need(len(lines) == 4 and lines[0] == "set -e"
                and lines[1] == "for replay_rank in 0 1 2 3 4 5 6 7; do" and lines[3] == "done",
                "serial8 loop/order changed")
    tokens = shlex.split(lines[2])
    replay.need(tokens[:2] == ["python", "recovery/check_cross_real_matrix.py"]
                and len(tokens[2:]) % 2 == 0, "unexpected replay interpreter/command")
    pairs = list(zip(tokens[2::2], tokens[3::2], strict=True))
    options = dict(pairs)
    replay.need(len(options) == len(pairs), "duplicate CLI option in launcher")
    expected = {"--capture-plan": matrix["capture_plan"]["path"],
                "--matrix-audit": matrix["capture_audit"]["path"], "--mode": "timing",
                "--round": str(number), "--correctness-dir": str(Path(correctness_dir).resolve()),
                "--rank": "$replay_rank", "--device": "$replay_rank",
                "--output": str(Path(directory).resolve() / "rank$replay_rank.json")}
    replay.need(options == expected, "exact replay CLI is not bound to this capture/round/output")
    return result, dict(launch=replay.pin(launch_path), result=replay.pin(path))


def summarize(correctness_dir, timing_dirs, supervisor_results):
    replay.need(len(timing_dirs) == 3 and len({Path(p).resolve() for p in timing_dirs}) == 3,
                "three distinct timing directories required")
    replay.need(len(supervisor_results) == 3, "three clean launch receipts required")
    reference = json.loads((Path(correctness_dir) / "rank0.json").read_text())
    matrix, sources = reference["matrix"], reference["sources"]
    launches, leader_identities, nonces = [], set(), set()
    for number, (directory, path) in enumerate(zip(timing_dirs, supervisor_results, strict=True), 1):
        launch, launch_pins = serial_launch(path, directory, number, matrix, correctness_dir)
        leader = launch["leader"]
        leader_identities.add((leader["pid"], leader["start"]))
        nonces.add(launch["nonce"])
        launches.append(launch_pins)
    replay.need(len(leader_identities) == len(nonces) == 3, "three independent fresh launches required")
    replay.need(sources == replay.source_pins(), "frozen replay sources changed")
    correctness = replay.verify_correctness(correctness_dir, matrix, sources)
    summaries, all_case_rows, report_pins = [], [], []
    for round_number, directory in enumerate(timing_dirs, 1):
        rows, boot_ids = [], set()
        for rank in range(8):
            path = Path(directory) / f"rank{rank}.json"
            report = json.loads(path.read_text())
            replay.need(report["schema"] == replay.SCHEMA and report["mode"] == "timing"
                        and report["passed"] is True and report["round"] == round_number
                        and report["rank"] == report["device"] == rank, "timing mode/rank/round differs")
            replay.need(report["matrix_sha256"] == matrix["sha256"]
                        and report["manifest"] == matrix["manifest"] and report["sources"] == sources
                        and report["correctness_receipts"] == correctness, "timing admission pins differ")
            replay.need(report["visible_devices"] == "8,9,10,11,12,13,14,15"
                        and report["actual_cross_cases_completed"] == 24
                        and report["full_attention_forward"] is True
                        and report["constructor_context"] == "isolated_from_captured_config"
                        and report["all40_payloads"] is False, "timing device/caller/scope differs")
            boot_ids.add(report["boot_id"])
            cases = report["cases"]
            replay.validate_order([case["identity"] for case in cases])
            for case, wanted in zip(cases, matrix["ranks"][rank]["cases"], strict=True):
                replay.need(case["identity"] == wanted["identity"] and case["fixture"] == wanted["fixture"]
                            and case["spy_enabled"] is False and case["full_owner_sha256_verified"] is True
                            and case["numerical_checks_in_timing"] is False,
                            "wrong real case or correctness work in timing")
                order = ["combined", "plain"] if round_number == 2 else ["plain", "combined"]
                replay.need(case["order"] == order and set(case["arms"]) == set(replay.ARMS),
                            "missing arm or incorrect A/B order")
                means = {}
                previous_phase_end = None
                for arm in order:
                    record = case["arms"][arm]
                    replay.need(type(record["phase_start_ns"]) is int and type(record["phase_end_ns"]) is int
                                and record["phase_start_ns"] < record["phase_end_ns"]
                                and (previous_phase_end is None or previous_phase_end <= record["phase_start_ns"]),
                                "invalid or overlapping A/B phase intervals")
                    previous_phase_end = record["phase_end_ns"]
                for arm in replay.ARMS:
                    record = case["arms"][arm]
                    replay.need(record["warmup"] == 20, "warmup differs")
                    wall, device = samples(record["wall_ms"], "wall"), samples(record["device_ms"], "device")
                    replay.need(len(record["wall_windows_ns"]) == 30, "missing host timing windows")
                    previous_end = record["phase_start_ns"]
                    for latency, (start, end) in zip(wall, record["wall_windows_ns"], strict=True):
                        replay.need(type(start) is int and type(end) is int
                                    and previous_end <= start < end <= record["phase_end_ns"]
                                    and math.isclose((end-start)/1e6, latency, rel_tol=1e-12),
                                    "invalid, repeated or overlapping wall interval")
                        previous_end = end
                    wall_mean, device_mean = statistics.mean(wall), statistics.mean(device)
                    replay.need(math.isclose(wall_mean, record["wall_mean_ms"], rel_tol=1e-12)
                                and math.isclose(device_mean, record["device_mean_ms"], rel_tol=1e-12),
                                "saved means differ from actual samples")
                    means[arm] = dict(wall_ms=wall_mean, device_ms=device_mean)
                rows.append(dict(rank=rank, **{key: case["identity"][key] for key in ("kv_len", "probe")},
                                 **means,
                                 wall_speedup_fraction=1-means["combined"]["wall_ms"]/means["plain"]["wall_ms"],
                                 wall_regression_fraction=means["combined"]["wall_ms"]/means["plain"]["wall_ms"]-1))
            report_pins.append(replay.pin(path))
        replay.need(len(boot_ids) == 1 and len(rows) == 192, "one complete same-host actual192 round required")
        plain = statistics.mean(row["plain"]["wall_ms"] for row in rows)
        combined = statistics.mean(row["combined"]["wall_ms"] for row in rows)
        speedup = 1-combined/plain
        worst = max(rows, key=lambda row: row["wall_regression_fraction"])
        gate = speedup >= .10 and worst["wall_regression_fraction"] <= .05
        summaries.append(dict(round=round_number, actual_cases=192, plain_wall_mean_ms=plain,
                              combined_wall_mean_ms=combined, wall_speedup_percent=100*speedup,
                              plain_device_mean_ms=statistics.mean(row["plain"]["device_ms"] for row in rows),
                              combined_device_mean_ms=statistics.mean(row["combined"]["device_ms"] for row in rows),
                              worst_case=worst, worst_case_regression_percent=100*worst["wall_regression_fraction"],
                              average_10percent_passed=speedup >= .10,
                              worst_case_5percent_passed=worst["wall_regression_fraction"] <= .05,
                              gate_passed=gate))
        all_case_rows.append(dict(round=round_number, cases=rows))
    return dict(schema="lingbot_cross_actual192_timing_aggregate_v1", complete_three_rounds=True,
                gate_passed=all(item["gate_passed"] for item in summaries), rounds=summaries,
                matrix_sha256=matrix["sha256"], manifest=matrix["manifest"], sources=sources,
                correctness_receipts=correctness, timing_receipts=report_pins, case_results=all_case_rows,
                full_call_scope=reference["full_call_scope"], real_payload_layers=[0], all40_payloads=False,
                model_admitted=False, end_to_end_performance_validated=False,
                statistical_significance_claimed=False, historical_full_combo_reproduced=False,
                weighting="equal weight per actual rank/history/probe case; no sum of cross-rank durations",
                method="20warm +30 synchronized complete-call wall +30 separate device-event samples per arm",
                caveats=reference["timing_caveats"],
                independent_process_launches_verified=True, supervisor_results=launches,
                serial_rank_launcher_verified=True,
                rank_execution_scope="8 serial fresh Python workers per independent round; not concurrent model performance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness-dir", type=Path, required=True)
    parser.add_argument("--timing-dirs", nargs=3, type=Path, required=True)
    parser.add_argument("--supervisor-results", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay.need(not args.output.exists(), "refusing to replace earlier aggregation")
    result = summarize(args.correctness_dir, args.timing_dirs, args.supervisor_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(gate_passed=result["gate_passed"], rounds=result["rounds"]), indent=2))
