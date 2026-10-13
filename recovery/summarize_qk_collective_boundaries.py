"""New-wrapper host-boundary diagnostic only; no gate, network or lock inference.

Explicit adaptation: reuse only scalar statistics, interval/GC clipping and pin
helpers from summarize_qk_host_dependencies. New schema/correctness/row admission
is independent; old broad diagnostic reports cannot be relabelled or admitted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import types

import summarize_qk_host_dependencies as scalar
import observe_qk_collective_host as fingerprint_api

need = scalar.need
SCHEMA = "lingbot_qk_collective_boundaries_diagnostic_v2"
METHOD = "frozen_per_call_sync_private_constructor_and_scoped_module_delegates_v2"
OBSERVER_SCHEMA = "lingbot_prepared_collective_host_boundaries_v2"
MARKS = ("call_begin", "allocation_begin", "allocation_end", "dist_begin", "dist_end", "call_end")
INTERVALS = (("guard_before_allocation", 0, 1), ("receive_allocation", 1, 2),
             ("allocation_to_dist_bridge", 2, 3), ("public_dist_call", 3, 4),
             ("output_layout_and_return", 4, 5))
STATUS = {"call_raised": 1, "unexpected_boundary": 2, "foreign_thread": 4,
          "concurrent_invocation": 8, "globals_conflict": 16, "observation_failed": 32}
EXPECTED = [(kv, p) for kv in (4680, 9360, 14040, 18720, 23400, 28080) for p in range(4)]
CHECKS = ["expanded_coefficients", "captured_reference_hccl", "baseline_send_and_receive",
          "sentinel_send", "full_arena_after_pack", "sentinel_receive", "full_arena_after_hccl",
          "fresh_owner_extent_offset_and_distinctness", "fresh_sends_and_receives", "first_owner_not_overwritten",
          "full_source_and_reference_owners", "source_descriptors_unchanged", "missing_snapshot_refuses_without_producer_mutation"]
INSTRUMENTED = ["instrumented_send_and_original_receive_for_both_arms_and_scopes", "instrumented_source_owners_unchanged"]


def check_pin(pin):
    result, actual = scalar.read_pinned(pin["path"])
    need(actual == pin, "receipt pin changed")
    return result


def source_pin(pin):
    need(scalar.read_pinned_source(pin["path"]) == pin, "source/build pin changed: " + pin["path"])


def claim_flags(report):
    for key, wanted in (("diagnostic_only", True), ("performance_timing_eligible", False),
                        ("gate_passed", False), ("model_admitted", False), ("original_hccl_included", True),
                        ("all40_payloads", False)):
        need(report.get(key) is wanted, "invalid scope/claim: " + key)
    need(report.get("real_payload_layers") == [0] and report.get("actual_payload_cases_per_rank") == 24,
         "wrong actual-case scope")


def common_report(report, rank, mode):
    need(report.get("schema") == SCHEMA and report.get("method") == METHOD and report.get("mode") == mode
         and report.get("rank") == rank and report.get("passed") is True, "new successful wrapper report required")
    claim_flags(report)
    cases = report.get("cases", [])
    need(len(cases) == 24 and [(x["identity"]["kv_len"], x["identity"]["probe"]) for x in cases] == EXPECTED, "actual24 order incomplete")
    matrix = report["matrix_rank"]
    need(matrix["rank"] == rank and len(matrix["cases"]) == 24, "rank matrix missing")
    for item, expected in zip(cases, matrix["cases"]):
        need(item["identity"]["rank"] == rank and item["identity"]["layer"] == 0 and item["passed"] is True
             and item["full_owner_sha256_verified"] is True and item["identity"] == expected["identity"]
             and item["fixture"] == expected["file"], "case or fixture binding differs")
    return cases


def boundary_rows(data, expected_count, code):
    need(data.get("schema") == OBSERVER_SCHEMA and data.get("diagnostic_only") is True and data.get("performance_gate") is False
         and data.get("row_count") == expected_count and tuple(data.get("marks", [])) == MARKS
         and tuple(data.get("columns", [])) == scalar.COLS and data.get("intervals") == [list(v) for v in INTERVALS]
         and data.get("status_bits") == STATUS and data.get("terminal_failure") is False
         and data.get("all_rows_complete") is True, "new observer schema or coverage differs")
    need(data.get("pins") == {k: code[k] for k in ("source_path", "source_sha256", "code_sha256", "code_fingerprint_method")}, "observed code differs")
    need(data.get("status") == [0] * expected_count and data.get("visited") == [1] * expected_count
         and data.get("restored_original_bindings") == [1] * expected_count
         and data.get("complete_rows") == [True] * expected_count, "incomplete/failed/restoration-missing boundary")
    for key, expected in (("status", 0), ("visited", 1), ("restored_original_bindings", 1)):
        need(all(type(v) is int and v == expected for v in data[key]), "integer observer flags required")
    need(all(type(v) is bool and v for v in data["complete_rows"]), "boolean completion flags required")
    flat = data["data"]
    need(len(flat) == expected_count * 24 and all(type(v) is int and v >= 0 for v in flat), "invalid boundary array")
    rows = []
    for number in range(expected_count):
        row = [flat[number * 24 + i * 4:number * 24 + (i + 1) * 4] for i in range(6)]
        need(all(all(a <= b for a, b in zip(x, y)) for x, y in zip(row, row[1:])), "nonmonotonic boundary observation")
        rows.append(row)
    need(type(data["native_thread_id"]) is int and data["native_thread_id"] > 0, "observer thread missing")
    return rows


def outer_rows(report):
    """Independent new-schema validator; never mutates a record to fit old schema."""
    cases = common_report(report, report["rank"], "profile")
    trace = report["trace"]
    need(tuple(trace["marks"]) == scalar.MARKS and tuple(trace["columns"]) == scalar.COLS
         and trace["row_count"] == 5760 and len(trace["data"]) == 5760 * 56, "outer trace layout/extent differs")
    need(all(type(v) is int for v in trace["data"]), "outer observations must be integers")
    alive = trace["old_wrapper_alive_after_replace"]
    need(trace["old_wrapper_columns"] == ["lease", "receive_tensor"] and len(alive) == 5760 * 2
         and all(type(v) is int and v in (0, 1) for v in alive), "release flags missing")
    rows, expected_row, previous_end = [], 0, -1
    for case in cases:
        need(len(case["phases"]) == 4, "four phases required")
        for phase, (scope, arm) in zip(case["phases"], [(s, a) for s in scalar.SCOPES for a in scalar.ARMS]):
            need((phase["scope"], phase["arm"], phase["first_row"], phase["row_count"], phase["warmup"])
                 == (scope, arm, expected_row, 60, 20), "phase mapping differs")
            bounds = phase["phase_boundaries"]
            need(len(bounds) == 2 and [b["name"] for b in bounds] == ["warm20", "diagnostic_calls"]
                 and all(type(b[k]) is int for b in bounds for k in ("enter_ns", "ready_ns")), "phase boundaries invalid")
            need(bounds[0]["enter_ns"] <= bounds[0]["ready_ns"] <= bounds[1]["enter_ns"] <= bounds[1]["ready_ns"], "phase times unordered")
            producer = scope == scalar.SCOPES[0]
            for sample in range(60):
                index = expected_row + sample
                flat = trace["data"][index * 56:(index + 1) * 56]
                row = [flat[i:i + 4] for i in range(0, 56, 4)]
                points = list(range(14)) if producer else [0, 1] + list(range(4, 14))
                need(all(v >= 0 for point in points for v in row[point]), "outer mark missing")
                need(producer or row[2] == row[3] == [-1] * 4, "cached producer marks must be absent")
                need(all(all(a <= b for a, b in zip(row[x], row[y])) for x, y in zip(points, points[1:])), "outer counters unordered")
                need(row[0][0] >= max(previous_end, bounds[1]["ready_ns"]), "outer samples overlap/precede phase")
                previous_end = row[13][0]
                rows.append(dict(rank=report["rank"], row=index, kv_len=case["identity"]["kv_len"],
                    probe=case["identity"]["probe"], scope=scope, arm=arm, sample=sample, producer=producer,
                    marks=row, old_wrapper_alive=alive[index * 2:index * 2 + 2]))
            expected_row += 60
    need(expected_row == 5760, "incomplete row coverage")
    return rows


def code_binding(reference):
    code = reference["expected_collective_code"]
    source = reference["sources"]["original_collective"]
    source_pin(source)
    need(code["source_path"] == source["path"] and code["source_sha256"] == source["sha256"]
         and code["python"] == sys.version and type(code["optimize"]) is int and code["optimize"] in (0, 1, 2), "code/platform pin differs")
    need(code["code_fingerprint_method"] == fingerprint_api.CODE_FINGERPRINT_METHOD == "python_typed_code_fields_v1",
         "old raw marshal fingerprint is not admitted")
    need(scalar.read_pinned_source(fingerprint_api.__file__) == reference["diagnostic_sources"]["observer"],
         "fingerprint helper differs from the new observer source")
    compiled = compile(Path(source["path"]).read_bytes(), source["path"], "exec", dont_inherit=True, optimize=code["optimize"])
    candidates = [x for x in compiled.co_consts if isinstance(x, types.CodeType) and x.co_name == "all_to_all_5D"]
    need(len(candidates) == 1 and fingerprint_api.code_fingerprint(candidates[0]) == code["code_sha256"], "trusted code fingerprint differs")


def correctness_chain(reports):
    reference = reports[0]
    need(len(reference["correctness_receipts"]) == len(reference["diagnostic_correctness_receipts"]) == 8, "eight-rank correctness chains required")
    fresh = []
    for rank in range(8):
        base = check_pin(reference["correctness_receipts"][rank])
        need(base.get("schema") == "lingbot_qk_real_matrix_replay_v1" and base.get("mode") == "correctness"
             and base.get("rank") == rank and base.get("passed") is True, "original actual192 correctness missing")
        new = check_pin(reference["diagnostic_correctness_receipts"][rank])
        cases = common_report(new, rank, "correctness")
        for data in (base, new):
            need(data["matrix_sha256"] == reference["matrix_sha256"] and data["manifest"] == reference["manifest"]
                 and data["sources"] == reference["sources"] and data["matrix_rank"] == reports[rank]["matrix_rank"], "correctness provenance differs")
        need(new["diagnostic_sources"] == reference["diagnostic_sources"] and new["expected_collective_code"] == reference["expected_collective_code"]
             and new["correctness_receipts"] == reference["correctness_receipts"], "new wrapper chain differs")
        need(len(base["cases"]) == 24, "original actual192 cases missing")
        for index, (old_case, case, current) in enumerate(zip(base["cases"], cases, reports[rank]["cases"])):
            for item in (old_case, case):
                need(item["identity"] == current["identity"] and item["fixture"] == current["fixture"]
                     and item["passed"] is True and item["checks"] == CHECKS and item["full_owner_sha256_verified"] is True,
                     "original/new correctness case incomplete")
            need(case["instrumented_checks"] == INSTRUMENTED and case["boundary_rows"] == dict(first_row=index * 4, row_count=4),
                 "new observer correctness rows/checks missing")
        boundary_rows(new["collective_boundaries"], 96, reference["expected_collective_code"])
        fresh.append(new)
    return fresh


def bind_supervisor(directory, mode, output, reference):
    launch, lp = scalar.read_pinned(Path(directory) / "launch.json")
    result, rp = scalar.read_pinned(Path(directory) / "result.json")
    need(all(launch[k] == result[k] for k in ("nonce", "leader", "command", "mode", "manifest")), "supervisor identity differs")
    need(result["mode"] == mode and result["exit_code"] == 0 and result["fatal"] is None and launch["fatal"] is None
         and result["remaining_owned_processes"] == [], "diagnostic supervisor not cleanly retired")
    command = result["command"]
    prefix = ["torchrun", "--standalone", "--nnodes=1", "--nproc-per-node=8", "recovery/diagnose_qk_collective_boundaries.py", "--execute"]
    need(command[:6] == prefix and (len(command) - 6) % 2 == 0, "new wrapper launcher differs")
    options = dict(zip(command[6::2], command[7::2]))
    keys = {"--mode", "--fixtures", "--manifest", "--correctness-dir", "--output"}
    if mode == "profile":
        keys.add("--diagnostic-correctness-dir")
    need(len(options) == (len(command) - 6) // 2 and set(options) == keys, "launcher options duplicated/different")
    need(options["--mode"] == mode and Path(options["--output"]).resolve() == Path(output).resolve()
         and options["--manifest"] == reference["manifest"]["path"], "launcher/report mapping differs")
    manifest = check_pin(reference["manifest"])
    need(result["manifest"] == manifest["provider_manifest"]["path"], "supervisor provider differs")
    check_pin(manifest["provider_manifest"])
    for rank in range(8):
        _, actual = scalar.read_pinned(Path(options["--correctness-dir"]) / f"rank{rank}.json")
        need(actual == reference["correctness_receipts"][rank], "base correctness directory differs")
        if mode == "profile":
            _, actual = scalar.read_pinned(Path(options["--diagnostic-correctness-dir"]) / f"rank{rank}.json")
            need(actual == reference["diagnostic_correctness_receipts"][rank], "new correctness directory differs")
    # Exact source metadata path prefix, not a substring check or payload rehash.
    root = Path(options["--fixtures"]).resolve()
    for rank, receipt in enumerate(reference["diagnostic_correctness_receipts"] if mode == "profile" else []):
        data = check_pin(receipt)
        for entry in data["cases"]:
            need(Path(entry["fixture"]["path"]).resolve().parent == root / f"rank{rank}", "fixture CLI differs from actual matrix")
    return dict(launch=lp, result=rp, nonce=result["nonce"], options=options)


def peers(outer, inner, index):
    values = []
    for rank in range(8):
        record, marks = outer[rank][index], inner[rank][index]
        item = dict(rank=rank, inner_timestamps=dict(zip(MARKS, [v[0] for v in marks])),
                    outer_timestamps=dict(zip(scalar.MARKS, [v[0] for v in record["marks"]])),
                    inner_deltas={name: dict(wall_ns=marks[end][0] - marks[begin][0],
                        thread_cpu_ns=marks[end][1] - marks[begin][1],
                        voluntary_switches=marks[end][2] - marks[begin][2],
                        involuntary_switches=marks[end][3] - marks[begin][3]) for name, begin, end in INTERVALS})
        values.append(item)
    starts = [x["inner_timestamps"]["dist_begin"] for x in values]
    returns = [x["inner_timestamps"]["dist_end"] for x in values]
    return dict(public_dist_host_start_skew_ns=max(starts) - min(starts), latest_host_start_rank=starts.index(max(starts)),
                public_dist_host_return_skew_ns=max(returns) - min(returns), latest_host_return_rank=returns.index(max(returns)), peers=values)


def summarize(profiles, profile_run, correctness_run, top=20):
    reports, pins, outer, inner = [], [], [], []
    for rank in range(8):
        report, pin = scalar.read_pinned(Path(profiles) / f"rank{rank}.json")
        common_report(report, rank, "profile")
        outer.append(outer_rows(report))
        inner.append(boundary_rows(report["collective_boundaries"], 5760, report["expected_collective_code"]))
        need(report["collective_boundaries"]["native_thread_id"] == report["trace"]["main_native_thread_id"], "inner/outer thread differs")
        for outside, inside in zip(outer[-1], inner[-1]):
            need(all(a <= b <= c <= d for a, b, c, d in zip(outside["marks"][7], inside[0], inside[5], outside["marks"][8])),
                 "inner observation not inside original outer collective span")
        reports.append(report); pins.append(pin)
    reference = reports[0]
    for report in reports[1:]:
        for key in ("manifest", "sources", "diagnostic_sources", "expected_collective_code", "correctness_receipts",
                    "diagnostic_correctness_receipts", "matrix_sha256", "boot_id", "main_call_changes"):
            need(report[key] == reference[key], "rank provenance differs: " + key)
        for a, b in zip(reference["cases"], report["cases"]):
            need({k: v for k, v in a["identity"].items() if k != "rank"} == {k: v for k, v in b["identity"].items() if k != "rank"}, "cross-rank case differs")
    for pin in list(reference["sources"].values()) + list(reference["diagnostic_sources"].values()):
        source_pin(pin)
    code_binding(reference)
    fresh = correctness_chain(reports)
    new_dir = Path(reference["diagnostic_correctness_receipts"][0]["path"]).parent
    runs = dict(profile=bind_supervisor(profile_run, "profile", profiles, reference),
                correctness=bind_supervisor(correctness_run, "correctness", new_dir, fresh[0]))
    need(runs["profile"]["nonce"] != runs["correctness"]["nonce"], "profile reused correctness service")
    need(runs["profile"]["options"]["--fixtures"] == runs["correctness"]["options"]["--fixtures"], "profile/correctness fixture roots differ")
    stats, by_group = defaultdict(scalar.Stats), defaultdict(scalar.Stats)
    gc_status, stalls, totals, gc_hits = [], [], [], []
    release = dict(lease_alive=0, receive_wrapper_alive=0)
    for rank in range(8):
        intervals, status = scalar.gc_intervals(reports[rank]["gc"])
        gc_status.append(dict(rank=rank, **status))
        thread = reports[rank]["trace"]["main_native_thread_id"]
        for outer_item, marks in zip(outer[rank], inner[rank]):
            identity = {key: outer_item[key] for key in ("rank", "row", "kv_len", "probe", "scope", "arm", "sample")}
            parts = {name: scalar.interval(marks, begin, end, intervals, thread) for name, begin, end in INTERVALS}
            total = scalar.interval(marks, 0, 5, intervals, thread)
            need(sum(x["wall_ns"] for x in parts.values()) == total["wall_ns"], "five inner stages do not partition inner call")
            for name, value in parts.items():
                stats[name].add(value)
                by_group[(rank, outer_item["scope"], outer_item["arm"], name)].add(value)
                if value["wall_ns"] > 2_000_000:
                    stalls.append(dict(**identity, stage=name, **value))
            stats["inner_call_total_overlaps_five_stages"].add(total)
            if total["wall_ns"] > 2_000_000:
                totals.append(dict(**identity, **total, inner_stages=parts))
            if total["same_thread_gc_overlap_ns"]:
                gc_hits.append(dict(**identity, **total, overlapping_stages={k: v for k, v in parts.items() if v["same_thread_gc_overlap_ns"]}))
            release["lease_alive"] += outer_item["old_wrapper_alive"][0]
            release["receive_wrapper_alive"] += outer_item["old_wrapper_alive"][1]
    stalls.sort(key=lambda x: x["wall_ns"], reverse=True)
    totals.sort(key=lambda x: x["wall_ns"], reverse=True)
    examples = [dict(**x, same_call=peers(outer, inner, x["row"]),
                     next_call=peers(outer, inner, x["row"] + 1) if x["sample"] < 59 else None) for x in totals[:top]]
    return dict(schema="lingbot_qk_collective_boundary_summary_v2", diagnostic_only=True, performance_timing_eligible=False,
                gate_passed=False, model_admitted=False, total_rank_samples=46080, logical_samples_per_rank=5760,
                actual_cases=192, layer_payloads=[0], all40_payloads=False, profiles=pins, supervisors=runs,
                correctness_receipts=reference["correctness_receipts"], diagnostic_correctness_receipts=reference["diagnostic_correctness_receipts"],
                diagnostic_schema=SCHEMA, diagnostic_method=METHOD, diagnostic_sources=reference["diagnostic_sources"],
                expected_collective_code=reference["expected_collective_code"], sources=reference["sources"],
                manifest=reference["manifest"], matrix_sha256=reference["matrix_sha256"], boot_id=reference["boot_id"],
                analyzer_source=scalar.read_pinned_source(__file__), scalar_helper_source=scalar.read_pinned_source(scalar.__file__),
                fingerprint_helper_source=scalar.read_pinned_source(fingerprint_api.__file__),
                stage_summary={k: v.export() for k, v in stats.items()},
                by_rank_scope_arm=[dict(rank=r, scope=s, arm=a, stage=n, **v.export()) for (r, s, a, n), v in by_group.items()],
                gc_observation_complete=all(x["observation_complete"] for x in gc_status), gc_by_rank=gc_status,
                samples_overlapping_same_thread_gc=gc_hits, old_python_wrapper_alive=release,
                inner_stage_stalls_gt2ms=stalls, inner_total_stalls_gt2ms=totals, examples=examples,
                caveats=["New wrapper and observer overhead change dispatch/GC timing; diagnostic data never admit the failed QK candidate.",
                         "Only five inner host stages are disjoint. Their total, enclosing outer call and outer sync overlap and cannot be added as savings.",
                         "Receive allocation measures initial empty_like only; output processing may itself allocate.",
                         "Public dist host entry/return is not device entry/completion, transport time, or an identified lock wait.",
                         "Same/next-call eight-rank timestamps show host ordering association, not a standalone causal device timeline.",
                         "GC is matched only on the same native thread and clipped to each interval. Missing/dropped pairs remain partial.",
                         "Wall/CPU/context observations are sequential, not atomic; signed wall-minus-CPU is not an idle estimate.",
                         "Scalar helpers are reused with explicit source pins; old broad schema, raw-marshal-v1 or old PASS is never relabelled as v2 correctness."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--profile-run", type=Path, required=True)
    parser.add_argument("--correctness-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    need(args.top > 0 and not args.output.exists(), "positive top and fresh output required")
    result = summarize(args.profiles, args.profile_run, args.correctness_run, args.top)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("total_rank_samples", "gc_observation_complete", "old_python_wrapper_alive", "stage_summary")}, indent=2))


if __name__ == "__main__":
    main()
