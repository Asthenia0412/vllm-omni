"""CPU-only scalar analysis of instrumented QK host observations; never a speed gate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

SCHEMA = "lingbot_qk_host_dependency_diagnostic_v1"
METHOD = "integer_boundaries_normal_gc_per_call_sync_v1"
MARKS = ("pre_sync_begin", "pre_sync_end", "producer_begin", "producer_end",
         "prepare_begin", "prepare_end", "exchange_begin", "collective_begin",
         "collective_end", "exchange_end", "call_return_before_replace",
         "old_held_replace_end", "post_sync_begin", "post_sync_end")
COLS = ("wall_ns", "thread_cpu_ns", "thread_voluntary_context_switches",
        "thread_involuntary_context_switches")
SCOPES = ("producer_plus_layer0_and_original_consumer", "cached_snapshot_and_original_consumer")
ARMS = ("baseline", "combined")
# These partition pre-sync begin through post-sync end. Totals below overlap them.
STAGES = (("pre_sync", 0, 1), ("pre_call_gap", 1, None),
          ("producer", 2, 3), ("producer_to_prepare", 3, 4),
          ("prepare", 4, 5), ("prepare_to_exchange", 5, 6),
          ("exchange_pre_collective", 6, 7), ("collective_host_call", 7, 8),
          ("exchange_post_collective", 8, 9), ("call_return_gap", 9, 10),
          ("old_held_release", 10, 11), ("replace_to_post_sync", 11, 12),
          ("post_sync", 12, 13))
TOTALS = (("host_call_including_old_release", 1, 11), ("sample", 0, 13))


def need(value, message):
    if not value:
        raise ValueError(message)


def integer(value):
    return type(value) is int


def read_pinned(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def gc_intervals(record):
    """Pair only matching thread/generation; orphan or missing observations stay partial."""
    need(record["columns"] == ["wall_ns", "native_thread_id", "phase_start0_stop1", "generation"], "GC columns differ")
    data = record["data"]
    need(len(data) == record["count"] * 4 and all(integer(v) for v in data), "GC buffer malformed")
    need(all(integer(record[k]) and record[k] >= 0 for k in ("count", "capacity", "dropped", "errors")), "GC counters invalid")
    need(record["count"] <= record["capacity"], "GC buffer overflow unreported")
    need(record["complete"] is (record["dropped"] == 0 and record["errors"] == 0), "GC complete flag differs")
    active, paired, orphan = {}, [], []
    previous = -1
    for index in range(0, len(data), 4):
        stamp, thread, phase, generation = data[index:index + 4]
        need(stamp >= previous and thread > 0, "GC timestamp/thread invalid")
        previous = stamp
        key = (thread, generation)
        if phase == 0:
            if key in active:
                orphan.append(dict(kind="duplicate_start", thread=thread, generation=generation, start_ns=active[key]))
            active[key] = stamp
        elif phase == 1 and key in active:
            paired.append(dict(thread=thread, generation=generation, start_ns=active.pop(key), end_ns=stamp))
        else:
            orphan.append(dict(kind="unmatched_stop_or_invalid_phase", thread=thread, generation=generation, wall_ns=stamp, phase=phase))
    orphan.extend(dict(kind="unmatched_start", thread=thread, generation=generation, start_ns=stamp)
                  for (thread, generation), stamp in active.items())
    return paired, dict(event_count=record["count"], paired_count=len(paired), dropped=record["dropped"],
                        errors=record["errors"], unmatched=orphan,
                        observation_complete=record["complete"] and not orphan)


def overlap(start, end, intervals, thread):
    clipped = [(max(start, x["start_ns"]), min(end, x["end_ns"])) for x in intervals
               if x["thread"] == thread and x["start_ns"] < end and x["end_ns"] > start]
    clipped.sort()
    total, previous_end = 0, start
    for lo, hi in clipped:
        total += max(0, hi - max(lo, previous_end))
        previous_end = max(previous_end, hi)
    return total


def interval(row, begin, end, gc, thread):
    first, last = row[begin], row[end]
    delta = [b - a for a, b in zip(first, last)]
    need(all(v >= 0 for v in delta), "non-monotonic host observation")
    return dict(start_ns=first[0], end_ns=last[0], wall_ns=delta[0], thread_cpu_ns=delta[1],
                wall_minus_thread_cpu_ns=delta[0] - delta[1], voluntary_switches=delta[2],
                involuntary_switches=delta[3], same_thread_gc_overlap_ns=overlap(first[0], last[0], gc, thread))


def stages_for(row, producer, gc, thread):
    result = {}
    for name, begin, end in STAGES:
        if not producer and name in ("producer", "producer_to_prepare"):
            continue
        if name == "pre_call_gap":
            end = 2 if producer else 4
        result[name] = interval(row, begin, end, gc, thread)
    return result


def validate_report(report, rank):
    need(report.get("schema") == SCHEMA and report.get("method") == METHOD, "schema/method differs")
    need(report.get("mode") == "profile" and report.get("rank") == rank and report.get("passed") is True, "incomplete/wrong rank profile")
    for key, wanted in (("diagnostic_only", True), ("performance_timing_eligible", False),
                        ("gate_passed", False), ("model_admitted", False), ("original_hccl_included", True)):
        need(report.get(key) is wanted, "invalid diagnostic claim: " + key)
    trace = report["trace"]
    need(tuple(trace["marks"]) == MARKS and tuple(trace["columns"]) == COLS, "trace layout differs")
    need(trace["row_count"] == 5760 and len(trace["data"]) == 5760 * 56, "trace extent differs")
    need(all(integer(v) for v in trace["data"]), "trace values must be integers")
    need(trace["old_wrapper_columns"] == ["lease", "receive_tensor"] and len(trace["old_wrapper_alive_after_replace"]) == 5760 * 2,
         "release observations incomplete")
    need(all(integer(v) and v in (0, 1) for v in trace["old_wrapper_alive_after_replace"]), "release flags invalid")
    need(integer(trace["main_native_thread_id"]) and trace["main_native_thread_id"] > 0, "thread ID invalid")
    expected_cases = [(kv, probe) for kv in (4680, 9360, 14040, 18720, 23400, 28080) for probe in range(4)]
    need(len(report["cases"]) == 24, "case count differs")
    rows, next_row, previous_end = [], 0, -1
    for case, (kv, probe) in zip(report["cases"], expected_cases):
        identity = case["identity"]
        need(identity["rank"] == rank and identity["kv_len"] == kv and identity["probe"] == probe and identity["layer"] == 0,
             "case identity/order differs")
        need(case["passed"] is True and case["full_owner_sha256_verified"] is True, "case failed")
        need(len(case["phases"]) == 4, "phase count differs")
        for phase, (scope, arm) in zip(case["phases"], [(s, a) for s in SCOPES for a in ARMS]):
            need(phase["scope"] == scope and phase["arm"] == arm and phase["row_count"] == 60
                 and phase["warmup"] == 20 and phase["first_row"] == next_row, "phase order/extent differs")
            bounds = phase["phase_boundaries"]
            need(len(bounds) == 2 and [b["name"] for b in bounds] == ["warm20", "diagnostic_calls"], "phase boundaries differ")
            need(all(integer(b[k]) for b in bounds for k in ("enter_ns", "ready_ns")), "noninteger phase time")
            need(bounds[0]["enter_ns"] <= bounds[0]["ready_ns"] <= bounds[1]["enter_ns"] <= bounds[1]["ready_ns"], "phase times unordered")
            producer = scope == SCOPES[0]
            for sample in range(60):
                number = next_row + sample
                flat = trace["data"][number * 56:(number + 1) * 56]
                row = [flat[i:i + 4] for i in range(0, 56, 4)]
                valid = list(range(14)) if producer else [0, 1] + list(range(4, 14))
                need(all(v >= 0 for index in valid for v in row[index]), "missing mark")
                need(producer or row[2] == row[3] == [-1] * 4, "cached producer must be absent")
                for before, after in zip(valid, valid[1:]):
                    need(all(a <= b for a, b in zip(row[before], row[after])), "marks not monotonic")
                need(row[0][0] >= max(previous_end, bounds[1]["ready_ns"]), "overlapping sample/phase")
                previous_end = row[13][0]
                rows.append(dict(rank=rank, row=number, kv_len=kv, probe=probe, scope=scope, arm=arm,
                                 sample=sample, producer=producer, marks=row,
                                 old_wrapper_alive=trace["old_wrapper_alive_after_replace"][number * 2:number * 2 + 2]))
            next_row += 60
    need(next_row == 5760, "row coverage differs")
    return rows


class Stats:
    def __init__(self):
        self.n = self.total = self.cpu = self.voluntary = self.involuntary = self.gc = self.gc_samples = 0
        self.max = self.gt2 = self.gt5 = self.negative_residual = 0

    def add(self, value):
        self.n += 1
        self.total += value["wall_ns"]
        self.cpu += value["thread_cpu_ns"]
        self.voluntary += value["voluntary_switches"]
        self.involuntary += value["involuntary_switches"]
        self.gc += value["same_thread_gc_overlap_ns"]
        self.gc_samples += value["same_thread_gc_overlap_ns"] > 0
        self.max = max(self.max, value["wall_ns"])
        self.gt2 += value["wall_ns"] > 2_000_000
        self.gt5 += value["wall_ns"] > 5_000_000
        self.negative_residual += value["wall_minus_thread_cpu_ns"] < 0

    def export(self):
        return dict(count=self.n, mean_wall_ms=self.total / self.n / 1e6, mean_thread_cpu_ms=self.cpu / self.n / 1e6,
                    max_wall_ms=self.max / 1e6, gt2ms=self.gt2, gt5ms=self.gt5, voluntary_switches=self.voluntary,
                    involuntary_switches=self.involuntary, same_thread_gc_overlap_ns=self.gc,
                    samples_with_same_thread_gc=self.gc_samples, negative_wall_minus_cpu_samples=self.negative_residual)


def peer_view(rows, index):
    """Same ordinal collective, with no claim that host timestamps are device entry."""
    peers = []
    for rank in range(8):
        item = rows[rank][index]
        m = item["marks"]
        peers.append(dict(rank=rank, timestamps=dict(zip(MARKS, [x[0] for x in m])),
                          collective_wall_ns=m[8][0] - m[7][0], collective_thread_cpu_ns=m[8][1] - m[7][1],
                          collective_voluntary_switches=m[8][2] - m[7][2],
                          collective_involuntary_switches=m[8][3] - m[7][3],
                          post_sync_wall_ns=m[13][0] - m[12][0],
                          previous_sample_post_end_ns=rows[rank][index - 1]["marks"][13][0] if item["sample"] else None))
    starts = [p["timestamps"]["collective_begin"] for p in peers]
    ends = [p["timestamps"]["collective_end"] for p in peers]
    return dict(collective_host_start_skew_ns=max(starts) - min(starts),
                latest_host_collective_start_rank=starts.index(max(starts)),
                collective_host_return_skew_ns=max(ends) - min(ends),
                latest_host_collective_return_rank=ends.index(max(ends)), peers=peers)


def bind_run(run_dir, profiles, reference):
    launch, lp = read_pinned(Path(run_dir) / "launch.json")
    result, rp = read_pinned(Path(run_dir) / "result.json")
    for key in ("nonce", "leader", "command", "mode", "manifest"):
        need(launch[key] == result[key], "launch/result identity differs")
    need(result["exit_code"] == 0 and result["fatal"] is None and launch["fatal"] is None
         and result["remaining_owned_processes"] == [], "supervisor not cleanly retired")
    need(result["mode"] == "profile", "supervisor mode differs")
    command = result["command"]
    prefix = ["torchrun", "--standalone", "--nnodes=1", "--nproc-per-node=8", "recovery/diagnose_qk_host_dependencies.py", "--execute"]
    need(command[:6] == prefix and (len(command) - 6) % 2 == 0, "launcher differs")
    options = dict(zip(command[6::2], command[7::2]))
    need(len(options) == (len(command) - 6) // 2 and set(options) == {
        "--mode", "--fixtures", "--manifest", "--correctness-dir", "--diagnostic-correctness-dir", "--output"}, "launcher flags differ")
    need(options["--mode"] == "profile" and Path(options["--output"]).resolve() == Path(profiles).resolve()
         and options["--manifest"] == reference["manifest"]["path"], "launcher/report binding differs")
    manifest, pin = read_pinned(options["--manifest"])
    need(pin == reference["manifest"] and result["manifest"] == manifest["provider_manifest"]["path"], "provider binding differs")
    for field, option in (("correctness_receipts", "--correctness-dir"), ("diagnostic_correctness_receipts", "--diagnostic-correctness-dir")):
        need(len(reference[field]) == 8, "correctness evidence count differs")
        for rank, wanted in enumerate(reference[field]):
            _, actual = read_pinned(Path(options[option]) / f"rank{rank}.json")
            need(actual == wanted, "correctness pin differs")
    return dict(launch=lp, result=rp, nonce=result["nonce"], options=options)


def summarize(profiles, run_dir, top=20):
    reports, pins, rows, gc_by_rank, gc_status = [], [], [], [], []
    for rank in range(8):
        report, pin = read_pinned(Path(profiles) / f"rank{rank}.json")
        rows.append(validate_report(report, rank))
        gc, status = gc_intervals(report["gc"])
        gc_by_rank.append(gc)
        gc_status.append(dict(rank=rank, **status))
        reports.append(report)
        pins.append(pin)
    reference = reports[0]
    for report in reports[1:]:
        for name in ("schema", "method", "boot_id", "manifest", "matrix_sha256", "sources", "diagnostic_source",
                     "correctness_receipts", "diagnostic_correctness_receipts"):
            need(report[name] == reference[name], "rank provenance differs: " + name)
        for first, other in zip(reference["cases"], report["cases"]):
            need({k: v for k, v in first["identity"].items() if k != "rank"}
                 == {k: v for k, v in other["identity"].items() if k != "rank"}, "cross-rank case mismatch")
    run = bind_run(run_dir, profiles, reference)
    stats, per_arm, per_rank = defaultdict(Stats), defaultdict(Stats), defaultdict(Stats)
    stage_stalls, host_stalls, gc_hits = [], [], []
    release = dict(lease_alive_samples=0, receive_alive_samples=0)
    for rank, rank_rows in enumerate(rows):
        thread = reports[rank]["trace"]["main_native_thread_id"]
        for item in rank_rows:
            parts = stages_for(item["marks"], item["producer"], gc_by_rank[rank], thread)
            totals = {name: interval(item["marks"], begin, end, gc_by_rank[rank], thread) for name, begin, end in TOTALS}
            need(sum(v["wall_ns"] for v in parts.values()) == totals["sample"]["wall_ns"], "stage partition lost time")
            identity = {key: item[key] for key in ("rank", "row", "kv_len", "probe", "scope", "arm", "sample")}
            for name, value in {**parts, **totals}.items():
                stats[name].add(value)
                per_arm[(item["scope"], item["arm"], name)].add(value)
                per_rank[(rank, item["arm"], name)].add(value)
                if name in parts and value["wall_ns"] > 2_000_000:
                    stage_stalls.append(dict(**identity, stage=name, **value))
            if totals["host_call_including_old_release"]["wall_ns"] > 2_000_000:
                host_stalls.append(dict(**identity, **totals["host_call_including_old_release"], stages=parts))
            if totals["sample"]["same_thread_gc_overlap_ns"]:
                gc_hits.append(dict(**identity, **totals["sample"], stages={name: value for name, value in parts.items()
                                        if value["same_thread_gc_overlap_ns"]}))
            release["lease_alive_samples"] += item["old_wrapper_alive"][0]
            release["receive_alive_samples"] += item["old_wrapper_alive"][1]
    host_stalls.sort(key=lambda item: item["wall_ns"], reverse=True)
    stage_stalls.sort(key=lambda item: item["wall_ns"], reverse=True)
    examples = []
    # Preserve complete per-marker timestamps for all ranks for the largest host stalls.
    for item in host_stalls[:top]:
        examples.append(dict(**item, same_ordinal_collective=peer_view(rows, item["row"]),
                             next_ordinal_collective=peer_view(rows, item["row"] + 1) if item["sample"] < 59 else None))
    return dict(schema="lingbot_qk_host_dependency_summary_v1", diagnostic_only=True, performance_timing_eligible=False,
                gate_passed=False, model_admitted=False, total_rank_samples=8 * 5760, profile_pins=pins,
                analyzer_source=read_pinned_source(__file__), run=run, manifest=reference["manifest"],
                sources=reference["sources"], diagnostic_source=reference["diagnostic_source"], matrix_sha256=reference["matrix_sha256"],
                boot_id=reference["boot_id"], gc_observation_complete=all(x["observation_complete"] for x in gc_status),
                gc_by_rank=gc_status, old_python_wrapper_release=release,
                stage_summary={k: v.export() for k, v in stats.items()},
                by_scope_arm=[dict(scope=s, arm=a, stage=n, **v.export()) for (s, a, n), v in per_arm.items()],
                by_rank_arm=[dict(rank=r, arm=a, stage=n, **v.export()) for (r, a, n), v in per_rank.items()],
                stage_stalls_gt2ms=stage_stalls, host_call_stalls_gt2ms=host_stalls,
                examples_with_eight_rank_timestamps=examples, samples_overlapping_same_thread_gc=gc_hits,
                caveats=["Instrumentation changes work and GC scheduling; these durations never admit a performance candidate.",
                         "Disjoint stages partition samples; host_call and sample totals overlap stages and must not be added again.",
                         "Thread CPU and context counters are sequential observations; wall-minus-thread-CPU is signed, not a CPU-idle estimate.",
                         "GC matching uses only same native thread and clips overlap to each sample/stage; correlation is not causation.",
                         "Missing/dropped GC observations imply partial coverage, not evidence that no collection occurred.",
                         "collective_host_call spans original all_to_all_5D, including metadata admission, output allocation, distributed call and output layout processing.",
                         "Same-ordinal eight-rank host call skew is not device entry, HCCL transfer duration, or proof of a causal wait.",
                         "Weakref death is Python wrapper release, not device storage reclamation.",
                         "Layer0 actual fixtures and isolated WORLD reproduce call structure, not whole-model or chunk performance."])


def read_pinned_source(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    need(args.top > 0, "top must be positive")
    result = summarize(args.profiles, args.run_dir, args.top)
    need(not args.output.exists(), "refuse to overwrite summary")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("total_rank_samples", "gc_observation_complete", "old_python_wrapper_release", "stage_summary")}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
