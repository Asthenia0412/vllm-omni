"""Independent private-retention replay of 24 actual layer0 QK cases per rank.

This is an isolated operator/consumer experiment, not model admission. Import is
CPU-only. No missing payload is synthesized; all40-layer capture descriptors
are not40-layer payloads. Run only through the launch-owned supervisor.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

from load_real_fixtures import contract, load_case, validate_case_metadata

SCHEMA = "lingbot_qk_private_retention_matrix_replay_v3"
METHOD = "phase_barrier_then_sync_per_call_sync_v2"
EXPECTED = tuple((length, probe) for length in contract.LENGTHS for probe in range(4))
SHAPE = (8, 585, 3, 1, 5, 128)
STRIDE = (1123200, 1920, 640, 640, 128, 1)
OWNER_ELEMENTS = 8986112
CHECKS = (
    "expanded_coefficients", "captured_reference_hccl", "baseline_send_and_receive",
    "sentinel_send", "full_arena_after_pack", "sentinel_receive",
    "full_arena_after_hccl", "fresh_owner_extent_offset_and_distinctness",
    "fresh_sends_and_receives", "first_owner_not_overwritten",
    "full_source_and_reference_owners", "source_descriptors_unchanged",
    "missing_snapshot_refuses_without_producer_mutation",
    "production_run_auto_reclaim", "burst_four_fresh_outputs_and_original_hccl",
    "released_sources_and_private_churn",
)
_QUARANTINE = []


def need(value, message):
    # These checks are deliberately retained under python -O.
    if not value:
        raise RuntimeError("QK repeatable matrix: " + message)


def pin(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def case_key(case):
    return case["kv_len"], case["probe"]


def identity(case):
    keys = ("rank", "request_id", "chunk", "kv_len", "probe", "layer", "coefficient_generation")
    return {key: case[key] for key in keys}


def validate_order(cases):
    need(tuple(case_key(case) for case in cases) == EXPECTED, "24 real cases must have canonical history/probe order")
    need(all(case["layer"] == 0 for case in cases), "only captured layer0 payload is supported")


def preflight(root):
    """Bind all192 actual QK metadata records, before importing torch/allocating.

    Full owner SHA verification happens again in load_case on the owning rank.
    This function does not claim byte verification or candidate correctness.
    """
    root = Path(root).resolve(strict=True)
    ranks, locations, capture_builds = [], [], set()
    for rank in range(8):
        directory = root / f"rank{rank}"
        coverage_path = directory / "coverage.json"
        coverage = json.loads(coverage_path.read_text())
        need(coverage["schema"] == "lingbot_real_qk_cross_capture_v1"
             and coverage["rank"] == rank and coverage["complete"] is True
             and coverage["failed"] is False and coverage["timing_eligible"] is False,
             "complete, successful actual capture required")
        # Bind the24 known layer0 names directly. Re-reading all40-layer JSONs
        # from all8 workers adds NFS startup traffic without adding QK payload
        # evidence; check_real_fixtures.py owns the broader descriptor audit.
        names = [f"qk_kv{length}_p{probe}_l00.json" for length, probe in EXPECTED]
        selected = [name for name in coverage["cases"] if name.startswith("qk_") and name.endswith("_l00.json")]
        need(len(selected) == 24 and set(selected) == set(names), "missing/duplicate actual24 QK payload names")
        paths = [(directory / name).resolve(strict=True) for name in names]
        need(all(path.is_relative_to(directory.resolve()) for path in paths), "case path escapes rank directory")
        cases = [json.loads(path.read_text()) for path in paths]
        validate_order(cases)
        pins = []
        rank_locations = []
        for path, case in zip(paths, cases):
            validate_case_metadata(case)
            need(case["rank"] == rank and case["kind"] == "qk", "rank/kind mismatch")
            need(case["actual_backend"] == "lingbot_recovery_v1::qk_send"
                 and case["epsilon"] == 1e-6 and case["head_num"] == 40 and case["ring_size"] == 1,
                 "captured arithmetic/backend differs")
            need(type(case["coefficient_generation"]) is int and case["coefficient_generation"] > 0,
                 "missing actual captured coefficient generation")
            need(case["dispatch_receipt"]["build_manifest_sha256"] == coverage["build_manifest"]["sha256"],
                 "case/capture build differs")
            capture_builds.add(coverage["build_manifest"]["sha256"])
            rank_locations.append((case["kv_len"], case["probe"], case["request_id"], case["chunk"]))
            pins.append(dict(file=pin(path), identity=identity(case)))
        locations.append(rank_locations)
        ranks.append(dict(rank=rank, coverage=pin(coverage_path), cases=pins))
    need(len(capture_builds) == 1 and all(items == locations[0] for items in locations),
         "rank capture builds/actual selected occurrences differ")
    result = dict(root=str(root), ranks=ranks, captured_provider_sha256=next(iter(capture_builds)),
                  actual_qk_payload_cases=192, real_payload_layers=[0], all40_payloads=False)
    result["sha256"] = digest(result)
    return result


def source_pins():
    base = Path(__file__).resolve().parents[1]
    return {name: pin(base / relative) for name, relative in (
        ("harness", "recovery/check_qk_private_retention_matrix.py"),
        ("loader", "recovery/load_real_fixtures.py"),
        ("capture_contract", "vllm_omni/diffusion/layers/lingbot_real_fixture_capture.py"),
        ("original_collective", "vllm_omni/diffusion/distributed/comm.py"),
        ("coefficient_producer", "vllm_omni/diffusion/layers/lingbot_native_bf16_qk.py"),
        ("baseline_wrapper", "vllm_omni/diffusion/layers/lingbot_recovery.py"),
        ("pool_wrapper", "vllm_omni/diffusion/layers/lingbot_qk_private_retention.py"),
        ("original_pool_wrapper", "vllm_omni/diffusion/layers/lingbot_qk_repeatable_pool.py"),
        ("private_retention_contract", "recovery/qk_private_retention_contract.py"),
        ("padded_consumer_hook", "vllm_omni/diffusion/layers/qk_single_host_integration_v1.py"),
    )}


def verify_correctness(directory, matrix, manifest_pin, sources):
    """Timing admission requires all8 matching actual24 correctness receipts."""
    receipts = []
    for rank in range(8):
        path = Path(directory) / f"rank{rank}.json"
        report = json.loads(path.read_text())
        need(report.get("schema") == SCHEMA and report.get("mode") == "correctness"
             and report.get("passed") is True and report.get("rank") == rank,
             "successful same-rank correctness receipt required")
        need(report.get("matrix_sha256") == matrix["sha256"] and report.get("manifest") == manifest_pin
             and report.get("sources") == sources, "correctness fixture/build/source pins differ")
        need(report.get("original_hccl_included") is True and report.get("all40_payloads") is False
             and report.get("real_payload_layers") == [0], "correctness scope differs")
        cases = report.get("cases", [])
        validate_closed(report.get("pool_after_close", []))
        validate_order([entry["identity"] for entry in cases])
        wanted = matrix["ranks"][rank]["cases"]
        for entry, expected in zip(cases, wanted):
            need(entry.get("identity") == expected["identity"] and entry.get("fixture") == expected["file"]
                 and entry.get("passed") is True and entry.get("checks") == list(CHECKS)
                 and entry.get("full_owner_sha256_verified") is True,
                 "incomplete real-case correctness evidence")
            validate_lifecycle(entry.get("lifecycle", {}))
        receipts.append(pin(path))
    return receipts


def phase(dist, torch, phase_name):
    # Boundary only. No per-sample rank barrier: preserve arrival skew evidence.
    enter = time.perf_counter_ns()
    dist.barrier()
    torch.npu.synchronize()
    return dict(name=phase_name, enter_ns=enter, ready_ns=time.perf_counter_ns())


def equal(torch, actual, expected, name):
    torch.npu.synchronize()
    left = actual.detach().cpu().contiguous()
    right = expected.detach().cpu().contiguous()
    need(left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape)
         and torch.equal(left.view(torch.uint8), right.view(torch.uint8)), name + " is not bitwise equal")


def check_owners(torch, loaded, provider):
    for section in ("inputs", "outputs"):
        for oid, owner in loaded["owners"][section].items():
            equal(torch, owner, loaded["host_owners"][section][oid], f"{section} full owner {oid}")
        for name, view in loaded[section].items():
            actual = dict(zip(contract.FIELDS, provider.base.describe(view)))
            expected = loaded["metadata"][section]["descriptors"][name]
            for field in contract.FIELDS[:9]:
                value = actual[field]
                if isinstance(value, (tuple, list)):
                    value = list(value)
                need(value == expected[field], f"{section}/{name}/{field} changed")


class Replay:
    def __init__(self, torch, backend, candidate, provider, loaded, group, collective):
        from vllm_omni.diffusion.layers.lingbot_native_bf16_qk import SharedCoefficients
        from vllm_omni.diffusion.layers.lingbot_recovery import prepare_send
        self.torch, self.backend, self.candidate, self.provider = torch, backend, candidate, provider
        self.loaded, self.group, self.collective, self.baseline_prepare = loaded, group, collective, prepare_send
        self.SharedCoefficients = SharedCoefficients
        inputs = loaded["inputs"]
        self.raw = inputs["qkv"]
        self.original = (inputs["cosine"], inputs["sine"])
        self.shared = SharedCoefficients()
        self.shared.prepare(*self.original)
        self.values = {candidate.CONFIG_KEY: True, "lingbot_npu_native_bf16_qk": True}
        self.module = SimpleNamespace(ulysses_world_size=8, num_local_heads=40, head_dim=128,
            norm_q=SimpleNamespace(eps=1e-6, weight=inputs["query_weight"]),
            norm_k=SimpleNamespace(eps=1e-6, weight=inputs["key_weight"]),
            _lingbot_native_bf16_qk_coefficients=self.shared)
        self.config = SimpleNamespace(model_config=self.values,
            model_class_name="LingBotWorldCausalDMDPipeline", enforce_eager=True,
            parallel_config=SimpleNamespace(tensor_parallel_size=1, sequence_parallel_size=8,
                                            ulysses_degree=8, ring_degree=1))
        self.route = candidate.configure_operator_module(self.module, self.config)
        self.reference = loaded["outputs"]["native_send"]
        self.metadata = self.raw.view(1, 585, 3, 40, 128)
        self.live = []
        self.lifecycle = {}
        for name, value in (("expanded_cosine", self.shared.expanded_cosine),
                            ("expanded_sine", self.shared.expanded_sine)):
            equal(torch, value, inputs[name], name)

    def prepare(self, arm):
        lease = (self.baseline_prepare(self.module, self.raw, self.original, 1)
                 if arm == "baseline" else self.route.prepare(self.raw, self.original, 1))
        need(lease is not None, arm + " refused actual captured fixture")
        return lease

    def call(self, arm, producer):
        if producer:
            self.shared.prepare(*self.original)
        lease = self.prepare(arm)
        output = lease.exchange(self.collective, self.group)
        return lease, output  # Keep output and every source/target owner live until sync.

    def correctness(self):
        torch, candidate = self.torch, self.candidate
        need(tuple(self.reference.shape) == SHAPE and self.reference.storage_offset() == 0,
             "actual native_send must satisfy the original offset0 ABI")
        expected = self.collective(self.metadata, scatter_idx=3, gather_idx=1,
                                   group=self.group, _prepared_send=self.reference)
        self.live.append(expected)
        baseline = self.prepare("baseline")
        self.live.append(baseline)
        equal(torch, baseline.send, self.reference, "baseline vs captured native send")
        baseline_result = baseline.exchange(self.collective, self.group)
        self.live.append(baseline_result)
        equal(torch, baseline_result, expected, "baseline receive")

        first = self.prepare("pool")
        self.live.append(first)
        args = first.args
        arena = torch.full((OWNER_ELEMENTS,), -7, dtype=torch.bfloat16, device=self.raw.device)
        target = arena.clone()
        target[256:-256].copy_(self.reference.flatten())
        self.live.extend((arena, target))
        send, owner = self.provider.ops.run_with_test_arena(self.provider.ensure_pool(args), *args, arena)
        self.live.extend((send, owner))
        need(owner.data_ptr() == arena.data_ptr(), "test arena owner was replaced")
        equal(torch, send, self.reference, "sentinel send")
        equal(torch, arena, target, "full arena including canaries after pack")
        sentinel = candidate.PoolLease(self.provider, self.route, self.metadata, send, owner,
                                   args, (self, self.loaded, arena, target, *args))
        self.live.append(sentinel)
        sentinel_result = sentinel.exchange(self.collective, self.group)
        self.live.append(sentinel_result)
        equal(torch, sentinel_result, expected, "sentinel receive")
        equal(torch, arena, target, "full arena including canaries after HCCL")

        second = self.prepare("pool")
        self.live.append(second)
        need(len({first.owner.data_ptr(), second.owner.data_ptr(), arena.data_ptr()}) == 3,
             "live calls reused a target owner")
        for label, lease in (("first", first), ("second", second)):
            need(tuple(lease.owner.shape) == (OWNER_ELEMENTS,) and lease.owner.storage_offset() == 0
                 and lease.owner.untyped_storage().nbytes() == OWNER_ELEMENTS * 2
                 and tuple(lease.send.shape) == SHAPE and tuple(lease.send.stride()) == STRIDE
                 and lease.send.storage_offset() == 256
                 and lease.send.untyped_storage().data_ptr() == lease.owner.data_ptr(),
                 label + " fresh padded owner extent/offset differs")
            equal(torch, lease.send, self.reference, label + " fresh send")
            result = lease.exchange(self.collective, self.group)
            self.live.append(result)
            equal(torch, result, expected, label + " fresh receive")
        equal(torch, first.send, self.reference, "first live owner after second consumer")
        equal(torch, arena, target, "sentinel arena after subsequent calls")
        # A third production call must reclaim completed retired Calls through
        # run(), without invoking the explicit support-test reclaim helper.
        handle = self.provider.ensure_pool(args)
        before_third = pool_stats(self.provider.ops, handle)
        third = self.prepare("pool")
        self.live.append(third)
        third_result = third.exchange(self.collective, self.group)
        self.live.append(third_result)
        equal(torch, third_result, expected, "third production receive")
        stats = pool_stats(self.provider.ops, handle)
        need(stats["producer_reclaimed_calls"] > before_third["producer_reclaimed_calls"]
             and stats["producer_reclaim_tid"] == stats["producer_tid"],
             "production run did not exercise producer automatic reclaim")
        self.lifecycle["automatic_reclaim"] = dict(before=before_third, after=stats)
        # Actual unchanged HCCL calls, no inserted per-call device drain. A
        # successful burst is evidence only for its observed queue depth.
        burst = [self.call("pool", False) for _ in range(4)]
        self.live.extend(burst)
        torch.npu.synchronize()
        self.lifecycle["burst"] = dict(before=stats, after=pool_stats(self.provider.ops, handle),
            calls=4, per_call_device_drain=False, inflight_coverage_claim=False)
        owners = [first.owner, second.owner, third.owner, arena, *[lease.owner for lease, _ in burst]]
        need(len({owner.data_ptr() for owner in owners}) == len(owners),
             "burst reused a live output allocation")
        for lease, result in burst:
            equal(torch, lease.send, self.reference, "burst fresh send")
            equal(torch, result, expected, "burst original receive")
        equal(torch, first.send, self.reference, "first owner retained across burst")
        equal(torch, arena, target, "sentinel canaries retained across burst")
        check_owners(torch, self.loaded, self.provider)
        missing = self.SharedCoefficients()
        self.module._lingbot_native_bf16_qk_coefficients = missing
        before_miss = pool_stats(self.provider.ops, handle)
        need(self.route.prepare(self.raw, self.original, 1) is None and missing.generation == 0
             and missing.expanded_cosine is None and missing.expanded_sine is None,
             "missing producer snapshot mutated state or submitted")
        after_miss = pool_stats(self.provider.ops, handle)
        need(before_miss["submitted"] == after_miss["submitted"],
             "missing snapshot submitted native work")
        self.lifecycle["missing_snapshot"] = dict(before=before_miss, after=after_miss)
        self.module._lingbot_native_bf16_qk_coefficients = self.shared
        self.retained_send = first.send
        self.retained_arena_pair = (arena, target)
        self.lifecycle["private_churn"] = self.private_churn(expected, handle)
        return list(CHECKS)

    def _fresh_sources_call(self):
        """Return only outputs; fresh input/route/lease Python locals expire here.

        Copies are device-to-device on the admitted stream. Preserve full base
        owners rather than cloning views into a different physical signature.
        Native Call retention and original HCCL protection must carry the work
        after this function returns. This is untimed correctness stress only.
        """
        inputs, owners = {}, {}
        snapshot = self.loaded["metadata"]["inputs"]
        for name in ("qkv", "query_weight", "key_weight"):
            desc = snapshot["descriptors"][name]
            oid = snapshot["view_owner_ids"][name]
            if oid not in owners:
                source = self.loaded["owners"]["inputs"][oid]
                owner = self.backend.empty_with_format(
                    desc["base_sizes"], dtype=source.dtype, device=source.device,
                    acl_format=desc["npu_format"])
                owner.copy_(source)
                owners[oid] = owner
            view = owners[oid].as_strided(desc["shape"], desc["stride"], desc["storage_offset"])
            actual = dict(zip(contract.FIELDS, self.provider.base.describe(view)))
            for field in contract.FIELDS[:9]:
                value = actual[field]
                if isinstance(value, (tuple, list)):
                    value = list(value)
                need(value == desc[field], "fresh source descriptor differs: " + name + "/" + field)
            need(view.untyped_storage().data_ptr() != self.loaded["inputs"][name].untyped_storage().data_ptr(),
                 "fresh stress input aliases captured source owner")
            inputs[name] = view
        shared = self.SharedCoefficients()
        shared.prepare(*self.original)
        module = SimpleNamespace(
            ulysses_world_size=8, num_local_heads=40, head_dim=128,
            norm_q=SimpleNamespace(eps=1e-6, weight=inputs["query_weight"]),
            norm_k=SimpleNamespace(eps=1e-6, weight=inputs["key_weight"]),
            _lingbot_native_bf16_qk_coefficients=shared)
        route = self.candidate.configure_operator_module(module, self.config)
        lease = route.prepare(inputs["qkv"], self.original, 1)
        need(lease is not None, "fresh-source stress route refused")
        receive = lease.exchange(self.collective, self.group)
        return lease.send, receive

    def private_churn(self, expected, handle):
        """Bounded original-HCCL burst, releasing Python sources before drain."""
        torch = self.torch
        before = pool_stats(self.provider.ops, handle)
        retained = []
        pressure_owners = []
        pressure_stream = torch.npu.Stream(device=self.raw.device)
        need(pressure_stream != torch.npu.default_stream(self.raw.device), "distinct pressure stream required")
        churn_operations = 0
        churn_shapes = (
            ((1, 585, 5120), torch.bfloat16), ((1, 585, 5120), torch.bfloat16),
            ((1, 585, 1), torch.float32), ((1, 585, 1), torch.float32),
            ((1, 585, 40, 128), torch.bfloat16), ((1, 585, 40, 128), torch.bfloat16))
        for _ in range(12):
            # No lease, route, module or fresh source owner escapes this helper.
            retained.append(self._fresh_sources_call())
            # Allocate and fill on the same independent pressure stream. A
            # default-stream caching-allocator block can still have valid prior
            # uses; moving its new writer to another stream without a dependency
            # would create a test-induced race. Keep pressure owners until drain.
            with torch.npu.stream(pressure_stream):
                churn = [self.backend.empty_with_format(shape, dtype=dtype,
                         device=self.raw.device, acl_format=2) for shape, dtype in churn_shapes]
                for value in churn:
                    value.record_stream(pressure_stream)
                    value.fill_(-19)
                    churn_operations += 1
            pressure_owners.extend(churn)
            del value, churn
        torch.npu.synchronize()  # One drain after the entire burst, outside timing.
        after = pool_stats(self.provider.ops, handle)
        pointers = [send.untyped_storage().data_ptr() for send, _ in retained]
        need(len(set(pointers)) == len(retained), "stress reused retained send owner")
        for send, receive in retained:
            equal(torch, send, self.reference, "released-source stress send")
            equal(torch, receive, expected, "released-source stress original receive")
        equal(torch, self.retained_send, self.reference, "earlier send across allocator churn")
        equal(torch, *self.retained_arena_pair, "earlier full arena/canaries across allocator churn")
        check_owners(torch, self.loaded, self.provider)
        self.live.extend(retained)
        record = dict(before=before, after=after, calls=12, churn_allocations=churn_operations,
            per_call_device_drain=False, fresh_native_source_owners=True,
            python_source_locals_released=True, python_lease_locals_released=True,
            retained_send_receive_pairs=12, all_retained_outputs_bitwise=True,
            prior_arena_canaries_bitwise=True, original_sources_unchanged=True,
            original_hccl=True, fresh_source_contents_after_release_observed=False,
            native_private_reuse_addresses_observed=False, inflight_coverage_claim=False,
            pressure_allocated_on_pressure_stream=True, pressure_stream_distinct=True,
            pressure_submitted_on_separate_stream=True, device_overlap_observed=False)
        validate_private_churn(record)
        return record

    def timing(self, dist, round_number):
        torch = self.torch
        records = {}
        order = ("pool", "baseline") if round_number == 2 else ("baseline", "pool")
        for scope, producer in (("producer_plus_layer0_and_original_consumer", True),
                                ("cached_snapshot_and_original_consumer", False)):
            records[scope] = {}
            for arm in order:
                # Each scope/arm begins with its own snapshot of this exact case.
                # Cached calls reuse it only within this same fixture.
                self.shared.prepare(*self.original)
                start_generation = self.shared.generation
                boundaries = [phase(dist, torch, "warm20")]
                for _ in range(20):
                    held = self.call(arm, producer)
                boundaries.append(phase(dist, torch, "wall30"))
                wall_samples = []
                for index in range(30):
                    torch.npu.synchronize()
                    start_ns = time.perf_counter_ns()
                    held = self.call(arm, producer)
                    returned_ns = time.perf_counter_ns()
                    torch.npu.synchronize()
                    end_ns = time.perf_counter_ns()
                    wall_samples.append(dict(index=index, start_ns=start_ns, return_ns=returned_ns,
                                             end_ns=end_ns, wall_ms=(end_ns - start_ns) / 1e6))
                boundaries.append(phase(dist, torch, "event30_separate_batch"))
                event_samples = []
                for index in range(30):
                    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    host_start_ns = time.perf_counter_ns()
                    start.record()
                    held = self.call(arm, producer)
                    returned_ns = time.perf_counter_ns()
                    end.record()
                    end.synchronize()
                    end_ns = time.perf_counter_ns()
                    event_samples.append(dict(index=index, start_ns=host_start_ns, return_ns=returned_ns,
                                              end_ns=end_ns, event_ms=start.elapsed_time(end)))
                torch.npu.synchronize()
                records[scope][arm] = dict(warmup=20, wall_samples=wall_samples, event_samples=event_samples,
                    phase_boundaries=boundaries, generation_start=start_generation,
                    generation_end=self.shared.generation, events_are_separate_batch=True)
                del held
        check_owners(torch, self.loaded, self.provider)
        return records


def pool_stats(ops, handle):
    names, values = list(ops.stat_names()), list(ops.stats(handle))
    need(len(names) == len(values) and len(set(names)) == len(names), "invalid pool statistics")
    return dict(zip(names, values))


def validate_lifecycle(record):
    need(set(record) == {"automatic_reclaim", "burst", "missing_snapshot", "private_churn"},
         "private-retention lifecycle evidence missing")
    validate_private_churn(record["private_churn"])
    need(record["private_churn"]["before"] == record["missing_snapshot"]["after"],
         "private churn does not continue this case's lifecycle")
    reclaim = record["automatic_reclaim"]
    need(reclaim["after"]["producer_reclaimed_calls"] > reclaim["before"]["producer_reclaimed_calls"]
         and reclaim["after"]["producer_reclaim_tid"] == reclaim["after"]["producer_tid"] > 0,
         "case did not observe automatic producer reclaim")
    burst = record["burst"]
    need(burst["calls"] == 4 and burst["per_call_device_drain"] is False
         and burst["inflight_coverage_claim"] is False
         and burst["after"]["submitted"] - burst["before"]["submitted"] == 4
         and burst["after"]["callbacks_done"] - burst["before"]["callbacks_done"] == 4,
         "case burst evidence is incomplete")
    miss = record["missing_snapshot"]
    need(miss["before"]["submitted"] == miss["after"]["submitted"],
         "missing snapshot submitted work")
    for stage in (reclaim, burst, miss):
        for stats in (stage["before"], stage["after"]):
            need(all(stats[name] == 0 for name in ("pending", "poisoned", "pool_exhaustions",
                                                  "retirement_overflows")),
                 "case lifecycle includes incomplete or failed work")


def validate_private_churn(record):
    need(record.get("calls") == record.get("retained_send_receive_pairs") == 12
         and record.get("churn_allocations") == 72
         and record.get("per_call_device_drain") is False
         and record.get("inflight_coverage_claim") is False,
         "bounded private churn contract differs")
    for name in ("fresh_native_source_owners", "python_source_locals_released",
                 "python_lease_locals_released", "all_retained_outputs_bitwise",
                 "prior_arena_canaries_bitwise", "original_sources_unchanged", "original_hccl",
                 "pressure_allocated_on_pressure_stream", "pressure_stream_distinct",
                 "pressure_submitted_on_separate_stream"):
        need(record.get(name) is True, "private churn witness missing: " + name)
    for name in ("fresh_source_contents_after_release_observed", "native_private_reuse_addresses_observed",
                 "device_overlap_observed"):
        need(record.get(name) is False, "private churn exceeds observed scope: " + name)
    before, after = record["before"], record["after"]
    need(after["submitted"] - before["submitted"] == 12
         and after["callbacks_done"] - before["callbacks_done"] == 12
         and after["plan_reuses"] > before["plan_reuses"]
         and after["producer_reclaimed_calls"] > before["producer_reclaimed_calls"]
         and after["producer_reclaim_tid"] == after["producer_tid"] > 0,
         "private churn did not exercise completed-plan reuse/producer reclaim")
    for stats in (before, after):
        need(all(stats[name] == 0 for name in ("pending", "poisoned", "pool_exhaustions", "retirement_overflows")),
             "private churn left failed or pending work")


def pool_snapshots(provider):
    return [dict(handle=handle, stats=pool_stats(provider.ops, handle))
            for handle in provider.ops.handles()]


def validate_closed(snapshots):
    need(len(snapshots) == 1, "one provider-wide pool required")
    stats = snapshots[0]["stats"]
    need(stats["closed"] == stats["native_closed"] == 1
         and stats["finalize_tid"] == stats["producer_tid"] > 0
         and all(stats[name] == 0 for name in ("workspace_resident_bytes", "pending",
            "poisoned", "pool_exhaustions", "retirement_overflows", "retained_call_slots",
            "retired_count", "accepting", "finalizing")),
         "pool cleanup failed or retained owners/workspace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("correctness", "timing"), default="correctness")
    parser.add_argument("--round", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Fresh QK repeatable runtime binding")
    parser.add_argument("--correctness-dir", type=Path, help="Required for timing; matching complete8-rank receipts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        print("NOT EXECUTED: explicit --execute and root-owned torchrun8 supervisor required")
        return
    need(os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "8,9,10,11,12,13,14,15",
         "this experiment owns only physical dies8 through15")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    need(int(os.environ["WORLD_SIZE"]) == 8 and rank == local_rank and rank in range(8),
         "single-host torchrun8 logical0 through7 required")
    need(os.environ.get("ROOT_LINGBOT_SP8_EXPERIMENT_MODE") == args.mode,
         "supervisor experiment mode differs")
    for name in ("ROOT_LINGBOT_SP8_FATAL_DIR", "ROOT_LINGBOT_SP8_FATAL_NONCE"):
        need(bool(os.environ.get(name)), "launch-owned supervisor required")
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"rank{rank}.json"
    need(not output.exists(), "refuse to overwrite an earlier replay receipt")
    matrix, sources, manifest_pin = preflight(args.fixtures), source_pins(), pin(args.manifest)
    correctness = []
    if args.mode == "timing":
        need(args.correctness_dir is not None, "timing needs matching actual192 correctness first")
        correctness = verify_correctness(args.correctness_dir, matrix, manifest_pin, sources)
    manifest = json.loads(args.manifest.read_text())
    need(pin(manifest["provider_manifest"]["path"]) == manifest["provider_manifest"], "parent manifest changed")
    need(os.environ.get("LINGBOT_RECOVERY_MANIFEST") == manifest["provider_manifest"]["path"]
         and os.environ.get("LINGBOT_RECOVERY_MANIFEST_SHA256") == manifest["provider_manifest"]["sha256"],
         "supervisor provider build differs from candidate")
    report = dict(schema=SCHEMA, mode=args.mode, method=METHOD, rank=rank, round=args.round,
        matrix_sha256=matrix["sha256"], matrix_rank=matrix["ranks"][rank], manifest=manifest_pin,
        sources=sources, correctness_receipts=correctness, passed=False, cases=[],
        original_hccl_included=True, actual_payload_cases_per_rank=24, real_payload_layers=[0],
        actual192_candidate_correctness_global=False, all40_payloads=False, model_admitted=False,
        gate_passed=False, historical_single_host_identity_proven=False,
        supervisor_nonce=os.environ["ROOT_LINGBOT_SP8_FATAL_NONCE"],
        physical_devices=os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        host_clock="time.perf_counter_ns: same-host monotonic clock; no cross-host epoch claim",
        timing_caveats=["wall and event batches are distinct", "ranks share30 collective samples, not240 independent samples",
                        "per-call synchronization changes model submission", "no end-to-end gain inferred"],
        full_call_scope="optional coefficient producer, Python policy/admission, repeatable pool admission/reclaim/native arithmetic/pack, lease, unchanged all_to_all_5D HCCL and receive reorder")
    provider, replay, loaded = None, None, None
    try:
        import torch
        import torch_npu as backend
        import torch.distributed as dist
        torch.npu.set_device(local_rank)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = True
        need(torch.npu.get_device_name() == "Ascend910_9392", "arch22 device required")
        candidate = importlib.import_module("vllm_omni.diffusion.layers.lingbot_qk_private_retention")
        os.environ[candidate.MANIFEST] = manifest_pin["path"]
        os.environ[candidate.MANIFEST_SHA] = manifest_pin["sha256"]
        provider = candidate.load()
        dist.init_process_group("hccl")
        from vllm_omni.diffusion.distributed.comm import all_to_all_5D
        with torch.inference_mode():
            for case_pin in matrix["ranks"][rank]["cases"]:
                need(pin(case_pin["file"]["path"]) == case_pin["file"], "fixture metadata changed after preflight")
                source = load_case(case_pin["file"]["path"])
                loaded = source.materialize(torch, f"npu:{local_rank}", provider.base.describe,
                    lambda shape, dtype, device, fmt: backend.empty_with_format(
                        shape, dtype=dtype, device=device, acl_format=fmt))
                replay = Replay(torch, backend, candidate, provider, loaded, dist.group.WORLD, all_to_all_5D)
                phase(dist, torch, "case_ready")
                result = dict(identity=identity(source.metadata), fixture=case_pin["file"],
                    captured_coefficient_generation=source.metadata["coefficient_generation"],
                    replay_generation_is_fresh_local_counter=True, full_owner_sha256_verified=True,
                    captured_provider_sha256=source.metadata["dispatch_receipt"]["build_manifest_sha256"])
                if args.mode == "correctness":
                    result.update(checks=replay.correctness(), lifecycle=replay.lifecycle, passed=True)
                    validate_lifecycle(result["lifecycle"])
                else:
                    result.update(records=replay.timing(dist, args.round), passed=True)
                phase(dist, torch, "case_complete")
                report["cases"].append(result)
                print(f"rank={rank} case={len(report['cases'])}/24 mode={args.mode} complete", flush=True)
                # Release each full case only after all local consumers completed.
                replay.live.clear()
                replay = loaded = source = None
            # Cleanup is outside timing. SDK resources close on the consumer;
            # framework owners release on the producer, followed by final drain.
            phase(dist, torch, "before_pool_close")
            report["cold_initializations"] = provider.cold_records
            report["pool_before_close"] = pool_snapshots(provider)
            provider.close_all_drained()
            torch.npu.synchronize()
            provider.finalize_all_drained()
            torch.npu.synchronize()
            report["pool_after_close"] = pool_snapshots(provider)
            validate_closed(report["pool_after_close"])
            report["passed"] = True
            report["actual192_candidate_correctness_global"] = args.mode == "timing"
            # This field in timing means prerequisite receipts verified all8; a
            # standalone correctness rank report never claims the other7 passed.
            phase(dist, torch, "all_cases_complete")
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(str(output), flush=True)
        dist.barrier()
        dist.destroy_process_group()
    except BaseException as error:
        owners = (provider, replay, loaded, error)
        _QUARANTINE.append(owners)
        if provider is not None:
            provider.fail(error, owners)
        report.update(passed=False, error=repr(error), traceback=traceback.format_exc())
        output.write_text(json.dumps(report, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
