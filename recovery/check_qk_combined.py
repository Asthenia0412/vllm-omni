# SPDX-License-Identifier: Apache-2.0
"""Explicit isolated replay: full Python wrapper, snapshot producer and lease.

Current fixtures contain exactly request2/chunk11/probe0/layer0 on eight ranks.
No actual192 or complete40-layer payload claim. Timing is diagnostic pre-HCCL:
it includes producer.prepare (once for this first-layer case), route admission,
shared.get, the typed native call, lease creation, route-specific consumer guard
and record_stream; it excludes transfer/receive. Cached-snapshot timing is also
reported separately. Do not sum these isolated timings into chunk gains.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--mode", choices=("correctness", "timing"), default="correctness")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--hccl", action="store_true", help="torchrun8 only: original collective and receive reorder included")
    p.add_argument("--round", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not a.execute:
        print("NOT EXECUTED: root-owned supervisor and explicit --execute are required")
        return
    from qk_replay_fixture import Runtime, pin, require
    import torch
    with torch.inference_mode():
        group = None
        if a.hccl:
            import torch.distributed as dist
            require(int(os.environ["WORLD_SIZE"]) == 8, "HCCL requires all eight ranks")
            a.device = int(os.environ["LOCAL_RANK"])
            require(a.device == int(os.environ["RANK"]), "rank mapping changed")
        r = Runtime(a.device, a.fixtures)
        if a.hccl:
            dist.init_process_group("hccl")
            group = dist.group.WORLD
        manifest = json.loads(a.manifest.read_text())
        module_pin = next(x for x in manifest["runtime_sources"] if x["path"].endswith("/lingbot_qk_combined.py"))
        require(pin(module_pin["path"]) == module_pin, "candidate module changed")
        name = "vllm_omni.diffusion.layers.lingbot_qk_combined"
        if name in sys.modules:
            candidate = sys.modules[name]
            require(pin(candidate.__file__) == module_pin, "already imported candidate source differs")
        else:
            spec = importlib.util.spec_from_file_location(name, module_pin["path"])
            candidate = importlib.util.module_from_spec(spec)
            sys.modules[name] = candidate
            spec.loader.exec_module(candidate)
        os.environ[candidate.MANIFEST] = str(a.manifest.resolve())
        os.environ[candidate.MANIFEST_SHA] = pin(a.manifest)["sha256"]
        from vllm_omni.diffusion.layers.lingbot_native_bf16_qk import SharedCoefficients
        from vllm_omni.diffusion.layers.lingbot_recovery import prepare_send as baseline_prepare
        original = (r.source_views["cosine"], r.source_views["sine"])
        shared = SharedCoefficients()
        shared.prepare(*original)
        values = {candidate.CONFIG_KEY: True}
        module = SimpleNamespace(ulysses_world_size=8, num_local_heads=40, head_dim=128,
            norm_q=SimpleNamespace(eps=1e-6, weight=r.qw), norm_k=SimpleNamespace(eps=1e-6, weight=r.kw),
            _lingbot_native_bf16_qk_coefficients=shared)
        # This is explicitly an isolated leaf route, not forged model startup.
        route = candidate.Route(module, SimpleNamespace(model_config=values), values)
        provider = candidate.load()
        def consume(metadata, *, scatter_idx, gather_idx, group, _prepared_send, _qk_padded_owner=None):
            # Same canonical prepared-send checks before original all_to_all_5D.
            require((scatter_idx, gather_idx) == (3, 1) and tuple(metadata.shape) == (1,585,3,40,128),
                    "canonical prepared metadata required")
            if _qk_padded_owner is not None:
                candidate.validate_padded_send(metadata, _prepared_send, _qk_padded_owner, 8)
            else:
                require(tuple(_prepared_send.shape) == (8,585,3,1,5,128)
                        and _prepared_send.dtype == metadata.dtype == torch.bfloat16
                        and _prepared_send.device == metadata.device and metadata.device.type == "npu"
                        and _prepared_send.is_contiguous() and _prepared_send.storage_offset() == 0,
                        "canonical offset0 baseline send required")
            return _prepared_send  # Diagnostic stops before allocation/HCCL/receive.
        if a.hccl:
            from vllm_omni.diffusion.distributed.comm import all_to_all_5D
            consume = all_to_all_5D  # Untouched actual transfer+original receive order.
        def baseline():
            lease = baseline_prepare(module, r.raw, original, 1)
            require(lease is not None, "baseline refused actual fixture")
            return lease.exchange(consume, group)
        def combined():
            lease = route.prepare(r.raw, original, 1)
            require(lease is not None, "combined refused actual fixture")
            return lease.exchange(consume, group)
        if a.mode == "correctness":
            native_original, expected = r.native(check_original_rms=True)
            expected_consumer = consume(native_original, group=group) if a.hccl else expected
            arena = torch.full((8986112,), -7, dtype=torch.bfloat16, device=r.raw.device)
            target = arena.clone()
            target[256:-256].copy_(expected.flatten())
            actual, owner = provider.ops.run_with_test_arena(*r.arguments, arena)
            require(owner.data_ptr() == arena.data_ptr(), "test owner changed")
            r.equal(arena, target, "combined_full_target_arena_canaries")
            r.equal(actual, expected, "combined_sentinel_send")
            for name, fn in (("baseline", baseline), ("combined", combined)):
                got = fn()
                r.equal(got, expected_consumer, name + "_full_wrapper_consumer_output")
            r.check_sources()
            # A missing immutable producer snapshot refuses without expansion.
            shared2 = SharedCoefficients()
            module._lingbot_native_bf16_qk_coefficients = shared2
            require(route.prepare(r.raw, original, 1) is None and shared2.generation == 0,
                    "ordinary producer miss mutated snapshot")
            report = dict(**r.report(), passed=True, timing_validated=False)
        else:
            records = {}
            for scope in ("producer_plus_layer0_and_consumer_guard", "cached_snapshot_wrapper_and_consumer_guard"):
                arms = {"baseline": baseline, "combined": combined}
                order = ("combined", "baseline") if a.round == 2 else ("baseline", "combined")
                records[scope] = {}
                for name in order:
                    def call():
                        if scope.startswith("producer_"):
                            shared.prepare(*original)
                        return arms[name]()
                    for _ in range(20):
                        held = call()
                    torch.npu.synchronize()
                    samples = {"wall_ms": [], "event_ms": []}
                    for _ in range(30):
                        torch.npu.synchronize()
                        start = time.perf_counter_ns()
                        held = call()
                        torch.npu.synchronize()
                        samples["wall_ms"].append((time.perf_counter_ns()-start)/1e6)
                    for _ in range(30):
                        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                        start.record()
                        held = call()
                        end.record()
                        end.synchronize()
                        samples["event_ms"].append(start.elapsed_time(end))
                    records[scope][name] = samples
            report = dict(**r.report(), round=a.round, records=records, gate_passed=False,
                          scope=("producer/shared.get/admission/typed native call/lease/original HCCL/receive reorder" if a.hccl else __doc__),
                          reason="retained layer0 only; actual192/model admission remains false")
        report["original_hccl_included"] = a.hccl
        report.update(manifest=pin(a.manifest), harness=pin(__file__), fixture_loader=pin(Path(__file__).with_name("qk_replay_fixture.py")))
        if a.hccl:
            a.output = a.output / f"rank{a.device}.json"
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2)+"\n")
        print(a.output)
        if a.hccl:
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
