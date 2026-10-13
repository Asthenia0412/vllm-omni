# SPDX-License-Identifier: Apache-2.0
"""Explicit in-service operator screening; never a model timing mode.

Retain the actual 24 forward inputs, including a clone of the otherwise reused
staging value. After the session bank is complete, restore each captured value
to its original staging buffer and replay the complete 40-layer preparation.
Restaging, StageB admission and all lookups belong to the measured region.
Nothing here advances KV, camera, VAE or DMD state. The interrupted forward's
staging bytes are restored before returning to its normal block consumers.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

LENGTHS = (4680, 9360, 14040, 18720, 23400, 28080)
EXPECTED = tuple((length, probe) for length in LENGTHS for probe in range(4))
MAX_SNAPSHOT_BYTES = 24 * 128 * 1024


def need(ok, message):
    if not ok:
        raise RuntimeError("StageB operator screen: " + message)


def configure(pipeline, model_config):
    directory = os.environ.get("LINGBOT_STAGEB_OPERATOR_SCREEN_DIR")
    host_profile = os.environ.get("LINGBOT_STAGEB_HOST_PROFILE", "0")
    need(host_profile in ("0", "1"), "host profile mode must be explicit0/1")
    pipeline.transformer._lingbot_stageB_operator_screen = None
    if not directory:
        need(host_profile == "0", "host profile requires an explicit diagnostic directory")
        return
    if host_profile == "1":
        need(model_config.get("lingbot_npu_fixed_modulation_cpp_v2") is True
             and model_config.get("lingbot_npu_fixed_modulation_stageB", False) is False,
             "host profile requires the independent CPPv2 candidate")
    need(model_config.get("lingbot_npu_fixed_modulation_stageB") is True
         or model_config.get("lingbot_npu_fixed_modulation_cpp_v2") is True,
         "explicit StageB flag is required")
    need(os.environ.get("ROOT_LINGBOT_SP8_EXPERIMENT_MODE") == "profile",
         "screening is permitted only in an owned diagnostic service")
    need(os.environ.get("VLLM_LINGBOT_NPU_ASYNC_METADATA") == "0",
         "metadata remains OFF for this independent experiment")
    path = Path(directory)
    root = Path(os.environ["LINGBOT_RECOVERY_ROOT"]).resolve(strict=True)
    need(path.is_absolute() and path.resolve().is_relative_to(root / "audits"),
         "persistent audit directory required")
    need(not any(os.environ.get(k) for k in ("LINGBOT_CAPTURE_DIR", "LINGBOT_PROFILE_CONTROL")),
         "operator timing must not include tensor capture or profiler")
    pipeline.transformer._lingbot_stageB_operator_screen = Screen(path, host_profile=host_profile == "1")


class Screen:
    def __init__(self, directory, *, host_profile=False):
        self.directory = directory
        self.host_profile = host_profile
        self.cases = {}
        self.session = None
        self.snapshot_bytes = 0
        self.completed = False
        self.failed = False
        self.retained_failure = []

    def capture(self, transformer, token, timestep, source, projection, uniform, length, forward):
        try:
            return self._capture(transformer, token, timestep, source, projection, uniform, length, forward)
        except BaseException as error:
            if not self.failed:
                self.failed = True
                self.retained_failure.append((transformer, token, timestep, source, projection, forward, error))
                from . import lingbot_sp8_fatal
                lingbot_sp8_fatal.report_failure("stageB_operator_capture", repr(error), (self, error))
            raise

    def _capture(self, transformer, token, timestep, source, projection, uniform, length, forward):
        if self.completed:
            return forward
        need(not self.failed, "failed screen requires worker restart")
        from . import lingbot_fixed_timestep_bank_stageA_v1 as stageA
        from .lingbot_fixed_modulation_dispatch import for_transformer
        stageB = for_transformer(transformer)
        need(type(token) is stageA._CallToken and length in LENGTHS and uniform,
             "actual fixed-step context and one of six actual histories required")
        session = token.probe.session
        if self.session is None:
            self.session = session
        need(self.session is session, "operator corpus cannot mix sessions")
        key = length, token.probe.index
        import torch
        if key not in self.cases:
            snapshot = projection.clone()
            extent = snapshot.untyped_storage().nbytes()
            need(self.snapshot_bytes + extent <= MAX_SNAPSHOT_BYTES, "snapshot budget exceeded")
            self.snapshot_bytes += extent
            self.cases[key] = dict(args=(transformer, token, timestep, source, projection, uniform),
                                   snapshot=snapshot, extent=extent)
        if len(self.cases) != len(EXPECTED) or forward is None:
            return forward
        # All four probes and all40 layers must already be published, as
        # confirmed through the core's read-only diagnostic predicate.
        if not stageB.bank_published(forward):
            return forward
        saved_current = projection.clone()
        try:
            self.run(transformer, stageB, torch)
            self.completed = True
        except BaseException as error:
            self.failed = True
            self.retained_failure.append((saved_current, transformer, token, projection, error))
            # Execution has started: preserve the sources; never resume this
            # request by selecting native arithmetic after a screening failure.
            from . import lingbot_sp8_fatal
            lingbot_sp8_fatal.report_failure("stageB_operator_screen", repr(error), (self, error))
            raise
        finally:
            if not self.failed:
                projection.copy_(saved_current)
                torch.npu.synchronize()
                self.cases.clear()
        # The diagnostic replay deliberately prepared other genuine tokens.
        # Restore core invocation identity as well as the original buffer bytes.
        restored = stageB.prepare_forward(transformer, token, timestep, source, projection, uniform)
        need(restored is not None, "original model forward refused after diagnostic restoration")
        return restored

    def run(self, transformer, stageB, torch):
        need(os.environ.get("LINGBOT_STAGEB_HOST_PROFILE", "0") == ("1" if self.host_profile else "0"),
             "host profile policy changed after construction")
        if self.host_profile:
            from .lingbot_fixed_modulation_host_profile import run
            return run(self, transformer, stageB, torch, EXPECTED)
        import torch.distributed as dist
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
        rank = int(get_sp_group().ulysses_rank)
        output = self.directory / f"rank{rank}.json"
        self.directory.mkdir(parents=True, exist_ok=True)
        need(not output.exists(), "refuse earlier screening receipt overwrite")
        need(tuple(sorted(self.cases)) == EXPECTED, "incomplete or duplicate actual corpus")
        report = dict(schema="lingbot_actual_stageB_full_region_screen_v1", rank=rank,
                      operator_only=True, model_timing_eligible=False, model_admitted=False,
                      source=str(Path(__file__).resolve()),
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      nonce=os.environ["ROOT_LINGBOT_SP8_FATAL_NONCE"], cases=[],
                      warmups=20, wall_measurements=30, event_measurements=30,
                      actual_layers=40, actual_histories=list(LENGTHS), actual_probes=4,
                      snapshot_bytes=self.snapshot_bytes, passed=False,
                      scope="restage actual compact BF16 input; native40 original expressions OR StageB prepare_forward +40 get; retain outputs through full-region completion",
                      caveats=["wall/event are separate batches", "profile supervisor is an ownership label; torch/CANN profiling is disabled",
                               "whole-model latency is not measured", "no per-layer synchronization", "all 40 actual layers included"])
        for key in EXPECTED:
            case = self.cases[key]
            args = case["args"]
            projection, snapshot = args[4], case["snapshot"]
            sources = [block.modulation.detach().clone() for block in transformer.blocks]
            source_before, timestep_before = args[3].clone(), args[2].clone()
            snapshot_before = snapshot.clone()
            native = self.invoke("native", transformer, stageB, case)
            torch.npu.synchronize()
            candidate = self.invoke("stageB", transformer, stageB, case)
            torch.npu.synchronize()
            need(len(native) == len(candidate) == 40, "all40 outputs required")
            for left, right in zip(native, candidate):
                need(len(left) == len(right) == 6, "six original output views required")
                for a, b in zip(left, right):
                    need(a.dtype == b.dtype == torch.float32 and tuple(a.shape) == tuple(b.shape)
                         and torch.equal(a.view(torch.int32), b.view(torch.int32)),
                         "original and cached modulation bits differ")
                # Compare all six logical views and their original layout.
                # This alone does not prove allocation padding or canaries.
                need([tuple(t.stride()) for t in left] == [tuple(t.stride()) for t in right]
                     and [t.storage_offset() for t in left] == [t.storage_offset() for t in right],
                     "borrowed view layout differs")
            entry = dict(history=key[0], probe=key[1], correctness=True, arms={})
            native = candidate = None
            for arm in ("native", "stageB"):
                dist.barrier()
                torch.npu.synchronize()
                for _ in range(20):
                    held = self.invoke(arm, transformer, stageB, case)
                    torch.npu.synchronize()
                wall, event = [], []
                for _ in range(30):
                    torch.npu.synchronize()
                    start = time.perf_counter_ns()
                    held = self.invoke(arm, transformer, stageB, case)
                    torch.npu.synchronize()
                    wall.append(time.perf_counter_ns() - start)
                for _ in range(30):
                    torch.npu.synchronize()
                    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    begin.record()
                    held = self.invoke(arm, transformer, stageB, case)
                    end.record()
                    end.synchronize()
                    event.append(begin.elapsed_time(end))
                entry["arms"][arm] = dict(wall_ns=wall, event_ms=event)
            for block, original in zip(transformer.blocks, sources):
                need(torch.equal(block.modulation.detach().view(torch.int16), original.view(torch.int16)),
                     "readonly modulation parameter changed")
            for a, b in ((args[3], source_before), (snapshot, snapshot_before)):
                need(torch.equal(a.view(torch.int16), b.view(torch.int16)), "BF16 source changed")
            need(torch.equal(args[2].view(torch.int32), timestep_before.view(torch.int32)), "timestep source changed")
            reference_after = self.invoke("native", transformer, stageB, case)
            torch.npu.synchronize()
            for left, right in zip(reference_after, held):
                for a, b in zip(left, right):
                    need(torch.equal(a.view(torch.int32), b.view(torch.int32)), "cached result changed after replay")
            projection.copy_(snapshot)
            torch.npu.synchronize()
            need(torch.equal(projection.view(torch.int16), snapshot.view(torch.int16)),
                 "restaged input differs")
            report["cases"].append(entry)
        report["passed"] = True
        dist.barrier()
        torch.npu.synchronize()
        temporary = output.with_suffix(".tmp")
        with temporary.open("w") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
        dist.barrier()

    @staticmethod
    def invoke(arm, transformer, stageB, case):
        args = case["args"]
        projection = args[4]
        projection.copy_(case["snapshot"])
        if arm == "native":
            return tuple((block.modulation.unsqueeze(1) + projection.float()).chunk(6, dim=2)
                         for block in transformer.blocks)
        forward = stageB.prepare_forward(*args)
        need(forward is not None and stageB.bank_published(forward), "warm actual StageB bank refused")
        return tuple(forward.get(index, block, projection) for index, block in enumerate(transformer.blocks))
