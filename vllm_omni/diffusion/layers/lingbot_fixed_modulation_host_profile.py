# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only cProfile observation of unchanged, actual StageB calls.

Only the explicit Screen host mode imports this module. No sealed callable or
code object is replaced. These instrumented host times cannot qualify a speed
candidate and do not identify actual NPU enqueue/arrival times.
"""
from __future__ import annotations

import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import sys

ITERATIONS = 20
LAYERS = 40
_QUARANTINE = []


def need(ok, message):
    if not ok:
        raise RuntimeError("StageB host profile: " + message)


def profiling_active():
    """Conservatively detect legacy or monitoring-based instrumentation.

    Python3.12 cProfile can occupy monitoring tool2 while getprofile() is None.
    Reject every reserved monitoring tool, including tools with local events or
    no globally enabled events; this diagnostic never owns their registration.
    Python3.11 has no sys.monitoring and keeps its original getprofile check.
    """
    if sys.getprofile() is not None:
        return True
    monitoring = getattr(sys, "monitoring", None)
    return monitoring is not None and any(monitoring.get_tool(index) is not None for index in range(6))


def function_key(key):
    filename, line, name = key
    return dict(filename=filename, line=line, name=name)


def profile_enqueue(invoke, synchronize):
    """Observe20 enqueues; all returned owners survive the unprofiled drain.

    The caller keeps the returned batch until its correctness checks finish.
    On any exception, preserve owners and rethrow for Screen's fatal channel.
    There is no retry, fallback, polling, file write or stage/device sync inside
    the profile interval. Existing profiling is rejected, not overwritten.
    """
    need(not profiling_active(), "existing thread profiler cannot be replaced")
    held = []
    profiler = cProfile.Profile()
    try:
        synchronize()
        need(not profiling_active(), "thread profiler changed before activation")
        profiler.enable()
        try:
            for _ in range(ITERATIONS):
                held.append(invoke())
        finally:
            profiler.disable()
        need(not profiling_active(), "unexpected profiler after observation")
        synchronize()
        stats = pstats.Stats(profiler)
        methods = []
        for key, (cc, nc, tt, ct, callers) in sorted(stats.stats.items()):
            caller_rows = []
            # cProfile stores caller tuples nc,cc,tt,ct (unlike function rows).
            for caller, (caller_nc, caller_cc, caller_tt, caller_ct) in sorted(callers.items()):
                caller_rows.append(dict(function=function_key(caller), primitive_calls=caller_cc,
                                        total_calls=caller_nc, tottime_s=caller_tt, cumtime_s=caller_ct))
            methods.append(dict(function=function_key(key), primitive_calls=cc, total_calls=nc,
                                tottime_s=tt, cumtime_s=ct, callers=caller_rows))
        return dict(enqueue_calls=ITERATIONS, actual_layers_per_call=LAYERS,
                    cprofile_total_tottime_s=stats.total_tt, methods=methods), held
    except BaseException as error:
        _QUARANTINE.append((held, profiler, error, error.__traceback__))
        raise


def file_pin(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def write_report(output, report):
    need(not profiling_active(), "receipt serialization cannot occur under a profiler")
    need(not output.exists(), "refuse earlier host profile receipt overwrite")
    temporary = output.with_suffix('.tmp')
    need(not temporary.exists(), "refuse unfinished host profile receipt overwrite")
    with temporary.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)


def _equal_outputs(torch, reference, actual):
    need(len(reference) == len(actual) == LAYERS, "all40 output layers required")
    for left, right in zip(reference, actual):
        need(len(left) == len(right) == 6, "six original views required")
        for a, b in zip(left, right):
            need(a.dtype == b.dtype == torch.float32 and tuple(a.shape) == tuple(b.shape)
                 and torch.equal(a.view(torch.int32), b.view(torch.int32))
                 and tuple(a.stride()) == tuple(b.stride()) and a.storage_offset() == b.storage_offset(),
                 "profiled region changed logical output bits/layout")


def run(screen, transformer, stageB, torch, expected):
    """Only called by explicit host-mode Screen.run; no formal timing path."""
    need(screen.host_profile is True and os.environ.get('LINGBOT_STAGEB_HOST_PROFILE') == '1',
         "explicit retained host profile mode required")
    need(os.environ.get('ROOT_LINGBOT_SP8_EXPERIMENT_MODE') == 'profile'
         and os.environ.get('VLLM_LINGBOT_NPU_ASYNC_METADATA') == '0',
         "owned profile service with metadataOFF required")
    need(not any(os.environ.get(key) for key in ('LINGBOT_CAPTURE_DIR', 'LINGBOT_PROFILE_CONTROL')),
         "tensor/device profilers cannot be combined")
    need(not profiling_active(), "existing profiler cannot be replaced")
    need(stageB.FLAG == 'lingbot_npu_fixed_modulation_cpp_v2', "actual CPPv2 core required")
    exact = tuple((length, probe) for length in (4680, 9360, 14040, 18720, 23400, 28080) for probe in range(4))
    need(tuple(expected) == exact and tuple(sorted(screen.cases)) == exact and len(transformer.blocks) == LAYERS,
         "complete actual24-context/40-layer matrix required")
    import torch.distributed as dist
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
    rank = int(get_sp_group().ulysses_rank)
    need(rank in range(8), "actual eight-rank SP group required")
    output = screen.directory / f'rank{rank}.json'
    screen.directory.mkdir(parents=True, exist_ok=True)
    need(not output.exists(), "refuse earlier diagnostic result overwrite")
    source = file_pin(__file__)
    screen_source = file_pin(screen.invoke.__code__.co_filename)
    report = dict(schema='lingbot_actual_stageB_host_profile_v1', rank=rank,
                  nonce=os.environ['ROOT_LINGBOT_SP8_FATAL_NONCE'],
                  source=source['path'], source_sha256=source['sha256'], screen_source=screen_source,
                  diagnostic_only=True, operator_only=True, model_timing_eligible=False,
                  speed_gate_eligible=False, performance_admitted=False, model_admitted=False,
                  profile_iterations=ITERATIONS, actual_layers=LAYERS, actual_histories=sorted({x[0] for x in exact}),
                  actual_probes=4, snapshot_bytes=screen.snapshot_bytes, cases=[], passed=False,
                  method='independent cProfile per case/arm around20 unchanged Screen.invoke calls; retain all20 outputs until unprofiled sync',
                  caveats=['instrumented host default wall timer; includes blocking inside host calls, not threadCPU time',
                           'cProfile overhead changes execution; cannot replace clean operator or model timing',
                           'current Python thread only; consumer threads/device execution are not decomposed',
                           'restage+actual40 native expressions OR restage+prepare_forward+40get are observed',
                           'no per-call or per-stage device synchronization inside cProfile; final drain is excluded',
                           'allocation/queue backpressure may block an enqueue; this is not proof of hardware arrival',
                           '20 outputs stay live through each drain; retention differs from formal timing',
                           'tottime/cumtime/caller intervals overlap; do not sum nested intervals',
                           'actual active-session tokens and snapshots; modulation is independent of history length',
                           'logical/source checks do not prove full arena/canary integrity or lifecycle acceptance'])
    for key in exact:
        case = screen.cases[key]
        args, snapshot = case['args'], case['snapshot']
        sources = [block.modulation.detach().clone() for block in transformer.blocks]
        source_before, timestep_before, snapshot_before = args[3].clone(), args[2].clone(), snapshot.clone()
        native = screen.invoke('native', transformer, stageB, case)
        torch.npu.synchronize()
        candidate = screen.invoke('stageB', transformer, stageB, case)
        torch.npu.synchronize()
        _equal_outputs(torch, native, candidate)
        entry = dict(history=key[0], probe=key[1], correctness=True, arms={})
        candidate = None
        for arm in ('native', 'stageB'):
            dist.barrier()
            stats, held = profile_enqueue(lambda: screen.invoke(arm, transformer, stageB, case), torch.npu.synchronize)
            _equal_outputs(torch, native, held[0])
            _equal_outputs(torch, native, held[-1])
            entry['arms'][arm] = stats
            held = None  # Only after profile-disabled drain and logical checks.
        for block, original in zip(transformer.blocks, sources):
            need(torch.equal(block.modulation.detach().view(torch.int16), original.view(torch.int16)),
                 "readonly modulation parameter changed")
        for a, b in ((args[3], source_before), (snapshot, snapshot_before)):
            need(torch.equal(a.view(torch.int16), b.view(torch.int16)), "BF16 source changed")
        need(torch.equal(args[2].view(torch.int32), timestep_before.view(torch.int32)), "timestep source changed")
        report['cases'].append(entry)
    dist.barrier()
    torch.npu.synchronize()
    report['passed'] = True
    write_report(output, report)
    dist.barrier()
