"""Bounded host-dependency observations of real QK/HCCL replay; NEVER a speed gate.

CPU-only import. Device execution requires an owned torchrun8 supervisor. The
frozen checker and runtime are imported unchanged. Only layer0 has payloads.
"""
from __future__ import annotations

import argparse
from array import array
import gc
import importlib
import json
import os
from pathlib import Path
import resource
import threading
import time
import traceback
import weakref

import check_qk_real_matrix as replay_api

SCHEMA = "lingbot_qk_host_dependency_diagnostic_v1"
METHOD = "integer_boundaries_normal_gc_per_call_sync_v1"
MARKS = ("pre_sync_begin", "pre_sync_end", "producer_begin", "producer_end",
         "prepare_begin", "prepare_end", "exchange_begin", "collective_begin",
         "collective_end", "exchange_end", "call_return_before_replace",
         "old_held_replace_end", "post_sync_begin", "post_sync_end")
COLS = ("wall_ns", "thread_cpu_ns", "thread_voluntary_context_switches",
        "thread_involuntary_context_switches")
SCOPES = (("producer_plus_layer0_and_original_consumer", True),
          ("cached_snapshot_and_original_consumer", False))
ARMS = ("baseline", "combined")
PRE_BEGIN, PRE_END, PRODUCER_BEGIN, PRODUCER_END, PREPARE_BEGIN, PREPARE_END, EXCHANGE_BEGIN, COLLECTIVE_BEGIN, COLLECTIVE_END, EXCHANGE_END, BEFORE_REPLACE, AFTER_REPLACE, POST_BEGIN, POST_END = range(14)
_QUARANTINE = []
need = replay_api.need


class IntegerTrace:
    """Only scalars; it must never retain a lease/tensor or a frame."""
    def __init__(self, rows):
        need(hasattr(resource, "RUSAGE_THREAD"), "RUSAGE_THREAD is required; process counters are not substituted")
        self.rows = rows
        self.data = array("q", [-1]) * (rows * len(MARKS) * len(COLS))
        self.alive = array("q", [-1]) * (rows * 2)
        self.row = -1
        self.native_thread_id = threading.get_native_id()

    def mark(self, point):
        # These four observations are sequential, not an atomic snapshot.
        base = (self.row * len(MARKS) + point) * len(COLS)
        self.data[base] = time.perf_counter_ns()
        self.data[base + 1] = time.thread_time_ns()
        usage = resource.getrusage(resource.RUSAGE_THREAD)
        self.data[base + 2] = usage.ru_nvcsw
        self.data[base + 3] = usage.ru_nivcsw

    def export(self):
        return dict(row_count=self.rows, marks=MARKS, columns=COLS,
                    data=list(self.data), main_native_thread_id=self.native_thread_id,
                    old_wrapper_alive_after_replace=list(self.alive),
                    old_wrapper_columns=["lease", "receive_tensor"],
                    allocation="fixed signed int64 arrays, allocated before device replay",
                    observation_order="wall then thread CPU then RUSAGE_THREAD; not atomic")


class GCTrace:
    """Process-global callbacks, bounded scalar-only writes; never collect."""
    def __init__(self, capacity=32768):
        self.capacity = capacity
        self.data = array("q", [-1]) * (capacity * 4)
        self.count = self.dropped = self.errors = 0
        self.callback = self._callback  # Stable identity for removal.

    def _callback(self, phase, info):
        try:
            if self.count >= self.capacity:
                self.dropped += 1
                return
            base = self.count * 4
            self.data[base] = time.perf_counter_ns()
            self.data[base + 1] = threading.get_native_id()
            self.data[base + 2] = 0 if phase == "start" else 1 if phase == "stop" else -1
            self.data[base + 3] = info.get("generation", -1)
            self.count += 1
        except BaseException:
            self.errors += 1  # Never interfere with the runtime's collection.

    def __enter__(self):
        gc.callbacks.append(self.callback)
        return self

    def __exit__(self, *_):
        # Preserve callbacks independently added/removed while this one is live.
        for index in range(len(gc.callbacks) - 1, -1, -1):
            if gc.callbacks[index] is self.callback:
                del gc.callbacks[index]

    def export(self):
        return dict(columns=["wall_ns", "native_thread_id", "phase_start0_stop1", "generation"],
                    data=list(self.data[:self.count * 4]), count=self.count,
                    capacity=self.capacity, dropped=self.dropped, errors=self.errors,
                    complete=self.dropped == 0 and self.errors == 0,
                    scope="process-global callbacks, including other Python threads",
                    interpretation="overlap is correlation, not proof that collection caused a stall")


class CollectiveProbe:
    def __init__(self, collective, trace):
        self.collective, self.trace = collective, trace

    def __call__(self, metadata, *, scatter_idx=3, gather_idx=1, group=None,
                 _prepared_send=None, _qk_padded_owner=None):
        self.trace.mark(COLLECTIVE_BEGIN)
        # Preserve the exact baseline call as well as the candidate padded call.
        if _qk_padded_owner is None:
            result = self.collective(metadata, scatter_idx=scatter_idx, gather_idx=gather_idx,
                                     group=group, _prepared_send=_prepared_send)
        else:
            result = self.collective(metadata, scatter_idx=scatter_idx, gather_idx=gather_idx,
                group=group, _prepared_send=_prepared_send, _qk_padded_owner=_qk_padded_owner)
        self.trace.mark(COLLECTIVE_END)
        return result  # No argument/result is stored on this object.


class InstrumentedCall:
    def __init__(self, replay, trace):
        self.replay, self.trace = replay, trace
        self.collective = CollectiveProbe(replay.collective, trace)

    def __call__(self, arm, producer):
        lease = result = None
        try:
            if producer:
                self.trace.mark(PRODUCER_BEGIN)
                self.replay.shared.prepare(*self.replay.original)
                self.trace.mark(PRODUCER_END)
            self.trace.mark(PREPARE_BEGIN)
            lease = self.replay.prepare(arm)
            self.trace.mark(PREPARE_END)
            self.trace.mark(EXCHANGE_BEGIN)
            result = lease.exchange(self.collective, self.replay.group)
            self.trace.mark(EXCHANGE_END)
            return lease, result
        except BaseException as error:
            _QUARANTINE.append((self.replay, lease, result, error))
            raise


def run_arm(replay, arm, producer, trace, first_row, phase_boundary, *, calls=60, warmup=20):
    """The sole old-held tuple reference is local to this function.

    Do not move replacement into a helper taking held: its caller would retain
    the old tuple past the observed release span. No old_held alias is allowed.
    """
    need(calls > 0 and warmup > 0 and first_row >= 0
         and first_row + calls <= trace.rows, "invalid diagnostic buffer extent")
    instrumented = InstrumentedCall(replay, trace)
    held = new_held = None
    try:
        replay.shared.prepare(*replay.original)
        generation_start = replay.shared.generation
        phase_boundary("warm20")
        # Exactly the original checker warmup: no new per-call synchronization.
        for _ in range(warmup):
            held = replay.call(arm, producer)
        phase_boundary("diagnostic_calls")
        for index in range(calls):
            trace.row = first_row + index
            # No callbacks and no reference to the referent. These are prepared
            # before the original pre-sync/start boundary.
            old_lease = weakref.ref(held[0])
            old_result = weakref.ref(held[1])
            trace.mark(PRE_BEGIN)
            replay.torch.npu.synchronize()
            trace.mark(PRE_END)
            new_held = instrumented(arm, producer)
            trace.mark(BEFORE_REPLACE)
            held = new_held  # THIS STORE_FAST decrefs the sole previous tuple.
            trace.mark(AFTER_REPLACE)
            del new_held  # Only drops the extra reference to the NEW tuple.
            trace.alive[trace.row * 2] = old_lease() is not None
            trace.alive[trace.row * 2 + 1] = old_result() is not None
            trace.mark(POST_BEGIN)
            replay.torch.npu.synchronize()
            trace.mark(POST_END)
        # The final held remains live through its original post-sync.
        return dict(first_row=first_row, row_count=calls, warmup=warmup,
                    generation_start=generation_start, generation_end=replay.shared.generation)
    except BaseException as error:
        # Failure may occur after prepare or communication. Never retry/fallback.
        _QUARANTINE.append((replay, held, locals().get("new_held"), error))
        raise


def instrumented_correctness(replay, trace):
    """Check the wrapper's real send/receive and sources; not a speed sample."""
    for scope_index, (_, producer) in enumerate(SCOPES):
        for arm_index, arm in enumerate(ARMS):
            trace.row = scope_index * 2 + arm_index
            held = InstrumentedCall(replay, trace)(arm, producer)
            replay.torch.npu.synchronize()
            try:
                replay_api.equal(replay.torch, held[0].send, replay.reference, "instrumented send")
                replay_api.equal(replay.torch, held[1], replay.live[0], "instrumented original HCCL receive")
            except BaseException as error:
                _QUARANTINE.append((replay, held, error))
                raise
            del held
    replay_api.check_owners(replay.torch, replay.loaded, replay.provider)
    return ["instrumented_send_and_original_receive_for_both_arms_and_scopes", "instrumented_source_owners_unchanged"]


def host_snapshot():
    """Non-hot evidence; only current worker/CPU cgroup, never process sweeping."""
    result = dict(pid=os.getpid(), native_thread_id=threading.get_native_id(),
                  affinity=sorted(os.sched_getaffinity(0)), gc_enabled=gc.isenabled(),
                  gc_thresholds=gc.get_threshold(), gc_callback_ids=[id(item) for item in gc.callbacks])
    try:
        result["proc_self_cgroup"] = Path("/proc/self/cgroup").read_text()
        result["cpu_cgroup_files"] = {}
        for line in result["proc_self_cgroup"].splitlines():
            _, controllers, relative = line.split(":", 2)
            if "cpu" not in controllers.split(","):
                continue
            base = Path("/sys/fs/cgroup/cpu")
            current = base / relative.lstrip("/")
            while current.is_relative_to(base):
                for name in ("cpu.stat", "cpu.cfs_quota_us", "cpu.cfs_period_us"):
                    path = current / name
                    result["cpu_cgroup_files"][str(path)] = path.read_text() if path.exists() else None
                if current == base:
                    break
                current = current.parent
        # v2 is reported explicitly if this host has moved to a unified mount.
        for line in result["proc_self_cgroup"].splitlines():
            if line.startswith("0::"):
                base = Path("/sys/fs/cgroup")
                current = base / line[3:].lstrip("/")
                while current.is_relative_to(base):
                    for name in ("cpu.stat", "cpu.max"):
                        path = current / name
                        result["cpu_cgroup_files"][str(path)] = path.read_text() if path.exists() else None
                    if current == base:
                        break
                    current = current.parent
    except (OSError, ValueError) as error:
        result["cgroup_observation_error"] = repr(error)
    return result


def verify_diagnostic_correctness(directory, matrix, manifest_pin, sources, diagnostic_pin):
    pins = []
    for rank in range(8):
        path = Path(directory) / f"rank{rank}.json"
        data = json.loads(path.read_text())
        need(data.get("schema") == SCHEMA and data.get("method") == METHOD
             and data.get("mode") == "correctness" and data.get("passed") is True
             and data.get("rank") == rank and data.get("matrix_sha256") == matrix["sha256"]
             and data.get("manifest") == manifest_pin and data.get("sources") == sources
             and data.get("diagnostic_source") == diagnostic_pin, "diagnostic correctness binding differs")
        cases = data.get("cases", [])
        replay_api.validate_order([entry["identity"] for entry in cases])
        for actual, wanted in zip(cases, matrix["ranks"][rank]["cases"]):
            need(actual.get("identity") == wanted["identity"] and actual.get("fixture") == wanted["file"]
                 and actual.get("checks") == list(replay_api.CHECKS)
                 and actual.get("instrumented_checks") == ["instrumented_send_and_original_receive_for_both_arms_and_scopes", "instrumented_source_owners_unchanged"]
                 and actual.get("passed") is True, "diagnostic actual-case correctness incomplete")
        pins.append(replay_api.pin(path))
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("correctness", "profile"), default="correctness")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--correctness-dir", type=Path, required=True,
                        help="Unchanged check_qk_real_matrix actual192 receipt directory")
    parser.add_argument("--diagnostic-correctness-dir", type=Path, help="Required for profile")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        print("NOT EXECUTED: explicit --execute and root-owned torchrun8 supervisor required")
        return
    need(hasattr(resource, "RUSAGE_THREAD"), "RUSAGE_THREAD unavailable")
    need(os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == "8,9,10,11,12,13,14,15", "only owned dies8 through15")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    need(int(os.environ["WORLD_SIZE"]) == 8 and rank == local_rank and rank in range(8), "torchrun8 required")
    need(os.environ.get("ROOT_LINGBOT_SP8_EXPERIMENT_MODE") == args.mode, "supervisor mode differs")
    for name in ("ROOT_LINGBOT_SP8_FATAL_DIR", "ROOT_LINGBOT_SP8_FATAL_NONCE"):
        need(bool(os.environ.get(name)), "owned supervisor required")
    output = args.output / f"rank{rank}.json"
    args.output.mkdir(parents=True, exist_ok=True)
    need(not output.exists(), "refuse to overwrite a diagnostic receipt")
    matrix = replay_api.preflight(args.fixtures)
    sources, manifest_pin, diagnostic_pin = replay_api.source_pins(), replay_api.pin(args.manifest), replay_api.pin(__file__)
    prior = replay_api.verify_correctness(args.correctness_dir, matrix, manifest_pin, sources)
    diagnostic_prior = []
    if args.mode == "profile":
        need(args.diagnostic_correctness_dir is not None, "profile requires matching diagnostic correctness")
        diagnostic_prior = verify_diagnostic_correctness(args.diagnostic_correctness_dir, matrix, manifest_pin, sources, diagnostic_pin)
    manifest = json.loads(args.manifest.read_text())
    need(replay_api.pin(manifest["provider_manifest"]["path"]) == manifest["provider_manifest"], "parent build changed")
    need(os.environ.get("LINGBOT_RECOVERY_MANIFEST") == manifest["provider_manifest"]["path"]
         and os.environ.get("LINGBOT_RECOVERY_MANIFEST_SHA256") == manifest["provider_manifest"]["sha256"],
         "supervisor provider build differs")
    trace = IntegerTrace(24 * 2 * 2 * 60 if args.mode == "profile" else 4)
    gc_trace = GCTrace()
    report = dict(schema=SCHEMA, method=METHOD, mode=args.mode, rank=rank, passed=False,
        matrix_sha256=matrix["sha256"], matrix_rank=matrix["ranks"][rank], sources=sources,
        diagnostic_source=diagnostic_pin, manifest=manifest_pin, correctness_receipts=prior,
        diagnostic_correctness_receipts=diagnostic_prior, cases=[], host_start=host_snapshot(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        diagnostic_only=True, performance_timing_eligible=False, gate_passed=False, model_admitted=False,
        actual_payload_cases_per_rank=24, real_payload_layers=[0], all40_payloads=False,
        original_hccl_included=True, collective_group="isolated torchrun8 WORLD, not live model Ulysses instance",
        caveats=["markers/proxy/weakrefs/counters change host work and possibly GC schedule; no speed inference",
                 "pre/post-call NPU sync retained; no extra per-stage NPU sync",
                 "frozen checker return_ns includes old held destruction; diagnostic replacement end is only an analogue",
                 "Python weakref death proves wrapper release, not device storage reclamation",
                 "thread CPU/context switches are host observations, not proof of device execution or CPU cause",
                 "normal GC enablement/thresholds preserved; callback overlap is not causal proof"])
    provider = active = loaded = None
    row = 0
    try:
        import torch
        import torch_npu as backend
        import torch.distributed as dist
        torch.npu.set_device(local_rank)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = True
        need(torch.npu.get_device_name() == "Ascend910_9392", "arch22 required")
        candidate = importlib.import_module("vllm_omni.diffusion.layers.lingbot_qk_combined")
        os.environ[candidate.MANIFEST], os.environ[candidate.MANIFEST_SHA] = manifest_pin["path"], manifest_pin["sha256"]
        provider = candidate.load()
        dist.init_process_group("hccl")
        from vllm_omni.diffusion.distributed.comm import all_to_all_5D
        with torch.inference_mode(), gc_trace:
            for entry in matrix["ranks"][rank]["cases"]:
                need(replay_api.pin(entry["file"]["path"]) == entry["file"], "fixture changed")
                source = replay_api.load_case(entry["file"]["path"])
                loaded = source.materialize(torch, f"npu:{local_rank}", provider.base.describe,
                    lambda shape, dtype, device, fmt: backend.empty_with_format(shape, dtype=dtype, device=device, acl_format=fmt))
                active = replay_api.Replay(torch, backend, candidate, provider, loaded, dist.group.WORLD, all_to_all_5D)
                replay_api.phase(dist, torch, "case_ready")
                result = dict(identity=replay_api.identity(source.metadata), fixture=entry["file"], passed=False,
                              full_owner_sha256_verified=True)
                if args.mode == "correctness":
                    result["checks"] = active.correctness()
                    result["instrumented_checks"] = instrumented_correctness(active, trace)
                else:
                    result["phases"] = []
                    for scope, producer in SCOPES:
                        for arm in ARMS:
                            boundaries = []
                            phase_boundary = lambda name: boundaries.append(replay_api.phase(dist, torch, name))
                            item = run_arm(active, arm, producer, trace, row, phase_boundary)
                            item.update(scope=scope, arm=arm, phase_boundaries=boundaries)
                            result["phases"].append(item)
                            row += item["row_count"]
                    replay_api.check_owners(torch, loaded, provider)
                replay_api.phase(dist, torch, "case_complete")
                result["passed"] = True
                report["cases"].append(result)
                active.live.clear()
                active = loaded = source = None
            replay_api.phase(dist, torch, "all_cases_complete")
        report.update(passed=True, host_end=host_snapshot(), gc=gc_trace.export())
        if args.mode == "profile":
            need(row == trace.rows, "diagnostic row count differs")
            report["trace"] = trace.export()
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(output, flush=True)
        dist.barrier()
        dist.destroy_process_group()
    except BaseException as error:
        owners = (provider, active, loaded, _QUARANTINE, error)
        _QUARANTINE.append(owners)
        if provider is not None:
            provider.fail(error, owners)
        report.update(passed=False, host_end=host_snapshot(), gc=gc_trace.export(),
                      error=repr(error), traceback=traceback.format_exc())
        output.write_text(json.dumps(report, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
