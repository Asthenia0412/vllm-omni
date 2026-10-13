"""Separate actual192 host diagnostic with allocation/dist/layout boundaries.

Never a performance gate or model admission. Requires fresh eight-rank
correctness for THIS wrapper before profile. The frozen diagnostic and runtime
files remain unchanged. No device work occurs on import or without --execute.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
import types

import diagnose_qk_host_dependencies as frozen
import observe_qk_collective_host as observer_api

replay_api = frozen.replay_api
need = replay_api.need
SCHEMA = "lingbot_qk_collective_boundaries_diagnostic_v2"
METHOD = "frozen_per_call_sync_private_constructor_and_scoped_module_delegates_v2"
SCOPES, ARMS = frozen.SCOPES, frozen.ARMS
INSTRUMENTED_CHECKS = ["instrumented_send_and_original_receive_for_both_arms_and_scopes",
                       "instrumented_source_owners_unchanged"]


def diagnostic_sources():
    return {"wrapper": replay_api.pin(__file__), "observer": replay_api.pin(observer_api.__file__),
            "frozen_diagnostic": replay_api.pin(frozen.__file__)}


def expected_collective_code(source_pin):
    """Compile trusted, pinned source without executing imports or model code."""
    need(replay_api.pin(source_pin["path"]) == source_pin, "collective source changed")
    module = compile(Path(source_pin["path"]).read_bytes(), source_pin["path"], "exec",
                     dont_inherit=True, optimize=sys.flags.optimize)
    codes = [item for item in module.co_consts
             if isinstance(item, types.CodeType) and item.co_name == "all_to_all_5D"]
    need(len(codes) == 1, "one top-level original all_to_all_5D required")
    return {"source_path": source_pin["path"], "source_sha256": source_pin["sha256"],
            "code_fingerprint_method": observer_api.CODE_FINGERPRINT_METHOD,
            "code_sha256": observer_api.code_fingerprint(codes[0]),
            "python": sys.version, "optimize": sys.flags.optimize,
            "derivation": "compile pinned full module without execution; unchanged imported code must match"}


class BoundaryIntegerTrace(frozen.IntegerTrace):
    def __init__(self, mode):
        need(mode in ("correctness", "profile"), "unknown boundary mode")
        super().__init__(24 * 2 * 2 * 60 if mode == "profile" else 4)
        self.mode, self.case_index, self.boundary_observer = mode, 0, None

    def boundary_row(self):
        return self.row if self.mode == "profile" else self.case_index * 4 + self.row


class CollectiveProbe(frozen.CollectiveProbe):
    def __call__(self, metadata, *, scatter_idx=3, gather_idx=1, group=None,
                 _prepared_send=None, _qk_padded_owner=None):
        self.trace.mark(frozen.COLLECTIVE_BEGIN)
        observer = self.trace.boundary_observer
        need(observer is not None and observer.collective is self.collective,
             "observer must wrap the original replay collective")
        if _qk_padded_owner is None:
            result = observer(self.trace.boundary_row(), metadata, scatter_idx=scatter_idx,
                gather_idx=gather_idx, group=group, _prepared_send=_prepared_send)
        else:
            result = observer(self.trace.boundary_row(), metadata, scatter_idx=scatter_idx,
                gather_idx=gather_idx, group=group, _prepared_send=_prepared_send,
                _qk_padded_owner=_qk_padded_owner)
        self.trace.mark(frozen.COLLECTIVE_END)
        return result


class InstrumentedCall(frozen.InstrumentedCall):
    # Inherit the exact producer/prepare/exchange call body and error quarantine.
    # Replace only the collective observer constructed before each run_arm.
    def __init__(self, replay, trace):
        self.replay, self.trace = replay, trace
        self.collective = CollectiveProbe(replay.collective, trace)


def private_constructor_binding(function):
    """Reuse frozen bytecode with an EXPLICIT private globals substitution.

    The frozen module's globals never change. Its quarantine list is shared so
    the existing fatal-owner policy still retains owners on any terminal error.
    """
    namespace = dict(function.__globals__)
    namespace["InstrumentedCall"] = InstrumentedCall
    clone = types.FunctionType(function.__code__, namespace, function.__name__,
                               function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return clone


run_arm = private_constructor_binding(frozen.run_arm)
instrumented_correctness = private_constructor_binding(frozen.instrumented_correctness)


def validate_boundaries(data, *, rows, code_pin):
    need(data.get("schema") == observer_api.SCHEMA and data.get("diagnostic_only") is True
         and data.get("performance_gate") is False and data.get("row_count") == rows
         and data.get("marks") == list(observer_api.MARKS)
         and data.get("columns") == list(observer_api.COLUMNS)
         and data.get("intervals") == [list(item) for item in observer_api.INTERVALS]
         and data.get("status_bits") == observer_api.STATUS
         and data.get("terminal_failure") is False
         and data.get("all_rows_complete") is True,
         "boundary schema/completeness differs")
    need(data.get("pins") == {name: code_pin[name] for name in
         ("source_path", "source_sha256", "code_fingerprint_method", "code_sha256")},
         "observed collective code/source differs")
    need(data.get("status") == [0] * rows and data.get("visited") == [1] * rows
         and data.get("restored_original_bindings") == [1] * rows
         and data.get("complete_rows") == [True] * rows, "boundary row incomplete")
    values = data.get("data", [])
    need(len(values) == rows * 6 * 4 and all(type(value) is int and value >= 0 for value in values),
         "invalid boundary integer array")
    for row in range(rows):
        for column in range(4):
            stamps = [values[(row * 6 + point) * 4 + column] for point in range(6)]
            need(stamps == sorted(stamps), "boundary counters are not monotonic")


def verify_diagnostic_correctness(directory, matrix, manifest_pin, sources, diagnostic_pins, code_pin):
    """Old broad diagnostic PASS cannot authorize this narrower profile."""
    pins = []
    for rank in range(8):
        path = Path(directory) / f"rank{rank}.json"
        data = json.loads(path.read_text())
        need(data.get("schema") == SCHEMA and data.get("method") == METHOD
             and data.get("mode") == "correctness" and data.get("passed") is True
             and data.get("rank") == rank and data.get("matrix_sha256") == matrix["sha256"]
             and data.get("manifest") == manifest_pin and data.get("sources") == sources
             and data.get("diagnostic_sources") == diagnostic_pins
             and data.get("expected_collective_code") == code_pin
             and data.get("diagnostic_only") is True and data.get("performance_timing_eligible") is False
             and data.get("gate_passed") is False and data.get("model_admitted") is False,
             "new diagnostic correctness binding differs")
        cases = data.get("cases", [])
        replay_api.validate_order([entry["identity"] for entry in cases])
        for index, (actual, wanted) in enumerate(zip(cases, matrix["ranks"][rank]["cases"])):
            need(actual.get("identity") == wanted["identity"] and actual.get("fixture") == wanted["file"]
                 and actual.get("checks") == list(replay_api.CHECKS)
                 and actual.get("instrumented_checks") == INSTRUMENTED_CHECKS
                 and actual.get("full_owner_sha256_verified") is True
                 and actual.get("boundary_rows") == {"first_row": index * 4, "row_count": 4}
                 and actual.get("passed") is True, "new diagnostic actual-case correctness incomplete")
        validate_boundaries(data.get("collective_boundaries", {}), rows=96, code_pin=code_pin)
        pins.append(replay_api.pin(path))
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=("correctness", "profile"), default="correctness")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--correctness-dir", type=Path, required=True,
                        help="Unchanged check_qk_real_matrix actual192 correctness directory")
    parser.add_argument("--diagnostic-correctness-dir", type=Path,
                        help="THIS wrapper's fresh eight-rank correctness; required for profile")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        print("NOT EXECUTED: explicit --execute and root-owned torchrun8 supervisor required")
        return
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
    sources, manifest_pin = replay_api.source_pins(), replay_api.pin(args.manifest)
    diagnostic_pins = diagnostic_sources()
    code_pin = expected_collective_code(sources["original_collective"])
    prior = replay_api.verify_correctness(args.correctness_dir, matrix, manifest_pin, sources)
    diagnostic_prior = []
    if args.mode == "profile":
        need(args.diagnostic_correctness_dir is not None, "profile requires matching new diagnostic correctness")
        diagnostic_prior = verify_diagnostic_correctness(args.diagnostic_correctness_dir, matrix,
            manifest_pin, sources, diagnostic_pins, code_pin)
    manifest = json.loads(args.manifest.read_text())
    need(replay_api.pin(manifest["provider_manifest"]["path"]) == manifest["provider_manifest"], "parent build changed")
    need(os.environ.get("LINGBOT_RECOVERY_MANIFEST") == manifest["provider_manifest"]["path"]
         and os.environ.get("LINGBOT_RECOVERY_MANIFEST_SHA256") == manifest["provider_manifest"]["sha256"],
         "supervisor provider build differs")
    trace, gc_trace = BoundaryIntegerTrace(args.mode), frozen.GCTrace()
    report = dict(schema=SCHEMA, method=METHOD, mode=args.mode, rank=rank, passed=False,
        matrix_sha256=matrix["sha256"], matrix_rank=matrix["ranks"][rank], sources=sources,
        diagnostic_sources=diagnostic_pins, expected_collective_code=code_pin, manifest=manifest_pin,
        correctness_receipts=prior, diagnostic_correctness_receipts=diagnostic_prior,
        cases=[], host_start=frozen.host_snapshot(),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        diagnostic_only=True, performance_timing_eligible=False, gate_passed=False, model_admitted=False,
        actual_payload_cases_per_rank=24, real_payload_layers=[0], all40_payloads=False,
        original_hccl_included=True, collective_group="isolated torchrun8 WORLD, not live model Ulysses instance",
        main_call_changes={"frozen_files_modified": False, "frozen_module_globals_modified": False,
            "reused_frozen_bytecode": ["run_arm", "instrumented_correctness", "InstrumentedCall.__call__"],
            "private_function_globals_replaced": {"InstrumentedCall": "this wrapper.InstrumentedCall"},
            "collective_observer": "this wrapper.CollectiveProbe -> PreparedCollectiveObserver -> original all_to_all_5D",
            "temporary_runtime_module_bindings": ["comm.torch delegate", "comm.dist delegate"],
            "torch_dist_public_attributes_patched": False,
            "boundary_rows": "correctness case_index*4 + outer_row; profile outer_row"},
        caveats=["extra proxy/markers/counters change host work and possibly GC schedule; no speed inference",
                 "original pre/post-call NPU sync retained; no per-stage NPU sync",
                 "inner five intervals partition inner call span; never add them to enclosing collective time",
                 "outer collective span additionally contains observer setup/checks/restoration",
                 "dist return includes framework submission/wait and is not device completion or HCCL arrival",
                 "allocation interval is initial receive empty_like only; later output processing may allocate",
                 "thread CPU/context switches do not identify a lock or prove scheduling cause",
                 "normal GC preserved; weakref death proves wrapper release, not device storage reclamation"])
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
        trace.boundary_observer = observer_api.PreparedCollectiveObserver(all_to_all_5D,
            trace.rows if args.mode == "profile" else 96,
            expected_source_sha256=code_pin["source_sha256"], expected_code_sha256=code_pin["code_sha256"])
        with torch.inference_mode(), gc_trace:
            for case_index, entry in enumerate(matrix["ranks"][rank]["cases"]):
                trace.case_index = case_index
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
                    result["boundary_rows"] = dict(first_row=case_index * 4, row_count=4)
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
        report.update(host_end=frozen.host_snapshot(), gc=gc_trace.export(),
                      collective_boundaries=trace.boundary_observer.export())
        # Normalize tuples to JSON lists for the exact same validator used at admission.
        validate_boundaries(json.loads(json.dumps(report["collective_boundaries"])),
            rows=trace.rows if args.mode == "profile" else 96, code_pin=code_pin)
        if args.mode == "profile":
            need(row == trace.rows, "diagnostic row count differs")
            report["trace"] = trace.export()
        report["passed"] = True
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(output, flush=True)
        dist.barrier()
        dist.destroy_process_group()
    except BaseException as error:
        owners = (provider, active, loaded, frozen._QUARANTINE, error)
        frozen._QUARANTINE.append(owners)
        if provider is not None:
            provider.fail(error, owners)
        report.update(passed=False, host_end=frozen.host_snapshot(), gc=gc_trace.export(),
                      error=repr(error), traceback=traceback.format_exc())
        if trace.boundary_observer is not None:
            report["collective_boundaries"] = trace.boundary_observer.export()
        output.write_text(json.dumps(report, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
