"""Replay actual layer0 cross fixtures through the complete Attention wrapper.

Only --mode correctness/timing imports torch or touches the NPU. The normal
Attention constructor is used under an isolated configuration context; this is
not service/model startup. There is no Q/K projection, output projection, HCCL
or end-to-end performance claim. Run device modes only under the owned launcher.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from load_real_fixtures import case_paths, contract, load_case, validate_case_metadata

SCHEMA = "lingbot_cross_real_matrix_replay_v1"
EXPECTED = tuple((length, probe) for length in contract.LENGTHS for probe in range(4))
ARMS = ("plain", "combined")
OPERATIONS = {"plain": "cross", "combined": "try_cross_combined"}
CHECKS = (
    "actual_full_wrapper_dispatch", "finite_outputs", "captured_native_output_bitwise",
    "plain_combined_bitwise", "full_output_owner_bitwise", "fresh_output_owners",
    "full_source_and_reference_owners", "source_descriptors_unchanged",
)
_QUARANTINE = []


def need(value, reason):
    if not value:
        raise RuntimeError("Cross real matrix: " + reason)


def pin(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identity(case):
    return {key: case[key] for key in ("rank", "request_id", "chunk", "kv_len", "probe", "layer")}


def validate_order(cases):
    need(tuple((case["kv_len"], case["probe"]) for case in cases) == EXPECTED,
         "all24 real cases must have canonical history/probe order")
    need(all(case["layer"] == 0 for case in cases), "only actual layer0 payloads are admitted")


def validate_capture_case(case, rank, build):
    validate_case_metadata(case)
    need(case["rank"] == rank and case["kind"] == "cross", "wrong actual role/rank")
    need(case["actual_backend"] == "lingbot_recovery_v1::cross"
         and case["scale"] == 128 ** -.5 and case["text_kv_len"] == 512,
         "captured arithmetic/backend is not plain H40 BSND")
    need(case["dispatch_receipt"]["build_manifest_sha256"] == build, "case/build mismatch")
    for name, shape in (("query", [1, 585, 40, 128]), ("key", [1, 512, 40, 128]),
                        ("value", [1, 512, 40, 128])):
        desc = case["inputs"]["descriptors"][name]
        need(desc["shape"] == shape and desc["dtype"] == "torch.bfloat16",
             "actual operand shape/dtype differs")


def read_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    need(len(config["stages"]) == 1, "one captured diffusion stage required")
    stage = config["stages"][0]
    need(stage["model_class_name"] == "LingBotWorldCausalDMDPipeline"
         and stage["enforce_eager"] is True, "captured LingBot eager configuration required")
    parallel = stage["parallel_config"]
    need(tuple(parallel[key] for key in ("tensor_parallel_size", "sequence_parallel_size",
                                        "ulysses_degree", "ring_degree")) == (1, 8, 8, 1),
         "captured SP8/TP1/ring1 required")
    need(stage.get("diffusion_attention_config") is None
         and stage.get("diffusion_kv_cache_dtype") in (None, "auto", "float"),
         "this constructor contract does not adapt explicit attention/KV policies")
    values = stage["model_config"]
    need(values.get("lingbot_npu_recovery_cross") is True
         and values.get("lingbot_npu_recovery_cross_combined") is False
         and values.get("lingbot_npu_cross_combined_guard_cpp_v1") is False,
         "capture configuration is not the plain BSND reference")
    return stage


def preflight(plan_path, audit_path):
    """Verify all192 actual metadata records without reading tensor payloads."""
    plan = json.loads(Path(plan_path).read_text())
    need(plan["schema"] == "lingbot_real_fixture_launch_plan_v1" and plan["phase"] == "full",
         "frozen full capture launch plan required")
    config_pin, manifest_pin = pin(plan["config"]), pin(plan["manifest"])
    need(config_pin["sha256"] == plan["config_sha256"]
         and manifest_pin["sha256"] == plan["manifest_sha256"], "capture config/build changed")
    for saved in plan["harness_pins"]:
        need(pin(saved["path"]) == saved, "frozen capture harness changed")
    stage = read_config(plan["config"])
    audit = json.loads(Path(audit_path).read_text())
    need(audit["schema"] == "lingbot_real_fixture_matrix_audit_v1"
         and audit["complete_input_matrix"] is True and audit["actual192_global_proven"] is True
         and audit["cross_real_layer0_payload_cases"] == 192
         and audit["all40_payloads"] is False and audit["payload_sha256_verified"] is True
         and audit["build_manifest_sha256"] == manifest_pin["sha256"],
         "successful globally hashed actual192 capture audit required")
    root = Path(plan["fixture_root"]).resolve(strict=True)
    ranks, locations = [], []
    for rank in range(8):
        coverage_path = root / f"rank{rank}" / "coverage.json"
        coverage = json.loads(coverage_path.read_text())
        need(coverage["schema"] == "lingbot_real_qk_cross_capture_v1"
             and coverage["rank"] == rank and coverage["complete"] is True
             and coverage["failed"] is False
             and coverage["build_manifest"]["sha256"] == manifest_pin["sha256"],
             "rank capture/build differs")
        paths = case_paths(root, "cross", rank)
        cases = [json.loads(path.read_text()) for path in paths]
        validate_order(cases)
        entries, selected = [], []
        for path, case in zip(paths, cases, strict=True):
            validate_capture_case(case, rank, manifest_pin["sha256"])
            entries.append(dict(fixture=pin(path), identity=identity(case)))
            selected.append((case["kv_len"], case["probe"], case["request_id"], case["chunk"]))
        ranks.append(dict(rank=rank, coverage=pin(coverage_path), cases=entries))
        locations.append(selected)
    need(all(values == locations[0] for values in locations), "rank actual occurrences differ")
    result = dict(root=str(root), capture_plan=pin(plan_path), capture_audit=pin(audit_path),
                  config=config_pin, manifest=manifest_pin, ranks=ranks,
                  cross_real_layer0_payload_cases=192, all40_payloads=False)
    result["sha256"] = digest(result)
    return result, stage


def source_pins():
    base = Path(__file__).resolve().parents[1]
    return {name: pin(base / relative) for name, relative in (
        ("harness", "recovery/check_cross_real_matrix.py"),
        ("loader", "recovery/load_real_fixtures.py"),
        ("capture_contract", "vllm_omni/diffusion/layers/lingbot_real_fixture_capture.py"),
        ("configuration_context", "vllm_omni/diffusion/config.py"),
        ("forward_context", "vllm_omni/diffusion/forward_context.py"),
        ("model_metadata", "vllm_omni/diffusion/model_metadata.py"),
        ("caller", "vllm_omni/diffusion/attention/layer.py"),
        ("backend_selector", "vllm_omni/diffusion/attention/selector.py"),
        ("backend_dispatch", "vllm_omni/diffusion/attention/backends/abstract.py"),
        ("parallel_factory", "vllm_omni/diffusion/attention/parallel/factory.py"),
        ("local_input_output_handling", "vllm_omni/diffusion/attention/parallel/base.py"),
        ("backend", "vllm_omni/diffusion/attention/backends/sdpa.py"),
        ("integration", "vllm_omni/diffusion/attention/backends/cross_combined_integration_v1.py"),
        ("provider", "vllm_omni/diffusion/layers/lingbot_recovery.py"),
    )}


def verify_correctness(directory, matrix, sources):
    """No timing until all8 ranks have passed every same-build real case."""
    receipts = []
    for rank in range(8):
        path = Path(directory) / f"rank{rank}.json"
        receipt = json.loads(path.read_text())
        need(receipt.get("schema") == SCHEMA and receipt.get("mode") == "correctness"
             and receipt.get("passed") is True and receipt.get("rank") == rank,
             "successful all8 same-rank correctness receipts required")
        need(receipt.get("matrix_sha256") == matrix["sha256"]
             and receipt.get("manifest") == matrix["manifest"] and receipt.get("sources") == sources,
             "correctness fixture/build/source pins differ")
        need(receipt.get("full_attention_forward") is True
             and receipt.get("constructor_context") == "isolated_from_captured_config"
             and receipt.get("all40_payloads") is False,
             "correctness wrapper/scope differs")
        entries = receipt.get("cases", [])
        validate_order([entry["identity"] for entry in entries])
        for entry, wanted in zip(entries, matrix["ranks"][rank]["cases"], strict=True):
            need(entry.get("identity") == wanted["identity"]
                 and entry.get("fixture") == wanted["fixture"] and entry.get("passed") is True
                 and entry.get("checks") == list(CHECKS)
                 and entry.get("full_owner_sha256_verified") is True,
                 "missing real-case correctness evidence")
        receipts.append(pin(path))
    return receipts


def construct_wrappers(stage, provider):
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
    from vllm_omni.diffusion.layers.lingbot_recovery import Cross, CrossCombined
    wrappers, descriptions = {}, {}
    for arm, expected in (("plain", Cross), ("combined", CrossCombined)):
        values = copy.deepcopy(stage["model_config"])
        values["lingbot_npu_recovery_cross"] = arm == "plain"
        values["lingbot_npu_recovery_cross_combined"] = arm == "combined"
        config = SimpleNamespace(model_class_name=stage["model_class_name"], enforce_eager=True,
                                 model_config=values, diffusion_attention_config=None,
                                 parallel_config=SimpleNamespace(**stage["parallel_config"]))
        with set_current_diffusion_config(config):
            module = Attention(num_heads=40, head_size=128, num_kv_heads=40,
                               softmax_scale=128 ** -.5, causal=False, role="cross",
                               qkv_layout="BSND", prefix="blocks.0.cross_attn.attn",
                               skip_sequence_parallel=True, disable_kv_quant=True)
        module.eval()
        need(module.attn_backend.get_name() == "SDPA" and type(module.attention) is SDPAImpl,
             "normal constructor did not select the captured SDPA integration")
        adapter = module.attention._lingbot_combined_cross
        need(type(adapter) is expected and module.skip_sequence_parallel and not module.use_ring,
             "actual configured adapter/parallel route differs")
        wrappers[arm] = module
        descriptions[arm] = dict(attention_class=type(module).__module__ + "." + type(module).__name__,
                                 backend_class=type(module.attention).__name__,
                                 adapter_class=type(adapter).__name__,
                                 operation=OPERATIONS[arm], qkv_layout="BSND",
                                 scale=128 ** -.5, model_config=values)
    return wrappers, descriptions


def described(provider, tensor):
    result = dict(zip(contract.FIELDS, provider.describe(tensor)))
    for key in ("base_sizes", "base_strides", "physical_sizes", "shape", "stride"):
        result[key] = list(result[key])
    return result


def equal(torch, left, right, reason):
    torch.npu.synchronize()
    lc, rc = left.detach().cpu().contiguous(), right.detach().cpu().contiguous()
    need(lc.dtype == rc.dtype and tuple(lc.shape) == tuple(rc.shape)
         and torch.equal(lc.view(torch.uint8), rc.view(torch.uint8)), reason)


def unchanged(torch, provider, materialized):
    for section in ("inputs", "outputs"):
        for oid, owner in materialized["owners"][section].items():
            equal(torch, owner, materialized["host_owners"][section][oid], f"{section}/{oid} full owner changed")
        for name, tensor in materialized[section].items():
            need(described(provider, tensor) == materialized["live_descriptors"][section][name],
                 f"{section}/{name} descriptor changed")


def dispatch_checked(provider, module, arm, operands):
    """Correctness-only dispatch spy; always restore the provider before timing."""
    need("call" not in provider.__dict__, "unexpected provider override before dispatch audit")
    native_call, records = provider.call, []

    def spy(name, *args):
        value = native_call(name, *args)
        records.append((name, args, value))
        return value

    provider.call = spy
    try:
        output = module(*operands)
    finally:
        del provider.call
    need(len(records) == 1 and records[0][0] == OPERATIONS[arm], "wrong/refused actual cross dispatch")
    args, dispatched = records[0][1:]
    need(all(actual is wanted for actual, wanted in zip(args[:3], operands, strict=True))
         and args[3] == 128 ** -.5 and output is dispatched and output is not None,
         "full caller changed operands or bypassed expected FIA output")
    if arm == "combined":
        need(args[4] is True, "combined route was not admitted")
    return output


def correctness_case(torch, provider, wrappers, materialized, state):
    operands = tuple(materialized["inputs"][name] for name in ("query", "key", "value"))
    reference = materialized["outputs"]["native_output"]
    descriptor = materialized["metadata"]["outputs"]["descriptors"]["native_output"]
    forbidden = {owner.data_ptr() for owners in materialized["owners"].values() for owner in owners.values()}
    outputs = state["live_outputs"] = []
    unchanged(torch, provider, materialized)
    for arm in ARMS:
        for repeat in range(2):
            output = dispatch_checked(provider, wrappers[arm], arm, operands)
            outputs.append(output)  # Keep every output live until all comparisons/consumers finish.
            torch.npu.synchronize()
            actual = described(provider, output)
            need(actual["storage_data_ptr"] not in forbidden, "output owner reused while a consumer is live")
            forbidden.add(actual["storage_data_ptr"])
            need(all(actual[field] == descriptor[field] for field in contract.FIELDS[:9]),
                 "native output physical contract differs")
            need(bool(torch.isfinite(output.detach().cpu()).all()), "nonfinite output")
            equal(torch, output, reference, f"{arm}/{repeat} differs from captured native BSND")
            base = output.as_strided(descriptor["base_sizes"], descriptor["base_strides"], 0)
            oid = materialized["metadata"]["outputs"]["view_owner_ids"]["native_output"]
            equal(torch, base, materialized["host_owners"]["outputs"][oid], "full output owner differs")
            unchanged(torch, provider, materialized)
    equal(torch, outputs[0], outputs[2], "plain and combined differ")
    torch.npu.synchronize()
    state["live_outputs"] = []
    return dict(checks=list(CHECKS), source_full_owner_verified=True, full_output_owner_verified=True,
                destination_canary_verified=False, actual_operations=OPERATIONS,
                passed=True, full_owner_sha256_verified=True)


def timing_case(torch, provider, wrappers, materialized, state, round_number):
    need("call" not in provider.__dict__, "correctness spy leaked into timing")
    operands = tuple(materialized["inputs"][name] for name in ("query", "key", "value"))
    result = {}
    order = ARMS if round_number != 2 else tuple(reversed(ARMS))
    for arm in order:
        module = wrappers[arm]
        phase_start = time.perf_counter_ns()
        for _ in range(20):
            state["last_output"] = module(*operands)
            torch.npu.synchronize()
        walls, events, windows = [], [], []
        for _ in range(30):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            state["last_output"] = module(*operands)
            torch.npu.synchronize()
            end = time.perf_counter_ns()
            walls.append((end - start) / 1e6)
            windows.append([start, end])
        # Separate device-event pass; no event creation/recording in wall samples.
        begin, end_event = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        for _ in range(30):
            torch.npu.synchronize()
            begin.record()
            state["last_output"] = module(*operands)
            end_event.record()
            end_event.synchronize()
            events.append(float(begin.elapsed_time(end_event)))
        result[arm] = dict(warmup=20, wall_ms=walls, device_ms=events,
                           wall_mean_ms=statistics.mean(walls), device_mean_ms=statistics.mean(events),
                           wall_windows_ns=windows, phase_start_ns=phase_start,
                           phase_end_ns=time.perf_counter_ns())
    torch.npu.synchronize()
    state.pop("last_output", None)
    return dict(arms=result, order=list(order), spy_enabled=False,
                full_owner_sha256_verified=True, numerical_checks_in_timing=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-plan", type=Path, required=True)
    parser.add_argument("--matrix-audit", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "correctness", "timing"), default="preflight")
    parser.add_argument("--rank", type=int, choices=range(8), default=0)
    parser.add_argument("--device", type=int, choices=range(8), default=0)
    parser.add_argument("--visible-devices", default="8,9,10,11,12,13,14,15")
    parser.add_argument("--round", dest="round_number", type=int, choices=(1, 2, 3))
    parser.add_argument("--correctness-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), "refusing to overwrite an existing experiment receipt")
    matrix, stage = preflight(args.capture_plan, args.matrix_audit)
    sources = source_pins()
    result = dict(schema=SCHEMA, mode=args.mode, rank=args.rank, passed=False,
                  matrix_sha256=matrix["sha256"], matrix=matrix, manifest=matrix["manifest"], sources=sources,
                  full_attention_forward=True, constructor_context="isolated_from_captured_config",
                  full_model_startup=False, original_hccl_included=False, all40_payloads=False,
                  real_payload_layers=[0], actual192_global_proven=False,
                  model_admitted=False, performance_validated=False, end_to_end_performance=False,
                  destination_canary_verified=False, gate_passed=False,
                  full_call_scope="Attention.forward: native layout handling, admission, FIA allocation/dispatch and output handling",
                  timing_excludes=["fixture loading/materialization", "constructor/first-call initialization",
                                  "Q/K projection", "self/cross output projection", "HCCL", "model scheduling"],
                  timing_caveats=["per-call synchronization differs from model submission",
                                  "wall and device events use separate sample batches",
                                  "no cross-rank phase barrier; do not assume ranks sampled simultaneously",
                                  "three independently launched rounds and all8 ranks are required for screening"],
                  host_clock="time.perf_counter_ns, same-host monotonic clock",
                  boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(), cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    try:
        if args.mode == "preflight":
            result["metadata_preflight_passed"] = True
            result["device_executed"] = False
            result["passed"] = True
            return
        need(os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == args.visible_devices,
             "explicit owned visible-device set differs")
        need(len(set(args.visible_devices.split(","))) == 8 and args.device == args.rank,
             "rank must map to its matching logical device in the owned eight-die set")
        need(os.environ.get("ROOT_LINGBOT_SP8_EXPERIMENT_MODE") == args.mode,
             "launch-owned experiment mode differs")
        need(os.environ.get("LINGBOT_REAL_FIXTURE_CAPTURE", "0") == "0"
             and not os.environ.get("LINGBOT_OUTPUT_CAPTURE_DIR")
             and not os.environ.get("LINGBOT_SP8_OBSERVE_MODE"), "capture/observer must be disabled for replay")
        need(os.environ.get("LINGBOT_RECOVERY_MANIFEST") == matrix["manifest"]["path"]
             and os.environ.get("LINGBOT_RECOVERY_MANIFEST_SHA256") == matrix["manifest"]["sha256"],
             "owned launcher must pin the same recovery manifest")
        if args.mode == "timing":
            need(args.round_number in (1, 2, 3) and args.correctness_dir is not None,
                 "timing requires round1..3 and all8 correctness receipts")
            result["correctness_receipts"] = verify_correctness(args.correctness_dir, matrix, sources)
            result["round"] = args.round_number
        import torch
        import torch_npu
        torch.npu.set_device(args.device)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = True
        from vllm_omni.diffusion.layers import lingbot_recovery
        provider = state["provider"] = lingbot_recovery.load()
        need(provider.fatal.ensure_protocol(), "launch-owned fatal channel required")
        result["device_executed"] = True
        result["device"] = args.device
        result["visible_devices"] = args.visible_devices
        result["device_name"] = torch.npu.get_device_name(args.device)
        result["physical_package_mapping_verified"] = False
        with torch.inference_mode():
            wrappers, descriptions = construct_wrappers(stage, provider)
            state["wrappers"] = wrappers
            result["wrappers"] = descriptions
            for row in matrix["ranks"][args.rank]["cases"]:
                need(pin(row["fixture"]["path"]) == row["fixture"], "fixture metadata changed after preflight")
                loaded = state["loaded"] = load_case(row["fixture"]["path"])
                materialized = state["materialized"] = loaded.materialize(
                    torch, f"npu:{args.device}", describe=provider.describe,
                    allocate_owner=lambda shape, dtype, device, fmt: torch_npu.empty_with_format(
                        tuple(shape), dtype=dtype, device=device, acl_format=fmt))
                torch.npu.synchronize()
                materialized["live_descriptors"] = {
                    section: {name: described(provider, value) for name, value in materialized[section].items()}
                    for section in ("inputs", "outputs")}
                if args.mode == "correctness":
                    outcome = correctness_case(torch, provider, wrappers, materialized, state)
                else:
                    # Reconfirm actual dispatch once per arm outside the timed region.
                    operands = tuple(materialized["inputs"][key] for key in ("query", "key", "value"))
                    for arm in ARMS:
                        state["dispatch_output"] = dispatch_checked(provider, wrappers[arm], arm, operands)
                        torch.npu.synchronize()
                    state.pop("dispatch_output", None)
                    outcome = timing_case(torch, provider, wrappers, materialized, state, args.round_number)
                result["cases"].append(dict(**row, **outcome))
                torch.npu.synchronize()
                state.pop("materialized"); state.pop("loaded")
                del materialized, loaded
        need(source_pins() == sources, "replay source changed during execution")
        result["actual_cross_cases_completed"] = 24
        result["passed"] = True
    except BaseException as error:
        result["error"] = repr(error)
        provider = state.get("provider")
        if provider is not None:
            provider.poisoned = True
            _QUARANTINE.append((state, error))
            provider.fatal.report_failure("cross_real_matrix", repr(error), (state, error))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: result[key] for key in ("mode", "rank", "passed")}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
