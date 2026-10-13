"""Prepare NOT-ADMITTED paired80 configs; never imports torch or launches work.

The new record-once QK candidate must independently pass its actual192
correctness and three-round complete-call screen before these plans may run.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

FLAGS = (
    "lingbot_npu_recovery_qk", "lingbot_npu_recovery_qk_combined",
    "lingbot_npu_recovery_cross", "lingbot_npu_recovery_cross_combined",
)
REFERENCE = dict(zip(FLAGS, (True, False, True, False)))
CANDIDATE = dict(zip(FLAGS, (False, True, False, True)))
NOTICE = ("# PREPARED ONLY — NOT ADMITTED. No service has been launched.\n"
          "# Requires the NEW record-once QK actual192 correctness + three-round screen.\n"
          "# Bind fresh source-pinned manifests and VLLM_LINGBOT_NPU_ASYNC_METADATA=0.\n")


def need(value, message):
    if not value:
        raise RuntimeError("Paired80 preparation: " + message)


def pin(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        found = []
        for key in sorted(set(left) | set(right)):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                found.append(name)
            else:
                found.extend(differences(left[key], right[key], name))
        return found
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [name for index, (a, b) in enumerate(zip(left, right))
                for name in differences(a, b, f"{prefix}[{index}]")]
    return [] if left == right and type(left) is type(right) else [prefix]


def build(source):
    need(source["dtype"] == "bfloat16" and len(source["stages"]) == 1,
         "captured BF16 single-stage source required")
    stage = source["stages"][0]
    need(stage["model_class_name"] == "LingBotWorldCausalDMDPipeline"
         and stage["enforce_eager"] is True and stage["devices"] == "0,1,2,3,4,5,6,7",
         "captured eager LingBot8 configuration differs")
    p = stage["parallel_config"]
    need(tuple(p[key] for key in ("tensor_parallel_size", "sequence_parallel_size", "ulysses_degree", "ring_degree"))
         == (1, 8, 8, 1), "captured SP8/TP1/ring1 required")
    v = stage["model_config"]
    need(all(v.get(key) is value for key, value in REFERENCE.items()), "source is not the plain reference")
    need((v["ar_diffusion_width"], v["ar_diffusion_height"], stage["default_sampling_params"]["num_inference_steps"])
         == (832, 480, 4), "captured resolution/DMD changed")
    need(v["lingbot_npu_fixed_timestep_bank"] is True and v["lingbot_npu_recovery_vae_pack"] is True
         and v["lingbot_npu_quant"] == "w8a8" and v["lingbot_npu_quant_scope"] == "ffn"
         and v["lingbot_npu_quant_weight_format"] == "nz"
         and v["lingbot_npu_pixel_output_mode"] == "root_uint8_p2p"
         and v["lingbot_npu_pixel_device_interleave"] is True,
         "source lost StageA/FFN/VAE/pixel settings")
    need(v["ar_diffusion_kv_config"]["reuse_history_staging"] is True
         and v["ar_diffusion_kv_config"]["npu_history_staging_layers"] == list(range(40)),
         "full-history staging settings differ")
    need(v["lingbot_npu_qk_single_host_admission_v1"] is False
         and v["lingbot_npu_cross_combined_guard_cpp_v1"] is False,
         "lost historical providers must remain disabled")
    reference, candidate = copy.deepcopy(source), copy.deepcopy(source)
    reference["stages"][0]["model_config"].update(REFERENCE)
    candidate["stages"][0]["model_config"].update(CANDIDATE)
    need(reference == source, "reference changed more than explicitly specified flags")
    expected = sorted(f"stages[0].model_config.{key}" for key in FLAGS)
    changed = differences(reference, candidate)
    need(changed == expected, "candidate changed fields outside the four requested flags")
    return reference, candidate, changed


def write_same_or_new(path, text):
    if path.exists():
        need(path.read_text() == text, f"refusing to overwrite different preparation: {path}")
    else:
        path.write_text(text)


def main():
    worktree = Path(__file__).resolve().parents[1]
    root = worktree.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-plan", type=Path,
                        default=root / "audits/capture-fixtures/launch-v1/full/plan.frozen-before-request.json")
    parser.add_argument("--output-dir", type=Path, default=worktree / "recovery/qk-record-once-paired80-prepared")
    args = parser.parse_args()
    capture = json.loads(args.capture_plan.read_text())
    need(capture["schema"] == "lingbot_real_fixture_launch_plan_v1" and capture["phase"] == "full",
         "full capture plan required")
    source_pin = pin(capture["config"])
    need(source_pin["sha256"] == capture["config_sha256"], "frozen source config changed")
    reference, candidate, changed = build(yaml.safe_load(Path(capture["config"]).read_text()))
    destination = args.output_dir.resolve()
    need(destination.is_relative_to(worktree / "recovery"), "prepare only within this worktree/recovery")
    destination.mkdir(parents=True, exist_ok=True)
    configs = {}
    for name, values in (("reference", reference), ("candidate", candidate)):
        path = destination / f"{name}.yaml"
        write_same_or_new(path, NOTICE + yaml.safe_dump(values, sort_keys=False))
        configs[name] = pin(path)
    plan = dict(schema="lingbot_qk_record_once_paired80_preparation_v1", prepared_only=True,
                admitted=False, ready_to_launch=False, service_started=False, operator_gates_passed=False,
                model_admitted=False, performance_validated=False,
                source_capture_plan=pin(args.capture_plan), source_config=source_pin,
                generator=pin(__file__), worktree=str(worktree), configs=configs,
                only_config_differences=changed, reference_flags=REFERENCE, candidate_flags=CANDIDATE,
                model=str(root / "models/lingbot-world"), model_download_required=False,
                model_identity_caveat="Use retained current model/integrity receipt; original historical full checkpoint identity remains unproven.",
                new_host_manifest=None, new_qk_manifest=None,
                prerequisites=[
                    "New record-once QK source-pinned actual192 correctness/HCCL/full-owner/canary and finite gates pass.",
                    "New record-once QK three independent complete-call rounds pass the pre-frozen mean and worst-case gates; old failed QK receipts do not admit this route.",
                    "Bind fresh host/QK manifests to this worktree; provider_manifest relationship and actual loaded module/library paths must match.",
                    "Cross combined actual192 correctness/full-call screen passed previously; verify unchanged arithmetic/runtime or repeat required checks for the new pinned build.",
                    "Root owns and verifies all8 available dies; prepare fresh separate service and capture directories for both arms.",
                ],
                source_environment=str(worktree / "recovery/env.sh"),
                pythonpath_after_source=dict(prepend=str(worktree),
                    reason="The copied env.sh prepends the old main worktree; the isolated worktree must take precedence.",
                    verify_actual_import_paths=True),
                environment_after_source={
                    "ASCEND_RT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15",
                    "VLLM_LINGBOT_NPU_ASYNC_METADATA": "0",
                    "VLLM_LINGBOT_NPU_HISTORY_STAGING": "1",
                    "VLLM_LINGBOT_NPU_PLANAR_RGB": "1", "VLLM_LINGBOT_NPU_PINNED_RGB": "0",
                    "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                    "LINGBOT_REAL_FIXTURE_CAPTURE": "0", "LINGBOT_CAPTURE_RANK": "-1",
                    "LINGBOT_CAPTURE_ACTIVATIONS": "0",
                },
                per_arm_capture_environment="LINGBOT_CAPTURE_DIR must be a new dedicated persistent path for that arm.",
                unset_environment=["LINGBOT_OUTPUT_CAPTURE_DIR", "LINGBOT_REAL_FIXTURE_DIR",
                                   "LINGBOT_SP8_OBSERVE_MODE", "LINGBOT_COMPUTE_CAPTURE_DIR"],
                no_extra_instrumentation="Disable any other profiler/activation/fixture dump not required by paired80.",
                runtime_import_gate=dict(must_resolve_under=str(worktree), modules=[
                    "vllm_omni.diffusion.layers.lingbot_recovery",
                    "vllm_omni.diffusion.layers.lingbot_qk_combined",
                    "vllm_omni.diffusion.attention.layer"]),
                supervisor=dict(mode="correctness", recv=True, host_manifest="REQUIRED_NEW_PINNED_MANIFEST"),
                candidate_manifest_environment=["LINGBOT_QK_COMBINED_MANIFEST", "LINGBOT_QK_COMBINED_MANIFEST_SHA256"],
                request_protocol=dict(chunks=80, requests_per_arm=1, warmup_requests=0,
                                      expected_latent_chunks_per_rank=80, expected_root_rgb_frames=957,
                                      other_rank_rgb_frames=0, comparison="bitwise latent and pre-encode RGB"),
                pair_scope="Reference plain QK/plain cross versus candidate new QK combined/cross combined; this pair alone cannot attribute separate QK and cross speedups.")
    write_same_or_new(destination / "preparation.json", json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(prepared_only=True, admitted=False, service_started=False,
                         preparation=str(destination / "preparation.json"), config_differences=changed), indent=2))


if __name__ == "__main__":
    main()
