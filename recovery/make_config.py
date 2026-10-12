"""Materialize portable native rollback or explicitly reconstructed candidate."""
import argparse
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    value = yaml.safe_load(Path(__file__).with_name("historical_candidate.yaml").read_text())
    config = value["stages"][0]["model_config"]
    for name in ("lingbot_npu_qk_single_host_admission_v1", "lingbot_npu_cross_combined_guard_cpp_v1",
                 "lingbot_npu_vae_exact_pack"):
        config[name] = False
    config.pop("lingbot_npu_vae_exact_pack_library", None)
    config.pop("lingbot_npu_vae_exact_pack_manifest", None)
    for feature in ("qk", "cross", "vae_pack", "descriptor"):
        config["lingbot_npu_recovery_" + feature] = args.candidate
    config["lingbot_npu_fixed_timestep_bank"] = args.candidate
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False))


if __name__ == "__main__":
    main()
