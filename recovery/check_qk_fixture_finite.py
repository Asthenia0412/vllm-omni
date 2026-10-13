"""CPU-only BF16 logical-view finite proof, bound to actual192 bitwise replay.

No torch import, NPU call, padding-finite claim, or performance admission. Native
reference finite plus device-proven bitwise candidate equality implies candidate
finite; this is explicitly not a candidate-side isfinite kernel measurement.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import check_qk_real_matrix as replay
from load_real_fixtures import load_case


def logical_nonfinite(raw, descriptor):
    replay.need(descriptor["dtype"] == "torch.bfloat16" and descriptor["element_bytes"] == 2,
                "this finite proof admits only BF16 logical views")
    view = np.ndarray(tuple(descriptor["shape"]), dtype="<u2", buffer=raw,
        offset=descriptor["storage_offset"] * 2,
        strides=tuple(stride * 2 for stride in descriptor["stride"]))
    # BF16 exponent is bits14..7. All ones denotes either infinity or NaN.
    count = int(np.count_nonzero(np.bitwise_and(view, 0x7f80) == 0x7f80))
    return dict(logical_elements=int(view.size), nonfinite_elements=count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--correctness-dir", type=Path, required=True)
    parser.add_argument("--supervisor-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay.need(not args.output.exists(), "refuse to overwrite finite receipt")
    # Verify logical-view treatment with nonfinite padding and strided selection.
    raw = np.array([0x7f80, 0, 0x7fc1, 0x3f80, 0xff80], dtype="<u2").tobytes()
    descriptor = dict(dtype="torch.bfloat16", element_bytes=2, shape=[2], stride=[2], storage_offset=1)
    replay.need(logical_nonfinite(raw, descriptor) == dict(logical_elements=2, nonfinite_elements=0),
                "finite guard incorrectly included padding")
    descriptor.update(shape=[5], stride=[1], storage_offset=0)
    replay.need(logical_nonfinite(raw, descriptor)["nonfinite_elements"] == 3,
                "finite guard missed NaN/infinity")

    matrix = replay.preflight(args.fixtures)
    first = json.loads((args.correctness_dir / "rank0.json").read_text())
    correctness = replay.verify_correctness(args.correctness_dir, matrix, first["manifest"], replay.source_pins())
    supervisor = json.loads(args.supervisor_result.read_text())
    replay.need(supervisor["exit_code"] == 0 and supervisor["fatal"] is None
                and supervisor["remaining_owned_processes"] == [] and supervisor["mode"] == "correctness",
                "clean owned correctness supervisor required")
    command = supervisor["command"]
    replay.need(command[command.index("--output") + 1] == str(args.correctness_dir)
                and command[command.index("--fixtures") + 1] == matrix["root"]
                and command[command.index("--manifest") + 1] == first["manifest"]["path"],
                "supervisor receipt belongs to a different replay")
    rows = []
    for rank in matrix["ranks"]:
        for item in rank["cases"]:
            replay.need(replay.pin(item["file"]["path"]) == item["file"], "fixture metadata changed")
            loaded = load_case(item["file"]["path"])
            values = {}
            for section in ("inputs", "outputs"):
                snapshot = loaded.metadata[section]
                for name, desc in snapshot["descriptors"].items():
                    oid = snapshot["view_owner_ids"][name]
                    checked = logical_nonfinite(loaded.owner_bytes[(section, oid)], desc)
                    replay.need(checked["nonfinite_elements"] == 0, f"nonfinite {section}/{name}: {item['file']['path']}")
                    values[f"{section}/{name}"] = dict(**checked, owner_sha256=snapshot["owners"][oid]["raw_sha256"])
            rows.append(dict(identity=item["identity"], fixture=item["file"], views=values))
            del loaded
    result = dict(schema="lingbot_qk_real192_finite_implication_v1", passed=True,
        actual_qk_payload_cases=192, real_payload_layers=[0], all40_payloads=False,
        matrix_sha256=matrix["sha256"], correctness_receipts=correctness,
        supervisor_result=replay.pin(args.supervisor_result), supervisor_summary=supervisor,
        sources=dict(script=replay.pin(__file__), loader=replay.pin(Path(__file__).with_name("load_real_fixtures.py")),
                     replay_harness=first["sources"]["harness"]), numpy_version=np.__version__,
        cases=rows, full_owner_sha256_verified=True, all_logical_inputs_finite=True,
        captured_native_send_finite=True, candidate_finite_proven_by_bitwise_implication=True,
        candidate_isfinite_kernel_executed=False, padding_finite_claim=False,
        device_executed_by_this_script=False, performance_validated=False, model_admitted=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), passed=True, actual_cases=len(rows),
                         sha256=replay.pin(args.output)["sha256"])), flush=True)


if __name__ == "__main__":
    main()
