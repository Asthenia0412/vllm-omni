"""Independent combined-cross checks; synthetic inputs are not model admission.

Use --cpu-only for registered-op refusal checks without initializing the NPU.
Run the device mode serially under the experiment controller on each owned die.
"""
import argparse
import hashlib
import json
from pathlib import Path


def pin(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for row in [manifest["source"], manifest["library"], manifest["kernels"],
                *manifest["kernel_sources"], *manifest["runtime_sources"], *manifest["abi_headers"]]:
        assert pin(row["path"]) == row, f"Source/build pin changed: {row['path']}"
    import torch
    import torch_npu

    assert (torch.__version__, torch_npu.__version__) == (manifest["torch"], manifest["torch_npu"])
    torch.ops.load_library(manifest["library"]["path"])
    op = torch.ops.lingbot_recovery_v1.try_cross_combined
    results = []
    if args.cpu_only:
        assert not torch.npu.is_initialized()
        cpu = torch.zeros(1, dtype=torch.bfloat16)
        assert op(cpu, cpu, cpu, 128**-.5, True) is None
        results.append("grad_enabled_refuses_without_device")
        with torch.inference_mode():
            assert op(cpu, cpu, cpu, 128**-.5, False) is None
            results.append("incompatible_route_refuses_without_device")
            assert op(cpu, cpu, cpu, 128**-.5, True) is None
            results.append("cpu_input_refuses_without_device")
            try:
                op(cpu, cpu, cpu, float("nan"), True)
            except RuntimeError as error:
                assert "Invalid attention scale" in str(error)
            else:
                raise AssertionError("Invalid scale was accepted")
            results.append("invalid_scale_raises_before_device")
        assert not torch.npu.is_initialized()
    else:
        torch.npu.set_device(args.device)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = True
        device = f"npu:{args.device}"
        torch.manual_seed(1327)

        def equal(a, b, label):
            torch.npu.synchronize()
            ac, bc = a.cpu().contiguous(), b.cpu().contiguous()
            assert torch.isfinite(ac).all() and torch.isfinite(bc).all(), label
            assert torch.equal(ac.view(torch.int16), bc.view(torch.int16)), label
            results.append(label)

        with torch.inference_mode():
            # Offset views inside complete owners cover lifetime/address/extent,
            # while native/combined/public calls all consume the same payload.
            owners, tensors = [], []
            for size in (585, 512, 512):
                count = size * 40 * 128
                owner = torch.full((count + 512,), -7, dtype=torch.bfloat16, device=device)
                tensor = owner[256:-256].view(1, size, 40, 128)
                tensor.copy_(torch.randn_like(tensor))
                owners.append(owner)
                tensors.append(tensor)
            before = [owner.clone() for owner in owners]
            q, k, v = tensors
            public = torch_npu.npu_fused_infer_attention_score(q, k, v,
                num_heads=40, num_key_value_heads=40, input_layout="BSND", scale=128**-.5,
                pre_tokens=2147483647, next_tokens=2147483647, sparse_mode=0,
                inner_precise=0, block_size=0, antiquant_mode=0, key_antiquant_mode=0,
                value_antiquant_mode=0, softmax_lse_flag=False)[0]
            legacy = torch.ops.lingbot_recovery_v1.cross(q, k, v, 128**-.5)
            combined = op(q, k, v, 128**-.5, True)
            assert combined is not None, "Supported H40 BSND unexpectedly refused"
            equal(combined, public, "combined_vs_public_bsnd")
            equal(combined, legacy, "combined_vs_existing_cpp_bsnd")
            repeated = op(q, k, v, 128**-.5, True)
            assert repeated.data_ptr() != combined.data_ptr(), "Output allocation reused while live"
            equal(repeated, combined, "repeat_fresh_output")
            assert op(q, k, v, 128**-.5, False) is None
            results.append("incompatible_route_prestart_refusal")
            assert op(q[..., ::2], k, v, 128**-.5, True) is None
            results.append("noncontiguous_input_prestart_refusal")
            assert op(q, k[:, :511], v[:, :511], 128**-.5, True) is None
            results.append("nonhistorical_shape_prestart_refusal")
            assert op(q[:, :0], k, v, 128**-.5, True) is None
            results.append("empty_input_prestart_refusal")
            with torch.enable_grad():
                assert op(q, k, v, 128**-.5, True) is None
            results.append("grad_mode_prestart_refusal")
            stream = torch.npu.Stream(device=args.device)
            with torch.npu.stream(stream):
                assert op(q, k, v, 128**-.5, True) is None
            results.append("nondefault_stream_prestart_refusal")
            for index, (owner, initial) in enumerate(zip(owners, before, strict=True)):
                equal(owner, initial, f"source_full_owner_and_canary_{index}")
    receipt = dict(passed=results, cpu_only=args.cpu_only,
                   device=None if args.cpu_only else args.device,
                   manifest=pin(args.manifest), actual_operator=str(op),
                   scope="registered CPU refusals" if args.cpu_only else "synthetic H40 BSND calls",
                   historical_exact_options_proven=False, real_model_inputs=False,
                   model_validated=False, performance_validated=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
