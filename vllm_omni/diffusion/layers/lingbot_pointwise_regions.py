"""Opt-in larger tensor-only regions; no communication or request state writes."""
from __future__ import annotations

import os


def compute(module, region, *inputs):
    if region == "post_attention":
        hidden, attention, gate, camera_scale, camera_shift = inputs
        frames = gate.shape[1]
        grid = hidden.unflatten(1, (frames, hidden.shape[1] // frames))
        attention_grid = attention.unflatten(1, (frames, hidden.shape[1] // frames))
        hidden = (grid + attention_grid * gate).flatten(1, 2).to(hidden.dtype)
        hidden = ((1 + camera_scale) * hidden + camera_shift).to(hidden.dtype)
        return hidden, module.norm3(hidden)
    if region == "ffn_tail":
        hidden, scale, shift, gate = inputs
        frames = scale.shape[1]
        grid = hidden.unflatten(1, (frames, hidden.shape[1] // frames))
        normalized = module.norm2(hidden.float()).unflatten(1, (frames, hidden.shape[1] // frames))
        if os.environ.get("VLLM_LINGBOT_NPU_FUSED_POINTWISE", "0") == "1":
            import torch

            normalized = torch.addcmul(shift, normalized, 1 + scale)
        else:
            normalized = normalized * (1 + scale) + shift
        normalized = normalized.flatten(1, 2).to(hidden.dtype)
        output = module.ffn(normalized).unflatten(1, (frames, hidden.shape[1] // frames))
        return ((grid + output * gate).flatten(1, 2).to(hidden.dtype),)
    raise ValueError("Unknown pointwise graph region")


def dispatch(module, region, *inputs):
    state = getattr(module, "_lingbot_pointwise_regions", None)
    if state is None or region not in state:
        return None
    item = state[region]
    if item["policy"] != (os.environ.get("VLLM_LINGBOT_NPU_FUSED_POINTWISE", "0"),
                          os.environ.get("VLLM_LINGBOT_NPU_LOSSY_POINTWISE", "0")):
        raise RuntimeError("Pointwise graph execution policy changed; reload all workers")
    hidden = inputs[0]
    # Ordinary unsupported input falls back before capture or execution.
    if hidden.device.type != "npu" or tuple(hidden.shape) != (1, 585, 5120):
        return None
    import torch

    if hidden.dtype != torch.bfloat16 or any(x.device != hidden.device for x in inputs):
        return None
    parameters = tuple(p for owner in item["owners"] for p in (*owner.parameters(), *owner.buffers()))
    identities = tuple((id(p), p.data_ptr(), p._version) for p in parameters)
    if identities != item["identities"]:
        raise RuntimeError("Pointwise graph owners changed; reload all workers")
    try:
        if item["runner"] is None:
            result = compute(module, region, *inputs)
        else:
            result = item["runner"].run(region, inputs, (region, identities),
                                         lambda *args: compute(module, region, *args))
        if item["capture"] is not None:
            item["capture"](inputs, result)
        return result
    except BaseException as error:
        from .lingbot_sp8_fatal import report_failure

        report_failure("pointwise_region", str(error), (module, inputs, item, parameters))
        raise


def install(transformer, regions=()):
    from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

    capture_root = os.environ.get("LINGBOT_POINTWISE_CAPTURE_DIR")
    allowed = ("post_attention", "ffn_tail")
    if len(set(regions)) != len(regions) or any(r not in allowed for r in regions):
        raise ValueError("Pointwise graph regions must be unique post_attention/ffn_tail entries")
    if os.environ.get("VLLM_LINGBOT_NPU_LOSSY_POINTWISE", "0") != "0":
        raise ValueError("Pointwise graph candidate requires the existing exact camera arithmetic")
    for index, block in enumerate(transformer.blocks):
        selected = allowed if capture_root and index == 0 else regions
        if not selected:
            continue
        state = {}
        for region in selected:
            # Camera values are graph inputs; their producer weights are not
            # accessed here. Pin exactly the modules this region executes.
            owners = (block.norm3,) if region == "post_attention" else (block.norm2, block.ffn)
            parameters = tuple(p for owner in owners for p in (*owner.parameters(), *owner.buffers()))
            capture = _capture(block, region, capture_root) if capture_root and index == 0 else None
            runner = NPUExactGraphRunner(max_graphs=4, component_name=f"LingBot {index} {region}",
                                        disable_config_hint="lingbot_npu_pointwise_graph_regions=[]") if region in regions else None
            state[region] = dict(owners=owners, identities=tuple((id(p), p.data_ptr(), p._version) for p in parameters),
                                 runner=runner, capture=capture,
                                 policy=(os.environ.get("VLLM_LINGBOT_NPU_FUSED_POINTWISE", "0"),
                                         os.environ.get("VLLM_LINGBOT_NPU_LOSSY_POINTWISE", "0")))
        block._lingbot_pointwise_regions = state


def _capture(block, region, root):
    import hashlib
    import json
    from pathlib import Path

    import torch
    import torch.distributed as dist

    folder = Path(root).resolve() / f"rank{dist.get_rank()}"
    if not str(folder).startswith("/tmp/lingbot-generated-"):
        raise ValueError("Pointwise payload spool must be an explicit temporary non-model path")
    folder.mkdir(parents=True, exist_ok=True)
    describe = torch.ops.lingbot_storage_descriptor.describe
    fields = ("npu_format", "origin_format", "base_sizes", "base_strides", "physical_sizes",
              "shape", "stride", "storage_offset", "storage_bytes")
    norm = block.norm3 if region == "post_attention" else block.norm2
    parameters = {f"norm.{k}": v for k, v in norm.named_parameters()}
    if region == "ffn_tail":
        parameters.update({f"ffn.{k}": v for k, v in block.ffn.named_parameters()})
        for i in (0, 2):
            parameters[f"ffn.{i}.scale"] = block.ffn[i]._lingbot_quant_weight_scale
    path = folder / f"{region}_weights.pt"
    torch.save({k: v.detach().cpu() for k, v in parameters.items()}, path)
    descriptions = {k: dict(zip(fields, describe(v))) for k, v in parameters.items()}
    weight_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    call = 0

    def capture(inputs, outputs):
        nonlocal call
        index = call
        call += 1
        if index >= 24:
            return
        payload = folder / f"{region}_{index:03d}.pt"
        torch.save(dict(inputs=[v.detach().cpu() for v in inputs], outputs=[v.detach().cpu() for v in outputs]), payload)
        row = dict(rank=dist.get_rank(), layer=0, region=region, call=index, chunk=index // 4, probe=index % 4,
                   timing_eligible=False, payload=str(payload), payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
                   weights=str(path), weights_sha256=weight_sha, parameters=descriptions,
                   inputs=[dict(zip(fields, describe(v))) for v in inputs], eps=norm.eps)
        (folder / f"{region}_{index:03d}.json").write_text(json.dumps(row, indent=2) + "\n")
    return capture
