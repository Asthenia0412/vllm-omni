"""Independent correctness-run capture of actual QKV/FFN inputs and owners."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def install(transformer):
    import torch
    import torch.distributed as dist
    import torch_npu

    root = Path(os.environ["LINGBOT_REGION_CAPTURE_DIR"]).resolve()
    if not str(root).startswith("/tmp/lingbot-generated-"):
        raise ValueError("Region payloads require an explicit temporary non-model spool")
    rank = dist.get_rank()
    folder = root/f"rank{rank}"
    folder.mkdir(parents=True, exist_ok=True)
    describe = torch.ops.lingbot_storage_descriptor.describe
    fields = ("npu_format", "origin_format", "base_sizes", "base_strides", "physical_sizes",
              "shape", "stride", "storage_offset", "storage_bytes")
    limit = int(os.environ.get("LINGBOT_REGION_CAPTURE_CALLS", "24"))
    hooks = []
    block = transformer.blocks[0]
    for region, module in (("qkv", block.self_attn.qkv), ("ffn", block.ffn)):
        state = {"call": 0}
        params = {}
        for name, value in module.named_parameters():
            params[name] = value.detach().cpu()
        if region == "ffn":
            for index in (0, 2):
                params[f"{index}.scale"] = module[index]._lingbot_quant_weight_scale.detach().cpu()
        weight_file = folder/f"{region}_weights.pt"
        torch.save(params, weight_file)
        descriptions = {name: dict(zip(fields, describe(value))) for name, value in module.named_parameters()}
        if region == "ffn":
            for index in (0, 2):
                descriptions[f"{index}.scale"] = dict(zip(fields, describe(module[index]._lingbot_quant_weight_scale)))

        def after(mod, args, output, *, region=region, state=state, descriptions=descriptions,
                  weight_file=weight_file):
            index = state["call"]
            state["call"] += 1
            if index >= limit:
                return
            value = args[0]
            result = output[0] if region == "qkv" else output
            payload = folder/f"{region}_{index:03d}.pt"
            torch.save({"input": value.detach().cpu(), "output": result.detach().cpu()}, payload)
            metadata = {"rank": rank, "layer": 0, "region": region, "call": index,
                        "chunk": index//4, "probe": index%4, "timing_eligible": False,
                        "payload": str(payload), "weights": str(weight_file),
                        "input": dict(zip(fields, describe(value))), "parameters": descriptions,
                        "allow_internal_format": str(torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")),
                        "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
            (folder/f"{region}_{index:03d}.json").write_text(json.dumps(metadata, indent=2)+"\n")

        hooks.append(module.register_forward_hook(after))
    transformer._lingbot_region_capture_hooks = hooks
