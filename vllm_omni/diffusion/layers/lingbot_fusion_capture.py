# SPDX-License-Identifier: Apache-2.0
"""Bounded actual-operand captures sharing the diagnostic writer's gates.

Disabled captures perform no copies, file writes, or NPU calls. Enabled sessions
are diagnostic-only, never valid performance samples. First block/session only.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_first_session: str | None = None
_saved: set[tuple[str, int, int, str]] = set()


def capture_fusion_operands(module, kind: str, tensors: dict[str, torch.Tensor], **metadata) -> None:
    if os.environ.get("LINGBOT_CAPTURE_ACTIVATIONS", "0") != "1":
        return
    if not getattr(module, "_lingbot_fusion_capture_first_block", False):
        return
    from . import lingbot_capture

    writer = lingbot_capture._active_capture
    if writer is None or not writer.capture_activations:
        return
    if not 0 <= writer.chunk < min(writer.activation_limit, 6) or writer.probe not in range(4):
        return
    global _first_session
    if _first_session is None:
        _first_session = writer.session_id
    if writer.session_id != _first_session:
        return
    signature = (writer.session_id, writer.chunk, writer.probe, kind)
    if signature in _saved:
        return
    _saved.add(signature)
    path = Path(writer.directory) / f"fusion_chunk{writer.chunk:03d}_probe{writer.probe}_{kind}.pt"
    temporary = path.with_suffix(".tmp")
    values = {name: tensor.detach().to("cpu").contiguous() for name, tensor in tensors.items()}
    torch.save({"kind": metadata.get("screen_kind", kind), "tensors": values, "metadata": metadata,
                "chunk": writer.chunk, "probe": writer.probe, "session_id": writer.session_id,
                "timing_eligible": False}, temporary)
    temporary.replace(path)
    writer.activation_records.append({"kind": "first_layer_fusion_operands", "region": kind,
                                     "path": path.name, "chunk": writer.chunk, "probe": writer.probe,
                                     "shapes": {name: list(tensor.shape) for name, tensor in values.items()}})


def install_fusion_capture(transformer) -> None:
    """Attach one first-block GELU pre-hook when the capture gate is set."""
    if os.environ.get("LINGBOT_CAPTURE_ACTIVATIONS", "0") != "1" or not transformer.blocks:
        return
    first = transformer.blocks[0]
    first._lingbot_fusion_capture_first_block = True
    first.self_attn._lingbot_fusion_capture_first_block = True
    if hasattr(first, "_lingbot_gelu_capture_hook"):
        return
    def capture_gelu(module, arguments):
        if arguments:
            capture_fusion_operands(first, "gelu_quant", {"value": arguments[0]})
    first._lingbot_gelu_capture_hook = first.ffn[1].register_forward_pre_hook(capture_gelu)


def install_vae_capture(vae) -> None:
    """Capture one real norm operand per selected decoder channel/chunk.

    Prioritize first residual norm1 at C192/C384 and norm_out at C96. Inputs
    stay in their original channel-first layout; no sampled/reencoded substitute.
    """
    if os.environ.get("LINGBOT_CAPTURE_ACTIVATIONS", "0") != "1" or hasattr(vae, "_lingbot_vae_fusion_capture_hooks"):
        return
    candidates = []
    for name, module in vae.named_modules():
        if module.__class__.__name__ not in {"RMSNormVAE", "WanRMS_norm"}:
            continue
        gamma = getattr(module, "gamma", None)
        if not isinstance(gamma, torch.Tensor) or not getattr(module, "channel_first", False):
            continue
        channels = int(gamma.numel())
        if channels not in {96, 192, 384, 1024} or "decoder" not in name:
            continue
        priority = 0 if ((channels == 96 and name.endswith("norm_out"))
                         or (channels in {192, 384, 1024} and name.endswith("norm1"))) else 1
        candidates.append((priority, name, module, channels))
    chosen = set()
    hooks = []
    for _, name, module, channels in sorted(candidates, key=lambda item: (item[0], item[1])):
        if channels in chosen:
            continue
        chosen.add(channels)
        module._lingbot_fusion_capture_first_block = True
        def capture_norm(norm, arguments, name=name, channels=channels):
            if not arguments or not isinstance(arguments[0], torch.Tensor) or arguments[0].ndim != 5:
                return
            tensors = {"value": arguments[0], "gamma": norm.gamma}
            if isinstance(getattr(norm, "bias", None), torch.Tensor):
                tensors["bias"] = norm.bias
            npu_source = norm.__class__.__name__ == "RMSNormVAE"
            capture_fusion_operands(norm, f"vae_rms_silu_C{channels}", tensors,
                screen_kind="vae_rms_silu", source_module=name, channels=channels,
                source_contract="npu_rms_norm" if npu_source else "l2_normalize",
                eps=float(norm.epsilon) if npu_source else 1e-12,
                scale=float(norm.scale), upcast=not npu_source)
        hooks.append(module.register_forward_pre_hook(capture_norm))
    vae._lingbot_vae_fusion_capture_hooks = hooks
