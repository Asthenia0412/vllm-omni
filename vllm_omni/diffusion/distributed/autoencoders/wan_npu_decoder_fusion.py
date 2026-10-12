# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in A3 decoder norm/SiLU adapter, preserving the deployed NPU norm.

The RMSNormVAE NPU implementation uses npu_rms_norm, whereas its native
implementation uses F.normalize. The source contracts must not be conflated.
Channel-first layouts, streaming decoder state and halo collectives stay intact.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class NPUWanNormSiLU(nn.Module):
    def __init__(self, norm: nn.Module) -> None:
        super().__init__()
        self.gamma = norm.gamma
        if isinstance(norm.bias, torch.Tensor):
            self.bias = norm.bias
        else:
            self.bias = None
            if norm.bias not in (None, 0, 0.0):
                raise ValueError("NPU VAE fusion supports tensor bias or zero scalar bias")
        # Keep checkpoint parameter names under norm1.gamma / norm2.gamma.
        # This module is installed after loading; retain the original CPU
        # fallback without registering a second copy of its parameters.
        object.__setattr__(self, "_original_norm", norm)
        self.source_contract = "npu_rms_norm" if norm.__class__.__name__ == "RMSNormVAE" else "l2_normalize"
        self.upcast = norm.__class__.__name__ == "WanRMS_norm"
        self.eps = float(getattr(norm, "epsilon", 1e-12 if self.upcast else 1e-6))
        self.scale = float(norm.scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.device.type != "npu":
            return torch.nn.functional.silu(self._original_norm(value))
        if torch.is_grad_enabled() and value.requires_grad:
            raise ValueError("NPU VAE fusion is an inference-only kernel")
        from vllm_omni.diffusion.layers.lingbot_npu_fusion import vae_rms_silu

        return vae_rms_silu(
            value, self.gamma, self.bias, eps=self.eps, scale=self.scale,
            upcast=self.upcast, source_contract=self.source_contract,
        )


def install_wan_npu_decoder_fusion(vae: Any) -> dict[str, Any]:
    """Replace only verified norm→SiLU chains, with atomic validation.

    This is independent of the CUDA fused fast path and changes no convolution
    layout or weight dtype. Kernels require NPU parity/quality screening.
    """
    installed = getattr(vae, "_lingbot_npu_decoder_fusion", None)
    if installed is not None:
        return installed
    decoder = getattr(vae, "decoder", None)
    if not isinstance(decoder, nn.Module):
        raise ValueError("NPU decoder fusion requires a decoder module")
    blocks = [module for module in decoder.modules() if module.__class__.__name__ == "WanResidualBlock"]
    entries = []
    for parent, names in [(block, ("norm1", "norm2")) for block in blocks] + [(decoder, ("norm_out",))]:
        activation = getattr(parent, "nonlinearity", None)
        if not isinstance(activation, nn.SiLU) or activation.inplace:
            raise ValueError("NPU decoder fusion requires a plain non-inplace SiLU")
        for name in names:
            norm = getattr(parent, name, None)
            if norm.__class__.__name__ not in ("WanRMS_norm", "RMSNormVAE") or not norm.channel_first:
                raise ValueError(f"NPU decoder fusion requires a channel-first Wan norm at {name}")
            if not isinstance(norm.gamma, torch.Tensor) or norm.gamma.ndim != 4:
                raise ValueError("NPU decoder fusion requires 5D decoder norm parameters")
            entries.append((parent, name, NPUWanNormSiLU(norm)))
    for parent, name, norm in entries:
        setattr(parent, name, norm)
    for parent in [*blocks, decoder]:
        parent.nonlinearity = nn.Identity()
    counts = {"norm_silu_regions": len(entries), "contracts": sorted({norm.source_contract for _, _, norm in entries})}
    vae._lingbot_npu_decoder_fusion = counts
    return counts
