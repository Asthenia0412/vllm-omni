# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in Ascend fast paths for LingBot's exact inference layouts.

Norm and RoPE arithmetic stays FP32. The gates allow end-to-end A/B testing
without changing checkpoint parameters, cache semantics, or other platforms.
"""

from __future__ import annotations

import logging
import os

import torch

from .lingbot_npu_attention import _BACKEND_FAILURES, run_attention

logger = logging.getLogger(__name__)
_INFER_FAILURES = _BACKEND_FAILURES
_ROTARY_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def fused_pointwise_enabled(value: torch.Tensor) -> bool:
    return value.device.type == "npu" and os.environ.get("VLLM_LINGBOT_NPU_FUSED_POINTWISE", "0") == "1"


def lossy_pointwise_enabled() -> bool:
    return os.environ.get("VLLM_LINGBOT_NPU_LOSSY_POINTWISE", "0") == "1"


def npu_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    head_num: int,
    scale: float,
    backend: str | None = None,
    num_splits: int | None = None,
    strict: bool | None = None,
) -> torch.Tensor:
    """Unmasked noncausal BSND attention over exactly the supplied visible KV.

    No sequence padding is introduced. This supports the changing history
    lengths during cache ramp-up as well as the steady-state window.
    """
    if os.environ.get("LINGBOT_CAPTURE_ACTIVATIONS", "0") == "1":
        from .lingbot_capture import capture_attention

        capture_attention(query, key, value, head_num=head_num, scale=scale)
    if os.environ.get("LINGBOT_CAPTURE_COMPUTE", "0") == "1":
        from .lingbot_compute_capture import record_attention

        record_attention(query, key, value, input_layout="BSND", attention_kind="self")
    return run_attention(
        query, key, value, head_num=head_num, scale=scale, backend=backend, num_splits=num_splits, strict=strict
    )


def fp32_rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Replace the TP=1 pointwise RMSNorm sequence with one FP32 norm kernel."""
    import torch_npu

    result = torch_npu.npu_rms_norm(value.float(), weight.float(), epsilon=eps)[0]
    return result.to(value.dtype)


def modulate(normalized: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Apply FP32 norm modulation with one product-plus-add kernel."""
    return torch.addcmul(shift, normalized, 1 + scale)


def gated_residual(hidden: torch.Tensor, update: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return torch.addcmul(hidden, update, gate)


def fp32_rotary_interleaved(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    """Interleaved Wan RoPE, with FP32 coefficients and arithmetic.

    The repeated full-width coefficients are reused across all transformer
    layers sharing this chunk's position table. One entry per device retains
    the source tensors, avoiding allocator-address reuse and unbounded caches.
    """
    import torch_npu

    if value.ndim != 4 or cosine.ndim != 2 or cosine.shape != sine.shape:
        raise ValueError("LingBot fused RoPE expects BSND and sequence×half-head coefficients")
    if cosine.shape != (value.shape[1], value.shape[-1] // 2) or value.shape[-1] % 2:
        raise ValueError("LingBot fused RoPE coefficients do not match the sequence/head dimensions")
    cached = _ROTARY_CACHE.get(value.device)
    if cached is None or cached[0] is not cosine or cached[1] is not sine:
        full_cosine = cosine.float().repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(2)
        full_sine = sine.float().repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(2)
        cached = (cosine, sine, full_cosine, full_sine)
        _ROTARY_CACHE[value.device] = cached
    return torch_npu.npu_rotary_mul(value.float(), cached[2], cached[3], rotary_mode="interleave").to(value.dtype)
