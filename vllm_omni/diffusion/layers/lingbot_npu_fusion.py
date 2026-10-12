# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in precision-preserving fusion contracts for the LingBot A3 path.

Importing this module does not import torch_npu or Triton or initialize an NPU.
CPU entry points are independent numerical oracles, not device fallbacks.
Device kernels are experimental until exact-shape payload/error/timing gates
pass. An explicitly enabled backend raises on unsupported inputs or compilation
failure; it never silently substitutes a different precision or eager sequence.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

REGIONS = frozenset({"qknorm_rope", "layernorm_modulation", "gelu_quant", "vae_rms_silu"})
_runtime_driver_so: str | None = None
_kernels_module = None


@dataclass(frozen=True)
class PrecisionFusionConfig:
    backend: str = "off"
    regions: tuple[str, ...] = ()
    driver_so: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"off", "triton"}:
            raise ValueError("LingBot precision fusion backend must be off or triton")
        if len(set(self.regions)) != len(self.regions) or any(region not in REGIONS for region in self.regions):
            raise ValueError(f"LingBot precision fusion regions must be unique members of {sorted(REGIONS)}")
        if self.backend == "off" and self.regions:
            raise ValueError("Fusion regions require an explicit enabled backend")
        if self.driver_so is not None and not isinstance(self.driver_so, str):
            raise ValueError("Fusion driver_so must be a string path")

    @classmethod
    def from_sources(cls, model_config: Mapping[str, Any] | None = None,
                     env: Mapping[str, str] | None = None) -> PrecisionFusionConfig:
        model = model_config or {}
        values = os.environ if env is None else env
        backend = model.get("lingbot_npu_fusion_backend", values.get("VLLM_LINGBOT_NPU_FUSION_BACKEND", "off"))
        regions = model.get("lingbot_npu_fusion_regions", values.get("VLLM_LINGBOT_NPU_FUSION_REGIONS", ""))
        if not isinstance(backend, str):
            raise ValueError("lingbot_npu_fusion_backend must be a string")
        if isinstance(regions, str):
            regions = tuple(part.strip() for part in regions.split(",") if part.strip())
        elif isinstance(regions, (list, tuple)) and all(isinstance(part, str) for part in regions):
            regions = tuple(regions)
        else:
            raise ValueError("lingbot_npu_fusion_regions must be a string or a list of strings")
        driver_so = model.get("lingbot_npu_fusion_driver_so", values.get("VLLM_LINGBOT_NPU_FUSION_DRIVER_SO"))
        return cls(backend=backend.lower(), regions=regions, driver_so=driver_so)

    def enabled(self, region: str, value: torch.Tensor) -> bool:
        return self.backend != "off" and region in self.regions and value.device.type == "npu"


def install_precision_fusion(transformer: torch.nn.Module, config: PrecisionFusionConfig) -> None:
    global _runtime_driver_so
    if config.backend == "triton" and config.regions:
        if _kernels_module is not None and config.driver_so != _runtime_driver_so:
            raise RuntimeError("Cannot switch fusion driver after device kernels have initialized")
        _runtime_driver_so = config.driver_so
    transformer._lingbot_precision_fusion = config
    for block in transformer.blocks:
        block._lingbot_precision_fusion = config
        block.self_attn._lingbot_precision_fusion = config
    from .lingbot_fusion_capture import install_fusion_capture

    install_fusion_capture(transformer)


def fusion_enabled(module: torch.nn.Module, region: str, value: torch.Tensor) -> bool:
    config = getattr(module, "_lingbot_precision_fusion", None)
    return config is not None and config.enabled(region, value)


def _device_kernels():
    global _kernels_module
    if _kernels_module is None:
        from .lingbot_triton_cann91 import configure_driver
        configure_driver(_runtime_driver_so)
        from . import lingbot_npu_fusion_kernels
        _kernels_module = lingbot_npu_fusion_kernels
    return _kernels_module


def _floating_contract(value: torch.Tensor) -> None:
    if value.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
        raise ValueError("Fusion input must be BF16, FP16, or FP32")
    if value.device.type not in {"cpu", "npu"}:
        raise ValueError("LingBot fusion supports explicit CPU references and Ascend kernels only")


def qkv_norm_rope(
    qkv: torch.Tensor, query_weight: torch.Tensor, key_weight: torch.Tensor,
    cosine: torch.Tensor, sine: torch.Tensor, *, heads: int, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full-feature Q/K RMSNorm -> dtype rounding -> FP32 interleaved RoPE.

    Norm reduces over the complete 5120 features before SP head sharding.
    Its BF16 output boundary is retained before the FP32 rotary arithmetic.
    Q/K/V are returned in BSND with identical head ordering and value bits.
    """
    _floating_contract(qkv)
    if qkv.ndim != 3 or qkv.shape[-1] % 3 or heads <= 0 or eps <= 0 or not math.isfinite(eps):
        raise ValueError("QKV fusion requires nonempty BSD with 3×features and positive heads/epsilon")
    width = qkv.shape[-1] // 3
    if width % heads or width // heads % 2 or qkv.numel() == 0:
        raise ValueError("QKV fusion requires an even head dimension and nonempty input")
    head_dim = width // heads
    if query_weight.shape != (width,) or key_weight.shape != (width,):
        raise ValueError("QK norm weights must span full features before head sharding")
    if cosine.shape != (qkv.shape[1], head_dim // 2) or sine.shape != cosine.shape:
        raise ValueError("RoPE coefficients must match local sequence×half-head")
    if any(t.device != qkv.device for t in (query_weight, key_weight, cosine, sine)):
        raise ValueError("All QKV fusion tensors must share a device")
    if qkv.device.type == "npu":
        return _device_kernels().qkv_norm_rope(qkv, query_weight, key_weight, cosine, sine, heads, eps)
    query, key, value = qkv.split(width, dim=-1)
    def norm_and_rotate(x, weight):
        floating = x.float()
        rms = torch.sqrt(floating.square().sum(-1, keepdim=True) / width + eps)
        rounded = (floating / rms * weight.float()).to(qkv.dtype)
        pairs = rounded.unflatten(-1, (heads, head_dim)).float().unflatten(-1, (-1, 2))
        even, odd = pairs.unbind(-1)
        c, s = cosine.float()[None, :, None, :], sine.float()[None, :, None, :]
        return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).flatten(-2).to(qkv.dtype)
    return norm_and_rotate(query, query_weight), norm_and_rotate(key, key_weight), value.unflatten(-1, (heads, head_dim))


def layernorm_modulation(
    value: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, *, eps: float,
) -> torch.Tensor:
    """FP32 LayerNorm -> separate FP32 multiply/add -> original dtype.

    scale/shift use B×frames×1×D, with one frame entry per token under SP.
    No FMA is allowed to erase the original product rounding boundary.
    """
    _floating_contract(value)
    if value.ndim != 3 or scale.ndim != 4 or scale.shape != shift.shape:
        raise ValueError("LayerNorm modulation requires BSD and equal BF1D scale/shift")
    b, tokens, width = value.shape
    frames = scale.shape[1]
    if scale.shape != (b, frames, 1, width) or frames <= 0 or tokens % frames or eps <= 0:
        raise ValueError("LayerNorm modulation frame/feature dimensions do not align")
    if scale.device != value.device or shift.device != value.device:
        raise ValueError("LayerNorm modulation tensors must share a device")
    if value.device.type == "npu":
        return _device_kernels().layernorm_modulation(value, scale, shift, eps)
    normalized = F.layer_norm(value.float(), (width,), eps=eps).unflatten(1, (frames, tokens // frames))
    return (normalized * (1 + scale.float()) + shift.float()).flatten(1, 2).to(value.dtype)


def per_token_quant_reference(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU oracle for native symmetric INT8; zero-token convention is gated on device."""
    if value.device.type != "cpu" or value.ndim < 2 or not value.is_floating_point():
        raise ValueError("Per-token quant reference requires a CPU floating matrix")
    floating = value.float()
    maximum = floating.abs().amax(-1)
    # Native all-zero convention is not documented; the NPU screen must compare
    # payload AND scale on all-zero/tiny/saturation/tie cases before enabling.
    scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
    integers = torch.round(floating / scale[..., None]).clamp(-127, 127).to(torch.int8)
    return integers, scale


def gelu_quant(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """tanh GELU in FP32 -> original BF16 rounding -> per-token native-contract quant."""
    _floating_contract(value)
    if value.ndim < 2 or value.shape[-1] <= 0:
        raise ValueError("GELU quant requires a nonempty feature dimension")
    if value.device.type == "npu":
        if value.dtype != torch.bfloat16:
            raise ValueError("GELU quant NPU candidate currently requires BF16")
        return _device_kernels().gelu_quant(value)
    rounded = F.gelu(value.float(), approximate="tanh").to(value.dtype)
    return per_token_quant_reference(rounded)


def vae_rms_silu(
    value: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor | None,
    *, eps: float, scale: float, upcast: bool = True, source_contract: str = "l2_normalize",
) -> torch.Tensor:
    """Channel-first VAE L2-normalize -> input rounding -> affine -> SiLU.

    l2_normalize is F.normalize's max(norm, eps), NOT sqrt(mean(x²)+eps).
    upcast selects the source VAE contract; a BF16 gamma retains BF16 affine
    rounding while FP32 gamma promotes the affine and activation to FP32.
    npu_rms_norm preserves RMSNormVAE.forward_npu: FP32 RMS reduction and
    gamma inside the norm, input-dtype norm output, optional bias, then SiLU.
    """
    _floating_contract(value)
    if source_contract not in {"l2_normalize", "npu_rms_norm"}:
        raise ValueError("VAE norm source_contract must be l2_normalize or npu_rms_norm")
    if value.ndim != 5 or gamma.numel() != value.shape[1] or eps <= 0 or not math.isfinite(scale):
        raise ValueError("VAE fusion requires BCTHW, one gamma per channel and valid epsilon/scale")
    if bias is not None and bias.numel() != value.shape[1]:
        raise ValueError("VAE bias must contain one entry per channel")
    if gamma.device != value.device or (bias is not None and bias.device != value.device):
        raise ValueError("VAE fusion tensors must share a device")
    if value.device.type == "npu":
        return _device_kernels().vae_rms_silu(value, gamma, bias, eps, scale, upcast, source_contract)
    if source_contract == "npu_rms_norm":
        floating = value.float()
        rms = torch.sqrt(floating.square().mean(1, keepdim=True) + eps)
        result = (floating / rms * gamma.float().reshape(1, -1, 1, 1, 1)).to(value.dtype)
        if bias is not None:
            result = result + bias.reshape(1, -1, 1, 1, 1)
        return F.silu(result)
    normalized = F.normalize(value.float() if upcast else value, p=2, dim=1, eps=eps).type_as(value)
    normalized = normalized * scale
    result = normalized * gamma.reshape(1, -1, 1, 1, 1)
    if bias is not None:
        result = result + bias.reshape(1, -1, 1, 1, 1)
    return F.silu(result)
