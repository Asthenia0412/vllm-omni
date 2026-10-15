# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Incremental Matrix conditioning over Omni's Wan transformer components.

The checkpoint supplies a 148-channel patch embedding and causal weights. Wan
owns projections, feed-forward layers, norms and attention dispatch. This adapter
owns only the released temporal/sink policy and checkpoint-name translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from vllm.model_executor.layers.linear import QKVParallelLinear

from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel


class MatrixRMSNorm(RMSNorm):
    """Wan checkpoint semantics round to BF16 before multiplying the weight."""

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return value.to(x.dtype) * self.weight

    forward_cuda = forward_native


class MatrixQKV(QKVParallelLinear):
    """Reuse packed weight loading, preserving the reference's three BF16 GEMMs.

    Packing Q/K/V into one GEMM changes accumulation/rounding on SM120. The
    resident single-rank correctness path therefore evaluates the three slices.
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        assert self.bias is not None
        output = [F.linear(x, weight, bias) for weight, bias in zip(self.weight.chunk(3), self.bias.chunk(3))]
        return torch.cat(output, dim=-1), None


@dataclass
class TemporalCache:
    key: torch.Tensor
    value: torch.Tensor
    sink_key: torch.Tensor | None = None
    position: int = -1
    slot: int = 0
    length: int = 0

    def write(self, key: torch.Tensor, value: torch.Tensor, position: int) -> None:
        """Keep slot zero as sink; rewrite the current block during denoising."""
        tokens = key.shape[1]
        capacity = self.key.shape[1] // tokens
        if position != self.position:
            if position != self.position + 1:
                raise ValueError("Temporal KV writes must be consecutive or overwrite the current block")
            if self.length == capacity:
                # Preserve the reference's chronological attention reduction
                # order. A ring/permuted layout needs separate qualification.
                for buffer in (self.key, self.value):
                    buffer[:, tokens:-tokens].copy_(buffer[:, 2 * tokens :].clone())
            self.slot = min(position, capacity - 1)
            self.position = position
            self.length = min(capacity, self.length + 1)
        start = self.slot * tokens
        self.key[:, start : start + tokens].copy_(key)
        self.value[:, start : start + tokens].copy_(value)


@dataclass
class TextCache:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None


def checkpoint_name(name: str) -> str | None:
    """Translate released student names into the existing Wan weight loader."""
    for prefix in ("pipe.dit.", "model."):
        name = name.removeprefix(prefix)
    if name.startswith("ref_conv."):
        return None  # Training-only reference branch, absent from causal inference.
    roots = {
        "text_embedding.0.": "condition_embedder.text_embedder.linear_1.",
        "text_embedding.2.": "condition_embedder.text_embedder.linear_2.",
        "time_embedding.0.": "condition_embedder.time_embedder.linear_1.",
        "time_embedding.2.": "condition_embedder.time_embedder.linear_2.",
        "time_projection.1.": "condition_embedder.time_proj.",
        "head.head.": "proj_out.",
    }
    if name == "head.modulation":
        return "output_scale_shift_prepare.scale_shift_table"
    for source, target in roots.items():
        if name.startswith(source):
            return target + name.removeprefix(source)
    if name.startswith("blocks."):
        name = name.replace(".self_attn.", ".attn1.").replace(".cross_attn.", ".attn2.")
        for source, target in (("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("o", "to_out")):
            for attention in ("attn1", "attn2"):
                name = name.replace(f".{attention}.{source}.", f".{attention}.{target}.")
        name = name.replace(".norm3.", ".norm2.")
        name = name.replace(".ffn.0.", ".ffn.net_0.proj.").replace(".ffn.2.", ".ffn.net_2.")
        name = name.replace(".modulation", ".scale_shift_table")
    return name


def load_transformer(checkpoint: Path, *, device: torch.device) -> MatrixRuntime:
    model = WanTransformer3DModel(
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=148,
        out_channels=48,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        rope_max_seq_len=8193,
    ).to(dtype=torch.bfloat16)
    # The shared Conv3d GEMM fast path changes the reduction order. Use its
    # existing convolution path for reference qualification.
    model.patch_embedding.enable_linear = False
    for index, block in enumerate(model.blocks):
        block.attn1.to_qkv = MatrixQKV(
            hidden_size=3072,
            head_size=128,
            total_num_heads=24,
            bias=True,
            params_dtype=torch.bfloat16,
            prefix=f"blocks.{index}.attn1.to_qkv",
        )
        for attention in (block.attn1, block.attn2):
            attention.norm_q = MatrixRMSNorm(3072, eps=1e-6, dtype=torch.bfloat16)
            attention.norm_k = MatrixRMSNorm(3072, eps=1e-6, dtype=torch.bfloat16)
    expected = set(dict(model.named_parameters()))
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        translated = {name: checkpoint_name(name) for name in source.keys()}
        for target in translated.values():
            if target is None:
                continue
            packed = target
            for projection in ("q", "k", "v"):
                packed = packed.replace(f".attn1.to_{projection}.", ".attn1.to_qkv.")
            if packed not in expected:
                raise ValueError(f"Unsupported Matrix checkpoint weight: {target}")
        loaded = model.load_weights(
            (target, source.get_tensor(name)) for name, target in translated.items() if target is not None
        )
    if missing := expected - loaded:
        raise ValueError(f"Incomplete Matrix checkpoint: {sorted(missing)}")
    model.to(device=device).eval().requires_grad_(False)
    return MatrixRuntime(model)


class MatrixRuntime:
    """Single-session causal execution using shared Wan layers and Attention."""

    def __init__(self, model: WanTransformer3DModel):
        self.model = model
        self.rotations: torch.Tensor | None = None

    def ensure_rope_positions(self, positions: int) -> None:
        if self.rotations is not None and self.rotations.shape[0] >= positions:
            return
        # Preserve the checkpoint's FP64 44/42/42 temporal/height/width split.
        device = self.model.patch_embedding.weight.device
        axes = []
        for channels in (44, 42, 42):
            frequencies = 10000 ** (-torch.arange(0, channels, 2, dtype=torch.float64, device=device) / channels)
            angles = torch.arange(positions, dtype=torch.float64, device=device)[:, None] * frequencies
            axes.append(torch.polar(torch.ones_like(angles), angles))
        self.rotations = torch.cat(axes, dim=-1)

    def init_kv_cache(self, batch_size: int, frame_tokens: int, device: torch.device, dtype: torch.dtype):
        shape = (batch_size, 21 * frame_tokens, 24, 128)
        return [
            TemporalCache(
                torch.empty(shape, device=device, dtype=dtype), torch.empty(shape, device=device, dtype=dtype)
            )
            for _ in self.model.blocks
        ]

    def init_crossattn_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        return [TextCache() for _ in self.model.blocks]

    def _rotate(self, value: torch.Tensor, position: int, height: int, width: int) -> torch.Tensor:
        assert self.rotations is not None
        temporal, vertical, horizontal = self.rotations.split((22, 21, 21), dim=-1)
        rotation = torch.cat(
            (
                temporal[position].expand(height, width, -1),
                vertical[:height, None].expand(height, width, -1),
                horizontal[None, :width].expand(height, width, -1),
            ),
            dim=-1,
        ).reshape(1, height * width, 1, 64)
        pairs = torch.view_as_complex(value.double().reshape(*value.shape[:-1], 64, 2))
        return torch.view_as_real(pairs * rotation).flatten(-2).to(value.dtype)

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor,
        kv_cache: list[TemporalCache],
        crossattn_cache: list[TextCache],
        current_start: int,
        kv_only: bool = False,
        sink_position: int | None = None,
    ) -> torch.Tensor | None:
        model = self.model
        hidden = model.patch_embedding(torch.cat((x, y), dim=1))
        batch, dim, frames, height, width = hidden.shape
        if frames != 1:
            raise ValueError("Matrix causal execution accepts one latent frame per call")
        position = current_start // (height * width)
        hidden = hidden.flatten(2).transpose(1, 2)
        embedder = model.condition_embedder
        frequencies = 10000 ** (-torch.arange(128, dtype=torch.float64, device=x.device) / 128)
        angles = t.flatten().double()[:, None] * frequencies
        time = embedder.time_embedder(torch.cat((angles.cos(), angles.sin()), dim=-1).to(hidden.dtype))
        modulation = embedder.time_proj(embedder.act_fn(time)).reshape(batch, 6, dim)
        text = None if all(cache.key is not None for cache in crossattn_cache) else embedder.text_embedder(context)
        for index, (block, cache, cross) in enumerate(zip(model.blocks, kv_cache, crossattn_cache)):
            shift, scale, gate, ff_shift, ff_scale, ff_gate = (modulation + block.scale_shift_table).chunk(6, dim=1)
            attn = block.attn1
            normalized = block.norm1.forward_native(hidden, scale, shift)
            qkv, _ = attn.to_qkv(normalized)
            query, key, value = qkv.chunk(3, dim=-1)
            query = attn.norm_q(query).reshape(batch, -1, 24, 128)
            key = attn.norm_k(key).reshape(batch, -1, 24, 128)
            value = value.reshape(batch, -1, 24, 128)
            if position == 0:
                cache.sink_key = key
            query = self._rotate(query, position, height, width)
            cache.write(self._rotate(key, position, height, width), value, position)
            if sink_position is not None:
                assert cache.sink_key is not None
                cache.key[:, : height * width].copy_(self._rotate(cache.sink_key, sink_position, height, width))
            if kv_only and index + 1 == len(model.blocks):
                return None
            length = cache.length * height * width
            attended = attn.attn(query, cache.key[:, :length], cache.value[:, :length])
            hidden = hidden + attn.to_out(attended.flatten(2)) * gate
            attn = block.attn2
            query = attn.norm_q(attn.to_q(block.norm2(hidden))).reshape(batch, -1, 24, 128)
            if cross.key is None:
                assert text is not None
                cross.key = attn.norm_k(attn.to_k(text)).reshape(batch, -1, 24, 128)
                cross.value = attn.to_v(text).reshape(batch, -1, 24, 128)
            assert cross.value is not None
            attended = attn.attn(query, cross.key, cross.value)
            hidden = hidden + attn.to_out(attended.flatten(2))
            hidden = hidden + block.ffn(block.norm3.forward_native(hidden, ff_scale, ff_shift)) * ff_gate
        shift, scale = (model.output_scale_shift_prepare.scale_shift_table + time[:, None]).chunk(2, dim=1)
        output = model.proj_out(model.norm_out.forward_native(hidden, scale, shift))
        return (
            output.reshape(batch, 1, height, width, 1, 2, 2, 48)
            .permute(0, 7, 1, 4, 2, 5, 3, 6)
            .reshape(batch, 48, 1, height * 2, width * 2)
        )
