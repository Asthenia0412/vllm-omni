# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in online INT8 linears for the LingBot SP4 / TP1 Ascend path.

Weights are quantized once after checkpoint loading, on the CPU, with one
symmetric scale per output channel. W8A8 uses native per-token activation
quantization plus INT8 QuantMatmul; W8A16 uses native weight-only matmul.
Checkpoint files and the default BF16 path are unchanged. This is an inference
experiment, not a calibrated checkpoint or a claim of unchanged video quality.
"""

from __future__ import annotations

import gc
import fnmatch
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Any, Mapping

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# These are the only projections eligible for controlled realtime expansion.
# Fused QKV counts as one physical module; cross-attention text K/V stay BF16.
CONTROLLED_PROJECTIONS = {
    "ffn": frozenset({"ffn.0", "ffn.2"}),
    "self_qkv": frozenset({"self_attn.qkv"}),
    "self_out": frozenset({"self_attn.o"}),
    "cross_q": frozenset({"cross_attn.q"}),
    "cross_out": frozenset({"cross_attn.o"}),
}
CONTROLLED_SCOPES = {
    "ffn": ("ffn",),
    "qkv": ("self_qkv",),
    "self_out": ("self_out",),
    "cross_q": ("cross_q",),
    "cross_out": ("cross_out",),
    "ffn_qkv": ("ffn", "self_qkv"),
    "ffn_self": ("ffn", "self_qkv", "self_out"),
    "ffn_self_crossq": ("ffn", "self_qkv", "self_out", "cross_q"),
    "ffn_self_crosso": ("ffn", "self_qkv", "self_out", "cross_out"),
    "attention": ("self_qkv", "self_out", "cross_q", "cross_out"),
    "hot": tuple(CONTROLLED_PROJECTIONS),
}


def _patterns(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        result = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, (list, tuple)) and all(isinstance(part, str) for part in value):
        result = tuple(value)
    else:
        raise ValueError(f"{name} must be a comma-separated string or a list of strings")
    if any(not part or part != part.strip() for part in result):
        raise ValueError(f"{name} cannot contain empty or whitespace-padded entries")
    return result


@dataclass(frozen=True)
class NpuQuantizationConfig:
    mode: str = "off"
    scope: str = "hot"
    weight_format: str = "nd"
    keep_bf16_cpu: bool = False
    require_npu: bool = True
    projections: tuple[str, ...] = ()
    allowlist: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"off", "w8a8", "w8a16"}:
            raise ValueError("LINGBOT_NPU_QUANT must be off, w8a8, or w8a16")
        if self.scope not in {*CONTROLLED_SCOPES, "all"}:
            raise ValueError(f"Unknown LINGBOT_NPU_QUANT_SCOPE: {self.scope}")
        if any(name not in CONTROLLED_PROJECTIONS for name in self.projections):
            raise ValueError(f"Projection selectors must be one of {tuple(CONTROLLED_PROJECTIONS)}")
        if self.scope == "all" and (self.projections or self.allowlist or self.exclude):
            raise ValueError("Legacy scope=all cannot be combined with controlled selectors; use hot")
        if len(set(self.projections)) != len(self.projections):
            raise ValueError("Projection selectors must not repeat")
        if self.weight_format not in {"nd", "nz"}:
            raise ValueError("LINGBOT_NPU_QUANT_WEIGHT_FORMAT must be nd or nz")
        if self.mode == "w8a16" and self.weight_format == "nz":
            raise ValueError("W8A16 currently uses transposed ND weights; choose weight_format=nd")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> NpuQuantizationConfig:
        values = os.environ if env is None else env
        mode = values.get("LINGBOT_NPU_QUANT", "off").lower()
        mode = {"none": "off", "bf16": "off", "int8": "w8a8"}.get(mode, mode)
        keep = values.get("LINGBOT_NPU_QUANT_KEEP_BF16_CPU", "0")
        if keep not in {"0", "1"}:
            raise ValueError("LINGBOT_NPU_QUANT_KEEP_BF16_CPU must be 0 or 1")
        return cls(
            mode=mode,
            scope=values.get("LINGBOT_NPU_QUANT_SCOPE", "hot").lower(),
            weight_format=values.get("LINGBOT_NPU_QUANT_WEIGHT_FORMAT", "nd").lower(),
            keep_bf16_cpu=keep == "1",
            projections=_patterns(values.get("LINGBOT_NPU_QUANT_PROJECTIONS", ""), "quant projections"),
            allowlist=_patterns(values.get("LINGBOT_NPU_QUANT_ALLOWLIST", ""), "quant allowlist"),
            exclude=_patterns(values.get("LINGBOT_NPU_QUANT_EXCLUDE", ""), "quant exclusions"),
        )

    @classmethod
    def from_sources(
        cls, model_config: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None,
    ) -> NpuQuantizationConfig:
        """Explicit model settings override corresponding environment settings."""
        values = dict(os.environ if env is None else env)
        for key, suffix in {
            "lingbot_npu_quant": "", "lingbot_npu_quant_scope": "_SCOPE",
            "lingbot_npu_quant_weight_format": "_WEIGHT_FORMAT",
            "lingbot_npu_quant_projections": "_PROJECTIONS",
            "lingbot_npu_quant_allowlist": "_ALLOWLIST", "lingbot_npu_quant_exclude": "_EXCLUDE",
        }.items():
            if key not in (model_config or {}):
                continue
            value = model_config[key]
            if suffix in {"_PROJECTIONS", "_ALLOWLIST", "_EXCLUDE"}:
                value = ",".join(_patterns(value, key))
            elif not isinstance(value, str):
                raise ValueError(f"model_config.{key} must be a string")
            values["LINGBOT_NPU_QUANT" + suffix] = value
        return cls.from_env(values)


def selected_linear(
    name: str, scope: str, *, projections: tuple[str, ...] = (),
    allowlist: tuple[str, ...] = (), exclude: tuple[str, ...] = (),
) -> bool:
    """Select transformer block projections, leaving embeddings/head untouched."""
    match = re.fullmatch(r"blocks\.\d+\.(.+)", name)
    if not match:
        return False
    suffix = match.group(1)
    if scope in CONTROLLED_SCOPES:
        selectors = projections or CONTROLLED_SCOPES[scope]
        eligible = frozenset().union(*(CONTROLLED_PROJECTIONS[selector] for selector in selectors))
        # An allowlist only narrows eligibility; it can never enable conditioning
        # or output modules even when a wildcard explicitly names those paths.
        return (
            suffix in eligible
            and (not allowlist or any(fnmatch.fnmatchcase(name, pattern) for pattern in allowlist))
            and not any(fnmatch.fnmatchcase(name, pattern) for pattern in exclude)
        )
    if scope == "all":
        return suffix in {
            "self_attn.qkv", "self_attn.o", "cross_attn.q", "cross_attn.k",
            "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2",
            "cam_injector_layer1", "cam_injector_layer2", "cam_scale_layer", "cam_shift_layer",
        }
    raise ValueError(f"Unknown quantization scope: {scope}")


def useful_matrix_ledger(
    config: NpuQuantizationConfig, *, layers: int = 40, forwards: int = 4, ranks: int = 4,
    local_tokens: int = 1170, hidden: int = 5120, ffn_hidden: int = 13824,
    attention_query_tokens: int = 4680, attention_kv_tokens: int = 28080,
    heads_per_rank: int = 10, head_dim: int = 128, text_tokens: int = 512,
    served_seconds: float | None = None, bf16_peak_tflops: float = 363.7248,
    int8_peak_tops: float | None = None, int8_peak_status: str = "unverified",
) -> dict[str, Any]:
    """Selected useful-matrix lower-bound, recalculated for exact selectors.

    No padding or replicated text K/V work enters the numerator. Camera linears
    run once per chunk while hot projections run once per probe. Logical-rank
    arithmetic scales by ranks; rank elapsed durations must never be summed.
    A supplied unverified INT8 peak gives a separately labeled conditional proxy.
    """
    dimensions = (layers, forwards, ranks, local_tokens, hidden, ffn_hidden,
                  attention_query_tokens, attention_kv_tokens, heads_per_rank, head_dim, text_tokens)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions):
        raise ValueError("Useful-matrix dimensions and counts must be positive integers")
    if attention_query_tokens != local_tokens * ranks or hidden != heads_per_rank * ranks * head_dim:
        raise ValueError("Useful-matrix ledger requires matching SP token/head geometry and TP1")
    if served_seconds is not None and (not math.isfinite(served_seconds) or served_seconds <= 0):
        raise ValueError("Served duration must be finite and positive")
    if bf16_peak_tflops <= 0 or (int8_peak_tops is not None and int8_peak_tops <= 0):
        raise ValueError("Theoretical precision peaks must be positive")
    rows = [{"name": "self_attention_qk_pv", "precision": "bf16", "launches_per_rank": layers*forwards,
             "operations_per_rank": layers*forwards*4*attention_query_tokens*attention_kv_tokens*heads_per_rank*head_dim},
            {"name": "cross_attention_qk_pv", "precision": "bf16", "launches_per_rank": layers*forwards,
             "operations_per_rank": layers*forwards*4*local_tokens*text_tokens*(heads_per_rank*ranks)*head_dim}]
    matrices = {
        "self_attn.qkv": (hidden, 3*hidden, forwards),
        "self_attn.o": (hidden, hidden, forwards),
        "cross_attn.q": (hidden, hidden, forwards),
        "cross_attn.o": (hidden, hidden, forwards),
        "ffn.0": (hidden, ffn_hidden, forwards),
        "ffn.2": (ffn_hidden, hidden, forwards),
        "cam_injector_layer1": (hidden, hidden, 1),
        "cam_injector_layer2": (hidden, hidden, 1),
        "cam_scale_layer": (hidden, hidden, 1),
        "cam_shift_layer": (hidden, hidden, 1),
    }
    for index in range(layers):
        for suffix, (k, n, repeats) in matrices.items():
            name = f"blocks.{index}.{suffix}"
            quantized = config.mode == "w8a8" and selected_linear(
                name, config.scope, projections=config.projections, allowlist=config.allowlist, exclude=config.exclude,
            )
            rows.append({"name": name, "precision": "int8" if quantized else "bf16",
                         "launches_per_rank": repeats,
                         "operations_per_rank": repeats*2*local_tokens*k*n,
                         "matrix_mkn": [local_tokens, k, n]})
    operations = {precision: sum(row["operations_per_rank"] for row in rows if row["precision"] == precision)
                  for precision in ("bf16", "int8")}
    global_ops = {precision: value*ranks for precision, value in operations.items()}
    ideals = {"bf16": operations["bf16"]/(bf16_peak_tflops*1e12),
              "int8": None if int8_peak_tops is None else operations["int8"]/(int8_peak_tops*1e12)}
    mixed = None
    conditional = None
    if served_seconds is not None and ideals["int8"] is not None:
        proxy = (ideals["bf16"]+ideals["int8"])/served_seconds
        if int8_peak_status == "verified":
            mixed = proxy
        else:
            conditional = proxy
    return {
        "metric": "selected useful-matrix lower-bound; MAC=2 operations",
        "config": asdict(config), "logical_ranks": ranks, "forwards_per_chunk": forwards,
        "self_attention_launches_per_rank_per_chunk": layers*forwards,
        "operations_per_rank": operations, "operations_global": global_ops,
        "useful_operations_total_global": sum(global_ops.values()),
        "ideal_seconds_per_rank": ideals, "bf16_peak_tflops_per_rank": bf16_peak_tflops,
        "int8_peak_tops_per_rank": int8_peak_tops, "int8_peak_status": int8_peak_status,
        "served_seconds": served_seconds,
        "bf16_e2e_selected_proxy": None if served_seconds is None else ideals["bf16"]/served_seconds,
        "int8_e2e_tops_global": None if served_seconds is None else global_ops["int8"]/served_seconds/1e12,
        "precision_weighted_selected_proxy": mixed,
        "conditional_precision_weighted_selected_proxy": conditional,
        "whole_model_mfu": None,
        "exclusions": ["VAE/RGB", "conditioning/text/prefill", "smaller unlisted matrices",
                       "norm/softmax/layout/quantizer arithmetic", "communication", "SP-replicated text K/V"],
        "rows": rows,
    }


@torch.no_grad()
def quantize_weight_per_channel(weight: torch.Tensor, *, chunk_rows: int = 256,
                                shared_maximum: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """CPU-only quantization and error statistics with bounded temporary storage."""
    if weight.ndim != 2 or not weight.is_floating_point() or weight.device.type == "meta":
        raise ValueError("Quantization requires a materialized floating-point 2D weight")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    rows, cols = weight.shape
    if rows == 0 or cols == 0:
        raise ValueError("Weight dimensions must be nonempty")
    source = weight.detach().to(device="cpu")
    if shared_maximum is not None:
        if (shared_maximum.shape != (rows,) or shared_maximum.device.type != "cpu"
                or shared_maximum.dtype != torch.float32 or not torch.isfinite(shared_maximum).all()
                or (shared_maximum < 0).any()):
            raise ValueError("Shared TP weight maximum requires finite nonnegative CPU FP32 channel values")
    quantized = torch.empty((rows, cols), dtype=torch.int8, device="cpu")
    scale = torch.empty(rows, dtype=torch.float32, device="cpu")
    squared_error = squared_reference = 0.0
    max_error = 0.0
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        block = source[start:stop].float()
        if not bool(torch.isfinite(block).all()):
            raise ValueError("Cannot quantize a weight containing NaN or infinity")
        maximum = block.abs().amax(dim=1)
        if shared_maximum is not None:
            if (shared_maximum[start:stop] < maximum).any():
                raise ValueError("Shared TP weight maximum is smaller than its local shard")
            maximum = shared_maximum[start:stop].float()
        local_scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
        integers = torch.round(block / local_scale[:, None]).clamp(-127, 127).to(torch.int8)
        quantized[start:stop] = integers
        scale[start:stop] = local_scale
        error = block - integers.float() * local_scale[:, None]
        squared_error += error.square().sum(dtype=torch.float64).item()
        squared_reference += block.square().sum(dtype=torch.float64).item()
        max_error = max(max_error, error.abs().max().item())
    return quantized, scale, {
        "elements": rows * cols,
        "max_abs_weight_error": max_error,
        "weight_rmse": math.sqrt(squared_error / (rows * cols)),
        "weight_relative_rmse": math.sqrt(squared_error / max(squared_reference, 1e-30)),
        "weight_squared_error": squared_error,
        "weight_squared_reference": squared_reference,
    }


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Full-output FP64 diagnostics with bounded temporaries, on the CPU.

    FP32 BLAS dot products can lose substantial precision for large BF16
    output vectors. Accumulate elementwise products and squared norms in
    FP64 chunks so an unchanged output reports cosine one at actual shapes.
    This measures tensor error, not perceptual video quality.
    """
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate shapes differ")
    reference_cpu = reference.detach().cpu().reshape(-1)
    candidate_cpu = candidate.detach().cpu().reshape(-1)
    squared_reference = squared_candidate = squared_error = product = maximum_error = 0.0
    for start in range(0, reference_cpu.numel(), 1 << 20):
        a = reference_cpu[start:start + (1 << 20)].double()
        b = candidate_cpu[start:start + (1 << 20)].double()
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            return {"max_abs_error": float("inf"), "relative_rmse": float("inf"),
                    "cosine_similarity": float("nan")}
        difference = a - b
        squared_reference += a.square().sum().item()
        squared_candidate += b.square().sum().item()
        squared_error += difference.square().sum().item()
        product += (a * b).sum().item()
        maximum_error = max(maximum_error, difference.abs().max().item())
    if squared_reference > 0 and squared_candidate > 0:
        cosine = max(-1.0, min(1.0, product / math.sqrt(squared_reference * squared_candidate)))
    else:
        cosine = float(squared_error == 0)
    return {
        "max_abs_error": maximum_error,
        "relative_rmse": math.sqrt(squared_error / max(squared_reference, 1e-60)),
        "cosine_similarity": cosine,
    }


def _native_ops():
    import torch_npu

    return torch_npu


def _internal_format_policy(ops) -> str:
    raw_policy = ops._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    if isinstance(raw_policy, bytes):
        raw_policy = raw_policy.decode("ascii")
    if raw_policy not in ("enable", "disable"):
        raise RuntimeError(f"Cannot faithfully restore ALLOW_INTERNAL_FORMAT state: {raw_policy!r}")
    return raw_policy


def _convert_weight_to_nz(weight, ops):
    """One-time post-load layout conversion; restore the execution flag exactly.

    Model loading is serialized before serving begins. Timed forward execution
    retains its previous internal-format policy; this never toggles it per call.
    """
    # torch_npu 2.10 exposes a write-only config descriptor. Read the same
    # underlying C option its setter writes, and reject unknown state rather
    # than silently changing the serving execution policy.
    previous = _internal_format_policy(ops) == "enable"
    try:
        ops.npu.config.allow_internal_format = True
        converted = ops.npu_format_cast(weight, 29)
        if ops.get_npu_format(converted) != 29:
            raise RuntimeError("Requested FRACTAL_NZ conversion fell back to a base format; use ND weights")
        roundtrip = ops.npu_format_cast(converted, 2)
        if not torch.equal(weight, roundtrip):
            raise RuntimeError("FRACTAL_NZ layout conversion changed the INT8 weight payload")
        return converted
    finally:
        ops.npu.config.allow_internal_format = previous


def _matrix_weight(layer: nn.Module) -> torch.Tensor:
    return layer.weight if layer._lingbot_quant_weight_layout == "kn" else layer.weight.transpose(0, 1)


@torch.no_grad()
def _quantized_linear(layer: nn.Module, value: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    shape = value.shape
    if not shape or shape[-1] != layer._lingbot_quant_in_features:
        raise ValueError("Quantized linear input feature dimension does not match its weight")
    if value.device != layer.weight.device:
        raise ValueError("Quantized linear input and weight must be on the same device")
    matrix = value.reshape(-1, shape[-1])
    if getattr(layer, "_lingbot_tp_integer_reduce", False):
        return _tp_row_quantized_linear(layer, matrix, shape, value.dtype)
    weight = _matrix_weight(layer)
    scale = layer._lingbot_quant_weight_scale
    mode = layer._lingbot_quant_mode
    if matrix.shape[0] == 0:
        return value.new_empty((*shape[:-1], layer._lingbot_quant_out_features))
    if value.device.type == "cpu":
        # A numerical reference for CPU tests; never selected on an NPU failure.
        if mode == "w8a8":
            floating = matrix.float()
            maximum = floating.abs().amax(dim=1)
            activation_scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
            integers = torch.round(floating / activation_scale[:, None]).clamp(-127, 127).to(torch.int32)
            result = (integers @ weight.to(torch.int32)).float()
            result = result * activation_scale[:, None] * scale.float()[None, :]
        else:
            result = matrix.float() @ (weight.float() * scale.float()[None, :])
        if bias is not None:
            result = result + bias.float()
        result = result.to(value.dtype)
    elif value.device.type == "npu":
        if value.dtype not in {torch.bfloat16, torch.float16}:
            raise ValueError("A3 quantized linear activations must be BF16 or FP16")
        ops = _native_ops()
        if mode == "w8a8":
            integers, activation_scale = ops.npu_dynamic_quant(matrix, dst_type=torch.int8)
            result = ops.npu_quant_matmul(
                integers, weight, scale, pertoken_scale=activation_scale.reshape(-1),
                bias=bias, output_dtype=value.dtype,
            )
        else:
            native_bias = None
            if bias is not None:
                native_bias = layer._lingbot_quant_bias_fp32 if bias is layer.bias else bias.float()
            result = ops.npu_weight_quant_batchmatmul(
                matrix, weight, scale, bias=native_bias, antiquant_group_size=0,
            )
    else:
        raise ValueError("LingBot NPU quantization supports NPU execution and CPU reference tests only")
    return result.reshape(*shape[:-1], layer._lingbot_quant_out_features)


class NpuInt8LinearMethod:
    """Apply interface used by existing vLLM parallel Linear forwards."""

    uses_meta_device = False

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return _quantized_linear(layer, x, bias)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # The post-load installer already owns layout and scales. The ordinary
        # diffusion loader skips this non-QuantizeMethodBase adapter; callers
        # using duck typing must not transpose or quantize it a second time.
        return


def _nn_linear_forward(layer: nn.Module, value: torch.Tensor) -> torch.Tensor:
    return _quantized_linear(layer, value, layer.bias)


@torch.no_grad()
def quantized_linear_from_payload(
    layer: nn.Module, payload: torch.Tensor, activation_scale: torch.Tensor, *, output_dtype: torch.dtype,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    if os.environ.get("LINGBOT_CAPTURE_COMPUTE", "0") == "1":
        from vllm_omni.diffusion.layers.lingbot_compute_capture import compute_module_scope

        with compute_module_scope(layer, payload) as diagnostic:
            result = _quantized_linear_from_payload(layer, payload, activation_scale, output_dtype=output_dtype)
            if diagnostic is not None:
                diagnostic["output"] = result
            return result
    return _quantized_linear_from_payload(layer, payload, activation_scale, output_dtype=output_dtype)


def _quantized_linear_from_payload(
    layer: nn.Module, payload: torch.Tensor, activation_scale: torch.Tensor, *, output_dtype: torch.dtype,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """Consume a screened native-contract payload without quantizing it again.

    Used only by an explicitly enabled GELU+quant fusion. Bias handling matches
    both nn.Linear and vLLM skip_bias_add/return_bias, and output remains BF16.
    The caller must validate payload AND FP32 scales against npu_dynamic_quant.
    """
    if getattr(layer, "_lingbot_quant_mode", None) != "w8a8":
        raise ValueError("Prequantized payload requires an installed W8A8 linear")
    if getattr(layer, "_lingbot_tp_integer_reduce", False):
        raise ValueError("TP2 FFN requires a globally scaled activation; prequantized GELU payload is unsupported")
    if payload.dtype != torch.int8 or payload.shape[-1] != layer._lingbot_quant_in_features:
        raise ValueError("Prequantized payload must be INT8 with matching input features")
    if activation_scale.shape != payload.shape[:-1] or activation_scale.dtype != torch.float32:
        raise ValueError("Prequantized per-token scales must be FP32 and match leading dimensions")
    if payload.device != layer.weight.device or activation_scale.device != payload.device:
        raise ValueError("Payload, scales and W8A8 weights must share a device")
    if output_dtype not in {torch.bfloat16, torch.float16, torch.float32}:
        raise ValueError("Prequantized output must use the original floating dtype")
    shape = payload.shape
    matrix = payload.reshape(-1, shape[-1])
    weight = _matrix_weight(layer)
    bias = None if getattr(layer, "skip_bias_add", False) else layer.bias
    if payload.device.type == "cpu":
        result = (matrix.to(torch.int32) @ weight.to(torch.int32)).float()
        result = result * activation_scale.reshape(-1, 1) * layer._lingbot_quant_weight_scale[None, :]
        if bias is not None:
            result = result + bias.float()
        result = result.to(output_dtype)
    elif payload.device.type == "npu":
        if output_dtype not in {torch.bfloat16, torch.float16}:
            raise ValueError("A3 W8A8 fused output must remain BF16 or FP16")
        result = _native_ops().npu_quant_matmul(
            matrix, weight, layer._lingbot_quant_weight_scale,
            pertoken_scale=activation_scale.reshape(-1), bias=bias, output_dtype=output_dtype,
        )
    else:
        raise ValueError("Prequantized linear supports CPU reference and Ascend execution only")
    result = result.reshape(*shape[:-1], layer._lingbot_quant_out_features)
    if getattr(layer, "return_bias", False):
        return result, layer.bias if getattr(layer, "skip_bias_add", False) else None
    return result


def _tp_group():
    from vllm.distributed import get_tp_group

    return get_tp_group().device_group


def _tp_row_quantized_linear(layer, matrix, shape, dtype):
    """Reduce INT32 dot products before dequantization and the single bias add.

    Independent shard scales or BF16 partial reductions change the existing
    W8A8 contract. Shared activation/weight scales preserve the integer payload;
    the outer RowParallelLinear reduction is disabled for this path.
    """
    import torch.distributed as dist

    maximum = matrix.float().abs().amax(dim=1)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=_tp_group())
    activation_scale = torch.where(maximum > 0, maximum/127.0, torch.ones_like(maximum))
    integers = torch.round(matrix.float()/activation_scale[:, None]).clamp(-127, 127).to(torch.int8)
    weight = _matrix_weight(layer)
    if matrix.device.type == "cpu":
        accum = integers.to(torch.int32) @ weight.to(torch.int32)
    elif matrix.device.type == "npu":
        accum = _native_ops().npu_quant_matmul(integers, weight, layer._lingbot_quant_unit_scale,
                                              output_dtype=torch.int32)
    else:
        raise ValueError("TP W8A8 supports CPU reference and Ascend only")
    dist.all_reduce(accum, op=dist.ReduceOp.SUM, group=_tp_group())
    result = accum.float()*activation_scale[:, None]*layer._lingbot_quant_weight_scale[None, :]
    if not getattr(layer, "skip_bias_add", False) and layer.bias is not None:
        result = result+layer.bias.float()
    return result.to(dtype).reshape(*shape[:-1], layer._lingbot_quant_out_features)


@torch.no_grad()
def _install_on_linear(layer: nn.Module, config: NpuQuantizationConfig) -> dict[str, Any]:
    if hasattr(layer, "_lingbot_quant_mode"):
        raise ValueError("Linear is already quantized; restore or reload BF16 weights before changing mode")
    tp_size = int(getattr(layer, "tp_size", 1))
    if tp_size not in (1, 2) or (tp_size == 2 and config.mode != "w8a8"):
        raise ValueError("LingBot TP quantization supports TP1 or TP2 W8A8")
    original_weight = layer.weight
    if config.require_npu and original_weight.device.type != "npu":
        raise ValueError("LINGBOT_NPU_QUANT requires weights loaded on an NPU")
    if original_weight.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError("Online INT8 expects original BF16/FP16/FP32 weights, not an already quantized checkpoint")
    old_method = getattr(layer, "quant_method", None)
    if old_method is not None and "Unquantized" not in type(old_method).__name__:
        raise ValueError(f"Cannot replace an existing quantization method: {type(old_method).__name__}")
    out_features, in_features = original_weight.shape
    shared_maximum = None
    tp_row = tp_size == 2 and type(layer).__name__ == "RowParallelLinear"
    if tp_size == 2 and type(layer).__name__ not in ("RowParallelLinear", "ColumnParallelLinear"):
        raise ValueError("TP2 W8A8 requires reviewed row/column FFN linears")
    if tp_row:
        import torch.distributed as dist

        maximum = original_weight.detach().float().abs().amax(dim=1)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=_tp_group())
        shared_maximum = maximum.cpu()
    qweight, scale, errors = quantize_weight_per_channel(original_weight, shared_maximum=shared_maximum)
    device = original_weight.device
    layout = "kn" if config.mode == "w8a8" else "nk"
    if layout == "kn":
        qweight = qweight.transpose(0, 1).contiguous()
    qweight = qweight.to(device)
    effective_weight_format = config.weight_format
    if tp_row and device.type == "npu":
        # This installed INT32 QuantMatmul path interprets FRACTAL_NZ input
        # through the newer affinity-weight ABI. Its ordinary ND path accepts
        # the original KN integers. Allocate a real ND tensor at startup;
        # changing a shared storage's format label would corrupt its owners.
        qweight = _native_ops().npu_format_cast(qweight, 2)
        effective_weight_format = "nd"
    elif config.weight_format == "nz":
        if device.type != "npu":
            raise ValueError("FRACTAL_NZ format requires an NPU")
        ops = _native_ops()
        qweight = _convert_weight_to_nz(qweight, ops)
    scale_dtype = torch.float32 if config.mode == "w8a8" else original_weight.dtype
    scale = scale.to(device=device, dtype=scale_dtype)
    if tp_row:
        layer._lingbot_tp_integer_reduce = True
        layer._lingbot_quant_unit_scale = torch.ones(out_features, device=device, dtype=torch.float32)
        layer._lingbot_bf16_reduce_results = layer.reduce_results
        layer.reduce_results = False
    if config.keep_bf16_cpu:
        layer._lingbot_bf16_weight_cpu = original_weight.detach().cpu().clone()
    layer._lingbot_bf16_dtype = original_weight.dtype
    layer._lingbot_bf16_requires_grad = original_weight.requires_grad
    layer._lingbot_bf16_quant_method = old_method
    layer._lingbot_bf16_custom_op = getattr(layer, "custom_op", None)
    layer._lingbot_quant_in_features = in_features
    layer._lingbot_quant_out_features = out_features
    layer._lingbot_quant_mode = config.mode
    layer._lingbot_quant_weight_layout = layout
    layer.register_buffer("_lingbot_quant_weight_scale", scale, persistent=False)
    if config.mode == "w8a16" and getattr(layer, "bias", None) is not None:
        # Installed CANN9.1 WeightQuantBatchMatmulV2 requires FLOAT bias for
        # this BF16 path. Cache it once, retaining vLLM's optional-bias contract.
        layer.register_buffer("_lingbot_quant_bias_fp32", layer.bias.detach().float(), persistent=False)
    original_weight.requires_grad_(False)
    original_weight.data = qweight
    if old_method is not None:
        layer.quant_method = NpuInt8LinearMethod()
        if hasattr(layer, "custom_op"):
            layer.custom_op = None
    elif isinstance(layer, nn.Linear):
        layer._lingbot_bf16_forward = layer.forward
        layer.forward = MethodType(_nn_linear_forward, layer)
    else:
        raise ValueError("Selected module is neither an nn.Linear nor a vLLM parallel Linear")
    return {
        "shape_out_in": [out_features, in_features],
        "mode": config.mode, "weight_format": effective_weight_format,
        "requested_weight_format": config.weight_format,
        "original_weight_bytes": out_features * in_features * torch.tensor([], dtype=layer._lingbot_bf16_dtype).element_size(),
        "quantized_weight_and_scale_bytes": qweight.numel() * qweight.element_size() + scale.numel() * scale.element_size(),
        **errors,
    }


@torch.no_grad()
def install_lingbot_npu_quantization(
    transformer: nn.Module, config: NpuQuantizationConfig | None = None,
) -> dict[str, Any]:
    """Post-load hook; parameter names survive checkpoint completeness checks.

    Changing quantization mode on an already loaded quantized model requires
    restoring retained CPU BF16 weights or reloading the original checkpoint.
    The loader may call the hook more than once with the same configuration.
    """
    config = NpuQuantizationConfig.from_env() if config is None else config
    previous = getattr(transformer, "_lingbot_npu_quantization_config", None)
    if previous is not None:
        if previous != config:
            raise ValueError("Reload or restore the BF16 model before changing quantization configuration")
        return transformer._lingbot_npu_quantization_summary
    if config.mode == "off":
        return {"config": asdict(config), "layers": [], "selected_layers": 0}
    candidates = [
        (name, layer) for name, layer in transformer.named_modules()
        if selected_linear(name, config.scope, projections=config.projections,
                           allowlist=config.allowlist, exclude=config.exclude)
    ]
    if not candidates:
        raise ValueError("No LingBot block linears matched the requested quantization scope")
    # Validate the complete selection before mutating any weights.
    for name, layer in candidates:
        if not isinstance(getattr(layer, "weight", None), torch.Tensor) or layer.weight.ndim != 2:
            raise ValueError(f"Selected LingBot module has no 2D weight: {name}")
        tp_size = int(getattr(layer, "tp_size", 1))
        if tp_size != 1 and (tp_size != 2 or config.mode != "w8a8" or config.scope != "ffn"
                            or type(layer).__name__ not in ("RowParallelLinear", "ColumnParallelLinear")):
            raise ValueError("TP2 W8A8 is restricted to the existing FFN scope")
        if config.require_npu and layer.weight.device.type != "npu":
            raise ValueError(f"Selected weight is not loaded on NPU: {name}")
        old_method = getattr(layer, "quant_method", None)
        if old_method is not None and "Unquantized" not in type(old_method).__name__:
            raise ValueError(f"Selected layer already uses quantization: {name}")
        if old_method is None and not isinstance(layer, nn.Linear):
            raise ValueError(f"Selected module is not a supported linear: {name}")
        if layer.weight.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError(f"Selected weight is already quantized or has an unsupported dtype: {name}")
        if config.weight_format == "nz" and layer.weight.device.type != "npu":
            raise ValueError("FRACTAL_NZ format requires an NPU")
    has_npu_weights = any(layer.weight.device.type == "npu" for _, layer in candidates)
    execution_policy_before = _internal_format_policy(_native_ops()) if has_npu_weights else None
    layers = [{"name": name, **_install_on_linear(layer, config)} for name, layer in candidates]
    execution_policy_after = _internal_format_policy(_native_ops()) if has_npu_weights else None
    if execution_policy_after != execution_policy_before:
        raise RuntimeError("Quantization installation changed ALLOW_INTERNAL_FORMAT execution policy")
    error = sum(row["weight_squared_error"] for row in layers)
    reference = sum(row["weight_squared_reference"] for row in layers)
    summary = {
        "config": asdict(config), "layers": layers, "selected_layers": len(layers),
        "original_weight_bytes": sum(row["original_weight_bytes"] for row in layers),
        "quantized_weight_and_scale_bytes": sum(row["quantized_weight_and_scale_bytes"] for row in layers),
        "global_weight_relative_rmse": math.sqrt(error / max(reference, 1e-30)),
        "quality_semantics": "Weight reconstruction error only; activation/latent/video quality requires measured validation.",
        "protected_conditioning_and_head": config.scope != "all",
        "selected_names": [name for name, _ in candidates],
        "activation_output_contract": "BF16/FP16 outputs; Q/K/V/cache/attention dtype unchanged",
        "allow_internal_format_option_before": execution_policy_before,
        "allow_internal_format_option_after": execution_policy_after,
        "allow_internal_format_option_preserved": execution_policy_before == execution_policy_after,
        "allow_internal_format_option_source": "torch_npu._C._npu_getOption(ALLOW_INTERNAL_FORMAT)" if has_npu_weights else "CPU path: not read",
    }
    if config.scope == "all":
        logger.warning("Legacy LingBot quantization scope=all includes camera/text K/V; use controlled hot scopes for realtime")
    transformer._lingbot_npu_quantization_config = config
    transformer._lingbot_npu_quantization_summary = summary
    if has_npu_weights:
        # KV budgeting must see free BF16 storage after online conversion,
        # rather than the old blocks retained by the caching allocator.
        gc.collect()
        _native_ops().npu.synchronize()
        _native_ops().npu.empty_cache()
    logger.warning(
        "LingBot experimental %s quantized %d %s linears; weight relative RMSE %.6g; storage %.3f -> %.3f GiB",
        config.mode, len(layers), config.scope, summary["global_weight_relative_rmse"],
        summary["original_weight_bytes"] / 2**30, summary["quantized_weight_and_scale_bytes"] / 2**30,
    )
    if has_npu_weights:
        logger.warning("LingBot quantization ALLOW_INTERNAL_FORMAT before=%s after=%s preserved=%s weight_format=%s",
                       execution_policy_before, execution_policy_after, execution_policy_before == execution_policy_after,
                       config.weight_format)
    return summary


@torch.no_grad()
def restore_lingbot_bf16(transformer: nn.Module) -> None:
    """Restore exact retained CPU weights; otherwise require checkpoint reload."""
    layers = [layer for layer in transformer.modules() if hasattr(layer, "_lingbot_quant_mode")]
    if any(not hasattr(layer, "_lingbot_bf16_weight_cpu") for layer in layers):
        raise ValueError("Exact BF16 weights were not retained; reload the unchanged original checkpoint")
    for layer in layers:
        if getattr(layer, "_lingbot_tp_integer_reduce", False):
            layer.reduce_results = layer._lingbot_bf16_reduce_results
            del layer._lingbot_tp_integer_reduce
        layer.weight.data = layer._lingbot_bf16_weight_cpu.to(layer.weight.device)
        layer.weight.requires_grad_(layer._lingbot_bf16_requires_grad)
        if layer._lingbot_bf16_quant_method is not None:
            layer.quant_method = layer._lingbot_bf16_quant_method
            if hasattr(layer, "custom_op"):
                layer.custom_op = layer._lingbot_bf16_custom_op
        else:
            layer.forward = layer._lingbot_bf16_forward
            del layer._lingbot_bf16_forward
        for attr in ("_lingbot_quant_weight_scale", "_lingbot_quant_bias_fp32"):
            layer._buffers.pop(attr, None)
        for attr in list(vars(layer)):
            if attr.startswith(("_lingbot_quant_", "_lingbot_bf16_")):
                delattr(layer, attr)
    for attr in ("_lingbot_npu_quantization_config", "_lingbot_npu_quantization_summary"):
        if hasattr(transformer, attr):
            delattr(transformer, attr)


class NpuQuantLinear(nn.Linear):
    """Standalone adapter for isolated kernel/numerical tests at actual shapes."""

    @classmethod
    def from_linear(
        cls, layer: nn.Linear, mode: str = "w8a8", *, weight_format: str = "nd",
    ) -> NpuQuantLinear:
        if mode not in {"w8a8", "w8a16"}:
            raise ValueError("Standalone quantized linear mode must be w8a8 or w8a16")
        result = cls(layer.in_features, layer.out_features, bias=layer.bias is not None, device="meta")
        result.weight = nn.Parameter(layer.weight.detach(), requires_grad=False)
        result.bias = None if layer.bias is None else nn.Parameter(layer.bias.detach(), requires_grad=False)
        _install_on_linear(result, NpuQuantizationConfig(mode=mode, weight_format=weight_format, require_npu=False))
        return result
