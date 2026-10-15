# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-tensor FP8 for loaded linear weights with independent output partitions."""

from collections.abc import Sequence

import torch
from torch import nn
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import PerTensorTorchFP8ScaledMMLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTensorSym, kFp8StaticTensorSym
from vllm.platforms import current_platform


class PartitionedFP8LinearMethod(LinearMethodBase):
    """Convert a loaded BF16/FP16 linear without merging its weight scales.

    Each output partition keeps its own weight scale and GEMM; all partitions
    share one dynamic per-tensor input quantization. The upstream Torch kernel
    dispatches scaled-MM on the active device. No GPU architecture or model
    geometry is selected here. Use this when preserving separate projection
    scales is required; ordinary online FP8 should use the existing quantization
    config instead.

    Input and partition widths must be multiples of 16 for Torch scaled-MM.
    Conversion runs once, after checkpoint loading. ``min_amax`` lets callers
    preserve their checkpoint's scale-floor convention.
    """

    def __init__(
        self,
        layer: LinearBase,
        output_partition_sizes: Sequence[int] | None = None,
        *,
        min_amax: float = 1e-12,
    ) -> None:
        rows, cols = layer.weight.shape
        self.output_partition_sizes = tuple(output_partition_sizes) if output_partition_sizes is not None else (rows,)
        if (
            not self.output_partition_sizes
            or any(type(size) is not int or size <= 0 for size in self.output_partition_sizes)
            or sum(self.output_partition_sizes) != rows
        ):
            raise ValueError("FP8 output partition sizes must be positive integers summing to the output width")
        if cols % 16 or any(size % 16 for size in self.output_partition_sizes):
            raise ValueError("FP8 input and output partition widths must be multiples of 16")
        if layer.weight.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FP8 conversion requires loaded BF16 or FP16 weights")
        if not 0 < min_amax < float("inf"):
            raise ValueError("min_amax must be positive and finite")
        supported, reason = PerTensorTorchFP8ScaledMMLinearKernel.is_supported()
        if not supported:
            raise ValueError(f"Per-tensor FP8 scaled-MM is unavailable on this platform: {reason}")
        self.out_dtype = layer.weight.dtype
        config = FP8ScaledMMLinearLayerConfig(
            weight_quant_key=kFp8StaticTensorSym,
            activation_quant_key=kFp8DynamicTensorSym,
            weight_shape=(self.output_partition_sizes[0], cols),
            input_dtype=self.out_dtype,
            out_dtype=self.out_dtype,
        )
        self.kernel = PerTensorTorchFP8ScaledMMLinearKernel(
            config, ("weight", "weight_scale", "input_scale", "input_scale_ub")
        )
        fp8_dtype = current_platform.fp8_dtype()
        matrices, scales = [], []
        # Preserve CPU FP32 scale calculation independently of the device GEMM.
        for weight in layer.weight.detach().cpu().float().split(self.output_partition_sizes):
            scale = weight.abs().max().clamp_min(min_amax) / torch.finfo(fp8_dtype).max
            matrices.append((weight / scale).to(fp8_dtype))
            scales.append(scale)
        device = layer.weight.device
        layer.weight = nn.Parameter(torch.cat(matrices).to(device).t(), requires_grad=False)
        layer.register_buffer("weight_scale", torch.stack(scales).to(device))

    def create_weights(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("Partitioned FP8 conversion requires weights to be loaded first")

    def apply(self, layer: LinearBase, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        quantized, scale = self.kernel.quant_fp8(x.reshape(-1, x.shape[-1]))
        outputs = []
        start = 0
        for index, width in enumerate(self.output_partition_sizes):
            stop = start + width
            outputs.append(
                self.kernel.apply_scaled_mm(
                    A=quantized,
                    B=layer.weight[:, start:stop],
                    out_dtype=self.out_dtype,
                    As=scale,
                    Bs=layer.weight_scale[index : index + 1],
                    bias=None if bias is None else bias[start:stop],
                    output_shape=[*x.shape[:-1], width],
                )
            )
            start = stop
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
