# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Matrix per-tensor FP8 scaling through vLLM quantization and GEMM kernels."""

import torch
from torch import nn
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import PerTensorTorchFP8ScaledMMLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import kFp8DynamicTensorSym, kFp8StaticTensorSym


class MatrixFP8Method(LinearMethodBase):
    """Keep Q/K/V weight scales separate and quantize their shared input once."""

    def __init__(self, layer: LinearBase, partitions: int = 1):
        self.partitions = partitions
        rows, cols = layer.weight.shape
        self.width = rows // partitions
        config = FP8ScaledMMLinearLayerConfig(
            weight_quant_key=kFp8StaticTensorSym,
            activation_quant_key=kFp8DynamicTensorSym,
            weight_shape=(self.width, cols),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
        )
        self.kernel = PerTensorTorchFP8ScaledMMLinearKernel(
            config, ("weight", "weight_scale", "input_scale", "input_scale_ub")
        )
        # Quantize each released matrix separately, preserving its CPU FP32
        # scale calculation. Embeddings and the output head remain BF16.
        matrices, scales = [], []
        for weight in layer.weight.detach().cpu().float().chunk(partitions):
            scale = weight.abs().max().clamp_min(1 / 512) / 448
            matrices.append((weight / scale).to(torch.float8_e4m3fn))
            scales.append(scale)
        device = layer.weight.device
        layer.weight = nn.Parameter(torch.cat(matrices).to(device).t(), requires_grad=False)
        layer.register_buffer("weight_scale", torch.stack(scales).to(device))

    def create_weights(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("Matrix FP8 conversion follows strict BF16 checkpoint loading")

    def apply(self, layer: LinearBase, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        quantized, scale = self.kernel.quant_fp8(x.reshape(-1, x.shape[-1]))
        outputs = []
        for index in range(self.partitions):
            start, stop = index * self.width, (index + 1) * self.width
            outputs.append(
                self.kernel.apply_scaled_mm(
                    A=quantized,
                    B=layer.weight[:, start:stop],
                    out_dtype=torch.bfloat16,
                    As=scale,
                    Bs=layer.weight_scale[index : index + 1],
                    bias=None if bias is None else bias[start:stop],
                    output_shape=[*x.shape[:-1], self.width],
                )
            )
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)


def quantize_blocks(model: nn.Module) -> None:
    for block in model.blocks:
        block.attn1.to_qkv.quant_method = MatrixFP8Method(block.attn1.to_qkv, 3)
        for layer in (
            block.attn1.to_out,
            block.attn2.to_q,
            block.attn2.to_k,
            block.attn2.to_v,
            block.attn2.to_out,
            block.ffn.net_0.proj,
            block.ffn.net_2,
        ):
            layer.quant_method = MatrixFP8Method(layer)
