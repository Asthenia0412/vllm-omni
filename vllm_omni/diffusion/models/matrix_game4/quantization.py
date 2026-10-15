# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Choose Matrix block projections for the shared partitioned FP8 method."""

from torch import nn

from vllm_omni.quantization.partitioned_fp8 import PartitionedFP8LinearMethod


def quantize_blocks(model: nn.Module) -> None:
    for block in model.blocks:
        qkv = block.attn1.to_qkv
        qkv.quant_method = PartitionedFP8LinearMethod(qkv, qkv.output_partition_sizes, min_amax=1 / 512)
        for layer in (
            block.attn1.to_out,
            block.attn2.to_q,
            block.attn2.to_k,
            block.attn2.to_v,
            block.attn2.to_out,
            block.ffn.net_0.proj,
            block.ffn.net_2,
        ):
            layer.quant_method = PartitionedFP8LinearMethod(layer, min_amax=1 / 512)
