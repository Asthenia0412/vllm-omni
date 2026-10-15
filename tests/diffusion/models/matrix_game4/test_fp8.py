# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed Matrix QKV retains independent FP8 scales and one input quantization."""

from unittest.mock import patch

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import ReplicatedLinear

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.matrix_game4.quantization import MatrixFP8Method
from vllm_omni.diffusion.models.matrix_game4.transformer import MatrixQKV

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@torch.inference_mode()
def test_fp8_qkv_preserves_independent_projection_scales(monkeypatch):
    # This unit test loads whole matrices and does not exercise collectives.
    for module in ("vllm.model_executor.parameter", "vllm.model_executor.layers.linear"):
        monkeypatch.setattr(f"{module}.get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(f"{module}.get_tensor_model_parallel_world_size", lambda: 1)
    with set_current_vllm_config(VllmConfig()):
        torch.manual_seed(42)
        packed = MatrixQKV(64, 64, 1, bias=True, params_dtype=torch.bfloat16, disable_tp=True).to("cuda")
        projections = [ReplicatedLinear(64, 64, bias=True, params_dtype=torch.bfloat16).to("cuda") for _ in range(3)]
        for index, layer in enumerate(projections):
            layer.weight.copy_(torch.randn_like(layer.weight) * (0.01, 1.0, 100.0)[index])
            layer.bias.copy_(torch.randn_like(layer.bias))
        packed.weight.copy_(torch.cat([layer.weight for layer in projections]))
        packed.bias.copy_(torch.cat([layer.bias for layer in projections]))
        packed.quant_method = MatrixFP8Method(packed, partitions=3)
        for layer in projections:
            layer.quant_method = MatrixFP8Method(layer)
        assert packed.weight_scale.unique().numel() == 3
        inputs = torch.randn(1, 128, 64, device="cuda", dtype=torch.bfloat16)
        expected = torch.cat([layer(inputs)[0] for layer in projections], dim=-1)
        kernel = packed.quant_method.kernel
        with patch.object(kernel, "quant_fp8", wraps=kernel.quant_fp8) as quantize:
            actual, _ = packed(inputs)
        assert quantize.call_count == 1
        assert torch.equal(actual, expected)
