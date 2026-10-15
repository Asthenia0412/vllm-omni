# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-independent partition geometry, precision and capability checks."""

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import ReplicatedLinear

from tests.helpers.mark import hardware_test
from vllm_omni.quantization.partitioned_fp8 import PartitionedFP8LinearMethod

pytestmark = [pytest.mark.core_model]


@pytest.fixture(autouse=True)
def single_rank_linears(monkeypatch):
    # The test constructs resident weights directly, without a process group.
    for module in ("vllm.model_executor.parameter", "vllm.model_executor.layers.linear"):
        monkeypatch.setattr(f"{module}.get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(f"{module}.get_tensor_model_parallel_world_size", lambda: 1)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "input_size, output_size, partitions, dtype, min_amax, error",
    [
        (64, 128, (), torch.bfloat16, 1e-12, "positive integers"),
        (64, 128, (64,), torch.bfloat16, 1e-12, "summing"),
        (64, 128, (0, 128), torch.bfloat16, 1e-12, "positive integers"),
        (64, 128, (-64, 192), torch.bfloat16, 1e-12, "positive integers"),
        (64, 128, (64.0, 64.0), torch.bfloat16, 1e-12, "positive integers"),
        (63, 128, None, torch.bfloat16, 1e-12, "multiples of 16"),
        (64, 128, (63, 65), torch.bfloat16, 1e-12, "multiples of 16"),
        (64, 128, None, torch.float32, 1e-12, "BF16 or FP16"),
        (64, 128, None, torch.bfloat16, 0, "positive and finite"),
        (64, 128, None, torch.bfloat16, float("nan"), "positive and finite"),
        (64, 128, None, torch.bfloat16, float("inf"), "positive and finite"),
    ],
)
def test_invalid_conversion_does_not_replace_weights(input_size, output_size, partitions, dtype, min_amax, error):
    with set_current_vllm_config(VllmConfig()):
        layer = ReplicatedLinear(input_size, output_size, params_dtype=dtype)
        weight = layer.weight
        with pytest.raises(ValueError, match=error):
            PartitionedFP8LinearMethod(layer, partitions, min_amax=min_amax)
        assert layer.weight is weight


@pytest.mark.cpu
def test_unsupported_platform_reports_kernel_reason(mocker):
    mocker.patch(
        "vllm_omni.quantization.partitioned_fp8.PerTensorTorchFP8ScaledMMLinearKernel.is_supported",
        return_value=(False, "test device lacks FP8 scaled-MM"),
    )
    with set_current_vllm_config(VllmConfig()):
        layer = ReplicatedLinear(64, 64, params_dtype=torch.bfloat16)
        weight = layer.weight
        with pytest.raises(ValueError, match="test device lacks FP8 scaled-MM"):
            PartitionedFP8LinearMethod(layer)
        assert layer.weight is weight


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("widths", [(64,), (32, 64, 96)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("leading_shape", [(17,), (2, 3)])
@pytest.mark.usefixtures("supported_fp8_device")
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_partitioned_linear_matches_independent_linears(dtype, widths, bias, leading_shape, mocker):
    with set_current_vllm_config(VllmConfig()):
        torch.manual_seed(42)
        layer = ReplicatedLinear(64, sum(widths), bias=bias, params_dtype=dtype).to("cuda")
        separate = [ReplicatedLinear(64, width, bias=bias, params_dtype=dtype).to("cuda") for width in widths]
        for index, projection in enumerate(separate):
            projection.weight.copy_(torch.randn_like(projection.weight) * 10**index)
            if bias:
                projection.bias.copy_(torch.randn_like(projection.bias))
        layer.weight.copy_(torch.cat([projection.weight for projection in separate]))
        if bias:
            layer.bias.copy_(torch.cat([projection.bias for projection in separate]))
        layer.quant_method = PartitionedFP8LinearMethod(layer, widths)
        for projection in separate:
            projection.quant_method = PartitionedFP8LinearMethod(projection)
        inputs = torch.randn(*leading_shape, 64, device="cuda", dtype=dtype)
        expected = torch.cat([projection(inputs)[0] for projection in separate], dim=-1)
        quantize = mocker.spy(layer.quant_method.kernel, "quant_fp8")
        actual, _ = layer(inputs)
        assert quantize.call_count == 1
        assert actual.dtype == dtype
        assert actual.shape == (*leading_shape, sum(widths))
        assert torch.equal(actual, expected)


@pytest.fixture
def supported_fp8_device():
    from vllm.model_executor.kernels.linear.scaled_mm.pytorch import PerTensorTorchFP8ScaledMMLinearKernel

    supported, reason = PerTensorTorchFP8ScaledMMLinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)
