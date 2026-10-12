# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU fusion contracts including protected precision and broadcast boundaries."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_path = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/layers/lingbot_npu_fusion.py"
_spec = importlib.util.spec_from_file_location("lingbot_precision_fusion_tests", _path)
fusion = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fusion
_spec.loader.exec_module(fusion)


def test_default_fusion_off_and_explicit_region_validation():
    config = fusion.PrecisionFusionConfig.from_sources({}, {})
    assert config.backend == "off"
    assert not config.enabled("qknorm_rope", torch.ones(1))
    enabled = fusion.PrecisionFusionConfig.from_sources({
        "lingbot_npu_fusion_backend": "triton", "lingbot_npu_fusion_regions": ["qknorm_rope", "gelu_quant"],
    }, {})
    assert enabled.regions == ("qknorm_rope", "gelu_quant")
    with pytest.raises(ValueError, match="explicit enabled"):
        fusion.PrecisionFusionConfig(backend="off", regions=("gelu_quant",))
    with pytest.raises(ValueError, match="unique"):
        fusion.PrecisionFusionConfig(backend="triton", regions=("qknorm_rope", "qknorm_rope"))


def test_full_5120_feature_qk_norm_and_intermediate_bf16_rounding():
    torch.manual_seed(243)
    qkv = torch.randn(1, 5, 15360, dtype=torch.bfloat16)
    wq, wk = torch.randn(5120), torch.randn(5120)
    c, s = torch.randn(5, 64), torch.randn(5, 64)
    actual = fusion.qkv_norm_rope(qkv, wq, wk, c, s, heads=40, eps=1e-6)
    source = qkv.split(5120, -1)
    expected = []
    for x, weight in zip(source[:2], (wq, wk)):
        xf = x.float()
        normalized = (xf / torch.sqrt(xf.square().sum(-1, keepdim=True) / 5120 + 1e-6) * weight).to(x.dtype)
        even, odd = normalized.float().reshape(1, 5, 40, 64, 2).unbind(-1)
        rotated = torch.stack((even*c[None, :, None]-odd*s[None, :, None],
                               even*s[None, :, None]+odd*c[None, :, None]), -1).flatten(-2).to(x.dtype)
        expected.append(rotated)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[2].flatten(-2), source[2])
    assert actual[0].shape == (1, 5, 40, 128)
    # Per-head RMSNorm changes the operation and must never substitute for full
    # feature norm merely because the attention kernel consumes head-sharded data.
    x = source[0].float().reshape(1, 5, 40, 128)
    per_head = x / torch.sqrt(x.square().mean(-1, keepdim=True)+1e-6)
    full_feature = x / torch.sqrt(x.square().mean((-1, -2), keepdim=True)+1e-6)
    assert not torch.equal(per_head.to(torch.bfloat16), full_feature.to(torch.bfloat16))


@pytest.mark.parametrize("frames", [1, 3, 6])
def test_layernorm_modulation_frame_and_sp_token_broadcast(frames):
    torch.manual_seed(43)
    value = torch.randn(2, 6, 5120, dtype=torch.bfloat16)
    scale, shift = torch.randn(2, frames, 1, 5120), torch.randn(2, frames, 1, 5120)
    actual = fusion.layernorm_modulation(value, scale, shift, eps=1e-6)
    reference = F.layer_norm(value.float(), (5120,), eps=1e-6).reshape(2, frames, 6//frames, 5120)
    reference = (reference * (1+scale) + shift).reshape(value.shape).to(value.dtype)
    assert torch.equal(actual, reference)
    assert actual.dtype == torch.bfloat16


def test_gelu_rounds_before_quant_and_keeps_fp32_scales():
    torch.manual_seed(531)
    value = torch.randn(3, 13824, dtype=torch.bfloat16) * 0.5
    payload, scales = fusion.gelu_quant(value)
    expected_payload, expected_scales = fusion.per_token_quant_reference(F.gelu(value.float(), approximate="tanh").bfloat16())
    assert torch.equal(payload, expected_payload)
    assert torch.equal(scales, expected_scales)
    unrounded_payload, unrounded_scales = fusion.per_token_quant_reference(F.gelu(value.float(), approximate="tanh"))
    assert not torch.equal(scales, unrounded_scales)
    assert bool((payload != unrounded_payload).any())


def test_quant_zero_tie_tiny_and_saturation_contract():
    value = torch.tensor([[0., 0., 0., 0.], [127., 0.5, 1.5, -2.5],
                          [1e-15, -1e-15, 0., 0.]])
    payload, scales = fusion.per_token_quant_reference(value)
    assert torch.equal(payload[0], torch.zeros(4, dtype=torch.int8))
    assert scales[0] == 1
    assert payload[1].tolist() == [127, 0, 2, -2]
    assert payload[2].tolist() == [127, -127, 0, 0]
    assert scales.dtype == torch.float32


@pytest.mark.parametrize("gamma_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_present", [False, True])
def test_vae_npu_rms_contract_rounds_norm_before_bias_and_silu(gamma_dtype, bias_present):
    torch.manual_seed(52)
    value = torch.randn(1, 32, 2, 3, 5, dtype=torch.bfloat16)
    gamma = torch.randn(32, dtype=gamma_dtype)
    bias = torch.randn(32) if bias_present else None
    actual = fusion.vae_rms_silu(value, gamma, bias, eps=1e-6, scale=32**0.5, source_contract="npu_rms_norm")
    xf = value.float()
    norm_output = (xf / torch.sqrt(xf.square().mean(1, keepdim=True)+1e-6)
                   * gamma.float()[None, :, None, None, None]).bfloat16()
    if bias is not None:
        norm_output = norm_output + bias[None, :, None, None, None]
    assert torch.equal(actual, F.silu(norm_output))
    assert actual.dtype == (torch.float32 if bias_present else torch.bfloat16)


def test_vae_l2_epsilon_is_not_rms_epsilon_and_promotion_is_preserved():
    value = torch.full((1, 4, 1, 2, 3), 1e-5, dtype=torch.bfloat16)
    gamma = torch.ones(4)
    l2 = fusion.vae_rms_silu(value, gamma, None, eps=1e-6, scale=2, source_contract="l2_normalize")
    rms = fusion.vae_rms_silu(value, gamma, None, eps=1e-6, scale=2, source_contract="npu_rms_norm")
    assert l2.dtype == torch.float32
    assert rms.dtype == torch.bfloat16
    assert l2.mean() > rms.float().mean() * 10


def test_invalid_full_norm_weights_and_modulation_frames_fail_closed():
    with pytest.raises(ValueError, match="full features"):
        fusion.qkv_norm_rope(torch.randn(1, 2, 15360), torch.ones(128), torch.ones(128),
                             torch.ones(2, 64), torch.ones(2, 64), heads=40, eps=1e-6)
    with pytest.raises(ValueError, match="do not align"):
        fusion.layernorm_modulation(torch.randn(1, 5, 32), torch.ones(1, 3, 1, 32),
                                     torch.ones(1, 3, 1, 32), eps=1e-6)
