# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Attention dispatch contracts without requiring an Ascend device."""

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.layers import lingbot_npu_ops as ops


def _inputs(length=7):
    return (
        torch.randn(1, 3, 2, 16, dtype=torch.bfloat16),
        torch.randn(1, length, 2, 16, dtype=torch.bfloat16),
        torch.randn(1, length, 2, 16, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("history_length", [1, 7, 19])
def test_inference_dispatch_uses_actual_visible_history(monkeypatch, history_length):
    inputs = _inputs(history_length)
    calls = []

    def infer(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return q.clone(), None

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_fused_infer_attention_score=infer))
    monkeypatch.setenv("VLLM_LINGBOT_NPU_INFER_ATTENTION", "1")
    ops._INFER_FAILURES.clear()
    result = ops.npu_attention(*inputs, head_num=2, scale=.25)
    q, k, v, kwargs = calls[0]
    assert result.shape == inputs[0].shape
    assert k.shape[1] == history_length and v.shape[1] == history_length
    assert all(t.is_contiguous() for t in (q, k, v))
    assert kwargs == dict(
        num_heads=2, num_key_value_heads=2, input_layout="BSND", scale=.25,
        sparse_mode=0, inner_precise=0, softmax_lse_flag=False,
    )


def test_unsupported_history_falls_back_without_disabling_other_lengths(monkeypatch):
    calls = []

    def infer(q, k, v, **kwargs):
        calls.append(("infer", k.shape[1]))
        if k.shape[1] == 7:
            raise NotImplementedError("test capability limitation")
        return q, None

    def fusion(q, k, v, **kwargs):
        calls.append(("fusion", k.shape[1]))
        assert kwargs["keep_prob"] == 1 and kwargs["sparse_mode"] == 0
        return q, None

    monkeypatch.setitem(
        sys.modules, "torch_npu",
        SimpleNamespace(npu_fused_infer_attention_score=infer, npu_fusion_attention=fusion),
    )
    monkeypatch.setenv("VLLM_LINGBOT_NPU_INFER_ATTENTION", "1")
    ops._INFER_FAILURES.clear()
    for length in (7, 7, 19):
        ops.npu_attention(*_inputs(length), head_num=2, scale=.25)
    assert calls == [("infer", 7), ("fusion", 7), ("fusion", 7), ("infer", 19)]
    ops._INFER_FAILURES.clear()


def test_gate_off_preserves_fusion_parameters(monkeypatch):
    calls = []
    monkeypatch.setenv("VLLM_LINGBOT_NPU_INFER_ATTENTION", "0")
    monkeypatch.setitem(
        sys.modules, "torch_npu",
        SimpleNamespace(npu_fusion_attention=lambda q, k, v, **kw: (calls.append(kw) or q, None)),
    )
    ops.npu_attention(*_inputs(), head_num=2, scale=.25)
    assert calls[0]["pre_tockens"] == calls[0]["next_tockens"] == 2147483647
    assert calls[0]["input_layout"] == "BSND"


@pytest.mark.parametrize("error", ["heads", "dtype", "length", "shape"])
def test_invalid_attention_contract_is_rejected(error):
    q, k, v = _inputs()
    heads = 2
    if error == "heads":
        heads = 4
    elif error == "dtype":
        k = k.float()
    elif error == "length":
        k, v = k[:, :0], v[:, :0]
    elif error == "shape":
        v = v[..., :-1]
    with pytest.raises(ValueError):
        ops.npu_attention(q, k, v, head_num=heads, scale=.25)
