"""CPU checks for NPU page gathering, without importing or running NPU kernels."""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    _npu_paged_attention,
    _reference_paged_attention,
)


def test_npu_gather_respects_page_order_and_excludes_padding(monkeypatch):
    calls = []

    def fused(q, k, v, **kwargs):
        calls.append(kwargs)
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=kwargs['scale']
        ).transpose(1, 2)
        return (output,)

    monkeypatch.setitem(sys.modules, 'torch_npu', types.SimpleNamespace(npu_fusion_attention=fused))
    query = torch.randn(12, 2, 8)
    key = torch.randn(4, 16, 2, 8)
    value = torch.randn_like(key)
    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    offsets = torch.tensor([0, 5, 12], dtype=torch.int32)
    lengths = torch.tensor([19, 27], dtype=torch.int32)
    key[0, 3:] = 1000
    value[0, 3:] = 1000
    key[1, 11:] = 1000
    value[1, 11:] = 1000
    args = (query, key, value, table, offsets, lengths, 8 ** -0.5)
    expected = _reference_paged_attention(*args, causal=False)
    actual = _npu_paged_attention(*args, causal=False)
    torch.testing.assert_close(actual, expected)
    assert len(calls) == 2
    assert all(call['input_layout'] == 'BSND' and call['sparse_mode'] == 0 for call in calls)


def test_npu_gather_rejects_unsupported_causal_mode():
    with pytest.raises(NotImplementedError):
        _npu_paged_attention(None, None, None, None, None, None, 1.0, causal=True)
