# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU parity and invalidation checks for NPU metadata/history optimizations.

Only the hardware attention operator is replaced by CPU SDPA. These tests run
the real pool manager, forward contexts, writes and staging dispatch.
"""

import sys
import types

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig, paged_write_attn
from vllm_omni.experimental.ar_diffusion.kv_cache import paged_attention as module
from vllm_omni.experimental.ar_diffusion.kv_cache.config import (
    KV_GATHER_ENV,
    NPU_HISTORY_STAGING_ENV,
    NPU_HOST_METADATA_ENV,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
from vllm_omni.experimental.ar_diffusion.kv_cache.manager import select_npu_staging_layers

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
BLOCK, HEADS, DIM = 16, 2, 8


@pytest.fixture
def fake_npu(monkeypatch):
    calls = []

    def attention(q, k, v, **kwargs):
        calls.append((q.clone(), k.clone(), v.clone()))
        result = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=kwargs["scale"]
        ).transpose(1, 2)
        return (result,)

    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(npu_fusion_attention=attention))
    monkeypatch.delenv("VLLM_LINGBOT_NPU_INFER_ATTENTION", raising=False)
    return calls


def make_state(monkeypatch, *, sink=0, window=3, reset_at_boundary=False):
    # The CPU manager uses the existing gather gate to allocate the same
    # budgeted staging tensors; no NPU allocator/operator is invoked.
    monkeypatch.setenv(KV_GATHER_ENV, "1")
    monkeypatch.setenv(NPU_HISTORY_STAGING_ENV, "1")
    cfg = ARDiffusionKVConfig(
        enable=True, chunk_size=BLOCK, window_chunks=window, sink_chunks=sink,
        reset_at_boundary=reset_at_boundary, reuse_history_staging=True,
    )
    kv = ARDiffusionKVCache(
        cfg, num_layers=2, num_kv_heads=HEADS, head_size=DIM, dtype=torch.float32,
        block_size=BLOCK, max_model_len=4096, available_bytes=1 << 24,
        kv_branches=(ARDiffusionKVBranchSpec("positive", 0), ARDiffusionKVBranchSpec("negative", 1)),
        session_capacity=2, frames_per_block=1, max_scratch_tokens_per_branch=BLOCK,
        device=torch.device("cpu"),
    )
    return kv, ARDiffusionKVState(
        kv, "npu-test", {branch: kv.begin_request(branch) for branch in ("positive", "negative")}, num_layers=2,
    )


def npu_dispatch(query, key, value, **kwargs):
    return module._npu_paged_attention(
        query, key, value, kwargs["block_table"], kwargs["query_start_loc"], kwargs["seq_lens"],
        kwargs["softmax_scale"], causal=kwargs["causal"], host_kv_len=kwargs["host_kv_len"],
        stage_key=kwargs["stage_key"], stage_value=kwargs["stage_value"],
        reuse_history=kwargs["reuse_history"], current_tokens=kwargs["current_tokens"],
        stage_first_block=kwargs["stage_first_block"],
    )


def run_forward(state, *, branch="positive", commit=False, action_len=0):
    ctx = state.get_kv_caches(branch, seq_len=BLOCK, commit_current=commit)[0].forward_ctx
    ctx.prepare(torch.device("cpu"), action_len=action_len, query_len=BLOCK)
    for layer in range(2):
        q, k, v = (torch.randn(BLOCK, HEADS, DIM) for _ in range(3))
        k_act, v_act = (
            (None, None)
            if not action_len
            else (torch.randn(action_len, HEADS, DIM), torch.randn(action_len, HEADS, DIM))
        )
        inputs = ctx.layer_inputs(layer)
        actual = paged_write_attn(inputs, q, k, v, k_act, v_act, DIM**-0.5)
        expected = module._reference_paged_attention(
            q, ctx.kv_cache.key_cache(layer), ctx.kv_cache.value_cache(layer),
            inputs.block_table, inputs.query_start_loc, inputs.seq_lens, DIM**-0.5, causal=False,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    # Successful transient probes publish staging without committing pages.
    # #7816 advances the KV window only after the final probe.
    ctx.mark_history_staged()
    if commit:
        state.commit_paged_context(branch)
    return ctx


def test_four_probes_reuse_history_without_early_page_commits(monkeypatch, fake_npu):
    _, state = make_state(monkeypatch, sink=1)
    monkeypatch.setattr(module, "ar_diffusion_paged_attention", npu_dispatch)
    for _ in range(7):
        before = state.adapter("positive").absolute_num_computed_tokens
        contexts = []
        for _ in range(3):
            ctx = run_forward(state)
            contexts.append(ctx)
            assert not ctx._committed
            assert state.adapter("positive").absolute_num_computed_tokens == before
        final = run_forward(state, commit=True)
        assert [ctx.reuse_history for ctx in contexts] + [final.reuse_history] == [False, True, True, True]
        assert final._committed
        assert state.adapter("positive").absolute_num_computed_tokens == before + BLOCK


def test_host_lengths_do_not_read_device_metadata_or_padded_pages(monkeypatch, fake_npu):
    q = torch.randn(5, HEADS, DIM)
    k, v = torch.randn(4, BLOCK, HEADS, DIM), torch.randn(4, BLOCK, HEADS, DIM)
    table = torch.tensor([[2, 0, 1, 3]])
    offsets, lengths = torch.tensor([0, 5]), torch.tensor([19])
    expected = module._reference_paged_attention(q, k, v, table, offsets, lengths, DIM**-0.5, causal=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("host-known path must not read tensor values with .tolist()")

    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    actual = module._npu_paged_attention(
        q, k, v, table, offsets, lengths, DIM**-0.5, causal=False, host_kv_len=19,
    )
    torch.testing.assert_close(actual, expected)
    assert fake_npu[0][1].shape[1] == 19
    assert torch.equal(fake_npu[0][1][0], torch.cat((k[2], k[0, :3])))


@pytest.mark.parametrize("sink", [0, 1])
def test_staged_probe_commit_sliding_branch_and_reset_parity(monkeypatch, fake_npu, sink):
    _, state = make_state(monkeypatch, sink=sink)
    monkeypatch.setattr(module, "ar_diffusion_paged_attention", npu_dispatch)
    for _ in range(7):
        first = run_forward(state)
        assert not first.reuse_history
        second = run_forward(state)
        assert second.reuse_history
        committed = run_forward(state, commit=True)
        assert committed.reuse_history
        assert committed.adapter.absolute_num_computed_tokens > 0
    # Shared per-layer staging must be invalidated when the CFG branch changes.
    assert not run_forward(state, branch="negative").reuse_history
    assert not run_forward(state).reuse_history
    state.reset()
    assert not run_forward(state).reuse_history
    assert run_forward(state).reuse_history


def test_staged_gather_refreshes_only_current_blocks(monkeypatch, fake_npu):
    kv, state = make_state(monkeypatch)
    monkeypatch.setattr(module, "ar_diffusion_paged_attention", npu_dispatch)
    for _ in range(2):
        run_forward(state, commit=True)
    selected = []
    index_select = torch.index_select

    def count_gather(source, dim, indices, **kwargs):
        selected.append(indices.numel())
        return index_select(source, dim, indices, **kwargs)

    monkeypatch.setattr(torch, "index_select", count_gather)
    assert not run_forward(state).reuse_history
    assert selected == [3] * 4  # two layers, K and V, no action-capacity padding
    selected.clear()
    assert run_forward(state).reuse_history
    assert selected == [1] * 4
    assert kv.history_staging_reserved_bytes > 0


def test_abandoned_prepare_cannot_publish_unstaged_history(monkeypatch, fake_npu):
    _, state = make_state(monkeypatch)
    monkeypatch.setattr(module, "ar_diffusion_paged_attention", npu_dispatch)
    run_forward(state, commit=True)
    # Commit changes the history signature. Merely preparing its next forward
    # must not claim the new history is already present in every layer buffer.
    abandoned = state.get_kv_caches("positive", seq_len=BLOCK, commit_current=False)[0].forward_ctx
    abandoned.prepare(torch.device("cpu"), action_len=0, query_len=BLOCK)
    assert not abandoned.reuse_history
    assert not run_forward(state).reuse_history
    assert run_forward(state).reuse_history


def test_action_scratch_disables_staging_but_keeps_exact_host_lengths(monkeypatch, fake_npu):
    _, state = make_state(monkeypatch)
    monkeypatch.setattr(module, "ar_diffusion_paged_attention", npu_dispatch)
    ctx = run_forward(state, action_len=3)
    assert not ctx.staging_enabled
    assert ctx.host_kv_len == BLOCK + 3
    assert fake_npu[-1][1].shape[1] == BLOCK + 3


@pytest.mark.parametrize("host_kv_len,table_rows", [(-1, 1), (65, 1), (19, 2)])
def test_host_metadata_rejects_incompatible_capacity(fake_npu, host_kv_len, table_rows):
    with pytest.raises(ValueError):
        module._npu_paged_attention(
            torch.randn(5, HEADS, DIM), torch.randn(4, BLOCK, HEADS, DIM), torch.randn(4, BLOCK, HEADS, DIM),
            torch.zeros((table_rows, 4), dtype=torch.int64), torch.tensor([0, 5]), torch.tensor([19]),
            DIM**-0.5, causal=False, host_kv_len=host_kv_len,
        )


def test_metadata_gate_default_and_opt_in(monkeypatch):
    _, state = make_state(monkeypatch)
    monkeypatch.delenv(NPU_HISTORY_STAGING_ENV)
    monkeypatch.delenv(NPU_HOST_METADATA_ENV, raising=False)
    ctx = state.get_kv_caches("positive", seq_len=BLOCK, commit_current=False)[0].forward_ctx
    ctx.prepare(torch.device("cpu"), action_len=0, query_len=BLOCK)
    assert ctx.layer_inputs(0).host_kv_len == 0
    monkeypatch.setenv(NPU_HOST_METADATA_ENV, "1")
    ctx = state.get_kv_caches("positive", seq_len=BLOCK, commit_current=False)[0].forward_ctx
    ctx.prepare(torch.device("cpu"), action_len=0, query_len=BLOCK)
    assert ctx.layer_inputs(0).host_kv_len == BLOCK


def test_pinned_subset_retains_exact_full_history_on_unstaged_layers(monkeypatch, fake_npu):
    kv, state = make_state(monkeypatch, sink=1)
    monkeypatch.setattr(module, "ar_diffusion_paged_attention", npu_dispatch)
    # Pin layer1 only. Layer0 must use full gather, without indexing a missing
    # layer0 buffer to determine capacity or reusing another layer's data.
    kv.history_staging[0] = None
    for _ in range(5):
        assert not run_forward(state).reuse_history
        assert run_forward(state).reuse_history
        run_forward(state, commit=True)
    state.reset()
    assert not run_forward(state).reuse_history


def test_npu_staging_budget_pins_largest_subset_and_counts_reserved_workspace():
    gib = 1024**3
    layers = select_npu_staging_layers(
        usable_bytes=16*gib - 2*gib - 4*gib, base_session_bytes=9*gib,
        layer_bytes=gib//4, num_layers=40, requested=None,
    )
    assert layers == (0, 1, 2, 3)
    assert select_npu_staging_layers(
        usable_bytes=9*gib, base_session_bytes=9*gib, layer_bytes=gib//4,
        num_layers=40, requested=None,
    ) == ()


def test_explicit_staging_layers_are_pinned_and_never_silently_truncated():
    arguments = dict(usable_bytes=100, base_session_bytes=60, layer_bytes=10, num_layers=8)
    assert select_npu_staging_layers(**arguments, requested=(2, 6)) == (2, 6)
    with pytest.raises(ValueError, match="exceed"):
        select_npu_staging_layers(**arguments, requested=(0, 1, 2, 3, 4))
    for requested in ((1, 1), (-1,), (8,), (True,)):
        with pytest.raises(ValueError, match="unique valid"):
            select_npu_staging_layers(**arguments, requested=requested)
