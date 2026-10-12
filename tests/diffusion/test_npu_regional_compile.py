# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion import compile as module

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
        self.self_attn = nn.Module()
        self.self_attn.qkv = nn.Linear(4, 12)
        self.cam_injector = nn.Linear(4, 4)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.head = nn.Linear(4, 4)


def test_pure_regions_have_parity_and_leave_stateful_parent_eager(monkeypatch):
    model = _Model()
    value = torch.randn(2, 4)
    expected = [(block.ffn(value), block.self_attn.qkv(value)) for block in model.blocks]
    parent_forward = model.blocks[0].forward
    camera_forward = model.blocks[0].cam_injector.forward
    calls = []

    def compile(forward, **kwargs):
        calls.append(kwargs)
        return forward

    monkeypatch.setattr(torch, "compile", compile)
    module.regionally_compile_npu_pure(model)
    for block, (ffn, qkv) in zip(model.blocks, expected, strict=True):
        torch.testing.assert_close(block.ffn(value), ffn)
        torch.testing.assert_close(block.self_attn.qkv(value), qkv)
        torch.testing.assert_close(block.ffn(value + 1), block.ffn._omni_npu_region.forward(value + 1))
        assert block.ffn._omni_npu_region.hits == 1
    assert model.blocks[0].forward == parent_forward
    assert model.blocks[0].cam_injector.forward == camera_forward
    assert calls == [dict(backend="npugraph_ex", fullgraph=True, dynamic=False,
                         options={"capture_limit": 6, "pattern_fusion_pass": False})] * 4


def test_signature_limit_does_not_evict_and_unknown_shape_is_eager(monkeypatch):
    compiled = []
    monkeypatch.setattr(torch, "compile", lambda forward, **kwargs: compiled.append(forward) or forward)
    runner = module.NPURegionalGraphCallable(lambda x: x + 1, max_graphs=2)
    for size in (2, 3, 4, 2):
        value = torch.zeros(size)
        torch.testing.assert_close(runner(value), value + 1)
    assert len(compiled) == len(runner.graphs) == 2
    assert runner.eager_misses == 1
    assert runner.hits == 1


def test_shared_graph_pool_is_forwarded_without_changing_math_fusion(monkeypatch):
    pool = (1, 2)
    calls = []
    monkeypatch.setattr(torch, "compile", lambda forward, **kwargs: calls.append(kwargs) or forward)
    runner = module.NPURegionalGraphCallable(lambda x: x + 1, graph_pool=pool)
    runner(torch.ones(2))
    assert calls[0]["options"] == {"capture_limit": 6, "pattern_fusion_pass": False, "use_graph_pool": pool}


def test_signature_distinguishes_stride_and_constants(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "compile", lambda forward, **kwargs: calls.append(kwargs) or forward)
    runner = module.NPURegionalGraphCallable(lambda x, gain=1: x * gain)
    x = torch.arange(4.0).reshape(2, 2)
    for value, gain in ((x, 1), (x.T, 1), (x, 2), (x, 1)):
        torch.testing.assert_close(runner(value, gain=gain), value * gain)
    assert len(calls) == 3
    assert runner.hits == 1


def test_capture_failure_cannot_fall_back_or_reset_live_process(monkeypatch):
    calls = []

    def compile(forward, **kwargs):
        calls.append(1)
        return lambda *args: (_ for _ in ()).throw(RuntimeError("capture failed"))

    monkeypatch.setattr(torch, "compile", compile)
    runner = module.NPURegionalGraphCallable(lambda x: x)
    with pytest.raises(RuntimeError, match="restart the stage process"):
        runner(torch.ones(2))
    with pytest.raises(RuntimeError, match="cannot continue"):
        runner(torch.ones(3))
    assert len(calls) == 1
    assert not runner.graphs


def test_replay_failure_is_fatal(monkeypatch):
    count = 0

    def compiled(x):
        nonlocal count
        count += 1
        if count > 1:
            raise RuntimeError("replay failed")
        return x + 1

    monkeypatch.setattr(torch, "compile", lambda forward, **kwargs: compiled)
    runner = module.NPURegionalGraphCallable(lambda x: x + 1)
    runner(torch.ones(2))
    with pytest.raises(RuntimeError, match="replay failed.*restart"):
        runner(torch.ones(2))
    assert runner.failed


@pytest.mark.parametrize("invalid", [0, 7, True, "6"])
def test_graph_limit_is_bounded(invalid):
    with pytest.raises(ValueError):
        module.NPURegionalGraphCallable(lambda x: x, max_graphs=invalid)


def test_install_validation_is_atomic(monkeypatch):
    model = _Model()
    original = model.blocks[0].ffn.forward
    model.blocks[1].self_attn.qkv.tp_size = 2
    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        module.regionally_compile_npu_pure(model)
    assert model.blocks[0].ffn.forward == original
    assert not hasattr(model.blocks[0].ffn, "_omni_npu_region")


def test_offload_hooks_are_rejected_before_activation():
    model = _Model()
    model.blocks[0].ffn._hook_registry = SimpleNamespace()
    with pytest.raises(ValueError, match="offload hooks"):
        module.regionally_compile_npu_pure(model)


def test_stateful_child_activation_capture_is_rejected_before_installation(monkeypatch):
    model = _Model()
    original = model.blocks[0].ffn.forward
    monkeypatch.setenv("LINGBOT_CAPTURE_ACTIVATIONS", "1")
    with pytest.raises(ValueError, match="diagnostic child hooks are stateful"):
        module.regionally_compile_npu_pure(model)
    assert model.blocks[0].ffn.forward == original
    module.regionally_compile_npu_pure(model, regions=("qkv",))
    assert model.blocks[0].ffn.forward == original
