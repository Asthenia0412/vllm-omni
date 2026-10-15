# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""UVA staging of the runner's per-step uploads matches the upstream uploads."""

import numpy as np
import pytest
import torch
from vllm.utils.platform_utils import is_uva_available
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

from tests.helpers.mark import hardware_marks
from vllm_omni.worker_v2.uva_staging import GrowOnlyUvaBufferPool, stage_write_contents_in_uva

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available() or not is_uva_available(), reason="needs CUDA UVA"),
]


def test_growth_keeps_replaced_buffers_alive():
    pool = GrowOnlyUvaBufferPool(4, torch.int32, max_concurrency=2)
    first = pool.copy_to_uva(np.arange(3))
    replaced = pool._uva_bufs[(pool._curr + 1) % 2]
    pool.copy_to_uva(np.arange(9))  # grows the other slot
    grown = pool.copy_to_uva(np.arange(20))  # grows the first slot
    assert any(buf is replaced for buf in pool._retired)
    assert len(pool._retired) == 2
    torch.accelerator.synchronize()
    assert first.tolist() == [0, 1, 2] and grown.tolist() == list(range(20))


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
def test_staged_writes_through_uva_match_the_upload_path(dtype):
    device = torch.device("cuda")
    reference = StagedWriteTensor((6, 8), dtype, device)
    candidate = StagedWriteTensor((6, 8), dtype, device)
    stage_write_contents_in_uva([candidate])
    assert reference.write_contents is None and candidate.write_contents is not None
    rng = np.random.default_rng(1)
    for _ in range(5):
        for tensor in (reference, candidate):
            tensor.stage_write_elem(1, 7)
        rows = rng.choice(6, 3, replace=False)
        for row in rows:
            start = int(rng.integers(0, 4))
            values = rng.integers(-50, 50, int(rng.integers(1, 9 - start))).tolist()
            for tensor in (reference, candidate):
                tensor.stage_write(int(row), start, values)
        for tensor in (reference, candidate):
            tensor.apply_write()
    torch.accelerator.synchronize()
    assert torch.equal(reference.gpu, candidate.gpu)


def test_uva_staging_leaves_non_moss_runner_unchanged(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as upstream_runner

    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

    runner = object.__new__(OmniGPUModelRunner)
    runner.model_state = object()
    runner.device = torch.device("cuda")
    original = upstream_runner.async_tensor_h2d
    monkeypatch.setattr("vllm_omni.worker_v2.omni_model_runner.is_uva_available", lambda: True)
    runner._configure_uva_staging()
    assert upstream_runner.async_tensor_h2d is original


def test_uva_staging_runs_after_cache_initialization(monkeypatch):
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

    runner = object.__new__(OmniGPUModelRunner)
    config = KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])
    order = []
    monkeypatch.setattr(GPUModelRunner, "initialize_kv_cache", lambda *args: order.append("cache"))
    runner._configure_uva_staging = lambda: order.append("uva")
    runner.initialize_kv_cache(config)
    assert order == ["cache", "uva"]
