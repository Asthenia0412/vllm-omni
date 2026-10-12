# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded receive storage for the final spatial Wan decoder all-gather.

Only collective receive tensors are borrowed. The caller must assemble an
owned output before ending the lease; causal feature caches never live here.
Reuse is restricted to one submitting thread and one device stream. Shape,
byte-budget or ownership misses use the existing transient allocation path.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import Lock, get_ident
from typing import Any, Iterator

import torch


def _stream_owner(x: torch.Tensor) -> tuple[int, Any]:
    if x.device.type == "cpu":
        stream = None
    elif x.device.type == "cuda":
        stream = torch.cuda.current_stream(x.device).cuda_stream
    elif x.device.type == "npu":
        # No torch_npu import or device initialization at module import time.
        stream = torch.npu.current_stream(x.device).npu_stream
    else:
        raise ValueError(f"Unsupported Wan gather workspace device: {x.device.type}")
    return get_ident(), stream


class FinalGatherWorkspace:
    """Single-stream receive pool; all returned decoder outputs stay owned.

    The normal one-frame/four-frame output shapes fit in two entries. The
    default 32 MiB cap also fits both shapes in FP32 at 832x480, four ranks.
    A collective/cat failure poisons the pool: restart the stage rather than
    reusing potentially incomplete device writes. No synchronization or RNG,
    model arithmetic, state mutation or process-group creation is added.
    """

    def __init__(self, group: Any, world_size: int, *, max_shapes: int = 2,
                 max_bytes: int = 32 * 1024 * 1024) -> None:
        if world_size < 2 or max_shapes < 1 or max_bytes < 1:
            raise ValueError("Gather workspace requires at least two ranks and positive bounds")
        self.group, self.world_size = group, world_size
        self.max_shapes, self.max_bytes = max_shapes, max_bytes
        self._buffers: dict[tuple, list[torch.Tensor]] = {}
        self._lock = Lock()
        self._owner: tuple | None = None
        self._failed = False
        self.allocated_bytes = 0
        self.allocations = self.reuses = self.fallbacks = 0

    @contextmanager
    def lease(self, x: torch.Tensor, *, group: Any, world_size: int) -> Iterator[list[torch.Tensor] | None]:
        if self._failed:
            raise RuntimeError("Wan gather workspace failed; restart the stage process")
        if group is not self.group or world_size != self.world_size:
            raise ValueError("Wan gather workspace is bound to its original process group")
        owner = _stream_owner(x)
        if (self._owner is not None and owner != self._owner) or not self._lock.acquire(blocking=False):
            self.fallbacks += 1
            yield None
            return
        try:
            if self._failed:
                raise RuntimeError("Wan gather workspace failed; restart the stage process")
            key = (tuple(x.shape), x.dtype, x.device)
            buffers = self._buffers.get(key)
            if buffers is None:
                required = x.numel() * x.element_size() * world_size
                if len(self._buffers) >= self.max_shapes or self.allocated_bytes + required > self.max_bytes:
                    self.fallbacks += 1
                    buffers = None
                else:
                    buffers = [torch.empty_like(x, memory_format=torch.contiguous_format)
                               for _ in range(world_size)]
                    self._buffers[key] = buffers
                    self.allocated_bytes += required
                    self.allocations += world_size
                    self._owner = owner
            else:
                self.reuses += 1
            try:
                yield buffers
            except Exception:
                if buffers is not None:
                    self._failed = True
                raise
        finally:
            self._lock.release()
