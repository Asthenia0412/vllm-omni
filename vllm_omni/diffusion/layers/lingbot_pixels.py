# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact planar conversion of one LingBot decoded chunk to host RGB bytes."""

import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch


def _interleave_planar(planar: np.ndarray) -> np.ndarray:
    _, frame_count, height, width = planar.shape
    output = np.empty((frame_count, height, width, 3), dtype=np.uint8)
    for channel in range(3):
        output[..., channel] = planar[channel]
    return output


@dataclass
class _PendingRGB:
    session_id: str
    chunk_index: int
    host: torch.Tensor
    source: torch.Tensor  # keep device storage alive until its copy event completes
    event: Any


class PinnedPlanarRGBQueue:
    """Bounded, lossless D2H FIFO with explicit readiness and buffer ownership.

    The copy is enqueued on the decoder's current stream, retaining causal
    operation order. Consumers wait only for this chunk's event. Returned RGB
    arrays own their bytes; pinned slots can then be reused safely. No decode
    or HCCL overlap is introduced by this helper.
    """

    def __init__(
        self, *, capacity: int = 2, max_shapes: int = 2,
        allocate: Callable[[torch.Tensor], torch.Tensor] | None = None,
        event_factory: Callable[[], Any] | None = None,
        copy: Callable[[torch.Tensor, torch.Tensor, Any], None] | None = None,
    ) -> None:
        if capacity < 1 or max_shapes < 1:
            raise ValueError("RGB FIFO capacity and max_shapes must be positive")
        self.capacity, self.max_shapes = capacity, max_shapes
        self._allocate = allocate or (
            lambda source: torch.empty(source.shape, dtype=torch.uint8, device="cpu", pin_memory=True)
        )
        self._event_factory = event_factory or (lambda: torch.npu.Event())
        self._copy = copy or self._enqueue_copy
        self._pending: deque[_PendingRGB] = deque()
        self._free: dict[tuple[int, ...], list[torch.Tensor]] = {}
        self._allocated: dict[tuple[int, ...], int] = {}
        self._session_id: str | None = None
        self._last_submitted_index = -1
        self._failed = False

    @staticmethod
    def _enqueue_copy(host: torch.Tensor, source: torch.Tensor, event: Any) -> None:
        host.copy_(source, non_blocking=True)
        event.record()

    def enqueue(self, planar: torch.Tensor, *, session_id: str, chunk_index: int) -> None:
        if self._failed:
            raise RuntimeError("RGB transfer FIFO is failed; restart the stage process")
        if planar.dtype != torch.uint8 or planar.ndim != 4 or planar.shape[0] != 3:
            raise ValueError("RGB FIFO expects uint8 [3,F,H,W]")
        if self._session_id is not None and session_id != self._session_id:
            raise ValueError("RGB FIFO session changed; reset the queue before the new session")
        if chunk_index != self._last_submitted_index + 1:
            raise ValueError("RGB FIFO chunks must be submitted once in session order, starting at zero")
        if len(self._pending) >= self.capacity:
            raise BufferError("RGB transfer FIFO is full; consume the oldest chunk before retrying")
        shape = tuple(planar.shape)
        if shape not in self._allocated and len(self._allocated) >= self.max_shapes:
            raise BufferError("RGB transfer FIFO reached its shape limit; reset after consuming pending chunks")
        free = self._free.setdefault(shape, [])
        if free:
            host = free.pop()
        else:
            host = self._allocate(planar)
            self._allocated[shape] = self._allocated.get(shape, 0) + 1
        event = self._event_factory()
        self._pending.append(_PendingRGB(session_id, chunk_index, host, planar, event))
        try:
            self._copy(host, planar, event)
        except Exception:
            self._failed = True
            raise
        self._session_id = session_id
        self._last_submitted_index = chunk_index

    def pop(self) -> tuple[str, int, np.ndarray]:
        if self._failed:
            raise RuntimeError("RGB transfer FIFO is failed; restart the stage process")
        if not self._pending:
            raise IndexError("RGB transfer FIFO is empty")
        item = self._pending[0]
        try:
            item.event.synchronize()
        except Exception:
            self._failed = True
            raise
        return self._consume_ready()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def try_pop(self) -> tuple[str, int, np.ndarray] | None:
        """Poll the oldest copy without a device wait; never reorder chunks.

        Only a completed event permits host access or buffer reuse. This is
        for a future producer/consumer stage handoff; the synchronous pixel
        API deliberately retains its existing blocking contract.
        """
        if self._failed:
            raise RuntimeError("RGB transfer FIFO is failed; restart the stage process")
        if not self._pending:
            return None
        try:
            ready = self._pending[0].event.query()
        except Exception:
            self._failed = True
            raise
        return self._consume_ready() if ready else None

    def _consume_ready(self) -> tuple[str, int, np.ndarray]:
        item = self._pending[0]
        try:
            output = _interleave_planar(item.host.numpy())
        except Exception:
            self._failed = True
            raise
        self._pending.popleft()
        self._free[tuple(item.host.shape)].append(item.host)
        return item.session_id, item.chunk_index, output

    def reset(self) -> None:
        # Drain before releasing/reusing source or pinned host storage. Failed
        # device events propagate; reset must not conceal a broken device.
        if self._failed:
            raise RuntimeError("RGB transfer FIFO cannot reset failed device copies; restart the stage process")
        for item in self._pending:
            item.event.synchronize()
        self._pending.clear()
        self._free.clear()
        self._allocated.clear()
        self._session_id = None
        self._last_submitted_index = -1
        self._failed = False


_pinned_queues: dict[str, tuple[PinnedPlanarRGBQueue, int]] = {}


def planar_uint8_tensor(video: torch.Tensor) -> torch.Tensor:
    """Prepare exact [3,F,H,W] uint8 pixels without a host transfer."""
    if video.ndim != 5 or video.shape[:2] != (1, 3) or not video.is_floating_point():
        raise ValueError("LingBot pixel conversion expects floating-point [1,3,F,H,W] video")
    frames = (video[0] / 2 + 0.5).clamp(0, 1)
    return frames.float().mul_(255).round_().to(torch.uint8)


def planar_uint8_frames(video: torch.Tensor) -> np.ndarray:
    """[1,3,F,H,W] decoder output -> contiguous host [F,H,W,3] uint8.

    Preserve the VideoProcessor arithmetic boundaries: divide/add/clamp in
    decoder precision, then FP32 multiply and round. Keeping device arithmetic
    planar avoids CANN's repeated transposes around noncontiguous FHWC ops.
    """
    frames = planar_uint8_tensor(video)
    if video.device.type == "npu" and os.environ.get("VLLM_LINGBOT_NPU_PINNED_RGB", "0") == "1":
        key = str(video.device)
        queue, index = _pinned_queues.get(key, (PinnedPlanarRGBQueue(), 0))
        if tuple(frames.shape) not in queue._allocated and len(queue._allocated) >= queue.max_shapes:
            queue.reset()
            index = 0
        queue.enqueue(frames, session_id="synchronous", chunk_index=index)
        _, _, output = queue.pop()
        _pinned_queues[key] = (queue, index + 1)
        return output
    return _interleave_planar(frames.cpu().numpy())
