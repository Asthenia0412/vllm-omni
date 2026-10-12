# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm_omni.diffusion.layers.lingbot_pixels import PinnedPlanarRGBQueue

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _Event:
    def __init__(self):
        self.recorded = False
        self.waited = False
        self.finish = lambda: None

    def synchronize(self):
        assert self.recorded
        self.finish()
        self.waited = True


def make_queue(**kwargs):
    events, allocations = [], []

    def allocate(source):
        value = torch.empty_like(source)
        allocations.append(value)
        return value

    def event_factory():
        event = _Event()
        events.append(event)
        return event

    def enqueue_copy(host, source, event):
        # Model an asynchronous copy whose destination is unreadable until
        # event synchronization. The real implementation enqueues NPU D2H.
        event.finish = lambda: host.copy_(source)
        event.recorded = True

    queue = PinnedPlanarRGBQueue(allocate=allocate, event_factory=event_factory, copy=enqueue_copy, **kwargs)
    return queue, events, allocations


def source(value=1, frames=2):
    return torch.full((3, frames, 2, 4), value, dtype=torch.uint8)


def test_fifo_waits_for_own_event_and_returned_bytes_survive_reuse():
    queue, events, allocations = make_queue(capacity=2)
    queue.enqueue(source(1), session_id="one", chunk_index=0)
    queue.enqueue(source(2), session_id="one", chunk_index=1)
    session, index, first = queue.pop()
    assert (session, index) == ("one", 0)
    assert events[0].waited and not events[1].waited
    assert (first == 1).all()
    queue.enqueue(source(3), session_id="one", chunk_index=2)
    assert len(allocations) == 2
    assert queue.pop()[1] == 1
    assert (queue.pop()[2] == 3).all()
    assert (first == 1).all()


def test_full_queue_applies_backpressure_without_advancing_chunk():
    queue, _, _ = make_queue(capacity=1)
    queue.enqueue(source(), session_id="one", chunk_index=0)
    with pytest.raises(BufferError, match="full"):
        queue.enqueue(source(2), session_id="one", chunk_index=1)
    queue.pop()
    queue.enqueue(source(2), session_id="one", chunk_index=1)
    assert (queue.pop()[2] == 2).all()


def test_reset_drains_copies_and_prevents_cross_session_reuse():
    queue, events, _ = make_queue()
    queue.enqueue(source(), session_id="old", chunk_index=0)
    with pytest.raises(ValueError, match="session changed"):
        queue.enqueue(source(), session_id="new", chunk_index=0)
    queue.reset()
    assert events[0].waited
    assert not queue._pending and not queue._allocated
    queue.enqueue(source(7), session_id="new", chunk_index=0)
    assert queue.pop()[:2] == ("new", 0)


def test_shape_cache_is_bounded_and_reset_releases_slots():
    queue, _, allocations = make_queue(max_shapes=2)
    for index, frames in enumerate((1, 2)):
        queue.enqueue(source(frames=frames), session_id="one", chunk_index=index)
        queue.pop()
    with pytest.raises(BufferError, match="shape limit"):
        queue.enqueue(source(frames=3), session_id="one", chunk_index=2)
    assert len(allocations) == 2
    queue.reset()
    queue.enqueue(source(frames=3), session_id="new", chunk_index=0)
    assert queue.pop()[2].shape == (3, 2, 4, 3)


def test_duplicate_and_skipped_chunks_are_rejected():
    queue, _, _ = make_queue()
    with pytest.raises(ValueError, match="session order"):
        queue.enqueue(source(), session_id="one", chunk_index=1)
    queue.enqueue(source(), session_id="one", chunk_index=0)
    with pytest.raises(ValueError, match="session order"):
        queue.enqueue(source(), session_id="one", chunk_index=0)


def test_failed_copy_retains_buffers_and_requires_process_restart():
    queue, _, _ = make_queue()
    queue._copy = lambda *args: (_ for _ in ()).throw(RuntimeError("copy failed"))
    with pytest.raises(RuntimeError, match="copy failed"):
        queue.enqueue(source(), session_id="one", chunk_index=0)
    assert len(queue._pending) == 1
    with pytest.raises(RuntimeError, match="restart"):
        queue.pop()
    with pytest.raises(RuntimeError, match="restart"):
        queue.reset()
