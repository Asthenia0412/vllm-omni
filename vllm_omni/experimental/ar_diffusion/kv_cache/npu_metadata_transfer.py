# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded pinned host ownership for eager NPU metadata H2D.

No device destination is pooled: its existing forward context owns it. A host
slot remains immutable until its post-copy event completes, including across
session changes. This module imports no torch/device backend until enabled.
"""
from collections import Counter
from dataclasses import dataclass
import json
import logging
import os
import threading
from typing import Any, Callable


logger = logging.getLogger(__name__)
MAX_BYTES = 2 * 1024**2
MAX_SLOTS = 24
MAX_SIGNATURES = 6


@dataclass
class _HostSlot:
    signature: tuple[Any, ...]
    host: Any
    nbytes: int
    event: Any = None
    pending: bool = False
    sequence: int = -1


def _allocate_host(source: Any) -> Any:
    import torch
    return torch.empty(source.shape, dtype=source.dtype, device="cpu", pin_memory=True)


def _copy_to_owned_device(host: Any, device: Any) -> Any:
    return host.to(device=device, non_blocking=True)


def _record_copy_event(device: Any) -> Any:
    import torch
    event = torch.npu.Event()
    event.record(torch.npu.current_stream(device))
    return event


class BoundedNPUMetadataTransfer:
    """Single-producer host pool; event completion permits host mutation only.

    Injectable callbacks make ownership/backpressure contracts testable without
    importing torch. Device buffers are never reused. Failure of a submitted
    copy/event poisons the pool because its source must not be overwritten.
    """
    def __init__(self, *, max_bytes: int = MAX_BYTES, max_slots: int = MAX_SLOTS,
                 max_signatures: int = MAX_SIGNATURES,
                 allocate: Callable[[Any], Any] = _allocate_host,
                 copy_to_device: Callable[[Any, Any], Any] = _copy_to_owned_device,
                 record_event: Callable[[Any], Any] = _record_copy_event) -> None:
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
               for v in (max_bytes, max_slots, max_signatures)):
            raise ValueError("Metadata host-pool limits must be positive")
        self.max_bytes, self.max_slots, self.max_signatures = max_bytes, max_slots, max_signatures
        self._allocate, self._copy_to_device, self._record_event = allocate, copy_to_device, record_event
        self._slots: list[_HostSlot] = []
        self._allocated_bytes = 0
        self._failed = False
        self._owner_thread = threading.get_ident()
        self._device: str | None = None
        self._sequence = 0
        self._stats: Counter[str] = Counter()
        self._warned: set[str] = set()

    def _check(self) -> None:
        if self._failed:
            raise RuntimeError("NPU metadata copy pool failed; restart the stage process")
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("NPU metadata pool requires one producer thread")

    def _note_fallback(self, reason: str) -> None:
        self._stats[reason] += 1
        if reason not in self._warned:
            logger.warning("LingBot NPU metadata transfer fallback: %s", reason)
            self._warned.add(reason)

    def _reap(self) -> None:
        try:
            for slot in self._slots:
                if slot.pending and slot.event.query():
                    slot.pending = False
                    slot.event = None
        except Exception:
            self._failed = True
            raise

    def _wait_oldest(self, *, draining: bool = False) -> None:
        oldest = min((s for s in self._slots if s.pending), key=lambda s: s.sequence)
        if draining:
            self._stats["drain_copy_event_waits"] += 1
        else:
            self._note_fallback("pool_full_wait_oldest_copy_event")
        try:
            oldest.event.synchronize()
        except Exception:
            self._failed = True
            raise
        oldest.pending = False
        oldest.event = None

    def transfer(self, source: Any, device: Any) -> Any:
        self._check()
        if source.device.type != "cpu":
            raise ValueError("NPU metadata H2D requires a CPU source")
        if self._device is None:
            self._device = str(device)
        elif str(device) != self._device:
            raise ValueError("One metadata pool cannot copy to different devices")
        signature = (tuple(source.shape), source.dtype)
        nbytes = source.numel() * source.element_size()
        self._stats["transfers"] += 1
        if nbytes == 0:
            self._stats["empty_transfers"] += 1
            return source.to(device=device)
        if nbytes > self.max_bytes:
            self._note_fallback("oversize_synchronous_copy")
            return source.to(device=device)
        self._reap()
        signatures = {s.signature for s in self._slots}
        if signature not in signatures and len(signatures) >= self.max_signatures:
            self._note_fallback("signature_limit_synchronous_copy")
            return source.to(device=device)
        while True:
            slot = next((s for s in self._slots if not s.pending and s.signature == signature), None)
            if slot is not None:
                break
            if len(self._slots) < self.max_slots and self._allocated_bytes + nbytes <= self.max_bytes:
                host = self._allocate(source)
                # Never accept an allocator that silently returns pageable memory.
                if not host.is_pinned():
                    raise RuntimeError("NPU metadata allocator returned unpinned host memory")
                slot = _HostSlot(signature, host, nbytes)
                self._slots.append(slot)
                self._allocated_bytes += nbytes
                self._stats["host_allocations"] += 1
                break
            # A completed slot of a different signature may be safely retired.
            retired = next((s for s in self._slots if not s.pending), None)
            if retired is not None:
                self._slots.remove(retired)
                self._allocated_bytes -= retired.nbytes
                continue
            self._wait_oldest()
            self._reap()
        try:
            slot.host.copy_(source)
            destination = self._copy_to_device(slot.host, device)
            # Pending is set before event creation: if recording fails, retain
            # the host source and poison the pool instead of reusing it.
            slot.pending = True
            slot.sequence = self._sequence
            self._sequence += 1
            slot.event = self._record_event(device)
        except Exception:
            self._failed = True
            raise
        self._stats["max_pending_slots"] = max(self._stats["max_pending_slots"], sum(s.pending for s in self._slots))
        if os.environ.get("LINGBOT_NPU_METADATA_DIAGNOSTIC", "0") == "1" and self._stats["transfers"] % 192 == 0:
            logger.warning("LingBot NPU metadata pool: %s", json.dumps(self.snapshot(), sort_keys=True))
        return destination

    def snapshot(self) -> dict[str, Any]:
        return {**dict(self._stats), "allocated_host_bytes": self._allocated_bytes,
                "allocated_host_slots": len(self._slots), "pending_host_slots": sum(s.pending for s in self._slots),
                "max_bytes": self.max_bytes, "max_slots": self.max_slots, "max_signatures": self.max_signatures,
                "device": self._device, "failed": self._failed, "device_destinations_pooled": False}

    def drain(self) -> None:
        """For shutdown/reset: finish copies before releasing any host sources."""
        self._check()
        while any(s.pending for s in self._slots):
            self._wait_oldest(draining=True)
        self._slots.clear()
        self._allocated_bytes = 0


_DEVICE_POOLS: dict[str, BoundedNPUMetadataTransfer] = {}


def transfer_npu_metadata(source: Any, device: Any) -> Any:
    key = str(device)
    pool = _DEVICE_POOLS.get(key)
    if pool is None:
        pool = BoundedNPUMetadataTransfer()
        _DEVICE_POOLS[key] = pool
        logger.warning("LingBot NPU async metadata enabled: host budget=%d bytes/%d slots; destination pooling disabled",
                       pool.max_bytes, pool.max_slots)
    # Pools carry no session/KV/action state. Immutable old-session bytes stay
    # pending until their copy completes; reuse always fills current values.
    return pool.transfer(source, device)
