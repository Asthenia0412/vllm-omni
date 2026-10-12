"""Opt-in bounded SP8 tail diagnostics; never valid performance evidence.

Host timestamps share CLOCK_MONOTONIC across local workers. Device events
measure within-rank stream intervals only; no per-layer synchronization or
tensor readback is introduced. JSON serialization and I/O run in a writer.
"""
from __future__ import annotations

import atexit
from collections import deque
from functools import wraps
import gc
import json
import os
from pathlib import Path
import resource
import threading
import time

ENABLED = bool(os.environ.get("LINGBOT_TAIL_DIAGNOSTIC_DIR"))
_observer = None


def geometry(value):
    if value is None or not hasattr(value, "shape"):
        return None
    description = dict(shape=list(value.shape), stride=list(value.stride()),
                offset=value.storage_offset(), dtype=str(value.dtype),
                data_ptr=value.data_ptr(), storage_ptr=value.untyped_storage().data_ptr(),
                storage_bytes=value.untyped_storage().nbytes())
    if value.device.type == "npu":
        import torch_npu
        description["npu_format"] = int(torch_npu.get_npu_format(value))
    return description


class Observer:
    def __init__(self, rank, directory, device_events=True, capacity=20000):
        self.rank, self.capacity = rank, capacity
        self.directory = Path(directory) / f"rank{rank}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.records, self.pending = deque(), deque()
        self.dropped = 0
        self.context = dict(session=None, chunk=-1, probe=-1)
        self.device_events = device_events
        if device_events:
            import torch
            self.device = torch.npu.current_device()
        self.stop = threading.Event()
        self.writer = threading.Thread(target=self._write, daemon=True,
                                       name="lingbot-tail-writer")
        self.writer.start()
        gc.callbacks.append(self.gc_callback)
        atexit.register(self.close)

    def add(self, kind, **fields):
        if len(self.records) >= self.capacity:
            self.dropped += 1
            return
        self.records.append(dict(kind=kind, rank=self.rank, pid=os.getpid(),
                                 host_ns=time.monotonic_ns(), **self.context, **fields))

    def gc_callback(self, phase, info):
        if threading.current_thread() is threading.main_thread():
            self.add("gc", phase=phase, generation=info["generation"],
                     collected=info.get("collected", 0))

    def bind(self, state, probe=None):
        if state is not None:
            self.context = dict(session=state.request_id, chunk=int(state.chunk_index),
                                probe=int(state.step_in_chunk) if probe is None else probe)

    def wrap(self, function, label, *, device=False, state_argument=False):
        @wraps(function)
        def measured(*args, **kwargs):
            state = args[0] if state_argument and args else None
            states = kwargs.get("states")
            if states:
                state = states[0]
            self.bind(state)
            context = dict(self.context)
            usage = resource.getrusage(resource.RUSAGE_THREAD)
            start = time.monotonic_ns()
            events = None
            if device and self.device_events:
                import torch
                if torch.npu.is_initialized():
                    events = (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
                    events[0].record()
            failed = True
            try:
                result = function(*args, **kwargs)
                failed = False
                return result
            finally:
                end = time.monotonic_ns()
                if events is not None:
                    events[1].record()
                    if len(self.pending) < self.capacity:
                        self.pending.append((events, context, label, start))
                    else:
                        self.dropped += 1
                after = resource.getrusage(resource.RUSAGE_THREAD)
                if len(self.records) < self.capacity:
                    self.records.append(dict(kind="phase", rank=self.rank, pid=os.getpid(),
                        host_ns=start, end_ns=end, phase=label, failed=failed, **context,
                        cpu_ms=1000*((after.ru_utime+after.ru_stime)-(usage.ru_utime+usage.ru_stime)),
                        voluntary_switches=after.ru_nvcsw-usage.ru_nvcsw,
                        involuntary_switches=after.ru_nivcsw-usage.ru_nivcsw))
                else:
                    self.dropped += 1
        return measured

    def _write(self):
        if self.device_events:
            import torch
            torch.npu.set_device(self.device)
        with (self.directory / "events.jsonl").open("x") as output:
            while not self.stop.wait(2):
                self._drain(output)
            self._drain(output)
        (self.directory / "receipt.json").write_text(json.dumps(dict(
            rank=self.rank, dropped=self.dropped, unresolved_device_events=len(self.pending),
            timing_eligible=False, device_synchronizations=0, tensor_readbacks=0)) + "\n")

    def _drain(self, output):
        for _ in range(min(len(self.records), self.capacity)):
            output.write(json.dumps(self.records.popleft(), separators=(",", ":")) + "\n")
        # No synchronization: unresolved event pairs remain owned until ready.
        for _ in range(min(len(self.pending), self.capacity)):
            events, context, label, start = self.pending.popleft()
            if events[1].query():
                output.write(json.dumps(dict(kind="device_phase", rank=self.rank,
                    phase=label, host_ns=start, elapsed_ms=events[0].elapsed_time(events[1]),
                    **context)) + "\n")
            else:
                self.pending.append((events, context, label, start))
        output.flush()

    def close(self):
        if self.stop.is_set():
            return
        if self.gc_callback in gc.callbacks:
            gc.callbacks.remove(self.gc_callback)
        self.stop.set()
        self.writer.join(timeout=10)


def install(pipeline):
    global _observer
    if not ENABLED:
        return
    import torch
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() != 8:
        raise ValueError("Tail diagnostics require eight initialized SP8 workers")
    if _observer is not None:
        raise RuntimeError("Tail observer already installed in this worker")
    _observer = Observer(torch.distributed.get_rank(), os.environ["LINGBOT_TAIL_DIAGNOSTIC_DIR"])
    for name, device, state in (("_prepare_next_chunk", True, True),
                                ("denoise_step", True, False),
                                ("_decode_chunk_to_pixels", True, False),
                                ("post_decode", True, True)):
        setattr(pipeline, name, _observer.wrap(getattr(pipeline, name), name,
                                              device=device, state_argument=state))
    # Retain only slow CPU dispatch regions, plus block0 Q/K/V metadata.
    for index, block in enumerate(pipeline.transformer.blocks):
        for role in ("self_attn", "cross_attn"):
            module = getattr(block, role)
            observe_module(module, index, role, _observer)
    targets = {"decoder.up_blocks.1.resnets.0.conv2",
               "decoder.up_blocks.1.resnets.1.conv1", "decoder.up_blocks.1.resnets.1.conv2",
               "decoder.up_blocks.1.resnets.2.conv1", "decoder.up_blocks.1.resnets.2.conv2"}
    for name, module in pipeline.vae.named_modules():
        if name in targets:
            observe_vae_buffer(module, name, _observer)
    _observer.add("installed", cpu_affinity=sorted(os.sched_getaffinity(0)),
                  device_events=True, timings_include_diagnostics=True)


def observe_vae_buffer(module, name, observer):
    original = module._assemble_input
    previous_owner = None
    reuse_number = 0

    @wraps(original)
    def assembled(value, cache_x=None):
        nonlocal previous_owner, reuse_number
        result = original(value, cache_x)
        destination = result[0]
        owner = destination.untyped_storage().data_ptr()
        if owner != previous_owner:
            reuse_number += 1
            previous_owner = owner
        observer.add("vae_geometry", module=name, pointer_reuse_number=reuse_number,
            input=geometry(value), cache=geometry(cache_x), assembled=geometry(destination),
            persistent=geometry(getattr(module, "_input_buf", None)),
            halo_top=geometry(getattr(module, "_halo_recv_top_buf", None)),
            halo_bottom=geometry(getattr(module, "_halo_recv_bottom_buf", None)))
        return result
    module._assemble_input = assembled


def observe_module(module, index, role, observer):
    # Keep forward.__func__ and its caller code untouched: recv ingress checks
    # that exact execution boundary before allowing writes to managed owners.
    stack = []
    def before(_module, args, kwargs):
        if index == 0 and role == "self_attn":
            cache = kwargs.get("cache")
            if cache is not None:
                observer.add("kv_geometry", host_kv_len=getattr(cache, "host_kv_len", None),
                    reuse_history=getattr(cache, "reuse_history", None),
                    owners={name:geometry(getattr(cache, name, None)) for name in
                            ("key_pool", "value_pool", "stage_key", "stage_value")})
        stack.append(time.monotonic_ns())
    def after(_module, args, kwargs, result):
        if stack:
            begin = stack.pop()
            elapsed = time.monotonic_ns()-begin
            if elapsed >= 5_000_000:
                observer.add("slow_module", layer=index, role=role,
                             start_ns=begin, elapsed_ns=elapsed, failed=result is None)
    module.register_forward_pre_hook(before, with_kwargs=True)
    module.register_forward_hook(after, with_kwargs=True, always_call=True)
