"""Candidate template: post-load, encoder-only residency. Default disabled.

This file is copied into an isolated vLLM-Omni source by the preparation script.
It has no framework imports at module scope. It never changes transformer/VAE
placement, arithmetic, the public offload selector, or global NPU policies.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

# Worker shutdown logs backend errors and deletes/gc-collects its runner.
# A failed fence therefore needs an independent, process-lifetime owner.
# There is deliberately no inline recovery/reuse after a quarantined failure.
_QUARANTINE = []


def _quarantine(owner):
    owner.poisoned = True
    if not any(value is owner for value in _QUARANTINE):
        _QUARANTINE.append(owner)


def _enabled(value):
    if value in (True, 1, "1", "true"):
        return True
    if value in (None, False, 0, "0", "false", ""):
        return False
    raise ValueError("text block offload accepts only explicit true/false")


class TextBlockLifecycle:
    """Framework-independent owner, also exercised by CPU lifecycle tests."""

    def __init__(self, runtime, encoder, hooks):
        self.runtime, self.encoder, self.hooks = runtime, encoder, tuple(hooks)
        self.lock = threading.RLock()
        self.active = self.poisoned = self.closed = False
        self.invocation_running = False
        self.stream = None
        self.encoder_calls = self.encoder_drains = self.failed_calls = 0
        self.original_forward = encoder.forward

    def before_encode(self):
        with self.lock:
            if self.closed or self.poisoned or self.active:
                raise RuntimeError("text block residency is closed, poisoned, or concurrently active")
            stream = self.runtime.current_stream()
            # Same-stream allocation/copy/use/free avoids relying on an unproved
            # cross-stream caching-allocator lifetime. A new stream is selected
            # only after the previous encoder invocation has been fully drained.
            for hook in self.hooks:
                hook.copy_stream = stream
            self.stream = stream
            self.active = True
            self.invocation_running = True
            self.encoder_calls += 1

    def finish_encode(self, failed=False):
        with self.lock:
            if not self.active:
                raise RuntimeError("encoder finish without active invocation")
            self.poisoned |= failed
            self.failed_calls += int(failed)
            try:
                self.stream.synchronize()
                self.encoder_drains += 1
                # The ring normally prefetches block 0 again from block 23.
                # Release that residual block after the encoder-only fence.
                self.runtime.clear_blocks(self.hooks)
            except BaseException:
                _quarantine(self)
                # Keep active=true, the stream, hooks and pinned masters alive.
                # shutdown may retry the fence; no later encode is admitted.
                raise
            self.active = False
            self.runtime.publish(self)

    def forward(self, *args, **kwargs):
        self.before_encode()
        try:
            try:
                output = self.original_forward(*args, **kwargs)
            finally:
                with self.lock:
                    self.invocation_running = False
        except BaseException as original:
            try:
                self.finish_encode(failed=True)
            except BaseException as cleanup:
                if hasattr(original, "add_note"):
                    original.add_note("Encoder drain/cleanup also failed: " + repr(cleanup))
            raise
        self.finish_encode()
        return output

    def shutdown(self):
        with self.lock:
            if self.closed:
                return
            try:
                # Do not detach CPU masters while an encoder thread can enqueue
                # more operations. Worker shutdown requires requests stopped.
                if self.invocation_running:
                    raise RuntimeError("shutdown requires the encoder invocation to finish")
                if self.stream is not None:
                    self.stream.synchronize()
                self.runtime.clear_blocks(self.hooks)
                self.encoder.forward = self.original_forward
                self.runtime.disable(self.encoder, restore_weights=False)
                self.active = False
                self.closed = True
                self.runtime.publish(self)
                self.hooks = ()  # Release pinned masters only after the fence.
            except BaseException:
                _quarantine(self)
                raise


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _rank(torch):
    return str(torch.distributed.get_rank()) if torch.distributed.is_initialized() else "0"


def _resident_tensors(encoder, blocks):
    blocked_modules = {id(module) for block in blocks for module in block.modules()}
    values = {}
    for module in encoder.modules():
        if id(module) not in blocked_modules:
            for tensor in tuple(module.parameters(recurse=False)) + tuple(module.buffers(recurse=False)):
                values[id(tensor)] = tensor
    return tuple(values.values())


class _Runtime:
    def __init__(self, torch, platform, device, disable, clear, summary, metadata):
        self.torch, self.platform, self.device = torch, platform, device
        self.disable, self.clear = disable, clear
        self.summary, self.metadata = summary, metadata

    def current_stream(self):
        return self.platform.current_stream()

    def clear_blocks(self, hooks):
        for hook in hooks:
            self.clear(tuple(hook.block_parameters.values()) + tuple(hook.block_buffers.values()))
            hook._prefetch_done = None

    def publish(self, owner):
        if self.summary is None:
            return
        free, total = self.torch.npu.mem_get_info(self.device)
        value = dict(
            self.metadata,
            enabled=not owner.closed,
            poisoned=owner.poisoned,
            encoder_calls=owner.encoder_calls,
            encoder_drains=owner.encoder_drains,
            failed_calls=owner.failed_calls,
            active=owner.active,
            free_bytes=int(free),
            total_bytes=int(total),
            allocated_bytes=int(self.torch.npu.memory_allocated(self.device)),
            reserved_bytes=int(self.torch.npu.memory_reserved(self.device)),
        )
        value["global_peak_allocated_bytes_not_reset_by_candidate"] = int(
            self.torch.npu.max_memory_allocated(self.device)
        )
        _atomic_json(self.summary, value)


def _install_prompt_capture(pipeline, directory):
    """Capture up to two distinct final masked prompt tensors, both arms alike."""
    if not directory:
        return
    import torch

    root = Path(directory)
    if not root.is_absolute() or str(root).startswith(("/tmp/", "/root/")):
        raise ValueError("prompt capture requires a persistent absolute output directory")
    rank = _rank(torch)
    original = pipeline.encode_prompt
    seen = set()

    def capture(prompt, *, max_sequence_length, dtype):
        owner = getattr(pipeline, "_lingbot_text_block_residency", None)
        calls_before = owner.encoder_calls if owner else None
        output = original(prompt, max_sequence_length=max_sequence_length, dtype=dtype)
        text = " ".join(prompt.strip().split())
        key = hashlib.sha256((text + "\0" + str(max_sequence_length) + "\0" + str(dtype)).encode()).hexdigest()
        if key not in seen and len(seen) < 2:
            seen.add(key)
            calls_after = owner.encoder_calls if owner else None
            drains_after = owner.encoder_drains if owner else None
            cached = original(prompt, max_sequence_length=max_sequence_length, dtype=dtype)
            if cached is not output or (
                owner and (owner.encoder_calls != calls_after or owner.encoder_drains != drains_after)
            ):
                raise RuntimeError("diagnostic prompt cache hit changed tensor identity or invoked encoder residency")
            cpu = output.detach().to("cpu").contiguous().clone()
            payload = cpu.view(torch.uint8).numpy().tobytes()
            stem = root / ("rank" + rank + "_prompt" + str(len(seen)) + "_" + key[:16])
            torch.save(cpu, stem.with_suffix(".pt"))
            _atomic_json(
                stem.with_suffix(".json"),
                dict(
                    prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    key_sha256=key,
                    payload_sha256=hashlib.sha256(payload).hexdigest(),
                    shape=list(cpu.shape),
                    dtype=str(cpu.dtype),
                    max_sequence_length=max_sequence_length,
                    encoder_calls_before=calls_before,
                    encoder_calls_after=owner.encoder_calls if owner else None,
                    offload_enabled=owner is not None,
                    masked_prompt_embedding=True,
                    cache_hit_tensor_identity=True,
                    cache_hit_added_encoder_calls=0 if owner else None,
                    cache_hit_added_encoder_drains=0 if owner else None,
                ),
            )
        return output

    root.mkdir(parents=True, exist_ok=True)
    pipeline.encode_prompt = capture


def install_lingbot_text_block_residency(od_config, pipeline, device, public_backend):
    """Install only after ordinary NPU load/quant postprocessing has completed."""
    if _QUARANTINE:
        raise RuntimeError("quarantined encoder residency requires worker process restart")
    config = getattr(od_config, "model_config", None) or {}
    enabled = _enabled(config.get("lingbot_npu_text_block_offload", False))
    capture_dir = os.environ.get("LINGBOT_TEXT_BLOCK_CAPTURE_DIR", "")
    if not enabled and not capture_dir:
        return public_backend
    if device.type != "npu" or od_config.model_class_name != "LingBotWorldCausalDMDPipeline":
        raise ValueError("text block residency diagnostics support only NPU LingBot")
    if public_backend is not None:
        raise ValueError("text block candidate cannot coexist with public/global offload")
    from .config import offload_enabled

    if offload_enabled(od_config):
        raise ValueError("ordinary NPU loading is required; public offload must be disabled")
    if not enabled:
        _install_prompt_capture(pipeline, capture_dir)
        return None
    if getattr(pipeline, "_lingbot_text_block_residency", None) is not None:
        raise RuntimeError("text block residency already installed")

    import torch

    from vllm_omni.platforms import current_omni_platform

    from .layerwise_backend import disable_plan_encoder_layerwise_offload, enable_plan_encoder_layerwise_offload
    from .plan_resolver import BlockStack, ResolvedComponent
    from .tensor_utils import clear_tensor_storage

    encoder = pipeline.text_encoder
    blocks = tuple(encoder.encoder.block)
    if type(encoder).__name__ != "UMT5EncoderModel" or len(blocks) != 24 or encoder.training:
        raise ValueError("candidate requires the original eval UMT5EncoderModel with exactly 24 blocks")
    block_tensor_ids = {id(t) for block in blocks for t in tuple(block.parameters()) + tuple(block.buffers())}
    resident = _resident_tensors(encoder, blocks)
    if not resident or any(t.device != device for t in resident):
        raise ValueError("shared embedding/final norm must already be resident on the original NPU")
    # Reject cross-block/shared parameter ownership; the ring cannot clear a
    # Parameter needed by another concurrently resident block or embedding.
    ids = [id(t) for block in blocks for t in tuple(block.parameters()) + tuple(block.buffers())]
    if len(ids) != len(set(ids)) or block_tensor_ids.intersection(id(t) for t in resident):
        raise ValueError("cross-block or resident/block shared tensors are unsupported")
    tensor_bytes = sum(
        t.numel() * t.element_size() for block in blocks for t in tuple(block.parameters()) + tuple(block.buffers())
    )
    before_free, before_total = torch.npu.mem_get_info(device)
    stream = current_omni_platform.current_stream()
    stream.synchronize()  # Post-load fence, once, before taking CPU masters.
    component = ResolvedComponent("text_encoder", encoder, True, (BlockStack(blocks),), False)
    if not enable_plan_encoder_layerwise_offload(component, device=device, stream=stream, pin_memory=True):
        raise RuntimeError("explicit encoder block plan was not installed")
    hooks = tuple(encoder._omni_layerwise_hooks)
    if len(hooks) != 24:
        raise RuntimeError("encoder offloader did not install exactly 24 block hooks")
    torch.npu.empty_cache()  # Post-load only, before AR KV budgeting.
    rank = _rank(torch)
    summary_dir = os.environ.get("LINGBOT_TEXT_BLOCK_SUMMARY_DIR", "")
    summary = Path(summary_dir) / ("rank" + rank + ".json") if summary_dir else None
    metadata = dict(
        kind="text_encoder_blocks_postload",
        block_count=24,
        block_logical_bytes=int(tensor_bytes),
        resident_tensor_count=len(resident),
        resident_logical_bytes=sum(t.numel() * t.element_size() for t in resident),
        copy_policy="same_current_encoder_compute_stream",
        pin_cpu_memory=True,
        before_install_free_bytes=int(before_free),
        total_bytes=int(before_total),
        public_offload_disabled=True,
        transformer_hooks_added=0,
        full_weights_restore_on_shutdown=False,
    )
    runtime = _Runtime(
        torch,
        current_omni_platform,
        device,
        disable_plan_encoder_layerwise_offload,
        clear_tensor_storage,
        summary,
        metadata,
    )
    owner = TextBlockLifecycle(runtime, encoder, hooks)
    encoder.forward = owner.forward
    pipeline._lingbot_text_block_residency = owner
    runtime.publish(owner)
    _install_prompt_capture(pipeline, capture_dir)
    return owner
