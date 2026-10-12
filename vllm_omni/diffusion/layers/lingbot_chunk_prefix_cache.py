# SPDX-License-Identifier: Apache-2.0
"""Default-off reuse of immutable camera/RoPE inputs within one AR chunk.

The caller-owned #7816 CameraModulationCache already guarantees that camera
input and model weights stay fixed across a chunk's probes. This extension
retains the original pre-SP values; it never retains mutable SP staging views.
It is deliberately independent of timestep, latent, sampler, and KV state.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import os

import torch


def _flag(value):
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value in ("0", "1", "false", "true", "False", "True"):
        return value in ("1", "true", "True")
    raise ValueError("LingBot chunk prefix cache requires an explicit boolean")


def configure_chunk_prefix_cache(transformer, model_config=None, *, environ=None):
    values = os.environ if environ is None else environ
    name = "lingbot_npu_chunk_prefix_cache"
    enabled = _flag(values.get("LINGBOT_NPU_CHUNK_PREFIX_CACHE", "0"))
    if isinstance(model_config, Mapping) and name in model_config:
        enabled = _flag(model_config[name])
    elif model_config is not None and hasattr(model_config, name):
        enabled = _flag(getattr(model_config, name))
    transformer._lingbot_chunk_prefix_cache = enabled
    return dict(enabled=enabled, default_enabled=False, lifetime="caller-owned single AR chunk",
                cached="original pre-SP camera patch/C2WS projection and absolute RoPE",
                arithmetic_changed=False, original_sp_prepare_preserved=True)


def prefix_cache_enabled(transformer, cache, camera):
    return bool(getattr(transformer, "_lingbot_chunk_prefix_cache", False)
                and cache is not None and camera.device.type == "npu")


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors have no version counter. Their immutability is the
        # same caller contract required by #7816's existing modulation cache.
        return None


def _signature(camera, start_frame, geometry, sp_size):
    return (start_frame, tuple(geometry), sp_size, tuple(camera.shape), tuple(camera.stride()),
            camera.storage_offset(), camera.dtype, camera.device, _version(camera))


@dataclass(frozen=True)
class ChunkPrefix:
    owner: object
    source: torch.Tensor
    signature: tuple
    camera: torch.Tensor
    rotary: tuple[torch.Tensor, torch.Tensor]


def lookup_chunk_prefix(transformer, cache, camera, *, start_frame, geometry, sp_size):
    prefix = cache.prefix
    if prefix is None:
        return None
    if (prefix.owner is not transformer or prefix.source is not camera
            or prefix.signature != _signature(camera, start_frame, geometry, sp_size)):
        raise RuntimeError("LingBot chunk prefix cache was reused with changed camera/chunk/model inputs")
    return prefix


def save_chunk_prefix(transformer, cache, source, camera, rotary, *, start_frame, geometry, sp_size):
    if cache.prefix is not None:
        raise RuntimeError("LingBot chunk prefix cache must be populated only once per chunk")
    if source.requires_grad or camera.requires_grad or any(value.requires_grad for value in rotary):
        raise RuntimeError("LingBot chunk prefix cache is an inference-only path")
    if len(rotary) != 2 or any(value.device != camera.device or value.dtype != camera.dtype for value in rotary):
        raise RuntimeError("LingBot chunk prefix cache requires original matching camera/RoPE device and dtype")
    cache.prefix = ChunkPrefix(transformer, source,
                               _signature(source, start_frame, geometry, sp_size), camera, rotary)
