# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in dense attention adapters; Q/K/V and cache arithmetic stay BF16.

Imports of NPU/community libraries are lazy. The default preserves the old
native gate. This module never appends KV, applies RoPE, pads a sequence, or
changes which history tokens are visible.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, NamedTuple

import torch

logger = logging.getLogger(__name__)
BACKENDS = (
    "native", "fusion", "fia_bsnd", "fia_tnd", "fia_bnsd", "fia_v2_bsnd", "fia_v2_tnd", "fia_v2_bnsd",
    "flash_v2", "flash_v3", "flash_v4",
)
FLASH_PIN = "ef0579112055c771d2ddafdc53e94986d3befec5"
_BACKEND_FAILURES: set[tuple] = set()
_METADATA: OrderedDict[tuple, tuple[torch.Tensor, Any]] = OrderedDict()
_METADATA_LIMIT = 48
_STATS = {"metadata_hits": 0, "metadata_misses": 0, "fallbacks": 0}
_COMMUNITY_MODULES: dict[tuple, Any] = {}


class AttentionPolicy(NamedTuple):
    backend: str | None = None
    num_splits: int | None = None
    strict: bool | None = None


def parse_attention_policy(model_config: dict | None) -> AttentionPolicy:
    """Validate per-model overrides without changing process environment."""
    config = model_config or {}
    backend = config.get("lingbot_npu_attention_backend")
    splits = config.get("lingbot_npu_attention_splits")
    strict = config.get("lingbot_npu_attention_strict")
    if backend is not None and backend not in BACKENDS:
        raise ValueError(f"unknown lingbot_npu_attention_backend {backend!r}; choose {BACKENDS}")
    if splits is not None:
        if isinstance(splits, bool) or not isinstance(splits, int) or splits not in (0, 1):
            raise ValueError("lingbot_npu_attention_splits must be integer 0 or 1 for dense full-history attention")
    if strict is not None and not isinstance(strict, bool):
        raise ValueError("lingbot_npu_attention_strict must be a boolean")
    return AttentionPolicy(backend, splits, strict)


def configure_lingbot_attention(transformer, model_config: dict | None) -> dict:
    """Install scalar dispatch choices on each self-attention instance.

    None leaves the corresponding environment choice active at call time.
    Explicit values travel with the owning module through paged attention.
    Version is encoded by the backend enum, e.g. flash_v2 / flash_v3 / flash_v4.
    """
    policy = parse_attention_policy(model_config)
    count = 0
    for block in transformer.blocks:
        attention = block.self_attn
        attention._lingbot_npu_attention_backend = policy.backend
        attention._lingbot_npu_attention_splits = policy.num_splits
        attention._lingbot_npu_attention_strict = policy.strict
        count += 1
    return dict(configured_layers=count, backend=policy.backend, num_splits=policy.num_splits, strict=policy.strict)


def reset_attention_state() -> None:
    """Clear only dispatch state, useful for independent screening processes."""
    _BACKEND_FAILURES.clear()
    _METADATA.clear()
    _STATS.update(metadata_hits=0, metadata_misses=0, fallbacks=0)


def attention_state() -> dict:
    return dict(
        _STATS,
        metadata_entries=len(_METADATA),
        failed_signatures=len(_BACKEND_FAILURES),
        community_modules={
            version: getattr(module, "__file__", None) for (version, _), module in _COMMUNITY_MODULES.items()
        },
    )


def _validate(query, key, value, head_num, scale) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("LingBot NPU attention requires four-dimensional BSND tensors")
    if key.shape != value.shape or query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("LingBot NPU attention Q/K/V batch, heads and head dimension must agree")
    if head_num != query.shape[2] or head_num <= 0 or query.shape[1] <= 0 or key.shape[1] <= 0:
        raise ValueError("LingBot NPU attention requires matching heads and nonempty visible sequences")
    if not (query.device == key.device == value.device and query.dtype == key.dtype == value.dtype):
        raise ValueError("LingBot NPU attention Q/K/V must share device and dtype")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("LingBot NPU attention scale must be finite and positive")


def _native(torch_npu, query, key, value, head_num, scale, backend):
    if backend == "fusion":
        return torch_npu.npu_fusion_attention(
            query,
            key,
            value,
            head_num=head_num,
            input_layout="BSND",
            scale=scale,
            keep_prob=1.0,
            pre_tockens=2147483647,
            next_tockens=2147483647,
            sparse_mode=0,
        )[0]
    tnd = backend.endswith("tnd")
    bnsd = backend.endswith("bnsd")
    if bnsd:
        # Q/K/V change on every probe/layer. Never cache converted activations:
        # full-call measurement includes these three copies and the output copy.
        q, k, v = (tensor.transpose(1, 2).contiguous() for tensor in (query, key, value))
    else:
        q, k, v = (tensor.flatten(0, 1) for tensor in (query, key, value)) if tnd else (query, key, value)
    batch, sq, _, _ = query.shape
    sk = key.shape[1]
    lengths_q = [sq * (i + 1) for i in range(batch)]
    lengths_k = [sk * (i + 1) for i in range(batch)]
    layout = "TND" if tnd else "BNSD" if bnsd else "BSND"
    if backend.startswith("fia_v2"):
        kwargs = dict(
            num_query_heads=head_num,
            num_key_value_heads=head_num,
            input_layout=layout,
            softmax_scale=scale,
            sparse_mode=0,
            inner_precise=0,
            return_softmax_lse=False,
        )
        if tnd:
            kwargs.update(actual_seq_qlen=lengths_q, actual_seq_kvlen=lengths_k)
        result = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, **kwargs)[0]
    else:
        kwargs = dict(
            num_heads=head_num,
            num_key_value_heads=head_num,
            input_layout=layout,
            scale=scale,
            sparse_mode=0,
            inner_precise=0,
            softmax_lse_flag=False,
        )
        if tnd:
            kwargs.update(actual_seq_lengths=lengths_q, actual_seq_lengths_kv=lengths_k)
        result = torch_npu.npu_fused_infer_attention_score(q, k, v, **kwargs)[0]
    if bnsd:
        return result.transpose(1, 2).contiguous()
    return result.reshape(query.shape) if tnd else result


def _community_module(version: str):
    # No implicit pip installation. The root-owned launcher chooses this
    # persistent source directory, whose extensions were built for torch2.10.
    source = os.environ.get("VLLM_LINGBOT_NPU_FLASH_SOURCE")
    key = (version, str(Path(source).resolve()) if source else None)
    if key in _COMMUNITY_MODULES:
        return _COMMUNITY_MODULES[key]
    if source:
        path = key[1]
        head = Path(path) / ".git/HEAD"
        if not head.exists() or head.read_text().strip() != FLASH_PIN:
            raise ImportError(f"community attention source must be detached at {FLASH_PIN}")
        manifest_path = Path(path) / "lingbot_variant.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("launch_forbidden") or manifest.get("status", "").startswith("rejected"):
                raise ImportError(f"community attention variant rejected by static audit: {manifest_path}")
        if path not in sys.path:
            sys.path.insert(0, path)
    package = "flash_attn_npu" if version == "v2" else f"flash_attn_npu_{version[1:]}"
    module = importlib.import_module(f"{package}.flash_attn_npu_interface")
    if source and not Path(module.__file__).resolve().is_relative_to(path):
        raise ImportError(f"community attention was already imported outside the selected source: {module.__file__}")
    _COMMUNITY_MODULES[key] = module
    return module


def _metadata(module, signature, query, key, head_num, scale, num_splits):
    entry = _METADATA.get(signature)
    if entry is not None:
        _METADATA.move_to_end(signature)
        _STATS["metadata_hits"] += 1
        return entry
    # Lengths are derived from host-known tensor shapes, never from .item(),
    # .tolist() or a device tensor copied back to the host. Both objects stay
    # alive and unchanged for every launch using this signature.
    lengths = torch.full((query.shape[0],), key.shape[1], dtype=torch.int32, device=key.device)
    metadata = module.get_scheduler_metadata(
        batch_size=query.shape[0],
        max_seqlen_q=query.shape[1],
        max_seqlen_k=key.shape[1],
        num_heads_q=head_num,
        num_heads_kv=head_num,
        headdim=query.shape[-1],
        cache_seqlens=lengths,
        qkv_dtype=query.dtype,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=scale,
        num_splits=num_splits,
    )
    _METADATA[signature] = (lengths, metadata)
    if len(_METADATA) > _METADATA_LIMIT:
        _METADATA.popitem(last=False)
    _STATS["metadata_misses"] += 1
    return lengths, metadata


def _community(query, key, value, head_num, scale, backend, num_splits, signature):
    module = _community_module(backend.removeprefix("flash_"))
    lengths, metadata = _metadata(module, signature, query, key, head_num, scale, num_splits)
    if backend in ("flash_v2", "flash_v3"):
        # Public inference API validates its metadata fingerprint. No append
        # tensors, rotary tables, or page table are supplied: dense read only.
        return module.flash_attn_with_kvcache(
            query,
            key,
            value,
            cache_seqlens=lengths,
            softmax_scale=scale,
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            num_splits=num_splits,
            scheduler_metadata=metadata,
        )
    # v4's low-level registered forward avoids Autograd/training wrappers;
    # cached immutable metadata is keyed by every argument that affects tiling.
    return module._flash_attn_forward(
        query,
        key,
        value,
        seqused_k=lengths,
        softmax_scale=scale,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        softcap=0.0,
        num_splits=num_splits,
        scheduler_metadata=metadata,
    )[0]


def _check_result(result, query):
    if not isinstance(result, torch.Tensor) or result.shape != query.shape:
        raise RuntimeError("attention backend returned an invalid output shape")
    if result.dtype != query.dtype or result.device != query.device:
        raise RuntimeError("attention backend changed output dtype or device")
    return result


def run_attention(
    query,
    key,
    value,
    *,
    head_num: int,
    scale: float,
    backend: str | None = None,
    num_splits: int | None = None,
    strict: bool | None = None,
) -> torch.Tensor:
    """Run full-visible-history attention; explicit candidates are opt-in.

    Strict mode is required for benchmarks: an unavailable candidate must be
    recorded as unavailable rather than silently timing another backend.
    """
    _validate(query, key, value, head_num, scale)
    backend = backend or os.environ.get("VLLM_LINGBOT_NPU_ATTENTION_BACKEND", "native")
    if backend not in BACKENDS:
        raise ValueError(f"unknown LingBot NPU attention backend {backend!r}; choose {BACKENDS}")
    if backend == "native":
        backend = (
            "fia_bsnd"
            if (
                os.environ.get("VLLM_LINGBOT_NPU_INFER_ATTENTION", "0") == "1"
                and query.dtype in (torch.bfloat16, torch.float16)
            )
            else "fusion"
        )
    if strict is None:
        strict = os.environ.get("VLLM_LINGBOT_NPU_ATTENTION_STRICT", "0") == "1"
    if num_splits is None:
        num_splits = int(os.environ.get("VLLM_LINGBOT_NPU_ATTENTION_SPLITS", "1"))
    if backend.startswith("flash_"):
        if query.dtype != torch.bfloat16:
            raise ValueError("LingBot community attention screening requires BF16 Q/K/V")
        if num_splits not in (0, 1):
            raise ValueError("dense community attention permits num_splits=0 or1 only")
    query, key, value = (tensor.contiguous() for tensor in (query, key, value))
    scale = float(scale)
    signature = (backend, str(query.device), query.dtype, tuple(query.shape), tuple(key.shape), scale, num_splits)
    import torch_npu

    if strict or signature not in _BACKEND_FAILURES:
        try:
            result = (
                _community(query, key, value, head_num, scale, backend, num_splits, signature)
                if backend.startswith("flash_")
                else _native(torch_npu, query, key, value, head_num, scale, backend)
            )
            return _check_result(result, query)
        except (RuntimeError, NotImplementedError, AttributeError, ImportError, TypeError) as exc:
            # An exhausted allocator cannot safely support another attention
            # launch. Keep the original failure visible to the model runner.
            if strict or backend == "fusion" or "out of memory" in str(exc).lower():
                raise
            _BACKEND_FAILURES.add(signature)
            logger.warning(
                "LingBot attention backend %s unavailable for %s; native fallback: %s", backend, signature, exc
            )
    _STATS["fallbacks"] += 1
    # Existing inference-first behavior remains available for optional kernels.
    # A failed native FIA goes straight to the historical fusion fallback.
    fallback = "fusion" if backend == "fia_bsnd" else "native"
    return run_attention(
        query, key, value, head_num=head_num, scale=scale, backend=fallback, strict=False, num_splits=1
    )
