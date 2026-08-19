# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase-I normalized FP8 cache support for DLO checkpoint mmap."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.model_loader.host_weight_plan import (
    _build_source_map,
    _collect_required_dit_tensors,
)
from vllm_omni.quantization.component_config import ComponentQuantizationConfig

logger = init_logger(__name__)

CACHE_SCHEMA_VERSION = 1
CACHE_KIND = "normalized_fp8_runtime"
CACHE_MANIFEST_NAME = "manifest.json"
CACHE_COMPLETE_NAME = "COMPLETE"
CACHE_COMPONENT_SUBFOLDER = "transformer"


def _transformer_quant_config() -> ComponentQuantizationConfig:
    """Build the serialized FP8 config used by a normalized cache."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    return ComponentQuantizationConfig(
        {
            "transformer": Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
            )
        }
    )


def is_online_fp8_config(quant_config: Any) -> bool:
    """Return whether the transformer component requests online FP8."""
    if quant_config is None:
        return False
    if hasattr(quant_config, "resolve"):
        quant_config = quant_config.resolve("transformer")
    get_name = getattr(quant_config, "get_name", None)
    if not callable(get_name) or str(get_name()).lower() != "fp8":
        return False
    return not bool(getattr(quant_config, "is_checkpoint_fp8_serialized", False))


def _source_fingerprint(files: set[str]) -> str:
    entries: list[dict[str, int | str]] = []
    for filename in sorted(files):
        stat = os.stat(filename)
        entries.append(
            {
                "path": os.path.abspath(filename),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_fp8_cache_manifest(cache_dir: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    """Return a validated Phase-I manifest, or ``None`` for a cache miss."""
    if cache_dir is None:
        return None
    root = Path(cache_dir)
    manifest_path = root / CACHE_MANIFEST_NAME
    if not (root / CACHE_COMPLETE_NAME).is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    if manifest.get("cache_kind") != CACHE_KIND:
        return None
    if manifest.get("component") != CACHE_COMPONENT_SUBFOLDER:
        return None
    quantization = manifest.get("quantization")
    if not isinstance(quantization, dict) or quantization != {
        "method": "fp8",
        "serialized": True,
        "activation_scheme": "dynamic",
    }:
        return None
    if manifest.get("component_subfolder") != CACHE_COMPONENT_SUBFOLDER:
        return None
    if manifest.get("source_prefix") != f"{CACHE_COMPONENT_SUBFOLDER}.":
        return None

    index_path = root / CACHE_COMPONENT_SUBFOLDER / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(index, dict):
        return None
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map or not all(isinstance(key, str) for key in weight_map):
        return None
    if manifest.get("tensor_count") != len(weight_map):
        return None
    for filename in weight_map.values():
        if not isinstance(filename, str):
            return None
        relative_path = Path(filename)
        if (
            relative_path.is_absolute()
            or relative_path.name != filename
            or relative_path.suffix != ".safetensors"
            or not (root / CACHE_COMPONENT_SUBFOLDER / relative_path).is_file()
        ):
            return None

    source_files = manifest.get("source_files")
    source_fingerprint = manifest.get("source_fingerprint")
    if not isinstance(source_files, list) or not source_files or not isinstance(source_fingerprint, str):
        return None
    try:
        if _source_fingerprint({str(filename) for filename in source_files}) != source_fingerprint:
            return None
    except (OSError, TypeError):
        return None
    return manifest


def _cache_key_for_runtime_name(runtime_name: str, prefix: str) -> str:
    if not runtime_name.startswith(prefix):
        raise ValueError(f"runtime tensor {runtime_name!r} is outside cache source prefix {prefix!r}")
    return runtime_name[len(prefix) :]


def build_normalized_fp8_cache(
    model: nn.Module,
    *,
    sources: tuple[object, ...],
    cache_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Write a normalized FP8 cache from a completed online-FP8 model.

    The builder writes tensors one at a time into individual safetensors shards,
    so it does not create a second full-model state dict. The first cache build
    still follows the existing online loader; warm launches consume the cache
    through the direct DLO mmap path.
    """
    cache_root = Path(cache_dir)
    existing_manifest = load_fp8_cache_manifest(cache_root)
    if existing_manifest is not None:
        return existing_manifest
    if cache_root.exists() and any(cache_root.iterdir()):
        raise FileExistsError(f"FP8 cache directory is not empty and is not valid: {cache_root}")
    cache_root.parent.mkdir(parents=True, exist_ok=True)

    dit_names = tuple(getattr(model, "_dit_modules", ()))
    dit_names = tuple(name for name in dit_names if hasattr(model, name))
    if len(dit_names) != 1:
        raise ValueError(f"Phase-I FP8 cache requires exactly one DiT module, got {dit_names}")
    dit_name = dit_names[0]
    if dit_name != CACHE_COMPONENT_SUBFOLDER:
        raise ValueError(
            f"Phase-I FP8 cache requires a top-level {CACHE_COMPONENT_SUBFOLDER!r} module, got {dit_name!r}"
        )
    source_prefix = f"{dit_name}."
    dit_module = model.get_submodule(dit_name)
    planned_sources = tuple(source for source in sources if getattr(source, "prefix", "") == source_prefix)
    if len(planned_sources) != 1:
        raise ValueError(f"Phase-I FP8 cache requires one source for {source_prefix!r}, got {len(planned_sources)}")

    source_map = _build_source_map(
        sources=planned_sources,
        model_path=None,
        remap_fn=getattr(type(model), "_remap_ckpt_key", None),
    )
    required = _collect_required_dit_tensors(((dit_name, dit_module),))
    quantized_modules: dict[str, nn.Module] = {}
    try:
        from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
    except ImportError as exc:  # pragma: no cover - vLLM 0.27 provides this class
        raise RuntimeError("Phase-I FP8 cache requires vLLM's online FP8 implementation") from exc

    for module_name, module in model.named_modules():
        if not module_name.startswith(source_prefix):
            continue
        if isinstance(getattr(module, "quant_method", None), Fp8PerTensorOnlineLinearMethod):
            if not hasattr(module, "weight_scale"):
                raise ValueError(f"online FP8 module {module_name!r} has no generated weight_scale")
            quantized_modules[module_name] = module

    tmp_root = cache_root.parent / f".{cache_root.name}.tmp-{uuid.uuid4().hex}"
    component_root = tmp_root / CACHE_COMPONENT_SUBFOLDER
    component_root.mkdir(parents=True, exist_ok=False)
    weight_map: dict[str, str] = {}
    source_files: set[str] = set()
    total_size = 0
    part_index = 0

    file_cache: dict[str, Any] = {}

    def write_tensor(key: str, tensor: torch.Tensor) -> None:
        nonlocal part_index, total_size
        if key in weight_map:
            raise ValueError(f"normalized FP8 cache has duplicate tensor key {key!r}")
        tensor = tensor.detach().to("cpu").contiguous()
        filename = f"part-{part_index:05d}.safetensors"
        part_index += 1
        save_file({key: tensor}, str(component_root / filename))
        weight_map[key] = filename
        total_size += tensor.numel() * tensor.element_size()

    try:
        for runtime_name, target in required.items():
            binding_info = source_map.get(runtime_name)
            module_name = runtime_name.removesuffix(".weight")
            quant_module = quantized_modules.get(module_name) if runtime_name.endswith(".weight") else None
            if runtime_name.endswith(".weight_scale"):
                owner_name = runtime_name.removesuffix(".weight_scale")
                if owner_name in quantized_modules:
                    # Generated online scales are published in the dedicated
                    # pass below; they have no raw checkpoint binding.
                    continue
            if quant_module is not None:
                if binding_info is None:
                    raise ValueError(f"no source binding for quantized weight {runtime_name!r}")
                qweight = quant_module.weight.detach()
                if qweight.ndim != 2:
                    raise ValueError(f"Phase-I FP8 cache only supports 2D weights, got {runtime_name!r}")
                # Online FP8 stores the final runtime weight as qweight.T. The
                # serialized FP8 method expects the pre-process qweight shape.
                qweight = qweight.t().contiguous()
                # Normalized caches contain the post-checkpoint-adapter QKV
                # order. The cache-hit adapter bypasses the raw-checkpoint
                # grouped-QKV transform, then the serialized FP8 finalizer
                # performs only the standard weight transpose.
                write_tensor(binding_info[0], qweight)
                source_files.add(binding_info[1])
                continue

            if binding_info is not None:
                checkpoint_key, file_path = binding_info
                source_files.add(file_path)
                handle = file_cache.get(file_path)
                if handle is None:
                    handle = safe_open(file_path, framework="pt", device="cpu")
                    file_cache[file_path] = handle
                write_tensor(checkpoint_key, handle.get_tensor(checkpoint_key))
            else:
                # Constructor-derived persistent buffers have no checkpoint
                # entry; preserve them as canonical cache tensors.
                write_tensor(_cache_key_for_runtime_name(runtime_name, source_prefix), target)

        for module_name, module in quantized_modules.items():
            scale = module.weight_scale.detach()
            logical_widths = getattr(module, "logical_widths", None)
            if scale.numel() == 1 and logical_widths is not None and len(logical_widths) > 1:
                # Serialized Fp8LinearMethod represents a fused per-tensor
                # weight with one scale slot per logical shard. Online FP8
                # produces one scalar for the fused tensor, so replicate that
                # scalar to the offline runtime shape without changing its
                # value.
                scale = scale.reshape(1).expand(len(logical_widths)).contiguous()
            write_tensor(
                _cache_key_for_runtime_name(f"{module_name}.weight_scale", source_prefix),
                scale,
            )

        index = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        }
        (component_root / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_kind": CACHE_KIND,
            "component": "transformer",
            "component_subfolder": CACHE_COMPONENT_SUBFOLDER,
            "source_prefix": source_prefix,
            "quantization": {
                "method": "fp8",
                "serialized": True,
                "activation_scheme": "dynamic",
            },
            "tensor_count": len(weight_map),
            "source_fingerprint": _source_fingerprint(source_files),
            "source_files": sorted(source_files),
        }
        (tmp_root / CACHE_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (tmp_root / CACHE_COMPLETE_NAME).write_text("phase1\n", encoding="utf-8")
        for handle in file_cache.values():
            close = getattr(handle, "__exit__", None)
            if callable(close):
                close(None, None, None)
        try:
            os.replace(tmp_root, cache_root)
        except FileExistsError:
            # Multiple DLO workers may build the same cache concurrently. If
            # another worker published a complete artifact first, treat this
            # publisher as a successful cache race rather than a build error.
            published = load_fp8_cache_manifest(cache_root)
            if published is None:
                raise
            shutil.rmtree(tmp_root, ignore_errors=True)
            logger.info("Normalized FP8 cache was published concurrently at %s", cache_root)
            return published
        logger.info("Published normalized FP8 cache at %s (%d tensors)", cache_root, len(weight_map))
        return manifest
    except BaseException:
        for handle in file_cache.values():
            close = getattr(handle, "__exit__", None)
            if callable(close):
                close(None, None, None)
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CACHE_KIND",
    "CACHE_COMPONENT_SUBFOLDER",
    "build_normalized_fp8_cache",
    "is_online_fp8_config",
    "load_fp8_cache_manifest",
    "_transformer_quant_config",
]
