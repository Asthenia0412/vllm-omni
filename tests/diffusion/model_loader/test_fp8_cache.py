# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the Phase-I normalized online-FP8 cache contract."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from vllm_omni.diffusion.model_loader.fp8_cache import (
    CACHE_COMPLETE_NAME,
    CACHE_COMPONENT_SUBFOLDER,
    CACHE_MANIFEST_NAME,
    _source_fingerprint,
    _transformer_quant_config,
    build_normalized_fp8_cache,
    is_online_fp8_config,
    load_fp8_cache_manifest,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _write_valid_cache(tmp_path):
    source = tmp_path / "source.safetensors"
    save_file({"weight": torch.ones(2, 2)}, str(source))

    cache = tmp_path / "cache"
    component = cache / CACHE_COMPONENT_SUBFOLDER
    component.mkdir(parents=True)
    shard = component / "part-00000.safetensors"
    save_file({"weight": torch.ones(2, 2, dtype=torch.float8_e4m3fn)}, str(shard))
    (component / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 4},
                "weight_map": {"weight": shard.name},
            }
        )
    )

    manifest = {
        "schema_version": 1,
        "cache_kind": "normalized_fp8_runtime",
        "component": "transformer",
        "component_subfolder": CACHE_COMPONENT_SUBFOLDER,
        "source_prefix": "transformer.",
        "quantization": {"method": "fp8", "serialized": True, "activation_scheme": "dynamic"},
        "tensor_count": 1,
        "source_files": [str(source)],
        "source_fingerprint": _source_fingerprint({str(source)}),
    }
    cache.mkdir(exist_ok=True)
    (cache / CACHE_MANIFEST_NAME).write_text(json.dumps(manifest))
    (cache / CACHE_COMPLETE_NAME).write_text("phase1\n")
    return cache, source, manifest


def test_online_and_serialized_fp8_configs_are_classified():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    assert is_online_fp8_config(Fp8Config())
    assert not is_online_fp8_config(Fp8Config(is_checkpoint_fp8_serialized=True))

    cache_config = _transformer_quant_config()
    assert not is_online_fp8_config(cache_config)
    assert cache_config.resolve("transformer").is_checkpoint_fp8_serialized
    assert cache_config.resolve("text_encoder") is None


def test_load_fp8_cache_manifest_validates_artifact_and_sources(tmp_path):
    cache, source, expected = _write_valid_cache(tmp_path)

    assert load_fp8_cache_manifest(cache) == expected

    source.write_bytes(source.read_bytes() + b"changed")
    assert load_fp8_cache_manifest(cache) is None


def test_load_fp8_cache_manifest_rejects_wrong_serialization_contract(tmp_path):
    cache, _, manifest = _write_valid_cache(tmp_path)

    manifest["quantization"]["serialized"] = False
    (cache / CACHE_MANIFEST_NAME).write_text(json.dumps(manifest))

    assert load_fp8_cache_manifest(cache) is None


def test_load_fp8_cache_manifest_rejects_path_traversal(tmp_path):
    cache, _, manifest = _write_valid_cache(tmp_path)
    index_path = cache / CACHE_COMPONENT_SUBFOLDER / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {"weight": "../outside.safetensors"}}))

    assert load_fp8_cache_manifest(cache) is None


def test_load_fp8_cache_manifest_rejects_non_object_json(tmp_path):
    cache, _, _ = _write_valid_cache(tmp_path)
    (cache / CACHE_MANIFEST_NAME).write_text("[]")

    assert load_fp8_cache_manifest(cache) is None


def test_build_normalized_fp8_cache_serializes_weight_and_scale(tmp_path):
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerTensorOnlineLinearMethod,
    )

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_file = source_dir / "model.safetensors"
    save_file({"block.weight": torch.ones((3, 2), dtype=torch.bfloat16)}, str(source_file))

    model = nn.Module()
    model._dit_modules = ("transformer",)
    model.transformer = nn.Module()
    model.transformer.block = nn.Module()
    model.transformer.block.weight = nn.Parameter(torch.ones((2, 3), dtype=torch.float8_e4m3fn), requires_grad=False)
    model.transformer.block.weight_scale = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
    model.transformer.block.quant_method = object.__new__(Fp8PerTensorOnlineLinearMethod)

    source = SimpleNamespace(
        model_or_path=str(source_dir),
        subfolder=None,
        revision=None,
        prefix="transformer.",
    )
    cache_dir = tmp_path / "cache"

    manifest = build_normalized_fp8_cache(
        model,
        sources=(source,),
        cache_dir=cache_dir,
    )

    assert manifest["cache_kind"] == "normalized_fp8_runtime"
    tensors = load_file(str(cache_dir / CACHE_COMPONENT_SUBFOLDER / "part-00000.safetensors"))
    scale_tensors = load_file(str(cache_dir / CACHE_COMPONENT_SUBFOLDER / "part-00001.safetensors"))
    assert tensors["block.weight"].dtype == torch.float8_e4m3fn
    assert tuple(tensors["block.weight"].shape) == (3, 2)
    assert torch.equal(scale_tensors["block.weight_scale"], torch.tensor([0.5]))


@pytest.mark.parametrize("missing", [CACHE_COMPLETE_NAME, CACHE_MANIFEST_NAME])
def test_load_fp8_cache_manifest_requires_completion_marker(tmp_path, missing):
    cache, _, _ = _write_valid_cache(tmp_path)
    (cache / missing).unlink()

    assert load_fp8_cache_manifest(cache) is None
