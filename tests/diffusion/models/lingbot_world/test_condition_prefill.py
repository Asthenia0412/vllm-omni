# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact conditioning prefill, bounded convergence, retry and source-key checks."""

import gc
import weakref

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.layers.lingbot_condition_prefill import (
    ConditionPrefillDidNotConverge,
    build_condition_prefill,
    condition_source_digest,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def identical(before, after):
    return before is not None and after is not None and len(before) == len(after) and all(
        x is y if x is None or y is None else torch.equal(x, y) for x, y in zip(before, after, strict=True)
    )


def finite_encoder(*, block_frames=3, dtype=torch.bfloat16, history=4):
    references, calls = [], []

    def encode(start, cache):
        calls.append(start)
        feature = torch.arange(1, history + 1, dtype=dtype) if cache is None else torch.cat(
            (cache[0][1:], cache[0].new_zeros(1))
        )
        references.append(weakref.ref(feature))
        condition = feature.sum().expand(1, 20, block_frames, 2, 4).clone()
        condition[:, :4] = 0
        if start == 0:
            condition[:, :4, 0] = 1
        return condition, [feature]

    return encode, references, calls


@pytest.mark.parametrize("block_frames", [1, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_transient_replay_and_fixed_point_match_unskipped_encoding(block_frames, dtype):
    encode, references, calls = finite_encoder(block_frames=block_frames, dtype=dtype)
    bank = build_condition_prefill(
        key=("session", "source", dtype), block_frames=block_frames,
        encode_chunk=encode, caches_identical=identical,
    )
    encoded_calls = len(calls)
    assert calls == [index * block_frames for index in range(encoded_calls)]
    assert 1 < encoded_calls < 32
    gc.collect()
    assert all(reference() is None for reference in references), "feature histories must not remain resident"

    reference_encode, _, _ = finite_encoder(block_frames=block_frames, dtype=dtype)
    cache = None
    for index in range(20):
        expected, cache = reference_encode(index * block_frames, cache)
        actual, pending = bank.replay(start_frame=index * block_frames, key=bank.key)
        assert torch.equal(actual, expected)
        assert actual.dtype == dtype and pending == []
    assert len(calls) == encoded_calls
    assert bank.resident_bytes == sum(value.numel() * value.element_size() for value in bank.conditions)


def test_prefill_retry_does_not_advance_or_expose_mutable_bank():
    encode, _, _ = finite_encoder()
    bank = build_condition_prefill(key="same-session", block_frames=3, encode_chunk=encode, caches_identical=identical)
    original, pending = bank.replay(start_frame=0, key=bank.key)
    expected = original.clone()
    original.zero_()
    pending.append(None)
    retried, retried_pending = bank.replay(start_frame=0, key=bank.key)
    assert torch.equal(retried, expected) and retried_pending == []
    with pytest.raises(ValueError, match="key changed"):
        bank.replay(start_frame=0, key="new-source-or-session")
    with pytest.raises(ValueError, match="align"):
        bank.replay(start_frame=1, key=bank.key)


def test_nonconvergence_is_bounded_and_does_not_publish_a_bank():
    calls = []

    def encode(start, cache):
        calls.append(start)
        return torch.zeros(1, 20, 3, 2, 4), [torch.tensor(start)]

    with pytest.raises(ConditionPrefillDidNotConverge, match="within 5 chunks"):
        build_condition_prefill(
            key="session", block_frames=3, encode_chunk=encode, caches_identical=identical, max_chunks=5
        )
    assert calls == [0, 3, 6, 9, 12]


def test_source_digest_tracks_values_dtype_shape_and_mutation():
    image = Image.new("RGB", (4, 2), (1, 2, 3))
    original = condition_source_digest(image)
    assert condition_source_digest(image.copy()) == original
    image.putpixel((0, 0), (4, 5, 6))
    assert condition_source_digest(image) != original
    tensor = torch.zeros(3, 2, 4, dtype=torch.bfloat16)
    original = condition_source_digest(tensor)
    assert condition_source_digest(tensor.clone()) == original
    assert condition_source_digest(tensor.float()) != original
    assert condition_source_digest(tensor.reshape(3, 4, 2)) != original
    tensor[0, 0, 0] = 1
    assert condition_source_digest(tensor) != original


def test_prefill_matches_real_causal_wan_encoder_for_first_and_steady_blocks():
    from diffusers import AutoencoderKLWan

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        vae = AutoencoderKLWan(
            base_dim=4, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
            temperal_downsample=[False, True, True],
        ).eval()
        image = torch.randn(1, 3, 1, 16, 32)
    zeros = torch.zeros(1, 3, 4, 16, 32)

    @torch.no_grad()
    def encode(start, cache):
        pending = [None] * vae._cached_conv_counts["encoder"] if cache is None else list(cache)
        features = [
            vae.encoder(image if start == 0 and frame == 0 else zeros, feat_cache=pending, feat_idx=[0])
            for frame in range(3)
        ]
        latents = vae.quant_conv(torch.cat(features, dim=2)).chunk(2, dim=1)[0]
        mask = latents.new_zeros(1, 4, *latents.shape[2:])
        if start == 0:
            mask[:, :, 0] = 1
        return torch.cat((mask, latents), dim=1).to(torch.bfloat16), pending

    bank = build_condition_prefill(key="real-wan", block_frames=3, encode_chunk=encode, caches_identical=identical)
    assert 0 < bank.settled_chunk_index < 31
    cache = None
    for index in range(16):
        expected, cache = encode(index * 3, cache)
        actual, _ = bank.replay(start_frame=index * 3, key=bank.key)
        assert torch.equal(actual, expected)
    assert torch.count_nonzero(bank.conditions[0][:, :4]) > 0
    assert torch.count_nonzero(bank.conditions[1][:, :4]) == 0
