# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact, bounded precomputation of LingBot's static causal conditioning.

The image/zero-input encoder path is independent of generated latents, camera,
text and sampling RNG. Retain its small output conditions, not feature-cache
snapshots, and stop only after the existing bitwise cache comparator proves a
fixed point. The pipeline owns the opt-in switch and session lifetime.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

import torch

EncoderCache = list[torch.Tensor | None]


class ConditionPrefillDidNotConverge(RuntimeError):
    """The bounded preparation did not prove a fixed point; normal encoding is required."""


def condition_source_digest(image: Any) -> str:
    """Fingerprint the exact source at session opening, without an identity cache.

    Tensor bytes include dtype/shape and preserve BF16 bit patterns. A device
    tensor is copied once to CPU for this opening fingerprint. PIL images are
    normalized to the same RGB interpretation used by image preparation.
    Callers should compute this at opening/retry, not hash images every layer.
    """
    digest = hashlib.sha256()
    if isinstance(image, torch.Tensor):
        source = image.detach().to(device="cpu").contiguous()
        digest.update(f"tensor:{source.dtype}:{tuple(source.shape)}:".encode())
        digest.update(source.view(torch.uint8).numpy().tobytes())
    else:
        from PIL import Image

        if not isinstance(image, Image.Image):
            raise TypeError("condition prefill source must be a tensor or PIL image")
        source = image.convert("RGB")
        digest.update(f"rgb:{source.size}:".encode())
        digest.update(source.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ConditionPrefillBank:
    """Per-session exact transient conditions followed by one verified steady block."""

    key: Hashable
    block_frames: int
    conditions: tuple[torch.Tensor, ...]

    @property
    def settled_chunk_index(self) -> int:
        return len(self.conditions) - 1

    @property
    def resident_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.conditions)

    def replay(self, *, start_frame: int, key: Hashable) -> tuple[torch.Tensor, EncoderCache]:
        """Read by absolute latent-frame offset; retries do not consume a cursor.

        The fresh empty pending-cache list is deliberately a completion token:
        the original rolling feature caches are no longer needed. A pipeline
        using this path must keep it active until session reset/close.
        """
        if key != self.key:
            raise ValueError("condition prefill key changed; reset/rebuild the session conditioning")
        if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0:
            raise ValueError("condition start_frame must be a nonnegative latent-frame index")
        if start_frame % self.block_frames:
            raise ValueError("condition start_frame must align with the configured latent block")
        index = min(start_frame // self.block_frames, self.settled_chunk_index)
        # The transformer/caller cannot mutate a later retry or the fixed point.
        return self.conditions[index].clone(), []


@torch.no_grad()
def build_condition_prefill(
    *,
    key: Hashable,
    block_frames: int,
    encode_chunk: Callable[[int, EncoderCache | None], tuple[torch.Tensor, EncoderCache]],
    caches_identical: Callable[[EncoderCache | None, EncoderCache | None], bool],
    max_chunks: int = 32,
) -> ConditionPrefillBank:
    """Run the original encoder callback in order and retain no feature snapshots.

    ``encode_chunk`` receives the absolute latent ``start_frame`` and only the
    preceding chunk's cache. It must follow Wan's existing replacement/list-copy
    semantics, preserving its input cache for the bitwise convergence check.
    No speculative fixed-point condition is substituted for an early chunk.
    Failure raises before publication; callers may opt for normal encoding.
    """
    if isinstance(block_frames, bool) or not isinstance(block_frames, int) or block_frames <= 0:
        raise ValueError("block_frames must be a positive integer")
    if isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks < 2:
        raise ValueError("condition prefill max_chunks must be an integer of at least two")
    hash(key)
    conditions = []
    cache: EncoderCache | None = None
    for index in range(max_chunks):
        condition, pending = encode_chunk(index * block_frames, cache)
        if not isinstance(condition, torch.Tensor) or condition.ndim != 5 or condition.shape[2] != block_frames:
            raise ValueError("condition prefill requires a [batch, channels, latent-frames, height, width] block")
        if conditions:
            previous = conditions[0]
            if (
                condition.shape != previous.shape
                or condition.dtype != previous.dtype
                or condition.device != previous.device
            ):
                raise ValueError("condition prefill shape/device/dtype changed within the session")
        conditions.append(condition.detach().clone())
        if index > 0 and caches_identical(cache, pending):
            return ConditionPrefillBank(key=key, block_frames=block_frames, conditions=tuple(conditions))
        cache = pending
    raise ConditionPrefillDidNotConverge(
        f"LingBot condition encoder did not prove a fixed point within {max_chunks} chunks"
    )
