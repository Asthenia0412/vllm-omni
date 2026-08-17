# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform-neutral contract for registering existing host mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import torch


class HostRegistrationError(RuntimeError):
    """Raised when existing host ranges cannot be registered safely."""


class HostRegistrationBudgetError(HostRegistrationError):
    """Raised before mutation when the complete mapping exceeds its budget."""


class HostRegistrationCleanupError(HostRegistrationError):
    """Raised when partial registration cannot be rolled back safely."""


class HostRegistration(Protocol):
    """Lifecycle owned by DLO for one platform's registered host ranges."""

    @property
    def total_bytes(self) -> int: ...

    @property
    def region_count(self) -> int: ...

    def close(self) -> list[str]: ...


def register_host_mappings(
    sources_by_mapping: Mapping[str, Sequence[torch.Tensor]],
    *,
    device: torch.device,
    max_bytes: int,
) -> HostRegistration:
    """Register mappings with the active platform or report unsupported use.

    CUDA is the first implementation. Other platforms retain DLO's bounded
    host-staging path until they provide an equivalent registration backend.
    """
    if device.type == "cuda":
        from .cuda_host_registration import CudaHostRegistration

        return CudaHostRegistration.create(sources_by_mapping, max_bytes=max_bytes)
    raise HostRegistrationError(f"host-mapping registration is not supported on {device.type}")


__all__ = [
    "HostRegistration",
    "HostRegistrationBudgetError",
    "HostRegistrationCleanupError",
    "HostRegistrationError",
    "register_host_mappings",
]
