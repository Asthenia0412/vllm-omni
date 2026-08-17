# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic identity helpers for the DLO runtime-weight cache."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import TypeAlias

import torch

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
PathIdentity: TypeAlias = str | os.PathLike[str] | os.PathLike[bytes]


class IdentityNormalizationError(ValueError):
    """Raised when a cache identity contains a process-unstable value."""


def _type_identity(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def canonicalize_existing_local_path(value: PathIdentity) -> str:
    """Resolve equivalent local paths while leaving model repository IDs alone."""
    original = os.fsdecode(os.fspath(value))
    candidate = Path(original).expanduser()
    try:
        if candidate.exists():
            return str(candidate.resolve())
    except OSError:
        pass
    return original


def normalize_identity(value: object) -> JsonValue:
    """Convert supported loader inputs into deterministic JSON-compatible data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return {"enum": _type_identity(value), "name": value.name}
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, os.PathLike):
        return canonicalize_existing_local_path(value)
    if isinstance(value, type):
        return _type_identity(value)
    if dataclasses.is_dataclass(value):
        return normalize_identity(dataclasses.asdict(value))

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except (TypeError, ValueError):
            dumped = model_dump()
        return normalize_identity(dumped)

    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise IdentityNormalizationError("runtime-cache identity mappings require string keys")
            normalized[key] = normalize_identity(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (set, frozenset)):
        return sorted((normalize_identity(item) for item in value), key=canonical_json)
    if isinstance(value, (list, tuple)):
        return [normalize_identity(item) for item in value]

    raise IdentityNormalizationError(f"runtime-cache identity does not support values of type {_type_identity(value)}")


def canonical_json(value: object) -> bytes:
    """Serialize a supported identity with stable ordering and separators."""
    return json.dumps(
        normalize_identity(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


__all__ = [
    "IdentityNormalizationError",
    "JsonValue",
    "canonical_json",
    "canonicalize_existing_local_path",
    "normalize_identity",
]
