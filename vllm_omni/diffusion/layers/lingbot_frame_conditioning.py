# SPDX-License-Identifier: Apache-2.0
"""Frame-run conditioning primitives for a separately gated NPU experiment.

The default-off uniform route is connected to the LingBot model after its
original BF16 frame expansion, contiguous SP shard and shape checks. The
general frame-run primitives remain separate from that route. Values are
never inspected on the host during normal inference; complete FP32
normalization remains outside these primitives. Combining uniform timestep
with the independent Cast-once flag has not passed NPU acceptance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os

import torch

_frame_capture_session = None
_frame_capture_probes = set()


def _flag(value):
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value in ("0", "1", "false", "true", "False", "True"):
        return value in ("1", "true", "True")
    raise ValueError("LingBot uniform timestep flag must be an explicit boolean")


def configure_uniform_timestep(transformer, model_config=None, *, environ=None):
    values = os.environ if environ is None else environ
    enabled = _flag(values.get("LINGBOT_NPU_UNIFORM_TIMESTEP", "0"))
    name = "lingbot_npu_uniform_timestep"
    if isinstance(model_config, Mapping) and name in model_config:
        enabled = _flag(model_config[name])
    elif model_config is not None and hasattr(model_config, name):
        enabled = _flag(getattr(model_config, name))
    transformer._lingbot_uniform_timestep = enabled
    return dict(uniform_timestep=enabled, default_enabled=False,
                per_frame_timestep_policy="original dense token path",
                arithmetic_contract="complete original FP32 norm, addcmul modulation, separate gate mul/add, BF16 rounding",
                sequence_parallel_input_dtype_changed=False)


def prepare_uniform_timestep(transformer, projection, timestep):
    """Compact only an enabled NPU request with one timestep per batch.

    This runs after the original BF16 SP preparation and shape checks. The
    four canonical probe inputs and original three-frame GEMM rows must first
    pass the separate pre-SP capture bit check. Per-frame timestep inputs
    take the untouched original token conditioning path.
    """
    enabled = getattr(transformer, "_lingbot_uniform_timestep", False)
    if not enabled or projection.device.type != "npu" or timestep.ndim != 1:
        return projection, False
    if projection.ndim != 4 or projection.shape[2] != 6 or timestep.shape[0] != projection.shape[0]:
        raise ValueError("Uniform timestep requires original SP4 projection and one timestep per batch")
    return projection.narrow(1, 0, 1), True


def capture_frame_projection(transformer, timestep, projection):
    """Diagnostic-only four small original pre-SP three-frame probe packs."""
    if os.environ.get("LINGBOT_CAPTURE_FRAME_TIMESTEP", "0") != "1":
        return
    from . import lingbot_capture

    writer = lingbot_capture._active_capture
    if writer is None or writer.chunk != 1 or writer.probe not in range(4):
        return
    global _frame_capture_session
    if _frame_capture_session is None:
        _frame_capture_session = writer.session_id
    if writer.session_id != _frame_capture_session or writer.probe in _frame_capture_probes:
        return
    from pathlib import Path

    _frame_capture_probes.add(writer.probe)
    values = dict(projection=projection.detach().to("cpu").contiguous(),
                  timestep=timestep.detach().to("cpu").contiguous(),
                  first_block_modulation=transformer.blocks[0].modulation.detach().to("cpu").contiguous())
    metadata = dict(source_module="transformer._timestep_embeddings.output.before_sp_prepare",
                    projection_dtype=str(projection.dtype), original_projection_stride=list(projection.stride()),
                    original_timestep_shape=list(timestep.shape), timestep_values=values["timestep"].tolist(),
                    uniform_route_requires="one timestep per batch; original frame projection rows bit-identical",
                    normal_inference_has_no_d2h=True, maximum_probe_packs=4)
    path = Path(writer.directory) / f"fusion_chunk001_probe{writer.probe}_frame_timestep_projection.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(dict(kind="frame_timestep_projection", tensors=values, metadata=metadata,
                    chunk=writer.chunk, probe=writer.probe, session_id=writer.session_id,
                    timing_eligible=False), temporary)
    temporary.replace(path)
    writer.activation_records.append(dict(kind="frame_timestep_projection", region="frame_timestep_projection",
        path=path.name, chunk=writer.chunk, probe=writer.probe,
        shapes={name: list(value.shape) for name, value in values.items()}))


def frame_token_runs(
    *, frames: int, tokens_per_frame: int, token_offset: int, token_count: int
) -> tuple[tuple[int, int], ...]:
    """Return (global frame index, local token count) for a contiguous shard."""
    if any(type(value) is not int for value in (frames, tokens_per_frame, token_offset, token_count)):
        raise ValueError("Frame-run geometry must contain integer values")
    if frames <= 0 or tokens_per_frame <= 0 or token_count <= 0 or token_offset < 0:
        raise ValueError("Frame-run dimensions/count must be positive and offset nonnegative")
    stop = token_offset + token_count
    if stop > frames * tokens_per_frame:
        raise ValueError("Frame-run shard exceeds the original frame sequence")
    result = []
    cursor = token_offset
    while cursor < stop:
        frame = cursor // tokens_per_frame
        length = min(stop, (frame + 1) * tokens_per_frame) - cursor
        result.append((frame, length))
        cursor += length
    return tuple(result)


def _lengths(lengths: Sequence[int], tokens: int) -> tuple[int, ...]:
    result = tuple(lengths)
    if not result or any(type(value) is not int or value <= 0 for value in result) or sum(result) != tokens:
        raise ValueError("Frame-run lengths must be positive integers spanning every local token")
    return result


def compact_timestep_projection(projection: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    """Select one original conditioning row per proven constant frame run.

    No arithmetic or dtype change occurs here. Projection extraction can
    launch NPU Slice/Concat kernels; their costs belong to the full call.
    """
    if projection.ndim != 4 or projection.shape[2] != 6:
        raise ValueError("Expected token projection (batch, tokens, 6, dim)")
    lengths = _lengths(lengths, projection.shape[1])
    parts = []
    offset = 0
    for length in lengths:
        parts.append(projection.narrow(1, offset, 1))
        offset += length
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


def _validate_operands(value: torch.Tensor, conditioning: torch.Tensor, lengths: Sequence[int]):
    if value.ndim != 3:
        raise ValueError("Expected hidden values (batch, tokens, dim)")
    lengths = _lengths(lengths, value.shape[1])
    if conditioning.shape != (value.shape[0], len(lengths), 1, value.shape[2]):
        raise ValueError("Expected compact conditioning (batch, runs, 1, dim)")
    if conditioning.dtype != torch.float32:
        raise ValueError("Frame-run modulation/gates must retain the original FP32 boundary")
    if conditioning.device != value.device:
        raise ValueError("Hidden values and conditioning must share a device")
    return lengths


def modulate_frame_runs(
    normalized: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    lengths: Sequence[int],
    *,
    output_dtype: torch.dtype,
    use_addcmul: bool,
) -> torch.Tensor:
    """Apply the original FP32 modulation to an already normalized full input.

    Preserve the serving path's choice of addcmul versus separate multiply
    and add. Each run is rounded to the original output dtype before concat;
    concatenation performs no arithmetic. Norm reductions are never split.
    """
    lengths = _validate_operands(normalized, scale, lengths)
    _validate_operands(normalized, shift, lengths)
    if normalized.dtype != torch.float32 or type(use_addcmul) is not bool:
        raise ValueError("Full normalization must be FP32 and addcmul selection explicit")
    parts = []
    offset = 0
    for run, length in enumerate(lengths):
        values = normalized.narrow(1, offset, length)
        # Select retains the singleton run axis to match the existing 4-D
        # broadcasting contract, then remove only the singleton token axis.
        run_scale = scale.narrow(1, run, 1).squeeze(2)
        run_shift = shift.narrow(1, run, 1).squeeze(2)
        factor = 1 + run_scale
        result = torch.addcmul(run_shift, values, factor) if use_addcmul else values * factor + run_shift
        parts.append(result.to(output_dtype))
        offset += length
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


def gated_residual_frame_runs(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    lengths: Sequence[int],
) -> torch.Tensor:
    """Keep FP32 gate multiply, residual add, and the final BF16/FP16 cast."""
    lengths = _validate_operands(hidden, gate, lengths)
    if update.shape != hidden.shape or update.device != hidden.device or update.dtype != hidden.dtype:
        raise ValueError("Residual update must match the original hidden tensor")
    parts = []
    offset = 0
    for run, length in enumerate(lengths):
        residual = hidden.narrow(1, offset, length)
        values = update.narrow(1, offset, length)
        run_gate = gate.narrow(1, run, 1).squeeze(2)
        # Do not replace with addcmul: the existing served residual path has
        # an explicit product and addition, followed by output rounding.
        parts.append((residual + values * run_gate).to(hidden.dtype))
        offset += length
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
