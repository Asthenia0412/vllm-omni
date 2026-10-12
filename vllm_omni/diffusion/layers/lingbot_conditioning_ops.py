# SPDX-License-Identifier: Apache-2.0
"""Opt-in reuse of identical conditioning conversions; no arithmetic change."""
from collections.abc import Mapping
import os

_projection_capture_session = None
_projection_captures = set()


def _flag(value):
    if isinstance(value, bool):
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        if value in ("1", "true", "True"):
            return True
        if value in ("0", "false", "False"):
            return False
    raise ValueError("LingBot timestep FP32-once flag must be an explicit boolean")


def configure_timestep_fp32_once(transformer, model_config=None, *, environ=None):
    values = os.environ if environ is None else environ
    enabled = _flag(values.get("LINGBOT_NPU_TIMESTEP_FP32_ONCE", "0"))
    name = "lingbot_npu_timestep_fp32_once"
    if isinstance(model_config, Mapping) and name in model_config:
        enabled = _flag(model_config[name])
    elif model_config is not None and hasattr(model_config, name):
        enabled = _flag(getattr(model_config, name))
    transformer._lingbot_timestep_fp32_once = enabled
    return {"timestep_fp32_once": enabled, "sequence_parallel_input_dtype_changed": False,
            "position": "after BF16 SP preparation, before optional static staging and block loop",
            "arithmetic_contract": "same deterministic cast reused; block modulation remains out-of-place FP32"}


def prepare_timestep_projection(transformer, projection):
    """Convert after communication and before staging to retain static address.

    No cache mutation: .float() preserves an FP32 input or allocates one FP32
    value for a BF16 input. Every block's subsequent .float() becomes identity.
    """
    if os.environ.get("LINGBOT_CAPTURE_TIMESTEP", "0") == "1":
        _capture_timestep_projection(transformer, projection)
    if not getattr(transformer, "_lingbot_timestep_fp32_once", False):
        return projection
    return projection.float()


def _capture_timestep_projection(transformer, projection):
    """Four original operands in chunk 1 of the first diagnostic session.

    The independent gate avoids extending normal activation captures. The raw
    writer exists only on rank 0; this helper restricts it to the first session.
    At SP4/1170 tokens,
    four BF16 projections occupy 274.22 MiB plus four small modulation weights.
    """
    from . import lingbot_capture

    writer = lingbot_capture._active_capture
    if writer is None or writer.chunk != 1 or writer.probe not in range(4):
        return
    global _projection_capture_session
    if _projection_capture_session is None:
        _projection_capture_session = writer.session_id
    if writer.session_id != _projection_capture_session or writer.probe in _projection_captures:
        return
    from pathlib import Path
    import torch

    _projection_captures.add(writer.probe)
    first = transformer.blocks[0]
    values = {"projection": projection.detach().to("cpu").contiguous(),
              "modulation": first.modulation.detach().to("cpu").contiguous()}
    metadata = dict(source_module="transformer.sp_prepare.output.timestep_projection",
                    projection_dtype=str(projection.dtype), projection_stride=list(projection.stride()),
                    cast_position="after BF16 SP preparation; before optional FP32 conversion/staging",
                    arithmetic_contract="block modulation + original_projection.float(), out-of-place",
                    maximum_projection_captures=4)
    path = Path(writer.directory) / f"fusion_chunk001_probe{writer.probe}_timestep_projection.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(dict(kind="timestep_projection", tensors=values, metadata=metadata,
                    chunk=writer.chunk, probe=writer.probe, session_id=writer.session_id,
                    timing_eligible=False), temporary)
    temporary.replace(path)
    writer.activation_records.append(dict(kind="timestep_projection", region="timestep_projection",
        path=path.name, chunk=writer.chunk, probe=writer.probe,
        shapes={name: list(value.shape) for name,value in values.items()}))
