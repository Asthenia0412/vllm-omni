# SPDX-License-Identifier: Apache-2.0
"""Opt-in parent media attribution. Standard library only; no network fields."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
import time

PROFILE_DIR = os.environ.get("LINGBOT_SP8_MEDIA_PROFILE_DIR")
ASYNC_DIR = os.environ.get("LINGBOT_TAIL_MEDIA_DIAGNOSTIC_DIR")
MEDIA_PROFILE_ENABLED = bool(PROFILE_DIR or ASYNC_DIR)
_LOCK = threading.Lock()
_WARNED = False
_ASYNC_OBSERVER = None
if ASYNC_DIR:
    # Create the bounded writer during import/loading. The encoding hot path
    # only appends in memory; JSON serialization and NFS I/O run in the writer.
    from vllm_omni.diffusion.layers.lingbot_tail_diagnostic import Observer

    _ASYNC_OBSERVER = Observer("media" + str(os.getpid()), ASYNC_DIR, device_events=False)


def emit_media_span(stage, start_ns, end_ns, *, request_id=None, generation_chunk_index=None,
                    num_frames=None, fps=None, byte_length=None, **details):
    """Write after the measured span. Diagnostic I/O never counts as encoding."""
    global _WARNED
    if not MEDIA_PROFILE_ENABLED:
        return
    row = {"schema_version": 1, "stage": stage, "pid": os.getpid(),
           "thread_id": threading.get_ident(), "request_id": request_id,
           "generation_chunk_index": generation_chunk_index, "num_frames": num_frames,
           "fps": fps, "byte_length": byte_length, "before_perf_counter_ns": start_ns,
           "after_perf_counter_ns": end_ns, "wall_ms": (end_ns - start_ns) / 1e6,
           "recorded_at_perf_counter_ns": time.perf_counter_ns(),
           "model_elapsed": False, "diagnostic_logging_outside_span": True, **details}
    if _ASYNC_OBSERVER is not None:
        # Observer supplies the same process identity itself.
        row.pop("pid")
        _ASYNC_OBSERVER.add("media_phase", **row)
        return
    try:
        line = json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
        with _LOCK:
            directory = Path(PROFILE_DIR)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"media_profile_pid{os.getpid()}.jsonl").open("a") as handle:
                handle.write(line)
    except (OSError, TypeError, ValueError):
        if not _WARNED:
            _WARNED = True
            logging.getLogger(__name__).exception("Optional media profiling could not write its diagnostic evidence")
