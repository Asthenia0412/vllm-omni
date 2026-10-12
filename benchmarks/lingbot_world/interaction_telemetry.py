# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only camera replay and evidence accounting; never pace generation."""
from __future__ import annotations

import math
from typing import Any

CAMERA_ACTIONS = ("forward", "left", "stop", "forward", "right", "stop")
CAMERA_DELAYS_MS = (0, 250, 750)


def camera_command(action: str) -> dict[str, Any]:
    if action not in CAMERA_ACTIONS:
        raise ValueError(f"Unknown camera action: {action}")
    yaw = math.radians(1) * (1 if action == "left" else -1 if action == "right" else 0)
    return {"mode": "velocity", "data": {
        "translation": [0.0, 0.0, 0.05 if action == "forward" else 0.0],
        "rotation": [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)],
    }}


def camera_replay_plan() -> list[dict[str, Any]]:
    """Trigger from full binary arrivals 0,10,...590, not compute boundaries."""
    return [{"event_id": f"camera-{index:02d}", "trigger_chunk_index": index * 10,
             "delay_ms": CAMERA_DELAYS_MS[index % 3], "action": CAMERA_ACTIONS[index % 6],
             "trigger_arrival_ms": None, "sent_at_ms": None, "queued_at_ms": None,
             "owner_samples": [], "errors": []} for index in range(60)]


def event_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {"type": "session.interaction", "interaction": {
        "event_id": row["event_id"],
        "event": {"multi_modal_data": {"camera": camera_command(row["action"])}},
    }}


def record_camera_owners(events: list[dict[str, Any]], metadata: dict[str, Any], arrival_ms: float) -> None:
    """Map real conditioning owners to encoder PTS, never claim visual change."""
    by_id = {row["event_id"]: row for row in events}
    timing = metadata.get("media_timing") or {}
    pts = timing.get("frame_pts", []) if timing.get("complete") is True else []
    for sample in metadata.get("camera_event_samples", []):
        row = by_id.get(sample.get("event_id"))
        if row is None:
            continue
        offset = sample.get("media_grid_frame_offset")
        pose = sample.get("translation", []) + sample.get("rotation", [])
        valid = (sample.get("evidence") == "camera_conditioning_pose_owner"
                 and sample.get("generation_chunk_index") == metadata.get("generation_chunk_index")
                 and isinstance(sample.get("latent_frame_index"), int)
                 and len(pose) == 7 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in pose))
        if not valid:
            row["errors"].append("invalid_camera_owner_sample")
            continue
        stamp = pts[offset].get("presentation_time_s") if isinstance(offset, int) and 0 <= offset < len(pts) else None
        row["owner_samples"].append(dict(sample, binary_arrival_ms=arrival_ms,
                                        control_grid_presentation_time_s=stamp))


def camera_replay_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    effective = [row for row in events if row["owner_samples"]]
    return {"planned_count": len(events),
            "sent_count": sum(row["sent_at_ms"] is not None for row in events),
            "queued_count": sum(row["queued_at_ms"] is not None for row in events),
            "effective_count": len(effective),
            "presentation_mapping_complete": all(
                any(sample.get("control_grid_presentation_time_s") is not None for sample in row["owner_samples"])
                for row in effective),
            "coverage_complete": len(effective) == len(events)
            and all(not row["errors"] and row["sent_at_ms"] is not None
                    and row["queued_at_ms"] is not None for row in events),
            "meaning": "Actual camera pose owners; PTS is a control-grid presentation proxy, not first visible pixel change",
            "events": events}
