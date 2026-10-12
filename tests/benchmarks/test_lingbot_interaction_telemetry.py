from __future__ import annotations

import importlib.util
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.lingbot_world.interaction_telemetry import camera_replay_plan, camera_replay_summary, record_camera_owners
from benchmarks.lingbot_world.benchmark_lingbot_world_realtime import run_session
from benchmarks.lingbot_world.workload import default_workload
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.interaction.modality_handlers.camera import SE3DeltaCameraHandler
from vllm_omni.diffusion.interaction.types import InteractionChunkMetadata, merge_interaction_metadata
from vllm_omni.diffusion.output_formatter import _stream_metadata_from_diffusion_output
from vllm_omni.diffusion.utils.media_utils import FragmentedMP4Muxer
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def state(enabled):
    return StepRequestState(request_id="cpu", prompt="hello",
        sampling=SimpleNamespace(extra_args={"camera_event_samples": enabled}))


def enqueue(handler, value, event_id, *, mode="velocity", translation=(0, 0, .05), received_at=0):
    handler.enqueue(value, event_id=event_id, received_at=received_at,
        payload={"mode": mode, "data": {"translation": list(translation), "rotation": [0, 0, 0, 1]}},
        transition_chunks=0)


def apply(handler, value):
    return handler.apply_at_chunk_boundary(value, boundary_at=1, chunk_index=7,
        num_media_frames=12, fps=12, num_latent_frames=3)


def test_samples_default_off_and_preserve_pose_values():
    handler = SE3DeltaCameraHandler()
    a, b = state(False), state(True)
    for value in (a, b):
        enqueue(handler, value, "forward")
    off, on = apply(handler, a), apply(handler, b)
    assert off.camera_event_samples is None
    assert "camera_event_samples" not in off.as_dict()
    assert torch.equal(a.interaction_sessions["camera"].last_absolute_poses,
                       b.interaction_sessions["camera"].last_absolute_poses)
    assert [x["latent_frame_index"] for x in on.camera_event_samples] == [0, 1, 2]
    assert [x["media_grid_frame_offset"] for x in on.camera_event_samples] == [0, 4, 8]


def test_immediate_target_is_owner_even_after_completion():
    handler, value = SE3DeltaCameraHandler(), state(True)
    enqueue(handler, value, "snap", mode="target", translation=(0, 0, 1))
    meta = apply(handler, value)
    assert meta.active_event_ids == [] and meta.completed_event_ids == ["snap"]
    assert len(meta.camera_event_samples) == 1
    assert meta.camera_event_samples[0]["event_id"] == "snap"
    assert meta.camera_event_samples[0]["translation"] == [0, 0, 1]


def test_started_but_superseded_event_is_not_a_pose_owner():
    handler, value = SE3DeltaCameraHandler(), state(True)
    enqueue(handler, value, "discarded")
    enqueue(handler, value, "stop", translation=(0, 0, 0))
    meta = apply(handler, value)
    assert meta.started_event_ids == ["discarded", "stop"]
    assert {x["event_id"] for x in meta.camera_event_samples} == {"stop"}
    assert all(x["translation"] == [0, 0, 0] for x in meta.camera_event_samples)


def test_metadata_forwarding_omits_opt_in_key_when_disabled():
    meta = InteractionChunkMetadata(camera_event_samples=[{"event_id": "camera"}])
    combined = merge_interaction_metadata([InteractionChunkMetadata(), meta])
    assert combined.camera_event_samples == meta.camera_event_samples
    off = _stream_metadata_from_diffusion_output(DiffusionOutput(finished=False))
    on = _stream_metadata_from_diffusion_output(DiffusionOutput(finished=False, camera_event_samples=[]))
    assert "camera_event_samples" not in off and on["camera_event_samples"] == []


def test_actual_encoder_pts_are_continuous_across_opening_and_regular_chunk():
    muxer = FragmentedMP4Muxer(width=32, height=32, fps=12, media_timing=True,
        video_codec_options={"preset": "ultrafast", "tune": "zerolatency"})
    all_pts = []
    for count in (9, 12):
        assert muxer.mux_video_frames(np.zeros((count, 32, 32, 3), dtype=np.uint8))
        timing = muxer.last_media_timing
        assert timing["complete"] and timing["source"] == "encoder_packet_pts"
        all_pts.extend(x["presentation_time_s"] for x in timing["frame_pts"])
    muxer.close()
    assert all_pts == pytest.approx([i / 12 for i in range(21)])


def test_replay_count_uses_pose_owners_and_not_queued_ack():
    events = camera_replay_plan()
    assert [x["trigger_chunk_index"] for x in events] == list(range(0, 600, 10))
    events[0]["sent_at_ms"] = 10
    events[0]["queued_at_ms"] = 20
    assert camera_replay_summary(events)["effective_count"] == 0
    record_camera_owners(events, {"generation_chunk_index": 1,
        "media_timing": {"complete": True, "frame_pts": [{"presentation_time_s": .75}]},
        "camera_event_samples": [{"event_id": "camera-00", "generation_chunk_index": 1,
            "latent_frame_index": 0, "media_grid_frame_offset": 0,
            "translation": [0, 0, .05], "rotation": [0, 0, 0, 1],
            "evidence": "camera_conditioning_pose_owner"}]}, 900)
    summary = camera_replay_summary(events)
    assert summary["effective_count"] == 1 and not summary["coverage_complete"]


def acceptance_module():
    path = Path(__file__).resolve().parents[3] / "profiling/lingbot_sp8_acceptance.py"
    spec = importlib.util.spec_from_file_location("acceptance_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_screen_and_final_gate_are_distinct():
    mod = acceptance_module()
    baseline = [{"chunks": 161, "mean_ms": x, "p99_ms": 1050} for x in (1000, 1002, 1001)]
    candidate = [{"chunks": 161, "mean_ms": x, "p99_ms": 1055} for x in (994, 996, 995)]
    assert mod.candidate_gate(baseline, candidate)["passed"]
    assert not mod.final_gate(candidate)["passed"]
    candidate[0]["p99_ms"] += .01
    assert not mod.candidate_gate(baseline, candidate)["passed"]


@pytest.mark.asyncio
async def test_fast_generation_done_does_not_fabricate_or_wait_for_sixty_events():
    from websockets.asyncio.server import serve
    received = []
    async def server(websocket):
        start = json.loads(await websocket.recv())
        assert start["extra_params"]["camera_event_samples"] is True
        assert "camera_action_script" not in start["extra_params"]
        await websocket.send(json.dumps({"type": "video.start", "request_id": "cpu-replay"}))
        async def controls():
            async for message in websocket:
                value = json.loads(message)
                if value["type"] == "session.interaction":
                    received.append(value["interaction"]["event_id"])
                    await websocket.send(json.dumps({"type": "session.interaction.queued",
                        "event_id": received[-1]}))
        control = asyncio.create_task(controls())
        try:
            for index in range(600):
                await websocket.send(json.dumps({"type": "video.chunk_metadata", "kind": "media",
                    "generation_chunk_index": index, "num_frames": 9 if index == 0 else 12}))
                await websocket.send(b"synthetic-cpu-media")
                await asyncio.sleep(.001)
            await websocket.send(json.dumps({"type": "session.done", "stopped": False}))
        finally:
            control.cancel()
            await asyncio.gather(control, return_exceptions=True)
    async with serve(server, "127.0.0.1", 0) as listener:
        port = listener.sockets[0].getsockname()[1]
        result = await run_session(f"ws://127.0.0.1:{port}/v1/realtime/video", "cpu",
            default_workload(image_reference="data:image/png;base64,Zm9v", num_chunks=600, fps=12),
            first_chunk_timeout=3, chunk_timeout=3, ping_interval=0, collect_media=False,
            print_chunks=False, camera_event_replay=True)
    assert result.wall_s < 3
    assert len(result.records) == 600
    telemetry = result.interaction_telemetry
    assert telemetry["planned_count"] == 60
    assert telemetry["effective_count"] == 0 and not telemetry["coverage_complete"]
    assert received == sorted(received)
    assert telemetry["sent_count"] < 60


def test_receipt_rejects_reused_service_and_browser_requires_actual_end():
    mod = acceptance_module()
    runs = [{"path": f"/cpu/B{index}.json", "chunks": 600} for index in range(1, 4)]
    receipt = {"services": [{"arm": "B", "pair": index, "benchmark": run["path"],
        "server_pid": 100, "server_starttime": "same", "warmup_chunks": 40,
        "measured_chunks": 600, "profiler_active": False} for index, run in enumerate(runs, 1)]}
    assert not mod.fresh_process_gate(receipt, [], runs)["passed"]
    assert not mod.browser_gate({"transport_complete": True, "playback_complete": False,
        "generation_done_ms": 100, "underrun_count": 0, "total_stall_ms": 0})["passed"]
