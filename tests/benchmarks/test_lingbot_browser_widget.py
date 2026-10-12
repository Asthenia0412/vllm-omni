"""Real Chromium + synthetic CPU fMP4 fixture, never accelerator evidence."""
from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import threading
from contextlib import contextmanager

import numpy as np
import pytest

from vllm_omni.diffusion.utils.media_utils import FragmentedMP4Muxer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def browser_probe():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("browser_probe_cpu", root / "profiling/lingbot_browser_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    try:
        import playwright  # noqa: F401
    except ImportError:
        pytest.skip("Optional Playwright tools are not installed")
    return probe


@contextmanager
def cpu_server(stream):
    from websockets.asyncio.server import serve
    ready, finished = threading.Event(), threading.Event()
    address = []
    async def main():
        async with serve(stream, "127.0.0.1", 0) as listener:
            address.append(listener.sockets[0].getsockname()[1])
            ready.set()
            while not finished.is_set():
                await asyncio.sleep(.05)
    worker = threading.Thread(target=lambda: asyncio.run(main()), daemon=True)
    worker.start()
    assert ready.wait(5)
    try:
        yield f"ws://127.0.0.1:{address[0]}/v1/realtime/video"
    finally:
        finished.set()
        worker.join(5)


def test_real_browser_keeps_playing_after_generation_done():
    from websockets.asyncio.server import serve
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("browser_probe_cpu", root / "profiling/lingbot_browser_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    try:
        import playwright  # noqa: F401
    except ImportError:
        pytest.skip("Optional Playwright tools are not installed")
    ready, finished = threading.Event(), threading.Event()
    address = []
    async def stream(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "video.start", "request_id": "synthetic-browser"}))
        muxer = FragmentedMP4Muxer(width=32, height=32, fps=12, media_timing=True,
            video_codec_options={"preset": "ultrafast", "tune": "zerolatency"})
        for index, count in enumerate((9, 12, 12)):
            chunk = muxer.mux_video_frames(np.full((count, 32, 32, 3), index * 50, dtype=np.uint8))
            await ws.send(json.dumps({"type": "video.chunk_metadata", "kind": "media",
                "generation_chunk_index": index, "num_frames": count, "media_timing": muxer.last_media_timing,
                "camera_event_samples": []}))
            await ws.send(chunk)
            await asyncio.sleep(.05)
        trailer = muxer.close()
        if trailer:
            await ws.send(json.dumps({"type": "video.chunk_metadata", "kind": "trailer", "num_frames": 0}))
            await ws.send(trailer)
        await ws.send(json.dumps({"type": "session.done", "stopped": False}))
    async def main():
        async with serve(stream, "127.0.0.1", 0) as listener:
            address.append(listener.sockets[0].getsockname()[1])
            ready.set()
            while not finished.is_set():
                await asyncio.sleep(.05)
    worker = threading.Thread(target=lambda: asyncio.run(main()), daemon=True)
    worker.start()
    assert ready.wait(5)
    try:
        result = probe.observe({"url": f"ws://127.0.0.1:{address[0]}/v1/realtime/video",
            "playback_buffer_chunks": 2, "interaction_telemetry": True,
            "payload": {"type": "session.start", "num_frames": 33, "fps": 12}}, timeout_s=10)
    finally:
        finished.set()
        worker.join(5)
    assert result["observation_source"] == "real_chromium_widget"
    assert result["playback_complete"] and result["transport_complete"]
    assert result["playback_ended_ms"] > result["generation_done_ms"] + 1500
    assert result["media_duration_s"] == pytest.approx(33 / 12, abs=1 / 12)
    assert result["underrun_count"] == 0 and result["total_stall_ms"] == 0
    assert not result["browser_page_errors"] and not result["observation_timed_out"]


def test_real_browser_controls_owners_pts_prompt_and_post_done_rejection():
    probe = browser_probe()
    received = []
    async def stream(ws):
        start = json.loads(await ws.recv())
        assert start["extra_params"] == {"camera_event_samples": True, "media_timing": True}
        await ws.send(json.dumps({"type": "video.start"}))
        active = None
        prompts = []
        async def controls():
            nonlocal active
            async for message in ws:
                value = json.loads(message)
                if value["type"] == "session.interaction":
                    interaction = value["interaction"]
                    received.append(interaction)
                    if "multi_modal_data" in interaction["event"]:
                        active = interaction
                    else:
                        prompts.append(interaction["event_id"])
                    await ws.send(json.dumps({"type": "session.interaction.queued", "event_id": interaction["event_id"]}))
        task = asyncio.create_task(controls())
        muxer = FragmentedMP4Muxer(width=32, height=32, fps=12, media_timing=True,
            video_codec_options={"preset": "ultrafast", "tune": "zerolatency"})
        try:
            for index in range(8):
                count = 9 if index == 0 else 12
                chunk = muxer.mux_video_frames(np.full((count, 32, 32, 3), index * 20, dtype=np.uint8))
                samples = []
                if active:
                    pose = active["event"]["multi_modal_data"]["camera"]["data"]
                    samples = [{"event_id": active["event_id"], "generation_chunk_index": index,
                        "latent_frame_index": 0, "media_grid_frame_offset": 0, **pose,
                        "evidence": "camera_conditioning_pose_owner"}]
                await ws.send(json.dumps({"type": "video.chunk_metadata", "kind": "media",
                    "generation_chunk_index": index, "num_frames": count, "media_timing": muxer.last_media_timing,
                    "camera_event_samples": samples, "started_event_ids": prompts,
                    "completed_event_ids": prompts}))
                prompts = []
                await ws.send(chunk)
                await asyncio.sleep(.35)
            trailer = muxer.close()
            if trailer:
                await ws.send(json.dumps({"type": "video.chunk_metadata", "kind": "trailer", "num_frames": 0}))
                await ws.send(trailer)
            await ws.send(json.dumps({"type": "session.done", "stopped": False}))
            await ws.wait_closed()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    plan = [{"trigger_chunk_index": index, "action": "camera", "value": action}
            for index, action in enumerate(("forward", "left", "right", "stop"))]
    plan.extend([{"trigger_chunk_index": 1, "action": "prompt", "value": "A bright boardwalk.", "transition_chunks": 0},
                 {"trigger": "generation_done", "action": "camera", "value": "forward"},
                 {"trigger": "generation_done", "action": "prompt", "value": "Post done."}])
    with cpu_server(stream) as url:
        result = probe.observe({"url": url, "playback_buffer_chunks": 2, "interaction_telemetry": True,
            "payload": {"type": "session.start", "num_frames": 93, "fps": 12}}, control_plan=plan, timeout_s=15)
    assert result["playback_complete"] and not result["browser_page_errors"]
    assert [row["accepted_by_client"] for row in result["control_actions"]] == [True] * 5 + [False] * 2
    assert len(received) == 5 and len(result["post_done_events"]) == 2
    assert len(result["camera_events"]) == 4
    for event in result["camera_events"]:
        assert event["queued_at_ms"] is not None and not event["errors"]
        assert any(sample["control_grid_presented_at_ms"] is not None for sample in event["owner_samples"])
        assert all(sample["control_grid_presentation_time_s"] is not None for sample in event["owner_samples"])
    prompt = result["prompt_events"][0]
    assert prompt["queued_at_ms"] is not None and prompt["started_media_arrival_ms"] is not None
    assert prompt["completed_media_arrival_ms"] is not None


def test_real_browser_cancel_records_server_stopped_ack():
    probe = browser_probe()
    stop_received = threading.Event()
    async def stream(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "video.start"}))
        async def controls():
            async for message in ws:
                if json.loads(message)["type"] == "session.stop":
                    stop_received.set()
        task = asyncio.create_task(controls())
        muxer = FragmentedMP4Muxer(width=32, height=32, fps=12,
            video_codec_options={"preset": "ultrafast", "tune": "zerolatency"})
        try:
            for index in range(20):
                if stop_received.is_set():
                    break
                count = 9 if index == 0 else 12
                chunk = muxer.mux_video_frames(np.zeros((count, 32, 32, 3), dtype=np.uint8))
                await ws.send(json.dumps({"type": "video.chunk_metadata", "kind": "media",
                    "generation_chunk_index": index, "num_frames": count}))
                await ws.send(chunk)
                await asyncio.sleep(.15)
            await ws.send(json.dumps({"type": "session.done", "stopped": stop_received.is_set()}))
            await ws.wait_closed()
        finally:
            muxer.close()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    with cpu_server(stream) as url:
        result = probe.observe({"url": url, "playback_buffer_chunks": 2,
            "payload": {"type": "session.start", "num_frames": 237, "fps": 12}},
            control_plan=[{"trigger_chunk_index": 2, "action": "cancel"}], timeout_s=10)
    assert stop_received.is_set() and result["control_actions"][0]["accepted_by_client"]
    assert result["cancel_requested_ms"] is not None and result["cancel_ack_ms"] is not None
    assert result["session_stopped"] and result["cancelled"] and not result["transport_complete"]
    assert not result["observation_timed_out"] and not result["browser_page_errors"]
