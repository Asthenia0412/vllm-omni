"""CPU-only actual encoder and parent attribution fixtures."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from vllm_omni.entrypoints.openai import lingbot_media_profile, serving_video_output_stream, video_api_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def rows(directory):
    return [json.loads(line) for path in directory.glob("*.jsonl") for line in path.read_text().splitlines()]


def test_opt_in_encoder_actual_nine_twelve_frames_is_byte_identical(monkeypatch, tmp_path):
    frames = [np.full((count, 32, 32, 3), index * 50, dtype=np.uint8) for index, count in enumerate((9, 12))]
    options = {"preset": "ultrafast", "threads": "0", "tune": "zerolatency"}
    monkeypatch.setattr(video_api_utils, "MEDIA_PROFILE_ENABLED", False)
    off = video_api_utils.FragmentedMP4VideoEncoder(fps=12, video_codec_options=options)
    expected = [off.encode(value) for value in frames]
    off.close()
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(video_api_utils, "MEDIA_PROFILE_ENABLED", True)
    monkeypatch.setattr(lingbot_media_profile, "PROFILE_DIR", str(tmp_path))
    on = video_api_utils.FragmentedMP4VideoEncoder(fps=12, video_codec_options=options)
    actual = []
    for index, value in enumerate(frames):
        on.set_media_profile_context(request_id="transport-cpu", generation_chunk_index=index, output_request_id="engine-cpu")
        actual.append(on.encode(value))
    on.close()
    assert actual == expected
    assert isinstance(on._muxer._stream.codec_context.thread_count, int)
    assert isinstance(on._muxer._stream.codec_context.name, str)
    evidence = rows(tmp_path)
    assert [row["num_frames"] for row in evidence] == [9, 12]
    assert [row["generation_chunk_index"] for row in evidence] == [0, 1]
    assert [row["initialized_this_chunk"] for row in evidence] == [True, False]
    for row in evidence:
        assert row["stage"] == "complete_encode" and row["fps"] == 12
        assert row["request_id"] == "transport-cpu" and row["output_request_id"] == "engine-cpu"
        assert row["before_perf_counter_ns"] <= row["normalized_at_perf_counter_ns"] <= row["mux_started_at_perf_counter_ns"] <= row["after_perf_counter_ns"]
        assert row["wall_ms"] == pytest.approx(row["normalization_ms"] + row["encoder_init_ms"] + row["mux_encode_ms"], abs=1e-9)
        assert row["model_elapsed"] is False and row["diagnostic_logging_outside_span"]


def test_jsonl_writer_is_thread_safe(monkeypatch, tmp_path):
    monkeypatch.setattr(lingbot_media_profile, "PROFILE_DIR", str(tmp_path))
    def writer():
        for index in range(20):
            lingbot_media_profile.emit_media_span("cpu_fixture", index, index + 1, num_frames=12, fps=12)
    workers = [threading.Thread(target=writer) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert len(rows(tmp_path)) == 40


@pytest.mark.asyncio
async def test_actual_parent_hook_binds_session_and_preserves_network_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(lingbot_media_profile, "PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(video_api_utils, "MEDIA_PROFILE_ENABLED", True)
    monkeypatch.setattr(serving_video_output_stream, "MEDIA_PROFILE_ENABLED", True)
    handler = serving_video_output_stream.OmniStreamingVideoOutputHandler(
        engine_client=SimpleNamespace(), model_name="cpu", stage_configs=[], stall_timeout=5, start_timeout=5)
    async def build(_request):
        return {}, SimpleNamespace(extra_args={}), SimpleNamespace(fps=12)
    async def generate(*_args):
        for index, count in enumerate((9, 12)):
            yield SimpleNamespace(error=None, request_id="engine-source", images=[], multimodal_output={
                "video": np.zeros((count, 32, 32, 3), dtype=np.uint8),
                "metadata": {"stream": {"generation_chunk_index": index}}})
    handler._build_prompt_and_sampling_params = build
    handler._iter_generation_outputs = generate
    class WebSocket:
        def __init__(self):
            self.queue = asyncio.Queue()
            self.queue.put_nowait(json.dumps({"type": "session.start", "prompt": "cpu", "fps": 12}))
            self.sent = []
        async def accept(self):
            pass
        async def receive_text(self):
            return await self.queue.get()
        async def send_json(self, value):
            json.dumps(value)
            self.sent.append(value.copy())
        async def send_bytes(self, value):
            self.sent.append(value)
    websocket = WebSocket()
    await handler.handle_session(websocket)
    messages = [value for value in websocket.sent if isinstance(value, dict)]
    assert messages[0]["type"] == "video.start" and messages[-1]["type"] == "session.done"
    request_id = messages[0]["request_id"]
    metadata = [value for value in messages if value["type"] == "video.chunk_metadata" and value["kind"] == "media"]
    assert [row["num_frames"] for row in metadata] == [9, 12]
    assert all("before_perf_counter_ns" not in row and "media_profile" not in row for row in metadata)
    evidence = rows(tmp_path)
    assert {row["stage"] for row in evidence} >= {"complete_encode", "chunk_metadata_build", "websocket_send_json", "websocket_send_bytes"}
    assert all(row["request_id"] == request_id and row["fps"] == 12 and not row["model_elapsed"] for row in evidence)
    assert [row["generation_chunk_index"] for row in evidence if row["stage"] == "complete_encode"] == [0, 1]
