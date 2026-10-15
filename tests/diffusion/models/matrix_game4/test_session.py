# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request and session ownership checks; no checkpoints or CUDA required."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.matrix_game4.pipeline_matrix_game4 import (
    DEFAULT_PROMPT,
    END_KEY,
    SESSION_KEY,
    START_KEY,
    MatrixGame4Pipeline,
    prepare_request,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def request(frames=1, session="one", start=0, end=False):
    media = {} if frames is None else {"video": torch.zeros(frames, 480, 864, 3, dtype=torch.uint8)}
    return OmniDiffusionRequest(
        prompt={"prompt": "", "multi_modal_data": media},
        request_id=f"{session}-{start}",
        sampling_params=OmniDiffusionSamplingParams(
            seed=42, num_inference_steps=1, extra_args={SESSION_KEY: session, START_KEY: start, END_KEY: end}
        ),
    )


class RecordingStream:
    def __init__(
        self,
        transformer,
        vae,
        prompt_embeds,
        *,
        seed,
        max_latents,
        steps,
        first_steps,
        sink_latent,
        sink_mode,
        sink_rope_mode,
    ):
        self.frame = self.index = 0
        self.closed = False
        self.fail = False
        self.offsets = []

    def push(self, pixels, *, frame_start):
        if self.fail:
            raise RuntimeError("inference failed")
        assert frame_start == self.frame
        self.offsets.append(frame_start)
        # Different chunks carry distinct values, exposing accidental storage reuse.
        output = torch.full((pixels.shape[0], 3, 480, 864), self.index - 1, dtype=torch.float32)
        self.frame += pixels.shape[0]
        self.index += 1
        return SimpleNamespace(pixels=output)

    def close(self):
        self.closed = True


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr("vllm_omni.diffusion.models.matrix_game4.stream.MatrixStream", RecordingStream)
    model = MatrixGame4Pipeline.__new__(MatrixGame4Pipeline)
    nn.Module.__init__(model)
    model.prompt_text = DEFAULT_PROMPT
    model.sink_mode, model.sink_rope_mode = "kv_only", "fixed"
    model.max_latents = 64
    model.runtime = model.vae = model.prompt_embeds = model.sink_latent = None
    model._stream = model._session_id = model._steps = None
    return model


def run(pipeline, req):
    return pipeline(DiffusionRequestBatch([req]))[0]


def test_continuation_owns_pixels_and_explicit_close(pipeline):
    first = run(pipeline, request())
    stream = pipeline._stream
    second = run(pipeline, request(4, start=1))
    assert stream.offsets == [0, 1]
    assert first.media.video.tensor.shape == (1, 3, 1, 480, 864)
    assert second.media.video.tensor.shape == (1, 3, 4, 480, 864)
    assert first.media.video.tensor.eq(-1).all()
    assert second.media.video.tensor.eq(0).all()
    first.media.validate()
    run(pipeline, request(None, start=5, end=True))
    assert stream.closed and pipeline._stream is None
    assert run(pipeline, request(session="two")).media.video.tensor.eq(-1).all()


def test_foreign_and_out_of_order_requests_preserve_owner(pipeline):
    run(pipeline, request())
    stream = pipeline._stream
    for req in [request(session="two"), request(None, session="two", end=True), request(4, start=5)]:
        with pytest.raises(ValueError):
            run(pipeline, req)
        assert pipeline._stream is stream and not stream.closed and stream.frame == 1
    run(pipeline, request(4, start=1, end=True))
    assert stream.closed


def test_failure_releases_session(pipeline):
    run(pipeline, request())
    stream = pipeline._stream
    stream.fail = True
    with pytest.raises(RuntimeError, match="inference failed"):
        run(pipeline, request(4, start=1))
    assert stream.closed and pipeline._stream is None
    assert run(pipeline, request(session="two")).media.video.tensor.shape[2] == 1


def test_complete_clip_uses_one_then_four_and_resets(pipeline):
    output = run(pipeline, request(9, session=None, end=True))
    assert pipeline._stream is None
    assert output.media.video.tensor.shape == (1, 3, 9, 480, 864)
    assert output.media.video.tensor[0, 0, :, 0, 0].tolist() == [-1] + [0] * 4 + [1] * 4
    assert output.custom_output["session_closed"]


@pytest.mark.parametrize(
    "frames,session,start,end",
    [
        (4, "one", 0, False),
        (1, "one", 1, False),
        (2, None, 0, True),
        (1, None, 1, True),
        (None, "one", 0, False),
        (1, "", 0, False),
        (1, "one", True, False),
        (1, "one", -1, False),
        (1, "one", 0, "yes"),
    ],
)
def test_invalid_session_geometry(frames, session, start, end):
    with pytest.raises(ValueError):
        prepare_request(request(frames, session, start, end))


@pytest.mark.parametrize(
    "field,value",
    [
        ("height", 720),
        ("width", 640),
        ("num_inference_steps", 4),
        ("num_inference_steps", 0),
        ("num_outputs_per_prompt", 2),
        ("guidance_scale", 7.0),
        ("output_type", "latent"),
    ],
)
def test_unsupported_sampling(field, value):
    req = request()
    setattr(req.sampling_params, field, value)
    with pytest.raises(ValueError):
        prepare_request(req)


def test_unknown_options_and_wrong_rgb_dtype():
    req = request()
    req.sampling_params.extra_args["typo"] = True
    with pytest.raises(ValueError, match="Unknown"):
        prepare_request(req)
    req = request()
    req.prompt["multi_modal_data"]["video"] = req.prompt["multi_modal_data"]["video"].float()
    with pytest.raises(ValueError, match="uint8"):
        prepare_request(req)


def test_budget_rejection_does_not_acquire_session(pipeline):
    pipeline.max_latents = 1
    with pytest.raises(ValueError, match="budget"):
        run(pipeline, request(9, session=None, end=True))
    assert pipeline._stream is None


def test_runner_prepares_float_media_for_transport(pipeline):
    from vllm_omni.diffusion.data import OmniDiffusionConfig, VideoOutputTransportConfig
    from vllm_omni.diffusion.postprocess.device_reduction import prepare_diffusion_media_for_transport

    req = request(9, session=None, end=True)
    output = run(pipeline, req)
    assert not output.media.prepared_for_transport
    prepared = prepare_diffusion_media_for_transport(
        output.media,
        od_config=OmniDiffusionConfig(
            video_output_transport=VideoOutputTransportConfig(enable_device_postprocess=True)
        ),
        sampling_params=req.sampling_params,
    )
    assert prepared.prepared_for_transport
    assert prepared.video.tensor.dtype == torch.uint8
    assert prepared.video.tensor.shape == (1, 9, 480, 864, 3)
    assert prepared.video.tensor[0, :, 0, 0, 0].tolist() == [0] + [128] * 4 + [255] * 4


def test_engine_generator_does_not_revalidate_prepared_request(pipeline):
    req = prepare_request(request())
    # The engine attaches its own generator after model-specific preprocessing.
    req.sampling_params.generator = torch.Generator().manual_seed(42)
    assert run(pipeline, req).custom_output["frame_count"] == 1


def test_omitted_seed_uses_native_default():
    req = request()
    req.sampling_params.seed = None
    assert prepare_request(req).sampling_params.seed == 0


def test_temporal_cache_rewrite_and_eviction_preserve_sink():
    from vllm_omni.diffusion.models.matrix_game4.transformer import TemporalCache

    cache = TemporalCache(torch.empty(1, 3, 1, 1), torch.empty(1, 3, 1, 1))

    def write(position, value):
        tensor = torch.full((1, 1, 1, 1), float(value))
        cache.write(tensor, -tensor, position)

    for position in range(3):
        write(position, position)
    write(2, 22)  # Clean KV replaces noisy KV without consuming another slot.
    assert cache.key.flatten().tolist() == [0, 1, 22]
    write(3, 3)
    assert cache.key.flatten().tolist() == [0, 22, 3]
    assert cache.value.flatten().tolist() == [0, -22, -3]
    before = cache.key.clone()
    with pytest.raises(ValueError, match="consecutive"):
        write(5, 5)
    assert torch.equal(cache.key, before) and cache.position == 3


def test_ring_cache_moves_previous_current_once_and_preserves_sink():
    from vllm_omni.diffusion.models.matrix_game4.transformer import TemporalCache

    cache = TemporalCache(torch.empty(1, 5, 1, 1), torch.empty(1, 5, 1, 1), ring=True)
    for position in range(6):
        tensor = torch.full((1, 1, 1, 1), float(position))
        cache.write(tensor, -tensor, position)
    assert cache.key.flatten().tolist() == [0, 4, 2, 3, 5]
    cache.write(torch.full((1, 1, 1, 1), 55.0), torch.full((1, 1, 1, 1), -55.0), 5)
    cache.prepare(6, 1)
    cache.prepare(6, 1)  # Denoise and clean passes must not advance the ring again.
    assert cache.key.flatten().tolist() == [0, 4, 55, 3, 55]
    assert cache.value.flatten().tolist() == [0, -4, -55, -3, -55]


@pytest.mark.parametrize("option", ["cuda_graph", "fp8"])
def test_optimization_options_require_booleans(option):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.matrix_game4.pipeline_matrix_game4 import validate_config

    config = OmniDiffusionConfig(
        model="matrix-resources",
        model_class_name="MatrixGame4Pipeline",
        enforce_eager=True,
        max_num_seqs=1,
        diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}},
        model_config={option: "false"},
    )
    with pytest.raises(ValueError, match="boolean"):
        validate_config(config)


@pytest.mark.parametrize("backend", [None, "TORCH_SDPA", "CUDNN_ATTN"])
def test_quality_path_requires_qualified_attention_backend(backend):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.matrix_game4.pipeline_matrix_game4 import validate_config

    attention = {} if backend is None else {"default": {"backend": backend}}
    config = OmniDiffusionConfig(
        model="matrix-resources",
        model_class_name="MatrixGame4Pipeline",
        enforce_eager=True,
        max_num_seqs=1,
        diffusion_attention_config=attention,
    )
    if backend == "TORCH_SDPA":
        validate_config(config)
    else:
        with pytest.raises(ValueError, match="quality qualification"):
            validate_config(config)
