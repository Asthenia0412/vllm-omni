# Matrix-Game 4.0 — single RTX 5090 deployment target

Native geometry-to-video inference through vLLM-Omni's experimental Python
`DiffusionEngine`. The deployment target is **one RTX 5090 with 32 GB VRAM**.
Local validation uses **one RTX PRO 5000 72GB Blackwell** (PCI ID
`2BB3:10DE:227A`, 73,415 MiB reported). RTX 5090 fit and performance are not yet
qualified. A separate whole-device NVML sample at 100 ms intervals observed
23.13 GiB during cold startup on the PRO 5000, including the CUDA context. The reference preset name `rtx5090` is not a measurement of that SKU.

Maintainer: @lishunyang12. Model:
[Skywork/Matrix-Game-4.0](https://huggingface.co/Skywork/Matrix-Game-4.0).
Status: experimental Python integration; default BF16 quality path and opt-in acceleration.

## What this runs

A game engine produces low-poly RGB frames. The pipeline defaults to 864×480,
accepting one opening frame followed by four frames per chunk. The first latent
is followed by one new latent per chunk, with 21 latent frames of temporal KV.
Castle and Egypt use a KV-only sink with relative positioning; Racing uses its
sink image as the first output frame and absolute positions.

Gameplay, collision, keyboard/mouse input, browser transport and public serving
are application responsibilities. GPT-6 or another coding-agent API is **not** a
runtime dependency. This change does not add an HTTP/WebSocket game endpoint.

## Install

Use Linux, Python 3.12 and an NVIDIA driver compatible with your selected vLLM /
PyTorch CUDA build. Start with a fresh environment. From this integration branch:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm==0.31.0 --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/db9527a46873454610df6dbedf79a36d6bf1a7f6
uv pip install -e .
```

See the [CUDA source installation guide](../../docs/getting_started/installation/gpu.md)
for a different CUDA build. The bounded local comparisons below used Python
3.12.13, PyTorch 2.13.0+cu130, vLLM 0.31.0, Transformers 5.14.1, Diffusers 0.40.0
and NVIDIA driver 580.95.05. Requalify outputs when changing the software stack.

No external Matrix-Game package is installed for Omni inference.

## Download and arrange resources

The source repositories use different directory names. The following commands
create the layout consumed by this pipeline; they do not invoke upstream code.
Download only the scenes you need. Each scene checkpoint is approximately 10 GB;
the shared text checkpoint is approximately 11.4 GB. Only one scene is loaded per
worker. Host RAM must accommodate checkpoint construction and text encoding.

```bash
MATRIX_RESOURCES="$PWD/matrix-resources"
MATRIX_REV=4419accdb7bb7881234c5914de8cd915266e0325
WAN_REV=b8bc1a65ab71d054ba4636dc0dac104aa4df2686

hf download alibaba-pai/Wan2.2-Fun-5B-Control \
  --revision "$WAN_REV" \
  --include 'models_t5_umt5-xxl-enc-bf16.pth' 'google/umt5-xxl/*' \
  --local-dir "$MATRIX_RESOURCES/base"

hf download Skywork/Matrix-Game-4.0 --revision "$MATRIX_REV" \
  --include 'base/taew2_2.pth' 'LICENSE' 'THIRD_PARTY_NOTICES.md' 'licenses/*' \
    'fps-castle/model.safetensors' 'fps-castle/sink.png' 'fps-castle/inference.json' \
    'fps-egypt/model.safetensors' 'fps-egypt/sink.png' 'fps-egypt/inference.json' \
    'driving_game/model.safetensors' 'driving_game/sink.png' 'driving_game/inference.json' \
  --local-dir "$MATRIX_RESOURCES"

mkdir -p "$MATRIX_RESOURCES/models"
ln -s ../fps-castle "$MATRIX_RESOURCES/models/castle"
ln -s ../fps-egypt "$MATRIX_RESOURCES/models/ancient-egypt"
ln -s ../driving_game "$MATRIX_RESOURCES/models/racing"
```

These commands assume a fresh resource directory. The required layout is:

```text
matrix-resources/
  base/
    models_t5_umt5-xxl-enc-bf16.pth
    google/umt5-xxl/                 # tokenizer files
    taew2_2.pth
  models/
    castle/                         # or ancient-egypt / racing
      model.safetensors
      sink.png
      inference.json
```

`inference.json` supplies the released scene caption. The model checks its input
geometry, temporal window, guidance and scheduler contract before loading.

## Run on one GPU

Save actual engine geometry as `geometry.npy`, a CPU uint8 RGB array of shape
`[T,480,864,3]`. Do not use neural output frames as geometry controls. A complete
clip contains `1+4k` frames. For a 241-frame qualification:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/diffusion/benchmark_matrix_game4.py \
  --scene castle --resources ./matrix-resources --input ./geometry.npy \
  --output ./qualification/omni --frames 241 --steps 1
```

Select `--scene ancient-egypt` or `--scene racing` with matching geometry input.
`--steps 2` selects timesteps 1000 and 500 before the shift-5 scheduler mapping.
Changing scene requires a new worker; each scene has its own checkpoint.

The benchmark constructs this engine configuration:

```python
config = OmniDiffusionConfig(
    model="./matrix-resources",
    model_class_name="MatrixGame4Pipeline",
    dtype="bfloat16",
    num_gpus=1,
    max_num_seqs=1,
    distributed_executor_backend="uni",
    enforce_eager=True,
    diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}},
    model_config={"scene": "castle", "max_latents": 8192},
    video_output_transport=VideoOutputTransportConfig(
        enable_device_postprocess=True,
    ),
)
```

Use the shared SDPA backend for the qualified BF16 path. `enforce_eager=True`
disables generic whole-pipeline graph wrapping. Keep it enabled even when using
the model-owned graph switch below; session bookkeeping must stay outside capture.
The text encoder runs once during startup and is released before gameplay.
No CPU inference offload or generic quantization is enabled.

## Experimental 1280×736 resolution

Use a new worker with `model_config.height=736` and `model_config.width=1280`.
Only 864×480 (default) and 1280×736 are currently accepted. Geometry input and
output use the selected dimensions, including the opening frame and every
continuation chunk. Request dimensions must match the worker or be omitted.
Changing resolution requires a new worker; 1280×720 is not accepted because the
codec and patch embedding require spatial multiples of 32.

Capture actual engine geometry at 1280×736, saved as uint8 `[T,736,1280,3]`:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/diffusion/benchmark_matrix_game4.py \
  --scene racing --resources ./matrix-resources --input ./geometry-736.npy \
  --height 736 --width 1280 --frames 1201 --steps 1 --cuda-graph --fp8 \
  --output ./qualification/racing-736-fp8-1
```

For a BF16 two-step comparison, omit `--fp8` and use `--steps 2`. Both use the
tiny TAEW2 codec; neither is the upstream full-Wan-VAE quality preset. Changing
both precision and steps does not isolate either factor's quality contribution.

The released checkpoint and profile remain unchanged. Sink resizing, latent
shape, KV capacity, spatial RoPE and graph descriptors follow the worker size:
405 tokens per latent frame at 864×480, 920 at 1280×736. This is native generation
at a larger latent grid, but it is **resolution extrapolation**, not a model
trained or reference-qualified at the larger size. More pixels do not establish
better texture fidelity or temporal stability. The pinned reference runtime
only accepts 864×480, so `--mode reference` rejects the larger size. Multi-GPU
execution is still unsupported.

Bounded Racing evaluation on one RTX PRO 5000 72GB Blackwell, same software stack
as above, seed 42, SDPA, graph replay, TAEW2, one 1,201-frame run per configuration:

| Native resolution | Precision / steps | Generated FPS | Chunk p50 / p95 (ms) |
| --- | --- | --- | --- |
| 864×480 | FP8 / 1 | 61.38 | 65.13 / 65.59 |
| 1280×736 | FP8 / 1 | 27.52 | 145.55 / 146.49 |
| 1280×736 | BF16 / 2 | 11.24 | 356.16 / 356.98 |

The game rendered 1920×1104 geometry, downsampled to 1280×736. The 480p control
was center-cropped and downsampled from that same capture, preserving the control
sequence but with a slightly different aspect ratio. The first 25 chunks are
excluded, leaving 276 measured four-frame chunks (1,104 generated frames).
Timing covers CPU RGB input through CPU RGB output, including the Omni engine,
codec and clean-KV update; it excludes loading, session close, game rendering,
video encoding and network transport. Saved clips play at the simulation's
60 FPS and last 20.02 seconds; that is not the generation rate. These single-run
measurements do not establish eight-GPU scaling, live response latency, RTX 5090
performance, or superior visual quality. A separate 481-frame 480p FP8 regression
remained byte-identical to the previous implementation's output.

## Input, output and session lifetime

| Item | Contract |
| --- | --- |
| Input field | `prompt["multi_modal_data"]["video"]` |
| Input tensor | CPU `torch.uint8`, RGB `[T,H,W,3]`; defaults to 480×864 |
| First chunk | One frame with `matrix_frame_start=0` |
| Continuation | Four frames with offsets `1,5,9,...` |
| Complete clip | Omit the session ID, supply `1+4k` frames; closes automatically |
| Output | `result.images[0]`: uint8 `[1,T,H,W,3]`, same frame count and worker dimensions |
| Sampling | Seed, one or two steps, guidance scale 1, `output_type="np"` |
| Caption | Fixed at startup; optionally override `model_config.prompt` |
| Budget | At most 8,192 generated latents, configurable downward |
| Concurrency | One resident session per worker; serialize requests |

Named sessions use sampling `extra_args`:

```python
{
    "matrix_session_id": "player-1",
    "matrix_frame_start": 1,
    "matrix_end_session": False,
}
```

The initial request uses offset zero. Set `matrix_end_session=True` on the last
chunk, or omit video to close the owning session explicitly. Close in a `finally`
block; this API has no idle timeout. Conflicting session IDs and incorrect
offsets are rejected without taking ownership. Inference failure clears state.

The current serial session path does not implement the framework's experimental
[AR-Diffusion session/paged-KV protocol](../../docs/design/feature/realtime_ar_diffusion.md).
Reusing `DiffusionEngine` does not imply its routing or concurrency guarantees.

## Compare quality against the reference

The reference is an isolated oracle, never an Omni runtime dependency. Keep
checkpoint, input, caption, seed, precision, codec, timesteps and software equal.
The current qualification uses **BF16 + TAEHV**. It does not compare the TAEHV
result to the upstream `quality` preset's full Wan VAE.

```bash
git clone https://github.com/SkyworkAI/Matrix-Game.git matrix-reference
git -C matrix-reference checkout 5446525c80da2a4a2f87801addbbc7faa8d2486e
PYTHONPATH=./matrix-reference/Matrix-Game-4 \
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/diffusion/benchmark_matrix_game4.py \
  --mode reference --reference-runtime native --scene castle \
  --resources ./matrix-resources --input ./geometry.npy \
  --output ./qualification/reference --frames 241 --steps 1 \
  --compare ./qualification/omni.npy --require-exact
```

The two commands write `.npy` output and JSON timings, pixel MAE, maximum error
and exact-match status. A 241-frame castle test at seed 42 matched every decoded
RGB byte, including KV-window eviction and sink repositioning. The shared UMT5
encoder also matched the reference embedding exactly. The final shared-codec implementation also matched every RGB byte for all six
121-frame pairs: Castle, Ancient Egypt and Racing, each at one and two steps
(seed 42, 864×480). All cases cross the 21-latent cache window. This is bounded
qualification, not a guarantee for arbitrary-length sessions. Three additional
1,201-frame (20.02-second) comparisons, one per scene at two steps, also matched
every RGB byte, including captured movement, firing, steering and braking.

The correctness path preserves original BF16 RMSNorm rounding, separate Q/K/V
GEMMs, chronological temporal KV order and the convolution implementation of the
shared patch embedding. Packed QKV and convolution-to-GEMM were measured to
change results on SM120. Optimizations need their own quality comparison before
being enabled. Do not mix FP8 runtime results with BF16 results.

## Opt-in acceleration

The default remains BF16 eager execution. Enable fixed-shape graph replay with
`model_config.cuda_graph=True` or the benchmark's `--cuda-graph`. It captures the
steady DiT window after fill and continued codec chunks using vLLM's existing
platform graph wrapper. Graph warmup preserves codec history, replay returns
owned output buffers, and closing a session releases its graphs and caches.

Add `model_config.fp8=True` or `--fp8` for the experimental throughput path:

```python
model_config={"scene": "castle", "cuda_graph": True, "fp8": True}
```

It uses vLLM's per-tensor E4M3 quantizer/scaled GEMM with separate Q/K/V weight
scales and one shared input quantization. PyTorch compiles the tensor-only
normalization/residual operations; temporal KV uses a ring instead of shifting
the whole window. Text KV is computed once. Embeddings, output head and the
shared TAEW2 codec remain BF16. The first run includes compilation and graph
warmup; neither is included in the reported steady FPS.

FP8 output is **not pixel-identical** to the original optimized FP8 runtime.
Fused reductions differ numerically, and long autoregressive clips exhibit
different effects even with identical inputs and seed. This is a throughput
comparison, not a passed visual-quality gate. Both switches are off by default;
use BF16 for exact-reference work. Do not pass `--require-exact` for FP8 expecting
it to succeed. The tested PyTorch/vLLM versions are listed above; other compiler
versions and GPUs need requalification.

Run the paired throughput comparison on the same idle GPU, in separate
processes, with matching scene/geometry, 481 frames and 25 warmup chunks:

```bash
PYTHONPATH=./matrix-reference/Matrix-Game-4 \
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
  python benchmarks/diffusion/benchmark_matrix_game4.py \
  --mode reference --reference-runtime rtx5090 --scene castle \
  --resources ./matrix-resources --input ./geometry.npy \
  --output ./qualification/reference-fp8 --frames 481 --steps 1

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
  python benchmarks/diffusion/benchmark_matrix_game4.py \
  --scene castle --resources ./matrix-resources --input ./geometry.npy \
  --output ./qualification/omni-fp8 --frames 481 --steps 1 \
  --cuda-graph --fp8 --compare ./qualification/reference-fp8.npy
```

Repeat with `--steps 2` for two-step denoising. The timer starts with owned CPU
uint8 RGB input and ends after CPU uint8 RGB output. It includes normalization,
encoding, denoising, decoding and clean-KV update; Omni includes engine scheduling
and output postprocessing as well. Loading, the first 25 chunks and explicit
session close are excluded from both sides. At 481 frames this measures 96
steady four-frame chunks (384 generated frames). This is generated throughput,
not a display frame rate or browser response-time measurement.

One RTX PRO 5000 72GB Blackwell, 864×480, seed 42, same scene checkpoints and
actual geometry capture; software versions as listed above. Each row is one
paired 481-frame run (not an average across repeated runs):

| Scene | Steps | Original FP8 FPS | Omni FP8 FPS | Original / Omni p95 (ms) |
| --- | --- | --- | --- | --- |
| castle | 1 | 59.72 | 60.50 | 69.27 / 66.91 |
| castle | 2 | 41.71 | 41.14 | 101.28 / 98.12 |
| ancient-egypt | 1 | 57.18 | 60.03 | 77.16 / 67.09 |
| ancient-egypt | 2 | 41.34 | 40.94 | 100.92 / 98.68 |
| racing | 1 | 59.52 | 61.37 | 71.00 / 65.72 |
| racing | 2 | 42.04 | 41.50 | 100.12 / 96.77 |

All six FP8 pairs are non-exact. RGB MAE ranges from 13.28 to 27.59 on the 0–255
scale; this measures pixel disagreement, not perceived quality. Throughput is
within 1.4% of the original in the slower cases; faster cases are not a claim of
a statistically significant speedup. RTX 5090 performance has not been measured.

Separately, BF16 with `--cuda-graph` and GPU input normalization matched every
reference RGB byte for all three 241-frame two-step scene tests (MAE/max error
zero, including KV eviction), at 26.36–26.71 generated FPS. The FP8 throughput
table above must not be attributed to this exact-output path.

## Alternative attention backends

The benchmark defaults to `--attention-backend TORCH_SDPA`. Its PyTorch
dispatcher used the built-in FlashAttention kernel on the tested SM120 stack.
Explicit `CUDNN_ATTN` and `FLASH_ATTN` reuse Omni's shared attention backends;
they do not change the model's projection precision or KV policy. Both self
and cross attention must resolve to an explicit supported backend. The
platform selector owns hardware and package checks and rejects unavailable
explicit backends.

For cuDNN, use the cuDNN installation supplied with the PyTorch environment:

```python
diffusion_attention_config={"default": {"backend": "CUDNN_ATTN"}}
```

For FA4 on SM120, install the optional package and select `FLASH_ATTN`. The
CUDA 13 qualification used FA4 `4.0.0b35`, CUTLASS DSL `4.7.1` and cuDNN `9.20`:

```bash
uv pip install 'flash-attn-4[cu13]==4.0.0b35'
```

```python
diffusion_attention_config={"default": {"backend": "FLASH_ATTN"}}
```

Compare the same racing capture, seed, precision and graph settings in separate
processes on the same GPU. Replace the backend flag to test cuDNN or FA4:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/diffusion/benchmark_matrix_game4.py \
  --scene racing --resources ./matrix-resources --input ./racing-geometry.npy \
  --output ./qualification/racing-sdpa --frames 481 --steps 1 \
  --cuda-graph --fp8 --attention-backend TORCH_SDPA

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/diffusion/benchmark_matrix_game4.py \
  --scene racing --resources ./matrix-resources --input ./racing-geometry.npy \
  --output ./qualification/racing-cudnn --frames 481 --steps 1 \
  --cuda-graph --fp8 --attention-backend CUDNN_ATTN \
  --compare ./qualification/racing-sdpa.npy
```

The JSON records the selected backend. This flag configures Omni only; it
cannot switch the separate reference runtime's kernel. Compare generated
outputs as well as timings: different reduction orders propagate through
autoregressive history, so neither alternative is an exact-reference path.
Retain `TORCH_SDPA` when pixel-identical reference reproduction is required.

On the same PRO 5000 and software stack above, three interleaved runs per
backend produced the following results. Each run uses Racing, seed 42, FP8
linears, CUDA Graph, one denoising step and 481 frames; 25 warmup chunks are
excluded, leaving 384 measured frames. Timings use the CPU-uint8-to-CPU-uint8
boundary described above, including the codec and clean-KV update.

| Backend | Mean generated FPS | FPS range across 3 runs | Mean chunk p50 (ms) |
| --- | --- | --- | --- |
| `TORCH_SDPA` | 61.12 | 60.95–61.39 | 65.39 |
| `CUDNN_ATTN` | 62.07 | 61.97–62.26 | 64.41 |
| `FLASH_ATTN` (FA4) | 61.93 | 61.84–62.07 | 64.55 |

cuDNN and FA4 improve mean end-to-end throughput by approximately 1.6% and
1.3%, respectively, in this bounded test. Their ranges overlap; this does not
establish a universal ranking. A separate racing-shaped Attention microbenchmark
showed approximately 6–7% lower full-window kernel latency, which must not be
reported as an end-to-end FPS gain.

Each backend reproduced identical output bytes across its three runs. Relative
to the SDPA run, cuDNN and FA4 RGB MAE was 13.73 and 13.53 on the 0–255 scale;
this measures pixel disagreement, not perceived quality. Matched-frame inspection
showed different textures and scene details. Neither alternative has passed an
exact-reference or perceptual-quality equivalence gate. The default and existing
BF16 qualification remain unchanged. These additional backend measurements cover
Racing at one step, not other scenes, two-step sampling or arbitrary long clips.

## Framework reuse and limits

The DiT reuses Omni's Wan modules, unified Attention, Linear weight loaders and
FFN. UMT5 checkpoint loading is shared with ABot-World. Request scheduling, typed
video media and output conversion use existing engine interfaces. Matrix owns
conditioning, the incremental flow schedule and its sink policy. The TAEW2 decoder is shared with ABot-World; its matching encoder and independent
encoder/decoder states are shared Wan codec components.

Unsupported: multi-GPU TP/SP/PP/CFG/HSDP, generic FP8/FP4 or LoRA, offload/sleep,
TeaCache/Cache-DiT/prefix caching, concurrent sessions, step execution, dynamic
captions and generic streaming-output APIs. Full Wan VAE and other devices are
not qualified. Attention choices are limited to the three explicit backends
above; their SM120 measurements do not qualify other hardware or kernels.

```bash
python -m pytest tests/diffusion/models/matrix_game4 \
  tests/diffusion/models/abot_world/test_offline_check.py \
  tests/diffusion/models/abot_world/test_pipeline_ci.py -q

# CI-like GPU subset: graph lifetime and independent FP8 QKV scales.
python -m pytest tests/diffusion/models/matrix_game4 \
  -m 'core_model and cuda' --run-level core_model -q
```

These regressions require the installed vLLM/Omni environment and a CUDA GPU
with FP8 support, but no downloaded model weights. The existing L2 diffusion
model sweep collects the GPU tests; the CPU sweep collects the session tests.

Timings synchronize each chunk and include clean KV; Omni additionally includes
engine scheduling and output postprocessing. The first 25 chunks are excluded.
Allocator peaks in the JSON are observed after initialization and per request;
internal engine resets can hide cold-start peaks, so use the separate NVML
measurement above when assessing startup capacity.
These measurements do not include game simulation, browser/network delivery or
action-to-photon latency. They do not establish a guaranteed 60 FPS game loop.

Model assets retain their original licenses. The shared TAEW2 codec retains its MIT
notice. Official source revision:
[`5446525c80da2a4a2f87801addbbc7faa8d2486e`](https://github.com/SkyworkAI/Matrix-Game/tree/5446525c80da2a4a2f87801addbbc7faa8d2486e/Matrix-Game-4).
