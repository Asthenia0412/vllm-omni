# Matrix-Game 4.0 castle rendering — NVIDIA SM120

## Summary

- Vendor/model: Skywork, `Skywork/Matrix-Game-4.0`, castle checkpoint.
- Task: render game-engine RGB geometry as stylized video.
- Runtime: native `MatrixGame4Pipeline`, experimental Python `DiffusionEngine` API.
- Hardware: one NVIDIA SM120 device reporting `NVIDIA Graphics Device`, 73,415 MiB.
- Profile: BF16 model with native FP8 block linears, one worker and one active session.
- Maintainer: @lishunyang12.

Use this recipe to qualify chunked neural rendering against the released runtime.
Game simulation, keyboard/mouse input, browser presentation and public serving
remain application responsibilities. This integration does not add a game UI,
an HTTP/WebSocket route, or the generic AR-Diffusion session protocol.

## Supported model contract

| Item | Contract |
| --- | --- |
| Task | Castle geometry-to-video; not text-only generation |
| Input | CPU `torch.uint8` RGB tensor `[T, 480, 864, 3]` in `prompt["multi_modal_data"]["video"]` |
| First chunk | One RGB frame, `matrix_frame_start=0` |
| Continuation | Four RGB frames, consecutive offsets `1, 5, 9, ...` |
| Complete clip | Omit the session ID and provide `1+4k` frames; state closes automatically |
| Output | Same frame count and geometry; one batched video in `result.images[0]` |
| Presentation | `output_type="np"`; device postprocessing yields uint8 `[1,T,480,864,3]` |
| Sampling | One or two denoising steps, seed, guidance scale 1 |
| Caption | Fixed at startup; optionally set `model_config={"prompt": "..."}` |
| Budget | Default 8,192 latent chunks, configurable downward with `model_config.max_latents` |
| Session | At most one resident session per worker; callers serialize chunk requests |

The default caption, KV-only sink, rolling sink position, shift-5 scheduler,
initial CPU noise and streaming TAEHV cadence follow the released castle preset.
No runtime import from the Matrix-Game repository is required. Model-owned ring
KV is separate from Omni's generic paged KV and prefix-cache interfaces.

## References

- [Model assets](https://huggingface.co/Skywork/Matrix-Game-4.0/tree/4419accdb7bb7881234c5914de8cd915266e0325).
- [Pinned reference implementation](https://github.com/SkyworkAI/Matrix-Game/tree/5446525c80da2a4a2f87801addbbc7faa8d2486e/Matrix-Game-4).
- [Supported models](../../docs/models/supported_models.md) and
  [diffusion features](../../docs/user_guide/diffusion_features.md).
- [Generic AR-Diffusion contract](../../docs/design/feature/realtime_ar_diffusion.md):
  not implemented by this initial pipeline.
- [Qualification entrypoint](../../benchmarks/diffusion/benchmark_matrix_game4.py).
  Existing shared text/image video examples do not supply this chunk-input contract.

## Hardware and software

| Hardware | Exercised profile |
| --- | --- |
| Accelerator | NVIDIA Graphics Device, SM120, 110 SMs, 73,415 MiB |
| Count/interconnect | One GPU, PCIe; no collective communication |
| Host staging | T5 checkpoint loads through host memory; no inference offload |
| Scope | Bounded castle qualification; other device SKUs are not qualified |

The upstream preset is named `rtx5090`. That name does not identify the hardware
used here, and these results are not RTX 5090 measurements.

| Software | Version |
| --- | --- |
| OS | Ubuntu 22.04.4 LTS, Linux 5.15.0-1069-nvidia |
| Python | 3.12.13 |
| NVIDIA driver / CUDA runtime | 580.95.05 / 13.0 |
| PyTorch / vLLM | 2.13.0 / 0.31.0 |
| Transformers / Diffusers | 5.14.1 / 0.40.0 |
| vLLM-Omni | This change on base `8201f614b` |

## Setup

Install this branch with its CUDA dependencies. Download only the castle
checkpoint, sink, tokenizer, text encoder and codec:

```bash
hf download Skywork/Matrix-Game-4.0 \
  --revision 4419accdb7bb7881234c5914de8cd915266e0325 \
  --include 'models/castle/model.safetensors' 'models/castle/sink.png' \
    'base/models_t5_umt5-xxl-enc-bf16.pth' 'base/google/umt5-xxl/*' 'base/taew2_2.pth' \
  --local-dir ./matrix-resources
```

Capture geometry frames from the game engine and save a uint8 RGB THWC NumPy
array as `geometry.npy`. A 241-frame qualification requires at least 241 frames
at exactly 480 by 864 pixels. Generated neural frames are not geometry controls.

## Commands

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/diffusion/benchmark_matrix_game4.py \
  --resources ./matrix-resources --input ./geometry.npy \
  --output ./qualification/omni --frames 241 --steps 1
```

This constructs `OmniDiffusionConfig` with `model_class_name="MatrixGame4Pipeline"`,
`num_gpus=1`, `max_num_seqs=1`, `distributed_executor_backend="uni"` and
`enforce_eager=True`. The latter disables the generic graph wrapper: Matrix owns
the DiT and codec graphs. `VideoOutputTransportConfig(enable_device_postprocess=True)`
lets the runner convert normalized device output to RGB bytes before download.

The entrypoint uses `DiffusionEngine.step_streaming()` for consecutive ordinary
requests. Each request carries these sampling `extra_args`:

| Key | Value |
| --- | --- |
| `matrix_session_id` | Nonempty string, at most 128 characters |
| `matrix_frame_start` | Expected consecutive frame offset |
| `matrix_end_session` | `True` on the final chunk |

To close without rendering, send the owning session ID with
`matrix_end_session=True` and omit video. Applications must close in a `finally`
block; abandoned sessions have no idle timeout in this initial API. A conflicting
session or incorrect offset is rejected without taking over the existing owner.
An inference failure clears the session. After closure, a fresh session begins
at frame zero. End-of-session cleanup includes a device synchronization.

## Reference comparison

The optional oracle requires the pinned upstream checkout only in the reference
process. Run both modes on the same device and Python environment:

```bash
git clone https://github.com/SkyworkAI/Matrix-Game.git matrix-reference
git -C matrix-reference checkout 5446525c80da2a4a2f87801addbbc7faa8d2486e
PYTHONPATH=./matrix-reference/Matrix-Game-4 \
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/diffusion/benchmark_matrix_game4.py \
  --mode reference --resources ./matrix-resources --input ./geometry.npy \
  --output ./qualification/reference --frames 241 --steps 1 \
  --compare ./qualification/omni.npy
```

## Supported features

| Feature | Status |
| --- | --- |
| Native FP8, CUDA graphs, ring KV, text KV reuse, streaming TAEHV | Included |
| Single GPU / single active session | Qualified profile |
| [TP/SP/PP/CFG/HSDP](../../docs/user_guide/diffusion/parallelism/overview.md) | Unsupported |
| [Generic quantization](../../docs/user_guide/quantization/overview.md), LoRA | Unsupported; native FP8 is fixed |
| [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md), sleep | Unsupported |
| TeaCache / Cache-DiT / prefix caching | Unsupported |
| Step execution / concurrent batching / generic streaming output | Unsupported; use serial chunk requests |
| Other scenes, dynamic captions, public session routing | Not implemented |
| Other accelerators, frame interpolation | Not qualified |

## Verification

```bash
python -m pytest tests/diffusion/models/matrix_game4/test_session.py -q
```

The qualification command writes `.npy` RGB output and `.json` timing/software
metadata. `--compare` additionally records pixel MAE, maximum error and exact
equality. Timings synchronize each chunk and include the clean KV pass; Omni
timings also include engine scheduling and output postprocessing. The first
25 chunks are excluded from steady timing. Browser/network latency, game-loop
timing and action-to-photon latency are not measured by this command.

### Initial qualification

One 241-frame castle capture, seed 42, one denoising step, concurrency one,
and 25 excluded warmup chunks produced the following on the environment above:

| Path | Steady FPS | Four-frame p50 / p95 |
| --- | --- | --- |
| Native pipeline through Omni | 59.81 | 66.79 / 69.61 ms |
| Pinned standalone runtime | 63.04 | 63.08 / 64.68 ms |

These are bounded single-run measurements, not a speedup claim. The through-engine
output is not yet qualified for reference parity: pixel MAE was 23.00 on the
0–255 RGB scale across the 241 frames. Direct pipeline execution matched the first
five reference frames exactly, localizing further investigation to engine
integration. Human visual review, longer sessions, two-step parity and peak memory
qualification remain outstanding. Do not use these timings to imply equivalent
image quality or a guaranteed 60 FPS browser experience.

## Source provenance

Native network, fused kernels, graph execution and codec streaming are adapted
from the pinned Matrix-Game-4 source. The tiny codec retains the MIT notice for
Ollin Boer Bohan's TAEHV, upstream revision
`011dfc2112197741c540e0bdd5b7b67bcc930771`. Unused H3 and larger-codec architectures
are excluded. Text encoding and scheduler arithmetic retain released checkpoint
semantics; this is not a Diffusers pipeline adapter.
