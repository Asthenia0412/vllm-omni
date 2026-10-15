# Matrix-Game 4.0 in vLLM-Omni

A native geometry-to-video integration for **one CUDA GPU** at 864×480. It has
no SM120-only model layers or architecture gate. The current qualification
device is one **RTX PRO 5000 72GB Blackwell**; other GPUs, including the RTX 5090
deployment target, require their own capacity, quality and speed measurements.

This is an experimental Python chunk API. It is not a released HTTP game server.
The game engine supplies low-poly RGB frames; no GPT-6 or other coding-agent API
is required while playing. A coding agent can help author the external game.

See the [deployment recipe](../../../../recipes/Skywork/Matrix-Game-4.0.md)
for installation, pinned asset downloads, scene selection, request configuration,
and reference comparisons. No Matrix-Game source checkout or package is needed
for Omni inference; the separate reference command uses it only as a test oracle.

## Framework reuse

| Component | Implementation |
| --- | --- |
| DiT weights and modules | Existing `WanTransformer3DModel`, with strict native-name mapping |
| Attention | Omni `Attention`; `TORCH_SDPA` reference path, opt-in `CUDNN_ATTN` / `FLASH_ATTN` |
| Projection and FFN parameters | Existing vLLM parallel Linear layers and Wan feed-forward blocks |
| Text encoder | Shared Wan checkpoint loader and Transformers `UMT5EncoderModel`; also used by ABot-World |
| Request execution | Existing `DiffusionEngine` and `DiffusionRequestBatch` |
| Video presentation | Typed `DiffusionMediaOutput`; runner-owned float-to-uint8 conversion and transport |
| Matrix-specific behavior | 148-channel conditioning, incremental flow sampling, sink and temporal-cache policy |
| Codec | Shared Wan TAEW2 encoder/decoder; explicit encoder/decoder state per session |
| Graph replay | Platform-selected vLLM graph wrapper, with session-owned stable inputs and codec history |
| Experimental FP8 | Existing `quantization_config="fp8"`, online weight loading and vLLM kernel selection |

Reusing a component does not automatically qualify every backend or optimization
it offers. Multi-GPU execution, paged KV and concurrent sessions are not enabled.

FP8 uses the existing `DiffusionFp8Config` and vLLM online linear method through
the standard `quantization_config` entry point. vLLM owns weight conversion,
activation scaling and kernel selection. No Matrix-specific quantizer, scale
floor, partitioned linear method or GPU architecture check is added. The packed
QKV projection follows the standard backend's scaling and GEMM policy. Optional
`ignored_layers` names use Wan runtime prefixes such as `blocks.0.attn1.to_qkv`.
Only the plain online FP8 config is qualified here; serialized checkpoints,
per-component maps and other methods are rejected before loading.

The remaining model-specific normalization, RoPE and session code preserves the
released checkpoint's rounding and causal state. Existing per-head fused RoPE
and generic diffusion caching have different semantics and are not substitutes.

## Quality contract

Reference quality is a gate, not a best-effort target. Compare the same checkpoint,
geometry frames, caption, seed, timestep grid, codec and precision. BF16 reference
comparisons must pass before enabling lossy quantization or a different attention
backend. Check both the first chunk and generation beyond the 21-latent window.

The correctness path keeps three separate Q/K/V GEMMs, the original BF16 RMSNorm
rounding, the shared Conv3d convolution path, unfused adaptive normalization and
chronological KV order. Packed QKV GEMM and convolution-to-GEMM produced measured
numerical differences on SM120; those optimizations cannot be presumed lossless.

A 241-frame castle comparison through the actual Omni engine matched all decoded
RGB bytes exactly at one step with TAEHV. This does not establish parity for a
full Wan VAE, other attention backends, quantization, hardware or arbitrary long
sessions. All six 121-frame scene/step pairs and three 1,201-frame two-step gameplay clips
also matched exactly. See the recipe for the complete qualification matrix.

## Acceleration switches

Graph replay defaults to `False` and `quantization_config` defaults to `None`.
Keep `enforce_eager=True` for the outer engine; the model controls capture of its
stateful, fixed-shape operations.

```python
model_config={"scene": "castle", "cuda_graph": True},
quantization_config="fp8",
```

`cuda_graph` captures the steady DiT window and continued TAEW2 chunks using the
existing platform graph wrapper. Online FP8 quantizes supported block linears;
this acceleration path also compiles normalization/residual chains with PyTorch
and uses a ring history buffer. Embeddings, output head and codec remain BF16.
`model_config.fp8` is rejected; quantization has one configuration entry point.

The FP8 path changes scaling and GEMM policy compared with the former custom
adapter. It is **not pixel-identical** to BF16 or the reference optimized FP8
runtime. Retain the default BF16 path for exact reference reproduction. The
recipe distinguishes current validation from historical throughput results.
A bounded 481-frame Racing run with standard online FP8 measured 76.37 generated
FPS on one PRO 5000 at 864×480, one step, graph replay and SDPA. This is not a
visual-quality parity claim or an RTX 5090 measurement.

## Experimental 1280×736 rendering

The default remains 864×480. Configure a new worker with
`model_config={"scene": "racing", "height": 736, "width": 1280}` and supply
CPU uint8 geometry frames of shape `[T,736,1280,3]`. Request dimensions must
match the worker (or be omitted). The benchmark accepts `--height 736 --width 1280`;
precision, step count and graph replay remain independent options.

This runs the released checkpoint at a larger latent grid; it does not upscale
the generated 480p output. It is an experimental resolution extrapolation, with
no reference-quality qualification at that size. The codec and patch embedding
require multiples of 32; 1280×720 is rejected. Resolution cannot change within a
worker. Multi-GPU execution remains unsupported.

Historical runs with the removed custom FP8 path on the PRO 5000 measured 27.52 FPS
at 1280×736 with FP8 / one step, 11.24 FPS with BF16 / two steps, and 61.38 FPS
at 864×480 with FP8 / one step. All use TAEW2, SDPA and graph replay. These are
generated FPS after warmup, not video playback FPS or interaction latency.
These historical FP8 numbers do not describe the current standard online FP8 path.
The recipe documents current measurements and their limits.

## Attention selection

Select the existing framework backend explicitly for both self and cross
attention, or use the shared `per_role` configuration for separate choices:

```python
diffusion_attention_config={"default": {"backend": "CUDNN_ATTN"}}
```

`TORCH_SDPA` remains the exact-reference configuration and benchmark default.
`CUDNN_ATTN` and `FLASH_ATTN` are experimental alternatives; they reuse the
framework implementations without model-specific kernels or fallback logic.
On SM120, `FLASH_ATTN` selects FA4 and requires the optional `flash-attn-4`
package. Missing or incompatible kernels are rejected by the platform selector.
FP8 linears, graph replay and temporal/text KV reuse remain independent options.

Changing the attention kernel changes floating-point reductions and can change
autoregressive output. The alternative backends are not pixel-identical to the
reference. See the recipe for paired end-to-end timings and installation.
