# Matrix-Game 4.0 in vLLM-Omni

A native geometry-to-video integration targeting **one RTX 5090 (32 GB)** at
864×480. The current qualification device is one **RTX PRO 5000 72GB Blackwell**;
RTX 5090 capacity and speed have not been measured for this implementation.

This is an experimental Python chunk API. It is not a released HTTP game server.
The game engine supplies low-poly RGB frames; no GPT-6 or other coding-agent API
is required while playing. A coding agent can help author the external game.

See the [deployment recipe](../../../../recipes/Skywork/Matrix-Game-4.0-SM120.md)
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
| Experimental FP8 | vLLM per-tensor quantization/scaled GEMM; shared QKV input quantization |

Reusing a component does not automatically qualify every backend or optimization
it offers. Multi-GPU execution, generic FP8, paged KV and concurrent sessions are
not enabled by this adapter.

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

Both switches default to `False`. Keep `enforce_eager=True` for the outer engine;
the model controls capture of its stateful, fixed-shape operations.

```python
model_config={"scene": "castle", "cuda_graph": True, "fp8": True}
```

`cuda_graph` captures the steady DiT window and continued TAEW2 chunks using the
existing platform graph wrapper. `fp8` additionally converts block projections
to per-tensor E4M3, compiles normalization/residual chains with PyTorch, and uses
a ring history buffer. Embeddings, the output head and codec remain BF16. These
switches reuse framework components and do not import the reference runtime.

The FP8 path is experimental and **not pixel-identical** to the reference FP8
runtime. Small fused-reduction differences can lead to different long-sequence
effects. Throughput alignment does not qualify its visual quality; retain the
default BF16 path where exact reference reproduction is required. See the
recipe for paired FPS measurements, reproducible commands and their scope.

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
