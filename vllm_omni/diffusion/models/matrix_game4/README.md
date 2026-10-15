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
| Attention | Omni `Attention`, qualified through its `TORCH_SDPA` backend |
| Projection and FFN parameters | Existing vLLM parallel Linear layers and Wan feed-forward blocks |
| Text encoder | Shared Wan checkpoint loader and Transformers `UMT5EncoderModel`; also used by ABot-World |
| Request execution | Existing `DiffusionEngine` and `DiffusionRequestBatch` |
| Video presentation | Typed `DiffusionMediaOutput`; runner-owned float-to-uint8 conversion and transport |
| Matrix-specific behavior | 148-channel conditioning, incremental flow sampling, sink and temporal-cache policy |
| Codec | Shared Wan TAEW2 encoder/decoder; explicit encoder/decoder state per session |

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
