# Distributed Layerwise Offload

This document describes distributed layerwise offload (DLO) for diffusion
models. DLO keeps only a small number of DiT blocks on the accelerator and
streams the remaining blocks from host memory. The distributed backend can
either shard those host-side weights across an existing parallel group or keep
complete rank-local block sources and avoid an additional collective.

For user-facing commands, see the
[distributed layerwise offloading guide](../../../user_guide/diffusion/offloader/distributed_layerwise_offload.md)
and the [Cosmos3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-DistOffload.md).

## Feature compatibility

Host-storage optimization and runtime compatibility are separate decisions.
When direct checkpoint mmap is unavailable, DLO can still use tensors produced
by the ordinary loader. "Compatibility path" below means that fallback is
implemented but has less end-to-end coverage than the primary path.

Legend: ✅ supported, ⚠️ compatibility path or limited validation, ❌ unsupported.

| Feature | DLO + AllGather | DLO without AllGather |
|---|---|---|
| **DP** | ✅ Primary path; host weights are sharded across the DP group. | ✅ Each DP rank streams complete rank-local blocks. |
| **SP** | ✅ When DP=1, DLO uses the SP group for weight sharding. | ✅ SP remains active without a DLO weight collective. |
| **TP > 1** | ⚠️ Ordinary TP-aware loader only; no direct checkpoint mmap. | ⚠️ Ordinary TP-aware loader only; no direct checkpoint mmap. |
| **HSDP** | ❌ Rejected to avoid double-sharding parameters. | ⚠️ Limited end-to-end coverage. |
| **Per-tensor online FP8 linears** | ✅ Ordinary loader finalizes weights and scales before DLO sharding. | ✅ Ordinary loader retains complete rank-local tensors. |
| **Other online quantization methods** | ❌ Rejected until runtime packing and scale layouts are validated. | ⚠️ Allowed through the ordinary loader; validation is method-specific. |
| **Model-level or standard layerwise CPU offload** | ❌ Disabled because DLO takes priority. | ❌ Disabled because DLO takes priority. |
| **Resident leading layers** | ❌ Rejected. | ✅ Requires eligible resident paths in the model's `OffloadPlan`. |

See [Parallelism compatibility](#parallelism-compatibility) and
[Request and loading constraints](#request-and-loading-constraints) for the
detailed contracts and validation boundaries.

## Status

DLO is implemented for multi-device diffusion execution. The default
AllGather path is the primary path for DP and SP deployments. The
`--dlo-no-use-allgather` path streams complete blocks independently and adds no
DLO weight collective.

Host storage is selected separately from the transfer protocol. The loader can
produce a direct-checkpoint mmap plan for a proven-compatible runtime layout;
otherwise it uses the ordinary loader. In no-AllGather mode, an opt-in Phase B
runtime cache can normalize those final ordinary-loader DiT tensors into an
immutable node-local mmap entry. Consequently, replicas can share either
proven-compatible checkpoint pages or equivalent final runtime layouts. An
opt-in CUDA registration budget can make the complete final mapping directly
H2D-copyable, avoiding the recurrent host packing otherwise required by mmap.

The Phase A shared-mmap support boundary is TP1. TP greater than one is an
ordinary-loader compatibility path: DLO can consume the resulting TP-local
tensors. Direct checkpoint mmap still does not cover TP, but the Phase B cache
can share each final TP coordinate among equivalent processes.

The compatibility matrix below describes the current implementation. The
unit-level guards are covered, but not every parallelism combination has a
full model-and-hardware end-to-end test.

## Design

### DLO consumes the existing parallel topology

DLO does not create a new DP, TP, or SP topology. It reads the configured
`DiffusionParallelConfig` and attaches offload hooks to the DiT blocks after the
standard distributed groups have been initialized.

The DLO weight-sharding group is selected as follows:

1. Use the existing DP group when `data_parallel_size > 1`.
2. When DP is one and SP is greater than one, use the SP group.
3. Otherwise, run rank-locally without a DLO process group.

TP is deliberately not used as DLO's AllGather group. HSDP has its own
parameter-sharding lifecycle and is not allowed to be sharded a second time by
DLO's AllGather path.

### The loader owns host-weight planning

Before it decides whether ordinary weight materialization can be skipped, the
diffusion loader builds one `HostWeightPlan`. A direct-checkpoint mmap plan is
accepted only when preflight proves all of the following:

- every required DiT parameter and persistent buffer has exactly one source;
- runtime names, checkpoint keys, shapes, and dtypes match;
- the runtime topology is TP1 without HSDP or online quantization; and
- every custom loader operation is represented by a loader-owned checkpoint
  adapter.

The exact plan object is handed to DLO. The backend does not rescan checkpoint
files, repeat the capability decision, or reconstruct names from its block
topology. If preflight fails, the loader materializes weights normally and DLO
consumes those runtime tensors.

The plan owns only dedicated DiT component sources. If a pipeline also exposes
ordinary sources for a text encoder or another non-DiT component, the loader
still consumes those sources and includes their loaded names in its strict
coverage check. Only the source prefixes covered by the plan skip ordinary
materialization. A source that mixes DiT and non-DiT weights fails closed to
the complete ordinary-loader path because it cannot be skipped safely as a
unit.

This boundary keeps checkpoint semantics out of DLO and avoids model-pipeline
flags such as `_supports_mmap_loading` or parameter attributes for mmap-only
transforms. Model-specific direct-layout knowledge, when required, lives in a
checkpoint adapter beside the ordinary loader.

### Post-load runtime cache

The Phase B cache preserves the same ownership boundary. DLO neither derives
cache keys nor interprets model loaders. When enabled with no-AllGather, every
rank first completes the ordinary loader, custom callbacks, post-load
processing, strict validation, and calibration. The loader then:

1. discovers final CPU DiT parameters and persistent buffers;
2. rejects unsupported non-contiguous, aliased, quantized, distributed, or
   ambiguous layouts before changing the model;
3. hashes tensor names, metadata, and final bytes;
4. builds or joins one content-addressed entry under a strict POSIX lock; and
5. returns a `HostWeightPlan(backing_kind="runtime_cache")` only after file,
   manifest, and remapped-tensor content validation succeeds.

The identity excludes DP size/rank and SP rank. It includes TP world size/rank
and conservative SP implementation/world-size guards. DP replicas and
equivalent SP ranks can therefore share; different TP coordinates cannot.
Final tensor content is authoritative, so an unmodeled loader/config change
that changes bytes creates a different entry instead of risking silent
sharing.

Publication writes sharded safetensors and a JSON manifest into a same-
filesystem temporary directory, flushes them, and atomically renames the
directory. The manifest includes per-file SHA-256 digests and the final tensor
content digest. Kernel `flock` ownership distinguishes a live writer from a
stale lock file: process death releases the lock automatically. Stale temporary
entries are removed only by the next lock owner. A bounded wait or any cache
error retains the already-valid ordinary tensors.

DLO maps this plan without moving the DiT to `meta` and without rerunning any
post-load hook. It changes tensor storage while preserving `Parameter` objects,
then runs the existing `gc.collect()`/`malloc_trim()` cleanup before installing
streaming hooks. The source mappings remain open through DLO disable. With the
default zero registration budget they feed the existing bounded two-slot host
staging path. Every streamed block is first packed from pageable mmap storage
into one of those pinned slots and is then copied to the device. This removes
model-sized private pinned storage, but adds a recurrent host-to-host copy and
can amplify TP synchronization waits when several ranks compete for host
bandwidth.

This version is deliberately opt-in and steady-state-oriented. It requires
roughly one local-disk model copy per runtime identity, has no eviction, and
still performs ordinary loading plus full content-hash passes in every rank.
Skip-load-on-hit is a separate follow-up after loader side effects can be
modeled safely.

### Registered runtime-cache transfer

`dlo_runtime_cache_pin_limit_gib > 0` opts a CUDA worker into complete
runtime-cache registration. It also changes loader selection: no-AllGather
uses the ordinary loader and final runtime cache even when a TP1 direct-
checkpoint plan is available. A checkpoint binding can carry a deferred
adapter such as grouped-QKV reordering. Registering that raw source would not
remove the adapter's recurrent CPU allocation/copy, whereas the runtime cache
contains the transform-complete final tensors.

After mapping and validating every final tensor, DLO groups source storages by
backing file, page-aligns their address ranges, and coalesces overlapping or
adjacent ranges within that mapping. The complete byte count is checked
against the per-worker budget before any CUDA call. It then registers every
range with the read-only CUDA host-registration flag and verifies that PyTorch
recognizes every mapped tensor as pinned. Failure rolls back any ranges already
registered and preserves the existing two-slot staging path. Partial direct
transfer is deliberately unsupported. A rollback failure aborts worker
initialization so a still-registered range is retained until process/context
teardown rather than being unsafely unmapped.

On success, the hooks copy each immutable tensor view directly into its offset
in the existing flattened rotating device buffer on the copy stream. Parameter
repointing, two-block HBM residency, and H2D/compute overlap are unchanged. The
implementation currently issues more H2D copy operations than block packing,
but avoids the much larger recurrent CPU copy. A packed cache schema should be
considered only if launch-count measurements justify its added format and
publication complexity.

Registration is per process and CUDA context; physical file-cache pages remain
shared and are not copied into one private host allocation per worker. The
operator must nevertheless budget page-locked memory and satisfy OS/CUDA
registration limits for every worker. During shutdown, DLO first synchronizes
outstanding device work, unregisters all ranges, and then closes the safetensors
mappings. Worker teardown invokes this cleanup before destroying the
distributed environment and CUDA context.

### AllGather path

With the default `dlo_use_allgather=True`, each rank stores approximately
`1 / group_size` of each streamable block in pinned host memory. The next
block's shard is copied to a device buffer and reconstructed with
`all_gather_into_tensor` on a communication stream while the current block is
executing.

```text
Compute:    [Block N]             [Block N+1]          [Block N+2]
H2D:                      [shard N+1]           [shard N+2]
AllGather:                [full N+1]             [full N+2]
Buffers:    [current slot]       [prefetch slot]       [current slot]
```

![DLO double-buffer prefetch pipeline](../../figures/dlo/dlo_pipeline.gif)

The backend uses two shared device buffers, so accelerator weight residency is
bounded by the largest streamed blocks rather than the complete model.

When direct checkpoint mmap is selected, the checkpoint mappings are only the
source used to prepare each rank's persistent shard. They can be closed after
shard preparation. Across the AllGather group, those private shards total
approximately one runtime model copy.

An effective DLO group size of one performs no collective, even when
`dlo_use_allgather=True`; it follows the rank-local transfer path described
below.

When DP is greater than one, the engine can process one request per DP rank in
the same denoising wave. Because AllGather is a collective, all participating
requests must take the same execution path at every denoising step.

Fast NVLink/NVSwitch changes collective cost, not this execution contract. For
independently scheduled replicas—even on one NVSwitch baseboard—no-AllGather
plus shared mmap storage avoids coupling their request schedules and failure
domains.

### Rank-local path without DLO AllGather

With `--dlo-no-use-allgather`, DLO forces its internal offload shard size to
one and streams complete blocks using H2D copies only. The host backing may be
a loader-approved checkpoint mapping, a normalized runtime-cache mapping, or
ordinary runtime tensors.

For direct mmap, each process retains immutable safetensors views and uses two
bounded pinned host staging slots. Processes on the same node that map the same
files share physical checkpoint pages through the OS page cache. This removes
the persistent private full-model copy per pure-DP process, but each process
still packs and transfers every complete block. Sharing is node-local; each
node has its own page cache.

For a runtime-cache mapping, a positive registration budget replaces those
host slots with direct copies from the shared final tensor views. A zero budget
or failed registration retains the bounded staging behavior.

When direct mmap preflight fails, the regular model loader remains responsible
for preparing each rank's weights, including TP-local tensors or HSDP-managed
parameters. Supported CPU layouts may then enter the opt-in runtime cache.
Unsupported layouts keep one private runtime copy per process.

This mode means:

- DP still provides independent replicas, but DLO does not shard weights
  across DP ranks.
- SP still performs its normal activation/attention collectives, but DLO does
  not shard weights across SP ranks.
- TP/HSDP/SP collectives, if configured, are not disabled by this flag; only
  DLO's additional weight AllGather is disabled.
- Pure DP deployments share one checkpoint-backed copy per node when direct
  mmap is selected. The runtime cache can share equivalent ordinary-loader
  outputs; its fallback keeps one private runtime copy per rank.
- The scheduler does not require a synchronized DP request wave for DLO.

## Parallelism compatibility

| Parallelism | DLO + AllGather | DLO without AllGather |
|---|---|---|
| **DP** | Supported primary path. DLO shards host weights across the DP group and can run DP multi-concurrency. | Supported rank-local path. Compatible TP1 replicas can share checkpoint pages; the optional runtime cache shares equivalent final layouts while excluding DP rank from its identity. CUDA workers may register the complete final mapping for direct H2D. |
| **SP** | Supported in the implementation. With DP=1, DLO uses the SP group for host-weight sharding; SP still shards sequence/activation work. | SP remains active without a DLO weight collective. Runtime-cache v1 excludes SP rank and includes conservative SP implementation/world-size guards. Registered transfer does not remove ordinary SP activation/attention collectives. |
| **TP > 1** | Outside the Phase A direct-mmap scope. The loader preserves TP-local layouts and DLO may apply DP/SP host sharding to ordinary runtime tensors. | The ordinary TP-aware loader produces rank-local tensors. Runtime-cache v1 creates one entry per TP coordinate, shareable by equivalent DP/SP processes. Registering those final mappings removes staging-driven TP skew but does not remove ordinary TP collectives. |
| **HSDP** | Rejected. HSDP has already sharded parameters, so DLO AllGather would double-shard them. | Accepted by configuration. HSDP owns parameter sharding and its own gathers; DLO only stages rank-local parameters. End-to-end coverage is limited. |

### Combined dimensions

- **DP + SP:** DLO uses the DP group for weight sharding when DP is greater
  than one; SP continues to use its own sequence-parallel group. If DP is one,
  the SP group becomes DLO's sharding group in AllGather mode.
- **DP + TP/SP without AllGather:** standard model loading defines the
  rank-local tensor layout. DLO adds no cross-DP, cross-TP, or cross-SP weight
  collective.
- **HSDP + SP:** the general parallel configuration permits HSDP over SP, but
  DLO must use `--dlo-no-use-allgather`. HSDP remains responsible for weight
  materialization and synchronization.
- **HSDP + DP or TP:** rejected independently by the diffusion parallel
  configuration.

## Request and loading constraints

AllGather DP multi-concurrency requires:

- explicit `num_inference_steps`;
- the same `num_inference_steps` for all requests in a wave; and
- identical request arguments that affect the collective execution path.

The no-AllGather path does not impose these DLO-specific synchronized-wave
requirements.

Direct checkpoint mmap can back either transfer path. It is currently limited
to proven TP1, non-HSDP, non-online-quantized layouts. Other layouts use the
ordinary loader. Per-tensor online FP8 linears can use DLO AllGather after the
ordinary loader finalizes their runtime weights and scales; DLO then shards and
reconstructs those tensors with their recorded layouts. Other online methods
must use `--dlo-no-use-allgather` or disable online quantization until their
runtime layouts are validated.

Runtime-cache v1 is no-AllGather only and rejects all quantization
configurations, HSDP/DTensor, expert parallelism, CFG parallelism, PP,
non-contiguous tensors, and
tied/shared storage.

## Validation coverage

Current source-level validation includes:

- HSDP + DLO + AllGather rejection;
- HSDP + DLO without AllGather acceptance at configuration level;
- loader preflight fallback for TP, HSDP, online quantization, unknown custom
  loaders, missing keys, and shape/dtype mismatches;
- ordinary-loader fallback for per-tensor online FP8 linears followed by DLO
  sharding of finalized weights and scales;
- exact loader-to-backend plan transfer and ordinary-loader fallback;
- rank-local mmap source retention, bounded two-slot staging, and adapter
  transforms without parameter-side flags;
- runtime-cache content identity, sharded atomic publication, concurrent
  writer election, corruption rebuild, lock timeout, and fail-closed layout
  rejection;
- loader ordering through final post-load mutation and DLO remapping without
  rerunning post-load hooks;
- all-or-nothing page-aligned registration, budget rejection, partial-failure
  rollback, direct-copy staging bypass, and explicit shutdown cleanup;
- resident-layer requests requiring no-AllGather;
- DP request-wave validation for denoising-step compatibility;
- sharding, double-buffer, AllGather-size, and heterogeneous-block regression
  tests.

### B300 parallel-topology smoke matrix

A four-GPU B300 smoke test covered MiniMax-H3 FL2VA with the same prompt, seed,
CUDNN attention backend, 256x256 output, two denoising steps, and
`dlo_resident_layers=0`. The TP2 rows used DiT DP2xTP2 with the text encoder and
VAEs at TP1. They validate the ordinary-loader fallback only, not direct mmap
or shared-mmap host-memory savings.

| Configuration | Result | Warm E2E | Peak device memory | Host PSS |
|---|---:|---:|---:|---:|
| DP4xTP1 AllGather | Passed, 4 concurrent requests | 2.87 s / 4 requests | 13.84 GiB | 211.99 GiB |
| DP4xTP1 no-AllGather | Passed, 1 request | 15.02 s | 13.23 GiB | 187.77 GiB |
| DP2xTP2 AllGather | Passed, 2 concurrent requests | 4.16 s / 2 requests | 12.50 GiB | 211.97 GiB |
| DP2xTP2 no-AllGather | Passed, 1 request | 3.51 s | 11.88 GiB | 314.01 GiB |

Within each topology, the AllGather and no-AllGather video and audio outputs
were byte-identical. All four runs completed without an `ERROR` or traceback
and released their device allocations. For DP4xTP1, no-AllGather direct mmap
reduced total PSS by 24.22 GiB (11.4%) and `Private_Dirty` from 211.33 to
125.32 GiB (40.7%) relative to AllGather. For DP2xTP2, preflight selected the
ordinary loader as designed; no-AllGather PSS was 314.01 GiB, about 48% above
AllGather, because DP replicas did not share checkpoint-backed runtime
weights. This is a functional and memory smoke test, not a production-quality
performance or output-quality benchmark.

### Host-memory measurement

A two-worker MiniMax-H3 FL2VA measurement on one L20X node compared the
ordinary-loader fallback with direct mmap. Both runs used
DP=2, TP=1, no DLO AllGather, BF16 weights, two denoising steps, and a
256x256 four-second request. The ordinary-loader workers were sampled after
initialization. The mmap workers were sampled after one completed request, so
the checkpoint working set had been faulted into the page cache; this is the
more conservative point for mmap.

The values below come from `/proc/<worker>/smaps_rollup` and include the whole
worker, not only the DiT. The stable rank-to-rank difference comes from other
pipeline components, so each worker should be compared with the same worker in
the other storage mode.

| Worker | Ordinary RSS | mmap RSS | Ordinary PSS | mmap PSS | PSS reduction |
|---|---:|---:|---:|---:|---:|
| DP worker 0 | 168.27 GiB | 132.76 GiB | 167.84 GiB | 101.43 GiB | 66.40 GiB |
| DP worker 1 | 116.19 GiB | 79.97 GiB | 115.73 GiB | 48.64 GiB | 67.09 GiB |
| **Two-worker total** | — | — | **283.56 GiB** | **150.08 GiB** | **133.48 GiB (47.1%)** |

The direct-mmap workers each reported 62.45 GiB `Shared_Clean` but only
31.20 GiB `Pss_File`, which is the proportional charge expected when the same
resident checkpoint pages are mapped by two workers. `Private_Dirty` also fell
from 167.53 to 70.24 GiB for worker 0 and from 115.40 to 17.44 GiB for worker
1, a reduction of about 97–98 GiB per worker. RSS understates this benefit
because it counts a shared physical page in every process that maps it; summed
PSS is the appropriate node-memory comparison.

### L20X runtime-cache smoke matrix

Phase B was also exercised on four L20X GPUs in one NVLink domain with
MiniMax-H3 FL2VA, CUDNN attention, eager execution, 256x256 four-second
outputs, two denoising steps, `dlo_resident_layers=0`, text-encoder TP1, and
VAE TP1. This is a functional, memory, and profiler smoke test rather than a
production-quality performance benchmark. Host values below are node-wide
maxima during request waves from descendant-process `smaps_rollup`; they do
not represent the private startup peak. HBM is the engine-reported request
peak because sampled `nvidia-smi` data can miss short-lived allocations.

One internal DP2xTP2 engine compared the three host-storage/transfer choices.
Each measured wave contained two requests. NVLink TX is the physical per-link
counter delta summed across the four GPUs; RX had the same value and is not
added again here.

| Mode | Two-request wave | Throughput | Host PSS | `Private_Dirty` | Engine peak HBM | NVLink TX / wave |
|---|---:|---:|---:|---:|---:|---:|
| AllGather, ordinary loader | 3.41 s | 0.587 req/s | 210.70 GiB | 209.72 GiB | 12,498 MB | 73.37 GiB |
| no-AllGather, private loader weights | 6.96 s | 0.287 req/s | 312.56 GiB | 311.56 GiB | 11,882 MB | 19.78 GiB |
| no-AllGather, runtime cache (cold publish) | 18.94 s | 0.106 req/s | 178.71 GiB | 116.00 GiB | 11,882 MB | 19.78 GiB |
| no-AllGather, runtime cache (warm reuse) | 24.23 s | 0.083 req/s | 178.29 GiB | 115.58 GiB | 11,882 MB | 19.78 GiB |

The no-AllGather cache preserved the no-AllGather NVLink traffic and HBM
footprint while reducing node PSS by about 134 GiB relative to private loader
weights. Cold and warm labels describe cache publication or reuse during
initialization; both request phases ran after initialization. A warm hit still
performed ordinary loading and full content hashing, so it was not a startup
fast path. All measured no-AllGather private/cache outputs were byte-identical
for matching seeds.

The deployment shape targeted by the cache was tested separately as two
independent DP1xTP2 engines on the same node. Both engines mapped the same two
TP-coordinate entries, including the same file device/inode tuples; the cache
occupied 66,281,995,903 bytes. A start barrier kept all four measured requests
ahead of profiling.

| Storage | Mean request latency, engines A / B | Combined throughput | Host PSS | `Private_Dirty` | Engine peak HBM |
|---|---:|---:|---:|---:|---:|
| Private loader weights | 3.50 / 3.60 s | 0.463 req/s | 366.20 GiB | 365.22 GiB | 11,882 MB |
| Shared runtime cache | 8.96 / 12.30 s | 0.142 req/s | 232.23 GiB | 169.52 GiB | 11,882 MB |

The shared cache reduced request-time PSS by 133.97 GiB (36.6%) and
`Private_Dirty` by 195.70 GiB (53.6%), but reduced combined throughput by
69.3% in this host-contention smoke. Across one profiled request per engine,
aggregate GPU compute was unchanged (1.359 versus 1.357 seconds) and H2D
payload was exactly 264.24 GiB in both modes. Aggregate CPU `aten::copy_` time
rose from 3.75 to 32.49 seconds, while communication-kernel time rose from
3.56 to 5.80 seconds as TP ranks waited on skewed staging. These durations are
summed across ranks and can overlap; they identify the mechanism, not wall
time. Matching private/cache outputs were byte-identical.

The result confirms that zero-budget runtime-cache staging is a host-capacity
option, not a latency-neutral replacement for private pinned weights. Broader
SP, model, platform, and production-quality validation remains tracked in
[issue #6231](https://github.com/vllm-project/vllm-omni/issues/6231). The
historical B300 rows above predate the runtime cache and must not be treated as
Phase B memory validation.

### L20X registered-runtime-cache smoke

The read-only registered path was then tested with the same MiniMax-H3
workload. These are functional and mechanism-validation measurements, not a
production benchmark. The TP1 pair used one request per wave. Both modes moved
exactly 121.86 GiB H2D in the profiled request and produced byte-identical video
and audio for both measured seeds.

| TP1 no-AllGather mode | Mean wave | Throughput | Request-time host PSS | `Private_Dirty` | Engine peak HBM | CPU `aten::copy_` | GPU compute |
|---|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint mmap + bounded staging | 33.87 s | 0.0295 req/s | 133.65 GiB | 71.16 GiB | 13,226 MB | 21.91 s | 419.28 ms |
| Final runtime cache + registered direct H2D | 4.88 s | 0.2055 req/s | 129.41 GiB | 66.93 GiB | 13,226 MB | 1.64 s | 419.05 ms |

Registration covered 61.73 GiB in 13 page-aligned ranges. Mean wave latency
fell by 85.6% and throughput increased 6.94x. GPU compute and H2D payload were
unchanged, identifying recurrent host transform/packing as the removed work.
The direct path issued 2,425 H2D events versus 1,945 for packed staging, but
that launch increase was much smaller than the eliminated CPU cost.

Two independent TP2 engines also registered the same two TP-coordinate cache
entries across four workers. Compared with the zero-budget cache, mean wave
latency improved from 9.21/12.55 seconds to 3.80/3.86 seconds for engines A/B;
the private-loader reference was 3.61/3.72 seconds. Registered request-time
host PSS was 224.26 GiB versus 232.23 GiB staged and 366.20 GiB private.
Aggregate CPU `aten::copy_` time was 3.83 seconds registered, 32.49 seconds
staged, and 3.75 seconds private; aggregate GPU compute and the 264.24 GiB H2D
payload were effectively unchanged. Ordinary TP communication remained, but
the extra wait caused by staging-driven rank skew largely disappeared.

All workers explicitly unregistered before releasing their mappings, the
system's acquired-minus-released pin counter returned to its pre-run value,
and every GPU allocation was released. Cold runtime-cache publication remained
expensive because it still performs ordinary loading and multiple full-content
passes; that startup problem is outside the registered transfer scope.

## Recommendations

- Use **DP + DLO AllGather** for the supported throughput and host-memory
  scaling path.
- Use **SP + DLO AllGather** for long-sequence workloads when DP concurrency is
  not the goal.
- Use **no-AllGather** when independent replica execution is required. TP1
  direct-mmap deployments can share checkpoint pages per node. Enable the
  runtime cache only when independent replicas have repeated final layouts and
  benchmark the target CPU/memory topology, especially for matching
  coordinates across multiple TP engines. On CUDA, provide an explicit
  registration budget large enough for the complete final layout when
  recurrent staging cost is unacceptable; leave it at zero when page-locking
  that layout is not operationally acceptable.
- Prefer **HSDP alone** for production HSDP deployments until the combined
  HSDP + DLO no-AllGather path has broader end-to-end coverage.
