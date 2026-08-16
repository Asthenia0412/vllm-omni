# Distributed Layerwise Offloading

Distributed layerwise offloading (DLO) extends block streaming to multi-device
deployments. With AllGather enabled, each rank stores roughly `1 / dp_size` of
the host weights and reconstructs each layer at runtime. Without AllGather,
each rank streams a complete block independently. Compatible TP1 deployments
can share checkpoint-backed host pages among processes on the same node;
otherwise DLO streams the ordinary loader's rank-local tensors. The optional
runtime-weight cache extends node-local sharing to final ordinary-loader
layouts, including matching TP coordinates across independent engines.

See the [DLO feature design](../../../design/feature/offloader/distributed_layerwise_offload.md)
for the implementation contract and compatibility matrix.

## Execution model

DLO overlaps three operations with a fixed two-block device buffer:

```text
Compute stream:  [Layer N]          [Layer N+1]        [Layer N+2]
H2D stream:      [H2D shard N+1]    [H2D shard N+2]
AllGather:       [AG N+1]           [AG N+2]
Slots:           slot 0: Layer N    slot 1: Layer N+1
```

AllGather communicates request-independent weight shards, but every member of
the transfer group must request the same next block in the same collective
order. A synchronized DP wave may contain different prompts only when those
requests follow the same denoising and block-execution path.

## Usage

```bash
# Four ranks with sharded host weights and AllGather
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 4

# Standard-loader rank-local weights, without DLO AllGather
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 4 \
  --dlo-no-use-allgather

# Independently scheduled replicas sharing final runtime weights on one node
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --tensor-parallel-size 2 \
  --dlo-no-use-allgather \
  --dlo-enable-runtime-cache \
  --dlo-runtime-cache-dir /var/cache/vllm-omni/dlo-runtime-weights

# Sequence parallel deployment
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --usp 4
```

```python
from vllm_omni import Omni

omni = Omni(
    model="/path/to/model",
    enable_distributed_layerwise_offload=True,
    dlo_use_allgather=False,
    dlo_enable_runtime_cache=True,
    dlo_runtime_cache_dir="/var/cache/vllm-omni/dlo-runtime-weights",
)
```

## Flags

| Flag | Meaning | Default |
| --- | --- | --- |
| `--enable-distributed-layerwise-offload` | Enable DLO | `false` |
| `--data-parallel-size N` | DP ranks and AllGather weight-sharding group | `1` |
| `--dlo-use-allgather` | Shard host weights and reconstruct with AllGather | `true` |
| `--dlo-no-use-allgather` | Stream complete rank-local blocks without a DLO weight collective | `false` |
| `--dlo-enable-runtime-cache` | Normalize final ordinary-loader DiT weights into a node-local mmap cache; requires no-AllGather | `false` |
| `--dlo-runtime-cache-dir PATH` | Shared local-disk cache root | `~/.cache/vllm-omni/dlo-runtime-weights` |
| `--dlo-runtime-cache-lock-timeout SECONDS` | Maximum wait for another process publishing the same layout | `600` |
| `--dlo-resident-layers N` | Keep N leading main-DiT blocks on device; requires no-AllGather and model-declared resident paths | `0` |

## Host-weight loading

The diffusion loader chooses host storage before DLO is enabled. It first
attempts to build a complete, validated direct-checkpoint mmap plan. If names,
coverage, shape, dtype, topology, or loader-callback compatibility cannot be
proven, it runs the ordinary model loader instead. DLO consumes that result and
does not make a second checkpoint-compatibility decision.

The shared-mmap optimization in this phase is supported only with TP1. TP
greater than one falls back before model mutation to ordinary TP-aware loading.
DLO may still consume those TP-local tensors, but this is a compatibility path:
it does not share checkpoint-backed runtime weights across DP replicas and
provides no shared-mmap host-memory guarantee.

When `--dlo-enable-runtime-cache` is set with no-AllGather, that fallback gains
a second storage opportunity. Every rank still completes ordinary loading,
loader callbacks, post-load processing, validation, and calibration. It then
hashes the final CPU DiT parameters and persistent buffers. One process per
equivalent runtime layout publishes sharded safetensors plus a checksummed
manifest; peers validate and mmap the same files. DLO replaces only tensor
storage, preserves the final `Parameter` objects, and releases the private
allocator-backed copies before installing its streaming hooks.

The runtime layout identity excludes DP rank and SP rank. It includes TP world
size/rank and conservative SP implementation/world-size guards. Therefore two
independent `TP=2` engines normally create two entries: both TP0 workers map
one entry and both TP1 workers map the other. Cache sharing does not create a
process group or an inference-time collective.

The mmap plan skips only dedicated DiT weight sources. Other component sources,
such as a text encoder loaded through the shared diffusion loader, continue to
use their ordinary component loader. A checkpoint source that mixes DiT and
non-DiT weights falls back completely rather than leaving an unplanned
component uninitialized.

With direct checkpoint mmap, the loader:

1. saves non-persistent buffers such as RoPE frequencies;
2. moves the normally created transformer to the meta device;
3. loads checkpoint tensors as mmap views backed by the shared OS page cache;
4. applies any loader-owned bounded layout adapters while packing blocks;
5. restores saved non-persistent buffers; and
6. preserves `post_load_weights()` and `validate_loaded_weights()` lifecycle
   hooks.

For AllGather with a group larger than one, each process copies only its
persistent shard and then releases the source mapping. For no-AllGather, each
process keeps the mapping open and packs complete blocks through two bounded
pinned staging slots. Processes mapping the same files on one node share the
immutable pages; no-AllGather still performs a complete-block H2D copy in each
process.

When the effective DLO group size is one, `dlo_use_allgather=True` does not
perform a collective and uses the same rank-local transfer behavior.

## Runtime-cache lifecycle

Runtime-cache publication uses a per-layout POSIX file lock, a temporary
directory on the same filesystem, file and final-tensor SHA-256 validation,
`fsync`, and an atomic directory rename. A crashed writer releases its kernel
lock; the next writer removes only that layout's stale temporary directories
while holding the lock. Lock timeout, unsupported layout, insufficient disk,
or validation failure keeps the already-valid ordinary tensors.

The cache root is deliberately independent of `VLLM_CACHE_ROOT`, because
vLLM-Omni may isolate that variable between stage replicas. All consumers that
should share must receive the same `--dlo-runtime-cache-dir` on a local,
disk-backed filesystem. Do not place the cache on tmpfs when the goal is lower
host-memory PSS, and do not use a cross-node directory in this version.

There is no automatic eviction. Budget roughly one final DiT copy per distinct
TP coordinate, dtype/layout, and model version, plus temporary space while an
entry is published. Operators may remove entries or the complete cache root
only after all workers using its mmap files have stopped. The feature is
opt-in for this reason.

## Declarative topology

Models may declare an `OffloadPlan` instead of embedding offload logic:

```python
from vllm_omni.diffusion.offloader import OffloadPlan


class MyPipeline(nn.Module):
    _dit_modules = ["transformer"]
    _offload_plan = OffloadPlan(
        block_attrs={"transformer": ("blocks",)},
        offload_submodules={"context_encoder": "layers"},
    )
```

When no plan exists, discovery falls back to
`_layerwise_offload_blocks_attrs` and then heuristic attribute lookup.

## Data-parallel concurrency

With `data_parallel_size > 1` and AllGather enabled, the scheduler can process
up to `dp_size` requests per denoising step. Every concurrent request must set
the same explicit `num_inference_steps`; `None` is rejected because every rank
must enter each collective.

## Limitations

- Direct checkpoint mmap currently requires TP1. TP greater than one falls
  back to the ordinary TP-aware loader. With the optional runtime cache,
  equivalent processes at the same TP coordinate can then share that final
  rank-local layout in no-AllGather mode.
- HSDP plus AllGather is rejected to avoid double sharding. HSDP without
  AllGather has limited end-to-end validation.
- Per-tensor online FP8 linears use the ordinary loader and can run with either
  DLO transfer path. With AllGather, every rank temporarily materializes the
  complete FP8 model in host memory before DLO retains only its shard. Other
  online quantization methods require no-AllGather until their runtime layouts
  are validated.
- Runtime-cache v1 rejects all quantized, non-contiguous, aliased/tied,
  device-only, HSDP, expert-parallel, CFG-parallel, and PP layouts.
- Resident leading layers require `--dlo-no-use-allgather` and a model
  `OffloadPlan` that declares eligible `resident_dit_paths`.
- DP concurrency requires an explicit, identical inference-step count.

The runtime cache improves steady-state host PSS, not startup peak: every rank
loads private weights before remapping, and validation adds full content-hash
passes. Skip-load-on-cache-hit and cache eviction remain follow-up work in
[RFC #6195](https://github.com/vllm-project/vllm-omni/issues/6195).

See the [Cosmos3 DistOffload recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-DistOffload.md)
for an end-to-end example.
