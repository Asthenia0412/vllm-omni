# Distributed Layerwise Offloading

Distributed layerwise offloading (DLO) streams DiT blocks through two device
buffers instead of keeping the complete DiT in HBM. It supports sharded host
weights with AllGather or independently streamed rank-local weights. The
no-AllGather path can optionally share node-local runtime weights through mmap.

See the [DLO feature design](../../../design/feature/offloader/distributed_layerwise_offload.md)
for loader contracts, compatibility details, failure handling, and benchmarks.

## Choose a mode

| Mode | Use when | Host weights | Runtime synchronization |
| --- | --- | --- | --- |
| DLO AllGather (default) | DP ranks execute the same block path in lockstep | About `1 / dp_size` per rank | DLO weight AllGather |
| no-AllGather | Ranks or engines must schedule independently | Complete rank-local layout | No DLO weight collective |
| no-AllGather + runtime cache | Equivalent independent workers share one node | Shared final mmap layout per TP coordinate | No DLO weight collective |
| runtime cache + host registration | The platform supports registration and recurrent staging is too expensive | Shared registered mmap layout | No DLO weight collective |

AllGather is normally the best choice for synchronized DP. Runtime-cache mode
targets independently scheduled replicas, especially repeated TP engines on
one node.

## Usage

```bash
# Sharded host weights with DLO AllGather
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 4

# Independently streamed rank-local weights
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather

# Share final runtime weights and register up to 80 GiB per worker for direct H2D
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --tensor-parallel-size 2 \
  --dlo-no-use-allgather \
  --dlo-enable-runtime-cache \
  --dlo-runtime-cache-pin-limit-gib 80
```

## Relevant options

| Flag | Meaning | Default |
| --- | --- | --- |
| `--enable-distributed-layerwise-offload` | Enable DLO | `false` |
| `--dlo-no-use-allgather` | Stream complete rank-local blocks independently | `false` |
| `--dlo-enable-runtime-cache` | Share final no-AllGather DiT weights through a node-local mmap cache | `false` |
| `--dlo-runtime-cache-pin-limit-gib GIB` | Per-worker registration budget; zero uses bounded host staging | `0` |
| `--dlo-resident-layers N` | Keep eligible leading DiT blocks resident in HBM | `0` |

The runtime cache uses `~/.cache/vllm-omni/dlo-runtime-weights`. Programmatic
configuration may override `dlo_runtime_cache_dir` and the writer lock timeout;
these advanced storage controls are intentionally not separate CLI flags.

## Operational notes

- All workers that should share must see the same local, disk-backed cache
  directory. Do not use tmpfs or a cross-node filesystem for this version.
- Cache entries are immutable and validated before use. A cache or registration
  failure keeps the ordinary loader weights or falls back to bounded staging.
- A positive registration budget must cover the complete page-aligned mapping
  reported in the worker log. Registration is all-or-nothing.
- CUDA is the first registration backend. Platforms without an equivalent
  implementation continue to use bounded staging.
- Registration is process-local but does not duplicate the underlying file
  pages. Each worker must still satisfy its platform and OS page-locking limits.
- Shutdown unregisters host ranges before closing their mmap handles.
- Runtime-cache v1 has no automatic eviction. Stop all users of an entry before
  deleting it.

## Scheduling constraints

AllGather ranks must request blocks in the same collective order. Concurrent DP
requests may use different prompts, but they must follow the same denoising and
block-execution path and set the same explicit `num_inference_steps`.

No-AllGather workers do not have this DLO lockstep requirement. Ordinary TP or
SP model collectives still synchronize ranks within each engine.

## Limitations

- Direct checkpoint mmap currently requires TP1. The runtime cache can share
  final ordinary-loader layouts between matching TP coordinates at TP1 or TP>1.
- Per-tensor online FP8 linears use the ordinary loader and can run with either
  DLO transfer path. Other online quantization methods require no-AllGather
  until their runtime layouts are validated.
- Runtime-cache v1 rejects quantized, non-contiguous, aliased/tied, device-only,
  HSDP, expert-parallel, CFG-parallel, and PP layouts.
- HSDP plus DLO AllGather is unsupported. HSDP without AllGather has limited
  end-to-end validation.
- The runtime cache lowers steady-state host PSS, not startup peak: each worker
  still performs ordinary loading and full content validation.
- Skip-load-on-hit, cache eviction, cross-node sharing, and broader SP hardware
  validation remain follow-up work in
  [RFC #6195](https://github.com/vllm-project/vllm-omni/issues/6195).

See the [Cosmos3 DistOffload recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-DistOffload.md)
for an end-to-end example.
