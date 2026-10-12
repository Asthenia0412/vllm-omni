# CANN Profiling Handbook — Kernel-Level Compute & Communication

**Site-specific handbook for fused-kernel creation and measurement.**
Machine: 8× Ascend910, CANN **8.0.RC3** (`/usr/local/Ascend/ascend-toolkit/latest`).
Verified on 2026-09-29. All commands below were smoke-tested on this host unless marked *(container-only)*.

---

## 1. Quick Reference

| Goal | Tool | One-liner |
|---|---|---|
| Whole-model trace incl. HCCL | `torch_npu.profiler` (Level1) | see §3 — *(container-only)* |
| Whole-app CLI profile | `msprof` | see §4 — works from host |
| Single fused-kernel, full PMU counters | `msprof op` (msopprof) | see §5 |
| Communication baseline (bw/latency) | `hccl_test` prebuilt binaries | see §6.3 — verified |
| Parse/analyze collected PROF dirs | `msprof.py` CLI + analyzers | see §7 — installed on host |

**Golden path for a fused-kernel A/B measurement:** §8 (protocol).

---

## 2. Environment Setup (this host)

```bash
# 1) CANN environment (required for every msprof/hccl command)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2) MPI for hccl_test multi-device runs
export LD_LIBRARY_PATH=/usr/local/mpich-4.1.3/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/mpich-4.1.3/bin:$PATH
```

Installed on host Python 3.9 (2026-09-29):

- `msprof-0.0.1` (analysis package) — from
  `/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/msprof-0.0.1-py3-none-any.whl`
- `hccl_parser-0.1` — from `/usr/local/Ascend/ascend-toolkit/latest/tools/hccl_parser-0.1-py3-none-any.whl`

> ⚠️ **Import quirk (CANN 8.0.RC3 wheel):** the analysis wheel installs a
> top-level package literally named `analysis`. Its internal imports
> (`analyzer.*`, `msinterface.*`) only resolve when the `analysis/` directory
> itself is on `sys.path`:
>
> ```python
> import sys
> sys.path.insert(0, '/home/zhoutaichang/.local/lib/python3.9/site-packages/analysis')
> from msinterface.msprof_analyze import AnalyzeCommand          # this version exports this
> from analyzer.communication_analyzer import CommunicationAnalyzer
> ```
>
> The PyPI-style `from msprof import ProfilerAnalyzer` belongs to CANN ≥ 8.1's
> pip package and **does not exist here**. Prefer the `msprof.py` CLI (§7.1).

Reinstall after a CANN upgrade:

```bash
pip3 install --user --force-reinstall \
  /usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/msprof-*-py3-none-any.whl \
  /usr/local/Ascend/ascend-toolkit/latest/tools/hccl_parser-*-py3-none-any.whl
```

**Container note:** `torch_npu` is *not* on host Python. In-framework profiling
(§3) runs inside the vllm-ascend runtime container (install the same two wheels
there if you want programmatic analysis; collection itself needs nothing extra).

---

## 3. Method 1 — In-framework: `torch_npu.profiler` *(container-only)*

The only method that gives you Python/operator ↔ kernel correlation plus HCCL
data in **one** trace. This is what vLLM-Ascend already wires up at
`vllm-ascend/vllm_ascend/profiler/torch_npu_profiler.py`.

**Key fact: there is no `with_hccl` flag.** Communication data is captured by
`profiler_level=Level1`, which emits `communication.json`,
`communication_matrix.json` and `api_statistic.csv` in addition to kernel data.

### Levels

| Level | Adds |
|---|---|
| `Level0` | app data, NPU data, partial operator info |
| `Level1` | **AscendCL data, AI Core metrics, HCCL communication files** ← default here |
| `Level2` | + CANN runtime data, `data_preprocess.csv` (heavier) |

### Standalone recipe

```python
import torch, torch_npu
from torch_npu.profiler import (
    profile, ProfilerActivity, _ExperimentalConfig, ExportType,
    ProfilerLevel, AiCMetrics, tensorboard_trace_handler,
)

exp = _ExperimentalConfig(
    export_type=ExportType.Text,            # .Db for programmatic analysis
    profiler_level=ProfilerLevel.Level1,    # ← REQUIRED for HCCL + AIC metrics
    aic_metrics=AiCMetrics.PipeUtilization, # or ArithmeticUtilization / Memory / MemoryUB / L2Cache ...
    msprof_tx=False, l2_cache=False, op_attr=False,
    data_simplification=True, record_op_args=False, gc_detect_threshold=None,
)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    schedule=None,                            # or torch.profiler.schedule(wait=1, warmup=2, active=6)
    on_trace_ready=tensorboard_trace_handler("./prof_out", worker_name="fused_ab"),
    experimental_config=exp,
    record_shapes=True,                       # tensor shapes in op_summary — useful for shape-sensitive fusion
    with_stack=False, profile_memory=False,
) as prof:
    for _ in range(warmup + iters):           # warmup is critical: skip JIT/tiling-cache effects
        step()
        prof.step()
```

Under vLLM: `vllm serve ... --profile --torch-profiler-dir ./prof_out` style
flags (see `vllm_ascend/profiler/`), or embed the snippet around the region of
interest in a bench script.

### What you get (Text export, under `ASCEND_PROFILER_OUTPUT/`)

| File | Content |
|---|---|
| `op_summary_*.csv` | **per-kernel**: name, task type (AI Cube / AI Vector / **Communication** / MixAIV ...), duration, start time, aic metrics columns |
| `trace_view.json` | Chrome trace (open in `chrome://tracing` / Perfetto) — shows HCCL kernel rows next to compute, overlap visible |
| `communication.json` | HCCL API-level: per communication op, per rank start/duration |
| `communication_matrix.json` | per-op × per-rank-group traffic matrix (bytes, time, bandwidth) |
| `api_statistic.csv` | host-side API time statistics |

**Fusion read-out:** in `op_summary`, HCCL collectives appear with task type
`Communication`. After fusing comm+compute, that row should disappear (or
shrink) and the fused kernel row grows — compare totals with
`summarize_profile.py` (§7.3).

---

## 4. Method 2 — Whole-app CLI: `msprof` (host)

Wraps any launch command (python bench script, vLLM server, etc.). No code
changes needed. **Use `--delay`/`--duration` to skip model load and JIT noise.**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

msprof --output=./prof_out \
       --application="python3 bench_fused.py" \
       --task-time=l1 \
       --ai-core=on --aic-metrics=PipeUtilization \
       --hccl=on \
       --runtime-api=on \
       --delay=30 --duration=60 \
       --output-profiler-off \
       --export=on --type=csv          # auto-parse at end; adds summary CSVs
```

Useful flags (full list: `msprof --help`):

| Flag | Purpose |
|---|---|
| `--task-time=l1` | task-level kernel timing (l0/l1/on/off) |
| `--ai-core --aic-metrics=<group>` | kernel HW metrics: `ArithmeticUtilization, PipeUtilization, Memory, MemoryL0, MemoryUB, L2Cache, ResourceConflictRatio`, or `Custom:reg1,reg2` |
| `--aic-mode=task-based` | per-task attribution (default) vs sample-based |
| `--hccl=on` | HCCL comm data collection |
| `--delay/--duration` | skip init; bound collection window |
| `--sys-hardware-mem=on` | HBM/LLC bandwidth sampling (910 supported) |
| `--export=on --type=csv\|db --summary-format=csv\|json` | auto-export summaries after collection |

Analyze a collected PROF dir afterwards (no re-run):

```bash
msprof --output=./prof_out/PROF_xxx --export=on --type=csv                 # kernel summaries
msprof --output=./prof_out/PROF_xxx --analyze=on \
       --rule=communication,communication_matrix                          # comm rule reports
```

---

## 5. Method 3 — Isolated single-kernel: `msprof op` (msopprof) ★ for fused-kernel A/B

The precision instrument: matches **one kernel by name**, skips warmup
launches, collects exactly N launches with deep hardware counters. Ideal to
measure a fused kernel vs. the unfused sequence at identical shapes.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # needs ASCEND_TOOLKIT_HOME

# Direct mode: wrap a python app that launches the kernel (torch_npu app → run in container)
msprof op --application="python3 launch_fused_kernel.py" \
          --output=./op_prof_out \
          --kernel-name=dispatch_ffn_combine_kernel \
          --launch-skip-before-match=20 \
          --launch-count=30 \
          --aic-metrics=ArithmeticUtilization,MemoryUB,Memory,L2Cache \
          --kill=off
```

Or config-file mode — see template `msprof_op_config.json` in this directory:

```bash
msprof op --config=msprof_op_config.json
```

| Flag | Meaning |
|---|---|
| `--kernel-name` | substring match on device kernel name |
| `--launch-skip-before-match` | skip N launches first (warmup/JIT/tiling-cache) — range 0–1000 |
| `--launch-count` | collect N launches — range 1–100 |
| `--aic-metrics` | `ArithmeticUtilization / MemoryUB / Memory / MemoryL0 / L2Cache / PipeUtilization / ResourceConflictRatio / Roofline / Occupancy / Default` |
| `--kill=on` | kill app once N kernels collected (good for long-running servers) |
| `--mstx` / `--mstx-include` | collect only ranges marked with mstx markers |

Outputs land in `op_prof_out/PROF_*/`: `op_summary_*.csv` with the extra metric
columns, plus per-launch details. **Metric groups are mutually exclusive per
run** — do one run per group (e.g., PipeUtilization run, then Memory run).

> For a torch workload, write `launch_fused_kernel.py` so it: builds inputs at
> fixed shapes → runs the op `skip+count` times on one stream → exits. Synchronize
> only at the end; no per-iteration syncs.

---

## 6. Communication Profiling

### 6.1 HCCL data inside framework traces — §3 Level1
`communication.json` (per-op per-rank timings), `communication_matrix.json`
(bytes/bw matrix). HCCL kernels also appear in `op_summary` typed
`Communication` → overlap with compute is directly measurable in the timeline.

### 6.2 CLI rule analysis
```bash
msprof --output=<PROF_dir> --analyze=on --rule=communication,communication_matrix
```
Produces per-collective time/traffic/bandwidth reports across ranks from an
existing PROF dir.

### 6.3 Pure communication baselines — `hccl_test` (**verified on this host**)

**Use the prebuilt binaries — do not rebuild** (a rebuild attempt in a copied
dir fails: the source tree is root-owned read-only, and the toolkit already
ships ARM64 binaries under `tools/hccl_test/bin/`).

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/mpich-4.1.3/lib:$LD_LIBRARY_PATH
HCCL_TEST=/usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test

# single device (sanity / loopback)
$HCCL_TEST/bin/all_reduce_test                      # default: 64MB, 5 warmup + 20 iters

# 8 devices via MPI
mpirun -np 8 $HCCL_TEST/bin/all_reduce_test -b 1M -e 128M -f 2 -w 5 -n 20
```

Flags: `-b/-e` min/max bytes (`8M` style accepted), `-f` multiply step,
`-w` warmup iters, `-n` measured iters, `-p` #devices.
Binaries: `all_reduce_test all_gather_test reduce_scatter_test reduce_test
broadcast_test scatter_test alltoall_test alltoallv_test`.

Reference numbers measured on this host (2026-09-29):

| Test | Size | Avg time | Algo bandwidth |
|---|---|---|---|
| all_reduce, 1 dev | 64 MB | 107.95 µs | 621.7 GB/s |
| all_reduce, 8 dev | 8 MB | 217.40 µs | 38.6 GB/s |

**Fusion judgement rule:** fused comm+compute kernel wins if
`t_fused < t_compute_best + t_comm_baseline(size)` at the same shapes, where
`t_comm_baseline` comes from these tables (interpolate for exact sizes).

---

## 7. Post-Processing & Programmatic Analysis

### 7.1 `msprof.py` analysis CLI (installed on host)

```bash
MSPROF_PY=~/.local/lib/python3.9/site-packages/analysis/msprof/msprof.py

python3 $MSPROF_PY import  --path=<PROF_dir>              # parse raw → db
python3 $MSPROF_PY export  --path=<PROF_dir> --format=csv # summaries as CSV
python3 $MSPROF_PY query   --path=<PROF_dir> ...          # structured queries
python3 $MSPROF_PY analyze --path=<PROF_dir> --rule=communication
```

### 7.2 Python analyzer API (host)

```python
import sys
sys.path.insert(0, '/home/zhoutaichang/.local/lib/python3.9/site-packages/analysis')
from analyzer.communication_analyzer import CommunicationAnalyzer
from analyzer.communication_matrix_analyzer import CommunicationMatrixAnalyzer
```

### 7.3 `summarize_profile.py` — quick kernel/comm split (this directory)

Stdlib-only; works on any `PROF_*` dir or `ASCEND_PROFILER_OUTPUT` dir
produced by §3/§4/§5:

```bash
python3 summarize_profile.py ./prof_out/PROF_2026xxxx --top 15
```

Prints: compute vs communication time split, top kernels by total duration,
per-collective HCCL table. Use it for before/after-fusion comparisons.

---

## 8. Fused-Kernel Workflow (creation → measurement)

### 8.1 Creation

Follow the established pattern in `vllm-ascend/csrc/` (Ascend C custom op):

```
csrc/<my_fused_op>/
  op_host/   *_proto.cpp  *_tiling.cpp  *_def.cpp   + aclnn op_api adapter
  op_kernel/ *.cpp *.hpp  (Ascend C kernel)
```
+ torch binding in `torch_binding*.cpp`, build with `csrc/build.sh`.

**Working reference for comm+compute fusion: `vllm-ascend/csrc/mc2/dispatch_ffn_combine*/`**
— dispatch → FFN → combine with in-kernel HCOM communication
(see `op_host/hcom_topo_info.h`). Standalone scaffolding alternative:
`msopgen` templates (`$ASCEND_HOME_PATH/tools/msopgen/template`).

### 8.2 Measurement protocol

1. **Locate the target region** — run a Level1 framework trace (§3) around the
   model region; identify kernel sequence + HCCL ops, their durations, and the
   gap/overlap between them (`summarize_profile.py`, timeline).
2. **Communication baseline** — `hccl_test` for the exact collective + message
   size (§6.3).
3. **Isolated unfused baseline** — `msprof op` per kernel of the unfused
   sequence (§5): fixed shapes, skip ≥ 20 warmup launches, 30+ measured,
   one metric-group per run.
4. **Fused kernel measurement** — same `msprof op` recipe with
   `--kernel-name=<fused_kernel>`; then an end-to-end Level1 trace to confirm
   the `Communication` op disappeared into the fused op and step time dropped.

### 8.3 Avoid these measurement traps

- **No warmup skip → garbage numbers.** First launches pay tiling/JIT/cache
  costs. Always use `launch-skip-before-match` (isolated) or
  `schedule(wait, warmup, active)` (framework).
- **Metric groups are per-run.** `PipeUtilization` and `Memory` cannot be
  collected in the same `msprof op` run.
- **Different shapes ≠ comparable.** Pin shapes; report them with results
  (`record_shapes=True`).
- **HCCL init pollutes traces.** Use `--delay`, or schedule-warmup, so the
  trace window excludes communicator setup.
- **`data_simplification=True` deletes raw files** after parsing — set it
  `False` if you need the original `PROF_*` internals for msprof.py re-import.

---

## 9. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `cannot find msopprof ... not set ASCEND_TOOLKIT_HOME` | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` first |
| `ModuleNotFoundError: msprof` | Wrong import path for 8.0.RC3 wheel — use §2 sys.path pattern or `msprof.py` CLI |
| `No module named 'analyzer'` | You imported `analysis.msinterface...` without putting the `analysis/` dir itself on `sys.path` — see §2 |
| `cannot open output file ... Permission denied` (hccl_test rebuild) | Toolkit tree is root-owned; use the prebuilt `bin/*_test` (§6.3) |
| torch_npu absent | Framework profiling runs in the vLLM-Ascend container, not on host Python |
| Empty HCCL files in trace | `profiler_level` must be `Level1`; also ensure ≥2 ranks actually communicated in the window |
| `--analyze` finds no data | Run `--export=on`/`import` first; only `PROF_*` dirs with parsed dbs can be analyzed |

## 10. Where things live

- Toolkit: `/usr/local/Ascend/ascend-toolkit/latest` (CANN 8.0.RC3)
- `msprof` CLI: `$ASCEND_HOME_PATH/bin/msprof`; op-mode binary `msopprof`
- Analysis package: `~/.local/lib/python3.9/site-packages/analysis` (+`hccl_parser`)
- hccl_test prebuilt: `/usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test/bin/`
- MPI: `/usr/local/mpich-4.1.3` (MPICH 4.1.3)
- This toolkit's profiler tooling home: `$ASCEND_HOME_PATH/tools/profiler/profiler_tool`
- vLLM-Ascend profiler integration: `vllm-ascend/vllm_ascend/profiler/torch_npu_profiler.py`
- Fused comm+compute reference kernel: `vllm-ascend/csrc/mc2/dispatch_ffn_combine/`
