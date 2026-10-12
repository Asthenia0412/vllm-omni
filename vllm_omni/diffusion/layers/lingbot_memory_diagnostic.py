"""Opt-in, once-per-session NPU memory evidence outside every graph region."""
import json
import os

GIB=1024**3


def report_lingbot_npu_memory(state, device, logger, *, ops=None):
    target=os.environ.get("LINGBOT_NPU_MEMORY_DIAGNOSTIC_CHUNK")
    if target is None or getattr(device,"type",None)!="npu": return None
    if not os.environ.get("LINGBOT_CAPTURE_DIR") and os.environ.get("LINGBOT_PROFILE_STAGE_RANGES","0")!="1":
        raise ValueError("NPU memory diagnostics require a diagnostic/profiling session, never clean timings")
    try: target=int(target)
    except ValueError as exc: raise ValueError("NPU diagnostic chunk must be a nonnegative integer") from exc
    if target<0: raise ValueError("NPU diagnostic chunk must be a nonnegative integer")
    completed=state.chunk_index-1
    if completed!=target or getattr(state,"_lingbot_npu_memory_reported",False): return None
    if ops is None:
        import torch
        ops=torch.npu
    free,total=ops.mem_get_info(device)
    row={"request_id":state.request_id,"completed_chunk_index":completed,"device":str(device),
         "allocated_bytes":ops.memory_allocated(device),"reserved_bytes":ops.memory_reserved(device),
         "max_allocated_bytes":ops.max_memory_allocated(device),"max_reserved_bytes":ops.max_memory_reserved(device),
         "free_bytes":int(free),"total_bytes":int(total),"minimum_free_bytes":4*GIB,
         "minimum_actual_headroom_passed":int(free)>=4*GIB,"timing_eligible":False,
         "scope":"Once after completed chunk; actual driver free memory and allocator current/lifetime peaks, including loaded weights, condition state and graph pool. No explicit synchronization is issued. Query overhead belongs only to this diagnostic."}
    state._lingbot_npu_memory_reported=True
    logger.warning("LingBot diagnostic NPU memory: %s",json.dumps(row,sort_keys=True))
    if not row["minimum_actual_headroom_passed"]:
        raise RuntimeError("Actual NPU free headroom is below4GiB after graph/condition warmup; restart stage and reduce graph/staging scope")
    return row
