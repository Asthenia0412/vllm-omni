"""Default-OFF SP8 receive/current-KV byte copies on the ordinary caller.

No framework import, DSO load or device call at module import. The original
manager supplies all pools, slots, staging and commit decisions. A failed
admitted forward is poisoned, never retried using partly written caches.
"""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading

FLAG = "LINGBOT_RECV_Q_CURRENT_KV_EAGER"
MANIFEST = "ROOT_LINGBOT_RECV_KV_SP8_MANIFEST"
MANIFEST_SHA = "ROOT_LINGBOT_RECV_KV_SP8_MANIFEST_SHA256"
_STATE = threading.local()
_BOUND = None
_FATAL = None
_QUARANTINE = []
META = {"video_slots": "current_video_slot_mapping", "action_slots": "action_slot_mapping",
        "block_table": "block_table", "query_start_loc": "query_start_loc", "seq_lens": "seq_lens"}
ROLES = ("key_pool", "value_pool", "stage_key", "stage_value")


def enabled():
    raw = os.environ.get(FLAG, "0")
    if raw not in ("0", "1"):
        raise ValueError(FLAG + " must be exactly 0/1")
    if _BOUND is not None and raw != "1":
        poison()
        raise RuntimeError("SP8 ingress cannot be disabled after binding; restart worker")
    if raw == "1" and getattr(_STATE, "poisoned", False):
        raise RuntimeError("SP8 ingress is poisoned; restart worker")
    return raw == "1"


def is_active():
    """Cheap default-OFF hook predicate; no environment or framework access."""
    return getattr(_STATE, "prepared", None) is not None


def _fatal_module():
    global _FATAL
    if _FATAL is not None:
        return _FATAL
    name = "vllm_omni.diffusion.layers.lingbot_sp8_fatal"
    existing = sys.modules.get(name)
    path = Path(existing.__file__).resolve() if existing is not None else Path(__file__).with_name("lingbot_sp8_fatal.py").resolve()
    if _BOUND is not None and "manifest" in _BOUND:
        pins = [digest for filename, digest in _BOUND["manifest"]["source_and_header_pins"].items()
                if Path(filename).name == path.name]
        if len(pins) != 1 or _sha(path) != pins[0]:
            raise RuntimeError("actual shared SP8 fatal module differs from its unique source pin")
    if existing is None:
        spec = importlib.util.spec_from_file_location(name, path)
        existing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(existing)
        sys.modules[name] = existing
    _FATAL = existing
    return existing


def poison(reason="admitted SP8 forward failed"):
    if not getattr(_STATE, "poisoned", False):
        owners = (getattr(_STATE, "prepared", None), getattr(_STATE, "receive", None), _BOUND,
                  getattr(_STATE, "dispatched", None))
        _QUARANTINE.append(owners)
        _STATE.poisoned = True
        if any(owner is not None for owner in owners):
            try:
                _fatal_module().report_failure("recv_ingress", str(reason), owners)
            except BaseException as error:
                # Preserve the original error even when a source pin/channel is
                # broken. The root also monitors this explicit stderr marker.
                try:
                    sys.stderr.write("LINGBOT_SP8_FATAL_REPORT_FAILED " + json.dumps(dict(
                        kind="lingbot_sp8_fatal_report_failed", version=1, pid=os.getpid(),
                        nonce=os.environ.get("ROOT_LINGBOT_SP8_FATAL_NONCE"), origin="recv_ingress",
                        reason=str(reason), report_error=str(error), restart_scope="all8")) + "\n")
                    sys.stderr.flush()
                except BaseException:
                    pass
    _STATE.poisoned = True


def require(condition, message):
    if not condition:
        poison(message)
        raise RuntimeError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_from_manifest():
    global _BOUND
    require(enabled() and _BOUND is None, "SP8 ingress binds once")
    path = Path(os.environ.get(MANIFEST, "")).resolve(strict=True)
    expected = os.environ.get(MANIFEST_SHA, "")
    require(len(expected) == 64 and _sha(path) == expected, "explicit SP8 build manifest SHA required")
    manifest = json.loads(path.read_text())
    require(manifest.get("status") == "CPU_BUILD_ONLY_SP8_RECV_KV" and
            manifest.get("build_passed") is True and manifest.get("NPU_execution") is False,
            "own completed SP8 CPU build required; device/quality admission is separate")
    for filename, digest in manifest["source_and_header_pins"].items():
        require(_sha(filename) == digest, "SP8 source/header pin changed: " + filename)
    artifacts = manifest["artifacts"]
    for pin in artifacts.values():
        require(Path(pin["path"]).stat().st_size == pin["bytes"] and _sha(pin["path"]) == pin["sha256"],
                "SP8 built artifact changed")
    # Root freezes identical helper bytes under candidate_source for each run.
    # Relocation must not bypass any original source/header pin or accept a
    # differently named module: bind the sole matching basename and same bytes.
    actual_helper = Path(__file__).resolve()
    helper_pins = [digest for filename, digest in manifest["source_and_header_pins"].items()
                   if Path(filename).name == actual_helper.name]
    require(len(helper_pins) == 1 and _sha(actual_helper) == helper_pins[0],
            "actual helper bytes must match the uniquely pinned helper basename")
    import torch
    import torch_npu
    require(torch_npu.npu.is_initialized(), "bind only after the original NPU initialization")
    require(int(torch._C._GLIBCXX_USE_CXX11_ABI) == manifest["torch_cxx11_abi"], "Torch ABI changed")
    internal = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    require(internal in (b"enable", "enable") and torch_npu.npu.is_jit_compile_false() is True,
            "SP8 ingress preserves serving internal-format enable/JITFalse")
    driver = Path(artifacts["driver"]["path"]).resolve(strict=True)
    torch.ops.load_library(str(driver))
    library = ctypes.CDLL(str(driver))
    class DlInfo(ctypes.Structure):
        _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                    ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    info = DlInfo()
    # Resolve the driver's actual linked relocation, not an unrelated dlsym.
    address = torch.ops.lingbot_recv_kv_sp8_eager.linked_raw_launch_address()
    require(dladdr(ctypes.c_void_p(address), ctypes.byref(info)) and info.filename,
            "cannot identify the actual linked SP8 raw kernel")
    actual = Path(info.filename.decode()).resolve(strict=True)
    require(actual == Path(artifacts["raw"]["path"]).resolve(strict=True) and
            _sha(actual) == artifacts["raw"]["sha256"], "SP8 raw symbol was interposed")
    _BOUND = {"torch": torch, "npu": torch_npu, "library": library, "manifest": manifest,
              "ingress": torch.ops.lingbot_recv_kv_sp8_eager.ingress_eager}


def describe(tensor):
    storage = tensor.untyped_storage()
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(), str(tensor.dtype),
            str(tensor.device), tensor.data_ptr(), storage._cdata, storage.data_ptr(), storage.nbytes())


def metadata_describe(tensor):
    # Detect same-owner framework in-place mutation without timing-path readback.
    # Pools/staging are writable and deliberately do not use this snapshot.
    return describe(tensor) + (tensor._version,)


def manager_geometry(manager):
    """Bind full rank-local allocations; managed pages exclude scratch pages."""
    values = tuple(getattr(manager, key) for key in (
        "managed_num_blocks", "scratch_num_blocks", "num_blocks_total", "block_size",
        "num_kv_heads", "head_size", "history_staging_tokens", "num_layers"))
    require(all(type(value) is int and value > 0 for value in values), "invalid original manager geometry")
    managed, scratch, total, block, heads, dim, stage, layers = values
    require(total == managed + scratch and (block, heads, dim, layers) == (1560, 5, 128, 40) and
            stage % block == 0 and stage >= 4680, "SP8 manager geometry changed")
    require(tuple(manager.history_staging_layer_indices) == tuple(range(40)), "all forty staging owners required")
    return values


def _ordinary_execution_supported():
    """Classify an unsupported caller before binding or custom cache writes.

    An admitted forward still uses _ordinary() for every write: changing its
    execution mode or stream is an error, never a mid-forward native retry.
    """
    if _BOUND is None:
        import torch
        import torch_npu as npu
    else:
        torch, npu = _BOUND["torch"], _BOUND["npu"]
    if torch.compiler.is_compiling() or npu.npu.is_current_stream_capturing():
        return False
    return npu.npu.current_stream() == npu.npu.default_stream()


def _ordinary():
    require(_BOUND is not None, "SP8 ingress has not bound its actual build")
    torch, npu = _BOUND["torch"], _BOUND["npu"]
    require(not torch.compiler.is_compiling() and not npu.npu.is_current_stream_capturing(),
            "SP8 ingress admits ordinary eager execution only")
    stream = npu.npu.current_stream()
    require(stream == npu.npu.default_stream(), "original managed default caller stream required")
    return stream


def host_mapping(ctx):
    geometry = manager_geometry(ctx.kv_cache)
    managed, _, total, block, _, _, stage, _ = geometry
    require(ctx.seq_len == 4680 and ctx.block_size == block and ctx._action_len == 0 and
            not ctx.action_scratch_block_ids and ctx.staging_enabled and
            type(ctx.reuse_history) is bool and ctx.query_len == 4680 and
            ctx.host_kv_len == ctx.kv_len and ctx.kv_len % block == 0,
            "original zero-action aligned staging required")
    all_ids = tuple(ctx.history_block_ids) + tuple(ctx.current_video_block_ids)
    maximum = ctx.max_video_tokens // block
    sink = int(ctx.kv_cache.spec.sink_chunks)
    require(ctx.max_video_tokens % block == 0 and 0 <= sink <= maximum, "original window policy changed")
    visible = all_ids if len(all_ids) <= maximum else all_ids[:sink] + (all_ids[-(maximum-sink):] if maximum > sink else ())
    current, first = tuple(ctx.current_video_block_ids), ctx.stage_first_block
    require(len(current) == 3 and len(visible) == ctx.kv_len // block and
            type(first) is int and first >= 0 and first % 3 == 0 and first + 3 == len(visible) and
            visible[first:] == current and (first + 3) * block <= stage,
            "current pages must be exact visible trailing three within actual staging")
    null = ctx.kv_cache.null_block_id
    require(all(type(page) is int and 0 <= page < total and page != null for page in visible) and
            len(set(visible)) == len(visible), "visible IDs must be valid, non-null, unique and disjoint")
    require(all(0 <= page < managed and page != null for page in ctx.history_block_ids),
            "history must contain original managed non-null pages")
    if ctx.commit_current:
        start = int(ctx.adapter.num_computed_tokens)
        table = tuple(ctx.kv_cache.block_table(ctx.adapter))
        require(start % block == 0 and table[start // block:start // block + 3] == current and
                all(0 <= page < managed and page != null for page in current),
                "original managed current write targets changed")
    else:
        require(tuple(ctx.kv_cache.scratch_block_ids(ctx.kv_branch, 0, 3)) == current,
                "original branch-private scratch write targets changed")
    return (visible, current, first, ctx.reuse_history, int(ctx.adapter.num_computed_tokens),
            int(ctx.adapter.absolute_num_computed_tokens), ctx.kv_branch, ctx.commit_current)


def prepare_context(ctx):
    if not enabled():
        return
    previous = getattr(_STATE, "prepared", None)
    require(getattr(_STATE, "receive", None) is None and getattr(_STATE, "dispatched", None) is None and
            (previous is None or (previous["completed"] == set(range(40)) and previous["published"])),
            "abandoned/partly written SP8 forward requires worker restart")
    # Ordinary unsupported contexts stay native before any custom prefix/write.
    manager = ctx.kv_cache
    if (manager.num_kv_heads != 5 or ctx.seq_len != 4680 or ctx._action_len != 0 or
            not ctx.staging_enabled or tuple(manager.history_staging_layer_indices) != tuple(range(40))):
        _STATE.prepared = None
        ctx._recv_kv_sp8_record = None
        return
    if not _ordinary_execution_supported():
        _STATE.prepared = None
        ctx._recv_kv_sp8_record = None
        return
    if _BOUND is None:
        bind_from_manifest()
    fatal = _fatal_module()
    fatal.ensure_protocol()
    stream = _ordinary()
    require(sys._getframe(1).f_code is type(ctx).prepare.__code__, "exact original prepare seam required")
    require(ctx._prepared and ctx._allocated_video and not ctx._committed, "original pre-commit prepare required")
    module = sys.modules["vllm_omni.experimental.ar_diffusion.kv_cache.manager"]
    require(type(manager) is module.ARDiffusionKVCache and type(ctx.adapter) is module.ARDiffusionRequestAdapter,
            "authoritative original manager and adapter required")
    refs = {name: getattr(ctx, attr) for name, attr in META.items()}
    require(all(t is not None for t in refs.values()), "original metadata owners unavailable")
    geometry, proof = manager_geometry(manager), host_mapping(ctx)
    width = refs["block_table"].shape[1]
    require(tuple(refs["video_slots"].shape) == (4680,) and
            tuple(refs["block_table"].shape) == (1, width) and
            tuple(refs["action_slots"].shape) == (0,) and ctx.max_seq_len == width * 1560 and
            ctx.max_seq_len <= geometry[6], "original metadata exceeds actual staging capacity")
    record = {"ctx": ctx, "manager": manager, "adapter": ctx.adapter, "proof": proof, "geometry": geometry,
              "refs": refs, "descriptions": {name: metadata_describe(t) for name, t in refs.items()},
              "stage_state": manager.history_staging_state, "stage_signature": ctx._history_stage_signature,
              "stage_owner": ctx._history_stage_owner, "stream": stream, "thread": threading.get_ident(),
              "layers": {}, "layer_indices": {}, "history_ids_owner": ctx.history_block_ids,
              "current_ids_owner": ctx.current_video_block_ids, "completed": set(), "published": False}
    ctx._recv_kv_sp8_record = record
    _STATE.prepared = record


def _current(record, *, check_stream=False):
    require(not getattr(_STATE, "poisoned", False) and record is getattr(_STATE, "prepared", None) and
            record["thread"] == threading.get_ident(), "current original forward/thread changed")
    if check_stream:
        require(_ordinary() == record["stream"], "current original write stream changed")
    ctx, proof = record["ctx"], record["proof"]
    require(ctx.kv_cache is record["manager"] and ctx.adapter is record["adapter"] and not ctx._committed and
            ctx.history_block_ids is record["history_ids_owner"] and
            ctx.current_video_block_ids is record["current_ids_owner"] and
            (ctx.stage_first_block, ctx.reuse_history, int(ctx.adapter.num_computed_tokens),
             int(ctx.adapter.absolute_num_computed_tokens), ctx.kv_branch, ctx.commit_current) == proof[2:],
            "current original manager/adapter/provider/scalars changed")
    require(manager_geometry(ctx.kv_cache) == record["geometry"], "rank-local manager allocation geometry changed")
    state = ctx.kv_cache.history_staging_state
    require(state is record["stage_state"] and state.forward_owner is record["stage_owner"] and
            state.signature == record["stage_signature"] and state.adapter is not None and
            state.adapter() is record["adapter"] and state.ready is False,
            "original staging preparation/publication owner changed")
    require(all(getattr(ctx, META[name]) is t for name, t in record["refs"].items()),
            "original metadata provider changed")
    require(all(metadata_describe(t) == record["descriptions"][name] for name, t in record["refs"].items()),
            "original metadata descriptor/version changed in place")


def record_layer(ctx, index, value):
    record = getattr(ctx, "_recv_kv_sp8_record", None)
    if record is None:
        return value
    _current(record)
    require(sys._getframe(1).f_code is type(ctx).layer_inputs.__code__ and type(index) is int and
            0 <= index < 40 and index not in record["layers"] and id(value) not in record["layer_indices"],
            "one original layer provider per forward required")
    require(all(getattr(value, name) is t for name, t in record["refs"].items()), "layer metadata owner changed")
    require(value.key_pool is ctx.kv_cache._k_pools[index] and value.value_pool is ctx.kv_cache._v_pools[index] and
            all(actual is original for actual, original in
                zip((value.stage_key, value.stage_value), ctx.kv_cache.history_staging[index])),
            "cache roles must be exact original manager-owned complete tensors")
    geometry = record["geometry"]
    for name in ROLES:
        tensor = getattr(value, name)
        rows = geometry[2] * geometry[3] if name in ("key_pool", "value_pool") else geometry[6]
        require(tuple(tensor.shape) == (rows, 5, 128) and tuple(tensor.stride()) == (640, 128, 1) and
                tensor.dtype == _BOUND["torch"].bfloat16 and tensor.device == record["refs"]["video_slots"].device and
                tensor.is_contiguous() and tensor.storage_offset() == 0 and
                tensor.untyped_storage().nbytes() == rows * 640 * 2,
                "original full pool/stage descriptor required before any prefix write")
    record["layers"][index] = (value, tuple(describe(getattr(value, name)) for name in ROLES))
    record["layer_indices"][id(value)] = index
    return value


def mark_forward_succeeded(ctx):
    """Called only by the original full-forward staging publisher, never the leaf."""
    record = getattr(ctx, "_recv_kv_sp8_record", None)
    if record is None:
        return
    require(record is getattr(_STATE, "prepared", None) and record["ctx"] is ctx and
            record["completed"] == set(range(40)) and getattr(_STATE, "receive", None) is None and
            getattr(_STATE, "dispatched", None) is None and
            not getattr(_STATE, "poisoned", False), "incomplete SP8 forward cannot publish readiness")
    if not record["published"]:
        _current(record)
        record["published"] = True


def record_receive(attention, cache, received, query, key, value):
    if not is_active():
        return
    record = _STATE.prepared
    _current(record)
    require(sys._getframe(1).f_code is attention.forward.__func__.__code__ and
            getattr(attention, "_lingbot_paged_graph_runners", None) is None and
            attention.ulysses_world_size == 8 and attention.head_dim == 128 and
            getattr(attention, "_lingbot_native_bf16_qk_enabled", False) is True,
            "original ordinary native SP8 attention caller required")
    index = record["layer_indices"].get(id(cache))
    require(index is not None and record["layers"][index][0] is cache and
            index not in record["completed"] and getattr(_STATE, "receive", None) is None,
            "one original receive per layer required")
    require(tuple(received.shape) == (1, 4680, 3, 5, 128) and received.is_contiguous() and
            received.storage_offset() == 0 and received.untyped_storage().nbytes() == 17971200,
            "completed original all-to-all receive allocation changed")
    require(all(t.untyped_storage()._cdata == received.untyped_storage()._cdata for t in (query, key, value)),
            "original unbind must alias the returned receive")
    visible = tuple(t[0] for t in (query, key, value))
    require(all(tuple(t.shape) == (4680, 5, 128) and tuple(t.stride()) == (1920, 128, 1) and
                t.storage_offset() == offset for t, offset in zip(visible, (0, 640, 1280))),
            "original received public views changed")
    _STATE.receive = {"record": record, "layer": index, "cache": cache, "attention": attention,
                      "received": received, "public": visible, "descriptions": tuple(map(describe, visible)),
                      "policy": (attention._lingbot_npu_attention_backend,
                                 attention._lingbot_npu_attention_splits, attention._lingbot_npu_attention_strict)}


def try_ingress(query, key, value, k_act, v_act, kp, vp, block_size, slots, action_slots, table,
                qstart, lens, max_query, max_seq, scale, ks, vs, reuse, first, kvlen, backend, splits, strict):
    if not is_active():
        return None
    receive = getattr(_STATE, "receive", None)
    require(receive is not None, "opaque op lacks original completed receive association")
    record, cache = receive["record"], receive["cache"]
    _current(record, check_stream=True)
    module = sys.modules["vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention"]
    require(sys._getframe(1).f_code is module._paged_write_attn_impl.__code__, "original opaque implementation only")
    require(tuple(map(describe, (query, key, value))) == receive["descriptions"], "current received alias changed")
    require(all(actual is getattr(cache, name) for actual, name in zip((kp, vp, ks, vs), ROLES)) and
            tuple(map(describe, (kp, vp, ks, vs))) == record["layers"][receive["layer"]][1],
            "original complete pool/stage descriptor changed")
    actual = dict(video_slots=slots, action_slots=action_slots, block_table=table,
                  query_start_loc=qstart, seq_lens=lens)
    require(all(t is record["refs"][name] and metadata_describe(t) == record["descriptions"][name]
                for name, t in actual.items()), "original fresh metadata provider changed")
    ctx, proof = record["ctx"], record["proof"]
    require(k_act is None and v_act is None and block_size == 1560 and first == proof[2] and reuse is proof[3] and
            kvlen == ctx.kv_len and max_query == 4680 and max_seq == ctx.max_seq_len and scale == 128 ** -.5 and
            (backend, splits, strict) == receive["policy"], "original native FIA scalar policy changed")
    effective = backend or os.environ.get("VLLM_LINGBOT_NPU_ATTENTION_BACKEND", "native")
    require(effective == "fia_bsnd" or (effective == "native" and
            os.environ.get("VLLM_LINGBOT_NPU_INFER_ATTENTION", "0") == "1"), "original BF16 FIA only")
    torch = _BOUND["torch"]
    try:
        if not reuse and first:
            blocks = kvlen // 1560
            ids = table[0, :first].long()
            torch.index_select(kp.unflatten(0, (-1, 1560)), 0, ids,
                               out=ks[:kvlen].view(blocks, 1560, 5, 128)[:first])
            torch.index_select(vp.unflatten(0, (-1, 1560)), 0, ids,
                               out=vs[:kvlen].view(blocks, 1560, 5, 128)[:first])
        dense_query = _BOUND["ingress"](query, key, value, kp, vp, ks, vs, slots, table, first, reuse)
        # Keep the completed receive and new Q owner through the FIA call. On
        # failure RPC cleanup may clear traceback references before root kills8.
        _STATE.dispatched = (receive, dense_query, kp, vp, ks, vs, slots, table)
        fatal = _fatal_module()
        if fatal.FAULT_ENABLED:
            fatal.fault_once_after_valid_write("recv_ingress")
        record["completed"].add(receive["layer"])
        _STATE.receive = None
        return dense_query, ks[:kvlen], vs[:kvlen]
    except BaseException as error:
        poison(str(error))
        raise


def attention_succeeded():
    """Release the normal-path leaf hold only after successful native FIA."""
    require(not getattr(_STATE, "poisoned", False) and getattr(_STATE, "dispatched", None) is not None,
            "SP8 FIA completion lacks its retained ingress owner")
    _STATE.dispatched = None
