"""Default-OFF byte-preserving Q/K/V SP send preparation.

The post-drain storage guard/raw kernel pair passed 48 complete-call and
original HCCL byte checks over three fresh four-rank rounds. This module
changes only send packing. Native projection/full5120 QK norm/RoPE, HCCL
and receive transforms stay with the original model. Each call allocates an
independent output. BF16 model parity/160 actual launches/E2E remain separate
admission gates; no preset or quality promotion follows from a leaf screen.

SP8 uses its independent namespace and screened physical source whitelist.
Ordinary unsupported inputs use native packing before dispatch. Ownership,
post-drain metadata and dispatched-kernel failures remain errors and never
retry with native packing after custom work may have written.
"""
from collections.abc import Mapping
import ctypes
import hashlib
import json
import os
from pathlib import Path

SCREENED_DRIVER_SHA256 = '1ba1248dddc08197362fa7fd646de3a26ce78b0807e5ee31b40f26982b79a3fc'
SCREENED_RAW_KERNEL_SHA256 = 'ebe60d7e8b3b8b250622e69eb4bd63c77ca4a9ce759f193226f7b9fbee4d099f'
_loaded_driver = None
_loaded_sp8_driver = None
_loaded_sp8_descriptor = None
_PHYSICAL_FIELDS = ("npu_format", "origin_format", "base_sizes", "base_strides",
                    "physical_sizes", "shape", "stride", "storage_offset", "storage_bytes")
_DESCRIPTOR_FIELDS = _PHYSICAL_FIELDS + ("storage_data_ptr", "data_ptr")
_SEQUENCE_FIELDS = frozenset(("base_sizes", "base_strides", "physical_sizes", "shape", "stride"))


def _description(value):
    """Decode the explicitly pinned read-only eleven-field descriptor ABI."""
    if isinstance(value, Mapping):
        result = {name: value[name] for name in _DESCRIPTOR_FIELDS}
    else:
        if len(value) != len(_DESCRIPTOR_FIELDS):
            raise ValueError("SP8 QKV descriptor ABI returned an unexpected field count")
        result = dict(zip(_DESCRIPTOR_FIELDS, value))
    for name, item in result.items():
        values = item if name in _SEQUENCE_FIELDS else (item,)
        if any(type(number) is not int for number in values):
            raise ValueError("SP8 QKV descriptor contains invalid integer metadata")
        if name in _SEQUENCE_FIELDS:
            result[name] = tuple(item)
    return result


def _physical_signature(value):
    if isinstance(value, Mapping):
        values = [value[name] for name in _PHYSICAL_FIELDS]
    else:
        if len(value) < len(_PHYSICAL_FIELDS):
            raise ValueError("SP8 QKV physical signature has an unexpected field count")
        values = value[:len(_PHYSICAL_FIELDS)]
    return tuple(tuple(item) if name in _SEQUENCE_FIELDS else int(item)
                 for name, item in zip(_PHYSICAL_FIELDS, values))


def _validate_owner(value, description):
    """Reject inconsistent ownership before classifying an unsupported layout."""
    shape, stride = description["shape"], description["stride"]
    base, base_stride = description["base_sizes"], description["base_strides"]
    physical = description["physical_sizes"]
    if (not shape or len(shape) != len(stride) or len(base) != len(base_stride)
            or not base or not physical or any(s <= 0 for s in shape + base + physical)
            or any(s < 0 for s in stride + base_stride)):
        raise ValueError("SP8 QKV descriptor has an invalid extent or stride")
    offset, size = description["storage_offset"], description["storage_bytes"]
    owner, pointer = description["storage_data_ptr"], description["data_ptr"]
    if (offset < 0 or size <= 0 or size % 2 or owner <= 0
            or owner + size > 1 << 48 or pointer != owner + offset * 2):
        raise ValueError("SP8 QKV descriptor has an invalid allocation owner or effective pointer")
    last = offset + sum((extent - 1) * step for extent, step in zip(shape, stride))
    physical_elements = 1
    for extent in physical:
        physical_elements *= extent
    if (2 * (last + 1) > size or 2 * physical_elements > size
            or tuple(value.shape) != shape or tuple(value.stride()) != stride
            or value.storage_offset() != offset):
        raise ValueError("SP8 QKV descriptor disagrees with its tensor or owned allocation")
    storage = value.untyped_storage()
    if (storage.nbytes() != size or storage.data_ptr() != owner
            or value.data_ptr() != pointer):
        raise ValueError("SP8 QKV descriptor disagrees with the original tensor storage")


def _load_sp8_admission(path, library):
    """Read a leaf-screen seal without importing the framework or touching a device."""
    if not isinstance(path, str) or not path:
        raise ValueError("SP8 QKV packing requires an explicit screened admission manifest")
    manifest_path = Path(path).expanduser().resolve(strict=True)
    admission = json.loads(manifest_path.read_text())
    expected = dict(version=1, kind="sp8_qkv_send_pack", world_size=8, status="admitted",
                    namespace="lingbot_qkv_sp8_pack", raw_launch_symbol="lingbot_qkv_sp8_pack_launch",
                    source_shape=[1, 585, 40, 128], send_shape=[8, 585, 3, 1, 5, 128])
    if any(admission.get(k) != v for k, v in expected.items()):
        raise ValueError("SP8 QKV admission does not describe the canonical screened binding")
    gates = admission.get("gates", {})
    if (any(gates.get(k) is not True for k in
            ("actual_capture", "all8ranks", "complete_call_bitexact", "original_hccl_bitexact"))
            or type(gates.get("fresh_rounds")) is not int or gates["fresh_rounds"] < 3):
        raise ValueError("SP8 QKV admission requires captured bytes and three fresh eight-rank screens")
    for name in ("driver_sha256", "raw_kernel_sha256"):
        value = admission.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("SP8 QKV admission contains an invalid binary pin")
    driver = Path(library).expanduser().resolve(strict=True)
    if _sha256(driver) != admission["driver_sha256"]:
        raise ValueError("SP8 QKV driver differs from the screened admission")
    evidence = admission.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("SP8 QKV admission requires pinned capture/screen evidence")
    for item in evidence:
        target = Path(item["path"]).expanduser()
        if not target.is_absolute():
            target = manifest_path.parent / target
        if _sha256(target.resolve(strict=True)) != item["sha256"]:
            raise ValueError("SP8 QKV capture/screen evidence changed after admission")
    descriptor = admission.get("descriptor_library", {})
    if (not isinstance(descriptor.get("path"), str)
            or not isinstance(descriptor.get("sha256"), str)
            or len(descriptor["sha256"]) != 64):
        raise ValueError("SP8 QKV admission requires the pinned read-only descriptor ABI")
    descriptor_path = Path(descriptor["path"]).expanduser()
    if not descriptor_path.is_absolute():
        descriptor_path = manifest_path.parent / descriptor_path
    descriptor_path = descriptor_path.resolve(strict=True)
    if _sha256(descriptor_path) != descriptor["sha256"]:
        raise ValueError("SP8 QKV descriptor library differs from the captured ABI")
    whitelist = set()
    for record in admission.get("source_descriptors", []):
        if not isinstance(record, list) or len(record) != 3:
            raise ValueError("SP8 QKV whitelist requires actual Q/K/V descriptor triples")
        physical = tuple(_physical_signature(value) for value in record)
        if any(value[5] != (1, 585, 40, 128) for value in physical):
            raise ValueError("SP8 QKV whitelist contains an unscreened logical shape")
        whitelist.add(physical)
    if not whitelist:
        raise ValueError("SP8 QKV physical whitelist is empty")
    admission.update(manifest_path=str(manifest_path), descriptor_path=str(descriptor_path),
                     source_whitelist=whitelist)
    return admission


def _flag(value):
    if type(value) is bool:
        return value
    if type(value) is str and value in ("0", "1"):
        return value == "1"
    raise ValueError("QKV SP packing flag must be bool (environment 0/1)")


def enabled_from_sources(model_config=None, environ=None):
    values = os.environ if environ is None else environ
    enabled = _flag(values.get("LINGBOT_NPU_QKV_SP_PACK", "0"))
    if isinstance(model_config, Mapping) and "lingbot_npu_qkv_sp_pack" in model_config:
        enabled = model_config["lingbot_npu_qkv_sp_pack"]
        if type(enabled) is not bool:
            raise ValueError("Model QKV SP packing flag must be bool")
    return enabled


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Backend:
    def __init__(self, library, *, _sp8_admission=None):
        global _loaded_driver, _loaded_sp8_driver, _loaded_sp8_descriptor
        driver_pin = SCREENED_DRIVER_SHA256 if _sp8_admission is None else _sp8_admission["driver_sha256"]
        raw_pin = SCREENED_RAW_KERNEL_SHA256 if _sp8_admission is None else _sp8_admission["raw_kernel_sha256"]
        namespace = "lingbot_qkv_sp_pack" if _sp8_admission is None else "lingbot_qkv_sp8_pack"
        symbol = namespace + "_launch"
        # Fail before framework/device imports while the new dependency is
        # pending. Preparation is not an authorization to reuse the old SO.
        if any(len(x) != 64 or any(c not in "0123456789abcdef" for c in x)
               for x in (driver_pin, raw_pin)):
            raise RuntimeError("QKV SP post-drain dependency has not been sealed after rescreen")
        import torch
        import torch_npu
        if not torch_npu.npu.is_initialized():
            raise RuntimeError("Configure QKV SP packing after existing NPU initialization")
        internal = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
        jit_false = torch_npu.npu.is_jit_compile_false()
        if internal not in (b"enable", "enable") or jit_false is not True:
            raise RuntimeError("QKV SP packing requires serving internal-format enable/effective JITFalse; policy is never changed")
        path = Path(library).expanduser().resolve(strict=True)
        digest = _sha256(path)
        if digest != driver_pin:
            raise ValueError("QKV SP driver differs from the post-drain screened binary")
        loaded = _loaded_driver if _sp8_admission is None else _loaded_sp8_driver
        if loaded is None:
            torch.ops.load_library(str(path))
            if _sp8_admission is None:
                _loaded_driver = (str(path), digest)
            else:
                _loaded_sp8_driver = (str(path), digest)
        elif loaded != (str(path), digest):
            raise RuntimeError("A different QKV SP driver already owns this process; restart required")
        # Retain the handle and identify the actual loaded raw dependency.
        self._library_owner = ctypes.CDLL(str(path))
        class DlInfo(ctypes.Structure):
            _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                        ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]
        dladdr = ctypes.CDLL(None).dladdr
        dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        dladdr.restype = ctypes.c_int
        info = DlInfo()
        entry = ctypes.cast(getattr(self._library_owner, symbol), ctypes.c_void_p)
        if not dladdr(entry, ctypes.byref(info)) or not info.filename:
            raise RuntimeError("Cannot identify actual loaded QKV SP kernel")
        raw_path = Path(info.filename.decode()).resolve(strict=True)
        raw_digest = _sha256(raw_path)
        if raw_digest != raw_pin:
            raise RuntimeError("Loaded QKV SP raw kernel differs from screened binary; restart required")
        self._pack = getattr(torch.ops, namespace).pack
        self._scope = torch.profiler.record_function
        self._profile = os.environ.get("LINGBOT_PROFILE_STAGE_RANGES") == "1"
        self.summary = dict(driver_path=str(path), driver_sha256=digest,
                            actual_raw_kernel_path=str(raw_path), actual_raw_kernel_sha256=raw_digest,
                            raw_internal_format=internal.decode() if isinstance(internal, bytes) else internal,
                            effective_is_jit_compile_false=jit_false, policy_mutated=False)
        if _sp8_admission is not None:
            from vllm_omni.diffusion.layers.lingbot_sp8_fatal import ensure_protocol
            ensure_protocol()
            descriptor = (_sp8_admission["descriptor_path"], _sp8_admission["descriptor_library"]["sha256"])
            if _sha256(Path(descriptor[0])) != descriptor[1]:
                raise RuntimeError("SP8 QKV descriptor ABI changed before configuration")
            if _loaded_sp8_descriptor is None:
                torch.ops.load_library(descriptor[0])
                _loaded_sp8_descriptor = descriptor
            elif _loaded_sp8_descriptor != descriptor:
                raise RuntimeError("A different SP8 QKV descriptor ABI owns this process; restart required")
            self.describe = torch.ops.lingbot_storage_descriptor.describe
            self._source_whitelist = _sp8_admission["source_whitelist"]
            self._torch, self._npu = torch, torch_npu.npu
            self.summary.update(namespace=namespace, admission_manifest=_sp8_admission["manifest_path"],
                                descriptor_library=descriptor[0], descriptor_sha256=descriptor[1],
                                physical_source_triples=len(self._source_whitelist),
                                ordinary_unsupported="native before dispatch", post_dispatch_fallback=False)

    def can_pack(self, query, key, value):
        """SP8-only eligibility; no launch, output allocation, copy or policy change."""
        if (self._torch.compiler.is_compiling() or self._npu.is_current_stream_capturing()
                or self._npu.current_stream(query.device) != self._npu.default_stream(query.device)):
            return False
        descriptions = tuple(_description(self.describe(source)) for source in (query, key, value))
        for source, description in zip((query, key, value), descriptions):
            _validate_owner(source, description)
        # Addresses vary across ranks and calls. The complete allocation and
        # original physical format remain part of the captured whitelist.
        return tuple(_physical_signature(d) for d in descriptions) in self._source_whitelist

    def pack(self, query, key, value):
        if self._profile:
            with self._scope("lingbot.qkv_sp_send_pack"):
                return self._pack(query, key, value)
        return self._pack(query, key, value)


def configure_qkv_sp_send_pack(transformer, model_config=None, *, device,
                               environ=None, logger=None, _backend_factory=None):
    values = os.environ if environ is None else environ
    enabled = enabled_from_sources(model_config, values)
    active = enabled and getattr(device, "type", None) == "npu"
    modules = [block.self_attn for block in transformer.blocks]
    world = getattr(modules[0], "ulysses_world_size", None) if modules else None
    sequence = {4: 1170, 8: 585}.get(world)
    policy = dict(enabled=enabled, active_on_this_platform=active,
                  default_enabled=False, dtype="bfloat16", tp_required=1,
                  sp_required=world, source_shape=[1, sequence, 40, 128],
                  send_shape=[world, sequence, 3, 1, 40 // world if world in (4, 8) else None, 128],
                  destination_lifetime="fresh owned output per call; no shared pool",
                  admission=("SP8 screened physical whitelist; ordinary unsupported native before dispatch"
                             if world == 8 else "strict screened storage guards; no fallback"),
                  jit_policy_mutated=False)
    backend = None
    if active:
        if len(modules) != 40:
            raise ValueError("QKV SP packing admits exactly 40 self-attention layers")
        if world not in (4, 8):
            raise ValueError("QKV SP packing admits only canonical SP4 or SP8")
        for module in modules:
            if (module.ulysses_world_size, module.num_local_heads, module.head_dim,
                    module.tp_inner_dim) != (world, 40, 128, 5120):
                raise ValueError("QKV SP packing requires uniform canonical SP4/SP8 TP1/40x128 self attention")
        library = (model_config.get("lingbot_npu_qkv_sp_pack_library")
                   if isinstance(model_config, Mapping) else None)
        library = library or values.get("LINGBOT_NPU_QKV_SP_PACK_DRIVER_SO")
        if not isinstance(library, str) or not library:
            raise ValueError("Enabled QKV SP packing requires the explicitly screened driver path")
        admission = None
        if world == 8:
            manifest = (model_config.get("lingbot_npu_qkv_sp_pack_manifest")
                        if isinstance(model_config, Mapping) else None)
            manifest = manifest or values.get("LINGBOT_NPU_QKV_SP_PACK_MANIFEST")
            admission = _load_sp8_admission(manifest, library)
        if _backend_factory is not None:
            backend = _backend_factory(library)
        else:
            backend = _Backend(library, _sp8_admission=admission)
        policy.update(backend.summary)
    for module in modules:
        module._lingbot_npu_qkv_sp_pack_enabled = active
        module._lingbot_npu_qkv_sp_pack_backend = backend
    transformer._lingbot_qkv_sp_send_pack_policy = policy
    policy["configured_self_attention_layers"] = len(modules)
    if logger is not None:
        logger.info("LingBot QKV SP send packing runtime policy: %s", policy)
    return policy


def prepare_qkv_sp_send(module, query, key, value):
    """Return owned prepared bytes plus a shape-only view, never read as data."""
    if not getattr(module, "_lingbot_npu_qkv_sp_pack_enabled", False):
        raise RuntimeError("QKV SP preparation was called while disabled")
    sequence = {4: 1170, 8: 585}.get(module.ulysses_world_size)
    if sequence is None:
        raise ValueError("QKV SP preparation requires SP4 or SP8")
    backend = getattr(module, "_lingbot_npu_qkv_sp_pack_backend", None)
    if module.ulysses_world_size == 8:
        if backend is None:
            raise RuntimeError("Enabled QKV SP preparation has no admitted backend")
        supported = all(tuple(source.shape) == (1, 585, 40, 128)
                        and str(source.dtype) == "torch.bfloat16"
                        and getattr(source.device, "type", None) == "npu"
                        and source.device == query.device for source in (query, key, value))
        if supported:
            supported = (tuple(query.stride()) == (2995200, 5120, 128, 1)
                         and tuple(key.stride()) == (2995200, 5120, 128, 1)
                         and tuple(value.stride()) == (8985600, 15360, 128, 1)
                         and backend.can_pack(query, key, value))
        if not supported:
            import torch
            return torch.stack((query, key, value), dim=2), None
    for source in (query, key, value):
        if tuple(source.shape) != (1, sequence, 40, 128) or str(source.dtype) != "torch.bfloat16":
            raise ValueError("QKV SP preparation requires canonical original BF16 Q/K/V")
        if getattr(source.device, "type", None) != "npu" or source.device != query.device:
            raise ValueError("QKV SP preparation requires the same NPU device")
    if backend is None:
        raise RuntimeError("Enabled QKV SP preparation has no admitted backend")
    if module.ulysses_world_size == 8:
        try:
            send = backend.pack(query, key, value)
        except BaseException as error:
            try:
                from vllm_omni.diffusion.layers.lingbot_sp8_fatal import report_failure
                report_failure("send_pack", str(error), (query, key, value, backend))
            except BaseException:
                # Diagnostic reporting must preserve the original dispatch
                # failure. The configured helper retains owners and reports
                # its own filesystem failures to the supervisor on stderr.
                pass
            raise
    else:
        send = backend.pack(query, key, value)
    # The comm function only consumes this view's dimension/shape/dtype/device;
    # _prepared_send bypasses its original reshape/transpose/contiguous path.
    metadata = query.unsqueeze(2).expand(-1, -1, 3, -1, -1)
    return metadata, send
