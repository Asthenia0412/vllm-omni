"""Default-OFF exact VAE packing policy, frozen separately from live serving.

Only seven stage/geometry pairs screened on three fresh rounds/four dies are
admitted. Unsupported shapes stay on the original native copy path; a failure
after admitted dispatch propagates and requires worker restart.
"""
from __future__ import annotations
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path

ELIGIBLE = frozenset({('halo',
            ((1, 96, 6, 482, 210), (58302720, 607320, 101220, 210, 1), 0),
            ((1, 96, 6, 482, 1), (58302720, 607320, 101220, 210, 1), 0),
            ((1, 96, 6, 482, 1), (277632, 2892, 482, 1, 1), 0),
            ((1, 96, 6, 482, 1), (58302720, 607320, 101220, 210, 1), 209),
            ((1, 96, 6, 482, 1), (277632, 2892, 482, 1, 1), 0)),
           ('halo',
            ((1, 192, 4, 122, 54), (5059584, 26352, 6588, 54, 1), 0),
            ((1, 192, 4, 122, 1), (5059584, 26352, 6588, 54, 1), 0),
            ((1, 192, 4, 122, 1), (93696, 488, 122, 1, 1), 0),
            ((1, 192, 4, 122, 1), (5059584, 26352, 6588, 54, 1), 53),
            ((1, 192, 4, 122, 1), (93696, 488, 122, 1, 1), 0)),
           ('halo',
            ((1, 192, 6, 242, 106), (29551104, 153912, 25652, 106, 1), 0),
            ((1, 192, 6, 242, 1), (29551104, 153912, 25652, 106, 1), 0),
            ((1, 192, 6, 242, 1), (278784, 1452, 242, 1, 1), 0),
            ((1, 192, 6, 242, 1), (29551104, 153912, 25652, 106, 1), 105),
            ((1, 192, 6, 242, 1), (278784, 1452, 242, 1, 1), 0)),
           ('halo',
            ((1, 384, 4, 122, 54), (10119168, 26352, 6588, 54, 1), 0),
            ((1, 384, 4, 122, 1), (10119168, 26352, 6588, 54, 1), 0),
            ((1, 384, 4, 122, 1), (187392, 488, 122, 1, 1), 0),
            ((1, 384, 4, 122, 1), (10119168, 26352, 6588, 54, 1), 53),
            ((1, 384, 4, 122, 1), (187392, 488, 122, 1, 1), 0)),
           ('interior',
            ((1, 16, 3, 62, 28), (83328, 5208, 1736, 28, 1), 0),
            ((1, 16, 2, 60, 26), (83328, 5208, 1736, 28, 1), 29),
            ((1, 16, 2, 60, 26), (49920, 3120, 1560, 26, 1), 0),
            ((1, 16, 1, 60, 26), (83328, 5208, 1736, 28, 1), 3501),
            ((1, 16, 1, 60, 26), (24960, 1560, 1560, 26, 1), 0)),
           ('interior',
            ((1, 192, 6, 242, 106), (29551104, 153912, 25652, 106, 1), 0),
            ((1, 192, 2, 240, 104), (29551104, 153912, 25652, 106, 1), 107),
            ((1, 192, 2, 240, 104), (9584640, 49920, 24960, 104, 1), 0),
            ((1, 192, 4, 240, 104), (29551104, 153912, 25652, 106, 1), 51411),
            ((1, 192, 4, 240, 104), (19169280, 99840, 24960, 104, 1), 0)),
           ('interior',
            ((1, 384, 3, 62, 28), (1999872, 5208, 1736, 28, 1), 0),
            ((1, 384, 2, 60, 26), (1999872, 5208, 1736, 28, 1), 29),
            ((1, 384, 2, 60, 26), (1198080, 3120, 1560, 26, 1), 0),
            ((1, 384, 1, 60, 26), (1999872, 5208, 1736, 28, 1), 3501),
            ((1, 384, 1, 60, 26), (599040, 1560, 1560, 26, 1), 0))})

SCREENED_DRIVER_SHA256 = "a00aef6d1a1f0af77cae56465bfc5ca2dd2d43480e2c73ad47d1e81a4bfe7ce0"
SCREENED_RAW_KERNEL_SHA256 = "413455ad672d9a64a91539856931e51401d8ae519a96c5bdf0674cbc86fef300"
_loaded_driver = None
_loaded_descriptor = None
_PHYSICAL_FIELDS = ("npu_format", "origin_format", "base_sizes", "base_strides",
                    "physical_sizes", "shape", "stride", "storage_offset", "storage_bytes")


def _physical_signature(description):
    """Portable physical whitelist: addresses vary between ranks and calls."""
    if not isinstance(description, dict):
        description = dict(zip(_PHYSICAL_FIELDS, description[:9]))
    return tuple(tuple(description[name]) if name in
                 ("base_sizes", "base_strides", "physical_sizes", "shape", "stride")
                 else int(description[name]) for name in _PHYSICAL_FIELDS)


def _load_sp8_admission(path):
    manifest_path = Path(path).expanduser().resolve(strict=True)
    admission = json.loads(manifest_path.read_text())
    expected = dict(version=1, kind="sp8_vae_exact_pack", world_size=8, status="admitted",
                    driver_sha256=SCREENED_DRIVER_SHA256, raw_kernel_sha256=SCREENED_RAW_KERNEL_SHA256)
    if any(admission.get(k) != v for k, v in expected.items()):
        raise ValueError("SP8 VAE admission must reuse the screened original driver and raw kernel")
    gates = admission.get("gates", {})
    if (any(gates.get(k) is not True for k in
            ("actual_capture", "all8ranks", "complete_call_bitexact", "descriptor_stable_after_drain"))
            or type(gates.get("fresh_rounds")) is not int or gates["fresh_rounds"] < 3):
        raise ValueError("SP8 VAE admission requires actual descriptors and three fresh eight-rank screens")
    evidence = admission.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("SP8 VAE admission requires pinned capture/screen evidence")
    for item in evidence:
        target = Path(item["path"]).expanduser()
        if not target.is_absolute():
            target = manifest_path.parent / target
        if hashlib.sha256(target.resolve(strict=True).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("SP8 VAE capture/screen evidence changed after admission")
    library = admission.get("descriptor_library", {})
    descriptor_path = Path(library["path"]).expanduser()
    if not descriptor_path.is_absolute():
        descriptor_path = manifest_path.parent / descriptor_path
    descriptor_path = descriptor_path.resolve(strict=True)
    if hashlib.sha256(descriptor_path.read_bytes()).hexdigest() != library["sha256"]:
        raise ValueError("SP8 VAE descriptor library differs from captured ABI")
    whitelist = {}
    for pair in admission.get("descriptor_pairs", []):
        if pair["stage"] not in ("interior", "halo") or len(pair["operands"]) != 5:
            raise ValueError("SP8 VAE whitelist requires five captured operands per pair")
        physical = tuple(_physical_signature(x) for x in pair["operands"])
        logical = (pair["stage"],) + tuple((x[5], x[6], x[7]) for x in physical)
        whitelist.setdefault(logical, set()).add(physical)
    if not whitelist:
        raise ValueError("SP8 VAE whitelist is empty; unscreened geometry is never admitted")
    admission.update(manifest_path=str(manifest_path), descriptor_path=str(descriptor_path),
                     whitelist=whitelist)
    return admission


def enabled_from_sources(model_config):
    if "lingbot_npu_vae_exact_pack" in model_config:
        value = model_config["lingbot_npu_vae_exact_pack"]
        if type(value) is not bool:
            raise ValueError("model_config.lingbot_npu_vae_exact_pack must be a bool")
        return value
    value = os.environ.get("LINGBOT_NPU_VAE_EXACT_PACK", "0")
    if value not in ["0", "1"]:
        raise ValueError("LINGBOT_NPU_VAE_EXACT_PACK must be0 or1")
    return value == "1"


def _descriptor(tensor):
    return tuple(tensor.shape), tuple(tensor.stride()), int(tensor.storage_offset())


def _decoder_conv(name, module):
    return type(module).__name__ == "WanDistCausalConv3d" and (
        name == "decoder.conv_in" or (name.startswith("decoder.") and ".resnets." in name
                                      and name.rsplit(".", 1)[-1] in ["conv1", "conv2"]))


class ExactPackingPolicy:
    def __init__(self, backend, *, profile=False, physical_whitelist=None):
        self.backend = backend
        self.profile = profile
        self.physical_whitelist = physical_whitelist
        self.eligible = ELIGIBLE if physical_whitelist is None else physical_whitelist.keys()

    def try_pair(self, stage, base, dst_a, src_a, dst_b, src_b):
        values = [base, dst_a, src_a, dst_b, src_b]
        if stage not in ["interior", "halo"] or any(x is None for x in values):
            return False
        if any(str(x.dtype) != "torch.bfloat16" or x.device.type != "npu" for x in values):
            return False
        key = (stage,) + tuple(_descriptor(x) for x in values)
        if key not in self.eligible:
            return False
        if self.physical_whitelist is not None:
            actual = tuple(_physical_signature(self.backend.describe(x)) for x in values)
            if actual not in self.physical_whitelist[key]:
                return False
        # Every pointer, actual physical format/extent, alias, destination
        # rectangle and lifetime check occurs in the guarded C++ call before
        # mutation. NEVER recover an admitted-call exception by copying again.
        scope = self.backend.scope("lingbot.vae_exact_pack." + stage) if self.profile else nullcontext()
        if self.physical_whitelist is None:
            with scope:
                self.backend.copy_pair(*values)
        else:
            try:
                with scope:
                    self.backend.copy_pair(*values)
            except BaseException as error:
                try:
                    from vllm_omni.diffusion.layers.lingbot_sp8_fatal import report_failure
                    report_failure("vae_pack", str(error), tuple(values) + (self.backend,))
                except BaseException:
                    pass
                raise
        return True


class _Backend:
    def __init__(self, library, *, _sp8_admission=None):
        global _loaded_driver, _loaded_descriptor
        import torch
        import torch_npu
        if not torch_npu.npu.is_initialized():
            raise RuntimeError("Configure packing only after existing NPU initialization")
        internal = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
        jit_false = bool(torch_npu.npu.is_jit_compile_false())
        if internal not in [b"enable", "enable"] or not jit_false:
            raise RuntimeError("Packing requires serving internal-format enable/effective JITFalse; policy is never changed")
        path = Path(library).expanduser().resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SCREENED_DRIVER_SHA256:
            raise ValueError("Packing driver differs from the three-round screened binary")
        if _loaded_driver is None:
            torch.ops.load_library(str(path)); _loaded_driver = (str(path), digest)
        elif _loaded_driver[1] != digest:
            raise RuntimeError("A different packing backend already owns this process; restart required")
        # Verify the actual loaded dependency rather than assuming LD_LIBRARY_PATH
        # selected the raw kernel archived alongside this host driver.
        import ctypes
        class DlInfo(ctypes.Structure):
            _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                        ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]
        self._library_owner = ctypes.CDLL(str(path))
        dladdr = ctypes.CDLL(None).dladdr
        dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        dladdr.restype = ctypes.c_int
        info = DlInfo()
        entry = ctypes.cast(self._library_owner.lingbot_vae_exact_pack_launch, ctypes.c_void_p)
        if not dladdr(entry, ctypes.byref(info)) or not info.filename:
            raise RuntimeError("Cannot identify actual loaded packing kernel")
        raw_path = Path(info.filename.decode()).resolve(strict=True)
        raw_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if raw_digest != SCREENED_RAW_KERNEL_SHA256:
            raise RuntimeError("Loaded raw packing kernel differs from screened binary; restart required")
        self.copy_pair = torch.ops.lingbot_vae_exact_pack.copy_pair
        self.scope = torch.profiler.record_function
        self.summary = dict(driver_path=str(path), driver_sha256=digest,
            actual_raw_kernel_path=str(raw_path), actual_raw_kernel_sha256=raw_digest,
            raw_internal_format=internal.decode() if isinstance(internal, bytes) else str(internal),
            effective_is_jit_compile_false=jit_false, policy_mutated=False)
        if _sp8_admission is not None:
            from vllm_omni.diffusion.layers.lingbot_sp8_fatal import ensure_protocol
            ensure_protocol()
            descriptor = (_sp8_admission["descriptor_path"],
                          _sp8_admission["descriptor_library"]["sha256"])
            if _loaded_descriptor is None:
                torch.ops.load_library(descriptor[0])
                _loaded_descriptor = descriptor
            elif _loaded_descriptor != descriptor:
                raise RuntimeError("A different descriptor ABI owns this process; restart required")
            self.describe = torch.ops.lingbot_storage_descriptor.describe
            self.summary.update(admission_manifest=_sp8_admission["manifest_path"],
                                descriptor_library=descriptor[0], descriptor_sha256=descriptor[1],
                                sp_required=8)


def configure_exact_packing(vae, model_config, *, device, logger=None):
    recovery = model_config.get("lingbot_npu_recovery_vae_pack", False)
    if type(recovery) is not bool:
        raise ValueError("lingbot_npu_recovery_vae_pack must be bool")
    if recovery and device.type == "npu":
        if model_config.get("lingbot_npu_vae_exact_pack", False):
            raise ValueError("Choose old screened packing or reconstructed packing")
        from vllm_omni.diffusion.layers.lingbot_recovery import ExactPacking
        owners = [(name, module) for name, module in vae.named_modules() if _decoder_conv(name, module)]
        if not owners:
            raise ValueError("Reconstructed packing requires spatial decoder owners")
        import torch.distributed as dist
        if {dist.get_world_size(module.group) for _, module in owners} != {8}:
            raise ValueError("Reconstructed packing is limited to SP8")
        policy = ExactPacking()
        for _, module in owners:
            module._lingbot_exact_pack_policy = policy
            module._lingbot_exact_pack_enabled = True
        return dict(enabled=True, reconstructed=True, historical_admission_inherited=False,
                    modules=[name for name, _ in owners])
    enabled = enabled_from_sources(model_config)
    if not enabled or device.type != "npu":
        return dict(enabled=False, platform=device.type, eligible_geometry_pairs=7)
    library = model_config.get("lingbot_npu_vae_exact_pack_library") or os.environ.get("LINGBOT_NPU_VAE_EXACT_PACK_DRIVER_SO")
    if not isinstance(library, str) or not library:
        raise ValueError("Enabled packing requires an explicit screened driver library path")
    owners = []
    for name, module in vae.named_modules():
        if _decoder_conv(name, module):
            owners.append((name, module))
    if not owners:
        raise ValueError("Packing requires the existing spatially sharded Wan decoder")
    manifest = model_config.get("lingbot_npu_vae_exact_pack_manifest") or os.environ.get("LINGBOT_NPU_VAE_EXACT_PACK_MANIFEST")
    admission = _load_sp8_admission(manifest) if manifest else None
    # Actual spatial group membership determines the policy. A SP4 manifest
    # cannot enable SP8, and the old SP4 whitelist is never reused on SP8.
    world = 4
    if any(hasattr(module, "group") for _, module in owners):
        import torch.distributed as dist
        worlds = {dist.get_world_size(module.group) for _, module in owners}
        if len(worlds) != 1:
            raise ValueError("VAE exact packing requires one uniform spatial group")
        world = worlds.pop()
    if world not in (4, 8) or (world == 8) != (admission is not None):
        raise ValueError("SP8 VAE packing requires its captured descriptor admission; SP4 keeps its original policy")
    backend = _Backend(library) if admission is None else _Backend(library, _sp8_admission=admission)
    whitelist = None if admission is None else admission["whitelist"]
    policy = ExactPackingPolicy(backend, profile=os.environ.get("LINGBOT_PROFILE_STAGE_RANGES") == "1",
                                physical_whitelist=whitelist)
    for _, module in owners:
        module._lingbot_exact_pack_policy = policy
        module._lingbot_exact_pack_enabled = True
    summary = dict(enabled=True, modules=[name for name, _ in owners], eligible_geometry_pairs=len(policy.eligible),
        missing_cache_or_unscreened_geometry="original native path", failure="admitted-call error propagates; restart required",
        default_current_stream_only=True, **backend.summary)
    if logger is not None:
        logger.info("LingBot exact VAE packing: %s", summary)
    return summary


def try_exact_packing(owner, stage, base, dst_a, src_a, dst_b, src_b):
    policy = getattr(owner, "_lingbot_exact_pack_policy", None)
    return bool(policy is not None and policy.try_pair(stage, base, dst_a, src_a, dst_b, src_b))
