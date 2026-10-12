# SPDX-License-Identifier: Apache-2.0
"""Explicit, default-off reconstructed operators; no inherited admission.

Only load source-pinned builds. This namespace deliberately differs from the
lost experiment libraries. Old manifests and old flags keep their old meaning.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading

_LOCK = threading.RLock()
_BOUND = None
_QUARANTINE = []


def _check(value, message):
    if not value:
        raise RuntimeError("LingBot recovery: " + message)


def _pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load():
    global _BOUND
    with _LOCK:
        policy = (os.environ.get("LINGBOT_RECOVERY_MANIFEST"),
                  os.environ.get("LINGBOT_RECOVERY_MANIFEST_SHA256"))
        if _BOUND is not None:
            _check(_BOUND.policy == policy and not _BOUND.poisoned, "provider changed/poisoned; restart eight workers")
            return _BOUND
        path = Path(policy[0] or "")
        _check(path.is_absolute() and _pin(path)["sha256"] == policy[1], "explicit manifest SHA required")
        manifest = json.loads(path.read_text())
        _check(manifest.get("schema") == "lingbot_recovery_build_v1"
               and manifest.get("registration_passed") is True
               and manifest.get("historical_admission_inherited") is False, "unsupported build manifest")
        for pin in [manifest["source"], manifest["library"], manifest["kernels"],
                    *manifest["kernel_sources"], *manifest["runtime_sources"], *manifest["abi_headers"]]:
            _check(_pin(pin["path"]) == pin, "changed source/library/header: " + pin["path"])
        import torch
        import torch_npu
        _check(torch.__version__ == manifest["torch"] and torch_npu.__version__ == manifest["torch_npu"],
               "framework ABI changed")
        _check(torch_npu.npu.is_initialized() and torch_npu.npu.is_jit_compile_false(), "existing JITFalse NPU runtime required")
        _check(torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") in (b"enable", "enable"), "internal format policy changed")
        _check(torch_npu.npu.get_device_name() == "Ascend910_9392", "arch22/910_9392 only")
        from . import lingbot_sp8_fatal as fatal
        _check(fatal.ensure_protocol(), "launch-owned failure channel required")
        torch.ops.load_library(manifest["library"]["path"])
        import ctypes
        class DlInfo(ctypes.Structure):
            _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                        ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]
        dladdr = ctypes.CDLL(None).dladdr
        dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        dladdr.restype = ctypes.c_int
        info = DlInfo()
        address = torch.ops.lingbot_recovery_v1.linked_copy_address()
        _check(dladdr(ctypes.c_void_p(address), ctypes.byref(info)) and info.filename,
               "cannot identify actual kernel dependency")
        _check(_pin(info.filename.decode()) == manifest["kernels"], "kernel dependency interposed")
        _BOUND = Provider(torch, torch_npu, fatal, manifest, policy)
        return _BOUND


class Provider:
    def __init__(self, torch, backend, fatal, manifest, policy):
        self.torch, self.backend, self.fatal = torch, backend, fatal
        self.manifest, self.policy = manifest, policy
        self.ops = torch.ops.lingbot_recovery_v1
        self.describe = self.ops.describe
        self.poisoned = False

    def ordinary(self, tensor):
        _check(not self.poisoned, "worker poisoned; restart all eight workers")
        return (tensor.device.type == "npu" and tensor.dtype == self.torch.bfloat16
                and not self.torch.is_grad_enabled() and self.torch.is_inference_mode_enabled()
                and not self.torch.npu.is_current_stream_capturing()
                and self.torch.npu.current_stream(tensor.device) == self.torch.npu.default_stream(tensor.device))

    def call(self, name, *args):
        _check(not self.poisoned, "worker poisoned")
        try:
            return getattr(self.ops, name)(*args)
        except BaseException as error:
            self.poisoned = True
            owners = (self, args, error)
            _QUARANTINE.append(owners)
            self.fatal.report_failure("recovery_" + name, repr(error), owners)
            raise


def configured(feature):
    from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
    config = get_current_diffusion_config_or_none()
    values = getattr(config, "model_config", None) or {}
    key = "lingbot_npu_recovery_" + feature
    enabled = values.get(key, False)
    _check(type(enabled) is bool, key + " must be bool")
    if not enabled:
        return False
    _check(getattr(config, "model_class_name", None) == "LingBotWorldCausalDMDPipeline", "LingBot causal model only")
    parallel = config.parallel_config
    _check((parallel.tensor_parallel_size, parallel.sequence_parallel_size,
            parallel.ulysses_degree, parallel.ring_degree) == (1, 8, 8, 1), "SP8/TP1/ring1 only")
    _check(bool(config.enforce_eager), "eager execution required")
    _check(not values.get("lingbot_npu_qk_single_host_admission_v1", False)
           and not values.get("lingbot_npu_cross_combined_guard_cpp_v1", False), "old and rebuilt providers are mutually exclusive")
    return True


class Send:
    def __init__(self, provider, metadata, send, owners):
        self.provider, self.metadata, self.send, self.owners = provider, metadata, send, owners
        self.used = False
        self.stream = provider.torch.npu.current_stream(send.device)

    def exchange(self, collective, group):
        _check(not self.used and not self.provider.poisoned, "send lease already consumed/poisoned")
        _check(self.provider.torch.npu.current_stream(self.send.device) == self.stream, "send stream changed")
        self.used = True
        try:
            result = collective(self.metadata, scatter_idx=3, gather_idx=1, group=group, _prepared_send=self.send)
            for owner in (*self.owners, self.send, result):
                owner.record_stream(self.stream)
            return result
        except BaseException as error:
            self.provider.poisoned = True
            _QUARANTINE.append((self, error))
            self.provider.fatal.report_failure("recovery_hccl", repr(error), (self, error))
            raise


def prepare_send(module, raw, rotary, ring_size):
    provider = load()
    if (not provider.ordinary(raw) or tuple(raw.shape) != (1, 585, 15360)
            or not raw.is_contiguous() or module.ulysses_world_size != 8 or ring_size != 1
            or module.num_local_heads != 40 or module.head_dim != 128
            or module.norm_q.eps != 1e-6 or module.norm_k.eps != 1e-6):
        return None
    cosine, sine = module._lingbot_native_bf16_qk_coefficients.get(*rotary)
    owners = (raw, module.norm_q.weight, module.norm_k.weight, cosine, sine)
    send = provider.call("qk_send", *owners)
    return Send(provider, raw.view(1, 585, 3, 40, 128), send, owners)


class Cross:
    def try_forward(self, q, k, v, *, role, scale, attn_mask, dropout_p, is_causal, enable_gqa):
        provider = load()
        if (role != "cross" or attn_mask is not None or dropout_p != 0 or is_causal or enable_gqa
                or not provider.ordinary(q) or tuple(q.shape) != (1,585,40,128)
                or k.ndim != 4 or tuple(k.shape) != tuple(v.shape)
                or tuple(k.shape[:1] + k.shape[2:]) != (1,40,128) or not 0 < k.shape[1] <= 512
                or any(not t.is_contiguous() for t in (q,k,v))):
            return None
        return provider.call("cross", q, k, v, scale)


class ExactPacking:
    def try_pair(self, stage, base, da, sa, db, sb):
        values = (base, da, sa, db, sb)
        if stage not in ("interior", "halo") or any(t is None for t in values):
            return False
        provider = load()
        if any(not provider.ordinary(t) or t.ndim not in (4,5) or t.stride(-1) != 1 for t in values):
            return False
        if not base.is_contiguous() or any(tuple(t.stride()) != tuple(base.stride()) for t in (da,db)):
            return False
        for t in values:
            desc = provider.describe(t)
            if desc[0] not in (0,2,30) or desc[1] != desc[0] or desc[2] != desc[4]:
                return False
        provider.call("copy_pair", *values)
        return True
