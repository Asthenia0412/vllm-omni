# SPDX-License-Identifier: Apache-2.0
"""Default-off reconstructed QK host admission; never inherits old PASS.

The native arithmetic remains two full-width RMSNorm calls and two interleaved
RotaryV2 calls, followed by exact send packing. This is a host admission
experiment, with independent per-call output ownership and unchanged HCCL.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import threading

CONFIG_KEY = "lingbot_npu_recovery_qk_combined"
MANIFEST = "LINGBOT_QK_COMBINED_MANIFEST"
MANIFEST_SHA = MANIFEST + "_SHA256"
_LOCK = threading.RLock()
_BOUND = None
_QUARANTINE = []
_POLICY = (MANIFEST, MANIFEST_SHA, "LINGBOT_RECOVERY_MANIFEST", "LINGBOT_RECOVERY_MANIFEST_SHA256",
           "ASCEND_CUSTOM_OPP_PATH", "ASCEND_RT_VISIBLE_DEVICES", "TASK_QUEUE_ENABLE",
           "PYTORCH_NPU_ALLOC_CONF", "ROOT_LINGBOT_SP8_FATAL_DIR", "ROOT_LINGBOT_SP8_FATAL_NONCE",
           "ROOT_LINGBOT_SP8_EXPERIMENT_MODE")


def need(condition, message):
    if not condition:
        raise RuntimeError("LingBot QK combined admission: " + message)


def pin(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def policy():
    return tuple(os.environ.get(key) for key in _POLICY)


def configure_module(module):
    """Bind only the actual construction context; this initializes no NPU."""
    from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
    config = get_current_diffusion_config_or_none()
    values = getattr(config, "model_config", None) or {}
    enabled = values.get(CONFIG_KEY, False)
    need(type(enabled) is bool, CONFIG_KEY + " must be bool")
    if not enabled:
        return None
    need(not values.get("lingbot_npu_recovery_qk", False)
         and not values.get("lingbot_npu_qk_single_host_admission_v1", False),
         "QK providers are mutually exclusive")
    need(os.environ.get("ROOT_LINGBOT_QK_SINGLE_HOST_INTEGRATION_V1", "0") == "0",
         "historical QK route may not overlap")
    need(getattr(config, "model_class_name", None) == "LingBotWorldCausalDMDPipeline"
         and config.enforce_eager is True and values.get("lingbot_npu_native_bf16_qk") is True,
         "eager LingBot native BF16 QK is required")
    p = config.parallel_config
    need((p.tensor_parallel_size, p.sequence_parallel_size, p.ulysses_degree, p.ring_degree) == (1, 8, 8, 1),
         "SP8/TP1/ring1 required")
    return Route(module, config, values)


def load():
    global _BOUND
    with _LOCK:
        if _BOUND is not None:
            _BOUND.check_policy()
            return _BOUND
        values = policy()
        path = Path(values[0] or "")
        need(path.is_absolute() and pin(path)["sha256"] == values[1], "explicit source-pinned manifest required")
        manifest = json.loads(path.read_text())
        need(manifest.get("schema") == "lingbot_qk_combined_build_v1"
             and manifest.get("registration_passed") is True
             and manifest.get("historical_admission_inherited") is False, "unsupported reconstruction manifest")
        for item in (manifest["source"], manifest["library"], manifest["provider_manifest"],
                     *manifest["dependencies"], *manifest["runtime_sources"], *manifest["abi_headers"]):
            need(pin(item["path"]) == item, "source/dependency changed: " + item["path"])
        need(str(Path(values[2]).resolve()) == manifest["provider_manifest"]["path"]
             and values[3] == manifest["provider_manifest"]["sha256"], "recovery provider manifest differs")
        from .lingbot_recovery import load as load_recovery
        base = load_recovery()  # Shared fatal/poison owner; forces actual kernel ELF resolution.
        t, n = base.torch, base.backend
        need((t.__version__, n.__version__, n.version.git_version) ==
             (manifest["torch"], manifest["torch_npu"], manifest["torch_npu_git"]), "framework ABI differs")
        need(n._C._npu_get_soc_version() == 251, "arch22 required")
        t.ops.load_library(manifest["library"]["path"])
        ops = t.ops.lingbot_qk_combined_v1
        need([str(Path(p).resolve()) for p in ops.dependencies()] ==
             [p["path"] for p in manifest["dependencies"]], "actual native SDK/kernel resolution differs")
        _BOUND = Provider(base, ops, manifest, values)
        _BOUND.check_policy()
        return _BOUND


class Provider:
    def __init__(self, base, ops, manifest, values):
        self.base, self.ops, self.manifest, self.policy = base, ops, manifest, values
        self.torch, self.backend = base.torch, base.backend

    def check_policy(self):
        need(not self.base.poisoned and policy() == self.policy, "provider poisoned or launch policy changed")
        # These effective getters are not exported as linkable C++ symbols in
        # this exact wheel. Read once per QK call; stream/owner checks are C++.
        need(self.backend.npu.is_jit_compile_false()
             and self.backend._C._npu_getOption("ALLOW_INTERNAL_FORMAT") in (b"enable", "enable"),
             "JIT/internal format policy changed")

    def fail(self, error, owners):
        self.base.poisoned = True
        keep = (self, owners, error)
        _QUARANTINE.append(keep)
        try:
            self.base.fatal.report_failure("recovery_qk_combined", repr(error), keep)
        except BaseException:
            pass


class Route:
    def __init__(self, module, config, values):
        self.module, self.config, self.values = module, config, values
        self._provider = None

    def prepare(self, raw, rotary, ring_size):
        provider = self._provider
        keep = (self, raw, rotary)
        try:
            if provider is None:
                provider = load()
                self._provider = provider
            provider.check_policy()
            need(self.config.model_config is self.values and self.values.get(CONFIG_KEY) is True,
                 "construction config owner or enablement changed")
            need(not self.values.get("lingbot_npu_recovery_qk", False)
                 and not self.values.get("lingbot_npu_qk_single_host_admission_v1", False),
                 "QK modes changed after startup")
            module = self.module
            compatible = (module.ulysses_world_size == 8 and ring_size == 1
                          and module.num_local_heads == 40 and module.head_dim == 128
                          and module.norm_q.eps == module.norm_k.eps == 1e-6)
            # Shape/scope refusal occurs before a leaf-only coefficient expansion.
            if not compatible or tuple(raw.shape) != (1, 585, 15360):
                return None
            shared = module._lingbot_native_bf16_qk_coefficients
            from .lingbot_native_bf16_qk import SharedCoefficients
            need(type(shared) is SharedCoefficients, "shared coefficient provider changed")
            # The canonical transformer has already prepared one fresh snapshot
            # for all 40 layers. Admit only that immutable snapshot. A leaf miss
            # refuses before get() could expand or mutate any producer state.
            if not (rotary[0] is shared.cosine and rotary[1] is shared.sine):
                return None
            need(type(shared.generation) is int and shared.generation > 0,
                 "shared coefficient generation is invalid")
            cosine, sine = shared.get(*rotary)
            args = (raw, module.norm_q.weight, module.norm_k.weight, cosine, sine)
            keep = (*keep, shared, shared.generation, *args)
            send, owner = provider.ops.try_run(*args, compatible)
            keep = (*keep, send, owner)
            need((send is None) == (owner is None), "partial optional result after QK call")
            if send is None:
                return None
            metadata = raw.view(1, 585, 3, 40, 128)
            return Lease(provider, self, metadata, send, owner, args, keep)
        except BaseException as error:
            if provider is not None:
                provider.fail(error, keep)
            raise


class Lease:
    def __init__(self, provider, route, metadata, send, owner, args, keep):
        self.provider, self.route = provider, route
        self.metadata, self.send, self.owner, self.args = metadata, send, owner, args
        self.owners = (*keep, metadata, send, owner)
        self.stream = provider.torch.npu.current_stream(send.device)
        self.used = self.entering = False
        raw = args[0]
        self.pointers = (raw.untyped_storage().data_ptr(), raw.data_ptr(), owner.data_ptr())

    def exchange(self, collective, group):
        try:
            need(not self.used and not self.entering and not self.provider.base.poisoned,
                 "lease reused or provider poisoned")
            self.entering = True
            result = collective(self.metadata, scatter_idx=3, gather_idx=1, group=group,
                                _prepared_send=self.send, _qk_padded_owner=self)
            need(self.used, "collective omitted the required padded-lease admission")
            for owner in (*self.args, self.metadata, self.send, self.owner, result):
                owner.record_stream(self.stream)
            return result
        except BaseException as error:
            self.provider.fail(error, (self, collective, group))
            raise
        finally:
            self.entering = False


def validate_padded_send(metadata, send, lease, world_size):
    need(type(lease) is Lease and lease.provider is _BOUND and type(lease.route) is Route
         and lease.route._provider is _BOUND and lease.entering and not lease.used
         and not _BOUND.base.poisoned and world_size == 8,
         "new registered QK lease required")
    need(metadata is lease.metadata and send is lease.send, "QK consumer changed tensors")
    need(_BOUND.torch.npu.current_stream(send.device) == lease.stream, "QK consumer stream changed")
    _BOUND.ops.validate_lease(lease.args[0], metadata, send, lease.owner, *lease.pointers)
    lease.used = True  # Set before the unchanged original collective can submit.
