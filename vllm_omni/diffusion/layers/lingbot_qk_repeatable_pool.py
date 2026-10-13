# SPDX-License-Identifier: Apache-2.0
"""Independent operator-only repeatable QK adapter, disabled by default.

Uses the frozen v2 native executor pool. Production construction is rejected
until a separately reviewed model admission exists. No inherited numeric/speed
PASS, no combined compute entry, and no implicit device synchronization.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time

CONFIG_KEY = "lingbot_npu_recovery_qk_repeatable_pool"
MANIFEST = "LINGBOT_QK_REPEATABLE_POOL_MANIFEST"
MANIFEST_SHA = MANIFEST + "_SHA256"
SCHEMA = "lingbot_qk_repeatable_runtime_binding_v1"
NAMESPACE = "lingbot_qk_repeatable_pool_v2"
POOL_POLICY = dict(max_pools=1, plan_capacity=8, retirement_capacity=32,
                   workspace_bytes_per_plan=79092224, workspace_budget_bytes=632737792)
_OTHERS = ("lingbot_npu_recovery_qk", "lingbot_npu_recovery_qk_combined",
           "lingbot_npu_qk_single_host_admission_v1")
_POLICY = (MANIFEST, MANIFEST_SHA, "LINGBOT_RECOVERY_MANIFEST", "LINGBOT_RECOVERY_MANIFEST_SHA256",
           "ASCEND_CUSTOM_OPP_PATH", "ASCEND_RT_VISIBLE_DEVICES", "TASK_QUEUE_ENABLE",
           "PYTORCH_NPU_ALLOC_CONF", "ROOT_LINGBOT_SP8_FATAL_DIR", "ROOT_LINGBOT_SP8_FATAL_NONCE",
           "ROOT_LINGBOT_SP8_EXPERIMENT_MODE", "ROOT_LINGBOT_QK_SINGLE_HOST_INTEGRATION_V1")
_LOCK = threading.RLock()
_BOUND = None
_QUARANTINE = []
_OPERATOR_TOKEN = object()


def need(condition, message):
    if not condition:
        raise RuntimeError("LingBot QK repeatable pool: " + message)


def pin(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def policy():
    return tuple(os.environ.get(key) for key in _POLICY)


def read_pinned(item):
    need(type(item) is dict and set(item) == {"path", "bytes", "sha256"}
         and Path(item["path"]).is_absolute() and pin(item["path"]) == item,
         "changed or missing evidence pin")
    return json.loads(Path(item["path"]).read_text())


def validate_binding(manifest):
    """CPU/source-only validation; compilation ancestry differs from runtime binding."""
    need(manifest.get("schema") == SCHEMA and manifest.get("namespace") == NAMESPACE
         and manifest.get("binding_only") is True and manifest.get("recompiled") is False
         and all(manifest.get(k) is False for k in ("historical_admission_inherited", "device_admitted",
                                                     "model_admitted", "performance_admitted")),
         "independent operator-only runtime binding required")
    need(manifest.get("pool_policy") == POOL_POLICY, "fixed provider-wide pool budget differs")
    ancestor = read_pinned(manifest["compilation_ancestor"])
    need(ancestor.get("schema") == "lingbot_qk_repeatable_event_pool_cpu_build_v2"
         and ancestor.get("registration") is True and ancestor.get("v1_qualification_inherited") is False,
         "v2 compilation ancestry required")
    for key in ("source", "library", "namespace", "dependencies", "statistics_fields", "ownership_policy"):
        need(manifest[key] == ancestor[key], "v2 compilation identity differs: " + key)
    guard = manifest["guard"]
    need(guard["namespace"] == "lingbot_qk_combined_v1"
         and guard["allowed_operations"] == ["validate_lease"]
         and guard["resolution_only_operations"] == ["dependencies"], "guard compute execution is forbidden")
    old_guard = read_pinned(guard["compilation_ancestor"])
    need(old_guard.get("schema") == "lingbot_qk_combined_build_v1"
         and old_guard.get("registration_passed") is True, "guard compilation ancestry required")
    for key in ("source", "library", "dependencies", "abi_headers"):
        need(guard[key] == old_guard[key], "guard compilation identity differs: " + key)
    base = read_pinned(manifest["provider_manifest"])
    need(base.get("schema") == "lingbot_recovery_build_v1"
         and base.get("registration_passed") is True
         and base.get("historical_admission_inherited") is False, "current base binding required")
    for parent in (ancestor, old_guard):
        old_base = read_pinned(parent["provider_manifest"])
        need(base.get("compilation_ancestor") == parent["provider_manifest"]
             and all(base["source"][key] == old_base["source"][key] for key in ("bytes", "sha256"))
             and all(base[key] == old_base[key] for key in ("library", "kernels")),
             "current base must preserve compilation ancestry and source bytes")
    freeze = read_pinned(manifest["freeze_manifest"])
    # The binder owns the freeze schema; every current runtime source is also
    # explicitly checked here, including this actual imported adapter path.
    need(type(freeze) is dict and bool(manifest["runtime_sources"]), "source freeze missing")
    items = [manifest["source"], manifest["library"], manifest["binder"],
             guard["source"], guard["library"], *manifest["dependencies"], *manifest["abi_headers"],
             *guard["dependencies"], *guard["abi_headers"], *manifest["runtime_sources"],
             *manifest["test_sources"]]
    checked = {}
    for item in items:
        if item["path"] not in checked:
            checked[item["path"]] = pin(item["path"])
        need(checked[item["path"]] == item, "source/dependency changed: " + item["path"])
    need(pin(__file__) in manifest["runtime_sources"], "actual imported adapter is not pinned")
    return ancestor


def _config_values(config):
    values = getattr(config, "model_config", None)
    need(isinstance(values, dict), "explicit model_config owner required")
    enabled = values.get(CONFIG_KEY, False)
    need(type(enabled) is bool, CONFIG_KEY + " must be bool")
    return values, enabled


def _operator_config(config, values):
    need(config.model_config is values and values.get(CONFIG_KEY) is True, "construction config changed")
    need(all(type(values.get(k, False)) is bool and not values.get(k, False) for k in _OTHERS),
         "QK providers are mutually exclusive")
    need(os.environ.get("ROOT_LINGBOT_QK_SINGLE_HOST_INTEGRATION_V1", "0") == "0",
         "historical route may not overlap")
    need(getattr(config, "model_class_name", None) == "LingBotWorldCausalDMDPipeline"
         and config.enforce_eager is True and values.get("lingbot_npu_native_bf16_qk") is True,
         "eager LingBot native BF16 QK config required")
    p = config.parallel_config
    need((p.tensor_parallel_size, p.sequence_parallel_size, p.ulysses_degree, p.ring_degree) == (1, 8, 8, 1),
         "SP8/TP1/ring1 required")


def configure_module(module):
    """Actual model construction hook: default off, enabled mode NOT admitted."""
    from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
    config = get_current_diffusion_config_or_none()
    if config is None or getattr(config, "model_config", None) is None:
        return None
    _, enabled = _config_values(config)
    if not enabled:
        return None
    raise RuntimeError("LingBot QK repeatable pool has operator-only evidence; model construction is not admitted")


def configure_operator_module(module, config):
    """Explicit standalone replay entry; never grants production model admission."""
    values, enabled = _config_values(config)
    need(enabled, "standalone operator config must explicitly enable the route")
    _operator_config(config, values)
    return PoolRoute(module, config, values, _token=_OPERATOR_TOKEN)


class LeaseGuard:
    """Expose only the verified read-only guard, never combined compute ops."""
    __slots__ = ("_validate",)

    def __init__(self, validate):
        self._validate = validate

    def validate_lease(self, *args):
        return self._validate(*args)


def load():
    global _BOUND
    with _LOCK:
        if _BOUND is not None:
            _BOUND.check_policy()
            return _BOUND
        values = policy()
        path = Path(values[0] or "")
        need(path.is_absolute() and pin(path)["sha256"] == values[1], "explicit runtime manifest SHA required")
        manifest = json.loads(path.read_text())
        validate_binding(manifest)
        need(values[2] and str(Path(values[2]).resolve()) == manifest["provider_manifest"]["path"]
             and values[3] == manifest["provider_manifest"]["sha256"], "current base provider binding differs")
        need(values[10] in ("correctness", "timing"), "root-owned operator experiment required")
        from .lingbot_recovery import load as load_base
        base = load_base()
        t, n = base.torch, base.backend
        need((str(t.__version__), str(n.__version__), n.version.git_version) ==
             (manifest["torch"], manifest["torch_npu"], manifest["torch_npu_git"]), "framework ABI differs")
        need(n._C._npu_get_soc_version() == 251, "arch22 only")
        t.ops.load_library(manifest["library"]["path"])
        ops = getattr(t.ops, NAMESPACE)
        need([str(Path(p).resolve(strict=True)) for p in ops.dependencies()] ==
             [p["path"] for p in manifest["dependencies"]], "pool SDK/kernel resolution differs")
        need(list(ops.stat_names()) == manifest["statistics_fields"] and not list(ops.handles()),
             "fresh v2 registration/statistics required")
        guard_manifest = manifest["guard"]
        t.ops.load_library(guard_manifest["library"]["path"])
        guard_ops = getattr(t.ops, guard_manifest["namespace"])
        need([str(Path(p).resolve(strict=True)) for p in guard_ops.dependencies()] ==
             [p["path"] for p in guard_manifest["dependencies"]], "guard kernel resolution differs")
        provider = PoolProvider(base, ops, LeaseGuard(guard_ops.validate_lease), manifest, values)
        provider.check_policy()
        _BOUND = provider
        return provider


def _tuple(value):
    return tuple(_tuple(v) for v in value) if isinstance(value, (tuple, list)) else value


class PoolProvider:
    def __init__(self, base, ops, guard, manifest, values):
        self.base, self.ops, self.guard, self.manifest, self.policy = base, ops, guard, manifest, values
        self.torch, self.backend = base.torch, base.backend
        self.producer_tid = threading.get_native_id()
        self.pools = {}  # Non-address descriptor keys -> handles; shared across all routes/layers.
        self.cold_records = []  # create wall only; no explicit synchronization or hot instrumentation.
        self._stat_names = tuple(manifest["statistics_fields"])
        self._closing = self._closed = False

    def check_policy(self):
        need(not self.base.poisoned and not self._closing and not self._closed,
             "provider poisoned or closing/closed")
        need(threading.get_native_id() == self.producer_tid and policy() == self.policy,
             "producer thread or launch policy changed")
        need(self.backend.npu.is_jit_compile_false()
             and self.backend._C._npu_getOption("ALLOW_INTERNAL_FORMAT") in (b"enable", "enable"),
             "JIT/internal format policy changed")

    def fail(self, error, owners):
        self.base.poisoned = True
        keep = (self, owners, error)
        _QUARANTINE.append(keep)
        try:
            self.base.fatal.report_failure("recovery_qk_repeatable_pool", repr(error), keep)
        except BaseException:
            pass

    def ordinary(self, args):
        """Ordinary unsupported inputs refuse before pool creation or producer mutation."""
        t = self.torch
        if len(args) != 5 or any(x.device.type != "npu" or x.dtype != t.bfloat16 for x in args):
            return False
        raw = args[0]
        need(all(x.device == raw.device for x in args), "mixed device owners")
        need(t.npu.current_device() == raw.device.index, "current device differs from input owner")
        if (t.is_grad_enabled() or not t.is_inference_mode_enabled() or t.are_deterministic_algorithms_enabled()
                or t.npu.is_current_stream_capturing()
                or t.npu.current_stream(raw.device) != t.npu.default_stream(raw.device)):
            return False
        shapes = ((1, 585, 15360), (5120,), (5120,), (1, 585, 1, 128), (1, 585, 1, 128))
        return (all(tuple(x.shape) == shape for x, shape in zip(args, shapes)) and raw.is_contiguous()
                and all(self.backend.get_npu_format(x) in (0, 2, 30) for x in args))

    def ensure_pool(self, args):
        """Cold cache bounded across layers; no eviction, synchronization or implicit growth."""
        try:
            self.check_policy()
            need(self.ordinary(args), "ensure_pool requires admitted operator inputs")
            descriptors = []
            for value in args:
                desc = self.base.describe(value)
                need(len(desc) == 11, "physical descriptor ABI differs")
                descriptors.append((str(value.dtype), str(value.device), *_tuple(desc[:9])))
            # Context is deliberately NOT guessed in Python. The frozen C++
            # create/run producer_check compares the true bound ACL context on
            # every call before refresh/launch. A context change is terminal.
            # Never read/hash Stream.npu_stream here: its pointer getter may
            # drain the framework queue. ordinary() already requires the
            # current device default stream; C++ verifies true stream/context.
            key = (self.producer_tid, str(args[0].device), tuple(descriptors))
            if key in self.pools:
                return self.pools[key]
            need(len(self.pools) < POOL_POLICY["max_pools"], "bounded descriptor pool exhausted; no fallback")
            started = time.perf_counter_ns()
            handle = self.ops.create(*args, POOL_POLICY["plan_capacity"], POOL_POLICY["retirement_capacity"])
            ended = time.perf_counter_ns()
            self.pools[key] = handle  # Retain registry handle before any post-create validation can fail.
            stats = self.snapshot(handle)
            self.cold_records.append(dict(handle=handle, create_start_ns=started, create_end_ns=ended,
                create_wall_ns=ended-started, explicit_sync=False, workspace_bytes=stats["workspace_resident_bytes"]))
            need(stats["producer_tid"] == self.producer_tid and stats["initialized"] == POOL_POLICY["plan_capacity"]
                 and stats["workspace_initial_bytes"] == POOL_POLICY["workspace_budget_bytes"]
                 and stats["workspace_resident_bytes"] == stats["workspace_initial_bytes"],
                 "cold pool capacity/workspace/thread differs")
            return handle
        except BaseException as error:
            self.fail(error, (args, self.pools))
            raise

    def snapshot(self, handle):
        values = tuple(self.ops.stats(handle))
        need(len(values) == len(self._stat_names), "statistics length differs")
        return dict(zip(self._stat_names, values))

    def snapshots(self):
        return [{"handle": handle, "stats": self.snapshot(handle)} for handle in self.pools.values()]

    def close_all_drained(self):
        """Caller drains first; this only enqueues SDK close. Caller drains again."""
        try:
            self.check_policy()
            self._closing = True
            for handle in self.pools.values():
                self.ops.close(handle)
            return tuple(self.pools.values())
        except BaseException as error:
            self.fail(error, self.pools)
            raise

    def finalize_all_drained(self):
        """After the second caller drain, release owners. Caller performs final drain."""
        try:
            need(self._closing and not self._closed and not self.base.poisoned
                 and policy() == self.policy and threading.get_native_id() == self.producer_tid,
                 "successful drained close on original producer required")
            for handle in self.pools.values():
                state = self.snapshot(handle)
                need(state["native_closed"] == 1 and state["closed"] == state["pending"] == state["poisoned"] == 0,
                     "SDK close has not completed")
                self.ops.finalize_close(handle)
                final = self.snapshot(handle)
                need(final["closed"] == final["native_closed"] == 1
                     and final["workspace_resident_bytes"] == final["retired_count"] == final["retained_call_slots"] == 0,
                     "producer owner finalization incomplete")
            self._closed = True
            return self.snapshots()
        except BaseException as error:
            self.fail(error, self.pools)
            raise


class PoolRoute:
    def __init__(self, module, config, values, *, _token=None):
        need(_token is _OPERATOR_TOKEN, "use explicit standalone configure_operator_module")
        _operator_config(config, values)
        self.module, self.config, self.values = module, config, values
        self._provider = None

    def prepare(self, raw, rotary, ring_size):
        provider = self._provider
        keep = (self, raw, rotary)
        try:
            _operator_config(self.config, self.values)
            module = self.module
            if (module.ulysses_world_size != 8 or ring_size != 1 or module.num_local_heads != 40
                    or module.head_dim != 128 or module.norm_q.eps != module.norm_k.eps
                    or module.norm_q.eps != 1e-6 or tuple(raw.shape) != (1, 585, 15360)):
                return None
            from .lingbot_native_bf16_qk import SharedCoefficients
            shared = module._lingbot_native_bf16_qk_coefficients
            need(type(shared) is SharedCoefficients, "shared coefficient provider changed")
            if rotary is None or len(rotary) != 2 or rotary[0] is not shared.cosine or rotary[1] is not shared.sine:
                return None
            need(type(shared.generation) is int and shared.generation > 0, "shared coefficient generation invalid")
            # Read only the canonical immutable snapshot. Never call get() on
            # a miss or expand/update coefficients as a side effect of refusal.
            args = (raw, module.norm_q.weight, module.norm_k.weight, shared.expanded_cosine, shared.expanded_sine)
            keep = (*keep, shared, shared.generation, *args)
            if provider is None:
                provider = load()
                self._provider = provider
            provider.check_policy()
            if not provider.ordinary(args):
                return None
            handle = provider.ensure_pool(args)
            send, owner = provider.ops.run(handle, *args)
            keep = (*keep, send, owner)  # Quarantine returned owners even if subsequent validation fails.
            need(send is not None and owner is not None, "pool returned incomplete output after submission")
            metadata = raw.view(1, 585, 3, 40, 128)
            return PoolLease(provider, self, metadata, send, owner, args, keep)
        except BaseException as error:
            if provider is not None:
                provider.fail(error, keep)
            raise


class PoolLease:
    def __init__(self, provider, route, metadata, send, owner, args, keep):
        self.provider, self.route = provider, route
        self.metadata, self.send, self.owner, self.args = metadata, send, owner, args
        self.owners = (*keep, metadata, send, owner)
        self.stream = provider.torch.npu.current_stream(send.device)
        self.used = self.entering = False
        self.pointers = (args[0].untyped_storage().data_ptr(), args[0].data_ptr(), owner.data_ptr())

    def exchange(self, collective, group):
        result = None
        try:
            need(not self.used and not self.entering and not self.provider.base.poisoned
                 and not self.provider._closing and not self.provider._closed, "lease reused or provider poisoned/closed")
            self.entering = True
            result = collective(self.metadata, scatter_idx=3, gather_idx=1, group=group,
                                _prepared_send=self.send, _qk_padded_owner=self)
            need(self.used, "collective omitted padded-lease admission")
            # v2 already recorded input/send storages; HCCL owns its own stream
            # protection. Retain the separately allocated consumer result too.
            result.record_stream(self.stream)
            return result
        except BaseException as error:
            self.provider.fail(error, (self, collective, group, result))
            raise
        finally:
            self.entering = False


def validate_padded_send(metadata, send, lease, world_size):
    need(type(lease) is PoolLease and lease.provider is _BOUND and type(lease.route) is PoolRoute
         and lease.route._provider is _BOUND and lease.entering and not lease.used and world_size == 8,
         "independent registered pool lease required")
    provider = lease.provider
    need(not provider.base.poisoned and not provider._closing and not provider._closed,
         "pool provider poisoned/closed before HCCL")
    need(metadata is lease.metadata and send is lease.send, "consumer changed lease tensors")
    need(threading.get_native_id() == provider.producer_tid
         and provider.torch.npu.current_stream(send.device) == lease.stream,
         "consumer producer-thread/stream changed")
    provider.guard.validate_lease(lease.args[0], metadata, send, lease.owner, *lease.pointers)
    lease.used = True  # Set before the unchanged original collective submits.
