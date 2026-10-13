# SPDX-License-Identifier: Apache-2.0
"""Independent operator-only private6 native provider; no model admission.

Use exact unchanged Route/Lease/consumer types through an explicitly recorded
empty-registry bridge. No original function or validator is replaced. The
original standalone config/environment keys retain their complete hot checks;
this independently pinned v3 manifest is the explicit candidate selection.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import lingbot_qk_repeatable_pool as original
from recovery import qk_private_retention_contract as source_contract_module

CONFIG_KEY, MANIFEST, MANIFEST_SHA = original.CONFIG_KEY, original.MANIFEST, original.MANIFEST_SHA
POOL_POLICY = original.POOL_POLICY
PoolRoute, PoolLease = original.PoolRoute, original.PoolLease
SCHEMA = "lingbot_qk_private_retention_runtime_binding_v3"
NAMESPACE = "lingbot_qk_private_retention_v3"
REGISTRATION_POLICY = source_contract_module.REGISTRATION_POLICY
BRIDGE = dict(policy="install_validated_v3_into_empty_original_registry_only",
              route_type="lingbot_qk_repeatable_pool.PoolRoute",
              lease_type="lingbot_qk_repeatable_pool.PoolLease", unchanged_consumer=True)
_LOCK, need, pin, policy, read_pinned = original._LOCK, original.need, original.pin, original.policy, original.read_pinned
LeaseGuard = original.LeaseGuard


class PrivateProvider(original.PoolProvider):
    """Only the independently loaded ops differ; all warm/lifetime code is inherited."""


def configure_module(module):
    # Default-off original policy still rejects enabled production construction.
    return original.configure_module(module)


def configure_operator_module(module, config):
    values, enabled = original._config_values(config)
    need(enabled, "standalone private-retention config must explicitly enable the route")
    original._operator_config(config, values)
    provider = load()
    route = original.configure_operator_module(module, config)
    route._provider = provider
    return route


def validate_binding(manifest):
    """CPU/source-only validation; compilation ancestry differs from runtime binding."""
    need(manifest.get("schema") == SCHEMA and manifest.get("namespace") == NAMESPACE
         and manifest.get("binding_only") is True and manifest.get("recompiled") is False
         and all(manifest.get(k) is False for k in ("historical_admission_inherited", "device_admitted",
                                                     "model_admitted", "performance_admitted")),
         "independent operator-only runtime binding required")
    need(manifest.get("pool_policy") == POOL_POLICY, "fixed provider-wide pool budget differs")
    ancestor = read_pinned(manifest["compilation_ancestor"])
    need(ancestor.get("schema") == "lingbot_qk_private_retention_cpu_build_v3"
         and ancestor.get("registration") is True and ancestor.get("npu_initialized") is False
         and all(ancestor.get(k) is False for k in ("v1_qualification_inherited", "v2_qualification_inherited",
                "device_admitted", "model_admitted", "performance_admitted", "production_entrypoint")),
         "private-retention v3 compilation ancestry required")
    for key in ("source", "library", "namespace", "dependencies", "statistics_fields", "ownership_policy"):
        need(manifest[key] == ancestor[key], "private-retention v3 compilation identity differs: " + key)
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
             ancestor["builder"], ancestor["design"], ancestor["source_contract_helper"],
             guard["source"], guard["library"], *manifest["dependencies"], *manifest["abi_headers"],
             *guard["dependencies"], *guard["abi_headers"], *manifest["runtime_sources"],
             *manifest["test_sources"]]
    checked = {}
    for item in items:
        if item["path"] not in checked:
            checked[item["path"]] = pin(item["path"])
        need(checked[item["path"]] == item, "source/dependency changed: " + item["path"])
    need(pin(__file__) in manifest["runtime_sources"], "actual imported adapter is not pinned")
    need(manifest.get("v2_qualification_inherited") is False
         and manifest.get("registration_policy") == REGISTRATION_POLICY,
         "independent private6 policy required")
    need(manifest.get("registry_bridge") == BRIDGE,
         "unchanged exact Route/Lease registry bridge required")
    need(manifest.get("original_operator_adapter") == pin(original.__file__)
         and pin(original.__file__) in manifest["runtime_sources"],
         "actual original operator adapter is not pinned")
    need(pin(source_contract_module.__file__) in manifest["runtime_sources"],
         "actual source-contract helper is not pinned")
    parent = read_pinned(manifest["parent_v2_build"])
    need(parent.get("schema") == "lingbot_qk_repeatable_event_pool_cpu_build_v2"
         and parent.get("namespace") == "lingbot_qk_repeatable_pool_v2"
         and parent.get("registration") is True and parent.get("npu_initialized") is False
         and parent.get("v1_qualification_inherited") is False
         and parent["source"] == manifest["parent_v2_source"]
         and parent["provider_manifest"] == ancestor["provider_manifest"],
         "frozen v2 source ancestry differs")
    need(pin(parent["source"]["path"]) == parent["source"], "frozen v2 source pin changed")
    expected = source_contract_module.validate_source(parent["source"]["path"], manifest["source"]["path"])
    need(manifest.get("source_contract") == ancestor.get("source_contract") == expected
         and ancestor.get("registration_policy") == REGISTRATION_POLICY
         and ancestor.get("v2_qualification_inherited") is False
         and ancestor.get("parent_v2_build") == manifest["parent_v2_build"],
         "private6 native source/compilation contract differs")
    return ancestor



def load():
    with _LOCK:
        if original._BOUND is not None:
            need(type(original._BOUND) is PrivateProvider,
                 "original provider registry is already occupied; fresh worker required")
            original._BOUND.check_policy()
            return original._BOUND
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
             "fresh private-retention v3 registration/statistics required")
        guard_manifest = manifest["guard"]
        t.ops.load_library(guard_manifest["library"]["path"])
        guard_ops = getattr(t.ops, guard_manifest["namespace"])
        need([str(Path(p).resolve(strict=True)) for p in guard_ops.dependencies()] ==
             [p["path"] for p in guard_manifest["dependencies"]], "guard kernel resolution differs")
        provider = PrivateProvider(base, ops, LeaseGuard(guard_ops.validate_lease), manifest, values)
        provider.check_policy()
        original._BOUND = provider
        return provider

