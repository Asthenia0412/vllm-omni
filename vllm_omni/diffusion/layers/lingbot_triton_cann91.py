# SPDX-License-Identifier: Apache-2.0
"""Audited process-local host-driver selection for opt-in precision fusion.

No installed package mutation. A successful private host build is necessary
but never suffices to authorize a device numerical/performance candidate.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

_selected_binary: str | None = None


def configure_driver(binary: str | None = None) -> dict:
    global _selected_binary
    path = Path(binary or os.environ.get("VLLM_LINGBOT_NPU_FUSION_DRIVER_SO", "")).resolve()
    if not path.is_file() or not path.is_relative_to(Path("/mnt/data")):
        raise RuntimeError("Opt-in fusion requires an audited private CANN9.1 driver .so on persistent NFS")
    manifest_path = path.parent/"driver_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    import torch
    toolkit = Path(os.environ.get("ASCEND_HOME_PATH", "")).resolve()
    if (not manifest["success"] or manifest["installed_files_modified"]
        or str(toolkit) != manifest["ascend_home_path"]
        or importlib.metadata.version("triton-ascend") != manifest["triton_version"]
        or importlib.metadata.version("torch-npu") != manifest["torch_npu_version"]
        or torch.__version__ != manifest["torch_version"]
        or int(torch._C._GLIBCXX_USE_CXX11_ABI) != manifest["cxx11_abi"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["binary_sha256"]
        or hashlib.sha256(Path(manifest["patched_source"]).read_bytes()).hexdigest() != manifest["patched_source_sha256"]
        or hashlib.sha256(Path(manifest["installed_source"]).read_bytes()).hexdigest() != manifest["installed_source_sha256"]):
        raise RuntimeError("CANN9.1 driver provenance/runtime mismatch; refuse device launch")
    for runtime in manifest["runtime_inputs"]:
        if hashlib.sha256(Path(runtime["path"]).read_bytes()).hexdigest() != runtime["sha256"]:
            raise RuntimeError("Private Triton driver runtime/header changed; rebuild and revalidate")
    if _selected_binary is not None and _selected_binary != str(path):
        raise RuntimeError("One process cannot switch audited Triton driver binaries")
    if _selected_binary is None:
        from triton.backends.ascend.driver import NPUUtils
        from triton.backends import backends
        # Triton3.2.2 discovers backend files with a separate importlib module
        # instead of the normal package import. Its driver owns another class.
        # Cover both classes before either singleton loads a shared library.
        discovered = backends["ascend"].driver.__init__.__globals__["NPUUtils"]
        classes = {NPUUtils, discovered}
        # The singleton's existing method resolves a source-built cached .so.
        # Replace only that resolver; kernel compilation/launch logic is intact.
        for cls in classes:
            instance = cls()
            if instance.npu_utils_mod is not None:
                raise RuntimeError("Triton host driver already initialized before precision fusion selection")
        for cls in classes:
            cls._build_or_get_cached_so = lambda self: str(path)
            cls()._cache_path = str(path)
        manifest = dict(manifest, process_local_resolver_classes=sorted(cls.__module__ for cls in classes))
        _selected_binary = str(path)
    return manifest
