# SPDX-License-Identifier: Apache-2.0
"""Source-bound loader for the read-only per-layer CPP metadata guard."""
import hashlib
import json
import os
from pathlib import Path
import threading

MANIFEST = "LINGBOT_STAGEB_CPP_MANIFEST"
MANIFEST_SHA = "LINGBOT_STAGEB_CPP_MANIFEST_SHA256"
_LOCK = threading.RLock()
_BOUND = None


def need(ok, message):
    if not ok:
        raise RuntimeError("StageB CPP provider: " + message)


def pin(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def policy():
    return os.environ.get(MANIFEST), os.environ.get(MANIFEST_SHA)


def checked(item):
    need(pin(item["path"]) == item, "Changed source/library/header: " + item["path"])
    return json.loads(Path(item["path"]).read_text())


class Provider:
    def __init__(self, classes, settings):
        self.Frame, self.Entry = classes.Frame, classes.Entry
        self.policy = settings


def check_policy(provider):
    need(provider is _BOUND and provider.policy == policy(), "CPP guard provider/manifest changed; restart8")


def load():
    global _BOUND
    with _LOCK:
        if _BOUND is not None:
            check_policy(_BOUND)
            return _BOUND
        settings = policy()
        path = Path(settings[0] or "")
        need(path.is_absolute() and pin(path)["sha256"] == settings[1], "Explicit runtime manifest SHA required")
        manifest = json.loads(path.read_text())
        need(manifest.get("schema") == "lingbot_stageB_guard_runtime_v2"
             and manifest.get("namespace") == "lingbot_stageB_guard_v2"
             and manifest.get("device_admitted") is False
             and manifest.get("model_admitted") is False
             and manifest.get("historical_admission_inherited") is False,
             "Independent, unqualified runtime binding required")
        ancestor = checked(manifest["compilation_ancestor"])
        need(ancestor.get("schema") == "lingbot_stageB_guard_cpu_build_v2"
             and ancestor.get("namespace") == manifest["namespace"]
             and ancestor.get("registration") is True and ancestor.get("npu_initialized") is False,
             "Actual CPU compilation/registration ancestry required")
        for key in ("source", "library", "abi_headers", "abi_libraries", "cxx11_abi", "torch", "torch_npu", "sdk"):
            need(manifest[key] == ancestor[key], "CPP compilation identity differs: " + key)
        for item in (manifest["source"], manifest["library"], *manifest["abi_headers"],
                     *manifest["abi_libraries"], *manifest["runtime_sources"]):
            need(pin(item["path"]) == item, "CPP runtime/ABI source changed: " + item["path"])
        need(pin(__file__) in manifest["runtime_sources"], "Actual loader is not source-pinned")
        from . import lingbot_fixed_modulation_stageB_cpp
        need(pin(lingbot_fixed_modulation_stageB_cpp.__file__) in manifest["runtime_sources"],
             "Actual CPP cache core is not source-pinned")
        import torch
        import torch_npu
        need((torch.__version__, torch_npu.__version__) == (manifest["torch"], manifest["torch_npu"]), "Framework ABI changed")
        need(int(torch._C._GLIBCXX_USE_CXX11_ABI) == manifest["cxx11_abi"]
             and str(Path(os.environ["ASCEND_HOME_PATH"]).resolve()) == manifest["sdk"], "Compiler/SDK ABI changed")
        need(torch.npu.is_initialized() and torch.npu.is_jit_compile_false()
             and torch.npu.get_device_name() == "Ascend910_9392", "Initialized JITFalse arch22 runtime required")
        from . import lingbot_sp8_fatal
        need(lingbot_sp8_fatal.ensure_protocol(), "Launch-owned fatal channel required")
        torch.ops.load_library(manifest["library"]["path"])
        provider = Provider(torch.classes.lingbot_stageB_guard_v2, settings)
        _BOUND = provider
        return provider
