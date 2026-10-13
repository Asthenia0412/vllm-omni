# SPDX-License-Identifier: Apache-2.0
"""CPU build and provenance only; this does not grant device/model admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def pin(path):
    p = Path(path).resolve(strict=True)
    b = p.read_bytes()
    return dict(path=str(p), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--source-root", type=Path, default=root)
    p.add_argument("--runtime-root", type=Path, default=root)
    p.add_argument("--provider-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    import torch
    import torch_npu
    from torch.utils.cpp_extension import include_paths, library_paths
    if (torch.__version__, torch_npu.__version__, torch_npu.version.git_version) != (
            "2.10.0+cpu", "2.10.0.post4", "5dd8ef3f9b375b5ae4a83538d5785754148c3302"):
        raise RuntimeError("Revalidate the framework ABI for a different version")
    if torch_npu.npu.is_initialized():
        raise RuntimeError("Use a fresh CPU-only process")
    source = a.source_root.resolve() / "csrc/lingbot_qk_combined/host.cpp"
    module = a.source_root.resolve() / "vllm_omni/diffusion/layers/lingbot_qk_combined.py"
    runtime = a.runtime_root.resolve()
    base = json.loads(a.provider_manifest.read_text())
    if base.get("schema") != "lingbot_recovery_build_v1" or base["historical_admission_inherited"] is not False:
        raise RuntimeError("Source-pinned reconstruction provider required")
    for key in ("library", "kernels"):
        if pin(base[key]["path"]) != base[key]:
            raise RuntimeError("Provider build changed: " + key)
    kernel = Path(base["kernels"]["path"])
    sdk = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    npu = Path(torch_npu.__file__).resolve().parent
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    lib = out / "liblingbot_qk_combined.so"
    includes = include_paths() + [str(npu / "include"), str(sdk / "include")]
    libraries = library_paths() + [str(npu / "lib"), str(sdk / "lib64"), str(kernel.parent)]
    cmd = ["g++", "-std=c++17", "-shared", "-fPIC", "-O2",
           f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}", str(source), "-o", str(lib)]
    cmd += ["-I" + s for s in includes] + ["-L" + s for s in libraries] + ["-Wl,-rpath," + s for s in libraries]
    cmd += [str(kernel), "-ltorch_npu", "-ltorch_cpu", "-ltorch", "-lc10", "-lascendcl", "-ldl"]
    subprocess.run(cmd, check=True)
    torch.ops.load_library(base["library"]["path"])
    torch.ops.load_library(str(lib))
    ops = torch.ops.lingbot_qk_combined_v1
    dependencies = [pin(item) for item in ops.dependencies()]
    if dependencies != [base["kernels"]] or torch_npu.npu.is_initialized():
        raise RuntimeError("Unexpected actual kernel resolution or NPU initialization")
    # Unsupported CPU input is a true prestart refusal. It must not initialize
    # a device, allocate NPU output or grant a device correctness claim.
    with torch.inference_mode():
        x = torch.empty(1)
        if ops.try_run(x, x, x, x, x, True) != (None, None):
            raise RuntimeError("CPU input did not refuse before allocation")
    if torch_npu.npu.is_initialized():
        raise RuntimeError("Unsupported admission initialized NPU")
    headers = ["core/NPUStorageImpl.h", "aten/CustomFunctions.h", "core/npu/NPUStream.h",
               "core/npu/NPUFunctions.h", "core/npu/NPUGraphsUtils.h", "core/npu/NPUCachingAllocator.h", "framework/OpCommand.h"]
    runtime_paths = [runtime / "vllm_omni/diffusion" / x for x in (
        "layers/lingbot_recovery.py", "layers/qk_single_host_integration_v1.py",
        "layers/lingbot_native_bf16_qk.py", "layers/lingbot_sp8_fatal.py",
        "layers/lingbot_real_fixture_capture.py", "layers/lingbot_output_capture.py",
        "models/lingbot_world/transformer.py", "distributed/comm.py")]
    manifest = dict(schema="lingbot_qk_combined_build_v1", source=pin(source), library=pin(lib),
        provider_manifest=pin(a.provider_manifest), dependencies=dependencies,
        runtime_sources=[pin(module), *[pin(x) for x in runtime_paths]],
        abi_headers=[*[pin(npu / "include/torch_npu/csrc" / h) for h in headers], pin(npu / "lib/libtorch_npu.so"),
                     pin(Path(torch.__file__).resolve().parent / "include/c10/core/GradMode.h"),
                     pin(Path(torch.__file__).resolve().parent / "include/c10/core/InferenceMode.h")],
        torch=torch.__version__, torch_npu=torch_npu.__version__, torch_npu_git=torch_npu.version.git_version,
        command=cmd, build_script=pin(__file__), registration_passed=True, cpu_refusal_passed=True,
        npu_initialized=False, historical_admission_inherited=False, device_admitted=False, model_admitted=False,
        execution="public native RMSQ/RMSK/RotaryV2Q/RotaryV2K + queued exact send; consolidated admission")
    (out / "qk_combined_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out / "qk_combined_manifest.json")


if __name__ == "__main__":
    main()
