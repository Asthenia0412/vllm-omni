"""Rebuild missing LingBot host operators using the installed pinned SDK."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def pin(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu
    from torch.utils.cpp_extension import include_paths, library_paths

    if (torch.__version__, torch_npu.__version__) != ("2.10.0+cpu", "2.10.0.post4"):
        raise RuntimeError("Revalidate ABI before building for another framework version")
    root = Path(__file__).resolve().parents[1]
    sdk = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    npu = Path(torch_npu.__file__).resolve().parent
    source = root / "csrc/lingbot_recovery/host.cpp"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    library = output / "liblingbot_recovery_host.so"
    includes = include_paths() + [str(npu / "include"), str(sdk / "include")]
    kernels = args.kernels.resolve(strict=True)
    libraries = library_paths() + [str(npu / "lib"), str(sdk / "lib64"), str(kernels.parent)]
    command = ["g++", "-std=c++17", "-shared", "-fPIC", "-O2",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
               str(source), "-o", str(library)]
    command += ["-I" + p for p in includes] + ["-L" + p for p in libraries]
    command += ["-Wl,-rpath," + p for p in libraries]
    command += [str(kernels), "-ltorch_npu", "-ltorch_cpu", "-ltorch", "-lc10", "-lascendcl"]
    subprocess.run(command, check=True)
    torch.ops.load_library(str(library))
    manifest = {"schema": "lingbot_recovery_build_v1", "source": pin(source),
                "library": pin(library), "kernels": pin(kernels),
                "kernel_sources": [pin(p) for p in sorted((root / "csrc/lingbot_recovery").glob("*")) if p.is_file()],
                "torch": torch.__version__, "torch_npu": torch_npu.__version__,
                "sdk": str(sdk), "command": command, "registration_passed": True,
                "device_validated": False, "model_validated": False, "historical_admission_inherited": False}
    modules = ["layers/lingbot_recovery.py", "layers/lingbot_recv_kv_sp8.py",
               "layers/lingbot_sp8_fatal.py", "layers/qk_single_host_integration_v1.py",
               "layers/lingbot_fixed_timestep_bank_stageA_v1.py",
               "models/lingbot_world/transformer.py", "attention/backends/cross_combined_integration_v1.py",
               "distributed/autoencoders/wan_npu_exact_pack.py"]
    manifest["runtime_sources"] = [pin(root / "vllm_omni/diffusion" / p) for p in modules]
    manifest["abi_headers"] = [pin(npu / "include/torch_npu/csrc/core/NPUStorageImpl.h"),
                               pin(npu / "include/torch_npu/csrc/aten/CustomFunctions.h")]
    (output / "host_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
