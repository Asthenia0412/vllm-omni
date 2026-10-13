"""Independent private6 v3 CPU build/registration; no device qualification."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qk_private_retention_contract import (BUILD_SCHEMA, NAMESPACE, REGISTRATION_POLICY, validate_source)


def pin(path):
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-manifest", type=Path, required=True)
    parser.add_argument("--provider-manifest-sha256", required=True)
    parser.add_argument("--parent-v2-build", type=Path, required=True)
    parser.add_argument("--parent-v2-build-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().with_name("host.cpp")
    document = source.parents[2] / "QK_PRIVATE_RETENTION_V3_DESIGN_20261013.txt"
    original = source.parent.parent / "qk_single_queue" / "host.cpp"
    v1 = source.parent.parent / "qk_repeatable_pool" / "host.cpp"
    provider_path = args.provider_manifest.resolve(strict=True)
    provider = json.loads(provider_path.read_text())
    if pin(provider_path)["sha256"] != args.provider_manifest_sha256:
        raise RuntimeError("Explicit provider manifest SHA differs")
    parent_path = args.parent_v2_build.resolve(strict=True)
    parent_pin = pin(parent_path)
    if parent_pin["sha256"] != args.parent_v2_build_sha256:
        raise RuntimeError("Explicit frozen v2 build SHA differs")
    parent = json.loads(parent_path.read_text())
    if (parent.get("schema") != "lingbot_qk_repeatable_event_pool_cpu_build_v2"
            or parent.get("namespace") != "lingbot_qk_repeatable_pool_v2"
            or parent.get("registration") is not True
            or parent.get("npu_initialized") is not False
            or parent.get("v1_qualification_inherited") is not False
            or parent["provider_manifest"] != pin(provider_path)):
        raise RuntimeError("Exact unqualified frozen v2 compilation ancestry required")
    for name in ("source", "builder", "design", "library", "provider_manifest", "kernel"):
        if pin(parent[name]["path"]) != parent[name]:
            raise RuntimeError("Frozen v2 build pin changed: " + name)
    source_contract = validate_source(parent["source"]["path"], source)
    kernel_pin = provider["kernels"]
    kernel = Path(kernel_pin["path"]).resolve(strict=True)
    if pin(kernel) != kernel_pin:
        raise RuntimeError("Provider kernel pin differs; no implicit substitution")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    # Imports and registration are CPU-only. Explicit assertions reject a
    # caller that has already initialized the NPU or a load with side effects.
    import torch
    import torch_npu
    from torch.utils.cpp_extension import include_paths, library_paths
    if torch_npu.__version__ != "2.10.0.post4":
        raise RuntimeError("This isolated storage/queue ABI is pinned to torch-npu 2.10.0.post4")
    if torch_npu.version.git_version != "5dd8ef3f9b375b5ae4a83538d5785754148c3302":
        raise RuntimeError("Unexpected torch-npu source version")
    if torch_npu.npu.is_initialized():
        raise RuntimeError("Build process already initialized NPU")
    npu = Path(torch_npu.__file__).resolve().parent
    sdk = Path(os.environ["ASCEND_HOME_PATH"]).resolve(strict=True)
    lib = output / "liblingbot_qk_private_retention_v3.so"
    includes = include_paths() + [str(npu / "include"), str(sdk / "include")]
    libraries = library_paths() + [str(npu / "lib"), str(sdk / "lib64"), str(kernel.parent)]
    command = ["g++", "-std=c++17", "-shared", "-fPIC", "-O2", "-pthread",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
               str(source), "-o", str(lib)]
    command += ["-I" + value for value in includes]
    command += ["-L" + value for value in libraries]
    command += ["-Wl,-rpath," + value for value in libraries]
    command += [str(kernel), "-ltorch_npu", "-ltorch_cpu", "-ltorch", "-lc10",
                "-lopapi", "-lnnopbase", "-lascendcl", "-ldl"]
    headers = [sdk / "include/aclnn/acl_meta.h", sdk / "include/acl/acl_rt.h",
               npu / "include/torch_npu/csrc/core/NPUStorageImpl.h",
               npu / "include/torch_npu/csrc/core/npu/NPUStream.h",
               npu / "include/torch_npu/csrc/framework/OpCommand.h"]
    header_pins = [pin(path) for path in headers]
    before = {"source": pin(source), "builder": pin(__file__), "design": pin(document),
              "original_nonrepeatable_source": pin(original), "parent_v1_source": pin(v1),
              "provider_manifest": pin(provider_path), "kernel": kernel_pin,
              "parent_v2_build": parent_pin, "parent_v2_source": parent["source"],
              "source_contract_helper": pin(Path(__file__).resolve().parents[2] / "qk_private_retention_contract.py")}
    subprocess.run(command, check=True)
    torch.ops.load_library(str(lib))
    operator = getattr(torch.ops, NAMESPACE)
    dependencies = operator.dependencies()
    if torch_npu.npu.is_initialized():
        raise RuntimeError("CPU registration unexpectedly initialized NPU")
    for key in ("source", "builder", "design", "original_nonrepeatable_source", "parent_v1_source", "provider_manifest", "kernel", "parent_v2_build", "parent_v2_source", "source_contract_helper"):
        if pin(before[key]["path"]) != before[key]:
            raise RuntimeError(f"Source/dependency changed while building: {key}")
    if any(pin(item["path"]) != item for item in header_pins):
        raise RuntimeError("Pinned SDK/framework header changed while building")
    receipt = {
        "schema": BUILD_SCHEMA,
        "namespace": NAMESPACE, **before,
        "library": pin(lib), "dependencies": [pin(path) for path in dependencies],
        "headers": header_pins, "command": command,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "torch_npu_git": torch_npu.version.git_version,
        "statistics_fields": operator.stat_names(),
        "initialization_status_stride": ["plan", "stage", "stage1_status", "repeatable_status", "workspace_bytes"],
        "registration": True, "npu_initialized": False, "device_admitted": False,
        "model_admitted": False, "performance_admitted": False,
        "production_entrypoint": False, "historical_identity_proven": False,
        "v1_qualification_inherited": False, "v2_qualification_inherited": False,
        "source_contract": source_contract, "registration_policy": REGISTRATION_POLICY,
        "ownership_policy": "bounded_consumer_retirement_producer_reclaim_two_phase_close_v2",
    }
    (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
