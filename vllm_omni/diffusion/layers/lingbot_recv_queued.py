"""Independent opt-in queue binding for the unchanged SP8 ingress kernel."""
import ctypes
import hashlib
import json
from pathlib import Path


def configure(manifest_path):
    if not manifest_path:
        return
    from . import lingbot_recv_kv_sp8 as original

    if not original.enabled() or original.is_active():
        raise ValueError("Queued ingress must bind before request preparation on the existing SP8 path")
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("kind") != "lingbot_sp8_recv_queued" or not manifest.get("build_passed"):
        raise ValueError("Queued ingress requires its independent successful build manifest")
    for pin in manifest["pins"]:
        if hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest() != pin["sha256"]:
            raise ValueError("Queued ingress source or raw kernel changed")
    driver = Path(manifest["library"]).resolve(strict=True)
    if hashlib.sha256(driver.read_bytes()).hexdigest() != manifest["library_sha256"]:
        raise ValueError("Queued ingress driver changed")
    if original._BOUND is None:
        original.bind_from_manifest()
    if "queued_binding" in original._BOUND:
        raise ValueError("Queued ingress is already installed")
    import torch

    torch.ops.load_library(str(driver))
    library = ctypes.CDLL(str(driver))
    namespace = torch.ops.lingbot_recv_kv_sp8_eager_queued
    address = namespace.linked_raw_launch_address()
    class DlInfo(ctypes.Structure):
        _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                    ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    info = DlInfo()
    if not dladdr(ctypes.c_void_p(address), ctypes.byref(info)) or not info.filename:
        raise ValueError("Cannot resolve queued driver's actual raw launch relocation")
    actual = Path(info.filename.decode()).resolve(strict=True)
    if actual != Path(manifest["raw"]).resolve(strict=True) or hashlib.sha256(actual.read_bytes()).hexdigest() != manifest["raw_sha256"]:
        raise ValueError("Queued driver's raw kernel was interposed")
    # Original provider/manager/staging checks and its unique commit/publisher
    # remain in place. Only the already guarded copy driver's launch is changed.
    original._BOUND["queued_binding"] = (library, manifest, namespace)
    original._BOUND["ingress"] = namespace.ingress_eager
