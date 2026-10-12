"""Independent, default-off 4D VAE interior pack with captured descriptors."""
import hashlib
import json
import os
from pathlib import Path

FIELDS = ("npu_format", "origin_format", "base_sizes", "base_strides", "physical_sizes",
          "shape", "stride", "storage_offset", "storage_bytes")


def describe(value):
    import torch

    return dict(zip(FIELDS, torch.ops.lingbot_storage_descriptor.describe(value)))


def capture(owner, base, destination, source):
    import torch

    if tuple(source.shape) != (2, 384, 120, 26):
        return
    count = getattr(owner, "_lingbot_vae4d_capture_count", 0)
    if count >= 6:
        return
    owner._lingbot_vae4d_capture_count = count+1
    name = owner._lingbot_vae4d_name
    folder = Path(os.environ["LINGBOT_VAE4D_CAPTURE_DIR"])/f"rank{torch.distributed.get_rank()}"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{count:03d}"
    payload = folder/(stem+".pt")
    descriptors = [describe(value) for value in (base, destination, source)]
    torch.save({"base": base.detach().cpu(), "source": source.detach().cpu()}, payload)
    metadata = {"owner": name, "call": count, "rank": torch.distributed.get_rank(),
                "descriptors": descriptors, "payload": str(payload), "timing_eligible": False,
                "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
    (folder/(stem+".json")).write_text(json.dumps(metadata, indent=2)+"\n")


def try_pack(owner, base, destination, source):
    import torch

    policy = owner._lingbot_vae4d_policy
    if tuple(source.shape) != (2, 384, 120, 26) or source.dtype != torch.bfloat16:
        return False
    if base.untyped_storage().data_ptr() == source.untyped_storage().data_ptr():
        raise RuntimeError("4D VAE source aliases the managed destination arena")
    signature = [describe(value) for value in (base, destination, source)]
    if not any(row["owner"] == owner._lingbot_vae4d_name and row["descriptors"] == signature
               for row in policy["cases"]):
        return False
    owners = (owner, base, destination, source, policy)
    try:
        operation = (torch.ops.lingbot_vae_exact_pack4d_queued.copy_pair if policy.get("queued", False)
                     else torch.ops.lingbot_vae_exact_pack4d.copy_pair)
        operation(
            base, destination[:1], source[:1], destination[1:], source[1:])
        return True
    except BaseException as error:
        from vllm_omni.diffusion.layers.lingbot_sp8_fatal import report_failure

        report_failure("vae_exact_pack4d", str(error), owners)
        raise


def configure(vae, model_config):
    import torch

    manifest = model_config.get("lingbot_npu_vae4d_exact_pack_manifest")
    capture_dir = os.environ.get("LINGBOT_VAE4D_CAPTURE_DIR")
    if not manifest and not capture_dir:
        return
    descriptor = os.environ.get("LINGBOT_VAE4D_DESCRIPTOR_LIBRARY")
    if not descriptor:
        raise ValueError("4D VAE path requires a pinned descriptor library")
    torch.ops.load_library(descriptor)
    policy = None
    if manifest:
        policy = json.loads(Path(manifest).read_text())
        if policy.get("kind") != "lingbot_vae4d_exact_pack" or not policy.get("micro_passed"):
            raise ValueError("4D VAE requires a passing independent micro-screen manifest")
        for item in policy["pins"]:
            if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("4D VAE source/library/capture changed")
        torch.ops.load_library(policy["library"])
    if capture_dir and not str(Path(capture_dir).resolve()).startswith("/tmp/lingbot-generated-"):
        raise ValueError("4D payloads require a temporary generated-data spool")
    for name, module in vae.decoder.named_modules():
        if type(module).__name__ == "WanDistConv2d":
            module._lingbot_vae4d_name = name
            module._lingbot_vae4d_observer = bool(capture_dir)
            module._lingbot_vae4d_policy = policy
