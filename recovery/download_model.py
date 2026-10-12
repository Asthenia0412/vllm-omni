"""Download to persistent storage; record the resolved immutable revision."""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repo", default="robbyant/lingbot-world-v2-14b-causal-fast-diffusers")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(args.repo, revision=args.revision, files_metadata=True)
    manifest = {"repo": args.repo, "revision": info.sha, "complete": False,
                "historical_weights_identity_verified": False,
                "files": [{"path": item.rfilename, "size": item.size,
                           "blob_id": item.blob_id,
                           "lfs_sha256": getattr(item.lfs, "sha256", None)}
                          for item in info.siblings]}
    receipt = args.destination / "recovery_download.json"
    receipt.write_text(json.dumps(manifest, indent=2) + "\n")
    snapshot_download(args.repo, revision=info.sha, local_dir=args.destination, max_workers=4)
    manifest["complete"] = True
    receipt.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"destination": str(args.destination), "revision": info.sha, "complete": True}))


if __name__ == "__main__":
    main()
