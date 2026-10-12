"""Save committed source and the exact current runtime closure, excluding weights."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    repo = Path(__file__).resolve().parents[1]
    root = repo.parent
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo).strip():
        raise RuntimeError("Commit the recovery before creating its portable backup")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    bundle = root / "lingbot-recovery.bundle"
    subprocess.run(["git", "bundle", "create", str(bundle), branch], cwd=repo, check=True)
    subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=repo, check=True)
    manifest_path = Path((root / "build/latest-manifest.txt").read_text().strip())
    manifest = json.loads(manifest_path.read_text())
    dependencies = json.loads((repo / "recovery/ascend_dependency.json").read_text())
    for pin in [manifest["library"],manifest["kernels"],*dependencies["libraries"],*dependencies["additional_artifacts"]]:
        if hashlib.sha256(Path(pin["path"]).read_bytes()).hexdigest() != pin["sha256"]:
            raise RuntimeError("Runtime artifact changed: " + pin["path"])
    archive = root / "lingbot-recovery-source-runtime.tar.gz"
    sources = [bundle, manifest_path.parent, Path(manifest["kernels"]["path"]).parent,
               root/"build/latest-manifest.txt", root/"dependencies/vllm-ascend",
               root/"models/lingbot-world/recovery_download.json"]
    with tarfile.open(archive,"w:gz") as output:
        for path in sources:
            output.add(path,arcname=str(path.relative_to(root)))
    # Read every archive member to detect truncation/compression failure.
    with tarfile.open(archive,"r:gz") as check:
        for item in check:
            if item.isfile():
                with check.extractfile(item) as stream:
                    while stream.read(1024*1024):
                        pass
    result={"head":head,"branch":branch,"weights_included":False,"files":[]}
    for path in (bundle,archive):
        result["files"].append({"path":str(path),"bytes":path.stat().st_size,
                                "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (root/"BACKUP_MANIFEST.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
