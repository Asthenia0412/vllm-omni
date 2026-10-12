"""Own one service process group; fatal post-write errors retire every worker.

Never select processes by name or device. Only this launch's uid/pgid/start-time
identities may receive a signal. Each launch gets a fresh persistent directory.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time


def identity(pid):
    try:
        path = Path("/proc") / str(pid)
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "uid": path.stat().st_uid, "pgid": int(fields[2]), "start": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def members(leader):
    found = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            current = identity(int(path.name))
            if (current and current["uid"] == os.getuid() and current["pgid"] == leader["pgid"]
                    and current["start"] >= leader["start"]):
                found.append(current)
    return found


def retire(leader):
    for sig, seconds in ((signal.SIGTERM, 10), (signal.SIGKILL, 2)):
        owned = members(leader)
        for process in owned:
            if identity(process["pid"]) == process:
                try:
                    os.kill(process["pid"], sig)
                except ProcessLookupError:
                    pass
        until = time.monotonic() + seconds
        while time.monotonic() < until and members(leader):
            time.sleep(.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--recv", action="store_true")
    parser.add_argument("--mode", choices=("correctness", "timing", "profile"), default="correctness")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a service command after -- is required")
    nonce = secrets.token_hex(16)
    run = args.runs.resolve() / nonce
    run.mkdir(parents=True, mode=0o700)
    env = os.environ.copy()
    env.update(ROOT_LINGBOT_SP8_FATAL_BASE=str(args.runs.resolve()), ROOT_LINGBOT_SP8_FATAL_DIR=str(run),
               ROOT_LINGBOT_SP8_FATAL_NONCE=nonce, ROOT_LINGBOT_SP8_EXPERIMENT_MODE=args.mode)
    if args.manifest:
        import hashlib
        env.update(LINGBOT_RECOVERY_MANIFEST=str(args.manifest.resolve()),
                   LINGBOT_RECOVERY_MANIFEST_SHA256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    if args.recv:
        if not args.manifest:
            parser.error("--recv requires the reconstructed build manifest")
        env.update(LINGBOT_RECOVERY_RECV="1", LINGBOT_RECV_Q_CURRENT_KV_EAGER="1")
    else:
        env.update(LINGBOT_RECOVERY_RECV="0", LINGBOT_RECV_Q_CURRENT_KV_EAGER="0")
    # Historical experiment launch flags cannot silently select lost providers.
    env.update(ROOT_LINGBOT_QK_SINGLE_HOST_INTEGRATION_V1="0", ROOT_LINGBOT_CROSS_COMBINED_INTEGRATION_V1="0")
    env["LINGBOT_NPU_VAE_EXACT_PACK"] = "0"
    with (run / "service.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        leader = identity(process.pid)
        if leader is None:
            raise RuntimeError("Service exited before its identity could be recorded")
        receipt = {"nonce": nonce, "leader": leader, "command": command, "mode": args.mode,
                   "manifest": str(args.manifest), "fatal": None}
        (run / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(run, flush=True)
        code = 1
        known = {leader["pid"]: leader}
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        try:
            while process.poll() is None:
                owned = {item["pid"]: item for item in members(leader)}
                known.update(owned)
                for path in run.glob("fatal-*.json"):
                    record = json.loads(path.read_text())
                    sender = known.get(record.get("pid"))
                    if (sender and record.get("nonce") == nonce
                            and record.get("process_start_ticks") == sender["start"]
                            and record.get("boot_id") == boot_id
                            and record.get("restart_scope") == "all8"):
                        receipt["fatal"] = str(path)
                        break
                if receipt["fatal"]:
                    break
                time.sleep(.5)
            code = process.poll() if not receipt["fatal"] else 1
        finally:
            retire(leader)
            process.wait(timeout=15)
            receipt.update(exit_code=code, remaining_owned_processes=members(leader))
            (run / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    raise SystemExit(code or 0)


if __name__ == "__main__":
    main()
