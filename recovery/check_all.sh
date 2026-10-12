#!/usr/bin/env bash
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/env.sh"
set -u
cd "$script_dir/.."
python - <<'PY'
import subprocess
for card in range(4):
    output=subprocess.check_output(['npu-smi','info','-t','proc-mem','-i',str(card)],text=True)
    if output.count('No process in device.') != 2:
        raise SystemExit(f'Physical NPU {card} is occupied; no device tests started')
PY
read -r manifest < "$LINGBOT_RECOVERY_ROOT/build/latest-manifest.txt"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for die in 0 1 2 3 4 5 6 7; do
    python recovery/check_operators.py --device "$die" --manifest "$manifest" \
        --output "$LINGBOT_RECOVERY_ROOT/check-operators-die${die}.json" \
        > "$LINGBOT_RECOVERY_ROOT/check-operators-die${die}.log" 2>&1
done
python recovery/supervise.py --runs "$LINGBOT_RECOVERY_ROOT/hccl-checks" \
    --manifest "$manifest" -- python -m torch.distributed.run --standalone \
    --nproc-per-node=8 recovery/check_hccl.py > "$LINGBOT_RECOVERY_ROOT/hccl-check.log" 2>&1
