#!/usr/bin/env bash
# Explicit operator qualification only. Each invocation owns one fresh eight-worker launch.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source recovery/env.sh
export PYTHONPATH="$PWD:$LINGBOT_RECOVERY_ROOT/dependencies/vllm-ascend"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export VLLM_LINGBOT_NPU_ASYNC_METADATA=0
unset RANK LOCAL_RANK WORLD_SIZE LINGBOT_SP8_OBSERVE_MODE LINGBOT_REAL_FIXTURE_CAPTURE_DIR LINGBOT_OUTPUT_CAPTURE_DIR
LINGBOT_POOL_MODE="${1:-}"
LINGBOT_POOL_ROUND="${2:-1}"
if [[ "$LINGBOT_POOL_MODE" != correctness && "$LINGBOT_POOL_MODE" != timing ]]; then
    echo 'Usage: bash recovery/run_qk_pool_warm_handle.sh correctness|timing [1|2|3]' >&2
    exit 2
fi
if [[ ! "$LINGBOT_POOL_ROUND" =~ ^[123]$ ]]; then
    echo 'Round must be 1, 2, or 3' >&2
    exit 2
fi
LINGBOT_POOL_BUILD="$LINGBOT_RECOVERY_ROOT/build/qk-pool-warm-handle.20261013v1"
LINGBOT_POOL_CHECKS="$LINGBOT_RECOVERY_ROOT/qk-pool-warm-handle-checks"
LINGBOT_POOL_EXTRA=()
if [[ "$LINGBOT_POOL_MODE" == timing ]]; then
    LINGBOT_POOL_OUTPUT="$LINGBOT_POOL_CHECKS/timing-round$LINGBOT_POOL_ROUND"
    LINGBOT_POOL_EXTRA=(--correctness-dir "$LINGBOT_POOL_CHECKS/correctness")
else
    LINGBOT_POOL_OUTPUT="$LINGBOT_POOL_CHECKS/correctness"
fi
exec python recovery/supervise.py --runs "$LINGBOT_RECOVERY_ROOT/qk-pool-warm-handle-runs" \
    --mode "$LINGBOT_POOL_MODE" --manifest "$LINGBOT_POOL_BUILD/host_manifest.json" -- \
    python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    recovery/check_qk_repeatable_matrix.py --execute --mode "$LINGBOT_POOL_MODE" \
    --round "$LINGBOT_POOL_ROUND" \
    --fixtures "$LINGBOT_RECOVERY_ROOT/audits/capture-fixtures/launch-v1/full/fixtures" \
    --manifest "$LINGBOT_POOL_BUILD/qk_repeatable_runtime_manifest.json" \
    "${LINGBOT_POOL_EXTRA[@]}" --output "$LINGBOT_POOL_OUTPUT"
