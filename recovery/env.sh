#!/usr/bin/env bash
# Source with explicit persistent roots. Never place weights/builds in /tmp.
export LINGBOT_RECOVERY_ROOT="${LINGBOT_RECOVERY_ROOT:-/mnt/sfs_turbo/lingbot-recovery-zhoutaichang-20261012}"
export LINGBOT_CANN_ROOT="${LINGBOT_CANN_ROOT:-/home/zhoutaichang/tools/ascend-9.1-a3/usr/local/Ascend}"
export LINGBOT_VENV="${LINGBOT_VENV:-/home/zhoutaichang/venvs/vllm-omni}"
source "$LINGBOT_CANN_ROOT/cann-9.1.0/set_env.sh"
source "$LINGBOT_CANN_ROOT/nnal/atb/9.1.0/atb/set_env.sh" --cxx_abi=1
source "$LINGBOT_VENV/bin/activate"
export PYTHONPATH="$LINGBOT_RECOVERY_ROOT/worktree:$LINGBOT_RECOVERY_ROOT/dependencies/vllm-ascend${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$LINGBOT_RECOVERY_ROOT/hf-cache"
# Match the historical per-worker CPU thread budget; avoid host-wide defaults.
export OMP_NUM_THREADS="${LINGBOT_OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${LINGBOT_MKL_NUM_THREADS:-8}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TARGET_DEVICE=npu
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_LINGBOT_NPU_ATTENTION_BACKEND=fia_bsnd
export VLLM_LINGBOT_NPU_ATTENTION_STRICT=1
export VLLM_LINGBOT_NPU_ATTENTION_SPLITS=1
# Enable the manager-owned history buffers requested by the SP8 configs.
export VLLM_LINGBOT_NPU_HISTORY_STAGING=1
export VLLM_LINGBOT_NPU_CROSS_H40_BSND_V1=1
export VLLM_LINGBOT_NPU_FUSED_POINTWISE=1
export VLLM_LINGBOT_NPU_LOSSY_POINTWISE=0
# Required by the retained device-interleave eligibility contract.
export VLLM_LINGBOT_NPU_PLANAR_RGB=1
export VLLM_LINGBOT_NPU_PINNED_RGB=0
export ASCEND_CUSTOM_OPP_PATH="$LINGBOT_RECOVERY_ROOT/dependencies/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
# Device selection, candidate enablement and fatal nonce belong to each launch.
