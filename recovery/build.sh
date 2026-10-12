#!/usr/bin/env bash
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/env.sh"
set -u
cmake -S "$script_dir/../csrc/lingbot_recovery" \
  -B "$LINGBOT_RECOVERY_ROOT/build/kernels-release" -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_PYTHON_EXECUTABLE="$LINGBOT_VENV/bin/python"
cmake --build "$LINGBOT_RECOVERY_ROOT/build/kernels-release" -j 2
python "$script_dir/build_host.py" --output "$LINGBOT_RECOVERY_ROOT/build/host" \
  --kernels "$LINGBOT_RECOVERY_ROOT/build/kernels-release/lib/liblingbot_recovery_copy.so"
