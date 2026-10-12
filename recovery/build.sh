#!/usr/bin/env bash
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/env.sh"
set -u
# CANN's legacy CMake flow rewrites intermediates in place; a second incremental
# link can consume an already transformed object. Use a fresh build directory
# and retain previous binaries so running workers never see an overwritten DSO.
mkdir -p "$LINGBOT_RECOVERY_ROOT/build"
kernel_build=$(mktemp -d "$LINGBOT_RECOVERY_ROOT/build/kernels.XXXXXXXX")
host_build=$(mktemp -d "$LINGBOT_RECOVERY_ROOT/build/host.XXXXXXXX")
cmake -S "$script_dir/../csrc/lingbot_recovery" \
  -B "$kernel_build" -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_PYTHON_EXECUTABLE="$LINGBOT_VENV/bin/python"
cmake --build "$kernel_build" -j 2
python "$script_dir/build_host.py" --output "$host_build" \
  --kernels "$kernel_build/lib/liblingbot_recovery_copy.so"
printf '%s\n' "$host_build/host_manifest.json" > "$LINGBOT_RECOVERY_ROOT/build/latest-manifest.txt"
