#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: $0 OUTSIDE_REPO_BUILD_DIR (D-032 cross SDK environment required)" >&2
  exit 2
fi
: "${CXX:?}" "${LLD:?}" "${SYSROOT:?}" "${CUDA_ROOT:?}"
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p -- "$1"
build_dir=$(cd -- "$1" && pwd)
printf '#!/usr/bin/env bash\nexec %q --target=aarch64-linux-gnu --sysroot=%q --gcc-toolchain=%q -march=armv8-a "$@"\n' \
  "$CXX" "$SYSROOT" "$SYSROOT/usr" > "$build_dir/clang++"
chmod +x "$build_dir/clang++"
"$CUDA_ROOT/bin/nvcc" -std=c++23 -O3 -arch=sm_121 --target-directory sbsa-linux \
  -ccbin "$build_dir/clang++" -cubin "$source_dir/kernel.cu" -o "$build_dir/kernel.cubin"
"$build_dir/clang++" -std=c++23 -O2 -Wall -Wextra -Werror "--ld-path=$LLD" \
  "-I$CUDA_ROOT/targets/sbsa-linux/include" "$source_dir/main.cc" \
  -lcuda -o "$build_dir/vmm-microbench"
