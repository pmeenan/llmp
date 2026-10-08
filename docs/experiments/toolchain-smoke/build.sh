#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Experiment profiles, not the M1 application build or SDK provisioner.
if [[ $# != 2 || ! $1 =~ ^(native|cross|spark)$ ]]; then
  echo "Usage: $0 {native|cross|spark} OUTSIDE_REPO_BUILD_DIR" >&2
  exit 2
fi
profile=$1
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p -- "$2"
build_dir=$(cd -- "$2" && pwd)
: "${CXX:?Set CXX to an absolute Clang 22 compiler path}"
flags=(-std=c++23 -Wall -Wextra -Werror)
case "$profile" in
  native) flags+=(--target=x86_64-linux-gnu -march=x86-64) ;;
  spark) flags+=(--target=aarch64-linux-gnu -march=armv8-a) ;;
  cross)
    : "${SYSROOT:?Set SYSROOT to the Spark snapshot}"
    : "${LLD:?Set LLD to the absolute ld.lld-22 path}"
    # NVCC invokes the wrapper even for preprocessing/compiler identification.
    printf '#!/usr/bin/env bash\nexec %q --target=aarch64-linux-gnu --sysroot=%q --gcc-toolchain=%q -march=armv8-a "$@"\n' \
      "$CXX" "$SYSROOT" "$SYSROOT/usr" > "$build_dir/clang++"
    chmod +x "$build_dir/clang++"
    CXX="$build_dir/clang++"
    flags+=("--ld-path=$LLD")
    ;;
esac
"$CXX" "${flags[@]}" "$source_dir/main.cc" -o "$build_dir/cpu-smoke"
if [[ $profile != native ]]; then
  : "${CUDA_ROOT:?Set CUDA_ROOT to the CUDA compiler toolkit}"
  cuda_flags=("-std=${CUDA_STD:-c++23}" -arch=sm_121 -ccbin "$CXX")
  if [[ ${CUDA_STD:-c++23} == c++23 ]]; then
    cuda_flags+=(-DREQUIRE_CUDA_CXX23)
  fi
  if [[ $profile == cross ]]; then
    cuda_flags+=(--target-directory sbsa-linux)
  else
    cuda_flags+=(-Xcompiler=-march=armv8-a)
  fi
  "$CUDA_ROOT/bin/nvcc" "${cuda_flags[@]}" -c "$source_dir/kernel.cu" -o "$build_dir/kernel.o"
  "$CXX" "${flags[@]}" -DWITH_CUDA "$source_dir/main.cc" "$build_dir/kernel.o" \
    "-L$CUDA_ROOT/targets/sbsa-linux/lib" -lcudart -o "$build_dir/cuda-smoke"
fi
