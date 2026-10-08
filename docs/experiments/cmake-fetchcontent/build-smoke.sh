#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Retained experiment only; M1 supplies production toolchain files/presets.
if [[ $# != 2 || ! $1 =~ ^(native|cross|spark)$ ]]; then
  echo "Usage: $0 {native|cross|spark} OUTSIDE_REPO_BUILD_DIR" >&2
  exit 2
fi
profile=$1
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p -- "$2"
build_dir=$(cd -- "$2" && pwd)
: "${CMAKE:?Set CMAKE to the absolute pinned CMake binary}"
: "${CXX:?Set CXX to the absolute D-032 Clang compiler}"
args=(-S "$source_dir" -B "$build_dir" -G 'Unix Makefiles'
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
flags='-Wall -Wextra -Werror'
case "$profile" in
  native) flags+=' --target=x86_64-linux-gnu -march=x86-64' ;;
  spark) flags+=' --target=aarch64-linux-gnu -march=armv8-a' ;;
  cross)
    : "${SYSROOT:?Set SYSROOT to the D-032 Spark snapshot}"
    : "${LLD:?Set LLD to the absolute D-032 ld.lld-22 path}"
    # Compiler identification may link through NVCC's host compiler. Supply
    # LLD only for linking, so compile-only steps retain strict warnings.
    printf '#!/usr/bin/env bash\nlink_flags=(--ld-path=%q)\nfor arg in "$@"; do\n  case "$arg" in -c|-E|-S|-M|-MM|-fsyntax-only) link_flags=();; esac\ndone\nexec %q --target=aarch64-linux-gnu --sysroot=%q --gcc-toolchain=%q -march=armv8-a "${link_flags[@]}" "$@"\n' \
      "$LLD" "$CXX" "$SYSROOT" "$SYSROOT/usr" > "$build_dir/clang++"
    chmod +x "$build_dir/clang++"
    CXX="$build_dir/clang++"
    args+=(-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64
           "-DCMAKE_SYSROOT=$SYSROOT" -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER
           -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY
           -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY
           -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY
           '-DCMAKE_CUDA_FLAGS=--target-directory sbsa-linux')
    ;;
esac
args+=("-DCMAKE_CXX_COMPILER=$CXX" "-DCMAKE_CXX_FLAGS=$flags")
if [[ $profile != native ]]; then
  : "${CUDA_ROOT:?Set CUDA_ROOT to the D-032 compiler toolkit}"
  args+=(-DSMOKE_CUDA=ON "-DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc"
         "-DCMAKE_CUDA_HOST_COMPILER=$CXX" -DCMAKE_CUDA_ARCHITECTURES=121-real)
  if [[ $profile == spark ]]; then
    args+=('-DCMAKE_CUDA_FLAGS=-Xcompiler=-march=armv8-a')
  fi
else
  args+=(-DSMOKE_CUDA=OFF)
fi
"$CMAKE" "${args[@]}"
"$CMAKE" --build "$build_dir" --parallel 2
