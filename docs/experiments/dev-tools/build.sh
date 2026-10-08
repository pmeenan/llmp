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
: "${NINJA:?Set NINJA to the absolute pinned Ninja binary}"
: "${CXX:?Set CXX to the absolute D-032 Clang compiler}"
: "${GOOGLETEST_SOURCE_DIR:?Set GOOGLETEST_SOURCE_DIR to prepared GoogleTest source}"
: "${GCC_INSTALL_DIR:?Set GCC_INSTALL_DIR to the selected GCC lib/gcc/<triple>/<major> directory}"
args=(-S "$source_dir" -B "$build_dir" -G Ninja "-DCMAKE_MAKE_PROGRAM=$NINJA"
      -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
      "-DGOOGLETEST_SOURCE_DIR=$GOOGLETEST_SOURCE_DIR"
      "-DSMOKE_FAILING=${FAILING:-OFF}")
flags="--gcc-install-dir=$GCC_INSTALL_DIR"
link_flags=''
if [[ ${STATIC_RUNTIME:-0} == 1 ]]; then
  link_flags+=' -static-libstdc++ -static-libgcc'
fi
if [[ ${SANITIZE:-0} == 1 ]]; then
  flags+=' -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer'
  link_flags+=' -fsanitize=address,undefined'
  args+=(-DSMOKE_ASAN_PROBE=ON)
fi
case "$profile" in
  native) flags+=' --target=x86_64-linux-gnu -march=x86-64' ;;
  spark) flags+=' --target=aarch64-linux-gnu -march=armv8-a' ;;
  cross)
    : "${SYSROOT:?Set SYSROOT to the D-032 Spark snapshot}"
    : "${LLD:?Set LLD to the absolute D-032 ld.lld-22 path}"
    : "${REMOTE:?Set REMOTE to the SSH host that runs the AArch64 tests}"
    printf '#!/usr/bin/env bash\nlink_flags=(--ld-path=%q)\nfor arg in "$@"; do\n  case "$arg" in -c|-E|-S|-M|-MM|-fsyntax-only) link_flags=();; esac\ndone\nexec %q --target=aarch64-linux-gnu --sysroot=%q -march=armv8-a "${link_flags[@]}" "$@"\n' \
      "$LLD" "$CXX" "$SYSROOT" > "$build_dir/clang++"
    chmod +x "$build_dir/clang++"
    CXX="$build_dir/clang++"
    # CTest runs each test command on REMOTE in the same absolute directory;
    # the caller first copies the build tree there at the same path.
    printf '#!/usr/bin/env bash\ncmd=$(printf "%%q " "$@")\nexec ssh -o BatchMode=yes %q "cd $(printf %%q "$PWD") && ${REMOTE_ENV:-} $cmd"\n' \
      "$REMOTE" > "$build_dir/remote-run"
    chmod +x "$build_dir/remote-run"
    args+=(-DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64
           "-DCMAKE_SYSROOT=$SYSROOT" -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER
           -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY
           -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY
           -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY
           "-DCMAKE_CROSSCOMPILING_EMULATOR=$build_dir/remote-run")
    ;;
esac
args+=("-DCMAKE_CXX_COMPILER=$CXX" "-DCMAKE_CXX_FLAGS=$flags"
       "-DCMAKE_EXE_LINKER_FLAGS=$link_flags")
"$CMAKE" "${args[@]}"
"$CMAKE" --build "$build_dir" --parallel 4
