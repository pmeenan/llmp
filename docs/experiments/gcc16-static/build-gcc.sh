#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Builds GCC 16.2's C/C++ runtimes (libstdc++, libgcc, libatomic) natively for
# the current host into PREFIX. Only those runtimes and headers are used; llmpalooza
# compiles with Clang. Retained experiment, not M1 provisioning.
if [[ $# != 3 ]]; then
  echo "Usage: $0 PREPARED_GCC_SOURCE OUTSIDE_REPO_BUILD_DIR NEW_PREFIX" >&2
  exit 2
fi
src=$(cd -- "$1" && pwd)
mkdir -p -- "$2"
build=$(cd -- "$2" && pwd)
prefix=$3
[[ $prefix == /* && ! -e $prefix ]] || { echo "PREFIX must be absolute and new" >&2; exit 2; }
[[ -e $src/gmp && -e $src/mpfr && -e $src/mpc && -e $src/isl ]] ||
  { echo "Run contrib/download_prerequisites in the source first" >&2; exit 2; }
case "$(uname -m)" in
  x86_64) triple=x86_64-linux-gnu ;;
  aarch64) triple=aarch64-linux-gnu ;;
  *) echo "unsupported host" >&2; exit 2 ;;
esac
cd "$build"
"$src/configure" --prefix="$prefix" \
  --build="$triple" --host="$triple" --target="$triple" \
  --enable-languages=c,c++ --disable-multilib --disable-bootstrap \
  --with-gcc-major-version-only --enable-default-pie \
  --enable-threads=posix --enable-clocale=gnu --enable-libstdcxx-time=yes \
  --with-default-libstdcxx-abi=new --enable-gnu-unique-object \
  --disable-libsanitizer --disable-libgomp --disable-libitm --disable-libssp \
  --disable-libvtv --disable-libquadmath --disable-libquadmath-support \
  --disable-nls --disable-werror
make -j"$(nproc)"
make install-strip
