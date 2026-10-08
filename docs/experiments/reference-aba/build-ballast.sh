#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'Usage: CXX=... LLD=... SYSROOT=... build-ballast.sh /outside/repo/ballast' >&2
  exit 2
fi
: "${CXX:?Select the D-032 Clang 22.1.8 compiler}"
: "${LLD:?Select the matching LLD 22.1.8 linker}"
: "${SYSROOT:?Select the verified D-032 AArch64 sysroot}"
aba_source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
"$CXX" --target=aarch64-linux-gnu --sysroot="$SYSROOT" \
  --gcc-toolchain="$SYSROOT/usr" --ld-path="$LLD" -march=armv8-a \
  -std=c++23 -O2 -Wall -Wextra -Werror "$aba_source_dir/ballast.cc" -o "$1"
# The runnable helper is verified again before privileged execution.
python3 - "$aba_source_dir/artifacts.json" "$1" <<'PY'
import hashlib, json, sys
with open(sys.argv[2], "rb") as binary:
    actual = hashlib.file_digest(binary, "sha256").hexdigest()
expected = json.load(open(sys.argv[1]))["ballast_sha256"]
if actual != expected:
    raise SystemExit(f"Helper build differs from the recorded binary: {actual}")
print(actual)
PY
