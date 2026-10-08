#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds an instrumented copy of the FP16 toolchain bridge that logs GGML's
# CUDA pool high-water mark after every graph compute (fp16-pool-peak.patch),
# without touching the bridge build or the pinned source:
#   fp16_pool_peak.sh SDK LLAMA_SOURCE BRIDGE_BUILD PATCH OUT
# It copies BRIDGE_BUILD and LLAMA_SOURCE under OUT, applies PATCH to the
# copied source, recompiles only ggml-cuda.cu with the exact command recorded
# in the bridge's compile_commands.json (source path substituted), replaces
# that member of the copied libggml-cuda.a with the SDK's llvm-ar, and relinks
# fp16_reference with the bridge's own link command. Every other object is the
# bridge's. The instrumented executable is OUT/build/fp16_reference.
set -eu
[ ! -e "$5" ] || { echo "OUT must be new" >&2; exit 1; }
mkdir -p "$5"
SDK=$(realpath "$1") SOURCE=$(realpath "$2") BRIDGE=$(realpath "$3") PATCH=$(realpath "$4") OUT=$(realpath "$5")
cp -a "$BRIDGE" "$OUT/build"
cp -a "$SOURCE" "$OUT/source"
(cd "$OUT/source" && patch -p1 --forward < "$PATCH")
python3 - "$OUT/build/compile_commands.json" "$SOURCE" "$OUT/source" "$BRIDGE" "$OUT/build" > "$OUT/compile.sh" <<'PY'
import json, shlex, sys
db, src, new_src, bridge, new_build = sys.argv[1:]
entry = next(e for e in json.load(open(db)) if e["file"].endswith("/ggml-cuda/ggml-cuda.cu"))
if src not in entry["command"]:
    sys.exit("the recorded compile command does not name LLAMA_SOURCE; the patched source would not be compiled")
command = entry["command"].replace(src, new_src)
print("set -eu")
print("cd " + shlex.quote(new_build))
print(command)
PY
sh -x "$OUT/compile.sh"
cd "$OUT/build"
"$SDK/bin/llvm-ar" r llama.cpp/ggml/src/ggml-cuda/libggml-cuda.a \
  llama.cpp/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/ggml-cuda.cu.o
link=$("$SDK/bin/ninja" -t commands fp16_reference | tail -n 1)
rm -f fp16_reference
sh -xc "$link"
