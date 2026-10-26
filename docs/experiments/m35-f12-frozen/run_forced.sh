#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
# Builds forced_rows.cc against the digest-pinned llama.cpp v0.6.0 image (d8123504) and
# runs it on a Spark over the UD-Q2_K_XL DeepSeek V4 Flash 0731 GGUF (fbbb5b93):
#   run_forced.sh HARNESS_DIR HEADERS_DIR MODEL_DIR OUT [fused|unfused]
# HEADERS_DIR holds that release's include/llama.h and ggml/include (the image ships
# only the libraries); OUT receives the binary and rows-MODE.jsonl.
set -euo pipefail
here=$1; headers=$2; model_dir=$3; out=$4; mode=${5:-fused}
image=ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db
shard=$(cd "$model_dir" && ls ./*-00001-of-*.gguf | head -1)
mkdir -p "$out"
docker=(sudo -n docker)
mounts=(--mount "type=bind,src=$here,dst=/harness,readonly"
        --mount "type=bind,src=$headers,dst=/headers,readonly"
        --mount "type=bind,src=$model_dir,dst=/model,readonly"
        --mount "type=bind,src=$out,dst=/output")
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=1g "${mounts[@]}")
gpu=(--device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1)
if [ ! -x "$out/forced_rows" ] || [ "$here/forced_rows.cc" -nt "$out/forced_rows" ]; then
  "${docker[@]}" "${common[@]}" --entrypoint g++ "$image" \
    -std=c++17 -O2 -Wall -Wextra -Werror -I/headers/include -I/headers/ggml/include \
    /harness/forced_rows.cc -L/app -Wl,-rpath,/app -lllama -lggml -lggml-base -o /output/forced_rows
fi
env=()
[ "$mode" = unfused ] && env=(--env GGML_CUDA_DISABLE_FUSION=1)
"${docker[@]}" "${common[@]}" "${gpu[@]}" "${env[@]}" --entrypoint /output/forced_rows "$image" \
  "/model/$shard" 563,42,509,818,5279,529,7001,42 \
  "563,42;42,62684;509,818;818,5279;5279,529;529,7001;7001,17960;42,563" > "$out/rows-$mode.jsonl"
