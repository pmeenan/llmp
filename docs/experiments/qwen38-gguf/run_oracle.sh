#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds oracle.cc against llama.cpp b11254 (the source tree's headers, the
# local image's libraries) and runs it on a Spark (README.md): the prompts
# and the perplexity text, with upstream's CUDA fusion on (the default) and
# off; or llama-bench at 8K prefill and decode.
#
#   run_oracle.sh SOURCE MODEL_DIR INPUTS OUT [fused|unfused|bench|bench-ub2048]
#
# SOURCE is the llama.cpp source tree at 8019dc563 (b11254), MODEL_DIR the
# directory of the GGUF's shards, INPUTS the directory holding prompts.tsv
# and ppl.tsv (docs/experiments/qwen38-native/), OUT a new directory.
set -euo pipefail
source_dir=$1
model_dir=$2
inputs=$3
out=$4
mode=${5:-fused}
here=$(cd "$(dirname "$0")" && pwd)
image=llmp-llamacpp:b11254-cuda13
shard=$(cd "$model_dir" && ls ./*-00001-of-*.gguf | head -1)
mkdir -p "$out"
docker=(sudo -n docker)
mounts=(--mount "type=bind,src=$here,dst=/harness,readonly"
        --mount "type=bind,src=$source_dir,dst=/source,readonly"
        --mount "type=bind,src=$model_dir,dst=/model,readonly"
        --mount "type=bind,src=$inputs,dst=/inputs,readonly"
        --mount "type=bind,src=$out,dst=/output")
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=1g "${mounts[@]}")
gpu=(--device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1)
if [ ! -x "$out/oracle" ]; then
  "${docker[@]}" "${common[@]}" --entrypoint g++ "$image" \
    -std=c++17 -O2 -march=armv8-a -Wall -Wextra -Werror \
    -I/source/include -I/source/ggml/include /harness/oracle.cc \
    -L/app -Wl,-rpath,/app -lllama -lggml -lggml-base -o /output/oracle
fi
case "$mode" in
  fused)
    "${docker[@]}" "${common[@]}" "${gpu[@]}" --entrypoint /output/oracle "$image" \
      "/model/$shard" /output/fused --prompts /inputs/prompts.tsv --generate 32 --ppl /inputs/ppl.tsv ;;
  unfused)
    "${docker[@]}" "${common[@]}" "${gpu[@]}" --env GGML_CUDA_DISABLE_FUSION=1 \
      --entrypoint /output/oracle "$image" \
      "/model/$shard" /output/unfused --prompts /inputs/prompts.tsv --generate 32 --ppl /inputs/ppl.tsv ;;
  bench)
    # Upstream's defaults (fusion and CUDA graphs on, batch 2048, ubatch
    # 512); 8,192-token prefill and 128 decode steps, three repetitions.
    "${docker[@]}" "${common[@]}" "${gpu[@]}" --entrypoint /app/llama-bench "$image" \
      -m "/model/$shard" -ngl 99 -fa on -p 8192 -n 128 -r 3 -lm none -o json ;;
  bench-ub2048)
    # The same with 2,048-token ubatches (llama.cpp's faster prefill setting).
    "${docker[@]}" "${common[@]}" "${gpu[@]}" --entrypoint /app/llama-bench "$image" \
      -m "/model/$shard" -ngl 99 -fa on -p 8192 -n 0 -b 2048 -ub 2048 -r 3 -lm none -o json ;;
  *)
    echo "unknown mode $mode" >&2; exit 2 ;;
esac
