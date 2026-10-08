#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds oracle.cc against the pinned llama.cpp image and runs it on a Spark
# (README.md): the prompts and the perplexity text, with upstream's CUDA
# fusion on (the default) and off, and optionally a dump of named tensors.
#
#   run_oracle.sh SOURCE MODEL_DIR OUT
#                 [fused|unfused|dump NAMES|bench|bench-unfused|bench-unfused-nographs]
#
# The bench modes run llama-bench with upstream's defaults (fusion and CUDA
# graphs on), with fusion off, and with fusion and CUDA graphs off.
#
# SOURCE is the llama.cpp source tree at b29c606e (its headers), MODEL_DIR
# the directory of the GGUF's shards, OUT a new directory.
set -euo pipefail
source_dir=$1
model_dir=$2
out=$3
mode=${4:-fused}
here=$(cd "$(dirname "$0")" && pwd)
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
shard=$(cd "$model_dir" && ls ./*-00001-of-*.gguf | head -1)
mkdir -p "$out"
docker=(sudo -n docker)
mounts=(--mount "type=bind,src=$here,dst=/harness,readonly"
        --mount "type=bind,src=$source_dir,dst=/source,readonly"
        --mount "type=bind,src=$model_dir,dst=/model,readonly"
        --mount "type=bind,src=$out,dst=/output")
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=1g "${mounts[@]}")
gpu=(--device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1)
if [ ! -x "$out/oracle" ]; then
  "${docker[@]}" "${common[@]}" --entrypoint g++ "$image" \
    -std=c++23 -O2 -march=armv8-a -Wall -Wextra -Werror \
    -I/source/include -I/source/ggml/include /harness/oracle.cc \
    -L/app -Wl,-rpath,/app -lllama -lggml -lggml-base -o /output/oracle
fi
# The perplexity text: docs/async-model.md at e7973e5 (README.md), copied by
# the caller to OUT/ppl.txt with its SHA-256 checked.
case "$mode" in
  fused)
    "${docker[@]}" "${common[@]}" "${gpu[@]}" --entrypoint /output/oracle "$image" \
      "/model/$shard" /output/fused --prompts /harness/prompts.tsv --generate 32 --ppl /output/ppl.txt ;;
  unfused)
    "${docker[@]}" "${common[@]}" "${gpu[@]}" --env GGML_CUDA_DISABLE_FUSION=1 \
      --entrypoint /output/oracle "$image" \
      "/model/$shard" /output/unfused --prompts /harness/prompts.tsv --generate 32 --ppl /output/ppl.txt ;;
  dump)
    "${docker[@]}" "${common[@]}" "${gpu[@]}" --env GGML_CUDA_DISABLE_FUSION=1 \
      --entrypoint /output/oracle "$image" \
      "/model/$shard" /output/dump --prompts /harness/prompts.tsv --generate 1 --dump "$5" ;;
  bench)
    # No speculation; flash attention on; prefill 512 and decode 64, three
    # repetitions (llama-bench's own).
    "${docker[@]}" "${common[@]}" "${gpu[@]}" --entrypoint /app/llama-bench "$image" \
      -m "/model/$shard" -ngl 99 -fa on -p 512 -n 64 -r 3 -o json -lm none ;;
  bench-unfused)
    "${docker[@]}" "${common[@]}" "${gpu[@]}" --env GGML_CUDA_DISABLE_FUSION=1 \
      --entrypoint /app/llama-bench "$image" \
      -m "/model/$shard" -ngl 99 -fa on -p 512 -n 64 -r 3 -o json -lm none ;;
  bench-unfused-nographs)
    "${docker[@]}" "${common[@]}" "${gpu[@]}" --env GGML_CUDA_DISABLE_FUSION=1 \
      --env GGML_CUDA_DISABLE_GRAPHS=1 --entrypoint /app/llama-bench "$image" \
      -m "/model/$shard" -ngl 99 -fa on -p 512 -n 64 -r 3 -o json -lm none ;;
  *)
    echo "unknown mode $mode" >&2; exit 2 ;;
esac
