#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
scratch="$HOME/.local/share/llmp/gemma26-swa-ring-transfer"
source_root="$HOME/src/llmp-wt/m3gm31"
models="$HOME/.local/share/llmp/reference-models"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=1g
        --mount "type=bind,src=$scratch,dst=/scratch"
        --mount "type=bind,src=$source_root/docs/experiments/gemma26-swa-ring-transfer,dst=/tool,readonly")
case "${1:-}" in
  build)
    [[ $# == 1 ]]
    sudo -n docker "${common[@]}" --entrypoint g++ "$image" -std=c++23 -O2 -march=armv8-a \
      -Wall -Wextra -Werror -I/scratch/headers /tool/llama_prefill_ring.cc -L/app -Wl,-rpath,/app \
      -lllama -lggml -lggml-base -o /scratch/llama_prefill_ring
    sudo -n docker "${common[@]}" --entrypoint sha256sum "$image" /app/libllama.so /app/libggml.so /app/libggml-base.so /app/libggml-cuda.so > "$scratch/libraries.sha256"
    sha256sum "$scratch/llama_prefill_ring"
    ;;
  replay)
    [[ $# == 2 && "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]
    sudo -n docker "${common[@]}" --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 \
      --mount "type=bind,src=$models,dst=/model,readonly" --entrypoint /scratch/llama_prefill_ring "$image" \
      /model/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf /scratch/ids.i32 "/scratch/$2"
    ;;
  *) exit 2 ;;
esac
