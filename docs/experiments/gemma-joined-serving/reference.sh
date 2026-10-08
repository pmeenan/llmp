#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
scratch="$HOME/.local/share/llmp/gemma-joined"
source_root="$HOME/src/llmp-wt/m3gm31"
models="$HOME/.local/share/llmp/reference-models"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)"
        --tmpfs /tmp:rw,size=1g --mount "type=bind,src=$scratch,dst=/scratch"
        --mount "type=bind,src=$source_root/docs/experiments/gemma-joined-serving,dst=/tool,readonly"
        --mount "type=bind,src=$models,dst=/model,readonly")
if [[ "$1" == build ]]; then
  sudo -n docker "${common[@]}" --entrypoint g++ "$image" -std=c++23 -O2 -march=armv8-a \
    -Wall -Wextra -Werror -I/scratch /tool/llama_joined.cc -L/app -Wl,-rpath,/app \
    -lllama -lggml -lggml-base -o /scratch/llama_joined
else
  case "$1" in
    26) raw=gemma-4-26B-A4B-it-UD-Q4_K_M.gguf ;;
    31) raw=gemma-4-31B-it-UD-Q4_K_XL.gguf ;;
    *) exit 2 ;;
  esac
  inputs=()
  if [[ $# == 5 ]]; then inputs=("/scratch/$5"); fi
  sudo -n docker "${common[@]}" --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 \
    --entrypoint /scratch/llama_joined "$image" "/model/$raw" "/scratch/$2" "$3" "$4" "${inputs[@]}"
fi
