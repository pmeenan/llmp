#!/bin/bash
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
scratch="$HOME/.local/share/jitllm/gemma-assistant-reference"
source_root="$HOME/src/jitLLM-wt/m3gm31"
models="$HOME/.local/share/jitllm/reference-models"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)"
        --tmpfs /tmp:rw,size=1g --mount "type=bind,src=$scratch,dst=/scratch"
        --mount "type=bind,src=$source_root/docs/experiments/gemma-assistant-reference,dst=/tool,readonly"
        --mount "type=bind,src=$models,dst=/model,readonly")
case "${1:-}" in
  build)
    [[ $# == 1 ]]
    sudo -n docker "${common[@]}" --entrypoint g++ "$image" -std=c++23 -O2 -march=armv8-a \
      -Wall -Wextra -Werror -I/scratch/headers/include -I/scratch/headers/src \
      -I/scratch/headers/ggml/include /tool/llama_assistant.cc -L/app -Wl,-rpath,/app \
      -lllama -lggml -lggml-base -o /scratch/llama_assistant
    sha256sum "$scratch/llama_assistant"
    ;;
  acquire1|acquire-serial2|acquire-batch2)
    [[ $# == 2 && "$2" =~ ^[a-zA-Z0-9_-]+$ ]]
    owners=1 mode=serial
    if [[ "$1" != acquire1 ]]; then owners=2; fi
    if [[ "$1" == acquire-batch2 ]]; then mode=batch; fi
    sudo -n docker "${common[@]}" --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 \
      --entrypoint /scratch/llama_assistant "$image" \
      /model/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf /model/mtp-gemma-4-26B-A4B-it.gguf \
      /scratch/inputs.i32 "/scratch/$2" "$owners" "$mode"
    ;;
  *) exit 2 ;;
esac
