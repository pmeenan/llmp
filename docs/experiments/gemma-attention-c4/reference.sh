#!/bin/bash
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
scratch="$HOME/.local/share/jitllm/gemma-attention-c4"
source_root="$HOME/src/jitLLM-wt/m3gm31"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)"
        --tmpfs /tmp:rw,size=1g --mount "type=bind,src=$scratch,dst=/scratch"
        --mount "type=bind,src=$source_root/docs/experiments/gemma-attention-c4,dst=/tool,readonly")
case "${1:-}" in
  build)
    [[ $# == 1 ]]
    sudo -n docker "${common[@]}" --device nvidia.com/gpu=all --entrypoint g++ "$image" -std=c++23 -O2 -march=armv8-a \
      -Wall -Wextra -Werror -I/scratch/headers /tool/llama_attention.cc -L/app -Wl,-rpath,/app \
      -lggml-cuda -lggml -lggml-base -o /scratch/llama_attention
    sha256sum "$scratch/llama_attention"
    ;;
  replay)
    [[ $# == 2 && "$2" =~ ^[a-zA-Z0-9_-]+$ ]]
    sudo -n docker "${common[@]}" --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 \
      --entrypoint /scratch/llama_attention "$image" /scratch/packed "/scratch/$2"
    ;;
  trace)
    [[ $# == 1 ]]
    observer=/opt/nvidia/nsight-systems/2025.3.2
    [[ -x "$observer/bin/nsys" ]]
    sudo -n docker "${common[@]}" --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 \
      --mount "type=bind,src=$observer,dst=$observer,readonly" --entrypoint "$observer/bin/nsys" "$image" \
      profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=false \
      --output=/scratch/original-launches /scratch/llama_attention /scratch/packed /scratch/original-traced
    [[ -f "$scratch/original-traced/completion.json" ]]
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r=={"phase":"completed_after_backend_release","owners":4,"paid_waves":32,"completed_units":128}' "$scratch/original-traced/completion.json"
    cmp "$scratch/original-traced/first.f32" "$scratch/original-traced/repeat.f32"
    ;;
  *) exit 2 ;;
esac
