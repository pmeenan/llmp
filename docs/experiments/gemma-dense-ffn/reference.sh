#!/bin/bash
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
scratch="$HOME/.local/share/jitllm/gemma-dense-ffn"
source_root="$HOME/src/jitLLM-wt/m3gm31"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)"
        --tmpfs /tmp:rw,size=1g --mount "type=bind,src=$scratch,dst=/scratch"
        --mount "type=bind,src=$source_root/docs/experiments/gemma-dense-ffn,dst=/tool,readonly")
case "${1:-}" in
  build)
    [[ $# == 1 ]]
    sudo -n docker "${common[@]}" --device nvidia.com/gpu=all --entrypoint g++ "$image" -std=c++23 -O2 -march=armv8-a \
      -Wall -Wextra -Werror -I/scratch/headers /tool/llama_ffn.cc -L/app -Wl,-rpath,/app \
      -lggml-cuda -lggml -lggml-base -o /scratch/llama_ffn
    sha256sum "$scratch/llama_ffn"
    ;;
  replay)
    [[ $# == 3 && "$2" =~ ^[a-zA-Z0-9_-]+$ && ( "$3" == A || "$3" == A0 ) ]]
    sudo -n docker "${common[@]}" --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 \
      --entrypoint /scratch/llama_ffn "$image" /scratch/operands "/scratch/$2" "$3"
    ;;
  trace)
    [[ $# == 2 && ( "$2" == A || "$2" == A0 ) ]]
    observer=/opt/nvidia/nsight-systems/2025.3.2
    [[ -x "$observer/bin/nsys" ]]
    sudo -n docker "${common[@]}" --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 \
      --mount "type=bind,src=$observer,dst=$observer,readonly" --entrypoint "$observer/bin/nsys" "$image" \
      profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=false \
      --output="/scratch/original-launches-$2" /scratch/llama_ffn /scratch/operands "/scratch/original-traced-$2" "$2"
    [[ -f "$scratch/original-traced-$2/completion.json" ]]
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r=={"phase":"completed_after_backend_release","paid_chains":32}' "$scratch/original-traced-$2/completion.json"
    cmp "$scratch/original-traced-$2/first.f32" "$scratch/original-traced-$2/repeat.f32"
    ;;
  *) exit 2 ;;
esac
