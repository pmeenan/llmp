#!/bin/bash
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
scratch="$HOME/.local/share/jitllm/gemma26-prefill-profile"
source_root="$HOME/src/jitLLM-wt/m3gm31"
models="$HOME/.local/share/jitllm/reference-models"
nsight=/opt/nvidia/nsight-systems/2025.3.2
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=1g
        --mount "type=bind,src=$scratch,dst=/scratch"
        --mount "type=bind,src=$nsight,dst=$nsight,readonly"
        --mount "type=bind,src=$source_root/docs/experiments/gemma26-prefill-profile,dst=/tool,readonly")
case "${1:-}" in
  build)
    [[ $# == 1 ]]
    sudo -n docker "${common[@]}" --entrypoint g++ "$image" -std=c++23 -O2 -march=armv8-a \
      -Wall -Wextra -Werror -I/scratch/headers -I"$nsight/target-linux-sbsa-armv8/nvtx/include" \
      /tool/llama_prefill_profile.cc -L/app -Wl,-rpath,/app -lllama -lggml -lggml-base -ldl \
      -o /scratch/llama_prefill_profile
    sudo -n docker "${common[@]}" --entrypoint g++ "$image" -std=c++23 -O2 -march=armv8-a \
      -Wall -Wextra -Werror /tool/profile_child.cc -o /scratch/profile_child
    sudo -n docker "${common[@]}" --entrypoint sha256sum "$image" /app/libllama.so /app/libggml.so /app/libggml-base.so /app/libggml-cuda.so > "$scratch/libraries.sha256"
    sha256sum "$scratch/llama_prefill_profile" "$scratch/profile_child"
    ;;
  cleanup)
    [[ $# == 2 ]]
    python3 -B "$source_root/docs/experiments/gemma26-prefill-profile/container_retire.py" "$scratch" "$2"
    ;;
  control|trace)
    [[ $# == 2 && "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]
    [[ ! -e "$scratch/$2" && ! -e "$scratch/$2-completion.json" && ! -e "$scratch/$2.sqlite" && ! -e "$scratch/$2.nsys-rep" && ! -e "$scratch/$2.cid" && ! -e "$scratch/$2-container-retired.json" ]]
    output_name="$2"
    cleanup() {
      local previous_status=$?
      trap - EXIT INT TERM
      if ! python3 -B "$source_root/docs/experiments/gemma26-prefill-profile/container_retire.py" "$scratch" "$output_name"; then
        exit 1
      fi
      exit "$previous_status"
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    launch=(/scratch/profile_child "/scratch/$2-completion.json" /scratch/llama_prefill_profile
            /model/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf /scratch/ids.i32 "/scratch/$2")
    if [[ "$1" == trace ]]; then
      launch=("$nsight/bin/nsys" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
              --cuda-graph-trace=node --wait=all --stop-on-exit=true --kill=none
              --force-overwrite=false --export=sqlite "--output=/scratch/$2" "${launch[@]}")
    fi
    sudo -n docker "${common[@]}" --cidfile "$scratch/$output_name.cid" \
      --name "jitllm-gemma26-prefill-profile-$output_name" --label jitllm.observer=gemma26-prefill-profile \
      --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 \
      --mount "type=bind,src=$models,dst=/model,readonly" --entrypoint "${launch[0]}" "$image" "${launch[@]:1}"
    [[ -s "$scratch/$2-completion.json" ]]
    ;;
  *) exit 2 ;;
esac
