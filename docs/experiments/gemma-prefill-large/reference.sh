#!/bin/bash
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
# Reproduce the measured original-image thin-client commands; math is not rebuilt.
# VARIANT build | VARIANT screen NEW_OUTPUT UBATCH
# Exact b29 llama/ggml headers and input1/ids.i32 must exist in external scratch.
set -euo pipefail
[[ "$#" -ge 2 ]] || exit 2
variant="$1"
mode="$2"
shift 2
case "$variant" in
  26) scratch="$HOME/.local/share/jitllm/gemma-prefill"; model=gemma-4-26B-A4B-it-UD-Q4_K_M.gguf ;;
  31) scratch="$HOME/.local/share/jitllm/gemma31-prefill"; model=gemma-4-31B-it-UD-Q4_K_XL.gguf ;;
  *) exit 2 ;;
esac
source_root="${JITLLM_GEMMA_PREFILL_SOURCE_ROOT:-$HOME/src/jitLLM-wt/m3fixb}"
models="$HOME/.local/share/jitllm/reference-models"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=1g --mount "type=bind,src=$scratch,dst=/scratch" --mount "type=bind,src=$source_root/docs/experiments/gemma-prefill,dst=/tool,readonly" --mount "type=bind,src=$models,dst=/model,readonly")
if [[ "$mode" == build ]]; then
  [[ "$#" == 0 ]] || exit 2
  sudo -n docker "${common[@]}" --entrypoint g++ "$image" -std=c++23 -O2 -march=armv8-a -Wall -Wextra -Werror -I/scratch /tool/llama_prefill.cc -L/app -Wl,-rpath,/app -lllama -lggml -lggml-base -o /scratch/llama_prefill
else
  [[ "$mode" == screen && "$#" == 2 ]] || exit 2
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 2
  case "$2" in 128|256|512|1024|2048) ;; *) exit 2 ;; esac
  sudo -n docker "${common[@]}" --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 --entrypoint /scratch/llama_prefill "$image" "/model/$model" /scratch/input1/ids.i32 "/scratch/$1" screen "$2"
fi
