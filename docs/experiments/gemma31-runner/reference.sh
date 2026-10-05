#!/bin/bash
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
# quality build|prepare|score|score-unfused INPUT NEW_OUTPUT [CHUNK]
# prefill build|prepare|screen [NEW_OUTPUT CHUNK]
# Exact b29 headers must already be installed in each external scratch directory.
set -euo pipefail
[[ "$#" -ge 2 ]] || exit 2
kind="$1"
mode="$2"
shift 2
case "$kind" in quality|prefill) ;; *) exit 2 ;; esac
scratch="$HOME/.local/share/jitllm/gemma31-$kind"
source_root="${JITLLM_GEMMA31_SOURCE_ROOT:-$HOME/src/jitLLM-wt/m3gm31}"
models="$HOME/.local/share/jitllm/reference-models"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,size=1g --mount "type=bind,src=$scratch,dst=/scratch" --mount "type=bind,src=$source_root/docs/experiments/gemma-$kind,dst=/tool,readonly" --mount "type=bind,src=$models,dst=/model,readonly")
if [[ "$mode" == build ]]; then
  [[ "$#" == 0 ]] || exit 2
  sudo -n docker "${common[@]}" --entrypoint g++ "$image" -std=c++23 -O2 -march=armv8-a -Wall -Wextra -Werror -I/scratch "/tool/llama_$kind.cc" -L/app -Wl,-rpath,/app -lllama -lggml -lggml-base -o "/scratch/llama_$kind"
elif [[ "$kind" == quality ]]; then
  [[ "$#" -ge 2 && "$#" -le 3 ]] || exit 2
  case "$mode" in prepare|score|score-unfused) ;; *) exit 2 ;; esac
  diagnostic_env=()
  if [[ "$mode" == score-unfused ]]; then diagnostic_env=(--env GGML_CUDA_DISABLE_FUSION=1); fi
  sudo -n docker "${common[@]}" "${diagnostic_env[@]}" --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 --entrypoint /scratch/llama_quality "$image" /model/gemma-4-31B-it-UD-Q4_K_XL.gguf "/scratch/$1" "/scratch/$2" "$mode" "${3:-128}"
elif [[ "$mode" == prepare ]]; then
  [[ "$#" == 0 ]] || exit 2
  sudo -n docker "${common[@]}" --mount "type=bind,src=$HOME/.local/share/jitllm/m3lc/prompts/ppl.txt,dst=/corpus.txt,readonly" --entrypoint /scratch/llama_prefill "$image" /model/gemma-4-31B-it-UD-Q4_K_XL.gguf /corpus.txt /scratch/input1 prepare 512
else
  [[ "$mode" == screen && "$#" == 2 ]] || exit 2
  sudo -n docker "${common[@]}" --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 --entrypoint /scratch/llama_prefill "$image" /model/gemma-4-31B-it-UD-Q4_K_XL.gguf /scratch/input1/ids.i32 "/scratch/$1" screen "$2"
fi
