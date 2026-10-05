#!/bin/bash
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
[[ "$#" == 2 ]] || exit 2
case "$1" in all|none|norm_rope|norm_add|both) ;; *) exit 2 ;; esac
controller="${JITLLM_REFERENCE_CONTROLLER:-controller.so}"
case "$controller" in controller.so|controller-both.so) ;; *) exit 2 ;; esac
scratch="${JITLLM_REFERENCE_ROOT:-$HOME/.local/share/jitllm/gemma31-reference-fusions}"
quality="$HOME/.local/share/jitllm/gemma31-quality"
models="$HOME/.local/share/jitllm/reference-models"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
sudo -n docker run --rm --network none --read-only --user "$(id -u):$(id -g)" --device nvidia.com/gpu=all --tmpfs /tmp:rw,size=1g --mount "type=bind,src=$scratch,dst=/scratch" --mount "type=bind,src=$quality,dst=/quality,readonly" --mount "type=bind,src=$models,dst=/model,readonly" --env CUDA_DISABLE_PTX_JIT=1 --env "LD_PRELOAD=/scratch/$controller" --env "JITLLM_REFERENCE_FUSION_POLICY=$1" --env JITLLM_REFERENCE_TRACE=1 --entrypoint /quality/llama_quality "$image" /model/gemma-4-31B-it-UD-Q4_K_XL.gguf /quality/input1/ids.i32 "/scratch/$2" score 128
