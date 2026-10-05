#!/bin/bash
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root="$HOME/.local/share/jitllm/gemma-moe-dispatch/reference-gates"
stock="$HOME/.local/share/jitllm/gemma-reference-stock"
quality="$HOME/.local/share/jitllm/gemma-quality"
models="$HOME/.local/share/jitllm/reference-models"
sdk="$HOME/.local/share/jitllm/sdk/aarch64-c09daba6ac31edee"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
[[ ! -e "$root" ]] || exit 2
mkdir "$root"
rsync -rlpc "$stock/ggml/" "$root/ggml/"
python3 "$(dirname "$0")/gate_trace.py" "$stock/controller.cu" "$root/controller.cu"
sudo -n docker run --rm --network none --read-only --user "$(id -u):$(id -g)" --device nvidia.com/gpu=all --tmpfs /tmp:rw,size=2g --mount "type=bind,src=$root,dst=/scratch" --mount "type=bind,src=$sdk,dst=$sdk,readonly" --entrypoint bash "$image" -c '
set -euo pipefail
nvcc="$1/pkgs/cuda/usr/local/cuda-13.4/bin/nvcc"
"$nvcc" --version
"$nvcc" -ccbin /usr/bin/g++ -shared -Xcompiler -fPIC -O3 -DNDEBUG -std=c++17 -arch=sm_121a -use_fast_math -extended-lambda -DGGML_CUDA_USE_GRAPHS -DGGML_BACKEND_DL -I/scratch/ggml/include -I/scratch/ggml/src -I/scratch/ggml/src/ggml-cuda --cudart none /scratch/controller.cu -L/app -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/nvidia/lib64 -Xlinker -rpath -Xlinker /app -lggml-cuda -lggml-base -l:libcudart.so.13 -l:libcublas.so.13 -l:libcuda.so.1 -o /scratch/controller.so
sha256sum /scratch/controller.cu /scratch/controller.so /app/libggml-cuda.so
ldd -r /scratch/controller.so
' bash "$sdk"
sudo -n docker run --rm --network none --read-only --user "$(id -u):$(id -g)" --device nvidia.com/gpu=all --tmpfs /tmp:rw,size=1g --mount "type=bind,src=$root,dst=/scratch" --mount "type=bind,src=$quality,dst=/quality,readonly" --mount "type=bind,src=$models,dst=/model,readonly" --env CUDA_DISABLE_PTX_JIT=1 --env LD_PRELOAD=/scratch/controller.so --env JITLLM_REFERENCE_FUSION_POLICY=all --env JITLLM_REFERENCE_TRACE=1 --entrypoint /quality/llama_quality "$image" /model/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf /quality/input1/ids.i32 /scratch/all-first score 128
cmp "$root/all-first/logits.f32" "$quality/reference128-first/logits.f32"
sha256sum "$root/all-first/logits.f32"
