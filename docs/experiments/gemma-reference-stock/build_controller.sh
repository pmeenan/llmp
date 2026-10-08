#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
scratch="${LLMP_REFERENCE_ROOT:-$HOME/.local/share/llmp/gemma-reference-stock}"
sdk="$HOME/.local/share/llmp/sdk/aarch64-c09daba6ac31edee"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
if [[ ! -e "$scratch/controller.cu" ]]; then python3 "$scratch/patch_controller.py" "$scratch/ggml/src/ggml-cuda/ggml-cuda.cu" "$scratch/controller.cu"; fi
printf '%s  %s\n' fce9dd32edde4891ed180a4081d155ef921a9d1b50e619375caac58e2f6ba291 "$scratch/controller.cu" | sha256sum --check --status
sudo -n docker run --rm --network none --read-only --user "$(id -u):$(id -g)" --device nvidia.com/gpu=all --tmpfs /tmp:rw,size=2g --mount "type=bind,src=$scratch,dst=/scratch" --mount "type=bind,src=$sdk,dst=$sdk,readonly" --entrypoint bash "$image" -c '
set -euo pipefail
nvcc="$1/pkgs/cuda/usr/local/cuda-13.4/bin/nvcc"
"$nvcc" --version
"$nvcc" -ccbin /usr/bin/g++ -shared -Xcompiler -fPIC -O3 -DNDEBUG -std=c++17 -arch=sm_121a -use_fast_math -extended-lambda -DGGML_CUDA_USE_GRAPHS -DGGML_BACKEND_DL -I/scratch/ggml/include -I/scratch/ggml/src -I/scratch/ggml/src/ggml-cuda --cudart none /scratch/controller.cu -L/app -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/nvidia/lib64 -Xlinker -rpath -Xlinker /app -lggml-cuda -lggml-base -l:libcudart.so.13 -l:libcublas.so.13 -l:libcuda.so.1 -o /scratch/controller.so
sha256sum /scratch/controller.cu /scratch/controller.so /app/libggml-cuda.so
ldd -r /scratch/controller.so
' bash "$sdk"
