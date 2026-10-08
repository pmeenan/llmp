#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
# build|run NEW_OUTPUT. External oracle only; installed GPU supervisor required.
set -euo pipefail
[[ "$#" -ge 1 ]] || exit 2
root="${LLMP_GEMMA_MOE_ORACLE_ROOT:-$HOME/.local/share/llmp/gemma-moe-oracle}"
source_root="${LLMP_GEMMA_MOE_SOURCE_ROOT:-$HOME/src/llmp-wt/m3fixb}"
ggml="$HOME/.local/share/llmp/gemma-reference-stock/ggml"
sdk="$HOME/.local/share/llmp/sdk/aarch64-c09daba6ac31edee"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)" --device nvidia.com/gpu=all --tmpfs /tmp:rw,size=1g --mount "type=bind,src=$root,dst=/scratch" --mount "type=bind,src=$ggml,dst=/ggml,readonly" --mount "type=bind,src=$source_root/docs/experiments/gemma-moe-primitives,dst=/tool,readonly")
case "$1" in
  build)
    [[ "$#" == 1 ]] || exit 2
    sudo -n docker "${common[@]}" --mount "type=bind,src=$sdk,dst=$sdk,readonly" --entrypoint bash "$image" -c '
set -euo pipefail
nm -D -C /app/libggml-cuda.so | grep -E "ggml_cuda_op_(topk_moe|moe_weighted_reduction)"
"$1/pkgs/cuda/usr/local/cuda-13.4/bin/nvcc" -ccbin /usr/bin/g++ -O2 -std=c++17 -arch=sm_121a -DGGML_CUDA_USE_GRAPHS -DGGML_BACKEND_DL -I/ggml/include -I/ggml/src -I/ggml/src/ggml-cuda --cudart none /tool/oracle.cu -L/app -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/nvidia/lib64 -Xlinker -rpath -Xlinker /app -lggml-cuda -lggml-base -l:libcudart.so.13 -l:libcublas.so.13 -l:libcuda.so.1 -o /scratch/oracle
sha256sum /tool/oracle.cu /scratch/oracle /app/libggml-cuda.so
' bash "$sdk"
    ;;
  run)
    [[ "$#" == 2 ]] || exit 2
    sudo -n docker "${common[@]}" --env CUDA_DISABLE_PTX_JIT=1 --entrypoint /scratch/oracle "$image" /scratch/native "/scratch/$2"
    ;;
  *) exit 2 ;;
esac
