#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
scratch="$HOME/.local/share/llmp/gemma26-dispatch-observation"
source_root="$HOME/src/llmp-wt/m3fixb"
previous="$HOME/.local/share/llmp/gemma26-packed-attention-c4"
sdk="$HOME/.local/share/llmp/sdk/aarch64-c09daba6ac31edee"
models="$HOME/.local/share/llmp/reference-models"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
tool="$source_root/docs/experiments/gemma26-dispatch-observation"
[[ $# == 1 ]] || exit 2
case "$1" in
  build)
    [[ ! -e "$scratch" ]]
    mkdir -m 700 "$scratch"
    cp -R "$HOME/.local/share/llmp/gemma-reference-stock/ggml" "$scratch/ggml"
    cp "$previous/llama_joined" "$scratch/llama_joined"
    cp "$previous/ids.i32" "$scratch/ids.i32"
    python3 "$tool/observe_controller.py" "$scratch/ggml/src/ggml-cuda/ggml-cuda.cu" "$scratch/controller.cu"
    printf '%s  %s\n' \
      01889d8c4a281110f611cedcab43235a9ac80bcc21abf094d982f14dd2112f61 "$scratch/llama_joined" \
      b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610 "$scratch/ids.i32" \
      4c042fcd8a0f72c4dcb1cb4cffe018f45e309175bca073e11e86e4f732e534b2 "$scratch/controller.cu" | sha256sum --check
    sudo -n docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
      --device nvidia.com/gpu=all --tmpfs /tmp:rw,size=2g \
      --mount "type=bind,src=$scratch,dst=/scratch" \
      --mount "type=bind,src=$sdk,dst=$sdk,readonly" --entrypoint bash "$image" -c '
set -euo pipefail
nvcc="$1/pkgs/cuda/usr/local/cuda-13.4/bin/nvcc"
"$nvcc" --version
"$nvcc" -ccbin /usr/bin/g++ -shared -Xcompiler -fPIC -O3 -DNDEBUG -std=c++17 \
  -arch=sm_121a -use_fast_math -extended-lambda -DGGML_CUDA_USE_GRAPHS -DGGML_BACKEND_DL \
  -I/scratch/ggml/include -I/scratch/ggml/src -I/scratch/ggml/src/ggml-cuda \
  --cudart none /scratch/controller.cu -L/app \
  -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/nvidia/lib64 \
  -Xlinker -rpath -Xlinker /app -lggml-cuda -lggml-base \
  -l:libcudart.so.13 -l:libcublas.so.13 -l:libcuda.so.1 -o /scratch/controller.so
sha256sum /scratch/controller.cu /scratch/controller.so /app/lib{llama,ggml,ggml-base,ggml-cuda}.so
ldd -r /scratch/controller.so
' bash "$sdk"
    ;;
  run)
    printf '%s  %s\n' \
      01889d8c4a281110f611cedcab43235a9ac80bcc21abf094d982f14dd2112f61 "$scratch/llama_joined" \
      b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610 "$scratch/ids.i32" \
      4c042fcd8a0f72c4dcb1cb4cffe018f45e309175bca073e11e86e4f732e534b2 "$scratch/controller.cu" | sha256sum --check
    sudo -n docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
      --device nvidia.com/gpu=all --tmpfs /tmp:rw,size=1g \
      --mount "type=bind,src=$scratch,dst=/scratch" \
      --mount "type=bind,src=$models,dst=/model,readonly" \
      --env CUDA_DISABLE_PTX_JIT=1 --env LD_PRELOAD=/scratch/controller.so \
      --entrypoint /scratch/llama_joined "$image" \
      /model/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf /scratch/observed 4 128 /scratch/ids.i32
    # Success returns only after the unchanged C API client's explicit batch,
    # context, model and public backend free calls. This is not a new device fence.
    printf '%s\n' 'OBSERVATION_CLIENT_EXIT0 explicit_public_backend_teardown=source_verified'
    ;;
  *) exit 2 ;;
esac
