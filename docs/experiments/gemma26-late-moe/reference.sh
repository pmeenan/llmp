#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
# External only. Every invocation uses the installed GPU supervisor.
set -euo pipefail
umask 077
scratch="$HOME/.local/share/llmp/gemma26-late-moe"
source_root="$HOME/src/llmp-wt/m3fixb"
previous="$HOME/.local/share/llmp/gemma26-packed-attention-c4"
sdk="$HOME/.local/share/llmp/sdk/aarch64-c09daba6ac31edee"
models="$HOME/.local/share/llmp/reference-models"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
tool="$source_root/docs/experiments/gemma26-late-moe"
[[ $# -ge 1 ]] || exit 2
output_name="$1"
cleanup() {
  local previous_status=$?
  trap - EXIT INT TERM
  if ! python3 -B "$tool/container_retire.py" "$scratch" "$output_name"; then
    exit 1
  fi
  exit "$previous_status"
}
owned_container() {
  [[ ! -e "$scratch/$output_name.cid" && ! -e "$scratch/$output_name-container-retired.json" ]]
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
}
case "$1" in
  build)
    [[ $# == 1 ]]
    [[ ! -e "$scratch" ]]
    mkdir -m 700 "$scratch"
    cp -R "$HOME/.local/share/llmp/gemma-reference-stock/ggml" "$scratch/ggml"
    cp -R "$previous/headers" "$scratch/headers"
    cp "$previous/llama_joined" "$scratch/untouched-client"
    cp "$previous/ids.i32" "$scratch/ids.i32"
    cp "$tool/capture.h" "$scratch/capture.h"
    cp "$tool/capture_main.cc" "$scratch/capture_main.cc"
    cp "$source_root/docs/experiments/gemma-joined-serving/llama_joined.cc" "$scratch/client.cc"
    printf '%s  %s\n' \
      01889d8c4a281110f611cedcab43235a9ac80bcc21abf094d982f14dd2112f61 "$scratch/untouched-client" \
      c80b73c85453ece170eda57bc0725cdef5b4e0af3d7cacf3d1a8bee0d1cfa19e "$scratch/client.cc" \
      b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610 "$scratch/ids.i32" | sha256sum --check
    PYTHONDONTWRITEBYTECODE=1 python3 "$tool/capture_controller.py" \
      "$scratch/ggml/src/ggml-cuda/ggml-cuda.cu" "$scratch/controller.cu"
    owned_container
    sudo -n docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
      --cidfile "$scratch/$output_name.cid" --name "llmp-gemma26-late-moe-$output_name" \
      --label llmp.observer=gemma26-late-moe \
      --device nvidia.com/gpu=all --tmpfs /tmp:rw,size=2g \
      --mount "type=bind,src=$scratch,dst=/scratch" \
      --mount "type=bind,src=$tool,dst=/tool,readonly" \
      --mount "type=bind,src=$source_root/src,dst=/native-src,readonly" \
      --mount "type=bind,src=$sdk,dst=$sdk,readonly" --entrypoint bash "$image" -c '
set -euo pipefail
nvcc="$1/pkgs/cuda/usr/local/cuda-13.4/bin/nvcc"
"$nvcc" --version
g++ --version
"$nvcc" -ccbin /usr/bin/g++ -shared -Xcompiler -fPIC -O3 -DNDEBUG -std=c++17 \
  -arch=sm_121a -use_fast_math -extended-lambda -DGGML_CUDA_USE_GRAPHS -DGGML_BACKEND_DL \
  -I/scratch/ggml/include -I/scratch/ggml/src -I/scratch/ggml/src/ggml-cuda \
  --cudart none /scratch/controller.cu -L/app \
  -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/nvidia/lib64 \
  -Xlinker -rpath -Xlinker /app -lggml-cuda -lggml-base -ldl \
  -l:libcudart.so.13 -l:libcublas.so.13 -l:libcuda.so.1 -o /scratch/controller.so
g++ -std=c++23 -O2 -march=armv8-a -Wall -Wextra -Werror \
  -I/scratch/headers/include -I/scratch/headers/ggml/include -Dmain=joined_main \
  -c /scratch/client.cc -o /scratch/client.o
g++ -std=c++23 -O2 -march=armv8-a -Wall -Wextra -Werror \
  -c /scratch/capture_main.cc -o /scratch/capture_main.o
g++ /scratch/client.o /scratch/capture_main.o -L/app -Wl,-rpath,/app \
  -lllama -lggml -lggml-base -ldl -o /scratch/capture-client
g++ -std=c++20 -O2 -march=armv8-a -I/native-src -c /native-src/base/sha256.cc -o /scratch/sha256.o
g++ -std=c++20 -O2 -march=armv8-a -I/native-src -c /native-src/base/check.cc -o /scratch/check.o
"$nvcc" -ccbin /usr/bin/g++ -O2 -std=c++20 -arch=sm_121a \
  -DGGML_CUDA_USE_GRAPHS -DGGML_BACKEND_DL -I/tool -I/native-src \
  -I/scratch/ggml/include -I/scratch/ggml/src -I/scratch/ggml/src/ggml-cuda \
  --cudart none /tool/oracle.cu /scratch/sha256.o /scratch/check.o -L/app \
  -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/nvidia/lib64 \
  -Xlinker -rpath -Xlinker /app -lggml-cuda -lggml-base -ldl \
  -l:libcudart.so.13 -l:libcublas.so.13 -l:libcuda.so.1 -o /scratch/oracle
sha256sum /scratch/{controller.cu,capture.h,controller.so,client.cc,capture_main.cc,capture-client} \
  /tool/{oracle.cu,graphs.h,inputs.h} /native-src/base/{sha256.cc,sha256.h,check.cc,check.h} /scratch/oracle \
  /app/lib{llama,ggml,ggml-base,ggml-cuda}.so
sha256sum /app/lib{llama,ggml,ggml-base,ggml-cuda}.so > /scratch/libraries.sha256
nm -D -C /app/libggml-cuda.so | grep -E "ggml_cuda_op_(topk_moe|moe_weighted_reduction)"
ldd -r /scratch/controller.so
ldd -r /scratch/capture-client
ldd -r /scratch/oracle
' bash "$sdk"
    ;;
  capture)
    [[ $# == 1 ]]
    # A separate authenticated pre-acquisition receipt and explicit root release
    # are prerequisites. This script does not grant model admission.
    owned_container
    sudo -n docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
      --cidfile "$scratch/$output_name.cid" --name "llmp-gemma26-late-moe-$output_name" \
      --label llmp.observer=gemma26-late-moe \
      --device nvidia.com/gpu=all --tmpfs /tmp:rw,size=1g \
      --mount "type=bind,src=$scratch,dst=/scratch" \
      --mount "type=bind,src=$models,dst=/model,readonly" \
      --env CUDA_DISABLE_PTX_JIT=1 --env LD_PRELOAD=/scratch/controller.so \
      --entrypoint /scratch/capture-client "$image" \
      /model/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf /scratch/capture 4 128 /scratch/ids.i32
    ;;
  oracle)
    [[ $# == 4 && "$2" =~ ^(route|reduce)$ && "$3" =~ ^(primitive|fused)$ && "$4" =~ ^[0-9a-f]{64}$ ]]
    output_name="$2-$3"
    owned_container
    sudo -n docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
      --cidfile "$scratch/$output_name.cid" --name "llmp-gemma26-late-moe-$output_name" \
      --label llmp.observer=gemma26-late-moe --device nvidia.com/gpu=all --tmpfs /tmp:rw,size=1g \
      --mount "type=bind,src=$scratch,dst=/scratch" --env CUDA_DISABLE_PTX_JIT=1 \
      --entrypoint /scratch/oracle "$image" \
      "/scratch/$2-inputs" "$4" "/scratch/original-$output_name" "$2" "$3"
    ;;
  cleanup)
    [[ $# == 1 ]]
    # Fixed names only: a supervised recovery step must authenticate the CID,
    # label and name before removing daemon descendants left by SIGKILL.
    python3 -B "$tool/container_retire.py" "$scratch" build
    python3 -B "$tool/container_retire.py" "$scratch" capture
    for output_name in route-primitive route-fused reduce-primitive reduce-fused; do
      python3 -B "$tool/container_retire.py" "$scratch" "$output_name"
    done
    ;;
  *) exit 2 ;;
esac
