#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds the bridge's census harness (fp16_census.cc) on a Spark against an
# existing FP16 bridge build (../backend-proof-p0/bridge/bridge.sh), with the
# compiler, flags and link line that build used for fp16_reference, so the
# llama.cpp and GGML objects are the bridge's own:
#   census_bridge.sh SDK LLAMA_SOURCE CUDA BRIDGE_BUILD OUTPUT
set -eu
SDK=$1 SOURCE=$2 CUDA=$3 BUILD=$4 OUT=$5
GCC=$SDK/gcc/aarch64-linux-gnu/lib/gcc/aarch64-linux-gnu/16
HERE=$(cd "$(dirname "$0")" && pwd)
CXX="$SDK/bin/clang++ --target=aarch64-linux-gnu -march=armv8-a --gcc-install-dir=$GCC"
$CXX -O3 -DNDEBUG -std=gnu++23 -DGGML_USE_CPU -DGGML_USE_CUDA \
  -I"$SOURCE/include" -I"$SOURCE/ggml/include" -I"$SOURCE/src" -I"$CUDA/targets/sbsa-linux/include" \
  -c "$HERE/fp16_census.cc" -o "$OUT.o"
$CXX -O3 -DNDEBUG -fuse-ld=bfd -static-libstdc++ -static-libgcc "$OUT.o" -o "$OUT" \
  -Wl,-rpath,"$CUDA/lib64" \
  "$BUILD/llama.cpp/src/libllama.a" "$BUILD/llama.cpp/ggml/src/libggml.a" \
  "$BUILD/llama.cpp/ggml/src/libggml-cpu.a" "$BUILD/llama.cpp/ggml/src/ggml-cuda/libggml-cuda.a" \
  "$BUILD/llama.cpp/ggml/src/libggml-base.a" -lm \
  "$CUDA/lib64/libcudart.so" "$CUDA/lib64/libcublas.so" "$CUDA/lib64/libcublasLt.so" \
  "$CUDA/targets/sbsa-linux/lib/stubs/libcuda.so" -ldl /usr/lib/aarch64-linux-gnu/librt.a \
  -L"$CUDA/targets/sbsa-linux/lib/stubs" -L"$CUDA/targets/sbsa-linux/lib" \
  -lcudadevrt -lcudart_static -lrt -lpthread -ldl
rm -f "$OUT.o"
