#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# Builds the reference-only GGML operation shim (CMakeLists.txt) on a Spark,
# inside the EXL3 reference container (image llmp-exl3-reference:20260922),
# so it loads into the same process as upstream ExLlamaV3:
#
#   build.sh NAME
#
# EXL_RUN is M0's reference directory (for cxx-target, the container's g++
# with -march=armv8-a); P0 holds this harness in harness/ and receives the
# build in ggmlops/NAME; LLAMA_SOURCE is the pinned llama.cpp source
# (b29c606e2), of which only ggml/ is built, unmodified; SDK is llmpalooza's
# aarch64 SDK, used only for its CMake (the container has none). CUDA is the
# toolkit mounted at CUDA_MOUNT whose NVCC compiles the kernels: by default the
# host's CUDA 13.0 at the same path, as the reference's extension build uses;
# the bridge variant mounts the SDK's NVCC 13.4 tree at /cuda (exl3_run.sh).
# The container's GCC 13.3 is the C, C++ and CUDA host compiler. GGML's own
# CMake sets the CUDA flags (-use_fast_math -extended-lambda
# -compress-mode=...); CMAKE_CUDA_ARCHITECTURES=121-real, which GGML maps to
# 121a-real as in the bridge. Options: shared libraries, CPU backend without
# OpenMP, no NCCL, no CUDA graphs, no tests or examples, Release.
# compile_commands.json in the build records every compiler invocation.
set -eu
: "${EXL_RUN:?} ${P0:?} ${LLAMA_SOURCE:?} ${SDK:?} ${CUDA:=/usr/local/cuda-13.0} ${CUDA_MOUNT:=/usr/local/cuda-13.0}"
name=$1
mkdir -p "$P0/ggmlops/$name"
exec sudo -n docker run --rm --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$EXL_RUN,dst=/experiment,readonly" \
  --mount "type=bind,src=$P0,dst=/p0" \
  --mount "type=bind,src=$LLAMA_SOURCE,dst=/llama.cpp,readonly" \
  --mount "type=bind,src=$SDK/cmake,dst=/sdk-cmake,readonly" \
  --mount "type=bind,src=$CUDA,dst=$CUDA_MOUNT,readonly" \
  -e HOME=/tmp --entrypoint sh llmp-exl3-reference:20260922 -c "
set -eu
/sdk-cmake/bin/cmake -S /p0/harness/ggml_shim -B /p0/ggmlops/$name -G Ninja -DCMAKE_MAKE_PROGRAM=/usr/local/bin/ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_C_FLAGS=-march=armv8-a -DCMAKE_CXX_FLAGS=-march=armv8-a \
  -DCMAKE_CUDA_COMPILER=$CUDA_MOUNT/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/experiment/cxx-target \
  -DCMAKE_CUDA_ARCHITECTURES=121-real -DCUDAToolkit_ROOT=$CUDA_MOUNT \
  -DGGML_SOURCE=/llama.cpp/ggml -DGGML_SHIM_SOURCE_ID=llama.cpp-b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml \
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=OFF -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF \
  -DGGML_CUDA_GRAPHS=OFF -DGGML_OPENMP=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
/sdk-cmake/bin/cmake --build /p0/ggmlops/$name --target ggml_shim
$CUDA_MOUNT/bin/nvcc --version
"
