#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
# Native Spark build using extracted dependencies; no package installation.
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [[ $(uname -m) != aarch64 ]]; then
  echo 'This recipe runs on the ARM64 Spark.' >&2
  exit 1
fi
mkdir -p "$LLMP_NET_ROOT"
python3 "$(dirname "$0")/prepare-packages.py" --root "$LLMP_NET_ROOT"
if [[ ! -d "$LLMP_NET_ROOT/nccl" ]]; then
  git clone --branch v2.30.7-1 --depth 1 https://github.com/NVIDIA/nccl.git "$LLMP_NET_ROOT/nccl"
fi
test "$(git -C "$LLMP_NET_ROOT/nccl" rev-parse HEAD)" = 73cf112295c33aee2b895f329f592f2a9b4b0f97
if [[ ! -d "$LLMP_NET_ROOT/nccl-tests" ]]; then
  git clone https://github.com/NVIDIA/nccl-tests.git "$LLMP_NET_ROOT/nccl-tests"
  git -C "$LLMP_NET_ROOT/nccl-tests" checkout --detach b4d5beebca8a76cf01335f724d154b9b9d394d96
fi
test "$(git -C "$LLMP_NET_ROOT/nccl-tests" rev-parse HEAD)" = b4d5beebca8a76cf01335f724d154b9b9d394d96
test -z "$(git -C "$LLMP_NET_ROOT/nccl" status --porcelain)"
test -z "$(git -C "$LLMP_NET_ROOT/nccl-tests" status --porcelain)"
export NVCC_PREPEND_FLAGS=-Xcompiler=-march=armv8.2-a
CXXFLAGS=-march=armv8.2-a make -C "$LLMP_NET_ROOT/nccl" -j8 src.build \
 CUDA_HOME=/usr/local/cuda NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121'
export NVCC_APPEND_FLAGS=-lmpi_cxx
make -C "$LLMP_NET_ROOT/nccl-tests/src" -j4 \
 BUILDDIR="$LLMP_NET_ROOT/nccl-tests/build" MPI=1 CUDA_HOME=/usr/local/cuda \
 NCCL_HOME="$LLMP_NET_ROOT/nccl/build" MPI_HOME="$OPAL_LIBDIR/openmpi" \
 CXXFLAGS='-std=c++17 -DNCCL_OS_LINUX -O3 -g -march=armv8.2-a' \
 NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121' \
 "$LLMP_NET_ROOT/nccl-tests/build/all_reduce_perf" \
 "$LLMP_NET_ROOT/nccl-tests/build/all_gather_perf" \
 "$LLMP_NET_ROOT/nccl-tests/build/sendrecv_perf"
mkdir -p "$LLMP_NET_ROOT/nccl-runtime/lib" "$LLMP_NET_ROOT/nccl-tests-runtime"
cp -a "$LLMP_NET_ROOT"/nccl/build/lib/libnccl.so* "$LLMP_NET_ROOT/nccl-runtime/lib/"
cp "$LLMP_NET_ROOT"/nccl-tests/build/{all_reduce,all_gather,sendrecv}_perf "$LLMP_NET_ROOT/nccl-tests-runtime/"
g++ -std=c++23 -O2 -march=armv8.2-a -I/usr/local/cuda/include \
 "$(dirname "$0")/cuda-capabilities.cc" -lcuda -o "$LLMP_NET_ROOT/cuda-capabilities"
