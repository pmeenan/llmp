#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# Runs linear_reference.py for one fixture and arm in the M0 reference
# container (../exl3-reference, image llmp-exl3-reference:20260922), on
# upstream's extension built by the SDK's NVCC 13.4.92 (P0's bridge build,
# build-cache-nvcc134, SHA-256 aa8b9f16...) with the SDK's cuBLAS 13.8.0.4
# bind-mounted over PyTorch's, as P0's BP-F2 reference arm runs:
#
#   run_reference.sh FIXTURE ARM CACHE OUTPUT [linear_reference.py options...]
#
# FIXTURE is 4.0bpw or 4.5bpw, ARM G (EXL3_GEMV=0) or O (GEMV on), CACHE the
# tuning cache file under $P3 (created or extended by a run that misses a
# key, reused unchanged by one that does not), OUTPUT the result file under
# $P3. EXL_RUN is M0's reference directory (source, models, cxx-target), P0
# holds the bridge's CUDA tree and extension cache, SDK the llmpalooza SDK whose
# cuBLAS is mounted, P3 this experiment's directory outside Git.
set -eu
fixture=$1 arm=$2 cache=$3 output=$4
shift 4
: "${EXL_RUN:=$HOME/.local/share/llmp/exl3-reference-20260922}"
: "${P0:=$HOME/.local/share/llmp/p0-20260925}"
: "${P3:=$HOME/.local/share/llmp/p3a-20260927}"
: "${SDK:=$HOME/.local/share/llmp/sdk/aarch64-e0a0c85c42806fb1}"
here=$(cd "$(dirname "$0")" && pwd)
lib=$SDK/pkgs/cuda/usr/local/cuda-13.4/targets/sbsa-linux/lib
wheel=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
gemv=""
[ "$arm" = G ] && gemv="-e EXL3_GEMV=0"
# shellcheck disable=SC2086
exec sudo -n docker run --rm --gpus all --shm-size 2g --memory 48g --memory-swap 48g --network none \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$EXL_RUN,dst=/experiment,readonly" \
  --mount "type=bind,src=$P0,dst=/p0" \
  --mount "type=bind,src=$P3,dst=/p3" \
  --mount "type=bind,src=$here,dst=/harness,readonly" \
  --mount "type=bind,src=$P0/cuda-bridge-torch,dst=/cuda,readonly" \
  --mount "type=bind,src=$(readlink -f "$lib/libcublas.so.13"),dst=$wheel/libcublas.so.13,readonly" \
  --mount "type=bind,src=$(readlink -f "$lib/libcublasLt.so.13"),dst=$wheel/libcublasLt.so.13,readonly" \
  -e CUDA_HOME=/cuda -e TORCH_CUDA_ARCH_LIST=12.1 -e TORCH_EXTENSIONS_DIR=/p0/build-cache-nvcc134 \
  -e PYTHONPATH=/experiment/source -e CXX=/experiment/cxx-target -e CUDAHOSTCXX=/experiment/cxx-target \
  -e CUDA_DISABLE_PTX_JIT=1 -e OMP_NUM_THREADS=4 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -e EXL3_HGEMM_F16ACC=0 $gemv -e "EXLLAMAV3_TUNE_CACHE=/p3/$cache" \
  --entrypoint python3 llmp-exl3-reference:20260922 /harness/linear_reference.py \
  --model "/experiment/models/$fixture" --fixture "$fixture" --arm "$arm" --output "/p3/$output" "$@"
