#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# Runs one of P3's reference-side scripts (op_plan_record.py probe, op_tier_e.py,
# test_op_tier_e.py; also tierc_check.py and pack_run.py, which need numpy) in
# the M0 reference container, with P0's
# ggml_ops_run.sh mounts and environment and the SDK's cuBLAS 13.8.0.4
# bind-mounted over PyTorch's:
#
#   run_container.sh SCRIPT [script options...]
#
# SCRIPT is relative to this directory. Mounts: EXL_RUN (M0's reference
# directory) at /experiment, read-only; P0 (P0's working directory: tuning
# caches, extension caches, shim builds, reference runs, held-out IDs) at
# /p0; OUT (this experiment's raw outputs) at /out; docs/experiments at
# /harness, read-only. CUDA, CUDA_MOUNT and EXT_CACHE are as in
# ../backend-proof-p0/exl3_run.sh (default: the host's CUDA 13.0 and
# build-cache, the extension build P0's operation plan probe used). TUNE is
# the tuning cache's path in the container, if the script loads the model.
# GPUS="" runs without the GPU (tierc_check.py, pack_run.py).
# DOCKER_EXTRA adds docker options, split on spaces (read-only mounts of an
# artifact store or a native recording, GGML_OPS_LIB).
set -eu
: "${EXL_RUN:=$HOME/.local/share/llmp/exl3-reference-20260922}"
: "${P0:=$HOME/.local/share/llmp/p0-20260925}"
: "${OUT:=$HOME/.local/share/llmp/p3b-20260927}"
: "${SDK:=$HOME/.local/share/llmp/sdk/aarch64-e0a0c85c42806fb1}"
: "${CUDA:=/usr/local/cuda-13.0}" "${CUDA_MOUNT:=/usr/local/cuda-13.0}" "${EXT_CACHE:=build-cache}"
here=$(cd "$(dirname "$0")" && pwd)
script=$1
shift
lib=$SDK/pkgs/cuda/usr/local/cuda-13.4/targets/sbsa-linux/lib
wheel=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
# shellcheck disable=SC2086
exec sudo -n docker run --rm ${GPUS---gpus=all} --shm-size 2g --memory 48g --memory-swap 48g --network none \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$EXL_RUN,dst=/experiment,readonly" \
  --mount "type=bind,src=$P0,dst=/p0" \
  --mount "type=bind,src=$OUT,dst=/out" \
  --mount "type=bind,src=$(dirname "$here"),dst=/harness,readonly" \
  --mount "type=bind,src=$CUDA,dst=$CUDA_MOUNT,readonly" \
  --mount "type=bind,src=$(readlink -f "$lib/libcublas.so.13"),dst=$wheel/libcublas.so.13,readonly" \
  --mount "type=bind,src=$(readlink -f "$lib/libcublasLt.so.13"),dst=$wheel/libcublasLt.so.13,readonly" \
  -e "CUDA_HOME=$CUDA_MOUNT" -e TORCH_CUDA_ARCH_LIST=12.1 -e MAX_JOBS=8 \
  -e "TORCH_EXTENSIONS_DIR=/p0/$EXT_CACHE" -e PYTHONPATH=/experiment/source \
  -e CXX=/experiment/cxx-target -e CUDAHOSTCXX=/experiment/cxx-target \
  -e CUDA_DISABLE_PTX_JIT=1 -e OMP_NUM_THREADS=4 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  ${TUNE:+-e EXLLAMAV3_TUNE_CACHE=$TUNE} \
  ${DOCKER_EXTRA:-} --entrypoint python3 llmp-exl3-reference:20260922 "/harness/backend-proof-p3/$script" "$@"
