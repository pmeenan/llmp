#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# Runs one of the GGML operation study's scripts (ggml_ops_capture.py,
# ggml_ops_score.py) in the M0 reference container with the GPU, with
# exl3_run.sh's mounts and environment:
#
#   ggml_ops_run.sh SCRIPT [script options...]
#
# EXL_RUN, P0, CUDA, CUDA_MOUNT, EXT_CACHE and DOCKER_EXTRA as in
# exl3_run.sh; TUNE names the tuning cache in P0/exl3/ (a copy of a frozen
# one) for scripts that load the model.
set -eu
: "${EXL_RUN:?} ${P0:?} ${CUDA:=/usr/local/cuda-13.0} ${CUDA_MOUNT:=/usr/local/cuda-13.0} ${EXT_CACHE:=build-cache}"
script=$1
shift
exec sudo -n docker run --rm --gpus all --shm-size 2g --memory 32g --memory-swap 32g --network none \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$EXL_RUN,dst=/experiment,readonly" \
  --mount "type=bind,src=$P0,dst=/p0" \
  --mount "type=bind,src=$CUDA,dst=$CUDA_MOUNT,readonly" \
  -e "CUDA_HOME=$CUDA_MOUNT" -e TORCH_CUDA_ARCH_LIST=12.1 -e MAX_JOBS=8 \
  -e "TORCH_EXTENSIONS_DIR=/p0/$EXT_CACHE" -e PYTHONPATH=/experiment/source \
  -e CXX=/experiment/cxx-target -e CUDAHOSTCXX=/experiment/cxx-target \
  -e CUDA_DISABLE_PTX_JIT=1 -e OMP_NUM_THREADS=4 -e HOME=/tmp \
  ${TUNE:+-e EXLLAMAV3_TUNE_CACHE=/p0/exl3/$TUNE} \
  ${DOCKER_EXTRA:-} --entrypoint python3 llmp-exl3-reference:20260922 "/p0/harness/$script" "$@"
