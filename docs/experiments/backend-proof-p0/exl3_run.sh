#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# Runs one arm of the EXL3 held-out trajectory (exl3_heldout.py) in the M0
# reference container (../exl3-reference, image llmp-exl3-reference:20260922):
#
#   exl3_run.sh NAME FIXTURE TUNE_CACHE [exl3_heldout.py options...]
#
# EXL_RUN is M0's reference directory (source, models, cxx-target); P0 holds
# this harness in harness/, the held-out IDs, pins.json in exl3/ and the
# outputs. CUDA is the toolkit that compiles the extension, mounted at
# CUDA_MOUNT: the reference mounts the host's /usr/local/cuda-13.0 at the
# same path with EXT_CACHE=build-cache, as M0 did; the bridge mounts the
# SDK's CUDA 13.4 (README) at /cuda with its own EXT_CACHE, so the two
# builds never share a cache. The tuning cache P0/exl3/TUNE_CACHE is created
# when missing (a tuning run) and reused when present (a frozen run).
# DOCKER_EXTRA adds docker options, split on spaces: the cuBLAS-substitution
# arm mounts the SDK's libcublas.so.13 and libcublasLt.so.13 (13.8.0.4) over
# PyTorch's copies, so the process maps only those (README).
set -eu
: "${EXL_RUN:?} ${P0:?} ${CUDA:=/usr/local/cuda-13.0} ${CUDA_MOUNT:=/usr/local/cuda-13.0} ${EXT_CACHE:=build-cache}"
name=$1 fixture=$2 tune=$3
shift 3
exec sudo -n docker run --rm --gpus all --shm-size 2g --memory 32g --memory-swap 32g --network none \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$EXL_RUN,dst=/experiment,readonly" \
  --mount "type=bind,src=$P0,dst=/p0" \
  --mount "type=bind,src=$CUDA,dst=$CUDA_MOUNT,readonly" \
  -e "CUDA_HOME=$CUDA_MOUNT" -e TORCH_CUDA_ARCH_LIST=12.1 -e MAX_JOBS=8 \
  -e "TORCH_EXTENSIONS_DIR=/p0/$EXT_CACHE" -e PYTHONPATH=/experiment/source \
  -e CXX=/experiment/cxx-target -e CUDAHOSTCXX=/experiment/cxx-target \
  -e CUDA_DISABLE_PTX_JIT=1 -e OMP_NUM_THREADS=4 -e HOME=/tmp \
  -e "EXLLAMAV3_TUNE_CACHE=/p0/exl3/$tune" \
  ${DOCKER_EXTRA:-} --entrypoint python3 llmp-exl3-reference:20260922 /p0/harness/exl3_heldout.py \
  --model "/experiment/models/$fixture" --ids /p0/heldout-ids.i64le --pins /p0/exl3/pins.json \
  --output "/p0/exl3/$name" "$@"
