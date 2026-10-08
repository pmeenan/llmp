#!/bin/sh
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# One timing session of the EXL3 kernel cases (the 176 cases of the M0
# reference, ../exl3-reference/measure.py unchanged) with two arms:
#
#   timing_session.sh SESSION ARM_A ENV_A ARM_B ENV_B
#
# ENV_x is the EXL3_* settings of that arm as a JSON object, written into a
# copy of M0's protocol as its "optimized" profile. Each arm first tunes in
# one discarded process per case set; its tuning cache is then frozen and
# every timed block runs on a copy. An arm with the same settings as arm A
# (an A/A session) reuses A's caches, so both arms run one plan; with
# FROZEN_FROM=DIR, an arm whose DIR/tune-ARM-SET.bin exists reuses that cache
# instead of tuning (matched by arm name only: the caller keeps the name
# bound to the same settings). The timed blocks run in ORDER (default the
# approved primary order "A1 B1 B2 A2 B3 A3 A4 B4"; a confirmation passes the
# mirrored "B1 A1 A2 B2 A3 B3 B4 A4"; the early sessions aa and gemv passed
# "A1 B1 B2 A2"), each block a fresh
# process per case set (4.0 bpw real, 4.5 bpw
# real, synthetic). One discarded warm-up process (arm A, 4.0 bpw real) runs
# first, so no timed block starts on an idle GPU. The GPU's SM clock and temperature are recorded at every
# block boundary. Needs EXL_RUN and P0 as for exl3_run.sh; output goes to
# P0/timing/SESSION. DOCKER_EXTRA adds docker options to every process (the
# sessions from c1 on bind-mount the SDK's cuBLAS 13.8.0.4 over
# PyTorch's, as exl3_run.sh's substitution arm does). CUDA, CUDA_MOUNT and
# EXT_CACHE choose the toolkit that builds the extension, as in exl3_run.sh:
# the NVCC 13.4.92 reference mounts the SDK's tree (cuda-bridge-torch) at
# /cuda with EXT_CACHE=build-cache-nvcc134. MEASURE names another harness
# with measure.py's arguments in its place, as the container sees it
# (BP-F2's P3-entry case set: ../backend-proof-p3/bpf2_measure.py, with
# BPF2_CASES passed through DOCKER_EXTRA; every process gets its case set as
# BPF2_SET). timing_stats.py summarizes a session.
set -eu
: "${EXL_RUN:?} ${P0:?} ${CUDA:=/usr/local/cuda-13.0} ${CUDA_MOUNT:=/usr/local/cuda-13.0} ${EXT_CACHE:=build-cache}"
session=$1
out=$P0/timing/$session
mkdir -p "$out"
[ -z "$(ls -A "$out")" ] || { echo "session directory not empty: $out" >&2; exit 1; }

protocol() {  # ARM ENV
  python3 - "$EXL_RUN/experiment/protocol.json" "$out/protocol-$1.json" "$2" <<'EOF'
import json, sys
protocol = json.load(open(sys.argv[1]))
protocol["profiles"]["optimized"] = json.loads(sys.argv[3])
open(sys.argv[2], "w").write(json.dumps(protocol, indent=2) + "\n")
EOF
}

observe() {
  echo "$(date -u +%FT%TZ) $1 $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader)" >> "$out/observations.txt"
}

measure() {  # ARM BLOCK SET CACHE
  arm=$1 block=$2 set=$3 cache=$4
  case $set in
    real-40) model=4.0bpw mode=kernels ;;
    real-45) model=4.5bpw mode=kernels ;;
    synthetic) model=4.0bpw mode=synthetic ;;
  esac
  sudo -n docker run --rm --gpus all --shm-size 2g --memory 32g --memory-swap 32g --network none \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$EXL_RUN,dst=/experiment,readonly" \
    --mount "type=bind,src=$P0,dst=/p0" \
    --mount "type=bind,src=$CUDA,dst=$CUDA_MOUNT,readonly" ${DOCKER_EXTRA:-} \
    -e "CUDA_HOME=$CUDA_MOUNT" -e TORCH_CUDA_ARCH_LIST=12.1 -e MAX_JOBS=8 \
    -e "TORCH_EXTENSIONS_DIR=/p0/$EXT_CACHE" -e PYTHONPATH=/experiment/source \
    -e CXX=/experiment/cxx-target -e CUDAHOSTCXX=/experiment/cxx-target \
    -e CUDA_DISABLE_PTX_JIT=1 -e OMP_NUM_THREADS=4 -e HOME=/tmp \
    -e "EXLLAMAV3_TUNE_CACHE=/p0/timing/$session/$cache" -e "BPF2_SET=$set" \
    --entrypoint python3 llmp-exl3-reference:20260922 "${MEASURE:-/experiment/experiment/measure.py}" \
    --model "/experiment/models/$model" --mode "$mode" --protocol "/p0/timing/$session/protocol-$arm.json" \
    --pins /experiment/experiment/pins.json --output "/p0/timing/$session/$arm-$block-$set" \
    > "$out/$arm-$block-$set.log" 2>&1
}

A=$2 B=$4
protocol "$A" "$3"
protocol "$B" "$5"
for arm in "$A" "$B"; do
  for set in real-40 real-45 synthetic; do
    if [ -n "${FROZEN_FROM:-}" ] && [ -f "$FROZEN_FROM/tune-$arm-$set.bin" ]; then
      cp "$FROZEN_FROM/tune-$arm-$set.bin" "$out/tune-$arm-$set.bin"
    elif [ "$arm" = "$B" ] && [ "$5" = "$3" ]; then
      cp "$out/tune-$A-$set.bin" "$out/tune-$arm-$set.bin"
    else
      measure "$arm" tune "$set" "tune-$arm-$set.bin"
    fi
  done
done
cp "$out/tune-$A-real-40.bin" "$out/frozen-$A-warm-real-40.bin"
measure "$A" warm real-40 "frozen-$A-warm-real-40.bin"
cmp -s "$out/tune-$A-real-40.bin" "$out/frozen-$A-warm-real-40.bin" || {
  echo "tuning cache changed during warm-up" >> "$out/observations.txt"
  exit 1
}
for step in ${ORDER:-A1 B1 B2 A2 B3 A3 A4 B4}; do
  case $step in
    A[1-9]) arm=$A ;;
    B[1-9]) arm=$B ;;
    *) echo "bad ORDER token: $step" >&2; exit 1 ;;
  esac
  set -- "$arm" "${step#?}"
  observe "before $1-$2"
  for s in real-40 real-45 synthetic; do
    cp "$out/tune-$1-$s.bin" "$out/frozen-$1-$2-$s.bin"
    measure "$1" "$2" "$s" "frozen-$1-$2-$s.bin"
    cmp -s "$out/tune-$1-$s.bin" "$out/frozen-$1-$2-$s.bin" || {
      echo "tuning cache changed: $1-$2-$s" >> "$out/observations.txt"
      exit 1
    }
  done
  observe "after $1-$2"
done
echo "session $session done"
