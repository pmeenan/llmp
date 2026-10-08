#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
#
# L2 counters behind the host-VMM diagnosis (README.md in this directory):
# Nsight Compute's L2 lookup hits and misses for the GPU's reads, per memory
# arm, for the microbenchmark's L2-resident reread and streaming scan, and
# for two GGML products under three placements. Counts, not timings (ncu
# serializes and replays kernels). Needs ncu and, with the driver's
# RmProfilingAdminOnly=1, root (sudo -n).
#
#   ncu_counters.sh BENCH OUT_DIR
set -euo pipefail
bench=$1
out=$2
mkdir -p "$out"
ncu=${NCU:-/usr/local/cuda/bin/ncu}
metrics=gpu__time_duration.sum,lts__t_sectors_srcunit_tex_op_read.sum
for aperture in device sysmem; do
  for part in "" _lookup_hit _lookup_miss; do
    metrics+=",lts__t_sectors_srcunit_tex_aperture_${aperture}_op_read${part}.sum"
  done
done
"$ncu" --version > "$out/ncu-version.txt"
for test in reread-4m scan16; do
  for arm in malloc dvmm hvmm hvmm-gpu hvmm-2m pinned pageable registered-thp managed; do
    sudo -n "$ncu" --csv --metrics "$metrics" -k 'regex:Reread|ScanVector' --launch-skip 2 \
      --launch-count 1 "$bench" micro --arm "$arm" --rounds 1 --test "$test" \
      > "$out/micro-$test-$arm.csv" 2> "$out/micro-$test-$arm.err"
  done
done
for case in linear.lm_head@1 linear.gate_up@512; do
  for placement in "malloc malloc malloc malloc" "hvmm hvmm hvmm hvmm" "hvmm dvmm dvmm dvmm"; do
    set -- $placement
    # GGML's and cuBLAS's kernels only (not the benchmark's fills): skip
    # the eager invocation's, profile six from the graph replays.
    sudo -n "$ncu" --csv --metrics "$metrics" --kernel-name-base mangled \
      -k 'regex:mul_mat|gemm|convert_unary' \
      --launch-skip 4 --launch-count 6 \
      "$bench" ggml --weights "$1" --acts "$2" --scratch "$3" --workspace "$4" --case "$case" \
      > "$out/ggml-$case-$1-$2.csv" 2> "$out/ggml-$case-$1-$2.err"
  done
done
