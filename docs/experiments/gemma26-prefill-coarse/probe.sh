#!/bin/bash
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
[[ $# == 1 && "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]
scratch="$HOME/.local/share/jitllm/gemma26-prefill-coarse"
source_root="$HOME/src/jitLLM-wt/m3gm31"
nsight=/opt/nvidia/nsight-systems/2025.3.2
[[ ! -e "$scratch/$1-completion.json" && ! -e "$scratch/$1.sqlite" && ! -e "$scratch/$1.nsys-rep" ]]
"$nsight/bin/nsys" profile --trace=nvtx,osrt --sample=none --cpuctxsw=none \
  --wait=all --stop-on-exit=true --kill=none --force-overwrite=false --export=sqlite \
  "--output=$scratch/$1" "$scratch/profile_child" "$scratch/$1-completion.json" \
  "$source_root/build/spark-native/benchmarks/jitllm_gemma26_clock_probe"
[[ -s "$scratch/$1-completion.json" ]]
