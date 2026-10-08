#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
[[ $# == 2 && "$1" =~ ^(control|trace)$ && "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]
scratch="$HOME/.local/share/llmp/gemma26-prefill-profile"
source_root="$HOME/src/llmp-wt/m3gm31"
nsight=/opt/nvidia/nsight-systems/2025.3.2
artifact="$HOME/.local/share/llmp/m3-artifacts/4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3"
[[ ! -e "$scratch/$2" && ! -e "$scratch/$2-completion.json" && ! -e "$scratch/$2.sqlite" && ! -e "$scratch/$2.nsys-rep" ]]
launch=("$scratch/profile_child" "$scratch/$2-completion.json"
        "$source_root/build/spark-native/benchmarks/llmp_gemma26_prefill_profile"
        "$artifact" "$scratch/ids.i32" "$scratch/$2" 26 all 1024)
if [[ "$1" == trace ]]; then
  launch=("$nsight/bin/nsys" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none
          --cuda-graph-trace=node --wait=all --stop-on-exit=true --kill=none
          --force-overwrite=false --export=sqlite "--output=$scratch/$2" "${launch[@]}")
fi
"${launch[@]}"
[[ -s "$scratch/$2-completion.json" ]]
