#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
ulimit -c 0
[[ $# == 2 && "$1" =~ ^(native|reference)$ && "$2" =~ ^(first|repeat|timing-first|timing-repeat)$ ]] || exit 2
[[ "${LLMP_C2_SOURCE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
root="$HOME/.local/share/llmp/gemma-release-c2"
engine="$1" arm="$2"
if [[ "$arm" == timing-* ]]; then
  [[ "${LLMP_C2_NATIVE_OWN_SHA256:-}" =~ ^[0-9a-f]{64}$ &&
     "${LLMP_C2_REFERENCE_OWN_SHA256:-}" =~ ^[0-9a-f]{64}$ &&
     "${LLMP_C2_COMPARISON_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
  python3 -B "$root/analyze.py" timing-ready "$root" "$LLMP_C2_SOURCE_SHA256" \
    "$LLMP_C2_NATIVE_OWN_SHA256" "$LLMP_C2_REFERENCE_OWN_SHA256" "$LLMP_C2_COMPARISON_SHA256"
fi
python3 -B "$root/analyze.py" guard "$root" "$LLMP_C2_SOURCE_SHA256"
[[ ! -e "$root/$engine/$arm" && ! -e "$root/$engine/$arm.log" ]] || exit 2
# Exact binary/raw/artifact/input identities are supplied by preregistered source.json.
if [[ "$engine" == native ]]; then
  env LD_LIBRARY_PATH=/home/pmeenan/src/llmp-wt/m3gm31/build/spark-native/cublas \
    "$HOME/.local/share/llmp/ggml-release-refresh/native1" "$HOME/.local/share/llmp/m3-artifacts/32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08" \
    "$root/native/$arm" 31 2 joined norm "$root/inputs.i32" production \
    > "$root/native/$arm.log" 2>&1
  grep -xF 'JOINED_NATIVE_RETIRED production=1' "$root/native/$arm.log"
else
  [[ "${LLMP_C2_NATIVE_OWN_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
  python3 -B "$root/analyze.py" native-ready "$root" "$LLMP_C2_SOURCE_SHA256" "$LLMP_C2_NATIVE_OWN_SHA256"
  name="release-c2-$arm"
  [[ ! -e "$root/$name.cid" && ! -e "$root/$name-container-retired.json" ]] || exit 2
  cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if ! python3 -B "$root/container_retire.py" "$root" "$name"; then exit 1; fi
    exit "$status"
  }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  sudo -n docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
    --cidfile "$root/$name.cid" --name "llmp-gemma26-late-moe-$name" \
    --label llmp.observer=gemma26-late-moe --tmpfs /tmp:rw,size=1g \
    --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 --ulimit core=0 \
    --mount "type=bind,src=$root,dst=/scratch" \
    --mount "type=bind,src=$HOME/.local/share/llmp/reference-models,dst=/model,readonly" \
    --entrypoint /scratch/llama-c2 \
    ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db \
    /model/gemma-4-31B-it-UD-Q4_K_XL.gguf "/scratch/reference/$arm" 2 256 \
    /scratch/inputs.i32 production 31 joined > "$root/reference/$arm.log" 2>&1
  python3 -B "$root/container_retire.py" "$root" "$name"
  grep -xF 'JOINED_REFERENCE_RETIRED production=1' "$root/reference/$arm.log"
  cp --no-clobber "$root/$name-container-retired.json" "$root/reference/$arm-container-retired.json"
fi
python3 -B "$root/analyze.py" guard "$root" "$LLMP_C2_SOURCE_SHA256"
