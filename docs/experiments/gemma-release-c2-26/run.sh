#!/bin/bash
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
ulimit -c 0
[[ $# == 2 && "$1" =~ ^(native|reference)$ && "$2" =~ ^(first|repeat|timing-first|timing-repeat)$ ]] || exit 2
[[ "${JITLLM_C2_SOURCE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
root="$HOME/.local/share/jitllm/gemma-release-c2-26"
engine="$1" arm="$2"
if [[ "$arm" == timing-* ]]; then
  [[ "${JITLLM_C2_NATIVE_OWN_SHA256:-}" =~ ^[0-9a-f]{64}$ &&
     "${JITLLM_C2_REFERENCE_OWN_SHA256:-}" =~ ^[0-9a-f]{64}$ &&
     "${JITLLM_C2_COMPARISON_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
  python3 -B "$root/analyze.py" timing-ready "$root" "$JITLLM_C2_SOURCE_SHA256" \
    "$JITLLM_C2_NATIVE_OWN_SHA256" "$JITLLM_C2_REFERENCE_OWN_SHA256" "$JITLLM_C2_COMPARISON_SHA256"
fi
python3 -B "$root/analyze.py" guard "$root" "$JITLLM_C2_SOURCE_SHA256"
[[ ! -e "$root/$engine/$arm" && ! -e "$root/$engine/$arm.log" ]] || exit 2
# Exact binary/raw/artifact/input identities are supplied by preregistered source.json.
if [[ "$engine" == native ]]; then
  env LD_LIBRARY_PATH=/home/pmeenan/src/jitLLM-wt/m3gm31/build/spark-native/cublas \
    "$HOME/.local/share/jitllm/ggml-release-refresh/native1" "$HOME/.local/share/jitllm/m3-artifacts/4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3" \
    "$root/native/$arm" 26 2 joined all "$root/inputs.i32" production \
    > "$root/native/$arm.log" 2>&1
  grep -xF 'JOINED_NATIVE_RETIRED production=1' "$root/native/$arm.log"
else
  [[ "${JITLLM_C2_NATIVE_OWN_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
  python3 -B "$root/analyze.py" native-ready "$root" "$JITLLM_C2_SOURCE_SHA256" "$JITLLM_C2_NATIVE_OWN_SHA256"
  name="release-c2-26-$arm"
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
    --cidfile "$root/$name.cid" --name "jitllm-gemma26-late-moe-$name" \
    --label jitllm.observer=gemma26-late-moe --tmpfs /tmp:rw,size=1g \
    --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 --ulimit core=0 \
    --mount "type=bind,src=$root,dst=/scratch" \
    --mount "type=bind,src=$HOME/.local/share/jitllm/reference-models,dst=/model,readonly" \
    --entrypoint /scratch/llama-c2 \
    ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db \
    /model/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf "/scratch/reference/$arm" 2 1024 \
    /scratch/inputs.i32 production 26 joined > "$root/reference/$arm.log" 2>&1
  python3 -B "$root/container_retire.py" "$root" "$name"
  grep -xF 'JOINED_REFERENCE_RETIRED production=1' "$root/reference/$arm.log"
  cp --no-clobber "$root/$name-container-retired.json" "$root/reference/$arm-container-retired.json"
fi
retirement_sha=none
if [[ "$engine" == reference ]]; then
  read -r retirement_sha _ < <(sha256sum "$root/reference/$arm-container-retired.json")
fi
python3 -B "$root/analyze.py" admit "$root" "$engine" "$arm" "$JITLLM_C2_SOURCE_SHA256" "$retirement_sha"
