#!/bin/bash
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
ulimit -c 0
[[ $# == 4 && "$1" =~ ^(native|reference)$ && "$2" =~ ^(1|4)$ && "$3" =~ ^(quality|cycle|corpus)$ && "$4" =~ ^(once|first|repeat)$ ]] || exit 2
[[ "${JITLLM_SERVING_SOURCE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
engine="$1" count="$2" mode="$3" arm="$4"
[[ ( "$mode" == cycle && "$arm" != once ) || ( "$mode" != cycle && "$arm" == once ) ]] || exit 2
[[ "$mode" != corpus || "$count" == 1 ]] || exit 2
root="$HOME/.local/share/jitllm/gemma31-serving-bridge"
analysis="$root/source/docs/experiments/gemma31-serving-bridge/analyze.py"
key="$engine-c$count-$mode"
[[ "$mode" != cycle ]] || key="$key-$arm"
python3 -B "$analysis" guard "$root" "$JITLLM_SERVING_SOURCE_SHA256"
[[ "$(df -B1 --output=avail "$root" | tail -1)" -ge 107374182400 ]] || exit 2
[[ ! -e "$root/$key" && ! -e "$root/$key.log" ]] || exit 2
if [[ "$mode" == cycle ]]; then
  [[ "${JITLLM_SERVING_NATIVE_OWN_SHA256:-}" =~ ^[0-9a-f]{64}$ && "${JITLLM_SERVING_REFERENCE_OWN_SHA256:-}" =~ ^[0-9a-f]{64}$ && "${JITLLM_SERVING_COMPARISON_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
  python3 -B "$analysis" timing-ready "$root" "$JITLLM_SERVING_SOURCE_SHA256" "$count" \
    "$JITLLM_SERVING_NATIVE_OWN_SHA256" "$JITLLM_SERVING_REFERENCE_OWN_SHA256" "$JITLLM_SERVING_COMPARISON_SHA256"
fi
if [[ "$engine" == native ]]; then
  [[ ! -e "$root/enrollment" && ! -L "$root/enrollment" ]] || exit 2
  env LD_LIBRARY_PATH=/home/pmeenan/src/jitLLM-wt/m3gm31/build/spark-native/cublas \
    "$root/native-serving1" proof "$mode" "$count" "$root/inputs.i32" "$root/$key" \
    --config "$root/config-c$count.toml" --anchor "$root/enrollment" > "$root/$key.log" 2>&1
  grep -xF 'SERVING_PROOF_RETIRED' "$root/$key.log"
else
  [[ "${JITLLM_SERVING_NATIVE_OWN_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
  python3 -B "$analysis" native-ready "$root" "$JITLLM_SERVING_SOURCE_SHA256" "$count" \
    "${mode/cycle/quality}" "$JITLLM_SERVING_NATIVE_OWN_SHA256"
  input=/scratch/inputs.i32
  if [[ "$mode" == quality ]]; then
    [[ "${JITLLM_SERVING_ANCHOR_PROOF_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
    python3 -B "$analysis" carrier-ready "$root" "$JITLLM_SERVING_SOURCE_SHA256" "$count" \
      "$JITLLM_SERVING_NATIVE_OWN_SHA256" "$JITLLM_SERVING_ANCHOR_PROOF_SHA256"
    input="/scratch/anchors-c$count.i32"
  fi
  name="serving-$key"
  [[ ! -e "$root/$name.cid" && ! -e "$root/$name-container-retired.json" && ! -e "$root/$key-container-retired.json" ]] || exit 2
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
    --entrypoint /scratch/llama-serving2 \
    ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db \
    /model/gemma-4-31B-it-UD-Q4_K_XL.gguf "/scratch/$key" "$count" "$input" "$mode" > "$root/$key.log" 2>&1
  python3 -B "$root/container_retire.py" "$root" "$name"
  grep -xF 'SERVING_REFERENCE_RETIRED' "$root/$key.log"
  cp --no-clobber "$root/$name-container-retired.json" "$root/$key-container-retired.json"
  if [[ "$mode" == quality ]]; then
    python3 -B "$analysis" carrier-ready "$root" "$JITLLM_SERVING_SOURCE_SHA256" "$count" \
      "$JITLLM_SERVING_NATIVE_OWN_SHA256" "$JITLLM_SERVING_ANCHOR_PROOF_SHA256"
  fi
fi
python3 -B "$analysis" guard "$root" "$JITLLM_SERVING_SOURCE_SHA256"
