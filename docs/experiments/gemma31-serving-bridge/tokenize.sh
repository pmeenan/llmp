#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
ulimit -c 0
root="$HOME/.local/share/llmp/gemma31-serving-bridge"
[[ "${LLMP_SERVING_INPUT_FRAME_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
printf '%s  %s\n' "$LLMP_SERVING_INPUT_FRAME_SHA256" "$root/input.sha256" | sha256sum --check
sha256sum --check "$root/input.sha256"
[[ ! -e "$root/native-input" && ! -e "$root/public-input" && ! -e "$root/input-pack" ]] || exit 2
mkdir -m 700 "$root/native-input" "$root/public-input"
metadata="$HOME/.local/share/llmp/m3-artifacts/32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08/meta/gemma-4-31B-it-UD-Q4_K_XL.kv.gguf"
for index in 1 2 3 4; do
  "$root/native-tokenizer1" "$metadata" "$root/texts/paragraph-$index.txt" "$root/native-input/paragraph-$index" literal
  name="serving-input-$index"
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
    --label llmp.observer=gemma26-late-moe --tmpfs /tmp:rw,size=1g --ulimit core=0 \
    --mount "type=bind,src=$root,dst=/scratch" \
    --mount "type=bind,src=$HOME/.local/share/llmp/reference-models,dst=/model,readonly" \
    --entrypoint /scratch/llama-serving \
    ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db \
    prepare /model/gemma-4-31B-it-UD-Q4_K_XL.gguf "/scratch/texts/paragraph-$index.txt" "/scratch/public-input/paragraph-$index" \
    > "$root/$name.log" 2>&1
  python3 -B "$root/container_retire.py" "$root" "$name"
  grep -xF 'SERVING_INPUT_RETIRED vocab_only=1 model_contexts=0 supplied_rows=8192' "$root/$name.log"
  trap - EXIT INT TERM
done
python3 -B "$root/source/docs/experiments/gemma31-serving-bridge/input.py" pack "$root/texts" \
  6f4c968af84d0bcc68cc85697a305e8bdf1c4ed7c0dc7f7f967bdfd842d71abb \
  "$root/native-input" "$root/public-input" "$root/input-pack"
printf '%s  %s\n' "$LLMP_SERVING_INPUT_FRAME_SHA256" "$root/input.sha256" | sha256sum --check
sha256sum --check "$root/input.sha256"
