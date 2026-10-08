#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
scratch="$HOME/.local/share/llmp/gemma-assistant-reference"
source_root="$HOME/src/llmp-wt/m3gm31"
models="$HOME/.local/share/llmp/reference-models"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)"
        --tmpfs /tmp:rw,size=1g --mount "type=bind,src=$scratch,dst=/scratch"
        --mount "type=bind,src=$source_root/docs/experiments/gemma-assistant-reference,dst=/tool,readonly"
        --mount "type=bind,src=$models,dst=/model,readonly")
# Closed 31B C1 extension; historical 26B modes retain their original paths.
if [[ "${1:-}" == build31 || "${1:-}" == acquire31 ]]; then
  [[ $# == 2 && "$2" =~ ^assistant31-[A-Za-z0-9_-]+$ ]] || exit 2
  name="$2"
  scratch="$HOME/.local/share/llmp/gemma31-assistant-reference"
  [[ ! -e "$scratch/$name" && ! -e "$scratch/$name.cid" &&
     ! -e "$scratch/$name-container-retired.json" ]] || exit 2
  printf '%s  %s\n' d93c10c58c792d9cad39efa816a3d94c11ccd24918a0de27c2ddc0c2643bb994 \
    "$scratch/container_retire.py" \
    b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610 \
    "$scratch/inputs.i32" | sha256sum --check
  cleanup31() {
    local previous_status=$?
    trap - EXIT INT TERM
    if ! python3 -B "$scratch/container_retire.py" "$scratch" "$name"; then exit 1; fi
    exit "$previous_status"
  }
  trap cleanup31 EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  common31=(run --rm --network none --read-only --user "$(id -u):$(id -g)"
    --cidfile "$scratch/$name.cid" --name "llmp-gemma26-late-moe-$name"
    --label llmp.observer=gemma26-late-moe --tmpfs /tmp:rw,size=1g
    --mount "type=bind,src=$scratch,dst=/scratch")
  if [[ "$1" == build31 ]]; then
    [[ ! -e "$scratch/llama_assistant" ]] || exit 2
    sudo -n docker "${common31[@]}" --entrypoint g++ "$image" -std=c++23 -O2 -march=armv8-a \
      -Wall -Wextra -Werror -I/scratch/headers/include -I/scratch/headers/src \
      -I/scratch/headers/ggml/include /scratch/llama_assistant.cc -L/app -Wl,-rpath,/app \
      -lllama -lggml -lggml-base -o /scratch/llama_assistant
    chmod 500 "$scratch/llama_assistant"
    sha256sum "$scratch/llama_assistant" > "$scratch/reference-binary.sha256"
  else
    [[ "${LLMP_GEMMA_ASSISTANT_REFERENCE_SHA256:-}" =~ ^[0-9a-f]{64}$ &&
       "${LLMP_GEMMA_ASSISTANT_PRE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
    printf '%s  %s\n' "$LLMP_GEMMA_ASSISTANT_REFERENCE_SHA256" "$scratch/llama_assistant" \
      "$LLMP_GEMMA_ASSISTANT_PRE_SHA256" "$scratch/reference-pre1.json" | sha256sum --check
    [[ $(stat -c %s "$models/gemma-4-31B-it-UD-Q4_K_XL.gguf") == 18822970304 &&
       $(stat -c %s "$models/mtp-gemma-4-31B-it.gguf") == 514687104 ]] || exit 2
    sudo -n docker "${common31[@]}" --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 \
      --mount "type=bind,src=$models,dst=/model,readonly" \
      --entrypoint /scratch/llama_assistant "$image" \
      /model/gemma-4-31B-it-UD-Q4_K_XL.gguf /model/mtp-gemma-4-31B-it.gguf \
      /scratch/inputs.i32 "/scratch/$name" 1 serial 31
  fi
  python3 -B "$scratch/container_retire.py" "$scratch" "$name"
  [[ -f "$scratch/$name-container-retired.json" ]] || exit 1
  exit 0
fi
case "${1:-}" in
  build)
    [[ $# == 1 ]]
    sudo -n docker "${common[@]}" --entrypoint g++ "$image" -std=c++23 -O2 -march=armv8-a \
      -Wall -Wextra -Werror -I/scratch/headers/include -I/scratch/headers/src \
      -I/scratch/headers/ggml/include /tool/llama_assistant.cc -L/app -Wl,-rpath,/app \
      -lllama -lggml -lggml-base -o /scratch/llama_assistant
    sha256sum "$scratch/llama_assistant"
    ;;
  acquire1|acquire-serial2|acquire-batch2)
    [[ $# == 2 && "$2" =~ ^[a-zA-Z0-9_-]+$ ]]
    owners=1 mode=serial
    if [[ "$1" != acquire1 ]]; then owners=2; fi
    if [[ "$1" == acquire-batch2 ]]; then mode=batch; fi
    sudo -n docker "${common[@]}" --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 \
      --entrypoint /scratch/llama_assistant "$image" \
      /model/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf /model/mtp-gemma-4-26B-A4B-it.gguf \
      /scratch/inputs.i32 "/scratch/$2" "$owners" "$mode"
    ;;
  *) exit 2 ;;
esac
