#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
[[ $# == 2 && "$2" =~ ^task70-[A-Za-z0-9_-]+$ ]] || exit 2
mode="$1"; name="$2"
case "$mode" in build|31|26) ;; *) exit 2 ;; esac
scratch="$HOME/.local/share/llmp/gemma-owner-variable"
previous="$HOME/.local/share/llmp/gemma-packed-attention-c4"
models="$HOME/.local/share/llmp/reference-models"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
[[ ! -e "$scratch/$name" && ! -e "$scratch/$name.cid" && ! -e "$scratch/$name-container-retired.json" ]] || exit 2
printf '%s  %s\n' d93c10c58c792d9cad39efa816a3d94c11ccd24918a0de27c2ddc0c2643bb994 "$scratch/container_retire.py" \
  6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b "$scratch/reference-ids.i32" | sha256sum --check
cleanup() {
  local previous_status=$?
  trap - EXIT INT TERM
  if ! python3 -B "$scratch/container_retire.py" "$scratch" "$name"; then exit 1; fi
  exit "$previous_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)"
  --cidfile "$scratch/$name.cid" --name "llmp-gemma26-late-moe-$name"
  --label llmp.observer=gemma26-late-moe --tmpfs /tmp:rw,size=1g
  --mount "type=bind,src=$scratch,dst=/scratch"
  --mount "type=bind,src=$previous/headers,dst=/headers,readonly")
if [[ "$mode" == build ]]; then
  [[ ! -e "$scratch/llama_joined_8k" ]] || exit 2
  sudo -n docker "${common[@]}" --entrypoint g++ "$image" -std=c++23 -O2 -march=armv8-a \
    -Wall -Wextra -Werror -I/headers/include -I/headers/ggml/include \
    /scratch/llama_joined_8k.cc -L/app -Wl,-rpath,/app \
    -lllama -lggml -lggml-base -o /scratch/llama_joined_8k
  chmod 500 "$scratch/llama_joined_8k"
  sha256sum "$scratch/llama_joined_8k" > "$scratch/reference-binary.sha256"
else
  [[ "${LLMP_GEMMA_REFERENCE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
  printf '%s  %s\n' "$LLMP_GEMMA_REFERENCE_SHA256" "$scratch/llama_joined_8k" | sha256sum --check
  if [[ "$mode" == 31 ]]; then raw=gemma-4-31B-it-UD-Q4_K_XL.gguf; ubatch=256
  else raw=gemma-4-26B-A4B-it-UD-Q4_K_M.gguf; ubatch=1024; fi
  sudo -n docker "${common[@]}" --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 \
    --mount "type=bind,src=$models,dst=/model,readonly" \
    --entrypoint /scratch/llama_joined_8k "$image" \
    "/model/$raw" "/scratch/$name" 4 "$ubatch" /scratch/reference-ids.i32 8k
fi
python3 -B "$scratch/container_retire.py" "$scratch" "$name"
[[ -f "$scratch/$name-container-retired.json" ]]
