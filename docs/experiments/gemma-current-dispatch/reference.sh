#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
# One existing-controller observation of the unchanged Task57 reference recipe.
set -euo pipefail
umask 077
[[ $# == 0 ]] || exit 2
name=task-dispatch26-observed
scratch="$HOME/.local/share/llmp/gemma-current-quality26"
retire="$scratch/tools/container_retire.py"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
[[ ! -e "$scratch/$name.cid" && ! -e "$scratch/$name-container-retired.json" && ! -e "$scratch/observed-dispatch" ]] || exit 2
printf '%s  %s\n' \
  d93c10c58c792d9cad39efa816a3d94c11ccd24918a0de27c2ddc0c2643bb994 "$retire" \
  225c40affb625e02e7e381d75c306180b444a63b520a7ca84bfa8e3dd023973b "$scratch/llama_quality" \
  b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610 "$scratch/input1/ids.i32" \
  aef05aaf605e2cfdcab7227b05903d62a22c505a1767c75f785a4c594d1d5d29 "$scratch/controller-all.so" | sha256sum --check
cleanup() {
  local previous_status=$?
  trap - EXIT INT TERM
  if ! python3 -B "$retire" "$scratch" "$name"; then exit 1; fi
  exit "$previous_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
sudo -n docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,size=1g --cidfile "$scratch/$name.cid" \
  --name "llmp-gemma26-late-moe-$name" --label llmp.observer=gemma26-late-moe \
  --mount "type=bind,src=$scratch,dst=/scratch" \
  --mount "type=bind,src=$HOME/.local/share/llmp/reference-models,dst=/model,readonly" \
  --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 \
  --env LD_PRELOAD=/scratch/controller-all.so --env LLMP_REFERENCE_FUSION_POLICY=all \
  --env LLMP_REFERENCE_TRACE=1 --entrypoint /scratch/llama_quality "$image" \
  /model/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf /scratch/input1/ids.i32 \
  /scratch/observed-dispatch score-ring 1024
python3 -B "$retire" "$scratch" "$name"
[[ -s "$scratch/$name-container-retired.json" ]]
# The unchanged scorer returns after explicit batch/context/model/backend release.
printf '%s\n' 'OBSERVED_SCORER_EXIT0 explicit_public_backend_teardown=source_verified'
