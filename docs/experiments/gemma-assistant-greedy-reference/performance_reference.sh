#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
[[ $# == 2 && "$2" =~ ^throughput31-[A-Za-z0-9_-]+$ ]] || exit 2
scratch="$HOME/.local/share/llmp/gemma-assistant-throughput"
headers="$HOME/.local/share/llmp/gemma31-assistant-reference/headers"
models="$HOME/.local/share/llmp/reference-models"
image=ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7
name="$2"
[[ ! -e "$scratch/$name" && ! -e "$scratch/$name.cid" &&
   ! -e "$scratch/$name-container-retired.json" ]] || exit 2
printf '%s  %s\n' d93c10c58c792d9cad39efa816a3d94c11ccd24918a0de27c2ddc0c2643bb994 \
  "$scratch/container_retire.py" | sha256sum --check
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
  --mount "type=bind,src=$scratch,dst=/scratch")
case "$1" in
  build)
    [[ ! -e "$scratch/llama_throughput1" ]] || exit 2
    sudo -n docker "${common[@]}" --mount "type=bind,src=$headers,dst=/headers,readonly" \
      --entrypoint /bin/sh "$image" -c 'sha256sum --check /scratch/libraries.sha256 && \
      exec g++ -std=c++23 -O2 -march=armv8-a -Wall -Wextra -Werror \
      -I/headers/include -I/headers/src -I/headers/ggml/include \
      /scratch/source/docs/experiments/gemma-assistant-greedy-reference/llama_greedy.cc \
      -L/app -Wl,-rpath,/app -lllama -lggml -lggml-base -o /scratch/llama_throughput1'
    chmod 500 "$scratch/llama_throughput1"
    ;;
  plain31|unit31)
    [[ "${LLMP_GREEDY_REFERENCE_SHA256:-}" =~ ^[0-9a-f]{64}$ &&
       "${LLMP_GREEDY_REFERENCE_PRE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
    printf '%s  %s\n' "$LLMP_GREEDY_REFERENCE_SHA256" "$scratch/llama_throughput1" \
      "$LLMP_GREEDY_REFERENCE_PRE_SHA256" "$scratch/reference-pre32.json" \
      b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610 \
      "$scratch/inputs.i32" | sha256sum --check
    python3 -B - "$scratch/reference-pre32.json" <<'PYGUARD'
import hashlib,json,pathlib,sys
receipt=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert set(receipt['native_owns'])=={'plain32','unit32'}
for mode,item in receipt['native_owns'].items():
    path=pathlib.Path(item['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    own=json.loads(path.read_text())
    assert own['engine']=='native' and own['mode']==mode and own['profile']=='31'
    assert own['full_quality_repeat_exact'] and own['timed_matches_quality'] and own['all_retained_outputs_finite']
PYGUARD
    mode=unit32
    [[ "$1" != plain* ]] || mode=plain32
    profile=31 target=gemma-4-31B-it-UD-Q4_K_XL.gguf assistant=mtp-gemma-4-31B-it.gguf
    [[ $(stat -c %s "$models/$target") == 18822970304 &&
       $(stat -c %s "$models/$assistant") == 514687104 ]] || exit 2
    sudo -n docker "${common[@]}" --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1 \
      --mount "type=bind,src=$models,dst=/model,readonly" \
      --entrypoint /scratch/llama_throughput1 "$image" \
      "/model/$target" "/model/$assistant" /scratch/inputs.i32 "/scratch/$name" "$profile" "$mode"
    ;;
  *) exit 2 ;;
esac
python3 -B "$scratch/container_retire.py" "$scratch" "$name"
[[ -f "$scratch/$name-container-retired.json" ]] || exit 1
sha256sum "$scratch/llama_throughput1"
