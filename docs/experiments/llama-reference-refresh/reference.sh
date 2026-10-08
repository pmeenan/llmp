#!/bin/bash
# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
ulimit -c 0
[[ $# == 2 && "$2" =~ ^llama-refresh-[A-Za-z0-9_-]+$ ]] || exit 2
scratch="$HOME/.local/share/llmp/llama-reference-v060"
image=ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db
sdk="$HOME/.local/share/llmp/sdk/aarch64-c09daba6ac31edee"
mode="$1" name="$2"
sha256sum --check "$scratch/stage1.sha256"
if [[ "$mode" == pull ]]; then
  [[ ! -e "$scratch/image-inspect2.json" ]] || exit 2
  sudo -n docker pull --platform linux/arm64 "$image"
  sudo -n docker image inspect "$image" > "$scratch/image-inspect2.json"
  python3 -B - "$scratch/image-inspect2.json" <<'PYCHECK'
import json,pathlib,sys
items=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert len(items)==1
j=items[0]; assert j['Architecture']=='arm64' and j['Os']=='linux'
assert j['Id'] in {'sha256:c45165320931c3c957baef9edf518d731870f9c4f9eb98f749170d5009afe2bd',
                   'sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db'}
assert 'ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db' in j['RepoDigests']
expected=json.loads((pathlib.Path(sys.argv[1]).parent/'full-cuda13-b11429-arm64-config.json').read_text())
assert j['Config']==expected['config'] and j['RootFS']['Layers']==expected['rootfs']['diff_ids']
l=j['Config']['Labels']; assert l['org.opencontainers.image.revision']=='d81235049384534c167caea52b85a694f6103d14'
assert l['org.opencontainers.image.version']=='b11429'
assert l['org.opencontainers.image.source']=='https://github.com/ggml-org/llama.cpp'
PYCHECK
  exit 0
fi
if [[ "$mode" == compile ]]; then
  printf '%s  %s\n' 7b8167cdbe7455b25cb41c681f321562f34683bed26dac2b64e47650b987e02d "$sdk/sdk.json" | sha256sum --check
  [[ -f "$scratch/image-closure1/receipt.json" && ! -e "$scratch/compiled1" ]] || exit 2
  python3 -B - "$scratch" <<'PYCHECK'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]);j=json.loads((r/'image-closure1/receipt.json').read_text())
for name,item in j['app_files'].items():
 if '.so' in name:
  p=r/'image-closure1/lib'/name;assert p.stat().st_size==item['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
PYCHECK
  mkdir -m 700 "$scratch/compiled1"
  "$sdk/bin/clang++" --version > "$scratch/compiled1/compiler.txt"
  for helper in joined quality assistant; do
    "$sdk/bin/clang++" --target=aarch64-linux-gnu -march=armv8-a \
      --gcc-install-dir="$sdk/gcc/aarch64-linux-gnu/lib/gcc/aarch64-linux-gnu/16" \
      -std=c++23 -O2 -Wall -Wextra -Werror -static-libstdc++ -static-libgcc \
      -I"$scratch/headers/include" -I"$scratch/headers/src" -I"$scratch/headers/ggml/include" \
      -I"$scratch/headers/ggml/src" "$scratch/helpers/$helper.cc" \
      -L"$scratch/image-closure1/lib" -Wl,-rpath,/app -Wl,--allow-shlib-undefined \
      -lllama -lggml -lggml-base -o "$scratch/compiled1/$helper"
    chmod 500 "$scratch/compiled1/$helper"
  done
  sha256sum "$scratch/compiled1/joined" "$scratch/compiled1/quality" "$scratch/compiled1/assistant" > "$scratch/compiled1/binaries.sha256"
  exit 0
fi
[[ "$mode" == inspect || "$mode" == usage ]] || exit 2
[[ ! -e "$scratch/$name.cid" && ! -e "$scratch/$name-container-retired.json" ]] || exit 2
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if ! python3 -B "$scratch/container_retire.py" "$scratch" "$name"; then exit 1; fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
common=(run --rm --network none --read-only --user "$(id -u):$(id -g)"
  --cidfile "$scratch/$name.cid" --name "llmp-gemma26-late-moe-$name"
  --label llmp.observer=gemma26-late-moe --tmpfs /tmp:rw,size=64m
  --ulimit core=0 --mount "type=bind,src=$scratch,dst=/scratch"
  --device nvidia.com/gpu=all --env CUDA_DISABLE_PTX_JIT=1)
# CUDA device/runtime capability is checked; no model/checkpoint directories are mounted.
sudo -n docker "${common[@]}" --entrypoint python3 "$image" -B /scratch/probe.py "$mode"
python3 -B "$scratch/container_retire.py" "$scratch" "$name"
[[ -f "$scratch/$name-container-retired.json" ]] || exit 1
