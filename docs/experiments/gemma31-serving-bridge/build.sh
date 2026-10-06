#!/bin/bash
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
ulimit -c 0
[[ $# == 1 && "$1" =~ ^(compile|usage)$ && "${JITLLM_SERVING_STAGE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || exit 2
root="$HOME/.local/share/jitllm/gemma31-serving-bridge"
reference="$HOME/.local/share/jitllm/llama-reference-v060"
sdk="$HOME/.local/share/jitllm/sdk/aarch64-c09daba6ac31edee"
printf '%s  %s\n' "$JITLLM_SERVING_STAGE_SHA256" "$root/stage.sha256" \
  63ccae40a696fb574bedf1145a75f9576a0f7914efbd8081fc6b085babe3cf64 "$reference/image-closure1/receipt.json" \
  7b8167cdbe7455b25cb41c681f321562f34683bed26dac2b64e47650b987e02d "$sdk/sdk.json" | sha256sum --check
sha256sum --check "$root/stage.sha256"
if [[ "$1" == compile ]]; then
  [[ ! -e "$root/compiled1" && ! -e "$root/llama-serving" ]] || exit 2
  mkdir -m 700 "$root/compiled1"
  "$sdk/bin/clang++" --version > "$root/compiled1/compiler.txt"
  "$sdk/bin/clang++" --target=aarch64-linux-gnu -march=armv8-a \
    --gcc-install-dir="$sdk/gcc/aarch64-linux-gnu/lib/gcc/aarch64-linux-gnu/16" \
    -std=c++23 -O2 -Wall -Wextra -Werror -static-libstdc++ -static-libgcc \
    -I"$root/headers" -I"$reference/headers/include" -I"$reference/headers/ggml/include" \
    "$root/source/docs/experiments/gemma31-serving-bridge/llama_serving.cc" -L"$reference/image-closure1/lib" \
    -Wl,-rpath,/app -Wl,--allow-shlib-undefined -lllama -lggml -lggml-base -ldl \
    -o "$root/compiled1/llama-serving"
  chmod 500 "$root/compiled1/llama-serving"
  sha256sum "$root/compiled1/llama-serving" > "$root/compiled1/binary.sha256"
else
  [[ ! -e "$root/serving-usage1.cid" && ! -e "$root/serving-usage1-container-retired.json" ]] || exit 2
  name=serving-usage1
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
    --label jitllm.observer=gemma26-late-moe --tmpfs /tmp:rw,size=16m --ulimit core=0 \
    --mount "type=bind,src=$root,dst=/scratch" --entrypoint python3 \
    ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db \
    -B -c 'import subprocess; p=subprocess.run(["/scratch/compiled1/llama-serving"],capture_output=True); assert p.returncode==1 and b"MODEL OUT" in p.stderr; print("SERVING_USAGE_REFUSAL_PASS")'
  python3 -B "$root/container_retire.py" "$root" "$name"
fi
sha256sum --check "$root/stage.sha256"
