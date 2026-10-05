#!/bin/bash
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
# Installed GPU supervisor required. quality|ordinary|paid|paid-ordinary NAME.
set -euo pipefail
[[ "$#" == 2 ]] || exit 2
mode="$1"
name="$2"
[[ "$name" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
source_root="${JITLLM_GEMMA_NORM_SOURCE_ROOT:-$HOME/src/jitLLM-wt/m3gm31}"
root="${JITLLM_GEMMA_NORM_ROOT:-$HOME/.local/share/jitllm/gemma-native-norm}"
artifact="$HOME/.local/share/jitllm/m3-artifacts/32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08"
mkdir -p "$root"
case "$mode" in
  quality|ordinary)
    ids="$HOME/.local/share/jitllm/gemma31-quality/input1/ids.i32"
    expected=b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610
    policy=both
    [[ "$mode" == quality ]] || policy=ordinary
    actual="$(sha256sum "$ids")"
    [[ "${actual%% *}" == "$expected" ]] || exit 2
    exec "$source_root/build/spark-native/benchmarks/jitllm_gemma_quality" "$artifact" "$ids" "$root/$name" 128 "$policy" 31
    ;;
  paid|paid-ordinary)
    ids="$HOME/.local/share/jitllm/gemma31-prefill/input1/ids.i32"
    expected=6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b
    policy=both
    [[ "$mode" == paid ]] || policy=ordinary
    actual="$(sha256sum "$ids")"
    [[ "${actual%% *}" == "$expected" ]] || exit 2
    exec "$source_root/build/spark-native/benchmarks/jitllm_gemma_prefill" "$artifact" "$ids" "$root/$name" 31 "$policy"
    ;;
  *) exit 2 ;;
esac
