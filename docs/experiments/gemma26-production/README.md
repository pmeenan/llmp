<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 bounded production recipe

The approved Gemma 4 26B-A4B artifact (`4ddb360c…`) now has a bounded
production recipe, adopted on the evidence below. It applies at resolved
context ≤8,192 and ≤4 slots:

- an uncalibrated 1024-row prefill fallback; smaller explicit overrides stay
  intact;
- both norm chains;
- MoE route/reduce fusion;
- joined waves with owner attention.

Through that path, native reproduces official llama.cpp v0.6.0's greedy
tokens for every owner and runs within 0.6% of it. Measured on `spark-b`
(GB10), 2026-10-06, with the same 4×8,192-ID inputs as the
[dense31 bridge](../gemma31-serving-bridge/README.md), against the same
image. The reference is the bridge helper's diagnostic copy for 26B (layer
and width check, `n_batch`/`n_ubatch` 1024, F16 K/V, independent sequences,
normal ring), kept external.

## Before

Before adoption, Gemma26 kept the scalar route with a 128-row prefill cap:

- **C1:** paid cycles 7.91/7.94 s against stock's 4.97/5.04 s (+57%), with
  prefill 5.22 s against 2.33 s.
- **C4:** the scalar cohort decoded each owner separately, 20.24 s against
  14.29/14.46 s (+41%).
- **Tokens:** both diverged from stock's from the second token.

The cap alone doubled prefill time: 64 chunks against stock's 8.

## Each piece

These C1 screens were run in fresh processes with temporary toggles. The
toggles are not committed.

| Recipe | C1 second cycle | Tokens equal to stock's |
| --- | --- | --- |
| Default recipe, 128-row chunks | 7.91 s | no |
| Default recipe, 1024-row chunks | 5.71 s | no |
| Norm chains and MoE route/reduce, 1024-row chunks | 5.02 s | all 129 |
| Same with the fused quantized FFN | 5.01 s | all 129 |

For C4 at 1024-row chunks with the norm/MoE recipe:

- Joined waves without owner attention took 14.54 s, and every owner's
  tokens differed from stock's.
- With owner attention they took 14.41/14.42 s, and every owner's tokens
  matched.

The fused quantized FFN is neutral for Gemma26 (its shared-expert FFN is
small) and stays off. Exact agreement depends on the matched geometry, the
routing arithmetic and owner attention together.

## Production path

These bookends ran reference/native/native/reference in fresh processes,
with the new defaults and no toggles. Native ran warm, second and third
cycles; the reference ran warm and paid cycles.

| Workload | Reference paid | Native second | Native third | Native − reference |
| --- | --- | --- | --- | --- |
| C1 | 5.013 / 5.023 s | 5.051 / 5.047 s | 5.002 / 4.993 s | +0.6% (third −0.3%) |
| C4 | 14.374 / 14.374 s | 14.462 / 14.451 s | 14.351 / 14.313 s | +0.6% (third −0.3%) |

All 129 emitted tokens match stock's in every owner and every cycle of both
native processes, for C1 and for C4.

**Corpus control.** Native `ScorePrompt` and the reference's scalar one-row
queries were compared over the first owner's first 1,024 authentic IDs:

- all 1,024 complete heads are byte-identical;
- zero strict choice differences;
- identical mean NLL (10.60303447) over 1,023 targets.

Each engine's first and repeat run is byte-exact, as is native's initialized
state. The high absolute perplexity on this corpus text is the same in both
engines. It shows parity, not semantic quality.

## Scope

This is bounded: approved 26B-A4B, context ≤8,192, ≤4 slots, ordinary
arithmetic. Larger contexts or cohorts and explicit diagnostics keep their
prior recipe and calibration identity. Default context 262,144 is not
reduced, so the recipe applies only where configured within bounds.
Thinking/tools, assistants, sustained performance and HTTP lifecycle
controls on this recipe remain owed. A natural two-turn `jitllm-runtime chat`
through the default selection used 1024-row chunks and stopped naturally ("The
capital of France is Paris." / "The capital of Italy is Rome."). Among the GPU
suites, only the opposite-profile switch case in `gemma4_serving_gpu_test`
reaches this recipe (at 16-row chunks, 2 slots), and it passes; no unit test
covers the 1024-row, four-owner geometry, which this report's runs exercise.

## Provenance

Native ran on main `171cadb` plus this change (spark-b build tree, NVCC 13.4
SDK). The reference is the official image
`ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f…0607db` (v0.6.0/d812350) with
`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`. Its helper is the dense31 bridge's
`llama_serving.cc`, changed only to accept 26B's shape, take its batch from
`JITLLM_DIAG_CHUNK` and allow profiler injection. Raw logs, heads and token
files stay external on spark-b.
