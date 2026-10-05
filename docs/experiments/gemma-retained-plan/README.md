<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma cold versus retained prefill plans

Retaining all prefill plans reduced elapsed prefill by 7.12% for Gemma26 and
4.77% for Gemma31 in this one-pass diagnostic. Complete prefill/final heads,
initialized state, and all 32 incoming-head choices remained byte exact with
both the cold pass and each model's existing native baseline.

| Model / policy | Cold / retained prefill (s) | Full planning (ms) | State growth (ms) | Completed execution (s) | Cold misses / retained hits |
| --- | --- | --- | --- | --- | --- |
| 26B all1024 | 2.717236 / 2.523736 | 125.853 / 0.003 | 63.612 / 59.527 | 2.520972 / 2.456943 | 8 / 8 |
| 31B both256 | 12.064975 / 11.489527 | 526.341 / 0.036 | 164.280 / 162.258 | 11.342839 / 11.294893 | 32 / 32 |

Planning is a material cold cost, but completed execution remains the largest
retained interval. The entire cold/retained difference is not a planning-cost
estimate: execution also changed by 64 ms / 48 ms. No fresh reference run or
competitive speed, quality, or batching qualification is claimed.

The copied manual helper disables graphs in both passes and uses the existing
8K input, F16 KV, context 16384, six discarded weight-warming tokens, three
untimed anchors, and 32 forced completed units (8227 positions). The policies
are unchanged: Gemma26 norm/RoPE + norm/residual + routing/reduction at cap1024;
Gemma31 norm/RoPE + norm/residual at cap256; plain norm fusion and other optional
policies are off. It still publishes 8 / 32 prefill heads. After weight warmup
it drops plans and clears state for the cold pass, then only clears state for
the retained pass. The retained caches contain 9 / 33 entries, including one
decode shape; every prefill call hits and none rebuilds.

Off-by-default runner accounting encloses the full `Planned()` call, including
Find/Add and hit paths, rather than summing historical stored build durations.
Checks, input construction excluding state growth, state growth, planning,
staging, completed execution, and publication/cleanup are disjoint caller wall
intervals. `TakeTimes` resets immediately before paid prefill. Its dispatch,
submission, and completion fields subdivide the job wall; they are not added
to the enclosing phase wall. CUDA event stream elapsed includes submission
gaps and is not kernel-active time. Neither measurement establishes CPU busy
or critical-path attribution. Printed `last_built` counters can refer to the
prior decode plan in the retained pass and do not claim current dispatch.

The next candidate is explicit state-only intermediate prefill: preserve every
final-layer KV write while omitting its otherwise discarded query, attention,
output, FFN and head work. Its gain and exact state/continuation behavior still
require a separate implementation and matched screen; toggling `head=false`
does not provide this contract.

Build only `jitllm_gemma_retained_prefill` on the Spark-native preset. Invoke
`jitllm_gemma_retained_prefill ARTIFACT IDS_I32 NEW_DIRECTORY 26` (or `31`)
under the installed supervisor with `umask 077`. Supply the approved prepared
artifact and the canonical 8227-token IDs listed in [results](results.json),
then compare both arm directories' two heads and streamed initialized-state
hashes to those records. Input/recipe ancestry is retained in the
[26 ring recipe](../gemma26-swa-ring-transfer/README.md) and
[31 ring recipe](../gemma-swa-ring-h1/README.md).

Both supervised model runs and exact-output checks passed; the narrow build
passed. No unit suite was run for this diagnostic. Results record the native
binary, dependency receipt, source hashes, actual hosts/driver, and fresh
[TensorFold](https://github.com/ashhart/TensorFold) pin (609ca419, 0.6.5; Gemma26
recipe remains MLX). Raw logs, state, vectors and job records remain external.
