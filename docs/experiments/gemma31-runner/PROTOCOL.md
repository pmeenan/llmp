<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded Gemma 31B ordinary runner control

This task generalizes the shared native `Gemma4Runner` to the two approved
26B-A4B and dense 31B profiles. It introduces no separate scheduler or state
lifecycle. The engine-only variant defaults to 26B-A4B; unsupported enum values
refuse before artifact opening or state allocation. The exact artifact binder
still checks every tensor role, shape, type, readable span and optional group.
The HTTP adapter remains restricted to 26B-A4B. Assistants, optimized batching,
long-context qualification and supported-model status remain separate work.

The dense source is `unsloth/gemma-4-31B-it-GGUF` at
`c1ac76e99d5513b141e8adde7288b85c3f9c32ec`, file
`gemma-4-31B-it-UD-Q4_K_XL.gguf`, 18,822,970,304 bytes, SHA-256
`9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575`.
The fully verified prepared artifact is
`32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08`,
with 9,010 chunks and seven files, importer
`m3-1+layout-a0d1980a9eddd1ad`. The checked dense contract has 60 layers,
width 5,376, 32 query heads, local/global KV heads 16/4, D256/D512 attention,
FFN width 21,504, no experts, vocabulary 262,144 and F16 state. The source
contains F32, Q4_K, Q5_K and Q6_K tensors; embedding and output head are tied.

At **2026-10-05 02:36:24 UTC (October 4 EDT)**, a fresh `git ls-remote`
resolved TensorFold HEAD to
[609ca419abecebdc5a059498a613680bd3aa847f](https://github.com/ashhart/TensorFold/tree/609ca419abecebdc5a059498a613680bd3aa847f).
Freshly inspected package version is **0.6.5**. Its
[pinned README](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md)
and [Gemma recipe](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/docs/recipes/gemma-4.md)
declare Gemma 26B-A4B on MLX, with affine group-32/64 weights and an 8-bit
router. They declare no Gemma 31B or CUDA Gemma comparator. Retain the
same-format GPU reference and do not treat the MLX recipe's rates as Spark
measurements.

Use llama.cpp `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, b10964, image
`ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.
Build the existing explicit-ID C API quality and prefill harnesses against the
image's `/app` libraries and exact pinned headers. Never modify the owner's
reference checkout or rebuild CUDA math to establish this comparison.

Use canonical `ppl.txt`, 3,274,124 bytes, SHA-256
`c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d`.
Freshly tokenize this complete corpus with the **31B raw GGUF tokenizer**,
`add_special=true`, `parse_special=false`. Observed token count is 774,039.
The first 1,024 I32 IDs include BOS 2 once and hash to
`b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610`.
The independently prepared first 8,227 IDs hash to
`6b6567ca51a3fbe5000521cb71fcf168ef485623bdbfea2abab30d57f414d96b`.
Equality with the earlier 26B token file is observed after fresh preparation;
no 26B model outputs, calibration bounds or quality results apply to 31B.

The numerical screen publishes every full-vocabulary head for eight
teacher-forced chunks of 128 rows, context 4,096, one sequence, F16 KV,
FlashAttention, no implicit tokens, template, drafts or EOG stopping. Native
ordinary products and device masks remain selected. Shared-Q8, row-invariant,
RoPE/store, lane and upstream fusion policies stay off. A separately requested
native norm-fused arm is a calibration diagnostic only. Acquire native ordinary,
norm and independent ordinary repeat before reference model scoring. Check the
ordinary repeat's complete bytes, then freeze a **new 31B** native-only top-two
margin p99 and row hashes using the existing quality analyzer in a separate
external directory. Never replace a freeze or widen it after oracle acquisition.

Acquire pinned full-fusion reference and its repeat at the same 128-row shape.
Score row j against input j+1 for j=0..1,022: exactly 1,023 targets, unscored BOS
and an unscored final retained head. Report FP64-summed mean NLL/PPL, strict
argmax differences, full-softmax variation, raw/target/chosen deltas and actual
byte hashes. If an unfused diagnostic is needed, its own repeat and full-file
byte comparison are separate controls; its agreement cannot waive a production
reference failure. These prose scores do not qualify general answer quality.

The lifecycle control funds caller output capacity before allocation and uses
the shared catalog-backed pinned allocator. Ordinary waves of 1/2/4 requests
must reproduce their own complete heads and initialized KV bytes after fresh
state, then resume exact continuations through checkpoint, clear, spill and
restore. This is same-shape replay, not scalar-versus-joined equivalence.
Checkpoint metadata retains a source-layout ID with bytes; 26B's existing
`gemma26-f16-kv-scalar-device-v1:...` stays unchanged and 31B uses a distinct
`gemma31-...` ID. All three restore/adopt entries reject wrong, empty or malformed
source identity before mutation; tests cover both cross-variant directions,
nonzero valid footprints, cursor, state extents, occupancy and peer state.
Trusted callers must preserve real source metadata; a fabricated matching tag
is not authentication of arbitrary raw bytes. Diagnostic external checkpoint
files now require their saved bounded `.layout` sidecar. Untagged historical
files are refused; their historical measured source and receipts remain intact.

Competitive timing is a separate paid 8K prefill plus 32 completed forced-prefix
decode calls. Reuse the paid-prefill harness protocol, fresh 31B IDs and new
output directories. Screen supported reference ubatches independently, then
bookend the selected arm around native. Warm weights on six discarded rows,
clear/reset, pay fresh 8,192-row prefill, and append three identical untimed IDs
before 32 identical paid IDs. Both arms publish full vocabulary outputs inside
timers; native's intermediate chunk heads remain paid. Native's 128-row envelope
never constrains the reference screen. Report retention funding, completed
counts, retained-head limitations and a measured gap without adopting a policy.

All Spark-b work runs through installed `spark-job --gpu`, timeout at most 600
seconds per job, stop on failure and official waits. The task's distinct warm
tree is `m3gm31`, base `c8edc06`; source/binary receipts and raw data remain in
external `gemma31-quality`, `gemma31-prefill` and job directories. Local heavy
checks and full-logit analysis require `hostlock shared`. Aggregate results and
source identities accompany the final report; planned charges are not measured
whole-node peak-memory qualification.
