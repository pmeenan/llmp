<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 packed C4 with checked routing and reduction

Adding the existing checked routing and scaled-reduction contracts to packed
attention reduced the Gemma26 screen from **68/128 to 2/128 positive-margin
argmax differences**. **92/128 complete vocabulary heads are byte exact**
against both fresh original-reference bookends. The compound candidate remains
unselected: two strict choices remain outside its frozen zero own-repeat
movement, and its paid latency is **7.1498% above** the reference mean. This is
one short C4 diagnostic, not model support, corpus PPL, or a performance gate.

The additive benchmark is based on `c309575`. The existing packed26 wrapper,
production graph, planner, kernels, runtime, weights and primitive defaults are
unchanged. Control and candidate both pack local/global attention and select
norm/RoPE plus norm/residual; the candidate only enables the existing checked
Gemma routing and scaled reduction throughout prefill and decode. Generic
products, row-invariant products, shared-Q8, standalone norm fusion, cache-store
fusion and feature retention remain off. The exact setup and funding contract
is in [PROTOCOL.md](PROTOCOL.md).

## Results

| Compared with fresh stock C4/u128 | Packed norm control | Packed compound |
| --- | ---: | ---: |
| Complete heads checked | 128 | 128 |
| Complete heads byte exact | 0 | 92 |
| Strict argmax differences | 68 | 2 |
| Differences with positive reference-winner-over-native-choice margin | 68 | 2 |
| Strict choices at a reference tie | 0 | 0 |
| Maximum raw F32 logit delta | 32.1520195 | 4.36018276 |
| Selected 16-head maximum TV | 0.939187272 | 0.0318467752 |
| Selected 16-head maximum absolute forced-target NLL delta | 5.66798093 | 0.0647973804 |
| Selected 16-head signed mean NLL delta (native minus reference) | 0.264771831 | -0.00563730346 |

Selected likelihood rows are all four owners at waves 0, 1, 2 and 31. Their
next forced target is the canonical input at query position plus one. The 16
rows are not a PPL corpus or an allowance for the other 112 heads. Strict and
byte comparisons cover every head, using all 262,144 logits; byte equality also
distinguishes signed zero. Both reference arms have the historical stock head
SHA `5808306e…`, with complete byte-repeat equality.

The candidate first wave has 2/4 exact heads and no strict misses. Its first
three waves have 8/12 exact heads and no strict misses; the later 29 waves have
84/116 exact heads and two strict misses. Owner exact-head counts are
27/32, 23/32, 19/32 and 23/32. Their first nonexact query positions are 67, 68,
70 and 72 respectively (waves 0, 0, 1 and 2). Query positions below are zero-based consumed input positions; the next
forced-target position and completed-position boundary are query position plus
one. The two strict misses are:

| Wave | Owner | Query position | Native ID | Reference ID | Reference winner over native choice |
| --- | --- | --- | --- | --- | --- |
| 3 | 1 | 71 | 107 | 108 | 0.0279254913 |
| 4 | 3 | 74 | 249598 | 246977 | 0.365622997 |

The indexed misses, per-owner/per-wave counts and aggregate comparisons are in
[results.json](results.json). Full row records and logits stay external.

Each policy's independent first/repeat run is byte exact over all 128 heads and
four complete initialized state witnesses. Each owner contributes 57,671,680
bytes across 60 ranges; completed positions are 99, 100, 101 and 102. These are
same-policy repeat proofs, not cross-policy or cross-engine state equality.
The fresh control also exactly matches the previous packed26 head and state
hashes. Candidate paid heads, initialized states and actual 4,096-byte canonical
input file match its own frozen acquisition exactly.

Actual first-built plans for every prefill of 64/65/66/67 rows and final four-row
decode select 60 norm/RoPE and 90 norm/residual steps. Control selects zero
routing/reduction steps; compound selects 30 of each. Other optional counters
are zero. Both retain the paid pack copies, complete computed roots, state writes
and full-head publication; each timed native arm uses 32 graph replays and
zero new captures.

## Paid bookend

| Arm | 32 completed C4 waves / 128 heads |
| --- | ---: |
| Fresh original reference A | 0.997136 s |
| Compound native | 1.067470 s |
| Fresh original reference B | 0.995346 s |

All arms use the same physical Spark, canonical histories, four independent
owners, per-owner context 256, F16 KV, one-query decode and full vocabulary
publication followed by CPU argmax. Native timing pays packing, source staging,
model execution, completion and publication. Finite scans, state snapshots and
raw-file writes occur outside the timer. The original image retains production
fusion and allows CUDA graphs; its reported 32 GGML graph reuses are host graph
observations, not proof of 32 CUDA replays. Supervisor wall times are not the
comparison. No comparable physical peak or throughput ladder was measured.

## Interpretation and reproduction

[Stage A](../gemma26-dispatch-observation/README.md) observed that stock selects
all 30 routing/reduction fusions for four-row decode. For each 64–67-row prefill,
stock selects routing at layers 0–27 but refuses layers 28 and 29 through the
physical-memory gate; scaled reduction remains selected at all 30 layers. This
candidate uniformly selects both contracts at all 30 layers, so its prehistory
is not an identical stock recipe. The residual misses do not prove an alias
bug, a specific late-layer cause, or an attention-only error. No layer whitelist
or incidental-allocation imitation is used. A genuine common-input block probe
is needed before attributing the remaining difference.

[provenance.json](provenance.json) records the native/source/build receipt,
original image/client/header/math-library identities, approved artifact and
raw-model pins, immutable own freeze and official job-log hashes. All paid arms
ran on Spark A (`spark-c4e2`, NVIDIA GB10). The inherited earlier after-batch
driver observation is 580.178.04; this batch does not provide a fresh driver
trace. The authenticated native SDK pins CUDA 13.4.92 and Clang 22.1.8; the
same immutable original image inherits its previously authenticated CUDA
13.3.0/CUDART 13.3.29-1 environment. TensorFold was
refreshed at this task's entry on 2026-10-05 at 14:48:25 UTC: official HEAD
`609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5, still provides the
Gemma26 MLX recipe and no matching CUDA reference. The digest-pinned original
llama.cpp image remains the same-format comparator. Native acquisition sources
are separately authenticated from the post-acquisition `breakdown.py` analyzer;
no executable or model math changed after original exposure.

Build the additive `jitllm_gemma26_attention_compound` target and the unchanged
`jitllm_gemma26_attention_packed_metadata` target with the declared Spark SDK.
Under installed GPU supervision, run `test_controls.py NATIVE_BINARY`, then
`prepare.py inputs EXPECTED_CHECKOUT_MANIFEST`, `reference.sh build`, and
`prepare.py prefreeze`. Use `PYTHONDONTWRITEBYTECODE=1` for these Python commands.
Each native invocation is:

```text
JITLLM_GEMMA26_PACKED_C4=1 NATIVE_BINARY ARTIFACT OUTPUT 26 4 joined norm|compound IDS
```

Acquire control-first/control-repeat with `norm`, candidate-first/candidate-repeat
with `compound`, then run `own_freeze.py ROOT` exclusively before any fresh
reference acquisition. The five-step bookend runs `verify.py reference FREEZE_SHA`,
`reference.sh run reference-a`, the compound native arm into candidate-paid,
`reference.sh run reference-b`, and `verify.py paid FREEZE_SHA`. Compare both
policies using `compare.py ROOT control|candidate REF_A REF_B FREEZE_SHA OUTPUT`.
`breakdown.py ROOT FREEZE_SHA` supplies the post-acquisition indexed aggregates.
All raw logs, states and vectors remain outside Git.

The locked native build, 24 descriptor controls, 10 CLI refusals, comparator
controls and sparse binary-size boundary control passed. Official preparation,
four native arms, exclusive own freeze, five-step bookend, two comparison passes
and indexed breakdown retired successfully. Author light checks passed:
1,520 REUSE/header files, 392 boundary files, changed format/diff, Python/JSON
parsing, shell syntax and comparator controls.
