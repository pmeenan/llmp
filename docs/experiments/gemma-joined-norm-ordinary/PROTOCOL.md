<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense31 ordinary-product norm C4 first screen

This new benchmark-only axis starts from `fdf6d1a`. It preserves the failed
[row-invariant/norm screen](../gemma-joined-norm/README.md), its immutable own
freeze and all prior ordinary/stock evidence. Production defaults stay off.

## Candidate and source-led rationale

The closed `norm` helper mode accepts only dense31, genuine joined C4 and the
same supplied 1,024-ID control. Select existing checked `fuse_norm_rope` and
`fuse_norm_add`; set `row_invariant=false`. Legacy two-node/generic fusion,
sharedQ8, RoPE/store and MoE policies stay off. No engine, graph, kernel,
registry, runtime or source-lock changes are part of this screen. Retain the
original helper funding, complete-output publication, stable owner lifetime,
completion-aware teardown and paid timer.

Native `Gemma4Runner::Choices` gates row invariance to at most eight rows.
`PlanGraph` otherwise calls `SelectMulMatQ` with the actual type/device/column
count: original MMVQ eligibility, then MMQ eligibility. Row-invariant dispatch
instead forces one-column sums for every column. The native row kernel's source
records that Q8_0/Q4_K/Q5_K/Q6_K one-column GB10 products use twice the warps of
ordinary two-to-four-column products. Disabling that override restores the
existing ordinary selection; it does not prove stock head identity.

Pinned stock `mmvq.cu:318` selects MMVQ for GB10 Q2_K up to six columns and the
other supported quant types up to its ordinary maximum batch. Stock also has
float selectors and possible graph fusions. These are source-predicate
inferences, not a captured list of actual CUDA launches. The runner exposes
last-built policy counters, not a public per-operation plan observer. No eval
callback that keeps intermediate nodes or changes aliases/fusion eligibility
is installed. Actual first-built norm/row/count metadata is recorded unchanged;
full-head comparison remains the math evidence.

## Inputs, shape and funding

Use the dense31 prepared artifact
`32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08`
and approved same-format GGUF revision
`c1ac76e99d5513b141e8adde7288b85c3f9c32ec`, size 18,822,970,304 bytes,
SHA-256 `9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575`.
External little-endian 1,024 IDs have SHA-256
`b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610`,
BOS 2 once and the same complete War and Peace corpus identity: 3,274,124 bytes,
SHA-256 `c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d`.

Owner i prefills IDs [0,64+i) in one native chunk. Discard eight warm units,
reset/prefill identically and process three untimed anchors. Paid step s=0..31
processes ID 67+i+s at its absolute position; target likelihood is for ID
68+i+s. Native context is 256 per owner, max_rows128, one real four-segment
wave per completed unit. Timer includes all 128 full-vocabulary heads and
argmax publications; output to disk stays outside. Retained/working head heap
is charged 142,606,336 bytes separately from native state, plan, scratch and
pinned output. Capacity charges do not establish physical peak.

Acquire unchanged joined `rows-norm-control`, then joined `norm` first and
repeat. Authenticate exact source/input/binary identities, finite complete
heads and first/repeat full-byte equality. Freeze externally with exclusive
creation before any stock comparison/acquisition. Candidate versus control
movement is an arithmetic policy change, never noise. No C1=C4 identity is
required: different ordinary column counts may use different reduction sums.

## Primary competitive control and stop boundary

After own freeze, acquire fresh stock **ubatch128**, candidate and stock
ubatch128 bookends. Stock uses original llama.cpp
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` and original math libraries in digest
`837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.
Its unchanged thin C-API client uses four actual sequences, F16 KV,
fusion/graphs enabled, batch128 and effective context256 per owner. Physical
ubatch128 can execute each 64..67-ID prefix as one chunk, reducing the native
versus reference prefill chunk difference. This is recipe alignment, not a
claim that every selected operation is identical.

Authenticate complete stock repeats and candidate bookend identity. Retained
stock ubatch4/32 heads are additional quality comparisons only after exact
source, checkpoint, input, position and shape recipe authentication; identify
them as retained. No retained timing substitutes for fresh primary bookends.
Compare all 128 complete rows for byte identity, finite values, raw delta and
lowest-index argmax. Keep strict differences, positive stock winner margins
outside frozen zero own movement and exact reference ties separate. Summarize
only eight labelled TV/target-NLL rows (steps0/7/15/30, owners0/3); no corpus PPL
claim follows. Raw heads/IDs/per-step telemetry stay external.

A negative first screen gets no C12/context ladder or adoption. Positive
bounded C4 evidence still requires C1 matched-reference math and repeat/state/
continuation controls before selection; previous128 teacher proof, row-policy
solo agreement and C12 8+4 evidence do not qualify this different product policy.
No routine whole-unit suite is required for this benchmark-only first screen
under the experimental workflow.

At task entry 2026-10-05 08:28:50 UTC, latest primary TensorFold HEAD is
`609ca419abecebdc5a059498a613680bd3aa847f`, version0.6.5. Its pinned
[Gemma recipe](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/docs/recipes/gemma-4.md)
still describes 26B-A4B on MLX, with no dense31 CUDA comparator. The exact
primary refresh and pinned GGML source identities remain with the evidence.
