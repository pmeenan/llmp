<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek community IQ2_XXS scalar incumbent evidence

The exact community checkpoint is
`antirez/deepseek-v4-gguf@f71f23d552d664e523b422157b2befbf74040380`,
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`,
86,720,111,488 bytes, SHA-256
`ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`.
Its published v1 artifact is
`e7c0d0e453aab30765c884ab610835936e80312646f18383538bd4d9b428dc59`.
The importer/source inventory and semantic spec authenticate its actual formats
and roles, including IQ2_XXS routed gate/up, Q2_K routed down, Q8_0 shared
experts/output and F16 compressor/router/mixing tables. The UD checkpoint's
numerical evidence is separate and supplies no authority here.

On 2026-10-18, same-host A (`spark-56f5`, GB10) controls scored the identical
967 supplied token IDs and 966 targets with final R6 and integrated R7. Both
completed normally with finite losses; the R7 score run also retained its
actual one-row stage-0 plan and zero refusal/coverage violations. The source
and runtime identities were respectively `75e22afc…`/`1e601fb1…` and
`1b0cab33…`/`ac53498c…`. The unchanged diagnostic pair failed: PPL increased
0.8049388% against its predeclared 0.5% bound, mean absolute target difference
0.0871218, maximum 2.5918649 and 70.186% of targets within 0.1.

One independent same-checkpoint reference reused the reviewed b11254
collector (source `8019dc563b1ecbae6b161a70c3a1359f1b206c1e`, executable
`3938b46b7c7469b59acd99f8c3af59c52c185bf91e66170de5d77c28c6d4d5d0`)
and pinned image `sha256:6dd025913fdf0ac258850ac1405dde3edbdb64c7c6d60533ee63e16448f66607`.
Before loading, the original source's full bytes passed the hash above. It
scored the same 967 IDs without tokenization, in batches of 128, context3072,
FA, F16 K/V, one sequence, default fusion/graphs, `swa_full=false` and
`kv_unified=false`. All 967 full rows were finite and all 966 target losses
were retained. Task-local swap and memory events were zero, the process/cache
cap and live host-memory margin held, and the container/target retired.
Host-global swap counters were informational under the resource-observer
correction fixed before these reference metrics were inspected.

| Same 966 targets | Mean NLL | PPL versus reference | Mean absolute distance | Maximum distance | Within 0.1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference | 2.84880928048 | — | — | — | — |
| R6 incumbent | 2.84903127636 | +0.022202% | 0.0900577 | 0.984396 | 690/966 |
| R7 without IQ2 authority | 2.85704844062 | +0.827320% | 0.09147 | 2.83528 | 660/966 |

The new maximum occurs at position608, target9348. These observations show a
consistent loss relative to both incumbent and independent reference, not an
accepted numerical change. The original pair remains failed and the original
frozen discrepancies remain evidence.

The retained R6 scalar IQ2 fixture selected 388 VecQ products and 302 Q8_1
preparations. The actual R7 teacher plan selects 302 prepared ordinary MMVQ
and 129 routed MMVQ products. It reports 86 missing Q8-vector qualifications
at ordinary-product anchors and 43 at routed-product anchors. These are
separate inventories, not a one-to-one recipient mapping, and the retained
fixture's long-prefix geometry is not identical to the teacher passage.
Original-GGML MMVQ component equivalence does not prove quality equivalence
to the accepted VecQ schedule for this checkpoint.

The orchestrator authorized restoration of evidenced incumbent arithmetic
through the existing typed numerical domain, limited to explicit `rows={1}`.
The source approval's existing context262144/slots4 limits describe its
configured envelope; they do not authorize two or more actual rows. Projection
binds the exact source receipts, inventory, semantic spec, active stage,
component, original weight role and exact K/N. Stored formats remain part of
that authenticated inventory; kernel capability and physical input predicates
still apply. The compound-product/GLU anchors preserve the same two independent
product contracts rather than granting a whole component indiscriminately.
No UD domain, blanket flag or multirow allowance is added.

This is incumbent-retention evidence, not qualification of a new candidate.
The restoration still requires an actual same-ID passage and state controls,
plus independent delta review, before adoption or any IQ2 frozen disposition.
Raw records live outside Git under `r7-iq2xxs-paired-teacher/results` and
`C/r7-iq2xxs-exact967-reference-results/A`; no broad quality gain, every-context
quality proof or completed family gate is claimed.
