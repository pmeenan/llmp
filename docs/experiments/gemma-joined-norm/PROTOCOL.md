<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense31 joined norm/row-policy first screen

This is a separate benchmark-only candidate at the natural C4 shape from the
joined-serving experiment. It does not mutate that experiment's source, own
freezes, failed quality results or calibration. Production defaults remain off.
Acquire on the integrated joined-serving commit once its final root job retires;
source acquisition is based on 422c687 after its completed final gate.

## Candidate and closed contracts

Select only existing `Gemma4Options::row_invariant`, `fuse_norm_rope` and
`fuse_norm_add`. Legacy two-node norm fusion, generic fusion, sharedQ8,
RoPE/store and MoE policies remain false. No new kernel, graph matcher,
executor, source-lock patch or runtime option is introduced.

Existing norm/RoPE eligibility requires checked packed F32 D256/D512 full NEOX
rotation and original finite frequency parameters; actual positions/factors
cannot depend on elided norm/product nodes. Residual eligibility requires
approved 5376 width, checked F32 shape/strides and disjoint actual inputs/output.
The deferred final GET_ROWS remains paid/materialized before ADD. Planner
OnlyReader guards preserve kept/view consumers and raw K-as-V; checked bounded
chains and primitive fallback remain unchanged. Original compiled norm launchers
perform the math. This source eligibility does not assert actual model selection.

Record actual counts for each fresh initial 64+i-row prefill and first built one-/four-row
decode plan: rows, segments, row products, norm/RoPE, norm/add, legacy norm,
sharedQ8, RoPE/store and MoE choices. Row-invariant products are disabled on
these prefills because they exceed 8 rows; checked norm policies apply wherever
the existing predicates admit them. Decode stays within 4 rows and 32 routed-pair
capacity, though dense31 has no routed experts. No C12 expansion is part of
this screen.
`last_built_policy()` updates only on a cache miss. Do not present cached reset
prefill reads as fresh policy selection; guard logged first-build rows/segments
and label final counts as last-built selection. Paid capture/replay counts and
the unchanged shape keys identify reused plans without new runtime tracing.

## Matched shape and paid work

Use the approved dense31 prepared artifact 32c92e07… and same-format raw
`gemma-4-31B-it-UD-Q4_K_XL.gguf`, revision c1ac76e99d5513b141e8adde7288b85c3f9c32ec,
SHA-256 9e92cb6236044c6a9870af406029c74a76e0571c157a6f95df724dcc8c7a1575.
The exact retained 1,024-ID little-endian tokenizer control has SHA-256
b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610 and
BOS 2 exactly once. Complete corpus identity remains 3,274,124 bytes/SHA-256
c7156148ecaa12b6416cf816540d8dede2014982554a835e61076f0dd8bf0c2d.

Owner i receives IDs [0,64+i), with independent sequence/state. Discard 8 warm
units, reset and prefill the identical prefix, then 3 untimed anchors. Paid
steps 0..31 process supplied ID 67+i+step at that absolute position; the labelled
next target is ID 68+i+step. Every completed full-vocabulary head and argmax
publication stays inside the timer, with disk output outside. Context 256 per
owner/max_rows128 and 32*C=128 paid frontier rows match the prior screen.
Retained head and working-row heap is explicitly charged 142,606,336 bytes;
native state/plan/workspace/pinned output funding remains separate. No peak
claim follows from these capacity charges.

Acquire unchanged rows-only control, candidate same-policy scalar-a,
joined-first, joined-repeat and scalar-b. All four candidate full-head files
must be finite and byte exact before freezing own-repeat/source/input/binary
identities. Candidate versus unchanged control movement is an arithmetic
change, never own-repeat noise. Freeze before stock inspection/acquisition.
Abort an own-equivalence failure; do not widen a tolerance or discard the arm.

Then acquire a fresh stock ubatch4/candidate/stock ubatch4 paid bookend using the
original pinned b29c606e28a01b1bc8c1351026a0fa6e616bf6c4 C API and digest image
837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7.
Stock uses genuine four-sequence batches, effective 256 context per owner,
batch 128, F16 KV, fusion/graphs enabled. Authenticate first/repeat heads and
the exact supplied inputs. Retained ubatch4/32/128 stock vectors are eligible
for quality comparison only after checking exact input/position/format/source
recipe compatibility; label them retained, not fresh timing evidence.

Compare all 128 heads for actual byte identity, finite values, raw delta,
lowest-index strict argmax and positive reference-winner margin relative to
frozen zero own-repeat movement. Keep exact ties separate. Summarize eight
labelled TV/target-NLL rows (steps 0/7/15/30 × owners0/last); do not claim PPL or
reuse the earlier128-row teacher-forced quality result as this shape's pass.
Raw heads, IDs and per-step output remain external; Git retains aggregates,
identities, conditions and reproducible source only.

## Current source refresh and stop boundary

At this task entry 2026-10-05 07:44:19 UTC, primary TensorFold HEAD is
609ca419abecebdc5a059498a613680bd3aa847f, version 0.6.5. Its pinned
[README](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md)
and [Gemma recipe](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/docs/recipes/gemma-4.md)
remain 26B-A4B on MLX, with no dense31 CUDA comparator. Pinned same-format stock
llama.cpp is the GB10 comparator.

One representative short screen precedes any expansion. A negative candidate
gets no context/concurrency ladder or selection. Positive bounded evidence
still leaves state/capture/spill, broader quality, physical peak, competitive
C12 and full optimized-batching/model-support qualification separate. No
routine whole-unit suite is needed for this initial benchmark-only axis.
