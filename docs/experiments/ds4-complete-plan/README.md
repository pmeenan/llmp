<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Complete configured ds4 comparison

The owner requested an initial complete ds4 pipeline in jitLLM under the
same community GGUF, value/storage precisions, chunking and output scope,
followed by mechanical one-piece A/Bs. Measure that first complete baseline
independently of the quality gates; a diagnostic quality failure remains
visible and does not suppress its performance result. Quality gates still
precede adoption or a production default change.

The literal path is temporary benchmark scaffolding (owner, 2026-09-30).
After it matches the configured reference pipeline and performance, restore
native stages one at a time under the same inputs and settings to identify
which pieces explain the speed. Charge each stage's preparation, conversion,
layout and completion work; test interactions where independent gains do
not add up. Adapt the useful techniques into jitLLM's existing architecture
and check other model/kernel consumers. Retain our native state, dispatch,
catalog and provider contracts wherever they do not cause the difference.
Retire unused literal code from shipping builds after the comparisons;
durable provenance, controls and aggregate results remain. A permanent ds4
runtime or parallel serving architecture is not the intended deliverable.

That complete plan is not implemented yet. The existing HCA and Q2 product
experiments are individual transfers over native inputs/state. Their speed
ratios do not demonstrate complete ds4 performance parity.

## Cache and quantization-aware transforms

The first numerical stages are explicit native cache adapters:
`dsv4_ds4_cache.h/.cc/.cu`, with the pinned original MIT numerical core in
`dsv4_ds4_cache_core.cuh` and extraction provenance beside it. The source
is Entrpi/ds4 `76d51ef82a81b70b78e51a3a6ea11946286de976`, original
`ds4_cuda.cu` SHA-256
`8d5de76a7aaaf9131ba8f9cf35412863fef88299b39676ea4bcfb07aef7386e3`.
The original numerical CUDA options are retained. No original runtime,
stream, allocator, global sidecar registry or cache counter is imported.

| Stage | Original value and storage contract |
| --- | --- |
| KV QAT | F32 in place, seven 64-channel power-of-two-scaled E4M3 blocks over 448 non-RoPE channels; 64 rotary channels remain F32 |
| Packed KV | 704-byte row: 448 code bytes plus 64 F32 rotary values; seven F32 scales add 28 bytes |
| Indexer QAT | Original normalized D128 Hadamard, then four 32-channel power-of-two-scaled E2M1 blocks, retaining rounded F32 values |
| Packed indexer | 64 code bytes and four F32 scales (16 bytes) per row; query preparation can emit only the scales |
| Raw store | Every QAT row round-trips through F16 into physically F32 ring storage, with original chronological wrap |
| Packed expansion | Original packed-row reconstruction into an explicit F32 destination; the caller accounts and retains the immutable 512-byte KV decode table |

All operands are caller-owned checked logical views. Launchers allocate no
scratch and queue on jitLLM's stream; completion remains the caller's
responsibility. They refuse invalid sizes, overflow, alignment, aliases,
incomplete packed outputs and repeated raw write slots. The raw store's
host position cannot be replayed safely, so it explicitly refuses capture.

CPU controls exercise those refusals and the decode table. GPU controls
exercise analytical ties, signs and scales, exact packed/expanded bytes,
Hadamard impulses, query scale-only preparation, F16 rounding/wrap, guard
bytes and raw capture refusal. These are operator controls, not model
quality or performance evidence. On Spark A (2026-09-30), the pinned
original source, all eleven verbatim functions, scalar ABI and both
table-pointer adaptations were verified; the locked native build and full
1,059-test suite passed, with SDK format/tidy, boundaries and REUSE/header
checks. The complete configured model plan remains unqualified.

## Normalization and hyper-connections

`dsv4_ds4_hc.h/.cc/.cu` exposes twelve complete original numerical functions
under the same pin and CUDA options: plain/weighted RMS with F16 and compact
Q8 sidecars, the HC4 twenty-iteration Sinkhorn split, weighted sums, fused
preparation, residual expansion and final-head weights. Original wide-row
eligibility is explicit. Fused next-layer expansion can consume six guarded
expert contributions in ascending slot order and emit the F16 normalized
input for the next projection. Its arithmetic remains the original fused
path; equivalence to a separate expansion and RMS is not claimed.

Checked views exclude partial aliases and account complete sidecar extents;
only an eligible RMS source/F32 output may alias exactly. CPU and GPU controls
cover refusals, independent F64 references, compact Q8 scales/signed codes,
nonfinite expert guards, reduction order, ragged rows, current-operand graph
replays and exact own repeats. The final combined Spark A check above covers
these operators and all twelve original source ranges, with 1,027 headers
checked. Physical model state and the complete pipeline remain separate work.

## Compressor frontiers and aligned weight preparation

`dsv4_ds4_comp` retains nine complete original compressor/RoPE functions.
It exposes original zero-prefix, aligned and ragged pooling/state updates,
learned RMS, rotary-tail coordinates and mandatory FP8/FP4 QAT. Boot/Clear
uses finite score `-1e30`; original later resets use negative infinity.
Ratio-four bulk calls explicitly require the caller's separate small
last-four-token projection products and refresh before continuation.
Reusing wide projection tails would change the original state rounding.

`dsv4_ds4_repack` exposes three complete original preparation kernels from
`cuda/mmq/ds4_repack.cu` (SHA-256
`ce29f7c50e6bc9922a08bf456d5204e684ae74f5618e63dbd583a81edab15153`).
It converts bounded raw IQ2_XXS, paired-row Q2_K and dense Q8 blocks into
the original aligned sections. One initialization establishes padding;
chunks preserve every scale/code bit and require proved completion before
staging reuse or file output. No original allocator or weight server is used.

Spark B's final combined check (2026-09-30) verifies all 35 original
cache/HC/compressor/repack functions, both explicit decode-table adaptations
and scalar ABI. The locked native build passes 1,110 tests, including 223
GPU tests, in 69.78 seconds, with 24 SDK format checks, all twelve host-unit
tidies, boundaries and REUSE/header checks (1,047 headers). Source inventory
and bytes remained unchanged. Compressor controls cover F64 pooling,
4,096-row norm/RoPE/QAT, distinct refresh inputs, ragged frontiers and graph
operands; repack controls cover independent full byte permutation/inverse,
special scale bits, chunks across experts, padding/guards and graph inputs.
Check receipt SHA-256:
`2eb67bf9247a8c7982a4dda97954045c21a63cabb8bcf5bd0a94f14921c1e3e5`.

These qualify primitives. The file producer, complete alternate weight set,
physical model state and whole configured pipeline are still pending.
Workstation checks remain deferred until the implementations settle.

## Original dense and vector products

`dsv4_ds4_product` supplies original F16 embedding/HC lookup, the explicit
F32-to-F16 input conversion, fused wide Q/KV RMS, the wide F16 cuBLAS product, small
F32-activation/F16-weight vector tiers, compact D4 and canonical Q8_1
activation producers, Q8 MMQ and aligned dense D2R, small full-row Q8
products, head norm/RoPE and the grouped own out-a HMMA with optional D4
output. Each producer carries its source pointer, shape and generation;
scratch, physical tile padding and input conversions are explicit.
The original numerical definitions and thirteen retained MMQ headers are
isolated in a private namespace. The existing production D2R unit continues
to use the locked GGML headers through explicit includes.

Spark A's final locked slice passes 1,135 tests, including 234 GPU tests,
the actual SDK's format/tidy, boundaries and REUSE/header checks. Complete
numerical spans and all thirteen original headers match the authenticated
source. Controls cover independent double products, halfway/subnormal F16
rounding, signed positions, byte-exact activation forms, every full-head
logit, padded/ragged reads and writes, current graph inputs and refusal
before submission. Independent whole-source and adversarial reviews are
clean. No model speed or quality result follows from these products alone;
workstation checks remain deferred until the implementations settle.

## Remaining complete-plan work

Use one alternate prepared expert representation, with canonical GGUF and
per-tensor provenance, rather than keeping full raw and aligned expert
weight sets resident together. Add separate accounted raw/packed cache and
frontier resources, actual per-layer visibility counters and a distinct
state identity. Assemble the checked HC/compressor stages and port the
original indexer, all sparse attention branches, router,
routed/shared FFN, fused epilogues and full
requested head, including their preparation and completion boundaries.

The first complete-model control is all 43 layers at 8K, with matched
4,096-row chunks and actual original default cache/fast-path settings.
Retain a dispatch receipt for every stage and fail closed on a missing
implementation; ordinary native arithmetic is not a silent substitute.
Then measure a fresh same-configuration baseline and change one piece at a
time. Long-context, own-repeat, same-GGUF oracle/PPL, practical-answer and
state gates remain separate subsequent evidence.
