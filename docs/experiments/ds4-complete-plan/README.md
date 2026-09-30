<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Complete configured ds4 comparison

The owner requested an initial complete ds4 pipeline in jitLLM under the
same community GGUF, value/storage precisions, chunking and output scope,
followed by mechanical one-piece A/Bs. Measure that first complete baseline
independently of the quality gates; a diagnostic quality failure remains
visible and does not suppress its performance result. Quality gates still
precede adoption or a production default change.

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

## Remaining complete-plan work

Use one alternate prepared expert representation, with canonical GGUF and
per-tensor provenance, rather than keeping full raw and aligned expert
weight sets resident together. Add separate accounted raw/packed cache and
frontier resources, actual per-layer visibility counters and a distinct
state identity. Port the original dense/HC, compressor, indexer, all sparse
attention branches, router, routed/shared FFN, fused epilogues and full
requested head, including their preparation and completion boundaries.

The first complete-model control is all 43 layers at 8K, with matched
4,096-row chunks and actual original default cache/fast-path settings.
Retain a dispatch receipt for every stage and fail closed on a missing
implementation; ordinary native arithmetic is not a silent substitute.
Then measure a fresh same-configuration baseline and change one piece at a
time. Long-context, own-repeat, same-GGUF oracle/PPL, practical-answer and
state gates remain separate subsequent evidence.
