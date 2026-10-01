<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Temporary complete ds4 benchmark

The goal is to match the pinned ds4 pipeline inside jitLLM at the same
model, value/storage precisions, dispatch, chunking and output scope, then
restore native stages one at a time to identify the causes of the performance
gap. Useful techniques can then be adapted to jitLLM's architecture and the
literal reference retired. jitLLM keeps its own streams, paging, dispatch
and completion ownership throughout.

The initial scope is all 43 layers, two 4,096-row chunks at 8K, and one
final full-vocabulary head. Measure this initial performance baseline
independently of the quality gates; quality qualification still precedes
adoption or a production default change. The matched reference now also
completes [32K on the same Spark](../ds4-matched-32k/README.md), exercising
deep original selection: all four complete native heads match all four
original heads byte for byte, with throughput 0.71% lower. Both results
cover one request on its ordinary stream; ordinary suffix/decode, concurrent
requests and larger contexts remain separate work.

The complete 8K native reference now matches all 129,280 original logits
byte for byte and runs within 2.4% of original ds4 throughput. This establishes
the initial pipeline baseline, rather than a production quality qualification.
The first native attempt exposed a binding mismatch: the canonical importer
already flattens `out_a` to 4,096 × 8,192. The corrected plan preserves that
physical order, refuses duplicate tensor identities, and validates every
bounded read before payload preparation or CUDA initialization.

## Complete native 8K result

Spark A runs the same authenticated community artifact and exact 8,192 IDs
with the original precisions and dispatch, context 8,192, two 4,096-row
chunks, 43 layers and one final full-vocabulary head. An explicit warmup
precedes three fresh, initialized passes. Each resolved chunk records its
actual stage choices and paid scratch. Preparation and weight loading are
outside the prefill timer, as they are for the original reference.

| Engine | Fresh pass seconds, including final result copy | Mean seconds | Tokens/s |
| --- | --- | ---: | ---: |
| Original ds4, Spark B | 7.355353121 / 7.368073092 / 7.387095276 | 7.370173830 | 1111.507027 |
| Native reference, Spark A | 7.547550773 / 7.551976415 / 7.559512026 | 7.553013071 | 1084.600268 |

The native reference takes 2.48% longer and delivers 2.42% fewer tokens/s,
inside D-085's 10% performance threshold. This initial comparison uses
separate Sparks; causal stage comparisons will run on the same Spark.
All three native heads are finite and byte-identical to the original
head: 129,280 equal entries, maximum absolute error 0, normalized mean
square error 0, softmax total variation 0, and the same greedy ID 554.
Head SHA-256 is
`499a05df44162d26dd151f44003388a68a494c86265ea756fecdabed85ae13b8`.
These are fixed-input pipeline results, not a task-quality suite or a
long-context/decode qualification.

The metadata-only preflight covers 474 unique tensor entries, all 129
expert arrays, 11,008 replaced groups and 33,377 source chunks. Preparing
84,512,276,480 stored bytes takes 293.073 s; subsequent native page-in takes
6.396 s. Preparation uses bounded staging rather than resident raw and
aligned expert replicas. The process samples a maximum MemAvailable drop
of 102,627,532,800 bytes and peak RSS of 898,461,696 bytes at 250 ms intervals;
the node measure includes paging/cache. No original memory trace was
collected in this matched run, so it establishes no comparative memory gate.
Native and supervisor exit 0 and the strong retirement gate clears with
117.101 GiB available. Source and binary hashes are unchanged after execution.

Measured on Spark A, 2026-09-30. Raw source, qualification and model records
remain outside Git under
`spark:~/scratch/m3-ds4-complete-r1/{merged-check-r2,model-r2}/`.

| Identity | SHA-256 |
| --- | --- |
| Native qualified source map, 1,224 files | `693707103ebbe3e694d4e074b734f73e5f2d6814fccbdf95f334502e82e8b9d3` |
| Final locked qualification receipt | `09633db943664801994ea2e657312386b0d462dcc4a4bd454b70487091575a14` |
| Native complete-plan binary | `2067673fd46132d7682d3427319ec412ffd8408abd3dca28681ddf7bc66600be` |
| Metadata-only plan | `523b9c9d0f618d443b445d72a179818a9a38aa09b18aabf6d6f73faef735066b` |
| Native model receipt | `2a9c962a8952e7af4577cf245613468b40e64ffccca88940629e0ae8fc10675f` |

The next comparisons restore native stages individually while holding the
model, precisions, inputs, chunking and final-head cadence fixed. Account
for preparation, physical layout, synchronization and interactions before
adapting any technique. Production defaults are unchanged. Retire literal
reference code that no longer serves a comparison after the study.

## Matched original 8K reference

Spark B runs the same community GGUF and exact 8,192 token IDs, with context
allocation 8,192, original default cache/product settings, `quality=false`,
no MTP or DSpark, and the full 129,280-entry head only after the second
chunk. No `DS4_*` tuning overrides are inherited. One explicit 8K warmup
precedes three measured passes; every pass has a newly allocated, initialized
and fenced session, so it cannot reuse a prompt prefix.

| Pass | Prefill seconds | Tokens/s |
| --- | ---: | ---: |
| 8K warmup | 12.490182830 | 655.875107 |
| Fresh 1 | 7.355353121 | 1113.746664 |
| Fresh 2 | 7.368073092 | 1111.823932 |
| Fresh 3 | 7.387095276 | 1108.960924 |
| Measured mean | 7.370173830 | 1111.507027 |

Mean throughput is 8,192 divided by the mean measured time. All four
complete 129,280-F32 logit rows are finite and byte-identical, SHA-256
`499a05df44162d26dd151f44003388a68a494c86265ea756fecdabed85ae13b8`.
This original own-repeat result supplies the head for the complete native
comparison above; neither result establishes task quality.

The public original sync API conservatively refuses a prompt whose length
equals its context allocation. A temporary external C frontend therefore
includes unchanged pinned `ds4.c` and calls its existing static
`metal_graph_prefill_chunked` entry directly. NULL progress/display
callbacks preserve the two 4,096-row, 43-layer chunks and final-only head.
Those cadence counts are source contracts, rather than instrumented counters.
The frontend commits the session timeline and copies the full logits after
the timer; it does not decode a token.

The timing includes the original routine's final GPU-to-host logits read.
Fresh state allocation/reset/completion, the extra public host logits copy
and file serialization are outside the timer. Original startup and its
normal boot warmup precede the explicit 8K warmup. For the native comparison,
report `prefill_wall_seconds` and `result_copy_seconds` separately and use
their sum for parity. Native `tok_s` alone excludes that copy; any additional
host vector copy in the native sum is charged conservatively.

The external frontend rebuilds only the host translation unit with SDK
Clang 22.1.8 and the exact original Make C flags:
`-O3 -ffast-math -g -march=armv8-a -Wall -Wextra -std=c99
-D_GNU_SOURCE -fno-finite-math-only`. It reuses the ten original numerical
and support objects without rebuilding CUDA. The actual CUDA 13.4.92
objects retain `-O3 --use_fast_math -lineinfo` and `sm_121a`; the resolved
cuBLAS/cuBLASLt 13.8.0.4 and CUDA runtime libraries belong to the checked
`aarch64-e0a0c85c42806fb1` SDK.

Measured on Spark B, 2026-09-30; native process and supervisor exit 0,
complete post-close receipt, strong retirement gate clear at 117.188 GiB.

| Identity | SHA-256 |
| --- | --- |
| Community GGUF | `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0` |
| Exact 8192-ID TSV | `0cbafc4bafd7a2c1e1c83f4a346815cdd500d2fb0bcb45afcfd826d591ae8e9a` |
| Original host `ds4.c` | `ebc28cabc062c347daf5811331778b4ac27bf70482472214289e4aed1db4d9db` |
| Original `ds4_cuda.cu` | `8d5de76a7aaaf9131ba8f9cf35412863fef88299b39676ea4bcfb07aef7386e3` |
| Reused `ds4_cuda.o` | `913de893e23880b7026e6783e68af3eb21aeb50071f4ff104f71980e3e5c8f82` |
| Temporary frontend | `3315aa882e436b373b7cdb81f3580579827bbe73e4b536f43ad589489e9f1e4b` |
| External source/input manifest | `7da032bc750427a5911c84c0c6cecb158e7a5ef66fb3077fa045238b9db1d70a` |
| Original oracle binary | `96648b30cd9cfc76ea7a343b86a1638f75d977171328004bf8db264e9340e8f7` |
| Complete build/object/library receipt | `e7ed8bdd2b93c0131a584b3d3645670cce60eec84ddd1208a35a46c7b2fa7d8e` |
| Complete model receipt | `83eee7d816cba2b67c0009d3236b4621f5659035e56d6d0b7627c014dcd69208` |

The original source is Entrpi/ds4
`76d51ef82a81b70b78e51a3a6ea11946286de976`, archive SHA-256
`731e037da1bed009da5db31e5170681b59f1e22af00a3e56fa29cff8706ed63c`.
External source, build and raw results remain under
`spark-b:~/scratch/m3-ds4-complete-original/{source-r1,build-r1,model-r1}/`.
The authenticated community model is
`~/.local/share/jitllm/models/antirez/deepseek-v4-gguf@f71f23d5/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`;
the TSV is `source-r1/jitllm-ds4-complete-original-8192.tsv`.
The build receipt records every reused object, exact compiler/link command,
resolved library and source hash. This external oracle runtime is temporary
and is not incorporated into jitLLM.

## Numerical components and physical state

The native reference uses reviewed MIT numerical derivatives and checked
borrowed-view launchers. Original runtime objects, global registries,
allocators and foreign dispatchers are excluded. Source provenance beside
each core identifies the exact original functions and adaptations.

| Component | Original contract retained |
| --- | --- |
| Embedding/dense/RoPE/head | Original F16 embedding/HC lookup, F32-to-F16 and fused Q/KV RMS, wide F16 cuBLAS, small F16/Q8 vector tiers, canonical Q8_1/D4 producers, Q8 MMQ/aligned D2R, grouped out_a, rotary transforms and full head; thirteen unchanged original MMQ headers |
| Cache/QAT | Eleven functions: E4M3 rounding over seven 64-channel non-RoPE blocks; 704-byte packed KV plus seven F32 scales; D128 Hadamard/E2M1 indexer rounding and 64-byte packed rows plus four scales; raw F16 round trip into physically F32 ring storage |
| Normalization/HC | Twelve functions: plain/weighted RMS, F16 and compact Q8 sidecars, HC4 twenty-iteration Sinkhorn, weighted sums, residual expansion, guarded six-slot expert sum, fused next RMS and final-head weights |
| Compressor | Nine functions: original ratio-4/ratio-128 state, pooling, normalization, RoPE and mandatory last-four-row refresh, with finite C boot/reset versus the original CUDA IEEE-infinity resets preserved |
| Aligned repack | Three original IQ2_XXS/Q2_K/Q8 kernels, bounded native direct reads and fenced staging into one alternate prepared representation |
| Indexer | Thirty-four complete score/select/query-preparation definitions, including the paid MXF4 producer chain and exact streaming/tie selection |
| Sparse attention | Sixty-eight complete definitions: raw/mixed prefill, selected/banked/live-count tiers, split/head-group combines and token-tile mirror, union/sort, causal records and HMMA; explicit decode table |
| Router/routed/shared FFN | 115 unchanged original numerical functions/templates: authoritative selected-six routing, original maps, D4/canonical Q8_1 input, IQ2 G1, weighted/clamped SwiGLU, D2S6 preparation, Q2 down and guarded sum |

Dense/vector products preserve the original numerical spans and thirteen
authenticated headers in a private namespace. Each sidecar producer carries
its source pointer, shape and generation; input conversion and physical tile
padding are paid. Spark A's earlier locked slice passed 1,135 tests,
including 234 GPU tests, plus SDK format/tidy, boundaries and REUSE/header
checks. Independent controls cover double-reference products, F16
halfway/subnormal rounding, byte-exact activation forms, every full-head
logit and current-operand graph replays.

The indexer/attention slice preserves all 102 complete definitions and
retained regions after the recorded decode-table/diagnostic substitutions
and MXF4 architecture guard restoration. Mirrors, query requantization,
unions, records, score/select storage and completion remain explicit paid
work. Spark A's locked slice passed 1,168 tests, including 248 GPU tests,
in 77.27 seconds, with the actual SDK format/tidy, boundaries and
REUSE/header checks. Controls cover packed QAT scores, exact deep/tied
selection, future-cell masking, ragged query-mirror bytes, F64
scalar/head-group results, partial/deep unions, sinks and graph operands.
Check receipt SHA-256:
`c4a950f4f03535206c0d5985d1a09313605fa8b3bac2e482f216c93c26a4e4b6`;
source-correspondence SHA-256:
`9aeb3249cc7d219d9f506beadc54a2075b5866d8762fe47d4578cb871a81bc66`.
These prior slice checks establish operator coverage, not a completed
configured model.

Complete declared extents account raw and packed state, compressor frontiers,
alternate weights, product sidecars, maps and scratch. Raw and aligned expert
weight sets are not both retained as complete resident copies. Packed
token-tile/MXF4 selection is explicit; an uninitialized logical F32 proxy
cannot become an arithmetic fallback. Each resolved stage must have its
own dispatch/paid-scratch receipt and current operands.

The FFN source-correspondence proof verifies 115 original functions unchanged
and ten unchanged prepared headers (SHA-256
`691bcddc9d74f09eebaee273df05c6265784cde4e9d701d4034a5ae6f49f8e86`).
Spark A's final combined locked check passes 1,197 tests, including 256 GPU
tests, with the actual SDK format/tidy, boundaries and REUSE/header checks.
The corrected FFN controls pass in that suite. Independent controls reconstruct
the original direct-F16 Q8_1 scale,
reciprocal-F32 D4 scale and reciprocal-F16 D2S6 scale, including a
minimum-half-subnormal witness. They retain the existing 0.5%/1e-7
physical bound and byte-exact tier/map/repeat/current-sidecar controls.
Direct/materialized D2R admits full 128-row output tiles and even tails
below 16 rows; larger partial tails are refused before packing because
their full/guarded warp paths have different CTA barriers.

Original serial live-scalar routing and the default materialized/shared-MMQ
tier below 1,024 assignments remain explicit follow-ons. The implemented
Classic tier has a different original activation and is diagnostic.
Primitive controls do not qualify ordinary decode, complete-model equality,
quality or performance. The complete fixed-input execution above supplies
the initial equality and timing evidence. Native-stage restoration,
long-context, quality and state gates remain separate.
