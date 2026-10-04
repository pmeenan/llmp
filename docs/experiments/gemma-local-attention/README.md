<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma local attention primitive controls

This bounded integration supplies D256/F16 vector attention and unchanged
D256/group2 MMA kernels for Gemma local GQA2. It follows the pinned overall
GGML selector on GB10: one query row and one sequence use vector attention;
multiple query rows or sequences use group2 MMA with query tiles 4/8/16/32.
Separate one-sequence graph segments retain the solo vector choice. This is
primitive evidence; it establishes no whole-model quality, throughput,
optimized batching, VMM cache ownership or supported-model status.

## Contract and correctness

Q/output are F32, K/V/masks are F16; Q is `[256,rows,heads,sequences]`,
K/V are `[256,cells,KV_heads,sequences]`, output is
`[256,heads,rows,sequences]`, and the query/KV head ratio is exactly 2.
Cells are padded to a multiple of 256. Scale 1 is preserved. The local paths
refuse sinks, ALiBi, softcap and sparse gathers. CPU graph planning without
a device selector uses the group2 primitive fallback. Existing D64 vector,
D128 MHA and D256/D512 group8 kernels retain their normal selected identities.

The device controls cover 16/8 and 32/16 query/KV heads, query rows
1/2/4/8/16/33, and 1/2/4 independent sequences. Q is a strided permutation of
packed `[D,heads,rows,sequences]` projection storage. Each sequence has its
own K/V and causal 1024-cell ring mask. Queries cross the ring wrap; future
and cells outside the window stay masked. Complete output is compared with
an independent FP64 QK/softmax/PV calculation using actual uploaded F16 KV.
The primitive accuracy bound is the existing GGML attention NMSE 5e-4.
The largest observed NMSE across all 36 ring cases is 1.20222e-06.
Own repeats are byte-exact, and every MMA result matches a separately
launched pinned case byte-for-byte. First-sequence output agrees with its
solo mathematical control within the same declared attention bound.

A 1025-query case funds 1056 mask rows and exercises the mask prepass's final
32-query tile. Unfunded 1025-row metadata refuses before submission, leaves
an output sentinel intact and does not fault the context. The funded
single-visible-cell case compares uploaded F16 V against both pinned and
registered output: both have NMSE 2.00564e-8, and pinned/registered bytes
match exactly. No cause for that pinned numerical deviation is inferred.
The general ring calculation, rather than this sparse diagnostic fixture,
qualifies dense local attention math.

Scratch planning pays for mask scanning, partial results and fixup. A
prepass requires rounded query-tile mask extents and a mask sequence extent
matching Q; ordinary extent broadcasting cannot substitute for those rows.
Shared strides with those complete extents remain eligible. Pool high-water
usage never exceeds the planned charge. Independent metadata controls probe
the last admitted and next padded KV block for D128/D256/group2/group8/D512,
including the final unaligned stream-k iteration advance,
and the conservative vector efficiency bound for D64/D256. Combined Q
byte spans, K/V head-byte products and vector mask-row byte starts are also
bounded, while K/V sequence addressing remains int64. A D128 unpadded
KV endpoint control also bounds the pinned ceil-division addition. Refused metadata is
never submitted to a device. These conservative guards close signed pinned
arithmetic limits; some refused vector shapes might have terminated their
trial search safely before its worst-case bound ([RE-046](../../rough-edges.md#re-046-ggml-flash-attention-total-iterations-use-signed-integers--2026-10-04-status-worked-around)).

## Paid pinned reference comparison

Measured 2026-10-04 on `spark` (`spark-c4e2`), NVIDIA GB10, driver 580.178.04,
NVCC 13.4.92/Clang 22.1.8, official locked `spark-native` RelWithDebInfo SDK
`aarch64-c09daba6ac31edee`; base jitLLM 39d4c22. GGML is llama.cpp b10964,
commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, prepared tree
`1ae467d0fced412beb16faca3d0a910441477a06f1e2bae4ccbe0cdb6230bd11`.
The new wrapper instantiates unchanged pinned kernels. The source lock's
license inventory records the additional wrapper; source payload, patches
and prepared-tree digest are unchanged.

[`jitllm_gemma_attention_bench`](../../../benchmarks/gemma_attention.cc)
uses 128 launches per arm, warmed original/selected output and A/B/A CUDA
event timings. Inputs are deterministic random F32 Q and resident F16 KV,
scale 1, 1280 padded cells and causal 1024-cell ring masks crossing wrap.
Each independent segment has distinct Q/K/V/mask buffers and ring positions;
segments execute sequentially with one shared scratch maximum. Neither
cross-segment lanes nor joined attention is claimed. All buffers and
workspace are cudaMalloc allocations, initialized before launch; this does
not qualify the serving VMM/cache path.

A calls the pinned arm selected by the original GB10 dispatcher through the
same LaunchContext. B calls the registered checked wrapper. Both pay the
same mask scan and fixup work. The uncaptured column includes wrapper
planning and launch submission; it excludes full graph construction,
operand transfers and host input generation. The graph column captures the
complete paid work and measures replay. Both captured graphs replay to the
separately warmed original output byte-exactly before timing. Every reported
peak is within the planned scratch charge. Scratch bytes are a per-wave
maximum reused between sequential segments; the benchmark provisioned a
16 MiB workspace. Times are µs per complete wave, not per token or segment.

| Heads/KV | Rows | Segments | Arm/tile | Planned/peak bytes | Nodes per captured arm | Launch A/B/A µs | Graph A/B/A µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16/8 | 1 | 1 | vector/1 | 82688/82560 | 2 | 13.912/13.871/13.941 | 16.292/16.313/16.372 |
| 16/8 | 4 | 1 | group2/4 | 399360/399360 | 2 | 11.794/11.792/11.751 | 14.262/14.260/14.295 |
| 16/8 | 8 | 1 | group2/8 | 1597440/1597440 | 2 | 14.667/14.616/14.620 | 16.389/16.388/16.391 |
| 16/8 | 16 | 1 | group2/16 | 3194880/3194880 | 2 | 18.459/18.466/18.395 | 23.999/23.237/20.582 |
| 16/8 | 33 | 1 | group2/32 | 6389760/6389760 | 2 | 29.788/29.844/29.777 | 32.736/32.713/32.726 |
| 32/16 | 1 | 1 | vector/1 | 99072/99072 | 2 | 26.389/26.411/26.499 | 29.155/29.212/29.122 |
| 32/16 | 4 | 1 | group2/4 | 399360/399360 | 2 | 28.379/28.288/28.106 | 31.381/31.363/31.317 |
| 32/16 | 16 | 1 | group2/16 | 3194880/3194880 | 2 | 49.770/49.728/52.217 | 55.375/52.970/54.916 |
| 32/16 | 33 | 1 | group2/32 | 6389760/6389760 | 2 | 76.300/81.498/76.162 | 83.657/78.017/83.762 |
| 16/8 | 1 | 2 | vector/1 | 82688/82560 | 4 | 39.860/39.836/39.763 | 47.707/41.696/41.630 |
| 16/8 | 1 | 4 | vector/1 | 82688/82560 | 8 | 193.323/199.180/202.978 | 195.387/198.758/193.412 |
| 32/16 | 4 | 4 | group2/4 | 399360/399360 | 8 | 334.166/340.046/345.982 | 334.428/339.948/334.508 |
| 16/8 | 1025 | 1 | group2/32 | 6390016/6390016 | 3 | 423.713/431.194/419.272 | 426.429/428.825/422.579 |

Solo launch results are close to their identical pinned arm. Captured and
multi-segment samples show variability, including slower selected samples;
these short screens establish no speedup or whole-engine parity. The four-
segment resident shape also scales poorly in both arms. A bounded repeat of
the differing shapes is recorded below; the model runner must separately
qualify any batching policy with actual caches and complete paid execution.

The initial representative 16-head screen gave solo launch A/B/A
13.837/13.825/13.797 µs and four-row 11.654/11.717/11.682 µs. Expanded controls
cover every added MMA query tile, both actual head counts, independent
segments and paid mask prepass. No dispatch choice is inferred from an
isolated kernel-only or uncharged scratch timing.


| Repeated heads/KV | Rows | Segments | Launch A/B/A µs | Graph A/B/A µs |
| --- | --- | --- | --- | --- |
| 32/16 | 4 | 1 | 24.145/24.052/24.024 | 26.563/26.579/27.163 |
| 16/8 | 1 | 4 | 186.191/191.714/186.416 | 191.641/186.451/191.127 |
| 32/16 | 4 | 4 | 359.267/368.222/367.702 | 359.976/359.967/364.372 |
| 16/8 | 1025 | 1 | 425.304/421.001/425.516 | 428.065/415.664/426.468 |

These repeats retain output equality and show that small captured differences
change across short samples. The measurements are complete primitive waves,
with sequential segment execution; they are not evidence to select four
segments as an optimized model policy.

## Verification limits

Focused Spark checks pass 23 CPU operand tests, 14 graph tests, nine
extended attention controls and two memory/registry controls including existing D64 vector, D128 MHA, D256 group8 and D512 global
references. The benchmark controls above pass output equality, scratch
funding, completion and captured replay. Local REUSE/header/boundary checks
and source-contract tests cover the added wrapper and audit inventory.
The source-contract set has three existing EXL3 closure controls skipped
because that prepared component is absent locally; no Spark GPU control is
skipped. No workstation GPU measurement, serving model, artifact inference,
full-model likelihood/greedy comparison or optimized-batching result is
claimed by this primitive study.
