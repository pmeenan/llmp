<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded quantized Gemma GeGLU writer

`VecQGlu::kGeGlu` supplies an explicit quantized gate/up output writer using
GGML's pinned GELU-tanh helper. Ordinary products plus split GeGLU remain the
model default and primitive fallback. This slice qualifies the writer at
actual Gemma widths; it does not qualify either checkpoint, a model dispatch,
whole-model quality/performance, optimized serving batches or paging.

The first one-row paid screen did not establish a writer gain. The warmed
screen and bounded 2/4-row overlap controls distinguish shared preparation
from writer fusion. No additional GeGLU/Q8 writer, Q4_K shared decode rewrite,
paired-MMQ prefill rewrite or universal selection is adopted.

## Numerical and representation contract

The writer calls `ggml_cuda_op_gelu_single(gate) * up` from locked
`ggml-cuda/unary.cuh`, retaining its F32 tanh approximation and build flags.
It uses existing VecQ dot products, reduction order and expert-pair scheduling.
The new enum requires a zero clamp limit; nonzero, infinite and NaN limits
are refused. Existing SwiGLU and clamp variants retain their behavior.

Expert controls use Q4_K fused gate/up `[2816,1408,128]` with two
`[2816,704,128]` views: gate first, up second. A row is 1,584 bytes; the up
view starts 1,115,136 bytes after gate. Both retain the complete expert stride
and owner. Q5_1 down is `[704,2816,128]`, with 528-byte rows.
The chain benchmark lays both members into one synthetic expert group,
funds canonical 512-value readable tails and zeroes every allocation.
Member starts/padding are 256 aligned; runtime slab pitch rounds stored group
bytes to `lcm(16,144,24)=144`, not universally 256. The measured pitch is
3,719,232 bytes, or 64 modulo 256. Separate direct controls use Q4_K and
Q5_1 pitches that are also not 256 aligned. No weight repacking is required.

Q8_0 shared gate/up controls use K2816/N2112. Routed and dense controls run
1/2/4 independent rows with `SetVecQOneToken`. Each joined output equals its
own selected solo output exactly. Direct captured preparation/writer replay
also equals the uncaptured result exactly. Finite near-zero and larger input
rows, duplicate/distinct routes, zero inactive columns, current views,
mismatched gates, output aliases, invalid enum/limits and excessive route
pairs have explicit controls. Refusal preserves sentinel outputs, draws zero
scratch and leaves the launch context usable; prefix/suffix output canaries
survive valid launches.

The fused writer is byte exact against native unfused products plus existing
GeGLU at these shapes. An independent FP64 tanh-GELU reference has NMSE at
most `1e-10` against actual downloaded gate/up products. Sampled actual-width
Q4_K products have independent nibble/scale/minimum decoding against uploaded
Q8_1 bytes; sampled Q5_1 down products independently decode nibbles/high bits,
Q8 stored sums and half-rounded affine products. Their scalar absolute-error
bound is `2e-5 * max(1, abs(reference))`. Existing all-format/variant controls
remain the separate quant-product evidence.

## Paid expert-chain comparison

`llmp_gemma_geglu_quant_bench ROWS OVERLAP REPEATS` constructs the actual
expert geometry with eight seeded random matrix patterns repeated across
128 physically distinct expert addresses. Inputs and route IDs are synthetic;
this is neither model weight execution nor observed model routing. Later rows
share exactly 0, 2 or 8 of their eight IDs with the first row. All eight IDs
within a row are distinct. Routes remain fixed within a timing block.

- **A:** ordinary pinned fused 1,408-row routed product, split GeGLU, ordinary
  Q5_1 down, expert scale, normalized positive route probabilities and ordered
  slot addition.
- **B0:** paid shared standard Q8_1 preparation, two native 704-row products,
  split GeGLU and the same down/scale/weight/add chain.
- **B1:** the same prepared input and one-token arithmetic with the GeGLU
  output writer, followed by the same down/scale/weight/add chain.

All arms pay input preparation, all selected experts and downstream work.
Expert scale is 0.75 and synthetic route probabilities are `(slot+1)/36`;
normalization/router construction, branch norms, shared FFN, residuals and
attention are outside this bounded chain. An independent FP64 scaled and
weighted ordered sum checks the downloaded down output, with NMSE `<=1e-12`.
DeepSeek routing/combination semantics are not used.

Eight warm launches precede each arm's event timing; captured replay has eight
warm replays. A/B0/B1/A bookends each contain 128 complete chains. Every
uncaptured and captured output/activation is checked. This small screen does
not estimate model-level throughput or establish a statistically significant
gain. Timings below are CUDA-event microseconds per complete chain.

| Rows | Shared IDs / 8 | Paid A bookends | Paid B0 | Paid B1 | Captured A bookends | Captured B0 | Captured B1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 134.415/141.562 | 145.076 | 136.568 | 138.020/143.163 | 135.556 | 140.190 |
| 2 | 0 | 328.002/322.083 | 307.399 | 309.355 | 323.830/318.085 | 306.864 | 307.137 |
| 2 | 2 | 281.219/287.077 | 274.484 | 281.922 | 280.885/277.752 | 266.943 | 273.353 |
| 2 | 8 | 135.299/134.975 | 141.618 | 139.983 | 138.145/132.125 | 131.669 | 139.012 |
| 4 | 0 | 636.620/630.544 | 597.601 | 606.831 | 639.284/631.812 | 590.898 | 603.760 |
| 4 | 2 | 525.067/516.497 | 508.350 | 504.181 | 519.746/514.044 | 494.257 | 501.537 |
| 4 | 8 | 167.651/167.452 | 197.990 | 194.105 | 164.120/164.144 | 191.120 | 194.316 |

All seven final chain shapes have original/native combined-output NMSE zero,
B0/B1 activation and combined-output byte equality, and captured replay
byte equality. The declared comparison bound against original is `5e-4`.
A result within the reference bookends is inconclusive; B0/B1 differences
separate preparation from the writer. The measured results do not support a
universal GeGLU writer default. One-row results changed across short screens;
the final table does not establish a repeatable solo gain. For multiple rows,
B0 preparation reuse often beats A, while adding B1 writer fusion does not
consistently improve B0. Full overlap retains the faster ordinary chain.

Scratch is fully paid: ordinary down peaks at 9,216 / 18,432 / 36,864 bytes
for 1/2/4 rows, and context peaks equal the plans. Product writer scratch is
zero; its input Q8 is separately owned at 3,456 bytes per row. Arm activation
storage, including the downstream chain and input Q8 where used, is A 416,832,
B0 420,288 and B1 375,232 bytes per row. The writer removes two 704-by-top8 F32
product buffers versus B0, saving 45,056 bytes per row. The benchmark retains
all three arms simultaneously: total cudaMalloc backing is 481,476,192 /
482,696,384 / 485,136,768 bytes for 1/2/4 rows. These are diagnostic allocation
charges, not VMM residency or a whole-engine memory qualification.

## Provenance and limits

Measured 2026-10-04 on `spark` GB10, driver 580.178.04, CUDA 13.4.92,
Clang 22.1.8, spark-native sm_121a, base 5198d2f. Same-format reference is
locked llama.cpp b10964/v0.4.1, commit
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, prepared GGML tree SHA-256
`1ae467d0fced412beb16faca3d0a910441477a06f1e2bae4ccbe0cdb6230bd11`.
The source lock and third-party payload are unchanged by this writer.

All 24 CPU extension-validator controls and all 19 fast-kernel GPU controls
passed without skips. After final setup ordering corrections, the new direct
GeGLU GPU control, 24 CPU controls and all seven chain screens passed again
(`gemma-geglu-final3`). Independent weighted-sum NMSE is at most
`3.9906744e-15`; every final observed scratch peak equals its plan.
Measured benchmark source SHA-256 is
`5d682ba06c3f1a714ba07389d1adc4c186155978997102e22e68724f2a6753cc`;
measured spark-native executable SHA-256 is
`2ae65b1f08658fb46efa4fb7be7c8cba6ddc1b7f297ed5ed49cd44f2fccbd9a8`.


At this task, TensorFold's default-branch HEAD was independently refreshed to
[`609ca419abecebdc5a059498a613680bd3aa847f`](https://github.com/ashhart/TensorFold/tree/609ca419abecebdc5a059498a613680bd3aa847f),
version [0.6.5](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/src/tensorfold/__init__.py),
on 2026-10-04. Its [declared Gemma coverage](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/README.md)
is MLX, not a CUDA GB10 comparator. No TensorFold performance is measured or
inferred here. Whole-model comparisons must refresh their references again.

Q4_K still uses generic per-column dot helpers; this factor does not add an
explicit shared Q4_K block load/decode. Routed GLU products retain P1 passes
and reload later pairs through cache. Dense non-GLU products can use their
existing P4 sibling; no new P4 writer policy is introduced. Standard Q8_1
output quantization still requires a separate preparation before down.
Actual checkpoint quality, model routing/selection, larger prefill chunks,
independent serving request lifetimes, VMM leases/capture and engine batching
remain later qualification. No additional format, EXL3 or model support is
claimed.
