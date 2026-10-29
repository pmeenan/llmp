<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# llmp.vecq against GGML's MMVQ at one to eight columns (GB10)

The review of M3.5 (C5) found that on the GB10 a dense `llmp.vecq` product
of five to eight columns ran two passes of four columns, re-reading and
re-decoding its weights, and that only Q6_K and Q8_0 were sent to GGML's
MMVQ there, by a capability veto (`PreferOrdinaryMmvq`). This screen
measured every encoding llmp.vecq computes against the pinned MMVQ at one
to eight columns, and a one-pass eight-column form, before choosing.

## Method

`llmp_vecq_bench --sweep 1 --launches 128` ([vecq_bench.cc](../../../benchmarks/vecq_bench.cc),
[vecq_upstream.cu](../../../benchmarks/vecq_upstream.cu)), twice, on `spark`
(GB10, driver 580.178.04, CUDA 13.0, SDK `aarch64-8486b6bf58a61039`),
2026-10-28, as exclusive jobs. Random weights in sixteen copies (each launch
reads another, so L2 holds none of it) and random Q8_1 activations, each
column's row padded to 512 values; the faster of the two runs is kept.
Arms: llmp.vecq's default schedule; GGML's MMVQ through its pinned
dispatcher (`ggml_cuda_op_mul_mat_vec_q_dispatch`) over the same prepared
Q8_1 rows, with the immutable-weights prefetch `llmp.mmvq.prepared` uses;
for the gated product, llmp.vecq's fused GLU against two MMVQ products (the
decomposition's GLU kernel, about 2 to 3 µs at these widths, not counted).
Shapes K x N 4096 x 4096 and 2560 x 10240.

## Results

MMVQ's time over llmp.vecq's (below 1: MMVQ faster), at 1 / 2 / 4 / 6 / 8
columns; for the gated rows, two MMVQ products over the fused GLU:

| Encoding | Dense, 4096 x 4096: c1 / c2 / c4 / c6 / c8 | Dense, 2560 x 10240 | Gated, 4096 x 4096 | Gated, 2560 x 10240 |
| --- | --- | --- | --- | --- |
| Q4_0 | 0.97 / 0.98 / 0.89 / 0.75 / 0.67 | 0.91 / 0.86 / 0.81 / 0.87 / 0.70 | 1.94 / 1.75 / 1.27 / 0.90 / 0.75 | 1.11 / 0.89 / 0.81 / 0.73 / 0.58 |
| Q4_1 | 1.04 / 0.99 / 0.91 / 0.84 / 0.73 | 0.98 / 0.95 / 0.83 / 0.87 / 0.73 | 1.94 / 1.69 / 1.25 / 0.97 / 0.78 | 1.02 / 0.93 / 0.79 / 0.79 / 0.63 |
| Q5_0 | 0.97 / 1.00 / 0.98 / 0.76 / 0.67 | 0.96 / 0.91 / 0.84 / 0.75 / 0.63 | 1.51 / 1.51 / 1.27 / 0.85 / 0.73 | 1.01 / 0.89 / 0.75 / 0.73 / 0.61 |
| Q5_1 | 0.98 / 1.02 / 0.97 / 0.84 / 0.72 | 0.99 / 0.95 / 0.81 / 0.82 / 0.69 | 1.30 / 1.36 / 1.21 / 0.93 / 0.77 | 1.00 / 0.89 / 0.77 / 0.75 / 0.65 |
| Q8_0 | 1.02 / 0.97 / 0.92 / 0.77 / 0.70 | 0.98 / 1.00 / 0.98 / 0.82 / 0.74 | 1.02 / 0.94 / 0.86 / 0.86 / 0.75 | 0.97 / 1.00 / 0.93 / 0.89 / 0.77 |
| Q2_K | 0.93 / 0.76 / 0.65 / 0.56 / 0.52 | 0.99 / 0.67 / 0.60 / 0.58 / 0.58 | 1.65 / 0.87 / 0.67 / 0.58 / 0.55 | 1.42 / 0.69 / 0.62 / 0.57 / 0.57 |
| Q3_K | 0.93 / 0.79 / 0.56 / 0.42 / 0.39 | 0.95 / 0.62 / 0.50 / 0.42 / 0.41 | 1.62 / 1.02 / 0.64 / 0.45 / 0.41 | 1.40 / 0.75 / 0.54 / 0.43 / 0.42 |
| Q4_K | 0.81 / 0.96 / 0.83 / 0.57 / 0.52 | 0.80 / 0.83 / 0.71 / 0.57 / 0.55 | 1.59 / 1.41 / 0.92 / 0.69 / 0.57 | 0.94 / 0.83 / 0.64 / 0.57 / 0.52 |
| Q5_K | 0.97 / 0.97 / 0.83 / 0.61 / 0.54 | 0.96 / 0.78 / 0.71 / 0.66 / 0.57 | 1.49 / 1.33 / 0.94 / 0.68 / 0.59 | 0.88 / 0.73 / 0.62 / 0.60 / 0.52 |
| Q6_K | 0.86 / 0.89 / 0.79 / 0.60 / 0.50 | 0.91 / 0.85 / 0.75 / 0.60 / 0.49 | 1.13 / 0.98 / 0.78 / 0.61 / 0.49 | 0.96 / 0.79 / 0.66 / 0.57 / 0.49 |
| IQ2_XXS | 0.94 / 0.73 / 0.55 / 0.40 / 0.39 | 0.97 / 0.60 / 0.55 / 0.38 / 0.38 | 1.43 / 0.76 / 0.47 / 0.34 / 0.31 | 1.37 / 0.65 / 0.53 / 0.36 / 0.34 |
| IQ2_XS | 0.91 / 0.84 / 0.71 / 0.60 / 0.62 | 1.01 / 0.71 / 0.78 / 0.67 / 0.67 | 1.41 / 1.00 / 0.78 / 0.62 / 0.64 | 1.41 / 0.73 / 0.82 / 0.65 / 0.65 |
| IQ2_S | 0.96 / 0.72 / 0.48 / 0.39 / 0.36 | 0.97 / 0.54 / 0.46 / 0.38 / 0.36 | 1.52 / 0.83 / 0.51 / 0.42 / 0.38 | 1.49 / 0.66 / 0.50 / 0.40 / 0.37 |
| IQ3_XXS | 0.95 / 0.99 / 0.82 / 0.65 / 0.62 | 0.97 / 0.86 / 0.77 / 0.68 / 0.67 | 1.78 / 1.10 / 0.77 / 0.60 / 0.58 | 1.51 / 0.78 / 0.65 / 0.58 / 0.55 |
| IQ3_S | 0.99 / 0.94 / 0.56 / 0.47 / 0.38 | 0.98 / 0.69 / 0.48 / 0.38 / 0.33 | 1.85 / 1.03 / 0.61 / 0.49 / 0.39 | 1.40 / 0.79 / 0.51 / 0.39 / 0.34 |
| IQ4_NL | 0.98 / 0.90 / 0.86 / 0.72 / 0.64 | 1.00 / 0.84 / 0.75 / 0.80 / 0.68 | 1.79 / 1.65 / 1.22 / 0.86 / 0.72 | 1.08 / 0.90 / 0.77 / 0.66 / 0.55 |
| IQ4_XS | 0.90 / 0.83 / 0.72 / 0.66 / 0.53 | 0.98 / 0.91 / 0.78 / 0.53 / 0.46 | 1.83 / 1.24 / 0.97 / 0.70 / 0.55 | 1.06 / 0.83 / 0.73 / 0.52 / 0.46 |
| MXFP4 | 1.01 / 0.99 / 0.92 / 0.65 / 0.59 | 1.04 / 0.82 / 0.78 / 0.72 / 0.61 | 1.87 / 1.54 / 1.17 / 0.80 / 0.71 | 1.19 / 0.86 / 0.76 / 0.67 / 0.56 |

- **MMVQ is faster at two to eight columns for nearly every encoding**, by
  up to 2.8 times (the IQ2/IQ3 types and Q3_K at eight columns), and at one
  column llmp.vecq (whose GB10 schedule mirrors MMVQ's) is within a few
  percent except Q4_K (MMVQ 20% faster on both shapes).
- **The fused GLU** wins at one column and loses from four to seven
  columns on, depending on the encoding.
- **One pass of eight columns does not help** (`r2 w2 p8` / `r1 w2 p8`
  against the two-pass `r2 w2 p4` / `r1 w2 p4`, five to eight columns):
  dense, time ratio median 1.04 min 0.82 max 1.33; gated median 1.19 min 0.92 max 1.92.
  The second pass's weights come from L2, and the eight-column form's
  registers cost more than the re-decode; the forms were not kept.

## Adoption

- The veto is gone: `AcceptsVecQ` says only what the kernel computes.
- [measured_ranks.h](../../../src/kernels/ggml/measured_ranks.h) ranks
  `ggml.mul_mat.mmvq` and `llmp.mmvq.prepared` first for the GB10's dense
  products of each encoding `llmp.mmvq.prepared` takes, from the first
  column count at which MMVQ was more than 10% faster on both shapes through
  eight: Q6_K from 2; Q4_K from 3; Q4_0, Q5_K and IQ4_NL from 4; Q5_0,
  Q5_1 and Q8_0 from 5; Q4_1 from 6. Q4_K's single column (MMVQ 20% faster
  alone) was ranked in the first candidate and dropped: Gemma 4 31B's C1
  decode measured +0.80% slower with it (below, sd 1.39%), −0.05% without.
- Its fusion rows make llmp.vecq's fused GLU no candidate where its
  decomposition measured more than 5% faster on both shapes (below, "The
  fused GLU by measurement"): Q6_K from 4; Q8_0, Q4_K and Q5_K from 5;
  Q4_0, Q5_0, IQ4_NL and IQ3_XXS from 6; Q4_1, Q5_1 and IQ2_XXS from 7.
- Encodings `llmp.mmvq.prepared` does not take (IQ2_XXS, IQ2_XS, IQ2_S,
  IQ3_XXS, IQ3_S, Q2_K, Q3_K, IQ4_XS, MXFP4) keep llmp.vecq: their products
  share one Q8_1 preparation (`vecq_inputs`), which MMVQ over the F32 input
  would lose. Extending `llmp.mmvq.prepared` to them is the follow-up the
  table above argues for.
- llmp.vecq's GB10 L2 prefetch list now matches the pinned MMVQ's (Q4_0,
  Q5_0, Q3_K, IQ4_NL and IQ4_XS added; IQ1_M has no llmp.vecq kernel), and
  its prefetches compile only for the GB10, its pre-wait prefetch only where
  programmatic dependent launch exists, as upstream's; its kernel versions
  are 2.

## The fused GLU by measurement (2026-10-28)

The first rows took the fused GLU out only where two MMVQ products were
more than 15% faster on both shapes, which left Q8_0 fused at five and six
columns although the two products measured 11% to 14% faster there: the
five-row verify the old veto was added for. The rows were remade by one
rule for every encoding and column count: the faster wherever the gap is
beyond the sweep's repeat noise, at 5% (`execution::ClassifyFusionLatency`'s
band, narrowed from D-085's coarse 10% for this).

- **Data.** `llmp_vecq_bench --sweep 1 --launches 256`, eight runs as one
  exclusive job on `spark` (GB10, driver 580.178.04), each arm's fastest
  run. The sweep now also times GGML's SwiGLU of the two products
  (`sweepswiglu`, `UpstreamSwiGlu` in vecq_upstream.cu): 2.0 µs at 4096 x
  4096 and 1.8 to 2.0 µs at 2560 x 10240, at every column count.
- **The decomposition as the planner plans it.** Two products, each by
  the dense rank rows (GGML's MMVQ in its band above, llmp.vecq
  otherwise), plus the SwiGLU. Encodings without MMVQ rows (the IQ2/IQ3
  types, Q2_K, Q3_K, IQ4_XS, MXFP4) are compared with two llmp.vecq
  products, which is what their decomposition runs, not with MMVQ.
- **Noise.** The ratio of the decomposition's time to the fused GLU's,
  from the fastest of four runs, differed between the two halves of the
  eight by a median 1.0%, 3.2% at the 90th percentile, 4.3% at the 95th
  and 9.3% at most; against the eight runs of the build before the
  SwiGLU timing (with this run's SwiGLU), 1.1% median, 3.9% at the 95th.
  A 5% gap on both shapes is therefore beyond the repeat spread at the
  95th percentile.
- **Rows.** A band holds the column counts at which the decomposition was
  more than 5% faster on both shapes (ratio below 1 / 1.05), through 8;
  its recorded times are the band's narrowest gap (IQ2_XXS's at 2560 x
  10240, the rest at 4096 x 4096). Everything else keeps the fusion: it was
  faster or within 5% on at least one shape.

The decomposition's time over the fused GLU's (below 1: the decomposition
faster; bold: separate):

| Encoding | 4096 x 4096: c1 … c8 | 2560 x 10240: c1 … c8 | Separate at |
| --- | --- | --- | --- |
| Q4_0 | 1.80 / 1.74 / 1.52 / 1.30 / 1.02 / **0.95** / **0.85** / **0.78** | 1.27 / 1.11 / 1.09 / 0.85 / 0.82 / **0.76** / **0.68** / **0.63** | 6 to 8 |
| Q4_1 | 1.61 / 1.65 / 1.48 / 1.33 / 1.25 / 1.02 / **0.91** / **0.82** | 1.10 / 1.01 / 1.01 / 1.04 / 0.93 / 0.83 / **0.76** / **0.69** | 7 to 8 |
| Q5_0 | 1.45 / 1.50 / 1.47 / 1.31 / 1.00 / **0.91** / **0.87** / **0.77** | 1.06 / 1.00 / 0.98 / 0.97 / 0.90 / **0.80** / **0.73** / **0.67** | 6 to 8 |
| Q5_1 | 1.35 / 1.34 / 1.31 / 1.35 / 1.10 / 0.96 / **0.89** / **0.82** | 1.04 / 1.00 / 0.92 / 0.92 / 0.91 / 0.83 / **0.76** / **0.71** | 7 to 8 |
| Q8_0 | 1.04 / 1.03 / 1.02 / 0.99 / **0.95** / **0.89** / **0.84** / **0.79** | 1.02 / 1.02 / 1.02 / 0.98 / **0.92** / **0.88** / **0.83** / **0.79** | 5 to 8 |
| Q2_K | 1.81 / 1.18 / 1.09 / 1.05 / 1.04 / 1.06 / 1.04 / 1.04 | 1.57 / 1.11 / 1.02 / 1.00 / 0.98 / 1.02 / 1.00 / 1.01 | never |
| Q3_K | 1.84 / 1.38 / 1.28 / 1.14 / 1.08 / 1.06 / 1.05 / 1.05 | 1.37 / 1.24 / 1.14 / 1.11 / 1.06 / 1.05 / 1.05 / 1.03 | never |
| Q4_K | 1.79 / 1.49 / 1.18 / 1.01 / **0.80** / **0.73** / **0.66** / **0.61** | 1.31 / 1.02 / 0.76 / 0.67 / **0.69** / **0.62** / **0.58** / **0.54** | 5 to 8 |
| Q5_K | 1.45 / 1.41 / 1.27 / 1.07 / **0.81** / **0.73** / **0.66** / **0.61** | 1.04 / 0.95 / 0.95 / 0.66 / **0.68** / **0.63** / **0.59** / **0.55** | 5 to 8 |
| Q6_K | 1.23 / 1.08 / 0.98 / **0.88** / **0.71** / **0.64** / **0.59** / **0.55** | 1.04 / 0.85 / 0.79 / **0.73** / **0.66** / **0.60** / **0.57** / **0.52** | 4 to 8 |
| IQ2_XXS | 1.68 / 1.14 / 1.00 / 0.89 / 0.85 / 0.86 / **0.84** / **0.81** | 1.46 / 1.11 / 1.05 / 1.02 / 0.98 / 0.96 / **0.94** / **0.92** | 7 to 8 |
| IQ2_XS | 1.68 / 1.25 / 1.16 / 1.11 / 1.06 / 1.05 / 1.04 / 1.06 | 1.38 / 1.11 / 1.04 / 1.01 / 1.02 / 1.01 / 1.01 / 1.00 | never |
| IQ2_S | 1.69 / 1.19 / 1.12 / 1.09 / 1.05 / 1.05 / 1.05 / 1.03 | 1.52 / 1.18 / 1.12 / 1.08 / 1.05 / 1.04 / 1.04 / 1.04 | never |
| IQ3_XXS | 1.89 / 1.17 / 1.05 / 0.97 / 0.98 / **0.95** / **0.95** / **0.95** | 1.54 / 0.94 / 0.87 / 0.84 / 0.88 / **0.87** / **0.88** / **0.86** | 6 to 8 |
| IQ3_S | 1.89 / 1.19 / 1.12 / 1.08 / 1.07 / 1.04 / 1.02 / 1.03 | 1.39 / 1.18 / 1.08 / 1.07 / 1.03 / 1.03 / 1.03 / 1.03 | never |
| IQ4_NL | 1.75 / 1.69 / 1.51 / 1.22 / 0.96 / **0.90** / **0.82** / **0.75** | 1.19 / 1.15 / 1.14 / 0.86 / 0.78 / **0.73** / **0.68** / **0.63** | 6 to 8 |
| IQ4_XS | 1.84 / 1.57 / 1.45 / 1.35 / 1.14 / 1.12 / 1.04 / 1.01 | 1.22 / 1.02 / 0.99 / 0.96 / 0.97 / 0.99 / 0.98 / 0.99 | never |
| MXFP4 | 1.78 / 1.62 / 1.44 / 1.30 / 1.27 / 1.25 / 1.21 / 1.19 | 1.20 / 1.09 / 1.05 / 1.02 / 0.94 / 0.95 / 0.97 / 0.98 | never |

- The narrowest adopted gaps are Q8_0 at five columns (5.3% and 9.2%),
  IQ3_XXS at six to eight (5.5% to 5.7% at 4096 x 4096), Q4_0 at six
  (5.7%) and IQ2_XXS at seven (5.9% at 2560 x 10240).
- Mixed cells keep the fusion: Q5_K and Q4_K at four columns are 0.66 to
  0.67 at 2560 x 10240 but 1.01 to 1.07 at 4096 x 4096; MXFP4 and IQ4_XS
  separate faster only at 2560 x 10240.
- End to end, on the tree with these rows and the F13 follow-up, against
  main at `4a86b43f` (frozen copies, two-block ABBA on `spark`,
  2026-10-28): `qwen38-gguf` C1 prefill / decode / wall −1.10% / −0.19% /
  −0.77%, C2 −0.88% / −2.47% / −1.31%, C4 −0.61% / −4.30% / −1.33%;
  `quality` equal on both arms (NLL 2.8432114726, scores `d7491cec…`,
  replies equal). Its timed waves are at most four columns, so the rows
  that changed here (five columns and up) act on products of five to eight
  rows elsewhere (speculative verify, short prompt tails), not on its timed
  decode.

## End-to-end

Tier-1 two-block ABBA confirmations (`tools/speed_gate.py plan --blocks 2`,
[m36-baselines](../m36-baselines/README.md)) on `spark`, 2026-10-28, the old
arm main at `4a86b43f`, the new arm this change, both from frozen copies of
their builds, the `u2-v1-artifacts` store. Mean slowdown over the two blocks
(negative: the new arm faster); `quality` on both arms.

| Family | C1 prefill / decode / wall | C2 (gemma26/31: C4) prefill / decode / wall | C4 prefill / decode / wall | Quality (both arms) |
| --- | --- | --- | --- | --- |
| gemma2 | −0.52% / −0.12% / −0.29% | −0.75% / −0.40% / −0.58% | | NLL 3.5649091350, scores `a8b38e0b…`, replies equal |
| gemma3 | −0.64% / −0.05% / −0.22% | −1.31% / −0.05% / −0.58% | | NLL 3.6733320239, scores `2566efde…`, replies equal |
| gemma26 | +0.09% / −0.12% / −0.01% | −0.14% / −0.25% / −0.33% | | NLL 9.9205757226, scores `349987df…`, replies equal |
| gemma31 | +0.17% / −0.05% / +0.05% | −0.08% / **−3.98%** / −0.63% | | NLL 9.7004038387, scores `7904d829…`, replies equal |
| qwen38-gguf | −0.95% / +0.14% / −0.56% | −0.73% / **−2.08%** / −1.09% | −0.26% / **−3.49%** / −0.89% | NLL 2.8432114726, scores `d7491cec…`, replies equal |
| deepseek | −0.12% / −1.13% / −0.42% | −0.00% / −0.76% / −0.58% | −0.21% / **−2.84%** / −0.75% | NLL 2.7937082602, scores `7b96879c…`, replies equal |

- gemma26 and gemma31 are the final change's runs. gemma2, gemma3,
  qwen38-gguf and deepseek ran the first candidate, which also ranked MMVQ
  for Q4_K's single-row products (the final change selects as the old arm
  there); quality was also rerun on the final change for gemma26 and
  gemma31, equal.
- Every metric is within the tier-1 rule (mean slowdown at most +0.5%); no
  family needs the full gate. Peak memory moved by at most +0.06 GiB.
- Scores and replies are bit-identical on every family: the GB10's MMVQ and
  llmp.vecq share the reduction geometry at these column counts, and the
  multi-row decodes the gates time are not in the quality runs' greedy C1
  replies.
- The same runs' plan builds (`planned in`, each run's held plans at
  teardown) are 17% to 36% faster with the planner changes of this slice
  (memoized fused inference, no re-inference at Finish, lazy digests, the
  registry indexed by operation): per plan, gemma2 14.1 to 11.7 ms, gemma26
  28.0 to 22.0, gemma31 34.7 to 26.7, qwen38-gguf 99.4 to 64.0, deepseek 148
  to 125.
