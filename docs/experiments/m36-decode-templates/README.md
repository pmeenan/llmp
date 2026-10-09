<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Decode templates: the sm_86 proof and a first speed comparison — 2026-10-08

The format layer's generic CUDA kernels (`src/kernels/decode`, D-108): a
block decoder per encoding and layout under shared vector, GLU, matrix,
grouped and rows templates. This records the first proof (Qwen3.8's
native NVFP4 and MXFP8 tensors on the discrete `sm_86` target) and one
matched speed comparison per template against the tuned GB10 kernel it
would someday replace. The templates are not wired into any model; the
comparison is a starting point for selection, not a decision.

## The proof

`qwen38_decode_proof_test` reads one layer's worth of every product shape
from Qwen3.8's prepared artifacts through the v0 reader and
`BindQwen38`: layer 1's MXFP8 projections, shared expert and BF16
products (hyper-connections, n-gram key and value, router), layer 3's
MXFP8 QSA projections, 30 of layer 1's 512 routed experts (expert 17 i)
in the CUTLASS import (`c4fb47a9…`) and the GGML import (`67617f87…`),
3,072 rows of the n-gram table, and the output head. Every product runs
at 1 and 16 rows (vector, GLU) and 512 (matrix; grouped also at 1 and
16) on random N(0, 1) inputs and is checked against an F64 product of
the CPU reference decode, within (K + 4) · 2^-24 · Σ|w x| for F32 paths,
plus 2^-8 · Σ|w x| where mma.sync rounds inputs to BF16; the rows gather
bit-exactly, and the two imports' grouped products are equal bit for bit.
Device memory is read before each launch and after it completes, the
operands allocated: a run that kept memory, or a test holding more than
its allocations, fails (modules loaded eagerly first). That sees memory a
run keeps, not memory taken and freed before it completes; that no
product allocates at all is `decode_gpu_test`'s capture check, where each
template's run, captured in global mode (which refuses `cudaMalloc`), is
one kernel node.

| Host | GPU | Result |
| --- | --- | --- |
| workstation, SDK `x86_64-2c4eb5b2f68f3fa5`, `native` preset (targets only) | RTX 3080 Ti, `sm_86` | 4 of 4 pass (again after the review's fixes); largest error 0.263 of its bound (a BF16 K 320 product at 512 rows); no product took device memory (0.0 MiB), peak within the allocations |
| `spark-b` (`spark-56f5`), SDK `aarch64-2c4eb5b2f68f3fa5`, `spark-native` | GB10, `sm_121` | 4 of 4 pass on the full M3 store (jobs `m36dec-final2` and, after the review's fixes, `m36rev-test2`), the two imports bit-identical; the memory check is not judged on an integrated GPU, whose free memory is the host's (an earlier run, `m36dec-proof1`, failed only that check) |

The workstation read a partial copy of each artifact: manifest, index
and metadata whole, each shard at full size with only its header and the
groups and table rows the test reads present (sparse files, 1.6 GiB and
249 MiB on disk). `decode_gpu_test` (every decoder under every template
at rows 1 to 2048, row invariance, weights of several matrices, a grouped
product's empty, single and mismatched experts and its equality with the
matrix template, edge codes, graph capture, refusals, declarations)
passes on both GPUs: largest error 0.582 of its bound (MXFP8 split,
mma.sync).

## Speed against the tuned kernels

`llmp_decode_templates` on `spark-b`, base `f5eed53e` plus this slice,
`spark-native` build, job `m36dec-bench2`: each candidate one CUDA graph
of 64 launches, 7 replays alternating ABBA with its counterpart on one
stream, the median replay per launch (in every cell the fastest replay is
within 4% of the median; a few slowest replays are up to 40% above it). Weights are constant bytes at the model's sizes;
decode-shaped products rotate over weight copies (4 of qkv, 16 of the
shared expert) or 8 routings so that successive launches do not hit L2.
The grouped template reads a worklist built on the host; the tuned
prefill path's routing is timed alone beside it.

| Template | Shape | Rows | Tuned kernel | Tuned µs | Template µs | Ratio |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| vector | MXFP8 K 2560, N 10240 | 1 | `llmp.mxfp8.mul_mat_vec` | 113.7 | 113.5 | 1.00 |
| vector | MXFP8 K 2560, N 10240 | 16 | `llmp.mxfp8.mul_mat_vec` | 154.8 | 514.6 | 3.33 |
| vector_glu | MXFP8 K 2560, N 640 gate and up | 1 | two `mul_mat_vec` (no SwiGLU) | 15.5 | 17.7 | 1.14 |
| vector_glu | MXFP8 K 2560, N 640 gate and up | 16 | two `mul_mat_vec` (no SwiGLU) | 41.9 | 78.8 | 1.88 |
| matrix | MXFP8 K 2560, N 10240 | 512 | `llmp.mxfp8.quantize` + `.swizzle` + `.gemm` (CUTLASS FP8) | 305.1 (swizzle 13.9) | 1380.9 | 4.53 |
| matrix_fma | the same | 512 | (none) | | 3181.9 | |
| grouped | NVFP4 atoms, K 2560, N 1280, 10 of 512 experts | 1 | `llmp.moe.gemv` | 81.8 | 181.3 | 2.22 |
| grouped | the same | 16 | `llmp.moe.gemv` | 1156.3 | 1967.2 | 1.70 |
| grouped | the same | 512 | `llmp.moe.route` + `.quantize` + `.gemm` (CUTLASS FP4) | 4554.0 (route 92.8) | 7850.8 | 1.76 without the route |
| rows | NVFP4 row-interleaved, 160 values, 16 ids a token | 1 | `llmp.nvfp4.get_rows` | 0.90 | 1.19 | 1.32 |
| rows | the same | 16 | `llmp.nvfp4.get_rows` | 1.03 | 1.57 | 1.52 |
| rows | the same | 512 | `llmp.nvfp4.get_rows` | 33.8 | 19.0 | 0.56 |

The arithmetic differs: the tuned prefill products quantize their inputs
(to MXFP8, or to NVFP4 for the experts) and run on FP8 or FP4 tensor
cores; the templates keep inputs in F32 (vector) or round them to BF16
(mma.sync), which is more accurate and slower. A first-run variant
without weight rotation found the decode-shaped products partly in L2
(vector rows 1: 65 to 67 µs both); the table supersedes it.

Not measured, so only leads: the vector template at 16 rows re-reads
each input segment from L1 once per output row, with lanes 64 bytes
apart; the matrix and grouped tiles are single-stage (no `cp.async`
pipeline, no `ldmatrix`) at 64 by 64 or 16 by 64; the grouped template
launches a block per expert and 64 outputs whether or not the expert has
rows (10,040 of its 10,240 blocks are empty at one token). None of this
was tuned.

Raw logs stay on `spark-b` and the workstation's session scratch, not in
Git.
