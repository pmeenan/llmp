<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Decode templates: the sm_86 proof and a first speed comparison — 2026-10-08

The format layer's generic CUDA kernels (`src/kernels/decode`, D-108): a
block decoder per encoding and layout under shared vector, GLU, matrix,
grouped and rows templates. This records the first proof (Qwen3.8's
native NVFP4 and MXFP8 tensors on the discrete `sm_86` target) and one
matched speed comparison per template against the tuned GB10 kernel it
would someday replace. At that first comparison the templates were not wired
into any model; the comparison is a starting point for selection, not a
decision. The later bounded split-row integration is recorded below.

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

## Native split-row foundation — 2026-10-12

The shared `ModelRunner` now validates and binds every row-addressable part of
a native split NVFP4 table, requires identical lookup-order slot maps before
effects, and keeps codes and block scales compressed in bounded device slots.
The optional original tensor scale remains resident. A selected-operation
adapter calls the existing decoder under the ordinary ordered step program.
It accepts only the original paged weight and its direct checked side-row input.
The GGML GetRows capability now matches its actual supported formats and bounds,
so an unsupported native implementation cannot preempt this decoder.

Selected sources default to capture refusal. This allocation-free decoder
explicitly opts in; capture and replay retain the existing stream, launch
generation and owner checks and refuse a faulted selected runtime. No new
kernel arithmetic, widened resident representation or family-specific runner
is introduced. Exact row checks protect source decoding, independently of
another quantized engine's numerical recipe.

On Spark A, main's full native build passed in81 seconds, followed by117
focused host controls (four decoder descriptors,54 runner,12 execution-contract,
seven general lowering,11 Qwen and29 DeepSeek cases), with no failures/skips.
The two synthetic GPU cases passed in one supervised second: a32×9 split table
with four bounded row slots executes changed, repeated and reordered IDs through
eager, captured and replayed waves; all384 F32 values match an independent
code/scale oracle. Invalid IDs refuse before effects, and six incomplete or
inconsistent multipart configurations refuse before planning or reads.

The fixture admits a256MiB node domain and16MiB metadata allowance before
construction, charging8MiB for its fixed parser/compiler/result inventory.
Its tiny fixed state occupies one managed extent. The wrapper confines
process/cache memory to10GiB with zero swap and requires16GiB physical
availability before execution and6GiB afterward. All pre/post source, recipe
and binary guards passed; no compute PID remained and every max/OOM event was
zero. Observed cgroup peak was217,829,376 bytes, not total device occupancy.

Earlier failed attempts are preserved: a synthetic v1 spec missed its canonical
trailing newline; the mixed registry selected an overbroad GGML capability; and
a blanket selected-source capture refusal kept successful decoding eager.
Their fixes retain the strict artifact reader and complete output/replay checks.
One packaging attempt also failed before compilation because its source-map
script named the wrong file. These failures are not passing execution evidence.

This closes only the bounded row-operation foundation. Native ModelOpt target
admission, products, expert worklists, model quality and matched serving speed
remain open. It establishes no latency gain or new quality threshold. Raw
logs/XML remain outside Git under `/tmp/llmp-m36-coordination/`.

| Literal receipt | SHA-256 |
| --- | --- |
| `native-split-rows-main05/results/result.json` | `d8bf6a4fda0d9b9b0d411928c18e2d083d74d624bf6e3fdd37c799e1099d1046` |
| `native-split-rows-main-gpu04/results/result.json` | `988ca4a431320370dd162f71b70c603c8e891c70d4ed950eb88f7884e587e3d2` |
| `native-split-rows-main-gpu04/results/models.xml` | `9592b8aea7ee575b978a0de91d819f38fbe8f8fe3e6fe135eaa01d4655ab089a` |
| source05 independent capture challenge | `63c67ff808606a56d13d250ae9df7c338bbb30d0e7b26fe3a8f49f0b3715febb` |

## sm_86 proof on the v1 ModelOpt artifact (2026-10-16)

The Formats exit criterion, re-run on the workstation's discrete GPU
against the v1 artifact: `qwen38_decode_proof_test` (4 tests, labels
`gpu`, `models`, `gpu-discrete`) at `343d375e` (clean tree), `native`
preset `--locked`, `mise run test -- native --gpu`, restricted with
`-R Qwen38DecodeProof`, under `hostlock shared --label "llmp: sm_86
proof"`. GPU: RTX 3080 Ti, `sm_86`. Artifact: the v1 ModelOpt target
`d145900e058ff6470405c396affb9879b5c3c3770cc2b586d846e7e148c2058d`,
read through `LLMP_QWEN38_PROOF_ARTIFACT` from a partial copy (manifest,
index, spec and meta whole; every shard at full size, sparse, with its
header and only the byte ranges the test reads present; 1.5 GiB on disk,
in shards 00001, 00002, 00003 and 00019). All 4 pass, none skipped.
Errors are the largest observed over the checked outputs, as a fraction of
the test's per-element bound (1.0 is the tolerance):

| Product | Result | Largest error of its bound |
| --- | --- | --- |
| MXFP8 projections, layer 1 and 3 (13 tensors, vector 1 and 16 rows, matrix 512 rows) | pass | 0.124 (`ffn_down_shexp`, matrix 512); vector products at most 0.001 |
| Shared-expert gate and up as one vector GLU, 1 and 16 rows | pass | within its bound (the GLU asserts, it prints no ratio) |
| BF16 products (hyper-connection, n-gram key and value, router, shared gate, output head and its hyper-connection) | pass | 0.263 (`hc_attn_up`, K 320, matrix 512); head `output` 0.055 |
| NVFP4 routed experts, `scale_atoms` (grouped gate_up and down over 30 of 512 experts, 1, 16 and 512 tokens; one expert's vector GLU and down) | pass | grouped gate_up 0.063, grouped down 0.116, one expert's down (matrix 512) 0.104 |
| NVFP4 row-interleaved n-gram rows (3,072 rows, 1 to 3,072 ids) | pass | bit-exact against the reference decode |

Device memory: no product's run took device memory (0.0 MiB, bound 8 MiB),
and none exceeded its allocations (peak 250 MiB layers, 1,722 MiB head,
160 MiB experts). The test passed on first execution of the copy; the
earlier setup problems were environmental, not the test's: the copy needed
every shard's header (`Artifact::Open` checks them all), and this build
tree's discrete-GPU run needed `LD_LIBRARY_PATH=build/native/cublas`, since
the `native-gpu` test preset does not set it and some test binaries
(`expert_slab_test` and others) are linked without the cuBLAS run path, so
CTest's discovery failed without it. `nvidia-smi` reports a driver/library
mismatch (kernel module 595.91.07, user space 595.99.02); CUDA itself ran.
