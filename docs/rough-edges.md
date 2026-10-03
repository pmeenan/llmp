<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Rough edges — findings log

CUDA driver, DGX Spark platform, toolchain, and library bugs, quirks,
surprising limits, performance cliffs, and missing capabilities encountered
while building jitLLM. Log the ones that burned real debugging time and will
bite again — this is a save-future-you log, not a compliance artifact.

**Before adding:** grep for the API/library involved to avoid duplicates.
**Before debugging weirdness:** check here first — it may be known.

A good entry says what environment it happened in (workstation or Spark,
driver, toolkit, compiler versions) and what was observed vs. expected;
include a reproduction when it's cheap to capture. Platform constraints that
were known from documentation before any code existed (Spark's
compatibility-mode-only GDS, no GPUDirect RDMA) are recorded in
[environment.md](environment.md) and D-004/D-034, not here; this log is for
what the documentation did not tell us.

Format:

```
## RE-NNN: Title  (YYYY-MM-DD, status: open | fixed-upstream | worked-around | wontfix)
Environment / Repro or measurement / Observed / Expected / Impact / Links
```

Newest first. RE-numbers are never reused.

## RE-043: `rsync -a` of an older source over a build tree leaves stale objects  (2026-10-03, status: worked-around)

Environment: a Spark build tree (`ninja`, preset `spark-native`) fed by
`rsync -a` from more than one workstation worktree. `rsync -a` keeps each
source's modification time. A file copied from a worktree whose copy is
older than the one built last goes back in time, and `ninja` rebuilds only
an output older than its inputs. So the old object stays, and the build
links code the tree no longer holds, with no warning. Observed: a binary
built from one branch carried another branch's scheduler. It confounded
an HTTP comparison (`ds4-output-prefix`, "DSpark with output-A/HCA") and
its explanation, and the binary's checks ran over mixed objects.
Work-around: copy with checksums and fresh times (`rsync -rlpc`, no `-t`),
or `touch` what was copied, and compare a rebuilt binary's hash against
an expected one before measuring.

## RE-042: Clang's CUDA parser requires a cuRAND header absent from the trimmed SDK  (2026-10-02, status: worked-around)

On Spark A, SDK `aarch64-e0a0c85c42806fb1`, Clang22.1.8 and CUDA13.4.92,
the Q-head CUDA clang-tidy check first rejected NVCC-only compile-database
driver flags. Adapting that actual entry for Clang's CUDA host parser then
failed at `__clang_cuda_runtime_wrapper.h:505`: its unconditional
`curand_mtgp32_kernel.h` include is absent from the trimmed SDK, although
the checked kernel uses no cuRAND API.

For this lint check, retain the actual entry's definitions and SDK include
paths, replace NVCC-only driver flags with Clang CUDA host parsing flags,
then supplement headers with `/usr/local/cuda-13.0/include`, after the SDK
paths. The resulting changed-unit check passes. This workaround affects
parsing only: production still compiles with NVCC13.4.92, unchanged native
flags and the pinned SDK libraries. Preserve the original and adapted
entries; do not replace the production toolchain to repair lint.
See the [Q-head report](experiments/dsv4-qhead/README.md) and
[LLVM note](upstream/other.md#llvm-cuda-lint-parsing-needs-an-unused-curand-header-re-042).

## RE-041: CUDA current free memory can exclude reclaimable inactive GGUF file cache  (2026-10-01, status: worked-around)

Environment: `spark-b` (`spark-56f5`), GB10, kernel `7.0.0-1019-nvidia`,
driver 580.178.04, CUDA 13.4.92 in SDK `aarch64-e0a0c85c42806fb1`.
The matched 32K native ds4 reference refused its full-budget-plus-6-GiB
guard after preparation, before weight loading or inference. Linux
MemAvailable was about 114 GiB, but allocation-only diagnostics observed
CUDA current free memory about 87.11 GB below Linux MemAvailable. The
8K-to-32K budget increase was only 248 MiB.

The inactive original 86,720,111,488-byte GGUF had been fully hashed using
buffered reads. A pinned-FD `mincore` control, sampled after the first
allocation probe and immediately before advice, found all 21,171,903 pages
resident. File-scoped `POSIX_FADV_DONTNEED` reduced residency to zero.
The same 32K allocation probe then observed CUDA free memory after setup
increase from 23,445,520,384 to 110,266,376,192 bytes, while Linux
MemAvailable remained approximately 120.32 GB. Its unchanged 6-GiB guard
changed from refusal to admission. Restoring the original qualified source
and executable then completed the full 32K model with byte-exact logits.

The workaround belongs to this benchmark handoff: advise only the inactive,
authenticated original GGUF after its buffered verification, and retain the
normal native admission guard. No global cache flush, provider accounting
change or guard reduction was made. The diagnostic omits aligned-weight
virtual reservations/catalog entries and does not itself qualify a model
run. These observations establish a cache-sensitive current-free reading;
they do not establish whether a real CUDA allocation would automatically
reclaim that cache, nor identify a driver defect.

See the [matched 32K report](experiments/ds4-matched-32k/README.md#allocation-refusal-and-scoped-cache-control)
and [NVIDIA handoff](upstream/cuda.md#cuda-current-free-memory-excludes-inactive-file-cache-re-041).
Failed run, unchanged allocation probes and scoped cache control remain at
`spark-b:~/scratch/m3-ds4-matched-32k-{native-r1,budget-r1}/`.

On 2026-10-02, Spark A's private query-B control hit the same guard after
preparation, before loading weights or running a numerical sample. An
allocation-only probe with its unchanged paid buffers needed
100,797,513,728 bytes free (94,355,062,784 future weight/state bytes plus
6 GiB), but CUDA reported 97,618,493,440. Advice on only the inactive,
previously hashed 28,800,138,240-byte Mia packed PLE raised the same
probe's CUDA free to 116,689,006,592, admitting the unchanged guard.
Linux MemAvailable changed from 119,751,155,712 to 120,541,548,544 bytes.
The pinned-FD `mincore` vectors stayed all ones for this root-owned 0644
file under the unprivileged reader; those vectors do not substantiate an
evicted-page count. No global cache drop, guard reduction or operand removal
was used. Raw probes, advice and the failed preparation are retained at
`spark:~/scratch/m3-ds4-query-b-factor-r1/`.

## RE-040: FlashInfer accepts a Python profile override but its SM120 wrapper still autotunes both products  (2026-09-30, status: worked-around)

Environment: Spark GB10, FlashInfer0.6.17/a0a6b019 in Mia's pinned image
fc120ece…. `fused_moe/core.py` documents `profile_ids` as absolute product
tactic overrides, but the SM120 wrapper accepts that argument and then
unconditionally runs `AutoTuner.choose_one` for both products. A Python
call with a traced pair therefore does not prove that pair executed.
The source-visible native binding validates and applies absolute IDs;
`[-1,-1]` instead selects its first entries, without the Python autotuner.

The external complete-consumer comparison uses that actual native binding
with source/trace-proved19/56 and ordered19/36, pins all original sources,
and tests every captured graph against eager controls before timing.
This bypass belongs to an operator diagnostic, not the serving runtime.
Its neutral/slower measured result justifies no implementation import.
See [the complete report](experiments/qwen38-fi-down-stage/README.md#complete-consumer-follow-up)
and [the standalone upstream handoff](upstream/flashinfer.md#pinned-python-profile-override-is-ignored-re-040).

## RE-039: A bounded Nsight CLI capture ended before its profiled model, leaving the target outside the job's supervision  (2026-09-30, status: worked-around)

- **Environment:** `spark`, GB10, driver 580.178.04, CUDA 13.4.92,
  Nsight Systems 2025.3.2. Job `m3-final-ds-max-swap` starts at
  10:10:15 EDT, launching `jitllm_long_swap` through
  `nsys profile --trace=cuda --sample=none --cpuctxsw=none --delay=2700
  --duration=60 --kill=none --wait=all --export=sqlite`.
- **Observed:** after the bounded capture/export, the profiler exits zero
  and the supervised shell ends, but the model remains a GPU compute
  process with PPID 1 and its own process group. Its output descriptors
  are anonymous pipes. The redirected JSON file contains profiler/export
  text, not a completed native result; no complete native output was
  retained after the profiler exited. `spark-job busy` and the GPU/process probes correctly
  refuse to call the Spark free. A later attempt to adopt the observed
  process finds it already gone; there is no retained application exit
  code or complete swap record.
- **Impact:** a successful bounded profiler capture is neither application
  completion nor proof that the model remains inside the supervisor's
  process group. This maximum-state attempt is excluded from the gate.
  Its completed `.nsys-rep`/SQLite capture can support bounded GPU
  diagnostics, without qualifying the unfinished host/model test.
- **Workaround:** run long validation commands directly under
  `spark-job`; collect bounded profiles separately with application
  lifetime and output collection kept explicitly supervised. Always
  check `busy` and the full memory/process gate before handing off a
  Spark. Do not infer application success from the profiler's exit.
- **Limits:** this is one observed launch/option combination, not an
  isolated Nsight defect or a claim about why the target later exited.
  Pins, failed native output and completed trace remain in
  `spark:~/scratch/m3-extrapolation-ds/final-max-swap/`; see
  [the final context report](experiments/m3-final-context/README.md).
  The launch wrapper is
  `spark:~/scratch/m3-final-launch/jitllm-final-ds-max-swap.sh`.

## RE-038: GGML's concat has two kernels, and only the per-row one is bound by the grid's 65,535 channels, so a check that bounded both refused DeepSeek V4 past ~52K positions  (2026-09-29, status: fixed)

- **Environment:** `spark-b`, GB10, the `spark-native` build at `6c182c3`;
  the pinned llama.cpp `b29c606e2`'s `concat.cu`.
- **Observed:** `jitllm-runtime` with DeepSeek V4 at `context = 262144`
  failed the 64K prompt's prefill at the chunk at 51,200: "concat beyond
  the kernels' grid" on the CSA layers' concatenation of the window cells
  and the compressed rows, F16 [512, 1, 66,560]. `concat_cuda` launches
  a block per output row, channel and sample (`dim3(ne1, ne2, ne3)`, so at
  most 65,535 channels) only for strided operands; operands contiguous in
  their first three dimensions take `concat_cont`, a one-dimensional grid
  over the plane, and whole copies along dimension 3. jitLLM's operation
  check (`CheckConcat`) applied the per-row kernel's limits to both, so
  every CSA layer past about 52K attended cells (the window's cells plus a
  quarter as many compressed rows) was refused. llama.cpp runs the same
  concatenation at 256K.
- **Fixed:** the check follows `concat_cuda`'s dispatch: the grid limits
  apply only where the per-row kernel runs (unit test at 53,248 + 13,312
  channels, and a strided operand still refused).
- **Impact:** a check that is stricter than its kernel's dispatch is a
  silent context ceiling; the long-context ladder found this one at 64K.

## RE-037: GGML takes [n_kv, rows] strides as 32-bit ints (flash attention's mask, `ggml_permute`), so Qwen3.8 past ~147K tokens in 3,584- or 4,096-row chunks is refused  (2026-09-29, status: worked-around)

- **Environment:** `spark-b`, GB10, the `spark-native` build at `6c182c3`
  (the prefill-chunk slice); the pinned llama.cpp `b29c606e2`: the
  `fattn-mma-f16.cuh` launch signature (`nb31`, `nb32` are `int32_t`) and
  `ggml_permute` in `ggml.c` (its `ne` and `nb` locals are `int`).
- **Observed, twice:**
  - `jitllm-runtime` with Qwen3.8 at `context = 262144` refused to start:
    "flash attention beyond the kernel's 32-bit extents and strides". The
    chunk's F16 mask is [n_kv, rows]; at 262,144 cells and the default
    4,096 rows its plane is 2^31 bytes, one past `INT32_MAX`. The MMA
    kernel reads the mask only through `nb33` (int64, `ne32` is 1), so
    the truncated value would go unused; jitLLM's launcher check
    (`validate_ext.cc`) refuses it anyway, as intended.
  - With the chunk at 3,584 rows the service started, and the 256K
    prompt's prefill failed at the chunk at 146,944 (n_kv 150,528):
    "cont on an empty or unmeasurable tensor". The fallback QSA selection
    (past 32,768 cells, RE-031) permutes the indexer's expanded F32
    scores [rows, n_kv] (2,157,969,408 bytes); `ggml_permute` computed
    the view's plane stride in `int` and gave 18446744071572553728
    (2^64 − 2^31 + …). jitLLM's operation check refused it before any
    kernel read through it. Upstream fixed the truncation in PR #29227
    (after the pin).
- **Worked around:** `model::Qwen38State` and `Qwen38MostRows` bound a
  chunk so an F32 [padded context, rows] tensor stays under `INT32_MAX`
  bytes, which covers both: the widest chunk is 2,047 rows at 262,144
  (the runtime caps `prefill_chunk` at 2,040, whole 8-row tiles, and logs
  it), 4,095 at 131,072 and 8,191 at 65,536. DeepSeek V4's [n_kv, rows]
  tensors are F16 masks; its fast plan's attention mask spans the ring
  and the compressed cells, so at 1,048,576 positions a 4,096-row chunk's
  mask is 2,183,135,232 bytes and the operation check refuses the last
  chunks of a near-ceiling prompt (found in review, 2026-10-02, when the
  default became 4,096 rows; 2,048 rows stay under). `Dsv4MostRows`
  bounds the chunk under 2^31 bytes: 4,096 rows to 1,030,144 positions,
  4,029 at 1,048,576 (a configured 4,096 runs 4,024; the default there
  stays 2,048 for memory).
- **Impact:** any context × rows product past 2^29 cells in F32 trips it
  until the pin includes #29227; a sparse attention path with block
  tables instead of dense masks and expanded scores removes the tensors
  ([long-context](experiments/long-context/README.md)). Since phase 2
  Qwen3.8's default (fast) graph builds none of them, so its chunks are no
  longer bounded (the runtime's 4,096 rows at 262,144); the bound stays
  for its reference and unfused graphs, and the graph builder refuses a
  chunk of theirs past it. The executor's
  refusal now names the refused node and its operand, with shapes and
  strides, which is how the second one was found.

## RE-036: GGML's flash-attention mask pre-pass reads whole 8-row tiles from 1,024 query rows on, so a chunk of 1,024+ rows that is not a multiple of 8 is refused  (2026-09-29, status: worked-around)

- **Environment:** `spark`, GB10, the `spark-native` build at `c7c1ead`
  plus the prefill-chunk slice; the pinned llama.cpp `b29c606e2`'s
  `fattn-common.cuh` pre-pass as `kernels/ggml/fattn_mma.cu` dispatches it.
- **Observed:** `jitllm-runtime chat` with `prefill_chunk = 4096` (both
  DeepSeek V4 and Qwen3.8) failed its first chunk, a whole 2,164- or
  2,362-token prompt, with "the mask pre-pass reads whole column tiles
  past the mask's rows"; 512-, 1,024- and 2,048-row chunks of the same
  prompts ran (their chunks were under 1,024 rows or multiples of 8).
  For 1,024 query rows or more (and a KV length a multiple of 256) the
  kernel's pre-pass (`flash_attn_mask_to_KV_max`) reads the mask in whole
  tiles of up to 8 rows with no bound on the rows, and the models' graphs
  build masks of exactly the chunk's rows, as llama.cpp's own graph does
  (`llama-graph.cpp`, at the pin and at master `8019dc563`, b11254).
  llama.cpp escapes it only because its default micro-batch of 512 rows
  stays under the threshold, and a larger one over-reads into its compute
  buffer unnoticed; upstream CUDA still has the over-read (Metal fixed the
  same bug in llama.cpp PR 29220, merged 2026-09-21, a precedent for a
  CUDA fix). jitLLM's launcher check refuses rather than read past the
  mask, and in the runtime a refused chunk is a node failure.
- **Worked around:** the runtime's prefill (`runtime/prefill.h`) runs a
  chunk of 1,024 rows or more in whole 8-row tiles and its remainder as a
  chunk of its own, and its chunk sizes are multiples of 8. Since the
  long-context baseline the resident harnesses' prompts and perplexity
  chunks tile the same way (`jitllm_dsv4_exec`; `jitllm_qwen38_exec`'s
  64-row `ChunkRows`); their other `--max-rows` paths (synthetic prefill
  benchmarks) are not covered.
- **Impact:** anything that drives GGML attention with 1,024+ query rows
  must keep them a multiple of 8, or pad the mask's rows (in the models'
  chunk inputs and graphs), or bound the pre-pass's row reads (the
  upstream fix, as Metal's).

## RE-035: NVCC contracts `a*b + c*d` and `a*b - c*d` into different FMAs, so one expression copied into another kernel need not give its bits  (2026-09-28, status: worked-around)

- **Environment:** `spark-b`, the SDK's CUDA 13.4 NVCC, sm_121a, default
  `-fmad=true`; `kernels/image`, M3 image speed slice.
- **Observed:** the query norm fused into FlashAttention
  (`flash_attention.cu`) was meant to reproduce `HeadNormRopeComplexKernel`
  (`ops.cu`) bit for bit, from the same C++. The denoiser's latents moved
  (relative RMS to diffusers 0.0174017 → 0.0176318) while a 300-row unit
  test still matched. The SASS of the original showed the complex product
  `n0*c - n1*s` as `fma(n0, c, -(n1*s))` but `n0*s + n1*c` as
  `fma(n0, s, n1*c)`: the product fused is not the same one in the two, and
  a first explicit rewrite that guessed `fma(n1, c, n0*s)` changed the M3
  slice's pixels.
- **Worked around:** both kernels now spell every contraction out with
  `__fmaf_rn` / `__fmul_rn` as the original compiled (read from its SASS),
  the legacy plan's image is again `95fbcbc5…` and the fused plan's latents
  equal it byte for byte; the unit test runs 1,500 rows × 8 heads, enough
  that one query element in 10⁴ rounding differently shows.
- **Impact:** any "same values as that kernel" fusion should compare the
  fused output directly on production-sized data, and write the arithmetic
  with explicit intrinsics on both sides rather than trust the compiler to
  contract two copies alike.

## RE-034: CUTLASS's SM120 MXFP8 GEMM halves its speed on wide products at 8,192 rows unless its tiles are swizzled  (2026-09-28, status: worked-around)

`spark-b`, GB10, CUTLASS 4.7.1's SM120 block-scaled GEMM (MXFP8 × MXFP8,
128 × 128 × 128 tiles, ping-pong or cooperative, the persistent CLC tile
scheduler with its default rasterization), a quick A/B on an idle GPU at
Qwen3.8's shapes. At 4,096 rows the kernel is 1.9–2.5× cuBLAS's BF16
product; at 8,192 rows the wide products fall behind cuBLAS: QKV (n
10,240, k 2,560) 4.66 ms in BF16 out and 7.5 ms in F32 out against 5.3 ms,
Q (n 12,288) 5.6 and 9.0 against 6.3, while the narrow ones (n 2,560 or
6,144) stay ahead. Setting the scheduler's `max_swizzle_size` to 8 with
raster along N gives QKV 2.59 ms and Q 3.09 (F32 3.3 and 3.9), and at
4,096 rows the swizzle is slightly slower (QKV 1.31 against 1.18), which
fits an L2 working-set cliff (the A and B operands of a wave no longer
share the L2). Impact: `kernels/ggml/mxfp8_cutlass.cu` swizzles past 4,096
rows; anything else on CUTLASS's SM120 persistent GEMMs at this size
should A/B the swizzle rather than trust the default heuristic.

## RE-033: GGML's MMVQ changes its launch with the column count, so a multi-token verify's rows differ in their last bits from one-token decoding  (2026-09-28, status: worked-around)

`spark-b`, GB10, the pinned llama.cpp `b29c606e2`'s `mmvq.cu` as jitLLM
builds it. `calc_nwarps` and `calc_rows_per_block` depend on `ncols_dst`:
on the GB10's table one column of Q8_0, Q4_K, Q5_K or Q6_K runs 8 warps,
two to four columns run 4, so each dot product's partial sums are
reduced in a different order and a column of a 4-column product differs
in the last bits from the same column run alone. Flash attention's MMA
kernel tiles and splits by query rows too, and a float product above one
column leaves MMVF for MMF or cuBLAS. A speculative verify of k + 1 rows
through upstream's plan is therefore not bit-identical to k + 1 decode
steps, one reason llama.cpp's own speculation is not (its issue #25618).
Worked around by D-092's row-invariant plan
(`mmvq_rows.cu`: MMVQ's body with the one-column launch for every
column, attention per query row). Bites again: any new multi-row path
expected to equal one-row decoding, and any comparison of llama.cpp's
speculative and plain outputs.

## RE-032: GGML's ssm_conv reads up to 31 floats past its window when the tokens exceed 32 and are not a multiple of 32  (2026-09-28, status: worked-around)

Environment: `spark-b` (GB10), llama.cpp `b29c606e2`'s `ssm-conv.cu` under
jitLLM's dispatch (`kernels/ggml/ops_ext.cu SsmConv`), SDK CUDA 13.4.

Observed: `unit.Qwen38FusedTest.GdnConvIsTheUnfusedNodes` aborted once
under the Spark test tier's parallel run and never alone; the process
aborted without a message (the fence query's error through `.value()`).
`compute-sanitizer --tool memcheck` found 25 invalid global reads in
`ssm_conv_long_token_f32<false, 128, 4, 32>` at 40 tokens, 13–17 bytes past
the window's allocation. Past 32 tokens the launcher splits the tokens into
32-token blocks, and each block loads `d_conv - 1 + 32` columns of every
channel's row into shared memory whatever `local_n_t` is, so the last
block of the last channel reads up to `(32 - n_t % 32) % 32` floats past
the window. Only the loads are unbounded: the outputs read loaded columns
below `local_n_t + d_conv - 1`. In llama.cpp the window sits in a larger
compute buffer, so the over-read goes unnoticed.

Expected: loads bounded by the window's columns.

Impact: `CheckSsmConv` did not refuse it, so any unfused graph with an odd
chunk past 32 tokens (Qwen3.8 with `--unfused`, or a prompt's last chunk)
read past the concatenation it convolves, inside the activation region
(no fault seen there). The fused graph's `jitllm.gdn.conv` does not over-read
and takes every chunk of 3 or more tokens.

Workaround (the prefill slice's review): `CheckSsmConv` refuses more than
32 tokens that are not whole 32-token blocks, and the unfused graph pads
such a window to whole blocks with copies of its first columns and drops
their outputs (the others are the same bit for bit); the fused test builds
its unfused reference the same way, with no slack. Fix upstream: bound the
load by the window's columns.

## RE-031: GGML's radix top-k picks among tied values nondeterministically, so Qwen3.8's QSA selection varies run to run past 2,051 cells, and DeepSeek V4's indexer past 4,096 positions  (2026-09-28, status: worked-around in both models' fast plans at any depth (Qwen3.8's since its long-context phase 2); open for both reference (and unfused) graphs)

`spark-b`, GB10, driver 580.178.04, the pinned llama.cpp `b29c606e2`'s
`top-k.cu` as jitLLM builds it (no CUB). For rows over 1,024 columns
`top_k_radix_cuda` compacts the elements above the threshold and those
equal to it with `atomicAdd` on per-row counters (`top-k.cu:170-173`), up
to 64 blocks a row: which of several equal values land in the first k
depends on thread timing. Qwen3.8's QSA indexer selects the top
`budget + ratio − 1` = 2,051 cells from scores that are per block of 4
cells (every score appears four times) and ReLU'd (many exact zeros), so
wherever the k-th value is tied the selected cells, and with them the
attention, change between runs. Observed with M3's swap runner
(`jitllm_swap_pairs`, [swap.md](experiments/fast-swap/swap.md)): the same
8,192-token prefill (16 chunks of 512) in one process, weights and state
identical, gave logits that differ from the 11th to the 16th chunk on
(positions past 5,120) in seven of twelve reruns; two processes'
greedy continuations of the same context diverged at the 10th token. The
first 8 chunks (n_kv ≤ 4,096) matched in a repeat, and short prompts
(n_kv ≤ 2,051, no selection) always do, which is why qwen38-native's
repeats were identical. Mia's vLLM is not repeatable by default either,
and its deterministic mode, the oracle's, sets `VLLM_QSA_DET_TOPK=1`
(baselines.md). DeepSeek V4's top-k
(the lightning indexer's) showed no such difference until the
prefill-chunk slice (below).
Impact: jitLLM's Qwen3.8 is not repeatable past 2,051 cells; any
bit-identity check there must compare against the same state, not a rerun
(the swap runner snapshots the state and runs the unswapped continuation
from it). Fix: a top-k that breaks ties by index (a patch to GGML's radix
select, or jitLLM's own), part of the Qwen3.8 work. Since the second
prefill pass the fast graph (the default, in prefill and decode) selects
with jitLLM's own `jitllm.qsa.select`, which keeps the lower cell among
equals, and its perplexity run (3,557 positions, past the budget)
repeated exactly three times
([qwen38-native](experiments/qwen38-native/README.md#results-second-pass));
the reference (`--exact`) and unfused graphs keep GGML's top-k, and so
did the fast graph past 32,768 cells (the kernel's shared memory). Since
long context's phase 2 (2026-09-29) the fast graph selects on the device
at any depth up to its configured 262,144 (`jitllm.qsa.topk`: a
byte-wise radix select over tiles of 8,192 blocks, then over the tiles'
candidates, ties to the lower cell): two runs of the same 64K and 128K
forced prompts gave identical logits at all 512 steps, so **for Qwen3.8's
default graph RE-031 is closed**
([long-context](experiments/long-context/README.md#phase-2-qwen38-flash-next-flat-with-depth)).
The reference and unfused graphs keep GGML's top-k and are still not
repeatable past 2,051 cells.

**DeepSeek V4 too** (`spark`, 2026-09-29, the `spark-native` build of
the prefill-chunk slice): its lightning indexer keeps the top 512 of its
compressed cells with GGML's top-k, the radix path once there are more
than 1,024 of them (past 4,096 positions). `jitllm-runtime chat` prefilling
the same 8,088-token prompt from a cleared state in 4,096-row chunks gave
one of two last rows (top logit 22.639 or 22.914) across 7 runs; with the
radix gather replaced by an index-ordered one (an experiment on the
prepared source, not kept) 6 of 6 runs repeated exactly, so the ties are
the whole cause, not the chunking. 512-row chunks also gave two outcomes
(22.389 three times, 22.579 once) and 2,048-row chunks repeated 7 of 7,
but ties depend on the data, so no chunk size is immune. With the
index-ordered gather the prompt's result was also the same whether it
ran first or after another turn (512 and 4,096 rows), so no earlier
turn's state leaks into a cleared one.
Impact and fix as above: DeepSeek's prefill past 4,096 positions is not
repeatable either; a top-k that breaks ties by index would fix both
models.

**Worked around for DeepSeek V4's fast plan** (`spark`, 2026-09-29, the
long-context phase 2 slice): its indexer now scores and selects with
jitLLM's own `jitllm.dsv4.lid_topk` (`src/kernels/ggml/dsv4_sparse.cu`), a
radix select that keeps the lower row among equals, and two runs of the
32K forced prompt (31,705 tokens, 512 steps) gave the same logits bit for
bit (`judge.py repeat`: 0 of 512 steps differ), where before they differed
from the first step. The reference mode (`--exact on`) keeps GGML's top-k,
as llama.cpp runs it, and does not repeat.

## RE-030: GGML's tensor-core flash attention reads attention sinks past the last head when query heads per KV head are not a multiple of 8  (2026-09-28, status: worked-around)

Environment: `spark-b` (GB10), llama.cpp `b29c606e2`'s
`fattn-mma-f16.cuh` under jitLLM's dispatch (`kernels/ggml/fattn_mma*.cu`),
SDK CUDA 13.4.

Observed: with sinks, 24 query heads over 2 KV heads (12 per KV head, as
Qwen3.8's QSA) and D = 256, the kernel faulted with an illegal address when
the 24-float sinks tensor ended flush against an unmapped VMM granule (the
over-read probe in `tests/unit/ggml_ext_ops_test.cc`). The kernel groups 8
query heads per tile (`ncols2`) and reads `sinks_f[jc % ncols2]` from each
group's first head (`fattn-mma-f16.cuh:1402`, `1889`) with no bound, so the
last KV head's second group reads 4 floats past the tensor; the Q loads and
output writes of those padded heads are bounded, the sinks are not. In
llama.cpp a sinks tensor sits inside a larger weight buffer, so the
over-read goes unnoticed and only feeds heads that are discarded.

Expected: sinks read for existing heads only.

Impact: `CheckFlashAttnMma` refuses sinks unless the heads per KV head are
a multiple of 8 (DeepSeek V4's 64 are). A model with sinks and another GQA
ratio needs a padded sinks tensor, or a kernel fix upstream.

**Checked 2026-09-28 for Qwen3.8's QSA** (24 query and 2 KV heads of 256):
it has no sinks, so the over-read does not arise; without sinks the
padded heads' loads and writes are bounded and the result matches an FP64
reference (`unit.Qwen38OpsTest.TensorCoreAttentionAtTwelveQueryHeadsPerKvHead`),
and the check still refuses sinks at that ratio
([qwen38-native](experiments/qwen38-native/README.md#re-030)).

---

## RE-029: A job's kernel launches can block its lane while the stream is busy  (2026-09-27, status: worked-around)

On `spark-b` (GB10, driver 580.178.04), `jitllm_exl3_paged
--cancel-in-flight` queues the 1,023-row EXL3 prefill as one device job
behind a gate: a `cuStreamWaitValue32` on a host flag, queued first.
- The job's launches stopped returning to the submission lane while the
  gate held. They were still blocked 60 s later, and returned only once
  the gate opened. The likely cause, not measured further: the phase has
  more launches than the stream can hold pending, and the driver blocks the
  launching thread while that queue is full (its depth is not documented).
- Nothing synchronizes in jitLLM's code: `src/kernels/` has no
  synchronizing call.

Impact: `commands.h` says a job only queues and never waits. The driver can
still make it wait, whenever the stream is busy with earlier work and the
phase is long. The device submission lane is then held. The same lane
submits the zone's page-in copies, so a long phase queued behind a slow
one can delay page-ins.
- Correctness is unaffected: leases hold until the fence, and cancellation
  still drains (the run above).
- It matters for M3/M4 latency. Likely fixes are a separate submission
  lane for the zone's copies, or phases split into jobs that fit the
  queue. Measure before choosing.

**Checked 2026-09-28: the wait is not the cause; a full stream is.**
MiaAI-Lab's `patch_ple_offload.py` (at `b8439110`) states that the GB10
reports `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 0` and that after a
`cuStreamWaitValue32` "the *next* kernel launch on that stream blocks the
host thread". A probe on `spark` (GB10, driver 580.178.04, driver API
13000), built with the SDK (`aarch64-e0a0c85c42806fb1`, CUDA 13.4.92
headers, clang 22.1.8), gave, in two identical runs:
- `cuDeviceGetAttribute`: `CAN_USE_STREAM_MEM_OPS_V1` (92) = 0,
  `CAN_USE_64_BIT_STREAM_MEM_OPS_V1` (93) = 0,
  `CAN_USE_STREAM_WAIT_VALUE_NOR_V1` (94) = 0,
  `CAN_FLUSH_REMOTE_WRITES` (98) = 0, `CAN_USE_64_BIT_STREAM_MEM_OPS`
  (122) = 1, `CAN_USE_STREAM_WAIT_VALUE_NOR` (123) = 1. CUDA 13.4's
  `cuda.h` has no unsuffixed `CAN_USE_STREAM_MEM_OPS`; attribute 92 is now
  the deprecated `_V1`, and Mia's 0 is that value. The current memory
  operations are supported.
- A `cuStreamWaitValue32` on a mapped host flag returned `CUDA_SUCCESS` and
  gated the stream. Behind it, 1,020 empty-kernel launches returned at
  once (the first in 6–7 µs); the 1,021st blocked the host for the whole
  3 s gate and returned 7–12 µs after the flag was written.
- Behind a 3 s spinning kernel instead of the wait, the 1,022nd launch
  blocked the same way, until the kernel ended.
- With one stream holding 1,000 gated launches, 20,000 launches and 2,000
  small `cuMemcpyHtoDAsync` calls on a second stream never blocked.

So a stream holds about 1,020 pending operations, whatever it waits on,
and a launch into a full stream blocks the calling thread; other streams
are unaffected. Mia's "next launch" did not reproduce with the driver API.

**Also blocked, 2026-09-28 (`spark-b`, same driver):** while one thread is
blocked launching into a full stream, `cuEventCreate` and
`cuStreamCreate` on any other thread block too, for as long as it is
(30 s in the probe below: until the gate opened). `cuEventRecord`,
`cuEventQuery`, `cuMemcpyAsync` and a synchronize on another, idle stream
returned at once. So a separate thread and stream are not enough: a lane
that fences its work must not make events as it goes.

**Worked around 2026-09-28 (M3's swap path):** the zone's copies, in and
out, run on a copy lane of their own (a second device service with its own
submission and completion threads over the zone's stream,
`scheduler.h` `Lanes::copy`), and the CUDA device-execution provider takes
fences' events from a pool made when it opens and kept when a fence is
released (`cuda_device_execution.h`); the paged node makes as many as its
device and copy lanes can hold fences at once (1,042), so neither makes
one as it goes. A job blocked launching into a full
stream then holds up no page-in; a DeepSeek chunk is 4,972 launches.
Checked on `spark-b` by `unit.CudaPagedNodeTest.ACopyLaneLandsPageInsWhileAJobFillsItsStream`
(a job gated by `cuStreamWaitValue32` on a host flag stops after about
1,020 of its 1,500 operations, and a 6 MiB page-in completes in 11–13 ms
meanwhile; without the pool it waited for the gate) and by the swap
runner's overlap probe ([swap](experiments/fast-swap/swap.md)). The depth
was not measured again: the test asserts only that the job stops short.

What it implies for page-in copies sharing the submission lane: the lane
must never launch into a stream that may be full. A phase of more than
about 1,000 launches, queued behind unfinished work on its stream, stalls
the lane and every page-in copy waiting behind it, while the copy stream
itself stays free. The options are the two above, now with a number:
submit the zone's copies from their own thread, or keep each stream's
queued launches under the limit (split phases into jobs of well under
1,000 launches, or count a stream's outstanding launches before
submitting). The depth is observed, not documented, and may change with
the driver; whichever design M3 picks measures it again.

**Measured 2026-09-28: a graph replay is one entry** (`spark-b`, same
driver; `unit.CudaGraphTest.AGraphOfManyKernelsIsOneOperationInItsStream`).
Behind a `cuStreamWaitValue32` gate, a stream took 1,020 replays of a
1,500-kernel graph before the next `cudaGraphLaunch` blocked: the same
depth as plain launches, one entry per replay whatever the graph holds.
DeepSeek's decode steps now replay as graphs (D-090), so a decode job
queues one operation instead of 4,972 and never fills its stream; its
prefill chunks still do.

## RE-028: cuBLAS's handle keeps a 64 MiB default workspace pool that `cublasSetWorkspace` does not free, and nsys's memory trace hides who allocated it  (2026-09-27, status: worked-around)

Environment: `spark` and `spark-b` (GB10, kernel 7.0.0-1019-nvidia, driver
580.178.04), cuBLAS 13.8.0.4, Nsight Systems 2025.3.2
(`--trace=cuda --cuda-memory-usage=true`, SQLite export).

Observed:
- `cublasCreate` makes three `cudaMalloc`s: 1,024, 131,072 and 67,108,864
  bytes. They stay allocated until `cublasDestroy`, even after
  `cublasSetWorkspace` supplies a workspace (GGML's and jitLLM's 32 MiB)
  that later calls use. With `CUBLAS_WORKSPACE_CONFIG=:4096:2` the two
  large ones become one 8 MiB allocation, and with `:16:8` one 131,072-byte
  one, so they are cuBLAS's default workspace pool. The documentation says
  that pool is "allocated during the cuBLAS context creation" (cuBLAS docs,
  section 2.4.8). On the FP16 bridge these 64.1 MiB were the only
  API-visible memory beyond its declared buffers, identical in all 32 nsys
  passes on both Sparks.
- nsys's memory trace reports each allocation's size exactly, but not its
  caller. `--cudabacktrace` needs CPU sampling, which is unsupported here
  (`perf_event_paranoid` 4), so the pool was named only by resizing it.
- The trace's timestamps count CLOCK_MONOTONIC_RAW from the session's
  start. A program's CLOCK_REALTIME drifted from it by about 3 µs/s
  (0.24 ms after a minute).
- `nsys profile` starts `nsys --start-agent`, which leaves the launcher's
  process tree and holds `/dev/nvidia*` open.

Expected: `cublasSetWorkspace` to release the default pool, or the
documentation to say it does not. Trace time in a documented clock.

Impact: every cuBLAS handle costs 64.1 MiB beyond the workspace jitLLM
gives it, and jitLLM cannot free it (`CUBLAS_WORKSPACE_CONFIG` is refused as
a numerics switch). The owner decided on 2026-09-27 that it does not count
against the 32 MiB persistent-workspace figure, and that native may hold
what the bridge holds (backend-proof.md, "Memory and workspace"). To place
readings on an nsys trace, use CLOCK_MONOTONIC_RAW, anchored once by
CLOCK_REALTIME. A GPU-idleness monitor must count the nsys agent as the
run's own.

## RE-027: The Spark's SSD reads recently written data ~11% faster than data at rest  (2026-09-27, status: open)

On `spark-b` (Samsung `MZALC4T0HBL1-00B07`, ext4 root, kernel
7.0.0-1019-nvidia), 2 MiB `O_DIRECT` reads at four in flight ran at
14.7–14.9 GB/s from files written minutes to an hour before, and at
13.2–13.4 GB/s from files at rest, whatever read them (in place into host
VMM or through the landing zone, the runtime or the standalone probe).
In one run, a 9 GiB pattern file written seconds before read at
14.65–14.80, and a 16 GiB one that had read at 14.7–14.9 for the hour
after it was written read at 13.20–13.41, 90 minutes after; a GGUF shard
written six days before read at 13.17–13.41, and the FP16 artifact's shard (installed four
days before) at 12.2–13.3 against 14.3–14.9 for a `dd` copy of it.
Not fragmentation: `filefrag` on `spark-b` found the six-day-old GGUF
shards (32–49 GB) in 18–33 physical runs averaging 1.5–1.9 GB, the 16 GiB
pattern file after it slowed in 22 runs (~745 MiB on average), and the
FP16 artifact's shard in 15 runs for its 988 MB against 5 for the fresh
`dd` copy: all effectively contiguous for 2 MiB reads. The drive's write
cache (SLC) serving recent writes, displaced by later writes or by time,
fits the timing but is inferred, not proven. So the device's bandwidth for artifacts
at rest is ~13.3 GB/s, not the ~14.9 of earlier measurements, which all
read files just written (io-path, storage-queue, dmabuf-direct). Compare
paths on the same file in the same session, and state the file's age.
Evidence: [pagein-perf](experiments/pagein-perf/README.md).

## RE-026: The Spark's SSD reads 4 KiB-offset 2 MiB requests ~18% slower out of order  (2026-09-27, status: worked-around)

On `spark-b` (Samsung `MZALC4T0HBL1-00B07`, ext4 root, kernel
7.0.0-1019-nvidia, io_uring `O_DIRECT` reads of 2 MiB, four in flight),
reads of a pattern file whose offsets were 2 MiB multiples ran at
14.5–14.9 GB/s whether or not consecutive requests were in file order. With
every offset 4 KiB past a 2 MiB multiple, as an artifact's chunks are
(D-056 aligns groups to 4 KiB), the same reads ran at 14.4–14.8 GB/s in
order but 11.9–12.4 GB/s when about 60% of consecutive submissions were
locally swapped (a window of about eight). The runtime's direct reader
started queued reads in the order of their keys, which are reused mailbox
indices, so page-in through the zone hit this and in-place reads (fresh
keys, in order) did not. Worked around: the reader starts reads in arrival
order (`providers/direct_reader.h`). Keep reads of one load in file order.
Evidence: [pagein-perf](experiments/pagein-perf/README.md).

## RE-025: GB10 device memory cannot be exported as a dma-buf, and NVIDIA dma-buf mappings refuse direct I/O  (2026-09-27, status: open)

On `spark` (GB10, driver 580.178.04, CUDA 13.0 toolkit, kernel
7.0.0-1019-nvidia), `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` is 0.
`cuMemGetHandleForAddressRange(..., CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, ...)`
returns `CUDA_ERROR_INVALID_VALUE` (not `NOT_SUPPORTED`) for device VMM and
host VMM, with or without the PCIe flag, and for `cuMemAlloc`. The CUDA 13.4
header's `CU_DEVICE_ATTRIBUTE_DMA_BUF_MMAP_SUPPORTED` (152) is unknown to
this driver (`INVALID_VALUE`), and its "cached mapping on coherent ARM" note
does not apply. Only `cuMemAllocHost` exports. Its `mmap` is CPU-cached but
is a PFN map (`VM_PFNMAP | VM_IO`, smaps `pf io`), so `O_DIRECT`, io_uring
reads and `IORING_REGISTER_BUFFERS` into it fail with `EFAULT`.
`pin_user_pages` refuses such VMAs, and the open module's `nv_dma_buf_mmap`
says so; 610.57.04's source still maps by PFN (read, not tested). The
reverse direction works outside the documentation: a `udmabuf` of a shmem
`memfd` imports through `cuImportExternalMemory(..._DMABUF_FD)`
(documented for Jetson Thor only), and a hugetlb one is refused. The GPU treats it as host-located memory: no
L2 reuse, and 2.6 GB/s random reads with 4 KiB pages. Net: no route lands a
direct read in L2-cacheable memory on the Spark. Evidence:
[dmabuf-direct](experiments/dmabuf-direct/README.md).

## RE-024: MemAvailable misses host memory held on the per-CPU page lists  (2026-09-27, status: worked-around)

Environment: `spark-b` (kernel 7.0.0-1019-nvidia, 4 KiB pages, 20 CPUs,
driver 580.178.04), an otherwise idle host.

Observed: 64 MiB from `malloc`, written, then freed, moved `RssAnon` by
exactly 64 MiB every time, but `MemAvailable` (and `MemFree`) by anything
from 0 to 62 MiB, and by −15.8 to +70.6 MiB across the census runs' 144
control moves. Waiting 1 to 2.5 s did not help. Freed order-0 pages go to
the freeing CPU's page list and allocations are served from it, without
moving `NR_FREE_PAGES`, which `MemAvailable` reads. This kernel's adaptive
list sizes allow up to 198,125 pages (774 MiB) per CPU per zone
(`/proc/zoneinfo`, `high_max`). `cudaMalloc` and VMM backing, taken in
large blocks, moved `MemAvailable` within about 1 MiB of their size.

Expected: `MemAvailable` to follow an allocation of tens of MiB.

Impact: the M2 census rule (backend-proof.md) voided every run through its
host control. Adding the lists' pages (each zone's pagesets `count:` in
`/proc/zoneinfo`, readable by any user) to `MemAvailable` brings the host
control within 0.4 MiB of 64 MiB; the census reads that sum. Root can
drain the lists instead (`vm.percpu_pagelist_high_fraction`, compaction),
which an unprivileged harness cannot.

## RE-023: ext4 reuses inode numbers at once, and coarse timestamps hide the swap  (2026-09-27, status: worked-around)

Environment: the workstation (kernel 7.0.0-31-generic) with a loop-mounted
ext4 made with `mkfs.ext4 -I 128`, and `spark-b` (kernel 7.0.0-1019-nvidia,
ext4 with 256-byte inodes).

Observed: a file deleted and created again gets the same inode number
straight away on both. With 128-byte inodes ext4 has no room for
sub-second timestamps, so the new file's `st_ctim` was also identical to the
old one's (whole seconds, nanoseconds zero) and, at the same size, `fstat`
could not tell them apart. `FS_IOC_GETVERSION` did: the inode generation
differed (`dedc637f` against `7142ca29`); ext4 draws a new one for every
inode it creates. Overlayfs and tmpfs answer the ioctl with `ENOTTY`.

Expected: device, inode and status-change time to identify a file between
two opens.

Impact: the artifact reader's shard identity (`OpenShardForDirectRead`,
`src/artifact/artifact.h`) includes the generation where the file system
reports one; the replacement test fails on the 128-byte-inode ext4 without
it. Where there is none (overlayfs, tmpfs, NFS), a same-size replacement
within one timestamp tick is not detected. The request's declared argument
is a `long` even though ext4, btrfs and xfs write an `int`: FUSE copies back
up to the declared 8 bytes from its server, so the buffer must be a `long`.

## RE-022: The GB10's L2 does not keep host-located CUDA memory, so re-reads go to DRAM  (2026-09-27, status: worked-around)

On `spark` (GB10, driver 580.178.04), GPU reads of memory that CUDA
allocates at a host location miss L2 every time they re-read it. That
covers `cuMemCreate` at `HOST_NUMA` or `HOST` (jitLLM's host VMM, whether
mapped for the CPU or not) and `cudaMallocHost`. A kernel re-reading a
4 MiB buffer gets 0 of 8,388,608 L2 sector hits and 243 GB/s (DRAM rate).
The same kernel over `cudaMalloc` or device VMM gets 98.4% hits and
1,952 GB/s. Streaming reads run at ~240 GB/s from every kind, so a
bandwidth scan cannot show the difference; that is how D-034's evidence
missed it. Blocks that write to such memory also finish more slowly
(151,936 one-write blocks: 246 vs 96 µs). GEMMs, grouped attention and
matrix-vector products re-read through L2, and ran 1.1–4.9× slower with
all their buffers there (BP-F1). Ordinary memory read through ATS
(pageable, registered, managed) is cached in L2, but it streams at only
~165 GB/s. The CUDA 13.4 headers offer no allocation or access flag for
caching, and a persisting access-policy window does not change it.
`cuMemSetAccess` refuses to map device VMM for the CPU
(`CUDA_ERROR_NOT_SUPPORTED`), and it cannot be exported as a dma-buf
either (RE-025). Measurements, options and reproduction:
[host-vmm-diagnosis](experiments/host-vmm-diagnosis/README.md).
Before placing any buffer the GPU re-reads in host-located memory on the
Spark, measure it with a re-reading kernel, not a scan. D-081 keeps
weights and state in device VMM and uses host VMM only as a landing zone.

## RE-021: GGML's graph object trips UBSan on creation  (2026-09-27, status: worked-around)

In the `cross-asan` build on `spark-b` (address and undefined sanitizers,
GGML's `ggml.c` instrumented too), `ggml_new_graph_custom` and
`ggml_graph_overhead_custom` end the process: `ggml_graph_nbytes` sizes the
graph by advancing a null pointer (`ggml.c:7424`, "applying non-zero offset
96 to null pointer"). The plain builds never notice. So jitLLM builds no
`ggml_cgraph`: the fusion gates (`src/kernels/ggml/fusion.h`) take a node
list in GGML's order (`GraphOrder`) and count uses as GGML's graph does.
Anything else that needs a `ggml_cgraph` (the graph allocator, CPU
diagnostics) meets this first under the sanitizers.

## RE-020: The reference container denies io_uring setup  (2026-09-26, status: worked-around)

On the workstation, the three `unit.UringTest.*` tests pass, but the
default reference container used by `check:full` returns `EPERM` from
`io_uring_setup`, before any I/O. The tests now explicitly skip when setup
is unavailable (`ENOSYS`, as under qemu-user) or denied (`EPERM`), while
other setup errors still fail. Real I/O remains covered on the workstation
and by `check:spark`; an offline-container pass does not claim that coverage.

## RE-019: CUDA VMM backing escapes cgroup memory accounting on the Spark  (2026-09-25, status: open)

Environment: `spark-c4e2`, GB10, driver 580.178.04, kernel
7.0.0-1019-nvidia, cgroup v2; backing created with `cuMemCreate` through
jitLLM's CUDA provider.

Observed: 8 GiB of device-local or host-NUMA backing moved the process's
cgroup `memory.current` by at most 40 MiB, while `MemAvailable` fell by the
full 8 GiB at creation. Device backing never appears in the process's RSS;
host backing appears there (as `RssFile`) only while mapped with access.
The driver's per-extent bookkeeping (about 34 KiB of unreclaimable slab per
2 MiB extent) is not charged to the cgroup either.

Expected: memory a process pins to be charged to its cgroup.

Impact: `MemoryMax=` on `jitllm.service` would not bound the runtime's
backing, and an OOM decision based on the cgroup would not see it. The
runtime's own budget `B`, checked on every materialization, is the bound;
the memory breakdown reconciles against `MemAvailable`. Measurement:
[vmm-counters](experiments/vmm-counters/README.md).

## RE-018: Btrfs quietly serves misaligned or compressed direct I/O through the page cache  (2026-09-25, status: open)

Environment: the workstation's build tree, btrfs mounted with
`compress=zstd:1`, kernel 7.0.0-31-generic. A file opened with `O_DIRECT`
and read at file offset 1, through io_uring or `preadv`, returned the data
(4096 bytes) instead of `EINVAL`.

Expected: a refusal, as ext4 and XFS give. Btrfs falls back to buffered I/O
for direct I/O it cannot do in place (misaligned requests, and compressed
extents), so the page cache fills and the caller cannot tell.

Impact: a storage role on btrfs can pass D-034's direct-I/O probe
(`platform/direct_io.h`, which accepts btrfs) yet page through the cache,
against D-034's no-page-cache intent. Tests that expect the kernel to refuse
misaligned direct I/O accept either outcome on btrfs
(`unit.UringTest.*`). The Spark roles are ext4.

## RE-017: A sleeping thread takes hundreds of microseconds to wake on the Spark  (2026-09-24, status: worked-around)

Environment: `spark-c4e2`, GB10 (Cortex-X925/A725), DGX OS 7.6.0, kernel
7.0.0-1019-nvidia, cpuidle `acpi_idle` with the `menu` governor (LPI-0 to
LPI-3, exit latencies 0/42/231/433 µs), cpufreq `performance`; jitLLM's
`WakeFlag` (a mutex and condition variable) built with the pinned SDK.

Observed: after a 100–400 µs idle gap, a thread sleeping on a condition
variable took 207–283 µs at p50 and 451–485 µs at p99 to run after it was
signalled (three runs of 5,000 samples; earlier runs gave p50 from 88 to
370 µs). The workstation (i9-11900KF, `intel_idle`) took 2.7–72 µs at p50
across all runs. Polling with `yield` woke in under
1 µs on both, at a whole core's CPU.

Expected: tens of microseconds, as on the workstation.

Likely cause, not isolated: the governor choosing LPI-2 or LPI-3 for the
idle gap. Disabling idle states to confirm it is a system change that needs
the owner's approval.

Impact: a scheduler that sleeps between a launch and its completion can add
up to about half a millisecond per step at the tail. It should poll while a
critical-path completion is imminent, and sleep only when idle. Measurement
and harness: [task-lanes](experiments/task-lanes/README.md). The storage and
device submission lanes poll the same way: asleep between a load's reads
and copies, they took ~100–200 µs, and at the tail ~380 µs, to
wake for the next one, which kept the landing zone below depth
([pagein-perf](experiments/pagein-perf/README.md)). On decode (2026-09-28,
`spark-b`), a scheduler and device lane that slept between a request's
steps added 0.26–0.65 ms a step (DeepSeek, Qwen3.8); polling for 100 ms,
longer than a step, left 0.01 ms, at the cost of two cores (the
scheduler's and the device submission lane's threads) spinning while a
request steps and for 100 ms after its last step. That window is the
paged harness's (`NodeSettings::poll_window`); the scheduler's and the
lanes' defaults stay 200 µs
([swap](experiments/fast-swap/swap.md#a-lease-per-request)).

**Worked around 2026-09-28: the runtime wake (D-094,
[runtime-wake](experiments/runtime-wake/README.md)).** Waking on the GPU's
own signal is slower still on `spark-b` (same driver): a blocking-sync
event's wait returned 1.0–1.4 ms after the step's end at the median, a
host function's futex 1.4–1.7 ms, and a host function holds its stream
until the driver's callback thread runs it. So the device completion
lane sleeps through most of a fence's expected length (its stream's
recent lengths), spins only around its likely ends, and as it starts to
spin wakes the scheduler and the submission lane to poll ahead of the
completion; the scheduler polls after a step for about as long
as its client takes to ask for the next. With a synthetic 45 ms step,
the gap from a step's end to the next step's start fell from 0.56–0.67 ms
(200 µs windows, 2 cores busy) to 26–28 µs at the median, at 0.11–0.12 of
a core while stepping and none while idle; every thread polling (the
harness's 100 ms) took 9 µs and 4 cores. The idle states themselves are
unchanged: a thread that sleeps when it was not anticipated still pays
this.

**Cause confirmed 2026-09-28: the cores' power-down states (D-095,
[runtime-wake](experiments/runtime-wake/README.md#a-latency-hold)).** A
PM QoS request of 0 µs (`/dev/cpu_dma_latency`, only LPI-0, WFI) cut each
sleeping hop on `spark-b` to ~5 µs, D-094's gap from 28–31 to 8–9 µs,
the paged node's to ~10 µs, and a blocking-sync event's wake from 1.0–1.5
ms to 5–35 µs; a request of 50 µs, which still allows LPI-1 (42 µs exit
latency), changed nothing. DeepSeek's decode was no faster for it (the
~0.03 ms a step it saves is under the device's spread), so the runtime
does not hold one (D-095, not adopted); revisit for short-step
workloads.

## RE-016: Ubuntu's snapshot service has no ports archive, so arm64 packages cannot be pinned by date  (2026-09-24, status: worked-around)

`https://snapshot.ubuntu.com/ubuntu-ports/<timestamp>/` answers HTTP 401,
while `https://snapshot.ubuntu.com/ubuntu/<timestamp>/` serves the amd64
archive. With `APT::Snapshot` set in an arm64 Ubuntu 24.04 container, `apt-get
update` still fetches the live `ports.ubuntu.com` indexes, and `apt-get
install systemd` then fails with "Unable to locate package". So the arm64
install-test image (`packaging/install-test/`) takes systemd from the live
ports archive and the test prints the version it got. The SDK is unaffected:
it pins each arm64 `.deb` by URL and SHA-256 (D-070).

## RE-015: A multi-arch image digest can run the wrong architecture from the local image store  (2026-09-23, status: worked-around)

Workstation, Docker 29.8.1 with the containerd image store and qemu binfmt
registered. `docker run ubuntu:24.04@sha256:008173c2…` (the multi-platform
index digest) ran the **arm64** image under qemu-user, with only the warning
"The requested image's platform (linux/arm64/v8) does not match the detected
host platform". An arm64 `ubuntu:24.04` pulled for the D-061 qemu tests was
the local content for that digest. With binfmt registered, nothing fails, so
builds and tests silently run emulated for the wrong target. Workaround: the
reference container's `FROM` and every `docker run` or `docker build` of it
name `--platform linux/amd64` (the Dockerfile skips BuildKit's
`FromPlatformFlagConstDisallowed` check on purpose), and `doctor` inside the
container reports the architecture.

## RE-014: LeakSanitizer aborts AArch64 tests under qemu-user  (2026-09-23, status: worked-around)

Workstation (x86-64), Ubuntu `qemu-user-static` 1:8.2.2+ds-0ubuntu1.18 via
binfmt, running the D-059 GoogleTest suite cross-built with Clang 22.1.8,
`-fsanitize=address,undefined`, the D-060 static GCC 16.2 runtime and the
arm64 compiler-rt from the D-059 SDK, with `QEMU_LD_PREFIX` set to the
sysroot. All 10 tests pass. At exit, LeakSanitizer reports "LeakSanitizer has
encountered a fatal error" and the process exits 1, so a passing suite looks
failed. With `ASAN_OPTIONS=detect_leaks=0` it exits 0, and ASan still catches
the heap-overflow probe. Workaround (D-061): leak detection is off for
emulated AArch64 runs; LSan runs natively on x86-64 and on the Sparks.
ThreadSanitizer under qemu-user was not tried.

## RE-013: Ubuntu 24.04 blocks unprivileged network sandboxes (`unshare -rn`, `bwrap`)  (2026-09-23, status: worked-around)

The workstation, `spark` and `spark-b` (Ubuntu 24.04.5) all set
`kernel.apparmor_restrict_unprivileged_userns=1`. Two unprivileged ways to
deny a build its network both fail:

- `unshare -rn` stops at `write failed /proc/self/uid_map: Operation not permitted`.
- `bwrap --unshare-net` stops at `loopback: Failed RTM_NEWADDR: Operation not permitted`.

Both need an AppArmor profile or root. Workaround (D-061): the offline
build gate runs in the reference container with `docker run --network none`,
which works on the workstation. The owner's `spark` account is not in the
`docker` group; arm64 containers run on the workstation through qemu binfmt
(installed 2026-09-23, D-061). Changing the sysctl or adding an AppArmor
profile is a system change the owner has not made.

## RE-012: Pinned SGLang MiMo-V2 startup fails with a misleading processor error without torchcodec  (2026-09-22, status: worked-around)

`lmsysorg/sglang` nightly `0f6761b5` (arm64 digest `9e1fb4c3…`) on both
Sparks, MiMo-V2.6-Flash-RL. Startup, even for text-only use and with
`--enable-multimodal`, aborted in the tokenizer manager with
`No processor registered for architecture: ['MiMoV2ForCausalLM']`. The real
cause is earlier and swallowed: `multimodal/processors/mimo_v2.py` imports
`torchcodec`, which the image lacks, and SGLang's processor discovery logs and
skips modules that fail to import. Two boots were spent before the import
was tested directly. Workaround: a derived image adding hash-pinned
`torchcodec` 0.16.0 ([MiMo reference](experiments/mimo-reference/README.md));
the MiaAI recipe also installs it. The boot without `--enable-multimodal`
failed identically, so the engine builds this architecture's multimodal
processor regardless of that flag; dropping the flag with `torchcodec`
present was not tried. When a registry lookup reports "not
registered", import the module directly before debugging arguments.

## RE-011: Upstream GGML CUDA broadcast ops abort on strides above 2^32 elements  (2026-09-22, status: open)

stable-diffusion.cpp `c92d73c` built against upstream GGML `8846b79` (CUDA
13.0.3, `sm_121a`), GB10 `spark-b`, Qwen-Image-2.1 Q4_K_M at 2048²/40. All 40
denoising steps completed; the Wan VAE decode then failed
`GGML_ASSERT(s02 <= std::numeric_limits<uint32_t>::max())` in
`src/ggml-cuda/binbcast.cu:270` (exit 133). The same build matched the
patched fork's pixels exactly at 512² and 1024². The assert is unchanged in the
fork and in upstream master `179b60f` (2026-09-22), and upstream master still has
no CUDA `conv3d.cu`, which the fork adds with implicit-GEMM conv2d/conv3d.
Plausibly the fork's conv path never materializes the oversized intermediate
that upstream's im2col path passes to a broadcast op; not verified, and
upstream master was not executed. Impact: GGML CUDA elementwise/broadcast
kernels carry 32-bit stride limits, so very large activation or workspace
tensors (here a 38.3 GiB decode buffer) need chunking or different kernels.
Check tensor strides against these limits before planning GGML execution of
high-resolution image/video decoders. [Report](experiments/image-gguf/README.md).

## RE-010: Adding graph outputs changes logits with CUDA optimizations  (2026-09-22, status: open)

Pinned llama.cpp `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, GB10, driver
580.178.04, Gemma 4 26B A4B. A libllama patch that only appended copies of
each layer's routing IDs as graph outputs changed every logit row and 44 of
3,072 teacher-forced predictions with CUDA fusion and graphs enabled. It
changed nothing with both disabled, and the patched library matched the
official one bit-for-bit with the feature off. An eval callback that never requests data
also changed nothing. Extending a tensor's lifetime or adding outputs
changes the compute-buffer layout. A change in which fusions qualify is a
plausible explanation, but these controls disabled fusion and CUDA graphs
together: neither their individual roles nor the specific affected operation
was isolated.

Treat any graph-shape or lifetime change (instrumentation, extra outputs,
debug copies) as a potential numerical-plan change under CUDA optimizations.
Reference controls and jitLLM's own GGML integration must compare optimized
logits exactly before assuming an observation point is transparent. See the
[fused-routes experiment](experiments/fused-routes/README.md#rejected-design-routes-as-graph-outputs).

## RE-009: Pinned ExLlamaV3 compiles x86-only CPU helpers on Spark  (2026-09-22, status: worked-around)

ExLlamaV3 `6b84a21b6f1e5da3f291b9e1019061f0de788279`, Spark AArch64,
GCC 13.3.0, CUDA 13.0.88 and PyTorch 2.14.0+cu130. Import builds every
extension translation unit, including x86 CPU feature probes, CPU MoE and
CPU collectives. The unmodified build fails on `__builtin_cpu_supports` in
`avx512_target.cpp`; host spin waits also use `__builtin_ia32_pause`.

The [external reference patch](experiments/exl3-reference/arm-reference.patch)
returns false for x86 capability probes on ARM, makes unsupported CPU MoE
and CPU-reduction entry points fail explicitly, and uses the ARM `yield`
instruction for host spin waits. EXL3 GPU kernel bodies are unchanged.
This is a bounded **single-GPU reference build**, not a portable CPU backend
or validation of upstream CPU offload or tensor-parallel collectives.
The [baseline report](experiments/exl3-reference/README.md) records the build
and execution identities. Native jitLLM integration still adopts only its
audited operation closure; it does not inherit this whole external extension.

## RE-008: Extended reference runs do not always preserve exact top-1 predictions  (2026-09-21, status: open)

Pinned llama.cpp `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, GB10,
driver 580.178.04, CUDA fusion and graphs disabled. Two distinct limits surfaced
in the [full paging study](experiments/paging-feasibility/full-study.md):

- DeepSeek V4 Flash sequence snapshots reserialize identically after restore,
  but append-only continuation differs from live state on 16/1,536 top-1
  predictions (0/6/10 by turn). A same-host repeat reproduces those differences;
  restored predictions also match across the two Sparks. Raw/compressed KV and
  compressor state are present in the serializer. Cache compaction changes
  attention shapes, but neither a numerical explanation nor missing semantic
  state has been established. This is not the Gemma rollback failure in RE-007.
- Qwen3.8 Flash Next's identical untraced binary repeated on one host differs
  on 6/1,536 predictions (3/2/1). A restore probe differs on 10, including five
  before any restore, so restoration is not an isolated cause. Qwen also prunes
  the last prefill layer to output rows (`models/qwen4exp.cpp:400`); assuming
  every layer routes the whole input batch aborts capture. Record the actual
  final-layer output-only dependency, without reducing consumed input tokens.

No numerical tolerance is inferred from these counts. Capture defaults remain
strict; Qwen's explicit drift-recording mode labels failed equivalence, and
replay/locality analysis require a separate opt-in. Both large-model spill
returns use conservative recomputation in the study. Raw evidence remains in
external `paging/large-capture-2`, `deepseek-restore-probe-2`, `qwen-capture-1`,
`qwen-capture-3`, and `large-restore-probe-1`; their identities and comparisons
are retained in the study's aggregate evidence. Do not promote these reference
observations into a jitLLM numerical or restore-compatibility guarantee.

The [image-reference follow-up](experiments/image-reference/README.md)
(2026-09-22) also finds a short Gemma configuration difference with this
llama.cpp revision and its default CUDA fusion/graph settings: a 639-token
continuation restored from 627 saved tokens matches resident execution, but
fresh-context full prefill differs at 15 of 32 generated positions, starting
at zero-based position 17. Full-prefill controls before and after image
generation match each other, so image switching is not required to reproduce
the difference. This is a free-running token comparison, not a teacher-forced
logit diagnosis, and does not establish a common cause with the cases above.
Preserve the failed cross-mode comparison rather than silently declaring
cached and recomputed trajectories equivalent.

## RE-007: Gemma sequence snapshots lose SWA history needed after prompt rollback  (2026-09-21, status: worked-around)

On Spark GB10/driver 580.178.04 and pinned llama.cpp
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, the extended four-turn
Gemma restore probe changed **58/3,072** teacher-forced next-token argmax
predictions versus live full-SWA state (per turn: 0, 12, 19, 27). Ornith
changed **0/3,072**. CUDA fusion and graphs were disabled in both arms.
Each snapshot reserialized byte-for-byte identically after restoration into
a fresh context; that checks the stored bytes, not sufficient context coverage.
Raw negative evidence is `paging/restore-probe-1` on Spark, with the matched
live controls in `paging/small-capture-2`.

Pinned `src/llama-kv-cache.cpp:2080` drops cells outside the final SWA window
when serializing an individual sequence, even with full-SWA allocation.
Gemma uses standard SWA of 1,024 tokens. Its first snapshot ends at position
8,105 and retains SWA positions 7,082–8,105. The canonical next prompt shares
only 7,335 tokens: its first resumed query needs positions 6,312–7,335,
including **770 positions absent from the snapshot**. The next two returns
have the same missing-window count. Successful tail removal does not detect
this gap. Physical cache compaction also changes attention shapes
(`llama-kv-cache.cpp:1250`), but the missing dependencies alone invalidate
assuming equivalent restored execution; the prediction changes are not
classified as harmless numerical noise.

The native session and restore harnesses now check the earliest retained
position against the model's actual SWA window before reuse, including after
full-SWA restoration. They reset and recompute when coverage is insufficient,
following the conservative checkpoint-coverage principle in pinned
`tools/server/server-context.cpp:3297` and `:3349`. They have no older
checkpoint to restore. Replay uses captured recompute routes for Gemma
sequence-spill returns with prefix rollback, and rejects apparent reuse from
legacy normal-SWA captures by selecting the conservative recompute scenario.
Serialized byte identity and a successful short response do not establish
that a checkpoint supports arbitrary template rewrites. The earlier A→B→A
six-token rollback's matching output remains a narrow observation, not a
proof of complete retained-window coverage. Full-history snapshots would be
a different, larger spill contract and are not established by this probe.

## RE-006: Reading MoE routes through the llama.cpp callback changes the CUDA path  (2026-09-21, status: worked-around)

On the pinned reference `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`,
GB10/driver 580.178.04, a callback at `ffn_moe_topk-N` changed 51 of
3,072 teacher-forced next-token argmax predictions over four Gemma turns.
A repeated untraced control matched all 3,072 original control predictions;
the first traced difference occurred before any conversation reuse. The
short first-cut 118-prediction probe had matched, so it did not expose this.

The scheduler splits and synchronizes the graph at requested callback nodes.
The pinned CUDA implementation has a fused routing operation spanning the
selected top-k view, so this observation point changes the execution path.
Disabling both CUDA fusion and CUDA graphs in **both** traced and control
runs (`GGML_CUDA_DISABLE_FUSION=1`, `GGML_CUDA_DISABLE_GRAPHS=1`) restored
exact prediction equality in the extended retained and recomputed Gemma
captures. Those two settings were changed together; this does not isolate
their individual effects or prove bitwise-logit equivalence.

Keep instrumented-route experiments explicitly matched to their control
configuration, and keep their timing separate from the reference's normal
optimized path. See the
[paging-feasibility experiment](experiments/paging-feasibility/README.md).

**Fusion-preserving capture (2026-09-22).** Reading each layer's IDs at the
end of its gated-activation fusion group, before the down projection
consumes them, keeps upstream fusion and CUDA graphs: logits were
bit-identical to untraced runs on all 3,072 Gemma and 3,072 Ornith outputs,
using the unmodified image
([fused-routes experiment](experiments/fused-routes/README.md)). With fusion
and graphs off, the same read point reproduces the study's recorded routes
exactly. The optimized plan and the plan with both disabled select different
expert sets in 40–43% of token-layer rows, rising with depth. The study's
routes describe the plan with fusion and graphs disabled. New captures should
use the boundary read and re-check untraced equality per model/revision; the
recorded captures were not redone.

## RE-005: Spark perftest warmup option stalled an RDMA-CM sweep  (2026-09-21, status: worked-around)

Environment: both Sparks, kernel `7.0.0-1019-nvidia`, ConnectX firmware
`28.45.4028`, `rdma-core 50.0-2ubuntu0.2`, Ubuntu perftest
`24.01.0+0.38-1build2` (binary reports 6.20), RoCE v2 / MTU 1024.
An `ib_write_bw -d rocep1s0f1 -R -p 18700 -a -n 1000 --report_gbits
--perform_warm_up` server/client pair failed to finish within 90 seconds;
both remote timeouts returned 124. Removing the optional warmup flag
completed the sweep, and the subsequent repeated baseline passed. Duration
tests discard a one-second start/end margin instead.

This is an observed option-combination timeout, not an isolated root cause
or evidence that the DAC requires a reboot. The excluded pilot receipt and
logs remain in workstation `/tmp/jitllm-interconnect/host-sweep/`; the
[baseline report](experiments/interconnect/README.md) records the working
protocol. Also, `ib_write_bw --version` prints `Version: 6.20` but exits 1;
do not treat that informational exit as a failed transfer.

## RE-004: llama.cpp Gemma slot restore reports success but default SWA re-prefills  (2026-09-21, status: worked-around)

Environment: `spark-c4e2`, GB10, driver 580.178.04; pinned llama.cpp
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` in the CUDA 13.3 ARM64 container;
Gemma 4 26B A4B Unsloth UD-Q4_K_M. Full identities and the retained
[smoke harness](experiments/reference-setup/README.md) accompany the result.

With an 8192-token context, f16 K/V, one slot, batch/microbatch 512, and
default SWA retention, the slot save and fresh-process restore endpoints
both reported **627 tokens / 141,268,896 bytes**. A 639-token continuation
then re-evaluated **all 639 tokens**; the server logged a full prompt
re-processing fallback due to missing cache data. The equivalent resident
continuation evaluated 18 tokens after an internal checkpoint rollback.
The prompt was shorter than the 1024-token sliding window. Successful
serialization counters therefore do not establish useful continuation reuse.

The pinned [server source](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/tools/server/server-context.cpp)
uses SWA coverage thresholds and context checkpoints when deciding whether
the common prefix can be reused. `--swa-full` disables that SWA checkpoint
path. With it, two runs restored the 627-token prefix, evaluated only the
12-token extension, and matched all 32 resident-continuation output token
IDs. This is a measured workaround for this pinned configuration, not a
general fix for all hybrid models or proof of long-context coverage.

Cost: logged KV allocation grows from **460 MiB** (160 global + 300 SWA)
to **1760 MiB** (160 global + 1600 full-SWA) at this context. Keep the
normal/windowed and full-SWA reference configurations separate in the
A→B→A experiment and include their actual memory costs. To reproduce the
negative control, remove only `--swa-full` from an external copy of
`smoke.py`; its restore assertions must fail rather than report a pass from
the matching API counters. The two differing continuation token streams in
the negative control alone are not evidence of corrupted KV: they used
different prefill paths. No upstream fix is claimed.

Long-context follow-up: the [A→B→A report](experiments/reference-aba/README.md)
reproduces this behavior at context 32,768. Three default-SWA restore trials
reported 18,303 restored tokens but processed all 18,339 continuation input
tokens. Full-SWA restore reused 18,297 and processed 42, matching every
118-token output. The six-token difference between saved and reusable
counts is legitimate: Gemma's canonical chat template removes the empty
generation-only thinking marker from completed turns. Compare the actual
longest common token prefix, not just the save API's count. A separate
early/late notebook recall check passed. Also retain the measured decode
speed distinction (about 27–28 tokens/s live/recomputed full-SWA versus
46–47 after restore/default SWA); its cause was not isolated here.

## RE-003: nvme-cli 2.8 feature control requires --value, and zero has a different printed form  (2026-09-21, status: worked-around)

Environment: Spark, installed nvme-cli 2.8-1ubuntu0.1. During the bounded
interrupt-coalescing comparison, a command using `set-feature -f 8 -v 263`
set zero: `-v` is **verbosity**, not value, in this version; the value option
is `-V` or `--value`. Also, `get-feature` prints nonzero as
`Current value:0x00000107` but zero as `Current value:00000000`. A parser
requiring `0x` rejected zero, including during the attempted cleanup.

The hardware readback exposed the mismatch before any A/B measurement was
retained. The original value was restored with
`nvme set-feature /dev/nvme0 --feature-id=8 --value=263`, then independently
read back as `0x107`. The comparison was rerun in full with long options,
both output forms accepted, and verified restoration. Do not trust mocks
based on remembered short flags for a device-control command: check the
installed tool's help and read back the actual state. The retained
[coalescing harness](experiments/io-path/coalescing.py) records the original
value before changes and bounds/restores its temporary setting.

## RE-002: cuFile compatibility mode rejects a descriptor opened with O_NOFOLLOW  (2026-09-21, status: worked-around)

Environment: `spark`, GB10, driver 580.178.04, installed libcufile package
1.15.1.6-1 (reported API version 2.12), Linux 7.0.0-1019-nvidia, ext4.
The I/O spike opened its private regular file with
`O_RDONLY | O_DIRECT | O_CLOEXEC | O_NOFOLLOW`; `cuFileHandleRegister`
failed with **5019 / CU_FILE_INVALID_FILE_OPEN_FLAG**. The library log
reported unsupported open flags `229376`. Native direct reads on the same
descriptor worked.

The comparison harness keeps its validated original descriptor open,
reopens `/proc/self/fd/<fd>` with `O_RDONLY | O_DIRECT | O_CLOEXEC`, and
checks device/inode identity before registering that new descriptor with
cuFile. This preserves file identity without following the original user
pathname again. Registration and subsequent GPU-verified reads then passed.
Do not respond by removing path protections from the original file open.
See the [I/O experiment](experiments/io-path/README.md) and its retained
`main.cc`. This workaround is confined to the comparison backend; the
native direct-file candidate does not need it.

## RE-001: CUDA 13.0 NVCC rejects C++23 even with a C++23-capable Clang host  (2026-09-21, status: worked-around)

Environment: x86-64 Ubuntu 24.04, NVCC V13.0.88, Ubuntu Clang 18.1.3;
AArch64 cross target using the Spark DGX OS 7.6.0 snapshot, GB10 `sm_121`.
Passing `-std=c++23` to NVCC fails before compilation with
`Value 'c++23' is not defined for option 'std'`. Clang's support for C++23
does not establish NVCC front-end support. This corrects the expectation
in D-010's original context; its allowance for a separate CUDA dialect
already covers the fix.

The older-toolkit comparison used C++23 `.cc` files and C++20 `.cu`
files and passed cross-built and native Spark runs. The selected D-032
pin instead uses Toolkit 13.4.2 / NVCC 13.4.92, which accepts C++23;
both execution paths passed with an explicit C++23 host/device feature
probe. M1 should use C++23 throughout for that selected SDK. Do not
silently fall back to the installed 13.0 compiler and assume the same
dialect support. See the [reproduction and pins](experiments/toolchain-smoke/README.md)
and [NVCC 13.4 options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
