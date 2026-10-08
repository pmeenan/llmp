<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen paired shared-expert MXFP8 screen — 2026-10-04

Keep the existing shared-expert chain. A custom paired gate/up kernel with
SwiGLU reduces the representative complete operator's latency **28.64%**,
but the paid C4 verify-wave screen is neutral: **+0.0907% latency** against
mean controls whose medians move **−0.4472%**. All measured complete outputs,
wave token histories, target rows and initialized target/drafter states are
exact. The private candidate is rejected; no production kernel, graph rewrite,
setting, default or calibration changes land.

## Actual chain and arithmetic

The native NVFP4 target's shared expert builds two MXFP8 products at
K=2,560/N=640, then `ggml_swiglu_split(gate, up)` before its down product
(`qwen38_graph.cc`). Its twelve-column production launch takes four output
rows per warp and four warps per CTA: 40 CTAs per matrix, then one GLU launch.
The custom kernel instead takes two rows from each family per warp, four
warps per CTA and 80 CTAs, writing the final F32 GLU output directly.
Both distinct weight matrices are still read. The original per-output
sixteen-FMA order, scale FMA, XOR reductions and PDL synchronization remain.

This removes intermediate writes/reads and launches but changes family
grouping too. The original products already share input across four rows;
net input-traffic reduction, occupancy and sole-cause attribution are unproven.

The original GGML CUDA unary translation unit uses `-use_fast_math`, while
llmpalooza's MXFP8 product translation unit does not. Initially reusing the same
SiLU C++ expression in the precise product TU differs in **126,322 of
614,400** output values, with zero own-repeat differences or nonfinite values.
That first model-free screen stops before any timing; its failed receipt is
retained. The correction reproduces the actual pinned unary compile's five
PTX operations only in the GLU epilogue:

```text
mul.ftz.f32          exponent, gate, -log2(e)
ex2.approx.ftz.f32   exponential, exponent
add.ftz.f32          denominator, exponential, 1
div.approx.ftz.f32  silu, gate, denominator
mul.ftz.f32          result, silu, up
```

The first multiplier is the original F32 bit pattern `0xBFB8AA3B`.
A probe built with the actual locked unary compiler command records these
operations. The product TU retains its precise compilation; neither product
arithmetic nor a global fast-math flag changes. Complete output equality
against the actual compiled original chain qualifies this bounded correction.

## One complete operator

Spark A (`spark-c4e2`), NVIDIA GB10, 48 SMs, compute capability 12.1,
24-MiB L2, SDK `aarch64-e0a0c85c42806fb1`, pinned CUDA toolkit 13.4.92.
A post-run inventory confirms driver 580.178.04. The private operator fixture calls
actual `RunMxfp8MulMatVec(up)`, `RunMxfp8MulMatVec(gate)` and GGML `SwiGlu`
versus the fused kernel, at **C12/K2560/N640** with twelve padding floats per
input column. Padding contains NaN sentinels and is not read. Generated finite
MXFP8 codes/scales and F32 inputs are identical between paths.

There are 80 distinct physical gate/up weight pairs, rotated once per captured
cycle. Codes plus scales occupy **3,379,200 bytes per pair**, or
**270,336,000 bytes / 257.8125 MiB** in all, larger than L2. Runtime layout
records 123,456 input bytes, 61,440 original intermediate bytes, zero declared
operator workspace and 277,930,048 guarded allocation bytes. The inherited
fixture additionally allocates 32 MiB of unused scratch; graph-driver
allocations are separate. These are allocation/layout counts, not a measured
peak-memory claim.

All **614,400 outputs** (80 × 640 × 12) compare bit for bit with the actual
original chain and a fused repeat before timing; differences/nonfinite values
are zero. All 256-byte front/back guards remain intact, and all 7,680 final
outputs agree after each timed arm. These are complete byte comparisons.

| Chronological arm | Complete operator ms | Kernels per call | Captured nodes |
| --- | ---: | ---: | ---: |
| Original before | 0.03566479 | 3 | 240 |
| Paired | 0.02533372 | 1 | 80 |
| Original after | 0.03533811 | 3 | 240 |

Each graph captures 80 complete calls. CUDA-event timing uses three batches,
with the requested 4,096 repetitions rounded to **4,160 actual calls** per
batch and normalized by that actual count; the table reports their median.
A 20-ms warmup, allocation, host validation, capture and final output checks
are outside the operator clock. All two-product-plus-GLU GPU work is paid
in each original call, and all fused work in each candidate call.
Candidate latency is `(paired / mean(original medians) − 1) × 100 =
−28.640314%`; original medians move `−0.915987%`. This is an isolated captured
operator result, not a model or serving gain.

The final operator child exits zero and is reaped after completion-aware
fixture retirement. Strong 105-GiB admission/retirement probes report
116.493/116.436 GiB and clear GPU/container/native-model inventories.

## One paid C4 wave

The private composition runs **after the existing wave joins**. Four native
three-row verify slots first form the ordinary C12 gate/up products. A bounded
post-join rewrite replaces those two products and their four split-view
SwiGLUs with one full C12 fused output and four output views. Both original
F32 concat trees remain paid: the unused second packed input is an explicit,
validated sixth dependency, retained through topological planning and
activation lifetimes. Downstream per-slot operations retain their lanes;
the fused operation is on the context stream.

The rewrite checks membership, shape/input order, GLU parameters, split offsets
and exclusive consumers, authenticating eight original per-slot products and
four GLUs against their original implementations. Ragged/other phases retain
the original graph. The private rewrite assumes GB10; production device
fallback and broader contract/refusal qualification remain unimplemented.

Each fresh benchmark process uses four frozen short fast-swap prompts,
96 generated tokens per slot, context 16,384, chunk 4,096, four slots and wave
lanes/graphs on. All arms request runner draft capacity three and vocabulary
65,536 but execute **fixed shared depth two**, selected physical head
**47,172 rows**, F32 head inputs, no adaptive depth and window zero. The
production joined-drafting limit stays two; C4 drafts stay serial. Prefill
uses the ordinary benchmark chunk path (`runtime_prefill = false`).

| Chronological arm | C4 verify median ms | Draft median ms | Whole process s |
| --- | ---: | ---: | ---: |
| Original before | 106.655 | 21.915 | 31.972479 |
| Paired | 106.513 | 21.489 | 31.381338 |
| Original after | 106.178 | 21.505 | 31.043243 |

Verify latency changes `(106.513 / mean(106.655, 106.178) − 1) × 100 =
+0.090681%`; controls move `−0.447236%`. Medians include waves with all four
slots active. The verify clock pays `VerifyWave` input preparation, both
concat trees, planning, activation placement, capture/replay, execution and
full logit copies. Caller work-vector construction, subsequent row hashing,
prefill, draft and final state reads are outside it. Whole process wall also
includes load, prefill and checks; it is not served-token throughput.

Every arm completes exactly 96 generated tokens per slot, **43 waves**,
two verify captures and 38 replays. All four per-slot prompt-plus-generated
token-history SHA arrays, complete F32 verify-row SHA arrays and SHA arrays
read from actual initialized target-plus-drafter state match across all
three arms. The candidate reports a **maximum of 48 replaced target layers** and
**39 waves with fusion**; other geometries fall back. Composer statistics
are part of paid planning; the benchmark reads them after the verify clock,
including replays, without a launch-phase diagnostic counter or print.

All three children exit zero and are reaped without cleanup errors. Six strong
105-GiB gates pass with clear GPU/container/native-model probes and minimum
availability **115.790 GiB**. Installed GPU-supervised
`qwen-shared-mxfp8-wave-screen1` ends zero after approximately 96 s and is
waited on. The neutral result ends qualification: no HTTP, context/swap/quality
matrix or full suite follows under the M3 experimental override. Exact native
controls establish neither cross-engine parity nor a quality bound.

## Provenance and retained evidence

The operator and candidate sources start at `ab90044`; the clean report
worktree starts at `4c06b47`. Later calibration-only changes do not alter
these kernels. The original benchmark source exactly matches `ab90044`.
Its 487-file source/build-input inventory is authenticated; the candidate
inventory adds the actual benchmark source and differs in eight source
files. Benchmark changes only emit the post-clock fusion witness. The
operator's earlier two-file source and all linked archives are frozen
separately from the later nine-file private wave prototype.

| Item | SHA-256 |
| --- | --- |
| Operator source inventory | `0515327f28f8f53bcb4c15f2390d3614de1844e8b233911563648839d9638fef` |
| Operator executable | `6891d150966224a39e9ed38876fd47bef7f4b5ff87263591fb60572ece787a2e` |
| Operator harness source | `ef02931dfd442b326e37327f6da49006a772b132406ea4e4baaa53d3eec3e85a` |
| Operator archive/binary identity inventory | `8379b4c980f90c5633f01fb0d8f7882d994f5c805c170d8c32c63c06e5ac87ca` |
| Operator compile/link command receipt | `559ca27a6cf89e96fdcdbb742e79b2954ecfc3dbf5adede316efc1b0e7ade24a` |
| Operator controller | `9d8702e0ee6ddba69f6768882ea1a2cf5a9e31fd6446a2c889c4a742862bfa0a` |
| Exact operator aggregate | `2eacc0443c836fdf76b47cea47ed07d1cb4d5401111450a36c30dea52cb8a27b` |
| Failed precision aggregate, no timing | `2615d4a6951676d1557eae0c19c99ef0b640fd7bbfec845b9ecf8b9a1f9d1ad9` |
| Actual unary probe PTX | `ccedbd0d82e768f579eecf22c2e0a965040ba90ca1a7b55439e044bc045d18dc` |
| Probe compiler-command receipt | `1b83aa4b45aad11330f6f7fa71f7c1ba33349d7219dd5a710df8f743a22b5ae3` |
| Wave control source inventory | `c4332305fc048aa9b340c09751e460c2a1651e1cae188e61b2f57d534a368fe2` |
| Wave candidate source inventory | `effccbb77f3cf3132c22bf82f2faaf993c821c7ff1a6bd58e11d6d04c11181af` |
| Wave control executable | `b238b5be4b30318ae7ce5f8656fd1f10262e10bbaee934f641c1be6c9bef56ac` |
| Wave candidate executable | `325af2b5560be4686ffd74df97445a72bd6cad0558b9aea027edbbcdda8ad295` |
| Wave private patch | `ff7411d819a3a06a20fb6a9005e9e50bf6ce69fbdf7481a3293bbb30596e91ba` |
| Wave controller | `269b11d2d1e4d60c42ed7c88dd192e7170bcdbafaf1c9c4dca37ea0bad800fd6` |
| Wave aggregate | `36304d99c8ae796e6c70848083823abd88af993e259bba228597c8519077f686` |
| Frozen short prompt JSON | `c697236c56a09b0a3f2550f7514b3e4d826e1d14a96b4d1c79e3bd33a3a6f859` |
| Original benchmark source | `e7c95af532772e89c24d5315b36751e8ff8fab82bc10c2156a184aad3c700e72` |

Target and drafter artifacts are respectively
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`
and `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`;
these IDs are their manifest hashes. Their index hashes are
`5b65dcce8169374e638264d7ebdcb5cca517234b3dec26f1272a5f4f9c0e4c89`
and `5cd45fc5354ab224d281c2416027f224c61e32e2ae0acf8e7580d063fc274d99`.
The checkpoint/tokenizer path is
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`; tokenizer SHA is
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.

The frozen wave controller names these assets rather than hashing them.
Two **post-retirement supplemental** reads at 13:17:09 and 13:17:29 UTC
match manifest/index/tokenizer hashes from the preceding qualified alignment
receipt and match each other. Embedded target/drafter config metadata and
selected-head ID metadata are newly recorded and unchanged. The first file
is named `wave-assets-during.json`, but its timestamp is after the model
collection ended; it is not a pre-model or during-model authentication.
The second receipt also asserts every arm's actual selected-head/effective
metadata. No full weight payload is rehashed. Supplemental receipt SHAs are
`dcf3e74764a6daa99f6538df5419e5791e499274253e127bcfbcd44ef5230ef9`
and `3038229fb87fd06aac85a0f95c0c9a5c0d9eeec8bd94bbebe42422d0d5652448`.

Preparation failures (host-harness tidy, declaration order and const-pointer
lookup) are corrected before inference and retained without performance claims.
The final minimal benchmark build takes 14.925 s and is supervised/waited zero;
changed source is SDK-formatted, and the final operator harness passes SDK tidy.
All model/operator attempts carry retirement receipts. The compiler-only PTX
probe is supervised/waited but has no separate strong model-admission receipt.

Raw controllers, source snapshots, patches, commands, linked-archive pins,
frozen binaries, failed/successful logs, per-arm specs and retirement receipts
remain outside Git at `spark:~/scratch/qwen-shared-mxfp8/` and locally
`/home/pmeenan/scratch/llmp-m3-qwen-shared-mxfp8-2026-10-04/`.
The owned `mxg01` candidate and earlier `rdal1`/operator controls remain
immutable. Only this aggregate and an optimization-inventory row land.
