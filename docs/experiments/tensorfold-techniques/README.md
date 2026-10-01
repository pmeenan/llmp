<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorFold's techniques in jitLLM (M3): Qwen3.8's decode gap first

TensorFold decodes Qwen3.8 Flash Next about 1.3–1.45× faster than jitLLM
([baselines](../fast-swap/baselines.md#qwen38-flash-next-tensorfold-mlx-4-bit-cross-quantization)):
36.9–37.6 tok/s plain and 55.1–55.2 with its MTP drafts, against jitLLM's
25.7–25.9 and 42.5 / 39.1 (MTP depth 2,
[qwen38-mtp](../qwen38-mtp/README.md)). The owner asked to adopt its
techniques wherever they measurably help, speed before bit exactness
(D-085's note), within each model's format: a change of precision is the
separate quality/performance-modes item. This study profiles both engines,
estimates the format's contribution from checkpoint sizes, adopts measured
engine improvements that transfer, and records the remaining attribution limits.

This study read and ran TensorFold at `ashhart/TensorFold@beddbb7b`, in
`nvcr.io/nvidia/pytorch:26.07-py3` as the baseline ran it, on
`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa80` (MLX affine 4-bit, groups
of 32). The pin has since moved to `71377a53` (0.3.6.2), whose decode is
within 4% of it (plain 37.5, MTP 55.4–56.1 tok/s, the owner's tip runs on
`spark-b`), so the comparisons below stand. Its code is MIT
([licensing](../../licensing.md#tensorfold)); nothing of it is copied
here, only ideas.

## Checkpoint sizes suggest a substantial format contribution

**Modeled bytes per decode token**, computed from the two checkpoints'
safetensors headers (`spark`, 2026-09-28): routed experts count 10 of 512 a
layer, the token table one row, the n-gram table its 16 rows, everything
else whole.

| Part | jitLLM (Mia's NVFP4, MXFP8, BF16) | TensorFold (MLX 4-bit g32: 4 bits, a BF16 scale and bias per 32) |
| --- | ---: | ---: |
| Routed experts, 10 × 48 layers | 1,327 MB (NVFP4) | 1,475 MB |
| Shared expert, 48 | 243 MB (MXFP8) | 147 MB |
| Router and shared gate, 48 | 126 MB (BF16) | 126 MB (BF16) |
| Hyper-connection products, 97 mixes | 1,279 MB (BF16) | 400 MB |
| Gated DeltaNet projections, 36 | 2,150 MB (MXFP8) | 1,303 MB |
| QSA projections, 12 | 637 MB (MXFP8) | 386 MB |
| n-gram layer's key and value | 66 MB (BF16) | 21 MB |
| Head, 248,320 × 2,560 | 1,271 MB (BF16) | 397 MB |
| Gated DeltaNet state, read and written | 453 MB before this study, 226 MB after | 226 MB (double-buffered) |
| **Total** | **7.55 GB before, 7.33 GB after** | **4.48 GB** |

This full-read model assigns jitLLM 1.64–1.69× TensorFold's bytes. The MXFP8 dense layers (8.25
bits a weight against 5) and the BF16 hyper-connection products and head
(16 against 5) make up all of it; Mia's NVFP4 experts read *fewer* bytes
than MLX's (4.5 bits against 5).

**Effective bandwidth estimates.** jitLLM's plain step took 37.4 ms on the device
before this study ([qwen38-mtp](../qwen38-mtp/README.md#performance-and-memory)):
7.55 GB at 202 GB/s. TensorFold's 36.9–37.6 tok/s (the baseline) is 165–168
GB/s on its 4.48 GB, and its 39.4–39.8 tok/s in this study's session
(`"draft": false`, 32 and 256 tokens of `prose`) 177–178 GB/s. Applying
jitLLM's modeled 202 GB/s to TensorFold's 4.48 GB predicts 22.2 ms, 45 tok/s.
This counterfactual makes format a plausible substantial explanation of the
solo gap. Checkpoint bytes are not measured DRAM traffic, however, and
TensorFold's kernel profile was unavailable. Cache reuse, numerical products,
state and scheduling remain confounded; these estimates establish neither a
complete format attribution nor an engine-for-engine advantage. Concurrent
sharing also changes how often common weights need to be read.

**Where jitLLM's step went** (Nsight Systems, the decode graph's nodes
traced, `spark-b`, `jitllm_swap_pairs --a qwen38 --bench 16`, 8 steps;
38.1 ms busy of a 38.4 ms step under tracing):

| Kernel class | ms a step | Bytes | GB/s |
| --- | ---: | ---: | ---: |
| BF16 head (GGML MMVF) | 5.28 | 1,271 MB | 241 |
| MXFP8 vector products (`Mxfp8Gemv`) | 14.9 | 3,030 MB | 203 |
| – Gated DeltaNet's output (2,560 × 6,144), 36 | 3.6 (98–102 us each) | 16.2 MB each | 159–165 |
| – QSA's output, the same shape, 12 | 0.89 (73–75 us each) | 16.2 MB each | 217–222 |
| Routed experts (`Gemv`, gate and up / down) | 4.19 / 2.12 | 885 / 442 MB | 211 / 208 |
| Hyper-connection products (cuBLAS gemv and split-K reduce) | 6.15 | 1,270 MB | 206 |
| `HcPrepKernel` (one block a token) | 1.37 | – | 97 launches, 13–18 us each |
| The recurrence and its state's `set_rows` copy | 0.61 + 0.28 | 453 MB | – |
| The one-row short convolution (GGML's `concat`, `ssm_conv`, `silu`, two L2 norms) | about 0.6 | – | – |
| Gaps between kernels | 0.58 | – | – |

The same kernel reads the same shape at 217–222 GB/s in QSA's layers and
159–165 in Gated DeltaNet's. Alone and cold (a scratch microbenchmark, a
512 MB ring of weight copies, `spark-b`), `Mxfp8Gemv` reads that shape at
221 GB/s and Qwen3.8's other MXFP8 shapes at 222–229 (the 1.3–1.7 MB ones
at 140–163, launch-bound). The difference was the state: each Gated
DeltaNet step wrote its 3.1 MB state into the node's output and `set_rows`
copied it back (another 3.1 MB read and written), dirty lines L2 wrote back
while the next products streamed their weights. TensorFold double-buffers
the state (read one, write the other) and copies nothing.

**What TensorFold does differently** (its source at `beddbb7b`,
`families/qwen4_exp/cuda/` and `cuda/kernels/`):

- *Layouts made at load for its kernels:* the 4-bit words regrouped into
  64-column tiles, each group's block contiguous, or into MMA fragment
  order; each expert group's scales and biases beside its codes; the shared
  expert as expert 513 of the grouped expert kernel; the router's rows and
  the shared gate one matrix; Gated DeltaNet's q, k, v, z, b and a one
  matrix, QSA's q|gate, k, v and indexer projections one.
- *Decode products on tensor cores*, even for one row (`mma.sync` over
  16-row tiles), a 4-stage `cp.async` pipeline, split-K from the shape alone
  (at least 160 blocks) reduced in a thread-block cluster in slice order.
- *Fused hyper-connections:* the RMS norm inside the down product, the
  sigmoid mix inside the up product; 3 kernels a mix.
- *The recurrent state double-buffered;* verify rows committed by a replay.
- *The n-gram rows* from the host page cache (a memory map), copied each
  step.
- *CUDA graphs* for decode, verify windows and draft steps; about 17
  kernels a layer (from its forward pass), against jitLLM's 41 (34 after
  this study).
- *Its drafter:* its MTP layer over a 79,591-token draft vocabulary, 1 to
  6 drafts a round, a chain cut before a draft whose probability is under
  0.3; exact verification by row-invariant kernels.

The 4-bit dense layers and head are the format; the rest is engine, and
what transfers within Mia's format is below.

## Adopted

Each kept only where end-to-end decode moved. Stage 1 is the first four
rows (with the adaptive window's plumbing, below), stage 2 adds PDL, and
the final build keeps the BF16 vector kernel to one row.

| Technique (TensorFold's, or its lesson) | jitLLM change | Effect |
| --- | --- | --- |
| The recurrent state double-buffered | `jitllm.gdn.step`: the fast graph's gated delta rule up to 16 rows writes the new state over the old in place (the columns kernel's arithmetic, so the same state and output bit for bit); no state rows in the output, no `set_rows` copy. A verify's reads the state and writes none (the commit replays the kept rows) | 226 MB a token less; the output product after it 89 us instead of 98–102; `set_rows` (0.28 ms) gone |
| Enough blocks, not one | `jitllm.hc.prep` up to 8 tokens as a cluster of 8 blocks a token (`HcPrepClusterKernel`), its sums through distributed shared memory in block order, the second half's weights prefetched | 11.2 us instead of 13–18 a mix: −0.29 ms |
| Its own vector kernels for the mixes | `GemvBf16`: the fast graph's hyper-connection products at one row (a decode step; wider, cuBLAS, below) on a jitLLM BF16 vector kernel (a warp a row for the up product, 128 threads a row for the down one, every weight load issued before the arithmetic) instead of cuBLAS's gemv and split-K reduce | 5.73 ms instead of 6.15 |
| About 17 kernels a layer | the one-row short convolution on the fused `jitllm.gdn.conv` (the reference fusion's arithmetic), its history kept by `jitllm.gdn.history` reading the old history | 7 kernels a layer to 3: −0.4 ms, and fewer gaps (0.21 ms a step instead of 0.58) |
| Enough bytes in flight (stage 2) | programmatic dependent launch for `Mxfp8Gemv`, `GemvBf16` and the routed experts' `Gemv`, each prefetching its first weights into L2 before waiting for the kernel before; the small kernels before them trigger early | +1.6% end to end (below) |

A decode step went from 1,957 kernels to 1,644, and from 38.1 to 36.3 ms
busy under tracing (stage 1, `nsys`).

**Plain decode** (`jitllm_swap_pairs --a qwen38 --bench 128`, the runtime
wake, a request's lease, graphs; `spark-b`, no other GPU process, each
the mean of 3 passes of 128 steps):

| Build | Session 1 (15:28–16:30) | Session 2 (16:48–16:58, alternating) | Session 3 (18:30–18:40, alternating, rebased on `0560a8c`) |
| --- | ---: | ---: | ---: |
| main (`d7f0318`) | 25.79 tok/s (37.60 ms on the device) | 25.20 (38.32) | 25.83 (37.37) |
| stage 1 | 27.29 / 27.30 (35.48 / 35.45) | 26.81 (35.92) | – |
| stage 2 (+ PDL) | – | 27.26 / 27.23 (35.38 / 35.50) | – |
| final (stage 2, the BF16 vector kernel at one row only) | – | – | 27.39 / 27.36 (35.21 / 35.22) |

**Plain decode: +5.4–8.1% (2.0–3.0 ms a step)** against main in the same
session. The review's repeat (`spark-b`, alternating, the final build
rebased on `5e6794c`): main 25.88 / 25.82 tok/s (37.29 / 37.47 ms), final
27.29 / 27.45 (35.23 / 35.25), +5.4–6.3%. On `spark` the speculation harness's plain decode gave
25.49 / 26.06 → 26.94–27.30 / 27.52–27.56 tok/s (`prose` / `code`, stage
1). That is 0.73× TensorFold's baseline 36.9–37.6, against 0.69× before:
the remaining format contribution has not been isolated.

## Speculation: the adaptive window

The MTP draft's argmax now also gives the draft's softmax probability over
the draft head's rows (`jitllm.argmax` with probabilities, one more pass
over the row), which `Qwen38Runner::Draft` returns beside the drafts, and
`jitllm_qwen38_spec --window P` verifies the first draft and each later one
only while its probability is at least P, TensorFold's rule at its 0.3.
`spark-b`, stage 2, the greedy check (256 tokens of `prose` and `code`,
the six chat prompts' 32), one run each, 16:58–17:10:

| Depth, window | `prose` tok/s (acceptance) | `code` tok/s (acceptance) | A step, ms (`prose`) |
| --- | ---: | ---: | ---: |
| 2, off (the default) | 39.53 (0.581) | 40.49 (0.607) | 54.7 |
| 2, 0.3 | 39.03 (0.615) | 41.41 (0.638) | 54.9 |
| 3, 0.3 | 37.16 (0.533) | 41.33 (0.570) | 63.5 |
| 4, 0.3 | 36.95 (0.487) | 39.83 (0.525) | 67.7 |
| 3, off | 37.68 (0.463) | 42.64 (0.549) | 63.2 |
| main's build, 2, off | 41.02 (0.631) | 38.43 (0.539) | 55.0 |

**Not adopted as a default.** The window trims about one draft in
fifteen at depth 2 and moves the rate within run-to-run spread (−1.3% and
+2.3%); deeper windows lose, because every draft pass still runs (2.5–3 ms
each) and each verify row costs about 5 ms of its own experts while the
third and fourth positions accept 24–44%. The drafter's probability is
kept (it costs one pass over 65,536 logits) and the harness option stays
for a later drafter whose passes can stop early. Speculative decode itself
did not speed up with the plain step at first: a step stayed about 54 ms
(the verify ~47.7 ms, the draft ~6.2) in both builds, and the rate moved
with acceptance, which the changed kernels move token by token (`prose`
0.631 → 0.581, `code` 0.539 → 0.607). The verify's profile (`nsys`, 17
three-row verifies, `spark-b`) showed why: the verify graph fell from 47.1
to 45.9 ms, but the BF16 vector kernel for the mixes' products took 7.6 ms
at three rows against cuBLAS's 6.2 (its down product's blocks each read x
three times). So the vector kernel now takes one row only
(`kGemvBf16FastColumns`), and a verify's mixes stay on cuBLAS. The final
build against main in the same session (`spark-b`, the greedy check, two
speculative repeats each):

| | main | Final |
| --- | ---: | ---: |
| A step: draft / verify / all, `prose`, ms | 6.03 / 47.48 / 53.98 | 5.98 / 46.22 / 52.68 |
| `prose` tok/s (acceptance) | 41.03, 41.81 (0.631) | 40.01, 40.68 (0.571) |
| `code` tok/s (acceptance) | 38.63, 38.34 (0.539) | 40.84, 40.77 (0.576) |
| Plain, `prose` / `code` | 25.52 / 25.80 | 26.74 / 27.32 |

A speculative step is 2.4% shorter; the rates move with acceptance, which
the changed arithmetic shifts token by token (−10% on `prose`, +7% on
`code` here), so speculative decode is about where it was: 1.49–1.50× the
final build's plain rate, against main's 1.49–1.64×.

## Considered, not adopted

- **Kernel-first layouts at import.** The MXFP8 vector product streams the
  row-major artifact layout at 221–229 GB/s alone, the experts' layout is
  the prefill's grouped GEMM's (a second copy would cost 68 GB), and the
  GGUF path's vector kernel already reads 230–245 GB/s alone
  ([dsv4-decode](../dsv4-decode/README.md)). No import layout change had a
  measured gap to close; TensorFold's regrouping fixes a gap (107–130 to
  200–220 GB/s, creator-reported) its MLX layout had and Mia's does not.
- **Tensor-core decode products.** At one row they have the same modeled
  weight bytes. Native vector-kernel measurements suggest a memory-bound
  path; TensorFold's modeled per-byte rate does not isolate a kernel comparison.
- **Stacked projections** (q, k, v, z, b, a in one launch). DeepSeek's
  grouped multi-matrix launch gained nothing
  ([dsv4-decode](../dsv4-decode/README.md)); with PDL the launches between
  products overlap instead.
- **Partial-vocabulary draft heads.** Qwen3.8's draft head already reads
  65,536 rows ([qwen38-mtp](../qwen38-mtp/README.md)). DSpark's reads the
  target's whole Q4_K head (298 MB, about 1.3 ms of a 7.4–8.6 ms draft
  block): 65,536 rows would save about 0.65 ms of an 88 ms step (0.7%,
  computed), inside the runs' spread, at some cost in acceptance.
- **Draft trees.** Both drafters are chains; a tree's verify would need a
  recurrent state per branch in Qwen3.8's Gated DeltaNet layers and a
  compressor state per branch in DeepSeek's CSA and HCA, and neither has
  a tree-shaped verify plan. Not in this slice.
- **Per-request drafter choice.** Each model has one drafter here; the
  choice is "speculate or not", which belongs to the runtime's request
  path (m3rt).
- **Row-invariant exact speculation** (TensorFold's exactness). The owner's
  rule is speed before bit exactness; DeepSeek keeps D-092's row-invariant
  verify as `--exact on`.
- **A quantized draft head** (MXFP8 or NVFP4 rows of the BF16 head, about
  1.2 ms a step at 65,536 rows, [qwen38-mtp](../qwen38-mtp/README.md)): it
  changes only acceptance, but needs a cataloged extent of its own; left
  as a lead.

## DeepSeek V4 Flash

None of the adopted kernels is on DeepSeek's path (its hyper-connections,
products and routing are `dsv4_fast.cu`'s, already PDL-launched with L2
prefetch). Its bytes a token, computed the same way from the GGUF's
headers: **8.79 GB** (routed experts 2.10 GB, the attention's Q8_0
projections 4.6 GB, the Q4_K head 0.30 GB). At its 44.85 ms step on the
device that is 196 GB/s overall. The "200–210 GB/s in the model" of
[dsv4-decode](../dsv4-decode/README.md) divides bytes by kernel durations
that, under PDL, include each kernel's wait for the one before (the
kernels' busy time, 47.2 ms, exceeds the step's 45.7 under tracing); on
the step's span the products read about 215–220 GB/s (estimated: 8.6 GB in
the ~40 ms the products take), and the rest to the 230–245 alone is not
isolated. Its hyper-connection prep (`HcPreKernel`, one block a token,
0.82 ms a step) is the same shape of lever as Qwen3.8's (13–18 us to 11),
worth about 1%, and was not taken. Decode before and after (`spark-b`,
`jitllm_swap_pairs --a dsv4 --bench 64`, a request's lease, graphs, the
same session): main's build 21.99 tok/s (45.16 ms on the device), this
study's 22.15 (44.83): unchanged within the runs' spread, as expected;
DSpark was not rerun (its path is unchanged).

## Page migration and compaction (TensorFold item 6)

`/proc/vmstat` sampled every 0.5 s on `spark-b` (`vmstat.sh`, compaction
and migration counters, `MemAvailable`):

| Window | Pages migrated | Compaction |
| --- | ---: | --- |
| Qwen3.8 and DeepSeek loaded, two A→B→A cycles with eviction, decode between (165 s; 122 s in a second run) | 0 | no stall, no scan |
| After a load: warm-up, 3 × 128-step decode passes, then a swap-pairs run (208 s) | 0 | none |
| A 75 GB load while another agent's test suite (11–17 GPU processes) allocated beside it | 151,227 (113,000 in 5 s) | 326,000 pages isolated, 4.1 M scanned; the load read at 1.10 GB/s instead of 13.3 |
| A decode bench whose run another agent's 100 GB process started beside (the OOM killer ended it) | 731,160 | 291,801 direct-compaction stalls |

So on GB10 our runs are not disturbed by migration or compaction when
memory is not overcommitted: jitLLM's direct reads keep the page cache out
of it, and its extents are allocated once. The bursts TensorFold reports
come with memory pressure, which the memory gate exists to prevent; the
two contaminated windows are where two agents' processes met on one
Spark.

## Correctness

The owner's coarse rule (D-085's note). **The near-tie bound, recorded
before the new build was compared with the oracle** (16:22, from main's
build only): the top-1 to top-2 margin's move between main's two paths on
the oracle's tokens (`jitllm_qwen38_exec` as run and `--stepwise`) has
p50 0.23, p95 0.99, p99 1.94, maximum 2.17 over 192 steps; **B = 2.0** (the
p99 on the oracle's 0.125 grid; qwen38-native's p95 rule gave 1.0).

| Check | main (`d7f0318`) | Final build |
| --- | --- | --- |
| Greedy, teacher-forced, 192 steps (bound B) | 180 equal, 12 near-ties, 0 failures | 180 equal, 12 near-ties (oracle margins 0.0–1.0 and `french` step 3's 2.0), 0 failures |
| \|dlogprob\| RMS against the oracle, per prompt | 0.50–1.31 | 0.52–1.28 |
| Margin move against main's stepwise run | – | p95 0.76, p99 1.38 |
| Perplexity in 8-row chunks (decode's kernels), 3,557 positions | 14.4535 (−1.4%), top-1 agreement 85.9% | 14.4151 (−1.7%), 86.3% (stage 2: 14.3879, 86.6%) |
| Unit tests (`qwen38_*_test`, GB10) | – | 53 of 53 pass, among them the new `GdnStepIsTheColumnsRecurrenceInPlace` (the step's output and state equal the columns kernel's bit for bit; the verify's form leaves the state unchanged), `GemvBf16IsTheProductAtDecodeWidths` (NMSE against FP64 at most 1e-10 in F32, 2e-5 into BF16), `GdnHistoryShiftsTheOldHistoryForShortChunks` (exact) and the cluster form's `HcPrep` cases |
| Speculative greedy (every token the plain argmax or a near-tie, bound 1.0), 8 prompts | 0 violations | 0 violations; the repeats identical bit for bit |
| Forced rejections against the control (`capital`, 160 tokens, depth 2) | 86 steps, 0 states differing; 8 near-ties, largest 0.916, 0 past 1.0 | 85 steps, 0 states differing; 1 near-tie (0.20), 0 past 1.0 |
| The same, stages 1 and 2 (the BF16 vector kernel in the verify too) | – | 83 steps, 0 states differing; **2 tokens past qwen38-mtp's 1.0 margin** (steps 48 and 60: 2.51 and 1.09), the same bit for bit with and without PDL (below) |
| The same on other prompts (review, `spark-b`): `code` 200 tokens and `fibonacci` 200 at depth 2, `french` 160 and `prose` 256 at depth 3 | – | 108, 96, 99 and 133 steps, 0 states differing (at depth 3 "two accepted" 19 and 26 times); 0 tokens past 1.0 (largest near-tie 0.28); verify noise at most 0.55, 0.93, 1.21 and 0.61 |
| Swap with the commit owed (FP16 fixture as B) | (recorded) | 97 steps, 0 states differ, 0 of 160 tokens' logits differ, B's logits their recorded hash; graphs replayed across the swap |
| The Spark check set (`mise run test -- spark-native --locked`, `spark-b`, rebased on `0560a8c`) | – | 901 of 901 tests pass |

**The intermediate builds' two tokens.** Step 48 is the end-of-turn class that
DeepSeek's step 93 was: the plain engine prefers token 248045 by 2.51 nats
where the speculative run's verify row gave 248046 (main's build had the
same pair at step 42 with a margin of 0.48, and 248045 against 248044 at
step 97). Both pass under the forced run's own verify-against-plain noise
(p99 2.58, maximum 2.98; main's 2.37 and 2.39), the rule
[dsv4-decode](../dsv4-decode/README.md#the-bound-going-forward) recommends,
but that noise was measured with the run, not before it, and the recorded
test for Qwen3.8's speculation is 1.0, which is not loosened here. The
state check, the only one a wrong commit or rollback would fail, found no
difference in 83 steps. The final build, whose verify takes the mixes'
products from cuBLAS again, has none past 1.0 on its own trajectory (its
verify noise there: p99 4.28, so the margin test passing is itself partly
the trajectory's luck). On four more prompts and at depth 3 (the table
above) the final build had none past 1.0 and no state difference, its
verify noise at most 1.21: the class (verify noise against plain, not a
wrong commit) can recur on other text, since a verify and a plain step
take different kernels, but none was found. Not diagnosed: Qwen3.8 has no
probe like DeepSeek's; a probe is the lead if a later build meets it
again. (The forced check's control now avoids a token that the verify
would keep: at `prose` depth 3 its "other wrong" draft, one id past the
forced one, was the verify's own token, so the control kept a row the run
had rejected and the two diverged, a harness coincidence, not a state
difference.)

The stage 1 build's changes of arithmetic are the hyper-connection prep's
order of sums and the one-row BF16 products' order; the in-place state and
the fused one-row convolution repeat the earlier arithmetic bit for bit.
PDL changes no arithmetic.

## Judgement calls

- **A new operation for the in-place step** (`jitllm.gdn.step`), not a
  flag on GGML's `gated_delta_net` node: the node's output layout (rows
  then state) is GGML's, and an operation that writes its state operand is
  explicit in the plan and its check (the state disjoint from every other
  operand and the output).
- **The cluster prep only up to 8 tokens:** prefill keeps a block a token,
  which already fills the GPU.
- **The BF16 vector kernel at one row only,** after the verify's profile
  (above); its multi-column code stays for a later layout of the down
  product's blocks, but no node reaches it now, so the tests cover its
  four forms at one column only.
- **The cluster prep on compute capability 9.0 and later only:** the
  discrete sm_86 (D-082) has no thread-block clusters and keeps a block a
  token.
- **PDL through GGML's launcher** (`ggml_cuda_kernel_launch`), as
  DeepSeek's kernels: every kernel so launched waits
  (`cudaGridDependencySynchronize`) before reading what an earlier kernel
  wrote, and the small kernels that now trigger early are followed only by
  such kernels or by ordinary launches; x and y lose `__restrict__` where
  the wait guards them (GGML's rule).
- **The adaptive window kept off by default** and its probability output
  kept: measured, no gain at this drafter's cost per pass.
- **No import layout change,** though the brief listed it first: the
  measurements named no layout gap.
- **The near-tie bound for plain decode (B = 2.0) is main's p99** between
  two of its own paths, recorded before the new build met the oracle
  (dsv4-decode's rule). The speculation margin stays qwen38-mtp's 1.0; the
  intermediate builds' two exceptions are reported, not absorbed.

## Limits and what remains

- **Format remains a plausible substantial contributor:** Mia's MXFP8 dense
  layers and BF16 hyper-connection products and head add about 2.9 GB per token
  under the full-read model compared with TensorFold's 4-bit. The entire
  remaining gap is not causally assigned. Reading these weights in fewer bits is the
  quality/performance-modes item (a copy of the head in NVFP4 or MXFP8,
  a 4-bit import of TensorFold's format, D-087's M9 item).
- **Speculation did not gain** beyond the verify's one-column fix (its
  step is the target's per-row experts, about 5 ms a row, and the BF16
  draft head, 1.4 ms a pass). Leads: a quantized draft head (acceptance
  only), draft passes that stop early on the drafter's probability (the
  window would then save passes, not only rows), each expert read once for
  all a verify's rows (tried in qwen38-mtp and slower on the GB10).
- **Engine leads left:** the Gated DeltaNet output product still reads at
  about 180 GB/s in the model against 221 alone (the in-place state is
  still written back through L2); the small MXFP8 products (1.3–1.7 MB)
  are launch-bound at 140–163 GB/s; the alpha/beta chain (6 kernels a
  layer) could be one; DeepSeek's `HcPreKernel` could take the cluster
  form (about 1%).
- **TensorFold's kernel-level profile was not taken.** Nsight Systems
  inside its container (CUPTI 13.3 on driver 580.178.04) recorded no
  kernels, twice; the host's nsys mounted into the container was queued
  when `spark` was reserved. The TensorFold runs this study still needs:
  one decode step under the host's `nsys` (2025.3.2) in the container,
  plain and with drafts, to set its kernel classes beside jitLLM's.
- **The intermediate builds' two forced-run tokens** (above) were not
  diagnosed; the final build passes, and a Qwen3.8 probe like DeepSeek's
  is the tool if one comes back.
- **Harness, not runtime:** the window is `jitllm_qwen38_spec`'s; the
  runtime's chat path (D-096) gets the kernels, not the window.

## Reproduce

On `spark-b`, the `spark-native` build, the artifacts of
[qwen38-native](../qwen38-native/README.md) (`c4fb47a9…`) and
[qwen38-mtp](../qwen38-mtp/README.md) (`056a750e…`):

    jitllm_swap_pairs --a qwen38 --b dsv4 --cycles 0 --bench 128 ...   # plain decode
    jitllm_qwen38_spec ... --check greedy|forced|swap [--draft N] [--window P]
    jitllm_qwen38_exec --artifact Q --context 4096 --prompts prompts.tsv \
      --force forced.tsv --generate 32 [--stepwise] --out DIR          # greedy data
    jitllm_qwen38_exec --artifact Q --context 4096 --ppl ppl.tsv --max-rows 8 --out DIR
    python3 ../qwen38-native/compare.py ppl ../fast-swap DIR

The bytes tables come from the checkpoints' safetensors and GGUF headers;
the vmstat sampler reads `/proc/vmstat` every 0.5 s. Raw outputs,
profiles and the scratch scripts stay on `spark-b` under `~/scratch/m3tf/`.
