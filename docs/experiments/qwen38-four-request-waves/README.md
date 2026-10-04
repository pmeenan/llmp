<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 four-request decode waves

Qwen3.8 serving now decodes up to four requests in one wave, up from fixed
pairs. Matched 8K HTTP cells complete **+20% tokens/s at C4** and **+9.8% at
C2**. C1 is unchanged, with a byte-identical reply. Spark B (`spark-56f5`),
2026-10-02.

## What limited concurrency

An nsys trace of the private C2 benchmark (`jitllm_qwen38_batch --mode natural`)
was decisive. A two-request wave cost 92–104 ms against 57 ms for one
request's verify, so pairing saved only ~19%.

| Cause | What the trace showed |
| --- | --- |
| Routed experts | 35% of GPU time. `jitllm.moe.gemv` reads each (token, expert) slot's weights separately, so pairing barely helped (4.00 → 3.66 s an arm). |
| Fixed pairs of at most 8 rows | A 4-request wave read every dense weight twice. The earlier [exact16](../qwen38-mxfp8-sixteen/README.md) rejection was a register spill: the 8-column template was instantiated at 16 and used 1,744 bytes of local memory. It was not a limit of the approach. |
| Graph replay | Wave graph keys hold each slot's KV extent (256-cell alignment), rows and pending draft rows. With four slots the keys seldom repeated: 241K eager launches against 86 graph launches, and the GPU was only ~45% busy for long stretches. |

## Changes

All but the depth policy keep every request's arithmetic. Replies are
byte-identical across old/new binaries at C1, and between 2- and 4-slot builds.

- **Groups of up to four consecutive compatible slots** replace pairs: up
  to 16 rows a joined product, and target heads joined up to 16 columns.
  The preflight, authentication and per-slot state are as before.
- **Wide MXFP8 vector kernels** (9–16 columns): x is staged a chunk at a time
  in shared memory and weights are decoded to F32 once. Each output is
  bit-identical to the 8-column kernel. A standalone microbenchmark of every
  production shape picks the launch: 2 rows a warp with 16 warps for n ≥ 4096,
  2×8 for n ≥ 2048, and a register-only form below that. Large shapes reach
  ~170 GB/s against a ~220 GB/s 1-column ceiling.
- **Expert-major routed GEMV** for products of more than 4 tokens: each
  distinct expert is read once for every slot that chose it. A block leads for
  its expert and loops over 4 (gate/up) or 8 (down) output tiles. Outputs are
  bit-identical to the per-slot kernel.
  - C2 benchmark ABBA, 2 runs each way: +7.0% shared-wave throughput. Within
    one request L2 already dedups, so ≤4 tokens keep the per-slot form.
  - Tiles: down 458 → 312 µs a layer, gate/up 588 → 544 µs.
- **Wave KV alignment 2048 cells** for multi-slot target and draft waves.
  Cells past a row's position are masked, or hidden from selection by position,
  so no result changes. Those cells are still read (vector attention and the
  block scores read every cell), so a wave backs each slot's caches through
  the same alignment: up to 1,792 more cells of state a slot. The wave plan
  caches hold 64 entries.
- **Policy:**
  - 4 funded slots.
  - Each request in a shared wave drafts depth 2. A lone request keeps its
    adaptive 2–3 depth.
  - Past two requests, drafts run per slot on their cached scalar graphs.

## Results

All cells use a fresh service, the frozen common-v2 client and 8,256-token
prompts with 256 outputs each. Rates include prefill and queueing, as in
[serving-concurrent](../serving-concurrent/README.md).

| Cell | Old binary (bookends) | New (two runs) | Change |
| --- | --- | --- | ---: |
| C4 tok/s | 25.13, 27.06 | 32.00, 32.74 | about +20% |
| C2 tok/s | 26.96, 27.01 | 29.66, 29.56 | +9.8% |
| C1 tok/s | 25.65, 25.50 | 25.62, 25.68 | ±0 |

The cleaned production build gives 32.22 / 29.72 / 25.68. Against current
TensorFold 0.6.2 NVFP4 (31.87 / 24.79 / 21.46, the earlier session's
[matched comparison](../serving-concurrent/README.md)) native leads at C2 and C1
and is level at C4: +1.1% across sessions is within the run-to-run spread
above. Against legacy Mia (39.50 / 30.05 / 20.32) it trails 18% at C4 and
about 1% at C2.

### Draft depth

| Policy | C4 | C2 | C1 |
| --- | ---: | ---: | ---: |
| Adaptive 2–3, 4 slots | 26.76 | 26.93 | 25.69 |
| Fixed 3 | 29.70 | 28.25 | 24.68 |
| Fixed 2 | 31.29, 31.57 | 29.08 | 25.10, 25.10 |
| Selected: 2 when shared, adaptive alone | 31.67 | 29.52 | 25.59 |

Fixed depth 3 wins nowhere measured. Adaptive beats fixed 2 alone by about 2%.

Funding four slots raises the fixed budget from 3.89 to 7.45 GiB (workspace
3.25 → 6.80 GiB): the wave workspace is provisioned as four times the largest
scalar activation, including a 4,096-row prefill chunk, although waves run
only verify shapes. A tighter bound is a follow-up.

## State capacity

Four requests can outgrow the conversation-state capacity that two never
reached. Under a 30 GB host-memory hold (about 3.2–3.7 GB of state
capacity at startup), four concurrent ~10.7K-token prompts generating
~4–7K-token stories had all failed with a 500 at "the chunk at 8192". With
the cohort's capacity policy
([runtime-serving](../../runtime-serving.md#state-capacity-in-a-cohort))
the same attacks complete every request: equal 10,660-token prompts finish
in 442 s and staggered 10,480–12,280-token prompts in 430 s, each with
three waits, one preemption whose generation resumed from its rebuilt
state, and one cleared idle cache; the service stays healthy and answers a
follow-up. A matched 8K HTTP set after the change measured C4 / C2 / C1
32.45 / 30.04 / 25.61 completed tok/s, the C1 reply identical to the
earlier cell (spark-b, 2026-10-02; records under `~/scratch/ccqw-fix/`).

## Not adopted

- **Joining plain BF16 products** (draft heads, drafter products, router).
  GGML routes some joined products to cuBLAS, which the authentication
  refuses. A per-product selector probe is owed.
- **jitLLM's own HC kernels on every column count** (a cluster split-K down
  product, batch-invariant by construction). They are neutral solo and slower
  than cuBLAS at 12–16 columns, and they change solo trajectories. Reverted.
- **Grouped drafts past two requests**: 3.6% slower at C4.

## Open

- **Replies under C2/C4 vary run to run.** Wave composition depends on arrival
  timing, and the joined HC (cuBLAS) and head (GGML MMF) products are not
  column-count invariant. Solo runs repeat exactly. Batch invariance would
  need row-invariant head and HC products.
- **Per-wave budget at 12 rows** (~146 ms, ~10% gaps):

  | Component | ms |
  | --- | ---: |
  | Routed experts (near bandwidth) | ~52 |
  | MXFP8 | 19 |
  | Draft heads | 7.5 |
  | HC | 6.7 |
  | Target head | 5.4 |
  | HC prep | 4.4 |
  | GDN commit | 3.9 |

  The C4 gap to Mia is decode rate, about 58 vs ~100 tok/s. Prefill is similar.

Raw records are on spark-b under `~/scratch/cc-http/` and `~/scratch/cc-prof/`.
The microbenchmarks are in `~/scratch/cc-mxbench/`.

## Wave lanes

On 2026-10-03, the Qwen wave builder began tagging each request's private
attention and Gated DeltaNet work for a concurrent stream inside the wave.
Shared products, packing and cuBLAS work remain on the owning stream;
`LaneOrder` inserts dependencies for every overlapping read/write. The
planner keeps concurrent activation lifetimes separate, and the runner
charges separate per-lane scratch to its memory budget. `wave_lanes`
defaults to true and is an owner override; its value invalidates calibration
made with the other schedule. Waves below four requests retain one stream.

The inherited NVFP4/MTP screen with the curated 47,172-entry draft head
measured verify waves at 2/3/4 requests: lanes off 68.1/93.1/110.2 ms,
on 71.5/94.8/104.2 ms. This sets the four-request threshold. The following
qualification uses the production 65,536-entry prefix head instead.

### Correctness

`jitllm_qwen38_spec --check wave --slots 4 --tokens 96 --wave-lanes off|on`
produces exactly 96 greedy tokens per slot with depth two, context 16,384
and 4,096-row prefill chunks. Short inputs use the first two decode and
first two chat prompts in `fast-swap/prompts.json`. The long fixture repeats
350 public service records per prompt, exercising selected attention.
Its JSON SHA-256 is `5613e76b48e0d31b899099ae0d127f306afe36341b5bb523c177c80cbc5c6022` (external `qlanes/long-prompts.json`).
Regenerate it from the fast-swap JSON, replacing its decode list with four
entries named `long-0` through `long-3`: each has one user message made of
`Record {i:04}: a routine service ran for {i % 97} minutes with no errors.`
for i = 0..349, newline-separated, followed by a newline and the original
user message of the first two decode and first two chat entries respectively.
Serialize with Python `json.dumps` (default separators) plus a newline;
keep the other top-level fields.
All four token-history hashes, complete verify-logit hashes and final
initialized target/drafter state hashes match between schedules on both
fixtures. Both schedules capture and replay verify graphs: short 2 captures
and 38 replays, long 1 and 37. Each process completes and retires cleanly.

| Fixed-history control | One-stream verify median | Concurrent verify median |
| --- | ---: | ---: |
| Short | 112.299 ms | 107.196 ms |
| Long | 114.020 ms | 105.141 ms |

Medians count only waves with all four slots active. These clocks exclude
prefill and output handling; they are not serving rates.

The optional FP16 roundtrip supplies `--fp16-artifact`, `--fp16-tokens`
and `--fp16-expect` to the same wave control. After wave 25, all four slots
have accepted a partial verify and still owe restoring an actual saved
rejected row. The harness releases its request lease, spills every slot,
evicts target/drafter weights, evaluates the FP16 control, then restores
Qwen and continues. All token, complete verify-row and final state hashes
equal the unswapped lane control. All 12 graphs survive; the first verify
on return replays without capture, with 15 returning verify replays total.
The FP16 complete-logit hash is the pinned
`bb8ae5e7e3ac6da734173edb1111160a0c80a55c4279b94e67a8f90b142e7571`.

The UD-IQ3_XXS GGUF harness toggles lanes with the optional final argument
`jitllm_qwen38_gguf_wave ARTIFACT prompts.tsv OUT 2048 off|on`. Its short
and long attention controls cover widths two, three and four, paired and
unpaired products, eager execution and capture/replay. Each schedule passes
1,728 compared complete rows, 12 captures and 372 replays with zero coverage
violations and clean retirement. Across schedules the aggregate hash of all
wave logits is `fdf3b131f82586ea3df03002cc0a8ecc923fbd4bb03beba785eb39077b087977`,
and of final states `4734098a71fbf96b8ec5e53155363ab90e47f121685d15d634674e519797189b`.
At the production 2,048-cell alignment the reference is the unpaired wave;
this does not assert equivalence to the differently aligned scalar path.

### Serving screen

Each HTTP cell starts a fresh service, primes it, then releases the clients
at one barrier. Requests use 183-token prompts and complete 256 outputs
each with zero cached tokens. Rates pay prefill, graph preparation and
queueing. The NVFP4 screen compares main `a40388a` with the lane candidate;
the GGUF screen toggles `wave_lanes` in the same candidate binary.

| C4 cell | Off, first | On | Off, second | Gain over mean off |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 + MTP | 60.5895 tok/s | 62.6992 tok/s | 60.2670 tok/s | 3.76% |
| UD-IQ3_XXS, plain | 48.6823 tok/s | 52.4566 tok/s | 48.7776 tok/s | 7.65% |

Bookend drift is -0.53% and +0.20% respectively. HTTP replies still depend
on arrival-driven wave composition, as before; they are not this exactness
control. These short prompts do not close the 8K C4 comparison with Mia.

A subsequent same-binary plain NVFP4 control uses the owner switch, reverse
on/off order at short context and off/on at long context:

| Plain NVFP4 cell | Off | On | Change |
| --- | ---: | ---: | ---: |
| Short C1 | 25.6780 tok/s | 25.9266 tok/s | +0.97% |
| Short C2 | 35.0503 tok/s | 35.0695 tok/s | +0.05% |
| Short C4 | 40.2538 tok/s | 43.3880 tok/s | +7.79% |
| 8,258-token C4 | 26.2650 tok/s | 27.5971 tok/s | +5.07% |

C1/C2 differences are within normal spread; they execute the same one-stream
wave schedule. Long C4 first-token maximum is 15.621/15.676 s, median
completion 38.503/36.617 s. The conservative lane scratch raises provisioned
scratch from 174,063,616 to 868,220,928 bytes (662 MiB), paid from the budget;
short plain C4 peak MemAvailable drop is 76.624/77.291 GiB, long
77.635/78.555 GiB. This control funds four slots in both arms. A model capped
below four provisions no lane scratch. All requests finish the full output
budget, services exit zero, and there are no capacity waits or set-asides.

Final same-binary NVFP4/MTP checks toggle the same owner setting with
`speculation = true`, paying the same prompt/output work:

| MTP C4 cell | Off | On | Change |
| --- | ---: | ---: | ---: |
| Short | 61.1857 tok/s | 62.9863 tok/s | +2.94% |
| 8,258 tokens | 31.5163 tok/s | 31.6334 tok/s | +0.37% |

Short order is off/on and long on/off. Long median decode rate is
12.088/12.290 tok/s a request, maximum first-token
16.599/16.589 s, median completion
31.512/31.170 s, peak MemAvailable drop
80.400/81.175 GiB. Every request finishes 256 outputs and every
service exits zero. Arrival-driven changes in acceptance contribute to
these HTTP rates; the fixed-history verify medians above isolate scheduling.
The 8K difference is within spread, essentially neutral. Its rate still
trails the earlier legacy Mia cell, so that gap is open.

### Provenance

Spark B (`spark-56f5`), NVIDIA GB10 sm_121, driver 580.178.04,
CUDA toolkit 13.4.92; source based on `a40388a`, source-lock SHA-256
`440f03cdb52921c6c55843819e6ac950a5b2c4aafdc01055e52a0af3117ce32e`.
NVFP4 target `c4fb47a9…`, MTP `8600a998…`, checkpoint
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`; GGUF target
`5356b5b0…` (UD-IQ3_XXS). HTTP context 32,768, prefill chunk 4,096,
four funded slots, production draft head/policy. External supervised job
receipts and raw records are under `~/scratch/qlanes/{screen,quality,
gguf-screen,gguf-exact-off,gguf-exact-on,final-http,final-mtp}` on Spark B;
the inherited width screen is under `~/scratch/qx` on Spark.


## Fast-start Mia refresh — 2026-10-04

A fresh native/Mia/native C4 cell leaves native **15.01% below Mia's
completed-token rate**: 32.44 / 31.91 native bookends versus 37.86 tokens/s.
Native bookend wall movement is +1.66%. This refresh includes the landed
wave lanes and tighter slot workspace. It measures the remaining gap;
it does not qualify cross-engine output quality or the whole M3 exit.

Spark B (`spark-56f5`), driver 580.178.04, CUDA 13.4.92, SDK
`aarch64-e0a0c85c42806fb1`. All three arms use fresh services and state,
the frozen common-v2 buffered client, four 8,256-token rendered prompts,
greedy decoding and 256 outputs per request. Each service completes an
excluded one-output prime before the measured barrier-released burst.
Rates count usage-reported completed tokens from the first submission
through the last completed response, paying prefill, generation and queueing.
There is no streamed time to first token or isolated decode measurement.

| Arm | Burst wall s | Completed tokens/s | Request latencies s, in input order | Peak MemAvailable drop GiB | Ready s |
| --- | ---: | ---: | --- | ---: | ---: |
| Native before | 31.5652 | 32.4408 | 30.4729, 29.1460, 31.5638, 31.5644 | 81.285 | 3.145 |
| Fast-start Mia | 27.0467 | 37.8604 | 27.0424, 27.0464, 26.2092, 25.5980 | 101.421 | 172.239 |
| Native after | 32.0886 | 31.9117 | 30.9650, 29.6376, 32.0872, 32.0886 | 81.275 | 3.905 |

All 12 measured replies return HTTP 200, 256 outputs, length finishes,
8,256 prompt tokens and explicit zero cached tokens: 3,072 completed tokens.
Native bookends have identical content, reasoning and usage for every input.
Mia supplies complete 256-token output-ID records; native does not claim
API output IDs. Matching inputs is grounded in the frozen complete native/HF
rendered-ID controls and pinned template/tokenizer assets, with API counts
checked again here; Mia also returns complete prompt IDs, but fresh native engine-side capture
and a fresh cross-engine ID comparison were not performed.
Native and Mia use different draft and state arithmetic. The fixed quality
references remain unchanged, and no quality-bound verdict follows from this
throughput screen.

Native uses the production NVFP4 target `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`
and MTP drafter `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`,
context 33,792, chunk 4,096, max slots four and the 65,536-entry prefix draft
head. Shared waves draft depth two; a lone request remains adaptive.
Target KV is F16 and recurrent state F32. The source file inventory matches
main through `4412786` before and after both native arms; the runtime is the
final checked literal-cohort implementation (1,543 Spark tests passed).

Mia uses the [pinned fast-start launcher](../fast-swap/baselines.md#default-mia-launcher-for-new-runs-2026-10-03)
with the original performance policy: determinism off, fixed MTP depth three,
curated 47,172-entry head, FP8 KV and BF16 recurrent/draft IO, chunk 2,048,
context 33,792 and max sequences four. Compilation is disabled; decode graphs
use sizes 4/8/12/16. The image is
`sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`,
recipe `b8439110eec0230facbe4ddf0dffe01b8f769be0`, with the qualified
instanttensor 0.2.0 payload, `V2=1` and persistent Triton cache. There is no
observer patch. The offline HF snapshot and read-only checkpoint mount use
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`; actual code/vocabulary mount
hashes and five model assets match their qualification pins before and after.
All 38 checkpoint/PLE file stat identities also remain unchanged; this is
not a fresh hash of every weight payload. Readiness takes 172 seconds with
the retained compilation cache, without a cold page-cache claim.

Memory is sampled every 250 ms from before startup through shutdown. The
native mean peak MemAvailable drop is 0.801 times Mia's. These are host
budget measurements, not GPU allocator occupancy. All samplers finish
without errors. Both native services return zero and are reaped. The Mia launcher is
retired and reaped; the owned recipe stop returns zero and its container
is absent. All strong admission and retirement probes pass, and the final
GPU probe finds no compute process or busy/waiting job.

The installed GPU-supervised job `qwen-fast-serving-c4` completes successfully
and is waited on. Raw records and the controller remain outside Git at
`~/scratch/qwen-fast-serving/screen1/` on Spark B and
`/home/pmeenan/scratch/jitllm-m3-qwen-fast-serving-2026-10-04/` locally.
Reproduction uses that controller's native-before/Mia/native-after order,
the canonical launcher and frozen common-v2 inputs from
`~/scratch/m3-serving-concurrent-r1/inputs-v2/`. This is one bookended screen,
without a confidence interval or an updated C1/C2 result.

| Identity | SHA-256 |
| --- | --- |
| Native runtime | `cc7706fd1d17a3033374b87677ebd384644913b50f0d9426a73b74e394d92f24` |
| Production source inventory | `62b29cbd9f576cd44d94d5451534d055caac184030e75d0e99fa5640dc958339` |
| Controller | `c3c0ebe1072d7cd8127ab16d2866dafa148d57011fbce3b2a3f59a4e39c8da07` |
| Frozen client | `4f76adb8e36bb97e04c30f22d28d8d2ffc985d67aa4621f083ac92fc5e85d8f3` |
| Frozen input receipt | `d5a35e6341e53de0286cfd777e4fadd707c12cf9d18f95010a4a38ffac0ab59d` |


## Smaller expert-sharing groups — 2026-10-04

Reducing the private expert-major group cap from eight slots to four is
**not adopted**. The representative original/candidate/original control
measures verify medians 106.667 / 106.947 / 106.060 ms: the candidate is
0.549% slower than the mean original, with -0.569% original bookend movement.
This does not justify replacing the production eight-slot grouping or
expanding into a serving/context ladder.

The motivation was fewer live accumulators in `GemvExperts`; a repeated
expert selected more than four times instead needs several groups. Only
`kMatched = 8` changes to four, without changing dispatch, output tiles,
quantization, per-token reductions or solo verify. Register use, actual
occupancy and weight-traffic changes were not measured; none is inferred
from the neutral whole-wave result.

Spark B, driver 580.178.04, CUDA 13.4.92 and SDK
`aarch64-e0a0c85c42806fb1`. The private worktree is based on `4412786`.
`jitllm_qwen38_spec --check wave` uses the NVFP4/MTP artifacts above,
65,536-entry prefix head, four slots, lanes on, context 16,384,
4,096-row prefill, depth two within shared waves and 96 generated tokens per
slot. Inputs are the first two decode and first two chat entries from
`fast-swap/prompts.json`. This is an in-process short-prompt verify clock,
excluding prefill and draft time, not an HTTP throughput measurement.

All three arms produce 43 waves. Four token-history hashes, four complete
verify-row hashes and four final initialized target/drafter-state hashes
match exactly across all arms. Each captures two verify graphs and replays
38. Processes finish in 31.05–31.51 seconds, return zero and are reaped;
strong admission and retirement gates pass around every arm. The installed
GPU-supervised build and `qwen-expert-screen4` jobs finish successfully and
are waited on. No runtime source from this experiment lands; the diagnostic
optimization override defers full suites for rejected candidates.

Raw records, controller and one-line patch remain outside Git under
`~/scratch/qwen-expert-groups/` on Spark B and
`/home/pmeenan/scratch/jitllm-m3-qwen-expert-groups-2026-10-04/` locally.
The source worktree is `/home/pmeenan/src/jitLLM-qwen-expert-groups` and its
Spark copy `~/src/jitLLM-wt/qxgrp001/`. Repeat its `screen.py` in a fresh
output directory under the installed GPU supervisor and wait; it compares
complete per-slot hashes, completion and capture/replay as well as medians.

| Identity | SHA-256 |
| --- | --- |
| Original benchmark | `c2351f846370d201a2546f7781522b0fd9dd5f8c6fa877acde0fc3df9084164c` |
| Four-slot candidate | `d27c4569c88264478371f46ff71a622c8703c25330fbf0f3f00bde68a8937730` |
| Controller | `82a97a0ab9514cf87f64633550d30476dbdd14c9cc2f07a6e1938517e1567286` |
| Candidate patch | `a3bcd071b6c69c7c5cf54c6755f8d84f6f3a9266454b7f074976eae4b8ad703c` |
