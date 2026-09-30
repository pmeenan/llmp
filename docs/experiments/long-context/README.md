<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Long context: the baseline (2026-09-29)

This is phase 1 of M3's **Long context** item
([plan.md](../../plan.md#m3--single-spark-fast-full-swap--in-progress)).
It measures each LLM's one-Spark maximum context and runs the context
ladder through `jitllm-runtime` and through the same-format comparators on
the same prompts. The gaps it finds set the optimization slices of
phase 2; [phase 2's DeepSeek slice](#phase-2-deepseek-flat-with-depth-2026-09-29)
fixed gap 1. Coarse by design (D-085): one run per depth. Every number here is
**measured** on `spark` or `spark-b` (GB10, driver 580.178.04) unless
marked **computed**. jitLLM was measured at `6c182c3` plus this change's
working tree (the fixes below).

**Phase 2's Qwen3.8 slice** ([Qwen3.8 Flash Next flat with
depth](#phase-2-qwen38-flash-next-flat-with-depth)) fixed gap 2 and
supersedes the Qwen3.8 rows below: its per-token cost is now flat to
within the indexer, at least Mia's vLLM's speed plain at every depth,
repeatable, and speculative to 262,144.

The [final context checks](../m3-final-context/README.md) follow the
later kernel transfers, growing state and turn reuse. They distinguish
the earlier 1M request fit from final-path maximum, quality and swap gates;
pending results there are not passes.

## Phase 2: DeepSeek flat with depth (2026-09-29)

Phase 2's first slice fixes gap 1 (below) on DeepSeek V4 Flash's default
fast plan, in the order that gap proposes:

- **The window cache is a ring** of the window and a prefill chunk, 2,304
  cells at 2,048-row chunks (`model/dsv4.h` `Dsv4Window::kRing`), not a
  cell per position; the reference mode (`--exact on`) keeps llama.cpp's
  full cache. Each compressed layer's compressed cache follows its window
  cache in the state, so attention reads the two as one K in place: no
  concatenation.
- **Attention gathers what a token attends:** its 128 window cells and
  its layer's compressed rows (CSA: the indexer's 512; HCA: the visible
  ones), through the pinned MMA kernel's sparse gather, which jitLLM's
  graph now marks for every layer (`SetFlashAttnSparseAny`; upstream's
  takes it only past 4,096 cells). The masks are built on the device from
  each row's visible counts (`jitllm.dsv4.sparse_mask`): no host-built
  mask grows with the context.
- **The indexer is jitLLM's** (`jitllm.dsv4.lid_topk`,
  `kernels/ggml/dsv4_sparse.cu`): scores on tensor cores (a block scores
  four rows' 64 heads against each key tile, so prefill and a verify's
  rows share one pass), then a radix select that keeps the lower row
  among equals (two stages for decode's few rows over many keys).
  Deterministic: RE-031 is closed for DeepSeek's default plan. The scores
  of a prefill chunk go through pool scratch of at most 128 MiB, the rows
  in groups that fit it.
- **The memory guard** keeps 6 GiB (was 4) and counts each model's
  host-built chunk inputs beside it (`runtime/memory_guard.h`; gap 7).

Measured on `spark` (GB10, driver 580.178.04) at this change's working
tree; the comparator numbers are phase 1's (llama.cpp b11254 at 32K–128K,
b10964 at 8K, M3's earlier runs); one run per depth.

### Speed through the runtime

The runtime at `context = 262144`, plain and with DSpark, 512 greedy
tokens (the retrieval prompts fewer); the 8K rung is the Qwen3.8 corpus's
8K prompt (7,671 DeepSeek tokens: the builder cannot fill DeepSeek's own
8K rung within 1%). Phase 1's jitLLM numbers are "before"; at 128K the
route stopped phase 1's run, and the watchdog slice's run (plain, context
131,072, 64 tokens, 2026-09-29) is the "before".

| Depth | Prefill tok/s: before → after (× llama.cpp) | Plain decode tok/s: before → after (×) | DSpark decode tok/s: before → after (×) | Retrieval |
| --- | --- | --- | --- | --- |
| 8K | 463 → 438 (1.24× b10964's 352) | 21.9–22.2 → 21.9 (1.10× 19.9) | 31.6 → 37.3 (1.21× 30.8) | pass |
| 32K | 333 → **471 (1.65×)** | 14.7 → **21.6 (1.15×)** | 31.3 → **37.5 (1.22×)** | pass |
| 64K | 230 → **466 (1.69×)** | 10.8 → **21.2 (1.18×)** | 21.4 → **32.1 (1.11×)** | pass |
| 128K | 154 → **445 (1.72×)** | 7.1 → **20.5 (1.22×)** | refused → **36.6 (1.19×)** | pass |

DSpark's rate follows each answer's acceptance (llama.cpp's too: 0.59 at
64K against 0.62–0.71 elsewhere). Retrieval passed at every depth run, in
both modes (all three notes). The runs used this slice's first selection
kernel; the final one (a parallel bin choice and ballot compaction)
selects the same rows (its unit tests compare with a host reference; the
32K forced run's logits are the first build's bit for bit) in less time
(below). Re-measured on the final build (the review, one run each): 32K
prefill 473 / 470 tok/s and decode 21.7 plain, 37.7 DSpark; 64K 466 /
465 tok/s and 21.4 plain, 32.4 DSpark: within 1% of the table.

### The slope, per 1K tokens of context

From the rates above (a decode step 1 ÷ rate; prefill per token 1 ÷ rate):

| Cost | Before (8K → 64K) | After (8K → 64K → 128K) | llama.cpp b11254 (32K → 256K) |
| --- | --- | --- | --- |
| Decode step | +0.83 ms per 1K (45.4 → 92.7 ms) | **+0.026 ms per 1K** (45.7 → 47.2 → 48.9 ms) | +0.066 ms per 1K (53.1 → 67.9 ms) |
| Prefill, per token | +0.039 ms per 1K (2.16 → 4.35 ms) | **+0.0013 ms per 1K** (32K 2.12 → 64K 2.15 → 128K 2.25 ms) | +0.0039 ms per 1K (3.50 → 4.37 ms) |

So a decode step at 64K costs 3.2% more than at 8K (was 104%), at 128K
6.5%; prefill at 128K 5.5% more than at 32K (8K's rate includes a
larger share of fixed costs). What remains is the architecture's O(n)
part, measured in the profile:

### Where the time goes, after

`nsys` over the resident harness (the final build), the same prompts and
method as phase 1's profile (a chunk at 6,144 and 61,440 tokens of
context; a decode step at 8,195 and 63,491):

| ms | Prefill chunk 8K | 64K | Decode step 8K | 64K |
| --- | ---: | ---: | ---: | ---: |
| Flash attention (the sparse gather) | 903.4 | 745.9 | 0.87 | 0.90 |
| The masks (jitllm.dsv4.sparse_mask) and the gather's compaction | 6.4 | 19.9 | 0.33 | 0.41 |
| Indexer scoring (LidScoreKernel) | 16.8 | 135.2 | 0.22 | 0.58 |
| Indexer selection (TopKKernel, and its merge) | 7.7 | 43.8 | 0.16 | 0.45 |
| Everything else | 3,291.6 | 3,318.2 | 47.21 | 47.87 |
| **Total** | **4,225.9** | **4,262.9** | **48.79** | **50.21** |

From 8K to 64K the prefill chunk grows 37 ms (0.9%; phase 1: 7.6 s): the
indexer's scoring +118 ms (at about 81 TFLOP/s on the tensor cores, 11
TFLOP a chunk at 64K) and selection +36 ms, attention 158 ms less. The
decode step grows 1.4 ms (2.9%; phase 1: 40.5 ms): the indexer's scoring
+0.37 ms (the keys read, 84 MB at 64K) and selection +0.29 ms, the masks
+0.08; the quantized products (the weights' same bytes) +0.62 ms,
consistent across three profiles, which the attention does not explain (a
longer prefill before the step, and so a warmer device, is the likely
cause; not isolated). Attention itself is flat. The indexer's O(n)
scoring and selection, 1.4% of the step at 64K, is the architecture's
part; phase 1 computed about 1% from the keys' bytes alone.

### Correctness

On `spark`, with the resident harness (this change's fast plan) and
[judge.py](judge.py), phase 1's method:

| Check | Result |
| --- | --- |
| Near-tie bound, recorded first: the fast plan against the reference form (`--exact on`), 512 forced steps at 32K | p99 **0.947** (p50 0.173, max 1.243; 500 of 512 argmax equal); phase 1's was 1.24 |
| Greedy at 32K against llama.cpp b11254 | 493 of 512 equal, 19 near-ties (oracle margin at most 0.725), **0 outside: pass** (phase 1: step 249 outside) |
| Perplexity at 32K (16,383 tokens scored) | 1.8545 against 1.8528 (+0.09%): **pass** |
| Retrieval through the runtime | pass at 8K, 32K, 64K and 128K, plain and with DSpark |
| The same 32K forced run twice, bit for bit | **yes**: 0 of 512 steps differ (RE-031 closed for the fast plan) |

Speculation, with the fast plan's batched verify (`jitllm_spec_runner
--check forced --max-rows 128 --tokens 320`, so the window ring is 256
cells and the `capital` prompt's 336 positions wrap it): 164 steps, 140
with rejected rows (all rejected, one or two accepted, and at rows
completing a CSA or an HCA block), **0 stale bytes** after the rollbacks
in 14.9 GB compared; 318 of 320 speculative tokens the plain engine's
argmax on their prefix, 2 near-ties (verify noise p99 2.61). (The first
run's 160 tokens, 176 positions, did not wrap the ring.) The review's
runs, on the final build (`spark`, 2026-09-29):

| Check | Result |
| --- | --- |
| Rollback composes with swap (`--check swap --max-rows 128 --tokens 320`, the FP16 fixture as B), over the wrapping ring | A out and back after step 94 (rows rejected): 189 steps compared, **0 states differ**; B's logits its recorded hash |
| Greedy at short context against llama.cpp's record ([reference-deepseek-v4-flash-0731-llamacpp.json](../fast-swap/reference-deepseek-v4-flash-0731-llamacpp.json), six chat prompts, 32 tokens; `--check greedy`), the first divergence judged | plain: 4 of 6 equal throughout, `sky` from step 27 (oracle lead 0.25), `french` from step 0 (0.29); DSpark: 4 of 6, `sky` from step 20 (1.11, inside its verify noise p99 2.54), above the 32K bound of 0.947 and *accepted by the owner, 2026-09-29, as a near-tie*, `french` from step 0 (0.29); every speculative token the plain engine's argmax or within its near-tie margin: **pass** |
| The runtime's swap table, DeepSeek (DSpark on, `context = 36864`) holding 32,768 tokens of the book, so its ring has wrapped, against Qwen3.8 (`swap-table --pairs deepseek:qwen3.8 --context-tokens 32768 --cycles 2`) | A→B 7.88 / 7.83 s (first use / prepared; 0.38 GB spilled, phase 1's 3.35 GB at 64K), B→A 9.26 / 9.27 s (restore 0.09 s); **every row exact** (A's state hashing as it left, its 16 continued tokens and logits the unswapped continuation's), the prepared return replaying graphs captured before the swap (18 replays, none captured); lowest `MemAvailable` 7.95 GiB |

Not run: the greedy comparison at 8K (no DeepSeek 8K oracle record), the
128K teacher-forced comparison, perplexity at 128K, and the image pair
of the swap table (its reference noise is on `spark-b` only).

### Step 249

Phase 1's fast plan chose token 10386 at step 249 of the 32K prompt, where
llama.cpp b11254 prefers 82437 by 2.62 nats; the reference form agreed.
Diagnosed with the dsv4-decode probe's method on the resident harness
(`jitllm_dsv4_exec --probe-step 249`, a full window so both plans read one
state; [probe_step.py](probe_step.py)): the step run in both plans from
the fast plan's state (F/F, E/F) and from the reference's (E/E, F/E).

| Path | Lead of 82437 over 10386, nats |
| --- | --- |
| Phase 1 fast plan, two runs | −0.62, −0.84 |
| Phase 1 reference form | +0.72 |
| This fast plan (ring), two runs | +0.10, +0.10 |
| Probe F/F, E/F, E/E, F/E | +0.44, +0.16, +0.78, +0.22 |

- **The token is near-tied in every jitLLM path** (−0.84 to +0.78);
  llama.cpp's 2.62 is its own arithmetic (another build, fused, with its
  own sparse attention), 1.8 nats from jitLLM's reference form at this
  token.
- **On one state the plans differ continuously**: the fast plan against
  the reference on the fast state moves the lead by 0.28, the logits by
  1.8%, with no routing flip; the indexer selections differ in 20 CSA
  layers by 1 to 5 of 512 rows, at the selection's boundary (the reference
  keeps GGML's top-k, which breaks ties arbitrarily, over other sums).
- **Across states the flips are discrete**: the same plan on the two
  states selects another sixth expert in 7 to 8 layers (the reference's
  gaps between the sixth and seventh selection scores 0.0016–0.029) and
  other indexer rows (up to 14 of 512); the residual streams differ by up
  to 19% and the lead by 0.62.
- **Nothing points at a defect**: no mask, selection or state differs
  beyond near-tied choices, and the new plan's selection is exact against
  a host reference (unit tests).

**Verdict:** noise, carried by discrete flips (the indexer's boundary rows
and near-tied routing) at a token every jitLLM path holds within 0.9 nats
of a tie; phase 1's fast plan fell on the other side of it. Not a defect.

### Memory and the maximum

At `context = 262144` the runtime's fixed memory is 3.50 GiB plain
(phase 1: 20.36) and 3.89 GiB with DSpark, whose guard (weights 100.4 GiB,
host inputs 0.04, margin 6) now passes with 6.4 GiB to spare where it
refused above 143,360. Peak `MemAvailable` drop: 98.0 GiB plain (1.02×
llama.cpp's 95.8) and 109.0 GiB with DSpark (1.02× its 107.1). So the
one-Spark maximum is the configuration's bound, **262,144, plain and with
DSpark** (both served a 128K prompt; a 256K prompt was not run). Past the
bound (computed): the state is 6,880 bytes a position plus 99 MiB (6.8 GiB
at 1,048,576), and the prefill masks' workspace about 1 KiB a position at
2,048-row chunks: about 9.5 GiB fixed at a million tokens, which fits
plain beside the weights and the margin (about 106 of the 116 GiB
available) and would reach the guard's edge with DSpark.
The configuration refuses contexts above 262,144 (a configuration change,
gap 4). Two LLMs registered together are another matter: DeepSeek with
DSpark and Qwen3.8 both at `context = 65536` map 14.4 GiB fixed and are
refused (the old 4 GiB guard refused them too); at 36,864 each they
were refused by 0.1 GiB while the guard summed the models' host inputs,
which it no longer does (one model runs chunks at a time, so it counts
the largest; the review's fix).
The swap table above ran with Qwen3.8 at 8,704 (3.7 GiB fixed).

## Headline

Through `jitllm-runtime`'s chat route against the same-format comparator
on the same prompt, jitLLM first. DeepSeek against llama.cpp b11254
(UD-Q2_K_XL; speculative: DSpark on both sides); Qwen3.8 against Mia's
vLLM (NVFP4; prefill against the faster of its two launches, plain decode
against its deterministic MTP-off launch, speculative against MTP 3 with
jitLLM's MTP depth 2). "—": not run (below).

| Model, depth | Prefill, tok/s (ratio) | Plain decode, tok/s (ratio) | Speculative decode, tok/s (ratio) | Retrieval (jitLLM) |
| --- | --- | --- | --- | --- |
| DeepSeek, 8K (M3's earlier runs) | 463 / 352 (b10964) | 21.9–22.2 / 19.9 (b10964) | 31.6 / 30.8 | — |
| DeepSeek, 32K | 333 / 286 (1.16×) | 14.7 / 18.8 (**0.78×**) | 31.3 / 30.6 (1.02×) | pass |
| DeepSeek, 64K | 230 / 275 (**0.84×**) | 10.8 / 18.0 (**0.60×**) | 21.4 / 29.0 (**0.74×**) | pass |
| DeepSeek, 128K | — / 258 | — / 16.8 | — / 30.6 | — |
| DeepSeek, 256K | — / 229 | — / 14.7 | — / 28.4 | — |
| Qwen3.8, 8K (M3's earlier runs) | 2,320 / 2,101 | 27.4 / 25.1 | 40.1–42.5 / 37.9 | — |
| Qwen3.8, 32K | 2,053 / 1,729 (1.19×) | 21.8 / 24.6 (**0.89×**) | 35.4 / 37.3 (0.95×) | pass |
| Qwen3.8, 64K | 1,367 / 1,947 (**0.70×**) | 17.1 / 24.1 (**0.71×**) | refused / 40.5 | pass |
| Qwen3.8, 128K | 859 / 1,909 (**0.45×**) | 11.9 / 23.8 (**0.50×**) | refused / 48.7 | pass |
| Qwen3.8, 256K (its maximum) | 487 / 1,775 (**0.27×**) | 6.8 / 23.7 (**0.29×**) | refused / 37.7 | pass |

**The finding is the slope.** Both comparators are nearly flat with depth
(llama.cpp's decode falls 22% from 32K to 256K, Mia's 4%); jitLLM's
per-token cost grows linearly: DeepSeek's decode step costs 0.7–0.8 ms
more per 1K tokens of context (llama.cpp's 0.07), Qwen3.8's 0.3–0.4 ms
(Mia's 0.01; computed from the rates and the profile's step times). The
profile ([Where the time goes](#where-the-time-goes))
shows why: jitLLM's attention does dense work over every cached cell,
where both architectures only need a window and a fixed top-k.

**Maximum context on one Spark** (the runtime's memory guard with its 4 GiB
margin, `kUncountedMargin`; one model registered):

| Model | Configured | Plain | Speculative | Bound by |
| --- | ---: | ---: | ---: | --- |
| Qwen3.8 Flash Next | 262,144 | **262,144**, verified with a 258,633-token prompt (2,040-row chunks, RE-037) | **32,768** | plain: its configured maximum; speculative: the MTP drafter's selection works only up to 8,192 blocks, so registration refuses more |
| DeepSeek V4 Flash | 262,144 (trained 1,048,576) | **262,144**, served (32K and 64K prompts at that context); the guard passes by 1.2–1.3 GiB and refused once | **143,360** (the guard; not run with a prompt) | the configuration's bound, then memory: the full-size window cache is 49.7 KiB a token |

## Contents

- [Phase 2: DeepSeek flat with depth](#phase-2-deepseek-flat-with-depth-2026-09-29):
  before and after, the slope, the profile, correctness and the maximum.
- [The maximum contexts](#the-maximum-contexts): state per token, the
  guard, and the runs at each maximum.
- [Corpus and harness](#corpus-and-harness): the prompts, the retrieval
  check, the perplexity text, and how to rebuild them.
- [Comparators](#comparators): the llama.cpp pin with sparse
  flash-attention prefill, Mia's vLLM, and how each ran.
- [Results at depth](#results-at-depth): per model and depth; [Where the
  time goes](#where-the-time-goes): the per-kernel profile, and what the
  architectures allow.
- [Correctness at depth](#correctness-at-depth).
- [Swap with a long saved context](#swap-with-a-long-saved-context).
- [Turn-to-turn reuse](#turn-to-turn-reuse).
- [Memory and the guard's margin](#memory-and-the-guards-margin).
- [Fixed to measure](#fixed-to-measure): the blockers this phase fixed.
- [Gap list for phase 2](#gap-list-for-phase-2), ranked, with levers and
  effort.
- [Phase 2: Qwen3.8 Flash Next flat with
  depth](#phase-2-qwen38-flash-next-flat-with-depth) (gap 2 done).
- [Not run, and why](#not-run-and-why); [Reproduce](#reproduce).

## The maximum contexts

### State per token, from the model layer (computed)

| Model | Per token | Fixed | At 262,144 | At 1,048,576 |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek V4 Flash (`model/dsv4.h` `Dsv4State`) | 50,912 B: the window cache 44,032 (43 layers × 512 F16, a cell per position: `swa_full`), CSA keys 5,376 (21 layers, one row per 4 positions), indexer keys 1,344, HCA keys 160 (20 layers, per 128) | ~12 MiB of compressor rings; DSpark's window ring | 12.4 GiB | 49.7 GiB |
| … with the window cache as a ring of window + chunk rows (not built) | 6,880 B | ~95 MiB at 2,048-row chunks | 1.7 GiB | 6.7 GiB |
| Qwen3.8 Flash Next (`model/qwen38.h` `Qwen38State`) | 30,720 B: 12 QSA layers × (K and V 1,024 B each in F16, indexer keys 512 B in F32) | 118 MB: 36 Gated DeltaNet layers' recurrent (3 MiB) and convolution state | 7.5 GiB | — (configured maximum 262,144) |
| … its MTP drafter's layer | +2,560 B | — | +0.6 GiB | — |

Mia's vLLM keeps Qwen3.8's KV in FP8 (half of jitLLM's F16 per token) and
sizes its pool at 16–19 GiB for four sequences; llama.cpp keeps DeepSeek's
window as a ring.

### What the runtime reserves: the guard

At registration the runtime maps each model's own memory (its state at the
configured context, the workspace for the largest chunk, the chunk-input
staging) for the node's life, and refuses to serve unless the largest
model's weights plus a 4 GiB margin fit the memory then available
(`kUncountedMargin`, runtime-serving.md). Measured at startup on `spark-b`
(one model registered, nothing else running; "fixed" is the node's mapped
memory, the workspace part of it):

| Model, mode | Context | Prefill chunk | Fixed, GiB (workspace) | Available after, GiB | Weights + 4, GiB | Serves |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DeepSeek, DSpark | 65,536 | 2,048 | 5.90 (1.89) | 110.34 | 104.4 | yes |
| DeepSeek, DSpark | 131,072 | 2,048 | 10.66 (2.79) | 105.40 | 104.4 | yes |
| DeepSeek, DSpark | 143,360 | 2,048 | 11.72 (3.12) | 104.58 | 104.4 | yes |
| DeepSeek, DSpark | 147,456 | 2,048 | 12.0 | 104.3 | 104.4 | no |
| DeepSeek, plain | 262,144 | 2,048 | 20.36 (4.78) | 95.53–95.62 (93.8 once) | 94.3 | yes, by 1.2–1.3 GiB (refused once) |
| Qwen3.8, plain | 131,072 | 4,096 | 21.80 (10.38) | 94.2 | 73.9 | yes |
| Qwen3.8, plain | 196,608 | 4,096 | 31.86 (15.06) | 83.76 | 73.9 | yes |
| Qwen3.8, plain | 262,144 | 4,096 | — | — | — | no: RE-037 |
| Qwen3.8, plain | 262,144 | 3,584 | 37.65 (17.29) | 78.01 | 73.9 | yes (then failed at 147K: RE-037) |
| Qwen3.8, plain | 262,144 | 2,040 (the cap since RE-037) | 24.74 (9.84) | 89.71 | 73.9 | yes |
| Qwen3.8, MTP | 32,768 | 4,096 | 3.87 (1.77) | 112.49 | 79.3 | yes |
| Qwen3.8, MTP | 36,864 and up | 4,096 | — | — | — | no: "the drafter selects on the device only" |

So past the state itself, the runtime's fixed memory grows about 77 KiB
per token for DeepSeek (state 49.7, chunk-input staging ~12 for its
double-buffered F16 masks [n_kv, 2,048], workspace ~15) and 161 KiB per
token for Qwen3.8 at 4,096-row chunks (state 30; the rest the
[n_kv, rows] masks and scores in the staging and workspace), 94 KiB at
2,040 rows (computed from the table).

### The maxima, verified

- **Qwen3.8: 262,144 (its configured maximum), plain.** Verified by the
  256K retrieval prompt (258,633 tokens rendered) at `context = 262144`:
  prefilled in 2,040-row chunks, answered with all three codenames, peak
  101.2 GiB. It needed the RE-037 fix (below): before it the service
  refused to start at 4,096 rows and failed at 147K at 3,584.
  **Speculative: 32,768.** Past 8,192 selection blocks the MTP drafter's
  graph has no host-mask fallback (the target falls back to GGML's
  top-k; the drafter is refused at registration), so a context above
  32,768 cannot speculate at all.
- **DeepSeek V4 Flash: 262,144, plain** (the configuration's upper bound,
  and within 1.3 GiB of the guard): the runtime registered and served at
  `context = 262144`, with the 32K and 64K prompts; peak 116.0 GiB, the
  whole Spark. A prompt near the maximum was not run: the 128K prompt
  outran the chat route's 600 s request deadline (the state kept 108,544
  tokens), and then the owner stopped the rungs past 64K until the
  scaling is fixed. It needed the RE-038 fix: before it every prompt past
  about 52K failed. **With DSpark: 143,360** (the guard refuses 147,456),
  from the guard alone. The trained 1,048,576 would need
  49.7 GiB of state as built, beside 90.3 GiB of weights (100.4 with
  DSpark): it does not fit, and the configuration refuses a context above
  262,144 anyway. With the window cache as a ring, 1M is 6.7 GiB of state
  (computed), and fits beside the weights and the drafter.

## Corpus and harness

Built by [build_prompts.py](build_prompts.py) from
[corpus.json](corpus.json), both fixed before the first run, and run on a
Spark in the pinned PyTorch image (its `tokenizers`); the prompts stay
outside Git on both Sparks under `~/.local/share/jitllm/m3lc/prompts/`,
identified by their content hashes (below).

- **Coding context:** source files of llama.cpp at `8019dc563` (b11254,
  MIT, `LICENSE` SHA-256 `94f29bbe…`), in a fixed order (`tools/server`
  first, then `src`, `include`, `common`, `ggml`, the other tools,
  examples, tests), each whole file that still fits the rung's budget
  (files over 256 KiB skipped), each introduced by `=== FILE: path ===`,
  then one question: quote three notes' values, then explain how
  llama-server picks a slot and when it reuses a cached prompt. Rungs are
  sized with each model's own tokenizer so the rendered prompt plus 512
  generated tokens fits the rung: 32K (32,768), 64K, 128K, 256K (262,144)
  and 1M (1,048,576, DeepSeek only).
- **Retrieval check:** three notes (A at about 10% of the files' tokens,
  B at 50%, C at 90%, placed at file boundaries; recorded token offsets)
  in every rung. The first prompts called their values "maintenance
  passphrases", and at 256K Mia's vLLM reasoned they were planted secrets
  and declined to repeat them; so the check has its own prompts (`-r`, the
  same files) whose notes give neutral "release codenames" and whose one
  question asks for them in three lines. A rung passes when all three
  appear in the output (reasoning or answer).
- **Perplexity:** *War and Peace* (Project Gutenberg #2600, public domain
  in the USA; the download's SHA-256 `2d5bb2ad…`, its header and footer
  cut: `ppl.txt`, SHA-256 `c7156148…`, 777,232 DeepSeek tokens). Each
  oracle tokenizes it itself and jitLLM is fed the oracle's window of IDs;
  scored is the second half of one window (llama-perplexity's rule for one
  chunk): 32,768 and 131,072 tokens. The book is heavily memorized (a
  32K-window perplexity of 1.4–1.9), so only ratios mean anything.
- **Turn reuse:** `s64k`, a 64K rung built with 4,096 tokens of room for
  two follow-up questions (write a function; then what changes for
  multimodal chunks).

| Prompt | DeepSeek content tokens (SHA-256 of the content) | Qwen3.8 content tokens (SHA-256) |
| --- | --- | --- |
| `8k` | — | 7,534 (`b2d24aa0…`) |
| `32k` | 31,701 (`1676feba…`) | 31,691 (`c7928ea7…`) |
| `64k` | 64,443 (`92ff03c0…`) | 64,058 (`1aa0c238…`) |
| `128k` | 128,817 (`e94961ec…`) | 128,747 (`b7154f4b…`) |
| `256k` | 258,852 (`d1e6ac95…`) | 258,650 (`14cf7d90…`) |
| `1m` | 1,037,954 (`aad4e576…`) | — |
| `s64k` (turn reuse) | 61,111 (`71a7a214…`) | 60,850 (`607f2bb4…`) |
| `32k-r` … `256k-r` (retrieval) | 31,624 (`677ca618…`), 64,488 (`63da4e81…`), 128,740 (`7ff7d36f…`), 258,775 (`a534146a…`) | 31,622 (`b182da30…`), 63,989 (`82129ba6…`), 128,985 (`7ff7d36f…`), 258,581 (`ab1912e3…`) |

The engines' own counts, rendered with the template, are 30–70 tokens
more (`32k`: 31,705 for DeepSeek, 31,743 for Qwen3.8). The builder
writes a `manifest.json` per model with every prompt's counts, files and
note offsets.

The harness is [longctx.py](longctx.py) (it reuses the M3 baselines'
[baseline.py](../fast-swap/baseline.py) for its client and memory
sampler) and [judge.py](judge.py). Prefill = first streamed piece −
request sent (one decode step included); decode = (tokens − 1) ÷ (last
piece − first piece), 512 generated tokens greedy (fewer when the model
stopped); peak memory = the drop in `MemAvailable` from before the engine
started, sampled every 200 ms. jitLLM runs start with a short warm-up
request so no measured prefill includes paging the model in.

## Comparators

- **llama.cpp for DeepSeek**, UD-Q2_K_XL, same-format oracle and
  comparator: built by us on `spark-b` from `8019dc563` (b11254, which has
  sparse flash-attention prefill for DeepSeek V4, #29298, and for Qwen3.8,
  #28770) with upstream's `.devops/cuda.Dockerfile`, target `full`, CUDA
  13.4.1, `CUDA_DOCKER_ARCH=121a-real` (upstream's default list includes
  121a-real; the GB10's code is the same), local image
  `jitllm-llamacpp:b11254-cuda13`, `sha256:6dd02591…`, copied to `spark`
  ([pins.json](../fast-swap/pins.json)). Upstream publishes no image for
  b11254 yet (the newest is b11243). The b10964 pin stays the 8K oracle.
  Server arguments as the M3 baselines: `-ngl all -fa on -c 262144 -np 1
  --fit off -cram 0`; DSpark adds `-md dspark-…-Q8_0.gguf --spec-type
  draft-dspark --spec-draft-n-max 3 -ngld all`. Requests are the prompt's
  IDs (the server's own template and tokenizer) to `/completion` with
  `cache_prompt: false` and five log-probabilities.
- **Mia's vLLM for Qwen3.8**, NVFP4, oracle and comparator: the recipe
  and image the baselines pinned, on `spark` with its `.env` (262,144
  context, FP8 KV, 2,048-token prefill chunks). Two cold launches: the
  oracle's deterministic mode with MTP off (`MTP_NUM_SPECULATIVE_TOKENS=0
  VLLM_QSA_DET_TOPK=1 VLLM_MOE_DET_FINALIZE=1`; ready in 622 s), and the
  default MTP-3 launch (685 s). Each request a unique `cache_salt` (no
  prefix reuse), prompt IDs from `/tokenize` with the template.
- TensorFold was not run (below).

## Results at depth

One run per depth, 512 greedy tokens (fewer where the model stopped).
Prompt tokens are the engine's count; "peak" is the drop in
`MemAvailable` over the whole run (so it is the deepest rung's).

### DeepSeek V4 Flash (UD-Q2_K_XL)

| Depth | Engine | Prompt tokens | Prefill s (tok/s) | Decode tok/s | Speculative decode tok/s (acceptance) | Peak GiB (context) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32K | jitLLM (plain on `spark-b`, DSpark on `spark`; 2,048-row chunks) | 31,705 | 95.2 (333) | 14.66 | 31.34 (context 65,536; prefill 332 tok/s with DSpark) | 116.0 (262,144) |
| 32K | llama.cpp b11254 (`spark-b`; DSpark on `spark`) | 31,705 | 110.8 (286) | 18.82 | 30.61 (0.62) | 95.8 (262,144); 107.1 with DSpark |
| 64K | jitLLM | 64,447 | 279.8 (230) | 10.79 | 21.40 (prefill 242 tok/s) | 110.8 with DSpark (65,536) |
| 64K | llama.cpp | 64,447 | 234.1 (275) | 18.01 | 28.96 (0.59) | |
| 128K | jitLLM | 128,821 | the route's 600 s deadline stopped it at 108,544 tokens | — | — | |
| 128K | llama.cpp | 128,821 | 498.6 (258) | 16.76 | 30.59 (0.71) | |
| 256K | llama.cpp | 258,856 | 1,128.7 (229) | 14.73 | 28.41 (0.74) | |

The resident harness (`jitllm_dsv4_exec`, the same fast plan) ran the
128K prompt for the correctness check (below): prefill 812.5 s (159
tok/s), decode 7.4 tok/s (forced tokens, launch by launch).
llama.cpp's prefill with DSpark loaded was within 4% of without at every
depth (276, 269, 252, 225 tok/s). llama.cpp's own 8K numbers are the M3
baselines' at the b10964 pin (352 / 19.9); b11254 was not run at 8K.

### Qwen3.8 Flash Next (NVFP4)

| Depth | Engine | Prompt tokens | Prefill s (tok/s) | Decode tok/s | Speculative decode tok/s | Peak GiB (context) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32K | jitLLM (`spark-b`, 4,096-row chunks; MTP at context 32,768) | 31,743 | 15.5 (2,053) | 21.81 | 35.39 (MTP 2) | 100.6 (131,072); 79.1 with MTP (32,768) |
| 32K | Mia's vLLM (`spark`): deterministic, MTP off / MTP 3 | 31,743 | 19.0 (1,673) / 18.4 (1,729) | 24.62 | 37.25 (0.45) | 102.4 / 100.7 (262,144 pool) |
| 64K | jitLLM | 64,110 | 46.9 (1,367) | 17.10 | refused | |
| 64K | Mia's vLLM | 64,110 | 32.9 (1,947) / 33.9 (1,891) | 24.12 | 40.45 (0.50) | |
| 128K | jitLLM | 128,799 | 149.9 (859) | 11.90 | refused | |
| 128K | Mia's vLLM | 128,799 | 67.5 (1,909) / 69.2 (1,862) | 23.80 | 48.65 (0.69) | |
| 256K | jitLLM (context 262,144, 2,040-row chunks; the `256k-r` prompt) | 258,633 | 531.4 (487) | 6.83 | refused | 101.2 (262,144) |
| 256K | Mia's vLLM | 258,702 | 145.7 (1,775) / 149.6 (1,729) | 23.65 | 37.71 (0.44) | |

jitLLM's 32K row is the warm run (the first attempt's included the
model's 5.6 s page-in: 1,491 tok/s); its 64K and 128K rows followed a
warm request in the same process. Mia's acceptance is vLLM's accepted ÷
drafted tokens; the chat route does not report jitLLM's. Mia's 128K MTP
rate reflects a higher acceptance on that prompt's answer.

## Where the time goes

`nsys` (`--trace=cuda`) over the resident harnesses, which run the same
fast plan and kernels as the runtime, launch by launch: one run per depth
with a prompt of whole chunks and 2 decode steps
([profile.py](profile.py) splits the kernels at each chunk's logits copy).
Device busy time of the prompt's last prefill chunk (DeepSeek 2,048 rows,
Qwen3.8 4,096) and of the last decode step, by operation family, on
`spark` (2026-09-29):

**DeepSeek V4 Flash**, chunks at 6,144 / 28,672 / 61,440 tokens of
context:

| ms | Prefill chunk 8K | 32K | 64K | Decode step 8K | 32K | 64K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash attention | 1,289 | 4,034 | 7,856 | 2.8 | 7.6 | 15.0 |
| Concatenating the window cells and compressed rows into K (and copies) | 42 | 94 | 171 | 3.2 | 13.0 | 27.3 |
| Lightning indexer scoring (and its Hadamard) | 128 | 439 | 887 | 0.2 | 0.5 | 0.8 |
| Indexer top-k (GGML's radix select) | 9 | 41 | 82 | 0.6 | 0.5 | 0.5 |
| Sparse-index prep, masks and fills | 9 | 24 | 44 | 0.7 | 1.0 | 1.3 |
| Everything else (weights, MoE, norms, hyper-connections, compressors) | 3,228 | 3,300 | 3,291 | 47.8 | 49.8 | 50.7 |
| **Total** | **4,704** | **7,932** | **12,331** | **55.2** | **72.5** | **95.7** |

The lightning indexer's own share, split: scoring
2.7% / 5.5% / 7.2% of the prefill chunk at 8K / 32K / 64K and 0.4% /
0.7% / 0.8% of the decode step; its top-k 0.2% / 0.5% / 0.7% and 1.1% /
0.7% / 0.5%; its data movement nothing separate (its keys are read in
place from the state by the scoring kernel; the gathers and masks after
the top-k belong to attention, the "sparse-index prep" row).
From 8K to 64K the decode step grows 40.5 ms: 60% the K concatenation,
30% flash attention, 1% the indexer. The prefill chunk grows 7.6 s: 86%
flash attention, 10% the indexer. The weights' share is flat. The cause
is structural: jitLLM's DeepSeek state keeps a window-cache cell per
position (`swa_full`, as llama.cpp's library default and the exact-mode
oracle have it), and every layer's attention concatenates all of those
cells (43 layers) with its compressed rows into one K and attends over
them under a mask, so the work is O(context) per token in every layer,
though only the last 128 positions and 512 selected compressed rows
contribute. llama-server runs with `swa_full` off (a ring of window plus
micro-batch cells), and #28770/#29298's sparse flash attention gathers the
selected cells, so its per-token work is flat except the indexer.

**Qwen3.8 Flash Next**, prefill chunks at 4,096 / 28,672 (2,048 rows) /
57,344 tokens, decode at 8,195 / 30,723 / 61,443:

| ms | Prefill chunk 8K | 32K (2,048 rows) | 64K | Decode step 8K | 32K | 64K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash attention (dense, masked to the 2,051 selected cells) | 157 | 306 | 1,268 | 1.0 | 3.6 | 6.7 |
| QSA selection: jitLLM's device select (to 8,192 blocks) | 16 | 42 | — | 0.4 | 1.5 | — |
| QSA selection: GGML fallback (scores expanded to cells, ReLU, adds, transposes, radix top-k, masks) | — | — | 1,130 | — | — | 7.3 |
| Pooling the indexer's raw keys into blocks (gather) | 0.5 | 1.8 | 73 | 0.3 | 1.3 | 3.0 |
| Gated DeltaNet (lanes, convolution, norm and gate) | 168 | 85 | 170 | 1.2 | 1.3 | 1.4 |
| n-gram rows (the lookup kernel; the reads are host-side) | 4.1 | 2.3 | 4.1 | 0.01 | 0.01 | 0.01 |
| Everything else (weights, MoE, hyper-connections) | 1,194 | 739 | 1,175 | 38.2 | 39.6 | 39.2 |
| **Total** | **1,540** | **1,176** | **3,820** | **41.1** | **47.3** | **57.6** |

The Gated DeltaNet layers and the n-gram rows are flat (O(1) in depth, as
the architecture promises). Growth from 8K to 64K: decode +16.5 ms (flash
attention 5.7, the selection 6.9, block pooling 2.7); prefill +2.3 s per
4,096 rows (flash attention 1.1 s, the selection 1.1 s). Below 32,768
cells jitLLM's device select is cheap and only attention and pooling
grow; past it the GGML fallback also makes the indexer's scores
cell-sized ([n_kv, rows] F32: the RE-037 tensors) and builds both masks
on the host for every chunk (host time, not in these device times).

**Qwen3.8 with MTP, through the runtime** (its decode graphs, timed per
graph launch by `nsys`'s graph trace; the 8K and 32K prompts, context
8,704 and 32,768; a depth-2 draft then a batched verify each step; `spark`):

| Mean device time per launch, ms | 8K | 32K | Growth |
| --- | ---: | ---: | ---: |
| Plain decode step (MTP off) | 37.8 | 45.0 | +19% |
| Verify (the target over the anchor and drafts) | 48.5 | 55.2 | +14% |
| One drafter pass (its one QSA layer) | 5.1–5.4 | 6.2–6.6 | +21% |
| Speculative decode, tok/s (under `nsys`) | 39.1 | 34.8 | −11% |

The drafter grows like the target: its QSA layer has the same dense
attention and per-step pooling, and past 32,768 cells it has no selection
at all (gap 2.1). The verify grows by about as much as a plain step
(6.7 ms against 7.2), so speculation keeps its ratio but not the rate.

### What the architectures allow

Per token, what stays O(1) in the context and what cannot:

| | O(1) in depth | O(n) by design | Its cost at 64K / 128K / 1M (computed) |
| --- | --- | --- | --- |
| DeepSeek V4 Flash | the window (128 positions), the CSA attention over the indexer's top 512 compressed rows, the compressors' rings, MoE, hyper-connections | the lightning indexer's scoring over every compressed row (64 heads × 128 dims, F16 keys, one row per 4 positions, 21 layers: 1,344 B a token of context), and HCA's dense attention over one row per 128 positions (20 layers: 160 B a token) | indexer keys read per decode step 88 MB / 176 MB / 1.41 GB, HCA 10 / 20 / 160 MB, against about 10 GB of weights a token (the decode step's ~48 ms at ~205 GB/s, dsv4-decode): +1% / +2% / +16% |
| Qwen3.8 Flash Next | attention over the 2,051 selected cells, Gated DeltaNet's recurrent state (3 MiB a layer), the n-gram rows, MoE | the QSA indexer's scoring over every block of 4 cells (4 query heads, one 128-dim key a block, 12 layers: 1,536 B a token as F32 pooled keys, 768 in BF16; today jitLLM reads every cell's raw F32 key, 6,144 B, and pools them each step) | 100 / 200 MB per step, 400 MB at its 262,144 maximum (F32 pooled keys), against about 7.5 GB a token: +1.3% / +2.7% / +5.3% |
| Qwen3.8's MTP drafter | the same as one QSA layer | its one QSA layer's scoring | 1/12 of the target's |
| Dense attention (the M2 Qwen2 fixture) | — | reading every cell's K and V each step | exactly linear; "flat" is impossible on the exact path, and the lever is a quality mode (KV compression), not measured here |

In prefill the indexers are compute: DeepSeek's scoring at 64K is 5.8
TFLOP per 2,048-row chunk (computed), which the WMMA kernel runs at about
6.6 TFLOP/s (887 ms); as a batched tensor-core GEMM it would be tens of
milliseconds, about 1% of a flat chunk, and about 16% at 1M. So "flat" in
practice: DeepSeek within about 2% to 128K and about 15% at 1M; Qwen3.8
within about 3% at 128K and 5% at its maximum (half that with BF16 block
keys). Mia's vLLM, 6% slower at 256K than at 8K, is on that line.

## Correctness at depth

With the resident harnesses (the runtime's fast plan and kernels) on
`spark`, [judge.py](judge.py) in the pinned PyTorch image. Greedy: jitLLM
teacher-forced on the oracle's 512 greedy tokens after the oracle's own
prompt IDs, its argmax at each step against the oracle's token, a
difference passing as a near-tie when the oracle's log-probability margin
between the two is under the bound. **The bound was recorded before each
model's first comparison:** the 99th percentile of the top-two margin's
move between jitLLM's default fast plan and its reference form
(`--exact`), forced on the same tokens at 32K (dsv4-decode's rule).
Perplexity: the second half of one window of the book, against
llama-perplexity (DeepSeek) and vLLM's prompt log-probabilities (Qwen3.8),
on the oracle's own token IDs.

| Check | DeepSeek V4 Flash vs llama.cpp b11254 | Qwen3.8 vs Mia's vLLM (deterministic) |
| --- | --- | --- |
| Near-tie bound (p99 fast vs exact, 512 steps at 32K) | 1.24 (p50 0.17, max 1.56; 498/512 argmax equal) | 1.47 (p50 0.19, max 2.96; 495/512) |
| Greedy at 32K | 500/512 equal, 11 near-ties (oracle margin ≤ 1.02), **1 outside: step 249**, margin 2.62 (the same in a second run; the exact form passes, 498 equal + 14 near-ties) | 474/512 equal, 38 near-ties (≤ 1.13), 0 outside: **pass** |
| Greedy at 128K | 508/512 equal, 4 near-ties (≤ 0.46), 0 outside: **pass** | 482/512 equal, 30 near-ties (≤ 1.00), 0 outside: **pass** |
| Perplexity at 32K (16,383 tokens scored) | 1.8516 against 1.8528 (−0.1%): **pass** | 1.4352 against 1.4656 (−2.1%): **pass** |
| Perplexity at 128K (65,535 scored) | not run (stopped past 64K) | 3.9467 against 4.0042 (−1.4%): **pass** |
| Retrieval, through the runtime | passes at 32K and 64K (the first prompts' notes, quoted in the answer); deeper not run | passes at 32K, 64K, 128K and 256K (`-r` prompts) |
| The same run twice, bit for bit | **no**: two runs of the 32K forced prompt differ from the first step (largest logit difference 6.13; the margin moves by p99 0.93 between them): RE-031's radix top-k in the lightning indexer | **yes** at 32K (jitLLM's own device selection, ties by cell); past 32,768 cells the GGML fallback is not repeatable (RE-031; not re-run here) |

Step 249 on DeepSeek is like dsv4-decode's step 93: the fast plan's
margin there moves further than the bound on one token, repeatably, where
the reference form agrees with the oracle; not diagnosed in this phase.
The DeepSeek bound itself includes RE-031's run-to-run noise (the fast
plan does not repeat), so it is looser than a repeatable engine's would
be. The comparators' own retrieval: llama.cpp quoted all three notes at
every depth (32K–256K) without DSpark, and at 64K–256K with it (at 32K
its 512 tokens ended first); Mia's vLLM quoted them at 32K–128K in its
deterministic launch and declined at 256K (the "passphrase" wording).

## Swap with a long saved context

At 64K, not the plan's 128K and maximum (runs past 64K stopped): `jitllm-runtime
swap-table` with both LLMs registered at `context = 65536`, plain, A
holding 61,440 tokens of the book, one first-use cycle each way, every
check of the M3 swap table on (A's restored state must hash as it left;
its 16 continued tokens and logits must equal the unswapped
continuation's). On `spark`, 2026-09-29:

| A (61,440 tokens) | A→B total | Spilled | B→A total | Restore (bytes read back) | Exact |
| --- | ---: | ---: | ---: | ---: | --- |
| DeepSeek → Qwen3.8 | 8.16 s (evict with spill 2.14) | 3.35 GB | 8.57 s | 0.29 s (the state within the 100.35 GB paged in) | yes |
| Qwen3.8 → DeepSeek | 8.68 s (evict with spill 1.47) | 2.13 GB | 7.54 s | 0.18 s (within 77.13 GB) | yes |

Every swap stays under the ~10 s goal at 64K, the continuation is exact,
and restore runs at the page-in rate (about 11.5 GB/s). Two findings: the
spill is the whole state region at the configured context, not the used
part (DeepSeek at 262,144 would spill 13.3 GB for any conversation: gap
5), and the pair peaked at 114 GiB with `MemAvailable` down to 2.97 GiB,
both models' states being mapped at once.

## Turn-to-turn reuse

A 3-turn coding session through each chat route (`s64k`: ~61K tokens of
files and a question, then two short follow-ups), greedy, 512 tokens a
turn, the client sending each answer back. Both models think by default,
and 512 tokens end inside the reasoning, so each reply's `content` is
short or empty. "Drop" sends back only the content (what clients do);
"keep" also sends `reasoning_content`. On `spark`, plain decoding.

| Engine, client | Turn 1: prompt, reused, prefill | Turn 2 | Turn 3 |
| --- | --- | --- | --- |
| jitLLM Qwen3.8, drop | 60,902, 0, 43.1 s | 60,949, **0**, 42.9 s | 61,113, **0**, 43.8 s |
| jitLLM Qwen3.8, keep | 60,902, 0, 43.4 s | 61,462, 61,413, **0.27 s** | 62,007, 61,973, **0.23 s** |
| jitLLM DeepSeek, drop | 61,115, 0, 247.6 s | 61,152, **0**, 246.7 s | 61,173, **0**, 246.9 s |
| llama.cpp DeepSeek, drop (prompt cache on) | 61,115, 0, 231.5 s | 61,152, 61,111, **0.59 s** | 61,173, 61,148, **0.49 s** |

jitLLM reuses a conversation only when the re-rendered prompt extends
everything the state holds. With the reasoning dropped the re-rendered
history leaves out the last turn's reasoning, which the state holds, so
jitLLM clears it and prefills the whole prompt again: about 4 minutes a
turn for DeepSeek at 64K. llama.cpp keeps the longest common prefix
(61,111 of 61,152 tokens) and prefills only the rest. When the client
sends the reasoning back, the rendered prompt extends the state and
jitLLM prefills only the new tokens.

## Memory and the guard's margin

Peak memory (drop in `MemAvailable`) against the comparators at the same
configured context:

| Model, context | jitLLM, GiB | Comparator, GiB | Ratio |
| --- | ---: | ---: | ---: |
| DeepSeek, 262,144, plain | 116.0 | 95.8 (llama.cpp) | **1.21×** |
| DeepSeek, DSpark | 110.8 at 65,536 (the guard refuses above 143,360) | 107.1 at 262,144 | ≥ 1.04× (at a quarter of the context) |
| Qwen3.8, 131,072 (jitLLM) / 262,144 (Mia's pool) | 100.6 | 102.4 | 0.98× |
| Qwen3.8, 262,144 | 101.2 | 102.4 (deterministic) / 100.7 (MTP 3) | 0.99–1.00× |
| Qwen3.8, 32,768 with MTP | 79.1 | 100.7 (MTP 3, its 262,144 pool) | 0.79× (not like for like) |

The guard (`kUncountedMargin`, 4 GiB) holds back room for what the
catalog does not count. Measured here, peak minus the budget the runtime
logged:

| Run | Budget, GiB (fixed) | Peak, GiB | Uncounted, GiB |
| --- | ---: | ---: | ---: |
| Qwen3.8, 32,768, MTP | 75.27 (3.87) | 79.06 | 3.8 |
| DeepSeek, 65,536, DSpark | 106.34 (5.90) | 110.84 | 4.5 |
| DeepSeek, 262,144 (two runs) | 110.66 (20.36) | 115.95–116.01 | 5.3–5.4 |
| Qwen3.8, 262,144, 2,040 rows | 94.64 (24.74) | 101.16 | 6.5 |
| Qwen3.8, 131,072, 4,096 rows (two runs) | 91.70 (21.80) | 99.92–100.65 | 8.2–9.0 |

So the uncounted memory is not a constant: besides the decode graphs and
the driver's and cuBLAS's own memory (4.6–5.1 GiB at 8K, fix B's
review), it grows with the host-side chunk inputs a model builds before
they are staged, which the catalog does not see: Qwen3.8's fallback path
builds F16 and F32 masks of [n_kv, rows] on the host (3 GiB a chunk at
131,072 × 4,096). DeepSeek at 262,144 left `MemAvailable` within 0.2 GiB
of zero at its peak.

**Recommendation:** raise `kUncountedMargin` to 6 GiB now (it covers every
run here except Qwen3.8's fallback at 4,096 rows), and count the host-side
chunk inputs in each model's fixed memory (they are bounded by the same
planning that sizes the staging), so the margin is again a constant.
Phase 2's device-side masks and sparse attention remove most of those
inputs. With a 6 GiB margin DeepSeek's plain maximum would fall from
262,144 to about 250,000 and its DSpark maximum to about 118,000
(computed from the guard table), until the window ring shrinks its state.

## Fixed to measure

Two blockers, each fixed minimally with a unit test, and two aids (the
Spark check set ran on the final tree):

- **RE-037** (Qwen3.8 at 147K–262K): GGML takes the strides of [context,
  rows] tensors as 32-bit ints (flash attention's mask; `ggml_permute`,
  fixed upstream in #29227 after the pin). `Qwen38State` and
  `Qwen38MostRows` now bound a chunk so an F32 [context, rows] tensor stays
  under 2^31 bytes: at 262,144 the runtime caps `prefill_chunk` at 2,040
  rows and says so. `qwen38_test` checks the edges.
- **RE-038** (DeepSeek past ~52K): `CheckConcat` applied the per-row
  concat kernel's 65,535-channel grid to GGML's contiguous kernel too,
  which has no such limit, so the CSA layers' concatenation of window cells
  and compressed rows was refused. The check now follows `concat_cuda`'s
  dispatch; `ggml_ext_validate_test` checks both kernels at 66,560
  channels.
- The executor's refusal now names the refused node and its first
  operand with shapes and strides, which found RE-037's second case.
- The resident harnesses (`jitllm_dsv4_exec`, `jitllm_qwen38_exec`)
  prefill a `--prompts` prompt longer than `--max-rows` in chunks, so the
  teacher-forced checks run at 32K and 128K (benchmark code only; covered
  by those runs).

## Gap list for phase 2

Ranked; the owner's target is per-token cost flat with depth, fixed at
64K first. Each fix is on the exact default path (the same results as
the comparator's default configuration, not an approximation); effort is
a rough estimate of agent days, including tests.

1. **DeepSeek's per-token cost grows with the whole context** (*done in phase
   2*, [above](#phase-2-deepseek-flat-with-depth-2026-09-29): 1.1-1.3 by the ring,
   sparse attention and jitLLM's indexer; its scoring is a tensor-core pass per
   four rows, not a batched GEMM, and the selection is its own kernel after it)
   (prefill 463
   → 230 tok/s and decode 22 → 10.8 tok/s from 8K to 64K, against
   llama.cpp's nearly flat 286 → 229 and 18.8 → 14.7 from 32K to 256K).
   Cause, from the profile: the full-size window cache, concatenated with
   the compressed rows into one K in every layer and attended under a
   mask. Fix plan:
   1. *The window cache as a ring* of window plus chunk rows (what
      llama-server runs; `swa_full` stays the exact mode's):
      removes the concatenation's O(n) copies (60% of the decode growth)
      and the dense window attention (most of the rest and 86% of the
      prefill growth); state 50,912 → 6,880 B a token, so 1M fits beside
      the weights and DSpark. Touches the state layout, the chunk inputs,
      the graph, DSpark's verify snapshots and rollback, spill and restore.
      **2–3 days.**
   2. *Attention over the gathered selection:* CSA layers attend the ring
      plus the indexer's top 512 rows gathered by index through flash
      attention's sparse path (#28770, #29298), with no concatenation of
      every compressed row and no n/4-wide mask; HCA layers read their
      n/128 rows in place. **1–2 days** (a port, or the pin bump).
   3. *The lightning indexer, exact and fused:* the model's shapes are 64
      heads × 128 dims, a Hadamard-rotated query, F16 keys (as llama.cpp
      caches them), one key per 4 positions in 21 layers, the top 512
      kept. One pass scores q·k over block-contiguous keys (coalesced
      reads), applies the heads' weighted ReLU sum and keeps a streaming
      top 512 with ties broken by index, which is deterministic and closes
      RE-031 for DeepSeek. Prefill scores as a batched tensor-core GEMM
      (5.8 TFLOP a 2,048-row chunk at 64K, run today at 6.6 TFLOP/s:
      887 ms); a DSpark verify's rows share one pass. **2–3 days.**
   4. Expected after 1–3: decode within about 1% of 8K at 64K and about
      2% at 128K, prefill within a few percent; the floor is the
      indexer's scoring and HCA's n/128 rows (+16% of the weights' bytes
      at 1M, computed above).
2. **Qwen3.8's per-token cost grows with the whole context, and MTP stops
   at 32K** (prefill 2,053 → 487 tok/s and decode 21.8 → 6.8 from 32K to
   256K, against Mia's 1,729–1,947 and 23.6–24.6). **Done in phase 2**
   ([below](#phase-2-qwen38-flash-next-flat-with-depth)). Fix plan:
   1. *Selection on the device at any depth:* a tiled, deterministic
      select over blocks past 8,192 (TensorFold #93's technique, ties by
      index) that emits the selected cells: removes the GGML fallback
      (cell-sized scores, ReLU, adds, transposes, radix top-k, host-built
      masks: RE-031's and RE-037's tensors, about 40% of the decode growth
      at 64K), and lets the MTP drafter speculate at any depth. **1–2
      days.**
   2. *Sparse flash attention over the 2,051 selected cells* by gather,
      not dense masked attention over every cell (#28770's technique;
      about 35% of the decode growth and half of prefill's). **1–2 days.**
   3. *Pooled block keys cached* when a block completes (the pooling,
      norm and rotation computed once, the same arithmetic), laid out per
      block for coalesced reads: removes the per-step gather of every raw
      indexer key (about 16% of the decode growth). **1 day.**
   4. Prefill scoring as a batched GEMM with the streaming top-k; a
      verify's rows share one pass. Expected: within about 1% at 64K, 3% at
      128K and 5% at the maximum (the floor: block scoring).
3. **The chat route's 600 s request deadline** (D-097's `kDeadlineMs`):
   at DeepSeek's 220–230 tok/s a 128K prompt needs 9–10 minutes (the
   route stopped it at 108,544 of 128,821 tokens), 256K about 19, 1M
   hours even after fix 1; a client that does not resend the request
   fails (a resend resumes, fix B). To evaluate in phase 2: a deadline
   that scales with the prompt (a floor prefill rate), or none for a
   stream that is sending keepalives and making progress. **Hours, and a
   D-097 amendment.** *Closed 2026-09-29: both, as a progress watchdog
   (D-097's owner note,
   [runtime-serving.md](../../runtime-serving.md#progress-and-deadlines)).*
4. **DeepSeek's memory and maximum:** 1.21× llama.cpp's peak at 262,144;
   DSpark only to 143,360; 1M impossible as built; the configuration
   refuses more than 262,144. Fix 1.1 is the lever; then the
   configurable ceiling (a configuration-semantics change: a decision).
5. **State reserved at the ceiling:** each registered model maps its
   state at its configured context for the node's life, a swap spills
   the whole region (12.4 GiB for DeepSeek at 262,144 whatever the
   conversation), and two long-context models do not register together
   under the guard. Lever: state extents grown with the conversation and
   spilled only as used (the state is already whole extents; the pinned
   places of D-090 must hold). **3+ days.**
6. **Turn reuse:** with a client that drops the reasoning (the common
   case), every turn at 64K re-prefills about 61K tokens: 43 s on Qwen3.8
   and 4 min on DeepSeek, against llama.cpp's 0.5–0.6 s
   ([above](#turn-to-turn-reuse)). Lever: reuse the longest common prefix,
   not only an extension. The attention caches, the indexer keys and the
   compressed rows are append-only, so truncating to a prefix is a
   position change. Qwen3.8's recurrent and convolution state, and
   DeepSeek's compressor rings, cannot be rolled back, so they need
   checkpoints where a later turn may diverge: at the end of each
   rendered user message (before the assistant's reasoning). **2–3 days.**
7. **The guard's margin** (*done in phase 2*): 6 GiB and the host-side inputs counted
   ([above](#memory-and-the-guards-margin)). **Hours.**
8. **Repeatability (RE-031):** DeepSeek at 32K and Qwen3.8 past 32K
   (both *closed in phase 2*).
9. **The prefill chunk at depth:** [n_kv, rows] tensors force narrower
   chunks at depth (Qwen3.8 2,040 rows at 262,144, RE-037); with
   gathered attention and device selection they go away and the chunk can
   stay wide (Qwen3.8's is 4,096 rows at every context since phase 2).

**Future quality and performance modes** (never the default; each off,
behind a per-alias flag, and each needing a quality check against the
exact indexer): hierarchical selection (coarse super-blocks, then the
exact top-k within the chosen ones); reusing a step's selection for the
next few decode steps with a periodic full rescoring; approximate
nearest-neighbour search over the index keys; and, for dense-attention
models, KV compression.

## Phase 2: Qwen3.8 Flash Next flat with depth

Gap 2's fix plan, on the default (fast) graph, 2026-09-29, `spark-b`
(GB10, driver 580.178.04), the phase-2 working tree on `09f9015`. The
reference (`--exact`) and unfused graphs are unchanged.

### What changed

- **Block keys cached** (`jitllm.qsa.pool`): when a chunk completes a
  block of 4 cells, its raw indexer keys are pooled, normalized, rotated
  (the reference form's arithmetic) and kept in the state as BF16, 256
  bytes a block (`Qwen38StateTensor::kIndexerBlocks`, +768 B a token over
  12 layers; the drafter's likewise). Nothing re-pools every block each
  step, and no host table names the blocks. A verify saves the block keys
  it completes with its cells, so a rejected row's are restored.
- **Selection on the device at any depth** (`jitllm.qsa.topk`,
  TensorFold #93's technique): each complete block's score is its four
  heads' relu scores (the query rounded to BF16 times the block key, F32
  sums; BF16 tensor-core products past 16 rows) plus build_qsa_top_k's
  rule (the token's own incomplete block always kept, later cells never
  visible); then per token a byte-wise radix select for the 2,051st
  cell's order-preserving key in tiles of 8,192 blocks, ties to the lower
  cell, and once more over the tiles' candidates. Out: each token's kept
  cells, ascending. It replaces the GGML top-k fallback past 8,192 blocks
  and its host masks, closes RE-031 for this graph, and lets the MTP
  drafter select at any depth.
- **Attention over the kept cells alone** (`jitllm.qsa.attn`, #28770's
  gather): a warp a token's KV head, its 12 query heads one m16n8k16 tile,
  16 cells gathered at a time (K and V double-buffered), online softmax;
  decode and verify rows split their cells into shares combined in order.
  F16 products, F32 sums, the query scaled before rounding, as GGML's MMA
  kernel.
- **The MTP drafter** runs the same builder, so its QSA layer gets all
  three; it is no longer refused past 32,768.
- **No tensor of every cell by every row** in the fast graph, so RE-037's
  chunk bound applies only to the reference and unfused graphs (the graph
  builder refuses theirs past it): the runtime's chunk is 4,096 rows at
  every context. The fast graph's largest host input is now the causal
  mask of the widest chunk that does not select ([2,048 cells, rows]),
  which the runner's staging is sized for (the deep speculation check
  found it missing: a chunk ending at 2,048 cells was refused).

### Through the runtime, before and after

The chat route, as phase 1 ran it (`longctx.py`, 512 greedy tokens, the
same prompts), `context = 262144`, plain and with MTP depth 2, one run a
depth, the final build (13:12–13:25). 256K ran once, on an intermediate
build whose selection was slower (the same bits; of its 2,404 ms prefill
chunk at 256K the selection took 710 ms, where the final build's takes 53
ms at 64K and 78 ms at 128K); the owner asked to wrap up before a rerun,
so 256K's prefill is understated. Mia's numbers are phase 1's (8K: the M3
baselines').

| Depth | Prefill tok/s, before → after (× Mia) | Plain decode tok/s, before → after (× Mia) | MTP decode tok/s, before → after (× Mia's MTP 3) |
| --- | --- | --- | --- |
| 8K (7,586 tokens) | 2,320 → 2,314 (1.10×) | 27.4 → 26.8 (1.07×) | 40.1–42.5 → 42.3 (1.12×) |
| 32K | 2,053 → 2,375 (1.37×) | 21.8 → 26.5 (1.08×) | 35.4 → 39.7 (1.07×) |
| 64K | 1,367 → 2,359 (1.21×) | 17.1 → 26.0 (1.08×) | refused → 42.6 (1.05×) |
| 128K | 859 → 2,306 (1.21×) | 11.9 → 25.5 (1.07×) | refused → 44.4 (0.91×) |
| 256K (258,702 tokens; an intermediate build, below) | 487 → 2,178 (1.23×) | 6.8 → 24.4 (1.03×) | refused → 36.3 (0.96×) |

**The slope** (computed from these rates): before, a plain decode step
cost 0.39 ms more per 1K tokens of context (36.5 ms at 8K, 58.5 ms at
64K); after, 0.021 ms from 8K to 64K (37.3 → 38.5 ms) and 0.016 ms to 128K
(39.3 ms), about 1/20 as much, and 0.014 ms to 256K. The prompt's mean
cost per token rose 70% from 8K to 64K before; after it is −2% at 64K,
+0.4% at 128K and +6% at 256K. Every plain rate is at least Mia's at the
same depth. MTP's rate follows its acceptance, which the route does not
report: it is below Mia's MTP 3 at 128K (0.91×) and 256K (0.96×).

**MTP at depth, diagnosed** (the review, `spark-b`, the final build,
`jitllm_qwen38_spec --check greedy` on the 8K and 128K chat prompts; each
answer's first 32 tokens, so acceptance is coarse): a speculative step
costs the same at depth. At MTP depth 2 a step is 60.4 ms at 8K and 60.7
ms at 128K (the drafter's passes 7.8 → 8.3 ms, the verify 52.0 → 51.8
ms); acceptance 0.55 at 8K and 0.69 at 128K (by position 0.77 / 0.62),
2.07 and 2.38 tokens a step. This short-answer diagnosis identifies a
depth cap rather than a cost that grows with context: that answer accepts
well (Mia's MTP 3
accepted 0.69 of its drafts, 3.07 tokens a step at about 63 ms, from its
phase 1 rate), and depth 2 takes at most 3 tokens a step. MTP depth 3
at 128K: 3.10 tokens a step, 72.1 ms (drafter 11.8, verify 59.7), 43.0
tok/s against depth 2's 39.3 (+9%) through the harness. What is left is
the verify's cost per row (about 6 ms a row at 3 rows, 8 ms for the
fourth, the routed experts each row adds; Mia's MTP 3 step costs about
1.5× its plain step, jitLLM's depth-3 step 1.8×). At 256K Mia accepted
only 0.44, so there the gap is not acceptance: 256K ran on the
intermediate build, whose plain step was 5% slower (44.7 against 42.6
ms), and was not re-run. Levers: the draft depth chosen by acceptance
(depth 3 when it runs high), a cheaper verify row, and the draft head's
selected vocabulary. Mia's actual head is a curated 47,172-row BF16 product,
while jitLLM uses the first 65,536 rows; the
[draft-head study](../qwen38-draft-head/README.md) measures that distinction
at fixed depth. The 32-token diagnosis alone does not isolate it.

### Where the time goes now

`nsys` over the resident harness (the runtime's kernels, launch by
launch), the prompt's last 4,096-row chunk and the last decode step, the
final build; before: phase 1's table (the 64K chunk's rows were at 57,344
tokens, as here).

| ms | Prefill chunk 8K | 64K | 128K | Decode step 8K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention over the kept cells (`QsaAttnKernel`, combine) | 172.4 | 176.4 | 202.4 | 0.35 | 0.39 | 0.49 |
| Indexer scoring (queries to BF16, block scores) | 2.8 | 22.8 | 46.5 | 0.15 | 0.39 | 0.95 |
| Selection (tile radix select, merge) | 12.1 | 53.1 | 77.7 | 0.32 | 0.60 | 0.68 |
| Block keys (`QsaPoolKernel`) | 0.1 | 0.1 | 0.1 | 0.02 | 0.02 | 0.02 |
| QSA query and key prep | 11.7 | 11.5 | 11.6 | 0.10 | 0.10 | 0.12 |
| Gated DeltaNet | 166.5 | 167.1 | 167.6 | 1.26 | 1.28 | 1.35 |
| Everything else (weights, MoE, hyper-connections) | 1,170.8 | 1,188.7 | 1,185.8 | 38.12 | 38.06 | 39.03 |
| **Total** | **1,536** | **1,620** | **1,692** | **40.3** | **40.8** | **42.6** |
| *Phase 1's total* | *1,540* | *3,820* | — | *41.1* | *57.6* | — |

From 8K to 128K a decode step grows 2.3 ms, of which the QSA work is 1.3
(the indexer's scoring 0.8, the selection 0.36, attention 0.14) and the
rest run-to-run spread in the weights' products (38.1 at 64K); a prefill chunk
grows 156 ms (+10%): selection 66, scoring 44, attention 30. The
scoring's growth is what the architecture requires (every block, every
row); in prefill it is not yet at its floor (the keys, 4 bytes a block a
row, are written and read back through memory: 1 GB a layer at 256K), the
selection's is its constant, and attention's is the gathered rows falling
out of L2 as the cache outgrows it. An intermediate build with a slower
selection measured the same shape at 256K: prefill chunk 2,404 ms, of
which the scoring 95 and attention 225; decode 44.7 ms.

Levers, not taken here (the owner asked to wrap up): scoring fused with a
streaming first-tile select so the keys never leave the chip; the tile
select at 16-bit keys with an exact re-check; attention over the union of
a few neighbouring rows' cells (their selections overlap), which would
cut its gathers and its 8K cost too.

### Correctness

The resident harness (`jitllm_qwen38_exec`, the runtime's kernels) on
`spark-b`, [judge.py](judge.py), the phase-1 corpus and Mia's recorded
deterministic outputs. The final build's logits equal the first measured
build's bit for bit (32K forced run), so these hold for it.

| Check | Result |
| --- | --- |
| Near-tie bound, recorded first (p99 of the top-two margin's move, fast vs `--exact`, 512 forced steps at 32K) | **1.765** (p50 0.21, max 6.06; 494/512 argmax equal); phase 1: 1.47 |
| Greedy vs Mia's vLLM at 32K | 477/512 equal, 35 near-ties (oracle margin ≤ 1.125), 0 outside: **pass** (phase 1: 474 + 38) |
| Greedy vs Mia's vLLM at 128K | 486/512 equal, 26 near-ties (≤ 1.00), 0 outside: **pass** (phase 1: 482 + 30) |
| Perplexity at 32K (16,383 scored) | 1.4473 against 1.4656 (−1.2%): **pass** (phase 1: −2.1%) |
| Perplexity at 128K (65,535 scored) | 3.9476 against 4.0042 (−1.4%): **pass** (phase 1: −1.4%) |
| Retrieval through the runtime (`-r` prompts) | all three codenames at 32K, 64K, 128K and 256K (258,633 tokens): **pass** |
| The same run twice, bit for bit | **yes** at 64K and 128K (512 steps each, max difference 0): RE-031 closed for this graph |
| MTP: forced rejections against a control (`jitllm_qwen38_spec --check forced`, `capital`, 160 tokens, context 8,704) | 85 steps, 57 with rejected rows: **0 states differ** from the control; 1 near-tie, 0 violations |
| MTP: rollback across a swap (`--check swap`, the same, the FP16 fixture as B) | 97 steps compared, **0 states and 0 of 160 tokens' logits differ**; B's logits `bb8ae5e7…` as recorded; graphs captured before the swap replayed after it |
| The same two checks past 32K (the 64K coding prompt, 64,110 tokens, context 65,536) | forced: 78 steps, 47 with rejected rows, **0 states differ** (the block keys included), 2 near-ties, 0 violations; swap: 93 steps, **0 states and 0 logits differ**, graphs replayed across the swap |
| Swap with 61,440 tokens saved (`jitllm-runtime swap-table`, both LLMs at 65,536, plain, first use) | Qwen3.8 as A: A→B 9.06 s, B→A 7.52 s (restore 0.18 s), **exact**; DeepSeek as A: 8.04 s / 8.85 s, **exact** (phase 1: 8.68 / 7.54 and 8.16 / 8.57) |

The bound's p99 is above phase 1's (1.47): the BF16 scores move near-tied
blocks in and out of the selection, which moves some steps' margins
further. Every oracle margin at a divergence stays within phase 1's bound
too (1.125 at most).

### Memory and the maxima

At `context = 262144`, plain, one model: the runtime reserves 11.09 GiB
(its workspace 2.73) where phase 1 reserved 24.74 (9.84) at 2,040-row
chunks, and the run's peak is 84.7 GiB (phase 1: 101.2). With MTP: 11.90
GiB, peak 87.2. **The MTP maximum is the target's configured 262,144**
(registered, and run with the 128K and 256K prompts); it was 32,768.

### Reproduce

On `spark-b`, beside phase 1's setup (`W` its directory; `W2` this
phase's, with `harness.sh` and `judge.sh` pointed at this tree's build):

```sh
# Correctness: forced runs (fast; --exact for the bound), the judge.
harness.sh qwen $W2/raw/hq-32k-fast 33280 4096 --prompts q32k.prompt.tsv --force q32k.force.tsv --generate 512
harness.sh qwen $W2/raw/hq-32k-exact 33280 4096 --exact --prompts ... --force ... --generate 512
harness.sh qwen $W2/raw/hq-128k-fast 131072 4096 --prompts q128k.prompt.tsv --force q128k.force.tsv --generate 512
judge.sh noise $W2/raw/hq-32k-fast $W2/raw/hq-32k-exact q32k --vocab 248320
judge.sh greedy $W/raw/qw-mia-det/qwen3.8-32k.json $W2/raw/hq-32k-fast q32k --vocab 248320 --bound 1.765
judge.sh repeat $W2/raw/hq-128k-fast $W2/raw/hq-128k-fast2 q128k --vocab 248320
harness.sh qwen $W2/raw/hq-ppl-128k 131072 4096 --ppl ppl-131072.ids
judge.sh ppl $W/raw/qw-mia-det/ppl-131072.nll.json $W2/raw/hq-ppl-128k/ppl.nll.f64 --ctx 131072
# Speed: the runtime at context 262144, plain and with MTP.
python3 longctx.py jitllm $W2/raw/final-plain 8k.json 32k.json 64k.json 128k.json --port 18140 \
  --runtime build/spark-native/src/runtime/jitllm-runtime --config qw-262144-off.toml --model qwen3.8 --retries 3
# The profile: whole 4,096-row chunks, then the split.
nsys profile --trace=cuda --sample=none --cpuctxsw=none --export=sqlite -o qw3-64k \
  jitllm_qwen38_exec --artifact ... --context 65536 --max-rows 4096 --prompts p64k.tsv --generate 3
python3 profile.py qw3-64k.sqlite 248320
# Speculation at depth: a prompts file whose one chat prompt is 64k.json's messages.
jitllm_qwen38_spec ... --prompts deep.json --only deep64k --context 65536 --tokens 160 --check forced
# MTP at depth: acceptance and a step's parts (the same file shape, 128k.json's messages).
jitllm_qwen38_spec ... --prompts rv-128k.json --only deep128k --context 131072 --check greedy [--draft 3]
```

Raw outputs, logs and traces stay on `spark-b` under
`~/.local/share/jitllm/m3lc2/`.

## Not run, and why

- **Every jitLLM rung past 64K after 11:00** (the owner, 2026-09-29: we
  know enough at 64K; fix the scaling first). Qwen3.8's 128K and 256K and
  DeepSeek's 128K correctness had run by then; not run: DeepSeek at 128K,
  256K and its maximum through the runtime, DeepSeek's perplexity at 128K,
  the 1M comparator run (llama.cpp at `-c 1048576`), Qwen3.8's repeat at
  128K, and the swaps with 128K and maximum saved context (a 64K one
  instead, [above](#swap-with-a-long-saved-context)).
- **DSpark past 64K and MTP past 32K:** DSpark's maximum is 143,360
  under the guard; the MTP drafter is refused past 32,768 (gap 2.1).
- **TensorFold** (optional cross-quantization information): not run; its
  long-context changes (PR #93) are open upstream, and the comparison that
  matters is same-format.
- **llama.cpp b11254 at 8K:** the 8K reference is b10964's (M3
  baselines).
- **jitLLM's speculation acceptance at depth:** the chat route does not
  report it, and `jitllm-runtime chat` takes its prompt as an argument
  (128 KiB at most), too small for 32K.

## Reproduce

On a Spark, with this directory at `~/.local/share/jitllm/m3lc/lc/long-context`
and `docs/experiments/fast-swap` beside it, the M3 model store, the pinned
PyTorch image, and llama.cpp at `8019dc563` cloned to `~/src/lc/llama.cpp`:

```sh
W=~/.local/share/jitllm/m3lc
IMG=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
# The prompts (the tokenizer files as corpus.json pins them), and the book.
curl -sSLo $W/corpus/pg2600.txt https://www.gutenberg.org/cache/epub/2600/pg2600.txt
sudo docker run --rm --network none --user $(id -u):$(id -g) --entrypoint python3 \
  -v $W/lc/long-context:/tools:ro -v ~/src/lc/llama.cpp:/repo:ro \
  -v ~/.local/share/jitllm/models:/models:ro -v $W/corpus:/corpus -v $W/prompts:/out $IMG \
  -I /tools/build_prompts.py --repo /repo --model qwen3.8 \
  --tokenizer /models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6/tokenizer.json --out /out \
  --rung 8k=8704 --rung 32k=32768 --rung 64k=65536 --rung 128k=131072 --rung 256k=262144 \
  --session s64k=65536 --retrieval 32k-r=32768 ... --ppl-source /corpus/pg2600.txt
# (DeepSeek: --model deepseek with DeepSeek's tokenizer.json; add --rung 1m=1048576.)

# llama.cpp (the image built from 8019dc563, pins.json), and its perplexity.
python3 longctx.py llama $W/raw/ds-llama-plain $W/prompts/deepseek/{32k,64k,128k,256k}.json \
  --image jitllm-llamacpp:b11254-cuda13 --model .../DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --args "-ngl all -fa on -c 262144 -np 1 --fit off -cram 0"   # DSpark: add -md ... --spec-type draft-dspark ...
python3 longctx.py llama-ppl $W/raw/ds-llama-ppl --image ... --model ... --text $W/prompts/ppl.txt \
  --ctx 32768 --ctx 131072 --args "-ngl all -fa on --fit off"

# Mia's vLLM (as the M3 baselines set it up), the oracle's launch, then MTP 3.
python3 longctx.py vllm $W/raw/qw-mia-det $W/prompts/qwen3.8/{32k,64k,128k,256k}.json --port 8888 \
  --start "cd .../mia && MTP_NUM_SPECULATIVE_TOKENS=0 VLLM_QSA_DET_TOPK=1 VLLM_MOE_DET_FINALIZE=1 exec ./start.sh" \
  --stop "cd .../mia && ./stop.sh" --ppl $W/prompts/ppl.txt --ctx 32768 --ctx 131072

# jitLLM through the runtime (a configuration naming one model, its context).
python3 longctx.py jitllm $W/raw/qw-jit $W/prompts/qwen3.8/32k.json ... --port 18140 \
  --runtime build/spark-native/src/runtime/jitllm-runtime --config qwen-131072.toml --model qwen3.8 --retries 3

# Correctness: the oracle's IDs as harness input, the forced runs, the judge.
judge.py inputs raw/ds-llama-plain/deepseek-32k.json hin/ds d32k
jitllm_dsv4_exec --artifact ... --out raw/hd-32k-fast --context 33280 --max-rows 2048 \
  --prompts hin/ds/d32k.prompt.tsv --force hin/ds/d32k.force.tsv --generate 512   # and --exact on
judge.py noise raw/hd-32k-fast raw/hd-32k-exact d32k --vocab 129280
judge.py greedy raw/ds-llama-plain/deepseek-32k.json raw/hd-32k-fast d32k --vocab 129280 --bound B
jitllm_dsv4_exec ... --ppl raw/ds-llama-ppl/ppl-32768.ids; judge.py ppl 1.8528 raw/hd-ppl-32k/ppl.nll.f64 --ctx 32768

# The profile: one nsys run per depth, then the split.
nsys profile --trace=cuda --sample=none --cpuctxsw=none --export=sqlite -o prof/ds-64k \
  jitllm_dsv4_exec ... --context 65536 --max-rows 2048 --prompts ds-64k.tsv --generate 3
python3 profile.py prof/ds-64k.sqlite 129280

# Phase 2: the step probe, in each plan's run (a full window), then the comparison.
jitllm_dsv4_exec ... --context 33280 --max-rows 2048 --prompts hin/ds/d32k.prompt.tsv \
  --force hin/ds/d32k.force.tsv --generate 251 --probe-step 249 [--exact on]
python3 probe_step.py FAST_OUT EXACT_OUT 82437 10386
# Speculation over a wrapping ring.
jitllm_spec_runner --dsv4-artifact DSV4 --drafter DRAFTER --prompts ../fast-swap/prompts.json \
  --out DIR --check forced --max-rows 128 --tokens 160 --margin 6.11
```

The judge requires complete, nonempty captures: every oracle step for
greedy, equal row counts for repeatability and the noise bound, and exactly
`context - 1` NLL values on each side of a perplexity comparison. It refuses
partial values, partial rows and non-finite logits or NLL values.
Repeatability mismatches return a failing exit status. Run its synthetic
regression tests in the same NumPy image with `python3 -B test_judge.py`
from this directory.

Long runs went through `tools/spark-job` (`start --gpu --steps`), after
checking free memory and that no other model process was on the Spark.
Raw outputs (every request's record, server logs, traces) stay on the
Sparks under `~/.local/share/jitllm/m3lc/`.
