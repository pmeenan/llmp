<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma3 internal three-owner departure and rejoin

Actual H8/no-softcap C3 attention and one representative C4→C3→C4 model
schedule pass their original-MMA and same-schedule stock gates. Native and stock
have 147/148 bitwise exact full heads and zero greedy differences over 144
scored transitions. Paused and catch-up peers retain exact initialized state
and cursors; device/full-head replay and checkpoint restoration are exact.

Ordinary serving still admits at most two slots. This internal control does not
accept the [earlier solo/C4 geometry differences](../gemma-h8-c4/README.md),
qualify unequal C3/C4 model reads, transfer cap 50 to C3, or establish continuous
HTTP batching, speed parity, peak memory or sustained behavior. No new quality
allowance or timing measurement is introduced. The prior C4 prefill and total
latency gaps remain open.

## Contract and operand proof

The host now admits canonical D256/H8/GQA2 one-query attention with three actual
and logical roots at offset zero and cap 0. It reuses the existing four-bank
launcher and unchanged floating kernel/grid. Cap 50 and bounded actual-root
reads retain their C2-only restrictions; partial H8, cohorts 8/12 and unsupported
head dimensions remain refused. Paired prefill remains limited to two owners,
256 total rows and 128 rows per owner, with unchanged state/checkpoint layout.

The graph joins three frontier heads and preserves independent K/V roots.
Unequal internal C3 descriptors use funded real zero-tail activations, as C4
does. Setup measures both proper mixed C3 long-owner counts, 1 and 2. Graph/plan
controls cover C3 and C4, exact and one-byte-short source/activation funding,
independent source roots and the fallback for unsupported multirow waves.
Descriptor/funding admission is separate from unequal model qualification.

The operand control compares three real roots with original packed physical
C3 MMA at cells 256 and 1024, including fresh queries/masks, eager and captured
replays, FP64 and scratch refusals. Both are bitwise exact. FP64 NMSE is
1.13217e−7 / 2.0108e−7 against the existing 5e−4 bound. Original and owner
occupancy are 1/1, with 48 blocks and 399,616 scratch bytes in each case. Legacy C4,
cap 50, bounded-root and Gemma4 controls also pass.

## Actual partial cohort model control

Four real independently funded roots use two explicitly duplicated corpus
identities: owners 0/2 use input 0, owners 1/3 input 1. These are two corpus samples,
not four. Each authenticated input contains 295 native/stock-tokenizer-matched
IDs. Context is 4096 per owner with F16 KV and four funded head rows. Pairs 0/1
then 2/3 prefill 256 prompt rows per owner, followed by three supplied scalar
rows to position 259. Both engines then execute the same supplied history:

1. Eight C4 waves advance all owners to 267.
2. Eight actual C3 waves advance owners 0/1/2 to 275 while owner 3 stays at 267.
3. Eight scalar rows catch owner 3 up to 275 while the other peers remain paused.
4. Sixteen C4 waves rejoin all owners, followed by four scalar tail rows each.

The native caller checks owner 3's initialized-state SHA and cursor across C3,
then the first three peers' SHA/cursors across catch-up. Inactive-owner waves
must refuse without publication or mutation. Each computed full head is written
once at its canonical owner/position index; delayed catch-up fills its own rows
without publishing stale peer heads. All 148 rows and 144 scored choices must be
present exactly once; four final rows are unscored.

Two fresh native runs repeat exactly. Within each run, GPU-token replay matches
all 144 choices, final heads and initialized state; Clear retains backing and
spill/checkpoint restoration preserves exact continuations. Each run executes
16 actual C3 waves, 16 catch-up rows and 32 rejoined C4 waves across its two passes,
with 156 GPU token publications. The isolated C3 phases bind 34 owner-attention
plans each (68 total); this is a plan-selection witness, not a replay kernel
count. `rejoin_equal` denotes exact full-head/GPU/restore behavior within this
schedule, never equality with solo or uninterrupted C4 arithmetic.

The unchanged d812/v0.6.0 original reference uses four per-sequence greedy
samplers, context 16,384 total / 4096 per sequence, batch/ubatch 256, F16 KV, flash
attention, `swa_full=false`, `kv_unified=false` and off-clock logical reset false.
There is no extra state-only synchronization. Its original sampler exports
full 262,208-value sampled-logit rows to the host. Metadata-only observations
independently witness actual Q/K/mask sequence counts 3 and 4. Both teacher runs
repeat bitwise exactly and their containers have checked absence receipts.

| Same-schedule native vs original stock | Result |
| --- | ---: |
| Exact full heads | 147/148 |
| Greedy / positive-margin / tie differences | 0 / 0 / 0 |
| Scored transitions | 144 |
| Relative conditional loss delta | −0.00302168% |
| Mean total variation | 8.3897444e−6 |
| Maximum raw logit difference | 0.0205717087 |

This passes the existing zero-difference and conditional-loss gates. The single
nonexact head remains explicit. `quality.json` deliberately records
`concurrency_acceptance=false`; no solo/batch waiver follows from this result.

## Reproduction and measured binding

Use the [existing approved pin](../gemma3-execution/pins.json), retained original
CUDA image/library closure and external actual prompt texts. Preserve the two
295-ID histories and `inputs.json`; keep heads/state/logs outside Git. In a new
private output directory `OUT`, run the [native caller](../../../benchmarks/gemma3_batch_probe.cc)
twice as:

```sh
jitllm_gemma3_batch_probe ARTIFACT IDS0 IDS1 OUT/native-departure1 own departure
jitllm_gemma3_batch_probe ARTIFACT IDS0 IDS1 OUT/native-departure2 own departure
```

Run [the stock caller](../gemma3-execution/llama_batch_probe.cc) twice inside the
original image as `MODEL IDS0 TEXT0 IDS1 TEXT1 NEW_OUT departure`, named
`stock-departure1/2`. Use the historical retirement helper's checked name/label
and private CID files. The [aggregate helper](../gemma3-execution/analyze_batch.py)
runs `departure-own OUT MODEL_JOB_LOG` before stock, then
`departure-quality OUT MODEL_JOB_LOG` after both retirements. It authenticates
inputs, complete finite heads, row/choice mapping, counters, repeats and stock
shape witnesses before applying the existing strict quality gate.

Measured on Spark B (GB10, driver 580.178.04, SDK `aarch64-c09daba6ac31edee`,
`sm_121`) on 2026-10-07, true parent `20e0990`; operator prerequisite parent
`21c64d6`. Check1 and model1 finish official DONE0. XML records 34 passing executions:
15 prerequisite controls, then 19 current host controls (eight graph and
11 plan). The graph control was renamed when extended from C4 to C3/C4, so
those executions are not a distinct-test count. The model
job's nine steps build narrowly, authenticate retained model/input identities,
freeze native own behavior before stock, then apply strict quality. Every build
and inference uses installed GPU600/stop-on-fail supervision and official wait;
source sync is checksum-based with an empty dry-run. No full suite ran.

| Identity | SHA-256 |
| --- | --- |
| Approved source, 2,526,080,992 bytes | `ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a` |
| Prepared artifact | `8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb` |
| Input 0, 295 i32 | `c37e404d378484e6dbf4da9a7f7e812105b569eac9e8323e9dd49e47d57dc470` |
| Input 1, 295 i32 | `1e7793632795175c98b7835511b66cadbaa48312a87fdd297e1dd2186258850e` |
| Native caller | `008670496890eef9ac57e5aa836abb9858048e917e3d133eadf1d734f9f17b49` |
| Stock caller | `68336b8942daee7f25b1b142f17fb424bf485a09c8a88153bd2c4b3b340b99b5` |
| Build receipt | `874aaf7a5967cfbe91054e0d8fc1a0630f952e0e8eb54b831d09719f1e08ce89` |
| Complete measured model source inventory | `1dd59944cfa4e1a8b552473d3840e66f6715a294bd4ae27e8217d20b765ca310` |
| Strict quality aggregate | `817d9e3e02f21b1ebbbacc73e11db955c2a12151f66d95848233e996be5b20ef` |
| Native own aggregate | `fd6203dbce17af5768ad207387710cd1490c97d90e12c47da0d851f2b300f59e` |

Task-entry TensorFold HEAD is `041d14a94e951834470fd514ed33e65b8be1059a`,
version 1.0.0, observed 2026-10-07. Its documented GB10 CUDA recipe qualifies
Nemotron and supplies no matching Gemma3/Gemma2 CUDA GGUF comparator.
Historical pins are preserved.
