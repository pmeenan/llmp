<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma3 internal four-root batching foundation

The internal D256/H8/no-softcap C4 path passes its operand and representative
same-geometry stock quality gates. It reuses the existing four-bank MMA launcher;
no floating kernel changes. Four independent states and funded head outputs run
with the established paired-prefill limit of two owners and 256 total rows.
Ordinary serving still admits at most two slots.

This does **not** close concurrency qualification. Native and unmodified stock
both change 12 positive-margin choices between solo and joined C4 geometry on
these supplied histories. A short n=2 C4 timing screen is 3.29938% slower than
stock. Actual C3 arrival/departure waves, wider cohorts, public adoption, broader
quality, context, memory, swap and sustained performance remain open. No new
numerical allowance is introduced. A subsequent
[actual C3 departure/rejoin control](../gemma-h8-c3/README.md) passes its separate
operand and same-schedule model gates; the other limitations above remain open.

## Checked implementation and funding

The host contract now admits H8 only for D256/GQA2 at actual/logical C2 or C4,
offset zero. C4 requires zero softcap; cap50 remains C2 only. Bounded actual-root
reads also remain C2 only. Existing H16/H32, cap50 and copy-free C2 paths retain
their contracts. The graph selects independent four-root attention for four
one-row frontier outputs; other unqualified shapes retain their fallback.

Unequal internal C4 cache views use real, funded zero tails and concatenated
F16 activations. Every padded mask row is invisible. Setup measures equal
endpoints and all three proper long-owner counts, including three short roots
and one long root. Plan controls check each split, four output heads and the
source inputs at exact funding and one byte short. State layout and per-owner
128-row capacity are unchanged. The first model screen below uses equal read
widths; these descriptor/funding checks alone do not qualify unequal C4 model
execution or public serving.

The existing operand helper adds a closed C4 mode without changing its legacy
case set. At cells 256 and 1024 it checks original packed physical-stream MMA
against four real independent roots, fresh queries/masks, eager and captured
replays, FP64 and exact/short scratch funding. Both shapes are bitwise exact to
original MMA. FP64 NMSE is 1.33784e-7 / 2.23447e-7 against the existing 5e-4
bound. Original and owner occupancy are both 1; each uses 48 blocks, columns 4,
GQA group 2 and 399,616 scratch bytes. Cap50 C4, bounded C4, partial H8 and
H8 cohorts 8/12 remain refused.

## Representative model and geometry controls

The approved Gemma3 QAT Q4_0 source and prepared artifact are unchanged. Four
actual state owners use two authenticated corpus identities explicitly:
owner 0/2 receive input 0, owner 1/3 input 1. Each input contains 295 native
and stock-tokenizer-matched IDs: 256 prompt rows, three supplied off-clock
scalar rows, 32 supplied teacher steps and four departure steps. The four
owners have independent backing; duplicated input identities are not four
independent corpus samples.

The native caller uses context 4096, F16 KV, four slots/head rows, per-owner
rows 128 and wave rows 256. Prompt pairs 0/1 then 2/3 execute two 128-row chunks
each. Decode joins four owners; departure advances each owner alone while
checking all peers' cursors. The quality output has 148 full vocabulary heads
and 144 labeled transitions; four final rows have no targets. Two fresh joined
runs, an eager run, device/full-head choices, initialized state, Clear reuse,
spill/checkpoint restoration and publication refusals agree exactly within
policy. A separate solo policy uses the same supplied histories and paired
prefill, executing each decode owner independently.

The original v0.6.0/d812 reference uses four per-sequence greedy samplers,
context 16,384 total / 4096 per sequence, batch/ubatch 256, F16 KV, flash
attention, `swa_full=false` and `kv_unified=false`. Metadata-only teacher
observations witness four-sequence decode and two-sequence prefill attention.
There is no extra state-only synchronization. Logical reset uses
`llama_memory_clear(..., false)` off-clock. The original backend sampler still
copies full 262,208-value sampled-logit rows to the host; calling its token API
does not remove that transfer. Stock joined and solo teacher runs each repeat
bitwise exactly and retire their containers before aggregation.

| Comparison | Exact full heads / 148 | Greedy differences | Relative conditional loss delta |
| --- | ---: | ---: | ---: |
| Native joined vs stock joined | 146 | 0 | +0.00024884% |
| Native solo vs stock solo | 148 | 0 | 0 |
| Native joined vs native solo | 4 | 12 positive-margin | −0.421711% |
| Stock joined vs stock solo | 4 | 12 positive-margin | −0.421959% |

The first cross-geometry greedy difference in both engines is head row 8,
owner 0, next-token position 261, with solo-reference margin 0.1975917816.
Maximum raw cross-geometry logit difference is 1.05425167. All four native
initialized state hashes differ between solo and joined arithmetic, while
same-policy replay/restore remains exact. Native joined vs stock joined has
maximum raw difference 0.00072670, zero strict/tie differences and passes the
existing conditional-loss gate. `quality.json`'s PASS is this joined/stock
gate only; the descriptive cross-geometry result is not a concurrency PASS.
No attention-only cause, state corruption, near-tie waiver or quality bound is
inferred from the common stock/native geometry variance.

## Short paid timing screen

After the strict joined gate and official completion, fresh processes run
stock/native/native/stock. Each warms the same prompt, three supplied rows and
eight natural steps, then logically clears off-clock. Paid work includes all
four 256-row prompts (1024 rows total), followed by three off-clock supplied
rows per owner, then 32 joined natural decode steps and four final full heads.
Native device tokens and original stock sampled-logit transfers stay on their
normal paths. Both engines publish the matching final heads inside paid decode.

| Arm | Prefill ms | Decode ms | Paid ms |
| --- | ---: | ---: | ---: |
| Stock 1 | 160.073 | 460.998 | 621.071 |
| Native 1 | 174.927 | 464.709 | 639.636 |
| Native 2 | 176.091 | 465.299 | 641.390 |
| Stock 2 | 159.453 | 459.586 | 619.039 |
| Native mean | 175.509 | 465.004 | 640.513 |
| Stock mean | 159.763 | 460.292 | 620.055 |

All four arms have identical 128 natural choices and four final full heads.
Native paid latency is +3.29938% (+15.746 ms prefill, +4.712 ms decode). This is
one short n=2 screen, not parity, peak-memory, endpoint or sustained evidence.
There is no causal timing claim about attention in isolation. Paid prefill
includes the normal first capture beside eager execution for previously warmed
keys; its contribution to the prefill gap is not separately measured. No extra
warm passes or benchmark-only capture policy are introduced.

## Replay and provenance

The [native caller](../../../benchmarks/gemma3_batch_probe.cc),
[public-API stock caller](../gemma3-execution/llama_batch_probe.cc) and
[aggregate helper](../gemma3-execution/analyze_batch.py) are reusable. Supply the
exact approved source/artifact from [the existing pin](../gemma3-execution/pins.json),
original pinned reference image/library closure and the external corpus texts.
Prepare the first 295 tokenized IDs from each text; the stock caller checks
its original tokenizer against every supplied owner history. Keep all payloads,
heads, traces and logs outside Git, in a new private output directory `OUT`.
The native invocation is:

```sh
llmp_gemma3_batch_probe ARTIFACT IDS0 IDS1 OUT/native-own1 own joined
llmp_gemma3_batch_probe ARTIFACT IDS0 IDS1 OUT/native-own2 own joined
llmp_gemma3_batch_probe ARTIFACT IDS0 IDS1 OUT/native-eager own eager
llmp_gemma3_batch_probe ARTIFACT IDS0 IDS1 OUT/native-solo own solo
```

Run the stock caller as `MODEL IDS0 TEXT0 IDS1 TEXT1 NEW_OUT teacher|solo|cycle`
inside the unchanged pinned CUDA image. Use the historical retirement helper's
checked name/label and private CID files. Queue each stock mode twice as
`stock-teacher1/2` and `stock-solo1/2`; retain their checked absence receipts.
Run aggregate modes `own OUT MODEL_JOB_LOG`, `quality OUT`, then only after
quality PASS and official job DONE0 run the four `cycle` arms and aggregate
`cycle OUT PERFORMANCE_JOB_LOG`. The analyzer authenticates `inputs.json`,
full finite heads, scored-row mapping, own repeats, output lengths, exact
natural choices/finals, paid counters and stock retirement receipts. The
installed queue logs supply arm boundaries. All builds/inference use installed
`spark-job start --gpu --timeout 600 --stop-on-fail` and official `wait`;
checksum-sync the source tree and require an empty dry-run before rebuilding.

Measured on Spark B (GB10, driver 580.178.04, SDK
`aarch64-c09daba6ac31edee`, target `sm_121`), 2026-10-07, true parent
`fa6defd` (prerequisite `ee85bf3`).
The prerequisite, model retry and timing jobs finish official DONE0. Model
attempt 1 stops before inference on two test designated-initializer order
errors; the retry changes only their order and uses a fresh namespace. XML
records 32 passing executions / 25 unique controls: six GPU, eight graph and
11 plan controls. All six stock containers have checked absence receipts.

| Identity | SHA-256 |
| --- | --- |
| Approved source, 2,526,080,992 bytes | `ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a` |
| Prepared artifact | `8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb` |
| Input 0, 295 i32 | `c37e404d378484e6dbf4da9a7f7e812105b569eac9e8323e9dd49e47d57dc470` |
| Input 1, 295 i32 | `1e7793632795175c98b7835511b66cadbaa48312a87fdd297e1dd2186258850e` |
| Native probe | `4d0b3a561e87fe79f80de3e0a6a6bb7b06abd7164bcd9b097f89e7d604a62c84` |
| Stock public-API probe | `4a19fb324454e9565705e865772325587d472d67973b3691c0b540aaf6532f64` |
| Build receipt | `874aaf7a5967cfbe91054e0d8fc1a0630f952e0e8eb54b831d09719f1e08ce89` |
| Complete measured model source inventory | `10a54c6e3c2a41b6295e9ae691b11c4235ad2c7786ae52609eb1e5c8884fb907` |
| Original CUDA image | `c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db` |
| Exact natural choices, all four timing arms | `29a32893d8ada545193abd53ae773529a1e4a85fe15383dcf45f237dd5ccfec6` |
| Exact final heads, all four timing arms | `129f2ac8a6e63369d4eaacb4ac208227d3d1dee4a14c25abbadf0fce4bbc42d0` |

Task-entry TensorFold check observes HEAD
`041d14a94e951834470fd514ed33e65b8be1059a`, version 1.0.0, on 2026-10-07.
Its current documented GB10 CUDA recipe qualifies Nemotron and supplies no
matching Gemma3/Gemma2 CUDA GGUF comparator. Historical pins are preserved.
