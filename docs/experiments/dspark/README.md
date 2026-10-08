<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DSpark: speculative decoding for DeepSeek V4 Flash (M3)

M3's "Speculative decoding in the core"
([plan](../../plan.md#m3--single-spark-fast-full-swap-in-progress)), for
DeepSeek V4 Flash 0731 (UD-Q2_K_XL) with its DSpark drafter
(`dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf`, the pin in
[baselines](../fast-swap/baselines.md)), on the paged node
([swap](../fast-swap/swap.md)) with decode graphs (D-090). The verify's
numerical rule is D-092. Qwen3.8's MTP layer is not part of this slice.

*Since 2026-09-28 (the owner: speed before bit exactness, D-092's note):
the row-invariant verify described here is the optional exact mode
(`--exact on`); the default verifies on DeepSeek's fast plan, each routed
expert read once for all the rows that select it, with the checks coarse
where the kernels differ ([dsv4-decode](../dsv4-decode/README.md)). The
numbers below are the exact mode's, as measured then.*

## Design

**The drafter.** llama.cpp's DSpark (`dflash.cpp` `graph_dsv4` and
`common/speculative.cpp`'s `draft-dspark`, b29c606e, MIT) is mirrored op
for op in `kernels/ggml/dsv4_graph.cc` (`BuildDsparkGraph`), its profile
and state in `model/dspark.h`:

- 3 DeepSeek V4 blocks, window-only (compression ratio 0), every layer
  routed (256 experts, `exp_probs_b`), MXFP4 experts, Q8_0 dense, BF16
  router and Markov head; rope NORM at base 10000 without YaRN.
- Features: the hyper-connection mean of the target's residual streams
  entering layers 41 and 42 and leaving the last, `fc` [12288 → 4096] and
  `enc.output_norm`, projected by each block's `wkv` and `kv_norm`, roped
  and written into the drafter's KV ring (256 cells) at every committed
  position. The injection runs inside the target's chunk graph (prefill
  and verify), so it costs no job of its own.
- A draft block of 3 rows: the anchor and 2 `MASK` tokens (128799),
  non-causal among themselves and over the ring within the window of 128;
  slot i predicts position pos0 + i + 1 (`sample_from_anchor`), so 3 rows
  give 3 drafts. The Markov head (`w1[prev]` → `w2`
  bias) chains each slot's argmax into the next; the confidence head is
  bound and never read (p_min 0, as the card runs it). n-max 3, so a
  verify is 4 rows.
- Its state, the ring, is a D-068 representation (append, truncate) of
  its own, a second kPreserve state region spilled and restored with the
  target's.

**Import and binding** (D-089's note). The drafter GGUF is its own v0
artifact through `import_m3.py`'s GGUF path unchanged (architecture
`dflash`; 772 groups, 5,675 chunks, 10.89 GB on disk, 1.5 min). It holds
no copy of the target's token table or head: `BindDspark` binds the
target artifact's `token_embd.weight` and `output.weight`, so one set of
extents serves both. The drafter's dense groups and expert slabs are paged
beside the target's by the same runner (`src/engine/dsv4_runner.h`, in
`benchmarks/` until D-096).

**Verify** (D-092). A verify chunk of k + 1 rows (anchor and drafts) runs a
row-invariant plan, so each row computes exactly what its one-row decode
step computes: a llmpalooza copy of GGML's MMVQ with the one-column launch
configuration for every column (`mmvq_rows.cu`; Q8_0, MXFP4, Q4_K, Q5_K,
Q6_K, IQ2_XS, IQ3_XXS), per-token expert launches over grid z, MMVF for
float products up to 8 columns, flash attention per query row, and drafts
shortened so a verify never crosses a mask width
(`Dsv4SameWidths`). It returns every row's logits.

**Rollback.** Before a verify's work, the job copies every state byte the
verify will write into a snapshot region on the device (`CopyRanges`, one
kernel): per row, its window cells, its compressor ring rows, the
compressed rows of blocks it completes, and the drafter's ring cells;
chunk-wide, the dummy block's scratch rows (`Dsv4ChunkWrites`,
`DsparkWrites`). `Accept(keep)` queues the restore of rows ≥ keep and all
scratch before the next job's own work (or now, `Rollback`), which leaves
exactly the state a verify of the accepted rows alone would have left
(D-068 truncation; the representations declare kAppend | kTruncate when
speculating). Nothing is re-run. A verify that fails after its snapshot
was queued is undone whole the same way (every saved range restored
before the next job); any other failed job that may have written the
state, or a launch of unknown effect, quarantines the state until
`Clear` (`Dsv4Runner::Settle`), so it is never left half-written.

**One job a step.** `DraftVerify` chains the draft and its verify in one
device job: the draft's graph, its drafts copied into the verify's staged
tokens, their embedding rows looked up on the device (`ggml_get_rows` of
the Q5_K table) into the verify's staged rows, the snapshot, and the
verify's graph. The device lookup equals the host's dequantization for
every one of the 129,280 tokens (checked at each greedy run,
`CheckDeviceEmbedding`). Both graphs are captured on their second run and
replayed after (D-090), including across swaps.

**Acceptance.** Greedy: accept drafts while they equal the target's argmax,
then the target's token. Sampled: standard speculative sampling for a
deterministic drafter (q = δ): accept draft d with probability p(d);
on rejection sample from p with d removed and renormalized
(`execution/sampling.h` `VerifyDraft`); the acceptance uniform comes from
the seeded stream at the row's position xor `kAcceptStream`, the sample
from the plain stream, so it is reproducible per seed.

## Correctness

The harness is `llmp_spec_runner` (`benchmarks/spec_runner.cc`) on
`spark-b` (GB10, driver 580.178.04), DeepSeek artifact `8a355bfb…`,
drafter artifact `dd2d3f9c…`, context 8,704, graphs on. Prompts are
[prompts.json](../fast-swap/prompts.json)'s: `prose` and `code` for
decode, and the six chat prompts, all rendered by llmpalooza's DeepSeek
renderer with thinking on, which is what llama-server's template gives by
default (the chat prompts' token IDs equal the recorded llama.cpp
reference's, checked at each run). Every check compares against plain one-token
greedy decoding in the same process: tokens and every logits row, bit for
bit.

| Check (exit criterion) | Run | Result |
| --- | --- | --- |
| Greedy speculation equals greedy | `prose`, `code` (256 tokens, 3 repeats each) and 6 chat prompts (32 tokens) | every token and logits row bit-identical, every run; each prefill with the injection equals the plain prefill bit for bit |
| Device embedding lookup | all 129,280 tokens | the drafts' rows looked up on the device equal the host's dequantization bit for bit (0.6 s) |
| Forced rejections, against a control | `capital`, 160 tokens: 91 steps, 83 with rejected rows (all-reject 15, one accepted 15, two accepted 15, a CSA block's completing row rejected 14, an HCA block's 1) | tokens and logits equal greedy's; after every step's rollback, each of the target's state tensors and the drafter's ring hash equal to the control's, which drafted only the accepted tokens |
| Rollback composes with swap | `capital`, 160 tokens, 91 steps; A swapped out for the FP16 fixture after step 45 (rows rejected) and back | B's logits equal their recorded hash; A continued bit-identical to the unswapped run, every step's state equal; the draft and verify graphs replayed across the swap without capture |
| Sampled speculation preserves the distribution | `capital`, `haiku`, `sky`, `fibonacci`; seeds 0–255; the first 8 generated tokens; temperature 1, no truncation; 8,192 sampled tokens a mode | total variation over each prompt's 16 most frequent tokens plus "other": 0.0039, 0.0005, 0.0269, 0.0161 (bound 0.1). Plain sampling took 637.5 s (at a host load of 16–20 from other work), speculative 448.8 s |

The forced run changes a draft (to the next token ID) at a chosen
position: all-reject, one or two accepted, and, whenever a verify's rows
complete a compressor block (a CSA block every 4 positions, an HCA block
every 128), the rejection on that row or before it, so the block's
compressed row is written by the verify and must be restored. The control
replays the same steps drafting exactly the accepted tokens. State is
compared by hashing each state tensor and the ring after the rollback
has run (FNV-1a over 8-byte words).

Unit tests carry the parts: `unit.SpecRowsTest.*` (GB10) holds each row
kernel to GGML's one-column launch bit for bit for every weight type and
reduction length the two models bring, per-token experts, MMVF's columns,
the argmax and the range copies; `unit.DsparkTest.*` the profile, the
binding and its refusals, the draft block's inputs and mask, the verify's
writes and same-width rule, and both graphs' plans; `unit.VerifyDraft.*`
that speculative sampling reproduces `Sample`'s
distribution and handles greedy and filtered drafts.

## Performance and memory

`spark-b`, a clean run (no other GPU process and load under 3 throughout,
sampled every 5 s), 256 tokens, the same prompts and settings as the
llama.cpp baseline (DSpark n-max 3, [baselines](../fast-swap/baselines.md);
llama.cpp's numbers are medians of three from that page). Llmpalooza's
speculative rates are three repeats, median in bold.

| Measure | llmpalooza | llama.cpp (UD-Q2_K_XL, DSpark Q8_0) | Ratio |
| --- | ---: | ---: | ---: |
| Decode, `prose`, tok/s | 26.82 / **27.94** / 28.09 | 30.80 | 0.91 (repeats 0.87–0.91) |
| Decode, `code`, tok/s | 29.33 / 29.25 / **29.33** | 31.94 | 0.92 (repeats 0.92) |
| Decode without speculation, `prose` / `code`, tok/s | 18.59 / 18.75 | 19.90 / 19.95 | 0.93 / 0.94 |
| Speed-up over own plain decode | 1.50 / 1.56 | 1.55 / 1.60 | |
| Acceptance (accepted ÷ drafted), `prose` / `code` | 0.551 / 0.579 | 0.536 / 0.569 | |
| Tokens a verify, `prose` / `code` | 2.63 / 2.71 | | |
| Peak `MemAvailable` drop, GiB | 104.70 | 104.7 | 1.00 |
| Load | 8.66 s to read 107.9 GB into the paged node (the drafter's 10.9 GB) | 92.3 s from server start to the first token (one load) | |

Both sides speculate with the same drafter and the same greedy drafts;
acceptance is slightly higher here, perhaps because every verify row is
exact (llama.cpp's verify rows differ from its decode in the last bits,
RE-033; not isolated).

**Against the gate: a narrow pass.** On the medians, decode with
speculation is 9% (`prose`) and 8% (`code`) slower than llama.cpp's, so
it passes the exit's "not more than about 10% slower" (D-085) with one or
two points to spare. The margin is inside the run-to-run spread: the
slowest `prose` repeat (26.82 tok/s) is 0.87×, outside the gate; the
three repeats span 26.82–28.09 (5%); llama.cpp's medians come from
another session (baselines.md); and plain decode here was 18.6–18.8
tok/s against 19.05–19.53 in graphs.md's runs of the same code. The
review's rerun (same host, clean: no other GPU process, load under 2.7;
the rollback-failure and argmax fixes only) gave 27.81 / 28.75 / 28.85
and 29.60 / 29.72 / 29.70 tok/s (0.93× / 0.93× on the medians), plain
18.37 / 19.44. A rerun can land on either side of 0.90×. The leads for more speed are under
[Limits](#limits-and-what-remains). The chat prompts (32
tokens, acceptance 0.72–0.92) speculate at 30.9–38.6 tok/s against
18.6–19.1 plain. An earlier run with draft and verify as two jobs
(before `DraftVerify`) gave 26.9 / 28.0 / 28.7 and 29.2 / 29.4 / 29.4:
chaining them saves under 1 ms a step, because the device is already
busy for all but about 3 ms of it.

**With a lease per request and the runtime wake** (2026-09-28). The
figures above predate both: each chunk leased DeepSeek's whole closure
(1.3–2.6 ms a step, D-093) and the lanes slept between steps on their
200 µs windows; they were never harness-polled. The spec runner now runs
each generation as a request (one lease, every chunk under it) on the
runtime's own wake (D-094, [runtime-wake](../runtime-wake/README.md)).
`spark-b`, 10:54–10:58, one run each, the memory gate before each (no
other GPU process, load under 6 at the start; other agents used the host
between runs), 256 tokens, three speculative repeats, median in bold:

| Measure | Runtime wake | 100 ms windows (a diagnostic, same session) | llama.cpp (above) | Ratio (runtime wake) |
| --- | ---: | ---: | ---: | ---: |
| Decode, `prose`, tok/s | 28.57 / **29.67** / 29.82 | 28.71 / 29.63 / 29.74 | 30.80 | 0.96 (repeats 0.93–0.97) |
| Decode, `code`, tok/s | 30.84 / **30.85** / 30.82 | 30.35 / 30.27 / 30.33 | 31.94 | 0.97 (repeats 0.96–0.97) |
| Decode without speculation, `prose` / `code` | 19.13 / 20.42 | 20.13 / 20.20 | 19.90 / 19.95 | 0.96 / 1.02 |
| A step (draft and verify), `prose` / `code`, ms | 88.15 / 88.02 | 88.38 / 89.46 | | |
| Chat prompts, speculative, tok/s | 32.8–40.0 | 32.2–40.1 | | |

Every greedy check passed in both (tokens and logits bit-identical to
plain decoding, the injected prefill equal to the plain one); acceptance
and tokens a verify are unchanged (0.551 / 0.579; 2.63 / 2.71); 1,309
replays, 7 captures, none refused. Speculative decode is now 0.96× and
0.97× llama.cpp's on the medians, inside the exit's 10%, against 0.91×
and 0.92× before: a step's wall fell from 93.6 / 92.5 ms to 88.2 / 88.0
(mostly the lease per request: per step it alone cost DeepSeek 1.3–2.6
ms in graphs.md's runs, against the 0.26–0.65 ms of wakeups at the old
200 µs windows that the wake removes; the rest of the fall is not
isolated). The wake itself costs nothing measurable against polling (the
step's wall is within 1.5 ms either way, in the runs' noise). The first plain generation
(`prose`) includes the decode graphs' capture. Peak `MemAvailable` drop
106.5–106.8 GiB against 104.7 before; not isolated (the host is shared).
Raw outputs: `~/scratch/m3wake/models/spec-*` on `spark-b`.

**The step on the device** (Nsight Systems, `prose`, chained, graphs
replayed): the restore copy, the draft graph (7.8 ms), the drafts' copy,
lookup and snapshot save (together 0.03 ms), the verify graph (81–85 ms),
then 2.7–3.2 ms with the device idle while the host reads the logits,
judges the drafts and submits the next job.

**Where a step's time goes** (Nsight Systems, `prose`, the kernels' busy
time summed per step, so slightly inflated by tracing): a 4-row verify
keeps the device busy 89.3 ms against 56.1 ms for a one-row decode step.
The quantized products take 71.3 ms against 43.3 ms, and nearly all of
the difference is the routed experts: each token reads its own 6 experts
(IQ2_XS and IQ3_XXS), about 36 ms a verify against about 11 ms a decode
step (computed from the kernels' totals and counts), as upstream's
one-token-per-column MMVQ-with-ids path reads them too; the dense products
read their weights once for all 4 rows and cost what one row costs
(within 3 ms). Attention per query row adds 2 ms, the rest about 3 ms.
The draft block takes 7.4–8.6 ms. (This breakdown traced each graph
node, in the run before `DraftVerify`; the kernels are the same.)

## Judgement calls

Made under the owner's overnight delegation, for review with the slice:

- **Bit-identity over bounded divergence** (D-092). D-068 allowed a
  verify plan to diverge numerically; the exit asks for "the same tokens"
  and, after each rejection, output equal "bit for bit". A row-invariant
  verify gives that exactly, at the price of MMVQ's one-column reduction
  width in the verify and attention per query row.
- **Rollback by snapshot and restore**, not by re-running the accepted
  rows or keeping per-row state planes: the verify saves only the bytes it
  will write, so a rejection costs one small copy kernel queued ahead of
  the next job's work.
- **Drafts shortened at mask widths** (about one step in 64) rather than a
  verify plan whose rows differ in mask width.
- **The drafter is its own artifact, bound at load** (D-089's note); no
  composition document until one exists without model_index.json.
- **Prompts rendered with thinking on**, matching llama-server's
  `/apply-template` default, so the six chat prompts equal llama.cpp's
  recorded IDs token for token; `prose` and `code` are rendered the same
  way (the reference records no IDs for them).
- **Sampled speculation keeps the greedy drafter** (q = δ), as llama.cpp's
  DSpark drafts greedily; standard speculative sampling then reduces to
  accepting d with probability p(d).
- **One job per step** (`DraftVerify`), with the drafts' embedding rows
  looked up on the device, proven equal to the host's lookup; the separate
  `Draft` and verify jobs remain and are what the forced and swap checks
  run (their control needs chosen drafts).
- **Verify graphs are captured per shape** (1 to 4 rows) like decode's
  (D-090), and so is the draft block; the snapshot's save and restore
  copies run outside the graphs, from ranges the host writes each step.

## Limits and what remains

- **Qwen3.8's MTP layer** is the plan item's other half, not started.
- **Speed.** 0.91–0.92× llama.cpp's DSpark decode on the medians, a
  narrow pass of the 10% gate that one `prose` repeat (0.87×) misses;
  the verify's device time (81–85 ms against a 48 ms decode
  step) is most of a step. Two leads, neither measured: the paged node's
  per-step round trip (about 3 ms here; plain decode pays it too,
  graphs.md), and grouping a verify's tokens by expert so each distinct
  expert is read once with every column's arithmetic unchanged, whose gain
  depends on how often neighbouring tokens share experts.
- **A harness, not the runtime.** Speculation runs in
  `llmp_spec_runner` on the test harness's paged node, like the swap
  runner; the runtime process and the chat route take it with them.
- **Prefill with the injection** equals plain prefill bit for bit but is
  not timed here (llama.cpp's drafter cost it no measurable prefill).
- **What the forced and swap runs reach.** At 160 tokens both stay
  below position 256, where no mask width changes. The review's forced
  rerun at 300 tokens crossed it: 160 steps, 141 with rejected rows (two
  HCA blocks' rows among them), verifies shortened to 2 rows at 254 and
  1 at 255, and every step's state equal to the control's. The swap
  check swaps A out after its last step's restore has run (each step's
  state hash runs it); a restore still pending across a swap relies on
  D-090's pinned places and the snapshot staying resident, which no run
  exercises.
- **Ring cells past the committed prefix** keep the last draft's keys
  after a rollback, in the forced run and its control alike; the draft
  mask never reads them (they fall outside the window as the positions
  256 before), and the next draft or injection overwrites them.
- **The composition document** for a drafter and its target (D-089's
  note).
- **The verify's weight types.** A new weight type needs its row kernel
  (`mmvq_rows.cu`) before speculation runs on it; the plan refuses it
  otherwise (D-092).
