<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 speculative decoding with its MTP layer, and decode graphs (M3)

This slice finishes Qwen3.8's half of M3's "Speculative decoding in the core"
([plan](../../plan.md#m3--single-spark-fast-full-swap-in-progress)), and adds
Qwen3.8's part of "CUDA graphs for decode". The model is Qwen3.8 Flash Next
(ModelOpt NVFP4 experts and MXFP8 linears, the CUTLASS-layout artifact
`c4fb47a9…`,
[qwen38-native](../qwen38-native/README.md)). The drafter is the MTP block
stored in the same checkpoint. Both run on the paged node
([swap](../fast-swap/swap.md)) with a lease per request (D-093) and the
runtime's own wake (D-094).

The owner's rule (D-085's note, 2026-09-28) is speed before bit exactness.
So the default verify here is a batched fast plan, judged coarsely:

- each token must equal the plain engine's argmax, or be a near-tie;
- a forced rejection must leave exactly the control's state;
- sampled speculation must pass the total-variation bound.

The verify is not row-invariant (D-092's exact mode is DeepSeek's).

## Design

**The drafter** mirrors vLLM's `Qwen3_8FlashNextMTP`
(`vllm/models/qwen3_8_flash_next/nvidia/mtp.py` at 8e685d198, the oracle's
engine). It lives in `kernels/ggml/qwen38_graph.cc` (`BuildQwen38MtpGraph`),
with its binding and state in `model/qwen38.h`. For a target position q
whose next token t(q+1) is known, it computes:

- e = `fc_embedding`(GemmaRMSNorm(embed(t(q+1)))).
- h = `fc_hidden` applied per stream to the GemmaRMSNorm of the target's
  four hyper-connection streams at q (10,240 values, before the head's mix).
- The layer input is the streams h + e.
- One full-attention QSA layer runs over them, with its own K, V and
  indexer caches (cell q), the MoE (512 experts, top-10, NVFP4 in the
  CUTLASS layout) and the combine.
- Then its own final mixer, and the target's head.
- The argmax is taken over the draft vocabulary (below).

A further pass (vLLM's scheme A) takes the previous pass's draft as its
token and that pass's combined streams as its h, at the next position. The
drafter holds no token table or head: it binds the target's `token_embd`
and `output`, as vLLM loads the same two. Its attention, indexer, router
and shared expert are BF16 in the checkpoint and stay BF16; the o-proj
fast path is skipped for BF16.

**State.** The drafter's state is one kPreserve region, a D-068
representation spilled and restored with the target's. It holds the MTP
layer's F16 K and V caches, its F32 indexer keys, and a stream buffer H of
`max_rows + 1` rows of the target's streams.

- A prefill chunk with injection exports its rows' streams to H[1..rows].
  In the same job, it runs the drafter's pass over the positions whose next
  token the chunk knows. That is every row but the last, whose streams are
  carried to H[0] and H[1] as the pending row.
- A verify exports its rows' streams to H[1..rows] the same way.
- The next draft's catch-up pass reads H[1..R]. R is 1 after a prefill, and
  the number of rows the last verify kept after a verify.
- A prefill chunk with injection after a verify (a new turn) catches up the
  same R rows first: rows 1..R−1 in a pass of their own (a job before the
  chunk's, whose export overwrites them), and row R copied to H[0] at the
  start of the chunk's job, so its pass covers that position too
  (`Qwen38Injection` in `model/qwen38.h`; `jitllm_qwen38_spec --check turn`
  compares the cells with a draft's own catch-up).

**A step** is two jobs.

1. **Draft:** the catch-up pass over R rows, then k − 1 one-row passes, in
   one graph. The drafts stay on the device until one 512-byte copy.
2. **Verify:** the anchor and k drafts through the target in its *verify
   form*. It is a separate job because the drafts' n-gram (PLE) rows are
   read from the SSD by row (D-035) once the drafts are known. The verify
   returns every row's argmax, computed on the device. Logits are copied out
   only when sampling or checking; the lean graph variant copies argmaxes
   alone.

**Rollback by commit, not by snapshot of the recurrent state.** The verify
form reads the Gated DeltaNet recurrent state, the convolution histories and
the n-gram layer's convolution history, but never writes them. Per row it
saves the convolution output (q, k, v), the QKV input, the gate, beta and
the n-gram layer's input rows to a commit region (`Qwen38CommitLayout`).

`Accept(keep)` queues `Qwen38Commit` (`kernels/ggml/qwen38_commit.h`) ahead
of the next job's work:

- `GdnCommitKernel` replays the kept rows' recurrence in place, with
  arithmetic identical to the verify's `GdnColumnsKernel` (so the state is
  what the verify's rows computed, bit for bit).
- `HistoryCommitKernel` sets each convolution history to its last `taps`
  inputs.

The KV and indexer cells, which the verify does write, are saved first to a
snapshot (`CopyRanges`, DSpark's pattern). The rejected rows' cells are
restored in the same queued step. The MTP caches need no restore: their
cells past the committed positions are rewritten before any row reads them.

A failed verify is undone whole. Its stream rows may already overwrite the
rows the next draft would read, so no draft runs until a verify is
accepted or a chunk with the injection writes them again (a chunk without
the injection also clears them). A verify saves the stream rows it
overwrites with its cells, so one discarded after a failed host judgement
(`DiscardVerify`) leaves the previous pending rows in place. Any other failure that may have written
the state quarantines it until `Clear`, as DeepSeek's runner does. The
commit region is never spilled; it stays mapped across a swap, and a
commit still owed at the swap runs at the next job after it.

**The draft vocabulary.** The draft head computes only the first N rows of
the target's head (FR-Spec's idea; a view, no copy). This token-ID prefix is
a heuristic chosen by the sweep below, not a ranking of the model's output
frequencies. It does not use Mia's AGPL-licensed 47k list. A smaller head
can change acceptance, never the verify's authority over each token.
The [draft-head study](../qwen38-draft-head/README.md) checks this difference
separately from draft depth.

**Depth.** Mia's launch drafts 3 tokens. Here each verify row adds
4.7–5.8 ms to the step (mostly its own 10 experts; the dense weights are
read once for all rows). A third draft, accepted 18–27% of the time, costs
more than it returns. The defaults are k = 2 and
N = 65,536, the fastest of the sweep below.

**Decode graphs** (D-090) now cover Qwen3.8: the one-row decode chunk, each
verify shape and each draft shape are captured on their second run and
replayed. The n-gram row gather took its row count as a launch parameter.
It now reads the count from pinned memory the host writes before the job
(`GatherPleRows` launches `max_count` blocks, and the extra blocks return),
so one graph serves every step. `ReadPleRows` polls the io_uring completion
instead of sleeping on it, which saves the wake-up (RE-017).

**Multi-row MXFP8 products.** `Mxfp8Gemv` is templated on its columns: 1, up
to 4, or up to 8, with 1, 4 or 2 rows a lane. A lane loads each x vector
once for its rows, and each row's arithmetic is unchanged. This took 1.7 ms
off a 4-row verify.

## Import

The MTP block is its own drafter artifact, not part of a new target
import (D-089's note, as DSpark's is):

- **Size.** `import_m3.py drafter` reads and hashes only the checkpoint's
  last shard and `config.json`, and writes a 1.6 GB artifact. A new target
  import would re-read, repack and re-hash 104 GB (about 13 min), change the
  target's artifact ID, and need both Sparks to receive the new artifact
  again.
- **Existing records.** The swap pairs and the qwen38-native records all
  name `c4fb47a9…`, and they stay valid.
- **Optional.** The drafter can be left out: plain decode does not page it.

Converter: `m3-1+layout-…+modelopt_qwen38-<digest16>+mtp`, architecture
`qwen4exp-mtp` ([modelopt_qwen38.py](../artifact-layout/modelopt_qwen38.py)
`plan_mtp`; `check_mtp_config` refuses anything other than one hybrid
full-attention layer over the last streams with shared embeddings).

On `spark-b` it took 6.7 s at 229 MB peak RSS. The artifact is
`056a750e3a90be3ae6a4b12bb963ce45290aaa6f52b5ba9799e777d491f80aea`,
1,597,030,400 bytes (one layer group, 512 expert groups, the head's mixer),
and it was copied to `spark`. `test_modelopt_qwen38.py` (27 tests, 4 of
them new) and `test_import_m3.py` (12) pass on `spark-b`.

The target artifact is unchanged. The module's target path writes what it
wrote before (its tests are unchanged and pass), so the target is not
re-imported.

## Correctness

**Setup.** The harness is `jitllm_qwen38_spec` (`benchmarks/qwen38_spec.cc`),
run on `spark` (GB10), with the target `c4fb47a9…`, the drafter `056a750e…`,
context 8,704, graphs on, the defaults (depth 2, 65,536 draft rows), and the
memory gate before each load. The owner released `spark` part way through
the slice, and the earlier runs were on `spark-b`.

**Prompts.** They are from [prompts.json](../fast-swap/prompts.json):

- `prose` and `code` for decode, 256 tokens each;
- six chat prompts, 32 tokens each.

All are rendered by the native Qwen3.8 renderer with thinking on. The chat
prompts' IDs equal the oracle's (checked at each run).

**What the plain engine runs.** Its tokens come from one-row chunks with
decode graphs, in the same process. For the near-tie check, the plain
engine is teacher-forced on the speculative tokens. The margin is the
target's logit gap between its argmax and the token the speculation
emitted.

| Check (exit criterion, as amended) | Run | Result |
| --- | --- | --- |
| Greedy speculation equals greedy, near-ties aside | 8 prompts, 13:49–13:51 | every token is the plain argmax on its prefix except 9 near-ties: `prose` 4 (largest margin 0.147), `code` 3 (0.050), `capital` 1 (0.003), `primes` 1 (0.019); 0 violations of the 1.0 bound. The speculative repeats equal each other bit for bit. The prefill with the drafter's injection gives the plain prefill's logits bit for bit |
| Forced rejections, against a control | `capital`, 160 tokens, depth 2: 86 steps, 61 with rejected rows (17 all rejected, 17 one accepted) | after every step's commit, every target state tensor, the MTP caches and the stream rows the next draft reads hash equal to the control's (0 of 86 differ); the tokens hold the near-tie rule (8 near-ties, largest 0.916, 0 violations) |
| The same at depth 3 (earlier build, `spark-b`) | 96 tokens, 32,768 draft rows: 41 steps, 29 with rejected rows (8 all rejected, 8 one accepted, 8 two accepted) | 0 of 41 differ; near-ties 5, largest 0.916, 0 violations |
| Sampled speculation preserves the distribution | `capital`, `haiku`, `sky`, `fibonacci`; seeds 0–255; the first 8 tokens; temperature 1 | total variation over each prompt's 16 most frequent tokens plus "other": 0.0317, 0.0396, 0.0093, 0.0332 (bound 0.1), 8,192 sampled tokens a mode. Plain sampling took 594.7 s (inside D-085's 10 minutes, barely), speculative 452.1 s |
| Rollback composes with swap (review, merged on `051baa6`, `spark`, 14:39–14:40) | `capital`, 160 tokens, depth 2, forced as in `swap` below: the control's 97 steps, then the same run stopped after step 47 (rows rejected), its commit and restore still owed; Qwen3.8's state, weights and drafter out for the FP16 fixture and back | the owed commit ran after the swap; all 97 steps' states equal the control's (0 differ), all 160 tokens and their logits bit for bit; B's logits hash `bb8ae5e7…`, as recorded; every weight and state extent still pinned; after the swap 48 verifies and 50 drafts replayed graphs captured before it (2 new shapes captured) |
| Decode graphs across a swap (review, `spark`, 14:34–14:35) | `jitllm_swap_pairs --a qwen38 --b image --cycles 2 --continue 16` (8,192 context tokens) | the prepared return replayed the decode graph captured before the swap for all 16 continued steps (none captured or launched), each step's logits bit-identical to the unswapped continuation, the restored state byte-identical; every swap `exact` |

**The control.** The forced run changes a chosen draft (to the next token
ID). The control replays the same steps with the same row counts, but
drafts different tokens after the kept ones: the forced one is changed
again. So the same kernels run, and a byte a rejected row wrote would show
as a difference.

**The swap check** (`--check swap`, as DeepSeek's in `jitllm_spec_runner`)
forces all rejected, one accepted and as drafted in turn. It swaps with
the last step's commit still owed, so it also shows that the commit
region survives a swap and that the commit then runs on the restored
state.

#### The verify's own noise (post hoc)

The rule in dsv4-decode's "The bound, going forward" takes the near-tie
bound from the verify's own rows against one-row decoding, recorded before
the comparison. That was not done here; the review measured it afterwards
on the merged build (`spark`, `verify_noise` in `jitllm_qwen38_spec`, the
same runs as the table's). It is the move of the plain row's top-two
margin when the verify's row gave the token:

| Prompt | Steps | Median | p95 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prose` | 255 | 0.068 | 0.347 | 0.494 | 0.630 |
| `code` | 255 | 0.047 | 0.231 | 0.288 | 0.389 |
| `capital` (forced run) | 159 | 0.266 | 1.486 | 2.367 | 2.391 |
| six chat prompts | 31 each | 0.041–0.388 | 0.178–2.341 | 0.232–2.391 | 0.232–2.391 |

So 1.0 is not the rule's bound: it is below the verify's p99 on `capital`
(2.37) and above it on the rest. It is kept as the stricter test, never
loosened. Every exception passes under either: the largest is 0.916 (the
forced run's), under that run's own p99 of 2.37. The re-run gave the same
verdicts as the slice's: greedy 0 violations on all 8 prompts (9
near-ties), forced 0 of 86 states differing, decode 41.50 / 42.40 tok/s on
`prose` and 39.04 / 38.65 on `code`. As with DeepSeek's `capital` step 93,
the rows where the verify moves the margin by more than 2 are not
diagnosed.

**Unit tests** carry the parts:

- `unit.Qwen38CommitTest.*` (GB10): the commit's state equals the verify
  recurrence's bit for bit, its histories are exact, and the launcher's
  refusals hold.
- `unit.Qwen38MoeTest.SeveralTokensGetEachTokensOwnProducts`: a verify's
  expert products equal one-token launches bit for bit.
- `unit.Qwen38Test.*`: the drafter's binding and refusals, the MTP state and
  commit layouts, `Qwen38Rows`, and the drafter's and verify's plans.
- `unit.PleRowsTest.*`: the gather's device-read count.

## Performance and memory

**Comparators.** Mia's vLLM (the same NVFP4 and MXFP8 checkpoint) from
[baselines](../fast-swap/baselines.md): MTP 3 at 37.85 / 37.85 tok/s
(`prose` / `code`, acceptance 0.423 / 0.422, peak 103.4 GiB), and MTP off
(deterministic) at 25.12 / 25.33. TensorFold's 55.2 / 55.1 tok/s (55.4 /
56.1 at 0.3.6.2, 37.5 without drafts) speculates cross-quantization and
is information only.

**Speculative decode** (the greedy run above, `spark`, 256 tokens, three
repeats, median in bold):

| Measure | jitLLM, MTP depth 2 | Mia's vLLM, MTP 3 | Ratio |
| --- | ---: | ---: | ---: |
| Decode, `prose`, tok/s | 41.88 / **42.46** / 42.72 | 37.85 | 1.12 (repeats 1.11–1.13) |
| Decode, `code`, tok/s | 39.10 / 38.72 / **39.09** | 37.85 | 1.03 (repeats 1.02–1.03) |
| Acceptance (accepted ÷ drafted), `prose` / `code` | 0.631 / 0.539 | 0.423 / 0.422 | |
| Tokens a verify, `prose` / `code` | 2.26 / 2.07 | | |
| A step: draft / verify / all, ms (`prose`) | 5.98 / 46.82 / 52.83 | | |
| Plain decode in the same run, `prose` / `code`, tok/s | 25.78 / 26.11 | 25.12 / 25.33 (MTP off) | 1.03 / 1.03 |
| Speed-up over own plain decode | 1.65 / 1.50 | 1.51 / 1.49 | |
| Chat prompts (32 tokens), tok/s | 40.95–52.38 (acceptance 0.65–1.00) | | |
| Peak `MemAvailable` drop, GiB | 75.8 | 103.4 | 0.73 |

Acceptance is not like for like. Mia drafts 3 per step and we draft 2, and
the later positions accept less often.

Peak memory is not like for like either:

- the n-gram table (28.8 GB) stays on the SSD and is read by rows (D-035);
- vLLM's KV pool is 16.2 GiB, sized for its concurrency;
- the drafter adds 1.6 GB here.

**The sweep** that chose the defaults (`spark`, the same prompts and runs,
13:28–13:48, medians of three):

| Depth, draft rows | `prose` tok/s (acceptance) | `code` tok/s (acceptance) | A step, ms |
| --- | ---: | ---: | ---: |
| 2, 32,768 | 39.71 (0.539) | 37.15 (0.469) | 51.5 |
| **2, 65,536** | **42.56** (0.631) | **38.73** (0.539) | 52.5–52.9 |
| 2, 98,304 | 41.65 (0.641) | 38.52 (0.556) | 54.0 |
| 2, 131,072 | 40.78 (0.641) | 37.70 (0.556) | 55.4–55.6 |
| 2, all | 37.55 (0.641) | 34.53 (0.556) | 59.9–61.0 |
| 3, 32,768 | 39.29 (0.449) | 35.73 (0.378) | 58.8–59.0 |
| 3, 65,536 | 40.50 (0.503) | 37.21 (0.436) | 60.9 |

Past 98,304 rows, acceptance stops rising while each draft pass pays about
0.63 ms for every further 32,768 rows of BF16 head. A depth-2 verify is
46.7–47.0 ms and a depth-3 verify 52.5–52.7 ms, against 37.4 ms for a plain
step on the device.

**Plain decode, with decode graphs** (`jitllm_swap_pairs --a qwen38
--cycles 0 --bench 128`, `spark-b`, the runtime wake, each generation a
request), two runs at 13:53–13:58 of three measured passes each:

- **With graphs:** 25.88 and 25.65 tok/s (means; the passes span
  25.32–26.06). A step is 37.4–37.8 ms of wall time and 37.4–37.7 ms on
  the device, with a round trip of 0.04–0.07 ms.
- **Launch by launch, the same session:** 24.05 and 23.84 tok/s (40.3–40.6
  ms a step).
- **Against Mia's MTP off** (25.12 / 25.33): 1.01–1.03×.
- **Exactness:** 0 steps' logits differ between graphs and launch by launch.
- **The earlier build** (`spark-b`, 12:13–12:31): 25.91 / 25.92.
- **The speculation harness's plain runs** (`spark`, prompts from
  prompts.json): 25.5–26.1.

**Profile.** Nsight Systems, kernels' busy time summed per step, so slightly
inflated by tracing.

A **plain step before graphs** (`spark-b`, 23.97 tok/s, a wall of 41.7 ms,
40.4 ms on the device):

| Part | ms |
| --- | ---: |
| MXFP8 vector products | 15.3 |
| Hyper-connection products (cuBLAS) | 6.0 |
| Routed experts' vector products | 6.3 |
| BF16 head (MMVF) | 5.2 |
| `HcPrep` (one block a token) | 1.37 |
| Gaps between launches | 3.0 |

The n-gram rows' reads took 0.54–0.60 ms a chunk, outside the job; they
now poll their completion instead of sleeping on it.

What changed with graphs:

- The ~1.3 ms of host work outside the job that qwen38-native measured was
  already gone once a request leased its closure once (D-093). Its round
  trip is 0.03 ms.
- Graphs remove the launch gaps. The job's host time falls from 18.2 ms
  (launching into the stream) to 0.07 ms, and the device time from 40.4 to
  37.4 ms.
- The step's wall time now equals its device time within 0.05 ms.

A **depth-3 verify** (earlier build, 4 rows) kept the device busy about
55 ms, against 37.6 ms for a plain step:

| Part | Verify, ms | Plain step, ms |
| --- | ---: | ---: |
| Routed experts | 18.7 | 6.2 |
| MXFP8 products | 16.6 (after the multi-row kernel) | 14.9 |
| Float products (`mul_mat_f`: head rows and others) | 6.7 | |
| Commit | 0.9 | |

The dense weights cost about what one row costs. The routed experts, which
each row selects for itself, are the difference.

## Judgement calls

- **The drafter is its own artifact**, not a new target import (above).
- **Commit instead of snapshotting the recurrent state.** Unlike a KV
  cell, the recurrent state is one accumulator per layer, 113 MB over the
  36 layers (F32 128 × 128 × 48 each). A snapshot taken before the verify
  can only return it to the anchor. Returning it to "after the kept rows"
  would mean either re-running those rows, or writing one state per row
  (vLLM's approach: 4 × 113 MB each verify). The commit writes the state
  once, from the rows the verify saved (about 0.9 ms for a 4-row verify).
  Its arithmetic is the verify's own, so the kept rows' state is what the
  verify computed.
- **Two jobs a step, not one.** The verify's n-gram rows can only be read
  once the drafts exist. The host reads them from the SSD through io_uring
  between the draft job and the verify job. Chaining the two jobs would
  need the rows of every candidate token prefetched, or the 28.8 GB n-gram
  table resident.
- **Depth 2 and a 65,536-row draft head**, measured (above). Mia's recorded
  comparator used depth 3 and its curated 47,172-row head; the earlier
  description of that head as full vocabulary was incorrect.
- **The grouped expert product was tried and dropped.** A variant that
  read each distinct expert once for every verify row that chose it,
  bit-identical per slot, was slower on the GB10: 21.5–21.9 ms a
  4-row verify against 18.7 ms for the per-slot kernel, even after its x
  quantization was staged in shared memory and its sums were sized to the
  row count. The per-slot kernel already runs at about the bandwidth of the
  distinct experts, with L2 serving the duplicates. So "each expert read
  once per verify" holds in effect, through L2, not by construction.
- **The near-tie margin is 1.0 logit, and it was not set by the rule
  later slices use.** It is qwen38-native's bound: the 95th percentile of
  the margin move between two of jitLLM's plain-decode plans. dsv4-decode's
  rule ("The bound, going forward") asks for the 99th percentile of the
  verify's own rows against one-row decoding, recorded before the
  comparison. That noise was not measured before this slice's comparison;
  the review measured it afterwards ([below](#the-verifys-own-noise-post-hoc)).

## Limits and what remains

- **Rollback across a swap** passes (above), with the commit owed across
  the swap: it relies on D-090's pinned places and on the commit region
  staying mapped, as DSpark's pending restore does.
- **The forced check's control** runs the same commit, so it cannot see a
  commit that is wrong the same way in both runs. The kernel's unit test
  (against the verify's own recurrence) and the near-tie rule on the tokens
  cover that part.
- **A harness, not the runtime**: `jitllm_qwen38_spec`, like
  `jitllm_spec_runner`.
- **Speed leads, none measured:**
  - Plain decode's head is about 1.3 GB of BF16, 5.2 ms of a 37.4 ms step.
    An MXFP8 or NVFP4 copy would halve or quarter it, but it changes the
    logits, so it is an owner call.
  - The hyper-connection products run as small cuBLAS GEMMs, 6.0 ms a
    step. A fused kernel is the obvious lead.
  - A quantized draft head would affect acceptance only (about 1.2 ms a
    step at N = 65,536).
  - Prefetching the n-gram rows of each draft's top candidates would let
    draft and verify be one job.
- **Prefill with the injection** equals the plain prefill bit for bit. Its
  cost is not timed here (the drafter's pass adds one layer per chunk).
