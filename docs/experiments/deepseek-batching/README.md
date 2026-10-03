<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek V4 Flash concurrent request batching

DeepSeek chat now serves up to four requests at once in decode **waves**,
instead of one request at a time. On the matched 7K-token HTTP protocol of
[deepseek-concurrent](../deepseek-concurrent/README.md), the completed-token
rate at C4 rises **+29.0% plain** and **+4.5% with DSpark**. In these cells every reply is
byte-identical across C1, C2 and C4. Spark A (`spark-c4e2`), 2026-10-02.

## Design

- **Request slots.** `Dsv4Runner` keeps four request states. Each has its
  own live state (target caches and DSpark ring), verify snapshot, output
  staging and plans bound to its state. The weights, workspace, staging
  and launch context are shared. The cohort rules (active set, closures,
  fault handling) are the engine's new `request_cohort.h`. Serving maps
  four `Llm` branches onto the slots, as Qwen3.8 does, and the cooperative
  chat backend admits DeepSeek.
- **Wave graph.** `BuildDsv4WaveGraph` builds one fast-plan graph over
  every slot's rows. Row-local work runs once over all rows: every
  quantized product (`jitllm.vecq`, routed experts deduplicated across
  slots), the HC mixes, routing, combine, norms, rotations and the head.
  Each slot's compressors, indexer, attention, cache writes and DSpark
  injection run on its own state. The attention outputs are then joined
  again. A wave holds at most 16 rows.
- **Exactness.** A request's rows in a wave equal its rows alone, bit for
  bit:
  - **Verify rows.** `jitllm.vecq` gives each token the same arithmetic at
    any count from 2 to 16, so it now takes 16 tokens. GGML's float vector
    kernel is count-invariant only to 8 columns, so past 8 rows a wave
    runs the router, indexer-weight and head-mix products per slot.
  - **One-row steps.** A wave of these sets `SetVecQOneToken`, which gives
    each token the one-token launch. Weights are read again from cache for
    each token.
  - **Injection.** The drafter's injection products are GGML MMVQ, so they
    run per slot.
  - **One-row verifies.** These occur at a mask-width boundary or on the
    last token, and run alone.
- **Policy.** Plain models run decode waves. With DSpark, each slot drafts
  its own block, then one joined verify covers every slot's natural rows
  (up to 4). A lone request keeps its ordinary step. Prefill chunks
  alternate with peer waves (the cooperative backend's existing policy).
- **Capacity.** A slot's state growth that the execution budget refuses
  beside its peers' leased state is typed (`Slot::state_refused`), so the
  cohort's capacity policy
  ([runtime-serving](../../runtime-serving.md#state-capacity-in-a-cohort))
  clears an idle conversation's state first, then makes the request wait,
  and preempts the youngest when nobody can go on. A decode step grows its
  state in its preparation, through the most rows its verify may take, so a
  wave never grows a member's state.
- **Plan memory.** The slots share one cap of 24 chunk plans; waves keep 6
  plans; every graph (chunks', draft blocks', waves') counts toward 16
  graphs and 256 MiB, and a capture alone past that is not made. The runner
  reports the bound those caps allow, a swap drops the outgoing model's
  plans and graphs, and the memory guard sets apart the largest model's
  bound ([below](#plan-memory)).

## Controls

Harness: `jitllm_spec_runner --check wave --slots N`. The first N
decode/chat prompts of `fast-swap/prompts.json`, 96 tokens each.
Results are against each slot running alone in the same process.

| Check | 2 slots | 3 slots | 4 slots |
| --- | --- | --- | --- |
| Plain waves: rows byte-identical (teacher-forced) | 142/142 | — | 332/332 |
| Injected decode waves (DSpark loaded): rows byte-identical | 142/142 | — | 332/332 |
| DSpark waves: drafts and verify rows bit-identical, same tokens | all | all | all |
| DSpark: final state fingerprints equal | yes | yes | yes |
| DSpark: discarded wave verify, state restored (stale bytes) | 0 | 0 | 0 |
| DSpark: discarded step re-run equals solo | yes | yes | yes |
| A slot leaving the cohort: state untouched afterwards | yes | yes | yes |

The DSpark discard comparison excludes only the draft block's own ring
cells, which hold uncommitted positions.

On the final build (the plan caps and the cohort capacity policy, base
main `5a5a0b9`) every control above was run again with the same results.

Before these exactness fixes, the controls measured the following against
solo:

- Plain waves: argmax agreement 217/220, margin-move p99 1.82.
- DSpark: drafts differed at two steps because of the injection.

Unit tests:

- `dsv4_test` (wave structure): joined products, per-slot state, writes and
  refusals.
- `dsv4_fast_test` (vecq): 16-token and one-token launches bit for bit,
  dense and routed.
- `request_cohort_test`.

## HTTP cells

These use the [deepseek-concurrent](../deepseek-concurrent/README.md)
protocol unchanged:

- buffered non-streaming chat, greedy, 256 outputs;
- 7,043-token prompts u0–u3;
- a fresh service per cell, with an excluded prime;
- production config: artifact `8a355bfb…`, drafter `dd2d3f9c…`, context
  262,144, 4,096-row chunks.

Each cell is one sample. Every reply of the final build's cells is
byte-identical to the earlier run's, at every concurrency.

| Native | C1 tok/s | C2 tok/s | C4 tok/s | C4 vs before |
| --- | ---: | ---: | ---: | ---: |
| DSpark, waves | 13.312 | 14.069 | 14.268 | +4.5% |
| DSpark, waves, final build (plan caps, capacity policy) | 13.392 | 14.217 | 14.387 | +5.4% |
| DSpark, before (one at a time) | 13.442 | 13.515 | 13.654 | — |
| Plain, waves | 11.076 | 13.255 | 14.441 | +29.0% |
| Plain, waves, final build | 11.092 | 13.249 | 14.466 | +29.3% |
| Plain, before | 11.091 | 11.152 | 11.192 | — |
| llama.cpp b11254 DSpark / plain (same GGUF) | 8.839 / 7.635 | 9.503 / 8.949 | 10.078 / 9.793 | — |
| ds4 0.6.5, plain, community IQ2_XXS artifact | 13.177 | 14.590 | 16.395 | — |

- **llama.cpp.** Native now leads at every cell:
  - DSpark: +50.6% / +48.0% / +41.6%.
  - Plain: +45.1% / +48.1% / +47.5%.
- **ds4.** The comparison is not matched. ds4 ran a different artifact,
  through literal completions, which still run one at a time. On those
  numbers native plain trails by 15.9% / 9.2% / 11.9% (before: 16.1% /
  23.8% / 32.2%). Native plain C1 measured the same on both artifacts
  (11.06 / 11.09).
- **Latency.** All four requests now finish together (C4: 70.6–71.8 s).
  Before, the first finished at about 19–23 s and the last at 75–92 s.
- **Memory.** Peak `MemAvailable` drop at C4 is 109.85 GiB with DSpark
  (before: 107.83) and 98.44 plain (before: 97.04).
- **Remaining gap.** The cells are now prefill-bound: four 7K prefills take
  about 45 s of C4's 71 s. ds4 prefills about 1.7× faster.

Pure decode, from the harness (4 slots, waves of 3–4, short prompts):

| Mode | Solo | Waves |
| --- | ---: | ---: |
| Plain | 21.8 tok/s | 37.6 tok/s |
| DSpark | 32.8 tok/s | 37.4 tok/s |

DSpark gains less because a wave of four 4-row verifies reads about 78
distinct routed experts, plus four separate draft blocks.

## Plan memory

Plans and graphs are host and driver memory outside the catalog. Before
the caps, four slots' 32 chunk plans each, 32 wave plans and 16 graphs
(draft blocks' uncounted) could grow the service by about 2 GiB. The
adversarial churn-then-fill probe (`cf`, ten rounds of four distinct
prompt lengths, then four conversations filling the state budget) took
the service to 3.74 GiB RSS, `MemAvailable` to 0.40 GiB and 1.34 GiB into
swap: the 6 GiB margin meant for the driver and cuBLAS had been consumed.

Measured one new shape at a time (`--check plan-memory`, process heap by
`mallinfo2`; a kind's first shape also loads its kernels and is not
counted), against what the runner now counts (`PlannedHostBytes`: the
arena plus 512 bytes a launched node; a graph then 12 KiB a node):

| Kind (4 slots, fast plan) | Heap, MiB | `MemAvailable` drop, MiB | Counted, MiB |
| --- | ---: | ---: | ---: |
| One-row chunk plan | 9.9 | — | 10.6 |
| 2,048-row prefill plan | 10.1 | — | 11.2 |
| Decode wave plan | 18.4–18.6 | — | 19.7 |
| DSpark wave plan | 18.5–18.6 | — | 19.8 |
| One-row chunk graph | 15.6–17.0 | 13–25 | 29.5 |
| Decode wave graph | 33.0–38.3 | 43–56 | 56.6–57.6 |
| DSpark wave graph | 36.9–41.0 | ~50 | 61.6 |

Replays add nothing. A graph is counted at 16 KiB a launched node, 37% over
the largest drop measured (a decode wave graph's 56 MiB at 4,912 nodes;
its count is 77 MiB). The caps are sized from the working sets the
challenge measured in the C4 HTTP cells (up to 26 plans, 330 MiB, and 3
graphs, 143 MiB at 12 KiB a node): 24 chunk plans shared by the slots, 6
wave plans, one draft plan a slot, and 16 graphs within 256 MiB, least
recently used dropped. They bound the total at 647 MiB with DSpark.
Qwen3.8's runner gets the same treatment from its measured C4 working set
(24 plans, 250 MiB; 16 graphs, 105 MiB at 12 KiB a node): 24 chunk and 24
drafter plans shared by its slots, 4 target and 16 draft waves, and 16
graphs within 192 MiB, a bound of 628 MiB.

A swap drops the outgoing model's plans and graphs (D-090 as amended), so
the start's memory guard sets apart only the largest model's bound
(`plan_host_bytes` in its log line), not their sum. With every model's
kept, DeepSeek at 262K with DSpark beside Qwen3.8 at 33,792 with MTP (a
configuration that serves on the base) was refused at start (3.75 GiB of
plans set apart); now it starts with 0.63 GiB set apart and serves. Its
DeepSeek state room is the base's less 0.63 GiB: 1.35 GiB in these runs
(108.4–108.5 GiB available beside the fixed memory), and about 0.4 GiB on
the same host an hour earlier, when 1 GiB less was available.

| Two-model service (DeepSeek 262K DSpark, Qwen3.8 33,792 MTP) | Result |
| --- | --- |
| A DeepSeek cohort in flight, a Qwen3.8 request between its turns, swaps back and forth, against each conversation run alone | 24 replies identical, 0 different |
| DeepSeek → Qwen3.8 → DeepSeek, one idle DeepSeek conversation | 7.67 / 9.46–9.48 s (two runs) |
| The same, four idle DeepSeek conversations (every slot's state spilled and restored) | 7.72–7.75 / 9.51–9.52 s |
| `swap-table --pairs deepseek:qwen3.8` (DeepSeek plain at 32K), first use and prepared, 8K and 0 context | all six rows exact, 7.8–8.9 s; a prepared return plans again (0.04 s) |

Each return plans and captures again; the swaps stay far inside the 20 s
bound. (The same table with DSpark stops at its diagnostic state snapshot,
which the guard's margin refuses in this two-model configuration; the base
build refuses it the same way.) Matched HTTP cells after the tighter caps
(single model): DeepSeek
DSpark 13.41 / 14.28 / 14.39 and plain 11.11 / 13.23 / 14.50 tok/s at
C1/C2/C4; Qwen3.8 8K C4 32.70 and C2 29.89 against the base build's 32.56
and 30.02 in the same session.

`cf` again, after (with the first caps, 32 chunk and 8 wave plans and 768
MiB of graphs; two runs, the same probe and a hog sized for a similar
room): the service's RSS peaks at 2.44 GiB, and `MemAvailable` stays at
3.5–4.3 GiB through the churn (both runs) and the fill (the first; its four
conversations now all complete, [below](#capacity-under-pressure)). The only swap is the
hog's own idle pages, which the kernel pages out: in the second run its
RSS falls by exactly what swap gains, 0.59 GiB. One ~10-second dip to
0.58 GiB in the first run coincided with another session's 2m24s of CPU
on the host (the journal), as did similar dips during `react2` and
`fill3`; the service's RSS fell, not rose, through each.

## Capacity under pressure

The challenge's state-capacity probes, re-run against the cohort policy
(with the first caps).
A memory hog sizes the state room (1.6–1.9 GiB beside the 1.3 GiB plan
bound, about what the challenge had):

| Probe | Before | After |
| --- | --- | --- |
| `fill3`: four 27–29K-token conversations, 2,500 outputs each | three failed at the 16,384 chunk; the fourth completed, or failed mid-reply | all four complete: three wait, the youngest is preempted and rebuilt, a finished conversation's idle state is cleared |
| `react2`: an idle 40K conversation, three 24–33K growers, then its continuation | the three growers failed while the idle state stayed resident | all four complete: two growers and then the continuation wait, the continuation is preempted and rebuilt, a finished grower's idle state is cleared; the continuation prefilled its 40K tokens again (0 cached; the challenge's run reused 40,007) |
| `fill4`: two idle 26–27K ballast conversations, two decoders of 2,000+ tokens | both decoders failed mid-reply | both complete; a ballast's idle state is cleared |

Every probe ends with the service healthy and its short follow-up requests
answered. A request that waits runs nothing while it waits; a preempted
one rebuilds its state by prefill, so its later tokens may differ from an
uninterrupted run's, while tokens already streamed never change
([runtime-serving](../../runtime-serving.md#state-capacity-in-a-cohort)).

The runner's typed refusal itself (`--check capacity`, two slots and a
1 GiB state room): beside slot 0's 612 MiB, slot 1's growth to the same
is refused for capacity, its state usable and unchanged; a growth past
the context is refused, but not for capacity; a selected slot cannot be
cleared as idle; once idle slot 0 is cleared, slot 1 grows and runs.

## Not adopted, and open

- **Verify waves of 2 rows a slot at 3–4 slots.** These measured +8%
  pure-decode wave throughput (40.4 tok/s) and +4% at HTTP C4. They change
  each request's step form, so replies differ from C1. Not adopted, to
  keep byte-identical replies.
- **Injected decode waves instead of DSpark waves past 2 slots.** 14.27 at
  HTTP C4, no better than the adopted waves.
- **A multi-token vecq launch with the one-token reduction order.** Open.
  It would read each weight once while keeping one-row exactness, and could
  recover the ~10% cost of `SetVecQOneToken` on plain waves, as the wide
  MXFP8 kernel did for Qwen.
- **Joined DSpark draft blocks.** Open. A wave runs one draft per slot.
- **Literal completions (`/v1/completions`).** Open. They remain serial.

## Provenance

- **Host.** Spark A (`spark-c4e2`, GB10, driver 580.178.04).
- **Build.** The `spark-native` build of worktree `cc/dsv4-batching` (base
  main `bae7a0d`); after the plan caps and the cohort capacity policy, of
  worktree `cc/dsv4-batching-3` (base main `5a5a0b9`); after the swap-out
  drop and the measured caps, of worktree `cc/dsv4-batching-4` (base main
  `418887b`).
- **Harnesses.**
  - The HTTP cells used `deepseek-concurrent`'s `run.py`, pointed at this
    build.
  - The wave controls used `jitllm_spec_runner --check wave`; plan memory
    `--check plan-memory`, the runner's capacity refusal `--check capacity`.
  - The probes (`cf`, `fill3`, `react2`, `fill4`) are the challenge's
    `dsprobe.py`, with the state room read from the guard line (its plan
    bound included) and a memory hog sized for a target `MemAvailable`.
  - Supervised jobs `dsb-wave1`–`5`, `dsb-http1`–`3`; then `dsb2-pm4`,
    `dsb2-atk2`–`5`, `dsb2-final1`, then `dsb4-q1`–`q3` and
    `dsb4-swaptable4` (the base's refusal: `dsb4-swaptable-base`).
- **Admission.** Every model load passed the admission check: at least 105
  GiB available, no GPU process.
- **Raw records.** Kept outside Git in `spark:~/scratch/dsbatch/`
  (`wave*-*/spec.json`, `http/native-*-r3/`), `spark:~/scratch/dsb2/` and
  `spark:~/scratch/dsb4/`.
