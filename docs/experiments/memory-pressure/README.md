<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Memory use and reclaim under pressure (2026-10-02)

The owner's memory policy (2026-10-02): use memory fully, scale down
gracefully under pressure, and never use fixed cache counts. This report
records what each reclaimable thing costs to give back and restore on a
GB10, the order derived from those costs, the design built on it (D-055
and D-090 as amended 2026-10-02), and its checks. Everything ran on
`spark-b` (`spark-56f5`) with the `spark-native` build, against base main
`64faee6` (the work was then rebased onto `eb2bd43`, where the
sustained-pressure runs below ran, and onto `6052286` for the review
round's checks).

## Results

- **One reclaim order by measured cost** covers graphs, plans and idle
  conversation state (and idle weights once partial eviction produces
  them). Measured cost to restore a byte freed: idle weights 0.08 s a GiB,
  idle state 0.17 (0.07 unchanged since its last spill), graphs 0.2–0.4,
  plans 2–15. Recomputing state costs 64 s a GiB, so idle state is
  spilled, never dropped while spill has room. The order is GreedyDual-Size
  over those costs, so stale plans and graphs fall behind fresh idle state
  ([review round](#review-round-2026-10-03)).
- **No fixed counts.** Plans and graphs are charged inside the execution
  budget and grow into free room; a swap keeps both models'. A prepared
  return now plans nothing (0.000 s against 0.04–0.11 s).
- **Smaller plans.** A plan's arena holds what its graph uses: DeepSeek's
  plans shrink 3.2–4.1× (a chunk plan 9.9 → 2.4 MiB, a wave plan 18.5 →
  5.3 MiB), at 16–32% more planning time (a second, exact build).
- **More state room.** The start's guard sets apart one step's plans
  (0.03 GiB) instead of the caps' bound (0.63 GiB). DeepSeek 262K DSpark
  beside Qwen3.8 33,792 MTP: conversation state room 1.31 → 1.92 GiB
  (+0.61 GiB, +47%) in the same session.
- **The two-model swap table runs.** With DSpark, `swap-table --pairs
  deepseek:qwen3.8` stopped at its state snapshot on the base (the guard's
  margin counted twice). Now all six rows are exact, 8.0–9.8 s.
- **Spill, don't clear.** An idle conversation reclaimed under pressure
  resumes in 2.48 s with 20,012 cached tokens (control without pressure
  2.29 s). The base clears it and prefills again: 30.18 s, 0 cached. Every
  reply is identical to the control's.
- **Pressure from outside without thrash.** A process holding MemAvailable
  under the trigger got 2 trims from the new build (one target-sized
  reclaim each, then 60 s back-off) and none during a request. The
  pre-hysteresis build trimmed 70 times and rebuilt 142 plans and 123
  graphs ([below](#sustained-pressure-from-outside)).
- **No regression.** Swaps are within noise of the base (7.77 / 9.57 s
  against 7.80 / 9.54 s). DeepSeek's HTTP cells are within 0.4%, and
  every reply is identical. Qwen3.8's C4 is level, and its C1 and C2 are
  1.2–2.0% lower. That comes from the new slot choice warming an empty
  slot's plans, not from the memory work ([below](#http-cells)).

## Stage 1: what each thing costs

### Plans and graphs (DeepSeek)

`jitllm_spec_runner --check plan-memory --slots 4 --context 32768
--max-rows 4096 --tokens 16`, DSpark and plain. Each new shape is measured
alone: process heap (`mallinfo2`), the plan's arena, planning time,
capture and instantiation time, and what the runner counts. A kind's first
shape also loads its kernels and is left out.

| Kind (4 slots, fast plan) | Heap before, MiB | Arena before (used), MiB | Planning before, ms | Heap after, MiB | Counted after, MiB | Planning after, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| One-row chunk plan | 9.9 | 9.35 (1.84) | 23 | 2.4 | 3.07 | 26.8 |
| 4,096-row prefill plan | 10.1 | 9.35 (2.49) | 41.6 | 3.2 | 4.34 | 49.5 |
| Decode wave plan | 18.4–18.5 | 17.26 (4.02–4.07) | 72–75 | 5.2–5.4 | 6.47 | 95.5 |
| DSpark wave plan | 18.5 | 17.26 (4.28) | 80 | 5.5 | 6.85 | 106 |

**Why 10–20 MiB.** About 90% of a plan was the GGML tensor arena sized
for the builder's worst-case tensor estimate (26,656 tensors for a wave);
a graph used a fifth of it. The rest is the plan's steps and launch
records. `SizedArena` (`engine/planned.h`) now builds each plan twice: once
on a reusable scratch arena of the estimate, then on an exact arena of
what that build used (the arena answers the builder's up-front checks for
the estimate, then is sealed). The second build costs more planning
time once per shape (16–32%); nothing else in a plan was avoidable.

| Kind | Heap, MiB | Device, MiB | Counted, MiB | Capture and instantiate, ms |
| --- | ---: | ---: | ---: | ---: |
| One-row chunk graph | 15.6–20.5 | 20–27 | 39.3 | 8.2–13.7 |
| Decode wave graph | 33.7–38.3 | — | 76.8 | — |
| DSpark wave graph | 36.9–41.0 | — | 82.2–90.8 | — |

Graphs are not serializable; their cost is the capture. Replays add
nothing.

### State, weights and recomputation

| What | Measured | Seconds a GiB |
| --- | --- | ---: |
| Weight page-in (swaps, `swap-table`) | 13.3 GB/s | 0.08 |
| Spill of all state (four slots, 1.42–1.52 GB) | 0.132–0.146 s, 10.4–11.0 GB/s | 0.10 |
| Restore of the same | 0.097–0.105 s, 14.5–14.7 GB/s | 0.07 |
| Spill and restore together | | 0.17 |
| Re-prefill, DeepSeek, 16,384 tokens (368 MiB of state) | 22.9–23.0 s, 713 tok/s | 64 |

`--check plan-memory`'s `reclaim-costs` section: prefill timing, then
eviction and reload of all state through the node.

### The order derived

| Kind | Cost to restore, s a GiB | Order at equal recency |
| --- | ---: | ---: |
| Idle weights (another model's; none under a full swap) | 0.08 | 1 |
| Idle conversation state, spilled (0.07 when unchanged since its last spill) | 0.17 | 2 |
| Captured graphs | 0.2–0.4 | 3 |
| Plans | 2–15 | 4 |
| (Recomputing state by prefill) | 64 | never chosen |

The runtime measures its own costs as it runs: each plan's planning time,
each graph's capture, and its spills' and restores' rates (11.0 and 14.5
GB/s until it has measured its own). In the service, the logged costs
were graphs 0.33–0.45, plans 1.9–8.3 and idle state 0.17 s a GiB.

The first build sorted kinds by these costs first. The review showed why
that is wrong: plans and graphs grow with every shape a workload meets
(Qwen3.8 over 9 turns: 149 plans of 312 MiB, 93 graphs of 1,157 MiB;
DeepSeek C4 at 9K: 38 plans, 10 graphs, 416 MiB), so kind-first sorting
spilled an idle conversation for each of 16 one-off requests while half
the room held plans and graphs no request used again. The owner's intent
is the expected cost of a byte freed, which depends on the chance it is
used again. The order is now GreedyDual-Size: a candidate's priority is
the inflation value L at its last use plus its kind's cost a GiB, the
lowest goes first, and each reclaim raises L to what it took. The table's
order holds for candidates used at the same time. A plan left unused
falls behind an idle conversation used after reclaims raised L past their
cost difference. Within a kind the order is strict least recent use.

**Plans to disk.** A plan costs more to rebuild a byte (2–15 s a GiB) than
a spill and restore (0.17), so the owner's criterion is met per byte, and
plans grow with use. It is not built: a plan holds host pointers (tensor
data, launch records) that a page-out would have to keep valid. It stays
open.

## Design

- **`memory/reclaim.h`.** A candidate is a kind, an owner, an id, its
  bytes, its last use, the inflation value at that use, its measured
  restore seconds and whether its model is running. `ReclaimOrder` sorts by
  GreedyDual priority (inflation plus the kind's cost a GiB: sum of seconds
  over sum of GiB), then kind, the running model's last, least recently
  used, never largest first. `SelectReclaim` takes victims in that order
  until the bytes needed are covered, counting a plan and its graph once,
  only below a priority bound if given; it says whether they suffice.
  `Server::Reclaim` takes all of it or nothing (pressure from outside takes
  what it can), no more than needed, and raises the inflation to each
  victim's priority.
- **Incremental spill.** Each runner's slot keeps a `SpillTrack`: once a
  spill wrote its state back, the file holds it, and every write since is
  recorded (the first position a job wrote from, through the model's
  checkpoint write ranges; ranges a checkpoint copied in). A clear, a
  diagnostic restore or a failed spill loses the record. A spill, or a
  swap's write-back, writes only the extents the record covers; the rest
  are evicted with `EvictOptions::unchanged`, which the scheduler honours
  only if the catalog's `saved_generation` equals the extent's content
  generation, so nothing is written and the eviction completes preserved.
- **Pinned staging reclaims too.** `PagedNode::Pinned` asks the reclaimer
  for its shortfall before refusing, as `ChargeHost` does, and the runtime
  sets the turn checkpoints' staging apart at start
  (`PagedNode::ReserveStaging`), so a budget full of caches never refuses a
  checkpoint. A checkpoint that is not saved or restored is logged.
- **Plans and graphs are accounted.** `PlanCache` (`engine/planned.h`)
  keeps every shape it is given, as `unique_ptr` entries that never move.
  `PlanAccount` charges each plan's bytes when it is added (required) and
  each graph's before its capture (refused if it does not fit even after
  reclaiming what costs less to restore than a graph; that shape runs
  launch by launch). `PlanStep` marks a step:
  what a step under way has used is never a candidate. `PagedNode::ChargeHost`
  turns everything past the start's floor into one pinned `kRuntime`
  extent inside the budget, so the catalog sees it. When a charge does not
  fit, the node's reclaimer runs the order. Qwen-Image's recorded step is
  charged the same way.
- **The floor.** `Served::plan_floor_bytes` is what one step of a model
  holds at once: DeepSeek 7.5 MiB (a wave's plans, or a chunk's beside its
  draft), Qwen3.8 a paired unit's target and draft waves together. The
  guard sets apart the largest, and beside it the one scratch arena every
  plan's first build uses (shared by the process under a lock, at the
  largest estimate). It used to set apart the caps' bound, 0.63 GiB.
- **Spill, don't clear.** A runner's slot can `Spill` its idle state
  (leaving every closure, written back through the node, backing released)
  and `Restore` it (refused for capacity with the same typed refusal as a
  growth). `Llm::SpillIdle` spills a conversation the order chose;
  `Llm::ReusePrompt` restores it at the next turn's first unit, and a
  capacity refusal there is deferred and retried after a reclaim like a
  chunk's. A cohort member preempted for its peers is spilled
  (`SpillSetAside`) and resumes its generation from its restored state;
  only if its state cannot be spilled is it cleared and prefilled again.
- **Retention, spill budget and pressure** (`Server::Maintain`, every 250
  ms between units and while idle). Conversations unused for
  `[memory] retention_hours` are dropped, resident or spilled. Past
  `spill_budget_gib` the least recently used spilled conversation is
  deleted (a swap's written-back state included; before a swap-out the
  outgoing model's least recently used conversations are deleted if its
  write-back would pass the budget). Under 512 MiB MemAvailable one reclaim
  asks for what restores 1 GiB of headroom above the mark
  (`runtime/pressure_trim.h`); a full memory stall of 5% over 10 s
  (`/proc/pressure/memory`, `platform/memory_pressure.h`) only raises the
  trigger to that target. Pressure ends only back there; while it
  persists, trims are spaced by a back-off doubling from 1 s to 60 s (60 s
  at once, and one log line a minute at most, when too little was left).
- **The running model's floor.** No reclaim takes the running model's most
  recently used plans up to its `plan_floor_bytes`, with their graphs
  (`memory::ProtectFloor`), nor what a step under way uses, so its next
  step finds them.
- **The snapshot fix.** `Server::SaveSnapshot` set apart the guard's margin
  again beside the snapshot's pinned staging, which is already charged
  within the budget. It now asks the node, reclaims through the order on a
  refusal, and retries.
- **Slot choice.** A new conversation used to take the free branch with
  the longest common token prefix, which the chat template's own tokens
  gave to every idle conversation. A new conversation therefore clobbered
  an idle one, even while an empty branch was free. The choice is now by
  `Llm::ReusablePrefix` (live history continued, or a turn checkpoint),
  then an empty branch, then the least recently used.

## Checks

### State capacity, spill and restore (GPU)

`jitllm_spec_runner --check capacity --slots 2 --state-budget-mib 1024
--context 131072 --max-rows 4096`, DSpark and plain. A slot fills the
room, the other's growth is refused with typed `state_refused` and no
state used, the idle slot is spilled (about 710 MiB written), the refused
growth then fits, and the spilled slot's restore is refused for capacity
cleanly and then succeeds once room is made. Its next-token logits equal a
control's that was never spilled. Both: 0 failures.

### Idle conversations under pressure (HTTP, one model)

`pressure.py` (scratch): a fresh DeepSeek service (production
configuration, DSpark, 262K), a host-memory hog that leaves about 1 GiB of
state room (1.96 GiB logged), three 20,014-token conversations A1, B1, C1
(64 outputs), a fourth D1 that does not fit beside them, then A's second
turn A2. The control has no hog.

| Build | D1 | A2 seconds | A2 cached tokens | Replies against the control |
| --- | --- | ---: | ---: | --- |
| New, under pressure | 1 idle conversation spilled (451 MiB) | 2.48 | 20,012 | identical |
| New, control | — | 2.29 | 20,012 | — |
| Base, under pressure | idle state cleared | 30.18 | 0 | identical |

To restore A, the new build spilled two more idle conversations (902
MiB). The service logged graphs 0.37–0.38, plans 5.4–5.6 and idle state
0.17 s a GiB.

### Two-model configuration

DeepSeek 262K with DSpark beside Qwen3.8 at 33,792 with MTP, started in
the same session:

| Build | Budget | Fixed | Set apart for plans | Conversation state room |
| --- | ---: | ---: | ---: | ---: |
| Base | 109.47 GiB | 7.71 GiB | 0.63 GiB | 1.31 GiB |
| New | 110.08 GiB | 7.71 GiB | 0.03 GiB | 1.92 GiB |

(Room is the budget less the fixed memory and DeepSeek's 100.45 GiB of
weights. Later starts on the new build logged 1.99–2.05 GiB as
MemAvailable moved.)

`swap-table --pairs deepseek:qwen3.8 --context-text
context-decisions-4655685.md`, DSpark:

| Row | Base | New |
| --- | --- | --- |
| Start | stopped: "initialized state snapshot would exceed the memory guard" | — |
| A→B first use, 8K context | — | 8.127 s, exact |
| B→A first use, 8K | — | 9.618 s (planning 0.108), exact |
| A→B prepared, 8K | — | 8.048 s (planning 0.000), exact |
| B→A prepared, 8K | — | 9.565 s (planning 0.000), exact |
| A→B prepared, 0 | — | 8.003 s, exact |
| B→A prepared, 0 | — | 9.769 s, exact |

A prepared return now keeps its plans and graphs and replays graphs
captured before the swap (the table checks it). At teardown DeepSeek held
8 plans (25.7 MiB) and 5 graphs (161.5 MiB).

Swap away and back with four idle DeepSeek conversations (`dsprobe-swap.py
swaptime`, the same configuration, each conversation about 2,000 tokens):

| Build | DeepSeek → Qwen3.8 | Qwen3.8 → DeepSeek | Resumed turns cached | Replies |
| --- | ---: | ---: | --- | --- |
| Base | 7.797 s | 9.535 s (restore 0.165) | 3 of 4 | — |
| New | 7.772 s | 9.569 s (restore 0.166) | 3 of 4 | identical to the base's |

The fourth conversation's slot went to the new request between them,
least recently used, in both builds: a slot holds one conversation (M6's
entries lift this).

### HTTP cells

The [deepseek-batching](../deepseek-batching/README.md#http-cells)
protocol for DeepSeek (7,043-token prompts, 256 outputs, production
configuration, a fresh service a cell) and the frozen common-v2 client for
Qwen3.8 (8,256-token prompts, 33,792 context, MTP). New and base ran
alternately in one session, completed tok/s:

| Cells | C1 | C2 | C4 |
| --- | ---: | ---: | ---: |
| DeepSeek DSpark, new | 13.47 | 14.24 | 14.44 |
| DeepSeek DSpark, base | 13.46 | 14.27 | 14.40 |
| DeepSeek plain, new | 11.12 | 13.29 | 14.45 |
| DeepSeek plain, base | 11.15 | 13.31 | 14.50 |
| Qwen3.8, new (two runs) | 25.44 / 25.39 | 29.57 / 29.46 | 31.99 / 32.31 |
| Qwen3.8, base (two runs) | 25.74 / 25.82 | 30.00 / 30.05 | 32.00 / 32.44 |
| Qwen3.8, new with the base's slot choice (one run) | 25.61 | 30.05 | — |

Every reply is identical to the base's, cell for cell. DeepSeek is within
0.4%. The service's peak RSS fell 90–210 MiB at C2 and C4 with the
smaller plans.

Qwen3.8 is 1.2–2.0% slower at C1 and C2 because of the slot choice, not
the memory work. The cells' excluded prime leaves a conversation in slot
0. The base gave the measured request that slot, displacing the prime's
state, and with it the plans and graphs the prime's steps had warmed.
The new choice keeps the idle conversation and uses an empty slot, whose
per-slot plans and graphs are built cold. With the base's choice
restored for one run, C2 matches the base (30.05) and C1 is within 0.6%.
At C4 every slot is used in both builds, and the cells are level. Steady
state, with every slot warmed, does not pay this.

### Sustained pressure from outside

`hyst.py` and `keeper.py` (scratch) run a fresh DeepSeek service
(production configuration, DSpark, 262K) and send a one-token prime. A
keeper process then holds MemAvailable near 350 MiB, under the 512 MiB
trigger, with locked memory. It takes back whatever the runtime gives
and yields only when the runtime grows. Under it, two 7,043-token
requests (u0, u1; 256 outputs) run one after the other, then 30 s idle.
The pre-hysteresis build is the same tree before this change: a trim at
every 250 ms look, asking for twice the mark, with no floor kept.

| Build | Trims (reclaim lines) | Plans / graphs reclaimed | u0 tok/s | u1 tok/s |
| --- | ---: | ---: | ---: | ---: |
| New, no keeper (two runs) | 0 | 0 / 0 | 13.13 / 13.45 | 13.44 / 13.45 |
| New, keeper (two runs) | 2 / 2 | 17 / 4 each | 13.36 / 11.94 | 12.01 / 11.88 |
| Pre-hysteresis, keeper | 70 | 142 / 123 | 11.09 | 10.96 |

Every reply is identical across all five runs. In the new build, the
first trim spilled the prime's idle conversation. It came as the keeper
reached its level, before u0. It gave 327 MiB of the 1.1 GiB asked for,
so it waited 60 s and logged once. The second came during the idle tail,
after u1 (17 of the running model's plans past its floor, 4 graphs, 2
idle conversations). No trim fell inside a request, and no plan or graph
was rebuilt there. The pre-hysteresis build dropped and rebuilt the
running model's plans and graphs about 0.7 times a second.

Requests under the keeper still ran up to 12% slower than without it,
with no trim during them. That is the host at 230–340 MiB available, not
the runtime: the kernel reclaims page cache, and the logged full stall
was 0.4–1.4%. The pre-hysteresis build lost 17–18%; that build's
rebuild loop accounts for the extra 6–7%.

### Review round (2026-10-03)

An adversarial review found one high and three medium defects; all are
fixed here. Runs were on `spark-b`, with the tree on main `6052286` and a
base build of `6052286`.

**H1: turn checkpoints refused on a full budget.** A budget full of plans
and graphs is now the steady state. Checkpoint staging was refused without
a reclaim, and the failure went unlogged. DeepSeek, whose template drops
earlier reasoning, then lost continuation reuse in 2 of 4 two-model runs.
Now pinned staging reclaims first, the checkpoints' staging is set apart
at start, and failures are logged. Unit tests fill the budget with plan
charges and still capture and restore.

The review's two-model repro (`adv.py spillswap`: DeepSeek 262K DSpark
beside Qwen3.8; four 20,014-token conversations; a Qwen turn; A2 and B2
back on DeepSeek) ran five times:

| Run | A2 cached / s | B2 cached / s | Replies against the review's control |
| --- | --- | --- | --- |
| 2 | 20,012 / 12.00 | 20,012 / 2.28 | identical |
| 3 | 20,012 / 12.01 | 20,012 / 2.26 | identical |
| 4 | 20,012 / 12.08 | 20,012 / 2.35 | identical |
| 5 | 20,012 / 12.94 | 20,012 / 2.33 | identical (A1–D1 too) |

A2's time includes the swap back (about 9.5 s). Run 1 is excluded. It
started with 1.6 GiB less available on the host, so the state room was
0.37 GiB. Every 20K-token conversation was correctly refused as not
fitting alone; no reply was produced to compare. No checkpoint failure
was logged in any run.

**M1: kind-first order and whole-state spills.** The order is now
GreedyDual-Size (above). Spills and swaps write back only what changed.
In the counted repro runs the write-backs released 0, 454, 681 and 454
extents (up to 1.4 GB) unchanged, writing nothing for them. The service logged idle state at
0.07 s a GiB when unchanged. `--check capacity` stays exact (0 failures,
DSpark and plain) with a second spill after one more step: restored, it
continues bit for bit. That check reserves the state through 63,488
positions, so almost every page lies past the write. It wrote 710 of
712 MiB; the saving shows in the service instead.

**M2: reclaims took far more than needed, and partial counted as
success.** A reclaim now takes all it was asked for or nothing, and no
more. A graph's capture takes only what costs less to restore than a
graph. A tiny need takes a small candidate of another kind over a
conversation when that costs no more to restore. In the repro logs:
369.1 MiB freed for 348.0 needed, and 60.7 for 50.0. Reclaims that took
nothing (too little to take): 0 to 6 a run.

**M3: a stall alone triggered trims.** Low MemAvailable is now required.
A stall only raises the trigger to the target.

**Low.**

- Qwen3.8's floor counts a paired unit's target and draft waves together
  (27.5 MiB).
- The teardown's plan line no longer underflows for a graph without a
  plan.
- The scratch arena is one for the process and counted beside the floor.
- A swap's write-back respects the spill budget.
- The Stats call before start that crashed a refused start is guarded.

Graph charges against what captures measured:

| Model | Counted | Measured at capture | Ratio |
| --- | ---: | ---: | ---: |
| Qwen3.8, C4 HTTP | 240.4 MiB | 161.4 MiB | 1.49 |
| Qwen3.8, three-model table | 61.7 MiB | 21.2 MiB | 2.9 |
| DeepSeek, two-model table | 161.5 MiB | 58.8 MiB | 2.7 |

16 KiB a node is therefore an upper bound on both models. The image
pipeline's recorded step is charged at its 16 MiB floor, and its
measured drop is now in its report. It was 0 bytes over three
generations (image as A, `swap-table --pairs image:qwen3.8`, pixels
`3b7770ca…` as expected): the recording took nothing measurable, so the
floor is the whole charge.

Afterwards, on `6052286` against its base build, completed tok/s:

| Cells | C1 | C2 | C4 |
| --- | ---: | ---: | ---: |
| DeepSeek DSpark, new / base | 13.75 / 13.73 | 14.55 / 14.57 | 14.73 / 14.78 |
| DeepSeek plain, new / base | 11.26 / 11.30 | 13.52 / 13.54 | 14.75 / 14.85 |
| Qwen3.8, new / base | 25.50 / 25.59 | 29.58 / 30.03 | 32.66 / 32.28 |

Every reply is identical to the base's. (A first DSpark run, 13.08 at C1,
overlapped a clang-tidy run on the host and is not counted.)

Swaps on the same tree:

- The two-model table's six rows are exact, 7.97–9.73 s. Prepared
  returns plan nothing.
- Qwen3.8 ↔ image (three models configured): six rows exact,
  5.13–6.50 s.
- Away and back with four idle conversations: 7.76 / 9.53 s.
- The sustained-pressure keeper run gave 2 trims, at 12.29 / 12.03 tok/s.

### Review round 2 (2026-10-03)

The second challenge confirmed the following hold:

- the H1 checkpoint fix;
- incremental spill exactness, on its rotation test for DeepSeek and
  Qwen3.8;
- set-aside, now reached end to end and exact;
- the stall-trigger fix.

It found one new high-severity defect and two remaining medium ones,
fixed here. All runs below are on `spark-b`, with the tree on main
`4a250b4` (DeepSeek's output-A/HCA prefill defaults, which change
DeepSeek's prefill numerics). Every repro is therefore compared against
a new control built from the same tree.

**High: a swap short by under one extent aborted the process.** It
happened in 1 of the review's 6 two-model runs. The reclaim for the
incoming model reported 50.3 MiB freed of 50.0 needed, but the plans'
charge moves in whole extents, so occupancy dropped less. The swap then
failed to load, and the cooperative backend aborted. Three fixes:

- `Server::MakeRoomForSwap` now goes through `runtime/swap_room.h`. It
  asks the order in whole extents and re-checks the room against the
  catalog after each reclaim, at most four times. A swap it cannot make
  room for is refused before anything moves.
- A swap that fails partway is undone (`Server::UndoSwap`): what came in
  goes out, and the outgoing model loads back whole. As built in this
  round, every forced failure path still aborted the runtime (review
  round 3, below).
- `tests/unit/swap_room_test.cc` forces a swap short by half a MiB against
  a ledger whose charge moves in whole extents. It is re-checked until it
  fits. A swap the order cannot cover is refused before it starts, and
  the attempts are bounded.

The review's two-model repro (`adv.py spillswap`) ran 8 times; 2 runs
were refused at start because the host had less memory free (97.4 and
106.6 GiB available). In the other 6 runs:

- A2 and B2 hit 20,012 cached tokens every time.
- Every reply was identical to a new control from this tree (DeepSeek
  alone), A1–D1 too.
- No swap failed, was refused, or aborted.

**Medium M1: stale plans never taken.** The chance of reuse now decays
with staleness directly. It halves with every reclaim since a candidate's
last use and every 5 minutes idle, so a plan at ~6 s a GiB unused for
three reclaims ranks below an idle conversation used just now (a unit
test holds this). On the review's `pg-r12` (16 one-off 6,000-word
requests at 1.2 GiB of room):

| Build | Plans held at the end | Graphs | Reclaims: conversations / plans / graphs taken |
| --- | ---: | ---: | --- |
| Review's tree | 79 (261.8 MiB) | 6 (163.7 MiB) | 16 / 1 / 34 |
| This tree | 24 (70.1 MiB) | 4 (46.3 MiB) | 15 / 106 / 27 |

The conversations still spilled are each one-off request's own state
need (about 350 MiB a request at that room): no cache is left to cover
it.

**Medium M2: optional captures spilled conversations.** Cache charges (a
plan, a graph's capture) now displace only other plans and graphs; state
is displaced only for state, swaps and pressure from outside. Across
every run above, no plan or graph reclaim spilled a conversation. A
refused capture is not asked again for the next 1, 2, 4 … 256 uses of its
plan (a unit test). Reclaims that took nothing fell from the review's
262–389 a run to 0–53.

**Low.**

- A cohort reclaims for each refused member on its own.
- Spill rates now count what was written.
- Set-aside is documented as verified end to end:
  - `sa-g2b` set a member aside (spilled at 3,069 tokens, restored).
  - All three set-aside runs equal their new controls.
- The docs now say that incremental spill saves writes for Qwen3.8
  (`rot-qw` released 452 extents unchanged; DeepSeek with DSpark covers
  every used extent).
- A failed spill clears the conversation and is logged (the I/O-error
  path is untested).

Exactness against new controls from the same tree:

| Repro | Result |
| --- | --- |
| `rot-ds` (DeepSeek, 4 conversations × 3 turns at 1.3 GiB room) | 12/12 replies identical; turns 2 and 3 hit 8,012 / 8,025 cached |
| `rot-qw` (Qwen3.8, 3 × 3 at 1.5 GiB) | 9/9 identical; 8,110 / 8,182 cached |
| `sa-dl150`, `sa-g2b`, `sa-p100` (set-aside) | 4/4, 2/2, 4/4 identical |
| `spillswap` × 6 | all identical (above) |

Performance on `4a250b4` against its base build, completed tok/s:

| Cells | C1 | C2 | C4 |
| --- | ---: | ---: | ---: |
| DeepSeek DSpark, new / base | 14.97 / 14.97 | 16.08 / 16.09 | 16.18 / 16.06 |
| DeepSeek plain, new / base | 12.10 / 12.14 | 14.85 / 14.87 | 16.49 / 16.53 |
| Qwen3.8, new / base | 25.64 / 25.49 | 29.60 / 29.62 | 32.12 / 32.31 |

Every reply is identical to the base's.

### Review round 3 (2026-10-03)

The third challenge confirmed that every earlier repro still ran exact,
but it injected failures and found every swap-failure path aborting the
runtime (rc 134):

- **Refused before moving:** the cohort marked its native work touched
  before the swap. Its retirement then tried to fence a model that was
  not resident, and the server aborted.
- **Read error mid swap:** `UndoSwap` tried to evict the incoming model's
  whole closure, shared workspace and pinned memory included. It failed,
  and the non-refused failure was treated as a node failure.
- **Read error during a first load:** the same path, treated as a node
  failure.

Fixes:

- The cohort marks native work touched only once its model is resident.
- `UndoSwap` evicts only what a swap moves (`Server::EvictPaged`: the
  incoming model's weights and state).
- A first load, or an undo that fails, leaves no model resident. The next
  activation first evicts whatever another model still holds, then loads
  its model whole.
- A failed activation fails only its requests with a 503 while the node
  is healthy (`Server::NodeHealthy`, the scheduler not faulted). Only a
  faulted node ends the process, by abort (signal 6), for its supervisor
  to restart (D-102).

The check is the challenger's harness (`adv.py inject` on a Spark-side
copy patched by `inject.py`, never in the tree). It fails the Nth storage
completion and adds a 1 TiB room shortfall to a chosen swap. Each run
sends 12 turns across DeepSeek (two conversations) and Qwen3.8, compared
with a control from the same patched tree:

| Run | Failures forced | Result |
| --- | --- | --- |
| `inj-run` | read error mid Qwen3.8 → DeepSeek swap; room short; two read errors mid DeepSeek → Qwen3.8 swap | 3 × 503, each swap undone; all 9 other turns identical |
| `inj-short` | room short | 1 × 503; 11 identical |
| `inj-ds2qw` | room short; read error mid DeepSeek → Qwen3.8 | 2 × 503, undone; 10 identical |
| `inj-early` | early read error mid swap; room short | 2 × 503, undone; 10 identical |
| `inj-first` | read error in DeepSeek's first load | that request 503; the next loads DeepSeek whole; every turn whose history the 503 did not change identical |
| `inj-undo` | two read errors aimed at the undo (both landed in the swap) | 2 × 503, undone; 10 identical |

Every run exited 0 and served to the end. These injections never reached
the undo itself: a failed swap reads its whole page-in (about 52,000
storage completions for DeepSeek, 38,000 for Qwen3.8) before the undo
starts. The fourth review round placed a second error in the outgoing
model's reload, in both directions, four runs: "could not be undone … no
model is resident", then the next activation of either model loaded
whole and cleaned up what was left, and every later turn was exact with
its cache intact (6,012 / 6,025 / 2,110 cached tokens). A failed device
copy (a node fault) exits by abort (rc 134) without hanging, for the
supervisor to restart.

### Unit and host tests

`mise run test -- spark-native --locked` on `spark-b`, on the tree rebased
onto main `2c35087`: 100% of 1,417 tests passed. New and rewritten
tests cover:

- the plan cache: no fixed count, charges, step protection, sized arenas,
  the refused capture's back-off;
- the reclaim order: its GreedyDual aging and staleness decay, all of it
  or nothing, the priority bound, a tiny need's small candidate, a plan
  and its graph counted once, and the running model's floor;
- room for a swap short by under one extent;
- an unchanged write-back writing nothing and restoring the same bytes,
  and checkpoints under a budget full of plans (reclaimed, and the
  reserved staging);
- the pressure trim: target, hysteresis, back-off and log rate, from fake
  readings;
- the `[memory]` keys and the pressure readers;
- the LLM paths: an idle conversation spilled for its peers continues
  exactly; a generation set aside resumes from its spilled state without
  prefill; the reusable prefix is the live history continued; an idle
  branch counts only resident state.

clang-format and clang-tidy are clean on the 60 changed C++ files (tidy
on the 37 translation units). The REUSE, header and boundary checks are
clean: 1,387 files, 1,248 headers, 386 boundaries.

## Open

- Plans to disk: not built (above).
- A conversation displaced from its slot by a new one is dropped, not
  spilled: a slot holds one conversation's state and its spill file. M6's
  entries (D-031) give conversations their own spill.
- Per-slot fixed buffers (DSpark's ~327 MiB a slot) are not lazy; the
  slot's state spills instead.
- Pressure from outside only trims: the runtime gives back through the
  order (at the back-off above) but never waits for the other process.
- Per-slot plans and graphs are built cold on each slot's first use
  (the Qwen3.8 C1/C2 cells above). Sharing them across slots would
  remove both the warm-up and the bytes.
- The double build adds 16–32% to each new shape's planning time.
- Requests arriving together are admitted one at a time. One that reuses
  nothing takes the least recently used idle slot, and with it the
  conversation a request queued just behind it would have continued.
  The swap-away-and-back probe's resume wave lost three of four
  conversations this way in one run, against one in the earlier runs.
  Admission does not look at the queue yet.
- The spill write budget (bytes a rolling 24 hours, from drive endurance)
  is not built; the spill budget bounds capacity only.
- Incremental spill saves nothing for DeepSeek with DSpark: its write
  ranges and the whole drafter region cover every used extent. Tracking
  the drafter's ring incrementally is open.
- The path where a spill fails partway (an I/O error) is untested. By
  design it clears the conversation and logs it.

## Provenance

Scratch on `spark-b`: `~/scratch/mempress/` (step files, harness copies
under `h/`, outputs per run). Raw logs stay there.
