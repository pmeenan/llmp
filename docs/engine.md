<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The engine

`src/engine/` is the layer above the kernels that runs models on a paged
node (D-096; [architecture.md](architecture.md#layers-and-dependency-rules)).
It holds the node, one skeleton of shared mechanics every model runner is
built from, and each model family's runner, which adds only what is that
family's own. `llmp-runtime` serves through it
([runtime-serving.md](runtime-serving.md)) and the paged harnesses drive the
same code (`tests/support/paged_node.h`).

**Being replaced (M3.6, D-107).** The per-family runners, plans and graph
builders described below give way to one engine of shared components over
llmpalooza's own graph IR ([engine-components.md](engine-components.md)). This
file describes the code as it is, and changes as each piece lands.

**Two libraries** (D-096 as amended 2026-10-08). `llmp_engine_core` builds
in every profile and names no kernel library: the node, the weights'
paging, live state (with its spill, checkpoints and a verify's snapshot),
the runner's own memory (`runner_memory.h`), what plans hold
(`plan_account.h`: the charge, the step a reclaim spares, the reclaimer's
view of a cache), prefill lookahead, the request cohort, the shared slot
lifecycle (`slot_lifecycle.h`, below), the n-gram rows, and the one model runner
over step programs with its wave composer and prefill orchestrator
(`model_runner.h`, [below](#the-model-runner)). It reaches the device through the providers, the device runtime and
the paging kernels (`kernels/paging/paging.h`), and links no device
backend: a program links the CUDA one, or the fake device runtime
(`providers/fake/fake_device_runtime.h`: host memory, work queued on fake
streams that runs when a fence is queried) with the paging kernels' host
loops (`llmp_fake_device_backend`), which the core's tests use in every
profile. `llmp_engine` (CUDA builds) adds the GGML plans and runs
(`planned.h`, `graph_runs.h`), the runners' cuBLAS and launch context
(`runner_resources.h`), `ExpertSlab` (`expert_slab.h`), the launch-context
path of a verify's snapshot (`live_state_launch.h`) and each family's runner.

## The node

`paged_node.h`: device 0 with its providers, a stream per model and the
copy stream, the storage ring, one catalog domain, the landing zone
(D-081), the scheduler and its lanes, and the workspace the models share
(activations and the GGML pool; in the runtime it pages with the active
model, each model's closures and steps bounded to its own prefix of it,
[runtime-serving.md](runtime-serving.md#registration-and-the-swap); so
two models never execute concurrently on one node, a constraint M4's
placement must revisit). A runner registers its memory and page
sources with it and posts its work through it: `Job` (one device job on the
model's stream over a closure), `Load`, `Evict`, `Swap`, `Call` (a function
on the scheduler's thread), and requests: `BeginRequest`/`EndRequest`, or
`WithRequest` around a body, lease a model's closure once for all its steps
(D-093). Its teardown fences each model's stream (a job that leases
nothing, so a model swapped out is not paged in again), evicts the managed
extents, stops the scheduler and only then calls each model's `Release`
(`PagedModel`). `ReleaseMapped` and `PlaceCheck` are the node's memory
helpers every runner uses.

## The skeleton

Composition, not inheritance: a runner holds these as members and calls
them; nothing in them is virtual, and nothing runs per kernel.

| Part | File | What it does | The runner supplies |
| --- | --- | --- | --- |
| Weights | `paged_weights.h` | Opens an artifact; registers its shards as the model's weight files (`PagedNode::AddWeightFile`), open for direct reads only while the model is active and size-checked as they open; reserves and catalogs, in file order, a 2 MiB-aligned device region per dense group, a slab per layer of routed experts (`LayOutSlab`; a GGML runner's `ExpertSlab`, `expert_slab.h`) and host regions for groups the CPU reads; registers every page source and span; checks the places still pinned; the resource and expert-array addresses a plan binds | Which group goes where (`GroupPlace`), the slabs and their alignment |
| Live state | `live_state.h` | Stable virtual regions registered before the weights, physical backing only for used extents; successful source/pin checks reused under scheduler lifetime/epoch with mutation invalidation (not residency or leases); sparse unnamed direct-I/O spill files, clear, packed copies, pinned places and quarantine; a verify's snapshot, accept, owed restore and commit, rollback, and undoing a failed verify, queued on the job's stream as the paging kernels' range copies (a GGML runner queues them through its launch context, `live_state_launch.h`) | The state layout and each step's used ranges; which ranges a verify writes; a commit if kept rows need one |
| Planned shapes | `planned.h` (`plan_account.h`, `plan_cache.h`, `coverage.h`) | `PlannedGraph<Graph>`, `SizedArena` (a plan's arena holds what its graph uses), `PlaceAndPlan` (placeless plan, activation placement, the same plan again), `BindPlanned` (pool scratch checked, implementations bound, D-053), `PlanCache` (per key, variants, no fixed number, each plan's host bytes and planning time kept: `PlannedHostBytes`; the core's `BasicPlanCache` over GGML's `PlanRuns`); in the core, `PlanAccount` (what the plans and graphs hold, charged to the node: a plan as it is added, a graph before its capture, D-090 as amended), `PlanStep` (a step's plans spared by any reclaim while it runs), `PlanCacheBase`, `CollectPlans` and `ReclaimPlan` (the plans and graphs as candidates for the node's reclaim order, and one's reclaim); `CheckCoverage` (BP-A1) | The graph builder and its binding (`*_plan.h`), the cache key, the tensor classes for the coverage check |
| Runs and graphs | `graph_runs.h` | `GraphRuns`: stages a run's inputs in the pinned staging, queues copies, work between inputs and plan, the plan and the outputs; captures a shape on its second run (or when the runner asks), beside that run's launch-by-launch execution (so capture and instantiation overlap the device's work) unless work is queued between inputs and plan, replays its graph from then on with the staging checked, falls back to launch by launch on a refused capture; `CaptureAhead` records another planned shape's graph without running it, from the input layout `Layout` predicts; `GraphStats`, `RunPath` | When a run may be captured (decode steps, verifies, drafts, hinted prefill) and what it copies out |
| Prefill lookahead | `prefill_lookahead.h` | `PrefillLookahead` and fixed-capacity `PrefillLookaheadGroup`: independent optional host allowances before CPU-only future plans, sequential builds during current device work, ordered installation after successful completion, exact cache-charge transfer and destruction before unused-grant release | Shape prediction, graph construction, binding/coverage/cache installation and capture policy |
| Resources | `runner_memory.h`, `runner_resources.h` | The runner's own device memory (pinned) and pinned staging (`RunnerMemory`, the core's); for a GGML runner, cuBLAS and its workspace, a measuring launch context, the launch context over the pool and the registry (`RunnerResources`); and their completion-aware release (AGENTS.md rule 6) | Sizes and names |
| Request cohort | `request_cohort.h` | Several request slots of one model: the active set selected between completed units (several only under one held stream request), the closures (everything, the state fence, the execution closure, each slot's fence) and the held request's refresh, and the cohort's fault (every slot quarantined until retirement) | Its slots' live states and the shared extents |
| Slot lifecycle | `slot_lifecycle.h` (`model/state_contract.h`) | Request slots of one LiveState per stage (target, then each drafter) and every verb that moves their state, a verify's Accept, discard and settle among them, derived from the stages' state contracts, with the reconciled guards ([below](#the-shared-slot-lifecycle)); the model runner's, so Gemma 2's, Gemma 3's and Gemma 4's (the other families' runners move to it as they migrate) | The stages' contracts and read alignments, the shared extents, the spec's digest, its invalidation and plan-dropping hooks, and a write record before every job that writes a slot |

Startup measurement can skip greedy activation placement when a checked
disjoint root-storage bound fits the current exact maximum. Gemma2/3,
DeepSeek/DSpark and native/GGUF Qwen select it; Gemma4 keeps its optional
dense-sharing policy. Qwen scalar and wave ceilings are independent.
Measurement-only plans cannot bind, and ordinary runtime plans retain exact
placement. [Startup controls and scope](experiments/startup-placement-transfer/README.md).

`support.h` holds the small helpers (errors, addresses, rounding, seconds,
joined problems).

Gemma2, Gemma3, both Gemma4 profiles, DS/DSpark and native/MTP/GGUF Qwen use
the shared prefill lookahead lifecycle.
Their family adapters still choose future shapes, outputs and capture eligibility;
`GraphRuns::CaptureAhead` records an eligible future graph without executing it.
Gemma2/Gemma3 select two distinct missing futures. Both Gemma4 profiles now
compose the same fixed group, but ordinary policy retains one future and no
capture ahead: the [plain Gemma26 8192-row diagnostic screen](experiments/gemma26-capture-ahead/README.md)
replays six chunks without a meaningful paid gain. The separate
[retained-frontier opt-in](experiments/gemma26-feature-capture/README.md)
measures −0.484% prefill and exact assistant consumption; ordinary and
`SetupAssistant` capture policies still remain off/one. Cached current/next/after
keys are protected before optional capture funding in eligible adapters. Each
future has independent refusal and cleanup; the CPU callback publishes no future
logical state or cursor. DS/Qwen scalar adapters now select two-stage CPU planning
and independently prepare eligible fresh backing before Job, draining it after
completion and before current publication. Qwen funds two target plus two MTP
plans; its companion prediction uses post-success pending state. DS preserves
its external Status meanwhile contract.
[Actual controls and factors](experiments/ds-qwen-prefill-prediction/README.md).
`PromptSession::NextPrefillHint` supplies the same
checkpoint/scoring-aware descriptors to scalar and joined Gemma2/Gemma3 prefill
after current funding. Each future stage independently filters ended owners and
requires homogeneous head intent; a mixed next stage does not suppress a valid
after stage. Gemma4 serving still admits one prompt chunk at a time; its joined
prefill admission/funding remains separate work. [Focused lifecycle controls](experiments/prefill-transfer/README.md).

Gemma2/Gemma3 compatible two-owner 2–128-row prefill reads the real F16 K/V
roots through the existing MMA body, keeping the packed logical geometry and
cap50/cap0 arithmetic while removing K/V concatenation. Mask concatenation
remains. The cap0 primitive is qualified through 131072 cache cells; public
contexts and the 128-row chunk default are unchanged. Original selected
Columns4/8/16/32 geometry and padded-query tail semantics cover partial chunks.
Larger compatible rows, mixed-width roots and cohorts retain checked fallback
and explicit extension tasks. [Attribution and qualification](experiments/gemma-prefill-copies/README.md).

Gemma2/Gemma3 default to a shared per-graph exact-input Q8 cache for selected
original MMVQ products. Transient kernel-owned device choices gate the graph
before creating preparations; empty/false choices and MMQ routes remain ordinary.
The original quantizer, consumer reduction and one-column fused GeGLU are
preserved. Setup measures all ordered owner-row compositions through total eight
inside configured budgets, as well as its existing endpoint envelopes, so
intermediate MMVQ storage is funded independently of maximum-row MMQ plans.
The shared helper also preserves Gemma4's distinct diagnostic VecQOneToken
policy. [Bounds, native factors and startup cost](experiments/gemma-shared-q8/README.md).

## A runner's life

1. **Setup** (before the scheduler): open the artifacts
   (`PagedWeights::Open`), bind the model (`model/*.h`), lay out its state;
   add the live-state regions first, then any scratch the model keeps
   (`RunnerResources::Map`), reserve the weights, open cuBLAS; measure the
   largest shapes over placeless addresses (`MeasuringContext`,
   `PlaceAndPlan` with no activations) to size the activations, pool and
   staging; pin the staging (`RunnerResources::Pinned`, then
   `GraphRuns::SetStaging`); map a verify's snapshot if the model
   speculates (`LiveState::SnapshotAt`, `AllocateSnapshot`).
2. **Register** (after the node's Start): each shard a weight file of
   the model's and each weight's source reading it by identity
   (`PagedWeights::Register`; the shards open at the model's first
   page-in or activation, `PagedNode::SetActive`), the state's write-back places
   (`LiveState::RegisterSpill`), then every managed extent's place pinned
   (`Scheduler::PinPlaces`, D-090).
3. **Bind** (after the workspace): the closures (`everything`, the state's
   fence, any narrower ones; of the workspace, only the model's own prefix
   of each region, `ModelRunner::activation_bytes` and `pool_bytes`), the
   model's places as the plan's address functions, the commit hook, and
   the launch context over that pool prefix (`RunnerConfig::launch`, then
   `GraphRuns::SetLaunch`).
4. **Steps**: each chunk, draft or verify holds a `PlanStep` for its life,
   checks `LiveState::Usable` and
   `AwaitingAccept`, finds or plans its shape (`PlanCache::Find`, else plan,
   `BindPlanned`, `CheckCoverage`, `Add` with its `PlannedHostBytes` and
   planning time, charged as it is added), decides whether to capture
   (`PlanRuns::CaptureDue` plus the model's rule, then `ChargeGraph` for
   every capture of the job: a graph with no room even after a reclaim is
   not made), and
   materializes its used ranges (`LiveState::Use`) and renews the request
   closure if new extents were initialized, then posts one job that queues what the live state owes, a
   verify's saves (through its launch context: `QueueOwedThrough`,
   `QueueSavesThrough`), and the run (`GraphRuns::Queue`). A
   failure settles the state (`LiveState::Settle`: a verify undone, else a
   quarantine); success counts the path (`Count`).
5. **Release** (after the node's teardown): the plans and graphs first
   (they name the launch context and the memory), then
   `RunnerResources::Release`, `LiveState::Release`, each
   `PagedWeights::Release`.

## Independent request state

`Qwen38Runner` exposes stable request slots (`Qwen38Options::request_slots`,
the model's cap in serving; at most `kMaxRequestSlots`, 16,
`request_cohort.h`). Each owns its target and MTP
live state, verify snapshot, commit state, pending cursor and plan caches. The
weights, launch context, stream, staging and workspace remain shared. The scalar
runner methods use slot zero. `SelectSlots` selects an execution closure over the
active slots at a completed unit boundary; the runner's aggregate state and swap
closure include every initialized slot, idle retained conversations too, but
not a spilled one (a swap in does not restore it; its next turn does).
Every slot's plans and graphs are charged to one account. A slot outside the
selected set spills (`Slot::Spill`: its state out of every closure, written to
its spill file, its backing released) and restores (`Slot::Restore`) before
its next work; Clear discards a spilled slot's saved state too. Setup plans
the widest waves its slots can run (every slot at the context's end with
its most rows, target and draft) over placeless addresses, and the shared
workspace, pool, staging and host inputs are the larger of a chunk's
needs and those waves', with a quarter's margin: a slot adds its few-row
shapes, not another prefill chunk's (6.80 → 1.53 GiB of workspace at four
slots; [request slots](experiments/request-slots/README.md)).

`qwen38_wave_plan.h` composes fresh per-slot plans without rewriting scalar
caches. Consecutive compatible slots form a group while their rows fit
sixteen (more slots form more groups), whose
target/draft MXFP8 and routed vector products join at up to sixteen rows,
charging concatenation and retaining independent output views; the wide
MXFP8 and expert-major routed kernels give each request's products bit for
bit. Compatible two-to-four-row HC BF16 products of a target group share
their original cuBLAS path, retaining separate preparation and nonlinear
mixing. Eligible three- or four-row BF16 target heads join through the
ordinary MMF selector (up to sixteen columns); unsupported heads stay
original. Selected BF16 MTP heads (K 2,560, N 16,384–65,536) join up to
eight single columns through the original MMVF vector arithmetic, with
immutable backing and per-slot views. Other draft products keep their
selectors; stateful operations remain per request. See the
[head-sharing controls](experiments/qwen38-draft-head-waves/README.md). A
multi-slot wave reads attention cells aligned to 2,048, and backs each
slot's caches through them, so its graph key repeats. Each decode step's
own state reservation already backs through that alignment, so running out
of state capacity refuses that request rather than failing the shared wave
(which would stop the node). `TargetWave` and `DraftWave` validate placement
and bindings before dispatch, then publish outputs only after the completed job.
Each successful verify retains its own snapshot until acceptance or explicit
discard restores that branch's prior target and MTP state. Fast Qwen verifies
with 48 value heads, dimension 128 and up to sixteen rows fuse the alpha/beta
pointwise chains into two F32 planes. Both Linear products, recurrence and
saved-row consumers remain original; prefill and exact plans keep the
primitives ([controls](experiments/qwen38-gdn-gates/README.md)).

Clearing or restoring one destination first renews the request's protection of
its peers, then discards only the destination's eligible backing and renews the
selected closure. A separately held destination refuses discard. A proven local
failure while discarding the destination invalidates that slot; clean validation
refusals preserve the existing state. An unproven device or shared-execution failure
stops the cohort and preserves borrowed owners until retirement is established.
The serving adapter forwards each host branch to its corresponding native slot.
The production Qwen and DeepSeek chat backends drive prompt and generation units
through the shared driver seams; literal completions retain scalar entry points.
A native fence proves retirement before any borrowed owner is released.

DeepSeek V4 with DSpark runs on the same shared runner (`model_runner.h`):
its slots, cohort rules (`request_cohort.h`), wave composer and decoding
modes are the shared ones, each slot with its own target state, DSpark ring,
verify snapshot, output staging and plans. A wave holds at most 16 rows. The fast plan's
row-local operations (quantized products, routed experts, HC mixes, routing,
norms, the head) run once over every slot's rows through the planner's
owner-row products (`components/owner_products.h`), while each slot's
compressors, indexer, attention, cache writes and DSpark injection run on its
own state. The joined kernels keep each token's sums equal to a one-token
step's (`Dsv4FastTest.WaveLaunchesKeepEachTokensSumsBitForBit`); whole-model
equality of a joined wave with its requests' solo steps is not asserted by a
current test. The retired runner's wave construction and its evidence are in the
[DeepSeek batching report](experiments/deepseek-batching/README.md).

## The shared slot lifecycle

M3.6's replacement for the lifecycle each family wrote (D-107;
[engine-components.md](engine-components.md#execution)), in the core and
tested on the fake backend. The family runners do not use it yet: each
family's migration moves to it with the rest of its runner.

**State contracts** (`model/state_contract.h`). Each state tensor declares
its kind, encoding (`format::Encoding`), cell bytes, cell rule (a cell per
position; a window ring; a cell per block of positions, over the positions
or over the cells a per-position tensor reads, with an optional dummy row,
written on completion or by every row; fixed cells; a chunk's rows; a
position ring of fixed cells), its read alignment and its rollback
(truncate, snapshot, commit or none), whether the target's steps write it
in a drafter's stage (a feature tap), or that it reads the target's tensor
(borrowed). `Layout` derives a stage's region; from it come the ranges a
state of n positions reads (`UsedThrough`), a chunk's writes (flat, or by
verify row with the scratch every rejection restores: `ChunkWrites`,
`WritesFor`, below), the ranges later positions may overwrite
(`CheckpointWrites`), its D-068 representations and `KeptLayoutId`
(`llmp-state-v2:` and the SHA-256 of the spec's digest and every stage's
layout). Contracts reproduce Gemma's closed-form layouts, used state and chunk
writes (the M3 Gemma 2 and Gemma 3 layouts', deleted in G1d), Qwen3.8's (and its MTP region's)
layout, used state and checkpoint writes, DeepSeek V4's in both window
modes and DSpark's ring byte for byte, and the saves of Qwen3.8's,
DeepSeek V4's and DSpark's verifies (`state_contract_test`). The IDs are new: kept state
written under a runner's own ID does not match and is refused, not
migrated.

**Slots** (`SlotLifecycle`). A slot holds one LiveState per stage (target,
then each drafter), each with its own spill file and SpillTrack; at its
interface a range's `region` names the stage. The closures come from the
request cohort over every stage's extents. While one stage grows beside
another's fresh extents, the execution closure and the slot's own extents
are protected together. A runner calls `RecordWrite` before posting any job
that writes a slot (the first position, or a copy's ranges): an incremental
spill writes back only what a record covers. A record, like `SetPositions`,
is refused for a spilled or adopted slot (no job may write it), and
`CheckActive` refuses a slot between a checkpoint restore's Prepare and
Complete (only the restore's own copies and completion run then). While a
verify in any stage awaits its Accept or owes a restore or commit, no
stage's file is marked current (a stage's commit is the model's code and
may write another stage's state), and `LiveState::Save` refuses a range
outside its own state's regions.

**The reconciliation.** Where the five LLM runners' copies differed, one
behaviour was chosen:

| Verb | What the families did | What `SlotLifecycle` does, and why |
| --- | --- | --- |
| SelectSlots | Gemma2 refused dropping an active slot with a pending restore; Gemma4 refused dropping a borrowed or verify-pending slot, active or not; Gemma3 checked neither; Qwen/DeepSeek checked slot counts. All kept an unchanged selection's closure inside an open request | Refuses dropping any slot with a pending restore, a borrow or a verify awaiting its Accept, active or not (each needs its state protected); slot-count checks and the unchanged-selection shortcut kept |
| Clear | Gemma reset positions and the refusal only on success; Qwen also reset speculation; Qwen/DeepSeek always marked the spill file stale; Gemma4 guarded verify and borrow and invalidated features | The union: verify and borrow refused; peers protected while the slot's own lease is dropped; every stage zeroed for reuse, and a failed zeroing quarantines the whole slot (an earlier stage may already be zeroed); spill files always stale; positions, restore, adoption and refusal reset only once every stage cleared; the runner's derived data through the invalidation hook |
| ClearIdle | Only DeepSeek reset the refusal unconditionally and refused an unprovisioned slot | DeepSeek's, with Gemma4's verify and borrow guards |
| Spill | Gemma2/3 refused while restoring; Qwen/DeepSeek refused an unaccepted verify and spilled incrementally; Gemma wrote every extent | Refused while restoring, borrowed, unusable or awaiting an Accept; incremental for every stage through SpillTrack; a stage with nothing used counts as saved; owed restores stay owed, and a slot owing one in any stage is written whole by the next spill (what runs it, a step or a settle of its own, may record no write, and may write any stage) |
| Restore | Only Gemma2 refused a disk restore during a checkpoint restore and refreshed the closures after a failed adopted load; Qwen/DeepSeek returned before checking the cohort's fault; Gemma passed the execution closure to an adopted load, Qwen/DeepSeek none | The fault checked first; a pending restore refused; adopted stages loaded beside the execution closure and the slot's other stages, closures refreshed after a refusal; a clean capacity refusal (the budget's, state usable) is retryable; only the adopted path marks its files current |
| PrepareRestore / CompleteRestore / ValidateFootprint | Gemma only: layout and footprint checks, a pending restore with per-range progress, a clean capacity refusal separate from quarantine. Qwen/DeepSeek prepared without them and quarantined on any failure | Gemma's three steps for every family, refused while a verify owes a restore or commit (it would run over the checkpoint; the runtime settles first). Every stage grows to the footprint before any drops its newer tail, so a capacity refusal leaves the state, and what the runner derived from it, as it was. The pages a completion waits for are the checkpoint's (below) |
| CheckpointRanges | Gemma copied every used byte; Qwen/DeepSeek copied the used pages later positions may write | Qwen/DeepSeek's (`CheckpointPages` over `CheckpointWrites`): earlier pages never change, so they stay in the live state or its spill file |
| CopyState | Gemma2/3 enforced the direction against a pending restore and tracked progress; Gemma4 did neither; Qwen/DeepSeek quarantined on any failed copy; an unproven copy faulted the cohort except DeepSeek's single-slot runner | Gemma2's direction and progress; refused while a verify awaits its Accept (Gemma4's) or owes a restore or commit (out, the bytes are not the kept rows' state; in, a settle that follows would overwrite the pages copied); a failed copy to the device quarantines the slot (one to the host changed nothing); an unproven completion faults the cohort always; copies in are recorded as writes |
| Adopt | Gemma checked the layout and footprint in the runner; Qwen/DeepSeek left the identity to serving and counted whole extents | Layout ID and footprint checked here, into an empty, idle, usable, unborrowed slot with no restore pending or kept zeroed backing; whole extents; files current once Restore read them |
| CheckPlaces | The same pattern each time, over each family's drafter or assistant weights too | One pass over every stage's weights and every slot's stages |
| ReserveStateThrough / EnsureState | Gemma2/3 refused while restoring, Gemma4 not; Qwen kept the refusal flag; DeepSeek checked usability, Qwen not; Qwen/DeepSeek quarantined on a failed refresh | Refused while restoring, borrowed or awaiting an Accept; the refusal always reset; every stage usable; a failed refresh quarantines the slot |
| StateWrittenBack / kept_whole | Gemma: on disk if whole and usable, cleared by every write; Qwen/DeepSeek: SpillTrack, kept whole while on disk | SpillTrack, saved only if whole, usable, not mid-restore and no verify in any stage awaiting its Accept or owing a restore or commit; kept whole only with nothing recorded since (Gemma's) |
| RecoverInPlace / SplitForSpill | Qwen/DeepSeek only (Gemma fell back to dropping plans) | Qwen/DeepSeek's: an untouched spilled slot is kept (every stage preserved, or the slot goes whole), and so is one holding nothing; every other slot the failure may have touched is discarded and reported, and so is one whose verify awaits its Accept or whose checkpoint restore is pending (the failure may have skipped them; kept outside the default selection, it would refuse every selection dropping it); a failed discard faults the cohort; every borrow is dropped. The caller proves the launch retired first |
| Kept layout | Gemma's fixed strings checked in the runner; Qwen/DeepSeek's region strings checked by serving | `KeptLayoutId` from the spec and every stage's layout, checked here |

Positions are Gemma's ledger (set after a completed step, by a completed
restore, an adoption or a verify's Accept): the slot's committed
positions, the target's, by which every stage's state is indexed. Each
stage has its own context (its layout's) and read alignment
(`StageState::read_align`, else the options'). A step on the target backs
every stage (`ReserveStateThrough(slot, positions, align, stage)`: the
target at the step's alignment, each drafter at its own through its
context, since a target's step writes the drafters' feature taps); a
drafter's step backs its own stage alone. A slot's verify-pending state
is its stages' own (`LiveState::verify_rows`); `Accept(slot, keep)` takes
every verifying stage's Accept together (refused, nothing changed, unless
all can) and advances the positions, `DiscardVerify` undoes the verify
whole in every stage, now, and `Settle` runs what an accepted verify owes
now (a failure there checks the job and faults the cohort if a stage was
quarantined), so callers never reach into a stage's LiveState for them.
A borrow freezes one stage (`Borrow(slot, stage)`, a move-only
`StateBorrow` returned once or by its destruction): that stage cannot
grow, the slot cannot clear, spill, restore, be copied into or be dropped
by a selection, while its other stages (the reader's own) may grow; a
borrow is refused until a verify is accepted and what it owes has run
(Gemma4's settled at its Accept), and one taken before a recovery returns
nothing (`RecoverInPlace` drops every borrow and moves the epoch on).
`PrepareThrough` and `FinishPreparations` give every family's
prepare-ahead (`LiveState::Prepare` on each stage's fresh zero extents
for a known next chunk, drained after the current job) through the slots.

## The model runner

M3.6's one runner (D-107; [engine-components.md](engine-components.md#execution)),
in the core and tested on the fake backend (`model_runner_test`, over
`tests/support/fake_step_source.h`; `decoding_test` for the decoding
modes) and, over the component step source and GGML's lowering, on the
Gemma artifacts. No family serves on it yet: each family's migration
moves onto it with its components and is served through
[the serving adapter](#the-serving-adapter).

**Step programs** (`kernels/common/step_program.h`, vendor-free). A
kernel source's lowering makes a `StepProgram` from a planned step and
the `Addresses` it binds (weights by role and part, state by `StateRef`,
the activations; activations at 0 is a placeless measurement): what it
needs (scratch, pool, host bytes, launches), its input slots (each a
segment's tokens or positions, caller bytes per segment or step, or what
the runner computes from the shape by components/step.h's rules: the
head's rows, a segment's store cells, a host-built mask), its
output slots (a segment's own, or the step's joined), its tensor uses by
role for the coverage check, `Bind` (once per plan, on the driver after the
current job), `Run` (the one virtual call per step) and `Record`. A program
whose launch parameters take the step's positions (DeepSeek's HCA prefill
plan) says it is not `capturable`: it is never recorded, and `Position`
gives it each step's segment positions before its run.
`StepLaunch` is the backend's launch: `Capture` records what a body queues
and `Launch` replays it; `RecordingLaunch` does both over the device
runtime's recorded work (the fake's in tests, any backend that records a
stream), and `GgmlStepLaunch` (`kernels/ggml/step_program.h`) over GGML's
`LaunchContext`, whose captures are its CUDA graphs.

**The component step source** (`component_source.h`). A `StepSource` for a
model made of components: per shape, the stage's primitive step
(`BuildStep`), planned (`planner::Plan`, every input staged but the cells'
positions, which only masks read) over a kernel source's `StepBackend`
(`kernels/common/lower.h`: registry, device facts, the implementations
kept on the stream, what its implementations write beyond a tensor's
bytes, and its lowering), then lowered into a program, under the
architecture mapping's default policy for the source's values
(`spec::DefaultPolicy` over `spec::Mapping::policies`: the family's recipe
as data, a rule per gated variant) unless it is given another; its key base is
the spec's bound digest with the policy's patterns and digest. GGML's
backend (`kernels/ggml/step_program.h` `GgmlBackend`) gives the registry
with its capabilities, the device's architecture, the lowering of
[engine-components.md](engine-components.md#the-graph-ir) and the step
launch the runner makes at Bind. Over it the runner runs Gemma 2 and Gemma
3 steps of two owners under the full serving recipe (`component_runner_gpu_test`:
owner decode and prefill, bounded roots, the narrowed final layer, ring wrap,
logits and greedy heads, finite and exact from one runner to the next; they
were bit for bit the M3 runners', deleted after the gate); serving runs Gemma 2 and Gemma 3 on it
([the serving adapter](#the-serving-adapter)). A model of several stages
(a target and its drafters) has one `StagedStepSource`: each stage's
component source, the one a shape's stage names, and one key base for all
(the spec bound to every stage's artifacts, digests over every stage's
patterns and policy). Over it the runner runs Gemma 4 26B-A4B's and 31B's
recipes and their assistants' draft passes bit for bit the M3 runner's
(they were; the M3 runner is deleted, and `component_runner_gemma4_gpu_test`
now checks the outputs finite and identical from one run to the next,
the verify and draft cases included: the target's steps of the composed
spec write the assistant's feature tap, which its first pass reads); serving
runs Gemma 4 and its assistant on it (`Gemma4AssistantServingGpu`).

**Planner types** (`planner/step_shape.h`, header-only; the planner,
`planner/plan.h`, plans a step's graph and makes its key,
[engine-components.md](engine-components.md#the-graph-ir)). `StepShape`
{stage, `Phase` (prefill, decode, verify, draft, score), segments by
ascending slot, row invariance, read alignment};
`Segment` {slot, rows, past, read cells, outputs, group, want_head,
greedy}; `PlanKey` {spec digest, shape, pattern and policy digests,
partition}. A segment's `past` is data, not the plan: equality ignores it,
as every family's shape equality ignored `n_past`; `read_cells`, which
follows it at the shape's alignment, is part of the plan. A step source's
program therefore bakes nothing that follows `past`: positions reach it
as input data, or through `Position` for a program that is never
captured.

**The runner** (`model_runner.h`). `ModelRunner final : PagedModel` holds
a `SlotLifecycle`, the stages' `PagedWeights`, `RunnerMemory` (input and
output staging), `StepRuns`, the launch, a `PlanAccount` and one plan
cache, over a `StepSource` (`KeyBase` once at Bind; `Build`, host-only, per
plan miss). Setup lays out the slots' state first, reserves the weights,
builds every configured setup shape placeless to size the activations,
pool, staging and plan floor (a quarter's margin) and pins the staging;
Register registers sources and pins places; Bind makes the launch, binds
the account and builds the closures; Release drops plans and recordings,
then the launch, then the memory they name. A step (`RunStep`):
1. `PlanStep` (a segment wider than the chunk bound of a stage whose
   per-row cells it writes, its own or on the target a drafter's tap,
   refused); each owner admitted (`Admit`: active; on the target, at
   its committed positions whether or not the step advances them (a
   verify's rows start at its anchor); on a drafter, never advancing, its
   rows ending within its stage's chunk bound past the committed
   positions, and a drafter reading the target's state only while the
   target's stage is borrowed; state backed through every cell it reads,
   `past + rows` or a common read's wider `read_cells`, at the step's
   alignment; a refusal names its slot and leaves its peers), a joined
   step's owners first through what each reads alone (its rows at its
   stage's alignment, as a lone step's segment), so no owner's widening for
   the join takes what a later owner's own rows need and an owner refused
   only in the widening (`StepRefusal::widening`) steps alone with no
   growth; `CheckPlaces`. A widening refused for a later owner keeps the
   earlier owners' (clean growth, leased, read once their positions reach
   it): at most the join's extra reads per owner, which a peer's own growth
   may then lack until the runtime reclaims for it; an all-or-nothing
   widening (one acquisition for every owner's) is owed;
2. the plan found, or built, bound, coverage-checked (`coverage.h`, refused
   on a violation) and cached with its host bytes and planning time; a
   build binds only the state its stage may write (a target's step the
   target's and the drafters' `target_writes` tensors, a drafter's its own
   stage's and its view of the target's), so a verify's saves and a failed
   step's quarantine cover every stage its program can touch;
3. inputs staged by slot (`StepRuns::Stage`: a computed input by the
   shape, a tap's store cells for its segment's output rows alone, a
   frozen read's mask over the cells before its rows), output copies laid
   out, a program that is not capturable positioned; a drafter's features
   (the program's `kFeatures` input, never staged) copied on the device
   from where each segment's work names them (`FeatureFrom`: its stage's
   feature tap, `model::FeatureTap`, at positions the target has committed
   and its ring still holds, or the slot's carry), and its next features
   (an output marked `features`) carried to its slots' carries after its
   run, both in its job outside any recording (their sources move with the
   positions); a carry is forgotten when the slot's state changes outside
   its steps;
4. the futures (`StepAhead`): deduplicated against the shape and each
   other and protected by `Find` before any funding; a recording made
   ahead whose staging differs is dropped; the capture decision (second
   run in the phases `capture_phases` names, or the first run of a shape
   the next one repeats) and `ChargeGraph`; the next shape's recording
   captured beside this run when it differs and is planned; up to two
   missing futures funded (`PrefillLookaheadGroup`) and built in the job's
   meanwhile; the next shape's fresh state prepared;
5. a verify's saves (below), `RecordWrite` from each owner's first row,
   then one job: each owner's owed restore and commit, the verify's saves,
   the run (`StepRuns::Queue`), the capture ahead;
6. preparations drained; on failure `CheckFailedJob`, each owner's states
   settled (a verify's saving stages undone whole; any other stage the
   step may have written, its own and on the target the drafters' taps,
   quarantined if anything was queued; a stage changed only by what it
   owed ran that whole or still owes it) and the cohort faulted on a
   launch fault (a faulted cohort's refusal is never clean); on a failed
   drain every owner quarantined; a verify that ran awaits its Accept in
   every stage it writes (the target, the drafters' taps) or that
   commits, whether or not it saved anything there;
7. futures installed (bound, checked, the grant transferred to the
   cache's charge), the path counted, each owner's outputs copied to its
   own buffer, positions set.

A drafter stage that reads its target's state frozen and stores none of
its rows itself (`model::ReadsFrozen`: borrowed tensors and taps the
target writes, Gemma 4's assistant) reads only the cells before its rows:
its reads cover `past`, not `past + rows`, so a draft anchored at a past
that is a multiple of 256 reads exactly that many cells (the runner's
`CheckShape` and lone reads, the composer's `StageReads::frozen`). A
target's output rows each write a drafter's tap cell: Setup refuses a tap
ring narrower than a verify's rows, and a step refuses more output rows
than the ring holds.

The shared runner also stages bounded on-demand row tables (`RunnerConfig::row_tables`).
Their full artifact groups stay unplaced; each table owns a direct-read ring,
pinned landing and offsets, and pinned device weight slots sized from its
setup shapes. Every side-row input must be supplied in full and pass the
row-table planner before I/O starts. A completed read is gathered on the
GPU before either an eager program or a graph replay. An unretired read
prevents reuse; teardown retains its ring and landing until completion is
proven.

Components declare their verify intermediates through `StepSource::CommitBuffers`.
The runner maps one field buffer with a fixed slice per owner and verify
row. Joined saves stage owner-specific indices; a later owner cannot
replace a peer's pending rows. `Accept` leaves an owed commit. Before the
next job, each owed component commit is planned under the job's plan hold
and stages its computed inputs into a separate pinned slice per owner and
stage. The job runs those commits before new model work. The internal
`kCommit` phase is refused by ordinary steps and lookahead. Caller-provided
commit callbacks remain available for other step sources.

Feature inputs can read position rings or per-row chunk taps. For a chunk
tap, the runner tracks its committed logical span and carries its previous
tail into the leading cells. Verification snapshots those leading cells
with its first row: acceptance retains them, while discarding the entire
verify restores the previous chunk. Ordinary same-process spill and
restore retain this logical metadata. Existing position-ring taps derive
their cells from positions alone.

**Repeatable drafter progress** (G3c2a). A component may declare
`DraftProgressPolicy::kRepeatPending` and the relation between cache,
token and feature positions. With `kTokenAfterCache`, cache row `q`
consumes token `q+1` and feature `q`; with `kTokenAtCache`, it consumes
token `q` and feature `q-1`. The stage owns one fixed 64-byte I64 control
tensor containing its schema, committed positions, first pending cache
row, chunk-tap span, preceding rows, relation and reserved zero. This is
ordinary cataloged state, distinct from numeric KV encoding support.
Layout validation binds the policy to the real tap, carry and context
bounds and requires replayable private position/block caches; unsupported
borrowed or recurrent private state refuses. Its contract contributes to
the existing kept-layout digest. Version 1 component contracts coexist
with the explicitly opted-in version 2 `embedding.mtp` and `attention.qsa`.
Current architecture mappings keep version 1.

Multi-owner submissions require immutable numerical qualification from
**every** stage's step source when any stage declares repeat-pending
progress. Native component sources are currently unqualified: decoding,
prefill, prompt and raw-wave entry points refuse the entire multi-owner
submission before settling owed work, opening units or admitting state.
Direct steps and their supplied lookahead shapes have the same boundary;
a scalar prompt's lookahead is checked before its pre companion. The fake
backend explicitly qualifies its independent-owner arithmetic for lifecycle
proofs. Scalar native MTP is supported by the execution proof below.
Joined native MTP remains a required lowering slice before serving and
migration gates; failed matched 256/2048 joined recipes are not enabled.

A completed head-bearing catch-up leaves the durable pending interval
unchanged. Whole discard therefore repeats its original K-row schedule,
even after a changed anchor or restored checkpoint. Acceptance first
retires numerical restore/commit work, then copies each stage's candidate
control from its separate pinned buffer; only proven completion publishes
all host mirrors. Terminal completions carry owner/transition identity.
A late or unknown copy failure quarantines the owner and retains staging.
Restore and new-runner adoption exclude use until actual device-to-host
control reads complete and every stage's record validates. Clear resets
the control only after old work retires. Checkpoints include the bounded
private-cache tail starting at `max(0, positions - max_rows)`, rounded to
blocks where needed, plus fixed state, taps and dummy cells. Every opted-in
draft write requires an internal owner/unit permit for its exact cache,
token and feature coordinates before admission; arbitrary historical
writes cannot escape that checkpoint envelope.

The shared prompt helper runs headless companions before and after target
prefill, using authoritative full token history. Compound ownership starts
before the first companion and retains the old pending record until the
whole sequence and control publication complete. A proven pre-only
completion followed by a clean target refusal retries the original K;
leading-carry or target mutation with incomplete work quarantines. An
unchosen repeat-pending consumer also consumes its old features before the
target replaces them. Stages that use the original cursor policy keep their
existing orchestration. Without the durable policy, checkpoint replacement
and kept adoption still clear a chunk tap's transient span; lookup refuses
until a target step publishes it again.

Components may derive side-row IDs from authenticated full history through
immutable source metadata. Staging uses the declared token coordinate,
including a repeat-pending drafter's offset from its cache row. Explicit
row inputs remain authoritative. Missing, malformed or partial inputs
refuse before state admission for resident and on-demand tables alike;
duplicate input slots for one table refuse before paging, while a joined
input covers multiple owners.

G3c1 checks Qwen's plain execution against the legacy runner in all three
weight forms, native target verification against the legacy verifier,
and generated commits against the legacy commit kernel using the actual
saved verify inputs. The fast verifier is not row-invariant: its saved
recurrence, not a shorter plain step, defines exact committed state.
G3c2a's fake checks cover publication and hydration failure, independent
consumers, partial blocks and omitted checkpoint pages, and stale kept
records whose control stayed unchanged while numeric bytes changed. Native
MTP GPU checks compare exact scalar head logits, target logits, pending
features and private caches across accepted prefixes, whole discard,
changed anchors, the refused-prompt retry, checkpoint replacement and
fresh-node adoption, with changed-input replay of the same multirow MTP
graph. Runtime wiring and the unchanged quality/state/swap
and matched 1% speed gates remain G3c2's; serving defaults are unchanged.

`PlanCache` is now `BasicPlanCache<Key, Planned, Variants, Runs>` in the
core (`plan_cache.h`); `planned.h`'s `PlanCache` is the alias over GGML's
`PlanRuns`, and `StepPlanRuns` is the step programs'. `RunPath`,
`GraphStats` and the run copies moved to `run_stats.h`.

**The wave composer** (`wave_composer.h`). Pending work per slot becomes
step shapes by a `model::Batching` capability (`model/batching.h`, data
the components will supply): work of one kind (stage, phase, row
invariance, device choice, head intent unless mixing is allowed) joins in
slot order up to `max_segments` owners and `max_step_rows` rows, prompt
rows only with `join_prefill` (and equal rows and reads with
`equal_prefill`); within a step, consecutive owners form groups while
their rows fit `group_rows` and they number at most `group_segments`
(`invariant_group_segments` for a row-invariant step); a joined step reads
at `joined_read_align`, and with `common_read` every owner reads the
widest. `RunWave` runs the composition (refused whole when the composer's
alignment is not the state's): an owner's refusal (not active, at other
positions, out of state capacity) settles it alone and the rest are
composed again (an owner out of capacity only for the joined step's wider
reads, at a joined alignment or through a common read, first steps alone
at its own, after its kind's joined step); a joined step refused cleanly
(nothing queued, no owner's state changed beyond clean growth, the cohort
healthy) runs its owners alone; a step that may have written fails its
owners together, and none of them runs it again. Each outcome says where
its owner sat in its step (segment, the rows and outputs before it) to
find its share of the joined outputs. Each stage composes at its own
context and alignment (`WaveComposer::For` takes them from the slots). A
row-invariant step (an exact verify, D-092) reads what each owner reads
alone (its stage's alignment, no common read), so no row's arithmetic
follows a peer's reads; the runner refuses one whose reads follow its
peers.

**The prefill orchestrator** (`prefill.h`). One completed unit at a time:
each prompt's next chunk (the configured rows, bounded by the capability's
per-owner rows, the context and a checkpoint boundary; no time cap), cut
state-only unless it is the prompt's last, joined by the composer, with
its futures (the same owners' next and following chunks, composed;
an owner whose prompt ends drops out) built ahead, captured ahead or
captured first, and its fresh state prepared. Each outcome keeps its
step's joined outputs and its place in the step (segment, rows and outputs
before it).

**Decoding modes** (`decoding.h`, D-068). `DecodingOrchestrator` runs one
unit of generation for a set of slots on the runner:

- plain, sampled and scored modes are one step: a decode row a slot (its
  token chosen on the device when greedy), or a scored run of given
  tokens with every row's outputs;
- speculation drafts with any number of drafters, each a stage with its
  own state (`DrafterSpec`): an autoregressive drafter proposes a token a
  pass, a row a pass but its first; a block drafter proposes a block from
  its anchor and a fill token in one pass. As a **cascade**, drafter i
  extends drafter i − 1's proposal and one verify checks them all; as
  **alternatives**, each slot's unit takes one, named by the alias policy
  (`DecodingPolicy::choose`) or chosen by what each has committed a unit
  for its cost (`DrafterChoice`: four units each, in order, then the best
  past a 3% margin). Every draft pass and the verify run through
  `RunWave`, so one slot's refusal never fails its peers; a drafter
  refused for a slot leaves that slot's unit the proposals it has (none:
  a verify of the anchor alone);
- the verify (phase `kVerify`, on the target, never advancing; its plan
  row-invariant in the exact mode) saves, in its job before its run, what
  it writes in each stage (`model::WritesFor`) into that stage's snapshot
  (`RunnerConfig::max_verify_rows` sizes every slot's at Setup, by
  `model::VerifySaveBound`, in one mapping; `RunnerConfig::commit` gives a
  stage its commit). The caller's judge (`DecodingPolicy::judge`: greedy
  agreement, speculative sampling) says how many rows stay; the slot
  accepts them, or the verify is discarded whole when the judge refuses.
  A draft's own writes are never undone: what a drafter's step writes past
  the committed positions must lie outside every cell later steps read
  (DSpark's ring holds its window beside a block), and a verify's
  snapshot of a tap the drafter writes too restores the draft's rows,
  not what the cells held before the draft;
- an autoregressive drafter with `catch_up` (its state for committed
  positions is rewritten from the target's features: MTP) starts each
  unit's first pass at its cursor, which a verify moves to its first row,
  so the pass catches up the kept rows and the anchor (`Reset` puts the
  cursor at the positions after a prefill, a clear or a restore); one
  further behind than its chunk bound (an alternative not chosen for a
  while) catches up its last chunk's rows, its older rows left stale;
- a drafter with `borrows_target` borrows the target's stage for its
  passes, after settling what the last verify owes; a stage whose
  contract views the target's tensor (`borrowed_from_target`, resolved by
  `model::BorrowedItem` to the owner's target state) steps only so;
- an `anchored` drafter (Gemma 4's assistant) runs every pass at the
  committed positions, its one row the latest proposal's token (the
  anchor's first); one with `features` reads beside its tokens, on a
  unit's first pass, its stage's tap of the positions before its rows (the
  target's features there, which its steps with outputs write) and on a
  later pass its own last pass's next features (the runner's carry); none
  drafts at position 0;
- depth adapts per slot and drafter (`execution::AdaptiveDepth`), taught
  only by complete steps (not cut by the reply's end or the context, and
  in a cascade only once every earlier proposal was accepted); the wave's
  form (`DecodingPolicy::speculate`, `execution::AdaptiveWaveMode`'s
  choice) may make a unit one plain decode row a slot, which still feeds
  the drafters' taps;
- block diffusion is a declared kind: a mode of it is refused.

What bytes mean (a drafter's proposals, the verdict) is the caller's
policy; the tests' is a fake truth.

**A verify's writes** (`model::WritesFor`), by each tensor's rollback: a
`kTruncate` tensor saves nothing (rows past the kept prefix are never
read; refused for fixed state and partial blocks, which a verify writes
for the kept rows too, and valid for a position ring or per-row cells only
if the model never reads a rejected row's cell, which the contract cannot
check); a `kSnapshot` one saves its cells by verify row (a cell per
position, a ring's or position ring's cell, a per-row stream's row after
its carried rows, the block a row completes, or with `partial_blocks` the
block by the chunk's last row in it), its dummy row when the chunk writes
it (fewer blocks completed than ceil(rows / ratio)), and fixed state
whole; restored, fixed state and partial blocks hold none of the kept
rows' part, so the commit rebuilds it (refused without one); a `kCommit` tensor is
not written by a verify and needs the commit. Over a drafter's stage the
target's verify saves only the tensors it writes there (`target_writes`:
a feature tap, an injection). A verify writing one ring cell twice, or
saving cells that are not 16-byte multiples (the range copies'), is
refused. Per-row feature taps additionally save their leading carry cells
with the first accepted row. The remaining ranges reproduce byte for byte
what today's verifies save
(`state_contract_test`): Qwen3.8's (`Qwen38VerifySaves`, which its scalar
and wave verifies now both call: each row's K, V and indexer cells, the
block keys a row completes, the MTP streams rows), DeepSeek V4's
(`Dsv4ChunkWrites`: each row's window cell, the compressor ring rows it
persists, a position ring per compressor, the compressed rows it
completes, the dummy rows as scratch, no commit) and DSpark's injection
(`DsparkWrites`), compared as each row's set of ranges (the order only
places them in the snapshot).

What each family's speculation maps onto, and what its migration must
still bring:

| Family | Maps onto | Not yet |
| --- | --- | --- |
| Qwen3.8 MTP | An autoregressive drafter with `catch_up` over its own stage (K, V and indexer cells truncating; the streams a per-row tap the target writes, snapshot); the target's K, V, indexer and completed block keys snapshot by row, its linear layers' state `kCommit` with generated component commits; bounded on-demand n-gram rows and chunk-feature inputs | Runtime wiring and gates; the catch-up and passes in one job, the selected draft head's rows; the opt-in shared repeat-pending policy supplies durable chunk spans and prompt companions |
| DeepSeek V4 + DSpark | A block drafter (its fill the mask token) over its ring (a position ring the target's injection writes); the target's window cells, compressor rings and compressed rows snapshot by row, no commit; the exact mode a row-invariant verify | The draft and the verify chained in one job (drafts copied into the verify's staged tokens and embedded on the device), the Markov head, shortening a draft before a mask width (`Dsv4SameWidths`: policy) |
| Gemma4 assistant | An anchored autoregressive drafter with `borrows_target` and `features`, its contract viewing the target's KV (`borrowed_from_target`) and its tap of the target's final-norm rows; the target's KV truncates (a ring holds a chunk's rows beside its window, so a rejected row is never read) | The device's verify verdicts (`GreedyUnit`'s device judge); its adaptive depth (three drafts a unit, M3's screen's) |

What the shared pieces reproduce of each family, and what a migration
must still bring (as components' capabilities, planner patterns or
policy):

| Family | Reproduced | Not yet |
| --- | --- | --- |
| Gemma2/3 | Joined prompt waves with equal rows and reads (`equal_prefill`); two futures, first-run capture of a repeating shape, capture ahead, prepare ahead; head-or-state-only shapes; second-run capture in every phase; independent outputs; owner attention (owner decode, owner and flexible owner prefill, bounded roots, padding to a common width, Gemma3's four-root carriers: the planner's `owner_attention` from the step's shape) | The 2,048/256 head-capacity envelope (`max_head_rows`); the cohort's 2- or 12-owner limits become `max_segments`; the C12 bounded factor and the copying packed prefill are not ported |
| Gemma4 | Quads as `group_segments` 4, the 8/12 groups as `invariant_group_segments`/`group_segments`; one future without capture ahead is `lookahead` 1, `capture_ahead` false; per-group isolation (RunGemmaGroups) is RunWave's; the assistant's frozen borrows, tapped features and carry (decoding modes); prompts stepping alone and, outside the envelope, one owner a step (its mapping's rules, `prompts_alone`, `owners_alone`); bounded owner roots and common widths (the planner's owner-attention options) | Joined prompt admission (it would change the routed products' arithmetic; M3 prefilled alone) |
| Qwen3.8 | Groups of consecutive owners while rows fit 16 (`group_rows`), 2,048-aligned joined reads (`joined_read_align`), state backed through that alignment before the step (an owner out of capacity there steps alone at 256 rather than being refused), two futures and prepare ahead, state-only cuts; joined draft and verify waves (decoding modes) | Per-product joined limits (draft head 8, VecQ, MoE GEMV, MXFP8 columns, HC cuBLAS 16: planner patterns), BF16 head sharing by row count, the n-gram row gather between inputs and plan, the MTP lookahead group, rows ≤ 4 per slot; its prefill captures only one-row shapes (`capture_phases` without prefill) |
| DeepSeek V4 | Total wave rows ≤ 16 (`max_step_rows`), joined row-local work with per-owner state, scalar prefill (`join_prefill` false), two futures and prepare ahead, state-only cuts; joined drafts and verifies (decoding modes) | Float products over groups of whole owners fitting 8 columns and `SetVecQOneToken` for one-row waves (planner, from the shape), `Dsv4WaveSupport`'s fallback to one slot, the HCA plan's per-position patch (the lowering's `Position`; the runner never captures it), a draft and its verify in one job |

## The serving adapter

One adapter per pipeline kind (D-107; [runtime-serving.md](runtime-serving.md#engine-backed-models))
replaces the M3 serving classes: `runtime/engine_llm.h` (`EngineLlm final :
Llm`) and `runtime/engine_image.h` (`EngineImage final : Image`). Both
are in the runtime's `llmp_runtime_llm` library with `Llm`'s conversation
code (`runtime/llm.cc`, moved out of `serving.cc`), every build profile:
they reach the engine through the core and name no kernel library, so
`engine_llm_test` drives a whole `Llm` session over the fake backend.

An adapter is made with its model's settings (D-103) and an `EngineBuild`
that turns them into its parts (`EngineLlmParts`): the `RunnerConfig`
(every stage's layout, from the settings' context and chunk; the bound
spec's digest; the stages' weights, setup shapes, launch, verify rows and
commits), the step source, the components' `model::Batching`, the
`DecodingMode` with its drafters, the head's logit width, whether a greedy
step returns its tokens chosen on the device, and the prefill futures'
options, and what the source and launch borrow (`keep`: the spec, the
artifact's tensors, the kernel source's backend), released after the
runner. Serving's build (`runtime/component_build.h`, which the gate's
harnesses share) reads the spec its v1 artifact stores (D-118) and, when
the model speculates, composes its drafter's (`spec::ComposeDrafter`: Gemma
4's assistant, a stage of its own), takes each stage's layout (each state
tensor on its own paged extents, as the M3 layouts: packed tensors
straddle extents, so a prompt would grow its state at staggered positions
every chunk or two, each growth a host cost between steps) and the
target's batching from the components, and makes a `ComponentStepSource`
a stage (a `StagedStepSource` over several) planned under each
approval's policy rule for the serving envelope (its context and
owners: `ComponentAdmissionPolicy` over its mapping's `spec::DefaultRule`, whose rule also says how serving composes
the recipe's steps: prompts alone, owners alone, the prompt futures
planned and captured ahead) over a `GgmlBackend` created on the model's
own stream (the runner's job stream, which every launch checks; one
backend per runner, its launch made once at Bind). A layer's routed
experts are one slab (`engine::ExpertSlabsOf`, the M3 runners' pitch:
256 bytes and every array's block bytes), its pitch the lowering's expert
stride. The head's logit width is the spec head's
vocabulary (its logits' width); a greedy step returns each owner's token
chosen on the device. cuBLAS gets the workspace the runner maps for it
(`RunnerConfig::kernel_memory`: mapped and charged at Setup as the
runner's own memory, before any measurement, in every closure; the
handle made by `prepare` over it and lent to the backend's launch, and
retired at Release after the launch, before the memory), as
`RunnerResources` does for the M3 runners. Its setup shapes are a
prompt chunk and a decode row, joined and alone, with logits and with
device tokens, at the end of the context reading all of it, and with a
drafter a verify of its rows and a draft pass, joined and alone. Gemma 2
and Gemma 3 have it and are served by it since their migration, Gemma 4
26B-A4B and 31B and their assistants since theirs (the assistant an
anchored drafter of three drafts a unit reading the target's tapped
features; the M3 class's chat slice kept: thinking off and refused on,
tools refused, the channel and tool-call openers ending a reply);
any other architecture is refused at Setup. The tests'
build is the fake's. Speculation on with parts that have no drafter is
refused at Setup (the M3 classes refused it too), never served plain in
its place. At Setup the adapter makes
the `ModelRunner` (owner and stream its registration index; its slots
named after the model, `max_slots` of them, spilling to the spill role or
the kept places the server sets before Register), a `WaveComposer` over
the slots, a plain `DecodingOrchestrator` and, when the model has drafters
and speculation is on, a speculative one with the adapter's policy, and
prepares a branch per slot.

How `Llm`'s units map:

| `Llm` | The shared runner |
| --- | --- |
| A branch | The request slot of its index (`SelectBranches` → `SelectSlots`) |
| `RunPrefillChunkFor`, `RunPreparedPrefillWave` | Prefill `PendingWork` a unit through `RunWave`, joined by the composer; the last row's head as F32 logits (none for a state-only cut); the sessions' `PrefillHint`s composed as the step's two futures (the next chunks joined where `CompatiblePrefill` would join them; when they step apart, as a prompt's shorter last chunk does, the first two of those steps), under the parts' lookahead, capture-ahead and prepare-ahead |
| `RunChunkFor` | One row after the state: the plain mode's decode row (a scorer's, a generation's); more: prefill steps of the model's chunk |
| `RunGreedyChunkFor`, a greedy plain wave | The plain mode's greedy rows: one I32, the segment's own output (only where the parts say the step returns it) |
| `SpecStepFor`, a speculative wave | The speculative mode's unit: drafts from each drafter's rows (their argmax), one verify, the verdict by `Llm::Keep`/`Choose` (greedy agreement, or speculative sampling seeded by position); a refused or failed unit keeps its prefix when the state is usable (the verify was undone whole) |
| `PrepareDecodeStateFor`, `PreparePrefillStateFor` | `ReserveStateThrough` through a verify's rows or the chunk, so a capacity refusal comes before dispatch (`StateRefusedFor`) |
| `SettleFor`, clear, idle clear, spill, restore, copies | The slot lifecycle's verbs |
| Checkpoint ranges, restores, adoption | `CheckpointRanges` at the committed positions; `ValidateFootprint`, `PrepareRestore` against `KeptLayoutId` (the restored state's: a turn checkpoint keeps the layout it was saved or adopted under), `CompleteRestore`; `Adopt` with the kept record's layout ID, which the lifecycle holds against its own; the cursor is the committed positions |
| `CheckPlaces`, `RecoverInPlace`, reclaim, graphs | The lifecycle's place check and recovery (records of discarded slots removed first, their conversations forgotten); the runner's plans |
| Kept state (D-105) | Record version 2 (D-112): a spill file a stage, `KeptRegions`/`KeptLayouts` a stage each |

After anything that changes a slot's state outside a unit (a prefill, a
clear, a restore, an adoption, a recovery) the speculative mode's drafter
cursors are reset to the slot's positions before its next unit. What the
adapter does not carry yet, and each family's migration brings when it
needs it: the drafters' adaptive depths in kept records and turn
checkpoints (the record keeps `Llm`'s one depth; each drafter starts again
at its own), a drafter's pass inside the target's prefill (Qwen3.8's
injection: a catch-up drafter's own state for prompt positions is not
computed, since its cursor is reset to the positions after a prefill, which
costs acceptance only), and the adaptive wave form
(`DecodingPolicy::speculate`).


## Adding a model family

*From M3.6 (D-107, D-109) a family is added as an architecture mapping for
the importer, plus any component its shape still lacks
([engine-components.md](engine-components.md)); no family gets a runner of
its own, and the guidance below describes the M3/M3.5 runners until they
move. Gemma 2's and Gemma 3's are gone (slice G1d: `engine/gemma{2,3}_
{runner,plan}.*`, `kernels/ggml/gemma{2,3}_graph.*`, the serving classes),
and Gemma 4's (slice G2d: `engine/gemma4_{runner,plan,assistant,
assistant_plan,greedy,wave}.*`, `kernels/ggml/gemma4_graph.*`,
`gemma4_assistant_graph.*`, `gemma4_attention.cc`, the serving class);
their paragraphs below stay as the record of the techniques the shared
engine carries for them, and name files that no longer exist.* Start from the [optimization inventory's catalog](optimization-inventory.md#transfers-and-gaps):
a new runner adopts each technique there whose contract it meets before the
family's speed is compared with its reference. Shared mechanisms (graph
capture policy, held direct steps, state reuse across Clear, the pager's
lazy handoff and handle reserve) come with the skeleton; those still
written per runner retain their family descriptors while shared source,
lifetime and dispatch helpers carry compatible transfers. Plain greedy
publication now shares `Llm::RunPlainGenerationUnit` and
`engine/planned.h` output authentication: a successful optional scalar hook
publishes one token, nullopt retains rows, and joined runners validate all
completed IDs before publishing any owner. DeepSeek and native/GGUF Qwen
non-spec target paths join Gemma on host-compatible device argmax. Shared
row-free eligibility also serves DeepSeek adaptive joined plain waves, which
return `kept` IDs while preserving feature injection and drafter-ring writes;
ordinary `DeviceGreedy` still refuses speculative units. Sampling/scoring and
draft/verify retain their contracts. Shared prompt-completion first-token
publication remains a separate compatible extension for every family.
[Controls and open consumers](experiments/plain-gpu-tokens/README.md).

Host reference construction shares checked prefix/ring intervals in
`model/host_mask.h`; existing Gemma content checks reduce every F16 bit over
constant intervals. DeepSeek shares construction only, and Qwen QSA keeps its
three-valued bias and finite dead sentinel. Dense Qwen already uses prefix
fills. These host helpers do not add validation to device-only paths.
[Exact controls and CPU-only measurements](experiments/host-mask-intervals/README.md).

Causal/ring mask source validation and host staging are shared in
`engine/graph_mask_inputs.h`, used by Gemma2, Gemma3, both Gemma 4 profiles,
DeepSeek target raw masks and Qwen native/GGUF targets with native MTP.
The producer authenticates explicit padded
Gemma rows or exact contiguous DeepSeek/Qwen rows; sparse consumers retain their
real query count.
`GraphMaskSourceBytes` authenticates the complete graph-owned producer, positions,
per-segment row offset, capacity/window/context and unique node/input membership
and the expected F16/F32 output dtype before excluding device-mask bytes
from the host-input grant. Mask activations
remain funded. `StageHostGraphMask` retains the explicit host-reference path and
fills every padded F16 row with negative infinity when padding is required.
DeepSeek omits its raw host matrix in ordinary serving; optional compressed
visible-count inputs and DSpark noncausal block masks keep separate policies.
The [target transfer](experiments/deepseek-device-masks/README.md) preserves
complete heads/state in a joined ring screen and checks production verify/swaps.
Qwen's exact-row adapter omits only mask matrices from host construction and
staging: QSA block metadata remains independent, and existing device sparse
selection keeps its own producer. Headed MTP passes use their own four-section
position input; cache-write-only passes omit unused producers. The
[Qwen transfer](experiments/qwen-device-masks/README.md) checks complete heads,
joined execution and initialized-state spill/restore.
A new compatible adapter uses
these helpers with the existing mask kernel; sparse/select masks need their own
visibility contract. The [Gemma2 transfer](experiments/gemma2-serving/README.md#shared-gpu-masks-2026-10-07)
checks joined 128-row owners, ring wrap, fresh captured positions and restart.

The [Gemma2 runner](gemma2.md) reuses held requests, charged outputs and
capture ownership, retained Clear and initialized-state spill/restore. It
admits only the approved 2B Q8_0 artifact with one or two slots, separate F16 KV and
softcapped attention; checked state descriptors retain alternating
local rings/global caches. Explicit optimized probes select norm/Mul,
quantized GeGLU and width-2304 norm/ADD, with primitive fallback. The bounded
C1 full-head stock control is exact. The opt-in
[C2 decode control](experiments/gemma2-owner2/README.md) selects funded
true-softcap50 owner attention for equal-width one-row segments, with ordinary
fallback. Small/ring strict quality and restore-next controls pass; a short
n=2 cycle retains a 1.49% stock latency gap. Its internal
[checkpoint/adoption foundation](experiments/gemma2-checkpoint/README.md)
checks complete initialized footprints, copy retirement and kept layout IDs,
while protecting held peers and refusing incomplete-restore work. Its 19 focused
controls include exact wrapped-ring fresh-node adoption and peer continuation.
The [bounded serving transfer](experiments/gemma2-serving/README.md) now adds
compatible plain prefill under a separate 256-row wave and common-width decode
padding. Real F16 copies preserve actual state/source bounds and cap50; per-owner
rows 128 and local ring 4,352 remain unchanged. The representative 256/768-prefix
comparison has all 76 stock heads exact, and five HTTP lifecycle/restart cases
pass with actual joined groups. Its short native C2 cycle is 5.78% slower than
stock; broader context/batching, memory/swap and sustained qualification remain open.

The [Gemma 4 foundation](gemma4.md) supplies checked profiles, strict tensor
bindings, bounded independent-slot KV layouts and segmented host inputs.
It also supplies segmented text graphs and a checked plan adapter, with
complete-layer diagnostic controls. Its bounded native 26B-A4B runner reuses
the shared skeleton and has complete-model state/replay controls. Both profiles have a bounded serving adapter. The
[dense31 bridge](experiments/gemma31-serving-bridge/README.md) selects ordinary
joined execution and both norm chains only at context<=8192/slots<=4, with
current-pin quality/corpus and HTTP controls; the
[Gemma26 recipe](experiments/gemma26-production/README.md) adds MoE route/reduce
and a 1024-row prefill under the same bounds. Larger configurations retain their
scalar recipe. Broader quality, context, assistant and sustained
performance qualification remain open.

The [common-read repair](experiments/gemma4-common-width/README.md) restores
both approved profiles' unequal-prefix stock heads. Its subsequent
[bounded-root transfer](experiments/gemma4-bounded-owner-roots/README.md) preserves
exact SET_ROWS-derived actual roots and common logical partitions while removing
temporary K/V padding. Both short and physically wrapped C2 controls match all
68 stock heads and preserve initialized-state restore/replay. Ordinary serving
selects this recipe at context<=4096 with exactly two configured slots; equal
reads retain the previous no-copy path. Same-binary paid latency falls 6.01% /
5.24% (26B / 31B), within 0.1% of current stock in short n2 runner screens.
Wider configurations, partial cohorts and compatible multirow prefill remain
separate qualification work; no sustained or whole-serving parity follows.

Gemma4 [held-cohort selection](experiments/gemma-held-cohort/README.md) keeps
an unchanged selection's execution closure inside a held stream request. Mask
validation and frozen borrowed/verify-peer exclusions run first; changed masks,
growth, Clear and restore still refresh closures. Every submitted step also
checks that its actual execution closure is held.

Gemma's [immutable head capacity](experiments/gemma-head-capacity/README.md)
separates maximum input rows from pinned publication rows. Manual callers keep
`max_head_rows=0` (the existing all-row envelope); serving funds one head per
configured slot. Over-cap all-head waves refuse before inputs, state growth or
planning. Frontier, state-only and retained-feature input envelopes remain full.

Gemma's [prefill lookahead](experiments/gemma-prefill-lookahead/README.md)
accepts token-free next-chunk shape hints from scalar and wave callers. The
threaded node runs a host-only graph/placement callback after job submission
and before its normal completion wait. That callback cannot access node or
launch state, stage inputs or prepare future KV. Optional host funding covers
the temporary plan through post-completion binding and cache transfer; a
missing or refused prediction uses normal planning. Gemma2 and Gemma3 take the same
hints, extended to the chunk after next ([graphs captured ahead](experiments/gemma3-execution/README.md#prefill-lookahead-and-graphs-captured-ahead-2026-10-07)):
they build up to two distinct missing upcoming shapes under independent
allowances,
capture a shape the next chunk repeats on its first run, and capture the next
planned shape beside current execution. The [two-future transfer](experiments/prefill-transfer/README.md#two-distinct-future-shapes-2026-10-08)
checks cold changing-width replay and exact state; the CPU plan callback does
not prepare future KV.
Gemma4's explicit plain capture-ahead comparison uses the same group and
next/after descriptors; the scalar adapter forwards both stages. Ordinary
Gemma26/Gemma31 remain off/one-future. Explicit capture may now retain one
frontier feature per owner: the actual post-replay D2D copy remains outside
the recorded plan. Verification, greedy and all-output/all-feature capture
stay excluded; joined target and broader context/policy gates remain open.

Gemma2, Gemma3 and both Gemma4 profiles separately [prepare fresh zero backing](experiments/gemma-state-prepare-ahead/README.md)
for a known next chunk before the current Job. A LiveState-owned no-victim
acquisition ticket is funded and authenticated before submission; the host-only
meanwhile contract stays unchanged. Future backing does not enter used ranges,
advance cursors or acquire write-back authority. After the current fence, a
scoped existing-page-in drain proves retirement before collecting unused zeros
as reclaimable backing. Actual Use alone publishes initialized ranges. Clear,
source changes and teardown drain first; unknown completion retains the complete
owner, and the Runtime Server fails stop before member destruction if teardown
cannot retire it. Their defaults select eligible fresh zero sources; kept
restart sources and fresh sparse-file reads retain their current path. Other
growing adapters and broader context/performance qualification remain open.
The [Gemma4 transfer](experiments/gemma4-state-prepare/README.md) reserves up to
120 range descriptors, submits only after verify-save prechecks, and drains all
owners before rollback, features or cursors publish. Post-write preparation
errors quarantine all current owners and invalidate retained feature authority.
Actual approved 8K/C4 PromptSession controls match complete heads/history/state
through owner departure; prefill itself remains scalar.
The already admitted Gemma3 8448-context scalar route matches exact full
prompt/continuation heads, histories and initialized state at 8192/8193 with
positive default preparation; no new speed or admission claim follows.

Gemma's [assistant component](gemma4-assistant.md#native-component-and-protected-target-operands)
adds explicit post-finalnorm feature retention and scoped readonly cache
borrows to that skeleton. It shares the target launch/cohort and uses separate
funded recurrent storage, plans and staging. A separate default-off
[Gemma target verifier](experiments/gemma-target-verify/README.md) completes
one to four C1 rows, retains explicit heads/features and settles accepted or
rejected KV writes before publishing a cursor. Its focused transaction proof
does not establish scalar-prefix arithmetic parity, assistant serving
speculation or full model support. The optional
[Gemma C1 greedy transaction](experiments/gemma-assistant-greedy-unit/README.md)
composes the component and verifier under one held request: release the borrow,
verify anchor plus drafts, accept after retirement, then publish only committed
tokens and selected full head/feature. Its next anchor remains uncommitted.
Both-profile focused exact controls use a same-four-query independent target;
serving and scalar-width quality remain separate.

One [matched Gemma31 target-plus-assistant transaction](experiments/gemma-assistant-greedy-reference/README.md)
now also matches full original heads/features and acceptance at C1/P64/depth3
under the explicit native target norm chains. Its independent normal Wave4/reset
checks preserve the engine's semantic-prefix oracle. That one-unit control adds
no engine defaults, serving admission or scalar-width quality; the bounded
repeated performance screen follows below.

A subsequent [Gemma31 repeated assistant screen](experiments/gemma-assistant-throughput/README.md)
times the actual engine unit, including drafts, verify/rollback/accept and
retired publication, for 32 identical committed tokens across all four native
and original plain/assistant arms. Native assistant elapsed is 55.24% below
native plain and 1.25% above original assistant on this short C1/P64 suffix.
Own repeat/timed carry checks pass; EOG is ignored, with no terminal/serving,
other-prefix, sustained or 26 qualification and no engine default change.

What a new family writes, and nothing else:

- `model/<family>.h`: its profile, the binding to an artifact, its state
  layout (bytes per region, where each tensor lives), each chunk's
  host-built inputs, and for speculation which state ranges a verify
  writes, by row.
- `kernels/ggml/<family>_graph.h` (or another kernel module): the chunk
  graph builder and shape; its operations registered (D-053) with a
  primitive fallback for any fused one ([portability.md](portability.md#the-registry-rule)).
- `engine/<family>_plan.h`: `using <Family>Planned =
  PlannedGraph<Graph>`, the function that builds, binds the weights and
  state at the model's places, and calls `PlaceAndPlan`; the host inputs
  in the graph's copy order.
- `engine/<family>_runner.h`: a `PagedModel` holding `RunnerResources`,
  `LiveState`, `GraphRuns`, a `PagedWeights` per artifact and a
  `PlanCache` per kind of plan bound to its `PlanAccount` at Bind, its
  `plan_floor_bytes()` (what one step holds at once, from its kinds' largest
  plans measured at Setup: the memory
  guard sets it apart), its plans' `ReclaimCandidates` and `Reclaim` and its
  slots' `Spill` and `Restore`, with Setup, Register and Bind as above and
  the family's steps. Gemma3's bounded runner also composes this skeleton,
  with independent slot roots, graph-owned device masks with a funded host
  reference path, separate head-row capacity,
  state-only prefill, explicit kept device-argmax publication and exact
  Clear/restore controls. Its bounded serving adapter adds completion-aware
  checkpoint coverage, kept-conversation adoption and unchanged held-cohort
  reuse, with pending-restore refusal and peer protection. Compatible
  Gemma3 prefill uses the shared prepare/dispatch/retire/apply prompt seam:
  all participants stay borrowed until completion publication, clean capacity
  refusals omit only that owner, and failed shared work invalidates unpublished
  histories. Per-owner rows/checkpoint layouts stay 128 while total wave
  funding is 256. The internal H8/no-cap graph now joins canonical C2..C12
  one-row owners, grouping only attention into existing four-root carriers and
  retaining whole-wave products and one common read maximum across groups.
  Setup measures every proper unequal-padding multiplicity and funds twelve
  immutable heads. Cap50 remains C2-only; bounded roots additionally have a
  false-default whole-C12 H8/no-cap factor that retains the global mask maximum
  even in an all-short four-root carrier. Partial bounded carriers remain closed.
  An [actual C3 pause/catch-up/rejoin control](experiments/gemma-h8-c3/README.md)
  preserves peers' initialized state/cursors and exact own GPU/checkpoint replay;
  its same-schedule stock gate passes. The [wide continuous control](experiments/gemma-h8-wide/README.md)
  adds unequal/partial/departure/refill peers and exact own restore, with zero
  stock strict differences over 472 heads. Its separate exact-final-head cost
  gate remains failed despite exact natural choices; ordinary slots remain two,
  with no solo/batch allowance. The separate
  [whole-C12 bounded factor](experiments/gemma-h8-wide-bounded/README.md)
  preserves all native natural outputs exactly and lowers short n=2 decode by
  21.0667%; it does not alter that stock gate or public admission.
  Other families retain the default scalar seam. The DeepSeek
  and Qwen3.8 runners are the worked examples: DeepSeek with a host table, a chained draft-and-verify job and
  a snapshot of every written range; Qwen3.8 with rows read on demand and
  gathered between the inputs and the plan, a commit kernel, and two
  variants of a verify's plan.
- `runtime/serving.cc`: a `Served`/`Llm` adapter that forwards to the
  runner (its tokenizer and template, chunks, speculative step, its
  `CheckPlaces`), and the architecture name that selects it. The adapter
  takes every setting from the model's `ModelSettings`, never from a
  constant of its own (D-103).
- `runtime/model_settings.*` and `config/node_config.cc`'s `ModelKeys`:
  the family's settings. Its runner's context ceiling (derived from the
  model spec and stage position limits); the metadata its defaults
  derive from (the trained context, the drafter's block, sampling
  defaults: read by `ArtifactFactsOf`, never a checkpoint's name or hash);
  each setting of its own a key in `ModelKeys` governed by capability
  predicates, its fallback a constant in `model_settings.h` with what
  measured it, its resolution in `ResolveSettings` and its line in
  `ModelSettings::Lines` (the unit test checks every key is listed). A
  measured speed trade becomes a calibration once the machine can measure it.

Outside the engine, a family may also need its import to a prepared
artifact (`docs/experiments/artifact-layout/import_m3.py` today), its
tokenizer's pre-tokenizer if it is new ([tokenizer.md](tokenizer.md)), and
optionally a native chat renderer (`chat/`, chosen by template hash or
probe equivalence, D-067); without one, its template runs through the
sandboxed interpreter.

The checks a family gets by composing the skeleton: the pinned places
after each swap (the runner's `CheckPlaces` over its `PagedWeights` and
`LiveState`, forwarded by its adapter; the adapter's default checks
nothing), BP-A1's coverage of every planned shape, the quarantine, and
graphs that replay only with the staging they were captured with.

DeepSeek production prefill with the measured Q4_K head explicitly
requests one frontier head row;
its graph's default remains every row. Requested output count is part of
the planned shape, and copies use the graph's actual logit rows. The
gather narrows only the final mix/norm/head, after all target layers and
feature capture, so DSpark injection retains the full chunk's streams.
Reference, verify and named diagnostic plans retain all requested heads.
Qwen3.8 uses the same separation for its frontier and MTP streams.
Opted-in shared attention retains CSA and window sharing, while existing
count-based HCA masks select the registered ordinary MMA path. This is
an operation-semantic choice, independent of context or artifact identity.
Fresh 32K/128K head repeats, PPL and bounded state/rollback/swap controls
qualify that combination; final sampled and maximum-context runtime gates
remain separate.

What the skeleton assumes today, which a family that differs changes here
rather than works around:

- **GGML plans.** `planned.h` and `GraphRuns::Queue` run a GGML
  `BoundGraph`, and `RunnerResources::BindLaunch` binds against GGML's
  implementations only. A family run by another executor (EXL3's, or its
  own kernels as Qwen-Image's are) composes the weights, resources and
  live state but runs and captures its own steps, as the image runner
  does.
- **State access described by the model.** Each live-state region reserves
  its layout's virtual bytes. DeepSeek and Qwen3.8 materialize only the
  extents their padded cache prefixes and fixed rings or recurrent state
  use; page-in, spill and restore follow that initialized set. A verify's writes are ranges by row, and state
  a verify rewrites whole (recurrent) is restored and then rebuilt for the
  kept rows by the commit hook, as Qwen3.8's is.
- **One target and one drafter.** Regions, weights and plan caches are the
  runner's own members; nothing here counts them, but the runners and
  their adapters are written for a target with at most one drafter.

## Where the long-context work goes

- **State that grows with use**: `LiveState::AddGrowing` reserves stable
  virtual addresses; `Use` materializes named ranges before dispatch.
  Sources and graph address pins are registered once, with sparse zeros
  for unused extents. Initialized extents gain write-back; `Retain` drops
  discarded tail pages and their saved bytes. A clear within the slot's
  request first zeroes its resident used extents (`ZeroForReuse`), so the
  discard keeps their backing outside the state: invalidated in the
  catalog (neither leasable nor registrable), taken back by the next
  growth with no zero load (`Catalog::ReviveDiscarded`). Every runtime
  reclaim releases all kept backing before pricing anything else, and so
  do materialization's victims, a model's eviction, a swap, teardown and
  a discard that does not keep. Model footprint functions
  cover padded attention reads, dummy cells and fixed rings. Packed
  snapshots carry only used pages. A clean capacity refusal preserves the
  completed prefix; uncertain completion quarantines it.
- **Turn-boundary checkpoints**: the model's `CheckpointWrites` names
  future mutable ranges; the runner includes its drafter and
  `CheckpointPages` selects whole used physical pages overlapping them.
  `CheckpointFile` captures those pages into an unnamed private direct-I/O
  file with transient cataloged staging. Immutable earlier cache pages
  remain in the branch's live state or spill file. Restore first retains
  the original footprint and discards newer tail pages, then copies the
  saved pages back. Uncertain copies quarantine the state and retain the
  original staging allocation.
  Per-page continuation checks let the watchdog observe progress and
  limit cancellation to the transfer currently in flight.
- **Turn-to-turn prefix reuse**: the runtime's `PreparePrompt` compares
  exact tokens, reuses a complete live prefix or restores the nearest of
  two retained boundaries within their common prefix, and prefills the
  suffix. Boundaries precede the renderer's unstable assistant opening.
  Entries carry history position, cursor and adaptive decoding state,
  expire for reuse after 24 hours and disappear at clear or restart;
  cleanup during idle time is lazy. See
  [the native control](experiments/turn-reuse/README.md).
- **Resumable prompt preparation**: a runtime branch's `PromptSession` owns
  the bounded prompt and advances reuse/restore, one ordinary prefill chunk,
  or checkpoint capture as separate completed units. The legacy prompt path
  drives the same session. This seam applies to both LLMs and keeps native
  state, chunk arithmetic and completion ownership in their existing runners.
  The production Qwen chat backend uses this seam to interleave prompt units and
  peer decode without moving a borrowed request frame.
- **Deterministic top-k and sparse prefill attention**: graph builders and
  kernels (`kernels/ggml/`), selected per shape by the plan; the skeleton's
  plan cache and graphs take them unchanged, and a prefill shape's capture
  rule is the runner's. DeepSeek's fast plan has them (its window cache a
  ring, its window the spec's attention parameter; `llmp.dsv4.lid_topk` and
  `sparse_mask`, [long-context](experiments/long-context/README.md)), and
  so does Qwen3.8's fast graph (`llmp.qsa.pool`, `.topk`, `.attn`; its
  block keys a state tensor of the model's layout, their verify saves the
  runner's).

## Shared partial-weight foundation and completion news

Ordinary runtime switches still evict the outgoing prepared footprint. The
internal partial swap benchmark uses the shared server/node transaction to
retain clean inactive weights, select a global GreedyDual capacity deficit,
release incompatible handoff backing, and reload only missing closure members.
Whole outgoing state spills first. Protected generation, lease and registration
checks, zero-victim request retirement, saved-state proof and cache-preserving
rollback apply to every model adapter. Runtime-managed state acquisition
validates the entire closure before invoking the installed reclaimer; callback
refusal or reentry never enables the standalone LRU fallback. Positive completed
storage READ bytes are measured separately from writes, failures and zero-fill.

The internal partial path remains default-off because its measured cold-switch
tradeoff leaves the ordinary adoption gate open. The shared WakeFlag handshake
now notifies after unlocking; CompletionBoard removes at most 64 already queued
indices per news lock, unlocking before mailbox reads and observation. Neither
waits to fill a batch nor adds per-step scheduler calls. All current families
inherit these shared changes; actual performance was screened on DeepSeek/Qwen,
with concurrency/lifetime controls and whole-node lock counters. Their narrow
publication/eviction benefits establish no repair of the intermittent long tail
and no per-family speed claim. See [partial-weight evidence](experiments/partial-weight-eviction/README.md).

Independent GGML cache stores can use the shared checked physical grouping
implementation, preserving primitive conversion, dependency order, lane tags
and completion ownership. Gemma3 selects it; other measured recipients stay
off after neutral results. [T86 controls and scope](experiments/grouped-kv-stores/README.md).

### Scalar nonfinal output intent

DS and native/GGUF Qwen now carry actual PromptSession `want_head` through current
and predicted complete cache keys. Ordinary non-speculative serving selects
state-only nonfinal chunks; low-level plans stay explicitly headed. Plain final
blocks stop after required cache/compressor or convolution/recurrent/QSA writers.
DSpark final features and MTP exported streams require the full trunk, so their
checked head-only variants remain explicit opt-ins. Scoring and checkpoint
settlement preserve required heads, pending rows and carry. Both intent kinds
are measured for Setup funding; pruning does not assume a smaller placement.
[Focused cuts, actual Setup cost and replay](experiments/ds-qwen-state-only/README.md).
