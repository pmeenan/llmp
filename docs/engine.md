<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The engine

`src/engine/` is the layer above the kernels that runs models on a paged
node (D-096, D-107; [architecture.md](architecture.md#layers-and-dependency-rules),
[engine-components.md](engine-components.md)). It holds the node, one skeleton
of shared mechanics, the single token model runner (`ModelRunner`,
`src/engine/model_runner.h`) with its shared slot lifecycle (`SlotLifecycle`,
`src/engine/slot_lifecycle.h`), and the stateless staged runner
(`StagedExecution`, `src/engine/staged_execution.h`) for multi-stage pipelines
such as image generation. No per-family code lives in `src/engine/`: all LLMs
(Gemma 2, Gemma 3, Gemma 4, Qwen2, Llama, Mistral, Qwen3.8, DeepSeek V4 Flash)
and drafters (`gemma4_assistant`, `dspark`) execute on `ModelRunner`, while
`Qwen-Image-2.1` executes on `StagedExecution` over the same program, cache,
capture and memory lifecycle. `llmp-runtime` serves through it
([runtime-serving.md](runtime-serving.md)) and the paged harnesses drive the
same code (`tests/support/paged_node.h`).

**Two libraries** (D-096 as amended for M3.6). `llmp_engine_core` builds
in every profile and names no kernel library: the node (`paged_node.h`), the
weights' paging (`paged_weights.h`), live state (with its spill, checkpoints
and a verify's snapshot: `live_state.h`, `checkpoint_file.h`), the runner's own
memory and paid node root (`runner_memory.h`, `runner_owner.h`), what plans hold
(`plan_account.h`: the charge, the step a reclaim spares, the reclaimer's view
of a cache), prefill lookahead (`prefill_lookahead.h`), the request cohort
(`request_cohort.h`), the shared slot lifecycle (`slot_lifecycle.h`), on-demand
row tables (`ple_rows.h`), and the shared runners over step programs
(`model_runner.h`, `staged_execution.h`, `step_runs.h`, `step_inputs.h`,
`step_transaction.h`, `plan_cache.h`, `coverage.h`) with the wave composer
(`wave_composer.h`), prefill orchestrator (`prefill.h`), decoding modes with
any number of drafters (`decoding.h`, `draft_companions.h`),
and the component step source (`component_source.h`, `graph_plan.h`,
`execution_contract.h`, `program_lifecycle.h`, `supplemental_run.h`). It reaches
the device through the providers, the device runtime and the paging kernels
(`kernels/paging/paging.h`), and links no device backend: a program links the
CUDA one, or the fake device runtime (`providers/fake/fake_device_runtime.h`:
host memory, work queued on fake streams that runs when a fence is queried) with
the paging kernels' host loops (`llmp_fake_device_backend`), which the core's
tests use in every profile. `llmp_engine` (CUDA builds only) adds the runners'
cuBLAS and launch context (`runner_resources.h`) and the expert slabs
(`ExpertSlabsOf`, `expert_slab.h`).

## The node

`paged_node.h`: device 0 with its providers, a stream per model and the
copy stream, the storage ring, one catalog domain, the landing zone
(D-081), the scheduler and its lanes, and the workspace the models share
(activations and the GGML pool; in the runtime it is resident for the
node's life when it fits beside the largest model's weights, and otherwise
pages with the active model; either way each model's closures and steps
are bounded to its own prefix of it,
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
| Live state | `live_state.h` | Stable virtual regions registered before the weights, physical backing only for used extents; successful source/pin checks reused under scheduler lifetime/epoch with mutation invalidation (not residency or leases); sparse unnamed direct-I/O spill files, clear, packed copies, pinned places and quarantine; a verify's snapshot, accept, owed restore and commit, rollback, and undoing a failed verify, queued on the job's stream as the paging kernels' range copies | The state layout and each step's used ranges; which ranges a verify writes; a commit if kept rows need one |
| Planned shapes | `plan_cache.h` (`plan_account.h`, `coverage.h`) | `BasicPlanCache` (per key, found through an index of the keys' hashes, variants, no fixed number, each plan's host bytes and planning time kept); `PlanAccount` (what the plans and graphs hold, charged to the node: a plan as it is added, a graph before its capture, D-090 as amended), `PlanStep` (a step's plans spared by any reclaim while it runs), `PlanCacheBase`, `CollectPlans` and `ReclaimPlan` (the plans and graphs as candidates for the node's reclaim order, and one's reclaim); the coverage check (BP-A1) | The step source and its plan key |
| Runs and graphs | `step_runs.h` (`run_stats.h`) | `StepRuns`: stages a run's inputs in the pinned staging, queues copies, work between inputs and plan, the step program and the outputs; captures a shape (or when the runner asks), replays its graph from then on with the staging checked, falls back to launch by launch on a refused capture; records another planned shape's graph without running it; `GraphStats`, `RunPath` | When a run may be captured (decode steps, verifies, drafts, hinted prefill) and what it copies out |
| Prefill lookahead | `prefill_lookahead.h` | `PrefillLookahead` and fixed-capacity `PrefillLookaheadGroup`: independent optional host allowances before CPU-only future plans, sequential builds during current device work, ordered installation after successful completion, exact cache-charge transfer and destruction before unused-grant release | Shape prediction, graph construction, binding/coverage/cache installation and capture policy |
| Resources | `runner_memory.h`, `runner_resources.h` | The runner's own device memory (pinned) and pinned staging (`RunnerMemory`, the core's); for a GGML runner, cuBLAS and its workspace, a measuring launch context, the launch context over the pool and the registry (`RunnerResources`); and their completion-aware release (AGENTS.md rule 6) | Sizes and names |
| Request cohort | `request_cohort.h` | Several request slots of one model: the active set selected between completed units (several only under one held stream request), the closures (everything, the state fence, the execution closure, each slot's fence) and the held request's refresh, and the cohort's fault (every slot quarantined until retirement) | Its slots' live states and the shared extents |
| Slot lifecycle | `slot_lifecycle.h` (`model/state_contract.h`) | Request slots of one LiveState per stage (target, then each drafter) and every verb that moves their state, a verify's Accept, discard and settle among them, derived from the stages' state contracts, with the reconciled guards ([below](#the-shared-slot-lifecycle)); used by all models on `ModelRunner` (Gemma 2, Gemma 3, Gemma 4, Qwen2, Qwen3.8, DeepSeek V4 Flash, drafters, and image compositions) | The stages' contracts and read alignments, the shared extents, the spec's digest, its invalidation and plan-dropping hooks, and a write record before every job that writes a slot |

### Engine file inventory (`src/engine/`)

The module contains only shared engine code (no per-family code):

| Header | Implementation | Role |
| --- | --- | --- |
| `batch_execution.h` | — | Shared batch execution contracts and callbacks |
| `batch_inventory.h` | `batch_inventory.cc` | Batch shape and resource inventory |
| `batch_resources.h` | `batch_resources.cc` | Shared batch resource lifecycle |
| `checkpoint_file.h` | `checkpoint_file.cc` | Direct-I/O turn boundary checkpoint files |
| `component_source.h` | `component_source.cc` | Step source (`ComponentStepSource`, `StagedStepSource`) built from components |
| `copy_sink.h` | — | Output copy destination contracts |
| `coverage.h` | `coverage.cc` | Tensor role and coverage checks |
| `decoding.h` | `decoding.cc` | Decoding modes (`DecodingOrchestrator`): plain, sampled, speculative cascade/alternatives |
| `draft_companions.h` | `draft_companions.cc` | Shared prompt wave execution and batching |
| `execution_contract.h` | `execution_contract.cc` | Execution contract validation and lifecycle |
| `expert_slab.h` | `expert_slab.cc` | Expert slab layout and pitch (`llmp_engine`) |
| `graph_plan.h` | `graph_plan.cc` | Planned step graph binding and lowering |
| `host_buffer.h` | — | Bounded host buffer helpers |
| `live_state.h` | `live_state.cc` | Virtual state regions, sparse spill files, verify snapshots |
| `model_runner.h` | `model_runner.cc`, `model_runner_routed.cc` | The single shared model runner over step programs |
| `paged_node.h` | `paged_node.cc` | Paged node: device, streams, io_uring, scheduler, workspace |
| `paged_weights.h` | `paged_weights.cc` | Weight files, group reservation, slab layout, direct reads |
| `plan_account.h` | `plan_account.cc` | Plan memory accounting and charges |
| `plan_cache.h` | — | Basic plan cache template, indexed by its keys' hashes |
| `ple_rows.h` | `ple_rows.cc` | On-demand n-gram row gathering and direct reads |
| `prefill.h` | `prefill.cc` | `PrefillOrchestrator`: chunking, joins and futures |
| `prefill_lookahead.h` | — | Future shape prediction and lookahead groups |
| `program_lifecycle.h` | `program_lifecycle.cc` | Step program lifecycle and resource bindings |
| `request_cohort.h` | `request_cohort.cc` | Request slots, active sets, closures, fault isolation |
| `runner_memory.h` | `runner_memory.cc` | Runner device memory and pinned staging |
| `runner_owner.h` | `runner_owner.cc` | Node-retained paid runner root and authenticated retirement |
| `runner_resources.h` | `runner_resources.cc` | cuBLAS handles, workspace and launch context (`llmp_engine`) |
| `run_stats.h` | — | Execution statistics, graph paths, run counts |
| `slot_lifecycle.h` | `slot_lifecycle.cc` | Shared request slot lifecycle derived from state contracts |
| `specialization.h` | — | Step specialization descriptors |
| `staged_execution.h` | `staged_execution.cc` | Stateless multi-stage execution for image pipelines |
| `step_ahead.h` | — | Lookahead and capture-ahead descriptors for `RunStep` |
| `step_inputs.h` | `step_inputs.cc` | Computed step input staging and side-row derivation |
| `step_runs.h` | `step_runs.cc`, `step_source.cc` | Step program input staging, positions, graph capture/replay, outputs |
| `step_transaction.h` | `step_transaction.cc` | Per-step admission, state reservation, and rollback transactions |
| `supplemental_run.h` | `supplemental_run.cc` | Supplemental execution and commit steps |
| `support.h` | — | Address, rounding, and error helpers |
| `wave_composer.h` | `wave_composer.cc` | Wave composition across request slots |

Startup measurement can skip greedy activation placement when a checked
disjoint root-storage bound fits the current exact maximum. Gemma2/3,
DeepSeek/DSpark and Qwen3.8 select it; Gemma4 keeps its optional
dense-sharing policy. Qwen scalar and wave ceilings are independent.
Measurement-only plans cannot bind, and ordinary runtime plans retain exact
placement. [Startup controls and scope](experiments/startup-placement-transfer/README.md).

`support.h` holds the small helpers (errors, addresses, rounding, seconds,
joined problems).

All LLM architectures on `ModelRunner` use the shared prefill lookahead
lifecycle (`prefill_lookahead.h`, `prefill.h`, `step_ahead.h`), governed by
each architecture mapping's `PolicyRule` (`prefill_lookahead`, `capture_ahead`).
`StepRuns::CaptureAhead` records an eligible future step program without
executing it. Gemma2/Gemma3 select two distinct missing futures. Both Gemma4
profiles compose the same fixed group, with ordinary policy retaining one
future and no capture ahead: the
[plain Gemma26 8192-row diagnostic screen](experiments/gemma26-capture-ahead/README.md)
replays six chunks without a meaningful paid gain, and the separate
[retained-frontier opt-in](experiments/gemma26-feature-capture/README.md)
measures −0.484% prefill and exact assistant consumption. Cached
current/next/after keys are protected before optional capture funding. Each
future has independent refusal and cleanup; the CPU callback publishes no future
logical state or cursor. DeepSeek V4 and Qwen3.8 select two-stage CPU planning
and independently prepare eligible fresh backing before the job, draining it
after completion and before current publication. Qwen3.8 funds two target plus
two MTP plans (`draft_companions.h`); its companion prediction uses post-success
pending state. [Actual controls and factors](experiments/ds-qwen-prefill-prediction/README.md).
`PromptSession::NextPrefillHint` supplies the same checkpoint/scoring-aware
descriptors to scalar and joined prefill after current funding. Each future
stage independently filters ended owners and requires homogeneous head intent;
a mixed next stage does not suppress a valid after stage. Gemma4 serving
prefills one prompt chunk at a time (`prompts_alone = true`).
[Focused lifecycle controls](experiments/prefill-transfer/README.md).

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
   (`PagedWeights::Open`), read the stored model spec (`spec/stored.h`) and
   compose any external drafter (`spec::ComposeDrafter`), lay out each stage's
   state (`model::ExtentLayout`, `SlotLifecycle`); add the live-state regions
   first, then any runner-owned device memory (`RunnerConfig::kernel_memory`
   via `RunnerMemory`), reserve the weights and expert slabs
   (`engine::ExpertSlabsOf`), and build every configured setup shape placeless
   through `ComponentStepSource` (`planner/plan.h`) to size the activations,
   pool, staging and plan floor (with a quarter's margin); pin the staging
   (`StepRuns::SetStaging`); map verify snapshots and commit buffers if the
   model speculates (`LiveState::SnapshotAt`, `StepSource::CommitBuffers`).
2. **Register** (after the node's Start): each shard a weight file of
   the model's and each weight's source reading it by identity
   (`PagedWeights::Register`; the shards open at the model's first
   page-in or activation, `PagedNode::SetActive`), the state's write-back places
   (`LiveState::RegisterSpill`), then every managed extent's place pinned
   (`Scheduler::PinPlaces`, D-090).
3. **Bind** (after the workspace): the closures (`everything`, the state's
   fence, each slot's fence; of the workspace, only the model's own prefix
   of each region, `ModelRunner::activation_bytes` and `pool_bytes`), the
   model's places as the step program's bindings, the commit hooks, and
   the launch context over that pool prefix (`RunnerConfig::launch`, then
   `StepRuns::SetLaunch`).
4. **Steps** (`ModelRunner::RunStep`, `step_transaction.h`): each chunk, draft
   or verify holds a `PlanStep` for its life, admits its owners and reserves
   their state (`SlotLifecycle`, `Admit`), finds or builds its `StepPlan` and
   `StepProgram` (`BasicPlanCache::Find`, else `StepSource::Build`,
   `CheckCoverage`, `Add` with its host bytes and planning time, charged as it
   is added), decides whether to capture (`StepRuns` capture rules plus
   `ChargeGraph` for every capture of the job: a graph with no room even after
   a reclaim is not made), stages computed inputs and side-row lookups
   (`step_inputs.h`, `ple_rows.h`), funds up to two missing futures
   (`PrefillLookaheadGroup`) and prepares fresh state ahead, then posts one job
   that queues owed restores and component commits, verify saves, the step
   program (`StepRuns::Queue`), and any capture ahead. A failure settles the
   state (`SlotLifecycle` / `LiveState::Settle`: a verify undone, else a
   quarantine); success installs completed futures and counts the path.
5. **Release** (after the node's teardown): the plans and graphs first
   (they name the launch context and the memory), then the launch context and
   `RunnerMemory`, `SlotLifecycle::Release`, and each stage's
   `PagedWeights::Release`.

## Independent request state

`ModelRunner` exposes stable request slots (`RunnerConfig::slots`, at most
`kMaxRequestSlots`, 16, `request_cohort.h`) managed by `SlotLifecycle`
(`slot_lifecycle.h`). Each slot owns one `LiveState` per stage (target, then
each drafter), verify snapshots, component commit buffers,
and feature carries. The weights, launch
context, stream, staging, workspace and `BasicPlanCache` remain shared on the
runner. The scalar runner methods use slot zero. `SelectSlots` selects an
execution closure over the active slots at a completed unit boundary; the
runner's aggregate state and swap closure include every initialized slot, idle
retained conversations too, but not a spilled one (a swap in does not restore
it; its next turn does). Every plan and graph is charged to the runner's
`PlanAccount`. A slot outside the selected set spills (`SlotLifecycle::Spill`:
its state out of every closure, written to its spill files, its backing
released) and restores (`SlotLifecycle::Restore`) before its next work; `Clear`
discards a spilled slot's saved state too. Setup plans the widest waves its
slots can run (every slot at the context's end with its most rows, target and
draft) over placeless addresses, and the shared workspace, pool, staging and
host inputs are the larger of a chunk's needs and those waves', with a
quarter's margin: a slot adds its few-row shapes, not another prefill chunk's
(6.80 → 1.53 GiB of workspace at four slots;
[request slots](experiments/request-slots/README.md)).

`WaveComposer` (`wave_composer.h`) and the planner (`planner/plan.h`,
`planner/owner_products.h`, `planner/fusions.cc`) compose multi-slot waves from
the components' `model::Batching` rules and the architecture's `PolicyRule`
without per-family wave builders. Consecutive compatible slots form groups
(`group_rows`, `group_segments`, `invariant_group_segments`), joining stateless
products across owners while keeping stateful operations (attention, KV writes,
recurrent state, compressors, feature taps) per owner or per owner-attention
carrier:
- On Qwen3.8, consecutive compatible slots form a group while their rows fit 16,
  joining target/draft MXFP8 and routed vector products, eligible 2–4-row HC
  BF16 products, 3–4-row BF16 target heads, and up to 8 single-column BF16 MTP
  heads ([head-sharing controls](experiments/qwen38-draft-head-waves/README.md)),
  with 2,048-aligned joined reads (`joined_read_align`) and fused GDN
  alpha/beta pointwise chains on verify waves ([controls](experiments/qwen38-gdn-gates/README.md)).
- On DeepSeek V4 with DSpark, waves hold at most 16 rows (`max_step_rows`),
  running row-local operations (quantized products, routed experts, HC mixes,
  routing, norms, the head) once over every slot's rows through
  `planner/owner_products.h` while each slot's compressors, indexer, attention,
  cache writes and DSpark injection run on its own state
  ([DeepSeek batching report](experiments/deepseek-batching/README.md)).
- On Gemma 2, Gemma 3, and Gemma 4, owner-attention carriers (`fattn_owner.h`),
  bounded owner roots, and common-width reads join decode and eligible prefill
  waves within each mapping's policy envelope.

Clearing or restoring one destination first renews the request's protection of
its peers, then discards only the destination's eligible backing and renews the
selected closure. A separately held destination refuses discard. A proven local
failure while discarding the destination invalidates that slot; clean validation
refusals preserve the existing state. An unproven device or shared-execution failure
stops the cohort and preserves borrowed owners until retirement is established.
`EngineLlm` forwards each host branch to its corresponding slot in `ModelRunner`
and proves retirement through a native fence before any borrowed owner is released.

## The shared slot lifecycle

M3.6's replacement for the lifecycle each family wrote (D-107;
[engine-components.md](engine-components.md#execution)), in the core and
tested on the fake backend. All LLMs (Gemma 2, Gemma 3, Gemma 4, Qwen2,
Qwen3.8, DeepSeek V4 Flash), drafters (`gemma4_assistant`, `dspark`), and
`Qwen-Image-2.1` run on `SlotLifecycle` (`slot_lifecycle.h`).

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
modes) and, over the component step source and GGML's lowering, on all
supported models. All LLMs (Gemma 2, Gemma 3, Gemma 4, Qwen2, Qwen3.8,
DeepSeek V4 Flash), drafters (`gemma4_assistant`, `dspark`), and
`Qwen-Image-2.1` run on `ModelRunner` (`model_runner.h`) and are served
through [the serving adapter](#the-serving-adapter).

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

**Planner types** (`execution/step_shape.h`, header-only, re-exported by
`planner/step_shape.h`; the planner, `planner/plan.h`, plans a step's graph
and makes its key, [engine-components.md](engine-components.md#the-graph-ir)).
`StepShape` {stage, `Phase` (prefill, decode, verify, draft, score, commit),
segments by ascending slot, row invariance, read alignment}; `Segment` {slot,
rows, past, read cells, outputs, group, want_head, greedy}; `PlanKey` {spec
digest, shape, pattern and policy digests, partition, capturable}. A segment's
`past` is data, not the plan: equality ignores it, as every family's shape
equality ignored `n_past`; `read_cells`, which follows it at the shape's
alignment, and `capturable` (whether the step program avoids position-dependent
baked state and may be recorded in a device graph) are part of the plan. A
capturable step source's program therefore bakes nothing that follows `past`:
positions reach it as input data; uncapturable plans receive positions through
`Position`.

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
   run in permitted `capture_phases` also named by `automatic_capture_phases`,
   or an explicit hinted repeat/future in a permitted phase) and `ChargeGraph`; the next shape's recording
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
A table is paged this way only when it is larger than the pinned landing
and slots its reads need for a setup shape's lookups (`RowPagingPays`):
Qwen3.8's n-gram table is; DeepSeek V4's hash-routing tables are resident.
Their full artifact groups stay unplaced; each table owns a direct-read ring,
pinned landing and offsets, and pinned device weight slots sized from its
setup shapes. Landing byte offsets are explicitly 64-bit through the planner
and both gather implementations; row counts and slot IDs retain their bounded
32-bit representation. Every side-row input must be supplied in full and pass the
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

Components may derive side-row IDs from authenticated full history through
immutable source metadata (`step_inputs.h`). Staging uses the declared token
coordinate.
Explicit row inputs remain authoritative. Missing, malformed or partial inputs
refuse before state admission for resident and on-demand tables alike;
duplicate input slots for one table refuse before paging, while a joined
input covers multiple owners.

`PlanCache` is `BasicPlanCache<Key, Planned, Variants, Runs, Hash>` in
the core (`plan_cache.h`), found through an index of its keys' hashes
(`execution::HashOf` for a `PlanKey`); `StepPlanRuns` is the step
programs' runs. `RunPath`, `GraphStats` and the run copies are in
`run_stats.h`.

**The wave composer** (`wave_composer.h`). Pending work per slot becomes
step shapes by a `model::Batching` capability (`model/batching.h`, data
the components supply): work of one kind (stage, phase, row
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

How each family's speculation maps onto the shared engine (all migrations completed in M3.6):

| Family | Maps onto | Migration milestone |
| --- | --- | --- |
| Qwen3.8 MTP | An autoregressive drafter with `catch_up` over its own stage (K, V and indexer cells truncating; the streams a per-row tap the target writes, snapshot); the target's K, V, indexer and completed block keys snapshot by row, its linear layers' state `kCommit` with generated component commits; bounded on-demand n-gram rows and chunk-feature inputs | Completed in M3.6 (slice G3b/G3c); served via `EngineLlm` |
| DeepSeek V4 + DSpark | A block drafter (its fill the mask token) over its ring (a position ring the target's injection writes); the target's window cells, compressor rings and compressed rows snapshot by row, no commit; the exact mode a row-invariant verify | Completed in M3.6 (slice G4b); served via `EngineLlm` |
| Gemma4 assistant | An anchored autoregressive drafter with `borrows_target` and `features`, its contract viewing the target's KV (`borrowed_from_target`) and its tap of the target's final-norm rows; the target's KV truncates (a ring holds a chunk's rows beside its window, so a rejected row is never read) | Completed in M3.6 (slice G2c); served via `EngineLlm` |

What the shared pieces reproduce of each family (completed in M3.6):

| Family | Reproduced in the component engine | Notes / out of scope |
| --- | --- | --- |
| Gemma2/3 | Joined prompt waves with equal rows and reads (`equal_prefill`); two futures, first-run capture of a repeating shape, capture ahead, prepare ahead; head-or-state-only shapes and `max_head_rows`; second-run capture in every phase; independent outputs; owner attention (owner decode, owner and flexible owner prefill, bounded roots, padding to a common width, Gemma3's four-root carriers: the planner's `owner_attention` from the step's shape); cohort `max_segments` | The C12 bounded factor and the copying packed prefill are not ported |
| Gemma4 | Quads as `group_segments` 4, the 8/12 groups as `invariant_group_segments`/`group_segments`; one future without capture ahead (`prefill_lookahead` 1, `capture_ahead` false); per-group isolation in `RunWave`; the assistant's frozen borrows, tapped features and carry (decoding modes); prompts stepping alone and, outside the envelope, one owner a step (its mapping's rules, `prompts_alone`, `owners_alone`); bounded owner roots and common widths (the planner's owner-attention options) | Joined prompt admission (would change the routed products' arithmetic; M3 prefilled alone) |
| Qwen3.8 | Groups of consecutive owners while rows fit 16 (`group_rows`), 2,048-aligned joined reads (`joined_read_align`), state backed through that alignment before the step (an owner out of capacity there steps alone at 256 rather than being refused), two futures and prepare ahead, state-only cuts; joined draft and verify waves; per-product joined limits (draft head 8, VecQ, MoE GEMV, MXFP8 columns, HC cuBLAS 16: planner patterns), BF16 head sharing by row count, on-demand n-gram row gather (`ple_rows.h`), prefill capture policy | — |
| DeepSeek V4 | Total wave rows ≤ 16 (`max_step_rows`), joined row-local work with per-owner state (`planner/owner_products.h`), scalar prefill (`join_prefill` false), two futures and prepare ahead, state-only cuts; joined drafts and verifies (decoding modes); float products over groups of whole owners fitting 8 columns and `SetVecQOneToken` for one-row waves (planner, from the shape); HCA per-position patch (`Position`, uncapturable `PlanKey::capturable = false`) | — |
| Qwen2, Llama, Mistral | Generic pre-norm dense decoder (`embedding.token`, `attention.dense` with GQA, RoPE / `rope_freqs` / YaRN, `ffn.glu` (`silu`), `norm.rms`, `head.linear` tied or untied) across GGUF, plain (unquantized) safetensors and EXL3 (`spec/mappings/qwen2.h`), served via `kDefaultServingRecipe` | Served and verified only for Llama 3.2 1B (GGUF) and Qwen2.5 0.5B (GGUF FP16, EXL3); Mistral and the safetensors/EXL3 forms of Llama are synthetic-source tested only. HF-quantized safetensors (AWQ, GPTQ, compressed-tensors, ModelOpt, MLX) have no import path for these families. Qwen2 serves single-owner (`max_decode_rows 1`, `owners_alone`) |

## The serving adapter

One adapter per pipeline kind (D-107; [runtime-serving.md](runtime-serving.md#engine-backed-models))
replaces the M3 serving classes: `runtime/engine_llm.h` (`EngineLlm final :
Llm`) and `runtime/engine_image.h` (`EngineImage final : Image`). Both
are in the runtime's `llmp_runtime_llm` library with `Llm`'s conversation
code (`runtime/llm.cc`, `runtime/llm_prompt.cc`, `runtime/llm_generation.cc`,
`runtime/llm_mixed.cc`), every build profile: they reach the engine through
the core and name no kernel library, so `engine_llm_test` drives a whole `Llm`
session over the fake backend.

`EngineImage` (`runtime/engine_image.h`, built by
`runtime/component_image_build.h`) translates source requests and observations
over the engine's `StagedExecution` (`engine/staged_execution.h`). Each stage of
an image composition (`text_encoder`, `transformer`, `vae`) is a spec stage with
typed stage `links` and `exports` (`spec/graph_links.h`, D-122); `StagedExecution`
owns typed-stage invocation, per-stage closures, geometry replacement,
`BasicPlanCache`, `ProgramLifecycle`, `StepRuns` and `RunnerMemory`. Its paid
node-owned root retains plans and backing after frontend destruction until
authenticated fences and physical retirement.

An LLM adapter is made with its model's settings (D-103) and an `EngineBuild`
that turns them into its parts (`EngineLlmParts`): the `RunnerConfig`
(every stage's layout, from the settings' context and chunk; the bound
spec's digest; the stages' weights, setup shapes, launch, verify rows and
commits), the step source, the components' `model::Batching`, the
`DecodingMode` with its drafters, the head's logit width, whether a greedy
step returns its tokens chosen on the device, and the prefill futures'
options, and what the source and launch borrow (`keep`: the spec, the
artifact's tensors, the kernel source's backend), released after the
runner. Serving's build (`runtime/component_build.h`, decomposed across
`runtime/component_build.cc`, `runtime/component_domain.cc`, and
`runtime/component_bindings.cc`, with admission in
`runtime/component_admission.cc` and reference-fleet pinning in
`runtime/approvals.cc`) reads the spec its v1 artifact stores (D-118, D-119)
and, when the model speculates, composes its drafter's (`spec::ComposeDrafter`:
Gemma 4's assistant or DeepSeek V4's DSpark, a stage of its own), takes each
stage's layout (`model::ExtentLayout`: each growing state tensor of an extent or
more on its own paged extents, smaller or whole-used tensors packed at 256
bytes, and a compressed cache right after its raw window) and the target's
batching from the components, and makes a `ComponentStepSource` per stage (a
`StagedStepSource` over several) planned under the mapping's policy rule for the
serving envelope (`ComponentAdmissionPolicy` over `spec::DefaultRule`, whose
rule also says how serving composes the recipe's steps: `prompts_alone`,
`owners_alone`, `prefill_lookahead`, `capture_ahead`) over a `GgmlBackend`
created on the model's own stream (the runner's job stream, which every launch
checks; one backend per runner, its launch made once at Bind). A layer's routed
experts are one slab (`engine::ExpertSlabsOf`, 256-byte and block-byte aligned),
its pitch the lowering's expert stride. The head's logit width is the spec
head's vocabulary; a greedy step returns each owner's token chosen on the
device. cuBLAS gets the workspace the runner maps for it
(`RunnerConfig::kernel_memory`: mapped and charged at Setup as the runner's own
memory, before any measurement, in every closure; the handle made by `prepare`
over it and lent to the backend's launch, and retired at Release after the
launch, before the memory). Its setup shapes are a prompt chunk and a decode
row, joined and alone, with logits and with device tokens, at the end of the
context reading all of it, and with a drafter a verify of its rows and a draft
pass, joined and alone. All supported LLM architectures (Gemma 2, Gemma 3,
Gemma 4 26B-A4B and 31B, Qwen2, Llama, Mistral, Qwen3.8, and DeepSeek V4 Flash)
and drafters (`gemma4_assistant`, `dspark`) are built by `ComponentLlmBuild`
(`src/runtime/component_build.h`) and served by `EngineLlm`
(`src/runtime/engine_llm.h`); reference-fleet aliases are pinned to their frozen
digests in `src/runtime/approvals.cc`, while arbitrary v1 artifacts of mapped
architectures are admitted directly by `src/runtime/component_admission.cc`
(D-119) using `ServingRecipeOf(mapping)` (`kDefaultServingRecipe` when the
mapping sets no custom `serving` recipe). Speculation on with parts that have no
drafter is refused at Setup, never served plain in its place. At Setup the
adapter makes the `ModelRunner` (owner and stream its registration index; its
slots named after the model, `max_slots` of them, spilling to the spill role or
the kept places the server sets before Register), a `WaveComposer` over the
slots, a plain `DecodingOrchestrator` and, when the model has drafters and
speculation is on, a speculative one with the adapter's policy, and prepares a
branch per slot.

The serving runner has a paid `RunnerOwner` root retained by the node. Its
explicit parts owner outlives the complete runner and all source/config borrowers. Adapter
or private-view destruction cannot orphan its mappings or run driver cleanup
on a consumer thread. Node teardown visits these roots and its caller's models
once each. Real stream fences permit clearing internal retained flights;
external private views refuse teardown before eviction or stream destruction.
Dropping those views allows retry. A successful authenticated release retires
the root. Destroying a node with unretired adopted resources fails before its
providers are destroyed; the service already exits on unproven retirement.

A producer-minted private execution view preserves the canonical registry,
selection policy and numerical authority while owning distinct binding storage
and a main scratch context. It retains the actual parent launch and borrows its
unchanged paid lanes, selected kernel state and library handle. Private GGML
views execute eagerly; ordinary programs and recordings retain their original
stable binding. Source graph planning stays canonical and checks the private
view's exact producer before lowering.

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
cursors are reset to the slot's positions before its next unit.

What the skeleton assumes today, which a model that differs changes here rather
than works around:

- **Bound step programs.** Token `ModelRunner` and stateless `StagedExecution`
  share `BasicPlanCache`, `ProgramLifecycle`, `StepRuns` and `RunnerMemory`.
  Kernel sources lower programs and supply launch contexts; they do not own
  another stage dispatcher.
- **State access described by the components' state contracts.** Each
  live-state region reserves its layout's virtual bytes (`model::ExtentLayout`)
  and materializes only the extents its padded cache prefixes and fixed rings or
  recurrent state use; page-in, spill and restore follow that initialized set.
  A verify's writes are ranges by row, and recurrent state a verify rewrites
  whole is restored and then rebuilt for the kept rows by the component's
  `kCommit` graph.
- **Any number of drafter stages per target.** `ModelRunner`, `SlotLifecycle`,
  and `DecodingOrchestrator` take a target stage and zero or more drafter stages
  from the model spec (`spec::ComposeDrafter` or in-stage MTP).

## Extending the component engine

In the unified component engine (D-107, D-108, D-109, D-119), no model family
has its own runner, graph builder, wave planner, or serving adapter. For the
record of per-family optimizations transferred into the shared components,
planner patterns, and execution lifecycle, see
[optimization-inventory.md](optimization-inventory.md). Use the five checklists
below when extending the engine.

### How to add a model of a known shape

When a new model architecture uses existing components in `src/components/`
(for example, a pre-norm dense decoder like Llama, Mistral, or Qwen2):

- **What you do not write:** no `src/model/<family>.*`, no
  `src/engine/<family>_runner.*`, no `src/kernels/ggml/<family>_graph.*`, no
  custom `ServingRecipe` unless overriding measured defaults, and no
  `src/runtime/approvals.cc` row unless pinning a checkpoint in the reference
  test fleet.
1. **Declare the architecture mapping (`src/spec/mappings/`):**
   - Add or extend a mapping header/source (for example, `LlamaMapping()` and
     `MistralMapping()` alongside `Qwen2Mapping()` in
     `src/spec/mappings/qwen2.{h,cc}`; `LLMP_DEFINE_DENSE_FAMILY_MAPPINGS`
     defines four `Mapping` objects per family: the base one and one each for
     the `gguf`, `safetensors` and `exl3` containers, all registered)
     declaring `SlotRule`s (`embedding.token`,
     `attention.dense`, `ffn.glu`, `norm.rms`, `head.linear`), `TensorRule`s,
     `ParamRule`s (including RoPE scaling/frequency rules or sliding-window
     parameters), container `FoldRule`s, and default `PolicyRule`s.
   - Leave `Mapping::serving = nullptr` to inherit `kDefaultServingRecipe`
     (`src/spec/mapping.h`: batching on, shared route, 4,096 prefill rows).
     Override `Mapping::serving` only when the architecture requires a custom
     measured context/chunking envelope, built-in MTP/block drafting wave priors,
     or non-default chat stop/thinking policies.
   - Register every `Mapping` object in `Mappings()` (`src/spec/mapping.cc`).
   - Numerical qualifications and schedule products are not inherited:
     `PolicyRule`s must name any `product ...` option the family wants
     (only DeepSeek's mapping enables the schedule products today), and
     evidence-backed numerical changes are admitted by
     `runtime/component_admission.cc` for GB10 (`arch = 121`) only.
2. **Register container normalization tables (`src/import/source_mapping.cc`):**
   - Add `SourceMapping` entries in `FindSourceMapping` for each supported
     source container (`gguf`, `safetensors`, `exl3`; `mlx` and HF-quantized
     safetensors are served only by the pinned Qwen3.8 normalizers,
     `ValidateSourceMapping` admits no other), wiring Hugging Face
     `config.json` key aliases (`KeyAlias`), tensor name aliases
     (`TensorAlias`), auxiliary metadata prefixes (`AuxiliaryNamespace`),
     tied-head presence rules (`TensorPresenceRequirement`), and the GGUF
     `<arch>.` prefix in the hand-kept `kGgufSemantic` list
     (`source_mapping.cc`). **Omitting that prefix fails open:** unconsumed
     `<arch>.*` keys become auxiliary metadata with only a warning
     (`normalize.cc`), so a key the math needs is ignored instead of refused.
     A family with bounded side or numeric-list components also sets
     `extended_components`.
   - If `config.json` uses a new `model_type` string, map it to the canonical
     architecture name in the container readers (`src/import/readers/safetensors.cc`,
     `src/import/readers/exl3.cc`, `src/import/readers/mlx.cc`).
3. **Tokenizer and chat template (only if new):**
   - If the checkpoint introduces a new pre-tokenizer regex or normalizer, add
     it in `src/tokenizer/` ([tokenizer.md](tokenizer.md)).
   - If the family needs a native chat template renderer, add it in `src/chat/`
     chosen by template hash or probe equivalence (D-067); otherwise the bounded
     Jinja-subset interpreter renders its `chat_template` automatically.
4. **Admission and reference-fleet pinning (`src/runtime/`):**
   - Unpinned v1 artifacts of any registered mapping are admitted automatically
     by `src/runtime/component_admission.cc` (D-119).
   - Only if pinning a checkpoint into the reference test fleet, add an entry in
     `src/runtime/approvals.cc` and record its frozen spec/artifact IDs in
     `tests/unit/data/v1_frozen/v1-ids.json` (`tests/support/v1_frozen.h`).
5. **Tests:**
   - Add synthesis and mapping validation cases in `tests/unit/spec_test.cc`,
     container normalization tests in `tests/unit/import_normalize_test.cc`,
     admission checks in `tests/unit/component_admission_test.cc`, and a GPU
     import + `EngineLlm` serving smoke test (such as
     `tests/unit/dense_decoder_serving_gpu_test.cc`).

### How to add a component

When an architecture requires a new building block (attention variant, FFN,
recurrence, conditioning block, or embedding):

1. **Specify the component contract (`src/components/`):**
   - Declare the component kind and version in `src/components/defs.h` and
     implement its `ComponentDef` in the corresponding catalog unit
     (`src/components/attention.cc`, `mqa.cc`, `qsa.cc`, `ffn.cc`, `gdn.cc`,
     `moe.cc`, `ple.cc`, `hc.cc`, `inject.cc`, `blocks.cc`,
     `conditioning_blocks.cc`, `spatial_blocks.cc`, or `image_blocks.cc`),
     registered in `src/components/catalog.cc`.
   - Define its typed parameters (`ParamSpec` in `src/components/component.h`
     and validation via `Params::Of` in `src/components/params.cc`),
     weight roles (`WeightRole` with rank/layout constraints), state contract
     (`model::StateContract` in `src/model/state_contract.h`: growing KV, fixed
     ring, recurrent state, verify write/save ranges, and optional `kCommit`
     graph), `model::Batching` capabilities, TP/EP/PP split rules, and default
     planner patterns.
2. **Emit the primitive IR fragment and CPU reference:**
   - Implement the component's graph builder over `ir::Builder` (`src/ir/builder.h`)
     using primitive IR ops (`src/components/step.cc`) and, if stateful across
     accepted speculative rows, its `kCommit` fragment.
   - Implement the scalar CPU reference (`ComponentDef::reference`) so the
     component can be verified independent of GPU kernels.
3. **Tests:**
   - Add catalog and contract cases in `tests/unit/component_catalog_cases.h`
     and `tests/unit/components_test.cc`, and GPU vs. CPU reference parity cases
     in `tests/unit/components_catalog_gpu_test.cc`.

### How to add an IR op or fused kind

When a new component or optimization requires a new primitive IR op or a
planner-selected fused operation:

1. **For a new primitive IR op (`src/ir/` and `src/kernels/reference/`):**
   - Add the `ir::Op` enum value and metadata in `src/ir/op.{h,cc}`, shape and
     type inference in `src/ir/infer.cc`, structural verification in
     `src/ir/verify.cc`, and builder helper in `src/ir/builder.{h,cc}`.
   - If the op is a composite primitive, add its decomposition into foundational
     ops in `src/ir/decompose.cc`; implement its exact scalar CPU execution in
     `src/kernels/reference/exec.cc` (`tests/unit/ir_test.cc`,
     `tests/unit/reference_exec_test.cc`).
2. **For a planner pattern / fused kind (`src/planner/`):**
   - Every fused kind must have a primitive IR equivalent (the registry rule,
     [portability.md](portability.md#the-registry-rule)). Declare the DAG
     pattern in `src/planner/pattern.{h,cc}` and `src/planner/fusions.{h,cc}`
     and wire its selection in `src/planner/select.cc` and `src/planner/plan.cc`.
3. **Lower and register in kernel backends (`src/kernels/`):**
   - Declare capability predicates in `src/kernels/ggml/capabilities.cc` (or
     `src/kernels/decode/`, `src/kernels/exl3/`), register the `Implementation`
     in `implementations.cc`, validate operands in `validate.cc` /
     `validate_ext.cc`, and lower the op or fused pattern into the backend's
     step program in `src/kernels/ggml/lower.cc` (`step_program.cc`).
   - Add planner pattern tests (`tests/unit/planner_test.cc`) and GPU parity
     tests comparing the fused implementation against the primitive IR reference.

### How to add a weight format

When supporting a new quantization encoding, block layout, or multi-part tensor
representation (D-108):

1. **Define the encoding and layout (`src/format/`):**
   - Add the `Encoding` or `Layout` in `src/format/encoding.{h,cc}` and
     `src/format/layout.{h,cc}`, and define part roles, alignment, and exact
     byte-size rules in `src/format/tensor_format.{h,cc}`
     (`tests/unit/format_test.cc`).
2. **Add the CPU reference decoder (`src/kernels/reference/`):**
   - Implement exact block/tensor decoding in `src/kernels/reference/decode.cc`
     (or a companion like `decode_exl3.cc`) so `reference::Execute` can run any
     graph bound to weights in the new format (`tests/unit/reference_decode_test.cc`).
3. **Register device decode and product implementations (`src/kernels/` and `src/execution/`):**
   - Implement the CUDA block decoder under `src/kernels/decode/` (registered in
     `src/kernels/decode/implementations.cc` for the vector, GLU, matrix,
     grouped, and rows templates) and/or specialized backend kernels in
     `src/kernels/ggml/` or `src/kernels/exl3/`.
   - Wire the format into the planner's format selection and rank tables
     (`src/execution/registry.h`, `src/execution/measured_ranks.h`,
     `src/planner/select.cc`) and test GPU decode/product parity against the
     CPU reference (`tests/unit/decode_gpu_test.cc`).

### How to add a container

When teaching the native importer (`llmp import`, D-109) to read a new
checkpoint container or quantization family:

1. **Implement the untrusted container reader (`src/import/readers/`):**
   - Add `src/import/readers/<container>.{h,cc}` (or extend
     `src/import/readers/hf_quant.{h,cc}` for a `safetensors` + `config.json`
     quantization family such as AWQ, GPTQ, or compressed-tensors).
   - Parse headers, metadata, and tensor extents strictly as untrusted input
     (bounded counts, checked offsets, no code execution) into `Checkpoint`
     (`src/import/checkpoint.h`) and dispatch it from `src/import/reader.cc`.
2. **Wire part assembly and relayout (`src/import/`):**
   - If the container splits logical weights across multiple source tensors
     (such as `qweight`, `scales`, `qzeros`, or codebook parts) or needs a
     deterministic import-time transform (transpose, bit-unpacking, scale row/col
     folding, or routed-expert stacking), wire it through
     `src/import/transforms.{h,cc}`, `src/import/expert_relayout.{h,cc}`, or
     `src/import/parameter_folds.{h,cc}` and declare the container's
     `SourceMapping` rules in `src/import/source_mapping.cc` and
     `src/import/normalize.cc`.
3. **Tests:**
   - Add positive normalization and v1 writer tests in
     `tests/unit/import_normalize_test.cc`, `tests/unit/import_writer_test.cc`,
     and `tests/unit/artifact_v1_test.cc`, and adversarial malformed-input tests
     in `tests/unit/import_adversarial_test.cc`.
   - Add a representative seed input (with its `.license` sidecar) to
     `tests/unit/data/import_fuzz/`, wire the reader through
     `benchmarks/import_fuzz.cc`, and verify that `llmp_import_fuzz --corpus`
     and a bounded fuzz run pass.

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
- **Turn-boundary checkpoints**: the stage's state contract (`CheckpointRanges`)
  names future mutable ranges; the runner includes its drafters and
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
  or checkpoint capture as separate completed units (`runtime/llm_prompt.cc`,
  `engine/prefill.h`). This seam applies to all LLMs on `EngineLlm` and
  interleaves prompt units and peer decode without moving a borrowed request
  frame.
- **Deterministic top-k and sparse prefill attention**: planner patterns and
  kernels (`kernels/ggml/`), selected per shape by the plan; the runner's
  plan cache and graphs take them unchanged, and a prefill shape's capture
  rule is the runner's. DeepSeek V4's fast plan uses `llmp.dsv4.lid_topk` and
  `sparse_mask` ([long-context](experiments/long-context/README.md)), and
  Qwen3.8's fast plan uses `llmp.qsa.pool`, `.topk`, and `.attn` (with its
  block keys in a state tensor of the stage's layout and saved by verify).

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

`PromptSession` carries `want_head` through current and predicted complete cache
keys (`StepShape`). Ordinary non-speculative serving selects state-only nonfinal
chunks; low-level plans stay explicitly headed. Plain final blocks stop after
required cache/compressor or convolution/recurrent/QSA writers. DSpark final
features and MTP exported streams require the full trunk, so their checked
head-only variants remain explicit opt-ins. Scoring and checkpoint settlement
preserve required heads, pending rows and carry. Both intent kinds are measured
for Setup funding; pruning does not assume a smaller placement.
[Focused cuts, actual Setup cost and replay](experiments/ds-qwen-state-only/README.md).
