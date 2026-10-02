<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The engine

`src/engine/` is the layer above the kernels that runs models on a paged
node (D-096; [architecture.md](architecture.md#layers-and-dependency-rules)).
It holds the node, one skeleton of shared mechanics every model runner is
built from, and each model family's runner, which adds only what is that
family's own. `jitllm-runtime` serves through it
([runtime-serving.md](runtime-serving.md)) and the paged harnesses drive the
same code under their old names (`benchmarks/engine_names.h`,
`tests/support/paged_node.h`).

## The node

`paged_node.h`: device 0 with its providers, a stream per model and the
copy stream, the storage ring, one catalog domain, the landing zone
(D-081), the scheduler and its lanes, and the workspace the models share
(activations and the GGML pool). A runner registers its memory and page
sources with it and posts its work through it: `Job` (one device job on the
model's stream over a closure), `Load`, `Evict`, `Swap`, `Call` (a function
on the scheduler's thread), and requests: `BeginRequest`/`EndRequest`, or
`WithRequest` around a body, lease a model's closure once for all its steps
(D-093). Its teardown fences each model's stream, evicts the managed
extents, stops the scheduler and only then calls each model's `Release`
(`PagedModel`). `ReleaseMapped` and `PlaceCheck` are the node's memory
helpers every runner uses.

## The skeleton

Composition, not inheritance: a runner holds these as members and calls
them; nothing in them is virtual, and nothing runs per kernel.

| Part | File | What it does | The runner supplies |
| --- | --- | --- | --- |
| Weights | `paged_weights.h` | Opens an artifact and its shards for direct reads; reserves and catalogs, in file order, a 2 MiB-aligned device region per dense group, a slab per layer of routed experts (`LayOutSlab`, `ExpertSlab`) and host regions for groups the CPU reads; registers every page source and span; checks the places still pinned; the resource and expert-array addresses a plan binds | Which group goes where (`GroupPlace`), the slabs and their alignment |
| Live state | `live_state.h` | Stable virtual regions registered before the weights, physical backing only for used extents; sparse unnamed direct-I/O spill files, clear, packed copies, pinned places and quarantine; a verify's snapshot, accept, owed restore and commit, rollback, and undoing a failed verify | The state layout and each step's used ranges; which ranges a verify writes; a commit kernel if kept rows need one |
| Planned shapes | `planned.h` | `PlannedGraph<Graph>`, `PlaceAndPlan` (placeless plan, activation placement, the same plan again), `BindPlanned` (pool scratch checked, implementations bound, D-053), `PlanCache` (per key, variants, capped), `RoomForGraph` (the graph cap, D-090), `CheckCoverage` (BP-A1) | The graph builder and its binding (`*_plan.h`), the cache key, the tensor classes for the coverage check |
| Runs and graphs | `graph_runs.h` | `GraphRuns`: stages a run's inputs in the pinned staging, queues copies, work between inputs and plan, the plan and the outputs; captures a shape on its second run, replays its graph from then on with the staging checked, falls back to launch by launch on a refused capture; `GraphStats`, `RunPath` | When a run may be captured (decode steps, verifies, drafts) and what it copies out |
| Resources | `runner_resources.h` | The runner's own device memory (pinned), pinned staging, cuBLAS and its workspace, a measuring launch context, the launch context over the pool and the registry, and their completion-aware release (AGENTS.md rule 6) | Sizes and names |

`support.h` holds the small helpers (errors, addresses, rounding, seconds,
joined problems).

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
2. **Register** (after the node's Start): each weight's source
   (`PagedWeights::Register`), the state's write-back places
   (`LiveState::RegisterSpill`), then every managed extent's place pinned
   (`Scheduler::PinPlaces`, D-090).
3. **Bind** (after the workspace): the closures (`everything`, the state's
   fence, any narrower ones), the model's places as the plan's address
   functions, the commit hook, and the launch context
   (`RunnerResources::BindLaunch`, then `GraphRuns::SetLaunch`).
4. **Steps**: each chunk, draft or verify checks `LiveState::Usable` and
   `AwaitingAccept`, finds or plans its shape (`PlanCache::Find`, else plan,
   `BindPlanned`, `CheckCoverage`, `Add`), decides whether to capture
   (`PlanRuns::CaptureDue` plus the model's rule, then `RoomForGraph`), and
   materializes its used ranges (`LiveState::Use`) and renews the request
   closure if new extents were initialized, then posts one job that queues what the live state owes (`QueueOwed`), a
   verify's saves (`QueueSaves`), and the run (`GraphRuns::Queue`). A
   failure settles the state (`LiveState::Settle`: a verify undone, else a
   quarantine); success counts the path (`Count`).
5. **Release** (after the node's teardown): the plans and graphs first
   (they name the launch context and the memory), then
   `RunnerResources::Release`, `LiveState::Release`, each
   `PagedWeights::Release`.

## Independent request state

`Qwen38Runner` exposes four stable request slots. Each owns its target and MTP
live state, verify snapshot, commit state, pending cursor and plan caches. The
weights, launch context, stream, staging and workspace remain shared. The scalar
runner methods use slot zero. `SelectSlots` selects an execution closure over the
active slots at a completed unit boundary; the runner's aggregate state and swap
closure include every initialized slot, including idle retained conversations.
The graph cap applies across all slots. Serving provisions two concurrent
execution slots separately from these four stable state slots, with conservative
activation, scratch, staging and host-input bounds checked before work.

`qwen38_wave_plan.h` composes fresh per-slot plans without rewriting scalar
caches. Consecutive compatible slots (up to all four) form a group whose
target/draft MXFP8 and routed vector products join at up to sixteen rows,
charging concatenation and retaining independent output views; the wide
MXFP8 and expert-major routed kernels give each request's products bit for
bit. Compatible two-to-four-row HC BF16 products of a target group share
their original cuBLAS path, retaining separate preparation and nonlinear
mixing. Eligible three- or four-row BF16 target heads join through the
ordinary MMF selector (up to sixteen columns); unsupported heads stay
original. Stateful operations and draft heads stay independent. A
multi-slot wave reads attention cells aligned to 2,048, and backs each
slot's caches through them, so its graph key repeats. Each decode step's
own state reservation already backs through that alignment, so running out
of state capacity refuses that request rather than failing the shared wave
(which would stop the node). `TargetWave` and `DraftWave` validate placement
and bindings before dispatch, then publish outputs only after the completed job.
Each successful verify retains its own snapshot until acceptance or explicit
discard restores that branch's prior target and MTP state.

Clearing or restoring one destination first renews the request's protection of
its peers, then discards only the destination's eligible backing and renews the
selected closure. A separately held destination refuses discard. A proven local
failure while discarding the destination invalidates that slot; clean validation
refusals preserve the existing state. An unproven device or shared-execution failure
stops the cohort and preserves borrowed owners until retirement is established.
The serving adapter forwards each host branch to its corresponding native slot.
The production Qwen chat backend drives prompt and generation units through the
shared driver seams; other families and literal completions retain scalar entry
points. A native fence proves retirement before any borrowed owner is released.

## Adding a model family

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
  `PlanCache` per kind of plan, with Setup, Register and Bind as above and
  the family's steps. The DeepSeek and Qwen3.8 runners are the worked
  examples: DeepSeek with a host table, a chained draft-and-verify job and
  a snapshot of every written range; Qwen3.8 with rows read on demand and
  gathered between the inputs and the plan, a commit kernel, and two
  variants of a verify's plan.
- `runtime/serving.cc`: a `Served`/`Llm` adapter that forwards to the
  runner (its tokenizer and template, chunks, speculative step, its
  `CheckPlaces`), and the architecture name that selects it.

Outside the engine, a family may also need its import to a prepared
artifact (`docs/experiments/artifact-layout/import_m3.py` today), its
tokenizer's pre-tokenizer if it is new ([tokenizer.md](tokenizer.md)) and
its chat renderer (`chat/`, by template hash, D-067).

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
  discarded tail pages and their saved bytes. Model footprint functions
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
  ring, `model/dsv4.h` `Dsv4Window`; `jitllm.dsv4.lid_topk` and
  `sparse_mask`, [long-context](experiments/long-context/README.md)), and
  so does Qwen3.8's fast graph (`jitllm.qsa.pool`, `.topk`, `.attn`; its
  block keys a state tensor of the model's layout, their verify saves the
  runner's).
