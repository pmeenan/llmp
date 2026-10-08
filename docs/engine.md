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
| Weights | `paged_weights.h` | Opens an artifact and its shards for direct reads; reserves and catalogs, in file order, a 2 MiB-aligned device region per dense group, a slab per layer of routed experts (`LayOutSlab`, `ExpertSlab`) and host regions for groups the CPU reads; registers every page source and span; checks the places still pinned; the resource and expert-array addresses a plan binds | Which group goes where (`GroupPlace`), the slabs and their alignment |
| Live state | `live_state.h` | Stable virtual regions registered before the weights, physical backing only for used extents; successful source/pin checks reused under scheduler lifetime/epoch with mutation invalidation (not residency or leases); sparse unnamed direct-I/O spill files, clear, packed copies, pinned places and quarantine; a verify's snapshot, accept, owed restore and commit, rollback, and undoing a failed verify | The state layout and each step's used ranges; which ranges a verify writes; a commit kernel if kept rows need one |
| Planned shapes | `planned.h` | `PlannedGraph<Graph>`, `SizedArena` (a plan's arena holds what its graph uses), `PlaceAndPlan` (placeless plan, activation placement, the same plan again), `BindPlanned` (pool scratch checked, implementations bound, D-053), `PlanCache` (per key, variants, no fixed number, each plan's host bytes and planning time kept: `PlannedHostBytes`), `PlanAccount` (what the plans and graphs hold, charged to the node: a plan as it is added, a graph before its capture, D-090 as amended), `PlanStep` (a step's plans spared by any reclaim while it runs), `CollectPlans` and `ReclaimPlan` (the plans and graphs as candidates for the node's reclaim order, and one's reclaim), `CheckCoverage` (BP-A1) | The graph builder and its binding (`*_plan.h`), the cache key, the tensor classes for the coverage check |
| Runs and graphs | `graph_runs.h` | `GraphRuns`: stages a run's inputs in the pinned staging, queues copies, work between inputs and plan, the plan and the outputs; captures a shape on its second run (or when the runner asks), beside that run's launch-by-launch execution (so capture and instantiation overlap the device's work) unless work is queued between inputs and plan, replays its graph from then on with the staging checked, falls back to launch by launch on a refused capture; `CaptureAhead` records another planned shape's graph without running it, from the input layout `Layout` predicts; `GraphStats`, `RunPath` | When a run may be captured (decode steps, verifies, drafts, hinted prefill) and what it copies out |
| Prefill lookahead | `prefill_lookahead.h` | `PrefillLookahead` and fixed-capacity `PrefillLookaheadGroup`: independent optional host allowances before CPU-only future plans, sequential builds during current device work, ordered installation after successful completion, exact cache-charge transfer and destruction before unused-grant release | Shape prediction, graph construction, binding/coverage/cache installation and capture policy |
| Resources | `runner_resources.h` | The runner's own device memory (pinned), pinned staging, cuBLAS and its workspace, a measuring launch context, the launch context over the pool and the registry, and their completion-aware release (AGENTS.md rule 6) | Sizes and names |
| Request cohort | `request_cohort.h` | Several request slots of one model: the active set selected between completed units (several only under one held stream request), the closures (everything, the state fence, the execution closure, each slot's fence) and the held request's refresh, and the cohort's fault (every slot quarantined until retirement) | Its slots' live states and the shared extents |

`support.h` holds the small helpers (errors, addresses, rounding, seconds,
joined problems).

Gemma2, Gemma3 and both Gemma4 profiles use the shared prefill lookahead lifecycle.
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
future has independent refusal and cleanup; no future
state or cursor is prepared. DeepSeek and Qwen
adapters remain open transfers. `PromptSession::NextPrefillHint` supplies the same
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
2. **Register** (after the node's Start): each weight's source
   (`PagedWeights::Register`), the state's write-back places
   (`LiveState::RegisterSpill`), then every managed extent's place pinned
   (`Scheduler::PinPlaces`, D-090).
3. **Bind** (after the workspace): the closures (`everything`, the state's
   fence, any narrower ones), the model's places as the plan's address
   functions, the commit hook, and the launch context
   (`RunnerResources::BindLaunch`, then `GraphRuns::SetLaunch`).
4. **Steps**: each chunk, draft or verify holds a `PlanStep` for its life,
   checks `LiveState::Usable` and
   `AwaitingAccept`, finds or plans its shape (`PlanCache::Find`, else plan,
   `BindPlanned`, `CheckCoverage`, `Add` with its `PlannedHostBytes` and
   planning time, charged as it is added), decides whether to capture
   (`PlanRuns::CaptureDue` plus the model's rule, then `ChargeGraph` for
   every capture of the job: a graph with no room even after a reclaim is
   not made), and
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

`Dsv4Runner` has the same slots (`Dsv4Options::wave_slots`, up to 16; one,
the harnesses' default, provisions slot zero alone), each with its own target state,
DSpark ring, verify snapshot, output staging and plans, its cohort rules from
`request_cohort.h`. Its waves are not composed from per-slot plans: the graph
builder takes several slots' chunk shapes (`dsv4_graph.h` `Dsv4WaveGraph`) and
builds one fast-plan graph whose row-local operations (every quantized product,
routed experts read once per distinct expert, HC mixes, routing, norms, the
head) run once over every slot's rows, while each slot's compressors, indexer,
attention, cache writes and DSpark injection run on its own state. A wave holds
at most 16 rows and keeps each request's rows bit-identical to its steps alone:
`jitllm.vecq` is count-invariant from 2 to 16 tokens, float products past GGML's
8-column vector kernel run over groups of whole slots that fit its eight
columns, GGML's products of quantized
weights `jitllm.vecq` has no kernel for (the dense products, quantized HC
mixes and the attention output's grouped product) run per slot, a wave of one-row steps takes the vector
product's one-token configuration (`SetVecQOneToken`: each token's sums its
step's; dense products four tokens a pass), and the drafter's injection
(GGML MMVQ) runs per slot. A wave needs every layer in the fused form
(`Dsv4WaveSupport`: each layer's expert products `jitllm.vecq` types; the HC
mixing weights may be any type). Setup checks it before provisioning slots;
an artifact that fails it gets one slot and a `serial_reason()` for the
start's log, so it is served one request at a time rather than refused.
`DecodeWave` runs one step of each slot (injected
beside a drafter); `DraftVerifyWave` each slot's draft block and one joined
verify, whose rows await each slot's `Accept`. A family whose graph builder can
take per-slot segments gets waves this way at the cost of its stateful
operations alone ([report](experiments/deepseek-batching/README.md)).

## Adding a model family

Start from the [optimization inventory's transfer table](optimization-inventory.md#transfers-and-gaps):
a new runner adopts each technique there whose contract it meets, or the
table records why it cannot (workflow.md's transfer rule), before the
family's speed is compared with its reference. An existing kernel's narrow
selector does not rule out a near-fit consumer: investigate a compatible
extension with qualified arithmetic, bounds, funding and performance, or
retain an explicit current-milestone open item. This applies to older
families too; incompatible math or absent redundant work remains a concrete
non-applicability reason. Shared mechanisms (graph
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

Gemma3 separately [prepares fresh zero backing](experiments/gemma-state-prepare-ahead/README.md)
for a known next chunk before the current Job. A LiveState-owned no-victim
acquisition ticket is funded and authenticated before submission; the host-only
meanwhile contract stays unchanged. Future backing does not enter used ranges,
advance cursors or acquire write-back authority. After the current fence, a
scoped existing-page-in drain proves retirement before collecting unused zeros
as reclaimable backing. Actual Use alone publishes initialized ranges. Clear,
source changes and teardown drain first; unknown completion retains the complete
owner, and the Runtime Server fails stop before member destruction if teardown
cannot retire it. The Gemma3 default selects eligible fresh zero sources; kept
restart sources and fresh sparse-file reads retain their current path. Other
growing adapters and larger-context qualification remain open.

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
  the family's settings. Its runner's context ceiling
  (`RunnerContextCeiling`); the metadata its defaults derive from (the
  trained context, the drafter's block, sampling defaults: read by
  `ArtifactFactsOf`, never a checkpoint's name or hash); each setting of
  its own a key in `ModelKeys` naming the architectures that use it, its
  fallback a constant in `model_settings.h` with what measured it, its
  resolution in `ResolveSettings` and its line in `ModelSettings::Lines`
  (the unit test checks every key is listed). A measured speed trade
  becomes a calibration once the machine can measure it.

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
  ring, `model/dsv4.h` `Dsv4Window`; `jitllm.dsv4.lid_topk` and
  `sparse_mask`, [long-context](experiments/long-context/README.md)), and
  so does Qwen3.8's fast graph (`jitllm.qsa.pool`, `.topk`, `.attn`; its
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
