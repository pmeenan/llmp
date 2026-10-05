<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Coarse native Gemma26 critical-path diagnosis

Diagnostic implementation protocol. No build, model or trace has run yet.
The isolated measurement base is: d08eea75fac191bdb0e6ad6d6535ea1363d71f23.
The preceding20-file profile and its14 acquisition sources remain immutable.

Entry2026-10-05T16:34:17.696059Z freshly resolves primary TensorFold609ca419 /
0.6.5; four unchanged primary identities in tensorfold-entry.json. Gemma26 still
MLX-only there. CPU-sampling Stage0 is an actual completed unavailable-capability
finding: nsys status reports paranoid4/perf_event_open andsampling-trigger Fail.
Failed official metadata1/no-model proof870efeee is retained separately. Do not
change sysctl/security/permissions, enable system-wide sampling or substitute
OSRT wait unions for CPU time.

## Question

Locate elapsed critical-path phases in one unchanged26/all1024/C1 8K prefill.
Existing diagnosis has1.361091553s native wall without recorded GPU activity;
VMM API-body sum48.765856ms, launcher bodies811.335744ms are overlapping and
cannot be subtracted to identify CPUcost. Previous untraced native+57.6433%
latency remains the competitive result. The new run is not a speed comparison.

## Actual source boundaries (current d08 source, no proposed math change)

- benchmarks/gemma_prefill.cc113–139: WithRequest already holds the model lease,
  sixwarm+clear then eight1024-rowChunk calls paid; last_built_policy lookup is
  paid. Job normally reuses the held channel, but state-growth closure changes can
  end the prior lease and re-admit through RefreshRequest; this conditional
  lease replacement is part of paid work, not omitted admission.
- engine/gemma4_runner.cc932–1075: Chunk delegatesWave. Wave validates tuples,
  CheckPlaces, CheckFactors, ChargeHost/input descriptors; ReserveStateThrough;
  Planned; Gemma4Sources/Stage; node.Job; completed host publication/cursor.
- CheckPlaces544–561: ONE node.Call wraps weight/state-place walks. That call
  body executes on scheduler thread, while callerPost/Await waits. Existing
  moved-place refusal and all walks remain intact.
- ReserveStateThrough593–616 uses LiveState::Use. live_state.cc164–269 performs
  fresh-range validation/deduplication, node.Call closure description,
  node.Acquire initialization, then node.Call initialized write-back
  registration. RefreshClosures also follows actualfresh/partial work. Its node.Call rebuilds
  closures on scheduler; RequestCohort::Hold then calls RefreshRequest, which
  ends/reopens the held request when execution.extents differ. Add separate
  refresh.call ends immediately after node.Call returns; refresh.body is the
  same existing scheduler closure, and cohort.hold is a sibling caller scope
  inside state.grow. Keep their inclusive elapsed/charged CPU ownership
  separate, with no
  generic request instrumentation or assumption of re-admission on every call.
- Planned893–930: plan-cache lookup; on miss PlanGemma4Chunk, BindPlanned,
  policycounts, state-tensor construction, CheckCoverage, plan-cache insertion.
  Coverage does NOT post a node.Call per tensor: PagedNode::Covered819 reads
  sorted span/catalog descriptions synchronously on caller. Keep this source
  fact distinct from earlier speculation about scheduler roundtrips.
- gemma4_plan.cc111–180 builds measured-sized arena/graph, binds weights and
  keeps; calls PlaceAndPlan. planned.cc51–102 runs firstPlanGraph, activation
  placement, actual address/view binding, secondPlanGraph and SamePlan.
  BindPlanned105–140 plans scratch and binds implementation registry.
- graph_plan.cc147/896: eligibility/reader scans and activation lifetime placement
  are synchronous caller work; no kernel launch. Do not alter the algorithms.
- GraphRuns::Stage44–59 copies host sources into pre-funded pinned staging.
  Queue61–140 runs on device-submission worker, submits input copies/bound
  plan/outputD2H, or captures/instantiates/replays using unchanged gates.
- PagedNode::Job1101/Step1250 uses held RequestChannel and waits completed steps;
  Timed1120 wraps the existing job on submission worker. Its Timing.started/
  queued and Note record dispatch/job/after/device totals but are not emitted
  by this helper. The new scopes can bracket the same boundaries without
  changing marks, waits, Note or hang/completion behavior.
- PagedNode::Call1385 posts CallProgram; CallProgram::Advance(programs.h150)
  invokes the body on scheduler thread. CallerPost1019→Await951 includes queue,
  scheduling, body, report and20us sleep/wakeup overhead. These are elapsed
  boundaries, not busy CPU samples.
- RunnerResources maps/binds/pins at setup, outside paidprefill. Do NOT instrument
  its setup as the explanation of the paid interval. State initialization's
  node.Acquire can issue materialization/backing/copies through scheduler;
  pagein.cc179+ and services.cc333/629 verify the separate backing/submission
  lanes. First screen need not instrument each extent/command or schedulerTurn.

## Small scope set

Each paidChunk gets a bounded ordinal0..7, not prompt/token/KV values. Keep the
existing outer paid NVTX. Add fixed labels, coarse once-per-call/miss scopes:

| Caller scope | Exact body | Nested or cross-thread witness |
| --- | --- | --- |
| wave | entire Wave through HostGrant destruction | contains all caller phases |
| places.call | existing CheckPlaces node.Call | places.body inside SAME closure on scheduler |
| inputs | host charge + Gemma4Chunk/descriptors/frontier assembly | state phase separately nested |
| state.grow | ReserveStateThrough/Use + required RefreshClosures | state.describe/body and state.register/body inside existing Call closures; state.acquire around existingAcquire |
| plan.miss | actual cache miss from build through cacheAdd | PlanGemma4Chunk outer, plan.first, placement, plan.second, plan.bind, plan.coverage |
| stage | Gemma4Sources + GraphRuns::Stage | host descriptors/memcpy only, no invented devicecopycost |
| job.completed | node.Job call until checked result | job.submit around existing Queue invocation in SAME Wave job closure on submission thread |
| publish | completedlogits assign/cursor/feature metadata | existing HostGrant destruction tracked separately if not covered by final callerphase |

Cache hits get a tiny fixed plan.hit label/count, never treated as rebuild time.
The graph/buildbind remainder is the PlanGemma4Chunk outer interval excluding
PlaceAndPlan children, accurately labelled remaining graph/validation/binding;
no per-node/per-layer range. Plan.first/second bracket
actual calls in PlaceAndPlan, placement brackets PlaceActivations; BindPlanned
and CheckCoverage each once per miss. No ranges in numerical kernels, hot loops,
perextentcatalogchecks, completion polling or condition wait loops.

Use fixed-label publicNVTX+bounded integer payloadordinal; no hot-loop logging,
per-event host result allocation, callbacks that retain graphnodes, graph
producer/alias changes or CUDAcommand changes. The printed old metrics/timers
and snapshots remain byte-for-byte semantic contracts. No OSRT CPU attribution.

## Isolated implementation and charged-thread-clock proposal

Use a reviewed reversible instrumentation-only patch stored with the benchmark
experiment, applied ONLY to the isolated measurement tree. Main/defaultsource
and its object archives remain untouched. Patch three source units only:
engine/gemma4_runner.cc, planned.cc and live_state.cc. Do NOT instrument generic
PagedNode/scheduler/provider/kernel objects. PlanGemma4Chunk's outer scope is
at the runnercall site; PlaceAndPlan scopes are in planned.cc. CheckPlaces and
state Call bodies are annotated in their existing closures. Wave's existing
Queue callback carries job.submit on the actualsubmissionthread; PagedNode::Timed
and all submission/wait/patience/fence code remain unchanged.

The manual helper additionally reads existing PagedNode::TakeTimes(0) only
before/after the whole paid span, clearing priorwarm metrics before the
original startclock and publishing aggregate dispatch/job/after/device outside
the original elapsedclock/NVTX. This only resets independent diagnostic
accumulators; it does not reset predictor/request/channel state. There is no
public runnerplan-time getter; new planhit/miss counts come only from bounded
coarsescopes, not a production API. This existing elapsed summary crosschecks
new scopes without changing core Timed/Note instrumentation.

Each fixed coarse scope may also read CLOCK_THREAD_CPUTIME_ID and steadyclock
on ENTRY/EXIT of its actualexecutingthread and add count/elapsed/chargedCPU to
fixed bounded static diagnostic counters (e.g <=32phase slots, <=4KiB known
observer metadata, no per-event heap/string/log allocation). No kernel/node
loop sampling. Readouts occur only after paid span/completedjobs. No CPUclock
subtraction across different threads. Parent+child chargedCPU counts overlap
by nesting: publish inclusive and explicitly disjoint/exclusive figures only
where matching parent-child thread ownership supports subtraction. Across
threads, CPUcharged sums are not criticalpathwall. ThreadCPU includes scheduled
userspace/kernel work of that thread, not exact instruction/stack attribution;
elapsed minus threadCPU includes descheduling/waits, not proven blockedstate.

Before any build/model, root source review receives completepatch/header/helper/
CMake/protocol and reverse-patch reconstruction hashes of ALLthree original
sources. No operation/kernel/graph/check/memory-policy change. Only that
isolated native diagnosticbinary is built; do not hand patchedsource/objects
back to a production warmtree. Source manifest explicitly admits these three
instrumentation deltas; never claim all production-source bytes unchanged.
Storedpatch and reversed-originalsource maps make originalmath reconstructible.
PublicNVTX/clock glue is benchmark-only external interface, not a shipped core
dependency. Previousprofile immutable14-source frame is separately preserved.

The process-local paidflag/ordinal is snapshotted by VALUE before an existing
scheduler/device closure is posted. Scope metadata contains no borrowed input,
address or label string. Fixedone-active-nodeC1 scope is closed admission.
Existing CallProgram/Job borrow lifetime remains through done.gone/completedstep;
no new retained pointer/state/queue field. NVTX push/pop and CPUclockpairs remain
on actualexecutingthread. No worker metadata crosses into nextchunk or teardown.
The observation-only static counters/NVTX foreign runtime occupancy are reported
separately from native managed budget; do not present their known bytes aspeak.
If bounded counter ownership/retirement fails, stop instead of merging events.

A tiny model-free public-interface test must first establish both monotonic
threadCPU clock support and knownbusy-versus-condwait behavior, returning actual
elapsed/CPU deltas and balanced worker NVTX contexts. No perfprivileges needed;
no source/frame-pointer/security change. Knownbusy and knownwait are measured
separately on SAME worker, joined before successfulatomiccompletion. Clockread
failure marks the gate failed; no guessed CPUdata. Freeze exactvalidated source,
clocks/NVTX identities and flags before approved control/model.

## Acquisition gate and fixedwork

Exactly ONE annotated native control, then ONE native trace only after control
receipt authenticates Task40/the committed profile output. No original arm.
Each installedGPU600/stop-on-fail, expected10–30s plus export. Root must approve
control before trace. Source freezes include alloverlays+canonicalparents/
mathdependency manifests/buildreceipts and publicNVTX/header/bin identities.
Nsight flags remain samplingnone/contextswitchnone, CUDA/NVTX/OSRT with graphnode
activity; no sampledCPU/security changes. Model-free publicruntimeNVTX already
proven usable; any new worker-range payload uses a tiny no-model selfcontrol.

Same26artifact4ddb/indexe748/IDs6b656/rawf2c28/C1/context16384/max_rows1024/
local2048global16384/F16, sixwarm+clear,8192rows,3anchors,32INCOMINGheadargmaxes,
8completeheadpublications. Native policycounts60/90/30/30/allothers0. Keep all
math options, frontierhead, inputmasks, residency/staging/graph gates, closure
checks, originaltimer boundaries and exactcompletion-aware teardown. The
annotatedcontrol and trace must reproduce completeprefill/finalheads, all32
choices, full592445440B initializedstate/layout and actualcapacity/policycounts.
Atomic child exit0 after proven resource retirement and profiler+supervisor
successful finalrecords/actualbusyfree required separately(RE039).
Any fidelity/retirement/source/worker-context failure stops interpretation.

## Predicted distinguishable results and analysis limits

1. If plan.miss dominates caller GPUinactive time, split actualgraph/buildbind,
   first/second eligibility scans, placement, scratch/registrybinding and catalog
   coverage. Only their measured elapsed scopes establish a planning-phase lead.
2. If places.call is long but schedulerplaces.body short, waiting/dispatch/report
   dominates that call; ifbody is long, sourceplace walk is the elapsed lead.
   ThreadCPU deltas can distinguish chargedCPU from elapsed descheduling/wait;
   they still provide no sampled call-stack or exactblocked-state attribution.
3. If state.acquire spans VMM/copies plus a longGPUinactive tail, identify elapsed
   materialization/wait region. Description/registration worker bodies versus
   enclosing Call spans separate actualclosurework from transfer/report waits.
   No diskread attribution or exactstatepage cause without further witnesses.
4. Match callerjob.completed to submissionworkerjob.submit by capturedordinal.
   Callerstart→workerstart is dispatchelapsed; workerend→callerreturn includes
   GPUcompletion/schedulerreport/wakeup. ActualCUDAactivity union intersections
   further separate knownGPUoverlap; don't label allaftersubmit asGPUduration.
5. If stage/input/publication or unattributed caller gaps dominate, report that
   boundary specifically. If all coarsephases remain small/unknown, do not invent
   a criticalpath. No head suppression/CheckPlaces removal solely from a trace.

Analyze Nsight's common timestamp domain, authenticate modelPID and actualTIDs
with childreceipt/PROCESSES/NVTX; collect thread-owned ranges, payloads and all
balanced pairs. Clip toouterpaidrange and report perchunkcounts/critical-window
unions plus inclusive/exclusive nested durations. NEVERsum parent+child or
callerwait+parallelworker intervals twice. Across8sequentialchunks, matchordinals
and reject missing/duplicateworker associations before role claims. Steadyclock
old metrics remain separate from profilertime; avoid cross-domain subtraction.
ChargedthreadCPU is directly measured if the no-model clock gate passes;
blocked/ready state and sampledstack claims remain unavailable. Coarse scopes
identify elapsed operations/chargedCPU and concurrent waits, not scheduler state. Raw events/timestamps/threadIDs/
KV/heads/logs external; Git onlyaggregates/provenance/scope counts/limits.
No competitive/adoption/fullquality/memorypeak/othermodel result. This is a
bounded location diagnosis needed before any proposed production optimization.

Verified eligibility source lead (not measured attribution):
graph_plan.cc80–110 OnlyReader walks the whole graph plus source/view roots for
eligible norm chains; gemma_moe_fusion.cc73–104 Private walks wholegraph/keep and
repeatedly evaluates bounded RootOf/internal/elided predicates for route/reduce.
PlaceAndPlan performs PlanGraph twice per miss. Eight growing global_n_kv prefill
widths can therefore repeat these scans. n_past authenticates builds but is
not itself a plan-cache dimension. First/second PlanGraph coarsescopes
already isolate this combined eligibility cost; NO matcherloop/routine/node
annotations, eligibility weakening or cachedreader implementation is proposed.
A large measured phase would justify a later source-led experiment; similarity
to dense31 norm scans does not establish its causal contribution or timing.

## Source and observer reconstruction

The canonical three engine units remain unchanged in the author/integration
tree. apply_overlay.py authenticates the exact original or instrumented hashes
before applying/reversing instrumentation.patch in an explicitly supplied
measurement tree. overlay-identities.json records both versions and patch SHA.
The CMake option is false by default and refuses missing overlay markers; only
the two new manual targets are built in the isolated patched tree. The engine
archive there intentionally contains diagnostic calls resolved by the ONE
coarse_scopes.cc singleton linked into the manual helper. Do not build other
executables against that diagnostic archive or transfer its objects to main.

validate.py prepare uses the exact base Git objects to prove canonical src,
cmake, source-lock/toolchain and test-support inputs unchanged, and records the
three explicit measured deltas plus complete helper/script source frame. The
remote source/result gates re-read that entire measured map, fixed IDs,
artifact manifest/index, public NVTX tools/headers and Task40 baseline receipts.
No historical model-payload rehash is required. Source receipts include the
actual build receipt, two new binaries, atomic child helper and completed
no-model probe identities. Result gates reauthenticate the same environment
and successful supervisor records before trusting outputs.

The singleton owns four fixed thread slots and 28 fixed counters per owner,
less than 4KiB known static observer data; existing node Call/Job completion
boundaries precede counter readout. CLOCK_MONOTONIC is the diagnostic elapsed
clock; CLOCK_THREAD_CPUTIME_ID belongs to the executing thread. Public NVTX
uses its profiler time domain. Their raw timestamps are never subtracted.
Per-thread CPU aggregates are inclusive; no parent/child CPU sum is presented
as total CPU. Same-thread exclusive NVTX elapsed subtracts only the union of
recorded nested scopes on that thread; this does not remove observer cost.

The probe records busy and timed condition wait on one joined worker, with
balanced unsigned ordinal-zero NVTX ranges. analyze.py probe validates actual
SQLite PROCESSES/NVTX ownership against the checked child PID, worker/counter
TID agreement and positive charged CPU versus elapsed behavior. Runtime proof
is required before compiled-source freeze and any model. Analysis validates
all eight ordinals and caller/scheduler/submission associations, counts and
noncrossing same-thread nesting, rather than guessing kernel or CPU roles.
Unknown GPU table ownership remains explicit. Detailed timing/TID/event rows
stay external; only bounded aggregate evidence belongs in Git.

The analyzer enforces the exact successful source roster for each ordinal:
wave, places.call/body, inputs/state.grow, one hit or miss, stage, completed
job/submission, publish and host.release. Misses additionally require all
graph/build-bind, first-plan, placement, second-plan, registry-bind and coverage
scopes on the caller with declared parents/order; hits forbid those scopes.
Scheduler bodies and submission scopes must use their consistent distinct
source roles, not merely fit within a caller interval. Conditional fresh work
may be wholly absent. In this closed successful path, a fresh Use(true) causes
description/acquisition/registration and then refresh/hold, so all eight
witnesses are required when any appears. Failed/partial work cannot pass the
model fidelity/retirement gate and is never interpreted as a successful trace.
Publication's later declaration ends that scope before HostGrant destruction;
require publish.end <= host.release.start and both within wave. Synthetic tiny
controls exercise the actual summarizer with counters regenerated after each
omission/thread/parent mutation, rather than relying on counter/event mismatch
to reject missing source witnesses. They are parser controls, not measurements.
