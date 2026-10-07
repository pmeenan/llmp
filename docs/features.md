<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Feature matrix

The scope ledger, first triaged in the M0 planning conversations:

- **Confirmed** — stated project scope from the owner's design brief
  ([ideation.md](ideation.md)) and subsequent owner-approved decisions and
  review fixes. Milestone assignment happens in [plan.md](plan.md) as the plan
  firms up.
- **Proposed** — candidate additions awaiting triage by the project owner.
  This covers both approaches the brief itself labelled "proposed design" and
  agent suggestions (marked *agent-suggested*).
- **Deferred** — a candidate parked with a concrete revisit trigger and an
  earliest milestone. It does not block M0 or become accepted automatically
  when the trigger occurs. Confirmed scope can also have a deferred delivery
  date or trigger without being removed from scope.
- **Rejected** — a candidate the owner declined at triage; the row keeps the
  dated reason so it is not re-proposed unknowingly.
- **Open questions** — things that shape architecture and need an answer
  during M0 or an explicit deadline before dependent work begins.

Status legend: `confirmed` · `proposed` · `deferred` · `open` · `rejected`

Triage may confirm, reject, or defer a proposal. Rejection and deferral need a
brief reason; only load-bearing decisions need a D-NNN entry. Record a
deferred item's trigger and earliest milestone here or in plan.md.

A confirmed feature is scope, not a support promise: model support is earned
per checkpoint and configuration and tracked in the
[support matrix](model-support.md) (see "Testing and evidence").

The owner walked this matrix on 2026-09-21 (the M0 feature triage). Every
status below reflects that pass; a dated note in a row records the call. A
proposal added later starts as `proposed` again.

## Model switching and sessions

The primary workload (D-019): one user, or an agent plus subagents, switching
among a library of models larger than memory, with conversations that last
hours. State is retained within D-024's bounds, with safe recomputation from
client-supplied history when an idle cache entry is unavailable.

| Feature | Status | Notes |
| --- | --- | --- |
| Model library larger than memory, loaded on first request, evicted across the whole library | confirmed | D-019 |
| Time-sliced (TDMA-style) execution under memory contention; concurrent execution when a supported placement fits the working sets and complete execution envelopes within each node's budget | confirmed | D-019, D-020; aggregate pool capacity alone is insufficient; switch points are quiescent boundaries; concurrency-when-it-fits is required soon after v0 |
| Configurable switching policy: pause a running request only at a completed phase boundary; inputs are request class, per-model/alias configuration, estimated switch cost and deadlines | confirmed | Owner 2026-09-23, D-069, after an external architecture review. Default until M6 measures: priority-aware (same class runs to completion; interactive pauses background after a minimum run; a pre-admission 429 when a known deadline cannot be met); run-to-completion and time-sliced are configurable alternatives |
| Single front door: clients talk to the conductor, the cluster's one point of entry, which routes each request to a node running that model | confirmed | D-020, D-022, D-023 |
| Cluster topology discovered or configured at runtime; no node names, counts, or roles baked into the application | confirmed | D-023/D-038/D-039; M6a classifies single/pair/triangle/switched-N QSFP layouts and scans attached dedicated subnets for bootstrap candidates, then enrolls configured membership and one conductor; automatic membership changes and election remain deferred |
| Multiple replicas of the same model across nodes when concurrent demand on a small model justifies it; the conductor balances across replicas with session or prefix affinity | confirmed | D-023; delivery deferred until measured overlapping demand justifies replication after M6a; affinity follows compatible retained state |
| Conversation state preserved across a model switch (KV, compressed, or recurrent state kept resident, spilled to SSD, or reconstructed) | confirmed | D-019, D-024; M3 spills and restores it across a full swap; M6 proves retained-state reuse and correct fallback separately; idle cache retention is bounded |
| Switch-latency metrics: A-to-B first token, B-to-A resume with state, bytes moved per switch, and whether the concurrent or time-sliced path was taken | confirmed | D-019 headline metrics; reported per the comparison protocol |
| Compatible prefix reuse for standard clients; optional explicit session ID and release for cooperating clients | confirmed | D-024, D-031: shared system-prompt prefixes are cached independently of conversation continuations, with separate reuse/expiry policies under shared node bounds. Prefix identity establishes neither a conversation nor its lifetime; branches have isolated mutable state. D-055's [retention policy](retention-policy.md) sets identity, boundaries and refresh-by-branch rules |
| Bounded cache-memory, spill and metadata use; expiry and safe recomputation when reusable state is unavailable | confirmed | D-024; active/suspended admitted requests remain protected. D-055: capacity-driven with 24-hour per-class idle caps (owner decision); capacity values pinned at M5 exit from measured state sizes |
| Incoming-model prefetch at switch time (load B's non-expert weights and hot experts while A finishes its step) | deferred | *agent-suggested.* Revisit in M9 after M6's measured timeline shows overlap opportunity and spare capacity; request identity is known, timing is speculative |
| Client warm hints as an optional extension (an orchestrator announces a model it is about to use; optional pin or priority) | confirmed | D-022; ignorable by standard clients; M10 with the session extension |
| Conductor places models, or parts of models, per node and routes requests; placement preferred over paging when it suffices | confirmed | D-020; whole-model placement is M6a, independently of M7; sharding starts in M4 (TP2, with a minimal ready-before-collective rule), with coordinated admission in M8. The subagent's model on another node while the main model stays resident needs only the network |

## Runtime core: catalog, reservations, leases

| Feature | Status | Notes |
| --- | --- | --- |
| One native execution process per node managing all local models | confirmed | D-005 |
| Node-wide resource catalog: typed IDs, generations, dependency closure, shared extents charged once | confirmed | D-006; descriptor field groups in ideation §4 |
| Explicit CUDA VMM backing (reserve / create / map / access / unmap via the driver API) | confirmed | D-006 |
| Capacity reservations separate from residency leases; request transaction vs execution lease | confirmed | D-007; D-050 settles guaranteed bounded requests, retained-state/growth and complete phase envelopes; execution proof remains M2 |
| Lazy commitment: grants never eagerly evict useful cache | confirmed | D-007 |
| Separate commitment and occupancy ledgers | confirmed | D-007 |
| Completion service tracking GPU, I/O, and network consumers before reclaim | confirmed | ideation §3, §14 |
| Resumable continuations: suspend a model phase while I/O is pending and run other ready work | confirmed | ideation §7; D-050 bounds suspended phases and retains their full envelope during waits; another model phase needs a successful concurrent-envelope check |
| Per-class lifecycle policies (immutable weights, routed experts, dense/attention weights, sparse lookup tables, live KV/state, reusable prefix state, scratch, graph objects, comm buffers, staging) | confirmed | ideation §5 table is the initial policy set. Qwen3.8-Flash-Next's 51B n-gram embedding is the first concrete sparse lookup: row requests resolve to their containing 2 MiB chunks initially (D-035, D-056) |
| Architecture-specific adapters for live and reusable state (KV blocks, compressed attention, sliding window, recurrent) | confirmed | conservative semantics per architecture first; D-068 adds declared capabilities (truncation, snapshots, transient working state) |
| Model-shape extension points: composed model contexts, finite request programs of build-time phase kinds, state capabilities, decoding modes | confirmed | owner direction 2026-09-23, D-068: the resource core names no architecture; new shapes extend these contracts and pass the D-050/D-055 matrices. [Shape inventory](architecture.md#model-shapes) |
| Prefix-cache metadata always consistent with physical eviction (no stale hits) | confirmed | pager invariant 4 |
| Deterministic simulated (fake) resource backend for tests | confirmed | Stage 1 deliverable in ideation §19 |
| Conceptual native API: `register_resource` / `reserve_capacity` / `acquire_group` / `submit` / `retire_completed` / `reclaim` / `cancel` | confirmed | 2026-09-21: the starting shape (ideation §20), not a frozen interface. D-048 settles explicit native task states and completion ownership; the M2 backend proof settles the internal contract |
| Physical-backing pool to amortize allocation overhead | confirmed | D-033 baseline retains useful contents and hands compatible backing to admitted replacements; no per-read release/create requirement. D-035 records the owner's larger mapped-slab alternative (including 1 GiB) for comparison in the M2 paging proof. Physical capacity, suballocation, and I/O size stay distinct; all held backing is charged |
| Turn/step-scoped leases with eviction only at scheduler-established completion boundaries as the v1 lease model | confirmed | *agent-suggested*, confirmed 2026-09-21; D-050 records the full question 9 policy and adversarial cases. Owner 2026-09-22: leases may end at step boundaries, but the node switches requests/models only at client-facing request/response boundaries (D-069, 2026-09-23, makes this a configurable policy that pauses only at completed phase boundaries; by default same-class requests still run to completion) (concurrent only when both fit). A switch request alone establishes no quiescence; consumers must complete and suspended live state stays protected. Backend phase bounds need M2 evidence; MoE within-step misses get their progress proof in M7 |
| Core free of vendor types; device memory, paging, and transfer operations behind narrow provider interfaces, CUDA VMM the first and only implementation | confirmed | D-026; only where it adds no complexity or penalty on NVIDIA |
| Ledger keyed by memory domain (one domain on unified-memory platforms) so a discrete-GPU platform is a data difference, not a redesign | confirmed | *agent-suggested*, confirmed 2026-09-21: a domain field on the ledger, which passes D-026's zero-cost rule |

## Eviction and retention policy

| Feature | Status | Notes |
| --- | --- | --- |
| Global cross-model victim selection at extent granularity | confirmed | D-008; `shrink(model, N)` may exist as a convenience, not as the boundary |
| Release changes eligibility, not residency; hysteresis, minimum useful residency, reload-cost awareness | confirmed | ideation §9 |
| Real use tracked separately from prefetch and cancelled planned use | confirmed | |
| Per-model statistics combined with global comparison so no model monopolizes reclaimable bytes | confirmed | fairness |
| Compare global LRU vs frequency/recency vs cost-aware heuristic on identical recorded traces | confirmed | report miss bytes, reload repetition, write volume, waiting time — not hit count; M9, on natively recorded traces |
| Routing/access trace capture and offline replay simulator | confirmed | reference-engine feasibility spike in M0/early M1 before M2; native recording and validation in M7; ideation §7, §9 |
| Cost-aware eviction heuristic (`eviction_cost_per_reclaimed_byte`) | confirmed | 2026-09-21: as one candidate in the trace-replay comparison above, alongside global LRU and frequency/recency; it becomes the default only if replay shows fewer miss bytes and reloads. Ideation §9 calls it a first heuristic, not a measured predictor |
| Brief wait for an imminent completion instead of evicting expensive contents | deferred | 2026-09-21. Earliest after M6; trigger: traces show expensive eligible extents evicted shortly before other work completed and freed enough capacity, or made cheaper extents reclaimable, to satisfy the pending request. Extents with unfinished consumers remain protected. Ideation §9 says "consider" |
| Trace-learned expert co-occurrence prefetch | deferred | *agent-suggested.* Revisit in M9 after M7 traces show predictable misses and an overlap window; compare against demand-only execution including unused prefetch bytes |
| Residency warm-start across runtime restarts (persist heat/working-set metadata, re-warm on start) | deferred | *agent-suggested*, deferred 2026-09-21. Earliest M9; trigger: measured cold paging after a restart is a material share of the switch budget on the primary workload. Metadata only, never live state. If revisited, the live-state alternative has a precedent: vLLM 0.30.0's Fast Start (below). For jitLLM that would mean exporting VMM handles to a small holder process, whose support on GB10 is unmeasured |
| Dependency-group value functions for eviction scoring | deferred | *agent-suggested.* Revisit after the M6 baseline when trace replay demonstrates avoidable reload cost caused by ignoring dependency coupling; preserve extent-level reclamation and lifetime rules |

## Storage and I/O

| Feature | Status | Notes |
| --- | --- | --- |
| Common internal read/write completion interface across backends | confirmed | ideation §8 |
| Spark direct path: file DMA → bounded host-VMM landing zone → GPU copy into device VMM (and reversed for write-back) | confirmed | D-081 amends D-034's in-place consumption: the GB10's L2 does not cache host-located memory (RE-022); no CPU payload copies |
| Bounded, explicitly budgeted staging pool where a validated DMA fallback needs one | confirmed | on Spark this is D-081's landing zone (2 × depth × 2 MiB); never allocate unbudgeted RAM in order to evict RAM |
| Explicit handling of short transfers, checksum errors, storage exhaustion, alignment, retries, cancellation; bounded queues | confirmed | |
| Coalesce duplicate loads for the same content generation | confirmed | |
| Native direct-file I/O backend (queue depth, priority, cancellation control) | confirmed | D-034 selects bounded asynchronous direct I/O on validated Spark configurations; D-081 lands it in host VMM and copies into device VMM |
| cuFile compatibility-mode backend | confirmed | measured M0 comparison path; D-034 does not select it for the initial runtime |
| Buffered vs direct-I/O comparison; page-cache duplication and read amplification measured | confirmed | no system-wide cache flushing as runtime policy |
| Write-back only when preservation requires it; clean weights are never written | confirmed | |
| Native GDS backend on hardware where the direct path is supported | deferred | 2026-09-21. Earliest M9 (untriggered deferrals are reviewed at M9 planning, plan.md); trigger: a supported target with native GDS appears (D-026); never Spark |
| Remote extent transfer between nodes (logical object/version/extent over a supported transport) | deferred | 2026-09-21. Earliest after M8; trigger: measured direct-link bandwidth beats local NVMe read and a placed or sharded workload reloads data a peer already holds. Source stays leased until completion; never remote `cuMemMap` |
| Optional crash durability for spill as a separate policy | confirmed | 2026-09-21 deferred; landed 2026-10-03 under D-105 at the owner's direction (D-102's hang recovery): a conversation wholly on disk survives a restart of the same build, crash or graceful stop, in named owner-only files with a hashed, versioned record, adopted only when everything validates; `[memory] keep_across_restart = false` keeps nothing past the process. Resident state is still lost in a crash, and an upgrade starts cold (runtime-serving.md) |
| Spill encryption at rest | rejected | *agent-suggested*, rejected 2026-09-21. Nodes are single-owner and local (multi-tenant isolation is a stated non-goal); D-014's "protect spill files" is met by a restricted directory owned by the non-root service user, bounded retention, and cleanup on expiry and on start. Restore-path CPU on unified memory would compete with the model |
| Direct I/O (`O_DIRECT`) as the default payload read path on unified memory | confirmed | measured and selected in D-034; buffered metadata remains allowed, but payload paths must not silently introduce CPU copies; import must honor queried direct-I/O alignment |
| GPU in-place access to memory read straight from NVMe | rejected | confirmed 2026-09-21 (D-034), rejected 2026-09-27 by D-081. M0's scan matched device VMM, but kernels that re-read lost 1.1–4.9× (BP-F1: the GB10's L2 skips host-located memory), and no route lands a direct read in device memory on Spark (RE-025); D-081 copies each extent instead |
| Sustained-read thermal behaviour and a spill-write budget for the single NVMe | confirmed | *agent-suggested*, confirmed 2026-09-21 as I/O spike scope. Measure sustained throughput over minutes, not seconds; reads do not wear the drive but KV spill writes do |

## Model import and prepared artifacts

| Feature | Status | Notes |
| --- | --- | --- |
| Owned import pipeline: validate → select layout → pack/shard → index → hash → atomic publish | confirmed | D-009 |
| Prepared per-model paging artifact aligned with VMM backing; direct weight DMA | confirmed | D-035, amended by D-056: making a model available includes repacking into contiguous dependency groups (a dense layer, one expert's closure), 4 KiB-aligned on disk and paged in 2 MiB group-relative chunks through coalesced, vectored direct reads; small tensors are packed, experts/tensors indexed, and page-in does no CPU payload repacking. Sparse rows fetch their containing chunks; mutable spill stays separate |
| Versioned, hashed artifacts supporting bounded range reads without reprocessing | confirmed | D-009, D-018; experimental initially, with explicit rejection of incompatible versions; compatibility guarantees follow dense and MoE restore evidence |
| Checkpoints treated as untrusted input; no code execution; lengths, paths, hashes, metadata validated | confirmed | D-009 |
| Resumable import; free-space and peak-temp checks; interrupted imports never appear valid | confirmed | |
| Workstation-side import; target-assisted tuning as an explicit mode with separately keyed results | confirmed | x86 importer, ARM importer, and architecture-independent artifact format are distinct things |
| Immutable model files separate from mutable spill files | confirmed | |
| Artifact contents: manifest, tokenizer/config, representation catalog, resource index, immutable data, integrity/provenance, optional plan metadata | confirmed | 2026-09-21: the required content set (ideation §11); encoded by D-056's [v0 format](artifact-format.md). Plan metadata is not yet part of v0 |
| Multiple alternative backend layouts of the same resource per artifact | deferred | 2026-09-21. Earliest M9; trigger: measurement justifies storing alternatives' disk/import cost. D-052 already requires representation-aware descriptors and separate GGML/EXL3 prepared artifacts in M2; this deferral does not postpone EXL3 support |
| Reuse a known container for the immutable blobs (GGUF- or safetensors-style aligned tensor data) and own only the manifest and resource index | confirmed | *agent-suggested*, confirmed 2026-09-21 as the principle; D-056 chooses safetensors shards (2026-09-22), with GGUF kept only as the carrier of GGUF sources' metadata. Re-packing experts into contiguous aligned extents is justified; inventing a container is not. Keeps tooling available while D-018 keeps the format experimental |
| Standalone artifact verification tool (checksums, index bounds, manifest consistency) | confirmed | *agent-suggested*, confirmed 2026-09-21; delivered with the M5 importer. Cheap given per-chunk checksums (D-056); separates "bad artifact" from "pager bug" during bring-up |
| Model support matrix per checkpoint: unsupported → import-only → resident-correct → paged-correct → distributed-correct → performance-validated | confirmed | ideation §19; kept from M3, started there as [model-support.md](model-support.md) |
| Direct model download from the Hugging Face Hub in the importer and the management API | confirmed | owner request 2026-09-20; downloads are resumable and verified like any import input (D-009); an M5 import job |
| Hugging Face token from a `.env` or config file tied to the user's HF account, also settable from the web management UI | confirmed | owner request 2026-09-20. Secret handling: never logged, restricted file permissions, `.env` git-ignored, management stays local by default (D-014). Gated repositories download only with a token whose account already has access; the tool cannot grant it. Config/`.env` in M5; the web UI with M10's dashboard |
| Installed artifacts only on node-local storage that passes the direct-I/O probe; the runtime never pages from network or long-term storage | confirmed | owner request 2026-09-22, D-054; anything but a local block-device filesystem passing the probe (network, FUSE, memory-backed) is rejected at startup, never served through a slower path; installed files are opened without following links or crossing mounts |
| Optional long-term store (network mount such as a NAS, or a USB-attached drive) for source checkpoints and an optional prepared-artifact archive | confirmed | owner request 2026-09-22, D-054; absent by default. A plain filesystem path mounted by the OS; jitLLM ships no network-filesystem client. Only import/install jobs touch it; content-addressed, no symlinks, verified as untrusted input; never holds credentials, spill or conversation state |
| One import per cluster: one node pulls and imports, peers with a compatible profile receive the verified prepared artifact over the cluster link | confirmed | owner direction 2026-09-22, D-054; a management-plane transfer between enrolled nodes, distinct from the deferred remote extent transfer; needs M6a enrollment. Naive single-stream copy measured 1.05 GB/s over the DAC versus 117–118 MB/s per NAS client ([inventory](environment.md#long-term-model-store-2026-09-22)) |
| No implicit removal at install time: when a node lacks space, the user explicitly picks installed models to archive or delete before the install proceeds | confirmed | owner decision 2026-09-22, D-054; otherwise the install fails before transferring, reporting required space and candidates. The check leaves the spill budget and filesystem headroom free, and jobs allocate the space before writing. Archive verifies its copy before local removal; removal quiesces users and their I/O before space counts as free |

## Compute backends and execution

| Feature | Status | Notes |
| --- | --- | --- |
| Reuse of kernels, algorithms, and model semantics from vLLM, llama.cpp/GGML, ExLlamaV3/EXL3, FlashInfer, CUTLASS/CuTe under their licenses | confirmed | D-013; D-051 records the first dense slice's GGML/Qwen2 source-unit inventory and adoption gates; other models still need their own selection/audit |
| Keep a fully resident fused plan and a pageable split plan where both are useful | confirmed | ideation §7, §10 |
| KV-cache and output-head compression modes: lower-precision caches (FP8 as vLLM runs Qwen3.8), compression algorithms such as [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/), a quantized output head (and token table), and FP8 tensor-core linears for Qwen-Image's DiT, as per-model quality/performance modes | deferred | Owner, 2026-09-28: queued as future work, since several algorithms and modes are worth exploring. The output head joined it the same day: Qwen3.8's is BF16, 248,320 × 2,560 (about 1.27 GB), read every step (about 5.2 ms of a 37 ms step, measured), and Mia's checkpoint keeps it BF16 too, so quantizing it would be a lever past the reference, changing the logits. Qwen-Image's FP8 DiT linears joined it too (owner, 2026-09-28), as the image's quality/performance lever; the default image path takes only quality-preserving speedups. API shape (owner, 2026-09-28): each reduced mode is served as a model alias beside the full-quality default (for example `qwen3.8-flash` and `qwen3.8-flash:fast`), so unmodified clients pick it by model name and the response's `model` reports the mode that ran; the image endpoint also takes OpenAI Images' `quality` (`low`/`medium`/`high`). The default is always full quality; a reduced mode runs only when named. It is the context-length term of memory: Qwen3.8's F16 cache is ~24 KiB a token (about 6 GiB at 262K, computed), and it also sets long-context decode bandwidth and a swap's spill size. Recurrent state (Gated DeltaNet) stays F32. Needs long-context quality checks (perplexity on a long text, retrieval) per model and mode; llama.cpp's `q8_0` K cache is wrong for DeepSeek V4 (upstream #24162). Approximate sampling (drawing from a fixed top-N candidate pool, or an approximate top-p cutoff) would be a lever of the same kind; the default sampler draws from the exact distribution (tokenizer.md#sampling, 2026-09-29) |
| Lossless layout transforms separated from quantization/precision changes (the latter need explicit quality evaluation and metadata) | confirmed | |
| CUDA graphs with dynamic residency decisions outside captured segments; no CUDA API calls from host-function nodes | confirmed | initial approach; ideation §7 |
| One deliberately managed CUDA context per GPU; explicit streams and library handles | confirmed | initial |
| Backend operation contract (declares architectures, layouts, quantization, state, workspace, dependencies, graph restrictions, completion) | confirmed | 2026-09-21 starting shape (ideation §10); D-052 requires both real GGML and EXL3 M2 proofs before settling the contract. D-053 (2026-09-22): jitLLM owns dispatch; several implementations of an operation, from any compatible source or our own, coexist as build-time modules and are selected per operation, architecture and shape; implementation identity is part of plan and cache identity |
| Early backend integration proof using jitLLM-owned memory, explicit workspace and completion, then eviction and restoration | confirmed | D-051 FP16 control plus D-052 real small EXL3 quants in M2: complete packed dependencies, reference logits, restore/cancellation and upstream kernel performance on Spark; informs M3 without freezing a plugin ABI. [Scope, oracles and cases](backend-proof.md) recorded 2026-09-22 |
| Versioned C ABI for optional separately built backends | rejected | rejected 2026-09-21 (D-028). Optional implementation modules (D-017) are build-time modules behind the operation contract; the removable boundary is a build-profile property, not a runtime plugin ABI. Ideation §14 |
| Triton AOT as an optional build-time kernel route | deferred | 2026-09-21. Earliest M9 (alternative kernels and plans); trigger: a needed kernel exists only as Triton source. Keep provenance of generated code (ideation §10) |
| GPU-visible residency table plus compact miss notification | deferred | Earliest M9, after M7 demonstrates material host-boundary overhead; a Boolean check without protection against revocation is unsafe |
| Executing ready experts while other experts load | deferred | 2026-09-21. Earliest M9, after the M7 baseline; trigger: M7 traces show partial-availability windows that would hide material stall time. Not an assumed capability |
| Speculative decoding with MTP: layers stored in the checkpoint (Ornith, Qwen3.8, MiMo) and companion drafters (Gemma 4 assistants) | confirmed | Owner confirmed 2026-09-23 (D-068), replacing the 2026-09-21 deferral: designed now (M2's contract must express draft, verify and rollback phases and composed contexts), executed in M3 and M4 for their models (Qwen3.8's stored MTP and DeepSeek V4's DSpark drafter in M3; GLM-5.3's and DeepSeek v4.1's drafters in M4, D-087), M9 for the rest. Benchmarks still require matched decoding features plus the reference's normal configuration; Qwen3.8's matched comparison must state whether MTP is used |
| Block-diffusion text generation (canvas denoising over a causal prefix; DiffusionGemma) | confirmed | Owner confirmed 2026-09-23 (D-068): designed now, executed in M9. A bounded DiffusionGemma reference study measures per-step expert closures before M9 planning |
| Optimistic MoE execution: device-visible residency table, kernels flag a miss, restart from the missed layer | deferred | *agent-suggested.* Earliest M9, after the pessimistic M7 baseline is correct and measured host-boundary cost warrants it; must prove safe revocation, replay of mutable state, and progress when a step's leases fill memory |
| GGML/GGUF first with a required early EXL3 companion, both using jitLLM-owned backing | confirmed | D-028/D-051 retain the FP16 control; D-053 runs GGML's kernels under jitLLM dispatch, not GGML's backend runtime; owner requested early EXL3 on 2026-09-22 (D-052). Real packed EXL3 execution and kernel performance are M2 gates; flagship EXL3 serving and full swaps M4, paged switching/restore M6. [Fixtures and contract](exl3-bringup.md). Tokenizer/model semantics need their own audited units; other GGML backends preserve D-026's portability boundary |

## Two-node execution

| Feature | Status | Notes |
| --- | --- | --- |
| Conductor placement and request routing across nodes, ahead of sharding | confirmed | D-020, D-023, D-037/D-038; M4's minimal configured conductor loads its TP2 shards first; M6a detects paths on the existing network and enrolls members after M6, independently of M7 |
| Explicit sharding across nodes for the flagship; the conductor issues distributed phase IDs | confirmed | M4 (TP2 with NCCL); coordinated admission and asymmetric-failure handling M8; ideation §12; needs the direct link |
| Prepare/commit admission across ranks for sharded models; no rank enters a collective while another can wait indefinitely on an unapproved allocation | confirmed | placement-only execution does not need it |
| Separately budgeted, stable communication-buffer pool honouring NCCL registration and threading contracts | confirmed | |
| Ordered collective submission; completion fences for GPU and network consumers | confirmed | |
| Port the validated target recipe's parallelism first (TP, PP, EP are different plans) | confirmed | |
| Remote paging (see Storage) | deferred | 2026-09-21; same trigger and earliest milestone (after M8) as remote extent transfer under Storage |
| Master-decides admission with a mirrored replica ledger instead of full prepare/commit for the two-node case | deferred | *agent-suggested*, deferred 2026-09-21 to M8 scoping, when the two-node admission design is written; prepare/commit stays the general design if a third node ever appears |

## Management, inference API, and diagnostics

| Feature | Status | Notes |
| --- | --- | --- |
| CLI / status endpoint and structured events first | confirmed | ideation §13; a dashboard is not a prerequisite for validating the pager |
| Versioned local management API: import/list/remove, representation inspection, priorities, residency policies, cancellation, node health, trace capture | confirmed | budget changes are scheduler requests; removal quiesces users. D-064 (2026-09-23): anonymous on loopback, with browser guards (Host, Origin/`Sec-Fetch-Site`, JSON-only mutations, no CORS) |
| Explainable decisions: memory breakdown, model working sets, request state, eviction decisions (victims, expected cost, bytes recovered, why alternatives were kept), I/O timeline, backend choice | confirmed | ideation §13 views; explainable scheduling is a stated priority |
| Local-only binding by default; auth and transport protection for remote; no prompt/KV logging by default; opaque request IDs | confirmed | D-014 |
| Front-door TLS per served name from certificate files kept current by external tools: certbot (snap) with its Cloudflare DNS plugin through a jitLLM deploy hook, a packaged `tailscale cert` timer for the node's `ts.net` name, or user-supplied files; built-in name-constrained CA as fallback | confirmed | Owner request 2026-09-23, D-065; certbot chosen over an in-house ACME client the same day. jitLLM holds no Cloudflare token and manages no DNS records. M5 ships files, the local CA, reload, the certbot hook and the Tailscale timer |
| Public internet exposure of the front door (for Cursor, whose requests come from Cursor's servers) | deferred | 2026-09-23, D-065: none of the TLS sources makes a node publicly reachable; Tailscale Funnel (ports 443/8443/10000, bandwidth limits) or port forwarding would be a separate security decision. Trigger: the owner wants Cursor against a node |
| Streaming inference API compatible with standard web-API clients (Cursor, OpenCode, Codex, Claude Code named) | confirmed | D-022, D-030; D-040 documentation baseline and D-045 front-door contract; executed per-client compatibility remains M5 work |
| Web dashboard as a separate process over the management API | confirmed | M10 (Stage 6); must not own the scheduler or take runtime locks; calls management from its own server side, never from the browser (D-064) |
| OpenAI-compatible HTTP surface (chat completions at minimum) | confirmed | D-040: Chat Completions plus Responses, JSON/SSE and tools; model listing and Messages token counting included |
| Anthropic Messages API format alongside the OpenAI-compatible surface | confirmed | *agent-suggested*, confirmed 2026-09-21 (D-030): Claude Code is a named client and speaks it, so the format ships in the M5 baseline surface. Athena's Engine offers both flavours, mild evidence that Spark users expect it |
| Admission "explain / what-if" query (why can't this request be admitted now; what would need to be evicted) | confirmed | *agent-suggested*, confirmed 2026-09-21 for M6's basic status and diagnostics. Natural extension of explainability and a debugging tool for progress-envelope bugs |
| Trace export in Perfetto / Chrome trace-event format for the I/O timeline and scheduling | confirmed | *agent-suggested*, confirmed 2026-09-21 for M6's basic status and diagnostics. Structured events are confirmed; a standard viewer format avoids building a timeline UI early |
| Target capability probe tool (VMM granularity, GDS mode, RDMA availability, driver/toolkit versions, glibc/ABI) | confirmed | *agent-suggested*, confirmed 2026-09-21; first cut landed in M1 as `jitllm doctor` (D-072). Platform properties are probed capabilities, not constants (D-026) |
| Per-model memory quota and priority policy (minimum guarantee, maximum share) | deferred | *agent-suggested*, deferred 2026-09-21. Earliest after M6; trigger: fairness statistics show a model starved or the flagship displaced by small models on a named workload. Fairness itself is confirmed under Eviction |
| Model version hot-swap (publish a new artifact version, drain the old, no runtime restart) | deferred | *agent-suggested*, deferred 2026-09-21. Earliest after M6; trigger: re-importing a model in use forces a restart during a long conversation. Removal-with-quiesce is confirmed; this is the add-then-drain composition |

## Toolchain, build, and development environment

| Feature | Status | Notes |
| --- | --- | --- |
| C++23 host runtime, Clang-first; NVCC with Clang host compiler where validated; pinned libstdc++ initially | confirmed | D-010; D-060 links a source-built GCC 16.2 libstdc++/libgcc and cudart statically; only glibc, the NVIDIA driver libraries and rdma-core stay dynamic, plus cuBLAS (with the system `libgcc_s`), which D-076 links dynamically and the package will ship once a binary uses it |
| Cross-compile from x86-64 to Spark; deploy and test over SSH; explicit targets only | confirmed | D-011 |
| `mise.toml` + `mise.lock` for tool setup, environment, and tasks | confirmed | D-012 |
| Project-owned SDK provisioning; native Ubuntu and a reference dev container from the same logic | confirmed | D-012 |
| Locked CMake source acquisition with curated vendoring for adapted units | confirmed | D-057: audited source closure selected before acquisition, hash-verified local inputs, offline configure/build, profile exclusion and receipts; separate from the D-049 SDK. M1 implemented it ([third_party/](../third_party/README.md), `mise run prepare`, the build receipt) with GoogleTest 1.18.0 as the first pin; M2's first adapted source, GGML, came as a narrowed archive with patches instead of vendored units (D-077) |
| Copyleft-components-disabled check profile with audited dependency closure | confirmed | D-002, D-017; excludes optional implementation dependencies and records declared tools/platform runtimes separately. Runs in D-061's local `check:full` tier; no hosted CI yet |
| Optional implementation modules/plugins selectable at build time; incorporated core implementation uses Apache-2.0 / BSD / MIT / MPL-2.0 | confirmed | D-017; default build may use declared platform dependencies under their actual terms; classification never waives license obligations |
| Reference container pinned by digest; target driver recorded separately from toolkit and library versions | confirmed | ideation §16 |
| Core builds and its tests pass in a CPU-only configuration with no vendor SDK present | confirmed | ideation §16, D-026; the portability guardrail and the fake backend's home |
| CMake presets + Ninja + `compile_commands.json`; LLD where validated; pinned LLVM format/analysis tools | confirmed | 2026-09-21; D-032 pins validated compilers/LLD; D-058 pins CMake 4.4.3 with both-host FetchContent checks and native/cross/CUDA smoke. D-059 pins Ninja 1.13.2, GoogleTest 1.18.0, clang-format/clang-tidy/clangd/llvm-symbolizer 22.1.8 and the candidate style, check and warning sets. Application build validation is still owed (ideation §14) |
| File set: `toolchains/manifest.toml`, `toolchains/artifacts.lock.json`, `tools/setup-toolchain`, `tools/check-toolchain`, `cmake/toolchains/`, `CMakePresets.json`, `.devcontainer/` | confirmed | 2026-09-21 as M1 scope, minus the `dev` script (next row); ideation §16 |
| `setup / doctor / build / test / deploy` contributor entry point as mise tasks | confirmed | 2026-09-21: D-012 already makes mise the task runner, so these are mise tasks rather than a separate `./dev` script (ideation §16). M1 added `prepare` for D-057's sources, which `setup` also runs |
| REUSE-style file-level SPDX identifiers; NOTICE file; SBOM | confirmed | 2026-09-21 (D-029): copyright/license metadata with REUSE lint plus a separate embedded-header check for commentable source and docs, and a NOTICE file from M1; uncommentable files use sidecars or REUSE.toml. The SBOM lands with `.deb` packaging and is tied to the license profile (ideation §17). D-071 (M1): REUSE lint 6.2.0 and the header check in `check`, sidecars only (no REUSE.toml), `NOTICE`, and provenance records for the toolchain; the SBOM comes with the Package item |
| Benchmark trace replay with external captured inputs and regression thresholds in the check gate | confirmed | Owner's 2026-09-21 output policy: feasibility/M7 captured traces stay outside Git and are fetched or supplied by verified hash for replay. Keep harnesses, input identities, aggregate baselines, and regression thresholds in Git; set thresholds after measurement. Replay still catches policy regressions without Spark time for every change. The gate is local until hosted CI is adopted (D-061) |
| Native-on-Spark CMake preset kept as a fallback and diagnostic build alongside the cross build | confirmed | *agent-suggested*, confirmed 2026-09-21 as part of the toolchain smoke ([M0 record](m0-record.md)). The owner reaffirmed cross-compiling as primary (D-011): the C++ side is routine and the toolchain is needed anyway. The one piece with real uncertainty is NVCC with a cross Clang host compiler; a native preset costs little and keeps the first token unblocked if that drags |

## Release and project surface

| Feature | Status | Notes |
| --- | --- | --- |
| Shipped notices and source availability match the actual build configuration (core vs. enabled optional modules) | confirmed | ideation §17, D-017; includes any shipped platform components; a build that self-reports its license profile is the obvious mechanism |
| Versioned releases with a changelog and a compatibility policy for the artifact format and management API | confirmed | *agent-suggested*, confirmed 2026-09-21. D-062 (2026-09-23): SemVer 0.x, owner-signed tags, independent integer versions per public surface, Keep a Changelog `CHANGELOG.md` from M1; artifact compatibility guarantees additionally require D-018's dense and MoE evidence |
| Contribution policy: external PRs accepted with DCO sign-off; no CLA | confirmed | 2026-09-21 (D-029); merges still pass the human commit gate (D-016). External PR code never runs locally; the first external PR triggers hosted CI (D-061). Flagged in ideation §21 alongside the license |
| User installation through native package managers: a project-hosted, signed apt repository with arm64 packages for Spark first | confirmed | D-027; the user path, distinct from the developer setup path (D-012); M10 |
| Optional copyleft modules as separate packages in a separate repository component, mirroring D-017's tiers | confirmed | *agent-suggested*, confirmed 2026-09-21. Amended 2026-09-27 (D-080): jitLLM's own builds and packages include confirmed-copyleft modules by default, and a build-time flag excludes them; whether a module also gets its own package is M10's choice |
| systemd unit, non-root service user, FHS layout (config under `/etc`, state and artifacts under a configurable data directory), drain-before-restart upgrades | confirmed | *agent-suggested* consequences of D-027, confirmed 2026-09-21; recorded in D-063 (2026-09-23): `jitllm` system user, `jitllm.service`, a strict TOML node document at `/etc/jitllm/jitllm.toml` plus `jitllm.d/` fragments, data roles under `/var/lib/jitllm` ([layout](architecture.md#installed-layout)). Directories are created by the role's user and split by mode (0755 models, 1777 checkpoints, 0700 spill and state); defaults are front door `127.0.0.1:8114` and management `127.0.0.1:8115` (jitLLM's own, clear of other engines' defaults). Unit, user and layout M1; drain-before-restart upgrades with M10's apt repository |
| The check gate builds installable `.deb` packages from M1, before the repository is published | confirmed | *agent-suggested*, confirmed 2026-09-21. Late packaging is where notices, paths, and dependencies go wrong. D-061 moves this to the local `check:full` tier (arm64 package in D-063's layout), with its install test in an arm64 container on the workstation |
| An OCI container image built from the release `.deb`, with a published seccomp profile that allows io_uring and documented run flags (memlock, GPU through NVIDIA's container toolkit, storage-role bind mounts) | confirmed | Owner request 2026-09-29, D-098: secondary to the `.deb`, which stays the supported and measured configuration; serves only (no jobs without a delegated cgroup). Docker's default seccomp profile blocks io_uring since 25.0. M10 |
| Homebrew or `.pkg` (macOS), winget or MSI (Windows) | deferred | D-098: with their ports ([portability.md](portability.md)); follow their platforms (D-026, D-027) |
| Supported linkable library (public C++ headers or a C ABI) | rejected | Owner 2026-09-23, D-064: jitLLM is a service; the native API is internal. Public surfaces are D-062's list |
| Inventory which MiaAI-Lab files are actually AGPL versus MIT ExLlamaV3 upstream before designing the optional-module boundary; document AGPL's network clause for a served process | confirmed | *agent-suggested*, confirmed 2026-09-21; done in M0 ([record](m0-record.md), [licensing.md](licensing.md)). The AGPL exposure may be a small glue and patch set |

## Testing and evidence

| Feature | Status | Notes |
| --- | --- | --- |
| Layered tests: native CPU, deterministic simulated backend, single-Spark CUDA, model semantics, multi-model pressure, two-node, packaging/licensing | confirmed | ideation §18 |
| Mandatory pager invariants (eight, listed in [architecture.md](architecture.md)) enforced by tests | confirmed | |
| Numerical references against pinned known-working engines with documented tolerances; teacher-forced logits and intermediates, not generated text | confirmed | |
| Benchmark set: warm decode rate, inter-token latency distribution, bytes read/written per token, exposed stalls, peak occupancy, mixed-model throughput, per-model waiting/fairness, cold and partial-resume latency, switch and switch-back latency with bytes moved per switch (D-019) | confirmed | full provenance (artifact, backend, toolchain, driver, hardware, policy) with every result |
| Resident-hit path measured independently from the miss path; cold storage vs warm OS cache vs warm residency separated | confirmed | |
| Canonical A→B→A through an unmodified client, with resident reuse, forced spill/restore, and bounded-cache fallback cases | confirmed | M6's product gate, after M3's full-swap A→B→A; report elapsed time, bytes read/written, reused/recomputed prompt tokens, and numerical checks. D-055 names the Qwen2.5-0.5B FP16/EXL3 workload, arms and statistics ([acceptance workload](retention-policy.md#m6-acceptance-workload)) |
| Paging feasibility assessed before M2 against the full-swap floor; matched-configuration and normal reference-configuration comparisons | confirmed | [performance evidence](architecture.md#performance-evidence); D-021/D-025: first-cut estimates, then a measured reference A→B→A once setup runs; spike sizes M6/M7, M3/M4 validate full swaps, M6 validates switching and M7/M9 validate paging/optimizations |
| Athena's Engine as a closed-source comparator for the normal-reference view (46 s full swap and 2.1 s context restore on one GB10, creator-reported) | confirmed | *agent-suggested*, confirmed 2026-09-21 as an optional comparator install (M0 reference experiment, [record](m0-record.md)). Nothing reusable; numbers include speculative decoding; install on a Spark only if its terms allow and only as a comparator. The 46 s includes checkpointing the active session; the pair does not both fit in 128 GB |
| TensorFold as a benchmark target and source of optimization ideas (MIT; single-stream, exact speculative decoding on DGX Spark; creator-reported 1.6–3× vLLM-with-MTP decode) | confirmed | Owner-suggested 2026-09-26; [assessment](tensorfold-assessment.md) of `d7470ed`; confirmed as an M3 baseline and comparator by D-087 and run in M3 ([record](m3-record.md)). Scope: a pinned reference engine for overlapping models (Qwen3.8 Flash Next), normal and matched configurations, M3 and M9; reproduce its Spark numbers first. MLX 4-bit only (a like-for-like comparison needs that import format or an explicit cross-quant report). M3 and M4 report it as cross-quant; importing and optimizing its format is an M9 item, which then makes it a gated same-format comparator (D-087). Ideas mapped to the operation contract, D-056 layouts, BP-F4, M3, M4 and M9; its reported GB10 page-migration slowdowns deserve one measurement here |
| vLLM 0.30.0 Fast Start as a switching comparator (a per-GPU daemon holds post-quantized, sharded weights resident; restarted engines map them over CUDA IPC, `--load-format ipc_cache`; release notes report engine init 28.9 s to 8.2 s on H200) | proposed | Owner-noted 2026-09-26 ([release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0), [PR 54921](https://github.com/vllm-project/vllm/pull/54921)). Whole models held live in separate processes: no paging, eviction or cross-model policy, so it does not serve a library larger than memory. Proposed: include it in M6's switching comparisons as vLLM's best case whenever the models fit, after checking that it works on GB10 (unmeasured) |

## API additions and triage (2026-09-22)

Owner-requested [Ollama and API assessment](api-capabilities.md). Listing,
per-request model selection, on-demand activation, HF download/import, separate
system content and optional session/release build on confirmed scope above.
The two triage groups are owner-approved in D-041/D-042. Deferred work retains
explicit triggers; confirmed scope does not imply runtime support.

| Feature | Status | Notes |
| --- | --- | --- |
| Optional native Ollama API compatibility profile | confirmed | Owner approved 2026-09-22, D-041: listing/details and chat/generation, tested with a named client. Preload/unload and partial-residency mapping remain separate follow-ups; M10 |
| Ollama registry downloads and other model-management compatibility | deferred | Owner approved deferral 2026-09-22, D-041: revisit when a named client needs them; earliest after basic subset and relevant native management operation |
| Machine-readable API and per-model capability discovery | confirmed | Owner approved 2026-09-22, D-041: schemas/features/model limits in M5, cluster availability in M6a; discovery does not reserve capacity |
| Explicit final-turn flag and idempotent continuation close | confirmed | Owner approved 2026-09-22, D-041, M6: targeted conversation release with completion-safe cleanup and independent shared-prefix retention; D-055 fixes the semantics, the exact schema remains M6 API design |
| Asynchronous warm/install job surface | confirmed | Owner approved 2026-09-22, D-041: progress/status/cancellation/retry, validated prepared publication, warming subject to capacity, no implicit downloads from inference; D-054 adds staging, archive, peer replication and the explicit archive/delete selection. Import/download jobs M5, warm jobs M6, archive and peer replication M6a |
| Typed text resources, images and audio-file inputs | confirmed | Owner approved 2026-09-22, D-042: incremental delivery with validated models/backends and bounded preprocessing; output modalities separate; text resources M10. Images, video and audio files moved to M3.5 and M4 with their families (D-101) |
| Live audio/video input | deferred | Owner approved deferral 2026-09-22, D-042: concrete workload must establish streaming/synchronization needs; earliest M9 planning after initial file-input evidence, not an automatic deliverable |
| Optional MCP management adapter | confirmed | Owner approved 2026-09-22, D-042: after native management API, separate process exposing discovery/status and explicitly authorized actions; no arbitrary tool execution in runtime; M10 |
| Application permissions, priorities, queue waits, cancellation and bounded progress events | confirmed | Owner approved 2026-09-22, D-042: interactive/background request priority and maximum queue waits within single-owner scope, not production multitenant isolation. Priorities, queue waits, cancellation and progress events M6 (D-069 uses them); application permissions M10 |
| Embedding API | confirmed | Owner approved 2026-09-22, D-042: later scope with a validated embedding model/output contract; M10. Owner-requested [EmbeddingGemma 2 research](m35-families.md#owner-requested-research-addition-embeddinggemma-2), 2026-10-06, supplies a concrete multimodal candidate; model/media/output and client API qualification remain open, without changing this schedule or D-044 generic-pooling scope |
| Batch/background inference jobs | deferred | Owner approved deferral 2026-09-22, D-042: concrete workload must justify scheduling/storage requirements; earliest M9 planning after request scheduling is validated, not an automatic deliverable or implied by background request priority |

## vLLM API follow-up triage (2026-09-22)

The owner requested a [vLLM comparison](vllm-api-assessment.md) after the
D-041/D-042 triage. D-043 confirms direct wire compatibility as the default
and the first three additions; D-044 completes this triage.

| Feature | Status | Notes |
| --- | --- | --- |
| General tokenization/counting and authorized prompt preview | confirmed | Owner approved 2026-09-22, D-043: vLLM-compatible tokenize/detokenize/tokenizer_info and prompt rendering for supported formats; same tokenizer/template as inference, no admission guarantee; M10 |
| JSON/schema-constrained output and strict tool arguments | confirmed | Owner approved 2026-09-22, D-043: standard response_format JSON-object/schema and strict tools plus vLLM structured_outputs.json; documented subset, explicit unsupported/incomplete handling; M7, with the daily-driver models |
| Regex/grammar constrained-output extensions | deferred | Owner approved 2026-09-22, D-043: revisit for a concrete client requirement, earliest after validated JSON/schema support; no automatic milestone delivery |
| Per-model reasoning output/control contract | confirmed | Owner approved 2026-09-22, D-043: protocol-compatible reasoning/final/tool fields and streaming, including vLLM reasoning; advertise model-supported controls and reject unsupported settings; M7, with the daily-driver models |
| Reranking alongside embeddings | confirmed | Owner approved 2026-09-22, D-044: existing rerank/v1/v2 contracts where supported, tested with unmodified retrieval clients and validated models; M10 |
| Prometheus metrics and compatible health/load queries | confirmed | Owner approved 2026-09-22, D-044: /metrics, compatible queries, metric names reused only with matching meaning; paging measurements separate, bounded labels and authorization; M6 |
| Raw Completions and bounded token/logprob diagnostics | confirmed | Owner approved 2026-09-22, D-044: OpenAI-compatible /v1/completions, standard log-probability fields and bounded vLLM-compatible token diagnostics; M10 |
| LoRA adapters | deferred | Owner approved deferral 2026-09-22, D-044: concrete adapter workload required; earliest M9 planning after validated base-model execution, not automatic delivery |
| Classification, reward and generic pooling APIs | deferred | Owner approved workload-driven scope 2026-09-22, D-044: concrete model/task demand required; earliest M9 planning after validated base-model execution. Decision models over the Jev API are the exception, in M3.5 (D-101) |
| Generic worker RPC, training controls and split-serving deployment APIs in the client baseline | rejected | Owner approved exclusion 2026-09-22, D-044; specialized runtime/deployment controls do not belong in this baseline. Compatible prompt-rendering endpoints remain confirmed in D-043 |

## Front-door contract and review fixes (2026-09-22)

A review of the D-040–D-044 documents against live client documentation found
gaps the owner directed to be fixed; D-045 records the resulting
public-interface rules. Scope is the [client API baseline](client-api-baseline.md).

| Feature | Status | Notes |
| --- | --- | --- |
| Single inference front door with a separate local-only management listener; anonymous loopback access until a credential is configured; per-operation authorization | confirmed | Owner-directed 2026-09-22, D-045. Credentials are optional on every inference binding, loopback, the tailnet (the default with loopback) or any configured address (D-014's owner note of 2026-09-28, D-097); the Ollama profile shares the front door on a configurable port and 11434 is not claimed by default |
| Cross-origin policy: loopback origins by default, configured list otherwise, wildcard only on loopback with a credential; loopback `Host` check | confirmed | D-045; the equivalent of Ollama's `OLLAMA_ORIGINS` and host guard for browser-hosted local clients |
| Admission-outcome HTTP status contract and SSE-streaming keepalive rule (headers and first event on admission, pings or SSE comments through switches, `retry-after` at most 60 s); non-streaming returns one JSON outcome within the request deadline | confirmed | D-045/D-047; derived from documented Claude Code and Codex timeout and retry behavior; verified in M5 acceptance |
| Standard-client advisory signals: Claude Code request-class and context-compacted headers (opt-in on a custom base URL), `cache_control`/`prompt_cache_key`, Ollama `keep_alive` mapping | confirmed | D-045; hints steer priority and retention and never identify a conversation or grant retention (D-031) |
| Anthropic-shape `GET /v1/models` on the shared path; alias echo in `model` with resolved identity in an extension header; namespaced extension headers | confirmed | D-045; Claude Code's opt-in discovery and its `claude` ID filter; D-062 sets the `jitllm-` header prefix and a single `jitllm` body object; M5 fixes the individual names |
| Strip Claude Code's attribution block from prefix identity; alias auxiliary requests to the main model by default | confirmed | D-045; without the first, D-031's shared-prefix reuse never fires for Claude Code; without the second, side requests cause switch storms |

## OpenRouter extension vocabulary (2026-09-22)

Owner-requested [OpenRouter assessment](openrouter-api-assessment.md), triaged
by the owner on 2026-09-22 (D-046): spellings adopted on the existing
OpenAI-shaped routes, no fourth protocol. Execution evidence is still owed.

| Feature | Status | Notes |
| --- | --- | --- |
| OpenRouter model-metadata fields in `/v1/models` entries; per-model `endpoints` shape for M6a cluster availability | confirmed | Owner approved 2026-09-22, D-046: rides with M5 discovery (D-041) and M6a availability; values from the artifact, configured limits and the implemented profile only, pricing and uptime omitted |
| OpenRouter `reasoning` request object, `reasoning`/`reasoning_details` output with jitLLM-signed blocks, and `cached_tokens`/`cache_write_tokens` usage | confirmed | Owner approved 2026-09-22, D-046: the Chat Completions spelling under D-043's reasoning contract, current vLLM also uses `reasoning`; `reasoning_content` is legacy-only, and signed blocks use SDK-supported `format: "unknown"` (D-047); cache fields report real prefix reuse only; M7 with D-043's reasoning contract |
| `session_id`, `user` and `metadata` as advisory hints | confirmed | Owner approved 2026-09-22, D-046: affinity, attribution and retention preferences under D-045's signal rules; never identity, retention grants or authorization. Accepted as advisory preferences from M5 ([signals](client-api-baseline.md)); the hint extension lands in M10 with sessions |
| `models` array with `provider.require_parameters`/`quantizations` as the opt-in fallback spelling | deferred | Owner approved 2026-09-22, D-046: reserved as the spelling if alternative-model fallback is ever accepted; fallback itself stays a D-042 design suggestion, never silent; no earliest milestone until fallback is accepted |
| OpenRouter plugins, transforms, auto-router, routing suffixes, pricing, credits, service tiers and generation stats | rejected | Owner approved exclusion 2026-09-22, D-046: server-side tools, lossy prompt rewrites and billing have no local meaning; `transforms` and `plugins` are rejected explicitly, cost fields omitted |

## Decision models, media inputs and generation APIs (2026-10-02)

Owner-requested on 2026-10-02 and recorded in D-101. The facts behind it
(the Jev API, which planned checkpoints carry which encoders, and how
clients generate images) are in
[m35-families.md](m35-families.md#media-inputs-decision-models-and-generation-apis).

| Feature | Status | Notes |
| --- | --- | --- |
| Decision models over TypeSafe's Jev/SystemOne API (`POST /v1/systemone`) | confirmed | D-101, M3.5: wire-compatible with TypeSafe's OpenAPI spec and SDKs; Clef and Clef-flash as the test models; a prefill-only decision program feeding the model's joint schema head; independent requests to each model batch internally |
| Image, video and audio file inputs per family | confirmed | D-101: M3.5 for Qwen3.8 Flash Next and the M3.5 checkpoints whose files carry encoders, DeepSeek V4 Flash Vision-Exp and Gemma 4 E4B-it as the audio carrier; M4 for GLM-5.3 Flash and DeepSeek V4.1 Flash. Inline data only; remote URL fetching off by default |
| Image generation and editing (`/v1/images/generations`, `/v1/images/edits`) | confirmed | D-101, M3.5: OpenAI's shape with vLLM-Omni's diffusion fields; Qwen-Image-2.1 generation and edits, Ming-Image-0.1-Design generation including RGBA; compatible work batches across independent requests with their own settings and cancellation |
| Video generation jobs (`/v1/videos`) | confirmed | D-101, M3.5: the asynchronous job shape vLLM-Omni, SGLang and LiteLLM serve, for MiniMax H3 text-to-video and image-to-video; generated media kept until fetched or expired |
| Text-to-speech (`/v1/audio/speech`) | confirmed | D-101, M3.5: Breeze-TTS-2 and Kokoro-82M as testbeds (owner, 2026-10-02) |
| MCP media server | confirmed | D-101, M3.5: an optional separate process exposing image and video generation as MCP tools, for coding agents whose built-in image generation cannot point at a local server |
| Stable Diffusion WebUI routes (`/sdapi/v1/txt2img`, `/img2img`, `/sd-models`, `/options`) | confirmed | D-101, M3.5: for Open WebUI, SillyTavern and LibreChat over the image pipeline |
| Images in chat responses (OpenRouter `message.images` / `delta.images`) | confirmed | D-101, M3.5: a chat request to an image model returns its image |
| Speech recognition (`/v1/audio/transcriptions`) | confirmed | D-101, M3.5: on the audio carrier, for Open WebUI's and LibreChat's voice input |
| Google Gemini API (`generateContent`, streaming, `countTokens`, models) | confirmed | D-101, M5: for Gemini CLI and the google-genai SDKs by base URL; HTTPS off loopback |
| Fill-in-the-middle code completion (`suffix`, `/v1/fim/completions`, `/infill`) | confirmed | D-101, M5 (Ollama's `suffix` with the M10 profile): Continue, Tabby, Twinny, llama.vscode, Zed edit predictions; models with FIM tokens confirmed at entry |
| Embedding and rerank client shapes (TEI `/embed` and `/rerank`, Cohere `/v2/embed`, Ollama `/api/embed`, Cohere and Jina rerank) | confirmed | D-101, M10 with the pooled outputs |
| Responses hosted `image_generation` tool; `/v1/images/variations` | rejected | Owner approved exclusion 2026-10-02, D-101: no client sends the tool to a custom provider; variations are retired upstream |

## Platforms

| Platform | Status | Notes |
| --- | --- | --- |
| NVIDIA DGX Spark, one or more nodes | confirmed | D-004; the primary target |
| Discrete NVIDIA GPU on x86-64 (first: the workstation's RTX 3080 Ti, `sm_86`) | confirmed | D-082, a secondary target: one device-memory domain plus the SSD, fast whole-model swaps with one model active, partial paging within device memory, a configured budget, one GPU (device 0); direct reads land in a bounded host-VMM zone outside the device ledger and are copied in (D-081). On-demand expert paging from the SSD during a run is not in the first discrete round. The `native` build targets it, `jitllm doctor` judges it, and its `gpu-discrete` tests run on request. Fast-swap validation joins M6's single-Spark A→B→A stage, after the two-Spark swap (D-087); exactness stays GB10-referenced |
| Host RAM as a second tier on a discrete GPU | deferred | D-082: designed for as a separate memory domain with no core special case; not built. Trigger: the owner asks for it (D-082's reopen) |
| Apple silicon, single machine | deferred | D-082: in scope later in the project (unified memory like the Spark's, a different runtime); D-026's boundaries kept portable at no cost; memory API verified when the port starts |
| Intel GPUs | rejected | D-082: out for now |
| AMD, single machine | deferred | D-026: same posture |

## Load- and temperature-aware operating policy (proposal)

Owner-suggested 2026-09-22; no thermal fault is established on our Sparks.
The [community clock-cap report](https://github.com/tonyd2wild/DGX-Spark-Hard-Poweroff-Fix/tree/abb5372e4be8d6abc30281a18bf7469508212c97)
uses a fixed GPU clock ceiling and reports fewer hard power-offs after also
applying memory-pressure mitigations. That combined intervention does not
isolate the cause or demonstrate an adaptive controller. Treat it as a
measurement lead, not a validated fix for our hosts. No scripts, services or
host settings are adopted.

| Feature | Status | Notes |
| --- | --- | --- |
| Load/temperature telemetry and optional adaptive GPU clock ceiling | proposed | Observe first; evaluate fixed and adaptive policies on actual Spark workloads before selecting thresholds or promising stability/performance. Earliest M9 evaluation, or earlier if reproducible throttling or unexplained shutdowns justify diagnosis |

Candidate design: sample available temperatures, clocks, utilization,
throttling reasons, power readings and host memory pressure with timestamps
and freshness. Unsupported sensors remain unknown, not zero; GPU power is
not whole-node wall power. Keep memory admission under the existing unified
physical budget, independently of temperature. Compare stock operation with
fixed caps across prefill, decode, image denoising and two-node work; record
latency, throughput, temperature and energy where measurable. Do not borrow
another host's threshold or assume all phases have decode's clock sensitivity.

If measurements justify automatic control, make it an explicit opt-in policy
with a bounded clock range, hysteresis, minimum dwell time and gradual
recovery. Use load/phase and temperature trends rather than a single reading.
A polling loop cannot guarantee prevention of a sudden hardware power cut;
it supplements firmware protection. Define stale/missing-sensor and actuator
failure behavior before enabling it: never raise clocks on stale evidence;
retain a validated conservative ceiling or suspend new admission and surface
the fault if a safe setting cannot be established.

Keep any privileged actuator outside the native execution hot path behind a
narrow node-local supervisor interface. Capability-probe controls and verify
applied settings; [NVIDIA documents clock-lock/reset controls](https://docs.nvidia.com/deploy/nvidia-smi/),
but support and permissions must be checked on each target. Coordinate with
operator policies and other GPU users, define ownership and restart/exit
behavior, and avoid resetting an operator's existing cap. Publish the active
policy and its performance effect in diagnostics and benchmark provenance.
Scheduler responses must preserve completion-aware lifetimes and collective
ordering. Periodic global cache drops and killing unrelated workloads are
not part of this proposal; neither substitutes for our memory accounting.
No implementation milestone or public control contract is committed yet.

## Model targets

| Target | Status | Notes |
| --- | --- | --- |
| GLM-5.3-Flash, DeepSeek-v4.1-Flash, Qwen3.8-Flash-Next (via the MiaAI-Lab two-Spark references) | confirmed | as target families and reference recipes; pinned before porting; support earned per checkpoint. M3/M4's models: Qwen3.8 in M3 on one Spark, GLM and v4.1 in M4 |
| ~30B-class dense models (suitable Gemma / Llama variants) | confirmed | intended use case, not a promise for every checkpoint |
| Cloudflare Clef and Clef-flash decision models | confirmed test models | Owner-added 2026-10-02 (D-101): Qwen3.8-27B and Qwen3.5-9B backbones with a joint schema head, served over the Jev/SystemOne API in M3.5 |
| [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) image-generation/editing reference experiment | confirmed experiment target | [BF16 reference study complete](experiments/image-reference/README.md), including phase release and text switching; [GGUF study on a pinned GGML runner](experiments/image-gguf/README.md) complete (Q4_K_M/Q8_0, phase release, budgeted disk-backed denoising). M3 runs the native pipeline in BF16, like diffusers (D-087). The image generation and edit routes are M3.5's (D-101) |
| MiMo-V2.6-Flash-RL | confirmed experiment target | Owner-added 2026-09-22; [bounded text-only TP=2/EP=2 SGLang reference](experiments/mimo-reference/README.md) ran on both Sparks; smaller-quant follow-up remains a [revisit trigger](experiments/model-candidates.md). Native support not validated; sharding remains M8 |
| Gemma 4 MTP assistant drafters (26B-A4B and 31B targets) and DiffusionGemma-26B-A4B | confirmed shape targets | Owner-added 2026-09-23 (D-068) so that speculative and diffusion shapes are designed for from M0; DiffusionGemma and the 26B-A4B drafter build on the Gemma 4 26B-A4B architecture already used as a reference, and the 31B drafter targets the dense 31B. Execution M9; native support not validated |
| Small dense model plus synthetic/tiny MoE as the first bring-up vehicles | confirmed | ideation §19: separate execution, import, and pager bugs before a flagship architecture |
| First vertical-slice checkpoints and backends | confirmed | D-051: official Qwen2.5-0.5B-Instruct FP16 GGUF and llama.cpp reference ([contract](first-slice.md)); D-052 adds required same-model 4.0 bpw and mixed-rate 4.5 bpw EXL3 with ExLlamaV3 reference ([contract](exl3-bringup.md)). Both external references have run; EXL3 includes two quants and 176 kernel cases ([report](experiments/exl3-reference/README.md)); native support remains unvalidated for both |
| Reference engine for the feasibility spike: llama.cpp with MoE GGUFs | confirmed | decided 2026-09-20. Owner-provided candidates with card-verified configs in the [M0 record](m0-record.md): Qwen3.8-Flash-Next (512 experts, top-10 plus 1 shared, 6B active of 125B, plus a 51B n-gram table; 3-bit fits one node, 4-bit is borderline, 5-bit exceeds it), Gemma 4 26B-A4B (128 experts, top-8 plus 1 shared, hybrid sliding-window attention), Ornith-1.5-35B-A3B (`qwen35moe`). This does not decide the runtime substrate (open question 4) |

## Open questions (answer during M0)

Ordered by how much work a late answer would invalidate. Lesser open items
from ideation §21 (user-space baseline on the Sparks, graph integration proof,
public API scope) ride along as M0 tasks or later-milestone questions.

1. **VMM extent granularity, map/unmap cost, and physical-pool strategy on the
   real Spark driver.** Every pager data structure sizes itself on this.
   → Answered for the initial provider on 2026-09-21 (D-033): 2 MiB
   independently reclaimable extents, compatible backing handoff without a
   standing unused-handle cache. [Measured costs and retention](experiments/vmm-microbench/README.md).
   I/O/model traces may revise the policy; dependency-group scoring remains
   deferred until the Eviction row's measurement trigger.
2. **Storage I/O path.** cuFile compatibility mode vs native file I/O with
   pinned staging vs direct I/O, measured under concurrent compute and memory
   pressure, including page-cache duplication. Decides the storage service
   design and staging budget. → Initial path answered by the M0
   [comparison](experiments/io-path/README.md), D-034: direct regular files,
   bounded asynchronous submission, and GPU-accessible host VMM without a
   staging copy; D-081 then moved weights and state to device VMM behind a
   landing-zone copy. M2 validates GGML/lifetime behavior; model traces still
   tune queue policy and M6 spill limits.
3. **Async/task and completion model.** Hand-rolled executor with explicit
   continuations, C++20 coroutines, a sender/receiver library, or something
   else. C++23 does not supply the scheduler, and every interface signature
   depends on this. → Answered 2026-09-22 (D-048): explicit native task states,
   one scheduler/catalog writer per node, bounded provider services and
   completion-owned lifetimes. [Design and prototype](async-model.md);
   CPU-only event-order checks passed on the workstation and Spark. Actual
   concurrency/provider integration remains M2; D-050 answers the separate
   reservation-policy question 9.
4. **First vertical-slice model and backend.** Which checkpoint (revision,
   quantization, tokenizer, kernels, provenance) and which numerical reference
   engine. Decides M3 and gives the license audit its first real inputs.
   → Answered 2026-09-22 (D-051): [official Qwen2.5-0.5B-Instruct FP16
   GGUF and pinned llama.cpp numerical reference](first-slice.md), with exact
   embedded tokenizer/template identity and a source-unit provenance inventory.
   The bounded external reference passed on Spark; native support, numerical
   acceptance thresholds and Unicode-data clearance remain implementation
   gates. D-052 adds required small EXL3 fixtures and upstream performance
   gates; the bounded Spark reference is recorded. Prove both GGML and EXL3 can use
   jitLLM-owned memory alongside M2 before settling the contract (D-028/D-052).
5. **Experimental artifact schema and layout ABI.** Choose metadata encoding,
   alignment, integrity, sharding representation, and version rejection rules
   for bring-up. → M0 decision, likely after question 4. D-018 defers
   compatibility guarantees until a dense model and a small MoE have each
   passed import, execution, eviction, and restoration checks. Experimental
   revisions may require explicit re-import. The blob container is a reused
   known container, not bespoke (settled 2026-09-21); pick which one (see
   the Artifacts rows). → Answered 2026-09-22 (D-056): the
   [v0 format](artifact-format.md) uses safetensors shards, a strict JSON
   manifest/index, content-addressed atomic publication, exact version and
   profile rejection with re-import, 4 KiB-aligned dependency groups paged in
   2 MiB chunks, and no page-in hashing. The [layout study](experiments/artifact-layout/README.md)
   covers seven real models and built four verified artifacts. Model-parallel
   (TP/EP) partitioning is explicitly deferred, with a deadline of M4 entry:
   it depends on M4's sharding design, and v0 artifacts are whole-model.
   Compatibility guarantees stay behind D-018's gate.
6. **Exact toolchain pins validated as one unit.** Resolved 2026-09-21
   by D-032 and the [toolchain smoke](experiments/toolchain-smoke/README.md):
   LLVM 22.1.8, pinned libstdc++/glibc and Spark sysroot, NVCC 13.4.92
   (Toolkit 13.4.2) with Clang host compiler, `sm_121`, C++23 throughout.
   Native workstation, cross-to-Spark, and native Spark fallback passed;
   declarative provisioning and clean-container verification remain M1.
7. **C++ source-dependency mechanism.** Resolved 2026-09-23 by D-057:
   [CMake FetchContent with a source lock and curated vendoring](source-dependencies.md)
   for adapted units, separate from SDK provisioning. Audit and select the
   full profile closure before acquisition; configure/build uses verified
   local inputs. M1 implements and proves native/cross, offline and
   copyleft-disabled behavior; no dependency is admitted by this choice.
8. **Inference API surface.** Resolved 2026-09-20 by D-022: standard web-API
   compatibility (Cursor, OpenCode, Codex, and since 2026-09-21 Claude Code,
   D-030) is the baseline, with sessions and hints as optional extensions.
   Documentation baseline recorded in D-040 and [client-api-baseline.md](client-api-baseline.md);
   executed client compatibility and Cursor wire details remain M5 work. (The
   license half closed on 2026-09-20: Apache-2.0 in D-003, dependency policy
   in D-017.)
9. **Initial reservation guarantee and progress envelopes.** How conservative
   the first scheduler is about serializing phases, and what a "minimum
   feasible phase" envelope includes (activations, state growth, scratch,
   staging, comm, graphs, metadata). Include guaranteed versus opportunistic
   grants, retained continuations, growth limits, and impossible-phase
   handling. → Answered 2026-09-22 (D-050): guaranteed finite requests,
   maximum retained-state/growth allowance, initially one complete phase
   envelope held through waits and unwind, and full-envelope admission for
   supported concurrency. Spill does not reduce admitted state commitments;
   opportunistic work cannot consume guaranteed headroom. The
   [policy and adversarial cases](reservation-policy.md) implement the
   owner's completed-boundary lease direction. Numeric envelopes and runtime
   progress evidence remain M2/M6/M7 gates, not results of the design decision.
