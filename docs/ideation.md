<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llmpalooza — Project Ideation and Bootstrap Design

**Working tagline:** Just-in-time memory for LLMs.  
**Design snapshot:** September 20, 2026.  
**Status:** Pre-implementation architecture and bootstrap brief.  
**Initial target:** Local inference on one or two NVIDIA DGX Sparks.  
**Primary development host:** x86-64 Ubuntu Linux.  
**Implementation direction:** C++23, Clang-first, native runtime with replaceable compute backends.

> **Model implementations describe computation and dependencies. Llmpalooza owns storage, residency, scheduling, and execution lifetime.**

This document consolidates the latest project direction. It is not a claim that the runtime, example interfaces, build commands, model support, or performance results already exist. **Decided direction** means a project choice from the discussion; **proposed design** means an implementation approach to validate. Exact dependency revisions, hardware measurements, and the final license for original code remain open.

External implementation facts have source links. Those links were checked while preparing this snapshot; moving documentation and repository branches are not dependency pins. Record exact commits, package versions, artifact hashes, and target capabilities during bootstrap.

## Contents

- [1. Purpose and product goal](#1-purpose-and-product-goal)
- [2. Initial hardware and model references](#2-initial-hardware-and-model-references)
- [3. Runtime architecture and ownership](#3-runtime-architecture-and-ownership)
- [4. Core data model: resources, extents, and backing](#4-core-data-model-resources-extents-and-backing)
- [5. Memory classes and their lifecycle policies](#5-memory-classes-and-their-lifecycle-policies)
- [6. Capacity reservations and residency leases](#6-capacity-reservations-and-residency-leases)
- [7. On-demand experts and resumable execution](#7-on-demand-experts-and-resumable-execution)
- [8. CUDA VMM and storage-transfer design](#8-cuda-vmm-and-storage-transfer-design)
- [9. Global eviction and retention policy](#9-global-eviction-and-retention-policy)
- [10. Compute backends and optimization strategy](#10-compute-backends-and-optimization-strategy)
- [11. Import pipeline and prepared model artifacts](#11-import-pipeline-and-prepared-model-artifacts)
- [12. Two-node execution and communications](#12-two-node-execution-and-communications)
- [13. Management API, dashboard, and diagnostics](#13-management-api-dashboard-and-diagnostics)
- [14. Language, compiler, and build policy](#14-language-compiler-and-build-policy)
- [15. Development host and Spark cross-build environment](#15-development-host-and-spark-cross-build-environment)
- [16. Declarative setup and toolchain provisioning](#16-declarative-setup-and-toolchain-provisioning)
- [17. Licensing and provenance strategy](#17-licensing-and-provenance-strategy)
- [18. Correctness, testing, and performance evidence](#18-correctness-testing-and-performance-evidence)
- [19. Phased implementation plan and acceptance gates](#19-phased-implementation-plan-and-acceptance-gates)
- [20. Proposed repository shape and internal interfaces](#20-proposed-repository-shape-and-internal-interfaces)
- [21. Open decisions, risks, and immediate bootstrap tasks](#21-open-decisions-risks-and-immediate-bootstrap-tasks)
- [22. Sources and verification notes](#22-sources-and-verification-notes)

## 1. Purpose and product goal

Build an independent, completely open-source inference runtime for workloads in which many models should be available, only some components are active at a time, and aggregate model storage exceeds physical memory.

The system should retain useful portions of models instead of treating a model as an indivisible allocation. When another model needs capacity, reclaim the least valuable eligible extents across all models, not automatically unload an entire model. Bring missing weights or state back only when required, or when justified prefetching can hide a future miss.

The defining workload is **one large model, potentially sharded across two Sparks, plus smaller models handling mixed traffic**. The large candidate models need not all execute simultaneously. Smaller models in roughly the 30B class, including suitable Gemma or Llama variants, are an intended use case, not an assertion that every architecture or checkpoint will be supported initially.

The user is comfortable with admission delays and cold paging costs. Prioritize useful mixed-model execution, warm generation performance, memory efficiency, correctness, and explainable scheduling. Lower cold time to first token is desirable but is not the primary objective. Repeated paging stalls during generation still matter.

### Decided direction

| Area | Direction |
|---|---|
| Runtime ownership | Build our own runtime rather than make a vLLM fork or plugin the product architecture. |
| Process model | One modular native execution process per node, managing all local models. |
| Memory | Explicit CUDA VMM backing management, semantic metadata, and a global local-node resource catalog. |
| Admission | Separate logical capacity reservations from physical residency leases. |
| Paging | Partial model eviction and on-demand expert acquisition; unlocking does not automatically evict. |
| Reuse | Learn from and selectively reuse compatible implementations from inference projects. |
| Model storage | Prepare versioned, execution-ready on-disk representations at import time. |
| Implementation | C++23 host runtime, Clang-first, CUDA/native computation without an interpreted serving loop. |
| Development | Work on x86-64 Linux; cross-compile and deploy/test on Sparks over SSH. |
| Setup | Declarative toolchain setup, locked versions, project provisioning, and a reference development container. |
| Licensing | All llmpalooza-authored code open source; optional copyleft implementations must be identifiable and removable. |

### Non-goals for the first implementation

Do not attempt universal model support, an entire new tensor compiler, a new quantization scheme, arbitrary GPU page-fault interception, transparent cross-node shared virtual memory, or production-grade multi-tenant isolation. Training is outside the initial scope.

Do not rewrite a proven kernel merely to claim native ownership. Do not import an engine's allocator or scheduler merely because we reuse its kernel. Do not invent dense-model sparsity by rearranging weights: layout changes alone do not establish that a dense kernel can skip those bytes.

## 2. Initial hardware and model references

### Spark memory is one physical budget

A Spark has an Arm CPU, GB10 GPU, and 128 GB of unified system memory. CPU allocations, GPU backing, staging buffers, filesystem cache, and operating-system activity compete for the same DRAM. CPU offload is therefore not an additional physical capacity tier on this platform. NVIDIA lists GB10 at compute capability 12.1. [Hardware][spark-hardware] · [GPU capability][cuda-gpus]

Treat the two Sparks as **two local memory domains connected by a network**, not a coherent 256 GB allocation space. Model parallelism and remote object transfers must be explicit. CUDA fabric-memory features have separate platform requirements; a QSFP/RDMA connection does not establish them. [CUDA VMM overview][cuda-vmm-guide]

CUDA-only free-memory reporting is not the complete node budget. NVIDIA notes that `cudaMemGetInfo` does not include all memory Linux may reclaim on Spark. Combine a resource ledger, system measurements, pressure signals, and conservative headroom. Do not sum overlapping CPU/GPU measurements as though they described independent physical pools. [Spark optimization guide][spark-optimization]

### Storage and networking constraints to preserve

As documented at this snapshot, Spark supports GPUDirect Storage only in compatibility mode; NVIDIA says not to load `nvidia-fs` on Spark. Plan a staging path rather than assume native SSD-to-device-allocation DMA. [GDS release notes][gds-release]

NVIDIA also documents that GPUDirect RDMA into the relevant device allocations is not supported on Spark and recommends an appropriate host-buffer fallback. RDMA networking and GPUDirect RDMA are different capabilities. Probe the installed stack rather than infer one from the other. [GB10 FAQ][spark-faq]

These are capability constraints, not a reason to make the runtime storage backend Spark-specific. Other supported hardware may use more direct transfer paths.

### User-provided reference deployments

Use these as starting points for checkpoint selection, architecture details, numerical references, kernel adaptations, and target-side experiments:

| Target family | Reference |
|---|---|
| GLM-5.3-Flash | [MiaAI-Lab GLM Spark deployment][mia-glm] |
| DeepSeek-v4.1-Flash | [MiaAI-Lab DeepSeek Spark deployment][mia-deepseek] |
| Qwen3.8-Flash-Next | [MiaAI-Lab Qwen Spark deployment][mia-qwen] |

These repositories report working Spark configurations. This document does **not** independently validate their benchmarks or promise equivalent llmpalooza support. Preserve a known-working pinned reference before porting. A model family's name is insufficient: record checkpoint revision, quantization, tokenizer, attention/state format, sharding, kernels, and any speculative-decoding configuration.

Do not reintroduce the earlier assumption that default full-precision checkpoint sizes decide feasibility when the actual target is a prepared quantized representation. Measure the intended checkpoint's loaded and peak working-set costs.

## 3. Runtime architecture and ownership

### One execution process per node

The local runtime contains all model execution, scheduling, memory policy, VMM control, and completion tracking. A dashboard, importer, test controller, and supervisor may run separately. They are not in the per-expert hot path.

```text
x86-64 development workstation
    editor / indexer / builds / CPU tests / import tools
                         |
                   SSH deployment
                         |
        +----------------+----------------+
        |                                 |
Spark A: llmpalooza runtime              Spark B: llmpalooza runtime
  model/execution registry           model/execution registry
  node-wide resource catalog         node-wide resource catalog
  reservations + task scheduler <--> reservations + task scheduler
  VMM + storage + completion         VMM + storage + completion
  native compute backends   <------> native compute backends
        |                            model communication
        +---------- management API ----------+
                          |
                optional web dashboard
```

The two runtime instances have independent addresses, backing handles, and physical budgets. Cluster coordination exchanges logical resource identities, phase identifiers, grants, and transfer metadata—not naked pointers to remote model weights.

### Component responsibilities

| Component | Responsibilities |
|---|---|
| Model registry | Model identity, tokenizer/configuration, adapters, immutable manifests, loaded plans, and lifecycle. |
| Execution planner | Select compatible operations/fused segments, declare dependencies, workspace and yield boundaries. |
| Scheduler | Admit requests, advance ready continuations, apply fairness, and coordinate distributed phases. |
| Resource catalog | Describe every managed logical resource and its storage, backing, lifetime, aliases, and statistics. |
| Memory manager | Budgeting, reservations, residency leases, precise cross-model victim selection, mapping policy. |
| Storage service | File extents, read/write queues, staging, prefetch, integrity checks, and spill management. |
| Completion service | Track GPU, I/O, and network consumers before allowing reclamation. |
| Compute backends | Execute declared operations using caller-controlled state, streams, and workspace. |
| Management plane | Configuration, status, import requests, metrics, traces, and explainable control actions. |

Prefer one deliberately managed CUDA context per GPU initially, with streams and library handles assigned explicitly. Validate every backend's context, stream, and thread assumptions. Separate model contexts at the **application** level from CUDA contexts.

“Monolithic” means a single authority over local execution state, not a single thread, an undifferentiated codebase, or a global lock held during I/O. Start with understandable queues and synchronization. Introduce lock-free structures only where measurements justify their complexity.

### Earlier alternatives no longer define the product

The earlier vLLM extension, per-model worker, and external local broker proposals are reference/integration options, not the selected runtime architecture. There is no required local IPC transaction for an expert lease or block eviction. A coordinator may still exist across nodes, and optional adapters may eventually expose the native memory subsystem elsewhere.

## 4. Core data model: resources, extents, and backing

Keep logical identity separate from the current address and physical occupancy.

```text
Logical resource
    -> one or more tensor/storage byte ranges
    -> independently reclaimable backing extents
    -> zero or more valid stored representations
```

An expert can span multiple arrays: packed weights, scales, quantization auxiliaries, and pointer/dispatch metadata. Multiple logical resources can share an extent. The manager must know the complete dependency closure and charge each physical extent once.

### Proposed resource descriptor

| Field group | Required information |
|---|---|
| Identity | Model/checkpoint revision, representation version, shard, layer, component, logical block ID. |
| Content kind | Expert weight, dense weight, attention weight, lookup table, live sequence state, reusable prefix, scratch, graph/runtime, communication, staging. |
| Semantics | Read-only or mutable; preserve, reconstruct, recompute, or discard; access/lifetime rules. |
| Layout | Byte ranges, alignment, dtype, packing, aliases, tied storage, extent membership. |
| Recovery | File/object identity, offset/length, content generation, checksum, codec or reconstruction method. |
| Residency | Reserved address, mapped extents, load/write-back state, mapping generation. |
| Safety | Lease references, outstanding consumers, registrations, revocation and cancellation state. |
| Policy | Last actual use, decayed frequency, reuse estimates, recovery cost, priority, known next use. |

Metadata categories are orthogonal. “Read-only” does not mean “always resident.” “Unleased” does not mean “contents may be lost.” “Reconstructible” does not mean “already stored in the correct format.” “Resident” does not mean “available for a new non-preemptible operation without admission.”

Use strong typed IDs and generation checks instead of exposing raw pointers as permanent identities. Raw addresses belong in the device/backend boundary. Shared/tied weights require content and representation identity, not matching tensor names alone.

### Basic state machine

```text
NONRESIDENT -> LOADING -> RESIDENT_UNLEASED <-> RESIDENT_LEASED
     ^                        |
     |                        v
     +-------------------- EVICTING

LOADING -> FAILED
EVICTING -> RESIDENT_UNLEASED   when eviction is cancelled safely
```

This is a conceptual state machine. Real operations also need content generations, consumer counts, and cancellation tokens. A dirty extent cannot become nonresident until valid recovery is secured or an explicit discard decision has invalidated its logical contents.

Unknown or unclassified allocations are non-evictable by default. Registration happens at construction; immutable backing is finalized only after loading, packing, and postprocessing have finished.

## 5. Memory classes and their lifecycle policies

| Class | Initial policy |
|---|---|
| Immutable weights | Discard clean resident copies under pressure; restore execution-ready bytes from prepared storage. |
| Routed expert weights | Acquire only the selected experts' storage dependencies; release execution protection after consumption. |
| Dense and attention weights | Acquire the ranges the selected implementation actually reads; no assumed activation sparsity. |
| Sparse lookup tables / modality components | Use component-aware dependencies where semantics allow selective use. |
| Live KV, compressed attention, recurrent state | Preserve while a request may resume, through residency, valid spill, or explicit reconstruction. |
| Reusable completed-prefix state | Retain by expected reuse and recovery value; invalidate correctly when discarded. |
| Temporary activations and scratch | Retain until final consumers complete, then recycle; normally do not spill dead scratch. |
| Graph/runtime objects and kernel code | Keep out of fine-grained paging initially; account for their allocations and permit coarse cleanup later. |
| Communication buffers | Keep backing stable for registration and in-flight network lifetimes. |
| Transfer staging | Bounded, explicitly budgeted pool reserved before pressure becomes critical. |

The usual attention model reuses past keys and values across generated tokens. The current prompt is not disposable immediately after prefill. Exact reusable prefixes also depend on preceding context and model-related identifiers, not just repeated text. [Cache explanation][cache-explanation] · [Prefix caching][prefix-caching]

Represent live and reusable state through architecture-specific adapters. Conventional KV blocks, compressed attention state, sliding windows, and recurrent summaries do not share one universal lifetime rule. Initially retain conservative semantics for each supported architecture.

System prompts are often good retention candidates, not a permanently pinned memory partition. Prefix ownership and lookup metadata must agree with physical eviction: either the object remains valid with recoverable backing, or the cache entry is invalidated. A stale cache hit must never point at discarded contents.

## 6. Capacity reservations and residency leases

### Three separate concepts

1. **Virtual reservation:** an address-space operation. It does not provide physical capacity.
2. **Capacity reservation:** an admission/accounting commitment that a bounded operation can obtain the necessary memory under a defined progress policy.
3. **Residency lease:** protection for particular backing while its consumers execute.

Do not use “reserved memory” without identifying which meaning applies.

### Request transaction versus execution lease

A prompt/request owns a logical transaction and persistent-state budget. Smaller execution phases acquire concrete residency leases. The transaction may last an entire response; an expert lease should usually last only until its final relevant GPU consumers complete.

Transactions provide all-or-nothing **admission/acquisition** at a declared boundary. They do not promise database-style rollback of a model state update after a kernel has begun. Failed or cancelled acquisitions return unconsumed grants; in-flight operations complete or enter a controlled failure path before memory is reused.

### Lazy commitment

Granting capacity must not eagerly evict useful cached data. Unused reservation allowance may remain occupied by revocable cached contents until a real dependency miss requires it.

Illustrative, not measured:

```text
A has many resident but unleased expert extents.
B receives a 12 GiB execution allowance.
    Nothing is evicted solely because of the allowance.

B discovers 2 GiB of missing dependencies.
    Reclaim sufficient eligible extents across the node.
    Load those dependencies and run B's phase.

B's phase completes.
    Release leases; keep useful B contents cached.
    Preserve all untouched A extents.
```

A reservation is not a free duplicate claim on actively leased memory. Memory loaned as a cache can be revoked; memory protected by another in-flight consumer cannot. Classify grants as either guaranteed under a defined safe-progress schedule or opportunistic and explicitly deferrable.

### Admission and progress invariants

- Account for the union of shared extents, not the sum of every request's logical sizes.
- Charge newly committed physical backing before allocation/loading starts.
- Keep evicting or pending-write-back capacity charged until it is actually reclaimable.
- Do not add a reservation ceiling to already charged backing as though both consume physical bytes. Maintain commitment and occupancy ledgers separately.
- Include live activations, bounded state growth, scratch, staging, communications, graphs, and metadata in the progress envelope.
- A request whose minimum feasible phase can never fit must fail or select another valid plan, not wait forever.
- Acquisition and eviction must have mutually exclusive state transitions. Generation checks prevent stale completions from modifying reused storage.
- Never hold the global scheduling/catalog lock while waiting for GPU, disk, or network completion.

Dynamic expert sets make an exact full-request weight reservation unrealistic. Reserve a feasible execution envelope; acquire the selected extents once routing resolves them. If a phase's selected set exceeds its envelope, split work only through a validated execution plan, obtain a larger grant, or defer before unsafe submission.

A suspended continuation can itself retain large activations. Bound the number and size of suspended phases. For the first safe scheduler, conservatively serialize phases that cannot coexist; later admit more overlap based on measured, declared envelopes. Avoid cycles in which several continuations hold all capacity and each needs additional memory to finish.

## 7. On-demand experts and resumable execution

### Selected-expert boundary

The pageable path should make routing a distinct dependency-discovery stage:

```text
Prepare input and router dependencies
    -> execute routing
    -> obtain selected logical expert IDs
    -> resolve local shard IDs and storage ranges
    -> acquire residency for their dependency closure
    -> execute expert computation
    -> combine results / continue the model
    -> release leases after all consumers complete
```

vLLM's modular MoE interface is a useful reference for separating selected expert IDs, expert computation, workspace, and implementation capabilities. It is not the required llmpalooza API. [MoE implementation reference][vllm-moe]

Never substitute a resident expert for the expert the model selected. Never drop a selected contribution to avoid a miss. Prefetching is speculative; actual routing remains authoritative.

### Initial implementation

Allow a compact GPU-to-host report of selected experts and a native host residency decision. This costs a synchronization boundary; record its cost even on all-resident paths. The native scheduler should suspend the model continuation while I/O is pending and run other ready work, rather than block the only thread that can make progress.

Initially acquire all selected expert extents needed by the current invocation before launching the existing numerical implementation. Executing ready experts while other experts load is a later scheduling/kernel integration, not an assumed capability.

A batched phase's working set is the union of its token routes. For E experts, k selections per token, and T tokens, the distinct selection count is bounded by `min(E, k*T)`. This is a combinatorial bound, not a prediction of locality. Prefill may touch far more experts than a small decode batch. Chunk sizes are therefore memory and scheduling decisions as well as throughput decisions.

Collect real routing traces and replay them through simulated cache policies. Measure reuse distance and miss bytes; do not equate advertised active parameter count with a stable cache working set.

### Later fast paths

Investigate a GPU-visible residency table and compact miss notifications only after a correct baseline exists. A residency check must also acquire protection against revocation; a Boolean check followed by an unprotected launch is unsafe.

Retain an optimized fully resident path where it is valuable. Do not force every invocation to pay the maximum flexibility cost when an execution plan already guarantees its complete dependency set.

### CUDA graphs

Ordinary host code executed during capture is not automatically rerun with graph replay. Moving that code from Python into C++ does not change this. CUDA graphs represent explicitly captured/constructed operations; graph nodes and synchronization have their own contracts. [CUDA graphs][cuda-graphs]

Start with dynamic residency decisions outside captured segments:

```text
Captured segment through routing
    -> native admission/residency boundary
    -> expert compute segment
    -> remaining computation
```

CUDA does support host-function nodes, but CUDA API calls are prohibited from the relevant host callbacks. Do not build a pager that calls mapping APIs or waits on its own blocked GPU work from such a callback. Use ordinary native service threads and explicit dependencies. [CUDA host functions][cuda-execution]

Graph identity, workspace layout, pointer tables, and registrations need lifetime management. Stable virtual addresses are helpful, not proof that an executable graph and every associated external registration remain valid after backing changes.

## 8. CUDA VMM and storage-transfer design

### Explicit control is the selected approach

Use the Driver API to reserve addresses, create physical backing, map it, and set access permissions. Llmpalooza supplies backing-store policy and transfer operations. Explicit CUDA VMM is not an automatic SSD pager; accessing absent backing is a bug, not a request to our storage service. [VMM guide][cuda-vmm-guide] · [VMM API][cuda-vmm-api]

Maintain stable virtual addresses for a model's loaded lifetime where practical. Distinguish independently reclaimable allocation extents from tensor ranges and I/O chunks. The API's granularity, whole-mapping unmap rules, and handle lifetime requirements constrain implementation. An unmapped extent still held in a physical pool consumes RAM; report it as reusable pool capacity, not memory returned to the OS. [VMM API][cuda-vmm-api]

Do not assume map/unmap operations are free or fully asynchronous. Benchmark their cost and their interaction with concurrent work on the actual driver. A small physical-backing pool may amortize allocation overhead, but its retained capacity must remain in the ledger.

### Page-in lifecycle

```text
Reserve/commit capacity for actual missing extents
    -> obtain physical backing
    -> map and establish required access
    -> transfer prepared bytes
    -> verify completion and content identity
    -> publish resident state
    -> grant execution lease
```

No consumer sees an extent as ready before its required bytes and permissions are established. Coalesce duplicate requests for the same content generation into one load. Cancelled request interest does not justify reusing a transfer buffer while its I/O remains in flight.

### Eviction lifecycle

```text
Select specific eligible extents
    -> atomically exclude new leases
    -> wait for all consumers/registrations that require the backing
    -> write back only when preservation requires it
    -> commit recoverable state / invalidate discarded logical entries
    -> unmap and release or recycle backing
    -> update occupancy and generation
```

Clean weights with valid backing require no SSD write. Mutable snapshots require a consistent version and confirmed write completion. Dead scratch can be discarded. Optional crash durability is a separate policy from retaining data for a later read in the same process lifetime.

### Storage backends

Provide a common internal read/write completion interface. The initial Spark implementation should compare:

| Backend | Intended role |
|---|---|
| cuFile compatibility mode | A supported library-managed fallback path to benchmark. |
| Native file I/O plus pinned staging | Explicit queue depth, bounded buffers, priority, and cancellation control. |
| Native GDS | Optional path on hardware where the direct path is supported and validated. |
| Remote extent transfer | Later extension; explicit object requests, not transparent remote page faults. |

GDS supports file transfers involving VMM-allocated GPU memory on supported configurations, but its API does not supply a model pager or make unsupported hardware direct. [GDS overview][gds-overview]

For Spark, the expected staged paths are:

```text
Read:  SSD -> host staging -> CUDA copy -> mapped model backing
Write: mapped model backing -> CUDA copy -> host staging -> SSD
```

Reserve staging and completion infrastructure before pressure peaks. Handle short transfers, checksum errors, storage exhaustion, alignment, retries, and cancellation explicitly. Keep queues bounded. Avoid a design that must allocate unbudgeted RAM in order to evict RAM.

Compare buffered reads with direct-I/O approaches; filesystem cache duplication and read amplification must be measured, not assumed away. Do not use system-wide cache flushing as normal runtime policy.

## 9. Global eviction and retention policy

The local resource catalog gives the manager enough information to compare **specific extents across all models**. `shrink(model, N bytes)` may be a convenience API, but is not the fundamental optimization boundary.

A proposed first heuristic is:

```text
eviction_cost_per_reclaimed_byte =
    (cost_to_reclaim_now
     + probability_of_near_term_reuse * recovery_cost
     + expected_stall_cost_for_admitted_work)
    / physical_bytes_actually_reclaimed
```

This is a heuristic, not an optimal policy or a measured predictor. Account for grouped backing, aliases, I/O coalescing, uncertain reuse, priority, and fairness. A clean hot expert can be more valuable than a dirty low-value cache object. No content type always wins.

Track real use separately from prefetch and cancelled planned use. Keep safety data exact; recency and frequency estimates may be approximate. Combine per-model statistics with global comparisons so one busy model does not monopolize every reclaimable byte indefinitely.

Release of a lease changes **eligibility**, not residency. Add hysteresis, minimum useful residency where appropriate, and reload-cost awareness to avoid oscillation. Consider waiting briefly for an imminent completion instead of evicting expensive contents.

For the first version, compare simple global LRU, a frequency/recency policy, and the cost-aware heuristic against identical traces. Report miss bytes, reload repetition, write volume, and model waiting time—not merely cache-hit count.

## 10. Compute backends and optimization strategy

Use existing projects as sources of algorithms, kernels, model semantics, and test references. Do not make any one project's execution architecture mandatory.

| Reference | What to investigate |
|---|---|
| vLLM | Model details, attention/state integration, MoE dispatch, quantization, graph execution, batching techniques. |
| llama.cpp / GGML | Native model execution, tensor/backend contracts, import and quantization implementations. |
| Ollama | Product-facing model lifecycle and API behavior; not the memory-system architecture. |
| ExLlamaV3 and EXL3 integrations | Packed expert representations, kernels, and architecture-specific launch details. |
| FlashInfer | Attention and related native computation/planning implementations. |
| CUTLASS / CuTe | Matrix kernels, layouts, fusion, and hardware-specific implementation building blocks. |
| Other engines and research | Candidate optimizations evaluated against llmpalooza's contracts and workload. |

FlashInfer and CUTLASS expose useful kernel infrastructure; vLLM's MoE interface illustrates why activation layout, quantization, workspace, and dispatch compatibility matter. Reuse compatible implementation units, not isolated benchmark winners assembled without their assumptions. [FlashInfer][flashinfer] · [CUTLASS][cutlass] · [MoE reference][vllm-moe]

### Proposed operation contract

A backend declares supported architectures, tensor layouts, quantization, numerical behavior, persistent state, workspace, resource dependencies, graph restrictions, parallel configuration, and completion semantics. It executes using explicit streams, model state, communicator references, and caller-managed storage.

An implementation may cover one operation or a fused segment. Keep a fully resident fused plan and a pageable split plan when both are useful. Shape specialization and plan selection are valid; hidden ownership of large memory pools is not.

Define effective performance in context:

```text
compute + layout conversion + synchronization + unhidden I/O
        + interference with other admitted work
```

This expression describes costs to measure; overlapping stages are not simply summed for wall-clock prediction. Scratch footprint and retained state can make a slower isolated kernel better for mixed-model execution.

Separate lossless layout transformations from quantization or precision changes. The latter require explicit quality evaluation and user-visible representation metadata. A faster kernel is not acceptable if it silently changes routing, masking, state semantics, or numerical policy.

### Native runtime does not prohibit non-native build tools

No Python interpreter is required in the intended serving, paging, or scheduling hot path. Optional import/build tooling can use Python or another compiler frontend when it produces usable native kernels and metadata.

Triton's AOT tooling can produce native launch integration for supported kernels, but not every dynamically configured kernel can be adopted unchanged. Keep compiler versions, specializations, target metadata, workspace, and generated-code provenance. [Triton AOT source][triton-aot]

Avoid a universal compiler framework in the first milestone. Start with a small set of explicit model operations and validated backend adapters.

## 11. Import pipeline and prepared model artifacts

### Import is part of the runtime contract

Own the conversion from an original checkpoint into execution-ready storage:

```text
Source checkpoint + configuration + tokenizer
    -> validate architecture, revision, and licensing metadata
    -> select quantization-preserving backend layout
    -> pack and shard static data
    -> index resource dependencies and reclaimable extents
    -> hash and atomically publish prepared artifact
```

The artifact should enable bounded, range-based reads without reprocessing the whole checkpoint. “Mappable” means an indexed representation suitable for direct population of runtime extents. It does not mean `cuMemMap` directly maps an SSD file. [VMM guide][cuda-vmm-guide]

### Proposed artifact contents

| Element | Content |
|---|---|
| Manifest | Format version, architecture, source revision/hashes, converter identity, numerical policy. |
| Tokenization/configuration | Tokenizer, vocabulary, prompt/chat formatting, special tokens, positional and architectural parameters. |
| Representation catalog | Backend/layout ABI, quantization, hardware requirements, sharding, optional alternative layouts. |
| Resource index | Expert/tensor membership, ranges, aliases, alignments, file extents, expected reconstruction method. |
| Immutable data | Kernel-ready weights, scales, lookup data, and other persistent constants. |
| Integrity/provenance | Extent checksums, source/license references, transformation history. |
| Optional plan metadata | Compatible operation plans and specialization keys; rebuildable when compatibility changes. |

Do not serialize process addresses, C++ object layouts, live mutexes, allocator handles, or executable graph objects as durable model state. Rebuild pointer tables and runtime objects for the current process. Use explicit-width fields, defined byte order, checked offsets, and bounds validation.

Do not duplicate every possible backend layout by default. Permit alternatives only when their measured value justifies disk and import costs. Keep mutable spill files separate from immutable model files. Requantization must be an explicit transformation, not an incidental effect of switching kernels.

### Import and deployment workflow

Prefer the workstation for parsing and conversion that does not require target hardware. Permit target-assisted tuning or conversion as an explicit mode. Distinguish the x86 importer executable, ARM importer executable, and architecture-independent artifact format.

Build-time tools run on the build host. Target binaries are not executed during cross-configuration. Hardware-dependent tuning occurs on the Spark and writes a separately keyed result.

Check free disk space and peak temporary usage before conversion. Support resumable work and atomic publication so interrupted imports do not appear as valid models. Treat checkpoints as untrusted input: avoid arbitrary code execution and validate lengths, paths, hashes, and metadata.

## 12. Two-node execution and communications

Each Spark independently owns its local physical memory. A coordinator issues a distributed phase ID and obtains a consistent admission decision across participating nodes. Local eviction victims may differ.

Start with explicit, conservative prepare/commit semantics:

```text
Describe phase and local requirements on each rank
    -> reserve capacity on all required nodes
    -> establish required local residency
    -> commit distributed execution
    -> preserve collective order
    -> acknowledge completion/cancellation
```

A rank must not enter a collective while another rank can wait indefinitely on an unapproved allocation. Grant timeouts are not proof that in-flight memory is free. Cancellation must distinguish unstarted preparation from submitted execution.

Tensor parallelism, pipeline parallelism, and expert parallelism are different plans. Port the validated target recipe first; do not assume expert IDs or ownership are interchangeable across strategies. In tensor parallelism, a selected expert may require a local shard on every rank.

### Stable communication storage

Use a separately budgeted communication-buffer pool. Registrations and outstanding operations can depend on backing identity, not only a virtual pointer value. Stable virtual addresses do not authorize remapping underneath a registered buffer. NCCL's registration and thread-safety contracts must be honored by each adapter. [NCCL registration][nccl-registration] · [NCCL threading][nccl-threading]

Keep collective submission deliberately ordered. Maintain completion fences for all GPU and network consumers. Prefer stable host staging on Spark where required by its capabilities. A future direct path must explicitly manage registration retirement and recreation when backing changes.

Remote paging, if added, requests a logical object/version/extent and transfers it through a supported transport. The source remains leased until completion. This is not remote `cuMemMap`, and remote capacity is never counted twice.

## 13. Management API, dashboard, and diagnostics

The dashboard can be a separate process using a versioned local management API. It must not own the model scheduler or take runtime locks while rendering.

Expose model import/list/remove, representation inspection, admission priorities, residency policies, request cancellation, node health, and trace capture. Model removal must quiesce users and respect live references; changing a budget is a scheduler request, not an immediate unsafe unmap.

Useful views include:

| View | What it explains |
|---|---|
| Memory breakdown | Committed, mapped, leased, cached/revocable, dirty, loading, evicting, staging, and unmanaged/headroom bytes. |
| Model working sets | Resident expert distribution, usage history, shared storage, and missing dependencies. |
| Request state | Running, waiting for residency, waiting for capacity, waiting for a rank, cancelled, or failed. |
| Eviction decisions | Victims, expected cost, bytes actually recovered, and why alternatives were retained. |
| I/O timeline | Demand/prefetch reads, write-back, queue depth, staging copies, and exposed stall time. |
| Backend choice | Representation, execution plan, kernel specialization, and benchmark provenance. |

Start with a CLI/status endpoint and structured events; do not make a polished dashboard a prerequisite for validating the pager. A streaming inference API is a product goal, but its exact compatibility surface is an open choice.

Bind management locally by default. Require authentication and appropriate transport protection for remote access. Do not log prompts or KV contents by default. Protect spill files and make retention explicit. Diagnostic events should use opaque request IDs and aggregate routing information unless detailed capture is deliberately enabled.

## 14. Language, compiler, and build policy

### C++23 and Clang-first

The host runtime targets C++23. Use modern standard-library facilities where they improve clarity, with feature probes for the facilities actually required. This is not a commitment to follow every Chromium style restriction.

Clang is the primary compiler for code we own. NVCC remains the default CUDA compiler; use Clang as its host compiler where the validated native/cross configuration supports it. Localized GCC exceptions for a backend do not change the runtime's primary compiler. Current CUDA documentation lists Clang and C++23 support, but the pinned toolkit/compiler combination must pass our own integration tests. [CUDA installation/compiler support][cuda-install] · [NVCC][nvcc]

Use a pinned libstdc++ initially. Clang supports both libstdc++ and libc++; compiler choice and standard-library choice are separate. Avoid crossing incompatible C++ library/ABI boundaries. GCC's library status lists `flat_map` and `flat_set` starting at 15.1; this is a feature-availability reference, not an adopted toolchain pin. [Clang toolchain][clang-toolchain] · [Library status][libstdcpp-status]

Ordinary `.cc` files use the host compiler. CUDA-facing translation units stay narrow and can use a separately validated dialect when required by imported code. Do not expose heavy runtime containers through every CUDA header.

### Proposed engineering conventions

Use CMake presets, Ninja, `compile_commands.json`, and pinned LLVM formatting/analysis tools. Prefer LLD where validated. Use typed byte counts, spans/views, explicit error results, bounded queues, and move-only ownership wrappers. Select an exception policy deliberately; no exceptions may cross a C ABI.

C++ interfaces are appropriate within one build. Optional separately built backend modules should have a versioned C ABI with opaque handles and explicit descriptors; do not expose STL objects across it. This is an engineering boundary, not a licensing exemption.

GPU/I/O lifetime must be completion-aware. A host destructor is not evidence that submitted work is finished. A lease transfers into an in-flight record and becomes reclaimable only when all consumers complete.

## 15. Development host and Spark cross-build environment

### User-provided reference host

The user's `uname -a` output, preserved as supplied:

```text
Linux linux 7.0.0-31-generic #31~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Mon Aug 10 09:38:02 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
```

This establishes an x86-64 Linux reference host with an Ubuntu-packaged kernel. The userspace distribution/version is **not yet confirmed**. Read `/etc/os-release` before selecting exact APT repository versions or a matching container base. Do not infer the userspace release from the kernel string.

Treat this machine as the initial supported development baseline; supporting older environments is not a priority. Do not impose a hard kernel-version check unless a required feature actually needs it. Feature/capability checks are preferable to an arbitrary version barrier.

### Cross-development layout

```text
x86-64 host tools
    + Clang native build -> fast local unit tests / tools
    + Clang AArch64 target -> Spark CPU objects
    + NVCC cross setup -> Spark GPU code and ARM host objects
    + pinned ARM sysroot and libraries -> deployment package
                                              |
                                             SSH
                                              |
                                     Spark execution/tests
```

Clang supports target triples and sysroots. NVIDIA supplies x86-to-ARM cross-development packages, including `cuda-cross-sbsa`. Cross-compilation requires target headers/libraries and explicit compiler configuration; a target triple alone is not a complete SDK. [Clang cross-compilation][clang-cross] · [CUDA cross-development][cuda-install]

Explicitly configure CMake's target system, processor, sysroot, compiler target, find-root policies, and CUDA host compiler before enabling the language. `CMAKE_CUDA_HOST_COMPILER` selects the CPU compiler used by NVCC; in this build it must generate ARM code. The full cross combination remains a bootstrap validation item. [CMake toolchains][cmake-toolchains] · [CUDA host compiler][cmake-host-compiler]

Use explicit CPU/GPU targets, not workstation autodetection or `-march=native`. Verify the toolkit's supported GB10 architecture spelling and required feature suffixes for each backend; general compute capability does not imply every newer GPU kernel is supported. The workstation need not have a matching GPU just to perform offline compilation. [NVCC][nvcc]

### Local work versus target work

Keep editing, indexing, ordinary builds, static analysis, and CPU tests on the workstation. Run ARM concurrency tests, VMM validation, actual kernels, distributed tests, and tuning on Sparks. Remote debugging/profiling is part of target testing, not a requirement to host the development desktop there. NVIDIA documents SSH profiling for Spark. [Spark profiling][spark-optimization]

ARM memory ordering can expose synchronization mistakes hidden by x86 execution. Include ARM runs early; use the C++ memory model rather than hardware-specific assumptions. [ARM ordering][arm-ordering]

Keep model artifacts on the target between runs. Deploy changed executables, libraries, kernel artifacts, and tests rather than copying checkpoints repeatedly. Retain an optional native ARM build as a diagnostic comparison, not the primary workflow.

## 16. Declarative setup and toolchain provisioning

### Selected direction

Use a checked-in `mise.toml` and `mise.lock` for tool setup, environment selection, and tasks. Use project-owned provisioning for the complete LLVM/CUDA/AArch64 SDK where a generic tool declaration is insufficient. Provide both native Ubuntu setup and a reference development container using the same provisioning logic.

Mise documents APT package declarations and resolved tool lockfiles. Exact APT package versions must still be available in configured repositories; a tool lockfile is not a snapshot of the whole OS. Pin a mise release that supports the configuration syntax actually used. [Mise APT][mise-apt] · [Mise lockfile][mise-lock]

### Proposed files and responsibilities

| File | Purpose |
|---|---|
| `mise.toml` | Tools, setup/check tasks, controlled environment, basic host prerequisites. |
| `mise.lock` | Exact resolutions for supported mise-managed tools/platforms. |
| `toolchains/manifest.toml` | Validated LLVM, C++ library, CUDA, target ABI, kernel targets, and SDK component identities. |
| `toolchains/artifacts.lock.json` | URLs/repository identities, versions, checksums, and redistribution/license metadata. |
| `tools/setup-toolchain` | Idempotent installation/provisioning; explicit privileged actions. |
| `tools/check-toolchain` | Native C++23, cross-compile, link, architecture, dependency, and CUDA checks. |
| `cmake/toolchains/` | Native and Spark toolchain definitions/wrappers. |
| `CMakePresets.json` | Repeatable configure/build/test combinations. |
| `.devcontainer/` | Reference container and editor integration without a second package-policy source. |
| `dev` | Small convenience entry point for setup, build, tests, deployment, and diagnostics. |

These files are a plan, not files created by this document. Toolchain pins are intentionally not invented here. Choose them after one end-to-end Clang/C++23/native/ARM/CUDA smoke test succeeds.

### Setup contract

The setup command should preview machine changes, verify artifacts, install missing components, configure the project environment, and verify the result. It must be safe to rerun and must not silently change system-default compilers, update GPU drivers, accept model licenses, disable security controls, or download optional copyleft backends without the selected build profile allowing them.

Use APT for basic Ubuntu prerequisites and suitable signed SDK repositories. Snap is not a requirement. Prefer project-local SDK installs or a container for components needing strict pins. A development container can be driven through the Dev Container CLI; it need not require one editor. [Dev Container tools][devcontainers]

A proposed contributor experience, **to implement**, is:

```bash
./dev setup --dry-run
./dev setup
./dev doctor
./dev build native-debug
./dev test unit
./dev build spark-release
./dev deploy --target spark-a
./dev test --target spark-a gpu-smoke
./dev test --cluster sparks distributed-smoke
```

`doctor` should report the exact compiler/library/toolkit/SDK identities and actionable mismatches. Build-only setup should not require a local NVIDIA GPU or target SSH access. Remote tests should require an explicit target selection.

### Reproducibility and CI

Pin the reference container by digest and archive required SDK artifacts or repository snapshots where appropriate. Record the target driver separately from toolkit and library versions. Check that the target provides the needed glibc, C++ ABI symbols, and driver compatibility; do not assume a workstation-built library will run on the Spark.

Separate the **toolchain manifest** from the still-to-be-selected **C++ source dependency mechanism**. Mise is not being chosen as a replacement for every library dependency tool. The latter decision remains open.

Never execute an ARM build-time generator on x86 by accident. Produce host tools separately. CPU-only configuration should not require configuring every optional CUDA/backend dependency.

## 17. Licensing and provenance strategy

### Project intent

All llmpalooza-authored code is intended to be open source. The preferred proposal is a permissively licensed independently useful core, with optional copyleft components and accurately licensed combined builds. **The final original-code license is not selected by this document.** Apache-2.0 is a candidate, not an already-applied license.

The initial acceleration stack may require vendor-provided CUDA components. “Open-source llmpalooza” does not claim every driver, SDK, model weight, or tool in the deployment is open source. Track their separate terms and redistribution conditions.

### Optional copyleft is acceptable, but removability must be real

Keep affected implementation code, adapters, generated code, importer support, and transitive dependencies identifiable. A fork must be able to rebuild a useful core without them. Losing a particular model format or optimization is acceptable; losing the entire scheduler or allocator is not.

Do not claim that a directory, shared library, C ABI, or process boundary automatically avoids copyleft obligations. A build combining covered code must comply with its applicable obligations. AGPL has network-interaction source requirements for covered modified programs, and Apache-2.0 has notice and modification requirements. Exact distribution and compatibility questions need review against the actual dependency graph. [AGPL text][agpl] · [Apache-2.0 text][apache]

Rewriting or translating copied code does not erase its provenance. Keep independently authored interfaces distinct from imported implementation. Use each project's code under its actual license; “reference” is not a relicensing mechanism.

### Initial source inventory

vLLM's repository license is Apache-2.0; llama.cpp and ExLlamaV3 carry MIT licenses. These repository-level checks are not substitutes for file and transitive-dependency review. [vLLM license][vllm-license] · [llama.cpp license][llama-license] · [ExLlamaV3 license][exllama-license]

The MiaAI-Lab references include AGPL declarations and retained upstream/file-specific licensing. Resolve provenance per imported component rather than treating the entire ecosystem as uniformly licensed. Their model checkpoints also require separate review. [GLM reference][mia-glm] · [DeepSeek reference][mia-deepseek] · [Qwen reference][mia-qwen]

### Proposed compliance mechanics

Maintain source URLs, immutable revisions, file-level SPDX identifiers, modifications, licenses, notices, and a software bill of materials. REUSE provides a useful convention for file-level declarations. Generated native binaries and kernels retain provenance and any applicable source obligations. [REUSE][reuse]

CI should include a **copyleft-components-disabled profile** that neither fetches nor includes optional copyleft source, headers, code generation, or binaries. Audit its dependency closure; passing a build flag alone does not establish a license outcome. Maintain an appropriately compliant full-feature profile separately.

Ship notices, build instructions, relevant source availability, and model/vendor notices for the actual release configuration. Audit license compatibility before merging a new backend, not only before release. Keep unknown or ambiguous provenance out of distributed builds until resolved.

## 18. Correctness, testing, and performance evidence

### Layered test strategy

| Layer | Tests |
|---|---|
| Native CPU | Catalog/aliasing, reservation accounting, eviction policy, cancellation, parsers, artifact integrity. |
| Deterministic simulated backend | Delayed/out-of-order completion, capacity exhaustion, disk errors, concurrent revocation/acquisition, deadlock scenarios. |
| Single-Spark CUDA | Reserve/map/load/verify/unmap/remap; leases across streams; staging lifetime; actual kernel dependency reads. |
| Model semantics | Intermediate outputs, routing IDs, attention/state updates, teacher-forced logits, sampling configuration. |
| Multi-model pressure | Partial eviction, repeat reuse, fairness, cache sharing, budget changes, long-prefill interference. |
| Two-node | Shard agreement, ordered collectives, one-rank pressure, failure/cancellation, no speculative reuse after timeouts. |
| Packaging/licensing | Clean setup, restricted-dependency build, notices, reproducible artifact and backend identities. |

A fake backend tests our logic; it does not establish real GPU synchronization or performance. Use pinned known-working engines as numerical references. Define tolerances and distinguish expected floating-point differences from correctness failures. Generated text alone is insufficient evidence.

### Mandatory pager invariants

- A consumer never touches absent or incompletely loaded backing.
- An extent is not remapped, overwritten, or reused until all old consumers and relevant registrations are retired.
- A cancelled request cannot cause late I/O to corrupt a newly assigned extent.
- Mutable content is preserved or deliberately invalidated; cache metadata never falsely advertises a valid object.
- Shared physical backing is counted once, including aliases and pool-held allocations.
- Acquiring leases and revoking eligibility cannot both succeed for conflicting generations.
- The runtime retains enough resources to complete or safely unwind admitted work.
- An unavailable node or unacknowledged completion is not evidence of reclaimed memory.

### Benchmark priorities

Record warm decode rate, inter-token latency distribution, bytes read/written per token, exposed routing/paging stalls, peak physical occupancy, mixed-model throughput, per-model waiting/fairness, and cold/partial-resume latency.

Separate cold storage, warm OS cache, and warm model residency. Report whether speculative decoding, quantization changes, or prefix hits contributed. Record full artifact, backend, toolchain, driver, hardware, and policy identities with every result.

A resident-hit path should be measured independently from a miss path. A cache with many tiny hits can still perform badly if a few misses transfer huge extents. Include bytes and stall time, not only hit ratio.

Never assume that native code automatically outperforms an existing engine. Compare numerical equivalence and end-to-end workload performance before promoting a backend or policy.

## 19. Phased implementation plan and acceptance gates

No stage below has a promised completion date. Each stage should leave a usable, testable result.

| Stage | Deliverable | Acceptance gate |
|---|---|---|
| **0 — Bootstrap** | Repository skeleton, license decision, declarative SDK setup, C++23/Clang native and Spark cross-build. | Clean host/container setup; native tests pass; ARM/CUDA smoke binary runs over SSH; exact pins recorded. |
| **1 — Resource core** | Catalog, reservation/lease state machine, deterministic fake backend, real VMM smoke harness. | Adversarial completion/cancellation tests pass; repeated map/load/evict/restore checks succeed on Spark. |
| **2 — One resident model** | Import a manageable model; native backend execution; tokenizer/state/sampling baseline. | Teacher-forced and intermediate comparisons against a pinned reference; bounded and explainable memory usage. |
| **3 — Partial retention** | Two persistent model contexts, shared local budget, partial quiescent-model eviction, basic status API. | Only selected extents are displaced; untouched data remains resident; resumption reloads only missing dependencies. |
| **4 — Demand-paged MoE** | Routing boundary, selected-expert leases, asynchronous misses and resumable tasks. | No unselected expert load except declared metadata/granularity/read-ahead; no expert substitution; resident/miss overhead measured. |
| **5 — Two Sparks** | Explicit sharding, coordinated admission, stable communication buffers, ordered collectives. | Both ranks remain correct under asymmetric memory pressure, cancellation, and controlled failure tests. |
| **6 — Performance and product** | Alternative compatible kernels/plans, prefetch, selective graphs, dashboard, packaging. | Measurable mixed-workload benefits; no numerical/lifetime regression; compliant optional-backend builds. |

The first model need not be the largest target. A small dense model plus synthetic/tiny MoE tests can separate execution, import, and pager bugs before introducing a complex flagship architecture.

Target model support is earned per checkpoint/configuration. Keep a matrix of unsupported, import-only, resident-correct, paged-correct, distributed-correct, and performance-validated states.

## 20. Proposed repository shape and internal interfaces

### Repository sketch

```text
llmp/
  ideation.md
  README.md
  LICENSE                         # chosen original-code license
  LICENSES/
  NOTICE
  REUSE.toml
  CMakeLists.txt
  CMakePresets.json
  mise.toml
  mise.lock
  dev
  .devcontainer/
  toolchains/
    manifest.toml
    artifacts.lock.json
  cmake/toolchains/
  include/llmp/
    resource.h
    residency.h
    execution.h
    backend.h
    model_artifact.h
  src/
    runtime/
    scheduler/
    memory/
    storage/
    execution/
    distributed/
    management/
  backends/
    reference/
    cuda/
    optional/                     # explicit licenses and dependency closure
  importers/
  tools/
  tests/
    unit/
    simulation/
    cuda/
    model/
    distributed/
  benchmarks/
  dashboard/
  third_party/                    # provenance, notices, pinned imports
  docs/
    decisions/
    design/
    support-matrix.md
    licensing.md
```

This is a layout proposal, not a commitment to individual filenames or a generated codebase. Backend licensing should be explicit; directory names alone do not establish legal isolation.

### Conceptual native API

The following is an interface sketch, **not compilable API code**. Types and async primitives remain to be designed.

```cpp
ResourceId register_resource(ResourceDescriptor descriptor);

ReservationResult reserve_capacity(
    TransactionId transaction,
    PhaseEnvelope envelope);

AcquireResult acquire_group(
    ReservationId reservation,
    DependencySet dependencies);

// Submission transfers lease ownership to completion tracking.
CompletionToken submit(
    ExecutionPlan plan,
    AcquiredLease lease,
    ExecutionContext context);

void retire_completed(CompletionToken completion);
ReclaimResult reclaim(ExtentSet candidates);
void cancel(TransactionId transaction);
```

`AcquireResult` needs ready, deferred, impossible, cancelled, and failed outcomes. Deferred results refer to owned continuations, not a blocked global scheduler. Completion must cover all declared consumers. `DependencySet` resolves aliases and unique extents before accounting. `reclaim` validates the selected generations and reports actual capacity recovered.

Do not freeze a public plugin ABI before the first real backend, paging path, and distributed phase expose their requirements. Keep internal interfaces narrow enough to evolve.

## 21. Open decisions, risks, and immediate bootstrap tasks

### Open decisions

| Decision | Required evidence or next action |
|---|---|
| Original-code license | Choose a compatible permissive core license and contribution policy; document optional copyleft build obligations. |
| Exact toolchain | Validate Clang + libstdc++ + CUDA + ARM sysroot as a unit; then lock versions and artifacts. |
| User-space baseline | Capture `/etc/os-release` on the workstation and both target nodes. |
| First model/backend | Choose a manageable, well-referenced vertical slice with clear provenance and usable kernels. |
| Source dependency manager | Decide how C++ dependencies are pinned/fetched independently of toolchain provisioning. |
| Artifact schema | Choose metadata encoding, layout ABI, alignment, integrity, and sharding representation. |
| Extent size and physical pool | Measure granularity, map/unmap overhead, fragmentation, and I/O amplification. |
| I/O implementation | Compare cuFile compatibility with native staged I/O under concurrent compute and pressure. |
| Async framework | Choose a small native task/completion approach; C++23 does not by itself provide our scheduler. |
| Initial reservation guarantee | Define conservative progress envelopes and admissible overlap before lending active capacity. |
| Graph integration | Prove correctness across restore/replay boundaries and quantify all-resident overhead. |
| Public API/dashboard | Select compatibility scope after a CLI/status path can explain real runtime behavior. |

### Highest risks

The main risks are kernel accesses beyond declared dependencies, graph/registration assumptions surviving backing changes, inaccurate peak-memory envelopes, repeated expert misses overwhelming storage, numerical mistakes in ports, and unremovable licensing dependencies.

The mitigations are explicit contracts, conservative first implementations, tracing, target-side stress tests, pinned numerical references, and a continuously tested reduced-dependency build—not claims that the abstractions eliminate the underlying constraints.

### Immediate bootstrap sequence

1. Establish the source repository, license policy, directory skeleton, and this document as the design baseline.
2. Inventory the workstation and Spark software/capabilities without changing drivers or security settings.
3. Pin a working C++23/Clang/native/cross/CUDA toolchain through an automated setup and verification path.
4. Implement the fake resource backend, budget/lease invariants, and a standalone CUDA VMM restore test.
5. Choose one model and backend, record provenance, and validate fully resident numerical execution before adding paging.

### Instructions for implementation work derived from this document

Treat declared decisions as the baseline and proposals as hypotheses to test. Do not silently revert to a vLLM-controlled process architecture. Do not invent compiler pins, measured throughput, supported model combinations, or license permissions. Preserve the native hot path while permitting build-time tools from other ecosystems. Add an architecture decision record when measurements require a material change.

> **Reserve the ability to run. Materialize only the dependencies that are needed. Keep useful contents until better work needs their capacity.**

## 22. Sources and verification notes

Sources below support external API/platform/license facts, not claims that llmpalooza has been implemented. Project requirements and proposed interfaces come from the design discussion. No supplied Spark or development host was accessed to compile, benchmark, or probe capabilities while preparing this document.

### Hardware, memory, and execution

- [NVIDIA DGX Spark hardware][spark-hardware] — shared memory and node hardware.
- [NVIDIA GPU compute capabilities][cuda-gpus] — GB10 GPU target identification.
- [DGX Spark optimization guide][spark-optimization] — UMA accounting and target profiling.
- [DGX Spark / GB10 FAQ][spark-faq] — GPUDirect RDMA caveat and fallback guidance.
- [CUDA VMM guide][cuda-vmm-guide] — explicit allocation/mapping and platform capabilities.
- [CUDA VMM API][cuda-vmm-api] — granularity, mapping, access, and handle lifetime contracts.
- [GDS release notes][gds-release] — Spark compatibility-mode limitation.
- [GDS overview][gds-overview] — explicit storage transfers and supported allocation types.
- [CUDA graphs][cuda-graphs] — captured operations and execution boundaries.
- [CUDA execution API][cuda-execution] — host-function constraints.
- [NCCL user buffer registration][nccl-registration] and [thread safety][nccl-threading] — communication integration requirements.
- [Hugging Face cache explanation][cache-explanation] and [vLLM prefix caching][prefix-caching] — state reuse and exact-prefix identity.

### Implementation references

- [GLM Spark recipe][mia-glm], [DeepSeek Spark recipe][mia-deepseek], and [Qwen Spark recipe][mia-qwen] — user-selected target references; pin before reuse.
- [vLLM modular MoE source][vllm-moe] — dispatch/workspace/capability reference.
- [FlashInfer documentation][flashinfer] and [CUTLASS source overview][cutlass] — candidate numerical infrastructure.
- [Triton AOT compiler source][triton-aot] — optional build-time kernel route.

### Toolchain and development setup

- [CUDA Linux installation guide][cuda-install] and [NVCC documentation][nvcc] — compiler and cross-development support.
- [Clang cross-compilation][clang-cross] and [complete toolchains][clang-toolchain] — targets, sysroots, standard libraries, and ABI.
- [libstdc++ implementation status][libstdcpp-status] — feature availability independent of the language flag.
- [CMake toolchains][cmake-toolchains] and [CUDA host compiler selection][cmake-host-compiler] — explicit cross-build configuration.
- [ARM memory ordering][arm-ordering] — target concurrency validation.
- [Mise APT bootstrap][mise-apt] and [lockfiles][mise-lock] — declarative setup, with repository-retention limits.
- [Dev Container supporting tools][devcontainers] — optional reference development environment.

### Licensing

- [Apache-2.0 text][apache] and [AGPL-3.0 text][agpl] — governing license texts to review against actual builds.
- [vLLM license][vllm-license], [llama.cpp license][llama-license], and [ExLlamaV3 license][exllama-license] — initial repository-level checks.
- [REUSE specification][reuse] — file-level provenance conventions.

[spark-hardware]: https://docs.nvidia.com/dgx/dgx-spark/hardware.html
[cuda-gpus]: https://developer.nvidia.com/cuda/gpus
[spark-optimization]: https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html
[spark-faq]: https://forums.developer.nvidia.com/t/dgx-spark-gb10-faq/347344
[cuda-vmm-guide]: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html
[cuda-vmm-api]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html
[gds-release]: https://docs.nvidia.com/gpudirect-storage/release-notes/index.html
[gds-overview]: https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html
[cuda-graphs]: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
[cuda-execution]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html
[nccl-registration]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html
[nccl-threading]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
[cache-explanation]: https://huggingface.co/docs/transformers/main/cache_explanation
[prefix-caching]: https://docs.vllm.ai/en/latest/design/prefix_caching/
[mia-glm]: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks
[mia-deepseek]: https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks
[mia-qwen]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks
[vllm-moe]: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/fused_moe/modular_kernel.py
[flashinfer]: https://docs.flashinfer.ai/
[cutlass]: https://raw.githubusercontent.com/NVIDIA/cutlass/main/README.md
[triton-aot]: https://raw.githubusercontent.com/triton-lang/triton/main/python/triton/tools/compile.py
[cuda-install]: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
[nvcc]: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
[clang-cross]: https://clang.llvm.org/docs/CrossCompilation.html
[clang-toolchain]: https://clang.llvm.org/docs/Toolchain.html
[libstdcpp-status]: https://gcc.gnu.org/onlinedocs/libstdc++/manual/status.html
[cmake-toolchains]: https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html
[cmake-host-compiler]: https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_HOST_COMPILER.html
[arm-ordering]: https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/memorder.html
[mise-apt]: https://mise.jdx.dev/bootstrap/packages/apt.html
[mise-lock]: https://mise.jdx.dev/dev-tools/mise-lock.html
[devcontainers]: https://containers.dev/supporting
[apache]: https://www.apache.org/licenses/LICENSE-2.0
[agpl]: https://opensource.org/license/agpl-3.0
[vllm-license]: https://raw.githubusercontent.com/vllm-project/vllm/main/LICENSE
[llama-license]: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/LICENSE
[exllama-license]: https://raw.githubusercontent.com/turboderp-org/exllamav3/master/LICENSE
[reuse]: https://reuse.software/spec-3.3/
