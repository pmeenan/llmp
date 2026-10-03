<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Architecture

> **Status: first full draft, approved by the owner on 2026-09-23.** No application
> code exists yet; this describes the system that M1–M10 build. It is the map:
> processes, components, layers, the main flows and the boundaries between
> them, with links to the designs that govern each area. Settled choices live
> in [decisions.md](decisions.md). Where this document disagrees with a
> linked design or a decision, they govern and this document is the one to
> fix. Nothing here is a measurement unless it cites one, and open items are
> collected [at the end](#open-architecture-questions). The original reasoning
> and source links live in [ideation.md](ideation.md); section numbers (§)
> refer to it. The workstation and Spark inventories are in
> [environment.md](environment.md).

## Where the detail lives

| Area | Governing design | Decisions |
| --- | --- | --- |
| Task states, service lanes, completion ownership | [async-model.md](async-model.md) | D-048 |
| Capacity reservations, admission and progress | [reservation-policy.md](reservation-policy.md) | D-007, D-050 |
| Conversation-state retention and the M6 workload | [retention-policy.md](retention-policy.md) | D-024, D-031, D-055 |
| Prepared artifacts and page-in | [artifact-format.md](artifact-format.md) | D-009, D-018, D-035, D-056 |
| Kernel dispatch and the M2 backend proof | [backend-proof.md](backend-proof.md), [exl3-bringup.md](exl3-bringup.md), [first-slice.md](first-slice.md) | D-051–D-053 |
| Serving in the runtime: the engine, the configured models, the full swap and the serving commands (M3) | [runtime-serving.md](runtime-serving.md), [swap.md](experiments/fast-swap/swap.md) | D-086, D-090, D-093, D-096 |
| The engine: the paged node, the runner skeleton (weights, live state and spill, planned shapes, graph runs, speculation's snapshot, resources) and how a model family plugs in | [engine.md](engine.md) | D-053, D-090, D-093, D-096 |
| Other GPU platforms and operating systems: boundaries, the registry's primitive fallbacks, the runners' shared skeleton, distribution | [portability.md](portability.md) | D-026, D-053, D-082, D-098 |
| Cluster membership, transport and placement | [cluster-design.md](cluster-design.md) and [the conductor section](#conductor-ownership-and-admission) | D-037–D-039 |
| Inference API contract | [client-api-baseline.md](client-api-baseline.md) and the assessments it links | D-040–D-047 |
| Source dependencies and licensing | [source-dependencies.md](source-dependencies.md), [licensing.md](licensing.md) | D-017, D-057 |
| Toolchain, checks, versions, installed layout | [Build, repository and installation](#build-repository-and-installation) | D-032, D-049, D-058–D-063 |

## Fixed points (from decisions)

- **Ownership split.** Model implementations describe computation and
  dependencies; jitLLM owns storage, residency, scheduling, and execution
  lifetime. One native process per node (D-005).
- **Primary workload.** One user switching among a library of models larger
  than memory; an agent plus subagents on different models; conversations
  spanning minutes to hours; time-sliced under contention, concurrent when
  a supported placement fits each node's complete execution budget. Headline
  metrics are switch latency, switch-back with conversation state preserved,
  and decode parity with an all-resident run. A switch request signals intent;
  the scheduler establishes a completed handoff boundary before releasing
  residency leases or reclaiming backing (D-007, D-019).
- **Cluster coordination.** A single conductor, the cluster's one point of
  entry, places models, or parts of models, per node; routes requests to a
  node running the model; admits work cluster-wide. Placement is the first
  multi-node capability and is preferred over paging when it suffices;
  sharding follows for the flagship. Topology is discovered or configured at
  runtime, never baked into the application, and a busy small model may run
  as replicas across nodes (D-020, D-023). The conductor is an in-process
  role on the designated node; its cluster view cannot grant local capacity
  or bypass a node's admission authority (D-037).
- **Switching bar and API baseline.** Never worse than a full swap; seamless
  is the goal, validated against a measured reference cycle (D-021, D-025).
  Standard web-API clients work unmodified; the model field drives switching.
  Prefix matching enables bounded state reuse; it identifies neither a
  conversation nor its lifetime. Shared prompt prefixes and conversation
  continuations have independent reuse/expiry policies (D-024, D-031),
  capacity-driven with a 24-hour idle cap under D-055's
  [retention policy](retention-policy.md), which also names M6's acceptance
  workload. Sessions and hints are optional extensions (D-022). M5 serves Chat
  Completions, Responses and Messages with model listing and token counting
  under the [D-040 client contract](client-api-baseline.md); client compatibility
  requires per-version execution evidence, including Cursor reachability.
  D-041 adds discovery, targeted continuation close, download/warm jobs and
  a tested Ollama subset; [scope and design guidance](api-capabilities.md) keep
  model residency, conversation retention and artifact installation distinct.
  D-042 confirms staged file modalities, separate MCP management, single-owner
  sharing controls and later embeddings; delivery/evidence remain ahead.
  D-043 requires direct compatibility profiles tested with unmodified tooling
  and confirms tokenization/rendering, constrained output and reasoning contracts.
  D-044 adds compatible reranking, monitoring and raw Completions/token
  diagnostics; specialized runtime/deployment controls stay outside the baseline.
  D-045 fixes the front-door listener, auth and CORS defaults, the admission
  status and keepalive rules, and the standard-client signals the baseline
  honors. D-046 adopts OpenRouter's model-metadata, reasoning and hint
  spellings on the OpenAI-shaped routes ([assessment](openrouter-api-assessment.md))
  and excludes its hosted-routing features. D-047 corrects reasoning wire
  formats, rejects unsupported Responses storage, and distinguishes SSE
  keepalives from non-streaming JSON outcomes.
- **Vocabulary.** *Virtual reservation* = address space. *Capacity
  reservation* = admission commitment under a progress policy. *Residency
  lease* = protection of specific backing while consumers run. Never
  "reserved" unqualified (D-007). Likewise: read-only ≠ always resident;
  unleased ≠ may be lost; reconstructible ≠ already stored in the right
  format; resident ≠ available without admission (§4).
- **Memory model of the target.** One Spark = one 128 GB unified budget. Two
  Sparks = two domains over a network. Weights and state live in device
  VMM, which the GB10's L2 caches; validated Spark storage reads directly
  into a bounded host-VMM landing zone, and the GPU copies each extent into
  device VMM, with no CPU payload copy (D-004, D-034, D-081).
- **Explicit VMM plus catalog.** Driver-API VMM; accessing absent backing is
  a bug; every allocation registered; unknown allocations non-evictable;
  typed IDs and generations, raw pointers only at the backend boundary
  (D-006).
- **Paging semantics.** Extent-level cross-model eviction; selected experts
  acquired on demand; no substitution; release ≠ eviction; lazy commitment
  (D-007, D-008).
- **Task and completion ownership.** Explicit native task states, one
  scheduler/catalog writer per node, bounded provider services and owned
  completion records (D-048). Cancellation/client termination cannot retire
  unfinished consumers or registrations; [design](async-model.md).
- **Capacity progress.** D-050 grants bounded requests their maximum retained
  state/growth plus a complete phase envelope; initially one active request
  per node, holding the slot through all its phases, waits and unwind.
  D-069 makes switching a configurable policy: a request may be paused only
  at a completed phase boundary, with its retained state kept resident, and
  the default is priority-aware (same class runs to completion; interactive
  pauses background). Concurrent requests require their full envelopes to
  fit. [Policy and validation cases](reservation-policy.md).
- **Artifacts.** Prepared, versioned, hashed, atomically published; no
  process state serialized; checkpoints untrusted. Initial encoding and
  layout are experimental; compatibility guarantees require dense and MoE
  import/execution/eviction/restore evidence (D-009, D-018). Import repacks
  weights into contiguous dependency groups, 4 KiB-aligned in safetensors
  shards and paged in 2 MiB group-relative chunks, with an expert/tensor
  index ([v0 format](artifact-format.md); D-035, D-056).
- **Storage roles.** The runtime pages only from each node's installed store,
  a local block-device filesystem that must pass the direct-I/O probe.
  Source checkpoints and an optional prepared-artifact archive may live on
  an optional long-term store (a network mount or USB drive) that only
  import/install jobs touch. A cluster imports once and replicates the
  verified artifact to peers over the cluster link. Installs never delete
  models implicitly; the user chooses what to archive or delete when space
  is short (D-054).
- **Dependency policy.** Own code Apache-2.0; incorporated core implementation
  uses Apache-2.0 / BSD / MIT / MPL-2.0, with other implementation licenses in
  optional, fully removable modules. Declared tools and platform dependencies
  have separate terms and remain in the build audit (D-003, D-017).
- **Language and build.** C++23, Clang-first (LLVM 22.1.8), NVCC with Clang
  as host compiler, and a statically linked, source-built GCC 16.2 C++
  runtime; cross-built from x86-64 and tested on Spark over SSH; declarative
  pinned toolchain (D-010–D-012, D-032, D-060). D-049 selects a complete,
  persistent project SDK with declared system prerequisites, project-scoped
  tool selection and shared workstation/reference-container provisioning.
  D-058 and D-059 pin CMake, Ninja, GoogleTest and the LLVM developer tools;
  D-061 replaces hosted CI with local `check`, `check:full` and
  `check:spark` tiers (`tools/check`, the `mise run check` tasks).
- **Errors.** jitLLM code builds with `-fno-exceptions`. Fallible operations
  return `std::expected` values, and only a violated invariant or failed heap
  allocation is fatal (D-066; [errors and faults](#errors-faults-startup-and-shutdown)).
- **Source dependencies.** D-057 selects [locked CMake FetchContent acquisition
  and curated vendoring](source-dependencies.md) for adapted source units,
  separate from the SDK. Select the audited profile closure before fetching;
  configure/build uses verified local inputs. The lock, preparation, the
  configure checks and the build receipt are in [third_party/](../third_party/README.md);
  source adoption gates remain open.
- **First model/reference.** D-051 selects the official Qwen2.5-0.5B-Instruct
  FP16 GGUF and pinned llama.cpp CUDA reference on Spark, with CPU diagnostics;
  [identities and numerical contract](first-slice.md). GGML supplies operations,
  while jitLLM owns execution/backing and, under D-053, dispatch: kernels are
  swappable build-time implementations selected per operation and plan, from
  several sources at once. D-052 adds a required
  [small EXL3 companion and upstream performance gates](exl3-bringup.md)
  before M2 closes. Both external references have run; the
  [EXL3 baseline](experiments/exl3-reference/README.md) includes real and larger
  synthetic packed projections. Native support remains unvalidated for both.
- **Chat rendering.** A template renders through a native family renderer,
  chosen by the template's hash (checked against golden fixtures) or by
  probe equivalence, or otherwise through a bounded, sandboxed
  Jinja-subset interpreter in the runtime that marks only the control
  tokens the template's own text placed (D-067 as amended 2026-10-02).
- **Model shapes.** The resource core names no model architecture. Adapters,
  phase kinds, state capabilities, decoding modes and operations carry
  shape-specific behavior with declared bounds. Speculative (MTP) and
  block-diffusion decoding are designed for now; speculation executes from
  M3 and block diffusion in M9 (D-068, D-087;
  [model shapes](#model-shapes)).
- **Portability posture.** NVIDIA first. The core holds no vendor types;
  device memory, paging, and transport sit behind narrow provider interfaces
  with CUDA VMM as the only implementation for now; platform properties are
  probed capabilities. Apple silicon and AMD single machines are possible
  later targets; nothing is done or sacrificed for them now (D-026).
- **Distribution.** Users install from a signed apt repository, Spark first;
  the developer toolchain path is separate (D-027, D-012). The
  [installed layout](#installed-layout), service user and unit are settled
  in D-063; product and surface versioning in D-062.

### Mandatory pager invariants (§18)

Owner-stated; every one gets tests in the simulated backend and on hardware.

1. A consumer never touches absent or incompletely loaded backing.
2. An extent is not remapped, overwritten, or reused until all old consumers
   and relevant registrations are retired.
3. A cancelled request cannot cause late I/O to corrupt a newly assigned
   extent.
4. Mutable content is preserved or deliberately invalidated; cache metadata
   never falsely advertises a valid object.
5. Shared physical backing is counted once, including aliases and pool-held
   allocations.
6. Acquiring leases and revoking eligibility cannot both succeed for
   conflicting generations.
7. The runtime retains enough resources to complete or safely unwind admitted
   work.
8. An unavailable node or unacknowledged completion is not evidence of
   reclaimed memory.

### Reservation progress gate

D-050 settles question 9's initial [reservation policy](reservation-policy.md).
Guaranteed requests have finite context/output and plan bounds. Fixed
overhead, all admitted retained state and its maximum growth, non-revocable
background work, and the largest complete phase envelope must fit the node's
execution budget after OS/external headroom. Initially one client-facing
request owns the execution slot through all its phases, I/O suspension and
completion-aware unwind. The node switches requests or models at a phase
boundary only when D-069's switching policy pauses the running request
there. Concurrent cohorts additionally need all their full phase
envelopes to fit. Admitted
state retains its in-memory allowance even when spilled. Opportunistic work
cannot invalidate existing guarantees. Commitments stay separate from actual
occupancy, and grants never eagerly evict cache.

M2 must execute the policy's adversarial matrix, including request-boundary
switching, competing suspended phases, future state growth, impossible phases, spill/reclaim saturation,
fragmentation, sharing, envelope upgrades and cancellation with late I/O.
Assert bounded occupancy and eventual completion or explicit safe failure;
unknown completion remains charged and faults affected admission. The real
GGML/EXL3/VMM proof must validate allocation envelopes and lifetime assumptions.
M6 adds retention/concurrent-execution evidence and M7 routed expert closures.
The design decision is complete; merely avoiding OOM is not implementation
proof of progress.

## System overview

### Processes (§3)

```text
standard clients (Cursor, OpenCode, Codex, Claude Code, Ollama-native, ...)
        |  Chat Completions, Responses, Anthropic Messages, Ollama subset
        v
front door on the conductor node (loopback by default; TLS when exposed)
        |
        v
jitllm runtime: conductor role + node role (one process)
        | routed attempts over mTLS sessions (M6a)   | collectives (M4)
        v                                             v
jitllm runtime: node role, one process on each other enrolled node

jitllm CLI, later dashboard -> management listener (conductor or standalone)
import / install / archive jobs <- started and supervised by their runtime
topology: detected paths, enrolled membership; no fixed names or counts (D-038)
```

| Process | Runs as | Does | Never |
| --- | --- | --- | --- |
| Runtime (`/usr/libexec/jitllm/`, `jitllm.service`) | `jitllm` | All local execution, scheduling, memory policy, VMM, storage and completion tracking (D-005); the front door, management listener and conductor role on the designated node | Opens the long-term store (D-054) or runs checkpoint code |
| Import, install and archive jobs | `jitllm` | Download, stage and import sources; verify and publish artifacts; archive; replicate to peers (D-054) | Touches scheduler state or the hot path; runs beside a conflicting job for the same model or artifact |
| `jitllm` CLI | The invoking user | A management client over loopback HTTP (D-064) | Reads runtime state files or links runtime code |
| Setup tooling | An administrator | Enrollment, cluster documents, credentials, node identity (D-038, D-063) | Changes membership or trust while nodes serve ([cluster-design.md](cluster-design.md)) |
| Dashboard (M10) | Its own service | Browser UI that calls the management API from its server side (D-064) | Takes runtime locks or exposes the management API to the browser |
| Certificate helpers | root | Keep front-door certificate files current: the certbot deploy hook and the Tailscale timer (D-065) | Hold any jitLLM authority |

### Runtime components

| Component | Responsibilities | Runs on |
| --- | --- | --- |
| Front door | Listeners, TLS, request guards and credentials, protocol adapters, SSE and keepalives, bounded intake and output (D-045, D-065) | Network lane |
| Conductor role | Placement, routing and the advisory cluster view; admission stays with each node (D-037) | Network lane, conductor node only |
| Management plane | Management API, job supervision, configuration, status, the admission what-if query, trace capture (D-064) | Network lane; changes reach the scheduler as commands |
| Model registry | Installed artifacts, aliases, tokenizer and template identities, model contexts and their plans, lifecycle | Records owned by the scheduler |
| Execution planner | Executable plans: an implementation per operation, dependency closures, workspace, phase envelopes (D-050, D-053) | CPU workers; the scheduler validates and installs the result |
| Scheduler | Task states, admission, the execution slot and the switching policy, fairness, phase progression, retirement (D-048, D-050, D-069) | Scheduler thread, the single writer |
| Resource catalog | Every managed resource: storage, backing, generations, aliases, leases, claims, statistics | Scheduler thread |
| Memory manager | Commitment and occupancy ledgers, victim selection, backing handoff | Scheduler thread (policy code) |
| Retention cache | Continuation and shared-prefix entries, their identity index, claims, expiry and spill decisions (D-055) | Scheduler thread |
| Storage service | Direct-file reads and writes, spill files, completion harvesting (D-034) | Storage lane |
| Device services | VMM calls, launches in plan order, fence recording; separately, fence observation | Device submission and device completion lanes |
| Kernel implementations | Build-time units behind the operation contract (D-053) | Launched by the dispatcher on the device submission lane |
| Tokenizer, renderer, output parser | Requests in, token IDs and segment boundaries out; generated tokens in, text, tool calls and reasoning out (D-067) | CPU workers |
| Telemetry | Structured events, metrics, trace capture, all in bounded buffers | Written by every lane; read through snapshots |

"Monolithic" means one authority over local execution state, not one thread
and not a global lock held during I/O. Understandable queues first; lock-free
only where measured. One deliberately managed CUDA context per GPU initially;
application-level model contexts are separate from CUDA contexts.

### Layers and dependency rules

Modules form a directed acyclic graph: a module includes only modules in
lower layers, or modules listed before it in its own layer. Tests may
substitute a fake at any provider boundary.

| Layer | Modules | Holds |
| --- | --- | --- |
| Base | `base` | Typed IDs and generations, typed byte counts, `Error` and `std::expected` helpers (D-066), checked arithmetic, bounded containers and queues, monotonic clocks |
| Platform | `platform` | Linux services: files opened beneath a directory without following links, direct-I/O opens, io_uring, the event loop, wakers and signal watch, socket calls, processes, memory counters. With the Linux providers, the only place Linux-specific code appears ([portability.md](portability.md)) |
| Providers | `providers`, `providers/fake`, `providers/cuda` | The [provider interfaces](#providers) and their implementations; with kernel units, the only place vendor headers appear (D-026) |
| Resource core | `catalog`, `memory`, `retention`, `scheduler` | Resources, ledgers, victim selection, retention, tasks and admission |
| Model | `tokenizer`, `chat`, `artifact`, `model`, `execution` | The tokenizer; the chat renderers and, later, output parsers; the artifact reader and verifier; architecture and state adapters; the operation contract, planner and dispatcher, and sampling |
| Kernels | `kernels/<source>` | Build-time implementations of operations: `ggml` and `exl3` first (D-053); jitLLM's own `image` kernels, and `paging`, the engine's fill and row-gather kernels |
| Engine | `engine` | The paged node, which composes the providers, the resource core and its lanes on one GPU; the runner skeleton every model is built from: weights as extents, live state with its spill and a verify's snapshot, planned shapes and their cache, runs and decode graphs, the runner's own device resources; and each model family's runner, which adds its plans, state layout and steps (D-096, [engine.md](engine.md), [runtime-serving.md](runtime-serving.md)). It reaches the device only through the providers, the device runtime among them, and the kernels, and is built without CUDA's headers; CUDA builds only, because it links the kernel modules |
| Services | `config`, `api`, `cluster`, `management`, `jobs` | The node's configuration, storage roles and served models (D-073, D-096); protocol adapters, conductor and sessions, the management API, job processes |
| Programs | `runtime`, job executables, `cli`, `tools` | Process wiring, startup and shutdown; the import, install and archive processes; the CLI; build and diagnostic tools |

- The resource core and the model layer hold no vendor types. The CPU-only
  build compiles and tests every module except the CUDA provider, the CUDA
  kernel units and the engine, with no CUDA SDK visible (D-026, D-061's
  `check` tier); its runtime refuses the serving commands.
- CUDA translation units stay narrow. They include kernel sources and
  provider code, never scheduler or catalog headers (AGENTS.md rule 6).
- Vendor and OS code keeps to its modules: CUDA only in `providers/cuda`
  and the kernel modules, Linux-specific headers and calls only in
  `platform` and the Linux providers (`providers/uring_storage.*`). The
  light check tier's boundary check (`tools/jitllm_boundaries.py`) fails
  on anything else, and allows a named exception only with its reason
  ([portability.md](portability.md)).
- A build-generated table registers the compiled implementations, and the
  `runtime` program links it. The execution layer never includes a kernel
  module, and nothing is loaded at run time (D-028).
- An implementation under a license the core does not admit (copyleft;
  the core admits any permissive license, D-091) lives in its own optional
  module with its own license and CMake option. jitLLM's own
  builds and packages include it by default (D-080); the copyleft-disabled
  profile excludes it before any source is fetched (D-057), and the
  registry then never sees it.
- Headers are internal and live beside their sources. No public header tree,
  C ABI or linkable library ships (D-064).
- Only programs wire services together. Library code starts no threads and
  opens no files during static initialization. Tests wire fakes the same
  way.

### Threads and lanes

D-048 fixes the thread roles ([async-model.md](async-model.md)). The
scheduler thread is the node's single writer. It owns task states, the
catalog, both ledgers, leases and claims, retained entries and retirement
decisions. Other lanes send it bounded commands and completion
observations; they never change its records or run a continuation inline.

| Lane | From | Work |
| --- | --- | --- |
| Scheduler | M2 | Bounded batches of observations and ready tasks per turn; never blocks on disk, GPU, network or a full queue |
| Storage | M2 | Direct-I/O submission and completion harvesting |
| Device submission | M2 | Ordered copies and launches on owned streams (and VMM calls when there is no VMM lane) |
| VMM | M2 | Creating, mapping and releasing managed backing, so a page-in's copies never wait behind it; keeping it for a handoff (M3) |
| Copy | M3 | The landing zone's copies, in and out, on the zone's stream, so a job blocked launching into a full stream (RE-029) never holds them up |
| Device completion | M2 | Fence queries, independent of any blocking submission call |
| CPU workers | M2 | Plan preparation, hashing and verification, then rendering and tokenization in M3; other long host work |
| Network | M3 (loopback endpoint), M5 (front door), M6a (cluster) | Listeners, TLS, protocol parsing and writing, cluster sessions, backpressure |

CPU workers keep reserved capacity for admitted requests' work, such as
spill-restore verification. Intake work such as rendering and token counting
draws on bounded intake credits and cannot starve it. Readers such as
discovery, status and the management API use bounded snapshots that the
scheduler publishes, or ask it through a command. A request's output travels
through a bounded per-request buffer. A phase that can emit n tokens, such as
a verify or a canvas commit, starts only when that buffer has room for their
worst-case wire bytes, which are part of its output credits; the buffer is
sized at admission to hold at least the largest phase's worst-case output.
When the buffer fills, the scheduler stops production at the next completed
boundary, while completion harvesting continues (D-048). Worker counts, queue
sizes and polling intervals are M2 settings to measure. A sleeping thread
wakes slowly on the Spark (RE-017), so the scheduler, storage and device
submission lanes poll for a bounded window (200 µs, not tuned) after their
last progress before they sleep; page-in through the zone reached the
in-place reads' bandwidth only with them
([pagein-perf](experiments/pagein-perf/README.md)). Between a request's
steps they wake by anticipation (D-094,
[runtime-wake](experiments/runtime-wake/README.md)): the device completion
lane sleeps through most of a step and spins only around its likely ends,
waking the scheduler and the submission lane ahead of the completion, and
the scheduler polls after a step for about as long as its client takes to
ask for the next. That removed the 0.26–0.65 ms a step the sleeping lanes
cost, at about a tenth of a core while stepping and none while idle, where
polling through every step (the paged harness's old 100 ms window) took
four cores. Durations use monotonic clocks.

## Request path

One inference request on a single node, with the conductor role local. Each
step names the owner of its decisions; the linked designs govern the details.

1. **Accept (network lane).** The front door accepts the connection,
   completes TLS where configured (SNI selects the certificate, D-065) and
   applies D-045's guards: `Host`, `Origin`, content type, and the credential
   when one is required. It reads headers and body within fixed bounds.
   Before accepting the body it reserves intake memory from a node-wide
   intake pool: the larger of a fixed per-request minimum and a fixed
   multiple of the declared decoded size, or of the per-request cap when no
   size is declared (chunked or compressed bodies). Parsing, rendering and
   tokenization allocate only from that reservation, and input that would
   need more fails with a 413. An executing node reserves the same
   allowance from its own pool when it receives a routed request.
   cluster-design.md's initial values size the pool's decoded-body share,
   and M5 sets the multiple and the pool size
   ([client-api-baseline.md](client-api-baseline.md#shared-correctness-and-limits),
   [cluster-design.md](cluster-design.md#internal-messages-and-bounded-state)).
2. **Normalize (network lane).** The route's protocol adapter parses strict,
   depth-bounded JSON and checks every field against the model's implemented
   profile. It produces a protocol-neutral request: the requested model name,
   ordered messages and content blocks, tools, sampling and limit fields, the
   stream flag, advisory hints and the caller's scope. Malformed or
   unsupported input fails here, in the protocol's error shape.
3. **Resolve and place (conductor role).** The name resolves to an installed
   artifact identity through a catalog snapshot; unknown or uninstalled
   models fail here (D-045), and a request for an installed model still
   registering is deferred within the queue bounds or gets 503 with
   `retry-after`. The conductor chooses a node, creates a routed
   attempt and dispatches it. Locally, dispatch is a bounded command to the
   scheduler (D-037).
4. **Render and tokenize (CPU worker, executing node).** The D-067 renderer
   for the artifact's template hash renders the request, and the artifact's
   tokenizer produces token IDs with segment boundaries. A missing output
   limit resolves to the finite supported default; a request that cannot fit
   its context fails with a 400 (D-045, D-050). Only a node that holds the
   artifact renders.
5. **Admit (scheduler).** From intake until admission the request counts as
   queued for D-055's close and compaction records. The front door stamps
   each request with the current close sequence number at intake; M6 serves
   the close API there too (D-041), so every close is ordered against
   intake. Once rendering has produced its tokens,
   the request is matched against every later close, and a close keeps its
   released entries' recorded ends until no request stamped before it is
   still unmatched, within the intake and queue bounds. One admission
   transaction then looks up retained state (the
   longest valid match within identity and scope, D-055), computes the
   request's envelope from its plan and bounds, and checks D-050's
   inequality together with task, queue and output credits. The request is
   admitted, deferred in a bounded queue without a grant, or rejected as
   impossible. Admission protects and promotes any reused blocks atomically
   and issues a continuation handle. For SSE, the front door now sends
   headers and the first event, then keepalives (D-045).
6. **Take the execution slot (scheduler).** Initially one request per node
   holds the slot from its first phase until it retires, is terminated, or
   is paused at a completed phase boundary by the switching policy. A paused
   request keeps its retained state resident and resumes when the policy
   gives it the slot again (D-069). Another request runs beside it only in a
   concurrent cohort whose full envelopes fit (D-050). A request whose known
   deadline cannot be met behind the queue ahead of it is refused before
   admission with 429; once admitted it is never refused with 429 (D-045,
   D-047).
7. **Run phases.** The adapter turns the request into a finite program of
   phases ([request programs](#adapters-and-request-programs)): for
   autoregressive decoding, prefill chunks and then decode steps. Each phase
   declares its dependency closure: weight chunks, state blocks and
   workspace. Resident chunks are
   leased. Missing chunks are paged in: victims are chosen, backing is
   recovered or handed off, direct reads fill it, and completion publishes
   the chunks before they are leased
   ([lifecycles](#page-in-and-eviction-lifecycles-8)). The task waits
   suspended, and the scheduler serves other work meanwhile. Once the
   closure is complete, the dispatcher submits the phase's launches on the
   request's stream and records a completion fence. The phase reaches a
   completed boundary only once the fence has completed, its host readers
   (the sampler, and in M7 the routing report) have consumed its outputs,
   and every other consumer and registration of its resources has retired.
   Then its new state becomes retained request state and its leases end
   (D-050). When every phase's closure is the model's whole closure, as
   for a full-swap model (M3), the request leases it once, at its first
   phase, and holds that lease to its end: each phase is then device work
   under it, still ending on its own fence, with no closure walked and
   nothing leased or released per phase (D-093, `scheduler.h` HoldLease). The
   per-phase lease stays for closures that change from phase to phase (M7's
   routed experts).
8. **Sample and stream.** Sampling happens before that boundary, while the
   logits' workspace is still leased: the decoding mode's sampler reads the
   logits once the fence completes. The output parser detokenizes
   incrementally, applies stop conditions and parses tool calls and reasoning
   (D-067). The protocol adapter turns the result into wire events in the
   request's bounded output buffer.
9. **Retire and hand off (scheduler).** At the request's end every consumer
   of its resources retires, and a lease it held for the whole request is
   released once its last phase's fence is seen (so on a cancellation too);
   its extents stay resident, now eligible for eviction. Its retained
   entries are published, and their
   accounting moves atomically from the grant into the idle cache (D-055);
   transient working state is discarded.
   The switching policy passes the slot to the next eligible request,
   round-robin with aging within each class (D-050, D-069).

**Cancellation.** A disconnect, deadline or explicit cancel settles the client
outcome at once. The task accepts no new child work, requests best-effort
provider cancellation only for operations no other waiter still needs, and
drains the operations already accepted. It publishes only what D-055 allows a
cancelled request to publish. The slot and the allowance stay held until
retirement (D-048). A timeout never proves reclamation.

**Remote execution (M6a).** The conductor forwards the normalized request
over the chosen node's authenticated session. The node performs steps 4–9
and streams response chunks back under credits. The conductor owns the
client connection and its protocol semantics
([below](#conductor-ownership-and-admission)).

## Model lifecycle

A **model context** is the runtime's unit of execution. It composes one or
more components, each an installed artifact or a declared part of one, in
roles: the main model; a speculative drafter, either MTP layers stored in
the main artifact or a companion drafter artifact; modality encoders and
projectors; and, for pipelines, text encoders, denoisers and decoders.
A resource that components share, such as a drafter's use of its target's
embedding table, is one resource counted once. The context also binds its
executable plans (one per supported request profile) and its state
representations (D-068). The first contexts are single artifacts, and the
FP16 and EXL3 builds of one base model are separate contexts (D-052).
Aliases name contexts. A model context is not a CUDA context. The steps
below apply to each component artifact.

1. **Install (job).** A job imports the artifact and publishes it into the
   installed store with one rename, or receives a verified copy from a peer
   or an archive (D-054, [artifact-format.md](artifact-format.md#directory-identity-and-publication)).
   The [job rules](#import-install-and-archive-jobs) (records, locks and
   grants) ensure that a late, cancelled or superseded job never publishes,
   resurrects a deleted model or removes a reinstalled one.
2. **Register (runtime).** At startup, or once a publishing job is reaped, CPU
   workers parse the manifest strictly and run the verifier's shallow checks:
   the directory name is the SHA-256 of the manifest bytes, the file set is
   exact, files are regular and singly linked, and sizes match. Unknown format
   or profile versions are rejected and need a re-import (D-056). The result
   is a compact **installed-model index** entry: identity, sizes, template
   hash, operation set, support status and D-046's discovery metadata (context
   length, modalities), enough for discovery with no I/O. The manifest alone
   does not carry all of that, so the first registration of an artifact
   derives the entry from the manifest, kept metadata and index, runs one
   planning pass to settle support status, and drops those plans. That pass
   reserves its bound in `F` before it starts, like detail below. The entry
   goes into a disposable cache in `state`, keyed by artifact ID, runtime
   build, profile configuration and probed platform capabilities. It is not
   one of D-062's versioned durable records. Entries are written atomically
   with a checksum, and a torn, unreadable, unknown-version or mismatched-key
   entry is discarded and rebuilt, never refused. Entries for other keys or
   removed artifacts are pruned at startup. The cache replaces only the
   index parse and planning pass; every startup still runs the shallow checks
   above. Uncached artifacts, such as a whole library after an upgrade, finish
   registering after readiness and appear in discovery once their pass
   completes. A request for one whose registration is still pending is
   deferred within the queue bounds or answered 503 with `retry-after`
   (D-045), never 404 or `x-should-retry: false`. The installed library's
   steady cost at startup and in `F` then grows with this compact index, not
   with every artifact's full index. An artifact that fails is reported
   unavailable and skipped; it does not stop startup. Registration commits no
   physical backing.
3. **Detail and plan on first use (CPU workers).** Support status is settled
   at registration: every operation must resolve to an eligible implementation
   in this build profile and pass the plan's layout checks, or the context is
   reported unsupported, and nothing is substituted (D-053). Before a
   context's first admission, the scheduler reserves the detail's bound in
   `F`, computed from the compact entry's sizes and entry counts and rechecked
   against the admitted inequalities (D-050); a build that does not fit is
   deferred or refused. Within that reservation a worker parses the full index
   and bounds-checks it against the shard sizes, records groups, chunks and
   resources as nonresident, and builds a plan for each configured request
   profile with its phase envelopes and state allowances. The detail is then
   kept as revocable metadata cache. It can be dropped only when no chunk of
   the context is resident or loading; no operation, lease, claim or retained
   entry references its records; no other context's detail references its
   resources; and no admitted request uses it. Record generations continue
   across a rebuild, and any admission while detail is absent rebuilds it
   first. Plans are never serialized; only inputs such as separately keyed
   tuning results may be stored. Chunk hashes are not rechecked, because
   integrity was established at install. M5 measures startup time and metadata
   memory as the library grows, and the cold-switch cost that on-demand detail
   adds.
4. **Use.** The first admitted request pages in each phase's missing
   closure. A dense model's decode step reads every layer, so its closure is
   the whole weight set. After that, residency is kept or lost chunk by chunk
   under victim selection; there is no model-level "loaded" state.
5. **Partial eviction.** Another context's demand reclaims eligible chunks
   of this one, least valuable first. Untouched chunks stay resident, and a
   later request reloads only what is missing (D-008).
6. **Remove or replace.** Removing an artifact applies to every context that
   composes it or uses a resource it provides. Removal stops new placement and
   admission for those contexts, waits for their admitted requests and
   outstanding consumers to retire, and ends their retained entries (D-055).
   Only then does it release backing, shared backing only once no remaining
   context references it, and close its file handles. An artifact that another
   installed artifact's manifest references is removed only together with that
   dependent, by the user's explicit choice (D-054). While a removal is in
   progress, no context that composes the artifact or uses its resources is
   registered. A removal job, under the same job rules, renames the artifact
   directory out of the published namespace into a private directory under
   `.staging` in one rename, then deletes it. A crash therefore leaves either
   the intact artifact or an unpublished leftover for the staging sweep
   ([artifact-format.md](artifact-format.md#directory-identity-and-publication)).
   A new artifact version is a new identity. Until hot swap is built,
   replacing a version means removing the old context under these rules and
   installing the new one; publishing the new version and draining the old one
   while serving is the deferred hot swap ([features.md](features.md)).

## Data model (§4)

```text
Logical resource -> tensor/storage byte ranges -> independently reclaimable
backing extents -> zero or more valid stored representations
```

Descriptor field groups: identity, content kind, semantics, layout, recovery,
residency, safety, policy. The categories are orthogonal. Multiple logical
resources may share an extent; the manager knows the full dependency closure
and charges each physical extent once. Shared or tied weights need content
and representation identity, not matching tensor names.

Identities are typed and generation-checked. Raw addresses exist only at
provider boundaries and in executable views that are rebuilt at load and
never serialized (D-006):

| Identity | Scope | Notes |
| --- | --- | --- |
| Artifact ID | Global | SHA-256 of the exact manifest bytes (D-056) |
| Group, chunk, resource | Within an artifact | Index positions; a chunk is a group-relative 2 MiB slice |
| Model context | Node incarnation | Component artifacts, plans and state representations |
| Backing handle or slot, with generation | Node | The generation advances on every reassignment, so a stale completion cannot claim a reassigned range |
| Content generation | A logical range | Advances when contents are replaced or invalidated |
| Lease, retention claim | Node | Leases protect backing for consumers; claims tie retained entries to blocks (D-055) |
| Task, operation, with generation | Node | Never reused while anything refers to them (D-048) |
| Retained entry, with generation | Node and caller scope | D-055 |
| Request, routed attempt | Conductor and node | Opaque; attempts are qualified by session and sequence (D-038) |
| Runtime incarnation, authority epoch | Node, cluster | Fence stale traffic after a restart (D-038) |

Small tensors and state blocks may be suballocated within backing (D-035).
Their logical sizes, validity, and leases are distinct from the provider's
physical mapping/release granularity and from the storage request size.
Freeing a suballocation can create a reusable hole subject to address,
alignment, and lifetime constraints; the whole backing is still occupied
until every occupant and outstanding registration/consumer permits release.
Immutable weight slots retain their imported group/chunk layout and content
identity. Backing bytes beyond a chunk's stored length (the rest of a 2 MiB
handle) are not written by reloads; they may be reused only under the general
suballocation rules below (union protection, content generation, lifetime),
never as an unprotected free pool, and small state blocks never inherit the
2 MiB chunk size. General/mutable reuse must also respect restore
footprints, content generations, and representation compatibility.
The ledger distinguishes reusable suballocated bytes from physically released
bytes. Owning all model address spaces in one process does not change the
provider's minimum unmap/release unit. Moving live contents to consolidate
holes would require a separately validated relocation/completion policy;
compaction is not implied by suballocation.

Pool capacity is distinct from allocation and transfer size. Steady-state
paging reuses backing after old consumers complete; release/create is not
required per read. D-033's baseline retains useful contents and directly
hands compatible backing to admitted replacements. A large slab kept mapped
could instead host software-managed slots with tensor views; 1 GiB slabs do
not imply 1 GiB transfers. Compare this owner-proposed alternative against
retained small handles before changing the baseline (D-035). Include address
stability, backend views, registrations, fragmentation, and pressure-driven
shrink; the current CUDA mapping API does not promise arbitrary interior
offset remapping of a large handle. Growth/retention stays within the node
budget and OS headroom; artifact extents do not fix physical handle size.

Conceptual state machine (real transitions also carry content generations,
consumer counts, and cancellation tokens):

```text
NONRESIDENT -> LOADING -> RESIDENT_UNLEASED <-> RESIDENT_LEASED
     ^                        |
     +------ EVICTING <-------+          LOADING  -> FAILED
                                         EVICTING -> RESIDENT_UNLEASED (safe cancel)
```

A dirty extent cannot become nonresident until recovery is secured or an
explicit discard has invalidated its logical contents.

### Memory classes (§5)

Immutable weights (discard clean copies, restore from artifact) · routed
expert weights (acquire the selected closure only) · dense/attention weights
(acquire what the implementation reads; no assumed activation sparsity) ·
sparse lookup and modality components · live KV / compressed attention /
recurrent state (preserve while resumable: residency, valid spill, or
reconstruction) · reusable completed-prefix state (retain by reuse and
recovery value; invalidate correctly) · scratch (recycle after final
consumers; don't spill dead scratch) · graph/runtime objects and kernel code
(planned shapes and captured graphs: counted, reclaimed by measured cost,
D-090 as amended) · communication buffers (stable backing for
registrations) · transfer staging, including D-081's landing zone
(bounded, pre-reserved; a separately declared persistent pool, not part
of `F`) · transient
working state such as a drafter's state or a diffusion canvas (charged to
the request's `R(G)` allowance and discarded at retirement; never a D-055
entry unless its adapter declares it). Live and
reusable state goes through architecture-specific adapters with
conservative semantics.

## Memory and residency

### Budgets and ledgers

Each memory domain has one execution budget `B`: the physical memory jitLLM
may use after OS and external headroom. A Spark is one domain shared by CPU
allocations, GPU backing and page cache (D-004). The ledgers are keyed by
domain from the start, so for the ledgers a discrete-GPU platform is a data
difference, not a redesign (D-026). On a discrete GPU (D-082) the domain is
the device's memory, with a configured `B` (not all of it: the GPU may
drive a desktop) and the SSD as the only second tier. Host memory holds the
runtime and the landing zone for direct reads, a bounded pool declared
outside the device ledger and reported on its own, and is not a tier (host
RAM as one would be a second domain, designed for but not built). jitLLM
uses one GPU, device 0. Two ledgers stay separate
([reservation-policy.md](reservation-policy.md#admission-rule-and-separate-ledgers)):

- **Commitment:** fixed overhead `F`, admitted retained state `R(G)`,
  non-revocable background work `J` and the phase envelopes. Admission checks
  these against `B`. They are promises, not bytes in use.
- **Occupancy:** backing that actually exists, by class and state: resident
  and leased, resident and eligible, loading, evicting or awaiting
  write-back, quarantined, or held in a pool. Every materialization checks
  occupancy against `B`. Revocable cache may fill capacity that is committed
  but not yet used.

A budget reduction is a scheduler request. It waits for retirement or is
refused; the budget never drops below outstanding claims (D-050).

### Backing and addresses

D-033 starts with independent 2 MiB physical extents. Compatible backing is
handed directly to admitted loads that need it, with no standing cache of
unused handles. Weights and state are backed by device-located VMM:
the GB10's L2 does not cache host-located memory (RE-022). On validated
Spark configurations direct file reads land in a bounded host-VMM zone of
2 × depth 2 MiB extents, and the GPU copies each one into its device
extent, by the copy engine by default (D-081).

With D-033's independent handles, the runtime reserves a virtual range for
a context's weights before its first page-in; this is address space only.
Each group gets a 2 MiB-aligned region, and chunk *k* maps at that region's
base plus *k*·2 MiB. A resource's view then stays stable across eviction and
reload while its backing and generations change underneath
([executable views](artifact-format.md#executable-views)). Slab slots would
move a group's address with its slot; address stability is one of the
retained-backing comparison's criteria, so this scheme stands or falls with
D-033. A stable address does not by itself make a captured pointer or
registration valid (§7).

Retained state uses its own blocks, sized per state adapter rather than per
chunk. Spill writes go from backing through the landing zone (D-081), so a
state block's bytes past its valid length are zeroed before the block joins
a retained entry (D-055). Workspace comes from charged backing that the plan declares. The M2
[retained-backing comparison](backend-proof.md#retained-backing-comparison)
tests the owner-proposed slab alternative before D-033 is kept or amended.

### Page-in and eviction lifecycles (§8)

Page-in: commit capacity for the actual missing extents → obtain backing →
map and set access → transfer (direct read into the landing zone, then
the copy into device VMM, D-081) → verify completion, full length and the
current content generation (no page-in hashing, D-056) →
publish resident → grant lease. Duplicate requests for one content generation
are coalesced.

The scheduler runs each page-in in stages, each an operation of its own
whose proven completion alone moves it on (`scheduler.h`): the VMM lane
creates the extent's backing, maps it and sets access; the load waits, in
order, for a landing slot; the storage lane reads the chunk into the slot,
starting reads in the order they were published, so a load reaches the
device sequentially (RE-026);
the copy lane (or, without one, the device lane) copies it into place on
the zone's stream, in up to four pieces when the extent is not one range
of the read (a page of a resident expert slab, whose groups sit at a
stride that is not the file's); and the extent
is published, and the slot freed, only once that copy's fence has
completed. At most twice the zone's slots of landed loads are in flight,
so backing is mapped at most one zone ahead of the reads. A failed or
withdrawn load unmaps and releases what it mapped before the extent is
nonresident again; a read, copy or unmap whose completion is unproven,
or an unmap the provider refuses because an earlier unknown outcome left
its place undetermined, quarantines the extent and its slot, never
reused. Extents the CPU must
read (the FP16 token table's host copy) are read in place into host
backing instead, with no slot or copy.

Eviction: select specific eligible extents → atomically exclude new leases →
wait for all consumers and registrations → write back only if preservation
requires it → commit recoverable state / invalidate discarded entries → unmap
and release or recycle → update occupancy and generation. The unmap and
release run on the VMM lane while the extent is EVICTING (D-033: the
backing is released, not pooled). An eviction asked for with a handoff
(M3's full swap) unmaps the backing but keeps it, and parks, still
EVICTING and charged; a page-in in the same domain whose backing has the
same class and size takes it, and the parked eviction completes in the
same step, so the
charge moves from one extent to the other and occupancy never exceeds B.
The VMM lane then maps the kept backing with no create or release.
Backing no load took is released when the task that asked for the
eviction finishes: never an idle pool. Live mutable state is written back
first, by the reverse path through the zone: the extent is copied into a
landing slot, fenced, and the slot written to the state's place with
direct I/O. Only once the whole range is written is the backing released,
and the catalog then marks the contents preserved at the same content
generation, so a later load restores them (D-086). In M2 the place is a
process-private unnamed file; M3's service names each request slot's file
in the spill role and keeps a conversation wholly on disk across a
restart by a validated record (D-105); D-055's full retention arrives in
M6.

A request's lease (M3) is an ordinary catalog lease held longer, so no
eviction of its extents can begin while it is held (invariant 6). A swap
that needs them waits for its release (`AwaitRelease`: woken then, not
retrying), which comes at the request's end, or earlier when the swap is
asked for between the request's steps and the runtime ends the request
first; the wait has no bound, and a task holding a request's lease (or
whose ancestor does) is refused it, so holders never wait for each other
in a cycle. Ending the lease, or cancelling the request, releases it only once
the fence of the step in flight, if any, is seen (invariant 2); a step
whose completion stays unknown keeps it, and its extents, for good.

Storage backends sit behind one read/write completion interface. D-034 selects
native direct-file I/O on validated Spark configurations, with bounded
asynchronous submission and no CPU payload copy; D-081 lands it in a
host-VMM zone and copies each extent into device VMM (device VMM cannot be
mapped for the CPU or exported as a dma-buf on the Spark). cuFile
compatibility mode is a comparison path, not required for the initial
runtime. Native GDS (supported non-Spark targets) and remote extent transfer
remain later backends. GGML execution on device VMM and full
registration/reclaim lifetimes are part of the M2 integration proof.

### Victim selection (initial baseline)

Victims are chosen only when an admitted phase needs capacity or a retention
cap is exceeded. Allowances and grants never trigger eviction (D-007). The
baseline is deliberately simple, so that alternatives can be measured against
it on recorded traces (M9 in [plan.md](plan.md#milestone-ladder)):

- **Eligible:** resident and unleased, with no outstanding consumer or
  registration, and not quarantined. It must also be either clean
  (restorable from the artifact or a valid spilled copy) or revocable
  retained state that D-055 allows to be spilled or dropped. Blocks that
  admitted work uses are never candidates (D-055).
- **Order:** D-055's [victim order](retention-policy.md#victims-spill-and-exhaustion).
  Released and expired entries go first. Next come clean idle weight chunks,
  least recent *actual* use first across all models, while resident
  retained state stays within `M_state`. When it exceeds `M_state`, or no
  eligible weight chunk remains, the least recently refreshed entry is
  demoted: spilled when allowed, otherwise dropped. Prefetch and cancelled
  planned use do not count as use.
- **Credit:** a victim counts only the bytes it makes usable for the pending
  materialization, and only once no lease, claim, other occupant or
  consumer remains: a whole compatible handle or release unit, or a
  suballocated hole the destination fits (see the data model).
- **Ties** break by artifact, group and chunk, so replays and fake-backend
  tests are deterministic.

The baseline has no hysteresis, minimum residency or reload-cost weighting.
Those confirmed features, a frequency/recency policy and §9's cost-aware
heuristic each replace it only after beating it on the same recorded
traces. *As built in M3 (D-055 as amended 2026-10-02):* the runtime's
reclaim order (`memory/reclaim.h`, `Server::Reclaim`) already weighs
restore cost: every reclaimable kind, captured graphs, plans (host and
driver memory charged inside the budget past one step's floor, D-090 as
amended) and idle conversation state (spilled, not cleared, writing only
what changed), and idle weights once partial eviction produces them, goes
by GreedyDual-Size over its measured cost to restore a byte freed (stale
entries fall behind fresh cheap ones as reclaims go on), least recently
used within a kind, the running model's last; the scheduler's
extent-level choice (`memory/victims.h`) stays the baseline for what it
sees (discarded, then clean weights). Details in the
[retention policy](retention-policy.md#victims-spill-and-exhaustion) and
[runtime-serving](runtime-serving.md#registration-and-the-swap). Dependency-group scoring stays deferred (features.md). The M0 study
replayed global deterministic LRU with whole-closure acquisition. It
recommends first measuring an eager load of the active dense model that
preserves inactive extents
([full study](experiments/paging-feasibility/full-study.md)). Every eviction
emits an event naming the victims and their classes, expected and actual
bytes, and how many alternatives were passed over.

### Memory breakdown

Status output and the dashboard show one breakdown per memory domain. Each
line names its source, and lines from different sources are never silently
combined:

| Line | Source |
| --- | --- |
| Physical total and available memory | OS counters: `MemTotal` and `MemAvailable`, which on the Spark include all VMM backing, device-local and host alike, from creation to release |
| Configured headroom and execution budget `B` | Configuration |
| Commitments: `F`, `R(G)`, `J` and the active envelopes | Commitment ledger |
| Occupancy by class: weights (leased, eligible), admitted state, retained entries, workspace and activations, I/O and communication buffers, runtime metadata, loading, evicting or awaiting write-back, quarantined, pool-held, and non-evictable backend or unknown allocations | Occupancy ledger, with shared backing counted once |
| Spill: bytes held and bytes written in the rolling window | Retention cache and storage service |
| Plans and captured graphs (host heap and driver memory outside extents) | Each runner's account, counted (`engine/planned.h`); past one step's floor, a pinned runtime extent in the occupancy ledger (D-090 as amended 2026-10-02) |
| The runtime process's ordinary allocations | OS accounting for the process (RSS, its cgroup), which on the Spark misses VMM backing: device backing never appears, and host backing only while mapped with access |
| Driver bookkeeping for the backing | Derived: the kernel's unreclaimable slab, about 34 KiB per 2 MiB extent while it exists and more while mapped on the Spark; charged to `F` |
| Unattributed: the drop in `MemAvailable` not explained by cataloged occupancy, driver bookkeeping, ordinary allocations or page cache | Derived; a discrepancy to explain, never free memory |
| Page cache, jobs and other processes | OS counters; all outside `B` |
| Driver-reported free memory | Provider probe; informational, never an admission input |

On the Spark, VMM backing leaves `MemAvailable` when it is created and
returns when it is released, and the driver's free memory equals
`MemAvailable`. Per-process and cgroup counters miss it, so a cgroup limit
does not bound it and the budget `B` must
([measurement](experiments/vmm-counters/README.md)). The breakdown therefore
reconciles the catalog against the system-wide counters, not the process
footprint. D-034's direct reads kept the page
cache empty ([I/O follow-up](environment.md#io-path-follow-up-2026-09-21)),
but jobs and other processes can still fill page cache within the headroom.

## Conversation-state retention

D-024 and D-031 distinguish state required by admitted work from reusable
state kept between requests. A suspended continuation retains the resources
needed to complete or safely unwind; expiry of an idle cache entry cannot
invalidate those resources. After response completion, prefix retention is
subject to bounded memory and spill capacity. Optional sessions and hints can guide
policy without making storage unbounded.

D-055's [retention policy](retention-policy.md) settles the rules: entry
identity, restore boundaries, immutable shared blocks, refresh by branch,
release semantics, the initial victim order, spill storage and M6's named
acceptance workload. Shared prompt-prefix and conversation-continuation
entries have separate reuse statistics and retention/expiry decisions within
common bounds. Shared-prefix value comes from reuse across conversations;
continuation value comes from reuse of that history. A hit on the shared
prefix does not refresh unrelated continuations. Retention is
capacity-driven with a per-class maximum idle age of 24 hours by default.
The capacity values (resident state, spill bytes, entry counts, minimum
prefix length) are pinned at M5 exit from measured state sizes and headroom.
Spill-full or expiry invalidates only eligible reusable entries; active work
retains a valid recovery path or safely fails under the admission policy.
Spill is deleted at startup, so no crash durability or indefinite retention
is promised; M3's service keeps a conversation wholly on disk across a
restart only through a record it validates whole, within the same
retention (D-105).

Cache identity covers artifact/model version (every component artifact of a
composed context, D-068), relevant execution settings
(including position/attention configuration), state representation/layout,
and the exact rendered token prefix from the context origin plus non-text
input identity when supported, within one caller scope (D-055). The same
system-prompt text after different preceding input is not the same prefix;
matching a message label or its text alone never authorizes a hit.
Tokenizer and template changes must not produce an incompatible hit. Architecture-specific adapters define which boundaries
can be restored; do not assume a recurrent snapshot can be truncated like
full-attention KV.
Coverage includes the attention window required at the first resumed token,
not merely the snapshot's final token or byte identity. Template rewrites can
rewind a common prefix behind the window preserved by a sequence snapshot,
even when the source context used full-SWA allocation (RE-007). In that case
restore an earlier compatible checkpoint or recompute; successful deserialization
and tail removal do not authorize reuse.
Independent branches may share compatible immutable prefixes, with their
mutable continuation state isolated. Client-inserted per-conversation material
that the endpoint is documented to strip, such as Claude Code's attribution
block, is removed before prefix identity is computed, and a client's documented
post-compaction signal releases the prior continuation without touching shared
prefixes (D-045).

Retain the shared system-prompt prefix independently of longer conversation
snapshots at supported restore boundaries. With inputs `S + A` and `S + B`,
where `S` is the same compatible rendered prefix, both requests may reuse
the immutable state for `S`; neither may use the other's divergent suffix.
Expiring or releasing A drops only A's continuation retention, not S's cache
entry or B's state. Reusing S for a new conversation does not keep A alive.
S remains subject to its own bounded retention policy. Expiry of a cache
entry removes its retention claim, not backing still needed by admitted
work or other retained entries; shared extents are counted once. Eviction
updates affected residency and restore metadata; a cache hit requires a
valid resident or stored representation of every dependency needed to restore.

On a compatible hit, restore state and process new input plus any declared
cache-block tail. If a longer continuation is missing or expired, reuse a
compatible shorter prefix at a valid restore boundary and recompute only the
remaining supplied history. Without a valid prefix, recompute from the
request's full history. If required history is unavailable, fail explicitly.
Expose reused/recomputed token counts and miss reasons through diagnostics
without logging prompts or KV. M6 tests branching histories, edits to an
earlier message, incompatible cache identity, expiry, and spill exhaustion,
alongside both resident reuse and forced spill/restore. Include S+A and S+B
with independent release/expiry, continuation eviction while S remains,
shared-prefix expiry while a consumer is suspended, changed rendering or
preceding context that must miss, and shared-byte accounting. Compare each
branch's logits with its uncached reference and report shared-prefix reuse
separately from longer-history reuse.

## Artifacts, storage and jobs

### Prepared paging layout (§11)

Making a model available includes import into an immutable paging artifact,
with metadata describing architecture, tokenizer, execution representation,
and the index from logical resources to stored groups and chunks (D-009,
D-035, D-056). Publication follows complete validation; interrupted
preparation is not an available model. The artifact is a content-addressed
logical unit with safetensors file shards and a jitLLM manifest/index,
specified in [artifact-format.md](artifact-format.md). Runtime paging does
not inherit the source checkpoint's tensor ordering.

Each dependency group (a dense layer, one expert's closure in one layer, a
row table, the head) is one contiguous file range, 4 KiB-aligned, so disk
carries almost no padding. Groups divide into group-relative 2 MiB chunks,
the unit of closures, integrity records and independent 2 MiB backing
handles. Small tensors with compatible use/lifetimes share a group and may
share a chunk. Shared or tied weights keep one representation and shared
ownership. Repacking honors actual backend strides and quantization blocks,
with no CPU payload transformations at page-in. Whether backing is
per-chunk handles or slab slots, and whether GGML expert views use pointer
tables or uniform strides, are runtime choices that the file layout leaves
open.

Weight misses fetch their chunk closure. Adjacent missing chunks coalesce
into bounded, vectored direct reads, with one iovec per separately admitted
and protected destination. Resident chunks are never overwritten to
manufacture a sequential read. Page-in does not hash; integrity is checked
at install, replication and explicit verification (D-054). Sparse row
requests resolve to their containing chunks, with useful-byte/read
amplification measured separately. This layout favors sequential work inside
a resource group; routing can still select distant groups. No physical NAND
placement or all-sequential workload is promised. Mutable state has separate
spill files and generation/retention rules; metadata need not use 2 MiB I/O.

### Storage service

The storage service owns every direct-I/O submission and completion on the
node (D-034, [async-model.md](async-model.md)). It opens files only beneath
the installed and spill roles, without following links or crossing mounts
(D-054, D-063). It allocates no payload buffers. Every read or write names
admitted, protected backing ranges by backing identity and generation. The
scheduler records those ranges and their addresses when it prepares the
operation (async-model.md), and the service checks the backing generations
at submission. On the Spark those ranges are landing-zone extents, and the
copy to or from device VMM is device-execution work (D-081).

- **Classes, highest first:** demand reads for admitted phases, including
  restores of their retained state, together with any write-back or spill
  that frees a destination for such a read (same class, D-050); prefetch
  and warming; retention spill writes; maintenance. Queue entry is by class,
  with aging only within a class. Maintenance keeps a small reserved share
  so that expiry completes within D-055's pinned interval; apart from it, no
  lower class overtakes demand for admitted phases. Requests already in
  flight are not preempted.
- **Depths and sizes:** start from D-034's two 2 MiB requests in flight for
  latency-sensitive loads and four for bulk reads. Adjacent misses may
  coalesce into vectored runs (D-056), an option of the reader that is off
  by default: BP-P1's A/B measured it slower than one read per chunk
  (D-085). When on, reads already waiting for room that continue one
  another in a file start as one request, so coalescing never delays a
  read; a span's failure is retried read by read, and a span is cancelled
  only once no read in it is wanted (`providers/direct_reader.h`). Tuning
  is M2/M3/M6 measurement.
- **Completion:** every submission resolves as not started, accepted or
  unknown, and the original request's terminal completion is harvested even
  after cancellation (D-048). A short read or an error publishes nothing.
  Page-in does not hash; spill restores check their catalog digest (D-055).

### Import, install and archive jobs

Jobs do the long, untrusted-input work in separate processes, so a hung
mount or a hostile checkpoint cannot stall the scheduler (D-054). The
runtime starts them on management requests, limits how many run at once,
observes their progress and can ask them to cancel. Jobs keep their memory,
dirty file data included, within the node's headroom, and their I/O yields
to serving. These rules keep a job from publishing or deleting behind the
runtime's back:

- **Records and bounds.** The runtime records each job durably under
  `state` before starting it, within a bounded number of job records;
  beyond that, the request is refused. A job that conflicts with an
  unfinished job for the same model or artifact ID, or for any artifact
  that its manifest references or that references it, waits, and reports
  that it is waiting.
- **Containment.** Each job runs in its own cgroup, which
  `jitllm.service` delegates to the runtime (`<unit>/jobs/<id>`, beside the
  runtime's own), and the runtime is also the subreaper that reaps its
  jobs' orphans (D-074); a process group alone does not contain it. A job has ended only when every process it started has exited
  and been reaped, and its record lock is free. Cancellation, a timeout or a
  missing report does not end it.
- **Locks.** Each job record has its own lock file under `state`, never
  under `/run/jitllm`, which systemd removes when the unit stops. Every
  process of the job inherits a hold on it at spawn, across exec. Every
  process that can write a staging directory also holds that directory's
  lock (artifact-format.md).
- **Grants.** A job renames into or out of the published namespace only
  after the runtime grants it. Each grant names the model's current install
  generation and advances it. The runtime records the grant durably, with
  its artifact ID, before sending it, grants at most once per job, and never
  grants after recording the job's cancellation. A grant also rechecks the
  artifact's manifest references, which an import learns only at the end,
  against removals in progress. A job that has not received
  its grant never renames. A cancel that arrives after the grant takes the
  job's reaped outcome.
- **Registration** follows the reap, not the job's report. After a cancelled
  removal, the model is registered again only once that job has been reaped,
  and only if its artifact is still published.

Stages that parse untrusted input run confined: import's parse and repack,
and verification of an archived or received artifact. Such a stage can read
only its inputs and write only its own staging directory, with no network
and no access to `spill`, `state` (beyond its inherited lock descriptors),
credentials or other artifacts. Downloading is confined the same way, except
that it may read the Hugging Face credential and use the network.
Publishing an archived or received artifact never rests on a confined
stage's verdict. The unconfined part hashes each file after every process
that could write the copy has been reaped and before anything parses it,
checking the manifest bytes against the identity held outside the source
and every listed file's SHA-256 against the manifest. A verification stage
then gets the copy read-only, and nothing that can write the staging
directory runs between the hash check and the rename. A confined stage
applies a Landlock ruleset (read its inputs, write its staging directory,
no TCP, no signals or abstract sockets outside itself) and a seccomp
filter that refuses sockets and io_uring to itself, since RE-013 blocks
unprivileged `unshare` and `bwrap` on these hosts (D-074). M1's job proof
covers a child that outlives its job, inherited locks across exec, runtime
and unit restarts and a confined stage (`tools/job-proof`), before M5
builds the importer on it.

- **Import:** stage the source locally, validate it as untrusted input, plan
  groups and chunks, write shards into `.staging`, verify the complete
  artifact as an installer would, and publish with one rename
  ([artifact-format.md](artifact-format.md#directory-identity-and-publication)).
- **Install from an archive or a peer:** copy, verify against an identity
  held outside the copy's source, and publish atomically. A peer receives
  the artifact over the cluster link from the importing node, with its
  identity sent over an authenticated session (D-054).
- **Archive and delete:** only on the user's explicit choice and after
  quiescence. An archive copy is verified, read back from the store, before
  the local copy is removed, and a delete first renames the artifact out of
  the published namespace (above).
- **Verify:** the standalone verifier checks hashes, index bounds and
  manifest consistency on demand (features.md).

How jobs are launched and how they report back are M5 implementation
choices, as is the peer-transfer mechanism in M6a.

## Execution

### Operations, implementations and plans

jitLLM owns dispatch (D-053). The **operation contract** defines each
operation's operands (views over catalog resources, state blocks or plan
workspace), attributes and shape class. An **implementation** is a
build-time unit for one operation or a fused segment. It declares what D-053
lists: supported operations, layouts and quantization; shape and alignment
limits; numerical behavior; workspace and peak memory; library handles;
capture restrictions; and completion semantics. It launches only on the
stream and workspace it is given, and returns errors instead of aborting.

A **plan** binds one model context's components and a request profile
(context bound, prefill chunk sizes, batch, decoding-mode bounds) to what
this build contains. It lists the phase kinds the adapter may use; each
kind's operation invocations with their selected implementations and launch
configurations; its dependency closure and worst-case bound; the streams
and fences; and the envelope. Selection is deterministic per operation,
architecture, layout, shape range, device capability and build profile.
Fusion is an explicit choice between fused and unfused implementations. A
plan's identity covers every component artifact, the request profile, the
adapter version and every implementation's identity, and it is part of
retained-state identity (D-055). Plans change only at request boundaries or
through D-050's atomic envelope replacement.

Worst-case unions grow with phase width, so a model that fits with small
decode phases may not fit a wide prefill or verify phase. A plan therefore
carries a small set of validated widths for each wide phase kind, such as
prefill chunk sizes, each with its envelope. A request profile may pin its
width, and M6's acceptance profiles do; otherwise admission uses the widest
validated width that fits. Widths can change numerics within declared bounds,
so the chosen width is part of the plan identity recorded with retained state.
An entry produced at another width is a miss unless the widths are validated
as state-compatible (D-053, D-055). A rejection names the phase kind, width,
required bytes and shortfall. Discovery and the what-if query show supported
widths and their envelopes. M2 and M7 measure each phase kind's guaranteed
bound against its observed peak, to refine validated plans without weakening
the guarantee.

The **dispatcher** walks a phase's invocations on the device submission lane
and records a completion fence after the phase's last consumer. It fences
earlier only where the plan needs host visibility, such as an MoE routing
report (below). Initially each active request has one compute stream.
Decode steps replay as CUDA graphs captured per model, plan and shape
(D-090): each captured graph names fixed addresses, so the extents it reads
have pinned places the scheduler refuses to move, a swap maps whatever
backing it takes at those places, and what varies between steps is data
the graph copies in, never a launch parameter.
The M2 proof chooses, per GGML-derived operation, between a context adapter
and a lifted kernel, and settles the contract's exact types.

### Adapters and request programs

An **architecture adapter** maps one supported architecture family, Qwen2
dense first, from artifact metadata to: the operation graph of each phase
kind it uses, the binding of index resources to operands, its supported
request profiles, its decoding mode and its numerical profile. For each
admitted request it produces a **request program**: a finite sequence of
phases drawn from the phase kinds the build registers.

- **Phase kinds** each declare their dependency closure (static or
  discovered at run time), the worst-case bound on that closure, their
  envelope, their outputs and what their completed boundary means. The
  first kinds are the prefill chunk and the decode step; M7 adds routing and
  expert subphases (D-050). The rest are designed now and built later
  (D-068): draft, verify and accept or roll back for speculative decoding;
  canvas denoise and canvas commit for block diffusion; modality encode;
  pooled output for embeddings and reranking; image denoise and decode for
  pipelines.
- **Finite before admission.** Every program has bounds fixed at admission:
  output tokens, draft depth and verify width, denoising steps and block
  count, image steps. Run-time outcomes such as draft acceptance or adaptive
  stopping only shorten an admitted program, never lengthen it (D-050).
  Draft depth and verify width are clamped so that written positions never
  exceed the admitted context and output bounds; every narrower width a
  clamp can reach is itself a validated width of the plan. A canvas keeps
  the model's block size as transient working state unless its adapter
  validates a shorter one, and only its commit is clamped to the admitted
  output bound.
  Where a full block would cross the model's context limit, admission lowers
  the output bound to end on a block the context can hold; if no whole block
  fits, the request fails as context exhaustion (400, D-045). A program is
  represented by its bounds, counters and current phase state; nothing is
  allocated for each future phase.
- **Data-dependent closures.** A phase whose dependencies depend on computed
  values, such as MoE routing, sparse row tables like Qwen3.8's n-gram table
  or per-request adapters, uses the dependency-discovery boundary described
  for MoE below: compute the selector, acquire the closure, run, and never
  substitute (D-008). Its admission bound is the worst-case union over every
  position in the phase. With X routed experts per layer, top-r routing and
  T positions in the phase (a prefill chunk, a verify of every drafted token
  plus one, or a 256-token canvas), that is min(X, r·T) routed experts per
  layer plus any shared ones.

A **state adapter** defines one state representation and declares its
capabilities: block size and layout, bytes per token for `R(G)`, append,
truncation (to any position, or only at snapshots), snapshot and restore,
fork by copy-on-write, and the coverage checks that resuming needs (D-055).
A context may use different representations in different layers, as hybrid
models do. Speculative decoding needs truncation after a rejected draft;
recurrent state provides it through bounded per-step snapshots. **Transient
working state** is state a program needs across its own completed boundaries
but never publishes: a canvas between denoising steps, a drafter's state and
inputs, drafted positions awaiting acceptance, snapshots kept for rollback.
It is retained request state under D-050, charged to `R(G)` at its program's
bound and protected until the program discards it or the request retires,
when it is discarded. It never becomes a D-055 retained entry unless its
adapter declares and validates that. Retention uses only the capabilities
an adapter declares.

A **decoding mode** pairs a program shape with its sampler and output rules.
Autoregressive decoding produces one position per step. Speculative
decoding accepts drafted tokens by exact match against the verify pass, or
by rejection sampling, so its output follows the target's verify-path
distribution. That equals plain decoding only when the verify and decode
plans are numerically identical; a wider verify batch can change kernel
shapes and top-1 choices (RE-008). Block diffusion samples a canvas under a
step schedule with adaptive stopping. Each decoding mode declares which
request features it supports, such as constrained output, log-probabilities
and stop sequences inside a block, and the rest are rejected explicitly
(D-043). Output
therefore arrives as single tokens, accepted runs or whole blocks, and
protocol adapters stream whatever arrives, with keepalives covering the gaps
(D-045). Seeds and schedules are recorded.

Adapters, phase kinds, state capabilities and decoding modes are native C++
compiled into the build. Support is earned per checkpoint and recorded in
the [support matrix](model-support.md) (vision.md).

### Model shapes

D-068 requires the design to accommodate shapes before jitLLM executes
them. The resource core (catalog, ledgers, admission, leases, retention,
scheduler) never names an architecture. A new shape adds adapters, phase
kinds, state capabilities, decoding modes and operations with declared
bounds, and it must pass the existing D-050 and D-055 adversarial matrices.
A shape that those contracts cannot express is a contract change with its
own decision.

| Shape | Examples | Evidence so far | Design hooks | Execution |
| --- | --- | --- | --- | --- |
| Dense decoder, full-attention GQA KV | Qwen2.5-0.5B | D-051/D-052 fixtures and external references | The baseline adapters | M2–M6 |
| Hybrid sliding-window and global attention | Gemma 4 | Gemma 26B-A4B reference; RE-004, RE-007 | Per-layer state representations; coverage checks (D-055) | M7 |
| Linear-attention or recurrent layers mixed with attention | Ornith 1.5 (`qwen35moe`), Qwen3.8 | Ornith's recurrent state saved and restored in the A→B→A reference | Snapshot-only restore; truncation through snapshots | M3 (Qwen3.8); M7 (Ornith) |
| Compressed attention with an indexer | DeepSeek V4 Flash | Its compressed-attention and indexer state charged in the paging study | State adapter and operations | M3 |
| Routed experts, with or without shared experts | Gemma 4 26B-A4B, Ornith, Qwen3.8, DeepSeek V4, MiMo | References and route traces | Routing boundary; worst-case unions | M3 (resident); M7 (demand-paged) |
| Sparse row tables | Qwen3.8's n-gram table | Layout study | Data-dependent closures | M3 |
| Stored MTP layers | Ornith (one layer), Qwen3.8 (MTP head), MiMo | Stored and accounted in references, never executed | Draft, verify and rollback phase kinds; truncation | M3 (Qwen3.8); M9 (Ornith, MiMo) (D-068) |
| Companion MTP drafter | Gemma 4 assistant drafters; DeepSeek V4's DSpark; GLM-5.3's DFlash2 | None | Composed contexts; resources shared across artifacts | M3 (DSpark); M4 (DFlash2 or GLM's MTP, whichever is faster and correct, D-087); M9 (Gemma 4) (D-068) |
| Block diffusion over a causal prefix | DiffusionGemma-26B-A4B | None | Canvas phase kinds and sampler; transient canvas; bidirectional attention over cached KV; restore points only where the adapter validates them (vLLM describes the commit as a causal encoder pass) | M9 (D-068) |
| Modality encoders | Qwen3-VL tower (Qwen3.8 Flash Next first), Gemma 4, MiMo, DeepSeek-ViT, GLM ViT; Gemma 4 E4B audio | None | Encoder components and phase kinds (D-042's staged modalities) | M3.5; M4 (GLM-5.3 Flash, DeepSeek V4.1 Flash) (D-101) |
| Decision heads | Clef, Clef-flash | None | Prefill-only program exposing final-norm hidden states; head component | M3.5 (D-101) |
| Speech synthesis | Breeze-TTS-2, Kokoro-82M | None | Codec and vocoder components; audio output phase | M3.5 (D-101) |
| Image-generation pipelines | Qwen-Image-2.1 | BF16 and GGUF references | Multi-component contexts; per-phase release | M3 |
| Pooled outputs | Embeddings (D-042), reranking (D-044) | None | Pooled-output phase kind; bidirectional attention | M10 |
| Model-parallel sharding | GLM-5.3 and DeepSeek v4.1 TP2; MiMo TP=2/EP=2 | Two-Spark references | Per-rank plans | M4 (TP2); M8 (MiMo's EP) |

Shapes arrive with the first model that needs them: Gemma 4 and Ornith as
M7's daily drivers, DeepSeek V4 Flash and Qwen3.8 as M3's large pair
([plan.md](plan.md#milestone-ladder)). Three consequences matter already:

- **Wide phases on routed experts.** A k+1-token verify or a 256-token
  canvas multiplies the positions in one phase. With 128 experts and top-8
  routing, a canvas can plausibly touch most experts in each layer on every
  denoising step. If it does, demand paging cannot help those layers, and
  the plan must keep them resident. This is unmeasured; a bounded
  DiffusionGemma reference study measures per-step closures before M9
  planning. Google notes that MoE verification can load additional experts,
  and the llama.cpp contributor saw no MoE speedup from Gemma's drafter
  (D-068).
- **A numerical contract per mode.** Autoregressive decoding compares
  teacher-forced logits. The speculative verify path's teacher-forced logits
  are compared with plain decoding within declared bounds, with top-1
  agreement reported. Free-running greedy divergence is reported with its
  first position, and must be zero only for a plan chosen to be numerically
  identical to plain decoding (RE-008). Diffusion is compared step by step
  under a fixed seed. Image outputs need a quality metric beyond pixel
  agreement.
- **Artifacts and identity.** Stored MTP layers are ordinary tensors in
  their checkpoints, so v0 groups them like any layer. A companion drafter
  or a multi-component pipeline needs manifest references to another
  artifact by ID, which stays open in
  [artifact-format.md](artifact-format.md#deliberately-open). Retention
  identity covers every component and plan the retained state depends on
  (D-055).

### Tokenization, rendering and output

- **Tokenizer:** a native implementation of a fixed set of tokenizer,
  pre-tokenizer and normalizer kinds, selected by identifier or hash like
  renderers; patterns from artifact metadata are never compiled. The
  vocabulary, merges and special tokens come from the artifact, and the
  result must reproduce the reference token IDs on fixtures. M3's byte-level
  BPE with its UCD 15.1.0 tables is in [tokenizer.md](tokenizer.md),
  cleared for shipped binaries by D-088.
- **Renderer:** native family renderers chosen by template hash (golden
  fixtures, segment boundaries) or probe equivalence, else the bounded
  Jinja-subset interpreter of the checkpoint's template (D-067).
- **Output:** incremental detokenization that holds back incomplete UTF-8,
  stop conditions, and the model family's tool-call and reasoning parser.
  Protocol adapters turn the result into wire events.
- **Counting and preview** (`count_tokens` and D-043's tokenization routes)
  run the same renderer and tokenizer on a CPU worker, without admission or
  weight residency.

### Sampling

The decoding mode's sampler reads logits after the phase's completion fence:
the final position for autoregressive decoding, the verified positions for
speculative decoding, and the whole canvas for block diffusion. It runs on
the host first, reading the logits from host-accessible
backing that the plan allocates for them (a design choice, not yet
measured); a device sampler is an option to measure against D-052's decode
gates. The supported sampling parameters are part of the M5 contract
(D-040, D-043). Numerical acceptance compares logits, never sampled text
([first-slice.md](first-slice.md)).

### Routing boundary for MoE (§7)

Prepare input and router dependencies → route → selected expert IDs → resolve
local shards and ranges → acquire the dependency closure → expert compute →
combine → release after consumers complete. Initial implementation: compact
GPU-to-host selected-expert report, native residency decision, all selected
experts acquired before launch, continuation suspended while I/O is pending.
CUDA graphs: residency decisions sit outside captured segments; no CUDA API
calls from host-function nodes. The same boundary serves every
data-dependent closure ([request programs](#adapters-and-request-programs)),
and its admission bound covers every position in the phase.

### Early backend integration proof

Run D-051's Qwen2.5-0.5B-Instruct FP16 control and D-052's real EXL3 quants
from prepared experimental artifacts alongside M2's resource-core work,
before treating the internal backend contract or executable layout as settled.
jitLLM supplies the weight and state backing, controls the stream, accounts
for workspace and backend-owned allocations, and tracks completion before
reuse. Unknown allocations remain non-evictable and budgeted. Check
teacher-forced logits against a pinned reference, then evict and restore
weights and retained state at a completed boundary and repeat the comparison
on a Spark. Include cancellation with pending work to exercise lifetime rules.
EXL3 adds packed trellis/side-vector closures, per-tensor rates/codebooks,
bounded reconstruction workspace and pointer-generation checks; conversion
to FP16 does not satisfy packed execution. Its [acceptance contract](exl3-bringup.md)
requires kernel performance against upstream in M2, full resident performance
in M5 and EXL3 switch/restore evidence in M6.
This proof informs M5 and the interfaces; it does not claim support for
flagship architectures, and there is no runtime plugin ABI to freeze (D-028).
The [proof scope](backend-proof.md) records the stages, oracle ladder and
cases. D-053 puts dispatch in jitLLM. GGML's backend runtime keeps a hidden
scratch pool, cuBLAS workspace and its own streams, so it does not execute
model work. Instead, GGML- and ExLlamaV3-derived kernels, and later others or
our own, are build-time implementations of operations. Several coexist, and
the plan selects them per operation, architecture and shape. Implementation
identity is part of the numerical plan and of cached-state identity.

## Providers

The core reaches devices and the OS only through narrow provider interfaces
(D-026). Each interface has a CUDA or Linux implementation and a
deterministic fake (the device runtime, which only the CUDA-only engine
uses, has none yet). A call that can block runs on its provider's lane, never
on the scheduler thread. This is the minimal set the pager needs; exact
signatures follow the M2 proof.

| Provider | Operations | Notes |
| --- | --- | --- |
| Device memory | Report domains, granularity and allocation classes; reserve and free address ranges; create and release backing in a class; map, set access, unmap | CUDA VMM through the driver API (D-006, D-033); device-located backing, with a host-located landing zone for direct I/O on Spark (D-081) |
| Device execution | Create streams and library handles; give implementations their stream, workspace and handles; enqueue copies between backing ranges (landing zone to device VMM and back, D-081; relocation); record a fence after a phase's last consumer; query fences without blocking | Completion is observed on its own lane; destroying an event is not retirement ([async-model.md](async-model.md#provider-checks-and-validation-gates)) |
| Device runtime | Open the device with its two providers; within a device job, copies and fills on the job's stream, timing marks, recorded work (captured and replayed graphs), the thread's error state; pinned host memory; the device's architecture and free memory | Plain functions the build's one device backend defines (`providers/device_runtime.h`; CUDA's in `providers/cuda`), one direct call around the backend's own: what the engine uses of the device besides the kernels ([portability.md](portability.md)) |
| Storage I/O | Open beneath a role directory; vectored direct reads into, and writes from, protected backing ranges (the landing zone, D-081); reserve file space; cancel; harvest completions; probe direct-I/O support | io_uring (D-034), opened through `OpenStorage`; every request ends not started, accepted or unknown. Other systems' implementations: [portability.md](portability.md#storage-and-direct-io) |
| Transport | Authenticated sessions with bounded messages and streams; register and deregister communication buffers; report send, receive and deregistration completions as observations; in M4, collectives over those stable buffers | TLS 1.3 mutual authentication (D-038); the M0 baseline ran NCCL over mapped host buffers ([environment.md](environment.md#direct-dac-cluster-follow-up-2026-09-21)) |
| Platform probe | Driver and toolkit versions, device capability, VMM granularity, direct-I/O results, RDMA devices, memory totals | Feeds `jitllm doctor` (M1, D-072) and node capability reports. The M1 cut is split: the host half in `platform`, the device half behind `providers/device_probe.h`, which the CUDA provider implements through the linked driver; direct-I/O results come with node configuration |

The fakes keep backing in host memory filled with poison patterns, so a touch
of absent backing shows up in tests. They script completion order, delays,
short reads, errors and unknown outcomes, which is what the deterministic
simulation needs (§18; [async-model
experiment](experiments/async-model/README.md)). An address-only device-memory
fake keeps the same rules and capacity with no bytes behind its backing, so a
replay can count provider calls at real scale. A fake proves jitLLM's logic,
not GPU synchronization or performance.

## Errors, faults, startup and shutdown

jitLLM code builds without exceptions, and every expected failure is an
`std::expected` error value (D-066). The error's category decides what
happens next:

| Category | Examples | Outcome |
| --- | --- | --- |
| Invalid input | Malformed JSON, broken tool links, oversized input, context exhaustion | Protocol-shaped 4xx before admission ([client-api-baseline.md](client-api-baseline.md#front-door-timing-and-admission-d-045)) |
| Unsupported | Unknown model, missing implementation or renderer, unsupported feature | Explicit error, never substitution (D-053, D-067) |
| Capacity, temporary | Queue full, over budget now | Bounded deferral, then 429 or 503 with `retry-after` (D-045, D-050) |
| Capacity, impossible | The minimum phase cannot fit, even alone | Rejected without queueing (D-050) |
| Cancelled | Disconnect, deadline, explicit cancel | Completion-safe unwind (D-048) |
| Failed, completion known | I/O error, short read, a provider error that proves no further access | The operation fails and retires; the request fails with a protocol-shaped error, in-stream once streaming has started |
| Completion unknown | Ambiguous submission, lost device or transport state | Fault: affected backing is quarantined and stays charged, affected admission stops, and status shows the fault (D-048) |
| Integrity | Verification failure, spill digest mismatch | The artifact or entry is invalidated; retention falls back to recomputation (D-055) |
| Internal invariant | Impossible transition, ledger underflow, failed heap allocation | Fatal: a bounded diagnostic, then the process aborts |

The fatal path writes no prompt content, and an abort must not leave a memory
image that holds request bodies or state (D-014). M1 packaging disables core
dumps, and the runtime also marks itself non-dumpable at startup unless the
owner opts in, so development runs meet the same rule. A core-size limit alone
does not stop a pipe handler such as the hosts' apport
([environment.md](environment.md#crash-dump-handling-2026-09-23)); M1 verifies
on each host that an abort writes neither a core file nor an apport report. A
debug core is an explicit, documented owner opt-in. A restarted runtime has a
new incarnation, so peers reject its predecessor's sessions and grants, and
startup deletes spill (M3's service adopts the conversations it kept
whole, by validated records, D-105). The runtime handles every
core-dumping signal by exiting (`_exit(128 + signal)`) rather than
dumping, is non-dumpable from before `main` for the faults a handler
cannot catch, and every thread it starts installs its own signal stack.
The unit restarts it after a crash, after a hang its recovery could not
free in the process (D-102: cancel the stuck work, reset the model, then
exit 1; [runtime-serving](runtime-serving.md#hang-recovery)) and after
exit 75 (the host not ready yet), never giving up, but not after exit
78, a refusal a restart would only repeat (D-074).

**Startup** runs in this order; a failed step stops it:

1. Read the enrollment anchor at its fixed path, before anything else
   (D-063).
2. Parse and validate the node document with its fragments, and the cluster
   document if enrolled, before opening any listener (D-063,
   [cluster-design.md](cluster-design.md#configuration-v2)).
3. Take the per-node process lock: the runtime holds an exclusive lock on
   `<anchor>.lock`, `/var/lib/jitllm/enrollment.lock` when packaged, which
   no configuration moves and `/run/jitllm`'s removal at stop does not
   touch (D-074).
4. Resolve and check the runtime's own roles (`installed`, `spill`,
   `state`: ownership, modes, nesting), comparing the job-only paths by
   text alone so a hung mount cannot stall startup. Then check the anchor
   against the configuration and `state`'s records; a mismatch refuses
   startup. Without an anchor, a standalone configuration is still refused
   while `state` holds enrollment or epoch records (D-063).
5. Run the direct-I/O probe on `installed` and `spill`, check the spill
   marker and delete runtime-named spill files (D-055); M3's service
   adopts, once its models are set up, the conversations it kept whole and
   removes everything else it finds there (D-105).
6. For each unfinished job record from an earlier incarnation, try its
   lock without waiting. A free lock means the job has ended: settle the
   record and rescan the IDs it names. A held lock keeps the models and
   artifact IDs named in the record or its grant unregistered and closed to
   new jobs until a background retry finds the lock free (D-054); the retry
   then settles the record and sweeps that job's leftovers. Then run
   artifact-format.md's staging sweep, which removes only leftovers whose
   lock is free.
7. Probe the platform and refuse unsupported configurations.
8. Register installed artifacts in the compact index; one that fails is
   reported unavailable and skipped, and artifacts without a cached entry
   finish registering after readiness.
9. Load and validate the durable authority records (D-038). Every enrolled
   node loads its highest accepted conductor epoch and incarnation, and the
   conductor durably advances its own epoch. A missing or corrupt record, or
   a detected rollback, fails closed; normal startup never initializes one.
10. Open the listeners and report readiness to systemd.

**Shutdown** stops intake and fails queued requests with 503. Admitted
requests either finish or are cancelled at a drain deadline. Then accepted
operations and registrations drain, spill files whose I/O has retired are
deleted, and backing is released (D-048; drain-before-restart upgrades in
features.md). Startup deletion covers crashes. Work whose completion cannot
be reconciled faults the shutdown instead of reporting its capacity
reclaimed. M3's service instead spills its idle conversations at a
graceful stop and keeps them, recorded, for the next start (D-105).

## Front door and management

### Listeners and protocols

The conductor, or a standalone node, serves one inference front door (port
8114 on loopback and the tailnet by default, D-097) and the management
listener (`127.0.0.1:8115`) (D-037, D-045, D-063). Cluster-wide management
goes through the conductor (D-038); whether a worker also serves a loopback
listener for node-local operations is settled in M6a. A non-loopback
management binding requires credentials and TLS (D-064); no inference
binding requires credentials, an inference credential being optional (an
API key, as llama-server and vLLM offer; D-014's owner note, D-097). TLS
comes from certificate files that external tools keep current,
selected by SNI and reloaded on change, with the local CA as the fallback
(D-065). The proposed baseline is HTTP/1.1 with keep-alive and SSE streaming,
adding HTTP/2 only if a named client's tests require it. The HTTP and JSON
libraries are M3 dependency choices, and the TLS library M5's, under D-017,
D-057 and D-066:
no-exception APIs, bounded buffers, and non-blocking integration with the
network lane.

Protocol adapters sit at the edge: Chat Completions, Responses, Messages
with `count_tokens`, model listing, and later the Ollama subset (D-040,
D-041). Each one parses its wire format into the protocol-neutral request,
and turns the neutral event stream back into its own format, including
keepalives, in-stream errors and alias echo (D-045–D-047). Extensions use
the `jitllm-` header prefix and one `jitllm` body object (D-062). Discovery
answers from catalog snapshots with no I/O. Inference credentials never
carry management authority.

### Management API and CLI

The management API is versioned by its route prefix (D-062). On loopback it
is anonymous behind browser guards (D-064). It covers models (list, inspect,
install, remove, archive), jobs (progress, cancel), residency policy and
priorities, request cancellation, node status with the memory breakdown, the
admission what-if query and trace capture (features.md). Changes reach the
scheduler as commands, never as direct writes to runtime state. The
`jitllm` CLI is a client of this API.

## Cluster

### Conductor ownership and admission

D-037 places the conductor in its designated node's native runtime. The
single-node deployment uses that same role and local admission path.
Dashboard, importer and supervisor remain separate processes. Conductor
work uses bounded queues and buffers, charged to its node's budget, and
never enters per-expert dependency acquisition or residency decisions.
Cluster coordination does not hold a catalog/scheduling lock across network,
disk or GPU waits. D-048's [task/completion design](async-model.md) selects
explicit task states, a single scheduler/catalog writer and bounded service
lanes; exact worker counts and polling policy remain implementation choices.

The following are conceptual records, not a wire schema or public API:

| Record | Owner and meaning |
| --- | --- |
| Node view | Conductor's advisory snapshot: configured node identity, runtime incarnation, report revision/freshness, capabilities/compatible plans, health, budget and occupancy/commitment summaries. Reports from an old incarnation or older revision cannot overwrite newer state |
| Local capacity ledger | Node authority: its execution budget, outstanding capacity commitments and full execution envelopes under D-050. Physical occupancy is a separate ledger; cache and lazy commitments are not naively summed or counted as free memory |
| Placement | Conductor intent and node-confirmed model-instance identity: artifact/plan compatibility, node incarnation, readiness or unknown status. Separate instances can represent future replicas or M4 ranks. Weight residency and extent ownership remain in the node catalog |
| Retained-state hint | Node-issued, compatibility-scoped hint for placement affinity. The node revalidates existence, identity, permissions and expiry at use. A prefix hit is neither conversation identity nor a refresh of unrelated continuation retention (D-031) |
| Routed attempt | Conductor request/attempt identity, conductor incarnation, target node incarnation and model instance, dispatch/admission/start/terminal-or-unknown status, and stream progress. The node owns the matching execution record and any capacity grant; observations at the conductor may lag |

Snapshots include enough budget categories to interpret admission: active and
suspended state, weights/cache, workspace, I/O and communication buffers,
metadata, non-evictable allocations and OS/runtime headroom. Shared backing
is counted once locally. Pending retirement remains occupied until local
completion proves otherwise. Unavailable-node capacity cannot satisfy a
request elsewhere. Neither aggregate free bytes nor reported model residency
is permission to run. D-050 defines envelope guarantees; their numeric bounds
and implementation proof remain M2 work.

**Whole-model placement (M6a).** Filter candidates by configured membership,
current runtime identity, health and compatible executable plan/artifact.
Prefer a feasible placement that avoids paging, using compatible retained
state and existing model instances as affinity hints; preserve useful contents
on other nodes. A node with revocable cache is not automatically full, and
an affinity hit does not override admission. Ranking predicts suitability;
the chosen node checks reality. No numerical ranking formula is claimed here.

Dispatch one attempt to one node, including when that node is local. The
node validates identity and plan/state compatibility and serializes its
capacity decision with other local admissions. It either accepts under its
progress policy, defers with bounded waiting, or rejects an impossible or
invalid request. Deferral is not a capacity grant unless the node explicitly
issues one. Granting capacity alone never evicts useful cache; the node later
acquires the actual dependency closure and schedules at safe boundaries.
Switching away from an instance does not evict the whole model or expire its
retained state. Adding replica records here does not enable automatic replicas.

An attempt's identity is stable across transport retransmission. The node
must return its existing outcome or reject a retired identity, never turn a
duplicate into another execution or commitment. Deduplication/terminal records
are bounded, but pruning them must not make old requests executable again:
retire their admission namespace or retain a rejection watermark/equivalent
fence. D-038 uses session sequence high-water marks for this bound; M6a must
validate their implementation rather than rely on unbounded request tombstones. This is internal dispatch safety, not durable
exactly-once semantics for client retries.

A rejection or an acknowledged cancellation of queued work permits trying
another node only if the original node establishes **never started and no
longer startable**, including rejection of delayed dispatch messages. If
acceptance/start is uncertain, query or cancel the same attempt on that node;
if it cannot be resolved within bounded time, fail the client explicitly.
Do not reroute on a timeout, stale health, or absence of tokens: the node may
already have changed state or begun execution. A terminal failure after work
started likewise does not authorize transparent replay.

**Streaming and cancellation.** The executing node sends ordered response
chunks to the conductor, which owns the client connection and preserves the
protocol's terminal/error semantics. The local execution path obeys the same
ordering and buffer limits. Slow clients apply bounded backpressure; if the
configured buffer/time bound cannot be maintained, cancel or fail instead of
accumulating unbounded output. No global execution/catalog lock spans a stream
write. Client disconnect cancels the routed attempt. The node stops further
submission at supported boundaries and tracks pending GPU, I/O and network
consumers through completion; cancellation acknowledgement is not a physical
reclamation receipt. A request may be terminal for the client while cleanup
remains outstanding. Transport protection/authentication follow D-014; workers'
internal control endpoints are not extra public inference front doors.

**Restart and uncertain ownership.** Node and conductor process incarnations
qualify all attempt and control identities. Restarted nodes reject old grants
and dispatch; old state hints and placements become invalid until explicitly
revalidated. Process restart is not itself proof that driver/network resources
are reusable: local recovery must establish safe ownership and completion
before reporting a usable budget. On conductor connection loss, a worker
blocks new admission from that connection and initiates bounded cancellation/
unwind of its orphaned work when the connection-loss deadline is reached.
A deadline ends client waiting, never substitutes for device completion.
Resources whose consumers cannot be proven finished stay charged/quarantined
until a validated recovery establishes safety; report the node unavailable
instead of promising indefinite successful progress.

Reconnecting or restarting the conductor cannot reconstruct truth from its
old snapshots. Each worker reconciles or cancels outstanding attempts, fences
old control/admission sessions and reports current local status before new
admission through a replacement session. Old attempts may still be retiring;
new work can use only capacity the local ledger safely makes available. M6a
has no automatic election, failover, stream resumption or durable replay log.
Replacement of the configured conductor first requires fencing the previous
authority; an unreachable process is not proof it is dead. Authentication,
session fencing, bounded control-record retention and reconciliation mechanics
are specified in the [initial cluster design](cluster-design.md), D-038;
they still require implementation validation.

**M8 extension.** A sharded placement maps ranks to separate node domains
(M4's full swaps run one first, without this transaction).
The conductor coordinates a phase transaction with node-issued capacity
reservations and readiness for every rank; all required ranks must be ready
before a matching commit authorizes execution. Each node validates the current
transaction/incarnations and its own grant before collective submission.
Prepare failures cancel/unwind participating ranks; unknown rank completion
never releases another rank's still-consumed buffers. Collective ordering
arrives in M4; commit/abort races and failure recovery require the M8
protocol and tests; M6a whole-model routing does not require distributed
prepare/commit. No cluster ledger may replace these local authorities with a sum of free bytes.

**Required validation, not results.** The M6a fake transport/node tests and
Spark integration must cover: two attempts racing on stale reported capacity;
affinity pointing to expired state; out-of-order reports and stale incarnations;
duplicated/delayed dispatch after cancellation and dedup-record retirement;
lost acceptance before any token; slow/disconnected clients; conductor restart
with a surviving worker; and node loss with pending GPU/I/O consumers. Assert
no over-admission, duplicate execution, silent rerouting or early reuse, bounded
client/queue waiting, and continued accounting for unresolved cleanup. M8 adds
partial preparation, lost commit, mismatched rank generations and node loss
during a collective. These tests are owed at implementation, not run in M0.

### Two-node flow (§12)

Placement first (D-020, D-023, D-037, D-038): the conductor decides which node hosts each
model and routes requests there; a subagent's model on another node while the
main model stays resident needs no collective and no direct link. M6a starts
with enrolled membership and one configured conductor, detected network
paths, canonical QSFP layouts and bounded setup subnet scans under
[D-038/D-039](cluster-design.md), capability and
health probes, and request routing with affinity to retained compatible
state. The node runtime remains authoritative for local admission; a stale
cluster view cannot authorize unsafe local execution. Unavailable nodes
cause explicit request failure, not assumed reclamation or silent replay of
an already-started stream. Automatic replica placement, automatic membership changes, and
conductor election have separate revisit triggers in plan.md. A busy small
model may later run as replicas on several nodes. Concurrent execution
without paging requires a supported placement whose working sets and complete execution envelopes fit
each node's budget; aggregate pool capacity alone is insufficient. M6a depends
on M6, not on demand-paged MoE. Sharding, below, is M4 for the flagship models.

Describe the phase and local requirements per rank → reserve capacity on all
required nodes → establish local residency → commit distributed execution →
preserve collective order → acknowledge completion or cancellation. Local
eviction victims may differ per rank. Communication buffers come from a
separately budgeted pool with stable backing. TP, PP, and EP are different
plans; port the validated recipe's plan first. The first external sharded
reference ([MiMo TP=2/EP=2](experiments/mimo-reference/README.md)) moved about
0.84 MB per prefill token and 4 MB per decoded token each way at 18.5
tokens/s, a small fraction of the measured link: per-step collective latency,
not bandwidth, is the first transport question for M4.

## Configuration

Each node reads one strict TOML 1.0 document, `/etc/jitllm/jitllm.toml` plus
its `jitllm.d/` fragments, and the shared cluster document when enrolled
(D-063, [cluster-design.md](cluster-design.md#configuration-v2)). Unknown
keys, duplicates, and type or range errors are fatal before any listener
opens. Credentials are referenced by path, never inline. Configuration is
read only at startup; certificate files are the exception and reload on
change (D-065). Runtime policy changes, such as a budget reduction, go
through the management API as scheduler requests. Membership and trust
changes need cluster-design.md's coordinated restart. D-063 records the
`[storage]` keys and D-073 the rest of M1's spellings, the one-owner merge,
the trust checks on the files and roles, and the diagnostics; M5 adds the
front door's and TLS keys and M6 the switching policy's.

## Observability and privacy

- **Events.** Admission, deferral and rejection; page-in, eviction and
  spill; retention hits and misses with their reasons; faults; jobs. Each is
  a typed, bounded record keyed by opaque IDs, kept in a bounded in-memory
  ring behind the management API and summarized in counters. Operational
  logs go to the journal (D-063). A full ring drops its oldest detail and
  counts the drop; it never blocks the scheduler or grows.
- **Explanations.** The what-if query and eviction events read the same
  ledgers the scheduler decides from, so an explanation describes the
  decision actually made (vision.md).
- **Metrics** follow D-044's monitoring contract. **Traces** export the I/O
  timeline and scheduling in Perfetto/Chrome trace-event format for an
  explicitly requested capture window (features.md).
- **Privacy.** Prompts, token IDs, rendered text and state contents are never
  logged (D-014, D-055). Routing and expert choices are captured only on an
  explicit management request (D-064). The runtime writes nothing to the
  long-term store, and spill stays node-local with mode 0700 (D-054, D-055).

## Trust boundaries

| Input | Trust | Handling |
| --- | --- | --- |
| Client requests | Untrusted, even when authenticated | Bounded before any work; D-045's guards; strict parsing |
| Checkpoints, the long-term store, archives, peer transfers | Untrusted | Read only by job processes, in a confined parse stage; lengths, paths, hashes and metadata validated; no checkpoint code runs in a jitLLM process, and a chat template runs only in the runtime's bounded, sandboxed interpreter (D-009, D-054, D-067) |
| Installed artifacts | Verified at install, then protected by the store's permissions | The runtime still parses manifests and indexes strictly and bounds-checks every range before use |
| Cluster messages | Authenticated peers | Mutual TLS with pinned identities, bounded framing, validated records; no raw addresses or unvalidated paths cross (D-038) |
| Management requests | Local processes | Loopback plus browser guards; credentials and TLS when bound elsewhere (D-064) |
| Configuration, credentials, certificates | Files only root or the runtime's user can replace | Strict parsing; path, owner and mode checks (D-063, D-065) |
| Spill files | Written by this process | Restores are checked against catalog digests; spill is deleted at startup (D-055) |

The runtime and the jobs run as `jitllm`, never as root. The security model
is a single-owner node; multi-tenant isolation is a non-goal (vision.md).

## Performance evidence

The M0 paging-feasibility study used a reference engine because jitLLM's
execution path did not exist yet, and the protocol below still governs such
studies. Record checkpoint revisions, quantization/layout,
expert sizes, request ordering and timing, prefill chunks, decode batches,
context/output lengths, and memory reserved for non-pageable resources.
Capture routes only for deliberately enabled benchmark sessions (D-014).
Compare policies at the same total node budget; preserve the distinction
between cold storage, warm OS cache, and warm runtime residency. Record
assumptions about overlap, mapping overhead, and contention when turning
trace replay and measured I/O into estimates. These estimates guide scope;
actual end-to-end results must later validate them.

Once the pinned reference runs, measure A→B→A on the target (D-025). Start
with one reproducible conversation on A, a request to B, and a continuation of
A's history. Include an all-resident control and a constrained budget that
forces displacement. Measure the full switch and switch-back interval,
including any state writes, unload/load, restore or re-prefill, and first
returned token. Report queue delay behind a running request separately from
paging and switch time and from first-token compute (D-069). Enable applicable
reference routing and state-save features, verify them per checkpoint, and
record any harness actions needed to use them. Pin the trace and settings so
M3 and M6 can repeat the same experiment.

Every backend/paging performance comparison has two views:

- **Matched configuration:** align checkpoint, numerical policy, request
  workload, context lengths, prefix-cache conditions, and decoding features.
  Disable speculative decoding in both paths if jitLLM lacks it. Compare
  jitLLM's resident and paged paths separately to expose paging overhead.
- **Normal reference configuration:** also run the pinned reference's normal
  documented configuration, including its enabled optimizations. Report its
  actual settings and feature differences. This measures the user-visible
  gap; do not attribute the whole gap to paging.

For speculative runs, record drafter identity, settings, acceptance, and
memory use; report throughput per accepted output token. If a matched run
cannot be made, record why and leave its comparison unvalidated. D-036 records
owner-accepted targets for named supported workloads: M6's median/p95 floor
against the fastest correct full-swap reference arm in both directions; M7's
at most 10% added generation time, continuation time to first token included,
and 20 ms p95 / 100 ms p99 added token gaps; and M9's at least 25% median
return-switch benefit over jitLLM's own whole-model control on an agreed
partial-retention workload, with at least one named library exceeding
physical memory. Pin workloads, trial counts, and measurement methods before
acceptance runs, repeat the correct reference and the whole-model control
alongside them, and report uncertainty; inconclusive comparisons do not pass.
The resident generation control with matched state provenance, the
same-budget whole-model control, and the full-swap switching reference are
three distinct controls (D-036). Record latency distributions, bytes read and
written, peak memory/spill occupancy, and prompt tokens reused versus
recomputed. Never invent thresholds or measured results.
M6 validates switching, M7 validates MoE paging, and M9 validates subsequent
optimizations against these criteria; scope changes when evidence warrants it.

### Comparator: Athena's Engine (closed source, creator-reported)

Announced 2026-09-19 on the NVIDIA developer forum by its author; numbers
measured 2026-09-18 on a single GB10. Closed source, so nothing is reusable;
free for personal, research, and small-company use at the time of reading,
so installing it on a Spark as a second comparator is permitted (verify the
current terms first). All figures are the author's, not independently
verified, and include its speculative-decoding sidecar, so they belong in the
normal-reference view, never the matched one.

| Item | Reported |
| --- | --- |
| Models | DeepSeek V4 Flash (IQ2_XXS mix, Q8 projections); Qwen3.8 Flash Next (Unsloth UD-IQ4_XS); GGUF only |
| Prefill | about 1,070 to 1,126 tok/s at 8k context; about 950 to 960 tok/s at 256k |
| Decode, 256 tokens | DeepSeek 21.4 tok/s at 8k, 19.4 at 256k; Qwen3.8 29.9 at 8k, 32.1 at 256k; flat over context |
| Model switch | drain, flush checkpoints, release, verify memory is free, load the other: 46 s measured |
| Context restore | a 141,519-token conversation restored from a 619 MB file in 2.1 s, versus 2 min 20 s to re-prefill |
| API | OpenAI- and Anthropic-compatible, streaming and tool calls |
| Sessions | checkpoints the current agent or session to disk before unloading, so the switch preserves conversation state; one endpoint serves both models |
| Memory | the two models at those bit depths do not both fit in 128 GB, per the author's X thread |

What it tells us. The 46 s switch includes checkpointing the active session,
so it is a real-world floor for A→B→A with state preserved on this exact
model pair on one GB10 (D-021, D-025). The pair is M9's named large-model
configuration (D-036), where the target is to beat it clearly at comparable
bit depths; our own measured baseline still governs. Because
the pair does not both fit in 128 GB, it is the canonical two-large-model
switching workload for the feasibility spike. The restore figure implies roughly 4.4 KB of restorable state per
token for Qwen3.8's hybrid attention, a concrete datapoint for D-024's
retention budgets and the spill/restore gate, and it confirms D-025's
caution that a comparator need not lose conversation state on a swap. Its
"checks the memory is really there" step is the unified-memory accounting
problem of D-004 seen in the wild. Its offering both API flavours was early
evidence for the Anthropic Messages surface, since confirmed (D-030, D-040).

## Testing

Tests follow §18's layers, and D-061's local tiers decide where each runs:

| Layer | Covers | Tier |
| --- | --- | --- |
| Native CPU | Catalog and aliasing, ledgers, victim selection, retention identity, parsers, artifact verification, renderers against golden fixtures | `check`: x86-64, and AArch64 under qemu-user |
| Deterministic simulation | Fake providers with scripted delays, reordering, short reads, errors, unknown outcomes, full queues and cancellation races; the D-050, D-055 and cluster adversarial matrices | `check` |
| Sanitizers and packaging | ASan and UBSan, the copyleft-disabled build, offline source gates, `.deb` install | `check:full` |
| Spark device and I/O | Map, load, verify, evict, restore; leases across streams; actual kernel reads; direct I/O on the target NVMe; memory-ordering stress; LSan and TSan | `check:spark` |
| Model semantics | Teacher-forced logits and intermediates on the oracle ladder, before and after restoration ([backend-proof.md](backend-proof.md#numerical-oracles)); M3 adds the speculative contract and M9 the diffusion one ([model shapes](#model-shapes)) | `check:spark` |
| Multi-model pressure | Partial eviction, reuse, fairness, the M6 A→B→A workload | `check:spark` |
| Two-node | Placement, routing, fencing and failure; later sharding and collectives | `check:spark` on both Sparks |
| Client acceptance | Named clients against pinned profiles ([client-api-baseline.md](client-api-baseline.md#acceptance-owed-in-m5)) | Recorded runs; aggregate evidence in Git |

Every pager invariant has tests on the fake backend and on hardware. A test
that injects a fault asserts bounded occupancy and either eventual
completion or an explicit, safe failure; the absence of an out-of-memory
error is not a pass (D-050).

## Build, repository and installation

### Build

CMake 4.4.3 presets with Ninja drive the native x86-64 and cross AArch64
builds, linked with LLD, plus D-032's native Spark diagnostic profile with
GNU binutils (D-032, D-058, D-059). Every build names
explicit CPU and GPU targets (`sm_121` for GB10, and in the x86-64 `native`
build the discrete `sm_86`, D-082), never `-march=native`
(D-011). The C++ and CUDA runtimes link statically, so a binary needs only
glibc and, in CUDA builds, the NVIDIA driver's `libcuda.so.1` at run time
(D-060, D-072). The driver is a hard requirement; the build links NVIDIA's
stub from the SDK. The exception is cuBLAS: binaries that use it will link
it dynamically, and the package will then ship its two pinned libraries
(D-076). Once RDMA is linked, rdma-core joins them. Sources come through
D-057's locked acquisition, and tools through the mise-managed SDK (D-049).
Build profiles select optional modules; the copyleft-disabled profile
excludes them before any source is fetched. `jitllm --version` and the build
receipt carry the product version, commit, license profile and SDK identity
(D-062).

### Repository shape (§20; a proposal for M1)

```text
CMakeLists.txt  CMakePresets.json  mise.toml  mise.lock  .devcontainer/
LICENSE  LICENSES/  NOTICE  CHANGELOG.md
toolchains/{manifest.toml, artifacts.lock.json, provenance.toml}   cmake/toolchains/
src/base/  src/platform/  src/providers/{fake,cuda}/
src/catalog/  src/memory/  src/retention/  src/scheduler/
src/artifact/  src/model/  src/execution/  src/kernels/{ggml,exl3}/
src/api/  src/cluster/  src/management/  src/jobs/  src/runtime/  src/cli/
modules/      optional implementation modules, each under its own license
third_party/  curated vendored sources and patches with provenance (D-057)
packaging/    Debian package, systemd units, sysusers and tmpfiles (D-063)
tools/        setup, check, doctor and build-time tooling
tests/{toolchain,unit,simulation,cuda,model,distributed,packaging}/   benchmarks/
dashboard/ (M10)  docs/
```

The toolchain file set was confirmed on 2026-09-21. File metadata uses
embedded headers and `.license` sidecars, not REUSE.toml (D-071). The source directories
follow the [layers](#layers-and-dependency-rules), and M1 may rename them.
There is no public `include/` tree (D-064). `docs/` keeps this scaffold's
single-file decision and findings logs rather than §20's `docs/decisions/`
directory. Directory names do not establish legal isolation; each module's
license is explicit.

### Conceptual native API (§20; confirmed 2026-09-21 as the starting shape; a sketch, not compilable)

`register_resource(descriptor)`, `reserve_capacity(transaction, envelope)`,
`acquire_group(reservation, dependencies)` → ready | deferred | impossible |
cancelled | failed, `submit(plan, lease, context)` → owned submission record
(prepared before provider access, reconciled as not-started/accepted/unknown;
accepted leases stay with completion tracking), `retire_completed(token)`,
`reclaim(extents)` (validates generations, reports actual bytes recovered),
`cancel(transaction)`. Deferred results refer to owned continuations, not a
blocked global scheduler. D-048 specifies bounded task states and completion
ownership in [async-model.md](async-model.md). There is no runtime plugin ABI
(D-028); optional backends are build-time modules behind the operation contract, which is
finalized after the M2 GGML and EXL3 proofs (D-052).

### Installed layout

D-063 records the packaged layout, D-074 the package that implements it,
and D-062 versions the configuration schema; M10 adds the repository.

| Path | Owner / mode | Holds |
| --- | --- | --- |
| `/usr/bin/jitllm` | root | User-facing CLI |
| `/usr/libexec/jitllm/` | root | Node runtime process (`jitllm-runtime`, D-074), the import/install/archive job processes (D-005, D-054), and the certbot deploy hook and Tailscale certificate script (D-065) |
| `/usr/lib/systemd/system/jitllm.service` | root | The runtime's one unit; runs it as `jitllm`. An optional, disabled-by-default Tailscale certificate timer and service ship alongside it (D-065) |
| `/usr/lib/sysusers.d/jitllm.conf` | root | `jitllm` system user and group, no login shell |
| `/usr/lib/tmpfiles.d/jitllm.conf` | root | `d` lines for `/var/lib/jitllm` (`jitllm` 0755) and the default `checkpoints` (`jitllm` 1777, applied only on creation); no age, so never cleaned |
| `/etc/jitllm/jitllm.toml` | root, not shipped | Optional main file of the node document (`schema_version`, strict); with no main file or fragments, standalone loopback defaults, refused while `state` holds enrollment or epoch records |
| `/etc/jitllm/jitllm.d/*.toml` | root, not shipped | Fragments of the node document, read in lexical order; a key other than `schema_version` set in two files is fatal. Setup tooling owns its own fragment |
| `/etc/jitllm/cluster.toml` | root, not shipped | Shared membership document for cluster members (D-038/D-039) |
| `/usr/share/doc/jitllm/examples/` | root | Annotated example node document |
| `/etc/jitllm/credentials/` | `root:jitllm` 0750 | Credential files referenced by path, never inline in TOML |
| `/etc/jitllm/tls/` | `root:jitllm` 0750, files 0640 | Front-door certificate files (combined PEM or cert/key pairs), written atomically by the certbot deploy hook, the Tailscale timer or the owner (D-065) |
| `/var/lib/jitllm/` | `jitllm` 0755 | `storage.data_dir` (absolute), the base for relative role paths |
| `/var/lib/jitllm/enrollment` | `jitllm` | Enrollment anchor (node and cluster IDs, `state` path, enrollment ID) at a fixed path independent of configuration; while present, startup refuses a configuration or `state` that does not match it. Moving `state` or leaving the cluster is a setup step that rewrites or removes it (D-063) |
| `…/models/` | `jitllm` 0755 | `storage.installed`, including D-056's `.staging/` (0700); artifact directories 0755, files 0644; readable by all, written only by the runtime's user |
| `…/checkpoints/` | `jitllm` 1777 | `storage.checkpoints` (node-local default, created by the package's tmpfiles entry); anyone may add sources, which jobs treat as untrusted |
| `…/spill/` | `jitllm` 0700 + marker | `storage.spill` (D-055) |
| `…/state/` | `jitllm` 0700 | `storage.state`: durable runtime records (conductor epochs and floors, job records and their locks, install generations) and the disposable compact-index cache, plus the local CA's key and leaves (D-065) |
| `/var/lib/jitllm/enrollment.lock` | `jitllm` 0600 | The per-node process lock, held by the runtime (D-074) |
| `/run/jitllm/` | `jitllm` | Runtime sockets |
| `/usr/share/doc/jitllm/` | root | `copyright`, `NOTICE`, changelog, SBOM (D-029) |

The optional `storage.long_term` (unset by default) holds `archive`
(default `<long_term>/archive`; an archive requires it) and, only if the
user points it there, the checkpoint store. Only job processes open it; the
runtime opens only `installed`, `spill` and `state`, and `long_term` may not
equal, contain or sit inside any of them. After canonical resolution (links
followed, device and inode compared) no two role paths may be equal or
nested; the runtime checks the job-only paths' text against its resolved
roles, so a hung mount cannot stall it, and jobs repeat the full check. The
runtime refuses its roles if users other than root and its own user
(`jitllm` when packaged) could write or replace them, and probes
`installed` and `spill` with D-034's direct-I/O check at startup; those
checks, not the text comparison, reject a network mount aliased into its
roles.
`jitllm.toml` and its fragments form cluster-design.md's node-local document
extended with `[storage]`; a standalone node omits its cluster keys, including
`[credentials]`. Logs go to the journal. The package depends on glibc and
the versioned `libcuda.so.1` virtual package; the C++ and CUDA runtimes are
static (D-060). The process that uses a role creates it when missing. The
front door (conductor or standalone node) defaults to port 8114 on
loopback and the tailnet (M3's chat route, D-097) and the management API to
`127.0.0.1:8115`.

## Open architecture questions

The M0 open questions are answered or deferred in
[features.md](features.md#open-questions-answer-during-m0). This draft adds
the exception policy (D-066), chat rendering (D-067) and the model-shape
rules (D-068), and sets initial
designs for the [provider interfaces](#providers),
[victim selection](#victim-selection-initial-baseline) and the
[memory breakdown](#memory-breakdown). What remains needs implementation
evidence or a later choice:

| Question | Settled by |
| --- | --- |
| Worker counts, queue sizes, wakeup and polling for D-048's lanes; lost-wakeup and memory-ordering evidence | M2 |
| The operation contract's exact types, the implementation registry, per-operation GGML integration, phase envelopes and `F` | M2 backend proof (P6) |
| Keep or amend D-033: independent handles versus slab slots, under criteria approved before measuring | M2 [retained-backing comparison](backend-proof.md#retained-backing-comparison) |
| Which OS counters include VMM backing on the Spark driver, so the memory breakdown can reconcile | Settled in M2 ([vmm-counters](experiments/vmm-counters/README.md)) |
| Storage queue depths, run sizes and polling with real model traces; mixed read/write scheduling and the spill write budget | M2, M3 (the swap path), M6 |
| State block sizes and KV layouts per state adapter | M2/M3 |
| HTTP, TLS and JSON libraries (TOML: toml++, D-073; general JSON so far jitLLM's own `base/json.h`, M3) | M3 (HTTP, JSON); M5 (TLS), under D-017, D-057 and D-066 |
| How jobs are launched and report back (their containment and confinement are D-074's); the peer-replication transfer mechanism | M5; M6a |
| Switching-policy default and tuning (minimum run, pause cap, deadline handling) | M6 comparison of the D-069 policies |
| Whether a worker node serves its own loopback management listener for node-local operations | M6a |
| Host versus device sampling | M3, measured against D-052's decode gates |
| Retention capacity values (`M_state`, `S_spill`, entry counts, `L_prefix`, maintenance interval) | M5 exit (D-055) |
| Cluster dependencies: hardware profiles, bootstrap discovery, parsers, authenticated sessions, crash recovery | M6a (D-038) |
| Expert dispatch: a pointer table versus a uniform stride | M3, the initial choice per format for resident experts; M7 GGML proof, compaction and demand paging |
| Model-parallel artifact partitioning | M4 entry |
| Optimistic MoE execution, including a miss when the current step fills memory | After M7's pessimistic path is correct and measured |
| Companion and multi-component artifacts: manifest references by artifact ID, shared-resource accounting | M3 (D-068) |
| Per-step expert closures of wide phases (speculative verify, diffusion canvas) on routed experts | A bounded DiffusionGemma reference study, before M9 planning |
| Numerical bounds and trace protocols for speculative and diffusion decoding | Before execution: M3 (speculative), M9 (diffusion) |
