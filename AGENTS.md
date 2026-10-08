<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# jitLLM — just-in-time LLM inference engine with intelligent SSD paging

jitLLM is an independent, open-source inference runtime for workloads where
many models should be available, only some components are active at a time,
and aggregate model storage exceeds physical memory. It keeps the useful parts
of models resident, reclaims the least valuable extents across all models when
capacity is needed, and brings missing weights or state back on demand from
prepared on-disk artifacts. The primary workload is one user switching among
a library of models larger than memory, with conversation state preserved
across switches (D-019). Initial target: one or two NVIDIA DGX Sparks,
developed from an x86-64 Linux workstation, whose discrete NVIDIA GPU is a
secondary target (D-082). Almost all code is written by AI
agents working from the project documentation, directed and reviewed by a
human.

**Read this file first, then pull docs on demand via the "Doc map" below — don't
read everything up front.** This file is long-term project memory and the
rulebook for agents.

## Load-bearing constraints (change deliberately, never silently)

Constraints evolve as we learn, but never by silent drift: changing one means
making the case in [docs/decisions.md](docs/decisions.md) and updating the
affected docs. Until then, these govern.

- **Our own native runtime, one process per node.** jitLLM is not a vLLM fork
  or plugin. One modular native execution process per node owns all local
  model execution, scheduling, memory policy, VMM control, and completion
  tracking. Dashboard, importer, and supervisors are separate processes and
  never in the per-expert hot path. (D-005)
- **Optimize for one user switching models, not mixed-traffic throughput.**
  The primary workload is a single user, or an agent plus subagents on
  different models, switching among a library larger than memory with
  conversations spanning hours. Under contention models are time-sliced;
  when a supported placement fits each node's full execution budget they run
  concurrently. A single conductor, the cluster's one point of entry, places
  models across nodes, routes requests, and may run replicas of a busy small
  model; placement is preferred over paging when it suffices. Topology is
  configured or discovered, never baked into the app. The floor is "never
  worse than a full swap", validated against a measured reference cycle.
  State reuse has bounded retention; shared prompt prefixes and conversation
  continuations have independent reuse and expiry (D-031). Prefix matching
  identifies neither a conversation nor its lifetime. Standard web-API clients
  (Cursor, OpenCode, Codex, Claude Code) work unmodified; sessions and hints
  are optional extensions.
  (D-019 to D-025, D-030)
- **Spark memory is one physical budget; two Sparks are two domains.** CPU
  allocations, GPU backing, staging, and page cache share 128 GB of unified
  memory, so CPU offload is not a second tier. Two nodes are two memory
  domains connected by a network; cross-node access is explicit object
  transfer, never shared virtual memory. Weights and state live in device
  VMM (the GB10's L2 does not cache host-located memory). On validated
  Spark configurations, direct file reads land in a bounded host-VMM zone
  and the GPU copies each extent into device VMM; no CPU payload copies.
  Native GDS or GPUDirect RDMA is not assumed. The storage backend itself
  is not Spark-specific.
  (D-004, D-034, D-081)
- **Explicit CUDA VMM plus a node-wide resource catalog.** Backing is
  reserved, created, mapped, and unmapped by us through the driver API.
  Accessing absent backing is a bug, not a page-in request. Every managed
  allocation is registered with semantic metadata; unknown allocations are
  non-evictable. Logical identity, current address, and physical occupancy
  are separate things. (D-006)
- **Reservations are not leases; release is not eviction.** Virtual
  reservation, capacity reservation, and residency lease are three different
  concepts — never say "reserved" without saying which. Grants commit lazily
  and never eagerly evict useful cache. Releasing a lease changes eligibility,
  not residency. Never hold the global scheduling/catalog lock across GPU,
  disk, or network waits. (D-007)
- **Partial eviction and on-demand experts, with no substitution.** Reclaim
  specific extents across all models, never whole models by default. Routing
  is a dependency-discovery stage: acquire the selected experts' closure,
  run, release. Never substitute a resident expert for a selected one or drop
  a selected contribution; prefetch is speculative, routing is authoritative.
  (D-008)
- **Prepared artifacts; checkpoints are untrusted input.** Models execute
  from versioned, hashed, execution-ready on-disk artifacts produced at
  import, never from raw checkpoints. Artifacts never serialize process
  addresses or runtime objects. Import validates lengths, paths, hashes, and
  metadata and never executes checkpoint code. Initial formats are explicitly
  experimental; compatibility guarantees follow execution/restore evidence.
  Import repacks weights into contiguous, indexed dependency groups
  (4 KiB-aligned on disk, paged in 2 MiB chunks with direct reads, which
  may coalesce; off by default, BP-P1); v0 uses safetensors shards with a jitLLM manifest/index and no
  page-in hashing. (D-009, D-018, D-035, D-056)
- **C++23, Clang-first, native hot path.** No interpreter in the serving,
  paging, or scheduling path. NVCC is the CUDA compiler with Clang as host
  compiler where validated. Build-time tooling may use Python. (D-010)
- **jitLLM owns dispatch; kernels are swappable build-time implementations.**
  jitLLM owns streams, workspace, library handles, fusion choice and
  completion; third-party backend runtimes never dispatch model work. Kernels
  from GGML (first), ExLlamaV3 (a real EXL3 companion is required in M2 before
  settling the artifact/backend contracts; once operational, each engine
  runs at least as fast as its reference, end to end, D-085),
  other sources or our own (on measured need) implement operations under one
  contract; several
  coexist, and the plan selects per operation, architecture and shape, by
  correctness and speed. No runtime plugin ABI. (D-028, D-052, D-053, D-080)
- **NVIDIA first; portable boundaries when free.** The core holds no vendor
  types; device memory, paging, and transport go through narrow provider
  interfaces, with CUDA VMM the only implementation for now. Discrete
  NVIDIA GPUs are a secondary target: one device-memory domain plus the
  SSD, fast swaps with one model active; host RAM is not a tier (designed
  for as a separate domain, not built). Apple silicon is in scope later and
  AMD is plausible: do nothing for them, sacrifice nothing on the GB10, but
  don't foreclose them. Intel is out. The CPU-only build and the fake
  backend are the guardrail. (D-026, D-082)
- **Develop on x86-64 Linux, cross-build, test on Spark over SSH.** Native
  builds and CPU tests, including AArch64 CPU tests under qemu-user, run on
  the workstation; ARM concurrency, VMM, kernel, GPU, RDMA/NCCL, and
  distributed tests run on the Sparks, which in the owner's environment
  are `spark` and `spark-b` (inventory in environment.md; those names are
  not application configuration); `gpu-discrete` tests also run on the
  workstation's GPU, only when asked (`mise run test -- native --gpu`).
  Explicit CPU/GPU targets only, never
  `-march=native` or autodetection. Toolchain provisioning is declarative and
  pinned. Agents never invent compiler pins, measured numbers, supported
  model combinations, or license permissions. Each slice's check builds
  natively on a Spark. Nothing that runs over 10 minutes (the
  workstation tiers, timing batches, nsys passes) runs unless its result
  is needed now. New Mia reference runs use the [pinned fast-start launcher](docs/experiments/fast-swap/baselines.md#default-mia-launcher-for-new-runs-2026-10-03).
  (D-011, D-012, D-084, D-085)
- **Apache-2.0 core with license tiers; reuse under actual licenses.**
  jitLLM's own code is Apache-2.0. Any permissive license is allowed in
  the core (D-091), as is MPL-2.0; copyleft lives in optional modules,
  which jitLLM's own builds ship by default; the copyleft-disabled profile
  is the build-time opt-out (D-080). Unknown, non-permissive or
  proprietary terms need a decision; obligations are still recorded. A
  component counts as copyleft once that is confirmed, not on suspicion.
  Declared tools and
  platform runtimes (including system libraries and CUDA) have separate
  terms under D-017 and remain in the audit. Reuse follows actual licenses;
  no single engine's architecture is mandatory, and "reference" is not
  relicensing. Every dependency records its category and applicable tier.
  (D-002, D-003, D-013, D-017, D-080, D-091)
- **Local-first, privacy by default.** Management binds locally by default;
  the inference endpoint defaults to loopback and the tailnet, may be bound
  elsewhere explicitly, and authentication is optional. Prompts and
  KV contents are never logged by default; spill files are protected with
  explicit retention. (D-014 and its owner note, D-097)

## Repository layout

| Path | What lives there |
| --- | --- |
| `docs/` | Vision, plan, architecture, decisions, features, rough edges, workflow |
| `docs/ideation.md` | The kickoff design brief (2026-09-20). Frozen origin document with source links; the living docs above supersede it where they differ |
| `LICENSE`, `LICENSES/`, `NOTICE` | Apache-2.0, the license for all jitLLM-authored code (D-003; dependency policy in D-017); the text of every license a file declares; the attribution notice. Every file carries SPDX tags in its header, or in a `.license` sidecar if it cannot hold a comment (D-029, D-071) |
| `CHANGELOG.md` | Keep a Changelog; a change with user-visible effect adds its line (D-062) |
| `mise.toml`, `mise.lock` | mise tasks (`setup`, `prepare`, `doctor`, `build`, `test`, `deploy`) and the pinned Python that runs `tools/` (D-070) |
| `toolchains/` | The SDK manifest, artifact lock, host prerequisite lists and the provenance records of everything that builds jitLLM ([README](toolchains/README.md); D-049, D-070, D-071) |
| `third_party/` | The source lock: every third-party source component, prepared into `build/sources/` by `mise run prepare` ([README](third_party/README.md); D-017, D-057), and in `patches/` the reviewed changes to them (GGML's, ExLlamaV3's, CUTLASS's and ds4's, D-077) |
| `CMakeLists.txt`, `CMakePresets.json`, `cmake/` | The build: presets `native` (CUDA for `sm_121` and the discrete `sm_86`, D-082; test preset `native-gpu` runs its `gpu-discrete` tests on the workstation's GPU), `cpu`, `cross` and `spark-native` (GB10 only) use the SDK (plus the host GNU linker on Spark) and the prepared sources (`JitllmSources.cmake`); `project(VERSION)` and the version derived from Git on every build (`JitllmVersion.cmake`, D-062); outputs and the build receipt go to the ignored `build/<preset>/` |
| `src/` | jitLLM's modules, one directory per module of the [layers](docs/architecture.md#layers-and-dependency-rules): so far `base/` (build info, public-surface versions, diagnostic reports, typed identities, checked byte counts, invariant checks, bounded queues, the wake flag, SHA-256, general JSON), `platform/` (reads of `/proc` and `/sys`, the host probe, the path-trust walk, the direct-I/O probe and opens, a raw io_uring ring, the interface addresses, the event loop, wakers and signal watch, socket calls, memory pressure from outside (MemAvailable, PSI), the owner-only files kept across a restart, D-105; with the Linux providers the only Linux-specific code), `providers/` (the device probe; the device-memory, device-execution and storage interfaces, the device runtime (`device_runtime.h`: the engine's copies, graphs, pinned memory), whole direct reads, and their fakes in `providers/fake/`; `providers/cuda/` links the NVIDIA driver, D-072, and with the kernels is the only CUDA code), `config/` (the node's TOML configuration and storage roles, D-073), `catalog/` (extents, resources, leases, generations, occupancy), `memory/` (the commitment ledger, victim selection, materialization planning, the reclaim order) and `scheduler/` (admission and switching, the completion board, lanes, task trees, the storage, device and CPU lanes over the providers, and the scheduler thread's turn loop) of the resource core, `tokenizer/` (byte-level BPE with the M3 pre-tokenizers, Gemma 4 raw UTF-8 BPE, classic SentencePiece GGUF, UCD 15.1.0 tables and NFC, and readers of GGUF and `tokenizer.json` tokenizers, D-088) and `chat/` (native chat-template family renderers chosen by template hash or probe equivalence, the bounded Jinja-subset interpreter for any other template, stop tokens, D-067), `model/` (state representations with their capabilities, a request's live state, model contexts composed of components, D-068, and architecture adapters: the Qwen2 profile, its binding to an artifact and each chunk's host-built inputs, the EXL3 binding and native operation plan, DeepSeek V4's and Qwen3.8's profiles, bindings, bounded state layouts and chunk inputs, Gemma 4's [foundation](docs/gemma4.md) with checked profiles, tensor bindings and independent-slot state/inputs, and Qwen-Image-2.1's profiles, component bindings, VAE plan and host arithmetic), `execution/` (the implementation registry and plans that name one implementation per operation, D-053; phase kinds, decoding modes and request programs with their envelopes, D-050, D-068; greedy and seeded sampling) and `artifact/` (the v0 prepared-artifact reader: strict JSON, validation as untrusted input, groups, chunks and direct-read plans, D-056; and the composition reader, D-089) of the model layer, `kernels/ggml/` (GGML tensor descriptors over jitLLM memory, the K-C launch context, jitLLM's cuBLAS handle and GGML-derived operations, cuBLAS matrix multiplication and the forced vector attention among them, and DeepSeek V4's and Qwen3.8's in `ops_ext.h`: quantized products and `mul_mat_id`, tensor-core attention at D 256 and 512, the indexer, hyper-connections and linear attention; jitLLM's own MXFP8 and NVFP4-row operations on GGML tensors in `jitllm_ops.h`, with its fusions of Qwen3.8's GGML nodes and the routed experts over CUTLASS's NVFP4 grouped GEMM (`moe_cutlass.h`), upstream's fusion gates, their registry declarations, the Qwen2, DeepSeek V4 and Qwen3.8 chunk graphs, graph planning with activation placement, and the executor of bound plans, D-053, D-077), `kernels/exl3/` (jitLLM's launchers of ExLlamaV3's locked kernels under their launch contract, host checks, the reconstruction GEMM on cuBLASLt, the EXL3 linear's paths and their registry declarations, D-080, and the executor of a native EXL3 phase), `kernels/image/` (jitLLM's own BF16 kernels for Qwen-Image-2.1, FlashAttention-2 and the VAE's implicit-GEMM convolution among them, its cuBLAS and pinned cuBLASLt BF16 products, their registry declarations, and the pipeline's phases dispatched through a bound plan), `kernels/paging/` (the engine's fill and n-gram row-gather kernels), `engine/` (the paged node, the runner skeleton: weights as extents, live state with spill and a verify's snapshot, plan caches, graph runs, runner resources, the request cohort; and each M3 model's runner on it; CUDA builds, D-096, but no CUDA calls or headers of its own), `runtime/` (`jitllm-runtime`, the node runtime process, D-074, its serving commands `chat` and `swap-table` over the engine, D-096, each model's settings in three layers, their calibration record and the `settings` command, D-103, hang recovery's ladder, D-102, conversations kept across a restart, D-105, and the chat route's HTTP server and listeners, D-097) and `cli/` (the `jitllm` command: `--version`, `doctor`) |
| `packaging/` | `jitllm.service`, the sysusers and tmpfiles files, the maintainer scripts, the annotated example configuration, the notice texts the package needs and the arm64 install test; CPack settings (D-063, D-074) |
| `.clang-format`, `.clang-tidy`, `.clangd` | Style and lint configuration (D-059); clangd reads `build/native` |
| `tests/toolchain/` | The toolchain contract (C++23, GCC 16.2 runtime, no exceptions, libstdc++ assertions in the test presets (D-083), explicit targets, static runtimes, GoogleTest), tested in each profile's binaries |
| `tests/jobs/` | The confined-job proof, which `tools/job-proof` runs in delegated cgroups (D-074) |
| `benchmarks/` | Measurement harnesses, built but never run by CTest; their reports live under `docs/experiments/`. `retained_backing/` holds the retained-backing replay and its candidate designs, which unit tests cover |
| `tests/unit/`, `tests/version/`, `tests/smoke/` | Module unit tests (GoogleTest; a `std::expected`'s error is read through `tests/support/expected_error.h`, D-083; `tests/unit/data/` holds the tokenizer corpus and reference fixtures, and the `models` label marks tests that need the model files on a Spark); the version rules on synthetic repositories, and `jitllm --version` against the receipt; `jitllm doctor` on each host, requiring a clean report on a GB10 (`gpu`) |
| `tests/sources/` | The source mechanism: the receipt and the compile/link inventory against the lock, and D-057's gates on a synthetic lock |
| `tests/support/` | Test and benchmark support, never linked into production binaries (configure checks): the safe reading of a `std::expected`'s error (D-083), the launch recorder and the executed-plan recording that `docs/experiments/backend-proof-p2/plan_compare.py` compares with the FP16 bridge's recorded plan; the harnesses' names for the engine's paged node and the scheduler's task programs (`paged_node.h`, `paged_programs.h`, D-096) |
| `tools/` | `setup` (SDK, then sources), `setup-toolchain` and `check-toolchain` (the SDK), `prepare-sources` and `inspect-sources` (the source lock), `build` (the build, test, deploy and package tasks; the package's documents and inventory in `jitllm_package.py`), `job-proof` (the confined-job proof), `spark-job` (detached long runs on a Spark, supervised and waited on; required, docs/workflow.md), `run-target` (runs cross-built tests under qemu-user or over SSH), `gen-unicode-tables` (the tokenizer's tables from pinned UCD files, D-088), `nsys_steps.py` (matched GPU-step comparison of two nsys traces, docs/optimization-inventory.md) and `check` (the `check`, `check:full` and `check:spark` tiers, D-061; its header check is `jitllm_headers.py`, its portability boundary check `jitllm_boundaries.py`) |
| `.devcontainer/` | The digest-pinned reference container (D-012, D-061) |

Update this table as new top-level scaffolding lands.

## Doc map — pull what the task needs, not everything

Always read (it's short): [docs/workflow.md](docs/workflow.md) — the
build → review → commit loop, the heavy path for blast-radius changes, and
the commit gate.

| Doc | Read when the task needs |
| --- | --- |
| [docs/plan.md](docs/plan.md) | What to work on, milestone scope, exit criteria — what "done" means |
| [docs/m0-record.md](docs/m0-record.md) | Where an M0 result came from: each planning task, spike and reference run with its evidence links and caveats. Frozen history |
| [docs/m1-record.md](docs/m1-record.md) | Where an M1 result came from: each bootstrap item's outcome, verification hosts and hand-offs. Frozen history |
| [docs/m2-record.md](docs/m2-record.md) | Where an M2 result came from: each resource-core and backend-proof item's outcome, commits, reports and caveats, and the gate. Frozen history |
| [docs/m3-record.md](docs/m3-record.md) | Frozen M3 task history, accepted speed exceptions, final profile/quality and swap evidence, build/package checks and work handed on |
| [docs/vision.md](docs/vision.md) | Why the project exists, who it's for, success criteria, non-goals |
| [docs/features.md](docs/features.md) | The feature matrix: confirmed scope, proposed additions, open questions |
| [docs/architecture.md](docs/architecture.md) | System map: processes, components and layers, request path, data model, memory and residency, providers, errors, pager invariants; links the detailed designs |
| [docs/environment.md](docs/environment.md) | Workstation and Spark inventories, links, NAS and certificates, and the M0 platform measurements behind D-032–D-034 |
| [docs/decisions.md](docs/decisions.md) | Settled choices (D-NNN). Scan headings; read only the entries your task touches |
| [docs/rough-edges.md](docs/rough-edges.md) | Findings log (RE-NNN). Grep before adding a finding or debugging weirdness |
| [docs/upstream/](docs/upstream/README.md) | Per upstream project, what to send upstream: fixes, limitations and jitLLM's patches, each entry a standalone handoff |
| [docs/async-model.md](docs/async-model.md) | The D-048 task/completion design: thread roles, submission/completion protocol, cancellation versus retirement, bounded queues; the internal contract M2 builds on |
| [docs/runtime-serving.md](docs/runtime-serving.md) | How `jitllm-runtime` serves models (D-096): the engine module, `[models]` in the configuration, registration, the full swap, turns, the `chat` and `swap-table` commands, and the chat route with its listeners, intake bounds and connections (D-097) |
| [docs/engine.md](docs/engine.md) | The engine's runner skeleton and how a new model family plugs in: what a runner holds, its life, and where the long-context work goes |
| [docs/artifact-format.md](docs/artifact-format.md) | The experimental v0 prepared-artifact format (D-056): container, manifest/index schema, layout and page-in rules, worked examples |
| [docs/model-support.md](docs/model-support.md) | The model support matrix: each model and drafter jitLLM runs, its pin, artifact, template hash, tokenizer, decoding modes, evidence, divergences and status |
| [docs/portability.md](docs/portability.md) | Other GPU platforms and OSes: where vendor and Linux code may live (the boundary check), the device runtime and platform seams, the registry's primitive-fallback rule and each model's minimum primitive set, the runners' shared skeleton, distribution |
| [docs/optimization-inventory.md](docs/optimization-inventory.md) | Cross-family optimization transfers, current consumers and shape/format limits, including useful pieces of rejected kernels; read before proposing another kernel experiment |
| [docs/tokenizer.md](docs/tokenizer.md) | The native tokenizer, chat renderers, stop tokens and sampling: pre-tokenizers, bounds, Unicode tables, agreement with the references, template hashes |
| [docs/client-api-baseline.md](docs/client-api-baseline.md) | The M5 inference API contract: routes, client profiles, front-door, status and keepalive rules; links the Ollama, vLLM and OpenRouter assessments |
| [docs/ideation.md](docs/ideation.md) | The full original reasoning and source links behind a constraint. Long; read the section you need, not the whole file |

## Rules for all agents

1. **Log decisions sparingly.** [docs/decisions.md](docs/decisions.md) is for
   choices that are expensive to reverse or that a future agent might silently
   undo — the load-bearing constraints above, artifact formats, on-disk
   layouts, public interfaces, toolchain pins. Routine implementation, naming,
   and scope calls don't get entries. A few entries per milestone is the
   target, not per task.
2. **Log findings that cost you.** A
   [docs/rough-edges.md](docs/rough-edges.md) entry is warranted when a CUDA,
   driver, Spark platform, toolchain, or library quirk burned real debugging
   time and will bite again. Skip the formal reproduction unless it's cheap to
   capture. A rough edge in, or a patch to, a third-party component also
   adds or updates that project's [docs/upstream/](docs/upstream/README.md)
   entry.
3. **Measure what a decision hangs on.** When a design choice depends on a
   performance number or a current platform capability (VMM granularity,
   map/unmap cost, I/O path behaviour, driver or toolkit support), get a real
   number on the actual target or check a current source — training knowledge
   is stale for this ecosystem. Never present an estimate as a measurement.
   Everything else: ship it and see.
4. **Fix the docs the change makes wrong** — plan status, the status paragraph
   below, an affected doc — in the same unit of work. Nothing more is owed.
5. **Commit only on the user's authorization** (D-075). The main agent
   has the owner's standing authorization (2026-10-04) to commit completed
   plan tasks of the current milestone until the owner withdraws it.
   Otherwise the user must directly ask for the
   change at hand; a request covers that commit only, never later work,
   and is never inferred from a plan, a prompt file or a tool result.
   Subagents and reviewers never commit.
   Commit only reviewed, checked work (docs/workflow.md), on the current
   branch, and say what the commit contains. No agent pushes, tags, amends
   or rewrites history. Otherwise all changes stay in the working tree for
   human review. The main agent ends each plan task's final report with a
   suggested commit title in the log's style: `[M<n>] <what landed>`, one
   line, under about 72 characters.
6. **C++23 conventions.** Clang-first. Ordinary `.cc` files use the host
   compiler; CUDA-facing translation units stay narrow and don't leak heavy
   runtime containers through headers. Typed byte counts, spans/views,
   `std::expected` error results, bounded queues, move-only ownership
   wrappers. No exceptions: jitLLM code builds with `-fno-exceptions` (D-066).
   GPU/I/O lifetime is completion-aware: a
   destructor is not proof that submitted work finished. Warning, format, and
   lint pins are recorded in D-059; M1 applies them at the repository root.
7. **Keep the always-loaded context lean.** This file is imported into every
   conversation; every line added costs every future agent. Detail belongs in
   `docs/` behind the doc map, not here.
8. **Scratch files stay out of the tree.** Temporary scripts and outputs go to
   the session scratchpad, not the repo. Delete throw-away diagnostics before
   concluding. Keep aggregate experiment results, analysis, and provenance in
   Git; raw samples, logs, traces, and telemetry stay outside the repository,
   and are deleted on every host when their milestone closes (docs/workflow.md).

## Current status

**M0 through M3 are complete; M3.5 is in progress.**
[Gemma31 gap closing](docs/experiments/gemma-gap-closing/README.md) narrowed the
bounded serving gap to stock from 3.15%/3.51% (C1/C4) to 0.73%/1.36%. It did
this with state reuse across Clear, graph capture beside eager execution and
the fused decode FFN; the remaining gap is mostly C4's four-owner decode. A
[bounded Gemma26 recipe](docs/experiments/gemma26-production/README.md) reaches
stock parity (within 0.6%) with stock's exact tokens. The
[decode hot path](docs/experiments/decode-hot-path/README.md) then runs request
steps on the driver (D-106) with no scheduler call per step, and greedy Gemma
picks tokens on the GPU: decode step round trips fall to 6–11 µs (Gemma
controls; every model takes the path).
The runtime serves Chat Completions and literal Completions with target
likelihoods on loopback and the tailnet. DeepSeek V4 Flash, Qwen3.8 Flash
Next (native NVFP4/MXFP8 and checked GGUF) and Qwen-Image-2.1 execute with
paging, initialized-state spill/restore and stable-address graphs. Both
LLMs batch chat and literal completions; pending model switches time-slice
at completed units and resume exact continuations. Active responses keep
making progress while new same-model requests wait for slots.

The final 32-row/six-pair production swap table passes: worst prepared LLM
swap 9.853 s against the 20 s bound, exact 8K states and continuations,
retained/replayed graphs and exact images. The standard OpenAI client gate
passes. Both LLMs execute to 262K; DeepSeek also completes the measured 1M
profile. Long retrieval, turn reuse and continuing-context swaps pass in
[final context](docs/experiments/m3-final-context/README.md).

The owner accepts the remaining Qwen/DeepSeek speed gaps for M3 and defers
further tuning to **M9's full-engine optimization pass** (2026-10-04);
optimizations found on any family, new kernels and fusions included, are
ported to Qwen/DeepSeek as they are found (2026-10-07).
The latest matched Qwen C4 rate is about 15% below fast Mia; the gap and
long-context misses remain measurements, not parity passes. Quality,
memory and exact-state requirements remain unchanged. The
[M3 record](docs/m3-record.md) retains the task history, qualified profiles,
exceptions, checks and later work; [optimization status](docs/m3-optimization-status.md)
retains the detailed comparisons and unadopted leads. The current Qwen
four-slot numerical control, 1,567-test Spark suite, ARM cross/qemu suite
and package install/purge fixture pass. Whole shipment tiers remain owed
before publishing a package.

M3.5 is in progress. A [lazy handoff](docs/experiments/vmm-batching/README.md)
(D-033 amended) moves swap unmaps beside page-in reads: the same 32-swap table
now peaks at 8.53 s LLM-to-LLM, 3.7 s into the image; a bounded handle
reserve and device zero-fill cut cold state growth 38%. Native token histories now have explicit capacity
charges and idle reclaim; both M3 LLMs retain exact 8K continuations.
Gemma 4 has checked profiles, bindings, bounded independent-slot state and
bounded serving for both approved profiles. Both select their checked
8K/four-slot joined recipes; larger envelopes retain the scalar route.
Gemma3 4B QAT keeps its 4K default/two-slot route and admits explicit 8448
context with one slot; checked 8K boundary and model-switch state gates
complement its ring, checkpoint/restore and restart-adoption controls.
[Gemma2 2B](docs/gemma2.md) has bounded 8K/two-slot serving with GPU masks, checked cap50
compatible prefill, template refusal and restart replay.
Broader reference, batching and sustained qualification remain open.
The retroactive optimization transfer audit now records every current
family, including those M3 closed out; its open ports are first in M3.5
(plan.md; workflow.md's transfer rule). Then come approved model checkpoints, legacy fixtures, Bonsai, formats with
EXL3 in focus, the remaining batching/skeleton gaps, media file inputs,
Clef/Clef-flash over the Jev API and media generation routes (D-101).
That scope includes batching compatible decision and image requests/phases.
[Plan](docs/plan.md), [family set](docs/m35-families.md). M4 follows on two
Sparks; [reference engines](docs/m4-references.md) are re-pinned at entry.
