<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Vision

## What this is

jitLLM is an independent, open-source inference runtime built around one
premise: **model implementations describe computation and dependencies;
jitLLM owns storage, residency, scheduling, and execution lifetime.**

The workload jitLLM is built for first is one user, or one user's agent and
its subagents, switching among a library of models that is larger than
memory (D-019). A conversation spans minutes to hours; the main model is
expected to resume after a subagent on a different model finishes; models are
time-sliced under contention and run concurrently when their complete
execution budgets fit. Switching can require expensive weight reloads and
prompt recomputation. jitLLM keeps a node-wide catalog of every managed
extent, reclaims the least valuable eligible extents across all models when
capacity is needed, preserves conversation state through residency or spill
within explicit retention bounds (D-024), and brings missing weights or state back only
when the model actually needs them, or when justified prefetching can hide a
miss. Routed-expert models get exactly the experts the router selected,
loaded on demand from prepared on-disk artifacts, with execution suspended
and other work running while the I/O is in flight. With more than one node,
a single conductor places models across the pool and routes requests, so a
subagent's model can run on another node while the main model stays
resident, a busy small model can run as replicas, and the flagship model may
be sharded across nodes. Topology is configured or discovered, never baked
in. When an idle cached prefix has expired or cannot be retained, the runtime
recomputes from client-supplied history and reports that work; admitted
requests keep their state protection while suspended.

It began as the design brief in [ideation.md](ideation.md) (2026-09-20), which
consolidated the ideation discussion. The decided directions there are
recorded in [decisions.md](decisions.md); its proposed designs were triaged
with the owner in [features.md](features.md) on 2026-09-21.

Correctness is a prerequisite for every supported configuration and every
optimization. Within that constraint, priorities are fast and predictable
model switching with conversation state preserved, warm generation
performance at parity with an all-resident run, memory efficiency so the warm
library is as large as possible, and explainable scheduling. Cold
time-to-first-token for a never-loaded model is desirable but not primary;
repeated paging stalls during generation matter.

## Who it's for

jitLLM is built by one developer but meant to be consumed externally (D-016),
so the audience is anyone with the problem, not just the owner.

1. **People running local inference on one or two DGX Sparks**, the project
   owner first among them, who switch between models or run agents whose
   subagents use different models, and who want a large library available
   without losing conversation state, warm-generation performance, or the
   ability to explain what the runtime did.
2. **Owners of other hardware with a similar memory-versus-storage gap**,
   other CUDA hardware first (discrete NVIDIA GPUs are a secondary target
   for fast model swaps, D-082), then later in the project's life Apple
   silicon, Intel Arc GPUs and AMD, in that order (D-110),
   once a validated path makes their configuration a supported one.
3. **Inference-systems developers** interested in a memory-first runtime
   whose catalog, reservation/lease, and paging subsystems are usable and
   inspectable independently of any one engine's kernels.

## Success criteria

Each of these is checkable, and each is scoped to a *supported configuration*
in the [model support matrix](model-support.md), not to arbitrary
checkpoints.

- **Switching is fast and state survives it.** With a library larger than
  memory, switching from a resident model A to model B and back reloads only
  the missing dependencies; a compatible retained prefix resumes from KV or
  equivalent state through residency or spill and restore. M6 demonstrates
  both paths through an unmodified client, after M3 and M4 make full swaps
  of large models fast. Expiry, capacity limits, or edited
  history can require recomputation; these cases are measured separately and
  must remain correct (D-024). Switch and switch-back latency are measured
  against the reference's end-to-end whole-model-switching baseline
  for the agreed workloads, and are never worse than a full swap of one
  engine instance for another (D-021, D-025). Report bytes moved and prompt
  tokens reused versus recomputed. Models run concurrently when the
  working sets and complete execution envelopes fit each node's budget under
  a supported placement; otherwise they are time-sliced, and the runtime
  reports which happened.
- **Standard clients work unmodified.** Cursor, OpenCode, Codex, Claude
  Code, and other standard web-API clients talk to the conductor's
  endpoint with no jitLLM-specific changes; the request's model field
  drives switching, and compatible cached state is recovered by prefix
  identity. Prefix matching identifies neither a conversation nor its lifetime.
  A system-prompt prefix is reusable across compatible conversations, with
  independent continuation state and different reuse/expiry policies under
  common bounds; neither is retained indefinitely (D-031).
  Sessions and hints are optional extensions (D-022, D-024, D-030).
- **Partial retention works.** With two persistent model contexts on one
  Spark, when the second needs capacity, only selected extents of the first
  are displaced; untouched extents remain resident; resuming the first
  reloads only the missing dependencies. Verified from catalog state and
  structured events, not inferred from timing. For a dense model this pays
  off at resume time, not during active decode, and the docs say so.
- **Demand-paged MoE is exact.** On a supported routed-expert checkpoint, a
  phase loads only the experts the router selected plus declared
  metadata/granularity/read-ahead, never substitutes an expert, never drops a
  selected contribution, and the overhead of the routing/residency boundary
  on an all-resident path is measured and reported.
- **Numerics match a pinned reference.** Teacher-forced logits and declared
  intermediate outputs match a pinned known-working engine within a
  documented tolerance for each supported checkpoint. Generated text alone is
  not evidence.
- **Two Sparks stay correct under stress.** A sharded model on two nodes
  passes asymmetric-memory-pressure, cancellation, and controlled-failure
  tests with no speculative reuse after timeouts.
- **Placement is useful without sharding.** M6a routes a main model and its
  subagent across configured nodes through one conductor, with local budget
  enforcement, health checks, and state affinity. It follows M6 independently
  of demand-paged MoE; sharded execution starts with M4's full swaps and has
  its own M8 gate under pressure and failure.
- **Every decision is explainable.** For any eviction or admission decision
  the runtime can report victims, expected and actual bytes recovered, the
  cost estimate, and why alternatives were retained, from structured events
  keyed by opaque request IDs.
- **Performance claims carry provenance.** Warm decode rate, inter-token
  latency distribution, bytes per token, and exposed stall time are reported
  for each supported configuration with artifact, backend, toolchain, driver,
  hardware, and policy identities, and compared against the pinned reference
  deployment before a backend or policy is promoted. Report both matched
  decoding configurations and the reference's normal optimized configuration,
  following [the comparison protocol](architecture.md#performance-evidence).
  The early paging-feasibility spike establishes workload benefit and
  generation-stall criteria; measured implementation results must meet them.
  Native code is never assumed to outperform an existing engine.
- **Setup is reproducible by a stranger.** A new user reaches a running
  supported model on a fresh Spark by installing from the project's package
  repository and following only the checked-in docs (D-027); a developer
  reaches a working build through the checked-in setup path, which reports
  the exact toolchain, driver, and SDK identities it installed or found.
- **The default build meets its dependency policy.** Incorporated core
  implementation is under permissive licenses or MPL-2.0 (D-017, D-091);
  declared tools and platform runtimes are recorded under their separate
  terms (D-017). It and the copyleft-disabled profile build, pass the core
  tests, and have their full dependency closures audited. jitLLM's own
  builds include optional modules and their dependencies by default, with
  the copyleft-disabled profile as the build-time opt-out (D-080); builds
  that include them ship matching notices and source obligations.

## Non-goals

Stated by the owner for the first implementation, each with its reason.

- **Universal model support.** Support is earned per checkpoint and
  configuration; a family name is not a support claim.
- **A new tensor compiler, or a universal compiler framework in the first
  milestone.** Start with a small set of explicit operations and validated
  backend adapters.
- **A new quantization scheme.** Preserve the reference representation;
  requantization is an explicit, evaluated transformation.
- **Arbitrary GPU page-fault interception.** Explicit VMM: accessing absent
  backing is a bug, not a request.
- **Transparent cross-node shared virtual memory.** Two Sparks are two
  domains; transfers are explicit objects.
- **Production-grade multi-tenant isolation.** Single-owner local nodes.
- **Training.** Inference only.
- **A vLLM fork or plugin as the product architecture.** Reference and
  integration options only (D-005).
- **Rewriting proven kernels to claim native ownership**, or importing an
  engine's allocator/scheduler because we reuse its kernel. Reuse the unit
  with its assumptions; own the memory system.
- **Inventing dense-model sparsity by rearranging weights.** A layout change
  does not let a dense kernel skip bytes.
- **A Python interpreter in the serving, paging, or scheduling hot path.**
  Build-time tooling may use it (D-010).
