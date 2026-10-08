<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M0 record — Plan the plan

*Milestones renumbered (D-087, 2026-09-27): numbers here are those of
their date; the 2026-09-23 ladder's M3–M8 are now M5–M10, and M4a is M6a.*

M0 ran from 2026-09-20 to 2026-09-23 and exited on the owner's approval of the
plan on 2026-09-23. This is its task-by-task record, moved out of
[plan.md](plan.md) at exit so the plan stays lean: what each planning task,
spike and reference experiment did, its evidence links, and the caveats
recorded at the time.

**This is frozen history.** The living documents supersede it where they
differ: [plan.md](plan.md)'s milestone ladder owns everything M0 left open,
[decisions.md](decisions.md) the settled choices, and
[features.md](features.md) and [architecture.md](architecture.md) the current
scope and design. Numbers here are summaries; the linked experiment reports are
the evidence. Later corrections go in those documents, not here.

## Goal

Goal: turn the design brief ([ideation.md](ideation.md)) into a settled
vision, feature matrix, architecture, and milestone ladder — through planning
conversations with the project owner plus targeted spikes where a decision
needs evidence from the real hardware.

## Tasks

- [x] Repo scaffolding for the AI-directed workflow (this scaffold,
      2026-09-20).
- [x] Feature triage: walk [features.md](features.md) with the owner; confirm,
      reject, or defer proposals (2026-09-21: 46 proposed or open rows
      triaged; 30 confirmed, 13 deferred with a reason, trigger, and
      earliest milestone, 2 rejected, 1 still open. Load-bearing calls
      recorded as D-028 GGML-first substrate with build-time optional
      backends, D-029 contribution and compliance conventions, D-030
      Claude Code and the Anthropic Messages format). A deferral does not
      become approval when its trigger fires.
- [x] Confirm the original-code license (2026-09-20: Apache-2.0 accepted,
      D-003; dependency categories and tiers clarified in D-017, superseding
      D-015; process weight for an externally consumed project recorded,
      D-016).
- [x] Decide NOTICE and SPDX-header conventions and the contribution policy
      (2026-09-21: REUSE copyright/license metadata plus an embedded-header
      check in CI and a NOTICE file from M1, SBOM with packaging, external
      PRs accepted under DCO; D-029).
- [x] Inventory the environments without changing drivers or security
      settings (2026-09-20, read-only, no sudo): workstation baseline and both
      Sparks recorded in
      [environment.md](environment.md#target-nodes-dgx-sparks). The
      interconnect half is the separate cabling task below.
- [x] Spike — **toolchain smoke** (2026-09-21; open question 6,
      D-032): Clang C++23 native and AArch64 cross builds passed;
      NVCC 13.4.92 (Toolkit 13.4.2) with Clang 22.1.8 produced
      C++23 `sm_121` CUDA objects that ran on `spark`, as did the native
      Spark fallback. C++23 host/device
      feature checks and GPU results passed on driver 580.178.04 with
      PTX JIT disabled; the older 13.0 dialect limit is RE-001.
      Exact compiler, library, CUDA component, and target-sysroot pins,
      hashes, commands, and limits are in the
      [smoke report](experiments/toolchain-smoke/README.md).
      Matching compiler-rt packages and Clang ASan/UBSan CPU checks on the
      workstation and Spark were added 2026-09-22; M1 provisioning must
      include those runtimes. M1 still owns declarative provisioning and
      CMake presets.
- [x] Spike — **VMM microbench** (2026-09-21; open question 1, D-033):
      three runs on `spark` measured 2 MiB minimum/recommended granularity,
      allocation/map/access/unmap/release costs across 2–128 MiB extents,
      and costs with independent background kernels. A retained, unmapped
      1 GiB pool kept its physical footprint and contents; releasing its
      handles returned capacity. Initial policy: 2 MiB independent extents,
      completion-safe backing handoff, no standing unused-handle cache.
      [Aggregate report and harness](experiments/vmm-microbench/README.md);
      numbers in environment.md. I/O and model-load optimization remain
      separate measurements, not conclusions of this allocation experiment.
- [x] Spike — **I/O path comparison** (2026-09-21; open question 2, D-034):
      compared buffered/direct files, pinned staging, cuFile compatibility,
      native asynchronous I/O, and GPU in-place access on `spark`.
      Direct regular files into host VMM reached about 15 GB/s, sustained
      14.962 GB/s for 180 seconds, and matched device-VMM GPU scan speed.
      Concurrent compute, 100 GiB held-memory pressure, cache reclamation,
      sparse/small reads, verified write bursts, and DMA-bounce tracing are
      in the [aggregate report and harness](experiments/io-path/README.md).
      Initial budget: two to four 2 MiB destination slots, no extra Spark
      staging copy. Raw-device alternatives need a dedicated unmounted SSD;
      no raw performance claim is made. Controller interrupt coalescing was
      tested separately with its original setting restored. Actual GGML
      behavior and full registration/reclaim lifetimes remain M2 proof work;
      mixed read/write and sustained-write/endurance policy remain M4 inputs.
- [x] Pick the reference engine for the feasibility spike: llama.cpp with
      MoE GGUFs and a small router-logging patch (decided 2026-09-20; the
      lightest install and the owner's preference).
- [x] Install llama.cpp on a Spark in a container so the host baseline in
      environment.md stays clean (2026-09-21): digest-pinned ARM64 CUDA
      image, source `b29c606e2`, and hash-verified Gemma 4 UD-Q4_K_M execute
      on `spark`. GPU inference and cross-process slot restore passed;
      Gemma requires `--swa-full` for the tested reuse path (RE-004), with
      its additional memory cost recorded. Exact identities, expert/state
      accounting, reproducible harness, and limits are in the
      [reference setup report](experiments/reference-setup/README.md).
      No host toolchain/driver/security changes or HF key were needed.
      Ornith (MIT, confirmed by owner) is now validated in the switching
      experiment below; Qwen remains an unexecuted candidate. Installation
      alone did not complete switching or route-trace experiments.
      Candidate trace models, owner-provided, with facts from their model
      cards as read on 2026-09-20 (repository access/revisions/licenses
      rechecked 2026-09-21; Gemma's structure and memory accounting were
      verified in setup, Ornith's in the subsequent A→B→A experiment):
      - [unsloth Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):
        125B core parameters, 6B active, 512 experts, top-10 plus 1 shared, expert
        intermediate dim 640, plus a 51B n-gram embedding table and a 4B MTP
        head; Gated DeltaNet plus sparse MQA attention; 262K context. GGUFs
        run from about 72 GB
        (1-bit) through 82 to 90 GB (3-bit) to 111 GB (Q4_K_XL). On a
        121 GiB node that is a single-model budget sweep: 3-bit fits, Q4 is
        borderline, Q5 and up exceed the node, so it covers the
        forces-paging axis by itself. Its n-gram table is the first concrete
        sparse-lookup component: row requests resolve to containing extents
        (D-035), and its MTP head matters
        for matched-configuration comparisons (D-021).
      - [unsloth gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF):
        25.2B total, 3.8B active, 128 experts, top-8 plus 1 shared; hybrid
        1024-token sliding-window and global attention; 256K context.
        Q4_K_M about 17 GB. The owner has experience with Gemma.
        Fits easily: the subagent and switch-latency case, with mixed KV
        lifetimes for the spill study.
      - [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF):
        36B total, about 3B active on the card, `qwen35moe` architecture;
        the selected GGUF verifies 256 experts, top-8 plus a shared FFN,
        40 primary layers and one stored MTP layer; 262K context.
        Q4_K_M about 22 GB. A second small model for replica and
        multi-model-switching traces.
      Gaps these three leave, to fill only if cheap: a few-large-experts,
      low-top-k model (Mixtral-style) for the high per-miss-cost end, and a
      plain full-KV GQA model where KV grows linearly and spill cost is
      highest. Record for each model used: expert count, top-k, shared
      experts, expert bytes at the chosen quant, total bytes, KV or recurrent
      state bytes per token, and headroom on one node. The owner's local
      Ollama blobs are plain GGUFs usable for workstation-side dry runs on
      the RTX 3080 Ti where they fit.
- [x] **Image-generation reference experiment — Qwen-Image-2.1**
      (bounded reference study complete, 2026-09-22): the
      [report, pins and harness](experiments/image-reference/README.md)
      establish a standalone BF16 Diffusers reference on `spark`, with
      checkpoint/dependency provenance and research-license boundaries.
      Twelve image cases cover 512–2048 pixel resolutions, 4/40 steps,
      editing, repeated outputs, prefix reuse, completed-step cancellation
      and phase-boundary backing release. The full pipeline has 32.44 GB of
      parameter storage in execution, not just the advertised 7B generation
      component; the 2048²/40-step request takes 252.796 s and peaks at
      56.521 GiB of CUDA allocations. Exact pixels survive cancellation
      recovery and discarding encoder/denoiser backing after their completed
      phases; all weights were initially loaded, so lower-budget admission
      and subsequent image-state restoration remain unproven.
      Short text → image → text cycles return in 0.044 s with text resident
      and 3.103 s after state restoration, processing only 12 new tokens.
      Full recomputation takes 4.155 s in a separate diagnostic and matches
      its fresh-context control, **but differs from resident text**; the
      original failed comparison is preserved (RE-008). These are single
      observations, not latency distributions or measured llmpalooza speedups.
      Image loading is slow and variable; outward harness timings include
      checksum verification. No cold-cache, forced-pressure or
      larger-than-memory image result is claimed. Native GGML image execution,
      pending-work cancellation, output API scope and delivery milestones
      remain separate follow-up work; existing image-input API scope implies
      no native image-generation capability.
- [x] **Additional reference candidates** (owner-added; bounded references
      complete 2026-09-22; [candidate matrix](experiments/model-candidates.md)).
      **Qwen-Image-2.1 GGUF** ([report](experiments/image-gguf/README.md)):
      pinned stable-diffusion.cpp (GGML) runs the full pipeline from Q4_K_M,
      Q8_0 and a BF16 control on `spark`; repackaged components were
      reconciled tensor by tensor with the BF16 baseline. Repeats, cold/warm,
      a second node, phase-released parameters and budgeted disk-backed
      denoising (against a same-VAE-tiling control) give exact pixels;
      quantization shrinks denoiser parameters
      (4.29 vs 13.25 GiB) but not step time, and this runner is 2–3× slower per
      step than diffusers BF16. Its 2048² VAE decode buffer is 38.3 GiB;
      upstream GGML matches the fork through 1024² but aborts there on a
      32-bit stride assert (RE-011). Releasing encoder/VAE parameters outside
      their phases cut sampled host memory 30.4→17.2 GiB at 1024²/40; a 3 GiB
      budget ran the request in about 5 GiB at 2.6× the time. Pixel agreement
      is not a quality score. **MiMo-V2.6-Flash-RL**
      ([report](experiments/mimo-reference/README.md)): TP=2/EP=2 on both
      Sparks with packed MXFP4 experts, 81.9–83.8 GiB weights per rank, both
      RoCE HCAs, correct greedy smoke, about 2,360 prefill tokens/s, prefix-hit
      TTFT 8.68→0.36 s at 20K tokens, and 18.5 decode tokens/s (54 ms median,
      62 ms p99 gaps) without speculation or graphs. Startup needed audited
      config remote code and `torchcodec` (RE-012). One boot and single
      requests; no switching, state restore, pressure, concurrency or
      multimodal cases. Neither establishes native support, replaces the
      canonical DeepSeek/Qwen switching pair, or blocks M1; smaller MiMo
      quants remain a revisit trigger.
- [x] **Reference A→B→A experiment** (2026-09-21, D-025): 27 verified cycles
      on `spark`, Gemma 4 UD-Q4_K_M → Ornith 1.5 Q4_K_M → Gemma, with an
      18,339-token continuation. Three repeats of nine cases distinguish
      resident, warm/cold file cache, restore/recompute, 80 GiB locked
      physical pressure, and default-SWA/cache optimizations. Under the
      matched forced-displacement budget, median first-token waits are
      21.232 s outward and 18.304 s back with retained state; recomputing
      A's prompt takes 25.236 s on return. Warm restore returns in 4.062 s;
      live residency in 0.089 s. Native LRU, durable save/restore, actual
      block I/O, state bytes, memory and bounded observed swap are verified.
      A reuses 18,297 tokens and processes 42; all outputs match, and a
      separate early/late notebook recall probe passes. B's recurrent-state
      continuation also passes. Default SWA still re-prefills after restore
      (RE-004); decode speeds differ by state lifecycle and are reported.
      Exact pins, external trace identity, measured ranges, tooling and
      limitations are in the [aggregate report](experiments/reference-aba/README.md).
      Normal optimization probes use the same forced one-model policy;
      normal concurrent placement remains unvalidated. The later full feasibility
      study includes a separate large-model observation and finds that Gemma
      sequence snapshots can lack SWA coverage after prompt rollback (RE-007).
      The short matching restored output above is historical evidence, not a
      general continuation-correctness proof; the validated recompute arm is
      the usable correctness floor. M4 can replay this exact input trace.
      Comparator datapoint: Athena's Engine reports a 46 s measured switch
      between DeepSeek V4 Flash and Qwen3.8 Flash Next on one GB10 and a
      2.1 s restore of a 141k-token conversation from disk (creator-reported;
      see [architecture.md](architecture.md#comparator-athenas-engine-closed-source-creator-reported)).
      Optionally install it on a Spark as a second comparator for the
      normal-reference view if its terms still allow personal use; never cite
      its numbers in place of our own measurement.
      The author's X thread adds that the 46 s includes checkpointing the
      session and that the pair does not both fit in 128 GB at those bit
      depths, which makes DeepSeek V4 Flash plus Qwen3.8 Flash Next the
      canonical two-large-model switching workload for the feasibility spike
      if a DeepSeek V4 Flash GGUF is available at pick time.
- [x] Spike — **paging feasibility** (bounded study complete, 2026-09-22):
      the [full study](experiments/paging-feasibility/full-study.md) extends the
      [first cut](experiments/paging-feasibility/README.md) with varied longer
      Gemma/Ornith conversations, the exact reference A→B→A trace, concurrent
      four-sequence decode, and the larger-than-memory DeepSeek/Qwen library.
      Arrival schedules model minutes-to-hours sessions; they are not hardware
      soaks and do not test expiry. Matched budget sweeps compare demand paging,
      eager active-model loading with partial inactive retention, and whole-model
      switching, including non-expert extents, live state, workspace, headroom,
      bounded spill, restore/recompute, and isolated/packed 2 MiB layouts.
      Per-layer reuse, held-out prediction, switch-back dependencies, measured
      reference waits, and explicit storage/overlap sensitivities are recorded.
      Results are offline byte/service estimates, **not measured llmpalooza speedups**.
      Qwen comparisons remain conditional on captured trajectories because
      exact prediction equivalence failed; no numerical tolerance is invented.
      Gemma rollback can require state recomputation despite byte-identical
      snapshots (RE-007), and unvalidated large-model spill continuations use
      conservative recompute scenarios. The evidence supports M4 retention
      before M5 expert paging and makes restore coverage an explicit validation
      requirement; it neither proves runtime admission nor enables prefetch.
      Performance acceptance targets are recorded in D-036 below.
- [x] Agree switching-benefit and generation-stall criteria (2026-09-22,
      D-036): M4 onward must meet the fastest correct full-swap reference
      arm at median/p95 in both directions; M5 onward permits at most 10%
      added generation time, continuation time to first token included, and
      20 ms p95 / 100 ms p99 added token gaps against a resident control
      with matched state provenance; M7 requires at least 25% lower median
      return-switch latency than llmpalooza's own whole-model control on an
      agreed partial-retention workload, with at least one named library
      that exceeds physical memory. Targets apply to named supported
      configurations; correctness is mandatory and inconclusive comparisons
      do not pass. Pin workloads, trial counts, and measurement methods
      before acceptance runs; these are targets, not measured implementation
      results.
- [x] Complete the Spark-to-Spark direct-link baseline. The owner configured
      the `sparky` DAC cluster on 2026-09-21; the
      [baseline](experiments/interconnect/README.md) passed 78 host-buffer
      test pairs and 27 pinned NCCL runs. The two PCIe interfaces share one
      physical 200 Gb/s port: combined writes measured 184.76 Gb/s in either
      direction, reads 150.10 Gb/s with default queue settings. Large NCCL
      SendRecv/AllReduce reached 22.35/22.20 GB/s; supported result checks
      passed. Channel logs and counters verify actual HCA use; source and
      allocation logs establish GPU access to mapped host communication
      buffers, not GPUDirect RDMA. Message-size sweeps, ranges, pins and
      limitations are recorded with the report and in environment.md.
      Sharded-model, asymmetric-pressure and failure tests remain M6 work.
- [x] Inventory MiaAI-Lab reference licensing versus MIT ExLlamaV3 upstream
      (2026-09-22): [pinned file/group inventory](licensing.md) separates
      AGPL defaults, explicit MIT/Apache notices, historical MIT grants and
      unresolved downstream modifications. Nine vendored ExLlamaV3 headers
      matched pinned upstream bytes and hashes. AGPL network-source duties
      and adoption blockers are recorded; no code or module boundary is
      approved by this inventory.
- [x] Decide conductor location and admission/placement ownership (2026-09-22,
      D-037): conductor inside the designated node runtime, advisory per-node
      cluster view and placements, authoritative local admission, bounded
      streaming through the front door, and no replay of uncertain attempts.
      [Architecture](architecture.md#conductor-ownership-and-admission) records
      lifecycle/failure requirements and future validation cases; D-038 below
      supplies the configuration and wire/fencing design.
- [x] Define the initial cluster (2026-09-22, D-038/D-039):
      [design](cluster-design.md) settles experimental TOML v2, one enrolled
      conductor/membership, mutual authentication, session fencing and restart
      reconciliation, bounded queues/records/timeouts, node health/admission
      and independent retained-state affinity. Owner-requested initial
      interface/QSFP detection proposes the layout; read-only probes on both
      Sparks verified adapter/physical-port grouping for the two host paths
      sharing one cable. D-039 adds single/pair/triangle/switched-N layout
      classification and bounded dedicated-QSFP subnet scans. Setup discovery
      and enrolled-member path refresh
      are M4a scope; automatic membership changes, election and replicas
      remain deferred. Implementation and adversarial execution are still owed.
- [x] Verify the named-client API requirements against current official docs
      (2026-09-22, D-040): the [baseline](client-api-baseline.md) specifies
      Chat Completions, Responses (required by Codex), Messages and token
      counting, model listing, JSON/SSE and tool round trips. Cursor's exact
      custom-endpoint behavior is not fully documented and remains an explicit
      M3 validation gap, as does executed compatibility for every client.
      Local-only defaults remain intact; no serving implementation is claimed.
- [x] Triage the owner-requested [API capability assessment](api-capabilities.md)
      (2026-09-22, D-041/D-042): confirmed Ollama subset, discovery (M3/M4a),
      continuation close (M4), download/warm jobs, staged text-resource/image/
      audio-file inputs, MCP management, single-owner sharing controls and
      later embeddings. Ollama registry/management compatibility, live
      audio/video and batch/background jobs have explicit deferral triggers.
      Exact schemas and unassigned delivery milestones remain planning work;
      no implementation support is claimed.
- [x] Triage the follow-up [vLLM API comparison](vllm-api-assessment.md)
      (2026-09-22, D-043/D-044): direct compatibility tested with unmodified
      clients; tokenization/rendering, JSON/schema/strict tools, reasoning,
      reranking, metrics/health/load and raw Completions/token diagnostics
      confirmed. Regex/grammar, LoRA and classification/reward/pooling remain
      workload-driven deferrals; generic worker RPC, training and split-serving
      deployment controls are excluded from the client baseline. Delivery
      milestones and precise compatibility profiles remain planning work.
- [x] Review the API contract documents against live client docs and fix the
      gaps (2026-09-22, D-045): Claude Code's Anthropic-shape model discovery,
      default-on alias fields, error-wording recovery, request-class and
      context-compacted signals and the attribution block; Codex timeouts,
      retries and returned reasoning items; front-door listener, auth and
      CORS defaults; admission status codes and the keepalive rule;
      `keep_alive`, alias echo and extension carriage. Execution evidence is
      still owed in M3.
- [x] Triage the [OpenRouter assessment](openrouter-api-assessment.md)
      (2026-09-22, D-046): model-metadata fields ride with M3 discovery and
      the per-model endpoints shape with M4a availability; the `reasoning`
      object, `reasoning_details` and cached-token usage join D-043's
      Chat Completions contract; `session_id`/`user`/`metadata` are hints;
      the `models` array is reserved as the spelling should fallback ever be
      accepted, which it is not; plugins, transforms, auto-router, pricing,
      credits and generation stats are excluded. No fourth protocol;
      execution evidence still owed. D-047 corrects reasoning wire fields and
      signed-block format, rejects unsupported Responses storage, and scopes
      SSE keepalives separately from non-streaming JSON/deadline handling;
      the baseline carries their acceptance cases.
- [x] Decide the async/task and completion model (2026-09-22, D-048):
      [explicit native task states](async-model.md), one node-local
      scheduler/catalog writer, bounded provider services and operation-owned
      completion/cleanup storage. Cancellation never substitutes for resource
      retirement. The [CPU-only prototype](experiments/async-model/README.md)
      passed on the workstation and `spark`, including 216 event schedules,
      saturation, partial submission and uncertain completion. Real threading,
      provider and GGML/VMM lifetime validation remain M2; reservation progress
      is the separate question 9 decision below.
- [x] Decide the initial reservation guarantee and progress envelopes
      (2026-09-22, D-050): [policy and adversarial cases](reservation-policy.md)
      settle guaranteed bounded requests, maximum retained-state/growth
      allowances, complete phase envelopes held through waits and unwind,
      request/response-granularity switching (never mid-request; owner
      2026-09-22; D-069 later made switching a configurable policy that
      pauses only at completed phase boundaries) and full-envelope checks for supported concurrency. Grants remain lazy;
      opportunistic work cannot invalidate them, and spilled admitted state
      keeps its in-memory allowance. Impossible phases fail or use an already
      validated alternative. Numeric envelopes and executed progress/lifetime
      proof remain M2, with retention/concurrency in M4 and routed phases in M5.
- [x] Decide the first vertical-slice checkpoint and numerical reference
      (2026-09-22, D-051): [Qwen2.5-0.5B-Instruct official FP16 GGUF](first-slice.md),
      exact artifact/tokenizer/template identity and pinned llama.cpp CUDA
      reference, with CPU diagnostics. The [bounded reference check](experiments/first-slice/README.md)
      passes 76-token repeat/context-restore comparisons on Spark; native
      support, numeric acceptance thresholds and context-size validation
      remain M2/M3. Source-unit licenses/provenance are recorded; generated
      Unicode data needs explicit clearance before native tokenizer adoption.
- [x] Require a real EXL3 companion early (2026-09-22, D-052):
      [contract and pinned small fixtures](exl3-bringup.md) add same-model
      4.0 bpw and mixed-rate 4.5 bpw EXL3 to M2 before settling the artifact
      layout/operation contract. Preserve packed execution and side tensors;
      upstream kernel performance is an M2 gate, resident serving performance
      an M3 gate, and EXL3 switching/restore an M4 gate.
- [x] Run the **small EXL3 reference baseline** (2026-09-22): both full-hash
      D-052 quants execute on Spark in the pinned ExLlamaV3 environment;
      tokenizer/template identities, repeated-logit and in-place cache restore
      controls pass. The [report](experiments/exl3-reference/README.md) records
      four direct Model API profiles and 176 real/synthetic kernel cases,
      dispatch boundaries, tracked memory and statistical comparison rules.
      Full Generator serving controls remain M3 work. ARM host
      helpers require a bounded patch; device kernels are unchanged. Native
      numerical tolerances, complete physical-memory envelopes and reference
      cases flagged unstable remain acceptance gates, not inferred passes.
- [x] Scope the remaining **early backend integration proof**, executed alongside M2
      (2026-09-22): the [proof scope](backend-proof.md) fixes entry conditions,
      stages P0–P6, a five-rung numerical oracle ladder (reference, toolchain
      bridge, native dispatch on conventional memory, host VMM, restored) and
      the BP case matrix for backing/accounting, numerics, paging, lifetime,
      failure, performance and kernel coexistence/swapping.
      Source reading at both pins (not measurement): GGML's CUDA backend hides
      a never-shrinking, aborting scratch pool, cuBLAS workspaces, its own
      streams and a GB10 device-flag side effect, chooses kernels/fusions
      internally and has no custom operation. The owner therefore set
      **D-053**: llmpalooza owns dispatch, and GGML, ExLlamaV3, later or
      llmpalooza-authored kernels are swappable build-time implementations
      selected per operation and plan, several at once. GGML's operation
      launchers take a context whose pool, stream and handle llmpalooza can supply,
      with build-time patches for context ownership, device initialization,
      abort paths and the `static` matrix-multiply routing.
      ExLlamaV3 device kernels separate from their ATen wrappers; autotuned
      grids, output dtypes and compile flags are part of the numerical plan.
      Its GEMV kernel cites GPL-3.0 QTIP code as its structural model: an open
      provenance gate ([licensing](licensing.md#early-exl3-companion-d-052));
      until resolved the native plan runs the GEMM kernel where upstream
      selects GEMV. Execution, thresholds and the contract remain M2 work.
- [x] Defer the retained backing comparison to M2 (owner follow-up
      2026-09-21, D-035; deferred 2026-09-23). It compares D-033's small
      independent handles with larger persistently mapped slabs, including
      1 GiB, and serial with bounded asynchronous submission for mixed
      checkpoint batches. It needs the M2 catalog, leases, storage and completion
      services on the backend proof's P4 harness, so it cannot run in M0.
      Deadline: D-033 is explicitly retained or amended before the internal
      contract is settled and M2 closes.
      The [scope](backend-proof.md#retained-backing-comparison) records the
      open prerequisites: a cross-model swap trace (none exists yet),
      synthetic extents because the dense ~0.5B fixtures are too small to
      fragment slabs realistically, and retain/amend criteria set before
      measurement. It also records that expert compaction waits for M5's
      dispatch choice.
- [x] Define the M4 A→B→A acceptance trace and the bounded retention policy
      (2026-09-22, D-055): the [retention policy](retention-policy.md)
      settles entry identity, adapter restore boundaries, immutable shared
      blocks charged once, refresh by branch, D-041 close semantics,
      capacity-driven expiry with 24-hour per-class idle caps (owner
      decision), an initial victim order, lazy digest-verified spill deleted
      at startup, and fallback/reporting. The owner named M4's workload:
      Qwen2.5-0.5B FP16 GGUF and EXL3 4.0 bpw in both orientations on a
      frozen synthetic transcript under policy-forced budgets. It has six
      llmpalooza arms, including whole-model controls, fresh interleaved
      llama.cpp/ExLlamaV3 references, at least 72 repetitions per arm and
      distribution-free 97.5% bounds, plus exact outputs and logits against
      provenance-matched controls. The functional matrix covers shared
      prefixes, branches, release races, expiry, spill failures and an
      unmodified client. Capacity values are pinned at M3 exit from measured
      state sizes. The transcript, budgets and reference paths are pinned at
      M4 entry. No retention code or measurement exists yet.
- [x] Choose an experimental artifact encoding and layout ABI (open question
      5, 2026-09-22, D-056): the [v0 format](artifact-format.md) uses
      safetensors shards with explicit zero pads, strict JSON manifest/index,
      a manifest-digest artifact ID with deterministic import and one-rename
      publication, exact version/profile rejection with re-import, and
      verification at install/replication/on demand but never at page-in.
      After owner questions during the task, D-056 amends D-035: dependency
      groups (a dense layer, one expert's closure) are single 4 KiB-aligned
      file ranges paged in 2 MiB group-relative chunks, and adjacent misses
      coalesce into vectored direct reads.
      The [layout study](experiments/artifact-layout/README.md) covers seven
      real models: 4 KiB groups leave ≤0.083% disk padding versus 3.49–10.87%
      at 2 MiB. Worked examples cover dense, expert, tied (Qwen2.5's embedding
      and head stored once), small-tensor, sparse-row, tail, chunk-boundary,
      shared-chunk-lease, resident-hole and interrupted-import cases, plus
      EXL3 descriptors and closure rules, and GGML's row-padding over-read.
      Both D-051/D-052 fixtures (FP16 and EXL3) and Gemma 4 built, verified,
      paged back byte-exact with direct reads and passed the pinned upstream
      safetensors reader. The verifier was hardened until a tenth adversarial
      challenge round came back clean (104 unit tests). On `spark`, 4 KiB offsets
      cost at most 2.6% raw read throughput but win on useful bytes; SHA-256
      runs at 2.49 GB/s per core; one process could reserve 128 TiB of GPU
      VA in one range. The pinned GGML expert stride makes uniform-stride
      expert views cost large VA; a per-expert pointer table is recommended
      for M5. Model-parallel partitioning is deferred to M6 entry.
      The C++ importer/verifier are M3; compatibility guarantees stay behind
      D-018's gate.
- [x] Decide the C++ source-dependency mechanism (2026-09-23, D-057):
      [locked CMake FetchContent acquisition and curated vendoring](source-dependencies.md)
      for adapted source units, separate from D-049's SDK. Select and audit
      each profile's complete closure before fetching; configure/build uses
      verified local inputs, with no implicit downloads or implementation
      library substitution. Official tool documentation checked; M1 still
      owes implementation, concrete dependency pins, native/cross offline
      checks and proof of copyleft-disabled exclusion. Existing source
      provenance gates remain open.
- [x] Toolchain decisions: build-system conventions (CMake presets / Ninja /
      LLD, confirmed 2026-09-21), test framework, format and lint pins, CI
      shape including the copyleft-disabled profile, REUSE lint, a separate
      embedded-header check for commentable source/docs, and an
      installable `.deb` build from M1 (D-029), versioning/changelog
      conventions for an externally consumed project (D-016), and the
      installed layout that packaging will need (FHS paths, service user,
      systemd unit; D-027, confirmed 2026-09-21), including D-054's
      storage-role paths and defaults. Record the pins and the
      layout in decisions.md. Provisioning split settled 2026-09-22 (D-049):
      a complete, persistent project SDK, declared system prerequisites and
      shared workstation/container setup. CMake 4.4.3 pinned 2026-09-23
      (D-058), with both Linux archive hashes and seven FetchContent semantic
      checks passing on workstation and Spark; CMake-driven native CPU,
      AArch64 cross CPU/CUDA and native Spark CPU/CUDA smoke also passed
      ([report](experiments/cmake-fetchcontent/README.md)). D-059 (2026-09-23)
      pins Ninja 1.13.2, GoogleTest 1.18.0 (owner's choice), the LLVM 22.1.8
      formatter/linter/language server/symbolizer and GCC 14.2 libstdc++
      headers (amending D-032; the headers were replaced by D-060's static GCC
      16.2 runtime), with candidate style, check and warning sets.
      Native, sanitizer, cross-to-Spark CTest and native Spark runs passed
      ([report](experiments/dev-tools/README.md)). On 2026-09-23 the owner
      chose no hosted CI until the repository has external contributors
      (D-061). The "CI" checks become
      M1's local `check`/`check:full`/`check:spark` mise tiers, and the
      offline gate runs in the reference container with `--network none`,
      because Ubuntu's user-namespace restriction blocks `unshare`/`bwrap`
      on all three hosts (RE-013). AArch64 CPU tests and ASan/UBSan run
      locally under qemu-user (installed on the workstation 2026-09-23;
      leak detection off, RE-014), as does the arm64 package-install test in
      an arm64 container. The Sparks keep GPU, VMM, RDMA/NCCL, target I/O,
      performance and ARM concurrency work, and are never runners for the
      public repository. D-062 sets SemVer 0.x product versions (signed
      owner tags, `~dev` Debian builds), independent surface versions, the
      `llmp-` extension prefix and a Keep a Changelog `CHANGELOG.md` from
      M1. D-063 sets the [installed layout](architecture.md#installed-layout):
      a `llmp` service user, one unit, a strict TOML 1.0 node document
      at `/etc/llmp/llmp.toml` plus `llmp.d/` fragments
      (cluster-design's node-local schema, extended) and D-054's roles
      under `/var/lib/llmp` with their `[storage]` keys. Implementation,
      application build validation and the remaining key spellings are
      M1 work. On the owner's follow-up the same day: includes via
      `llmp.d/` fragments, loopback defaults of 8114 (front door) and
      8115 (management), and directory modes split by role.
- [x] Evaluate static runtime linking with GCC 16.2 (owner follow-up
      2026-09-23, D-060): GCC 16.2 built from GPG-verified source on both
      hosts. libstdc++, libgcc and cudart are linked statically, and Clang
      stays 22.1.8 (NVCC's host limit; owner declined a compiler split).
      Binaries need only glibc (`GLIBC_2.38` at most) plus the driver's
      `libcuda`, which is loaded at run time. The C++23 probe (`<flat_map>`,
      `<mdspan>`, `<print>`), NVCC, the CUDA smoke on the GB10, GoogleTest
      (native, cross over SSH, Spark) and sanitizers all passed
      ([report](experiments/gcc16-static/README.md)). NCCL must be static or
      built with `-static-libstdc++`; cuBLAS static pending its review. The
      owner confirmed that GCC's runtime exception permits the static
      linking. Release builds do not link `libstdc++exp.a`, so libbacktrace and
      its notice stay out.
- [x] First full draft of [architecture.md](architecture.md) (2026-09-23):
      a system map of processes, components, layers and lanes; the request
      path and model lifecycle; identities, ledgers, backing and addresses;
      storage and jobs; execution (operations, plans, adapters, rendering,
      sampling); providers; errors, startup and shutdown; front door,
      configuration, observability and trust boundaries; testing and build.
      It sets initial designs for the minimal provider interfaces, a
      deterministic LRU victim-selection baseline and a sourced memory
      breakdown, and lists the remaining questions with the milestone that
      settles each. The owner answered two drafting questions: D-066 (no
      exceptions; `std::expected` errors) and D-067 (native per-template
      chat renderers; checkpoint template code never runs in llmpalooza). Environment
      inventories moved to [environment.md](environment.md). Follow-up owner
      decisions added D-068 (speculative and block-diffusion shapes designed
      now, executed in M7) and D-069 (configurable switching at completed
      phase boundaries). The owner reviewed and approved the draft on
      2026-09-23.
- [x] Rewrite the provisional ladder into real milestones with exit
      criteria (2026-09-23): the [milestone ladder](plan.md#milestone-ladder)
      gives M1–M8 an entry, scope and exit contract each, carrying every
      obligation other documents assign to a milestone. Owner answers the
      same day: reasoning and constrained output ride with M5's Gemma
      4/Ornith daily drivers; the old M7 splits into M7 (performance and
      the D-068 decoding modes) and M8 (product); the first tagged 0.x
      release comes at M8's exit. Earlier owner input carried in: M3/M4
      stay on the small D-051/D-052 fixtures; M2 stays small, with future
      shapes only in fake-provider scenarios; M4 compares D-069's policies
      with B arriving while A generates; M2/M5 record phase widths,
      rejection reasons and bound-versus-peak; M3 measures startup time and
      metadata memory against library size; M1 proves a minimal confined
      job. Unassigned scope placed: install/download jobs and the Tailscale
      timer in M3; warm jobs, continuation close, request classes and queue
      waits, and metrics/health/load in M4; archive and peer replication in
      M4a; reasoning, D-046's reasoning and cache fields, and constrained
      output in M5; the Ollama subset, tokenization/rendering, raw
      Completions and token diagnostics, embeddings, reranking, file
      inputs, MCP, application permissions and D-046's hint fields in M8.
      "With its model" shapes: hybrid sliding-window and recurrent layers
      in M5, compressed attention and sparse rows in M7 with the large
      pair, modality encoders and pooled outputs in M8; the DiffusionGemma
      study precedes M7 planning.

## Exit criteria (as set)

The owner has walked features.md and says the plan is good
enough to build from; open questions 1–7 and 9 are answered or explicitly
deferred with a reason and a milestone deadline; toolchain pins exist; M1+
milestones have scopes. Proposed optimizations may remain deferred with a
reason and trigger; they need not be accepted or rejected to exit M0. The
paging-feasibility result, measured reference switching baseline, agreed
performance criteria, and question 9's policy are required before M2, even if
deferred out of M0; feasibility sizes M4/M5 rather than gating viability
(D-021, D-025). Question 8 is resolved by
D-022; endpoint documentation verification is recorded in D-040 and D-045;
executed client compatibility remains M3 work. M0 exits on the owner's call,
not on a checklist reaching zero. Both Sparks are reachable (`spark`, `spark-b`); the toolchain, VMM, I/O,
interconnect, reference switching, and bounded paging-feasibility evidence
is recorded above. Sharded two-node reference execution remains separate
from these single-node model captures and is still future work.

## How M0 exited

The owner approved the plan on 2026-09-23, after the milestone rewrite and its
review. Against the criteria above: the feature matrix was walked on
2026-09-21; open questions 1–7 and 9 are answered, or deferred with a
deadline, in [features.md](features.md#open-questions-answer-during-m0), and question 8 by
D-022 with D-040/D-045's documentation checks; the toolchain is pinned by
D-032 and D-058–D-060; M1–M8 have entry, scope and exit criteria in the
[milestone ladder](plan.md#milestone-ladder). The paging-feasibility study, the
measured reference cycle, D-036's criteria and D-050's reservation policy, all
required before M2, are recorded above. The retained-backing comparison was
deferred into M2 with a deadline. Sharded two-node execution is M6 work: the
bounded MiMo TP=2/EP=2 reference above ran across both Sparks but covered no
switching, restore or pressure cases.
