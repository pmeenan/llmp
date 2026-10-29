<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Plan

**This is a living document.** Milestones will be re-scoped, re-ordered, split,
or added as planning conversations and findings come in. That churn is
expected; what is *not* allowed is silent change. Update scope and progress
here as work lands. Only consequential choices that change a load-bearing
constraint or are expensive to reverse get a decision-log entry; routine
scope, ordering, and implementation changes do not (AGENTS.md rule 1).

Check a box only when the item is done and verified; partially done items stay
unchecked, optionally with a note. When a milestone exits, its section shrinks
to a short summary and its task-by-task history moves to a record beside this
file ([M0](m0-record.md), [M1](m1-record.md), [M2](m2-record.md), [M3](m3-record.md)).

**Status legend:** `pending` · `in progress` · `done` · `parked`

## M0 — Plan the plan  `done`

Ran 2026-09-20 to 2026-09-23 and exited on the owner's approval of the plan.
M0 turned the [design brief](ideation.md) into the vision, the triaged feature
matrix, the approved architecture and decisions D-001–D-069, and gathered the
evidence they rest on: toolchain, VMM, I/O and interconnect spikes on the
Sparks; the llama.cpp, EXL3, image and MiMo reference runs; the measured A→B→A
reference cycle; the paging-feasibility study; and the v0 artifact layout
study. The [M0 record](m0-record.md) keeps each task's outcome, evidence links
and caveats. Everything M0 left open is owned by the milestone exits below.

## Milestone ladder

Rewritten from the provisional ladder at the end of M0 (2026-09-23; see the
[M0 record](m0-record.md)), then re-sequenced after M2 by the owner on
2026-09-27 (D-087): the fast full model swap, the owner's first real work,
comes first, on one Spark at M3 and on two at M4, and everything after moved
back two places (the old M3 is M5, M4 is M6, M4a is M6a, and so on to M8,
now M10). After the swap work the order is by risk: one resident model end
to end, with the importer and the full front door, at M5; the first useful
product at M6 (A→B→A with retention); configured placement at M6a;
demand-paged MoE with the first daily-driver models at M7; sharding under
pressure and failure at M8; performance and the remaining decoding modes at
M9; and the remaining product scope with the first tagged release at M10.
On 2026-10-08 the owner inserted M3.6, one engine of shared components,
ahead of M3.5's remaining work, which was parked until M3.6 exited (done 2026-10-17, D-107; see the [M3.6 record](m36-record.md)).
Each milestone leaves a usable, testable result. None has a promised date.

| Milestone | Result | Needs |
| --- | --- | --- |
| M1 Bootstrap | Pinned SDK, builds, local check gate, package skeleton, confined-job proof | M0 (done) |
| M2 Resource core | Catalog, admission and leases on a fake backend and a Spark; the backend proof settles the operation contract | M1 (done) |
| M3 Single-Spark fast swap | DeepSeek V4 Flash, Qwen3.8 Flash Next and Qwen-Image-2.1 swap A→B→A on one Spark, aiming at ~10 s to first token, as correct, fast and lean as their references | M2 |
| M3.5 Model families | The core engine runs the major open model families, MoE and dense (Gemma, Llama, MiMo and the other top-tier families), each correct, as fast as its reference and flat with context, before the system is built around it; image, video and audio file inputs on the models that take them, decision models over the Jev API, and image, video and speech generation routes (D-101) | M3; parked 2026-10-08, resumed on M3.6's exit |
| M3.6 Engine components | One engine of shared components over llmpalooza's own graph IR: the format layer, one runner and serving adapter, the native importer, every current model migrated within 1% of its current speed, and capability-driven optimization/normalization without checkpoint pins (D-107 to D-110, D-123); `done` (2026-10-17, R14–R19 completed 2026-10-20), see the [record](m36-record.md) | M3 |
| M4 Two-Spark fast swap | GLM-5.3 Flash, then DeepSeek v4.1 Flash, sharded over both Sparks in the same cycle, with their image (and GLM's video) inputs | M3.5 |
| M5 One resident model | Importer, verifier, the three client protocols plus the Gemini API and code completion, TLS and management on the small fixtures | M4 |
| M6 First useful product | A→B→A with partial retention; switching policy chosen from measurement | M5 |
| M6a Configured placement | Conductor, enrolled nodes, whole-model placement and routing | M6 |
| M7 Demand-paged MoE | Exact expert paging; Gemma 4 and Ornith as daily drivers with reasoning and constrained output (their resident bring-up is M3.5's) | M6 |
| M8 Sharding under pressure | Sharded execution correct under asymmetric pressure, cancellation and failure, with coordinated admission | M6a, M7 |
| M9 Performance | D-036's benefit target on a library larger than memory; the remaining speculative and diffusion decoding | M7; M8 before exit |
| M10 Product and release | Dashboard, remaining API scope, signed apt repository, first tagged 0.x release | M9, for the release |

How to read the milestones below:

- **Exit criteria are the contract.** Scope lists say what a milestone
  builds; its entry work refines them into tasks. An exit criterion changes
  here, visibly and with the owner, never by drift.
- **Detail lives in the linked designs.** Where a gate cites a matrix or an
  acceptance section, every row that section assigns to the milestone is
  part of the gate.
- **Evidence rules apply to every gate.** Thresholds are declared before the
  measurement they judge, and inconclusive comparisons do not pass (D-036).
  Every result names the check tiers and hosts that produced it (D-061,
  [workflow.md](workflow.md)). No performance gate is met by giving up
  correctness.
- **Clean up at close.** Closing a milestone includes deleting its work files
  on every host used ([workflow.md](workflow.md), "Clean up when a milestone
  closes").
- **Heavy path.** Work in a [blast-radius area](workflow.md#blast-radius-changes-get-the-heavy-path-by-default)
  passes its adversarial challenge before the milestone exits.
- **Support is earned per checkpoint.** From M3 the
  [support matrix](model-support.md) records
  what each exit validated; a milestone exit names its configurations.
- **Release.** The first tagged 0.x release (an owner-signed tag, D-062) is
  cut at M10's exit, after every milestone has exited (owner, 2026-09-23). The
  repository is already public (D-061); until then it carries only `-dev`
  builds.

## M1 — Bootstrap  `done`

Ran 2026-09-23 to 2026-09-24 and exited on the owner's word. M1 built the
pinned SDK and its reference container (D-070), the CMake presets and build
tasks, the source lock with GoogleTest and toml++ (D-057, D-073), the local
check gate (`check`, `check:full`, `check:spark`; D-061), license and
provenance records (D-071), versioning (D-062), `llmp doctor` (D-072), the
node configuration and storage-role checks (D-073), and the arm64 package
with `llmp-runtime`, `llmp.service`, crash handling and confined jobs in
delegated cgroups (D-074), validated on `spark`. The
[M1 record](m1-record.md) keeps each item's outcome, verification and
hand-offs. What it handed on: the GPU, VMM, I/O and ARM stress suites in
`check:spark`, a system-call filter with io_uring, and a single reaper for jobs
started from other threads (M2; the last two moved to M5); front-door, TLS and switching-policy keys
(M5, M6); the importer on the confined-job mechanism and spill file names
(M5); cluster-member checks of credential files and enrollment records
(M6a); drain-before-restart upgrades and the apt repository (M10).
Milestone numbers here follow D-087's renumbering.

## M2 — Resource core and backend proof  `done`

Ran 2026-09-24 to 2026-09-27 and exited on the owner's word. M2 built the
resource core: the catalog, commitment ledger, LRU victim baseline and
materialization planning (D-006, D-007), admission with D-069's pauses
(D-050), D-048's task lanes and scheduler loop, and the providers with
their fakes, CUDA VMM and io_uring (D-026). Its backend proof (P0–P6) ran
GGML's and ExLlamaV3's kernels under llmpalooza's dispatch (D-077, D-080),
selected by plan (D-053), with cuBLAS on a llmpalooza handle (D-076): native
Qwen2.5-0.5B FP16 matches the bridge bit for bit and both EXL3 fixtures
match their approved record and bounds, paged at disk speed through a
host-VMM landing zone into device VMM (D-081), evicted, written back,
restored and relocated bit-identically, FP16 and EXL3 alternating on one
node. D-086 records the operation contract; D-033 is retained; D-068's
shapes are expressible; the `native` build also targets discrete `sm_86`
GPUs (D-082); and D-085 made performance and memory coarse end-to-end
checks, so BP-F2 did not run. The [M2 record](m2-record.md) keeps each
item's outcome, evidence and caveats, and the gate. What it handed on,
in D-087's numbering: BP-F4's per-token host cost, the D-033 handoff of an
evicted model's backing, and ReconGemm's descriptors prepared once per
bound GEMM (the swap work, M3 and M4); end-to-end parity of each engine
with its reference on the small fixtures (BP-F3), the importer,
suballocation with state blocks, D-050's moved rows (suballocation holes,
stalled-client termination, every queue full at once, capacity-loss
injection), and M1's io_uring system-call filter and single job reaper
(M5); victim selection on a miss, spill as retention with D-055's spill
format, the cohort pause with several paused peers, fork and
copy-on-write, cached-state promotion and fast-swap validation on the
discrete GPU (M6); the runtime closure-excess check (M7); repeated
speculation with prefetch (M9); and, with no milestone yet, `spark-native`
sanitizer presets and the EXL3 memory outside the catalog in the paged
harness.

<a id="m3--single-spark-fast-full-swap--in-progress"></a>
<a id="m3--single-spark-fast-full-swap-in-progress"></a>

## M3 — Single-Spark fast full swap  `done`

Completed 2026-10-04. DeepSeek V4 Flash, Qwen3.8 Flash Next and
Qwen-Image-2.1 execute natively with prepared swaps, paging, saved state
and stable-address graphs. The final 32-row/six-pair swap table passes
with a worst prepared LLM swap of 9.853 s against the 20 s bound. Both
LLMs batch chat and literal completions, and pending model switches
time-slice cohorts with exact resumed continuations. Standard-client,
long-context, numerical and memory controls pass in their named scopes.

The owner accepts remaining Qwen/DeepSeek speed gaps and defers further
tuning to **M9's full-engine optimization pass** (2026-10-04), preserving
correctness, memory and swap bounds. No speed miss becomes a parity claim.
The [frozen M3 record](m3-record.md) retains all task history, evidence,
exceptions and later work. Focused workstation, Spark and ARM package
checks pass; whole shipment tiers remain owed before publishing a package.

## M3.5 — Model families  `in progress`

**Parked 2026-10-08 for M3.6 (below)**
(owner, D-107); **resumed on M3.6's exit** (2026-10-17). M3.5's work runs on the component
engine: there a family is an architecture mapping and any components its
shape lacks, not a runner, so items below that name a runner, plan or graph
builder are re-expressed against the shared engine when they resume. Their
history and evidence stay valid as recorded. The transfer item is retired.
Kernel-level work behind the per-operation contract may continue meanwhile.

**Carried in from M3.6** ([record](m36-record.md#deferred-items-from-the-reviews)
holds the history; this list is the whole queue of what is still open, with
nothing left to "the record's carried list"; RE-058 is fixed in R6 and not
carried):

- Importer and formats: MLX serving qualification
  (imported and admitted, not qualified for serving); the split NVFP4 experts'
  own fast products (a grouped vector product for decode rows and the
  block-scaled grouped GEMM over split scales); the MXFP4/MXFP8 relayout of
  routed experts and dense NVFP4 linears; the stored per-geometry image spec
  (with U8); full-weight Gemma 2/3 27B qualification.
- Families: Nemotron 3.5 Super VL ([m35-families.md](m35-families.md)).
- Engine and runtime: masking head rows past `tokenizer_vocab`
  ([RE-050](rough-edges.md)); `Q4_0` as a KV cache encoding setting; Qwen2's
  single-owner policy, the DeepSeek-only schedule products and the kernel
  capability predicates that name DeepSeek, Qwen3.8 or Gemma dimensions (U7
  below).
- Test and build infrastructure: the `native` preset test binaries' cuBLAS
  `RUNPATH`; `cpu-asan` `-fsanitize=function` on GGML `to_float` calls and the
  importer mutation test under ASan (both in the `check:full` and
  `check:spark` shipment tiers, still owed); watching
  `CudaPageIn.RepeatedCancellationsNeverCorruptAReassignedSlot` for a repeat of
  the one-off `5b21fe48` segfault. M5 owns the standalone artifact verifier.

M3.6's other hand-offs go on from here: M4 owns multi-device lowering and the
partitioning residue (the record's "Carried to M4"), and M9's list below owns
the performance leads.

**M3.6 review follow-ups (owner, 2026-10-17).** A whole-milestone review of
M3.6 found the one-engine structure sound but family knowledge moved into
import pins, per-mapping fusion lists, kernel shape whitelists, admission
branches, duplicated attention components and an unwired decode ladder,
plus a set of correctness defects. The owner asked for all of it to be fixed
before new families land; each item lands as one reviewed commit.

- [x] R1 Runtime and engine correctness (`aba73a61`: HTTP intake, hang-ladder drain,
      approvals envelopes as defaults, kept-state and calibration fixes,
      spill-range growth, plan-cache index, the `Llm` interface, legacy code)
- [x] R2 Kernel, decode and IR correctness (`40bf23e3`: validator gaps, CUTLASS and EXL3
      error paths, alignment, per-device opt-ins, latent IR rewrite bugs,
      dead kernel code)
- [x] R3 Importer correctness and structural normalization (`f5ebd16a`, D-120: safetensors
      metadata, composition roles, open-world keys, no encoding or shape pins,
      kept tokenizer documents, reproducible artifact identity)
- [x] R4 Mapping-driven admission and spec-derived bounds (`fe17e4ae`, D-121: no architecture
      branches in the runtime or CLI, container variants, gates and folds as
      mapping data, row and vocabulary bounds from the spec, admissible
      formats from capabilities; native Gemma 2/3 HF mappings and tokenizer
      assets, with 27B structural/tokenizer checks and full-weight qualification
      still pending)
- [x] R5 Kernel gates by real limits (`80ede618`: F16 GQA attention at head sizes
      64–192, parametric norm and MoE fusions, EXL3 beyond one fixture)
- [x] R6 The decode ladder in every backend (`43f2f50a`: dense NVFP4, MXFP8 grouped,
      MXFP4; honest rungs, feature bits and measured ranks)
- [x] R7 Planner (`d3374c67`: quality-preserving fusions on by default, plan and
      coverage reports, family constants out of the planner, `Operation`
      retired, activation memory reduced)
- [x] R8 Tokenizer (`09ffb557`: GGUF and `tokenizer.json` pre-tokenizers from their
      regexes, post-processors)
- [x] R9 Components (`0d28a553`, D-122: one attention core with YaRN, sections, gate, sinks,
      compression and selection, merged fused kinds, format-independent specs,
      the image text encoder and denoiser from token components)
- [x] R10 Serving (`e6c3617a`: one LLM route, every model batches by default, prefill
      policy in the engine, build and step phases, the image staged executor
      in the engine)
- [x] R11 Generality canary (`b003c9a5`: a generic dense-decoder mapping and default
      serving recipe; Llama 3.2 1B Instruct and Qwen2 served)
- [x] R12 Test coverage (`aa92bd1f`: layer-level reference-against-GPU checks per family,
      the Qwen2/EXL3 controls, `RESOURCE_LOCK llmp_gpu`, maintainable frozen expectations)
- [x] R13 Docs: `engine.md` and `portability.md` describe the current code,
      "how to add" checklists, `AGENTS.md` trimmed, the record's carried items
      triaged, and closeout verification recorded
      ([m35-r13-closeout](experiments/m35-r13-closeout/README.md))
- [x] R14 Concrete correctness fixes across IR, planner, spec, components,
      format, engine and gate tools (`575b04a4`: `Op::kRoute` contiguity in `Decompose`
      and `CompactRouteReaders`; `Decompose` role/view/rounding invariants in
      `Route`, `HcPost`, `Pool`, `kGatedResidualModulateSplit`; `Graph::Digest`
      `kExl3Trellis` params; `Partitioner::LocalIds`/`ExpertMask` `int32_t`
      constant bytes; `planner::KeyOf` full `PlanKey` initialization;
      `attention.dense` `explicit_stateless` `output_gate`/`sinks` and
      `Reference` `scale_scores`; `StepContext::Save`/`Hadamard` builder
      segment preservation; `EmitKvCacheStore` `d.kv_encoding` and
      `ApplyKvCacheEncoding`/`KvCacheEncodingBytes` `kF16`/`kQ4_0`;
      `CheckApplied` fold tensor validation; `Exclusive`/`ValidateMapping`
      disjoint `layer_types` and `FindMapping` collision; `kScaleAtoms`
      stacked row atom alignment; `GrowMapped` host charge rollback;
      `TearDown` multi-model detach; `PromptBatchInventory` per-shape fallback
      flags; `StepRuns::QueueRouted` per-segment capture and `routed_outputs`
      check; runner-scoped `PlanStep`; Python gate tool store fallback and SSE
      error check)
- [x] R15 Unpinning runtime admission, numerical qualifications, policies and
      settings (`c0cdf812`, D-123: derive `NumericalQualifications` and changed-arithmetic
      fusion eligibility from component roles, tensor formats/encodings and op
      shapes rather than `approvals.cc` SHA-256 receipts, envelope bounds or
      fixed `1024x1024` / 25-token image geometry; unify `AdmittedPolicy`,
      `ProductPolicy` defaults, `ResolveSettings` default context and
      `wide_context` slot clamping onto `spec::Mapping` and `ServingRecipe`
      for pinned and unpinned checkpoints alike; admit supported encodings in
      `ReadImageAdmission`; deduplicate staging helpers across
      `component_build.cc` and `component_image_build.cc`; wire multi-drafter
      configurations in `ComponentLlmBuild`)
- [x] R16 Capability-driven kernel gates, on-chip decode ladder and EXL3
      (`d1356730`, D-123: replace family-specific shape, head-count, GQA-ratio and
      `sm_121` whitelists in `src/kernels/ggml/capabilities.cc`, `lower.cc`,
      `validate_ext.cc` and `fattn_owner.cu` with true hardware/algorithmic
      predicates; wildcard model and shape rules in `DeviceRank`; enable
      `zero_point` and `AffineScale::kF32` in `src/kernels/decode/`;
      generalize mixed-encoding `fused.mul_mat_glu` and paged `kGetRows`;
      keep shape-generic `exl3.linear.gemv` and `exl3.bias_add` active in
      `DeviceSelectedSource`, derive default `GemmPlan` concurrency from device
      SMs, and remove pageable `cudaMemcpyAsync` in `Bound::Run`)
- [x] R17 Component unification, hybrid layer mappings and spatial/token bridge
      (`8936e175`: support `placement = "pre"` and `"sandwich"` on `attention.gdn`,
      `attention.mqa` and `attention.qsa`; make `gdn.value_heads_tiled`
      conditional and remove hardcoded DeepSeek V4 `idim`/`ih`/`ratio` checks
      in `attention.mqa`; deduplicate `attention.borrowed` into
      `attention.dense`; allow `{kF32, kBF16, kF16}` and `kNormZeroCentered`
      across all norm roles in `ffn.moe`, `hc.mix` and `defs.h`; support
      character-per-layer hybrid pattern strings and $>64$ layers in
      `LayerTypes()` and declarative drafter stage synthesis in
      `spec/mapping.cc`; remove the `f.b.segment() != kJoinedSegment`
      side-channel in `conditioning_blocks.cc`/`image_blocks.cc`, and extend
      `conv.spatial` to arbitrary kernel/stride/padding)
- [x] R18 Mapping-driven importer, role-driven expert relayout and
      tokenizer/chat generalization (`8b3eafdd`, D-123: lifted rule and key caps in
      `SourceMapping` and `normalize.cc`, automatic `FoldDerivedParameters`
      across `F32`/`F16`/`BF16` and role-driven `RelayoutExpertTables` over
      `spec::Model` including `kMxfp8`/`kMxfp4`, unpinned semantic tensor
      digests in `modelopt_qwen38.cc` and `mlx_qwen38.cc`, and `--component mtp`
      drafter normalization on MLX; accept nested `rope_parameters` and
      `layer_types` in HF `safetensors` import; broaden HF `tokenizer.json`
      `ByteLevel`/`Digits`/`rstrip` and `sentencepiece.cc` trainer fields;
      single-pass `kPrefixEnd`/`kMessageEnd`/`kGenerationPrompt` boundary
      emission and multi-token stop discovery in `RenderInterpreted`)
- [x] R19 Engine simplification, `BackingService` unification (`RE-059`),
      workspace virtual capacity and closeout verification (`fa1d9073`: route
      `CarryRegion` and `UndoCarry` 2 MiB handles through `BackingService`
      `reserve_` and `creates_`; reserve 16 GiB default virtual capacity in
      `MapWorkspace` so runtime workspace growth does not relocate base VAs;
      skip redundant catalog closure RPCs in `ContinueRouted` for base-leased
      expert pages; deduplicate runner boilerplate between `ModelRunner`,
      `StagedExecution` and `SetupRowTables`; run Spark verification and update
      the M3.6 record and docs)
- [x] R20 Engine and runtime correctness fixes from the 2026-10-20 review
      (hang recovery rung 2 acknowledges its stream fence so an unproven
      step or route no longer disables the model, `PagedNode::FenceStream`
      proving a lost direct step's end; a routed flight counts as started
      only once queued; the cooperative cohort's drain and switch follow the
      current queue head; idle state priced as spill or drop as its take
      does, with deleted spills charged; restored conversations' retained
      spill files counted against and released for `spill_budget_gib`; a
      capacity-refused restore keeps its D-105 record; a partly failed spill
      credited by measured catalog gain; departed clients' waiting parses
      freed; one activation/hang helper; catalog reads in `PagedNode::Load`
      on the scheduler thread)

**Gemma31 gap closing, 2026-10-06:** [state reuse across Clear, capture beside
eager execution and the fused decode FFN](experiments/gemma-gap-closing/README.md)
take the bounded C1/C4 paid cycles from 3.15%/3.51% to 0.73%/1.36% slower than
stock in fresh bookends. Third cycles match stock within 0.1%, and native
tokens and histories match the frozen proof. C4's four-owner decode (+3.2%) is
the remaining gap. State reuse and capture beside also apply to DeepSeek V4 and
Qwen3.8. The [earlier handoff](experiments/gemma-performance-review/README.md)
keeps its context. The follow-up removes C4 owner-attention copies (−1.5 ms a
wave) and adopts a [bounded Gemma26 recipe](experiments/gemma26-production/README.md):
1024-row prefill, norm chains, MoE route/reduce and joined owner attention take
Gemma26 C1/C4 from 57%/41% slower to within 0.6% of stock, with stock's exact
tokens and byte-identical corpus heads. The [decode hot path](experiments/decode-hot-path/README.md)
then removes the host's share of each step (every model takes the path; Gemma
measured): request steps run
on the driver (D-106), the runners make no scheduler call per step, and greedy
Gemma steps choose their token on the GPU. The round trip falls from
85–188 µs to 6–11 µs, and Gemma26/31 C1 and Gemma31 C4 decode gain
2.8%/0.6%/1.1% with identical tokens; Gemma31 and Gemma26 now run level with
or ahead of stock's recorded cycles. Remaining milestone gates stay open.

**Gemma bounded-recipe status (parked with M3.5).**
The [bounded dense31 serving bridge](experiments/gemma31-serving-bridge/README.md)
now selects the checked ordinary recipe only at resolved context<=8192/slots<=4,
with current-pin C1/C4 quality, complete 1K corpus parity, HTTP controls and
focused default-selection checks. The [bounded Gemma26 recipe](experiments/gemma26-production/README.md)
reaches stock parity with stock's exact tokens under the same bounds; larger
envelopes retain their prior recipe; sustained performance and broader
qualification remain open.
The historical SOURCE14 serving/default proposal (a 14-file runtime, settings
and test change set whose receipt the
[Gemma31 C2 first screen](experiments/gemma31-production-c2/results.json)
records) was not adopted. Its original
[Gemma31 C2 first screen](experiments/gemma31-production-c2/README.md) retains eight
positive-margin strict differences; the [private local-MMA factor](experiments/gemma31-c2-local-mma/README.md)
retains nine. The opt-in [actual two-owner factor](experiments/gemma-small-owner-attention/README.md)
now passes strict zero-margin choices and separate 64-target conditional-loss
bounds for both approved C2 profiles. All 64 paid 31B heads match retained stock;
26B has 34/66 exact heads with zero choice differences. Whole2/3 primitive proof
passes heads16/32. The 31B C3 short screen passes strict choices; the
[equal-width partial adapter](experiments/gemma-partial-owner-attention/README.md)
recovers the preserved C5 failure (eight strict differences and +3.8918% conditional
loss) to zero differences, 160 paid exact heads and −0.20935% conditional loss.
The [Gemma26 C5 transfer](experiments/gemma-partial-owner-transfer/README.md)
retains two strict differences despite a passing 160-target loss gate; N7/N9
operator controls pass. The [private prefix-only keep factor](experiments/gemma26-c5-prefix-keep/README.md)
removes both paid differences and matches all 160 paid full heads, but retains one
strict frontier difference; conditional loss passes at +0.08119%. This is diagnostic
evidence, without a production layer policy or an observed routing selection over
stock's 992-row prefix ([prefix-keep factor](experiments/gemma26-c5-prefix-keep/README.md)).
C3 transfer, other partial model counts and unequal widths outside the
[bounded C2 common-root recipe](experiments/gemma4-bounded-owner-roots/README.md)
remain unqualified. That recipe restores all 68 stock heads on both short and
physically wrapped controls, preserves initialized states and removes temporary
KV copies. Same-binary paid latency falls 6.01% / 5.24% (26B / 31B), within 0.1%
of current stock on short n=2 screens. Ordinary context<=4096/exactly-two-slot
HTTP adoption passes; configured-four C2, wider/partial cohorts and compatible
multirow prefill remain owed. Fresh
31B timing is +1.47% with a stock spread larger than its mean gap; 26B is −1.064%
in its short matched bookends. Core opt-in controls retain their historical scopes and do not adopt SOURCE14
serving/default files. The separate current bridge above closes bounded dense31
HTTP and corpus gates; broader depth/context/cohorts and sustained qualification
remain open.

The [C1 phase attribution](experiments/gemma-state-phase-attribution/README.md)
measures paid Clear 57.45 ms, state growth 162.18 ms and required planning 0.286 ms
with 160 plan hits/no misses, while complete outputs/state match the fixed own
proof. It changes only optional runner diagnostics. The attribution result itself
selects no serving policy; bounded dense31 adoption is established by the
separate production bridge above; Gemma26 has its own bounded recipe since. Nested counters and stream wall do not establish a residual
kernel cause or a new reference speed gate.

Goal (the owner, 2026-09-29): build out the core engine across the major
open model families, MoE and dense, before the system is built around it
(M4 onward). Each family runs natively on one Spark from a prepared
artifact on M3's engine skeleton. Each is correct against its same-format
oracle, at least as fast as its same-format reference, and flat with
context wherever its architecture allows. A family that needs a new engine
mechanism exposes the gap now, while the engine is still cheap to change.

**Entry:** M3 exit, including its engine skeleton and its "adding a model
family" guide, and its long-context scaling work.

**Scope:**

- [x] **Physical memory modes (owner, 2026-10-16/17): the first M3.5 work
      item.** Implemented under D-124 (2026-10-21). The accepted M3.6 exception
      for DeepSeek DSpark's C4 wall has this as its fix ([RE-059](rough-edges.md#re-059-gb10-kernel-speed-depends-on-where-the-kernels-free-memory-puts-a-device-allocation--2026-10-17-status-resolved-in-d-124):
      the driver backs a resident workspace with scattered 64 KiB pieces on a
      long-running host, which costs big prefill steps about 1.3% GPU time),
      and the image's response to placement is not monotonic, so placement
      must be made deterministic rather than worked around. Scope
      ([features.md](features.md), "Physical-memory mode per node", now
      `confirmed`):
      - the **dedicated node pool**: physical handles allocated once at start
        while memory is clean, up to the node budget, recycled on eviction, so
        placement is deterministic and swaps never fragment host memory;
      - the **shared-machine mode**: inference brought up on demand, the
        working set idled out after a configurable time, and physical backing
        released by manual, idle-timer and pressure-driven levers, re-acquired
        on demand;
      - configuration per node (`[memory] mode = "dedicated" | "shared"`, `budget_gib`),
        platform-neutral behind the provider interfaces (D-110);
      - heavy-path design and review (workflow.md: it touches the pager, VMM
        mapping and the catalog; D-124);
      - **re-gate** DeepSeek DSpark's C4 wall, the Qwen-Image generation and
        Gemma 2's prefill with it, and retire M3.6's accepted DSpark exception
        when the first passes ([record](m36-record.md#accepted-exceptions)):
        passed on `spark` (2-block matched confirmation against `61f8426f`,
        2026-10-21: `dspark` `C1/C2/C4 prefill_s` +0.12% / +0.97% / +0.47%,
        `C4 wall_tok_s` −1.99% faster, `C2/C4 decode_tok_s` −10.27% / −17.57%
        faster with the `PrefillOrchestrator::Predict` multi-slot speculative
        continuation lookahead fix; `image` `generation_s` −0.01%; `gemma2`
        `C1/C2 prefill_s` +0.32% / −1.18%, `decode_tok_s` −1.97% / −3.20%
        faster), retiring M3.6's DSpark C4 wall exception.
      Its first sub-slice was U1's engine seams (the M3.6 generality
      follow-ups below).
- [ ] **Drafter confidence calibration screen (owner, 2026-10-20): early
      M3.5, beside the memory modes** (independent of them, and light: a
      diagnostic, no default changes). Bind DSpark's `conf_proj` (imported
      but unbound today, `src/spec/mappings/dsv4.cc`) through the shared
      drafter mapping as a per-row confidence output, and record predicted
      against realised acceptance (cumulative per row, calibration buckets,
      mean |confidence − acceptance|) on the `--check greedy` prompts and
      longer agentic prompts, C1 and C4. If it is calibrated, land
      **confidence-budgeted verification** in the shared speculation
      scheduler here: verify rows per request per step from the drafter's
      confidence, costed by the calibrated wave and depth costs (no clock
      reads), allowing depths past today's 3 up to `dflash.block_size`,
      replacing the whole-wave speculate/plain switch with per-request
      budgets, adopted on measured C1/C2/C4 decode gains under D-085.
      Greedy output is unchanged. Caution: Qwen3.8's confidence stopping
      was neutral (optimization-inventory.md, T81), while its counted
      adaptive depth 2–3 (T45) stays the selected policy there
      ([assessment](deepinfer-assessment.md#1-confidence-budgeted-verification)).
      Built with [Flare's](flare-assessment.md#3-speculation-details-calibration-screen-sampled-speculation)
      shapes (owner, 2026-10-20): one per-request, per-depth, per-source
      acceptance profile (conditional acceptance per depth, expected tokens
      at width W = 1 + p1 + p1·p2 + …), carried with conversation state as
      `AdaptiveDepth`'s is, replaces `AdaptiveDepth`'s two-depth choice and
      `AdaptiveWaveMode`'s whole-wave average where it measures no worse
      (T45 stays Qwen3.8's policy until then), and confidence refines it
      once a proposal exists; a draft position is admitted on confidence
      through that position only (causal prefix, so sampled drafts stay
      unbiased); a drafter that exposes confidence (DSpark first) runs
      draft and verify as two graphs so verify width follows confidence
      (one host decision, D-106's 6–11 µs); and verify graphs are keyed
      by total rows with per-request row counts and offsets as device
      inputs, packed by a small dynamic program, so per-request widths
      never multiply captured shapes.
- [ ] **Sampled speculation** (moved from M9 by the owner, 2026-10-20;
      with the calibration screen, which touches the same drafter outputs
      and verify path): in the shared verify path for every drafter
      (DSpark, Qwen3.8's MTP and later ones), drafts are sampled from the
      drafter's own distribution q at the request's settings and accepted
      by min(1, p/q) with residual resampling, on the device, replacing
      the greedy-drafter verdict (a greedy drafter stays the point-mass
      case). Today temperature 1 costs speculative decode 26% on Qwen3.8
      (32.2 against greedy's 43.5 tok/s) and 9% on DeepSeek (27.1 against
      29.8; tokenizer.md, "Sampling", which attributes the loss to sampled
      drafts being accepted less often; acceptance itself is unmeasured). Exact in distribution, so no quality mode: M9's
      speculative-sampling criterion (exact acceptance and residual rule
      against a reference implementation under fixed seeds, sampled
      outputs within declared statistical bounds, the PIT/KS check) applies
      here. Adopt on measured C1/C2/C4 sampled decode
      ([assessment](deepinfer-assessment.md#2-speculative-sampling-with-the-drafters-distribution)).
      Verify and commit state are also tested with acceptance forced by a
      one-token constraint, which removes drafter noise
      ([Flare](flare-assessment.md#3-speculation-details-calibration-screen-sampled-speculation)).
- [ ] **History-copy drafts, measure first** (owner, 2026-10-20, from
      [Flare](flare-assessment.md#7-history-copy-drafts)): a proposal source
      in the shared speculation scheduler that suffix-matches the
      request's own history (n-grams within a bounded window) and proposes
      what followed, merged with any learned drafter under one verify
      budget, acceptance tracked per source in the profile above. Its q is
      a point mass, so it is exact under today's verify rule and under
      sampled speculation. Measure acceptance and decode gain on recorded
      agent transcripts (code edits, repeated tool output) first; then
      offer it to models with no drafter where their verify shapes and
      rollback contracts exist.
**M3.6 generality follow-ups (owner, 2026-10-20).** The M3.6 review of
2026-10-20 found the one-engine structure met but much family knowledge
moved down a layer: kernel capabilities whitelist checkpoint shapes,
numerical admission keys on container and component-kind proxies and on
GB10 only, schedule products are opt-in per mapping, and HF-quantized and
HF-MoE sources import only through C++ pinned to single Qwen3.8
checkpoints. Its correctness defects landed in `101fde83`, `a2aa3b87`,
`6840b1db` and `639583b9`; the items below are its architecture work, which
makes D-107, D-110, D-123 and rule 11 true where the code does not yet meet
them (no load-bearing constraint changes). The owner took the review's
recommendations (2026-10-20): the generic import route comes before Nemotron
and every other new family; an agent proposes the second-architecture proof
checkpoints from the archive catalog and the owner approves; GB10 numerical
evidence does not carry to `sm_86`; measured selection is checked-in rank
data per device class, not autotuning; the second image model is chosen with
the media routes item; the default-off repeat-pending drafter ledger is
deleted. **Order:** U1 is the memory modes' first sub-slice; U2 and U3 run
beside the memory modes (separate code; Spark check capacity is the limit);
U4, U5 and U6 next; U7 after the memory-mode re-gate, so gates measure one
change at a time; U8 before a second image model; U9 at M4 entry. Every
item that touches selection keeps each M3.6 speed-table row within its band
(the tiered two-block confirmation per family, the full gate only near the
line), and a newly covered shape takes the fast path in its plan report.
Decision entries land with U3 (one mapping per architecture, transforms
planned by role and encoding, manifest mapping provenance; amends D-109,
D-113, D-120), U4 (numerical evidence as component data scoped by device
class; amends D-123) and U8 (the image stage graph as spec data; amends
D-122).

- [x] **U1 Engine seams for the memory modes** (`61f8426f`). One budget owner held
      live by the scheduler, one engine-side backend factory, one fence-proven
      unproven-work object, `PagedNode` split along its contracts, a kernel-library
      name boundary check. **Exit met** ([record](m35-record.md#memory-modes-and-u1u10)).
- [x] **U2 Test-only paths and v0 removed** (`dae227dd`). `PlanGraph`, `Qwen38Commit`,
      the v0 artifact reader and the repeat-pending ledger are deleted; the import
      fuzz entry reaches synthesis, normalization, planning and the v1 readers with a
      checked-in corpus. **Exit met**; the v0 stores were removed on 2026-10-23
      ([record](m35-record.md#memory-modes-and-u1u10)).
- [x] **U3 One generic import route** (completed 2026-10-23; D-125, D-127). U3a one
      mapping per architecture, U3b quantized parts and experts, U3c Llama3 RoPE
      scaling. **Exits met** at the recorded scope; no general Gemma NVFP4 quality,
      assistant numerics, new device or encoding, batching, maximum-context or speed
      claim ([closeout](experiments/m35-generic-import/canonical-mappings.md),
      [record](m35-record.md#memory-modes-and-u1u10)).
- [x] **U4 Numerics as component data, per device** (completed 2026-10-23; D-126).
      Admission joins component roles, encodings and device evidence for pinned and
      unpinned artifacts alike; a device without evidence runs the exact path and
      says so; no `sm_86` grant. **Exit met**; recorded failures and limits stay in
      [component-numerics](experiments/m35-component-numerics/README.md) and the
      [record](m35-record.md#memory-modes-and-u1u10).
- [x] **U5 Front-door and scheduling seams** (completed 2026-10-22). U5a route-owned
      stream events, U5b structured message parts, U5c scheduler-owned cohort policy,
      U5d named activation/reclamation phases, U5e one concrete `Llm`. Tool calls
      (shipped, bounded qualification only) and Messages stay separate items.
- [x] **U6 IR and planner structure** (completed 2026-10-25). One checked trait table
      (`src/ir/op_traits.inc`) for every IR op's dispatch; all registered rewrites
      have synthetic match and refusal controls. Revised by F6: exact-site comparison
      left production and the U6 rank rows are retired
      ([fusion-ranks](experiments/fusion-ranks/README.md)). **Exit met.**
- [ ] **U7 Fast paths by capability** (after the memory-mode re-gate and
      U4/U6; five sub-slices, each moving its fusions' patterns to matching
      against their decompositions, which three of the sixty use today):
      - **U7a:** the [first shared GDN gate spike](experiments/gdn-gates/RESULTS.md)
        removes the 48-head and 16-row whitelist with checked storage bounds,
        non-reference-shape correctness and a passing registered 48×1 speed
        screen. Remaining kernels: HC width and stream count, the
        weighted reduce's width and slots, QSA's head and indexer sizes,
        DeepSeek's Q-head and output-A shapes and encodings (output-A is Q8_0
        only), owner attention's head counts, sizes and softcap, and the MXFP8
        GEMV schedule by shape class (keyed on `ne[0] == 2560` today); move
        `lower.cc`'s shape-literal sub-selections into capabilities as
        separate implementations so plans and plan keys show them; merge or
        select by capability the ops with one kernel per family (`kRoute`,
        `kMoeCombine`, `kRmsNormMulRope`, the two HC sets); rename the
        family-named kernel files and implementation IDs.
      - **U7b:** a boundary-style lint rejecting model-dimension literals
        in `capabilities.cc`, `lower.cc` and kernel sources (an allowlist
        for hardware and algorithm constants), and "a fast path gated on one
        model's dimensions" added to workflow.md's review criteria.
      - **U7c:** screen the schedule products (`compact_experts`,
        `pair_experts`, `pair_glu`, `pair_glu_q8`, `vector_floats`,
        `d2r_experts`; only DeepSeek and DSpark enable them, and `dense_pair`
        also the generic dense decoders) on Gemma 26B-A4B GGUF, Qwen3.8 GGUF
        and NVFP4 and DeepSeek under D-085, and enable each by capability
        where it does not lose; per-mapping settings remain only for measured
        exceptions; `d2r` is not bit-identical and takes the same-quality
        check (rule 11).
      - **U7d:** encoding coverage (rule 11): a dequantize-to-F16 plus
        cuBLAS rung for any GGML type without a tile kernel (IQ1_M and Q1_0
        have no device path today); an FP8 E4M3 block-scaled product for the
        DeepSeek and Qwen FP8 safetensors U3b imports; EXL3 device instances
        for 3inst, mul1 and 2–3 bpw (MCG at 4/5/6/8 bpw only today);
        quantized KV at head size 128 by capability fallback and kernel
        instances instead of a refusal in component validation
        (`components/attention.cc`); the MXFP8 scale swizzle done once at
        import and recorded as a layout (D-108, D-113), not on every prefill
        step.
      - **U7e:** Qwen2 batches: remove its single-owner policy
        (`max_decode_rows 1`, `owners_alone`, kept to preserve frozen v1
        digests) and re-pin the digests (rule 11: every model batches, speed
        over bit-exactness); record any remaining single-owner envelopes
        (Gemma 4 outside its envelope) as exceptions with their reason.
      **Exit:** a capability test on a synthetic non-reference shape takes
      each formerly whitelisted fast path; the lint passes; the M3.6 speed
      rows hold; the format coverage report lists a device path for every
      encoding the importer accepts; Qwen2's C1/C2/C4 is measured against
      its comparator.
- [x] **U8 Image pipeline as data** (completed 2026-10-23; D-122). Stage-graph recipes
      are mapping-owned; the pinned BF16 image is identical and a second VAE/DiT
      configuration imports without family wiring (bounded: no composed Ming
      pipeline, serving or quality) ([evidence](experiments/m35-image-pipeline-data/RESULTS.md),
      [record](m35-record.md#memory-modes-and-u1u10)).
- [ ] **U9 Partitioning and per-model residency** (M4 entry; amends M4's
      "Concurrent models and the shared workspace" item).
- [x] **U10 Review leftovers** (re-verified 2026-10-22). Pinned regular-file readers,
      drafter carry across spill/restore, per-activation paid swap receipts, pinned
      kept and lock files ([record](m35-record.md#memory-modes-and-u1u10)).

**Foundation review follow-ups (owner, 2026-10-25).** A cross-verified review
of `ca32156b..265c821a` found that broken image serving, the memory modes, the
agent (tool-call) path and the exact-site comparison machinery in production keep
the U-series from being a clean foundation; the owner asked for every finding
and suggestion to be fixed. Each item lands as one reviewed commit.

- [x] **F1 Image binding and RoPE scaling:** a closed request-scalar vocabulary
      (every Qwen-Image request was refused at `265c821a`, and its text encoder
      could not import), image generation in the default models set, dense RoPE
      scaling types mapped or refused instead of run as YaRN, the IMROPE
      capability aligned with its check, and the 256-block scale-atom cap removed.
- [x] **F2 Memory policy:** a live pool target (dedicated pool charged within the
      budget; pressure and release free backings first), the shared-mode shrink
      formula and floor, `SetBudget` as a scheduler command, a device-aware
      default mode and discrete sizing, the follow-up switch release tail; D-124
      accepted.
- [x] **F3a Tool-call correctness:** bounded JSON depth, growth-funded tool
      output, one request-local refusal channel, one grammar shared by parser,
      sampler and judge, Llama leading-JSON calls, self-opened reasoning, marked
      forced spans, linear-time result association.
- [x] **F3b Tool-call hot path:** device greedy and verify kept when the mask is
      the identity, lazy and class-based masks, template-derived openers, every
      declared tool rendered.
- [x] **F4 Engine recovery:** one fence-proof unproven-work contract with
      recover-then-serve tests per owner, MTP drafting after plain steps,
      image fence acknowledgement and request layout, row-read settling,
      checkpoint staging re-reserved after a fence, history copies (the
      generated-input classification item was superseded by F6).
- [x] **F5 Serving:** output delivery as host-only state, one cohort decision
      function, kept-record commit by descriptor, clock-skew and lock-file
      edges, a startup allowance that no longer sums sequential phases, a
      shared-mode floor of the state requests hold (refused growth is lent
      budget back), and a memory release that waits at most one turn.
- [x] **F6 Comparison out of production:** exact-site recipe comparison moved to
      `benchmarks/` (dense decoders could not finish a prefill plan at
      `265c821a`); coarse, versioned rank rows in `src/`.
- [x] **F7 Decode GEMV and planner cost:** measured rank rows instead of
      capability vetoes (GGML's MMVQ at 2–8 columns where faster, fused GLUs
      split where the decomposition is >5% faster; one-pass 8-column VecQ was
      measured slower and dropped), upstream prefetch, a row-strided
      `llmp.argmax`, stable op identity in digests, cheaper inference and
      planning; geometric decode buckets built but off until attention skips
      masked tiles ([vecq-columns](experiments/vecq-columns/README.md)).
- [x] **F8 Boundaries and mapping language:** a vendor-free
      selected-operation interface, no kernel-source include cycles and a
      stricter boundary check; one mapping value language, one Qwen-Image VAE
      mapping and a shared tensor-name dictionary.
- [x] **F9 Import throughput and leftovers:** one parallel post-write hash
      pass, reuse provenance, unknown-key policy, fuzzing of the newer
      routes, component-declared import transforms.
- [x] **F14 Test-suite audit:** obsolete, disabled and ignored tests deleted;
      redundant unit tests (covered lines and assertions already held by
      another test) merged or removed and input variants made table tests,
      with per-file line and branch coverage of `src/` measured before and
      after and not reduced.
- [x] **F13 follow-ups:** padding head rows bounded on the device in the heads
      component: every greedy argmax reads each row's first tokenizer-size
      logits in place, so padded heads (Qwen3.8, Qwen2.5) keep device greedy
      in joined waves (RE-050).
- [ ] **F10 Hygiene, docs and final checks:** raw experiment dumps and unused
      scripts removed, one test-store resolver, `m35-record.md`, doc drift;
      the cuBLAS RUNPATH and `-fno-sanitize=function` fixes confirmed in the
      `native` and `cpu-asan` tiers with the importer mutation test under
      ASan; optionally a `llmp doctor` warning for storage roles on btrfs
      (RE-018) and pacing the dedicated pool's start-up fill (RE-049); then
      the full Spark check set and a speed confirmation of every family.
      Status (2026-10-29): hygiene, docs, the Spark check set, the IQ2_XXS
      score and the swap table are done; the speed confirmation found the
      regressions of F15 ([final verification](m35-record.md#campaign-final-verification-f10b-2026-10-29)),
      and the workstation `native`/`cpu-asan` tier confirmations are still owed.
- [ ] **F15 Regressions found by the final checks:** Qwen3.8 native + MTP
      decode (drafts accepted 79% → 51%, in `639583b9..381eec18`), DeepSeek
      plain-wave C2 decode (+10%, in `aa92bd1f..381eec18`), Qwen3.8 GGUF C1
      prefill (+1.8%), the Gemma 4 31B assistant screen (+3.6%), and the
      literal-route prompt scores of Qwen3.8 native (PPL +4.8%) and Gemma 4
      (PPL +13–14%) that changed at `e6c3617a` while greedy replies stayed
      byte-identical; each fixed against the `53c2610b` old arm, or accepted
      by the owner as an exception.
- [x] **F12 Known model-test failures and swap continuation:** every model
      test that already failed at `265c821a` (stale frozen digests, a Gemma Q8
      cache admission, Gemma 4 tool rendering, the gemma2 store artifact
      refused at admission, the Qwen3.8 quantized-head tests) is fixed,
      re-recorded with evidence or corrected, every env-gated skip runs by
      default or is justified,
      and the DeepSeek-to-Qwen3.8 8K continuation that differs (15 of 16
      logits) is bisected and fixed.
- [x] **F13 Known open defects:** RE-050 (head wider than the tokenizer),
      RE-052 (start-up guard against page cache), RE-031 (reference top-k
      ties), the M3.6 record's carried lifecycle, planner and test-infra
      defects verified and fixed or scheduled, and the tool-path speed
      follow-ups; no known, fixable defect left recorded as open.

- [x] **Family selection** (approved by the owner 2026-09-29: all 13
      checkpoints of [m35-families.md](m35-families.md), with the answers
      recorded there). The goal is
      capability coverage, not particular models (the owner, 2026-09-29).
      Survey each top-tier open family's current and older widely used
      generations, list every architectural feature they use, and mark
      which llmpalooza already supports in a capability matrix. Then choose
      the smallest set of checkpoints that fit one Spark, each with a
      same-format reference engine, that covers every feature still in
      wide use. A feature only a too-large model uses is flagged. Record each pick's architecture
      class:
      - attention: dense, sliding window with dense global layers,
        compressed and sparse, or linear and recurrent;
      - MoE or dense;
      - positional scheme, normalisation and activations;
      - tokenizer and chat template;
      - MTP or companion drafters;
      - maximum context.

      Named by the owner: Gemma (Gemma 4 26B-A4B, M7's daily driver, and a
      dense Gemma 4), Llama, and MiMo. Already in the repo's plans:
      Ornith 1.5 35B-A3B (M7), Qwen3.8-27B (dense) and Nemotron. GLM-5.3
      Flash stays in M4 (two Sparks). Agents propose; the owner approves
      the list (AGENTS.md: agents never invent supported model
      combinations). Weight licenses are informational (D-087).
      Approved selection: [m35-families.md](m35-families.md).
- [x] **Native token-history accounting and reclaim** (M3 review P2,
      2026-10-04; [accounting](runtime-serving.md#the-chat-route),
      [checks](experiments/m35-token-history/README.md)):
      completed before expanding the model library or request concurrency.
      Native session, retained, snapshot, verification and persistence token
      capacities have explicit ownership and pre-allocation funding, outside
      the fixed uncounted margin. Both M3 LLMs pass exact 8K state/continuation
      controls; small-budget CPU/fake and rounded-catalog GPU regressions pass.
      - Charge actual allocated capacity for `PromptSession::tokens_`,
        `GenerationSession::all_`, `Branch::history_` and
        `Branch::saved_history_`, including temporary copies on admission,
        generation, checkpoint save and restore. Use the existing host
        catalog/request-memory mechanisms with explicit ownership;
        shared storage is charged once and independent copies separately.
      - Acquire capacity before allocation or growth. Refusal leaves the
        completed prefix usable and starts no unfunded native work.
        Reclaim must protect active and held continuation histories and
        avoid recursively reclaiming the branch being admitted.
      - Reclaim idle histories with their retained state. Clear, forget,
        discard, expiry and snapshot invalidation release obsolete vector
        capacity and its charge once no session, continuation or valid
        snapshot needs it. A spilled continuation keeps the tokens needed
        for exact restore.
        Charges survive every transfer of ownership and appear in the
        memory breakdown, outside the fixed uncounted margin.
      - Add small-budget regressions for repeated long requests across a
        growing model library, concurrent sessions, cancellation, expiry,
        snapshot save/restore and model-switch continuations. Check actual
        vector capacity against charges, clean refusal and release after
        retirement, including CPU/fake coverage of the accounting and
        Spark checks of exact continuations on both M3 LLMs.
- [x] **llama.cpp v0.6.0 reference and native GGML refresh** (owner,
      2026-10-05): exact release `d812350` and ARM64 image, with
      [reference inspection/helper prerequisites](experiments/llama-reference-refresh/README.md)
      and [native GGML 0.26.0 integration](experiments/ggml-release-refresh/README.md).
      Owned dispatch, compact expert worklists and IQ2 occupancy bridge remain;
      attention launch and FP4 precision interfaces follow the new source.
      Focused operators and selected build/source closure pass on a GB10.
      Historical model results retain their original pins; fresh whole-model
      quality/performance qualification and native Clef support remain open.
- [x] **Fresh v0.6.0 Gemma31 C2 representative screen**:
      [own-first complete-head/state controls](experiments/gemma-release-c2/README.md),
      actual initialized CUDA/driver/cache admission, zero strict choices/ties
      and 64-target conditional-loss PASS. Short native elapsed is 1.71513% slower;
      scalar, other cohorts, 26B transfer, context and sustained parity remain open.
- [x] **Representative near-8K Gemma model switches**
      ([checks](experiments/gemma-near8k-swaps/README.md)): Gemma2 and both Gemma4
      profiles save 8063 tokens at context 8192 and repeat 64 continuation heads/tokens
      exactly after Qwen3.8 switches. All six actual saved-state return snapshots
      match SHA-256, and prepared returns retain/replay graphs. Worst prepared
      handoffs are 5.911946/6.200046/6.294532 s for Gemma2/26/31. Snapshot budget and
      sampled whole-node MemAvailable decreases remain distinct. These are
      near-8K repeated-text boundary controls, not corpus/retrieval or maximum-context
      qualification; broader pairs/cohorts and sustained gates remain open.
- [x] **Fresh v0.6.0 Gemma26 C2 representative transfer**:
      [native-own before FIRST stock and full bookend identity controls](experiments/gemma-release-c2-26/README.md),
      zero strict choices/ties and 64-target conditional-loss PASS with the
      actual all30 routing recipe. Short native elapsed is 2.153583% slower;
      scalar, other cohorts, context and sustained production parity remain open.
- [x] **Private ordinary MMVQ preparation-sharing operator screen**:
      [dense31 C2 Q4_K gate/up and Q6_K down](experiments/gemma31-mmvq-shared-prep/README.md)
      retain complete synthetic chain bytes under eager/changed-input capture
      and atomic refusal controls. One original input preparation is removed;
      the short mean improves 1.0361% with overlapping timing ranges. No model
      benefit, production path or Gemma26 transfer is selected.
- [ ] **Pager improvements** (owner, 2026-10-07): the costs every model
      pays in swaps and cold state growth.
      - [x] [Lazy handoff](experiments/vmm-batching/README.md) (D-033
        amended): a swap's evicted backing stays mapped until the incoming
        loads take it, so the unmaps run beside page-in reads; DeepSeek ↔
        Qwen3.8 swaps fall from 7.9–9.7 to 6.1–8.4 s, exact. VMM batching
        and larger extents measured and not adopted.
      - [x] [Internal shared partial-weight foundation](experiments/partial-weight-eviction/README.md):
        whole state spill, global GreedyDual deficit selection, mixed backing,
        missing-only reload and exact cache-preserving recovery; D-055/D-096
        amended. Shared wake/harvest and donor identity work are qualified.
      - [ ] Adopt partial eviction for ordinary serving only after its
        never-worse-than-full cold-switch gate passes. The final current-source
        factor saves 18.3% prepared first-output time and 19.5% read bytes,
        but cold first-arrival is 11.9% slower; ordinary default stays full.
      - [x] [Handle reserve and device zero-fill](experiments/vmm-batching/README.md#handle-reserve-and-device-zero-fill)
        (D-033 amended): 32 device handles created while the VMM lane is
        idle, charged as one pinned extent, and new state zeroed on the
        GPU; Gemma3's 131K cold state growth falls 38% (0.47 to 0.29 s),
        exact, now at or slightly ahead of stock; swaps unchanged.
- [x] **Retroactive optimization transfer inventory audit** (owner,
      2026-10-07; [complete dispositions](optimization-inventory.md#complete-cross-family-dispositions--2026-10-07)):
      every historical table technique and material later kernel, fusion,
      graph, serving and pager mechanism has explicit current-family
      dispositions, including native/GGUF Qwen and image T/D/V boundaries.
      This closes recording coverage, not the open transfers below. Every
      eligible port is M3.5 work, including families M3 closed out; none waits
      for M9. Screen one representative shape with focused correctness and
      matched paid A/B before broader qualification. *The ports were retired
      on 2026-10-08 (D-107; the retired item below).*
- [x] **DSpark noncausal block GPU-mask transfer screen** ([evidence](experiments/dspark-device-masks/README.md)):
      shared F16/F32 producer/source controls and actual C2 draft/verify
      heads, histories and states are exact. The short joined-call factor is
      neutral (−0.108%, overlapping ranges), whole diagnostic +0.224%; keep
      host masks selected. This closes only the block sub-investigation as
      not adopted; compressed visible-count masks keep composite T22/T69 open.
- [ ] **Publish the greedy prompt-completion token on device.** Extend the
      shared first-token contract across all LLM families, including injected
      prefill, retained assistant features, catch-up/export/carry and pending
      rows; preserve requested-score/sampling fallbacks and exact state/head
      controls. This is a compatible extension of decode publication, not
      completed T27 decode work or a new public capability.
- [ ] **Reduce remaining repeated Setup graph construction/selection cost.**
      Preserve exact configured funding and shape/selector distinctions; the
      selected T96 placement shortcut does not remove graph sizing/building.
      Qualify equivalent descriptor/template reuse before dropping any probes.
- [ ] **Reference-free quality tripwires** (owner, 2026-10-20, from
      [DeepInfer](deepinfer-assessment.md#9-reference-free-quality-tripwires)),
      shared tooling that every served family runs unchanged: a
      decode-against-prefill check (teacher-force about 1K tokens through
      decode, speculation included, rescore by chunked prefill, and bound
      the KL and top-1 disagreement, the bound declared from a measured
      baseline); a PIT/KS uniformity check of the sampler on live rollouts
      beside the 256-seed total-variation check; and a debug-build
      tripwire in the CUDA provider that fails a blocking driver call
      (stream, event or context synchronize, synchronous copy) inside a
      step's submit region. No default changes. From
      [Flare](flare-assessment.md#10-poison-on-remap-diagnostic) (owner,
      2026-10-20): the bounds use full-vocabulary KL, top-1 margin (a flip
      under the measured noise margin is benign) and top-10 overlap, with
      a three-way noise floor (reference against canonical) as their
      measured basis; a diagnostic provider mode fills discardable
      extents with a poison pattern on every (re)map and the swap, wave
      and verify model tests run under it (the RE-058 class, where a value
      prepared once in shared scratch is lost on re-backing); and
      concurrent lanes (T15) get delay-injection stress.
- [ ] **Small serving items from Flare** (owner, 2026-10-20,
      [assessment](flare-assessment.md#11-smaller-items)):
      - sampling defaults recorded at import from the model card for
        artifacts whose checkpoint carries none. The original Qwen3.8 NVFP4
        example artifact omitted its pinned `generation_config.json`; the
        [U3 re-import](experiments/m35-generic-import/original-parity.md)
        already retains temperature 1, top-k 20 and top-p 0.95 from that file.
        The card fallback remains open for sources without structured defaults,
        since agent clients omit them; temperature 0 stays greedy and
        requests are never clamped;
      - XXH3 (BSD-2, source-locked) instead of SHA-256 for spill and
        checkpoint integrity, hashed inline with the write (owner-only
        files, torn or corrupt data, not an adversary; artifacts keep
        SHA-256 under D-054), amending D-105's record format when it lands;
      - registered io_uring buffers on the host-VMM landing zone, and
        bounded batched copies on a low-priority stream for spills.
- [x] **Seven-day retention default** (owner, 2026-10-20; D-055 as
      amended): idle conversations and turn checkpoints survive an idle
      weekend. Move `[memory] retention_hours`' default from 24 to 168
      (`kDefaultRetentionHours` in `src/config/node_config.h`, the
      `kTurnCheckpointRetention` constant and its comment in
      `src/runtime/turn_reuse.h` (the checkpoint cache's own default, which
      `tests/unit/turn_reuse_test.cc` reads), `packaging/llmp.example.toml`
      and the default asserted in `tests/unit/config_test.cc`) and the
      runtime docs that state 24 (runtime-serving.md, engine.md, D-055's
      configuration bullet). The
      spill budget (`spill_budget_gib`, least recently used first) stays
      the binding limit. The [seven-day capacity calculation](retention-policy.md#bounds-and-defaults)
      retains its 128 GiB default: measured 451/712 MiB state footprints
      give scenarios of about 41/26 new idle states per day over seven
      days, or 20/13 with one equal-sized checkpoint each. These are
      capacity calculations, not measured weekly demand or guarantees for
      other contexts. A fixed-clock checkpoint regression retains a
      boundary after a 72-hour idle weekend and expires it at 168 hours.
- [ ] **Tool calls on the chat route** (moved from M5 by the owner,
      2026-10-20): **Interpreter foundation complete (2026-10-22):** one
      generated-output interpreter uses verified template syntax data for
      Qwen, DSML and Gemma, with owned definitions, resource accounting bounds,
      complete-call events and Flare's malformed-call controls. Whole and
      adversarial reviews are clean; native focused and integrated shared and
      no-CUDA checks pass. **Canonical tokenizer foundation complete
      (2026-10-22):** exact canonical span planning uses actual EOF-sensitive
      tokenizer blocks, holds unstable tails and funds owned results. Whole
      and adversarial reviews are clean. Spark B passes 179 unique targeted
      shared checks across the initial and fixture-repair runs, including all
      17 span controls, and 24 model-tokenizer checks without skips. The native
      route enablement spike and representative proof are complete
      (2026-10-22): six frozen-identity Qwen API tool/result transactions pass
      across auto, required and named choice, each in JSON and SSE, and pinned
      OpenCode 1.19.3 completes one seeded edit-and-test task with four actual
      tools; the initial seeded tests fail and the independent post-edit run
      passes all five without changing the tests. See the
      [results](experiments/m35-tools/RESULTS.md). This evidence does not close
      the item: qualification across remaining served aliases and template
      grammars remains open; final integrated source checks are separate. No
      performance, maximum-context, sampled-quality, or broad speculative coverage
      follows from these diagnostics. See [tool-calls.md](tool-calls.md).
      The first native mixed-encoding Mia named-tool diagnostic failed with
      an incomplete call; persisted history showed an alternate tagged header
      and call close. Declared structural aliases, terminal eligibility and
      legacy grammar-marker route-stop controls address that failure in the
      shared engine. The same native weights now pass one named-tool/result
      transaction, including the streamed result follow-up. The original sampled
      MTP attempt failed at architecture admission before inference; its inference
      controls remain **NOTRUN**. After canonical mapping-identity repair, fresh
      separate sampled-MTP required-tool and named-tool cells pass with actual
      drafting, marker reads and streamed follow-ups; see the
      [qualification evidence](experiments/m35-generic-import/original-parity.md#drafter-source-alias-admission-repair).
      These two representative cells leave broader alias/grammar and speculative
      tool qualification open, and the original failures are retained.
      `tools`/`functions` and `tool_choice` (`auto`, `none`,
      `required`, a named function) accepted on Chat Completions, assistant
      `tool_calls` returned, with streamed argument fragments and tool-result
      messages rendered back by D-067's renderers. The parser is one engine
      driven by each chat format's data (its tool-call begin and end
      sequences, from the renderer and tokenizer), in the trigger shape
      M7's constrained output reuses: free text until the begin sequence,
      and a bound token that forces it for `tool_choice: required` or a
      named function; no per-model parser code
      ([assessment](deepinfer-assessment.md#8-tool-calls-and-structured-output)).
      The forced begin sequence is fed in the same step as the token
      that selects it, as the tokenizer's canonical encoding of the forced
      bytes (checked to reproduce them, with an unstable last block held
      back), so it costs about one weight read and never feeds a token
      sequence the tokenizer would not produce
      ([Locals](locals-assessment.md#2-canonical-forced-spans-jump-forward)).
      Each format's data also declares, after
      [Flare](flare-assessment.md#5-tool-calls-and-the-messages-profile)
      (owner, 2026-10-20), the per-call begin and end, where the name sits,
      the argument encoding (a JSON object, or a tagged-parameter pattern
      with its value-trimming rule: Qwen's `<parameter=…>`, GLM's key/value
      tags, DeepSeek's DSML) and whether the tool opener ends reasoning;
      one interpreter consumes it. Each call streams whole, as one
      argument fragment, once it validates, so no incremental argument
      parser is needed.
      Flare's edge cases are the test suite: an unclosed parameter closed
      by the next boundary, omitted outer tags, one wrapping newline
      trimmed per side, schema-guided scalar conversion when not strict,
      undeclared names returned in auto mode, a truncated call dropped
      with a length finish, parallel tool results reordered into call
      order before rendering, and calls without IDs tolerated.
      Qwen's tool-call and thinking markers are user-defined tokens, which
      match in message content for reference parity, unlike control tokens
      (tokenizer.md); the parser reads only generated output, never rendered
      input, and the item records whether content forging a marker needs a
      renderer-side guard (Flare has none: its `rejected_tokens` refuses only
      media placeholders a package cannot serve). Every served LLM whose template declares tools
      (lifting Gemma 4's refusal of tool declarations, runtime-serving.md);
      checked with OpenCode, the named client on this route. The
      Responses route, and with it Codex, stays M5's; Messages moves with
      the next item; strict tool arguments stay M7's constrained output.
- [ ] **Messages route and the Claude Code profile** (moved from M5 by the
      owner, 2026-10-20), on the same front door and the tool-call engine
      above, as [client-api-baseline.md](client-api-baseline.md#messages)
      specifies them:
      - `POST /v1/messages` (beta query handled independently of routing)
        with system blocks kept apart from roles, `tool_use` and
        `tool_result` blocks linked by `tool_use_id`, stop reasons and
        truthful usage, and its SSE events (`message_start` on admission,
        indexed content blocks, `input_json_delta` tool inputs, `ping`
        during long silences including model switches, in-stream errors);
      - `POST /v1/messages/count_tokens` from the model's own tokenizer and
        rendering, with no decode or residency;
      - `/v1/models` and `/v1/models/{id}` in the Anthropic list shape
        (chosen by `anthropic-version` or `x-api-key`) for Claude Code's
        opt-in discovery, with `display_name` and `description` (both
        shapes stay M5's in full, with D-046's metadata);
      - the Claude Code profile: bearer or `x-api-key` credentials accepted
        and ignored on the anonymous listener (the optional API key stays
        M5's), `anthropic-version` and bounded beta headers, and each
        default field's handling: `thinking` rejected with the documented
        400 naming it on every model without a reasoning contract (D-043;
        all, until M7's); `context_management` and the beta tool-schema
        fields disabled in the pinned setup
        (`CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`), since the client does
        not retry their rejection, unless a captured request shows
        `context_management` is only `clear_thinking_20251015` with
        `keep: "all"`, which is accepted as Flare does and drops the
        switch; `output_config` effort mapped onto the template's effort
        levels where it takes them (found by trial render) and otherwise
        rejected naming the field; a mid-conversation system message the
        template would drop rendered as a delimited user turn (trial
        render); the prompt-too-long 400; `cache_control` advisory; the
        attribution block stripped when it stands alone as the first
        `system` entry, so shared-prefix identity excludes it; the hint
        headers tolerated, default policy until M6's release semantics;
        background requests aliased to the main model; `image` blocks with
        the media-input item;
      - the protocol's error shapes and wording with `retry-after` and
        `x-should-retry` (D-045), inside D-097's listener and intake
        bounds.

      Done when a pinned Claude Code build, with the pinned profile's
      settings, completes an agentic session with tool use against a served
      LLM, in the shape of Flare's coding-client test (isolated settings
      sources, an empty strict MCP config and a fixed tool list, normal and
      `--bare`): it fixes a seeded bug in a temporary repository, the test
      passes, the test file is untouched and a tool call ran; OpenCode runs
      the same task on the chat route
      ([Flare](flare-assessment.md#5-tool-calls-and-the-messages-profile)). M5 keeps the Responses
      route, the rest of D-045's front-door rules and its acceptance, which
      re-verifies Messages.
- [ ] **Shared-prefix reuse across conversations, minimal** (moved from
      M6 by the owner, 2026-10-20): a new conversation whose rendered
      tokens begin with a retained boundary's (the renderer-reported end of
      the leading system/tool segment, D-055's declared boundary, saved like
      a turn checkpoint) starts from a copy of that state and prefills only
      the rest. Generic over the state contract: whatever a turn checkpoint
      holds (full and compressed KV, windows, recurrent state, compressor
      carries, drafter state) is what is copied. Exact: identity as D-055
      (same artifact, state-producing plan, rendered tokens and caller
      scope). As D-031 requires, the retained boundary is its own entry
      with its own reuse count and expiry (`T_prefix`), neither a
      conversation's identity nor its lifetime, and a hit refreshes only it.
      First measure, with OpenCode and recorded Claude Code and Codex
      prompts, how long the shared prefixes are and whether they stay
      byte-stable across conversations (dates or repository status inside a
      system prompt break reuse); build only if the hit rate pays. At
      DeepSeek's measured ~460 tok/s prefill a hit saves about 2 s per 1K
      tokens. Block sharing, reference counting and the full retention
      policy stay M6's
      ([assessment](deepinfer-assessment.md#7-shared-prefix-cache-generic-over-state-contracts)).
      Keys and boundaries follow
      [Flare](flare-assessment.md#4-prefix-keys-and-boundaries-minimal-prefix-reuse)
      (owner, 2026-10-20): chained 128-bit hashes over 64-token blocks,
      independent of any contract's block grain and domain-separated by
      D-055's whole identity above, not model identity alone as in Flare;
      mid-block endpoints keyed exactly (parent digest,
      tail tokens, count); per-block buckets so lookup costs about one
      probe per 64 new tokens; conditioning inputs hashed at their
      positions; one shared transfer for concurrent waiters. Boundaries
      come from rendering each message prefix and keeping it only when it
      is a strict token prefix of the full prompt, which works for every
      template and locates D-055's declared boundary without per-template
      rules (cross-conversation byte stability is still measured as above). A hit
      check predicts how many prefill launches the hit leaves and compares
      the runtime's counters.
      Two more measurements first, from
      [Locals](locals-assessment.md#3-reply-boundary-checkpoints-for-recurrent-state)
      (owner, 2026-10-22), for stateless clients on recurrent models
      (Qwen3.8): whether a client's next request hits the retained
      continuation when the chat template renders an earlier reply
      differently from the open turn (Qwen3.5's drops the empty think block;
      check Qwen3.8's) and when the model's own reply is re-tokenized from
      resent text. [First bounded Qwen3.8 checks](experiments/m35-prefix-reuse/README.md)
      (2026-10-22) retain all processed reply tokens when reasoning is resent;
      omitting reasoning changes the prompt and recovers the earlier turn
      checkpoint. Actual OpenCode mock captures preserve reasoning in history.
      A seeded native OpenCode v1.19.3 task adds three bounded continuation
      observations: cached tokens equal each prior prompt plus generated count
      minus the final unprocessed anchor (4,853, 5,299, 5,396); its next
      prompts add 327, 20 and 33 tokens. See the
      [native continuation results](experiments/m35-prefix-reuse/README.md#actual-native-opencode-continuation).
      This is one task, not general native-client, long-prompt or cross-model
      qualification; broader reuse remains open and these cells select no cache
      change. If either misses, the renderer reports a second
      boundary per turn, the position where the next request's rendering of
      this reply diverges, found by differential rendering (two renderings
      with different marker strings, first differing position; no
      per-template rule), the state there is checkpointed (recurrent and
      window state only, with views of the KV), and the tokens of the
      model's own reply are retained with the continuation. Moving volatile
      per-session blocks (date, folder) to the end of a system prompt is
      considered only if the byte-stability measurement above shows they
      cause the misses; it changes what the model reads, so an opt-in
      client-profile mode, off by default (rule 11).
- [ ] **Checkpoints off the turn path, measure first** (owner,
      2026-10-20, from [Flare](flare-assessment.md#9-checkpoints-off-the-turn-path)):
      turn checkpoints are captured and restored synchronously on the turn
      path at queue depth 1 through one 2 MiB staging buffer
      (`src/engine/checkpoint_file.cc`), and every 2 MiB extent holding a
      mutable byte is captured whole. Measure capture time alone and
      mutable against captured bytes on Qwen3.8 and DeepSeek; then capture
      by device copy into a cataloged, charged snapshot extent published
      after a fence and written behind at queue depth above 1, pipeline
      restores (read N+1 while copying N), and capture only the state
      contract's mutable byte ranges. The prefix reuse above copies with
      the same machinery.
- [ ] **Token-keyed row read-ahead, measure first** (moved from M4 by the
      owner, 2026-10-20): measure the share of Qwen3.8's prefill and
      decode step time spent in its synchronous n-gram row reads
      (`src/engine/ple_rows.h`), and if it is material, build the pager
      capability M4's Engram then inherits: a mapping declares a row table
      whose indices depend only on token IDs; the pager computes indices
      at intake, reads rows ahead of the first consuming layer, overlapping
      earlier layers and, in prefill, the previous chunk (in decode a step's
      rows depend on the previous token, so only that overlap is possible);
      a read-ahead landing and its row slots are charged whole while in
      flight (D-006, D-007), as today's per-chunk ones are
      ([assessment](deepinfer-assessment.md#4-token-keyed-row-read-ahead)).
      The key function is also evaluable in the graph, after
      [Flare](flare-assessment.md#2-device-keyed-row-delivery-the-token-keyed-row-read-ahead)
      (owner, 2026-10-20), for decode and drafted rows whose keys the host
      does not know: the graph writes row IDs and a sequence number to a
      host-mapped mailbox, the pager's reader polls it (RE-017), issues
      direct reads into the landing (no CPU copy, D-081) and raises a
      flag, and the gather runs behind an event wait at the first
      consuming layer. Host keys keep cross-chunk read-ahead in prefill.
      A row cache only if the measurement shows reuse, as a reclaimable
      catalog entry priced at its re-read cost.
- [x] **Dependency audits for adopted DeepInfer items** (owner,
      2026-10-20), source and license only, no integration:
      - [x] **xgrammar** (structured output for M7's constrained output and
        the tool-call trigger shape): pinned its revision, audited the
        selected C++ core license/dependency closure, and recorded that the
        core fails to compile with `-fno-exceptions` in
        [the source audit](xgrammar-audit.md); no code was integrated. Any
        future use needs a reviewed D-066-compatible error boundary or
        source adaptation;
      - [x] **TileKernels** (`deepseek-ai/TileKernels`, the Engram hash and
        gate kernels V4.1 needs): pinned its revision and license and recorded
        the sourced formulas in [the source audit](tilekernels-audit.md); no
        integration was done.
- [ ] **NVFP4 scale compression (CSF) in the importer** (owner,
      2026-10-20), for [Qwen3.8 FP4-Local](m35-families.md#owner-requested-addition-qwen38-flash-next-fp4-local):
      Flare describes a lossless NVFP4 scale compression, CSF
      (`csf_32x2_v1` and the slab variant), and ships an independent
      reference decoder
      ([assessment](flare-assessment.md#1-nvfp4-scale-compression-csf-for-fp4-local));
      that it is the card's NVIDIA format is unconfirmed. Confirm the
      FP4-Local tensors carry its names (otherwise identify the format from
      ModelOpt's source), then decode the
      encoded scale planes in the format layer into the native scale-atom
      layout at import (AGENTS.md rule 11), checked byte for byte against
      the reference decoder. Reused by M4's GLM FP4-Local. If FP4-Local
      is not CSF, the Flare Qwen3.8 pin of the comparator item below is
      the byte-for-byte test input.
- [ ] **Flare's pinned checkpoints as same-checkpoint comparators**
      (owner, 2026-10-20): archive the exact revisions Flare `e7b8c01`
      pins (D-114; its recipes hash-pin them, so they are the only inputs
      it takes) for every one that fits our hosts, and run each on both
      engines: llmpalooza imports and serves it (every quant runs,
      AGENTS.md rule 11), which also tests the format layer's flexibility
      (CSF-compressed NVFP4 and MXFP4 scales, MXFP8, tensor-scaled and
      block-128 FP8, MXFP4 experts, FP8 indexers), and Flare, pinned under
      [reference comparisons](reference-comparisons.md), gives the
      same-checkpoint speed and memory comparison, its default quality
      trades (FP8 KV, GLM's 4-bit prefill activations) stated or matched
      ([assessment](flare-assessment.md#as-a-comparator)):

      | Checkpoint (Flare's pin) | Hosts | Milestone |
      | --- | --- | --- |
      | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD@dc5b8ab2fa14bc4fecd6886273b7ebc6095662e2` | one Spark | M3.5 |
      | `nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80` | one Spark | M3.5 |
      | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB@c6cc2e8ffa353403213d6b9b18aaf8eb684d705a`, and for its speculative runs the LiLiCorr drafter `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr@abd478065678ee83a93821c4b3f512da24e4cecb` | one Spark | M3.5, with Nemotron-H's components |
      | `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd33efa73c3bb7c95b74eee10cad34f9be` | two Sparks | M4 |
      | `local-inference-lab/DeepSeek-V4-Flash-Vision-Exp-lossless-CSF@45fd997626821b131a904b1ae2304bb673c9458d` | two Sparks | M4 |

      The one-Spark rows run in M3.5 as their families land; the
      two-Spark rows join M4's comparisons. Flare's other pins (DeepSeek
      V4.1 at TP4, Kimi K3, MiMo V2.6 Pro, Step-5) need four or more
      Sparks and are not archived for this. Where a pin is also the
      FP4-Local weights, it doubles as that import's same-checkpoint
      reference.
- [ ] **Humming screen** (owner, 2026-10-21; vLLM's Humming `5b500b4`,
      Apache-2.0, as a build-time kernel source under D-053,
      [assessment](humming-assessment.md)): a coarse D-085 screen before any
      integration, with its thresholds written down before the run (rule 3).
      Pin the revision as a screening-only entry in the source lock; the
      harness lives in `benchmarks/` and instantiates Humming's headers with
      NVCC, never its Python launcher, on a Spark (`sm_121a`):
      - **NVFP4 W4A4 grouped GEMM** against the CUTLASS 79d kernel
        (`moe_cutlass.cu`) at Qwen3.8's prefill shapes and routed row
        counts, BF16 output, the routed working set rotated (cold L2), with
        documented PTX formats only (cubin patching off);
      - **MXFP4 expert products** (W4A8 and W4A4 over relaid scales) at
        DSpark's decode and verify row counts against GGML's MXFP4 path,
        the owed MXFP4/MXFP8 products' first candidate;
      - **one W4A16 grouped decode comparison** (1 to 16 rows) against
        `llmp.decode.grouped.nvfp4.scale_atoms`.
      Each result checks the operands against the CPU references. An
      operator win of 10% or more is followed by one model-level A/B, since
      operator wins here have not always survived in the model; if Humming
      loses or only ties, record that and stop. A win leads to a
      per-shape-class instance in the registry with checked-in rank data
      (U6), an import-time relayout recorded in the artifact (D-108, D-113)
      and a provenance check of each file used (rule 12). `sm_86` coverage
      goes to U7d only after Humming builds with our toolchain. The hidden
      E0M3/E3M4 formats (cubin patching) and FP8 MoE activations are leads,
      not part of this screen
      ([assessment](humming-assessment.md#hidden-tensor-core-formats-a-research-lead)).
- [x] **Scalar DS/Qwen nonfinal state-only dependency cuts**
      ([evidence](experiments/ds-qwen-state-only/README.md)): ordinary plain DS and
      native/GGUF Qwen select checked cuts; DSpark/MTP head-only opt-ins remain
      unselected. Exact initialized state, full heads, checkpoints/scoring and
      continuations pass. Explicit both-intent Setup funding has unchanged maxima
      and added construction work; only DS/native Qwen have n=2 speed factors.

**Owner stop/handoff (2026-10-08):** complete only grouped physical KV stores
(7520f12) and the T54 batch above, then STOP. The inventory is recorded, but
128 family cells across 47 techniques remained OPEN (retired to M3.6 by D-107,
below). This transfer run stops here
and starts no further batch, phase, general refactoring or duplication cleanup.
A separate agent may refactor under its own assignment.
Focused checks are recorded; full regression/shipment tiers remain deferred.
Keep milestone raw stores, warm source/build trees and standing replay inputs:
M3.5 is not closed. [Concrete handoff](experiments/ds-qwen-state-only/README.md#owner-stop-and-handoff).

- [x] **Open optimization transfers from the audit: retired** (D-107,
      owner, 2026-10-08). At retirement the item held 128 open recipient
      cells across 47 techniques, matching the frozen
      [inventory matrix](optimization-inventory.md#complete-cross-family-dispositions--2026-10-07)
      cell for cell. Each technique now lands once in M3.6's shared engine;
      [engine-components.md](engine-components.md#retired-transfer-items)
      maps every open technique to where it lands, and lists the kernel-level
      ones that may continue meanwhile. The per-recipient history stays in
      Git (the item as of `cc70d69`).
- [ ] **Current TensorFold quality lead:** the pinned FP8-prompt p3
      Teacher32 comparison agrees 31/32; row 29 reference margin 1.5625
      exceeds the unchanged 1.0 bound. Native host/device-mask heads are
      exact. The completed same-history BF16-prompt isolation moves row29's
      reference margin to 0.6875 without recovering native's argmax, and
      introduces a strong posterior change at row5. The completed 1565-ID
      all-prompt comparison also retains the argmax disagreement (native 17723,
      TF BF16 47149; reference margin 1.3125). Further arithmetic
      qualification remains open; original FP8 quality finding unchanged,
      no parity or PPL pass. [Controlled diagnostic](experiments/qwen-device-masks/README.md#controlled-bf16-prompt-follow-up).
- [x] Isolate pinned TensorFold FP8/BF16 prompt policy with one untimed,
      same-conditioned Teacher32 and actual-consumer/retirement proof.
- [x] Isolate row 29 prompt/serial geometry with one same-conditioned 1565-ID
      native/BF16 TensorFold frontier, complete finite heads and authenticated
      consumers/retirement. Argmax remains different; the quality lead stays open.
- [x] Implement and qualify an optional same-47,172-ID Q4_1 selected MTP
      head, including odd-prefix scalar/joined OneToken capture/replay,
      proposals/confidence, initialized state and protected-peer restore.
      [Results](experiments/qwen38-quantized-draft-head/README.md): C1 generation
      −4.064%, C2 +1.482% versus BF16; BF16 remains ordinary. Fresh public TF
      C1 generation is −1.999%, n=2, with the target-quality lead above unchanged.
- [ ] **Selected draft-head performance limitation:** C2 Q4 generation
      regresses 1.482% against BF16 in the bounded format factor. Own-Q4
      draft sharing reduces its draft interval 5.049%, but whole generation
      −0.143% is inside +0.229% off-bookend movement. Preserve the optional
      artifact/default distinction: the earlier ordinary BF16 comparison still
      takes 3.237% more time than public TF. Wider context/cohort and production
      request qualification remain open. This does not establish that all
      remaining Qwen/TF performance gaps are closed.
- [ ] **Per family**, on the engine skeleton, using the "adding a model
      family" guide, which M3.5 tests and corrects:
      - [x] Gemma 4 26B-A4B and 31B architecture foundation:
        [verified profiles/bindings](gemma4.md), bounded state and independent
        request-segment host inputs, used by the bounded native runner and scalar
        serving route. Full model and optimized-batching qualification remain owed.
      - [x] Gemma GELU-tanh and split GeGLU primitive fallbacks,
        with separately gated floating MMVF fusion and a
        [bounded synthetic screen](experiments/gemma-activations/README.md).
        Quantized writers and model/optimized-batching qualification remain owed.
      - [x] Gemma D256/F16 local attention primitives, with checked
        GQA2 vector/MMA selection, independent ring masks and funded scratch;
        [pinned primitive controls](experiments/gemma-local-attention/README.md).
        Whole-model quality, performance and optimized batching remain owed.

      - [x] Gemma segmented GGML text graphs and checked bound plans, with
        complete-layer, independent-slot and captured-replay diagnostic controls;
        optimized serving and whole-model/reference qualification remain owed.
      - [x] Gemma graph-owned causal/ring device masks, with funded host
        reference inputs and [exact mask/replay controls](experiments/gemma-device-masks/README.md).
        Optimized serving and reference gates remain owed.
      - [x] Checked standalone Gemma MoE routing/scaled-reduction primitives
        and [structural matchers](gemma-moe-matchers.md), preserving selected
        IDs, full-sort backing and ordered expert scaling.
      - [x] [Default-off native Gemma MoE dispatch](experiments/gemma-native-moe/README.md),
        with unchanged ordinary graph-order control, complete-root placement,
        initialized state/replay and primitive keep fallback. Compound quality
        still has nine out-of-noise greedy differences; production selection,
        whole-model reference qualification and optimized batching remain owed.
      - [x] [Checked native Gemma norm/rotation and norm/residual chains](experiments/gemma-native-norm/README.md),
        with keep/view fallback, paid final residual gather and shared state/capture
        controls. Dense31's 128-row full heads match stock; policies remain off,
        paid 8K differences and full reference/context/batching gates stay open.
      - [x] Bounded native Gemma 26B-A4B runner with independent slots,
        device masks and [exact state/replay controls](experiments/gemma-runner/README.md).
        Full serving qualification, assistant, representative quality, long-context and optimized
        batching/performance qualification remain owed.
      - [x] Dense Gemma 31B on that shared runner, with approved-profile
        selection, cross-variant restore guards and ordinary 1/2/4-request
        [state and reference controls](experiments/gemma31-runner/README.md).
        Representative PPL is 14.61% above full-fusion llama.cpp; the unfused
        diagnostic matches all full heads. Full serving qualification, assistants, optimized
        batching and full reference/context qualification remain owed.
      - [x] Checked per-segment Gemma RoPE/cache-store policy with primitive
        fallback and [complete-layer controls](experiments/gemma-rope-store/README.md).
        Mixed paid results keep selection off by default.
      - [x] Bounded dense31 production recipe at context<=8192/slots<=4:
        [current serving bridge](experiments/gemma31-serving-bridge/README.md)
        completes C1/C4 8K quality, exact current-pin 1K corpus, full-cost bookends
        and HTTP stop/continuation/departed-client peer progress. Default prefill
        fallback 256, both norm chains, eligible owner attention and (since
        [gap closing](experiments/gemma-gap-closing/README.md)) the fused decode
        FFN; short paid cycles are 0.73%/1.36% slower than stock.
      - [x] [Bounded Gemma26 production recipe](experiments/gemma26-production/README.md)
        at context<=8192/slots<=4: 1024-row prefill fallback, norm chains, MoE
        route/reduce and joined owner attention. C1/C4 within 0.6% of stock with
        stock's exact greedy tokens; corpus heads byte-identical. Larger settings
        retain their prior recipe. Broader quality/context, sustained performance and assistants
        remain open.
      - [x] [Current corrected Gemma4 reference](experiments/gemma-current-backend-greedy/README.md):
        ordinary 8K C1/C4 recipes versus original d812 backend greedy with normal
        cache/reset policy; all 1,290 complete heads byte-exact and natural
        histories exact. Quality precedes short n=2 paid token-path bookends;
        stock sampled-logit transfers remain explicit. Sustained performance,
        peak memory and actual HTTP per-unit overhead remain separate gates.
      - [x] Bounded approved Gemma26/31 adapter on the shared serving driver,
        with [scalar independent cohorts](experiments/gemma31-serving/README.md),
        target likelihoods, checked checkpoint positions, exact continuation/
        restart and pending cross-profile switch controls. Thinking/tools, assistants,
        broader optimized joining and full model qualification remain owed.
      - [x] Bounded Gemma26 same-format reference diagnosis: the
        [short resident comparison](experiments/gemma-performance/README.md)
        [representative likelihood screen](experiments/gemma-quality/README.md)
        and [paid 8K screen](experiments/gemma-prefill/README.md).
        The short scalar rate is 4.33% below the bookended reference, and
        representative PPL is 10.03% higher than fusion-enabled llama.cpp.
        Paid 8K prefill takes 6.931 s versus the fastest screened reference
        at 2.612 / 2.605 s, while native fixed-prefix decode is faster.
        These failures keep model qualification open; no optional policy is selected.
      - [x] [Larger-row Gemma prefill diagnosis](experiments/gemma-prefill-large/README.md),
        with authenticated independent state/head repeats and matched reference
        bookends. Dense31's 256-row prefill is within reference timing movement;
        all screened policies retain strict token differences. Production caps,
        selected optimizations and whole-model qualification remain unchanged.
      - [x] Explicit default-off [joined serving controls](experiments/gemma-joined-serving/README.md),
        with C1/2/4/8/12 own head/state controls and historical ordered8+4 C12 subwaves,
        independent refusal/publication and actual HTTP continuation evidence.
        Real multi-sequence reference quality fails on natural prefixes;
        optimized-batching selection and full model qualification remain open.
      - [x] [Dense31 natural C4 norm/row first screen](experiments/gemma-joined-norm/README.md),
        with exact native scalar/joined repeats and frozen candidate heads.
        Fresh stock quality fails on 33/128 heads and latency is 1.7523% higher;
        no candidate is selected and full optimized-batching qualification remains open.
      - [x] [Dense31 ordinary-product norm C4 first screen](experiments/gemma-joined-norm-ordinary/README.md),
        with frozen native repeats and fresh recipe-aligned stock bookends.
        Quality fails on 15/128 heads and latency is 1.1161% higher;
        no candidate is selected and full optimized-batching qualification remains open.
      - [x] [Dense31 identical-operand C4 attention diagnosis](experiments/gemma-attention-c4/README.md),
        with exact capture controls, actual original launch observation and
        short matched bookends. Native four-stream MMA matches the original
        backend exactly; segmented vector and MMA arithmetic differ.
        Other shapes, production selection and Gemma26 transfer remain open.
      - [x] [Dense31 packed C4 full-head diagnosis](experiments/gemma-packed-attention-c4/README.md),
        with unchanged native repeats, initialized-state witnesses and fresh
        original bookends. The joint attention dispatch/stream geometry change
        makes all 128 complete heads byte-exact, but costs 12.2774% more latency.
        Copy-cost optimization, Gemma26 transfer and broader qualification remain open.
      - [x] [Dense31 actual-input scalar FFN diagnosis](experiments/gemma-dense-ffn/README.md),
        with exact capture/replay controls, observed original fused/separate
        launches and matched component bookends. All complete activation/down
        values match; the fusion-gap lead is rejected for this layer-zero input.
        Other inputs/layers, broader C1 quality and production selection remain open.
      - [x] [Gemma26 packed-attention transfer screen](experiments/gemma26-packed-attention-c4/README.md),
        with independently frozen full heads/states and fresh matched bookends.
        The candidate retains 68/128 positive-margin mismatches and costs 9.19%
        more latency. No production transfer is selected; quality and optimized
        batching remain open.
      - [x] [Gemma26 stock dispatch diagnosis](experiments/gemma26-dispatch-observation/README.md),
        with all 128 observed full heads byte-exact to untouched stock bookends.
        Routing is fused in all 30 decode layers, but prefill refuses layers
        28/29 at its memory gate; scaled reductions are fused throughout.
        This identifies a recipe difference without attributing the full gap;
        common-input diagnosis, competitive batching and qualification remain open.
      - [x] [Gemma26 packed compound-policy screen](experiments/gemma26-compound-packed-c4/README.md),
        with frozen same-policy complete heads/states and fresh stock bookends.
        Routing/reduction reduces positive-margin disagreements from 68 to 2
        of 128; 92 complete heads are byte-exact. Paid latency is 7.15% higher.
        No production policy is selected; common-input diagnosis, competitive
        batching and full qualification remain open.
      - [x] [Gemma26 common-input late-prefill MoE controls](experiments/gemma26-late-moe/README.md).
        Both native policies match the original operator bytes at layers 28/29;
        captured stock uses primitive routing and fused reduction. Tiny recipe
        rounding differs without changing selected IDs. The two full-model
        disagreements, competitive performance and qualification remain open.
      - [x] [Dense31 solo checked-norm first screen](experiments/gemma-dense31-c1-norm/README.md),
        with frozen native full-head repeats and fresh matched original bookends.
        All 32 complete heads match byte-for-byte; short decode latency is 1.92%
        higher. No production policy is selected; 8K/depth, state, batching and
        competitive performance qualification remain open.
      - [x] [Dense31 8K ring-cache reference screen](experiments/gemma-swa-ring-h1/README.md),
        with unchanged native head/state repeats and fresh matched bookends.
        All 32 decode choices agree and the final complete head is byte-exact;
        prefill quality differs and native prefill/decode take 16.20%/2.05% more time.
        Historical full-cache comparisons remain qualified by their recipe;
        Full quality/performance qualification remains open.
      - [x] [Gemma26 8K ring-cache reference transfer](experiments/gemma26-swa-ring-transfer/README.md),
        with unchanged native full-head/state repeats and actual local2,048/global16,384
        reference capacities. All 32 decode choices agree, but both retained full
        heads differ. Native prefill/decode take 57.64%/0.49% more time;
        full quality and competitive performance qualification remain open.
      - [x] [Gemma26 matched prefill profile](experiments/gemma26-prefill-profile/README.md),
        with annotated-control and traced head/state fidelity plus successful
        application/profiler retirement. Diagnostic GPU activity is close,
        while native has 1.361 s outside recorded GPU activity versus 0.085 s
        for stock. CPU and wait attribution, optimization and competitive
        qualification remain open; trace spans do not replace untraced timing.
      - [x] [Gemma26 coarse prefill diagnosis](experiments/gemma26-prefill-coarse/README.md),
        with exact prior head/state controls and checked thread/interval rosters.
        The two graph-plan passes use about 1.093 s of charged caller CPU;
        algorithm changes and paid optimization/transfer gates remain separate.
      - [x] [Call-local graph reader index](experiments/gemma-plan-index/README.md),
        with fresh pre/post-placement checks and exact Gemma26/31 head/state
        controls. Gemma26 prefill is 27.3% faster than its retained baseline;
        initial prefill gaps were 15.10%/13.83%. Full qualification stays open.
      - [x] [Exact source-use indexing](experiments/gemma-use-index/README.md),
        preserving local subgraph gates and exact native heads/state while
        reducing fresh Gemma26/31 prefill time by 2.80%/2.31%. Matched reference
        gaps remain 12.38%/11.40%; math policies and qualification stay unchanged.
      - [x] [Cold/retained prefill-plan comparison](experiments/gemma-retained-plan/README.md),
        with capture disabled, verified cache hits and exact native heads/state/choices.
        Full cold planning costs 126/526 ms at Gemma26/31; state growth costs
        64/164 ms. Retained elapsed prefill improves 7.12%/4.77%, with execution
        timing movement alongside removed planning. Full qualification stays open.
      - [x] [Plain RMSNorm/Mul first screen](experiments/gemma-normmul-screen/README.md):
        Gemma31 selects 121 additional norm fusions and preserves exact native
        heads/state/choices, but that earlier full-head screen showed no resolved
        speed gain; the subsequent state-only adoption is below.
      - [x] [Current Gemma31 prefill timeline](experiments/gemma31-current-timeline/README.md):
        32 GPU-idle gaps over 10 ms total 592 ms; extra vocabulary projections
        cost 145 ms. Product and attention duration sums are close to stock,
        with differing final-block row shapes and 3,872 extra standalone MUL
        kernels. This diagnostic establishes no new optimization or parity pass.
      - [x] [State-only intermediate prefill](experiments/gemma-state-only-prefill/README.md):
        both approved Gemmas preserve KV writes and omit unused final-layer work.
        Ordinary and optional-policy continuation/state controls cover scalar and
        captured unequal-row waves. Measured all1024/both256 prefill improves
        3.04%/2.42%; 26B reference movement and full quality remain unresolved.
      - [x] [Bounded graph traversal](experiments/ggml-graph-order/README.md):
        arena-funded membership preserves DFS and refusal semantics across
        Qwen/DeepSeek/Gemma factories. Exact Gemma26/31 head/state controls pass;
        state-only prefill improves 1.93%/0.99%, leaving 8.19%/7.54% reference
        latency gaps. Planning-only attribution and full qualification stay open.
      - [x] [Bounded next-plan lookahead](experiments/gemma-prefill-lookahead/README.md):
        funded CPU graph/placement construction overlaps current execution;
        binding and cache publication wait for completed work. Ordinary and
        optimized scalar/unequal-wave lifetime controls pass. Same-binary
        Gemma26/31 prefill improves 1.55%/2.36% with exact heads/state; a qualified
        31B comparison leaves 4.94% latency excess against preceding references.
        Full quality, competitive batching and remaining overhead stay open.
      - [x] [Checked plain norm default](experiments/gemma-state-only-norm-policy/README.md):
        both approved profiles retain exact ordinary off/on heads, initialized
        state and choices. Focused scalar/unequal-wave continuation and capture
        controls pass; other arithmetic defaults stay off. Research-policy
        prefill screens favor fusion, with noisy 31B magnitude. Corpus quality
        and optimized batching remain open.
      - [x] [Current post-lookahead phase attribution](experiments/gemma-current-phases/README.md):
        31/32 required plans hit; synchronous planning is about 12 ms. Across
        32 insertions, binding costs 175 ms, coverage 17 ms and insertion 5 ms,
        nested within planning/publication. State growth costs 162–185 ms;
        execution variance remains. Individual binding callees and parity
        are not attributed or qualified by these diagnostic counters.
      - [x] [Per-bind validated wrapper reuse](experiments/gemma-binding-wrapper-cache/README.md):
        immutable registry wrappers are reused within each bind, with fresh
        per-step descriptor, arity and lane checks. Gemma31/26 binding falls
        73.98%/69.29%, with exact prior heads/state and seven focused controls.
        Wall screens favor the change with material candidate spread; no
        sustained speed, corpus quality or reference parity claim is made.
      - [x] [Paid grouped state preparation diagnostic](experiments/gemma-state-preparation/README.md):
        one existing bulk preparation call stays inside paid prefill; later
        chunk checks use initialized state. Gemma31 state time falls 61.1245 ms
        / 33.45%, with exact heads/state, but wall spread prevents a speed claim.
        The [26B transfer](experiments/gemma26-state-preparation/README.md)
        reduces paid prefill by 20.375 ms / 0.822% with exact heads/state/choices.
        Pressure/lifetime qualification and a fresh reference comparison
        precede production selection.
      - [x] [Independent C4 attention owner roots](experiments/gemma-owner-root-attention/README.md):
        all 32,768 real D25631 outputs match packed four-stream MMA byte for byte,
        preserving original grid/reduction partitions and checked independent spans.
        [Real first-global D512 inputs](experiments/gemma-owner-root-global/README.md)
        also pass all 65,536 output values, fresh/restored and captured controls,
        preserving the 96-block grid. The [closed C4 consumer](experiments/gemma-owner-root-c4/README.md)
        removes K/V packing with 8.93% lower paid latency and exact 128 heads/four
        initialized states; plain-norm-on remains 2.10% slower than fresh stock,
        with all 128 heads exact. The [Gemma26 backward transfer](experiments/gemma26-owner-root-c4/README.md)
        lowers paid latency 7.13% with exact own heads/state; native is 0.265%
        slower than fresh stock, and two positive-margin choices still fail strict quality.
        The [variable-width native opt-in](experiments/gemma-owner-variable/README.md)
        removes K/V packing at 8K with exact native heads/four states/choices:
        paid C4 falls 53.95% for 31B and 43.71% for 26B. Fresh stock leaves 31B
        2.31% slower and 26B 5.59% faster, but strict quality fails 4/128 and 25/128
        positive-margin choices. Different untimed prefill and 26B ring capacities
        remain explicit. The owner option defaults off; wider cohorts, serving
        recipes and genuine quality qualification remain open.
      - [x] [Current C4 phase accounting](experiments/gemma-owner-c4-phases/README.md):
        all 32 plans hit; checks cost 24.74 ms and paid time outside execution
        averages 78.73 ms. Exact heads/state/choices remain. This diagnostic
        supplies no fresh-reference or active-GPU attribution; immutable-weight
        placement checks are a bounded lead, not the whole remaining gap.
      - [x] [Successful weight placement memo](experiments/weights-placement-memo/README.md):
        scheduler lifetime/mutation stamps remove repeated immutable-weight scans
        while keeping mutable-state validation and every scheduler Call. Current
        31B C4 checks fall 23.50→2.92 ms and paid elapsed falls 0.77%, with exact
        heads/states/choices and 17 focused controls; no fresh competitive or
        other-family speed claim. Full regression remains owner-deferred.
      - [x] [Successful live-state placement memo](experiments/gemma-state-placement-memo/README.md):
        shared Gemma/Qwen/DeepSeek source/pin checks reuse only successful scans
        under the scheduler lifetime/epoch, with mutation invalidation and every
        scheduler Call retained. Nineteen focused controls and exact complete C8
        head/state/choice controls pass on both Gemmas. The short 26B factor falls
        29.195 ms/1.7661%; 31B latency gain is unestablished within spread. No
        residency memo, fresh reference or Qwen/DeepSeek speed claim.
      - [x] [Immutable Gemma head capacity](experiments/gemma-head-capacity/README.md):
        serving pins one head per configured slot while manual callers retain
        their all-row default. Full input/feature envelopes and early over-cap
        refusal pass 18 focused controls across both profiles. Actual catalog
        bounds confirm reduced pinned backing; row defaults/arithmetic policies
        and all quality/batching gates remain unchanged.
      - [x] [Fresh Gemma input qualification](experiments/gemma-input-preparation/README.md):
        both checkpoints independently match twelve supplied 1,024-token prose
        histories and four natural chat prompts against the public tokenizer;
        actual native/Jinja renders agree and each input has one BOS. Texts
        were frozen before tokenization. Corpus scoring has 1,023 transitions;
        the final cohort frontier has no neighboring-owner target. This is
        input-only evidence, with no model quality or batching qualification.
      - [x] [Candidate natural answers](experiments/gemma-natural-answer/README.md):
        four short C1 prompts complete normally on both checkpoints. 31B
        matches every reference answer and generated ID; both return `45.0`,
        retaining the predeclared exact `45` format failure. 26B matches all
        exact targets; both explanations pass semantic review. This does not
        replace corpus, batching or retrieval-at-depth gates.
      - [x] [Bounded owner cohorts](experiments/gemma-owner-cohorts/README.md):
        opt-in complete C4/C8 attention quads preserve independent real cache roots,
        fund both planning passes and match whole-eight reduction partitions.
        The 31B C8 factor resolves every paid decode head; 26B passes its unchanged
        prior margin and conditional-score gates. Full backing-parent validation
        admits short reads at configured 262K while preserving 16K/64 MiB read
        limits. Fifteen focused controls pass; serving defaults, C12, depth and
        full model qualification remain open. The same measured 31B C8 candidate
        remains 3.57% slower than fresh stock (155.385 ms) in a short matched
        bookend with unchanged complete heads/state; this is not sustained parity.
      - [x] [Equal-width partial owner adapter](experiments/gemma-partial-owner-attention/README.md):
        whole logical5/6/7/9/10/11 keeps the original full grid with active real-root
        groups and sequence-offset fixups. N5/N6 exact operators cover general,
        uniform and metadata-free branches; startup/both-pass funding passes.
        Gemma31 C5 recovers its retained FIRST-stock strict and 160-target loss gates,
        with all 160 paid heads exact; the original failure stays recorded.
        Runtime/default changes, unequal widths, remaining cohort/profile model
        controls and sustained/depth performance remain open.
      - [x] [Partial-owner primitive and Gemma26 transfer controls](experiments/gemma-partial-owner-transfer/README.md):
        N5/N6/N7/N9 operator 16-shape/36-group proof passes eager and captured
        complete output with heads16/32. FIRST Gemma26 C5 has two strict
        differences /165 heads; its 160-target conditional loss passes independently.
        Primary failure is preserved; no defaults or wider quality admission.
      - [x] [Gemma26 C5 prefix-only keep diagnostic](experiments/gemma26-c5-prefix-keep/README.md):
        benchmark-only retained-output wrapper; four no-launch modes/40 assertions
        pass. All160 paid heads match retained FIRST stock, but one frontier choice
        remains strict FAIL. The independent160-target loss gate passes at +0.08119%;
        original two-paid-miss failure remains preserved. No routing whitelist/defaults.
      - [x] [Grouped independent cache-write screen](experiments/gemma-grouped-cache-writes/README.md):
        rejected report-only candidate. Original and fast-divider 31B C12 screens
        increased mean paid time 7.18%/3.22%; timing movement limits attribution.
        Complete native heads/state/choices remain exact; both six-control proofs
        pass. No prototype source/default adopted, 26B transfer or new oracle.
      - [x] [Ordinary whole-C12 factor](experiments/gemma-c12-single-wave/README.md):
        twelve shared product columns and three funded real-root attention quads;
        D-092 invariant waves keep eight and serving owner/joined defaults stay off.
        All 384 paid 31B heads match retained stock exactly; 26B has three strict
        differences, none outside its unchanged prior bound. Both 384-target
        conditional-loss gates pass. The original 8+4 failures remain recorded;
        short before/after timing includes source-vintage confounds and does not
        qualify sustained speed, corpus/depth quality or production batching.
      - [x] [Fresh Gemma26 scheduling and heldout control](experiments/gemma-fresh-quality-validation/README.md):
        fixed-capacity native 128/1,024 schedules freeze a genuine operational
        bound on untouched history 13 before independent history 14. All 17
        heldout disagreements fit that unchanged bound and PPL increases 0.1970%,
        passing the declared gates. Strict zero-difference still fails; the
        uncommitted candidate recipe, prior failures and broader gates remain separate.
      - [x] [Current full-head corpus screen](experiments/gemma-current-quality/README.md):
        31B both256/plain-norm-on matches all 1,024 fresh ring-reference heads
        and 1,023 likelihoods exactly. The 26B all1024 transfer has nine positive-
        margin choices and fails strict quality despite only +0.0528% PPL.
        Both own repeats are independently frozen before stock; no inherited
        128-row allowance or full-model/batching qualification is claimed.
      - [x] [Fresh current solo reference bookends](experiments/gemma-current-reference/README.md):
        26B retains stable 2.43% prefill / 0.92% decode latency gaps. Native 31B
        prefill is stable, but stock varies 1.3355 s; the closing-arm comparison
        remains 2.08% slower, with decode 2.13% slower. Exact prior native
        heads/state/choices remain; no parity, corpus or batching pass follows.
      - [x] [Actual current Gemma26 corpus dispatch](experiments/gemma-current-dispatch/README.md):
        observed stock output matches all 1,024 frozen reference heads exactly.
        Stock selects 29 routing fusions, with no layer-28 selection; native
        all1024 selects 30. Norm/reduction counts agree. This selected-chain
        observation did not establish refusal reason or causality; no policy changes.
      - [x] [Gemma26 routing keep factor](experiments/gemma-keep28-routing/README.md):
        keeping only layer 28's routing probabilities changes native routing
        from 30 to 29 fusions and matches all 1,024 stock heads exactly, with
        independently frozen finite own repeats. This resolves the nine
        fixed-corpus disagreements; general selection, other shapes and production
        policy remain open.
      - [x] [Actual Gemma26 routing refusal](experiments/gemma26-routing-gate/README.md):
        all 30 candidates pass structure/shape, but layer 28's weights overlap
        external router logits by 32,768 bytes. The original >8-row memory gate
        declines fusion. Complete stock heads remain exact; this allocation-
        dependent result supplies no production layer or token-count rule.
      - [x] [Strict Q8_0 Gemma assistant binding](gemma4-assistant.md) for both
        approved target pairs, including kept architecture semantics, complete
        shared-target contracts and native canonical-vocabulary comparison.
        Bounded component execution and feature/cache lifetime controls are below;
        serving speculation and full qualification remain owed.
      - [x] [Original-image Gemma26 assistant oracle seam](experiments/gemma-assistant-reference/README.md),
        with full-head/recurrent-feature own repeats and unchanged physical
        target caches at C1, serial C2 and genuine batch two. Native assistant
        quality, performance and speculation qualification remain open.
      - [x] [Bounded native Q8 assistant component](experiments/gemma-assistant-execution/README.md),
        with explicit post-finalnorm features, scoped frozen-cache borrows,
        original-input C1 arithmetic and native independent-slot controls.
        Serving verification, optimized batching and quality/performance remain open.
      - [x] [Bounded engine-only Gemma target verification](experiments/gemma-target-verify/README.md),
        default-off one to four C1 rows with explicit head/feature funding,
        exact independent four-row acceptance, rejected KV restoration and
        projected continuations on both profiles; 11 focused controls pass.
        Scalar-prefix parity, deterministic verify budget-pressure refusal,
        assistant serving and whole-chain quality/performance remain open.
      - [x] [Gemma26 assistant C2 arithmetic and timing diagnosis](experiments/gemma-assistant-c2/README.md),
        with frozen native serial/joined recurrence, identical-input original
        comparisons and matched paid bookends. Both native policies reproduce
        serial-reference full rows exactly; original batch-two distribution
        differences remain. Full optimized-batching and serving qualification
        remain open.
      - [x] [Gemma31 Q8 assistant frozen C1/P64 extension](experiments/gemma31-assistant-reference/README.md),
        with complete original-image heads/5,376-value recurrent features exact
        for one/three-step chains and both own repeats; protected complete
        original state and native borrowed caches remain unchanged. No C2,
        native-target chain, serving/speculation or performance qualification.
      - [x] [Bounded engine-only Gemma C1 greedy transaction](experiments/gemma-assistant-greedy-unit/README.md),
        with borrow release before anchor-plus-draft verification, retired-prefix
        commit, selected target head/feature and explicitly uncommitted next anchor.
        Both real target/Q8 assistant pairs pass 11 focused controls against an
        independent same-four-query target, rejected bytes and projected continuation.
        Default-off; serving, scalar-width quality, sampling and performance remain open.
      - [x] [Matched Gemma31 target-plus-assistant C1/P64 transaction](experiments/gemma-assistant-greedy-reference/README.md),
        with query64 prefill/query4 verify, complete target/assistant heads and
        retained features byte-exact under existing target norm chains, matching
        retired acceptance/pending carry and zero conditional loss increase.
        Plain-norm baseline strict3/acceptance/conditional-loss FAIL remains;
        26B, scalar-width and serving remain open; bounded repeated performance follows.
      - [x] [Gemma31 all-cost repeated assistant screen](experiments/gemma-assistant-throughput/README.md),
        resident C1/P64 fixed32 ignoring EOG: all four native/original plain/assistant
        continuations and pending anchors agree after independent own-freezes.
        Native actual GreedyUnit mean 1.315482 s versus plain 2.939108 s (55.24% less
        elapsed, 2.2342× rate), 1.25% slower than original assistant; 12 units,
        46 verified/34 drafted rows paid. Short screen only; serving, terminal
        behavior, other prefixes, sustained performance and 26 transfer remain open.
      - import to a v0 artifact;
      - its runner: plan, state layout and model-specific steps;
      - the native tokenizer and its chat template's rendering (native or
        interpreted, D-067), with the template hash
        recorded in the support matrix;
      - swaps in and out beside the M3 models;
      - speculation where the family ships MTP layers or drafters;
      - the D-053 rule: a primitive fallback for every fused operation;
      - **Optimization transfer and batching** (owner, 2026-10-04):
        adopt all applicable optimizations already selected by the Qwen
        and DeepSeek paths, using the
        [inventory](optimization-inventory.md) to check each new family
        and quantization. Record eligibility, actual dispatch and measured
        correctness, speed and memory; account for shape or format limits.
        A family or quantization's supported status includes optimized
        batching, with the concurrency evidence below.
        Before closing each model, review its learnings for the previously
        implemented models, particularly Gemma 26B, and apply eligible
        improvements with matched correctness and performance checks
        (owner, 2026-10-05). Record transfer limits in the inventory.
        *Since 2026-10-08 (D-107), a family resumed on the component
        engine inherits its pieces' optimizations; workflow.md's "optimize
        once" replaces the per-family adoption and review above. Its
        batching evidence stays required.*
- [ ] **Concurrent requests with continuous batching** (the owner,
      2026-09-29). The primary workload includes an agent plus
      subagents, which is several concurrent requests on the same
      resident model. The engine supports:
      - per-request state side by side;
      - decode steps that batch rows from different requests (plain and
        speculative);
      - chunked prefill interleaved with decode;
      - shared-prefix state with copy-on-fork for parallel agent
        branches;
      - admission within the memory budget.

      This applies to every added model family, including Clef's,
      Clef-flash's and Clef-Omni's prefill-only decisions and image
      generation (owner, 2026-10-03). Compatible work from independent requests batches in
      the encoders, decision heads, denoising steps and image decoders.
      Each request keeps its own state, seed, guidance, step count,
      size, edit inputs and cancellation. Different input lengths or
      image sizes form compatible groups and interleave; they do not
      disable batching for the model. Admission follows memory, and
      waiting groups age so compatible arrivals cannot starve them (by the
      starvation bound of the throughput-first item below).

      The chat route's one-request-at-a-time queue (D-097) becomes a
      batch scheduler. Batching applies to requests for the same model;
      different models still time-slice by swapping (D-019).
      M3's added concurrent-request comparisons now measure this gap;
      advance the necessary implementation into the optimization run
      where those comparisons require it (owner, 2026-10-01).
- [ ] **Throughput-first scheduling of concurrent work, within a
      starvation bound** (owner, 2026-10-20; AGENTS.md rule 11, D-102 as
      amended; [Flare](flare-assessment.md#owner-decision-aggregate-completion-over-cadence)):
      - **mixed prefill and generation waves for every model:** generating
        streams' decode or verify rows ride inside each prefill chunk,
        extending `RunCausalWave` (plain waves only today; a speculating
        model refuses it, `src/runtime/llm_mixed.cc`) to speculative ones,
        within each state contract's rules and on the verify shapes of the
        speculation items above (total-rows-keyed graphs, per-request row
        counts);
      - **chunk size for aggregate completion:** the fastest measured chunk
        stays the default; a smaller chunk is chosen only where a
        measurement shows it shortens total completion, never for decode
        cadence or fair time-sharing;
      - **a starvation bound in calibrated unit time:** no admitted
        request's next unit (a prefill chunk or a decode step) waits longer
        than the bound, default 60 s (a judgment default under D-085: a
        fifth of Claude Code's and Codex's 300 s idle timeouts; not
        measured), configurable, from calibrated unit costs (no clock
        reads). It applies among requests on one resident model; waits
        between models stay D-050's and D-019's. Request classes (D-042,
        M6) may order work within it, never past it. It replaces
        `cohort_schedule.h`'s unit-count bounds (a generation waits at most
        one prompt unit; a prompt ages in after 12), which stretch with
        chunk length. It reorders work, never refuses or cancels it, and
        keepalives continue meanwhile;
      - **measured** as the total completion time of a multi-request agent
        workload (C2/C4, long prompts arriving while others generate)
        against today's schedule, with the worst wait and worst
        inter-token gap reported and checked against the bound; adopted
        on aggregate gain under D-085.
- [ ] **Skeleton gaps the M3 cleanup's review named**
      ([engine.md](engine.md)), *moved to M3.6* (its IR, which names no
      kernel library, and its decoding-mode orchestration with any number
      of drafters):
      - planning, capture and launch binding are GGML-only, so a family
        run on EXL3 (or another non-GGML backend) needs a step family of
        its own that plugs into the same skeleton;
      - runners assume one target and at most one drafter.
      M3's growing-state work has closed the setup allocation gap for
      DeepSeek and Qwen3.8: stable virtual regions, backing only when used.
- [ ] **Kernels:** operations new to a family come from GGML first, with
      our own kernels on measured need (D-053). Upstream findings go to
      docs/upstream/.
- [ ] **Quantization formats** (the owner, 2026-09-29), especially the
      variable-bit ones. A format coverage matrix sits beside the
      capability matrix. The covering set runs every format still in wide
      use:
      - GGUF K-quants, I-quants and dynamic per-tensor mixes;
      - MXFP4, NVFP4, MXFP8 and FP8;
      - AWQ and GPTQ;
      - MLX affine;
      - EXL3.

      Each is as fast as its same-format reference. The bounded
      [legacy quant primitive slice](experiments/m35-legacy-quants/README.md)
      adds compiled/operand coverage and transfer controls; whole-engine
      format and model qualification remain open.
- [ ] **MLX affine import** (moved from M9 by the owner, 2026-09-29):
      import and run TensorFold's MLX 4-bit checkpoints (reconcile the
      group size, 32 or 64), starting with Qwen3.8 Flash Next's, so
      TensorFold is a same-format oracle and gated comparator (D-085)
      instead of cross-quantization information.
- [ ] **EmbeddingGemma 2 research addition** (owner, 2026-10-06): add
      Google's newly released multimodal embedding checkpoint to the
      [candidate list](m35-families.md#owner-requested-research-addition-embeddinggemma-2).
      Select the actual checkpoint/format and reference, resolve source metadata,
      and assess native embedding outputs, media, batching and API qualification.
      Listing does not adopt a runtime path or change D-042's M10 embedding API
      schedule or D-044's generic-pooling deferral.
- [ ] **Legacy-tier features** (the owner, 2026-09-29): list the older
      generations' features that are not subsets of the covered ones
      (Gemma 2, Phi-3.5, Mistral 7B, Command R7B, Llama 3.2 and others),
      with which models and vendors used them and whether each was
      abandoned or just not updated. Implement those worth keeping.
      [Study](m35-families.md#legacy-tier-features). *Owner, 2026-09-29:*
      implement the seven small features:
      - classic SentencePiece (native GGUF tokenization and pinned
        agreement for the three approved SentencePiece fixtures are in
        place; [tokenizer](tokenizer.md));
      - linear RoPE scaling;
      - attention logit soft-capping;
      - LongRoPE;
      - LayerNorm with parallel attention and FFN blocks;
      - the older GGUF block types (Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL).

      Their four GGUF fixtures (Gemma 3 4B QAT Q4_0, Gemma 2 2B, Phi-3.5
      mini, Command R7B; about 17.9 GB, llama.cpp as reference) join
      M3.5's set under the same exit criteria.
      Gemma 2 [profile and tensor foundation](gemma2.md) now recognizes all
      288 actual descriptors, tied output and width-2304 readable padding;
      its [bounded C1 execution slice](experiments/gemma2-execution/README.md)
      authenticates and deeply imports the full approved source, qualifies
      softcap occupancy/launches and state/graph/runner controls, and reproduces
      all 33 optimized stock heads exactly. The opt-in
      [C2 owner decode screen](experiments/gemma2-owner2/README.md) qualifies
      true-softcap50 attention, own/eager/fallback and wrapped-ring restore-next
      controls. Small/ring stock gates pass with zero strict differences and
      65/66 plus 66/66 exact heads; short n=2 paid latency is 1.49% above stock
      with exact natural histories/final heads in that independent-prefill screen.
      The [bounded serving transfer](experiments/gemma2-serving/README.md) adds
      true-cap50 compatible prefill/common-width decode under total wave 256 while
      retaining per-owner 128/ring 4,352. All 76 representative stock heads are exact;
      all 45 focused and five HTTP cases pass, including actual system refusal,
      joined groups and exact kept restart. Short n=2 C2 latency is 5.78% above
      stock with exact natural choices/final heads for that padded snapshot.
      The subsequent copy-free bounded-read policy preserves original cap50
      partitions with actual root bounds and neutral empty fixup data. All 22
      focused controls pass; small 76/76 and representative wrapped 73/76 stock
      heads have zero strict differences, and padded/bounded native state/heads
      remain byte exact. Same-binary native bookends reduce paid C2 latency
      5.2588%, to +0.09665% versus original stock (n=2). The bounded runtime
      recipe selects it and repeats all five HTTP cases/two clean restart
      epochs with actual bounded selections. Broader context/quality,
      additional ring/chunk geometries, memory/swap, wider cohorts and sustained
      qualification remain open.
      The [checkpoint/adoption foundation](experiments/gemma2-checkpoint/README.md)
      passes 19 focused checks for initialized extent coverage, pending copy
      retirement/cancellation, protected peers and wrapped-ring named-file
      adoption in a fresh node with exact next-head replay. It adds no serving
      admission or new reference/performance claim.
      Gemma 3’s [internal H8 C4 foundation](experiments/gemma-h8-c4/README.md) passes
      original-MMA/FP64/replay and four-state/head funding controls. One equal
      model screen has zero native/stock strict differences (146/148 exact
      heads); solo native is 148/148 exact to stock solo. Both engines retain
      the same 12 positive-margin solo/C4 differences, leaving concurrency
      open. Short n2 C4 paid latency is +3.29938% with exact natural choices
      and final heads. Its subsequent
      [actual C3 departure/rejoin control](experiments/gemma-h8-c3/README.md)
      passes packed-MMA/FP64/replay and mixed-padding funding controls. Paused
      peers, scalar catch-up and C4 rejoin retain exact own state/replay/restore;
      same-schedule stock has zero strict differences over 144 targets and
      147/148 exact heads. No new timing or concurrency allowance is introduced.
      The [internal grouped H8 C8/C12 foundation](experiments/gemma-h8-wide/README.md)
      now checks every partial C5..7/C9..11 and unequal C4 in one continuous
      pause/catch-up/clear/refill schedule. Whole-wave products and common
      attention width across groups retain every-split funding; 472 heads /
      468 targets have zero strict differences (276 exact), with exact own
      device/state/peer/restore controls. A separate C12 RNNR preserves all
      384 natural choices/cursors and own final heads, but its original final
      cross-engine byte gate remains FAIL (maximum delta 1.90735e-6, zero argmax
      differences). Descriptive n2 wall is +11.3219% total / +23.5760% decode.
      A subsequent [whole-C12 bounded-root factor](experiments/gemma-h8-wide-bounded/README.md)
      removes the representative 408 K/V CONCAT sources while retaining the
      global logical masks/partitions. The four native arms preserve all 384
      natural choices, cursors and final heads exactly; short n=2 decode falls
      21.0667%, total paid latency 8.26029%. The option defaults false, and the
      earlier stock final-head-byte FAIL remains. Ordinary slots remain two;
      other wider/partial bounded reads, cap50 wider transfer, solo/batch
      concurrency, corpus/memory/sustained and public adoption remain open.
      No threshold or prior twelve-flip allowance changes.
      Gemma 3 [profile and tensor foundation](gemma3.md) is checked:
      actual approved metadata and all 444 tensor descriptors, separate
      profile/binding and refusal controls; three focused CPU controls pass.
      Separate state/input and primitive descriptor graph/plan foundations are
      checked by 13 focused CPU/no-launch controls. The approved source now has
      a [deep-verified prepared artifact](experiments/gemma3-execution/README.md)
      and seven focused import controls. The shared native runner now passes a
      bounded C1 own control, including actual local/global RoPE, eager/graph
      heads, Clear and initialized-state restore. The primitive first stock screen
      retains one positive-margin greedy difference; the
      [checked norm chains](experiments/gemma3-execution/README.md#checked-norm-chains-exact-bounded-c1-quality)
      resolve it with all 33 complete stock heads byte-identical. Device greedy
      retains exact choices/state; allocation-free public binding validation
      narrowed its historical short C1 screen to 0.56% slower than stock backend
      greedy. A [bounded C2 joined-decode screen](experiments/gemma3-execution/README.md#independent-prefill-c2-decode-screen)
      now passes strict quality after independent prefill: zero greedy differences,
      64/66 byte-identical heads, exact own state and +2.08% paid latency versus
      stock. The [bounded serving unit](experiments/gemma3-execution/README.md#bounded-serving-unequal-widths-and-wrapped-rings)
      now admits context 4096/rows 128/C1-C2 with unequal-width/ring/departure,
      exact initialized checkpoint/restore, HTTP and restart adoption controls.
      Independent-prefill runner screens are +0.96% C1/+2.36% C2; a native HTTP
      old/new bookend improves C1/C2 by 3.62%/3.18% with exact cached responses.
      The [compatible-prefill unit](experiments/gemma3-execution/README.md#compatible-joint-prefill-and-common-attention-reads)
      adds plain equal 2–128-row chunks under a separate 256-row wave bound.
      Real K/V packing and funded common-width decode padding preserve actual
      per-owner state bounds; unequal/ring/short screens pass all 216 strict
      transitions, with 73/74, 70/74 and 74/74 exact heads. One matched short
      C2 cycle is 1.49% above stock. Actual joined-prefill HTTP/restart gates
      pass; a cold native C2 endpoint bookend improves 14.03% with exact
      generated IDs/usage at that separate boundary. The subsequent
      [copy-free bounded-read transfer](experiments/gemma3-execution/README.md#copy-free-bounded-owner-reads--2026-10-07)
      preserves exact padded/bounded native heads/state; fresh small/ring stock
      screens have 73/74 and 70/74 exact heads and zero strict differences.
      A same-binary short n=2 screen lowers native paid latency 5.77372%, to
      −0.29466% versus original stock, with identical histories/final heads.
      Four HTTP aggregate cases/two clean epochs pass with actual bounded plans
      and exact matched cached replay. Explicit scalar context 8448 now retains
      the default 4096/two-owner bounds at 4K. A public 8192-prefix/64-continuation
      HTTP control and one Gemma3/Qwen exact-state model-switch pair check the
      repeated-text boundary; corpus/retrieval, broader cohorts, maximum-context
      memory/swap and sustained quality/performance remain open. An
      [internal trained-maximum C1 screen](experiments/gemma3-execution/README.md#internal-trained-maximum-c1-screen-2026-10-07)
      initializes all 131,072 positions with normal reclamation and matches all
      65 stock heads exactly on one 64-target corpus tail. Public maximum
      admission, retrieval and performance qualification remain open, including
      the descriptive initial 48.7-second native versus 32.9-second stock gap.
      The [global host-mask factor](experiments/gemma3-execution/README.md#global-host-mask-factor-2026-10-07)
      retains all exact mask checks and 65 frozen heads, reducing one native
      first-traversal prefill bookend mean from 49.056 to 43.818 seconds; this is
      not a new stock-parity or sustained-performance result. The subsequent
      [GPU-mask transfer](experiments/gemma3-execution/README.md#graph-owned-gpu-masks-2026-10-07) preserves every frozen head and reduces
      native prefill from 43.401 to 35.304 seconds in one first-traversal
      bookend; ordinary bounded serving uses the checked device producer,
      without qualifying public maximum depth or sustained performance.
      [Prefill lookahead and graphs captured ahead](experiments/gemma3-execution/README.md#prefill-lookahead-and-graphs-captured-ahead-2026-10-07)
      replay 1,085 of 1,088 runs on a first traversal and cut it from 35.520 to
      32.180 seconds, 1.31% above an interleaved stock control (31.820), with
      every frozen head exact and warm repeats unchanged.
      The deferred and dropped
      features stay as recorded. *Owner, 2026-09-29:* PrismML's Bonsai
      join too, both 1-bit and 2-bit, because they are hugely popular:
      - Bonsai-27B Q1_0 (with its Q4_1 drafter);
      - a 2-bit Ternary Bonsai.

      Which 2-bit build is settled at M3.5's start. Ternary-Bonsai-27B's
      Q2_0 on stock llama.cpp is unconfirmed, and Ternary-Bonsai-2-27B
      (on Qwen3.8-27B) needs PrismML's llama.cpp fork, so its reference
      is that fork, pinned and license-audited like any baseline. The
      stock-llama.cpp Q2_0 is preferred if it works; the Bonsai-2 card
      warns that rotated weights load silently and give garbage, so
      correctness is checked against its reference, not assumed.
- [ ] **EXL3 optimization** (the owner's particular interest): the
      trellis-encoded quants across their codebooks, bitrates and
      per-layer mixed widths, dense and MoE (grouped mixed-width routed
      experts). The target is faster than ExLlamaV3 on the GB10, with
      TensorFold's EXL3 path reported beside it, at decode and prefill.
      It builds on M2's native EXL3 linear (D-080).
      Study first: [TensorFold #42](https://github.com/ashhart/TensorFold/pull/42)
      (merged 2026-09-28, MIT). It reads EXL3 on CUDA for any codebook
      (3inst, mcg, mul1), 1–8 bits and mixed-K packs, with:
      - a row-invariant dense EXL3 linear (1–128 rows), bit for bit with
        ExLlamaV3's `reconstruct`;
      - coalesced low-bit word reads with prefetch;
      - one grouped kernel for mixed-width routed experts (no atomics or
        host syncs, graph-safe).

      It reports 1.5–1.6× ExLlamaV3 on Qwen3.8-27B and Flash Next, 1.5–3.6×
      on mixed-K expert layers, and drafted output equal to serial
      (creator-reported). Its prefill is untuned (decode kernels in 64-row
      chunks), and mcg and 3inst are verified on partial packs only.
      Measure it against ExLlamaV3 and us, then adopt what transfers
      (D-091).
- [ ] **Multimodal file inputs** (D-101, moved from M10 by the owner,
      2026-10-02): images (several per request), video files and audio
      files on every model here that takes them, brought up with its
      family. Live streams stay deferred (D-042). Encoders are components
      of the model's composition (D-089), paged and released like other
      extents ([facts and sources](m35-families.md#media-inputs-decision-models-and-generation-apis)).
      - **Qwen3-VL vision tower** (27 layers, patch 16, 2×2 merge,
        interleaved M-RoPE): Qwen3.8 Flash Next (from M3), Qwen3.8-27B,
        Ornith 1.5, Bonsai-27B, Clef and Clef-flash. Images and video
        (2 fps frame sampling, text timestamps). Qwen-Image-2.1's edit
        mode uses the same family of tower for its reference images.
      - **The other M3.5 encoders** as their pinned files carry them:
        - Gemma 4 26B and 31B, with images and video as frames; their
          image tokens are bidirectional on sliding layers only;
        - Gemma 3 4B (SigLIP);
        - Llama 4 Scout (tiles);
        - Mistral Small 4 (Pixtral);
        - Muse Glimmer (images; its video is unendorsed);
        - MiMo-V2.6 (images and video; the pinned EXL3 build dropped its
          audio).
      - **DeepSeek V4 Flash Vision-Exp** (images; kept by the owner,
        2026-10-02, with V4.1 Flash's vision following in M4): V4 Flash
        0731 is text-only, and the vision model is a separate,
        continued-trained checkpoint whose GGUF and encoder sit in antirez's pinned
        repository. Its row-pair "N-layout" fits CSA compression.
      - **Qwen3-Omni encoders** (Clef-Omni, owner, 2026-10-09): the
        Qwen3-Omni thinker's vision and audio encoders, images, audio and
        video (2 fps, with its soundtrack when every video in the record
        has one). The talker and code2wav weights it ships are unused.
      - **Audio:** apart from Clef-Omni's, no pinned checkpoint keeps an
        audio encoder, so Gemma 4 E4B-it joins as the audio carrier
        (owner, 2026-10-02): 16 kHz,
        128 mels, 30 s per encoder window, sharing Gemma 4's tokenizer,
        template and vision. Longer audio is split into windows, not
        refused (D-102).
      - **Intake** on the chat route: Chat Completions `image_url`,
        `input_audio` and vLLM's `video_url`, inline data only, with
        remote URL fetching off by default. Byte, pixel, frame and
        duration bounds are set before decoding, derived from memory and
        the model's context (encoder tokens) rather than fixed counts,
        configurable, permissive by default (D-102). The image, audio and
        video decoders are chosen under D-017 and D-080, and they and the
        intake take the heavy path.
- [ ] **Decision models over the Jev API** (D-101, the owner,
      2026-10-02): `POST /v1/systemone`, wire-compatible with TypeSafe's
      Jev/SystemOne OpenAPI spec, which the TypeSafe SDKs, gateways and
      Workers AI's Clef share.
      - **Test models:** `Cloudflare/clef` (Qwen3.8-27B backbone) and
        `Cloudflare/clef-flash` (Qwen3.5-9B backbone), each with a joint
        schema head of about 125M parameters, and `Cloudflare/clef-omni`
        (owner, 2026-10-09: Qwen3-Omni-30B-A3B mixture-of-experts thinker
        with vision and audio encoders, the same kind of joint schema
        head; states may carry text, JSON, images, audio and video).
      - **Decision program:** one prefill with no retained state. The head
        reads every position's final-norm hidden state and the output-head
        rows of each option's tokens. It answers all of a request's
        questions jointly, with images and videos as Clef accepts them.
        Independent requests to each decision model batch internally,
        including the joint head, without requiring a batch API route.
      - **Wire behavior:** confidence follows TypeSafe's published
        formulas; Clef's reference reports the top probability instead,
        and that difference is recorded.
      - **Intake bounds** (D-102 revises D-101's starting bounds): no
        question or image counts (Workers AI's 1–64 questions and 4
        images are not adopted); TypeSafe's wire ranges stay (2–255
        options, 2–10 score levels); image bytes and pixels follow memory
        and the context, as the chat route's media do.
      - **Clients:** the TypeSafe Python and JavaScript SDKs, unmodified,
        pointed at llmpalooza by base URL.
- [ ] **Media generation routes** (D-101, the owner, 2026-10-02):
      - **Image batching:** Qwen-Image-2.1 and
        Ming-Image-0.1-Design join compatible phase work across requests,
        including generation and edits where supported, under the
        concurrency requirement above. Latents and random-number state
        belong to each request; cancelling one retires its work without
        cancelling its peers.
      - `POST /v1/images/generations` and `/v1/images/edits` for
        Qwen-Image-2.1, in OpenAI's shape with vLLM-Omni's diffusion
        fields. Edits take reference images and a mask.
      - **Ming-Image-0.1-Design** (owner, 2026-10-03), a second
        text-to-image model to flesh out the image pipelines and API
        ([details](m35-families.md#generative-media-video-and-image)):
        a 6B DiT conditioned by a Bailing MoE multimodal encoder
        through a Qwen2 1.5B connector, and Qwen-Image's
        VAE with four channels, so it generates RGBA. Through
        `/v1/images/generations`, including OpenAI's
        `background: "transparent"`. Size, step count, guidance and seed
        are the caller's per request (OpenAI's `size`, vLLM-Omni's
        diffusion fields), as for Qwen-Image; the card's defaults (12
        steps, CFG 1.0, 2,048² or 1,024²) are the defaults and the tested
        points, not limits. Sizes are adjusted, not refused (owner,
        2026-10-03), as diffusers and vLLM-Omni do: a width or height
        that isn't a multiple of the model's granule (16 here) is
        rounded down to one (at least one granule), so an invalid size
        matches the reference engines' output; a size past the model's
        maximum (a side limit or a pixel-count limit, whichever binds)
        is first scaled down by one factor for both sides, keeping the
        aspect ratio. The image comes back at the size generated, and
        the adjustment is logged. Only malformed or non-positive values
        are refused. The same rule applies to every image model's
        routes. Its components run as one composition (D-089), sharing
        the image phases, VAE and route code with Qwen-Image's rather than
        a second pipeline.
      - `/v1/videos` asynchronous jobs for MiniMax H3 text-to-video and
        image-to-video, on D-041's job machinery, with generated media kept
        only until fetched or expired.
      - `/v1/audio/speech` on three text-to-speech testbeds (owner,
        2026-10-02; Qwen3-TTS added 2026-10-09):
        - **Breeze-TTS-2:** 3.47B in BF16. A T5Gemma2 text encoder (26
          layers), a Qwen3 backbone (28 layers, 2048 wide), a depth
          decoder over 16 codebooks and a Mimi-style codec at 24 kHz.
          It does voice design from instructions, and voice cloning from
          a reference clip, which goes through the codec's encoder.
          Classifier-free guidance doubles its rows. Streaming output.
        - **Kokoro-82M:** StyleTTS 2 with an ISTFTNet vocoder and a
          PL-BERT encoder, over misaki phonemes, with 54 voice packs, at
          24 kHz. Its weights and voices are PyTorch pickles, so import
          reads them without executing them. misaki's grapheme-to-phoneme
          step runs natively, with no interpreter in serving (D-010); its
          espeak-ng fallback is GPL and belongs in the optional copyleft
          tier (D-080).
        - **Qwen3-TTS** (Apache-2.0, ungated; pinned at bring-up, the
          commits current on 2026-10-09 recorded here): a talker LLM that
          emits 12 Hz codec codes (about 83 ms of audio each) and a
          lightweight code-to-wave decoder, `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
          (`5d839924`) and `-1.7B-Base` (`fd4b2543`) with voice cloning,
          `-0.6B-CustomVoice` (`85e237c1`) and `-1.7B-CustomVoice`
          (`0c0e3051`), `-1.7B-VoiceDesign` (`5ecdb673`), and the codec
          `Qwen/Qwen3-TTS-Tokenizer-12Hz` (`7dd38ad4`). Qwen3-Omni (Clef-Omni)
          ships the same kind of talker and code2wav, so the components
          are shared. Realtime voice is a target, after
          [Qwen TTS latency](https://www.storminthecastle.com/posts/qwen_tts_latency/?v=3)
          (owner, 2026-10-09; vLLM-Omni on an RTX 5090: 47–64 ms to the
          first audio chunk, 54–68 ms with a cached cloned voice):
          - time to first audio (request to first audio chunk) is
            measured and reported beside the real-time factor;
          - the first chunk is one codec frame, decoded as soon as the
            talker emits it with priority over steady-state work; later
            chunks grow for throughput. Talker and decoder interleave in
            the one runtime process (D-005), with no inter-process hop;
          - every graph the first request needs is captured and warmed
            before the model reports ready, so no request pays capture
            or tuning (the article's warmup run took 781 ms);
          - speaker embeddings (and reference codec frames for in-context
            cloning) are computed once per voice and cached, and voices
            can be added without a restart (per-request encoding of a 5 s
            clip costs 50–80 ms, about doubling time to first audio);
          - client playback buffering and resampling stay the client's
            (the article's clicks and artifacts were playback, not
            synthesis) and are documented for client authors.
      - Stable Diffusion WebUI's `/sdapi/v1/txt2img`, `/img2img`,
        `/sd-models` and `/options` over the same pipeline, for Open
        WebUI, SillyTavern and LibreChat.
      - Images in chat responses (OpenRouter's `message.images` and
        `delta.images`, D-046), so a chat request to an image model
        returns its image.
      - `POST /v1/audio/transcriptions` on the audio carrier, in OpenAI's
        shape, for voice input in Open WebUI and LibreChat.
      - An optional **MCP media server**, a separate process calling these
        routes. It is how coding agents whose built-in image generation
        cannot point at a local server mix generated images into chat:
        Claude Code, OpenCode, Codex, Gemini CLI, and Antigravity where it
        accepts the server. Open WebUI and LibreChat call the routes
        directly, and Codex's own image tool is tried against them too.
- [ ] **Resident only.** Demand-paged experts stay M7's; M7 keeps its
      daily-driver usability work and builds on the families brought up
      here.

**Exit criteria**, per approved family and form (MoE, dense). D-101's
decision and speech models meet their own criteria below in place of
the correctness, speed and long-context ones:

After each new model or quantization is integrated, run and record a matched
quality and performance comparison against its pinned same-format reference
engine (owner, 2026-10-04). The final comparison covers the production solo
and optimized batched paths and gates supported status under the criteria
below.

Use TensorFold as an additional competitive performance target, checking its
current upstream revision for each comparison or optimization task and freezing it under
[reference comparisons](reference-comparisons.md) (owner, 2026-10-04).
Same-format correctness oracles remain required; cross-format
TensorFold comparisons report speed, memory and separate quality controls.

- **Correctness:** greedy tokens match the same-format oracle except
  near-ties, under the recorded-first noise bound; perplexity within a few
  percent; speculation, where present, meets M3's speculation criteria.
- **Speed and memory** (D-085): prefill and decode at least as fast as the
  same-format reference, at 8K and at depth; report peak memory and permit
  worthwhile speed or quality benefits to use more than the reference when
  the complete workload safely fits (owner, 2026-10-13).
- **Long context:** M3's scaling criterion applies to the family's
  maximum context on one Spark, with any architectural floor measured and
  named. Dense global attention's per-token KV read is such a floor.
- **Swap:** each family swaps A→B→A with an M3 model within M3's swap
  goals, exact on return.
- **Concurrency:** with batching on, single-stream prefill and decode stay
  within noise of M3's. Aggregate decode throughput rises with the
  number of concurrent requests (reported at 1, 2, 4, 8 and 12, against
  ds4's batched serving and vLLM where they run the same model and
  format). Each request's greedy output equals its output when run
  alone, except near-ties. Forked branches share their prefix state
  without copying it until they diverge.
- **Host token memory:** the native token-history task above is complete.
  Session, retained-history and snapshot vector capacities are charged
  for their full lifetime; under a small budget, growth refuses or
  reclaims before allocation without corrupting a completed prefix.
  Repeated use and expiry across a growing model library leave no
  accumulating obsolete history capacity, and exact continuations survive
  reclaim and model switches. The memory breakdown explains these bytes
  without spending the fixed uncounted margin.
- **Formats:** every format in the approved covering set runs with the
  same correctness and speed criteria against its same-format reference.
  EXL3 decode and prefill are faster than ExLlamaV3's on the GB10, dense
  and MoE, at the covering set's bitrates, including mixed widths.
- **Media inputs:** for each model and modality, encoder outputs and the
  language model's teacher-forced logits on image, video and audio
  prompts match the same-format reference within bounds declared before
  evaluation, with preprocessing matched to the reference's pinned
  processor. Requests with several images, and video files, are among
  them. Generated text alone is not evidence. Inputs past their bounds
  are refused before any decode, and the intake passes its adversarial
  challenge.
- **Decision models:** Clef's, Clef-flash's and Clef-Omni's per-option
  probabilities match their pinned reference within declared bounds,
  text-only and with images (and, for Clef-Omni, audio and video). The
  TypeSafe SDKs complete requests unmodified. Decision requests to each model batch with each other, including requests with
  different prompt lengths and question sets; their probabilities meet
  the same bounds as solo requests. No state outlives a request.
- **Speech:** with the same inputs, seed and sampling, Breeze's codebook
  logits match its reference's within declared bounds (teacher-forced),
  and its codec turns given codes into the reference's waveform within
  bounds. Kokoro's waveform for given phonemes and voice matches its
  reference within bounds. Time to first audio and the real-time factor
  are at least as good as each reference's on the GB10 (D-085). Open
  WebUI speaks through `/v1/audio/speech` unmodified.
- **Generation routes:** Qwen-Image's generated and edited images from
  the routes equal the native pipeline's for the same seed;
  Ming-Image-0.1-Design's opaque and transparent images match its
  reference pipeline's for the same seed within bounds, alpha included,
  at least as fast on the GB10 (D-085). Both image models batch compatible
  work across concurrent requests. Batched outputs meet the same
  correctness bounds as solo outputs with each request's seed and
  settings preserved; controls cover differing sizes, step counts and
  cancellation of one peer. Report latency, aggregate throughput and
  memory at concurrency 1, 2 and 4 where memory permits, against each
  same-format reference, and record the measured memory limit and any
  phase or shape restrictions. Open
  WebUI generates through them unmodified, and through the WebUI routes,
  SillyTavern does too. Open WebUI shows an image returned in a chat
  response. Transcriptions from the audio carrier match its reference
  output for the same clips. MiniMax H3 text-to-video and image-to-video
  jobs each run from create to download and cancel cleanly. Through the MCP
  media server, Claude Code and at least one other coding agent show a generated image
  in a chat turn.
- The support matrix lists every approved family with its evidence.

## M3.6 — Engine components  `done`

Initial migration completed 2026-10-17; post-review capability-unification and
defect-fix slices R14–R19 completed 2026-10-20 (D-123, listed under
[M3.6 review follow-ups](#m35--model-families--in-progress) above and below).
Every model M3 served (Gemma 2, Gemma 3, Gemma 4
26B-A4B and 31B with their assistants, Qwen3.8 native with MTP and GGUF,
DeepSeek V4 Flash with DSpark, Qwen-Image-2.1) and the Qwen2 FP16 and EXL3
fixtures run from a stored spec on one engine of shared components over
llmpalooza's own graph IR, within the matched 1% speed gate of the frozen
`1facc957` baseline, bar the accepted exceptions listed in the record. The format layer keeps
weights in their source encoding and decodes on-chip, the native importer
publishes format-v1 artifacts from every migrated model's verified source,
serving is v1-only with approval as data (D-118), the legacy runners,
plans, graph builders, bindings and serving classes are deleted, and the
Python importers are out of the production path. The [M3.6 record](m36-record.md) retains the task history,
the exit criteria with their evidence, the owner's decisions, the accepted
exceptions, the classified review follow-ups and the work handed on. The
design is [engine-components.md](engine-components.md); the protocol and
baselines are [m36-baselines](experiments/m36-baselines/README.md) and the
final verification [m36-final](experiments/m36-final/README.md).

Goal (the owner, 2026-10-08; D-107 to D-110): replace the per-family
runners, plans, graph builders, bindings, serving classes and Python
importers with one engine of shared components over llmpalooza's own graph IR,
so every optimization, format and platform is written once and every model
benefits. Take the architecture hit now, before more families, formats or
platforms.

**Entry:** owner direction of 2026-10-08. M3.5 was parked at `cc70d69`; its
open families, formats and pipelines resume on this engine.

**Scope** (every box's sub-tasks, with their commits and reports, are in the
[record](m36-record.md#task-history-by-slice); the original nested scope text
is [retained there](m36-record.md#retained-scope-history)):

- [x] **Design and decisions:** [engine-components.md](engine-components.md),
      D-107 (components, IR, execution, partitioning, migration gate),
      D-108 (formats), D-109 (importer), D-110 (platform order); M3.5's
      transfer item retired with its [mapping](engine-components.md#retired-transfer-items).
- [x] **Baselines and the 1% protocol:** the matched, bookended protocol and
      the frozen `1facc957` baseline (`1facc957-fixed` with the DSpark
      cohort and swap-table continuation fixes), harness `tools/speed_gate.py`
      ([m36-baselines](experiments/m36-baselines/README.md)).
- [x] **Reference-model quality and numerical recovery** (owner, 2026-10-12):
      the numerical-policy audit, the shared teacher-forced scorer, the
      same-quantized-checkpoint references (Mia's vLLM for Qwen native, the
      old arm's per-target losses for Qwen GGUF and DeepSeek), image output
      pins per implementation, and the quality rules and bounds of the final
      verification ([audit](experiments/m36-numerical-audit/README.md),
      [m36-final](experiments/m36-final/README.md)). Comparisons against the
      original unquantized models are not required for same-quantized-model
      qualification and were not run; they stay optional for closer-to-original
      claims (M3.5, per family).
- [x] **Format layer** (D-108): encodings and layouts, CPU reference decoders,
      the fallback ladder, CUDA block decoders under shared vector, matrix,
      grouped and GLU templates, the `sm_86` proof, the EXL3 trellis through
      the same layer, and lossy state encodings end to end as a per-alias
      setting (`kv_cache_encoding = "q8_0"`, D-117) on every migrated LLM with
      its long-context quality measured (K1,
      [q8-state.md](experiments/m36-numerical-audit/q8-state.md)).
- [x] **Shared slot lifecycle, one runner interface, one serving adapter per
      pipeline kind**, with decoding modes over any number of drafters, tested
      on the fake backend with two drafters.
- [x] **IR and planner** (D-107): the op set, construction from a spec, fusion
      patterns covering the selected fused paths (the image's included), implementation
      selection, activation placement, lowering and plan keys, a CPU reference and
      fake-backend tests for every component.
- [x] **Partitioning in the planner:** TP, PP, EP and mixed layouts chosen by
      modelled traffic; equivalence on four or more fake devices. Two-Spark
      execution is M4's.
- [x] **Migrate Gemma2 and Gemma3**, then delete their runners, plans, graph
      builders, bindings and serving classes (G1).
- [x] **Migrate Gemma4** 26B-A4B and 31B with their assistants (G2).
- [x] **Migrate Qwen3.8**: native NVFP4/MXFP8 with MTP, and GGUF (G3), the
      legacy runner deleted. Open from it, moved to **M3.5**: the split NVFP4
      experts' own fast products (a grouped vector product for decode rows and
      the block-scaled grouped GEMM over split scales), and a relayout of
      MXFP4/MXFP8 routed experts and dense NVFP4 linears.
- [x] **Migrate DeepSeek V4 Flash** with DSpark (G4), the legacy runner, plans
      and graph builder deleted (G4d).
- [x] **Migrate Qwen2 FP16 and EXL3 fixtures** (G5): shared components and
      selected-operation lowerings, unserved, no speed baseline; the legacy
      executors deleted.
- [x] **Native importer** (D-109): readers for GGUF, safetensors with
      `config.json`, MLX, EXL3 and diffusers pipelines (Qwen-Image as a v1
      composition), architecture mappings, folds and relayouts recorded in the
      artifact, the v1 format (D-113), `llmp import` from the verified archive
      or a staged copy (D-114, D-116), every migrated model re-imported and
      admitted by an approval row (D-118), the adversarial challenge (N6).
      Moved to **M3.5**: the family-specific native readers (Qwen3.8's ModelOpt
      and MLX normalizers and the CLI's dispatch to them) generalized into
      mapping-driven normalizers when Nemotron NVFP4 brings a second ModelOpt
      model; MLX serving qualification (the checkpoint imports and is admitted,
      not yet qualified for serving); a Qwen3.8 drafter with another head list.
- [x] **Qwen-Image-2.1 onto the shared blocks**, with qualified numerical
      choices: the complete shared generation core, request-driven geometry and
      schedules, the fusions as planner patterns, the family as stored spec
      data; the legacy pipeline deleted. A stored per-geometry spec waits for
      M3.5.
- [x] **Structural techniques land once:** each technique landed in the
      component, pattern or lifecycle piece it belongs to as the migrations
      needed it, with every family's production selection reproduced and gated.
      The per-technique adoption audit against the retired transfer table was
      not run; the remaining adoptions move to **M9's** optimization pass
      ([optimization-inventory.md](optimization-inventory.md)), with the other
      performance leads (Qwen3.8 MTP activation sizing, about 12 against 9 ms
      of planning per new shape, the roughly 0.25 s workspace mapping in a
      Qwen3.8 swap, the decode-template leads).
- [x] **Persisted state and settings:** shared kept-state identity bound to the
      spec and the prepared execution identity, spec-bound calibration, and
      settings, applicability and defaults derived from the spec and its mapping
      (R1, `e3d7be83`; serving from data, `47cde2ed`, `d918ada4`); no
      per-architecture selector remains.
- [x] **Post-review capability unification and defect fixes (D-123, R14–R19):**
      fix the concrete IR, planner, component, spec, format, engine, importer
      and gate-tool defects found in the post-R13 review (R14, `575b04a4`); replace
      SHA-256 and geometry pins in `AdmittedNumerics`, `AdmittedImageNumerics`,
      `AdmittedPolicy` and `ResolveSettings` with capability-, format- and
      mapping-driven rules (R15, `c0cdf812`); replace family-specific dimension and `sm_121`
      whitelists in GGML capabilities, the on-chip decode ladder and EXL3 with
      true hardware/algorithmic predicates (R16, `d1356730`); unify attention placement,
      norm encodings, hybrid layer mappings and the 1D/4D spatial/token bridge
      (R17, `8936e175`); generalize parameter folds across `F32`/`F16`/`BF16`,
      role-driven `RelayoutExpertTables`, MLX MTP drafter normalization and
      single-pass chat boundary tracking (R18, `8b3eafdd`); and unify
      `BackingService` 2 MiB handle pooling (`RE-059`), 16 GiB workspace virtual
      reservations, base-leased routed expert closures and engine runner
      boilerplate with closeout verification (R19, `fa1d9073`).

**Exit criteria:** met, with the accepted exceptions (DeepSeek DSpark's C4 wall and the Gemma 3, Qwen3.8 GGUF and Qwen3.8 native quality tripwires), each with its evidence, in the
[record's mapping](m36-record.md#final-gate-and-evidence-mapping): one engine;
speed within 1% of the recorded baselines (final results in the
[record](m36-record.md#final-speed-verification)); quality and state against
the same-quantized references with the swap table and the Gemma near-8K
switch controls; formats and the `sm_86` proof; state encodings; import,
including the MLX admission and the adversarial challenge; partitioning on the
fake backend; and the Spark check set, with its skipped tiers named
(the shared set 2,839 of 2,839, the CPU-only guardrail 2,273 of 2,273 and the 214 model and timing tests all passed on `53c2610b`). Whole shipment tiers remain owed before publishing a
package.

**Handed on:** M3.5 resumes on this engine (below), physical memory modes first; M4 gains the deferred item
on concurrent models and a shared pageable workspace; M9 takes the
optimization leads.

## M4 — Two-Spark fast full swap  `pending`

Goal: the same cycle for models too big for one Spark, sharded across both.
Each node holds its shard on disk, and the conductor loads both shards at
once.

**Entry:** M3.5 exit (M3 exit before 2026-09-29). By entry, model-parallel artifact partitioning is
decided ([artifact-format.md](artifact-format.md#deliberately-open); moved
from M8's entry), and each model's checkpoint, recipe and baselines are
pinned and audited as in M3. That decision includes the partitioning
choice the planner costs for every MoE it places (moved ahead of entry by
the owner, 2026-10-20): expert parallelism with replicated attention (one
all-reduce a layer, at the cost of replicated attention weights) against
TP (two collectives a layer). Before entry, lift M3.6's rule that a TP
stage never mixes split and replicated layers, give DeepSeek's attention
component split options (V4.1 has one KV head, so by query heads with that
head replicated), and estimate both layouts for V4.1 and GLM-5.3 from
measured collective latency on the two Sparks and each model's weights
and state; M4 then confirms the choice end to end
([assessment](deepinfer-assessment.md#6-expert-parallelism-with-replicated-attention)).
First measure captured in-graph NCCL all-reduce at 8–64 KiB against a
minimal one-shot RDMA prototype on `spark` and `spark-b`
([Flare](flare-assessment.md#6-two-node-transport-and-the-ep-vs-tp-decision-m4-entry)).
If one-shot lands near the RDMA write latency (an estimate until that
measurement), the collective count stops deciding, and the
estimate prices bytes read per node (about 242 GB/s measured), the
expected busiest node's routed experts from real routing traces, foreign
expert slots, and paging granularity (EP keeps whole experts as paged
groups; TP splits each), using the planner's execution-cost objective,
not traffic alone. Entry also adds an intermediate-column split option
to the MoE component, without which TP cannot be expressed.

**Models, in this order:**

*Note, 2026-09-29:* the recipe moves fast; re-pin it at M4 entry. Its
2026-09-28 changes (#281 FP8 on most of the path with KDA in BF16:
decode 3.9–5.2% and 16K/64K prefill TTFT 11.5–12.6% faster; #292
`GLM53_MODEL_PRESET=dense-h3`: dense EXL3 "H3" targets with matched
6-bpw DFlash2 drafts, up to 16% decode, estimated; all
creator-reported, @plotarmordev on X) are techniques to study: whether
the FP8 split is a default or a quality mode under D-085's note, and
quantized drafters matched to EXL3 targets (M3.5's EXL3 work).

1. **GLM-5.3 Flash** (`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks@c1b7d4c`;
   EXL3 at about 4 bpw, TP2, about 80 GiB of weights per node,
   creator-reported): KDA linear attention, DSA sparse MLA with an indexer,
   mHC, and 288 routed experts.
2. **DeepSeek v4.1 Flash** (Mia's EXL3 at 2.9 bpw, TP2, 99.5 GiB of weights
   per node, creator-reported): a 552B encoder-decoder with CSA2 attention
   and a hierarchical indexer, which no GGML code covers. Its ~190 GiB of
   Engram lookup tables may stay row-paged from each node's SSD (D-035).
   *Owner update, 2026-10-17:* target the updated
   [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
   (MIT), a "minor update" of V4.1 Flash with substantial reasoning, coding,
   agentic and visual gains and shorter outputs; the card names no
   architecture change (1M context; FP8/BF16 safetensors; it reports 763B
   parameters, which may count the Engram tables). At entry: confirm the
   architecture matches, pick a 1017 quant that fits two Sparks (Mia's EXL3
   if re-made from 1017, else one of the card's linked quants) with its
   same-format reference, and keep the original V4.1 Flash only as a
   fallback if no 1017 quant fits.
   *DeepInfer, 2026-10-20:* DeepSeek's own engine runs 1017; its
   [assessment](deepinfer-assessment.md#v41-flash-as-deepinfer-implements-it)
   lists the shapes V4.1 adds over V4 Flash. Each lands as a parameter or
   capability of a shared component, never a V4.1 path (D-107;
   [generalization](deepinfer-assessment.md#generalization)): compression
   ratio, position embedding and overlap as compressor parameters, the
   indexer's key source, head count and transform as parameters,
   cross-layer KV and top-k sharing as graph edges any mapping may
   declare, hierarchical candidate blocks as an indexer stage, YOCO
   consumer layers as a mapping of shared KV sources, shifted
   hyper-connections as an HC variant, Engram as a token-keyed row-table
   component, a per-modality routing bias, and DSpark stored as `mtp.*`
   through the existing drafter mapping.

*Owner addition, 2026-10-16:* a second GLM-5.3 Flash quant,
[`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local)
(NVIDIA Open Model Agreement over GLM's MIT license; recorded, never
blocking, D-087), served in the same cycle as the EXL3 target (AGENTS.md
rule 11: every quant runs). It is ModelOpt QAD: NVFP4 routed experts,
MXFP8 decoder attention, NVFP4 MTP experts with BF16 activations, and a
vision tower (MXFP8 attention, NVFP4 MLP and merger). It uses the same
lossless compression of NVFP4 scale factors (probably CSF, as
[Flare](flare-assessment.md#1-nvfp4-scale-compression-csf-for-fp4-local)
describes it; unconfirmed)
as M3.5's
[Qwen3.8 FP4-Local](m35-families.md#owner-requested-addition-qwen38-flash-next-fp4-local)
(that work lands first and is reused here). The card validates two DGX
Sparks over ConnectX with an FP8 KV cache to 2M tokens (about 166 GiB of
weights, creator-reported). Reference: the card's community vLLM
configuration, or the latest TensorFold if it supports the checkpoint at
M4's entry re-pin.

**Oracles and comparators,** under M3's rule (cross-quantization is speed
and memory only):

Also study and measure the owner's additional
[Kindling GLM reference](m4-references.md), pinned provisionally on
2026-09-29. Its NVFP4 TP2 recipe is a speed/memory comparator for the
EXL3 target; re-pin and audit it at entry, and distinguish its dense-layer
requantization from optimizations at unchanged quality.

| Model | Correctness oracle (same format) | Performance comparators | Cross-quantization (speed and memory only) |
| --- | --- | --- | --- |
| GLM-5.3 Flash | Same-format EXL3 configuration, re-pinned at entry | Latest TensorFold where the EXL3 layout matches; Mia's configuration | TensorFold (MLX affine) |
| DeepSeek v4.1 Flash | Mia's configuration | Mia's configuration | none |
| GLM-5.3 Flash FP4-Local | The card's vLLM configuration, or TensorFold if it supports it at entry | Same; the EXL3 target's results; Flare at TP2 if its CSF-QAD checkpoint is the same weights, its default quality trades stated ([assessment](flare-assessment.md#as-a-comparator)) | none |
| Flare's pinned GLM-5.3 CSF-QAD and DeepSeek V4 Flash lossless-CSF (M3.5's comparator item) | none: Flare's FP8 KV and 4-bit prefill activations make it a speed and memory comparator only | Flare at TP2 on the same checkpoint, quality trades stated | none |

**Scope:**

- [ ] **Provenance and licenses:** re-audit the GLM recipe if its baseline
      moves past the pin (HEAD is 65 commits ahead); record the GLM
      checkpoint mirror's license and the DFlash2 drafter's (CC BY-NC-ND
      4.0) for information (D-087); Mia's E3 and cooperative MoE kernels
      (AGPL or mixed, D-080); and whether Mia's ExLlamaV3 revisions'
      formats match our `6b84a21`.
- [ ] **Baselines:** Mia's two-Spark configurations and TensorFold, run by
      us, measured as in M3. Refresh TensorFold at M4's entry and exit
      under [reference comparisons](reference-comparisons.md).
- [ ] **Conductor, minimal** (pulled from M6a): one configured two-node
      topology that loads both shards at once and runs each phase on both
      ranks. Discovery, enrollment, cluster trust and general placement
      stay in M6a. It is development-only (owner, 2026-09-28; D-087): it
      runs over the direct Spark-to-Spark link, trusted like loopback, is
      off by default, and is documented as development-only, not a
      supported deployment. Mutual TLS (D-038) arrives with M6a (D-014).
- [ ] **Sharded execution** (pulled from M8): TP2 over NCCL with
      conductor-issued distributed phase IDs, a separately budgeted
      communication-buffer pool that honors NCCL's registration and
      threading contracts, ordered collective submission and completion
      fences, and per-step collective latency measured before bandwidth.
      Each model runs the expert-parallel or TP layout chosen at entry,
      confirmed end to end. Collectives go through a provider with NCCL
      and, if entry's measurement shows collective latency above about 5%
      of step time, a host-staged one-shot RDMA implementation, chosen per
      op and size by quick A/B (D-053); one transport per node, registered
      once at a stable address and bound by every model's graphs, so a
      swap never pays NCCL or queue-pair setup; a vocabulary-parallel
      greedy pick ([Flare](flare-assessment.md#6-two-node-transport-and-the-ep-vs-tp-decision-m4-entry)).
- [ ] **Coordinated readiness, minimal** (the start of M8's coordinated
      admission): no collective or execution of B begins until both ranks
      report ready, their shard loaded and its memory obtained. A
      preparation failure on either rank, or no report within the stated
      timeout (default 30 s, owner-adjustable), aborts the swap: the other
      rank releases or rolls back what it prepared within that timeout, no
      rank enters a collective alone, and the swap reports failure cleanly.
      Memory a rank has not confirmed released stays counted as held; no
      timeout proves reclaim. General recovery, prepare/commit and the
      broader pressure and failure matrix stay in M8. After Flare: ranks
      agree on identity (artifact digest, graph I/O schema, state-layout
      hash) and capacity (the minimum across ranks) before ready; a GPU
      collective wait is bounded (D-102's hang detection, well above any
      healthy wait) and writes an error word, so a dead peer becomes an
      error, not a wedged GPU (with NCCL, an async-error
      watchdog and `ncclCommAbort`); transport health is checked before
      completions are published; rank 0 stops first.
- [ ] **Kernels:** EXL3 MoE, the codebooks and rates these checkpoints use,
      sparse MLA decode and prefill on `sm_121`, KDA, the DSA indexer, CSA2
      and Engram row gathers, each the fastest correct implementation under
      D-053, chosen by a quick A/B. Also, as shared capabilities:
      - **Engram on the token-keyed row read-ahead** M3.5 measures and
        builds for Qwen3.8 (or, if Qwen3.8's share did not justify it,
        built here for Engram's 190 GiB tables): Engram's key function
        declared in its mapping, with the hash and gate formulas from
        M3.5's TileKernels audit and the compressed-vocabulary table built
        at import; across nodes, tables split by hash head, each node
        reading only its heads' rows and joining one all-reduce, as Flare
        does, so no rows cross nodes (a row-table split option)
        ([assessment](deepinfer-assessment.md#4-token-keyed-row-read-ahead),
        [Flare](flare-assessment.md#6-two-node-transport-and-the-ep-vs-tp-decision-m4-entry)).
      - **FP4 indexer scoring** (e2m1, UE8M0 per 32, BF16 logits;
        DeepSeek's production numerics) as an operand encoding of the
        shared sparse-scoring operation, selectable for every indexer that
        uses it (V4, V4.1, GLM-5.3's DSA), qualified per model against F16:
        the default only at the same quality, otherwise a per-alias mode
        (D-085; [assessment](deepinfer-assessment.md#5-fp4-indexer-scoring)).
      - **FP8 MoE activations and quantized all-to-all** (leads from
        [Humming](humming-assessment.md#m4-leads-fp8-activations-for-moe-and-quantized-all-to-all);
        vendor figures only): an MXFP4-expert product with an FP8
        activation operand for V4.1-Flash, an opt-in per-alias activation
        mode qualified against F16 (rule 11), measured after M3.5's MXFP4
        screen; an FP8 dispatch only as a measured input to the EP-versus-TP
        decision above.
- [ ] **Per model,** as in M3: graphs, state adapters, templates, the
      resident expert layout for EXL3 MoE, and speculation with M3's
      forced-rejection checks where the model supports it (for GLM, MTP or the
      DFlash2 drafter that Mia's GLM configuration uses, whichever is faster
      and correct; the
      nextn layers or DSpark head for v4.1). ReconGemm prepares its
      cuBLASLt descriptors once per bound GEMM (M2's hand-off).
      Two shared capabilities this needs:
      - **Confidence-budgeted verification** (M3.5's calibration screen
        decides whether the shared speculation scheduler has it by then):
        V4.1's DSpark reports its head's per-row confidence from its first
        run through the shared drafter mapping, and uses the scheduler's
        budgeting if it landed; if V4's head failed calibration, screen
        V4.1's head the same way
        ([assessment](deepinfer-assessment.md#1-confidence-budgeted-verification)).
      - **Row-liveness pruning of prompt rows** in the planner,
        generalizing T54: from the graph's own dependencies (which layers'
        KV each layer reads, windowed attention's reach, drafter taps and
        requested outputs, scores and in-prompt state snapshots), compute
        per layer the rows any later consumer needs and execute only
        those, exact against full prefill (no approximation). V4.1's
        YOCO consumer layers (20–39, about 2.5K rows at layer 20 narrowing
        to about 134 at layer 39; estimated up to about 2× long-prompt
        prefill) are the large case; trailing windowed layers after a
        model's last global layer (Gemma) are screened when it lands
        ([assessment](deepinfer-assessment.md#3-row-liveness-pruning-exact-yoco-truncation)).
        The same pass also offers an approximate mode, opt-in per alias
        and off by default (owner, 2026-10-20; D-085, AGENTS.md rule 11):
        a configured keep count replaces the exact cone, so windowed
        layers past the cut see only the kept rows (DeepInfer's YOCO
        truncation, which DeepSeek runs in production). It applies to any
        model the exact pass applies to, is qualified per model (long-
        context quality against the exact pass, and measured speed), and
        is offered only where it is measurably faster than the exact cone
        at an acceptable quality cost
        ([assessment](deepinfer-assessment.md#3-row-liveness-pruning-exact-yoco-truncation)).
- [ ] **Media inputs** (D-101), on M3.5's intake:
      - GLM-5.3 Flash's images and video: the GLM ViT (24 layers, patch
        14, 2D RoPE inside), video at 2 fps with per-frame timestamps;
      - DeepSeek V4.1 Flash's images: the DeepSeek-ViT with its 3×3
        aligner, at most 1,024 tokens an image in reading order.

      Both checkpoints keep their towers. Mia's V4.1 build says vision was
      not its validation target, and llama.cpp has no V4.1 architecture,
      so V4.1's oracle is its repository's own inference code. That code
      is now [DeepInfer](deepinfer-assessment.md), which runs only on SM100
      and Ascend: at entry, either obtain teacher-forced references from it
      on such hardware or use it as read-only reference code beside Mia's
      configuration, and say which.
- [ ] The minimal `/v1/chat/completions` serves the sharded models.
- [ ] **Concurrent models and the shared workspace** (deferred from M3.6,
      [record](m36-record.md#deferred-items-from-the-reviews)): the node's
      shared pageable workspace (activations and the GGML pool) pages with
      the active model and every model draws on it from its base, so a shared
      pageable workspace rules out two models executing concurrently on one
      node. Revisit with placement: a workspace per executing model, or
      disjoint ranges of one, before the conductor places two models' work on
      one node at once (D-019 to D-025). Also from M3.6's partitioning: a TP
      stage never mixes split and replicated layers (lifted before entry,
      per Entry), the device cap is 16, no
      embedding split by vocabulary, expert parallelism is local experts plus
      a mask plus one all-reduce (no all-to-all, no DP-attention mode), and
      the lowering's per-runner backend and stream assumptions
      ([engine-components.md](engine-components.md#partitioning)). The
      single-resident-model assumption is wider than the workspace
      (`Server::resident_`, the cohort model, one swapped model and its parts,
      `SwapBytes`, `SpillView`; the M3.6 review of 2026-10-20), and the
      partitioner is a CPU model nothing executes yet (no non-reference
      collectives; mqa, hc and ple declare no splits). M3.5's U9 (owner,
      2026-10-20) lands here before entry: `PlanKey::partition` as a digest
      of the assignment and device rank rather than a bare index,
      non-reference collectives, split declarations for `attention.mqa`,
      `hc.*`, GDN and the routing and pooling ops, rewrite rules for the ops
      `planner/partition.cc` cannot split today (`kGatedDeltaNet`,
      `kSsmConv`, `kPool`, `kIndexScore`, `kHc*`, `kRoute`), and per-model
      residency records replacing the single-model fields above.

**Exit criteria:**

- **Swap time,** as in M3 across both nodes, with its owner-adjustable
  defaults:

  | Item | Rule |
  | --- | --- |
  | Pairs | GLM→v4.1→GLM and v4.1→GLM→v4.1, each swap reported separately; the headline is the worse |
  | Saved context | A holds 8K tokens; its KV, recurrent and indexer state on both ranks are spilled and restored, and count. The bound applies here; a 0-context swap is reported too |
  | Graphs | As in M3: the ~10 s goal and ~20 s bound apply to previously prepared swaps; first use no worse than about 2× the bound |
  | Endpoint | From the swap request to B's first generated token for a short prompt |
  | Also per swap | Per node: bytes read, read throughput and peak memory; the time between the two ranks' ready reports |

  The report sets each swap beside the baselines, Mia's GLM among them at
  65–70 s cold (creator-reported).
- M3's correctness, speculation, performance and memory criteria against
  each model's oracle and comparators above, with memory judged per node.
  Swap and restore are bit-identical on both ranks.
- **Readiness:** a load failure and a memory failure injected on one rank,
  each on either rank in turn, start no collective, the other rank rolls
  back within the timeout, the swap reports failure cleanly, neither
  catalog keeps a lease, grant or backing from B, and the next swap
  succeeds.
- **Media inputs:** M3.5's media-input criterion holds for both models'
  modalities.

## M5 — One resident model, end to end  `pending`

Goal: import the small fixtures, serve them resident through native GGML and
EXL3 execution, and complete chats from named clients over the three
baseline protocols, with bounded, explainable memory.

**Entry:** M4 exit. The native tokenizer, stop rules and sampling come from
M3.

**Scope:**

- [ ] **Importer and verifier** (D-009, D-056, D-109): M3.6 builds the
      native C++ importer; here it becomes a product, with the standalone
      verifier, running on the workstation and on a Spark.
      Confined import jobs take sources only from the configured stores
      (the `checkpoints` role or the long-term store; D-054, D-063) or the
      Hugging Face Hub (resumable and verified; the token comes from a
      `.env` or config file and is never logged); `llmp install` fetches
      into the [model archive](model-archive.md) with libcurl, writes its
      manifest and catalog events, then imports (D-114). Hard-linked and symlinked
      sources are rejected, and the user docs say to copy them in or
      download with `--local-dir` (D-063). Removal waits for
      quiescence. Jobs keep their records, locks and grants and report
      progress, status, cancellation and retry (D-041). An install that
      lacks space fails before transferring, reporting the space it needs
      and the removal candidates.
- [ ] **Registration and first use** ([model lifecycle](architecture.md#model-lifecycle)):
      the compact installed-model index cache, shallow checks on every
      startup, and detail and plans built on first use inside an `F`
      reservation. Startup rejects an installed store that fails the
      direct-I/O probe (D-054).
- [ ] **Resident execution:** the FP16 and both EXL3 fixtures under llmpalooza
      dispatch, finite context and chunk profiles within the 8K context,
      and state block sizes and KV layouts for each adapter. Decode graphs
      and their relocation proof come from M3.
- [ ] **Tokenization and output:** D-067 renderers for both fixture
      template hashes on M3's native tokenizer, stop rules and sampling
      (moved to M3), and the tool-call parser (moved to M3.5 by the owner,
      2026-10-20, with Messages; here it serves the Responses route too);
      host versus device sampling is settled against D-052's decode
      gates.
- [ ] **Front door** ([M5 surface](client-api-baseline.md#m5-surface)),
      growing from M3's minimal Chat Completions:
      Chat Completions, stateless Responses, Messages with token counting
      and the Claude Code profile (both moved to M3.5 by the owner,
      2026-10-20; re-verified here), `/v1/models` in both shapes with
      D-046's metadata, and the discovery
      document (D-041); D-045's listener, authentication, CORS, `Host`,
      status and keepalive rules; D-047's non-streaming and storage rules;
      and explicit rejection of `transforms` and
      `plugins`. Numeric intake bounds are fixed before any external input
      is accepted, with the total input limit reconciled with named-client
      tests ([cluster design](cluster-design.md)).
- [ ] **Gemini API and fill-in-the-middle** (D-101, the owner,
      2026-10-02):
      - Google's Gemini API (`v1beta/models`, `:generateContent`,
        `:streamGenerateContent?alt=sse`, `:countTokens`, image
        `inlineData` in and out) for Gemini CLI and the google-genai SDKs
        by base URL.
      - Code completion with a suffix: `suffix` on `/v1/completions` (and
        the Ollama profile's `/api/generate` in M10), Mistral's
        `/v1/fim/completions` and llama.cpp's `/infill`, on models whose
        tokenizers carry FIM tokens, confirmed at entry.
- [ ] **Local management API and CLI** (D-064): versioned routes for import,
      listing, representation inspection, removal, jobs and node health.
- [ ] **Service hardening** (D-074; M1's hand-offs, moved from M2 by the
      owner on 2026-09-27): a system-call filter for `llmp.service`
      that admits io_uring, and a single reaper for jobs started from
      other threads.
- [ ] **TLS** (D-065): per-name certificate files selected by SNI, reload on
      change, key-match and expiry checks, the name-constrained local CA,
      the certbot deploy hook, and the Tailscale certificate timer. The
      root-run hook and timer touch only `/etc/llmp/tls/`, their units are
      sandboxed to it, and they take the heavy path.
- [ ] **Surface definitions:** front-door, alias and TLS configuration keys;
      the individual `llmp-` header and body-field names (D-062); the
      HTTP, TLS and JSON libraries, chosen under D-017, D-057 and D-066.
- [ ] Move the fixtures' rows in the model support matrix (started in M3,
      where they are listed as fixtures) to served, with their templates.

**Exit criteria:**

- Teacher-forced logits and declared intermediates for the three fixtures
  match their pinned references within bounds declared before evaluation
  ([first-slice.md](first-slice.md), [exl3-bringup.md](exl3-bringup.md)).
  Rendered bytes and token IDs match the golden fixtures.
- EXL3 resident prefill, time to first token and decode inter-token
  p50/p95/p99 meet the predeclared upstream parity bounds against upstream
  serving controls in matched and normal views, with memory inside the
  declared bounds; a regression needs a fix or an explicit owner-approved
  tradeoff (D-052).
- Each engine, served resident, is at least as fast as its reference end
  to end (GGML against llama.cpp, EXL3 against ExLlamaV3; D-085's coarse
  comparison), with BP-F3's resident timings reported in it (moved from M2
  by the owner, 2026-09-27; BP-F4's per-token host cost is measured in
  M3).
- The [M5 acceptance cases](client-api-baseline.md#acceptance-owed-in-m5)
  pass as scoped there. Gemini CLI completes a chat unmodified through
  the Gemini API, and Continue or llama.vscode completes code through
  fill-in-the-middle, with the inserted text equal to the model's greedy
  output for the same prompt. At least one named client completes a chat
  unmodified with each representation, and Cursor stays an explicit gap
  unless resolved. The context-compacted release moves to M6, which
  delivers D-041's close, and the Ollama-native checks move to M10 with the
  Ollama profile.
- Finite default context and output bounds bound every admitted request,
  and the M5 rows of D-050's matrix pass, with the parts moved to M5
  (suballocation holes, stalled-client termination, every queue full at
  once, capacity-loss injection). Memory use is bounded and
  explained by the memory breakdown.
- The importer, verifier and front-door parsers pass their adversarial
  challenge; an interrupted import never appears valid.
- Measured and recorded: startup time and metadata memory as the installed
  library grows, the cold-switch cost of on-demand detail, and both
  contexts' state bytes, from which D-055's
  [capacity values](retention-policy.md#bounds-and-defaults) are pinned.

## M6 — First useful product: A→B→A with partial retention  `pending`

Goal: two small model contexts share one local budget. Switching to B
displaces only what B needs, retained conversation state lets A resume
without a full re-prefill, and the switching policy is chosen from
measurement. Useful without MoE or sharding. M3 already restores one
model's state across a full swap; M6 adds partial retention and the
retention policy around it.

**Entry:** M5 exit with its capacity values pinned, plus everything
[M6 entry pins](retention-policy.md#what-m6-entry-pins): the frozen
transcript, budgets and reference paths among them, and the spill write
budget from the drive's rated endurance. EXL3 switching budgets come from
new matched controls ([exl3-bringup.md](exl3-bringup.md)).

**Scope:**

- [ ] **Retention** ([policy](retention-policy.md); D-024, D-031, D-055):
      prefix and continuation entries with their identity, restore
      boundaries, branches, sharing and refresh; capacity-driven expiry with
      7-day idle caps; the victim-order baseline; spill with its protected
      directory, preallocation, direct I/O, digest check on restore and
      deletion at startup; miss reasons and fallback reporting. Prefix
      sharing grows from M3.5's minimal copy-based reuse; it is generic
      over the state contract, after DeepInfer's split
      for compressed attention: state a contract marks immutable once
      written (full and compressed KV, indexer keys) is shared in blocks
      keyed by a token hash chain at the contract's grain, reference-counted
      and charged once (D-006, D-007); mutable state (windows, recurrent
      state, compressor carries) is snapshotted per boundary; a fork copies
      only the tail block; and a prefill chunk may end at a shared prefix's
      end. Continuations keep their exact-position checkpoints
      ([assessment](deepinfer-assessment.md#7-shared-prefix-cache-generic-over-state-contracts)).
- [ ] **Partial eviction** of a quiescent model, reloading only missing
      dependencies (D-008).
- [ ] **Swap pipelining and load order** (owner, 2026-10-02): explore
      overlapping page-in with execution, so each layer runs once its
      extents are resident instead of after the whole closure (graphs wait
      on per-layer load completion or split per layer), and order page-in
      by first use: dense and attention weights and restored state before
      later layers, a drafter (DeepSeek's DSpark, Qwen3.8's MTP) after the
      first token. Measure time to first token after a swap at several
      prompt lengths, cached continuation included, against M3's
      load-then-run full swap (DeepSeek's page-in is about 8 s of a
      9.7 s worst swap,
      [fast-swap](experiments/fast-swap/swap.md)); keep what pays.
- [ ] **Concurrency when it fits:** all-resident cohorts under
      full-envelope checks, reporting whether requests ran concurrently or
      time-sliced.
- [ ] **Switching policies** (D-069): the priority-aware default,
      run-to-completion and time-slicing with their guards, their
      configuration keys and per-alias overrides.
- [ ] **Request control** (D-042): interactive and background classes,
      maximum queue waits, cancellation and bounded progress events.
- [ ] **Release** (D-041, D-045): the final-turn flag, idempotent
      continuation close and the context-compacted release, with their wire
      names fixed here.
- [ ] **Warm jobs** (D-041): capacity-constrained, never an implicit
      download.
- [ ] **Diagnostics:** status, the admission what-if query, Perfetto trace
      export and eviction explanations; management controls for priorities,
      residency policies and trace capture; Prometheus `/metrics` with
      compatible health and load queries (D-044).
      - [ ] **vLLM-compatible Prometheus `/metrics` for sparkDash** (owner,
            2026-10-22; D-044): this item's first slice.
            [sparkDash](https://github.com/MiaAI-Lab/sparkDash) (read at
            `df6a223`) has no plugin or push API. Its `LlmProbe` polls each LLM
            port, identifies the engine by `owned_by` on `/v1/models` and by
            endpoints unique to each engine, and otherwise treats it as vLLM,
            whose `/metrics` it diffs per poll into decode and prefill tok/s,
            requests, KV use and TTFT/ITL p95. Llmpalooza (`owned_by: "llmp"`)
            already lands on that fallback but serves no `/metrics`, so its card
            reads 0 tok/s; vLLM-named series fix that with no sparkDash change.
            - **The route:** `GET /metrics` on the chat listener (the port
              sparkDash probes), Prometheus text format 0.0.4, behind the
              listener's authentication when it has one. Rendered on the I/O
              thread from a snapshot the driver publishes at each completed unit:
              no wait on the driver, no catalog or scheduler lock (D-007). Counts
              and alias names only, never request or output text (D-014). Labels
              are bounded: `model_name` takes the configured aliases (`[models]`),
              and every registered LLM's series are written whether or not it is
              resident, so counters never vanish across swaps.
            - **vLLM-named series, only where the meaning matches vLLM's**
              (D-044). Names, types and histogram buckets are checked against a
              pinned vLLM release's metrics source when built, not from memory
              (rule 3). Where each count comes from:
              - `vllm:prompt_tokens_total` (counter): the turn's reused tokens at
                admission (`Usage::cached_tokens`, `api.h`, set from the branch's
                reuse in `service_backend.cc`), then each prefill chunk's rows as
                it completes (`RunPrefillChunks`, `prefill.cc`; joined waves in
                `llm_execution.cc`), so a long prefill shows a live rate rather
                than a jump at its end. Its total is the prompt's tokens, cache
                hits included, as vLLM counts them.
              - `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total`
                (counters, in tokens): the turn's prompt tokens and its reused
                tokens, at admission. sparkDash subtracts hits, so reuse never
                reads as prefill speed.
              - `vllm:generation_tokens_total` (counter): the tokens each unit
                commits, counted at the driver's commit (a plain step's token, a
                verify's `kept`, `llm_execution.cc`), not when the response ends.
              - `vllm:num_requests_running`, `vllm:num_requests_waiting` (gauges):
                `api::Server`'s `running_` and `queue_` (`api_server.h`); a
                request yielded to a pending switch is queued again, so it counts
                as waiting.
              - `vllm:kv_cache_usage_perc` (gauge, a 0–1 fraction as vLLM's): for
                each model with live state, its active branches' committed
                positions over `slots_` × `usable_context()` (`serving.h`);
                absent for a model without live state.
              - `vllm:time_to_first_token_seconds`,
                `vllm:inter_token_latency_seconds` (histograms): from the
                channel's `queued_at` to the stream's first emission, and between
                later emissions (`api_stream.cc`).
              - `vllm:spec_decode_num_drafts_total`,
                `vllm:spec_decode_num_draft_tokens_total`,
                `vllm:spec_decode_num_accepted_tokens_total` (counters):
                `Llm::Counts`' `speculative_units`, `drafted` and `accepted`
                (`serving.h`), counted already.
            - **Not published under vLLM names** (the meanings differ):
              `num_preemptions_total` (a request yielded at a model switch keeps
              its state; it is not evicted and recomputed), `cache_config_info`
              (our KV backing commits lazily and has no fixed block pool), and
              `iteration_tokens_total`.
            - **Llmpalooza's own series** (`llmp_`, D-044's separate paging
              measurements; sparkDash ignores them): `llmp_build_info{version}`;
              `llmp_model_resident{model_name}`; `llmp_request_yields_total`;
              `llmp_swaps_total`, a `llmp_swap_seconds` histogram and
              `llmp_swap_read_bytes_total` (`SwapParts`, `serving.h`); spills,
              restores, their bytes and seconds (`Llm::SpillStats`);
              `llmp_page_in_events_total` (`Server::page_in_events()`); and the
              node budget with the memory charged against it (`Server::budget()`
              and its parts).
            - **Known limits, left to sparkDash:** it sums every label set of a
              series, so two models resident at once add their KV fractions, and
              it names the port's model after the first `/v1/models` entry
              rather than the resident one. An `llmp` backend in sparkDash keyed
              on `owned_by`, as it has for Radiance, fixes both; contributing it
              upstream is the owner's call, and the runtime's `/v1/models` order
              does not change for it. The listener binds loopback and the tailnet
              by default, so sparkDash's per-Spark `llmHost` names the tailnet
              address unless the owner binds another (runtime-serving.md).
            - **Checks:** unit tests on the exposition text (types, `HELP`, label
              escaping, counters monotonic across a swap and a yield, no request
              text), the counters against `Usage` on fixture turns, and a
              `models` test that reads `/metrics` around a short chat and a swap
              on a Spark. Heavy path: a public route on the listener.
              **Done** when a sparkDash instance pointed at a Spark's runtime
              shows decode and prefill tok/s, requests, KV use and TTFT/ITL p95
              on its LLM card (recorded with the sparkDash commit), and
              runtime-serving.md and the CHANGELOG describe the route.
- [ ] **Storage scheduling:** demand reads mixed with spill write-back,
      inside the pinned write budget.
- [ ] Add the A→B→A workload and its regression thresholds to `check:spark`.
- [ ] **Discrete GPU, secondary** (D-082; after the two-Spark swap, D-087):
      the A→B→A fast swap, one model
      active and partial retention within device memory, on the
      workstation's discrete GPU (`mise run test -- native --gpu`), with
      its PCIe restore rate reported, within a configured device budget and
      with the landing zone reported apart from it. Whole-model swaps and
      paging within device memory only; no on-demand expert paging from the
      SSD there. Not an exit criterion.

**Exit criteria:**

- D-055's [timed workload](retention-policy.md#m6-acceptance-workload)
  passes its pass rule: Qwen2.5-0.5B FP16 and EXL3 4.0 bpw in both
  orientations, six llmpalooza arms against fresh interleaved references, 72
  accepted repetitions per arm, orientation and cache condition. For each
  floor arm (J-partial, J-spill), orientation, direction and cache
  condition, at the median and at p95, llmpalooza's one-sided 97.5% upper bound
  is at most the smallest one-sided 97.5% lower bound among the valid
  reference arms; a comparison with no valid reference arm does not pass
  (D-036). The report covers latency distributions, bytes read and written,
  peak memory and spill, prompt tokens reused versus recomputed, and deltas
  against llmpalooza's whole-model control (M9's comparator).
- The [correctness gates](retention-policy.md#correctness-gates) pass: exact
  outputs and bit-identical teacher-forced logits against
  provenance-matched controls, and catalog state and events show that only
  selected extents were displaced.
- The [functional and adversarial cases](retention-policy.md#functional-and-adversarial-cases)
  and the M6 rows of D-050's matrix, with the parts moved to M6 (fork and
  copy-on-write, cached-state promotion), pass
  with an EXL3 context in the
  matrix; cache expiry never destroys admitted suspended work.
- Through at least one unmodified named client: a long conversation on A,
  B under pressure, then A resumed, over both resident reuse and forced
  spill/restore. The context-compacted release case deferred from M5
  passes.
- B arriving while A is still generating is measured under each D-069
  policy, with queue delay reported apart from paging and switch time and
  from first-token compute, together with pauses and bytes reloaded. The
  owner keeps or changes the default on these results.
- An all-resident control shows concurrent progress when both complete
  envelopes fit.

## M6a — Configured placement across nodes  `pending`

Goal: one configured conductor places whole models on enrolled nodes and
routes requests with retained-state affinity, so a subagent's model runs on
the other Spark while the main model stays resident. It follows M6,
independently of M7, and grows M4's minimal two-node conductor into the
cluster's placement layer; sharding under pressure and failure is M8.

**Entry:** M6 exit.

**Scope:**

- [ ] **Setup and discovery** (D-038, D-039, [cluster design](cluster-design.md)):
      inventory, trusted SSH, the mDNS window, layout classification,
      bounded dedicated-QSFP subnet scans, one-time enrollment and path
      re-detection for enrolled members.
- [ ] **Configuration and trust:** the shared and node-local v2 documents,
      the private CA with TLS 1.3 mutual authentication, epochs, session
      fencing and restart reconciliation, and protocol v1 with its bounded
      state, schema tests and exact per-message field catalog.
- [ ] **Conductor** (D-037): placement, affinity routing, single-attempt
      dispatch, credit-based streaming, health states and authoritative
      per-node admission; stale or aggregate reports never admit.
- [ ] **Availability** (D-041, D-046): the per-model `endpoints` shape.
- [ ] **One import per cluster** (D-054): peer replication of verified
      prepared artifacts over the cluster link, archive to and install from
      the long-term store, and the explicit archive-or-delete choice when a
      node lacks space.
- [ ] The package gains its rdma-core dependencies when llmpalooza first links
      them (D-063).
- [ ] Decide whether a worker node serves its own loopback management
      listener.

**Exit criteria:**

- The [required validation](cluster-design.md#required-validation-and-handoff)
  challenge conditions and the conductor's fake-transport and Spark
  scenarios ([architecture](architecture.md#conductor-ownership-and-admission))
  pass.
- An unmodified standard client completes A→B→A through one endpoint, with B
  placed on the other node while A stays resident. Each node enforces its
  full local budget and compatible state reuse, and models run concurrently
  when placement permits.
- Stale capacity reports, node loss and cancellation end in bounded failure
  or unwind, never in unsafe admission or silent replay of a started
  stream.
- Remote management and cluster access require authentication and
  transport protection (D-014, D-065); inference authentication stays
  optional (D-014's owner note). A replicated or archived artifact is published only after
  verification against an identity held outside its source.

## M7 — Demand-paged MoE and the first daily drivers  `pending`

Goal: exact demand-paged routed-expert execution on the named Gemma 4
26B-A4B and Ornith 1.5 35B-A3B pair, and those two models usable day to day
through the named clients, reasoning and constrained output included (owner,
2026-09-23). Resident MoE execution arrives earlier, with M3's full swaps.

**Entry:** M6 exit; M6a is independent. The pair's checkpoints and
representations are pinned, and their GGML source closures, tokenizers and
chat templates are selected and audited (D-013, D-057, D-067). Approved
before measurement: the performance protocols; a resident-performance bound
against the pinned llama.cpp reference (prefill, time to first token and
decode inter-token p50/p95/p99, matched and normal views); the named
budgets, each with its loading policy, including at least one per model at
which its routed experts do not all fit beside its other extents, state and
headroom, so selected experts miss during both prefill and decode; and the
sustained-use schedule and duration.

**Scope:**

- [ ] **Routing boundary** (D-008, [architecture](architecture.md#routing-boundary-for-moe-7)):
      selected-expert leases, asynchronous misses and resumable tasks,
      brought up on a synthetic or tiny MoE before the named pair
      (features.md). Envelopes cover the worst-case union of every allowed
      closure and assume no fixed number of positions per phase (D-068).
- [ ] **Loading policies:** eager active-model loading that keeps inactive
      extents (the feasibility study's recommended first policy) and routed
      demand paging, both selectable, compared at the named budgets.
      With M6's swap pipelining: after a swap, load the routed experts
      each layer selects ahead of a background load of the rest, demand
      reads taking priority over background reads on the shared SSD
      bandwidth, and measure time to first token for a continuation
      against loading everything first (owner, 2026-10-02).
- [ ] **Expert layout:** expert compaction and demand-paged dispatch from
      the M7 GGML proof, building on M3's initial pointer-table or
      uniform-stride choice per format, and the MoE mapping in v0 artifacts.
- [ ] **Shapes that come with these models:** hybrid sliding-window and
      global attention (Gemma 4) and recurrent or linear-attention layers
      (Ornith), with their state adapters and explicit restore coverage
      (RE-004, RE-007), building on M3's Gated DeltaNet layers and state
      adapters. M6's retention matrix extends to them.
- [ ] **Traces:** native routing-trace capture and policy replay, checked
      against the M0 reference experiment; captured traces stay outside Git
      and replay by verified hash in the gate.
- [ ] **Reasoning** (D-043, D-046, D-047): protocol-specific reasoning,
      final and tool fields and streaming, model-supported thinking
      controls, signed blocks, `reasoning` and `reasoning_details`, and
      cached-token usage from real prefix reuse.
- [ ] **Constrained output** (D-043): `response_format` JSON object and
      schema, strict tool arguments and vLLM's `structured_outputs.json`,
      over a documented schema subset with explicit rejection of the rest.
      Design from [DeepInfer](deepinfer-assessment.md#8-tool-calls-and-structured-output),
      model-independent: xgrammar's C++ core as the first candidate
      (Apache-2.0; checked against D-066's no-exceptions build);
      asynchronous compilation that fails only its request; a compile cache
      bounded by bytes; constraint starting at the chat format's trigger
      sequence so reasoning stays free; the mask filled while the step
      runs; and one mask per draft step with rollback, in the shared
      sampler and verify path.
      From [Locals](locals-assessment.md) (owner, 2026-10-22), generic
      over the grammar engine: budget-aware masking, a per-grammar-state
      minimum-completion table built when the grammar compiles (forced
      spans and the pushdown stack included) that masks any token after
      which no valid completion fits in the remaining `max_tokens`, so a
      length finish always leaves valid JSON or a closed call (check
      whether xgrammar exposes the table, else build it above it); and
      canonical forced spans (jump-forward) as certain-acceptance
      proposals in the speculation scheduler's verify rows, tokenized
      canonically with the boundary rule and healing, generalizing M3.5's
      forced begin sequence. Leads, measured after the above exist and
      only if they pay: tree decoding of a small candidate set with its
      forced spans in one pass (needs tree-masked attention rows; Locals
      has no measurement), computing the output head only for the allowed
      rows when few are allowed, and prompt-lookup drafts walked through
      the automaton and cut at the first disallowed token (beside the
      history-copy drafts above).
- [ ] **Sustained use:** a bounded multi-hour agent/subagent session on the
      pair with repeated switches, branches, cancellations (during I/O and
      while paused included), expiry by capacity and by test-shortened idle
      caps, and spill and restore.
- [ ] Demand-paged EXL3 MoE only if claimed, with its own routed-expert
      closure, kernel, quality and performance baselines (D-052); resident
      EXL3 MoE arrives with M4's models.
- [ ] Before the first shipment, define release artifact compatibility guarantees
      using the dense and MoE evidence, in a separate decision. Development
      artifacts require no migration or backward compatibility (D-018, D-121).

**Exit criteria:**

- Acceptance exercises real demand misses. At each miss-forcing budget,
  catalog state and events record nonzero selected-expert misses in prefill
  and in decode, with their count, bytes and wait time and the phase and
  layer at which each suspended. There, a measured window without misses is
  inconclusive for the paging gates, and the next two criteria are judged
  on runs with misses.
- No unselected expert loads beyond declared metadata and read-ahead, and
  no expert is substituted or its contribution dropped, verified from
  catalog state and events.
- Numerics stay correct against the pinned references after eviction and
  restoration and across within-step misses, and the M7 rows of D-050's
  matrix, with the runtime closure-excess check moved to M7, pass with
  real routes.
- D-036's generation limits hold on the pair: at most 10% added generation
  time, continuation time to first token included, and at most 20 ms p95 /
  100 ms p99 added token gaps against a resident control with matched state
  provenance, at every named budget, the miss-forcing ones included. Pause
  gaps are reported separately (D-069). M6's switching floor still holds.
- Resident prefill, time to first token and decode inter-token p50/p95/p99
  on both models meet the approved bound against the pinned llama.cpp
  reference, or the owner approves an explicit tradeoff, as D-052 requires
  for EXL3. Paging limits compare llmpalooza with itself, so they cannot stand
  in for this.
- The sustained-use run ends with every completed, cancelled and paused
  request retired and no lease, grant, task or I/O outstanding. Memory, the
  catalog and retained-entry metadata, task and job records and queue
  depths stay within their bounds and level off instead of growing with
  elapsed time; task and job records and queue depths return to their idle
  levels after each switch cycle. Spill stays within `S_spill` and the write
  budget, and the memory breakdown reconciles with OS counters at the end.
- Measured and reported: the routing boundary's resident-hit overhead, each
  routed phase kind's bound against its observed peak, and whether idle
  retained state starves weight residency.
- Named clients complete reasoning and tool round trips on both models,
  including acceptance case 7's reasoning checks, and constrained-output
  requests pass their pinned compatibility fixtures. Any OpenRouter-mode
  claim rests on D-046's pinned-client run. The support matrix records the
  client versions.

## M8 — Sharded execution under pressure and failure (two Sparks)  `pending`

Goal: a flagship model sharded across both Sparks, correct under asymmetric
pressure, cancellation and controlled failure. M4 already runs TP2 sharded
full swaps (its communication-buffer pool, collective ordering and
collective latency moved there, with a minimal ready-before-collective rule
and bounded abort), and placement-only use works at M6a.

**Entry:** M6a and M7 exits. By entry: the flagship checkpoint and its
validated parallelism recipe are named (M4's models, and the
MiMo-V2.6-Flash-RL TP=2/EP=2 reference, are the candidates); and the
two-node admission design chooses between prepare/commit and the deferred
mirrored-ledger shortcut (features.md). Model-parallel artifact
partitioning is decided at M4's entry.

**Scope:**

- [ ] Parallelism beyond M4's TP2, porting the recipe's first (TP, PP and
      EP are different plans), with conductor-issued distributed phase IDs.
- [ ] Coordinated admission, generalizing M4's minimal readiness rule:
      node-issued reservations and every rank ready before commit, with
      prepare failures unwound; an unknown completion never frees another
      rank's buffers.
- [ ] Sharded models under partial eviction and paging, not only full
      swaps.

**Exit criteria:**

- Both ranks stay correct against the pinned reference under asymmetric
  pressure, cancellation and controlled failure, including partial
  preparation, a lost commit, mismatched rank generations and node loss
  during a collective. No timeout is treated as proof of reclaimed memory.
- Sharded performance is reported against the recipe's reference deployment
  in both views.

## M9 — Performance and new decoding modes  `pending`

Goal: meet D-036's benefit target on a library larger than memory, and
execute the speculative and block-diffusion shapes designed since M0
(D-068) that M3 and M4 did not.

**Entry:** M7 exit. M9 may start before M8 exits, but MiMo's stored MTP
layers run only on M8's sharded execution, so that work waits for M8 and M9
exits after it. Before M9 planning, the bounded DiffusionGemma reference
study measures the per-step expert closures of wide phases, and untriggered
deferrals are reviewed ([features.md](features.md) and the table below).
Before the new modes execute, numerical and statistical bounds and trace
protocols for speculative sampling and diffusion decoding are declared;
manifest references to another artifact by ID are settled in M3. The named
configurations, including the over-memory library, and the
partial-retention benefit workload are pinned before acceptance runs
(D-036).

**Scope:**

- [ ] **Full-engine optimization pass** (owner, 2026-10-04): resume the
      remaining Qwen/DeepSeek prefill, decode, concurrent and long-context
      speed/scaling work accepted as M3 exceptions. Use the retained
      [comparisons and leads](m3-optimization-status.md), including whole
      producer/consumer chains, draft acceptance and state/head delivery.
      Custom kernels and new operations may replace or fuse part or all of
      library kernels, including producer/consumer work around GEMM, when
      the measured whole-engine benefit warrants them (owner, 2026-10-04).
      Re-pin profiles and comparators at entry; qualify correctness and
      paid end-to-end benefit before selecting a change. Frozen unmeasured
      prototypes carry no adoption or speed claim. Optimizations land once
      in the shared engine and reach these models as they are found, not
      held for this pass (D-107, [workflow.md](workflow.md)'s "optimize
      once"); this pass takes what remains of the accepted M3 gaps.
      M3.6 hands on these leads, all still open ([record](m36-record.md#deferred-items-from-the-reviews)
      holds their history): Qwen3.8 MTP activation sizing (8.05 GiB at four
      slots at `53c2610b`, cut by 4.53 / 4.37 GiB for native / GGUF by R7,
      `d3374c67`, against 1.42 GB in M3; the remainder), planning cost (R10
      collapsed the duplicate per-owner step builds of joined prompt batches;
      what remains is the per-new-shape plan build, about 12 against 9 ms,
      and decode-first lookahead for a new shape), the roughly 0.25 s
      workspace mapping in a Qwen3.8 swap, the decode-template leads (vector
      template register blocking at 16 rows, a grouped warp-per-row decode
      variant, multi-stage tiles), one-shape-per-band probes for the graph
      reader index in place of the 25% margin, moving turn-checkpoint capture
      off the first-token path, DeepSeek's budget cliff (about 1.1 GiB less
      `MemAvailable` took qwen3.8 to deepseek 16.1 s against 8.15 s; it did
      not recur in the final table), Qwen's long-context acceptance cost, the
      per-technique structural adoption audit against
      [optimization-inventory.md](optimization-inventory.md), kept-record
      per-drafter adaptive depths, adaptive depth and a device verdict for the
      Gemma 4 assistant, a convert-then-tensor-core prefill path for quantized
      KV caches, a full-context Q8_0 KV scoring path for Qwen3.8 and
      DeepSeek, the per-extent workspace mapping hypothesis (range-wide access
      over runs of extents), the image to deepseek first-use swap at 27.354 s,
      RE-057 (drop the artifact stores' page cache before timing) and the
      heat-soak note on timing gates. (The dedicated physical handle pool is
      M3.5's first item, not M9's.) DeepInfer's leads, each as a shared pattern selected by
      capability for every attention kind whose chain matches, not one
      model's ([assessment](deepinfer-assessment.md#10-fusion-patterns-m9-leads)):
      inverse RoPE and quantization in the decode attention epilogue (T91
      covers DeepSeek prefill only); one fused norm/RoPE/rotation/cast/store
      transform for KV and compressor writers; and merged same-input
      projections as an import relayout chosen by tensor role. Measure
      first, adopt only on a material share: two decode steps in flight,
      decode graphs keyed without context length, and wave costs
      calibrated by context depth
      ([measure first](deepinfer-assessment.md#11-measure-first)).
      Flare's lead ([assessment](flare-assessment.md#8-overlapping-one-requests-independent-branches)):
      a planner pass that runs one request's independent subgraphs on
      lanes with join edges (the shared expert beside the routed experts,
      an indexer beside QKV and the KV write, a compressor beside query
      preparation), enabled per row band by measured gain with scratch
      disjoint by lifetime; T79's fused shared expert lost 0.09% in the
      model, so measure. A bounded C1/C4 decode screen is eligible before
      M9.
- [ ] **Evaluate lossless weight compression with fused on-chip decode**
      (owner, 2026-10-17): M9 candidate, with a bounded experiment eligible
      earlier after M3.6 if measured weight-bandwidth pressure justifies it.
      Keep active weights compressed in device memory and reconstruct their
      exact values inside the consuming compute kernel; inactive-RAM
      compression and expansion into a separate bulk weight buffer are out
      of scope. Start with Qwen3.8's BF16 MTP draft head (the
      [prior lossy-head experiment](experiments/qwen38-quantized-draft-head/README.md)
      is context, not evidence for lossless speedups), then consider MXFP8
      linears. Rank further candidates by measured bytes read per generated
      token and their contribution to decode time; do not assume savings
      for Q8, low-bit GGUF, NVFP4 or EXL3 from BF16 results.
      [NWC](https://github.com/parda21/NWC) and
      [ZipServ](https://arxiv.org/abs/2603.17435) are research leads, not
      adopted dependencies or Spark performance evidence. Screen one
      representative operand on a Spark against its current kernel, counting
      metadata/alignment bytes and decoding overhead; stop if savings or
      potential whole-engine benefit do not justify further work. Adoption
      requires exact weight reconstruction, numerical and draft-acceptance
      checks, and measured end-to-end decode benefit. Production work lands
      in the shared format/kernel path (D-107/D-108), with prepared-artifact,
      paging and memory accounting, prefill and multi-token verification
      paths qualified before selection; a negative screen closes the
      evaluation without requiring a new format to ship.
- [ ] **Larger-than-memory library:** DeepSeek V4 Flash with Qwen3.8 Flash
      Next on one node is the canonical pair (D-036). M3 runs both, with
      their compressed attention and indexers, Qwen3.8's sparse n-gram rows
      and its linear-attention layers, as full swaps; M9 adds partial
      retention and paging on them. If the pair cannot be validated, name
      another whose prepared weights exceed physical memory rather than
      pass M9 on the small pair alone.
- [ ] **Speculative decoding** (D-068), beyond M3's and M4's: stored MTP
      layers for Ornith (and MiMo's once it runs sharded under M8) and
      Gemma 4 companion drafters, with speculative sampling at every
      supported setting (M3.5's shared sampled-draft verify,
      [assessment](deepinfer-assessment.md#2-speculative-sampling-with-the-drafters-distribution)),
      and draft-length and acceptance tuning; and confidence-budgeted
      verification for the remaining drafters that report a confidence
      (it lands with M3.5's calibration screen if DSpark's head is
      calibrated;
      [assessment](deepinfer-assessment.md#1-confidence-budgeted-verification)).
- [ ] **Block diffusion** (D-068): DiffusionGemma-26B-A4B.
- [ ] **Execution speed:** CUDA graphs beyond M3's decode graphs, further
      kernels and plans (D-053), and target-assisted import tuning as an
      explicit, separately keyed mode.
- [ ] **TensorFold's format** (owner, 2026-09-28; D-087; *moved to M3.5 by the
      owner, 2026-09-29, as its "MLX affine import"*): import and run
      MLX-style affine 4-bit weights (TensorFold's checkpoints; reconcile
      the group size, 32 or 64) for Qwen3.8 Flash and GLM-5.3 Flash, and
      optimize them. TensorFold is then a same-format oracle and a gated
      comparator for those models (D-085's ~10% and ~1.1× bounds). Until
      then, in M3 and M4, it is a cross-quantization comparator: reported,
      not gated.
- [ ] **Victim policy:** compare global LRU, frequency/recency and the
      cost-aware heuristic on identical recorded traces, with the confirmed
      hysteresis, minimum-residency and reload-cost refinements
      ([baseline](architecture.md#victim-selection-initial-baseline)) and
      per-model statistics so no model monopolizes reclaimable bytes. A
      candidate replaces the baseline, and the cost-aware heuristic becomes
      the default, only if replay shows fewer miss bytes and reloads.
- [ ] Deferred optimizations whose triggers have fired (prefetch, residency
      warm-start and the others in features.md), each against its
      demand-only control.

**Exit criteria:**

- D-036's benefit target: at least 25% lower median return-switch latency
  than llmpalooza's own whole-model control, with identical state handling at
  the same budget, on the agreed partial-retention workload, with at least
  one named library exceeding physical memory, while the switching floor
  and generation limits still hold.
- Measured and reported on the over-memory library: retention under
  physical pressure and whether idle retained state starves weight
  residency (D-055).
- The speculative verify path's teacher-forced logits match plain decoding
  within declared bounds, with top-1 agreement reported and free-running
  divergence reported at its first position (RE-008). These gates cover
  M3's and M4's drafters too. The targeted checks those drafters need
  (forced rejection, rollback across a swap and a coarse sampled
  distribution) moved to M3's exit and apply to M9's drafters as well.
- Speculative sampling preserves the target distribution. On recorded target
  and draft distributions, acceptance, rejection and residual resampling
  match a reference implementation of the rule exactly under fixed seeds;
  sampled outputs match plain sampling's distribution within declared
  statistical bounds, at the supported sampling settings.
- Rollback leaves exactly the accepted prefix. After rejected drafts,
  teacher-forced continuation matches a control that drafted only the
  accepted tokens, bit-identical where the plan is the same at both verify
  widths and otherwise within the declared bounds. No rejected position's
  KV, recurrent update or drafter state survives, none enters a retained
  entry (D-055, D-068), cancelling between draft and verify retires all
  draft work, and a pause there commits none of it before verify.
- Diffusion stays within declared bounds of its reference, compared step by
  step under a fixed seed. A cancelled canvas never commits, and a paused one
  commits nothing until it resumes and finishes.
- Matched and normal reference comparisons are reported, with no numerical
  or lifetime regression in any supported configuration.
- If prefetch is built, the part of D-050's
  [matrix](reservation-policy.md#worked-cases-and-implementation-gates)
  moved to M9 passes: repeated speculation keeps its full peak in `J` or
  the owning phase.

## M10 — Product and first release  `pending`

Goal: the remaining confirmed product scope, and a first tagged 0.x release
that a stranger can install from the project's package repository and run
(owner, 2026-09-23).

**Entry:** M9 exit. The owner may pull an item forward once its dependencies
exist (for example the Ollama profile or tokenization endpoints after M5, or
the dashboard after M6's management controls); the release itself waits for
every earlier exit. Each model added for embeddings or reranking is
named at entry with its pinned reference engine and numerical
bounds, declared before native evaluation.

**Scope:**

- [ ] **Dashboard** (D-064): a separate service that calls the management
      API from its own server side, including setting the Hugging Face
      token.
- [ ] **MCP management adapter** (D-042): a separate process exposing
      discovery, status and explicitly authorized actions.
- [ ] **Application permissions** (D-042): per-application credentials and
      scopes for inference, read-only status and model administration.
- [ ] **Session and hint extensions** (D-022, D-046): the optional session
      ID and release, client warm hints, and D-046's `session_id`, `user`
      and `metadata` hints. M5's front door already accepts them as
      advisory preferences, and M6 records `session_id` for release lookup.
- [ ] **Ollama subset** (D-041, D-045): listing, details, chat and
      generation with the `keep_alive` mapping, tested with a named
      Ollama-native client including its load-time bound.
- [ ] **Tokenization and diagnostics** (D-043, D-044): vLLM-compatible
      tokenize, detokenize, tokenizer information and prompt rendering; raw
      Completions with standard log-probabilities and bounded token
      diagnostics.
- [ ] **Pooled outputs** (D-042, D-044): embeddings and reranking on
      validated models named at entry, in the shapes their clients send
      (D-101): OpenAI's `/v1/embeddings`, Hugging Face TEI's `/embed` and
      `/rerank`, Cohere's `/v2/embed` and `/v2/rerank`, Jina's rerank,
      and Ollama's `/api/embed`.
- [ ] **File inputs** (D-042): text resources. Images, video and audio
      files moved to M3.5 and M4 with their families (D-101); here the
      Ollama profile carries `images` for the models validated there.
- [ ] **Packaging** (D-027): the signed arm64 apt repository and its signing
      keys, optional copyleft modules in the default install with a
      build-time opt-out (D-080), and drain-before-restart upgrades; and,
      secondary (D-098), an OCI image built from the release `.deb` with
      its published io_uring seccomp profile and documented run flags,
      checked by serving a request on a Spark.
- [ ] **Release readiness** (D-061, D-062): the release checklist, the
      published support matrix, notices and source obligations for every
      shipped profile, and user documentation.

**Exit criteria:**

- Each new route passes its pinned compatibility fixtures, including
  negative and interrupted-stream cases, with the unmodified clients its
  decision names; unsupported features fail explicitly.
- Every new model and output path meets numerical acceptance against its
  pinned reference within the declared bounds before any support claim:
  embedding vectors with the model's pooling and normalization; rerank
  scores and orderings (media inputs are judged in M3.5 and M4, D-101).
  Raw Completions log-probabilities agree with the validated
  teacher-forced logits, and tokenize and render output matches inference
  byte for byte and ID for ID.
  Generated text alone is not evidence.
- On a fresh Spark, installing from the apt repository and following only
  the checked-in docs reaches a running supported model (vision.md).
- The default and copyleft-disabled builds meet D-017 with a complete,
  audited closure; builds with optional modules enabled ship matching
  notices and source.
- The release commit passes `check`, `check:full` from a fresh clone and
  `check:spark` (D-061), and the owner tags the first 0.x release.

## Deferred delivery and proposals

Confirmed scope stays confirmed when its implementation is deferred. A
candidate stays unapproved until revisited; reaching a trigger is a reason
to evaluate it. Deferred optimization triggers (predictive prefetch,
dependency-group scoring, optimistic MoE) live in features.md.

| Item | Earliest work / revisit trigger | Scope |
| --- | --- | --- |
| Quality/performance modes: KV-cache and output-head compression (FP8 caches, TurboQuant and other algorithms; a quantized output head) and FP8 linears for Qwen-Image's DiT | After M3's speed work; earliest when long contexts or several resident conversations press on memory, or to shrink swap spill | [Per-model options with long-context quality checks](features.md#compute-backends-and-execution) (owner, 2026-09-28)  Exposed as model-mode aliases and the image endpoint's `quality` (owner, 2026-09-28) |
| Load/temperature-aware GPU operating policy | Earliest M9 evaluation, or earlier diagnosis if reproducible throttling or unexplained shutdowns occur | [Proposed telemetry and optional adaptive clock ceiling](features.md#load--and-temperature-aware-operating-policy-proposal); measure stock/fixed/adaptive policies first, no assumed fault or automatic host changes |
| Ollama registry and other management compatibility | After the basic subset and relevant native management operation, when a named client needs them | Deferred D-041 candidate; lifecycle mapping needs separate proof |
| Regex/grammar constrained output | After validated JSON/schema support, when a concrete client requires it | D-043 deferral; no automatic milestone delivery |
| LoRA adapters | Earliest M9 planning after validated base-model execution, when a concrete adapter workload needs them | D-044 deferral; no automatic delivery |
| Classification/reward/generic pooling APIs | Earliest M9 planning after validated base-model execution, when a concrete model/task workload needs them | D-044 deferral; not implied by embedding/reranking support. Decision models over the Jev API are the exception, in M3.5 (D-101) |
| Live audio/video input | Earliest M9 planning after initial file-input evidence, when a concrete workload establishes streaming/synchronization requirements | Deferred D-042 candidate; not an automatic M9 deliverable |
| Batch/background inference jobs | Earliest M9 planning after validated request scheduling, when a concrete workload justifies scheduling/storage needs | Deferred D-042 candidate; ordinary background request priority is already confirmed |
| Automatic membership changes | After M6a, when configured enrollment and explicit restart cannot reasonably serve membership churn | Candidate mechanism under D-023/D-038; bootstrap discovery and path refresh for enrolled nodes are already M6a scope |
| Conductor election | After M6a, when conductor failover becomes an explicit requirement; first define fencing and in-flight request handling | Candidate mechanism under D-023; one configured conductor initially |
| Automatic replica placement and balancing | After M6a, when measured overlapping demand on a small model causes waiting while another node has sufficient headroom | Confirmed D-023 scope with deferred delivery; preserve affinity and include duplicated weights/state in budgets |
| Apple-silicon port design notes | When the Apple port is scheduled (AGENTS.md: Apple first among later ports) | Compute dtype as a device-capability plan input (M1/M2 emulate bfloat16), row-count classes in rank data, measuring against current MLX on the target chip, per-platform relayouts for formats Metal reads natively ([Locals assessment](locals-assessment.md#apple-silicon-notes-for-the-later-port)) |
| Discrete-GPU fast swap of large models | After the two-Spark swap (D-087); small models only fit a 12 GB card | M6 validates the A→B→A fast swap on the workstation's RTX 3080 Ti (D-082); a later slice or milestone takes the swap path further there |

Review untriggered items during M9 and M10 planning; they do not
automatically enter either milestone's scope or block earlier milestone exits.
