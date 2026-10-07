<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
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

The [bounded dense31 serving bridge](experiments/gemma31-serving-bridge/README.md)
now selects the checked ordinary recipe only at resolved context<=8192/slots<=4,
with current-pin C1/C4 quality, complete 1K corpus parity, HTTP controls and
focused default-selection checks. The [bounded Gemma26 recipe](experiments/gemma26-production/README.md)
reaches stock parity with stock's exact tokens under the same bounds; larger
envelopes retain their prior recipe; sustained performance and broader
qualification remain open.
The historical SOURCE14 serving/default proposal was not adopted. Its original
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
removes both paid differences and matches all160 paid full heads, but retains one
strict frontier difference; conditional loss passes at +0.08119%. This is diagnostic
evidence, without a production layer policy or observed stock992 routing selector.
C3 transfer, other partial model counts and unequal widths outside the
[bounded C2 common-root recipe](experiments/gemma4-bounded-owner-roots/README.md)
remain unqualified. That recipe restores all 68 stock heads on both short and
physically wrapped controls, preserves initialized states and removes temporary
KV copies. Same-binary paid latency falls 6.01% / 5.24% (26B / 31B), within 0.1%
of current stock on short n2 screens. Ordinary context<=4096/exactly-two-slot
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
Each milestone leaves a usable, testable result. None has a promised date.

| Milestone | Result | Needs |
| --- | --- | --- |
| M1 Bootstrap | Pinned SDK, builds, local check gate, package skeleton, confined-job proof | M0 (done) |
| M2 Resource core | Catalog, admission and leases on a fake backend and a Spark; the backend proof settles the operation contract | M1 (done) |
| M3 Single-Spark fast swap | DeepSeek V4 Flash, Qwen3.8 Flash Next and Qwen-Image-2.1 swap A→B→A on one Spark, aiming at ~10 s to first token, as correct, fast and lean as their references | M2 |
| M3.5 Model families | The core engine runs the major open model families, MoE and dense (Gemma, Llama, MiMo and the other top-tier families), each correct, as fast as its reference and flat with context, before the system is built around it; image, video and audio file inputs on the models that take them, decision models over the Jev API, and image, video and speech generation routes (D-101) | M3 |
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
provenance records (D-071), versioning (D-062), `jitllm doctor` (D-072), the
node configuration and storage-role checks (D-073), and the arm64 package
with `jitllm-runtime`, `jitllm.service`, crash handling and confined jobs in
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
GGML's and ExLlamaV3's kernels under jitLLM's dispatch (D-077, D-080),
selected by plan (D-053), with cuBLAS on a jitLLM handle (D-076): native
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

- [x] **Family selection** (approved by the owner 2026-09-29: all 13
      checkpoints of [m35-families.md](m35-families.md), with the answers
      recorded there). The goal is
      capability coverage, not particular models (the owner, 2026-09-29).
      Survey each top-tier open family's current and older widely used
      generations, list every architectural feature they use, and mark
      which jitLLM already supports in a capability matrix. Then choose
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

      This applies to every added model family, including Clef and
      Clef-flash's prefill-only decisions and image generation (owner,
      2026-10-03). Compatible work from independent requests batches in
      the encoders, decision heads, denoising steps and image decoders.
      Each request keeps its own state, seed, guidance, step count,
      size, edit inputs and cancellation. Different input lengths or
      image sizes form compatible groups and interleave; they do not
      disable batching for the model. Admission follows memory, and
      waiting groups age so compatible arrivals cannot starve them.

      The chat route's one-request-at-a-time queue (D-097) becomes a
      batch scheduler. Batching applies to requests for the same model;
      different models still time-slice by swapping (D-019).
      M3's added concurrent-request comparisons now measure this gap;
      advance the necessary implementation into the optimization run
      where those comparisons require it (owner, 2026-10-01).
- [ ] **Skeleton gaps the M3 cleanup's review named**
      ([engine.md](engine.md)):
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
      Ordinary slots remain capped at two; unequal C3/C4 model reads, continuous
      arrival, cap50 transfer, C8/C12 and public adoption follow separately.
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
      [copy-free bounded-read transfer](experiments/gemma3-execution/README.md#copy-free-bounded-owner-reads-2026-10-07)
      preserves exact padded/bounded native heads/state; fresh small/ring stock
      screens have 73/74 and 70/74 exact heads and zero strict differences.
      A same-binary short n=2 screen lowers native paid latency 5.77372%, to
      −0.29466% versus original stock, with identical histories/final heads.
      Four HTTP aggregate cases/two clean epochs pass with actual bounded plans
      and exact matched cached replay. Wider cohorts, broader memory/swap/context
      and sustained quality/performance remain open. The deferred and dropped
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
      - **Audio:** no pinned checkpoint keeps an audio encoder, so Gemma
        4 E4B-it joins as the audio carrier (owner, 2026-10-02): 16 kHz,
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
        schema head of about 125M parameters.
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
        pointed at jitLLM by base URL.
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
      - `/v1/audio/speech` on two text-to-speech testbeds (owner,
        2026-10-02):
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
  same-format reference, at 8K and at depth; peak memory at most about
  1.1× the reference's.
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
- **Decision models:** Clef's and Clef-flash's per-option probabilities
  match their pinned reference within declared bounds, text-only and with
  images. The TypeSafe SDKs complete requests unmodified. Decision
  requests to each model batch with each other, including requests with
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

## M4 — Two-Spark fast full swap  `pending`

Goal: the same cycle for models too big for one Spark, sharded across both.
Each node holds its shard on disk, and the conductor loads both shards at
once.

**Entry:** M3.5 exit (M3 exit before 2026-09-29). By entry, model-parallel artifact partitioning is
decided ([artifact-format.md](artifact-format.md#deliberately-open); moved
from M8's entry), and each model's checkpoint, recipe and baselines are
pinned and audited as in M3.

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

**Scope:**

- [ ] **Provenance and licenses:** re-audit the GLM recipe if its baseline
      moves past the pin (HEAD is 65 commits ahead); record the GLM
      checkpoint mirror's license and the DFlash2 drafter's (CC BY-NC-ND
      4.0) for information (D-087); Mia's E3 and cooperative MoE kernels
      (AGPL or mixed, D-080); and whether Mia's ExLlamaV3 revisions'
      formats match our `6b84a21`.
- [ ] **Baselines:** Mia's two-Spark configurations and TensorFold, run by
      us, measured as in M3. Refresh TensorFold for each applicable task
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
- [ ] **Coordinated readiness, minimal** (the start of M8's coordinated
      admission): no collective or execution of B begins until both ranks
      report ready, their shard loaded and its memory obtained. A
      preparation failure on either rank, or no report within the stated
      timeout (default 30 s, owner-adjustable), aborts the swap: the other
      rank releases or rolls back what it prepared within that timeout, no
      rank enters a collective alone, and the swap reports failure cleanly.
      Memory a rank has not confirmed released stays counted as held; no
      timeout proves reclaim. General recovery, prepare/commit and the
      broader pressure and failure matrix stay in M8.
- [ ] **Kernels:** EXL3 MoE, the codebooks and rates these checkpoints use,
      sparse MLA decode and prefill on `sm_121`, KDA, the DSA indexer, CSA2
      and Engram row gathers, each the fastest correct implementation under
      D-053, chosen by a quick A/B.
- [ ] **Per model,** as in M3: graphs, state adapters, templates, the
      resident expert layout for EXL3 MoE, and speculation with M3's
      forced-rejection checks where the model supports it (for GLM, MTP or the
      DFlash2 drafter that Mia's GLM configuration uses, whichever is faster
      and correct; the
      nextn layers or DSpark head for v4.1). ReconGemm prepares its
      cuBLASLt descriptors once per bound GEMM (M2's hand-off).
- [ ] **Media inputs** (D-101), on M3.5's intake:
      - GLM-5.3 Flash's images and video: the GLM ViT (24 layers, patch
        14, 2D RoPE inside), video at 2 fps with per-frame timestamps;
      - DeepSeek V4.1 Flash's images: the DeepSeek-ViT with its 3×3
        aligner, at most 1,024 tokens an image in reading order.

      Both checkpoints keep their towers. Mia's V4.1 build says vision was
      not its validation target, and llama.cpp has no V4.1 architecture,
      so V4.1's oracle is its repository's own inference code.
- [ ] The minimal `/v1/chat/completions` serves the sharded models.

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

- [ ] **Importer and verifier** (D-009, D-056): the C++ importer and the
      standalone verifier, with M0's Python prototype as the exact
      accept/reject oracle, running on the workstation and on a Spark.
      Confined import jobs take sources only from the configured stores
      (the `checkpoints` role or the long-term store; D-054, D-063) or the
      Hugging Face Hub (resumable and verified; the token comes from a
      `.env` or config file and is never logged). Hard-linked and symlinked
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
- [ ] **Resident execution:** the FP16 and both EXL3 fixtures under jitLLM
      dispatch, finite context and chunk profiles within the 8K context,
      and state block sizes and KV layouts for each adapter. Decode graphs
      and their relocation proof come from M3.
- [ ] **Tokenization and output:** D-067 renderers for both fixture
      template hashes on M3's native tokenizer, stop rules and sampling
      (moved to M3), and the tool-call parser; host versus device sampling
      is settled against D-052's decode gates.
- [ ] **Front door** ([M5 surface](client-api-baseline.md#m5-surface)),
      growing from M3's minimal Chat Completions:
      Chat Completions, stateless Responses, Messages with token counting,
      `/v1/models` in both shapes with D-046's metadata, and the discovery
      document (D-041); D-045's listener, authentication, CORS, `Host`,
      status and keepalive rules; D-047's non-streaming and storage rules;
      the Claude Code profile; and explicit rejection of `transforms` and
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
      owner on 2026-09-27): a system-call filter for `jitllm.service`
      that admits io_uring, and a single reaper for jobs started from
      other threads.
- [ ] **TLS** (D-065): per-name certificate files selected by SNI, reload on
      change, key-match and expiry checks, the name-constrained local CA,
      the certbot deploy hook, and the Tailscale certificate timer. The
      root-run hook and timer touch only `/etc/jitllm/tls/`, their units are
      sandboxed to it, and they take the heavy path.
- [ ] **Surface definitions:** front-door, alias and TLS configuration keys;
      the individual `jitllm-` header and body-field names (D-062); the
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
      24-hour idle caps; the victim-order baseline; spill with its protected
      directory, preallocation, direct I/O, digest check on restore and
      deletion at startup; miss reasons and fallback reporting.
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
  orientations, six jitLLM arms against fresh interleaved references, 72
  accepted repetitions per arm, orientation and cache condition. For each
  floor arm (J-partial, J-spill), orientation, direction and cache
  condition, at the median and at p95, jitLLM's one-sided 97.5% upper bound
  is at most the smallest one-sided 97.5% lower bound among the valid
  reference arms; a comparison with no valid reference arm does not pass
  (D-036). The report covers latency distributions, bytes read and written,
  peak memory and spill, prompt tokens reused versus recomputed, and deltas
  against jitLLM's whole-model control (M9's comparator).
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
- [ ] The package gains its rdma-core dependencies when jitLLM first links
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
- [ ] **Sustained use:** a bounded multi-hour agent/subagent session on the
      pair with repeated switches, branches, cancellations (during I/O and
      while paused included), expiry by capacity and by test-shortened idle
      caps, and spill and restore.
- [ ] Demand-paged EXL3 MoE only if claimed, with its own routed-expert
      closure, kernel, quality and performance baselines (D-052); resident
      EXL3 MoE arrives with M4's models.
- [ ] Assess artifact compatibility guarantees with M6's dense and M7's
      MoE evidence, in a separate decision (D-018).

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
  for EXL3. Paging limits compare jitLLM with itself, so they cannot stand
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
      prototypes carry no adoption or speed claim.
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
      supported setting, and draft-length and acceptance tuning.
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
  than jitLLM's own whole-model control, with identical state handling at
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
| Discrete-GPU fast swap of large models | After the two-Spark swap (D-087); small models only fit a 12 GB card | M6 validates the A→B→A fast swap on the workstation's RTX 3080 Ti (D-082); a later slice or milestone takes the swap path further there |

Review untriggered items during M9 and M10 planning; they do not
automatically enter either milestone's scope or block earlier milestone exits.
