<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3.6 handoff

Working state of the M3.6 refactor ([plan](plan.md), "M3.6 — Engine components",
[design](engine-components.md)) for whichever agent continues it. The original
orchestrator paused on 2026-10-10 for usage quota; G3b has since been
resumed, reviewed and checked on both Sparks. **This file is
temporary**: update it when you land or start a slice, and delete it when
M3.6 closes. AGENTS.md's rules govern all of it (rules 5, 9–13 especially:
one reviewed squashed commit per sub-task, heavy-path review rounds, Spark
jobs through `spark-job`, the full Spark set once per slice).

## How a slice runs

1. Builder in its own git worktree (`.claude/worktrees/<name>` or any other),
   with its own source copy on a Spark (`~/src/m36-<slice>`; never copy a
   build directory between paths).
2. Heavy-path review: a separate reviewer challenges it, rounds repeat until a
   round is clean or finds only low-severity issues (fixed without another
   round). Whoever changes code last runs the full Spark set on the final
   state (AGENTS.md rule 10).
3. Merge: root carries the reviewed, checked slice onto `main`, preserving
   its exact source identity and both sides of any integration. Semantic
   conflicts need review. Main builds and passes affected integration checks
   on a dedicated Spark copy before root commits.
4. Commit on `main`: `[M3.6] <what landed>`, recording the review and checks.
   Tick the plan sub-item. Keep unmerged worktrees and milestone evidence
   until their work lands or the owner directs otherwise; final cleanup
   follows `workflow.md`.

Family migrations add the 1% speed gate (docs/experiments/m36-baselines/README.md:
protocol, frozen baselines `~/.local/share/llmp/m36-baseline/1facc957-fixed`
on both Sparks, candidate installs under `~/.local/share/llmp/m36-candidates/`,
`tools/speed_gate.py`; the new arm runs with
`LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS=1` where the baseline's turn
checkpoints are broken, defect 3). Gate runs are timing work: exclusive
`spark-job --gpu` jobs, never beside a build on that host.

## Current direction, 2026-10-12

The pinned Mia reference and current-main manual Qwen A/C diagnostic score
the same quantized checkpoint and fixed 1,016 targets. Mia PPL is
35.144838403; native A/C differ by +0.389325%/−0.248639%. Full native build,
32 host controls, both actual cases and quality checks pass. Every native
loss also matches the historical observation; exact matching was not a gate.
Both experimental arms disable `owner_row_invariant`, so ordinary-default
policy remains unqualified. Next is a matched one-row-per-owner policy pair
on the same histories, after measured-class verification. Cache precision
differences are disclosed; no family adoption, serving-speed or
closer-to-original claim follows.
[Reference and manual replay](experiments/m36-numerical-audit/qwen-quantized-reference.md).

The bounded dense Q8_0 cache opt-in (D-117) keeps F16 default and uses actual
source/state/kept identities. Main native build and 519 host controls plus four
actual D64/D256 cases pass. Same-weights paired short PPL rises 4.598% and
1.452%; Gemma logical bytes shrink but its padded reservation does not. This
is explicit functionality with a measured quality cost, not an automatic
selection or whole-K1 closure. Long context, other families and serving speed
remain open. [Evidence](experiments/m36-numerical-audit/q8-state.md).

Spec-bound calibration now uses execution's stored/synthesized spec authority,
keys measurements before resolving calibrated values, and rechecks identity at
factory construction. Legacy keys, artifact-named paths and settings defaults
are preserved. Spark A main native build and 321 host controls across 15 affected
targets pass, followed by six existing Gemma target/assistant model cases with
no failures or skips. Source/binary guards and complete child retirement pass;
no compute processes or memory events remain. The observed 793,939,968-byte
peak is process/cache accounting, not device occupancy. Evidence is retained in
`spec-calibration-main02` and `spec-calibration-gpu02` under the coordination
scratch directory. The earlier Spark B native result also passes; its model
preflight stopped before execution because the publications were absent there.
Source checks pass: formatting, boundaries, headers and REUSE 6.2. No new quality
or speed claim is made.

Companion metadata parsing has a separate source-derived 384 MiB startup
envelope, independent of head-observation storage. Shared kept-state identity
starts with the bound spec and stage layouts. A present prepared execution
identity is now folded into the same opaque layout before registration;
durable records and slot restore/adoption reject another declared configuration.
Legacy/null sources retain their original layouts, including ordinary serving
without a required domain. No on-disk fields/version change or new OS-process/
GPU restart qualification is claimed. Main Spark A native build and all 229
host controls across nine targets pass with no failures/skips and source guards
and child retirement intact (`kept-execution-main01`). Whole-operation admission
remains open.
Architecture-based settings applicability/defaults and legacy retirement remain
separate migration work.

The reviewed owned Qwen target/MTP joined execution foundation passes main native
build and 305 affected host controls, then the actual two-owner screen on Spark B:
13 units, 132 finite heads, checkpoint/spill restoration and changed-input capture
replay. It includes complete v0 row-table reconstruction through the existing
artifact assembler and actual-width/selected-ID draft sampling. See the
[joined MTP report](experiments/m36-numerical-audit/qwen-joined-mtp.md).
Original-reference quality, durable old/new persistence, production serving and
the family speed/swap gate remain open. The full DeepSeek target screen now
passes main native build, 307 affected host controls and actual finite-head /
snapshot continuation on Spark B. The actual spatial-image screen passes
on main after full native build, 129 affected host controls and source checks.
Neither result closes its whole-family migration gate.

G4c1 qualifies the closed DeepSeek target factory at context 3072/chunk 512:
18 finite complete head rows, nine restored rows and completed frontier 2568.
Actual setup pays 106,614,572,256 bytes under the unchanged 110 GiB cap and
6 GiB physical margin. The reader reuses already-funded compiler space, and
no-head prefill skips dead final expert products while keeping all persistent
state writes. The actual case passes in 28.516 seconds including setup/storage;
this is not a throughput result. DSpark's composed execution now passes the bounded screen below;
original-reference quality, persistence, production serving and matched family
speed/swap gates remain open. See [the target report](experiments/m36-dsv4-target/README.md).

G4c2 now passes complete composed target/DSpark execution on main: full native
build, 308 affected host controls and the actual block-drafter/snapshot screen
on Spark B. Each of two continuations completes seven steps, all 17 requested
proposals and one accepted draft; 18 finite complete target heads and nine
restored rows reach frontier 2568. Actual setup pays 117,595,233,088 bytes under
the unchanged 110 GiB cap and preserves the 6 GiB physical margin. The case
takes 35.309 seconds including setup/storage, not a speed qualification.
Draft-head finiteness, original-model quality, durable persistence, selected-only
expert paging, production serving and matched family speed/swap gates remain
open ([report](experiments/m36-dsv4-target/dspark.md)).

Main through `7e0bf848` includes split NVFP4 paged rows, checked route/part
helpers and bounded selected-expert grouped execution. Full native main build,
eight helper controls and both actual grouped GPU cases pass, including repeated
contributions and changed-routing replay. Native ModelOpt admission remains open. The supplemental base-lease prerequisite landed in `ccd5a556` and passes full native main build and
145 focused host controls, including cancellation and unknown-retirement cases.
N4b6d now passes its bounded real shared routed-step foundation: full native
main build, 101 routed host controls, 34 fusion/validation controls and three GPU
cases. The fixture executes 288 complete outputs across eager/captured/replayed
steps, selected-only residency and state frontier 9. This does not qualify
production native expert/model admission or a full family. N4b6e ordered groups
now pass full native main build, 104 routed plus 34 fusion/validation host
controls and four GPU cases. The two-group step checks 288 full outputs and
36 actual routed IDs across nine completed segments, with final-only output
publication and fresh selected admission per group. Native model admission,
full-family quality/performance and inventory scaling remain open
([evidence](experiments/m36-native-grouped-foundation/README.md#ordered-routed-groups--2026-10-12)).

N4b6f's allocation-free page ordinal lookup passes main native build, 157 host
controls and four routed GPU cases. The same-host CPU screen supports the large
selected-device inventory case without persistent memory; small dense and host
fallback regressions are explicitly accepted and reported. Only routed selected
closure discovery calls this helper. It does not qualify model latency or close
native admission ([results and tradeoffs](experiments/m36-native-grouped-foundation/README.md#ordinal-page-lookup-cpu-screen--2026-10-12)).

N4b7 now consumes one prepaid stored spec/source and the shared component
role/owner visitor in admitted native Gemma builds. Main full native build and
191 affected host controls pass; the four existing published Gemma 2/3/26/31
execution cases and mapped DeepSeek source case pass without new artifact
publication or payload hashing. Real setup measurement and ordinary admission
are unchanged. Native Qwen/ModelOpt/MLX serving, multipart expert placement,
shared row-group ownership and full-family gates remain open. Source and map
copies/discovery graphs removed here are a structural simplification, not a
measured latency improvement.

The generic reserved-weight alignment landed in `b9833111` and passes main native build and
108 focused host controls. It provides actual checked physical facts without
installing tuning. The [Qwen secondary screen](experiments/m36-numerical-audit/qwen-joined-score.md)
records 4.56% lower perplexity on 128 targets and no clear short-screen timing
separation. Broader decode continuation, quality and serving gates
remain open; the original unquantized reference is still missing.

The shared real-program validator, full-head row witness and closed target-only
scoring factory landed in `5167cc36` and pass main native build and 127 focused host controls. They
activate no numerical policy or ordinary serving domain. The narrow private
Qwen51 actual Plain decode continuation passes eight generated steps and both
checkpoint/spill replays: 32 reference and 32 exact replay heads/tokens, with
eight checkpoint and four spill restores. Broader quality and family gates
remain open; no experimental policy is adopted.

Qwen52 extends the same pre-fixed four passages to 1,016 targets. One fresh
A/C pair passes all input/head/program controls; aggregate PPL improves 0.64%,
but the new 888 targets improve only 0.057%, with two documents regressing.
Most gain comes from the old prefix. This mixed secondary observation adopts
no policy and warrants no further context/timing run by itself. The bounded one-group routed
continuation passes; ordered-group execution now passes its bounded two-group controls. Native GGUF target normalization passes
a full native build, 103 affected host controls and one real-header-only case:
all 1,224 encoded routes are preserved, without payload traversal, hashing or
publication. The bounded reader prefixes can include bytes beyond the headers.

The shared phase-admission correction landed in `d8aa6388`, with a full native
build and 60 affected host controls: scoring and decode use their actual declared
modes, while compound prompt permission remains unchanged. Qwen50 arithmetic
qualification proceeds separately; no serving claim follows from this bug fix.

The bounded DeepSeek slice landed in `7e0bf848` and passes full native main build, 59 affected host controls
and one actual bounded mapped attention-source GPU case. It executes the literal
owned source/lower callback, verifies 320 finite head values and 376832 state bytes
through changed-token eager/capture replay. The default-off policy is unchanged;
full model admission, fusion and family quality/speed gates remain open. See the
[consumer report](experiments/m36-dsv4-rounded-consumer/README.md).
The shared owned-backend declaration also closes G4b2b2b3b2; its existing DS
consumer uses the same prepared owner and source/plan/snapshot/kept identity
machinery. Whole-operation admission, DSpark and complete-model kept/state,
quality and speed gates remain open.

Main through `0d127ac0` includes the explicitly tuned BF16/F32 vector-product
capability, all four EXL3 profiles' original-model scoring and shared lifecycle
controls, and the active FP16 benchmark adapter over shared components. The
legacy fixture executors have been retired and bounded G5 migration is
complete. Original-reference quality and same-implementation
state checks replace cross-library bit equality as its prospective gate; the
fixtures remain unserved with no speed baseline. Details and final integration
evidence live in [the G5 report](experiments/m36-exl3-selected/README.md).

Qwen49 owns preparation and source/launch compatibility in
`.claude/worktrees/m36-owned-backend`. Main passes a full native build and
148 focused host/refusal controls, one actual synthetic backend control and
three existing FP16 adapter controls. Metadata is funded before node start;
programs/recordings bind stable launch generations. The design avoids recursive
SDK/library byte attestation and retains qualified provenance, actual readiness
and execution identities. No numerical selection or joined-serving quality/speed
qualification follows. See [the source integration proof](experiments/m36-numerical-audit/qwen-products.md#owned-backend-configuration).

The owner requested a [retrospective numerical audit](experiments/m36-numerical-audit/README.md)
and quality/performance evaluation rather than mandatory bit parity.
The subsequent clarification uses the same quantized checkpoint's reference
implementation as the baseline; original weights are needed only to claim
closer agreement with the original. Missing original weights do not block
same-quantized-model qualification. Stop adding
legacy-rounding corrections solely to satisfy strict diagnostic screens.
Qwen45 and DS52 completed strict diagnostics with numerical differences;
DS53/53b remains unapplied/unbuilt. Preserve their raw evidence and unmerged
worktrees. Ordinary joined head/product kernels are recovery candidates,
not yet quality/performance-qualified defaults. Exact source/selection safety
and same-implementation restore controls remain. The matched 1% gate stands.

The bounded shared teacher-forced scorer landed in `a38b5ec0`; its real
Qwen2 FP16 adapter passes main native integration and both natural-ID policies.
The [original Qwen2 BF16 smoke](experiments/m36-original-qwen2/README.md)
authenticates and executes revision `7ae557604adf67be50417f59c2c2f167def9a775`
and aligns all 42 target losses on a 43-token natural passage. The EXL3 G adapter
also uses the common scorer and recorder, complete original-tokenizer identity
and supported seven/one-row tuning; absent three-row tuning explicitly refuses.
Broader quality
and performance remain open. Its existing official F16 artifact has no proved original
conversion lineage. Compare small reversible candidates by quality and matched
speed against a reference implementation of the same quantized checkpoint.
Qwen3.8/DeepSeek original BF16 execution remains missing from inspected
evidence, limiting original-agreement claims rather than this baseline.
The selected FP16 fixture now uses fused vector attention (`9b45ecc7`). Its
1,020-target comparison to the original BF16 model records a +0.003826683%
perplexity change versus the native baseline, accepted as a fixture-only
tradeoff; same-implementation 76/577-row state controls and all ten historical
controls pass. This is neither a general quality threshold nor a serving gate.

Qwen46/47 bounded product screens have completed on Spark B. Ordinary joined
BF16 tensor-core products contract F32 activations to BF16 internally; the
existing joined vector launcher retains F32 activations and much smaller
error against the stored-weight F64 product reference. The paired vector
candidate is slower for router/value and slightly faster for PLE key than
tensor cores in this small replay. Seven peer columns repeat one captured
owner input synthetically. Neither screen establishes original-model quality
or serving latency, and no blanket policy replacement follows. The minimal
shared selection capability landed in `79dc643b`; keep the frozen 45 diagnostic arm.

Current parallel slices: Qwen50 screens ordinary joined execution against
three narrowly tuned BF16/F32 products using secondary target-only likelihood
and matched timing. Native split-row paging now passes its bounded shared-runner
foundation: main native build, 117 focused host controls and two tiny GPU cases,
including changed-ID capture replay and multipart refusals. N4b6d also passes
the bounded one-group native expert continuation on a real ComponentStepSource.
N4b6e's ordered-group extension also passes main native and bounded two-group
execution; production native expert/model admission and full-family
quality/performance remain open. DeepSeek's default-off
rounded consumer has separate real component/source host controls and a passing
declared-backend synthetic GPU corridor. Main integration passes a full native
build, 193 focused host controls and the same one-case GPU corridor on Spark B.
The bounded mapped attention source now passes actual execution/replay; full
model admission and fusion remain open. Root corrected lazy framework inputs that accidentally
reset an owner's graph-builder context; main native build and 162 focused
controls pass on Spark. The fix carries no parity policy.
Each slice receives its own scoped review and checks before landing.

Both Sparks are available for parallel development/checks; actual GPU jobs
remain exclusive on their host. Do not continue the old parity-only loop.

## State on 2026-10-10 (historical)

On `main`: format layer, engine core, IR and planner, partitioning, spec and
mappings; Gemma 2 and 3 served on the shared engine and their M3 code deleted
(`4a6710ce`); Gemma 4 served with its assistants on the shared engine and
its M3 code deleted (`04339dec`); Qwen3.8 components, mappings, patterns and GGML
lowering (G3a and G3b; see `git log`); artifact format v1 and its reader
(`5bce159d`, D-113); kept files and weight shards relaxed and opened only
while active (`4b8b35a9`, `961fdefe`); license tags for ported code
(`019dd77b`); AGENTS.md working rules (`db6cdff0`); `.env` through mise
(`171717ef`).

Latest completed work (2026-10-11): sole-owner Qwen3.8 serving (`e2fd5bab`),
archive writer ownership/progress (`21ebd562`), and owned GGUF Gemma target
normalization N3b1 (`f7401054`) are on main. N4a's copy-only v1 writer and CLI
have clean independent challenge, exact real Gemma 2 publication/reuse, and
2,530 native / 1,985 CPU / all 114 model checks with no skips; main build, 43 integration checks and quality gates passed. Remaining N3b domains, N4 admission and N5 execution
qualification remain open.

Recent slices and handoffs (`git worktree list` shows current worktrees;
update this section as each slice lands):

- **G2c and G2d: Gemma 4 26B-A4B and 31B with their assistants on the shared
  engine, and the deletion of the M3 Gemma 4 code** landed on `main` (see
  `git log`). The 1% gate passed: 26B at 9 blocks, 31B at 7, and the 31B
  assistant screen 4.4% faster. Records are in
  `spark:~/scratch/m36-gate/g2c-c9f7f2db*` and in the baselines README's
  G2c section, which includes the 4-row verify's bit identity with M3 for
  both profiles. Open from G2d: the assistant's draft-pass logits are no
  longer compared with the reference executor, only run to run
  (`component_runner_gemma4_gpu_test`) and by the served acceptance test
  (`Gemma4AssistantServingGpu`).
- **G3b: Qwen3.8 patterns, policy and GGML lowering** landed on `main`
  (see `git log`). Current main was integrated before
  review. Plain mode is bit-identical to today's `Qwen38Runner` fast form on
  all three forms (ModelOpt CUTLASS, ModelOpt GGML experts, GGUF): each
  compares 31 outputs and 242 state tensors across two owners and 24 waves,
  including two captured replays and a step past the QSA budget
  (`component_runner_qwen38_gpu_test`, final run on `spark-b`, 2026-10-10).
  All 96 setup/MTP shapes lower launch for launch on the CPU, with separate
  checks for multiple owners and commits of one to three kept rows.
  Review fixed save-index lifetime, reads across in-place writes, dropped
  fusion parameters, incomplete side-row staging and scalar head NaN policy.
  Specialized QSA lowering proves its typed staging and state/mask structure;
  unsupported external staging semantics are refused before execution
  (as-built limits in `engine-components.md`). Primitive exact masks prove
  their own position source, held-cell metadata and integer bounds, and
  refuse overwritten roots. The final independent challenge is clean.

- **G3c1: Qwen3.8 shared-runner execution** landed on `main` (see
  `git log`). Verify saves and accepted-prefix commits have stable per-owner
  staging and remain pinned until GPU retirement. N-gram rows materialize on
  demand into bounded device slots; the full source table need not be resident.
  Focused GPU checks prove native verify and commits of one to three rows,
  including joined owners and captured replay. All three formats' commits
  match the legacy commit executor on cloned state. Plain-mode equality
  remains covered across all three formats.

  G3c1 supplied the initial fake-backend chunk-feature lifecycle controls.

- **G3c2a: Qwen3.8 MTP progress and persistence** landed on `main` (see
  `git log`). Opt-in component contracts declare the cache/token/feature
  relation and repeat-pending schedule. Completion-owned control publication,
  prompt replay companions, authenticated private writes and layout-derived checkpoint
  tails preserve pending work through checkpoint, spill and fresh-node adoption.
  Exact native single-owner GPU controls cover 24 acceptance/discard, changed
  anchor, retry and persistence actions, including changed-input multirow graph
  replay. Qualified fake joined consumers cover shared lifecycle and failure
  controls. Native joined repeat-pending execution remains unqualified:
  whole operations and both lookahead hints refuse before settlement, unit
  changes or admission. Default mappings and serving remain unopted.

  G3c2b1a adds default-off typed owner-product proofs, per-owner primitive
  rewrites before width-sensitive preparation and bound native-recipe audits.
  Structural/descriptor controls include combined allocation-fact forwarding
  through owner splits, patterns and primitive fallback. Current component
  sources and runtime do not opt in; joined repeat-pending remains refused.
  G3c2b1b1 adds a default-off selected-HC recipe correction: actual canonical
  output rows protect the final low/high vector pair; ordinary wide fallback
  remains unchanged. The bounded native screen matches both owners' K1/K3
  full-vocabulary heads and final HC products. A normal model-free `hc.mix`
  regression matches complete mixed outputs for ragged owners and replays
  changed inputs through the same recorded program. See
  [the selected-HC evidence](experiments/m36-selected-owner-hc/README.md).
  G3c2b1b2 still owes native joined target/drafter heads and state through each
  arm's own outputs, full continuation/replay, old/new persistence and bounded
  admission. G3c2b1b and G3c2b1 remain open; native joined admission stays closed.

  Required G3c2b1 qualifies joined MTP quality and routing against a reference
  implementation of the same quantized checkpoint; original-model evidence
  is needed for original-agreement claims. Preserve source
  bindings, owner coverage, every selected contribution and same-implementation
  state restoration/capture. Choose kernel widths and arithmetic by quality,
  measured speed, reuse, simplicity and resource cost. Equal quality/speed
  with simpler shared code, or measured CPU/memory-traffic savings hidden by
  another bottleneck, can justify a switch within the migration gates.
  G3c2b2 then composes native, GGUF and MTP serving and passes state, quality,
  swap and the matched 1% gate at every qualified cohort. G3d deletes the old
  path after those gates. G3c2b2a now has a reviewed sole-owner serving
  candidate in `.claude/worktrees/m36-qwen-solo`, copied to
  `spark:~/src/m36-g3c`. Actual `qwen-solo-screen06` and `07` pass four
  model comparisons and two startup controls; `compile08` passes the context
  margin and invalid-token controls after review corrections. Final Spark
  checks pass 2,500 native non-model and 1,955 CPU-only tests; all 124
  model/timing registrations pass after two authenticated selected-head fixture
  reruns. Raw receipts are `qwen-solo-final-parallel`, `final-cpu`,
  `final-models02` and `fixture-final02` in the matching Spark scratch
  directories and root coordination scratchpad.
  The broader joined, state, quality, swap and matched 1% gates remain open.
- **G4a: DeepSeek V4 components, mappings and the DSpark stage** landed on
  `main` (see `git log`). G4b1 has landed: complete plan keys include
  each owner's baked HCA block count and compressed read width; equivalent
  dimensions reuse plans while actual positions refresh the typed inputs.
  The fragment, reference and key share the state's compressed-cell padding,
  independent of the raw read envelope. Primitive lowering and its descriptor
  checks pass with the native indexer enabled or decomposed. G4b2a has
  landed four local, opt-in forms: Q-head, compressor pooling, full-carry HC
  pre and flat-norm HC post. Their guarded primitive fallbacks, descriptor
  accounting and existing-kernel operands are checked, including reuse of
  the same capture after changed inputs and pitched compressor views at
  ratio-four and ratio-128 block boundaries. The primitive dummy rows are
  literal constants; masked/nonfinite pooling agrees with the reference.
  The graph-order comparator authenticates only that exact dummy correction
  and continues to check all other nodes and dataflow. The indexer
  contract remains G4b2b2. G4b2b1 adds opt-in state-bank attention: exact
  owned-layout adjacency, bound-address checks and whole-span state use
  remove the primitive cache copy while preserving its indexer, TopK, masks
  and Attention parameters. Unsupported views, missing proof, gaps, writes
  and rounded output requirements retain primitive fallback. Operand GPU
  comparisons at D256/D512 are bit-identical with unequal owners and changed
  inputs replaying the same capture; state admission and resident backing
  are checked independently. Default policies and whole-artifact GPU parity
  remain G4b2c's. Owed by G4c:
  - binding the hash-routing side ids, checked against the expert count as
    `CheckDsv4HashRouting` does;
  - the compressor plan inputs;
  - the injection's features, with a drafter slot for every target slot.

  G4b is split into G4b1 (past-safe plan-shape/key contract, matching MQA
  reference width, focused primitive-lowering correctness) and G4b2 (fast
  patterns/default policy, structural parity and GPU execution). Its local
  forms are complete in G4b2a and state banks in G4b2b1. G4b2b2a adds
  exact visible-prefix selection, duplicate-safe bit-preserving masks and
  owner-authenticated completed-block count inputs. Scalar/native operand
  controls cover arbitrary score bits and ID views, unequal owners and
  changed-input captured replay. Existing score/TopK, MQA version1 and
  default policies retain their semantics. G4b2b2b1 adds typed half-rounded
  query scoring, an independent wider reference and literal prefix overwrite;
  the portable score path funds an explicit F32-to-F16-to-F32 query conversion
  and accepts pitched half keys through the checked shared copy. G4b2b2b2 shares
  the actual native Lid score producer with standalone scoring, funds literal
  tails and authenticates alignment facts before native selection. Exact
  original-helper controls cover 462 native operand scenarios, pitches, tiles,
  row groups and changed-input replay; state reads retain the full declared key
  closure. No component/default consumer is introduced. G4b2b2b3a adds physical
  reservation guarantees, exact owned/borrowed state facts and authenticated
  planner-pass forwarding, with placement and actual-binding audits. It introduces
  no new component recipe. G4b2b2b3b1 adds owned generic execution/domain values
  and a single accounted opaque snapshot allocation with issuer, generation,
  metadata and completion authentication. Legacy raw APIs retain their absent-
  identity behavior; no production source declares a required identity yet.
  G4b2b2b3b2 must qualify the full execution build and actual late resources;
  G4b2b2b3b3 must enforce source/plan/kept compatibility and complete operation
  admission, including both futures and companions, before G4b2b2b3c makes
  an opt-in rounded-score component or full-mask fusion reachable. Output-A/
  products, mixed encodings and full policy remain G4b2c. G4c then
  binds the hash-routing IDs, compressor inputs and injection features and
  runs serving/state/quality/swap/1% controls; G4d deletes the M3 path. The
  active checkout is `.claude/worktrees/m36-g4b`; its assigned Spark copy
  is `spark-b:~/src/m36-g4b`, with jobs in `~/scratch/m36-g4b`.

  Smaller items from the review (in the deferred list below):
  - `route_scale` 0 should map to 1 in the importer's mapping;
  - `silu_clamp` refuses a single unset limit;
  - `WritesFor` over-refuses verifies on recent rings.
- **N3a: bounded import transform primitives** are implemented and reviewed
  in `.claude/worktrees/m36-native-import`. The internal API covers the
  historical BF16-to-F32 `norm.plus_one` and `ssm.a_neg_exp` operations and
  complete F32/MXFP8 `gdn.value_heads_tiled` permutations. Scalar refusal
  leaves output unchanged; integer nearest-even output encoding preserves
  F32 subnormals, and each packed part has an independent monotone cursor.
  Source-normalization consumers and the remaining folds are N3b; the
  writer, embedded-spec integration, CLI and archive events are N4;
  re-import and family admission remain N5. No runtime path is activated.
  Spark-b passed all seven focused tests with ordinary floating-point
  settings and with flush-to-zero enabled, including both exhaustive
  65,536-input historical BF16 reference hashes. The external control
  checked actual gradual/flush arithmetic and restoration of the original
  floating-point environment. This is target-libm evidence, with no
  cross-libm qualification.
- **N3b1: owned GGUF Gemma 2/3/4 target normalization** is implemented in
  `.claude/worktrees/m36-native-import` on `codex/m36-native-normalize`;
  independent source review and adversarial challenge found no blocking
  issue. Compiled source tables classify all keys and tensors, retain
  independent byte-copy routes, and validate common component roles,
  formats and shapes. Tied heads, Gemma 2/3's already-applied GGUF norm
  fold, Gemma 4's integer layer lists, dense/MoE gates, shared rotary table
  and rank-three expert formats are preserved. Gemma 4 adds no fold.
  Quantization calibration provenance has the explicit `quantize.imatrix.`
  auxiliary disposition; unknown quantization keys still refuse.
  Allocation-free preflight bounds new metadata and work; no payload
  allocation, publication, CLI or runtime path is added.
  On Spark, 2026-10-11, checkpoint02 (`d0c1a9cc` source manifest) passed
  70 host controls, including all 16 normalization cases, and four real
  target GGUF normalizations: Gemma 2 2B, Gemma 3 4B QAT, Gemma 4 26B-A4B
  and Gemma 4 31B. The model controls compare complete normalized specs
  with the existing v0 profiles, formats and source routes. Raw XML/logs
  were audited with zero failures/skips and source guards passed. The
  earlier Gemma 2 provenance-key refusal remains recorded as the reason
  for the explicit namespace correction. Final Spark checks passed: 2,516
  native, 1,971 CPU and all 114 model tests, with no skips. Main build and integration checks passed. Assistants, Qwen/DeepSeek, raw
  containers and remaining folds keep N3b open; writer/publication and
  re-import/serving admission are N4/N5, not evidence from these checks.
- **Model archive design** landed on `main` (`340e65c4`). This is the
  owner's direction, not part of the M3.6 plan proper:
  - `docs/model-archive.md`, with layout, manifest, catalog (meta-index)
    and a downloader quickstart;
  - a decision entry, with libcurl accepted for a native `llmp install`;
  - an AGENTS.md rule;
  - interim `tools/archive-model`, `stage-model` and `archive-catalog`.

  The owner's download agent (Gemini via Antigravity, which reads
  AGENTS.md) uses it from the workstation under `hostlock shared`, with
  `HF_TOKEN` in the ignored `/.env`. N4's native `llmp import` reads a
  verified archive or staged source and records an import event in the
  archive catalog. Native `llmp install` remains M5 per D-114.

Next, in order: G3c2 (Qwen3.8 composed MTP, serving and the 1% gate for
native, GGUF and MTP); G3d (deletion); G4b2, G4c and G4d;
Qwen2 G5a–d. N3b/N4 (remaining folds and source normalization, writer and
`llmp import` from verified
archive/staged source) can proceed alongside Qwen-Image Q1–Q3; N5's
all-artifact re-import and family gates follow both, then N6's adversarial
challenge. T-slices, K1 and R1 follow their shared-engine prerequisites.
Native `llmp install` is M5, not N4.

Also worth a slice: a DeepSeek budget cliff. With about 1.1 GiB less
MemAvailable at start, qwen3.8→deepseek took 16.1 s instead of 8.15 s and
deepseek→qwen3.8 took 19.7 s, against the 20 s bound. It was measured on
spark-b during the weight-handle review.

## Remaining slices (blueprint, 2026-10-08)

The slice plan as designed before work began. Where it differs from what was
built, docs/engine-components.md's "as built" text and the commits win.
Done so far: waves 0–2, P1, G1a–d, G2a–d, G3a–b, G3c1 and G3c2a, G4a, G4b1, G4b2a, G4b2b1, G4b2b2a, G4b2b2b1 and G4b2b2b2, N1.

#### Wave 3: families, one at a time through the gate

Component authoring for the next family (its `a` slice) may run ahead in new files while the previous family is being gated.

**Gemma2 + Gemma3 (G1):**
- **G1a.** `spec/mappings/gemma2.*`, `gemma3.*`; `SynthesizeV0` from `Gemma2Profile`/`Gemma3Profile`; approval as data. Add `SpecFacts` (context ceiling, applicable settings) used by `model_settings.cc` for these two architectures.
- **G1b.** Patterns and wave variants for the production recipes (`serving.cc:1833-1858`, `2278-2304`):
  - norm-mul, norm-mul-rope, norm-mul-add;
  - quantized GeGLU (MMVQ);
  - rope-store;
  - grouped KV stores (`set_rows_group.h`);
  - shared Q8 preparation (`shared_q8.h`);
  - owner-root attention with packed, owner and flexible prefill and bounded roots (`fattn_owner.h`);
  - device masks.

  Test: the I6 parity test, extended to every wave shape the runners plan at setup.
- **G1c.** `ComponentStepSource` in `ModelRunner`; `EngineLlm` for gemma2/gemma3 behind the switch. On Spark:
  - stock heads exact (the current controls);
  - state, spill, restore and kept/restart controls, with old kept state refused under the new IDs;
  - swap controls and the near-8K switch;
  - the 1% A/B against S0's frozen binary.
- **G1d.** Delete `engine/gemma{2,3}_{runner,plan}.*`, `kernels/ggml/gemma{2,3}_graph.*`, `model/gemma{2,3}.*` (keep tokenizer and chat), the `serving.cc` classes, `ServingOptions` `gemma2_*`/`gemma3_*`, and the dependent tests and benchmarks. Remove their boundary-checker exceptions.

**Gemma4 26B-A4B/31B + assistants (G2):**
- **a.** Components: softmax top-8/128 router with expert scales; routed products + ordered reduce; parallel dense FFN; global K=V tie + V-norm; proportional frequencies. The assistant is a drafter stage that `reads` target KV (borrowed contract) and the post-final-norm tap.
- **b.** Patterns: Gemma route/reduce (`gemma_moe_fusion.h`), owner-root rewrite (`gemma4_attention.cc`), common-width reads.
- **c, d.** Wiring and gate, then deletion: landed.

**Qwen3.8 native + GGUF + MTP (G3):**
- **a.** Components:
  - HC residual (4 streams, sigmoid mix);
  - GDN + convolution state (snapshot rollback + commit hook);
  - QSA indexer + block pooling + sparse attention;
  - output gate;
  - top-10/512 router + sigmoid-gated shared expert;
  - n-gram side input (`rows_on_demand`, `engine/ple_rows.h`);
  - MTP drafter.
- **b.** Formats and patterns: MXFP8 vector/CUTLASS GEMM, NVFP4 CUTLASS grouped + `llmp.moe.*`, `hc.*`, `gdn.*`, `qsa.*`; GGUF encodings through the same spec.
- **c.** Waves (joined products ≤16 rows, head sharing T34/T19, 2,048 read alignment), gates for native, GGUF and MTP.
- **d.** Deletion, including `qwen38_wave_plan.*`.

**DeepSeek V4 + DSpark (G4):**
- **a.** Components:
  - Sinkhorn HC (the same HC component, mixing variant);
  - CSA/HCA compressed KV + lightning indexer top-512;
  - MQA with sinks + window ring;
  - √softplus + bias routing;
  - token-hash side input (host table);
  - clamped SwiGLU;
  - YaRN NORM rotation;
  - DSpark block drafter with injection + Markov head.
- **b.** Patterns: `vecq`, D2R, compact, pair-GLU, HC-post experts, output-A, the Hadamard product. Mixed-encoding fused operands go through F4 composition.
- **c.** The `Dsv4WaveGraph` equivalent through `WaveComposer`, `--exact` as "patterns off", and the gate.
- **d.** Deletion.

Each `a` slice is about a day of CPU work. Each `b` slice is one to two slices plus the structural parity test against that family's builder. Each `c` slice is Spark-heavy. Each `d` slice is mechanical.

#### Wave 4: native importer (heavy path; D-entry, version bump)

| ID | Content | Notes |
| --- | --- | --- |
| N1 | v1 format: `llmp-artifact` manifest with spec, per-tensor `TensorFormat` + parts, folds, importer identity; reader accepts v0 and v1 | `artifact/` + `docs/artifact-format.md`; schema decision |
| N2a–e | Readers: GGUF; safetensors + `config.json` (ModelOpt NVFP4/MXFP8, AWQ/GPTQ, compressed-tensors); diffusers → D-089 composition; MLX; EXL3 | `src/import/readers/*`; parallel, disjoint |
| N3 | Fold catalog and mapping application, reusing wave 3's tables | Partial: N3a implemented; N3b remains open |
| N3a | Internal BF16-to-F32 `norm.plus_one` / `ssm.a_neg_exp` primitives and F32/MXFP8 `gdn.value_heads_tiled` permutation cursors | Reviewed primitive scope only; validation status above |
| N3b | Source normalization and remaining folds: fused-QK splitting, ties, expert slicing, NVFP4 scale-atom relayout and the other migrated family/container mappings | Open; N3b1 GGUF Gemma 2/3/4 targets complete and reviewed; 70 focused host + four actual targets, 2,516 native + 1,971 CPU + all 114 model tests passed, no skips; main build and integration checks passed |
| N4 | Writer (deterministic, one rename), `llmp import` CLI, admission-gate harness; import from verified archive or staged source and record catalog import event | Public CLI → decision; native `llmp install` is M5 (D-114) |
| N5 | Re-import every artifact; lossless-equivalence checks against v0 bytes; family gates; the owner-chosen MLX checkpoint | After G3/G4/G5 and Q execution paths and N3/N4; Qwen2 fixture correctness gate follows the protocol exception below |
| N6 | Adversarial challenge rounds | |

**Qwen2 fixtures (G5, required by plan exit).** The Qwen2.5-0.5B FP16 and
two EXL3 fixtures need shared-engine execution even though they are not served.
G5a is complete: both encodings synthesize shared-component specs and all
three installed-artifact metadata controls pass. Shared FP16's fused and
unfused 76-token control trajectories are byte-exact; its screen/full cases
are not the historical control/heldout four-arm gate
([evidence](experiments/m36-qwen2/README.md)). G5b adds shared GGML/EXL3 lowering
with the approved reconstruction tuning facts/shapes; G5c runs them through
the shared runner and proves their existing correctness controls plus state,
spill, restore and kept-state behavior; G5d removes the legacy execution path.
The fixed M3.6 protocol explicitly gives these fixtures no speed baseline
because they are not served; their gate is the FP16 bridge and EXL3 Tier C
correctness controls ([protocol exception](experiments/m36-baselines/README.md#quality-controls),
BP-F3 moves resident speed to M5). G5a/b can proceed alongside other component
authoring; G5c depends on the lowering, and G5d follows its controls. N5's
all-artifact re-import/gates and K1's every-migrated-LLM criterion include
these fixtures where applicable.

#### Waves 5 to 7

- **Q1–Q3, Qwen-Image.** Q1a's shared stateless `norm.channel` component is
  complete: source-encoded F32/BF16 scales, clamped L2 normalization and
  optional SiLU with F32 arithmetic, without a legacy rounding gate. Main
  native build and all 96 affected host cases pass on Spark B (2026-10-12).
  Q1b's shared `norm.adaptive` and `residual.gated` fragments now preserve the
  original carry and consume explicit dynamic conditioning with F32 arithmetic.
  Main native build and all 99 affected host cases pass on Spark B (2026-10-12).
  Q1c's typed WHCN graphs now compose nearest2 upsampling and channel-changing
  `conv.spatial`, reusing the native role visitor, common planner and CPU
  reference executor. Main native build and all 239 affected host cases pass
  on Spark B (2026-10-12), including three spatial and 45 planner controls;
  source quality checks pass. This adds no standalone executor or token phase.
  Q3a's shared spatial lowering now passes a full main Spark A native build and
  129 host controls plus one actual VAE-span screen: 4,737,024 finite outputs,
  two eager executions, two captures and two same-selected replays. It reuses
  existing F32 kernels, actual source parts, the owned backend and common
  completion path; the selected physical convolution does not support BF16.
  Source quality checks pass; see the
  [bounded report](experiments/m36-shared-spatial-execution/README.md).
  Q3b's complete first-frame decoder now passes through the same shared path:
  109 components preserve source roles and owner-local residual values, with
  optional unused scalar-reference payment omitted for the GPU graph. Focused
  native compilation and 52 host controls pass, followed by the full main
  Spark B build and 227 host cases across 17 targets. The actual 2x2-to-32x32
  decoder case passes all 4,096 finite outputs plus 4,608 local source-reference
  outputs, with two eager executions, two captures and two same-selected replays.
  These bounded execution witnesses do not establish original image quality or
  comparative speed.
  Q3c's complete timestep conditioner now passes through shared projections
  and explicit owner-ordered output ports. Four actual BF16 source parts feed
  two unequal rows and both full branches. Main Spark B native build and all
  443 host cases across 25 targets pass; the actual case checks 204,800 finite
  values over five completions, with one eager execution, one capture and three
  retained replays, including changed/restored input and unchanged zero-time rows.
  Q3d now consumes that modulation in one complete shared block 0 graph, using
  existing adaptive/gated formulas, three NORMAL rotation slices and explicit
  causal-text/all-image attention. Scoped native compilation and 63 host controls
  pass; both actual conditioner and block cases pass, the latter checking 229,376
  finite values over seven steps with changed/restored inputs and retained replay.
  The full main Spark B native build and 452 host cases across 25 targets pass,
  with the actual-tested model binary unchanged; final source quality checks pass.
  All 32-block flow, broader geometries/reference quality and `EngineImage` remain
  open. Qualify numerical choices rather than imposing the legacy
  pipeline's rounding recipe; retire `qwen_image_runner.*` and `pipeline.cc`
  roles after their shared replacements pass.
- **T-slices, structural techniques.** One per row of `engine-components.md`'s retired table: format layer T04/T11/T58/T95; wave composer T15/T16/T20/T29/T59/T64/T88; patterns T10/T31/T56/T86/T92 (+T09/T41/T57 adoption); prefill T23/T26/T55/T93; state T67; decoding T39/T45/T81; components T22/T69/T54/T71/T72/T61; planner T96; pager T68. Adopt where qualified comparisons improve speed, quality, reuse, simplicity or resource cost.
- **K1, state encodings run.** A lossy KV encoding as a per-alias setting (public config key → D-entry), plus the long-context quality report.
- **R1, re-keying cleanup.** Remove `ModelKeySpec::architectures` (replaced by spec capability predicates); remove `RunnerContextCeiling` (from components' position limits); `ArtifactFactsOf` from the spec. The layout-ID bump itself already lands per family through `KeptLayoutId`.

#### Dependency graph (critical path)

```
F1 ─┬─ F2 ─┐            ┌─ F6 (discrete)
    ├─ F3 ─┼─ F4 ───────┘
    ├─ F5 ─┴─ F7
    └─ L1 ─┐
L2 ────────┴─ L3 ─┬─ L4 (opt)
                  └─ L5a ─ L5b ─ L5c ─ L6
F5 ─ I1 ─┬─ I2 ─┐
         ├─ I3 ─┴─ I4 ─┐
         └─ I5 ────────┴─ I6 ─ G1a..d ─ G2 ─ G3 ─ G4 ─ G5 ─ Q ─┐
                           N1 ─ N2 ─ N3 ─ N4 ───────────────────┴─ N5 ─ N6
                           T, K1, R1 follow their shared-engine prerequisites
                    P1 (any time after I5)
```

#### The sequencing questions answered

- **v0 mapping at read time: yes.** F2 plus `SynthesizeV0` keep the runtime off any new on-disk format until N1. The cost is one data table per architecture, which N3 reuses. Parameters come from the compiled profiles, so admission stays profile equality until the importer makes the approved list data.
- **Should the lifecycle reconcile precede the IR? Yes, for L1 to L3.**
  - The IR's state tensors consume L1's contract.
  - L3's semantics (guards, quarantine, recovery) are where swap and kept-state controls can fail, and that should be isolated from 1%-speed debugging.
  - L4 is the cheap proof on real models. The deciding factor is cost: about a day for the twins. Skip L4 for Qwen and DeepSeek.
- **Coexistence.**
  - During a family's task, the legacy runner stays the default and `component_engine` selects the new one.
  - New layout IDs make each engine refuse the other's kept state cleanly; test that refusal.
  - The task ends with the deletion slice, so no release ships with both.
  - Shared-skeleton changes (L2, L3) also touch legacy runners. Spot-check them with S0's protocol on a legacy arm before the family's own gate.
- **How the A/B compares binaries.** The gate arm is S0's frozen pre-M3.6 binary against the candidate, alternated ABBA on `spark-b` with the same artifacts and settings. An in-binary legacy arm is only diagnostic, for attributing a gap to orchestration rather than skeleton drift. Runs over 10 minutes need D-085 justification; split them into supervised `spark-job` batches of 600 s or less.

---

### 5. Risks and open questions for the owner

1. **Commit granularity.** workflow.md says one commit per plan task, but a family migration is 4 or more slices of large diffs. Split the M3.6 checkboxes into sub-tasks (for example G1a to G1d each a task) so each lands reviewed and committed?
2. **Pattern policy: global, or each family's current selection?** Optimize-once means one global pattern policy. Today Gemma2 serves with `fuse_norm_rope=false` and Gemma3 with `true`. Under a global policy, Gemma2's first migration may change its arithmetic. Two options:
   - accept that under the "equal or better quality" gate; or
   - require first migrations to reproduce current selections exactly (expected bit-identical), and widen patterns in the structural slices.

   I recommend the second option: it makes the 1% A/B cleaner.
3. **Retire default-off diagnostic options instead of porting them.** Examples: `bounded_whole12`, `gemma_row_invariant`, the packed-prefill controls, `ds4_hca`, benchmark-only probes, and most of the 21 Gemma4 and 10 Qwen benchmark harnesses tied to deleted runners. I recommend porting only what the production recipe, the gate or the swap table uses.
4. **GGML graph as the private lowering target.** Confirm this is consistent with D-107: the IR stays the source of truth, `ggml_cgraph` exists only inside `kernels/ggml`, and the engine names no library. It carries GGML's rank-4 and I32-stride limits into the IR verifier for now.
5. **Engine core CPU-buildable.** This amends D-096's "CUDA builds only" for `llmp_engine_core` and adds a fake device runtime. It is required for the fake-backend tests the plan names.
6. **Multi-drafter semantics.** Is "N drafters" a cascade (drafter i extends drafter i−1's tokens, one verify), a per-unit choice among alternatives, or both? L5c needs this before it is written.
7. **The sm_86 proof logistics (F6).** It needs a Qwen3.8 artifact reachable from the workstation and time on the 3080 Ti, which is shared with latency-sensitive measurements. The owner should pick the window.
8. **Public surfaces that will change and need D-entries and version bumps:**
   - `llmp-runtime`'s per-family `ServingOptions` flags (`commands.h:71-145`);
   - the `[models]` keys' architecture gating;
   - a new per-alias KV-encoding key;
   - `llmp import`;
   - the v1 artifact format.

   Confirm whether the per-family runtime flags count as public CLI or as diagnostics that can be removed freely.

Risks without a decision:
- Wave-shape plan-key growth from one-graph waves. This is Qwen's composed per-slot plans becoming one graph per combination. Mitigate with read-cell bucketing.
- DeepSeek's Sinkhorn, HC-post and lightning-indexer CPU references must be written from upstream semantics; no primitive composition exists today (portability.md).
- Peak memory at the gate depends on IR placement matching `PlaceActivations`. I5's extent test is the early warning.

#### Critical files for implementation
- /home/pmeenan/src/llmp/src/execution/registry.h
- /home/pmeenan/src/llmp/src/kernels/ggml/graph_plan.h
- /home/pmeenan/src/llmp/src/engine/live_state.h
- /home/pmeenan/src/llmp/src/artifact/representation.h

## Deferred items from the reviews

Every builder's and reviewer's open item that was not fixed in its own slice,
by the slice that raised it. Take each into the slice it names, or into its
own slice when it names none.

# M3.6 follow-ups queue (main agent notes)
- LEGACY BUG (from L3 review, 2026-10-08): Qwen3.8 and DeepSeek legacy runners' Spill and StateWrittenBack mark a stage saved while a verify still owes restore/commit or awaits Accept; their Rollback records no write → next incremental spill evicts rewritten extent as unchanged (stale bytes on restore). Paths: ReadState, a turn-checkpoint lookup returning before SettleFor, set-aside spill, spill with no keeper. Fix in place (bug fix allowed) with a test.
- L5c precondition: WritesFor not usable as-is for DeepSeek compressor rings (per-row persist via Dsv4ChunkWrites persist_src/dst; no commit) and Qwen per-row K/V/indexer saves + completed blocks only + MTP stream rows. Need ring-shaped cell rule or per-row attribution, and byte-for-byte tests vs Dsv4ChunkWrites/DsparkWrites and qwen38_runner.cc:3495-3535 verify saves.
- L6: kept record version bump (D-105) + decisions entry: record names each stage's file + identity; old IDs refused. KeptLayoutId digest must include weights/artifact identity; Setup should refuse all-zero digest once wired.
- Heads component: mask logit rows past tokenizer (Qwen3.8 248,077–248,319 selectable today).
- Decode templates speed leads: vector template register blocking at 16 rows; grouped warp-per-row variant for decode; multi-stage tiles.
- v1 manifest: stacked experts' tensor scale as one F32 [matrices][stacked] part; Exl3Codebook names; member alignment lives in artifact placement (Qwen-Image 256 B).
- format_map: v0 Q8_1 size 40 vs GGML 36 (refused today).
- SlotLifecycle (for L5c/L6): no DiscardVerify/Settle verb (callers must Settle+Rollback via live() first); CheckActive doesn't refuse mid-restore steps; Borrow/Return carry no token (stale Return after recovery could decrement a new borrower).
- Planning (shared): one-shape-per-band probes for graph reader index instead of relying on the 25% margin; qwen38_wave_plan.cc:1012 plans at runtime with activations==0 (can grow index_bytes after startup guard).
- fused.mul_mat_glu operand order (gate 0, up 1, input 2) assumed by decode declarations; IR must define it. Qwen3.8 token_embd gather not in proof.
- KeptLayoutId must take spec::BoundDigest(spec, artifact IDs) (wire at L6/G1c).
- Spec JSON: decisions.md entry when the importer first writes it to disk; v1 manifest holding the spec needs the strict-JSON nesting limit raised (spec uses all 8 levels) or the spec as its own document.
- Gemma2 mapping uses rsqrt(key_length) (llama.cpp's rule below 27B); 27B needs its own rule. norm.plus_one asserted for GGUF sources (llama.cpp converter applies it).
- EXL3: capabilities declare nothing row-invariant/capturable; check vs the M3 Qwen2 EXL3 path's capture when the fixtures migrate. Reconstruction capability limited to sm_121 (GB10-only cuBLASLt pins).
- KV encodings: the bounded dense Q8_0 D64/D256 opt-in now wires existing quantized kernels, blocked state layouts and per-alias settings/kept identity (D-117). Broader context, all migrated families and D512/specialized or borrowed state remain open; the kernel inventory does not qualify them.
- Registry: capability needing tuning data (EXL3 reconstruction only works at pinned cuBLASLt shapes m=145/1023/1024) — planner (I5) must drop candidates without a pin, or registry gains a 'needs tuning data' notion. EXL3 priority order differs from upstream (recon >144 rows, fused >=1024): migrating fixtures need measured ranks.
- Importer/mapping language: Gemma2 46-layer / Gemma3 62-layer (27B) query scale 1/sqrt(width/heads) not expressible (layer-count-dependent rule or refusal needed); require the target stage first in specs; check cycles between stage reads; Gemma3 sliding_window 0 → all layers global incl. base; verify mappings against the pinned llama.cpp (reviewer used 8019dc56).
- L5c: Admit with advance=false (verify, draft) doesn't check past vs slot positions (drafter stages may differ: define the rule); PrefillOutcome drops the step's joined outputs.
- Test infra: GPU test flakes under `--parallel 20` (CUDA "no context", subprocess abort, CudaPagedNodeSleep wake lag) likely memory/GPU contention — only models-labelled tests hold RESOURCE_LOCK llmp_gpu_model. Consider a GPU resource lock/parallel cap for all gpu-label tests. "no context" message now carries the CUDA error name (in L5 slice).
- Lossy KV: exit criterion says every migrated LLM — D 512 (Gemma4 global, DeepSeek V4, DSpark) has no upstream quantized attention: needs our own D512 quantized vec path (or define the opt-in per component). Convert-then-tensor-core prefill path for quantized caches: measure vs vector kernel when a family opts in (D-108).
- RE-051 ASan + InstallThreadSignalStack crash at thread exit: fix (keep an existing signal stack, or sanitizer presets set use_sigaltstack=0) before the next ASan tier run. GemmaJoinedGpu HostCopies could get the same destructor check.
- I6/G1b: kOwnerAttention defined with queries [D, owners·R, H, 1] and mask rows at owner·R; GGML owner carriers lay queries [d,1,heads,owners] with 32-row-padded masks concatenated on dim 3 — lowering repacks or amend the fused kind.
- Components: attention K/V cell_bytes should use model::CellBytes (lossy KV) after merging d3279fcd. End-to-end GPU tolerance can't see softcap/query-scale/global-rope-scale errors — parity test pins them; the reference-vs-production 0.46–0.70 logit gap is not Q8_1 activation rounding (matches earlier cross-engine Gemma3 0.72) — unexplained, investigate if a migration quality control disagrees.
- BEFORE Qwen3.8/DeepSeek migrate onto 2,048 joined alignment: all-or-nothing widening acquisition in SlotLifecycle (one node_.Acquire over every owner's widening, then each owner's Use; refusal carries no slot → RunWave runs all alone). Today a refused later widening leaves earlier owners' widening kept (bounded; = current Qwen3.8 behaviour).
- L5c: per-stage read alignment and context in SlotLifecycle (drafter stages); own phase and LoneRead must use the stage's value.
- Family migrations onto decoding modes: spec `reads` → state contract target_writes/borrowed_from_target (components/spec emit them); Gemma4 retained features as a tap + FrozenBorrow epoch/generation guards; DeepSeek draft+verify chained in one job + Dsv4SameWidths shortening; Qwen catch-up + passes in one job, injection pass inside prefill; no multi-level verification yet.
- I6: DeviceFacts.tuned populated from exl3-recon-pin.json; real GGML capabilities: group ggml.mul_mat, batch on llmp.set_rows.grouped, accepts predicates for norm-rope/norm-add width limits and MMVQ routing. G1b: OwnerAttnParams gains carriers' logical cohort + owner offset (op-set change); device masks as a kMaskBuild lowering choice; per-family default policy text in the architecture mapping (G1a).
- Importer/Qwen3.8 mapping: the existing `gdn.value_heads_tiled` fold already denotes the grouped-to-tiled permutation of linear-attention heads (rows of in_proj_qkv/z/b/a, columns of out_proj), and `spec/mappings/qwen38.cc` asserts it for GGUF and the historical ModelOpt converter. N3a supplies bounded permutation primitives; N3b still must select/apply them to source tensors. No additional fold ID is needed. spec::kMaxSourceTensors (2^18) < Qwen3.8 ModelOpt's 301,730 tensors and kMaxSourceValues 4,096 may be < flattened config → raise caps or stack per-expert tensors. Readers follow links outside the checkpoint dir (HF cache) by design — revisit at N6 challenge.
- I6/G1b (from planner review): CapabilityQuery carries no node parameters → real GGML fused capabilities can't refuse rotation mode/dims or softcap≠50 — add params to the query or pattern-side checks; lanes port has no default on_stream list (legacy kept cuBLAS and BF16 GEMM on the stream) — the lowering must pass it; PlanKey omits PlanOptions::capturable, device facts and registry (OK for in-process cache only if capture follows from Phase).
- MLX admission gate checkpoint (owner-approved 2026-10-09): Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa80 (needs affine CUDA decoder + Qwen3.8 migration to pass the gate).
- Spec-reads linking (family migrations): BuildStep/ComponentStepSource must refuse stores into borrowed state refs and target-step stores into non-tap drafter state (StateRef/TensorUse carry no r/w flag); clamp borrowed reads to the target's committed positions (Gemma4 assistant) or check in Admit. Failed-job quarantine counts owed work as "wrote" (over-quarantine only) — a separate "run may have written" flag would make it exact. MTP catch-up clamp leaves stale rows until Reset (acceptance cost on long-context MTP).
- G1b: port frontier_head / narrow_final (production default: output rows gathered after the last layer's attention projection; last FFN + head at output-row width; gather + gathered-residual norm-add) — needed for bit-identity with serving. rope_store (view-only fused result) refused by the lowering. EXL3 lowering: populate DeviceFacts.tuned from exl3-recon-pin.json. CapabilityQuery has no output encodings (kPrepare llmp.q8_1 can't tell targets apart).
- Partitioning (for M4): TP stage never mixes split and replicated layers (memory/traffic trade); device cap 16; no embedding split by vocabulary; EP is local experts + mask + one all-reduce (no all-to-all; DP-attention + all-to-all EP mode not built).
- G1c wiring (from lowering review): GgmlStepLaunch ignores the stream it's given — backend must be built on the runner's job stream (or add a check); GgmlBackend::Launch callable once and dangles after runner releases its launch (one backend per runner; matters for swaps); lowering gives placeholder for 0 weight/state address (coverage check refuses later) — refuse directly; MmvqRouted counts ne1·ne2·ne3 vs upstream ne11 (3-D inputs only); rope.ext/soft_max/rms_norm_mul capabilities accept more than the lowering builds (refused, not wrong).
- Parallel-GPU flake confirmed: ComponentsModelsGpu "no context: cudaErrorMemoryAllocation" under parallel run — memory pressure; consider gpu-label RESOURCE_LOCK / parallel cap (test infra).
- NEXT IMPORTER SLICE (owner direction 2026-10-09, SUPERSEDES the descriptor-cache and link-containment items): remove the trust walk from the importer entirely — plain opens, links followed, no ownership/permission/FUSE/device-inode checks, no fd cache, no RaiseOpenFileLimit; open per read or briefly. Keep all content hardening (bounds, caps, fuzzing, D-009). Keep only: non-weight files copied verbatim into an artifact must resolve inside the checkpoint tree (HF cache blob links allowed); others skipped, not refused. Update engine-components.md. D-073's runtime config/storage-role walk: pending owner answer.
- AFTER the trust-walk slice (owner suggestion 2026-10-09, heavy path, pager/catalog): weight artifact file handles open only while the model is active (registered for serving and any extent may need page-in); close when the model leaves the active set with nothing resident; reopen on activation with a size check against the installed manifest (functional, not trust). First confirm the current open/close lifetime of artifact files. Socket fd limit raising (D-102) stays.
- G1c wiring (from L6): ComponentLlmBuild lends no cuBLAS handle (plans needing cuBLAS refused) — needs a mapped, charged workspace; device_greedy false for GGML parts until the lowering returns greedy I32 per segment; setup shapes measured at past=context-rows, read_cells=context — confirm the lowering accepts them; policy strings duplicated in serving.cc ComponentLlmBuild (copied from component_runner_gpu_test) — must come from the architecture mapping (G1a/b owns). Kept record v2: per-drafter adaptive depths not persisted; drafter cursors reset to positions after prefill/restore/adopt (acceptance cost only); adaptive wave form not wired.
- G1c (from L6 review): ComponentLlmBuild hard-codes policy strings and vocab (serving.cc:126+) — take policy from Mapping::policy (now on main via G1a/b) and logit width from the spec head; BoundDigest must include drafter artifact IDs once drafters exist; EngineLlm::AdoptPositionFor/PreparePositionRestoreFor pass the model's own KeptLayout (tautological; real check is kept::Check identity.layout) — thread the record's layout through; cuBLAS handle nullptr; device_greedy false.
- From G2a: EP partition: kView of an expert-local value doesn't propagate (stacked gate_up); assistant's shared_kv_layers not required == block_count; mapping hardcodes stacked gate_up and kv_tie by layer type.
- From G2a review: CheckReads accepts only post-final-norm taps and only stage 0 as target — widen for MTP drafters reading pre-final-norm hidden state (Qwen3.8, DeepSeek) and drafter cascades (i reads i−1) before those migrations; CheckStateWrites checks only primitive kCacheStore (fine before patterns; extend when fused stores appear in drafter steps); attention.borrowed splits by heads unchecked against target's split; ffn.moe declares pattern gemma_route_reduce not yet a fused kind. 26B gap explained: production arithmetic error + frontier-row cancellation (prompt), routing near-ties ~half of decode gap.
- G2c: Gemma4 runner turns narrow_final off when retaining features (keeps every row's feature); components' tap keeps only output rows so components still narrow — decide whether the assistant needs every row's feature; if so NarrowPoint must stay off when a stage has taps.
- From G1c review: V0Values uses the compiled 2B/4B-QAT profile for any artifact of that architecture (M3 classes accepted only the approved artifact) — approval list as data comes with the importer's admission gate; engine path with speculation=true and no drafter silently serves plain (M3 refused) — decide at G2c/G3c; plan.md 2026-10-09 note should mention the gate-only checkpoint switch when G1 lands. AFTER G1: shared improvement — move turn-checkpoint capture off the first-token path (overlap with decode), for every family.
- From G2b review: launch-level CPU compare can't see read/store order inside owner-attention steps (multiset) nor addresses/placement — only the GPU bit-identity test covers them. EP with one expert per device (E == devices) refused at device-graph verification (partition.cc:185 Local keeps rank 3 with ne[2]=1) — fix before a 128-expert × 128-device placement; EP mask-by-zero turns a non-finite other-device product into NaN (pre-existing; use select/where instead of mul if it matters).
- From G1c gate debug: building a plan for a new shape costs 5.6 ms on the engine vs 3.3 ms on M3 (BuildStep 1.8 + Plan 2.2 + lower 1.3 + bind 0.9); still on the critical path for the first decode step and the 3-row tail after the checkpoint boundary (NextPrefillHint stops at stable_boundary_). Follow-ups: planner/BuildStep speed; decode-first lookahead. Gate passes without them.
- From N1 review: CheckImporter duplicated (artifact.cc/composition.cc); runtime doesn't read v1 yet (ComponentLlmBuild uses SynthesizeV0 — switch at N4/integration); N4 writer must reproduce test writer rules, options_sha256 canonical doc, tensor name mapping, drafter-pair compositions; no independent v1 oracle until M5's verifier; ASan run of the mutation test owed at a workstation tier.
- From G1d review: engine_llm.cc ~682 RunPrefill future join check uses only CompatiblePrefill; Llm::PromptSession::CanJoin also needs equal want_head (wrong prediction = wasted ahead build only). Gemma 2/3 numerical oracle after deletion is ComponentsModelsGpu.StepsMatchTheServedModel (CPU ref vs served, 6-token prompt + 3 decodes) — consider a longer independent reference. Recovery ladder and mid-prefill cancellation tested on the fake backend only for Gemma 2/3. OWNER QUESTION pending: do src/components carry llama.cpp's MIT notice (docs/licensing.md).
- From G3a round 2: component code nests builder calls as arguments (qsa.cc b.Add(b.Reshape(..), b.View(..)), Attention(cache_view(K), cache_view(V))) → node order and graph digest depend on argument-evaluation order (compiler-dependent; order tests pin Clang). Sequence them into locals across components (Gemma too) before a second compiler matters. Pre-existing UBSan -fsanitize=function failure in cpu-asan preset: reference/decode.cc:257 and reference_decode_test.cc:368/421 call GGML to_float through ggml_to_float_t (GGML's own pattern) — cpu-asan preset fails 15 tests; decide: suppress for those calls or log RE.
- License (after G3a merge): tag gdn.cc, qsa.cc, hc.cc, ple.cc MIT AND Apache-2.0 + "2023-2026 The ggml authors"; add them to docs/licensing.md and the GGML lock scope. Sweep remaining untagged src/kernels/ggml files (validate.cc, validate_ext.cc restate GGML asserts; gemma_moe_fusion.cc judged independent). ExLlamaV3 notice guard: decode_exl3.cc in llmp_reference_kernels (tests/fuzz only today) — add IN_TREE_UNITS/provenance entry so a shipped link carries Turboderp's notice. YaRN attribution comment in exec.cc/kv_quant_rope.cu (package notice already ships it).
- DeepSeek budget cliff (fd review, spark-b): with ~1.1 GiB less MemAvailable at start (119.0 vs 120.4 GB), qwen3.8→deepseek took 16.1 s (vs 8.15) and deepseek→qwen3.8 19.7 s (evict 9.2 s) — near the 20 s bound; investigate. Optional: re-hash shard header region on reopen (same-path same-size replacement now read silently, D-056 amendment).
- From G2c: perf leads: decode +0.2–0.4% (plan build ~11 vs 8.8 ms/shape; joined steps admit each owner twice). Assistant: depth fixed at 3 (no adaptive, draft_rows unused for gemma4, no device verdict); 26B assistant served speed has no baseline.
