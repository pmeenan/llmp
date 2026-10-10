<!--
SPDX-FileCopyrightText: 2026 llmpalooza contributors
SPDX-License-Identifier: Apache-2.0
-->

# M3.6 handoff

Working state of the M3.6 refactor ([plan](plan.md), "M3.6 — Engine components",
[design](engine-components.md)) for whichever agent continues it. The
orchestrating agent paused on 2026-10-10 for usage quota; other agents
(Codex, Gemini via Antigravity) may continue meanwhile. **This file is
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
3. Merge: commit the worktree's work as a WIP commit on its own branch, then on
   `main` `git merge --squash <branch>`; resolve conflicts (keep both sides'
   intent; semantic conflicts need the reviewers' merge notes); run the merge
   check on a dedicated copy (`spark:~/src/m36-main` or
   `spark-b:~/src/m36-main`, steps files in `~/scratch/m36-main/`): shared
   set `-LE 'models|timing'`, CPU-only unit tests, and the touched areas'
   model tests in exclusive jobs under 600 s.
4. Commit on `main` with a message file: `[M3.6] <what landed>`, the body
   saying what changed, the review rounds and the check counts. Tick the plan
   sub-item. Remove the worktree and its Spark copies.

Family migrations add the 1% speed gate (docs/experiments/m36-baselines/README.md:
protocol, frozen baselines `~/.local/share/llmp/m36-baseline/1facc957-fixed`
on both Sparks, candidate installs under `~/.local/share/llmp/m36-candidates/`,
`tools/speed_gate.py`; the new arm runs with
`LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS=1` where the baseline's turn
checkpoints are broken, defect 3). Gate runs are timing work: exclusive
`spark-job --gpu` jobs, never beside a build on that host.

## State on 2026-10-10

On `main`: format layer, engine core, IR and planner, partitioning, spec and
mappings; Gemma 2 and 3 served on the shared engine and their M3 code deleted
(`4a6710ce`); Gemma 4 planned and lowered bit for bit (`6ff8a262`); Qwen3.8
components and mappings (`88efa86c`); artifact format v1 and its reader
(`5bce159d`, D-113); kept files and weight shards relaxed and opened only
while active (`4b8b35a9`, `961fdefe`); license tags for ported code
(`019dd77b`); AGENTS.md working rules (`db6cdff0`); `.env` through mise
(`171717ef`).

In flight (uncommitted worktrees under `/home/pmeenan/src/llmp/.claude/worktrees/`;
`git worktree list` shows the current set, and this section is updated as
each lands):

- **G2c: Gemma 4 26B-A4B and 31B with their assistants on the shared engine**
  landed on `main` (see `git log`). The 1% gate passed: 26B at 9 blocks,
  31B at 7, and the 31B assistant screen 4.4% faster. Records are in
  `spark:~/scratch/m36-gate/g2c-c9f7f2db*` and in the baselines README's
  G2c section, which includes the 4-row verify's bit identity with M3 for
  both profiles.
- **G2d: delete the Gemma 4 M3 code.** Mostly done: the work is on the
  local branch `m36-g2d-wip`, one WIP commit on top of the G2c commit,
  uncommitted on `main`. Its builder's state:
  - The code deletions and engine-only test conversions are done.
  - L3 (served four owners at the 8K/C4 envelope) and L4 (cross-variant
    kept tag with the real identities) passed before a last assertion fix.
  - A new served-assistant acceptance test is included.
  - The calibration key no longer carries `gemma31_production=true`, so
    stored Gemma 4 calibrations re-key; that needs a CHANGELOG line.

  Left to do:
  - Docs: plan ticks, engine.md, runtime-serving.md, engine-components.md,
    optimization-inventory.md, CHANGELOG Removed lines, the coverage map in
    the commit message, AGENTS.md.
  - clang-format, `tools/llmp_boundaries.py` and the header check.
  - The full Spark set on the final state, then a review.

  Open question: the assistant's draft-pass logits are no longer compared
  with the reference executor, only run to run plus the served
  acceptance test.
- **G3b: Qwen3.8 patterns, policy and GGML lowering**
  (`agent-a1d41dad140560dc5`, based on `88efa86c`; Spark copy
  `spark-b:~/src/m36-g3b`, helper `~/scratch/g3b/tgt.sh`). Stopped at a
  building checkpoint on 2026-10-10. Its `G3B-HANDOFF.md`, at the
  worktree root, has the detail. Plain mode is bit-identical to today's
  `Qwen38Runner` fast form on all three forms (ModelOpt CUTLASS, ModelOpt
  GGML experts, GGUF): 31 outputs and 242 state tensors, two owners, 24
  waves including captured replays and a step past the QSA budget
  (`component_runner_qwen38_gpu_test`). All three forms and the MTP
  drafter lower launch for launch on the CPU (`ggml_lower_qwen38_test`).
  Brief items 1–3 and 5–9 are done. Item 4 is done for plain mode only:
  speculation's GPU numerics (verify, commits of k of n rows, catch-up,
  draft passes) wait on G3c's wiring, because ModelRunner can't yet bind
  verify saves, run commit steps or stage drafter features. Left before
  review:
  - rerun the shared set, clang-format, `tools/llmp_boundaries.py` and the
    header check on the final state;
  - fill in the two `(STATUS)` placeholders in engine-components.md and
    plan.md;
  - merge main (it has moved since `88efa86c`);
  - a heavy-path review;
  - the full Spark set.

  Caveat: the GPU test keeps the n-gram table resident (+28.8 GB), which
  puts the ModelOpt run near the Spark's memory limit.
- **G4a: DeepSeek V4 components, mappings and the DSpark stage** landed on
  `main` (see `git log`). Owed by G4b, before the lowering accepts
  DeepSeek's new inputs: the plan key ignores `past`, but an HCA layer's
  block count and a compressed cache's read width depend on it. Either key
  `CompressorBlocks(rows, past, ratio, overlap)` and the compressed width, or
  make both independent of `past` by padding; `src/planner/step_shape.h`'s
  header gives both options. The mqa reference's hard-coded 256-cell
  compressed width must follow whichever is chosen. Owed by G4c:
  - binding the hash-routing side ids, checked against the expert count as
    `CheckDsv4HashRouting` does;
  - the compressor plan inputs;
  - the injection's features, with a drafter slot for every target slot.

  Smaller items from the review (in the deferred list below):
  - `route_scale` 0 should map to 1 in the importer's mapping;
  - `silu_clamp` refuses a single unset limit;
  - `WritesFor` over-refuses verifies on recent rings;
  - `HadamardValues` may differ by 1 ulp at n = 128.
- **Model archive design** (its own worktree). This is the owner's
  direction, not part of the M3.6 plan proper:
  - `docs/model-archive.md`, with layout, manifest, catalog (meta-index)
    and a downloader quickstart;
  - a decision entry, with libcurl accepted for a native `llmp install`;
  - an AGENTS.md rule;
  - interim `tools/archive-model`, `stage-model` and `archive-catalog`.

  The owner's download agent (Gemini via Antigravity, which reads
  AGENTS.md) uses it from the workstation under `hostlock shared`, with
  `HF_TOKEN` in the ignored `/.env`. Native import from the archive and
  install into it go to the importer's N4 slice.

Next, in order: G2c land; G2d; G3b review and land; G3c (Qwen3.8 wiring
and the 1% gate for native, GGUF and MTP); G3d (deletion); G4a review and
land; G4b–d; N3–N6 (the importer's fold catalog, writer, `llmp import`
and install from the archive, re-import of every artifact, adversarial
challenge); Qwen-Image Q1–Q3; T-slices; K1; R1.

Also worth a slice: a DeepSeek budget cliff. With about 1.1 GiB less
MemAvailable at start, qwen3.8→deepseek took 16.1 s instead of 8.15 s and
deepseek→qwen3.8 took 19.7 s, against the 20 s bound. It was measured on
spark-b during the weight-handle review.

## Remaining slices (blueprint, 2026-10-08)

The slice plan as designed before work began. Where it differs from what was
built, docs/engine-components.md's "as built" text and the commits win.
Done so far: waves 0–2, P1, G1a–d, G2a–b, G3a, N1; G2c, G3b, G4a in flight.

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
- **c, d.** Wiring and gate, then deletion. G2c runs the assistant transaction controls.

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
| N3 | Fold catalog (`norm_plus_one`, `a_neg_exp`, `value_head_tiling`, `split_fused_qk`, ties, expert slicing, NVFP4 scale-atom relayout, …) + mapping application, reusing wave 3's tables | |
| N4 | Writer (deterministic, one rename), `llmp import` CLI, admission-gate harness | Public CLI → decision |
| N5 | Re-import every artifact; lossless-equivalence checks against v0 bytes; family gates; the owner-chosen MLX checkpoint | |
| N6 | Adversarial challenge rounds | |

#### Waves 5 to 7

- **Q1–Q3, Qwen-Image.** Components (adaLN, gated residual, conv2d, channel norm, upsample, flow-match step, 3-axis rope, bidirectional attention) with a rounding-policy parameter; `kernels/image/lower.*` as per-op implementations; `EngineImage`; delete `qwen_image_runner.*` and `pipeline.cc` roles.
- **T-slices, structural techniques.** One per row of `engine-components.md`'s retired table: format layer T04/T11/T58/T95; wave composer T15/T16/T20/T29/T59/T64/T88; patterns T10/T31/T56/T86/T92 (+T09/T41/T57 adoption); prefill T23/T26/T55/T93; state T67; decoding T39/T45/T81; components T22/T69/T54/T71/T72/T61; planner T96; pager T68. Each is adopted where its matched comparison gains.
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
         └─ I5 ────────┴─ I6 ─ G1a..d ─ G2 ─ G3 ─ G4 ─ N1..N6 ─ Q ─ T, K1, R1
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
- KV encodings item: GGML quantized fattn-vec instances (D 64/128/256) must be compiled and declared; quantizing cache store (llmp set_rows/rope-store/grouped stores are F16/F32 only); kept layout ID must carry state encodings; per-alias [models.NAME] key per state kind via D-103 layers (decision entry).
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
- Importer/Qwen3.8 mapping: v0 reordered linear-attention heads (rows of in_proj_qkv/z/b/a, columns of out_proj) without recording it (manifest lists only expert-slice) → the Qwen3.8 mapping needs a permutation fold. spec::kMaxSourceTensors (2^18) < Qwen3.8 ModelOpt's 301,730 tensors and kMaxSourceValues 4,096 may be < flattened config → raise caps or stack per-expert tensors. Readers follow links outside the checkpoint dir (HF cache) by design — revisit at N6 challenge.
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
- G2b/G2c (from G2a): EP partition: kView of an expert-local value doesn't propagate (stacked gate_up); lowering of kRoute/kExpertReduce/kMatMulId to the legacy GGML sequences, kFeatures/kOutputCells inputs, frozen masks (host and device); engine: anchored draft passes (past = target committed positions), feature binding, CheckShape read_cells ≥ past+rows vs frozen reads at pad256(prefix), tap ring size vs verify rows; assistant's shared_kv_layers not required == block_count; mapping hardcodes stacked gate_up and kv_tie by layer type.
- From G2a review: CheckReads accepts only post-final-norm taps and only stage 0 as target — widen for MTP drafters reading pre-final-norm hidden state (Qwen3.8, DeepSeek) and drafter cascades (i reads i−1) before those migrations; CheckStateWrites checks only primitive kCacheStore (fine before patterns; extend when fused stores appear in drafter steps); attention.borrowed splits by heads unchecked against target's split; ffn.moe declares pattern gemma_route_reduce not yet a fused kind. 26B gap explained: production arithmetic error + frontier-row cancellation (prompt), routing near-ties ~half of decode gap.
- G2c: Gemma4 runner turns narrow_final off when retaining features (keeps every row's feature); components' tap keeps only output rows so components still narrow — decide whether the assistant needs every row's feature; if so NarrowPoint must stay off when a stage has taps.
- From G1c review: V0Values uses the compiled 2B/4B-QAT profile for any artifact of that architecture (M3 classes accepted only the approved artifact) — approval list as data comes with the importer's admission gate; engine path with speculation=true and no drafter silently serves plain (M3 refused) — decide at G2c/G3c; plan.md 2026-10-09 note should mention the gate-only checkpoint switch when G1 lands. AFTER G1: shared improvement — move turn-checkpoint capture off the first-token path (overlap with decode), for every family.
- G2c (from G2b handback): WriteComputed must stage a store-cells slot with outputs!=0 as only the last `outputs` rows' cells (taps with outputs<rows refused today); frozen host masks need a frozen CellKey (lowering refuses unless LowerOptions::frozen_host_masks); serving must pick the policy rule (gemma*_production off → outside recipe even within envelope); cuBLAS for >16-row router product (ComponentLlmBuild lends none); expert slabs reserved like Gemma4Runner::ReserveWeights + GgmlBackendOptions.lower.expert_stride; multi-stage StepSource in the engine (test-local Stages takes KeyBase from stage 0); draft read widths at Pad256 prefixes (test uses prefix 37); TensorUse::routed_experts left false (slabs cataloged as weights).
- From G2b review: launch-level CPU compare can't see read/store order inside owner-attention steps (multiset) nor addresses/placement — only the GPU bit-identity test covers them. EP with one expert per device (E == devices) refused at device-graph verification (partition.cc:185 Local keeps rank 3 with ne[2]=1) — fix before a 128-expert × 128-device placement; EP mask-by-zero turns a non-finite other-device product into NaN (pre-existing; use select/where instead of mul if it matters).
- From G1c gate debug: building a plan for a new shape costs 5.6 ms on the engine vs 3.3 ms on M3 (BuildStep 1.8 + Plan 2.2 + lower 1.3 + bind 0.9); still on the critical path for the first decode step and the 3-row tail after the checkpoint boundary (NextPrefillHint stops at stable_boundary_). Follow-ups: planner/BuildStep speed; decode-first lookahead. Gate passes without them.
- From N1 review: CheckImporter duplicated (artifact.cc/composition.cc); runtime doesn't read v1 yet (ComponentLlmBuild uses SynthesizeV0 — switch at N4/integration); N4 writer must reproduce test writer rules, options_sha256 canonical doc, tensor name mapping, drafter-pair compositions; no independent v1 oracle until M5's verifier; ASan run of the mutation test owed at a workstation tier.
- From G1d review: engine_llm.cc ~682 RunPrefill future join check uses only CompatiblePrefill; Llm::PromptSession::CanJoin also needs equal want_head (wrong prediction = wasted ahead build only). Gemma 2/3 numerical oracle after deletion is ComponentsModelsGpu.StepsMatchTheServedModel (CPU ref vs served, 6-token prompt + 3 decodes) — consider a longer independent reference. Recovery ladder and mid-prefill cancellation tested on the fake backend only for Gemma 2/3. OWNER QUESTION pending: do src/components carry llama.cpp's MIT notice (docs/licensing.md).
- spark-b lacks the Gemma 2/3 v0 artifacts in ~/.local/share/llmp/m3-artifacts (eb18d30d…, 8c710341…), so GemmaEngineServingGpu.FailedPartialLoad… skips there; copy them over so both hosts run the full Gemma set.
- From G3a round 2: component code nests builder calls as arguments (qsa.cc b.Add(b.Reshape(..), b.View(..)), Attention(cache_view(K), cache_view(V))) → node order and graph digest depend on argument-evaluation order (compiler-dependent; order tests pin Clang). Sequence them into locals across components (Gemma too) before a second compiler matters. Pre-existing UBSan -fsanitize=function failure in cpu-asan preset: reference/decode.cc:257 and reference_decode_test.cc:368/421 call GGML to_float through ggml_to_float_t (GGML's own pattern) — cpu-asan preset fails 15 tests; decide: suppress for those calls or log RE.
- License (after G3a merge): tag gdn.cc, qsa.cc, hc.cc, ple.cc MIT AND Apache-2.0 + "2023-2026 The ggml authors"; add them to docs/licensing.md and the GGML lock scope. Sweep remaining untagged src/kernels/ggml files (validate.cc, validate_ext.cc restate GGML asserts; gemma_moe_fusion.cc judged independent). ExLlamaV3 notice guard: decode_exl3.cc in llmp_reference_kernels (tests/fuzz only today) — add IN_TREE_UNITS/provenance entry so a shipped link carries Turboderp's notice. YaRN attribution comment in exec.cc/kv_quant_rope.cu (package notice already ships it).
- DeepSeek budget cliff (fd review, spark-b): with ~1.1 GiB less MemAvailable at start (119.0 vs 120.4 GB), qwen3.8→deepseek took 16.1 s (vs 8.15) and deepseek→qwen3.8 19.7 s (evict 9.2 s) — near the 20 s bound; investigate. Optional: re-hash shard header region on reopen (same-path same-size replacement now read silently, D-056 amendment).
- From G2c: G2d deletes M3 Gemma class, gemma4_runner/plan/assistant/assistant plan/greedy, gemma4 + assistant graph builders, model/gemma4 builders (keep profile/binding), ServingOptions gemma diagnostics and gemma26/31_production (move 256/1024 prefill-chunk fallback into data), Gemma benchmarks, legacy tests incl. gemma_joined_serving_gpu_test, Make routing; CHANGELOG lines. Perf leads: decode +0.2–0.4% (plan build ~11 vs 8.8 ms/shape; joined steps admit each owner twice). Assistant: depth fixed at 3 (no adaptive, draft_rows unused for gemma4, no device verdict); 26B assistant served speed has no baseline.
