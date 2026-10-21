<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3.6 record — Engine components

**Frozen 2026-10-17: M3.6 is complete, with accepted exceptions (listed below).** M3.6 began on the owner's
direction of 2026-10-08 (D-107 to D-110), with M3 complete and M3.5 parked.
It replaced the per-family runners, plans, graph builders, bindings, serving
classes and Python importers with one engine of shared components over
llmpalooza's own graph IR, a format layer that keeps weights compressed, one
runner and one serving adapter per pipeline kind, and a native importer that
publishes format-v1 artifacts. Every model that M3 served now runs on it
within the matched 1% speed gate of its frozen baseline (final results
below). This record retains the milestone's task history, its owner
decisions, the accepted exceptions and the work handed on.

The living [plan](plan.md) owns later work, [model support](model-support.md)
claims only its qualified profiles, and [decisions](decisions.md) owns the
contracts (D-107 to D-118). The design as built is
[engine-components.md](engine-components.md) and [engine.md](engine.md).
Experiment reports retain their measured binaries, source pins, profiles and
limitations; a later documentation commit does not relabel those runs as
executions of the final Git revision. Raw samples and logs stay outside the
repository (AGENTS.md, rule 8).

## Final gate and evidence mapping

| Exit criterion | Evidence at exit |
| --- | --- |
| **One engine** | Every model M3 served, and the Qwen2 fixtures, runs from a stored spec on the shared components: Gemma 2, Gemma 3, Gemma 4 26B-A4B and 31B with their assistants, Qwen3.8 native (NVFP4/MXFP8 with MTP) and GGUF, DeepSeek V4 Flash with DSpark, Qwen-Image-2.1, and the Qwen2 FP16 and EXL3 fixtures. The legacy runners, plans, graph builders, bindings and serving classes were deleted in G1d, G2d, G3d, G4d, G5d and the image slice (`22741330`); `src/model/` holds no LLM family code (D-118); serving dispatches by the stored spec's pipeline kind, and what an architecture serves with is its mapping's `ServingRecipe` and `DraftRecipe`, not a branch on its name. The image pipeline's run-time fusion is deleted: every fusion is a planner pattern (`db78764b`). Every catalog component has a CPU reference and fake-backend tests compare its fragment with it ([engine-components.md](engine-components.md#components)). |
| **Speed** | The fixed 1% protocol and frozen `1facc957-fixed` baseline ([m36-baselines](experiments/m36-baselines/README.md)); per-family results in [the table below](#final-speed-verification). Each family also passed its own slice gate on the binary of its migration, recorded there. The DeepSeek DSpark waves' C4 wall did not pass and is an accepted exception (below). |
| **Quality and state** | Quality is judged against a reference of the same quantized checkpoint (owner, 2026-10-12): the Gemma families equal the baseline's scores, the Qwen native family is paired with Mia's vLLM losses, Qwen GGUF and DeepSeek against the old arm's per-target losses, all under bounds fixed before the run ([m36-final](experiments/m36-final/README.md#quality)). On `53c2610b` the Gemma 2, Gemma 4 26B-A4B and 31B and DeepSeek scores, NLLs and reply digests equal the baseline's (DeepSeek's 966 losses are identical to the old arm), and the assistant configurations' scores equal the plain arm's. Gemma 3, Qwen3.8 GGUF and Qwen3.8 native exceed pre-registered per-target tripwires and are recorded as an accepted exception (see below); the aggregate bounds hold except Qwen GGUF's +0.58% against the old arm (bound 0.5%), judged against llama.cpp. Exact same-implementation state, continuation, spill, restore and kept-state controls pass: v0 and v1 artifacts execute bit for bit alike and the facts are frozen as digests ([v1-equivalence](experiments/m36-numerical-audit/v1-equivalence.md)); the kept-across-restart controls for DeepSeek and Qwen GGUF (`ba2d4b19`). The M3 swap table (32 rows, worst LLM-to-LLM 8.665 s) and the Gemma near-8K switch controls (24 rows) pass on the final candidate. Peak memory is reported beside each family below (owner, 2026-10-13: more memory than the reference is permitted when speed or quality warrants it and the workload safely fits). |
| **Formats** | The `sm_86` proof ([decode-templates](experiments/m36-decode-templates/README.md), `f8cd7a96`): every product shape of Qwen3.8's native NVFP4 and MXFP8 tensors (and its BF16 products) runs through the on-chip decoders on the workstation's RTX 3080 Ti within tolerance of the CPU references, 4 of 4 cases, largest error 0.263 of its bound, no product allocated device memory (2026-10-08). Rerun against the v1 ModelOpt artifact `d145900e…` at `343d375e` (`655013ec`): PASS 4 of 4, none skipped, largest error 0.263 of its bound (`hc_attn_up`), NVFP4 `scale_atoms` experts at most 0.116, n-gram rows bit-exact. No resident weight is larger than its source encoding: bulk weights never widen, and the importer's small per-channel folds (Qwen3.8: about 2 MB of 97 GB) are recorded in the artifact and chosen by tensor role (D-108 as amended, 2026-10-15). DeepSeek V4's mixed-type GGUF runs its fused patterns across differently encoded operands (G4b2 and G4 fast forms, `a0e874cf`, `10416743`), checked against the CPU references. The EXL3 trellis runs through the same layer (F5, `64c59823`; G5). |
| **State encodings** | `kv_cache_encoding = "q8_0"` is an opt-in per-alias mode on every migrated LLM (Gemma 2, 3, 4 with assistants, Qwen3.8 native and GGUF with MTP, DeepSeek with DSpark), off by default (D-117). Long-context quality against F16 (6K to 16K scored tokens, one passage per family): whole-passage perplexity −0.55% to +0.52%, no depth bucket worse than 1.7%, drafters accept on the quantized cache ([q8-state.md, K1](experiments/m36-numerical-audit/q8-state.md#long-context-q8_0-vs-f16-quality-across-migrated-llms-k1)). Qwen3.8 and DeepSeek were scored far short of their 262K contexts; see the accepted exceptions. |
| **Import** | `llmp import` publishes a v1 artifact from the verified source of every migrated model: Gemma 2, 3, 4 and both assistants, Qwen3.8 GGUF, Qwen3.8 native ModelOpt with its MTP drafter (`--component mtp [--draft-head IDS]`), DeepSeek V4 and DSpark, the Qwen2 FP16 and EXL3 fixtures, and the Qwen-Image diffusers pipeline as a v1 composition (stored family spec, `2170db12…`). Each is admitted by an approval row (`runtime/approvals.cc`, D-118) and passes the gates above; re-imports reproduce the published IDs ([v1-equivalence](experiments/m36-numerical-audit/v1-equivalence.md), [model-support](model-support.md)). The MLX checkpoint `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa80` imports against its real 3,747 tensors (`7c9279c4`) and passes the admission gate with a pinned approval row; its serving qualification is handed to M3.5. The importer and the format passed the adversarial challenge, slice N6 (`adb8eb13`, `tests/unit/import_adversarial_test.cc`). `benchmarks/import_fuzz.cc` is a fuzz entry for the container readers (GGUF, safetensors, MLX, EXL3, diffusers), `SourceOf`/`ValuesOf` and the reference decoder only: no corpus is kept and no fuzzing run is recorded, and it stops before `Synthesize`, the normalizers, `archive_input` and the v1 reader, so those are covered by the adversarial unit tests alone. |
| **Partitioning** | `planner/partition.h` (`216b7481`): tensor, pipeline and expert parallelism of any split count and mixed layouts, chosen by modelled traffic for a described topology; TP, PP, EP, TP2×PP2 and six-device mixes equal the unpartitioned graphs on the reference executor over four or more fake devices ([as built](engine-components.md#partitioning)). Two-Spark execution is M4's. |
| **Check set** | On `53c2610b`, all green: the `spark-native --locked` build; the shared set 2,839 of 2,839; the CPU-only guardrail 2,273 of 2,273; the exclusive `models\|timing` set, 214 tests in 8 area groups, all passed. Skipped, each read with `-V`: 19 Qwen2/EXL3/FP16 fixture tests (sealed corpus, trajectory and held-out oracles not wired to the run), five importer tests on real checkpoints, two Qwen3.8 quantized-head tests, `ImageGenerationGpu.SameSourceMigrationScreen`. Tiers not run: `check`, `check:full`, `check:spark` and the package fixture ([m36-final](experiments/m36-final/README.md#check-set-final-tree-53c2610b)). |

## Final speed verification

The pre-registration and its amendment are
[m36-final](experiments/m36-final/README.md): tier 1 is a 2-block ABBA
confirmation of every family (a family passes when every gated metric's mean
slowdown is at most +0.5%); tier 2, the registered full gate, runs for a
family with a metric above that and for every family not validly gated on the
migrated engine. Each family runs wholly on one Spark against that host's
frozen `1facc957-fixed` install. A negative number is faster than the
baseline. The slice gate column is the family's own gate on the binary of its
migration (earlier trees, kept as history).

| Family (host) | Slice gate (binary of its migration) | Final verification on `53c2610b` (tier, blocks, verdict; worst metric: mean, upper bound) | Peak memory, new minus old | Quality control |
| --- | --- | --- | --- | --- |
| Gemma 2 2B (spark) | PASS at 5 blocks (G1c; worst upper bound +0.90%) | tier 2, 7 blocks, **PASS**: C1 prefill +0.09% (+0.73%), C2 prefill +0.10% (+0.57%), decode −0.55%/−0.66%, wall −0.30%/−0.20% (CONTINUE at 5 blocks) | +0.034 GiB | NLL 3.5635004927, scores `cf15e993`: equal to the baseline |
| Gemma 3 4B QAT (spark) | PASS at 7 blocks (G1c; worst upper bound +0.73%) | tier 1, 2 blocks, PASS: C1 prefill −0.21%, decode −1.05%; C2 prefill −0.47%, decode −2.72% | +0.054 GiB | NLL 3.6681064003 against 3.6671293402: tripwires exceeded; **accepted exception** (below) |
| Gemma 4 26B-A4B (spark) | PASS at 9 blocks (G2c) | tier 2, 9 blocks, **PASS**: C4 decode +0.50% (+0.70%), C4 wall +0.44% (+0.61%), C1 decode +0.20% (+0.42%), prefill +0.08%/+0.20% | −0.227 GiB | NLL 9.7916674254, scores `3d69b78c`: equal to the baseline |
| Gemma 4 31B (spark) | PASS at 7 blocks (G2c) | tier 1, 2 blocks, PASS: C1 prefill +0.19%, wall +0.07%; C4 decode +0.10%, wall +0.07% | −0.008 GiB | NLL 9.5749963928, scores `12c22f37`: equal to the baseline |
| Gemma 4 31B assistant (spark) | screen 4.41% faster, transactions byte-exact (G2c) | `assistant_gate.py`, 8 blocks, **PASS**: −3.85%, bound [−4.34%, −3.37%] (every block −3.0% to −4.9%) | peak 28.468 GiB served | scores and replies equal the plain arm's |
| Gemma 4 26B-A4B assistant (spark) | served and controlled, no baseline to gate against | not gated | peak 21.522 GiB served | scores equal the plain arm's; replies differ (a verify's four-row arithmetic), as G2c recorded |
| Qwen3.8 native + MTP (spark-b) | PASS at 6 blocks on the v1 pair, every cell faster ([qwen-modelopt-v1](experiments/m36-numerical-audit/qwen-modelopt-v1.md)) | tier 1, 2 blocks, PASS: every metric faster (C1 prefill −2.11%, C2 decode −5.74%, C4 prefill −1.74%, C4 wall −3.76%) | +7.932 GiB (82.634 to 90.566; MTP activations, below) | against Mia: aggregate PPL +0.92% (ok), one document +6.91%, per-target tripwires exceeded; the old arm fails the same bounds: **accepted exception** |
| Qwen3.8 GGUF (spark-b) | PASS (G3c; raw Spark outputs only, re-gated here) | tier 1, 2 blocks, PASS: every metric faster (C1 prefill −0.78%, C2 decode −13.03%, C4 prefill −0.88%) | +7.684 GiB (58.851 to 66.535) | NLL 2.8596571307 against 2.8538864116: PPL +0.58% against the old arm, but +0.02% against llama.cpp b11254 (old −0.55%): **accepted exception** |
| DeepSeek V4 Flash, plain waves (spark-b) | 2-block `deepseek-smoke11` incomplete only by `min_blocks`, every rate at or above the old arm | tier 1, 2 blocks, PASS: C4 prefill +0.29%, C2 prefill +0.31%, decodes −1.06% to −2.69% | +0.282 GiB | NLL 2.7939955482, scores `6adb9558`: all 966 losses identical to the old arm |
| DeepSeek V4 Flash, DSpark waves (spark-b) | never validly gated on the migrated engine | tier 1 then 2, 6 blocks (the registered maximum), **NOT PASSED**: C4 wall +0.94%, upper +1.20% (limit +0.995%), **accepted exception** (below); C4 prefill +0.55% (+0.71%) and every other metric PASS | +0.106 GiB | as the plain waves |
| Qwen-Image-2.1 (spark) | PASS −0.36% (upper −0.08%, 6 blocks, `f05323d1`); PASS −0.84% (upper −0.28%) on the stored-spec composition ([m36-baselines](experiments/m36-baselines/README.md#qwen-image-21-on-the-shared-engine)) | tier 2, 8 blocks, **PASS**: `generation_s` +0.13%, upper +0.73% (old 34.373 s, new 34.418 s; CONTINUE at 6 blocks, +0.25%, upper +1.08%), pin `93356574…` | +0.816 GiB | PSNR 41.26 dB, luma SSIM 0.99419 against diffusers (floors 32 dB, 0.98) |

The Qwen2 fixtures are not served and have no speed baseline (the protocol's
exception, BP-F3; their gate is the FP16 bridge and EXL3 Tier C correctness
controls). The first image gate, `image-check02`, was invalid (its new arm set
a switch no source reads, so both arms ran the legacy pipeline); the
closeout audit withdrew it and the valid screens and attribution are in
m36-baselines.

Swap and switch controls on `53c2610b` (`spark`, `swap_table.py`, v1 IDs):
the M3 production table passes all 32 rows (every row exact and state-exact,
every restored 8K hash and continuation equal, prepared returns kept and
replayed their graphs, every image regeneration equal to the pin). Worst
LLM-to-LLM row 8.665 s (qwen3.8 to deepseek, first use, 8K) against 8.494 s on
the fixed baseline and 9.853 s in M3; worst prepared LLM-to-LLM row 8.505 s
against the 20 s bound; largest tracked peak 111.73 GiB; lowest MemAvailable
4.69 GiB. The slowest row is the image to deepseek first use, 27.354 s
(the workspace pages for the first load beside DeepSeek; first-use rows are
bounded at 40 s), its prepared return 8.804 s. The four Gemma near-8K cases
(gemma2, gemma3, gemma26, gemma31 against Qwen3.8 plain, 24 rows) all pass,
first use at most 6.421 s and prepared at most 5.961 s (the baseline's
5.99 to 6.09 s). Full rows in
[m36-final](experiments/m36-final/README.md#swap-tables-spark-swap_tablepy-v1-ids-image-pin-93356574).

## Owner decisions

Made during M3.6 and its closeout; each is recorded where it governs
(AGENTS.md, decisions.md, workflow.md).

- **2026-10-08.** M3.6 inserted ahead of M3.5 (D-107 to D-110): shared
  components over our own IR, a format layer, a native importer; "N
  drafters" means both a cascade and a per-unit choice among alternatives;
  each reviewed sub-task lands as one squashed commit; the per-family
  `ServingOptions` flags are diagnostics, removable with a CHANGELOG line.
- **2026-10-09.** The importer's admission gate uses the MLX checkpoint
  `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa80`. The importer's trust
  walk is removed: plain opens, links followed, no ownership, permission, FUSE
  or inode checks (content hardening stays; AGENTS.md rule 11, "permissive
  import").
- **2026-10-12.** Quality is judged against a reference implementation of the
  same quantized checkpoint, with frozen provenance and bounds; original-model
  evidence is required only for a closer-to-original claim; bit-identity with
  old code is not required (engine-components.md, "Migration and its gate").
- **2026-10-13.** A worthwhile speed or quality benefit may use more memory
  than the reference when the complete workload safely fits; workspace sizes
  are implementation choices, not product ceilings; image dimensions and
  generation counts follow the request and the hardware.
- **2026-10-15, closeout audit (`505df41a`).** The 2026-10-15 handoff
  overstated completion: the image gate was invalid, the Qwen gate passes
  existed only as raw outputs, and K1's Gemma 4 rows were meaningless (RE-056).
  The owner had the closeout re-gate every family on the final binary,
  coordinated by Claude with cheaper models for mechanical work.
- **2026-10-15, serving is v1-only (D-118).** Format-v1 artifacts only; a v0
  artifact is re-imported from its archived source, never migrated; the
  frozen baseline binary keeps v0 for the gate's old arm. Approval is data
  (`runtime/approvals.cc`, a lock file pinning each source and the spec digest
  it executes), and serving choices are the mapping's `ServingRecipe` and
  `DraftRecipe`: slice 4a (`47cde2ed`) made them data while v0 still served,
  slice 4b (`d918ada4`) deleted v0 synthesis, the binders and the per-family
  profiles. Qwen3.8's v0 GGML-experts artifact has no v1 import and is no
  longer served.
- **2026-10-15, Qwen3.8 draft head.** The MTP drafter's selected draft head
  is an optional `llmp import` input (`--draft-head IDS`), pinned by approval
  rows for Mia's list; a drafter with another list, or none, has no approval.
- **2026-10-15, image constants are data.** No Qwen-Image number is compiled
  in: the image mapping synthesizes a family spec stored in the v1
  composition (`103686f5`) and an approval pins it; a stored per-geometry
  spec waits for M3.5.
- **2026-10-15, growth rule (`c2bafcf8`, D-108 amended).** Bulk weights
  (products, experts, embeddings) never widen beyond their source encoding;
  small per-channel parameters (norm weights, gates, decay constants) may be
  folded wider at import when that removes run-time work and the model still
  fits with reasonable context, recorded in the artifact and chosen by
  tensor role, never by architecture.
- **2026-10-15, platform-specific work happens at import (`d6d5e023`).**
  Lossless relayouts a fast kernel needs (for example block-scale atom order
  for stacked NVFP4 experts) are done once by the importer, chosen by role
  and encoding, recorded as layouts (D-108, D-113) and decodable by the CPU
  references; never at load time.
- **2026-10-15, no compatibility owed (`dcd606d8`).** No bit or format
  compatibility with v0 artifacts or retired runners: early development, no
  users. Acceptance is the speed gate and reference quality; the v0≡v1
  proofs are diagnostics and frozen digests regression detectors that may be
  re-recorded when a change is legitimate.
- **2026-10-15, housekeeping.** Source checkpoints already in the NAS archive
  and diagnostics of completed work may be deleted from the Sparks. Nemotron
  3.5 Super VL listed as an owner-requested M3.5 family (`71b34fab`, its
  release day). The importer's family-specific native readers (Qwen3.8's
  ModelOpt and MLX normalizers and the CLI's dispatch to them) remain as M3.5
  debt: generalize them into mapping-driven normalizers when Nemotron NVFP4
  brings a second ModelOpt model (the owner's reviewer).
- **2026-10-16, physical-memory modes (`0fb9bb7c`).** Dedicated (a handle
  pool sized to the node budget at start) against shared-machine (brought up
  on demand, backing released when idle) proposed as a per-node mode, on the
  owner's direction; amended by `be5c4db6` (features.md).
- **2026-10-17, DSpark C4 wall accepted as an exception.** The owner
  accepted the NOT PASSED DeepSeek DSpark C4 wall (+0.94%, upper +1.20%) with
  its root cause (physical placement of the activations, RE-059) and made the
  dedicated physical-memory pool M3.5's first work item (plan.md, features.md).
- **2026-10-17, quality exceptions accepted.** The Gemma 3, Qwen3.8 GGUF and
  Qwen3.8 native pairings that exceed their per-target tripwires are accepted
  as legitimate numerics or pre-existing differences, with the bounds
  unchanged (the coordinator's acceptance; see the accepted exceptions).
- **2026-10-16, tiered final gates (`ef99bc52`).** Cut the final pass from
  about 9.5 GPU-hours on one host to a tiered pass on both Sparks: every
  family a 2-block confirmation, the full gate only where a metric is above
  +0.5% or the family was never validly gated. Registered before any gate run
  ([m36-final](experiments/m36-final/README.md#amendment--tiered-gates-on-both-sparks-owner-2026-10-16)).

## Task history by slice

Slice IDs are the blueprint's (F format layer, L lifecycle and runner, I IR
and planner, P partitioning, G families, N importer, Q Qwen-Image, K1 state
encodings, R1 re-keying). One commit lands each reviewed sub-task
(`[M3.6] …`); the commit titles are the full account, the reports the
evidence. The nested scope text, verbatim, is
[at the end of this record](#retained-scope-history).

### Foundation

| Slice | Outcome | Commits and reports |
| --- | --- | --- |
| Design, decisions | [engine-components.md](engine-components.md), D-107 to D-110; M3.5's transfer item retired with its mapping | `ff6315ae` |
| Baselines and the 1% protocol (S0) | Matched, bookended protocol; frozen `1facc957` baseline; DSpark four-slot waves and the swap table's Qwen continuation fixed (the `-fixed` baseline); shared and exclusive `spark-job` classes | `2bc946b1`, `14b6d610`, `f4c65767`; [m36-baselines](experiments/m36-baselines/README.md) |
| Engine core, fake device runtime (F1) | `llmp_engine_core` builds in every profile with a fake device runtime and host paging kernels (D-096 amended) | `7cd41eba` |
| Format layer (F2 to F5) | Encodings, layouts and the size rule; CPU reference decoders; the fallback ladder; CUDA block decoders under vector, matrix, grouped and GLU templates; the `sm_86` proof; the EXL3 trellis; state encodings | `f5eed53e`, `f8cd7a96`, `64c59823`, `555563c4` (affine); [decode-templates](experiments/m36-decode-templates/README.md) |
| Graph IR | Op set v1, verifier, reference executor | `cab294de` |
| Slot lifecycle (L1 to L3) | State contracts and the shared slot lifecycle with reconciled guards; owed-verify incremental spill fixed | `e87d369a`, `7a852282` |
| Spec, mappings (I2, I3) | Model spec, mappings as data, v0 synthesis | `5b7e7b8d` |
| Quantized KV attention | GGML vector attention over Q8_0 and Q4_0 K and V at D 64 and 256, quantizing stores | `d3279fcd` |
| Runner, decoding modes (L5) | Step programs, wave composer, prefill, decoding modes with any number of drafters | `16181eec`, `0ccab851` |
| Components, planner, lowering (I4 to I6) | Component framework and Gemma 2/3 components; planner core; lowering onto GGML with structural parity | `fc2dc30d`, `453f0718`, `6d2cee22` |
| Partitioning (P1) | TP, PP, EP, mixed layouts, chosen by modelled traffic | `216b7481` |
| Serving adapter (L6) | One serving adapter per pipeline kind; kept record v2 (D-112) | `3d4c1e9e` |
| Weight shards open only while active | Artifact files open only while their model is active | `961fdefe` |

### Families

| Slice | Outcome | Commits and reports |
| --- | --- | --- |
| G1 Gemma 2, Gemma 3 | Components and the mapping; serving on the shared engine; the 1% gate passed (5 and 7 blocks); runners deleted (G1d) | `74136a5d`, `4a6710ce`; [m36-baselines, G1c](experiments/m36-baselines/README.md#gemma-2-and-gemma-3-on-the-shared-engine-slice-g1c) |
| G2 Gemma 4 26B-A4B, 31B, assistants | Components (MoE, borrowed attention, projected embedding), patterns and lowering bit for bit, serving with anchored assistants; gate passed (9 and 7 blocks); runners deleted (G2d) | `a2553b9e`, `6ff8a262`, `aec496b0`, `04339dec`; [G2c](experiments/m36-baselines/README.md#gemma-4-26b-a4b-and-31b-on-the-shared-engine-slice-g2c) |
| G3 Qwen3.8 native, GGUF, MTP | Components (HC, GDN, QSA, n-gram, PLE, MTP), fast-form lowering, verify and prefix commits, joined MTP arithmetic, sole-owner and joined serving, decode recovery (multi-slot draft depth, MoE scale memsets, fused GDN commit history: every cohort faster than the legacy runner), runner deleted (G3d) | `88efa86c`, `3c74d425`, `72abcee8`, `9e2453c4`, `e2fd5bab`, `25711ad9`, `7ad389b5`, `d69c8036`; [qwen-joined-mtp](experiments/m36-numerical-audit/qwen-joined-mtp.md), [qwen-products](experiments/m36-numerical-audit/qwen-products.md), [qwen-quantized-reference](experiments/m36-numerical-audit/qwen-quantized-reference.md) |
| G4 DeepSeek V4, DSpark | Components, compressed-attention shape safety, guarded fast forms, state banks, exact selection, rounded score and native index score, owned execution identity, DeepSeek serving on the shared engine, the closeout's thinking-default fix, legacy runner, plans and graph builder deleted (G4d) | `0ad1880a`, `333b35a0`, `9b72a090`, `56ca74fc`, `156c7be4`, `a0e874cf`, `10416743`, `495a1cdb`, `88eb0da4`, `2decc784`; [m36-dsv4-target](experiments/m36-dsv4-target/README.md), [m36-dsv4-rounded-consumer](experiments/m36-dsv4-rounded-consumer/README.md) |
| G5 Qwen2 FP16, EXL3 fixtures | Shared components, GGML and EXL3 lowerings, original-model quality observations, same-implementation state, spill, kept-state and weight-reload controls; legacy executors deleted (G5d) | `ab21dd76`, `04470706`, `0d127ac0`; [m36-qwen2](experiments/m36-qwen2/README.md), [m36-exl3-selected](experiments/m36-exl3-selected/README.md), [m36-original-qwen2](experiments/m36-original-qwen2/README.md) |

### Native importer

| Slice | Outcome | Commits and reports |
| --- | --- | --- |
| N1 format v1 | The v1 artifact format and reader: tensor formats and parts, the spec as a hashed document, applied folds (D-113) | `5bce159d` |
| N2 readers | GGUF, safetensors with `config.json`, diffusers, MLX, EXL3 | `4078761b`, `046f29bc` |
| N3 normalization | Bounded transform primitives; owned GGUF Gemma, Qwen3.8 and DeepSeek normalizers; BF16 operands and norm offsets; head permutations; indexed multipart assembly; the closed ModelOpt Qwen3.8 normalizer; MLX Qwen3.8 | `b9b4fc73`, `f7401054`, `a30bb57f`, `a33a4f82`, `06290abf`, `7c9279c4` |
| N4 writer and CLI | Deterministic v1 publication, `llmp import` from the archive or a staged copy with authenticated bounded streaming and catalog events (D-114, D-116); native Gemma, Qwen GGUF, DeepSeek GGUF, ModelOpt, Qwen2 and drafter publication and admission | `49641b52`, `d9681846`, `d5323ce7`, `926a4f40`, `359887e6`, `b8ed7a44`, `67eed4b6`, `13c42cf0`, `3e2f3b5a`; [m36-native-v1-admission](experiments/m36-native-v1-admission/), [m36-modelopt-normalization](experiments/m36-modelopt-normalization/), [m36-qwen-gguf-normalization](experiments/m36-qwen-gguf-normalization/README.md) |
| N5 re-import, admission | Every migrated model re-imported from its verified source; v0 and v1 execute bit for bit alike (frozen digests); Qwen3.8 ModelOpt target and MTP drafter with an expert relayout into scale atoms and role-chosen F32 folds (v1 pair served as the v0 pair, `qwen38` gate passed) | `cef6e1e4`, `5b21fe48`, `5f4bb2ce`, `f09e7994`; [v1-equivalence](experiments/m36-numerical-audit/v1-equivalence.md), [qwen-modelopt-v1](experiments/m36-numerical-audit/qwen-modelopt-v1.md) |
| N6 adversarial challenge | Untrusted-input suite over the readers and the writer (841-line `import_adversarial_test.cc`) and the hardening it drove | `adb8eb13` |

### Qwen-Image-2.1

| Slice | Outcome | Commits and reports |
| --- | --- | --- |
| Q1 to Q3 shared blocks | Channel normalization, adaptive normalization and gated residuals, typed spatial graphs, the VAE, the timestep conditioner, the 32-block flow and Euler step, the complete shared generation core, request-driven geometry and schedules, retained resources that grow, an ordinary installed-source service | `db8fa1c3`, `0b55601d`, `28d99ebd`, `cb4326ee`, `3a90e79e`, `cb8bed69`, `a0c784b4`, `eb602bce`, `44080740`; [m36-shared-spatial-execution](experiments/m36-shared-spatial-execution/README.md) |
| Speed and fusion | Run-time fusion first, then every matcher became a planner pattern and the run-time path was deleted; invariant inputs, load-time capture, driver and per-stage host costs; the gate passed | `b37f763f`, `7f4b0547`, `db78764b`, `b29380a7`, `c8f0f80b`; [m36-runtime-fusion](experiments/m36-runtime-fusion/README.md), [m36-baselines](experiments/m36-baselines/README.md#qwen-image-21-on-the-shared-engine) |
| Legacy retirement, v1 | Legacy pipeline and runner deleted (before its first valid gate: audited in the closeout); v1 composition with the family as stored spec data and an approval | `22741330`, `86ddce28`, `103686f5` |

### State encodings, cleanup and closeout

| Slice | Outcome | Commits and reports |
| --- | --- | --- |
| K1 | Dense Q8_0 opt-in (D-117), then every migrated LLM (D512, QSA, MQA, borrowed and drafter state) with long-context quality; Gemma 4 rows rerun as the model's own turn after the first run's were withdrawn (RE-056) | `ca64d434`, `b19ddb08`, `ed0ee03b`, `82dcb0ad`, `78cd2f8a`, `4e490771`, `d36922cb`; [q8-state](experiments/m36-numerical-audit/q8-state.md) |
| R1 | Settings applicability and defaults derive from the spec's capabilities; `ModelKeySpec::architectures` and `RunnerContextCeiling` removed | `e3d7be83` |
| Serving from data (4a, 4b) | Approvals as data, serving recipes, no per-architecture selectors; v0 synthesis, binders and profiles deleted; every model serves from an ordinary configuration | `47cde2ed`, `d918ada4`, `093efe3c` |
| Numerical audit and quality policy | Retrospective audit; same-quantized-reference gates; Qwen and DeepSeek reference losses | `fb4264d1`, `7197b022`, `f87bab73`; [m36-numerical-audit](experiments/m36-numerical-audit/README.md) |
| Closeout fixes | DeepSeek's default thinking restored; the shared workspace pages with the active model (the swap table starts again from an ordinary configuration); late state regions reserved on the backing lane; small row tables resident (DSpark joined waves back to M3 speed); forwarded alignment facts sealed by graph identity (Gemma 4 plan builds about twice as fast) | `495a1cdb`, `f05323d1`, `a32ea923`, `91228703`, `343d375e` |
| Planner policy, startup payment | Row gates are policy data (step rules in a family's serving policy, `planner::ForStep`; DeepSeek's byte for byte the old gates, Qwen3.8 GGUF's 9 to 16-row solo steps keep `vecq_glu` again, a kept behaviour change: C2 decode about +14% on a one-run screen); every LLM registers under the automatic startup payment and the funded diagnostic tuple is an optional override; every model serves from an ordinary configuration | `a9eea3bb`, `093efe3c`; `m36-planner-fix`, `m36-runtime-fix` |
| State layout, resident workspace | Per-tensor state layout (each growing tensor of 2 MiB or more on its own extents, the rest packed, a compressed cache contiguous with its raw window): Gemma 2 prefill back to M3 speed after the 256-byte packing made it 1 to 2% slower; the shared workspace maps resident whenever it fits and pages only otherwise (the image's denoiser had run up to 5% slower with extent-by-extent backing); late state regions reserved on the backing lane (a VMM race fix); a dense-vector-product check against GGML at Gemma 3's Q4_0 shapes | `0c6bcc9a`, `be5c4db6`, `a32ea923`, `f6442ebe`; [m36-final, State layout](experiments/m36-final/README.md#state-layout-2026-10-16-investigation-of-the-tier-1-regressions) |
| sm_86 proof on v1 | The Formats criterion re-run on the workstation's RTX 3080 Ti against the v1 ModelOpt artifact `d145900e…`: 4 of 4 | `655013ec`; [decode-templates](experiments/m36-decode-templates/README.md) |
| Final verification | The gate tools serve the v1 image composition by default; results on candidate `53c2610b` | `53c2610b`, `9a3ebc20`; [m36-final](experiments/m36-final/README.md#final-candidate-53c2610b-tree-d373a1bf-the-verification) |
| Gate tools and pre-registration | `speed_gate.py`, `assistant_gate.py`, `quality_pair.py`, `swap_table.py`; the final verification pre-registered and amended | `bf0fb2a4`, `210d0a1a`, `ef99bc52`; [m36-final](experiments/m36-final/README.md) |

## Accepted exceptions

- **Qwen3.8 and DeepSeek K1 contexts.** Q8_0 KV quality was measured at
  9.9K to 15.8K scored tokens, not their 262K contexts: one-row teacher
  forcing costs 20 to 30 minutes an arm past 32K. A full-context claim needs a
  scoring path that takes the rows' heads from prefill chunks. Handed to M9.
- **Gemma turn checkpoints in the gate.** The frozen M3 Gemma classes never
  save a turn checkpoint past about 1K tokens (baseline defect 3), so the
  Gemma families' new arm ran the gate with
  `LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS=1` for equal work; the quality
  controls run without it. The new engine saves them.
- **No speed gate for the Gemma 4 26B-A4B assistant** (no baseline records
  one; the harnesses refuse it); it is served, controlled and its peak memory
  recorded. The Qwen2 fixtures have no speed baseline (BP-F3).
- **Image pixels.** The image gate pins each implementation's own output
  digest before timing, with reference quality (PSNR and SSIM against
  diffusers) as the separate control; bits are not compared across
  implementations.
- **Bit equality is diagnostic.** Quality is judged against the same-quantized
  reference; Qwen3.8's numerics changed with the expert relayout and planner
  and are not bit-equal to v0.
- **DeepSeek DSpark waves.** The old arm's four-slot waves were defective and
  fixed in the baseline; DeepSeek's near-tie numerics make reply texts differ
  from OLD at varying depths (OLD also differs from itself between solo and
  joined forms).
- **Qwen3.8 MTP activations.** Four slots measure 8.05 GiB of activations on
  the shared engine against 1.42 GB in the M3 runner (a planner sizing
  question); it fits and is accepted under the memory rule. R7 (`d3374c67`)
  has since cut 4.53 GiB (native) and 4.37 GiB (GGUF) of it.
- **MLX serving.** MLX is imported and admitted, not qualified for serving.
- **Qwen2 serves single-owner.** `kQwen2Policies` sets `max_decode_rows 1` and
  `owners_alone` (`src/spec/mappings/qwen2.cc`) to preserve the frozen v1
  digests of the Qwen2.5-0.5B GGUF and EXL3 fixtures, so every Qwen2
  checkpoint decodes one row at a time and does not batch. This departs from
  AGENTS.md rule 11 (every model batches; speed over bit-exactness) and was
  not a registered exception; Gemma 4 outside its envelope and the
  repeat-pending drafter mode are single-owner too. Removing the policy and
  re-pinning the digests is M3.5 work.
- **Quality tripwires exceeded, accepted (coordinator's acceptance decision,
  2026-10-17).** Three registered pairings exceed their per-target bounds on
  the final candidate; the bounds stay as pre-registered and are not
  adjusted. Evidence: [m36-final](experiments/m36-final/README.md#quality) and
  `spark-b:~/scratch/m36-quality-reg/` (outside Git; determinism checked).
  - *Gemma 3* (new against old): PPL +0.098% but mean absolute per-target
    difference 0.0869 (bound 0.05) and 70.6% of targets within 0.1 (bound
    95%). The change is `vecq.dense` in place of MMVQ from `3c74d425` (G3b).
    Against llama.cpp b11254 on the same GGUF the old arm differs by PPL
    −0.52% and the new by −0.42% (mean absolute 0.0765 and 0.0825): the same
    distance, so legitimate numerics at the same reference quality.
  - *Qwen3.8 GGUF* (new against old): PPL +0.58% (bound 0.5%), mean absolute
    0.231. Against llama.cpp b11254 the new arm is +0.02% and the old −0.55%;
    llama.cpp's own fused and unfused builds differ by −1.75%. The change
    came with the legacy-to-component move (`093efe3c`): legitimate numerics,
    closer to the reference.
  - *Qwen3.8 native against Mia's vLLM*: aggregate PPL +0.92% (bound 1.0%
    holds), one document +6.91% (bound 4.0%), mean absolute 0.524 and maximum
    6.35 (tripwires 0.25 and 5.0). The old arm fails the same bounds (PPL
    +0.73%, mean absolute 0.521, maximum 6.32; document 1 +5.01%): a
    pre-existing difference (Mia's FP8 KV and BF16 SSM against ours), not a
    regression.
  - *The per-target bounds proved uncalibrated:* llama.cpp against itself
    (fused against unfused) fails the 95%-within-0.1 agreement bound (92.9%).
    A future pre-registration should set per-target bounds from a measured
    same-reference noise floor.
- **DeepSeek DSpark waves' C4 wall (owner decision, 2026-10-17; retired in M3.5
  D-124, 2026-10-21).** NOT PASSED at +0.94% mean, upper bound +1.20% (limit
  +0.995%), 6 blocks (the registered maximum), on candidate `53c2610b`; every
  other DSpark metric passes. The owner accepted it as an exception at M3.6
  closeout, and M3.5's dedicated physical pool (D-124, `RE-059`) plus the
  `PrefillOrchestrator::Predict` multi-slot speculative continuation lookahead
  fix retired it on 2026-10-21: in a 2-block matched confirmation on `spark`
  against `61f8426f`,
  DSpark passed every metric (`C1/C2/C4 prefill_s` +0.12% / +0.97% / +0.47%,
  `C4 wall_tok_s` −1.99% faster, `C1/C2/C4 decode_tok_s` −1.20% / −10.27% /
  −17.57% faster), alongside `image` (`generation_s` −0.01%) and `gemma2`
  (`C1/C2 prefill_s` +0.32% / −1.18%).
  - *Root cause:* the physical placement of the activations. `be5c4db6` maps
    the 2.6 GiB shared workspace resident at startup, and on a long-running
    host the driver backs it with about 51,000 scattered 64 KiB pieces taken
    from the buddy allocator's smallest free blocks, so big prefill steps
    lose about 1.3% of GPU time ([RE-059](rough-edges.md#re-059-gb10-kernel-speed-depends-on-where-the-kernels-free-memory-puts-a-device-allocation--2026-10-17-status-open)).
  - *Ruled out:* the state layout (`0c6bcc9a`) and the access pattern
    (per-extent against range-wide `cuMemSetAccess`).
  - *Workaround tried, not landed:* a startup "borrow until 2 MiB blocks"
    step recovered DSpark (−0.38%), but the image's response to placement is
    not monotonic (−0.74% to +1.25%), so it is no placement policy.
  - *Fix:* the dedicated physical-memory pool, **M3.5's first work item**
    (owner), which makes placement deterministic; the re-gate of DSpark's C4
    wall, the image and Gemma 2 with it retires this exception.

## Retained limitations

- Every model draws on one shared workspace from its base (resident as one
  region when it fits beside the largest model's weights since `be5c4db6`,
  paged with the active model only when it does not), so **two models cannot
  execute concurrently on one node** (runtime-serving.md, "Registration and
  the swap"). Concurrent placements (D-019 to D-025) need a workspace per
  executing model, or disjoint ranges of one, before M4 places two models'
  work on one node.
- RE-050 (Qwen3.8's head is wider than its tokenizer; rows 248,077 to 248,319
  are not masked), RE-056 (Gemma 4 instruct checkpoints are not raw-text
  language models; K1 scores them as their own turn), RE-057 (a timing run
  begun with a build's page cache swaps during its first large load; the gate
  drops it first). RE-058 (EXL3's lock slots) is fixed in R6: each runner
  funds its selected sources' lock slots separately from the shared pool.
- **What "family-neutral" does not yet cover** (the M3.6 review of
  2026-10-20; the code reads as follows):
  - *Schedule products are DeepSeek-only.* `compact_experts`, `d2r_experts`,
    `pair_experts`, `pair_glu`, `dense_pair` and `vector_floats` are default-off
    policy options that only `kDeepSeekServingPolicy` (and DSpark's subset,
    `src/spec/mappings/dsv4.h`) enables, although `graph_plan.h` describes the
    pair and stage ones as byte-identical. Gemma 4 26B-A4B and Qwen3.8 GGUF
    have no recorded screen for them (optimization-inventory T04, T07, T12).
  - *Fast-path predicates still name checkpoint dimensions.*
    `src/kernels/ggml/capabilities.cc` admits `llmp.gdn.gates` at 48 value
    heads only (the kernel hard-codes `i % 48`), the DeepSeek Q-head, output-A,
    hyper-connection post-norm, weighted-reduce and QSA kernels at their
    DeepSeek V4 Flash dimensions, owner attention at Gemma's head counts, and
    the MXFP8 GEMV schedule at one width. A same-architecture checkpoint of
    another size runs but without these paths. D-123's "real limits" wording
    is the target, not the state.
  - *Numerical qualification is GB10-only and approval-shaped.* Every
    `NumericalEvidenceDomain` defaults to `arch = 121` and `planner/select.cc`
    requires equality, so on `sm_86` no qualified fused path applies to any
    model. A pinned approval with empty `numerical` receives no
    qualifications, while an unpinned artifact derives them from container and
    component-kind proxies (`runtime/component_admission.cc`), so pinned and
    unpinned checkpoints do not receive the same policy.
  - *HF-quantized and HF-MoE import has two normalizers.* ModelOpt
    (`modelopt_qwen38.cc`) and MLX (`mlx_qwen38.cc`) are pinned to single
    Qwen3.8 checkpoints (exact tensor counts, part counts and quantization
    strings). The table normalizer refuses any other quantization, patterns
    have no `{expert}` field, and the role-driven folds and expert relayouts
    run only from those two. NVFP4/MLX of Llama, Qwen2 or Gemma, AWQ/GPTQ,
    DeepSeek FP8 safetensors, another Qwen3.8 size in ModelOpt/MLX and every
    HF MoE checkpoint are refused: rule 11's "every quant of every supported
    model" is not met for HF-quantized sources.
  - *The image pipeline's structure is code.* The VAE decoder builder
    (`model/image_pipeline.cc`), the DiT wiring (`AppendImageBlock`) and the
    spatial component switches are C++, not stored data; another image
    architecture or a quantized Qwen-Image is new code.
  - *On-demand expert paging is not live in serving.* Serving never sets
    `ModelRunnerConfig::routed_metadata`, so D-008's acquire-the-selected-
    closure path is exercised by tests and benchmarks, not by `rt::Server`.
  - *The single-model assumption runs past the workspace.* `Server::resident_`,
    the cohort model, one swapped model and its parts, `SwapBytes` and
    `SpillView` all assume one resident model (not only the shared
    workspace above); M4's placement and M3.5's memory modes must lift them.
- The shipment tiers were not run for this milestone: `check`, `check:full`
  (sanitizers, container, offline copyleft-disabled), `check:spark`
  (cross, ASan, TSan) and the package fixture; also the K1 llama-perplexity
  pairing for Gemma and the llama.cpp b11254 and DeepSeek `collect_reference`
  reruns (owed by the registration). All are owed before a package ships
  (D-085).
- Timing is sensitive to the host's thermal state: the migrated image path
  slows more than M3's when the GPU is heat-soaked (0 to 5% across a session
  at unchanged SM clocks; 41 C idle against 63 to 72 C under the queue), so a
  gate's verdict can differ between a cold and a soaked host. The final gates
  logged the GPU temperature before every session and after every run.

## Post-gate changes

Four M3.6 review commits landed immediately after the gated candidate
`53c2610b` (`c1dcc8c1`, `fa1f7cad`, `235ff8e4`, `7283b964`), changing 95 `src`
files (`c1dcc8c1`: component mask metadata, output-A fast-path geometry checks
and format-rule inventory validation; `fa1f7cad`: partitioner `StreamBytes` and
`MakeFull` deduplication, `PrologueOf` and written-state root checks,
`QuantAttention` and `selected_grouped`/`decode::Accepts` gates, `ffn.moe`
`hidden = 0` and `"pre"` placement, data-driven `KvCacheEncodingBytes` /
`ApplyKvCacheEncoding`, `FromRatio`/`FromOrRatio` mapping rules, and deleting
orphaned `live_state_launch.{h,cc}`; `235ff8e4`: D-119 direct admission of
unpinned v1 artifacts, family-neutral `model/image_pipeline.{h,cc}` and
retiring `src/kernels/image/`, Qwen2 safetensors mapping, unified indexed-part
normalizer dispatch, atomic `RenameNoReplace` publication, `execution/step_shape.h`,
and `sigmoid_topk` routing; `7283b964`: effective draft vocabulary in settings
and factory identities, 64-bit gathered-row offsets, partitioner side-source and
shared-weight deduplication, shared MoE branch clamp, VAE bounds from levels,
and independent product policy choices). At `7283b964` the shared (`2,899/2,899`)
and `models` (`214` entries: `190` passed, `24` skipped, `0` failed) sets were
rerun on `spark-b` with frozen digests unchanged; speed was not re-gated at
`7283b964`.

The M3.5 review follow-up campaign (`R1`–`R13`, tracked in [plan.md](plan.md))
then resolved the remaining review items across `aba73a61` (R1), `40bf23e3` (R2),
`f5ebd16a` (R3, D-120), `80ede618` (R5), `fe17e4ae` (R4, D-121), `43f2f50a` (R6),
`09ffb557` (R8), `d3374c67` (R7), `e6c3617a` (R10), `0d28a553` (R9, D-122),
`b003c9a5` (R11: generic dense-decoder mapping and `kDefaultServingRecipe`; Llama 3.2 1B
Instruct and Qwen2 served through `EngineLlm`), and `aa92bd1f` (R12: `components_catalog_gpu_test`,
wired Qwen2/EXL3/FP16 controls, `record_to_json.py` re-record workflow,
configurable `LLMP_TEST_V1_STORE`, and `RESOURCE_LOCK llmp_gpu`). Two-block
ABBA tier-1 speed and teacher-corpus quality screens ran in R6 (`43f2f50a`),
R7 (`d3374c67`), R10 (`e6c3617a`), and R9 (`0d28a553`), followed by a full
closeout verification on `aa92bd1f` against `53c2610b`
([m35-r13-closeout](experiments/m35-r13-closeout/README.md)): all `3,375` shared
tests, `2,789` CPU-only guardrail tests (`LLMP_CUDA=OFF`), and `216`
`models|timing` tests pass; all 32 M3 swap-table rows (`image -> deepseek`
first use `27.354 s → 8.733 s`) and 24 Gemma near-8K switch rows pass; peak
memory drops by `−4.30 GiB` on `qwen38`, `−4.44 GiB` on `qwen38-gguf`, and
`−1.68 GiB` on `gemma31`; and every served family confirms speed against the
M3.6 final table and M3 baseline.

A subsequent whole-tree review against `origin/main` (2026-10-20, D-123) added
six capability-unification and defect-fix slices (`R14`–`R19`):
- `575b04a4` (R14): IR `Decompose` contiguity and role/view/rounding checks,
  `Graph::Digest` trellis parameters, `Partitioner` 32-bit constant sizes,
  `planner::KeyOf` full `PlanKey` initialization, `attention.dense` stateless
  gates/sinks and `scale_scores`, `StepContext` builder segment preservation,
  `kF16`/`kQ4_0` KV cache encoding rules, `CheckApplied` fold validation,
  disjoint `layer_types`, `kScaleAtoms` stacked alignment, `GrowMapped` host
  charge rollback, multi-model `TearDown` detach, per-shape fallback flags,
  per-segment routed capture, runner-scoped `PlanStep`, and gate-tool fixes.
- `c0cdf812` (R15, D-123): capability- and format-derived `AdmittedNumerics`
  and `AdmittedImageNumerics` (no `approvals.cc` SHA-256 or `1024x1024`
  geometry gates), unified `AdmittedPolicy` and `ResolveSettings` over
  `spec::Mapping` and `ServingRecipe`, multi-encoding `ReadImageAdmission`,
  deduplicated staging helpers, and multi-drafter `ComponentLlmBuild`.
- `d1356730` (R16, D-123): hardware/algorithmic GGML capability predicates
  (owner attention on any 32-lane Turing+ arch, unpinned `Bf16VectorProduct`
  and `Dsv4HcPre`/`CheckDsv4HcMix`), wildcard `DeviceRank` rules, on-chip
  decode `AffineScale::kF32` and `zero_point`, generalized mixed-encoding
  `fused.mul_mat_glu` and paged `kGetRows`, shape-generic `exl3.device.*`
  `gemv`/`bias_add`, SM-derived multi-GEMM concurrency, and device-kernel
  `WritePointerTable` replacing pageable `cudaMemcpyAsync`.
- `8936e175` (R17): `placement = "pre"` and `"sandwich"` on `attention.gdn`,
  `attention.mqa` and `attention.qsa`, conditional `gdn.value_heads_tiled`,
  generalized `attention.mqa` sparse options, `attention.dense` `target_layer`
  KV borrowing, `{kF32, kBF16, kF16}` and `kNormZeroCentered` across `ffn.moe`,
  `hc.mix` and `attention.mqa`, string-valued `layer_type_key` ($>64$ layers)
  and declarative `Mapping::companion` synthesis, explicit
  `Fragment::spatial_layout`, and arbitrary kernel/stride/padding in
  `conv.spatial`.
- `8b3eafdd` (R18, D-123): `FoldScalar` across `kBF16`/`kF16`/`kF32`,
  role-driven `RelayoutExpertTables` over `spec::ModelSpec` including
  `kMxfp8`/`kMxfp4`, unpinned semantic tensor digests, `NormalizeMlxDrafter`
  (`--component mtp` on MLX), lifted `kRuleLimit` (`256`), nested
  `rope_parameters` and `layer_types` in HF `safetensors` import, HF
  `tokenizer.json` `ByteLevel`/`Digits`/`rstrip` and SentencePiece trainer
  fields, and single-pass `kPrefixEnd`/`kMessageEnd`/`kGenerationPrompt`
  boundary tracking in `RenderInterpreted`.
- `fa1d9073` (R19): `CarryRegion` and `UndoCarry` unified with `HandleReserve`
  and `BackingCreateCounters` (`RE-059`), 16 GiB default workspace virtual
  capacity in `MapWorkspace`, skipped redundant catalog closure RPCs in
  `ContinueRouted` when all selected expert pages are held in the base lease,
  and deduplicated `FormatPlanCoverageReport` and `SetupRowTables`. All `3,410`
  shared `spark-native` tests (`432` GPU, `2,948` unit, `21` toolchain, `10`
  timing, `5` sources, `4` smoke, `2` version) pass on `spark`.
- R20: engine/runtime correctness fixes from the same review. Hang
  recovery's rung 2 (`Server::RecoverModel`) now acknowledges its stream
  fence to the runner (`PagedNode::RecoverAfterFence`,
  `PagedModel::AcknowledgeFence`), so a step or routed continuation whose
  end was unproven no longer refuses every later step and plan drop until
  node teardown; on the direct path a later fence of the stream proves a
  lost step's end (`PagedNode::FenceStream`). A routed flight counts as
  started only once work was queued. The cooperative cohort's drain and
  switch are recomputed from the queue's head each pass. Idle state is
  priced as what its take does (spill with the deleted spills'
  recomputation charged, or drop); a restored conversation's retained
  spill files count toward `spill_budget_gib` and are released first under
  it (the catalog then forgets what they saved, `Catalog::ForgetSaved`); a restore refused for capacity keeps its D-105 record; a partly
  failed spill is credited by measured catalog gain; a waiting parse whose
  client left is freed; the three activation/hang copies are one
  `NodeBackend::ActivateFor`; `PagedNode::Load` reads the catalog on the
  scheduler thread and `PagedNode::Covered` without residency reads no
  catalog state.

## Deferred items from the reviews

The reviews' follow-up queue (the main agent's notes, formerly in the
temporary handoff file, deleted at closeout), classified at closeout and updated
after the post-gate M3.6 commits (`c1dcc8c1`–`7283b964`) and the M3.5/M3.6 review
follow-up slices (`R1`–`R19`, `aba73a61`–`fa1d9073`). Remaining open items are tracked in
[plan.md](plan.md) and [rough-edges.md](rough-edges.md).

**Fixed or moot in M3.6**

- The legacy Qwen3.8 and DeepSeek runners' spill and written-back bug (a
  verify's owed restore marking a stage saved) and every legacy-runner item:
  moot (the runners are deleted, G3d and G4d); the shared slot lifecycle owns
  the rule, and CHANGELOG records the fix.
- L5c's `WritesFor` ring shapes, per-row Qwen and DeepSeek saves, drafter
  stages' read alignment; L6's kept record version bump and the kept layout
  ID taking the bound spec and prepared execution identity (`c7534e88`, D-112).
- Family migrations onto decoding modes, spec `reads` linking, joined MTP and
  DSpark chains, the G1c wiring items (cuBLAS handle, device greedy, policy
  strings and vocabulary taken from the mapping and the spec): done by G2c,
  G3c, G4c and the serving-recipe slices.
- The importer's trust walk, descriptor cache and `RaiseOpenFileLimit` in the
  importer (verified: the importer no longer calls it; the socket limit stays,
  D-102); weight files open only while their model is active (`961fdefe`).
- `spec::kMaxSourceTensors` raised to 2^20 (verified); the Qwen3.8
  `gdn.value_heads_tiled` fold applied by N3b; the MLX admission checkpoint.
- License tags of `gdn.cc`, `qsa.cc`, `hc.cc` and `ple.cc` (verified:
  `MIT AND Apache-2.0` with the ggml notice).
- Legacy-only items: `qwen38_wave_plan.cc` planning with zero activations
  (deleted with the runner, G3d); family-specific `V0Values`, approval lists
  and the "speculation with no drafter" fallback (v0 profiles deleted, D-118;
  approval is data).
- Lossy KV's D512 gap (Gemma 4 global layers, DeepSeek) and attention cell
  bytes taking `model::CellBytes`: closed by K1 (D512 Q8_0 vector attention,
  `ggml.flash_attn_ext.vec_q8_0`).
- The D-073 question (the runtime's configuration and storage-role trust
  walk): settled by the owner (AGENTS.md rule 11: files the runtime creates are
  private, another owner or looser modes are warned about, not refused).
- RE-051 (ASan and the signal stack): worked around. The M3.6 batch's tests
  synthesize families from `tests/support/families/`.

**Carried to M3.5** (the family, importer and formats track)

- Importer and mapping language:
  - *Closed in `c1dcc8c1`–`b003c9a5`:* indexed-part normalizer dispatch
    unified in `src/cli/import.cc` (`235ff8e4`) and central recognition/dispatch
    registry in `import/source_mapping.cc` (`fe17e4ae`, R4); `FromRatio` /
    `FromOrRatio` ratio rules (`fa1f7cad`) and native Gemma 2/3 HF import with
    explicit `query_pre_attn_scalar` and GGUF `<arch>.attention.scale`
    (`fe17e4ae`, R4); requiring stage 0 as target, refusing stage-read cycles,
    Gemma 3 `sliding_window` handling, and verifying mappings against the pinned
    converter/source (`fe17e4ae`, R4); structural metadata normalization,
    `Exl3Codebook` marker/format validation, member alignment, `Q8_1` block size,
    spec JSON bounds, and `PartsOf` multipart expert placement (`f5ebd16a`, R3;
    `fe17e4ae`, R4); dense NVFP4, MXFP8 grouped, and MXFP4 decode-ladder paths
    wired (`43f2f50a`, R6); unpinned v1 artifacts and drafters with another head
    list or none admitted structurally (`235ff8e4`, D-119; `aba73a61`, R1;
    `fe17e4ae`, R4); generic dense-decoder mapping for `qwen2`, `llama` and
    `mistral` across GGUF, safetensors and EXL3 (`b003c9a5`, R11); served and
    verified only for Llama 3.2 1B (GGUF) and Qwen2.5 0.5B (GGUF FP16, EXL3),
    the rest by synthetic-source tests (model-support.md).
  - *Still open in M3.5 ([plan.md](plan.md)):* generalizing the retained
    Qwen3.8 ModelOpt and MLX normalizers into mapping-driven normalizers when
    Nemotron NVFP4 brings a second ModelOpt model; split-NVFP4 experts' own
    fast products (a grouped vector product for decode rows and the
    block-scaled grouped GEMM over split scales); MXFP4 and MXFP8 routed-expert
    and dense-NVFP4 relayout; MLX serving qualification; full-weight Gemma 2/3
    27B qualification.
- Components and lifecycle:
  - *Closed in `c1dcc8c1`–`aa92bd1f`:* vocabulary bounds derived from
    the stored target head and enforced (`fe17e4ae`, R4; `0d28a553`, R9);
    `fused.mul_mat_glu` operand order (gate 0, up 1, input 2) defined and
    verified by the IR (`40bf23e3`, R2); `SlotLifecycle` `Admit` position
    checks, `Settle`/`DiscardVerify`, mid-restore `CheckActive`, borrow tokens,
    all-or-nothing widening acquisition, `PrefillOutcome` joined outputs,
    `want_head` join checks, failed-job quarantine accounting, and feature copy
    coalescing (`aba73a61`, R1; `e6c3617a`, R10); multi-level verification and
    engine-owned `PrefillOrchestrator`, `PrefillLookahead`, `WaveComposer`,
    `DecodingOrchestrator` and `StepTransaction` (`e6c3617a`, R10); layer-level
    CPU reference vs GPU checks across every catalog component and family
    (`components_catalog_gpu_test`, `aa92bd1f`, R12) and Gemma 3 grouped cache-store /
    shared planner pattern qualification (`43f2f50a`, R6; `d3374c67`, R7);
    parametric `CapabilityQuery` node/format predicates and capture-aware
    `PlanKey` (`80ede618`, R5; `43f2f50a`, R6; `d3374c67`, R7); general
    `CheckReads` stage taps, borrowed-state store refusals and committed-position
    clamping (`fa1f7cad`; `aba73a61`, R1; `fe17e4ae`, R4; `0d28a553`, R9);
    `ffn.moe` `gemma_route_reduce`, `kView` of expert-local values and MTP
    catch-up rows (`fa1f7cad`; `40bf23e3`, R2; `80ede618`, R5; `e6c3617a`, R10);
    builder evaluation-order independence and unified `attention.dense` v1 with
    merged fused kinds and format-independent specs (`40bf23e3`, R2; `0d28a553`,
    R9).
  - *Still open ([plan.md](plan.md), [rough-edges.md](rough-edges.md)):* RE-050
    (masking logit rows between `tokenizer_vocab` and `vocab_size` when a head is
    wider than its tokenizer); adaptive depth / device verdict for the Gemma 4
    assistant (M9).
- EXL3 serving:
  - *Closed in R5 (`80ede618`), R6 (`43f2f50a`) and R11 (`b003c9a5`):* EXL3 fallback launch geometry derived from
    device facts (`80ede618`, R5); EXL3 sources wired through capability
    queries, capture-aware `PlanKey` and separately paid backend state, dropping
    reconstruction candidates without a pin (`43f2f50a`, R6); EXL3 Qwen2 served
    end-to-end through `rt::Server` / `EngineLlm` (`dense_decoder_serving_gpu_test`,
    `b003c9a5`, R11).
  - *RE-058 closed in R6 (`43f2f50a`):* `SelectedSource`'s lock slots are
    separate per-runner kernel memory, so an EXL3 model can serve beside GGML
    models; nothing remains open here.
- Lossy KV:
  - *Closed in `fa1f7cad` and R9:* `KvCacheEncodingBytes` and
    `ApplyKvCacheEncoding` made data-driven over component parameter
    declarations (`fa1f7cad`), and attention unified onto `attention.dense` v1
    (`0d28a553`, R9).
  - *Still open ([plan.md](plan.md)):* `Q4_0` as a setting (M3.5), and a
    convert-then-tensor-core prefill path for quantized caches (M9).
- Licensing sweep residue:
  - *Closed in `c1dcc8c1`, `235ff8e4`, `40bf23e3` (R2) and `0d28a553` (R9):*
    SPDX sidecars and attribution comments added for M3.6 preregistration files,
    CUTLASS BSD-3-Clause headers, `validate.cc`/`validate_ext.cc`,
    `decode_exl3.cc` and YaRN/component sources; REUSE and header gates clean.
- Test infrastructure:
  - *Closed in R3 (`f5ebd16a`) and R12 (`aa92bd1f`):* `CheckImporter` deduplicated into shared bounded
    validation in `src/artifact/` (`f5ebd16a`, R3); `RESOURCE_LOCK llmp_gpu`
    applied to all `gpu`-labeled CTest targets to prevent concurrent GPU context
    exhaustion under `--parallel` (`aa92bd1f`, R12); all 19 Qwen2/EXL3/FP16 fixture tests,
    importer model tests, tokenizer/chat model tests, and `gemma_engine_greedy`
    wired to published v1 artifacts and `LLMP_TEST_V1_STORE` with `0` skips in
    the shared 3,373-test suite (`aa92bd1f`, R12).
  - *Still open ([plan.md](plan.md)):* `CudaPageIn.RepeatedCancellationsNeverCorruptAReassignedSlot`
    (monitor if the one-off `5b21fe48` segfault recurs); `native` preset test
    binaries' cuBLAS `RUNPATH`; `cpu-asan` `-fsanitize=function` on GGML
    `to_float` calls and running the importer mutation test under ASan (`check:full`
    / `check:spark` shipment tiers); M5's standalone artifact verifier.

**Carried to M4** (two Sparks)

- A shared pageable workspace rules out two models executing concurrently on
  one node: revisit with placement (a workspace per executing model, or
  disjoint ranges of one).
- Partitioning: side sources and stream mixing, `StreamBytes` scaling by
  `max(1, stage.streams)`, `MakeFull` collective deduplication, sparse side
  transfers, shared-weight deduplication in search budgets, and device
  bandwidth/compute accounting were closed in `fa1f7cad`, `7283b964` and
  `d3374c67` (R7). Remaining for M4 ([plan.md](plan.md)): mixing split and
  replicated layers within a TP stage; raising the 16-device cap if needed;
  vocabulary-split embeddings; EP all-to-all / DP-attention mode, single-expert-per-device
  device-graph verification, and `select` in place of mask-by-zero for
  non-finite other-device products.
- Lowering with several devices: `GgmlStepLaunch` stream binding, one backend
  per runner across swaps, `MmvqRouted`'s 3-D count, and any remaining
  multi-device lowering checks remain M4's ([plan.md](plan.md)).

**Carried to M9** (the full-engine optimization pass)

- Performance leads: Qwen3.8 MTP activation memory was reduced by −4.53 GiB on
  `qwen38` and −4.37 GiB on `qwen38-gguf` in `d3374c67` (R7: backend aliasing
  and liveness facts in backend placement), and duplicate per-owner step builds
  in joined prompt batches were collapsed to a single build in `e6c3617a` (R10).
  Remaining for M9 ([plan.md](plan.md)): further MTP activation and plan-build
  reductions and decode-first lookahead for a new shape; the 0.25 s that a swap into
  Qwen3.8 loses to mapping the workspace after the reads; decode
  template leads (vector template register blocking at 16 rows, a grouped
  warp-per-row decode variant, multi-stage tiles); one-shape-per-band probes
  for the graph reader index instead of the 25% margin; moving turn-checkpoint
  capture off the first-token path; DeepSeek's budget cliff (about 1.1 GiB less
  `MemAvailable` took qwen3.8 to deepseek 16.1 s against 8.15 s: near the 20 s
  bound; it did not recur in the final table, whose worst LLM-to-LLM row is
  8.665 s with a lowest MemAvailable of 4.69 GiB); Qwen's long-context
  acceptance cost; the per-technique structural adoption audit against
  [optimization-inventory.md](optimization-inventory.md); kept-record
  per-drafter adaptive depths; the full-context K1 scoring path.
- Workspace backing and memory modes (from the image slowdown, 2026-10-17):
  a workspace backed extent by extent during a page-in, while the kernel
  reclaims a full page cache, ran the image's denoiser products up to 5%
  slower (the down projection +24% in its slowest run); one mapped resident
  at the start did not, so a node maps it resident whenever it fits
  (`be5c4db6`). Open: (a) the **per-extent mapping hypothesis**: range-wide
  access on contiguous lane-backed runs of extents (one `cuMemSetAccess`
  over a run, as the resident map does) to test whether the per-extent
  mapping pattern, rather than the backing's timing, slows the products (an
  extent-by-extent backing ahead of the weights did not remove the tail and
  slowed image swaps 1.4 to 2.5 s); (b) the **dedicated physical handle
  pool** (owned by M3.5's first work item, not M9; listed for history) beside the shared-machine mode ([features.md](features.md), the
  "Physical-memory mode per node" row, proposed on the owner's direction
  and amended in `be5c4db6`); (c) **Qwen3.8 MTP's activation footprint**
  at four slots (8.05 GiB at `53c2610b`, reduced by 4.53 / 4.37 GiB in R7
  `d3374c67`; 1.42 GB in M3); (d) the **image to deepseek first-use swap at 27.354 s** (the
  workspace pages for the first load beside DeepSeek; inside the 40 s
  first-use bound, its prepared return 8.804 s).
- The remaining Qwen and DeepSeek speed gaps accepted at M3 (m3-record.md)
  stay M9's.

## Work handed on

- **M3.5 resumes** on this engine: a family is an architecture mapping, any
  components its shape lacks, and (for the reference fleet) an approvals row
  with its evidence (unpinned v1 artifacts of supported architectures admit
  directly under D-119, `235ff8e4`). Following the `R1`–`R13` review follow-up
  campaign (which added the generic dense-decoder mapping and served Llama 3.2
  1B Instruct and Qwen2 in R11, `b003c9a5`), the next work items are physical memory modes
  (below), Nemotron 3.5 Super VL (owner-requested), generalizing the importer's
  ModelOpt and MLX normalizers, MLX serving qualification, split-NVFP4 fast
  products, and MXFP4/MXFP8 relayout. [plan.md](plan.md)'s M3.5 "Carried in"
  and M9 lists name every item still open, one by one; this record keeps only
  the history, and [m35-families.md](m35-families.md) holds the family queue.
- **M4:** two-Spark execution of the planner's partitions; concurrent models
  on one node need the placement work above.
- **M9:** the optimization leads above, on top of the shared engine.
- **M3.5's first item, physical memory modes** (owner, 2026-10-16/17): the
  dedicated node pool and the shared-machine mode (features.md, now
  `confirmed`), the fix for the DSpark C4 wall exception above (RE-059).
- **Workspace and memory modes (the rest, M9):** the per-extent mapping
  hypothesis and any remaining Qwen3.8 MTP activation gap after R7
  (`d3374c67`); the image to deepseek first-use swap at 27.354 s; RE-057 (drop
  the artifact stores' page cache before timing); the heat-soak note above.
  (The dedicated physical handle pool is M3.5's physical-memory-modes item,
  not M9's; RE-058 is fixed.)
- **Tooling and environment:** the `native` preset's test binaries lack the
  cuBLAS RUNPATH (they run only with `LD_LIBRARY_PATH` set); the
  workstation's NVIDIA driver is mismatched, kernel 595.91.07 against
  userspace 595.99.02 (the owner's to fix; the sm_86 proof ran through it).
- **Skipped tests (resolved in R12, `aa92bd1f`):** the 19 Qwen2/EXL3/FP16 fixture tests,
  importer model tests, and `gemma_engine_greedy` are now wired to the
  published v1 store (`LLMP_TEST_V1_STORE`) and deterministic controls (`aa92bd1f`, R12);
  only opt-in full-checkpoint publication tests and explicit diagnostic screens
  remain env-gated in `models|timing`, and the shared suite (`-LE 'models|timing'`)
  runs 3,373/3,373 with 0 skips.
- **Shipment:** the tiers `check`, `check:full` (sanitizers, container,
  offline copyleft-disabled), `check:spark` (cross, ASan, TSan) and the
  package install fixture, plus the confined-job proof, are owed before
  publishing a package (D-085).
- **Cleanup:** per AGENTS.md rule 8, raw outputs, logs and old worktrees are
  deleted when the milestone closes, except worktrees with unmerged work,
  which are listed for the owner and never dropped.

## Retained scope history

The plan's M3.6 scope as it stood when the milestone closed, moved here
verbatim from [plan.md](plan.md). Nested sub-task boxes keep the status they
had when the work moved: an unchecked sub-task here was closed by a later
slice, folded into the final verification, or handed on above, and the
sections above and the plan's final scope list are authoritative.

- [x] **Design and decisions:** [engine-components.md](engine-components.md),
      D-107 (components, IR, execution, partitioning, migration gate),
      D-108 (formats), D-109 (importer), D-110 (platform order); M3.5's
      transfer item retired with its [mapping](engine-components.md#retired-transfer-items).
- [x] **Baselines and the 1% protocol.** Before any family moves, fix the
      matched, bookended timing protocol that resolves 1% for each family's
      prefill and decode, solo and at its qualified cohorts, and record the
      current implementation's speed, quality controls, peak memory and
      swap rows as each family's baseline.
      *2026-10-08:* protocol, frozen `1facc957` baseline, same-binary
      spread and baseline rows recorded in
      [m36-baselines](experiments/m36-baselines/README.md) (harness
      `tools/speed_gate.py`). Its two defects (DeepSeek's four-slot DSpark
      waves, the swap table's Qwen3.8 continuation) are fixed; the
      `1facc957-fixed` baseline (the frozen build plus exactly those
      fixes) holds DeepSeek's DSpark cohort rows and the swap table's
      Qwen3.8-as-A rows (its 32 rows all exact there, worst LLM-to-LLM
      8.494 s).
- [ ] **Reference-model quality and numerical recovery** (owner, 2026-10-12):
      compare candidate and baseline with a reference implementation of the
      same quantized checkpoint using NLL/perplexity and behavioral controls.
      Use original source-precision evidence when available; it is required
      for a closer-to-original claim, not for same-quantized-model qualification.
      Preserve the 1% speed and same-implementation state gates.
      - [x] Retrospective numerical-policy audit and prospective gate clarification
            ([audit](experiments/m36-numerical-audit/README.md)).
      - [x] Image timing output controls use a frozen pin per implementation,
            preserving the old arm and historical records without requiring
            candidate pixels to match legacy bits. Reference image quality
            remains separate; workload and the 1% speed protocol are unchanged.
      - [x] Shared target-only teacher-forced scorer foundation: paid finite
            IDs/outputs, global next-target alignment, full-vocabulary NLL and
            guarded head/owner semantics; ten host controls pass. This is
            test support, without a real-model quality qualification.
      - [x] Real Qwen2 FP16 scorer adapter: bounded per-target reports through
            the actual shared backend, selected implementation identities and
            sink refusal/cancellation controls. Main native build, 114 affected
            controls and both actual natural-ID policies pass without skips.
      - [x] [Original Qwen2 BF16 likelihood smoke](experiments/m36-original-qwen2/README.md):
            authenticated complete original weights and 43 natural token meanings,
            42 aligned target losses through original and shared FP16 recipes.
            This is a tiny observation, not a quality acceptance gate.
      - [x] Real EXL3 4.0-bpw G scorer adapter: shared scorer/plan recorder,
            complete original-tokenizer identity and all 42 natural targets.
            Supported seven/one-row tuning is retained; missing three-row tuning
            refuses. Main native build, 124 affected registrations and all three
            FP16/EXL3 scoring cases pass without skips. Original-model observation
            is bounded, without a quality gate.
      - [ ] Broader fixed corpus and same-checkpoint reference comparisons;
            optional original-reference comparisons for original-agreement claims;
            production-phase, speculation, context and cohort controls for chosen candidates.
            - [x] Pinned Mia Qwen same-quantized-checkpoint baseline and
                  current-main manual A/C diagnostic: all 1,016 fixed targets
                  independently checked. Mia PPL 35.144838403; native A/C
                  +0.389325%/−0.248639% relative to it. Main native build,
                  32 host controls and both actual cases pass. Experimental
                  policies and cache differences are explicit
                  ([reference and replay](experiments/m36-numerical-audit/qwen-quantized-reference.md)).
            - [x] Matched one-row Qwen mapping-policy screen: unchanged policy
                  PPL 35.402897009 versus 35.281665930 with only the owner-row
                  restriction disabled (−0.342433%, same 1,016 targets).
                  Full native build, 32 host controls and both actual cases
                  pass. This whole-policy observation does not adopt a default.
            - [x] Bounded ordinary Qwen policy/state follow-up: real C1/C4
                  Prompt/Decode at read256, 20 independently measured classes,
                  checkpoint/spill replay and complete final initialized-state
                  hashes. Main native build, 33 host controls and both actual
                  cases pass. No cross-cohort bits gate, new NLL or memory-saving
                  claim; default adoption, MTP, context and family gates remain open.
      - [x] First Qwen2 fused-attention recovery screen: actual selected attention,
            aligned natural smoke losses and four fresh-fixture ABBA cases pass.
            Timing includes setup/I/O and baseline drift; no quality, serving-speed
            or production-adoption gate follows ([observation](experiments/m36-original-qwen2/README.md)).
      - [ ] Evaluate recovery candidates for quality, speed, reuse, simplicity,
            CPU cost and memory traffic. Equal quality/speed with greater reuse
            or simpler code, or measured resource savings hidden by another
            bottleneck, can justify a switch within the migration gates. Retire
            dominated machinery after functional and quality qualification.
- [ ] **Format layer** (D-108): encodings named independently of kernel
      libraries and layouts recorded separately; a block decoder per
      encoding on CUDA under shared vector, matrix and grouped templates;
      the fallback ladder; decoders composable inside fused kernels for
      mixed per-tensor encodings; state encodings, lossy ones as per-alias
      settings. First proof: Qwen3.8's native NVFP4/MXFP8 tensors on the
      discrete `sm_86` target, product by product against CPU references
      (D-082; the whole model does not fit the 3080 Ti). The Qwen2
      EXL3 fixture proves the trellis encoding through the same layer.
      Sub-tasks (owner, 2026-10-08: each lands as one reviewed commit):
      - [x] Encodings, layouts and the size rule; v0 representations
            mapped at read time; CPU reference decoders; registry
            capabilities and the ladder's ordering.
      - [x] CUDA block decoders under the vector, matrix, grouped and GLU
            templates; the `sm_86` proof.
      - [x] The EXL3 trellis through the layer; state encodings (the
            format half: which encodings, exact or lossy, CPU encode and
            decode).
      - [ ] Lossy state encodings end to end: attention that reads them,
            quantizing cache stores, the state contracts' sizes and
            kept-state layout, and the per-alias setting
            (engine-components.md).
            *2026-10-09:* the kernel half landed: GGML's vector attention
            over Q8_0 and Q4_0 K and V at D 64 and 256, the quantizing
            stores (plain, grouped, fused with RoPE) byte for byte with
            the reference encoder, cell sizes by the size rule, with
            capabilities and graph planning
            ([engine-components.md](engine-components.md#weight-and-state-formats)).
            - [x] Bounded shared dense Q8_0 opt-in at D64/D256: blocked
                  stores/readers, source/state/kept identity and explicit
                  per-alias `kv_cache_encoding` (D-117), preserving F16 by
                  default. Main native build and 519 host controls pass;
                  four actual cases cover short paired NLL and same-Q8
                  checkpoint/spill/kept restoration. PPL rises 4.598%/1.452%
                  in the two short samples; no quality threshold or speed
                  claim. [Evidence](experiments/m36-numerical-audit/q8-state.md).
            K1 extends it to every migrated LLM (D512, QSA, MQA, borrowed
            and drafter state) with 6K–16K-token quality against F16;
            Qwen3.8/DeepSeek at their full contexts remain open
            ([report](experiments/m36-numerical-audit/q8-state.md#long-context-q8_0-vs-f16-quality-across-migrated-llms-k1)).
- [x] **Shared slot lifecycle, one runner interface, one serving adapter
      per pipeline kind**, derived from state contracts; reconcile today's
      divergent copies (Gemma2's restore guards against Gemma3's).
      Decoding-mode orchestration takes any number of drafters and the
      planning and capture path is backend-neutral (M3.5's moved skeleton
      gaps), tested on the fake backend with two drafters.
      Sub-tasks:
      - [x] The engine core builds in every profile, with a fake device
            runtime and host paging kernels (D-096 amended).
      - [x] State contracts and the shared slot lifecycle, with the
            reconciled guards (the legacy runners do not delegate yet).
      - [x] One runner interface: step programs, wave composition,
            prefill and decoding-mode orchestration with any number of
            drafters (cascaded or alternative, chosen by policy).
            *2026-10-09:* `engine/model_runner.h`, `wave_composer.h`,
            `prefill.h`, `decoding.h`, on the fake backend with two
            drafters; no family uses them yet.
      - [x] One serving adapter per pipeline kind.
            *2026-10-09:* `runtime/engine_llm.h` and `engine_image.h` (the
            image one a shell until Qwen-Image moves), `Server::Make`'s
            dispatch by pipeline kind behind a diagnostic set of
            architectures (empty: each family's migration wires its parts),
            kept conversations in a file a stage (record version 2, D-112);
            `engine_llm_test` on the fake backend.
- [x] **IR and planner** (D-107): the op set and tensor descriptors,
      construction from a spec, fusion patterns covering today's selected
      fused paths, implementation selection, activation placement, lowering
      to recorded work, plan keys; a CPU reference and fake-backend tests
      for every component.
      Sub-tasks:
      - [x] The IR graph, its verifier and the reference executor; the
            model spec and its v0 synthesis.
            *2026-10-09:* `src/ir`, `kernels/reference/exec.h`, `src/spec`
            with the Gemma 2 and Gemma 3 mappings
            ([as built](engine-components.md#the-model-spec)).
      - [x] The component framework and the planner; lowering to the
            GGML kernel source with structural parity (EXL3's and the
            image kernels' lowerings land with their families'
            migrations).
            *2026-10-09:* `src/components`, `src/planner`, the GGML
            module's capabilities and lowering and the component step
            source ([as built](engine-components.md#the-graph-ir)), at
            parity with Gemma 2's and Gemma 3's builders; their
            owner-attention forms and narrowed final layer land with the
            Gemma 2/3 migration.
- [x] **Partitioning in the planner:** tensor, pipeline and expert
      parallelism with any split count and mixed, chosen by modelled
      traffic for a described topology; equivalence tests over four or more
      fake devices. Two-Spark execution is M4's.
      *2026-10-09:* `planner/partition.h`, the components' splits as data;
      TP, PP, EP, TP2×PP2 and six-device mixes equal the unpartitioned
      graphs on the reference executor
      ([as built](engine-components.md#partitioning)).
- [x] **Migrate Gemma2 and Gemma3**, then delete their runners, plans,
      graph builders, bindings and serving classes.
      - [x] Components, patterns and the mapping.
      - [x] Wiring and the gate, then deletion of the old code.
            *2026-10-09:* wired: Gemma 2 and Gemma 3 serve on the shared
            engine by default (the mapping's policy, the head's width,
            cuBLAS over a runner-mapped workspace, device greedy;
            [as built](engine.md#the-serving-adapter)); same-binary heads
            and tokens bit-identical to the M3 classes, the quality
            control's scores and the near-8K switch rows' continuations
            equal the frozen baseline's. The 1% gate passes (Gemma 2 at 5
            blocks, Gemma 3 at 7; the new arm runs with the gate-only
            switch `LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS=1`, which saves no
            turn checkpoint, as the baseline's M3 classes never could;
            [the results](experiments/m36-baselines/README.md#gemma-2-and-gemma-3-on-the-shared-engine-slice-g1c)).
            *2026-10-10:* deleted: the M3 runners, plans, graph builders and
            serving classes with their `ServingOptions` flags, tests and
            benchmarks; `model/gemma{2,3}.h` keep the approved profile and
            tensor binding the admission and the mappings use.
- [x] **Migrate Gemma4** 26B-A4B and 31B with their assistants.
      - [x] Components, patterns and the mapping.
            *2026-10-09:* the components and mappings landed (slice G2a:
            `ffn.moe`, `attention.borrowed`, `embedding.projected`,
            `scale.learned`, Gemma 4's attention and head forms; the
            `gemma4` and `gemma4-assistant` mappings, the assistant a
            drafter stage composed with its target; [as
            built](engine-components.md#initial-catalog)), at parity with
            today's builders' graphs (the narrowed final layer included;
            the assistant's head reads every row and does not narrow).
            The patterns and lowering (slice G2b: the fused routing and
            reduce as `route_reduce`, owner attention's Gemma 4 forms, the
            narrowed final layer as a policy option, the gemma4 mapping's
            policy per serving envelope, `kMatMulId`, the drafter's
            features, tap cells and frozen masks; [as
            built](engine-components.md#the-graph-ir)) plan and lower
            every setup shape of today's three recipes to today's
            launches, and the 26B-A4B, 31B and both assistants' draft
            passes run bit for bit today's through the model runner.
      - [x] Wiring and the gate, then deletion of the old code.
            *2026-10-10:* wired and gated (slice G2c). Both profiles serve
            on the shared engine, and with a `drafter` their assistants (an anchored drafter reading the
            target's tapped features: `FeatureFrom`, the runner's carry,
            frozen reads, `StagedStepSource`; [as
            built](engine.md#the-model-runner)); the quality control's
            scores and replies equal the frozen baseline's, the near-8K
            switch rows are exact and the frozen build's kept records are
            refused. The 1% gate passes (26B-A4B at 9 blocks, 31B at 7, the
            new arm with `LLMP_DIAGNOSTIC_SKIP_TURN_CHECKPOINTS=1`: the M3
            class has defect 3 too), and the 31B assistant's screen runs
            4.41% faster with its transactions byte-exact
            ([the results](experiments/m36-baselines/README.md#gemma-4-26b-a4b-and-31b-on-the-shared-engine-slice-g2c));
            the 26B-A4B's assistant has no baseline to gate against.
            *2026-10-10:* deleted (slice G2d): the M3 `Gemma` serving class,
            `engine/gemma4_*`, the GGML graph builders and the owner
            transform, the state and chunk builders (`model/gemma4.h` keeps
            the approved profiles and the tensor binding), the
            `ServingOptions` Gemma diagnostics and
            `gemma26_production`/`gemma31_production` (the 256- and
            1,024-row prompt chunks are the mapping's `prefill_chunk`),
            their benchmarks and tests. Coverage is on the engine: the
            run-to-run and reference checks of `component_runner_gemma4_gpu_test`
            and `components_gemma4_models_gpu_test`, the served four-owner
            8K check (`Gemma4EnvelopeServingGpu`), the cross-variant kept
            tag and the served assistant (`Gemma4AssistantServingGpu`).
- [ ] **Migrate Qwen3.8**: native NVFP4/MXFP8 with MTP, and GGUF.
      - [x] Components, patterns and the mapping.
            *2026-10-10:* the components and mappings landed (slice G3a:
            `residual.hc`, `hc.mix`, `hc.combine`, `attention.gdn`,
            `attention.qsa`, `side.ngram`, `residual.ple`, `embedding.mtp`,
            `ffn.moe`'s shared-gated form; streams, side inputs, fixed,
            per-block and per-row cells, a verify's saves and its commit;
            the `qwen4exp` mapping over the ModelOpt and GGUF imports and
            `qwen4exp-mtp`, the MTP drafter a stage composed with its
            target; [as built](engine-components.md#initial-catalog)), at
            parity with the unfused builder's graphs but for the n-gram
            layer's node order. The fast form's patterns, the default
            policies and the lowering (slice G3b: [as
            built](engine-components.md#the-graph-ir)) plan and lower
            every setup shape of the three imports, with the MTP drafter,
            to today's fast-form launches; on the real artifacts the
            model runner's 31 outputs and 242 state tensors are bit-identical
            to today's fast form in each import, including joined owners,
            selection beyond the QSA budget and captured replays. G3c1
            proves native verify and all-three-format commits; G3c2a proves
            native single-owner MTP and persistence. Joined MTP arithmetic
            remains G3c2b1 before serving and the family gates.
      - [ ] Wiring and the gate, then deletion of the old code.
            - [x] **G3c1 — shared runner execution:** bind verify saves and
                  commit execution; bind n-gram side tables on demand; stage
                  drafter features with fake-backend lifecycle controls.
                  Prove focused GPU numerical correctness for native target
                  verify and commits of k of n rows across all three formats.
            - [ ] **G3c2 — composed MTP, serving and gate:** prove actual
                  MTP catch-up and draft-pass GPU numerics and checkpoint/
                  kept-state feature restoration, then wire native, GGUF
                  and MTP through serving; pass state, quality and swap
                  controls and the matched 1% gate for all three paths.
                  - [x] **G3c2a — MTP progress and persistence:** declare the
                        cache/token/feature relation and pending replay
                        range as shared stage policy; preserve it with the
                        feature span through checkpoint, spill and adoption;
                        run the prefill companion and prove actual MTP GPU
                        and restoration behavior before serving.
                        *2026-10-10:* opt-in replay progress, completion-owned
                        control, prompt companions and layout-derived checkpoint
                        tails pass native single-owner MTP/persistence proof.
                        Unsupported joined operations and lookahead shapes refuse
                        before mutation; joined arithmetic remains G3c2b1.
                  - [ ] **G3c2b — joined lowering, serving and gate:**
                        qualify joined MTP arithmetic before serving and the
                        cohort gates.
                        - [ ] **G3c2b1 — joined arithmetic:** qualify shared
                              arithmetic and routing for joined MTP steps,
                              including expanded streams and unequal pending
                              ranges. Use a reference implementation of the same
                              quantized checkpoint as the quality baseline;
                              original-model evidence is needed for claims of
                              closer agreement with the original.
                              Preserve owner coverage and every selected expert
                              contribution; prove same-implementation state
                              restoration and changed-input capture replay.
                              - [x] **G3c2b1a — default-off owner products:**
                                    typed physical-column proof, per-owner
                                    primitive rewrite and bound native-recipe
                                    audit; structural and descriptor controls.
                                    No source/runtime opt-in or numerical/
                                    persistence qualification.
                              - [ ] **G3c2b1b — native qualification:** qualify
                                    joined full-head quality and routing,
                                    same-implementation state and changed-input
                                    replay, old/new persistence and bounded
                                    admission before enabling repeat-pending
                                    execution.
                                    - [x] **G3c2b1b1 — selected-HC recipe
                                          correction:** keep source policy
                                          default-off; prove actual selected-row
                                          handling and the native recipe/selector;
                                          pass the bounded first screen and
                                          permanent host, direct-GPU and
                                          integrated regression checks (clean03:
                                          169 host and six GPU checks passed,
                                          including model-free `hc.mix` complete
                                          outputs and ragged/replay coverage).
                                          The bounded 09c screen covers both owners' full
                                          selected HC products and K1/K3 heads;
                                          joined state and runtime admission
                                          remain outside this correction.
                                    - [ ] **G3c2b1b2 — remaining joined
                                          qualification:** prove joined quality,
                                          same-implementation state, changed-input
                                          replay, old/new persistence and bounded
                                          admission. The G3c2b1b parent remains
                                          open after the selected-HC correction.
                                          The owned target/MTP diagnostic now
                                          carries explicit joined-repeat-pending
                                          source eligibility and a two-owner
                                          unequal-pending recovery/capture test.
                                          Main native build and 305 affected host
                                          controls pass; the actual two-owner
                                          target/MTP screen passes 13 units,
                                          132 finite heads, checkpoint/spill
                                          restoration and changed-input capture
                                          replay (2026-10-12). The source accepts
                                          complete v0 multipart row tables through
                                          the shared artifact assembler, and draft
                                          heads use their actual selected width.
                                          Broader same-quantized-reference quality, durable old/new
                                          persistence and serving admission remain
                                          open ([evidence](experiments/m36-numerical-audit/qwen-joined-mtp.md)).
                              - [x] **G3c2b1c — inactive accurate dense-product
                                    capability:** expose the existing joined
                                    BF16-weight/F32-input vector kernel through
                                    exact typed tuning and physical descriptor
                                    checks, independently of owner-width legacy
                                    recipes. No runtime source installs pins;
                                    wider shapes keep ordinary selection.
                                    Full main Spark B native build and all98 focused
                                    owner/registry/planner/identity controls pass
                                    (2026-10-12). The inherited bounded kernel
                                    [replays46/47](experiments/m36-numerical-audit/qwen-products.md)
                                    measure local error and paired path timing,
                                    not original-model quality or serving gates.
                              - [x] **G3c2b1d — owned backend configuration:**
                                    immutable qualified build/runtime and tuning
                                    declarations, paid preparation before source
                                    identity, and source/plan/snapshot/kept bindings.
                                    Explicit pre-start metadata allowance and
                                    stable launch generations protect retries and
                                    stale direct/captured work. Main passes the full
                                    native build, 148 focused host/refusal controls,
                                    one model-free native backend control and the
                                    three existing FP16 adapter controls. No new
                                    numerical policy or joined capability follows
                                    ([evidence](experiments/m36-numerical-audit/qwen-products.md#owned-backend-configuration)).
                              - [x] **G3c2b1e — shared phase admission:**
                                    qualify actual prefill/decode/score target
                                    waves before effects, retaining compound
                                    permissions. Main native build and all 60
                                    affected host controls pass; arithmetic
                                    qualification remains separate ([evidence](experiments/m36-numerical-audit/qwen-products.md#shared-phase-admission-correction--2026-10-12)).
                              - [x] **G3c2b1f — checked weight reservation facts:**
                                    carry actual single-part reserved-address
                                    alignment through shared measurement, planning
                                    and lowering. Unknown facts preserve ordinary
                                    selection; invalid promises refuse. Full native
                                    main build and 108 focused host controls pass,
                                    including eleven valid/invalid/no-tuning modes.
                                    No arithmetic or tuning policy is activated
                                    ([evidence](experiments/m36-numerical-audit/qwen-products.md#checked-weight-alignment--2026-10-12)).
                              - [x] **G3c2b1g — real-program and full-head witnesses:**
                                    validate the actual lowerer's unchanged program
                                    and stream finite full-head rows with exact
                                    document/owner/position coordinates. Admit only
                                    the closed target-only diagnostic factory domain,
                                    measuring all fallback score shapes. Main native
                                    build and 127 focused controls pass; no tuning or
                                    serving default changes ([evidence](experiments/m36-numerical-audit/qwen-products.md#shared-quality-witnesses--2026-10-12)).
                              - [x] **G3c2b1h — short actual decode diagnostic:**
                                    independently measure and execute the private
                                    C4 Decode256 class; eight generated steps plus
                                    checkpoint/spill replay pass with 32 reference
                                    and 32 exact replay heads/tokens, eight checkpoint
                                    and four spill restores. This closes only the
                                    bounded diagnostic, with broader quality and
                                    family gates open ([evidence](experiments/m36-numerical-audit/qwen-joined-score.md#actual-decode-continuation--2026-10-12)).
                              - [x] **G3c2b1i — broader fixed natural secondary corpus:**
                                    four pre-fixed 255-input histories, 1,016 targets;
                                    complete actual Score8/4 program and finite-head
                                    controls pass in one fresh A/C pair. Aggregate
                                    PPL changes −0.635489%, but new 888 targets only
                                    −0.057291% with two documents regressing. This
                                    descriptive result adopts no policy or threshold;
                                    broader same-quantized-reference quality and family gates remain open
                                    ([report](experiments/m36-numerical-audit/qwen-joined-score.md#broader-fixed-natural-corpus--qwen52)).
                        - [ ] **G3c2b2 — serving and gate:** compose the
                              native, GGUF and MTP serving paths, pass state,
                              quality and swap controls, then the matched 1%
                              gate at every qualified cohort.
                              - [x] **G3c2b2a — sole-owner serving:** compose
                                    native MTP and GGUF through the shared
                                    adapter, with bounded startup accounting
                                    and complete-draft context admission.
                                    Focused serving and final Spark regression
                                    checks pass (2026-10-11).
                                    Joined execution and the family gates
                                    remain G3c2b1b2 and G3c2b2.
                              - [x] **G3c2b2b — cold kept-state hydration:**
                                    preserve an adopted prefix while its drafter
                                    progress awaits restoration. Permit reuse
                                    and selection only for valid cold tickets;
                                    execution still requires completed hydration.
                                    Native build and 77 focused host/fake controls
                                    pass on Spark B (2026-10-12), including partial
                                    restore capacity refusal/retry without
                                    recomputation
                                    and independent restoration of two cold peers.
                                    The actual owned C2 target/MTP durable
                                    control also passes on Spark A (2026-10-13):
                                    two sequential EngineLlm owners, four kept
                                    stage files, cold hydration without prefix
                                    execution, eight fresh continuation heads,
                                    and complete boundary/final initialized
                                    state within the same selected implementation
                                    ([report](experiments/m36-numerical-audit/qwen-joined-mtp.md#fresh-owner-durable-c2-continuation)).
                                    The internal Server route additionally derives
                                    its paid domain from the actual composition
                                    and passes default lookahead/capture, retained
                                    turn checkpoints and fresh-owner C2 hydration
                                    with eight finite new continuation heads
                                    ([Server control](experiments/m36-numerical-audit/qwen-joined-mtp.md#internal-server-c2-defaults-and-reopen)).
                                    A short current-implementation C2 MTP ABBA
                                    observes shared prompt/generation walls
                                    8.53%/2.17% lower, with slower startup and
                                    equal observed token work. This is a coarse
                                    diagnostic, not the fixed-baseline 1% gate
                                    ([timing screen](experiments/m36-numerical-audit/qwen-joined-mtp.md#short-c2-mtp-whole-implementation-timing-screen)).
                                    An explicitly funded internal HTTP selector
                                    also passes two overlapping streamed requests:
                                    331 fresh prompt and 128 output tokens each,
                                    with 29 joined MTP waves and graceful retirement
                                    ([HTTP control](experiments/m36-numerical-audit/qwen-joined-mtp.md#supervised-shared-http-c2-control)).
                                    The same internal selector now derives an
                                    ordinary target-only Prompt/Decode domain.
                                    Native C2 HTTP requests complete 128 tokens
                                    each through 127 joined ordinary waves;
                                    repeating the first prompt reuses 326 tokens
                                    and completes 16 more
                                    ([plain control](experiments/m36-numerical-audit/qwen-joined-mtp.md#supervised-native-target-only-http-c2-control)).
                                    The same source also passes native C4: four
                                    overlapping 128-token responses, an actual
                                    four-owner joined wave and a 325-token reused
                                    prefix before 16 more outputs
                                    ([C4 control](experiments/m36-numerical-audit/qwen-joined-mtp.md#native-target-only-http-c4-control)).
                                    The existing GGUF target-only route also
                                    passes C2: two overlapping 128-token replies,
                                    127 joined ordinary waves and a 326-token
                                    reused prefix before 16 more outputs, with
                                    rendered counts authenticated before inference
                                    ([GGUF control](experiments/m36-numerical-audit/qwen-joined-mtp.md#gguf-target-only-http-c2-control)).
                                    GGUF C4 also passes with four overlapping
                                    128-token replies, a proven four-owner wave
                                    and a 325-token reused prefix before 16 more
                                    outputs, on the same qualified runtime
                                    ([GGUF C4](experiments/m36-numerical-audit/qwen-joined-mtp.md#gguf-target-only-http-c4-control)).
                                    Automatic internal startup accounting now
                                    derives parser, compiler/domain and retained
                                    tokenizer/chat payments from enforced limits
                                    and actual source/settings; explicit grants
                                    remain supported. Full native compilation and
                                    456 combined host controls pass, including
                                    355 startup-affected controls. The existing
                                    33,792-context family quality/geometry screen
                                    passes admission and complete work; its
                                    same-host current legacy/shared pair confirms
                                    a 15.55% C4 decode throughput gap, with
                                    6.38% lower wall throughput. Bounded tracing
                                    motivated a generic owner-lane candidate,
                                    but its fresh pair remains 16.66% slower in
                                    C4 decode and does not justify adoption.
                                    The experiment is preserved privately; only
                                    its independent unknown-join safety fix lands,
                                    without a family gate verdict
                                    ([startup accounting](experiments/m36-numerical-audit/qwen-joined-mtp.md#automatic-startup-accounting-and-family-geometry)).
                                    A default-off Verify capture policy now reuses
                                    the first eager run before recording. One shared
                                    off/on screen observes 3.03% higher C4 decode
                                    throughput but 0.48% lower wall throughput and
                                    extra graph/plan memory, with complete work
                                    and no reclaims. The option remains experimental
                                    ([capture screen](experiments/m36-numerical-audit/qwen-joined-mtp.md#default-off-first-verify-capture-screen)).
                                    Local commands now consume the same strict
                                    internal selector. A grouped check captures
                                    exact 996-target IDs/losses and observes PPL
                                    15.743782 versus 16.354833 from the pinned
                                    same-quantized reference (target-only; cache
                                    precision differs). Existing 8K/16 swap-table
                                    controls pass both ordered native+MTP/GGUF
                                    pairs, including real spill and exact
                                    same-selected state/full-head continuation.
                                    Full native, 156 host and 48 tool controls pass
                                    ([grouped evidence](experiments/m36-numerical-audit/qwen-joined-mtp.md#grouped-exact-quality-and-8k-saved-state-swaps)).
                                    Automatic public selection, wider mixed-family
                                    swaps, broader quality and family gates remain open.
            - [ ] **G3d — deletion:** remove the Qwen3.8 M3 runner, plans,
                  graph builders, bindings and serving class after G3c2.
- [ ] **Migrate DeepSeek V4 Flash** with DSpark.
      - [ ] Components, patterns and the mapping.
            *2026-10-10:* the components and mappings landed (slice G4a:
            the Sinkhorn mix of `residual.hc`, `hc.mix` and `hc.combine`,
            `attention.mqa` with the CSA/HCA compressors and the lightning
            indexer, `ffn.moe`'s √softplus, hash and clamped-SwiGLU forms,
            `side.hash`, `embedding.inject` and the Markov head; the
            `deepseek4` mapping and `dflash`, DSpark a stage composed with
            its target, its injection run in the target's steps;
            [as built](engine-components.md#initial-catalog)), at parity
            with the reference-form builder's one-owner graphs but for the
            mixes' carries and the hash lookups' node order. The patterns,
            the policy, owner-safe verification and the lowering of the
            new inputs are slice G4b's.
      - [ ] Wiring and the gate, then deletion of the old code.
            - [x] **G4b1 — shape contract and primitive lowering:** make the
                  HCA block count and compressed read width safe for `past`
                  (key both in the plan key or make both past-independent by
                  padding), align the MQA reference width, then prove focused
                  primitive lowering correctness for the new inputs.
            - [ ] **G4b2 — fast forms:** add patterns and default policy,
                  then prove graph/source coverage and GPU execution for the
                  fast forms and owner-safe verification. Numerical choices use
                  same-checkpoint reference quality, measured speed and
                  same-implementation state controls; external bit equality
                  is diagnostic.
                  - [x] **G4b2a — local opt-in forms:** Q-head, compressor
                        pooling, full-carry HC pre and flat-norm HC post;
                        prove guarded fallback and existing-kernel operand
                        execution without changing the default policy.
                  - [x] **G4b2a1 — lazy input ownership:** preserve the
                        caller's graph-builder component and owner when creating
                        framework inputs; prove cold/cache equivalence and all
                        four main/indexer compressor assemblies. This changes
                        metadata correctness, not arithmetic recipes. Main
                        native build and 162 focused IR/component/planner/state
                        controls pass on Spark; no whole-model run is required
                        to prove this graph metadata invariant.
                  - [ ] **G4b2b — indexer and sparse banks:** prove their
                        ownership, backing and execution contracts.
                        - [x] **G4b2b1 — state banks:** derive and bind
                              contiguous spans of owned state; prove sparse
                              attention and primitive fallback with existing
                              indexer/TopK/mask semantics retained.
                              *2026-10-10:* opt-in bank lowering derives exact
                              layout spans, checks bound addresses and whole
                              state-use coverage, and preserves primitive
                              fallback. D256/D512 operand execution matches
                              the original copy bit for bit, including unequal
                              owners and changed-input capture replay. Default
                              policy and whole-artifact parity remain G4b2c.
                        - [ ] **G4b2b2 — indexer contract:** make query
                              rounding, visibility and nonfinite selection
                              explicit; prove the native recipe without
                              silently changing the primitive TopK contract.
                              - [x] **G4b2b2a — exact selection and counts:**
                                    define visible-prefix selection and a
                                    bit-preserving mask for arbitrary IDs;
                                    stage owner-authenticated completed-block
                                    counts within logical real capacity.
                                    Scalar/native operands and changed-input
                                    replay pass; existing score, TopK and
                                    default component semantics are retained.
                              - [ ] **G4b2b2b — score and fusion:** define
                                    rounded-query scoring with a real primitive
                                    fallback and recorded numerical relation;
                                    expose one native score producer and measure
                                    unfused/fused/legacy numerical relations before
                                    opting in a quality/performance-qualified recipe.
                                    - [x] **G4b2b2b1 — typed score and fallback:**
                                          define half-rounded query scoring
                                          with an independent wider reference;
                                          overwrite invisible scores with a
                                          literal value and fund both query
                                          conversions in the portable path.
                                          B1 introduces no specialized native
                                          score; existing component/default
                                          semantics are unchanged.
                                    - [x] **G4b2b2b2 — native score producer:**
                                          share the actual Lid score kernel and
                                          its launch grouping with standalone
                                          scoring; fund defined tails, prove
                                          launch/backing guards and exact
                                          original-helper operand execution.
                                          Authenticated root facts fail closed;
                                          462 native operand scenarios cover
                                          pitches, tiles, row groups and replay.
                                          No component/default consumer yet.
                                    - [ ] **G4b2b2b3 — component and fusion:**
                                          add the opt-in component recipe and
                                          full-mask fusion with observer/owner
                                          guards; enforce separate execution
                                          compatibility on kept/adopt/restore
                                          paths; prove source, selection and lifetime
                                          controls and measure numerical relations.
                                          - [x] **G4b2b2b3a — physical facts:**
                                                declare provider reservation
                                                alignment, bind owned/borrowed
                                                state facts and forward exact
                                                leaf maps through planner passes;
                                                audit placement and binding.
                                                No new component consumer.
                                          - [ ] **G4b2b2b3b — execution compatibility:**
                                                enforce execution identity,
                                                authenticated snapshots and
                                                numerical-domain refusals before
                                                any new recipe becomes reachable.
                                                - [x] **G4b2b2b3b1 — identity values and snapshots:**
                                                      add owned generic identity/domain
                                                      values and one accounted opaque
                                                      snapshot allocation with issuer,
                                                      generation and completion checks;
                                                      preserve legacy raw API behavior.
                                                      No production declaration or
                                                      numerical admission consumer yet.
                                                - [x] **G4b2b2b3b2 — backend declaration:**
                                                      own qualified build/runtime and
                                                      immutable tuning declarations; bind
                                                      actual paid handle/workspace readiness
                                                      through source, plan, snapshot and kept
                                                      identity. Full operation admission is
                                                      separate; no recursive invocation or
                                                      runtime loader-attestation prerequisite.
                                                      Complete through the shared owned-backend
                                                      configuration and its actual mapped DS
                                                      attention-source consumer: main native
                                                      build, 59 host controls and one GPU case
                                                      pass, comparing 320 complete head values
                                                      and 376,832 initialized state bytes
                                                      ([evidence](experiments/m36-dsv4-rounded-consumer/README.md#bounded-mapped-attention-source-execution--2026-10-12)).
                                                      Whole-operation admission, DSpark and
                                                      full-model kept/state/quality/speed gates
                                                      remain separate.
                                                - [ ] **G4b2b2b3b3 — admission enforcement:**
                                                      bind source/plan/kept identities;
                                                      enforce owned whole-operation and
                                                      declared-unit admission on all current,
                                                      ahead, companion and restore paths.
                                                      Present prepared execution identities
                                                      now also bind the shared durable kept
                                                      layout before registration; null/legacy
                                                      sources keep their existing identity.
                                                      Complete operation-shape enumeration
                                                      and broader admission remain open.
                                          - [ ] **G4b2b2b3c — component and fusion:**
                                                add the opt-in rounded-score
                                                component and guarded full-mask
                                                fusion; prove source, selection and lifetime
                                                controls, then evaluate quality and speed.
                                                - [x] **G4b2b2b3c1 — bounded consumer foundation:**
                                                      default-off rounded score, actual-ID
                                                      selection and one full mask; owned
                                                      source/domain host admission controls
                                                      and a paid declared-backend synthetic
                                                      changed-input capture pass. Full source
                                                      GPU/model admission and fusion remain
                                                      open ([evidence](experiments/m36-dsv4-rounded-consumer/README.md)).
                                                - [x] **G4b2b2b3c2 — mapped attention source execution:**
                                                      execute a bounded synthetic mapped
                                                      attention prefix through the actual
                                                      owned source/lower callback; 320 finite
                                                      head values and 376832 state bytes
                                                      match eager/capture across changed
                                                      tokens. Main native build, 59 host
                                                      controls and one GPU case pass.
                                                      Full model admission/fusion remains
                                                      open; no family quality/speed claim.
                  - [ ] **G4b2c — full policy:** output-A and product forms,
                        mixed encodings and default policy; same-checkpoint
                        reference quality, all selected contributions and same-implementation
                        state/verify correctness, without mandatory legacy bits.
            - [ ] **G4c — serving and gate:** bind hash-routing IDs with
                  expert-count checks, compressor inputs and injection
                  features with one drafter slot per target slot; pass the
                  family state, quality, swap and matched 1% controls.
                  Internal Server admission now derives actual composition bounds
                  and shares a paid transient compiler across aliases. At context
                  8448/chunk 256, three serial speculative HTTP requests complete
                  384 outputs. The two-client gate remains FAIL: another 506 MiB
                  of prompt state cannot join the active request, so zero joined
                  waves execute. Fixed-plain and the planned 8K/16 swap are unrun;
                  public/default adoption and the family gates remain open
                  ([service outcome](experiments/m36-dsv4-target/dspark.md#shared-server-admission-and-the-remaining-c2-capacity-wall)).
                  - [x] **G4c1 — full target factory execution screen:**
                        the fixed prepared GGUF executes through ComponentLlmBuild,
                        EngineLlm and ModelRunner with real hash/compressor inputs
                        and the ordinary complete expert closure. Main native build
                        and 307 affected host controls pass; 18 finite complete
                        head rows and nine snapshot-restored rows pass at frontier
                        2568 on Spark B. The unchanged 110 GiB cap and 6 GiB
                        physical margin admit actual setup. Broader same-checkpoint
                        quality, serving, swap and matched speed remain open
                        ([evidence](experiments/m36-dsv4-target/README.md)).
                  - [x] **G4c2 — composed DSpark execution screen:**
                        the exact prepared target/companion execute through shared
                        block drafting, injected features and combined snapshots.
                        Main native build and 308 affected host controls pass;
                        two continuations each complete seven steps, 17 required
                        proposals and one accepted draft, with 18 finite complete
                        target heads and nine restored rows at frontier 2568.
                        Actual paid setup fits the unchanged 110 GiB cap and
                        6 GiB physical margin. Draft-head finiteness, broader same-checkpoint
                        quality, joined serving and family speed/swap gates
                        remain open ([evidence](experiments/m36-dsv4-target/dspark.md)).
                  - [x] **G4c3 — bounded same-checkpoint target scoring:**
                        the real target factory admits a closed Score domain and
                        independently measures its full-head shapes. Shared
                        CorpusScore and current fused llama.cpp score the same
                        256 natural targets; native PPL changes
                        +1.265162% relative to that quantized reference.
                        A separate 2,560-target history reaches selective indexer
                        execution beyond past 2048: native PPL changes +0.803531%
                        overall and −0.034072% on its final 512 targets. Full native
                        build/474 combined host controls (39 DS-related), repository
                        checks and both actual cases pass; a stale wrapper overhead
                        constant is corrected by a source-backed offline audit.
                        Complete-key plan owners and one replacement are measured
                        and paid before payload. Broader sparse histories, DSpark
                        quality, joined serving and family speed/swap remain open
                        ([evidence](experiments/m36-dsv4-target/README.md#same-quantized-checkpoint-scoring)).
            - [ ] **G4d — deletion:** remove DeepSeek V4's M3 runner, plans,
                  graph builders, bindings and serving class after G4c.
- [x] **Migrate Qwen2 FP16 and EXL3 fixtures** (required by the exit criteria).
      The approved fixtures use shared components and selected operation
      lowerings. They are unserved with no speed baseline. Original-model
      corpus observations and same-implementation checkpoint/spill/kept/reload
      controls qualify the bounded fixture scope; historical bridge/Tier C
      comparisons remain recorded diagnostics. Changed virtual addresses,
      OS-process restart/crash recovery and broader serving qualification are
      future capabilities rather than claims of this fixture slice. The final
      integrated legacy deletion build and affected controls pass.
      - [x] **G5a — components and mapping:** the approved FP16 and both
            EXL3 fixtures synthesize specs over the shared catalog; all three
            installed-artifact metadata controls pass. Shared FP16 execution
            matches the frozen fused and unfused 76-token control trajectories
            ([evidence](experiments/m36-qwen2/README.md)); historical heldout
            and lifecycle controls remain G5c's.
      - [x] **G5b — lowering:** implement shared GGML and EXL3 operation
            lowerings; supply and enforce approved EXL3 reconstruction tuning
            facts and supported shapes. All four profiles have selected-source
            and actual execution controls; final deletion integration passes.
            - [x] **G5b1 — selected operations and reconstruction screen:**
                  the shared step program binds approved EXL3 products and
                  pinned tuning facts. All 145 full-vocabulary rows of the
                  4.0-bpw G fixture match a newly sealed native reference
                  bit for bit; 33 focused controls and two admission/input
                  controls passed with zero skips ([evidence](experiments/m36-exl3-selected/README.md)).
                  Final Spark checks passed: 2,554 native, 2,003 CPU and all
                  134 model/timing tests (179 raw GTest arms), with zero failures
                  or skips; quality gates passed. Main built and passed 91
                  focused integration checks and the quality gates. Other
                  profiles, packed execution and lifecycle gates stay open.
            - [x] **G5b2a — full 4.0-bpw G suffix trajectory:** the shared
                  runner matches all 161 full-vocabulary rows of the original
                  145-token prefill and 16 decode steps, bit for bit against a
                  fresh native reference. The original 145-row control and
                  38 focused input, selected-operation and state/lifecycle
                  checks also pass, with zero skips; native build and quality
                  gates pass ([evidence](experiments/m36-exl3-selected/README.md)).
                  Other profiles, packed execution and Tier C remain open.
            - [x] **G5b2b — mixed-rate 4.5-bpw G scoring screen:** the
                  shared runner executes the approved packed 4/5/6/8-bit
                  products and all 24 paired gate/up products on the same
                  authenticated 43 teacher IDs as the original BF16 reference.
                  Both 7-row and 1-row plans select the pinned G implementations;
                  all 42 next-token scores are finite. Eight focused controls
                  and the actual case pass without skips. Main builds and passes
                  all eight checks plus both EXL3 scoring cases
                  ([evidence](experiments/m36-exl3-selected/README.md)). This
                  small primary-reference observation adds no quality threshold,
                  performance, O-profile, Tier C or lifecycle qualification.
            - [x] **G5b2c — 4.0-bpw O GEMV scoring screen:** the shared
                  runner executes the existing O profile on the same 43 teacher
                  IDs and 7/1 chunk schedule as G. Eight queried occupancy facts
                  are rechecked at launch; each actual plan records 120 GEMVs,
                  one head GEMM, 24 paired MGEMMs and 72 bias operations.
                  All 42 full-vocabulary target scores are finite; nine focused
                  controls and the actual case pass without skips. Main builds
                  and passes the same nine controls and scoring case
                  ([evidence](experiments/m36-exl3-selected/README.md)).
                  This short primary-reference observation adds no speed or
                  broad quality claim. Broader O shapes and lifecycle remain open.
            - [x] **G5b2d — mixed-rate 4.5-bpw O scoring screen:** the same
                  shared scorer executes all 169 products/676 source parts on
                  the original 43 teacher IDs and 7/1 chunks. Each actual plan
                  records 14 GEMVs, 107 GEMMs, 24 paired MGEMMs and 72 bias
                  operations, with all eight occupancy facts checked at launch.
                  All 42 full-vocabulary scores are finite; nine focused
                  controls and the actual case pass without skips. Main builds
                  and passes the same nine controls and scoring case
                  ([evidence](experiments/m36-exl3-selected/README.md)).
                  No tuning, kernel or production default changes; broader
                  quality, performance and O lifecycle controls remain open.
      - [x] **G5c — execution controls:** run FP16 and both EXL3 fixtures
            through the shared runner, with original-model quality observations
            and same-implementation state, spill, restore and kept-file controls.
            Historical FP16 bridge / EXL3 Tier C bit comparisons remain evidence
            for their source-era recipes. This fixture slice has no serving,
            process-restart, changed-address or timing claim.
            Grouped primary-reference scoring also records all four EXL3 profiles
            on the existing four 256-token passages (1,020 targets per profile):
            14 focused controls and all 16 actual cases pass without skips,
            with unchanged tuning/arithmetic and no new quality threshold
            ([observations](experiments/m36-exl3-selected/README.md#grouped-original-reference-observations)).
            Main passes the full native build, 14 focused controls and all 16
            document/profile cases, with no failures or skips.
            - [x] **G5c1 — FP16 checkpoint and spill continuation:** both
                  fused and unfused full control trajectories restore checkpoint
                  pages after eight decode writes, replay all eight full-vocabulary
                  rows, and restore the complete used state after spill. Exact
                  bytes, counters, footprints and remaining logits pass
                  ([evidence](experiments/m36-qwen2/README.md)). Historical
                  heldout, repeated/kept-state and EXL3 controls remain open.
            - [x] **G5c2 — historical FP16 heldout trajectories:** both fusion
                  modes match all 577 frozen full-vocabulary rows, including the
                  original 512-row prefill and suffix. Checkpoint rewind at 33,
                  eight replayed rows and spill continuation at 49 also pass.
                  Weight relocation, kept-state/restart and EXL3 controls remain
                  open ([evidence](experiments/m36-qwen2/README.md)).
            - [x] **G5c3 — repeated FP16 execution and weight eviction:** both
                  complete trajectories repeat four times in each fusion mode,
                  clearing state between evaluations. Two full 490-extent weight
                  evictions per case release all backing; reload advances backing
                  generations and preserves content generations and complete
                  checkpoint state before writes. All full-vocabulary rows match.
                  Address relocation, kept-state/restart and EXL3 remain open
                  ([evidence](experiments/m36-qwen2/README.md)).
            - [x] **G5c4 — EXL3 same-implementation checkpoint/spill:** both
                  approved G fixtures create an uninterrupted 161-row reference
                  with their own selected programs, then compare all 161 rows
                  plus eight rewind/replay rows exactly. Checkpoint at 145,
                  spill/restore at 157 and final state at 161 each preserve
                  6,291,456 initialized state bytes, with unchanged weights
                  and recorded plan identities. Nine focused controls and both
                  actual cases pass without skips; main builds and passes the
                  same nine checks and both lifecycle cases
                  ([evidence](experiments/m36-exl3-selected/README.md)).
                  Kept-state/restart, relocation/eviction, broader shapes,
                  Tier C and deletion remain open.
            - [x] **G5c5 — FP16 shared-attention recovery:** select the existing
                  fused attention operation instead of materialized scores for
                  the fixture default. Four fixed natural passages provide 1,020
                  original/baseline/candidate targets; the measured +0.003826683%
                  candidate perplexity change versus native baseline is accepted
                  prospectively as a fixture tradeoff, not a new quality threshold.
                  Same-plan 76/577-row controls preserve exact heads, checkpoint,
                  spill and two full weight reloads. Historical Bridge recipes
                  remain explicit compatibility diagnostics. Final main passes
                  six host and twelve actual controls with no failures/skips
                  ([quality](experiments/m36-original-qwen2/README.md),
                  [state](experiments/m36-qwen2/README.md)). F16 conversion
                  lineage, restart/relocation and serving qualification remain open.
            - [x] **G5c6 — EXL3 O checkpoint/spill continuation:** both
                  approved O profiles use the same uninterrupted 161-row control
                  and paid buffers as G5c4. All 169 compared rows and 6,291,456
                  initialized state bytes match each profile's own reference
                  after checkpoint rewind/replay and spill/restore. Actual
                  occupancy facts and reconstruction/scalar selected operations
                  are recorded with unchanged 292/302 weight extents. Nine
                  focused controls and both actual cases pass on main without skips
                  ([evidence](experiments/m36-exl3-selected/README.md)).
                  Kept-state/restart, eviction/relocation, broader shapes,
                  Tier C, performance and deletion remain open.
            - [x] **G5c7 — EXL3 O kept-file adoption into a fresh owner:** both
                  approved O profiles publish authenticated v2 state at 157,
                  destroy the first native owner after proven retirement, and
                  adopt into a newly constructed empty owner. All 161 compared
                  full heads and 6,291,456 initialized state bytes match each
                  profile's uninterrupted reference. Model/layout/file identity,
                  selected recipes and refusal controls are checked. Thirteen
                  focused controls and both actual cases pass on main without
                  failures or skips
                  ([evidence](experiments/m36-exl3-selected/README.md)).
                  Process restart/crash recovery, serving history, weight
                  eviction/relocation, broader shapes and deletion remain open.
            - [x] **G5c8 — EXL3 four-profile weight reload and kept continuation:**
                  40G/45G/40O/45O evict all 292/302 immutable weight extents
                  between completed requests, reload at unchanged addresses,
                  and retain both selected programs. All 798 resource addresses,
                  content generations and 6,291,456 initialized state bytes
                  survive before continuation. Fresh-owner adoption at 157 then
                  completes all 161 compared rows and final state against each
                  candidate's uninterrupted reference. Fourteen focused controls
                  and all four actual cases pass on main, with no failures or
                  skips ([evidence](experiments/m36-exl3-selected/README.md)).
                  Actual address relocation, process restart/crash, serving
                  history, broader shapes and deletion remain open.
      - [x] **G5d — deletion:** remove the legacy Qwen2 FP16/EXL3 execution
            path after G5c. Both standalone executor families and graph/phase
            builders are removed; generic kernels, immutable source bindings and
            active swap/spec comparison consumers remain. The full native build,
            30 focused host/refusal controls and eight actual affected checks
            pass without failures or skips
            ([evidence](experiments/m36-exl3-selected/README.md#legacy-executor-retirement)).
            - [x] **G5d1 — active FP16 benchmark adapter:** move the swap
                  and speculation harnesses' FP16 comparison to the shared
                  component source/runner. Keep full control/heldout heads,
                  same-implementation repeats and weight reloads; retire the
                  separate graph executor. Explicit host inventory and distinct
                  setup classes replace implicit accounting and repeated
                  compilation. Main passes a full native build and three actual
                  controls (76/577 rows, three evaluations, complete 490-extent
                  weight reload, and pre-construction host refusal), without
                  failures or skips. No serving or speed claim
                  ([evidence](experiments/m36-exl3-selected/README.md#active-fp16-benchmark-adapter)).
- [ ] **Native importer** (D-109): container readers for GGUF,
      safetensors with `config.json`, diffusers pipelines (Qwen-Image's
      components, composed under D-089), MLX and EXL3; architecture mappings
      and the fold catalog for every migrated family; the new artifact
      format version, with every current artifact re-imported; the standard
      admission gate. The Python importers leave the production path.
      *2026-10-09:* format version 1 and its reader, v0 still read
      (D-113, [artifact-format.md](artifact-format.md#version-1)); the
      container readers ([as built](engine-components.md#the-importer)).
      The BF16 norm/A fold primitives and F32/MXFP8 value-head permutation
      are implemented; their source-normalization tables, the rest of the
      fold catalog, the writer, the CLI, the re-import and the gate remain.
      *2026-10-10 (D-114):* the writer and `llmp import` slice (N4)
      also imports straight from the [model archive](model-archive.md) or a
      staged copy, verifying every file against its manifest and reading
      each needed byte once, and records an `import` event in its catalog;
      the interim archive tools landed. Native `llmp install` (fetching
      into the archive with libcurl) stays in M5's importer item.

      - [x] **N3a — bounded transform primitives:** internal BF16-to-F32
            `norm.plus_one` and `ssm.a_neg_exp` scalar folds, with atomic
            refusal and integer nearest-even output encoding; complete
            F32/MXFP8 `gdn.value_heads_tiled` permutations with independent
            per-part cursors and constant-size metadata. Existing fold
            identities are unchanged; source normalization and checkpoint
            import remain N3b/N4.
      - [ ] **N3b — source normalization and remaining folds:** declarative
            mappings and transforms for every migrated family/container.
            - [x] **N3b1 — owned GGUF Gemma 2/3/4 target normalization:**
                  implemented and independently reviewed with exhaustive
                  key/tensor classification, tied heads, per-layer lists,
                  dense/MoE roles, byte-preserving routes and common
                  component validation. Focused native checks passed:
                  70 host controls and four real target profiles (Gemma 2
                  2B, Gemma 3 4B QAT, Gemma 4 26B-A4B and 31B), no skips.
                  Final Spark checks passed: 2,516 native, 1,971 CPU and
                  all 114 model tests, with no skips. Main build and integration checks passed.
                  Gemma 2/3's GGUF norm fold is asserted already applied;
                  Gemma 4 adds no fold. Assistants, Qwen/DeepSeek, other
                  containers and remaining transforms stay open in N3b.
            - [x] **N3b2a — source-preserving BF16 parameter operands:**
                  shared HC, QSA, PLE, MTP and GDN components accept their
                  declared small BF16 parameters and compute with transient
                  F32 values. GDN's explicit logarithmic-A option computes
                  its negative exponential without expanding resident weights.
                  Independent challenge and 30 focused native controls passed
                  (24 component and six lowering cases), with zero failures or
                  skips; all ten historical F32 graph digests match the parent.
                  Final Spark checks passed 2,561 native and 2,010 CPU tests
                  with zero failures or skips; format, boundary, header and
                  REUSE gates passed. Main built and passed 30
                  integration checks and the quality gates.
                  MLX source normalization, norm/head-order authentication,
                  publication and runtime admission remain open.
            - [x] **N3b2b — explicit zero-centered component norms:**
                  optional norm offsets for sigmoid HC, QSA/indexer, PLE
                  and MTP compute source-plus-one in paid F32 workspace,
                  preserving F32/BF16 resident bytes and default graph order.
                  Contradictory applied folds, Sinkhorn and GDN are refused.
                  Independent review, 36 focused native controls and all ten
                  historical F32 graph digests passed. Isolated final checks
                  passed 2,567 native and 2,016 CPU cases with zero failures
                  or skips, plus quality gates. Final main checks passed
                  2,582 native and 2,027 CPU cases, 36 integration controls,
                  and 135 model/timing cases (182 raw GoogleTest runs), all
                  with zero failures or skips. Source, recipe and model-binary
                  guards and the quality gates passed.
                  This component foundation does not establish source
                  normalization, converter/head-order authority, publication
                  or runtime admission.
            - [x] **N3b2c — encoding-preserving BF16 head permutation:**
                  generic dense BF16 rank-one/two support in the
                  existing bounded grouped-to-tiled copy plan. Synthetic full
                  payload controls cover vectors, prefixed rows, reshaped
                  convolution rows and columns, with byte coverage and refusal
                  guards. Independent review and final main checks passed:
                  2,576 native, 2,021 CPU and 135 model/timing cases
                  (182 raw GoogleTest runs), with zero failures or skips.
                  Format, boundary, header and REUSE gates passed;
                  source, recipe and selected model-binary guards passed.
                  No source head-order authority, checkpoint normalization,
                  publication or runtime admission is established by this foundation.
            - [x] **N3b2d — bounded streaming part-route execution:**
                  checked copies, per-part head permutation and historical
                  scalar-fold streams with caller-funded workspace and exact
                  sink-acknowledged coverage. Existing writer copy routes
                  use the executor. Independent review, 44 focused controls,
                  2,581 isolated native and 2,026 CPU tests and quality gates
                  passed with no failures or skips. Combined checks passed
                  2,587 native and 2,032 CPU cases; main build, 44 integration
                  controls and all 135 model/timing cases (182 raw GoogleTest
                  runs) passed with zero failures or skips. Source, recipe,
                  model-binary guards and quality gates passed. Source mappings, gathers,
                  head-order authority, normalization and new publication/
                  runtime admission remain open.
            - [x] **N3b2e — owned indexed multipart assembly:**
                  prepaid indexed source ownership, complete dispositions,
                  numeric expert/row gathers, shared tensor scales and
                  encoding-preserving reshape/permutation routes. Eight
                  full-payload and refusal controls, 2,595 native and 2,040
                  CPU checks and quality gates passed with no failures or
                  skips. Main build, eight integration controls and all
                  135 model/timing cases (182 raw GoogleTest runs) passed
                  with zero failures or skips. Source, recipe, model-binary
                  guards and quality gates passed. Source mapping authority,
                  byte authentication, publication and admission remain separate.
            - [x] **N3b2f — closed native ModelOpt target normalization:**
                  owned indexed multipart assembly and a finite source mapping
                  for the pinned Qwen3.8 ModelOpt target. Preserve BF16,
                  split MXFP8/NVFP4 and embedded expert scales; derive numeric
                  expert/ngram gathers and byte-preserving head permutations.
                  Validate all source roles, excluded auxiliary entries,
                  configuration and the 280-byte semantic input against a
                  complete component spec under prepaid metadata bounds.
                  Independent challenge, 26 main focused controls, confined
                  actual-source metadata normalization, 2,599 native and
                  2,044 CPU checks, all 136 model/timing cases (183 raw
                  GoogleTest runs) and quality gates passed with zero failures
                  or skips (2026-10-12). Publication, MTP/vision import, runtime admission
                  and migration gates remain separate.
            - [x] **N3b2g — owned GGUF Qwen3.8 target normalization:**
                  preserve all 1,224 source tensors in the supported target
                  profile and seven encodings, source-owned n-gram lists and
                  existing GGUF fold assertions. Validate complete component
                  roles and prepaid construction; check common writer placement
                  for row tables, mixing and all expert slices. Registering the
                  mapping also enables the ordinary CLI writer path. Full native build, 103 affected host controls and one
                  actual-header-only check pass on spark-b with no failures or
                  skips (2026-10-12). Subsequent N4b8 completes ordinary GGUF
                  target publication and stored-target admission; family
                  migration and other source domains remain open.
                  [Scope and source evidence](experiments/m36-qwen-gguf-normalization/README.md).
      - [ ] **N4 — native v1 writer, CLI and archive input:** deterministic
            publication, embedded specs, verified source streaming and
            catalog import events.
            - [x] **N4a — closed copy-only writer and CLI:** Gemma target
                  plans, embedded specs, exact archived/staged variant closure,
                  authenticated bounded streaming and independent import events.
                  79 focused Spark host tests and repeat independent challenge
                  pass (2026-10-11); final combined checks passed 2,530 native,
                  1,985 CPU, all 114 model tests and quality gates, with no skips. Actual Gemma 2 Q8_0 publication
                  and reuse verified all 288 tensors, 2,778,448,896 payload bytes
                  and 1,340 chunks with the same artifact ID. Main build, 43 integration checks and quality gates passed.
            - [x] **N4b1 — Gemma 2 native v1 runtime admission:**
                  shared admission reads the stored spec and native tensor
                  formats, checks the approved source/profile and preserves
                  declared folds, tied heads and policy. Registration and the
                  component build share it; v0 remains supported. Focused
                  host checks (31) and the actual v0/v1 full-logit, initialized-state
                  and spill-continuation comparison passed with zero skips.
                  Final regression passed 2,544 native, 1,999 CPU and all 131 model/timing
                  tests with zero skips; quality gates passed. Main built and passed
                  31 integration checks and quality gates. This historical
                  qualification covers Gemma 2; the three-target extension follows.
            - [x] **N4b2 — approved Gemma 3/4 native target admission:**
                  exact source and native tensor inventories for
                  Gemma 3 4B QAT, Gemma 4 26B-A4B and 31B, stored-spec equality,
                  profile-gated policy and native expert slab geometry. Host
                  controls (44) and all three complete copy-only publications
                  passed. Actual v0/v1 serving passed for each target: 24 full
                  vocabulary rows, tokens/history, initialized state and
                  spill/restore continuation, with three joined two-owner decode
                  waves per arm. Gemma 4 keeps prompts-alone and its defaults.
                  Final Spark checks passed 2,567 native, 2,012 CPU and all 135
                  model/timing tests (182 raw GoogleTest cases), with zero failures
                  or skips; quality gates passed. Main build, 74 integration controls,
                  all three actual comparisons and quality gates passed;
                  unsupported identities and drafter compositions stay closed.
            - [x] **N4b3 — indexed-part publication foundation:**
                  one native writer path for owned copy-only and closed ModelOpt
                  plans, whole-member routing, multipart expert scatter and
                  byte-preserving permutations. Prepaid metadata bounds,
                  complete binding/output coverage and same-stream semantic
                  reauthentication precede applied-fold receipts and verified
                  durable publication. Independent challenge, 21 main integration
                  controls, 2,601 native and 2,046 CPU checks, all 137
                  model/timing cases (184 raw GoogleTest runs) and quality
                  gates passed with zero failures or skips (2026-10-12).
                  Actual publication authenticated all 38 source files
                  (105,872,821,791 bytes), preserved 103,797,070,596 weight
                  bytes and verified all 25 files and 66,261 chunks.
                  All 468 permuted parts (2,153,140,992 bytes) matched
                  an independent destination-order oracle. The confined
                  8 GiB, zero-swap run completed in 31 minutes without OOM.
                  CLI source expansion, MTP/vision import, runtime admission
                  and migration qualification remain separate.
            - [x] **N4b4 — closed native ModelOpt CLI dispatch:**
                  route the existing import command's exact supported
                  safetensors architecture/quantization combination through
                  the closed normalizer and common writer/archive flow;
                  unsupported source contracts refuse without fallback.
                  Preserve Gemma dispatch, flags and metadata accounting.
                  Independent challenge and main checks passed: 22 import
                  integration controls, 2,602 native and 2,047 CPU checks,
                  all 137 model/timing cases (184 raw GoogleTest runs) and
                  quality gates, with zero failures or skips (2026-10-12).
                  Confined actual CLI checks passed the 1 MiB normalization
                  and 512 MiB writer funding refusals without store creation,
                  followed by funded ordinary-source publication using the
                  default 4 GiB metadata ceiling. All 38 complete source hashes,
                  canonical bindings and 468 full permuted parts matched the
                  pinned target under the 8 GiB, zero-swap process/cache cap.
                  Runtime admission, MTP/vision import and migration
                  qualification remain separate.
            - [x] **N4b5 — multipart native row execution foundation:**
                  bind every row-addressable format part separately and
                  require identical lookup-order slot maps before effects;
                  execute split NVFP4 GetRows through the existing selected
                  decoder and shared ModelRunner. Check a bounded synthetic
                  table, changed/repeated/reordered IDs, refusals and completed
                  capture replay. Main native build, 117 focused host controls
                  and both tiny GPU cases pass ([evidence](experiments/m36-decode-templates/README.md#native-split-row-foundation--2026-10-12)).
                  Products, expert worklists and native ModelOpt target
                  admission remain separate.
            - [x] **N4b6a — completed-route and multipart descriptor helpers:**
                  validate the complete route bijection and preserve original
                  routed slots, including repeated expert contributions; bind
                  all expert descriptors from exact part extents and padded
                  pitches without expanding source encodings. Caller-sized
                  storage and refusal-before-output-write controls pass:
                  eight focused host cases and full native main build on Spark,
                  with independent source challenge clean
                  ([evidence](experiments/m36-native-grouped-foundation/README.md)).
            - [x] **N4b6b — owned grouped operation execution:** connect those
                  helpers to an actual selected-expert closure with current
                  leases, bounded preparation and completed capture replay.
                  Both bounded synthetic GPU cases pass on the isolated source
                  and main (288 output values each), with full native main
                  build and all eight helper controls passing.
                  A shared ModelRunner mid-step routing/admission continuation
                  and native ModelOpt target admission remain open separately.
            - [x] **N4b6c — base-lease-preserving selected closure:** expose
                  supplemental admission through the ordinary scheduler while
                  the original request lease remains held. Retain the complete
                  payload until both device work and scoped page-ins positively
                  retire; unknown completion prevents continuation and teardown.
                  Full native main build and 145 focused host controls pass,
                  including six new malformed-input, funding, cancellation and
                  retirement cases. Production routed ModelRunner integration
                  and native ModelOpt target admission remain separate
                  ([evidence](experiments/m36-native-grouped-foundation/README.md#supplemental-selected-closure--2026-10-12)).
            - [x] **N4b6d — one shared routed ModelRunner step:** connect a
                  completed routing prefix, checked native split expert bindings
                  and supplemental selected closure to the original step's
                  continuation. Preserve the base lease, whole-plan placement,
                  fresh routing on capture replay and complete ownership on
                  uncertain completion. Full native main build, 101 routed host
                  and 34 fusion/validation controls, and all three GPU cases pass
                  without skips. The bounded real ComponentStepSource fixture
                  checks 288 complete outputs across three steps, selected-only
                  residency and state frontier 9; this is not full-model quality
                  or performance qualification
                  ([evidence](experiments/m36-native-grouped-foundation/README.md#shared-routed-step--2026-10-12)).
            - [x] **N4b6e — ordered routed groups:** extend the same whole-plan
                  placement and complete owner through multiple ordered router
                  boundaries, with fresh selected closure admission per group
                  and final-only output/carry publication. Full native main build,
                  104 routed and 34 fusion/validation host controls, and all four
                  GPU cases pass without skips. The two-group fixture checks
                  288 outputs, 36 actual routed IDs, nine completed segments and
                  state frontier 9 across eager/captured/replayed steps. Family
                  admission and full-model quality/performance remain separate
                  ([evidence](experiments/m36-native-grouped-foundation/README.md#ordered-routed-groups--2026-10-12)).
            - [x] **N4b6f — allocation-free page ordinal lookup:** derive selected
                  multipart page ordinals from checked current device reservation
                  geometry, retaining the original fallback and atomic output
                  validation. Main native build/157 host controls and four routed
                  GPU cases pass. A bounded same-host A/B/B/A CPU screen measures
                  large-inventory selected-span savings with explicit small dense
                  and host-fallback regressions; no model-speed claim or new index
                  ([evidence](experiments/m36-native-grouped-foundation/README.md#ordinal-page-lookup-cpu-screen--2026-10-12)).
            - [x] **N4b7 — paid consumable stored-spec bindings:** one prepaid
                  spec/source and the shared role/owner visitor replace redundant
                  binding-discovery graphs in admitted stored Gemma targets;
                  actual setup measurement remains. Main native build and 191
                  affected host controls, four existing published Gemma execution
                  cases, and the mapped DeepSeek source case pass. Admission,
                  native multipart expert placement and shared row-group ownership
                  are not widened by this foundation. ModelOpt/MLX serving and
                  family qualification remain open ([scope and checks](engine-components.md)).
            - [x] **N4b8 — ordinary Qwen GGUF publication and stored target:**
                  publish the complete approved three-shard UD-IQ3_XXS source
                  through the native CLI; retain exclusive embedding/PLE groups
                  and execute the admitted stored spec. Actual v0/v1 comparison
                  passes 36 complete finite head rows, six initialized-state
                  observations and two spill/zero-compute restores with
                  continuation. Main native/476 host controls pass. Public
                  defaults, stored drafter, ModelOpt/MLX serving, family speed
                  gates and full N5 remain open
                  ([evidence](experiments/m36-numerical-audit/qwen-joined-mtp.md#native-gguf-publication-and-target-serving--2026-10-13)).
            - [x] **N4b9 — native DeepSeek GGUF target publication and serving:**
                  normalize the complete approved three-shard UD-Q2_K_XL target
                  (1,328 tensors, 129 expert arrays, 43 layers), publish through
                  the ordinary CLI and execute its validated stored spec.
                  Actual v0/v1 comparison passes 36 full finite head rows,
                  six initialized-state observations and two spill/zero-compute
                  restores with continuation at 2,560 prompt tokens. Final native
                  build/66 affected controls pass; prior full normalization,
                  admission and writer controls remain separately recorded.
                  Other formats/folds, stored DSpark, public defaults, family
                  matched 1% gates and full N5 remain open
                  ([evidence](experiments/m36-dsv4-target/README.md#native-gguf-publication-and-stored-target--2026-10-13)).
            The remaining fold catalog, normalized publication domains and
            runtime import admission remain open in N4; N5's all-artifact
            re-import and serving qualification remain separate.
      - [ ] **N5 — re-import and admission gates:** every current artifact,
            lossless equivalence and qualified family execution.

      N5's all-artifact re-import and family gates follow G3, G4, G5,
      Qwen-Image and N3/N4;
      K1's lossy-state work covers every migrated LLM, including Qwen2 once
      G5 has supplied its shared execution path.
- [ ] **Qwen-Image-2.1 onto the shared blocks**, with qualified numerical choices.
      - [x] **Q1a — channel-normalization component:** stateless shared
            channel L2 normalization, learned scaling and optional SiLU;
            independent scalar and primitive-graph controls, F32 arithmetic
            without a legacy rounding gate. Main native build and all 96
            affected host cases pass on Spark B (2026-10-12), including four
            new channel-normalization controls. Image GPU lowering,
            pipeline wiring and family qualification remain open.
      - [x] **Q1b — adaptive normalization and gated residual components:**
            shared scale-only LayerNorm and tanh-gated residual with an explicit
            saved pre-branch carry and matching per-owner conditioning rows.
            CPU/IR `BuildComponent` composition and refusal controls pass review;
            main native build and all 99 affected host cases pass on Spark B
            (2026-10-12). Whole-stage conditioning, image
            GPU lowering and pipeline qualification remain open. No mandatory
            host row expansion or legacy intermediate-rounding gate.
      - [x] **Q1c — typed spatial graph composition:** the shared
            component-to-plan path consumes explicit image geometry, nearest2
            upsampling and channel-changing `conv.spatial` fragments. Source
            weight encodings and owner-specific dimensions remain intact.
            Main native build and all 239 affected host cases pass on Spark B
            (2026-10-12), including three new spatial and 45 planner controls;
            source quality checks pass. No new token phase or production
            executor. Whole-pipeline wiring and image qualification remain open.
      - [x] **Q3a — shared spatial execution:** typed graph ports and optional
            token metadata use the existing planner, `StepProgram`, `StepRuns`
            and owned completion/capture path. The full main Spark A native build,
            129 affected host controls and one actual F32 VAE-span screen pass
            (2026-10-12); source quality checks pass. The actual decoder
            prefix and a reduced nearest2/resample span cover 4,737,024 finite
            outputs and two eager/capture/replay paths each. This physical
            convolution implementation supports F32 source weights; BF16 remains
            unsupported here without changing the catalog's source formats.
            Flow/timestep composition, broader VAE qualification, original-image
            quality, comparative speed and serving remain open
            ([bounded evidence](experiments/m36-shared-spatial-execution/README.md)).
      - [x] **Q3b — complete first-frame VAE composition:** reuse shared
            spatial components, backward value links, source-owned roles and
            the existing program/completion path for the stored F32 decoder.
            A 2x2 latent produces all 32x32x4 decoder outputs; a small actual
            prefix has a separate source-reference control. Focused native
            compilation and all 52 host controls pass, followed by the full
            main Spark B native build and 227 host cases across 17 targets.
            The actual Spark B case passes all 4,096 finite decoder outputs,
            4,608 local source-reference outputs and two eager/capture/replay
            paths each (2026-10-12). No full-decoder scalar quality oracle,
            original-image quality, comparative-speed or serving claim
            ([scope and evidence](experiments/m36-shared-spatial-execution/README.md#complete-first-frame-decoder)).
      - [x] **Q3c — complete timestep conditioner:** shared source-typed
            projections and bounded exported values retain the prior single-output
            default. Four actual BF16 matrices produce both modulation and final
            scale for two unequal conditioning rows, without a mandatory legacy
            intermediate-rounding policy. Full main Spark B native build and
            443 host cases across 25 targets pass, including the four-target
            55-case affected subset. One actual case checks all 40,960 output
            values at five completed steps (204,800 finite values), both changed
            first-row branches, unchanged zero-time rows and retained capture
            replay (2026-10-12). Full flow, image quality against a same-source
            reference, comparative speed and serving remain open
            ([bounded evidence](experiments/m36-shared-spatial-execution/README.md#complete-timestep-conditioner)).
      - [x] **Q3d — conditioned transformer block:** complete conditioner and
            one source-shaped block share typed ports, backward links and saved
            residual carries. Existing NORMAL rotations implement three axes;
            explicit F32 attention applies causal-text/all-image masking. Scoped
            Spark B compilation and all 63 affected host controls pass, including
            a reduced independent CPU oracle. The actual 13-BF16-part block checks
            229,376 finite values across seven eager/captured/replayed steps,
            changed/restored time and image inputs, masked text and zero-time
            isolation; the existing conditioner regression also passes.
            Full main Spark B native build and all 452 host cases across 25
            targets pass; format, boundaries, headers and REUSE checks pass.
            Broader shapes, same-source reference quality, speed and serving
            remain open; the complete 32-block forward follows in Q3e
            ([bounded evidence](experiments/m36-shared-spatial-execution/README.md#conditioned-transformer-block)).
      - [x] **Q3e — complete denoiser forward and Euler step:** source-owned
            text/image projections and all 32 shared blocks feed final adaptive
            normalization, noise projection and the original-latent Euler carry.
            Generic learned RMS, row slicing/concatenation and scaled residuals
            preserve BF16 source encodings and existing fusions. Full main Spark B
            native build and 455 host controls across 25 targets pass, including
            the reduced independent two-owner complete-flow reference and complete
            source-shaped physical planning. All three actual cases pass: the
            complete flow checks 3,584 finite values and 1,792 Euler contributions
            across seven eager/captured/replayed steps with changed/restored inputs;
            the conditioner and single-block regressions also pass. Broader
            geometries, same-source reference quality, repeated sampling,
            text encoder/VAE integration and serving remain separate. Final
            formatting, boundaries, headers and REUSE checks pass.
            ([bounded evidence](experiments/m36-shared-spatial-execution/README.md#complete-denoiser-forward-and-euler-step)).
      - [x] **Q3f — complete shared generation core:** the real 36-layer text
            encoder, two full 32-block flow/Euler steps and complete first-frame
            VAE execute through `EngineImage` at 1024×1024. `ProgramLifecycle`
            shares paid measurement, binding, coverage, plan accounting and
            retained `StepRuns` with `ModelRunner`; token state and cache keys
            remain unchanged. Typed device handoffs complete under the current
            stage lease, and unknown completion retains the complete owner.
            Full main Spark B native build and all 197 affected host controls
            across eight targets pass. The actual case completes two generations
            and eight stages, checks all 4,194,304 finite decoded values per
            generation and reproduces its own first-flow and pixel digests.
            The measured 41,224,157,952-byte envelope fits the unchanged 48GiB
            node cap. Same-source image quality, broader prompts/schedules,
            serving/swap qualification and the matched 1% migration gate remain
            open; legacy serving is retained
            ([bounded evidence](experiments/m36-shared-spatial-execution/README.md#complete-shared-image-generation)).
      - [x] **Q3g — two-step native migration agreement screen:** compare
            completed legacy/shared images from the same checkpoint, prompt,
            noise and 1024×1024 schedule. Two generations per arm retain their
            own pins; PSNR46.438 dB / SSIM0.99873 pass the fixed32 dB / 0.98
            screen. The existing native pipeline is a migration baseline,
            not an independent upstream reference. Descriptive quality-run
            counters expose a shared remainder cost; no matched speed result
            or default adoption is claimed. Independent upstream quality,
            broader schedules and the 40-step/1% family gate remain open
            ([report](experiments/m36-shared-spatial-execution/README.md#two-step-native-migration-quality-screen)).
      - [x] **Q3h — shared pointwise convolution reuse:** supported F32 1×1
            spatial components emit existing layout copies and matrix products;
            padded/BF16/3×3 cases preserve their existing path. A source-backed
            64×64 pointwise comparison passes its F64 enclosure and reduces
            five warm completed-job median from249.584 to73.552µs (70.53%).
            Complete1024 two-step output passes retained native comparison at
            PSNR46.477 dB / SSIM0.99873. This is bounded reuse evidence, not
            independent upstream quality, isolated decoder timing or the1%
            family gate ([report](experiments/m36-shared-spatial-execution/README.md#shared-pointwise-convolution-reuse)).
      - [x] **Q3i — shared tiled F32 convolution reuse:** existing F32 im2col,
            cuBLAS, transpose and pitched copies initially qualify at64MiB;
            a1GiB experiment reduces calls7,706→870 and prepaid capture
            metadata106.8125MiB, but adds1,102,249,984 admission bytes without
            meaningful observed latency or measuredCPU/BW gain. Keep64MiB
            default. Quality remains46.495dB/.99873; decoder2.752s versus
            9.653s prefix directs next optimization toward text/flow.
            Exact measured sm_121 physical geometries become ordinary shared
            candidates; other geometries keep direct fallback. Representative
            completed-job median falls90.64%; the full1024 two-step sample
            passes retained native agreement at46.495dB /0.99873. Output-disabled
            warm complete generation is12.459s versus53.164s, a descriptive
            76.57% reduction, not the matched1% gate. Same-source upstream
            quality, serving/swap and broader schedules remain open
            ([report](experiments/m36-shared-spatial-execution/README.md#shared-tiled-f32-convolution-reuse)).
      - [x] **Packed masked-attention candidate:** shared Attention emission
            retains dense fallback and adds explicitly selected F16 transient
            packing. Empty-row SoftMax/MMA fixes preserve finite-row arithmetic.
            Representative1,057-row completed median falls41.14%; seven-row
            control regresses8.35%, so no universal shape policy follows.
            Complete1024/two-step native agreement passes48.229dB/.99877.
            Wide finite V overflows F16: the approximation remains opt-in,
            with unrestricted dense default. Independent upstream quality and
            family gates stay open
            ([report](experiments/m36-shared-spatial-execution/README.md#opt-in-packed-masked-attention)).
      - [x] **Q3j1 — request image geometry and schedules:** shared composition
            accepts paired source width/height or the existing square setting,
            actual prompt/drop/noise and configured steps. Rectangular VAE
            mappings, independent odd-axis positions and checked payment replace
            incidental diagnostic ceilings. One Euler step explicitly goes from
            source sigma 1 to 0; the existing multi-step scalar contract remains.
            Native 32 controls pass. A 2048×512 request with 35 native prompt
            tokens, actual drop 14 and one Euler step completes twice through
            six stages with all 4,194,304 pixel values finite each time and own
            repeat digests ([report](experiments/m36-shared-spatial-execution/README.md#request-driven-image-composition)).
      - [x] **Q3j2a — completed-request content and counts:** replace paid
            prompt/noise/schedule payloads only after successful Finish and
            release of the request lease. Same-layout requests retain model,
            weights, backend, ports and program/capture identity; active or
            unknown completion and incompatible payloads refuse atomically.
            Serial callers release prior request content without a count cap
            or accumulated images. Thirty-five focused host controls pass,
            including 33 replacements with stable captures and failure-atomic
            refusal. One retained native model completes three requests with
            changed/restored prompt and noise, 1/2/1 Euler steps, two released
            request owners and warmed resource reuse; all 12,582,912 decoded
            values are finite
            ([report](experiments/m36-shared-spatial-execution/README.md#completed-request-content-and-count-reuse)).
      - [x] **Q3j2b — geometry and resource growth:** paid backing grows
            behind stable shared workspace addresses and extent IDs; geometry
            ports, staging and plans retire after positive completion. The
            existing Server adapter retains a validated prepared source and
            Gemma state through four images: 512×512 → 1024×512 → repeat →
            512×512, with six activation calls and two retired geometry owners.
            Activation backing grows from 1,132,462,080 to 2,264,924,160 bytes;
            own repeated/restored pixels and retained Gemma state pass. The
            combined native build passes 474 host controls; the placement-pin
            correction passes ten targeted controls and the actual retry.
            Ordinary installed-source preparation now consumes the same Server
            route under the internal shared-engine selector, retaining paid
            sources while four requests change prompt, geometry and 1/2/2/1
            steps. All 14 image stages, six activation calls and retained Gemma
            state pass. The combined native build passes 476 unique controls,
            including 144 current image controls. One ordinary 1024²/40-step
            request passes retained same-checkpoint diffusers quality at
            37.059 dB PSNR/.99041 luma SSIM (floors 32/.98). Public defaults,
            legacy retirement and matched 1% family qualification remain open
            ([report](experiments/m36-shared-spatial-execution/README.md#ordinary-installed-source-image-service)).
- [ ] **Structural techniques land once**: each retired transfer technique
      in its component, pattern or lifecycle piece, adopted where qualified
      comparisons improve speed, quality, reuse, simplicity or resource cost.
- [ ] **Persisted state and settings:**
      - [x] **Shared kept-state identity:** the bound spec and every stage's
            layout derive `llmp-state-v2`. When the source provides a prepared
            execution identity, setup folds it into that same opaque layout
            before registration, so durable records and slot restore/adoption
            refuse another declared configuration. Legacy/null sources retain
            their original layouts; ordinary serving does not implicitly gain
            a required domain. Record fields/version stay unchanged (D-105).
            Main Spark A native build and all 229 host controls across nine
            targets pass, including current-identity fresh-owner continuation
            and changed-identity refusal before adoption/restore effects.
      - [x] **Spec-bound calibration:** key shared serving measurements by
            the actual bound spec, preserving artifact storage paths, default
            settings and legacy-runner handling (D-103). Main native build,
            321 host controls across 15 affected targets and six existing
            Gemma serving/assistant model cases pass without failures or skips.
            The companion metadata parser has a separate 384 MiB startup
            envelope; this is not a quality or performance qualification.
      - [ ] **Component-derived settings and legacy retirement:** replace
            architecture-based applicability and defaults as the remaining
            serving paths migrate; retain their qualification limits.

### Exit criteria as registered

- **One engine:** every current model (Gemma2, Gemma3, Gemma4 26B-A4B and
  31B with assistants, Qwen3.8 native with MTP and GGUF, DeepSeek V4 Flash
  with DSpark, Qwen-Image-2.1, and the Qwen2 FP16/EXL3 fixtures) runs from
  a spec on the shared components. No family-specific runner, plan, graph
  builder, binding or serving class remains, and serving dispatches by
  pipeline kind, not architecture name. Every catalog component has a CPU
  reference, and fake-backend tests compare its fragment with it.
- **Speed:** against its recorded baseline, each LLM's prefill and
  decode, solo and at its qualified cohorts, and Qwen-Image's generation
  latency, are no more than 1% slower at worst under the fixed protocol,
  and preferably at least as fast.
- **Quality and state:** prospective numerical improvements are judged
  against a reference implementation of the same quantized checkpoint
  (owner clarification, 2026-10-12), with frozen reference provenance,
  coverage and bounds, NLL/perplexity and existing behavioral controls.
  Original-model evidence is required only for claims of closer agreement
  with that original; missing original weights do not block this gate;
  historical completed gates are not reclassified. Exact same-implementation
  state, continuation, spill, restore and kept-state controls pass; the M3
  swap table and the Gemma
  near-8K switch controls pass within their bounds. Report peak memory;
  a worthwhile speed or quality benefit may use more than the baseline
  when the complete workload safely fits the hardware (owner, 2026-10-13).
- **Formats:** Qwen3.8's NVFP4 and MXFP8 tensors, read from its
  artifacts, run every product shape on `sm_86` through on-chip decode
  within tolerance of the CPU references; no resident weight
  is larger than its source encoding. DeepSeek V4's mixed-type GGUF runs
  its fused patterns across differently encoded operands where they gain,
  checked against the CPU references.
- **State encodings:** a lossy KV encoding runs as an opt-in per-alias mode
  on every migrated LLM, off by default, with its long-context quality
  against F16 measured and reported (an opt-in mode carries no parity
  claim).
- **Import:** the native importer imports every migrated model from its
  source checkpoint (archived or staged, verified against its archive
  manifest, D-114) into the new format version, and each passes the
  admission gate and the gates above; the MLX checkpoint of a migrated
  architecture, `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@dadefa80` (chosen
  with the owner, 2026-10-09), imports and passes the admission
  gate; the
  importer and the new artifact format pass their adversarial challenge
  (workflow.md's heavy path).
- **Partitioning:** the planner's split choices and their equivalence pass
  on the fake backend for TP, PP, EP and a mixed TP×PP layout.
- The Spark check set passes; skipped tiers are named.
