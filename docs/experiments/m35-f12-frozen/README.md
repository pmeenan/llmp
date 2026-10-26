<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# F12: the model tests that failed at 2e38631c, and the env-gated skips

Slice F12 of the M3.5 review fixes (2026-10-26) set out to leave a model
suite with no known failures and no silent skips. Each failure was bisected
to a commit on `spark`. The bisection used full builds of exported trees in
`~/src/llmp-bisect` (about 4 minutes each), with the frozen tests or one
swap-table pair run at each step. Each failure was then classified as one of:

- **(a)** an accepted numerics change: re-recorded with evidence;
- **(b)** a regression: fixed, with a test;
- **(c)** a test or fixture problem: the test fixed.

All runs were on `spark` (GB10, driver 580.178.04, CUDA 13.0, SDK
`aarch64-2c4eb5b2f68f3fa5`). Raw logs stay outside Git.

## Outcomes

| Test | Cause (commit) | Class | Outcome |
| --- | --- | --- | --- |
| Gemma2 refused at admission in the swap table and speed gate ("stored stage folds disagree with preprocessing receipts") | `tools/speed_gate.py` and `k1_quality.py` hard-coded v1 IDs. Those IDs predate 84ab300c, which republished Gemma2, Gemma3 and Qwen GGUF with preprocessing receipts. The later republications (DeepSeek 1d99893c, drafters 381eec18, Gemma4 6b465510) were missed as well. The old Gemma2 import carries no receipts. | (c) | The tools read the pinned `tests/unit/data/v1_frozen/v1-ids.json` by name. Tool tests assert this. |
| `GemmaEngineQ8CacheGpu` ("numerical semantic identity differs") | 3cf3118c compares the KV-encoded spec's digest with the admitted spec's. | (b) | `AdmittedNumerics` accepts the admitted spec under the alias Q8_0 KV encoding (`WithoutKvCacheEncoding`). New test: `ComponentAdmissionTest.AliasQ8KvEncodingKeepsTheAdmittedNumericalIdentity`. |
| `Qwen38EngineServingGpu.PublishedGgufMatchesFrozenHeadsStateAndSpill` | 4533033c: GB10's upstream scalar VecQ geometry for every encoding | (a) | Re-recorded. Tokens and history are unchanged; heads, logits and state moved. |
| `Dsv4EngineTargetGpu.Published{Gguf,Dspark,Iq2xxs}MatchesFrozenHeadsStateAndSpill` | 425ebf23: GB10 scalar Q8_0 one-token schedule | (a) | Re-recorded. DSpark keeps its tokens and draft counts; the target and IQ2_XXS loops take other tokens. |
| `Dsv4EngineTarget.FixedPromptHeadAndGreedyTokensMatchTheReferences` | 2b3452de: GB10 multi-column 2..8 schedule. It moved row 7's competitor logit to 2.05 from the M3 runner's (bound 2.0). | (a) | Re-anchored on the same checkpoint's reference implementation, with the same bounds (below). |
| `Gemma4ServingGpu.LiteralTeacherForcingAndChatRefusalsUseTheTargetRoute` (Gemma26, Gemma31) | c7b1b142 enabled tool rendering; the test still expected a refusal. | (c) | The test asserts that the tool's declaration is rendered. |
| `Qwen38QuantHeadTest` (both) | 4533033c: Q4_1 scalar geometry | (a) | The fixture is re-recorded as `operand-fixture-v2` (below). The test resolves and authenticates it by default. |
| Swap table: DeepSeek → Qwen3.8 B→A at 8K, "15 of 16 continued logits differ" (was F11) | dae227dd removed the drafter-progress ledger. A diagnostic snapshot restore then forgot which positions the drafter's per-row feature tap held, so the continuation drafted 17 tokens, not 19, and its verifies' widths, and so their bits, changed. Tokens were equal. | (b) | Snapshots record the tap bookkeeping beside the state and restore it (`ModelRunner::SaveTapRows`, `RestoreTapRows`). New test: `ChunkFeatureTest.CheckpointRestoreWithItsTapRowsReadsTheCapturedFeatures`. The swap-table problem line now prints tokens, steps, drafted and accepted. |
| Swap table: image control pixels `93356574…` against the pin `3b7770ca…` | The table defaulted to the M3.6 protocol's old-arm pin. The published composition's pixels have been `93356574…` since 2026-10-16. | (c) | `swap_table.py` pins the frozen `qwen_image.json` pixels. |
| Gemma frozen and assistant digests (U4's recorded Gemma26/Gemma31 assistant differences) | — | — | These already passed at 2e38631c. |
| `GemmaEngineServingGpu.PartialCachedSwitch`, `DiagnosticBudgetCap` | — | — | F5 fixed these. They pass. |

### Bisection points

Abbreviations: Q = Qwen GGUF frozen, D = the three DeepSeek frozen tests,
F = FixedPrompt.

| Commit | Q | D | F |
| --- | --- | --- | --- |
| 5ec811a4 | pass | pass | pass |
| 3cf3118c | pass | pass | pass |
| 16b03c19 | pass | pass | pass |
| 425ebf23 | pass | fail | pass |
| 381eec18 | pass | fail | pass |
| 4533033c | fail | fail | pass |
| e1832086 | fail | fail | pass |
| 2b3452de | fail | fail | fail |

The swap-continuation bisection ran with one slot each and DeepSeek at a
16K context, so that the older builds' memory guard admits the pair:

- 53c2610b and 639583b9 (`m36-v1-artifacts`) are exact.
- dae227dd and 5ec811a4 (`u2-v1-artifacts`) differ.

The two stores cannot be crossed: each build refuses the other's layout. So
the bisection narrows to dae227dd together with its re-import. The fix makes
HEAD exact with both the bisection settings and the default configuration.

## Quality of the re-recorded families

`speed_gate.py quality` scores the fixed 4,000-character passage and takes two
greedy C1 replies:

| Family | Build | Targets | Mean NLL (nats) | Scores SHA-256 |
| --- | --- | ---: | ---: | --- |
| DeepSeek | 16b03c19 (before 425ebf23) | 966 | 2.7968790810 | `3d2483c9…` |
| DeepSeek | 2e38631c + F12 | 966 | 2.7937082602 | `7b96879c…` |
| Qwen GGUF | 16b03c19 | 996 | 2.8432114726 | `d7491cec…` |
| Qwen GGUF | 2e38631c + F12 | 996 | 2.8432114726 | `d7491cec…` |

- **DeepSeek:** the HEAD value equals U4's arm A exactly (LLM-RESULTS.md) and
  is 0.0032 nats better than the pre-change build.
- **Qwen GGUF:** the score path is unchanged. The changed schedules are
  decode rows, so the greedy reply digests differ.
- **Replies:** U4's remaining-domain-transfer controls reviewed replies under
  these schedules and passed them.
- **Repeatability:** the re-recorded digests were identical over three runs.

## FixedPrompt's reference

[forced_rows.cc](forced_rows.cc) and [run_forced.sh](run_forced.sh) build
against the digest-pinned llama.cpp v0.6.0 image with that release's headers.
They teacher-force `Tokens(0, 1285)` and the M3 runner's eight tokens through
the UD-Q2_K_XL checkpoint, with flash attention, F16 caches and 512-row
ubatches. Each run records both watched logits of every row. Two runs gave
identical rows ([reference-rows.json](reference-rows.json)).

Both reference arms rank the M3 runner's token first in every row. Largest
absolute difference over the 16 watched logits:

| Arm | vs v0.6.0 unfused | vs v0.6.0 fused |
| --- | ---: | ---: |
| M3 runner (the old anchor) | 0.424 | 1.593 |
| 16b03c19 | 1.124 | 2.519 |
| e1832086 | 1.533 | 2.907 |
| 2e38631c + F12 | 1.814 | 2.880 |
| v0.6.0 fused | 1.929 | — |

The current engine stays within the reference's own fused-to-unfused spread
of the unfused reference.

The test now anchors on the unfused reference. It keeps the M3 bounds: 2.0 at
each watched logit and the 0.5 near-tie rule for a changed argmax. The M3
runner itself lies within 0.42 of the new anchor.

## The quant-head fixture

`references/qwen-q4-head/operand-fixture-v2`, in the standing models directory
on both Sparks, is v1 with `first.logits.f32` and `changed.logits.f32`
re-recorded from the current joined/scalar Q4_1 VecQ:

- **Size of the change:** at most 1.9e-6 from v1 (9,100 and 7,140 of
  141,516 values), and every column's argmax is unchanged.
- **Joined against scalar:** each joined column still equals its scalar run
  bit for bit, and the graph replay equals the eager run.
- **Pinned hashes:** the test pins all eight SHA-256s. The two new ones are
  `e54fa85d…` (first) and `24a45633…` (changed); the other six are v1's.

## Swap table after the fixes

On 2026-10-26, on `spark`, with the default configuration from
`swap_table.py plan --case m3 --pairs-per-job 3`, the full M3 table (all six
pairs) passes `swap_table.py check`:

- worst LLM↔LLM row: 8.908 s
- worst prepared row: 8.764 s
- largest tracked peak: 114.27 GiB

The `qwen3.8:deepseek` pair was exact on three further runs: one with the
bisection settings and two with the defaults.

The near-8K Gemma cases:

- **gemma2, gemma26, gemma31:** pass.
- **gemma3:** still fails, on different rows from run to run. A fresh
  gemma3 plan is refused with "the plan report inventory could not be
  funded" (`engine/graph_plan.cc`).
  - **Cause:** while serving, growing `ExecutionMetadataAllowance` requires
    physical `MemAvailable` above the 6 GiB uncounted margin. With both
    models resident it is about 2.3 GiB, and growth asks no reclaim of the
    inactive model's weights.
  - **Not F12's:** this is a memory-policy defect (handed to F13), not a
    numerics or fixture one.

## Env-gated tests

Every input-gated skip in `tests/unit` was inventoried on 2026-10-26.

**Run by default in the models set**, resolving inputs from the standing
models directory and the v1 store, and failing when an input is missing:

- `Qwen38QuantHeadTest` (2)
- `ImportGenericModelsTest` NAS proofs (6)
- the publication halves of `ImportDrafterNormalizeModelsTest` (they now
  publish into the test scratch)
- `ImageGenerationGpu`: CompleteTextFlow…, RectangularRequest…,
  CompletedRequestsReplace…, ServerSwapsRetainState…, OrdinaryServerConsumes…
  and ImportedQ4Completes…
- `Qwen38EngineServingGpu.NativeC2ServerDefaultOptionsReopen`

The Q4 image composition `7959b4cd…` (transformer `a4d7ce4e…`) and its 512²
fixture were staged into both Sparks' standing store and `references`.

**Deleted as one-off diagnostics:**

- the four `DISABLED_U4Diagnostic*` tests and their enrollment support
- `ImageGenerationGpu.SameSourceMigrationScreen`
- `ImportedBf16RecipeMatchesPublishedPixelsAndSource`: moot, since the
  published composition is the import
- `NativeC2DurablePendingReopen` and `NativeC2Speed{Legacy,Shared}`. The
  durable case's exact continuation across a restart with the MTP drafter
  (a per-row tap, whose held positions are host bookkeeping) is kept by the
  review's new `Qwen38MtpKeptGpu.ConversationKeptAcrossARestartContinuesExactly`
  (default in the models set; passed three runs on `spark`).
- the duplicate `InstalledComposition.QwenImage`

**Host-gated, not input-gated.** These skip where there is no v1 store (the
workstation, qemu) and run on both Sparks:

- `QwenImage.InstalledComponentsBind`
- `ImageFamily.ThePublishedPipelineIsAdmitted`

`ApprovalsModels` covers the same composition in the models set.

### Opt-in tests

These keep an explicit skip:

| Test | Needs | Why it is not default |
| --- | --- | --- |
| `ImportModelOptModelsTest.ClosedNativeTargetOwnsEverySourceAndPreservesFormats` | `LLMP_TEST_MODELOPT_SOURCE`, `LLMP_TEST_MODELOPT_CLI_STORE`, and a prepaid 8 GiB `memory.max` cgroup | Its memory-confinement proof must run inside a delegated cgroup (`tools/job-proof`), which CTest does not create. |
| `ImportModelOptModelsTest.PublishesPinnedNativeTargetWithCompleteSourceAuthentication` | the same, plus `LLMP_TEST_MODELOPT_PUBLICATION_STORE` | A confined publication of the 105.9 GB source, under the same cgroup. |
| `ImportMlxNormalizeModelsTest.RealStagedCheckpointNormalizesPublishesAdmitsAndPlansAffineSplitWeights`, publication half | `LLMP_TEST_MLX_PUBLICATION_STORE` | It writes the whole 106 GB checkpoint, which took longer than the 10-minute job bound on `spark`. Its normalization half runs in every models run; the test then skips, saying so. Its write budget is now `llmp import`'s default (4 GiB less normalization); the earlier 1 GiB was refused with "artifact metadata has not been funded". |
