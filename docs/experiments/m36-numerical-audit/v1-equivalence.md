<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# v0 and v1 equivalence (M3.6 exit, serving becomes v1-only)

Every migrated model, except the two held back below, was imported from its
staged, archive-manifest-verified source into format v1 by `llmp import`
(D-113, D-114) and served against its v0 artifact on the current binary. The
v1 artifact executes bit for bit as v0 does, and the facts that show it are
frozen as digests, so the evidence outlives the v0 serving path.

Not in this report (owned elsewhere): Qwen3.8 native NVFP4 and MTP (its expert
layout is changing) and the Qwen-Image composition (its spec is changing).

## Imports

Spark A, `~/scratch/v1imp/llmp` built from main `ba2d4b19` (importer build
string `unknown`: the Spark tree has no `.git`, which makes the IDs
reproducible; the Gemma 2, Gemma 3, Gemma 4 targets, Qwen3.8 GGUF and
DeepSeek target IDs equal the ones their N4 slices published earlier).
Command: `llmp import SOURCE --store ~/.local/share/llmp/m36-v1-artifacts`, with
SOURCE a file or variant directory of the staged revision
(`~/.local/share/llmp/models/<org>/<repo>@<rev8>/`, `llmp-archive.json`
beside it). Every source file read was listed present in the manifest and its
SHA-256 matched; no manifest-listed file the importer does not read was
refused (no ModelOpt `amax*` file is involved here). Jobs `v1imp-small`
(Gemma 2/3/4, assistants and Qwen2 fixtures; 9 steps, all `published`) and
`v1imp-big2` (Qwen3.8 GGUF 21 min, DeepSeek target 29 min, DSpark 2 min);
`--shared`. One import owns a store at a time (the store lock), so the big
three ran into a second store and were moved into the first by `mv`.

| Model | Source | v0 ID | v1 ID |
| --- | --- | --- | --- |
| Gemma 2 2B Q8_0 | `bartowski/gemma-2-2b-it-GGUF@855f67ca` `gemma-2-2b-it-Q8_0.gguf` | `eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870` | `f176f912bc7761beee21cda0c22aff2080b8b84ed99faa6b962846c7e1379076` |
| Gemma 3 4B QAT | `ggml-org/gemma-3-4b-it-qat-GGUF@bbcac0d0` `gemma-3-4b-it-qat-Q4_0.gguf` | `8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb` | `634bc193468f5968ac142c2e5547f72777b182f3f694873075d7a50660f46626` |
| Gemma 4 26B-A4B UD-Q4_K_M | `unsloth/gemma-4-26B-A4B-it-GGUF@c099eb48` | `4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3` | `f0f72ab3d6c733d15472ceeecd05fdf200c266101c537bd3253548fc38559ab7` |
| Gemma 4 26B-A4B assistant | same revision, `mtp-gemma-4-26B-A4B-it.gguf` | `1040a0299a459e00ad0a77efd77bd319ac593986ba2c9ef29eb03d07ce97db42` | `23aa1c353091e1ce80309fe8dd55a90f679758525d4abaca2b974819aeae3ede` |
| Gemma 4 31B UD-Q4_K_XL | `unsloth/gemma-4-31B-it-GGUF@c1ac76e9` | `32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08` | `4d9a255ab47519b8061fcbc38bb5e72b9008d8e3538aad5edf0e7530fa82adc8` |
| Gemma 4 31B assistant | same revision, `mtp-gemma-4-31B-it.gguf` | `447a5c20a0a25632bf35e118d5dde1867a182cd209b9ccd3afe93866c3696120` | `07067ae0547848e4dcf904148a917df300354f31eb9bed88c78d1b92c83602c0` |
| Qwen3.8 Flash Next GGUF UD-IQ3_XXS | `unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee` `UD-IQ3_XXS/` | `5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78` | `18a671534a4355a892a7691ad610f90003cc456620353f466e7a08522f1774c2` |
| DeepSeek V4 Flash 0731 UD-Q2_K_XL | `unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93` `UD-Q2_K_XL/` | `8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234` | `698b1cd8696f75355d8a130b71369099c2c5d1d241932330565d33480cbc0bfd` |
| DSpark Q8_0 | same revision, `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` | `dd2d3f9c66f070fb231d27d5a11f38ff22c78dc8f089cecedbb67721e9b4bec5` | `5c9d95e24769c4d0a06088b0b6a8cc4afd4979b5ffc6d9357b4116f26fa583aa` |
| Qwen2.5-0.5B FP16 | `Qwen/Qwen2.5-0.5B-Instruct-GGUF@9217f5db` | `b93cdc326ba4f4c1c71da613503ecd848cd2a0caf122214f26e04588a12a9073` | `a96092c9fe6ab8512f1c8fdd4daaf69622f2a8dd211d9056a23d285da22c5e84` |
| Qwen2.5-0.5B EXL3 4.0 bpw | `blockblockblock/...-exl3-4.0bpw@7009334d` | `6e96e499ea9b859ea726327d4f82fb1739cb116ce4e9d3541d1ba67506efb4b4` | `ce8defd9150a7a3fa38c448dee4fb8177e94a1acfb04719263dd7882ec3deadf` |
| Qwen2.5-0.5B EXL3 4.5 bpw | `blockblockblock/...-exl3-4.5bpw@030d3a41` | `00d77caf056c45b6fa05ebdbeff8b38da79d0556a2fd4f1b661e7e21ef9ed83a` | `b62e2d22452de82c27a6ecd903f58cb8263af5dc220aa3e9319cb7eace2626fa` |

A target and its drafter are two artifacts paired by the model entry's
`artifact` and `drafter` keys; no `llmp-composition` document is written for
them (the CLI writes one only for the Qwen-Image pipeline). The "composition
IDs" are therefore the pairs: Gemma 26B-A4B `f0f72ab3…` + `23aa1c35…`, Gemma
31B `4d9a255a…` + `07067ae0…`, DeepSeek `698b1cd8…` + `5c9d95e2…`. The store's
`v1-ids.json` holds the map, the names and the pairs; the tests read it
(`tests/support/v1_frozen.h`, `LLMP_TEST_V1_STORE`).

## What was compared

For each family the same server, settings and plan run on the v0 and on the v1
artifact in one process (v0 first), and both are asserted against frozen
digests. The compared facts per family:

| Family | Test | Prefill heads | Greedy tokens and logit rows | Complete state | Spill and restore |
| --- | --- | --- | --- | --- | --- |
| Gemma 2, 3, 4 26B and 31B targets | `GemmaEngineNativeV1Gpu.PublishedGemma2Matches…`, `GemmaNativeV1ServingGpu.PublishedMatchesV0FullRowsStateAndSpillContinuation` (3) | two owners, full vocabulary | 4 tokens and 4 full rows per owner, three joined decode steps | every initialized page and coordinate, SHA-256 | state spilled, restored with no model work, then continued |
| Gemma 4 26B and 31B with assistant | `GemmaAssistantV1Gpu.PublishedPairMatchesV0AndTheFrozenDigests` (new, 2) | 144-token prompt | 48 then 16 speculative tokens, every row, drafted/accepted/steps | every stage including the assistant's, after turn, after restore and after the continuation | idle spill and restore, restored state equal, then 16 more tokens |
| Qwen3.8 GGUF | `Qwen38EngineServingGpu.PublishedGgufMatchesV0HeadsStateAndSpill` | 36 rows total | 2 turns of 8 tokens with rows | 3 images | spill, zero-compute restore, continuation |
| DeepSeek target | `Dsv4EngineTargetGpu.PublishedGgufMatchesV0HeadsStateAndSpill` | 2,560-token prompt | 2 turns of 8 tokens with rows | 3 images | same |
| DeepSeek with DSpark | `Dsv4EngineTargetGpu.PublishedDsparkMatchesV0HeadsStateAndSpill` (new) | same | speculative, drafted and accepted counts too | all stages | same |
| Qwen2 FP16 | `ComponentRunnerQwen2Gpu.FrozenFp16ControlIsBitExact` (fused, unfused, repeated; `LLMP_TEST_QWEN2_ARM=v1`) | 32-row prefill | 44 teacher-forced decode rows, all 76 rows against the sealed bridge oracle | checkpoint state at position 32 | checkpoint rewind, spill and restore, full weight eviction and reload |
| Qwen2 EXL3 4.0 and 4.5 | `ComponentRunnerExl3Gpu.SameImplementation{40G,45G,40O,45O}CheckpointSpillContinuation`, `Reconstruction145MatchesSealedLegacy` (4.0, sealed oracle) | 145-row prefill | 16 decode rows; all 161 rows hashed | checkpoint (145) and final (161) state | checkpoint rewind, spill and restore |

Qwen2 is served only through the runner (not the server), and its trajectories
are teacher-forced, so there are no greedy tokens there.

## Result

All bit-identical. Every v0 and v1 arm pair matched in head rows, tokens,
logit rows, histories, drafted and accepted counts, state images, restored
and resumed states. For Qwen2 the two arms were asserted against the same
sealed oracle and frozen digests in separate invocations, and the recorded
v0 and v1 digests agreed line for line (the recorder refuses a conflict).

Digests: `tests/unit/data/v1_frozen/<family>.json` (11 files; 14 to 16 per
Gemma and assistant family, 11 Qwen3.8/DeepSeek target, 15 DSpark, 12 EXL3
across four selected profiles, 2 FP16). They were recorded from the v0 arm
(`LLMP_TEST_V1_RECORD`, `record_to_json.py`) and are asserted for both arms by
the same tests. With `LLMP_TEST_V1_FROZEN_ONLY=1`, or when the v0 artifact is
absent, only the v1 arm runs, against the digests. A deliberately wrong
digest fails the test (negative control, `v1imp-neg`).
A digest holds for the selected plans and kernels of the binary that
recorded it: a change that moves bits re-records the files.

## A defect found and fixed

DeepSeek with DSpark served from v1 failed at its first prefill chunk ("a
weight has no address (stage 1, fc.weight)"). The v0 arm and the v1 target
alone (and v1 Gemma assistants) were fine. Cause: `component_build.cc`
skipped every stage-0 setup graph for a stored artifact, but DSpark's
feature projection (`fc.weight`) is bound inside the target stage's prompt
steps, so the drafter's weight places were never made. The fix builds those
graphs when the model speculates and takes only the drafter's weights from
them (the stored target's places still come from its validated roles). After
it DSpark v1 equals v0 bit for bit. The lowering error now names the stage
and tensor. The v1 FP16 repeated control also needed its v0-specific 490
weight-extent pin generalized (the v1 import lays the same bytes in its own
extent count; the pin is now the opened artifact's).

## Jobs (Spark A)

`v1imp-build1..6` (shared builds), `v1imp-small`, `v1imp-big2` (imports),
`v1imp-record` and `v1imp-rec2` (first record run: found the DSpark defect and
the FP16 pin), `v1imp-record2` (final record run, 10 steps, all pass),
`v1imp-assert` (the frozen run: both arms, plus the v1-only run of every
server test; 16 steps, all pass, no skips), `v1imp-neg`, `v1imp-sync`
(the copy to `spark-b` below). Exclusive jobs; Spark B untouched except as
the rsync target.

Copy to Spark B: `rsync -rlpc` over the direct link (`10.100.208.1`,
`aes128-gcm`), then each artifact's `manifest.json` SHA-256 checked against
its directory name there.
