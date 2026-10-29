<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ready RMSNorm/Mul matrix results

The checkpoints below retain their status at the time of execution. For the
current U6 outcome, installed data and final checks, see
[U6 final qualification](U6-FINAL-RESULTS.md).

The preregistered ready-body dataset passed independent evidence review. This document covers the first-screen cell and the eleven follow-up cells; the historical twelve whole-program rows remain unchanged. The twelve installed ready-body keys matched actual live descriptors and registry lookups at compiler implementation identity `94a79b540f8557318d52b02346b971a1843eb8e39c0f08707d944db74acbb5d9` (r35). Later compiler changes make that recognition historical: the measurements are not re-pinned, and current selection requires exact matching keys. The twelve reviewed rows remain installed in `data.json` with their original identities and measurements; raw candidate records remain outside Git.

The eleven-row follow-up matrix passed independent review, and the first-screen cell passed its separate independent review. Together, the twelve cells include eleven incumbent classifications and one neutral classification; maximum bookend drift was 1.283%. All runs used the registered local F32 RMSNorm/Mul body protocol. The run interval was `2026-10-25T00:34:30.634772+00:00` to `2026-10-25T00:51:32.419609+00:00` UTC.

## Shared run identity

- Follow-up registration: [`READY-MATRIX-REGISTRATION.md`](READY-MATRIX-REGISTRATION.md), commit `09178f4c05e20c7c149a2fb6a3f8f95d329aff1d` (SHA-256 `e99ca775510f961919d7cc182aba576ac71a8e2fa8fe7f3f93a471eee0c5e888`).
- Follow-up source: 113-path r34 manifest, SHA-256 `15d71c2e924c9bb87208f812f961aef2b3bf7242d2af013f6e4bff2172947a0b`.
- Follow-up binaries: Spark A `28ade247a2d2f16150317a3e84b1f85d23d515ad349a05b9434b3d3f61505cb1`; Spark B `36ab4c60f462f6ad7cf4f74e334da994534d47f8f83367bb383c8736c501f758`.
- Shared timing protocol SHA-256: `e013f06a8ebc77fb66c36654e3828c41a7650d1485922b1415852bddfe7d30c6`.
- The first-screen 1024×1 tuple followed the separate [first-screen registration](READY-FIRST-SCREEN-REGISTRATION.md), commit `d194bd66c1e2b006c801e7989ad265ebc0a0e2c3`, with a 112-path source manifest (`d7dc4f34a82ea5921e19dfb7244fd0c0e393af3ac7669ba18ea021f5e56915b7`) and Spark B binary `36ab4c60…`; its protocol is identical.

## Measured cells

Each median pair is incumbent / alternative nanoseconds per captured body. Bookend drift is the registered before/after alternative-bookend check.

| Width × rows | Host | Incumbent / alternative median (ns) | Classifier | Bookend drift |
|---:|---|---:|---|---:|
| 256 × 1 | Spark A | 977 / 1501 | incumbent | 0.467% |
| 256 × 4 | Spark A | 993 / 1498 | incumbent | 1.077% |
| 256 × 32 | Spark A | 1030 / 1467 | incumbent | 0.479% |
| 256 × 256 | Spark A | 1465 / 2489 | incumbent | 0.160% |
| 1024 × 1 | Spark B | 1379 / 1807 | incumbent | 0.221% |
| 1024 × 4 | Spark A | 1377 / 1835 | incumbent | 0.821% |
| 1024 × 32 | Spark A | 1678 / 2366 | incumbent | 0.467% |
| 1024 × 256 | Spark B | 7310 / 8558 | incumbent | 1.283% |
| 4096 × 1 | Spark B | 1910 / 2117 | neutral | 0.379% |
| 4096 × 4 | Spark B | 1972 / 2326 | incumbent | 0.302% |
| 4096 × 32 | Spark B | 2617 / 3432 | incumbent | 0.205% |
| 4096 × 256 | Spark B | 10668 / 20925 | incumbent | 0.417% |

## Row identities and receipts

Recipe-key digests identify the complete typed keys recorded with each row. Event receipts hash each run’s exact 36-event timing array.

| Width × rows | Cut | Recipe-key digest | Event receipt SHA-256 |
|---:|---|---|---|
| 256 × 1 | r34 matrix | `d55b0c79e21f4a135779be677525d5dc196335737a5beb22f4c54dc0145f4eb1` | `2c1d89af3c54e1bd773c6e2c9484b19d7cb992ff5f059ce1948afe53fd61c431` |
| 256 × 4 | r34 matrix | `ad06cca150cbf3f1076357cb2ab3bcd36d4dea5cda3919e9785a20879d9da04c` | `f00135adcca0a24ccaa175bef54051ca03e0e0a1248aa99cee24bfce7969ae37` |
| 256 × 32 | r34 matrix | `2855914fe765b06721fcc3960b448dd9292ee426658b581c888750d546c25c3c` | `1dae5cbc74a86decb06ccd4c9937ca1dbdebf50685d6f686642bd0b27d2553d5` |
| 256 × 256 | r34 matrix | `3102e77d5d520011ef41f744a5df3fb4c2a774ca50e9bfd3d06ade10ea02cf15` | `48c024d299d349893dfe8b1198f644ae2617dc3e484d67345bc15393ebe9df96` |
| 1024 × 1 | r32 first screen | `0a5b4a15c30da277991c2c4ca8413b7de68c87d4003a911cda3a4a1d1f74e93f` | `7dea755b056beaefdc2a06fb16da22154b82b88269d83e7ede90bd0c897945d8` |
| 1024 × 4 | r34 matrix | `4c3122c72d3ea2ffd47c822a1132ac1e12044fbdd85f318b02e3e34aca3fae14` | `6af3d7813480d58d5ccf04f975f644bbffd379a638a04050b2a0adb75c621420` |
| 1024 × 32 | r34 matrix | `d91038b54fbcf8d0863953de2e3b3b7f4c3ccd367403b2e929384e86dc1a2518` | `eaba11e2ce4a8556365e2f5a23a68647a395d58de031700956341dad0376c369` |
| 1024 × 256 | r34 matrix | `5a843b9911d530cc2e2adb81d0f54cbb2a2a86a2605b260de55c5b4fae2ce8e7` | `869047516deae48c27a22087e166ceab33c1fd1c492f6070a3746b2318d0a43a` |
| 4096 × 1 | r34 matrix | `3ae3c090caca0e5eda3c299635229bdbc7d52111fec263e51c7e2fec76f32620` | `c701e5c6f271df19c420abe541aa907c0e34680d60ae8e06cfe0d2edf535c5a1` |
| 4096 × 4 | r34 matrix | `98892d27be2a95ff391c087429d7dc7162c4c956cd6bce53bdc1588ee4413d46` | `6e650251addf4ab263bf17a45d93c4f52a8b72b35dcc987c64b0e786ae703355` |
| 4096 × 32 | r34 matrix | `2a8cbf9cba0bb7036c53585a9360e3c517c77281b11b890757bd93dc9703614d` | `88a349836d55816cf3503eb71277847a3c48b6d0837d6a18135329d1c041c1bc` |
| 4096 × 256 | r34 matrix | `f976071f8b566851335b97b6f6736ecd96fdba872f74e4f12f370fe741dbe6ab` | `37edef9efe8da77ae17e86007d36f5ba874aa3aee6df3e5a7d0226e8becea739` |

## Evidence and limits

- The eleven-row follow-up audit is `/tmp/llmp-m35-orchestration/u6-resume/ready-matrix-09178f4c/independent-review.json`, SHA-256 `b3c845548a54d77de88bad69cf2b9e925f61ab87195d9946577341786b281f70`. It independently checked that matrix’s 396 event positions, samples, medians, classifiers, bookend gates, event/protocol hashes, and complete typed key serialization.
- Matrix evidence: Spark A `spark/llmp-u6-ready-matrix-09178f4c-spark/matrix.json` (SHA-256 `ab7fd9f90e0841b092d3b5ed25724b9288c25646c3550c16463c445e71833d59`); Spark B `spark-b/evidence.json` (SHA-256 `19969d7cd0eb5416b6484dac24dd429637c2c0e59cdc3d39fb24397be6d4218f`).
- First-screen evidence and its separate one-row review: `/tmp/llmp-m35-orchestration/u6-resume/ready-first-screen-r1/evidence.json` (SHA-256 `4170e4fc46fabcbcc9c837d54f1842947eb5296b9b4cd6e05a345c2beb1cee72`) and `/tmp/llmp-m35-orchestration/u6-resume/ready-first-screen-r1/independent-review.json` (SHA-256 `50ea7cd31196148c44d08f1fbd6845abd122457c3053bce59cb9d9bfaa675af5`). That review checked the first screen’s 36 events separately.
- The complete typed key values, per-run source and binary checks, CPU-reference errors, retirement status, and supervisor receipts remain in the linked scratch evidence tree. Raw samples are not included here.
- This is local ready-body timing evidence only. It makes no full-model speed, broader operation-family, or U6 completion claim. No candidate row was installed by these runs.

## Installed-registry qualification

At r35, the integrated native data-only rebuild preserved compiler implementation
identity `94a79b540f8557318d52b02346b971a1843eb8e39c0f08707d944db74acbb5d9`
and all twelve measured recipe-key digests. On Spark A, actual descriptor
production and `Registry::FindRecipeComparison` recognize all twelve exact
rows and report their measured preferences. Width 2048, rows 1 remains
unmeasured. All thirteen validation-only invocations pass two changed-input
CPU-reference frames using the same 128-invocation recordings; no new timing
is collected. The comparison/rank suite passes 73/73 on Spark B with no skips.
The checked integration cut is `u6-integrated-ready-data-r35`, manifest
`f34e8c2697b333cc48d60302e2ec6eb7f318e8a954e9afac42a1a86eeb2b7a08`;
its 113 paths match root and both Sparks. Actual receipts are retained at
`/tmp/llmp-m35-orchestration/u6-resume/integrated-r35-evidence/`.

This qualified the r35 implementation and these exact local tuples.
Later implementation changes require fresh matching measurements; historical
observations are never relabeled. Other operation and format classes remain
open within U6.

## Subsequent row-table qualification

The later row-table checkpoint adds owner-backed logical table identity and
physical slot bounds to comparison admission. A genuine `ModelRunner` prepares
F32 and F16 artifact rows through its normal transaction, including duplicate,
out-of-order and changed logical IDs. Both actual RMS/Mul schedules match the
original full-table CPU reference; complete row slots and norm scales remain
unchanged. Descriptor keys are stable across the changed maps, and bound seals
preserve their descriptor facts. The F16 table still feeds an F32 body; this is
not an F16-output qualification.

A separate Q8_0 control verifies the unsupported `GetRowsExt` proof path: closed
local norm certificates request the actual lookup's source expansion, and the
expanded region refuses the unsupported typed-access recipe. Cumulative selection
retains the first ordinary native program, which still binds and executes. This
control does not qualify quantized artifact-row gathering or a new comparison
schema.

Across `spark` and `spark-b`, the checked shared inventory has 3,951 passing
CTest cases, no remaining failures or skips. The two Qwen3.8 saved-recurrence
cases also pass on `spark-b` for ModelOpt and GGUF (two GTests under one CTest
wrapper). The combined results retain earlier passing cases and replace only
failed checks after their corrections; they do not represent a second full run
on the final source. The corrections declare the GPU fixture's scratch directory,
update a stale empty-table expectation to the twelve installed recipe rows, and
refresh 32 compiled objects whose dependency receipts named an older prepared
GGML tree. No benchmark or additional model inference was run for that refresh.
Portability checks cover 736 files; license and embedded-header checks cover
2,657 files, all clean.

The final 114-path integration manifest is
`b9cc7d3bae77cf90e84dcc6820e389a46613133d1d335ec02580ee09cfeab0e1`.
All eight build/check jobs preserve matching source hashes before and after.
The aggregate receipt is
`97f8c6a871a671d879a1d92c7348313b81d69a031750c7b33dc986cba5c7e894`;
root independently reconstructed its current test inventory and source checks.
Raw receipts remain outside Git under
`/tmp/llmp-m35-orchestration/u6-resume/integrated-row-r40-evidence/`.
These functional checks collect no timings and do not refresh the r35 rows or
complete U6's remaining fusion and shape coverage.

## NormGate functional qualification

The subsequent F32 NormGate checkpoint closes the actual fused recipe and its
own decomposition, including the backend's gate reshape and exact typed unary
accesses. Ready comparisons admit dense F32 Features only when their original
source and physical slot agree; the legacy comparison protocol keeps its
existing restrictions. The zero-prefix 128-dimension, 48-head, one-row producer
captures one complete body per arm: one fused node versus three decomposition
nodes. Both arms pass two changed input frames against the original-IR CPU
reference, with maximum absolute error `2.384185791015625e-7`; immutable gamma
bytes remain unchanged.

Both Sparks run the same r44 source checkpoint through all 3,954 shared CTest
cases: 1,980 on `spark` and 1,974 on `spark-b`, with no failures, skips, missing
or duplicate cases. Each job verifies all 114 source paths before and after
execution. This is a complete shared regression run on that checkpoint;
unchanged ModelOpt/GGUF saved-recurrence paths retain their previous model
qualification. Raw logs and the combined result remain outside Git under
`/tmp/llmp-m35-orchestration/u6-resume/norm-gate-final-checks/`.
This checkpoint collects no timing and makes no whole-GDN adoption or published
checkpoint applicability claim.

A later real GDN BuildStep control exposed and corrected state validation that
missed fused read-modify-write operations. State access and write authorization
now use the canonical IR write enumeration, retaining both the read and write
of recurrent state and the borrowed/foreign-stage restrictions. Production
certificates can also describe authenticated local calls separated by outside
work; executable comparisons and remainder checks retain their contiguous
closure requirement. Independent review and six focused native checks on
`spark-b` pass for these corrections.

The real GDN and zero-prefix producer have equal semantic boundaries, planned
regions and incumbent physical recipes, but different alternative physical
recipes. In the real decomposition, the later gate producer reuses the
attention input's dead storage; the isolated fixture stages both operands at
entry and allocates them separately. Their complete keys correctly remain
distinct. Expanding the region to include the authentic gate producer makes
the larger value layouts agree, but the isolated producer still reuses its
small stream input where the actual GDN plan does not. One source-faithful
construction-order correction leaves this distinction intact. The next screen
therefore uses the two actual full BuildStep programs with identical input and
reset state; neither isolated fixture qualifies the real GDN physical recipe.

The local NEOX RMSNorm/Mul/RoPE producer also passes both normally selected
recipes with two changed input/position frames. Width 128 at one and two rows
uses one captured fused node versus two decomposition nodes, retaining the
RMSNorm/Mul child fusion. Maximum absolute CPU-reference error is below
`4.303e-6`; gamma and I32 positions remain unchanged. Ready comparisons admit
I32 position inputs only with exact original-source, slot, segment and extent
facts, including canonical singleton scalars, while legacy eligibility stays
unchanged. All 58 comparison-pair tests pass on `spark`.

Additional GPU controls qualify optional F32 frequency factors with the actual
fused recipe and its normally selected RMSNorm/Mul plus `ggml.rope.ext`
alternative. Descriptor and bound certificates, remainder checks and two-frame
CPU outputs pass with gamma, frequency and positions guards. Typed malformed
descriptor checks and genuine unsafe-binding requests refuse; the malformed
descriptors never execute. The accepted extended-RoPE proof is limited to
checked forward F32 NEOX with frequency factors. This qualification collects
no timing and makes no broader model applicability claim.

The combined r45 source passes a fresh complete shared regression: 3,959 cases
across both Sparks (1,982 on `spark`, 1,977 on `spark-b`), with no failures,
skips, missing or duplicate cases. Both actual ModelOpt/GGUF saved-recurrence
cases also pass on `spark-b` under exclusive admission. Each build/test job
verifies the same 115 source paths before and after. The r45 manifest SHA-256
is `8619865b01f93c75c247dc94d9af77f5155096cbbbc605707277a10693779682`;
raw results remain outside Git under
`/tmp/llmp-m35-orchestration/u6-resume/neox-gdn-integrated-r45/`.
The r46 NEOX benchmark-only timing support passes its separate focused
qualification; that r46 change does not represent another full regression run.

The actual full GDN programs now also pass a separate correctness screen on
`spark-b`. Both retained default plans run through ordinary Bind/Run and
completion. Each arm first poisons activation storage, then restores identical
nonzero F32 convolution and recurrent state and stages its own inputs. Every
logit and both complete updated state tensors match the original-IR CPU
reference within the prospectively fixed normalized maximum gap `1e-4`.
Nontrivial state updates are checked. Independent review and the final focused
native rebuild/test pass. This adds no timing or portable-key/adoption claim.

The later r47 measured-capture implementation and actual-GDN benchmark pass
all 3,961 shared cases across both Sparks (1,982 on `spark`, 1,979 on
`spark-b`) and both affected ModelOpt/GGUF cases, with no failures or skips.
All 118 source paths match before and after. Its registered
[NEOX result](READY-NEOX-CAPTURED-EVENTS-RESULTS.md) is an incumbent win;
its [actual full GDN result](ACTUAL-GDN-FIRST-SCREEN-RESULTS.md) is neutral.
Both have independent clean result reviews and normal teardown. Installing the
single reviewed NEOX row then passes 85 focused checks and live hit/miss
qualification on `spark`, with unchanged backend build identity. Historical
RMS rows retain their keys. Raw combined regression evidence is under
`/tmp/llmp-m35-orchestration/u6-resume/captured-events-r47/`.

## F16 product-fusion validation checkpoint

Typed local F16 MMVF/SwiGLU comparison now preserves the actual fused reads and
final write, zero-scratch vector dispatch, reserved parameters and complete
source group. Split calls use their own alias rules, permitting safe reuse of
dead original input storage. Whole and adversarial review is clean; six focused
native checks pass on `spark-b` with no skips.

The original-IR, standalone-component producer runs without `Keep` or forced
placement. On `spark`, widths 256 and 128 at one row pass complete original-IR
and double product-reference checks over two independently NaN-poisoned and
restaged frames. Fused/decomposed maximum normalized errors are respectively
`1.1442768719541228e-9` / `3.731873196740147e-6` and
`7.537318199911449e-10` / `2.020610162703175e-6`, within unchanged `1e-4` /
`2e-2` source-backed limits. Actual F16 weights and readable tails stay unchanged;
one-body captures contain 3/5 nodes, including two timing events. Validation
checks positive durations without collecting ranking samples.

Existing RMS, NormGate and NEOX validation controls also retire normally. Their
rows stay unchanged and miss at the new backend identity
`6dd80faece88fa94656671d353adf4a1b1401d1470e23bf0339dfa2377db300c`.
This adds proof and producer coverage, not measurements or a model-speed claim.
The two new key digests are
`95371c9a5a7b665de1bf596e59a45e45782b765abb554351c1549916f1490cfd` and
`e92201b82d9aae2f78df8d770780bbbaf5584edbb1e7e5dda1a6fb9786fcabc4`.
The complete 118-path source manifest has SHA-256
`e147ee3a5389be3e0a80595ea7419008fe79b4cc9921ce1a87e022c1e7cf69a3`;
benchmark binary SHA-256 is
`8496df6d0e66812180be40f0badcdfcf718d9b8f0b6823229fa2caafdbe301f1`.
Raw qualification remains outside Git in
`/tmp/sol-f16-matmul-glu-r54-summary.json` and matching build/functional logs.
U6's remaining physical contracts and measurement coverage remain open.

## Shared Add-fusion validation checkpoint

The RMSNorm/Mul/Add and F16 MatMul/Add recipes now use their existing typed
launch checkers for actual accesses, default precision, full graph closure,
vector selection, zero scratch and reserved parameters in both body and prefix.
Split calls retain their own lifetime and alias rules. Whole and adversarial
review is clean; eight focused native checks pass on `spark-b` without skips.
The standalone component route admits these two exact selected implementations;
three further native controls exercise actual GraphBuildIO stream/side ports,
changed dynamic residuals and the existing GLU/NormGate paths. F32 MatMul/Add
remains a default-planner negative under today's F16-only matcher.

One concrete benchmark helper now shares discovery, binding, capture and staging
between GLU and both Add fixtures. It selects the complete original primitive
member group, distinguishes rewrite tickets from later merged fused tickets,
and checks exact kernel counts from the accepted physical calls. Every arm
poisons the arena with NaNs, restages all original component inputs, replays one
complete body, checks every original-IR CPU output and immutable weight byte,
and repeats with changed inputs. The NormAdd alternative retains its ordinary
RMSNorm/Mul subfusion; both new alternatives use two actual kernels. Captures
contain 3/4 nodes including two timing events, with one body repetition.

| Fixture | Width × rows | Fused maximum normalized error | Alternative error | Both-arm bound |
| --- | --- | ---: | ---: | ---: |
| RMSNorm/Mul/Add | 256 × 1 | 1.37988118e-07 | 1.15540844e-07 | 1e-05 |
| RMSNorm/Mul/Add | 128 × 2 | 1.51582165e-07 | 1.18639647e-07 | 1e-05 |
| F16 MatMul/Add | 256 × 1 | 4.68844389e-05 | 4.68844389e-05 | 2e-02 |
| F16 MatMul/Add | 128 × 1 | 3.41426684e-05 | 3.41426684e-05 | 2e-02 |

All eight functional cases on `spark` retire normally: these four cells plus
GLU, NormGate, NEOX and RMS controls. No ranking samples are collected and all
fourteen unchanged historical recipe rows miss at backend identity
`2680a856d641564c300edc573b1a8bc850bf3dc50d5c360858b16a4a176b36cb`.
The qualified 118-path producer manifest has SHA-256
`f8fa8db1ac702061774cf33303ee9154e33646b28999166eb05050c99ae38aef`;
benchmark binary SHA-256 is
`4510ba053503a1485bb96f26901175f40b55a6b43a9070067a1fdcbee64266ec`.
Raw receipts and build/functional logs remain outside Git at
`/tmp/sol-add-producer-r59-summary.json` and matching paths. The final integrated native builds pass on both Sparks. All 3,964 shared
checks pass (1,982 per host), with no failures, skips, missing cases or
duplicates. Both actual ModelOpt/GGUF saved-recurrence cases pass on `spark-b`.
All 118 frozen source paths match before and after each job. Aggregate and raw
check evidence remains outside Git at
`/tmp/llmp-m35-orchestration/u6-resume/add-final/results.json`.
This closes the F16/Add validation wave; U6 remains open.

## Split-modulation validation checkpoint

Optional checked access descriptions in the existing recipe inventory now serve
whole-program and local comparison proofs. They preserve existing scoped dtype
rules, authenticated elided descriptors, actual dispatch variants and ordered
reads/writes, including both joined outputs and the width-above-8192 residual
read. The shared kernel checker also rejects residual writes over live
modulation vectors. Six focused native controls and actual CPU-only checks pass
on `spark-b`; whole and adversarial review is clean.

Authentic later-round discovery exposed a planner mismatch: grouped ports use
current fused-member ordinals, while the source precheck assumed primitive
ordinals and direct incidence. Grouped source references now validate bounds,
root and view; exact member lineage and complete current ports remain mandatory
at the authenticated checkpoint. CUDA and CPU-only planner controls pass six
cases each, with a further live mapped-tensor forgery control passing. The first
three joined producer refusals remain recorded; the final source passes them.

All ten final functional cases on `spark` retire normally. The three F32
families run at 320 × 37, split 12, and 256 × 17, split 1; joined also runs at
8193 × 2, split 1. Each arm poisons activation backing, restages independently
changed original inputs, captures one complete body and checks every original
CPU-reference output over two frames. Joined checks both outputs, comparing one
kernel with the ordinary two child kernels. Pure alternatives contain six or
five kernels as identity repeats disappear. The exact image error metric is
`abs(error)/max(abs(reference),1)`; worst errors are `7.748604e-6` for residual,
`2.871518e-7` for modulation and `1.987815e-5` for joined, within unchanged
`1e-4`, `1e-5` and `1e-4` limits. GLU and both Add regressions also pass.
These cells cover interior F32 splits; actual BF16 consumer-qualified producer
cells and broader shape/format measurements remain open.

No ranking samples or rows are added. All fourteen unchanged rows miss at
backend identity
`896205e247a66634c1bd88d7df479bd93ede20e998e99988c18c384365687351`.
The qualified 121-path manifest SHA-256 is
`19e8563dacfe2f4335473911b16c7f655d9d7f855f56df652db85417365c35fb`;
benchmark binary SHA-256 is
`63f2423ccdfe002f00842971b70044f695ecacc7cfaa0eabd554e402ca51cb5e`.
Raw receipts remain outside Git at `/tmp/sol-split-producer-r61-summary.json`
and matching build/functional logs. Both integrated native builds pass.
The final aggregate is 3,966 shared checks (1,982 on `spark`, 1,984 on
`spark-b`), with zero failures, skips, missing cases or duplicates. Seven
negative fixtures initially used newly supported operations as an unproved
suffix; their corrected ReLU suffix preserves the original refusal, complete
output and scratch-alias assertions. All seven reruns pass; the other passes
are retained. The actual image-block replay on `spark` and both ModelOpt/GGUF
saved-recurrence controls on `spark-b` also pass without skips. The corrected
three-file delta changes tests only. Source checks retain the 121-path cuts;
aggregate and raw evidence remain outside Git at
`/tmp/llmp-m35-orchestration/u6-resume/modulation-final/results.json`.
This closes the split-modulation validation wave; U6 remains open.

## Sectioned RoPE validation checkpoint

The checked-access inventory now covers forward RoPE section calls, actual
BF16-to-F32 gamma conversion, position copies and the existing sectioned
RMSNorm/Mul/RoPE kernel. Plain NORMAL rotation uses the existing sectioned
implementation, including its one-section case; the generic norm/RoPE
capability remains NEOX-only. Whole-program RoPE and copy domains retain their
previous coverage. Local access facts include the full copied prefix and suffix
of partial rotations, each independent I32 positions input and the conversion's
actual write destination.

GGML conversion returns a view of its separate destination. Both descriptors
now retain their genuine planned identity; lifetime validation authenticates
only that exact checked pair when identifying its storage and definition.
Other duplicate roots still refuse. Six native cases on `spark-b` check both
arms' whole descriptor and bound seals, local closure and bound self-remainder,
then capture and replay two changed frames against the complete original-IR
CPU output. Two actual CPU-only controls and existing NEOX/Qsa controls pass.
Whole and adversarial review is clean.

Integrated startup checks exposed an overly strict conversion contiguity check.
The narrow `CheckConvert` correction now matches the actual flat CUDA copy's
GGML contiguity predicate: strides in dimensions of extent one are never
addressed. Active stride gaps and overlapping operands still refuse; dtype,
shape, element count, current views, alignment and ownership checks remain.
The lowering identity guards and the shared `Packed` predicate are unchanged.
The full native validator suite passes 37/37, focused GPU controls pass 9/9
(including both startup regressions and the six-case RoPE control), and the new
validator control also passes in the actual CPU-only build. No tests skip.

All seven functional benchmark cases on `spark` retire normally. The six new
cells use three heads and five rows, with NORMAL sections `[64]` or `[32,32]`,
or NEOX sections `[16,56,56]`. Each arm stages every original input after NaN
poisoning, checks independently changed positions, validates all original CPU
outputs, and preserves gamma bytes, readable tails and positions. BF16 gamma
uses its actual rounded bytes and the alternative's ordinary widening call.
The unchanged NEOX 128-by-2 control also passes.

| Sections / gamma | Captured nodes, fused / alternative | Worst image error, either arm | RMS-relative error, either arm |
| --- | ---: | ---: | ---: |
| normal1-f32 | 3 / 4 | 6.9141388e-06 | 1.2182561e-06 |
| normal1-bf16 | 3 / 5 | 6.9737434e-06 | 1.2145206e-06 |
| normal2-f32 | 3 / 5 | 1.0669231e-05 | 1.0206197e-06 |
| normal2-bf16 | 3 / 6 | 1.0669231e-05 | 1.0183252e-06 |
| neox3-f32 | 3 / 6 | 2.0120293e-05 | 1.840286e-06 |
| neox3-bf16 | 3 / 7 | 2.0142645e-05 | 1.8383978e-06 |

Each capture contains one body and two timing-event nodes. The alternative
retains RMSNorm/Mul fusion, one ordinary call per rotation section, and one
additional conversion for BF16 gamma. The established sectioned-output limits
are RMS-relative `1e-5` and image error
`abs(error)/max(abs(reference),1) <= 1e-4`; no bound changed after execution.

No ranking samples or rows are added. All fourteen historical rows miss at
backend identity
`23178efc87f8f5fcd7672c61173ba98f14083a7cf19c92d6fedc90a7ae7d1c1a`.
The qualified 123-path producer manifest SHA-256 is
`ee0071ec87b55cd5681fcd16ef80dea9ee7e6dc15316d1bd2a3c71f3779c8825`;
benchmark binary SHA-256 is
`e79ec576c5f242e5ec84ee34205ee92219b12b9deaaaf3a438fefe28ed0ef44b`.
Raw receipts remain outside Git at
`/tmp/sol-sections-rope-producer-r65-summary.json` and matching logs. The seven
final producer cases pass with the same metrics and selected calls above.

The final audit records **3,969 unique shared checks**, with zero failures,
skips, missing cases or duplicates: retain 3,966 initial passes, replace the
two startup failures with corrected passes, and add the new validator case.
Corrected native builds pass on both hosts. Seven startup/validator controls
and the actual image conditioning/replay model case pass on `spark`; both
actual ModelOpt/GGUF saved-recurrence model cases pass on `spark-b`. The focused
validator and GPU runs, plus the CPU-only control above, are retained as
additional validation, not added again to the unique aggregate. The audit is
`/tmp/llmp-m35-orchestration/u6-resume/rope-sections-final/results.json`.
This closes the sectioned-RoPE validation wave. U6 and broader shape/format
measurements remain open; this checkpoint establishes no performance coverage.

## Spatial normalization validation checkpoint

The existing checked-access inventory now covers F32 spatial channel
normalization, primitive L2Norm and local F32 Cont copies. The Cont scope
preserves existing whole-program F16 coverage and local I32 coverage. Spatial
parameters seal epsilon, scale and optional SiLU; L2Norm retains its actual
`rsqrt(max(sum(x*x),eps*eps))` contract and legal exact in-place output. The
fused kernel retains its disjoint-output requirement. Copy and spatial kernel
routes depend on sealed shape, strides, dtype and semantic fields; no new
selector or numerical eligibility was introduced.

PERMUTE schema and hashing seal all four ordered axes and reject reserved
fields, invalid transforms and extra sources. Active strides must match the
source transform; inactive unit-dimension strides may use lowering's existing
canonicalization and remain hashed as actual descriptors. Zero-work PERMUTE
members are authenticated through selected real calls' metadata dependencies,
with paid canonical IDs, complete source-origin groups and counted typed
planned/physical records. Mapped input cuts remain boundaries. Actual call
intervals, kernel counts, alias checks and lifetimes are preserved; metadata
members are not invented launches or removed from the accepted source group.

Whole and adversarial review is clean. Six focused native controls on
`spark-b` pass without skips, including spatial schema/access/alias checks,
proper-subregion execution, nested schedules, nonpacked saved auxiliaries and
nonzero views. Two controls also pass in the actual CPU-only build. The
existing spatial GPU test passes all sixteen width/SiLU combinations.

The final r69 producer on `spark` passes ten authentic original-DAG cells:
F32 `[37,3,C]` input and owned F32 `[C]` gamma, epsilon `1e-12`,
scale `sqrt(C)` and `C = 1,7,144,513,1025`, each with and without SiLU. Channel one is a positive
cell. Each default fused plan and its own decomposition has verified whole
descriptor and bound seals with no missing proof, closed local certificates
and a same self-remainder. Two changed frames restage all original inputs
after full activation poisoning, include exact-zero and epsilon-clamped tiny
pixels, and check every original-IR CPU output, immutable gamma and readable
tails. The unchanged NormAdd and joined split-modulation controls also pass,
for twelve functional cases in total.

The fixed spatial metric is
`abs(error)/max(abs(reference),1e-3) <= 1e-5`.
The maximum over the ten spatial cells and both arms is
`4.308632053486497e-7`. The fused body has one kernel. Its own alternative
retains `Cont -> L2Norm -> Scale -> Mul -> Cont`, with an additional unary
call for SiLU: five or six kernels. Single-body captures contain two timing
event nodes, giving fused/alternative node counts `3/7` or `3/8`.
These are functional capture checks; no ranking samples or timing comparisons
were collected, and no ranking rows changed. All fourteen historical rows
miss at backend identity
`fe95534b4a7cc5f65332ee677be37b3b89a8c0b0b74f8bd5bb64363ff749416b`.

The qualified source manifest contains 123 paths, SHA-256
`2b231f0565c8355c257cf0f8ec4adc6dc0d66cfd10cb5c5f78b18f4e592e85d8`.
Benchmark source SHA-256 is
`43884811750b564ca29079fd3124340ee94f49574e70d933ab055e867709dbc0`;
binary SHA-256 is
`da404a26b92b59a3b0f510429be3b4ad37bc0e040cfe69a433053305a17969f2`.
Raw receipts remain outside Git at `/tmp/sol-spatial-producer-r69-summary.json`
and matching build/functional logs. The independent raw/source audit is
`/tmp/llmp-m35-orchestration/u6-resume/spatial-norm-final/producer-audit.json`.
Focused core receipts are `/tmp/sol-spatial-metadata-build-r3.log`,
`/tmp/sol-spatial-metadata-tests-r3.log` and
`/tmp/sol-spatial-metadata-cpu-r3.log`.

Final combined native builds pass on both hosts. The actual JUnit inventory
union contains **3,970 unique shared passes**: 1,982 on `spark` and 1,988 on
`spark-b`, with zero failures, skips, missing or extra cases, or duplicates.
Three actual image spatial cases and one whole-VAE case pass on `spark`;
both actual ModelOpt/GGUF saved-recurrence cases pass on `spark-b`. The latter
are two actual GoogleTest cases inside one CTest wildcard entry. Preserved
model `LastTest.log` files confirm every actual case ran and passed, with no
hidden skips. All 123 source paths verify before and after both hosts' build,
shared and model jobs and at final read-only audit. Focused core checks above
are retained separately and are not counted again in the unique shared total.
The complete audit and evidence paths are recorded in
`/tmp/llmp-m35-orchestration/u6-resume/spatial-norm-final/results.json`.

This closes the spatial normalization validation wave. U6 and broader
shape/format measurements remain open; this checkpoint establishes no
performance coverage.

## HC post/flat-normalization validation checkpoint

The existing optional access inventory now describes primitive HC post and
its fused post/flat-normalization implementation. Primitive post reads all
four original inputs and writes its post output. The fused call reads the
same four inputs and writes both post and normalized outputs; it does not
invent a global read of the intermediate flat view. Existing validators,
full-graph fusion closure, alias/lifetime checks and reserved-field rules
remain authoritative. The concrete fused recipe declaration now matches its
actual `Fused(kHcNormPost)` capability and selected plan operation.

Local comparison covers F32 post and normalized outputs; existing whole
F16-normalized coverage is preserved. The `hc_post_norm_f16` name does not
classify this F32 source cell as a numerical change. The experts/post/norm
helper remains unsupported by this comparison proof. Stateless ComponentPorts
lowering admits only the actual selected HC implementations.

The original primitive reshape is authenticated as a real planned metadata
dependency: its input/output dimensions must match retained planned shapes,
with exact type, packed storage, element count, view and root checks. Mapped
input cuts remain boundaries. The fused flat is a separate owned GGML
reshape descriptor with no IR tensor or producer; its full-graph checked
post/flat/norm chain, paid descriptor ID, exact zero-offset view and retained
post-output storage authenticate it without fabricated identity. Typed
metadata hashing and real call counts remain intact.

Whole and adversarial review is clean. Focused native checks on `spark-b`
pass 7/7 and actual CPU-only checks pass 3/3, without skips. These include
four-read/two-write facts, changed normalization-axis and alias refusals,
reserved parameters, whole-F16/local-F32 scope controls, and the existing
nested HC case in both token and ComponentPorts lowering modes. The nested
case retains an outside post consumer, all three original CPU outputs,
whole descriptor/bound seals, local closure, self-remainder and two changed
capture frames. Proper-subregion, nonzero-view and nonpacked-save controls
also pass. Raw focused receipts are
`/tmp/sol-hc-access-final-build-cpu-r6.log` and
`/tmp/sol-hc-access-final-tests-r6.log`.

The r70 producer on `spark` passes five functional cases: three authentic
original-DAG HC cells and the existing spatial C7/SiLU and NormAdd controls.
Each HC source is packed F32, width 4096, four streams and rows 1, 2 or 4:
`HcPost -> Reshape[16384,rows] -> RMSNorm(eps=1e-6)`, with both original
post and normalized outputs retained. All four stream/side ports have
distinct payloads and their actual extents. Both default fused and own
alternative plans have verified whole descriptor/bound seals with no missing
proof, closed local certificates and same self-remainder. Each arm poisons
activation storage, restages every port on two changed frames and compares
both complete original-IR CPU outputs.

The unchanged source-backed HC bound is
`abs(error) <= 1e-5 * (1 + abs(reference))`.
The maximum observed `abs(error)/(1+abs(reference))` over both arms is
`1.0245485416499217e-7`; maximum absolute error is `2.384185791015625e-7`.
The default body is one `llmp.dsv4.hc_post_norm_f16` kernel. Its own
alternative is `ggml.dsv4_hc_post` followed by `ggml.rms_norm`, two kernels.
Single-body captures contain two timing-event nodes, giving fused/alternative
counts `3/4` in all three cells. These are functional capture checks; no
ranking samples or timing comparisons were collected and no data rows changed.
All fourteen historical rows miss at backend identity
`6635bdd96567acbcb0ca69d81cee32360e37c847822bb0e5385b467000954914`.

The qualified 123-path source manifest SHA-256 is
`8921af2cfe8f0e9ab12a67b04c13e3a940b5bd1722de02835479bb40fa358314`.
Benchmark source SHA-256 is
`4bb0014e494b5545d6085077768407573205a13d5980de4161a4bf3e0eb84932`;
binary SHA-256 is
`8c00e89c2fbf01294593e840fd75216a0faf48c2625f265cb90b7d76849f875d`.
Raw receipts remain outside Git at `/tmp/sol-hc-producer-r70-summary.json`
and matching build/functional logs.

Both integrated native builds and all **3,971 unique shared checks** pass:
1,982 on `spark` and 1,989 on `spark-b`, with no failures, skips, missing or
extra cases, or duplicates. The two actual ModelOpt/GGUF saved-recurrence
cases on `spark-b` pass without hidden skips. These counts do not include
the DeepSeek frozen screen as a passing model test.

The published-GGUF DeepSeek frozen screen on `spark` remains **FAIL** after
368.355 seconds. Its ten actual/expected failure tuples exactly reproduce
the pre-U6 raw result, and its initial head matches in both runs: all eleven
observations are unchanged. State identity across spill/restore is preserved.
The independent comparison is
`/tmp/llmp-m35-orchestration/u6-resume/hc-final/historical-frozen-comparison.json`,
bound to the current full `a-models-LastTest.log` and historical
`/tmp/llmp-m35-orchestration/m35-u3-fp8-dsv4-old-artifact-control-a-r1.log`.
No frozen values changed and this failed test is not relabeled as passing.
All 123 source paths remain exact at qualification, including an explicit
read-only verification after the failed job.

This is the already documented historical negative in the
[U4 remaining-domain report](../m35-component-numerics/remaining-domain-transfer/RESULTS.md),
whose separate current/reference head/state comparison passes at its recorded
scope. The same target ID `01a5e5ba...` and its current manifest, spec and index
hashes match those U4 pins. That bridge is reused without claiming a new U4
run or extending its scope. Fresh HC source attribution shows no change to
kernel arithmetic, CUDA launchers or capability predicates. Implementation
registration and `Kernel::operation` already use the capability's operation,
so correcting the inventory row does not alter selection. The lowering additions are confined to stateless admission, and the other new
facts serve comparison proof. The current shared
`GgmlLowerDsv4Gpu.HcCarriesAndDownstreamOwnersMatchExistingHelpers` case also
passes, covering F32/F16/BF16 mixing weights, rows 1/4, unequal owners, full
carry and both downstream outputs. Together with the fresh original-IR CPU
cells and unchanged complete model observations, this supports bounded HC
closure without another large model run or an oracle refresh.

The raw/source producer audit is
`/tmp/llmp-m35-orchestration/u6-resume/hc-final/producer-audit.json`;
the integrated audit, retained known failure and exact evidence paths are in
`/tmp/llmp-m35-orchestration/u6-resume/hc-final/results.json`.
A pre-HC baseline build was stopped on hold and is not counted as validation;
no baseline model rerun or 91-GB artifact staging occurred.

This closes the bounded HC validation wave with the historical frozen failure
retained explicitly. U6 and broader shape/format measurements remain open;
this checkpoint establishes no new performance or general quality coverage.


## GDN gates and HC-combine validation checkpoint

The original Verify-phase GDN graph now closes its complete seven-member
pointwise group, with its two real matrix products outside the captured body.
The F32 fixture uses 64 features, 48 heads and rows 1/2/3/4. Rows 1/2/3 select
MMVF; row 4 selects the ordinary direct SGEMM path. Both original outputs,
both complete products and immutable weights, including readable tails, are
checked over two changed input frames against the original-IR CPU executor.
The prospective complete-tensor bound is
`max_abs_error / max(1, max_abs_reference) <= 1e-4`.

Each invocation restages its inputs and prepares the prefix before replaying
one captured body. The alternative legitimately reuses dead alpha-product
storage for the later beta output. Products are therefore checked **before**
the body consumes them. This does not qualify repeated consumption of those
same prepared values. The actual alternative refuses a two-invocation capture
before recording, then passes a fresh single-body replay. The shared capture
API now requires backend-authenticated input preservation for repetitions
above one; the real nonclobbering RMS fixture still passes 128 invocations.

The shared boundary checks retain exact public ports and alias partitions
when a fusion replaces a removed, private, full-span output root. Protected
and kept roots remain strict. The metadata walker accounts for terminal
output reshapes without adding GPU calls or dropping source members. GDN's
two packed output planes retain distinct logical identities and an authenticated
physical storage relation. Direct SGEMM facts include the actual library
configuration and disjoint workspace. The stable descriptor context now follows
the backend's late library loan and returns that borrower on release; declared
and undeclared release/re-lend lifecycle controls pass.

HC-combine uses its original eight-node graph, three independent F32 inputs,
width 4096, four streams and rows 1/2/4. Its actual block reshape is checked
through its existing paid parent and selected source operand. Both schedules
pass whole descriptor/bound, local closure, self-remainder and complete-output
CPU checks over two changed frames. The bound remains
`abs(error) / (1 + abs(reference)) <= 1e-5`.

On `spark`, the final functional checkpoint passes GDN rows 1/4 and HC rows
1/2/4. GDN rows 2/3 retain their earlier numerical qualification, explicitly at
that earlier source identity. Current GDN row 4 has maximum normalized product
error `1.02445e-5` and output error `4.70877e-6`; HC's maximum normalized output
error is `1.65173e-7`. The captured bodies contain 1 versus 4 kernels for GDN
and 1 versus 6 for HC, plus two event nodes. GDN's two matrix-product calls
remain outside the body; library API calls are not counted as GPU kernels.

The functional backend identity is
`d804adf3541d9140f6f236b989d59391cbc493da66527e718af7bb929421f4a2`.
GDN rows 2/3 retain backend
`6dfbee147ab66c51af21b8f6359f5002cd847e97f9440145adbcc8103270364d`.
Focused native and actual CPU-only checks on `spark-b`, including library
lifecycle and repeated-capture controls, pass. Independent whole/adversarial
source review and root review of the raw functional receipts are clean.
Both integrated native builds pass. All 3,980 unique shared checks pass
without skips, failures, missing cases or duplicates: 1,984 on `spark` and
1,996 on `spark-b`. The three image-spatial cases and whole-VAE case on
`spark`, plus both actual ModelOpt/GGUF saved-recurrence cases on `spark-b`,
pass. All 123 source paths verify before and after each build/test job. A
final row-1 GDN probe confirms the receipt-only field correction preserves
the exact key, backend identity, refusal and reference results. The bounded
validation batch is closed; its final audit is
`/tmp/llmp-m35-orchestration/u6-resume/gdn-hc-final/results.json`.

No ranking samples were collected and no rows were added or re-pinned. All
fourteen historical recipe rows miss the current keys. This qualifies these
bounded graphs and contracts; U6's remaining families, shape/format measurements
and production ranking adoption remain open.

## Native whole-program state comparison checkpoint

The ordinary shared-GDN `BuildStep` now compares the genuine default-selected
`gdn_conv`, `gdn_step` and `gdn_norm_gate` occurrences against their complete
own alternatives. Convolution includes its 19 original source members and
history update; history is not a separate rewrite. The fixed decode fixture
uses width 64, head dimension 128, 16 key heads, 48 value heads, four
convolution taps and F32 weights/state. Each arm independently restores both
complete state images, poisons activation storage and stages canonical inputs
before one ordinary or captured invocation. Two changed frames produce 72
complete tensor checks: eight logits, 30,720 convolution-state values and
786,432 recurrence-state values per check. Maximum normalized errors are
`8.94070e-8`, `1.09713e-7` and `2.98023e-8`, respectively, below the existing
`1e-4` bound. Immutable weights and teardown also pass on `spark`.

Native whole-program seals use a distinct restored-state, single-invocation
preparation domain. Descriptor and bound facts authenticate the actual native
schedule, complete physical resources, backend/device identity and canonical
generated-index recipes. They do not grant local Ready-region authority. The
three exact whole-program keys miss all fourteen historical recipe rows; no
samples or rows were added. The qualified backend build is
`a79097c1d6a4b86727ba336ab3afa62c99e8d5d6ca24414fe5e387e1cbe40ada`.

On `spark-b`, the actual production three-store adoption control passes for
alternative, incumbent and neutral rows. The retained adopted owners run
normally and replay their captures at changed positions 1 and 4, checking all
three complete state buffers. The test exposed and now covers forwarding of
generated-input updates through the ordinary reported-program wrapper. A
failed Bind blocks execution and input updates. Fifteen focused native
preparation/key/cumulative controls and 25 actual CPU-only controls also pass.
Normal generated store/output indices validate advertised capacity before
writes, including empty ranges, joined rows and ring wrap; whole-program
proof joins those recipes to actual indexed destinations. Independent whole
and adversarial source review is clean. All 123 source paths verify around
the final jobs. This is functional qualification of these domains; remaining
state/attention/routed domains and shape/format ranking measurements are open.


## F16 resident comparison reference contract

Before qualifying or timing the separately named F16 resident MoE source, its
reference gate is corrected to the F16 arithmetic domain. The initial `1e-5`
ideal-reference gate came from the F32 `MoePiecesMatchTheirReferences` test;
it does not account for the selected grouped F16 products rounding their
activation inputs to half precision. The failed attempts remain failures under
that initial gate. They are not ranking evidence.

The corrected prospective gate reuses the existing F16 product/GLU producer's
`2e-2` maximum-error envelope, normalized by `max(1, max(abs(reference)))`,
against the complete original-IR CPU output. It additionally requires the
complete incumbent and its own alternative to agree within `1e-5` under that
same normalization on both changed frames. Thus a fusion cannot consume the
baseline's F16 precision allowance. Every output must be finite and every
immutable weight part must remain unchanged. The grouped-product precision
basis is the independent a-priori dot-product bound in
`tests/unit/decode_gpu_test.cc`: F32 accumulation error plus
`2^-11 * sum(abs(weight * input))` and the half-subnormal term. Observable
intermediates may be checked against that bound; dead intermediates are never
preserved by altering the source graph. The F32 ideal gaps remain reported.
This changes benchmark qualification, with no change to production arithmetic.


The separately named Q8 resident source uses the existing quantized-product
CPU sanity metric: each complete output tensor, independently for each arm and
changed frame, must be finite and have
`sum((actual-reference)^2) / sum(reference^2) <= 5e-4`. With zero reference
energy the numerator alone is checked, matching the existing `Nmse` helper.
The nontrivial-reference guard remains, as do normalized maximum-error and
worst-element diagnostics. Complete incumbent/own-alternative output agreement
must additionally stay within normalized maximum error `1e-5` on both frames.
This metric/envelope comes from `ggml_ext_ops_test.cc` and
`qwen38_moe_test.cc`; it is not a `5e-4` maximum-error gate. The initial borrowed
CPU-component `1e-4` maximum-error attempt remains a failed attempt. Its final
error does not identify an internal cause: its selected expert calls are
`ggml.mul_mat_id.mmvq`, and no grouped-F16 explanation is asserted. The operator
NMSE allowance supplies a coarse CPU sanity check, while strict peer agreement
checks the preparation change against the production baseline. Neither this
synthetic arithmetic fixture nor the F16 fixture claims model-quality evidence.
These contracts are fixed before their new qualification and ranking runs.


## Admitted numerical-change qualification contracts

Before executing the admitted ModelOpt and GGUF comparisons, the following
complete-output gates are fixed. Both arms use the actual source artifact,
production serving policy and admitted numerical grants; these arithmetic
fixtures do not establish model quality. Each changed frame must produce finite,
nontrivial outputs and preserve every immutable weight part. NMSE means
`sum((actual-reference)^2) / sum(reference^2)`, with the numerator used when
reference energy is zero.

For the ModelOpt NVFP4 expert chain at 17 rows, each arm must have CPU-reference
NMSE at most `3e-2`, peer NMSE at most `1e-3`, and incumbent CPU NMSE at most
`1.05` times alternative CPU NMSE. These are the existing whole MoE wider-prefill
contracts in `qwen38_moe_test.cc`, including routing, both quantized products,
SwiGLU and combination. At one row, the CPU ceiling remains `3e-2`; the peer
NMSE ceiling is `5e-4`, transferring the existing quantized-vector envelope
prospectively to this complete-chain comparison.

For admitted GGUF `vecq_glu`, both CPU-reference and peer NMSE must be at most
`5e-4`, the existing quantized-product sanity envelope. Its production numerical
grant permits changed reduction arithmetic. Unchanged combination and preparation
comparisons retain the stricter normalized maximum peer error `1e-5`. A failure
under these gates remains a failure; no limit is fitted to its observed error.

The admitted BF16 image block uses the existing image-fusion relative RMS ceiling
`5e-3`, independently for each complete output, arm and changed frame:
`sqrt(sum(error^2) / max(sum(reference^2), 1e-300))`. All values must be finite.
Maximum absolute error divided by reference RMS is diagnostic only. The block
retains genuine production image numerical grants and both original outputs.


## Resident, state and EXL3 functional qualification

On `spark-b`, all seven resident F16/Q8 comparisons pass both ordinary execution
and captured replay over two changed frames. F16 `moe_glu` and `moe_combine`
cover one and five rows; Q8 covers one-row `vecq_inputs`, five-row `prepared_inputs`
and five-row `vecq_inputs`. The 56 complete CPU comparisons and 28 peer checks
pass the prospectively stated contracts. Maximum F16 normalized CPU gaps are
`3.96621e-4` and `4.44134e-4`; maximum strict peer gap is `7.23026e-8`. Q8
maximum CPU NMSE is `1.39982e-6` at one row and `3.80205e-6` at five rows, with
zero peer difference. Both sources retain actual router-produced expert IDs and
native expert formats. They establish resident arithmetic, with no paged-route
comparison authority. The failed earlier borrowed-reference attempts remain
excluded from this qualification.

The same host passes complete quantized two/three-store groups, nested F16 and
quantized RopeStore changed-position controls, and genuine full token attention.
Dense one-row and two-owner attention execute 17 and 19 actual calls, respectively,
with complete logits, F16 state and immutable-weight checks after every ordinary
and captured arm. The dense four-row default schedule is identical to one row,
so it is retained as a dispatch control.

On `spark`, the installed Qwen2 EXL3 artifact
`4b995657242c6e74323bc7010b4d2adb3072096eb1688928bbc5d1dbdbbca77e`
supplies all eight actual gate/up parts for the one-row, K896/N4864 MCG4 pair.
The default paired product and its two-product alternative independently pass
complete original-IR CPU checks on both changed frames. Both seals require
ordinary execution; no capture authority is inferred. All 4,864 outputs, both
observable product outputs, immutable parts and readable tails pass. Maximum
propagated SwiGLU error/bound ratio is `0.042608`, below one. Rows 145 and 1,024
are actual default nonselection controls. The backend identity for this
qualification is `4d078529a18ce96e7d54a03130874dfccc1fb99ff274daf2d1f665635585a434`.

The later EXL3 qualification also passes with original F32 inputs, using the
provider's explicit F32 companion and the original F32 CPU oracle. Its actual
alternative retains one F32 companion and one existing converted F16 product,
as chosen by the default planner. All four complete outputs and eight observable
products pass across both arms and changed frames. Maximum propagated ratio is
`0.04899790815` for F32 and `0.04260782617` for F16. Both complete pairs use
ordinary execution. This later backend build is
`80e1bdbd160ae2d53dc4f722b0fc055fc6b4356fb7963d492267404f8554fe6b`;
twenty unique selected-source checks pass without skips.

On `spark-b`, the admitted NVFP4 expert cell at one row passes both ordinary and
captured complete programs over both changed frames: eight CPU checks and four
peer checks, with maximum NMSE `5.273975487942178e-4` and
`4.226846257787656e-4`, respectively. The original native parts and full prepared
slab remain unchanged. Actual captured programs contain 17 and 26 nodes. This
functional run used the temporary diagnostic build
`9539ca689bbf06fa3742c4cb224a7fd183032df3184c7d89254e08027e6d8795`;
its key is not installed or transferred to another build. The quality-rejection
path independently completes both ordinary GDN arms and captures only the
passing alternative, preserving the failed incumbent and peer outcomes. Its
production controls pass eleven native checks and ten generator checks, including
whole-quality precedence over a competing local timing row.

These are bounded functional qualifications, with normal teardown and independent
raw-result review. All exact keys miss the fourteen historical recipe rows. No
new ranking samples or rows were produced by these runs. Remaining image and
specialized execution, current-key measurements and the final integrated gate
remain due.


## Prospective specialized whole-source reference gates

Before executing the admitted MXFP8 HC→GDN→HC and HC→QSA→HC fixtures, each
complete original output and each complete state tensor must independently
pass CPU-reference NMSE `3e-3`. Each actually written state region must also
pass independently, preventing unchanged cache rows from diluting errors.
The synthetic Q8 DeepSeek specialized fixture uses `5e-4` for the same metric.
NMSE uses squared error divided by reference energy, or squared error alone
at zero reference energy. All decoded values must be finite.

These are prospective finite-fixture sanity gates grounded in the existing
MXFP8 and quantized-product controls (`qwen38_fast_test.cc:771–777` and
`ggml_ext_ops_test.cc`). They are not mathematical composition guarantees or
new model-quality grants. The source's genuine production numerical eligibility
remains mandatory. Same-semantics structural comparisons additionally require
complete incumbent/own-alternative normalized maximum gap `1e-5`, using
`max(abs(incumbent-alternative)) / max(1,max(abs(alternative)))`. Actual
numerical-changing sites require a separately stated contract.

Initial native state is decoded from the exact uploaded bytes, and CPU stores
retain the actual destination encoding. Untouched cells, padding, readable tails,
disabled writes and immutable weights remain byte-exact. Written subsets come
from original state-write/index semantics, including writes that happen to retain
the old value; observed byte differences are not a write mask. Existing F16
formula bounds are not extra error allowances on an already-rounded reference.
Integer protocol controls remain exact where the original-input oracle applies;
no general route-ID equality claim is inferred after newly quantized scores.
Any failure under these gates remains a failure, without fitting its threshold.


For actually selected `hc_prep`, `mxfp8_inputs` and `bf16_rows` numerical-changing
comparisons, fix the separate whole-peer gate prospectively at NMSE `3e-3`,
independently per complete output, state tensor and written state region. The
CPU gate remains `3e-3`. This deliberate finite-fixture transfer uses the existing
MXFP8 ideal-activation sanity envelope; isolated HC/BF16 rounded-operand tests
do not prove that it composes. Genuine production numerical grants remain
mandatory, and structural sites retain their stricter peer gate.

Use the actual accepted match's retained `Applied` numerical change and genuine
noncandidate proofs when choosing these gates. The registration table supplies
only a default: `prepared_head_gate` can declare `Mxfp8ProductInput` dynamically,
and `grouped_vecq` can declare `Q8VectorReduction`. The former uses the same
prospective admitted-MXFP8 chain CPU/peer NMSE `3e-3`; the synthetic Q8 DeepSeek
chain uses CPU/peer NMSE `5e-4` for its actual Q8 numerical-changing sites.
Neither a pattern name nor a missing table default establishes structural
equivalence or numerical eligibility.

QSA initial state also preserves the existing pooling invariant: previously
completed pooled blocks agree with the exact uploaded raw-key history, norm,
RoPE and BF16 rounding. Its fused pool may update only newly completed blocks
while the original graph recomputes earlier blocks. Original logical write
indices define the reference region; this does not claim every implementation
physically rewrites that region. Independently random, inconsistent histories
are not valid source inputs for this comparison.

## Conditioning projection reference correction, before further execution

The first admitted image R17 run fails the originally assigned `5e-3` relative
RMS limit on output 1: `0.003763215` on frame 0 and `0.008091754` on frame 1.
The failure is retained. Source inspection identifies this output as the extra
conditioning `kNormOutWeight` projection exported by `ImageTransformerBlockSpec`,
not a final image-flow output covered by `ImageFusionGpu::FlowCase`. Its ordinary
MMF prefix rounds F32 activations to BF16 before each product; the original CPU
graph uses those F32 activations directly. The first tested `split_modulate`
rewrite does not feed this output. Both arms have the same frame-0 error.

Before the fixed-payload rounding diagnostic or another quality run, fix this
separate conditioning contract. Preserve the original F32 CPU oracle, payloads
and the `5e-3` complete-flow gate on output 0. For every conditioning output-1
element, propagate an absolute forward-error envelope through the actual
time-linear1, SiLU, time-linear2, SiLU and final projection dependency chain.
Weights are the exact encoded BF16 values. Begin with input error `e_i=0`.
For each length-`k` product, with `u_b=2^-8`, `u_f=2^-24`, use
`gamma=2*k*u_f/(1-2*k*u_f)`, `u_d=2^-53`,
`gamma_d=2*k*u_d/(1-2*k*u_d)`, and

```
d_i = (1+u_b)*e_i + u_b*abs(x_ref_i) + 2^-126
E_j = sum_i abs(W_ji)*d_i
    + gamma*sum_i abs(W_ji)*(abs(x_ref_i)+d_i)
    + (u_f+gamma_d*(1+u_f))*sum_i abs(W_ji)*abs(x_ref_i) + 2^-126
```

This accounts for propagated input error, BF16 input rounding/flush, FP32 dot
roundoff and the reference's double-dot/F32-store roundoff. Before accepting
the envelope, require finite BF16 input rounding and the gamma-expanded native
dot magnitude within the finite F32 range. These conservative guard corrections
were fixed in the independent source/math review before further quality runs.
For either SiLU use
`E_out=1.2*E_in+2^-20*(abs(silu_ref)+1.2*E_in)+1e-30`, using the existing
propagated-SiLU derivative/arithmetic margins. The derivative bound is analytic;
the arithmetic allowance is a prospective implementation margin, not a universal
`expf` accuracy theorem inferred from its source. Require finite envelopes and
every absolute output error at most its own envelope. No value in this bound
depends on observed GPU error. Separately report the exact BF16-rounded-prefix
prediction against both the original CPU and GPU outputs on the same payload.
It is a diagnostic, not a replacement quality oracle. This corrects a mistaken
reference-contract transfer; it creates no additional production numerical grant.

## Numerical rejection before timing qualification

The admitted MXFP8 GDN prefill cell (derived width 128, 17 rows,
`mxfp8_inputs`, two fixed changed frames) fails the prospective `3e-3` NMSE
contract. Frame 1 incumbent logits have NMSE `0.0034330914447063174`;
the own alternative has `0.0029641064533`, and their peer NMSE is
`0.0066167833332`. Complete and logically written states pass in both
frames, as do finiteness, original encoded-state, immutable-weight and
untouched-byte checks. A diagnostic collected both arms before reporting the
failure; it collected no timings or captures and does not qualify the pair.
The original failure and fixed limit are retained.

This result rejects the incumbent for this exact complete recipe. It does not
revoke numerical eligibility for other shapes or sources. Before collecting
replacement evidence, the production decision record will distinguish a
quality rejection from a latency comparison. A quality rejection carries zero
timings and zero timing samples, its prospective protocol and evidence digest,
and the same complete source, policy, preparation, physical-recipe and backend
identities required by ordinary whole-program adoption. Both changed frames
must complete both ordinary arms; the own alternative must pass every CPU
quality check, and all nonnumeric guards remain mandatory for both arms. A
failed quality check is recorded explicitly, without widening the limit. Only
a current exact authenticated record may select the passing own alternative;
unknown or changed identities retain the ordinary fallback. No latency is
invented for this cell, and historical evidence is not re-pinned.


## Prospective ungranted Q8 GLU cell

The existing native `AutomaticPairedProductsNeedQualificationAcrossTheCompleteRegistry`
fixture supplies a concrete default-policy `mul_mat_glu` source: two
Q8_0 matrices of shape `[896,4864]`, one shared F32 `[896,1]` input, and
SwiGLU. The earlier `vecq_glu` rewrite requires a numerical grant, so this
ungranted source selects the ordinary quantized GLU fusion without disabling
policy. Before execution, fix CPU NMSE at `5e-4` for the complete original
output and peer normalized maximum error at `1e-5`, using the existing Q8
contracts above. Both changed frames must pass, decoding the actual encoded
weight bytes for the CPU reference and preserving every part and readable tail.
This cell adds neither an EXL3 claim nor a new numerical grant.


## Prospective admitted MQA decode cell

The existing admitted DeepSeek MQA source supplies its actual width 4096,
one decode row, past 256 and 512 read cells, with its complete HC/MQA/HC
assembly, original component parameters and native encodings. Use its actual
serving policy and numerical admission; no diagnostic candidates are permitted.
Before execution, fix complete output, complete state and logically written
state CPU NMSE at `5e-4`. Structural pairs retain peer normalized maximum
error `1e-5`; an actual accepted `Q8VectorReduction` declaration uses peer
NMSE `5e-4`. Old cache/ring images are independent finite native-encoded inputs
to this state transition, with exact aliases, indices and untouched bytes.
This does not assert that those images came from a particular token history.


## NVFP4 prefill attribution before further qualification

The existing width-128, 32-expert, 17-row admitted NVFP4 cell reaches both
whole-program seals, then fails the original `3e-2` CPU NMSE bound at incumbent
frame 0: `0.09904181308210103`. This failure is retained. The same explicit
untimed attribution mode will collect both original arms and changed frames,
keeping the payload, shape, CPU `3e-2`, peer `1e-3` and `1.05` nonregression
checks unchanged. A separate protocol records the outcomes. It may emit a
quality rejection only if the incumbent fails and the own alternative passes
every CPU check and its permitted untimed capture. If the alternative also
fails, the run reports no record. No latency, replacement limit or wider
quality claim follows from this diagnostic.


The unchanged NVFP4 prefill attribution run qualifies an exact quality rejection:
incumbent frame NMSE is `0.0990418131` / `0.0747515630`, while the own alternative
passes at `1.4101423e-5` / `1.4606745e-5`, including its untimed captured replay.
The peer NMSE (`0.13480849` / `0.08619348`) and nonregression failures remain
explicit. All immediate guards and teardown pass. Protocol
`e40394c4f95ecfaed0ebecd19d967e7e691c35ff3e0c340b981956fb2ad5ddc5`
records six complete CPU outcomes, four peer outcomes and zero timing samples.
This functional evidence is not re-pinned to the later shared closure build;
installation requires a fresh exact key and receipt there.


## Complete image, Q8 and GGUF functional qualification

The shared native closure passes seven complete selected image pairs at both
10 and 17 rows. Each row count checks 112 original output results and 28
captured executions across both implementations and both changed frames.
The final selection inventory distinguishes rewrites consumed by later fusions
from independent comparisons: two tickets are consumed at 10 rows and four at
17 rows. The complete-flow output retains its `5e-3` gate; the separate
conditioning projection passes its prospective elementwise forward bound.

The ungranted Q8 GLU cell passes all eight complete ordinary/captured output
checks against its exact encoded-byte CPU oracle and strict peer bound. Both
admitted GGUF routing and GLU pairs pass eight CPU and four peer checks each,
including four captures. Immutable weights, readable tails and teardown pass.
Both native builds and four focused native-closure controls pass. These are
functional results at their recorded source cuts; current ranking installation
still requires the final frozen identities and fresh measurements.

The admitted MQA decode `sinkhorn_hc_pre` pair also passes ordinary and captured
execution in both changed frames: 24 complete output/state CPU checks and
12 peer checks, with four captures and normal teardown. The CPU NMSE maximum
is `1.63745445e-4` against `5e-4`; structural peer error remains below `1e-5`.
Full native state, logical writes and unchanged bytes are checked separately.


## Shared quality failures in specialized whole sources

The admitted MQA `grouped_vecq` pair passes a further 24 CPU and 12 peer
checks, including both ordinary arms and four captured replays. Its maximum
CPU NMSE is `1.637454454116224e-4` against `5e-4`.

The fixed DeepSeek 64-row `state_bank_attention` comparison reaches verified
whole descriptors and binding with its declared 36,700,160-byte main pool.
Untimed attribution finds identical native arms in both frames (all 30 peer
checks are zero), but both frame-0 logits have CPU NMSE
`0.004916275935128997` against `5e-4`. Frame 1 passes, as do all 56 complete
and logically written state checks (maximum NMSE `1.990140083947512e-6`).
This is a shared baseline failure, not evidence against the state-bank rewrite.
No capture, latency or quality-rejection row is qualified.

Two source-bound diagnostics locate the shared discrepancy before the final
normalization. The native F32 output projection matches an independent dot
product of its actual input (NMSE below `7.2e-14`). A live snapshot then shows
the final RMS normalization matches an independent calculation from its actual
input and gamma (NMSE below `7.0e-15`), and its output exactly matches the
projection input. The weighted HC sum entering that normalization already
differs from the original reference. These diagnostics grant no quality or
ranking authority; temporary source changes were removed and production source
hashes rechecked. An exact inverse-RoPE projection comparison then retains
the same frame-0 failure in both arms (`0.00491627594` / `0.00492443518` NMSE).
Removing that selected projection does not remove the shared discrepancy.
Its structural peer check also fails; no timing or ranking row follows.

A live final-HC diagnostic then compares its complete gathered residual and
replays the HC calculation from that native input with the original F32
parameters. The residual already differs from the reference (NMSE
`0.00607385205` / `0.00352517259`), while the independent HC sum matches
the native sum below `6.2e-15` NMSE in both frames. This rules out the final
HC calculation as the source of the shared discrepancy. Both arms retain
the same failure and no ranking authority.

Both fixed admitted GDN17 `bf16_rows` occurrences pass genuine whole-only
checkpoint capture, descriptor and bound proofs. Untimed attribution checks
40 CPU and 20 peer outcomes across both pairs. Complete and logically written
states and all peer checks pass, but both frame-1 logits fail CPU NMSE `3e-3`:
the incumbent has `0.0034330914447063174`, and the alternatives have
`0.0030530695581163993` and `0.0033994516373751183`. Neither pair qualifies
a latency or quality-rejection row. A previously qualified whole-program
MXFP8 decision remains terminal; it supplies no local authority for a later
BF16-row choice. Its normal continuation may retain a smaller prepared-product
group, so these outcomes do not establish that BF16 rows disappear globally.


## Prospective QSA block-cache validity correction

The first native pooling comparison after the BF16-store repair reaches both
whole-program descriptors and binding, then fails the original full-cache CPU
comparison at block 1,754, channel 84. With past 7,000, 17 new rows and ratio 4,
that block is incomplete. The failure remains retained; it produces no rank.

The shared QSA contract writes a block key when the block completes
(`components/qsa.cc` and `QsaPoolWritesTheBlocksAChunkCompletes`). Native pooling
leaves incomplete and future keys unchanged. The primitive expansion can
materialize them; selection masks them until completion. They are different
valid cache representations. Before retrying, fix the following obligations,
without changing input bytes, shapes, numerical bounds or other state checks:

- Authenticate the actual selected native pool, its original typed BF16 block
  state, ratio, owner positions, block/store-cell recipes, raw-cache update and
  absence of another block writer. Pattern or source labels grant no exception.
- Compare every completed row `[0, floor((past + rows) / ratio))` against the
  original CPU result and its peer, retaining the existing metrics and bounds.
  Separately check the newly completed rows starting at `floor(past / ratio)`.
- For a native pool, require every other cache byte to equal its uploaded reset
  byte exactly. For a primitive store, retain the complete CPU state, declared
  writes and exact untouched-byte checks. Apply cross-arm comparison only to
  completed keys, including their separately checked newly written subset.
- Keep every other state and output obligation unchanged. Reuse the existing
  continuation control to verify that the incomplete row is overwritten with
  its correct key when a later invocation completes it.

Thus every byte still has an explicit check. This correction grants neither a
numerical allowance nor a general exemption for invalid or unobserved state.
The later qualification records its source and protocol separately from the
retained failed run.


With the source-backed cache obligations in place, the fixed admitted QSA17
mean-pool pair still fails the unchanged `3e-3` CPU bound in both ordinary
implementations. Frame-0 completed and newly written key NMSE is
`0.00403703229` / `0.00404274378`; frame 1 also fails the shared raw-key state
at `0.00315352113`. All four cache-obligation receipts, immutable data,
finiteness and exact unwritten-byte guards pass. The completed-key structural
peer bound also fails (maximum normalized gap `0.00245098`), and is reported
without adjustment. This is a numerically unqualified pair: no capture,
latency or quality-rejection row follows. Other comparisons with this identical
failing incumbent are blocked by its quality prerequisite; repeating that
incumbent does not qualify another fusion. The separate existing synthetic QSA source uses BF16 HC matrices and MXFP8
attention matrices. It retains its original gates and receives independent
qualification; it does not establish BF16-attention coverage.


## Prospective QSA preparation attribution and mixed-source comparisons

The existing synthetic QSA source at width 2,560 and 17 rows uses BF16 HC
matrices and MXFP8 attention matrices. Its `hc_mix` and `bf16_inputs` pairs
pass 72 original CPU results, 36 peer results and four captured replays each,
with CPU NMSE at most `3.78983221e-5`. The source inventory selects the
existing `3e-3` CPU gate; an independent audit also passes the stricter
prospectively stated `5e-4` condition. Retain both conditions for the three
additional occurrence-zero comparisons on this unchanged source: mean-pool,
interleaved norm/RoPE and block-sparse attention. Structural peer error retains
its `1e-5` normalized-maximum bound. This is independent source coverage,
not a BF16-attention domain or qualification of the failed admitted source.

One additional attribution targets the actual shared MXFP8 preparation upstream
of that failed source's raw/key caches. Use only the existing admitted QSA17
`mxfp8_inputs` occurrence zero and its complete own alternative, requiring the
actual `Mxfp8ProductInput` declaration and admitted source proof. Keep CPU NMSE
`3e-3`, the actual declared peer gate, original payloads, both changed frames,
completed-block-cache-v1 and all immediate ownership, finiteness, immutable and
exact unwritten-byte checks. Collect both ordinary arms before deciding.
Only an incumbent failure with every alternative CPU check passing may proceed
to the existing alternative-only untimed capture checks and qualify an exact
whole-program quality rejection. Both-arm failure produces no record; neither
outcome permits timing or a different bound. This tests whether shared input
preparation explains the already observed failure without repeating unrelated
comparisons with the same failing incumbent.


The admitted QSA17 `mxfp8_inputs` occurrence-zero attribution retains both-arm
CPU failure and emits no record. Its genuine `Mxfp8ProductInput` declaration
and source proof pass. Removing that preparation clears the frame-0 completed
block-key failure in the alternative, but both frame-1 raw-key states still
fail at `0.00315352112838` against `3e-3`. Ownership, finiteness and exact
unwritten-byte guards pass. No capture, timing or quality rejection follows.

The existing synthetic mean-pool comparison passes the additional `5e-4`
CPU condition but fails structural peer error (`0.0025` against `1e-5`).
It remains unqualified. The interleaved and block-sparse own alternatives
exposed shared fallback defects before execution: a padded physical row extent
was published as the logical extent, and mixed F32/F16 addition had no admitted
primitive. Their repairs and focused regressions remain prerequisites to
qualification; these failures are not converted into performance decisions.


## DeepSeek projection attribution closed

One final diagnostic reads the actual attention input, generated inverse-RoPE
table, out-A result and out-B input/output for both ordinary arms and both
changed frames. Original source/weight roots, complete physical geometry and
paid activation extents are joined to the actual selected calls. Independent
FP64 products decode the original Q8 weights. The staged replay also uses
the actual F16 conversion and the pinned MMQ D4 quantizer's output.

Out-A versus the ideal product of its actual live input has NMSE at most
`1.28390722e-6`; out-B has at most `1.13607768e-6`, including separate final-row
checks. Both satisfy the existing `5e-4` local product limit. Independent
staged replay matches native out-A below `1.52e-11` and out-B below `1.11e-13`.
Both arms have identical live payloads. The attention input already differs
from the original whole-source CPU reference; this is inherited disagreement,
not an isolated attention-kernel test or proof of its cause.

The fixed DS64 source remains numerically unqualified under its unchanged
whole-program bound. It emits no capture, timing, quality rejection or ranking
row. This bounded attribution is closed without fitting a new tolerance or
substituting a passing source. Temporary source, header, CMake changes and
payload snapshots were removed; all 142 production paths were rechecked.
Raw receipt SHA-256 is
`d89c9b48b293bf85402c3fcd63d81dd010aeb491bb16514f38de3fd1e17badaa`;
the diagnostic source manifest is
`c7766ca176d63f695f46d6242b23346ad08e52945b6bdb8225224397aa4a3f2a`,
against prepared GGML tree
`b1c096f614feada4bea2e21423e8c83eb05fde9e97da56ae15c876a0254603a6`.


## Shared fallback repairs

The interleaved alternative exposed a head/token layout mismatch. Its restored
packed input is `[256,24,17]`: 24 heads and 17 tokens. A constructor expecting
a two-dimensional token matrix read the head count as the token count. An
initial padded-two-dimensional-parent repair passed direct controls but refused
this actual three-dimensional input before execution. The corrected lowering
must derive a zero-offset token-matrix view from the authenticated logical
shape and actual head/token strides, retaining the complete paid parent. This
shares one contract with genuinely interleaved or padded physical storage;
native qualification is recorded below and in the final U6 results.

The sparse alternative requires F32 scores plus an F16 mask. The old native
broadcast dispatcher accepted that combination but cast the mask to `float*`.
Patch 0012 instantiates the existing float/half/float template in both ordinary
and fused broadcast entry points. Capability and bound validation admit the
additional addition with per-operand widths, alignment, strides and overlap
checks; mixed multiplication receives no new admission. Controls cover
contiguous, strided, broadcast and two-byte-aligned masks, immutable inputs,
wrong types, strides and aliases. Root and independent whole/challenge reviews
are clean. The native controls and unchanged mixed synthetic QSA source pass as recorded below.


The corrected stride-derived lowering passes six focused native controls with
no skips. The unchanged synthetic QSA `interleaved_norm_rope` occurrence zero
then passes 72 complete CPU checks, 36 peer checks and four captured replays
across both changed frames. Maximum CPU NMSE is `3.7898322010822035e-5`,
passing both its source-selected `3e-3` bound and the additional prospective
`5e-4` condition; peer error is zero. Immediate guards and normal teardown
pass. This qualifies only that existing mixed BF16/MXFP8 source, with no
transfer to the failed admitted source or mean-pool pair.


The sparse alternative next exposed an original F32 block-mask input that
only the F16 host stager could represent. The shared mask recipe now emits
either declared encoding with the same cell visibility, ring, frozen-read,
ahead, window and padded-row rules. Slot extents are checked at their actual
two- or four-byte element width; visible entries are zero and hidden entries
are negative infinity. Normal transaction staging supports both kinds. No
conversion or numerical allowance is added. Independent source review is
clean. Native staging, declared mask dtype/row controls and the complete sparse construction are qualified below; its numerical refusal remains explicit.


## Final shared fallback qualification

The final sparse-source construction reaches both genuine whole-program arms,
with 38 incumbent and 62 alternative scheduled steps. Shared fixes preserve
native BF16 batched products by flattening only their contiguous batch axes and
restoring the logical output view; causal masks use their original F16/F32 dtype
and exact or explicitly padded row extent. Native scalar copies accept supported
strided F16/BF16/F32 sources with checked owner extents. A staged host mask is
elided only after authenticating its original generator and actual input slot;
region boundaries follow actual GPU calls. Fill creates one native output tensor,
removing an unused, unfunded template while retaining strict library-closure
checks.

Independent source and adversarial reviews are clean. Focused native controls
pass for BF16 batching, mask dtype/rows, late staged masks and launch boundaries,
Fill, and native strided copies. The strided-copy GPU control covers all three
supported conversion directions, two nonzero-offset/gapped/permuted source-view
orders and two changed frames, checking complete outputs and immutable source,
gaps, tails and output guards. No new kernel or extra attention copy is added.

The original sparse comparison remains numerically unqualified: on frame zero,
incumbent CPU NMSE is at most `3.789832201e-5` and alternative logits CPU NMSE is
`1.321074091e-7`, but the first logits structural peer normalized maximum is
`1.9566002287e-5`, exceeding the unchanged `1e-5` limit. Execution stops before
alternative state/frame-one qualification or capture. It receives no timing,
quality-rejection or ranking row. Its raw receipt SHA-256 is
`c931410ba1e4352ee0ed1fefe382339595351061447eb6a1187bcd3e1af44e7a`.
This is the final bounded disposition; passing construction does not convert it
into a passing numerical result. Final current-key measurements and installed
production checks are recorded in [the U6 final results](U6-FINAL-RESULTS.md).


### Final production regression repairs

The closeout gate found two actual shared regressions. Converting the immutable
MoE shared gate after the Route anchor prevented its existing fusion. The
component now materializes that same small native parameter before routing;
three unchanged native selection checks and six full-reference F32/F16/BF16
parameter cases pass.

A first joined Verify with two owners also exposed an overly exact mapped
state-read check. The authentic GDN history rewrite at `new_rows >= taps`
retains the same state store but no longer reads obsolete history. Original
source declarations remain exact. A separate mapped-access check permits only
removed reads, forbids introduced reads, and preserves exact writes, root/ref
identity, complete inventory and borrowed/stage/tap authorization. The actual
two-owner Plan regression checks both sides of the threshold plus new-read,
changed-write, missing-store and wrong-owner refusals. Both original ModelOpt
and GGUF recurrence scripts pass all 33 outputs, 20 checked commits and the
required replays with the broad operation-discovery table present.

Three stale test fixtures were repaired without relaxing production or model
assertions. A warm source-inventory check was restored by recompiling only the
31 existing objects still referencing the previous prepared GGML tree. The
[shared-support pass](U6-SUPPORT-CLOSURE-RESULTS.md) preserves all preceding
measurements and failures; the final refresh uses new exact build identities,
never repinned evidence.
