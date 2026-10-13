<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Owned joined Qwen target/MTP recovery screen

2026-10-12, Spark B, main candidate based on `28d99ebd`. This is actual
shared-engine execution through ComponentStepSource, ModelRunner, WaveComposer
and DecodingOrchestrator. Ordinary production serving remains unopted.

The prepared target is `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`;
its MTP companion is `8600a99819ce583a719ebfb457de8cac40b4d0bd1ebe557ceb13dff5961aee40`.
The installed selected head has 47,172 rows, below the requested 65,536-row
ceiling. The immutable I32 map carries every selected row into the target
vocabulary; its SHA-256 is
`cfbfa2466c5c2b0246cdf62ca0ca2b6e7b94f72091fe650f62004909b1ac14b3`.
Ordinary truncated heads use their actual width with identity token mapping.

The physical v0 PLE table is U8 data plus a separate F32 tensor scale. The
source gate validates their complete inventory with the existing artifact
assembler against the logical NVFP4 row format. It retains exact v1 checks,
part/geometry/extent checks and the table's nonresident group requirement.
No forced intermediate rounding or new arithmetic policy is introduced.

## Observed controls

The full native build and all 306 selected host controls passed without skips
in `qwen-joined-mtp-main07`. The actual model registration
`Qwen38JoinedMtpGpu.OwnedUnequalPendingRestoresAndChangedInputCapture` passed
in `qwen-joined-mtp-gpu03`:

- Two owners with unequal pending ranges; 13 completed units and 132 finite
  full-head rows. Thirty reference rows and 60 restored rows are checked.
- Four checkpoint-restored owners, two spilled/restored owners, complete
  initialized-state checks and 30 changed-input head rows.
- Three captures and 33 replays. Changed inputs reuse the recorded program
  and produce different heads without another eager run.
- Thirty-five independently measured classes, complete physical admission
  91,628,901,123 bytes below the unchanged 104 GiB node cap, 12 GiB prepaid
  host allowance and 341,835,776 bytes per paid state-copy capacity.

The case took 20.499 seconds; this includes setup and storage and is not a
serving-performance measurement. Source, binary, artifact-metadata and recipe
guards passed before and after. The child process group retired, no GPU process
remained, and all memory limit/OOM event counters stayed zero. The cgroup's
2,156,793,856-byte process/page-cache peak is not total device residency.

Native recipe SHA-256: `d2ffd6efc15d7eaa589c608802f0606167898504c1e18ef997558f41ca4bd0dc`.
Actual recipe manifest SHA-256: `8158f99d116ad7a666179a6aef57ea782dd084f0e3b01bbcd3bd87f3b06b1434`.
Raw logs/XML remain outside Git under each host's session scratch.

This establishes same-implementation state restoration and owned-domain
execution. Broader quality and behavioral coverage against a reference of the
same quantized checkpoint, durable old/new restart compatibility, production
serving and matched family speed/swap remain open. The original Qwen
source-precision model remains absent from inspected local and NAS catalog
evidence, limiting original-agreement claims without blocking that quantized
baseline. Cross-library bit equality is diagnostic, not a quality requirement.

## Isolated commit check

The Qwen-only candidate excludes the in-flight DeepSeek bridge and reader-budget
changes. Its full native build and all 305 affected host controls passed in
`qwen-joined-mtp-main09`; the unchanged actual registration passed again in
`qwen-joined-mtp-gpu04`, with complete guards, retirement and zero memory-limit
or OOM events. The single removed host registration belongs to the unfinished
DeepSeek fixture. A final prose correction updates the stale pending status;
production and test code are identical to this checked candidate.

Isolated native recipe SHA-256: `94a970acb410e7e22b14d14eed280d7c0bdadb247708a861163bd5f8365b021f`.
Isolated actual recipe manifest SHA-256: `d6dca445587b2752f7d8d4d6be77489e7cc06cafe1457e0d0511a566b665b48e`.

## Fresh-owner durable C2 continuation

2026-10-13, Spark A, candidate based on `cb8bed69`. The manual
`Qwen38EngineServingGpu.NativeC2DurablePendingReopen` control uses the real
EngineLlm, StateKeeper and existing snapshot/slot lifecycle with the same
artifacts above. Two conversations begin with seven and five prompt tokens.
Each generates four tokens, saves its target and MTP boundary, and continues
to eight total tokens as the uninterrupted oracle. After restoring the
four-token boundary, both stage files per conversation are durably preserved. The complete first owner retires
before a fresh owner adopts and hydrates those four files.

Both cold conversations refuse execution before hydration. Hydration restores
the saved history and repeat-pending progress without executing a model prefix.
The reopened continuation produces eight fresh full target heads across the
two conversations. All 16 accumulated rows compare exactly, including eight
retained prefix rows; only the eight continuation rows are recomputed after
reopening. Boundary and final initialized-state comparisons include stage,
region, offset and length coordinates alongside complete payload digests.
Tokens and speculative acceptance counts compare only within this same selected
implementation. Existing host controls retain changed-identity/legacy-record
refusals and partial-hydration refusal/retry coverage.

The target disables the mapping's `owner_row_invariant` restriction; the drafter
retains its existing replay mapping, including that restriction. Other mapping
restrictions remain unchanged. No explicit product tuning or legacy owner-product
requests are added. Initial speculative prompts use the existing individual
session/companion path; generation uses joined waves. This is an experimental
owned policy, without changing Server's cohort admission or any default policy.

Each owner admits 953,736,576 bytes for measured retained/replacement programs.
Its complete physical closure is 86,567,109,091 bytes under the 104 GiB node
cap; each snapshot's complete payload/metadata ceiling is 341,842,075 bytes.
The current compiler inventory prepays 4,946,022,912 bytes, with separate
existing serving metadata and 128 MiB of observations. Positive retirement
separates the two model owners; their model domains do not overlap.

The focused native build and 126 host controls passed. The final fixture-only
rebuild reran four affected controls and authenticated the other 122 unchanged
source/binary results. The actual case passed in 38.454 seconds including
setup/storage, not a performance measurement. All source/binary/recipe guards
passed, the process group retired, no GPU process remained, and every memory
limit/OOM counter stayed zero. The 2,308,546,560-byte process/page-cache peak
is not device residency. Main integration passes a full Spark-native build and
all 126 affected host controls with no skips. The manual case is inventoried
once and remains absent from automatic CTest discovery. Pinned format, boundary,
whole-source header and REUSE checks also pass. The final merged binary repeats
the complete actual control successfully in 38.573 seconds with the same
resource envelopes, positive retirement, zero remaining GPU processes and zero
memory-limit/OOM events. Its process/page-cache peak is 2,309,382,144 bytes;
this remains a functional check, not a performance measurement.

Replay this built, manually supervised case with `LLMP_QWEN_C2_DURABLE=1`,
`LLMP_TEST_MODELS` pointing to the authenticated installed model root and
`LLMP_TEST_SCRATCH` pointing to a fresh private directory. Invoke
`qwen38_engine_serving_gpu_test` with
`--gtest_filter=Qwen38EngineServingGpu.NativeC2DurablePendingReopen` under the
existing exclusive Spark supervisor. It is absent from automatic CTest discovery.
The reviewed run uses a 112 GiB, zero-swap process/cache ceiling, 6 GiB physical
margin, 512 MiB per-file spill limit and 8 MiB log/XML/report limits.

Actual recipe SHA-256:
`12a4a2c421340cb10b8f4af6f13e44255e9274317776edcfd1251a6e70dc9705`.
Actual result SHA-256:
`ef29fe035c27827e5b5556c85040848a1d6a615fae2dc29903e29352707f87b1`.
Raw logs/XML stay outside Git in `qwen-c2-durable-models04` session scratch.
This closes the bounded fresh-owner durable continuation control. OS-process or
GPU restart, cross-format state conversion, broader same-quantized-reference
quality, joined production serving and family speed/swap gates remain separate.

Final-main actual receipt SHA-256: `3d80387174e6f8e2bd35ce1b04637bff1bb8cc939f93ac883dcb358076ec5227`.

## Internal Server C2 defaults and reopen

The internal shared Qwen selector now derives its immutable execution domain
from the resolved composition, state layouts, slots and phase bounds. It uses
the existing Server, ComponentLlmBuild, backend declaration and source seal;
explicit caller domains keep their existing meaning. Domain storage is paid
before construction and retained with its borrowers. Ordinary public model
selection is unchanged. This route uses the existing mapping defaults,
including `owner_row_invariant`, rather than the experimental target policy in
the preceding durable control.

A bounded actual Server control passes on Spark A (2026-10-13): context 512,
prefill chunk 16, two slots, draft depth two and read alignment 256. Unequal
55/53-token prompts retain the default lookahead, preparation, capture and turn
checkpoint options. Individual prompt advances save one checkpoint per branch;
joined generation completes four tokens per branch. After spill, keeper drain
and positive whole-owner retirement, a second Server adopts both conversations
and checkpoints. Hydration reuses each complete prefix without a new prefix
execution, preserves the pending drafter boundary, and completes four more
tokens per branch: eight fresh, finite full-vocabulary continuation heads.
The earlier same-selected boundary/final state oracle remains separate evidence;
this Server control does not add a second state-byte or uninterrupted-head oracle.

The run observes two built and captured prompt futures. Preparation remains
enabled but needs no additional backing for this short read256 schedule
(observed prepared count zero). Its recorded plan floor is 14,591,840 bytes;
actual retained programs and future captures use existing plan accounting.
The source-derived startup bound is rounded upward to the Server's required
2 MiB catalog quantum, inside the unchanged 104 GiB node cap. The wrapper keeps
112 GiB/zero swap, the 6 GiB physical margin, 512 MiB spill-file and 8 MiB
log/XML/report caps. Process-plus-page-cache peak is 1,557,827,584 bytes; this is
not a device-memory measurement. All source/binary guards, completion retirement,
zero memory-event and empty-GPU-process checks pass.

The actual native case takes 24.711 seconds including setup/storage; no latency
claim follows. Six current fixture host controls pass, with 122 unchanged
controls authenticated by their existing binary receipts. Final-main integration
passes the full Spark-native build and all 128 affected host controls, with the
manual case inventoried once and absent from automatic CTest discovery. Pinned
format, boundary, whole-source header and REUSE checks pass. The final model
binary exactly matches the actual-case binary, so no inference repeat is needed.
The case is manually supervised:
set `LLMP_QWEN_C2_SERVER=1`, the existing model/scratch paths, and select
`Qwen38EngineServingGpu.NativeC2ServerDefaultOptionsReopen` in
`qwen38_engine_serving_gpu_test`; it is absent from automatic CTest discovery.
Raw logs/XML remain in session scratch `qwen-c2-serving-models03`.
Recipe SHA-256: `6f95f9139b3d4a0315da244930a757f9b4da864a735360ff966807acd025e375`.
Result SHA-256: `22c95d7fdddee044422bb736f43f6d0378ce4d1e6ebad549ef5184eaaa9e4ee2`.

This closes the bounded internal Server default-option/reopen control, not
public Qwen admission, numerical-policy adoption, broader same-quantized-reference
quality, OS-process/GPU restart, or the family speed/swap gates.


## Short C2 MTP whole-implementation timing screen

2026-10-13, Spark A, candidate based on `a1f40645`. The existing Server fixture
compares the current legacy implementation with the shared mapping defaults,
using the same target/MTP artifacts above. This is one fresh-process
legacy/shared/shared/legacy bookend, not the frozen M3 baseline or the matched
1% family gate. Neither arm changes its qualified dispatch, and no cross-arm
token, acceptance-count or head-bit equality is required.

Both arms use context 512, chunk 16, two slots, draft depth two and read alignment
256. Four authenticated natural documents supply 255 native IDs each, without
chat templates or added BOS/EOS. Each process warms documents 0/1, clears both
branches and their kept/checkpoint reuse, then measures documents 2/3. Programs
and weight residency remain warm. Every pass completes 510 fresh prompt tokens,
zero reused tokens, 32 prompt chunks, two turn checkpoints and 64 generated
tokens per branch. Generation keeps tokens only, disables early stopping, and
records one completed cohort wall rather than summing branch times. Measured
passes happen to produce 127 draft proposals, 60 accepted proposals and 66 summed
branch steps in every arm; these are observed work counts, not equality gates.

| Phase, mean of two fresh processes | Current legacy | Shared defaults | Shared change |
| --- | ---: | ---: | ---: |
| Server setup | 1.077901 s | 2.023604 s | +87.736% |
| Weight materialization | 5.772398 s | 5.783211 s | +0.187% |
| Measured prompt completion | 3.234365 s | 2.958549 s | −8.528% |
| Measured cohort generation | 2.543479 s | 2.488331 s | −2.168% |

The individual measured generation walls are 2.542571, 2.489498, 2.487163 and
2.544387 seconds in execution order. First-to-last generation drift is +0.0714%
for legacy and −0.0938% for shared. Startup is slower for shared despite the
shorter measured prompt and generation phases. Setup, materialization, branch
reset and keeper drain are outside those phase clocks; checkpoint verification
is outside them too. These results locate no warmed-generation regression at
this short geometry and do not establish a general latency or quality win.

Graph and future counts are cumulative snapshots. Legacy captures rise from
9 after warmup to 14 after measurement, and replays from 42 to 102. Shared
captures rise from 17 to 21 and replays from 124 to 283; built/captured prompt
futures remain 2/2, with prepared count zero. Thus the measured passes still
include new graph captures, rather than measuring replay alone. Recorded plan
floors are 28,543,848 bytes for legacy and 14,591,840 bytes for shared; these
are not total memory measurements.

All four raw cases pass with zero skips and native exit zero. Source, binary
and recipe guards pass; every child process group retires, no GPU process
remains, and memory-limit/OOM events stay zero. The process/page-cache peak is
1,411,534,848 bytes, not device residency. The existing source-derived startup
payment and 104 GiB node cap remain unchanged, under the 112 GiB zero-swap
process/cache ceiling and 6 GiB physical margin. Focused native compilation and
six affected host controls pass, with the other 122 unchanged host controls
authenticated by their prior source/binary receipts. Final main Spark A native
build and all128 affected host controls pass, with formatting, boundaries,2256
headers and REUSE6.2. Main receipt:
`8caebf2bc800e4570fa88220e69dc43d41af0c292f6452c8f7cdcc1db9655dc3`.
The final binary differs after the unrelated spatial-convolution integration;
all Qwen execution and timing-fixture source remains identical to the measured
arm. The timing observations retain their original executed binary/source
provenance; no repeated timing run is needed for that unrelated integration.

Manual replay uses `LLMP_QWEN_C2_SPEED=1`, `LLMP_QWEN_C2_SPEED_IDS` pointing to
the fixed 4,080-byte little-endian ID file, and the existing model/fresh scratch
paths. Select `Qwen38EngineServingGpu.NativeC2SpeedLegacy` or
`Qwen38EngineServingGpu.NativeC2SpeedShared` in `qwen38_engine_serving_gpu_test`,
one fresh process per arm under the existing exclusive Spark supervisor. These
cases are absent from automatic CTest discovery. Input SHA-256:
`3cd946c3876501eccbe079b9d2700e542e66242be241a8549546df8d9b23e468`.
Recipe SHA-256: `ae3dd47b487a951b925de436a87f4294f6180adf380486abba438b0adb84a3a5`.
Result SHA-256: `957e270f3d31a64534b070543ea34852e1a590662267a176385d54378f12c00c`.
The complete four-process aggregate, raw logs/XML and source/binary receipts
remain in session scratch `qwen-c2-speed-models02`.

Public/HTTP shared selection, ordinary multi-owner admission, wider cohorts and
contexts, same-quantized-reference quality, state/swap qualification and the
fixed-baseline family gate remain open. This screen selects no new numerical
policy and makes no original-model agreement claim.
