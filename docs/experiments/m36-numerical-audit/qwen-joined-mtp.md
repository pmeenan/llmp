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

The supervised HTTP bridge is now covered below; automatic public selection,
ordinary multi-owner admission, wider cohorts and contexts,
same-quantized-reference quality, state/swap qualification and the fixed-baseline
family gate remain open. This screen selects no new numerical
policy and makes no original-model agreement claim.

## Supervised shared HTTP C2 control

2026-10-13, Spark A. The existing HTTP service now has an internal, explicitly
funded selector for the shared Qwen target/MTP route. It reuses `RunService`,
`Server`, the ordinary request scheduler and `speed_gate.Service/stream_chat`;
there is no additional serving adapter or numerical-policy selector. Public
model selection and the qualified mapping defaults remain unchanged.

A fresh process serves the same installed target/MTP identities listed above,
with context 512, prefill chunk 16, two slots, draft depth two and read alignment
256. Fresh private state and calibration directories exclude reused histories
or inherited calibration. Two authenticated natural texts go through the real
chat template, each producing 331 prompt tokens. Both streamed requests finish
at their requested 128-token limit with zero cached tokens: 662 fresh prompt
and 256 completion tokens total, with each history staying within 459 positions.
Both streams produce output before either completes. The runtime reports 29
joined MTP waves / 58 joined units, 96 speculative units, 191 draft proposals and
158 accepted proposals. These are actual-work observations, not cross-arm
identity or quality requirements.

The first 64-output-token control completed both individual requests but failed
the joined-overlap requirement. Speculative prompts cannot batch, and the
scheduler may finish one prompt before alternating its generation with the
second prompt's prefill. At chunk 16, the second 331-token prompt needs at most
22 chunk units across its stable boundary plus reuse and checkpoint units;
64 generated tokens can finish during those units. The 128-token control retains
the same overlap and joined-work requirements and uses the same paid context.
The failed receipt remains preserved; no production scheduling change was made.

The actual prepared execution identity is
`2c80ab538694d26a34204e01bb16a795ba5e708922db7417c4a858128b170f36`.
Final cumulative runner counters are 41 plans, 27 graphs, 29 eager runs,
18 captures, 242 replays and zero refusals; four prompt futures are built and
captured, with prepared count zero. Plan-host and graph accounting report
162,537,872 and 309,690,368 bytes respectively. These are component counters,
not a total memory or performance comparison.

The authenticated source/settings inventory derives a 5,909,774,336-byte startup
payment before Server construction work, within the unchanged 104 GiB node cap.
Existing plan/state/workspace and physical admission remain active. HTTP request
memory is separately bounded at 64 MiB, with 8 KiB request bodies and 64 KiB stream
buffers. The supervisor enforces 112 GiB process/page-cache memory with zero swap
and a 6 GiB physical margin. Observed process/page-cache peak is 1,490,305,024 bytes,
not device residency. Both source/binary guards pass; shutdown receives TERM and
exits zero, the complete process group retires, no GPU process remains and all
memory-limit/OOM counters stay zero. Failed native Server teardown now logs the
error and exits before destructors can free unretired owners; a subprocess
control exercises that production failure helper with a live owner sentinel.

Main native compilation and all 158 affected host controls pass (43 runtime,
109 API, six Qwen), with zero failures/skips. Pinned formatting, boundaries,
2,256 headers and REUSE 6.2 pass. The actual runtime binary exactly matches the
main build: SHA-256
`3644baa29d56289b5066ad575602822ad1279a3233f85b59331e90c4956d0170`,
271,061,904 bytes. Main receipt:
`0f0d925ce39c072e01ff7e81294790676a4826136f0ad776f083526f763e1667`.
Actual recipe:
`7ee206bed05c8489121329ce4a81ac7d469ecdee4ede39930a39ca6ce29d8a96`.
Actual result:
`d9c98eaac5cfadd3a80ea30aacd6ac5885da453ef3e1d5a377f2c16fc4ca30e4`.

Replay uses the existing `llmp-runtime` service with the all-or-none internal
tuple `LLMP_DIAGNOSTIC_COMPONENT_QWEN=1`,
`LLMP_DIAGNOSTIC_STARTUP_HOST_BYTES=5909774336` and
`LLMP_DIAGNOSTIC_NODE_CAP_BYTES=111669149696`, authenticated artifacts and fresh
private directories. Byte values require positive unsigned decimal, 2 MiB
alignment and startup no larger than the cap. Partial/invalid controls refuse
before startup; unsupported selected aliases cannot silently fall back. The
caller must rederive payment if source/configuration changes. The existing speed harness
client sends two simultaneous streaming chat requests with temperature zero and
max_tokens 128; verify actual usage, overlap, prepared identity, joined counters
and positive shutdown under an exclusive Spark supervisor. Captured texts,
config, complete logs, request-output digests and guard receipts remain in session
scratch `qwen-http-bridge-models02`; the earlier failed 64-token control remains
in `qwen-http-bridge-models01`.

This closes the bounded supervised HTTP route proof only. Automatic production
selection, ordinary C1/C4 and broader MTP cohorts, same-quantized-reference
quality, swap/state qualification and the matched fixed-baseline 1% family gate
remain open. The check makes no new quality or comparative speed claim.

## Supervised native target-only HTTP C2 control

2026-10-13, Spark A. The same internal diagnostic selector can derive an
ordinary Prompt/Decode domain from one actual native target and its configured
context, chunk and owner envelope. It requires no drafter, speculation disabled
and no competing caller-provided execution domain. The composed target/MTP and
closed Score routes keep their existing semantics. The generic single-target
path also accepts one configured owner; this slice makes no actual C1 claim.

The bounded control uses the existing installed native target, context 512,
prefill chunk 16, two slots and read alignment 256. It sends the same two
authenticated natural HTTP prompts through the real chat template, requesting
128 output tokens each. Both streams produce output before either ends. Each completes 331 fresh
prompt tokens and 128 output tokens; the runtime reports 22 joined prefill
steps / 662 rows and 127 ordinary joined waves / 254 units, with zero
speculative units. The subsequent exact first-prompt repeat completes 16
output tokens and reuses 326 of its 331 prompt tokens. Total completion work
is 272 tokens. This observes same-selected prefix reuse;
it is not a durable-state byte oracle or a cross-implementation equality gate.

The source/configuration inventory retains the conservative 5,909,774,336-byte
startup payment, including compiler capacity and authenticated metadata bounds,
within the unchanged 104 GiB node cap. Removing the drafter does not repurpose
its conservative metadata allowance. Actual setup and cached plans retain
measured resource admission and PlanAccount charges. The existing HTTP and
process supervisor limits and positive retirement requirements are unchanged.

The actual prepared execution identity is
`707d5b8030033a801b0df6ab7d6cc05ced115f18fcd23cddef5c91e9c8c966be`.
Final cumulative counters report seven plans, five graphs, four eager runs,
three captures, 158 replays and zero refusals. Plan-host and graph accounting
report 95,356,208 and 179,421,184 bytes; two prompt futures are built and
captured, with prepared count zero. These observations are not a comparative
speed or memory measurement. Process/page-cache peak is 1,319,759,872 bytes,
not device residency. TERM exits zero, the complete process group retires,
all source/runtime guards pass, no GPU process remains and all memory-limit/OOM
counters are zero.

Main native compilation and all 159 affected host controls pass (43 runtime,
109 API, seven Qwen), without failures/skips. Pinned formatting, boundaries,
2,256 headers and REUSE 6.2 pass. Runtime SHA-256:
`321822e33762fb09f9ab8e32dd47a9252fa7f5224fc04354de905bf1e1882034`,
271,071,104 bytes. Main receipt:
`ef53398fa4f6cfe29bc4d48a187393e329f5d2dfe0f36c46f3f4798e4b90ced4`.
Actual recipe:
`f8a0d05d0691689093f0070931e1f3259e7f2a7d9302028f8d6ba511f4417374`.
Actual result:
`27a55d0cf1fd4e10b743b37dba492f3be541e282253bb717ce7c48add0e68ea1`.

Replay uses the preceding internal selector/payment tuple with no drafter and
`speculation = false`, retaining the bounded context/chunk/slot settings. After
the simultaneous 128-token requests finish, repeat the exact first request body
with only `max_tokens` changed to 16. Check complete token counts, actual joined
ordinary work, positive cached prefix and graceful whole-process retirement.
The authenticated configuration and raw evidence belong in session scratch
`qwen-plain-serving-models02`. The preceding `models01` receipt remains
preserved: it stopped before requests because the controller expected an
override label instead of the source-defined `speculation=false (derived)`
for an alias with no drafter; setup and retirement were successful. The
correction changed only that observer, not source, configuration or payment.

This is bounded native target-only C2 coverage; the C4 follow-up is below. C1,
automatic public selection, broader same-quantized-reference quality and
the fixed-baseline 1% family gate remain open. It makes no quality or comparative speed claim.

## Native target-only HTTP C4 control

2026-10-13, Spark A. A recipe-only extension of the same qualified native
runtime uses four configured slots, with context 512, chunk 16, read alignment
256 and no drafter. Four authenticated natural texts render to 330, 331, 331
and 331 prompt tokens. All four fresh requests complete their requested 128
output tokens, and all four streams produce output before any finishes.
The runtime records 127 joined ordinary waves / 504 units with zero speculation.
Because each wave has at most four owners and the unit sum exceeds three times
the wave count, this proves actual four-owner execution. The first prompt's
repeat completes another 16 tokens, reusing 325 of its 330 prompt tokens. Total
completion work is 528 tokens; the largest history remains 459 positions.

The source/config-derived startup payment grows to 5,913,968,640 bytes: four
owners require at most 960 saved roots in the existing graph-capacity bound.
The 104 GiB node cap, 112 GiB zero-swap process/page-cache ceiling and 6 GiB
physical margin remain unchanged. Setup measures and admits the larger
activation/workspace/input/state closure; cached plans and futures retain
normal accounting. Final cumulative observations are ten plans, five graphs,
eight eager runs, four captures, 159 replays and zero refusals. Plan-host and
graph accounting are 144,152,544 and 202,194,944 bytes. Process/page-cache peak
is 1,372,598,272 bytes, not device residency. TERM exits zero, the whole process
group retires, source/runtime guards pass, GPU processes and memory-limit/OOM
events are zero. This is neither a speed comparison nor a memory-saving claim.

The actual prepared identity is
`e23d604208af841d4733db379ecb16cd8772c8db8b7bf513890df3dcd18574f6`.
Production, test and tool bytes and runtime SHA-256 remain exactly those of the
preceding successful main159 build; only four committed Markdown files differ
in the guarded source map at `ba20c1cb`. No duplicate compilation was needed.
Recipe SHA-256:
`ffeba388d0b45199b2f0e00c5a0d5d59d69e071818cf190024ec5245777a0a55`.
Actual result:
`3537803621f39e9c8a357d44d8227a1d7c890267217da6a292f068f316e86ea3`.
Raw evidence and replay inputs are retained in `qwen-plain-c4-models02`; the
preceding failed receipt preserves four completed responses before a controller
assumed 331 prompt tokens for every document instead of their actual rendered
counts, and stopped before the prefix repeat. Only that observer was corrected.

Replay keeps the existing supervised HTTP path, sets four slots and the derived
startup payment above, and starts all four 128-token requests together before
the 16-token prefix repeat. Require exact per-document rendered counts, all-four
output overlap, the four-owner counter witness, positive cached prefix and
positive retirement. Native C1, larger contexts, broader quality, public default
selection and the fixed-baseline 1% family gate remain open. GGUF C2 coverage follows below.

## GGUF target-only HTTP C2 control

2026-10-13, Spark A. The same qualified shared runtime and internal selector
complete a target-only C2 HTTP control for prepared GGUF artifact
`5356b5b05fd93d06419cd842c4946e0df9d57ec816916109af5c724cadf25b78`
(Qwen3.8 UD-IQ3_XXS). Context 512, chunk 16, read alignment 256 and two slots
remain unchanged; there is no drafter. Both fresh requests complete 128 output
tokens from 331-token prompts, with both streams producing output before either
finishes. Cumulative counters record 127 joined ordinary waves /
254 units and zero speculative work. Repeating the exact first
prompt completes 16 more tokens and reuses 326 of 331 prompt tokens. Total
completion work is 272 tokens. This is live retained-prefix evidence, not a
durable-state byte comparison or a cross-implementation output gate.

Before inference, a bounded native metadata check uses the existing GGUF
reader, tokenizer and chat renderer to record both complete rendered token
sequences. The runtime prefers the kept GGUF template
`12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`
over configured external fallback assets. Its independently measured counts
are 331/331; the native artifact's counts were not assumed. Metadata receipt:
`2032002e63c013fef22c09b98be3d5b9cf8210895fd1914e763b073d2c9240c3`.

The authenticated source has 1,080 resources and 24,627 groups, whose rounded
weight reservations total 85,983,232,000 bytes. The source/configuration-derived
startup payment remains 5,909,774,336 bytes within the 104 GiB node cap; actual
setup, workspaces, state, cached plans and futures retain ordinary accounting.
The 112 GiB zero-swap process/page-cache ceiling and 6 GiB physical margin are
unchanged. Observed process/page-cache peak is 1,221,808,128
bytes, not device residency. TERM exits zero, the complete process group retires,
source/runtime guards pass, and GPU processes and memory-limit/OOM events are
zero. No comparative speed, quality or memory-saving claim follows.

Actual prepared execution identity:
`c586531b75d58716979810a7647477338eaaf60c70c54a176463d4290df64508`.
The runtime and relevant source bytes are exactly those of the preceding
successful main159 build (runtime `321822e33762fb09f9ab8e32dd47a9252fa7f5224fc04354de905bf1e1882034`).
The guarded A source map retains the already authenticated four documentation
changes at `ba20c1cb`; later unrelated image changes are not part of this run.
No duplicate compilation was required. Recipe:
`6cdb4594719b7ab3d58fe5497c49b0756eb5ec031f8f7c5224bccf2b9f7b6ff1`.
Actual result:
`ff50307a1330c37ca77c5bc4207a279556efbec4c4239f672e5ca2723a57aab2`.

Replay uses the same supervised C2 HTTP protocol and startup payment, selects
this artifact under `qgguf-artifacts`, verifies the kept metadata and native
rendered-input receipt, and requires both 128-token responses to overlap before
the 16-token prefix repeat. Raw evidence and inputs remain in session scratch
`qwen-gguf-c2-models01`. GGUF C4 coverage follows below. Larger contexts,
broader same-quantized-reference quality, automatic public selection and the
fixed-baseline family speed/swap gates remain open. GGUF+MTP and native-v1 admission are unchanged.

## GGUF target-only HTTP C4 control

2026-10-13, Spark A. The existing GGUF route also passes four-owner ordinary
HTTP serving at context 512, chunk 16 and read alignment 256. The same native
metadata driver independently authenticates the remaining two texts before
inference; all four rendered prompt counts are 330, 331, 331 and 331.
Four fresh requests each complete 128 tokens, with all four streams producing
output before any request finishes. The runtime records 127 joined ordinary
waves / 504 units, proving an actual four-owner wave with four configured slots.
Speculative work is zero. Repeating the first prompt reuses 325 of 330 tokens
and completes 16 more, for 528 total output tokens.

This reuses the successful native C4 controller and the GGUF C2 source guards;
no production change or rebuild is involved. Four owners increase the existing
saved-root bound to 960 and the derived startup payment to 5,913,968,640 bytes.
Actual state, workspace, activation, input and plan/capture/future allocations
retain normal measured admission within the same 104 GiB node cap, 112 GiB
zero-swap process/page-cache ceiling and 6 GiB physical margin. Cumulative
observations are 11 plans, five graphs, eight eager runs, four captures,
159 replays and zero refusals. Process/page-cache peak is 1,292,857,344 bytes,
not device residency. TERM exits zero, the complete process group retires,
source/runtime guards pass, and GPU processes and memory-limit/OOM events are
zero. These are functional observations, not comparative speed, quality or
memory-saving claims.

Prepared execution identity:
`3cdcdc88220fefaf19496cc966065c229267cd99eb64020f8bfc7aa5226d6eb5`.
The exact main159 runtime and guarded source provenance are unchanged from
GGUF C2 above. Remaining-input metadata receipt:
`c4de428c8d3183ff1bad92ff6478623076b86617af7de03a6e477319dc75490f`.
Actual recipe:
`78153369ca24dad56b3e251e0aef451941218490c50bf755cc6ff1623b2c7a5d`.
Actual result:
`7f115b8ef6cedee0d4e7e395b83ddb16df6a22a8846a6d5428328d8d3e08371e`.

Replay selects four slots and the source-derived startup payment above, reuses
the authenticated GGUF metadata/input receipts and existing four-request
controller, and requires all-four output overlap, the four-owner work-counter
witness, exact completion counts, positive prefix reuse and complete retirement.
Raw evidence remains in `qwen-gguf-c4-models01`. This closes the bounded ordinary
C2/C4 HTTP coverage for both native and GGUF sources; it does not qualify larger
histories, automatic public admission, broader same-quantized-reference quality
or the fixed-baseline family speed/swap gate. Automatic internal startup accounting and the existing family-geometry screen
follow below; public selection and the family gates remain separate.


## Automatic startup accounting and family geometry

2026-10-13, Spark A. The internal shared selector accepts
`LLMP_DIAGNOSTIC_STARTUP_HOST_BYTES=auto` alongside its existing selector/node-cap
fields. Numeric grants and public model defaults are unchanged. Initial parsing
is paid from enforced reader/parser ceilings, including duplicate-key scratch;
it does not trust a disconnected file-size observation. Actual validated
source/settings and component-declared saved-root bounds then pay compiler/domain
metadata, while tokenizer capacities and chat geometry pay retained assets.
Those grants exist before the temporary parsing extent is released, after local
parser/settings owners die and before scheduling starts. Existing catalog,
plan/capture, future, workspace and growing-state accounting remain active.

The existing family harness uses native target `c4fb47a9…` and MTP `8600a998…`,
context 33,792, chunk 4,096 and four slots. Its requested draft-vocabulary cap
65,536 resolves to the artifact's 47,172 selected rows. The completed quality
process retires its parsing loan and reports 6,708,789,248 retained host bytes;
its complete allocation guard passes. Two greedy replies each complete 128
tokens. Literal prompt scoring, through existing one-row Decode/full-head
execution, records 996 target losses: mean NLL **2.756445494533136**, PPL
**15.743782013599764**. No external same-quantized-checkpoint reference on these
996 coordinates is established by this run; historical legacy-runtime quality
is migration context, not a replacement reference oracle.

The original wrapper receipt is FAIL because its settings observer expected the
requested vocabulary cap rather than the source-backed effective row count.
Native quality exited zero, and source guards, complete process/GPU retirement
and zero memory events passed. An independently reviewed offline audit corrects
only that observer; completed model work is not repeated.

The remaining existing `run --prime --session auto-locator` completes one warm
and two measured passes, each with C1/C2/C4. All 21 requests finish 128 tokens
with zero cached prompt tokens: 2,688 outputs and actual prompt lengths
6,835–8,478. This configures the full family envelope; it does not fill the
33,792-token context. The two measured-pass averages are:

| Cohort | Prefill seconds | Decode tokens/s | Wall tokens/s |
| --- | ---: | ---: | ---: |
| C1 | 3.203883 | 48.454922 | 21.907664 |
| C2 | 7.121829 | 61.824115 | 22.609637 |
| C4 | 16.755124 | 65.637934 | 21.334918 |

The C4 decode observation is about 14.7% below the historical frozen baseline's
76.93 tokens/s; prefill is about 6.0% longer and wall rate about 8.6% lower.
This single prime locates a meaningful gap for investigation, not a matched
statistical verdict. Cumulative counters show 270 joined waves / 744 units,
36 built and cached futures, eight state preparations, and zero runner refusals
or coverage violations. The existing memory sampler records an 89.8664 GiB
drop in system `MemAvailable`; the separate cgroup process/page-cache peak is
6,714,793,984 bytes. Neither is a device-only measurement. Both processes pass
hardware-fit admission, retain the same 6,708,789,248-byte startup grant, exit
TERM/zero and positively retire their groups/GPU work, with zero memory-limit
or OOM events. The recipe retains a 104 GiB node cap, 112 GiB zero-swap outer
ceiling and a checked live 6 GiB physical margin.

Prime result:
`6861fd4980be4cd44de000cdc7efa24648117ba048dd975c30d08165819396d4`.
Raw evidence remains in `shared-startup-admission-models03`; it runs only the
previously unstarted prime. Replay uses the existing `speed_gate.py quality`
and `run --prime` commands with the frozen family config and internal `auto`
startup tuple, under the supervised source/input/runtime and retirement guards.
The 996-loss observation and single prime do not establish an external-reference
quality pass or a family speed gate.

The source02 payment/lifetime review is clean. The integrated main build passes
full native compilation and 456 host cases (355 startup-affected plus 101
separate image controls), with no failures or skips. Main receipt:
`cec4f4af92ba9554911a1ba271f674722142c4cdff30a7e984a9b19d11191f29`.
Actual runtime:
`c4bc2c9507a59c0324d6ac54aaaf61ca684825420e75b33f35b224ed5ea5e6df`
(271,813,392 bytes), authenticated against the combined 2,542-file source map.
Quality raw/failed-wrapper evidence remains in `shared-startup-admission-models02`;
its corrected offline aggregate is
`5f399b7555d763fcf9f92998a4396c6340c662d93de65365faed054d7153a766`.
Final source QA passes: 22 formatted C++ paths, 19 boundary checks, 2,266
SPDX headers and REUSE 6.2.0, on an isolated 2,543-file Spark A export.
This qualifies bounded automatic internal
startup only; public migration, broader same-quantized-reference quality,
state/swap coverage and the matched 1% family gate remain open.
