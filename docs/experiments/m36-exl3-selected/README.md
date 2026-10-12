<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# EXL3 selected operations and first reconstruction execution

The shared GGML lowering can bind approved EXL3 linear operations into its
ordered step program. Single products and paired gate/up products preserve
all physical weight parts, existing F16 rounding and selected implementation
identity. Packed G/O variants use 98 authenticated P3 tuning records; all
69 earlier records are unchanged. The 18 output-specific reconstruction
facts remain authoritative. Unknown tuning keys and unsupported shapes refuse.
Resident norm scales remain BF16; the existing GGML copy kernel widens them
to transient F32 operands. Primitive expansion waits for all fusion inputs,
so a computed scale cannot split its selected RMSNorm/multiply operation.

Construction submits no work. The selected source owns stable pointer tables,
disjoint persistent locks and its first-use initialization on the owned stream.
The diagnostic explicitly selects uncaptured execution through a keyed option;
ordinary component-source capture requirements remain the default. Each run
retains complete ownership through completion, including on an uncertain result.

On Spark B, 26 focused host checks, six complete selected-operation controls
and one actual GPU conversion control passed with zero skips. The descriptor
controls cover all four profile identities and the approved reconstruction
boundaries; they do not qualify actual execution on all four profiles.

A newly sealed native reference completed the fixed 145-token prefill and
16 suffix steps of the approved 4.0-bpw G fixture. Its 145-row output then
matched the shared runner bit for bit across all 151,936 vocabulary entries:
22,030,720 F32 values, or 88,122,880 bytes. Both input/admission controls passed.
The same original 1,040-ID file is authenticated by both arms; the shared
screen consumes its first 145 IDs. The reference uses an externally admitted,
source-bounded workload under an exclusive job; it has no memory-cap option.
The candidate prepays its 4-GiB host inventory within its 8-GiB domain and
checks the separate physical margin before construction.

The actual screen took 964 ms; this is functional evidence, not a speed gate.
That reconstruction screen alone does not qualify packed products, other
profiles or lifecycle behavior; the later bounded packed screen is below.
Historical Tier C, broader lifecycle behavior and deletion remain open;
the bounded same-implementation checkpoint/spill control is recorded below.

On 2026-10-12, G5b2a extended the shared 4.0-bpw G control to the
complete 145-token prefill and 16 one-row suffix steps. A fresh native
reference completed all 17 phases using the same original 1,040 IDs and
approved typed plan. The shared runner matched all 161 full-vocabulary rows
bit for bit: 24,461,696 F32 values, or 97,846,784 bytes. The original
145-row control also passed. Both cases verify completed positions and
step/row counts; setup measures only the two distinct shape keys.

The full control retains the same 4-GiB host allowance and 8-GiB domain
with the separate 6-GiB physical margin. Its source-derived host bound is
4,109,194,928 bytes, including the second native program and the larger
oracle. Native build, two input/admission controls, six selected-operation
controls, 30 component/state/lifecycle checks and the quality gates passed.
Both actual model cases passed with zero failures or skips under a 16-GiB
zero-swap process limit. These historical checks cover reconstruction. The bounded packed scoring
screen below adds execution evidence; historical Tier C and model lifecycle
controls remain open.
The unrelated full regression suites were deferred under the M3.6
incremental check policy.

Final regression on the frozen checkpoint17 passed: Spark A completed 2,554
native and 2,003 CPU CTest cases plus format, boundary, header and REUSE gates.
Spark B completed all 134 model/timing CTest cases, containing 179 raw GTest
arms, including the actual reconstruction control and all four FP16 bridge
arms. All suites and every raw GTest arm had zero failures or skips. These
regressions preserve existing coverage; they do not close the remaining EXL3
profile or lifecycle migration gates. Main built and passed all 91 focused
integration checks and the quality gates, with zero failures or skips.
Source, recipe, input and binary guards passed before and after each job.
[results.json](results.json) records the aggregate evidence; raw data remain
in external scratch until the milestone closes.

## Bounded mixed-rate 4.5-bpw G scoring screen (G5b2b)

On 2026-10-12, the second approved EXL3 fixture completed the common shared
scorer on the same authenticated 43 teacher IDs as the
[original BF16 reference](../m36-original-qwen2/README.md). The fixed artifact
`00d77caf056c45b6fa05ebdbeff8b38da79d0556a2fd4f1b661e7e21ef9ed83a`
contains 169 logical products: 42 at 4 bits, 118 at 5 bits, eight at 6 bits and
one at 8 bits. All 676 code/sign/marker parts remain in their source encoding.
Both selected programs use 121 packed linear GEMMs, 24 paired gate/up MGEMMs
and 72 bias operations from the authenticated 4.5 G profile. No kernel,
tuning fact, arithmetic policy or production dispatch default changed.

The existing 7/1 schedule consumes six seven-row chunks and one final row.
The common scorer evaluates all 42 next-token targets using full-vocabulary
double logsumexp and retains bounded per-position NLLs. All logits are finite;
complete tokenizer JSON bytes match the original source. This matches the
teacher histories and target coordinates, while the FP16 scorer uses 8/3 and
the original BF16 evaluator uses its declared eager/no-cache recipe.

| Recipe | Targets | Mean NLL | Perplexity |
| --- | ---: | ---: | ---: |
| Original BF16, eager/no cache | 42 | 6.496176475891133 | 662.6033037486249 |
| Shared EXL3 4.5-bpw G | 42 | 6.515128758851711 | 675.2809043195413 |

This short observation is neither a corpus-wide quality gate nor a causal
comparison of quantization alone. No equality to quantized-library logits is
required. The existing 4-GiB host inventory, 8-GiB native domain, separate
6-GiB physical margin and complete-lifetime retirement remain unchanged.
The external process/cache cap is 10 GiB with zero swap; observed cgroup peak
was 276,840,448 bytes, which is not total GPU occupancy.

Spark A built the two affected targets and passed two input controls plus six
selected-operation controls, then passed the one actual scoring case. All raw
cases had zero failures or skips; source, binary, recipe and input guards
passed, with zero compute processes left afterward. Main then built natively
and passed the same eight affected controls plus both the 4.0- and 4.5-bpw G
scoring cases, again with zero failures or skips and all guards passing.
O profiles, broader packed shapes,
Tier C, broader EXL3 lifecycle behavior and legacy deletion remain open.

## Same-implementation checkpoint/spill continuation (G5c4)

Both approved G fixtures now pass a bounded state control through the same
shared runner. Each case first runs the fixed 145-row prefill and 16 scalar
suffix steps uninterrupted, retaining its own full-vocabulary output reference.
After clearing state, a second evaluation compares all 161 rows. A checkpoint
at 145 is restored after eight real decode writes; replaying those eight rows
also matches exactly. Spill/restore at 157 preserves the complete used state,
and the final state at 161 matches the uninterrupted run byte for byte.

Each fixture compares 169 full-vocabulary rows in total, executes 42 steps,
and restores/compares 6,291,456 bytes at each state check. The 40G case retains
292 immutable weight extents; 45G retains 302. Both record artifact-bound model
identity and actual selected plan identities/names: the 145-row plan runs twice
and the scalar plan runs 40 times. There is no external quantized-library output
oracle and no change to arithmetic or production dispatch.

One 97,846,784-byte output buffer holds the uninterrupted reference. Only after
the second prefill has compared all 145 rows can its retired prefix bytes store
the final-state reference; the still-live suffix rows remain untouched. Two
16-MiB pinned state-copy buffers are included explicitly in the pre-Start
8-GiB domain inequality and remain owned through completion. The original
4-GiB host allowance covers the bounded metadata and plan records, and the
separate 6-GiB physical margin remains. Unknown retirement retains the complete
owner graph. The external 10-GiB zero-swap process/cache cap is unchanged.

Spark A passed the native target build, three input controls and six selected
controls, followed by both actual lifecycle cases with zero failures or skips.
All source, input, recipe and binary guards passed, and no compute processes
remained after either case. Observed process/cache peaks were 943,190,016 bytes
(40G) and 943,714,304 bytes (45G); these are not total device occupancy.
Main built natively and passed the same nine affected controls and both
lifecycle cases, with all guards passing and zero failures or skips. This
qualifies only the stated same-implementation
checkpoint/spill continuation; kept-state/restart, eviction/relocation,
broader shapes/profiles, Tier C, performance and deletion remain open.

## Bounded 4.0-bpw O GEMV scoring screen (G5b2c)

The existing 4.0-bpw O profile now completes the common scorer on the same
43 authenticated teacher IDs and 7/1 chunks as the G profile. Before creating
the selected source, the fixture queries all eight native GEMV occupancy facts;
the source validates those facts and rechecks them when it creates the launch.
The observed counts are `[48,96,48,96,48,96,48,96]`, ordered by output
F16/F32, one/multiple rows, and narrow/wide configuration. Each actual program
records 120 four-bit GEMVs, one eight-bit head GEMM, 24 paired gate/up MGEMMs
and 72 bias operations. The original artifact and all physical weight parts
are unchanged; no tuning, kernel or production default changed.

All 42 next-token scores are finite. Mean NLL is 6.45690722611154 and
perplexity 637.0876384616802, compared with original BF16's 6.496176475891133
and 662.6033037486249 on the same teacher histories and target coordinates.
This short sample is a primary-reference observation, not a broader quality
qualification. The original uses its declared eager/no-cache recipe; no
quantized-library bit equality or speed gate is imposed here.

Spark A built the affected targets and passed three input controls plus six
selected-operation controls, then passed the one actual scoring case. Every
raw case had zero failures or skips, all source/input/recipe/binary guards
passed, and no compute processes remained afterward. The complete Lifetime,
4-GiB host allowance, checked 8-GiB native domain and 6-GiB physical margin are
unchanged. Observed 10-GiB zero-swap process/cache peak was 276,574,208 bytes;
this does not measure total GPU occupancy. Main then built natively and
passed the same nine controls and the one actual scoring case, with identical
scores and occupancy facts, zero failures/skips, all guards passing and no
compute processes left afterward.
Broader shapes, O lifecycle coverage, Tier C, performance and legacy
deletion remain open.

## Bounded mixed-rate 4.5-bpw O scoring screen (G5b2d)

The approved 4.5-bpw artifact now executes the same natural43 common-scorer
control through the existing O profile. Its 169 logical products retain all
676 code/sign/marker parts: 42 products at 4 bits, 118 at 5 bits, eight at
6 bits and the head at 8 bits. Of the four-bit products, 28 gate/up members
use 14 paired MGEMMs; the remaining 14 down products use GEMV. Each actual
7/1-row plan records 14 GEMVs, 107 GEMMs, 24 paired MGEMMs and 72 bias
operations. All eight queried occupancy facts are `[48,96,48,96,48,96,48,96]`
in the order recorded above and are rechecked at launch. No tuning, kernel,
weight encoding or production default changes.

All 42 full-vocabulary next-token scores are finite. Mean NLL is
6.5150540137704605 and perplexity 675.2304322797728 against original BF16's
6.496176475891133 and 662.6033037486249 on identical teacher histories and
target coordinates. The candidate uses 7/1 chunks; the original uses its
recorded eager/no-cache recipe. This short primary-reference observation is
not broader quality qualification, a speed result or a quantized-library
bit-equality gate.

Spark A built the affected targets and passed three input plus six selected
controls and the one actual scoring case, with zero failures/skips and all
source/input/recipe/binary guards passing. No compute processes remained.
The same complete Lifetime, 4-GiB host allowance, checked 8-GiB native domain
and 6-GiB physical margin apply. Observed process/cache peak under the
10-GiB zero-swap cap was 277,135,360 bytes, not total GPU occupancy. Main
then built natively and passed the same nine controls and the scoring case,
with identical scores and occupancy facts, zero failures/skips, all guards
passing and no compute processes left. Broader shapes, O lifecycle, Tier C,
performance and legacy deletion remain open.

## O-profile same-implementation checkpoint/spill continuation (G5c6)

Both approved O profiles pass the same G5c4 state control without new buffers
or arithmetic. Each uninterrupted 145-row prefill plus 16 scalar steps supplies
its own full-vocabulary reference. The second evaluation compares all 161
rows, rewinds from 153 to the checkpoint at 145 and replays eight rows, then
spills/restores at 157 and finishes at 161. Each case compares 169 output rows
and 6,291,456 initialized state bytes at each checkpoint, spill and final-state
check. The 40O/45O cases preserve 292/302 immutable weight extents and record
the artifact-bound model identity; this is no-eviction evidence.

Both 145-row plans run twice and select 169 reconstruction operations plus
72 bias operations. Scalar plans run 40 times: 40O selects 120 GEMVs and one
GEMM, while 45O selects 14 GEMVs and 107 GEMMs; both also select 24 paired
MGEMMs and 72 bias operations. All eight actual occupancy facts are
`[48,96,48,96,48,96,48,96]`, in the order above, and are rechecked at launch.
The exact comparisons use the same candidate's uninterrupted output and state,
not an external quantized-library oracle.

The sole 97,846,784-byte reference buffer, its retired-prefix reuse, two
16-MiB pinned state-copy buffers, 4-GiB host allowance, checked 8-GiB native
domain, 6-GiB physical margin and complete-owner retirement are unchanged.
Spark A built and passed three input and six selected controls, followed by
both actual lifecycle cases, with zero failures/skips, all guards passing and
no compute processes left. Observed process/cache peaks under the 10-GiB
zero-swap cap were 944,017,408 bytes for 40O and 944,025,600 for 45O; these do
not measure total device occupancy. Main also built and passed all nine
focused controls and both actual O cases with zero failures/skips, all guards
passing and no compute processes left; its binary matches the author binary.
Kept-state/restart, eviction/relocation, broader batches/shapes, Tier C,
performance and legacy deletion remain open.

## Kept files adopted by fresh O-profile owners (G5c7)

Both approved O profiles now pass a bounded kept-file control. The first native
owner executes an uninterrupted 145-row prefill plus 16 scalar steps. It clears
and repeats through position 157, comparing every full head, then spills and
atomically publishes an existing version 2 kept record. After successful
teardown and destruction, a freshly constructed node/runner adopts the exact
named file into an empty slot, restores all 6,291,456 initialized state bytes,
and finishes the remaining four scalar steps. All 161 compared output rows and
the complete final 161 state match the same candidate's uninterrupted reference.
There are 34 executed steps: 30 in the predecessor and four in the fresh owner.

Both owners independently synthesize the artifact-bound model, kept layout and
145/1 selected recipes. The record binds those recipes, the 40O/45O profile and
the checked executable identity. File identity, exact length and all initialized
extent hashes are checked before adoption; the actual scalar bound plan and
ordered implementation names are checked again before execution. Wrong profile,
plan, layout and incomplete footprint refuse. The control also confirms that a
cleared slot retaining backing cannot adopt the file. The predecessor preserves
292/302 weight extents; this does not test weight relocation or eviction.

The predecessor runs each 145-row reconstruction plan twice and its scalar plan
28 times; the fresh owner runs only the same scalar plan four times. Both
reconstruction plans select 169 reconstructions and 72 bias operations. Scalar
40O uses 120 GEMVs and one GEMM; 45O uses 14 GEMVs and 107 GEMMs; both use 24 paired
MGEMMs and 72 bias operations. Actual occupancy facts remain
`[48,96,48,96,48,96,48,96]`, in the order recorded above.

The sole 97,846,784-byte oracle transfers only after proven teardown. Two disjoint
consumed-prefix ranges hold the 157 and 161 state snapshots, and the existing
32-MiB pinned buffers are released before the fresh owner allocates its pair.
A checked 512-KiB record/receipt inventory fits within the existing 1-MiB state
metadata allowance. The 4-GiB host allowance, 8-GiB native domain, 6-GiB physical
margin and 10-GiB zero-swap process/cache cap remain. Unknown retirement retains
the complete owner and prevents starting its successor.

Spark A built and passed four input, six selected-operation and three existing
kept-record controls, then both actual cases, with zero failures/skips, all
source/input/recipe/binary guards passing and no compute processes left.
Observed process/cache peaks were 944,218,112 bytes for 40O and 943,742,976 for 45O;
these do not measure total device occupancy. Main then built natively and
passed all 13 focused controls and both actual cases, with zero failures/skips,
all guards passing and no compute processes left. Its binary includes the
independent cold-prefix hydration fix and is separately identified in the
result record; the selected recipes and full state controls pass unchanged.
This proves serial fresh native ownership within one process, not process
launch/crash recovery, serving-history restoration, broader shapes, performance
or legacy deletion. It adds no quantized-library bit-equality adoption gate.

## Four-profile weight reload and kept continuation (G5c8)

All four approved execution profiles, 40G/45G/40O/45O, pass one combined
weight-reload and fresh-owner control. The sole reference remains each
candidate's uninterrupted 145-row prefill and 16 scalar steps on the existing
sealed 1040-ID file's first 161 IDs. After comparing the repeated 145-row
prefill, the fixture fully ends the request before evicting every immutable
weight extent: 292 for each 40 profile and 302 for each 45 profile.

Every extent becomes nonresident and the physical backing count drops by
exactly that count. The next request reloads them with strictly advanced backing
generations and unchanged content generations. All 798 artifact resource
addresses remain unchanged; the authoritative placement check passes. Before
any subsequent model write, all 6,291,456 initialized state bytes at position
145 match the saved state. Both selected programs remain installed, with exact
bound recipe checks before execution and no replacement program constructed.
This is backing eviction/reload at the same virtual addresses, not relocation.

The continuation reaches 157, publishes the same authenticated version 2 kept
record, and destroys the predecessor only after successful retirement. A fresh
owner independently checks the model, layout, executable and G/O profile's
planned recipe, adopts the complete file into an empty lifecycle, and completes
four scalar steps. Each case executes 34 steps and compares all 161 full output
rows and the complete final state against its own uninterrupted reference.
This also extends kept-file adoption coverage to both G profiles. Existing
wrong-profile/plan/layout/footprint and retained-backing refusals remain.

Each predecessor runs its 145-row program twice and scalar program 28 times;
each successor runs the same scalar recipe four times. Prefill selects 169
reconstructions and 72 bias operations. G scalar recipes use 121 GEMMs, 24 paired
MGEMMs and 72 bias operations; 40O uses 120 GEMVs and one GEMM, while 45O uses
14 GEMVs and 107 GEMMs, with the same paired/bias counts. Both O profiles
retain all eight actual occupancy facts `[48,96,48,96,48,96,48,96]` and their
launch-time validation; G does not claim an occupancy-based selection.

The sole 97,846,784-byte reference and serially owned 32-MiB pinned state buffers
are unchanged. Checked reload witnesses require at most 53,248 additional bytes
inside a 64-KiB allowance; the complete state/kept/reload metadata upper bound is
884,736 bytes within the existing 1 MiB. The 4-GiB host allowance, 8-GiB native
domain, 6-GiB physical margin and 10-GiB zero-swap process/cache confinement remain.
Unknown completion retains the complete owner and prevents its successor.

Spark A built the affected targets and passed five input, six selected-operation
and three kept-record controls, then all four actual cases, with zero failures
or skips, all guards passing and no compute processes remaining. Observed
process/cache peaks were 943,767,552 bytes (40G), 943,501,312 (45G), 943,005,696
(40O) and 944,005,120 (45O); these are not total device occupancy. Main passed
a full native build, the same 14 controls and all four actual cases, with zero
failures or skips, all guards passing and no compute processes remaining. Main
process/cache peaks were 943,267,840 (40G), 943,255,552 (45G), 943,747,072 (40O)
and 943,742,976 bytes (45O). Actual address relocation, OS process restart/crash recovery,
serving-history restoration, broader shapes and legacy deletion remain open.
No corpus-quality or performance claim, or external quantized-library
bit-equality adoption gate, is added.


## Grouped original-reference observations

All four approved EXL3 profiles score the same four independently authenticated
256-token passages as the [original BF16 reference](../m36-original-qwen2/README.md#expanded-attention-quality-and-same-implementation-state).
Each passage starts with a fresh owner and state, without BOS or a chat template.
The shared scorer runs 32 eight-row chunks, checks every complete vocabulary
head and records all 255 next-token targets. That gives 1,020 aligned targets
per profile. Eight rows use the already authenticated packed bucket8 and the
existing O occupancy query; no tuning or arithmetic changes. The earlier
43-token controls retain their seven/one-row schedule.

| Profile | Mean NLL | Perplexity | PPL difference from original |
| --- | ---: | ---: | ---: |
| Original BF16 | 4.664608265494 | 106.124004644 | — |
| EXL3 4.0 G | 4.674358690231 | 107.163819846 | +0.979812% |
| EXL3 4.5 G | 4.674169702044 | 107.143569063 | +0.960729% |
| EXL3 4.0 O | 4.674219932654 | 107.148951085 | +0.965801% |
| EXL3 4.5 O | 4.674218332170 | 107.148779595 | +0.965639% |

These are bounded observations against the original weights, not a newly chosen
quality threshold or equal-or-better claim. Quantization, precision and execution
recipes differ, so the gaps do not isolate any single operation. No selection,
fusion or rounding changes were made to reproduce an external quantized
library's bits. The fixtures remain unserved with no speed baseline.

Each actual eight-row plan contains 217 selected EXL3 operations: G has 121 GEMMs,
24 paired MGEMMs and 72 bias operations; 40O replaces 120 GEMMs with GEMVs, and
45O replaces 14. All 169 source products and 676 parts are checked; O retains
its eight actual occupancy facts and launch-time validation. One score key fits
the existing paid two-program domain. The 21,108,224-byte scorer ceiling already
covers 256 IDs and eight rows; the other model/compiler/state and physical caps
are unchanged. No full corpus heads are retained.

Spark A passed the native target build and all 14 focused controls, then all 16
actual document/profile cases with zero failures or skips, all source/binary and
recipe guards passing, no compute processes remaining and zero memory events.
Main independently passes the full native build, the same 14 controls and all 16
actual cases. Every per-target NLL and selected-plan receipt equals the isolated
run. Main process/cache peaks range from 276,647,936 to 278,220,800 bytes;
format, header and license checks pass. Earlier checkpoint/spill/kept/reload
evidence is reused without rerunning those cases.


## Active FP16 benchmark adapter

The FP16 comparison used by the active swap and speculation benchmarks now
uses `ComponentStepSource` and `ModelRunner` with the current Qwen2 mapping,
including its selected fused attention. It returns the same full-vocabulary
row order, weight closures and first-chunk completion time through the existing
benchmark helper API. Metadata, backend/cuBLAS, compiler plans and completion
remain owned by the shared runner's lifetime. Setup measures two control or
five heldout classes once; positions remain runtime data.

The adapter pays a pinned host envelope before construction: 6 GiB for control
or 12 GiB for heldout, including temporary compilation, retained programs and
caller-owned result growth. Local limits enforce that inventory before and
after the node starts. Existing benchmark budgets include this pinned occupancy;
the node's global host policy stays unchanged. Full results remain paid until
the caller releases them and destroys the adapter. Unproven retirement keeps
the complete owner. Historical BP-P flags and launcher recording are explicitly
refused rather than emulated by another family executor.

Main passed a full native build and the insufficient-host-allowance refusal
control, then all three actual checks without failures or skips. The 76-row
control and 577-row heldout trajectories each execute three times and compare
all heads against their own first execution, with all 490 weight extents evicted
and reloaded before the third. The refusal control uses absent input paths and
an unopened node to verify that insufficient host inventory stops construction.
All source, binary, input and recipe guards pass, no compute process remains,
and all memory events are zero. The observed process/cache peak is 2,173,374,464
bytes under the existing 16-GiB zero-swap confinement; this is not device
occupancy. The unchanged mapping reuses its earlier original-reference quality
and state evidence. No external quantized-library bit agreement, serving
qualification or matched speed result is claimed. Legacy standalone executor
and graph deletion remains the following slice.
