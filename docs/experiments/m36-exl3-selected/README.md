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
The 4.5-bpw O profile, broader shapes, O lifecycle coverage, Tier C,
performance and legacy deletion remain open.
