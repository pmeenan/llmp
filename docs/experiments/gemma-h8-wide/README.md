<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Internal Gemma3 grouped whole and partial owner waves

The internal H8/no-softcap path now executes whole C8/C12 and every partial
cohort from five through eleven owners with existing four-root attention
carriers. One continuous unequal-prefix schedule passes the original same-format
stock strict gate: **472 full heads, 468 scored targets, zero greedy or tie
differences, and 276 byte-exact heads**. Native repeated heads, full/device
choices, initialized states and checkpoint restoration are exact.

Public admission remains two slots. A separate C12 natural performance screen
retains an **official failed exact-final-head gate**: all 384 consumed natural
choices and final cursors match across all four arms, and each engine's final
heads repeat exactly, but no cross-engine final row is byte-exact. The finite
final rows have zero argmax differences, maximum absolute difference
1.90735e-6 and mean total variation 9.44933e-9. No threshold or gate was changed.
Descriptive native paid latency is 11.3219% above stock, including 23.5760%
more decode time. This is a short n=2 measurement, with concurrency, sustained
performance, peak memory and wider-context batching still open.

## Implementation and operand proof

Only attention splits into groups of at most four actual roots. Products,
including twelve-column products, retain the whole wave. All attention groups
use the same logical cohort and the maximum global/local read width across
**all** owners. Actual short roots are padded using real funded F16 Fill/Concat
activations; initialized source state and direct SET_ROWS dependencies retain
bounds. Whole C8/C12 use the existing divided original grid with encoded
`owner_offset=0`; physical source groups start at 0/4/8. Partial C5..7/C9..11
instead use canonical encoded offsets 0/4/8, with one, two or three actual tail
roots and null unused slots.

No floating CUDA kernel, quantization, importer, state layout or public surface
changes. Cap50 and copy-free bounded roots remain C2-only. Paired prefill still
admits at most two 128-row chunks, with total wave 256. Setup funds equal
endpoints and every proper long-owner multiplicity for each admitted count;
host controls check exact and one-byte-short activation/source capacity, common
width from the last group, all padding outputs and immutable head capacity 12.

The new operand control compares independent grouped roots against the original
packed physical cohort for logical counts 5 through 12, at D256/H8/GQA2 and
512 cells. Eager, captured and fresh query/mask replays are byte-exact; poisoned
scratch/output and source immutability checks pass. FP64 maximum group NMSE is
3.28053e-7 under the existing 5e-4 bound. Whole C8 preserves original grid 48
as two groups of 24; whole C12 uses three groups of 16. Partial C9..11 include
small exact reduction scratch and one-byte-short refusal controls. Logged group
starts 4/8 for whole C8/C12 describe physical roots, not encoded offsets.

There are 39 passing focused test executions: 19 prerequisite executions
(one new operand, eight host controls and ten focused legacy controls), followed
by eight graph and twelve plan controls on the actual model build. No skipped
tests or full suite run.

## One continuous unequal/arrival schedule

Twelve real independently funded states use two authenticated input identities:
256- and 768-token prefixes, assigned by pairs as **short, long, short, short,
long, long**. The first quad is mixed, the second all short and the third all
long. This exercises a common maximum supplied outside the all-short group in
both C8 and C12. Duplicate identities are explicit; this is not a twelve-prompt
corpus or cross-conversation shared-prefix test.

The shared event schedule records every owner, cursor, target and published row:

1. Round-robin paired 128-row prefill and three supplied scalar rows per owner.
2. C12×4, C11/C10/C9×2, C8×4, C7/C6/C5×2 and C4×4 waves, checking every paused
   suffix before and after its active phase.
3. Scalar catch-up of owners 4..11, protecting the first four peers, then C12×8
   rejoin waves.
4. Clear and paired reprefill of owners 10/11, preserving all ten peers around
   each refill chunk, with two explicitly interleaved unequal C4 waves.
5. Three scalar startup rows for the returning pair and two scalar tails for
   every owner.

Independent analysis derives 472 publications and 468 scored transitions. Only
four final heads reach the supplied-history end; the other eight retain defined
next supplied targets. Final cursors are prefix+39 for owners 0..3, prefix+37
for 4..9 and prefix+5 for 10/11. The device pass has 460 token publications plus
twelve full final rows, reproducing the full-head pass exactly. All 36 protected
peer checks, repeated native final states and spill/restore checks pass. These
are same-schedule rejoin controls, without a new solo/batch equality claim.

Two native own runs freeze before either reference run. Two original stock
teachers repeat exactly and retire completely. Stock metadata-only callbacks
witness actual logical counts 4 through 12 and paired prefill without requesting
operand downloads. Same-schedule comparison has:

| Metric | Result |
| --- | ---: |
| Finite full heads / scored targets | 472 / 468 |
| Byte-exact native/stock heads | 276 / 472 |
| Greedy / positive-margin / exact-tie differences | 0 / 0 / 0 |
| Mean target NLL delta | 1.14872456e-5 |
| Relative conditional-loss delta | 1.14873116e-5 |
| Mean total variation | 1.20421060e-5 |
| Maximum absolute logit difference | 0.079269886 |

This passes the existing zero-choice-difference and 3% conditional-loss gate.
It does not accept the previously recorded twelve solo/C4 geometry changes or
introduce a numerical allowance for concurrency.

The standalone own harness funds its complete finite two-pass cache-key set,
including state-only chunks, without claiming an unregistered reclaimer. Both
runs retain 66 plans, accounting for 1,314,321,136 plan/graph bytes, with zero host
overcharges. The conservative cache funding is 9,980,118,720 bytes; total
all-owner virtual-state capacity is 3,925,868,544 bytes and startup capacity
16,795,865,792 bytes. These diagnostic commitments are **not peak-memory
measurements**. Exclusive head files stream outside any performance timer.

## Separate C12 natural cost screen

The separate callers run one warm prefix/startup/eight-wave pass, then logical
Clear without zeroing or discarding physical backing. The warm eight waves
preserve the existing C4 recipe: five token waves followed by three full-head
waves, matched in stock. There is no extra warm pass.

Paid prefill/startup includes all 24 paired chunks / 6,144 prefix rows **and all
36 supplied scalar writes**. Paid decode consumes 384 natural choices in 32
whole-C12 waves, with twelve final full heads paid on both engines. No state
hashing, head-file writes or extra head publication is timed. The original
stock backend greedy API still exports full sampled vocabulary rows on every
head; the caller does not suppress or replace that behavior. Stock uses F16 KV,
`swa_full=false`, `kv_unified=false`, logical reset `false`, no shape observer
and no added state-only synchronization.

The R1/N1/N2/R2 bookend completes and retires all four arms. Same-engine choices,
cursors and final heads repeat exactly; all engines consume identical 384
natural choices and end at prefix+35. The original aggregate then fails its
stricter cross-engine final-head byte-equality assertion. That failed receipt
and all outputs remain; there is no rerun, retroactive pass or new tolerance.
Read-only final-head analysis finds twelve finite rows, zero argmax/positive/tie
differences, maximum raw delta 1.90734863e-6 and mean TV 9.44932536e-9. No frozen
next-token labels exist for these final free-running histories, so a final-head
target-loss result is undefined.

The following elapsed values are descriptive for that exact-history screen:

| Paid phase | Native mean | Original stock mean | Native relative wall |
| --- | ---: | ---: | ---: |
| Prefill + 36 startup writes | 1.462820 s | 1.398585 s | +4.59286% |
| 32 C12 decode waves + final heads | 0.9490585 s | 0.7679955 s | +23.5760% |
| Sum of paid phases | 2.4118785 s | 2.1665805 s | +11.3219% |
| Aggregate decode rate | 404.612 tok/s | 500.003 tok/s | descriptive only |

Both native arms record twelve captured and 48 replayed prefill/startup paths,
then **32 replayed decode paths with no paid decode captures**. Thus paid decode
capture does not explain this screen's gap. Capture is recorded as an exclusive
path even though device work runs eager beside capture. Each timing arm retains
32 plans / 636,493,456 accounted bytes with zero overcharges, funded by a
conservative 160-event plan/graph envelope.

A source-backed next causal factor is common-width cache materialization: six
short roots at 512 cells are padded to 1,024 across 34 layers, producing 408 K/V
Concat nodes per decode wave. Derived output writes alone total 816 MiB/wave;
this is source volume, **not measured copy time or removable wall latency**.
The first separate optimization should preserve the whole logical partition
while bounding actual reads per carrier, including the all-short quad. Partial
carriers and cap50 remain separately guarded. Other structural leads, such as
per-owner SET_ROWS and grouped mask/output operations, remain unmeasured.

## Reproduction and binding

Use the approved [Gemma3 pin](../gemma3-execution/pins.json), original v0.6.0
CUDA image/library closure and external input texts/IDs. The
[native quality caller](../../../benchmarks/gemma3_wide_probe.cc) takes
`ARTIFACT IDS0 IDS1 NEW_OUT`; run two fresh directories before reference. The
[stock quality caller](../gemma3-execution/llama_wide_probe.cc) takes
`MODEL IDS0 TEXT0 IDS1 TEXT1 NEW_OUT` inside the original image. Then use
[the quality analysis](../gemma3-execution/analyze_wide.py) as
`own ROOT MODEL_JOB_LOG`, followed by `quality ROOT MODEL_JOB_LOG` after two
checked stock retirements. The [shared event map](../../../benchmarks/gemma3_wide_schedule.h)
and analysis independently define publication/target/cursor counts.

The [native cycle caller](../../../benchmarks/gemma3_wide_cycle.cc) and
[stock cycle caller](../gemma3-execution/llama_wide_cycle.cc) take the same
respective arguments in new R1/N1/N2/R2 output directories. Run the
[original cycle aggregate](../gemma3-execution/analyze_wide_cycle.py) as
`ROOT PERFORMANCE_JOB_LOG`; it deliberately reproduces the preserved
cross-engine final-head assertion failure on these outputs. Complete aggregate
values and source/binary identities are in [results.json](results.json).

Measured on Spark B, GB10/driver 580.178.04/SDK
`aarch64-c09daba6ac31edee`, `sm_121`, on 2026-10-07. Prerequisite true parent
`af71732`; model and cost source true parent `08dbd94`. Official prerequisite
and model jobs finish DONE0; the performance job retains FAIL at aggregate
step six after five successful steps. Every job uses installed GPU600/stopfail
supervision, checksum source sync with an empty dry-run, actual build/source/
binary/receipt binding and checked process retirement. Original preoptimization
ELFs and receipts are retained separately before further builds.

Final source is additively composed onto `b9a35fc`, preserving its trained-max
C1 source/settings/report. The twelve preexisting measured source paths are
byte-exact; benchmark CMake only gains the new targets, and final docs/results
are additive. There is no rebuild for unrelated parent changes and no claim
that those incoming paths executed in the frozen wide measurements.

| Identity | SHA-256 |
| --- | --- |
| Approved source, 2,526,080,992 bytes | `ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a` |
| Prepared artifact | `8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb` |
| Short input, 295 i32 | `c37e404d378484e6dbf4da9a7f7e812105b569eac9e8323e9dd49e47d57dc470` |
| Long input, 807 i32 | `60231d6717a7508c267f30302942d7ad2cd95291ba206ce300ec4769cc1295c1` |
| Native quality ELF | `6398a1a572b435b3dfa16914e4782548a95eb5dd30fadbe72fe108e324e837f2` |
| Native cycle ELF | `b079bb8ba9f34f3970427ec4dd0ec6ff38042b7800efbe6824799d2547ac1c69` |
| Build receipt | `874aaf7a5967cfbe91054e0d8fc1a0630f952e0e8eb54b831d09719f1e08ce89` |
| Complete model measured source inventory | `bd9435cacf742c0f5965313cc9399ae17ea0028cdcb019cf8bb8ef4d93016da1` |
| Strict model quality aggregate | `d58657901779bcc83184912b1eb6b734163cb325377d4fdea37ae7692593886f` |

Task-entry TensorFold HEAD `041d14a94e951834470fd514ed33e65b8be1059a`, version
1.0.0, was observed on 2026-10-07. Its documented GB10 CUDA recipe qualifies
Nemotron, without a matching Gemma3/Gemma2 CUDA GGUF comparator. Historical
pins remain unchanged.
