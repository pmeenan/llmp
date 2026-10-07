<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma3 whole-C12 bounded-root factor

The internal whole-C12 factor removes short-cache materialization while keeping
all 384 natural choices, twelve cursors and twelve final native heads byte-exact
across both policies and repetitions. One short same-binary baseline/candidate/
candidate/baseline bookend reduces paid decode by **21.0667%** and total paid
latency by **8.26029%**. The option remains false by default; ordinary public
admission remains two slots. This result does not close the earlier stock
final-head byte gate, concurrency, full-corpus, wider memory or sustained gates.

## Checked factor

`Gemma3Options::bounded_whole12` and the corresponding graph option select only
no-cap D256/H8/GQA2, one-query, whole-C12 attention. All twelve product columns,
the cohort-wide local/global mask maximum and the original stream-K grid remain.
Attention alone uses three four-root carriers. Each can address aligned actual
K/V prefixes directly, including an all-short carrier below another group's
larger mask maximum. The graph pads absent mask lanes with `-Inf`; fresh source
validation still checks each owner's causal/window masks. Address, stride,
independent-root and alias checks remain in force.

The low-level exception is closed to actual four roots, logical cohort twelve,
zero offset, D256/H8 and zero softcap. Existing C2 maximum-width validation is
unchanged. Bounded partial cohorts and other wider signatures remain refused;
all their real padding stays funded. No CUDA floating kernel, tile, occupancy
policy, reduction or empty-partition implementation changed. Equal carriers
still use the legacy specialization when no root is shorter than the mask.

The fixed operand control has logical width 1024 and actual widths 512/1024.
Its groups are mixed, all-short and all-long. NaN guard backing lies beyond each
actual root. Original packed padded MMA and each bounded carrier match byte for
byte, including eager, captured repeats and fresh Q/mask replay; sources and
poisoned guards remain unchanged. The original 48-block grid divides into
16 blocks per carrier; original and bounded occupancy are one block per SM,
with 16 bytes of scratch per carrier. FP64 NMSE is 3.60884e-7 overall and at most
4.91039e-7 per group, within the unchanged 5e-4 bound. Exact and one-byte-short
scratch controls pass.

The host plan control independently verifies that 408 K/V CONCAT sources become
zero, while whole-wave products, logical masks, actual cache bounds and source
writer dependencies remain. Exact and one-byte-short activation/source funding
controls pass. The supervised prerequisite completes with 14 passing executions:
one new operand, one plan, eight graph and four affected legacy controls; none
are skipped.

## Matched native bookend

Both policies use the same newly built executable. The strict optional final
argument `bounded-whole12` is the only candidate difference; no argument keeps
the padded baseline. There are twelve real private states and two authenticated
histories, duplicated as in the [wide foundation](../gemma-h8-wide/README.md):
long owners 2, 3, 8, 9, 10 and 11; short owners 0, 1, 4, 5, 6 and 7. Context is
4096, per-owner chunk capacity 128, paired prefill capacity two, total wave
capacity 256 and immutable frontier capacity twelve. Model, format, fusion,
attention and head-publication policies otherwise remain fixed.

One ordinary warm traverses 24 paired chunks, 36 supplied scalar writes and
eight natural C12 waves (five device-token waves then three full-head waves).
Logical Clear is off-clock and retains backing/plans. Paid prefill includes
6144 paired-prefix token rows **and all 36 supplied scalar writes**. Paid decode
consumes 32 natural C12 waves / 384 choices and publishes twelve final full heads
on the last wave. Output files, finiteness checks and hashes are off-clock.

| Arm | Prefill including scalar writes (s) | Decode including final heads (s) |
| --- | ---: | ---: |
| Padded B1 | 1.466130 | 0.948853 |
| Bounded N1 | 1.465670 | 0.749269 |
| Bounded N2 | 1.469310 | 0.749599 |
| Padded B2 | 1.468040 | 0.950051 |
| Padded mean | 1.467085 | 0.949452 |
| Bounded mean | 1.467490 | 0.749434 |

Mean total paid latency is 2.416537 → 2.216924 seconds, a 199.613 ms reduction
(8.26029%). Decode falls 200.018 ms (21.0667%); prefill changes +0.405 ms
(+0.02761%). Aggregate decode rate is 404.444 → 512.387 tokens/s. These are n=2
short native measurements, including policy allocation/storage/launch effects;
they do not equate a summed copy-kernel duration from another run to saved wall
time.

All four processes retire successfully. All 384 choices, twelve cursors and
twelve finite final heads are byte-exact across policies and repetitions, and
match the immutable earlier native outputs. Both modes have twelve paid
prefill captures / 48 replays and 32 paid decode replays / zero decode captures.
Selected bounded attention is 136 for each candidate and zero for each baseline;
these are bound-plan selection counts, not replay kernel counts. Both retain
32 plans with zero host overcharges. Accounted plan/graph bytes fall from
636,493,456 to 621,982,864; this is accounting evidence, not measured peak memory.
Both official jobs finish DONE0.

The [earlier stock comparison](../gemma-h8-wide/README.md) remains FAIL at its
preregistered cross-engine final-head byte gate. It had exact natural choices
but maximum final-head delta 1.90735e-6 and zero argmax differences. The present
factor reproduces those original native heads exactly; it does not fix or waive
that gate. No fresh stock process ran here. Earlier stock timings remain
historical context, not a new matched parity result.

## Provenance and remaining gates

Measured source is `14bc98ecdc153b1f28f568d9f44b8628d0658449` plus this
bounded-root factor. The separate `cba9ec0` host-mask factor is retained in the
combined source but was not part of this timing.
[Aggregate results](results.json) bind actual source, method, binary, receipt
and official outcomes. Raw arrays, logs, XML and private methods remain external.

Actual native cycle SHA-256 is
`f4ab5d1fbf0ffe03ef55ec70ac76b86da0e49fe4f0626dc535e8ee10635934cf`;
source inventory SHA-256 is
`c0a6926b74e2eb8e27be07d0b88d9961b427553cba71549dfd6c2b2716152ab9`.
Official jobs are `m35-gemma-h8-wide-bounded-check1` and
`m35-gemma-h8-wide-bounded-performance1`, both DONE0. The original native ELF
`b079bb8ba9f34f3970427ec4dd0ec6ff38042b7800efbe6824799d2547ac1c69`
is retained separately. The measured strict native analyzer is checked in as
[analyze_wide_bounded_cycle.py](../gemma3-execution/analyze_wide_bounded_cycle.py).

The approved Q4_0 model source SHA-256 is
`ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a`;
artifact ID is `8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb`.
Inputs retain the wide report's exact 295/807-token source identities. Execution
uses Spark B GB10, driver 580.178.04 and SDK `aarch64-c09daba6ac31edee`; receipt
SHA-256 is `874aaf7a5967cfbe91054e0d8fc1a0630f952e0e8eb54b831d09719f1e08ce89`.

Task-entry TensorFold HEAD was
`5a73b85289b58c0998d7754822145466687a551b` (version 1.0.0), read on
2026-10-07. Its README documents CUDA GB10 Nemotron and no matching Gemma3
CUDA/GGUF reference. Historical pins are preserved.

Remaining gates include fresh reference/quality/public and sustained
qualification, broader bounded carriers, and the unchanged solo/batch
concurrency question. Per-owner cache stores and grouped attention/mask/output
work remain source-backed leads, not attributed residual costs. Earlier Gemma31
store grouping reduced kernel sums but was end-to-end neutral; it must not be
assumed to explain this model's remaining cost.
