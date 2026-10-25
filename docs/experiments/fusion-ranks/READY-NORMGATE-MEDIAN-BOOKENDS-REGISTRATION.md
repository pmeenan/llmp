<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ready NormGate median-bookend screen registration

Prospective registration, 2026-10-25. The
[earlier single-bookend screen](READY-NORMGATE-CAPTURED-EVENTS-RESULTS.md)
was refused for 20% drift; that refusal remains unchanged. This revised
stability check uses fixed medians of nine bookend observations, matching the
nine-sample estimator used for each arm. No comparison under this protocol has
run. Screen the same F32 NormGate local body with
dimension 128, 48 heads and one token row, RMSNorm epsilon `1e-6` and sigmoid
gating, against its own normally planned decomposition. The complete accepted four-member `gdn_norm_gate` group is
retained; its alternative preserves the RMSNorm/Mul child fusion. This is an
isolated Ready-boundary comparison. Its physical key differs from the
[actual full GDN programs](ACTUAL-GDN-FIRST-SCREEN-RESULTS.md); no row or
model-speed claim transfers between them. No accepted NormGate ranking exists.

## Frozen implementation and qualification

| Fact | Value |
| --- | --- |
| Development Git base | `f0ba0a0fc0d48cdb73cb484c1b3c2900f6c8064e` |
| Frozen source | `u6-normgate-median-r50-b`, 118 paths |
| Source manifest SHA-256 | `3bb0af91385a47668b9522584822e2b4d7973305e5d04ae0830a484babf020d8` |
| Native binary SHA-256 | `458023dc847674449a89ad84a49579baf8402393f2f40e7e3570abe6d3ff570a` |
| Backend build | `5bfcfa9ed5b5d01720b9aedf656d9051b93e066930d1ca694827094ff0645ed5` |
| Complete recipe-key digest | `e55a5bb1499432c3147f39f18730844ac8638a137603bdd91ff3fc98406f5452` |
| Preparation domain | `4924c9434424f6bdf10171e29f90fb2b5b27ff9fb34a123368f6ff0142adc710` |
| Host / GPU / SDK | `spark-b` / NVIDIA GB10 / `aarch64-2c4eb5b2f68f3fa5` |
| Protocol SHA-256 | `b06453b54106e89271ee79e8c2b2670cbc31f9de25c12780d43971a7620ca395` |

The shared measured-capture API has clean whole-change and adversarial ownership
reviews. The benchmark-only NormGate migration has its own clean independent
whole/source challenge review. The median-bookend delta also passes independent
whole/source challenge review, native build and functional validation of the
registered cell and existing NEOX exact hit. The preceding capture qualification
also checked an unmeasured two-row NormGate cell. NEOX still recognizes its installed row in the 13-row recipe table;
backend identity is unchanged. Original-IR references validate two changed
out/gate input frames; maximum absolute error is `2.384185791015625e-7` for
both arms, within `1e-5 * (1 + abs(expected))`. Functional duration reads only
check completion and positivity; no ranking sample is collected.

Each recording contains one complete local body and two external event nodes,
for actual graph counts 3/5 (one versus three body kernels). Both arms use the
benchmark's common 32 MiB activation arena and 16 MiB scratch pool. No memory
pressure or model fit conclusion follows from this synthetic cell.

## Run and decision

After this registration is committed, run
`build/spark-native/benchmarks/llmp_fusion_ranks --ready-norm-gate-time 128 48 1`
once in an installed-supervisor exclusive job, bounded to 600 seconds. Verify
frozen source and binary before and after; retain normal completion following
teardown and a separate host-free check.

Repeat two-frame reference validation before comparison. Before every arm,
poison activation storage and restore both original F32 Features inputs.
Finish the empty pure prefix before replay. Read the captured event duration
only after owning-stream completion, then check every CPU output, immutable
gamma and actual capture inventory. Each replay uses fresh original inputs.
The event interval excludes host projection/launch validation and submission;
no adjustment is made for timing-event instrumentation.

Retain all **52** observations in this fixed order:

1. Four fixed AB warmup pairs (eight records).
2. Nine alternative-arm observations (`bookend_before`, pair `-1`, order 0–8).
3. Four alternating AB/BA warmup pairs (eight records).
4. Nine alternating AB/BA sample pairs beginning AB (eighteen records).
5. Nine alternative-arm observations (`bookend_after`, pair `-1`, order 0–8).

Compute each bookend as its nine-observation median. Refuse if the larger
bookend median exceeds 1.1 times the smaller, retaining the same 10% limit and
all individual readings, with no filtering or automatic retry. Compute the two
arm medians only from the nine sample pairs and apply the existing strict 10%
`ClassifyRecipeLatency` band. Median calculations sort copies; raw order and
values remain intact. Refuse nonpositive or invalid intervals.

This tests movement in typical latency and reduces sensitivity to one reading.
Correlated drift can still fail, and medians can conceal occasional slow
executions; every observation remains visible. A pass would not establish the
cause of the earlier refusal. Existing RMS and NEOX measurement protocols are
unchanged.

Export the complete typed key, protocol text/hash, compact ordered-event
receipt/hash, medians and classification. Keep raw candidate provenance
`UNREVIEWED` until independent numerical, source, event-order, schema, drift
and retirement review. Review this screen before expanding or installing data.
Historical rows retain their original keys. This does not complete U6 or
qualify a whole-model optimization.

The exact protocol text is:

```text
llmp-ready-norm-gate-in-capture-single-body-timing-v3;prep=4924c9434424f6bdf10171e29f90fb2b5b27ff9fb34a123368f6ff0142adc710;frames=2-original-ir-validated-before-samples;input=poison-arena-then-restage-both-original-f32-features-before-each-replay;prefix=prepare-once-finish-before-timed-body;body=captured-complete-local-region;body-replays=1;fixed-warmup-pairs=4:incumbent-then-alternative;bookends=alternative-nine-before-and-nine-after:median-of-nine;alternating-warmup-pairs=4:start-incumbent-then-alternate;alternating-sample-pairs=9:start-incumbent-then-alternate;events=two-external-event-nodes-inside-original-body-capture;weights-and-cpu-output-check=after-every-arm;ordered-retained-records=52;filter=none;bookend-drift=10pct;retry=none
```
