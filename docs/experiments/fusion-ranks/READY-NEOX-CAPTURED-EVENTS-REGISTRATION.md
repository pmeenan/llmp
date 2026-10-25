<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ready NEOX norm/RoPE captured-event screen registration

Prospective registration, 2026-10-25. This is a new measurement method after the
[first external-event screen](READY-NEOX-FIRST-SCREEN-RESULTS.md) was refused
for 14.573% bookend drift. That result remains refused and is not retried.
No comparison with the captured-event method has run.

Screen the same F32 width-128, two-row, no-frequency NEOX RMSNorm/Mul/RoPE
complete accepted group against its own normally planned decomposition on
`spark`. The alternative preserves its RMSNorm/Mul child fusion and competing
rewrite exclusions. Retain the original 10% drift and decision bands.

## Implementation and qualification

| Fact | Value |
| --- | --- |
| Development Git base | `f0ba0a0fc0d48cdb73cb484c1b3c2900f6c8064e` |
| Frozen source | `u6-captured-events-r47-a`, 118 paths |
| Source manifest SHA-256 | `68e84aee0b115d519d1959b4cc06bd755325031212663e0363ec67bd13deafac` |
| Native binary SHA-256 | `dc2406f618c66c76ec1e0438c1772624bbdae575c8a3dd781625dc8cb75b9c2f` |
| Backend build | `5bfcfa9ed5b5d01720b9aedf656d9051b93e066930d1ca694827094ff0645ed5` |
| Complete recipe-key digest | `4c8c0fafa6b61b47e0f6624cd337d6c0e6ed8d749fc8b78b207083d264b203fc` |
| Ready preparation domain | `4924c9434424f6bdf10171e29f90fb2b5b27ff9fb34a123368f6ff0142adc710` |
| Host / GPU / SDK | `spark` / NVIDIA GB10 / `aarch64-2c4eb5b2f68f3fa5` |
| Protocol SHA-256 | `33b6481d291371d38fb42ce6b29113a3c64c4c2a450f5471024b54899cae208c` |

Independent whole-change and adversarial CUDA ownership reviews are clean after
fixes. Native qualification passes all 16 focused checks, including the pending
replay with an older completed stop event, failed measured capture followed by
ordinary capture, default zero-event behavior and current-owner checks. The
instrumented NEOX 128-by-2 functional invocation validates both changed frames
with maximum absolute errors `4.287809133529663e-6` and
`4.302710294723511e-6`, within `1e-5 * (1 + abs(expected))`.
Functional duration reads only verify positivity and completion; no ranking
sample or performance comparison has run. The combined shared regression wave
is separate from this first screen.

Timing is default-off in ordinary captures. For this screen only, each original
complete body capture contains an external start-event node, its unchanged body,
and an external stop-event node. Report actual graph node counts 3/4 (body
kernels 1/2 plus two timing nodes); each recording contains exactly one body.
Duration is read only after successful replay and known completion on the
owning stream. The interval excludes CPU projection/launch validation and graph
submission. The event nodes may perturb execution; this screen does not claim
uninstrumented model speed or guarantee stable measurements.

## Run and decision

After this registration is committed, run
`build/spark-native/benchmarks/llmp_fusion_ranks --ready-norm-rope-time 128 2`
once in an installed-supervisor exclusive job with a 600-second bound. Verify
frozen source and binary before and after; retain normal completion following
teardown and a separate host-free check.

Validate both changed original-IR frames before comparison. Before every arm,
poison the activation arena and restore original F32 Features and I32 Positions.
Finish the pure prefix outside the timed body, replay once, finish, read the
captured interval, then check every CPU output, immutable gamma, positions and
actual capture inventory. No repeated body uses previously overwritten inputs.

Use four fixed AB warmup pairs, one alternative bookend, four alternating warmup
pairs, nine alternating AB/BA sample pairs beginning AB, then one alternative
bookend. Retain all 36 records in order. Refuse if the larger bookend exceeds
1.1 times the smaller, with no automatic retry or threshold change. Compute
medians from copies of the nine sample values per arm, using the existing
`ClassifyRecipeLatency` 10% band. Refuse invalid/nonpositive intervals.

Export exact complete recipe key, source/binary/backend/protocol identities,
ordered event receipt and hash, medians and classification. Candidate provenance
stays `UNREVIEWED` until independent numerical, schema, event-order, drift and
retirement review. Do not install data or expand the matrix before reviewing
this screen. Historical RMS measurements keep their original identities.
No model applicability, adoption or U6 completion claim follows from this run.

The exact protocol text is:

```text
llmp-ready-neox-norm-rope-in-capture-single-body-timing-v2;prep=4924c9434424f6bdf10171e29f90fb2b5b27ff9fb34a123368f6ff0142adc710;frames=2-original-ir-validated-before-samples;frequency=none;input=poison-arena-then-restage-original-f32-features-and-i32-positions-before-each-replay;prefix=prepare-once-finish-before-timed-body;body=captured-complete-local-region;body-replays=1;native-nodes=3,4;body-nodes=1,2;fixed-warmup-pairs=4:incumbent-then-alternative;bookends=alternative-before-and-after;alternating-warmup-pairs=4:start-incumbent-then-alternate;alternating-sample-pairs=9:start-incumbent-then-alternate;events=two-external-event-nodes-inside-original-body-capture;gamma-positions-and-cpu-output-check=after-every-arm;bookend-drift=10pct;retry=none
```
