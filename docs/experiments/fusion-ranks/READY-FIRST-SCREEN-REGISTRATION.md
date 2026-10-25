<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ready local RMSNorm/Mul: first-screen registration

Prospective registration, 2026-10-24. No ready-body timing has run at this
registration. Run one F32 width-1024, one-row comparison on `spark-b` (GB10),
then review it before expanding the matrix or installing data. This compares
the actual fused local recipe with its own separately planned decomposition.
The original twelve whole-program rows keep their historical identities.

## Frozen implementation and functional qualification

| Fact | Value |
| --- | --- |
| Git base | `f0ba0a0fc0d48cdb73cb484c1b3c2900f6c8064e` |
| Reviewed integration cut | `u6-integrated-ready-body-r32`, 112 source paths |
| Source manifest SHA-256 | `d7dc4f34a82ea5921e19dfb7244fd0c0e393af3ac7669ba18ea021f5e56915b7` |
| Native benchmark SHA-256 | `36ab4c60f462f6ad7cf4f74e334da994534d47f8f83367bb383c8736c501f758` |
| Host / GPU / driver | `spark-b` (`spark-56f5`) / NVIDIA GB10 / `580.178.04` |
| Pinned SDK | `aarch64-2c4eb5b2f68f3fa5` |
| Ready preparation domain | `5f7efa3f25e0b95af4c4ddad66fa2093343a767104c0804a2e73a278d1b1de15` |
| Exact recipe-key digest | `0a5b4a15c30da277991c2c4ca8413b7de68c87d4003a911cda3a4a1d1f74e93f` |
| Protocol SHA-256 | `e013f06a8ebc77fb66c36654e3828c41a7650d1485922b1415852bddfe7d30c6` |

The integration passes 54 focused CPU/GPU checks with no skips. Its
validation-only benchmark captures the same 128 complete local-body invocations
used by timing: 128 fused GPU nodes versus 256 decomposition nodes. Both
schedules pass two changed-original-input frames against the original-IR CPU
reference, with maximum absolute error `2.384185791015625e-7` in each arm.
Prepared prefix operands and immutable weight bytes remain unchanged. No
CUDA timing event or sample was collected by this qualification.

## Run and decision

Run `build/spark-native/benchmarks/llmp_fusion_ranks --ready-norm-time 1024 1`
in one installed-supervisor exclusive job, at most 600 seconds. Expected runtime
is seconds after the validated native build. Authenticate the frozen source
manifest and binary before and after the run. Retain the supervisor's actual
completion record, flushed events, candidate row, protocol and receipt digest
in external scratch; completion must follow teardown.

Every invocation first repeats the two-frame CPU-reference validation on its
actual 128-invocation recordings. Before each timed block, restage the original
input and run its actual pure prefix outside the CUDA events. One event pair
times one launch of the complete captured local body, normalized by 128.
Check the prepared operand and weight bytes again after each block.

Use four fixed-order paired warmups (incumbent then alternative), the
alternative-arm first bookend, four alternating paired warmups, nine alternating
AB/BA sample pairs, and the alternative-arm final bookend. Each alternating
group starts with the incumbent. Preserve all 36 warmup/sample/bookend events
in their original order; medians use copies of the nine measured values per
arm. Refuse evidence when the larger bookend exceeds 1.1 times the smaller;
keep a refused run's observations and do not change the bound afterward.
Classify the medians with the existing `ClassifyRecipeLatency` coarse 10% band.

Export the complete typed `RecipeComparisonKey` and both complete arm identities,
medians, sample count, exact versioned protocol digest/text, and SHA-256 of the
emitted compact JSON event-array bytes. The candidate retains `UNREVIEWED`
provenance until its source, numeric controls, event order, drift, schema and
retirement are independently checked. Review this first result before deciding
further measurement scope. This screen makes no model-speed or full-U6 claim.

The exact protocol text hashed above is:

```text
llmp-ready-body-timing-v1;prep=5f7efa3f25e0b95af4c4ddad66fa2093343a767104c0804a2e73a278d1b1de15;frames=2-original-ir-validated-before-events;input=restage-original-before-each-block;prefix=prepare-once-before-events;body=captured-complete-local-region;body-replays=128;fixed-warmup-pairs=4:incumbent-then-alternative;bookends=alternative-before-and-after;alternating-warmup-pairs=4:start-incumbent-then-alternate;alternating-sample-pairs=9:start-incumbent-then-alternate;events=one-start-stop-per-captured-block;bookend-drift=10pct
```
