<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ready local RMSNorm/Mul: remaining shape registration

Prospective registration, 2026-10-25 UTC. The width-1024, one-row
[first screen](READY-FIRST-SCREEN-RESULTS.md) passed independent review;
it used 23.6857% less time for the fused recipe, with 0.2206% bookend drift.
No additional ready-body timing has run at this registration. Measure only
the eleven remaining tuples below. The first-screen tuple is not repeated.

## Frozen implementation

| Fact | Value |
| --- | --- |
| Git implementation base | `f0ba0a0fc0d48cdb73cb484c1b3c2900f6c8064e` plus reviewed uncommitted U6 source |
| Source cut | `u6-integrated-authored-r34`, 113 paths |
| Source manifest SHA-256 | `15d71c2e924c9bb87208f812f961aef2b3bf7242d2af013f6e4bff2172947a0b` |
| Compiler-owned implementation provenance, both hosts | `94a79b540f8557318d52b02346b971a1843eb8e39c0f08707d944db74acbb5d9` |
| Spark A benchmark SHA-256 | `28ade247a2d2f16150317a3e84b1f85d23d515ad349a05b9434b3d3f61505cb1` |
| Spark B benchmark SHA-256 | `36ab4c60f462f6ad7cf4f74e334da994534d47f8f83367bb383c8736c501f758` |
| Hosts / device / driver | `spark` (`spark-c4e2`), `spark-b` (`spark-56f5`) / NVIDIA GB10 / `580.178.04` |
| Pinned SDK | `aarch64-2c4eb5b2f68f3fa5` |
| Protocol SHA-256 | `e013f06a8ebc77fb66c36654e3828c41a7650d1485922b1415852bddfe7d30c6` |

The integrated real-runner selection/cache qualification passes 1/1 on Spark A,
with no skips. Its validation-only benchmark checks two changed-original-input
frames against the original-IR CPU reference on the actual 128-invocation
recordings: 128 fused nodes versus 256 decomposition nodes, with maximum
absolute error `2.384185791015625e-7` in each arm. Prefix and immutable weight
bytes remain unchanged. Its width-1024, one-row recipe key and both physical
arm identities match the original Spark B first-screen identities exactly.
The earlier integrated 54-check comparison qualification remains applicable.
All 113 frozen paths match root and both hosts. A complete tracked-file audit
found only documentation differences after restoring a preserved temporary
Spark A diagnostic; compiler-owned declaration and object identities agree.
The remote Git HEAD labels remain inherited dirty-checkout labels. Each
host's independently recorded binary hash authenticates its actual executable;
the binary hashes need not equal across hosts.

## Scope and protocol

Measure the eleven remaining F32 width/row tuples in the ready RMSNorm/Mul
class, using each fused local recipe's own separately planned decomposition.
Preserve the first screen's exact protocol: two changed-input original-IR
reference frames, 128 captured local-body invocations per arm, original-input
restaging and actual pure-prefix preparation outside timing, immutable-weight
and prepared-prefix checks, 36-event sequence, and 10% bookend drift gate.
Use the existing `ClassifyRecipeLatency` coarse 10% band. Numerical validation
uses the benchmark's existing relative RMS reference bound; do not tighten or
relax it for this matrix. Each timing invocation validates its actual recordings
before collecting events; separate per-tuple validation-only runs are unnecessary.

```text
llmp-ready-body-timing-v1;prep=5f7efa3f25e0b95af4c4ddad66fa2093343a767104c0804a2e73a278d1b1de15;frames=2-original-ir-validated-before-events;input=restage-original-before-each-block;prefix=prepare-once-before-events;body=captured-complete-local-region;body-replays=128;fixed-warmup-pairs=4:incumbent-then-alternative;bookends=alternative-before-and-after;alternating-warmup-pairs=4:start-incumbent-then-alternate;alternating-sample-pairs=9:start-incumbent-then-alternate;events=one-start-stop-per-captured-block;bookend-drift=10pct
```

## Per-host tuples

| Host | Width | Rows |
| --- | ---: | ---: |
| Spark A (`spark`) | 256 | 1 |
| Spark A (`spark`) | 256 | 4 |
| Spark A (`spark`) | 256 | 32 |
| Spark A (`spark`) | 256 | 256 |
| Spark A (`spark`) | 1024 | 4 |
| Spark A (`spark`) | 1024 | 32 |
| Spark B (`spark-b`) | 1024 | 256 |
| Spark B (`spark-b`) | 4096 | 1 |
| Spark B (`spark-b`) | 4096 | 4 |
| Spark B (`spark-b`) | 4096 | 32 |
| Spark B (`spark-b`) | 4096 | 256 |

## Run and evidence

Run only after this registration is committed and reviewed.

Run one exclusive supervised job per host, each at most 600 seconds, with no
concurrent build or functional work on that host. For each assigned tuple run
`build/spark-native/benchmarks/llmp_fusion_ranks --ready-norm-time WIDTH ROWS`
in a fresh process. Authenticate the same source manifest and each host's
registered binary hash before and after every tuple. Copy this committed
registration to each host and authenticate its exact bytes, recording its
commit without changing the remote Git checkout.

Retain all 36 ordered events: four fixed-order incumbent/alternative warmup
pairs, the alternative first bookend, four alternating warmup pairs, nine
alternating AB/BA sample pairs, and the alternative final bookend. Each
alternating group starts with the incumbent. Compute medians from copies of
the nine samples per arm. Refuse evidence when the larger bookend exceeds
1.1 times the smaller. Preserve refused observations and stop that host's
remaining tuples on any refusal, identity mismatch, numeric failure or failed
retirement. Do not retry automatically or change a threshold after observing it.

Preserve each actual complete typed recipe key and both arm identities; keys
vary by shape and must not be inferred from historical whole-program query
hashes. Export the exact protocol text/digest and SHA-256 of compact emitted
event-array bytes. Keep full stdout/stderr, supervisor completion and GPU
retirement receipts outside Git. Independently audit these and the source,
class, schema, medians and classifier before installing any candidate.
Candidates remain `UNREVIEWED` until that audit; this registration installs
no rows, relabels no historical data and makes no model-speed or full-U6 claim.
