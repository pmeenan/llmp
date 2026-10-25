<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Ready local RMSNorm/Mul: first-screen result

The registered F32 width-1024, one-row screen completed on 2026-10-25 UTC. The
measured candidate classified as `incumbent`: the fused incumbent median was
`1.379 us` per captured body, versus `1.807 us` for the decomposition. This is
one local ready-body screen, not a whole-model speed result or a completion
claim for U6. The candidate remains `UNREVIEWED` and was not installed in the
comparison data.

## Measurement and validation

| Item | Result |
| --- | --- |
| Host / device / driver | `spark-b` (`spark-56f5`) / NVIDIA GB10 / `580.178.04` |
| Supervised job | `u6-ready-first-screen-d194bd66-r1`, exclusive, rc 0 |
| Command | `llmp_fusion_ranks --ready-norm-time 1024 1` |
| Source cut | `u6-integrated-ready-body-r32`, 112 paths |
| Source manifest SHA-256 | `d7dc4f34a82ea5921e19dfb7244fd0c0e393af3ac7669ba18ea021f5e56915b7` |
| Remote Git HEAD label | `61f8426f7e2f45931d91645b016069e13f6f2236` (inherited dirty-checkout label) |
| Benchmark binary SHA-256 | `36ab4c60f462f6ad7cf4f74e334da994534d47f8f83367bb383c8736c501f758` |
| SDK | `aarch64-2c4eb5b2f68f3fa5` |
| Protocol registration commit | `d194bd66c1e2b006c801e7989ad265ebc0a0e2c3` (freezes the protocol; not the measured code HEAD) |
| Preregistered protocol SHA-256 | `e013f06a8ebc77fb66c36654e3828c41a7650d1485922b1415852bddfe7d30c6` |
| Emitted event-array receipt SHA-256 | `7dea755b056beaefdc2a06fb16da22154b82b88269d83e7ede90bd0c897945d8` |

Both arms were checked against the original-IR CPU reference on two changed
input frames before timing. Maximum absolute error was `2.384185791015625e-7`
for each arm. The recordings contain 128 fused nodes and 256 decomposition
nodes. Prepared prefix operands and immutable weights remained unchanged.

The run retained all 36 events: four fixed-order paired warmups, the first
bookend, four alternating warmup pairs, nine alternating sample pairs, and
the final bookend. Each arm has nine measured samples. The alternative-arm
bookends were `1.817 us` and `1.813 us`, a `0.2206%` drift against the
registered 10% limit. The medians were `1.379 us` (incumbent) and `1.807 us`
(alternative); thus the incumbent used `23.6857%` less time than the
alternative, and the existing classifier selected the incumbent.

## Exact comparison identity

Recipe-key digest:
`0a5b4a15c30da277991c2c4ca8413b7de68c87d4003a911cda3a4a1d1f74e93f`

The complete key is version 1, region canonicalizer 2, discovery operation
`fused.rms_norm_mul`, mode `full`, with these shared fields:

| Field | Digest |
| --- | --- |
| Subject | `6704cc8c5087b64bc18ba38703ca7b989af11b18e579b046faf1a281467b62fb` |
| Boundary | `35a589915d31c09e26858b42309a927a928851f86cfd707c28416610f1e3ddfb` |
| Policy | `d0d25029e0977e49dcdd65f9079450325f2b1dbc64de635d9cdc527fb47a0c2c` |
| Device | `c2bed68b635e351b839f829745987ea1284ec08f18df2618ed6aa90b297e6cf2` |
| Selection environment | `004d81e229c150d86308c618e8153234b22609c1d9fa37063229775ddf89241f` |
| Preparation domain | `5f7efa3f25e0b95af4c4ddad66fa2093343a767104c0804a2e73a278d1b1de15` |

| Arm | Planned region | Physical recipe | Backend recipe | Backend build | Backend selection |
| --- | --- | --- | --- | --- | --- |
| Incumbent | `b6458780505be40ea71bf0a887f0f5036f73b0c7b7ce8a1e9f6ff7e67337bfc2` | `d3f1f8f0f322baf39888fe37fed5a7dd11f5a1f622f460efa6a9f9b7bb038653` | `6c907db04cdb63ac0b925b814a135dc0f4c872d5a25277a930958bd730e14efc` | `827a0d1d746b15dda52feffe2e6a5a120f47feacd85f3e90652afa6ca2d480c4` | `b0ba63c6a152539f148ae8b5f82b4ae9235d3fb2993b188063c85bb8e0f247a3` |
| Alternative | `88cc92a5f2f04e40cfc68e7205217475eade9a6f12dd7d5ae4fdd18834ae5a98` | `7d235f17da3a502831e6fb52e5e20fb57094c976f9af55cf36e67b916b34029b` | `6c907db04cdb63ac0b925b814a135dc0f4c872d5a25277a930958bd730e14efc` | `827a0d1d746b15dda52feffe2e6a5a120f47feacd85f3e90652afa6ca2d480c4` | `b0ba63c6a152539f148ae8b5f82b4ae9235d3fb2993b188063c85bb8e0f247a3` |

## Evidence and scope

The machine-readable evidence and raw supervisor receipts are retained outside
the repository at
`/tmp/llmp-m35-orchestration/u6-resume/ready-first-screen-r1/`. The evidence
records matching pre- and post-run hashes for all 112 source paths and the
binary. Those authenticated source and binary hashes, rather than the
inherited dirty-checkout Git HEAD label, identify the measured code. The
registration commit identifies the prospective protocol only. The root's
independent event, receipt, schema, median, bookend, and retirement audit is
`/tmp/llmp-m35-orchestration/u6-resume/ready-first-screen-r1/root-review.json`;
it found the run evidence clean within the registered scope. The raw
candidate remains `UNREVIEWED`; the audit did not adopt or install it. The
supervisor reported successful retirement and no remaining GPU process.

This result does not modify the historical twelve whole-program rows, install
the candidate, or establish performance for a model or other shape. Any
additional tuples require a separate prospective registration and source and
binary authentication on both Sparks before timing.

These observations are now included among the reviewed installed copies in the
[ready-body matrix](READY-MATRIX-RESULTS.md), after the separate matrix audit
and live-registry qualification. The original raw screen record is unchanged.
