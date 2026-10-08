<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Matched standalone IQ2 gate/up consumers

The current native compact IQ2 gate/up consumer runs at 0.880357× the
original standalone consumer's resident rate on the same captured inputs.
Its mean bookend median is 20.061968 ms versus 17.661696 ms. Activation
quantization, activation layout and routing maps are held fixed; the
remaining comparison combines weight loading, tile/schedule, paired-launch
structure, arithmetic and compiler choices. Keep original Direct in the
temporary complete benchmark while isolating those factors. No production
dispatch, full-model quality or concurrent-request claim follows.

## Matched operands and paid work

Spark A uses the authenticated community-model chunk 1, layer 21: T4096,
K4096, M2048, E256 and six selected experts, with 24,576 assignments. Both
consumers borrow the same complete original D4 activation, destination IDs
and expert bounds. The preceding [preparation capture](../ds4-native-iq2-prepared/README.md)
established full native/original D4 equality. Original forward and native
inverse source maps remain complete audit inputs; neither consumer reads
them. Owned activation/destination guards are initialized and all retained
ranges remain disjoint through completion.

CPU derivation proves complete inverse round trips between raw IQ2 records
and the original scale/code planes for both banks. One resident process
uses the same two 553,648,128-byte device weight allocations throughout,
uploading raw records for native or original SoA bytes for original only
after completion. There is no second simultaneous resident weight layout.
Both consumers preserve the same original half scales and IQ2 codes.

| Actual dispatch | Native | Original |
| --- | ---: | ---: |
| Tile / threads | I128 × J128 / 256 | M128 × N64 / 256 |
| Worklist builders / products | 2 / 2 | 1 / 1 paired launch |
| Product grid | (16,448,1), twice | (16,640,2) |
| Work capacity / bytes | 448 / 3,584 | 640 / 2,564 |
| Dynamic shared bytes | 57,856 | 0 |
| Static shared bytes | unavailable | 37,744 |
| Registers / local bytes | unavailable | 128 / 128 |

The original live worklist has 533 entries. Capacity is not the live count.
Native selection is authenticated from the existing host selector and
actual compiled instance, rather than inferred from a maximum guard size.
Both compact consumers bypass Stream-K. Original attributes come from its
defining numerical TU. The SDK lacks the requested object resource-dump
tool, so native compiled resources remain explicitly unavailable; these
values establish no occupancy result.

The native host wrapper uses its existing O2 flags; the actual IQ2
numerical instance uses O3, fast math and extended lambda. The literal
original TU uses O3, fast math and line info. Actual compile entries,
sm_121a machine code, object bytes and static closure are retained. No
numerical kernel or compiler flag changes in this comparison.

## Resident timing and output controls

One process runs N/O/O/N. Each block pays one warmup and nine positive GPU
event/wall samples, with full output checks after completion outside the
timer. Each arm pays its own worklist builder(s) and product(s); activation
quantization, routing, activation/down/sum, uploads, guards and readbacks
are excluded from the resident interval and reported separately.

| Block | Median ms |
| --- | ---: |
| Native before | 20.229696 |
| Original first | 17.884800 |
| Original second | 17.438593 |
| Native after | 19.894239 |

Native bookend max/min is 1.016862; original is 1.025587. Every own repeat
and both same-arm bookends match their authenticated complete gate/up
calibrations byte for byte. All 50,331,648 F32 entries per product are finite.
The complete original live worklist and all borrowed inputs remain exact.

| Native versus original | Differing F32 entries | Maximum absolute error | NMSE |
| --- | ---: | ---: | ---: |
| Gate | 35,453,307 | 1.192093e−6 | 1.088888e−14 |
| Up | 37,006,158 | 1.430511e−6 | 1.361259e−14 |

These are the same full-output differences found in the prepared-factor
control. Twenty-four fixed FP64 physical dots use the identical captured
D4 and completely authenticated weight bytes. They are descriptive
witnesses with no registered acceptance bound, not a model-quality pass.

The child allocates 2,227,492,872 CUDA bytes, including a symmetric
536,870,912-byte workspace. Resident gate/up banks total 1,107,296,256
bytes; the 249 selected experts' footprint is 1,077,018,624 bytes. Initial
uploads cost 180,669,444 bytes/0.003250 s; four bank uploads cost
4,429,185,024 bytes outside timing. Total uploaded bytes are 4,609,854,468
and bank-upload wall time is 0.078154 s. These inventories do not qualify a
comparative memory gate.

The following [J128/J64 control](../ds4-native-iq2-j64/README.md) holds raw
weights/D4/maps/arithmetic fixed and queries the actual O3 defining kernels.
J64 is 4.92% slower with every output byte equal, so keep J128. The
original/native comparison still combines loading, paired launch structure
and remaining arithmetic/compiler choices; these timings do not identify
one as the cause. Preparation/capture controls can transfer to other compact
products after separate type/shape qualification; this result qualifies
only the captured IQ2 geometry.

## Provenance

Spark A, 2026-10-01, SDK `aarch64-e0a0c85c42806fb1`, actual SDK
cuBLAS/cuBLASLt. Target-only build and matched controls follow the owner's
experimental-check override; no routine unit/full-suite/style rerun or
full-model execution is claimed. The original 115-function/ten-header
correspondence proof passes. The source baseline is `fb3d5ec` plus the
seven bounded consumer/dispatch-description paths, separate from later
shipping-reference cleanup.

| Identity | SHA-256 |
| --- | --- |
| Original capture metadata | `6b26061eac3b5970cddaafad3f6fd2cb1fc3091e06be51be9df9624bb14c395e` |
| CPU raw-weight preparation | `7ecd5d3b279a39f4972f3df3bfd082ebbdbd92a25b358f1e52c92d0f74db2088` |
| Target build | `767f4025b56d83fba84f2f1a470139a8a3650e51048e12d57b280d94dbf10d0d` |
| Unchanged 1,247-file source map | `c1930e0ae3d6528aaa29ea1184c39190b94afc3b266feb51f896ac8ed548bcde` |
| Replay binary | `dcb4d648364d815d08f6af9ca532a006fcd0dcb6128f9a25b4c6533a71cb40f9` |
| Actual native O3 IQ2 instance object | `b184cc137ef567c68cd782045844164baf90f1df63de71a2973d39aaf1d6840f` |
| External study source | `b629be43c1af9a18229ab6ec07353cc42757dc1461fb954740427350606ab7ac` |
| Operator controller source | `4398ca39075e4277c203106ffee9cf33abe28f6a049c6c820c535a280b3ad0f1` |
| Analyzer source | `161dcbbacb669f11f221732677c63b917de07681c255df06c7cbad4467db938d` |
| Four-block operator receipt | `1d39d369e1b8e379fb844c1b3ca19277c653cbd85dec6cab2915ebc515ff6a8f` |
| Full-output/FP64 analysis | `44322d0a9609ed1ad26717154dce73739522c8aa0802c264a8268e3de3027381` |
| Completed preservation | `836563262f1a759346df30b1a5ba9d352dcaed7c77bcfb0868e9b7ceca3c20d1` |

External records remain at
`spark:~/scratch/m3-ds4-native-iq2-consumer-r1/`. Preservation retains
1,366 source/build/controller/evidence files and authenticates 39 large
files in place: fourteen operator outputs/audit buffers, 23 original
captures and two derived raw banks. Its `controllers/`, `build-control/`
and source/closure inventories bind the complete replay and original
parent. The child is reaped, both operator retirement checks pass, and
final preservation reports 117.334 GiB available with a clear node.
