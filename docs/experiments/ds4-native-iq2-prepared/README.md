<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Captured native IQ2 preparation and borrowed consumers

The complete native and original gathered D4 activation payloads are byte
equal in this fixture. Replaying either payload through the same native IQ2
consumer also produces byte-equal full gate/up outputs. Activation
quantization and activation layout therefore do not explain the small
native-versus-original product differences observed in the
[preceding restoration experiment](../ds4-native-iq2/README.md).

Removing native preparation reduces the resident pair time by about 4–6%
in paired bookends, a descriptive difference of separate medians. This
does not measure the standalone original consumer's speed. Keep original
Direct pending an actual matched consumer comparison; no model-quality,
production-default or concurrent-request result follows.

## Operand and lifetime controls

Spark A replays the same authenticated community-model chunk 1, layer 21:
4,096 F32 tokens of width 4,096, six selected experts per token, 256 experts
and 2,048 output columns. The 24,576 selections and original maps form a
complete checked bijection. CPU derivation preserves every raw IQ2 scale
and code byte and verifies full round trips for both 553,648,128-byte
gate/up banks. Only raw banks are resident in this child.

An optional capture copies the native operation's actual activation and map
buffers after its unchanged preparation, before its two unchanged compact
MMQ consumers. Two complete captures must repeat and reproduce the
ordinary operation's full gate/up hashes. A separate borrowed operation
runs those same consumers without allocating or executing quantization or
mapping. Captures, copies, hashing, full comparisons and file writes remain
outside ordinary and borrowed operator timers.

The native source map is token-slot to sorted-pair; the original source map
is sorted-pair to token. Both full representations are validated against
the original destination bijection. Source IDs are retained audit inputs
and are unused by the borrowed consumers. Destination IDs and expert
bounds match the original byte for byte. No GPU permutation or numerical
kernel change is introduced.

All 786,432 D4 blocks, totaling 113,246,208 payload bytes, match exactly:
zero differing scale bytes, code bytes or scale values. Both payload hashes
are `4e6a6328046a25df1ebd10ba6ddf9098402c8c2a5d47f010baeba653d4c75ea1`.
The captured private guard bytes happen to repeat and be zero; this is an
observation. Replay explicitly zeroes only its owned activation and
destination guards, 37,888 bytes, and verifies unchanged complete operands
after consumption. Checked ranges keep captures, replay inputs, outputs,
live inputs and workspace disjoint through fenced retirement.

## Resident timings

One process pays one upload/setup, ordinary warmup and nine positive event
and wall samples per phase. Borrowed-native calibration, borrowed-original
and final bookends each have their own warmup. No profile marks or model
weights outside this captured fixture are loaded.

| Arm | First median ms | Final median ms |
| --- | ---: | ---: |
| Ordinary native pair, including preparation | 21.028481 | 20.929344 |
| Native consumer borrowing native D4/maps | 20.195616 | 19.722944 |
| Native consumer borrowing original D4/maps | 20.401888 | — |

Ordinary bookend max/min is 1.004737; borrowed-native is 1.023966.
The mean ordinary median is 20.978912 ms and the mean borrowed-native
median is 19.959280 ms: a descriptive 1.019632 ms difference, 4.86% of
ordinary time. Corresponding first/final reductions are 3.96% and 5.76%.
There are no internal preparation timestamps, and the historic complete
Direct/Materialized/suffix timings are not standalone consumer controls.

The child allocates 2,340,953,096 CUDA bytes, including a symmetric
536,870,912-byte workspace and retained capture/replay buffers. It uploads
1,401,391,112 bytes in 0.023852 s and pays 0.004203 s for owned guard
adaptation outside consumer timers. The selected experts' gate/up footprint
is 1,077,018,624 bytes; resident raw banks total 1,107,296,256 bytes. These
are allocation and upload inventories, not a comparative memory gate.

## Numerical result and next factor

All thirty retained output/capture/replay files pass full extents, hashes,
finite-output checks and independent repeats. Ordinary, both captures and
both borrowed-native phases reproduce the preceding native full gate/up
hashes exactly. Borrowed-original also matches them exactly, isolating the
remaining difference to consumer arithmetic and its weight-layout/launch
implementation rather than a different D4 producer.

| Native consumer versus captured original Materialized | F32 entries | Maximum absolute error | NMSE |
| --- | ---: | ---: | ---: |
| Gate | 50,331,648 | 1.192093e−6 | 1.088888e−14 |
| Up | 50,331,648 | 1.430511e−6 | 1.361259e−14 |

Twenty-four fixed FP64 physical dots use the actual captured D4 separately
for each consumer. The activation-dot difference is zero. Maximum absolute
consumer error is 3.535916e−7 for native and 5.329133e−7 for original
Materialized in these samples. These are descriptive witnesses with no
registered acceptance bound; they do not qualify full-model answers or
perplexity. The experiment runs no downstream activation/down/sum chain.

The [same-D4 standalone comparison](../ds4-native-iq2-consumer/README.md)
now measures the native gate/up consumer at 0.880357× original resident
rate, paying each worklist and product. A subsequent [J128/J64 control](../ds4-native-iq2-j64/README.md)
keeps every output byte equal but is slower at J64. Weight loading,
paired-launch structure and remaining arithmetic/compiler choices remain
distinct possible causes.
The existing capture/borrow seam can isolate native preparation for other
compact GGML expert products after separate shape/type qualification; this
fixture establishes only IQ2_XXS at this geometry.

## Provenance

Measured on Spark A, 2026-10-01, with SDK
`aarch64-e0a0c85c42806fb1`, actual SDK cuBLAS/cuBLASLt and unchanged compiled
native numerical paths. The target-only build retains source maps, static
link closure, actual compile commands and four native prepared-header
identities. The original 115-function/ten-header correspondence proof
passes. No routine unit/full-suite/style rerun is claimed under the owner's
experimental comparison override.

| Identity | SHA-256 |
| --- | --- |
| Original capture metadata | `6b26061eac3b5970cddaafad3f6fd2cb1fc3091e06be51be9df9624bb14c395e` |
| Full-roundtrip CPU preparation | `7ecd5d3b279a39f4972f3df3bfd082ebbdbd92a25b358f1e52c92d0f74db2088` |
| Target-only build | `043c3bba8127942d4a759ba0a2f929b2f530f004131799ab67adb87075c9a291` |
| Unchanged 1,245-file source map | `5108e27215109e20e4496e738ef381e1f02e09cbbcb11cfe615509f81fafcd81` |
| Stateless replay binary | `cedfbde267d1806b73bec2e9ecb5be1ac582a7ee5e934da546346a8a54cb00bf` |
| External replay host source | `45b29ca5897c0d7737b7dba23db073e99eb5c91aa59bc3ec1b680557f4978c1f` |
| External replay support header | `0503e49937b6fce146d02affff3f651bbdf14a238aaf49c6a3e8592b5c30a35c` |
| Operator controller source | `03f35bb3aeeccb7998a93186188c9576d45b9070985c1c472215cc9492b68366` |
| Data-only analyzer source | `f36dc28a16333f6eda4c267e3e66a0efd8592e5bdd78feab4bf382fda17abd7c` |
| Five-phase operator controller receipt | `aae7bd5fa6763832631076de87303c7cdf2654ffdb5267e95ff11413e9aa30f9` |
| Complete operand/output/FP64 analysis | `f2b1b26e7c3124b9a7debc60a2ca5a6c7a53a4a78f17d0ccbb42efd635741e6e` |
| Completed source/build preservation | `253365ba54971cb95e927e8bd5a442e409a8dcb311003dc18fb789d24ea121fd` |

Raw records remain outside Git at
`spark:~/scratch/m3-ds4-native-iq2-prepared-r1/`; preservation retains 1,355
source/build/controller files and authenticates all 55 original/derived/raw
files in place. `preservation-r2/controllers/` contains the authenticated
replay host, support header, controllers and protocol; `build-control/`
retains their actual compile/link and source proof. The successful operator
is reaped with both child and
terminal strong retirement checks; final preservation reports 117.249 GiB
available and a clear node.

An initial external compile failure is preserved. The first operator
attempt refused the mistaken forward-versus-inverse source-map assumption
after fourteen raw files, before borrowed consumption. Its exact source,
binary and those partial captures are preserved separately, with a reaped
child and terminal strong retirement. It establishes neither a numerical
failure nor a timing result. The correction changes host audit validation
and comments, preserving numerical CUDA bodies and consumer arguments.
