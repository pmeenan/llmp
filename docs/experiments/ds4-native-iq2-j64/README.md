<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native IQ2 column tile: J128 versus J64

Keep the current J128 choice. On Spark A, forcing J64 reduced resident
gate/up throughput by 4.92% for the captured DeepSeek routed-expert fixture.
Every output byte was identical between the two tiles. Lower register and
shared-memory counts alone did not improve this workload.

This is the next causal control after the [standalone original/native
consumer comparison](../ds4-native-iq2-consumer/README.md). It measures a
native scheduling factor and provides no whole-model speed, quality,
memory-gate or concurrent-request qualification. It changes no default.

## Fixed inputs and paid work

The same authenticated community-model capture supplies layer 21 of the
second 4,096-token chunk of the 8K complete-plan run: K=4,096, M=2,048,
256 experts, six selections/token and 24,576 routed pairs. Its 249 active
experts produce 358 live J128 work items and 533 live J64 work items.
Both arms borrow the exact original D4 activation and native inverse map,
destination map and expert bounds proved in the preceding prepared-input
study. Both use the same two raw IQ2 weight allocations, whose complete
inverse from the original aligned layout passed its round-trip control.

Each arm pays two native worklist builders and two native products. The
activation producer and routing-map producer are outside this control;
uploads, allocations and owned guard initialization are reported separately
from the resident event clock. The same 512 MiB workspace and full output
buffers are funded throughout. Resident gate/up weights total
1,107,296,256 bytes; the selected-expert footprint is 1,077,018,624 bytes,
larger than the device's 25,165,824-byte L2.

Only the fixed diagnostic entry selects J64. The ordinary native selector,
numeric definitions, weight loader, packed arguments and compiler flags
remain unchanged. The guarded host bridge lives in the existing defining
O3 IQ2 instance; it creates no second numerical translation unit.

## Completed paired control

Spark A (`spark-c4e2`), 2026-10-01, 14:46:31–14:47:21 EDT. One resident
child ran J128/J64/J64/J128, with a warmup and nine positive CUDA-event and
wall samples per phase. Both products were read in full after each phase.

| Phase | Median CUDA event time, ms |
| --- | ---: |
| J128 before | 20.330463 |
| J64 first | 21.034336 |
| J64 second | 21.051264 |
| J128 after | 19.683489 |

Mean of the two phase medians: J128 **20.006976 ms**, J64 **21.042800 ms**.
The resident rate ratio is **0.95077538**. Same-arm bookend max/min ratios
were 1.032869 and 1.000805. These are operator measurements; the previous
original-engine timing is a separate run, so this test supplies no new
same-process J64/original ratio.

The runtime queried the actual kernels in their authenticated O3 defining
instance before and after warmup:

| Attribute / dispatch | J128 | J64 |
| --- | ---: | ---: |
| Registers/thread | 254 | 244 |
| Static shared bytes | 0 | 0 |
| Local bytes | 0 | 0 |
| Launched dynamic shared bytes | 57,856 | 48,384 |
| Threads/block | 256 | 256 |
| Product grid | 16×448×1 | 16×640×1 |
| Worklist scratch bytes | 3,584 | 5,120 |

Both kernels report binary/PTX version 121. The device opt-in shared-memory
limit is 101,376 bytes. Occupancy was not measured. The selected SDK has no
`cuobjdump`, so unavailable offline resource-dump fields remain explicitly
unavailable; the runtime attributes above are separate direct observations.

## Numerical and lifetime controls

All 50,331,648 F32 values per product compare exactly across tiles: zero
different values, maximum absolute error 0 and NMSE 0. Both own repeats and
same-arm bookends are exact. J128's full gate/up outputs also match the
preceding authenticated native control. Full D4 payload, maps, bounds and
owned zero guards are unchanged; all complete outputs are finite.

The 24 fixed FP64 dot witnesses retain the original captured result and
both native results against the same decoded D4 operands. They are
descriptive and have no registered acceptance bound. No whole-model quality
exception follows from these operator controls.

One GPU child completed rc0, was reaped, and passed independent strong
retirement; the final gate reported 116.709 GiB available with GPU,
container and native-model probes clear. No model inference or broad test
suite ran for this experimental factor.

## Source and evidence

The original 320-entry prepared GGML tree remains byte-identical to its
lock. A full external copy has only the reviewed instance tail and two
bridge headers added/changed. This uses the supported explicit source
override with lock enforcement OFF and an **unofficial diagnostic receipt**;
it is not an unchanged official locked build. The literal ds4 numerical
proof still authenticates 115 complete functions and ten original headers.
The native defining instance uses the original O3/fast-math/extended-lambda
flags and sm_121a code. No numerical body or compiler flag was changed.

The initial direct edit of prepared source was correctly refused by the
lock check. The next configure correctly refused the new private header
missing from `module_files`; adding that inventory entry resolved it.
Both failed attempts remain outside Git with their receipts and sources.

| Evidence | SHA-256 |
| --- | --- |
| Completed target receipt | e62135df8c540fefa08942781114f663cfea1113aeaccb037a8a7b061533286d |
| Equal 1,248-file source maps | b1926c4dd9aa34032c6191870888e801be1a4b18cd3db8a641db7b38d3f41c49 |
| Binary | 7092d442b7cfa919060cca8258cd5f5b8af2ef48cae1882f310f321a115ce011 |
| O3 defining-instance source | ff21e28a6f6e417081b37fa575b96d31da965fa562d3ce6d75c5130ce3784014 |
| O3 defining-instance object | a689db0966a1bccdeb2015015b324456aee804a28c0a47cb16d889ae776d0a21 |
| Explicit override proof | e86e3b568e807a8a7247a38eed457cc5de830aaada3df914425102b01d7fcf65 |
| Operator receipt | bad5b712a5cadcb48eee36ab459ab3a8ca46da28f8137995de792f475c4331e0 |
| Native operator receipt | 97c7089fe838e87d77c4f29e14fd77351df0afbe78d6cf0ff69eff97fae6fde4 |
| Full-output / FP64 analysis | e96997bf6b9f463df6068b7e633e96f5d75b0e07cf06f8b717fec7f212065df6 |
| Completed preservation receipt | ebd64e795b0f4ca36587d58ce1597279b3038ce8df6131b8193533d9c06bea1d |
| Actual Spark SDK receipt | f38891fcc394ddf956aaa196c6895d7309148b8b5159b0c3a0459721946992bf |
| External study source | 005f25ade5eff91d8b6a826e5f3fac2a7e3326a2c952f95d221baaf4619c2fdd |
| Shared controller source | 8cd7b50fd49de445626c12670454d36b9231d77e9d166a591f685e9f231e9966 |
| Operator controller source | 2cb82d6b61112f44ee322d49d172c9c69280dbdac2dd3c9492b9aa79db934ea8 |
| Full-output analyzer source | ac9a301fa3f8885f06f2e8cac000cf60f5c2291507773ef87781f772ec775ec3 |

Raw evidence, the exact override/controller sources and the full twelve-file
operator output inventory remain under
`spark:~/scratch/m3-ds4-native-iq2-j64-r1/`; aggregate copies of build,
operator and analysis metadata are retained on the workstation under
`~/scratch/m3-ds4-native-iq2-j64-*`. These are reproducibility artifacts,
not shipping kernel dependencies.

`preservation-r1/` authenticates 1,696 copied source/build/controller/library
files and 37 complete raw files retained in place (twelve operator outputs,
23 original capture files and two derived raw weights). The copied sources
include the entire explicit GGML override. Its terminal strong gate reports
117.327 GiB available and all process probes clear; the earlier completed
consumer archive and both failed J64 build attempts remain intact.

Keep the full-byte operand controls and the actual defining-TU attribute
bridge as useful diagnostic pieces. The negative tile result applies to
this exact shape and expert distribution; it does not justify changing
other IQ2 consumers or predicting their performance from resource counts.
