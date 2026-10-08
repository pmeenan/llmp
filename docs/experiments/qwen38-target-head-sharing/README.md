<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen four-request BF16 full-target-head sharing

The private four-request generation control improved pooled decode throughput
from 48.2545 to 52.5332 tokens/s, an 8.8670% gain. Four complete fixed-history
target-head vectors were byte-identical, and all 1024 generated IDs, natural
acceptance traces, initialized state pages and pending cursors matched across
head-off/on/on/off. This is a positive mechanism result from two observations
per setting in one matrix. It does not enable a production default or qualify
HTTP throughput, Mia/TensorFold parity or broader model quality.

The run used Spark B, four unchanged 8192-ID spec-C4 fixtures, context 33792,
chunk 4096, fixed draft depth 3 and the selected 47172-ID drafter 8600a998….
Each request produced exactly 256 literal-stop IDs, including its prefill
anchor; the loop timed the remaining 255 per request, or 1020 aggregate.
Every cold loop paid planning/capture, normal output copies, packing,
commit/rollback and final settlement. Prefill, checkpoint/page controls and
full-vector diagnostic copies are outside this decode clock and included in
the full native wall time of 156.1192 seconds. The supervised job completed
2026-10-01 18:08:58–18:11:53 EDT in 175.312 seconds.

| Chronological arm | Head sharing | Paid loop seconds | Aggregate tokens/s | Draft seconds | Verify seconds | Settlement seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OFF before | Off | 21.251592 | 47.9964 | 3.215220 | 17.578708 | 0.452005 |
| ON first | On | 19.370217 | 52.6582 | 2.966070 | 15.936894 | 0.450068 |
| ON second | On | 19.462354 | 52.4089 | 2.967550 | 16.028403 | 0.449561 |
| OFF after | Off | 21.024269 | 48.5154 | 2.991173 | 17.563622 | 0.452306 |

OFF and ON mean times were 21.137930 and 19.416285 seconds. OFF bookend
max/min was 1.010812; ON repeat max/min was 1.004757. Both ON observations
were faster than both OFF observations, with pairwise throughput gains from
8.0253% to 9.7127%. The frozen factor protocol has no automatic minimum-gain
or bookend rejection rule; these ratios are descriptive evidence of limited
movement within each setting. The first OFF draft phase was slower
than the other draft phases and is retained. Verify mean fell from 17.571165
to 15.982648 seconds; this is a complete host Verify-phase observation,
not an isolated head timer or hardware bandwidth measurement. Two repeats
do not establish a broader confidence interval or serving-workload gain.

Only complete four-by-four target waves share the head. The ordinary selector
actually chose `ggml.mul_mat.mmf` for both the original four-column products
and the combined sixteen-column product. Each group retains four independent
normalizations and states, pays three ascending-slot F32 concatenations, runs
one ordinary BF16-weight/F32-input/F32-output product and exposes four bounded
views. The immutable full head is 1,271,398,400 bytes. Each shared group adds
368,640 packed bytes; the sixteen-column F32 result occupies 15,892,480 bytes.
Ragged or one-active waves keep the original head path. MXFP8 products remain
paired at eight columns; the rejected exact16 MXFP8 kernel is absent.

Each ON loop recorded 94 shared head groups and exactly 34,652,160 added packed
bytes. OFF/ON packed totals were 9,392,856,368/9,427,508,528 bytes. All four
loops retained 75,264 MXFP8 pairs, 19,992 routed launch pairs, 101 waves,
84 eager/11 captured/107 replayed plans and zero dropped/refused plans or
coverage violations. The mapped activation envelope was 899,678,208 bytes
with a 4,194,304-byte pool; all 15 active masks and every actual runtime
activation/scratch bound were checked.

The four fixed-history full controls each contain 993,280 target F32 values:
four rows over all 248320 original IDs. OFF/ON changed words, changed values,
maximum absolute error, RMS and NMSE were all zero; all sixteen greedy margins
and runner-ups agreed. These complete vectors cover the untimed shape wave,
not every timed target vector. The natural loops separately require exact
IDs, acceptance/work trajectories, initialized pages/range geometry and pending
cursors. The controller reconstructs every timed trace and raw ID array.
The 72 retained raw arrays support 16 output controls, 32 full-state controls,
8 canonical captures and 24 complete ordered-page/cursor comparisons. The
four-slot baseline was 2,720,432,128 bytes; diagnostic device checkpoint copies
were 27,204,321,280 bytes and state reads 21,763,457,024 bytes. Page capture and
comparison took 16.878512 and 4.459937 seconds outside decode timing. No full
host state replica is archived.

The first attempt is preserved as a setup refusal: the private validator
incorrectly required a tensor name, but the original builder records graph
names without setting `tensor.name`. It stopped before Start/Load. The retry
removed only that redundant name guard, retained actual graph-logits identity
and all type/shape/stride/leaf/parameter checks, and first obtained a no-load
plan proving all five actual MMF descriptors. Neither baseline inputs nor
ordinary selector/kernel bodies or numerical flags were changed. Finite
target-logit drift would have been retained descriptively; semantic IDs,
draft inputs, trajectories and state controls remained strict.

The source is an independent f4bdf654… derivative with six benchmark host
changes and exactly 1256 source files. The measured R3 patch is Git
`1af38b1ec58ba812bdd4fb476c717e305b7c53b7`; controller Git is
`175a6309c93297bd236458e4d060837446441f99`, SHA-256
`5270523d47a1c0807e69aca4f85c8a472bab81afad1d46284620cadd2c440191`.
Actual source map is `ca6183aa16ccd5ae4db8d70c0207a5e6ea18d2b34e2df7f2bb0ddc79810419a1`
and binary `277c1b33895023a22cdd5c68f1a2aebd4f8e5a8485f2fefc7cb24aaf5b4ed6ab`.
Official six-component locks, actual SDK 4432b0abad… and resolved SDK cuBLAS
payloads, DB/cache/native receipts and unchanged selector/MMF/MMVF sources,
objects and normalized flags were authenticated before and after the control.
Target-only diagnostic builds ran no unit suite or routine style cycle under
the owner's experimental override.

Raw and receipts remain outside Git under
`/home/pmeenan/scratch/m3-qwen-target-head-r3/` on Spark B. Actual build receipt
SHA-256 is `dfdc6066baa31e3e62997cfc0beba70f80826f8ee0b69ac5e6e27b12dcbba92a`;
no-load plan is `ffc7309450d055b1f10712b26ac6e0025dab30204d771f5b0a57936a916d248a`
with native `ee37e5d0b55e6f11e3dd277236660639501f546b6301b717b1e24c1a050f6df8`.
Completed control is `9cf0de4a7c0d00965bd36bc0fab28f2ebd429ef21acb67ed3b5066354fc07675`
with native `9a1eb3414895ef0d16b00bbbf33fd30b7a6ef0c48868732f716c2739d8035602`.
Supervisor 3206989 and all five children completed with rc0, inherited process
group and reaping. Terminal strong retirement reported 116.950 GiB clear.
The earlier failed attempt, checked pairwise C2/C4 archives and negative
exact16 archive remain separate and unchanged. The completed data-only archive
is `preserved-r1/` under that same scratch root, receipt SHA-256
`444b44b4ff74d4005f3a27748cba783b0c1a34a9985992d000f71f96511b107d`
and inventory `1ffe6efb4ac5b6baf36702176f56984c4d267fe2b5a7118132480c6c796e3b42`.
Its 1648 copied files include all 1256 source files, 256 target objects/archives,
72 raw arrays, seven ordinary head numerical source/header members, the binary,
relative-RPATH SDK cuBLAS payloads, actual tools, DB/cache/native receipts and
build/plan/control evidence. Live and copied identities and both earlier
archives were reauthenticated at completion. This is not a full SDK installation
or state-page replica. Supervisor 3212961 completed in 24.655 seconds; all three
children were reaped with rc0 and final strong retirement was 116.983 GiB clear.

Retain this factor as a candidate for the independent Slot-wave path while
keeping paired-eight MXFP8. Its actual serving integration, independent-state
lifetime proof and matched request performance remain future work. The
private fixed-history and natural controls do not change production defaults.
