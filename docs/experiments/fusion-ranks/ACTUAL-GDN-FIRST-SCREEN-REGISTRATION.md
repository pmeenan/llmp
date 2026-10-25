<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Actual full GDN first-screen registration

Prospective registration, 2026-10-25. Compare the two actual ordinary complete
programs from shared `BuildStep`: the default NormGate fusion and its own
normally planned decomposition, with GdnStep retained in both. This F32
synthetic token model includes embedding, GDN and output head; it is a full
program comparison for this fixed shape, not a checkpoint or end-to-end model
speed claim. The isolated Ready producer has a different physical key and its
measurements are not transferred here. No full-GDN performance comparison has
run.

## Frozen implementation and correctness

| Fact | Value |
| --- | --- |
| Development Git base | `f0ba0a0fc0d48cdb73cb484c1b3c2900f6c8064e` |
| Frozen source | `u6-captured-events-r47-b`, 118 paths |
| Source manifest SHA-256 | `bb98e764a488502657a06f339cd80de26d572b00cacef5b3d31ffdd6b96d5517` |
| Native binary SHA-256 | `5897e27e80e3c07fbbe4161ab353541aa65da562b20636686c91597f2ae29819` |
| Backend build | `5bfcfa9ed5b5d01720b9aedf656d9051b93e066930d1ca694827094ff0645ed5` |
| Original source graph | `ddc5b25148264f40d40fe7fae743fca38fad7b937ceee7ad072b6f1d384bf0b2` |
| Selection environment | `b6636d1620fdb552a465526349b20cb4c4c4910b7c316c2bee3331e262fbe357` |
| Host / GPU / SDK | `spark-b` / NVIDIA GB10 / `aarch64-2c4eb5b2f68f3fa5` |
| Protocol SHA-256 | `1fe490da95280eff980acda68a94aeab3d5c32ccec5aa6758c5457283dd694bd` |

Shape: stream width 64, head dimension 128, 48 value heads, 16 key heads,
convolution width 4, one decode row, F32 weights/state. Deterministic weights
include negative folded A. Original-IR CPU references cover two changed token
and nonzero state frames. Both ordinary and captured actual programs validate
all eight logits, all 30,720 convolution values and all 786,432 recurrence
values. Each state tensor updates nontrivially. Worst normalized max gap is
`1.0971276102037256e-7`, below the prospective `1e-4` convention used by the
existing component reference checks. Validation and measured-capture ownership
reviews are clean; native functional checks complete normally after teardown.

Each arm has 205,824 required activation bytes, within a 33,554,432-byte arena;
the common scratch pool is 33,554,432 bytes and complete state is 3,268,608
bytes. Actual captures contain 21/23 nodes, including the two timing-event
nodes and exactly one complete program. Functional tests verify positive
completed durations without collecting performance samples.

## Screen and decision

After this registration is committed, run
`build/spark-native/benchmarks/llmp_fusion_gdn --screen` once in an installed
supervisor exclusive job, bounded to 600 seconds. Verify frozen source and
binary before and after, require normal completion after teardown, and retain
a separate host-free check. Repeat ordinary and captured two-frame validation
before the comparison.

Before every replay, NaN-poison the complete activation arena, restore the
identical complete initial state image and stage each arm's actual input slots
on the owning stream. Finish staging before replay. Time one original complete
program using its internal external-event nodes; finish replay and read its
completed duration. Check complete logits and both complete state arrays after
every arm, outside the timed body. Use frame zero for all comparison arms.

Use four fixed AB warmup pairs, an alternative bookend, four alternating AB/BA
warmup pairs, nine alternating AB/BA sample pairs beginning AB, and an
alternative bookend. Preserve all 36 observations in order. Refuse if the larger
bookend exceeds 1.1 times the smaller, with no automatic retry or threshold
change. Compute each median from the nine sample values and round positive medians
to the nearest integer nanosecond (half up). Use `ClassifyRecipeLatency`: a
difference strictly greater than `floor(alternative_ns / 10)` selects the
faster arm; differences within that band are neutral. Retain both raw medians
and memory costs regardless of classification.

The result remains diagnostic pending independent source, event-order,
numerical, drift and retirement review. It creates no Ready comparison key,
installs no rank, transfers no isolated producer row and authorizes no model
adoption or U6 completion claim. Timing-event overhead remains part of the
instrumented interval.

The exact protocol text is:

```text
llmp.u6.actual-gdn.full-program.v1;shape=w64,d128,hv48,hk16,c4,decode1,f32;capture=single-ordinary-full-program;duration=internal-external-event-nodes;restage=full-arena-nan-poison,full-identical-state,own-input-slots-before-every-replay;reference=full-original-ir-logits-and-both-f32-state,normalized-max-gap-1e-4;timed-frame=0;fixed-ab-warmups=4;alternative-bookend-before;alternating-ab-ba-warmups=4;alternating-ab-ba-sample-pairs=9;alternative-bookend-after;drift=10pct;retry=none;adoption=none
```
