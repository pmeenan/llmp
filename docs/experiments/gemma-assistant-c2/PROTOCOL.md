<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# C2 assistant identical-input and paid short diagnostic

Base a6dd81f. Task-entry TensorFold observation 2026-10-05 10:29:07 UTC:
609ca419abecebdc5a059498a613680bd3aa847f, version0.6.5; current primary
Gemma26 recipe is MLX-only, no same-format CUDA Q8 assistant comparator.
Same-format original image and b29c606e source pin remain the oracle.

Scope and ownership: NEW benchmarks/gemma_assistant_wave_fixture.cc and narrow
additive benchmark target; new experiment client/scripts/report only. Calibrated
C1 gemma_assistant_fixture.cc remains byte unchanged. No production graph,
engine, kernel, selector, verifier or serving changes. Ordinary primitive policy
only, per-segment local Q.ne3=1 and existing native selector retained. No forced
MMA. Physical batch-two reference and serial reference remain distinct arms.

Admission: two authenticated stage0-only manifests from original serialC2,
owner0P64 and owner1P65 (featureP-1, distinct pending anchors), local layer28
D256/KV8, global29 D512/KV2, capacities1280/4096, read256. Native target and
assistant artifact identities, previously qualified canonical vocabulary pairing inherited for these exact
artifacts (no fresh vocabulary parser scan), tensor contracts,
source memberships/positions, finite feature/cache and initialized zero padding
are authenticated. No stock C2 recurrent feature/anchor/head/projection is read
before native source/input/binary and endogenous own-repeat freeze. Previously
observed C1 evidence is historical and explicitly not a blinded C2 oracle.

Native correctness: separate mapped/funded readonly slot operands, no checkpoint
retagging. Build actual two-segment ordinary assistant graph, plus separate
single-owner graphs for serial controls. Same shared PagedNode/resources/stream,
held closure including weights/cache/workspace/staging; complete coverage before
uploads/launches. Fund two-owner complete heads/projections, masks, host inputs,
staging, plan/scratch/capture and exact byte witnesses before allocation. Cache
uploads use node-owned pinned spans retired before reuse. Unknown completion
retains whole heap owner bundle. P constant per owner; no target KV writes.
One-step and three-step first/repeat start from the same stage0 inputs. Native
recurrence consumes its own feature/argmax, owners independently. Per-owner
cache bytes before/after every correctness chain and complete finite output byte witnesses
are retained. Native serial and joined arithmetic are measured independently;
no assumption they must equal. Official retirement and exclusive native own
receipts precede stock exposure, with separate movement/allowance per policy.

Posthoc: compile bounded per-step incoming override before endogenous freeze.
After freeze, admit exact original serialC2 and physical batch2 incoming tensors
and outputs separately. Preserve unchanged native compiled graph/math/helper,
compare each native policy at identical full incoming features/anchors rather
than different endogenous prefixes. Full byte counts, max raw differences,
strict winners, positive-margin/outside-own-noise counts and full-row normalized
TV/chosen NLL are reported if byte identity fails. Never widen frozen noise or
force kernel choice in response. Independent reference repeat/frozen physical
state receipts already exist; bind comparisons to those exact identities.

Paid short screen after correctness: new original-image draft-step-only timing
companion links exact original /app math libraries (no rebuilt floating kernel).
Build/freeze its source before C2 raw recurrent output exposure. Setup both
original contexts before target prefill; prove rebuilt stage0 matches admitted
inputs. Target prefill/model setup/disk output/capture warmup excluded. Initial
3 fixed-shape warm calls in each arm allow build/capture/replay transition;
record actual native capture/replay and reference graph policy/ggml reuse only.
Use same constantP, C2 complete heads/projections and exact incoming sequence
for both timed arms, with CPU finite scans/lower-index argmax/full publication
and stage costs inside timer, no disk writes. Preallocate/fund all output copies.
Cache witnesses run immediately before/after timed blocks in BOTH engines, outside
the paid interval; no full cache hashing per timed wave. For initial bookend cycle the same three-step incoming sequence for 32 completed
C2 waves (64 complete head/projection rows) after warmup; no target
state evolves. Reference/native/reference on SAME SparkA, installed supervision
<=600, expected20–45s including setup. Retain full correctness-chain heads/projections, all 64 paid winner IDs and
the final paid full heads/projections outside the timer in each arm. Intermediate
paid full arrays are published and finite-checked but are not retained or claimed
as a full paid-vector comparison.
Name host-feature staging versus component device recurrence as a diagnostic
cost boundary; do not claim serving/component performance qualification or peak
memory from this manual companion. If negative, stop without depth/C12 ladders.
