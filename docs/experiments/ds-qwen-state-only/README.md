<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek and Qwen state-only prefill — 2026-10-08

The two repeated plain-model factors take **1.928% less DeepSeek prefill time**
and **2.282% less native Qwen prefill time**, with exact same-native publication
and initialized state. The three companion/GGUF pairs are n=1 diagnostics.
[Aggregate, complete payload/state identities and provenance](results.json).

| Recipe | OFF / ON prefill seconds | Scope / movement |
| --- | --- | --- |
| DeepSeek plain | 11.274866 / 11.057457 | n=2, -1.928%; OFF -0.026% |
| Qwen native plain | 2.267784 / 2.216044 | n=2, -2.282%; OFF -0.095% |
| DSpark injected | 11.281108 / 11.284783 | n=1 diagnostic |
| Qwen native MTP | 2.300102 / 2.266725 | n=1 diagnostic |
| Qwen GGUF plain | 4.117646 / 4.034004 | n=1 diagnostic |

Repeated OFF/ON ranges are 11.273422–11.276309 / 11.046933–11.067981s
for DeepSeek and 2.266710–2.268858 / 2.214823–2.217264s for native Qwen;
both pairs of ranges do not overlap. These short n=2 comparisons concern first
scalar prefill with the existing planning/backing policy. They do not establish
complete generation throughput or individual speed gains for the n=1 recipients.

Ordinary non-speculative DeepSeek and native/GGUF Qwen select this scalar
state-only policy. Private unset overrides resolve after actual speculation
selection; explicit true still enables the checked speculative opt-in, while
low-level graph/runner defaults stay headed. DSpark is neutral in its n=1 pair;
MTP is a single diagnostic with unmeasured extra Setup cost. Both speculative
head-only consumers remain unselected, so DS/Qwen composite T54 cells stay OPEN.

Scalar nonfinal prompt chunks can omit downstream work that has no consumer.
Plain DeepSeek preserves final raw/compressed cache writers, then omits unused
query/attention, HC post, FFN/combine and head work. Plain Qwen preserves final
KV/QSA pool or convolution/recurrent state writers, then omits unused query,
selection, attention/NormGate/output, FFN/combine and head work. DSpark final-layer
features and Qwen MTP exported streams are required consumers: those recipes
retain the complete final trunk and omit only the nonfinal head.

All-layer headed eligibility still determines DeepSeek output-A/HCA arithmetic,
including the removed final output. Pruning cannot broaden persistent-state
arithmetic. Unsupported token, verify, named-capture and joined target shapes
refuse state-only intent. Current and predicted cache keys include intent;
scoring requests remain headed. Qwen CatchUp, pending rows, hidden carry and
checkpoint settlement remain required even without published logits.

Both matched arms provision the same headed and state-only Setup envelope using
T96's exact-maximum shortcut. OFF forces current and predicted chunks headed;
ON forwards actual PromptSession intent. Removing roots is not assumed to reduce
greedy placement. Setup time and additional planning work are recorded separately
from paid prefill; this is not a zero-cost policy or a matched startup comparison.

Seven unique focused cases cover required state writers, all-layer HCA eligibility,
DSpark features, native/GGUF Qwen recurrence/QSA/MTP exports, unsupported wave
refusal and default policies. One supervised acquisition runs three diagnostic
OFF/ON pairs (DSpark, MTP, GGUF) and two repeated OFF/ON/ON/OFF factors (plain DS,
native plain Qwen). Each process compares 22 complete finite vocabulary heads,
choices and histories, every initialized target/drafter state range and digest,
actual C2-to-C1 owner departure, protected peer, spill/restore, continuation and
an actual nonfinal checkpoint/SettleFor path. Four scoring likelihood rows plus
a final scoring head stay headed. Observation/copy/hash/checkpoint/generation
work is outside paid prefill. No reference-arithmetic, PPL or HTTP claim follows.

Replay uses the checked-in prefill_prediction_probe, the same approved artifact,
SDK/private-library pins and standing little-endian I32 inputs as
[the preceding planning/backing batch](../ds-qwen-prefill-prediction/README.md#replay-after-raw-result-cleanup).
The new optional last argument is headed or state-only:

```text
llmp_prefill_prediction_probe ds|dspark|qn|qmtp|qg STORE TARGET DRAFTER|- TOKENIZER_DIR|- IDS0 IDS1 NEW_OUT on headed|state-only
```

Keep actual contexts 16384/2048, chunk sizes 4096/512, two slots and prefill
capacity one. DSpark uses the supported speculative wave form only for its
separate off-paid companion proof. Short Qwen prompts keep QSA off; focused
geometry/graph controls cross its actual boundary. Use GPU-exclusive installed
supervision, per-child 120s bounds and per-arm boot/kernel-error/retirement/GPU
absence checks. Require the exact 22-file full-head set, same-native bytes,
initialized state, choices, work and funding, and actual intent/node reduction.
Three pairs are n=1 diagnostics; only the two repeated factors yield means,
ranges and off-bookend movement. The prior raw bundle is not required to replay.

Compatible capture ahead remains T23 OPEN. Production wide joined prefill remains
T55 OPEN (DS one-row decode waves; Qwen target waves cap at four); the scalar
transfer does not close wider admission/funding. Broader T67 state preparation
and the original Qwen fast-FP8 row29 quality lead remain OPEN.

## Setup and source continuity

Both intent envelopes are measured explicitly; the selected graph counter is
planned work, not a GPU launch counter. Actual Setup elapsed ranges and total
scalar planning counts are below. These elapsed observations have no matched
state-only-unprovisioned control, so the additional latency is unmeasured.

| Recipe | Setup seconds range | Scalar plans / added state-only probes |
| --- | --- | --- |
| DeepSeek plain | 0.196317–0.202246 | 12 / +4 |
| DSpark injected | 0.258378–0.264822 | 19 / +6 |
| Qwen native plain | 0.253318–0.260895 | 12 / +5 |
| Qwen native MTP | 0.254866–0.259459 | 21 / +8 |
| Qwen GGUF plain | 0.285484–0.286766 | 12 / +5 |

Qwen additionally measures two joined target waves. Added probes derive directly
from the configured Setup loop, excluding verify/token/capture intents. All five
recipe activation, scratch, staging and plan-floor budgets match the preceding
qualified planning/backing policy exactly; the maximum is preserved, while CPU
graph construction/selection work increases. No zero-startup-cost claim is made.

The factor used manifest c812354f at base 05974c8 plus exact qualified T86 patch
ee924461. Final source carries T54 additively onto 7520f12, preserving all grouped
store source and the selected Gemma3 default. Only the two private runtime policy
paths differ from measured T54 code: unset resolves to non-speculative ON; the
probe still sets explicit true in both matched arms. Low-level flags remain false.
The final incremental runtime/probe compile qualifies that type/wiring delta;
unchanged seven focused cases and fourteen model processes are reused. An
initial prerequisite compile failed on a fixture shadowed local; its sole mechanical rename was reviewed and
the five-target incremental build plus all seven previously unrun cases passed.
No failed run enters timing. All fourteen children and the installed supervisor
returned 0, with stable boot, no new kernel messages and no remaining GPU compute.

## Owner stop and handoff

The owner directed completion of only grouped stores (7520f12) and this T54 batch,
then STOP. The inventory audit is complete; its transfer backlog is not. There
were 129 OPEN family cells before T54; target-only GGUF closes one, leaving
128 OPEN cells across 96 techniques. DS/DSpark and native/MTP Qwen remain composite
OPEN with selected plain children and checked, unselected speculative children.
This transfer run stops here and starts no further batch, phase or refactoring.
A separate agent may refactor under its own assignment. Full regression/shipment
tiers remain deferred; only this focused union and recorded actual
controls/factors are claimed. Keep milestone raw stores because M3.5 is
not closed. Standing prepared artifacts, pinned checkpoints and replay inputs stay.

Spark A warm tree is `/home/pmeenan/src/llmp-wt/gemma-state-phase-attribution`,
measured T86 source 5f80afcf, official receipt 0296e41b. Its old Gemma3 default is
false; the qualified explicit ON branch is selected by 7520f12. Spark B warm tree
is `/home/pmeenan/src/llmp-wt/m3clean-main`, selected source manifest a36b5d3e
and official receipt 874aaf7a, with the final runtime/probe compile recorded in
the aggregate. Both hosts report no GPU jobs, waiting jobs or compute processes
after positive retirement. Local raw records are retained outside Git; the
checked-in probe and standing input/metadata pins above suffice for fresh replay.
