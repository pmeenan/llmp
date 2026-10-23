<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 ordinary multirow repair: numerical result

**PASS: all 1,070 teacher scores match the retained incumbent exactly.** The
single repaired R run and the single registered pair pass the unchanged five
criteria. The original [A/D failure](../gemma31-multirow/RESULTS.md) remains
failed; neither its input nor its bounds were replaced. Matched C2 speed and
assistant/state transfer remain separate obligations.

## Registered contrast and actual execution

Registration commit `440d4b3d3e7817d7414df8bfab371fb4482a21b4` fixed the
[protocol](PROTOCOL.md), original A, source/compiler ledger and identity map
before R collection. Spark B (`spark-56f5`) ran one exclusive 600-second job,
`u4-gemma31-multirow-r-r1`, with normal TERM/returncode0 retirement. The retained
R runtime is `0ba727678e726b0c1bd9800d175358db8ac59207c01f10ce5d34f5ec2fc8d28b`;
the complete 2,933-file source manifest is
`0a53dea6a1df1b79002b4e3f1af689ed724ee376c9d8ded94c9d8e4f7b0eb99a`.
Both pre/postflight identities agree. The original artifact, F16 KV, context8192,
two slots, prefill_chunk2, 1,071 IDs and two 128-token requests are unchanged.

The generic GB10 repair selects incumbent reduction geometry for ordinary,
non-ID, ungrouped columns2–8, including GLU. Scalar, explicit OneToken, routed,
grouped, other-device and above-eight paths retain their earlier schedules.
Independent operand/replay checks pass 178 ordinary cases: all18 encodings at
C2/C4/C5, 17 atC8, with Q2_K C8 outside the pinned incumbent oracle's domain.
Q2_K C5/K4096 remains checked. These are generated operands, not captured model
weights; the real model contrast below supplies the separate live control.

The source difference from original A is fully classified:46 changed and13
added paths, including prior FP8 format/ABI and DeepSeek sampling changes.
It is not represented as a two-file-only runtime difference. Production compiler
serialization reconstructs all196 original translation-unit primary identities
by substituting only authenticated old source/configuration hashes;127 consume
changed inputs. The ten loaded dependencies match. The authenticated production
IdentityOf map retains all declaration fields except the exact old/new module
token; all20 observed A and22 observed R concrete/logical pairs match it.

## Full-plan and reply review

Independent review reparsed the actual raw log and matched all15 R plans to
retained A by semantic domain and complete operand relationships. It checked
encoding, multipart layout, shape, stride, views, backing, roots, dependency
edges and unaffected arithmetic; graph-local IDs and incidental chat completion
counts are not acceptance conditions. Norm, RoPE and attention remain unchanged.
Allowed differences are explicit Q8_1 input preparation and fused/separate GLU.

Every full two-row model plan includes196 Q4_K,24 Q5_K and70 Q6_K dense
products,60 GLU products (108 Q4_K/12 Q5_K weight operands), and one prepared
Q5_K output head. Actual K values include5376/8192/16384/21504; the tied head
is K5376,N262144. These actual operands, rather than recipe totals alone,
establish the contrast.

Score completes535 two-row uses plus one scalar tail:536 uses/1071 input rows
and1070 aligned teacher targets. Actual joined decode is phase1/stage0,
rows2/segments2, installed plan102 with112 completed uses. The observed112 is
reported, not a fixed requirement. The two chat streams match A's text, hashes
and usage exactly. Their openings are coherent, but both end by length at128
tokens; neither completes its requested task. This is no complete-answer or
drafter-quality qualification.

Root approved the independent full-plan gate before pairing. Its receipt is
`d70e9717fa5c3404d263fb883236e5eda7d45be8acd6b8ebd7f05a4917e2c765`;
the relational report is
`0e7eed2e42643198db037878ed0c59ccf1109b88d613bef550b0a1ef9eed9bfc`.

## Unchanged numerical criteria

| Criterion | Observed | Fixed bound | Result |
| --- | ---: | ---: | --- |
| Aligned teacher targets |1070, all identical |1070 |PASS |
| Absolute PPL delta |0% |≤0.5% |PASS |
| Mean absolute NLL difference |0 nats |≤0.05 |PASS |
| Maximum absolute NLL difference |0 nats |≤1.0 |PASS |
| Fraction within0.1 nat |1.0 |≥0.95 |PASS |

Both mean NLL values are9.596567088186333. The original quality_pair ran once
as supervised metadata-only job `u4-gemma31-multirow-pair-r1`; report SHA-256
`507546bceb36a44d27e5712761a6c01349ff7fe1cddb718cd9c387f06eccaf68`.
No numerical bound, oracle, grant or source input was changed after collection.
No further repair quality run was made. Debug collection durations are not
speed results.

## Evidence and limits

Raw records, full plans/logs, requests/SSE, launch/pre/postflight, proposal and
approved manual receipts remain outside Git under
`/tmp/llmp-u4-gemma31-multirow-repair-r1/` on Spark B and the workstation.
The R plans hash is `6451e425f1d598f3cf10079949e0fff30e347ae6908f370ccdfc10d8a19d5d73`;
audit `2cb1c373b971e8419048485a5920e7b1abab4d828e03d5843eea5021267c8c01`;
quality receipt `4e7c6c93162fdc06b87c1320fb6266d0361d41c98bf6aa35411264e1981a220e`.
Original A/D records remain untouched in their original namespace.

Preparation failures were confined to read-only validator/wrapper setup and
retained separately; none launched inference. The successful R and pair each
ran once. This bounded result qualifies the registered ordinary target transfer;
assistant draft/verify/state, matched adoption speed, other actual changed
domains and whole U4 closure retain their own checks. It does not claim general
Gemma model quality, a new encoding/device, C4 performance or full context.

## Comparator packaging after the run

The executed comparator bytes remain pinned at SHA-256
`866642ca7502ae363bdd3c77aff84c57d69e2f2259e52d3232fc698878dd19f6`
in registration commit `440d4b3d3e7817d7414df8bfab371fb4482a21b4`.
A later license-header check required embedding the existing Apache-2.0 SPDX
notice in the Python file and removing its redundant sidecar. Only those three
comment/blank lines were prepended; the entire executable body is byte-identical.
The packaged file now hashes to `b13aa466d5e0bbd001ea44b098145dfaf024c7ad1df4ffa7504fbcad1d237d75`.
Historical protocol pins intentionally retain the executed file and sidecar
identities; reproduce that run from its registration commit. No comparison or
model observation was rerun or changed by this packaging correction.
