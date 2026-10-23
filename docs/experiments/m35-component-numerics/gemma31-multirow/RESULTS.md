<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 ordinary multirow quality screen — failed

**FAIL under the unchanged, preregistered numerical bounds.** The candidate
VecQ arm completed all 1,070 teacher targets, but failed four of the five pair
criteria. Only the target-count check passed. The frozen inputs, bound file,
oracles and numerical grants were not changed. This result grants no numerical
qualification and does not qualify the Gemma31 family generally.

The prospective [protocol](PROTOCOL.md) fixed a two-arm comparison of the
existing ordinary two-row VecQ path (D) against incumbent MMVQ (A). Both used
the same retained runtime, published target, corpus, 1,071 token IDs and two
chat requests. A ran once, then passed its full-plan review, which authorized
D. D ran once, then passed its separate full-plan comparison, which
authorized the unchanged registered pair. The pair ran once; its failure ends this screen. No
retry, adaptive workload change, bound change or follow-up inference was made.

## Frozen source, runtime and artifact

| Input | Identity used by both arms |
| --- | --- |
| Registration | Commit `9e71a8008a3b80384263a688b500397bb6ec7053`; preregistered pair file SHA-256 `d915aa85471f553a283d572d868a49ce60da2ada33d3e5550a1ed787f1dbdd64` |
| Host | Spark B, `spark-56f5`, driver `580.178.04` |
| Source | Immutable 2,920-file snapshot at `/home/pmeenan/.local/share/llmp/baselines/m35-gemma31-multirow-r1/source`; manifest SHA-256 `425ca8f4df184d19ce8c5a467cfc7d03b4c3665d70370e65577970bed48bdef3`. The Spark checkout's dirty Git ID `61f8426f7e2f45931d91645b016069e13f6f2236` is lineage only, not the source identity. |
| Runtime | Retained `llmp-runtime` SHA-256 `84510e35229fe96255737700abde85472706df51ed6f9d71ebc5f6298e3c3a37`; pre/postflight hashes match in A and D. |
| Build lineage | Pinned build receipt SHA-256 `5390ed59a7019a6e3b14f355b60b6c2fe3bb884adf4153f6f0665f2ca7d89578`; compiler identity receipt SHA-256 `881fc140fbdeefff54c70119648920473cf817a008a3fa0280fce661f77f2074`; compiled `ggml/module_digest.cc.o` SHA-256 `dd7749d20e8b01a60fd01ed25dda39b9ecbeab853083e4518cdb95e3a96c4454`. The complete resolved dependency closure and retained receipts are bound in the registered [pins](pins.json). |
| Target | Artifact ID `090809981d2909085798273e665a0db58351a8e6029b13bc04ced857c9130943`; semantic digest `995a4afb665d34f27c86b76530e20de97c51a6c2f9d295e98f360373f80f9317`; manifest SHA-256 `b29cf16db21ff94b72340d77347c531c1b7a4e5a26a3e4938aebf7439fc96cdd`; payload 18,807,134,448 bytes, SHA-256 `dcad28fa7c93595035aa0d45059ee9660137e20bcb1d3052816d402ff9927b2b`. |
| Scoring input | Corpus SHA-256 `55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb`; full input-ID int32-LE SHA-256 `e3c3bcd233290b3ec0219b64ada504eb90363d995af9a59b5acd89541cce40af`; 1,071 IDs including one BOS and 1,070 teacher targets. |

The arm policies differ only as registered: A disables `grouped_vecq`,
`owner_padded_reads`, `vecq_glu` and `vecq_inputs`; D uses ordinary VecQ with
`owner_padded_reads` disabled. Both retain the 120,259,084,288-byte node cap,
startup host budget `auto`, `max_decode_rows=16`, F16 KV and no drafter.
Identical pre/postflight source, runtime and artifact hashes bind the two
executions to these frozen identities.

## Actual plans and request behavior

Independent A and D reviews each re-parsed all 15 plans from the complete
retained runtime log and checked actual operand relationships, not aggregate
recipe totals. The reviews bind the full plans, logs, requests, quality files,
configuration and runtime bytes. A's manual gate is
`6da9dd76e9769fe4de0dddab0b42ae6671b89e78585e860d450d7e8b5f3e039b`; D's is
`1c26f575feaa4fc031b763e62464489308c94c04e6121c07c8ee004e3875c61c`. The
independent full-plan review digests are `82de6fc4a6401a268a1c0c44668d3ec449c26ce036e0fcd39dcc675aa7c94d57`
and `9b627c546523c3ffe1f079f1cc40788476adb04ccb4674727686e9ceb269b638`.

Each arm completed 536 score-plan uses over 1,071 rows: 535 uses of two rows
and one final use of one row. That is distinct from the 1,070 aligned teacher
targets. In both arms the joined decode plan explicitly reports installed
plan 102, stage 0, phase 1, two rows and two segments, with 112 completed
uses. The retained caller-chain and plan geometry establish two one-row chat
owners in the joined decode. Request or slot IDs are not explicitly emitted in
the runtime plan log; the owner attribution comes from the independent source
and operand review, not from an asserted logged owner-ID field.

The per-operand review matched all weight encodings, physical and semantic
dimensions, block strides, activation/output shapes and source edges against
the complete artifact index. It also checked every unaffected arithmetic and
descriptor relationship. For each full score/decode model plan, D has 290
dense VecQ weight operands (Q4_K 196, Q5_K 24, Q6_K 70), 60 fused GLU layers
(54 Q4_K and 6 Q5_K; two independently checked gate/up operands per layer),
and one prepared tied Q5_K output head: 411 quantized weight operands total.
The dense products use 241 shared Q8_1 preparations. No Q8_0 weight occurs;
Q6_K is not a gate/up format. A's full two-row score/decode plans have 411 ordinary
incumbent MMVQ products, including the 120 gate/up products consumed by its
60 separate GEGLU layers, each linked to its actual operands and shared
activation. Its scalar tail has 291 MMVQ products plus 60 incumbent fused
GEGLU products. Both arms retain the tied Q5_K head at K=5,376, N=262,144; D's
prepared form retains the original F32 third operand. The actual GLU layers
are K=5,376, N=21,504 and use Q4_K or Q5_K. These operand-by-operand
relationships, including the allowed input preparation and fused/separate
GLU differences, are the reviewed contrast; counts alone are not evidence.

There were exactly two concurrent chat requests in each arm, with identical
input bodies, 26 prompt tokens each and 128 generated tokens each. Both replies
were byte-identical across A and D: SHA-256
`5318071dc3e76684c535cebabadfe431c70373aff025cb416cca17748a58b252` and
`fd56b9a14e2e890f4d313102419f96afd1f16b381a39fe58687e00e45e47b357` in
request order. Independent review found both openings coherent, but both
streams ended with `finish_reason=length` at the 128-token cap. Neither is a
complete response to its task; matching bytes or coherent openings do not
constitute behavior or quality qualification. Raw reply text and SSE samples
remain outside Git.

## Registered score comparison

The table reports D candidate minus A reference where applicable. The original
fixed limits in the preregistration are unchanged.

| Criterion | Observed | Registered bound | Result |
| --- | ---: | ---: | --- |
| Aligned teacher targets | 1,070 | exactly 1,070 | PASS |
| Candidate mean NLL | 9.721212697334503 | — | — |
| Reference mean NLL | 9.596567088186333 | — | — |
| NLL delta (standard error) | +0.12464560914816936 (0.03843357013298855) | — | — |
| Absolute perplexity delta | 13.274694677057486% | ≤0.5% | FAIL |
| Mean absolute per-target NLL difference | 0.5751016687706009 nats | ≤0.05 nats | FAIL |
| Maximum absolute per-target NLL difference | 19.098237628581543 nats (document 0, position 522, target 236812) | ≤1.0 nat | FAIL |
| Targets within 0.1 nat | 361/1,070 (0.33738317757009345) | ≥0.95 | FAIL |

D's perplexity is 16,667.445044714426; A's is 14,714.182273660306.
Candidate scores are lower than A for 437 targets and higher for 633; none
are identical. The pair report returns `pass=false`. This is a genuine failure
of the fixed quality gate; no threshold, oracle or numerical authority was
relaxed.

## Setup failure, retained evidence and scope

The first A job, `u4-gemma31-multirow-a-r1`, failed before inference with zero
replies because the retained runtime copy had mode 0644 while its pinned
retention receipt required 0755. The original failure summary
(`4f6658c99b823a1691b0310eb2b31dbb9f2e4a60762467ff3ba068967cf8fae8`) and
all first-attempt outputs are preserved under the external
`/tmp/llmp-u4-gemma31-multirow/setup-failure-r1/`. The authorized repair
changed only the runtime executable's mode to 0755; its SHA-256 remained
`84510e35229fe96255737700abde85472706df51ed6f9d71ebc5f6298e3c3a37`, and
the 2,920-file source manifest, runtime dependencies and artifact metadata
were revalidated unchanged. Repair receipt SHA-256:
`6f27734cafd8c811f20d2fa41f7f22df96d7d21411ebf62de202104109a8dca6`.
The subsequent A run, D run and pair were each executed once; no inference
retry followed a valid arm or the failed pair.

The external evidence tree is `/tmp/llmp-u4-gemma31-multirow/`. It retains
the full A and D plans/logs, request/score records, SSE files, manual-review
receipts, failed setup attempt, pair report and supervisor logs. The complete
plan digests are A `79c9a52a820aec254009aaba99d35c338c9a201b6d8b3885bd76a9d26dcab0d9`
and D `5e94839b583e9faee2b66cc2e026e63412b169681cdf1a75424a106bd205c94d`;
the pair report SHA-256 is
`73e56d81aa57ad0399d4a56a76ea1a5da3545056d4a64ffd2fdd0cc64dcc748c`.
The per-file transfer manifests and arm receipts also remain outside Git.
Raw response text, raw samples and execution logs are not copied into this
report.

This screen does not change production code, frozen numerical expectations,
family policy, or numerical grants. It establishes a reproducible failure
for the registered Gemma31 ordinary multirow comparison only. The general
cause and repair remain under investigation; this report makes no claim that
the current path is qualified or that this single result generalizes to
other architectures, encodings, shapes or contexts.
