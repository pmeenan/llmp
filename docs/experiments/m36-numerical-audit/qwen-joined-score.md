<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 joined scoring: bounded candidate screen

The candidate warrants further qualification: it improves secondary
same-artifact teacher loss on these 128 targets, while one A–C–C–A screen
shows no clear timing separation. This is not an adoption or migration
result. Original unquantized Qwen3.8 BF16 remains unavailable.

## Policy and execution scope

Both policies explicitly disable inherited `owner_row_invariant` and retain
all other mapping restrictions. C differs only by three owned typed
physical-eight-column BF16-weight/F32-input/F32-output product keys: K2560,
N512/2560/10240. There are no owner/head splits, new kernels or cross-policy
head-bit requirements. These choices remain in the unmerged diagnostic;
providing checked reserved-weight alignment does not install them.

The actual target-only factory uses the owned backend, four independent
33-token histories and 128 next-token targets. Plaintext prefixes and all
132 token IDs were authenticated against the actual artifact tokenizer.
Both arms use the same prepared NVFP4 target artifact
`c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`.
Each was run twice in fresh serial processes on Spark B (GB10), in A–C–C–A
order, on one unchanged source snapshot and binary.

## Observed loss and timing

| Metric | A | C |
| --- | ---: | ---: |
| Mean NLL, 128 identical coordinates | 5.077858728615236 | 5.031231524314546 |
| Perplexity | 160.43016335829185 | 153.12146943995893 |
| First teacher time (s) | 8.115218860 | 8.117131517 |
| Second teacher time (s) | 8.062725807 | 8.085617079 |
| Pair mean teacher time (s) | 8.0889723335 | 8.1013742980 |

C−A is −0.04662720430069012 nats and −4.5556856425% perplexity. Three of
four documents improve; one worsens. Individual losses: 72 lower, 56 higher,
none equal. Same-arm losses repeat exactly. This small secondary comparison
does not establish original-model quality or a general acceptance threshold.

C's pair mean is 0.1533194% above A; A decreases 0.6468470% across its
bookends and C decreases 0.3882460% between runs. The difference is smaller
than the observed drift. There is no steady-serving, cold-storage or 1%
migration claim, and no extra repeats were run.

Fixture walls in A–C–C–A order are 12.711/12.702/12.650/12.571 seconds.
Fixture minus teacher is 4.595781140/4.584868483/4.564382921/4.508274193
seconds. That residual includes construction, setup, reporting and retirement;
it is not an isolated construction timer. Teacher time includes normal
scheduling, completed head transfer/hashing and scoring.

## Selection, state and limits

Each arm records 2,482 ordered implementation entries for both actual
physical-four and physical-eight score shapes. At eight columns, exactly
50 names change from `ggml.mul_mat.mmf` to `llmp.mul_mat.mmvf_rows`:
48 routers and two PLE products. No other names change. The four-column
remainder is unchanged. Owner maps, parameters, tensor geometry, strides and
source WeightRef/part-role bindings agree; selected registry indices differ
only on those 50 products. Full same-arm selected-program text hashes repeat.
C also reproduces its earlier completed same-binary qualification receipt.

The separate completed C state case retains 140 full-head row comparisons,
four checkpoint restores and four spill restores with exact initialized
state. State replay was not repeated in this timing batch. Wider physical
12/16, long-context, MTP, serving and migration qualifications remain open.

All source/binary/input/light-artifact guards pass, memory events are zero,
no GPU PID remains and complete owners retired. The observed 1,086,496,768-byte
cgroup peak is a host/process/cache observation, not total unified GPU
occupancy. The existing 104 GiB node allowance, including 8 GiB host metadata,
112 GiB process/cache ceiling and 6 GiB physical margin were unchanged.

## Provenance and receipt correction

The bookend used binary
`3f32688180bc15d1dfd308f4ab9123bdb75706187bf154a73ffa4fa6e48305bd`,
source-map digest
`79bc9def39cf02de5dd08e1aaf007494f216c9724d6df5c31efd6846812a4602`,
and token-ID digest
`e9baa394c28c44359f2a67eca03aba57b5d7e62626326be0085e584bbd327bdd`.
Raw data remains outside Git under `/tmp/llmp-m36-coordination/`.

| Evidence | SHA-256 |
| --- | --- |
| `qwen-recovery50-candidate-models05/actual/result.json` | `640256b7cb04b0616c9f6c2b7d7e92108adf25d28798e52b646cfc4fde304ba0` |
| `qwen-recovery50-abba-models06/actual/result.json` | `9e8e9d187ee0b2681ad0d31b08364e55fad017a3920f977b42d56bad125a4199` |
| `qwen-recovery50-abba-receipt07/result.json` | `7833013a4fcce57fd42437b0e4b811f79b36c547be2dea9afb9ce0f773e6f59d` |

Wrapper06 remains **FAIL** despite four completed native PASS cases: its
extra prior-C check compared Python tuples with equivalent JSON lists.
Reviewed receipt07 reproduces that representation-only mismatch and
re-audits every raw XML/log, guard, coordinate and same-arm result. Its
corrected offline audit is PASS. No inference rerun or numerical acceptance
change occurred. Independent final arithmetic/report review is
`0767efc9caed9a2515f39c2ffc6931fff455a4f6aeb9c056c67e3ff71d56f410`.

## Actual decode continuation — 2026-10-12

The separate C-only Qwen51 control passes on Spark B in 15.629 seconds
(installed exclusive job: 16 seconds). Four authenticated 33-token histories
seed the actual target scorer; their terminal heads supply greedy anchors.
Real Plain decoding then requests complete F32 vocabulary heads for eight
four-owner steps. It checkpoints at position 37, restores and replays the next
four steps, then restores again, spills/restores all owners and replays those
four. All 32 reference heads are finite; 32 replayed full-head hashes and chosen
tokens match exactly within C. Eight checkpoint restores and four spill restores
pass, with positions 33 → 37 → 41. Restored initialized state is compared at 37;
final state bytes at 41 are not independently saved or compared.

The new actual four-owner Decode/read256 class independently passes source
measurement, lowering, complete output-owner coverage, closure reservation and
Bind. Decode/read512 supplies only a shape template. All 17 classes are measured
before execution, with 8,432,673,280 bytes of derived construction/retention
funding inside the unchanged 8 GiB host allowance. Five separately admitted
state/readback buffers are reused. The actual Score4, Score8 and Decode4 programs
each record 2,482 ordered implementation entries before execution; typed MMVF
counts are 0, 50 and 0. Score4 and Decode4 have identical operation lists and
operation-list identities, with distinct phase keys. No copied score program
is used as decode. The seed's 128 losses reproduce prior C as provenance only.

All source, recipe, binary and input guards pass, memory events are zero,
no GPU process remains and the complete owner retires. The observed
2,776,756,224-byte cgroup peak is host/process/cache usage, not total unified
device occupancy. The private fixture remains outside main. This closes the
short actual-decode/checkpoint/spill diagnostic, with no adoption, original-model
quality, broader context/owner/MTP/swap or matched 1% qualification.

| Evidence | SHA-256 |
| --- | --- |
| Frozen fixture source manifest | `c3f3c528a60733e79d9fb3241d9d5cd8ac00c58f8af9864d8aa5717811684a37` |
| Literal focused build receipt | `f59e94f60b6c27e590fcf2aca9b9a8f2dc4832ed51271c73a6eaddc49ad64d30` |
| Native binary | `9bbca9a251afd059b66312d01b4c4289e9123353fbd566e1438ef2f6b322e340` |
| Actual completion receipt | `cb3fd1db196fecc5477df9a197f7eb778b03c7eb10a0e47d38dfc8f3c30bc244` |
| Independent raw/result audit | `32339e6ca74ff7599c745687d5bb3b7fb7788f0728b8587ac3ea54fc6a3f1ccb` |

Raw evidence remains outside Git in `qwen-recovery51-decode-models01/actual`
under the coordination root. Further work should address broader quality or an
unqualified lifecycle/domain requirement; these short checks need no repeat.
