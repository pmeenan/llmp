<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Remaining U4 transfer results

**All 14 registered head/state/transport arms passed, and their actual-plan
relationships are independently clean. Two current replies per family
(four samples total) also passed bounded semantic review. The matched speed
screens and final source checks now pass at the recorded scope.** Arm-level
receipts and review bindings are recorded in [results.json](results.json).

## Registered head and state screens

The fixed one-process-per-arm screen completed 14/14 with return code 0,
zero skips, complete exports, unchanged pinned inputs and clean retirement:
four Gemma31, four Gemma26, four DeepSeek/DSpark, and two plain Qwen GGUF arms.
No prompt, bound, policy override, or frozen expectation changed.

Strong's independent actual-plan review
(`u4-remaining-model-plan-independent-review-r1.json`, SHA-256
`9f24242f1132d2972726af0dceeaab9ec061071bfe37afd58dc1164826b2adbc`)
is clean for Gemma26, Qwen and DeepSeek. It checks complete operand and plan
relationships across the actual domains. Gemma31's four-arm review is also
clean (`g31-four-arm-plan-independent-review-r1.json`, SHA-256
`ef1e0ababf70b828e2165f7600415c27b0ff3e25aca281ef999e80faa1089a6c`). The
separate family relational supplements pass for Gemma26 (all seven domains,
170 uses per arm), Qwen (three domains, 16 uses per arm) and DeepSeek (six
domains, 20 uses per arm). These are plan/source relationship proofs, not
kernel traces or new numerical grants.

DeepSeek's separate dose note is only a historical source-and-operand bridge.
It identifies a historical two-row prefill geometry, but does not claim that
the current registered workload executed a two-row prefill. Its actual captured
domains and target verify4/stage1-draft plans are covered by the independent
plan review. Do not describe the prospective two-row bridge as an executed
two-row qualification.

## Historical wide-score reuse and current replies

The read-only teacher-reuse audit (SHA-256
`5a9bba23ab67cd32703df3b55158703e103dd32cc3e15d108ac3a7df31193001`)
supports reuse of the historical Qwen phase-4 score plan with 997 rows and
996 aligned targets, and the historical DeepSeek phase-4 score plan with 967
rows and 966 aligned targets. This applies only to the unchanged wide-score
scope. No current teacher was run, and this audit does not qualify assistant
state or reply quality.

After the head/state gates passed, each family received exactly two serial
current-candidate replies under the registered temperature-zero, 128-token
bound. Both runs passed operational checks and independent semantic review:

| Family | Prompt tokens | Completion tokens | Semantic outcome |
| --- | ---: | ---: | --- |
| Qwen GGUF | 66 each | 128 each | Both bounded prefixes are coherent and relevant with no observed material regression; neither completes its task. |
| DeepSeek + DSpark | 17 each | 128 each | Both prefixes are reasoning-only with empty final content; neither answers the prompt. |

The response hashes differ from retained A in both families. Hash equality was
not a registered quality requirement. The raw SSE and request hashes, runtime
receipts, and semantic-review bindings are in `results.json`; raw response
records remain outside Git. These samples establish no full-answer or general
model-quality claim.

Frozen Gemma26/Gemma31 assistant, DSpark, DeepSeek target and Qwen GGUF
differences remain recorded failures with unchanged expected values; the positive
upstream controls do not relabel those historical results. The prior strict Mia
failures and unchanged-target limitation remain in the
[LLM report](../LLM-RESULTS.md).

The historical Gemma31 A/D numerical screen remains a frozen failure: it
completed 1,070 targets but failed four registered bounds (13.2747% absolute
perplexity delta, 0.57510 mean absolute NLL difference, 19.09824 maximum
difference, and 361/1,070 within 0.1 nat). The later repaired R result is a
separate run and does not rewrite or replace that negative. See the original
[A/D results](../gemma31-multirow/RESULTS.md).

## Final checks and speed qualification

The initial shared CTest XML contains 3,693 tests, zero failures, zero errors
and zero skips; CTest returned 0 in 553.84 seconds. Its wrapper returned 1 only
afterward because the summarizer looked for a nonexistent `shared/junit.xml`.
The actual XML and `LastTest.log` were independently validated; that source
state was not rerun to repair the wrapper.

A real speed regression then required the narrow GGML selection change. Its
final source builds natively on both Sparks, passes 174 related lowering/planner
checks on Spark B, and passes the new final 3,696-test shared suite on Spark A
in 553.64 seconds with zero failures/errors/skips. Source guards before/after
the build and tests match all 2,935 pinned paths. This later source-state check
is separate from the earlier 3,693-case receipt.

The [initial C1 screen](../remaining-c1-speed/RESULTS.md) passes Gemma26,
Gemma31 and DeepSeek and preserves Qwen's failed first-token median (1.121499,
bound 1.10). The [separately registered Qwen routing repair](../qwen-mmvq-routing-repair/RESULTS.md)
passes one new five-ID head/state arm, full completed-plan review, and the same
six-arm speed comparison: medians are 0.997506 TTFT, 0.972506 decode and
0.977721 wall. Source/domain proof supports reuse of the other three families
and the existing bounded replies and wide teachers; no unrelated qualification
was repeated or broadened. The earlier Gemma31 C2 speed result also passes.

U4's component/evidence contract and bounded transfer qualifications are
complete. M3.5 remains in progress. Historical failures, incomplete replies and
all device/encoding/context limits retain their original classification.
