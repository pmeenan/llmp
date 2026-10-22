<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen27 MLX source-quality results

The frozen Qwen27 MLX checkpoint passes the bounded likelihood and six-case
behavior screens registered in [PROTOCOL.md](PROTOCOL.md), commit
`fa3fed7955e618ddb06e0270c406a8c002ab332d`. Both fixed repeat pairs pass the
+1% native PPL bound, and each engine's own repeats are byte-identical. All six
behavior controls complete correctly in both engines. Collections ran on Spark A
(`spark-c4e2`) on 2026-10-22; an independent audit checked the raw records,
requests, launch receipts and supervised retirement before this verdict.

This qualifies the registered checkpoint, bounded histories and raw-ID behavior
pack. It does not qualify competitive speed, batching, maximum context, ordinary
chat rendering, every Qwen format, or all of U3/M3.5. The earlier
[Llama 256-token Fibonacci failure](../RESULTS.md) remains unresolved. These new
controls also do not turn the earlier [64-token Qwen chat smoke](../QWEN27-GATE.md)
with empty final content into a completed answer.

## Likelihood results

Both arms received the same frozen integer histories, without inserted BOS or a
corpus chat template. Short uses 1,024 IDs and scores positions 1–1,023; depth
uses all 9,216 IDs and scores positions 8,192–9,215. Positions are zero-based.
Every scored record has the exact expected document, position and teacher token;
all scores are finite. There are no missing, duplicate, reordered or extra
records. The source depth run retains the complete prefix state, rather than
starting at the scored suffix.

Each row below applies separately to repeat 1 and repeat 2: neither repeat was
selected or averaged away. NLL is the mean actual-target loss in natural-log
units; PPL is its exponential.

| Phase | Targets per run | Source mean NLL | Native mean NLL | Source PPL | Native PPL | Native PPL change | Both pairs, bound ≤ +1% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Short | 1,023 | 1.960453639399 | 1.961371481211 | 7.102548330209 | 7.109070338669 | +0.091826316% | PASS |
| Depth | 1,024 | 2.331134818494 | 2.333038465027 | 10.289611751716 | 10.309218191394 | +0.190545962% | PASS |

The absolute own-repeat PPL change is **0%** for both phases in both engines,
within the preregistered 0.1% bound. The two loss files for each engine/phase are
byte-identical. The cross-engine per-target differences below are descriptive;
no new per-token acceptance bound was introduced.

| Phase | Mean absolute NLL difference | Maximum absolute NLL difference | Absolute position of maximum |
| --- | ---: | ---: | ---: |
| Short | 0.020935766700 | 0.301451357672 | 155 |
| Depth | 0.026516896485 | 0.402955641991 | 8,693 |

Native request bodies contain the exact frozen IDs, and responses report all
1,024 or 9,216 prompt-score slots, a null unscored first slot, the expected actual
target key at every later position, zero generated tokens and zero cached prompt
tokens. Each retained loss was independently checked against its raw response
entry. Responses do not contain a separate `prompt_token_ids` echo; alignment is
established by the authenticated request history, response geometry and actual
target entries, not an invented token-ID echo.

## Source validation and execution domain

The source adapter validation passed **before** the governing collections. On
fresh states for the first 513 and 1,024 frozen IDs, normal source prefill and the
adapter chose the same first tokens: respectively 13 and 864. At both lengths,
the final row's full-vocabulary logits were bit-identical for 16-row versus
one-row output-head calls. The validation identity binds the immutable image,
preregistration, protocol bytes and exact helper inventory.

The adapter used the pinned engine's own packed MLX weights, prefill, state,
output head and full-vocabulary FP32 normalization. Its recorded source spans
partition the entire 1,023- or 9,215-token teacher prefix without gaps; every
chunk has at most 512 rows. Final source state positions are respectively 1,023
and 9,215. The admitted prompt-row budget was 4,096, then the registered 512-row
limit was applied. Scoring output-head batches remained at most 16 rows.

**Precision differs as preregistered:** TensorFold uses BF16 prompt arithmetic
with FP8 prefill disabled and BF16 attention KV. Native uses its production
component plan and F16 KV. Both use context 12,288, one stream/request slot,
prefill limit 512 and no drafter or optional lossy mode. The results compare
these declared execution domains; they are not evidence of bit-identical
intermediates or proof that KV precision alone caused the loss differences.

Source capacity receipts retain explicit context 12,288, 12,304 cache slots and
a serving working-set estimate of 18,150,476,800 bytes, below each admitted
budget. That is the source admission estimate, not measured peak memory or
isolated GPU allocation accounting. Native startup guards passed, adopted zero
prior conversations, reported speculation disabled and no tensor-coverage
violations. Every run used a fresh state directory.

## Completed behavior controls

Each engine received the same frozen source-rendered IDs, ending in the completed
empty thinking block (`enable_thinking=False`). Each case ran in a separate fresh
process, greedily, with the unchanged 512-token maximum and EOS stopping. All
12 replies have `finish_reason=stop`, not `length`, with zero cached prompt tokens.
The independent review inspected all actual source and native texts.

| Control | Prompt IDs | Source / native completion tokens, including stop | Actual outcome in both engines |
| --- | ---: | ---: | --- |
| Capital | 26 | 2 / 2 | Only `Paris` |
| Arithmetic | 27 | 4 / 4 | Only `391` |
| Sorting | 37 | 13 / 13 | JSON array `[1,2,2,9]` |
| Structured data | 35 | 21 / 15 | Exactly `{"planet":"Mars","moons":2}` as a parsed object |
| Local context | 47 | 6 / 6 | Only `AZ-417` |
| Fibonacci | 59 | 67 / 67 | One complete `fib(n)` function, no markdown or explanation |

Five answer texts are byte-identical between engines. The structured-data texts
use different whitespace but parse to the same exact two-key object. Both
Fibonacci functions are identical: a zero-input return and a bounded `range(1,n)`
loop appending successive values. After inspection, restricted local execution
checked n=0 → `[]`, n=1 → `[0]`, and n=7 → `[0,1,1,2,3,5,8]` for both copies.
The completed, sensible nonnegative-n implementation passes the registered
functional control. No generated code was executed before inspection.

## Provenance and completion

The [frozen protocol](PROTOCOL.md#immutable-identities) records the full source,
image, checkpoint, archive, native manifest and ModelSpec identities. This audit
verified the actual source receipts' 449-file authentication, immutable arm64
image inspection, frozen metadata and input digests, staged helper bytes against
the preregistration commit, and native executable hashes before and after every
run. The earlier official stage verified 17 checkpoint files totaling
16,081,520,742 bytes; this audit reused that retained provenance and rechecked
small metadata, without rehashing the large weights.

| Executed identity | Value |
| --- | --- |
| TensorFold source | `cb2ebf0540f42604e2759b2ddef497861e928248` |
| Source image | `sha256:1010df9519efb5fb2ca8a09c1504e95829206e4fff8c8e8f3b671b4d71a6737a` |
| Source 449-file inventory SHA-256 | `8605fa47ae126abb91a79688ab72cb440032679b741ce3bcd7aeb75ae65a7cd3` |
| Protocol JSON SHA-256 | `a82d8fd1e24115037ea64edbd74c5e3e0ef130b328cd6a0f3cadfcc834f30240` |
| Helper inventory SHA-256 | `17f8868ab69705e4177a2403f3343149562e1d06feb349b1d9f2b6cac63610bf` |
| Successful validation JSON SHA-256 | `f2ebb52b522e7b55cebaa8d870858da5991945be03b9c5052a00dc48d78e6859` |
| Native runtime SHA-256 | `078a8da5a78242477891757b88675ed685b4ebe6d687db634c8c116cef1f3181` |
| Native artifact | `bbf67ee688746792a1fbb06552b59bc0132e74bc79ced2333302b431549d0fa5` |
| Native service helper SHA-256 | `0f1135a2283fe63797e15c8dffe48c9df39b0baae91a57f925b5ba21c3e3252e` |

The native binary is the preserved [gate-repair build](../QWEN27-GATE.md), based
on `2683b9070da87d759ccd2b6f3c507f0a5a0287c4` plus source patch
`7a519a525d900ceb34f6ac81d27e6467ef15c649d7712113bddf151b62220d62`.
The checkout's later commit or dirty version label is not its source identity.
Every native configuration hash matches the retained actual configuration;
only the fresh output/state path varies among these otherwise identical recipes.

| Phase / engine | SHA-256 of each repeat's `losses.jsonl` |
| --- | --- |
| Short source | `1722f665a3abc26f3568291650e1e91e092dcf0d01c4f2a9476f5762a56dc216` |
| Short native | `6a56aca38db4ffc89d80b89531d90f2a87c7c0acd8869a4e449b2e405ee90277` |
| Depth source | `8f66032933a8b89cd2296b8699bbe97f9ca77b1cca2a8253461447812352ca44` |
| Depth native | `d1e0d002bcc40a1dcc5e21d460b4e03f18f7b678307ffc8dafac3b129b65721c` |

All 21 successful model jobs—one adapter validation, eight likelihood runs and
12 behavior runs—used exclusive 600-second Spark supervision, completed with
return code zero and did not overlap one another. Source launch receipts confirm
exact container-name absence after retirement; every native service retired
through TERM with return code zero. Actual raw receipt, request, response and log
files are 0600, with their output/collection directories 0700 on the host.

One attempted source depth launch is retained as a negative operational receipt:
`m35-u3-qwen27-source-depth-r1` received `depth-prereg.json` (the pairing-bound
file) instead of `prereg.json` (the collector protocol). It failed the frozen
protocol check with return code 1 before output-directory creation or container
launch. The corrected invocation used a new directory and job name,
`m35-u3-qwen27-source-depth-r1-attempt2`. It changed no helper, input, model setting,
repeat, target or bound. No collected quality cell was dropped or replaced.

Raw receipts remain outside Git under
`spark:~/.local/share/llmp/m35-u3-quality/`: validation in
`qwen27-source-validate-r1`, likelihood in `qwen27-{source,native}-{short,depth}-r{1,2}`
(with the source depth repeat-1 `-attempt2` suffix), and behavior in
`qwen27-{source,native}-behavior-CASE-r1`. Their supervised job names have the
`m35-u3-` prefix, except validation's `u3-qwen27-source-validate-r1`.
The independent machine-readable audit and read-only receipt copies remain in
`/tmp/llmp-qwen27-quality-review/`; the audit recomputed all 8,188 governed loss
records and bounded function checks under a local shared hostlock. Raw artifacts
follow the milestone cleanup policy; the aggregate results and identities above
remain this record.
