<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3.5 native tool-call diagnostics

These are bounded behavior diagnostics for Qwen GGUF tool calls, one native
mixed-encoding Mia named-tool transaction, and one actual OpenCode client task. They are representative evidence for the native
route enablement spike, not the full M3.5 tool-call exit qualification. The
complete item remains open for the rest of the served aliases and grammars
and client coverage. Final integrated source checks are tracked separately.

## Frozen run identity

The six API transactions and the OpenCode task used the same frozen W3
native runtime identity: source-manifest SHA-256
`f5a2ecf7769124b24425746323d7c37c01a2fc48498f8269effd15d9062b328d`, with
2,812 source-file hashes unchanged pre/post; runtime SHA-256
`39665bb40bd90c43830d6f2250773d6e8858e936307ac8d38ec00d24f427e7e5`; and
approved diagnostic-helper SHA-256
`bd2410d633524a10039ca441fe6f7a92d7d651e20482215ea21db7ace86c3243`. The
artifact manifest SHA-256 is
`f60d801a1695079ba4cd9c17d32967315bf77fc26580c1f8245b37f95d03d027` (the
source manifest and artifact manifest are distinct identities).

The tested weight artifact is `unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee`
(UD-IQ3_XXS); its artifact manifest SHA-256 is
`f60d801a1695079ba4cd9c17d32967315bf77fc26580c1f8245b37f95d03d027`.
Tokenizer and chat-template assets came from
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`, with recorded SHA-256 values
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` and
`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`,
respectively. The NVFP4 checkpoint supplies tokenizer/template assets only;
those W3 diagnostics do not qualify NVFP4 weights. OpenCode was pinned at version
1.19.3, client SHA-256
`e5309bc8df8deac0e4050fb4a65989839522703cf86f8a4561c2694f5157b8ca`.
Pre/post source, model metadata, runtime, helper, and client identities were
equal in the retained receipts.

The API diagnostics used context 16,384, one serving slot, prefill chunk 512,
no speculation, and a 1,024-token output budget per request. Sampling was fixed
at temperature 0, top-p 1, min-p 0, top-k 0, seed 7319. The task asked the
model to read a local marker through a declared tool and report the returned
value. These controls describe the tested run; they do not establish maximum
context, performance, sampling quality, or speculative-decoding coverage.

## Native API tool/result transactions

All six unique transactions passed: `auto`, `required`, and named `read_marker`
choice, each with non-streaming JSON and SSE streaming. Every first response
finished with `tool_calls` and issued exactly one `read_marker` call for
`marker.txt`. The stable call ID linked the parsed request to the one executed
local tool and its result message. The tool returned
`LLMP_TOOL_NATIVE_RESULT_7319`; every follow-up reported that value. Usage in
the parsed responses matched the receipts, token totals reconciled, and each
request stayed within its 1,024-token budget. Each runtime exited zero and was
reaped with no live members, kills, retirement errors, or unexpected
descendants; all ports closed.

The earlier `auto`/JSON helper run `tool-qwen-auto-json-diagnostic-r1` remains
**FAIL**: its owned-group retirement race left the leader unreaped. It is
retained as a separate failure and is not relabeled by the six passing r2 runs.

## Actual OpenCode task

The pinned OpenCode client received a seeded bug-fix task and produced four
successful tool events: read `clamp.py`, read `test_clamp.py`, edit only
`clamp.py`, and run exactly `python3 -m unittest -q` through `bash`. The
initial seeded test run failed three of five tests against the intentionally
buggy implementation. After the edit, the independent rerun passed all five.
The test file remained byte-identical (SHA-256
`38fc1467d220c8b41d3d253399fdfcf973d84ce8945b9da9634234c8dce33c18`). All
runtime, client, helper, model, template, artifact, and source identities
matched before and after. Runtime, client, setup, and test children were reaped;
there were no kills, retirement errors, or unexpected descendants, and the
port closed.

## Native mixed-encoding Mia qualification

The native artifact is
`7175c03c1748aa96af84667f2e51e53d56e954d7fb3b6ccc1083f882b41501a2`,
from `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6c14c6c9442ef83e8f05b5a3c39304f69`.
Its manifest-file SHA-256 is
`6c79f701b97f087171f8765f001bd2f0ebd20782eff53275726d062b6b678c56`
and StoredSpec SHA-256 is
`02d130bce7462794251cc29e5554565a0ad83a398ce2776fd8078ef4b683618a`.
This is one native ModelOpt checkpoint with mixed NVFP4/MXFP8 encodings,
not independent qualification of every checkpoint in either format. Tokenizer
and template hashes are the Mia assets recorded above.

The first named-tool request failed on the frozen W5 source/runtime with
`tool_output_invalid: incomplete tool call at natural completion`. Its
persisted history retained 52 processed generated IDs after the 347-token
prompt; the unprocessed final anchor is absent. Reversible byte-alphabet
inspection of those IDs recovered an unambiguous call using
`<parameter name="path">marker.txt</parameter>` and `</invoke>` instead of
the declaration's canonical Qwen spellings. This was generated output, not an
input-parser finding. The original HTTP 500 and its retirement records remain
**FAIL**. An error-path usage count of zero did not mean the model generated no
tokens.

The shared interpreter now declares these alternatives only at structural
boundaries, preserves their control-token encodings, and retains literal alias
text inside open raw values and quoted JSON. Actual stop IDs cannot end an open
or unfulfilled required call. Exact declared tool markers continue through
legacy route-only stops, and speculative judging ends at its first accepted
natural terminator. A scripted common-engine control reproduces the old
post-EOS `sampling: no-logits` failure and passes with the fix, checking public
tokens, score rows, retained history and retirement. Native focused checks pass
344 unique CTest entries without skips, including both review regression
controls and plain/sampled-speculative route-marker transitions.
The final source also passes the complete shared set: 3,611 unique cases,
combining those 344 focused checks with 3,267 other non-model/non-timing cases.
Source closure and the actual compiled implementation-identity check pass.

The fresh native plain named-tool transaction passes on `spark-b`, source
manifest SHA-256
`26e2a203348eb45a60e0c1584fcec285f9e6f96bc9890d2ff44927f1147ac911`
(2,817 files), runtime SHA-256
`107fc9cc22cb9f6d9e7980fe0587e48a9ce117ff5568d00753a20d01822c5843`,
and unchanged qualification-helper SHA-256
`324d8441d60f7893c2817f00644a70fe828148c6a5234dbd6d09338dcb3312d9`.
The initial JSON response issues exactly one `read_marker({"path":"marker.txt"})`
call with 347 prompt and 53 completion tokens. The actual file read supplies
`LLMP_TOOL_NATIVE_RESULT_7319`; the streamed follow-up reports that value with
474 prompt and 35 completion tokens. Both requests retain the 1,024-token budget,
context 16,384, one slot, prefill 512, F16 KV, no speculation, and explicit
T0/top-k0/top-p1/min-p0/seed7319. Pre/post source, runtime and model-asset
identities match; the runtime exits zero and is reaped without kills or live
descendants, and the port closes. Raw receipts are outside Git under
`/tmp/llmp-m35-orchestration/native-tool-fix-r4/model-qualification/plain`.

The original sampled-MTP `required` cell failed before readiness: legacy companion
`2cb4cdb8d4c8bba3bf7a258d56edbae1b456d9268c238948f00210f7b1e9d0ef`
is declared for `qwen4exp`, while the new target manifest names
`qwen3_8_flash_next`. The process exited one, was reaped, and left no live group
or open port. No request or speculative unit ran; the original required and
named inference controls remain **NOTRUN**, and the admission result remains
**FAIL**.

After the canonical mapping-identity repair, fresh separate sampled-MTP
`required` and `named` cells passed on the same target/companion pair. Each
executed the actual marker read and completed its streamed result follow-up.
Required recorded 53 speculative units and 58 drafted tokens; named recorded
51 and 64, with zero coverage violations in both. The requested draft vocabulary
was 65,536; the effective value was 47,172, capped to the selected companion
head. Source/runtime/assets matched before and after, and both runtimes retired
cleanly with closed ports. See the [expanded qualification evidence](../m35-generic-import/original-parity.md#drafter-source-alias-admission-repair)
for frozen identities, settings and counters. These two sampled-MTP behavior
controls do not establish performance, maximum-context or broad quality coverage.

## Qualification boundary

This evidence establishes six representative GGUF Qwen API tool/result
transactions, one plain native mixed-encoding Mia named-tool/result transaction,
two sampled native MTP tool/result transactions (`required` and `named`), and
one actual OpenCode 1.19.3 task on their recorded frozen identities. It does not
qualify every served model alias, each template grammar, all client variants,
maximum context, performance, sampled quality, or broad speculation coverage. In particular,
it does not establish full served-model tool coverage or make the open plan
item complete. Final integrated source checks remain separate from this diagnostic.
