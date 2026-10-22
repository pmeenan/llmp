<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# M3.5 native tool-call diagnostics

These are bounded behavior diagnostics for native Qwen tool calls and one
actual OpenCode client task. They are representative evidence for the native
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
these diagnostics do not qualify NVFP4 weights. OpenCode was pinned at version
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

## Qualification boundary

This evidence establishes six representative Qwen API tool/result transactions
and one actual OpenCode 1.19.3 task on the frozen run identity. It does not
qualify every served model alias, each template grammar, all client variants,
maximum context, performance, sampled quality, or speculation. In particular,
it does not establish full served-model tool coverage or make the open plan
item complete. Final integrated source checks remain separate from this diagnostic.
