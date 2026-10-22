<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded recurrent reply-reuse diagnostic

The first current-source Qwen3.8 checks retain the model's own reply exactly
when its reasoning is resent. Omitting reasoning changes the rendered history;
the existing turn checkpoint still preserves the preceding user prefix. These
short diagnostics do not justify a new cache mechanism. The broader M3.5
shared-prefix and client qualification items remain open.

## Native observations

Both supervised exclusive jobs ran on `spark-b`, with no concurrent build or
GPU work. The model was the existing Qwen3.8 Flash Next GGUF v1 artifact
`f60d801a1695079ba4cd9c17d32967315bf77fc26580c1f8245b37f95d03d027`,
using its ordinary pinned Mia tokenizer/template assets. Geometry was context
4096, one slot, prefill chunk 512, F16 KV, plain greedy generation and maximum
512 output tokens. The questions were `What is 2 + 2? Answer briefly.` and
`What is 3 + 3? Answer briefly.` All observed completions stopped normally;
the answer content was `4`, then `6`.

| Path | First prompt / output tokens | Follow-up prompt tokens | Actual reused tokens |
| --- | ---: | ---: | ---: |
| Existing CLI, reasoning retained in history | 63 / 35 | 120 | 97 |
| Stateless Chat Completions, `reasoning_content` resent | 63 / 35 | 120 | 97 |
| Stateless Chat Completions, reasoning omitted | 63 / 35 | 89 | 58 |

The CLI report contains the actual prompt and generated IDs. All 98 IDs of the
first prompt plus reply are an exact prefix of the next rendered/tokenized
prompt. Its final selected stop token was unprocessed, so 97 reused tokens is
complete reuse of the processed prefix. Both CLI prefills used two chunks.
This cell shows no own-reply re-tokenization or re-rendering miss; it does not
prove that arbitrary generated strings round-trip to identical token IDs.

Each HTTP cell used a fresh service. Their first returned message, including
reasoning, was identical. The follow-up differed only in whether that reasoning
was resent. The service accepted the compatibility field `reasoning_content`;
its response used `reasoning`. Omitting reasoning deliberately produces a
different prompt, so the lower reuse count alone is not evidence of incorrect
caching. The existing checkpoint recovered 58 tokens and the follow-up still
completed correctly. HTTP usage supplies these reuse counts; HTTP prefill-launch
counts and the omitted-history token divergence were not instrumented here.

Qwen3.8's current renderer preserves thinking by default. Locals' observation
about a Qwen3.5 template dropping an empty think block is not assumed to apply
to this template. No renderer, tokenizer, checkpoint or default changed.

## Actual OpenCode protocol capture

The official immutable Linux ARM64 OpenCode v1.19.3 client was provisioned from
[release v1.19.3](https://github.com/anomalyco/opencode/releases/tag/v1.19.3),
source `e3c4eb242585dcd77be1a867391b7c25fd47ce52`, under its
[MIT license](https://github.com/anomalyco/opencode/blob/e3c4eb242585dcd77be1a867391b7c25fd47ce52/LICENSE).
This is an external qualification tool, not a runtime dependency or vendored
implementation. Its custom provider used `@ai-sdk/openai-compatible` and a
loopback `/v1` URL, with one permitted read tool, isolated XDG directories and
automatic updates/model fetching disabled.

Two supervised shared captures used a local mock server and a private marker
file. Both completed a real client tool-call/read/result/final-answer exchange.
The client sent streamed `/v1/chat/completions`, `tool_choice: auto`, modern
function definitions, `stream_options.include_usage: true`, assistant
`tool_calls` and matching `tool_call_id` result messages. A separate title
request had no tools. In the second capture, server `delta.reasoning` was
preserved by the client as `reasoning_content` on the next assistant history
message. The marker file was actually read and its contents returned.

The two actual system strings were 8,867 and 8,877 UTF-8 bytes, sharing the first
8,744 bytes; the first difference was the controlled working-directory path.
Their tool definitions were identical. This is one restricted client profile
on one day in two private projects. No native rendering/token count, cache hit,
latency saving, date-instability measurement or native model/tool qualification
is claimed by this mock capture. Volatile-block reordering remains off.

## Actual native OpenCode continuation

A seeded coding task adds one actual native OpenCode observation to the earlier
CLI, stateless HTTP and mock-client checks. The run used context 16,384,
prefill chunks of 512, one F16 KV slot and speculation off. Its weights were the
`unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee` UD-IQ3_XXS artifact
`f60d801a1695079ba4cd9c17d32967315bf77fc26580c1f8245b37f95d03d027`.
`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6` supplied tokenizer and
chat-template assets only; no NVFP4 weight qualification follows. OpenCode
v1.19.3, the runtime, source manifest and asset pins are recorded in the
[native tool-task report](../m35-tools/RESULTS.md#actual-opencode-task).

The runtime also logged a separate request with 652 prompt, 155 generated and
0 cached tokens. No HTTP request bodies were captured, so its purpose is
unknown and it is not classified as part of the task flow. The following four
request records accompany the retained client event stream, which shows four
actual tool calls: two reads, an edit and a bash test. Across those task turns,
the next prompt's reported cached count equals the previous prompt plus
completion count minus one:

| Previous prompt p | Generated g | Processed prefix p+g−1 | Next prompt | Cached | New prompt tokens |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,755 | 99 | 4,853 | 5,180 | 4,853 | 327 |
| 5,180 | 120 | 5,299 | 5,319 | 5,299 | 20 |
| 5,319 | 78 | 5,396 | 5,429 | 5,396 | 33 |

The count relationship matches the source contract: `prompt_tokens` is the
whole rendered prompt, `completion_tokens` counts generated tokens including a
stop token, and `cached_tokens` counts prompt tokens already held; a live-state
continuation reuses the exact common token prefix ([API completion counts](../../../src/runtime/api_server.h),
[ReusePrompt](../../../src/runtime/llm.cc)). Generation keeps its final selected
token as the next anchor, not yet processed into state
([generation boundary](../../../src/runtime/llm_generation.cc)); therefore
`p + g − 1` is the processed prefix at these completed turn boundaries. The
three observed cached counts match it exactly. This is evidence of live
within-conversation reuse in one OpenCode task. It does not measure prefill
launches or speed, qualify long prompts or other models and clients, or
establish a global prompt cache. No cache or recurrent-state change is selected
from this observation.

## Provenance and limits

The frozen native executable remained in its original build directory:
`build/spark-native/src/runtime/m35-wave-t-runtime`, SHA-256
`d396ff1ae933aa58c8ee2ea534109889ce6eda3097cb08c917f4f503fc5de721`.
Its built source is base `d53347d9` plus the interpreter foundation, recorded by
patch SHA-256 `ad679aee92f7fae942ca4952bfc1d4d435b19c21a9737ea156fd69eaff807f0c`.
All 2,796 tracked files matched the authenticated built-source manifest,
SHA-256 `0537a81a1bf3304a76fe7b15b3cd5c9c7b259fd426cc97a7500cb58745a36afd`.
The CLI launch's `source_patch_sha256` records `79a1f65e…`, the later final
patch containing a plan-status edit; the built-source patch above excludes
that uncompiled documentation edit. Production source bytes are identical.
The remote checkout's old Git label is not the compiled-source identity.

| Receipt | SHA-256 |
| --- | --- |
| Native CLI full report | `c4e2cebe48724ae4e4ced1d62816b4a313db41110cf22a260d719922b10b2b79` |
| Native HTTP aggregate | `8a34b39d49fc96b6944813950788b3ca99cf5fdbfccdb5cdea89eca45d282490` |
| CLI collection helper | `4e8f1aafb9478414754530756a6f55627dae138965df0fe877b92a155226d9da` |
| HTTP collection helper | `e91efb9d96a1e913d5b32d07e0f3d5a4395210b5896384cd3f4a82dbff8f3000` |
| OpenCode archive | `7900818a6a98881edbbadab4b7525940eff3b9750040c200481f9f56cacee405` |
| OpenCode executable | `e5309bc8df8deac0e4050fb4a65989839522703cf86f8a4561c2694f5157b8ca` |

Jobs `m35-prefix-current-cli-r1`, `m35-prefix-stateless-http-r1`,
`m35-opencode-client-provision`, `m35-opencode-client-mock-r1` and
`m35-opencode-client-reasoning-mock-r1` returned zero. Both HTTP services retired
with TERM/zero and closed their listener. Root observed the final Spark
GPU/busy check as free in retained parent tool output; it was not separately
redirected to a contemporaneous host receipt. Raw private requests, outputs, launch/source/metadata receipts, scripts
and supervisor logs remain outside Git under the orchestration scratchpad and
on Spark B until milestone cleanup. There is no throughput, quality, long
context, cross-family or broad prefix-cache qualification in these observations.
