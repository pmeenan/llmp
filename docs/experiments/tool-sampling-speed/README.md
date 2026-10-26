<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tool-call sampling: host cost a token and agent-style requests

The M3.5 review's B3, B4, B9 and B10 ([tool calls](../../tool-calls.md)):
a step restricts only what its constraint restricts (device greedy and
verify where nothing is restricted, lazy host choices where something is),
a full mask judges plain tokens once a state, one forced-span plan answers
for a whole forced text, forced openers spell the template's whitespace, and
every declared tool renders whatever `tool_choice` says. Coarse D-085
checks: no family slower beyond noise, greedy outputs unchanged.

**Host and build.** `spark-b` (GB10, driver 580.178.04), `spark-native
--locked`. Before: the tree at `e0aa811c`; after: the same tree with this
change. Both runtimes ran from copies of their own `llmp-runtime`, one model
alone, the v1 store `u2-v1-artifacts`.

## Per-token host cost of the sampler

`benchmarks/tool_mask_bench.cc` (`llmp_tool_mask_bench QWEN_TOKENIZER_JSON
GEMMA4_GGUF 10`) drives `chat::ToolSampling` on the real vocabularies
(Qwen3.8's `tokenizer.json`, 248,077 tokens; Gemma 4 26B's GGUF tokenizer,
262,144) to each state by accepting a fixed text (a forced opener by its own
stable spans), then times, over 10 repetitions after a warm one, the greedy
choice of a row whose argmax the constraint allows: the full mask and its
argmax (before: the only path), and the lazy choice (after: the argmax
checked first). `ValidateLogits` is included in both (about 0.17 ms). Two
tools are declared; `auto-serial` is auto with `parallel_tool_calls: false`.
Milliseconds a token:

| family | choice | state | allowed tokens | before | after, full mask | after, lazy |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.8 | auto | reasoning or text | all | 0.34 | 0.34 | 0.34 |
| Qwen3.8 | auto | in a call (name or value) | all but 2 terminals | 0.79 | 0.35 | 0.34 |
| Qwen3.8 | auto-serial | reasoning | all | 21.2 | 0.34 | 0.34 |
| Qwen3.8 | auto-serial | in a value | all but 2 | 22.3 | 0.35 | 0.34 |
| Qwen3.8 | required | reasoning (waiting) | all but 2 | 35.9 | 6.3 | 0.34 |
| Qwen3.8 | required | the call's name | 609 | 21.6 | 15.8 | 0.34 |
| Qwen3.8 | required | in a value | all but 2 | 25.4 | 5.8 | 0.34 |
| Qwen3.8 | named | reasoning (waiting) | all but 2 | 36.0 | 6.3 | 0.34 |
| Gemma 4 | auto | in a call | all but 2 | 0.84 | 0.37 | 0.36 |
| Gemma 4 | auto-serial | text, name or value | all (but 2) | 19.1–24.6 | 0.36–0.37 | 0.36 |
| Gemma 4 | required | the call's name | 70 | 23.2 | 16.7 | 0.36 |
| Gemma 4 | required or named | in a value | all but 2 | 28.9 | 6.5 | 0.36 |

When the argmax is refused (a terminal inside a call or before a required
call) the lazy choice costs a full mask (after: 0.69 ms unconstrained, 6–17 ms
constrained). Name positions keep probing every token (no plain verdict
there); they last a few tokens a call. The speculative judge's copy of the
cursor costs 0.07 µs, and 0.12 µs with a 4 KiB argument open, so it is kept.
The bench's named Qwen in-value state is not reached by its fixed text (both
builds): not measured.

## Agent-style requests

[`agent_e2e.py`](agent_e2e.py) starts the runtime on loopback (Qwen3.8 NVFP4
`02354af1…` with its MTP drafter `bb49b7da…`, context 16,384, 4 slots;
Gemma 4 26B-A4B `a1f1c523…`, context 8,192, 4 slots, no drafter), and sends
streamed Chat Completions at temperature 0, `max_tokens` 256, a short agent
system prompt and three declared tools (`get_weather`, `search`,
`read_file`): a text answer with tools present (`text/auto`) and the same
without tools (`text/none`, the reference), the agent's call turn
(`call/auto`) and its answer after a tool result (`answer/auto`). C1 sends
each three times, one at a time; C4 sends four different questions at once,
three times. The tables are the means of the warm repetitions 2 and 3 (the
first compiles plans and graphs); C4 is the wave's completion tokens over its
wall time. Runs were ordered after, before, before, after (Qwen first, then
Gemma), each warm.

Decode tokens a second, C1 (first streamed delta to the end):

| family | text/auto | text/none | call/auto | answer/auto |
| --- | --- | --- | --- | --- |
| Qwen3.8 before → after | 37.3 → 39.1 | 42.8 → 42.8 | 38.9 → 40.6 | 43.5 → 45.4 |
| Gemma 4 before → after | 55.9 → 56.9 | 57.8 → 57.6 | (one event) | (1 token) |

With tools present and a text answer Qwen still decodes 9% under the
no-tools reference (Gemma 1%); its tool prompt is 578 tokens against 161,
and the rest of the gap was not investigated.

Forced choices, C1, one request each (seconds a request; Qwen's decode
tokens a second in parentheses):

| family | call/required | answer/required | call/named | answer/named |
| --- | --- | --- | --- | --- |
| Qwen3.8 before → after | 4.34 → 2.01 (17.2 → 39.0) | 2.42 → 1.28 (20.6 → 46.4) | 4.63 → 1.91 (18.0 → 41.1) | 4.47 → 1.15 (19.1 → 53.9) |
| Gemma 4 before → after | 0.66 → 0.33 | 0.98 → 0.46 | 0.75 → 0.30 | 0.66 → 0.40 |

C4 wave tokens a second:

| family | text/auto | text/none | call/auto | answer/auto |
| --- | --- | --- | --- | --- |
| Qwen3.8 before → after | 55.3 → 57.0 | 61.6 → 62.6 | 43.9 → 42.7 | 40.9 → 42.5 |
| Gemma 4 before → after | 108.7 → 113.0 | 121.8 → 118.3 | 41.4 → 41.9 | 9.8 → 10.1 |

The no-tools path is unchanged code; its C4 differences (±3%) are run noise.

**Per-owner result kinds, not adopted.** With each owner of a joined wave
keeping its own result kind (`Batching::mixed_greedy`, a device token for an
unrestricted greedy owner beside rows for the others), the same after build
measured C4 call turns 8% (Gemma, 40.0 → 36.8) and 7% (Qwen, 41.6 → 38.6)
slower than one kind for the wave, and answer turns 10% slower for Qwen
(41.2 → 37.1), with text turns equal (one A/B pair of runs each, 3 waves).
The engine's own matched decode and verify waves
(`GemmaEngineNativeV1Gpu.MixedCausalAndMixedOutputs…` and
`MixedGreedyAndSampledSpeculativeVerify…`) measure equal (14.05 versus
13.99 ms a decode wave, 41.3 versus 40.3 ms a verify wave): the cost is in
short, changing compositions (each new per-owner combination is another plan
and graph). Serving keeps one kind a wave; the per-owner path stays a
diagnostic.

## Constrained steps on the device (review)

With every declared tool rendered (B10), `tool_choice: none` and the
restricting steps of auto, required and named replies first ran on the host
path at every step (the logits row copied back and judged there). The review
keeps the device's greedy choice and verify verdict wherever no forced span
is pending: the host checks the chosen IDs with `Allowed`, and only a choice
the constraint refuses is made on the host, from that row's logits, which the
runner reads back right after the step (`StepProgram::retained`,
`DecodingPolicy::readback`; [tool calls](../../tool-calls.md)). Same host,
store and harness, `--cells text:none,text:auto,text:off,call:auto,call:off,
call:required,answer:required` (`off`: tools present, `tool_choice: none`),
three repetitions, the warm two averaged; each build ran warm (a first run
after a build, weights cold, was discarded). Before: `e0aa811c`; B: the
builder's build above; R: with the readback.

Qwen3.8, C1 decode tokens a second and C4 wave tokens a second:

| cell | C1 before | C1 B | C1 R | C4 before | C4 B | C4 R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text/none (no tools) | 42.7 | 42.7 | 42.5 | 62.7 | 61.2 | 62.6 |
| text/auto | 37.4 | 38.8 | 38.6 | 55.9 | 57.0 | 57.2 |
| text/off | 42.7* | 37.4 | 38.5 | 63.9* | 63.2 | 66.6 |
| call/auto | 38.8 | 40.2 | 40.7 | 41.9 | 43.5 | 43.5 |
| call/off | 44.9* | 40.9 | 42.4 | 46.6* | 49.7 | 53.1 |
| call/required | 17.5 | 40.7 | 42.3 | 18.4 | 56.8 | 59.6 |
| answer/required | 20.7 | 46.6 | 48.2 | 19.8 | 45.5 | 47.7 |

Gemma 4 26B-A4B, C1 (text: decode tokens a second; calls: seconds a
request) and C4 wave tokens a second:

| cell | C1 before | C1 B | C1 R | C4 before | C4 B | C4 R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text/none (no tools) | 57.9 | 57.9 | 57.9 | 122.3 | 121.4 | 122.2 |
| text/auto | 55.6 | 57.1 | 57.2 | 108.2 | 114.2 | 116.3 |
| text/off | 57.9* | 54.7 | 57.2 | 122.6* | 116.3 | 126.2 |
| call/auto (s) | 0.328 | 0.327 | 0.316 | 41.7 | 41.8 | 43.8 |
| call/off (s) | * | 0.372 | 0.344 | 6.8* | 63.4 | 65.2 |
| call/required (s) | 0.652 | 0.320 | 0.310 | 22.3 | 67.5 | 71.6 |
| answer/required (s) | 0.951 | 0.460 | 0.445 | 27.1 | 69.9 | 74.4 |

\* Before B10, `none` rendered no tools: a different prompt and reply (Gemma's
call/off a single token), not comparable. `tool_choice: none` now decodes as
fast as auto with the same prompt (Qwen C1 37.4 → 38.5, Gemma 54.7 → 57.2),
and required replies gain 3–6%. The C4 waves include prefill: `off` follows
`auto` with the same rendered prompt (B10), so its prompt is reused
(`cached_tokens`), which `before` had only from `none`. Rows read back (the
engine report's `head_rows`): Qwen 15 in its C1 run (1,470 speculative
units) and 41 at C4 (5,843); Gemma 3 at C1 against 2,466 device choices and
12 at C4 against 9,859. Every C1 reply (21 a family) equals the builder's
build's.

## Greedy outputs

Every C1 auto and no-tools request (12 a family, both families) produced the
same reasoning, content, calls and finish before and after. Required choice
produced the same reasoning, content and calls with one more token (the
newline the template writes after `<tool_call>`, now forced). Named choice
renders all three tools now (prompt 435 → 570 tokens for Qwen, 191 → 299 for
Gemma), so its outputs legitimately differ; its calls name and argue the same
tool. C4 outputs differ between repetitions of the same build (joined-wave
numerics at near-ties), before as after, and are not compared.

## Limits

One pair of warm runs a family for the final tables (three earlier pairs
agree within the noise above; a run taken right after other work, its
weights cold, was 20–50% slower in both builds and is discarded). Sampled
decoding was not measured end to end; its lazy paths (top-k from the leading
allowed tokens, temperature-only rejection) are exact in distribution, not in
draws (`LlmTools.LazyChoicesMatchTheMaskedRow` checks their kept sets and
choices). DeepSeek (DSML) was not run end to end;
its spelling is checked against its renderer in `chat_template_test`.
