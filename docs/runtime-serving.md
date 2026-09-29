<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Serving in the runtime (M3)

How `jitllm-runtime` serves M3's models, since M3's swap path moved out of
the benchmark harnesses (D-096). The swap path itself, its measurements and
its checks are in [swap.md](experiments/fast-swap/swap.md); the decisions it
builds on are D-048 (tasks and lanes), D-081 (the landing zone), D-086 and
D-093 (jobs under a request's lease), D-090 (decode graphs at pinned
places) and D-094 (the runtime wake).

## Where the code lives

| Layer ([architecture](architecture.md#layers-and-dependency-rules)) | Module | What moved there |
| --- | --- | --- |
| Resource core | `scheduler/programs.h` | The task programs a driver posts: run, call, evict, a full swap (`SwapProgram`), a request's lease (`RequestProgram`), BP-S3's acquisition. Vendor-free; the fake backend's tests check them |
| Engine (new, above the kernels) | `engine/` | The paged node (`paged_node.h`: device 0's providers, streams, io_uring, one catalog domain, the landing zone, the scheduler and its lanes on their threads, the shared workspace, requests), weights as extents (`paged_weights.h`), Qwen3.8's n-gram rows (`ple_rows.h`), each M3 model's chunk planning (`dsv4_plan.h`, `qwen38_plan.h`) and runner (`dsv4_runner.h`, `qwen38_runner.h`, `qwen_image_runner.h`). CUDA builds only (it links the kernels); it reaches the device through the providers and the device runtime (`providers/device_runtime.h`), never the CUDA runtime itself ([portability.md](portability.md)) |
| Services | `config` | The models a node serves: `[models.<name>]` (below) |
| Programs | `runtime` | `commands.h` (the serving commands' arguments, vendor-free) and `serving.h` (the configured models on one node, the full swap, each model's turns, the commands; CUDA builds) |

The harnesses (`benchmarks/`) drive the same engine: `engine_names.h`
gives the engine's runners their harness names, and `tests/support`'s
`paged_node.h` and `paged_programs.h` do the same for the node and the
programs (with `LaunchOnlyProgram`, which only tests post). Nothing in
`tests/support` is linked into a shipped binary, as before.

What productizing them changed: nothing in them depends on test support;
the programs are the scheduler's; every thread the node starts installs
its own signal stack first (D-074's crash policy); and failures are values
up to the command, which exits 1 with them in the log. Lifetimes stay as
the harnesses proved them (D-048): the node's driver posts a program to a
bounded control queue (waiting while it is full) and never returns while
the program, or a job it queued, may still refer to its frame; past ten
minutes it cancels the request and waits for the drain, and if even that
does not come it aborts the process rather than free memory a job may
still use. A swap or eviction asked for between a request's steps ends
that request first; teardown ends every request, fences each model's
stream, evicts every managed extent and checks every backing released.

## Configuration

A node names the models it serves in its configuration (D-073's document,
`schema_version = 2`; the keys are new and compatible):

```toml
[models.deepseek]
artifact = "8a355bfb…"   # an installed artifact's ID, under storage.installed
drafter = "dd2d3f9c…"    # optional: its speculative drafter (DSpark, MTP)
# speculation = true     # the default when there is a drafter
# context = 8704         # tokens of conversation state, 512 to 262,144
# prefill_chunk = 2048   # rows of a prefill chunk; default by model (below)
# prefill_floor_tok_s = 100  # tokens a second: the floors the chat route figures
# decode_floor_tok_s = 5      #   a request's work at (progress and deadlines, below)

[models."qwen3.8"]
artifact = "c4fb47a9…"
drafter = "056a750e…"
tokenizer = "/path/to/tokenizer.json"          # when the artifact keeps none
chat_template = "/path/to/chat_template.jinja"  # likewise

[models.image]
composition = "eca21baa…"  # a pipeline (D-089)
```

A model names exactly one artifact or composition; the artifact-only keys
(drafter, speculation, context, prefill chunk, the floors, tokenizer, chat
template) are refused on a composition, an artifact serves one model, and
a node names at most 16. The runner follows the artifact's architecture
(`deepseek4`, `qwen4exp`) or the composition's (Qwen-Image); another is
refused at registration. Every
artifact is opened under the store's trust rules (only root and the
runtime's user may change it), and the tokenizer and template files the
configuration names are read under the configuration's (D-073). A chat
template is rendered only if a native renderer has its hash (D-067).

## Registration and the swap

The serving commands and the service with models run the runtime's
startup steps (anchor, configuration, process lock, storage roles,
platform), then register every configured
model on one paged node: its artifacts opened, its runner set up on its own
stream, its tokenizer and renderer found, the shared workspace mapped at the
largest model's need, and the scheduler started with the budget of
everything fixed (the zone, each model's own memory, the workspace, the
staging) plus the largest model's weights, which must fit what the host has
available with the largest chunk inputs a model builds on the host (one
model runs at a time; bounded by the staging they are sized for) and a
6 GiB margin for what else the catalog does not count (decode graphs, the driver's and
cuBLAS's own memory; [long-context](experiments/long-context/README.md#memory-and-the-guards-margin)).
No weights are paged yet.

One model is resident at a time (M3's full swap). Activating another
(`Server::Activate`) is one `SwapProgram`: the resident LLM's conversation
state written back through the zone to an unnamed spill file in
`storage.spill` if it holds a conversation (a model with none keeps its
state resident), its weights evicted with their backing parked for the
incoming loads (D-033's handoff), and the incoming model's whole closure
paged in, its state restored if it had been spilled. Then the model's own
checks (DeepSeek's hash-routing tables, Qwen3.8's n-gram hash) and its
places checked still pinned (D-090). The backing no load took is released
after the first output, off the swap's path. Each part is timed.

## Turns

An LLM holds one conversation: the tokens its state has seen. A turn's
tokens (the conversation rendered by the model's chat template) extend it
when they start with it, and only the rest is prefilled; otherwise the
state is cleared first. A turn is one request (D-093): the model's closure
leased once, the prefill chunks ([below](#prefill-chunks-and-cancellation))
and every decode step jobs under it. Decoding is greedy and, where the
model has a drafter, speculative by default (D-092's batched verify:
DeepSeek's DSpark draft and verify as one job, Qwen3.8's MTP draft then
verify), each step accepting the drafts the target agrees with; `--plain` decodes one token a step. The chat route may
sample instead (a `temperature` above 0): seeded, each token drawn at its
position in the conversation (execution/sampling.h), and when speculating
each draft accepted by speculative sampling (`VerifyDraft`), so the
tokens are distributed as plain sampling's; a seed repeats a reply.
Qwen3.8 greedy speculation chooses depth two or three using a moving
acceptance average and a measured relative step cost of 1.16. It tries
four complete steps at each depth, then probes the other depth for four
steps after 32 observations; a change needs a predicted 3% gain. Output-
or context-truncated verifies do not train the policy. Its schedule is
saved with conversation state, independent of timing. Seeded sampling
keeps depth two, as does a configured prefill chunk of only three rows.
Generation stops at the template's end-of-turn tokens, the token limit, or
when the route ends it (a stop string, the client gone, the backend
stalled, a non-streaming request's deadline, the runtime stopping;
[progress and deadlines](#progress-and-deadlines)), always between
steps; the same ends a prefill between its chunks (below). A model whose
chat template has no renderer is refused at registration, naming its
hash. A job that failed after it may
have run leaves the conversation unknown, so the next turn clears the
state first.

The image pipeline generates the prompt and initial latents it registered
with: this slice's image runner (being reworked by the image-speed slice)
takes them at setup, so a process serves one image prompt, and the latents
come from a file (the reference's for its seed; a native seeded generator
is still to come).

## Prefill chunks and cancellation

A turn's prefill runs in chunks (`runtime/prefill.h`). The chunk is the
model's `prefill_chunk` if configured (1 to 262,144 rows), else the
runtime's default for the model, and in either case at most what the
model's state layout admits at its context (DeepSeek: the window cache's
cells less its 128-position window; Qwen3.8: 8,192 rows, and its F32
[context, rows] tensors under 2^31 bytes, RE-037: 4,095 rows at 131,072,
2,047 at 262,144) and below the context, in whole 8-row tiles. So every
context the configuration accepts has a chunk: at the minimum, 512,
DeepSeek's chunk is 384 rows and Qwen3.8's 504. A chunk of 1,024 rows or
more runs in whole tiles and its few remaining rows as a chunk of their
own, since GGML's attention reads the mask in whole 8-row tiles from
1,024 rows on (RE-036). Registration logs each model's chunk.

**The defaults** come from the runtime's own prefill (`jitllm-runtime
chat`, speculative, so each chunk also feeds the drafter; one model
configured, context 8,704; `spark`, GB10, 2026-09-29; the best of two
turns each, from a cleared state):

| Model, chunk rows | 8K-token prompt | ~2K-token prompt | Longest chunk (8K) | Fixed memory (workspace) | Peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek, 256 | 220 tok/s | 224 | 1.29 s | 0.76 GiB (0.24) | 105.6 GiB |
| DeepSeek, 512 | 312 | 318 | 1.77 s | 1.01 (0.47) | 105.4 |
| DeepSeek, 1,024 | 391 | 392 | 2.78 s | 1.52 (0.94) | 106.2 |
| **DeepSeek, 2,048** | **463** | **458** | **4.66 s** | **2.54 (1.87)** | **107.3** |
| DeepSeek, 4,096 | 459–481 | 505 | 8.98 s | 4.58 (3.74) | 109.2 |
| DeepSeek, 8,192 | 461 | 503 | 17.5 s | 8.65 (7.47) | 113.6 |
| Qwen3.8, 512 | 1,301 | 1,285 | 0.41 s | 0.72 (0.18) | 75.6 |
| Qwen3.8, 1,024 | 1,711 | 1,635 | 0.61 s | 0.98 (0.35) | 75.9 |
| Qwen3.8, 2,048 | 2,065 | 1,923 | 0.99 s | 1.50 (0.70) | 76.3 |
| **Qwen3.8, 4,096** | **2,320** | **2,199** | **1.72 s** | **2.56 (1.39)** | **77.5** |
| Qwen3.8, 8,192 | 2,426 | 2,214 | 3.18 s | 4.68 (2.77) | 79.7 |

(8,088 and 2,164 prompt tokens for DeepSeek, 8,553 and 2,362 for
Qwen3.8; peak is the host's `MemAvailable` drop with the weights
resident.) The policy: from 512 rows, double the chunk while that gains
10% or more at 8K tokens and its longest chunk stays within 5 s, since a
chunk is how soon a prefill notices a cancellation. That gives DeepSeek
2,048 rows (4,096 gains 0–4% for twice the wait and memory) and Qwen3.8
4,096 (8,192 gains 5%). Against the fixed 512 rows before, an 8K-token
prompt prefills 1.48× faster on DeepSeek and 1.78× on Qwen3.8, for
1.4 GiB more shared workspace (sized for the larger model's need,
DeepSeek's). Per-chunk time grows with the position, so at contexts past
8,704 (not verified) a chunk takes longer; `prefill_chunk` sets another.

The prefill's result depends a little on the chunk (the fast plans are
not bit-exact across chunk shapes: DeepSeek's top logit after the 8K
prompt was 21.5–22.9 across the sizes). Its top token agreed in every
run, and greedy tokens agreed except at near-ties. The one seen early,
DeepSeek's third token after the 8K prompt, is a near-tie at every size
and in the oracle: llama.cpp (the pinned image, fusion off, 512-row
micro-batches) prefers token 3287 to 304 by 0.31; jitLLM's margins run
from 3287 by 0.38 to 304 by 1.16 across 512, 2,048 and 4,096 rows,
speculative or plain. The largest move from the oracle's margin is 1.47,
inside the fast plan's near-tie bound of about 2.5
([dsv4-decode](experiments/dsv4-decode/README.md#the-bound-going-forward)).
DeepSeek's prefill past 4,096 positions did not repeat bit for bit from
run to run at some chunk sizes (on this prompt, 4,096 and 512 rows, not
2,048 in 7 runs): RE-031's tie-breaking in GGML's top-k, which its
indexer used, not the chunking. The fast plan's own indexer
(long-context phase 2) breaks ties by row and repeats; the reference
mode (`--exact on`) still does not.

**Cancellation.** Whatever ends a chat request (the client gone, the
backend stalled or a non-streaming request's deadline passed
([below](#progress-and-deadlines)), the runtime stopping on SIGTERM or
SIGINT) is noticed before each prefill chunk and each generation step,
and after the swap that made its model resident (a swap is one program,
about 10 s, and is not interrupted). A stopped prefill is an ordinary end, not a
node failure, and the service goes on serving: the conversation's state
holds exactly the chunks that ran, a prefix of the turn's tokens, so a
retry of the request continues from it (its chunks at the same places an
uninterrupted prefill's would be) and any other request clears it as
usual; a swap spills and restores it like any conversation. The log says
how many of the prompt's tokens the state holds (counts only, D-014). A
cancellation waits at most for the chunk under way.

**Measured** (`spark`, driver
580.178.04, the service with both models on loopback at their default
chunks and context, greedy streamed requests of an ~8K-token prompt; two
runs, the same to 0.03 s, and a review's third of the disconnects):

| Case | Stopped after | From the event to the prefill's stop | Then |
| --- | --- | ---: | --- |
| DeepSeek, the client closes 3.0 s into the prefill | 1 chunk (2,048 of 8,107 tokens) | 1.04–1.57 s | 499 logged; the service serves on |
| Qwen3.8, the client closes 1.0 s in | 1 chunk (4,096 of 8,574) | 0.66 s | likewise |
| Qwen3.8, SIGTERM 1.0 s in | 1 chunk (4,096 of 8,370) | 0.65 s | in-stream 503 at 0.65 s; exit 0 after 3.7 s |
| DeepSeek, SIGTERM 3.0 s in | 1 chunk (2,048 of 7,991) | 1.07–1.08 s | in-stream 503 at 1.08 s; exit 0 after 5.0 s |
| Qwen3.8, SIGTERM during its swap in (first run) | before any chunk | — | the swap ran out: 503 after 6.7 s, exit 0 after 10.2 s |

Each disconnected request, sent again, continued from the state's
chunks (2,048 and 4,096 tokens cached; DeepSeek's partial state spilled
by a swap to Qwen3.8 and restored by the swap back), and its greedy reply
equalled the same request's from an empty state. In the review's run both
partial states went through swaps (each spilled by the swap to the other
model and restored by the swap back) and were continued by a different
request that shares their prefix (the prompt with a question after it):
its reply equalled that request's from an empty state for both models.
Shutdown, teardown
included, stays far inside `jitllm.service`'s 90 s stop allowance (the
longest chunk, 4.7 s, plus a swap of about 10 s and the teardown). With
these chunks, the swap table's DeepSeek ↔ Qwen3.8 pair (`swap-table
--pairs deepseek:qwen3.8`, both models configured: 4.23 GiB fixed, the
workspace 1.87) was exact in all six swaps, every one under ~10 s (7.87–8.92
s; peak 109.3 GiB, against 108.0–108.4 with 512-row chunks in
[swap](experiments/fast-swap/swap.md#through-jitllm-runtime-d-096), on
`spark-b`).

## The commands

    jitllm-runtime [--config FILE] [--anchor PATH] chat [--max-tokens N]
        [--ignore-stop] [--fresh] [SERVING] --turn MODEL TEXT...
    jitllm-runtime [--config FILE] [--anchor PATH] swap-table [--pairs A:B,...]
        [--context-text FILE] [--context-tokens N] [--continue N] [--cycles N]
        [--zero-context on|off] [--handoff on|off] [--short-prompt TEXT]
        [--image-expect SHA256] [SERVING]
    SERVING: [--plain] [--image-prompt TEXT] [--image-noise FILE] [--report FILE]

Both run in the runtime's own process, holding its process lock (so never
beside the service), and open no listener (D-014); the service serves the
chat route (below). `chat` sends each turn to its model in order, swapping
as needed, and prints the reply, the swap's parts, the first token's latency
from the request, the prefill and decode speeds and speculation's
acceptance. `swap-table` is M3's swap table (plan.md) in one process: every
ordered pair of the registered models (or those named), A→B→A, first use
(the incoming model's plans and graphs dropped) and prepared, with 8K and 0
tokens of A's context, each part timed and the endpoints as the table
specifies; A's restored state must hash as it left and its continuation
equal, token and logit, the same state's unswapped continuation; B's first
output must repeat; a prepared return must replay graphs captured before
the swap; an image A's regenerated pixels must equal its control's. Both
write every number to `--report` as JSON. Run by hand, a command stops on
SIGINT or SIGTERM at once; the kernel frees its memory and spill files.

## The chat route

With models configured, the service (no command) registers them as the
commands do, then listens where `[client]` says (D-097, as the owner
amended it on 2026-09-28) and reports readiness. Without models it
starts, checks and waits as before; a CPU-only build refuses to start with
models configured.

```toml
[client]
bind = ["loopback", "tailscale"]  # the default; a string or a list of 1 to 16
# bind = ["loopback", "192.168.1.5:9000", "[::]"]   # any address, with a port or not
port = 8114                       # the port of "loopback", "tailscale" and a bare address
max_connections = 1024            # open connections, idle ones included: 1 to 65,536
max_queued = 64                   # requests waiting behind the running one: 1 to 1,024
stall_seconds = 120               # no progress for this long fails a request: 30 to 3,600
deadline_cap_seconds = 14400      # a non-streaming request's deadline at most: 60 to 86,400
```

**Where it listens.** `"loopback"` is 127.0.0.1 and, where the node has
it, ::1. `"tailscale"` is the node's tailnet addresses, found at startup
from its interfaces (getifaddrs; the tailscale CLI is not run): an
interface named `tailscale*` or holding an address in Tailscale's
fd7a:115c:a1e0::/48, and its addresses in that range or 100.64.0.0/10.
The same /10 on any other interface is a carrier's shared address space
and is not served as the tailnet. Without Tailscale the route serves
loopback (and any explicit address) and logs so; a restart picks up a
tailnet that came up later. An explicit address may be anything, a
wildcard (`0.0.0.0`, `[::]`, which cover their family's other entries on
the same port) included. No binding needs authentication, as with
Ollama, llama-server, SGLang and LM Studio (D-014's owner note): the
start log names every address and port, the host names accepted, and,
as information, each listener beyond loopback and the tailnet as served
without authentication. An optional API key is M5's.

    POST /v1/chat/completions      one conversation turn, JSON or SSE
    GET  /v1/models                the configured models (the image among them)
    GET  /v1/models/{id}
    GET  /jitllm/v1/ignored-fields the unknown fields seen (loopback peers only)

It is a strict subset of client-api-baseline.md's Chat Completions profile,
not M5's front door. A request is stateless, as OpenAI's are: the whole
conversation is rendered by the model's template, and the state's tokens
are reused when they are a prefix of it (as `chat` does, including its
limit for thinking models). The model named is made resident first (a full
swap when another is), and the turn is one request under one lease
(D-093), greedy or sampled, speculative where the model has a drafter.

**Fields.** Honored: `model` (a configured name; unknown is a 404, the
image pipeline a 400 `model_not_supported`), `messages` (system;
developer, read as system; user; assistant, whose `reasoning` or
`reasoning_content` goes back to the template; content as a string or text
parts; the last message the user's), `max_tokens` or
`max_completion_tokens` (default: the rest of the context), `temperature`
(default 1, OpenAI's; 0 is greedy), `top_p`, `top_k` (0 or −1: off),
`min_p`, `seed` (default: random), `stop` (matched in the answer, not in
the reasoning), `stream`, `stream_options.include_usage`. Known fields
that would change the answer are accepted only at their "off" value, and
otherwise refused with a 400 naming them: `n` 1, `presence_penalty` and
`frequency_penalty` 0, `repetition_penalty` 1, `logprobs` false,
`top_logprobs` 0, `tools` and `functions` empty, `tool_choice` and
`function_call` "none" or "auto", `response_format` text, `logit_bias`
empty, `modalities` ["text"], `audio` null, `store` false; OpenRouter's
`transforms` and `plugins` are refused whatever their value (D-046).
Known metadata is ignored: `user`, `safety_identifier`,
`prompt_cache_key`, `metadata`, `service_tier`, `parallel_tool_calls`, a
message's `name`, a text part's `cache_control`, and an assistant's null
`refusal`, `audio` and `function_call`, empty `tool_calls` and
`annotations` (echoes of a response). **Any other field is ignored with a
200**, at the top, in a message, a text part or `stream_options`; its
name, never its value, is counted (`x`, `messages[].x`,
`messages[].content[].x`, `stream_options.x`; at most 64 names a request,
cut to 64 bytes) in a table of at most 256 names with each one's count
and first and last time seen, and logged once, when first seen. `GET
/jitllm/v1/ignored-fields` returns the table
(`{"object":"list","data":[{"name","count","first_seen","last_seen"}],"unrecorded":N}`,
Unix seconds; `unrecorded` counts names past the 256th) to loopback peers
only (a 404 to others); M5's management listener takes it over.

**Output.** The reasoning a thinking template opens (its prompt ends
inside `<think>`) goes out as `reasoning`, up to the `</think>` token; the
answer, less its leading whitespace, as `content`. `finish_reason` is
`stop` at the template's stop token or a stop string, else `length`.
`usage` counts the whole rendered prompt, the generated tokens (a stop
token included) and, as `prompt_tokens_details.cached_tokens`, the
prompt's tokens the state already held. A non-streaming response is one
JSON body once the outcome is known. Streaming sends the headers and the
role chunk once the request is admitted (its tokens counted against the
context, before the swap), a chunk per step's text, the finish chunk, the
usage chunk if asked for, and `data: [DONE]`; a failure after the headers
is a `data: {"error":...}` event, and the stream ends without `[DONE]`.

**Keepalives** (D-045). A streamed request that has waited 15 s in the
queue is admitted then: its headers and role chunk go out, so the client
sees it accepted. From its headers on, a stream that has sent nothing for
15 s gets a `: keepalive` comment line, queued, swapping or prefilling,
well inside the named clients' 300 s stream-idle bounds
(client-api-baseline.md). A stream has no fixed queue wait: it waits its
turn as long as the backend makes progress. A stream admitted early can
then only fail in the stream: the model's refusal (the context exceeded)
is an in-stream `invalid_request_error`, and the backend stalling an
in-stream `server_error`, both without `[DONE]`; a stream that starts
within 15 s gets the refusal as a 400 before any header. A non-streaming
request waits silently until its turn or the queue's wait (120 s, then a
429).

**Intake bounds** (client-api-baseline.md#shared-correctness-and-limits),
checked before any model work (runtime/api.h):

| Bound | Value | Over it | Why this value |
| --- | --- | --- | --- |
| Request line and headers | 16 KiB, 64 headers | 413 | Clients send a few hundred bytes; a bounded buffer per connection |
| Target | 2 KiB | 414 | Routes and a short query |
| Body | 4 MiB, by Content-Length only | 413 before it is read (chunked: 501; none on a POST: 411) | A 262,144-token context at ~4 bytes a token with JSON escaping, and a bounded buffer |
| Bodies arriving at once | 64 MiB | 503, `Retry-After: 10` | Sixteen largest bodies; what 1,024 connections could otherwise hold (4 GiB) comes out of the Spark's one memory budget |
| JSON | depth 16, 262,144 values | 400 | The request's own nesting is 5 deep; the parser's allocation stays under ~4 MiB of nodes |
| Messages | 1 to 1,024 | 400 | Several times any conversation that fits the default 8,704-token context |
| A message's text | 1 MiB, from at most 64 parts | 400 | Bounded by the body anyway; stops one field taking it all |
| `model` | 1 to 64 bytes | 400 | A configured name's limit (D-096) |
| `max_tokens` | 1 to 262,144 at parse; prompt + it ≤ the model's usable context | 400 `context_length_exceeded` | The context the model's state holds (`context`, less Qwen3.8's MTP draft rows when it speculates) |
| Prompt | under the usable context | 400 `context_length_exceeded` | As above; counted by the model's own tokenizer and template |
| `temperature`, `top_p`, `top_k`, `min_p` | [0, 2], (0, 1] (`top_p` not rounding to 0 as a float), −1 to 2³¹−1, [0, 1] | 400 | OpenAI's ranges and vLLM's for `top_k` and `min_p`; sampling.h's, which takes floats |
| `seed` | a 64-bit signed integer | 400 | OpenAI's type |
| `stop` | at most 4 strings of 1 to 128 bytes | 400 | OpenAI's count; the held-back text stays short |
| Unknown fields | 64 names a request, 64 bytes a name; 256 names kept | ignored | The table stays small whatever a client sends |
| Head, body arrival | 10 s, 30 s from the request's first byte | 408, then the connection closes | A local client sends at once; a stalled one holds only its own connection |
| Idle connection | 60 s between requests, told to the client (`Keep-Alive: timeout=60`) | closed | An idle connection costs a descriptor and a small buffer (at most 16 KiB each way: a larger one, a body's or a response's, is freed once its request is done); a minute spans a client's pauses between turns |
| Connections | 1,024 (`[client] max_connections`) | the oldest idle one is closed for the new one; with none idle, 503 | Agents and their subagents keep pools; each is a descriptor, and the open-file limit is raised to fit |
| Output not taken | 30 s without progress, or 1 MiB of a stream | the connection is dropped; the generation ends at its next step | A reader that stops reading cannot grow a buffer |
| Queue | 64 waiting behind the running request (`[client] max_queued`); a non-streaming one 120 s, a stream as long as the backend makes progress | 429, `Retry-After: 10`, `x-should-retry: true`; 503 with the same headers (in-stream once a stream has started) when the backend stalls | One user; a subagent's request waits for the main one instead of failing, a stream (held by keepalives) for as long as a long request ahead of it takes |
| A request | No fixed deadline: 120 s without progress (`[client] stall_seconds`), each unit of work allowed its expected time; a non-streaming one also its work at the model's floors three times over, at most 4 hours (`deadline_cap_seconds`) | 504 (in-stream error when streaming); after a stall, 503 to every request until the backend moves | A long prefill at depth is healthy and a stuck backend is not ([progress and deadlines](#progress-and-deadlines)) |

**Guards and errors.** No credential (D-014 and its owner note); an
`Authorization` header is ignored. The `Host` must name the node as it
listens, and a second `Host` is a 400: loopback addresses and `localhost`
always; the node's host name; with the tailnet, its addresses, its
MagicDNS name (the name the node's resolver gives a tailnet address,
under ts.net) and that name's first label; an explicit address and the
name its resolver gives it (for an address in Tailscale's ranges, only a
name under ts.net); with a wildcard, every non-link-local address of its
family and their names (at most 16 lookups, within 5 s together at
startup: a resolver that does not answer costs no more, and its names
are not accepted). An `Origin` must name the same, on a listening port,
and a cross-site or same-site `Sec-Fetch-Site` is refused (403): D-064's
browser guards, without M5's CORS. A JSON route needs `Content-Type: application/json`
(415). Errors are OpenAI's `{"error": {message, type, param, code}}`. A
client that disconnects, a non-streaming request's deadline (504), the
backend stalling (504, [below](#progress-and-deadlines)) or the runtime
stopping (SIGTERM or SIGINT, 503) ends the request at its next prefill
chunk or generation step, and after a swap before any model work (a
stall answers the client at once, without waiting for that step); the
state keeps
what it processed ([cancellation](#prefill-chunks-and-cancellation)), and
the service goes on. A client that shuts only its sending side after a whole
request (a half-close) has not disconnected: its response is finished and
the connection then closes. The two look alike until something is sent,
which a closed socket answers with a reset, ending the generation as a
disconnect; so the route sends something at once. A stream gets what any
client parses: its start (headers and role chunk, as after `keepalive`
in the queue; a later refusal is then an in-stream error) or a
`: keepalive` comment. A non-streaming response has nothing to send
before its head but an interim 1xx, so on HTTP/1.1 it gets `HTTP/1.1 102
Processing`; on HTTP/1.0 it runs to its end. No 1xx is safe for every
client (checked on `spark-b`, 2026-09-29, a 1xx before a 200 on a live
connection): httpx and the OpenAI Python SDK, aiohttp, Node's `http`,
Go's `net/http` and curl skip 100, 102 and 103; Node's `fetch` (undici)
and the OpenAI Node SDK fail on an unasked 100 and skip 102 and 103;
Python's `http.client` (and so `requests` and `urllib3`) skips only 100
and takes a 102 or 103 as the final response. The 102 only reaches a
client that half-closed and is still reading, which none of those
libraries does, or one that has gone; a client that half-closes should
parse 1xx (RFC 9110, section 15.2). A half-close before the request is
whole is a disconnect. A failure of the node itself (a swap or a job
that failed) ends the request with a 500 or 503 and stops the service
with status 1.

**Connections.** HTTP/1.1 connections persist: a response carries
`Connection: keep-alive` unless the client asked to close, an HTTP/1.0
client did not ask to keep it, the request was refused before it was
whole, or the runtime is stopping; a stream's body is chunked (on
HTTP/1.0, it ends with the connection). A request sent before the
previous response ended (pipelining) is not read: that response carries
`Connection: close` and the connection closes after it, so the client
retries the request on a new one (RFC 9112's rule for unanswered
requests). `Expect: 100-continue` is answered before the body.

**Logging.** One line a request: an opaque ID, the model, the status, the
token counts and times; one a swap, with its parts; once per unknown field
name. Never a prompt, completion, stop string or field value (D-014).

**Threads.** An I/O thread runs one epoll loop over the listeners and
every connection, all non-blocking: it reads requests, answers the model
list, the table and every refusal itself, and queues valid chat requests;
it never waits on the model, and no client's pace (a stalled head, a
reader that stops) holds up another's. The node's driver thread (the main
thread) takes the queue first come, first served, and runs one request at
a time, watching the runtime's signals (a signalfd) between requests and
between generation steps. It never touches a socket: it appends each
response's bytes to a buffer, whole events at a time, which the I/O
thread writes out as the client takes them. A connection that closes
(not one only half-closed, above) marks its request gone; the generation
ends at its next step and the request's lease is released as the backend
returns, while the buffer
lives until the driver lets go of it. Parsing a request's JSON (at most 4
MiB) is the one piece of CPU work on the I/O thread.

## Progress and deadlines

A chat request has no fixed deadline (D-097, the owner's note of
2026-09-29): the 600 s it had failed healthy long prefills (DeepSeek's
128K prompt stopped at 108,544 of 128,821 tokens,
[long-context](experiments/long-context/README.md)), and the
response time is ours to own, since the runtime can tell a backend that
is working from one that is not. Instead a request fails when the backend
stops making progress, and only a non-streaming request, whose client
hears nothing until the end, also has a deadline, scaled to its work
(`runtime/watchdog.h`, vendor-free and tested on a synthetic clock and
with the fake backend). Through the route on `spark` (2026-09-29,
DeepSeek plain at context 131,072), the 128,821-token prompt the old
deadline stopped streamed to its end: 836.8 s to the first token,
`[DONE]` at 845.8 s, 55 keepalive comments 15 s apart, no stall.

**Progress.** Each unit of the backend's work that ends is a beat: the
request taken up and admitted, the swap, each prefill chunk, each decode
step (a speculative step's draft and verify together), the response's
end. Before a unit that may be long the backend names it and its size: a
swap with the bytes it pages in (the incoming weights and state, and the
outgoing conversation's state written back), a prefill chunk with its
rows, decoding. A unit may take `stall_seconds` plus three times its
expected time at the model's floors: a chunk's rows at
`prefill_floor_tok_s`, a swap's bytes at 1 GB/s (a Spark pages in at
about 13 GB/s, [swap.md](experiments/fast-swap/swap.md)); a decode step,
which is short, the stall time (`decode_floor_tok_s` scales the
non-streaming deadline, below). So a unit that is long but healthy (a
configured chunk of many thousand rows, a swap from a slow disk) is not
cut short by the stall time; a swap, one program without beats inside,
is allowed for whole.

**A stall.** The I/O thread, which never waits on the model, watches the
beats. When a unit passes its allowance it acts at once, whether or not
the backend ever returns:

- the running request ends: a 504 `backend_stalled` before its headers,
  or after them an in-stream `server_error` event with that code and no
  `[DONE]`;
- its generation is cancelled the normal way: marked ended, as when a
  client leaves, so it stops at the backend's next step and its lease is
  released as the backend returns;
- the queued requests get a 503 `backend_unresponsive` (`Retry-After:
  10`, `x-should-retry: true`; in-stream for a stream that started);
- the backend is marked unhealthy.

**Unhealthy.** Until the backend's next beat (the unit it hung in
returning), every new chat request gets that 503 at once rather than
waiting behind a backend that may never return; `GET /v1/models` still
answers. The next beat makes it healthy again, and the log says so. A
unit that never returns does not hang the service: the node's own
patience for a step (`engine/paged_node.h`) cancels its request after ten
minutes, the failed step stops the service with status 1 (a node
failure, as before), and if even the cancellation never drains the
process aborts ten minutes later; `jitllm.service` restarts it either way
(`Restart=on-failure`).

**Health.** The watchdog keeps the backend's health: healthy or not, the
phase (idle, starting, swapping, prefilling, decoding, finishing), the
last progress, the stalls counted and when the last began. The log has a
line at each stall and each recovery, the service manager's status line
(`systemctl status jitllm`) says it, and `api::Server::health()` holds it
for M5's management listener.

**Deadlines.** A stream has none: keepalive comments hold its client's
idle timeout, and it runs until it is done, its client leaves (the
half-close rule above unchanged) or the watchdog fires. A non-streaming
request's deadline, from when it starts running, is

    min(deadline_cap_seconds,
        stall_seconds + 3 × (swap bytes / 1 GB/s
                             + prompt tokens / prefill_floor_tok_s
                             + max_tokens / decode_floor_tok_s))

counting the whole prompt (cached tokens too) and the request's
`max_tokens` or, without one, the rest of the context. At the defaults,
DeepSeek resident with a 131,072-token prompt and `max_tokens` 4,096:
120 + 3 × (1,310.7 + 819.2) s = 6,510 s, about 1 hour 49 minutes; with
no `max_tokens` at a 262,144-token context, the 4-hour cap. The watchdog
applies inside it.

**The queue.** A non-streaming request still waits at most 120 s, then
gets a 429 with `Retry-After`: it waits in silence, its client's own
timeout (600 s in OpenAI's SDKs) would otherwise decide, and its deadline
counts from when it runs. A stream in the queue, started after 15 s and
held by keepalives, has no fixed wait: it waits as long as the backend
makes progress, as a running stream runs, within the queue's 64 places.
A stall refuses the queue (above).

**The defaults.** `stall_seconds` 120 (30 to 3,600): a decode step takes
well under a second and a default chunk seconds at 8K (the table above),
so two minutes without either is a stuck backend, not a slow one. The
floors, 100 prefill and 5 decode tokens a second, are below every speed
measured here at 8K (DeepSeek's slowest prefill in the table, 220 tok/s
at 256-row chunks; its plain decode at 8K, 19 tok/s, plan.md), but not
at DeepSeek's deepest context, where the stall time and the margin of
three carry it. Extrapolated, not measured: its 2,048-row chunk's device
time grows about 0.14 s per 1,000 tokens of context (4.7, 7.9 and 12.3 s
at 8K, 32K and 64K, [long-context](experiments/long-context/README.md);
the same line gives 794 s of device time for the 128K stream, which
took 836.8 s to its first token), so about 40 s (52 tok/s) at 262,144,
inside its 181 s allowance (120 + 3 × 20.48); its decode step about
0.25 s there (4 tok/s), inside the deadline's 0.6 s a token; its whole
258,856-token prefill about 45 minutes, inside the deadline's 2 hours 11
minutes or more. A
swap's allowance covers a disk that pages in at a third of 1 GB/s or
more; a slower one (a hard disk, a NAS) needs a larger `stall_seconds`.
The node's own patience for a step, ten minutes, bounds every unit
whatever its allowance, so a `stall_seconds` above 600 leaves a hung step
to it alone. `deadline_cap_seconds` 14,400 (60 to 86,400).

## Limits

- One process, one model resident at a time: M3's full swap. Partial
  eviction, admission and the switching policy come with M5 and M6.
- The chat route is M3's minimal one: no tools, no reasoning controls,
  no credentials (the optional API key is M5's), no CORS, no Responses or Messages routes, one request
  at a time. The front door is M5's.
- The tailnet is found at startup; a node whose Tailscale comes up later
  serves it after a restart. `jitllm.service` is ordered after
  `tailscaled.service` (ordering only, no dependency) for that reason.
- A name the Host check does not derive (a LAN DNS name without a reverse
  record) cannot reach the route; a `[client] host_names` key could name
  such names later.
- The image serves one prompt a process, from a latents file (above), and
  only through the commands.
- A conversation is reused only when its re-rendered tokens extend what the
  state holds; a thinking model's re-rendered history usually does not
  (a client rarely sends the reasoning back), and the turn prefills again.
  Measured at 64K on Qwen3.8: a client that drops the reasoning
  re-prefills all ~61K tokens every turn (43 s); one that sends it back
  prefills only the new ones (0.2–0.3 s)
  ([long-context](experiments/long-context/README.md#turn-to-turn-reuse)).
- Per-token prefill and decode cost grows with the context (dense
  attention over every cached cell; the long-context report's gaps 1 and
  2), so a long prompt is slow: DeepSeek prefills about 230 tok/s at 64K,
  and a 128K prompt takes longer than the 600 s deadline the route had
  until 2026-09-29 (it now runs on while it makes progress,
  [above](#progress-and-deadlines)).
