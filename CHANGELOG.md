<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to jitLLM are recorded here, in the format of
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). jitLLM
follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html) and
stays at 0.x, where a minor release may break compatibility, until 1.0
(D-062). A change to a public surface names its version bump here.

## [Unreleased]

### Added

- Native GGML Q4_1/Q5_0/Q5_1 product primitives and legacy Q4_0/Q4_1/Q5_0/
  Q5_1/IQ4_NL row-preserving and joined vector products, with readable
  padding and independent column controls. Model qualification remains separate.

- Native classic SentencePiece GGUF tokenization for the approved Phi 3.5,
  Gemma 2 and Gemma 3 fixtures, with score-ordered merging, byte fallback,
  bounded working memory and pinned reference agreement.

- Native prompt, generation, retained and snapshot token histories are
  charged by allocated capacity before growth. Idle histories across the
  model library can release that memory with their state; active and
  paused exact continuations retain their funded tokens.

- Native Gemma 4 GGUF tokenization: raw UTF-8 BPE, space markers, byte
  fallback, pinned reference agreement and mode-aware request memory bounds.

- Qwen speculative verifies fuse the recurrent alpha/beta pointwise chains
  with original F32 arithmetic and saved-row restoration. A matched four-
  request 8K HTTP screen improves completed-token throughput by 1.34%.

- Calibration records track explicit chunk/slot overrides, including an
  override equal to the fallback. Qwen also tracks its effective draft head,
  shared draft cap, joined-draft limit, read alignment and depth-cost ratio,
  so reused measurements follow the execution policy.

- Qwen settings list the effective selected MTP head size and explain a
  capped request. Benchmark diagnostics record actual head kind and rows;
  the selected artifact's execution remains unchanged.

- Qwen MTP waves share selected BF16 heads while retaining each column's
  ordinary vector arithmetic and independent state. A matched two-request
  HTTP screen improves completed-token throughput by 1.77%; the existing
  two-request joined-drafting limit remains in force.

- Literal `/v1/completions` requests join Qwen and DeepSeek's native
  cohorts alongside chat requests. Prompt and generated likelihoods,
  Unicode offsets, response state and sampling survive model switches;
  zero-token scoring is supported at the full context. Unsupported
  continuations fail explicitly rather than restarting. Compatible with
  inference API version 1, a minor change in the 0.x line (D-100).

- A pending model switch pauses a running chat cohort at completed units
  after `[client] model_turn_seconds` of resident work (default 30), serves
  one substitute cohort, then resumes the original requests ahead of later
  arrivals. Paused state spills with its exact branch identity protected;
  response text, sampling and usage continue across the switch. Filled
  same-model slots still wait without preemption. Compatible with schema 2,
  a minor change in the 0.x line (D-069).

- Qwen3.8 waves run each request's attention and recurrence on its own
  stream from four requests, with shared products and dependencies ordered
  inside the same graph. NVFP4 and GGUF retain exact logits and state.
  `wave_lanes` in the model table defaults to true and can disable it;
  compatible with configuration schema 2, a minor bump of the 0.x line.

- Every model setting resolves in three layers (D-103): a default derived
  from the artifact (the trained context, the checkpoint's sampling
  defaults, the drafter's block, the vocabulary's reasoning markers; never
  a checkpoint's name), a calibration on the machine, or an override in
  `[models.<name>]`, else the constant measured on a GB10. Calibration is
  measured from a model's first uses (the progress floors, the recompute
  cost, DeepSeek's wave costs per width, Qwen3.8's draft depth cost) and
  recorded under the state directory per artifact, device, driver, build
  and the settings the measurements depend on (a new on-disk record,
  `jitllm-model-calibration-v1`; delete it to measure again), in force
  from the next start; a stale record is measured again, a corrupt one
  refused and replaced. The reclaim order's cost of recomputing dropped
  state is now a cost a token (`recompute_ms_per_token`), scaled by each
  conversation's tokens. The model table's schema is one table of keys, types, ranges and
  the models that take them, with new keys for every setting the runtime
  had fixed (sampling defaults, reasoning markers, draft rows, DeepSeek's
  wave costs and output-A/HCA prefill switches, Qwen3.8's draft
  vocabulary, depth cost, shared-wave depth, joined-draft width and wave
  read alignment, the reclaim order's recompute cost, the image's size and
  steps): compatible with schema version 2, a minor bump of the 0.x line.
  A request that sends no temperature, top_p, top_k or min_p now samples
  with the model's default, which is the checkpoint's own where its
  artifact keeps one (`do_sample: false` is greedy): unchanged for
  DeepSeek V4 Flash's GGUFs (temperature 1, top_p 1) and the Qwen3.8
  NVFP4 artifact (none kept), but Qwen3.8's GGUF quantizations now sample
  with their checkpoint's top_k 20 and top_p 0.95 when a request sends
  neither. The start logs each value and its source, and the new
  `jitllm-runtime settings [--json]` lists them with what each came from,
  beside a running service. A key a composition does not take is now
  refused naming that key; the throughput floors are resolved once per
  model instead of looked up per request.
- A genuine hang is recovered (D-102): the engine's fixed ten-minute step
  patience is gone, and a hang is work under way with no progress of any
  kind (no unit ending, no fence completing, no read or write landing)
  for `[client] hang_seconds` past its unit's allowance, so slow healthy
  work (a swap from a slow disk, a long prefill chunk, a long rendering
  or tokenization, which now beat as they go) never counts. The runtime
  then cancels the stuck work, and once the cancellation drained (nothing
  submitted before it still in flight: a read a drive holds is not ended
  by its cancellation), fails only the requests that needed it (503
  `backend_hung`) and resets the model in place (its stream fenced, the
  touched conversation state discarded, plans and graphs dropped, the
  model reloaded whole on its next request) while the service goes on;
  only when nothing less frees it (a read the drive never returns, a hung
  device, a stuck driver, a model that hangs again before serving a
  request, a failed reset) does it exit for systemd to restart it. Each
  step is logged with why.
- Conversations survive a restart of the service (D-105, a new versioned
  on-disk format, version 1, a minor bump of the 0.x line): a
  conversation whose state is wholly on disk (spilled,
  written back by a swap, or spilled at a graceful stop) is kept in named
  owner-only files beneath the spill role with a hashed record, and the
  next start adopts it after checking the build, model, layout, files and
  their SHA-256 digests, so its next turn restores the state exactly
  instead of prefilling again (measured: the same reply as an
  uninterrupted run, all 265 prior tokens reused, after a SIGTERM or a
  SIGKILL). A record that does not validate is refused and its files
  removed; retention and the spill budget apply while the service runs
  and across the restart, and a spill budget of 0 keeps nothing. The
  records hold the conversations' tokens and their files the state, on
  disk while the service is stopped: the new `[memory]
  keep_across_restart = false` (compatible with schema version 2) keeps
  nothing past the process, and purging the package removes the default
  spill directory's kept conversations. The records are hashed in the
  background at the lowest priority, waiting while a swap or a prefill
  runs.
- `jitllm.service` retries a failing start forever, backing off from 5 s
  to 30 s (it gave up after ten starts in five minutes), and its
  start and stop timeouts are extended while the start registers and
  adopts conversations and the stop spills and records them.
- A model's concurrent chat requests are no longer fixed at four (D-104):
  each LLM has request slots up to a cap, `[models.<name>] max_slots` (1
  to 16; a new key compatible with schema version 2, a minor bump of the
  0.x line), by default the measured knee (4 for DeepSeek V4 Flash and
  Qwen3.8 Flash Next with today's 16-row joined products; DeepSeek with
  DSpark takes at most 8, its waves of six to eight plain by the wave-form
  choice's measured costs), and the start logs the value and its source.
  Below the cap memory decides: a request joins the running ones only
  when the budget holds its prompt's state beside theirs, free or freed
  through the reclaim order, and otherwise waits first in the queue
  instead of being admitted to wait or be set aside. Qwen3.8's wave
  workspace is now sized from its widest measured wave instead of four
  prefill chunks: 7.45 → 2.17 GiB fixed with four slots, 5.45 GiB more
  room for conversation state, and a slot costs about 16 MiB of fixed
  buffers (DeepSeek with DSpark about 4 MiB). The reclaim order now covers
  a need at the least total expected cost to restore, so a small need
  takes a few stale plans and graphs instead of spilling a whole idle
  conversation, and admission never spills the conversation its request
  continues.

- Memory is used fully and given back gracefully under pressure (D-055
  and D-090 as amended 2026-10-02): plans and graphs no longer have fixed
  counts and survive a swap, growing into free memory; idle conversations
  are spilled to disk (writing only what changed since their last spill)
  and restored exactly at their next turn instead of being cleared and
  prefilled again (a request set aside for its peers' capacity likewise
  resumes from its spilled state, choosing no token again); everything
  reclaimable goes through one order, GreedyDual-Size over measured
  restore costs with stale entries aging out, taking all that is needed
  or nothing (caches never spill conversations); turn checkpoints are
  never refused for a budget full of caches; a swap that cannot make room,
  or fails (a read error) and is undone, or a first load that fails,
  fails only its requests (503) and the service goes on; and memory
  pressure from other processes (MemAvailable, the kernel's pressure-stall
  information) has the runtime give back what it can, in one trim to a
  target headroom and at a growing back-off while the pressure persists,
  never what the running model's next step uses. New configuration
  keys, compatible with schema version 2 (a minor bump of the 0.x line):
  `[memory] retention_hours` (default 24, the former fixed constant) and
  `spill_budget_gib` (default 128). A DeepSeek plan now holds 2.4 MiB
  instead of 9.9 (its arena holds what its graph uses), and the start's
  guard sets apart only one step's plans, so the DeepSeek DSpark beside
  Qwen3.8 configuration gains conversation-state room and can take a
  diagnostic state snapshot (`swap-table`), which the guard's margin was
  counted twice against before.

- Any chat template now gets chat routes when it can be rendered exactly
  (D-067 as amended): a native family renderer serves a template with a
  pinned hash or one it reproduces on a probe corpus (so repackaged copies
  and Unsloth's Qwen3.8 GGUF variant render natively), and any other
  template renders through a bounded, sandboxed Jinja-subset interpreter
  that matches Hugging Face transformers byte for byte on 29 real templates
  (Gemma, Llama, Mistral, gpt-oss, GLM, Kimi, Phi and others). Only control
  tokens in the template's own text become control tokens. Models whose
  templates were refused at registration now register; the request and
  response formats and `jitllm-inference-version` are unchanged.
- Native renderers for the Gemma 3 and Gemma 4 chat templates (template
  support only: jitLLM has no Gemma runner yet). By hash: Google's Gemma 4
  template (`ae53464b…`, 26B-A4B, 31B, 12B and its QAT GGUFs), the E2B and
  E4B one (`0a2c8073…`) and Gemma 3's (`7de1c58e…`, Unsloth's copies and
  the converted GGUFs surveyed); by probe: Unsloth's Gemma 4 templates,
  Google's of 2026-04-28 and 2026-05-18 (NVIDIA's NVFP4 checkpoints carry
  the first), Gemma 3n's and 270m's. They equal transformers byte for
  byte, in time linear in the conversation (the templates themselves are
  quadratic). Gemma 4 stops at `<turn|>` and at `<|tool_response>` (as its
  generation_config.json lists), so a model given tools ends its turn at a
  call instead of writing the tool's response itself.
- A case a native chat renderer does not implement now renders through
  the interpreter from the template's own text instead of failing the
  request, and native renderings are bounded as interpreted ones are
  (32 MiB of text).

- The community DeepSeek V4 Flash IQ2_XXS GGUF's "chat-v2" template
  (`antirez/deepseek-v4-gguf@f71f23d5`, SHA-256 `87249207…`) has a native
  renderer, so that model registers with chat routes instead of being
  refused for its template.
- Qwen3.8 Flash Next runs from its GGUF quantizations, imported verbatim
  (unsloth's UD-IQ3_XXS checked against llama.cpp b11254 on the same
  file: greedy equal but for near-ties, perplexity within 0.3%); served
  with the NVFP4 checkpoint's tokenizer and template, plain only. Through
  the runtime: 8K prefill 1.65× and decode 1.08× llama-server's. The build
  compiles GGML's tile kernels for Q4_0, Q2_0, Q3_K, IQ1_S, IQ2_S, IQ3_S,
  IQ4_NL and IQ4_XS too.
- DeepSeek chat serves up to four requests at once: their decode steps (and
  DSpark draft-and-verify steps) run in waves that read each weight once for
  all of them, each request's steps bit-identical to the same steps alone (a
  request preempted for state capacity is rebuilt by prefill, and a turn
  reuses whichever branch's cached prefix is longest, so those can still
  change a reply; so can concurrency with a drafter, since DSpark waves may
  take plain decode steps, below). At four matched 7K-token requests the
  completed-token rate rises 29% plain and 4.5% with DSpark. A request whose state cannot grow beside
  its peers' waits (an idle conversation's cached state is cleared first)
  instead of failing; only one that cannot fit alone fails for capacity.
- DeepSeek prefill uses ds4's stage mechanisms by default on GB10: F16 HC
  and attention rows, fused expert sums and pair activations, shared
  quantizations and the D2R Q2_K down product, each where its measured
  shape applies. Served 8K prefill gains 1.26% with identical logits; the
  community IQ2 checkpoint gains 7.21% at 2,048-row chunks, also exact.
- DeepSeek's qualified 4K GB10 IQ2 expert pairs use a scoped J64 kernel,
  improving the measured native 8K prefill rate by 4.15% with exact logits.
- An opt-in native DeepSeek output-A operation fuses inverse rotation,
  Q8 weight preparation and the grouped projection on the qualified
  4,096-row GB10 shape, preserving the diagnostic's complete logits.

- Qwen3.8 chat requests share decode work across up to four independent
  conversations (previously two), drafting two tokens each in a shared wave;
  matched 8K HTTP screens complete 20% more tokens a second at four requests
  and 9.8% more at two, with single-request output unchanged. A request whose
  conversation state does not fit beside its peers' waits, or is set aside and
  rebuilt once they finish, instead of failing; only a request that cannot fit
  alone is refused.
- Qwen3.8 chat requests share decode work across two independent conversations,
  retaining separate state, sampling and cancellation while reusing model weights.
  Compatible multirow HC products and unequal verification heads also share weights.
- `/v1/completions` accepts raw text or exact token IDs, legacy OpenAI
  echo/logprobs and vLLM prompt_logprobs with zero-token scoring. The
  bounded inference subset establishes `jitllm-inference-version: 1`.
- A bounded Qwen draft-head capture diagnostic supplies real operands for
  isolated product comparisons without changing serving arithmetic.
- A bounded Qwen routed-down capture diagnostic preserves common-input
  operands for isolated expert-product comparisons without changing serving.
- DeepSeek V4 Flash's community IQ2_XXS/Q2_K GGUFs run natively, including
  F16 attention-compressor APE tables, without requantizing their weights.
- LLM turns reuse stable checkpoints when clients remove earlier reasoning,
  with two private disk checkpoints per model and a 24-hour reuse limit.
- LLM conversation state allocates and spills only used cache extents;
  long context ceilings no longer allocate the whole cache at registration.
- The `jitllm` command, with `--version`: the product version (`X.Y.Z` for a
  release, `X.Y.Z-dev.N+g<commit>` otherwise), the commit, the license
  profile, the SDK identity and the target. The build receipt records the
  same version, with its Debian form (`X.Y.Z~dev.N+g<commit>-1`).
- `jitllm doctor`, a capability probe: the build, the host (kernel, glibc,
  memory), RDMA ports, the NVIDIA driver and each GPU's compute capability,
  compute mode, VMM support and backing granularity, and whether it is
  unified (the GB10) or discrete. It exits 1 when the host cannot run the
  build: a GPU the build has code for is needed, with VMM and host-backed
  VMM.
- CUDA builds of `jitllm` need the NVIDIA driver (`libcuda.so.1`) to start.
- Discrete NVIDIA GPUs are a secondary target (D-082): the x86-64 build
  also has code for compute capability 8.6 (`sm_86`), and `jitllm doctor`
  and the runtime accept such a GPU with VMM and host-backed VMM. Arm64
  builds stay GB10-only. jitLLM uses one GPU, device 0: `jitllm doctor`
  judges only it and warns on a host with more.
- The node's configuration, `schema_version = 2`: `/etc/jitllm/jitllm.toml`
  and the fragments in `/etc/jitllm/jitllm.d/`, strict TOML 1.0 in which
  every key has one owning file, with the `[storage]` roles, `[limits]` and
  a cluster member's keys. Every problem is reported with its file, line
  and column, and files or directories that other users could change are
  refused.
- An arm64 Debian package, `jitllm`: the `jitllm` command, the node runtime
  `/usr/libexec/jitllm/jitllm-runtime` and `jitllm.service`, which runs it
  as the new `jitllm` system user with `/var/lib/jitllm` as its data
  directory. It depends on the NVIDIA driver 580 or newer. The runtime reads
  its configuration, prepares its storage, checks the host and waits; it
  serves nothing yet. It exits on a fatal signal instead of dumping core.
- `jitllm doctor --config FILE` reads a configuration other than the
  default, and doctor now reports the configuration and the storage roles:
  their owners, modes and filesystems.
- The configuration names the models a node serves, `[models.<name>]`
  (still `schema_version = 2`: the keys are new): an installed artifact or
  composition by ID, a speculative drafter, whether to speculate, the
  conversation's context, its prefill chunk, and a tokenizer and chat
  template where the artifact keeps none (D-096).
- `jitllm-runtime` serves models by hand, in its own process (D-096):
  `jitllm-runtime chat --turn MODEL TEXT...` sends each turn to its model,
  swapping models as needed, and `jitllm-runtime swap-table` measures M3's
  swap table between the configured models. Replies are greedy, and
  speculative where a model has a drafter (`--plain` turns it off). CUDA
  builds only.
- With models configured, the runtime service serves a minimal
  OpenAI-compatible chat route (D-097): `POST /v1/chat/completions` (text
  messages; `max_tokens` or `max_completion_tokens`, `temperature`,
  `top_p`, `top_k`, `min_p`, `seed`, `stop`, `stream` with
  `stream_options.include_usage`; reasoning returned as `reasoning`),
  `GET /v1/models` and `GET /v1/models/{id}`, one request at a time,
  first come first served, behind a queue of 64, swapping models as
  requested. It listens on loopback and the node's Tailscale addresses by
  default (`[client] bind = ["loopback", "tailscale"]`, `port = 8114`), or
  on any configured address; none needs authentication, as with other
  engines, and the start log names each listener beyond loopback and the
  tailnet as served without it. The Host and Origin must name the node as
  it listens (its tailnet MagicDNS name included), and cross-site browser
  requests are refused. Connections persist (HTTP/1.1 keep-alive, up to
  `[client] max_connections`, 1,024 by default) on an event loop where no
  slow client holds up another; streams get `: keepalive` comments while
  they wait, swap or prefill. Known fields the route does not implement
  are refused when they would change the answer; unknown fields are
  ignored, their names (never values) counted at
  `GET /jitllm/v1/ignored-fields` for loopback clients. Every intake
  bound, timeout and refusal is listed in docs/runtime-serving.md. The
  configuration's schema version stays 2. The runtime can sample (seeded,
  with speculative sampling where a model has a drafter) as well as decode
  greedily. A CPU-only build refuses to start with models configured.
- A DeepSeek artifact whose chat template has no native renderer is now
  refused when it registers, as Qwen3.8's already was, naming the
  template's SHA-256.
- The arm64 package ships NVIDIA's cuBLAS (`libcublas.so.13`,
  `libcublasLt.so.13`, unmodified) in `/usr/lib/jitllm` for the runtime, and
  depends on `libgcc-s1`; its third-party notices now include GGML's and
  CUTLASS's, whose kernels the runtime links (D-076, D-096).
- The model support matrix, `docs/model-support.md`: the models jitLLM
  runs and serves (DeepSeek V4 Flash 0731 with DSpark, Qwen3.8 Flash Next
  NVFP4 with its MTP drafter, Qwen-Image-2.1, and the M2 fixtures), each
  with its checkpoint, artifact, chat template hash, decoding modes,
  evidence and known divergences.

### Changed

- Qwen3.8 GGUF concurrent decode now shares dense and routed quantized
  products and the full target head across requests, with byte-identical
  full logits and state against the original waves, including graph replay.
- DeepSeek's concurrent requests now run their own attention and state
  work at the same time, one CUDA stream each, inside a wave, with each
  request's results unchanged bit for bit. At four 124-token requests
  plain waves complete 4.7–4.8% more tokens a second, and with DSpark
  4.6–5.1%, whose wave-form costs were measured again (at 7K, where
  prefill dominates, the gains are within noise).
- A DeepSeek DSpark wave now drafts every request's block in one graph,
  reading the drafter's and the head's weights once for all of them, with
  each request's drafts unchanged bit for bit: a DSpark wave of four or
  five requests takes 4–5% less time (same session). End to end at four
  and five requests the rate is unchanged within noise, since the chosen
  form mostly runs plain waves there.
- With its DSpark drafter, DeepSeek now chooses per wave of concurrent
  requests between draft-verify waves and plain decode waves: the tokens
  draft-verify waves accept against a cost measured for each wave width.
  No clock enters the choice, so the same requests in the same waves choose
  the same forms. At four requests this is up to 7% faster than DSpark
  waves alone and 1–4% under plain waves without the drafter. A lone request
  keeps speculating (31–54% faster), and a wave with a sampling request
  always speculates, so seeded replies repeat. A greedy reply at several
  requests may now differ from the same request alone, and between runs
  whose requests arrive in a different order (the forms' arithmetic
  differs, and arrival order decides which waves a request joins). A
  model's new `wave_form` key (`"auto"`, the default, `"speculative"` or
  `"plain"`) fixes the form; with `"speculative"` a reply at several
  requests equals the reply alone.
- Concurrent chat prompts now prefill shortest remaining first, so a short
  prompt no longer waits behind a long one that arrived first, and equal
  prompts each stream their first token in turn instead of all at once at
  the end. A long prompt still runs after at most 12 others' prefill units,
  and a generating request waits for at most one prefill unit between its
  tokens. With one ~32K-token DeepSeek prompt followed by three short
  ones, the short prompts' first tokens come at 4.8–6.9 s instead of
  18.6–19.0 s; four 124-token chats' at 0.9 / 1.7 / 2.5 / 3.4 s instead
  of 3.0–3.5 s. Throughput is 1–2% lower for four equal prompts (their
  first finishers decode between the others' prefill units). DeepSeek's
  replies are unchanged.
- DeepSeek's output-A/HCA prefill now also takes a prompt's last, partial
  chunk (64 rows or more), which passes every quality control under the
  owner's tie-aware greedy rule: 7K-token chat completes 5% faster alone
  and 7.3% (community GGUF) and 8.5% (0731) faster four at a time (first
  token 8.3 → 7.3 s on the
  community GGUF, 9.4 → 8.3 s on the 0731). Replies change with the
  arithmetic.
- The chat route's limits follow real resources and default to permissive
  (D-102): no message, content-part or stop-string counts, no 8 MiB
  message or 128-byte stop caps, and no queue count, queue wait or
  non-streaming deadline unless `[client]` sets one (`max_queued`,
  `queue_wait_seconds`, `deadline_cap_seconds`). Requests' host memory is
  one pool charged by what each request really holds, bodies as they
  arrive, parses, queued requests, renderings, tokenizations and a
  stream's unread output included: 256 MiB is set apart at the start, and
  past it the pool grows within the memory budget, displacing caches and
  idle conversations, and shrinks as requests end (`request_memory_bytes`
  caps it); what cannot fit is refused before the work (413, or 503 to
  retry); a prompt that cannot fit the model's context is refused before
  it is queued. A body follows the pool and the largest configured context (on
  a Spark serving DeepSeek V4 Flash at 300K tokens, 221 MiB instead of 16
  MiB; `max_body_bytes`), connections the open-file hard limit. A stream
  whose client stops reading pauses, and if it keeps others waiting
  yields its place, continuing once its client reads. Large bodies parse off the I/O
  thread; stop strings of any number are matched together in one shared
  automaton; a chat text longer than any prompt that fits is refused
  before rendering, and tokenization works in bounded windows. JSON
  values follow the body (a 1M-token ID prompt parses) and nest to 64.
  `max_tokens` and literal top scores (`logprobs`, `prompt_logprobs`, now
  up to the vocabulary) are bounded by the model, not fixed caps. A stall
  is reported (log, health, service status) without failing requests
  (`stall_action = "fail"` restores that); head and body timeouts count
  inactivity (`request_inactivity_seconds`); a stream whose client stops
  reading pauses at a completed step instead of being dropped after 30 s
  (`write_inactivity_seconds` restores a drop). Chat templates have no
  step or work caps: a long rendering ends when its request does, and
  rendering bounds follow the model's context and memory. The
  configuration no longer caps models at 16 or contexts at 1,048,576. All
  earlier configurations and requests stay valid: `schema_version` 2 and
  `jitllm-inference-version` 1 are unchanged.
- DeepSeek serves its output-A/HCA prefill by default on full 4,096-row
  chunks, where it passes every registered quality bound on both GGUFs:
  7K-token chat completes 7% faster alone and 12% faster four at a time
  (first token 9.8 → 8.3 s on the community GGUF, 10.9 → 9.4 s on the
  0731). Replies change with the arithmetic. Those plans are no longer keyed by chunk position, so chunks of
  one shape share a plan. Four-request decode waves read each dense
  weight once for all requests and find their routed experts' rows with
  one warp's ballots instead of one thread's scan (bit-identical to
  before, waves 7% faster). With output-A/HCA, four concurrent 7K-token
  chats complete 15–16% faster on both GGUFs, and four 124-token chats
  12% faster.
- DeepSeek prefill is 3–4% faster on both GGUFs through the runtime
  (matched 7K-token prompts, one and four at a time). A prompt's last,
  partial prefill chunk of 64 rows or more now takes the stage mechanisms
  a full chunk takes (D2R; the IQ2 pair's activation write-back on
  compact chunks; with the opt-in output-A/HCA prefill, those too: the
  community GGUF's 7K prompt then prefills 21% faster, in 6.8 s). On the
  community GGUF partial chunks' logits move with D2R (32K perplexity
  +0.53%, inside the 3% bound). The 0731 UD-Q2_K_XL GGUF's F32
  hyper-connection mixes, IQ2_XS gate/up experts and Q5_K/Q6_K shared
  experts now take the fused HC norm, expert sum, write-back and shared
  quantization; its replies are unchanged byte for byte.
- Chat templates' case mapping (`upper`, `lower`, `capitalize`, `title`,
  case-insensitive `dictsort`, `sort` and `unique`, the `lower` and `upper`
  tests) is now Python's in full, as transformers renders it, instead of
  ASCII-only: `ß` upper-cases to `SS`, Cyrillic and Greek map (with the
  final sigma), and the `title` filter follows Jinja2's own rule. The
  Unicode tables gain the case data of UCD 15.1.0's SpecialCasing.txt and
  DerivedCoreProperties.txt.
- DeepSeek prefills in 4,096-row chunks by default (was 2,048): 12-15%
  faster at 8K-32K tokens on the 0731 GGUF for 1.28 GiB more fixed memory.
  Above a 262,144-token context the default stays 2,048 rows, whose
  smaller attention mask leaves a 1M conversation its state (there 4,096,
  capped to 4,024 rows, fixed 2.14 GiB more). Any DeepSeek chunk is now also capped
  where that mask fits the kernel's 32-bit strides (13,528 rows at a
  262,144-token context, 4,024 at 1M); wider chunks refused a deep
  prompt's last chunks.

- DeepSeek fast prefill fuses eligible Q-head normalization and rotation,
  preserving native F32 arithmetic without a normalization intermediate.

- DeepSeek's fast wide prefill combines its six expert outputs in one F32
  kernel with the original multiplication and addition order; unsupported
  shapes and reference plans keep their ordinary operations.
- DeepSeek fast production prefill with the measured Q4_K head computes
  only its frontier head row, while retaining every DSpark feature row;
  reference, verify and scoring paths keep their requested heads.
- DeepSeek count-based HCA attention retains the ordinary MMA path;
  selected-list CSA and window attention keep shared sparse gathers.
- Qwen3.8's small-output MXFP8 verify products use measured GB10 warp
  schedules, preserving their arithmetic and adding no workspace.
- DeepSeek fast prefill can share sparse KV gathers across eight queries
  and expert input preparation across gate/up products, and use compact
  expert tile lists from 2,048 rows, retaining F16 caches and the original
  quantized weights. Primitive/reference plans keep their original
  dispatch choices.
- Models default to 262,144 tokens of context (was 8,704). DeepSeek V4
  Flash accepts up to its trained 1,048,576-token ceiling; Qwen3.8 Flash
  Next remains capped at 262,144, checked before model allocation. Growing
  state remains subject to the physical memory guard. The chat route accepts
  bodies up to 16 MiB and message text up to 8 MiB, with `max_tokens` parsed
  up to 1,048,576; its aggregate intake bounds and prefill chunk policy stay
  unchanged. Schema version remains 2.
- Qwen3.8 greedy MTP decoding chooses depth two or three from acceptance;
  state restore preserves the schedule. Draft confidence is computed only
  when requested. The experimental importer can prepare a selected BF16
  draft head and original token-ID map from an external vocabulary list.
- Sampling with `top_k = 1` uses the greedy path, including speculative
  verification, avoiding candidate selection and random draws while keeping
  parameter and logit validation and the same token choices.
- DeepSeek V4 Flash's per-token cost no longer grows with the
  conversation: its window cache is a ring of the window and a prefill
  chunk, its attention reads only each token's window and the indexer's
  selected (or HCA's visible) compressed rows, and its indexer scores on
  tensor cores and selects deterministically. Through the runtime on a
  GB10, prefill runs at 471 / 466 / 445 tokens a second and plain decode
  at 21.6 / 21.2 / 20.5 at 32K / 64K / 128K tokens of context (before:
  333 / 230 / 154 and 14.7 / 10.8 / 7.1); a long prompt now repeats bit
  for bit, and its greedy tokens may differ from the previous build's at
  near-ties. At `context = 262144` it maps 3.5 GiB beside its weights
  (was 20.4), and with its DSpark drafter it now starts there (it was
  refused above 143,360).
- The runtime's start keeps 6 GiB (was 4) beside the largest model's
  weights for memory the node does not count, and also counts the
  largest chunk inputs a model builds on the host; a configuration that
  started within 2 GiB of the old limit may now be refused, with a
  message naming both.
- Sampling (temperature above 0, the chat route's default) no longer sorts
  the vocabulary for each token: a draw takes 0.45 ms instead of 5.9 ms at
  DeepSeek V4 Flash's vocabulary and 0.87 ms instead of 11.9 ms at
  Qwen3.8's, on a GB10, speculative verification included. A seed still
  draws the same tokens every time within a build, from the same
  distribution, but may draw other tokens than the previous build did.
- On the chat route, a client that shuts its sending side after a whole
  request now gets its response (a stream starts at once; a non-streaming
  response on HTTP/1.1 is preceded by an interim `102 Processing`), where
  before the request was cancelled; a client that closes
  its connection still cancels. A connection kept alive between requests
  no longer keeps its last request's body or response allocated.
- The runtime prefills in wider chunks by default, chosen per model
  (DeepSeek V4 2,048 rows, Qwen3.8 4,096): an 8K-token prompt prefills
  1.48× (DeepSeek) and 1.78× (Qwen3.8) as fast as in the 512-row chunks
  before, for about 1.4 GiB more workspace. `[models.<name>]
  prefill_chunk` sets a model's chunk; it is capped at what the model
  allows at its context, so every context the configuration accepts
  starts (at 512, DeepSeek was refused and Qwen3.8 with its MTP drafter
  failed at startup).
- The chat route no longer ends a request after a fixed 600 s, which
  failed healthy long prefills: a request fails only when the model
  backend makes no progress (no swap, prefill chunk or decode step
  ending) for `[client] stall_seconds` (default 120), with a 504 before
  the headers or an in-stream error after; the backend is then reported
  unhealthy (in the log and `systemctl status`) and requests get a 503
  until it makes progress again. A stream has no deadline; a
  non-streaming request's is scaled to its work at the model's
  `prefill_floor_tok_s` and `decode_floor_tok_s` (100 and 5 by default),
  three times over, at most `[client] deadline_cap_seconds` (4 hours). A
  stream waiting in the queue no longer runs out of time there. The
  configuration's schema version stays 2 (new keys).
- The chat route now ends a request whose client left, whose deadline
  passed or whose runtime is stopping between prefill chunks (and after a
  swap), not only once generation starts; the service keeps serving, and
  the conversation keeps the chunks that ran, so a retried request
  continues from them.
- Qwen3.8 Flash Next no longer slows with context: it caches each
  block's indexer key once, selects its attended cells on the GPU at any
  depth, and attends those 2,051 cells alone instead of every cell. Through
  the runtime on a GB10, plain decode is 26.8 / 26.0 / 24.4 tok/s at 8K /
  64K / 256K (before: 27.4 / 17.1 / 6.8) and prefill 2,314 / 2,359 / 2,178
  tok/s (before: 2,320 / 1,367 / 487). Its selection now repeats bit for
  bit at any depth, speculation with its MTP drafter works to its
  configured 262,144 (it was refused above 32,768), the prefill chunk
  stays 4,096 rows at every context (it was capped at 2,040 at 262,144),
  and at 262,144 the memory it reserves falls from 24.7 to 11.1 GiB (its
  peak from 101.2 to 84.7 GiB). Greedy tokens may differ from the
  previous build's at near-ties.
- Qwen3.8 Flash Next's plain decode is 5–8% faster (its recurrent state
  updated in place, and fused and clustered one-row kernels); its greedy
  tokens may differ from the previous build's at near-ties.
- Qwen-Image-2.1 generates about 8% faster (33.4 s against 36.2 s at
  1,024², 40 steps, weights resident, on a GB10) with the same BF16
  numerics (pinned cuBLASLt products, fused norms and attention,
  implicit-GEMM VAE convolutions, a CUDA graph per denoising step), its
  operations run through a plan bound against the implementation
  registry; its pixels differ from the previous build's in the last bits
  of the VAE's convolutions (still within the bounds of diffusers' image).

### Removed

- The temporary complete ds4 prefill reference (the `jitllm_ds4_complete`
  benchmark and its original cache, HC/norm, compressor, repack, product,
  indexer, sparse-attention and router/FFN stages, weight preparation and
  state layout) no longer builds into jitLLM's libraries; production keeps
  the original output-A and F16-Q token-tile HCA cores it selects. Commit
  `ec8af04` holds the rest
  ([report](docs/experiments/ds4-complete-plan/README.md)).

### Fixed

- Package purge deletes kept conversations as the service user, so a
  service-owned parent path swapped for a symlink cannot redirect a root
  deletion.

- Spark jobs keep their GPU lock in the child as well as the supervisor;
  losing the supervisor no longer admits another job alongside orphaned
  work, and orphaned or waiting jobs report the host busy.

- Paused streams no longer accumulate keepalive comments behind pending
  output. Prepared prompt tokens keep their request-memory charge through
  shared continuation ownership; continuation copies and growing request
  token histories are charged until they retire.

- Incremental stop matching compacts retained text only after enough
  bytes are emitted, avoiding repeated copies of a long stop prefix.

- Kept-conversation work queues retain only the latest record for each
  model slot and release invalidated queued copies immediately, bounding
  stale history storage while hashing yields to inference.

- Jinja scope variables count toward live memory. Mixed integer/float
  comparisons preserve integer precision and unordered NaN comparisons;
  float floor division and remainder match Python, while complex powers
  and overflowing float powers are refused.

- A failed diagnostic DeepSeek tensor dump keeps its pinned destination
  alive when submitted DMA retirement is unproven.

- The engine's paging kernels (range fills, n-gram row gathers, draft-id
  clamps) clear the CUDA error a failed launch leaves, so it no longer
  surfaces in a later, unrelated check, and the DeepSeek prefill HCA
  kernel refuses graph capture, whose replay would keep a stale first
  position.

- A model whose post-load checks fail is evicted instead of staying
  resident unchecked; a first load after a failed swap makes room through
  the reclaim order; a graceful stop's spills and an oversized idle spill
  respect the spill budget; and a refused swap keeps the incoming model's
  conversations kept on disk.

- The chat route answers `Expect: 100-continue` once even when the body
  waits for request memory, and a half-closed client whose body waits no
  longer spins the I/O loop.

- Kept-conversation records with a speculation cursor past their tokens are
  refused, a failed record removal is retried, and a record whose write
  failed after its rename is still removed when its state changes.

- Jinja `map` of `map` counts against the evaluation depth bound (a hostile
  template could overflow the stack); `%d` and `int` of non-finite or
  out-of-range floats, `float` of large integers, `'1e500'|float` and
  `split(None, n)` remainders now match Python or refuse.

- Jinja strings and renderings count their trusted ranges (16 bytes each)
  toward the string and output bounds as they grow, so a template that
  alternates its own text and a client's byte by byte can no longer hold
  several times those bounds uncharged; the rendering's ranges pass to the
  caller without a copy. The control tokens an interpreted template places
  count toward the same output bound, so a template repeating a short
  control token is refused rather than holding several times its output in
  spans, and a container's repr and `tojson`'s text are charged while they
  are copied.

- Qwen drafts record every catch-up row for incremental spill, so kept
  drafter cells are never stale on disk; DeepSeek's `pair_glu_q8` accepts
  partial prefill chunks; and several kernel host checks are tightened.

- A Qwen prefill chunk with the drafter's injection after a verify (a new
  turn after speculative decoding) catches the MTP drafter up on every row
  the verify kept, each from its own streams, instead of running the
  previous position from a stale stream row and skipping the others; turns
  no longer lose acceptance at their start (target output was unaffected).
  A scalar Qwen verify saves the stream rows it overwrites, as a wave's
  does, and a one-request speculative step (Qwen or DeepSeek) whose host
  judgement fails undoes its verify and keeps the conversation's prefix,
  as a wave already did, instead of clearing it.

- Package purge refuses a symlinked spill directory before its root `rm`.

- Adopting a kept conversation empties every extent of its spill file that
  its record does not list, so a record whose removal had not reached the
  disk before a power loss can no longer hand a later conversation's state
  to the adopted one. The keeper writes and syncs a record and syncs its
  directory outside its lock (only the rename is under it), so a turn
  restoring a spilled conversation never waits on a disk sync, no
  stale temporary record is left behind, and a temporary record is renamed
  into place only while it is still the file the keeper wrote.

- The spill budget counts what a spill or swap will hold on disk: an idle
  spill and a graceful stop's count the conversation's whole state, and a
  swap's write-back the outgoing model's whole state, never deleting the
  incoming model's conversations for it, nor any until the swap's room is
  made; a spill that cannot fit deletes nothing, and a graceful stop keeps
  the most recently used conversations. A reclaim whose victim gives back
  nothing selects again without it, and idle state that could only be
  dropped while a continuation holds it is no longer offered.

- Jinja container comparisons and member lookups poll request cancellation
  while scanning their values, including repeated references to large strings.

- The cross SDK includes a pinned ARM shared GCC support runtime for
  cuBLAS, allowing CPU test discovery under qemu without changing the
  compiler or jitLLM's static C++ runtime.

- Package provenance checks read license sidecars for JSON and other
  noncommentable source files, while code still requires embedded headers.

- CUDA-free builds exclude the CUDA-only DS4 source component from their
  receipts, so the strict source inventory agrees with the CPU build.

- DeepSeek automatic calibration exploration respects explicitly supplied
  wave-cost prefixes, including zero, which always speculates. Widths
  outside the override prefix retain bounded first-use exploration.

- Calibration records become stale when effective context or DeepSeek
  automatic wave-cost overrides change. Equal fallback-valued overrides
  and different override-prefix lengths have distinct dependencies. The
  existing record format and fresh serving defaults are unchanged.

- Fresh LLM rows in `swap-table` now generate one greedy token before
  ending their clocks. Their complete prefill-head hashes still check
  repeatability; saved-context returns retain their decode-step endpoint.
  Previously fresh rows stopped after prefill, before token selection.

- Concurrent chat prompts again prefill in order of what is really left:
  a conversation's stale history no longer counts as reused, a unit that
  prefills nothing ages no other prompt, and a started prompt about to
  finish goes before an aged one. At four 7K-token requests the second
  first token comes at 14.7 s again instead of 18.8 s.
- DeepSeek's DSpark drafter failed with the community IQ2_XXS GGUF ("get_rows
  of quantized rows into F32"): the verify's device lookup of the drafts'
  rows now takes its F16 token table. With the drafter, one 7K-token chat
  completes in 24% less time than without it (16.98 against 12.93 tok/s),
  a short one 35% less. Draft ids are now bounded to the vocabulary before
  any lookup, so a drafter's non-finite logits can no longer index past a
  table.
- A chat prompt over 4 MiB of rendered text, or over 4,194,304 tokens, was
  refused by the tokenizer's library defaults below the body limit and
  DeepSeek's 1M context; the chat and literal routes now bound
  tokenization by the rendering and the model's context.
- `jitllm-runtime` refused to start the community DeepSeek V4 Flash
  IQ2_XXS GGUF ("a DeepSeek V4 wave needs every layer in the fast plan's
  fused form"): its F16 hyper-connection mixing weights kept every layer
  off the fused form that four-request waves need. The fused form now
  takes F16 and BF16 mixing weights (and quantized ones through GGML's
  product), so that artifact starts, serves four requests in waves
  (+41% at four 7K-token requests over one at a time), and its
  single-request decode takes the fused form too (+4%). An artifact whose
  weights cannot form a wave is now served one request at a time, with a
  log line at start, instead of being refused.
- Stopping `jitllm-runtime` with two models configured, after a swap,
  logged "a compute stream could not be fenced" and exited with a failure
  status: the stop's fence of the swapped-out model's stream tried to page
  its spilled state back in, beyond the execution budget. The fence now
  leases nothing, so it neither fails nor reads anything back, and that
  stream's work is proven complete before its model is released; if a
  fence still fails, the stop leaves the models' memory as it is rather
  than release it under work that may still be running.
- A `/v1/completions` request (or a serial chat turn) refused for
  conversation-state capacity stopped the runtime, even when idle chat
  conversations' retained state held the capacity it needed. It now clears
  that idle state, the largest first, and goes on; one that cannot fit
  alone fails with a 500 and the service keeps serving.
- Cached plans and graphs, which live outside the catalog, are now
  bounded: each runner caps them (chunk and drafter plans shared by its
  request slots, wave plans, and every graph, its drafter's and waves'
  included, by count and by bytes; least recently used dropped) and reports
  the most they can hold, and a swap drops the outgoing model's. The
  runtime's memory guard sets apart the largest model's bound (DeepSeek
  with DSpark: 0.63 GiB), so many distinct request shapes can no longer eat
  its uncounted margin and push the host into swap. A model returning from
  a swap plans and captures its graphs again.
- Routed quantized products allocate scratch for partial expert tiles'
  dummy columns, including raw FP4, so bounded workspace covers every read.
- Qwen3.8's sparse attention rejects unequal key/value cache row strides
  before launch, preventing incorrect reads of padded caches.
- The long-context correctness judge refuses incomplete or non-finite
  captures instead of passing shortened comparisons; a repeatability
  mismatch now also returns a failing exit status.
- Qwen3.8 Flash Next at its configured maximum context (262,144 tokens)
  was refused at startup, and past about 147K tokens its prefill failed:
  its prefill chunk is now capped where a chunk's [context, rows] tensors
  would pass 2^31 bytes (2,040 rows at 262,144; the startup log says so),
  and a 258,633-token prompt prefills and answers.
- DeepSeek V4 Flash's prefill failed past about 52K tokens of context (an
  operation check stricter than the kernel it guards); it no longer does
  (checked with a 64K prompt, and 128K in the resident harness).
