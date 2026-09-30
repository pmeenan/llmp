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

### Fixed

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
