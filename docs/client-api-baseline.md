<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Client API baseline

Documentation check: 2026-09-22. Decisions: D-040 (routes and client subset)
and D-045 (front-door listener, auth and CORS defaults, admission status and
keepalive contract, standard-client signals, alias echo), with D-047's
reasoning, storage and non-streaming corrections, following D-022/D-030.
This is the M5 implementation contract and test plan for the
inference front door, not a claim that llmpalooza serves these clients. M3
serves a minimal `/v1/chat/completions` and non-streaming raw
`/v1/completions` with supplied-token likelihoods on loopback and the tailnet
([runtime-serving.md](runtime-serving.md#the-chat-route), D-097), checked
with curl, not with these clients; the surface here arrives in M5 (D-087).
M3's route already does three things this document assigns to M5:
persistent HTTP/1.1 connections, SSE keepalive comment lines on Chat
Completions (below), and ignoring unknown fields by name under a
documented rule (the last section). Sources are live official documentation, not
pinned client binaries; recheck and record exact client versions and
configurations when running acceptance.

D-100 adds the bounded [literal likelihood contract](runtime-serving.md#literal-completions-and-likelihoods):
OpenAI echo/logprobs and vLLM prompt_logprobs, including max_tokens zero.
This does not claim full client acceptance. The initial one-row scorer is
memory bounded and has decode-like throughput; generation quality suites
still need ordinary long prefill to exercise large attention tiles.

Related assessments: the [Ollama and API capability assessment](api-capabilities.md)
(D-041/D-042: Ollama subset, discovery, continuation close, download/warm jobs,
file modalities, MCP management, sharing controls, embeddings), the
[vLLM API gap assessment](vllm-api-assessment.md) (D-043/D-044: direct
compatibility, tokenization/rendering, constrained output, reasoning,
reranking, metrics, raw Completions) and the
[OpenRouter assessment](openrouter-api-assessment.md) (D-046: model-metadata,
reasoning and hint spellings adopted on the OpenAI-shaped routes; hosted-routing
features excluded). Delivery milestones for scope outside M5 are in
[plan.md's milestone ladder](plan.md#milestone-ladder). D-040's JSON response
envelopes do not imply schema-constrained generation; that comes from D-043.
Advertise only the implemented feature profile; llmpalooza extensions stay separate.

## Named clients

| Client | Documented connection and selected baseline | Evidence limits |
| --- | --- | --- |
| OpenCode | Custom provider with `/v1` base URL and configured model IDs; select Chat Completions. The main docs use `provider`, `npm: @ai-sdk/openai-compatible`, and `options.baseURL`; they explicitly distinguish `/v1/chat/completions` from `/v1/responses` via `@ai-sdk/openai`. Model limits and capabilities come from models.dev or the user's `models` config, not from `/v1/models`. | The separately published v2 docs use different configuration/package names. Pin the installed version and follow its matching docs; do not combine schemas. |
| Codex | Custom provider with `base_url`, `env_key`, and `wire_api = "responses"`; serve `POST /v1/responses` over HTTP/SSE. The config reference documents `responses` as the only `wire_api` value and the default. Documented defaults: `stream_idle_timeout_ms` 300000, `request_max_retries` 4, `stream_max_retries` 5. `ollama` and `lmstudio` are reserved built-in provider IDs; Ollama's Codex guide configures `http://localhost:11434/v1/` with Responses, and `codex --oss` selects a local provider. | Validate the chosen CLI build first; desktop/IDE compatibility is not established. With response storage disabled, Codex returns `reasoning` items, including `encrypted_content`, on later turns; the profile must accept items llmpalooza produced. Whether the built-in `ollama` provider can target another host, and exactly what it sends, is unverified. |
| Claude Code | `ANTHROPIC_BASE_URL` selects Messages; serve `POST /v1/messages`, including `?beta=true`, and `POST /v1/messages/count_tokens`. With `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` it also calls `GET /v1/models?limit=1000` in the Anthropic list shape with a 3 s default timeout, treats any redirect as failure, and keeps entries whose `id` contains `claude` or `anthropic`. `x-claude-code-session-id` and, on subagent traffic, `x-claude-code-agent-id` arrive on every request. The gateway hint headers are off by default on a custom base URL and arrive only with `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` (v2.1.273+): then every request carries `x-claude-code-request-class` (`main`, `subagent`, `workflow`, `compaction`, `auxiliary`) and the first main-conversation request after a compaction carries `x-claude-code-context-compacted`. Startup may also send a best-effort `HEAD /api/hello` probe that can be rejected. | Use a gateway credential, configured local model IDs and supported capabilities. Unrecognized aliases receive the full current-model field set: adaptive `thinking`, `output_config` effort and `context_management` with its beta header, plus `cache_control` markers and a conversation-fingerprinted attribution block as the first `system` entry. Claude Code aborts a stream silent for 300 s, reads integer `retry-after` and stops retrying above 60 s, reads `x-should-retry`, and matches error wording to recover from capability rejections. Background requests use the main model unless `ANTHROPIC_DEFAULT_HAIKU_MODEL` pins one, except that a credential supplied through `ANTHROPIC_API_KEY` with `ANTHROPIC_AUTH_TOKEN` unset can send them to the default Haiku ID; the pinned profile uses `ANTHROPIC_AUTH_TOKEN` or pins the Haiku variable. |
| Cursor | Target the OpenAI-compatible chat path, pending an actual custom-endpoint capture. | Official BYOK docs cover chat models, exclude Tab, and say requests pass through Cursor servers. They do not specify a complete custom-endpoint wire contract. Arbitrary local-model Agent compatibility remains unverified. |

Sources: [OpenCode providers](https://opencode.ai/docs/providers/),
[OpenCode v2 providers](https://opencode.ai/v2/docs/providers),
[Codex configuration](https://learn.chatgpt.com/docs/config-file/config-reference),
[Codex custom providers](https://learn.chatgpt.com/docs/config-file/config-advanced),
[Ollama's Codex guide](https://docs.ollama.com/integrations/codex),
[Claude Code connection](https://code.claude.com/docs/en/llm-gateway-connect),
[Claude Code gateway protocol](https://code.claude.com/docs/en/llm-gateway-protocol),
[Cursor BYOK](https://cursor.com/help/models-and-usage/api-keys).

Cursor's server-mediated path implies that a workstation loopback address is
not a sufficient deployment route. Keep D-014's local default; any reachable
remote deployment requires explicit configuration and protected transport,
with authentication optional there too (D-014's owner note, 2026-09-28).
This check does not authorize publishing a server or sending prompts to
Cursor. Cursor remains a named target with an M5 validation gap, not a
reason to claim all four clients already work.

## M5 surface

All routes terminate at the single inference front door (D-045 defines its
listener, authentication and CORS defaults). This is a bounded compatibility
subset, not a clone of either hosted platform. The following choices are
llmpalooza requirements informed by the linked protocol references.

| Route | Baseline behavior |
| --- | --- |
| `GET /v1/models`, `GET /v1/models/{id}` | List of configured, supported model IDs, including unloaded models, in the OpenAI shape by default and in the Anthropic list shape when the request carries `anthropic-version` or `x-api-key`. Claude Code's opt-in discovery sends `x-api-key` only when an API key or helper is configured and a bare bearer `Authorization` otherwise; both shapes carry `data[].id`, `display_name` and `description`, which is all Claude Code reads, so a missed discriminator cannot break discovery. Answered from the catalog with no I/O and no redirect, inside Claude Code's 3 s bound. Entries carry `display_name` and `description` naming the actual model behind an alias. Listing does not load weights or promise admission. Entries carry the D-046 OpenRouter metadata fields (`context_length`, `architecture`, `top_provider.max_completion_tokens`, `supported_parameters`, `default_parameters`, `per_request_limits`, `hugging_face_id`) with values from the artifact, configured limits and the implemented profile only; pricing and uptime are omitted. |
| `POST /v1/chat/completions` | Text conversations, function tools and tool results, JSON responses and SSE streaming. Image, audio and video content parts arrive on M3's chat route in M3.5 (D-101) and carry over. |
| `POST /v1/responses` | Text input/history, instructions, function calls/results, JSON responses and SSE streaming; stateless full-history operation. `input_image` parts for models validated for images (D-101). |
| `POST /v1/messages` | Text/system content blocks, tools/results, JSON responses and SSE streaming. Honor supported version semantics and handle the beta query independently of routing. `image` blocks for models validated for images (D-101). |
| `POST /v1/messages/count_tokens` | Count supported input using the selected model's actual tokenizer and prompt rendering, including system/tool overhead. No decode or weight residency is needed. Counts describe the local model, not Claude tokenization. |
| Gemini API: `GET /v1beta/models`, `POST /v1beta/models/{model}:generateContent`, `:streamGenerateContent?alt=sse`, `:countTokens` | Added by D-101 for Gemini CLI and the google-genai SDKs. Text and tool turns, image `inlineData` in and out. Gemini CLI requires HTTPS off loopback. |
| Fill-in-the-middle: `suffix` on `POST /v1/completions`, `POST /v1/fim/completions`, `POST /infill` | Added by D-101 for code-completion clients (Continue, Tabby, Twinny, llama.vscode, Zed). Only on models whose tokenizers carry FIM tokens; others refuse explicitly. |

Token counting follows the [Messages count contract](https://platform.claude.com/docs/en/api/messages/count_tokens).
Its inclusion is our usability choice; Claude Code can estimate when absent.
Unknown model IDs fail explicitly; never silently map a Claude/OpenAI name to a
different model. Owner-configured aliases identify their actual model, and the
response echoes the requested alias while the resolved artifact identity
travels in an extension header (D-045). Aliases meant to appear in Claude
Code's picker contain `claude`, because its discovery filter drops the rest.

### Chat Completions

Preserve system/developer/user roles, ordered history, tool-call IDs and
tool-result links. Keep system instructions separate through normalization and
model-template rendering; include role/template identity in cache matching.
Return indexed `choices`, assistant content or `tool_calls`, appropriate
`finish_reason` and truthful usage. Streaming uses `chat.completion.chunk`
objects with deltas, indexed tool-call argument fragments, a finishing chunk,
optional requested usage and the `[DONE]` terminator. Send the first chunk
with the assistant role delta as soon as the request is admitted, and SSE
comment lines as keepalives while weights load or prefill runs (D-045). After
headers, a failure is a data event carrying an `error` object followed by
termination without `[DONE]`. A partial argument fragment is not yet a
complete JSON object. Implement and test the selected model's tool template;
an HTTP-shaped response alone does not prove tool use. Reasoning output on
this route uses `reasoning` for both current vLLM and OpenRouter; only a
pinned legacy profile may use `reasoning_content`. OpenRouter also uses
`reasoning_details`, with llmpalooza-signed blocks carrying `format: "unknown"`
and llmpalooza identity/version inside their opaque signatures (D-047). The
version is that of the signature's representation, an integer independent
of the product version (D-062; version 1, `src/base/surface_versions.h`),
whose contents M5 fixes. A signature of a version the runtime does not
implement is one it cannot verify. D-043
owns the contract; D-046 adds the `reasoning` request object and
`cached_tokens`/`cache_write_tokens` usage reporting. OpenRouter's
`transforms`, `plugins` and routing suffixes are rejected explicitly, never
ignored (D-046).
See the [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

### Responses

Keep output item IDs distinct from `call_id`; associate each
`function_call_output` with its call. Support function definitions and choices
used by the pinned client profile, including multiple calls and result turns.
Custom/free-form tools are separate protocol types: either implement the
profile's types or configure a verified function-only profile; do not relabel
or silently drop them. See [function calling](https://developers.openai.com/api/docs/guides/function-calling).

For SSE streaming requests, emit typed lifecycle events (`response.created`, `response.in_progress`)
immediately on admission, ordered output-item/content-part events, text or
function-argument delta/done events, and the appropriate completed/incomplete/
failed terminal response with usage. SSE comment lines serve as keepalives
between lifecycle and delta events (D-045). Do not translate Chat Completions
by merely changing the URL, or signal Responses completion with `[DONE]`
alone. See [streaming Responses](https://developers.openai.com/api/docs/guides/streaming-responses).

Statelessness (D-047): accept `store: false`; omission means false in this
profile. Reject explicit `store: true`, non-boolean values, non-null
`previous_response_id` and conversation references with a protocol-shaped
400 before admission. Nothing is retrievable afterwards. Accept `reasoning` input items that llmpalooza produced, or
empty ones, on later turns; encrypted reasoning continuity, if ever offered,
uses llmpalooza's own opaque blobs and never a provider-shaped imitation.
`prompt_cache_key` is an advisory retention hint (D-045). M5 does not promise
hosted tools, background jobs, stored-response retrieval, remote compaction or
WebSockets. Reject unsupported semantic requests explicitly. In particular, if
a pinned Codex build requires compaction or a tool variant that cannot be
disabled through supported configuration, that profile fails compatibility
until implemented or D-040 is amended. Stateless wire operation still permits
bounded internal prefix/continuation reuse (D-031).

### Messages

Keep top-level system content separate from message roles. Preserve ordered
text and `tool_use` blocks and user `tool_result` blocks linked through
`tool_use_id`; return correct stop reasons such as `end_turn`, `tool_use` and
`max_tokens`, and truthful usage. See the
[Messages reference](https://platform.claude.com/docs/en/api/messages/create).

SSE uses `message_start`, indexed content-block start/delta/stop events,
`message_delta`, and `message_stop`. Send `message_start` as soon as the
request is admitted, since its input count needs only the tokenizer, then
`ping` events during long silent intervals including model switches (D-045);
support in-stream errors and never disguise an error as successful completion.
Tool inputs stream as `input_json_delta` fragments. See
[Messages streaming](https://platform.claude.com/docs/en/build-with-claude/streaming).

Accept configured gateway credentials through bearer or `x-api-key` auth;
validate `anthropic-version`. Bound headers, including comma-separated beta
values. Llmpalooza terminates inference rather than forwarding to Claude: it must
implement, explicitly reject, or document safe advisory handling for each
profile's beta/body feature. Disable unsupported client features through
documented settings and verify the actual requests. Pin main and auxiliary
model choices and real context limits in that profile.

Fields Claude Code sends by default to an unrecognized alias, and their
handling:

- `thinking` with `type: adaptive`: honor on models whose reasoning contract
  (D-043) supports it; otherwise reject with a 400 that names the `thinking`
  field, which the client documents as its trigger to disable thinking for
  the conversation. Llmpalooza signs its own thinking blocks; a block it cannot
  verify is rejected with the documented `bound to a different conversation`
  wording so the client drops earlier thinking blocks and retries. Never
  fabricate a provider's signature or accept one blindly.
- `context_management` and its beta header, and beta tool-schema fields: the
  client does not retry these rejections, so the profile's default
  configuration must either implement the requested strategy or document
  `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` as part of the pinned setup.
- `output_config` effort and structured output: honor where the model and
  D-043 contracts support it; otherwise reject naming the field.
- Context exhaustion: a 400 in the documented prompt-too-long form, or one
  carrying the gateway doc's stable `capability_rejected:` token, so the
  client's reactive compaction runs instead of a dead end.
- `cache_control` markers: advisory retention hints, never identity or a
  retention grant (D-045); usage reports real cache reads only, and writes
  as tokens submitted for retention (D-055).
- The attribution block: stripped when it arrives unchanged as the first
  `system` entry, matching the behavior the client documents for the
  first-party endpoint, so shared-prefix identity excludes its per-conversation
  fingerprint (D-045). Only an entry consisting solely of the block is
  stripped; an entry that merges it with other text is left intact so no
  user system content is dropped. Its contents are never logged.
- `x-claude-code-request-class` and `x-claude-code-context-compacted`:
  scheduling and release signals under D-045, present only when the pinned
  profile sets `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`; absence means default
  policy. The compaction request itself is interactive work.
- Background and auxiliary requests: alias them to the main model by default;
  a different model turns every side request into a switch. The profile
  records the choice and acceptance counts switches caused by side requests.

Error bodies are part of this contract: the client matches wording to decide
whether to retry and what to disable, so keep the protocol's error shape and
the documented phrases, and set `retry-after` and `x-should-retry` per D-045.

## Front door, timing and admission (D-045)

**Listeners, auth and CORS.** One inference front door per conductor serves
`/v1/*`, the Ollama `/api/*` profile and read-only discovery on one
configurable port; management is a separate listener, local-only by default
(D-014). A loopback-bound front door accepts requests without a credential
until an inference credential is configured, because Ollama-native clients
send none, and while anonymous access is on a presented credential is ignored
rather than checked, because OpenAI SDK clients send a placeholder. No
binding requires a credential (D-014's owner note, 2026-09-28, amending
D-045): an inference credential is an optional feature, as llama-server's
and vLLM's API keys are, never a precondition for a binding. Once a
credential is required, a request carrying both `Authorization` and
`x-api-key` must validate on each. Authorization is decided per operation,
never by path prefix, and an inference credential never carries management
authority. Cross-origin requests are allowed from loopback origins by default,
matching Ollama, and from a configured origin list otherwise; a request whose
`Origin` is outside the list is refused before any work, JSON routes require
`Content-Type: application/json` so a browser cannot trigger inference or a
release with a no-preflight request, and on a loopback binding the `Host`
header must name a loopback address, the machine's hostname or a configured
name, the DNS-rebinding guard Ollama applies. A wildcard origin is accepted
only on a loopback binding with a credential configured; anonymous plus
any-origin would let any web page drive local inference. M3's chat route
listens on loopback and the tailnet by default, without a credential, its
Host guard accepting the node's tailnet addresses and MagicDNS names, and
on any address configured explicitly, also without one (D-097).

**Time to first byte and keepalives.** A model switch can take many seconds
(D-036), so for requests selecting SSE streaming the front door sends response
headers and the first protocol event as soon as the request is validated and
admitted, before weights load, then keepalives at a pinned interval:
`ping` events on Messages, SSE comment
lines on Chat Completions and Responses. Ollama's NDJSON has no keepalive
frame; that profile's clients tolerate load time by design and the tested
profile records the bound. Long switches are never signalled through
`retry-after`.

M3's chat route does this for Chat Completions now (D-097): `: keepalive`
comment lines every 15 s without output, from the headers on, through the
queue, a swap and prefill. A stream that waits 15 s in the queue is
admitted then, with its headers and role chunk, so the queue counts as
admission for it; a stream admitted within 15 s keeps the 400s before
headers. Since D-102 (2026-10-03) the route's queue has no count and no
wait by default, so it sends no 429; an owner who sets `[client]
max_queued` or `queue_wait_seconds` gets the 429 below for those limits.
A stream whose client stops reading pauses at its next completed step
(backpressure) instead of being cut off.

Non-streaming requests, including `stream: false`, receive one JSON result
or protocol-shaped JSON error; hold headers until the outcome is known and
never send SSE keepalives. Pin and test each client's non-streaming timeout
bound separately. Document when long switches exceed that bound instead of
silently enabling streaming. A server request deadline, where one is
configured, produces a protocol-shaped 504 before headers and initiates
completion-safe cancellation; disconnects also initiate cancellation, never
premature backing reclamation. Pre-admission queue expiry, where configured,
remains 429. By default (D-102) neither applies: a non-streaming request
waits its turn and runs to its end, its client's own timeout the only
clock. A timeout does not authorize automatic replay or promise
exactly-once inference (D-047).

Documented streaming client bounds the profiles must stay inside:

| Client | Documented bound | Consequence |
| --- | --- | --- |
| Claude Code | Aborts a stream silent for 300 s; pings and comment lines count as traffic | Keepalive interval well inside that, including before the first token |
| Claude Code | `retry-after` above 60 s stops retries; integer seconds only | Admit and stream through a switch; 429/503 carry `retry-after` of at most 60 |
| Claude Code | Model discovery times out after 3 s and fails on any redirect | `/v1/models` answers from the catalog at the exact base URL |
| Codex | Stream idle timeout 300 s by default; 4 request retries, 5 stream retries | Same keepalive rule; the duplicate-retry policy must expect several re-attempts of one turn |

**Admission outcomes and HTTP status.** Before headers are sent:

| Outcome | Response |
| --- | --- |
| Malformed request, unsupported semantic feature, context exhaustion | 400 in the protocol's error shape, using documented phrases where a client acts on them |
| Missing or invalid credential; operation not permitted | 401 / 403 |
| Unknown model ID | 404 in the protocol's shape (`model_not_found`, `not_found_error`, Ollama's `model 'x' not found`) |
| Known model with no prepared artifact on any node | 503 with `x-should-retry: false`; the message names the install job route; inference never downloads |
| Oversized request or headers | 413 before expensive work |
| Queue full, maximum queue wait exceeded, over budget now | 429 with integer `retry-after` of at most 60 and `x-should-retry: true`; only for limits an owner configured (`[client] max_queued`, `queue_wait_seconds`; none by default, D-102), and the message names the key |
| Overloaded or draining | 503 with `retry-after` of at most 60 (a request's memory past what the request memory has free now, too many connections for the descriptors, the runtime stopping); a request larger than the whole request memory is a 413 |
| Switch, warm or prefill in progress | Not an error: for SSE, admit, send headers and the first event, keep alive; for non-streaming, await the JSON outcome (within a configured request deadline, D-047; none by default, D-102) |
| Non-streaming server request deadline expires | Only with `[client] deadline_cap_seconds` set (D-102): 504 in the protocol's JSON error shape before headers; initiate completion-safe cancellation (D-047) |

For streaming responses, after headers, failures use the protocol's in-stream error form and terminate
without a success marker. Retried requests are not proof that an earlier
attempt stopped; the duplicate policy in the shared section applies.

**Advisory signals from standard clients.** These steer policy and never
identify a conversation or grant retention (D-031):

- `x-claude-code-request-class`: `main`, `subagent`, `workflow` and
  `compaction` are interactive; `auxiliary` is background (D-042 priority).
  Both this and the next header are opt-in hint headers (see the client
  table); absence means default policy.
- `x-claude-code-context-compacted`: drop retention of the prior continuation.
  The pre-compaction prefix is no longer in the request, so the continuation
  is located by affinity through the `x-claude-code-session-id` and
  `x-claude-code-agent-id` values Claude Code sends on every request, within
  the caller's authorized scope; with no match the signal is a no-op. This is
  release, not erasure; shared-prefix retention and admitted work are
  untouched (D-041 close semantics), so a forged signal costs at most a
  recomputation.
- `cache_control`, `prompt_cache_key` and OpenRouter's `session_id`, `user`
  and `metadata` (D-046): retention, affinity and attribution preferences,
  never authorization.
- Ollama `keep_alive`: `0` releases the requester's own residency lease once
  its request completes, which changes eligibility, not residency (D-007), and
  never touches another consumer's lease or admitted work; a positive duration
  is an advisory retention preference; a negative value asks for an indefinite
  pin the budget cannot honor and is rejected explicitly.

D-055's [retention policy](retention-policy.md) fixes how these signals act
on retained state: breakpoints nominate shared-prefix boundaries only inside
the leading system/tool segment, hosted-cache TTLs are ignored, and
`keep_alive` concerns weight residency, not conversation state.

**Alias echo and extensions.** The response `model` field echoes the
requested alias exactly, and the resolved immutable artifact identity travels
in a llmpalooza response header and in native discovery and diagnostics. Llmpalooza
extensions travel as namespaced request and response headers on every
protocol, and as namespaced body fields only where the protocol tolerates
unknown keys. D-062 sets the lowercase `llmp-` header prefix and a single
top-level `llmp` body object; M5 fixes the individual names.

## Shared correctness and limits

M3's minimal endpoint fixed its bounds in
[runtime-serving.md](runtime-serving.md#the-chat-route) (D-097), since
derived from real resources (D-102: memory, descriptors, the model's
context, a parser's stack; counts and times that bounded only abuse are
gone or owner options); M5's front door extends them to its routes the
same way. Before M5's front door accepts external input, specify the bound
and the resource it protects for request/header bytes, history, tools,
argument size, output, queued requests and stream buffers. Reject
oversized or unsupported input before expensive work. Test
malformed JSON, invalid tool links, unsupported modalities and unavailable
models. No prompt/tool payload logging by default. Usage and cache accounting
must reflect actual work; cache hints do not grant retention or identify a
conversation. Harmless metadata may be ignored only under a documented rule.
M3's route documents one (D-097): a field it does not know is ignored and
its name counted, never its value; a field it knows but does not
implement is refused whenever honoring it would change the answer.

Use protocol-shaped non-success errors before streaming; after streaming has
started use its error/failure mechanism and terminate without a success marker.
Client disconnect requests cancellation; release backing only after device/I/O
completion. Slow readers must not grow buffers without bound or hold the
catalog lock across waits. Keepalives indicate liveness, not inference progress.
Do not silently replay an uncertain execution when a client or node disconnects
(D-037). Retried requests are not proof that an earlier attempt stopped.

## Acceptance owed in M5

1. Pin each client build, provider/SDK version where applicable, model artifact,
   tokenizer/template and complete configuration; keep credentials out of the
   report. Record supported and disabled features. Begin with OpenCode's chat
   profile, then Codex's Responses and Claude Code's Messages profiles.
2. Run text chat, incremental streaming, cancellation, and a complete
   client-executed tool round trip through each claimed profile. Exercise two
   calls, fragmented JSON/UTF-8, long arguments, token limits, error paths,
   context exhaustion/compaction, auxiliary requests and model selection.
   Use opt-in synthetic traffic captures outside Git; retain aggregate evidence.
3. Test each route's JSON/SSE schema independently even if only one named client
   is needed for the existing M5 end-to-end gate. A passing client does not
   certify another; publish a per-version/profile support matrix.
4. Resolve Cursor's custom endpoint, model selection, exact routes, streaming,
   tools and reachability with an owner-enabled test deployment. Until then,
   retain the explicit compatibility gap. Do not claim Tab support.
5. Challenge disconnects during tool emission, duplicate retries, slow readers,
   admission failure after headers, unknown beta fields and oversized inputs.
   Add M6's A→B→A trace through a validated client, without session extensions.
6. Verify the D-045/D-047 contract: discovery answers inside the client bound with no
   I/O; an induced switch longer than the keepalive interval completes through
   each streaming protocol with a defined keepalive. Non-streaming calls
   through each route return one valid
   JSON result after a switch, or a JSON 504 on server deadline expiry, without
   SSE bytes or premature success headers. Verify safe cancellation on deadline
   and disconnect. Each status-table row produces the documented client
   behavior, including Claude Code's thinking, signature and prompt-too-long
   recovery paths; side requests cause no extra switches under the default
   alias; a named client works on loopback without a credential, ignoring a
   placeholder one; cross-origin requests from a non-listed origin are
   refused. Two checks wait for their scope: in M6, with D-041's close,
   `x-claude-code-context-compacted` (hint headers enabled in the pinned
   profile) releases the prior continuation while a shared prefix stays
   reusable; in M10, with the Ollama profile, an Ollama-native client works
   on loopback without a credential and its separate load-time bound holds.

7. Verify D-047 storage validation: omitted/false `store` succeeds with full
   history, while true, non-boolean values and non-null continuation references
   fail before admission. When reasoning support lands (M7), test current vLLM's
   `reasoning` and any separately advertised legacy spelling, plus the pinned
   OpenRouter SDK's preservation of `format: "unknown"` signed blocks through
   streamed tool calls and subsequent tool-result requests. Check text,
   signatures, ordering and indices, not just visible final answers.

Provenance: written and reviewed 2026-09-22 on the x86-64 workstation from
live documentation; no runtime, client or Spark test was possible because no
endpoint exists. Handoff and review notes travel with the commit (workflow.md).
