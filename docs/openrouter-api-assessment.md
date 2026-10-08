<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# OpenRouter API compatibility assessment

Checked 2026-09-22 against OpenRouter's live documentation, at the owner's
request after the D-040–D-045 API work. Status: **owner-triaged 2026-09-22,
D-046**: groups 1–3 of the recommendation are accepted as spellings on the
existing OpenAI-shaped routes and group 4 is excluded. This is a public-API
comparison of one hosted aggregator, not a claim that any client works
against llmpalooza; execution evidence is still owed.

## What OpenRouter is, and why it matters here

OpenRouter is a hosted router in front of many providers. Its API is the
OpenAI shape plus a well-known extension vocabulary, and because so many local
and hosted models are reached through it, most agent clients ship an
"OpenRouter" provider mode that understands those extensions. That vocabulary,
not the hosted service, is what could matter to llmpalooza: where it fills a gap
our accepted contracts leave open, adopting the same spelling lets unmodified
clients use the feature (D-043's direct-compatibility rule).

Surface, from the [API overview](https://openrouter.ai/docs/api-reference/overview):
base path `/api/v1`, `POST /chat/completions`, `POST /completions`, a
stateless `POST /responses`, `GET /models` with rich metadata,
`GET /models/{author}/{slug}/endpoints`, `GET /generation?id=` stats,
`GET /key`, plus credits and key-management routes. Clients authenticate with
a bearer key and may send `HTTP-Referer` and `X-Title` attribution headers.
OpenCode lets any provider's `baseURL` be overridden, so its OpenRouter mode
can be pointed at another server
([OpenCode providers](https://opencode.ai/docs/providers/)); other clients
hardcode `openrouter.ai` and are out of reach.

## Fit against accepted contracts

| OpenRouter surface | Fit for llmpalooza |
| --- | --- |
| Chat Completions with normalized `finish_reason` (`stop`, `length`, `tool_calls`, `content_filter`, `error`) plus `native_finish_reason`; a final usage chunk before `[DONE]`; `: OPENROUTER PROCESSING` SSE comment keepalives; mid-stream errors as a chunk carrying an `error` object and `finish_reason: "error"`, then termination. Sources: [streaming](https://openrouter.ai/docs/api-reference/streaming), [errors](https://openrouter.ai/docs/api-reference/errors). | Confirms D-045's SSE-streaming keepalive rule (scoped by D-047) and in-stream error form with a widely deployed precedent. Nothing new to adopt beyond what D-040/D-045 already specify. |
| `GET /models` entries: `id` (`author/slug`), `name`, `description`, `context_length`, `architecture` (`modality`, `input_modalities`, `output_modalities`, `tokenizer`, `instruct_type`), `top_provider` (`context_length`, `max_completion_tokens`, `is_moderated`), `pricing`, `supported_parameters`, `default_parameters`, `per_request_limits`, `hugging_face_id`, `expiration_date`. Source: [list models](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties.md). | **Strongest fit.** D-041 requires per-model capability and limit discovery in M5 and leaves the schema open. This is the most widely parsed capability schema among local-model clients, and OpenAI SDKs ignore the extra keys. Recommend embedding these fields in our OpenAI-shaped `/v1/models` entries: `context_length` and `max_completion_tokens` from the artifact and configured limits, `supported_parameters` derived from the implemented profile only, modalities from validated support, `hugging_face_id` from import provenance, and `pricing` omitted, never invented or reported as zero (D-046). The native discovery document remains the authoritative superset. |
| `GET /models/{author}/{slug}/endpoints`: one entry per serving endpoint with `provider_name`, `context_length`, `max_completion_tokens`, `quantization`, `supported_parameters`, `status`, `uptime_last_30m`. Source: [endpoints](https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model.md). | Natural shape for M6a's cluster availability (D-041): one entry per node or replica holding a prepared artifact, `quantization` from the artifact representation, `status` from admission readiness. Uptime is omitted, never fabricated (D-046). |
| `reasoning` request object (`effort`, `max_tokens`, `exclude`, `enabled`); `reasoning` text and `reasoning_details` array (`reasoning.text` with optional `signature`, `reasoning.summary`, `reasoning.encrypted`, each with `format` and `index`) in messages and stream deltas; clients pass `reasoning_details` back unmodified for tool-call continuity. Source: [reasoning tokens](https://openrouter.ai/docs/use-cases/reasoning-tokens). | Fills D-043's missing Chat Completions spelling: current vLLM and OpenRouter share `reasoning`; OpenRouter adds `reasoning_details`, while `reasoning_content` is a pinned legacy option only (D-047). Recommend accepting the `reasoning` request object as the cross-model control surface, emitting the profile-selected field set, and using `reasoning_details[].signature` with `format: "unknown"` for our own signed reasoning blocks, carrying llmpalooza identity/version inside the opaque signature (D-047); an invented format enum would be discarded by the official SDK. Ordering and immutability rules on pass-back match D-043's parser-state requirement. |
| `cache_control` objects inside OpenAI-shaped content parts; `usage.prompt_tokens_details.cached_tokens` and `cache_write_tokens`. Source: [prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching.md). | Gives the Chat Completions profile a standard place to report reused versus recomputed prefix tokens (D-024 diagnostics) and a standard advisory hint already covered by D-045. Recommend both fields; report real reuse only. |
| `session_id` sticky-routing key; `user` per-end-user identifier; `metadata` (16 keys). Source: [chat completion reference](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion.md). | `session_id` is a standard spelling for D-022's optional session hint: an affinity and retention preference, never conversation identity or a retention guarantee (D-031). `user` can feed D-042's per-application attribution but never authorization. Recommend accepting both as hints. |
| `models` priority array with the served model reported in `model`; `provider` preferences (`order`, `only`, `ignore`, `allow_fallbacks`, `require_parameters`, `quantizations`, `sort`, `max_price`, `zdr`, `data_collection`). Sources: [model fallbacks](https://openrouter.ai/docs/guides/routing/model-fallbacks.md), [provider selection](https://openrouter.ai/docs/guides/routing/provider-selection.md). | `models` is an explicit client opt-in listing acceptable alternatives, which is exactly the form D-042 leaves as a design suggestion: no silent fallback, identity reported. If the owner accepts fallback at all, this is the spelling. `require_parameters` matches our explicit-failure rule and `quantizations` maps to artifact representation filters; `order`, `only`, `sort`, price and data fields have no llmpalooza meaning and would be ignored under a documented rule or rejected. |
| Error envelope `{ "error": { "code", "message", "metadata" } }` with the HTTP status equal to `code`; documented 402 credits, 408 timeout, 502 model down, 503 "no available model provider that meets your routing requirements". Source: [errors](https://openrouter.ai/docs/api-reference/errors). | Compatible with the OpenAI envelope clients already parse. D-045 chose 429 plus `retry-after` for exceeded queue waits rather than 408, and 503 with `x-should-retry: false` for an artifact absent everywhere; no change recommended. |
| Model ID convention `author/slug` with routing suffixes (`:free`, `:nitro`, `:floor`, `:exacto`, `:online`). | Aliases in `author/slug` form match what OpenRouter-mode clients expect and models.dev catalogs use. Suffixes encode hosted-routing choices and should be rejected as unknown models rather than reinterpreted. |
| `usage: { "include": true }`, `cost`, `cost_details`, `openrouter_metadata`, `service_tier`, `GET /generation?id=`, `GET /key`, credits. | Aggregator billing and observability. Omit cost fields rather than report zero as fact. `GET /generation` resembles D-044's authorized per-request diagnostics and could be a later spelling for them; not recommended now. |
| `plugins` (web search, PDF parsing, response healing, auto-router), `transforms` (middle-out compression), `:online`, `prediction`, `stop_server_tools_when`. | Server-side tools and lossy prompt rewrites. Outside accepted scope (vLLM assessment: no server-side agent loop; MCP is management only). `transforms` changes the prompt and must be rejected explicitly, not ignored. |
| Stateless `POST /responses` that rejects `store: true` and `previous_response_id` with 400. Source: [Responses overview](https://openrouter.ai/docs/api_reference/responses/overview.md). | Matches D-040's stateless Responses baseline. D-047 accepts omitted/false `store` (omission means false in this profile) and rejects true, non-boolean values, non-null `previous_response_id` and conversation references before admission; no retrieval is promised. |

## Recommendation

Accepted by the owner in D-046, with wire corrections in D-047. Do not add an "OpenRouter profile" as a
fourth protocol. Adopt OpenRouter's extension vocabulary on the existing
OpenAI-shaped routes where it fills a gap we already own, in three groups:

1. **Model metadata.** Embed the `/models` entry fields and the per-model
   `endpoints` shape in `/v1/models` and the M6a cluster-availability view.
   This is the highest-value item: it turns D-041's discovery requirement into
   a schema clients already read.
2. **Reasoning and cache reporting.** Accept the `reasoning` request object;
   emit `reasoning`/`reasoning_details` with SDK-supported `format: "unknown"`
   for llmpalooza-signed blocks and test preservation on pass-back; reserve
   `reasoning_content` for a pinned legacy profile (D-047). Report
   `cached_tokens`/`cache_write_tokens`.
3. **Hints and opt-in alternatives.** Accept `session_id`, `user` and
   `metadata` as hints; if fallback is ever accepted, use the `models` array
   and `provider.require_parameters`/`quantizations` semantics.

Excluded: plugins, transforms, auto-router, pricing, credits, generation
stats and provider-preference fields with no local meaning. Evidence rule
unchanged: a named client run in OpenRouter mode against a custom base URL
(OpenCode is the candidate) before any OpenRouter-mode compatibility is
claimed, with the exact client version, provider package and configuration
pinned. Honesty rules: `supported_parameters` lists only what the profile
implements; `context_length`, modalities and quantization come from the
artifact and validated support; uptime, pricing and cost are omitted, never
fabricated.

Provenance: documentation-only research on the x86-64 workstation,
2026-09-22; no OpenRouter account, client run or Spark test involved.
