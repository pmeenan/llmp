<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# API capability assessment: Ollama, lifecycle and cluster sharing

Owner-requested assessment, 2026-09-22, building on the
[D-040 client baseline](client-api-baseline.md). Status: D-041 accepted the
Ollama listing/details/chat/generation subset, discovery (M5, cluster
availability in M6a), targeted continuation close (M6) and download/warm jobs;
D-042 accepted text resources, images and audio files, an optional MCP
management adapter, application permissions, priorities, queue waits,
cancellation and progress events, and later embeddings. Ollama registry and
other management compatibility, live audio/video and batch/background
inference are deferred behind concrete triggers (plan.md). D-045 settles the
front-door listener, auth and CORS defaults, the `keep_alive` mapping and the
Ollama liveness/version routes. The [vLLM assessment](vllm-api-assessment.md)
and the [OpenRouter assessment](openrouter-api-assessment.md) are separate.
Detailed mechanisms below are design guidance: exact paths, schemas and
numeric limits are still to specify, and nothing here claims implementation
or client compatibility. Delivery milestones are in
[plan.md's milestone ladder](plan.md#milestone-ladder).

## Recommendation

Keep the three accepted inference protocols and add a versioned llmpalooza
management/discovery surface. An optional Ollama-compatible subset is useful
for local clients that speak its native API; it does not replace Responses or
Messages. Start with catalog queries and text inference, adding lifecycle and
registry operations only when their semantics can be fulfilled honestly. No
Ollama runtime dependency is needed: adapters call the same native scheduler.
Test a named Ollama client before claiming compatibility; an endpoint inventory
alone does not establish that its startup and model-management flows work.

## What Ollama offers and how it maps

| Ollama surface | Fit for llmpalooza |
| --- | --- |
| `GET /api/tags` | Available-model catalog; maps to prepared, supported models, whether resident or not. Keep downloading/import-only entries in management, outside the selectable inference list. |
| `POST /api/show` | Model metadata and capabilities; useful alongside a richer native description. Only expose truthful metadata; do not invent an Ollama Modelfile or raw-file digest for a repacked artifact. |
| `GET /api/ps` | Running/loaded-model view. Partial residency and replicas do not have a clean one-row, one-expiry mapping. Defer this route until tested field meanings exist. |
| `POST /api/chat` | Good candidate: model selection, role-labelled history, tools and JSON/streaming output. |
| `POST /api/generate` | Useful for simpler clients: prompt plus separate `system` and optional images. Raw/template/options variants need explicit support or rejection. |
| Empty chat/generate request | Ollama preloads a model. A llmpalooza warm operation can prepare a supported execution configuration, subject to admission. Do not claim full residency if it only warmed metadata or some extents. |
| `keep_alive` | Ollama model-residency duration: default 5 minutes, zero unloads, negative pins indefinitely. Not a conversation-end signal. D-045 maps it faithfully: `0` releases the requester's own residency lease after its request (eligibility, not eviction, D-007; other consumers and admitted work untouched), a positive duration is an advisory retention preference, and a negative value is rejected explicitly because the node budget cannot honor an indefinite pin. |
| `POST /api/pull` | Registry download with resumable progress. Useful later, but success must also await llmpalooza validation/repacking and atomic publication of an executable artifact. |
| `POST /api/embed` | Worth considering for retrieval workloads; requires a validated embedding model/output contract. Not supplied by merely wrapping chat inference. |
| `GET /api/version`, `GET /` | Ollama's CLI and common clients probe these at startup: `/api/version` returns `{"version": ...}` and the root returns the `Ollama is running` text on `GET` and `HEAD` ([server source](https://github.com/ollama/ollama/blob/main/server/routes.go)). Served only with the profile enabled (D-045): `version` carries the Ollama release the profile was tested against, as an API compatibility level, alongside a `llmp` field with the real server version. Never report a release the profile was not tested against. |
| Create/blob upload/copy/delete/push | Not in the recommended first subset. Mutation and registry semantics expand the surface substantially. |

Official sources checked: [tags](https://docs.ollama.com/api/tags),
[ps](https://docs.ollama.com/api/ps), [chat](https://docs.ollama.com/api/chat),
[generate](https://docs.ollama.com/api/generate),
[preloading and retention](https://docs.ollama.com/faq),
[embeddings](https://docs.ollama.com/api/embed).
For show/pull and the broader endpoint inventory, the new documentation pages
were unavailable/incomplete in this check; used Ollama's still-published
[repository API reference](https://github.com/ollama/ollama/blob/main/docs/api.md#show-model-information)
([pull section](https://github.com/ollama/ollama/blob/main/docs/api.md#pull-a-model)).
These are live sources; pin a server/client reference at implementation.

Ollama uses newline-delimited JSON (`application/x-ndjson`), unlike the SSE
streams in D-040. Its adapter needs its own framing, terminal records, errors
and tool parsing. See [streaming](https://docs.ollama.com/api/streaming).
Its API is not strictly versioned, making a tested compatibility profile more
useful than a blanket claim; see [introduction](https://docs.ollama.com/api/introduction).

The difficult part is semantics, not routing. Common clients send a positive
`keep_alive` by default, so the profile honors the D-045 mapping rather than
rejecting the field: zero releases, positive durations are preferences, only
the indefinite pin is refused, and default residency may differ from Ollama.
That is a documented partial compatibility profile, not full Ollama lifecycle
compatibility. If a chosen client depends on preload/unload/pinning, it does
not pass until those operations have faithful, completion-safe behavior.
Indefinite pinning cannot bypass the node budget. Whole-model unloading must be
an explicit authorized operation, never the default eviction policy, and
cannot destroy admitted work.

Deployment shape under D-045: the profile shares the inference front door and
its configurable port; llmpalooza does not claim 11434 by default because Ollama
may be installed, and Ollama-native clients are pointed at the front door with
`OLLAMA_HOST`. Loopback requests need no credential until one is configured,
and cross-origin requests are allowed from loopback origins by default, with a
configured list as the equivalent of Ollama's `OLLAMA_ORIGINS` for anything
else and the same loopback `Host` check Ollama applies. Codex reaches Ollama
through `/v1/responses` on that port, so the Responses profile covers Codex;
Ollama-native routes are not required for it.

## Requested capabilities

| Request | Assessment and delivery recommendation |
| --- | --- |
| Model list | Already in D-040 for M5. Add native catalog details: immutable artifact ID, display alias, support level, capabilities, context/output limits and nodes where prepared artifacts exist. D-046 puts the client-facing subset in OpenRouter's `/v1/models` metadata shape. |
| Model selection | Already confirmed: the request's `model` selects it. No global selected-model variable shared by clients. Resolve aliases to immutable identity for the attempt; echo the requested alias in the response `model` field and report the resolved identity in the extension header and diagnostics (D-045), so strict clients keep working. |
| Activate an inactive model | Already confirmed on first request. Recommend an explicit asynchronous warm operation as the API form of the existing warm hint; define what was warmed, TTL and status. Availability, residency and admission are separate. |
| Download/install | HF download/import is already confirmed. Expose a job with download → verify → repack → publish stages, progress, cancel/resume and failure reasons. Ollama registry ingestion would be a separate proposed source adapter. |
| Separate system prompt | Already supported structurally by D-040: system/developer roles in Chat Completions, instructions and role-labelled inputs in Responses, top-level system in Messages. Preserve roles in normalization/template rendering and cache identity. |
| End conversation | Optional session/release is already confirmed. Recommend both a final-request flag for a named continuation and an idempotent explicit release operation. Claude Code already sends `x-claude-code-context-compacted` after a compaction, which D-045 accepts as an advisory release of the prior continuation. See semantics below. |
| Multimodal inputs | Interpret the owner's “multi-model inputs” as text/resources, images, audio and video. Typed content is sensible; execution support must be earned per modality/model, not inferred from a model family name. Stage text/resources, then images, then audio files; real-time audio/video are separate projects. |
| API discovery | Recommend a machine-readable API schema plus a runtime capability document; a model list alone cannot describe operations or limits. |
| MCP | Recommend an optional external management adapter after that API settles. Expose catalog/capacity/job information and narrowly scoped actions; keep ordinary inference and client tool execution in their existing roles. |

### Lifecycle without cache confusion

Use a server-issued opaque continuation handle, bound to the caller's authorized
scope, branch and generation, for targeted release. Exact wire names remain
proposed. An end flag means “do not retain this continuation after this final
request,” not “unload these weights” or “delete every matching prefix.” A
request-scoped no-retain hint without a handle can govern newly produced
private state, but cannot identify older continuations to delete.

A close must serialize with admission on that handle: stop accepting new work
for the closing generation, allow already-admitted work to complete/cancel
safely, then drop the continuation's retention references and spill records.
Late completions cannot recreate a closed continuation. Repeating close is
idempotent; a new branch needs a new identity. Disconnect is not proof of
conversation end. Return closing/completed status without claiming immediate
physical reclamation. Shared-prefix entries retain their independent policy;
release is not secure erasure and does not imply shared bytes disappeared.
Test two conversations with identical system prompts, branches, close racing
with a queued turn, and delayed completion after close (D-007/D-031).

Standard clients already emit release-adjacent signals that need no handle:
Claude Code's `x-claude-code-context-compacted` marks the first request after
a compaction, when the pre-compaction prefix is dead. D-045 treats it as an
advisory release of the prior continuation, located by affinity through the
always-sent session and agent ID headers, with the same serialization,
shared-prefix independence and no-erasure caveats as an explicit close. Its
`x-claude-code-request-class` values feed the D-042 priority classes. Both
are opt-in hint headers that the pinned profile enables; neither identifies a
conversation for reuse or authorization.

### Install and activation

Inference selects an installed artifact; a miss returns a useful unavailable
error, never silently downloads a large model. Downloads require management
authority, bounded storage/temp space and verified source identity. Pin the
resolved source revision/digests; verify every staged object and publish only
complete supported artifacts. A successful download is not successful install.
Cancellation of a shared download detaches that job's interest, not other users'
work. Crash/retry must not publish incomplete artifacts or duplicate unbounded
jobs. Never execute checkpoint code or accept arbitrary server filesystem paths.

For a cluster, report per-node artifact availability and replication-job state.
An artifact on one node is not installed everywhere. The conductor can route to
that node; a transfer/import operation is explicit, budgeted and verified.
Warm requests are speculative and subordinate to admitted inference, with
bounded concurrency and cancellation. Read-only list/discovery does no loading.

### Multimodal scope

Ollama documents images alongside text for vision-capable models
([vision](https://docs.ollama.com/capabilities/vision)); the inspected native
chat/generate docs do not establish a general audio-file, live-audio or video
contract. Ollama compatibility therefore does not settle the whole request.

Prefer ordered typed content parts, preserving system versus user/resource
provenance. For resources, support bounded inline text and later uploaded,
caller-scoped resource handles with content hashes, MIME types and expiry.
A resource reference is data, not authorization to fetch arbitrary URLs or read
local paths. Parsing documents/OCR is additional capability, not implicit text
input. Account for upload storage, decoded pixels, media duration, preprocessing,
encoder/workspace/state and generated modality tokens in admission. Cache
identity includes content hashes and preprocessing/template versions. Keep
encoders behind native operation contracts; no interpreter enters serving.

Images need encoder/projector import and numerical tests. Audio files need
explicit tasks (transcription versus audio reasoning), formats and limits.
Streaming audio additionally needs input framing, timestamps, backpressure,
interruption/cancellation and bounded live state. Video needs timestamps,
sampling/ordering policy and audio/video synchronization. Extracting frames or
transcribing audio is an explicit lossy transformation, never silently called
native video/audio support. Advertise input and output modalities separately;
this request does not implicitly add image/audio/video generation (D-101
adds it explicitly).

### Discovery and MCP

Recommend OpenAPI/JSON Schema for native request/response structures plus
runtime discovery listing implemented protocol profiles, API/schema versions,
auth requirements, extensions, limits and per-model capabilities (tools,
structured output, modalities, context, quantization and support evidence).
Separate static support from changing health/capacity. Include freshness and
unknown/stale status; discovery is not admission or automatic cluster enrollment.
Filter it by caller authority and avoid exposing secrets or private model paths.

MCP offers [discoverable tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
and [resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources).
A proposed llmpalooza MCP server could expose model/capacity resources and tools
for explain-admission, warming, import-job status and continuation release.
It would be a separate process over the native management API (D-005), with
read-only access by default and explicitly granted mutation authority. Schema
annotations are not authorization. Tool calls cannot elevate an inference key
to install/delete/control privileges; returned model metadata is untrusted data.

This differs from llmpalooza acting as an MCP client that executes arbitrary model
requested tools, which is not recommended for the runtime. Existing agent
clients can handle tools and return results through D-040. MCP
[sampling](https://modelcontextprotocol.io/specification/2026-07-28/client/sampling)
is a client-side generation facility; it does not make an inference endpoint an
MCP server automatically. Pin the MCP revision when implementing the adapter;
older lifecycle/negotiation assumptions need not match this checked revision.

## Additional cluster API recommendations

D-042 accepts application permissions, interactive/background priorities, maximum
queue waits, cancellation and bounded progress events. D-041 accepts retry
support for management jobs. Other mechanisms in this table, including capability
filters/alternative-model fallback and affinity controls, remain design suggestions.
If fallback is ever accepted, D-046 reserves OpenRouter's `models` array with
`provider.require_parameters`/`quantizations` as its opt-in spelling.

These serve the existing single-owner/agent workload, not production multitenant
hosting. Such hosting remains a non-goal (vision.md).

| Priority | Proposed exposure | Constraint |
| --- | --- | --- |
| First | Readiness, admission explain/what-if, request/job status, cancellation, queue/load/cache counters | Builds on confirmed diagnostics. Distinguish inference readiness, management liveness and model availability; no fabricated ETA or automatic reservation from a query. |
| First | Per-application credentials and scopes for inference, read-only status and model administration | Prevent accidental cross-application mutation; not a claim of hostile-tenant isolation. Gate detailed cluster state and resource handles too. |
| First | Bounded idempotency keys for install/warm/release operations | Bind to caller plus payload, reject conflicting reuse, publish retention/expiry. Inference needs a separate uncertain-outcome policy; never promise exactly-once replay of an interrupted stream. |
| Next | Interactive/background class, maximum queue wait/deadline, output budget, optional affinity | Scheduler enforces priorities and starvation bounds. Timeout initiates cancellation, not backing reclamation. Affinity is a preference; explicit node constraints may fail admission. |
| Next | Capability filters and explicit fallback policy | Keep exact-model selection as default. Any alternative model/quantization requires client opt-in and reports actual identity; no silent fallback. |
| Next | Progress/events for import, warm, placement, drain and request state | Bounded replay window/cursors, gap indication and snapshot recovery; no prompt contents. Disconnecting an observer does not cancel an independent job. |
| Later | Embeddings, batch/background jobs, repeatable generation configuration | Require concrete workloads, numerical/output contracts and resource budgets before adding endpoints. Tokenizer/template identity improves reproducibility but is not a determinism promise. |

## Staging and validation

D-041 adds discovery to M5, targeted continuation close to M6, and cluster
availability to M6a. The milestone ladder (plan.md, renumbered by D-087)
places import and download jobs in M5; warm jobs, priorities, queue waits, cancellation and
progress events in M6; archive and peer replication in M6a; and the Ollama
subset, text resources, MCP, application permissions and embeddings in M10.
D-101 moves image, video and audio file inputs to M3.5 and M4 with their
model families, and adds media generation routes in M3.5. MCP follows the
native management API.
Live audio/video and batch/background jobs remain deferred until concrete
workloads establish their requirements; earliest M9 planning, not automatic M9
obligations. All accepted scope still needs execution evidence.

Before implementation acceptance, specify numeric bounds and adversarial tests:
repeated/oversized metadata queries, cross-scope handle access, malicious
URLs/redirects/path traversal, decompression bombs, interrupted imports, duplicate
job submission, stale capacity, shared-download cancellation, warm storms,
close/continue races, slow event readers, and failed/cancelled media preprocessing.
Tests must cover the actual backend/catalog lifetimes, not only HTTP schemas.

Provenance: documentation research and triage on the x86-64 workstation,
2026-09-22; no models downloaded, services exposed or runtime APIs
implemented. Reviewed per workflow.md; handoff and review notes travel with
the commit.
