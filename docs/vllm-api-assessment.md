<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# vLLM API gap assessment

Checked 2026-09-22 against official vLLM `latest` documentation. This is a
public HTTP/API-feature comparison, not an audit of every internal Python
method, installed release or optional plugin. Enabled routes depend on model,
runner and server flags. No vLLM server was installed or executed. Recheck and
pin the relevant version before implementing a compatibility profile.

D-043 now accepts the first three additions: compatible tokenization/rendering,
JSON/schema-constrained output with strict tools, and protocol-specific reasoning
output/controls. Direct compatibility with unmodified tooling is the default;
select and test a pinned version/feature profile, with separate llmpalooza extensions.
Regex/grammar extensions are deferred until a client needs them, after validated
JSON/schema support. D-044 completes triage: compatible reranking, monitoring
and raw Completions/token diagnostics are confirmed. LoRA and classification/
reward/pooling are deferred behind concrete workloads; generic worker RPC,
training and split-serving deployment controls are excluded from the client
baseline. Delivery milestones are in [plan.md](plan.md#milestone-ladder). No
vLLM runtime dependency, process architecture or blanket extension
compatibility is approved.

## Surface inventory

The [online-serving index](https://docs.vllm.ai/en/latest/serving/online_serving/)
lists these API families:

| Family | Examples |
| --- | --- |
| Generation | `/v1/completions`, `/v1/chat/completions`, `/v1/chat/completions/batch`, `/v1/responses` plus response retrieval/cancel |
| Anthropic | `/v1/messages`, `/v1/messages/count_tokens` |
| Embedding/scoring | `/v1/embeddings`, `/v2/embed`, `/score`, `/v1/score`, `/rerank`, `/v1/rerank`, `/v2/rerank` |
| Other pooling | `/classify`, `/pooling`, `/generative_scoring` |
| Speech | `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/realtime` |
| Inspection | `/tokenize`, `/detokenize`, `/tokenizer_info`, `/v1/models`, `/version`, `/health`, `/load`, `/metrics` |
| Specialized surfaces | LoRA management, profiling, SageMaker adapters, scale-out/rendering and elastic expert-parallel controls |

This inventory describes available families, not a claim that every loaded
model can execute them. Familiar endpoint names alone are insufficient evidence
of full provider compatibility.

## Useful gaps and recommendations

### 1. Tokenization and prompt inspection — confirmed (D-043)

Implement vLLM-compatible `/tokenize`, `/detokenize`, `/tokenizer_info` and
prompt-rendering endpoints for supported formats, extending the existing
Messages token-count capability. Let tools ask for the actual rendered
prompt size, context limit and remaining output allowance before requesting
inference. Return tokenizer/template identities and distinguish plain-text
counts from full history/system/tools rendering. No weights need to load for
ordinary text tokenization; multimodal processing has its own costs and bounds.

vLLM also offers rendering without inference, including a Responses renderer
that applies the same prompt construction and requires matching model,
tokenizer, template and preprocessing configuration. See
[renderer APIs](https://docs.vllm.ai/en/latest/serving/online_serving/renderer/).
For llmpalooza, recommend an optional prompt preview for caller-supplied content,
not a new distributed rendering service. Never disclose private server prompt
material through an unprivileged preview. Counts do not reserve capacity;
reject stale identity on a subsequent request that requires an exact match.

### 2. Schema-constrained generation — JSON/schema scope confirmed (D-043)

vLLM supports constrained JSON, choices, regex and grammars through its
[structured-output API](https://docs.vllm.ai/en/latest/features/structured_outputs/).
D-040's “JSON responses” means the HTTP response envelope; D-043 separately
adds guarantees for supported generated-content schemas. Likewise, tool
argument parsing alone does not establish strict schema enforcement.

Support standard `response_format` JSON-object/JSON-schema requests, strict
function arguments and vLLM's `structured_outputs.json` form, with a documented
JSON Schema subset. Reject unsupported schema keywords;
do not silently weaken constraints. Discovery should distinguish JSON syntax,
schema enforcement and tool support. Bound schema size/depth, compilation time,
compiled-state cache and per-token overhead. Truncation/cancellation must report
incomplete output, not success against the schema. Defer arbitrary regex/grammar
execution until a concrete client needs it, earliest after validated JSON/schema
support (owner-approved D-043 deferral).

### 3. Reasoning output and controls — confirmed (D-043)

vLLM separates model-generated reasoning from final content, including streaming,
using model-specific parsers; some models support thinking controls/budgets.
See [reasoning outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/).
Implement a per-model/profile compatible contract for distinct reasoning/final/tool output,
supported thinking controls and token accounting. Do not assume that suppressing
reasoning in the response prevents the model from spending tokens on it.
Do not claim uniform effort/budget semantics across model families or fabricate
provider signatures/encrypted reasoning. Parser state and template/option
identity must survive resumptions correctly; reasoning text must never be
mistaken for a tool invocation. This fills in the existing unsupported-feature
policy without accepting every vLLM field. Current vLLM and OpenRouter both
use `reasoning` on Chat Completions; `reasoning_content` is the legacy vLLM
spelling, supported only for a pinned, tested older client profile (D-047).
OpenRouter also uses `reasoning_details`; D-046/D-047 adopt that extension
under the same D-043 contract ([OpenRouter assessment](openrouter-api-assessment.md)).

### 4. Reranking — confirmed later alongside embeddings (D-044)

vLLM exposes pair scoring and reranking with model-dependent scoring backends;
see [scoring APIs](https://docs.vllm.ai/en/latest/models/pooling_models/scoring/).
Embeddings are already confirmed in D-042, but ranking retrieved documents for
a query is a separate useful operation. Implement the existing `/rerank`, `/v1/rerank` and `/v2/rerank` contracts where
supported, with unmodified-client tests, bounded query/documents, a validated
model, stable result indices and explicit truncation and score semantics. Scores are not automatically probabilities or comparable
between models. A model-switching library could use a small embedding model,
a reranker and a generator without requiring the server to own a retrieval DB.
Classification, reward/token-level outputs and generic hidden-state pooling
remain workload-driven deferrals under D-044, earliest M9 planning after
validated base-model execution; not implied by embedding approval.

### 5. Standard metrics and compatible health/load — confirmed (D-044)

vLLM exports Prometheus metrics through `/metrics`; it also documents optional
per-request timings with attribution limitations for multiple sequences/turns.
See [production metrics](https://docs.vllm.ai/en/latest/usage/metrics/) and
[per-request metrics](https://docs.vllm.ai/en/latest/features/per_request_metrics/).
Our diagnostics, status and progress scope already covers the underlying need.
Implement Prometheus `/metrics` and compatible health/load queries, with
bounded model/node labels. Reuse metric names only for matching units, labels
and aggregation semantics; paging-specific metrics remain separate. Distinguish
liveness/readiness from per-model admission availability. Never use prompt text, tool arguments, conversation
IDs or unbounded request IDs as metric labels. Report queue, preparation,
restore, prefill and decode costs with clear units; distinguish measured values
from estimates and missing observations. Per-request details need authorization;
health must not promise that an arbitrary model can currently be admitted.

### 6. Generation diagnostics and raw completions — confirmed (D-044)

vLLM's [OpenAI-compatible surface](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
includes raw completions and additional sampling/token/logprob fields. Recommend
bounded token IDs/logprobs and explicit effective sampling settings for numerical
debugging and evaluation. Seeds must not be presented as cross-backend determinism
guarantees; define whether scores are before or after sampling transforms.
OpenAI-compatible `/v1/completions` and standard log-probability fields are
confirmed, along with bounded vLLM-compatible token diagnostics. Preserve the raw-versus-chat-template distinction. Supporting Ollama
`generate` does not already provide this separate wire contract.

## Already covered or deliberately outside the current baseline

- Chat, Responses, Messages, model listing and tools: D-040 covers the basic
  families, with actual client/model evidence still owed. Stored Responses,
  `previous_response_id` and durable background jobs remain outside that
  stateless baseline; ordinary request cancellation does not require adopting
  response storage. A chat batch request is also not a durable batch job.
- Embeddings, file audio, permissions, priority, queue limits and progress:
  D-042 covers scope, not every API spelling. vLLM's
  [speech APIs](https://docs.vllm.ai/en/latest/serving/online_serving/speech_to_text/)
  provide candidate transcription/translation routes when the audio model/task
  is chosen. Its realtime route is an ASR interface, not evidence for universal
  voice-agent, speech-output or video support. Keep the live-input deferral.
- LoRA: vLLM has dynamic load/unload endpoints, gated by configuration; see
  [LoRA adapters](https://docs.vllm.ai/en/latest/features/lora/). This is a
  deferred library feature under D-044. Revisit when a concrete adapter workload
  needs it, earliest M9 planning after validated base-model execution. Adoption would need
  prepared validated adapters, base/adapter identity in cache keys, capacity
  accounting, and completion-safe replacement; never accept arbitrary paths
  from an inference credential.
- Sleep/wake and collective RPC: vLLM documents development-only HTTP controls
  in [sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode/). Do not
  copy generic worker RPC or model-global cache release into the client API.
  Our targeted continuation release, scheduler-controlled reclamation and
  drain-before-maintenance meet different needs. CPU offload cannot create a
  second physical memory tier on Spark. Admin invalidation must still preserve
  admitted work and outstanding GPU/I/O consumers.
- Split render/derender deployment, token-only scale-out, elastic scaling,
  weight transfer and training integration remain outside accepted scope.
  D-043 does accept compatible prompt-rendering endpoints; it does not adopt
  a split serving pipeline. vLLM's
  [derenderer](https://docs.vllm.ai/en/latest/serving/online_serving/derenderer/)
  handles postprocessing for a split serving pipeline; D-005 still governs our
  own native runtime. No arbitrary caller-provided embedding tensors, serialized
  worker objects or external executable processors should be adopted by analogy.
- MCP: the confirmed adapter is for managing llmpalooza. A server-side agent/tool
  execution loop is a separate feature and remains outside accepted scope.

## Cross-cutting validation

For any adopted route, enforce authorization by operation, including aliases
and nonstandard paths. Test parser/CPU/memory/output bounds before expensive
work, slow readers, cancellations and malformed inputs. Auth must not depend
only on an endpoint having a `/v1` prefix. Schema compilation, tokenization and
media preprocessing consume budget even without GPU inference.

Test identical tokenization/template results between preview and generation,
malicious or unsatisfiable schemas, interrupted structured output, streamed
reasoning/tool delimiters, large rerank document sets, unauthorized diagnostics
and unbounded metric cardinality. Confirm a client/configuration before adding
its route to discovery. Unknown options that affect behavior fail explicitly.

Provenance: documentation-only investigation on the x86-64 workstation,
2026-09-22; no vLLM installation, model download, runtime endpoint or Spark
experiment. Reviewed per workflow.md; handoff and review notes travel with
the commit.
