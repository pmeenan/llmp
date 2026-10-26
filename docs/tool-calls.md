<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tool-call interpretation

The M3.5 implementation connects the protocol-neutral interpreter to Chat
Completions intake, shared rendering, common generation and route events.
A bounded qualification passed on 2026-10-22 (six native Qwen API
transactions and one OpenCode task; [D-097's amendment](decisions.md) and
[the results](experiments/m35-tools/RESULTS.md)); full served-model and client
qualification remains open. `strict: true` is accepted and ignored, as vLLM
does: arguments are not validated against their schema (M7's constrained
output).

`ToolRequest` owns normalized definitions and their JSON documents. Its neutral
choice is auto, none, required or named, with a parallel-call flag. Every
declared definition reaches the template whatever the choice (as OpenAI and
vLLM render them), so a later turn that changes the choice keeps the
conversation's rendered prefix (and its kept state); sampling enforces the
choice. Under `none` a template with a verified tool format renders the tools
and the sampler refuses a call's opener; without one nothing renders. The
template receives each client function
object as sent (no field the client omitted is printed). Duplicate names and
missing required/named declarations fail before generation; both name lookups
and result association are hash-based, linear in the request. Active names must
round-trip exactly through the selected format: delimiter collisions and
stripped names fail before rendering or native work. They must also be text to
the model's tokenizer: names that are not UTF-8, not NFC under an NFC tokenizer,
or that contain a control token's text are refused (400 `invalid_tool_name`).
JSON envelopes retain escaped names; `none` bypasses these checks. Schemas guide
conversion, not argument validation: `type` (a name or list) and, minimally,
the alternatives of `anyOf`/`oneOf` choose JSON for admitted non-string values;
an unconvertible scalar stays a string.
Implicit tagged values use structured scanning only for a leading container
kind admitted by their property schema. Missing or non-container schemas keep
raw strings raw; an explicit JSON encoding always uses JSON syntax.

`ToolInterpreter` consumes **generated output only**, never rendered prompt
text. It emits owned text, reasoning-begin, reasoning-end and complete-call
events. Output that starts with the format's reasoning opener (Gemma 4's
`<|channel>thought`, `<think>`) opens reasoning the template did not prefill,
as the non-tools path does; anywhere else it is text. A completed
call carries its entire validated JSON argument object in one event, ready for
U5's existing route encoder to serialize as one argument fragment. Failed or
unfinished calls never publish partial arguments. A natural end with an
unfinished call fails; an output-limit end drops that unfinished call. A
required/named request fails at a natural end without a qualifying call; an
output-limit end without one ends the reply by length (no calls, no error).

Each verified native template registration supplies syntax data: optional
outer wrappers, call delimiters, name position, argument encoding, parameter
boundaries, wrapping-newline policy, explicit string/JSON suffixes where used,
and the reasoning boundary. The interpreter has no family dispatch. JSON
objects, tagged parameters and object notation are generic syntax encodings.
Qwen, DSML and Gemma declarations come from their checked-in renderers and
pinned templates; all native variants are covered by renderer-call round trips.
The source-verified Llama 3.3 template hash
`e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65`
(`nvidia/Llama-3.3-70B-Instruct-NVFP4@f792b707`) independently binds a bare
JSON object envelope with `name` and `parameters`, even when rendering is
interpreted. Only an object at the start of the answer (after whitespace) or
right after a call is a candidate, as in vLLM and llama.cpp; JSON in prose is
answer text and is not withheld. In auto mode a candidate that is not a call
(no name or arguments, or not valid JSON) is answer text; required and named
modes refuse it. Arguments given as a JSON string that holds an object are
accepted. An interpreted template with unknown output syntax exposes no grammar. A
synthetic key/value-tag grammar tests extensibility without claiming support
for an unverified GLM template.

**One grammar.** The sampler's cursor is a `ToolInterpreter` fed exactly the
bytes the output parser receives (each token decoded by the same stream
decoder: hidden control tokens dropped, ill-formed UTF-8 replaced). Its probe
(`Allows`) runs the parser's own code on a copied cursor over the buffered
bytes plus the candidate, and `CanStop` is `Finish`'s own decision, so the
sampler, the speculative judge and the parser cannot disagree. Object notation
is one byte-stepped grammar used both to check sampled prefixes and to convert
complete calls (strings use the declared quote only; keys are quoted or bare).
The cursor is incremental: probes scan only new bytes, apart from the bounded
name checks and the full call check when a candidate closes a call. Malformed
text between tagged parameters is refused at once. One refusal is left to the
parser: a call that closes but cannot be converted (invalid explicit JSON, a
repeated parameter or key, a second call with parallel calls disabled). Its
body is fixed by then, so masking the closing token could only run a
constrained reply on to its token allowance; the probe allows the close and
the request is refused at once (500 `tool_output_invalid`, the error starting
`the closed tool call is invalid:`).

In unconstrained auto mode no token is refused: only terminals are masked
while a call is open, and the output parser alone judges the reply. Auto with
parallel calls disabled is unconstrained until its first call closes (only a
second call can then be refused, so the grammar masks from there on). `none`
refuses only call openers. Required/named choices mask terminal tokens during
reasoning and force the opening after the declared reasoning end, or
immediately outside reasoning. The forced opener spells the template's own
whitespace around it (`ToolSpellingFor`, read from the template's rendering of
a probe call when the template is chosen): Qwen3.8 forces
`</think>\n\n<tool_call>\n<function=NAME>\n`, as its renderer writes the call,
so the model continues the layout it was trained on and the next turn
re-renders the reply byte for byte. When the template leaves reasoning to the model (the generation
prompt neither opens nor closes it, as Gemma 4's thinking turn), the model may
open reasoning first; the opener is forced after it ends. Every candidate's
decoded bytes pass the same grammar before both sampling and speculative
acceptance; a bare JSON name is checked as a prefix of an allowed name while
it is written. Original native rows are
validated first, so masks cannot hide NaN or positive infinity; negative
infinity remains legal padding. Names are constrained on
subsequent calls too; no argument payload is copied per vocabulary candidate.
Actual stop IDs are also masked while a call is open or a required call has
not completed, with the same check in selection and speculative acceptance.
An ordinary auto-mode answer can still end normally. Exact grammar markers
declared as continued route stops bypass legacy route-only stopping; model/EOS
stops and client stop strings retain their own meaning.
Speculative judging ends at the first accepted natural terminator, before
later verified rows could reopen or reject the already completed tool stream.

**A step costs what it restricts.** A greedy step keeps the device's own
choices and verify verdict wherever no forced span is pending
(`Llm::DeviceChooses`): the host checks each chosen ID with `Allowed` (an
allowed argmax is the masked argmax, so the choice is exact) and accepts it.
A device choice the constraint refuses is made on the host from its own row,
which the runner reads back right after the step (`DecodingPolicy::readback`,
the head logits the device chose from, [engine](engine.md)): a plain step
then chooses from that row as a host step would, and a device verify keeps its
rows up to the refused one and makes that row's choice from it, which ends the
verdict there. Without a readback (a runner that keeps no head rows) only a
step that restricts nothing (`ToolSampling::Identity`: unconstrained, able to
stop, no forced span pending) keeps the device path, and a refused verify row
ends the verdict before it. Forced spans and sampled steps return rows. A
joined wave still takes one result kind: rows for every owner while any
owner's step needs them (per-owner kinds, `Batching::mixed_greedy`, measured
slower for tool-call waves and stay a diagnostic). Where a host step does
restrict, it judges lazily: greedy takes the row's argmax when
`Allowed` passes it; top-k sampling judges the row's leading tokens until it
has the k best allowed ones, all the masked distribution keeps; temperature
alone keeps an allowed draw and redraws a refused one from the masked row by
an independent stream (rejection sampling). Only top-p or min-p without top-k,
and a greedy argmax the constraint refuses, mask the whole row. A full mask
judges every plain token of a state at once: where the grammar's state treats
all bytes it does not attend to alike (answer text, reasoning, a raw or string
value: `ToolInterpreter::PlainVerdict`), a token made only of such bytes takes
that one verdict without a probe. A name check stops once the name has ended.
While a forced span is owed only tokens starting with its next byte are
probed, and one canonical plan answers for all of its prefix candidates
(`Tokenizer::PlanForcedPrefixes`) instead of one plan each. When the token
after a stable span is the held tail's only candidate, sampling could choose
nothing else: it joins the span as a certain token instead of a sampled step.
[The measurements](experiments/tool-sampling-speed/README.md).

The [canonical span planner](tokenizer.md) finds an exact cover before the selecting token is
published. Forced spans use marked encoding: the format markers' control tokens
are specials, and the client's name between them is always text. A plan cut
by the remaining token allowance publishes the selected token unchanged and
the reply ends by length. Proven stable IDs run with that selecting anchor in one common
causal unit, including ordinary models and target-written drafter state. An
unstable final block remains ordinary target sampling: unchanged selected IDs
advance the held grammar, with no injected unstable IDs or claim of canonical
segmentation under unknown future raw-BPE merges. Verification stops before
accepting a proposal whose cover changes its ID or introduces a certain span;
already public anchors are never replaced. Usage and optional scoring follow
actual published IDs and their real causal rows.

The caller prepays `ToolRequestBytes` for the request. Tool output is funded
as it grows, never from `max_tokens`: `ToolOutputBytes(open)` bounds what the
parser, the sampler's cursor with its speculative copies and one call's
conversion hold for an open call (or held text) of `open` raw bytes, and
serving grows the request's charge to it before each token's bytes reach the
parser (within the grant, else past it while the request pool's capacity holds,
else a memory refusal for that request alone). A request with `max_tokens`
omitted therefore reserves nothing for output it has not produced. The sink
separately owns and funds retained events after delivery. The parser limits
total raw generated bytes to the token allowance, not an arbitrary call count,
duration or per-call quota. Generated JSON nests at most `api::kMaxJsonDepth`
(64) levels: a deeper value is a request-local refusal (or, for a schema-typed
tagged value, stays its raw string), and no parser or serializer recurses per
level, so no output can exhaust the driver's stack. The streaming scanner keeps its lexical position across fragments and
withholds partial UTF-8 and marker prefixes before emitting text. Bare-envelope
name/argument recognition also retains fixed, copyable metadata: probes of an
unfinished call advance over appended bytes and any held delimiter suffix,
without rescanning its accepted argument prefix. A separate whole-pending projection preserves callback-pause
and ordinary-answer fallback semantics. Argument payloads remain borrowed by
probes; bounded keys and names use the existing JSON checks. `ToolOutputBytes`
funds both cursor snapshots through the actual interpreter size. This removes
whole-prefix rescanning structurally; no measured parser or model speedup is
claimed.

A tool result must follow an assistant message with tool calls (its own
group of results directly after it); a result after a user message, first in
the conversation or after an assistant without calls is refused with 400
before rendering, as OpenAI's API validates history, even where an upstream
template would render it. The frozen upstream fixtures that render such
orphan results (`orphan-tool`, `tool-first`, `tool-after-user`,
`tool-runs-interleaved`, `g3-system-ws`) keep their source bytes and are
asserted as refusals.

Parallel result groups are reordered by IDs into their assistant's call order.
Their text and error flag move together. Missing IDs preserve supplied order,
without inventing a stable association; duplicate or unknown supplied IDs are
an explicit ambiguity/error. The shared conversation adapter resolves supplied IDs first. A complete
missing-ID group can use ordinal order only consistently with those IDs; partial
same-name results resolve the name alone and preserve supplied order. Different
possible names are ambiguous. Gemma renders the resolved name instead of the
source template's empty-ID last match. Verified name-only templates receive
transient, collision-free association keys in both native/probed and interpreted
rendering; original IDs and pinned raw reference fixtures remain unchanged.

Qwen's user-defined markers still tokenize in message content for reference
parity (D-067). No blanket renderer-side marker refusal is introduced: it would
also reject legitimate tool history. Feeding only newly generated output into
the interpreter prevents input markup from being directly classified as calls;
it does not prevent a model from echoing supplied text. Tools remain client-
executed, and this foundation makes no stronger prompt-isolation claim.

The malformed-call cases follow the pinned Flare assessment: closed malformed
calls fail, optional Qwen outer tags are tolerated, the next parameter/call
boundary can close an unclosed parameter, and exactly one wrapping newline is
trimmed at each side where declared. A call whose last parameter is never
closed ends at the first `</invoke>` alias inside that value when the output
ends there (a natural end is then allowed); a value closed later keeps the
alias as its text. Unknown names survive auto mode. See
[Flare's pinned API semantics](https://github.com/NVIDIA/flare-local-inference/blob/e7b8c016474f3b58d390910a7368f58a792b4744/docs/api.md#conversation-and-tool-semantics)
and [the project assessment](flare-assessment.md#5-tool-calls-and-the-messages-profile).
No upstream parser implementation is copied; the shared interpreter implements
the format descriptions above.

Qwen's declaration also accepts `<parameter name="…">` headers and an
`</invoke>` call close at structural boundaries, alongside the canonical
rendered `<parameter=…>` and `</function>` spellings. These aliases are read
only before a parameter or after an explicit `</parameter>`; their text inside
an open raw value remains literal, and quoted JSON stays opaque. Undeclared
boundaries and malformed closed arguments still fail. A native mixed-encoding
Mia artifact (`7175c03c…`) produced this unambiguous alternate spelling in its
first named-tool diagnostic. The original request failed with an incomplete
call at natural completion; recovered persisted history identified the variant.
The parser and terminal fixes pass deterministic controls and a fresh plain
native named-tool/result transaction on the same weights. The original sampled
MTP attempt stopped at target/drafter architecture admission before inference;
its failed startup and **NOTRUN** inference controls remain recorded in
[the diagnostic results](experiments/m35-tools/RESULTS.md). After the canonical
mapping-identity repair, fresh separate sampled-MTP required-tool and named-tool
transactions pass with actual drafting, marker reads and streamed follow-ups;
see the [qualification evidence](experiments/m35-generic-import/original-parity.md#drafter-source-alias-admission-repair).
These two representative cells do not establish broad speculative tool coverage
or close qualification across the remaining aliases and template grammars.

## Route ownership and response contract

Modern `tools`/`tool_choice` and legacy `functions`/`function_call` normalize to
one owned request. Assistant history with a valid nonempty modern or legacy
call may omit content, with the same absence of text as explicit null. Ordinary
messages still require content. Contradictory forms fail; `none` remains harmless on templates
without a verified output grammar. Generated calls emit modern `tool_calls` in
both response modes, with stable request-owned IDs and one validated argument
fragment per call. Clients execute tools. Result messages return through the
same shared conversation adapter used by native and interpreted rendering.

A tool request that names no tools with `auto` (`tools: []`) is a request
without tools. Non-streaming replies with calls and no text send
`content: null`. Interpreted templates receive a tool result's `name` when
the client sent one (a legacy function result), as transformers does; native
renderers resolve the name they need from the association.

An accepted-and-paused output callback retains the parser's unread input tail
and token batch. Serial yield and cooperative pause transfer the same paid
output state; neither replays an accepted call nor prematurely finishes a
partial call. Payments follow request definitions, parser/cursor snapshots,
canonical planning, unread output and serialized escaping through retirement.
Explicit stop strings match contiguous answer segments, excluding reasoning and
arguments, and cannot join a partial match across a call. Natural completion
with calls uses `tool_calls`; token exhaustion retains `length` and drops a
truncated final call. Required/named mode still refuses if no call completed.
Any refusal by the request's tool grammar (an empty constrained candidate set,
a token or forced span outside the grammar) fails that request alone after
native retirement, on the serial path as in a cohort (500
`tool_output_invalid`); it is not evidence that the model's execution failed.
When the exchange itself ends the reply (the client gone, a deadline, the
runtime stopping), the reply carries the exchange's own status, not a tool
error.
