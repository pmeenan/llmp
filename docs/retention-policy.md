<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Conversation-state retention and the M6 acceptance trace

D-055 defines the bounded retention policy that D-024 and D-031 require
before M6, and names M6's A→B→A acceptance workload under D-036. It says how
reusable state outlives the request that produced it, who may reuse it, what
ends it, how it is bounded and what M6 must demonstrate. It does not
implement a cache, choose a durable spill format or report measurements.
M3 spills and restores one model's state across a full swap through M2's
write-back path, and (D-055 as amended 2026-10-02) an idle conversation's
state under memory pressure, with a configured retention and spill budget
([below](#victims-spill-and-exhaustion)); the rest of the policy here
arrives in M6.
D-050's [reservation policy](reservation-policy.md) remains the authority for
admitted work, and D-048's [completion protocol](async-model.md) for
lifetimes.

## Admitted state and retained entries

**Admitted state** belongs to an admitted request and lives inside its
retained-state allowance `R(G)` (D-050). It stays protected while the request
waits for the slot, runs, pauses at a completed boundary, stalls on output or
unwinds. Nothing in this document can expire, release, drop or evict it.

A **retained entry** is revocable cache that outlives its request. At the
D-050 handoff, a retiring request's state moves atomically from its
allowance into the idle cache as:

- a **continuation entry:** immutable state for the tokens the request
  actually processed into state, valued by reuse of that conversation
  branch. For autoregressive decoding that is the prompt and all but the
  final generated token; with speculative decoding (D-068), the same after
  rejected drafts are truncated; with block diffusion, the prompt and every
  committed block. Working state never published, such as a canvas or a
  drafter's state, is not part of it; and
- **shared-prefix entries:** immutable state at declared prefix boundaries
  the request covered, valued by reuse across conversations. Only a request
  that starts a new branch creates or refreshes them (below).

Entries are not capacity grants. They occupy otherwise unused capacity as
revocable cache (D-050) and end whenever this policy allows. A request that
retires successfully publishes its entries. A cancelled request may publish
only the longest prefix, at an adapter-valid boundary, whose producing
operations all retired successfully under D-048, and nothing if there is
none. A failed operation, contract excess or uncertain completion publishes
nothing, and its backing follows D-048's quarantine rules. Final, no-retain
and auxiliary requests publish no continuation entry (below); like any
request, they still use shared prefixes and, when they start a new branch,
create or refresh them.

## Identity

An entry is usable only when every component matches exactly:

| Component | Content |
| --- | --- |
| Artifact | The prepared artifact's content identity (D-035, D-054), never an alias, model name or source checkpoint name. For a context composed of several artifacts, such as a target and its drafter (D-068), every component artifact the state depends on |
| Execution | The state-producing numerical plan: selected kernel implementations (D-053), data types, state representation and layout version, position/attention configuration and context profile. FP16 and EXL3 contexts of one base model never share entries (D-052) |
| Tokens | The exact rendered token sequence from the context origin, after documented client stripping such as Claude Code's attribution block (D-045). Entries keep token IDs and chained block digests for longest-prefix lookup |
| Other inputs | Non-text input identity once a modality is supported. Until then, a request with non-text input neither creates nor uses entries |
| Caller scope | The request's authorized inference scope (D-042). No hit crosses scopes, so reuse timing cannot reveal one application's prompts to another; anonymous loopback is one scope |

Sampling parameters, request IDs, message labels, session or agent IDs and
all hints are not identity (D-031). Matching system-prompt text is never a
hit by itself: the same text after different preceding tokens is a different
prefix. A tokenizer, template, plan or artifact change produces misses,
never an incompatible hit.

## Restore boundaries

Each state representation's adapter declares where state can resume and what
coverage a resume needs; entries record enough to check it.

- **Full-attention KV** (both M6 models): state for positions below `n`
  supports resuming at any `m ≤ n`. Positions from `m` on are never attended;
  the partly reused block is copied before appending, with only positions
  below `m` kept and the rest zeroed.
- **Sliding-window attention:** resume at `m` only if the entry holds every
  position the first resumed query attends to. Deserializing and trimming a
  snapshot does not establish that (RE-007); use an earlier valid checkpoint
  or recompute.
- **Recurrent or compressed state:** resume only at recorded snapshot
  positions, with no truncation. The adapter bounds how many snapshots a
  branch keeps.
- **Unvalidated representations:** no reuse. An adapter earns reuse with
  evidence that restored state continues exactly as resident state does under
  the same plan; byte-identical round trips are not that evidence (RE-007,
  RE-008).

Reuse always leaves at least the final prompt token to process, so the first
output token comes from logits computed in this request.

## Hits, branches and sharing

Lookup, protection and promotion happen in one catalog transaction (D-048's
single writer). The lookup takes the longest valid match across both classes
within identity and scope. The reuse length is the largest valid boundary at
or below the common prefix. Before reuse, the adapter's coverage check passes
at that boundary, every block the reuse needs has a valid resident or spilled
copy of the recorded generation, and the request's admission covers its full
state, reused blocks included (D-050). A merely queued request pins nothing;
it repeats the lookup when admitted, and an entry lost meanwhile is a miss.

Retained blocks are immutable. An admitted request appends into private
blocks, so two branches from one entry never see each other's suffix
(D-031). Entries hold **retention claims** on the blocks they cover, and
admitted requests hold leases. Physical blocks are charged once and freed
only when no claim, lease or outstanding consumer remains (pager invariants
2 and 5). The bytes that dropping or demoting an entry frees are its blocks
with no other claim, lease or outstanding consumer; victim selection credits
only those, never the entry's logical size. Blocks leased by admitted work
count in `R(G)`, not against `M_state`, so a request can never wait on the
reclamation of its own leased blocks (pager invariant 7).

**Shared-prefix boundaries.** At retirement, a request that did not continue
a continuation entry creates a shared-prefix entry at each boundary it
covered, at the largest adapter-valid position at or below it, aligned down
to a whole block for block-organized representations (adapters that need
snapshots take one there during prefill), at least the minimum prefix
length `L_prefix`, and either:

- the end of the leading system, developer and tool-definition segment, as
  the template renderer reports it; or
- a client cache breakpoint, such as Anthropic `cache_control`, inside that
  leading segment; at most four per request, Anthropic's own limit.

The entry claims the request's existing whole blocks rather than copying
them, so it never holds a conversation's tokens past its boundary.
Other positions create no shared-prefix entries, so edits and per-turn
breakpoints do not multiply them. Clients such as Claude Code also mark
their latest messages; those breakpoints concern one conversation and at
most nominate snapshot positions for representations that need them.

**Continuing, branching and refreshing.** Each continuation entry records
where its request's input ended.

- If a later request's reuse reaches that point, it continues that branch.
  The hit refreshes the entry, unless the request is final, no-retain or
  auxiliary, and the new request's continuation entry supersedes it at a
  successful retirement if the entry's generation is unchanged since the lookup.
  Otherwise, as with a retry or fork racing another successor, the later
  request's entry starts a new branch in the same lineage. Only the
  generated tail, which a template may rewrite, can need recomputation.
- If reuse stops earlier, because of an edited message or a new conversation
  sharing the same system prompt, the request starts a different branch. It
  may reuse the matching blocks but neither refreshes nor supersedes that
  entry.
- A request that does not continue a continuation entry, and whose reuse
  covers a shared-prefix entry, refreshes it. Continuing a branch refreshes
  only that branch and neither creates nor refreshes shared prefixes.

Hits on a system prompt therefore never refresh unrelated continuations
(D-031), each class keeps its own reuse statistics, and a conversation stays
retained only through its own use. A shared prefix that expires while a
conversation still claims its blocks remains reusable through those blocks
and is recreated by the next new conversation that covers it.

## Retention, release and expiry

The owner chose capacity-driven retention with a long idle cap (2026-09-22).
An entry lasts until one of these ends it:

1. **Capacity:** victim selection drops or spills it (below).
2. **Idle cap:** it goes unrefreshed for longer than its class's maximum idle
   age: `T_cont` for continuations, `T_prefix` for shared prefixes. Both
   default to 24 hours, are configured separately, may be lowered and apply
   to resident and spilled copies alike. Bounded maintenance processes expiry
   within a pinned interval, so spilled prompts do not linger until the next
   pressure event, and deallocates spill ranges no other entry claims.
   Lookup treats an entry
   past its cap, or released, as ended whether or not maintenance has run.
   The idle clock keeps running through system suspend.
3. **Release:** an explicit close, a final request, or a compaction signal
   (below). No-retain suppresses only the current request's publication.
4. **Invalidation:** the artifact is removed or replaced (after D-054's
   quiescence), a plan or configuration change alters identity, or a restore
   fails its integrity or coverage check.
5. **Restart:** the M6 design's retention is same-process: startup
   deletes every spill file and no entry survives a restart. M3's runtime
   keeps a conversation whose whole state is on disk across a restart, by
   a validated record (D-105, below); M6's entry classes decide theirs
   when they arrive.

Ending an entry drops its claims. It never frees blocks that other entries
claim or admitted work uses, and ending one class's entries never ends the
other's.

**Release** (D-041, D-045; wire names remain M6 API design):

- At admission, a request that may leave a continuation entry receives an
  opaque, server-issued handle for the lineage it continues or starts, bound
  to the caller's scope. Return it in a namespaced response header on the
  protocol's normal schedule: at admission for SSE, before publication is
  known (D-045), but only once the outcome is known for non-streaming
  requests (D-047). Assigning a handle does not send headers early. A lineage
  is a branch plus any branches its racing successors start.
- Closing a handle releases every continuation entry published by requests
  that received it. The close record also covers every request admitted or
  queued at close time that received the handle or whose input reaches a
  released entry's recorded end: none of them publishes a continuation
  entry, so a late completion or a turn queued before the close cannot
  recreate one. Admitted work completes or cancels normally. The record
  sets an idempotent publication-suppression flag on each matched request
  until it retires or leaves the queue without admission; suppression
  survives a queued request's admission. Repeated closes coalesce in the
  bounded request and queue records. Requests arriving after the close start
  a new branch.
- Close is idempotent. An unknown, expired, released or foreign handle gets
  the same success result, which reveals nothing about other scopes.
- A final request publishes no continuation entry and, if it continues a
  branch, releases that branch's current entry at retirement. A
  request-scoped no-retain hint withholds only the request's own entry and
  names no older continuation.
- `x-claude-code-context-compacted` releases every continuation entry
  published before the signalled request whose recorded session and agent
  affinity matches, within the caller's scope. It also suppresses continuation
  publication by every matching request already admitted or queued when the
  signal arrives, even if no matching entry has been published or survives.
  Suppression follows those requests through admission and retirement as for
  close. An absent agent ID matches only entries and requests without one.
  With no matching entry or request it is a no-op (D-045).
- Release never touches shared-prefix entries, other branches, admitted work
  or weight residency. It is release, not secure erasure, and promises no
  immediate physical reclamation.

**Hints** only lower retention or nominate boundaries (D-045, D-046).
`cache_control` breakpoints nominate boundaries as above. Their TTL values
describe a hosted cache's pricing tiers, not a request to discard state, and
are ignored rather than shortening the owner's idle cap. `prompt_cache_key`,
`session_id` and Claude Code's session and agent IDs are recorded on
continuation entries only as fixed-size keyed hashes, counted in the
metadata bound: for release lookup now and for placement in M6a. `user` and
`metadata` are not stored. Claude Code's `auxiliary` request class publishes no
continuation entry by default: such side requests are typically one-shot and
would displace conversations that continue. Ollama's `keep_alive` governs
weight residency leases only and neither creates nor extends state
retention. No hint pins an entry, raises a cap or changes identity.

## Victims, spill and exhaustion

The idle cache holds clean idle weight extents, restorable from artifacts,
and retained entries, resident, spilled or both. When an admitted phase needs
capacity (D-050) or a cap is exceeded, the initial rule is:

1. Released and expired entries go first.
2. Evict clean idle weight extents under the global extent policy (D-008)
   while resident retained state is within its cap `M_state`.
3. When resident state exceeds `M_state`, or no eligible weight extent
   remains, demote the least recently refreshed resident entry. Spill it only
   if spill is enabled, its adapter has a validated restore path, spill space
   and the write budget allow it, and its cleanup resources are already
   budgeted (D-050); otherwise drop it.
4. When spill space or the write budget runs out, drop the least recently
   refreshed spilled entry, or drop instead of spilling.

Candidates are ordered by their own refresh times, so the class-specific
refresh rules carry through, and each is credited only with the bytes it
actually frees. This is a baseline to measure in M6, not a tuned policy;
cost-aware alternatives are compared on recorded traces before replacing it
(features.md).

**As built in M3 (D-055 as amended 2026-10-02).** The runtime replaced the
baseline with the owner's cost-aware order before M6, for what M3 holds:
one conversation branch per request slot and no shared-prefix entries. One
order (`memory/reclaim.h`) covers everything reclaimable, whoever holds it:
captured graphs, plans and idle conversation state (idle weights once
partial eviction produces them). It is GreedyDual-Size over measured
costs: a candidate's priority is the inflation value at its last use plus
its kind's cost to restore a GiB freed, the lowest goes first, and each
reclaim raises the inflation to what it took. So a costly kind outlasts a
cheap one while both are used, one left unused falls behind fresher cheap
ones, and within a kind the order is strict least recent use (never
largest first), the running model's last and never its in-use floor. A
candidate's cost is measured, not assumed: a plan's planning time, a
graph's capture, idle state's spill (of what changed) and restore at the
node's measured rates, weights' page-in. On a GB10: idle weights (0.08 s a
GiB), idle state (0.17 written whole, 0.07 unchanged), graphs (0.2–0.4),
plans (2–15); recomputing state costs 64 s a GiB, so idle state is
spilled, not dropped
([memory-pressure](experiments/memory-pressure/README.md)). Spill is the
swap's write-back path for one slot: the slot's state leaves every
closure, the initialized extents written since its spill file last held
it are written there (the rest released with the file's copy kept) and
their backing released, and its next turn's first unit restores them
at the generation they were written at before the state is reused (a
refusal for capacity is typed and retried after a reclaim). A cohort
member set aside for its peers is spilled the same way and resumes from
its restored state. State is dropped instead only when its contents are
not exactly known, past the spill budget, or with a budget of 0. The
spill write budget below is not built yet: the spill budget bounds
capacity, and writing only what changed reduces the writes.
Retention is `[memory] retention_hours` (below), enforced by the driver
between units and while idle for resident and spilled state alike; the
spill budget is `[memory] spill_budget_gib`, past which the least
recently used spilled conversation, a swapped-out model's included, is
deleted first. A conversation whose whole state is on disk (spilled, or
written back by a swap, or spilled at a graceful stop) survives a
restart of the service: its slot's spill file is a named owner-only file
with a record the next process validates (build, model, layout, the
files' identity and SHA-256 digests) and adopts, with its turn
checkpoints; retention runs on from its recorded last use (each turn
checkpoint from its creation, enforced while the service runs as well as
at the next start), and the spill budget counts it (D-105,
[runtime-serving](runtime-serving.md#conversations-kept-across-a-restart)).
While the service is stopped nothing enforces retention until it starts
again; `[memory] keep_across_restart = false` keeps nothing past the
process, and purging the package removes the default spill directory's
kept conversations.
Plans and graphs are charged inside the budget past one
step's floor (D-090 as amended), so growing state reclaims them through
the same order. M6's state is small next to its weights, so it cannot show
whether idle state within a large `M_state` starves weight residency. M7
and M9 measure that on their larger workloads. Entry caps also bound each
caller scope's share of metadata; they do not isolate resident or spill
capacity, whose global pressure may evict another scope's entries.
Spill is lazy: an entry is written only when demoted, never merely because
it exists. Writing entries early, to take writes off the switch path, is a
candidate for later measurement, not the default.

**Spill storage.** Spill lives in a configured node-local directory owned by
the service user with mode 0700, on a filesystem that passes the D-034
direct-I/O probe. It never overlaps D-054's storage roles or uses a long-term
store. Llmpalooza creates that directory with a marker file. Startup refuses a
spill directory that is a link, has the wrong owner or mode, or is non-empty
without the marker, then deletes only runtime-named spill files in it, so a
misconfigured path cannot wipe unrelated data. Files get runtime-generated
names, are created exclusively and are never opened through links. Space is
allocated before writing, so concurrent spill or installs cannot exhaust it
mid-write, and D-054's install checks leave the spill budget free.

Spill writes and restore reads use the storage service's bounded queues and
D-034 direct I/O. Spill is block-granular: a block shared by several entries
is written and charged to `S_spill` once, and an entry is restorable only
while every block it needs has a valid resident or spilled copy. Demotion
writes every block the entry needs that has no spilled copy, shared blocks
included, before dropping its resident claims. Spilled
blocks hold only positions below the entry's length, with the rest zeroed.
Source blocks stay charged until the write retires, and a failed, cancelled
or uncertain write never publishes a spilled copy (D-048); an uncertain
write's range stays charged until restart. A spill range is reused only
after every I/O touching it retires. Spill promises no crash durability, so
it needs no flush. The catalog holds each spilled block's content identity,
generation, length and content digest; restore verifies the data against
that record, not against a header in the file, before the state reaches a
request. A mismatch, short
read or I/O error invalidates the entry and falls back to recomputation,
with the reason reported. Because spill is deleted at startup and never read
by another process or version, its encoding is an internal implementation
detail, not an on-disk compatibility format; adopting crash durability would
need a versioned format decision. M3's runtime made that decision for its
conversations (D-105): named owner-only spill files with a versioned,
hashed record a slot, read back only by the same build
([runtime-serving](runtime-serving.md#conversations-kept-across-a-restart)).

Spill and weight transfers share one drive and one memory bandwidth. Demand
from an admitted phase outranks retention writes, and a write that frees a
destination for admitted demand inherits that demand's priority (D-050).

## Bounds and defaults

Only the idle caps are fixed now. The capacity values are pinned at M5 exit,
before M6 implementation, from llmpalooza's measured state bytes for its
supported models and the node's measured headroom, and recorded here with
their provenance.

| Parameter | Meaning | Rule | Pinned |
| --- | --- | --- | --- |
| `T_cont`, `T_prefix` | Maximum idle age per class | 24 hours each (owner decision); configurable: M3's runtime has `[memory] retention_hours`, 1 to 8,760, default 24, for its conversations and their turn checkpoints | Now |
| `M_state` | Resident retained-state cap | At most `B − F − J − max_m(R_m + E_m)`, so idle state never stops the node from holding its largest supported request | M5 exit |
| `S_spill` | Spill capacity | Within the spill filesystem's free space less a fixed reserve; held back from installs (D-054). M3's runtime: `[memory] spill_budget_gib`, default 128 GiB, the least recently used deleted past it | M5 exit |
| Spill write budget | Bytes spilled per rolling 24 hours | From the drive's rated endurance, with its source recorded, and the measured M6 workload; past it, entries are dropped instead of spilled and the event is reported | M6 entry |
| `N_prefix`, `N_cont` | Entry-count caps per class, with a per-scope share of each | Worst-case index metadata (token IDs, block digests and maps at maximum context) fits inside `F` | M5 exit |
| `L_prefix` | Minimum shared-prefix length | Below it, recomputation costs less than an entry's metadata and lookup | M5 exit |
| Maintenance interval | Maximum delay before expiry and release are processed | Bounded maintenance work, independent of inference queues | M5 exit |

`B`, `F`, `J`, `R_m` and `E_m` follow D-050: the execution budget, fixed
overhead, non-revocable maintenance work such as spill writes and expiry,
and model `m`'s retained-state allowance and largest phase envelope, which
includes the weights the phase needs, for its supported request profile.

For scale, not measurement: Qwen2.5-0.5B's F16 KV state is 2 × 24 layers ×
2 KV heads × 64 dimensions × 2 bytes = 12,288 bytes per token, consistent
with the EXL3 reference's 50,331,648-byte cache at 4,096 tokens. A full
8,192-token context is 96 MiB. The published weight files are 1,266,425,696
bytes (FP16) and 588,951,098 bytes (EXL3 4.0 bpw); prepared artifacts will
differ by padding and deduplication. M6's pair therefore exercises the policy
with state small next to weights; state-dominated budgets arrive with longer
contexts and larger models. The paging study's 8 GiB spill ceiling was a
scenario allowance with replayed peaks of 551.75 MiB (small models) and
228.763 MiB (large); it is not a default.

## Misses, fallback and reporting

On a miss, or a hit shorter than the history, recompute from the request's
supplied history: from the longest valid shorter prefix, otherwise from the
start. Standard clients resend history, so a miss costs time, not
correctness. If a request lacks history needed for reconstruction, which
only a future extension that omits history could cause, fail explicitly;
never continue from partial or unrelated state (D-024).

Each request reports reused tokens, recomputed tail tokens (cached but not
reusable, such as a rewritten generated tail) and new tokens. Usage fields
such as `cached_tokens` and `cache_read_input_tokens` count actual reuse
only (D-046). Where a protocol reports cache writes, they count tokens
submitted for retention, which stays revocable. Diagnostics give the owner
miss reasons (none, expired, released, evicted, spill full, write budget,
incompatible identity, insufficient coverage, integrity failure), per-class
hits, resident and spilled bytes, spill traffic and demotions. Clients see
a scope mismatch as an ordinary miss. Token IDs, prompts and state contents
are never logged (D-014).

## M6 acceptance workload

### Models, orientations and budgets

The owner named the pair on 2026-09-22: Qwen2.5-0.5B-Instruct FP16 GGUF
(D-051) and its EXL3 4.0 bpw quant (D-052), both contexts M5 must support.
Every measurement runs in both orientations: **o1**, A = FP16 and B = EXL3;
**o2**, A = EXL3 and B = FP16. The mixed-rate 4.5 bpw fixture, also in M2/M5
scope, is not part of this workload.

At these sizes, a physical pressure holder cannot safely force displacement,
so pressure is policy-forced on both sides, as in the reference cycle's
zero-pressure arms. The reference keeps one model loaded at a time, plus
whatever page cache the OS keeps; llmpalooza gets a configured execution budget,
less memory than the reference can use. Budgets use M6-entry measurements
of `F`, `J`, `R_m` and `E_m` for the pinned plans and trace, with `W_m` the
weight part of `E_m` and `S_m` the retained-entry bytes at the switch points,
all rounded up to whole extents:

- **`B_all = F + J + R_A + R_B + E_A + E_B`**: both complete requests fit as
  one concurrent cohort, so nothing is displaced.
- **`B_half = F + J + max_m(R_m + E_m) + S_A + S_B + ⌈½ · min_m W_m⌉`**:
  either model's request fits beside both retained entries, and part of the
  outgoing model's weights stays resident. Neither whole-model placement
  fits.

The M6-entry record shows, for each arm and direction, how many of the
outgoing model's weight bytes `B_half` displaces. Arms without resident
entries gain their bytes as weight headroom, and a returning request's own
entry already sits inside its `R_m`, so displacement differs by arm. An
extent-level arm and direction that displaces none or all of the outgoing
weights is not a partial-retention test; adjust the fraction and record it
before any acceptance run.

### Trace

A frozen, message-level transcript is fixed before any run, rendered
separately for each model with that model's authoritative tokenizer and
template, and pinned by hash per model. It uses synthetic text, like the
reference cycle's notebook. Let `C` be the smaller M5-validated context limit
of the two models.

1. **History (untimed):** three A turns under a system prompt that ends at a
   declared shared-prefix boundary. The user turns carry notebook records;
   the assistant turns are **synthetic replies fixed in the transcript**.
   A's rendered prompts reach about 40%, 55% and 70% of `C`.
2. **Outward switch (timed):** one B request with its own system prompt and
   a task of about 10% of `C`.
3. **Return (timed):** A's full history, the synthetic reply to turn 3 and a
   new user message, each about 2% of `C`.

Decoding is greedy with a pinned seed. Every response is capped at 128
tokens on A and 64 on B, and the prompts ask for more than the cap, so every
response ends at it. Pinning checks this for llmpalooza's plan and each reference
engine; a response that stops early needs a new transcript identity, not a
rejected trial. The fixed replies give every arm and engine identical token
arrays whatever it generated, and credit no engine for reusing its own
generated tail. A hit therefore processes the final synthetic reply and the
new message, less any leading tokens that reply shares with the generated
one. Reuse of a client-returned generated tail is covered by the functional
client case below.

### Arms

| Arm | Budget | Weight policy | State policy | Role |
| --- | --- | --- | --- | --- |
| J-all | `B_all` | — | Resident | Provenance control; concurrent cohort |
| J-partial | `B_half` | Extent-level (D-008) | Entries resident | Floor gate |
| J-spill | `B_half` | Extent-level | `M_state = 0`: spilled at publication, restored on return | Floor gate |
| J-whole | `B_half` | Whole-model control (D-036) | As J-partial | Whole-model comparator |
| J-whole-spill | `B_half` | Whole-model control | As J-spill | Whole-model comparator for spill |
| J-recompute | `B_half` | Extent-level | No retention of either class | Fallback control |

Reference arms are measured fresh and interleaved with these (D-025, D-036).
The pinned llama.cpp runs the FP16 model and the pinned ExLlamaV3
environment runs EXL3, with one model loaded at a time. llama.cpp's router
starts a model child process on each load, so its arms include child
startup, as in the reference cycle. The EXL3 process starts without a model
before the timed switches and keeps its interpreter and CUDA context across
model loads, which excludes interpreter startup and makes it the stricter
comparator. Its compile and tuning caches are populated before the run with
pinned kernel grids, so no timed load pays compilation or tuning. Arms cover
restore and recompute, cold and warm page cache, and matched and normal
configurations. Restore uses llama.cpp slot files, or for ExLlamaV3 a
harness serialization of its cache. A restore arm must first match the same
engine's live-state continuation, and a recompute arm its own fresh-context
run; a path that fails is recorded as unsupported and cannot set the floor.
The harness adds no durability flush beyond what conditioning a saved state
file cold requires, and that write-back counts inside the reference's timer,
as llmpalooza's spill write counts inside its own.

### Timing protocol

- Interleave every arm in each repetition in a seeded random order recorded
  before the run.
- Every trial starts clean: a freshly started llmpalooza runtime, with its spill
  deleted and no entries, or fresh reference processes with no model state
  or prompt cache from earlier trials; pinned compile and tuning caches are
  not state. After a verified fresh start, a nonzero reused-token count on
  the outward request is a llmpalooza correctness failure.
- Cold means no page of the incoming model file resident by `mincore`; warm
  means at least 99.9%. Reference arms also condition their saved state files
  on return, as the reference cycle did. For llmpalooza, condition both model
  files before the cycle and recheck at each timer start. Direct I/O should
  leave cold files cold, so a llmpalooza file that gains cached pages is a
  defect, not a rejected trial.
- Issue requests back to back: each timed request is sent when the previous
  response's terminal event arrives. The trial trace must show llmpalooza
  emitting that event before the outgoing handoff completes and before any
  write of its entries starts, and llmpalooza's timer starts at the earlier of
  that emission and the timed request's send; otherwise the trial fails.
  Outgoing retirement work thus always falls inside the timed switch.
  Reference arms follow the reference cycle's protocol, with their state
  save inside the timer.
- Llmpalooza's timer runs from that start to the first nonempty token event at
  the client; the reference's runs from the switch decision to the
  same event.
- Record per trial: first-token time, decode inter-token gaps, bytes read and
  written by class (weights, spill) and whole-node block traffic, peak
  occupancy from llmpalooza's ledger and sampled node memory, spill occupancy,
  reused, recomputed and new tokens, OOM and swap counters, and output
  token IDs.
- Fix rejection criteria before the run: a cache condition not met for
  reasons outside llmpalooza, concurrent external load, or a harness failure.
  Decide them from timing-blind signals recorded the same way for every
  arm, reject the whole repetition, rerun it and report rejection counts per
  arm. A correctness failure is never a rejection.
- Gate the pressure itself, not just the ledger: llmpalooza's attributable
  sampled node memory stays within the arm's budget plus its measured
  baseline outside it, and whole-node reads are at least the ledger's weight
  and spill reads. J-spill and J-whole-spill returns hold no resident copy of
  A's entry at timer start and read every spilled block required by the
  validated reused prefix, with its physical byte count pinned at M6 entry.
  Blocks belonging only to the rewritten generated tail need not be read.
  J-partial reads nothing from spill.

### Statistics and pass rule

Run **72 accepted repetitions** per arm, orientation and cache condition,
with the count and a single analysis pinned at M6 entry; no trials are added
after looking at results. At n = 72, the sample maximum is a one-sided
97.5% distribution-free upper bound on the 95th percentile
(0.95^72 < 0.025).

The floor is checked for each orientation, direction, cache condition and
floor arm (J-partial, J-spill), separately at the median and at p95. A
comparison passes only if llmpalooza's one-sided 97.5% upper bound for the
statistic is at most the smallest one-sided 97.5% lower bound among all
valid reference arms: restore and recompute, matched and normal views. Both
bounds come from binomial order statistics with no distribution assumed.
Beating every arm's lower bound includes beating the truly fastest arm's, so
a single comparison passes falsely with probability at most 5%, and the
gate needs every comparison to pass. A comparison with no valid reference
arm is inconclusive. Anything else is a failure or inconclusive, and neither
passes (D-036).

Report every arm's median and p95 with their bounds, J-whole's and
J-whole-spill's differences from the retained arms (the baseline for M9's
benefit target), bytes moved, peaks and token counts. Inter-token gaps are
reported; D-036's generation limits start at M7.

### Correctness gates

- Every timed trial's output token IDs match its provenance control exactly.
  J-partial, J-spill, J-whole and J-whole-spill match J-all, because
  restored state and reloaded weights are the same bytes under the same plan.
  That requires the plan, chunk schedule and kernel selection not to vary
  with the budget or state placement. J-recompute matches a fresh-context
  control with the same history and chunk schedule. Any mismatch fails the
  gate.
- Untimed teacher-forced runs compare all raw logits at fixed positions after
  the return, with the same pairing, and must be bit-identical.
  Nondeterministic kernels need separately measured controls and bounds
  declared before these runs (first-slice.md); a declared bound applies to
  both gates.
- Reuse against recomputation is judged only against a cross-schedule bound
  declared from M2/M5 controls. Without one, the difference is reported and no
  equivalence is claimed (RE-008).
- Reference arms must match their own frozen-output controls, or they cannot
  set the floor.
- J-partial and J-spill returns reuse exactly the longest common prefix of
  the return prompt and A's entry, recorded per model and plan at M6 entry,
  and process the rest; J-recompute reuses nothing. J-partial's return
  reads only the weight extents its outward switch displaced, and the ledger
  shows the rest of A's extents stayed resident throughout. J-whole reads
  A's full weights.

### Functional and adversarial cases

These run on the M6 pair unless marked; fake-backend cases also run
without a vendor SDK, as M2's did.

| Case | Required result |
| --- | --- |
| Conversations S+A and S+B on one model; B's first request | B reuses at least S and none of A's divergent suffix, and matches its uncached control; S is refreshed; A is neither refreshed nor superseded |
| Close A, start S+C, then resume A | C reuses S; A recomputes from its supplied history with reason "released"; B still hits; a repeated close is a no-op |
| New conversations keep hitting S past A's idle cap (test-shortened caps) | A expires on schedule and S does not; neither refreshes the other |
| S expires while B, which reused S, is admitted or suspended | B is unaffected; blocks stay charged once; S is recreated at the next new-branch retirement that covers it |
| A's continuation is evicted under capacity while S remains | A's next request reuses S and recomputes the rest, with reason "evicted" and counts reported |
| Edit an earlier A message | Reuse stops at the edit; the old branch is neither refreshed nor superseded; both branches match their uncached teacher-forced controls |
| Different content before the same system text; changed template, plan or artifact; FP16 versus EXL3 on the same text; another caller scope | Every case misses; no stale hit; identity changes invalidate affected entries |
| Spill full, write budget exhausted, injected spill I/O error, corrupted, truncated or stale-generation spill object | Drop or recompute with the reason; no false hit; admitted work unaffected; bad data never reaches a request |
| Close racing an in-flight request, and a turn queued before the close, on its branch; late completion after close or compaction; compaction signal matching several entries by affinity; unknown or foreign handle | Nothing is published on a closed branch; admitted work completes or cancels normally; no resurrection; every earlier matching entry is released; identical no-op results |
| Compaction while a matching earlier request is admitted or queued, with no published entry (or after its entry expired); queued work admitted after close or compaction | Matching pre-signal requests publish no continuation even after admission; suppression lasts through retirement; later arrivals may start new branches |
| Concurrent successors of one entry (a retry or fork), then close of the handle both received | At most one supersedes it; the other starts a new branch in the same lineage; the close releases both |
| Final-request flag, no-retain hint, auxiliary request class | No continuation entry and no refresh of a reused continuation entry; a final request that continues a branch also releases that branch's entry; shared prefixes are still used, and created or refreshed when the request starts a new branch |
| Continuation handle on SSE and non-streaming requests, including a non-streaming deadline after admission | SSE receives the handle at admission; non-streaming headers wait for the outcome and still allow a protocol-shaped 504; handle assignment does not commit HTTP success |
| Shared-prefix boundary inside a state block; spill of an entry whose blocks were reused from freed backing; lookup of an entry past its cap before maintenance runs | The prefix entry rounds down and holds no later tokens; spilled blocks carry no bytes past the entry length; the expired entry misses |
| Demotion while an admitted branch leases most of the victim's blocks | Only unleased, unclaimed blocks are credited; the admitted request never waits on its own blocks |
| Cancellation during decode with late I/O | Only a prefix confirmed by retirement is published; uncertain completion publishes nothing and quarantines (D-048) |
| Expiry, release or eviction aimed at an admitted or suspended request's state | Refused; admitted state is outside retention (D-050) |
| Ten times more distinct conversations than the entry caps | Metadata stays within its bound; least recently refreshed entries go; no unbounded growth |
| Restart with spill present; a planted link or wrong owner in the spill directory | Startup deletes spill and refuses an unsafe directory; no entry survives (M3's runtime keeps a conversation only through a record it validates whole, D-105: a link, a wrong owner or mode, another build or a digest that differs is refused and removed) |
| Entry chosen for a queued request is evicted before admission | Lookup repeats at admission; a miss recomputes; nothing was pinned while queued |
| Fake backend: sliding-window coverage (RE-007's rollback), recurrent snapshot positions, unvalidated adapter | Resume only at valid boundaries, otherwise an earlier checkpoint or recomputation |
| Concurrent A and B requests at `B_all` | Admitted as one cohort; both progress; outputs match serial controls |
| A request for B arrives while A is still generating, at a budget where they cannot run as a cohort; repeated under each D-069 policy with B interactive and A background, and with both interactive | Queue delay, paging/switch time and first-token compute are reported separately, with pause counts and reloaded bytes; a paused A keeps its admitted state and resumes correctly; outputs match serial controls. These results decide D-069's default |
| An unmodified named client builds a conversation on A, switches to B and resumes A, returning its own generated replies | Completes without session extensions; diagnostics show continuation reuse including the returned tail, less any template rewrite |

### What M6 entry pins

Before any acceptance run: the M5-exit capacity values above and the spill
write budget; the transcript, per-model token-array hashes, output caps,
decoding mode and seed, the expected reuse and required spill-read bytes per
model and plan; `F`, `J`,
`R_m`, `E_m`, `S_m` and the resulting budgets with the per-arm displacement
record; llmpalooza's measured memory baseline outside its budget; each arm's
`M_state` and spill enablement (J-partial needs
`S_A + S_B ≤ M_state` and no spill); the reference engine pins,
configurations, validated state paths, kernel caches and cache conditioning;
the repetition count, arm order seed, rejection criteria and analysis; and
any declared cross-schedule bound. Later changes need a new trace identity, not
an edited pin.

Crash durability, spill encryption (rejected), cross-node state transfer and
early spill writing are outside this policy. Physical-pressure and
state-dominated evidence waits for larger supported models: M7's named
Gemma/Ornith configuration and M9's larger-than-memory library.
