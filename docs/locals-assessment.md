<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Locals assessment

On 2026-10-22 the owner asked whether
[Locals](https://github.com/Generated-Computers/Locals), an open-source
inference server for Apple silicon that claims "up to 3×" over mlx-lm for
agent workloads, has any new ideas worth adopting. This page is a source and
documentation review of commit
[`f901c6d`](https://github.com/Generated-Computers/Locals/tree/f901c6df04e3ca1b585c4bd1a88150bfb7871692)
(2026-10-22; 151 commits since 2026-10-12, one author) and of the abstract of
its paper, [arXiv:2610.09139](https://arxiv.org/abs/2610.09139) (Asadulaev,
submitted 2026-10-06), whose proofs were not read. The checkout lived in a
session scratchpad and was treated as untrusted data; nothing from it was built
or run. Every Locals number below is the author's, from one M2 Pro (and a few
M2 Max and M5 Pro runs); none is an llmpalooza measurement. File references are
relative to that commit. The plan items this review produced are listed
[at the end](#what-the-plan-takes).

## What it is

- **Target:** Apple silicon only (M1 to M5), macOS, Python on top of MLX and
  mlx-lm (`pyproject.toml`: `mlx`, `mlx-lm`, `llguidance`, `outlines-core`).
  Metal kernels are source strings compiled through MLX.
- **Workload:** coding agents (OpenCode, OpenHands, Cline, Aider, Claude Code
  and others) with tool calls, long growing prompts and many concurrent
  agents on one small model (Qwen3.5-2B/4B in most tables).
- **Pieces:** batch engine with mixed prefill/decode passes
  (`hybrid_batch.py`, `mixed.py`); small-batch 4-bit matmul kernels
  (`kernels.py`, `qmv_multi.py`, `qmv_frag.py`, `mma_kernel.py`,
  `nax_kernel.py`); fused projections and a one-kernel Gated DeltaNet step
  (`fuse.py`, `gdn_kernel.py`); constrained decoding from JSON schemas, regexes
  and per-family tool-call formats (`grammar.py`, `regex.py`, `constraint.py`,
  `forced.py`, `budget.py`, `pda.py`, `walk.py`, `formats.py`); a prompt cache
  with checkpoints (`hybrid_batch.py`, `server.py`); a multi-agent harness
  (`locals run`).
- **License:** MIT. Its kernels adapt Apple's MLX under MIT
  (`THIRD_PARTY_NOTICES.md`). Permissive, so core-eligible under D-091, but the
  code is Python/MLX, so only techniques port.

## Where the headline speedup comes from

The repository's own ablation (`docs/why-faster/README.md`, Qwen3.5-4B, M2 Pro,
author's numbers) shows the speedup is in model computation, not scheduling:
Locals' model under mlx-lm's own scheduler is as fast as Locals' scheduler.

| Variant, 8 requests at once | Tokens/s (author's) |
| --- | --- |
| Plain mlx-lm | 86.2 |
| Locals | 272.7 |
| Locals' model on mlx-lm's scheduler | 272.9 |
| Without the small-batch kernels | 119.2 |
| bfloat16 instead of float16 | 97.9 |
| Without fused projections | 246.8 |
| Without the Gated DeltaNet kernel | 267.6 |
| Without pipelining | 236.5 |

- **Small-batch 4-bit kernels** work around MLX's matrix-vector kernel
  re-reading the weights once per row and its switch to 32-row tiles at about
  6 rows. Our decode and verify products already read compressed weights once
  for their rows and are selected per shape (T38, U6's rank data).
- **float16 on M1/M2** is specific to GPUs that emulate bfloat16.
- **Fused projections, the one-kernel Gated DeltaNet step and pipelined
  launches** are things the engine already does (fused pairs and GLU, `llmp.gdn.*`,
  asynchronous steps).
- **The caveats are the author's own:** one request alone is parity (65.6
  against 60.5 tokens/s); an M5 Pro gains about 1.5× because MLX has kernels
  for its neural accelerators; prefill-bound work is not faster; in the
  paper's 8-agent suite the wall time was 22.1 s against 19.4 s for the stock
  engine (0.87×), and the 16-agent 2.5× compares 12 against 9 of 16 agents
  finished.
- **Grammar memory.** The paper's "3 MB instead of up to 1.5 GB" compares a
  per-step DFA walk kernel (`walk.py`) with Locals' own full state-by-token
  table. xgrammar and llguidance never build that table, so the saving is
  against a baseline we would not use.

Nothing in that list is a new idea for us. The ideas worth recording are
elsewhere.

## Ideas worth taking

Each says how it lands in the shared engine (D-107), by capability, never
per model.

### 1. Budget-aware masking

`budget.py` computes, per grammar state, the fewest tokens that still finish
(a shortest-path fixpoint over the token-step graph, forced spans included) and
masks any token after which no valid completion fits in the remaining
`max_tokens`. By induction an output cut by the limit always ends in an
accepting state. The pushdown path uses a byte-count bound over the stack
(`pdactl.py`). The author reports (from the paper) 90 of 90 JSON outputs valid
under every limit that admits one, against 3 of 90 without it, and 54 of 61
against 2 of 61 for recursive schemas.

**Why it matters here.** Our tool-call plan drops a call cut off by a length
finish, as Flare does, and M7's strict tool arguments and `response_format`
would otherwise return invalid JSON on a length finish. With an agent client's
`max_tokens` that is a failed turn. The technique is independent of the
automaton library.

**Generic form:** a per-grammar-state minimum-completion table, built when a
grammar compiles and kept in its cache entry, that the mask step consults with
the tokens left. To check first: whether xgrammar's matcher exposes the
shortest completion per state (it may not for its pushdown form), otherwise
llmpalooza builds the table above it. Locals applies the rule only on its
table path (not in `WalkController` or its mlx-lm logits processor).

### 2. Canonical forced spans (jump-forward)

Where a grammar leaves one live byte (the `<tool_call>`/`<function=` syntax,
JSON keys, brackets), `constraint.py` and `forced.py` compute the forced byte
chain, and the engine feeds it in the same forward pass as the sampled token
instead of sampling each token (the author's docs: m known tokens cost one
weight-streaming pass). Two details are the useful part:
- **Canonical tokenization.** `ForcedPlanner` re-encodes a forced span the way
  the tokenizer would, checks that the tokens reproduce the bytes, and holds
  back an unstable last block (one that later bytes could merge into). Without
  this, jump-forward feeds token sequences the tokenizer would never produce,
  which is off-distribution for the model.
- **Boundary cover rule and healing.** The token sampled just before a forced
  span must be at least as long as the single token that the span's first block
  would be, so a non-canonical split (`":` then `"To`) cannot be sampled; if
  the canonical split merges the last sampled token with the span, the planner
  re-encodes them together, so no KV rollback is needed.

The author reports 1.31× to 1.60× over mlx-lm with llguidance on JSON jobs
(2B/4B models, M2 Pro) when the schema's syntax is left to the engine; that
number is theirs.

**Generic form.** On a bandwidth-bound GB10 a forced span of a few tokens costs
close to one weight read. In our speculation scheduler a forced span is a
proposal source whose acceptance is certain, beside Flare's history-copy drafts
(that mapping is this review's, not Locals'). M3.5's bound-token forcing of
the tool-call begin sequence is the first, smallest use and needs the same
canonical-tokenization rule; M7's constrained output generalizes it.

### 3. Reply-boundary checkpoints for recurrent state

A recurrent layer's state (Gated DeltaNet) cannot be cut back to an earlier
position, so a cached conversation is reusable only from exactly where it was
stored (`docs/prompt-cache/README.md`). Locals found that Qwen3.5's chat
template renders an earlier assistant turn differently from the open turn that
produced it (it drops the empty think block), so request k's prompt plus its
reply is not a prefix of request k+1, and the next turn re-read the last
question and the whole reply. Its fix:
- store the recurrent state where the next request's rendering diverges from
  this one, found by **differential rendering**: render the conversation plus
  an assistant message and a user message with two different marker strings and
  take the first position where the renderings differ (`Server._checkpoints`);
- keep the exact token IDs of recent replies, so re-tokenizing a client's
  resent history gives the tokens the model produced and the lookup does not
  miss (`Server._remember`, `_tokens_for`);
- keep a small reply entry, only the recurrent state (a few MB per layer) at
  the reply position joined at retirement with views of the finished row's KV.

The author's figures (before and after against their own earlier build, not
another engine): a follow-up on a 60,564-token document 137.4 s → 1.5 s; on a
138,352-token document about 500 s → 2.4 s; prompt tokens from cache across four
multi-turn conversations 28% → 83%.

**Why it matters here.** Our plan's minimal shared-prefix item measures the
system-prompt boundary and the strict-prefix rendering test (Flare), and D-031
keeps a continuation's exact model-owned branch identity, but stateless clients
resend text. Whether a stateless client's next request hits our retained
continuation across the template's re-rendering and across re-tokenization of
the model's own reply is a measurement we have not recorded. Qwen3.8 is a
Gated DeltaNet model, so it applies if its template behaves like Qwen3.5's
(not checked here).

**Generic form:** the renderer reports two boundaries per turn (end of the
shared leading segment, and the position at which the next request's rendering
of this reply diverges), found by differential rendering, so it needs no
per-template rule. The tokens of a model's own reply are retained with the
continuation.

### 4. Volatile blocks moved to the end of the system prompt

Some agents put per-session text (date and time to the microsecond, the
project folder) in the middle of a long system prompt, which ends reuse at that
point. Locals' `_volatile_to_end` moves such blocks to the end of the system
message; two OpenHands sessions in different folders then share 13,714 of
13,795 prompt tokens instead of 1,555 (author's). It changes what the model
reads (the same text in a different place), so it is a quality trade:
under AGENTS.md rule 11, at most an opt-in client-profile mode, off by default,
and only if the plan's measurement of real prompts shows byte-instability is
the cause of misses.

### Leads with no evidence yet

- **Tree decoding over a grammar's candidates (`treedec.py`).** At a decision
  with at most 8 allowed tokens (the default cap), every candidate plus its forced span is
  appended to one forward pass as a branch, with an ancestor attention mask and
  per-branch positions. One pass then yields this decision and, for each
  candidate, the next one; the right branch's KV is kept. Nothing is drafted or
  rejected, so greedy output is unchanged. It suits bandwidth-bound decode,
  where extra rows are nearly free, but the repository has no measurement of it
  (no table in the docs, no test that names it), and it needs tree-masked
  attention rows in our kernels. A lead for M9 once constrained output exists.
- **Sparse LM head (`kernels.head_rows`).** When 4,096 or fewer tokens are
  allowed at a state, only those rows of the output head are computed. It
  helps constrained single-stream decode only, saves head weight traffic, and
  is small; measure after constrained output exists.
- **Prompt-lookup drafts under a constraint (`lookup.py`).** A draft proposed
  from the request's own history is first walked through the automaton on the
  host and cut at the first disallowed token. It joins Flare's history-copy
  drafts (plan.md) as a detail when constrained output exists.

## Apple-silicon notes for the later port

AGENTS.md puts Apple silicon first among later ports ("do nothing for them yet,
sacrifice nothing on the GB10, but don't foreclose them"). What Locals teaches,
all from its docs and not run:

- **Compute dtype is a device property.** M1/M2 emulate bfloat16 (MLX's fp16
  kernels run up to 1.7× faster there); M3 and later have native bfloat16.
  Locals casts bf16 scales and activations to fp16 only when every value fits
  fp16's range and a 12-token probe agrees on the argmax and nearly on the
  logits (`runtime.set_compute_dtype`). Our plans should take compute dtype as
  a device-capability input, as `sm_121` against `sm_86` already is.
- **Small-row quantized products need several implementation families** with a
  per-row-count crossover: a multi-row matrix-vector kernel, dequantizing into
  matrix-unit fragments, and dequantize-then-dense from about 128 rows. That
  matches D-053's ordered implementation lists; check that rank data can key on
  a row-count class (U6).
- **A reference engine closes gaps on new silicon.** On an M5 Pro, MLX's own
  neural-accelerator kernels cut Locals' gain to about 1.5×. At port time
  measure against current MLX on the target chip; do not carry over M2-era
  conclusions.
- **Format support differs from CUDA's.** Metal 4.1's matrix multiply takes FP4
  and FP8 operands and MXFP4 with E8M0 scales but not NVFP4's E4M3 block scales
  (engine-components.md), so import-time relayouts (rule 11) will be per
  platform, chosen by tensor role and encoding.
- **Unified memory.** Locals' admission control, growth guard and prompt-cache
  byte budget solve the same single-pool problem as D-004's one physical budget;
  nothing in them beats exact catalog accounting (D-006, D-007).

## Not adopted

| Item | Why |
| --- | --- |
| MLX small-batch, fragment and NAX kernels; fp16-on-M1/M2; the MLX-shaped fused projections and Gated DeltaNet kernel | Patches for MLX's gaps; our kernels already cover the shapes on CUDA; Metal comes in the later port |
| GPU-resident automaton state and the per-step DFA walk kernel | We sample on the host (cheap mask application, D-106); the walk kernel's saving is against Locals' own full table, not xgrammar or llguidance |
| Per-field and per-sibling parallel decoding (`fields.py`, `siblings.py`) | Approximations: a branch does not see earlier siblings' values, so conditioning changes. A per-alias mode at most; no schema-extraction client has asked |
| Background autotuning on the user's machine | Selection is checked-in rank data per device class (owner, 2026-10-20) |
| `locals run` multi-agent harness (planning, racers, test-picked winner) | Product layer above the engine |
| Shared-prefix tree KV with log-sum-exp merge of the root segment (`tree.py`) | Our shared-prefix reuse copies state exactly; block sharing stays M6's |
| A disk cache of system-prompt snapshots across restarts | Already covered by spill and restore (D-055, 7-day retention) |

## What the plan takes

| Item | Shared home | Plan placement |
| --- | --- | --- |
| 1. Budget-aware masking | Constrained-output grammar cache, mask step | M7's constrained output |
| 2. Canonical forced spans | Speculation scheduler (certain-acceptance proposals), tokenizer planner | M3.5's tool-call item (the forced begin sequence), M7's constrained output |
| 3. Reply-boundary checkpoints, differential rendering, retained reply tokens | Renderer boundary reports, prefix store and continuations | M3.5, measure first, with the minimal shared-prefix item |
| 4. Volatile-block reordering | Client profile | M3.5, only if the prefix measurement shows byte-instability is the cause; opt-in |
| Tree decoding over candidates, sparse head, constrained lookup drafts | Speculation scheduler, constrained output | M7 and M9, leads after constrained output exists |
| Apple-silicon notes | Portability design | [Deferred delivery table](plan.md#deferred-delivery-and-proposals) |
