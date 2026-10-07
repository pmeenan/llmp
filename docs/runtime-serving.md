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
the program, or a job it queued, may still refer to its frame. Each of
its waits watches the node's progress and asks its patience
([hang recovery](#hang-recovery)): when nothing moves it cancels the
request and waits for the drain, and if even that does not come the
process ends rather than free memory a job may still use (the service's
ladder exits for its supervisor; the commands' own patience, ten minutes
without progress each way, aborts). A swap or eviction asked for between a request's steps ends
that request first; teardown ends every request, fences each model's
stream, evicts every managed extent and checks every backing released.

## Configuration

Both approved Gemma26 and dense31 artifacts have a bounded native route on
this same driver. Complete trusted-artifact binding selects the existing
immutable engine profile before adapter construction and is rechecked at setup.
It accepts chat and literal completions with target likelihoods. Context is capped
at 262,144 and settings fallbacks are uncalibrated. Plain chat disables thinking;
generated tools and assistants/speculation are refused.
The [bounded dense31 production bridge](experiments/gemma31-serving-bridge/README.md)
selects ordinary joined serving, both norm chains and eligible owner attention for
approved 31B artifacts with resolved context at most 8,192 and at most four slots.
Its uncalibrated prefill fallback is 256; smaller explicit overrides remain intact.
The [bounded Gemma26 recipe](experiments/gemma26-production/README.md) selects
joined serving, both norm chains, MoE route/reduce and owner attention for the
approved 26B-A4B artifact under the same bounds, with a 1024-row prefill fallback;
its recorded adoption C1/C4 cycles were within 0.6% of that stock method with
stock's exact tokens, and its
1,024-row corpus heads are byte-identical. Both bounded Gemma4 recipes retain the
full final FFN before frontier-head publication, matching stock's ordinary row
shape. Three small same-geometry heads per profile are byte-identical after
that correction; its paired C1/C4 cost screens are native versus native.
Matched Gemma4 stock backend-sampler timing remains open; the recorded
stock bookends use CPU-head publication.
Larger configurations retain the prior scalar recipe and 128-row cap. The recorded earlier dense31 recipe has zero
predicted-ID differences in current-pin C1/C4 8K continuations and
128/129 and 512/516 byte-exact complete heads; its 1,024-row corpus has
complete head parity. The correction adds three small exact heads per profile
and unchanged 8K tokens/histories, without a new full-head 8K screen. Whole serving cycles were 3.15%/3.51% slower than the
reference at adoption and 0.73%/1.36% after [gap closing](experiments/gemma-gap-closing/README.md). The corrected ordinary HTTP gate passes same-geometry literal token/score
repeats and stock top2 IDs, cached chat response replay, stops and
client-observed departed-client peer completion. These are bounded controls, not sustained performance, broad semantic
quality, assistant admission or long-context qualification.
The [two-profile route controls](experiments/gemma31-serving/README.md) retain
owned continuation, kept restart and pending cross-profile switch evidence.
[Gemma's contract](gemma4.md) lists the remaining gates.

The approved Gemma3 4B QAT Q4_0 artifact has a
[bounded ordinary route](experiments/gemma3-execution/README.md#bounded-serving-unequal-widths-and-wrapped-rings):
context<=4096, prefill chunks<=128 and one or two request slots. Defaults are
4096/128/1; more owners, larger contexts and speculation refuse. It uses its
own checked optimized recipe, classic SentencePiece tokenizer and recognized
Gemma3 template. Chat/literal completions, scoring, device greedy, initialized
checkpoint/restore and two kept conversations across restart pass focused
controls. Two-owner decode can join.
[Compatible plain prefill](experiments/gemma3-execution/README.md#compatible-joint-prefill-and-common-attention-reads)
now joins equal 2–128-row chunks under a separately funded 256-row wave;
per-owner state/checkpoint rows remain 128. Scoring, one-row or incompatible
chunks and checkpoint/reuse units remain scalar. Funded temporary K/V and
mask padding handles unequal decode reads without widening actual cache
roots or cursors. Actual successful joined groups, ring likelihoods, queued
third requests and exact matched cached restart replay pass HTTP controls.
A short cold C2 endpoint bookend improves native latency 14.03% with exact
generated IDs/usage; this is distinct from the +1.49% matched runner gap to
stock. Broader cohorts, memory/swap and sustained qualification remain open.

A node names the models it serves in its configuration (D-073's document,
`schema_version = 2`; the keys are new and compatible):

```toml
[models.deepseek]
artifact = "8a355bfb…"   # an installed artifact's ID, under storage.installed
drafter = "dd2d3f9c…"    # optional: its speculative drafter (DSpark, MTP)
# speculation = true     # the default when there is a drafter
# wave_form = "auto"     # concurrent waves: "auto", "speculative" or "plain" (below)
# context = 262144       # default tokens of conversation state (bounds below)
# prefill_chunk = 4096   # rows of a prefill chunk; default by model (below)
# prefill_floor_tok_s = 100  # tokens a second: the floors the chat route figures
# decode_floor_tok_s = 5      #   a request's work at (progress and deadlines, below)
# max_slots = 4          # requests at once at most, 1 to 16; default by model
                         #   (request slots, below); memory decides below it
# ... and every other setting (model settings, below): each absent key
#     derived from the artifact, calibrated, or its measured fallback

[models."qwen3.8"]
artifact = "c4fb47a9…"
drafter = "056a750e…"
tokenizer = "/path/to/tokenizer.json"          # when the artifact keeps none
chat_template = "/path/to/chat_template.jinja"  # likewise

[models.image]
composition = "eca21baa…"  # a pipeline (D-089)

[memory]                     # D-055 as amended 2026-10-02 (below)
# retention_hours = 24       # idle conversations, resident or spilled, and their
                             #   turn checkpoints stay reusable this long: 1 to 8,760
                             #   (across a restart too, D-105)
# spill_budget_gib = 128     # spilled conversation state kept on disk at most,
                             #   the least recently used deleted first: 0 to 1,048,576
                             #   (0: idle state is dropped, never kept spilled,
                             #   and nothing is kept across a restart)
# keep_across_restart = true # conversations wholly on disk survive a restart
                             #   (D-105); false: nothing outlives the process
```

A model names exactly one artifact or composition; every other key is a
setting (below), each taken by an artifact's model, a composition's or
both, and refused on the other kind; an artifact serves one model. A
node names as many models as it likes: the
library may exceed memory (D-102). The runner follows the artifact's architecture
(`deepseek4`, `qwen4exp`, bounded `gemma4` or `gemma3`) or the composition's (Qwen-Image); another is
refused at registration. Every
artifact is opened under the store's trust rules (only root and the
runtime's user may change it), and the tokenizer and template files the
configuration names are read under the configuration's (D-073). A chat
template renders natively when a native renderer has its hash or reproduces
it on the probe corpus, and otherwise through the sandboxed Jinja-subset
interpreter; a model is refused only when neither accepts its template,
and the runtime logs which way a template it did not pin renders (D-067 as
amended 2026-10-02, [tokenizer.md](tokenizer.md#chat-templates)).

The omitted context defaults to 262,144 tokens. The configuration takes any
context from 512 (a 32-bit count; no generic ceiling since D-102);
registration checks the supported checkpoint's trained ceiling, its
position tables', before opening the device node or constructing any model:
DeepSeek V4 Flash permits 1,048,576 and Qwen3.8 Flash Next 262,144.
An explicit context that exceeds its checkpoint is refused with the model's
name and allowed range. These are virtual ceilings: initialized state grows
inside the physical budget and can be refused when it no longer fits.
The 1M DeepSeek ceiling is not a claim that a 1M conversation fits one Spark.
The widening preserves `schema_version = 2` and the HTTP API version (D-062).

## Model settings

Users bring any model, so no setting is chosen for a named checkpoint
(D-103). Every setting of a model resolves once at registration, before
the model is constructed (`runtime/model_settings.h`), from three layers,
later ones winning:

1. **derived** from the artifact itself: its architecture, its kept
   metadata (the GGUF's keys, or `config.json` and
   `generation_config.json`), its drafter's metadata and its vocabulary;
   never a checkpoint's name or hash;
2. **calibrated** on this machine, for a measured speed trade, recorded
   per artifact, drafter, device, driver and build (calibration, below);
3. **override**: the key in `[models.<name>]`.

A setting none of them gives takes its **fallback**: the constant M3
measured on a GB10 (`spark-b`). The schema is one table
(`config::ModelKeys`): each key's type and range and which models take it;
an unknown key is refused, as is one the model's kind does not take. A key
the model's architecture does not use (a DeepSeek wave setting on Qwen3.8)
is logged as ignored. An override the artifact cannot honour (a context
past the checkpoint's ceiling, more drafts than the drafter proposes) is
refused before the device node opens, naming the model and the bound.

The start logs every effective value with its source once the model is
set up (`model NAME: settings context=262144 (fallback) speculation=true
(derived) ...`), and `jitllm-runtime settings [--json]` lists each with
what it came from, reading only the configuration and the installed
artifacts (no process lock, no device), so it runs beside the service.

| Key | Models | Type and range | Default and its source |
| --- | --- | --- | --- |
| `context` | LLM | tokens, 512 to 2^32−1 | derived: the trained context (GGUF `<arch>.context_length`, `config.json` `max_position_embeddings`, or the original context times a rope scaling factor where that is longer) when below 262,144, within the runner's own ceiling; else fallback 262,144. An override past the ceiling is refused |
| `speculation` | LLM | boolean | derived: true with a drafter, false without (`--plain`: false) |
| `prefill_chunk` | LLM | rows, 1 to 262,144 | fallback: 4,096 (DeepSeek above 262,144 tokens of context: 2,048); then capped by the model at its context (below) |
| `max_slots` | LLM | 1 to 16 | fallback: 4, the measured knee; DeepSeek with DSpark at most 8 (below) |
| `prefill_floor_tok_s` | LLM | tokens a second, 1 to 1,000,000 | calibrated: a third of the measured prefill speed at short context; else fallback 100 (progress and deadlines, below) |
| `decode_floor_tok_s` | LLM | tokens a second, 1 to 100,000 | calibrated: a third of the measured decode speed a request at short context (its steps', or its waves' with the peers sharing them); else fallback 5 |
| `recompute_ms_per_token` | LLM | milliseconds, above 0 to 10^6 | calibrated: the slowest of three whole prefills of 8,192 tokens or more, a token; else fallback 1.4, DeepSeek's prefill (16,384 tokens in 22.9 s). The reclaim order's cost of dropping an idle conversation is its tokens at this cost |
| `temperature`, `top_p`, `top_k`, `min_p` | LLM | as a request's | derived: the checkpoint's `generation_config.json` (`do_sample: false`: temperature 0) or GGUF `general.sampling.*`: DeepSeek V4 Flash's GGUFs temperature 1 and top_p 1, Qwen3.8's GGUF quantizations top_k 20 and top_p 0.95 (the UD-IQ3_XXS); else fallback: OpenAI's (1, 1, off, off), as for the Qwen3.8 NVFP4 artifact, which keeps none. A request's own field always wins |
| `reasoning_start`, `reasoning_end` | LLM | a token's text, at most 256 bytes; `""`: none | derived: `<think>` and `</think>` where the vocabulary has both; else none. An override must be a token of the vocabulary |
| `draft_rows` | DeepSeek, Qwen3.8 | 1 to 16 | DeepSeek: fallback 3 (its verify one row more), derived the drafter's `dflash.block_size` when smaller; more than the block is refused. Qwen3.8: fallback 3, 2 to 3 only (its MTP passes) |
| `wave_form` | DeepSeek | `"auto"`, `"speculative"`, `"plain"` | fallback `"auto"` (waves, below) |
| `wave_costs` | DeepSeek | 1 to 7 numbers, 0 to 1,000 | calibrated: each width's measured draft-verify wave time over its plain one's; else fallback, measured on a GB10 with wave lanes and joined drafts (2.08, 2.52, 2.81, 2.40, 2.12, 2.21, 2.23); calibrated widths replace theirs, and an override's widths (from 2) replace both, the rest keeping theirs; 0 always speculates at that width |
| `prefill_outa_hca` | DeepSeek | boolean | fallback true: the output-A/HCA prefill on full 4,096-row chunks, qualified ([ds4-output-prefix](experiments/ds4-output-prefix/README.md#default-on-acceptance)); false for the ordinary arithmetic |
| `prefill_outa_hca_partial` | DeepSeek | boolean | fallback true: and on every chunk of 64 rows or more; listed false when `prefill_outa_hca` is off, which it needs |
| `draft_vocab` | Qwen3.8 | rows, 0 to 4,194,304; 0: all | fallback 65,536; with speculation, a selected head caps the effective rows at its physical row count (0: every selected row). The listing includes the requested cap and keeps an override's source; it does not select a prefix layout |
| `depth_cost_ratio` | Qwen3.8 | above 0 to 100 | calibrated: a three-draft step's measured time over a two-draft step's; else fallback 1.16 (adaptive depth, below) |
| `shared_wave_depth` | Qwen3.8 | drafts, 1 to 16 | fallback 2: a request's drafts in a wave of several; at most `draft_rows` (listed so) |
| `draft_wave_max` | Qwen3.8 | requests, 1 to 16 | fallback 2: past it each request drafts alone |
| `wave_lanes` | Qwen3.8 | boolean | fallback true: independent work within a wave runs on per-request streams from four requests; smaller waves keep one stream |
| `wave_read_align` | Qwen3.8 | a power of two, 256 to 65,536 | fallback 2,048: the cells a wave reads, rounded up (coarser keeps plans replayable; GGUF equivalence is qualified at the same alignment) |
| `image_size` | composition | pixels, 64 to 2,048, a multiple of 32 | fallback 1,024 (its latents file must match) |
| `image_steps` | composition | 2 to 100 | fallback 40 |

`artifact`, `composition`, `drafter`, `tokenizer` and `chat_template` name
the model's files rather than settings. The quality-qualified choices
(`prefill_outa_hca`, its partial form) keep their qualification (D-085):
their defaults are what passed, and an owner may turn them off for
exactness. Kernel schedules (the D2R, IQ2 pair, output-A and MXFP8 tables)
are still chosen by hand-tuned tables for the GB10 and the measured
shapes; their calibration on the machine is later work.

### Calibration

[Controlled Qwen chunk screens](experiments/qwen38-prefill-chunks/README.md)
retain the 4,096-row fallback: 8,192 improves one fresh 8K C4 cell by
2.69% with 4.58% more memory, but loses on another prompt family. These
runs do not install a new calibrated value.

A model's measured speed trades are calibrated on the machine from its
first uses, passively, with nothing run at startup
(`runtime/calibration.h`): its prefill chunks' speed and its decode
steps' and waves' speed a request at short context (under 32,768 tokens;
prefill chunks of 1,024 rows or more, since a short prompt's chunk is
mostly fixed cost; the floors are a third of each, the margin for depth),
whole prefills' cost a token (the recompute cost; prefills of 8,192 tokens
or more), DeepSeek's draft-verify and plain wave times at each width, and
Qwen3.8's three-draft and two-draft step times. A value is the median of
at least eight samples; the recompute cost is the slowest of the first
three whole prefills, since prefill slows with depth and a long
conversation's cost is what the reclaim order must not underrate (a cost
a byte of state measured on short prefills did: 12 s a GiB for DeepSeek
at 3,549 tokens, 6 s for Qwen3.8 at 9,000, against 64 at 16,384, so it
counts tokens instead). A wave counts only when steady and
matched between the forms: none that planned or captured a graph (a
shape's first waves), and a draft-verify wave only when every member's
full verify joined it (not one cut by a mask width or the reply's end,
or run alone). Until a width's cost is calibrated, DeepSeek alternates
the two forms at that width, at most 32 waves, until each form has its
samples (auto `wave_form` only, never with a sampling member). Widths
within an explicit `wave_costs` prefix keep their configured adaptive
choice instead: zero always speculates, and a finite cost follows counted
acceptance. Unoverridden widths still explore; after exploration the
counted choice runs as before.

The alternation is not visible in throughput, and adds no new kind of
reply variation. Measured on `spark-b` (wave lanes build, four
concurrent 124-token chats, 512 tokens each, two bursts per fresh
service, alternating cells): with the community GGUF the first burst of
a service that calibrated ran 42.76 and 43.27 completed tokens a second
against 42.74 and 43.85 for services with the calibration in force, and
the warm bursts 43.14 and 43.79 against 43.80 and 43.28; with the Unsloth
UD-Q2_K_XL, 41.36 against 41.65 (warm 42.22 against 42.66). Width 4's
calibrated cost was 2.94 (community) and 2.90 (Unsloth) against the
harness's 2.99, so the same waves ran plain; forcing every wave
speculative on the Unsloth GGUF ran 38.39 to 38.86 against plain's 42.75
to 43.23, confirming that choice. In auto `wave_form` the replies to
concurrent requests already vary between runs with the waves they join
(no two of these services' bursts gave the same replies, calibrated or
not); the exactness controls `wave_form = "speculative"` and `"plain"`
never alternate.

Once a value is measured it is written, between units and at teardown, to
`<storage.state>/calibration/<artifact ID>.json` (0600 in a 0700
directory, written to a new file, synced and renamed over the old):

```json
{"format":"jitllm-model-calibration-v1","artifact":"8a355bfb…","drafter":"dd2d3f9c…",
 "device":"NVIDIA GB10 sm_121, driver 580.95.05, CUDA 13.0","build":"0.3.0-dev.12+g…",
 "settings":"speculation=true draft_rows=3 prefill_chunk=4096 max_slots=4 prefill_outa_hca=true/true wave_form=auto prefill_chunk_override=false max_slots_override=false context=262144 wave_costs=[2.08,2.52,2.81,2.4,2.12,2.21,2.23] wave_costs_override=false wave_costs_override_count=0",
 "values":{"prefill_floor_tok_s":333,"decode_floor_tok_s":7,
           "recompute_ms_per_token":1.25,"wave_costs":[2.08,null,2.81],"depth_cost_ratio":1.1}}
```

`wave_costs` holds widths from 2, `null` where unmeasured. `settings`
names what the measurements depend on, as resolved without calibration
(an override, a derived value or the fallback): speculation (off with
`--plain`), effective context, draft rows, prefill chunk rows, request slots,
the output-A/HCA prefill and the wave form. Context matters even when the
configured prefill chunk is unchanged: the runtime caps its actual rows by
the context. DeepSeek also keys the seven wave costs resolved without
calibration, override presence and the number of leading widths explicitly
overridden. An equal fallback-valued override can hide a calibrated cost;
two such prefixes of different lengths can hide different widths, even
when their uncalibrated vectors are identical. Recorded wave costs do not
invalidate their own key, since registration and `settings` resolve the
lookup policy without calibration.
For Qwen it also includes wave lanes, effective `draft_vocab`,
`shared_wave_depth`, `draft_wave_max`, `wave_read_align` and the
uncalibrated `depth_cost_ratio`. These alter the paid work or adaptive
policy; a change makes the old measurements stale. Override presence for
the calibrated chunk, slot cap and depth ratio is also keyed: setting a
fallback value explicitly can bypass a different calibrated value. The
ratio used in the key is the override or fallback, so a recorded ratio does not invalidate
itself. The record format stays v1; earlier dependency strings are
stale and are measured again.

`build` is the version; for a tree without Git metadata or with uncommitted
changes, whose version stays the same between builds (`0.1.0-dev+unknown`),
it adds the executable's size
and modification time, so each rebuild measures again. The record is
in force from the next registration, never mid-service, so a running
schedule never changes with wall time; the start logs whether it is in
force. A record for another drafter, device, driver, build or settings is
stale: not used, and measured again. To measure again on purpose, delete
the file (with the runtime stopped, or before its next start). One that is not this format, names another
artifact, holds a key other than the calibrated ones (the five above,
`prefill_chunk` and `max_slots`) or a value outside its key's range, or that
users other than root and the runtime's could change, is refused, logged
and replaced the same way. Prefill chunk rows and the request-slot knee
need controlled runs; the record already takes them. The
[controlled Qwen slot screen](experiments/qwen38-slot-knee/README.md) retains
four for the measured selected-head/4,096-row profile and proves its
calibrated source in isolated owned state. No production record or default
changes. The [controlled DeepSeek comparison](experiments/deepseek-slot-knee/README.md)
also retains four for the community+DSpark automatic profile: six raises
throughput but delays mean/median completion, and remains an explicit
throughput/tail option. A separate current-build prime supplies the actual
key for an owned slot-only record; explicit slot or wave-cost-prefix
overrides and a changed context make it stale. The performance measurements keep their original
build/key. Broader profiles still need controlled calibration. An override
always wins over a calibration.

## Registration and the swap

The serving commands and the service with models run the runtime's
startup steps (anchor, configuration, process lock, storage roles,
platform), then register every configured
model on one paged node: its artifacts opened, its runner set up on its own
stream, its tokenizer and renderer found, the shared workspace mapped at the
largest model's need. The scheduler's physical cap is the fixed allocations
plus the post-setup available memory, less the largest host-built chunk
inputs, every model's plan budget and a 6 GiB guard, rounded down to 2 MiB
extents. Fixed allocations, weights, used state and dynamic staging are
charged within that cap; the largest model's weights must fit beside the
fixed allocations at admission. The guard covers memory outside the catalog
(the driver and cuBLAS; [long-context](experiments/long-context/README.md#memory-and-the-guards-margin)).
A model's plans and graphs are host and driver memory outside the
catalog's extents, but counted (D-090 and D-055 as amended 2026-10-02):
each runner charges what each plan and graph holds to the node as it
keeps it (`engine/planned.h`). The guard sets apart only what one step of
the largest model holds at once (`Served::plan_floor_bytes`: a chunk
beside its draft block, or one wave); everything past that floor is a
charge inside the budget, so plans and graphs grow into room the budget
leaves free, with no fixed number, and are given back through the node's
one reclaim order (below) when conversation state, a swap's incoming
model or pressure from outside needs the room. A swap keeps both models'
plans and graphs.
Conversation ceilings reserve virtual addresses; backing grows in 2 MiB
extents with each padded cache prefix. The budget admits that growth from
the remaining physical capacity, preserving the guard. Spill and restore
move only initialized extents. Diagnostic snapshots allocate cataloged,
pinned staging lazily and copy only used pages (charged within the
budget, not set apart again from the guard's margin, so the DeepSeek
DSpark beside Qwen3.8 configuration can take one). No weights are paged
yet.

**The reclaim order** (`Server::Reclaim`, `memory/reclaim.h`; D-055 as
amended 2026-10-02). Whatever can give memory back goes through one
order: captured graphs, plans, and idle conversations' state (a branch
with no session whose state no request leases), spilled to its slot's
spill file (idle weights too, once partial eviction produces them; M3's
full swap evicts an inactive model's weights whole). The order is
GreedyDual-Size over expected costs: a candidate's priority is the
inflation value at its last use plus its kind's measured cost to restore
a GiB (a plan's planning time, a graph's capture, a spill's write of what
changed and its restore at the node's rates, 11.0 and 14.5 GB/s on a
GB10 until the runtime has measured its own) times its chance of reuse,
which halves with every reclaim since its last use and every 5 minutes
idle. The lowest goes first, and each reclaim raises the inflation to
what it took. Measured on `spark-b`, idle state (0.17 s a GiB) goes
before graphs (0.2–0.4) and plans (2–15) used at the same time, while a
plan or graph unused for three reclaims or a quarter of an hour falls
behind a conversation used just now; recomputing state (its tokens at the
model's `recompute_ms_per_token`, 1.4 ms until calibrated: 23 s for
16,384 tokens) is never chosen while spill has room
([memory-pressure](experiments/memory-pressure/README.md)). Within a kind
the least recently used goes first, the running model's last; never its
in-use floor (its most recently used plans up to its `plan_floor_bytes`,
with their graphs) nor anything a step under way holds. A reclaim takes
all it was asked for or nothing (the caller then waits or refuses), and
no more, with one exception: a victim that gives back less than it
counted (held, or gone meanwhile) has the order select again for the rest
without it, and if the rest cannot be covered the reclaim ends short,
keeping what it took (its caller waits or refuses as for nothing). An
idle conversation that could only be dropped while a continuation holds
it is never a candidate. The order ranks a GiB, so a reclaim takes the
same order over each smaller set of the kinds too and keeps whichever covers the need at
the least total expected cost to restore: 55 MiB comes from a few stale
graphs, not a 704 MiB idle conversation (D-104). Cache charges (a plan, a graph's capture) displace only other
plans and graphs, never conversation state; a graph's capture takes only
what costs less to restore than a graph, and one refused is not asked
again for the next 1, 2, 4 … 256 uses of its plan. Conversation state is
displaced only for state (a growth, a restore, pinned staging), a swap,
or pressure from outside. The order runs when a growth or restore is
refused for capacity (below; each refused member's need on its own),
when a plan, a graph or pinned staging does not fit (from inside its
step, which it spares), before a swap whose incoming model would not
fit, and under pressure from outside. A swap's room is checked against
the catalog again after each reclaim, and a swap it cannot make room
for is refused before anything moves. One that fails partway (a read
error) is undone: what the swap brought in (the incoming model's weights
and state, never the shared workspace or a runner's pinned memory) goes
out again and the outgoing model loads back. If the undo fails too, or a
model's first load fails, no model is resident and the next activation
loads its model whole; a first load makes its room through the same
order first. A model whose checks after its load fail (its places not
pinned, or checks of what its kernels index, D-090) is evicted with its
state written back rather than left resident, and its next activation
loads and checks it again. In every case only the requests that needed the
swap fail (503); the backend goes on (D-102: recovery first). Only a
faulted node ends the process, by abort (signal 6) rather than an
orderly stop, for its supervisor to restart. The turn checkpoints' staging is set apart at
start, so a full budget never refuses one; a checkpoint that is not
saved or restored is logged.

**Spill writes only what changed** (D-055 as amended). Each runner
records what was written to a slot's state since its spill file last held
it: the first position a job wrote from (covering the ranges writes from
there may change, as a turn checkpoint does) and ranges copied in.
Anything else that changes the state loses the record. A spill, or a
swap's write-back, then writes only those extents. The rest are released
with the file's copy kept, which the catalog confirms by the content
generation the file saved (`scheduler::EvictOptions::unchanged`). This
saves writes for Qwen3.8 today. DeepSeek with DSpark writes everything:
its write ranges and the whole drafter region cover every used extent.
A spill that fails partway (an I/O error) leaves the state not exactly
known, so the conversation is cleared instead, the log says so, and its
next turn prefills.

**Retention, the spill budget and pressure** (`[memory]`, below).
Between units and while idle, the driver deletes idle conversations,
resident or spilled, unused for `retention_hours`, and while the spilled
state (a swap's written-back conversations included) passes
`spill_budget_gib`, the least recently used spilled one; a spill that
would pass the budget deletes those first, counting the whole state it
will hold on disk, not only what it writes (and deletes none when even
all of them would not make room), and with a budget of 0 (or a
state larger than the whole budget) idle state is dropped instead of
spilled. A swap counts the outgoing model's state as its write-back
leaves it (all of it on disk) and the incoming model's as resident:
none of the incoming model's conversations is deleted for it, and one a
request holds is never deleted; a swap that still does not fit is
refused before anything moves, and the deletions wait until the swap's
room is made, so a swap refused for room deletes nothing. It also reads MemAvailable and the
kernel's pressure-stall information (`runtime/pressure_trim.h`): under 512
MiB available, one reclaim through the same order asks for what would
bring MemAvailable back to 1.5 GiB (the mark plus 1 GiB of headroom),
taking whatever part of it the order has. A full memory stall of 5% of
the last 10 seconds only raises the trigger to that target: a stall
alone (a compile or a copy with plenty available) is not pressure on the
runtime. Pressure ends only back at
that target, and nothing more is trimmed between the mark and the target.
While it persists, trims are spaced by a back-off that doubles from 1 s to
60 s, and a trim that found too little left goes straight to 60 s and logs
once a minute at most. No reclaim takes the running model's in-use floor
(its most recently used plans up to its `plan_floor_bytes`, with their
graphs; `memory::ProtectFloor`) or anything a step under way uses. So
pressure that another process keeps up drops what lies past the floor at
that growing back-off (once a minute at most once it settles), never four
times a second. Each reclaim, deletion and pressure event is one log line
of counts, bytes and the measured costs (D-014).

One model is resident at a time (M3's full swap). Activating another
(`Server::Activate`) is one `SwapProgram`: the resident LLM's conversation
state written back through the zone to each slot's spill file in
`storage.spill` (named and kept across a restart in the service,
[below](#conversations-kept-across-a-restart); unnamed for the commands)
if it holds a conversation (a model with none keeps its
state resident), its weights evicted with their backing parked for the
incoming loads (D-033's handoff), and the incoming model's whole closure
paged in, its state restored if it had been spilled. Then the model's own
checks (DeepSeek's hash-routing tables, Qwen3.8's n-gram hash) and its
places checked still pinned (D-090). The backing no load took is released
after the first output, off the swap's path. Each part is timed.

## Turns

An LLM holds one default conversation branch: the tokens its state has seen. A turn's
tokens (the conversation rendered by the model's chat template) extend it
when they start with it, and only the rest is prefilled. Otherwise it
restores the nearest retained turn checkpoint inside the exact common
token prefix, then prefills the suffix; without a matching checkpoint it
clears first. A turn is one request (D-093): the model's closure
leased once, the prefill chunks ([below](#prefill-chunks-and-cancellation))
and every decode step jobs under it. Decoding is greedy and, where the
model has a drafter, speculative by default (D-092's batched verify:
DeepSeek's DSpark draft and verify as one job, Qwen3.8's MTP draft then
verify), each step accepting the drafts the target agrees with; `--plain` decodes one token a step. The chat route may
sample instead (a `temperature` above 0): seeded, each token drawn at its
position in the conversation (execution/sampling.h), and when speculating
each draft accepted by speculative sampling (`VerifyDraft`), so the
tokens are distributed as plain sampling's; a seed repeats a reply, but
for a DeepSeek request in waves of more than four, whose verify is cut to
its share of the wave's rows and so accepts drafts at other positions,
depending on its peers ([request slots](#request-slots)). (Plain
and speculative sampling turn one seed into different tokens, so DeepSeek
keeps every wave with a sampling member speculative, below.)
Qwen3.8 greedy speculation chooses depth two or three using a moving
acceptance average and a measured relative step cost of 1.16 (its
`depth_cost_ratio`, model settings above). It tries
four complete steps at each depth, then probes the other depth for four
steps after 32 observations; a change needs a predicted 3% gain. Output-
or context-truncated verifies do not train the policy. Its schedule is
saved with conversation state, independent of timing. Seeded sampling
keeps depth two, as does a configured prefill chunk of only three rows.
Generation stops at the template's end-of-turn tokens, the token limit, or
when the route ends it (a stop string, the client gone, the runtime
stopping, and only where the owner configured them, a stall with
`stall_action = "fail"` or a non-streaming deadline;
[progress and deadlines](#progress-and-deadlines)), always between
steps; the same ends a prefill between its chunks (below). A stream whose
client stops reading pauses between steps until it reads again
(backpressure, [the chat route](#the-chat-route)). A model whose
chat template neither a native renderer nor the interpreter accepts is
refused at registration, naming its hash. A job that failed after it may
have run leaves the conversation unknown, so the next turn clears the
state first.

The host-side `Llm::Branch` owns that conversation's history, turn checkpoints,
sampling scratch and saved cursor. Its move-disabled generation session borrows
stable options and callbacks, prepares one step, and applies its completed
result through the same acceptance and history logic as scalar generation.
The legacy methods forward to this model-owned default branch. Only this branch
is exposed; native execution still uses the runner's default state.

The turn cache keeps two checkpoints per model on the current branch,
captured before the renderer's assistant opening. That boundary survives
a client removing earlier reasoning; the opening itself can change when
the assistant becomes a history message. Checkpoints copy whole used
physical pages that later work can overwrite: recurrent and ring state,
partly filled pool blocks, padded cache tails, DeepSeek dummy cells and
the drafter's mutable state. Immutable earlier cache pages stay in the
current branch's live state or its ordinary spill file. Rollback restores
the complete footprint and drops newer tail pages, then restores the
speculation cursor and adaptive schedule. It also discards checkpoints
newer than the selected boundary.

Each checkpoint owns a private direct-I/O file in the spill directory:
in the service a named owner-only file kept across a restart with its
conversation (removed when the checkpoint is dropped,
[below](#conversations-kept-across-a-restart)), unnamed for the
commands. One cataloged 2 MiB staging buffer plus alignment space is
allocated only during a transfer. The cache does not keep pinned payload
copies between turns. A failed allocation or disk write skips publication;
a failed restore clears before recomputing. Uncertain device completion
quarantines state and retains staging until process exit.
Checkpoint transfers check cancellation and beat the progress watchdog
between each page. A cancelled capture drops its unpublished file and
keeps the completed prefix; a cancelled partial restore clears before
reuse. The page currently in flight completes first.

Checkpoints expire for reuse `[memory] retention_hours` (24 by default)
after capture; looking one up does not renew it. Idle live history, and
an idle conversation's spilled state, expire after as long unused: the
driver deletes them between units and while idle, and a lookup treats
them as gone whether or not it has. Physical file cleanup of checkpoints
is lazy at the model's next request or destruction, and `Clear`,
`Forget` and a diagnostic full-state restore drop all entries; a restart
keeps those of conversations whose whole state was on disk (D-105).

An idle conversation the reclaim order spills keeps its history: its
next turn's first unit restores its state exactly from the slot's spill
file before reusing it (refused for capacity like a chunk, and run again
once the order made room) and prefills only its new tokens. A turn that
does not continue it discards the spilled state as a clear would.
This is computation reuse within one model's current branch, not session
identity or the independent shared-prefix and branch cache planned under
D-031. The native control and measurements are in
[turn reuse](experiments/turn-reuse/README.md).

The image pipeline generates the prompt and initial latents it registered
with: this slice's image runner (being reworked by the image-speed slice)
takes them at setup, so a process serves one image prompt, and the latents
come from a file (the reference's for its seed; a native seeded generator
is still to come).

## Conversations kept across a restart

The service keeps a request slot's conversation across a restart of the
process (graceful, a crash, or [hang recovery](#hang-recovery)'s
restart) when its whole state is on disk: spilled by the reclaim order,
written back by a swap, or spilled at a graceful stop (D-105). The next
process adopts it with its turn checkpoints, and its next turn restores
the state exactly from disk instead of prefilling it again: the reply
equals the one an uninterrupted process gives. State resident at a crash
or a hang's exit is not kept (its next turn prefills). The serving
commands keep nothing and leave what the service kept alone.

**The files** (beneath the spill role, `conversations/<artifact ID>/`,
mode 0700; each file 0600, owned by the runtime's user, one link, never
opened through a link): `slot-N.state` is slot N's spill file (each used
2 MiB extent at its place, region by region), `slot-N.record` its record,
`slot-N.turn-K.state` its turn checkpoints (whole state pages, each at a
4 KiB-rounded place). A model's directory is named by its artifact's ID.

**The record** (`runtime/kept_record.h`, format version 1) is strict JSON,
written whole (a temporary file synced and renamed over it, only if it
is still the file written, the directory synced) and ending with the SHA-256 of everything before it:

| Field | What it holds | Refused when |
| --- | --- | --- |
| `format`, `version` | `jitllm-kept-conversation`, 1 | another format or version |
| `build` | the version, commit, SDK, target and the executable file's device, inode, inode generation, size and modification time | not this build's (another build, or this one rebuilt or reinstalled) |
| `artifact`, `drafter`, `layout` | the model's artifact and drafter IDs; the runner's state format version, context, drafter and each region's name and bytes | not this model's or layout's (a changed context is another layout) |
| `slot`, `file`, `device`, `inode`, `generation`, `file_bytes`, `regions` | the slot, its spill file's name, identity and size, each region's bytes in it | another slot's, or not the file it names, or another size |
| `extents` | each used extent (region, index) with the SHA-256 of its 2 MiB in the file | outside its region or layout, out of order, none, or a digest the file does not match |
| `tokens`, `cursor`, `decoding`, `used_unix_ms` | every token the state has seen, the speculation cursor and adaptive draft depth that go with it (its relative cost is the machine's calibration's now, D-103, never a reason to refuse), and the last use (wall clock) | a token outside the vocabulary, more than the context, a draft depth of another maximum, past `retention_hours`, or more than a minute in the future |
| `checkpoints` | each turn checkpoint: its file and identity, position, creation, pages, footprint, cursor, draft depth and each page's SHA-256 | (that checkpoint alone is left out) |
| `digest` | the SHA-256 of the record before it | the record cut short or edited |

A record exists only while its slot's state is wholly on disk and nothing
has written the file since: the driver removes it before anything may
change the file or make the state live again (a turn restoring it, a swap
bringing the model in, a clear, a discard, retention, the spill budget),
and after a spill or a swap's write-back completes, with the state
settled (a verify's owed restore run first, where no request holds the
model's stream), queues a new one. Only the latest queued record for each
slot stays, and invalidation drops its queued token and checkpoint copies
immediately; the worker may hold one other record while hashing. A keeper
thread (`runtime/state_keeper.h`) syncs the files, hashes them on four threads and writes
the record unless the slot was invalidated meanwhile, off the driver's
path (written and synced beside its name, renamed over it under the
keeper's lock only while the slot is still current, its directory synced
after, so a turn that invalidates a slot never waits on a sync); turn
checkpoints, which never change, are hashed once. The hashing is
background work: its threads run at the lowest CPU and I/O priority
(nice 19, the idle I/O class), wait between extents while a swap or a
prefill runs and for 250 ms after, and stop at the next extent once their
slot is invalidated. A crash at any moment leaves a record that
describes its files exactly, or none, or one removed just before it (a
removal is not synced) whose spill file may have changed since: the next
start refuses it unless every extent it lists still hashes as recorded,
and adopting it empties every extent of the file it does not list, so the
conversation reads its own state and zeros beyond it, as a fresh slot
does. Turn checkpoint files are covered whole by their digests.

**At start**, before the models register, each slot's record is read,
checked and its files hashed; what validates is adopted (its spill file
kept as it is but for the extents its record does not list, which are
emptied; its history, cursor, draft depth and last use restored, the
slot marked spilled until a turn restores it), and everything else in
the directory is removed: refused records and their files, other slots'
files, other models' directories, temporary files. The start logs each
refusal with its reason and each adoption with its token count, never
content (D-014), and extends the service manager's start timeout as it
goes. **At a graceful stop** (SIGTERM) the resident model's idle
conversations are settled and spilled within the spill budget, the
most recently used first, each deleting only spilled state used before
it (the least recently used first); one that cannot fit is not kept, and the keeper writes every record
queued while it makes progress (a minute without any ends the wait),
extending the stop timeout as it goes; the log says how many and how much
was hashed. **At a hang's exit** the records already queued get at most
10 s; nothing new is spilled.

Retention and the spill budget apply while the service runs and across
the restart: an idle conversation, adopted or not, expires
`retention_hours` after its last use and each turn checkpoint
`retention_hours` after its creation (deleted with their files, the
record rewritten without an expired checkpoint), an adopted
conversation's state counts toward `spill_budget_gib`, and a budget of
0, or `[memory] keep_across_restart = false`, keeps nothing (the
directory is removed at start).

**What this exposes** (D-105, D-014's note). The records hold each kept
conversation's tokens (its text, through the tokenizer) and the spill
and checkpoint files its KV state, owner-only, where before D-105
nothing outlived the process. While the service is stopped, or after a
crash, nothing enforces retention until the next start; a graceful stop
writes every idle conversation still resident to disk; removing the
package keeps the files, and purging it removes the default spill
directory's `conversations/` (one configured elsewhere is the owner's to
empty). An owner who wants nothing on disk past the process sets
`keep_across_restart = false`.

**Measured** (`spark-b`, GB10, 2026-10-03; DeepSeek V4 Flash with DSpark
and Qwen3.8 with MTP at contexts 32,768 and 33,792, greedy;
[hang recovery](experiments/hang-recovery/README.md)): a Qwen3.8
conversation of two turns (265 tokens), written back by a swap to
DeepSeek, then the service stopped with SIGTERM (5.0 s, 1.8 GiB hashed in
2.1 s at the hashing's background priority), killed with SIGKILL once its
record existed, or exited by hang recovery's rung 3 and restarted by
systemd, then started again: each adopted it (the start 3.5 s, under the
calibration the first process recorded), and its third turn reused all 265
tokens and replied exactly as the uninterrupted control did, reasoning
and answer. DeepSeek's conversation, spilled by the graceful stop, was
adopted too and its second turn restored its turn checkpoint from the
adopted file and replied as the control did (killed, it was resident and
prefilled again). One byte flipped in the spill file, or the restart with
another context, refused Qwen3.8's (`its digests differ`, `its state
layout is not this runner's`; DeepSeek's still adopted) and the turn
prefilled from the start. The background hashing cost the swaps nothing
measurable: three alternations of the two models swapped in 9.42–9.47 s
and 7.64–7.70 s with records kept, 9.47–9.51 s and 7.73–7.76 s without.

## Prefill chunks and cancellation

A turn's prefill runs in chunks (`runtime/prefill.h`). The chunk is the
model's `prefill_chunk` if configured (1 to 262,144 rows), else the
runtime's default for the model, and in either case at most what the
model's state layout admits at its context (DeepSeek: the window cache's
cells less its 128-position window, and its fast plan's F16 attention
mask over the ring and compressed cells under 2^31 bytes, RE-037:
13,528 rows at a 262,144-token context, 4,096 to 1,030,144 and 4,024
at 1,048,576;
Qwen3.8's fast graph: 8,192 rows) and below the context, in whole 8-row
tiles. The reference graph's host
masks have the separate RE-037 bound of F32 [context, rows] tensors under
2^31 bytes: 4,095 rows at 131,072 and 2,047 at 262,144. The configured
prefill chunk's cap stays 262,144 rows, as do the
runtime's default chunks: Qwen3.8 4,096; DeepSeek 4,096 to a 262,144-token
context and 2,048 above it, since its attention mask is sized for the
whole context (at 1,048,576 its capped 4,024 rows fix 4.50 GiB against
2,048 rows' 2.36 GiB, room the recorded 1M conversation's state needs). Every supported
context has a chunk: at the minimum, 512,
DeepSeek's chunk is 384 rows and Qwen3.8's 504. A chunk of 1,024 rows or
more runs in whole tiles and its few remaining rows as a chunk of their
own, since GGML's attention reads the mask in whole 8-row tiles from
1,024 rows on (RE-036). Registration logs each model's chunk.

**The defaults** were first sized from the runtime's own prefill
(`jitllm-runtime chat`, speculative, so each chunk also feeds the
drafter; one model configured, context 8,704; `spark`, GB10, 2026-09-29;
the best of two turns each, from a cleared state; bold, that sizing's
choice, before the DeepSeek stage mechanisms):

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
resident.) The policy chooses each default chunk for prefill throughput
within the memory bound; a chunk is also how soon a prefill notices a
cancellation, but that latency does not cap it (owner, 2026-10-02). The
first sizing gave DeepSeek 2,048 rows and Qwen3.8 4,096. Against the
fixed 512 rows before, an 8K-token prompt prefilled 1.48× faster on
DeepSeek and 1.78× on Qwen3.8. After the DeepSeek stage mechanisms
landed, 4,096 rows prefill the 0731 GGUF 12.1% faster than 2,048 at 8K
and 14.9% at 32K, for 1.28 GiB more fixed memory at a 262,144-token
context (2.70 GiB; 8,192 is no
faster and adds 2.9 GiB more): the
[DeepSeek concurrent report](experiments/deepseek-concurrent/README.md)
has the sweep. Per-chunk time grows with the position; the
[long-context report](experiments/long-context/README.md) checks deeper
contexts. `prefill_chunk` sets another chunk.

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
runtime stopping on SIGTERM or SIGINT, and where configured a stall with
`stall_action = "fail"` or a non-streaming deadline
([below](#progress-and-deadlines))) is noticed while its chat template
renders (an interpreted template asks every few thousand steps or
megabytes), before each prefill chunk and each generation step,
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
context and the default chunks then, DeepSeek's 2,048 rows before the
4,096-row default, greedy streamed requests of an ~8K-token prompt; two
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
output must repeat; a prepared return that kept its graphs through the swap
(models keep their plans and graphs unless the reclaim order took them)
must replay graphs captured before it; an image A's regenerated pixels must equal
its control's. Both
write every number to `--report` as JSON. `swap-table --context-tokens`
accepts 32 and up; the requested context must fit the selected model's
usable configured context. The commands' turns, texts and pairs have no
caps of their own (D-102): the command line's limits bound them, `chat
--max-tokens` generates at most what the context has left after each
turn's prompt, and `--context-text` is read whole and tokenized, so the
text and its tokenization's working set are charged to the request memory
the start reserved, as the route's requests are (below). Run by hand, a
command stops on
SIGINT or SIGTERM at once; the kernel frees its memory and spill files.

    jitllm-runtime [--config FILE] settings [--json]

`settings` lists every configured model's settings (above), each with its
value, source and what it came from, as a table or one JSON object
(`{"models":[{"name","architecture","settings":{KEY:{"value","source",
"basis"}},"ignored":[...],"calibration":NOTE}]}`, the note whether a
calibration record is in force). It reads the configuration, the installed
artifacts and the calibration records only, under the same trust rules:
no process lock, no device memory or context of its own, so it runs in any
build and beside the running service. Run it as the runtime's user. It exits 1 when a model's settings
cannot be resolved (each reason logged), 78 for an invalid configuration.

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
# Limits (D-102): every default permissive; set one to be stricter.
stall_seconds = 120               # no progress this long is reported as a stall: 1 to 2,592,000
stall_action = "report"           # "report" (log, health, service status) or "fail"
idle_seconds = 60                 # a kept-alive connection idle between requests
request_inactivity_seconds = 60   # a head or body with no byte arriving this long: 408
# max_connections = 1024          # absent: the open-file hard limit less 256
# max_queued = 64                 # absent: no count (the request memory bounds what they hold)
# queue_wait_seconds = 120        # absent: a non-streaming request waits its turn
# deadline_cap_seconds = 14400    # absent: no non-streaming deadline
# write_inactivity_seconds = 30   # absent: a client that stops reading gets backpressure
# hang_seconds = 600              # no progress this long, past the unit's allowance: a hang,
                                  #   recovered; absent: the larger of 600 s and five stall times; at least 60
# request_memory_bytes = 2147483648  # absent: no cap (256 MiB set apart, the rest charged to the budget)
# max_body_bytes = 134217728      # absent: 1/16 of the request memory and the largest context's bytes
# stream_buffer_bytes = 4194304   # absent: 1/64 of the 256 MiB floor, 1 to 64 MiB
```

Seconds are 1 to 2,592,000 (thirty days, the watchdog's arithmetic bound),
counts 1 to 2^32 − 1, bytes from a least that keeps the route usable (16
MiB of request memory, 1 KiB a body, 4 KiB a stream buffer) to the type's;
a body larger than the request memory is refused. The start log names
every limit in force.

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
    POST /v1/completions           literal prompt, JSON; echo and token likelihoods
    GET  /v1/models                the configured models (the image among them)
    GET  /v1/models/{id}
    GET  /jitllm/v1/ignored-fields the unknown fields seen (loopback peers only)

It is a strict subset of client-api-baseline.md's Chat Completions profile,
not M5's front door. A request is stateless, as OpenAI's are: the whole
conversation is rendered by the model's template, and the state's tokens
are reused directly or through a matching turn checkpoint (as `chat`
does). The model named is made resident first (a full
swap when another is), and the turn is one request under one lease
(D-093), greedy or sampled, speculative where the model has a drafter.

The inference responses advertise `jitllm-inference-version: 1` (D-100).
The following fields/output describe the chat route; literal completions
have their own contract below.

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
15 s gets a `: keepalive` comment line while queued, swapping or prefilling,
when no earlier response bytes await delivery, well inside the named
clients' 300 s stream-idle bounds (client-api-baseline.md). A stream has no fixed queue wait: it
waits its turn. A stream admitted early can then only fail in the stream:
the model's refusal (the context exceeded) is an in-stream
`invalid_request_error`, and with `stall_action = "fail"` the backend
stalling an in-stream `server_error`, both without `[DONE]`; a stream that
starts within 15 s gets the refusal as a 400 before any header. A
non-streaming request waits silently until its turn: by default as long as
that takes (D-102), with `queue_wait_seconds` set at most that long, then
a 429.

**Intake bounds** (client-api-baseline.md#shared-correctness-and-limits),
checked before any model work (runtime/api.h, runtime/intake_limits.h).
Only real resources bound a request (D-102): what remains names the
resource it protects; counts that only bounded abuse are gone, and each
time limit is an inactivity limit or an owner's option.

Requests' host memory is one pool, the **request memory**
(runtime/intake_limits.h), charged by what each request actually holds or
is about to build, before it builds it: a body as its bytes arrive (not as
declared), its parse's working set (the document's nodes and strings and
the request built from them, up to about thirteen times a body's bytes for
a dense token-ID array), the parsed request for as long as it is queued or
running, a stream's unread output as it grows (released as the socket
takes it), the conversation's rendering and tokenization while they run
(the copy of the messages, the rendering's bound, the interpreter's values
where the template may interpret, the tokenizer's longest window and the
tokens), a literal completion's score rows, a non-streaming response's
text as it grows, and a completed body until the socket has taken it.
Prepared prompt IDs remain charged through their shared ownership by active
work and paused requests. Continuations charge their copied generation and
history, and active request token vectors are charged as completed steps grow
them. Keepalive comments are not queued while response bytes already wait
for a socket, so they cannot grow a paused reader's output indefinitely.

Native prompt and generation sessions, retained branch histories, diagnostic
snapshot histories and Qwen's speculative verify inputs have independent
capacity charges. A dedicated zero-floor host pool charges those capacities
inside the catalog's execution budget, separately from the request pool and
the uncounted margin. Growth funds the replacement while the old allocation
still lives; refusal preserves the completed prefix and dispatches no native
unit. Generation retirement transfers its buffer and charge into the branch.
Geometric capacity growth keeps long generation amortized under pressure;
a refused growth preserves its completed prefix.

Idle host histories are reclaim candidates across all registered models,
including swapped and spilled models. A recency prefix frees just enough
whole catalog extents to cover the requested need. Reclaim measures the
catalog decrease; during native admission it also credits a prospective
extent that deleting idle tokens avoids allocating. Admission always checks
the resulting catalog target again before granting capacity.
Reclaim discards the history with its
retained state; spilling alone preserves and charges the tokens needed to
restore an exact continuation. Active sessions, held continuations, the
branch being admitted and valid diagnostic snapshots are protected. Clear,
forget, discard and expiry release obsolete vector capacity; a diagnostic
snapshot owns its separate history until explicit invalidation. Queued and
worker-owned persistence records hold separately funded token copies through
supersession, invalidation and worker retirement. At startup, kept-record
tokens are funded before decoding against available memory after the startup
reserve, then charged to the catalog before adoption. The startup allocation
report includes `native_token_history_bytes`; ordinary chat and swap-table
reports also expose it and `native_token_catalog_bytes`. The dedicated pool reports
allocated token bytes, and its catalog occupancy rounds their aggregate to
whole 2 MiB extents.

The request pool's first 256 MiB (the floor) are set apart at the start beside the memory
guard's margin. Past the floor the driver charges what requests hold to
the catalog's budget like any other need, through the reclaim order (it
displaces plans, graphs and idle conversations, D-055; request memory is
never reclaimed itself), and gives it back, a 64 MiB slack aside, once
nothing has wanted more for 10 s (so a burst of requests does not grow and
shrink it, displacing idle conversations, again and again); so the state
room is cut by the floor alone, not up front by a share of memory (on
`spark`, two models: 6.70 against 6.45 GiB before the change, the 0.25 GiB
floor). The driver's own charges (rendering, tokenizing, responses) grow it
at once; the I/O and parse threads' charges wait for the driver to grow
it, which it does between units, between a serial request's steps and
while it waits for a reader (only a single unit, such as a swap or a long
prefill chunk, holds them up): a body that cannot grow waits, unread (its
inactivity clock held), and a parse that cannot is set aside, so the
bodies behind it are parsed meanwhile, and tried again only when the grant
changes, a denial is counted or what it wanted fits. What could never fit
(past the capacity: `request_memory_bytes` when set, otherwise the floor
and the state room) is refused before the work with a 413; what the
driver could not grant now with a 503 and `Retry-After: 10`
(`request_memory_busy`); both name `request_memory_bytes`. A prompt that
cannot fit the model's context (text longer than a prompt that fits could
render to, or more token IDs than the context) is refused before it is
queued (400 `context_length_exceeded`). Queued requests may hold at most
three quarters of the request memory's current grant (past it a 503), not
of what it could grow to, leaving the driver headroom for the request it
takes up; the headroom is not a promise, since growth needs the reclaim
order to free room, and an active conversation's state is never taken
for it. After a request that held 16 MiB or more, the C library's free
heap is returned to the system, a refused one's too. The figures in
brackets are a DGX Spark's serving DeepSeek V4 Flash:

| Bound | Value | Over it | What it protects |
| --- | --- | --- | --- |
| Request memory | a 256 MiB floor set apart, growing within the budget to the floor and the state room (`request_memory_bytes` caps it) | 413 past the capacity, else 503 `request_memory_busy`, `Retry-After: 10` | Host memory: everything a request holds (above), whatever the number of requests, queued ones included |
| Request line and headers | 16 KiB, 64 headers | 413 | A connection's head buffer; clients send a few hundred bytes |
| Target | 2 KiB | 414 | The same buffer; routes and a short query |
| Body | 1/16 of the request memory, at most the largest configured context's bytes (its usable context × the longest token's bytes, each escaped to six, plus 1 MiB) and 2³² − 1 (`[client] max_body_bytes`), by Content-Length | 413 before it is read, naming the key (chunked: 501; none on a POST: 411) | Host memory (a JSON body's working set, as charged, is up to about thirteen times its bytes); a body no configured context could hold is refused unread; the JSON parser's 32-bit offsets |
| JSON | depth 64; values and string bytes at most the body's bytes | 400 | The parser's stack (it recurses); a token-ID prompt is one value a token, so a 1M-token one parses; its tables are reserved exactly (a pre-scan counts the values), never doubled |
| Messages, content parts | at least one message; any number of each | 400 for none | — (their bytes are the body's, their tokens the context's) |
| A message's text, reasoning, a text prompt | any length | — | — (the body's bytes; the rendering's and the context's below) |
| `model` | 1 to 64 bytes | 400 | A configured name's limit (D-096) |
| `max_tokens` | 1 to 2³² − 1 at parse; prompt + it ≤ the model's usable context | 400 `context_length_exceeded` | The model's context (`context`, less Qwen3.8's MTP draft rows when it speculates), the checkpoint's ceiling; `max_completion_tokens` has the same bound |
| Prompt | under the usable context | 400 `context_length_exceeded` | As above; counted by the model's own tokenizer and template, which stop at the context |
| A conversation's text | at most what a prompt that fits the usable context could render to: the context × the longest token's bytes (four times under NFC, which composes text at most threefold), at least 1 MiB [DeepSeek V4: 32 MiB] | 400 `context_length_exceeded`, before it is queued | Host memory and the driver's time: a longer conversation cannot fit |
| A rendering | at most four times its messages' bytes, 4 KiB a message and 1 MiB, within the bound above; an interpreted template's values at most that and 16 MiB | 400 | Host memory, charged before rendering |
| Tokenization | the text refused unread past max_tokens × the longest token (four times under NFC); encoded in windows of about 1 MiB, cut where no pre-tokenizer or NFC joins across (tokenizer.md), 96 bytes of working set a window byte | 400 `context_length_exceeded` for text whose longest stretch without a line break or space between words could never be held (its window past the request memory's capacity); 503 for the request memory now | Host memory: the tokenizer's working set follows its longest window, not the text |
| `temperature`, `top_p`, `top_k`, `min_p` | [0, 2], (0, 1] (`top_p` not rounding to 0 as a float), −1 to 2³¹−1, [0, 1] | 400 | OpenAI's ranges and vLLM's for `top_k` and `min_p`; sampling.h's, which takes floats |
| `seed` | a 64-bit signed integer | 400 | OpenAI's type |
| `stop` | any number of non-empty strings, any length | 400 for an empty one | — (the body's bytes; matched together by one automaton built at parse and shared with the output, a few steps an output byte whatever their number; its tables, a node a distinct prefix, charged to the request memory; the held-back text is at most the longest string) |
| Unknown fields | 64 names a request, 64 bytes a name; 256 names kept | ignored | The diagnostic table's memory, whatever a client sends |
| Head, body | no byte arriving for 60 s (`request_inactivity_seconds`), however long the whole takes | 408, naming the key, then the connection closes | A dead client's buffer and descriptor; a slow one that keeps sending is never cut off |
| Idle connection | 60 s between requests (`idle_seconds`), told to the client (`Keep-Alive: timeout=60`) | closed | An idle connection costs a descriptor and a small buffer (at most 16 KiB each way: a larger one, a body's or a response's, is freed once its request is done); a minute spans a client's pauses between turns |
| Connections | the open-file hard limit less 256 (`max_connections`; 524,032 under systemd's default hard limit of 524,288) | the oldest idle one is closed for the new one; with none idle, 503 | Descriptors: the open-file limit is raised to its hard limit (to `max_connections` + 256 when set) |
| A stream's unread output | 1/64 of the request memory's floor, 1 to 64 MiB [4 MiB] (`stream_buffer_bytes`), charged as it grows; nothing reserved while the stream waits its turn | the request pauses at its next completed step until its client has taken it (backpressure), and also when the request memory cannot take more; nothing is dropped; if it keeps queued requests waiting for 10 s it yields its place (below) | Host memory: a reader that stops cannot grow a buffer; with `write_inactivity_seconds` set, one that takes nothing that long is dropped, its generation ending at the next step. The kernel's socket buffers take several megabytes first, so on loopback a reader that stops is noticed only after that much output |
| Queue | no count, no wait (`max_queued`, `queue_wait_seconds`); queued requests hold at most three quarters of the request memory's current grant | with them set, 429, `Retry-After: 10`, `x-should-retry: true`; past the share, 503 `request_memory_busy` | The driver's headroom in the request memory, and descriptors for the connections |
| A response | a non-streaming one's text and body, and a literal one's score rows, within the request memory | 413/503 for the request memory | Host memory, charged as they are made and until the socket has taken the body |
| A request | no deadline; a stall (120 s without progress, `stall_seconds`, each unit allowed its expected time) is reported (`stall_action`) | with `stall_action = "fail"`, 504 (in-stream error when streaming), then 503 to every request until the backend moves; with `deadline_cap_seconds` set, a non-streaming request's scaled deadline, 504 | — (detection is not a limit; [progress and deadlines](#progress-and-deadlines)) |

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
(415). Errors are OpenAI's `{"error": {message, type, param, code}}`; a
refusal for a limit names its `[client]` key. A client that disconnects
or the runtime stopping (SIGTERM or SIGINT, 503), and where the owner
configured them a non-streaming request's deadline (504) or a stall with
`stall_action = "fail"` (504, [below](#progress-and-deadlines)), ends the
request while its template renders, at its next prefill chunk or
generation step, and after a swap before any model work (a failed stall
answers the client at once, without waiting for that step); the state
keeps what it processed ([cancellation](#prefill-chunks-and-cancellation)),
and the service goes on. By default a stall only reports, and a client
that stops reading a stream only pauses it (backpressure, above). A client that shuts only its sending side after a whole
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
whole is a disconnect. Work that hung and was cancelled ends the
requests that needed it with a 503 `backend_hung` and the model is reset
in place ([hang recovery](#hang-recovery)); any other failure of the
node itself (a job that failed) ends the request with a 500 or 503 and
stops the service with status 1. A request whose conversation state does not fit the
execution budget even alone is not one: it fails with a 500 and the
service goes on ([state capacity](#state-capacity-in-a-cohort)).

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
thread) takes the queue first come, first served, sharing compatible Qwen and DeepSeek chat
requests in completed units and running other requests one at a time. It
watches the runtime's signals (a signalfd) between units. It never touches a
socket: it appends each
response's bytes to a buffer, whole events at a time, which the I/O
thread writes out as the client takes them. A connection that closes
(not one only half-closed, above) marks its request gone; the generation
ends at its next step and the request's lease is released as the backend
returns, while the buffer
lives until the driver lets go of it. Parsing a request's JSON (up to the
body limit; tens of MB a second for dense token-ID arrays) is the one
piece of CPU work outside the driver: a body under 64 KiB is parsed on the
I/O thread, a larger one on a parse thread of its own, so a large body
holds up no other connection, keepalive or the watchdog. A stream's
response buffer past `stream_buffer_bytes` pauses its request
(backpressure): on the serial path the driver waits at that completed
step; a cooperative cohort leaves the paused member out of its units and
the others go on. A paused request that keeps queued requests waiting (on
the serial path, or in a cohort that must drain for another model or is
full) for 10 s yields its place: its generation ends at that completed
step, its conversation state stays as a finished turn's (resident, or
spilled when memory needs it, under the reclaim order), and it is parked
with its response's progress (its stream, its held text, its decoder's
partial character) until its client reads: only then does it go to the
back of the queue, so a client that still is not reading never has its
model swapped back in only to yield again. Taken up again, its state is
reused (restored when spilled, prefilled only for what it lost) and its
generation continues from its last token, nothing chosen or sent twice.
One whose client never reads again holds only its parked place, its unread
output and its connection (or is dropped, with
`write_inactivity_seconds`). Rendering
runs on the driver too: an interpreted template asks its request's
cancellation every few thousand steps or megabytes, and a configured
deadline is set before it renders, but other requests wait for it, as for
any unit.

An optional internal `CooperativeBackend` interface lets the same driver own
as many stable request frames for one model as it has request slots
([below](#request-slots)), admitting new work between completed units.
Each request retains its own response, deadline, cancellation
and backpressure (a member whose client is behind is left out of decode units;
when every runnable member is, the backend declares a paused unit and the server
waits for a reader, a new request or the runtime stopping);
switching models pauses the active group at a completed unit after its
resident-work quantum (below). Retirement must prove that no
work still borrows a frame before it can be freed. The production Qwen chat
backend's active requests decode in shared waves at draft depth 2 (a lone
request keeps its adaptive depth; past two, drafts run per request). A
request past the model's slots, or one that finds no memory for its
state beside its peers', waits for a retirement and then refills
the group. Literal completions join the same cohort, with their own raw
prompts and score metadata.
The prompt with the fewest tokens left to prefill (after what its
conversation's reuse would actually keep: a stale history that only shares
a few tokens counts for nothing) gets the next prompt unit, the oldest of
equals. So a short prompt is not held behind a long one that arrived
first, and prompts of equal length finish one after another, each
streaming its first token in turn. A prompt passed over for 12 other
prompt units that prefilled rows (a reuse or checkpoint unit prefills
none) goes next, so a long prompt is not starved, unless the shortest is
a started prompt with at most 256 tokens left, which finishes first. A
generating request waits
for at most one prompt unit between its waves
(`runtime/cohort_schedule.h`;
[prompt order](experiments/deepseek-batching/README.md#prompt-order-adopted)).
Compatible small-row target/draft products of consecutive requests share
weights while their rows fit sixteen (four requests of a depth-2 verify's
three rows, five at most; more requests form more groups; MXFP8 and routed
outputs bit for bit); independent
attention, recurrence, logits and commits remain branch-owned. Eligible groups
of three- or four-row BF16 target heads use one ordinary MMF product of up to
sixteen columns with paid input concatenations. Compatible multirow HC
BF16 products also share immutable weights through their original cuBLAS path,
with independent preparation and nonlinear mixing. Unsupported shapes keep
their original products.

GGUF Qwen's plain, one-row waves also join compatible dense and routed
`jitllm.vecq` products, including the quantized full head. Paid F32 input
packing and Q8 preparation feed the existing one-token reduction path;
joined routed groups fit the 128-pair kernel bound (twelve requests with
ten experts each; wider waves form multiple groups). Multirow products
keep their original launches. Full heads and final state match unjoined
waves byte for byte with eager execution and graph replay
([GGUF qualification](experiments/qwen38-gguf/README.md#one-row-gguf-waves-2026-10-03)).

The production DeepSeek chat backend's active requests (one native slot
each, `engine/dsv4_runner.h`; [request slots](#request-slots)) decode in
waves: plain, one row each. With DSpark, a wave of two or more is either
each request's own draft block then one joined verify of every request's
rows (its natural four up to four requests, sixteen in all; past four,
each its share of sixteen, at least two, so DSpark takes at most eight
slots), or one plain row each with the drafter still fed. The tokens
draft-verify waves accept, against a per-width cost measured for the
model (widths 2 to 8), choose between them (`execution/adaptive_wave_mode.h`). The
choice reads no clock, so the same requests in the same waves since the
service started choose the same forms, and a wave with a sampling member
always speculates. A lone request, and a request whose verify is one row
(a mask-width boundary or its last token), keeps its ordinary step. The
model's `wave_form` key fixes the form of waves of two or more instead
(`"auto"`, the default, chooses as above). With `"speculative"` every wave
is draft-verify, so each request's reply at four requests equals its reply
alone: the control a cohort is checked against. With `"plain"` every wave
is plain decode, sampling members included, while a lone request still
speculates (to compare the two wave forms). Neither is a tuning knob.

Each request's rows, drafts, accepted tokens and state equal the same steps
alone bit for bit (wave checks at 2 to 8 slots; past four requests a
DSpark verify takes fewer rows than alone, so its steps are not the same
steps). Without a drafter, concurrency itself therefore does not
change a reply. With DSpark it can: a greedy request may take plain steps in
a wave where alone it speculates, and the two forms' arithmetic differs, so
its reply at four requests may differ from its reply alone (both greedy;
a speculative token is the plain argmax or a near-tie within the verify's
noise). Which waves a request joins follows its peers' arrival (equal
prompts prefill in arrival order), so such a reply may also differ between
two runs whose concurrent requests arrive in another order; a sampled
request, always speculating, does not. Two more things can change a reply: a request
preempted for state capacity whose state could not be spilled rebuilds it
by prefill (below; one spilled resumes exactly), and a turn that reuses a
cached prefix continues from whichever free branch it can reuse the most
of (`Llm::ReusablePrefix`: live history continued, or a turn checkpoint;
with none, an empty branch, then the least recently used, so a new
conversation leaves idle ones alone while an empty branch is free), whose
earlier prefill may have been chunked differently
([report](experiments/deepseek-batching/README.md): C4 +29.0% plain, +4.5%
DSpark on the matched 7K protocol).
A wave needs every layer in the fast plan's fused form, which takes HC
mixing weights in F32, F16 or BF16 (through its own mix kernel) or
quantized (through GGML's product, a request at a time), but needs each
layer's expert products to be `jitllm.vecq` types with gate and up alike in
type and shape. An artifact whose experts are not is still served, one
request at a time, and the start logs `model NAME: serves one request at a
time (no waves of N requests: layer N: ...)`. A model is never refused
only because its artifact cannot batch.

### Request slots

A model's request slots are how many of its requests run at once, each
with its own conversation branch and native state; they are also how many
idle conversations it keeps for reuse. Their number resolves as D-103's
settings do (D-104): `[models.<name>] max_slots` (1 to 16) overrides it;
otherwise the fallback, each model's measured knee, applies until
calibration on the machine measures its own: **4 for DeepSeek V4 Flash and
4 for Qwen3.8 Flash Next**. The knee is where another slot stops raising
the completed-token rate enough to pay for slowing every request. With
plans warm (a second burst in one service), past four DeepSeek DSpark
(its waves of six to eight chosen plain, above) gains 9.3% at six and
2.6% more at eight with short prompts, nothing with long ones, while each
request decodes 24% and then 12% slower; Qwen3.8 gains nothing at any
width (its depth-2 verifies split into groups that each read the
weights, and a row's routed experts are mostly its own) while each
request decodes 36–48% slower ([request
slots](experiments/request-slots/README.md#warm-plans)). Those historical
DeepSeek gains use the original artifact and unmatched counts. A later
[matched community-artifact control](experiments/deepseek-slot-knee/README.md)
always pays six requests: six slots gain 16.36% / 21.79% short throughput
with 25.74% / 22.88% higher median latency; a long cell gains 6.70% with
9.30% / 10.26% higher mean/median latency. It retains four for the primary
latency-focused profile, while six remains a throughput/tail option. Its
second burst includes automatic width exploration and acceptance state,
not just warmed plans. A larger cap
also plans each wave composition when first met and meets far more of
them, so planning goes on through the service's life and grows steeply
with the cap: across two bursts Qwen3.8 planned 1.8 s at four slots and
68 s at sixteen, holding 0.2 and 2.7 GiB of plans (charged inside the
budget and reclaimable, but taken from conversation-state room). The start logs the
value and its source (`model NAME: 4 request slots (fallback: the knee
measured on a GB10 (D-104))`). Sixteen
is the most a model takes: sixteen one-row decode steps fill the joined
products' sixteen rows; with DSpark, DeepSeek takes at most eight (a
verify of at least two rows a request), and a larger `max_slots` is
lowered with the reason logged.

Below the cap, memory decides. While its model is resident, a request
joins its model's running requests only when the execution budget holds
its prompt's state and what its peers' prompts have yet to take (each
`Llm::StateBytesThrough` its tokens and first step, less what its branch
holds already, which it continues or clears): free in the budget, or
freed through the reclaim order (idle conversations spilled, stale plans
and graphs dropped; all of it or nothing, `Server::RoomFor`), never
spilling the branch the request takes. That reclaim runs in the
cooperative backend's `Start`, between completed units, the one native
work admission does. Otherwise the request waits first in the queue,
logged (`waits for memory for a request slot`), and is looked at again
when a running request retires. A lone request always starts, and one
that arrives while its model is being made resident joins as before. In
a matched A/B with Qwen3.8 at 1.45 GiB of state room, admission kept
requests waiting in the queue rather than in the cohort at a 0.6% lower
rate, the median request completing 5.8% sooner.
Under pressure the cohort shrinks as before ([state
capacity](#state-capacity-in-a-cohort)): idle conversations spill first,
and a running request is set aside (spilled) only when no member can go
on. A slot's fixed buffers (its verify snapshot, output staging and
descriptors) are set up at start for the cap: about 16 MiB a slot for
Qwen3.8 and 4 MiB for DeepSeek with DSpark, measured. Its state, which
grows to hundreds of MiB or GiB with its conversation, takes memory only
as it is used and goes back through the reclaim order when idle.
Qwen3.8's wave workspace is sized from its widest wave as measured at
setup, not from four prefill chunks: 6.80 → 1.53 GiB at four slots, so
more slots cost no workspace and its conversation-state room grows by
5.45 GiB.

Cancellation ends only its request at a completed boundary. Before releasing a
frame or admitting its replacement, an explicit native stream fence proves
that copies and jobs have retired. Unknown completion stops shared execution
and retains the borrowed owners. The image pipeline and models without
native generation waves retain their ordinary serial entry points.

### Pending model switches

When a different model reaches the queue's head, the running cohort stops
refilling. After `[client] model_turn_seconds` of resident work (default
30, configurable from 1 to 2,592,000 seconds), its members pause at their
next completed unit. Initial swap time is excluded from that turn. The
30-second fallback is a scheduling policy, not a measured throughput knee
or a request deadline. A full set of slots is still checked for a pending
other model; same-model newcomers wait for a slot without preempting a
running response.

Prompt and generation sessions retire through the normal native fences,
including a verify's owed restore. Their original response buffers,
sampling seed, decoder/stop state and usage stay with their continuations.
A continuation holds its exact model-owned branch identity; prefix matching
does not choose a replacement. Its initialized state and spill files cannot
expire or be deleted while the request waits, although its GPU residency
and leases are released for the swap. Partial prompts restore the completed
prefix and prefill only the remaining input. That prefix does not become
new cached-token credit for its own request. Generation resumes from the
already reported anchor without choosing or sending it again. A final
prefill checkpoint chooses that anchor before pausing; runnable generating
peers restore before their resumed decode wave begins.

One substitute cohort runs: compatible requests ready at its first
admission pass may join, then it accepts no later refill and cannot itself
be paused for another model. The original cohort resumes ahead of later
arrivals after that turn ends. Disconnects, individual substitute errors
and shutdown retain the ordinary completion/retirement rules. A failed
swap that cannot fit its protected state within the spill budget refuses
the substitute and leaves the original continuations intact. Returning
an admitted continuation to the queue cannot impose the initial
`queue_wait_seconds` refusal again. D-069's M3 amendment records this policy;
no response is truncated by its turn interval.

### State capacity in a cohort

Admission to a cohort does not reserve conversation state: each member's
state grows as its prompt and generation run, and every member's state is
leased while the cohort is selected. An idle branch keeps its finished
conversation's state (its reuse cache) resident until the reclaim order
spills it, or a swap does, or its retention ends. A member's growth (or
its spilled state's restore) can therefore be refused when the execution
budget is full. Such a
refusal is typed: the runner reports it (`WorkError::kOverBudget` from the
acquisition, `Llm::StateRefusedFor`) only when it came before any dispatch
and left the state usable as it was, with any fresh zero pages that
completed retained and protected (Qwen3.8's and DeepSeek's runners, each
slot's `state_refused`; a DeepSeek decode step grows its state in its
preparation, through the most rows its verify may take, so a shared wave
never grows, nor refuses, a member). The cooperative backend defers it
(`PromptSession::Advance` and `RunGenerationWave` with `defer_capacity`):
the session stays resumable at its completed prefix, and its unit runs
again later. Every other failure ends its request as before. The policy
(`runtime/cohort_capacity.h`):

- The reclaim order goes first (above): what the refused units asked for
  is freed from plans, graphs and idle conversations outside the cohort,
  spilled, in the order's priority, and the refused member runs again;
  only when all of it was freed (otherwise nothing is taken and it
  waits). A spilled conversation's next turn restores it exactly.
- Then a refused member waits while any peer holds state, a waiting peer or
  one about to retire included. It retries when a peer retires (its state
  is then idle, and reclaimable) or is preempted.
- A member refused while no other branch holds state cannot fit alone and
  fails with the refusal (a 500 naming the chunk or step, as a lone
  request's).
- When no member can go on (every one holding state waits and none is about
  to retire), the youngest waiting member is set aside (preempted): its
  sessions end at their completed boundary, the host keeps its tokens (the
  prompt, and in a generation every generated token but the unprocessed
  anchor), and its state is spilled (`Llm::SpillSetAside`). The others
  retry at once. A preempted member begins again, the oldest first, once
  no other member can run or waits: its first unit restores its state
  from the spill file, exactly as it left, and a resumed generation
  continues from its anchor without choosing or streaming any token again
  (sampling stays keyed by absolute position). Its state is exactly as it
  left, so its reply equals an uninterrupted run's wherever the family's
  replies do not depend on wave composition: DeepSeek's waves keep each
  request bit-identical to running alone, while Qwen3.8's replies under
  concurrency vary with arrival timing, set aside or not. (The review's
  set-aside runs reached it end to end on DeepSeek under memory pressure,
  and every resumed reply equalled its control's.) Only a state that
  cannot be spilled is cleared and rebuilt by prefill (whose rounding may
  then differ; tokens already streamed never change).
- While any member waits or is preempted, no new request joins the cohort;
  it stays first in the queue. Admission reserves nothing, but a request
  whose prompt's state does not fit beside its peers' waits in the queue
  instead of joining ([request slots](#request-slots)); one that grows
  past its estimate waits as above.

So only a request that cannot fit alone fails for capacity; under pressure
the cohort serves fewer requests at a time. A waiting member keeps its
deadline, cancellation and client checks between every unit. A cohort of
one reclaims first, then fails with the refusal.

The serial path (`Complete`: chat or literal requests for models without
a cohort) runs alone, as a cohort of one. Its prefill chunk,
scored row, decode step or spilled state's restore refused this way
(DeepSeek's runner types the same refusal) has the reclaim order free what
it asked for and runs again (`Llm::set_capacity_reclaim`); once nothing is
left, the request fails with the refusal (a 500 naming the chunk or
step), its state usable at its completed prefix, and the service goes on.
Any other failure still stops the service, as below. The runtime
logs each wait, preemption, reclaim and refusal with slots, token counts
and bytes only.

An LLM's stable `Branch` owns its prompt history, sampling key, session guard,
turn checkpoints and adaptive draft-depth policy. Qwen3.8 and DeepSeek map a branch
to each independent native request slot ([request slots](#request-slots)),
with one shared set of model weights. Other
families retain their default branch. Each resumable generation session forwards
its completed units to its own branch; saving, restoring or clearing a branch
does not change another branch's history or policy. Prefix matching chooses a
reuse opportunity among free branches; it does not identify a conversation.
Every slot both runs a request and retains an idle one's state for
reuse; a model whose artifact cannot batch has one.

`Branch::BeginPrompt` owns a bounded prompt copy without native work. Its
`PromptSession` advances one reuse/restore, prefill chunk or turn-checkpoint unit
at a time, allowing the cooperative backend to schedule peer decode
between completed units. The ordinary `PreparePrompt` drives this same session.
Cancellation preserves the completed token prefix; a failed helper that discarded
host history still requires a native clear before reuse. A branch admits only
one prompt or generation session. Explicit `Finish` releases host ownership;
the runner separately proves native completion.

The prompt-session slice passed all 1,242 locked Spark-native tests, including
six focused prompt controls and 256 GPU tests, plus SDK format/tidy, portability,
REUSE and 1,139 embedded-header checks. Spark B's completed check receipt is
`5a3b8826ce72f00f8338f97e85225f546ff673747eec9b1dedcd7781bc16691c`
under `~/scratch/m3-cooperative-prefill-records/check-r1/`; its unchanged
source map is `f9de799fc86c4aaf1d09be9a85ed2c4fcb3e43068c871bfcd8237122ba34ea20`.
The deferred workstation checks remain part of the final optimization gate.

## Progress and deadlines

A chat request has no fixed deadline (D-097, the owner's note of
2026-09-29): the 600 s it had failed healthy long prefills (DeepSeek's
128K prompt stopped at 108,544 of 128,821 tokens,
[long-context](experiments/long-context/README.md)), and the
response time is ours to own, since the runtime can tell a backend that
is working from one that is not. Since D-102 (2026-10-03) no time limit
stops legitimate work by default: the watchdog detects a backend that
makes no progress and reports it, and a request runs until it is done or
its client leaves. An owner may opt into the earlier behaviour: with
`stall_action = "fail"` a stall fails the request, and with
`deadline_cap_seconds` a non-streaming request, whose client hears nothing
until the end, has a deadline scaled to its work (`runtime/watchdog.h`,
vendor-free and tested on a synthetic clock and with the fake backend).
A genuine hang is recovered: the stuck work cancelled, the model reset in
place, and only when nothing less frees it the process restarted
([hang recovery](#hang-recovery)). Through the route on `spark` (2026-09-29,
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
is allowed for whole. A request paused for its client to read
(backpressure) is not watched: the phase says so, and a pause is never a
stall.

**A stall.** The I/O thread, which never waits on the model, watches the
beats. When a unit passes its allowance it marks the backend unhealthy at
once, whether or not the backend ever returns, and says so (the log, the
service manager's status, `health()`). By default (`stall_action =
"report"`) that is all: the running request, the queue and new requests
go on, and a backend that was only slow finishes them. With
`stall_action = "fail"`:

- the running request ends: a 504 `backend_stalled` before its headers,
  or after them an in-stream `server_error` event with that code and no
  `[DONE]`;
- its generation is cancelled the normal way: marked ended, as when a
  client leaves, so it stops at the backend's next step and its lease is
  released as the backend returns;
- the queued requests get a 503 `backend_unresponsive` (`Retry-After:
  10`, `x-should-retry: true`; in-stream for a stream that started);
- until the backend's next beat, every new chat request gets that 503 at
  once rather than waiting behind a backend that may never return.

**Unhealthy.** Until the backend's next beat (the unit it hung in
returning) the health says so; `GET /v1/models` always answers. The next
beat makes it healthy again, and the log says so. A unit that never
returns does not hang the service: it is a hang, recovered as below.

### Hang recovery

D-102 (the owner, 2026-10-03: "A genuine hang should be recovered …
minimizing data loss but recovery is the main goal"). One hang ladder
(`runtime/hang_ladder.h`, vendor-free and tested on a clock of its own)
tells a hang from slow work and escalates; the chat route's I/O thread,
which never waits on the model, and the node's driver both drive it.

**What is progress.** Any of: a unit's beat (above), the node's progress
count moving (`engine::PagedNode::progress`: every completion a lane
publishes, a fence seen complete, a read or write landed, a VMM
operation; and every wait of the driver's that ends), or the driver's
long CPU work beating its pulse (`base/work_pulse.h`: a template's
rendering at each of its cancellation checks, a tokenization at each
window; a graph's capture and instantiation run on the lane thread
inside a device job, covered by that unit's allowance). **Work under way**
is a unit (not idle, not paused for a client) or a wait of the driver's
on the node between units (housekeeping, a teardown). **A confirmed
hang** is work under way with no progress of any kind for `hang_seconds`
(absent: the larger of 600 s and five stall times; at least 60) that has
also passed its unit's allowance. So healthy slow work is never one,
however long: a swap from a slow disk keeps landing reads, and one long
job (a wide prefill chunk) is allowed its expected time at the floors.

**The rungs**, each logged with why:

1. **Cancel the stuck work.** The driver's wait on the node cancels the
   request it waits for (a program, a request's lease, a step;
   `engine::Patience`): the wait fails, flagged as a hang's. Outside any
   node wait, the driver's CPU work is asked to stop at its next
   cancellation check (its pulse): a rendering or tokenization fails with
   a 503 `backend_hung`. **The cancellation has drained** once the driver
   moved again and no storage operation submitted before it is still in
   flight (`engine::PagedNode::oldest_io`). A wait's cancellation ends
   only the request's interest: a read still queued is cancelled, but one
   the drive (or a hung mount) holds is not (io_uring's cancellation is
   best effort), and the wait returns without it. Until that read
   completes, nothing the work touched is freed or reused. A request's
   direct step (D-106) is the one wait that does not return on its
   cancellation: its request ends, but the driver goes on watching the
   step's fence, since the step's memory stays leased until it is seen.
   A step that completes then drains rung 1 as any wait does; a device
   that truly hangs never lets the driver move, and the grace leads to
   rung 3.
2. **Reset the model in place**, only once the cancellation drained. The
   requests that needed the work fail (a 503 `backend_hung` to retry; a
   cohort's members all), and once they retire the model's stream is
   fenced (proof that nothing it queued still runs), every slot the
   failure may have touched has its state discarded (one spilled before
   it is kept), its cohort's fault lifts, its plans and graphs are
   dropped, its calibration samples not yet recorded are dropped (the
   hung unit's time is no measure, D-103), and its weights and the state
   it kept are evicted (their records written, D-105), so its next
   request loads it whole. The service goes on; a swap whose reads hung
   fails only the requests that needed it, as any failed swap does, once
   its cancellation drained.
3. **Restart the process.** When the cancellation does not drain within
   the shorter of `hang_seconds` and 60 s (a read the drive still holds, a
   device that hangs: a fence that never completes cannot be cancelled,
   CPU work that never reaches a cancellation check), when a model hangs
   again before it served a request since rung 2 reset it (resetting it
   again would loop), or when rung 2 cannot reset the model, the ladder's
   last resort runs on the thread that saw it: the conversation
   records already queued get at most 10 s, and the process exits with
   status 1 for `jitllm.service` to restart it. Nothing is torn down (the
   driver may be the thread that hangs); requests under way and queued
   end with their connections. Conversations whose whole state was on
   disk survive the restart ([kept](#conversations-kept-across-a-restart));
   resident ones prefill again.

The serving commands have no ladder: their waits' own patience cancels
after ten minutes without progress and aborts after ten more.

**Measured** (`spark-b`, GB10, 2026-10-03; [hang
recovery](experiments/hang-recovery/README.md)). Through the service
(Qwen3.8 and DeepSeek V4 Flash registered, `hang_seconds = 60`): with the
node's reads held by a test hook (`JITLLM_TEST_HOLD_READS`) in its
cancellable form (reads still queued), a new conversation's state growth
made no progress; rung 1 cancelled it at 60 s and it drained at once;
rung 2 failed its request (503 after 62.6 s) and reset and evicted
Qwen3.8; once reads flowed the next request reloaded it and continued
the earlier conversation from its kept state (172 of 192 prompt tokens
cached); the process never exited. With the hook in its default form (a
read the drive holds, which its cancellation does not end), under a
systemd user unit: rung 1 cancelled the wait at 60 s and the wait
returned, but the read stayed in flight, so rung 2 never reset the model;
rung 3 exited at 120 s, systemd restarted the service (ready 131.8 s
after the request), and it adopted the kept Qwen3.8 conversation, whose
next turn reused all 265 tokens and replied as the uninterrupted control
did. On the device (`cuda_paged_node_test`): a held page-in, and a
request's lease, cancelled after 0.5 s quiet and drained; a page-in on a
drive that holds its read returned failed while the read stayed in
flight (`oldest_io`), and loaded whole once it completed; a stream gated
on a value that never comes could not be drained, and the patience
reached rung 3 after another 0.3 s quiet (recorded in place of the exit,
then the gate opened and the node went on).

**Health.** The watchdog keeps the backend's health: healthy or not, the
phase (idle, starting, swapping, prefilling, decoding, finishing, waiting
for a client to read), the last progress, the stalls counted and when the
last began. The log has a line at each stall and each recovery, the
service manager's status line (`systemctl status jitllm`) says it, and
`api::Server::health()` holds it for M5's management listener.

**Deadlines.** None by default (D-102): a stream runs until it is done or
its client leaves (the half-close rule above unchanged), keepalive
comments holding its client's idle timeout, and a non-streaming request
likewise, its client's own timeout (600 s in OpenAI's SDKs) the only clock.
With `deadline_cap_seconds` set, a non-streaming request's deadline, from
when it starts running, is

    min(deadline_cap_seconds,
        stall_seconds + 3 × (swap bytes / 1 GB/s
                             + prompt tokens / prefill_floor_tok_s
                             + max_tokens / decode_floor_tok_s))

counting the whole prompt (cached tokens too) and the request's
`max_tokens` or, without one, the rest of the context. At the default
floors, DeepSeek resident with a 131,072-token prompt and `max_tokens`
4,096: 120 + 3 × (1,310.7 + 819.2) s = 6,510 s, about 1 hour 49 minutes.
The floors feed only this estimate and the stall allowances.

**The queue.** No count and no wait by default (D-102): each request waits
its turn, a stream started after 15 s and held by keepalives, a
non-streaming one in silence; the request memory bounds what queued
requests hold (each parsed request is charged to it while it waits) and
descriptors their connections. With `max_queued` set, a request past it gets
a 429 with `Retry-After`; with `queue_wait_seconds` set, a non-streaming
request that waits longer gets the same. With `stall_action = "fail"`, a
stall refuses the queue (above).

**The defaults.** `stall_seconds` 120 (1 to 2,592,000): a decode step takes
well under a second and a default chunk seconds at 8K (the table above),
so two minutes without either is a stuck backend, not a slow one. The
floors, 100 prefill and 5 decode tokens a second, are below every speed
measured here at 8K (DeepSeek's slowest prefill in the table, 220 tok/s
at 256-row chunks; its plain decode at 8K, 19 tok/s, plan.md), but not
at DeepSeek's deepest context, where the stall time and the margin of
three carry it. On the fast plan, measured: 2,048-row chunks' longest is
14.7 s in a 1,048,512-token prefill
([final context](experiments/m3-final-context/README.md)); 4,024-row
chunks (what a 1M context caps 4,096 to) take 5.9–6.6 s in the first
32K positions and 9.4–9.8 s at 233K–253K (`spark`, 2026-10-02, DSpark,
context 1,048,576), about 0.015 s more per 1,000 tokens, which
extrapolates (not measured) to about 21 s at 1M, inside the 240.7 s
allowance (120 + 3 × 40.24). The first, more pessimistic extrapolation,
from before the fast plan: a 2,048-row chunk's device time grows about
0.14 s per 1,000 tokens of context (4.7, 7.9 and 12.3 s
at 8K, 32K and 64K, [long-context](experiments/long-context/README.md);
the same line gives 794 s of device time for the 128K stream, which
took 836.8 s to its first token), so about 40 s (52 tok/s) at 262,144,
inside its 181 s allowance (120 + 3 × 20.48); its decode step about
0.25 s there (4 tok/s), inside a configured deadline's 0.6 s a token; its
whole 258,856-token prefill about 45 minutes, inside such a deadline's 2
hours 11 minutes or more. A
swap's allowance covers a disk that pages in at a third of 1 GB/s or
more; a slower one (a hard disk, a NAS) is reported as stalled unless
`stall_seconds` is larger (with the default report, nothing fails), and
since its reads keep landing it is never a hang.

## Literal completions and likelihoods

`POST /v1/completions` accepts one `prompt`: raw UTF-8 text or a nonempty
array of exact nonnegative int32 token IDs. It applies no chat template,
reasoning split or prefix reuse. Each request starts from its own cleared
branch, under the same bounded queue, browser guards, swap and request
lease as chat. Each model's literal requests share native cohorts with
its chat requests. Scored prompts advance in completed
one-row teacher-forcing units, interleaved with peer work; generated rows
join the ordinary plain or speculative decode waves. Idle chat branches'
retained state gives way to it when the state's
capacity refuses it; one that cannot fit alone fails with a 500
([state capacity](#state-capacity-in-a-cohort)). Admission validates IDs
against the actual model vocabulary, refuses unused padding IDs and checks
prompt plus requested output against the usable context. Supplied EOS/stop
IDs are ordinary teacher-forced inputs.

Besides the shared model/sampling/stop controls, it honors `echo` (false),
`logprobs` and `prompt_logprobs` (null, or integers from 0 to the model's
vocabulary size; was 0–5 before D-102),
`add_special_tokens` (true) and `return_tokens_as_token_ids` (false).
Text adds BOS only when the tokenizer enables it and the prompt does not
already start with it; no EOS is added. Exact IDs are never changed.
An empty text prompt needs an enabled BOS; otherwise it is a 400.
`max_tokens` defaults to 16 and accepts zero, including a prompt filling
the whole usable context when no output is requested. `stream: true`,
multiple prompts in one request,
suffix insertion, embeddings, truncation and restricted token-ID sets are
refused; `n`/`best_of` must be 1, `min_tokens` 0, `ignore_eos` and beam
search false, and `skip_special_tokens` true. Other known unsupported
controls follow chat's off-value rule; unknown names are counted as before.

For supplied-token scores without generation or echo:

```json
{"model":"your-alias","prompt":[1,42,73],"max_tokens":0,"prompt_logprobs":1}
```

The JSON object is `text_completion`; `choices[0].text` is empty,
`finish_reason` is `stop`, and `usage.completion_tokens` is zero. Its
`prompt_logprobs` has one entry per supplied token: first null, then objects
keyed by decimal token ID. Each value has `logprob`, `rank` and
`decoded_token`. The actual supplied ID is always included, with the
requested highest-scoring alternatives; zero requests only the actual ID.
These are vLLM's [prompt score and zero-token conventions](https://github.com/vllm-project/vllm/blob/v0.12.0/vllm/entrypoints/openai/serving_completion.py).

For the legacy OpenAI echo interface:

```json
{"model":"your-alias","prompt":[1,42,73],"max_tokens":1,"temperature":0,"echo":true,"logprobs":1}
```

`choices[0].logprobs` contains equally sized `tokens`, `token_logprobs`,
`top_logprobs` and `text_offset` arrays. Echo includes supplied-token rows
before the generated suffix, so lm-evaluation-harness's
[`[ctxlen:-1]` parser](https://github.com/EleutherAI/lm-evaluation-harness/blob/ddd67220430a2470529f25fd5c05a576ca1057a0/lm_eval/models/openai_completions.py)
sees the supplied continuation and excludes the one generated row.
The first supplied token has null score/top scores because no preceding
distribution exists. With text and an implicit BOS, that BOS remains the
first null metadata row but is omitted from echoed text. `echo: false`
omits prompt rows from the legacy arrays; `prompt_logprobs` remains usable
independently. An absent `logprobs` returns a null legacy object.

Scores use double-precision log-sum-exp over the natural F32 target logits,
before temperature or sampling filters. Padded negative infinity is excluded
from normalization; NaN, positive infinity, an all-masked row or a masked
actual token produces an error, never an invented finite score. `top_logprobs`
includes the actual token even outside top-k; top-1 refers to the unfiltered
target maximum, with stable lowest-ID ties. No draft probabilities are scored.
A generated EOS/stop token retains its score and usage count while its text
is omitted. A stop string truncates returned text; the tokens that completed
the stop retain their scores, and later accepted verify rows are not emitted.

`text_offset` counts Unicode characters in the decoded stream, not bytes
or token-ID label lengths. Byte tokens inside a UTF-8 character can share
an offset. Trailing partial UTF-8 is replaced at each prompt/output boundary,
matching the returned text; individually decoded byte-token labels can still
be replacement characters. `return_tokens_as_token_ids` labels them
`token_id:N` and keeps identity unambiguous. Scored text must round-trip
through the tokenizer; normalization that changes it is refused, with an
exact-ID prompt as the alternative. Clients handle BPE boundaries and
rolling windows explicitly; no prompt truncation occurs.

The chat route's body limit applies (from the request memory, D-102); a
text prompt's bytes are the body's, an exact-ID prompt is one JSON value a
token, and the model's context bounds both (prompt plus `max_tokens`).
Score rows are therefore at most the context, and with their top scores
the request memory bounds them (D-102; was a fixed 131,072 rows, 5 top
scores and a 64 MiB envelope): one response may be up to the whole pool
(`[client] request_memory_bytes`), each row charged conservatively (2 ×
(256 + 128 a top score) bytes, and six bytes a text byte). Admission
figures the rows' charge before any work and refuses a request past the
pool (400 `score_limit_exceeded`, naming the key), then charges it to the
pool while the request runs (413/503 when it does not fit); a response
that grows beyond its allowance ends with an error. A completed body is
charged by its allocation until its socket buffer drains or is dropped;
one that does not fit beside what others hold gets a 503
`request_memory_busy`, so slow readers cannot retain an unbounded
collection of large score replies. A text prompt's tokenization is
charged too. Many top scores
are selected by a partial sort (O(V log k) a row), so asking for the whole
vocabulary costs a sort of the row, not V². One completed
target row is reduced and discarded at a time, retaining only bounded token
metadata. The initial scorer has decode-like throughput, uses the decode
floor for its watchdog allowance (and a configured deadline) and does not exercise 2,048-row HCA or other
large prefill tiles. Ordinary unscored generation keeps tiled prefill.
Cancellation stops between target steps and retires owed commits/rollback
before the request lease is released. Full recurrent and drafter-injection
state remains intact for generation after a scored prompt. A paused
literal request retains its score rows, decoder offsets and funded
response state across the other model's turn, then resumes its own branch
without repeating rows or generated tokens. Unsupported carried requests
fail explicitly rather than restarting from the prompt. Faster tiled or
jointly batched prompt scoring and streamed literal output are future work.

Validated on Spark on 2026-09-30 against completed native target rows for
short DeepSeek and Qwen prompts, with speculation on and off.

## Limits

- One process, one model resident at a time: M3's full swap. Partial
  eviction, admission and the switching policy come with M5 and M6.
- The chat route is M3's minimal one: no tools, no reasoning controls,
  no credentials (the optional API key is M5's), no CORS, no Responses or
  Messages routes. Qwen and DeepSeek chat and literal completions share up to their request slots
  (four by default; [request slots](#request-slots)); other
  families run one at a time. The front door is
  M5's.
- The tailnet is found at startup; a node whose Tailscale comes up later
  serves it after a restart. `jitllm.service` is ordered after
  `tailscaled.service` (ordering only, no dependency) for that reason.
- A name the Host check does not derive (a LAN DNS name without a reverse
  record) cannot reach the route; a `[client] host_names` key could name
  such names later.
- The image serves one prompt a process, from a latents file (above), and
  only through the commands.
- Turn reuse keeps two stable boundaries on the current branch. A client
  editing a prefix before those boundaries, or returning after their
  expiry, needs a fresh prefill. Independent shared-prefix caching and
  retained forks follow under D-031; prefix matching does not identify a
  conversation.
- Long-context prefill and decode are now nearly flat with depth on the
  fast plans ([long-context phase 2](experiments/long-context/README.md));
  no time limit stops a long prefill (D-102), and the watchdog reports one
  that makes no progress.
- A stream whose client stops reading yields its place after 10 s when it
  keeps others waiting (the serial path, a cohort draining or full), and
  continues from its state once its client reads; until then those
  requests wait behind it. Pausing needs several megabytes of unread
  output (the kernel's socket buffers take that much first). Chat
  rendering runs on the driver: cancellable, charged and bounded, but not
  concurrent with other requests' units.
- A genuine hang is found only after `hang_seconds` without progress
  (streams get keepalives meanwhile, non-streaming requests silence). A
  device that hangs cannot be recovered in-process: the restart loses the
  queue and the conversations then resident (those wholly on disk are
  kept, D-105). Conversations are kept only by the same build; an upgrade
  starts cold.
