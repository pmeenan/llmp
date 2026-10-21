<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Flare assessment

On 2026-10-20 the owner asked whether llmpalooza should adopt anything from
[Flare](https://github.com/NVIDIA/flare-local-inference), NVIDIA's newly
open-sourced local inference server. This page is a source review of commit
[`e7b8c01`](https://github.com/NVIDIA/flare-local-inference/tree/e7b8c016474f3b58d390910a7368f58a792b4744)
(2026-10-20). Nothing was run. Flare publishes no speed numbers, so every
speed figure below is either llmpalooza's own measurement or labelled an
estimate. File references are relative to that commit. The plan items this
review produced are listed [at the end](#what-the-plan-takes).

## What it is

- **Target hardware:** SM120/SM121 GPUs. Its one bandwidth reference is the RTX PRO 6000 (SM120); DGX Spark (SM121) is a supported second target.
- **Workload:** local serving for long conversations, coding agents and repeated tool use.
- **Architecture:**
  - Ignition (Python) prepares a model with PyTorch, Transformers and FlashInfer/b12x into a runtime package of CUDA graphs.
  - Helios (Rust) serves that package, with no Python and no capture at load.
- **Scope:**
  - One model per process; multi-model hosting is explicitly out of scope (`AGENTS.md`).
  - Cross-node tensor parallelism is in scope.
  - Recipes: DeepSeek V4 and V4.1, GLM-5.3, Kimi K3, MiMo V2.6 Pro, Nemotron-H, Qwen3.8 (Flash Next and 27B) and Step-5.
- **License:** Flare's own code, including its Triton kernels, is Apache-2.0 (copyright NVIDIA and Flare contributors, `NOTICE`); ports keep that tag beside ours (AGENTS.md rule 12).
  - b12x is the `b12x` extra of `local-inference-lab/flashinfer@622cec10`, CuTe-DSL code that needs the proprietary `nvidia-cutlass-dsl` (licensing.md). Its techniques are reimplemented, not vendored.

Flare overlaps llmpalooza more than [DeepInfer](deepinfer-assessment.md)
does: same hardware class, the same models, the same agent-client
workload. Its kernels are not ahead of ours. Its value is in:
- engine mechanisms;
- a documented checkpoint format;
- front-door detail;
- two-node transport.

## Generalization

As with DeepInfer, nothing is adopted in Flare's per-model recipe form
(D-107; owner, 2026-10-20). Each item below says how it lands in the shared
engine, selected by capability, tensor role or encoding.

## Changes to items already planned

### 1. NVFP4 scale compression (CSF) for FP4-Local

Our FP4-Local items call NVIDIA's NVFP4 scale compression undocumented
(m35-families.md, plan.md M4). Flare describes a lossless NVFP4 scale
compression, CSF; the normative specification is in its FlashInfer fork
(`docs/csf-scale-blocks.md` there, not in this checkout):
- **Formats:** `csf_32x2_v1`, with tensors `csf32_{bases,codes,page_bases,offsets,exceptions}`, and the slab variant `nvfp4_csf_{fixed,exceptions}` (`docs/tp-checkpoint-layout.md`).
- **Reference decoder:** an independent pure-torch decoder for both (`ignition/reference/qwen38_quant.py`, `decode_csf_scales`, `decode_csf_slab_scales`).
- **MXFP4 containers:** `ignition/preparation/csf_checkpoint.py` validates MXFP4-CSF containers; their decoding runs in b12x on the GPU.
- **Verification:** the codec is lossless, so a byte comparison against that decoder is the whole check.

**Unconfirmed:** that CSF is the format FP4-Local's card calls NVIDIA's
"new lossless compression". Flare's CSF checkpoints are
`local-inference-lab` re-hosts, and nothing in Flare names ModelOpt or
FP4-Local. Confirm first that the FP4-Local tensors carry these names.

**Generic form:** an encoded-scale-plane decoder in the format layer that
writes the native scale-atom layout at import (AGENTS.md rule 11). Flare
also keeps scales compressed in memory and expands them per layer into
shared buffers, planning an estimated 3–6% less resident weight storage
(`docs/model-preparation.md`). That is a later encoding candidate, not
part of the import.

### 2. Device-keyed row delivery (the token-keyed row read-ahead)

**What Flare does:**
- The graph computes n-gram and Engram row IDs on the GPU, draft tokens included. It publishes them to a host-mapped mailbox with a system-scope release (`ignition/components/ngram_delivery.py`).
- A reader thread fetches the rows: disk through io_uring with registered buffers, or RAM (`helios/src/engram/reader.rs`).
- The captured graph waits on an external event only at the consuming layer (`ignition/components/engram.py`, `ignition/recipes/qwen38/ple.py`).

**What llmpalooza's plan lacks:** our planned read-ahead computes keys on the host at intake. That covers prefill and host-known tokens only. It does not cover:
- decode, where keys depend on the token just picked on the GPU (D-106);
- DSpark's verify rows, which depend on drafts made inside the job.

**Generic form:** the row table's key function, declared in its mapping, is
evaluable on the host (intake, cross-chunk read-ahead) or in the graph
(mailbox). The pager's reader polls the mailbox (RE-017), issues direct
reads into the landing (no CPU copy, D-081), and raises a flag. The
existing gather kernel then runs behind an event wait at the first
consuming layer.

Flare's optional row cache (set-associative with frequency admission) is
off by default. Build one only if the measurement shows row reuse, and then
as a reclaimable catalog entry priced at its re-read cost.

### 3. Speculation details (calibration screen, sampled speculation)

From `docs/mtp-design.md` and `helios/src/scheduler/{speculation,split}.rs`:

- **Causal-prefix admission** (`ignition/components/split_draft.py`): admit draft position i using only confidence through i. Looking ahead biases sampled drafts. It becomes required once confidence budgeting and sampled drafts combine.
- **Split proposal and verify:** Flare runs DSpark's draft and verify as separate graphs, so verify width is chosen after confidence is known. Generically: any drafter that exposes confidence. The extra host decision costs one round trip (6–11 µs, D-106).
- **Verify graphs keyed by total rows:** per-request row counts and offsets are device inputs. A small dynamic program packs requests, padding included, into the captured shapes (`split.rs`, `split/packed.rs`; `docs/scheduling-runtime-design.md`). Per-request widths then don't multiply captured shapes.
- **Per-request, per-depth, per-source acceptance profiles:** conditional acceptance per depth and per proposal source, with prior 0.5 and memory 128. Expected tokens at width W is `1 + p1 + p1·p2 + …`. This one estimator can replace llmpalooza's two-depth `AdaptiveDepth` and whole-wave `AdaptiveWaveMode` averages, where it measures no worse, and it is what confidence refines once a proposal exists. Like `AdaptiveDepth`'s state, it travels with conversation state, so restored continuations schedule as uninterrupted ones do. Costs stay the calibrated ones (no clock reads; Flare's are measured).
- **Forced acceptance as a test** (`qualify/speculative_state_parity.py`): a one-token grammar forces every draft to be accepted, which exercises verify and commit state without drafter noise.

### 4. Prefix keys and boundaries (minimal prefix reuse)

From `helios/src/cache/hash.rs` and `docs/core-design.md` §4:

- **Hash chain:** chained 128-bit hashes over 64-token blocks, independent of page size. Keys are domain-separated by model identity.
- **Mid-block endpoints:** an endpoint that does not fall on a block boundary is keyed exactly as parent digest + tail tokens + count.
- **Lookup:** buckets from each block to the endpoints inside it make a lookup cost about one probe per 64 new tokens.
- **Conditioning:** image identities are hashed in at their positions.
- **Shared transfers:** restore tickets let waiters on the same prefix share one transfer.
- **Boundary discovery:** render each message prefix and keep it only if it is a strict token prefix of the full prompt (`docs/core-design.md`, "Exact endpoint capture"). This is renderer-agnostic. It checks that a template renders earlier messages stably within one prompt; it does not show that a system prompt stays byte-stable across conversations, which the minimal item measures separately.

**Generic form:** these are the key and boundary rules of the shared prefix
store over state contracts (D-031). Hash grain stays independent of each
contract's block grain. Keys are domain-separated by D-055's whole identity
(artifact, state-producing plan, non-text inputs, caller scope), not by
model identity alone as in Flare.

### 5. Tool calls and the Messages profile

Flare serves Chat Completions, Completions, Messages with `count_tokens`,
stateless Responses and `/v1/models`, all through one internal
conversation type (`docs/api.md`, `helios/src/server/`).

**Argument encodings differ.** Flare parses tool calls with per-family
parser code chosen by an enum in the package (`helios/proto/runtime_package.proto`,
`ToolParserKind` and `ReasoningParserKind`, with a few parameters such as
wrapping newlines): Qwen-coder `<function=…><parameter=…>` tags (Qwen,
Nemotron, MiMo, Step), GLM key/value argument tags, DeepSeek's DSML and
Kimi's XTML. A begin/end pair per format is not enough.

**Generic form:** each chat format declares, as data:
- the calls-block and per-call begin and end;
- where the name sits;
- the argument encoding (a JSON object, or a tagged-parameter pattern with its value-trimming rule);
- whether the tool opener ends reasoning.

One interpreter consumes that description.

**Edge cases Flare handles** (`docs/api.md`; parser tests), our test suite:
- **Malformed calls:**
  - a missing `</parameter>` is closed by the next boundary;
  - omitted outer tool tags are accepted;
  - one wrapping newline is trimmed at each end, per format;
  - scalars are converted best-effort from the schema when it is not strict.
- **Undeclared names:** in auto mode, calls to undeclared names are returned for the client to handle.
- **Truncation:** a call cut off by the output limit is dropped with a length finish; an incomplete call at natural completion is an error.
- **Parallel calls:**
  - parallel tool results are reordered into call order before rendering;
  - `parallel_tool_calls: false` fails the response when a second call starts.
- **Lenient input:** calls without IDs and empty assistant content are tolerated.

**Streaming:** each call is emitted whole once it validates, which needs no
incremental argument parser. The planned bound-token forcing improves on
Flare's prompt-and-validate handling of `tool_choice: required`.

**Forged markers:** not answered by Flare. Its `rejected_tokens`
(`helios/src/model/prompt.rs`) refuses a rendered prompt containing IDs
outside the package's frontend contract, and the recipes list only media
placeholders a package cannot serve (DeepSeek V4.1's image token 129264,
GLM's image and video tokens in a text-only package). Refusing tool-call
marker IDs that way would refuse every rendered conversation with an
earlier tool call, so the M3.5 item's forged-marker question stays open.

**Messages and Claude Code:**
- **Acceptance test** (`tests/server/test_coding_clients.py`):
  - the client runs under isolated settings (`--setting-sources ""`, an empty strict MCP config, a fixed tool list), in normal and `--bare` modes;
  - it must fix a seeded bug in a temporary git repo;
  - checks: the test passes, the test file is untouched, and a tool call ran.
- **`context_management`:** Flare accepts `clear_thinking_20251015` with `keep: "all"`. If a captured Claude Code request shows that is all it sends, the pinned profile may not need to disable experimental betas.
- **Effort:** where a template takes effort levels, Flare finds them by trial-rendering and maps `output_config.effort` (and Anthropic thinking budgets) onto them instead of rejecting.
- **System messages:** a mid-conversation system message the template would drop becomes a delimited user turn, decided by trial render.
- **Not adopted:** Flare's `ANTHROPIC_API_KEY` and model-name-agnostic setup. llmpalooza pins the Haiku variable or aliases background requests to the main model (client-api-baseline.md).

**Codex (M5):** `docs/clients/codex.toml` and `codex-models.json` are a
working Codex profile: freeform `apply_patch`, `unified_exec`, parallel
tool calls, context window and truncation policy.

### 6. Two-node transport and the EP-vs-TP decision (M4 entry)

**Collectives:** Flare does not use NCCL. RoCEnante is a one-shot RDMA collective.
- The captured kernel stages into a registered, pinned host region and rings a doorbell. A CPU proxy (`helios/native/roce_proxy.c`, libibverbs) posts RDMA writes striped over two rails, each stripe followed by a sequence write.
- The receiving GPU spins on the flags and reduces locally.
- Above about 1–2 MiB it switches to a pipelined two-shot.
- There is no GPUDirect, the same path NCCL takes on the Spark (environment.md).

**Latency:**
- llmpalooza measured NCCL's all-reduce at a median of 30.6 µs at 32 B (uncaptured, nccl-tests), against host-buffer RDMA write latencies (perftest RTT/2) of 2.34 µs at 1 KiB and 8.4 µs at 64 KiB (`experiments/interconnect/README.md`).
- A one-shot design should land near the wire time; that is an estimate.
- At that latency the per-layer collective count stops deciding EP against TP. Bytes read per node, routing imbalance and paging granularity decide it instead:
  - EP keeps whole experts as paged dependency groups;
  - TP splits every expert.

**Sharding:**
- Flare is plain Megatron TP: attention by heads with KV heads replicated when ranks outnumber them, and experts by intermediate width aligned to 32.
- V4.1 runs at TP4 only (four Sparks), with attention by query heads and the single latent KV head replicated.
- Engram is split by hash head: each rank holds and reads only its heads' rows, computes a partial projection and joins one all-reduce (`ignition/recipes/deepseek41/engram.py`). No rows cross nodes.
- GLM decode context parallelism shards KV by token and combines by log-sum-exp; that is a later option.

**Failure handling:**
- **Rendezvous:** rank 0 admits each rank once per export ID. Capacity is agreed as the minimum across ranks. QP records are exchanged and every connection confirmed.
- **Bounded waits:** a timed-out GPU wait writes an error word, so a dead peer becomes an error, not a wedged GPU. Here that is D-102's hang detection, bounded well above any healthy wait.
- **Health gating:** transport health is checked before completions are published.
- **Stop:** rank 0 stops first.

**Generic forms:**
- a collective provider with NCCL and host-staged one-shot RDMA implementations, chosen per op and size by quick A/B (D-053);
- one transport per node, registered once at a stable address and bound by every model's graphs, so swaps never pay for NCCL init or queue-pair setup;
- an intermediate-column split option in the MoE component;
- a hash-head split option for row tables;
- a vocabulary-parallel greedy pick;
- bounded waits and identity/capacity agreement in coordinated readiness.

Porting `roce_proxy.c` needs its upstream license confirmed first (it has
no SPDX header and comes from the FlashInfer fork). The GPU side is b12x's
and is rewritten. libibverbs (rdma-core) is taken under its BSD-2 option.

## New techniques

### 7. History-copy drafts

Flare proposes continuations by suffix-matching the request's own history:
8- or 4-grams within the last 4,096 tokens (`helios/src/scheduler/speculation.rs`).
It is off by default and wired only into the Qwen recipe (`--history-drafts`,
`docs/mtp-design.md`), so Flare offers no measurement of its gain.
- **Merging:** proposals are merged with the learned drafter under one verify budget, with acceptance tracked per source.
- **Exactness:** a copy draft's q is a point mass, so it fits llmpalooza's current verify rule exactly, and stays exact after sampled speculation.
- **Workload:** code edits and repeated tool output.
- **Reach:** it gives models with no drafter some speculation, where their verify shapes and rollback contracts exist.

**Generic form:** a proposal source in the shared speculation scheduler.
Measure first on agent transcripts.

### 8. Overlapping one request's independent branches

Flare runs these on an auxiliary stream inside the graph, each in its row band:
- the shared expert beside routing and the routed experts (up to 256 rows);
- the indexer/selector beside QKV, norm, RoPE and KV write;
- DeepSeek's compressor beside query preparation;
- Gated DeltaNet input projections.

Sources: `ignition/components/{expert_overlap,projection_overlap,compressor_overlap}.py`; rules in `docs/performance-work.md`.

**llmpalooza today:** our lanes run per request (T15). The paired
shared-expert GLU operator (T79, Qwen MXFP8) won 28.6% at the operator and
lost 0.09% in the model, so this needs measurement. Decode-sized work underfills the GB10's
48 SMs, which is why it may pay.

**Generic form:** a planner pass over the graph IR that assigns independent
subgraphs to lanes with join edges, enabled per row band by measured gain,
with scratch kept disjoint by lifetime.

### 9. Checkpoints off the turn path

**What Flare does:**
- A checkpoint is complete KV pages plus one dense state row holding all mutable state.
- It is captured by a device-to-device graph into a resident slot and published after a fence.
- Host offload, opt-in and off by default until its own latency gates pass, writes behind on the lowest-priority stream, and restores are pipelined through two staging windows in flight (`docs/prefix-cache-offload.md`, `helios/src/offload/`).

**llmpalooza today:** we capture and restore synchronously on the turn path,
at queue depth 1, through one 2 MiB staging buffer
(`src/engine/checkpoint_file.cc`). We also capture every 2 MiB extent that
contains a mutable byte.

**Generic form:**
- capture by device copy into a cataloged, charged snapshot extent, published after a fence, written behind at queue depth above 1;
- restore page N+1 while copying page N;
- capture only the contract's mutable byte ranges. Measure mutable against captured bytes on Qwen3.8 and DeepSeek first.

This is the same machinery the minimal prefix reuse copies with.

### 10. Poison-on-remap diagnostic

Flare's GLM batched-speculation defect (`docs/glm-batched-speculation-defect.md`):
a value written into scratch once at preparation was later zeroed, and only
the tiled route above 16 rows read it. llmpalooza hit the same class once
(RE-058).

Our shared pageable workspace is re-backed across swaps, and retained
graphs replay against it. Re-backed pages often still hold old values, so
such a bug hides. Only the fake provider poisons memory today.

**Proposed:**
- A diagnostic CUDA-provider mode that fills discardable extents with a poison pattern on every (re)map, run under the swap, wave and verify model tests.
- Delay-injection stress for concurrent lanes (T15), since Flare's still-open case is a timing-sensitive NaN, cause unknown, in layers whose projections run on an auxiliary stream.

### 11. Smaller items

- **Sampling defaults:** artifacts with no generation config get model-card defaults recorded at import. Qwen3.8 NVFP4 has none today, and agent clients don't send them. Keep greedy at temperature 0; never clamp, unlike Flare.
- **Faster integrity hashing:** XXH3 (BSD-2) instead of SHA-256 for spill and checkpoint integrity. The threat is torn or corrupt files in owner-only directories, not an adversary. Artifacts keep SHA-256 (D-054).
- **Small I/O improvements:** registered io_uring buffers on the landing zone, and bounded batched copies on a low-priority stream for spills.
- **Tripwire metrics** (`qualify/text_oracle.py`, `qualify/server.py`):
  - full-vocabulary KL, top-1 margin and top-10 overlap, with a three-way noise floor (reference against canonical) as the measured basis for a bound;
  - prefix-reuse checks that predict how many prefill launches a hit should leave and compare against counters.
- **FP8 E4M3 KV:** a state encoding to qualify as a per-alias mode under D-108. Flare uses it by default, as a quality trade.

## As a comparator

Flare publishes no speed numbers. Its recipes hash-pin their checkpoints,
so the owner chose (2026-10-20) to archive the pinned revisions that fit
our hosts and run each on both engines. That gives a same-checkpoint
comparison, and llmpalooza's import of them tests the format layer's
flexibility (plan.md, M3.5's comparator item). Where Flare runs on our
hosts:

| Model | Flare's checkpoint | Hosts | Use |
| --- | --- | --- | --- |
| Qwen3.8 Flash Next | `local-inference-lab/…-NVFP4-MXFP8-CSF-QAD@dc5b8ab2` only: the recipe pins its config, index and header hashes (`ignition/preparation/qwen38.py`), so it does not take our Mia pin. Check whether it is the FP4-Local weights (QAD, n-gram table, vision, as that card) | One Spark (TP1) | M3.5 comparator, same-checkpoint for FP4-Local if confirmed |
| GLM-5.3 Flash | `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD@dec48abd` (check whether it is the FP4-Local weights) | Two Sparks (TP2, RoCE) | M4 comparator for FP4-Local |
| DeepSeek V4 Flash | `local-inference-lab/DeepSeek-V4-Flash-Vision-Exp-lossless-CSF@45fd9976`, TP2/TP4 | Two Sparks | Same-checkpoint comparator once llmpalooza serves it (M4); against our 0731 GGUF, speed and memory only |
| Qwen3.8-27B | `nvidia/Qwen3.8-27B-NVFP4@482ca0f3` | One Spark | M3.5 comparator |
| Nemotron 3.5 Bolt 74B-A9B | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB@c6cc2e8f` | One Spark | M3.5 comparator, with Nemotron-H's components; speculative runs also pin its LiLiCorr drafter `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr@abd47806` (`ignition/recipes/nemotron_h/draft_config.py`) |
| DeepSeek V4.1, Kimi K3, MiMo V2.6 Pro, Step-5 | TP4 or more | No | Not comparators |

Flare's default quality trades, which any comparison must state or match:
- FP8 E4M3 KV with unit descales, although the checkpoint declares unquantized KV;
- V4.1's bounded decoder SWA replay in prefill, an approximation from DeepSeek's report that its production prefill also runs ([DeepInfer](deepinfer-assessment.md#3-row-liveness-pruning-exact-yoco-truncation)), always on;
- 4-bit prefill activations on GLM;
- Step-5's load-time conversions.

Running Flare needs a PyTorch/FlashInfer preparation environment on a
Spark, and preparation is long (the ten-minute rule applies).

## Owner decision: aggregate completion over cadence

Flare chooses prefill size by value per time, with a waiting cost for
active decoders, so a smaller chunk can win while others generate. That
trades prefill throughput for decode cadence.

The owner decided against it (2026-10-20). Many workloads are expected to be
prefill-bound, and all the work must finish, so concurrent streams are
scheduled for aggregate completion:
- **Mixed waves:** generating streams' rows ride along inside prefill chunks. That is pure utilization, so it is adopted.
- **Chunk size:** chunks stay the fastest measured option in aggregate. A smaller chunk is chosen only if it measurably shortens total completion, never for cadence.
- **Starvation bound:** default 60 s of calibrated unit time (a judgment default, not a measurement), configurable. It keeps every admitted request on the resident model moving so no client times out, by reordering only: it never refuses or cancels, and keepalives continue (AGENTS.md rule 11, D-102 as amended; plan.md's M3.5 throughput-first scheduling item). Request classes may order work within it, not past it.

## Not adopted

| Item | Why |
| --- | --- |
| Python/torch.compile preparation, capturing every graph at startup, fixed-address serialized graphs, one static memory plan per node, snapshots aliased into KV planes | One model per process; they conflict with fast swaps and several models sharing one address space. llmpalooza's native import and lazy catalog already cover the need |
| A host-RAM cache tier | Same physical pool on Spark; Flare rejects it there too |
| Online timing averages and funded re-measurement; the full fairness-price utility | Against llmpalooza's no-clock-reads rule; calibration plays that role. Revisit priority weights with M6's request classes (D-042) |
| Flare's half-prefill fetch rule | llmpalooza's reclaim order by measured restore cost already covers it |
| Clamping temperature to the model card; ignoring the request's model string | Rules out greedy decoding; breaks aliases |
| Flare's context-exceeded wording | llmpalooza's planned prompt-too-long 400 is what triggers Claude Code's compaction |
| Vendoring b12x | Needs the proprietary CUTLASS DSL; reimplement the techniques |
| Triton kernels as such | No interpreter at runtime; port to CUDA C++ where measured useful |

## What the plan takes

| Item | Shared home | Plan placement |
| --- | --- | --- |
| 1. CSF scale decoder | Format layer, importer | M3.5's Qwen3.8 FP4-Local; reused by M4's GLM FP4-Local |
| 2. Device-keyed row delivery | Pager, row-table capability | M3.5's row read-ahead item; M4's Engram |
| 3. Speculation details | Speculation scheduler | M3.5's calibration screen and sampled-speculation items |
| 4. Prefix keys and boundaries | Prefix store over state contracts | M3.5's minimal prefix reuse; M6's block sharing |
| 5. Tool-format data, edge cases, client acceptance | Chat-format data, front door | M3.5's tool-call and Messages items; M5's Codex profile |
| 6. Transport, splits, readiness | Collective provider, partitioning | Before M4's entry (latency, splits), M4's sharded execution and readiness |
| 7. History-copy drafts | Speculation scheduler, proposal source | M3.5, measure first |
| 8. Branch overlap within a request | Planner pass, lanes | M9's optimization pass, with a screen eligible earlier |
| 9. Checkpoints off the turn path | Checkpoint store | M3.5, measure first |
| 10. Poison-on-remap, lane stress | CUDA provider diagnostics | M3.5's quality tripwires |
| Aggregate completion over cadence | Cohort scheduler, mixed waves | M3.5's throughput-first scheduling item |
| 11. Smaller items | As listed | M3.5 (sampling defaults, XXH3, tripwire metrics); FP8 KV under D-108's per-alias modes |
| Flare as a comparator | Reference comparisons | M3.5 (Qwen3.8), M4 (GLM FP4-Local) |
