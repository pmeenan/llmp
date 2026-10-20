<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepInfer assessment

On 2026-10-20 the owner asked whether llmpalooza should adopt anything from
[DeepInfer](https://github.com/deepseek-ai/DeepInfer), DeepSeek's newly
open-sourced production inference engine. This page is a source review of commit
[`774ce24`](https://github.com/deepseek-ai/DeepInfer/tree/774ce244b87cc0eb8d4b94ab9f5de2ab22049f01)
(2026-10-20); its submodules (FlashMLA, DeepGEMM, DeepEP, DeepSelect) were not
read. Nothing was run, so every DeepInfer number below is DeepSeek-reported,
and every llmpalooza speed figure is an estimate to measure, not a result. File
references are relative to that commit. The plan items this review produced
are linked [at the end](#what-the-plan-takes).

## What it is

- **Scope.** DeepInfer is offline inference for `deepseek-ai/DeepSeek-V4.1-Flash-1017` only, the
  checkpoint M4 targets (plan.md, M4). It was built on vLLM. The engine core
  (scheduler, block manager, input preparation) is C++ under `csrc/engine/`;
  the model, worker and kernels are Python with TileLang ops
  (`deepinfer/model_executor/layers/tilelang_ops/`).
- **Not released:** the KV cache store (3FS), the load balancers, and
  disaggregated serving. `doc/inference_deployment_qa.md` describes them.
- **Hardware:** NVIDIA SM100 and Huawei Ascend 950. FlashMLA, DeepGEMM and
  DeepEP target SM90/SM100, so DeepInfer cannot run on a GB10 (`sm_121`) or
  on the workstation's `sm_86`.
- **License:** Apache-2.0, so it may be used in the core. A file that ports
  its code keeps its tag beside ours (AGENTS.md rule 12).
- **Workload:** datacenter scale. Prefill uses 2–8 ranks of expert and data
  parallelism and decode uses 64–128, with prefill and decode disaggregated
  and KV cache retained for 72 h or more on SSD clusters. Most of the engine
  serves that shape and is irrelevant to one user on one or two Sparks
  ([not adopted](#not-adopted)).

Its value to llmpalooza is threefold:

- the readable reference implementation of V4.1 Flash;
- a source of model-level ideas;
- a few scheduling and quality techniques that transfer.

## Generalization

DeepInfer is one model's engine, so most of its techniques are written for
V4.1. llmpalooza adopts none of them that way (D-107; owner, 2026-10-20).
Each item below says where it lands in the shared engine:

- as a parameter or capability of a shared component, planner pass, pager
  feature or the shared speculation and sampling path;
- selected by capability, tensor role or encoding, never by architecture
  name;
- naming the current families that use it besides DeepSeek.

A technique that only one model can use still lands as a capability its
mapping declares, so the next model with the same shape gets it from data.

## V4.1 Flash as DeepInfer implements it

These details come from DeepInfer's code together with the config of M4's
archived quant (`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw@64ba41b`). Each
row is a V4.1 shape that llmpalooza's DeepSeek V4 mapping (`src/spec/mappings/dsv4*`,
`src/components/{mqa,hc,moe}.cc`) does not have yet. The last column is the
generic form it takes in llmpalooza.

| V4.1 shape | DeepInfer | llmpalooza today | Lands as |
| --- | --- | --- | --- |
| **Compression ratios 0/1/2** per layer (layers 0–1 window only, 2–19 C2, 20–39 C1). No position embedding (APE) and no overlap; C1 passes the latent through without the gate | `models/v4_modules/compress.py`; refuses ratios above 2 and APE (`model_metadata_utils.py`, `mla_attn.py`) | C4 with overlap and C128, both with APE | Compressor parameters: ratio (including 1 and none), position embedding and overlap |
| **Indexer keys from the main compressed KV**, through a small projection and norm with no Hadamard. 32 index heads, top-512 | `v4_modules/index_attn.py`; `tilelang_ops/fused_transform.py` | Its own indexer compressor with a Hadamard; the score kernel is fixed at 64 heads | Indexer parameters: key source, head count, transform; the score kernel takes the head count |
| **Cross-layer sharing.** KV sources are layers 2/8/14/20, so layers 20–39 all reuse layer 20's KV. Index sources are 2/8/14/20/24/28/32/36; other layers reuse their source's top-k | `v4_modules/cross_layer.py` | No sharing between layers of one stage | Cross-layer KV and top-k sharing as graph edges any mapping may declare (read-only borrows within a stage) |
| **Hierarchical candidate blocks.** Layer 20 scores blocks of 8 by their maximum and keeps 2,048; later index sources score only those candidates | `index_attn.py`; `tilelang_ops/candidate_blocks.py` | None | An optional indexer stage (block-max scoring, block top-k, candidate scoring) |
| **YOCO.** `n_prefill_layer` is the first C1 layer (20). Layers 20–39 are consumers whose compressed KV comes from layer 20's attention-normed hidden state. Their sliding-window KV comes from each layer's own hidden state: `yoco_swa_share_hidden` defaults to false and the config does not set it | `v4_modules/yoco.py`, `attn.py`, `hugging_face/config.py` | None | A mapping of shared KV sources; its prefill gain comes from [row-liveness pruning](#3-row-liveness-pruning-exact-yoco-truncation) |
| **Shifted hyper-connections.** Each sub-layer applies the previous one's pre-mix; there is no HC head, and the last mix is applied at the model's end | `v4_modules/hc.py`, `deepseek.py` | Sinkhorn HC with a head (`hc.cc`) | A variant of the shared HC component |
| **Engram** at layers 1 and 14. n-gram orders 2–4 over a 99,092-entry compressed vocabulary, 8 hashed heads per order, rows 256 B wide in FP8, about 384M rows per layer (about 190 GiB in all with their per-32 scales). Hashes depend only on token IDs, so DeepInfer gathers every Engram layer's rows before layer 0. Tables stay in host memory by default | `v4_modules/engram*.py`, `deepseek.py`; the hash and gate kernels are in TileKernels, not in this repository | None (Qwen's per-layer embedding (PLE) n-gram rows are a different shape) | A token-keyed row-table component with [read-ahead](#4-token-keyed-row-read-ahead), shared with Qwen3.8's n-gram rows |
| **MoE.** 384 experts, top-6, sqrt-softplus router, no hash-routed layers, and a separate routing bias for image tokens | `v4_modules/moe.py`, `mlp.py` | Hash routing and the router exist; no image routing bias | A per-modality routing bias in the shared router |
| **DSpark stored in the checkpoint** as `mtp.{0..2}`. Block of 5, taps target layers 37–39, Markov rank 256, a drafter MoE of 128 experts top-3, and the confidence head always present | `models/mtp/dflash.py`, `hugging_face/names.py`, `config.py` | DSpark from a separate GGUF; its `conf_proj` is imported but unbound (`dsv4.cc`) | The existing drafter mapping over `mtp.*`; the head feeds [confidence-budgeted verification](#1-confidence-budgeted-verification) |

**State size.** With the KV shared across layers 20–39 and the
compression, V4.1's whole compressed and index state at 1M tokens is a few
GiB at F16. DeepInfer's FP8/FP4 KV formats serve datacenter batch sizes,
not one user's capacity.

## Techniques worth adopting

### 1. Confidence-budgeted verification

**What DeepInfer does:**
- Each DSpark row has a confidence head:
  `sigmoid([h_prenorm, markov_w1(prev)]·w / T)` (`models/mtp/dflash.py`, `ConfidenceHead`).
- The cumulative product over rows 1..j predicts that all of them are accepted.
- `MTPVarlen::schedule` (`csrc/engine/mtp_varlen.hpp`; opt-in through `DEEPINFER_ENABLE_MTP_VARLEN`, which DeepSeek's eval script sets) sorts every (sequence, row) pair in the batch by that cumulative confidence. It then grows the total number of verify rows while (sequences + Σ confidence) / predicted step time keeps improving. So each request verifies 0 to 5 drafts per step, and a doubtful batch falls back to plain decode.
- The predicted step time comes from a cost curve measured at startup, corrected online by a fitted per-row slope and a median sampler time.
- After each verify DeepInfer computes the mean |accept probability − confidence| (`avg_confidence_error`), an online check of calibration.

**llmpalooza today:**
- DSpark always verifies 3 drafts (`default_draft_rows`).
- Under concurrency, `adaptive_wave_mode.h` switches a whole wave between speculative and plain, using calibrated wave costs.
- llama.cpp never reads the head, so its calibration on the 0731 GGUF is unknown.

**Caution from our own history:**
- Confidence control for Qwen3.8's MTP was neutral (optimization-inventory.md, T45, T81 and "Rejected kernels: pieces worth retaining"): a confidence window that only shortened the verify, and incremental stopping that also skipped later draft passes, both at 128K.
- DSpark differs: it drafts its whole block in one pass, so the saving is on the verify side, where each extra verify row brings in up to six more routed experts per MoE layer. Skipping a doubtful row therefore saves weight bandwidth.
- Confident steps may also verify beyond 3 drafts, up to `dflash.block_size`.

**Generic form:** a drafter capability in the shared speculation scheduler. Any drafter that reports a per-row confidence gets verify rows per request per step from it, and every request in a joined wave gets its own row budget.
- DSpark's head is the first source: V4 now, V4.1 from its first run.
- A drafter without a trained head may supply its draft probability, but only where a measurement shows that pays; Qwen3.8's MTP did not.

**Proposed approach:**
- **Screen first:** bind the head and compare predicted with realised acceptance.
- **If it is calibrated:** choose rows per request, costed by the calibrated wave and depth costs (llmpalooza reads no clock to make plan choices).
- **Quality:** greedy output is unchanged.

### 2. Speculative sampling with the drafter's distribution

**What DeepInfer does:**
- Drafts are sampled at the request's own temperature, top-p and seed. The drafter's full-vocabulary log-probabilities q are kept per draft row (`layers/mtp_info_pool.py`).
- A draft is accepted when min(1, p/q) > U; on rejection the token is resampled from (p − q′)+, with q′ = q·e^−5e-5 so the residual is never empty (`DRAFT_PROB_PENALTY`, `common.py`).
- Uniform draws are keyed by (seed, position). It all runs on the GPU in one fused kernel (`tilelang_ops/fused_sample.py`, `common.py`).

**llmpalooza today:**
- `VerifyDraft` (`src/execution/sampling.h`) treats the drafter as greedy: q is a point mass, so a draft is accepted with probability p(draft).
- That is exact, but at temperature 1 (DeepSeek's default) it is expected to accept fewer drafts than sampling from q, whose acceptance is 1 − TV(p, q); neither acceptance has been measured, but the cost shows: speculative decode at temperature 1 fell to 27.1 tok/s against greedy's 29.8 for DeepSeek, and to 32.2 against 43.5 for Qwen3.8 (tokenizer.md, sampling).
- Sampled requests also copy each verify's logits to the host.

**Proposed approach:** sample drafts from q, keep q per draft row on the device (about 0.5 MB per row at a 129K vocabulary), and move acceptance onto the GPU. The result stays exact in distribution.

**Generic form:** one verify rule in the shared sampling and verify path, for every drafter (DSpark, Qwen3.8's MTP and every later drafter). A drafter exposes its distribution as an output. A greedy drafter becomes the special case where q is a point mass.

### 3. Row-liveness pruning (exact YOCO truncation)

**What DeepInfer does:**
- Its "compact" mode (`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE`, `environ.py`, `input_metadata/yoco.py`) runs the consumer layers 20–39 only on the last `KEEP_TOKENS` (140) prompt rows.
- It also narrows their sliding window to those rows. Because consumer windows come from each layer's own hidden state, this is **approximate**.
- It is off by default in this release, though the deployment Q&A (`doc/inference_deployment_qa.md`, "Decoder SWA bounded replay") says production prefill runs it. `examples/yoco/` checks only that the mask and compact modes agree with each other.

**The exact version:**
- The only cross-row dependency inside the consumer layers is the 128-row sliding window. Compressed and index KV come from layer 20, for every row.
- The last consumer layer needs the final block plus backtracking (about 134 rows, which also covers the drafter's taps at layers 37–39). Each earlier consumer layer needs about 127 rows more.
- So the cone is about 134 + 19 × 127 ≈ 2.5K rows at layer 20, narrowing to about 134 at layer 39. The Q&A gives the same figure (exact reconstruction "would require running the last 2,560 tokens through 20 decoder layers").
- Layer 20's compressed and index KV still take its attention-normed hidden for every row; only the consumer layers' query, window, MoE and HC work narrows.
- Running only that cone is exact: every kept row computes what full prefill computes (bits may differ only where a smaller row count selects a different kernel, as a chunk size does). On prompts much longer than 2.5K, it skips almost all of half the model's layers, including their MoE. Estimate: up to about 2× prefill; the Q&A reports its approximate version "nearly halving" prefill.
- It extends the state-only nonfinal prompt dependency cut (T54) and requested-row narrowing (T36) from the head to whole layers.

**Generic form:** a planner pass over the graph IR, not a V4.1 mode.
- From the graph's own dependencies, it computes per layer which prompt rows any later consumer needs. The inputs are:
  - which layers' KV each layer reads (shared KV sources included);
  - windowed attention's reach;
  - drafter taps, over the drafter's own attention reach;
  - requested outputs and scores (prompt log-probabilities keep every row);
  - state snapshots requested inside the prompt (prefix-sharing boundaries, checkpoints).
- It executes only those rows.
- V4.1's consumer layers are the large case. A model's trailing windowed layers after its last global-attention layer (in Gemma, for example) also qualify, and are screened when the pass lands.
- Full-attention and recurrent layers whose state every row feeds keep all their rows, so the pass is exact by construction.

**Approximate mode (owner, 2026-10-20):**
- The same pass takes an opt-in keep count that replaces the exact cone, so windowed layers past the cut see only the kept rows. This is DeepInfer's truncation in generic form, available to any model the exact pass covers.
- It trades quality for speed, so it is a per-alias mode, off by default (D-085, AGENTS.md rule 11).
- It is qualified per model: long-context quality against the exact pass, and measured speed.
- It is offered only where it is measurably faster than the exact cone. On prompts much longer than the cone, the exact pass already captures nearly all of the gain.

### 4. Token-keyed row read-ahead

Because Engram hashes depend only on token IDs, a prompt's rows are known
at intake:
- hash on the host and issue row reads before layer 0, as DeepInfer gathers them before layer 0 (`deepseek.py`);
- overlap layer 14's reads with layers 0–13 (layer 1's can overlap only layer 0, or the previous chunk);
- under TP2, shard the tables by row across both nodes' SSDs (D-035).

In decode the newest token is part of every n-gram, so a step's rows are
known only once the previous step's token is; there the reads overlap the
layers before the consuming one and nothing more.

Per token this is 48 rows of 256 B across the two layers. The importer builds
the compressed-vocabulary table; the hash and gate formulas need
TileKernels' source or DeepSeek's reference code.

**Generic form:** a pager capability for any row table whose indices depend
only on token IDs. The mapping declares the table's key function, and the
pager:
- computes indices at intake;
- reads rows ahead of the first layer that consumes them, overlapping earlier layers;
- under TP, shards rows across the nodes' SSDs.

Qwen3.8's n-gram (PLE) table is already read by rows from the SSD
(`src/engine/ple_rows.h`, D-035), but as Qwen3.8's own path: hashed before
each chunk's job and read synchronously before it runs. It is the first
consumer of the generic form and may take it before M4; with its table at
layer 1, its gain is reading the next chunk's rows while the current chunk
runs. Engram's hash, and per-layer embedding tables, are further key
functions, declared as data where the mapping language can express them.

### 5. FP4 indexer scoring

**What DeepInfer does:** it scores the indexer with Q and K in e2m1, a UE8M0 scale per 32 values, and BF16 logits. It has no other mode (`index_attn.py`, `tilelang_ops/cache_common.py`).

**llmpalooza today:** we score in F16 with F32 accumulation (`src/kernels/ggml/dsv4_sparse.cu`). The inventory keeps ds4's FP4 indexer as a different precision contract that needs same-quality evidence (T82).

**What's new for V4.1:**
- These are DeepSeek's own production numerics: DeepInfer has no other indexer precision.
- GB10's block-scaled `mma.sync` has a much higher FP4 than F16 rate on paper; we have not measured it on this kernel shape.

**Estimate:** at 1M context V4.1 scans about 2.6M keys per token, about 2.4 ms at F16 against about 0.6 ms at FP4.

**Proposed approach:** qualify it against F16 on V4.1. It becomes the default only at the same quality, otherwise a per-alias mode (D-085).

**Generic form:** an operand encoding of the shared sparse-scoring operation, selectable for every indexer that uses it: V4, V4.1, GLM-5.3's DSA, and any later one. Qualification is per model, because quality is.

### 6. Expert parallelism with replicated attention

**What DeepInfer does:** it replicates attention and splits experts across ranks.

**Why it may suit V4.1 on two Sparks:**
- V4.1 has one KV head, and its compressor and indexer state is small. Replicating attention costs about 2 GB of weights per node.
- The expert layer then needs one all-reduce, against TP2's two collectives per layer.

**Fit with our design:** M3.6's planner already places experts whole with one all-reduce (`moe.cc`'s split option; plan.md lists "local experts plus a mask plus one all-reduce" among M3.6's partitioning limits). Two limits stand in the way: a TP stage never mixes split and replicated layers (plan.md, M4's concurrent-models item), and DeepSeek's attention (`mqa.cc`) declares no split option, so neither EP-with-replicated-attention nor TP2 of V4.1's attention is expressible today.

**Proposed approach:** lift the mixing limit and give the attention component its split options (query heads with the single KV head replicated), then measure EP against TP2 before committing to TP2.

**Generic form:** a choice the partitioning planner costs for every MoE it places across devices, not a V4.1 setting. GLM-5.3 is measured as well.

### 7. Shared-prefix cache, generic over state contracts

**What DeepInfer does** (`csrc/engine/block_manager.hpp`, `scheduler.hpp`, `worker/cache_engine.py`, `tilelang_ops/state_cache.py`):
- **Blocks:** compressed KV and indexer keys live in reference-counted paged blocks, keyed by a token hash chain at a 128-token grain.
- **Window state:** each sequence has one state slot (window ring plus compressor accumulators). The window is snapshotted at 128-token boundaries.
- **Lookup:** the blocks decide the hit length; a missing window snapshot backs off one block.
- **Forks:** a fork copies only the partly filled tail block and clones the slot.
- **Boundaries:** callers can force a prefill chunk to end at chosen positions (`chunk_checkpoints`), so a system prefix can be snapshotted mid-prompt.

**llmpalooza today:**
- Exact turn checkpoints on one branch only, with no sharing across conversations (runtime-serving.md).
- V4's ratio-4 layers use overlap, so a block boundary also carries the previous block's projections. That carry is small but not zero as it is in V4.1.

**Value:**
- An agent and its subagents share long system and tool prefixes. At DeepSeek V4's measured ~460 tok/s prefill (runtime-serving.md, chunk table), every 1K tokens of shared prefix saves about 2 s per new conversation.
- Keep llmpalooza's exact-position checkpoints for continuations. Rounding them down to 128 would re-prefill up to 127 tokens on every restore.

**Generic form:** sharing works from each component's state contract (D-031), not from DeepSeek's layout.
- State a contract marks immutable once written is shared in blocks at the contract's grain: full KV, compressed KV, indexer keys.
- Mutable state is snapshotted per boundary: windows, recurrent and linear-attention state, compressor carries.
- Every family gets prefix sharing from its existing contract: Gemma's windowed layers, Qwen3.8's Gated DeltaNet and QSA, DeepSeek's compressed attention.

### 8. Tool calls and structured output

DeepInfer's structured output (`deepinfer/structure_output/`,
`tilelang_ops/structure_output.py`) uses xgrammar (Apache-2.0) for JSON
schema, regex and EBNF.

**The design to copy:**
- **Asynchronous compile:** grammars compile in a thread pool, and a request is scheduled only once its matcher is ready. A compile timeout or error fails only that request.
- **Bounded cache:** the compile cache is limited by bytes, after CVE-2025-32381.
- **Trigger tokens:** constraint starts only after a trigger token sequence, so reasoning and free text stay unconstrained. Once an optional bound token (such as the end of reasoning) appears, the rest of the trigger is forced, which is how `tool_choice: required` or a named function can work (`structure_output/matcher.py`).
- **Mask path:** the CPU fills the token mask while the GPU computes logits.
- **Speculation:** the matcher tentatively accepts each draft and fills one mask per draft step, then rolls back.

**Generic form:** one model-independent engine.
- Trigger, begin and end sequences come from each chat format's data (renderer and tokenizer), never per-model parser code.
- The mask applies in the shared sampler and verify path.

**Fit with llmpalooza:**
- We sample on the host, so applying a mask is cheap; rows with a grammar leave GPU greedy picking.
- xgrammar must be checked against the no-exceptions build (D-066), in an isolated unit if needed.
- Today the chat route refuses any non-empty `tools` (`src/runtime/api.cc`), so every named agent client fails as soon as it sends tools.

### 9. Reference-free quality tripwires

- **Decode against prefill.** `tests/models/test_kl.py` decodes about 1K tokens (speculation included) after 16K-token image and text prompts, rescores the whole sequence by one prefill, and requires KL = 0 in batch-invariant mode. With a declared bound instead of 0, this catches decode/prefill kernel divergence and KV-encoding, paging-restore and chunk-boundary bugs without a reference engine.
- **Sampler uniformity.** `tests/models/test_sampler.py` runs a PIT and KS test on live rollouts: Σ smaller probs + U·p(token) must be uniform. It is cheaper and stronger than our total-variation check over 256 seeds, and it is the natural method for M9's speculative-sampling distribution criterion.
- **Sync guard.** `forbid_cuda_sync()` wraps the model step (`worker/model_runner.py`, `tests/test_cuda_sync_guards.py`). Our equivalent is a debug-build counter in the CUDA provider that rejects blocking driver calls inside a step's submit region.
- **Generic form:** shared tooling that every served family runs unchanged.

### 10. Fusion patterns (M9 leads)

Each is a shared pattern that the planner selects by capability, for every attention kind or writer whose chain matches, never inside one model's mapping.

- **Decode attention epilogue.**
  - DeepInfer runs inverse RoPE and quantization in the attention epilogue, feeding output-A (FlashMLA's fused path, `attention_backend.py`).
  - We fuse this only in prefill of 64 or more rows (T91); decode keeps inverse RoPE, the Q8 product and a layout copy.
  - Estimate: 1–2% of decode.
- **One transform kernel around attention:** norm → RoPE → optional Hadamard → cast → paged store, with side outputs (`fused_transform.py`). Our V4 compressor pools, norms, ropes and stores in separate steps. Every KV and compressor writer with a norm/RoPE/cast/store chain is a candidate, not only DeepSeek's.
- **Merged projections by input** (`v4_modules/attn.py`). Mainly a prefill gain; a lossless import-time relayout where the quant types match, chosen by tensor role and encoding (AGENTS.md rule 11).

### 11. Measure first

Adopt each of these only if its measured share is material:
- **Two decode steps in flight** (`csrc/engine/zos.cpp`, `worker/worker.py`): the next step is scheduled before the current step's tokens are known. Our host round trip is 6–11 µs a step (D-106). Measure inter-step gaps with `tools/nsys_steps.py` first.
- **Decode graphs that don't depend on context length:** DeepInfer keys graphs by query rows only. Count first-time plans and captures over a long DeepSeek decode first.
- **Wave costs by context depth:** ours are calibrated below 32K; DeepInfer instead fits, online, a per-row slope over its startup curve from measured step times, which absorbs context growth (`mtp_varlen.hpp`).

## Owner decisions and open questions

- **Retention (decided).** DeepSeek's deployment keeps KV for 72 h or more (3 days for agent traffic, 7 for chatbot); a 3-day lifetime costs chatbot traffic 0.5–1% of its hit rate (deployment Q&A). The owner raised our idle cap from 24 hours to 7 days on 2026-10-20, so state survives an idle weekend (D-055 as amended). The spill budget stays the binding limit.
- **Approximate prompt-row skipping (decided):** an opt-in quality mode ([item 3](#3-row-liveness-pruning-exact-yoco-truncation)).
- **Moved earlier (decided).** Tool calls on the chat route, sampled speculation, minimal prefix reuse, Qwen3.8's row read-ahead and the dependency audits go to M3.5. The partitioning analysis goes ahead of M4's entry. Claude Code and Codex use the Messages and Responses routes, which stay M5's, so the chat route's tool calls unblock OpenCode first (Cursor stays an M5 validation gap).

## Not adopted

| Item | Why |
| --- | --- |
| Wide expert parallelism, EPLB, DeepEP, data-parallel attention, context parallelism, disaggregated prefill/decode/vision, load balancers | Datacenter scale; one user on one or two nodes |
| FlashMLA, DeepGEMM, DeepSelect kernels | SM90/SM100 instructions (wgmma, tcgen05), not available on `sm_121`; only their ideas port |
| A TileLang build toolchain | The TileLang ops are SIMT element-wise and reduction kernels. Porting the ideas into our own CUDA is cheaper than pinning TileLang and TVM, though build-time Python would be allowed (D-010) |
| Approximate sliding-window replay on prefix hits | Lossy. Exact window state is megabytes, while exact replay of a 128-token window costs seconds. A per-alias mode at most |
| FP8/FP4 main KV | Quality trade; V4.1's state is already small. A per-alias mode at most |
| Batch-invariant mode | Speed over bit-exactness (AGENTS.md rule 11); diagnostic value only |
| Forced top-k sampling (`DEEPINFER_SAMPLER_FORCE_TOPK`) | Lossy |
| A GPU sampler port | Our host sampler costs 0.45–2 ms per draw, and plain sampled decode is within 1–5% of greedy. Revisit only if [item 2](#2-speculative-sampling-with-the-drafters-distribution) or measured C4 sampling time warrants it; then its radix-select threshold and fixed-point CDF are the pieces to take |

## What the plan takes

| Item | Shared home | Plan placement |
| --- | --- | --- |
| 1. Confidence-budgeted verification | Speculation scheduler, drafter capability | M3.5, early: calibration screen on V4's DSpark, then the capability if calibrated (owner, 2026-10-20); M4's V4.1 drafter uses it; M9 extends it to later drafters |
| 2. Speculative sampling with q | Sampling and verify path, every drafter | M3.5, with the calibration screen (moved from M9 by the owner, 2026-10-20) |
| 3. Row-liveness pruning | Planner pass over the graph IR | M4, per model (V4.1 the large case; Gemma screened); the exact pass by default, plus an opt-in approximate mode per alias (owner, 2026-10-20) |
| 4. Token-keyed row read-ahead | Pager | M3.5, measured on Qwen3.8 first (moved by the owner, 2026-10-20); M4's Engram uses it |
| 5. FP4 indexer | Sparse-scoring operand encoding | M4, kernels (qualified per model) |
| 6. Expert parallelism with replicated attention | Partitioning planner | Before M4's entry, with the partitioning decision (V4.1 and GLM; moved by the owner, 2026-10-20) |
| 7. Shared-prefix cache | State contracts and retention | M3.5, minimal copy-based reuse after measuring real clients' prefixes (moved by the owner, 2026-10-20); M6, block sharing and the full policy |
| 8. Tool calls and structured output | Chat-format data, sampler | M3.5, tool calls on the chat route (moved from M5 by the owner, 2026-10-20); M5, the Responses and Messages routes; M7, constrained output |
| 9. Quality tripwires | Shared tooling | M3.5 |
| 10, 11. Fusions and measure-first leads | Planner patterns, engine | M9, full-engine optimization pass |
| Dependency audits: xgrammar, TileKernels | Source lock, licensing | M3.5 (owner, 2026-10-20) |
