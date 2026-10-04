<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorFold assessment

This assessment retains historical source reviews and measurements. New
tasks use the latest TensorFold under [reference comparisons](reference-comparisons.md),
which refreshes its pin for each applicable task and records current
coverage observations separately.

The owner asked, on 2026-09-26, that [TensorFold](https://github.com/ashhart/TensorFold)
be considered as a benchmark target and as a source of optimization ideas.
This is a source review of commit
[`d7470ed`](https://github.com/ashhart/TensorFold/tree/d7470ed8f6365ad3c7c7268f3c32da548eb1c343)
(version 0.3.1). Every number about TensorFold in the sections up to
[Proposed use](#proposed-use) is creator-reported and unverified; only the
`/proc` observations in item 6 are ours. TensorFold has since been pinned
and measured on our Sparks as an M3 baseline
([baselines](experiments/fast-swap/baselines.md#qwen38-flash-next-tensorfold-mlx-4-bit-cross-quantization)),
its techniques measured against jitLLM's
([below](#measured-and-adopted-m3-2026-09-28)), and its later commits
surveyed [at the end](#upstream-to-0362-2026-09-28).

## What it is

- **Engine.** A single-stream local inference engine with an
  OpenAI-compatible `/v1/chat/completions` endpoint, for Apple Silicon
  (MLX) and NVIDIA (CUDA).
- **Maturity.** The repository was created on 2026-09-25, with 14 commits
  and "Development Status :: 3 - Alpha". The CUDA engines for the DGX Spark
  landed on 2026-09-26.
- **Runtime.** On CUDA, each supported model family has its own PyTorch
  forward pass with Triton and CUDA kernels written for that family
  (`src/tensorfold/families/<name>/cuda/`). It runs inside NVIDIA's PyTorch
  container; kernel `.cu` sources build on first use with the container's
  NVCC.
- **Models.** On CUDA: Qwen3.8-27B (dense), Qwen3.8 Flash Next, and
  GLM-5.3-Flash (two Sparks only). Nemotron 3.5 Lightning appears in its
  Mac results. Families are recipes chosen from the checkpoint's
  `config.json`, and a checkpoint without a recipe is refused.
  - Qwen3.8 Flash Next is already one of our candidate models
    ([features](features.md), MoE reference candidates).
- **Formats.** MLX 4-bit checkpoints from Hugging Face only. It refuses
  EXL3, NVFP4, GPTQ and AWQ.
  - The CUDA path reads the *format*, not the MLX library: `mlx` and
    `mlx-lm` are macOS-only dependencies, and no CUDA module imports them.
  - The format is 4-bit affine group quantization: groups of 64, eight
    values packed per 32-bit word, and a bf16 scale and bias per group
    (`families/qwen3_5/cuda/qmm.py`). That is close to GGUF `Q4_1` or
    asymmetric GPTQ. That file is the dense Qwen3.8-27B family's; Qwen3.8
    Flash Next's checkpoint and family use groups of 32 (checked at
    `beddbb7b`, [licensing.md](licensing.md#tensorfold)).
  - Its Triton matmul computes
    `Σ_g (scale·(x·q) + bias·Σx)`, over the groups in order, after
    regrouping the packed words at load.
- **Hardware.** Per its README, tested on one and two DGX Sparks (GB10),
  with tensor parallelism over NCCL across the direct link. That is our
  exact target.

## License

- **Code.** TensorFold's own code is MIT, which is core-eligible under
  D-017. It adapts code from mlx-lm (MIT), transformers (Apache-2.0) and
  z-lab's DFlash (MIT, vendored unmodified). PyTorch (BSD-3-Clause) and
  Triton (MIT) come from NVIDIA's container, not bundled.
- **Model weights.** It ships none. The Qwen3.8-27B DFlash2 drafter is
  Apache-2.0 per its model card. GLM-5.3-Flash's optional drafter is
  CC BY-NC-ND 4.0 (non-commercial, no derivatives), so it must not enter
  any jitLLM artifact or default. As a benchmark-only component it needs
  the owner's decision. *Owner, 2026-09-28 (D-087): allowed. jitLLM never
  distributes weights, so a model's weight license gates nothing; M4's GLM
  may use DFlash2 or MTP, whichever is faster and correct.*
- **Architecture.** Its runtime is Python and PyTorch. jitLLM's hot path
  stays native (D-010), so reuse would mean porting ideas or individual
  kernels under their licenses (D-013), never adopting its runtime.

## Its claims on DGX Spark

The README and `docs/recipes/cuda.md` compare it with vLLM using MTP=3,
through the same OpenAI client:
- one stream, 64-token replies;
- the median of seeds 1234–1238;
- code and chat prompts, sampled (T 1, top-k 20, top-p 0.95) and greedy.

"Each TensorFold number is byte-identical to its own serial decoding", and
the recipe notes that vLLM's drafted output is not.

| Model | Sparks | vs vLLM with MTP (range over the four columns) |
| --- | --- | --- |
| Qwen3.8-27B + DFlash2 | 1 | 2.70–3.05× (for example 49.6 vs 17.7 tok/s, code sampled) |
| Qwen3.8-27B + DFlash2 | 2 | 1.94–2.49× |
| Qwen3.8 Flash Next | 1 | 1.60–1.79× |
| Qwen3.8 Flash Next | 2 | 1.74–2.24× |
| GLM-5.3-Flash | 2 | 1.78–2.06× |

These are decode rates for short replies to short prompts: the benchmark
prompts were 14 and 31 tokens, per its own notes. They say nothing about:
- prefill;
- long contexts;
- model switching;
- concurrency.

The comparison also mixes engines *and* drafters: DFlash2 draft trees
against vLLM's MTP.

## Ideas worth taking

Mapped to where they would land in jitLLM:

1. **Exact speculative decoding by row-invariant kernels.**
   - By its design notes, every kernel on the verify path gives a row the
     same bits whether it runs alone or in a window of *n* rows. Split-K depends only on the
     weight's shape; slices are added in order; attention is chunked by the
     row's own key range; there is no reduction across rows.
   - Drafts are then accepted exactly when serial decoding would produce
     them.
   - This is stronger than our Tier E exactness and rung-5 restore checks,
     which compare the same plan and trajectory across paging. They do
     not establish equality between serial and batched execution. For
     D-068 (M3 onward), exact speculative verification would need that additional
     guarantee across its draft and verify shapes. The M2 operation
     contract should record row-invariance per kernel, which the P0 study
     partly shows:
     - GGML's vector attention kernel is row-invariant by construction;
     - cuBLAS's algorithm choice is not.
2. **Lay weights out for the kernel's read.** Regrouping MLX's packed
   4-bit words into contiguous per-program blocks took the 27B's matmuls
   from 107–130 to 200–220 GB/s, against 240 it reports measuring for
   GB10 (creator-reported). That bears
   on D-056's import-time repacking: the prepared artifact can carry the
   layout the chosen kernel reads best, not only paging-friendly groups.
3. **Account host time against GPU time before reaching for graphs.** Its
   notes report that the 27B's 918 kernels take 12–14 ms of host time, hidden behind 50–85 ms of
   GPU work, so graphs would buy little. Small MoE forwards are
   launch-bound and need them. This is BP-F4's decision rule, with a worked
   example.
4. **Deterministic tensor parallelism across two Sparks.**
   - Row-parallel partial sums are all-gathered and added in rank order.
     NCCL's all-reduce order is an implementation detail, so it is not
     used.
   - The head is split by vocabulary, with a top-k merge.
   - This is directly relevant to M4 sharding if restored or migrated
     state must reproduce bit for bit.
5. **Choose drafters and window widths by measurement.**
   - The engine picks per request whichever drafter commits more tokens
     per millisecond.
   - The window grows only while committed tokens per round-time rise.
   - Draft heads read a partial vocabulary. Its notes report that 98,304
     ids cover 99.6% of committed tokens.

   These are inputs to M3's drafting and to M9.
6. **A trap to check on our Sparks.** Its notes report GB10's kernel
   migrating pages under a running model. Some runs went at half speed
   during bursts of 45,000–265,000 migrated pages, and dropping the page
   cache after load moved about 25 GB. Our design leans on unified memory
   and the page cache (D-004, D-034), and this could also explain some of
   P0's between-process timing noise.

   A first look at `spark` on 2026-09-26 (read-only, `/proc`):
   - automatic NUMA balancing is off (`kernel.numa_balancing=0`), and
     `numa_pages_migrated` is 0;
   - memory compaction has been moving pages. Since the 2026-09-21 boot:
     - `pgmigrate_success` reached 58,470,221. That counts every source of
       migration;
     - `compact_isolated` and `compact_success`/`compact_fail` tie most of
       it to compaction;
     - there were 162,635 direct-compaction stalls (`compact_stall`) and
       443,297 wake-ups of the compaction daemon;
     - `compaction_proactiveness` is 20, the kernel default;
   - over one minute during a P0 timing session, `pgmigrate_success` did
     not change.

   So compaction, not NUMA balancing, is the likely mover here. The
   migrations may cluster around large allocations and page-cache churn.
   It is worth one measurement in M3: the counters sampled across
   loads, evictions and a steady decode.
7. **Benchmark method.**
   - The same client for both engines.
   - Medians over seeds, since, per its notes, single seeds varied up to
     2×.
   - Nothing else on either GPU.
   - A hash check of the drafted output against serial decoding on every
     run.

   This matches our frozen protocol's spirit, and the hash check is a good
   addition wherever we compare decoding with drafts.

## Measured and adopted (M3, 2026-09-28)

The techniques study ([tensorfold-techniques](experiments/tensorfold-techniques/README.md))
profiled both engines on Qwen3.8 Flash Next and took what transfers within
Mia's format:

- **The decode gap is format, not engine.** A decode token reads 7.3–7.6 GB
  in Mia's NVFP4, MXFP8 and BF16 against 4.5 GB in TensorFold's MLX 4-bit
  (its dense layers, hyper-connection products and head are 4-bit where
  Mia's are 8- and 16-bit; its experts read slightly more). jitLLM reads
  about 202 GB/s against TensorFold's 165–178, so its engine is already the
  faster reader; the 1.3–1.45× is bytes.
- **Item 2 (layouts for the kernel):** jitLLM's MXFP8 vector product
  already streams 221–229 GB/s alone at Qwen3.8's shapes in the artifact's
  row-major layout, and the experts' layout is fixed by the prefill's
  grouped GEMM; no new import layout was warranted. What cost bandwidth in
  the model was other traffic: the recurrent state copied back after each
  Gated DeltaNet step. Adopted: the state updated in place (TensorFold
  double-buffers it), half the state bytes.
- **Items 2 and 3 (bytes in flight, host time):** decode already runs as
  CUDA graphs with a 0.04 ms round trip, so host time is not the lever.
  Adopted: the hyper-connection products of a decode step on jitLLM's own
  BF16 vector kernel instead of cuBLAS's gemv, the hyper-connection prep
  across a cluster of blocks instead of one, the one-row convolution
  fused, and programmatic
  dependent launch with L2 prefetch of the next product's weights (as
  DeepSeek's `jitllm.vecq`).
- **Item 5 (adaptive window, partial vocabulary):** the MTP drafter now
  reports each draft's probability and the harness can cut a round's
  drafts below a threshold (TensorFold's 0.3); at depths 2–4 it gained
  nothing measurable (every draft pass still runs), so it stays off.
  Qwen3.8's draft head already reads 65,536 rows; DSpark's whole head
  costs about 0.7% of a step.
- **Result:** Qwen3.8's plain decode +5.4–8.1% (35.2 ms a step against
  37.4 in the same session), 0.73× TensorFold's instead of 0.69×; the rest
  is the format.
- **Item 1 (row-invariant exact speculation):** not adopted; the owner's
  rule is speed before bit exactness (D-085's note), and DeepSeek keeps
  D-092's row-invariant verify as its exact mode.
- **Item 6 (page migration):** measured on `spark-b`: loads, evictions,
  swaps and steady decode migrated no pages and triggered no compaction
  when nothing else held memory; bursts (up to 113,000 pages in 5 s) came
  only when another process's allocations overlapped a load, which then
  read at 1.1 GB/s instead of 13.

## Proposed use

- **As a benchmark target.**
  - It becomes a pinned reference engine for the models it supports that
    overlap ours: Qwen3.8 Flash Next now, others as they enter both
    libraries.
  - Its comparisons would use its normal configuration (drafts on) and a
    matched one (`--no-drafts`, its serial reference), as D-021 and D-068
    ask. They land where drafting does: M3 for resident decode and
    Qwen3.8's stored MTP, and M9 for other drafters.
- **Format.** It reads only MLX 4-bit. A like-for-like comparison needs
  either jitLLM support for that format, which is a new import format and
  so a feature question, or an explicit cross-quantization comparison
  reported as such.
- **Before relying on its numbers.**
  - Pin a commit, and run its own benchmark client on `spark` against its
    serial mode and a pinned vLLM, as a reference-only run.
  - Record the result under `docs/experiments/`, as with the other
    references.
  - This waits until the Spark is free of P0's timing sessions.

## Upstream to 0.3.6.2 (2026-09-28)

The pin moved from `beddbb7b` (0.3.5.1) to main's tip
[`71377a53`](https://github.com/ashhart/TensorFold/tree/71377a5373ed7b394f1b480ba2a6a3986b03af1c)
(0.3.6.2): 58 commits, still MIT ([licensing](licensing.md#tensorfold)). On
our Qwen3.8 Flash Next baseline the tip measures within 4% of the first pin
([baselines](experiments/fast-swap/baselines.md#qwen38-flash-next-tensorfold-mlx-4-bit-cross-quantization)).
Of the load study's four cold-start patches, one is upstream: JIT kernels
built for the GPU present (`34bae79`, 0.3.6.1). The O_DIRECT reader, the
int32 nibble shuffle and the RUNBOOK's cache volumes still apply as they
were. The owner is taking those upstream.
2026-09-29: the owner opened
[TensorFold#82](https://github.com/ashhart/TensorFold/pull/82), which builds
on the reader and the repack; status in
[upstream/tensorfold.md](upstream/tensorfold.md).

What the commits offer jitLLM, read from the code. Every TensorFold number
here is creator-reported, on one GB10 unless stated:

1. **Grouped EXL3 routed experts, every codebook and mixed widths**
   (`1bbd2dd`; `cuda/exl3/experts.cu`, `experts_grouped.cuh`).
   - What it does: one launch per projection covers a whole MoE layer.
     Each expert has its own width (1–8 bits, half-bit steps) and is read
     in place through per-expert pointer tables. Grouping, the input
     rotation, the SwiGLU epilogue and down plus combine are fused. It
     uses no atomics and no host sync, so it can be graph-captured.
   - Gain (micro-benchmark on MiMo's real experts): 1.5–3.6× ExLlamaV3's
     `exl3_moe_mixedk`, for example 179.6 against 49.9 GB/s at one row.
     It is within 2% of `exl3_moe_coop` on uniform layers, which coop
     alone can dispatch.
   - jitLLM has no EXL3 MoE yet (M4 kernels). The pointer-table read suits
     demand-paged experts (M7). **Lands in M4, then M7. Highest value.**
2. **A latent MLA cache for GLM** (`fb985b8`, `279d8f7`;
   `glm5_next/cuda/latent.py`, `sparse.py`).
   - What it does: stores only the 512-wide latent, about 1 KB a token and
     layer. Queries absorb `kv_b`'s key blocks and outputs expand through
     its value blocks. A radix select picks the top 512 pools, with a
     graph per pool bucket.
   - Gain (two Sparks, with MTP): 256K context. Prompt 1,138 / 898 / 849
     tok/s and decode 50.9 / 47.2 / 31.9 tok/s at 32K / 131K / 256K. Loss
     within 0.001 nats of the per-head cache.
   - The M4 GLM cache design, and it sizes GLM's swap spill. **M4.**
3. **Confidence-gated MTP chains** (`decode.py:draft()`, present at
   `beddbb7b`; `163da97` only adds `--mtp-confidence`).
   - What it does: keeps the first draft, then stops before any draft whose
     probability is under 0.30, up to 6 a round. It costs a host sync per
     draft. The commit calls 0.60 and 0.75 "the measured policies", with
     no numbers.
   - On our prompts this gives 2.21 / 2.35 tokens a round (measured).
     jitLLM's Qwen3.8 drafts a fixed 2 inside one graph. A depth-3 verify
     costs about 5.7 ms more there, and depth 3 measured slower overall. A
     device-side stop could recover depth without paying for rejected
     rows. **M3 speculation, or M9's draft-length tuning.**
4. **Quantized KV caches, int8 and int4** (`549b7a7`, `22649b7`, `b8612e4`,
   `6ed268b`).
   - What it does: ExLlamaV3's `-cq 8` / `-cq 4` arithmetic. Groups of 32
     are Hadamard-rotated with an fp16 absmax scale. The query is rotated
     to match and the output rotated back, so the stored codes stay
     rotated. Indexer and pooled keys stay bf16.
   - Gain: 30,784 → 18,304 → 11,648 bytes a token (1.68× / 2.64×). No
     speed or quality number. The test checks scales bit for bit against
     ExLlamaV3's quantizer, written independently.
   - A checkable candidate for the deferred quality/performance modes
     item. If M4's Mia oracles run `-cq`, it becomes a same-format need.
     **Deferred modes; check at M4.**
5. **A row-invariant EXL3 linear for every codebook and width** (`e9780ed`,
   `b16fb91`, `38235a9`; `cuda/exl3/linear.cu`, `decode.cuh`).
   - What it does: K-split ranges depend only on (K, N), for 1 to 128 rows.
     `38235a9` loads each k step's low-bit words as one coalesced run a
     warp, shuffled out, with the next step in flight.
   - Gain (micro-benchmark, one row): 2-bit o_proj and down 76–79 → 186–193
     GB/s, level with ExLlamaV3's own linear (176–233).
   - jitLLM already runs ExLlamaV3's kernels at their speed
     (`src/kernels/exl3/`), for mcg at integer widths only, and they are
     not row-invariant. This is an MIT reference for more codebooks and
     widths, and for exact speculative verify on EXL3 (D-092). **M4.**
6. **Flash Next and the 27B on EXL3 checkpoints, with MTP** (`d7d18e0`,
   `5b4b343`; measured in `654e4ad`, `docs/recipes/qwen3.8-flash-next.md`).
   - Flash Next on turboderp's 3.05 bpw pack decodes at 59.4–80.8 tok/s,
     against 62.7–76.5 for its MLX 4-bit and 33.2–42.4 for vLLM MTP 3.
     Those are 64-token replies to its own prompts.
   - Without drafts it runs 1.51–1.55× ExLlamaV3 on a mixed-width pack and
     5–6% behind it on the uniform 3.05 bpw pack. Top-1 agreement with
     ExLlamaV3 is 0.954–0.959. Prefill is 930–970 tok/s, about 0.4× MLX's.
     Weights take 52 GB against MLX's 81 GB.
   - The pack traps are directly reusable in M4: norms stored as gamma−1,
     the MTP mixer kept outside the index, and ExLlamaV3's n-gram row
     codec. **M4; an EXL3 Flash Next is an M9 format option.**
7. **Streaming routed experts from SSD into a GPU pool** (`--ssd-experts`,
   0.3.6, `src/tensorfold/streaming/`; Metal only).
   - What it does: the GPU signals each MoE layer's picks to a host thread
     through a shared event inside the command stream, then waits. The host
     loads missing experts into an LRU of slots and writes the slot table.
     The kernels are the resident ones with a different address, so the
     tokens are identical.
   - Gain (M3 Ultra, held to a 64 GB Mac's budget, 24 GiB pool): decode
     0.31–0.39× resident, prefill about 0.33×.
   - This is D-008's routing-as-dependency-discovery on another stack,
     with a working in-stream wait. **M7.**
8. **Smaller items.**
   - The prefill head on the final chunk only (`96c0b0e`): jitLLM's chunk
     graphs already compute only the last rows' logits.
   - Prompt kernels at startup (`416106f`): 26.4 s of first-start compiles
     moved ahead of `/health` (measured). jitLLM compiles ahead of time, so
     this does not apply.
   - Evicted-prefix spill to disk (`675d4c2`, Mac): 2.2–2.4 GiB written in
     0.18–0.19 s and read back in 0.24 s, answering in 2.3 s against
     26.2 s cold. It corroborates M6's retention spill.
   - N-gram rows read a chunk ahead while the previous chunk computes
     (`ea709da`, Mac): a small lever for our synchronous row reads in M3
     prefill.
   - Concurrent 27B serving, and a request that stops or fails leaving its
     neighbours exact (`2fbebf7`, `9a1e6f1`, `24afe5e`, `ce09822`). These
     are M5 and M6 test cases, for client disconnects and admission
     failure.

Ranked by likely value to jitLLM: the grouped mixed-width EXL3 experts,
the latent MLA cache, confidence-gated drafts, the quantized KV caches, and
the row-invariant EXL3 linear.
