<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek V4 Flash, native and resident (M3)

M3's first model slice ([plan](../../plan.md#m3--single-spark-fast-full-swap-in-progress)):
DeepSeek V4 Flash 0731 UD-Q2_K_XL run by llmpalooza natively and resident on one
Spark from its D-056 prepared artifact, against llama.cpp on the same GGUF.

*Since 2026-09-28 (the owner: speed before bit exactness, D-085's note):
the node-for-node graph planned unfused, which this slice measured, is
the optional reference mode (`llmp_dsv4_exec --exact on`); the default
is DeepSeek's fast plan, judged coarsely against llama.cpp
([dsv4-decode](../dsv4-decode/README.md)).*

## Pre-registration (fixed before llmpalooza's first run on the model)

Inputs, fixed in the repository (D-087's entry rule):

- **Checkpoint:** `unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93`, UD-Q2_K_XL,
  three shards pinned in [fast-swap/pins.json](../fast-swap/pins.json).
- **Oracle:** llama.cpp `b29c606e` (build 10964), the digest-pinned image
  `ghcr.io/ggml-org/llama.cpp@sha256:837fc732…`, through
  [oracle.cc](oracle.cc) ([run_oracle.sh](run_oracle.sh)): flash attention on,
  F16 caches, one sequence, context 4,096, ubatch 512, every layer on the
  GPU, no speculation. Two arms: upstream's CUDA fusion on (the default) and
  off (`GGML_CUDA_DISABLE_FUSION=1`). Llmpalooza's plan runs the graph unfused
  (the model-level fused operations, `dsv4_hc_*` and the lightning indexer,
  are graph nodes in both), so the unfused arm is the like-for-like one; the
  fused arm is reported beside it.
- **Tokens:** llama.cpp tokenizes (the native tokenizer is a parallel slice);
  BOS first, no chat template. Llmpalooza reads the oracle's token IDs.
- **Prompts:** the eight lines of [prompts.tsv](prompts.tsv), 32 greedy tokens
  each. The last prompt is long enough (over 128 tokens) to complete an HCA
  block.
- **Perplexity text:** `docs/async-model.md` at commit `e7973e5` (17,210
  bytes, SHA-256 `772a8b2d503df4fefd1a0a974ecae3ad58f8a4d1f2940fbc6bd440b5c86fb431`),
  BOS first, cut to 4,096 tokens, evaluated in 512-token chunks; every token
  after the first is scored. Over 2,048 tokens, so the indexer's top-512
  selection (512 of the visible compressed rows) is exercised.

Bounds:

1. **Greedy tokens.** Llmpalooza is teacher-forced on the unfused oracle's 32
   generated tokens per prompt, and its argmax at each of the 256 steps must
   equal the oracle's token, except at a step where the oracle's top-1 to
   top-2 logit margin is below twice the largest absolute logit difference
   between the two engines at that step (a near-tie); every exception is
   listed. Free-running greedy continuations are reported, not gated.
2. **Logits.** Reported per prompt: maximum absolute and RMS difference from
   each oracle arm, beside the two oracle arms' own difference (their
   fusion is the only change between them), which sets the scale of "small".
3. **Perplexity** within 2% (relative) of the unfused oracle's; the fused
   arm's is reported beside it.
4. **Resident expert layout:** every layer's routed products over the
   repacked slab equal those over the reference layout bit for bit.

Performance and peak memory are reported only (the gate comes with the swap
work): prefill of 512 tokens and 64 decode steps against `llama-bench -p 512
-n 64 -fa 1` in the same image, neither speculating; peak memory as the drop
in `MemAvailable` over the run.

## What runs

- **Artifact.** [import_m3.py](../artifact-layout/import_m3.py) runs the M0
  prototype (`layout.py`, unchanged; its SHA-256 stays the one the
  retained-backing trace pins) on the three GGUF shards after checking each
  against the pins' name, size and SHA-256, with converter
  `artifact-layout/import_m3.py` `m3-1+layout-a0d1980a9eddd1ad`. On
  `spark-b` it wrote artifact
  `8a355bfb27c90e1150fbd7fa62ea6e63f6bf34fcca33934e52d22773f1508234` to
  local NVMe in 11 min 16 s (full verify included, peak RSS 1.1 GB):
  96,841,173,712 bytes in 23 shards; 11,053 groups (1 token table, 43
  layers, 11,008 expert groups of 8,060,928, 9,306,112 or 10,878,976
  bytes, 1 head), 48,171 chunks, 0.000% disk padding: the plan of
  [artifact-format.md](../../artifact-format.md#worked-examples-measured-plans-of-real-files)'s
  worked example, which the 0731 revision shares.
- **Load** (`llmp_dsv4_exec`): the artifact is opened as untrusted input,
  bound to the compiled-in profile (`model/dsv4.h`;
  [gguf_profile_check.py](gguf_profile_check.py) finds its 37 key/values
  in both the 0731 and the `e3aa0d6a` GGUF) and read with
  coalesced direct reads through double-buffered pinned staging into
  `cudaMalloc` memory: 96,827,269,120 bytes in 7.1–7.3 s (13.2–13.3 GB/s).
  The token table (Q5_K) stays on the host, as llama.cpp keeps it on the
  CPU. The three hash-routed layers' token-to-expert tables, which the
  kernels index with unchecked, are refused unless every entry names one
  of the 256 experts.
- **Resident expert layout:** uniform stride over a per-layer slab, stock
  GGML kernels
  ([artifact-format.md](../../artifact-format.md#executable-views)): 41
  layers at 8,064,224 bytes, one at 9,309,200 and one at 10,888,976, 37.9 MB
  over the groups' stored bytes. `--layout-proof` ran every layer's gate,
  up and down products at 1, 5 and 64 tokens (387 cases, 258 on MMVQ and
  129 on MMQ) with random activations and routes over the slab and over the
  reference layout (the GGUF's packed `[k, n, 256]` tensor, rebuilt from
  the artifact's slices, which the import copied byte for byte from their
  source ranges): 147,947,520 outputs, 0 differing.
- **Graph** (`kernels/ggml/dsv4_graph.h`): `deepseek4.cpp`'s graph node for
  node, with its expand order, planned with fusion off through the
  registry (4,972 steps per chunk) and run on one stream. The state
  (`Dsv4StateLayout`, 225,329,152 bytes at context 4,096) is the window
  cache, the compressed and indexer caches and the compressors' rings, as
  three D-068 representations; the chunk inputs follow llama.cpp's
  compressor plans, dummy blocks included, so decode graphs keep one shape
  and are planned once per shape (14 plans in the correctness run).

## Results (`spark-b`, 2026-09-28)

GB10, driver 580.178.04. Raw outputs in
`~/.local/share/llmp/m3dsv4-20260928/` on `spark-b`.

**Correctness** (`compare.py`, bounds 1–4 above): every bound passes, with
no difference at all from the unfused oracle.

| Check | Result |
| --- | --- |
| Greedy, teacher-forced (bound 1) | 256 of 256 argmax equal to the unfused oracle's tokens; no exceptions |
| Logits vs unfused oracle (bound 2) | max abs 0.0 and RMS 0.0 on all 8 prompts × 32 steps (bit-identical) |
| Fused vs unfused oracle (the scale) | max abs 3.1–68.4, RMS 0.18–3.85 per prompt; llmpalooza vs fused is the same |
| Free-running greedy | 8 of 8 prompts' 32 tokens identical to the oracle's |
| Perplexity (bound 3), 3,540 tokens | llmpalooza 20.2311, unfused oracle 20.2311, fused 20.2149; every per-token NLL equal to the unfused oracle's |
| Prefill intermediates, prompt 1 | 38 of 41 named tensors, from `hc_init` through every layer kind to `result_output`, bit-identical; the other 3 do not compare (llama.cpp reuses two names for earlier tensors, and the indexer's scores hold -inf) |
| Expert layout (bound 4) | 0 of 147,947,520 outputs differ |

The first perplexity run differed after position 592 (PPL 20.2104, NLL max
difference 1.21) because llmpalooza's window cache was a 768-cell ring while
llama.cpp's contexts default to a full-size SWA cache: the attention length,
and with it the summation order, differed. The cache now holds a cell per
position as llama.cpp's does, and every NLL matches.

**Performance and memory** (reported, not gated; neither side speculates):

| | Llmpalooza | llama.cpp (`llama-bench`, fusion and CUDA graphs on) |
| --- | --- | --- |
| Prefill, 512 tokens | 368.7 tok/s (best of 3) | 340.5 ± 27.0 tok/s (pp512, mean of 3) |
| Decode, 64 tokens from empty | 19.41 tok/s (mean of 3, after a warm-up) | 20.62 ± 0.09 tok/s (tg64) |
| Decode in the prompt runs (31 steps) | 19.4–19.5 tok/s | 19.8–20.4 tok/s (oracle, unfused and fused) |
| Load | 7.1–7.3 s (artifact, direct reads) | 86–95 s (GGUF, plain reads, oracle) |
| Peak memory (drop in `MemAvailable`) | 92.4 GiB (96,910,964 kB) | 93.1 GiB (97,585,588 kB) |

So prefill is 1.08× llama.cpp's, decode 0.94×, peak memory 0.99×. Llmpalooza
launches each of the 4,972 steps from the host every token (no CUDA
graphs yet, an M3 item), where llama-bench replays a captured graph; that
is the likely share of the 6% decode gap, not measured. *Measured since*
([graphs](../fast-swap/graphs.md)): graphs are worth 4–5% on the paged
node, and most of the gap was fusion (llama.cpp with fusion off: 20.04
tok/s with graphs, 19.86 without); what remained was the paged node's
per-step round trip. With a lease per request (D-093,
[swap](../fast-swap/swap.md#a-lease-per-request)) that is gone: the paged
node decodes at 20.34–20.46 tok/s with graphs, 1.016–1.022× llama.cpp with
fusion off and graphs on (20.02, the same session) and 0.990–0.996× with
fusion on (20.54). Those runs were harness-polled (every lane spinning
through each step). With the runtime's own wake (D-094,
[runtime-wake](../runtime-wake/README.md)) it decodes at 20.42–20.50
tok/s, a round trip of 0.03–0.05 ms a step, against 20.32 with the 100 ms
windows in the same session: 1.020–1.024× llama.cpp with fusion off
(20.01, the same session) and 0.999–1.003× with fusion on (20.43).

## Judgement calls

- **Importer:** option (a), a new file that imports the pinned `layout.py`
  rather than copying it, so nothing is duplicated and the trace's pin
  stays valid.
- **Expert dispatch:** uniform stride with stock kernels, no A/B: the
  pointer table needs a GGML kernel patch (a source-lock change) and buys a
  resident model nothing; it stays M7's for demand-paged experts.
- **Oracle arm:** Llmpalooza's plan is unfused (the model-level fused operations
  are graph nodes), so the unfused arm is the like-for-like one; upstream's
  fusion changes logits by up to 68 on these prompts.
- **Window cache:** full size, as llama.cpp's default; a ring of window +
  chunk cells would use 180 MB less at context 4,096 but reorders
  attention's sums.
- **Resident on `cudaMalloc`** like the backend proof's rung 3; jobs over
  leased closures on the paged node need the slab's placement in the
  catalog's extents, which is the swap path's. *Done since:* the paged
  runner's logits equal this harness's bit for bit on all 8 prompts
  ([swap](../fast-swap/swap.md)).

## Limits

- One sequence, context 4,096, chunks of at most 512 tokens, no rollback
  planes; long-context (beyond 4,096) and multi-sequence behaviour are
  untested.
- The oracle's tokenizer stands in for the native one; no chat template.
- The executed plan is not recorded against llama.cpp's launch by launch
  (M2's comparator); the bit-identical intermediates and logits are the
  evidence instead.
- The prompts' decode steps and the perplexity text exercise the indexer's
  top-512 selection only past 2,048 positions (the perplexity text).
