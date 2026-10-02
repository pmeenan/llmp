<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorFold

- **Repository:** [ashhart/TensorFold](https://github.com/ashhart/TensorFold),
  MIT through 0.5.0, Apache-2.0 from 0.6.0 with the earlier MIT notice
  retained. It serves MLX, EXL3 and NVFP4 checkpoints through family-specific
  engines, including concurrent Flash Next CUDA execution on a DGX Spark.
  It is an M3 baseline for Qwen3.8 Flash Next and a
  source of techniques ([tensorfold-assessment.md](../tensorfold-assessment.md)).
- **Historical baseline pin:** `71377a53` (0.3.6.2), measured as a baseline
  ([baselines](../experiments/fast-swap/baselines.md#qwen38-flash-next-tensorfold-mlx-4-bit-cross-quantization)).
  None of jitLLM's code comes from TensorFold.
- **The load study** is outside this repository, on the workstation at
  `/home/pmeenan/src/tensorfold-load-study/` (2026-09-28). Its `README.md`
  has the method and phase tables; `tensorfold/PATCHES.md` is the
  self-contained handoff, with diffs against `beddbb7b` and `71377a53` in
  `tensorfold/patches*/` and the tests in `patches/tests-test_load_io.diff`.

## Cold start: buffered reads and an int64 repack take ~130 of ~145 s

- **Status:** landed in 0.5.0 as `8ae247f`, according to the maintainer's
  2026-09-29 reply on
  [ashhart/TensorFold#82](https://github.com/ashhart/TensorFold/pull/82),
  "perf: Improve cuda startup/loading performance (3-15x faster on DGX
  spark)", by the owner. Follow-up `da8a5df` releases the direct reader's
  pinned staging after each load. Rechecked against upstream on 2026-10-02.
- **Found:** 2026-09-28, TensorFold `beddbb7b` (0.3.5.1), `spark`.
- **Problem:** Flash Next's `weights.load` reads 80 GB through small buffered
  reads (the kernel's 128 KiB readahead at a queue depth under 1:
  1.1–1.2 GB/s from an SSD that reads ~13 GB/s), then repacks expert nibbles
  with an int64 loop (~1 s of GPU time a layer). The two do not overlap.
- **The study's patches (measured):** O_DIRECT reads into two pinned 64 MiB
  pieces with asynchronous copies, and the nibble shuffle in int32
  (bit-identical). Start to ready 145.1 → 45.5 s, same greedy output token
  for token.
- **What the PR does**, building on the study: O_DIRECT checkpoint reads
  with read-ahead across neighbouring tensors
  (`tensorfold/cuda/direct_read.py`); a fused CUDA expert-pack kernel
  (`tensorfold/cuda/experts_pack.cu`) replacing about 40 elementwise
  operations; GLM experts stacked on the GPU; fewer device synchronisations.
  Reported in the PR: 2.9–15× across models; Flash Next MLX 146 → 24 s to
  first token. Not re-measured in jitLLM.
- **Proposed action:** use the upstream implementation in the updated
  baseline; no reapplication of the old PR is needed.

## Same-checkpoint NVFP4 comparison

- **Status:** measured at 8K with one, two and four concurrent requests;
  [conditions and results](../experiments/serving-concurrent/README.md).
- **Source:** release 0.6.2, commit
  `56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21`, inspected 2026-10-02.
- **Capability:** the [current Flash Next recipe](https://github.com/ashhart/TensorFold/blob/56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21/docs/recipes/qwen3.8-flash-next.md#nvfp4-checkpoints)
  explicitly supports `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`,
  the checkpoint already used by jitLLM, including its MTP head. The
  old 0.3.6.2 baseline accepts MLX/EXL3 and cannot run this ModelOpt export.
- **Actual conditions:** same model files and full rendered prompt IDs,
  context 33,792 and 256 generated tokens per request, zero cached prompt
  tokens. TensorFold uses BF16 KV, a 79,591-entry draft vocabulary and MTP
  cap 6/confidence 0.30; native uses F16 KV, 47,172 entries and depth 2/3.
  Flash Next's released loader uses BF16 activations even when the global
  `--precision checkpoint` flag is passed. Matching weight format does not
  establish matching arithmetic or quality. The old affine-format and Mia
  measurements retain their original provenance.
- **Result:** native's completed-token rate, including prefill and queueing,
  is 19.22% higher at C1 and 8.63% higher at C2, but 14.55% lower at C4.
  Current TensorFold starts in 198.870 s with the first kernel compilation,
  then 73.367/73.431 s with its compiled cache retained. One sample per cell;
  the remaining concurrency gap needs attribution.

## RUNBOOK: persist the kernel caches (study patch 4)

- **Status:** open.
- **Found:** 2026-09-28, `beddbb7b`; still applies at `71377a53`.
- **Problem:** in NVIDIA's container nothing persists `~/.cache` and
  `~/.triton`, so every start in a new container spends ~60 s compiling
  kernels a cache would keep (baseline difference).
- **Proposed action:** a small documentation PR: named volumes for both
  caches in `RUNBOOK.md` (`patches/patch-4-runbook-persist-kernel-caches.diff`
  in the study, untested as written). It is not among PR #82's listed
  changes; check the PR before sending it separately. Minutes.

## JIT kernels built for every architecture fail in NVIDIA's container (fixed upstream)

- **Status:** fixed upstream at `34bae79` (0.3.6.1, "CUDA builds inside
  NVIDIA's containers again", fixes TensorFold #56).
- **Problem:** the first start compiled for `TORCH_CUDA_ARCH_LIST`, which in
  NVIDIA's container starts at sm_80, and `qmm.cu` failed there.
- **Proposed action:** none.

## Upstream techniques to adopt

- **Tiled, bit-exact QSA block selection past 131,072 keys:**
  [ashhart/TensorFold#93](https://github.com/ashhart/TensorFold/pull/93)
  (open, by MovieMaker93, 2026-09-29), "perf: Flash Next lists QSA blocks
  past 131,072 keys in tiles (256k prompts 2.1x faster, same bits)", in
  `families/qwen4_exp/cuda/attention.py`.
  - TensorFold's `_select` kept the block scores in registers up to 32,768
    blocks. Past that they spilled, and a 256-row prompt block's layer took
    26–27 ms instead of 0.8 ms on a DGX Spark.
  - The fix, `_select_tiles`, processes longer rows in tiles (4,096 blocks
    for prompt rows, 8,192 for decode windows) with a radix select for the
    512th-largest order-preserving key, one byte per pass by histogram,
    keeping the block order identical.
  - Reported: 27.2 → 1.5 ms per block at 262K keys; a 250K-token prompt
    311 → 145 s and a 140K one 86 → 74 s; identical state after 140K and
    250K prompts, with BF16 and int8 KV. All creator-reported.
  - **Relevance to jitLLM:** `jitllm.qsa.select`
    (`src/kernels/ggml/jitllm_ops.h`, 8,192 blocks in 32 KiB of shared
    memory, 32,768 cells at Qwen3.8's ratio of 4) fell back to GGML's
    nondeterministic top-k past that
    ([ggml.md](ggml.md#radix-top-k-breaks-ties-nondeterministically-re-031),
    RE-031). TensorFold is MIT, so porting the idea is fine (D-091).
  - **Adopted (the technique, jitLLM's own code),** long context's phase 2
    (2026-09-29): `jitllm.qsa.topk` (`src/kernels/ggml/qsa_sparse.cu`)
    selects each token's 2,051 cells in tiles of 8,192 blocks (a byte-wise
    radix select for the width-th order-preserving key, ties to the lower
    cell), then once more over the tiles' candidates; Qwen3.8 repeats bit
    for bit at 64K and 128K and speculates to 262,144
    ([long-context](../experiments/long-context/README.md#phase-2-qwen38-flash-next-flat-with-depth)).
  - **Proposed action:** none; no upstream action.
- Other TensorFold techniques, ranked for jitLLM, are in
  [tensorfold-assessment.md](../tensorfold-assessment.md#upstream-to-0362-2026-09-28).
