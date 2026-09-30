<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ds4

- **Repository:** [Entrpi/ds4](https://github.com/Entrpi/ds4), fork of
  [antirez/ds4](https://github.com/antirez/ds4), MIT.
- **Study pin:** `76d51ef82a81b70b78e51a3a6ea11946286de976` (M3,
  2026-09-29), CUDA 13.4.92, `sm_121`, on `spark`.
- **Evidence:** [same-GGUF study](../experiments/ds4-study/README.md).
  No ds4 source is linked into jitLLM.

## Cold-context measurements require a fresh session

- **Status:** no upstream action; measurement method corrected.
- **What:** `ds4_bench.c` reuses one session as `ctx-start` advances to
  `ctx-max`. Its reported prefill rate is the added frontier tokens,
  rather than a fresh prefill of each reported context. jitLLM's first
  ladder used that result as a cold rung and excluded it after inspection.
- **Reproduction:** a rung uses a fresh `ds4-bench` process with
  `--ctx-start N --ctx-max N --ctx-alloc N_PLUS_1024`; study commands,
  model hash, prompt hash and chunk policy are in the report.
- **Proposed action:** none. Distinguish incremental and cold measurements
  in future comparisons.

## DSpark and MTP use different interfaces

- **Status:** no upstream action; invocation corrected.
- **What:** `ds4-bench --mtp FILE` expects MTP head tensors.
  `DeepSeek-V4-Flash-DSpark-support-0731.gguf` is a DSpark block drafter;
  the benchmark refuses it with missing `mtp.0.hc_head_base.weight`.
  `ds4-server --dspark` supplies the block-drafter interface.
- **Proposed action:** use plain decode for the serial same-format
  benchmark. Do not treat the refusal as a DSpark speed result.

## Transferable prefill techniques

- **Status:** study recorded; no upstream patch.
- **What:** isolated profiles show fused IQ2 gate/up products and Q2 down
  products, expert-major tiles, and token-tiled attention. jitLLM shares
  sparse attention gathers across up to eight queries using GGML's
  query-tile union, an explicit fast-plan choice that defaults off for
  unknown sparse patterns after disjoint D512 lists regressed. Its paired-MMQ path shares the expert maps and
  quantized input preparation while retaining the two ordinary products.
- **Rejected direct transfer:** `cuda/mmq/ds4_mmq_d2r.cu` requires dense
  whole-array SoA `[half scales, pad64, uint2 codes]`; jitLLM's resident
  experts use raw GGUF blocks at padded per-expert strides. ds4's fused
  epilogue weights the activation before Q8 quantization and the down
  product; jitLLM weights after down. It also replaces nonfinite values
  with zero. A direct kernel copy would change layout and arithmetic.
  No SoA replica, early weighting, sanitization or cache precision change
  is adopted to close the remaining speed gap.
- **Q2 prototype assessment:** GGML's D2S6 activation layout retains
  original F32 subgroup sums for its affine minimum correction. A simple
  fully dequantized F16 product would discard that correction. Matching
  it through per-16-value integer dots and epilogues adds synchronization
  to an existing integer-MMA path; no speed advantage is established, so
  the source proposal is not implemented.
- **License:** the fork's root LICENSE is MIT. Its D2R product source
  also attributes Marco Palaferri's MIT code from
  `xangel82/DS4-GB10-GX10-DSpark-CUDA` at `910501e`. This study ports no
  D2R source; future reuse must retain those notices.
- **Precision:** default compressed KV and indexer caches use FP8 and
  FP4 respectively. `DS4_CUDA_FP8_KV=0` and `DS4_CUDA_FP4_INDEX=0`
  restore F32 primary storage, retaining FP8/FP4-rounded values.
  jitLLM keeps its F16 caches; default ds4 timings are a same-weight
  comparator with the different transforms and storage disclosed.

## Cache-off is a storage control, not an unrounded quality oracle

- **Status:** source clarification; no upstream patch proposed.
- **What:** at the study pin, `fp8_kv_quantize_kernel` and
  `indexer_hadamard_fp4_kernel` in `ds4_cuda.cu` always write rounded,
  dequantized values into F32 tensors. Null packed-mirror pointers only
  disable the additional packed writes. The model graph still calls both
  transforms when `DS4_CUDA_FP8_KV=0 DS4_CUDA_FP4_INDEX=0`.
  `--quality` changes several product/attention kernels, not every fused
  path, and does not disable these transforms.
- **Interpretation:** the official DeepSeek inference code also simulates
  these quantization-aware-training transforms. A packed/F32-storage
  comparison alone neither measures their quality effect nor proves a
  long-window gate. The study's earlier higher/full-precision labels are
  corrected; same-GGUF oracle/perplexity qualification remains pending.
- **Proposed action:** document storage and value precision independently
  in future benchmark comparisons; retain the pinned original device
  objects in the external full-window scoring adapter.
