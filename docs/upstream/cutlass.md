<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CUTLASS

- **Repository:** [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass),
  BSD-3-Clause.
- **Llmpalooza's pin:** v4.7.1, commit `cb4247394dd82148787aed73e5dc7cef33cbf862`
  ([sources.lock.json](../../third_party/sources.lock.json)), headers only.
  Llmpalooza's own `sm_121a` units include them: Qwen3.8's routed experts over
  the NVFP4 grouped GEMM (`src/kernels/ggml/moe_cutlass.cu`) and the MXFP8
  products (`src/kernels/ggml/mxfp8_cutlass.cu`).
- **Last upstream check:** 2026-09-29. v4.8.0 (2026-09-22) mostly adds Rubin
  (SM107). No dense SM120 block-scaled or grouped-GEMM header changed between
  4.7.1 and 4.8.0, so no bump is needed.

## The SM120 block-scaled GEMM's default rasterization halves speed on wide products at 8,192 rows (RE-034)

- **Status:** open (a limitation to track).
- **Found:** 2026-09-28, v4.7.1, `spark-b` (GB10).
- **Problem:** the SM120 MXFP8 × MXFP8 GEMM (128 × 128 × 128 tiles,
  ping-pong or cooperative, the persistent CLC tile scheduler with its
  default rasterization) runs 1.9–2.5× cuBLAS's BF16 product at 4,096 rows,
  but at 8,192 rows the wide products fall behind it. N 10,240, K 2,560 takes
  4.66 ms with BF16 output and 7.5 ms with F32, against cuBLAS's 5.3 ms;
  N 12,288 takes 5.6 and 9.0 against 6.3. The narrow products (N 2,560 or
  6,144) stay ahead. Setting the scheduler's `max_swizzle_size` to 8 with
  raster along N gives 2.59 and 3.09 ms (F32 output 3.3 and 3.9). At 4,096
  rows the swizzle is slightly slower (1.31 against 1.18 ms for N 10,240).
  It fits an L2 working-set cliff.
  Repro: the kernel at M 8,192, N 10,240, K 2,560, with and without the
  swizzle.
- **Llmpalooza's workaround:** `src/kernels/ggml/mxfp8_cutlass.cu` swizzles past
  4,096 rows. No cost.
- **Upstream master:** not checked beyond the release notes above.
- **Upstream refs:** none found.
- **Proposed action:** optional: an issue with the A/B table, suggesting
  the SM120 persistent scheduler's heuristic consider swizzling for large
  M × N. Low priority.
- **Links:** RE-034 in [rough-edges.md](../rough-edges.md);
  [qwen38-native](../experiments/qwen38-native/README.md#prefill-second-pass-speed-before-bit-exactness).

## Llmpalooza's build glue (patch 0001)

- **Status:** carry.
- **What:** `third_party/patches/cutlass/0001-llmp-build.patch` adds
  `llmp/CMakeLists.txt`, an interface target over the kept header trees.
  It compiles nothing and changes no CUTLASS file.
- **Proposed action:** none.
