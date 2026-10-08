// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8's dense MXFP8 products on tensor cores (mxfp8_cutlass.cu):
// CUTLASS 4.7.1's SM120 block-scaled GEMM (BSD-3-Clause, the source lock's
// `cutlass`), built for sm_121a only, as the routed experts' grouped GEMM
// is (moe_cutlass.h). This header holds no CUDA or CUTLASS type: the
// llmp.mxfp8.gemm operation (llmp_ops.h) calls it with device
// addresses.
//
//   D[m, n] = A[m, k] · B[n, k]^T, where
//   - A is the activations quantized to MXFP8 (RowsLayout): m rows of k
//     E4M3 codes, each 32-element block's E8M0 scale in the swizzled layout
//     CUTLASS's SM1xx block-scaled kernels read (moe_cutlass.h SfOffset,
//     with k / 32 scales a row) over the rows rounded up to 128;
//   - B is the checkpoint's MXFP8 weights: n rows of k E4M3 codes as the
//     artifact holds them, and their scales swizzled the same way
//     (SwizzledScaleBytes, rows rounded up to 128);
//   - D is F32 or BF16, row r of D the products of row r of A, n values
//     apart; accumulation in F32.

#ifndef LLMP_KERNELS_GGML_MXFP8_CUTLASS_H_
#define LLMP_KERNELS_GGML_MXFP8_CUTLASS_H_

#include <cstddef>
#include <cstdint>

namespace llmp::kernels::ggml::mxfp8 {

// The values an E8M0 scale covers.
inline constexpr std::uint64_t kBlock = 32;
// The rows a scale atom holds; scales are padded to whole atoms.
inline constexpr std::uint64_t kAtomRows = 128;

constexpr std::uint64_t PaddedRows(std::uint64_t rows) {
  return (rows + kAtomRows - 1) / kAtomRows * kAtomRows;
}

// Activations quantized to MXFP8 (llmp.mxfp8.quantize's output), bytes:
// `rows` rows of k codes, then (256-byte aligned) their scales, swizzled,
// for PaddedRows(rows) rows; the padding rows' scales are zero.
struct RowsLayout {
  std::uint64_t k = 0;
  std::uint64_t rows = 0;
  static constexpr std::uint64_t codes() { return 0; }
  constexpr std::uint64_t scales() const { return ((rows * k) + 255) / 256 * 256; }
  constexpr std::uint64_t bytes() const { return scales() + (PaddedRows(rows) * (k / kBlock)); }
};

// A weight's scales swizzled for the product (llmp.mxfp8.swizzle's
// output): PaddedRows(n) rows of k / 32 scales, the padding rows zero.
constexpr std::uint64_t SwizzledScaleBytes(std::uint64_t n, std::uint64_t k) {
  return PaddedRows(n) * (k / kBlock);
}

struct Gemm {
  int m = 0;
  int n = 0;
  int k = 0;
  const void* a = nullptr;         // codes, m rows of k
  const void* a_scales = nullptr;  // swizzled
  const void* b = nullptr;         // codes, n rows of k
  const void* b_scales = nullptr;  // swizzled
  void* d = nullptr;               // m rows of n
  bool bf16 = false;               // D in BF16, else F32
  // Device scratch of Scratch(gemm) bytes, 256-byte aligned: CUTLASS's
  // workspace.
  void* scratch = nullptr;
};

// Whether the kernels are in this build (sm_121a code compiled).
bool Available();
// The scratch the product needs on a device of `sms` SMs.
std::size_t Scratch(const Gemm& gemm, int sms);
// Runs the product on `stream` (a cudaStream_t). Returns 0, or a nonzero
// CUTLASS status if CUTLASS refuses the problem.
int Run(const Gemm& gemm, int sms, void* stream);

}  // namespace llmp::kernels::ggml::mxfp8

#endif  // LLMP_KERNELS_GGML_MXFP8_CUTLASS_H_
