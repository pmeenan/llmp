// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// CUTLASS's block-scaled NVFP4 grouped GEMM for routed experts
// (moe_cutlass.cu; CUTLASS 4.7.1, BSD-3-Clause, the source lock's
// `cutlass`), built for sm_121a only. This header holds no CUDA or CUTLASS
// type: the llmp.moe.gemm operation (llmp_ops.h) calls it with device
// addresses.
//
// One group per expert e: D_e[M_e, n] = A_e[M_e, k] · B_e[n, k]^T, where
//   - A holds every routed row, sorted by expert: expert e's rows are rows
//     offsets[e] .. offsets[e + 1] of the codes (k / 2 bytes a row, element
//     2i in the low nibble), and its E4M3 scales (one per 16 elements) sit
//     in the swizzled layout (SfOffset) of a block of
//     sf_rows[e + 1] - sf_rows[e] rows (M_e rounded up to 128) starting at
//     row sf_rows[e] of the scales;
//   - B_e is expert e's weights at weights + e · expert_stride: n rows of
//     codes at codes_offset, their scales at scales_offset in the swizzled
//     layout of n rows;
//   - D is BF16, row r of D the product of row r of A; accumulation in F32.
// offsets and sf_rows are device arrays of groups + 1 entries.

#ifndef LLMP_KERNELS_GGML_MOE_CUTLASS_H_
#define LLMP_KERNELS_GGML_MOE_CUTLASS_H_

#include <cstddef>
#include <cstdint>

// Callable from device code where CUDA compiles this header.
#ifdef __CUDACC__
#define LLMP_MOE_HOST_DEVICE __host__ __device__
#else
#define LLMP_MOE_HOST_DEVICE
#endif

namespace llmp::kernels::ggml::moe {

// The byte offset of the E4M3 scale of row `row`, 16-element block `block`
// in the swizzled layout CUTLASS's SM1xx block-scaled kernels read (128-row
// by 4-scale atoms of 512 bytes, the atoms along k first; `blocks`, the
// scales a row, a multiple of 4).
LLMP_MOE_HOST_DEVICE constexpr std::uint64_t SfOffset(std::uint64_t row, std::uint64_t block,
                                                      std::uint64_t blocks) {
  return ((((row / 128) * (blocks / 4)) + (block / 4)) * 512) + ((row % 32) * 16) +
         (((row % 128) / 32) * 4) + (block % 4);
}

struct GroupedGemm {
  int groups = 0;
  int n = 0;
  int k = 0;
  const std::int32_t* offsets = nullptr;
  const std::int32_t* sf_rows = nullptr;
  const void* a = nullptr;
  const void* a_scales = nullptr;
  const void* weights = nullptr;
  std::uint64_t expert_stride = 0;
  std::uint64_t codes_offset = 0;
  std::uint64_t scales_offset = 0;
  void* d = nullptr;
  // Device scratch of GroupedGemmScratch(groups) bytes, 256-byte aligned:
  // the groups' arguments, then CUTLASS's workspace.
  void* scratch = nullptr;
};

// Whether the kernel is in this build (sm_121a code compiled).
bool GroupedGemmAvailable();
// The scratch the GEMM needs for `groups` groups on a device of `sms` SMs.
std::size_t GroupedGemmScratch(int groups, int sms);
// Builds the groups' arguments on the device from the offsets, then runs
// the grouped GEMM, both on `stream` (a cudaStream_t). Returns 0, or a
// nonzero CUTLASS status if CUTLASS refuses the problem.
int RunGroupedGemm(const GroupedGemm& gemm, int sms, void* stream);

}  // namespace llmp::kernels::ggml::moe

#endif  // LLMP_KERNELS_GGML_MOE_CUTLASS_H_
