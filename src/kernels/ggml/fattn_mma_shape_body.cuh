// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// The body of MmaShape (fattn_mma_shape.cuh), without an include guard or a
// namespace of its own: included once inside the namespace whose copy of
// GGML's MMA flash attention it measures, whose names (the configuration
// functions and flash_attn_ext_f16) it takes unqualified. fattn_mma_shape.cuh
// includes it in llmp::kernels::ggml::detail for GGML's kernel;
// fattn_mma_d512_q16.cu in llmp_fattn_q16 for the F16-Q copy's, whose
// occupancy may differ. The includer has included <algorithm>, <cstddef>,
// <expected>, <string> and fattn_mma.h.

// kGroup is ncols2: query heads per KV head in a tile (8 for the grouped
// kernels, 1 for multi-head attention).
template <int D, int kColumns, bool kSparse, int kGroup = 8, bool kSoftcap = false>
std::expected<::llmp::kernels::ggml::detail::MmaKernelShape, std::string> MmaShape(int device) {
  constexpr int ncols = kColumns * kGroup;
  const int cc = ggml_cuda_info().devices[device].cc;
  const int nthreads = ggml_cuda_fattn_mma_get_nthreads(D, D, ncols, cc);
  const int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(D, D, ncols, cc);
  const int nbatch_k2 = ggml_cuda_fattn_mma_get_nbatch_K2(D, D, ncols, cc);
  const int nbatch_v2 = ggml_cuda_fattn_mma_get_nbatch_V2(D, D, ncols, cc);
  const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(D, D, ncols, cc);
  const bool q_in_reg = ggml_cuda_fattn_mma_get_Q_in_reg(D, D, ncols, cc);
  const int nstages = ggml_cuda_fattn_mma_get_nstages(D, D, kColumns, kGroup, cc);
  const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
  const int warp_size = ggml_cuda_info().devices[device].warp_size;
  if (warp_size <= 0 || nthreads % warp_size != 0) {
    return std::unexpected(std::string("no MMA configuration for this device"));
  }
  const int nwarps = nthreads / warp_size;
  const bool swizzled = ggml_cuda_fattn_mma_get_swizzled(D, D, ncols, cc);
  const int stride_tile_k = swizzled ? nbatch_k2 : nbatch_k2 + 4;
  const int stride_tile_v = swizzled ? nbatch_v2 : nbatch_v2 + 4;
  const auto size = [](int value) { return static_cast<std::size_t>(value); };
  const std::size_t half2_size = sizeof(half2);
  const std::size_t kv_1stage =
      size(nbatch_fa) * size(std::max(stride_tile_k, stride_tile_v)) * half2_size;
  const std::size_t kv_2stage = size(nbatch_fa) * size(stride_tile_k + stride_tile_v) * half2_size;
  const std::size_t q_bytes = size(ncols) * size(D / 2 + 4) * half2_size;
  const std::size_t mask_bytes = size(kColumns) * size(nbatch_fa / 2 + 4) * half2_size;
  const std::size_t combine_bytes =
      size(nwarps) * size(cols_per_warp) * size(nbatch_combine + 4) * half2_size;
  const std::size_t kv_bytes = nstages <= 1 ? kv_1stage : kv_2stage;
  const std::size_t shared =
      std::max(combine_bytes, q_in_reg ? std::max(q_bytes, kv_bytes + mask_bytes)
                                       : q_bytes + kv_bytes + mask_bytes);
  const auto kernel = flash_attn_ext_f16<D, D, kColumns, kGroup, /*use_logit_softcap=*/kSoftcap,
                                         /*V_is_K_view=*/false, kSparse>;
  // As the case does before launch_fattn queries the occupancy.
  cudaError_t error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(shared));
  int per_sm = 0;
  if (error == cudaSuccess) {
    error =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, warp_size * nwarps, shared);
  }
  if (error != cudaSuccess || per_sm <= 0) {
    // Left pending, so the context's next run faults (launch.cu Begin).
    return std::unexpected(std::string("the MMA kernel's occupancy query failed: ") +
                           cudaGetErrorString(error));
  }
  return ::llmp::kernels::ggml::detail::MmaKernelShape{
      .kv_batch = nbatch_fa, .blocks_per_sm = per_sm, .async_kv_preload = nstages == 2 && !kSparse};
}
