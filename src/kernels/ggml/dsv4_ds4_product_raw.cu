// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 The ds4.c authors
// SPDX-FileCopyrightText: 2026 Entrpi <entrpi@proton.me> (batched-serving fork modifications)
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// This translation unit MUST resolve this header to the original Entrpi/ds4
// numerical closure, not jitLLM's different locked GGML. Root source preparation
// retains the authenticated original members at their exact include paths. Only the
// device templates are used; no original context, registry, pool, allocator or
// host dispatcher is instantiated. The native adapter supplies facts/scratch.
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kernels/ggml/dsv4_ds4_product_raw.h"

namespace jitllm::kernels::ggml::ds4_product::original {
// Pre-included standard/vendor dependencies stay outside this namespace.
// The named implementation namespace isolates original types and templates
// from native GGML while permitting the original extern template declarations.
#include "cuda/mmq/mmq.cuh"
#include "cuda/mmq/mmvq.cuh"
#include "cuda/mmq/quantize.cuh"
#include "cuda/mmq/unary.cuh"
#include "kernels/ggml/dsv4_ds4_product_f16.cuh"
#include "kernels/ggml/dsv4_ds4_product_original.cuh"
#include "kernels/ggml/dsv4_ds4_product_q8.cuh"
#include "kernels/ggml/dsv4_ds4_product_quantize.cuh"

template <int N, bool Small = false>
cudaError_t LaunchQ8Raw(const void* raw, const void* quantized, float* output, int m, int k,
                        cudaStream_t stream) {
  // Original dense dispatch at mmvq.cu:736-896, specialized to the
  // authenticated sm_121 Q8_0/no-ID/no-fusion path. No runtime query.
  constexpr auto type = GGML_TYPE_Q8_0;
  const auto dims = calc_launch_params<type>(N, m, 1, 1, 32, MMVQ_PARAMETERS_GENERIC, Small);
  const auto one = init_fastdiv_values(1);
  const auto col = ((k + 511LL) / 512) * 16;
  mul_mat_vec_q<type, N, false, Small><<<dims.first, dims.second, 0, stream>>>(
      raw, quantized, nullptr, {}, output, static_cast<std::uint32_t>(k), make_uint3(0, 0, 0),
      static_cast<std::uint32_t>(k / QK8_0), static_cast<std::uint32_t>(col),
      static_cast<std::uint32_t>(m), one, 0, static_cast<std::uint32_t>(N * col), 0, one, 0, 0, 0,
      0);
  return cudaGetLastError();
}

template <int N>
cudaError_t LaunchQ8Aligned(const void* scales, const void* codes, const void* quantized,
                            float* output, int m, int k, cudaStream_t stream) {
  q8_0_aligned_dense_vec_nc_kernel<N><<<static_cast<unsigned>(m), 32, 0, stream>>>(
      output, static_cast<const int4*>(codes), static_cast<const __half*>(scales),
      static_cast<const block_q8_1*>(quantized), m, k / QK8_0);
  return cudaGetLastError();
}

template <int N>
cudaError_t LaunchQ8(const void* raw, const void* scales, const void* codes, const void* quantized,
                     float* output, int m, int k, bool aligned, cudaStream_t stream) {
  if (aligned) {
    if constexpr (N == 1) {
      q8_0_aligned_dense_vec_kernel<<<static_cast<unsigned>(m), 32, 0, stream>>>(
          output, static_cast<const int4*>(codes), static_cast<const __half*>(scales),
          static_cast<const block_q8_1*>(quantized), m, k / QK8_0);
      return cudaGetLastError();
    }
    return LaunchQ8Aligned<N>(scales, codes, quantized, output, m, k, stream);
  }
  if constexpr (N == 1) {
    if (k < 1024) return LaunchQ8Raw<N, true>(raw, quantized, output, m, k, stream);
  }
  return LaunchQ8Raw<N>(raw, quantized, output, m, k, stream);
}

template <int X, bool Check>
cudaError_t LaunchMmq(const void* raw, const void* quantized, float* output, int m, int n, int k,
                      const jitllm::kernels::ggml::ds4_product::MmqPlan& p, float* fixup,
                      cudaStream_t stream) {
  constexpr auto type = GGML_TYPE_Q8_0;
  auto status = cudaFuncSetAttribute(mul_mat_q<type, X, Check>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, p.shared);
  if (status != cudaSuccess) return status;
  const auto one = init_fastdiv_values(1);
  const auto kb = init_fastdiv_values(static_cast<std::uint64_t>(k / QK8_0));
  const auto ntx = init_fastdiv_values(static_cast<std::uint64_t>(p.column_tiles));
  const auto sy = static_cast<int>(static_cast<std::int64_t>(n) * k * 36 /
                                   (32 * static_cast<std::int64_t>(sizeof(int))));
  const dim3 blocks(static_cast<unsigned>(p.blocks), 1, 1);
  const dim3 threads(32, static_cast<unsigned>(p.warps), 1);
  if (p.fixup != 0) {
    status = cudaMemsetAsync(fixup, 0, p.fixup, stream);
    if (status != cudaSuccess) return status;
  }
  // Descriptor checks bound positive dimensions, signed weight/output
  // indices and the complete Q8 payload before these interface casts.
  // Original ds4_mmq.cu:548-568 dense arguments and mmq.cuh:4259-4264
  // stream-K launch geometry, with the native borrowed fixup replacing pool.
  mul_mat_q<type, X, Check><<<blocks, threads, static_cast<std::size_t>(p.shared), stream>>>(
      static_cast<const char*>(raw), static_cast<const int*>(quantized), nullptr, nullptr, output,
      fixup, kb, m, n, k / QK8_0, n, m, one, one, 0, sy, 0, one, one, 0, sy, 0, ntx, nullptr, 0);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  if (p.fixup != 0) {
    const dim3 fixup_blocks(static_cast<unsigned>(p.blocks), static_cast<unsigned>(p.rows / 32), 1);
    const dim3 fixup_threads(32, static_cast<unsigned>(p.warps / 2), 1);
    mul_mat_q_stream_k_fixup<type, X, Check><<<fixup_blocks, fixup_threads, 0, stream>>>(
        nullptr, nullptr, output, fixup, kb, m, n, m, one, 0, one, 0, ntx);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
  }
  ds4_mmq_sanitize_f32_kernel<<<
      static_cast<unsigned>((static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(n) + 255) /
                            256),
      256, 0, stream>>>(output, static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(n));
  return cudaGetLastError();
}
template <int X>
cudaError_t LaunchMmq(const void* raw, const void* quantized, float* output, int m, int n, int k,
                      const jitllm::kernels::ggml::ds4_product::MmqPlan& p, float* fixup,
                      cudaStream_t stream) {
  if (m % p.rows == 0)
    return LaunchMmq<X, false>(raw, quantized, output, m, n, k, p, fixup, stream);
  return LaunchMmq<X, true>(raw, quantized, output, m, n, k, p, fixup, stream);
}
}  // namespace jitllm::kernels::ggml::ds4_product::original

namespace jitllm::kernels::ggml::ds4_product {

using namespace original;

cudaError_t Sanitize(float* output, std::uint64_t count, cudaStream_t stream) {
  ds4_mmq_sanitize_f32_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
      output, count);
  return cudaGetLastError();
}

cudaError_t Embedding(const std::int32_t* tokens, const void* weights, float* output,
                      std::uint32_t vocab, std::uint32_t rows, std::uint32_t width,
                      std::uint32_t hc, cudaStream_t stream) {
  const auto count = static_cast<std::uint64_t>(rows) * width * hc;
  embed_tokens_hc_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
      output, tokens, static_cast<const __half*>(weights), vocab, rows, width, hc);
  return cudaGetLastError();
}

cudaError_t F16Conversion(const float* input, void* output, std::uint64_t count,
                          cudaStream_t stream) {
  f32_to_f16_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
      static_cast<__half*>(output), input, count);
  return cudaGetLastError();
}

cudaError_t QkvNorm(const float* q, const float* qw, float* qo, std::uint32_t qn, const float* kv,
                    const float* kvw, float* kvo, std::uint32_t kvn, std::uint32_t rows,
                    float epsilon, cudaStream_t stream) {
  const dim3 grid(rows, 2, 1);
  dsv4_qkv_rms_norm_rows_kernel<<<grid, 256, 0, stream>>>(qo, q, qw, qn, kvo, kv, kvw, kvn, rows,
                                                          epsilon, nullptr);
  return cudaGetLastError();
}

cudaError_t Q81(const float* input, void* output, int rows, int columns, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || rows <= 0 || rows > 8 || columns <= 0 ||
      columns % 256 != 0)
    return cudaErrorInvalidValue;
  const auto padded = ((static_cast<std::int64_t>(columns) + 511) / 512) * 512;
  const dim3 grid(static_cast<unsigned>(padded / CUDA_QUANTIZE_BLOCK_SIZE),
                  static_cast<unsigned>(rows), 1);
  quantize_q8_1<<<grid, CUDA_QUANTIZE_BLOCK_SIZE, 0, stream>>>(
      input, output, columns, columns, static_cast<std::int64_t>(columns) * rows,
      static_cast<std::int64_t>(columns) * rows, padded, static_cast<std::uint32_t>(rows),
      init_fastdiv_values(1));
  return cudaGetLastError();
}

cudaError_t Q8Vector(const void* raw, const void* scales, const void* codes, const void* quantized,
                     float* output, int m, int n, int k, bool aligned, cudaStream_t stream) {
  if (quantized == nullptr || output == nullptr || m <= 0 || n <= 0 || n > 8 || k <= 0 ||
      k % 256 != 0 || (aligned && (scales == nullptr || codes == nullptr || k % 1024 != 0)) ||
      (!aligned && raw == nullptr))
    return cudaErrorInvalidValue;
  // Raw original dense wrapper clears output then sanitizes it. The aligned
  // original wrapper performs neither operation. Preserve that distinction.
  if (!aligned) {
    const auto status = cudaMemsetAsync(
        output, 0, static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(n) * 4, stream);
    if (status != cudaSuccess) return status;
  }
  cudaError_t status = cudaErrorInvalidValue;
  switch (n) {
    case 1:
      status = LaunchQ8<1>(raw, scales, codes, quantized, output, m, k, aligned, stream);
      break;
    case 2:
      status = LaunchQ8<2>(raw, scales, codes, quantized, output, m, k, aligned, stream);
      break;
    case 3:
      status = LaunchQ8<3>(raw, scales, codes, quantized, output, m, k, aligned, stream);
      break;
    case 4:
      status = LaunchQ8<4>(raw, scales, codes, quantized, output, m, k, aligned, stream);
      break;
    case 5:
      status = LaunchQ8<5>(raw, scales, codes, quantized, output, m, k, aligned, stream);
      break;
    case 6:
      status = LaunchQ8<6>(raw, scales, codes, quantized, output, m, k, aligned, stream);
      break;
    case 7:
      status = LaunchQ8<7>(raw, scales, codes, quantized, output, m, k, aligned, stream);
      break;
    case 8:
      status = LaunchQ8<8>(raw, scales, codes, quantized, output, m, k, aligned, stream);
      break;
  }
  if (status != cudaSuccess || aligned) return status;
  ds4_mmq_sanitize_f32_kernel<<<
      static_cast<unsigned>((static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(n) + 255) /
                            256),
      256, 0, stream>>>(output, static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(n));
  return cudaGetLastError();
}

cudaError_t F16Vector(const void* weights, const float* input, float* output, int m, int n, int k,
                      int split, void* partial, std::uint64_t partial_bytes, cudaStream_t stream) {
  if (weights == nullptr || input == nullptr || output == nullptr || m <= 0 || n <= 0 || n > 8 ||
      k <= 0 || split <= 0 || split > 2048 ||
      (split > 1 &&
       (partial == nullptr ||
        static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(split) * 4 > partial_bytes)))
    return cudaErrorInvalidValue;
  const auto* w = static_cast<const __half*>(weights);
  if (split == 1) {
    matmul_f16_kernel<<<dim3(static_cast<unsigned>(m), static_cast<unsigned>(n), 1), 256, 0,
                        stream>>>(output, w, input, static_cast<std::uint64_t>(k),
                                  static_cast<std::uint64_t>(m), static_cast<std::uint64_t>(n));
    return cudaGetLastError();
  }
  const bool vec = k % 8 == 0 && reinterpret_cast<std::uintptr_t>(weights) % 16 == 0 &&
                   reinterpret_cast<std::uintptr_t>(input) % 16 == 0;
  // Original default loops each token through the same one-row partials,
  // with the ascending combine completing before the next token writes.
  for (int row = 0; row < n; ++row) {
    matmul_f16_splitk_kernel<<<dim3(static_cast<unsigned>(m), static_cast<unsigned>(split), 1), 256,
                               0, stream>>>(
        static_cast<float*>(partial), w,
        input + static_cast<std::uint64_t>(row) * static_cast<std::uint64_t>(k),
        static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(m),
        static_cast<std::uint32_t>(split), vec ? 1 : 0);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    matmul_f16_splitk_combine_kernel<<<(static_cast<unsigned>(m) + 255) / 256, 256, 0, stream>>>(
        output + static_cast<std::uint64_t>(row) * static_cast<std::uint64_t>(m),
        static_cast<const float*>(partial), static_cast<std::uint64_t>(m),
        static_cast<std::uint32_t>(split));
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
  }
  return cudaSuccess;
}

bool PlanMmq(Device d, int m, int n, int k, MmqPlan& p) {
  if (d.cc != 1210 || d.multiprocessors <= 0 || d.warp != 32 || m <= 0 || n <= 8 || n > 65535 ||
      k <= 0 || k % 512 != 0)
    return false;
  const int warps = mmq_get_nwarps_host(d.cc, d.warp);
  const int y = get_mmq_y_host(d.cc);
  // Original default on the qualified sm_121 is 128. Do not call the
  // upstream host helper's process-global cached environment override.
  constexpr int maximum = 128;
  int best = 0, best_tiles = INT_MAX;
  // Original mmq.cuh:4307-4323 selection, using the caller's device facts.
  for (int x = 8; x <= maximum && best_tiles > 1; x += 8) {
    const int granularity = mmq_get_granularity_host(x, d.cc);
    if (x % granularity != 0 ||
        mmq_get_nbytes_shared<GGML_TYPE_Q8_0>(x, y, d.cc, d.warp, warps) > d.shared_optin)
      continue;
    const int tiles = (n + x - 1) / x;
    if (tiles < best_tiles) {
      best = x;
      best_tiles = tiles;
    }
  }
  if (best == 0 || y <= 0 || warps <= 0 || warps % 2 != 0) return false;
  const auto nty = (static_cast<std::int64_t>(m) + y - 1) / y;
  const auto tiles = nty * best_tiles;
  if (tiles * (k / QK8_0) >= (1LL << 30)) return false;
  const auto waves = (tiles + d.multiprocessors - 1) / d.multiprocessors;
  const auto efficiency = 100 * tiles / (d.multiprocessors * waves);
  const auto blocks = efficiency >= 90 ? tiles : d.multiprocessors;
  if (blocks <= 0 || blocks > INT_MAX) return false;
  p = {.columns = best,
       .rows = y,
       .warps = warps,
       .blocks = static_cast<int>(blocks),
       .column_tiles = best_tiles,
       .shared =
           static_cast<int>(mmq_get_nbytes_shared<GGML_TYPE_Q8_0>(best, y, d.cc, d.warp, warps)),
       .fixup = tiles % blocks != 0
                    ? static_cast<std::uint64_t>(blocks) * static_cast<std::uint64_t>(best) *
                          static_cast<std::uint64_t>(y) * 4
                    : 0};
  return true;
}

cudaError_t Mmq(const void* raw, const void* quantized, float* output, int m, int n, int k,
                const MmqPlan& p, void* fixup, std::uint64_t fixup_bytes, cudaStream_t stream) {
  if (raw == nullptr || quantized == nullptr || output == nullptr || p.columns <= 0 ||
      p.rows <= 0 || p.warps <= 0 || p.blocks <= 0 ||
      p.column_tiles != (n + p.columns - 1) / p.columns || p.fixup > fixup_bytes ||
      (p.fixup != 0 && fixup == nullptr))
    return cudaErrorInvalidValue;
  switch (p.columns) {
#define JITLLM_DS4_COLUMN(X) \
  case X:                    \
    return LaunchMmq<X>(raw, quantized, output, m, n, k, p, static_cast<float*>(fixup), stream)
    JITLLM_DS4_COLUMN(8);
    JITLLM_DS4_COLUMN(16);
    JITLLM_DS4_COLUMN(24);
    JITLLM_DS4_COLUMN(32);
    JITLLM_DS4_COLUMN(40);
    JITLLM_DS4_COLUMN(48);
    JITLLM_DS4_COLUMN(56);
    JITLLM_DS4_COLUMN(64);
    JITLLM_DS4_COLUMN(72);
    JITLLM_DS4_COLUMN(80);
    JITLLM_DS4_COLUMN(88);
    JITLLM_DS4_COLUMN(96);
    JITLLM_DS4_COLUMN(104);
    JITLLM_DS4_COLUMN(112);
    JITLLM_DS4_COLUMN(120);
    JITLLM_DS4_COLUMN(128);
#undef JITLLM_DS4_COLUMN
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t D4(const float* input, void* output, int rows, int columns, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || rows <= 0 || rows > 65535 || columns <= 0 ||
      columns % 512 != 0 || columns / 512 > 65535)
    return cudaErrorInvalidValue;
  const dim3 blocks(static_cast<unsigned>(rows), static_cast<unsigned>((columns + 511) / 512), 1);
  quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4><<<blocks, 128, 0, stream>>>(
      input, nullptr, output, columns, columns, 0, 0, columns, rows, 1);
  return cudaGetLastError();
}

std::uint64_t DenseD2rSharedBytes() { return dq8::kDqSmemTotalBytes; }

cudaError_t DenseD2r(const void* scales, const void* codes, const void* quantized, float* output,
                     int m, int n, int k, cudaStream_t stream) {
  using namespace dq8;
  auto status =
      cudaFuncSetAttribute(dense_q8_d2r_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(kDqSmemTotalBytes));
  if (status != cudaSuccess) return status;
  const dim3 grid(static_cast<unsigned>((m / kDqMTile) * ((n + kDqNTile - 1) / kDqNTile)), 1, 1);
  const dim3 block(32, kDqWarps, 1);
  dense_q8_d2r_kernel<<<grid, block, kDqSmemTotalBytes, stream>>>(
      static_cast<const char*>(scales), static_cast<const char*>(codes),
      static_cast<const block_q8_1_mmq*>(quantized), output, m, n, k, k <= 2048 ? 1 : 4);
  return cudaGetLastError();
}

cudaError_t HeadRope(float* input, std::uint32_t rows, std::uint32_t heads, std::uint32_t width,
                     Rope r, bool normalize, float epsilon, cudaStream_t stream) {
  if (normalize) {
    head_rms_norm_rope_tail_kernel<<<rows * heads, 256, 0, stream>>>(
        input, rows, heads, width, r.rotary, r.first, r.positions, r.step, r.original_context,
        r.inverse ? 1 : 0, r.base, r.scale, r.extension, r.attention, r.beta_fast, r.beta_slow,
        epsilon);
  } else {
    const auto pairs = rows * heads * (r.rotary / 2);
    rope_tail_kernel<<<(pairs + 255) / 256, 256, 0, stream>>>(
        input, rows, heads, width, r.rotary, r.first, r.positions, r.step, r.original_context,
        r.inverse ? 1 : 0, r.base, r.scale, r.extension, r.attention, r.beta_fast, r.beta_slow);
  }
  return cudaGetLastError();
}

cudaError_t OutA(const void* scales, const void* codes, const float* heads, float* low, void* table,
                 void* quantized, std::uint32_t rows, Rope r, cudaStream_t stream) {
  const auto table_rows = ((static_cast<std::uint64_t>(rows) + 127) / 128) * 128;
  // The original core reads table entries for its dummy CTA rows and stores
  // whole WMMA rows. Explicit physical padding closes those source bounds.
  auto status = cudaMemsetAsync(table, 0, table_rows * 32 * sizeof(float2), stream);
  if (status != cudaSuccess) return status;
  attention_outa_rope_cs_table_kernel<<<(rows * 32 + 255) / 256, 256, 0, stream>>>(
      static_cast<float2*>(table), rows, r.first, r.positions, r.original_context, r.base, r.scale,
      r.extension, r.attention, r.beta_fast, r.beta_slow);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  const dim3 grid(kOARank / kOATileN, (rows + kOATileM - 1) / kOATileM, kOAGroups);
  constexpr auto shared = (2 * kOATileM + 2 * kOATileN) * (kOATileK + kOAPad) * sizeof(__half);
  attention_outa_fused_own_kernel<<<grid, kOAWarps * 32, shared, stream>>>(
      low, heads, static_cast<const float2*>(table), static_cast<const __half*>(scales),
      static_cast<const std::int8_t*>(codes), static_cast<char*>(quantized), rows);
  return cudaGetLastError();
}
}  // namespace jitllm::kernels::ggml::ds4_product
