// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 The ds4.c authors
// SPDX-FileCopyrightText: 2026 Entrpi <entrpi@proton.me> (batched-serving fork modifications)
// SPDX-FileCopyrightText: 2026 Marco Palaferri
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include <cooperative_groups.h>
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
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "kernels/ggml/dsv4_ds4_moe_raw.h"

namespace jitllm::kernels::ggml::ds4_moe::original {
// ROOT must be the authenticated original Entrpi/ds4 header component.
// This private closure is separate from native GGML; no original host
// context, stream, allocator, symbol registry or graph runtime is used.
// A named implementation namespace permits original extern template
// declarations while isolating their types from native GGML.
#include "cuda/mmq/mmq.cuh"
#include "kernels/ggml/dsv4_ds4_moe_core.cuh"
#include "kernels/ggml/dsv4_ds4_moe_original.cuh"

using Device = jitllm::kernels::ggml::ds4_moe::Device;
using MmqPlan = jitllm::kernels::ggml::ds4_moe::MmqPlan;

#define DS4_TRY(EXPR)                                 \
  do {                                                \
    const auto ds4_status = (EXPR);                   \
    if (ds4_status != cudaSuccess) return ds4_status; \
  } while (false)

__global__ void PackIds(const std::int32_t* source, std::int32_t* out, int rows, int stride) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < rows * 6) out[i] = source[(i / 6) * stride + i % 6];
}
__global__ void PackWeights(const float* source, float* out, int rows, int stride) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < rows * 6) out[i] = source[(i / 6) * stride + i % 6];
}
cudaError_t Q81(const float* input, void* quantized, int rows, int k, cudaStream_t stream) {
  quantize_q8_1<<<dim3(static_cast<unsigned>((k + 255) / 256), 1, static_cast<unsigned>(rows)), 256,
                  0, stream>>>(input, quantized, k, k, k, static_cast<std::int64_t>(k) * rows, k, 1,
                               init_fastdiv_values(static_cast<std::uint64_t>(rows)));
  return cudaGetLastError();
}
template <mmq_q8_1_ds_layout Layout>
cudaError_t Quant(const float* input, const std::int32_t* map, void* quantized, int rows, int k,
                  cudaStream_t stream) {
  quantize_mmq_q8_1<Layout>
      <<<dim3(static_cast<unsigned>(rows), static_cast<unsigned>((k + 511) / 512), 1), 128, 0,
         stream>>>(input, map, quantized, k, k, k, static_cast<std::int64_t>(k) * rows, k, rows, 1);
  return cudaGetLastError();
}
template <ggml_type Type>
bool PlanMmq(Device d, int m, int n, int k, MmqPlan& p) {
  if (d.cc != 1210 || d.sm <= 0 || d.warp != 32 || m <= 0 || n <= 0 || n > 4096) return false;
  const int warps = mmq_get_nwarps_host(d.cc, d.warp);
  const int y = get_mmq_y_host(d.cc);
  int best = 0, tiles_x = INT_MAX;
  for (int x = 8; x <= 128 && tiles_x > 1; x += 8) {
    const int g = mmq_get_granularity_host(x, d.cc);
    if (x % g != 0 || mmq_get_nbytes_shared<Type>(x, y, d.cc, d.warp, warps) > d.shared) continue;
    const int tiles = (n + x - 1) / x;
    if (tiles < tiles_x) {
      best = x;
      tiles_x = tiles;
    }
  }
  if (best == 0 || y <= 0 || warps <= 0 || warps % 2 != 0) return false;
  const auto tiles = static_cast<std::int64_t>((m + y - 1) / y) * tiles_x * 256;
  if (tiles * (k / 256) >= (1LL << 30)) return false;
  const auto waves = (tiles + d.sm - 1) / d.sm;
  const auto efficiency = 100 * tiles / (d.sm * waves);
  const auto blocks = efficiency >= 90 ? tiles : d.sm;
  if (blocks <= 0 || blocks > INT_MAX) return false;
  p = {.x = best,
       .y = y,
       .warps = warps,
       .blocks = static_cast<int>(blocks),
       .column_tiles = tiles_x,
       .shared = static_cast<int>(mmq_get_nbytes_shared<Type>(best, y, d.cc, d.warp, warps)),
       .fixup = tiles % blocks != 0
                    ? static_cast<std::uint64_t>(blocks) * static_cast<std::uint64_t>(best) *
                          static_cast<std::uint64_t>(y) * 4
                    : 0};
  return true;
}
template <ggml_type Type, int X, bool Check>
cudaError_t Classic(const void* weight, const void* q, const std::int32_t* destination,
                    const std::int32_t* bounds, float* output, int m, int rows, int k,
                    const MmqPlan& p, float* fixup, cudaStream_t stream) {
  DS4_TRY(cudaFuncSetAttribute(mul_mat_q<Type, X, Check>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, p.shared));
  const int pairs = rows * 6;
  const auto one = init_fastdiv_values(1);
  const auto experts = init_fastdiv_values(256);
  const auto kb = init_fastdiv_values(static_cast<std::uint64_t>(k / 256));
  const auto ntx = init_fastdiv_values(static_cast<std::uint64_t>(p.column_tiles));
  // Checked shapes cap k at 4096 and rows at 4096; pairs * sy fits int.
  const auto sy = static_cast<int>(static_cast<std::int64_t>(k) * 36 /
                                   (32 * static_cast<std::int64_t>(sizeof(int))));
  const auto blocks =
      static_cast<std::int64_t>(256) * m * (k / 256) / (Type == GGML_TYPE_Q2_K ? 2 : 1);
  if (p.fixup != 0) DS4_TRY(cudaMemsetAsync(fixup, 0, p.fixup, stream));
  // Original routed stream-K invocation, original SoA loader selected.
  // The original x raw address is ignored when x_soa is supplied.
  mul_mat_q<Type, X, Check>
      <<<static_cast<unsigned>(p.blocks), dim3(32, static_cast<unsigned>(p.warps), 1),
         static_cast<std::size_t>(p.shared), stream>>>(
          static_cast<const char*>(weight), static_cast<const int*>(q), destination, bounds, output,
          p.fixup != 0 ? fixup : nullptr, kb, m, pairs, k / 256, pairs, m, one, experts,
          m * (k / 256), sy, 0, one, one, 0, pairs * sy, 0, ntx, static_cast<const char*>(weight),
          blocks);
  DS4_TRY(cudaGetLastError());
  if (p.fixup != 0) {
    mul_mat_q_stream_k_fixup<Type, X, Check>
        <<<dim3(static_cast<unsigned>(p.blocks), static_cast<unsigned>(p.y / 32), 1),
           dim3(32, static_cast<unsigned>(p.warps / 2), 1), 0, stream>>>(
            destination, bounds, output, fixup, kb, m, pairs, m, experts, 0, one, 0, ntx);
    DS4_TRY(cudaGetLastError());
  }
  return cudaSuccess;
}
template <ggml_type Type, int X>
cudaError_t Classic(const void* w, const void* q, const std::int32_t* dst,
                    const std::int32_t* bounds, float* out, int m, int rows, int k,
                    const MmqPlan& p, float* fixup, cudaStream_t stream) {
  if (m % p.y == 0)
    return Classic<Type, X, false>(w, q, dst, bounds, out, m, rows, k, p, fixup, stream);
  return Classic<Type, X, true>(w, q, dst, bounds, out, m, rows, k, p, fixup, stream);
}
template <ggml_type Type>
cudaError_t Classic(const void* w, const void* q, const std::int32_t* dst,
                    const std::int32_t* bounds, float* out, int m, int rows, int k,
                    const MmqPlan& p, float* fixup, cudaStream_t stream) {
  switch (p.x) {
#define DS4_MOE_X(X) \
  case X:            \
    return Classic<Type, X>(w, q, dst, bounds, out, m, rows, k, p, fixup, stream)
    DS4_MOE_X(8);
    DS4_MOE_X(16);
    DS4_MOE_X(24);
    DS4_MOE_X(32);
    DS4_MOE_X(40);
    DS4_MOE_X(48);
    DS4_MOE_X(56);
    DS4_MOE_X(64);
    DS4_MOE_X(72);
    DS4_MOE_X(80);
    DS4_MOE_X(88);
    DS4_MOE_X(96);
    DS4_MOE_X(104);
    DS4_MOE_X(112);
    DS4_MOE_X(120);
    DS4_MOE_X(128);
#undef DS4_MOE_X
    default:
      return cudaErrorInvalidValue;
  }
}
cudaError_t Vector(const jitllm::kernels::ggml::ds4_moe::Call& c, const std::int32_t* ids,
                   const float* weights, cudaStream_t stream) {
  if (!c.produced) DS4_TRY(Q81(c.input, c.input_quant, c.rows, c.input_width, stream));
  const auto blocks = static_cast<std::uint64_t>(256) * static_cast<std::uint64_t>(c.middle_width) *
                      static_cast<std::uint64_t>(c.input_width / 256);
  const auto scales = (blocks * 2 + 63) & ~std::uint64_t{63};
  const auto* gate_scales = static_cast<const __half*>(c.gate_weights);
  const auto* up_scales = static_cast<const __half*>(c.up_weights);
  const auto* gate_codes =
      reinterpret_cast<const uint2*>(static_cast<const char*>(c.gate_weights) + scales);
  const auto* up_codes =
      reinterpret_cast<const uint2*>(static_cast<const char*>(c.up_weights) + scales);
  const dim3 grid(static_cast<unsigned>(c.middle_width), static_cast<unsigned>(c.rows * 6), 1);
  if (c.rows >= 2 && c.rows <= 8) {
    switch (c.rows) {
#define DS4_MOE_DEDUP(N)                                                             \
  case N:                                                                            \
    iq2_xxs_aligned_moe_gate_up_mid_dedup_kernel<N><<<grid, 32, 0, stream>>>(        \
        c.middle, gate_codes, gate_scales, up_codes, up_scales,                      \
        static_cast<const block_q8_1*>(c.input_quant), ids, weights, c.middle_width, \
        c.input_width / 256, c.input_width / 32, 6, c.rows * 6, 10.0F);              \
    break
      DS4_MOE_DEDUP(2);
      DS4_MOE_DEDUP(3);
      DS4_MOE_DEDUP(4);
      DS4_MOE_DEDUP(5);
      DS4_MOE_DEDUP(6);
      DS4_MOE_DEDUP(7);
      DS4_MOE_DEDUP(8);
#undef DS4_MOE_DEDUP
    }
  } else {
    iq2_xxs_aligned_moe_gate_up_mid_kernel<<<grid, 32, 0, stream>>>(
        c.middle, gate_codes, gate_scales, up_codes, up_scales,
        static_cast<const block_q8_1*>(c.input_quant), ids, weights, c.middle_width,
        c.input_width / 256, c.input_width / 32, 6, 10.0F);
  }
  DS4_TRY(cudaGetLastError());
  DS4_TRY(Q81(c.middle, c.down_quant, c.rows * 6, c.middle_width, stream));
  const auto pair_blocks = static_cast<std::uint64_t>(256) *
                           static_cast<std::uint64_t>(c.output_width / 2) *
                           static_cast<std::uint64_t>(c.middle_width / 256);
  const auto dm = (pair_blocks * 8 + 63) & ~std::uint64_t{63};
  const auto sc = (pair_blocks * 32 + 63) & ~std::uint64_t{63};
  // Original aligned Q2 down clears output unconditionally; distinct
  // from wide path's default-OFF blanket output reset.
  DS4_TRY(cudaMemsetAsync(
      c.down, 0,
      static_cast<std::uint64_t>(c.rows) * 6 * static_cast<std::uint64_t>(c.output_width) * 4,
      stream));
  for (int first = 0; first < c.rows * 6; first += 8) {
    const int cols = std::min(8, c.rows * 6 - first);
    q2_k_aligned_moe_vec_kernel<<<dim3(static_cast<unsigned>(c.output_width / 2), 1, 1),
                                  dim3(32, static_cast<unsigned>(cols), 1), 0, stream>>>(
        static_cast<const uint2*>(c.down_weights),
        reinterpret_cast<const int4*>(static_cast<const char*>(c.down_weights) + dm),
        reinterpret_cast<const uint2*>(static_cast<const char*>(c.down_weights) + dm + sc),
        static_cast<const block_q8_1*>(c.down_quant) +
            static_cast<std::uint64_t>(first) * static_cast<std::uint64_t>(c.middle_width / 32),
        ids + first,
        c.down + static_cast<std::uint64_t>(first) * static_cast<std::uint64_t>(c.output_width),
        static_cast<std::uint32_t>(c.middle_width), static_cast<std::uint32_t>(c.output_width),
        static_cast<std::uint32_t>(c.middle_width / 32), static_cast<std::uint32_t>(c.output_width),
        static_cast<std::uint32_t>(cols));
    DS4_TRY(cudaGetLastError());
  }
  // The caller launches the literal original sanitize pass before any
  // optional six-slot sum, preserving continuation slot-buffer bytes.
  return cudaSuccess;
}
}  // namespace jitllm::kernels::ggml::ds4_moe::original

namespace jitllm::kernels::ggml::ds4_moe {
using namespace original;

bool PlanClassic(Device d, int rows, int input, int middle, int output, Plan& p) {
  if (!PlanMmq<GGML_TYPE_IQ2_XXS>(d, middle, rows, input, p.gate) ||
      !PlanMmq<GGML_TYPE_Q2_K>(d, output, rows, middle, p.down))
    return false;
  p.fixup = std::max(p.gate.fixup, p.down.fixup);
  return true;
}
bool CooperativeFits(int sm, int rows) {
  if (sm <= 0 || rows < 1 || rows > 8) return false;
  int dev = 0, enabled = 0, blocks = 0;
  if (cudaGetDevice(&dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&enabled, cudaDevAttrCooperativeLaunch, dev) != cudaSuccess ||
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, router_fused_rows_coop_kernel, 256,
                                                    0) != cudaSuccess)
    return false;
  return enabled != 0 && static_cast<std::uint64_t>(blocks) * static_cast<std::uint64_t>(sm) >= 48;
}
cudaError_t Select(const Router& r, cudaStream_t stream) {
  const int hb = r.bias != nullptr, hm = r.hash != nullptr;
  if (r.select == 0) {
    router_select_warp_topk_kernel<<<static_cast<unsigned>((r.rows + 3) / 4), dim3(32, 4, 1), 0,
                                     stream>>>(r.selected, r.weights, r.probabilities, r.bias,
                                               r.hash, r.logits, r.tokens, 0, r.hash_rows,
                                               static_cast<std::uint32_t>(r.rows), hb, hm, nullptr);
  } else if (r.select == 1) {
    router_select_parallel_kernel<<<static_cast<unsigned>(r.rows), 256, 0, stream>>>(
        r.selected, r.weights, r.probabilities, r.bias, r.hash, r.logits, r.tokens, 0, r.hash_rows,
        static_cast<std::uint32_t>(r.rows), hb, hm, nullptr);
  } else {
    router_select_kernel<<<static_cast<unsigned>(r.rows), 1, 0, stream>>>(
        r.selected, r.weights, r.probabilities, r.bias, r.hash, r.logits, r.tokens, 0, r.hash_rows,
        static_cast<std::uint32_t>(r.rows), hb, hm, nullptr);
  }
  return cudaGetLastError();
}
cudaError_t Cooperative(const Router& r, const float* input, const void* weights, float* partials,
                        cudaStream_t stream) {
  int dev = 0, sm = 0, active = 0;
  DS4_TRY(cudaGetDevice(&dev));
  DS4_TRY(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev));
  DS4_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, router_fused_rows_coop_kernel, 256,
                                                        0));
  const int grid = std::min(128, sm * active);
  if (grid < 48) return cudaErrorCooperativeLaunchTooLarge;
  auto logits = r.logits;
  auto selected = r.selected;
  auto route = r.weights;
  auto probabilities = r.probabilities;
  auto w = static_cast<const __half*>(weights);
  auto bias = r.bias;
  auto hash = r.hash;
  auto tokens = r.tokens;
  auto hash_rows = r.hash_rows;
  auto rows = static_cast<std::uint32_t>(r.rows);
  int hb = bias != nullptr, hm = hash != nullptr;
  void* args[] = {&logits, &partials, &selected, &route,     &probabilities, &w,  &input,
                  &bias,   &hash,     &tokens,   &hash_rows, &rows,          &hb, &hm};
  return cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(router_fused_rows_coop_kernel),
                                     static_cast<unsigned>(grid), 256, args, 0, stream);
}
cudaError_t Shared(const float* g, const float* u, float* out, int rows, int width,
                   cudaStream_t stream) {
  const auto n = static_cast<std::uint32_t>(rows * width);
  swiglu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(out, g, u, n, 10.0F, 1.0F);
  return cudaGetLastError();
}
cudaError_t Sum(const float* down, float* out, int rows, int width, cudaStream_t stream) {
  const auto n = static_cast<std::uint32_t>(rows * width);
  moe_sum_kernel<<<(n + 255) / 256, 256, 0, stream>>>(out, down, static_cast<std::uint32_t>(width),
                                                      6, static_cast<std::uint32_t>(rows), 1);
  return cudaGetLastError();
}
cudaError_t Moe(Device device, const Call& c, const Plan& p, void* fixup, cudaStream_t stream) {
  const std::int32_t* ids = c.selected;
  const float* weights = c.weights;
  if (c.selected_stride != 6) {
    PackIds<<<static_cast<unsigned>((c.rows * 6 + 255) / 256), 256, 0, stream>>>(
        ids, c.compact_ids, c.rows, c.selected_stride);
    DS4_TRY(cudaGetLastError());
    ids = c.compact_ids;
  }
  if (c.weight_stride != 6) {
    PackWeights<<<static_cast<unsigned>((c.rows * 6 + 255) / 256), 256, 0, stream>>>(
        weights, c.compact_weights, c.rows, c.weight_stride);
    DS4_TRY(cudaGetLastError());
    weights = c.compact_weights;
  }
  const int pairs = c.rows * 6;
  if (c.tier == 0) {
    DS4_TRY(Vector(c, ids, weights, stream));
    ds4_mmq_sanitize_f32_kernel<<<
        static_cast<unsigned>(
            (static_cast<std::uint64_t>(pairs) * static_cast<std::uint64_t>(c.output_width) + 255) /
            256),
        256, 0, stream>>>(
        c.down, static_cast<std::uint64_t>(pairs) * static_cast<std::uint64_t>(c.output_width));
    DS4_TRY(cudaGetLastError());
  } else {
    DS4_TRY(cudaMemsetAsync(c.ids_source, 0, static_cast<std::uint64_t>(pairs) * 4, stream));
    DS4_TRY(cudaMemsetAsync(c.ids_destination, 0, static_cast<std::uint64_t>(pairs) * 4, stream));
    const auto shared = static_cast<std::uint64_t>(c.rows) * sizeof(mm_ids_helper_store);
    if (shared <= device.shared) {
      DS4_TRY(cudaFuncSetAttribute(mm_ids_helper<6>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(device.shared)));
      mm_ids_helper<6><<<256, 32, shared, stream>>>(ids, c.ids_source, c.ids_destination, c.bounds,
                                                    c.rows, 6, 1, 6, 1);
    } else {
      mm_ids_helper_global<6><<<256, 32, 0, stream>>>(ids, c.ids_source, c.ids_destination,
                                                      c.bounds, c.rows, 6, 1, 6, 1);
    }
    DS4_TRY(cudaGetLastError());
    const int input_rows = c.tier == 1 ? c.rows : pairs;
    const auto input_payload = static_cast<std::uint64_t>(input_rows) *
                               static_cast<std::uint64_t>(c.input_width / 128) * 144;
    const auto down_payload =
        static_cast<std::uint64_t>(pairs) * static_cast<std::uint64_t>(c.middle_width / 128) * 144;
    // Explicit charged safety guard; original YBUF blanket memset is OFF.
    if (!c.produced)
      DS4_TRY(
          cudaMemsetAsync(static_cast<char*>(c.input_quant) + input_payload, 0, 128 * 144, stream));
    DS4_TRY(cudaMemsetAsync(static_cast<char*>(c.down_quant) + down_payload, 0, 128 * 144, stream));
    if (!c.produced)
      DS4_TRY(Quant<MMQ_Q8_1_DS_LAYOUT_D4>(c.input, c.tier == 1 ? nullptr : c.ids_source,
                                           c.input_quant, input_rows, c.input_width, stream));
    if (c.tier == 1 || c.tier == 2) {
      const int capacity = (pairs + (c.tier == 1 ? 31 : 63)) / (c.tier == 1 ? 32 : 64) + 256;
      int* count = c.work + capacity;
      if (c.tier == 1) {
        d2r_build_worklist_kernel<32><<<1, 256, 0, stream>>>(c.bounds, c.work, count, 256);
        DS4_TRY(cudaGetLastError());
        gateup_iq2_swiglu_q8_d2r_kernel<32, 32><<<dim3(static_cast<unsigned>(c.middle_width / 128),
                                                       static_cast<unsigned>(capacity), 1),
                                                  dim3(32, 8, 1), 0, stream>>>(
            c.gate_weights, c.up_weights, static_cast<const block_q8_1_mmq*>(c.input_quant),
            c.ids_source, c.ids_destination, c.bounds, weights, c.work, count,
            static_cast<block_q8_1_mmq*>(c.down_quant), c.middle_width, c.input_width, pairs,
            c.rows, 256, 10.0F);
      } else {
        d2r_build_worklist_kernel<64><<<1, 256, 0, stream>>>(c.bounds, c.work, count, 256);
        DS4_TRY(cudaGetLastError());
        gateup_iq2_d2r_pair_kernel<<<dim3(static_cast<unsigned>((c.middle_width + 127) / 128),
                                          static_cast<unsigned>(capacity), 2),
                                     dim3(32, 8, 1), 0, stream>>>(
            c.gate_weights, c.up_weights, static_cast<const block_q8_1_mmq*>(c.input_quant),
            c.ids_destination, c.bounds, c.work, count, c.gate, c.up, c.middle_width, c.input_width,
            pairs, 256);
      }
      DS4_TRY(cudaGetLastError());
    } else {
      DS4_TRY((Classic<GGML_TYPE_IQ2_XXS>(c.gate_weights, c.input_quant, c.ids_destination,
                                          c.bounds, c.gate, c.middle_width, c.rows, c.input_width,
                                          p.gate, static_cast<float*>(fixup), stream)));
      DS4_TRY((Classic<GGML_TYPE_IQ2_XXS>(c.up_weights, c.input_quant, c.ids_destination, c.bounds,
                                          c.up, c.middle_width, c.rows, c.input_width, p.gate,
                                          static_cast<float*>(fixup), stream)));
    }
    if (c.tier != 1) {
      const auto n = static_cast<std::uint64_t>(pairs) * static_cast<std::uint64_t>(c.middle_width);
      if (c.tier == 3) {
        moe_mmq_swiglu_weighted_clamp_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0,
                                               stream>>>(
            c.middle, nullptr, nullptr, c.gate, c.up, weights,
            static_cast<std::uint32_t>(c.middle_width), static_cast<std::uint32_t>(c.rows), 6,
            10.0F);
      } else {
        ds4_swiglu_weighted_f32<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
            c.gate, c.up, weights, c.middle, n, c.middle_width, 10.0F);
      }
      DS4_TRY(cudaGetLastError());
      DS4_TRY(Quant<MMQ_Q8_1_DS_LAYOUT_D2S6>(c.middle, c.ids_destination, c.down_quant, pairs,
                                             c.middle_width, stream));
    }
    if (c.tier == 3) {
      DS4_TRY((Classic<GGML_TYPE_Q2_K>(c.down_weights, c.down_quant, c.ids_destination, c.bounds,
                                       c.down, c.output_width, c.rows, c.middle_width, p.down,
                                       static_cast<float*>(fixup), stream)));
    } else {
      const int capacity = (pairs + 63) / 64 + 256;
      int* count = c.work + capacity;
      d2r_build_worklist_kernel<64><<<1, 256, 0, stream>>>(c.bounds, c.work, count, 256);
      DS4_TRY(cudaGetLastError());
      down_q2k_d2r_kernel<64, 64><<<dim3(static_cast<unsigned>((c.output_width + 127) / 128),
                                         static_cast<unsigned>(capacity), 1),
                                    dim3(32, 8, 1), 0, stream>>>(
          c.down_weights, static_cast<const block_q8_1_mmq*>(c.down_quant), c.ids_destination,
          c.bounds, c.work, count, c.down, c.output_width, c.middle_width, pairs, 256);
      DS4_TRY(cudaGetLastError());
    }
  }
  if (c.sum != nullptr) return Sum(c.down, c.sum, c.rows, c.output_width, stream);
  return cudaSuccess;
}
#undef DS4_TRY
}  // namespace jitllm::kernels::ggml::ds4_moe
