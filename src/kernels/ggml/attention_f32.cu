// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <cstring>
#include <limits>

#include "base/bytes.h"
#include "base/check.h"
#include "common.cuh"
#include "cpy.cuh"
#include "fill.cuh"
#include "kernels/ggml/attention_f32.h"
#include "kernels/ggml/fattn_mma.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
#include "kernels/ggml/validate_util.h"
namespace llmp::kernels::ggml {
namespace {
struct PackedAttention {
  std::array<ggml_tensor, 3> padded{}, real{};
  ggml_tensor output{};
};
void Packed(ggml_tensor& t, std::array<std::int64_t, 4> dims, void* data, float fill) {
  t = {};
  t.type = GGML_TYPE_F16;
  t.op = GGML_OP_FILL;
  t.data = data;
  std::memcpy(t.op_params, &fill, sizeof(fill));
  std::size_t stride = sizeof(ggml_fp16_t);
  for (std::size_t i = 0; i < dims.size(); ++i) {
    t.ne[i] = dims[i];
    t.nb[i] = stride;
    stride *= static_cast<std::size_t>(dims[i]);
  }
}
void Make(PackedAttention& a, const ggml_tensor* node, const AttentionF32Plan& p,
          std::byte* scratch) {
  const auto cells = static_cast<std::int64_t>(p.cells);
  const auto rows = static_cast<std::int64_t>(p.rows);
  Packed(a.padded[0], {128, cells, node->src[1]->ne[2], 1}, scratch, 0);
  Packed(a.padded[1], {128, cells, node->src[2]->ne[2], 1}, scratch + p.kv_bytes, 0);
  Packed(a.padded[2], {cells, rows, 1, 1}, scratch + 2 * p.kv_bytes,
         -std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < a.real.size(); ++i) {
    a.real[i] = a.padded[i];
    // Same number/order of real elements as the source; padded row/head
    // pitches stay intact. Existing scalar cpy converts to those addresses.
    for (int d = 0; d < GGML_MAX_DIMS; ++d) a.real[i].ne[d] = node->src[i + 1]->ne[d];
  }
  a.output = *node;
  for (std::size_t i = 0; i < a.padded.size(); ++i) a.output.src[i + 1] = &a.padded[i];
}
std::expected<FlashAttnMmaPlan, KernelFailure> Physical(const LaunchContext& launch,
                                                        const ggml_tensor* node,
                                                        const AttentionF32Plan& plan,
                                                        std::byte* scratch) {
  PackedAttention packed;
  Make(packed, node, plan, scratch);
  for (const auto& t : packed.padded)
    if (auto check = CheckFill(&t); !check) return std::unexpected(check.error());
  return PlanFlashAttnMma128(launch, &packed.output);
}
template <typename SrcT>
__global__ void PackAttentionKvContigKernel(const char* __restrict__ src, half* __restrict__ dst,
                                            std::int64_t real_cols, std::int64_t padded_cols,
                                            std::int64_t heads, std::size_t nb1, std::size_t nb2) {
  const int d = threadIdx.x * 4;
  const std::int64_t col = blockIdx.x;
  const std::int64_t head = static_cast<std::int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (col >= padded_cols || head >= heads) return;
  uint2 packed = make_uint2(0U, 0U);
  if (col < real_cols) {
    const auto* row = reinterpret_cast<const SrcT*>(src + static_cast<std::size_t>(col) * nb1 +
                                                    static_cast<std::size_t>(head) * nb2);
    if constexpr (std::is_same_v<SrcT, nv_bfloat16>) {
      const nv_bfloat162 b0 = *reinterpret_cast<const nv_bfloat162*>(row + d);
      const nv_bfloat162 b1 = *reinterpret_cast<const nv_bfloat162*>(row + d + 2);
      const float2 f0 = __bfloat1622float2(b0);
      const float2 f1 = __bfloat1622float2(b1);
      const half2 lo = __floats2half2_rn(f0.x, f0.y);
      const half2 hi = __floats2half2_rn(f1.x, f1.y);
      packed.x = *reinterpret_cast<const unsigned int*>(&lo);
      packed.y = *reinterpret_cast<const unsigned int*>(&hi);
    } else {
      const float4 f4 = *reinterpret_cast<const float4*>(row + d);
      const half2 lo = __floats2half2_rn(f4.x, f4.y);
      const half2 hi = __floats2half2_rn(f4.z, f4.w);
      packed.x = *reinterpret_cast<const unsigned int*>(&lo);
      packed.y = *reinterpret_cast<const unsigned int*>(&hi);
    }
  }
  *reinterpret_cast<uint2*>(dst + d + 128 * (col + padded_cols * head)) = packed;
}

__global__ void PackAttentionKvTiledKernel(const char* __restrict__ src, half* __restrict__ dst,
                                           std::int64_t real_cols, std::int64_t padded_cols,
                                           std::size_t nb0, std::size_t nb1, std::size_t nb2) {
  __shared__ half tile[32][33];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const std::int64_t base_col = static_cast<std::int64_t>(blockIdx.x) * 32;
  const std::int64_t base_d = static_cast<std::int64_t>(blockIdx.y) * 32;
  const std::int64_t head = blockIdx.z;
  const char* head_src = src + static_cast<std::size_t>(head) * nb2;
  for (int j = 0; j < 32; j += 8) {
    const std::int64_t d = base_d + ty + j;
    const std::int64_t col = base_col + tx;
    half val = __float2half(0.0f);
    if (d < 128 && col < real_cols) {
      const float f = *reinterpret_cast<const float*>(head_src + static_cast<std::size_t>(d) * nb0 +
                                                      static_cast<std::size_t>(col) * nb1);
      val = __float2half(f);
    }
    tile[ty + j][tx] = val;
  }
  __syncthreads();
  for (int j = 0; j < 32; j += 8) {
    const std::int64_t col = base_col + ty + j;
    const std::int64_t d = base_d + tx;
    if (col < padded_cols && d < 128) {
      dst[d + 128 * (col + padded_cols * head)] = tile[tx][ty + j];
    }
  }
}

__global__ void PackAttentionMaskKernel(const char* __restrict__ src, half* __restrict__ dst,
                                        std::int64_t real_cols, std::int64_t real_rows,
                                        std::int64_t padded_cols, std::int64_t padded_rows,
                                        std::size_t nb0, std::size_t nb1) {
  const std::int64_t row = blockIdx.x;
  if (row >= padded_rows) return;
  half* row_dst = dst + padded_cols * row;
  if (row >= real_rows) {
    if (threadIdx.x == 0) {
      *reinterpret_cast<std::uint16_t*>(row_dst) = 0x8000U;
    }
    return;
  }
  const char* row_src = src + static_cast<std::size_t>(row) * nb1;
  int any_nonzero = 0;
  if (nb0 == sizeof(float)) {
    const auto* f_src = reinterpret_cast<const float*>(row_src);
    for (std::int64_t col = threadIdx.x; col < real_cols; col += 256) {
      if (f_src[col] != 0.0f) {
        any_nonzero = 1;
        break;
      }
    }
  } else {
    for (std::int64_t col = threadIdx.x; col < real_cols; col += 256) {
      const float f =
          *reinterpret_cast<const float*>(row_src + static_cast<std::size_t>(col) * nb0);
      if (f != 0.0f) {
        any_nonzero = 1;
        break;
      }
    }
  }
  if (__syncthreads_or(any_nonzero) == 0) {
    if (threadIdx.x == 0) {
      *reinterpret_cast<std::uint16_t*>(row_dst) = 0x8000U;
    }
    return;
  }
  for (std::int64_t col = threadIdx.x; col < padded_cols; col += 256) {
    half val = __float2half(-INFINITY);
    if (col < real_cols) {
      const float f =
          *reinterpret_cast<const float*>(row_src + static_cast<std::size_t>(col) * nb0);
      val = (f == 0.0f) ? __ushort_as_half(0x0000U) : __float2half(f);
    }
    row_dst[col] = val;
  }
}

void LaunchPackAttentionKv(cudaStream_t stream, const ggml_tensor* src, ggml_tensor& dst,
                           bool src_is_bf16 = false) {
  const auto real_cols = src->ne[1];
  const auto padded_cols = dst.ne[1];
  const auto heads = dst.ne[2];
  auto* out = reinterpret_cast<half*>(dst.data);
  const auto* in = static_cast<const char*>(src->data);
  if (src_is_bf16) {
    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned>(padded_cols), static_cast<unsigned>((heads + 7) / 8), 1);
    PackAttentionKvContigKernel<nv_bfloat16><<<grid, block, 0, stream>>>(
        in, out, real_cols, padded_cols, heads, src->nb[1] / 2, src->nb[2] / 2);
  } else if (src->nb[0] == sizeof(float)) {
    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned>(padded_cols), static_cast<unsigned>((heads + 7) / 8), 1);
    PackAttentionKvContigKernel<float><<<grid, block, 0, stream>>>(in, out, real_cols, padded_cols,
                                                                   heads, src->nb[1], src->nb[2]);
  } else {
    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned>((padded_cols + 31) / 32), 4U,
                    static_cast<unsigned>(heads));
    PackAttentionKvTiledKernel<<<grid, block, 0, stream>>>(in, out, real_cols, padded_cols,
                                                           src->nb[0], src->nb[1], src->nb[2]);
  }
}

void LaunchPackAttentionMask(cudaStream_t stream, const ggml_tensor* src, ggml_tensor& dst) {
  const auto real_cols = src->ne[0];
  const auto real_rows = src->ne[1];
  const auto padded_cols = dst.ne[0];
  const auto padded_rows = dst.ne[1];
  const dim3 block(256, 1, 1);
  const dim3 grid(static_cast<unsigned>(padded_rows), 1, 1);
  PackAttentionMaskKernel<<<grid, block, 0, stream>>>(
      static_cast<const char*>(src->data), reinterpret_cast<half*>(dst.data), real_cols, real_rows,
      padded_cols, padded_rows, src->nb[0], src->nb[1]);
}
constexpr int kF16D = 128;
constexpr int kF16Br = 128;
constexpr int kF16Bc = 64;
constexpr int kF16Warps = kF16Br / 16;
constexpr int kF16Threads = kF16Warps * 32;
constexpr int kF16Chunks = kF16D * 2 / 16;
constexpr int kF16TileBytes = kF16Bc * kF16D * 2;
constexpr int kF16Depth = 2;
constexpr int SmemForF16(int stages) { return stages * 2 * kF16TileBytes; }

__device__ __forceinline__ std::uint32_t SwizzleF16(int row, int chunk) {
  return static_cast<std::uint32_t>(row * kF16Chunks + (chunk ^ (row & 7))) * 16U;
}

__device__ __forceinline__ void CpAsync16F16(std::uint32_t dst, const void* src, bool valid) {
  const int size = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(size));
}
__device__ __forceinline__ void CpAsyncCommitF16() { asm volatile("cp.async.commit_group;\n" ::); }
template <int kPending>
__device__ __forceinline__ void CpAsyncWaitF16() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
}

__device__ __forceinline__ void LdMatrixX4F16(std::uint32_t address, std::uint32_t& r0,
                                              std::uint32_t& r1, std::uint32_t& r2,
                                              std::uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(address));
}
__device__ __forceinline__ void LdMatrixX4TransF16(std::uint32_t address, std::uint32_t& r0,
                                                   std::uint32_t& r1, std::uint32_t& r2,
                                                   std::uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(address));
}

__device__ __forceinline__ void MmaF16(float (&d)[4], const std::uint32_t (&a)[4], std::uint32_t b0,
                                       std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ std::uint32_t PackF16Pair(float lo, float hi) {
  const half2 h2 = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const std::uint32_t*>(&h2);
}

template <int kStages, bool kQIsF16 = false, bool kOutIsBf16 = false>
__global__ void __launch_bounds__(kF16Threads, 1)
    FlashForwardF16Kernel(const void* __restrict__ q, const half* __restrict__ k,
                          const half* __restrict__ v, const half* __restrict__ mask,
                          void* __restrict__ out, int q_rows, int kv_rows, int padded_kv_cols,
                          int heads, float scale) {
  extern __shared__ __align__(128) unsigned char smem[];
  const std::uint32_t base = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
  const auto k_smem = [&](int stage) { return base + (stage * kF16TileBytes); };
  const auto v_smem = [&](int stage) { return base + ((kStages + stage) * kF16TileBytes); };
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int group_id = lane >> 2;
  const int tig = lane & 3;
  const int head = static_cast<int>(blockIdx.y);
  const int m0 = static_cast<int>(blockIdx.x) * kF16Br + warp * 16;

  const half* kh = k + static_cast<std::int64_t>(head) * padded_kv_cols * kF16D;
  const half* vh = v + static_cast<std::int64_t>(head) * padded_kv_cols * kF16D;
  const int tiles = (kv_rows + kF16Bc - 1) / kF16Bc;

  auto load_tile = [&](int tile, int stage) {
    const int key0 = tile * kF16Bc;
#pragma unroll
    for (int i = 0; i < kF16Bc * kF16Chunks / kF16Threads; ++i) {
      const int item = tid + i * kF16Threads;
      const int row = item / kF16Chunks;
      const int chunk = item % kF16Chunks;
      const int key = key0 + row;
      const bool valid = key < kv_rows;
      const std::int64_t at = static_cast<std::int64_t>(valid ? key : 0);
      CpAsync16F16(k_smem(stage) + SwizzleF16(row, chunk), kh + at * kF16D + chunk * 8, valid);
      CpAsync16F16(v_smem(stage) + SwizzleF16(row, chunk), vh + at * kF16D + chunk * 8, valid);
    }
    CpAsyncCommitF16();
  };

#pragma unroll
  for (int t = 0; t < kStages - 1; ++t) {
    if (t < tiles) {
      load_tile(t, t);
    } else {
      CpAsyncCommitF16();
    }
  }

  const int r0 = m0 + group_id;
  const int r1 = r0 + 8;
  std::uint32_t qa[8][4];
  if constexpr (kQIsF16) {
    const half* q0 = static_cast<const half*>(q) +
                     (static_cast<std::int64_t>(head) * q_rows + (r0 < q_rows ? r0 : 0)) * kF16D;
    const half* q1 = static_cast<const half*>(q) +
                     (static_cast<std::int64_t>(head) * q_rows + (r1 < q_rows ? r1 : 0)) * kF16D;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int c = kk * 16 + tig * 2;
      qa[kk][0] = r0 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q0 + c) : 0U;
      qa[kk][1] = r1 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q1 + c) : 0U;
      qa[kk][2] = r0 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q0 + c + 8) : 0U;
      qa[kk][3] = r1 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q1 + c + 8) : 0U;
    }
  } else {
    const float* q0 = static_cast<const float*>(q) +
                      (static_cast<std::int64_t>(head) * q_rows + (r0 < q_rows ? r0 : 0)) * kF16D;
    const float* q1 = static_cast<const float*>(q) +
                      (static_cast<std::int64_t>(head) * q_rows + (r1 < q_rows ? r1 : 0)) * kF16D;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int c = kk * 16 + tig * 2;
      const float2 v0a =
          r0 < q_rows ? *reinterpret_cast<const float2*>(q0 + c) : make_float2(0.0f, 0.0f);
      const float2 v1a =
          r1 < q_rows ? *reinterpret_cast<const float2*>(q1 + c) : make_float2(0.0f, 0.0f);
      const float2 v0b =
          r0 < q_rows ? *reinterpret_cast<const float2*>(q0 + c + 8) : make_float2(0.0f, 0.0f);
      const float2 v1b =
          r1 < q_rows ? *reinterpret_cast<const float2*>(q1 + c + 8) : make_float2(0.0f, 0.0f);
      qa[kk][0] = PackF16Pair(v0a.x, v0a.y);
      qa[kk][1] = PackF16Pair(v1a.x, v1a.y);
      qa[kk][2] = PackF16Pair(v0b.x, v0b.y);
      qa[kk][3] = PackF16Pair(v1b.x, v1b.y);
    }
  }

  const half* mask_row0 = mask + static_cast<std::int64_t>(r0 < q_rows ? r0 : 0) * padded_kv_cols;
  const half* mask_row1 = mask + static_cast<std::int64_t>(r1 < q_rows ? r1 : 0) * padded_kv_cols;
  const bool has_mask0 =
      (r0 < q_rows) && (*reinterpret_cast<const std::uint16_t*>(mask_row0) != 0x8000U);
  const bool has_mask1 =
      (r1 < q_rows) && (*reinterpret_cast<const std::uint16_t*>(mask_row1) != 0x8000U);

  float o[16][4];
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
  }
  float row_max[2] = {-INFINITY, -INFINITY};
  float row_sum[2] = {0.0f, 0.0f};

  const int lm_row = lane % 8;
  const int lm_mat = lane / 8;
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;

  for (int tile = 0; tile < tiles; ++tile) {
    const int stage = tile % kStages;
    const int ahead = tile + kStages - 1;
    if (ahead < tiles) {
      load_tile(ahead, ahead % kStages);
    } else {
      CpAsyncCommitF16();
    }
    CpAsyncWaitF16<kStages - 1>();
    __syncthreads();

    float s[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
    }
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        const int key = np * 16 + (lm_mat >> 1) * 8 + lm_row;
        const int chunk = kk * 2 + (lm_mat & 1);
        std::uint32_t b0, b1, b2, b3;
        LdMatrixX4F16(k_smem(stage) + SwizzleF16(key, chunk), b0, b1, b2, b3);
        MmaF16(s[np * 2], qa[kk], b0, b1);
        MmaF16(s[np * 2 + 1], qa[kk], b2, b3);
      }
    }
    if (!has_mask0 && !has_mask1) {
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        s[n][0] *= scale_log2;
        s[n][1] *= scale_log2;
        s[n][2] *= scale_log2;
        s[n][3] *= scale_log2;
      }
      if (tile * kF16Bc + kF16Bc > kv_rows) {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          const int key = tile * kF16Bc + n * 8 + tig * 2;
          if (key >= kv_rows) {
            s[n][0] = s[n][2] = -INFINITY;
          }
          if (key + 1 >= kv_rows) {
            s[n][1] = s[n][3] = -INFINITY;
          }
        }
      }
    } else {
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int key = tile * kF16Bc + n * 8 + tig * 2;
        const float2 m0_f2 = has_mask0
                                 ? __half22float2(*reinterpret_cast<const half2*>(mask_row0 + key))
                                 : make_float2(0.0f, 0.0f);
        const float2 m1_f2 = has_mask1
                                 ? __half22float2(*reinterpret_cast<const half2*>(mask_row1 + key))
                                 : make_float2(0.0f, 0.0f);
        s[n][0] = (s[n][0] * scale + m0_f2.x) * kLog2e;
        s[n][1] = (s[n][1] * scale + m0_f2.y) * kLog2e;
        s[n][2] = (s[n][2] * scale + m1_f2.x) * kLog2e;
        s[n][3] = (s[n][3] * scale + m1_f2.y) * kLog2e;
        if (key >= kv_rows) {
          s[n][0] = s[n][2] = -INFINITY;
        }
        if (key + 1 >= kv_rows) {
          s[n][1] = s[n][3] = -INFINITY;
        }
      }
    }

    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      tile_max[0] = fmaxf(tile_max[0], fmaxf(s[n][0], s[n][1]));
      tile_max[1] = fmaxf(tile_max[1], fmaxf(s[n][2], s[n][3]));
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffU, tile_max[r], 1));
      tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffU, tile_max[r], 2));
    }
    float correction[2];
    float safe_max[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float next = fmaxf(row_max[r], tile_max[r]);
      safe_max[r] = (next == -INFINITY) ? 0.0f : next;
      correction[r] = exp2f(row_max[r] - safe_max[r]);
      row_max[r] = next;
    }
    float tile_sum[2] = {0.0f, 0.0f};
    std::uint32_t pa[4][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const float p0 = exp2f(s[n][0] - safe_max[0]);
      const float p1 = exp2f(s[n][1] - safe_max[0]);
      const float p2 = exp2f(s[n][2] - safe_max[1]);
      const float p3 = exp2f(s[n][3] - safe_max[1]);
      tile_sum[0] += p0 + p1;
      tile_sum[1] += p2 + p3;
      const int j = n / 2;
      if (n % 2 == 0) {
        pa[j][0] = PackF16Pair(p0, p1);
        pa[j][1] = PackF16Pair(p2, p3);
      } else {
        pa[j][2] = PackF16Pair(p0, p1);
        pa[j][3] = PackF16Pair(p2, p3);
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      row_sum[r] = row_sum[r] * correction[r] + tile_sum[r];
    }
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      o[n][0] *= correction[0];
      o[n][1] *= correction[0];
      o[n][2] *= correction[1];
      o[n][3] *= correction[1];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int np = 0; np < 8; ++np) {
        const int key = j * 16 + (lm_mat & 1) * 8 + lm_row;
        const int chunk = np * 2 + (lm_mat >> 1);
        std::uint32_t b0, b1, b2, b3;
        LdMatrixX4TransF16(v_smem(stage) + SwizzleF16(key, chunk), b0, b1, b2, b3);
        MmaF16(o[np * 2], pa[j], b0, b1);
        MmaF16(o[np * 2 + 1], pa[j], b2, b3);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    row_sum[r] += __shfl_xor_sync(0xffffffffU, row_sum[r], 1);
    row_sum[r] += __shfl_xor_sync(0xffffffffU, row_sum[r], 2);
  }
  const float inv[2] = {row_sum[0] > 0.0f ? 1.0f / row_sum[0] : 0.0f,
                        row_sum[1] > 0.0f ? 1.0f / row_sum[1] : 0.0f};
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    const int c = n * 8 + tig * 2;
    if constexpr (kOutIsBf16) {
      auto* out_b = static_cast<nv_bfloat16*>(out);
      if (r0 < q_rows) {
        *reinterpret_cast<nv_bfloat162*>(
            out_b + (static_cast<std::int64_t>(r0) * heads + head) * kF16D + c) =
            __floats2bfloat162_rn(o[n][0] * inv[0], o[n][1] * inv[0]);
      }
      if (r1 < q_rows) {
        *reinterpret_cast<nv_bfloat162*>(
            out_b + (static_cast<std::int64_t>(r1) * heads + head) * kF16D + c) =
            __floats2bfloat162_rn(o[n][2] * inv[1], o[n][3] * inv[1]);
      }
    } else {
      auto* out_f = static_cast<float*>(out);
      if (r0 < q_rows) {
        *reinterpret_cast<float2*>(out_f + (static_cast<std::int64_t>(r0) * heads + head) * kF16D +
                                   c) = make_float2(o[n][0] * inv[0], o[n][1] * inv[0]);
      }
      if (r1 < q_rows) {
        *reinterpret_cast<float2*>(out_f + (static_cast<std::int64_t>(r1) * heads + head) * kF16D +
                                   c) = make_float2(o[n][2] * inv[1], o[n][3] * inv[1]);
      }
    }
  }
}

__device__ __forceinline__ void MmaBf16(float (&d)[4], const std::uint32_t (&a)[4],
                                        std::uint32_t b0, std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ std::uint32_t PackBf16Pair(float lo, float hi) {
  const nv_bfloat162 b2 = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const std::uint32_t*>(&b2);
}

__device__ __forceinline__ void NormRopeRowBf16(std::uint32_t (&a)[8], std::uint32_t (&b)[8],
                                                const nv_bfloat16* w, const float* freqs, int tig,
                                                float eps) {
  float2 va[8];
  float2 vb[8];
  float sum_sq = 0.0f;
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    va[kk] = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&a[kk]));
    vb[kk] = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&b[kk]));
    sum_sq += va[kk].x * va[kk].x + va[kk].y * va[kk].y + vb[kk].x * vb[kk].x + vb[kk].y * vb[kk].y;
  }
  sum_sq += __shfl_xor_sync(0xffffffffU, sum_sq, 1);
  sum_sq += __shfl_xor_sync(0xffffffffU, sum_sq, 2);
  const float inv_rms = rsqrtf(sum_sq * (1.0f / 128.0f) + eps);
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int d = kk * 16 + tig * 2 + h * 8;
      const float2 v = (h == 0) ? va[kk] : vb[kk];
      const float2 wf = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(w + d));
      const float2 cs = *reinterpret_cast<const float2*>(freqs + d);
      const float n0 = v.x * inv_rms * wf.x;
      const float n1 = v.y * inv_rms * wf.y;
      const float o0 = n0 * cs.x - n1 * cs.y;
      const float o1 = n0 * cs.y + n1 * cs.x;
      const std::uint32_t packed = PackBf16Pair(o0, o1);
      if (h == 0) {
        a[kk] = packed;
      } else {
        b[kk] = packed;
      }
    }
  }
}

template <int kStages, bool kOutIsBf16 = false, bool kNormQ = false>
__global__ void __launch_bounds__(kF16Threads, 1)
    FlashForwardBf16RowMajorKernel(const nv_bfloat16* __restrict__ q,
                                   const nv_bfloat16* __restrict__ k,
                                   const nv_bfloat16* __restrict__ v, const half* __restrict__ mask,
                                   void* __restrict__ out, int q_rows, int kv_rows,
                                   int padded_kv_cols, int heads, int heads_k, float scale,
                                   const nv_bfloat16* __restrict__ q_weight,
                                   const float* __restrict__ freqs, float q_eps) {
  extern __shared__ __align__(128) unsigned char smem[];
  const std::uint32_t base = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
  const auto k_smem = [&](int stage) { return base + (stage * kF16TileBytes); };
  const auto v_smem = [&](int stage) { return base + ((kStages + stage) * kF16TileBytes); };
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int group_id = lane >> 2;
  const int tig = lane & 3;
  const int head = static_cast<int>(blockIdx.y);
  const int group = heads / heads_k;
  const int kv_head = head / group;
  const int m0 = static_cast<int>(blockIdx.x) * kF16Br + warp * 16;

  const std::int64_t kv_stride = static_cast<std::int64_t>(heads_k) * kF16D;
  const nv_bfloat16* kh = k + static_cast<std::int64_t>(kv_head) * kF16D;
  const nv_bfloat16* vh = v + static_cast<std::int64_t>(kv_head) * kF16D;
  const int tiles = (kv_rows + kF16Bc - 1) / kF16Bc;

  auto load_tile = [&](int tile, int stage) {
    const int key0 = tile * kF16Bc;
#pragma unroll
    for (int i = 0; i < kF16Bc * kF16Chunks / kF16Threads; ++i) {
      const int item = tid + i * kF16Threads;
      const int row = item / kF16Chunks;
      const int chunk = item % kF16Chunks;
      const int key = key0 + row;
      const bool valid = key < kv_rows;
      const std::int64_t at = static_cast<std::int64_t>(valid ? key : 0);
      CpAsync16F16(k_smem(stage) + SwizzleF16(row, chunk), kh + at * kv_stride + chunk * 8, valid);
      CpAsync16F16(v_smem(stage) + SwizzleF16(row, chunk), vh + at * kv_stride + chunk * 8, valid);
    }
    CpAsyncCommitF16();
  };

#pragma unroll
  for (int t = 0; t < kStages - 1; ++t) {
    if (t < tiles) {
      load_tile(t, t);
    } else {
      CpAsyncCommitF16();
    }
  }

  const int r0 = m0 + group_id;
  const int r1 = r0 + 8;
  const nv_bfloat16* q0 =
      q + (static_cast<std::int64_t>(r0 < q_rows ? r0 : 0) * heads + head) * kF16D;
  const nv_bfloat16* q1 =
      q + (static_cast<std::int64_t>(r1 < q_rows ? r1 : 0) * heads + head) * kF16D;
  std::uint32_t qa[8][4];
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    const int c = kk * 16 + tig * 2;
    qa[kk][0] = r0 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q0 + c) : 0U;
    qa[kk][1] = r1 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q1 + c) : 0U;
    qa[kk][2] = r0 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q0 + c + 8) : 0U;
    qa[kk][3] = r1 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q1 + c + 8) : 0U;
  }
  if constexpr (kNormQ) {
    std::uint32_t a0[8], b0[8], a1[8], b1[8];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      a0[kk] = qa[kk][0];
      b0[kk] = qa[kk][2];
      a1[kk] = qa[kk][1];
      b1[kk] = qa[kk][3];
    }
    const float* f0 = freqs + static_cast<std::int64_t>(r0 < q_rows ? r0 : q_rows - 1) * kF16D;
    const float* f1 = freqs + static_cast<std::int64_t>(r1 < q_rows ? r1 : q_rows - 1) * kF16D;
    NormRopeRowBf16(a0, b0, q_weight, f0, tig, q_eps);
    NormRopeRowBf16(a1, b1, q_weight, f1, tig, q_eps);
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      qa[kk][0] = a0[kk];
      qa[kk][2] = b0[kk];
      qa[kk][1] = a1[kk];
      qa[kk][3] = b1[kk];
    }
  }

  const half* mask_row0 = mask + static_cast<std::int64_t>(r0 < q_rows ? r0 : 0) * padded_kv_cols;
  const half* mask_row1 = mask + static_cast<std::int64_t>(r1 < q_rows ? r1 : 0) * padded_kv_cols;
  const bool has_mask0 =
      (r0 < q_rows) && (*reinterpret_cast<const std::uint16_t*>(mask_row0) != 0x8000U);
  const bool has_mask1 =
      (r1 < q_rows) && (*reinterpret_cast<const std::uint16_t*>(mask_row1) != 0x8000U);

  float o[16][4];
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
  }
  float row_max[2] = {-INFINITY, -INFINITY};
  float row_sum[2] = {0.0f, 0.0f};

  const int lm_row = lane % 8;
  const int lm_mat = lane / 8;
  constexpr float kLog2e = 1.4426950408889634f;
  const float scale_log2 = scale * kLog2e;
  const float inv_scale = 1.0f / scale;

  for (int tile = 0; tile < tiles; ++tile) {
    const int stage = tile % kStages;
    const int ahead = tile + kStages - 1;
    if (ahead < tiles) {
      load_tile(ahead, ahead % kStages);
    } else {
      CpAsyncCommitF16();
    }
    CpAsyncWaitF16<kStages - 1>();
    __syncthreads();

    float s[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
    }
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        const int key = np * 16 + (lm_mat >> 1) * 8 + lm_row;
        const int chunk = kk * 2 + (lm_mat & 1);
        std::uint32_t b0, b1, b2, b3;
        LdMatrixX4F16(k_smem(stage) + SwizzleF16(key, chunk), b0, b1, b2, b3);
        MmaBf16(s[np * 2], qa[kk], b0, b1);
        MmaBf16(s[np * 2 + 1], qa[kk], b2, b3);
      }
    }
    if (!has_mask0 && !has_mask1) {
      if (tile * kF16Bc + kF16Bc > kv_rows) {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          const int key = tile * kF16Bc + n * 8 + tig * 2;
          if (key >= kv_rows) {
            s[n][0] = s[n][2] = -INFINITY;
          }
          if (key + 1 >= kv_rows) {
            s[n][1] = s[n][3] = -INFINITY;
          }
        }
      }
    } else {
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int key = tile * kF16Bc + n * 8 + tig * 2;
        const float2 m0_f2 = has_mask0
                                 ? __half22float2(*reinterpret_cast<const half2*>(mask_row0 + key))
                                 : make_float2(0.0f, 0.0f);
        const float2 m1_f2 = has_mask1
                                 ? __half22float2(*reinterpret_cast<const half2*>(mask_row1 + key))
                                 : make_float2(0.0f, 0.0f);
        s[n][0] += m0_f2.x * inv_scale;
        s[n][1] += m0_f2.y * inv_scale;
        s[n][2] += m1_f2.x * inv_scale;
        s[n][3] += m1_f2.y * inv_scale;
        if (key >= kv_rows) {
          s[n][0] = s[n][2] = -INFINITY;
        }
        if (key + 1 >= kv_rows) {
          s[n][1] = s[n][3] = -INFINITY;
        }
      }
    }

    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      tile_max[0] = fmaxf(tile_max[0], fmaxf(s[n][0], s[n][1]));
      tile_max[1] = fmaxf(tile_max[1], fmaxf(s[n][2], s[n][3]));
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffU, tile_max[r], 1));
      tile_max[r] = fmaxf(tile_max[r], __shfl_xor_sync(0xffffffffU, tile_max[r], 2));
    }
    float correction[2];
    float scaled_max[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float next = fmaxf(row_max[r], tile_max[r]);
      const float safe_next = (next == -INFINITY) ? 0.0f : next;
      correction[r] = exp2f((row_max[r] - safe_next) * scale_log2);
      row_max[r] = next;
      scaled_max[r] = safe_next * scale_log2;
    }
    float tile_sum[2] = {0.0f, 0.0f};
    std::uint32_t pa[4][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const float p0 = exp2f(s[n][0] * scale_log2 - scaled_max[0]);
      const float p1 = exp2f(s[n][1] * scale_log2 - scaled_max[0]);
      const float p2 = exp2f(s[n][2] * scale_log2 - scaled_max[1]);
      const float p3 = exp2f(s[n][3] * scale_log2 - scaled_max[1]);
      tile_sum[0] += p0 + p1;
      tile_sum[1] += p2 + p3;
      const int j = n / 2;
      if (n % 2 == 0) {
        pa[j][0] = PackBf16Pair(p0, p1);
        pa[j][1] = PackBf16Pair(p2, p3);
      } else {
        pa[j][2] = PackBf16Pair(p0, p1);
        pa[j][3] = PackBf16Pair(p2, p3);
      }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      row_sum[r] = row_sum[r] * correction[r] + tile_sum[r];
    }
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      o[n][0] *= correction[0];
      o[n][1] *= correction[0];
      o[n][2] *= correction[1];
      o[n][3] *= correction[1];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int np = 0; np < 8; ++np) {
        const int key = j * 16 + (lm_mat & 1) * 8 + lm_row;
        const int chunk = np * 2 + (lm_mat >> 1);
        std::uint32_t b0, b1, b2, b3;
        LdMatrixX4TransF16(v_smem(stage) + SwizzleF16(key, chunk), b0, b1, b2, b3);
        MmaBf16(o[np * 2], pa[j], b0, b1);
        MmaBf16(o[np * 2 + 1], pa[j], b2, b3);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    row_sum[r] += __shfl_xor_sync(0xffffffffU, row_sum[r], 1);
    row_sum[r] += __shfl_xor_sync(0xffffffffU, row_sum[r], 2);
  }
  const float inv[2] = {row_sum[0] > 0.0f ? 1.0f / row_sum[0] : 0.0f,
                        row_sum[1] > 0.0f ? 1.0f / row_sum[1] : 0.0f};
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    const int c = n * 8 + tig * 2;
    if constexpr (kOutIsBf16) {
      auto* out_b = static_cast<nv_bfloat16*>(out);
      if (r0 < q_rows) {
        *reinterpret_cast<nv_bfloat162*>(
            out_b + (static_cast<std::int64_t>(r0) * heads + head) * kF16D + c) =
            __floats2bfloat162_rn(o[n][0] * inv[0], o[n][1] * inv[0]);
      }
      if (r1 < q_rows) {
        *reinterpret_cast<nv_bfloat162*>(
            out_b + (static_cast<std::int64_t>(r1) * heads + head) * kF16D + c) =
            __floats2bfloat162_rn(o[n][2] * inv[1], o[n][3] * inv[1]);
      }
    } else {
      auto* out_f = static_cast<float*>(out);
      if (r0 < q_rows) {
        *reinterpret_cast<float2*>(out_f + (static_cast<std::int64_t>(r0) * heads + head) * kF16D +
                                   c) = make_float2(o[n][0] * inv[0], o[n][1] * inv[0]);
      }
      if (r1 < q_rows) {
        *reinterpret_cast<float2*>(out_f + (static_cast<std::int64_t>(r1) * heads + head) * kF16D +
                                   c) = make_float2(o[n][2] * inv[1], o[n][3] * inv[1]);
      }
    }
  }
}
}  // namespace
std::expected<std::uint64_t, KernelFailure> PlanAttentionF32(const LaunchContext& launch,
                                                             const ggml_tensor* node) {
  auto plan = CheckAttentionF32(node);
  if (!plan) return std::unexpected(plan.error());
  auto output = *node;
  auto query = *node->src[0];
  output.data = reinterpret_cast<void*>(0x100000000ULL);
  output.view_src = nullptr;
  output.view_offs = 0;
  query.data = reinterpret_cast<void*>(0x200000000ULL);
  query.view_src = nullptr;
  query.view_offs = 0;
  output.src[0] = &query;
  auto flash = Physical(launch, &output, *plan, reinterpret_cast<std::byte*>(0x1000000000ULL));
  if (!flash) return std::unexpected(flash.error());
  const int smem = SmemForF16(kF16Depth);
  if (cudaFuncSetAttribute(FlashForwardF16Kernel<kF16Depth, false, false>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess ||
      cudaFuncSetAttribute(FlashForwardF16Kernel<kF16Depth, true, false>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess ||
      cudaFuncSetAttribute(FlashForwardF16Kernel<kF16Depth, false, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess ||
      cudaFuncSetAttribute(FlashForwardF16Kernel<kF16Depth, true, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess ||
      cudaFuncSetAttribute(FlashForwardBf16RowMajorKernel<kF16Depth, false, false>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess ||
      cudaFuncSetAttribute(FlashForwardBf16RowMajorKernel<kF16Depth, true, false>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess ||
      cudaFuncSetAttribute(FlashForwardBf16RowMajorKernel<kF16Depth, false, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess ||
      cudaFuncSetAttribute(FlashForwardBf16RowMajorKernel<kF16Depth, true, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) {
    return detail::Rejected("attention shared memory limit");
  }
  return plan->packing_bytes + flash->scratch;
}
std::expected<void, KernelFailure> AttentionF32(LaunchContext& launch, ggml_tensor* node) {
  return AttentionF32WithOptions(launch, node, {});
}
std::expected<void, KernelFailure> AttentionF32WithOptions(LaunchContext& launch, ggml_tensor* node,
                                                           AttentionF32Options opts) {
  auto plan = CheckAttentionF32(node);
  if (!plan) return std::unexpected(plan.error());
  const auto workspace = launch.workspace();
  auto* scratch = reinterpret_cast<std::byte*>(workspace.base);
  auto flash = Physical(launch, node, *plan, scratch);
  if (!flash) return std::unexpected(flash.error());
  const auto bytes = plan->packing_bytes + flash->scratch;
  if (bytes > workspace.size.value()) return detail::Rejected("attention exceeds lent scratch");
  if (auto clear = CheckClearOf(node, workspace.base, bytes); !clear) return clear;
  const auto run = detail::FlashAttnMmaCase128(flash->columns);
  if (run == nullptr) return detail::Rejected("attention has no compiled query tile");
  const float scale = detail::ParamF32(node, 0);
  return launch.Run(base::Bytes(bytes), [&, opts](ggml_backend_cuda_context& context) {
    ggml_cuda_pool_alloc<std::byte> memory(context.pool(), plan->packing_bytes);
    base::Check(reinterpret_cast<std::uintptr_t>(memory.get()) == workspace.base,
                "attention packing starts at the paid pool base");
    PackedAttention packed;
    Make(packed, node, *plan, memory.get());
    cudaStream_t stream = context.stream();
    if (!opts.qkv_bf16_row_major) {
      if (!opts.k_prepacked) {
        LaunchPackAttentionKv(stream, node->src[1], packed.padded[0], false);
        CUDA_CHECK(cudaGetLastError());
        if (internal::CudaErrorPending()) return;
      }
      LaunchPackAttentionKv(stream, node->src[2], packed.padded[1], opts.v_is_bf16);
      CUDA_CHECK(cudaGetLastError());
      if (internal::CudaErrorPending()) return;
    }
    if (!opts.mask_prepacked) {
      LaunchPackAttentionMask(stream, node->src[3], packed.padded[2]);
      CUDA_CHECK(cudaGetLastError());
      if (internal::CudaErrorPending()) return;
    }
    const int q_rows = static_cast<int>(node->src[0]->ne[1]);
    const int kv_rows = static_cast<int>(node->src[1]->ne[1]);
    const int padded_kv_cols = static_cast<int>(plan->cells);
    const int heads = static_cast<int>(node->src[0]->ne[2]);
    const int heads_k = static_cast<int>(node->src[1]->ne[2]);
    const dim3 grid(static_cast<unsigned>((q_rows + kF16Br - 1) / kF16Br),
                    static_cast<unsigned>(heads));
    const auto smem = static_cast<std::size_t>(SmemForF16(kF16Depth));
    const auto* m_ptr = reinterpret_cast<const half*>(packed.padded[2].data);
    if (opts.qkv_bf16_row_major) {
      const auto* q_bf16 = static_cast<const nv_bfloat16*>(node->src[0]->data);
      const auto* k_bf16 = static_cast<const nv_bfloat16*>(node->src[1]->data);
      const auto* v_bf16 = static_cast<const nv_bfloat16*>(node->src[2]->data);
      const auto* q_w_bf16 = static_cast<const nv_bfloat16*>(opts.q_weight);
      const std::uint64_t freqs_bytes =
          ((static_cast<std::uint64_t>(q_rows) * 128 * sizeof(float)) + 255) & ~std::uint64_t{255};
      const auto* freqs_ptr = reinterpret_cast<const float*>(
          reinterpret_cast<const std::byte*>(packed.padded[2].data) - freqs_bytes);
      if (opts.norm_q_in_attn) {
        if (opts.bf16_dst) {
          FlashForwardBf16RowMajorKernel<kF16Depth, true, true>
              <<<grid, kF16Threads, smem, stream>>>(q_bf16, k_bf16, v_bf16, m_ptr, node->data,
                                                    q_rows, kv_rows, padded_kv_cols, heads, heads_k,
                                                    scale, q_w_bf16, freqs_ptr, opts.q_eps);
        } else {
          FlashForwardBf16RowMajorKernel<kF16Depth, false, true>
              <<<grid, kF16Threads, smem, stream>>>(q_bf16, k_bf16, v_bf16, m_ptr, node->data,
                                                    q_rows, kv_rows, padded_kv_cols, heads, heads_k,
                                                    scale, q_w_bf16, freqs_ptr, opts.q_eps);
        }
      } else if (opts.bf16_dst) {
        FlashForwardBf16RowMajorKernel<kF16Depth, true, false><<<grid, kF16Threads, smem, stream>>>(
            q_bf16, k_bf16, v_bf16, m_ptr, node->data, q_rows, kv_rows, padded_kv_cols, heads,
            heads_k, scale, nullptr, nullptr, 0.0f);
      } else {
        FlashForwardBf16RowMajorKernel<kF16Depth, false, false>
            <<<grid, kF16Threads, smem, stream>>>(q_bf16, k_bf16, v_bf16, m_ptr, node->data, q_rows,
                                                  kv_rows, padded_kv_cols, heads, heads_k, scale,
                                                  nullptr, nullptr, 0.0f);
      }
      CUDA_CHECK(cudaGetLastError());
      return;
    }
    const auto* k_ptr = reinterpret_cast<const half*>(packed.padded[0].data);
    const auto* v_ptr = reinterpret_cast<const half*>(packed.padded[1].data);
    if (opts.q_is_f16) {
      if (opts.bf16_dst) {
        FlashForwardF16Kernel<kF16Depth, true, true><<<grid, kF16Threads, smem, stream>>>(
            node->src[0]->data, k_ptr, v_ptr, m_ptr, node->data, q_rows, kv_rows, padded_kv_cols,
            heads, scale);
      } else {
        FlashForwardF16Kernel<kF16Depth, true, false><<<grid, kF16Threads, smem, stream>>>(
            node->src[0]->data, k_ptr, v_ptr, m_ptr, node->data, q_rows, kv_rows, padded_kv_cols,
            heads, scale);
      }
    } else {
      if (opts.bf16_dst) {
        FlashForwardF16Kernel<kF16Depth, false, true><<<grid, kF16Threads, smem, stream>>>(
            node->src[0]->data, k_ptr, v_ptr, m_ptr, node->data, q_rows, kv_rows, padded_kv_cols,
            heads, scale);
      } else {
        FlashForwardF16Kernel<kF16Depth, false, false><<<grid, kF16Threads, smem, stream>>>(
            node->src[0]->data, k_ptr, v_ptr, m_ptr, node->data, q_rows, kv_rows, padded_kv_cols,
            heads, scale);
      }
    }
    CUDA_CHECK(cudaGetLastError());
  });
}
}  // namespace llmp::kernels::ggml
