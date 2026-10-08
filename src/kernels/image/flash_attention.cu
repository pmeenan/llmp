// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// FlashAttention-2's forward pass (Dao, 2023), written for llmpalooza (ops.h
// FlashAttention): BF16, head dimension 128, no mask. Each block takes 128
// query rows of one head, eight warps of 16 rows each; the block streams
// the head's keys and values in tiles of 64 through shared memory, two
// stages deep (cp.async), and each warp keeps its queries, running maxima,
// sums and output in registers. Products are mma.sync m16n8k16 BF16 with
// F32 accumulation; shared memory rows are XOR-swizzled in 16-byte chunks,
// so that ldmatrix reads eight rows without bank conflicts. The last key
// tile is zero-filled and masked past kv_rows.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <expected>
#include <format>
#include <string>

#include "kernels/image/ops.h"

namespace llmp::kernels::image {
namespace {

constexpr int kD = 128;
constexpr int kBr = 128;  // query rows per block
constexpr int kBc = 64;   // keys per tile
constexpr int kWarps = kBr / 16;
constexpr int kThreads = kWarps * 32;
constexpr int kChunks = kD * 2 / 16;  // 16-byte chunks per row: 16
constexpr int kTileBytes = kBc * kD * 2;

__device__ __forceinline__ std::uint32_t Swizzle(int row, int chunk) {
  return static_cast<std::uint32_t>(row * kChunks + (chunk ^ (row & 7))) * 16U;
}

__device__ __forceinline__ void CpAsync16(std::uint32_t dst, const void* src, bool valid) {
  const int size = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(size));
}
__device__ __forceinline__ void CpAsyncCommit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int kPending>
__device__ __forceinline__ void CpAsyncWait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
}

__device__ __forceinline__ void LdMatrixX4(std::uint32_t address, std::uint32_t& r0,
                                           std::uint32_t& r1, std::uint32_t& r2,
                                           std::uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(address));
}
__device__ __forceinline__ void LdMatrixX4Trans(std::uint32_t address, std::uint32_t& r0,
                                                std::uint32_t& r1, std::uint32_t& r2,
                                                std::uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(address));
}

// d += a (16x16 row) * b (16x8 col), BF16 in, F32 accumulate.
__device__ __forceinline__ void Mma(float (&d)[4], const std::uint32_t (&a)[4], std::uint32_t b0,
                                    std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two floats as a BF16 pair (round to nearest even), low half first.
__device__ __forceinline__ std::uint32_t PackBf16(float lo, float hi) {
  std::uint32_t out = 0;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "f"(hi), "f"(lo));
  return out;
}

__device__ __forceinline__ float B2F(std::uint32_t half) { return __uint_as_float(half << 16U); }
__device__ __forceinline__ std::uint16_t F2B(float f) {
  std::uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((u >> 16U) | 0x40U);
  }
  u += 0x7fffU + ((u >> 16U) & 1U);
  return static_cast<std::uint16_t>(u >> 16U);
}
__device__ __forceinline__ float R(float f) { return B2F(F2B(f)); }

// HeadNormRopeComplex (ops.cu) on one query row's fragments, in place: the
// row's 128 dimensions are spread over the four threads of a quad, each
// holding, per k-step kk, the pairs at 16 kk + 2 tig and 16 kk + 2 tig + 8
// (a[kk], b[kk]). The sum of squares is taken in the order ops.cu's warp
// takes it (each four consecutive dimensions summed in order, then the
// warp's butterfly over those 32 sums), so the result is its bits.
__device__ __forceinline__ void NormRopeRow(std::uint32_t (&a)[8], std::uint32_t (&b)[8],
                                            const std::uint16_t* w, const float* freqs, int tig,
                                            int lane, float eps) {
  float va[8][2];
  float vb[8][2];
  float pa[8];
  float pb[8];
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    va[kk][0] = B2F(a[kk] & 0xffffU);
    va[kk][1] = B2F(a[kk] >> 16U);
    vb[kk][0] = B2F(b[kk] & 0xffffU);
    vb[kk][1] = B2F(b[kk] >> 16U);
    // The first two of four dimensions (an even tig), then the last two
    // added to them (an odd tig, whose sums are the ones used).
    const float xa = __fmaf_rn(va[kk][1], va[kk][1], __fmul_rn(va[kk][0], va[kk][0]));
    const float xb = __fmaf_rn(vb[kk][1], vb[kk][1], __fmul_rn(vb[kk][0], vb[kk][0]));
    const float ya = __shfl_xor_sync(0xffffffffU, xa, 1);
    const float yb = __shfl_xor_sync(0xffffffffU, xb, 1);
    pa[kk] = __fmaf_rn(va[kk][1], va[kk][1], __fmaf_rn(va[kk][0], va[kk][0], ya));
    pb[kk] = __fmaf_rn(vb[kk][1], vb[kk][1], __fmaf_rn(vb[kk][0], vb[kk][0], yb));
  }
  // The butterfly's tree: partial sums of groups g and g + 16, then + 8,
  // + 4 (all within this thread), + 2 (a's with b's) and + 1 (tig 1 with 3).
  float s1a[4];
  float s1b[4];
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    s1a[kk] = __fadd_rn(pa[kk], pa[kk + 4]);
    s1b[kk] = __fadd_rn(pb[kk], pb[kk + 4]);
  }
  const float s3a = __fadd_rn(__fadd_rn(s1a[0], s1a[2]), __fadd_rn(s1a[1], s1a[3]));
  const float s3b = __fadd_rn(__fadd_rn(s1b[0], s1b[2]), __fadd_rn(s1b[1], s1b[3]));
  const float s4 = __fadd_rn(s3a, s3b);
  const float total = __fadd_rn(s4, __shfl_xor_sync(0xffffffffU, s4, 2));
  const float sum = __shfl_sync(0xffffffffU, total, (lane & ~3) | 1);
  // ops.cu's contractions, as its kernel compiles them (SASS): the mean and
  // eps one FFMA, then each rotated pair's product fused as there.
  const float rrms = rsqrtf(__fmaf_rn(sum, 0.0078125f, eps));
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int d = (kk * 16) + (tig * 2) + (h * 8);
      const float* v = h == 0 ? va[kk] : vb[kk];
      const float n0 = R(R(__fmul_rn(v[0], rrms)) * B2F(w[d]));
      const float n1 = R(R(__fmul_rn(v[1], rrms)) * B2F(w[d + 1]));
      const float2 f = *reinterpret_cast<const float2*>(freqs + d);
      const float o0 = __fmaf_rn(n0, f.x, -__fmul_rn(n1, f.y));
      const float o1 = __fmaf_rn(n0, f.y, __fmul_rn(n1, f.x));
      const std::uint32_t packed =
          static_cast<std::uint32_t>(F2B(o0)) | (static_cast<std::uint32_t>(F2B(o1)) << 16U);
      (h == 0 ? a[kk] : b[kk]) = packed;
    }
  }
}

// The key pipeline's depth: two stages of K and V tiles (three, 96 KiB,
// measured no faster end to end in the speed slice).
constexpr int kDepth = 2;
constexpr int SmemFor(int stages) { return stages * 2 * kTileBytes; }

template <bool kNormQ, int kStages>
__global__ void __launch_bounds__(kThreads, 1)
    FlashForwardKernel(const std::uint16_t* __restrict__ q, std::int64_t q_stride,
                       const std::uint16_t* __restrict__ k, std::int64_t k_stride,
                       const std::uint16_t* __restrict__ v, std::int64_t v_stride,
                       std::uint16_t* __restrict__ out, std::int64_t out_stride, int q_rows,
                       int kv_rows, int group, float scale_log2,
                       const std::uint16_t* __restrict__ kp, const std::uint16_t* __restrict__ vp,
                       int prefix, const std::uint16_t* __restrict__ q_norm,
                       const float* __restrict__ freqs, float eps) {
  extern __shared__ __align__(128) unsigned char smem[];
  const std::uint32_t base = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
  // K stage s at s tiles, V stage s after the K stages.
  const auto k_smem = [&](int stage) { return base + (stage * kTileBytes); };
  const auto v_smem = [&](int stage) { return base + ((kStages + stage) * kTileBytes); };
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int group_id = lane >> 2;
  const int tig = lane & 3;
  const int head = static_cast<int>(blockIdx.y);
  const int kv_head = head / group;
  const int m0 = static_cast<int>(blockIdx.x) * kBr + warp * 16;

  const std::uint16_t* kh = k + static_cast<std::int64_t>(kv_head) * kD;
  const std::uint16_t* vh = v + static_cast<std::int64_t>(kv_head) * kD;
  // Keys below `prefix` from kp and vp (same strides), the rest from k and v.
  const std::uint16_t* kph = kp + static_cast<std::int64_t>(kv_head) * kD;
  const std::uint16_t* vph = vp + static_cast<std::int64_t>(kv_head) * kD;
  const int tiles = (kv_rows + kBc - 1) / kBc;

  // Stage one key and value tile: 64 rows x 16 chunks each, 4 per thread.
  auto load_tile = [&](int tile, int stage) {
    const int key0 = tile * kBc;
#pragma unroll
    for (int i = 0; i < kBc * kChunks / kThreads; ++i) {
      const int item = tid + i * kThreads;
      const int row = item / kChunks;
      const int chunk = item % kChunks;
      const int key = key0 + row;
      const bool valid = key < kv_rows;
      const bool front = key < prefix;
      const std::int64_t at = static_cast<std::int64_t>(!valid ? 0 : front ? key : key - prefix);
      CpAsync16(k_smem(stage) + Swizzle(row, chunk), (front ? kph : kh) + at * k_stride + chunk * 8,
                valid);
      CpAsync16(v_smem(stage) + Swizzle(row, chunk), (front ? vph : vh) + at * v_stride + chunk * 8,
                valid);
    }
    CpAsyncCommit();
  };

  // The first kStages - 1 tiles in flight (an empty group for each tile
  // past the end, so that every iteration's wait counts the same groups).
#pragma unroll
  for (int t = 0; t < kStages - 1; ++t) {
    if (t < tiles) {
      load_tile(t, t);
    } else {
      CpAsyncCommit();
    }
  }

  // This warp's queries as A fragments, 8 k-steps of 16 dimensions.
  std::uint32_t qa[8][4];
  {
    const int r0 = m0 + group_id;
    const int r1 = r0 + 8;
    const std::uint16_t* q0 =
        q + static_cast<std::int64_t>(r0 < q_rows ? r0 : 0) * q_stride + head * kD;
    const std::uint16_t* q1 =
        q + static_cast<std::int64_t>(r1 < q_rows ? r1 : 0) * q_stride + head * kD;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int c = kk * 16 + tig * 2;
      qa[kk][0] = r0 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q0 + c) : 0U;
      qa[kk][1] = r1 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q1 + c) : 0U;
      qa[kk][2] = r0 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q0 + c + 8) : 0U;
      qa[kk][3] = r1 < q_rows ? *reinterpret_cast<const std::uint32_t*>(q1 + c + 8) : 0U;
    }
    if constexpr (kNormQ) {
      // The raw projections, normed and rotated here (rows past the end
      // are zeros, with the last row's frequencies).
      std::uint32_t a0[8], b0[8], a1[8], b1[8];
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        a0[kk] = qa[kk][0];
        b0[kk] = qa[kk][2];
        a1[kk] = qa[kk][1];
        b1[kk] = qa[kk][3];
      }
      const float* f0 = freqs + static_cast<std::int64_t>(r0 < q_rows ? r0 : q_rows - 1) * kD;
      const float* f1 = freqs + static_cast<std::int64_t>(r1 < q_rows ? r1 : q_rows - 1) * kD;
      NormRopeRow(a0, b0, q_norm, f0, tig, lane, eps);
      NormRopeRow(a1, b1, q_norm, f1, tig, lane, eps);
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        qa[kk][0] = a0[kk];
        qa[kk][2] = b0[kk];
        qa[kk][1] = a1[kk];
        qa[kk][3] = b1[kk];
      }
    }
  }

  float o[16][4];
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
  }
  float row_max[2] = {-INFINITY, -INFINITY};
  float row_sum[2] = {0.0f, 0.0f};

  // ldmatrix lanes: lane l addresses row (l % 8) of matrix (l / 8).
  const int lm_row = lane % 8;
  const int lm_mat = lane / 8;

  for (int tile = 0; tile < tiles; ++tile) {
    const int stage = tile % kStages;
    // The tile kStages - 1 ahead into the stage the last iteration read
    // (the barrier ending it made that safe), then this tile's arrival.
    const int ahead = tile + kStages - 1;
    if (ahead < tiles) {
      load_tile(ahead, ahead % kStages);
    } else {
      CpAsyncCommit();
    }
    CpAsyncWait<kStages - 1>();
    __syncthreads();

    // S = Q K^T: 16 x 64 per warp, 8 n-tiles of 8 keys.
    float s[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
    }
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        // Matrices: (keys np*16 + 0..7, dims kk*16 + 0..7), (same keys,
        // dims +8), (keys np*16 + 8..15, dims kk*16), (those keys, +8).
        const int key = np * 16 + (lm_mat >> 1) * 8 + lm_row;
        const int chunk = kk * 2 + (lm_mat & 1);
        std::uint32_t b0, b1, b2, b3;
        LdMatrixX4(k_smem(stage) + Swizzle(key, chunk), b0, b1, b2, b3);
        Mma(s[np * 2], qa[kk], b0, b1);
        Mma(s[np * 2 + 1], qa[kk], b2, b3);
      }
    }
    // Keys past the end: -inf.
    if (tile * kBc + kBc > kv_rows) {
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int key = tile * kBc + n * 8 + tig * 2;
        if (key >= kv_rows) {
          s[n][0] = s[n][2] = -INFINITY;
        }
        if (key + 1 >= kv_rows) {
          s[n][1] = s[n][3] = -INFINITY;
        }
      }
    }
    // Online softmax: rows group_id (elements 0, 1) and group_id + 8 (2, 3).
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
      correction[r] = exp2f((row_max[r] - next) * scale_log2);
      row_max[r] = next;
      scaled_max[r] = next * scale_log2;
    }
    float tile_sum[2] = {0.0f, 0.0f};
    std::uint32_t pa[4][4];  // P as A fragments: 4 k-steps of 16 keys
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
        pa[j][0] = PackBf16(p0, p1);
        pa[j][1] = PackBf16(p2, p3);
      } else {
        pa[j][2] = PackBf16(p0, p1);
        pa[j][3] = PackBf16(p2, p3);
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
    // O += P V: 16 n-tiles of 8 dimensions, 4 k-steps of 16 keys.
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int np = 0; np < 8; ++np) {
        // Transposed matrices: (keys j*16 + 0..7, dims np*16 + 0..7), (keys
        // +8, same dims), (keys 0..7, dims +8), (keys +8, dims +8).
        const int key = j * 16 + (lm_mat & 1) * 8 + lm_row;
        const int chunk = np * 2 + (lm_mat >> 1);
        std::uint32_t b0, b1, b2, b3;
        LdMatrixX4Trans(v_smem(stage) + Swizzle(key, chunk), b0, b1, b2, b3);
        Mma(o[np * 2], pa[j], b0, b1);
        Mma(o[np * 2 + 1], pa[j], b2, b3);
      }
    }
    __syncthreads();  // the stage is refilled next iteration
  }

  // The row sums over the four threads of each row, then O / l.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    row_sum[r] += __shfl_xor_sync(0xffffffffU, row_sum[r], 1);
    row_sum[r] += __shfl_xor_sync(0xffffffffU, row_sum[r], 2);
  }
  const float inv[2] = {1.0f / row_sum[0], 1.0f / row_sum[1]};
  const int r0 = m0 + group_id;
  const int r1 = r0 + 8;
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    const int c = n * 8 + tig * 2;
    if (r0 < q_rows) {
      *reinterpret_cast<std::uint32_t*>(out + static_cast<std::int64_t>(r0) * out_stride +
                                        head * kD + c) =
          PackBf16(o[n][0] * inv[0], o[n][1] * inv[0]);
    }
    if (r1 < q_rows) {
      *reinterpret_cast<std::uint32_t*>(out + static_cast<std::int64_t>(r1) * out_stride +
                                        head * kD + c) =
          PackBf16(o[n][2] * inv[1], o[n][3] * inv[1]);
    }
  }
}

Status Launch(const Bf16* q, std::int64_t q_stride, const Bf16* k, std::int64_t k_stride,
              const Bf16* v, std::int64_t v_stride, Bf16* out, std::int64_t out_stride,
              std::int64_t q_rows, std::int64_t kv_rows, std::int64_t heads, std::int64_t kv_heads,
              float scale, const Bf16* kp, const Bf16* vp, std::int64_t prefix, const Bf16* q_norm,
              const float* freqs, float eps, Stream stream) {
  const auto aligned = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; };
  if (q_norm != nullptr && (freqs == nullptr || !aligned(freqs) || !std::isfinite(eps))) {
    return std::unexpected(std::string("FlashAttention: the query norm's frequencies"));
  }
  if (q_rows <= 0 || q_rows > (std::int64_t{1} << 30) || kv_rows <= 0 ||
      kv_rows > (std::int64_t{1} << 30) || heads <= 0 || heads > 65535 || kv_heads <= 0 ||
      heads % kv_heads != 0 || q_stride < heads * kD || k_stride < kv_heads * kD ||
      v_stride < kv_heads * kD || out_stride < heads * kD || q_stride % 8 != 0 ||
      k_stride % 8 != 0 || v_stride % 8 != 0 || out_stride % 8 != 0 || !aligned(q) || !aligned(k) ||
      !aligned(v) || !aligned(out) || prefix < 0 || prefix > kv_rows || !aligned(kp) ||
      !aligned(vp)) {
    return std::unexpected(std::string("FlashAttention: sizes, strides or alignment"));
  }
  // A positive, finite scale: the running maximum is taken over unscaled
  // scores, and masked keys (-inf) times 0 would be NaN.
  if (!(scale > 0.0f) || !std::isfinite(scale)) {
    return std::unexpected(std::string("FlashAttention: the scale must be positive and finite"));
  }
  const bool norm = q_norm != nullptr;
  const auto kernel = norm ? FlashForwardKernel<true, kDepth> : FlashForwardKernel<false, kDepth>;
  const auto smem = static_cast<std::size_t>(SmemFor(kDepth));
  // Per call: the attribute is per device, and a cached flag would race
  // between threads.
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemFor(kDepth)) !=
      cudaSuccess) {
    return std::unexpected(std::string("FlashAttention: shared memory limit"));
  }
  const dim3 grid(static_cast<unsigned>((q_rows + kBr - 1) / kBr), static_cast<unsigned>(heads));
  constexpr float kLog2e = 1.4426950408889634f;
  kernel<<<grid, kThreads, smem, static_cast<cudaStream_t>(stream)>>>(
      q, q_stride, k, k_stride, v, v_stride, out, out_stride, static_cast<int>(q_rows),
      static_cast<int>(kv_rows), static_cast<int>(heads / kv_heads), scale * kLog2e, kp, vp,
      static_cast<int>(prefix), q_norm, freqs, norm ? eps : 0.0f);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return std::unexpected(std::format("FlashAttention: {}", cudaGetErrorString(error)));
  }
  return {};
}

}  // namespace

Status FlashAttention(const Bf16* q, std::int64_t q_stride, const Bf16* k, std::int64_t k_stride,
                      const Bf16* v, std::int64_t v_stride, Bf16* out, std::int64_t out_stride,
                      std::int64_t q_rows, std::int64_t kv_rows, std::int64_t heads,
                      std::int64_t kv_heads, float scale, Stream stream) {
  return Launch(q, q_stride, k, k_stride, v, v_stride, out, out_stride, q_rows, kv_rows, heads,
                kv_heads, scale, k, v, 0, nullptr, nullptr, 0.0f, stream);
}

Status FlashAttentionPrefixed(const Bf16* q, std::int64_t q_stride, const Bf16* k_prefix,
                              const Bf16* v_prefix, std::int64_t prefix, const Bf16* k,
                              std::int64_t k_stride, const Bf16* v, std::int64_t v_stride,
                              Bf16* out, std::int64_t out_stride, std::int64_t q_rows,
                              std::int64_t kv_rows, std::int64_t heads, std::int64_t kv_heads,
                              float scale, Stream stream, const QueryNorm* q_norm) {
  return Launch(
      q, q_stride, k, k_stride, v, v_stride, out, out_stride, q_rows, kv_rows, heads, kv_heads,
      scale, k_prefix, v_prefix, prefix, q_norm != nullptr ? q_norm->weight : nullptr,
      q_norm != nullptr ? q_norm->freqs : nullptr, q_norm != nullptr ? q_norm->eps : 0.0f, stream);
}

}  // namespace llmp::kernels::image
