// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8's QSA at any depth (llmp_ops.h llmp.qsa.pool, .topk and
// .attn; the fast graph's, qwen38_graph.h): llmpalooza's own kernels, none of
// them GGML's arithmetic bit for bit, each deterministic (no atomics on
// floats, every sum in a fixed order), so a run repeats bit for bit.
//
// - The pool writes each block's key once, when the block completes, as
//   llmp.qsa.prep computes a pooled key (qwen38_graph.cc's reference
//   form pools every block again at every step).
// - The selection scores every complete block against the token's four
//   indexer heads (a thread a block up to kQsaTopKVecRows tokens; past them
//   BF16 tensor-core products, the block keys held in registers while the
//   tokens stream through), into one order-preserving key a block, then
//   selects the width best cells with a byte-wise radix select over tiles
//   of kQsaTopKTile blocks, ties to the lower cell, and the tiles'
//   candidates once more (TensorFold #93's technique, written for llmpalooza).
// - The attention gathers each token's kept cells, 16 at a time, and runs
//   its KV head's query heads over them as one m16n8k16 tile (llama.cpp
//   #28770's gather, one token a warp).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"

namespace llmp::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Refused(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

constexpr unsigned kFull = 0xffffffffU;

__device__ __forceinline__ float WarpSum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(kFull, x, offset, 32);
  }
  return x;
}

// A float's bits in an unsigned order that is the floats' (-inf lowest).
__device__ __forceinline__ std::uint32_t OrderKey(float f) {
  const std::uint32_t b = __float_as_uint(f);
  return (b & 0x80000000U) != 0 ? ~b : (b | 0x80000000U);
}

// The key of a token's own incomplete block (always kept: build_qsa_top_k's
// 1e9 bias), and of a block the token cannot see (never counted).
constexpr std::uint32_t kOwnBlock = 0xffffffffU;
constexpr std::uint32_t kHiddenBlock = 0U;

// The visible cells of block `b` for a token at `pos`: every cell of a
// complete block, the cells up to the token of its own, none later.
__device__ __forceinline__ std::uint32_t Visible(int b, int pos, int ratio) {
  const std::int64_t v =
      static_cast<std::int64_t>(pos) + 1 - (static_cast<std::int64_t>(b) * ratio);
  return v <= 0 ? 0U
                : (v >= ratio ? static_cast<std::uint32_t>(ratio) : static_cast<std::uint32_t>(v));
}

// The key a block's summed score gives, for a token at `pos`.
__device__ __forceinline__ std::uint32_t BlockKey(int b, int pos, int ratio, float score) {
  const int own = (pos + 1) / ratio;
  if (b < own) {
    return OrderKey(score);
  }
  return b == own && (pos + 1) % ratio != 0 ? kOwnBlock : kHiddenBlock;
}

__device__ __forceinline__ float Relu(float x) { return x > 0.0f ? x : 0.0f; }

// ---- PTX: asynchronous copies, matrix loads and products (sm_80 on).

__device__ __forceinline__ void CopyAsync16(void* shared, const void* global, bool valid) {
  const auto to = static_cast<unsigned>(__cvta_generic_to_shared(shared));
  const int bytes = valid ? 16 : 0;  // 0: the 16 bytes are zeroed, nothing read
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(to), "l"(global),
               "r"(bytes));
}
__device__ __forceinline__ void CopyCommit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int kPending>
__device__ __forceinline__ void CopyWait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
}

__device__ __forceinline__ void Ldmatrix4(std::uint32_t (&r)[4], const void* shared) {
  const auto at = static_cast<unsigned>(__cvta_generic_to_shared(shared));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(at));
}
__device__ __forceinline__ void Ldmatrix4Trans(std::uint32_t (&r)[4], const void* shared) {
  const auto at = static_cast<unsigned>(__cvta_generic_to_shared(shared));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(at));
}

// d += a · b, m16n8k16, F32 sums.
__device__ __forceinline__ void MmaF16(float (&d)[4], const std::uint32_t (&a)[4], std::uint32_t b0,
                                       std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void MmaBf16(float (&d)[4], const std::uint32_t (&a)[4],
                                        std::uint32_t b0, std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ std::uint32_t PackHalf2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const std::uint32_t*>(&h);
}

// ---------------------------------------------------------------- pool

constexpr int kPoolSlots = 16;  // d / 32, at most

// A warp a block the chunk completes: its raw keys summed in cell order and
// scaled by 1 / ratio (the reference form's adds and scale), then
// llmp.qsa.prep's norm and rotation at the block's first position, in
// BF16. Blocks past the positions' range, the cache or the table write
// nothing.
__global__ void __launch_bounds__(32)
    QsaPoolKernel(const float* __restrict__ raw, std::int64_t raw_row, int cells,
                  __nv_bfloat16* __restrict__ blocks, int n_blocks,
                  const float* __restrict__ weight, const std::int32_t* __restrict__ positions,
                  int t, int d, int ratio, float eps, float theta_scale) {
  const int lane = static_cast<int>(threadIdx.x);
  const int first = positions[0];
  const int last = positions[t - 1];
  if (first < 0 || last < first) {
    return;
  }
  const int b = (first / ratio) + static_cast<int>(blockIdx.x);
  const std::int64_t end = static_cast<std::int64_t>(last) + 1;
  if (b >= end / ratio || b >= n_blocks ||
      (static_cast<std::int64_t>(b) + 1) * ratio > static_cast<std::int64_t>(cells)) {
    return;
  }
  const int slots = d / 32;
  const float inverse = 1.0f / static_cast<float>(ratio);
  const float* src = raw + (static_cast<std::int64_t>(b) * ratio * raw_row);
  float v[kPoolSlots];
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < kPoolSlots; ++j) {
    v[j] = 0.0f;
    if (j < slots) {
      const int i = (32 * j) + lane;
      float s = src[i];
      for (int k = 1; k < ratio; ++k) {
        s += src[(k * raw_row) + i];
      }
      v[j] = s * inverse;
    }
    sum = fmaf(v[j], v[j], sum);
  }
  const float s = rsqrtf(fmaf(WarpSum(sum), 1.0f / static_cast<float>(d), eps));
#pragma unroll
  for (int j = 0; j < kPoolSlots; ++j) {
    if (j < slots) {
      v[j] = (v[j] * s) * weight[(32 * j) + lane];
    }
  }
  const float theta = static_cast<float>(static_cast<std::int64_t>(b) * ratio) *
                      powf(theta_scale, static_cast<float>(lane));
  const float c = cosf(theta);
  const float sn = sinf(theta);
  const float x0 = v[0];
  const float x1 = v[1];
  v[0] = (x0 * c) - (x1 * sn);
  v[1] = (x0 * sn) + (x1 * c);
  __nv_bfloat16* dst = blocks + (static_cast<std::int64_t>(b) * d);
#pragma unroll
  for (int j = 0; j < kPoolSlots; ++j) {
    if (j < slots) {
      dst[(32 * j) + lane] = __float2bfloat16_rn(v[j]);
    }
  }
}

// ---------------------------------------------------------------- scores

constexpr int kQueryValues = static_cast<int>(kQsaIndexDim * kQsaIndexHeads);  // a token's

// The queries in BF16, an element a thread.
__global__ void QsaQueryBf16Kernel(const float* __restrict__ q, __nv_bfloat16* __restrict__ out,
                                   std::int64_t n) {
  const std::int64_t i = (static_cast<std::int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (i < n) {
    out[i] = __float2bfloat16_rn(q[i]);
  }
}

// Up to kQsaTopKVecRows tokens: a thread a block, its key read once for
// every token (their BF16 queries in shared memory as floats), each head's
// sum in dimension order, the heads' relu scores summed in head order.
constexpr int kVecThreads = 128;
__global__ void __launch_bounds__(kVecThreads)
    QsaScoreVecKernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ blocks,
                      const std::int32_t* __restrict__ positions, std::uint32_t* __restrict__ keys,
                      int n_blocks, int t, int ratio) {
  extern __shared__ float query[];  // [t][4][128]
  for (int i = static_cast<int>(threadIdx.x); i < t * kQueryValues; i += kVecThreads) {
    query[i] = __bfloat162float(q[i]);
  }
  __syncthreads();
  const int b = (static_cast<int>(blockIdx.x) * kVecThreads) + static_cast<int>(threadIdx.x);
  if (b >= n_blocks) {
    return;
  }
  constexpr int kRows = static_cast<int>(kQsaTopKVecRows);
  float acc[kRows][kQsaIndexHeads];
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
#pragma unroll
    for (int h = 0; h < kQsaIndexHeads; ++h) {
      acc[r][h] = 0.0f;
    }
  }
  const auto* row =
      reinterpret_cast<const uint4*>(blocks + (static_cast<std::int64_t>(b) * kQsaIndexDim));
  for (int c = 0; c < static_cast<int>(kQsaIndexDim) / 8; ++c) {
    const uint4 raw = row[c];
    const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
    float k[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(pairs[i]);
      k[2 * i] = f.x;
      k[(2 * i) + 1] = f.y;
    }
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      if (r < t) {
#pragma unroll
        for (int h = 0; h < kQsaIndexHeads; ++h) {
          const float* qh = query + (((r * kQsaIndexHeads) + h) * kQsaIndexDim) + (c * 8);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            acc[r][h] = fmaf(qh[i], k[i], acc[r][h]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    if (r < t) {
      float s = Relu(acc[r][0]);
#pragma unroll
      for (int h = 1; h < kQsaIndexHeads; ++h) {
        s += Relu(acc[r][h]);
      }
      keys[(static_cast<std::int64_t>(r) * n_blocks) + b] = BlockKey(b, positions[r], ratio, s);
    }
  }
}

// Past kQsaTopKVecRows tokens: a block of 4 warps holds 128 block keys, a
// warp's 32 in registers (the product's B operand), and streams its tokens
// through in tiles of 16 (A: their 64 heads, head-major, so that a thread
// holds one token's four heads), double-buffered; BF16 products with F32
// sums. Blocks no token of a tile sees completely are not multiplied.
constexpr int kScoreStride = static_cast<int>(kQsaIndexDim) + 8;  // BF16 a shared row
constexpr int kScoreBlocks = 128;
constexpr int kScoreTokens = 16;
constexpr int kScoreSteps = static_cast<int>(kQsaIndexDim) / 16;
constexpr std::size_t kScoreShared =
    static_cast<std::size_t>(kScoreBlocks) * kScoreStride * sizeof(__nv_bfloat16);
static_assert(kScoreShared >= 2 * 64 * kScoreStride * sizeof(__nv_bfloat16));

__global__ void __launch_bounds__(128)
    QsaScoreMmaKernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ blocks,
                      const std::int32_t* __restrict__ positions, std::uint32_t* __restrict__ keys,
                      int n_blocks, int t, int ratio, int tokens_per_block) {
  extern __shared__ __align__(16) __nv_bfloat16 staged[];
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int g = lane / 4;
  const int tig = lane % 4;
  const int nb0 = static_cast<int>(blockIdx.x) * kScoreBlocks;
  const int r_begin = static_cast<int>(blockIdx.y) * tokens_per_block;
  const int r_end = min(t, r_begin + tokens_per_block);
  if (r_begin >= r_end) {
    return;
  }
  for (int i = tid; i < kScoreBlocks * 16; i += 128) {
    const int b = nb0 + (i / 16);
    const bool valid = b < n_blocks;
    CopyAsync16(
        staged + ((i / 16) * kScoreStride) + ((i % 16) * 8),
        valid ? blocks + (static_cast<std::int64_t>(b) * kQsaIndexDim) + ((i % 16) * 8) : blocks,
        valid);
  }
  CopyCommit();
  CopyWait<0>();
  __syncthreads();
  // The warp's 32 blocks (4 n-tiles), each k-step's fragments.
  std::uint32_t bf[kScoreSteps][4][2];
#pragma unroll
  for (int kk = 0; kk < kScoreSteps; ++kk) {
#pragma unroll
    for (int np = 0; np < 2; ++np) {
      std::uint32_t x[4];
      Ldmatrix4(x, staged +
                       (((warp * 32) + (np * 16) + (lane % 8) + ((lane / 16) * 8)) * kScoreStride) +
                       (kk * 16) + (((lane / 8) % 2) * 8));
      bf[kk][2 * np][0] = x[0];
      bf[kk][2 * np][1] = x[1];
      bf[kk][(2 * np) + 1][0] = x[2];
      bf[kk][(2 * np) + 1][1] = x[3];
    }
  }
  __syncthreads();  // the staging holds the token tiles from here
  __nv_bfloat16* tiles[2] = {staged, staged + (64 * kScoreStride)};
  const auto load = [&](int r0, __nv_bfloat16* into) {
    for (int i = tid; i < 64 * 16; i += 128) {
      const int row = i / 16;  // head · 16 + token
      const int r = r0 + (row % 16);
      const bool valid = r < r_end;
      CopyAsync16(into + (row * kScoreStride) + ((i % 16) * 8),
                  valid ? q +
                              (((static_cast<std::int64_t>(r) * kQsaIndexHeads) + (row / 16)) *
                               kQsaIndexDim) +
                              ((i % 16) * 8)
                        : q,
                  valid);
    }
    CopyCommit();
  };
  const int wb0 = nb0 + (warp * 32);  // the warp's first block
  int stage = 0;
  load(r_begin, tiles[0]);
  for (int r0 = r_begin; r0 < r_end; r0 += kScoreTokens) {
    const bool more = r0 + kScoreTokens < r_end;
    if (more) {
      load(r0 + kScoreTokens, tiles[stage ^ 1]);
      CopyWait<1>();
    } else {
      CopyWait<0>();
    }
    __syncthreads();
    const __nv_bfloat16* a = tiles[stage];
    float acc[kQsaIndexHeads][4][4];
#pragma unroll
    for (int h = 0; h < kQsaIndexHeads; ++h) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          acc[h][nt][e] = 0.0f;
        }
      }
    }
    // The tile's last token sees the most blocks.
    const int last = min(r0 + kScoreTokens, r_end) - 1;
    if (wb0 < (positions[last] + 1) / ratio) {
#pragma unroll
      for (int kk = 0; kk < kScoreSteps; ++kk) {
#pragma unroll
        for (int h = 0; h < kQsaIndexHeads; ++h) {
          std::uint32_t af[4];
          Ldmatrix4(af, a + (((h * 16) + (lane % 8) + (((lane / 8) % 2) * 8)) * kScoreStride) +
                            (kk * 16) + ((lane / 16) * 8));
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            MmaBf16(acc[h][nt], af, bf[kk][nt][0], bf[kk][nt][1]);
          }
        }
      }
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {  // tokens g and g + 8
      const int r = r0 + g + (8 * e);
      if (r >= r_end) {
        continue;
      }
      const int pos = positions[r];
      std::uint32_t* out = keys + (static_cast<std::int64_t>(r) * n_blocks);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int b = wb0 + (nt * 8) + (2 * tig) + j;
          if (b >= n_blocks) {
            continue;
          }
          float s = Relu(acc[0][nt][(2 * e) + j]);
#pragma unroll
          for (int h = 1; h < kQsaIndexHeads; ++h) {
            s += Relu(acc[h][nt][(2 * e) + j]);
          }
          out[b] = BlockKey(b, pos, ratio, s);
        }
      }
    }
    __syncthreads();  // before the next load writes this tile's buffer
    stage ^= 1;
  }
}

// ---------------------------------------------------------------- selection

// A block of 256 threads selects a token's tile (or its candidates): each
// thread owns a run of consecutive blocks, which the emission's scans need
// in block order. Keys sit in shared memory with a word of padding every
// 32, so a thread's run and a warp's lanes at one step read distinct banks.
// Each warp counts into a histogram of its own, a thread adding once per
// run of equal digits: the keys crowd a few digits (scores of a binade
// share their high bytes), and adds to one address serialize.
constexpr int kSelectThreads = 256;
constexpr int kSelectWarps = kSelectThreads / 32;
constexpr int kTileItems = static_cast<int>(kQsaTopKTile) / kSelectThreads;         // 32
constexpr int kMergeItems = static_cast<int>(kQsaTopKCandidates) / kSelectThreads;  // 32
static_assert(kTileItems * kSelectThreads == kQsaTopKTile);
static_assert(kMergeItems * kSelectThreads == kQsaTopKCandidates);

__device__ __forceinline__ int Padded(int i) { return i + (i >> 5); }
constexpr int kTileWords = static_cast<int>(kQsaTopKTile + (kQsaTopKTile >> 5) + 1);
constexpr int kMergeWords = static_cast<int>(kQsaTopKCandidates + (kQsaTopKCandidates >> 5) + 1);

struct SelectShared {
  std::uint32_t hist[kSelectWarps][256];  // each warp's; the sums in hist[0]
  std::uint32_t warp[kSelectWarps];
  std::uint32_t chosen[2];
};

// An exclusive scan of one value a thread over the block, in thread order;
// `total` the sum.
__device__ std::uint32_t BlockScan(std::uint32_t v, SelectShared& s, std::uint32_t& total) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % 32;
  const int warp = tid / 32;
  std::uint32_t x = v;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const std::uint32_t y = __shfl_up_sync(kFull, x, offset);
    if (lane >= offset) {
      x += y;
    }
  }
  if (lane == 31) {
    s.warp[warp] = x;
  }
  __syncthreads();
  std::uint32_t before = 0;
  total = 0;
#pragma unroll
  for (int w = 0; w < kSelectWarps; ++w) {
    before += w < warp ? s.warp[w] : 0U;
    total += s.warp[w];
  }
  __syncthreads();
  return (x - v) + before;
}

// The width-th best visible cell's key among `n` blocks (keys through
// `key`, each block's visible cells its weight), by four byte-wise passes:
// the key K and how many cells of key K are kept (all of them, and the key
// 0, when fewer than `want` cells are visible). Each pass's digit is the
// highest whose cells, with those of the digits above, reach what remains.
template <int kItems, typename Key, typename Weight>
__device__ void RadixSelect(const Key& key, const Weight& weight, int n, std::uint32_t want,
                            SelectShared& s, std::uint32_t& prefix, std::uint32_t& remaining) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % 32;
  const int warp = tid / 32;
  const int base = tid * kItems;
  prefix = 0;
  remaining = want;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = lane; b < 256; b += 32) {
      s.hist[warp][b] = 0;
    }
    __syncwarp();
    std::uint32_t last = 0;
    std::uint32_t run = 0;
#pragma unroll 4
    for (int j = 0; j < kItems; ++j) {
      const int i = base + j;
      if (i >= n) {
        break;
      }
      const std::uint32_t w = weight(i);
      const std::uint32_t k = key(i);
      if (w == 0 || (shift != 24 && (k >> (static_cast<unsigned>(shift) + 8U)) != prefix)) {
        continue;
      }
      const std::uint32_t bin = (k >> static_cast<unsigned>(shift)) & 255U;
      if (run != 0 && bin != last) {
        atomicAdd(&s.hist[warp][last], run);
        run = 0;
      }
      last = bin;
      run += w;
    }
    if (run != 0) {
      atomicAdd(&s.hist[warp][last], run);
    }
    __syncthreads();
    // The warps' counts summed (integers: in any order the same).
    for (int b = tid; b < 256; b += kSelectThreads) {
      std::uint32_t sum = 0;
#pragma unroll
      for (int w = 0; w < kSelectWarps; ++w) {
        sum += s.hist[w][b];
      }
      s.hist[0][b] = sum;
    }
    __syncthreads();
    if (warp == 0) {
      // Lane L holds digits 255 - 8L down to 248 - 8L.
      std::uint32_t mine = 0;
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        mine += s.hist[0][255 - (8 * lane) - d];
      }
      std::uint32_t upto = mine;
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const std::uint32_t y = __shfl_up_sync(kFull, upto, offset);
        if (lane >= offset) {
          upto += y;
        }
      }
      const std::uint32_t reach = __ballot_sync(kFull, upto >= remaining);
      if (reach == 0) {
        // Fewer cells than wanted: every visible cell (digit 0).
        if (lane == 31) {
          s.chosen[0] = prefix << 8U;
          s.chosen[1] = remaining - (upto - s.hist[0][0]);
        }
      } else if (lane == __ffs(static_cast<int>(reach)) - 1) {
        std::uint32_t above = upto - mine;
        int digit = 255 - (8 * lane);
        for (int d = 0; d < 8; ++d, --digit) {
          if (digit == 0 || above + s.hist[0][digit] >= remaining) {
            break;
          }
          above += s.hist[0][digit];
        }
        s.chosen[0] = (prefix << 8U) | static_cast<std::uint32_t>(digit);
        s.chosen[1] = remaining - above;
      }
    }
    __syncthreads();
    prefix = s.chosen[0];
    remaining = s.chosen[1];
    __syncthreads();
  }
}

// Each block's kept cells, in block order: every visible cell of a block
// above K, of K's blocks the first `remaining` cells in cell order. A
// thread's kItems consecutive blocks; kept[] and the scans' offsets out.
template <int kItems, typename Key, typename Weight>
__device__ void Kept(const Key& key, const Weight& weight, int n, std::uint32_t prefix,
                     std::uint32_t remaining, SelectShared& s, std::uint32_t (&kept)[kItems],
                     std::uint32_t& cells_before, std::uint32_t& cells_total,
                     std::uint32_t& blocks_before, std::uint32_t& blocks_total) {
  const int base = static_cast<int>(threadIdx.x) * kItems;
  std::uint32_t equal = 0;
#pragma unroll 4
  for (int j = 0; j < kItems; ++j) {
    const int i = base + j;
    if (i < n && key(i) == prefix) {
      equal += weight(i);
    }
  }
  std::uint32_t total = 0;
  std::uint32_t equal_before = BlockScan(equal, s, total);
  std::uint32_t cells = 0;
  std::uint32_t blocks = 0;
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    const int i = base + j;
    kept[j] = 0;
    if (i >= n) {
      continue;
    }
    const std::uint32_t w = weight(i);
    const std::uint32_t k = key(i);
    if (w == 0) {
      continue;
    }
    if (k > prefix) {
      kept[j] = w;
    } else if (k == prefix) {
      const std::uint32_t left = equal_before < remaining ? remaining - equal_before : 0U;
      kept[j] = min(w, left);
      equal_before += w;
    }
    cells += kept[j];
    blocks += kept[j] != 0 ? 1U : 0U;
  }
  cells_before = BlockScan(cells, s, cells_total);
  blocks_before = BlockScan(blocks, s, blocks_total);
}

// A token's kept cells, ascending, then -1 to the row's end.
template <int kItems, typename Block>
__device__ void WriteCells(const std::uint32_t (&kept)[kItems], const Block& block,
                           std::uint32_t before, std::uint32_t total, int ratio,
                           std::int32_t* __restrict__ out, int row) {
  const int base = static_cast<int>(threadIdx.x) * kItems;
  std::uint32_t at = before;
#pragma unroll
  for (int j = 0; j < kItems; ++j) {
    for (std::uint32_t c = 0; c < kept[j]; ++c) {
      if (at < static_cast<std::uint32_t>(row)) {
        out[at] = (block(base + j) * ratio) + static_cast<int>(c);
      }
      ++at;
    }
  }
  for (int i = static_cast<int>(min(total, static_cast<std::uint32_t>(row))) +
               static_cast<int>(threadIdx.x);
       i < row; i += kSelectThreads) {
    out[i] = -1;
  }
}

// A token's tile of blocks (a block of threads each): its keys into shared
// memory, the radix select, then either the token's cells (one tile) or the
// tile's candidates (key, block) in block order.
template <bool kFinal>
__global__ void __launch_bounds__(kSelectThreads)
    QsaSelectTileKernel(const std::uint32_t* __restrict__ keys,
                        const std::int32_t* __restrict__ positions, std::int32_t* __restrict__ out,
                        uint2* __restrict__ pairs, std::uint32_t* __restrict__ counts, int n_blocks,
                        int width, int ratio, int row, int candidates) {
  __shared__ std::uint32_t key[kTileWords];
  __shared__ SelectShared s;
  const int tile = static_cast<int>(blockIdx.x);
  const int r = static_cast<int>(blockIdx.y);
  const int tiles = static_cast<int>(gridDim.x);
  const int b0 = tile * static_cast<int>(kQsaTopKTile);
  const int n = min(static_cast<int>(kQsaTopKTile), n_blocks - b0);
  const int pos = positions[r];
  const std::uint32_t* src = keys + (static_cast<std::int64_t>(r) * n_blocks) + b0;
  for (int i = static_cast<int>(threadIdx.x); i < n; i += kSelectThreads) {
    key[Padded(i)] = src[i];
  }
  __syncthreads();
  const auto at = [&](int i) { return key[Padded(i)]; };
  const auto weight = [&](int i) { return Visible(b0 + i, pos, ratio); };
  std::uint32_t prefix = 0;
  std::uint32_t remaining = 0;
  RadixSelect<kTileItems>(at, weight, n, static_cast<std::uint32_t>(width), s, prefix, remaining);
  std::uint32_t kept[kTileItems];
  std::uint32_t cells_before = 0;
  std::uint32_t cells_total = 0;
  std::uint32_t blocks_before = 0;
  std::uint32_t blocks_total = 0;
  Kept(at, weight, n, prefix, remaining, s, kept, cells_before, cells_total, blocks_before,
       blocks_total);
  if constexpr (kFinal) {
    WriteCells(
        kept, [&](int i) { return b0 + i; }, cells_before, cells_total, ratio,
        out + (static_cast<std::int64_t>(r) * row), row);
  } else {
    uint2* dst = pairs + (((static_cast<std::int64_t>(r) * tiles) + tile) * candidates);
    const int base = static_cast<int>(threadIdx.x) * kTileItems;
    std::uint32_t slot = blocks_before;
#pragma unroll
    for (int j = 0; j < kTileItems; ++j) {
      if (kept[j] != 0) {
        if (slot < static_cast<std::uint32_t>(candidates)) {
          dst[slot] = make_uint2(at(base + j), static_cast<std::uint32_t>(b0 + base + j));
        }
        ++slot;
      }
    }
    if (threadIdx.x == 0) {
      counts[(static_cast<std::int64_t>(r) * tiles) + tile] =
          min(blocks_total, static_cast<std::uint32_t>(candidates));
    }
  }
}

// A token's tiles' candidates, in tile order (so in block order), selected
// once more into its cells.
struct MergeShared {
  std::uint32_t key[kMergeWords];
  std::uint32_t block[kMergeWords];
  std::uint32_t offset[kQsaTopKMaxTiles + 1];
  SelectShared s;
};
constexpr std::size_t kMergeShared = sizeof(MergeShared);

__global__ void __launch_bounds__(kSelectThreads)
    QsaSelectMergeKernel(const uint2* __restrict__ pairs, const std::uint32_t* __restrict__ counts,
                         const std::int32_t* __restrict__ positions, std::int32_t* __restrict__ out,
                         int tiles, int width, int ratio, int row, int candidates) {
  extern __shared__ __align__(16) unsigned char merge_shared[];
  auto& m = *reinterpret_cast<MergeShared*>(merge_shared);
  const int r = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const std::uint32_t* count = counts + (static_cast<std::int64_t>(r) * tiles);
  if (tid == 0) {
    std::uint32_t at = 0;
    for (int i = 0; i < tiles; ++i) {
      m.offset[i] = at;
      at += min(count[i], static_cast<std::uint32_t>(candidates));
    }
    m.offset[tiles] = min(at, static_cast<std::uint32_t>(kQsaTopKCandidates));
  }
  __syncthreads();
  for (int i = 0; i < tiles; ++i) {
    const uint2* src = pairs + (((static_cast<std::int64_t>(r) * tiles) + i) * candidates);
    const std::uint32_t n = min(count[i], static_cast<std::uint32_t>(candidates));
    for (std::uint32_t j = static_cast<std::uint32_t>(tid); j < n; j += kSelectThreads) {
      const std::uint32_t at = m.offset[i] + j;
      if (at < static_cast<std::uint32_t>(kQsaTopKCandidates)) {
        const uint2 p = src[j];
        m.key[Padded(static_cast<int>(at))] = p.x;
        m.block[Padded(static_cast<int>(at))] = p.y;
      }
    }
  }
  __syncthreads();
  const int n = static_cast<int>(m.offset[tiles]);
  const int pos = positions[r];
  const auto at = [&](int i) { return m.key[Padded(i)]; };
  const auto block = [&](int i) { return static_cast<int>(m.block[Padded(i)]); };
  const auto weight = [&](int i) { return Visible(block(i), pos, ratio); };
  std::uint32_t prefix = 0;
  std::uint32_t remaining = 0;
  RadixSelect<kMergeItems>(at, weight, n, static_cast<std::uint32_t>(width), m.s, prefix,
                           remaining);
  std::uint32_t kept[kMergeItems];
  std::uint32_t cells_before = 0;
  std::uint32_t cells_total = 0;
  std::uint32_t blocks_before = 0;
  std::uint32_t blocks_total = 0;
  Kept(at, weight, n, prefix, remaining, m.s, kept, cells_before, cells_total, blocks_before,
       blocks_total);
  WriteCells(kept, block, cells_before, cells_total, ratio,
             out + (static_cast<std::int64_t>(r) * row), row);
}

// ---------------------------------------------------------------- attention

constexpr int kGather = 16;
constexpr int kHead = static_cast<int>(kQsaAttnHead);
constexpr int kAttnStride = kHead + 8;  // halves a shared row: 528 bytes
constexpr int kHeadSteps = kHead / 16;
constexpr std::size_t kAttnShared = static_cast<std::size_t>(2) * kGather * kAttnStride * 2;

// A warp a work item: token r's KV head kh (its query heads the rows of one
// 16-row tile) over one share of its cells. Each gather of 16 cells loads K
// and V rows into shared memory (cp.async, the next gather's K while this
// one's V is used and its V while the next K is), S = Q K^T, the online
// softmax per head, O += P V. One share writes the output; several write
// partial results that QsaAttnCombineKernel sums in share order.
__global__ void __launch_bounds__(32)
    QsaAttnKernel(const float* __restrict__ q, const __half* __restrict__ k,
                  const __half* __restrict__ v, std::int64_t kv_row, int cells,
                  const std::int32_t* __restrict__ idx, int row, float* __restrict__ out,
                  float* __restrict__ partial, int heads, int kv_heads, int shares, float scale) {
  extern __shared__ __align__(16) __half gathered[];
  __half* kbuf = gathered;
  __half* vbuf = gathered + (kGather * kAttnStride);
  const int lane = static_cast<int>(threadIdx.x);
  const int g = lane / 4;
  const int tig = lane % 4;
  const int item = static_cast<int>(blockIdx.x);
  const int share = item % shares;
  const int kh = (item / shares) % kv_heads;
  const int r = item / (shares * kv_heads);
  const int group = heads / kv_heads;
  const int gathers = row / kGather;
  const int per = (gathers + shares - 1) / shares;
  const int c0 = share * per;
  const int c1 = min(gathers, c0 + per);
  const std::int32_t* list = idx + (static_cast<std::int64_t>(r) * row);

  // The query heads, scaled, as F16 A fragments for every k-step.
  const float* qrow =
      q +
      (((static_cast<std::int64_t>(r) * heads) + (static_cast<std::int64_t>(kh) * group)) * kHead);
  const auto qpair = [&](int h, int col) -> std::uint32_t {
    if (h >= group) {
      return 0U;
    }
    const float2 x = *reinterpret_cast<const float2*>(qrow + (h * kHead) + col);
    return PackHalf2(x.x * scale, x.y * scale);
  };
  std::uint32_t qa[kHeadSteps][4];
#pragma unroll
  for (int kk = 0; kk < kHeadSteps; ++kk) {
    const int col = (kk * 16) + (2 * tig);
    qa[kk][0] = qpair(g, col);
    qa[kk][1] = qpair(g + 8, col);
    qa[kk][2] = qpair(g, col + 8);
    qa[kk][3] = qpair(g + 8, col + 8);
  }
  float o[kHead / 8][4];
#pragma unroll
  for (int nt = 0; nt < kHead / 8; ++nt) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      o[nt][e] = 0.0f;
    }
  }
  float m0 = -INFINITY;
  float m1 = -INFINITY;
  float l0 = 0.0f;
  float l1 = 0.0f;
  const std::int64_t head_offset = static_cast<std::int64_t>(kh) * kHead;
  // One gather's rows of `base` (K or V) into `buf`; a cell outside the
  // cache (the -1 past a token's cells) is zeros.
  const auto gather = [&](int c, __half* buf, const __half* base) {
    const int mine = lane < kGather ? list[(c * kGather) + lane] : -1;
#pragma unroll
    for (int i = 0; i < kGather; ++i) {
      const int cell = __shfl_sync(kFull, mine, i);
      const bool valid = cell >= 0 && cell < cells;
      CopyAsync16(buf + (i * kAttnStride) + (lane * 8),
                  valid
                      ? base + (static_cast<std::int64_t>(cell) * kv_row) + head_offset + (lane * 8)
                      : base,
                  valid);
    }
    CopyCommit();
  };
  if (c0 < c1) {
    gather(c0, kbuf, k);
    gather(c0, vbuf, v);
  }
  for (int c = c0; c < c1; ++c) {
    const int mine = lane < kGather ? list[(c * kGather) + lane] : -1;
    // A token's cells come first, then -1: a gather that starts with -1
    // ends it.
    if (__shfl_sync(kFull, mine, 0) < 0) {
      break;
    }
    const bool more = c + 1 < c1 && list[(c + 1) * kGather] >= 0;
    CopyWait<1>();  // this gather's K (its V may still be in flight)
    __syncwarp();
    float sc[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        sc[nt][e] = 0.0f;
      }
    }
#pragma unroll
    for (int kk = 0; kk < kHeadSteps; ++kk) {
      std::uint32_t b[4];
      Ldmatrix4(b, kbuf + (((lane % 8) + ((lane / 16) * 8)) * kAttnStride) + (kk * 16) +
                       (((lane / 8) % 2) * 8));
      MmaF16(sc[0], qa[kk], b[0], b[1]);
      MmaF16(sc[1], qa[kk], b[2], b[3]);
    }
    __syncwarp();
    if (more) {
      gather(c + 1, kbuf, k);
    }
    // Cells past the token's are not attended.
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int cell = __shfl_sync(kFull, mine, (nt * 8) + (2 * tig) + j);
        if (cell < 0 || cell >= cells) {
          sc[nt][j] = -INFINITY;
          sc[nt][2 + j] = -INFINITY;
        }
      }
    }
    // The online softmax of rows g (sc[.][0..1]) and g + 8 (sc[.][2..3]).
    float x0 = fmaxf(fmaxf(sc[0][0], sc[0][1]), fmaxf(sc[1][0], sc[1][1]));
    float x1 = fmaxf(fmaxf(sc[0][2], sc[0][3]), fmaxf(sc[1][2], sc[1][3]));
    x0 = fmaxf(x0, __shfl_xor_sync(kFull, x0, 1));
    x0 = fmaxf(x0, __shfl_xor_sync(kFull, x0, 2));
    x1 = fmaxf(x1, __shfl_xor_sync(kFull, x1, 1));
    x1 = fmaxf(x1, __shfl_xor_sync(kFull, x1, 2));
    const float n0 = fmaxf(m0, x0);
    const float n1 = fmaxf(m1, x1);
    const float a0 = n0 == -INFINITY ? 1.0f : __expf(m0 - n0);
    const float a1 = n1 == -INFINITY ? 1.0f : __expf(m1 - n1);
    float p[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      p[nt][0] = n0 == -INFINITY ? 0.0f : __expf(sc[nt][0] - n0);
      p[nt][1] = n0 == -INFINITY ? 0.0f : __expf(sc[nt][1] - n0);
      p[nt][2] = n1 == -INFINITY ? 0.0f : __expf(sc[nt][2] - n1);
      p[nt][3] = n1 == -INFINITY ? 0.0f : __expf(sc[nt][3] - n1);
    }
    float s0 = (p[0][0] + p[0][1]) + (p[1][0] + p[1][1]);
    float s1 = (p[0][2] + p[0][3]) + (p[1][2] + p[1][3]);
    s0 += __shfl_xor_sync(kFull, s0, 1);
    s0 += __shfl_xor_sync(kFull, s0, 2);
    s1 += __shfl_xor_sync(kFull, s1, 1);
    s1 += __shfl_xor_sync(kFull, s1, 2);
    l0 = (l0 * a0) + s0;
    l1 = (l1 * a1) + s1;
    m0 = n0;
    m1 = n1;
#pragma unroll
    for (int nt = 0; nt < kHead / 8; ++nt) {
      o[nt][0] *= a0;
      o[nt][1] *= a0;
      o[nt][2] *= a1;
      o[nt][3] *= a1;
    }
    const std::uint32_t pa[4] = {PackHalf2(p[0][0], p[0][1]), PackHalf2(p[0][2], p[0][3]),
                                 PackHalf2(p[1][0], p[1][1]), PackHalf2(p[1][2], p[1][3])};
    if (more) {
      CopyWait<1>();  // this gather's V (the next K may still be in flight)
    } else {
      CopyWait<0>();
    }
    __syncwarp();
#pragma unroll
    for (int np = 0; np < kHead / 16; ++np) {
      std::uint32_t b[4];
      Ldmatrix4Trans(b, vbuf + (((lane % 8) + (((lane / 8) % 2) * 8)) * kAttnStride) + (np * 16) +
                            ((lane / 16) * 8));
      MmaF16(o[2 * np], pa, b[0], b[1]);
      MmaF16(o[(2 * np) + 1], pa, b[2], b[3]);
    }
    __syncwarp();
    if (more) {
      gather(c + 1, vbuf, v);
    }
  }
  CopyWait<0>();
  if (shares == 1) {
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f;
    const float i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    float* dst =
        out + (((static_cast<std::int64_t>(r) * heads) + (static_cast<std::int64_t>(kh) * group)) *
               kHead);
#pragma unroll
    for (int nt = 0; nt < kHead / 8; ++nt) {
      const int col = (nt * 8) + (2 * tig);
      if (g < group) {
        *reinterpret_cast<float2*>(dst + (g * kHead) + col) =
            make_float2(o[nt][0] * i0, o[nt][1] * i0);
      }
      if (g + 8 < group) {
        *reinterpret_cast<float2*>(dst + ((g + 8) * kHead) + col) =
            make_float2(o[nt][2] * i1, o[nt][3] * i1);
      }
    }
    return;
  }
  // The share's partial result: each head's max and sum, then its output.
  float* part = partial + (static_cast<std::int64_t>(item) * group * (2 + kHead));
  if (tig == 0) {
    if (g < group) {
      part[g] = m0;
      part[group + g] = l0;
    }
    if (g + 8 < group) {
      part[g + 8] = m1;
      part[group + g + 8] = l1;
    }
  }
  float* rows = part + (2 * group);
#pragma unroll
  for (int nt = 0; nt < kHead / 8; ++nt) {
    const int col = (nt * 8) + (2 * tig);
    if (g < group) {
      *reinterpret_cast<float2*>(rows + (g * kHead) + col) = make_float2(o[nt][0], o[nt][1]);
    }
    if (g + 8 < group) {
      *reinterpret_cast<float2*>(rows + ((g + 8) * kHead) + col) = make_float2(o[nt][2], o[nt][3]);
    }
  }
}

// A token's query head a block, a thread a value column: its shares'
// outputs rescaled to their common max and summed in share order.
__global__ void __launch_bounds__(kHead)
    QsaAttnCombineKernel(const float* __restrict__ partial, float* __restrict__ out, int heads,
                         int kv_heads, int shares) {
  const int col = static_cast<int>(threadIdx.x);
  const int q = static_cast<int>(blockIdx.x) % heads;  // the query head
  const int r = static_cast<int>(blockIdx.x) / heads;
  const int group = heads / kv_heads;
  const int kh = q / group;
  const int h = q % group;
  const std::int64_t stride = static_cast<std::int64_t>(group) * (2 + kHead);
  const float* first =
      partial + ((((static_cast<std::int64_t>(r) * kv_heads) + kh) * shares) * stride);
  float m = -INFINITY;
  for (int s = 0; s < shares; ++s) {
    m = fmaxf(m, first[(s * stride) + h]);
  }
  float l = 0.0f;
  float acc = 0.0f;
  if (m != -INFINITY) {
    for (int s = 0; s < shares; ++s) {
      const float* part = first + (s * stride);
      const float ms = part[h];
      if (ms == -INFINITY) {
        continue;
      }
      const float w = __expf(ms - m);
      l += part[group + h] * w;
      acc += part[(2 * group) + (h * kHead) + col] * w;
    }
  }
  out[(((static_cast<std::int64_t>(r) * heads) + q) * kHead) + col] = l > 0.0f ? acc / l : 0.0f;
}

unsigned Blocks(std::int64_t items, int threads) {
  return static_cast<unsigned>((items + threads - 1) / threads);
}

}  // namespace

std::expected<void, KernelFailure> RunQsaPool(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckQsaPool(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* raw = node->src[0];
    const ggml_tensor* blocks = node->src[1];
    const int d = LlmpOpInt(node, 0);
    const int ratio = LlmpOpInt(node, 1);
    const int t = LlmpOpInt(node, 2);
    QsaPoolKernel<<<static_cast<unsigned>((t + ratio - 1) / ratio), 32, 0, context.stream()>>>(
        static_cast<const float*>(raw->data), static_cast<std::int64_t>(raw->nb[1] / sizeof(float)),
        static_cast<int>(raw->ne[1]), static_cast<__nv_bfloat16*>(blocks->data),
        static_cast<int>(blocks->ne[1]), static_cast<const float*>(node->src[2]->data),
        static_cast<const std::int32_t*>(node->src[3]->data), t, d, ratio, LlmpOpFloat(node, 3),
        LlmpOpFloat(node, 4));
  });
}

std::expected<void, KernelFailure> RunQsaTopK(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckQsaTopK(node); !checked) {
    return checked;
  }
  // The second selection's shared memory is past the default 48 KiB.
  static const cudaError_t merge_shared =
      cudaFuncSetAttribute(QsaSelectMergeKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(kMergeShared));
  if (merge_shared != cudaSuccess) {
    return Refused(std::string("the selection's shared memory: ") +
                   cudaGetErrorString(merge_shared));
  }
  const std::uint64_t scratch = PlanQsaTopK(node);
  return launch.Run(base::Bytes(scratch), [node, scratch](ggml_backend_cuda_context& context) {
    const int n_blocks = LlmpOpInt(node, 0);
    const int width = LlmpOpInt(node, 1);
    const int ratio = LlmpOpInt(node, 2);
    const int t = LlmpOpInt(node, 3);
    const QsaTopKLayout layout{.t = t, .n_blocks = n_blocks, .width = width, .ratio = ratio};
    const auto row = static_cast<int>(QsaTopKRow(width));
    const auto tiles = static_cast<int>(layout.tiles());
    const auto candidates = static_cast<int>(layout.candidates());
    ggml_cuda_pool_alloc<std::uint8_t> pool(context.pool(), scratch);
    std::uint8_t* base = pool.get();
    auto* keys = reinterpret_cast<std::uint32_t*>(base + QsaTopKLayout::keys());
    auto* query = reinterpret_cast<__nv_bfloat16*>(base + layout.query());
    auto* pairs = reinterpret_cast<uint2*>(base + layout.pairs());
    auto* counts = reinterpret_cast<std::uint32_t*>(base + layout.counts());
    const auto* positions = static_cast<const std::int32_t*>(node->src[2]->data);
    const auto* blocks = static_cast<const __nv_bfloat16*>(node->src[1]->data);
    auto* out = static_cast<std::int32_t*>(node->data);
    const cudaStream_t stream = context.stream();
    const std::int64_t values = static_cast<std::int64_t>(t) * kQueryValues;
    QsaQueryBf16Kernel<<<Blocks(values, 256), 256, 0, stream>>>(
        static_cast<const float*>(node->src[0]->data), query, values);
    if (t <= kQsaTopKVecRows) {
      QsaScoreVecKernel<<<Blocks(n_blocks, kVecThreads), kVecThreads,
                          static_cast<std::size_t>(values) * sizeof(float), stream>>>(
          query, blocks, positions, keys, n_blocks, t, ratio);
    } else {
      // Enough blocks for every multiprocessor twice: the tokens split
      // when the blocks are few.
      const unsigned columns = Blocks(n_blocks, kScoreBlocks);
      const unsigned groups =
          std::max(1U, std::min(Blocks(t, kScoreTokens), Blocks(192, static_cast<int>(columns))));
      const int per = static_cast<int>(Blocks(Blocks(t, static_cast<int>(groups)), kScoreTokens)) *
                      kScoreTokens;
      QsaScoreMmaKernel<<<dim3(columns, Blocks(t, per)), 128, kScoreShared, stream>>>(
          query, blocks, positions, keys, n_blocks, t, ratio, per);
    }
    if (tiles == 1) {
      QsaSelectTileKernel<true><<<dim3(1, static_cast<unsigned>(t)), kSelectThreads, 0, stream>>>(
          keys, positions, out, pairs, counts, n_blocks, width, ratio, row, candidates);
      return;
    }
    QsaSelectTileKernel<false>
        <<<dim3(static_cast<unsigned>(tiles), static_cast<unsigned>(t)), kSelectThreads, 0,
           stream>>>(keys, positions, out, pairs, counts, n_blocks, width, ratio, row, candidates);
    QsaSelectMergeKernel<<<static_cast<unsigned>(t), kSelectThreads, kMergeShared, stream>>>(
        pairs, counts, positions, out, tiles, width, ratio, row, candidates);
  });
}

std::expected<void, KernelFailure> RunQsaAttn(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckQsaAttn(node); !checked) {
    return checked;
  }
  const std::uint64_t scratch = PlanQsaAttn(node);
  return launch.Run(base::Bytes(scratch), [node, scratch](ggml_backend_cuda_context& context) {
    const ggml_tensor* k = node->src[1];
    const int heads = LlmpOpInt(node, 0);
    const int kv_heads = LlmpOpInt(node, 1);
    const int t = LlmpOpInt(node, 2);
    const int row = LlmpOpInt(node, 3);
    const auto shares = static_cast<int>(QsaAttnShares(t, kv_heads, row));
    const std::int64_t items = static_cast<std::int64_t>(t) * kv_heads * shares;
    ggml_cuda_pool_alloc<float> pool;
    float* partial = nullptr;
    if (scratch > 0) {
      pool.alloc(context.pool(), scratch / sizeof(float));
      partial = pool.get();
    }
    auto* out = static_cast<float*>(node->data);
    QsaAttnKernel<<<static_cast<unsigned>(items), 32, kAttnShared, context.stream()>>>(
        static_cast<const float*>(node->src[0]->data), static_cast<const __half*>(k->data),
        static_cast<const __half*>(node->src[2]->data),
        static_cast<std::int64_t>(k->nb[1] / sizeof(__half)), static_cast<int>(k->ne[1]),
        static_cast<const std::int32_t*>(node->src[3]->data), row, out, partial, heads, kv_heads,
        shares, LlmpOpFloat(node, 4));
    if (shares > 1) {
      QsaAttnCombineKernel<<<static_cast<unsigned>(t * heads), kHead, 0, context.stream()>>>(
          partial, out, heads, kv_heads, shares);
    }
  });
}

}  // namespace llmp::kernels::ggml
