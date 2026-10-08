// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4's sparse attention at depth (llmp_ops.h, "DeepSeek V4's
// sparse attention"): llmpalooza's own kernels for the lightning indexer's
// scores and its selection, and the attention mask the selection gives.
// Each is deterministic (fixed summation orders, no atomics whose order
// reaches a result), so a model's runs repeat bit for bit (RE-031).
//
//   LidScoreKernel<R>  R rows' 64 query heads (M = 64·R) against 64 keys a
//                      step, tensor cores (mma.sync m16n8k16, F16 in, F32
//                      accumulated; the query rounded to F16 as GGML's WMMA
//                      indexer rounds it), each score's heads' ReLU times
//                      their weights summed in a fixed order; the keys a
//                      row cannot see are skipped.
//   TopKKernel         a block a row: a radix select of the k-th largest
//                      score over the row's visible keys (four 8-bit
//                      passes over the scores' order-preserving bits),
//                      then every key above it and the lowest-indexed keys
//                      equal to it, written in ascending key order.
//   SparseMaskKernel   a block a row: the window's mask copied (-inf past it
//                      to the compressed rows' start), then the compressed
//                      rows' part -inf but at the selected keys (CSA) or
//                      below the row's visible count (HCA).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/llmp_ops.h"

namespace llmp::kernels::ggml {
namespace {

constexpr int kLidDim = 128;   // the indexer's head
constexpr int kLidHeads = 64;  // its heads
constexpr int kTileKeys = 64;  // keys a step
constexpr int kKeyPad = 136;   // halves a key's row takes in shared memory
constexpr int kTopKThreads = 512;
constexpr int kMaskThreads = 256;
constexpr int kMaskSpan = 4096;  // a mask block's columns

__device__ __forceinline__ std::uint32_t SharedAddress(const void* p) {
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
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

// d += a (16x16 row) * b (16x8 col), F16 in, F32 accumulated.
__device__ __forceinline__ void Mma(float (&d)[4], const std::uint32_t (&a)[4], std::uint32_t b0,
                                    std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ std::uint32_t PackHalf2(float lo, float hi) {
  const __half2 h = __float22half2_rn(make_float2(lo, hi));
  return *reinterpret_cast<const std::uint32_t*>(&h);
}

// ---------------------------------------------------------------- scores

struct LidScoreArgs {
  const float* q = nullptr;  // [128, 64, rows]
  int q_head = 0;            // floats between heads
  int q_row = 0;             // between rows
  const half* k = nullptr;   // [128, n_kv]
  int k_row = 0;             // halves between keys
  const float* w = nullptr;  // [64, rows]
  int w_row = 0;
  const std::int32_t* visible = nullptr;  // [rows]
  float* out = nullptr;                   // [n_kv, rows] (row r at r · out_row)
  std::int64_t out_row = 0;
  int rows = 0;
  int n_kv = 0;
  int keys_per_block = 0;  // a multiple of kTileKeys
};

// Block (x, y): keys [x · keys_per_block, +keys_per_block) of rows
// [y · R, +R). Warp v computes 16 heads ((v % 4) · 16 on) of row v / 4
// against every key, so a row's four warps hold its 64 heads.
template <int R>
__global__ void __launch_bounds__(128 * R) LidScoreKernel(const LidScoreArgs a) {
  constexpr int kWarps = 4 * R;
  constexpr int kThreads = 32 * kWarps;
  __shared__ __align__(16) half keys[2][kTileKeys * kKeyPad];
  __shared__ float part[kWarps][kTileKeys];
  __shared__ float weights[R][kLidHeads];
  __shared__ int seen[R];
  __shared__ int most;

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int row0 = static_cast<int>(blockIdx.y) * R;
  const int key0 = static_cast<int>(blockIdx.x) * a.keys_per_block;

  ggml_cuda_pdl_sync();
  if (tid < R) {
    const int row = row0 + tid;
    seen[tid] = row < a.rows ? min(max(a.visible[row], 0), a.n_kv) : 0;
  }
  for (int i = tid; i < R * kLidHeads; i += kThreads) {
    const int r = i / kLidHeads;
    const int row = row0 + r;
    weights[r][i % kLidHeads] = row < a.rows ? a.w[(row * a.w_row) + (i % kLidHeads)] : 0.0f;
  }
  __syncthreads();
  if (tid == 0) {
    int m = 0;
    for (int r = 0; r < R; ++r) {
      m = max(m, seen[r]);
    }
    most = m;
  }
  __syncthreads();
  const int key_end = min(key0 + a.keys_per_block, most);
  if (key0 >= key_end) {
    return;
  }
  const int tiles = (key_end - key0 + kTileKeys - 1) / kTileKeys;

  // The warp's query rows as A fragments, rounded to F16, for all 8 steps
  // of k.
  const int r = warp / 4;
  const int row = row0 + r;
  const int head_a = ((warp % 4) * 16) + (lane / 4);
  const int head_b = head_a + 8;
  std::uint32_t qa[8][4];
  {
    const bool valid = row < a.rows;
    const float* qa_row = a.q + (static_cast<std::int64_t>(row) * a.q_row) + (head_a * a.q_head);
    const float* qb_row = a.q + (static_cast<std::int64_t>(row) * a.q_row) + (head_b * a.q_head);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int k = (16 * s) + (2 * (lane % 4));
      float2 x0 = make_float2(0.0f, 0.0f);
      float2 x1 = x0;
      float2 x2 = x0;
      float2 x3 = x0;
      if (valid) {
        x0 = *reinterpret_cast<const float2*>(qa_row + k);
        x1 = *reinterpret_cast<const float2*>(qb_row + k);
        x2 = *reinterpret_cast<const float2*>(qa_row + k + 8);
        x3 = *reinterpret_cast<const float2*>(qb_row + k + 8);
      }
      qa[s][0] = PackHalf2(x0.x, x0.y);
      qa[s][1] = PackHalf2(x1.x, x1.y);
      qa[s][2] = PackHalf2(x2.x, x2.y);
      qa[s][3] = PackHalf2(x3.x, x3.y);
    }
  }
  const float wa = weights[r][head_a];
  const float wb = weights[r][head_b];

  // A tile of keys into shared memory: 64 keys of 16 16-byte pieces.
  const auto load = [&](int tile, int buffer) {
    const int first = key0 + (tile * kTileKeys);
    for (int i = tid; i < kTileKeys * (kLidDim / 8); i += kThreads) {
      const int kk = i / (kLidDim / 8);
      const int piece = i % (kLidDim / 8);
      const int key = first + kk;
      const bool valid = key < a.n_kv;
      const half* src = a.k + (static_cast<std::int64_t>(valid ? key : 0) * a.k_row) + (piece * 8);
      CpAsync16(SharedAddress(&keys[buffer][(kk * kKeyPad) + (piece * 8)]), src, valid);
    }
    CpAsyncCommit();
  };

  load(0, 0);
  for (int t = 0; t < tiles; ++t) {
    const int buffer = t & 1;
    if (t + 1 < tiles) {
      load(t + 1, buffer ^ 1);
      CpAsyncWait<1>();
    } else {
      CpAsyncWait<0>();
    }
    __syncthreads();
    // 8 column tiles of 8 keys: each ldmatrix.x4 gives two steps of k.
#pragma unroll
    for (int n = 0; n < kTileKeys / 8; ++n) {
      float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      const half* key_row = &keys[buffer][((n * 8) + (lane % 8)) * kKeyPad];
#pragma unroll
      for (int s = 0; s < 8; s += 2) {
        std::uint32_t b0 = 0;
        std::uint32_t b1 = 0;
        std::uint32_t b2 = 0;
        std::uint32_t b3 = 0;
        LdMatrixX4(SharedAddress(key_row + (16 * s) + (8 * (lane / 8))), b0, b1, b2, b3);
        Mma(acc, qa[s], b0, b1);
        Mma(acc, qa[s + 1], b2, b3);
      }
      // Each head's ReLU times its weight; the warp's 16 heads summed in a
      // fixed tree (the lanes of one column pair differ in lane / 4).
      float v0 = (fmaxf(acc[0], 0.0f) * wa) + (fmaxf(acc[2], 0.0f) * wb);
      float v1 = (fmaxf(acc[1], 0.0f) * wa) + (fmaxf(acc[3], 0.0f) * wb);
#pragma unroll
      for (int offset = 4; offset < 32; offset <<= 1) {
        v0 += __shfl_xor_sync(0xFFFFFFFFU, v0, offset);
        v1 += __shfl_xor_sync(0xFFFFFFFFU, v1, offset);
      }
      if (lane < 4) {
        part[warp][(n * 8) + (2 * lane)] = v0;
        part[warp][(n * 8) + (2 * lane) + 1] = v1;
      }
    }
    __syncthreads();
    // A row's four warps' sums, in order, for the keys it sees.
    for (int i = tid; i < R * kTileKeys; i += kThreads) {
      const int rr = i / kTileKeys;
      const int key = key0 + (t * kTileKeys) + (i % kTileKeys);
      if (key < seen[rr]) {
        const float sum = ((part[4 * rr][i % kTileKeys] + part[(4 * rr) + 1][i % kTileKeys]) +
                           part[(4 * rr) + 2][i % kTileKeys]) +
                          part[(4 * rr) + 3][i % kTileKeys];
        a.out[(static_cast<std::int64_t>(row0 + rr) * a.out_row) + key] = sum;
      }
    }
  }
}

// ---------------------------------------------------------------- selection

// A float's bits, ordered as the floats are (NaN lowest, -0 as +0: a head
// with a negative weight and no positive dot adds -0, and equal scores tie
// to the lower row whatever their zeros' signs).
__device__ __forceinline__ std::uint32_t Ordered(float x) {
  if (isnan(x)) {
    return 0;
  }
  std::uint32_t u = __float_as_uint(x);
  if (u == 0x80000000U) {
    u = 0;
  }
  return (u & 0x80000000U) != 0 ? ~u : (u | 0x80000000U);
}

constexpr int kTopKWarps = kTopKThreads / 32;

struct TopKShared {
  int histogram[kTopKWarps][256];  // a warp's own, so its adds rarely collide
  int warp_above[kTopKWarps];
  int warp_equal[kTopKWarps];
  std::uint32_t prefix;
  int remaining;  // -1: every valid candidate is kept
};

// The chosen bin of a radix pass (warp 0): from the top bin down, the first
// at which the candidates above it and in it reach `remaining`; each lane
// sums 8 bins of the warps' histograms, the lanes' sums prefix-summed from
// the top. With `first` (the first pass), when every valid candidate fits
// in k it keeps them all (remaining -1).
__device__ __forceinline__ void ChooseBin(TopKShared& sh, std::uint32_t prefix, int shift,
                                          int remaining, int k, bool first) {
  const int lane = static_cast<int>(threadIdx.x) % 32;
  int bins[8];
  int sum = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int b = 255 - ((lane * 8) + i);  // lane 0 the top bins, each lane downward
    int total = 0;
    for (int w = 0; w < kTopKWarps; ++w) {
      total += sh.histogram[w][b];
    }
    bins[i] = total;
    sum += total;
  }
  int inclusive = sum;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int other = __shfl_up_sync(0xFFFFFFFFU, inclusive, offset);
    if (lane >= offset) {
      inclusive += other;
    }
  }
  const int all = __shfl_sync(0xFFFFFFFFU, inclusive, 31);
  if (first && all <= k) {
    if (lane == 0) {
      sh.remaining = -1;
    }
    return;
  }
  const unsigned reached = __ballot_sync(0xFFFFFFFFU, inclusive >= remaining);
  const int at = __ffs(static_cast<int>(reached)) - 1;  // the lane holding the bin
  if (lane == at) {
    int above = inclusive - sum;
    int i = 0;
    for (; i < 7; ++i) {
      if (above + bins[i] >= remaining) {
        break;
      }
      above += bins[i];
    }
    sh.prefix = prefix | (static_cast<std::uint32_t>(255 - ((lane * 8) + i)) << shift);
    sh.remaining = remaining - above;
  }
}

// The k best of `n` candidates, by value and then the lower key: candidate
// e names key `key_of(e)` (-1: none) and its value s[key]. Their keys in
// candidate order to out[0, k), -1 after; candidates are in ascending key
// order, so the output is too. A radix select of the k-th value's ordered
// bits, 8 at a time, each warp's histogram counted by leaders of the lanes
// that share a bin (exact counts, so the result is deterministic), then a
// compaction that keeps every candidate above it and the first ones equal
// to it: each thread a contiguous run of candidates, counted, scanned once
// over the block, then written.
template <typename KeyOf>
__device__ void SelectTopK(const float* s, int n, const KeyOf& key_of, int k, std::int32_t* out,
                           TopKShared& sh) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid % 32;
  const int warp = tid / 32;
  std::uint32_t prefix = 0;
  std::uint32_t mask = 0;
  int remaining = k;
  bool all = false;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = tid; i < kTopKWarps * 256; i += kTopKThreads) {
      (&sh.histogram[0][0])[i] = 0;
    }
    __syncthreads();
    for (int base = 0; base < n; base += kTopKThreads) {
      const int e = base + tid;
      const int key = e < n ? key_of(e) : -1;
      const std::uint32_t u = key >= 0 ? Ordered(s[key]) : 0;
      const bool match = key >= 0 && (u & mask) == prefix;
      const unsigned bin = match ? (u >> shift) & 255U : 256U;
      const unsigned peers = __match_any_sync(0xFFFFFFFFU, bin);
      if (bin < 256U && lane == __ffs(static_cast<int>(peers)) - 1) {
        atomicAdd(&sh.histogram[warp][bin], __popc(peers));
      }
    }
    __syncthreads();
    if (warp == 0) {
      ChooseBin(sh, prefix, shift, remaining, k, shift == 24);
    }
    __syncthreads();
    if (sh.remaining < 0) {
      all = true;
      break;
    }
    prefix = sh.prefix;
    remaining = sh.remaining;
    mask |= 255U << shift;
    __syncthreads();
  }
  // Every candidate above the threshold (every valid one with `all`), and
  // the first `remaining` equal to it, in candidate order: each warp a
  // contiguous run of candidates, read 32 at a time and counted by ballots,
  // the warps' counts summed in order, then written.
  const std::uint32_t threshold = prefix;
  const int per_warp = (((n + kTopKWarps - 1) / kTopKWarps) + 31) / 32 * 32;
  const int first = min(n, warp * per_warp);
  const int last = min(n, first + per_warp);
  const unsigned below = (1U << lane) - 1U;
  const auto classify = [&](int e, bool& above_it, bool& equal_it, int& key) {
    key = e < last ? key_of(e) : -1;
    const std::uint32_t u = key >= 0 ? Ordered(s[key]) : 0;
    above_it = key >= 0 && (all || u > threshold);
    equal_it = key >= 0 && !all && u == threshold;
  };
  int above = 0;
  int equal = 0;
  for (int base = first; base < last; base += 32) {
    bool a = false;
    bool q = false;
    int key = -1;
    classify(base + lane, a, q, key);
    above += __popc(__ballot_sync(0xFFFFFFFFU, a));
    equal += __popc(__ballot_sync(0xFFFFFFFFU, q));
  }
  if (lane == 0) {
    sh.warp_above[warp] = above;
    sh.warp_equal[warp] = equal;
  }
  __syncthreads();
  int at = 0;          // what the warps before this one keep
  int equal_rank = 0;  // the equal candidates before this warp
  int taken = 0;
  int rank = 0;
  for (int w = 0; w < kTopKWarps; ++w) {
    const int kept = all ? 0 : max(0, min(sh.warp_equal[w], remaining - rank));
    if (w < warp) {
      at += sh.warp_above[w] + kept;
    } else if (w == warp) {
      equal_rank = rank;
    }
    taken += sh.warp_above[w] + kept;
    rank += sh.warp_equal[w];
  }
  const int kept_equal = all ? 0 : max(0, min(equal, remaining - equal_rank));
  int equal_seen = 0;
  for (int base = first; base < last; base += 32) {
    bool a = false;
    bool q = false;
    int key = -1;
    classify(base + lane, a, q, key);
    const unsigned equals = __ballot_sync(0xFFFFFFFFU, q);
    const bool take = a || (q && equal_seen + __popc(equals & below) < kept_equal);
    const unsigned takes = __ballot_sync(0xFFFFFFFFU, take);
    const int to = at + __popc(takes & below);
    if (take && to < k) {
      out[to] = key;
    }
    at += __popc(takes);
    equal_seen += __popc(equals);
  }
  for (int j = min(taken, k) + tid; j < k; j += kTopKThreads) {
    out[j] = -1;
  }
}

struct TopKArgs {
  const float* scores = nullptr;  // row r at r · score_row
  std::int64_t score_row = 0;
  const std::int32_t* visible = nullptr;
  std::int32_t* out = nullptr;  // [k, rows]
  int k = 0;
  int n_kv = 0;
  // Two stages (a few rows over many keys): each of `slices` blocks keeps
  // its slice's k best into `candidates` ([k, slices, rows]), then one block
  // a row selects among them. 0: one block a row over the whole row.
  int slices = 0;
  int slice = 0;  // keys a slice
  std::int32_t* candidates = nullptr;
};

// One block a row (grid x rows), or stage 1 (grid x slices, y rows).
__global__ void __launch_bounds__(kTopKThreads) TopKKernel(const TopKArgs a) {
  __shared__ TopKShared sh;
  ggml_cuda_pdl_sync();
  const bool sliced = a.slices > 0;
  const int row = static_cast<int>(sliced ? blockIdx.y : blockIdx.x);
  const int n = min(max(a.visible[row], 0), a.n_kv);
  const float* s = a.scores + (static_cast<std::int64_t>(row) * a.score_row);
  if (!sliced) {
    SelectTopK(
        s, n, [](int e) { return e; }, a.k, a.out + (static_cast<std::int64_t>(row) * a.k), sh);
    return;
  }
  const int first = static_cast<int>(blockIdx.x) * a.slice;
  const int count = max(0, min(n - first, a.slice));
  SelectTopK(
      s, count, [first](int e) { return first + e; }, a.k,
      a.candidates + ((static_cast<std::int64_t>(row) * a.slices) + blockIdx.x) * a.k, sh);
}

// Stage 2: one block a row, among its slices' candidates (slice order, so
// key order).
__global__ void __launch_bounds__(kTopKThreads) TopKMergeKernel(const TopKArgs a) {
  __shared__ TopKShared sh;
  ggml_cuda_pdl_sync();
  const int row = static_cast<int>(blockIdx.x);
  const float* s = a.scores + (static_cast<std::int64_t>(row) * a.score_row);
  const std::int32_t* candidates = a.candidates + (static_cast<std::int64_t>(row) * a.slices * a.k);
  SelectTopK(
      s, a.slices * a.k, [candidates](int e) { return candidates[e]; }, a.k,
      a.out + (static_cast<std::int64_t>(row) * a.k), sh);
}

// ---------------------------------------------------------------- mask

struct SparseMaskArgs {
  const half* window = nullptr;  // [window_width, rows]
  std::int64_t window_row = 0;   // halves
  int window_width = 0;
  // The compressed rows' part: the selection (I32 [k, rows]), or each row's
  // visible count (I32 [rows]) when `top` is null.
  const std::int32_t* top = nullptr;
  int k = 0;
  const std::int32_t* visible = nullptr;
  half* out = nullptr;  // [cells + n_kv, rows], packed
  int cells = 0;        // the compressed rows' start
  int n_kv = 0;
};

// Block (x, y): row x's columns [y · kMaskSpan, +kMaskSpan), so a decode's
// one row still spreads over the SMs.
__global__ void __launch_bounds__(kMaskThreads) SparseMaskKernel(const SparseMaskArgs a) {
  const int tid = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.x);
  const int width = a.cells + a.n_kv;
  const int c0 = static_cast<int>(blockIdx.y) * kMaskSpan;
  const int c1 = min(width, c0 + kMaskSpan);
  half* out = a.out + (static_cast<std::int64_t>(row) * width);
  const half* window = a.window + (static_cast<std::int64_t>(row) * a.window_row);
  ggml_cuda_pdl_sync();
  const half negative = __ushort_as_half(0xFC00U);
  const half zero = __ushort_as_half(0U);
  const int seen = a.top == nullptr ? min(max(a.visible[row], 0), a.n_kv) : 0;
  for (int c = c0 + tid; c < c1; c += kMaskThreads) {
    half v = negative;
    if (c < a.window_width) {
      v = window[c];
    } else if (c >= a.cells && c - a.cells < seen) {
      v = zero;
    }
    out[c] = v;
  }
  if (a.top == nullptr) {
    return;
  }
  __syncthreads();
  const std::int32_t* top = a.top + (static_cast<std::int64_t>(row) * a.k);
  for (int j = tid; j < a.k; j += kMaskThreads) {
    const int key = top[j];
    if (key >= 0 && key < a.n_kv && a.cells + key >= c0 && a.cells + key < c1) {
      out[a.cells + key] = zero;
    }
  }
}

// A decode's few rows over many keys select in two stages (TopKArgs): the
// keys a slice, from 2,048, at most 32 slices a row; 0 slices: one stage.
constexpr std::int64_t kTwoStageRows = 8;
std::int64_t SliceKeys(std::int64_t n_kv) {
  const std::int64_t per = (((n_kv + 31) / 32) + 1023) / 1024 * 1024;
  return std::max<std::int64_t>(2048, per);
}
std::int64_t Slices(std::int64_t rows, std::int64_t n_kv) {
  if (rows > kTwoStageRows) {
    return 0;
  }
  const std::int64_t slices = (n_kv + SliceKeys(n_kv) - 1) / SliceKeys(n_kv);
  return slices > 1 ? slices : 0;
}

// The rows a lid_topk launch scores at once: all of them when their
// scores fit kDsv4LidScratch, else a multiple of 4 that does (at least 4).
std::int64_t LidGroupRows(std::int64_t rows, std::int64_t n_kv) {
  const std::int64_t per_row = std::max<std::int64_t>(n_kv, 1) * 4;
  if (rows * per_row <= kDsv4LidScratch) {
    return rows;
  }
  return std::max<std::int64_t>(4, (kDsv4LidScratch / per_row) / 4 * 4);
}

}  // namespace

std::expected<std::uint64_t, KernelFailure> PlanDsv4LidTopK(const LaunchContext& /*launch*/,
                                                            const ggml_tensor* node) {
  if (auto checked = CheckDsv4LidTopK(node); !checked) {
    return std::unexpected(checked.error());
  }
  const std::int64_t rows = node->ne[1];
  const std::int64_t n_kv = node->src[1]->ne[1];
  // The pool hands each block out from a 256-byte boundary: the scores,
  // then a two-stage selection's candidates.
  const auto round = [](std::uint64_t b) { return (b + 255) / 256 * 256; };
  const std::int64_t group = LidGroupRows(rows, n_kv);
  const std::uint64_t scores = static_cast<std::uint64_t>(group * n_kv) * sizeof(float);
  const std::uint64_t candidates =
      static_cast<std::uint64_t>(group * Slices(group, n_kv) * node->ne[0]) * sizeof(std::int32_t);
  return round(scores) + (candidates > 0 ? round(candidates) : 0);
}

std::expected<void, KernelFailure> RunDsv4LidTopK(LaunchContext& launch, ggml_tensor* node) {
  auto scratch = PlanDsv4LidTopK(launch, node);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  return launch.Run(base::Bytes(*scratch), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* q = node->src[0];
    const ggml_tensor* k = node->src[1];
    const ggml_tensor* w = node->src[2];
    const ggml_tensor* visible = node->src[3];
    const std::int64_t rows = node->ne[1];
    const std::int64_t n_kv = k->ne[1];
    const std::int64_t group = LidGroupRows(rows, n_kv);
    ggml_cuda_pool_alloc<float> scores(context.pool(), static_cast<std::size_t>(group * n_kv));
    const std::int64_t slices = Slices(group, n_kv);
    ggml_cuda_pool_alloc<std::int32_t> candidates;
    if (slices > 0) {
      candidates.alloc(context.pool(), static_cast<std::size_t>(group * slices * node->ne[0]));
    }
    for (std::int64_t g0 = 0; g0 < rows; g0 += group) {
      const std::int64_t g_rows = std::min(group, rows - g0);
      LidScoreArgs s;
      s.q = static_cast<const float*>(q->data) + (g0 * static_cast<std::int64_t>(q->nb[2] / 4));
      s.q_head = static_cast<int>(q->nb[1] / sizeof(float));
      s.q_row = static_cast<int>(q->nb[2] / sizeof(float));
      s.k = static_cast<const half*>(k->data);
      s.k_row = static_cast<int>(k->nb[1] / sizeof(half));
      s.w = static_cast<const float*>(w->data) + (g0 * static_cast<std::int64_t>(w->nb[1] / 4));
      s.w_row = static_cast<int>(w->nb[1] / sizeof(float));
      s.visible = static_cast<const std::int32_t*>(visible->data) + g0;
      s.out = scores.get();
      s.out_row = n_kv;
      s.rows = static_cast<int>(g_rows);
      s.n_kv = static_cast<int>(n_kv);
      // Decode's few rows spread over more blocks; prefill's reuse each
      // key tile over four rows.
      s.keys_per_block = g_rows <= 8 ? 256 : 1024;
      const auto grid = [&](int per) {
        return dim3(static_cast<unsigned>((n_kv + s.keys_per_block - 1) / s.keys_per_block),
                    static_cast<unsigned>((g_rows + per - 1) / per));
      };
      if (g_rows >= 4) {
        ggml_cuda_kernel_launch(
            LidScoreKernel<4>,
            ggml_cuda_kernel_launch_params(grid(4), dim3(512), 0, context.stream()), s);
      } else if (g_rows >= 2) {
        ggml_cuda_kernel_launch(
            LidScoreKernel<2>,
            ggml_cuda_kernel_launch_params(grid(2), dim3(256), 0, context.stream()), s);
      } else {
        ggml_cuda_kernel_launch(
            LidScoreKernel<1>,
            ggml_cuda_kernel_launch_params(grid(1), dim3(128), 0, context.stream()), s);
      }
      TopKArgs t;
      t.scores = scores.get();
      t.score_row = n_kv;
      t.visible = s.visible;
      t.out = static_cast<std::int32_t*>(node->data) + (g0 * node->ne[0]);
      t.k = static_cast<int>(node->ne[0]);
      t.n_kv = static_cast<int>(n_kv);
      const auto rows_grid = dim3(static_cast<unsigned>(g_rows));
      if (slices == 0 || Slices(g_rows, n_kv) == 0) {
        ggml_cuda_kernel_launch(
            TopKKernel,
            ggml_cuda_kernel_launch_params(rows_grid, dim3(kTopKThreads), 0, context.stream()), t);
        continue;
      }
      t.slices = static_cast<int>(slices);
      t.slice = static_cast<int>(SliceKeys(n_kv));
      t.candidates = candidates.get();
      ggml_cuda_kernel_launch(
          TopKKernel,
          ggml_cuda_kernel_launch_params(
              dim3(static_cast<unsigned>(slices), static_cast<unsigned>(g_rows)),
              dim3(kTopKThreads), 0, context.stream()),
          t);
      ggml_cuda_kernel_launch(
          TopKMergeKernel,
          ggml_cuda_kernel_launch_params(rows_grid, dim3(kTopKThreads), 0, context.stream()), t);
    }
  });
}

std::expected<void, KernelFailure> RunDsv4SparseMask(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckDsv4SparseMask(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* window = node->src[0];
    const ggml_tensor* rows_of = node->src[1];
    const bool counts = LlmpOpInt(node, 1) == 1;
    SparseMaskArgs m;
    m.window = static_cast<const half*>(window->data);
    m.window_row = static_cast<std::int64_t>(window->nb[1] / sizeof(half));
    m.window_width = static_cast<int>(window->ne[0]);
    m.top = counts ? nullptr : static_cast<const std::int32_t*>(rows_of->data);
    m.k = counts ? 0 : static_cast<int>(rows_of->ne[0]);
    m.visible = counts ? static_cast<const std::int32_t*>(rows_of->data) : nullptr;
    m.out = static_cast<half*>(node->data);
    m.cells = LlmpOpInt(node, 0);
    m.n_kv = static_cast<int>(node->ne[0]) - m.cells;
    const auto spans = static_cast<unsigned>((node->ne[0] + kMaskSpan - 1) / kMaskSpan);
    ggml_cuda_kernel_launch(
        SparseMaskKernel,
        ggml_cuda_kernel_launch_params(dim3(static_cast<unsigned>(node->ne[1]), spans),
                                       dim3(kMaskThreads), 0, context.stream()),
        m);
  });
}

}  // namespace llmp::kernels::ggml
