// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Test-only retained original producer. Do not call the newly factored helper.
// Source: main156c7be4 src/kernels/ggml/dsv4_sparse.cu; original source SHA256:
// 8a5db69ddf1d6a9dd990614806ca6e25c7f4eb916751f686e980687ead5901a2
// Score helpers/kernel body and original argument/launch formation are copied
// verbatim (unused selection/mask constants omitted). Only surrounding output
// observation is added: fill the old score scratch, then copy every group out.
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
#include "tests/support/dsv4_score_original.h"

namespace llmp::test_support {
namespace {
using llmp::kernels::ggml::kDsv4LidScratch;
constexpr int kLidDim = 128;   // the indexer's head
constexpr int kLidHeads = 64;  // its heads
constexpr int kTileKeys = 64;  // keys a step
constexpr int kKeyPad = 136;   // halves a key's row takes in shared memory

// clang-format off
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

std::int64_t LidGroupRows(std::int64_t rows, std::int64_t n_kv) {
  const std::int64_t per_row = std::max<std::int64_t>(n_kv, 1) * 4;
  if (rows * per_row <= kDsv4LidScratch) {
    return rows;
  }
  return std::max<std::int64_t>(4, (kDsv4LidScratch / per_row) / 4 * 4);
}

// clang-format on

__global__ void InitializeOriginalScores(std::uint32_t* p, std::int64_t n) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0xff800000U;
}
}  // namespace

std::expected<void, kernels::ggml::KernelFailure> OriginalLidScores(
    kernels::ggml::LaunchContext& launch, ggml_tensor* node, ggml_tensor* output) {
  auto checked = kernels::ggml::CheckDsv4LidTopK(node);
  if (!checked) return checked;
  const std::int64_t rows = node->ne[1];
  const std::int64_t cells = node->src[1]->ne[1];
  if (output == nullptr || output->type != GGML_TYPE_F32 || output->ne[0] != cells ||
      output->ne[1] != rows || output->ne[2] != 1 || output->ne[3] != 1 ||
      !ggml_is_contiguous(output) || output->data == nullptr)
    return std::unexpected(kernels::ggml::KernelFailure{
        .error = kernels::ggml::KernelError::kRejected,
        .detail = "original score observation needs a bound packed F32 output"});
  const auto begin = reinterpret_cast<std::uintptr_t>(output->data);
  const auto size = ggml_nbytes(output);
  if (begin % 4 != 0 || size > UINTPTR_MAX - begin)
    return std::unexpected(kernels::ggml::KernelFailure{
        .error = kernels::ggml::KernelError::kRejected,
        .detail = "original score observation has an invalid output interval"});
  for (const auto* input : {node, node->src[0], node->src[1], node->src[2], node->src[3]}) {
    const auto other = reinterpret_cast<std::uintptr_t>(input->data);
    if ((begin <= other && other - begin < size) ||
        (other <= begin && begin - other < ggml_nbytes(input)))
      return std::unexpected(kernels::ggml::KernelFailure{
          .error = kernels::ggml::KernelError::kRejected,
          .detail = "original score observation overlaps an original operand/output"});
  }
  const auto bytes = static_cast<std::uint64_t>(LidGroupRows(rows, cells) * cells * 4);
  return launch.Run(
      base::Bytes((bytes + 255) / 256 * 256), [node, output](ggml_backend_cuda_context& context) {
        const ggml_tensor* q = node->src[0];
        const ggml_tensor* k = node->src[1];
        const ggml_tensor* w = node->src[2];
        const ggml_tensor* visible = node->src[3];
        const std::int64_t rows = node->ne[1];
        const std::int64_t n_kv = k->ne[1];
        const std::int64_t group = LidGroupRows(rows, n_kv);
        ggml_cuda_pool_alloc<float> scores(context.pool(), static_cast<std::size_t>(group * n_kv));
        for (std::int64_t g0 = 0; g0 < rows; g0 += group) {
          const std::int64_t g_rows = std::min(group, rows - g0);
          InitializeOriginalScores<<<static_cast<unsigned>((g_rows * n_kv + 255) / 256), 256, 0,
                                     context.stream()>>>(
              reinterpret_cast<std::uint32_t*>(scores.get()), g_rows * n_kv);
          // clang-format off
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
          // clang-format on
          CUDA_CHECK(cudaMemcpyAsync(static_cast<float*>(output->data) + g0 * n_kv, scores.get(),
                                     static_cast<std::size_t>(g_rows * n_kv) * sizeof(float),
                                     cudaMemcpyDeviceToDevice, context.stream()));
        }
      });
}
}  // namespace llmp::test_support
