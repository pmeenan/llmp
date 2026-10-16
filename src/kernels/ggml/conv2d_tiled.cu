// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>

#include "base/bytes.h"
#include "base/check.h"
#include "common.cuh"
#include "cpy.cuh"
#include "im2col.cuh"
#include "kernels/ggml/conv2d_tiled.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/mul_mat_cublas_internal.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {
namespace {
using detail::Rejected;

// These are private stack descriptors, not new IR nodes or retained allocations.
// Parameters and source pointers are consumed by the host launchers during Run;
// captured kernels retain only the paid device addresses and scalar dimensions.
struct Tile {
  ggml_tensor input{}, columns{}, weights{}, matrix{}, product{}, transpose{}, packed{}, output{};
};
void Packed(ggml_tensor& t, std::array<std::int64_t, 4> dims, void* data) {
  t = {};
  t.type = GGML_TYPE_F32;
  t.data = data;
  std::size_t stride = sizeof(float);
  for (std::size_t i = 0; i < dims.size(); ++i) {
    t.ne[i] = dims[i];
    t.nb[i] = stride;
    stride *= static_cast<std::size_t>(dims[i]);
  }
}
void MakeTile(Tile& t, const ggml_tensor* node, const Conv2dTilePlan& plan, std::int64_t batch,
              std::int64_t y, std::int64_t rows, std::byte* scratch) {
  const auto width = node->ne[0];
  const auto count = width * rows;
  const auto channels = node->ne[2];
  const auto k = 9 * node->src[0]->ne[2];
  t.input = *node->src[1];
  t.input.ne[3] = 1;
  t.input.data =
      static_cast<std::byte*>(t.input.data) + static_cast<std::size_t>(batch) * t.input.nb[3];
  t.input.view_src = nullptr;
  t.input.view_offs = 0;
  Packed(t.columns, {k, width, rows, 1}, scratch);
  t.columns.op = GGML_OP_IM2COL;
  t.columns.src[0] = node->src[0];
  t.columns.src[1] = &t.input;
  t.columns.op_params[0] = t.columns.op_params[1] = 1;
  t.columns.op_params[2] = 1;
  // Original IH and channel pitch remain intact. Only the requested output
  // stripe is shifted; the upstream halo checks use original input bounds.
  t.columns.op_params[3] = static_cast<std::int32_t>(1 - y);
  t.columns.op_params[4] = t.columns.op_params[5] = t.columns.op_params[6] = 1;
  Packed(t.weights, {k, channels, 1, 1}, node->src[0]->data);
  Packed(t.matrix, {k, count, 1, 1}, scratch);
  Packed(t.product, {channels, count, 1, 1}, scratch + plan.columns_bytes);
  t.product.op = GGML_OP_MUL_MAT;
  t.product.src[0] = &t.weights;
  t.product.src[1] = &t.matrix;
  Packed(t.transpose, {count, channels, 1, 1}, t.product.data);
  t.transpose.nb[0] = static_cast<std::size_t>(channels) * sizeof(float);
  t.transpose.nb[1] = sizeof(float);
  Packed(t.packed, {count, channels, 1, 1}, scratch + plan.columns_bytes + plan.product_bytes);
  t.packed.op = GGML_OP_CONT;
  t.packed.src[0] = &t.transpose;
  auto* destination = static_cast<std::byte*>(node->data) +
                      static_cast<std::size_t>(batch) * node->nb[3] +
                      static_cast<std::size_t>(y) * node->nb[1];
  Packed(t.output, {count, channels, 1, 1}, destination);
  // Same-shape source/destination, dimension0 contiguous, dimension1 carries
  // rows, higher dimensions1: exactly pinned cpy's cudaMemcpy2DAsync path.
  t.output.nb[1] = node->nb[2];
  t.output.nb[2] = t.output.nb[3] = node->nb[3];
}
std::expected<CublasMulMat, KernelFailure> CheckTile(Tile& t) {
  // The selected compound implementation explicitly uses the existing F32
  // cuBLAS body. Ordinary MulMat's upstream selector and checks are unchanged.
  auto product = CheckMulMatCublas(&t.product, GGML_TYPE_F32, true);
  if (!product) return std::unexpected(product.error());
  if (product->scratch != 0 || product->weights != CublasOperand::kDirect ||
      product->input != CublasOperand::kDirect)
    return Rejected("packed F32 convolution product unexpectedly requires conversion scratch");
  if (auto copy = CheckCont(&t.packed); !copy) return std::unexpected(copy.error());
  if (!detail::AllSane({&t.output}) || !detail::Disjoint(&t.output, &t.packed, false) ||
      t.output.nb[1] < static_cast<std::size_t>(t.output.ne[0]) * sizeof(float))
    return Rejected("convolution stripe is not a disjoint pitched destination");
  return product;
}
__global__ void TransposeStripeToPitchedKernel(const float* __restrict__ src,
                                               const float* __restrict__ bias,
                                               char* __restrict__ dst, std::int64_t channels,
                                               std::int64_t count, std::size_t dst_nb1) {
  __shared__ float tile[32][33];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const std::int64_t base_c = static_cast<std::int64_t>(blockIdx.x) * 32;
  const std::int64_t base_x = static_cast<std::int64_t>(blockIdx.y) * 32;
  for (int j = 0; j < 32; j += 8) {
    const std::int64_t c = base_c + tx;
    const std::int64_t x = base_x + ty + j;
    if (c < channels && x < count) {
      tile[ty + j][tx] = src[c + channels * x];
    }
  }
  __syncthreads();
  for (int j = 0; j < 32; j += 8) {
    const std::int64_t x = base_x + tx;
    const std::int64_t c = base_c + ty + j;
    if (x < count && c < channels) {
      const float b = bias != nullptr ? bias[c] : 0.0f;
      auto* row = reinterpret_cast<float*>(dst + static_cast<std::size_t>(c) * dst_nb1);
      row[x] = tile[tx][ty + j] + b;
    }
  }
}

constexpr int kTh = 4;                               // tile rows
constexpr int kTw = 32;                              // tile columns
constexpr int kBm = kTh * kTw;                       // 128 pixels
constexpr int kBn = 144;                             // output channels
constexpr int kCk = 16;                              // input channels per stage
constexpr int kPh = kTh + 2;                         // patch rows
constexpr int kPw = kTw + 4;                         // patch columns (from x0 - 2)
constexpr int kPatchPixels = kPh * kPw;              // 216
constexpr int kPatchBytes = kPatchPixels * kCk * 2;  // 6,912
constexpr int kWeightRows = kBn * 9;                 // (co, tap)
constexpr int kWeightBytes = kWeightRows * kCk * 2;  // 41,472
constexpr int kStageBytes = kPatchBytes + kWeightBytes;
constexpr int kConv3x3Smem = 2 * kStageBytes;  // 96,768
constexpr int kThreads = 256;
constexpr int kWarpN = 72;                     // channels per warp: 9 n8 tiles
constexpr int kPairs = kPw / 2;                // 18 pixel pairs per patch row
constexpr int kPatchItems = 2 * kPh * kPairs;  // (8-channel group, row, pair): 216

static_assert(kConv3x3Smem <= 101376, "two stages must fit the block's shared memory");
static_assert(kPatchItems <= kThreads, "one patch item per thread");
static_assert(kBn * kBm * sizeof(float) <= kConv3x3Smem,
              "the F32 output tile is staged in the stages' memory");

__device__ __forceinline__ std::uint32_t Half(int row, int half) {
  return static_cast<std::uint32_t>(row * 32 + ((half ^ ((row >> 2) & 1)) * 16));
}

__device__ __forceinline__ void CpAsync16(std::uint32_t dst, const void* src, bool valid) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int size = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(size));
#else
  (void)dst;
  (void)src;
  (void)valid;
#endif
}
__device__ __forceinline__ void CpAsyncCommit() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void CpAsyncWaitAll() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ void LdMatrixX4(std::uint32_t address, std::uint32_t& r0,
                                           std::uint32_t& r1, std::uint32_t& r2,
                                           std::uint32_t& r3) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(address));
#else
  r0 = r1 = r2 = r3 = 0;
  (void)address;
#endif
}
__device__ __forceinline__ void LdMatrixX2(std::uint32_t address, std::uint32_t& r0,
                                           std::uint32_t& r1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(address));
#else
  r0 = r1 = 0;
  (void)address;
#endif
}

__device__ __forceinline__ void Mma(float (&d)[4], const std::uint32_t (&a)[4], std::uint32_t b0,
                                    std::uint32_t b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
  (void)d;
  (void)a;
  (void)b0;
  (void)b1;
#endif
}

__device__ __forceinline__ std::uint16_t F2B(float f) {
  std::uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((u >> 16U) | 0x40U);
  }
  u += 0x7fffU + ((u >> 16U) & 1U);
  return static_cast<std::uint16_t>(u >> 16U);
}

// A thread's next patch slice, held as loaded (F32 pairs): it is rounded
// to BF16 only when stored, after the current chunk's products, so the
// global loads' latency hides behind them rather than stalling before.
struct Patch {
  float2 v[8];
  bool active;
};

__global__ void __launch_bounds__(kThreads, 1)
    Conv3x3ImplicitF32Kernel(const float* __restrict__ x, const std::uint16_t* __restrict__ w,
                             const float* __restrict__ bias, float* __restrict__ out, int ci_count,
                             int co_count, int height, int width, int tiles_x) {
  extern __shared__ __align__(128) unsigned char smem[];
  const std::uint32_t base = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int group_id = lane >> 2;
  const int tig = lane & 3;
  const int wm = warp / 2;
  const int wn = warp % 2;
  const int tile = static_cast<int>(blockIdx.x);
  const int y0 = (tile / tiles_x) * kTh;
  const int x0 = (tile % tiles_x) * kTw;
  const int co0 = static_cast<int>(blockIdx.y) * kBn;
  const std::int64_t plane = static_cast<std::int64_t>(height) * width;

  const bool patch_item = tid < kPatchItems;
  const int item_group = tid / (kPh * kPairs);
  const int item_row = (tid / kPairs) % kPh;
  const int item_pair = tid % kPairs;
  const int item_y = y0 - 1 + item_row;
  const int item_x = x0 - 2 + (item_pair * 2);
  const bool item_inside =
      patch_item && item_y >= 0 && item_y < height && item_x >= 0 && item_x + 1 < width;

  auto load_patch = [&](int c0, Patch& p) {
    p.active = patch_item;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      p.v[j] = make_float2(0.0f, 0.0f);
      if (item_inside) {
        const std::int64_t at = (static_cast<std::int64_t>(c0 + (item_group * 8) + j) * plane) +
                                (static_cast<std::int64_t>(item_y) * width) + item_x;
        p.v[j] = *reinterpret_cast<const float2*>(x + at);
      }
    }
  };
  auto store_patch = [&](int stage, const Patch& p) {
    if (!p.active) {
      return;
    }
    const std::uint32_t patch = base + (stage * kStageBytes);
    std::uint32_t v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      v[j] = static_cast<std::uint32_t>(F2B(p.v[j].x)) |
             (static_cast<std::uint32_t>(F2B(p.v[j].y)) << 16U);
    }
    uint4 lo;
    uint4 hi;
    lo.x = __byte_perm(v[0], v[1], 0x5410);
    lo.y = __byte_perm(v[2], v[3], 0x5410);
    lo.z = __byte_perm(v[4], v[5], 0x5410);
    lo.w = __byte_perm(v[6], v[7], 0x5410);
    hi.x = __byte_perm(v[0], v[1], 0x7632);
    hi.y = __byte_perm(v[2], v[3], 0x7632);
    hi.z = __byte_perm(v[4], v[5], 0x7632);
    hi.w = __byte_perm(v[6], v[7], 0x7632);
    const int pixel = (item_row * kPw) + (item_pair * 2);
    const std::uint32_t a = patch + Half(pixel, item_group);
    const std::uint32_t b = patch + Half(pixel + 1, item_group);
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(a), "r"(lo.x), "r"(lo.y),
                 "r"(lo.z), "r"(lo.w));
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(b), "r"(hi.x), "r"(hi.y),
                 "r"(hi.z), "r"(hi.w));
  };
  auto load_weights = [&](int c0, int stage) {
    const std::uint32_t weights = base + (stage * kStageBytes) + kPatchBytes;
    for (int i = tid; i < kWeightRows * 2; i += kThreads) {
      const int row = i / 2;
      const int half = i % 2;
      const int co = row / 9;
      const int tap = row % 9;
      const bool valid = co0 + co < co_count;
      const std::int64_t at =
          valid ? ((static_cast<std::int64_t>(co0 + co) * 9 + tap) * ci_count) + c0 + (half * 8)
                : 0;
      const std::uint32_t dst =
          weights + static_cast<std::uint32_t>(row * 32 + ((half ^ ((co >> 2) & 1)) * 16));
      CpAsync16(dst, w + at, valid);
    }
    CpAsyncCommit();
  };

  float acc[2][9][4];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < 9; ++n) {
      acc[m][n][0] = acc[m][n][1] = acc[m][n][2] = acc[m][n][3] = 0.0f;
    }
  }

  const int chunks = ci_count / kCk;
  Patch next;
  load_patch(0, next);
  load_weights(0, 0);
  store_patch(0, next);

  const int lm_row = lane % 8;
  const int lm_mat = lane / 8;

  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int stage = chunk & 1;
    CpAsyncWaitAll();
    __syncthreads();
    if (chunk + 1 < chunks) {
      load_weights((chunk + 1) * kCk, stage ^ 1);
      load_patch((chunk + 1) * kCk, next);
    }
    const std::uint32_t patch = base + (stage * kStageBytes);
    const std::uint32_t weights = patch + kPatchBytes;
#pragma unroll
    for (int tap = 0; tap < 9; ++tap) {
      const int ky = tap / 3;
      const int kx = tap % 3;
      std::uint32_t a[2][4];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int col = (m * 16) + lm_row + ((lm_mat & 1) * 8);
        const int pixel = ((wm + ky) * kPw) + col + kx + 1;
        LdMatrixX4(patch + Half(pixel, lm_mat >> 1), a[m][0], a[m][1], a[m][2], a[m][3]);
      }
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        const int co = (wn * kWarpN) + (np * 16) + ((lm_mat >> 1) * 8) + lm_row;
        const int row = (co * 9) + tap;
        const int half = lm_mat & 1;
        std::uint32_t b0, b1, b2, b3;
        LdMatrixX4(weights + static_cast<std::uint32_t>(row * 32 + ((half ^ ((co >> 2) & 1)) * 16)),
                   b0, b1, b2, b3);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          Mma(acc[m][np * 2], a[m], b0, b1);
          Mma(acc[m][(np * 2) + 1], a[m], b2, b3);
        }
      }
      {
        const int co = (wn * kWarpN) + 64 + lm_row;
        const int row = (co * 9) + tap;
        const int half = lm_mat & 1;
        std::uint32_t b0, b1;
        LdMatrixX2(weights + static_cast<std::uint32_t>(row * 32 + ((half ^ ((co >> 2) & 1)) * 16)),
                   b0, b1);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          Mma(acc[m][8], a[m], b0, b1);
        }
      }
    }
    if (chunk + 1 < chunks) {
      store_patch(stage ^ 1, next);
    }
  }
  CpAsyncWaitAll();
  __syncthreads();

  auto* tile_out = reinterpret_cast<float*>(smem);
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < 9; ++n) {
      const int co = (wn * kWarpN) + (n * 8) + (tig * 2);
      const int px = (wm * kTw) + (m * 16) + group_id;
      const float b0 = (bias != nullptr && co0 + co < co_count) ? bias[co0 + co] : 0.0f;
      const float b1 = (bias != nullptr && co0 + co + 1 < co_count) ? bias[co0 + co + 1] : 0.0f;
      tile_out[(co * kBm) + px] = acc[m][n][0] + b0;
      tile_out[((co + 1) * kBm) + px] = acc[m][n][1] + b1;
      tile_out[(co * kBm) + px + 8] = acc[m][n][2] + b0;
      tile_out[((co + 1) * kBm) + px + 8] = acc[m][n][3] + b1;
    }
  }
  __syncthreads();
  for (int i = tid; i < kBn * kTh * (kTw / 4); i += kThreads) {
    const int co = i / (kTh * (kTw / 4));
    const int r = (i / (kTw / 4)) % kTh;
    const int g = i % (kTw / 4);
    const int y = y0 + r;
    const int xx = x0 + (g * 4);
    if (co0 + co >= co_count || y >= height || xx >= width) {
      continue;
    }
    const float* src = tile_out + (co * kBm) + (r * kTw) + (g * 4);
    float* dst = out + (static_cast<std::int64_t>(co0 + co) * plane) +
                 (static_cast<std::int64_t>(y) * width) + xx;
    if (width % 4 == 0 && (reinterpret_cast<std::uintptr_t>(dst) & 15U) == 0) {
      *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
    } else {
      for (int e = 0; e < 4 && xx + e < width; ++e) {
        dst[e] = src[e];
      }
    }
  }
}

__global__ void KrscFromF32Kernel(const float* __restrict__ w, std::uint16_t* __restrict__ out,
                                  std::int64_t co_count, std::int64_t ci_count) {
  const std::int64_t n = co_count * ci_count * 9;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    const std::int64_t ci = i % ci_count;
    const std::int64_t tap = (i / ci_count) % 9;
    const std::int64_t co = i / (ci_count * 9);
    out[i] = F2B(w[(((co * ci_count) + ci) * 9) + tap]);
  }
}

}  // namespace

std::expected<std::uint64_t, KernelFailure> PlanConv2dTiledF32(const LaunchContext&,
                                                               const ggml_tensor* node) {
  const auto plan = CheckConv2dTiledF32(node);
  if (!plan) return std::unexpected(plan.error());
  // Descriptor-only planning: these disjoint addresses are never read. Actual
  // binding repeats the same checks with the lent pool and source addresses.
  auto original = *node;
  auto weight = *node->src[0];
  auto input = *node->src[1];
  original.data = reinterpret_cast<void*>(0x100000000ULL);
  weight.data = reinterpret_cast<void*>(0x200000000ULL);
  input.data = reinterpret_cast<void*>(0x300000000ULL);
  original.src[0] = &weight;
  original.src[1] = &input;
  const auto rows = static_cast<std::int64_t>(plan->rows);
  const auto tail = (node->ne[1] - 1) % rows + 1;
  for (const auto h : {rows, tail}) {
    Tile tile;
    MakeTile(tile, &original, *plan, 0, node->ne[1] - h, h,
             reinterpret_cast<std::byte*>(0x400000000ULL));
    if (auto checked = CheckTile(tile); !checked) return std::unexpected(checked.error());
  }
  if (cudaFuncSetAttribute(Conv3x3ImplicitF32Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kConv3x3Smem) != cudaSuccess) {
    return Rejected("tiled convolution shared memory limit");
  }
  return plan->scratch;
}

std::expected<void, KernelFailure> Conv2dTiledF32(LaunchContext& launch, ggml_tensor* node) {
  return Conv2dTiledF32WithBias(launch, node, node->src[2]);
}

std::expected<void, KernelFailure> Conv2dTiledF32WithBias(LaunchContext& launch, ggml_tensor* node,
                                                          const ggml_tensor* bias) {
  auto plan = CheckConv2dTiledF32(node);
  if (!plan) return std::unexpected(plan.error());
  if (launch.cublas() == nullptr)
    return Rejected("tiled convolution requires the lent cuBLAS handle");
  const auto workspace = launch.workspace();
  if (workspace.size.value() < plan->scratch)
    return Rejected("tiled convolution exceeds the lent workspace");
  for (const auto space : {workspace, launch.cublas()->workspace()}) {
    if (auto clear = CheckClearOf(node, space.base, space.size.value()); !clear) return clear;
    if (bias != nullptr) {
      if (auto clear = CheckClearOf(bias, space.base, space.size.value()); !clear) return clear;
    }
  }
  if (bias != nullptr && (!detail::AllSane({bias}) || !detail::Disjoint(node, bias, false) ||
                          bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
                          ggml_nelements(bias) != node->ne[2])) {
    return Rejected("tiled convolution bias is invalid or aliases output");
  }
  // The provider also protects the library workspace from overlap with the
  // launch pool. Check explicitly before creating any compound operand.
  const auto library = launch.cublas()->workspace();
  if (library.size.value() != 0 && workspace.base < library.base + library.size.value() &&
      library.base < workspace.base + workspace.size.value())
    return Rejected("convolution scratch overlaps cuBLAS workspace");
  const auto width = node->ne[0];
  const auto height = node->ne[1];
  const auto co_count = node->ne[2];
  const auto ci_count = node->src[0]->ne[2];
  const auto rows = static_cast<std::int64_t>(plan->rows);
  const auto tail = (height - 1) % rows + 1;
  std::array<CublasMulMat, 2> products{};
  // All refusals precede Run. Two shapes suffice: maximum stripe and tail;
  // output offsets/batch differences stay inside the checked original span.
  for (std::size_t i = 0; i < products.size(); ++i) {
    Tile tile;
    MakeTile(tile, node, *plan, i == 0 ? 0 : node->ne[3] - 1, i == 0 ? 0 : height - tail,
             i == 0 ? rows : tail, reinterpret_cast<std::byte*>(workspace.base));
    auto checked = CheckTile(tile);
    if (!checked) return std::unexpected(checked.error());
    products[i] = *checked;
  }
  const auto& dev = ggml_cuda_info().devices[launch.device()];
  const std::uint64_t w_krsc_bytes = static_cast<std::uint64_t>(co_count) *
                                     static_cast<std::uint64_t>(ci_count) * 9ULL *
                                     sizeof(std::uint16_t);
  const bool use_implicit =
      dev.cc >= 800 && ci_count >= kCk && ci_count % kCk == 0 && width >= 2 && width % 2 == 0 &&
      (reinterpret_cast<std::uintptr_t>(node->src[1]->data) % 8 == 0) &&
      (reinterpret_cast<std::uintptr_t>(node->data) % 16 == 0) && w_krsc_bytes <= plan->scratch;
  const float* bias_ptr = bias != nullptr ? static_cast<const float*>(bias->data) : nullptr;
  return launch.Run(base::Bytes(plan->scratch), [&](ggml_backend_cuda_context& context) {
    ggml_cuda_pool_alloc<std::byte> scratch(context.pool(),
                                            static_cast<std::size_t>(plan->scratch));
    base::Check(reinterpret_cast<std::uintptr_t>(scratch.get()) == workspace.base,
                "a compound convolution starts at its checked pool base");
    if (use_implicit) {
      auto* w_krsc = reinterpret_cast<std::uint16_t*>(scratch.get());
      const std::int64_t n_w = co_count * ci_count * 9;
      const auto w_blocks =
          static_cast<unsigned>(std::min<std::int64_t>((n_w + 255) / 256, 65535 * 64));
      KrscFromF32Kernel<<<w_blocks, 256, 0, context.stream()>>>(
          static_cast<const float*>(node->src[0]->data), w_krsc, co_count, ci_count);
      CUDA_CHECK(cudaGetLastError());
      if (internal::CudaErrorPending()) return;
      const auto tiles_x = static_cast<int>((width + kTw - 1) / kTw);
      const auto tiles_y = static_cast<int>((height + kTh - 1) / kTh);
      const dim3 grid(static_cast<unsigned>(tiles_x * tiles_y),
                      static_cast<unsigned>((co_count + kBn - 1) / kBn));
      for (std::int64_t batch = 0; batch < node->ne[3]; ++batch) {
        const auto* x_batch =
            reinterpret_cast<const float*>(static_cast<const std::byte*>(node->src[1]->data) +
                                           static_cast<std::size_t>(batch) * node->src[1]->nb[3]);
        auto* out_batch = reinterpret_cast<float*>(static_cast<std::byte*>(node->data) +
                                                   static_cast<std::size_t>(batch) * node->nb[3]);
        Conv3x3ImplicitF32Kernel<<<grid, kThreads, kConv3x3Smem, context.stream()>>>(
            x_batch, w_krsc, bias_ptr, out_batch, static_cast<int>(ci_count),
            static_cast<int>(co_count), static_cast<int>(height), static_cast<int>(width), tiles_x);
        CUDA_CHECK(cudaGetLastError());
        if (internal::CudaErrorPending()) return;
      }
      return;
    }
    for (std::int64_t batch = 0; batch < node->ne[3]; ++batch) {
      for (std::int64_t y = 0; y < height; y += rows) {
        Tile tile;
        const auto h = std::min(rows, height - y);
        MakeTile(tile, node, *plan, batch, y, h, scratch.get());
        ggml_cuda_op_im2col(context, &tile.columns);
        if (internal::CudaErrorPending()) return;
        internal::LaunchPlannedMulMatCublas(context, products[h == rows ? 0 : 1], &tile.product);
        if (internal::CudaErrorPending()) return;
        const auto count = tile.output.ne[0];
        const auto channels = tile.output.ne[1];
        const dim3 block(32, 8, 1);
        const dim3 grid(static_cast<unsigned>((channels + 31) / 32),
                        static_cast<unsigned>((count + 31) / 32), 1);
        TransposeStripeToPitchedKernel<<<grid, block, 0, context.stream()>>>(
            static_cast<const float*>(tile.product.data), bias_ptr,
            static_cast<char*>(tile.output.data), channels, count, tile.output.nb[1]);
        CUDA_CHECK(cudaGetLastError());
        if (internal::CudaErrorPending()) return;
      }
    }
  });
}
}  // namespace llmp::kernels::ggml
