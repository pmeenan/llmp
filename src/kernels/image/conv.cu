// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The VAE decoder's 3x3 convolutions as one implicit GEMM (ops.h
// Conv3x3Implicit): out[co, p] = bias[co] + sum over (tap, ci) of
// w[co, tap, ci] x[ci, p + tap], never writing the im2col matrix. The GEMM's
// M is the output pixels, N the output channels, K the taps and input
// channels (tap-major, channels fastest, the weights' KRSC order).
//
// Each block takes a tile of 4 image rows x 32 columns (128 pixels) and 144
// output channels, eight warps of 32 pixels (one tile row) x 72 channels.
// Per stage it holds 16 input channels: the tile's input patch with its
// one-pixel halo (6 rows x 36 columns, the columns from two left of the
// tile so that pixel pairs stay 4-byte aligned), each patch pixel's 16
// channels one 32-byte row, loaded through registers (two pixels of eight
// channels per thread, transposed with byte permutes), and the weights of
// all nine taps for those channels, [co][tap][16] rows, by cp.async; two
// stages. Products are mma.sync m16n8k16 BF16 with F32 accumulation;
// shared-memory rows are XOR-swizzled in 16-byte halves so that ldmatrix
// reads eight rows without bank conflicts. The epilogue adds the BF16 bias
// in F32 and rounds once (as the im2col path's bias fill and beta = 1 do),
// through shared memory into 16-byte stores (4-byte ones at a width not a
// multiple of 8).

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <expected>
#include <format>
#include <string>

#include "kernels/image/ops.h"

namespace llmp::kernels::image {
namespace {

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
constexpr int kSmem = 2 * kStageBytes;  // 96,768
constexpr int kThreads = 256;
constexpr int kWarpN = 72;                     // channels per warp: 9 n8 tiles
constexpr int kPairs = kPw / 2;                // 18 pixel pairs per patch row
constexpr int kPatchItems = 2 * kPh * kPairs;  // (8-channel group, row, pair): 216

static_assert(kSmem <= 101376, "two stages must fit the block's shared memory");
static_assert(kPatchItems <= kThreads, "one patch item per thread");
static_assert(kBn * kBm * 2 <= kSmem, "the output tile is staged in the stages' memory");

// A 16-byte half's place in a 32-byte row: halves swap every four rows.
__device__ __forceinline__ std::uint32_t Half(int row, int half) {
  return static_cast<std::uint32_t>(row * 32 + ((half ^ ((row >> 2) & 1)) * 16));
}

__device__ __forceinline__ void CpAsync16(std::uint32_t dst, const void* src, bool valid) {
  const int size = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(size));
}
__device__ __forceinline__ void CpAsyncCommit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void CpAsyncWaitAll() { asm volatile("cp.async.wait_group 0;\n" ::); }

__device__ __forceinline__ void LdMatrixX4(std::uint32_t address, std::uint32_t& r0,
                                           std::uint32_t& r1, std::uint32_t& r2,
                                           std::uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(address));
}
__device__ __forceinline__ void LdMatrixX2(std::uint32_t address, std::uint32_t& r0,
                                           std::uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
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

__device__ __forceinline__ float B2F(std::uint16_t b) {
  return __uint_as_float(static_cast<std::uint32_t>(b) << 16U);
}
__device__ __forceinline__ std::uint16_t F2B(float f) {
  std::uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((u >> 16U) | 0x40U);
  }
  u += 0x7fffU + ((u >> 16U) & 1U);
  return static_cast<std::uint16_t>(u >> 16U);
}

struct Patch {
  std::uint32_t v[8];  // eight channels' pixel pairs
  bool active;
};

__global__ void __launch_bounds__(kThreads, 1)
    Conv3x3Kernel(const std::uint16_t* __restrict__ x, const std::uint16_t* __restrict__ w,
                  const std::uint16_t* __restrict__ bias, std::uint16_t* __restrict__ out,
                  int ci_count, int co_count, int height, int width, int tiles_x) {
  extern __shared__ __align__(128) unsigned char smem[];
  const std::uint32_t base = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem));
  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int group_id = lane >> 2;
  const int tig = lane & 3;
  const int wm = warp / 2;  // tile row
  const int wn = warp % 2;  // channel half
  const int tile = static_cast<int>(blockIdx.x);
  const int y0 = (tile / tiles_x) * kTh;
  const int x0 = (tile % tiles_x) * kTw;
  const int co0 = static_cast<int>(blockIdx.y) * kBn;
  const std::int64_t plane = static_cast<std::int64_t>(height) * width;

  // This thread's patch item: (8-channel group, patch row, pixel pair).
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
      p.v[j] = 0U;
      if (item_inside) {
        const std::int64_t at = (static_cast<std::int64_t>(c0 + (item_group * 8) + j) * plane) +
                                (static_cast<std::int64_t>(item_y) * width) + item_x;
        p.v[j] = *reinterpret_cast<const std::uint32_t*>(x + at);
      }
    }
  };
  // Two pixels' eight channels each, as 16-byte halves of their rows.
  auto store_patch = [&](int stage, const Patch& p) {
    if (!p.active) {
      return;
    }
    const std::uint32_t patch = base + (stage * kStageBytes);
    uint4 lo;
    uint4 hi;
    lo.x = __byte_perm(p.v[0], p.v[1], 0x5410);
    lo.y = __byte_perm(p.v[2], p.v[3], 0x5410);
    lo.z = __byte_perm(p.v[4], p.v[5], 0x5410);
    lo.w = __byte_perm(p.v[6], p.v[7], 0x5410);
    hi.x = __byte_perm(p.v[0], p.v[1], 0x7632);
    hi.y = __byte_perm(p.v[2], p.v[3], 0x7632);
    hi.z = __byte_perm(p.v[4], p.v[5], 0x7632);
    hi.w = __byte_perm(p.v[6], p.v[7], 0x7632);
    const int pixel = (item_row * kPw) + (item_pair * 2);
    const std::uint32_t a = patch + Half(pixel, item_group);
    const std::uint32_t b = patch + Half(pixel + 1, item_group);
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(a), "r"(lo.x), "r"(lo.y),
                 "r"(lo.z), "r"(lo.w));
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(b), "r"(hi.x), "r"(hi.y),
                 "r"(hi.z), "r"(hi.w));
  };
  // The nine taps' weights for channels c0..c0+15: rows (co, tap).
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
      // Rows (co, tap): halves swap every four channels (Half's rule on co).
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

  // ldmatrix lanes: lane l addresses row (l % 8) of matrix (l / 8).
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
      // A: this warp's two m16 tiles (tile row wm, columns 0-15 and 16-31),
      // pixel (wm, c) reading patch pixel (wm + ky, c + kx + 1).
      std::uint32_t a[2][4];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int col = (m * 16) + lm_row + ((lm_mat & 1) * 8);
        const int pixel = ((wm + ky) * kPw) + col + kx + 1;
        LdMatrixX4(patch + Half(pixel, lm_mat >> 1), a[m][0], a[m][1], a[m][2], a[m][3]);
      }
      // B: nine n8 tiles of channels wn * 72 + 8j, rows (co, tap).
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

  // The output tile, [co][pixel] BF16 in shared memory, then 16-byte stores.
  auto* tile_out = reinterpret_cast<std::uint16_t*>(smem);
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#pragma unroll
    for (int n = 0; n < 9; ++n) {
      const int co = (wn * kWarpN) + (n * 8) + (tig * 2);
      const int px = (wm * kTw) + (m * 16) + group_id;
      const float b0 = co0 + co < co_count ? B2F(bias[co0 + co]) : 0.0f;
      const float b1 = co0 + co + 1 < co_count ? B2F(bias[co0 + co + 1]) : 0.0f;
      tile_out[(co * kBm) + px] = F2B(acc[m][n][0] + b0);
      tile_out[((co + 1) * kBm) + px] = F2B(acc[m][n][1] + b1);
      tile_out[(co * kBm) + px + 8] = F2B(acc[m][n][2] + b0);
      tile_out[((co + 1) * kBm) + px + 8] = F2B(acc[m][n][3] + b1);
    }
  }
  __syncthreads();
  // 144 channels x 4 rows x 4 groups of 8 pixels.
  for (int i = tid; i < kBn * kTh * (kTw / 8); i += kThreads) {
    const int co = i / (kTh * (kTw / 8));
    const int r = (i / (kTw / 8)) % kTh;
    const int g = i % (kTw / 8);
    const int y = y0 + r;
    const int xx = x0 + (g * 8);
    if (co0 + co >= co_count || y >= height || xx >= width) {
      continue;
    }
    const std::uint16_t* src = tile_out + (co * kBm) + (r * kTw) + (g * 8);
    std::uint16_t* dst = out + (static_cast<std::int64_t>(co0 + co) * plane) +
                         (static_cast<std::int64_t>(y) * width) + xx;
    if (width % 8 == 0) {
      // Rows 16-byte aligned, and a group never crosses the image's edge.
      *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    } else {
      // An even width: pixel pairs, 4-byte aligned, up to the edge.
      for (int e = 0; e < 8 && xx + e < width; e += 2) {
        *reinterpret_cast<std::uint32_t*>(dst + e) =
            *reinterpret_cast<const std::uint32_t*>(src + e);
      }
    }
  }
}

__global__ void KrscFromF32Kernel(const float* w, std::uint16_t* out, std::int64_t co_count,
                                  std::int64_t ci_count) {
  const std::int64_t n = co_count * ci_count * 9;
  for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
    // out[co][tap][ci] = w[co][ci][tap]
    const std::int64_t ci = i % ci_count;
    const std::int64_t tap = (i / ci_count) % 9;
    const std::int64_t co = i / (ci_count * 9);
    out[i] = F2B(w[(((co * ci_count) + ci) * 9) + tap]);
  }
}

}  // namespace

Status Conv3x3Implicit(const Bf16* x, std::int64_t in_channels, std::int64_t height,
                       std::int64_t width, const Bf16* w_krsc, const Bf16* bias, Bf16* out,
                       std::int64_t out_channels, Stream stream) {
  const auto aligned = [](const void* p, std::uintptr_t b) {
    return reinterpret_cast<std::uintptr_t>(p) % b == 0;
  };
  if (in_channels <= 0 || in_channels % kCk != 0 || in_channels > 65536 || out_channels <= 0 ||
      out_channels > 65536 || height <= 0 || width <= 0 || width % 2 != 0 ||
      height * width > (std::int64_t{1} << 26) || !aligned(x, 16) || !aligned(w_krsc, 16) ||
      !aligned(out, 16) || !aligned(bias, 2)) {
    return std::unexpected(
        std::string("Conv3x3Implicit: input channels a multiple of 16, an even width, 16-byte "
                    "aligned tensors"));
  }
  const auto tiles_x = static_cast<int>((width + kTw - 1) / kTw);
  const auto tiles_y = static_cast<int>((height + kTh - 1) / kTh);
  const dim3 grid(static_cast<unsigned>(tiles_x * tiles_y),
                  static_cast<unsigned>((out_channels + kBn - 1) / kBn));
  if (cudaFuncSetAttribute(Conv3x3Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) !=
      cudaSuccess) {
    return std::unexpected(std::string("Conv3x3Implicit: shared memory limit"));
  }
  Conv3x3Kernel<<<grid, kThreads, kSmem, static_cast<cudaStream_t>(stream)>>>(
      x, w_krsc, bias, out, static_cast<int>(in_channels), static_cast<int>(out_channels),
      static_cast<int>(height), static_cast<int>(width), tiles_x);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return std::unexpected(std::format("Conv3x3Implicit: {}", cudaGetErrorString(error)));
  }
  return {};
}

Status Conv3x3WeightsKrsc(const float* w, Bf16* out, std::int64_t out_channels,
                          std::int64_t in_channels, Stream stream) {
  if (out_channels <= 0 || in_channels <= 0 ||
      out_channels * in_channels > (std::int64_t{1} << 34)) {
    return std::unexpected(std::string("Conv3x3WeightsKrsc: size"));
  }
  const std::int64_t n = out_channels * in_channels * 9;
  const auto blocks = static_cast<unsigned>(std::min<std::int64_t>((n + 255) / 256, 65535 * 64));
  KrscFromF32Kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(w, out, out_channels,
                                                                           in_channels);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return std::unexpected(std::format("Conv3x3WeightsKrsc: {}", cudaGetErrorString(error)));
  }
  return {};
}

}  // namespace llmp::kernels::image
