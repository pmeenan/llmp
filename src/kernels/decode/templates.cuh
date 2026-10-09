// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The product templates over the block decoders (D-108): each takes a
// decoder (decoders.cuh) per weight and reads every weight byte it needs
// once per tile at its compressed size; what a decoder expands stays in
// registers or shared memory. Nothing is allocated, and no workspace is
// read or written beyond the operands.
//
//   Vector     y[t, j] = s(j) Σ_k W[first + j, k] x[t, k], t < rows <= 16:
//              a warp per output row, 8 warps a block; lane l decodes
//              segments l, l + 32, ... of the row into registers and
//              accumulates each input row in F32 (fmaf, in k order within
//              the lane), then a warp sum. One instantiation per row
//              bucket (1, 2, 4, 8, 16); a row's arithmetic is the same in
//              every bucket, so the product is row-invariant.
//   VectorGlu  the same over a gate and an up weight, each its own
//              decoder, then out = glu(s_g g, s_u u) (ir::GluKind).
//   Matrix     tiles of 64 outputs by 16 or 64 input rows, 32 values of k
//              a step: the input rows converted to the decoder's Mma
//              element (BF16 or F16) and the weight tile decoded into
//              shared memory, then mma.sync m16n8k16 with F32
//              accumulation (sm_80 on); every output in the same k order
//              whatever its tile, so it is row-invariant too.
//   MatrixFma  the same tiles in F32 shared memory, F32 FMA: devices
//              without mma.sync, and exact-input checks.
//   Grouped    Matrix's tiles over an expert's rows: a block per 64
//              outputs and expert, its pairs [offsets[e], offsets[e + 1])
//              of a worklist sorted by expert gathered from x and
//              scattered to y, 16 or 64 a tile; each expert's operand from
//              a device array (any base address per expert; one whose K or
//              N is not the product's is skipped).
//   Rows       rows gathered by index and decoded to F32, times the
//              tensor scale (the reference's rounding: bit-exact).
//
// s(j) is the weight's tensor scale of row first + j (1 without one),
// applied once to the sum.

#ifndef LLMP_KERNELS_DECODE_TEMPLATES_CUH_
#define LLMP_KERNELS_DECODE_TEMPLATES_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <type_traits>

#include "kernels/decode/decoders.cuh"
#include "kernels/decode/launch.h"
#include "kernels/decode/operand.h"

namespace llmp::kernels::decode {

inline constexpr int kVectorWarps = 8;
inline constexpr int kTileN = 64;
inline constexpr int kTileK = 32;
inline constexpr int kTileThreads = 128;

__device__ __forceinline__ float RowScale(const Operand& p, std::uint32_t row) {
  return p.tensor_scale != nullptr ? __ldg(p.tensor_scale + (row / p.rows_per_scale)) : 1.0f;
}

__device__ __forceinline__ float WarpSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xFFFFFFFFU, v, offset);
  }
  return v;
}

__device__ __forceinline__ float Silu(float g) { return g / (1.0f + expf(-g)); }

// ir::GluKind: kSwiGlu 0, kGeGlu 1, kGeGluTanh 2, kSwiGluClamp 3.
__device__ __forceinline__ float Glu(std::uint8_t kind, float g, float u, float limit) {
  switch (kind) {
    case 1:
      return 0.5f * g * (1.0f + erff(g * 0.70710678118654752f)) * u;
    case 2:
      return 0.5f * g * (1.0f + tanhf(0.79788456080286536f * (g + (0.044715f * g * g * g)))) * u;
    case 3: {
      const float gc = fminf(g, limit);
      const float uc = fminf(fmaxf(u, -limit), limit);
      return Silu(gc) * uc;
    }
    default:
      return Silu(g) * u;
  }
}

// x[t, 16 seg .. 16 seg + 16) times the decoded segment, into each row's
// sum, for the rows the product has.
template <int Cols>
__device__ __forceinline__ void Accumulate(const Args& a, std::uint32_t seg, const Values& w,
                                           float (&acc)[Cols]) {
#pragma unroll
  for (int t = 0; t < Cols; ++t) {
    if (static_cast<std::uint32_t>(t) < a.rows) {
      const auto* xp = reinterpret_cast<const float4*>(
          a.x + (static_cast<std::uint64_t>(t) * a.x_stride) + (seg * kSegment));
      float s = acc[t];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float4 x = __ldg(xp + q);
        s = fmaf(w[4 * q], x.x, s);
        s = fmaf(w[(4 * q) + 1], x.y, s);
        s = fmaf(w[(4 * q) + 2], x.z, s);
        s = fmaf(w[(4 * q) + 3], x.w, s);
      }
      acc[t] = s;
    }
  }
}

template <class D, int Cols>
__global__ void __launch_bounds__(kVectorWarps * 32) VectorKernel(const Args a) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t n = (blockIdx.x * kVectorWarps) + (threadIdx.x >> 5U);
  if (n >= a.outputs) {
    return;
  }
  const std::uint32_t row = a.first + n;
  const std::uint32_t segs = a.w.k / kSegment;
  float acc[Cols];
#pragma unroll
  for (int t = 0; t < Cols; ++t) {
    acc[t] = 0.0f;
  }
  for (std::uint32_t seg = lane; seg < segs; seg += 32U) {
    Values w;
    D::Load(a.w, row, seg, w);
    Accumulate<Cols>(a, seg, w, acc);
  }
  const float scale = RowScale(a.w, row);
#pragma unroll
  for (int t = 0; t < Cols; ++t) {
    if (static_cast<std::uint32_t>(t) < a.rows) {
      const float s = WarpSum(acc[t]);
      if (lane == 0) {
        a.y[(static_cast<std::uint64_t>(t) * a.y_stride) + n] = s * scale;
      }
    }
  }
}

template <class DG, class DU, int Cols>
__global__ void __launch_bounds__(kVectorWarps * 32) VectorGluKernel(const Args a) {
  const std::uint32_t lane = threadIdx.x & 31U;
  const std::uint32_t n = (blockIdx.x * kVectorWarps) + (threadIdx.x >> 5U);
  if (n >= a.outputs) {
    return;
  }
  const std::uint32_t gate_row = a.first + n;
  const std::uint32_t up_row = a.first2 + n;
  const std::uint32_t segs = a.w.k / kSegment;
  float g[Cols];
  float u[Cols];
#pragma unroll
  for (int t = 0; t < Cols; ++t) {
    g[t] = 0.0f;
    u[t] = 0.0f;
  }
  for (std::uint32_t seg = lane; seg < segs; seg += 32U) {
    Values w;
    DG::Load(a.w, gate_row, seg, w);
    Accumulate<Cols>(a, seg, w, g);
    DU::Load(a.w2, up_row, seg, w);
    Accumulate<Cols>(a, seg, w, u);
  }
  const float gate_scale = RowScale(a.w, gate_row);
  const float up_scale = RowScale(a.w2, up_row);
#pragma unroll
  for (int t = 0; t < Cols; ++t) {
    if (static_cast<std::uint32_t>(t) < a.rows) {
      const float gs = WarpSum(g[t]);
      const float us = WarpSum(u[t]);
      if (lane == 0) {
        a.y[(static_cast<std::uint64_t>(t) * a.y_stride) + n] =
            Glu(a.glu, gs * gate_scale, us * up_scale, a.limit);
      }
    }
  }
}

template <typename T>
__device__ __forceinline__ T FromFloat(float v);
template <>
__device__ __forceinline__ float FromFloat<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ __half FromFloat<__half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 FromFloat<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

// D = A B + C for one m16n8k16 tile: A row-major 16 x 16, B column-major
// 16 x 8, F32 accumulators (PTX ISA's fragment layouts).
template <typename T>
__device__ __forceinline__ void Mma(float (&c)[4], const std::uint32_t (&a)[4], std::uint32_t b0,
                                    std::uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if constexpr (std::is_same_v<T, __half>) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
#else
  (void)c;
  (void)a;
  (void)b0;
  (void)b1;
  __trap();  // the host never launches the mma path below sm_80
#endif
}

template <typename T>
__device__ __forceinline__ std::uint32_t Pair(const T* p) {
  return *reinterpret_cast<const std::uint32_t*>(p);
}

// One block's tiles: kGrouped over expert blockIdx.y's pairs, otherwise
// over the input rows of tile blockIdx.y.
template <class D, int BM, bool kMma, bool kGrouped>
__global__ void __launch_bounds__(kTileThreads) TileKernel(const Args a) {
  static_assert(BM == 16 || BM == 64);
  using T = std::conditional_t<kMma, typename D::Mma, float>;
  constexpr int kStride = kTileK + (kMma ? 8 : 1);
  __shared__ __align__(16) T as[BM * kStride];
  __shared__ __align__(16) T bs[kTileN * kStride];

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t n0 = blockIdx.x * kTileN;
  Operand w;
  std::uint32_t begin = 0;
  std::uint32_t end = 0;
  if constexpr (kGrouped) {
    const std::uint32_t e = blockIdx.y;
    w = a.experts[e];
    if (w.k != a.w.k || w.rows != a.w.rows) {
      return;  // not an operand of the product's format: its pairs are skipped
    }
    const std::int32_t lo = __ldg(a.offsets + e);
    const std::int32_t hi = __ldg(a.offsets + e + 1);
    begin = lo < 0 ? 0U : min(static_cast<std::uint32_t>(lo), a.pairs);
    end = hi < 0 ? begin : max(begin, min(static_cast<std::uint32_t>(hi), a.pairs));
  } else {
    w = a.w;
    begin = blockIdx.y * BM;
    end = min(begin + BM, a.rows);
  }
  const std::uint32_t segs = w.k / kSegment;

  constexpr int kWarpsM = BM == 16 ? 1 : 2;
  constexpr int kWarpsN = 4 / kWarpsM;
  constexpr int kWm = BM / kWarpsM;
  constexpr int kWn = kTileN / kWarpsN;
  constexpr int kMt = kWm / 16;
  constexpr int kNt = kWn / 8;
  constexpr int kTm = BM / 16;  // the FMA path's rows a thread
  [[maybe_unused]] const std::uint32_t lane = tid & 31U;
  [[maybe_unused]] const std::uint32_t warp = tid >> 5U;
  [[maybe_unused]] const std::uint32_t wm = warp % kWarpsM;
  [[maybe_unused]] const std::uint32_t wn = warp / kWarpsM;
  [[maybe_unused]] const std::uint32_t g = lane >> 2U;
  [[maybe_unused]] const std::uint32_t q = lane & 3U;
  [[maybe_unused]] const std::uint32_t tm = tid % 16U;
  [[maybe_unused]] const std::uint32_t tn = tid / 16U;

  for (std::uint32_t m0 = begin; m0 < end; m0 += BM) {
    float c[kMma ? kMt : kTm][kMma ? kNt : 8][kMma ? 4 : 1];
#pragma unroll
    for (int i = 0; i < (kMma ? kMt : kTm); ++i) {
#pragma unroll
      for (int j = 0; j < (kMma ? kNt : 8); ++j) {
#pragma unroll
        for (int l = 0; l < (kMma ? 4 : 1); ++l) {
          c[i][j][l] = 0.0f;
        }
      }
    }
    for (std::uint32_t k0 = 0; k0 < w.k; k0 += kTileK) {
      // The input tile: BM rows by 32, a float4 at a time.
#pragma unroll
      for (int i = 0; i < BM / 16; ++i) {
        const std::uint32_t idx = tid + (static_cast<std::uint32_t>(i) * kTileThreads);
        const std::uint32_t r = idx / (kTileK / 4);
        const std::uint32_t col = (idx % (kTileK / 4)) * 4;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const std::uint32_t m = m0 + r;
        if (m < end && k0 + col < w.k) {
          std::uint32_t xr = m;
          bool ok = true;
          if constexpr (kGrouped) {
            const std::int32_t pr = __ldg(a.pair_rows + m);
            ok = pr >= 0 && static_cast<std::uint32_t>(pr) < a.x_rows;
            xr = static_cast<std::uint32_t>(pr);
          }
          if (ok) {
            v = __ldg(reinterpret_cast<const float4*>(
                a.x + (static_cast<std::uint64_t>(xr) * a.x_stride) + k0 + col));
          }
        }
        T* dst = as + (r * kStride) + col;
        dst[0] = FromFloat<T>(v.x);
        dst[1] = FromFloat<T>(v.y);
        dst[2] = FromFloat<T>(v.z);
        dst[3] = FromFloat<T>(v.w);
      }
      // The weight tile: 64 rows by two segments, a segment a thread.
      {
        const std::uint32_t r = tid / 2;
        const std::uint32_t s = tid % 2;
        const std::uint32_t n = n0 + r;
        const std::uint32_t seg = (k0 / kSegment) + s;
        Values v;
        if (n < a.outputs && seg < segs) {
          D::Load(w, a.first + n, seg, v);
        } else {
#pragma unroll
          for (int i = 0; i < static_cast<int>(kSegment); ++i) {
            v[i] = 0.0f;
          }
        }
        T* dst = bs + (r * kStride) + (s * kSegment);
#pragma unroll
        for (int i = 0; i < static_cast<int>(kSegment); ++i) {
          dst[i] = FromFloat<T>(v[i]);
        }
      }
      __syncthreads();
      if constexpr (kMma) {
#pragma unroll
        for (int kk = 0; kk < kTileK; kk += 16) {
          std::uint32_t af[kMt][4];
#pragma unroll
          for (int mt = 0; mt < kMt; ++mt) {
            const std::uint32_t r0 = (wm * kWm) + (mt * 16) + g;
            const T* p = as + (r0 * kStride) + kk + (q * 2);
            af[mt][0] = Pair(p);
            af[mt][1] = Pair(p + (8 * kStride));
            af[mt][2] = Pair(p + 8);
            af[mt][3] = Pair(p + (8 * kStride) + 8);
          }
#pragma unroll
          for (int nt = 0; nt < kNt; ++nt) {
            const std::uint32_t c0 = (wn * kWn) + (nt * 8) + g;
            const T* p = bs + (c0 * kStride) + kk + (q * 2);
            const std::uint32_t b0 = Pair(p);
            const std::uint32_t b1 = Pair(p + 8);
#pragma unroll
            for (int mt = 0; mt < kMt; ++mt) {
              Mma<T>(c[mt][nt], af[mt], b0, b1);
            }
          }
        }
      } else {
#pragma unroll 4
        for (int k = 0; k < kTileK; ++k) {
          float av[kTm];
          float bv[8];
#pragma unroll
          for (int i = 0; i < kTm; ++i) {
            av[i] = static_cast<float>(as[((tm + (16 * i)) * kStride) + k]);
          }
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            bv[j] = static_cast<float>(bs[((tn + (8 * j)) * kStride) + k]);
          }
#pragma unroll
          for (int i = 0; i < kTm; ++i) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              c[i][j][0] = fmaf(av[i], bv[j], c[i][j][0]);
            }
          }
        }
      }
      __syncthreads();
    }

    // Each output once, times its row's tensor scale.
    auto store = [&](std::uint32_t r, std::uint32_t col, float value) {
      const std::uint32_t m = m0 + r;
      const std::uint32_t n = n0 + col;
      if (m >= end || n >= a.outputs) {
        return;
      }
      std::uint32_t out = m;
      if constexpr (kGrouped) {
        if (a.pair_out != nullptr) {
          const std::int32_t po = __ldg(a.pair_out + m);
          if (po < 0) {
            return;
          }
          out = static_cast<std::uint32_t>(po);
        }
        if (out >= a.y_rows) {
          return;
        }
      }
      a.y[(static_cast<std::uint64_t>(out) * a.y_stride) + n] = value * RowScale(w, a.first + n);
    };
    if constexpr (kMma) {
#pragma unroll
      for (int mt = 0; mt < kMt; ++mt) {
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
          const std::uint32_t r = (wm * kWm) + (mt * 16) + g;
          const std::uint32_t col = (wn * kWn) + (nt * 8) + (q * 2);
          store(r, col, c[mt][nt][0]);
          store(r, col + 1, c[mt][nt][1]);
          store(r + 8, col, c[mt][nt][2]);
          store(r + 8, col + 1, c[mt][nt][3]);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < kTm; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          store(tm + (16 * i), tn + (8 * j), c[i][j][0]);
        }
      }
    }
  }
}

template <class D>
__global__ void __launch_bounds__(128) RowsKernel(const Args a) {
  const std::uint32_t i = blockIdx.x;
  const std::int32_t id = __ldg(a.ids + i);
  float* out = a.y + (static_cast<std::uint64_t>(i) * a.y_stride);
  const std::uint32_t segs = a.w.k / kSegment;
  if (id < 0 || static_cast<std::uint32_t>(id) >= a.w.rows) {
    for (std::uint32_t j = threadIdx.x; j < a.w.k; j += blockDim.x) {
      out[j] = 0.0f;
    }
    if (threadIdx.x == 0 && a.bad != nullptr) {
      *a.bad = 1;
    }
    return;
  }
  const auto row = static_cast<std::uint32_t>(id);
  const float scale = RowScale(a.w, row);
  for (std::uint32_t seg = threadIdx.x; seg < segs; seg += blockDim.x) {
    Values v;
    D::Load(a.w, row, seg, v);
#pragma unroll
    for (int j = 0; j < static_cast<int>(kSegment); ++j) {
      out[(seg * kSegment) + j] = v[j] * scale;
    }
  }
}

// ---------------------------------------------------------------- launchers

inline Status Launched(const char* what) {
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return std::unexpected(std::string(what) + ": " + cudaGetErrorString(error));
  }
  return {};
}

inline unsigned Blocks(std::uint32_t count, std::uint32_t per) { return (count + per - 1) / per; }

template <class D>
Status LaunchVector(const Args& a, Stream stream) {
  const dim3 grid(Blocks(a.outputs, kVectorWarps));
  const dim3 block(kVectorWarps * 32);
  auto* s = static_cast<cudaStream_t>(stream);
  if (a.rows <= 1) {
    VectorKernel<D, 1><<<grid, block, 0, s>>>(a);
  } else if (a.rows <= 2) {
    VectorKernel<D, 2><<<grid, block, 0, s>>>(a);
  } else if (a.rows <= 4) {
    VectorKernel<D, 4><<<grid, block, 0, s>>>(a);
  } else if (a.rows <= 8) {
    VectorKernel<D, 8><<<grid, block, 0, s>>>(a);
  } else {
    VectorKernel<D, 16><<<grid, block, 0, s>>>(a);
  }
  return Launched("llmp.decode.vector");
}

template <class DG, class DU>
Status LaunchVectorGlu(const Args& a, Stream stream) {
  const dim3 grid(Blocks(a.outputs, kVectorWarps));
  const dim3 block(kVectorWarps * 32);
  auto* s = static_cast<cudaStream_t>(stream);
  if (a.rows <= 1) {
    VectorGluKernel<DG, DU, 1><<<grid, block, 0, s>>>(a);
  } else if (a.rows <= 2) {
    VectorGluKernel<DG, DU, 2><<<grid, block, 0, s>>>(a);
  } else if (a.rows <= 4) {
    VectorGluKernel<DG, DU, 4><<<grid, block, 0, s>>>(a);
  } else if (a.rows <= 8) {
    VectorGluKernel<DG, DU, 8><<<grid, block, 0, s>>>(a);
  } else {
    VectorGluKernel<DG, DU, 16><<<grid, block, 0, s>>>(a);
  }
  return Launched("llmp.decode.vector_glu");
}

template <class D, bool kMma>
Status LaunchMatrix(const Args& a, Stream stream) {
  auto* s = static_cast<cudaStream_t>(stream);
  const dim3 block(kTileThreads);
  if (a.tile_rows == 16) {
    TileKernel<D, 16, kMma, false>
        <<<dim3(Blocks(a.outputs, kTileN), Blocks(a.rows, 16)), block, 0, s>>>(a);
  } else {
    TileKernel<D, 64, kMma, false>
        <<<dim3(Blocks(a.outputs, kTileN), Blocks(a.rows, 64)), block, 0, s>>>(a);
  }
  return Launched(kMma ? "llmp.decode.matrix" : "llmp.decode.matrix_fma");
}

template <class D>
Status LaunchGrouped(const Args& a, Stream stream) {
  auto* s = static_cast<cudaStream_t>(stream);
  const dim3 grid(Blocks(a.outputs, kTileN), a.experts_count);
  if (a.tile_rows == 16) {
    TileKernel<D, 16, true, true><<<grid, kTileThreads, 0, s>>>(a);
  } else {
    TileKernel<D, 64, true, true><<<grid, kTileThreads, 0, s>>>(a);
  }
  return Launched("llmp.decode.grouped");
}

template <class D>
Status LaunchRows(const Args& a, Stream stream) {
  RowsKernel<D><<<a.rows, 128, 0, static_cast<cudaStream_t>(stream)>>>(a);
  return Launched("llmp.decode.rows");
}

template <class D>
constexpr DecoderKernels KernelsOf() {
  DecoderKernels k;
  k.by_template[static_cast<int>(Template::kVector)] = &LaunchVector<D>;
  k.by_template[static_cast<int>(Template::kVectorGlu)] = &LaunchVectorGlu<D, D>;
  k.by_template[static_cast<int>(Template::kMatrix)] = &LaunchMatrix<D, true>;
  k.by_template[static_cast<int>(Template::kMatrixFma)] = &LaunchMatrix<D, false>;
  k.by_template[static_cast<int>(Template::kGrouped)] = &LaunchGrouped<D>;
  k.by_template[static_cast<int>(Template::kRows)] = &LaunchRows<D>;
  return k;
}

}  // namespace llmp::kernels::decode

#endif  // LLMP_KERNELS_DECODE_TEMPLATES_CUH_
