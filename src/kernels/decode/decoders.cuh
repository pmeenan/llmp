// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The block decoders (D-108; docs/engine-components.md#weight-and-state-formats):
// one small struct per encoding and layout, which decodes a 16-value
// segment of one row of a matrix into registers. Nothing is written to
// global memory; the product templates (templates.cuh) stage what they
// decode in registers or shared memory only.
//
// Each decoder has:
//   kId                    its DecoderId (operand.h);
//   Mma                    the element the matrix templates decode it to
//                          for mma.sync: BF16 where every decoded value is
//                          exactly a BF16 (BF16 itself, NVFP4's code times
//                          its UE4M3 scale, MXFP8's E4M3 code times a power
//                          of two), which also keeps the activations'
//                          range; F16 where the weights need F16's
//                          precision (F16 itself, Q8_0's F16 d times an
//                          I8, as GGML's dequantization to F16 rounds it);
//   Load(p, row, seg, v)   values [16 seg, 16 seg + 16) of row `row` of
//                          the matrix `p`, each exactly as the CPU
//                          reference (kernels/reference/decode.h) decodes
//                          it before a tensor scale: code times block scale,
//                          one F32 rounding, which for these encodings is
//                          exact.
// The tensor scale is the templates' (RowScale), applied once per output.
//
// Element formats as the reference's: E2M1 ±{0, 0.5, 1, 1.5, 2, 3, 4, 6};
// OCP E4M3 (bias 7, subnormals, 0x7F and 0xFF NaN); NVFP4's UE4M3 scale
// (E4M3's value for 0x00-0x7E, NaN from 0x7F up: format::ScaleCode); E8M0
// 2^(x - 127), 0xFF NaN. Every Load reads only its segment's bytes: the host launch code
// (products.cc) checks each part's extent and alignment before a launch.

#ifndef LLMP_KERNELS_DECODE_DECODERS_CUH_
#define LLMP_KERNELS_DECODE_DECODERS_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "kernels/decode/float_formats.cuh"
#include "kernels/decode/operand.h"

namespace llmp::kernels::decode {

using Values = float[kSegment];

using precision::E2m1;
using precision::E4m3;
using precision::E8m0;
using precision::Nan;
using precision::Ue4m3;

__device__ __forceinline__ std::uint32_t Byte(const std::uint8_t* p) { return __ldg(p); }

// Eight E2M1 codes, two a byte (value 2i in byte i's low nibble), times s.
__device__ __forceinline__ void Fp4Pairs(std::uint32_t lo, std::uint32_t hi, float s, Values& v) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const std::uint32_t b = (lo >> (8 * i)) & 0xFFU;
    v[2 * i] = E2m1(b & 15U) * s;
    v[(2 * i) + 1] = E2m1(b >> 4U) * s;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const std::uint32_t b = (hi >> (8 * i)) & 0xFFU;
    v[8 + (2 * i)] = E2m1(b & 15U) * s;
    v[8 + (2 * i) + 1] = E2m1(b >> 4U) * s;
  }
}

struct Bf16Dense {
  static constexpr DecoderId kId = DecoderId::kBf16Dense;
  using Mma = __nv_bfloat16;
  __device__ static void Load(const Operand& p, std::uint32_t row, std::uint32_t seg, Values& v) {
    const auto* src = reinterpret_cast<const uint4*>(p.data + (row * p.row_bytes) + (seg * 32U));
    const uint4 a = __ldg(src);
    const uint4 b = __ldg(src + 1);
    const std::uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v[2 * i] = __uint_as_float(w[i] << 16U);
      v[(2 * i) + 1] = __uint_as_float(w[i] & 0xFFFF0000U);
    }
  }
};

struct F16Dense {
  static constexpr DecoderId kId = DecoderId::kF16Dense;
  using Mma = __half;
  __device__ static void Load(const Operand& p, std::uint32_t row, std::uint32_t seg, Values& v) {
    const auto* src = reinterpret_cast<const uint4*>(p.data + (row * p.row_bytes) + (seg * 32U));
    const uint4 a = __ldg(src);
    const uint4 b = __ldg(src + 1);
    const std::uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      v[2 * i] = __half2float(__ushort_as_half(static_cast<unsigned short>(w[i] & 0xFFFFU)));
      v[(2 * i) + 1] = __half2float(__ushort_as_half(static_cast<unsigned short>(w[i] >> 16U)));
    }
  }
};

// block_q8_0: d (F16), then 32 I8; a segment is half a block. Blocks are
// 34 bytes, so only two-byte loads are aligned.
struct Q8_0Blocks {
  static constexpr DecoderId kId = DecoderId::kQ8_0Blocks;
  using Mma = __half;
  __device__ static void Load(const Operand& p, std::uint32_t row, std::uint32_t seg, Values& v) {
    const std::uint8_t* block =
        p.data + (row * p.row_bytes) + (static_cast<std::uint64_t>(seg / 2U) * kQ8_0BlockBytes);
    const auto* half = reinterpret_cast<const unsigned short*>(block);
    const float d = __half2float(__ushort_as_half(__ldg(half)));
    const auto* q = reinterpret_cast<const unsigned short*>(block + 2 + ((seg & 1U) * 16U));
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const std::uint32_t pair = __ldg(q + i);
      v[2 * i] = d * static_cast<float>(static_cast<std::int8_t>(pair & 0xFFU));
      v[(2 * i) + 1] = d * static_cast<float>(static_cast<std::int8_t>(pair >> 8U));
    }
  }
};

// NVFP4 codes as rows of K / 2 bytes (kSplit's and the atoms' codes part):
// segment `seg` is 8 bytes, its scale block `seg`.
__device__ __forceinline__ void Nvfp4Codes(const Operand& p, std::uint32_t row, std::uint32_t seg,
                                           float s, Values& v) {
  const uint2 c = __ldg(reinterpret_cast<const uint2*>(p.data + (row * p.row_bytes) + (seg * 8U)));
  Fp4Pairs(c.x, c.y, s, v);
}

struct Nvfp4Split {
  static constexpr DecoderId kId = DecoderId::kNvfp4Split;
  using Mma = __nv_bfloat16;
  __device__ static void Load(const Operand& p, std::uint32_t row, std::uint32_t seg, Values& v) {
    const float s =
        Ue4m3(Byte(p.scales + (row * static_cast<std::uint64_t>(p.k / kNvfp4Block)) + seg));
    Nvfp4Codes(p, row, seg, s, v);
  }
};

struct Nvfp4Atoms {
  static constexpr DecoderId kId = DecoderId::kNvfp4Atoms;
  using Mma = __nv_bfloat16;
  __device__ static void Load(const Operand& p, std::uint32_t row, std::uint32_t seg, Values& v) {
    const float s = Ue4m3(Byte(p.scales + AtomOffset(row, seg, p.k / kNvfp4Block)));
    Nvfp4Codes(p, row, seg, s, v);
  }
};

// Each row: its K / 2 code bytes, then its K / 16 scales. A row's bytes
// (9K / 16) need not be aligned beyond one byte.
struct Nvfp4Interleaved {
  static constexpr DecoderId kId = DecoderId::kNvfp4Interleaved;
  using Mma = __nv_bfloat16;
  __device__ static void Load(const Operand& p, std::uint32_t row, std::uint32_t seg, Values& v) {
    const std::uint8_t* base = p.data + (row * p.row_bytes);
    const float s = Ue4m3(Byte(base + (p.k / 2U) + seg));
    const std::uint8_t* c = base + (seg * 8U);
    const std::uint32_t lo =
        Byte(c) | (Byte(c + 1) << 8U) | (Byte(c + 2) << 16U) | (Byte(c + 3) << 24U);
    const std::uint32_t hi =
        Byte(c + 4) | (Byte(c + 5) << 8U) | (Byte(c + 6) << 16U) | (Byte(c + 7) << 24U);
    Fp4Pairs(lo, hi, s, v);
  }
};

// block_nvfp4: 64 values, four UE4M3 scales, then 32 bytes, byte j of a
// 16-value sub-block holding values j (low nibble) and j + 8 (high). A
// segment is a sub-block; blocks are 36 bytes, so four-byte loads align.
struct Nvfp4GgmlBlocks {
  static constexpr DecoderId kId = DecoderId::kNvfp4GgmlBlocks;
  using Mma = __nv_bfloat16;
  __device__ static void Load(const Operand& p, std::uint32_t row, std::uint32_t seg, Values& v) {
    const std::uint8_t* block = p.data + (row * p.row_bytes) +
                                (static_cast<std::uint64_t>(seg / 4U) * kGgmlNvfp4BlockBytes);
    const std::uint32_t sub = seg & 3U;
    const float s = Ue4m3(Byte(block + sub));
    const auto* c = reinterpret_cast<const std::uint32_t*>(block + 4 + (sub * 8U));
    const std::uint32_t w[2] = {__ldg(c), __ldg(c + 1)};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const std::uint32_t b = (w[j / 4] >> (8 * (j % 4))) & 0xFFU;
      v[j] = E2m1(b & 15U) * s;
      v[j + 8] = E2m1(b >> 4U) * s;
    }
  }
};

// MXFP8 codes as rows of K bytes: segment `seg` is 16 bytes, half of scale
// block seg / 2.
__device__ __forceinline__ void Mxfp8Codes(const Operand& p, std::uint32_t row, std::uint32_t seg,
                                           float s, Values& v) {
  const uint4 c = __ldg(reinterpret_cast<const uint4*>(p.data + (row * p.row_bytes) + (seg * 16U)));
  const std::uint32_t w[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    v[i] = E4m3((w[i / 4] >> (8 * (i % 4))) & 0xFFU) * s;
  }
}

struct Mxfp8Split {
  static constexpr DecoderId kId = DecoderId::kMxfp8Split;
  using Mma = __nv_bfloat16;
  __device__ static void Load(const Operand& p, std::uint32_t row, std::uint32_t seg, Values& v) {
    const float s =
        E8m0(Byte(p.scales + (row * static_cast<std::uint64_t>(p.k / kMxfp8Block)) + (seg / 2U)));
    Mxfp8Codes(p, row, seg, s, v);
  }
};

struct Mxfp8Atoms {
  static constexpr DecoderId kId = DecoderId::kMxfp8Atoms;
  using Mma = __nv_bfloat16;
  __device__ static void Load(const Operand& p, std::uint32_t row, std::uint32_t seg, Values& v) {
    const float s = E8m0(Byte(p.scales + AtomOffset(row, seg / 2U, p.k / kMxfp8Block)));
    Mxfp8Codes(p, row, seg, s, v);
  }
};

// MXFP4 uses the same adjacent E2M1 nibbles as NVFP4 and one E8M0
// scale per 32 values; each 16-value segment uses half of that block.
struct Mxfp4Split {
  static constexpr DecoderId kId = DecoderId::kMxfp4Split;
  using Mma = __nv_bfloat16;
  __device__ static void Load(const Operand& p, std::uint32_t row, std::uint32_t seg, Values& v) {
    const float s = E8m0(Byte(p.scales + row * static_cast<std::uint64_t>(p.k / 32U) + seg / 2U));
    Nvfp4Codes(p, row, seg, s, v);
  }
};
struct Mxfp4Atoms {
  static constexpr DecoderId kId = DecoderId::kMxfp4Atoms;
  using Mma = __nv_bfloat16;
  __device__ static void Load(const Operand& p, std::uint32_t row, std::uint32_t seg, Values& v) {
    const float s = E8m0(Byte(p.scales + AtomOffset(row, seg / 2U, p.k / 32U)));
    Nvfp4Codes(p, row, seg, s, v);
  }
};

// Affine integer quantization: unsigned codes q (4 or 8 bits) per row,
// scale s and bias b (or zero point z) per group of `group` values along a row.
// Each 16-value segment lies entirely within one group g = (seg * 16) / group.
// Arithmetic matches kernels/reference/decode.h:
// s * q + b (std::fma) or s * (q - z).
struct AffineSplit {
  static constexpr DecoderId kId = DecoderId::kAffineSplit;
  using Mma = __nv_bfloat16;

  __device__ static void Load(const Operand& p, std::uint32_t row, std::uint32_t seg, Values& v) {
    const std::uint32_t g = (seg * kSegment) / p.group;
    const std::uint64_t scale_row_offset = row * static_cast<std::uint64_t>(p.scale_row_bytes);

    float s = 0.0f;
    const std::uint8_t scale_elem = p.scale_type & 0x7FU;
    if (scale_elem == 3U) {
      s = __ldg(reinterpret_cast<const float*>(p.scales + scale_row_offset + (g * 4U)));
    } else if (scale_elem == 2U) {
      const auto u =
          __ldg(reinterpret_cast<const unsigned short*>(p.scales + scale_row_offset + (g * 2U)));
      s = __uint_as_float(static_cast<std::uint32_t>(u) << 16U);
    } else {
      const auto u =
          __ldg(reinterpret_cast<const unsigned short*>(p.scales + scale_row_offset + (g * 2U)));
      s = __half2float(__ushort_as_half(u));
    }

    const bool zero_point = (p.scale_type & 0x80U) != 0;
    std::int32_t z = 0;
    float b = 0.0f;
    if (zero_point) {
      const std::uint64_t groups = p.k / p.group;
      const std::uint64_t zero_row = ((groups * p.bits) + 7) / 8;
      const std::uint8_t* z_base = p.biases + (row * zero_row);
      if (p.bits == 4) {
        const std::uint32_t zb = Byte(z_base + (g / 2U));
        z = static_cast<std::int32_t>((g & 1U) != 0 ? (zb >> 4U) & 15U : (zb & 15U));
      } else {
        z = static_cast<std::int32_t>(Byte(z_base + g));
      }
    } else if (p.biases != nullptr) {
      if (scale_elem == 3U) {
        b = __ldg(reinterpret_cast<const float*>(p.biases + scale_row_offset + (g * 4U)));
      } else if (scale_elem == 2U) {
        const auto u =
            __ldg(reinterpret_cast<const unsigned short*>(p.biases + scale_row_offset + (g * 2U)));
        b = __uint_as_float(static_cast<std::uint32_t>(u) << 16U);
      } else {
        const auto u =
            __ldg(reinterpret_cast<const unsigned short*>(p.biases + scale_row_offset + (g * 2U)));
        b = __half2float(__ushort_as_half(u));
      }
    }

    if (p.bits == 4) {
      const auto* src = p.data + (row * p.row_bytes) + (seg * 8U);
      const uint2 c = __ldg(reinterpret_cast<const uint2*>(src));
      if (zero_point) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const std::uint32_t pair = (c.x >> (8 * i)) & 0xFFU;
          v[2 * i] = s * static_cast<float>(static_cast<std::int32_t>(pair & 15U) - z);
          v[(2 * i) + 1] = s * static_cast<float>(static_cast<std::int32_t>(pair >> 4U) - z);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const std::uint32_t pair = (c.y >> (8 * i)) & 0xFFU;
          v[8 + (2 * i)] = s * static_cast<float>(static_cast<std::int32_t>(pair & 15U) - z);
          v[8 + (2 * i) + 1] = s * static_cast<float>(static_cast<std::int32_t>(pair >> 4U) - z);
        }
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const std::uint32_t pair = (c.x >> (8 * i)) & 0xFFU;
          v[2 * i] = fmaf(s, static_cast<float>(pair & 15U), b);
          v[(2 * i) + 1] = fmaf(s, static_cast<float>(pair >> 4U), b);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const std::uint32_t pair = (c.y >> (8 * i)) & 0xFFU;
          v[8 + (2 * i)] = fmaf(s, static_cast<float>(pair & 15U), b);
          v[8 + (2 * i) + 1] = fmaf(s, static_cast<float>(pair >> 4U), b);
        }
      }
    } else {
      const auto* src = p.data + (row * p.row_bytes) + (seg * 16U);
      const uint4 c = __ldg(reinterpret_cast<const uint4*>(src));
      const std::uint32_t w[4] = {c.x, c.y, c.z, c.w};
      if (zero_point) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const std::uint32_t code = (w[i / 4] >> (8 * (i % 4))) & 0xFFU;
          v[i] = s * static_cast<float>(static_cast<std::int32_t>(code) - z);
        }
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const std::uint32_t code = (w[i / 4] >> (8 * (i % 4))) & 0xFFU;
          v[i] = fmaf(s, static_cast<float>(code), b);
        }
      }
    }
  }
};

}  // namespace llmp::kernels::decode

#endif  // LLMP_KERNELS_DECODE_DECODERS_CUH_
