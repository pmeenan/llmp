// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_KERNELS_DEVICE_FLOAT_FORMATS_CUH_
#define LLMP_KERNELS_DEVICE_FLOAT_FORMATS_CUH_
#include <cuda_fp8.h>

#include <cstdint>
namespace llmp::kernels::precision {
__device__ __forceinline__ float Nan() { return __uint_as_float(0x7FC00000U); }

// E2M1's magnitudes times two, a nibble each: 0, 1, 2, 3, 4, 6, 8, 12.
__device__ __forceinline__ float E2m1(std::uint32_t code) {
  const float m = static_cast<float>((0xC8643210U >> ((code & 7U) * 4U)) & 15U) * 0.5f;
  return (code & 8U) != 0 ? -m : m;
}

__device__ __forceinline__ float E4m3(std::uint32_t bits) {
  const std::uint32_t mag = bits & 0x7FU;
  if (mag == 0x7FU) {
    return Nan();
  }
  const std::uint32_t e = mag >> 3U;
  const std::uint32_t m = mag & 7U;
  // Normal: 2^(e - 7) (1 + m / 8), F32 exponent e + 120; subnormal m · 2^-9.
  const float v = e != 0 ? __uint_as_float(((e + 120U) << 23U) | (m << 20U))
                         : static_cast<float>(m) * 0.001953125f;
  return (bits & 0x80U) != 0 ? -v : v;
}

// NVFP4's block scale, UE4M3 (format::ScaleCode::kUe4m3): E4M3's value for
// 0x00-0x7E, NaN for 0x7F and every byte with bit 7 set (no sign), as the
// CPU reference (kernels/reference/decode.h Ue4m3ToF32).
__device__ __forceinline__ float Ue4m3(std::uint32_t bits) {
  return (bits & 0xFFU) >= 0x7FU ? Nan() : E4m3(bits);
}

__device__ __forceinline__ float E8m0(std::uint32_t bits) {
  if (bits == 0xFFU) {
    return Nan();
  }
  return bits != 0 ? __uint_as_float(bits << 23U) : __uint_as_float(0x00400000U);  // 2^-127
}

// Preserve the packed hardware conversion used by the tuned GEMV.
__device__ __forceinline__ float2 E4m3x2(std::uint16_t pair) {
  const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3);
  return __half22float2(__half2(raw));
}
}  // namespace llmp::kernels::precision
#endif
