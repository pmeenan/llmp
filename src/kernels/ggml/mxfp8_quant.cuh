// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// MXFP8 quantization in device code (llmp_ops.h llmp.mxfp8.quantize and
// the fusions that end in it): a 32-value block's E8M0 scale is
// 2^ceil(log2(amax / 448)), so that no value exceeds E4M3's 448, and each
// value divided by it is rounded to the nearest E4M3 (saturating). The
// codes and scales are laid out for CUTLASS's product (mxfp8_cutlass.h,
// moe_cutlass.h SfOffset). CUDA only.

#ifndef LLMP_KERNELS_GGML_MXFP8_QUANT_CUH_
#define LLMP_KERNELS_GGML_MXFP8_QUANT_CUH_

#include <cuda_fp8.h>

#include <cstdint>

namespace llmp::kernels::ggml::mxfp8 {

// The E8M0 code of a block whose largest magnitude is `amax` (finite, not
// negative): the exponent of amax / 448 rounded up; 0 (2^-127) for zeros.
__device__ __forceinline__ std::uint32_t ScaleCode(float amax) {
  const std::uint32_t bits = __float_as_uint(amax * (1.0f / 448.0f));
  std::uint32_t e = (bits >> 23U) & 0xffU;
  if ((bits & 0x7fffffU) != 0U) {
    ++e;
  }
  return e > 253U ? 253U : e;
}

// 1 / 2^(e - 127), exactly (a power of two): what each value is multiplied
// by before its rounding.
__device__ __forceinline__ float InverseScale(std::uint32_t e) {
  return __uint_as_float((254U - e) << 23U);
}

// Four values, times `inverse`, as four E4M3 codes (the first in the low
// byte).
__device__ __forceinline__ std::uint32_t Pack4(float a, float b, float c, float d, float inverse) {
  const __nv_fp8x2_storage_t lo =
      __nv_cvt_float2_to_fp8x2(make_float2(a * inverse, b * inverse), __NV_SATFINITE, __NV_E4M3);
  const __nv_fp8x2_storage_t hi =
      __nv_cvt_float2_to_fp8x2(make_float2(c * inverse, d * inverse), __NV_SATFINITE, __NV_E4M3);
  return static_cast<std::uint32_t>(lo) | (static_cast<std::uint32_t>(hi) << 16U);
}

}  // namespace llmp::kernels::ggml::mxfp8

#endif  // LLMP_KERNELS_GGML_MXFP8_QUANT_CUH_
