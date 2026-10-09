// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// One block of 32 F32 values quantized exactly as GGML's reference
// quantizers do it (quantize_row_q8_0_ref and quantize_row_q4_0_ref,
// ggml-quants.c at the pin; kernels/reference EncodeRows): the same
// comparisons in the same order, and each arithmetic step an IEEE
// operation rounded to nearest, written as an intrinsic so that no unit
// flag (-use_fast_math's approximate division, -fmad's contraction) changes
// it. Only a unit built with -ftz flushes subnormal operands and results
// (kv_quant.h). The callers own the block's placement.

#ifndef LLMP_KERNELS_GGML_KV_QUANT_CUH_
#define LLMP_KERNELS_GGML_KV_QUANT_CUH_

#include <cstdint>

#include "common.cuh"

namespace llmp::kernels::ggml::kv {

inline constexpr int kBlockValues = 32;
static_assert(QK8_0 == kBlockValues && QK4_0 == kBlockValues);

// quantize_row_q8_0_ref: d = amax / 127, codes round(x / d), half-away.
__device__ __forceinline__ block_q8_0 QuantizeQ8_0(const float (&x)[kBlockValues]) {
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < kBlockValues; ++j) {
    const float v = fabsf(x[j]);
    amax = amax > v ? amax : v;  // MAX(amax, fabsf(v))
  }
  const float d = __fdiv_rn(amax, 127.0f);
  const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
  block_q8_0 y;
  y.d = __float2half_rn(d);
#pragma unroll
  for (int j = 0; j < kBlockValues; ++j) {
    y.qs[j] = static_cast<int8_t>(roundf(__fmul_rn(x[j], id)));
  }
  return y;
}

// quantize_row_q4_0_ref: the value of largest magnitude (the first among
// equals) over -8 as the scale, codes min(15, trunc(x / d + 8.5)), value j
// in byte j's low nibble and value j + 16 in its high one.
__device__ __forceinline__ block_q4_0 QuantizeQ4_0(const float (&x)[kBlockValues]) {
  float amax = 0.0f;
  float vmax = 0.0f;
#pragma unroll
  for (int j = 0; j < kBlockValues; ++j) {
    const float v = x[j];
    if (amax < fabsf(v)) {
      amax = fabsf(v);
      vmax = v;
    }
  }
  const float d = __fdiv_rn(vmax, -8.0f);
  const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
  block_q4_0 y;
  y.d = __float2half_rn(d);
#pragma unroll
  for (int j = 0; j < kBlockValues / 2; ++j) {
    const float x0 = __fmul_rn(x[j], id);
    const float x1 = __fmul_rn(x[(kBlockValues / 2) + j], id);
    const int xi0 = min(15, static_cast<int>(static_cast<int8_t>(__fadd_rn(x0, 8.5f))));
    const int xi1 = min(15, static_cast<int>(static_cast<int8_t>(__fadd_rn(x1, 8.5f))));
    y.qs[j] = static_cast<uint8_t>(xi0 | (xi1 << 4));
  }
  return y;
}

// The encoding's block bytes at `destination` (2-byte aligned, the blocks'
// alignment).
__device__ __forceinline__ void StoreBlock(ggml_type type, const float (&x)[kBlockValues],
                                           char* destination) {
  if (type == GGML_TYPE_Q8_0) {
    *reinterpret_cast<block_q8_0*>(destination) = QuantizeQ8_0(x);
  } else {
    *reinterpret_cast<block_q4_0*>(destination) = QuantizeQ4_0(x);
  }
}

}  // namespace llmp::kernels::ggml::kv

#endif  // LLMP_KERNELS_GGML_KV_QUANT_CUH_
