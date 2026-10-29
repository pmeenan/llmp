// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The scaled rotations GGML's rope kernels do not compute: Llama 3's
// frequency smoothing (GGML_OP_ROPE with ext_factor < 0 and an original
// context) and proportional rotation of the first `rotated_dims` of the
// rotated dimensions (op_params[11]), into F32 rows or, through row ids,
// F16 cache rows: one kernel for ops.cu's rope and rope-store and
// ops_ext.cu's rope_ext.
//
// Everything here has internal linkage on purpose: its includers compile
// with different math flags (ops_ext.cu with -use_fast_math, ops.cu
// without; CMakeLists.txt), and each keeps its own arithmetic.

#ifndef LLMP_KERNELS_GGML_ROPE_SCALED_CUH_
#define LLMP_KERNELS_GGML_ROPE_SCALED_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ggml.h"

namespace llmp::kernels::ggml {
namespace {  // NOLINT(cert-dcl59-cpp,google-build-namespaces): see above

template <typename T>
__device__ T ScaledRopeOut(float v);
template <>
__device__ float ScaledRopeOut<float>(float v) {
  return v;
}
template <>
__device__ half ScaledRopeOut<half>(float v) {
  return __float2half(v);
}

// One block a row; dimensions outside the rotation (before n_offs, past
// n_dims, or past the proportional `rotated_dims`) copied. With kSetRows,
// row i1 of token i2 goes to cache row row_ids[i2] (its heads packed).
template <typename DstT, bool kSetRows>
__global__ void ScaledRopeKernel(const char* x_bytes, const std::int32_t* pos_data,
                                 const float* freq_factors, const std::int64_t* row_ids,
                                 char* dst_bytes, std::int64_t ne0, std::int64_t ne1,
                                 std::int64_t ne2, std::size_t x_nb1, std::size_t x_nb2,
                                 std::size_t x_nb3, std::size_t dst_nb1, std::size_t dst_nb2,
                                 std::size_t dst_nb3, int n_dims, int n_offs, int n_ctx_orig,
                                 float freq_base, float freq_scale, float attn_factor,
                                 float beta_fast, float beta_slow, bool neox, bool inverse,
                                 bool llama3, int rotated_dims) {
  const std::int64_t flat = blockIdx.x;
  const std::int64_t i1 = flat % ne1;
  const std::int64_t rem = flat / ne1;
  const std::int64_t i2 = rem % ne2;
  const std::int64_t i3 = rem / ne2;
  const auto* x_row = reinterpret_cast<const float*>(
      x_bytes + static_cast<std::size_t>(i1) * x_nb1 + static_cast<std::size_t>(i2) * x_nb2 +
      static_cast<std::size_t>(i3) * x_nb3);
  DstT* dst_row = nullptr;
  if constexpr (kSetRows) {
    dst_row = reinterpret_cast<DstT*>(dst_bytes + static_cast<std::size_t>(row_ids[i2]) * dst_nb1) +
              i1 * ne0;
  } else {
    dst_row = reinterpret_cast<DstT*>(dst_bytes + static_cast<std::size_t>(i1) * dst_nb1 +
                                      static_cast<std::size_t>(i2) * dst_nb2 +
                                      static_cast<std::size_t>(i3) * dst_nb3);
  }
  const int pairs = n_dims / 2;
  const int active_pairs = rotated_dims >= 0 ? rotated_dims / 2 : pairs;
  for (int c = static_cast<int>(threadIdx.x); c < static_cast<int>(ne0);
       c += static_cast<int>(blockDim.x)) {
    const int relative = c - n_offs;
    if (relative < 0 || relative >= n_dims ||
        (neox ? relative % pairs : relative / 2) >= active_pairs) {
      dst_row[c] = ScaledRopeOut<DstT>(x_row[c]);
    }
  }
  const float pos = static_cast<float>(pos_data[i2]);
  const float old_ctx = static_cast<float>(n_ctx_orig);
  const float low_freq_wavelen = old_ctx / beta_slow;
  const float high_freq_wavelen = old_ctx / beta_fast;
  const float mscale = llama3 && attn_factor == 0.0f ? 1.0f : attn_factor;
  for (int pr = static_cast<int>(threadIdx.x); pr < active_pairs;
       pr += static_cast<int>(blockDim.x)) {
    const int idx0 = neox ? (n_offs + pr) : (n_offs + 2 * pr);
    const int idx1 = neox ? (idx0 + pairs) : (idx0 + 1);
    const float base_theta =
        powf(freq_base, (2.0f * static_cast<float>(pr)) / static_cast<float>(n_dims));
    const float freq = 1.0f / base_theta;
    const float wavelen = 6.28318530717958647692f / freq;
    float scaled_freq = llama3 ? freq : freq * freq_scale;
    if (llama3 && wavelen > low_freq_wavelen) {
      scaled_freq = freq * freq_scale;
    } else if (llama3 && wavelen > high_freq_wavelen && beta_fast > beta_slow) {
      const float smooth = (old_ctx / wavelen - beta_slow) / (beta_fast - beta_slow);
      scaled_freq = (1.0f - smooth) * freq * freq_scale + smooth * freq;
    }
    const float ff = freq_factors != nullptr ? freq_factors[pr] : 1.0f;
    const float theta = (pos * scaled_freq) / ff;
    const float c_val = cosf(theta) * mscale;
    const float s_val = (inverse ? -sinf(theta) : sinf(theta)) * mscale;
    const float x0 = x_row[idx0];
    const float x1 = x_row[idx1];
    dst_row[idx0] = ScaledRopeOut<DstT>(x0 * c_val - x1 * s_val);
    dst_row[idx1] = ScaledRopeOut<DstT>(x0 * s_val + x1 * c_val);
  }
}

// Llama 3's smoothing, as GGML's rope parameters carry it.
bool IsLlama3Rope(const ggml_tensor* rope) {
  float ext_factor = 0.0f;
  std::memcpy(&ext_factor, rope->op_params + 7, sizeof(float));
  return ext_factor < 0.0f && rope->op_params[4] > 0;
}

// `rope`'s rotation of its source into `rope` itself (set_rows null) or
// into set_rows's F16 cache rows at its row ids.
void LaunchScaledRope(cudaStream_t stream, const ggml_tensor* rope, ggml_tensor* set_rows) {
  const ggml_tensor* x = rope->src[0];
  const auto* pos = static_cast<const std::int32_t*>(rope->src[1]->data);
  const auto* ff =
      rope->src[2] != nullptr ? static_cast<const float*>(rope->src[2]->data) : nullptr;
  const int n_dims = rope->op_params[1];
  const int mode = rope->op_params[2];
  const int n_offs = rope->op_params[15];
  const int n_ctx_orig = rope->op_params[4];
  const int rotated_dims = rope->op_params[11] != 0 ? rope->op_params[11] - 1 : -1;
  float freq_base = 0.0f, freq_scale = 0.0f, attn_factor = 0.0f, beta_fast = 0.0f, beta_slow = 0.0f;
  std::memcpy(&freq_base, rope->op_params + 5, sizeof(float));
  std::memcpy(&freq_scale, rope->op_params + 6, sizeof(float));
  std::memcpy(&attn_factor, rope->op_params + 8, sizeof(float));
  std::memcpy(&beta_fast, rope->op_params + 9, sizeof(float));
  std::memcpy(&beta_slow, rope->op_params + 10, sizeof(float));
  const bool neox = (mode & GGML_ROPE_TYPE_NEOX) != 0;
  const bool inverse = rope->op == GGML_OP_ROPE_BACK;
  const bool llama3 = IsLlama3Rope(rope);
  const auto blocks = static_cast<unsigned>(x->ne[1] * x->ne[2] * x->ne[3]);
  if (blocks == 0) return;
  if (set_rows != nullptr) {
    const auto* ids = static_cast<const std::int64_t*>(set_rows->src[1]->data);
    ScaledRopeKernel<half, true><<<blocks, 128, 0, stream>>>(
        static_cast<const char*>(x->data), pos, ff, ids, static_cast<char*>(set_rows->data),
        x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], x->nb[3], set_rows->nb[1], 0, 0, n_dims,
        n_offs, n_ctx_orig, freq_base, freq_scale, attn_factor, beta_fast, beta_slow, neox, inverse,
        llama3, rotated_dims);
  } else {
    ScaledRopeKernel<float, false><<<blocks, 128, 0, stream>>>(
        static_cast<const char*>(x->data), pos, ff, nullptr, static_cast<char*>(rope->data),
        x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], x->nb[3], rope->nb[1], rope->nb[2],
        rope->nb[3], n_dims, n_offs, n_ctx_orig, freq_base, freq_scale, attn_factor, beta_fast,
        beta_slow, neox, inverse, llama3, rotated_dims);
  }
}

}  // namespace
}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_ROPE_SCALED_CUH_
