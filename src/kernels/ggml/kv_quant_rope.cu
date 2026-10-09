// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// NEOX RoPE fused with a quantizing cache store (kv_quant.h
// RopeSetRowsQuant): a thread per block of 32 values of a token's row, each
// value rotated as GGML's rope_neox computes it (rope.cu at the pin: the
// angle, YaRN's ramp and magnitude, the frequency factor, the pair's
// products; copied, as its functions are local to rope.cu), then the block
// quantized by kv_quant.cuh. Built with GGML's device flags
// (CMakeLists.txt), as rope.cu is, so that each rotated value is the F32 the
// RoPE node would hold.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>

#include "base/bytes.h"
#include "common.cuh"
#include "ggml.h"
#include "kernels/ggml/kv_quant.cuh"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/launch.h"

namespace llmp::kernels::ggml {
namespace {

struct CorrDims {
  float v[2];
};

// rope.cu's rope_yarn_ramp and rope_yarn<true>, verbatim but for names.
__device__ float RopeYarnRamp(const float low, const float high, const int i0) {
  const float y = (i0 / 2 - low) / max(0.001f, high - low);
  return 1.0f - min(1.0f, max(0.0f, y));
}

__device__ void RopeYarn(const float theta_extrap, const float freq_scale, const CorrDims corr_dims,
                         const int64_t i0, const float ext_factor, float mscale, float& cos_theta,
                         float& sin_theta) {
  float theta_interp = freq_scale * theta_extrap;
  float theta = theta_interp;
  if (ext_factor != 0.0f) {
    float ramp_mix = RopeYarnRamp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
    theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
    mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
  }
  cos_theta = cosf(theta) * mscale;
  sin_theta = sinf(theta) * mscale;
}

struct RopeStore {
  const float* x;
  const std::int32_t* pos;
  const float* freq_factors;
  const std::int64_t* ids;
  char* destination;
  std::uint64_t x_head_stride;    // floats
  std::uint64_t x_token_stride;   // floats
  std::uint64_t destination_row;  // bytes
  uint3 blocks_per_head;
  uint3 blocks_per_row;
  std::uint32_t total;  // blocks
  int n_dims;
  float freq_scale, ext_factor, attn_factor, theta_scale;
  CorrDims corr_dims;
  ggml_type type;
};

template <bool kHasFf>
__global__ void RopeQuantizedStore(const __grid_constant__ RopeStore s) {
  const std::uint64_t flat = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (flat >= s.total) {
    return;
  }
  auto split = fast_div_modulo(static_cast<std::uint32_t>(flat), s.blocks_per_row);
  const std::uint32_t block = split.y;  // within the token's row
  const std::uint32_t token = split.x;
  split = fast_div_modulo(block, s.blocks_per_head);
  const std::uint32_t head = split.x;
  const int first = static_cast<int>(split.y) * kv::kBlockValues;  // channel within the head
  const float* x = s.x + head * s.x_head_stride + token * s.x_token_stride;
  const int half_dims = s.n_dims / 2;
  float values[kv::kBlockValues];
#pragma unroll
  for (int j = 0; j < kv::kBlockValues; ++j) {
    const int channel = first + j;
    if (channel >= s.n_dims) {
      values[j] = x[channel];
      continue;
    }
    const int pair = channel < half_dims ? channel : channel - half_dims;
    // rope_neox's i0 (n_offs is 0): the pair's even index.
    const int iw = 2 * pair;
    const float theta_base = s.pos[token] * powf(s.theta_scale, iw / 2.0f);
    const float freq_factor = kHasFf ? s.freq_factors[iw / 2] : 1.0f;
    float cos_theta;
    float sin_theta;
    RopeYarn(theta_base / freq_factor, s.freq_scale, s.corr_dims, iw, s.ext_factor, s.attn_factor,
             cos_theta, sin_theta);
    const float x0 = x[pair];
    const float x1 = x[pair + half_dims];
    values[j] =
        channel < half_dims ? x0 * cos_theta - x1 * sin_theta : x0 * sin_theta + x1 * cos_theta;
  }
  const std::uint64_t block_bytes =
      s.type == GGML_TYPE_Q8_0 ? sizeof(block_q8_0) : sizeof(block_q4_0);
  char* destination = s.destination + s.ids[token] * s.destination_row + block * block_bytes;
  kv::StoreBlock(s.type, values, destination);
}

constexpr unsigned kThreads = 256;

}  // namespace

std::expected<void, KernelFailure> RopeSetRowsQuant(LaunchContext& launch, ggml_tensor* rope,
                                                    ggml_tensor* set_rows) {
  if (auto checked = CheckRopeSetRowsQuant(rope, set_rows); !checked) {
    return checked;
  }
  const ggml_tensor* x = rope->src[0];
  RopeStore s{};
  s.x = static_cast<const float*>(x->data);
  s.pos = static_cast<const std::int32_t*>(rope->src[1]->data);
  s.freq_factors =
      rope->src[2] != nullptr ? static_cast<const float*>(rope->src[2]->data) : nullptr;
  s.ids = static_cast<const std::int64_t*>(set_rows->src[1]->data);
  s.destination = static_cast<char*>(set_rows->data);
  s.x_head_stride = x->nb[1] / sizeof(float);
  s.x_token_stride = x->nb[2] / sizeof(float);
  s.destination_row = set_rows->nb[1];
  s.blocks_per_head = init_fastdiv_values(static_cast<std::uint32_t>(x->ne[0] / kv::kBlockValues));
  const auto per_row = static_cast<std::uint32_t>((x->ne[0] / kv::kBlockValues) * x->ne[1]);
  s.blocks_per_row = init_fastdiv_values(per_row);
  s.total = per_row * static_cast<std::uint32_t>(x->ne[2]);
  // ggml_cuda_op_rope_impl's parameters (rope.cu): n_dims, n_ctx_orig and
  // YaRN's from op_params, theta_scale and the correction dimensions on
  // the host as it computes them.
  s.n_dims = rope->op_params[1];
  const int n_ctx_orig = rope->op_params[4];
  float freq_base = 0;
  float beta_fast = 0;
  float beta_slow = 0;
  std::memcpy(&freq_base, rope->op_params + 5, sizeof(float));
  std::memcpy(&s.freq_scale, rope->op_params + 6, sizeof(float));
  std::memcpy(&s.ext_factor, rope->op_params + 7, sizeof(float));
  std::memcpy(&s.attn_factor, rope->op_params + 8, sizeof(float));
  std::memcpy(&beta_fast, rope->op_params + 9, sizeof(float));
  std::memcpy(&beta_slow, rope->op_params + 10, sizeof(float));
  ggml_rope_yarn_corr_dims(s.n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, s.corr_dims.v);
  s.theta_scale = powf(freq_base, -2.0f / static_cast<float>(s.n_dims));
  s.type = set_rows->type;
  const auto blocks = static_cast<unsigned>((std::uint64_t{s.total} + kThreads - 1) / kThreads);
  const bool has_ff = s.freq_factors != nullptr;
  return launch.Run(base::Bytes(0), [s, blocks, has_ff](ggml_backend_cuda_context& context) {
    if (has_ff) {
      RopeQuantizedStore<true><<<blocks, kThreads, 0, context.stream()>>>(s);
    } else {
      RopeQuantizedStore<false><<<blocks, kThreads, 0, context.stream()>>>(s);
    }
  });
}

}  // namespace llmp::kernels::ggml
