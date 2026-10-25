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
    const float freq_factor = kHasFf ? s.freq_factors[iw / 2] : 1.0f;
    float cos_theta;
    float sin_theta;
    const float theta_base = s.pos[token] * powf(s.theta_scale, iw / 2.0f);
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

}  // namespace

std::expected<void, KernelFailure> RopeSetRowsQuant(LaunchContext& launch, ggml_tensor* rope,
                                                    ggml_tensor* set_rows) {
  auto described = DescribeRopeSetRowsQuantLaunch(rope, set_rows);
  if (!described) return std::unexpected(described.error());
  const ggml_tensor* x = rope->src[0];
  RopeStore s{};
  s.x = static_cast<const float*>(x->data);
  s.pos = static_cast<const std::int32_t*>(rope->src[1]->data);
  s.freq_factors =
      rope->src[2] != nullptr ? static_cast<const float*>(rope->src[2]->data) : nullptr;
  s.ids = static_cast<const std::int64_t*>(set_rows->src[1]->data);
  s.destination = static_cast<char*>(set_rows->data);
  s.x_head_stride = described->x_head_stride;
  s.x_token_stride = described->x_token_stride;
  s.destination_row = described->destination_row_stride;
  s.blocks_per_head = init_fastdiv_values(described->blocks_per_head);
  s.blocks_per_row = init_fastdiv_values(described->blocks_per_row);
  s.total = described->total_blocks;
  s.n_dims = described->n_dims;
  s.freq_scale = described->freq_scale;
  s.ext_factor = described->ext_factor;
  s.attn_factor = described->attn_factor;
  s.theta_scale = described->theta_scale;
  s.corr_dims.v[0] = described->correction_dims[0];
  s.corr_dims.v[1] = described->correction_dims[1];
  s.type = described->type;
  const auto blocks = described->grid_x;
  const auto threads = described->block_x;
  const auto shared = described->dynamic_shared_bytes;
  const bool has_ff = described->has_frequency_factors;
  return launch.Run(base::Bytes(described->pool_bytes),
                    [s, blocks, threads, shared, has_ff](ggml_backend_cuda_context& context) {
                      if (has_ff) {
                        RopeQuantizedStore<true><<<blocks, threads, shared, context.stream()>>>(s);
                      } else {
                        RopeQuantizedStore<false><<<blocks, threads, shared, context.stream()>>>(s);
                      }
                    });
}

}  // namespace llmp::kernels::ggml
