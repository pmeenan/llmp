// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <expected>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/dsv4_qhead.h"
#include "kernels/ggml/launch.h"

namespace jitllm::kernels::ggml {
namespace {

// The locked GGML norm.cu's 256-thread reduction and normal-offset
// rope.cu's YaRN arithmetic. YaRN: MIT licensed, Copyright (c) 2023
// Jeffrey Quesnelle and Bowen Peng, as attributed by the locked source.
__global__ void QHeadKernel(const float* x, float* dst, const std::int32_t* pos, float eps,
                            float theta_scale, float freq_scale, float ext_factor,
                            float attn_factor, float corr0, float corr1) {
  const int tid = static_cast<int>(threadIdx.x);
  const auto row = static_cast<unsigned long long>(blockIdx.x);
  x += row * 512;
  dst += row * 512;
  float tmp = 0.0f;
  for (int col = tid; col < 512; col += 256) {
    const float xi = x[col];
    tmp += xi * xi;
  }
  // CUDA dynamic shared memory has a C array declaration by ABI.
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  extern __shared__ float sums[];
  tmp = block_reduce<block_reduce_method::SUM, 256>(tmp, sums);
  const float scale = rsqrtf((tmp / 512) + eps);
  const int i0 = 2 * tid;
  // The separate norm's F32 store rounds before either rotated product.
  const float x0 = __fmul_rn(scale, x[i0]);
  const float x1 = __fmul_rn(scale, x[i0 + 1]);
  if (i0 < 448) {
    dst[i0] = x0;
    dst[i0 + 1] = x1;
    return;
  }
  const int iw = i0 - 448;
  const auto position = pos[row / 64];
  const float theta_base =
      static_cast<float>(position) * powf(theta_scale, static_cast<float>(iw) / 2.0f);
  const float theta_interp = freq_scale * theta_base;
  float theta = theta_interp;
  float mscale = attn_factor;
  if (ext_factor != 0.0f) {
    const float y = ((static_cast<float>(iw) / 2.0f) - corr0) / max(0.001f, corr1 - corr0);
    const float ramp_mix = (1.0f - min(1.0f, max(0.0f, y))) * ext_factor;
    theta = (theta_interp * (1 - ramp_mix)) + (theta_base * ramp_mix);
    mscale *= 1.0f + (0.1f * logf(1.0f / freq_scale));
  }
  const float c = cosf(theta) * mscale;
  const float s = sinf(theta) * mscale;
  // Native rope_norm's actual SASS contracts x0's products. Source-level
  // equivalence alone allows the other operand to be contracted instead.
  dst[i0] = __fmaf_rn(c, x0, -__fmul_rn(s, x1));
  dst[i0 + 1] = __fmaf_rn(s, x0, __fmul_rn(c, x1));
}

}  // namespace

std::expected<void, KernelFailure> RunDsv4QHead(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckDsv4QHead(node); !checked) {
    return checked;
  }
  const auto p = Dsv4QHeadParamsOf(node);
  std::array<float, 2> corr{};
  ggml_rope_yarn_corr_dims(64, p.original_context, p.base, p.beta_fast, p.beta_slow, corr.data());
  const float theta_scale = powf(p.base, -2.0f / 64);
  const auto* x = static_cast<const float*>(node->src[0]->data);
  const auto* positions = static_cast<const std::int32_t*>(node->src[1]->data);
  auto* dst = static_cast<float*>(node->data);
  const auto blocks = static_cast<unsigned>(node->ne[2]) * 64U;
  const float corr0 = corr[0];
  const float corr1 = corr[1];
  return launch.Run(base::Bytes(0), [=](auto& context) {
    QHeadKernel<<<blocks, 256, 32 * sizeof(float), context.stream()>>>(
        x, dst, positions, p.eps, theta_scale, p.scale, p.extension, p.attention, corr0, corr1);
    CUDA_CHECK(cudaGetLastError());
  });
}

}  // namespace jitllm::kernels::ggml
