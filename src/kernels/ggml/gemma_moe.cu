// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <limits>

#include "kernels/ggml/gemma_moe.h"
#include "kernels/ggml/launch.h"
#include "moe-weighted-reduction.cuh"
#include "topk-moe.cuh"

namespace jitllm::kernels::ggml {
std::expected<void, KernelFailure> RunGemmaRouting(LaunchContext& launch,
                                                   const GemmaRouting& desc) {
  if (auto checked = CheckGemmaRouting(desc); !checked) return checked;
  return launch.Run(base::Bytes(0), [&desc](ggml_backend_cuda_context& ctx) {
    // The launcher reads only clamp metadata synchronously, not its data.
    ggml_tensor clamp{};
    const float bounds[] = {kGemmaRouteClamp, std::numeric_limits<float>::infinity()};
    std::memcpy(clamp.op_params, bounds, sizeof(bounds));
    const ggml_cuda_topk_moe_args args{.softmax = true, .norm = true};
    ggml_cuda_op_topk_moe(ctx, desc.logits.tensor, const_cast<ggml_tensor*>(desc.weights.tensor),
                          const_cast<ggml_tensor*>(desc.ids.tensor), &clamp, nullptr, nullptr,
                          args);
  });
}
std::expected<void, KernelFailure> RunGemmaScaledReduction(LaunchContext& launch,
                                                           const GemmaScaledReduction& desc) {
  if (auto checked = CheckGemmaScaledReduction(desc); !checked) return checked;
  return launch.Run(base::Bytes(0), [&desc](ggml_backend_cuda_context& ctx) {
    ggml_cuda_op_moe_weighted_reduction(ctx, desc.experts.tensor, desc.scales.tensor,
                                        desc.weights.tensor,
                                        const_cast<ggml_tensor*>(desc.values.tensor));
  });
}
}  // namespace jitllm::kernels::ggml
