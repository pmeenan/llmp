// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_KERNELS_GGML_GEMMA_MOE_FUSION_H_
#define JITLLM_KERNELS_GGML_GEMMA_MOE_FUSION_H_

#include <array>
#include <cstddef>
#include <optional>
#include <span>

#include "kernels/ggml/fusion.h"
#include "kernels/ggml/gemma_moe.h"

namespace jitllm::kernels::ggml {
// All descriptors remain in the eventual step for conservative placement.
// These matchers allocate nothing and select no execution policy. Operands'
// byte counts describe required logical backing, not catalog residency proof.
struct GemmaRoutingFusionNodes {
  std::array<ggml_tensor*, 10> nodes;
  GemmaRouting operands;
};
struct GemmaReductionFusionNodes {
  std::array<ggml_tensor*, 17> nodes;
  GemmaScaledReduction operands;
};
// Exact native Gemma 128/top-eight pattern, including full ARGSORT backing.
// First-eight IDs and normalized weights may be read/kept independently.
// Full sort, probabilities and other elided values require primitives.
std::optional<GemmaRoutingFusionNodes> GemmaRoutingFusionAt(
    GraphNodes graph, std::size_t index, std::span<ggml_tensor* const> keep = {});
// Two products (expert*scale)*weight, eight selected-slot views, seven
// ascending dependent additions. Every elided producer/view must be private.
std::optional<GemmaReductionFusionNodes> GemmaReductionFusionAt(
    GraphNodes graph, std::size_t index, std::span<ggml_tensor* const> keep = {});
}  // namespace jitllm::kernels::ggml
#endif  // JITLLM_KERNELS_GGML_GEMMA_MOE_FUSION_H_
