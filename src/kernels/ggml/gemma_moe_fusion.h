// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_GGML_GEMMA_MOE_FUSION_H_
#define LLMP_KERNELS_GGML_GEMMA_MOE_FUSION_H_

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "kernels/ggml/fusion.h"
#include "kernels/ggml/gemma_moe.h"

namespace llmp::kernels::ggml {
namespace detail {
class GraphReadIndex;
}
inline constexpr std::string_view kGemmaRouteName = "ggml.gemma.route.fused";
inline constexpr std::string_view kGemmaReduceName = "ggml.gemma.scaled_reduce.fused";

// All descriptors remain in the eventual step for conservative placement.
// The reduction descriptor list is bounded by the input graph window.
// These matchers select no execution policy. Operands'
// byte counts describe required logical backing, not catalog residency proof.
struct GemmaRoutingFusionNodes {
  std::array<ggml_tensor*, 10> nodes;
  GemmaRouting operands;
};
struct GemmaReductionFusionNodes {
  std::vector<ggml_tensor*> nodes;
  GemmaScaledReduction operands;
};
// Native normalized softmax/top-k pattern, including full ARGSORT backing.
// Selected IDs and normalized weights may be read/kept independently.
// Full sort, probabilities and other elided values require primitives.
std::optional<GemmaRoutingFusionNodes> GemmaRoutingFusionAt(
    GraphNodes graph, std::size_t index, std::span<ggml_tensor* const> keep = {},
    const detail::GraphReadIndex* reads = nullptr);
// Two products (expert*scale)*weight, the selected-slot views and their
// ascending dependent additions. Every elided producer/view must be private.
std::optional<GemmaReductionFusionNodes> GemmaReductionFusionAt(
    GraphNodes graph, std::size_t index, std::span<ggml_tensor* const> keep = {},
    const detail::GraphReadIndex* reads = nullptr);
}  // namespace llmp::kernels::ggml
#endif  // LLMP_KERNELS_GGML_GEMMA_MOE_FUSION_H_
