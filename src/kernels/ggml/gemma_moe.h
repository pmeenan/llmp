// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Standalone checked calls to the original pinned GGML Gemma MoE launchers.
// No graph rewrite, dispatch policy, allocation or completion ownership.
#ifndef LLMP_KERNELS_GGML_GEMMA_MOE_H_
#define LLMP_KERNELS_GGML_GEMMA_MOE_H_

#include <cstdint>
#include <expected>
#include <limits>

#include "base/bytes.h"
#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml {
class LaunchContext;
inline constexpr std::int64_t kGemmaMoeMaxRows = std::numeric_limits<int>::max();
// Exactly the specializations compiled by upstream topk-moe.cu.
constexpr bool GemmaRoutingShapeFits(std::int64_t experts, std::int64_t used, std::int64_t rows) {
  return experts > 0 &&
         (((experts & (experts - 1)) == 0 && experts <= 512) || experts == 288 || experts == 576) &&
         used > 0 && used <= experts && rows > 0 &&
         rows <= std::numeric_limits<int>::max() / experts &&
         rows <= std::numeric_limits<int>::max() - 7;  // upstream ceil(rows/8)
}
// Reduction uses uint64 element offsets and a 2-D grid, width tiled by 256.
constexpr bool GemmaReductionShapeFits(std::int64_t width, std::int64_t used, std::int64_t rows) {
  return width > 0 && width <= 65535LL * 256 && used > 0 &&
         used <= std::numeric_limits<int>::max() && rows > 0 &&
         rows <= std::numeric_limits<int>::max() &&
         static_cast<std::uint64_t>(rows) <= std::numeric_limits<std::uint64_t>::max() / 4 /
                                                 static_cast<std::uint64_t>(width) /
                                                 static_cast<std::uint64_t>(used);
}
inline constexpr float kGemmaRouteClamp = 0x1p-14f;

// Bytes mapped starting at tensor->data. The caller funds and protects this
// storage until completion, including every replay. A span is not residency
// proof; successful Run queues work, not proof that it finished. kUnknown
// retains all owners until recovery establishes quiescence.
struct GemmaMoeOperand {
  const ggml_tensor* tensor = nullptr;
  base::Bytes bytes;
};
enum class GemmaRouteIdsUse : std::uint8_t { kSelectedTop8, kFullSort };
struct GemmaRouting {
  // F32 [experts,rows,1,1], F32 [1,used,rows,1].
  GemmaMoeOperand logits, weights;
  // I32 [used,rows,1,1] zero-offset VIEW of a canonical [experts,rows,1,1]
  // root. bytes MUST cover the entire root, including its unwritten tail.
  // Only the first `used` entries in each row are initialized by this launcher.
  // A graph caller must refuse fusion if any consumer needs the full sort,
  // router probabilities or normalization intermediates; use their original
  // primitive producer chain. This standalone API cannot inspect consumers.
  GemmaMoeOperand ids;
  GemmaRouteIdsUse ids_use = GemmaRouteIdsUse::kSelectedTop8;
  float denominator_min = kGemmaRouteClamp;
};
struct GemmaScaledReduction {
  // Contiguous F32: [width,used,rows,1], [1,used,rows,1], [1,used,rows,1],
  // [width,rows,1,1]. All selected scales/contributions are required.
  GemmaMoeOperand experts, scales, weights, values;
};

std::expected<void, KernelFailure> CheckGemmaRouting(const GemmaRouting& desc);
std::expected<void, KernelFailure> CheckGemmaScaledReduction(const GemmaScaledReduction& desc);

// CUDA only; validation refusal queues nothing, scratch0. Routing selects
// softmax/top8/normalized weights with the exact Gemma clamp; reduction uses
// original (expert*scale)*weight and ascending selected-slot accumulation
// under pinned fast-math flags, including their possible contraction. Neither
// operation is the DeepSeek explicitly rounded two-input reducer. Finite
// operand contents are the caller's contract, not host metadata validation.
std::expected<void, KernelFailure> RunGemmaRouting(LaunchContext& launch, const GemmaRouting& desc);
std::expected<void, KernelFailure> RunGemmaScaledReduction(LaunchContext& launch,
                                                           const GemmaScaledReduction& desc);
}  // namespace llmp::kernels::ggml
#endif  // LLMP_KERNELS_GGML_GEMMA_MOE_H_
