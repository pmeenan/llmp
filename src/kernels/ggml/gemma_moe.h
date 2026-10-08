// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Standalone checked calls to the original pinned GGML Gemma MoE launchers.
// No graph rewrite, dispatch policy, allocation or completion ownership.
#ifndef LLMP_KERNELS_GGML_GEMMA_MOE_H_
#define LLMP_KERNELS_GGML_GEMMA_MOE_H_

#include <cstdint>
#include <expected>

#include "base/bytes.h"
#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace llmp::kernels::ggml {
class LaunchContext;
inline constexpr std::int64_t kGemmaMoeMaxRows = 8192;
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
  // F32 [128,rows,1,1], F32 [1,8,rows,1].
  GemmaMoeOperand logits, weights;
  // I32 [8,rows,1,1] zero-offset VIEW of a canonical [128,rows,1,1]
  // root. bytes MUST cover the entire root, including its unwritten tail.
  // Only the first8 entries in each row are initialized by this launcher.
  // A graph caller must refuse fusion if any consumer needs the full sort,
  // router probabilities or normalization intermediates; use their original
  // primitive producer chain. This standalone API cannot inspect consumers.
  GemmaMoeOperand ids;
  GemmaRouteIdsUse ids_use = GemmaRouteIdsUse::kSelectedTop8;
  float denominator_min = kGemmaRouteClamp;
};
struct GemmaScaledReduction {
  // Contiguous F32: [2816,8,rows,1], [1,8,rows,1], [1,8,rows,1],
  // [2816,rows,1,1]. All selected scales/contributions are required.
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
