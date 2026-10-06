// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Explicit independent cache roots for the bounded Gemma C4 diagnostic.
// No selector uses this operation automatically. The caller holds every real
// operand span, output and workspace through completion/captured graph lifetime.
#ifndef JITLLM_KERNELS_GGML_FATTN_OWNER_H_
#define JITLLM_KERNELS_GGML_FATTN_OWNER_H_

#include <array>
#include <cstdint>
#include <expected>

#include "kernels/ggml/ops_ext.h"

namespace jitllm::kernels::ggml {

struct FlashAttnOwners {
  const ggml_tensor* q = nullptr;
  const ggml_tensor* mask = nullptr;
  std::array<const ggml_tensor*, 4> k{}, v{};
  ggml_tensor* output = nullptr;
};

// F32 Q [D,1,heads,4], F16 mask [256,32,1,4], four actual F16 K/V
// views [D,256,KVheads,1], packed F32 result [D,heads,1,4]. D256/GQA2
// or D512/GQA8 only; heads 16 or 32. Scale is fixed at 1 (Q already
// normalized/scaled by the caller). No sinks, softcap or sparse gather.
std::expected<void, KernelFailure> CheckFlashAttnOwners(const FlashAttnOwners& inputs);

struct FlashAttnOwnersPlan {
  FlashAttnMmaPlan original;
  std::uint64_t shared_bytes = 0;
  int threads = 0;
  int original_blocks_per_sm = 0;
  int owner_blocks_per_sm = 0;
};

// Geometry comes from the original compiled packed MMA kernel. The owner's
// resource check may refuse, but never changes the grid/reduction partitions.
std::expected<FlashAttnOwnersPlan, KernelFailure> PlanFlashAttnOwners(
    const LaunchContext& launch, const FlashAttnOwners& inputs);
std::expected<void, KernelFailure> FlashAttnOwnerRoots(LaunchContext& launch,
                                                       const FlashAttnOwners& inputs);

}  // namespace jitllm::kernels::ggml
#endif  // JITLLM_KERNELS_GGML_FATTN_OWNER_H_
