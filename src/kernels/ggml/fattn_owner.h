// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Explicit independent cache roots for the checked Gemma C4 opt-in.
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
  // Four real roots per call; an eligible whole-eight wave splits its
  // original stream-K grid equally between two calls. Never a root count.
  std::uint32_t logical_cohort = 4;
};

// F32 Q [D,1,heads,4], F16 mask [cells,32,1,4], four actual F16 K/V
// views [D,cells,KVheads,1], packed F32 result [D,heads,1,4]. Cells are
// multiples of 256 through 16384; executed operand views fit 64 MiB.
// Full backing parents may fit 1 GiB, with checked address/view containment;
// this does not broaden the readable prefix or the shader index domain.
// D256/GQA2
// or D512/GQA8 only; heads 16 or 32. Scale is fixed at 1 (Q already
// normalized/scaled by the caller). No sinks, softcap or sparse gather.
std::expected<void, KernelFailure> CheckFlashAttnOwners(const FlashAttnOwners& inputs);

// Graph adapter for the distinct ten-source custom node. These never reinterpret
// a legacy FLASH descriptor or invent a contiguous K/V source allocation.
std::expected<FlashAttnOwners, KernelFailure> FlashAttnOwnersFromNode(ggml_tensor* node);
std::expected<void, KernelFailure> CheckFlashAttnOwnersNode(const ggml_tensor* node);

struct FlashAttnOwnersPlan {
  FlashAttnMmaPlan original;
  int cohort_blocks = 0;
  std::uint32_t effective_cohort = 4;
  std::uint64_t shared_bytes = 0;
  int threads = 0;
  int original_blocks_per_sm = 0;
  int owner_blocks_per_sm = 0;
};

namespace detail {
struct OwnerPartition {
  int cohort_blocks = 0, quad_blocks = 0;
  std::uint32_t effective_cohort = 4;
};
// Host-only original grid arithmetic; max_blocks is actual occupancy times
// actual SMs. Odd whole-eight grids preserve the supported four-root path.
std::expected<OwnerPartition, KernelFailure> PlanOwnerPartition(int max_blocks, int kv_tiles,
                                                                int kv_heads,
                                                                std::uint32_t logical_cohort);
}  // namespace detail

// Geometry comes from the original compiled packed MMA kernel. The owner's
// resource check may refuse, but never changes the grid/reduction partitions.
// Cohort eight derives the whole-eight grid before rounding, then divides it
// by two. The shader still receives only four actual roots, with no offset.
std::expected<FlashAttnOwnersPlan, KernelFailure> PlanFlashAttnOwners(
    const LaunchContext& launch, const FlashAttnOwners& inputs);
std::expected<void, KernelFailure> FlashAttnOwnerRoots(LaunchContext& launch,
                                                       const FlashAttnOwners& inputs);

}  // namespace jitllm::kernels::ggml
#endif  // JITLLM_KERNELS_GGML_FATTN_OWNER_H_
