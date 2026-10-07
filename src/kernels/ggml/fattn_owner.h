// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Explicit independent cache roots for checked one-query Gemma owner cohorts.
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
  // Two/three roots use their whole physical-stream grid directly. Four roots
  // per call in an eligible whole-eight or whole-twelve wave split
  // its original stream-K grid equally between two or three calls. Partial cohorts
  // keep that whole grid in each offset-filtered group. Never a root count.
  std::uint32_t logical_cohort = 4;
  // Actual independent roots. Unused fixed-carrier slots must be null.
  std::uint32_t owner_count = 4;
  // Global sequence origin for an equal-width partial cohort. Legacy paths use zero.
  std::uint32_t owner_offset = 0;
  // Closed Gemma2 H8/C2 specialization; zero preserves every earlier path.
  std::uint32_t logit_softcap = 0;
  // Opt-in Gemma2 cap50/H8 or no-cap Gemma3 H8/Gemma4 H16/H32 C2.
  // Mask width is logical; each aligned K/V
  // view is actual. No partial cohorts, sinks, sparse gather or non-cell-major layouts.
  bool bounded_roots = false;
};

// F32 Q [D,1,heads,N], F16 mask [cells,32,1,N], N actual F16 K/V
// views [D,cells,KVheads,1], packed F32 result [D,heads,1,N], N=1..4. A single
// root is allowed only as a canonical partial-cohort tail. Cells are
// multiples of 256 through 16384; executed operand views fit 64 MiB.
// Full backing parents may fit 1 GiB, with checked address/view containment;
// this does not broaden the readable prefix or the shader index domain.
// D256/GQA2
// or D512/GQA8 with heads 16 or 32; D256/H8/GQA2 is additionally
// admitted for canonical C2/C3/C4, whole C8/C12 four-root carriers and
// partial C5..7/C9..11 root groups. Wider cohorts require zero softcap;
// partial offsets/counts must identify a complete real group or its real tail.
// Scale is fixed at 1 (Q already
// normalized/scaled by the caller). Softcap50 is admitted only at D256/H8/C2;
// all other contracts require zero. Bounded roots require actual/logical C2
// at offset zero: cap50/H8/D256, no-cap/H8/D256 or no-cap/H16/H32 at D256/D512. Each
// actual width is aligned to256 and the common mask equals their maximum.
// No sinks or sparse gather.
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
constexpr bool PartialOwnerCohort(std::uint32_t count) {
  return count == 5 || count == 6 || count == 7 || count == 9 || count == 10 || count == 11;
}
struct OwnerPartition {
  int cohort_blocks = 0, quad_blocks = 0;
  std::uint32_t effective_cohort = 4;
};
// Host-only original grid arithmetic; max_blocks is actual occupancy times
// actual SMs. Indivisible whole-cohort grids preserve the supported four-root path.
std::expected<OwnerPartition, KernelFailure> PlanOwnerPartition(int max_blocks, int kv_tiles,
                                                                int kv_heads,
                                                                std::uint32_t logical_cohort,
                                                                bool prefer_whole_tiles = false);
}  // namespace detail

// Geometry comes from the original compiled packed MMA kernel. The owner's
// resource check may refuse, but never changes the grid/reduction partitions.
// Cohorts eight/twelve derive the whole-cohort grid before rounding, then
// divide it by two/three. Partial cohorts retain the whole original grid and
// filter global sequence ownership before addressing one to four actual roots.
std::expected<FlashAttnOwnersPlan, KernelFailure> PlanFlashAttnOwners(
    const LaunchContext& launch, const FlashAttnOwners& inputs);
std::expected<void, KernelFailure> FlashAttnOwnerRoots(LaunchContext& launch,
                                                       const FlashAttnOwners& inputs);

}  // namespace jitllm::kernels::ggml
#endif  // JITLLM_KERNELS_GGML_FATTN_OWNER_H_
