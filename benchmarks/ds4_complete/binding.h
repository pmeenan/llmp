// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Temporary original-pipeline binding. The runner maps/catalogs every range;
// this unit allocates only bounded host descriptors and never submits work.
#ifndef JITLLM_BENCHMARKS_DS4_COMPLETE_BINDING_H_
#define JITLLM_BENCHMARKS_DS4_COMPLETE_BINDING_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "benchmarks/ds4_complete/executor.h"
#include "model/dsv4_ds4_state.h"

namespace jitllm::artifact {
class Artifact;
}
namespace jitllm::engine {
class PagedWeights;
class Ds4PagedAlignedWeights;
struct Ds4PreparedWeightSet;
}  // namespace jitllm::engine

namespace jitllm::benchmarks::ds4_complete {

struct ScratchRequirement {
  std::string name;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 256;
};
struct ScratchPlan {
  // Independent concurrently live mappings; layers reuse them only after
  // the preceding native layer Job completes. No suballocation aliases.
  std::vector<ScratchRequirement> ranges;
  std::uint64_t logical_bytes = 0;
  std::uint64_t cublas_workspace_bytes = std::uint64_t{32} << 20U;
  // Conservative checked bound for the original MMQ/F16-vector launch pool.
  // Resolve still validates each actual dispatch need before submission.
  std::uint64_t launch_workspace_bytes = 0;
};
struct NamedScratch {
  std::string_view name;
  kg::Ds4CacheBuffer storage{};
};

// Initial matched scope: context8192, T4096, first0/4096, all43 Flash layers,
// packed original cache primaries. Both chunks share this worst-case plan.
// Paid logical F32 cache proxies are explicit scratch, never hidden state.
std::expected<ScratchPlan, std::string> PlanScratch(const model::Dsv4Profile& profile,
                                                    std::uint32_t context,
                                                    std::uint32_t device_sms);

struct BindingInputs {
  const artifact::Artifact* artifact = nullptr;
  const engine::PagedWeights* raw_weights = nullptr;
  const engine::Ds4PreparedWeightSet* prepared_plan = nullptr;
  const engine::Ds4PagedAlignedWeights* aligned_weights = nullptr;
  const model::Ds4BaselineStateLayout* state = nullptr;
  // One stable virtual state reservation with mapped/charged used ranges.
  // Packed planes retain capacity but only initialized prefix is a read.
  kg::Ds4CacheBuffer state_storage{};
  std::span<const NamedScratch> scratch;
  std::span<const HashTable> hash_tables;
  std::uint32_t first = 0;
  std::uint32_t device_sms = 0;
  std::uint64_t storage_generation = 0;
  std::uint64_t model_generation = 0;
};

// Root uploads scratch named "tokens" and initializes scalar/decode-table
// state before Resolve. Root leases all paid_ranges in the Job closure.
// Hash table HOST owners outlive this result/Resolve; no device payload is
// trusted merely because an address was supplied. Final head exists only
// for the second chunk, preserving the original requested output cadence.
// Refuses missing format, representation, mapping, scratch or original tier;
// no generic native arithmetic substitutes for an unavailable original path.
std::expected<Chunk, std::string> BindChunk(const model::Dsv4Profile& profile,
                                            const BindingInputs& inputs);

}  // namespace jitllm::benchmarks::ds4_complete
#endif  // JITLLM_BENCHMARKS_DS4_COMPLETE_BINDING_H_
