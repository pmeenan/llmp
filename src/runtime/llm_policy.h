// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// One mechanical serving policy, resolved from admitted component/layout and
// compiled backend capabilities. A request expresses preferences, never support
// or numerical authority. The value owns no borrowed source/backend objects.

#ifndef LLMP_RUNTIME_LLM_POLICY_H_
#define LLMP_RUNTIME_LLM_POLICY_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>

#include "engine/execution_contract.h"
#include "kernels/common/lower.h"
#include "model/batching.h"
#include "model/state_contract.h"

namespace llmp::runtime {

struct LlmSourceCapabilities {
  model::Batching batching;
  std::uint32_t context = 0, max_rows = 0;
  bool indexed_masks = false;
};

struct LlmPolicyRequest {
  std::uint32_t owners = 0, read_align = 256;
  std::uint32_t read_buckets = 0;  // read classes per doubling (execution::ReadCellsFor)
  bool wave_lanes = false;
  std::uint64_t lane_scratch_bytes = 0;
  // Admitted composition constraints, independently of the requested settings.
  bool owners_alone = false, prompts_alone = false, common_read = false;
  std::uint32_t lookahead = 0;
  bool capture_ahead = true, prepare_ahead = true, state_only = true;
  bool trial_mixed_outputs = false;
  bool trial_mixed_causal = false;
};

// Mechanical intersection for an internal supplied/derived domain. Complete
// source sealing and independent numerical authorization remain mandatory.
std::expected<void, std::string> ValidateServingEnvelope(
    const engine::NumericalDomain& domain, std::span<const model::StateLayoutDerived> layouts,
    std::span<const model::Batching> batching, std::uint32_t owners);

class LlmServingPolicy {
 public:
  static std::expected<LlmServingPolicy, std::string> Resolve(
      const LlmSourceCapabilities& source, const kernels::BackendCapabilities& backend,
      const LlmPolicyRequest& request);

  const model::Batching& batching() const { return batching_; }
  const kernels::LaneConfiguration& lanes() const { return lanes_; }
  std::uint32_t owners() const { return owners_; }
  std::uint32_t read_align() const { return read_align_; }
  std::uint32_t read_buckets() const { return read_buckets_; }
  std::uint32_t lookahead() const { return lookahead_; }
  bool device_tokens() const { return device_tokens_; }
  bool indexed_masks() const { return indexed_masks_; }
  bool capture_ahead() const { return capture_ahead_; }
  bool prepare_ahead() const { return prepare_ahead_; }
  bool state_only() const { return state_only_; }

 private:
  model::Batching batching_;
  kernels::LaneConfiguration lanes_;
  std::uint32_t owners_ = 0, read_align_ = 0, read_buckets_ = 0, lookahead_ = 0;
  bool device_tokens_ = false, indexed_masks_ = false;
  bool capture_ahead_ = false, prepare_ahead_ = false, state_only_ = false;
};

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_LLM_POLICY_H_
