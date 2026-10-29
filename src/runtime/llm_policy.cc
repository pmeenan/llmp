// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/llm_policy.h"

#include <algorithm>

#include "engine/request_cohort.h"
#include "planner/lanes.h"

namespace llmp::runtime {

std::expected<void, std::string> ValidateServingEnvelope(
    const engine::NumericalDomain& domain, std::span<const model::StateLayoutDerived> layouts,
    std::span<const model::Batching> batching, std::uint32_t owners) {
  if (owners == 0 || owners > engine::kMaxRequestSlots || layouts.empty() ||
      domain.stages().size() != layouts.size() || batching.size() != layouts.size())
    return std::unexpected("execution envelope needs every actual stage and configured owner");
  for (std::size_t i = 0; i < layouts.size(); ++i) {
    const auto& state = layouts[i];
    const auto& caps = batching[i];
    if (state.context == 0 || state.max_rows == 0 || caps.max_segments == 0 ||
        caps.max_segments > engine::kMaxRequestSlots || domain.stages()[i].stage != i)
      return std::unexpected("execution envelope lacks an actual state/component capacity");
    const auto rows = caps.max_segment_rows == 0 ? state.max_rows
                                                 : std::min(state.max_rows, caps.max_segment_rows);
    for (const auto& phase : domain.stages()[i].phases) {
      if (phase.max_total_rows > UINT32_MAX ||
          phase.max_owners > std::min(owners, caps.max_segments) || phase.max_owner_rows > rows ||
          (caps.max_step_rows != 0 && phase.max_total_rows > caps.max_step_rows))
        return std::unexpected("execution envelope exceeds the actual state/component capacity");
    }
  }
  return {};
}

std::expected<LlmServingPolicy, std::string> LlmServingPolicy::Resolve(
    const LlmSourceCapabilities& source, const kernels::BackendCapabilities& backend,
    const LlmPolicyRequest& request) {
  if (source.context == 0 || source.max_rows == 0 || source.max_rows > source.context ||
      source.batching.max_segments == 0 ||
      source.batching.max_segments > engine::kMaxRequestSlots || request.owners == 0 ||
      request.owners > engine::kMaxRequestSlots || request.read_align == 0 ||
      request.read_align % 256 != 0)
    return std::unexpected("serving policy needs actual row, context, owner and read bounds");
  if (backend.owner_lanes > planner::kMaxLanes)
    return std::unexpected("serving policy received an invalid backend lane declaration");
  if (request.wave_lanes && (backend.owner_lanes == 0 || request.lane_scratch_bytes == 0 ||
                             request.lane_scratch_bytes % 256 != 0 ||
                             request.lane_scratch_bytes > UINT64_MAX / backend.owner_lanes))
    return std::unexpected("requested wave lanes lack a backend implementation or valid workspace");

  LlmServingPolicy result;
  result.owners_ = request.owners;
  result.read_align_ = request.read_align;
  result.read_buckets_ = request.read_buckets;
  result.batching_ = source.batching;
  result.batching_.max_segments = std::min(request.owners, source.batching.max_segments);
  if (request.owners_alone) result.batching_.max_segments = 1;
  if (request.prompts_alone) result.batching_.join_prefill = false;
  const auto rows = source.batching.max_segment_rows;
  result.batching_.max_segment_rows = rows == 0 ? source.max_rows : std::min(rows, source.max_rows);
  result.batching_.common_read |= request.common_read;
  if (request.trial_mixed_outputs && !backend.device_tokens)
    return std::unexpected("mixed output trial requires actual device-token lowering");
  result.batching_.mixed_greedy = request.trial_mixed_outputs && backend.device_tokens;
  if (request.trial_mixed_causal && !source.batching.join_prefill)
    return std::unexpected("mixed causal trial requires actual joined causal component graphs");
  result.batching_.mixed_causal = request.trial_mixed_causal && result.batching_.join_prefill &&
                                  result.batching_.max_segments > 1;
  result.batching_.joined_read_align =
      std::max(source.batching.joined_read_align, request.read_align);
  if (request.wave_lanes && result.batching_.max_segments > 1)
    result.lanes_ = {.count = backend.owner_lanes,
                     .scratch_bytes_per_lane = request.lane_scratch_bytes};
  result.device_tokens_ = backend.device_tokens;
  result.indexed_masks_ = source.indexed_masks && backend.indexed_masks;
  result.lookahead_ = request.lookahead;
  result.capture_ahead_ = request.capture_ahead;
  result.prepare_ahead_ = request.prepare_ahead;
  result.state_only_ = request.state_only;
  return result;
}

}  // namespace llmp::runtime
