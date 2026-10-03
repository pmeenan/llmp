// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/request_cohort.h"

#include <bit>
#include <format>
#include <utility>
#include <vector>

#include "engine/support.h"

namespace jitllm::engine {

namespace {
using catalog::ExtentId;
using support::Error;
}  // namespace

Status RequestCohort::Check(const PagedNode& node, std::uint32_t stream, std::uint32_t slot) const {
  if (faulted_) {
    return Error(std::format("the {} cohort requires retirement", model_));
  }
  if (!IsActive(slot)) {
    return Error(std::format("the {} request slot {} is not active", model_, slot));
  }
  if (std::popcount(active_) > 1 && !node.InRequest(stream)) {
    return Error(std::format("active {} slots require one held stream request", model_));
  }
  return {};
}

std::expected<SlotMask, std::string> RequestCohort::MaskOf(
    std::span<const std::uint32_t> slots) const {
  if (slots.size() > slots_) {
    return Error(std::format("at most {} {} request slots", slots_, model_));
  }
  SlotMask mask = 0;
  for (const std::uint32_t slot : slots) {
    if (slot >= slots_) {
      return Error(std::format("a {} request slot outside the runner", model_));
    }
    const SlotMask bit = SlotMask{1} << slot;
    if ((mask & bit) != 0) {
      return Error(std::format("a {} request slot occurs twice", model_));
    }
    mask |= bit;
  }
  return mask;
}

std::expected<RequestCohort::Closures, std::string> RequestCohort::Build(
    const catalog::Catalog& catalog, std::span<const ExtentId> shared,
    std::span<const LiveState* const> states, SlotMask protected_mask) const {
  std::vector<ExtentId> all(shared.begin(), shared.end());
  std::vector<ExtentId> active(shared.begin(), shared.end());
  std::vector<ExtentId> state;
  Closures out;
  for (std::size_t slot = 0; slot < states.size() && slot < kSlots; ++slot) {
    if (states[slot] == nullptr) {
      continue;
    }
    const std::vector<ExtentId> live = states[slot]->extents();
    all.insert(all.end(), live.begin(), live.end());
    state.insert(state.end(), live.begin(), live.end());
    if ((protected_mask & (SlotMask{1} << slot)) != 0) {
      active.insert(active.end(), live.begin(), live.end());
    }
    auto fence = catalog.ClosureOfExtents(live);
    if (!fence) {
      return Error(std::format("a {} slot's state closure is no longer cataloged", model_));
    }
    out.slot_fences[slot] = std::move(*fence);
  }
  auto everything = catalog.ClosureOfExtents(all);
  auto fence = catalog.ClosureOfExtents(state);
  auto execution = catalog.ClosureOfExtents(active);
  if (!everything || !fence || !execution) {
    return Error(std::format("a {} cohort closure is no longer cataloged", model_));
  }
  out.everything = std::move(*everything);
  out.fence = std::move(*fence);
  out.execution = std::move(*execution);
  return out;
}

Status RequestCohort::Hold(PagedNode& node, std::uint32_t stream, const catalog::Closure& execution,
                           std::span<LiveState* const> states) {
  if (auto held = node.RefreshRequest(stream, execution); !held) {
    // RefreshRequest may have ended the previous lease before refusing its
    // replacement: no slot may dispatch after that loss of protection.
    Fault(states);
    return held;
  }
  return {};
}

void RequestCohort::Fault(std::span<LiveState* const> states) {
  faulted_ = true;
  for (LiveState* live : states) {
    if (live != nullptr) {
      live->Quarantine();
    }
  }
}

void RequestCohort::CheckFailedJob(PagedNode& node, std::uint32_t stream,
                                   const catalog::Closure& execution,
                                   std::span<LiveState* const> states) {
  // A job's Status alone does not tell a fenced refusal from a fault that
  // retains an operation: ask the scheduler and the catalog, on their thread.
  auto healthy = node.Call(
      [&]() -> Status {
        if (node.scheduler().fault()) {
          return Error("the shared scheduler faulted");
        }
        for (const auto& [id, generation] : execution.extents) {
          const auto view = node.catalog().Describe(id);
          if (!view || view->content_generation != generation ||
              view->state == catalog::ExtentState::kQuarantined) {
            return Error("the active closure is no longer usable");
          }
        }
        return {};
      },
      "checking a failed shared job");
  if (!healthy || (std::popcount(active_) > 1 && !node.InRequest(stream))) {
    Fault(states);
  }
}

}  // namespace jitllm::engine
