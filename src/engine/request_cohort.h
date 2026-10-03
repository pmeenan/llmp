// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A runner's independent request states (docs/engine.md, "Independent
// request state"): the skeleton's part of serving several conversations of
// one model at once. A runner keeps the request states its model was set up
// with (at most kMaxRequestSlots), each its own LiveState (its caches, its
// drafter's, its verify snapshot) at stable addresses, beside one set of
// weights, staging and launch context. This holds what is the same for
// every family:
//
// - the active set a driver selected between completed units, and the rule
//   that more than one active slot runs only under one held stream request
//   over the execution closure (shared extents and every active slot's
//   state), so no job runs over a peer's state unprotected;
// - the closures: everything (shared and every slot's state, a swap's
//   load), the state fence, the execution closure, each slot's own fence;
//   built in one catalog pass and the held request refreshed to the
//   execution closure;
// - the cohort's health: a lost protection, an unknown completion or a
//   failed shared job whose effect cannot be proven local faults the whole
//   cohort, quarantining every slot until retirement. A clean, fenced
//   refusal stays the refusing slot's.
//
// Nothing here dispatches work or owns memory; the runner's slots do.

#ifndef JITLLM_ENGINE_REQUEST_COHORT_H_
#define JITLLM_ENGINE_REQUEST_COHORT_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>

#include "catalog/catalog.h"
#include "engine/live_state.h"
#include "engine/paged_node.h"

namespace jitllm::engine {

// The most request slots a runner keeps: a model's configured cap
// (`[models.<name>] max_slots`, docs/runtime-serving.md#request-slots) is at
// most this. It is the joined products' row capacity (kDsv4WaveRows, the
// wave kernels' sixteen columns): sixteen one-row decode steps fill one
// wave. Wider waves raise it.
inline constexpr std::size_t kMaxRequestSlots = 16;
// A set of request slots, bit i slot i.
using SlotMask = std::uint32_t;
static_assert(kMaxRequestSlots <= sizeof(SlotMask) * 8);

class RequestCohort {
 public:
  using Status = engine::Status;
  static constexpr std::size_t kSlots = kMaxRequestSlots;

  // `model` names it in refusals ("DeepSeek"); `slots` the request slots
  // its runner provisioned (set_slots once it knows them).
  explicit RequestCohort(std::string_view model, std::size_t slots = 1)
      : model_(model), slots_(slots) {}

  SlotMask active() const { return active_; }
  bool faulted() const { return faulted_; }
  std::size_t slots() const { return slots_; }
  // Before any selection but the default request's (a runner's Setup).
  void set_slots(std::size_t slots) { slots_ = slots; }
  bool IsActive(std::uint32_t slot) const {
    return slot < slots_ && (active_ & (SlotMask{1} << slot)) != 0;
  }

  // Before a slot's work: the cohort healthy, the slot active, and several
  // active slots only under a held request on `stream`.
  Status Check(const PagedNode& node, std::uint32_t stream, std::uint32_t slot) const;
  // The mask of distinct provisioned slots, refused otherwise.
  std::expected<SlotMask, std::string> MaskOf(std::span<const std::uint32_t> slots) const;
  // The active set (Refresh then protects it).
  void Select(SlotMask mask) { active_ = mask; }

  // The closures over `shared` extents and the slots' states (`states[i]`
  // slot i's; null for a slot not provisioned), `protected_mask` the slots
  // the execution closure holds. On the scheduler's thread (a Call).
  struct Closures {
    catalog::Closure everything;
    catalog::Closure fence;
    catalog::Closure execution;
    std::array<catalog::Closure, kSlots> slot_fences;
  };
  std::expected<Closures, std::string> Build(const catalog::Catalog& catalog,
                                             std::span<const catalog::ExtentId> shared,
                                             std::span<const LiveState* const> states,
                                             SlotMask protected_mask) const;
  // The held request (if any) refreshed to `execution`; a refusal may have
  // ended the old lease, so it faults the cohort.
  Status Hold(PagedNode& node, std::uint32_t stream, const catalog::Closure& execution,
              std::span<LiveState* const> states);

  // Every slot quarantined; no further dispatch until retirement.
  void Fault(std::span<LiveState* const> states);
  // After a failed job: a scheduler fault, an execution extent no longer at
  // its generation or quarantined, or several active slots without a held
  // request fault the cohort. Otherwise the failure stays local.
  void CheckFailedJob(PagedNode& node, std::uint32_t stream, const catalog::Closure& execution,
                      std::span<LiveState* const> states);

 private:
  std::string_view model_;
  std::size_t slots_ = 1;
  SlotMask active_ = 1;  // the default request until selected
  bool faulted_ = false;
};

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_REQUEST_COHORT_H_
