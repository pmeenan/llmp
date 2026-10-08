// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef JITLLM_ENGINE_PREFILL_LOOKAHEAD_H_
#define JITLLM_ENGINE_PREFILL_LOOKAHEAD_H_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "engine/paged_node.h"
#include "engine/planned.h"

namespace jitllm::engine {

// One optional host-only plan built while the current device unit runs.
// Shape prediction and graph construction stay with the family. Fund before
// allocating its plan; Build must not bind implementations or query CUDA.
// The driver calls InstallAfterCompletion only after its current job succeeds,
// with its PlanStep still protecting borrowed plans and captures. The installer
// binds/checks coverage, then calls transfer immediately before the cache charge.
// A refused future plan never changes the already completed prefix.
template <typename Planned>
class PrefillLookahead {
 public:
  PrefillLookahead(PagedNode& node, std::uint64_t allowance)
      : PrefillLookahead(
            allowance,
            [&node](std::uint64_t bytes, bool required) {
              return node.ChargeHost(bytes, required);
            },
            [&node](std::uint64_t bytes) { node.UnchargeHost(bytes); }) {}
  PrefillLookahead(std::uint64_t allowance, PlanAccount::ChargeFn charge,
                   PlanAccount::ReleaseFn release)
      : allowance_(allowance) {
    account_.Bind(std::move(charge), std::move(release));
  }
  PrefillLookahead(const PrefillLookahead&) = delete;
  PrefillLookahead& operator=(const PrefillLookahead&) = delete;
  ~PrefillLookahead() { Abandon(); }

  bool Fund() {
    if (attempted_ || finished_) return false;
    attempted_ = true;
    funded_ = account_.Charge(allowance_, false);
    return funded_;
  }

  template <typename BuildFn>
  bool Build(BuildFn&& build) {
    if (!funded_ || built_ || finished_) return false;
    built_ = true;
    const auto started = std::chrono::steady_clock::now();
    auto result = std::forward<BuildFn>(build)();
    seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (!result || !*result || PlannedHostBytes(**result) > allowance_) return false;
    planned_ = std::move(*result);
    return true;
  }

  double seconds() const { return seconds_; }

  template <typename InstallFn>
  bool InstallAfterCompletion(InstallFn&& install) {
    if (!planned_ || finished_) return false;
    finished_ = true;
    // Ownership stays under the grant through binding/coverage failure. The
    // callback consumes the plan before returning, even when it cannot cache it.
    return static_cast<bool>(
        std::forward<InstallFn>(install)(std::move(planned_), seconds_, [this] { Release(); }));
  }

  void Abandon() {
    finished_ = true;
    // Explicit order: member destruction after this body's release is too late.
    planned_.reset();
    Release();
  }

 private:
  void Release() {
    account_.Uncharge(account_.bytes());
    funded_ = false;
  }

  PlanAccount account_;
  std::uint64_t allowance_;
  std::unique_ptr<Planned> planned_;
  double seconds_ = 0;
  bool attempted_ = false;
  bool funded_ = false;
  bool built_ = false;
  bool finished_ = false;
};

// A fixed number of independent optional plans, ordered nearest first. Each
// slot keeps the single-plan funding/destruction contract. The family deduplicates
// complete shapes and protects cached predictions before optional funding.
// BuildAll runs sequentially in one host callback; a later refusal never prevents
// an earlier ready plan from being installed after the current unit succeeds.
template <typename Planned, std::size_t Capacity>
class PrefillLookaheadGroup {
 public:
  static_assert(Capacity > 0);
  PrefillLookaheadGroup(PagedNode& node, std::uint64_t allowance)
      : PrefillLookaheadGroup(
            allowance,
            [&node](std::uint64_t bytes, bool required) {
              return node.ChargeHost(bytes, required);
            },
            [&node](std::uint64_t bytes) { node.UnchargeHost(bytes); }) {}
  PrefillLookaheadGroup(std::uint64_t allowance, const PlanAccount::ChargeFn& charge,
                        const PlanAccount::ReleaseFn& release)
      : PrefillLookaheadGroup(allowance, charge, release, std::make_index_sequence<Capacity>{}) {}

  bool Fund(std::size_t index) { return index < Capacity && slots_[index].Fund(); }

  template <typename BuildFn>
  std::array<bool, Capacity> BuildAll(BuildFn&& build) {
    std::array<bool, Capacity> built{};
    for (std::size_t i = 0; i < Capacity; ++i) built[i] = slots_[i].Build([&] { return build(i); });
    return built;
  }

  double seconds(std::size_t index) const { return index < Capacity ? slots_[index].seconds() : 0; }

  template <typename InstallFn>
  std::array<bool, Capacity> InstallAfterCompletion(InstallFn&& install) {
    std::array<bool, Capacity> installed{};
    for (std::size_t i = 0; i < Capacity; ++i)
      installed[i] =
          slots_[i].InstallAfterCompletion([&](auto planned, double seconds, const auto& transfer) {
            return install(i, std::move(planned), seconds, transfer);
          });
    return installed;
  }

  void Abandon() {
    for (auto& slot : slots_) slot.Abandon();
  }

 private:
  template <std::size_t... Index>
  PrefillLookaheadGroup(std::uint64_t allowance, const PlanAccount::ChargeFn& charge,
                        const PlanAccount::ReleaseFn& release, std::index_sequence<Index...>)
      : slots_{((void)Index, PrefillLookahead<Planned>(allowance, charge, release))...} {}

  std::array<PrefillLookahead<Planned>, Capacity> slots_;
};

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_PREFILL_LOOKAHEAD_H_
