// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_PAGED_NODE_INTERNAL_H_
#define LLMP_ENGINE_PAGED_NODE_INTERNAL_H_

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "base/bounded_queue.h"
#include "base/bytes.h"
#include "base/check.h"
#include "catalog/catalog.h"
#include "engine/live_state.h"
#include "engine/paged_node.h"
#include "engine/supplemental_run.h"
#include "engine/support.h"
#include "engine/unproven_work.h"
#include "kernels/common/execution_resources.h"
#include "memory/materialize.h"
#include "memory/reclaim.h"
#include "platform/crash_policy.h"
#include "providers/device_runtime.h"
#include "providers/direct_reader.h"
#include "providers/storage.h"
#include "scheduler/commands.h"
#include "scheduler/programs.h"
#include "scheduler/scheduler.h"

namespace llmp::engine {

namespace sc = llmp::scheduler;
using base::Bytes;
using catalog::ExtentId;
using catalog::MemoryClass;
using Done = sc::ProgramDone;
using sc::AcquireProgram;
using sc::AcquireReport;
using sc::CallProgram;
using sc::EvictProgram;
using sc::RequestProgram;
using sc::RunProgram;
using sc::SwapProgram;
using sc::SwapReport;
using support::Address;
using support::Error;
using support::Joined;

inline std::atomic<std::uint64_t> next_node_generation{0};

// A request's driver spins from this long before a step's expected end
// until this long after it (the device lanes' defaults, DeviceSettings).
inline constexpr auto kSpinAhead = std::chrono::microseconds(1000);
inline constexpr auto kSpinPast = std::chrono::microseconds(1000);
// A direct step that has outlasted every recent one: how long the driver
// sleeps between looks.
inline constexpr auto kLateSleep = std::chrono::microseconds(200);
// The longest a direct step's driver sleeps at once: nothing wakes it at
// the step's end, so a likely end predicted too late costs at most this
// (and a wake, RE-017).
inline constexpr auto kDirectNap = std::chrono::microseconds(1000);
// A direct step on a stream with no step history yet (its first): the
// driver spins this long, then naps from kFirstNap, doubling to kDirectNap,
// as the device lanes back off without history (DeviceService).
inline constexpr auto kFirstSpin = std::chrono::milliseconds(20);
inline constexpr auto kFirstNap = std::chrono::microseconds(50);
// A fence query or release refused this many times in a row (a pending
// result in between resets the count) is given up:
// its work's end unproven (the device lane's DeviceSettings::refusals).
inline constexpr std::uint32_t kFenceRefusals = 8;

// Each device-type lane's completion handoff; a lane holds at most twice
// that many fences unreleased (queued for its completion lane, and
// watched there) and one more waiting to be handed over.
inline constexpr std::size_t kLaneHandoff = 256;
// The provider's events made ahead: every fence the device and copy lanes
// can hold at once, and a few for the node's own fences, so neither lane
// ever makes one as it goes (RE-029: cuEventCreate blocks while another
// thread launches into a full stream).
inline constexpr std::size_t kEventsAhead = (2 * ((2 * kLaneHandoff) + 1)) + 16;

inline bool Fence(providers::DeviceExecution& execution, providers::StreamId stream,
                  std::chrono::milliseconds quiet) {
  const auto fence = execution.Record(stream);
  if (!fence) {
    return false;
  }
  const auto give_up = std::chrono::steady_clock::now() + quiet;
  while (std::chrono::steady_clock::now() < give_up) {
    const auto state = execution.Query(*fence);
    if (!state) {
      return false;  // unknown: never released or retried
    }
    if (*state == providers::FenceState::kComplete) {
      return execution.Release(*fence).has_value();
    }
    std::this_thread::yield();
  }
  return false;
}

template <typename Done, typename Round, typename Progress>
void DriveUntil(Done done, Round round, Progress progress, std::chrono::milliseconds quiet) {
  std::uint64_t seen = progress();
  auto give_up = std::chrono::steady_clock::now() + quiet;
  while (!done()) {
    const auto now = std::chrono::steady_clock::now();
    if (const std::uint64_t moved = progress(); moved != seen) {
      seen = moved;
      give_up = now + quiet;
    } else if (now > give_up) {
      return;
    }
    round();
  }
}

inline Status CheckReplacementClosure(const catalog::Catalog& catalog, catalog::DomainId domain,
                                      const catalog::Closure& closure) {
  base::Bytes actual;
  std::optional<catalog::ExtentId> previous;
  for (const auto& [extent, generation] : closure.extents) {
    const auto view = catalog.Describe(extent);
    if (!view || view->descriptor.domain != domain || view->content_generation != generation ||
        view->discarded || view->state == catalog::ExtentState::kQuarantined ||
        (previous && extent <= *previous))
      return Error("replacement closure is foreign, stale or noncanonical");
    auto total = actual.Plus(view->descriptor.size);
    if (!total) return Error("replacement physical occupancy overflows");
    actual = *total;
    previous = extent;
  }
  if (closure.extents.empty() ? !closure.bytes_by_domain.empty()
                              : closure.bytes_by_domain.size() != 1 ||
                                    closure.bytes_by_domain.begin()->first != domain ||
                                    closure.bytes_by_domain.begin()->second != actual)
    return Error("replacement closure occupancy disagrees with its actual extents");
  return {};
}

inline bool ReleaseMapped(providers::VmmProvider& memory, Mapped& mapped,
                          scheduler::BackingService* backing_lane) {
  if (!mapped.reservation.valid()) {
    return true;
  }
  const bool unmapped =
      mapped.backings.empty() ||
      memory.Unmap(mapped.reservation, Bytes(0), Bytes(mapped.backings.size() * kPagedExtent))
          .has_value();
  if (!unmapped) {
    return false;
  }
  bool released = true;
  for (const auto backing : mapped.backings) {
    if (backing_lane != nullptr && backing_lane->ReturnToReserve(backing)) {
      continue;
    }
    released = memory.Release(backing).has_value() && released;
  }
  if (!released || !memory.Free(mapped.reservation)) {
    return false;
  }
  mapped.backings.clear();
  mapped.reservation = {};
  return true;
}

// One wait of the driver's under its patience.
class PagedNode::Waiter {
 public:
  Waiter(PagedNode& node, std::string_view what)
      : node_(node), patience_(node.patience()), seen_(node.progress()) {
    const auto now = std::chrono::steady_clock::now();
    state_ = {.what = what, .began = now, .progressed = now, .cancelled = std::nullopt};
    next_ = now + kPatienceCheck;
    patience_.Begin();
  }
  Waiter(const Waiter&) = delete;
  Waiter& operator=(const Waiter&) = delete;
  Waiter(Waiter&&) = delete;
  Waiter& operator=(Waiter&&) = delete;
  // A wait that ends is the driver's progress.
  ~Waiter() {
    patience_.End();
    node_.waits_ended_.fetch_add(1, std::memory_order_relaxed);
  }

  // What to do now: kCancel at most once (the caller cancels its request),
  // kGiveUp, or kWait.
  WaitVerdict Poll(std::chrono::steady_clock::time_point now) {
    const std::uint64_t progress = node_.progress();
    if (progress != seen_) {
      seen_ = progress;
      state_.progressed = now;
    }
    if (now < next_) {
      return WaitVerdict::kWait;
    }
    next_ = now + kPatienceCheck;
    const WaitVerdict verdict = patience_.Check(state_, progress);
    if (verdict == WaitVerdict::kCancel) {
      if (state_.cancelled) {
        return WaitVerdict::kWait;
      }
      state_.cancelled = now;
      std::println(stderr, "{}: nothing moved for {:.0f} s: its request is cancelled (a hang)",
                   state_.what, std::chrono::duration<double>(now - state_.progressed).count());
    }
    return verdict;
  }
  bool cancelled() const { return state_.cancelled.has_value(); }

 private:
  PagedNode& node_;
  Patience& patience_;
  std::uint64_t seen_ = 0;
  WaitState state_;
  std::chrono::steady_clock::time_point next_;
};

struct RequestExtension::Data final : kernels::ResourceDependency {
  Data(HostPayment paid, PagedNode& owner, std::uint64_t issuer, std::uint32_t at,
       std::uint64_t request_id, catalog::LeaseId held)
      : payment(std::move(paid)),
        node(&owner),
        generation(issuer),
        stream(at),
        request(request_id),
        lease(held) {}
  kernels::ResourceKeep Publish() { return Adopt(); }
  void Destroy() override {
    auto paid = std::move(payment);
    delete this;  // consumer-safe metadata only; no driver or node callback
    paid.Reset();
  }
  // Forward and close use the existing serialized CallMetadata path.
  // Controls run sequentially, so one actual CallProgram/callback allocation
  // capacity suffices. Capture bounds are checked against the real closure types below.
  static constexpr auto kValidationCaptureBytes = 4 * sizeof(void*);
  static constexpr auto kRemovalCaptureBytes = 3 * sizeof(void*);
  static constexpr auto kReplacementCaptureBytes = 5 * sizeof(void*);
  static constexpr auto kControlBytes =
      sizeof(sc::CallProgram) + kReplacementCaptureBytes + 2 * std::uint64_t{64};
  HostPayment payment, installed_payment;
  std::vector<std::pair<catalog::ExtentId, std::uint64_t>> additions;
  catalog::Closure candidate;
  sc::Scheduler::LeaseReplacementStorage prepared;
  PagedNode* node;
  std::uint64_t generation;
  std::uint32_t stream;
  std::uint64_t request;
  catalog::LeaseId lease;
  enum class State : std::uint8_t { kPrepared, kActive, kClosed };
  State state = State::kPrepared;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_PAGED_NODE_INTERNAL_H_
