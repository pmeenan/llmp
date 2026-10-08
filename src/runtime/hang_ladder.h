// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// D-102's recovery of a genuine hang (docs/runtime-serving.md#hang-recovery):
// how the runtime tells a hang from slow, healthy work, and the rungs it
// escalates through. Vendor-free, and every call takes the time, so the
// tests drive it on a clock of their own. Thread-safe: the chat route's I/O
// thread and the node's driver both call it.
//
// - Progress of any kind keeps it watching: a unit of the backend's work
//   beginning or ending (the watchdog's beats, Unit), the node's progress
//   count moving (Activity: a completion any lane published, a fence seen
//   complete, a read or write landed, a VMM operation;
//   engine/paged_node.h PagedNode::progress), or the driver's long CPU
//   work beating its pulse (pulse(), base/work_pulse.h: a rendering, a
//   tokenization, a graph's instantiation).
// - Work is under way while a unit runs (not while idle or paused for a
//   client) or while the driver waits on the node (BeginWait: a wait
//   outside any unit, such as housekeeping between units or a teardown).
// - A confirmed hang is work under way with no progress of any kind for
//   `hang` that has also passed its unit's allowance (the watchdog's: the
//   stall time plus three times the unit's expected time at the model's
//   floors). A swap from a slow disk keeps landing reads, and a wide
//   prefill chunk, one long job, is allowed its expected time: healthy
//   slow work is never a hang, however long it takes.
//
// The rungs, each logged with why:
// 1. Cancel. With the driver in a node wait, that wait cancels the request
//    it waits for (the node's patience, engine/paged_node.h); otherwise
//    the driver's CPU work is asked to stop at its next cancellation
//    checkpoint (its pulse). The cancellation has drained once the driver
//    moved again (its wait failed, its work stopped) and no storage
//    operation submitted before it is still in flight (set_operations: a
//    read the drive holds is not ended by a cancellation, and the wait
//    returns without it). Then the runtime fails only the requests that
//    needed the work and recovers the model in place (rung 2); the ladder
//    watches again.
// 2. Reset or evict the model in place: the runtime's, after rung 1 drained
//    (runtime/serve_api.cc). A model reset again before it served a
//    request since its last reset is not reset twice (Resetting): rung 3.
// 3. Restart: the cancellation did not drain within its grace (an
//    operation still in flight, or nothing moving: the device or a drive
//    hangs, which nothing in the process can free), or the runtime could
//    not recover the model in place (Restart). The last resort runs once,
//    on the thread that saw it: what can be kept is kept, and the process
//    exits for its supervisor to restart it.

#ifndef LLMP_RUNTIME_HANG_LADDER_H_
#define LLMP_RUNTIME_HANG_LADDER_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "base/work_pulse.h"

namespace llmp::runtime {

// A cancellation that can drain does so at once (a read still queued, a
// lease never granted, a command no lane took, CPU work at its next
// checkpoint); one that has not drained this long after it (or `hang`, if
// shorter) never will.
inline constexpr std::chrono::seconds kCancelGrace{60};

class HangLadder {
 public:
  using Clock = std::chrono::steady_clock;
  enum class Rung : std::uint8_t {
    kWatching,  // nothing confirmed
    kCancel,    // rung 1: the stuck work cancelled, its draining awaited
    kRestart,   // rung 3: the last resort has run (the process exits)
  };
  // Runs once, on the thread that entered rung 3, with why; the runtime's
  // exits the process. A test's may return: the ladder then stays at rung 3.
  using LastResort = std::function<void(const std::string& why)>;
  using Log = std::function<void(std::string_view line)>;
  // When the oldest storage operation still in flight was submitted; none
  // when none is (engine/paged_node.h PagedNode::oldest_io). Cheap: it is
  // asked at every Check, under the ladder's lock.
  using Operations = std::function<std::optional<Clock::time_point>()>;

  // `hang`: no progress of any kind this long is a confirmed hang (D-102's
  // [client] hang_seconds); the cancellation's grace is the shorter of it
  // and kCancelGrace.
  HangLadder(std::chrono::milliseconds hang, Clock::time_point now);
  HangLadder(const HangLadder&) = delete;
  HangLadder& operator=(const HangLadder&) = delete;
  HangLadder(HangLadder&&) = delete;
  HangLadder& operator=(HangLadder&&) = delete;
  ~HangLadder() = default;

  std::chrono::milliseconds hang() const { return hang_; }
  std::chrono::milliseconds cancel_grace() const { return grace_; }
  // Before the threads that call the rest start.
  void set_last_resort(LastResort last_resort) { last_resort_ = std::move(last_resort); }
  void set_log(Log log) { log_ = std::move(log); }
  // Any time (once the node runs): what the cancellation's draining is
  // judged on beside the driver's own progress. Unset: nothing in flight.
  void set_operations(Operations operations);
  // The driver's pulse (base::SetThreadPulse on the driver's thread): its
  // beats are progress, and rung 1 asks the CPU work under way to stop.
  base::WorkPulse& pulse() { return pulse_; }

  // A unit of the backend's work begins (the watchdog's beat): `working`
  // false while idle or paused for a client (nothing watched); `allowance`
  // the watchdog's for it; `what` names it in the log (static storage).
  // A beat is progress.
  void Unit(bool working, std::chrono::milliseconds allowance, std::string_view what,
            Clock::time_point now);
  // The node's progress count as read now: a change is progress.
  void Activity(std::uint64_t count, Clock::time_point now);
  // The driver's node waits begin and end (they may nest).
  void BeginWait();
  void EndWait();

  // Advances the ladder: the rung it entered now, if any, logged with why.
  // Entering rung 3 runs the last resort first (outside the ladder's lock).
  std::optional<Rung> Check(Clock::time_point now);
  // Rung 2 could not recover the model in place: rung 3, with why (the last
  // resort runs). Once; later calls do nothing.
  void Restart(const std::string& why);
  // Rung 2 is about to reset `model`: why it must not (it was reset before
  // and has served no request since: resetting it again would loop), or
  // none, recorded. Served: `model` completed a request.
  std::optional<std::string> Resetting(std::string_view model);
  void Served(std::string_view model);

  // Any thread.
  Rung rung() const { return rung_.load(std::memory_order_acquire); }
  // Rung 1s entered so far, and those whose cancellation drained.
  std::uint64_t cancels() const { return cancels_.load(std::memory_order_relaxed); }
  std::uint64_t drained() const { return drained_.load(std::memory_order_relaxed); }
  // When rung 1 was last entered: a wait that began before it is the one
  // to cancel (a later wait is new work).
  Clock::time_point cancelled_at() const;

 private:
  // With mutex_ held: rung 3 entered (the caller runs the last resort).
  std::string EnterRestart(std::string why);
  void Say(std::string_view line) const;

  // In the order that packs them.
  const std::chrono::milliseconds hang_;
  const std::chrono::milliseconds grace_;
  Clock::time_point unit_began_;
  std::chrono::milliseconds allowance_{0};
  std::uint64_t count_ = 0;
  std::uint64_t beats_ = 0;
  Clock::time_point progress_;
  Clock::time_point cancel_at_;
  std::atomic<std::uint64_t> cancels_{0};
  std::atomic<std::uint64_t> drained_{0};
  std::string_view what_ = "idle";
  base::WorkPulse pulse_;
  LastResort last_resort_;
  Log log_;
  Operations operations_;
  mutable std::mutex mutex_;
  // Per model: reset by rung 2 and not served since.
  std::map<std::string, bool, std::less<>> reset_unserved_;
  int waits_ = 0;
  bool working_ = false;
  bool counted_ = false;
  bool cancelled_wait_ = false;  // rung 1 cancelled a node wait (else CPU work)
  std::atomic<Rung> rung_{Rung::kWatching};
};

std::string_view RungName(HangLadder::Rung rung);

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_HANG_LADDER_H_
