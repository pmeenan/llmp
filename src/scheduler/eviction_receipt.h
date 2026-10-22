// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_SCHEDULER_EVICTION_RECEIPT_H_
#define LLMP_SCHEDULER_EVICTION_RECEIPT_H_

#include <chrono>
#include <cstdint>

#include "base/check.h"

namespace llmp::scheduler {
// One driver's operation may issue several eviction programs (room, swap,
// rollback). Each eviction retains this owner beyond its program's retirement.
// All methods run on the scheduler thread, including Seal and Read via Call.
// The driver seals only after every issuing program retired. No new eviction
// may join after sealing; completion is the last owned terminal event, never
// a later observer's time or another operation's release counters.
class EvictionReceipt {
 public:
  using Clock = std::chrono::steady_clock;
  struct Snapshot {
    std::uint64_t pending = 0;
    std::uint64_t handed_off = 0;
    std::uint64_t released_unused = 0;
    std::uint64_t refused = 0;
    std::uint64_t quarantined = 0;
    std::uint64_t occupancy = 0;
    Clock::time_point finished;
    Clock::time_point progressed;
    bool complete = false;
    bool had_work = false;
    // Keep the actual endpoint above even when all owned work completed
    // before activation became ready. Empty receipts have no release tail.
    Clock::duration TailAfter(Clock::time_point ready) const {
      return had_work && finished > ready ? finished - ready : Clock::duration::zero();
    }
    // The node's existing quiet-time policy, scoped to this owner's work.
    // Callers poll this snapshot; observing unrelated work never refreshes it.
    bool Stalled(Clock::time_point now, std::chrono::milliseconds quiet) const {
      return !complete && pending != 0 && now - progressed >= quiet;
    }
  };
  void Seal(std::uint64_t occupancy) {
    base::Check(!sealed_, "an eviction receipt was sealed twice");
    sealed_ = true;
    const auto now = Clock::now();
    // Start watching leftover releases at activation exit. Preserve the last
    // actual terminal event if all work already finished before sealing.
    snapshot_.progressed = now;
    snapshot_.complete = snapshot_.pending == 0;
    if (snapshot_.complete && snapshot_.finished == Clock::time_point{}) {
      // An activation with no evictions has no terminal event: its endpoint
      // is explicitly the seal, with the occupancy observed at that point.
      snapshot_.occupancy = occupancy;
      snapshot_.finished = now;
    }
  }
  Snapshot Read() const { return snapshot_; }

 private:
  friend class Scheduler;
  void Begin() {
    base::Check(!sealed_, "an eviction joined a sealed receipt");
    snapshot_.had_work = true;
    ++snapshot_.pending;
    snapshot_.progressed = Clock::now();
  }
  // An already-owned parked allocation moves into a page-in. Preserve its
  // owner until that load maps it or proves its unwind, even after sealing.
  void Transfer() {
    ++snapshot_.pending;
    snapshot_.progressed = Clock::now();
  }
  void Handoff() {
    ++snapshot_.handed_off;
    snapshot_.progressed = Clock::now();
  }
  void Released() {
    ++snapshot_.released_unused;
    snapshot_.progressed = Clock::now();
  }
  void End(bool evicted, bool quarantined, std::uint64_t occupancy) {
    base::Check(snapshot_.pending != 0, "an eviction receipt settled twice");
    --snapshot_.pending;
    snapshot_.refused += !evicted && !quarantined;
    snapshot_.quarantined += quarantined;
    snapshot_.occupancy = occupancy;
    snapshot_.finished = Clock::now();
    snapshot_.progressed = snapshot_.finished;
    snapshot_.complete = sealed_ && snapshot_.pending == 0;
  }
  Snapshot snapshot_;
  bool sealed_ = false;
};
}  // namespace llmp::scheduler
#endif  // LLMP_SCHEDULER_EVICTION_RECEIPT_H_
