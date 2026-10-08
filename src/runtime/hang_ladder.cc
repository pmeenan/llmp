// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/hang_ladder.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace llmp::runtime {
namespace {

double Seconds(HangLadder::Clock::duration d) { return std::chrono::duration<double>(d).count(); }

}  // namespace

std::string_view RungName(HangLadder::Rung rung) {
  switch (rung) {
    case HangLadder::Rung::kWatching:
      return "watching";
    case HangLadder::Rung::kCancel:
      return "rung 1 (cancel the stuck work)";
    case HangLadder::Rung::kRestart:
      return "rung 3 (restart the process)";
  }
  return "unknown";
}

HangLadder::HangLadder(std::chrono::milliseconds hang, Clock::time_point now)
    : hang_(hang),
      grace_(std::min<std::chrono::milliseconds>(hang, kCancelGrace)),
      unit_began_(now),
      progress_(now),
      cancel_at_(now) {}

void HangLadder::Say(std::string_view line) const {
  if (log_) {
    log_(line);
  }
}

void HangLadder::set_operations(Operations operations) {
  const std::scoped_lock lock(mutex_);
  operations_ = std::move(operations);
}

void HangLadder::Unit(bool working, std::chrono::milliseconds allowance, std::string_view what,
                      Clock::time_point now) {
  const std::scoped_lock lock(mutex_);
  working_ = working;
  what_ = what;
  unit_began_ = now;
  allowance_ = allowance;
  progress_ = std::max(progress_, now);
}

void HangLadder::Activity(std::uint64_t count, Clock::time_point now) {
  const std::scoped_lock lock(mutex_);
  if (!counted_ || count != count_) {
    counted_ = true;
    count_ = count;
    progress_ = std::max(progress_, now);
  }
}

void HangLadder::BeginWait() {
  const std::scoped_lock lock(mutex_);
  ++waits_;
}

void HangLadder::EndWait() {
  const std::scoped_lock lock(mutex_);
  waits_ = std::max(waits_ - 1, 0);
}

HangLadder::Clock::time_point HangLadder::cancelled_at() const {
  const std::scoped_lock lock(mutex_);
  return cancel_at_;
}

std::string HangLadder::EnterRestart(std::string why) {
  rung_.store(Rung::kRestart, std::memory_order_release);
  std::string line = std::format(
      "hang recovery, {}: {}; the runtime exits for its supervisor to restart it ([client] "
      "hang_seconds {:.0f} s)",
      RungName(Rung::kRestart), why, Seconds(hang_));
  Say(line);
  return line;
}

std::optional<HangLadder::Rung> HangLadder::Check(Clock::time_point now) {
  std::string restart;
  {
    const std::scoped_lock lock(mutex_);
    const Rung current = rung_.load(std::memory_order_acquire);
    if (current == Rung::kRestart) {
      return std::nullopt;
    }
    if (const std::uint64_t beats = pulse_.beats(); beats != beats_) {
      beats_ = beats;
      progress_ = std::max(progress_, now);
    }
    const bool under_way = working_ || waits_ > 0;
    if (current == Rung::kCancel) {
      // Drained: the driver moved again (its wait failed, its work stopped,
      // or nothing is under way) and nothing the cancelled work submitted
      // is still in flight: a read the drive holds keeps the cancellation
      // from draining, though the wait that needed it returned.
      const std::optional<Clock::time_point> oldest =
          operations_ ? operations_() : std::optional<Clock::time_point>{};
      const bool stuck_io = oldest.has_value() && *oldest <= cancel_at_;
      const bool moved = progress_ > cancel_at_ || !under_way;
      if (moved && !stuck_io) {
        rung_.store(Rung::kWatching, std::memory_order_release);
        pulse_.Clear();
        drained_.fetch_add(1, std::memory_order_relaxed);
        Say(std::format(
            "hang recovery, {}: the cancellation drained {:.1f} s after it (the {} stopped and "
            "nothing it submitted is still in flight); the requests that needed it fail and the "
            "model is recovered in place",
            RungName(Rung::kCancel), Seconds(now - cancel_at_),
            cancelled_wait_ ? "node wait" : "CPU work"));
        return Rung::kWatching;
      }
      if (stuck_io) {
        if (now - cancel_at_ < grace_) {
          return std::nullopt;
        }
        restart = EnterRestart(std::format(
            "the cancellation of the work under way ({}) did not drain: a storage operation it "
            "submitted {:.0f} s ago is still in flight {:.0f} s after the cancellation (a stuck "
            "drive or mount, which no cancellation ends)",
            what_, Seconds(now - *oldest), Seconds(now - cancel_at_)));
      } else {
        const Clock::time_point since = std::max(progress_, cancel_at_);
        if (now - since < grace_) {
          return std::nullopt;
        }
        restart = EnterRestart(std::format(
            "the cancellation of the work under way ({}) did not drain: nothing moved for {:.0f} "
            "s after it (the device or a lane is hung, or the work never reached a cancellation "
            "checkpoint, which nothing in the process can free)",
            what_, Seconds(now - since)));
      }
    } else {
      if (!under_way) {
        return std::nullopt;
      }
      // Outside any unit (a wait between units), no allowance applies.
      const bool past = !working_ || now - unit_began_ >= allowance_;
      const Clock::duration quiet = now - progress_;
      if (quiet < hang_ || !past) {
        return std::nullopt;
      }
      rung_.store(Rung::kCancel, std::memory_order_release);
      cancel_at_ = now;
      cancelled_wait_ = waits_ > 0;
      cancels_.fetch_add(1, std::memory_order_relaxed);
      if (!cancelled_wait_) {
        pulse_.Cancel();
      }
      Say(std::format(
          "hang recovery, {}: the model backend made no progress of any kind for {:.0f} s ({}: "
          "no unit ended, no fence completed, no read or write landed, no CPU work beat): a "
          "confirmed hang; {} ([client] hang_seconds)",
          RungName(Rung::kCancel), Seconds(quiet), what_,
          cancelled_wait_ ? "the request the node waits for is cancelled"
                          : "the driver's CPU work is asked to stop at its next checkpoint"));
      return Rung::kCancel;
    }
  }
  if (last_resort_) {
    last_resort_(restart);
  }
  return Rung::kRestart;
}

void HangLadder::Restart(const std::string& why) {
  std::string restart;
  {
    const std::scoped_lock lock(mutex_);
    if (rung_.load(std::memory_order_acquire) == Rung::kRestart) {
      return;
    }
    restart = EnterRestart(why);
  }
  if (last_resort_) {
    last_resort_(restart);
  }
}

std::optional<std::string> HangLadder::Resetting(std::string_view model) {
  const std::scoped_lock lock(mutex_);
  auto found = reset_unserved_.find(model);
  if (found != reset_unserved_.end() && found->second) {
    return std::format(
        "model {} hung again before it served a request since rung 2 reset it: resetting it again "
        "would loop",
        model);
  }
  reset_unserved_.insert_or_assign(std::string(model), true);
  return std::nullopt;
}

void HangLadder::Served(std::string_view model) {
  const std::scoped_lock lock(mutex_);
  if (auto found = reset_unserved_.find(model); found != reset_unserved_.end()) {
    found->second = false;
  }
}

}  // namespace llmp::runtime
