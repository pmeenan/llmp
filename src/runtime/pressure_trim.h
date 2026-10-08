// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// When pressure on the host's memory from outside the runtime has it trim
// (D-055 as amended 2026-10-02; docs/runtime-serving.md, "Retention, the
// spill budget and pressure"): other processes share a Spark's memory
// (D-004), and what the runtime gives back goes through the node's one
// reclaim order (Server::Reclaim). Vendor-free and host-only, so the CPU
// tests drive it with readings of their own; Server::Maintain feeds it
// platform::ReadMemoryPressure() between units and while idle.
//
// - Trigger: MemAvailable under the low mark. A full memory stall over its
//   share of the last 10 seconds only raises the trigger to the target: a
//   stall alone (a compile or a copy stalling on page cache with plenty
//   available) is not pressure on the runtime's memory. Without a
//   MemAvailable reading nothing triggers.
// - Target: a trim asks for enough to bring MemAvailable to the low mark
//   plus the headroom, in one reclaim, not one candidate a look; whatever
//   part of it the order has.
// - Hysteresis: once triggered, pressure ends only when MemAvailable is
//   back at that target and nothing stalls; between the mark and the
//   target nothing more is trimmed.
// - Back-off: while the trigger holds, trims are spaced by a back-off that
//   doubles from its first step to its most after each one, so pressure a
//   trim did not relieve (another process takes what was given back)
//   cannot make the runtime drop and rebuild what it uses every look. A
//   trim that found too little left to give goes straight to the most,
//   and asks for one log line a log interval at most. The back-off resets
//   when pressure ends.

#ifndef LLMP_RUNTIME_PRESSURE_TRIM_H_
#define LLMP_RUNTIME_PRESSURE_TRIM_H_

#include <chrono>
#include <cstdint>
#include <string_view>

#include "platform/memory_pressure.h"

namespace llmp::runtime {

struct PressureTrimLimits {
  std::uint64_t low_bytes = std::uint64_t{512} << 20U;     // the trigger
  std::uint64_t headroom_bytes = std::uint64_t{1} << 30U;  // the target above it
  double full_stall_percent = 5.0;                         // of the last 10 s
  std::chrono::steady_clock::duration first_backoff = std::chrono::seconds(1);
  std::chrono::steady_clock::duration most_backoff = std::chrono::seconds(60);
  std::chrono::steady_clock::duration log_interval = std::chrono::seconds(60);
};

class PressureTrim {
 public:
  using Clock = std::chrono::steady_clock;

  explicit PressureTrim(PressureTrimLimits limits = {}) : limits_(limits) {}

  struct Trim {
    bool now = false;          // reclaim `needed` bytes now
    std::uint64_t needed = 0;  // what would restore the target
    std::string_view why;      // for the reclaim's log line
  };
  // One look at the readings.
  Trim Observe(const platform::MemoryPressure& reading, Clock::time_point now);
  // What the trim Observe asked for freed. True when the caller should log
  // that the pressure cannot be relieved (too little left to give), at most
  // once a log interval.
  bool Trimmed(std::uint64_t freed, std::uint64_t needed, Clock::time_point now);

  bool pressed() const { return pressed_; }
  Clock::duration backoff() const { return backoff_; }
  std::uint64_t trims() const { return trims_; }
  std::uint64_t short_trims() const { return short_trims_; }

 private:
  PressureTrimLimits limits_;
  bool pressed_ = false;
  Clock::duration backoff_{};
  Clock::time_point next_;
  Clock::time_point next_log_;
  bool logged_ = false;
  std::uint64_t trims_ = 0;
  std::uint64_t short_trims_ = 0;
};

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_PRESSURE_TRIM_H_
