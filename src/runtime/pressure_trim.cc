// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/pressure_trim.h"

#include <algorithm>
#include <cstdint>

#include "platform/memory_pressure.h"

namespace llmp::runtime {

PressureTrim::Trim PressureTrim::Observe(const platform::MemoryPressure& reading,
                                         Clock::time_point now) {
  const std::uint64_t target = limits_.low_bytes + limits_.headroom_bytes;
  const bool low = reading.available && *reading.available < limits_.low_bytes;
  // A stall alone is not pressure on the runtime's memory (a compile or a
  // copy stalls on page cache with plenty available): it only brings the
  // trigger up to the target.
  const bool stalled = reading.stall && reading.stall->full_avg10 >= limits_.full_stall_percent &&
                       reading.available && *reading.available < target;
  if (!pressed_) {
    if (!low && !stalled) {
      return {};
    }
    pressed_ = true;
    backoff_ = limits_.first_backoff;
    next_ = now;
  } else if (!low && !stalled) {
    // Hysteresis: pressure ends only back at the target; between the mark
    // and the target nothing more is trimmed.
    if (!reading.available || *reading.available >= target) {
      pressed_ = false;
      backoff_ = {};
      logged_ = false;
    }
    return {};
  }
  if (now < next_) {
    return {};
  }
  Trim trim;
  trim.now = true;
  trim.needed = target - *reading.available;  // low or stalled: known and under the target
  trim.why = low ? "memory pressure from outside (MemAvailable under its low mark)"
                 : "memory pressure from outside (a full memory stall, MemAvailable under its "
                   "target)";
  return trim;
}

bool PressureTrim::Trimmed(std::uint64_t freed, std::uint64_t needed, Clock::time_point now) {
  ++trims_;
  if (freed < needed) {
    // Too little left to give: wait the longest before looking again.
    ++short_trims_;
    backoff_ = limits_.most_backoff;
    next_ = now + backoff_;
    if (!logged_ || now >= next_log_) {
      logged_ = true;
      next_log_ = now + limits_.log_interval;
      return true;
    }
    return false;
  }
  next_ = now + backoff_;
  backoff_ = std::min(backoff_ * 2, limits_.most_backoff);
  return false;
}

}  // namespace llmp::runtime
