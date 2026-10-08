// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/watchdog.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>

namespace llmp::runtime {
namespace {

// Thirty days in milliseconds bounds every allowance: far past any cap,
// and clear of overflow when added to a time point.
constexpr double kMostMs = 86'400'000.0 * 30;

double PerSecond(double amount, std::uint32_t rate) {
  return amount / static_cast<double>(std::max<std::uint32_t>(rate, 1));
}

}  // namespace

std::string_view PhaseName(Phase phase) {
  switch (phase) {
    case Phase::kIdle:
      return "idle";
    case Phase::kStarting:
      return "starting a request";
    case Phase::kSwap:
      return "swapping";
    case Phase::kPrefill:
      return "prefilling";
    case Phase::kDecode:
      return "decoding";
    case Phase::kFinishing:
      return "finishing a request";
    case Phase::kPaused:
      return "waiting for a client to read";
  }
  return "unknown";
}

double ExpectedSeconds(Phase phase, std::uint64_t size, const Floors& floors) {
  const auto amount = static_cast<double>(size);
  switch (phase) {
    case Phase::kSwap:
      return amount / kSwapFloorBytesPerSecond;
    case Phase::kPrefill:
      return PerSecond(amount, floors.prefill);
    case Phase::kDecode:
      return PerSecond(amount, floors.decode);
    case Phase::kIdle:
    case Phase::kStarting:
    case Phase::kFinishing:
    case Phase::kPaused:
      break;
  }
  return 0;
}

std::chrono::milliseconds Allowance(std::chrono::milliseconds stall, double expected) {
  const double ms =
      static_cast<double>(stall.count()) + (kWorkMargin * std::max(expected, 0.0) * 1e3);
  return std::chrono::milliseconds(std::llround(std::min(ms, kMostMs)));
}

std::chrono::milliseconds ScaledDeadline(std::chrono::milliseconds stall,
                                         std::chrono::milliseconds cap, const Floors& floors,
                                         std::uint64_t swap_bytes, std::uint32_t prompt_tokens,
                                         std::uint32_t max_tokens) {
  const double expected = ExpectedSeconds(Phase::kSwap, swap_bytes, floors) +
                          ExpectedSeconds(Phase::kPrefill, prompt_tokens, floors) +
                          ExpectedSeconds(Phase::kDecode, max_tokens, floors);
  return std::min(cap, Allowance(stall, expected));
}

Watchdog::Watchdog(std::chrono::milliseconds stall, WatchClock::time_point now) : stall_(stall) {
  health_.last_progress = now;
}

bool Watchdog::Beat(Phase next, double expected, WatchClock::time_point now) {
  const bool recovered = !health_.healthy;
  health_.healthy = true;
  health_.last_progress = now;
  health_.phase = next;
  due_ = now + Allowance(stall_, expected);
  return recovered;
}

bool Watchdog::Idle(WatchClock::time_point now) { return Beat(Phase::kIdle, 0, now); }

bool Watchdog::Check(WatchClock::time_point now) {
  if (health_.phase == Phase::kIdle || health_.phase == Phase::kPaused || !health_.healthy ||
      now < due_) {
    return false;
  }
  health_.healthy = false;
  health_.stalled_at = now;
  ++health_.stalls;
  return true;
}

std::optional<WatchClock::time_point> Watchdog::due() const {
  if (health_.phase == Phase::kIdle || health_.phase == Phase::kPaused || !health_.healthy) {
    return std::nullopt;
  }
  return due_;
}

}  // namespace llmp::runtime
