// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The chat route's progress watchdog, its health record and the scaled
// non-streaming deadline (D-097 as the owner amended it on 2026-09-29;
// docs/runtime-serving.md#progress-and-deadlines). Vendor-free, and every
// call takes the time, so the tests drive it on a clock of their own.
//
// - Progress. A request fails when its backend makes no progress, not
//   when a clock runs out: each unit of work the backend returns from (a
//   swap, a prefill chunk, a decode step, the request's own start and end)
//   is a beat, and the unit that follows is allowed the stall time plus
//   its expected time at the model's floors times kWorkMargin. A unit
//   expected to take long (a swap of many bytes, a wide prefill chunk)
//   says so before it starts, so it is never cut short by the stall time
//   alone.
// - Health. A stall marks the backend unhealthy until its next beat (the
//   unit it hung in returned); the record keeps the phase, the last
//   progress, the stalls counted and when the last began. Detection is not
//   a limit (D-102): what a stall does is the server's choice
//   ([client] stall_action), by default a report only.
// - Paused. A request waiting for its client to read (backpressure) is
//   not the backend stalling: nothing is watched while paused.
// - Deadlines. A stream has none, and by default nor does a non-streaming
//   request (D-102). With [client] deadline_cap_seconds set, a
//   non-streaming request's is its whole work figured the same way,
//   capped: min(cap, stall + kWorkMargin × (swap bytes /
//   kSwapFloorBytesPerSecond + prompt tokens / prefill floor + max_tokens /
//   decode floor)).

#ifndef JITLLM_RUNTIME_WATCHDOG_H_
#define JITLLM_RUNTIME_WATCHDOG_H_

#include <chrono>
#include <cstdint>
#include <optional>
#include <string_view>

#include "config/node_config.h"

namespace jitllm::runtime {

using WatchClock = std::chrono::steady_clock;

// What the backend is doing.
enum class Phase : std::uint8_t {
  kIdle,       // no request: nothing is watched
  kStarting,   // a request taken up: rendered, counted, admitted
  kSwap,       // its model made resident
  kPrefill,    // a prefill chunk
  kDecode,     // generation steps
  kFinishing,  // work after a response (a swap's leftover backing released)
  kPaused,     // waiting for a client to read its stream (backpressure): not a stall
};
std::string_view PhaseName(Phase phase);

// Expected times are figured at these floors and multiplied by the margin.
// The swap's floor is far below the Spark's measured page-in (about 13
// GB/s, docs/experiments/fast-swap/swap.md) so that slower SSDs fit too.
inline constexpr double kWorkMargin = 3.0;
inline constexpr double kSwapFloorBytesPerSecond = 1e9;

// A model's throughput floors, in tokens a second (config::ModelEntry).
struct Floors {
  std::uint32_t prefill = config::kDefaultPrefillFloor;
  std::uint32_t decode = config::kDefaultDecodeFloor;
};

// Seconds a unit is expected to take at the floors: a swap paging in
// `size` bytes, a prefill chunk of `size` rows, `size` decode tokens;
// other phases, 0.
double ExpectedSeconds(Phase phase, std::uint64_t size, const Floors& floors);

// The stall time plus the margin times `expected` seconds, saturating.
std::chrono::milliseconds Allowance(std::chrono::milliseconds stall, double expected);

// A non-streaming request's deadline from when it starts running (above).
std::chrono::milliseconds ScaledDeadline(std::chrono::milliseconds stall,
                                         std::chrono::milliseconds cap, const Floors& floors,
                                         std::uint64_t swap_bytes, std::uint32_t prompt_tokens,
                                         std::uint32_t max_tokens);

// The backend's health as the watchdog sees it.
struct Health {
  bool healthy = true;
  Phase phase = Phase::kIdle;
  WatchClock::time_point last_progress;  // the last beat, or when the watch began
  WatchClock::time_point stalled_at;     // when the last stall was noticed
  std::uint64_t stalls = 0;              // stalls noticed so far
};

// Not thread-safe: its owner serializes the calls (the chat route's
// server, under its lock).
class Watchdog {
 public:
  Watchdog(std::chrono::milliseconds stall, WatchClock::time_point now);

  std::chrono::milliseconds stall() const { return stall_; }
  const Health& health() const { return health_; }

  // Progress: the backend took up work or returned from a unit, and now
  // does `next`, expected to take `expected` seconds at the floors.
  // Returns true when this ends a stall (the backend recovered).
  bool Beat(Phase next, double expected, WatchClock::time_point now);
  // Progress too: the backend has nothing more to do.
  bool Idle(WatchClock::time_point now);
  // Whether the work under way has passed its allowance since the last
  // beat: true once a stall, which marks the backend unhealthy until its
  // next beat. Never while idle or paused.
  bool Check(WatchClock::time_point now);
  // When Check turns true unless a beat comes first; none while idle,
  // paused or already stalled.
  std::optional<WatchClock::time_point> due() const;

 private:
  std::chrono::milliseconds stall_;
  Health health_;
  WatchClock::time_point due_;
};

}  // namespace jitllm::runtime

#endif  // JITLLM_RUNTIME_WATCHDOG_H_
