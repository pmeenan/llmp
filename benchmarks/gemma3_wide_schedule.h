// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Shared, fixed first-screen event geometry for the native and public-API probes.
#ifndef LLMP_BENCHMARKS_GEMMA3_WIDE_SCHEDULE_H_
#define LLMP_BENCHMARKS_GEMMA3_WIDE_SCHEDULE_H_
#include <array>
#include <cstdint>
#include <vector>
namespace llmp::benchmarks::gemma3_wide {
inline constexpr std::uint32_t kOwners = 12, kVocab = 262208;
// Mixed first group, all-short second group, all-long third group.
// C4 stays unequal while C8/C12 require a maximum outside the short group.
inline constexpr std::uint32_t SourceIndex(std::uint32_t slot) {
  return slot == 2 || slot == 3 || slot >= 8 ? 1U : 0U;
}
enum class Kind {
  kPrefill,
  kScalar,
  kWave,
  kPauseBegin,
  kPauseEnd,
  kCatchupBegin,
  kCatchupEnd,
  kRefillBegin,
  kRefillEnd
};
struct Event {
  Kind kind;
  std::uint32_t first, count, rows, phase;
  bool head;
  std::array<std::uint32_t, kOwners> before;
};
inline std::vector<Event> Schedule(const std::array<std::uint32_t, kOwners>& prefix) {
  std::vector<Event> out;
  std::array<std::uint32_t, kOwners> past{};
  const auto add = [&](Kind kind, std::uint32_t first, std::uint32_t count, std::uint32_t rows,
                       std::uint32_t phase, bool head = true) {
    out.push_back({kind, first, count, rows, phase, head, past});
    if (kind == Kind::kPrefill || kind == Kind::kScalar || kind == Kind::kWave)
      for (std::uint32_t s = first; s < first + count; ++s) past[s] += rows;
    if (kind == Kind::kRefillBegin) past[10] = past[11] = 0;
  };
  // Round-robin paired chunks, including two independent input identities.
  for (std::uint32_t at = 0; at < 768; at += 128)
    for (std::uint32_t first = 0; first < kOwners; first += 2)
      if (at < prefix[first]) add(Kind::kPrefill, first, 2, 128, 0, at + 128 == prefix[first]);
  for (std::uint32_t s = 0; s < kOwners; ++s)
    for (std::uint32_t i = 0; i < 3; ++i) add(Kind::kScalar, s, 1, 1, 1);
  // One schedule covers both whole cohorts, every partial carrier and unequal C4.
  std::uint32_t phase = 2;
  for (const auto count : {12U, 11U, 10U, 9U, 8U, 7U, 6U, 5U, 4U}) {
    add(Kind::kPauseBegin, 0, count, 0, phase);
    const auto waves = count == 12 || count == 8 || count == 4 ? 4U : 2U;
    for (std::uint32_t i = 0; i < waves; ++i) add(Kind::kWave, 0, count, 1, phase);
    add(Kind::kPauseEnd, 0, count, 0, phase++);
  }
  add(Kind::kCatchupBegin, 0, 4, 0, phase);
  for (std::uint32_t s = 4; s < kOwners; ++s)
    while (past[s] < prefix[s] + 3 + 24) add(Kind::kScalar, s, 1, 1, phase);
  add(Kind::kCatchupEnd, 0, 4, 0, phase++);
  for (std::uint32_t i = 0; i < 8; ++i) add(Kind::kWave, 0, kOwners, 1, phase);
  ++phase;
  // The returning pair is a new request. Every refill chunk must preserve all
  // ten peers; only the explicitly interleaved C4 waves write peers 0..3.
  add(Kind::kRefillBegin, 10, 2, 0, phase);
  for (std::uint32_t at = 0; at < prefix[10]; at += 128) {
    add(Kind::kPrefill, 10, 2, 128, phase, at + 128 == prefix[10]);
    if (at < 256) add(Kind::kWave, 0, 4, 1, phase);
  }
  for (std::uint32_t s = 10; s < kOwners; ++s)
    for (std::uint32_t i = 0; i < 3; ++i) add(Kind::kScalar, s, 1, 1, phase);
  add(Kind::kRefillEnd, 10, 2, 0, phase++);
  for (std::uint32_t s = 0; s < kOwners; ++s)
    for (std::uint32_t i = 0; i < 2; ++i) add(Kind::kScalar, s, 1, 1, phase);
  return out;
}
}  // namespace llmp::benchmarks::gemma3_wide
#endif
