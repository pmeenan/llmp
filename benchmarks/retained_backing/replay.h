// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The deterministic replay of the retained-backing comparison
// (docs/backend-proof.md#retained-backing-comparison, D-079): drives one
// design through a swap trace on the address-only fake device-memory
// provider and computes the frozen deterministic metrics exactly.
//
// A tick is one lease, use or shrink. After each, held backing never
// exceeds the budget less any outstanding shrink, equals what the fake
// provider has created, and covers the resident groups; waste is held
// backing minus the resident groups' stored bytes. The replay recomputes
// the reference (swap_trace.py's LRU) and requires the trace's evict and
// restore records to be exactly its choices, so the design's extra victims
// follow the same recency order. The design's resident set stays within
// the reference's. A replay stops at its first refusal.

#ifndef LLMP_BENCHMARKS_RETAINED_BACKING_REPLAY_H_
#define LLMP_BENCHMARKS_RETAINED_BACKING_REPLAY_H_

#include <cstdint>
#include <string>

#include "retained_backing/designs.h"
#include "retained_backing/trace.h"

namespace llmp::rb {

struct Metrics {
  std::string design;
  std::string file;
  std::uint64_t budget = 0;
  std::uint64_t ticks = 0;
  // Waste after each tick: its sum (mean = waste_sum / ticks) and peak.
  std::uint64_t waste_sum = 0;
  std::uint64_t waste_peak = 0;
  std::uint64_t held_peak = 0;
  // The design's restores, and the reference's on the same accesses.
  std::uint64_t restores = 0;
  std::uint64_t restored_bytes = 0;
  std::uint64_t reference_restores = 0;
  std::uint64_t reference_restored_bytes = 0;
  // Useful content lost: restores beyond the reference's, and evictions
  // beyond its victims at shrink probes.
  std::uint64_t extra_restores = 0;
  std::uint64_t extra_restored_bytes = 0;
  std::uint64_t shrink_extra_evictions = 0;
  std::uint64_t shrink_extra_evicted_bytes = 0;
  // Extra evictions for accesses (their cost shows as extra restores).
  std::uint64_t access_extra_evictions = 0;
  std::uint64_t access_extra_evicted_bytes = 0;
  // A refusal stops the replay; refused_at is the event's index.
  std::uint64_t refusals = 0;
  std::int64_t refused_at = -1;
  // Accesses whose restore waited for a relocation, and what moved.
  std::uint64_t delayed_admissions = 0;
  std::uint64_t relocations = 0;
  std::uint64_t relocated_bytes = 0;
  CallCounts calls{};
  // SHA-256 over every provider call, placement, eviction and relocation.
  std::string digest;

  std::uint64_t content_lost() const { return extra_restored_bytes + shrink_extra_evicted_bytes; }
  bool operator==(const Metrics&) const = default;
};

struct ReplayOptions {
  // Check every structure's invariants every this many ticks (0: only at
  // shrink probes and at the end).
  std::uint64_t check_every = 0;
};

Metrics Replay(const Trace& trace, const DesignSpec& spec, const ReplayOptions& options = {});

// One JSON object on one line.
std::string ToJson(const Metrics& metrics);

}  // namespace llmp::rb

#endif  // LLMP_BENCHMARKS_RETAINED_BACKING_REPLAY_H_
