// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What serving's translation units share (serving.cc, llm.cc,
// engine_llm.cc): error values, seconds, and the conversions a kept
// record's fields (D-105) take between the engine's types and the record's
// and between steady and wall-clock times. Private to src/runtime.

#ifndef LLMP_RUNTIME_SERVING_SUPPORT_H_
#define LLMP_RUNTIME_SERVING_SUPPORT_H_

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "engine/execution_contract.h"
#include "engine/live_state.h"
#include "engine/paged_node.h"
#include "platform/host_probe.h"
#include "platform/kept_files.h"
#include "runtime/kept_record.h"
#include "runtime/memory_guard.h"

namespace llmp::runtime::serving_support {

using Clock = std::chrono::steady_clock;

inline constexpr std::uint64_t kExtent = engine::kPagedExtent;

inline std::unexpected<std::string> Error(std::string what) {
  return std::unexpected(std::move(what));
}

// Startup only: a real pinned catalog owner, charged before construction.
// Neither local charge release nor a failed factory unpins this reservation.
inline std::expected<std::shared_ptr<engine::ExecutionMetadataAllowance>, std::string> AdmitStartup(
    engine::PagedNode& node, std::uint64_t bytes, std::uint64_t cap) {
  std::uint64_t total = 0;
  if (__builtin_add_overflow(bytes, engine::ExecutionMetadataAllowance::OverheadBytes(), &total) ||
      __builtin_add_overflow(total, kExtent - 1, &total))
    return Error("startup metadata payment overflows");
  total = total / kExtent * kExtent;
  const auto occupancy = node.catalog().OccupancyOf(node.domain()).Total().value();
  if (cap == 0 || occupancy > cap || total > cap - occupancy)
    return Error("startup metadata exceeds the node execution cap");
  auto owner = engine::ExecutionMetadataAllowance::Admit(
      node.catalog(), node.domain(), total, platform::AvailableMemoryBytes().value_or(0),
      kUncountedMargin);
  if (!owner) return Error(owner.error());
  if (!(*owner)->Charge(bytes)) return Error("startup metadata charge exceeds its payment");
  return owner;
}

inline std::expected<std::shared_ptr<engine::ExecutionMetadataAllowance>, std::string>
AdmitExecutionMetadata(engine::PagedNode& node, std::uint64_t bytes, std::uint64_t cap) {
  using Payment = std::shared_ptr<engine::ExecutionMetadataAllowance>;
  std::optional<std::expected<Payment, std::string>> completed;
  const auto admit = [&]() -> engine::Status {
    const auto budget = node.open_for_setup() ? cap : std::min(cap, node.budget().value());
    completed = engine::ExecutionMetadataAllowance::AdmitWithin(
        node.catalog(), node.domain(), bytes, budget, platform::AvailableMemoryBytes().value_or(0),
        kUncountedMargin);
    return *completed ? engine::Status{} : engine::Status{Error(completed->error())};
  };
  if (node.open_for_setup()) {
    (void)admit();
    return std::move(*completed);
  }
  const auto posted = node.Call(admit, "admitting retained execution metadata payment");
  if (completed) return std::move(*completed);
  return Error(posted ? "execution metadata admission did not complete" : posted.error());
}

// Growth is an explicit caller contract: absent node/cap keeps a diagnostic's
// fixed allowance. Catalog changes run under the node's serialized owner.
inline bool ChargeExecutionMetadata(
    const std::shared_ptr<engine::ExecutionMetadataAllowance>& allowance, std::uint64_t bytes,
    engine::PagedNode* node = nullptr, std::uint64_t cap = 0) {
  if (!allowance) return false;
  if (bytes <= allowance->capacity() - allowance->used()) return allowance->Charge(bytes);
  if (!node || cap == 0) return false;
  const auto grow = [&]() -> engine::Status {
    const auto budget = node->open_for_setup() ? cap : std::min(cap, node->budget().value());
    return allowance->ChargeOrGrow(node->catalog(), node->domain(), bytes, budget,
                                   platform::AvailableMemoryBytes().value_or(0), kUncountedMargin);
  };
  if (node->open_for_setup()) return grow().has_value();
  std::optional<engine::Status> completed;
  const auto posted = node->Call(
      [&]() -> engine::Status {
        completed = grow();
        return *completed;
      },
      "growing retained execution metadata payment");
  // Completed catalog admission remains authoritative after cancellation.
  return (completed ? *completed : posted).has_value();
}

inline double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

inline std::string Errno(int error) { return std::system_category().message(error); }

// A slot's regions' bytes in its spill file (whole extents), or their
// layouts' bytes.
inline std::vector<std::uint64_t> KeptRegionsOf(const engine::LiveState& live, bool mapped) {
  std::vector<std::uint64_t> out;
  out.reserve(live.regions());
  for (std::size_t i = 0; i < live.regions(); ++i) {
    out.push_back(mapped ? live.mapped_bytes(i) : live.bytes(i));
  }
  return out;
}

// Wall-clock milliseconds for a steady time point, and back: a record's
// times survive a restart, the steady clock's do not.
inline std::int64_t UnixMs(Clock::time_point at) {
  const auto unix = std::chrono::system_clock::now() - (Clock::now() - at);
  return std::chrono::duration_cast<std::chrono::milliseconds>(unix.time_since_epoch()).count();
}
inline Clock::time_point SteadyAt(std::int64_t unix_ms) {
  const auto unix = std::chrono::system_clock::time_point(std::chrono::milliseconds(unix_ms));
  return Clock::now() -
         std::chrono::duration_cast<Clock::duration>(std::chrono::system_clock::now() - unix);
}
inline std::int64_t NowUnixMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

inline kept::FileId KeptId(const platform::FileIdentity& id) {
  return {.device = id.device, .inode = id.inode, .generation = id.generation};
}
inline platform::FileIdentity FileIdentityOf(const kept::FileId& id) {
  return {.device = id.device, .inode = id.inode, .generation = id.generation};
}

inline std::vector<kept::Range> KeptRanges(std::span<const engine::LiveState::Range> ranges) {
  std::vector<kept::Range> out;
  out.reserve(ranges.size());
  for (const auto& r : ranges) {
    out.push_back(
        {.region = static_cast<std::uint32_t>(r.region), .offset = r.offset, .bytes = r.bytes});
  }
  return out;
}

inline std::vector<engine::LiveState::Range> LiveRanges(std::span<const kept::Range> ranges) {
  std::vector<engine::LiveState::Range> out;
  out.reserve(ranges.size());
  for (const auto& r : ranges) {
    out.push_back({.region = r.region, .offset = r.offset, .bytes = r.bytes});
  }
  return out;
}

}  // namespace llmp::runtime::serving_support

#endif  // LLMP_RUNTIME_SERVING_SUPPORT_H_
