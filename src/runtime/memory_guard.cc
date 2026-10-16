// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/memory_guard.h"

#include <algorithm>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <string>

#include "artifact/artifact.h"
#include "artifact/json.h"
#include "base/json.h"
#include "runtime/calibration.h"
#include "runtime/model_settings.h"

namespace llmp::runtime {

std::uint64_t ActiveFootprint(std::uint64_t weights, std::uint64_t activations,
                              std::uint64_t pool) {
  constexpr std::uint64_t extent = std::uint64_t{2} << 20U;
  constexpr std::uint64_t most = std::numeric_limits<std::uint64_t>::max();
  const auto region = [](std::uint64_t bytes) {
    return bytes > most - extent
               ? most / extent * extent
               : (std::max(bytes, std::uint64_t{1}) + extent - 1) / extent * extent;
  };
  std::uint64_t total = weights;
  for (const std::uint64_t part : {region(activations), region(pool)})
    if (__builtin_add_overflow(total, part, &total)) return most;
  return total;
}

std::uint64_t GuardReserve(const MemoryGuard& guard) {
  return guard.host_inputs + guard.plans + guard.requests + kUncountedMargin;
}

std::expected<void, std::string> CheckMemoryGuard(const MemoryGuard& guard) {
  const auto gib = [](std::uint64_t bytes) { return static_cast<double>(bytes) / (1ULL << 30U); };
  // Subtracted term by term, so no sum can wrap.
  const std::uint64_t a = guard.available;
  const bool fits =
      guard.largest <= a && guard.host_inputs <= a - guard.largest &&
      guard.plans <= a - guard.largest - guard.host_inputs &&
      kUncountedMargin <= a - guard.largest - guard.host_inputs - guard.plans &&
      guard.requests <= a - guard.largest - guard.host_inputs - guard.plans - kUncountedMargin;
  if (a != 0 && !fits) {
    return std::unexpected(std::format(
        "the largest model's weights and workspace ({:.1f} GiB), a model's largest host-built "
        "chunk inputs "
        "({:.1f} GiB), a model step's plans ({:.1f} GiB), the request memory ({:.1f} GiB, "
        "[client] request_memory_bytes) and a {:.0f} GiB margin do not fit the {:.1f} GiB "
        "available beside this node's fixed memory ({:.1f} GiB)",
        gib(guard.largest), gib(guard.host_inputs), gib(guard.plans), gib(guard.requests),
        gib(kUncountedMargin), gib(guard.available), gib(guard.fixed)));
  }
  return {};
}

std::expected<std::uint64_t, std::string> StartupBootstrapBytes(std::uint64_t aliases,
                                                                std::uint64_t configuration_bytes) {
  namespace a = artifact;
  // Canonical parser: old+new storage for nodes(24), kids(4), stack(4),
  // duplicate-key scratch(16). An extra node covers a rejected last value.
  constexpr std::uint64_t values = a::json::kMaxSeparators + a::json::kMaxContainers + 2;
  constexpr std::uint64_t canonical = 3 * values * (24 + 4 + 4 + 16);
  // Both manifest/index trees, plus transient spec parse. Text doubling,
  // index expansions/roles/maps, directory names and one rebuilt shard header.
  // The entry term includes the complete rejected-input growth inventory;
  // strings copied into it are separately bounded by the document bytes.
  constexpr std::uint64_t documents = a::kMaxManifestBytes + a::kMaxIndexBytes + a::kMaxSpecBytes;
  constexpr std::uint64_t artifact_open = 3 * canonical + 8 * documents + 4096 * a::kMaxEntries +
                                          3 * (a::kMaxShardHeader + a::kFileAlignment) +
                                          1536 * a::kMaxDirectoryEntries;
  constexpr base::json::Limits json;
  // General JSON node/kid/pending (20/value), duplicate-key string_view
  // scratch (16/value), copied text, input and bounded metadata source views.
  constexpr std::uint64_t metadata =
      36 * json.max_values + 4 * a::kMaxMetadataTotal + json.max_string_bytes;
  // One fact pass may have both opened artifacts. Settings retained for all
  // aliases are paid independently of transient parser reuse (including names,
  // calibration fields and their canonical/digest scratch).
  constexpr std::uint64_t fixed = 2 * artifact_open + 2 * metadata;
  // Retained settings/calibration, their copies and formatting. Parser scratch
  // is already in metadata; variable configured strings are paid separately.
  constexpr std::uint64_t per_alias =
      16 * (kMaxCalibrationBytes + sizeof(ModelSettings) + sizeof(Calibration));
  constexpr std::uint64_t alignment = a::kChunkBytes;
  std::uint64_t bytes = 0, configured = 0;
  if (aliases == 0 || __builtin_mul_overflow(configuration_bytes, std::uint64_t{16}, &configured) ||
      __builtin_mul_overflow(aliases, per_alias, &bytes) ||
      __builtin_add_overflow(bytes, configured, &bytes) ||
      __builtin_add_overflow(bytes, fixed, &bytes) ||
      __builtin_add_overflow(bytes, alignment - 1, &bytes))
    return std::unexpected("startup parser loan overflows");
  return bytes / alignment * alignment;
}

std::uint64_t MeasuredStartupCeiling(std::uint64_t available) {
  constexpr std::uint64_t extent = std::uint64_t{2} << 20U;
  return available > kUncountedMargin ? (available - kUncountedMargin) / extent * extent : 0;
}

std::expected<StartupBudget, std::string> OrdinaryStartupBudget(std::uint64_t aliases,
                                                                std::uint64_t available) {
  auto parser = StartupBootstrapBytes(aliases);
  if (!parser) return std::unexpected(parser.error());
  const std::uint64_t cap = MeasuredStartupCeiling(available);
  if (cap == 0)
    return std::unexpected("startup needs known physical memory past the uncounted margin");
  return StartupBudget{.parser_loan = *parser, .cap = cap};
}

std::expected<void, std::string> CheckStartupHostAllowance(std::uint64_t bytes, std::uint64_t cap,
                                                           std::uint64_t available) {
  constexpr std::uint64_t extent = std::uint64_t{2} << 20U;
  if (bytes == 0 || bytes % extent != 0 || cap == 0 || bytes > cap)
    return std::unexpected("invalid startup host allowance or execution cap");
  if (cap > available || kUncountedMargin > available - cap)
    return std::unexpected("startup host allowance requires known physical headroom");
  return {};
}

std::expected<void, std::string> CheckDiagnosticBudgetCap(const MemoryGuard& guard,
                                                          std::uint64_t dynamic_budget,
                                                          std::uint64_t cap,
                                                          std::uint64_t extra_pinned) {
  if (guard.available == 0)
    return std::unexpected("diagnostic budget cap requires known physical availability");
  if (auto checked = CheckMemoryGuard(guard); !checked) return checked;
  if (cap > dynamic_budget)
    return std::unexpected("diagnostic budget cap exceeds the ordinary dynamic budget");
  // Subtract the required terms rather than adding an overflowing minimum.
  if (guard.fixed > cap || guard.largest > cap - guard.fixed ||
      extra_pinned > cap - guard.fixed - guard.largest)
    return std::unexpected("diagnostic budget cap is below the required startup footprint");
  return {};
}

}  // namespace llmp::runtime
