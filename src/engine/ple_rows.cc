// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/ple_rows.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <limits>
#include <thread>
#include <utility>

#include "engine/support.h"
#include "providers/storage.h"

namespace llmp::engine {

namespace {

constexpr std::uint64_t kChunk = std::uint64_t{2} << 20U;

using support::Error;

std::uint64_t Down(std::uint64_t v, std::uint64_t to) { return v / to * to; }
std::uint64_t Up(std::uint64_t v, std::uint64_t to) { return (v + to - 1) / to * to; }

}  // namespace

std::expected<PleRowPlan, std::string> PlanPleRows(const PleTable& table,
                                                   std::span<const std::int32_t> rows,
                                                   std::uint64_t landing_capacity,
                                                   std::uint64_t max_slots,
                                                   std::uint64_t max_read) {
  if (table.row_bytes == 0 || table.rows == 0 || max_read < 2 * kPleBlock ||
      max_read % kPleBlock != 0 || table.file_offset < table.chunk_file_offset ||
      table.chunk_file_offset % kPleBlock != 0 || table.file_offset > table.file_bytes ||
      landing_capacity > std::numeric_limits<std::uint32_t>::max() ||
      table.rows > (table.file_bytes - table.file_offset) / table.row_bytes) {
    return Error("an n-gram table the row reads cannot address");
  }
  std::vector<std::int32_t> distinct(rows.begin(), rows.end());
  for (const std::int32_t r : distinct) {
    if (r < 0 || std::cmp_greater_equal(r, table.rows)) {
      return Error(std::format("n-gram row {} is outside the table's {} rows", r, table.rows));
    }
  }
  std::ranges::sort(distinct);
  const auto [end, last] = std::ranges::unique(distinct);
  distinct.erase(end, last);
  if (distinct.size() > max_slots) {
    return Error(std::format("{} distinct n-gram rows, more than the {} row slots", distinct.size(),
                             max_slots));
  }
  PleRowPlan plan;
  plan.slots.reserve(rows.size());
  for (const std::int32_t r : rows) {
    plan.slots.push_back(
        static_cast<std::int32_t>(std::ranges::lower_bound(distinct, r) - distinct.begin()));
  }
  // Reads over the rows in file order, blocks that touch or overlap merged.
  std::vector<std::size_t> read_of(distinct.size());
  std::int64_t last_chunk = -1;
  for (std::size_t i = 0; i < distinct.size(); ++i) {
    const std::uint64_t start =
        table.file_offset + (static_cast<std::uint64_t>(distinct[i]) * table.row_bytes);
    const std::uint64_t a = Down(start, kPleBlock);
    const std::uint64_t b = Up(start + table.row_bytes, kPleBlock);
    if (b > table.file_bytes) {
      return Error("an n-gram row's blocks pass the end of its group's stored range");
    }
    if (!plan.reads.empty()) {
      PleRead& back = plan.reads.back();
      const std::uint64_t back_end = back.file_offset + back.length;
      if (a <= back_end && std::max(b, back_end) - back.file_offset <= max_read) {
        back.length = std::max(b, back_end) - back.file_offset;
        read_of[i] = plan.reads.size() - 1;
      } else {
        plan.reads.push_back({.file_offset = a, .length = b - a, .landing = 0});
        read_of[i] = plan.reads.size() - 1;
      }
    } else {
      plan.reads.push_back({.file_offset = a, .length = b - a, .landing = 0});
      read_of[i] = 0;
    }
    // Whole-chunk accounting: the chunks holding the row's first and last
    // byte (rows ascend, so chunks do too).
    for (const std::uint64_t at : {start, start + table.row_bytes - 1}) {
      const auto chunk = static_cast<std::int64_t>((at - table.chunk_file_offset) / kChunk);
      if (chunk != last_chunk) {
        ++plan.extents;
        last_chunk = chunk;
      }
    }
  }
  for (PleRead& read : plan.reads) {
    read.landing = plan.landing_bytes;
    plan.landing_bytes += read.length;
  }
  if (plan.landing_bytes > landing_capacity) {
    return Error(std::format("the n-gram rows' reads need {} bytes, more than the landing's {}",
                             plan.landing_bytes, landing_capacity));
  }
  plan.sources.reserve(distinct.size());
  for (std::size_t i = 0; i < distinct.size(); ++i) {
    const PleRead& read = plan.reads[read_of[i]];
    const std::uint64_t start =
        table.file_offset + (static_cast<std::uint64_t>(distinct[i]) * table.row_bytes);
    plan.sources.push_back(static_cast<std::uint32_t>(read.landing + (start - read.file_offset)));
  }
  plan.useful_bytes = distinct.size() * table.row_bytes;
  return plan;
}

std::expected<void, std::string> ReadPleRows(providers::Storage& storage, int fd,
                                             const PleRowPlan& plan, std::byte* landing) {
  std::size_t next = 0;
  std::size_t in_flight = 0;
  std::string failure;
  std::array<providers::IoCompletion, 64> done{};
  // Nobody wakes this ring, so a harvest returns empty-handed only when the
  // ring could not wait; reads that make no progress for this long are
  // given up on, not spun on forever.
  constexpr auto kStall = std::chrono::seconds(30);
  auto progressed = std::chrono::steady_clock::now();
  while (next < plan.reads.size() || in_flight > 0) {
    while (failure.empty() && next < plan.reads.size() && in_flight < storage.depth()) {
      const PleRead& read = plan.reads[next];
      const auto submitted = storage.Submit({.token = next,
                                             .kind = providers::IoKind::kRead,
                                             .fd = fd,
                                             .offset = read.file_offset,
                                             .memory = landing + read.landing,
                                             .length = static_cast<std::uint32_t>(read.length),
                                             .segments = {}});
      if (submitted == providers::Submission::kNotStarted) {
        break;  // full: harvest first
      }
      // Accepted or unknown: in flight either way (storage.h), so its
      // completion is waited for; only a completion proves the row.
      ++next;
      ++in_flight;
    }
    if (in_flight == 0) {
      if (!failure.empty() || next >= plan.reads.size()) {
        break;
      }
      failure = "no n-gram row read could be submitted";
      break;
    }
    // Polled, not waited: the rows are on a decode step's critical path, and
    // a sleeping thread's wakeup on the Spark costs more than the reads
    // (RE-017; D-094's reason for polling the step path).
    const std::size_t got = storage.Harvest(done, false);
    if (got == 0) {
      std::this_thread::yield();
      if (std::chrono::steady_clock::now() - progressed > kStall) {
        // Reads still in flight may yet land: the caller must not reuse
        // the landing or destroy the ring (Qwen38Runner stops its rows).
        return Error(
            std::format("{} n-gram row reads made no progress in {} s", in_flight, kStall.count()));
      }
      continue;
    }
    progressed = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < got; ++i) {
      --in_flight;
      const std::uint64_t token = done.at(i).token;
      if (token >= plan.reads.size() ||
          std::cmp_not_equal(done.at(i).result, plan.reads[token].length)) {
        if (failure.empty()) {
          failure = std::format("an n-gram row read returned {}", done.at(i).result);
        }
      }
    }
    if (!failure.empty()) {
      next = plan.reads.size();  // drain what is in flight, submit nothing more
    }
  }
  if (!failure.empty()) {
    return Error(failure);
  }
  return {};
}

}  // namespace llmp::engine
