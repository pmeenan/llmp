// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/runner_memory.h"

#include <algorithm>
#include <bit>
#include <format>
#include <limits>
#include <memory>
#include <utility>

#include "providers/device_memory.h"

namespace llmp::engine {

RunnerMemory::Status RunnerMemory::Map(Mapped& mapped, std::string name, std::uint64_t bytes,
                                       catalog::MemoryClass memory_class, bool* over_budget) {
  if (over_budget) *over_budget = false;
  // Retain even a partially mapped attempt through fenced teardown.
  // MapResident can fail after reservation or some backing was acquired.
  mapped_.push_back(&mapped);
  if (auto r = node_.MapResident(mapped, std::move(name), bytes, providers::BackingKind::kDevice,
                                 memory_class, catalog::Recovery::kPinned, owner_, over_budget);
      !r) {
    return r;
  }
  return {};
}

std::expected<void*, std::string> RunnerMemory::Pinned(std::uint64_t bytes, bool* over_budget) {
  if (over_budget) *over_budget = false;
  const auto charged = std::max<std::uint64_t>(bytes, 256);
  if (charged > UINT64_MAX - pinned_bytes_)
    return std::unexpected("pinned staging accounting overflows");
  auto allocated = node_.Pinned(bytes, owner_, staging_, over_budget);
  if (allocated) {
    pinned_bytes_ += charged;
    pinned_.push_back({*allocated, staging_.back(), charged});
  }
  return allocated;
}

std::expected<AlignedPinned, std::string> RunnerMemory::PinnedAligned(std::uint64_t bytes,
                                                                      std::uint64_t alignment) {
  if (bytes == 0 || !std::has_single_bit(alignment) ||
      alignment > std::numeric_limits<std::size_t>::max() ||
      bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
    return std::unexpected("aligned pinned staging has an invalid size or alignment");
  }
  const auto allocation_bytes = std::max<std::uint64_t>(bytes + alignment - 1, 256);
  if (allocation_bytes > std::numeric_limits<std::uint64_t>::max() - pinned_bytes_) {
    return std::unexpected("aligned pinned staging accounting overflows");
  }
  auto allocation = Pinned(allocation_bytes);
  if (!allocation) return std::unexpected(allocation.error());
  void* view = *allocation;
  auto available = static_cast<std::size_t>(allocation_bytes);
  if (std::align(static_cast<std::size_t>(alignment), static_cast<std::size_t>(bytes), view,
                 available) == nullptr) {
    // The node still owns and charges the allocation through fenced teardown.
    return std::unexpected("aligned pinned staging does not fit its allocation");
  }
  return AlignedPinned{.allocation = *allocation,
                       .data = static_cast<std::byte*>(view),
                       .bytes = bytes,
                       .allocation_bytes = allocation_bytes};
}

std::uint64_t RunnerMemory::mapped_bytes() const {
  std::uint64_t bytes = 0;
  for (const Mapped* mapped : mapped_) {
    bytes += mapped->bytes;
  }
  return bytes;
}

std::expected<std::uint64_t, std::string> RunnerMemory::MetadataBytes(std::size_t mappings,
                                                                      std::size_t pinned) {
  std::uint64_t bytes = 0, n = 0;
  if (__builtin_mul_overflow(std::uint64_t{mappings}, std::uint64_t{sizeof(Mapped*)}, &bytes) ||
      __builtin_mul_overflow(std::uint64_t{pinned},
                             std::uint64_t{sizeof(PinnedAllocation) + sizeof(catalog::ExtentId)},
                             &n) ||
      __builtin_add_overflow(bytes, n, &bytes) ||
      __builtin_add_overflow(bytes, std::uint64_t{(mappings ? 1U : 0U) + (pinned ? 2U : 0U)} * 64,
                             &bytes) ||
      bytes > PTRDIFF_MAX)
    return std::unexpected("private memory descriptor storage overflows");
  return bytes;
}
void RunnerMemory::ReserveMetadata(std::size_t mappings, std::size_t pinned) {
  mapped_.reserve(mappings);
  pinned_.reserve(pinned);
  staging_.reserve(pinned);
}
std::size_t RunnerMemory::ExtentCount() const {
  std::size_t count = staging_.size();
  for (const auto* mapped : mapped_) count += mapped->extents.size();
  return count;
}
RunnerMemory::Status RunnerMemory::FillExtents(std::span<catalog::ExtentId> into) const {
  if (into.size() != ExtentCount())
    return std::unexpected("private memory extent count/fill differs");
  std::size_t at = 0;
  for (const auto* mapped : mapped_)
    for (const auto id : mapped->extents) into[at++] = id;
  for (const auto id : staging_) into[at++] = id;
  return {};
}

std::vector<catalog::ExtentId> RunnerMemory::extents() const {
  std::vector<catalog::ExtentId> all;
  for (const Mapped* mapped : mapped_) {
    all.insert(all.end(), mapped->extents.begin(), mapped->extents.end());
  }
  all.insert(all.end(), staging_.begin(), staging_.end());
  return all;
}

RunnerMemory::Status RunnerMemory::Retire() {
  while (!mapped_.empty()) {
    if (auto retired = node_.RetireMapped(*mapped_.back()); !retired) return retired;
    mapped_.pop_back();
  }
  while (!pinned_.empty()) {
    const auto& allocation = pinned_.back();
    if (auto retired = node_.FreePinned(allocation.pointer); !retired) return retired;
    std::erase(staging_, allocation.extent);
    pinned_bytes_ -= allocation.bytes;
    pinned_.pop_back();
  }
  return {};
}

void RunnerMemory::ReleaseMappings(std::vector<std::string>& problems) {
  for (Mapped* mapped : mapped_) {
    if (!ReleaseMapped(node_.memory(), *mapped)) {
      problems.push_back(std::format("{} could not be released", mapped->name));
    }
  }
  mapped_.clear();
}

}  // namespace llmp::engine
