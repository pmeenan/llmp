// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/live_state.h"

#include <unistd.h>

#include <algorithm>
#include <format>
#include <limits>
#include <system_error>
#include <utility>

#include "engine/runner_resources.h"
#include "engine/support.h"
#include "platform/direct_io.h"
#include "providers/device_runtime.h"
#include "scheduler/commands.h"

namespace jitllm::engine {

namespace {

namespace kg = jitllm::kernels::ggml;
namespace sc = jitllm::scheduler;
using base::Bytes;
using catalog::ExtentId;
using support::Error;
using support::Pointer;
using support::Round;

constexpr std::uint64_t kExtent = kPagedExtent;

}  // namespace

// ------------------------------------------------------------------ regions

LiveState::Status LiveState::Add(PagedNode& node, std::string name, std::uint64_t bytes,
                                 int owner) {
  Region& region = regions_.emplace_back();
  region.bytes = bytes;
  auto mapped =
      node.MapResident(region.mapped, std::move(name), bytes, providers::BackingKind::kDevice,
                       catalog::MemoryClass::kLiveState, catalog::Recovery::kPreserve, owner);
  region.used.assign(region.mapped.extents.size(), 1);
  return mapped;
}

LiveState::Status LiveState::AddGrowing(PagedNode& node, std::string name, std::uint64_t bytes,
                                        int owner) {
  Region& region = regions_.emplace_back();
  region.bytes = bytes;
  auto reserved = node.ReserveState(region.mapped, std::move(name), bytes, owner);
  region.growing = true;
  region.used.assign(region.mapped.extents.size(), 0);
  return reserved;
}

std::uint64_t LiveState::base(std::size_t region) const {
  return region < regions_.size() ? regions_[region].mapped.base : 0;
}

std::uint64_t LiveState::bytes(std::size_t region) const {
  return region < regions_.size() ? regions_[region].bytes : 0;
}

std::uint64_t LiveState::total_bytes() const {
  std::uint64_t total = 0;
  for (const Region& r : regions_) {
    total += r.bytes;
  }
  return total;
}

std::vector<ExtentId> LiveState::extents() const {
  std::vector<ExtentId> all;
  for (const Region& r : regions_) {
    for (std::size_t i = 0; i < r.mapped.extents.size(); ++i) {
      if (r.used[i] != 0) {
        all.push_back(r.mapped.extents[i]);
      }
    }
  }
  return all;
}

std::vector<ExtentId> LiveState::reserved_extents() const {
  std::vector<ExtentId> all;
  for (const Region& r : regions_) {
    all.insert(all.end(), r.mapped.extents.begin(), r.mapped.extents.end());
  }
  return all;
}

std::vector<LiveState::Range> LiveState::used_ranges() const {
  std::vector<Range> ranges;
  for (std::size_t ri = 0; ri < regions_.size(); ++ri) {
    const Region& r = regions_[ri];
    for (std::size_t i = 0; i < r.used.size(); ++i) {
      const std::uint64_t offset = i * kExtent;
      if (r.used[i] != 0 && offset < r.bytes) {
        ranges.push_back(
            {.region = ri, .offset = offset, .bytes = std::min(kExtent, r.bytes - offset)});
      }
    }
  }
  return ranges;
}

std::uint64_t LiveState::used_bytes() const {
  std::uint64_t total = 0;
  for (const Range& r : used_ranges()) {
    total += r.bytes;
  }
  return total;
}

std::expected<bool, std::string> LiveState::Use(PagedNode& node, std::span<const Range> ranges,
                                                const catalog::Closure* keep, bool* over_budget) {
  if (over_budget != nullptr) {
    *over_budget = false;
  }
  // Validate the complete request before making any range resident.
  std::vector<std::pair<std::size_t, std::size_t>> fresh;
  for (const Range& range : ranges) {
    if (range.region >= regions_.size()) {
      return Error("a used state range names no region");
    }
    const Region& r = regions_[range.region];
    if (range.offset > r.bytes || range.bytes > r.bytes - range.offset) {
      return Error("a used state range is outside its layout");
    }
    if (range.bytes == 0) {
      continue;
    }
    const auto first = static_cast<std::size_t>(range.offset / kExtent);
    const auto last = static_cast<std::size_t>((range.offset + range.bytes - 1) / kExtent);
    for (std::size_t i = first; i <= last; ++i) {
      if (r.used[i] == 0) {
        fresh.emplace_back(range.region, i);
      }
    }
  }
  std::ranges::sort(fresh);
  const auto [begin, end] = std::ranges::unique(fresh);
  fresh.erase(begin, end);
  if (fresh.empty()) {
    return false;
  }
  std::vector<ExtentId> loading = extents();
  if (keep != nullptr) {
    for (const auto& [id, generation] : keep->extents) {
      (void)generation;
      loading.push_back(id);
    }
  }
  for (const auto& [ri, i] : fresh) {
    if (regions_[ri].sources.size() != regions_[ri].mapped.extents.size()) {
      return Error("a growing region has no registered zero source");
    }
    loading.push_back(regions_[ri].mapped.extents[i]);
  }
  catalog::Closure closure;
  auto described = node.Call(
      [&]() -> Status {
        auto of = node.catalog().ClosureOfExtents(loading);
        if (!of) {
          return Error("a growing state's closure");
        }
        closure = std::move(*of);
        return {};
      },
      "describing growing state");
  if (!described) {
    return std::unexpected(described.error());
  }
  sc::AcquireReport report;
  bool refused_for_budget = false;
  auto loaded =
      node.Acquire(closure, report, "initializing conversation state", &refused_for_budget);
  // A kPreserve extent with no write-back source cannot be reclaimed
  // here: its initial load is complete, and no job has used it yet.
  // Switch to preservation before exposing it to a model's closure.
  auto registered = node.Call(
      [&]() -> Status {
        for (const auto& [ri, i] : fresh) {
          Region& r = regions_[ri];
          const auto view = node.catalog().Describe(r.mapped.extents[i]);
          if (!view || view->state == catalog::ExtentState::kQuarantined) {
            Quarantine();
          }
          if (!view || view->state != catalog::ExtentState::kResident) {
            continue;  // a failed load may have initialized only some extents
          }
          r.used[i] = 1;  // retain completed pages for clearing and teardown
          r.sources[i].write_back = true;
          auto set = node.scheduler().SetSource(r.mapped.extents[i], r.sources[i]);
          if (!set) {
            return Error(std::format("a state's initialized write-back place: {}",
                                     sc::ToString(set.error())));
          }
        }
        return {};
      },
      "registering initialized conversation state");
  if (!registered) {
    Quarantine();
    return std::unexpected(registered.error());
  }
  if (!loaded) {
    // A clean capacity refusal precedes model dispatch. Existing state
    // remains valid; completed fresh zero pages are registered above.
    if (over_budget != nullptr) {
      *over_budget = refused_for_budget && !quarantined_;
    }
    return std::unexpected(loaded.error());
  }
  return true;
}

LiveState::Status LiveState::RegisterSpill(PagedNode& node,
                                           const std::filesystem::path& directory) {
  std::filesystem::create_directories(directory);
  const auto opened = platform::OpenUnnamedDirectFile(directory);
  if (!opened) {
    return Error(std::format("the spill file in {}: {}", directory.string(),
                             std::generic_category().message(opened.error())));
  }
  spill_fd_ = *opened;
  std::uint64_t file_bytes = 0;
  for (const Region& r : regions_) {
    if (r.mapped.bytes >
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) - file_bytes) {
      return Error("the state's spill file exceeds its offset range");
    }
    file_bytes += r.mapped.bytes;
  }
  if (::ftruncate(spill_fd_, static_cast<off_t>(file_bytes)) != 0) {
    return Error("sizing the sparse conversation spill file");
  }
  std::uint64_t slot = 0;
  for (Region& region : regions_) {
    Mapped& mapped = region.mapped;
    region.sources.clear();
    for (std::size_t i = 0; i < mapped.extents.size(); ++i, ++slot) {
      const sc::PageSource source{
          .read = {.fd = spill_fd_, .offset = slot * kExtent, .memory = nullptr, .length = kExtent},
          .landed = true,
          .destination = mapped.base + (i * kExtent),
          .backing = sc::BackingPlace{.reservation = mapped.reservation,
                                      .offset = Bytes(i * kExtent),
                                      .size = Bytes(kExtent),
                                      .allocation_class = node.device_class()},
          .write_back = region.used[i] != 0};
      auto set = node.scheduler().SetSource(mapped.extents[i], source);
      if (!set) {
        return Error(std::format("the state's write-back place: {}", sc::ToString(set.error())));
      }
      region.sources.push_back(source);
    }
    mapped.backings.clear();  // the VMM lane releases them on eviction (D-033)
  }
  return {};
}

LiveState::Status LiveState::Retain(PagedNode& node, std::span<const Range> ranges) {
  std::vector<std::vector<std::uint8_t>> wanted;
  wanted.reserve(regions_.size());
  for (const Region& r : regions_) {
    wanted.emplace_back(r.used.size(), 0);
  }
  for (const Range& range : ranges) {
    if (range.region >= regions_.size()) {
      return Error("a retained state range names no region");
    }
    const Region& r = regions_[range.region];
    if (range.offset > r.bytes || range.bytes > r.bytes - range.offset) {
      return Error("a retained state range is outside its layout");
    }
    if (range.bytes == 0) {
      continue;
    }
    const auto first = static_cast<std::size_t>(range.offset / kExtent);
    const auto last = static_cast<std::size_t>((range.offset + range.bytes - 1) / kExtent);
    for (std::size_t i = first; i <= last; ++i) {
      if (r.used[i] == 0) {
        return Error("a retained state range was not initialized");
      }
      wanted[range.region][i] = 1;
    }
  }
  std::vector<std::pair<std::size_t, std::size_t>> removed;
  std::vector<ExtentId> drop;
  for (std::size_t ri = 0; ri < regions_.size(); ++ri) {
    const Region& r = regions_[ri];
    for (std::size_t i = 0; i < r.used.size(); ++i) {
      if (r.used[i] != 0 && wanted[ri][i] == 0) {
        removed.emplace_back(ri, i);
        drop.push_back(r.mapped.extents[i]);
      }
    }
  }
  if (drop.empty()) {
    return {};
  }
  auto invalidated = node.Call(
      [&]() -> Status {
        for (const ExtentId id : drop) {
          const auto view = node.catalog().Describe(id);
          if (!view || view->leases != 0 || view->registrations != 0 ||
              (view->state != catalog::ExtentState::kResident &&
               view->state != catalog::ExtentState::kNonresident)) {
            return Error("a state being trimmed is held or has an operation in flight");
          }
        }
        for (const ExtentId id : drop) {
          if (node.catalog().Describe(id)->state == catalog::ExtentState::kResident &&
              !node.catalog().InvalidateContents(id)) {
            return Error("invalidating a state's discarded tail");
          }
        }
        return {};
      },
      "trimming conversation state");
  if (!invalidated) {
    return invalidated;
  }
  quarantined_ = true;
  if (auto evicted = node.Evict(drop); !evicted) {
    return evicted;
  }
  for (const auto& [ri, i] : removed) {
    if (!platform::DiscardFileRange(spill_fd_, regions_[ri].sources[i].read.offset, kExtent)) {
      return Error("discarding a state's saved tail");
    }
  }
  auto forgotten = node.Call(
      [&]() -> Status {
        for (const auto& [ri, i] : removed) {
          Region& r = regions_[ri];
          if (!node.catalog().ForgetPreserved(r.mapped.extents[i])) {
            return Error("forgetting a state's discarded tail");
          }
          r.sources[i].write_back = false;
          if (!node.scheduler().SetSource(r.mapped.extents[i], r.sources[i])) {
            return Error("restoring the initial source of a discarded state tail");
          }
          r.used[i] = 0;
        }
        return {};
      },
      "forgetting discarded conversation state");
  if (!forgotten) {
    return forgotten;
  }
  quarantined_ = false;
  return {};
}

LiveState::Status LiveState::Copy(PagedNode& node, const catalog::Closure& fence,
                                  std::uint32_t stream, void* host, std::span<const Range> ranges,
                                  bool to_host, CopyRetirement* retirement) {
  if (retirement != nullptr) {
    *retirement = CopyRetirement::kProven;
  }
  std::uint64_t total = 0;
  for (const Range& range : ranges) {
    if (range.region >= regions_.size() || range.offset > bytes(range.region) ||
        range.bytes > bytes(range.region) - range.offset ||
        range.bytes > std::numeric_limits<std::uint64_t>::max() - total) {
      return Error("a state snapshot range is outside its layout");
    }
    if (range.bytes != 0) {
      const Region& r = regions_[range.region];
      const auto first = static_cast<std::size_t>(range.offset / kExtent);
      const auto last = static_cast<std::size_t>((range.offset + range.bytes - 1) / kExtent);
      for (std::size_t i = first; i <= last; ++i) {
        if (r.used[i] == 0) {
          return Error("a state snapshot would touch an unused extent");
        }
      }
    }
    total += range.bytes;
  }
  if (total == 0) {
    return {};
  }
  if (host == nullptr) {
    return Error("a state snapshot has no host buffer");
  }
  auto posted = node.Job(
      fence,
      [&, host, to_host](providers::NativeStream native) {
        std::uint64_t at = 0;
        for (const Range& range : ranges) {
          if (range.bytes == 0) {
            continue;
          }
          void* device = Pointer(base(range.region) + range.offset);
          auto* pinned = static_cast<std::byte*>(host) + at;
          const auto copied = to_host ? providers::CopyAsync(native, pinned, device, range.bytes,
                                                             providers::CopyKind::kDeviceToHost)
                                      : providers::CopyAsync(native, device, pinned, range.bytes,
                                                             providers::CopyKind::kHostToDevice);
          if (!copied.ok()) {
            return sc::JobResult::kUnknown;
          }
          at += range.bytes;
        }
        return sc::JobResult::kQueued;
      },
      to_host ? "saving used conversation state" : "restoring used conversation state", stream);
  if (!posted) {
    node.KeepPinned(host);
    if (retirement != nullptr) {
      *retirement = CopyRetirement::kUnproven;
    }
  }
  return posted;
}

void LiveState::CheckPlaces(const sc::Scheduler& scheduler, PlaceCheck& check) const {
  for (const Region& region : regions_) {
    if (region.sources.size() != region.mapped.extents.size()) {
      check.Missing(std::format("{}'s write-back places", region.mapped.name));
    }
    for (std::size_t i = 0; i < region.mapped.extents.size() && i < region.sources.size(); ++i) {
      check.Check(scheduler, region.mapped.extents[i], region.sources[i]);
    }
  }
}

LiveState::Status LiveState::DiscardGrowingState(PagedNode& node) {
  if (regions_.empty() ||
      !std::ranges::all_of(regions_, [](const Region& r) { return r.growing; })) {
    return Error("discarding requires nonempty growing state");
  }
  quarantined_ = true;
  const auto used = extents();
  auto discarded = node.Call(
      [&]() -> Status {
        for (const ExtentId id : used) {
          auto view = node.catalog().Describe(id);
          if (!view || view->leases != 0 || view->registrations != 0 ||
              (view->state != catalog::ExtentState::kResident &&
               view->state != catalog::ExtentState::kNonresident)) {
            return Error("a state being cleared is held or has an operation in flight");
          }
        }
        for (const ExtentId id : used) {
          if (node.catalog().Describe(id)->state == catalog::ExtentState::kResident &&
              !node.catalog().InvalidateContents(id)) {
            return Error("invalidating a conversation state");
          }
        }
        return {};
      },
      "discarding conversation state");
  if (!discarded) {
    return discarded;
  }
  if (auto evicted = node.Evict(used); !evicted) {
    return evicted;
  }
  auto forgotten = node.Call(
      [&]() -> Status {
        for (const ExtentId id : used) {
          if (!node.catalog().ForgetPreserved(id)) {
            return Error("forgetting a conversation state's saved contents");
          }
        }
        for (Region& r : regions_) {
          for (std::size_t i = 0; i < r.sources.size(); ++i) {
            r.sources[i].write_back = false;
            if (!node.scheduler().SetSource(r.mapped.extents[i], r.sources[i])) {
              return Error("restoring a growing state's initial source");
            }
          }
        }
        return {};
      },
      "forgetting conversation state");
  if (!forgotten) {
    return forgotten;
  }
  std::uint64_t file_bytes = 0;
  for (const Region& r : regions_) {
    file_bytes += r.mapped.bytes;
  }
  if (::ftruncate(spill_fd_, 0) != 0 ||
      ::ftruncate(spill_fd_, static_cast<off_t>(file_bytes)) != 0) {
    return Error("clearing the sparse conversation spill file");
  }
  for (Region& r : regions_) {
    std::ranges::fill(r.used, 0);
  }
  restore_count_ = 0;
  commit_keep_ = 0;
  save_count_ = 0;
  saved_.clear();
  verify_rows_ = 0;
  quarantined_ = false;
  return {};
}

LiveState::Status LiveState::Clear(PagedNode& node, const catalog::Closure& fence,
                                   std::uint32_t stream, std::string_view what) {
  if (!regions_.empty() &&
      std::ranges::all_of(regions_, [](const Region& r) { return r.growing; })) {
    // Jobs have completed before this call. End the request's lease before
    // invalidating its state; do not reload an evicted conversation to clear it.
    if (node.InRequest(stream)) {
      if (auto ended = node.EndRequest(stream); !ended) {
        return ended;
      }
    }
    return DiscardGrowingState(node);
  }
  std::vector<std::pair<std::uint64_t, std::uint64_t>> zeroed;
  zeroed.reserve(regions_.size());
  for (const Range& range : used_ranges()) {
    zeroed.emplace_back(base(range.region) + range.offset, range.bytes);
  }
  restore_count_ = 0;
  commit_keep_ = 0;
  save_count_ = 0;
  saved_.clear();
  verify_rows_ = 0;
  // Unusable until zeroed; a quarantine lifts once the clear has run.
  quarantined_ = true;
  auto cleared = node.Job(
      fence,
      [zeroed](providers::NativeStream native) {
        for (const auto& [base, bytes] : zeroed) {
          if (!providers::FillAsync(native, Pointer(base), 0, bytes).ok()) {
            return sc::JobResult::kUnknown;
          }
        }
        return sc::JobResult::kQueued;
      },
      what, stream);
  if (!cleared) {
    return cleared;
  }
  quarantined_ = false;
  return {};
}

void* LiveState::HostCopy(PagedNode& node, std::uint64_t bytes) {
  if (host_copy_unproven_) {
    return nullptr;
  }
  bytes = bytes == 0 ? total_bytes() : bytes;
  bytes = std::max<std::uint64_t>(bytes, 256);
  if (host_copy_ != nullptr && host_capacity_ >= bytes) {
    return host_copy_;
  }
  if (host_copy_ != nullptr) {
    if (!node.FreePinned(host_copy_)) {
      return nullptr;
    }
    host_copy_ = nullptr;
    host_capacity_ = 0;
  }
  std::vector<ExtentId> staging;
  auto allocated = node.Pinned(bytes, kShared, staging);
  if (!allocated) {
    return nullptr;
  }
  host_node_ = &node;
  host_copy_ = *allocated;
  host_capacity_ = bytes;
  return host_copy_;
}

LiveState::Status LiveState::Read(PagedNode& node, const catalog::Closure& fence,
                                  std::uint32_t stream, std::string_view what,
                                  std::span<std::vector<std::byte>* const> out,
                                  CopyRetirement* retirement) {
  if (retirement != nullptr) {
    *retirement = CopyRetirement::kProven;
  }
  const std::vector<Range> ranges = used_ranges();
  auto* host = static_cast<std::byte*>(HostCopy(node, std::max<std::uint64_t>(used_bytes(), 256)));
  if (host == nullptr) {
    return Error("pinned host memory for the state's copy");
  }
  std::vector<std::pair<std::uint64_t, std::uint64_t>> reads;  // base, bytes
  reads.reserve(ranges.size());
  for (const Range& range : ranges) {
    reads.emplace_back(base(range.region) + range.offset, range.bytes);
  }
  auto posted = node.Job(
      fence,
      [&reads, host](providers::NativeStream native) {
        std::uint64_t at = 0;
        for (const auto& [base, bytes] : reads) {
          if (bytes != 0 && !providers::CopyAsync(native, host + at, Pointer(base), bytes,
                                                  providers::CopyKind::kDeviceToHost)
                                 .ok()) {
            return sc::JobResult::kUnknown;
          }
          at += bytes;
        }
        return sc::JobResult::kQueued;
      },
      what, stream);
  if (!posted) {
    HostCopyUnproven();
    if (retirement != nullptr) {
      *retirement = CopyRetirement::kUnproven;
    }
    return posted;
  }
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i]->assign(static_cast<std::size_t>(bytes(i)), std::byte{0});
  }
  std::uint64_t at = 0;
  for (const Range& range : ranges) {
    if (range.region < out.size()) {
      std::copy_n(host + at, static_cast<std::size_t>(range.bytes),
                  out[range.region]->data() + range.offset);
    }
    at += range.bytes;
  }
  return {};
}

LiveState::Status LiveState::Usable() const {
  if (quarantined_) {
    return Error(
        std::format("the {} state is quarantined: a job failed after it may have written it "
                    "(Clear first)",
                    model_));
  }
  return {};
}

// ------------------------------------------------------------------ speculation

LiveState::Status LiveState::AllocateSnapshot(RunnerResources& resources, std::uint32_t capacity) {
  auto save = resources.Pinned(capacity * sizeof(kg::RangeCopy));
  auto restore = resources.Pinned(capacity * sizeof(kg::RangeCopy));
  if (!save || !restore) {
    return Error(std::format("pinned staging for {}'s verify snapshot", model_));
  }
  capacity_ = capacity;
  save_ = static_cast<kg::RangeCopy*>(*save);
  restore_ = static_cast<kg::RangeCopy*>(*restore);
  return {};
}

void LiveState::BeginSaves() {
  saved_.clear();
  save_count_ = 0;
  save_at_ = 0;
}

LiveState::Status LiveState::Save(std::uint64_t address, std::uint64_t bytes, std::int64_t row) {
  if (save_count_ == capacity_ || save_at_ + bytes > snapshot_bytes_) {
    return Error("a verify writes more than its snapshot holds");
  }
  const std::uint64_t saved = snapshot_base_ + save_at_;
  saved_.push_back({.address = address, .saved = saved, .bytes = bytes, .row = row});
  save_[save_count_++] = {.from = address, .to = saved, .bytes = bytes};
  save_at_ += Round(bytes, 256);
  return {};
}

std::expected<void, kg::KernelFailure> LiveState::QueueSaves(kg::LaunchContext& launch) const {
  return kg::CopyRanges(launch, save_, save_count_);
}

LiveState::Status LiveState::AwaitingAccept() const {
  if (verify_rows_ != 0) {
    return Error("the last verify awaits its Accept");
  }
  return {};
}

LiveState::Status LiveState::Accept(std::uint32_t keep) {
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (verify_rows_ == 0 || keep == 0 || keep > verify_rows_) {
    return Error(std::format("accepting {} rows of a verify of {}", keep, verify_rows_));
  }
  std::uint32_t count = 0;
  for (const Saved& s : saved_) {
    if (s.row < 0 || std::cmp_greater_equal(s.row, keep)) {
      restore_[count++] = {.from = s.saved, .to = s.address, .bytes = s.bytes};
    }
  }
  restore_count_ = count;
  commit_keep_ = commit_ ? keep : 0;
  verify_rows_ = 0;
  return {};
}

std::expected<void, kg::KernelFailure> LiveState::QueueOwed(kg::LaunchContext& launch) {
  if (restore_count_ != 0) {
    // Still owed until the copy is queued.
    if (auto r = kg::CopyRanges(launch, restore_, restore_count_); !r) {
      return r;
    }
    restore_count_ = 0;
  }
  if (commit_keep_ != 0) {
    if (auto r = commit_(launch, commit_keep_); !r) {
      return r;
    }
    commit_keep_ = 0;
  }
  return {};
}

LiveState::Status LiveState::Rollback(PagedNode& node, const catalog::Closure& closure,
                                      std::uint32_t stream, kg::LaunchContext& launch,
                                      std::string_view what) {
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (!owed()) {
    return {};
  }
  std::string failed;
  auto posted = node.Job(
      closure,
      [&](providers::NativeStream) {
        if (auto r = QueueOwed(launch); !r) {
          failed = r.error().detail;
          return sc::JobResult::kUnknown;
        }
        return sc::JobResult::kQueued;
      },
      what, stream);
  if (!posted || !failed.empty()) {
    // What was not queued stays owed; what was has completed (the job
    // retired); a launch of unknown effect quarantines.
    Settle(false, false, !failed.empty() || launch.faulted());
    return Error(failed.empty() ? posted.error() : failed);
  }
  return {};
}

bool LiveState::Settle(bool saved, bool wrote, bool unknown) {
  verify_rows_ = 0;
  if (unknown) {
    quarantined_ = true;
    return false;
  }
  if (saved) {
    // The whole verify undone: every range it saved, before the next job's
    // own work (its queued work has completed: the job has retired), and
    // nothing committed.
    std::uint32_t count = 0;
    for (const Saved& s : saved_) {
      restore_[count++] = {.from = s.saved, .to = s.address, .bytes = s.bytes};
    }
    restore_count_ = count;
    commit_keep_ = 0;
    return true;
  }
  if (wrote) {
    quarantined_ = true;
  }
  return false;
}

void LiveState::Release(providers::VmmProvider& memory, std::vector<std::string>& problems) {
  for (Region& r : regions_) {
    if (!ReleaseMapped(memory, r.mapped)) {
      problems.push_back(std::format("{} could not be released", r.mapped.name));
    }
  }
  if (host_copy_ != nullptr) {
    if (host_copy_unproven_) {
      problems.emplace_back("the state's pinned copy retained until exit: completion unproven");
    }
    host_copy_ = nullptr;
    host_node_ = nullptr;
    host_capacity_ = 0;
  }
  if (spill_fd_ >= 0) {
    (void)::close(spill_fd_);  // unnamed: nothing outlives the process
    spill_fd_ = -1;
  }
}

}  // namespace jitllm::engine
