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

bool SameZeroSource(const sc::PageSource* actual, const sc::PageSource& initial) {
  return actual != nullptr && initial.zero && !initial.write_back && actual->zero &&
         !actual->write_back && actual->landed && actual->piece_count == 0 && actual->backing &&
         sc::SamePlace(*actual, initial) && actual->read.fd == initial.read.fd &&
         actual->read.offset == initial.read.offset && actual->read.memory == initial.read.memory &&
         actual->read.length == initial.read.length;
}

}  // namespace

struct LiveState::Preparation {
  PagedNode* node = nullptr;
  sc::ProgramDone done, drain_done;
  sc::AcquireReport report;
  catalog::Closure closure;
  std::vector<std::pair<std::size_t, std::size_t>> fresh;
  std::vector<ExtentId> ids;
  std::uint64_t request = 0, drain_request = 0, allowance = 0;
  bool acquisition_counted = false, cancellation_requested = false;
};

LiveState::LiveState(std::string model) : model_(std::move(model)) {}

LiveState::~LiveState() {
  // A destructor is not a completion proof. The driver or node teardown must
  // drain first, while the complete state/source owner still exists.
  base::Check(!preparation_, "destroying live state with preparation in flight");
}

std::expected<bool, std::string> LiveState::Prepare(PagedNode& node,
                                                    std::span<const Range> ranges) {
  ++preparation_stats_.attempted;
  if (preparation_ || node.job_active_ || node.torn_down_ || !node.threaded() || quarantined_)
    return Error("state preparation requires an idle healthy driver and no pending ticket");
  // Validate/count without allocating. Duplicate ranges may overestimate the
  // descriptor allowance; this is optional headroom, never forced/reclaimed.
  std::uint64_t count = 0;
  for (const auto& range : ranges) {
    if (range.region >= regions_.size()) return Error("a prepared range names no region");
    const auto& r = regions_[range.region];
    if (range.offset > r.bytes || range.bytes > r.bytes - range.offset)
      return Error("a prepared range is outside its layout");
    if (!r.growing) return Error("preparation requires growing regions");
    if (range.bytes == 0) continue;
    const auto first = static_cast<std::size_t>(range.offset / kExtent);
    const auto last = static_cast<std::size_t>((range.offset + range.bytes - 1) / kExtent);
    for (auto i = first; i <= last; ++i) {
      if (r.used[i] || r.kept[i]) continue;
      // An adopted restart file is authoritative even for currently unpublished
      // ranges. Never turn it into a zero source or disposable prepared page.
      if (i >= r.sources.size() || !r.sources[i].zero || r.sources[i].write_back) {
        ++preparation_stats_.refused;
        return false;
      }
      if (count == UINT64_MAX) return Error("state preparation descriptor count overflow");
      ++count;
    }
  }
  if (count == 0) return false;
  // Covers the stable ticket/two Done records, fresh indices/IDs, both closure
  // copies and the AcquireProgram/materialization planner's transient vectors.
  // A drain borrows these IDs; its fixed program fits the 4096-byte envelope.
  // No-victim planning allocates no global victim list. Actual page
  // backing remains separately charged by the scheduler under B.
  constexpr std::uint64_t kDescriptorBytes = 256;
  constexpr std::uint64_t kTicketBytes = 4096;
  if (count > SIZE_MAX || count > (UINT64_MAX - kTicketBytes) / kDescriptorBytes)
    return Error("state preparation descriptor allowance overflow");
  const auto allowance = kTicketBytes + count * kDescriptorBytes;
  if (!node.TryChargeHost(allowance)) {
    ++preparation_stats_.refused;
    return false;
  }
  auto pending = std::make_unique<Preparation>();
  pending->node = &node;
  pending->allowance = allowance;
  pending->fresh.reserve(static_cast<std::size_t>(count));
  for (const auto& range : ranges) {
    if (range.bytes == 0) continue;
    const auto& r = regions_[range.region];
    const auto first = static_cast<std::size_t>(range.offset / kExtent);
    const auto last = static_cast<std::size_t>((range.offset + range.bytes - 1) / kExtent);
    for (auto i = first; i <= last; ++i)
      if (!r.used[i] && !r.kept[i]) pending->fresh.emplace_back(range.region, i);
  }
  std::ranges::sort(pending->fresh);
  const auto duplicate = std::ranges::unique(pending->fresh);
  pending->fresh.erase(duplicate.begin(), duplicate.end());
  bool eligible = true;
  auto checked = node.Call(
      [&]() -> Status {
        // Whole-source/identity validation precedes any optional page-in. Only
        // nonresident fresh state is selected; another owner or in-flight use
        // cannot become a preparation side effect.
        auto& ids = pending->ids;
        ids.reserve(pending->fresh.size());
        for (const auto& [ri, i] : pending->fresh) {
          const auto& r = regions_[ri];
          const auto id = r.mapped.extents[i];
          const auto view = node.catalog().Describe(id);
          const auto* source = node.scheduler().SourceOf(id);
          const auto& original = r.sources[i];
          if (!view || view->descriptor.domain != node.domain() ||
              view->descriptor.memory_class != catalog::MemoryClass::kLiveState ||
              view->descriptor.recovery != catalog::Recovery::kPreserve ||
              view->descriptor.size != Bytes(kExtent) || view->discarded || view->preserved ||
              view->state != catalog::ExtentState::kNonresident || view->leases != 0 ||
              view->registrations != 0 || !SameZeroSource(source, original)) {
            eligible = false;
            return {};
          }
          ids.push_back(id);
        }
        auto closure = node.catalog().ClosureOfExtents(ids);
        if (!closure) return Error("forming the state preparation closure");
        pending->closure = std::move(*closure);
        const auto plan = memory::PlanMaterialization(node.catalog(), node.domain(), node.budget(),
                                                      pending->closure, {}, false);
        eligible = plan.feasible && plan.shortfall.value() == 0;
        return {};
      },
      "checking optional state preparation");
  if (!checked || !eligible) {
    pending.reset();  // destroy all descriptors before returning their charge
    node.UnchargeHost(allowance);
    ++preparation_stats_.refused;
    return checked ? std::expected<bool, std::string>{false} : Error(checked.error());
  }
  preparation_ = std::move(pending);
  next_preparing_ = node.preparing_state_;
  node.preparing_state_ = this;
  preparation_->request = node.Submit(std::make_unique<sc::AcquireProgram>(
      preparation_->done, preparation_->closure, node.domain(), node.budget(), preparation_->report,
      std::vector<ExtentId>{}, false));
  ++preparation_stats_.submitted;
  return true;
}

LiveState::Status LiveState::FinishPreparation(bool cancel) {
  if (!preparation_) return {};
  auto& pending = *preparation_;
  auto& node = *pending.node;
  if (node.job_active_) return Error("state preparation settles only after the current Job");
  if (cancel && !pending.cancellation_requested) {
    if (!pending.done.gone.load()) node.Cancel(pending.request);
    pending.cancellation_requested = true;
    ++preparation_stats_.cancel_requested;
  }
  const auto acquired = node.Await(pending.done, "optional state preparation", pending.request);
  if (!pending.acquisition_counted) {
    if (!acquired) ++preparation_stats_.failed;
    pending.acquisition_counted = true;
  }
  // Acquire gone proves destruction of that program, not retirement of the
  // independent page-ins it withdrew from. A scoped waiter joins those loads
  // without starting absent pages. Keep the complete owner/charge/link until
  // the program is gone AND each scoped load has a positive no-access proof.
  bool retired = false;
  const auto check_retired = [&]() -> Status {
    return node.Call(
        [&]() -> Status {
          const auto result = node.scheduler().PageInsRetired(pending.ids);
          if (!result) return Error("prepared page-in retirement is unproven");
          retired = *result;
          return {};
        },
        "checking prepared page-in retirement");
  };
  auto proof = check_retired();
  if (proof && !retired) {
    if (pending.drain_request == 0 || pending.drain_done.gone.load()) {
      pending.drain_done.outcome.store(-1);
      pending.drain_done.retired.store(false);
      pending.drain_done.error.store(-1);
      pending.drain_done.gone.store(false);
      pending.drain_request =
          node.Submit(std::make_unique<sc::DrainPageInsProgram>(pending.drain_done, pending.ids));
    }
    const auto drained =
        node.Await(pending.drain_done, "prepared page-in drain", pending.drain_request);
    if (!drained) {
      Quarantine();
      return Error(drained.error());  // cancellation is not a retirement proof
    }
    proof = check_retired();
  }
  if (!proof || !retired) {
    Quarantine();
    return Error(proof ? "prepared page-ins have not retired" : proof.error());
  }
  for (const auto& [ri, i] : pending.fresh) regions_[ri].kept[i] = 2;
  auto collected = node.Call(
      [&]() -> Status {
        bool invalid = false;
        for (const auto& [ri, i] : pending.fresh) {
          const auto& r = regions_[ri];
          const auto id = r.mapped.extents[i];
          const auto expected = std::ranges::lower_bound(pending.closure.extents, id, {},
                                                         [](const auto& e) { return e.first; });
          const auto view = node.catalog().Describe(id);
          const auto* source = node.scheduler().SourceOf(id);
          if (expected == pending.closure.extents.end() || expected->first != id || !view ||
              view->content_generation != expected->second || view->discarded ||
              !SameZeroSource(source, r.sources[i])) {
            invalid = true;
            continue;  // changed authority: retain, never discard its contents here
          }
          if (view->state == catalog::ExtentState::kNonresident) continue;
          if (view->state != catalog::ExtentState::kResident ||
              (!view->discarded && !node.catalog().InvalidateContents(id))) {
            invalid = true;
            continue;
          }
          ++preparation_stats_.completed_extents;
        }
        return invalid ? Error("state preparation completion is uncertain; state quarantined")
                       : Status{};
      },
      "collecting unpublished prepared state");
  if (!collected) Quarantine();
  // Unlink only after all operations retired and partial backing is owned by
  // kept_extents(), including any uncertain extent that must survive teardown.
  auto** link = &node.preparing_state_;
  while (*link != this) link = &(*link)->next_preparing_;
  *link = next_preparing_;
  next_preparing_ = nullptr;
  const auto allowance = pending.allowance;
  preparation_.reset();
  node.UnchargeHost(allowance);
  return collected;
}

// ------------------------------------------------------------------ regions

LiveState::Status LiveState::Add(PagedNode& node, std::string name, std::uint64_t bytes,
                                 int owner) {
  if (auto finished = FinishPreparation(); !finished) return finished;
  InvalidatePlaces();
  Region& region = regions_.emplace_back();
  region.bytes = bytes;
  auto mapped =
      node.MapResident(region.mapped, std::move(name), bytes, providers::BackingKind::kDevice,
                       catalog::MemoryClass::kLiveState, catalog::Recovery::kPreserve, owner);
  region.used.assign(region.mapped.extents.size(), 1);
  region.kept.assign(region.mapped.extents.size(), 0);
  return mapped;
}

LiveState::Status LiveState::AddGrowing(PagedNode& node, std::string name, std::uint64_t bytes,
                                        int owner) {
  if (auto finished = FinishPreparation(); !finished) return finished;
  InvalidatePlaces();
  Region& region = regions_.emplace_back();
  region.bytes = bytes;
  auto reserved = node.ReserveState(region.mapped, std::move(name), bytes, owner);
  region.growing = true;
  region.used.assign(region.mapped.extents.size(), 0);
  region.kept.assign(region.mapped.extents.size(), 0);
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

void LiveState::SplitExtents(std::span<const Range> changed, std::vector<ExtentId>& written,
                             std::vector<ExtentId>& unchanged) const {
  for (std::size_t ri = 0; ri < regions_.size(); ++ri) {
    const Region& r = regions_[ri];
    for (std::size_t i = 0; i < r.mapped.extents.size(); ++i) {
      if (r.used[i] == 0) {
        continue;
      }
      const std::uint64_t begin = i * kExtent;
      const std::uint64_t end = begin + kExtent;
      const bool touched = std::ranges::any_of(changed, [&](const Range& c) {
        return c.region == ri && c.bytes != 0 && c.offset < end && begin < c.offset + c.bytes;
      });
      (touched ? written : unchanged).push_back(r.mapped.extents[i]);
    }
  }
}

std::vector<ExtentId> LiveState::kept_extents() const {
  std::vector<ExtentId> all;
  for (const Region& r : regions_) {
    for (std::size_t i = 0; i < r.mapped.extents.size(); ++i) {
      if (r.kept[i] != 0) {
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

std::expected<std::uint64_t, std::string> LiveState::UsedBytesOf(
    std::span<const Range> ranges) const {
  std::vector<std::pair<std::size_t, std::size_t>> extents;
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
      extents.emplace_back(range.region, i);
    }
  }
  std::ranges::sort(extents);
  const auto [begin, end] = std::ranges::unique(extents);
  extents.erase(begin, end);
  std::uint64_t total = 0;
  for (const auto& [ri, i] : extents) {
    const std::uint64_t offset = i * kExtent;
    total += std::min(kExtent, regions_[ri].bytes - offset);
  }
  return total;
}

std::expected<bool, std::string> LiveState::Use(PagedNode& node, std::span<const Range> ranges,
                                                const catalog::Closure* keep, bool* over_budget) {
  if (over_budget != nullptr) {
    *over_budget = false;
  }
  if (auto finished = FinishPreparation(); !finished) return Error(finished.error());
  if (quarantined_) return Error("state growth requires clearing quarantined state");
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
  InvalidatePlaces();
  zeroed_ = false;
  // Kept zeroed backing still resident is taken back as it is: nothing can
  // have written a discarded extent (no lease or registration takes one).
  // One reclaimed meanwhile grows from the zero source like any other (its
  // eviction cleared it and its source). One still being evicted would
  // leave the closure below stale and refuse this growth cleanly; the
  // driver's evictions all complete before it grows state again.
  if (std::ranges::any_of(fresh, [&](const auto& f) { return regions_[f.first].kept[f.second]; })) {
    auto took = node.Call(
        [&]() -> Status {
          for (const auto& [ri, i] : fresh) {
            Region& r = regions_[ri];
            if (r.kept[i] == 0) {
              continue;
            }
            const bool prepared = r.kept[i] == 2;
            r.kept[i] = 0;
            const ExtentId id = r.mapped.extents[i];
            const auto view = node.catalog().Describe(id);
            if (!view || view->state != catalog::ExtentState::kResident || !view->discarded ||
                !node.catalog().ReviveDiscarded(id)) {
              continue;
            }
            r.used[i] = 1;
            r.sources[i].write_back = true;
            if (auto set = node.scheduler().SetSource(id, r.sources[i]); !set) {
              return Error(std::format("a state's initialized write-back place: {}",
                                       sc::ToString(set.error())));
            }
            preparation_stats_.adopted_extents += prepared;
          }
          return {};
        },
        "taking back kept conversation state");
    if (!took) {
      Quarantine();
      return std::unexpected(took.error());
    }
    std::erase_if(fresh, [&](const auto& f) { return regions_[f.first].used[f.second] != 0; });
    if (fresh.empty()) {
      return true;
    }
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
    refused_bytes_ = refused_for_budget ? fresh.size() * kExtent : 0;
    return std::unexpected(loaded.error());
  }
  return true;
}

std::uint64_t LiveState::mapped_bytes(std::size_t region) const {
  return region < regions_.size() ? regions_[region].mapped.bytes : 0;
}

std::string_view LiveState::region_name(std::size_t region) const {
  return region < regions_.size() ? std::string_view(regions_[region].mapped.name)
                                  : std::string_view();
}

LiveState::Status LiveState::RegisterSpill(PagedNode& node,
                                           const std::filesystem::path& directory) {
  return RegisterSpill(node,
                       SpillPlace{.directory = directory, .dir = -1, .name = {}, .keep = false});
}

LiveState::Status LiveState::RegisterSpill(PagedNode& node, const SpillPlace& place) {
  // A source is registered once. Refusal must not settle preparation or
  // change backing that still belongs to the original spill file.
  if (spill_fd_ >= 0) return Error("state spill source is already registered");
  if (auto finished = FinishPreparation(); !finished) return finished;
  InvalidatePlaces();
  std::uint64_t file_bytes = 0;
  for (const Region& r : regions_) {
    if (r.mapped.bytes >
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) - file_bytes) {
      return Error("the state's spill file exceeds its offset range");
    }
    file_bytes += r.mapped.bytes;
  }
  if (place.dir >= 0) {
    // A named file kept across a restart (D-105): owner-only, never through
    // a link; its contents kept only when adopted, and then exactly whole.
    auto opened = platform::OpenPrivateFile(
        place.dir, place.name.c_str(),
        {.write = true, .create = true, .truncate = !place.keep, .direct = true});
    if (!opened) {
      return Error(std::format("the spill file {}: {}", place.name,
                               std::generic_category().message(opened.error())));
    }
    spill_fd_ = opened->fd;
    spill_identity_ = opened->identity;
    if (place.keep && opened->bytes != file_bytes) {
      return Error(std::format("the kept spill file {} is not the state's size", place.name));
    }
  } else {
    std::filesystem::create_directories(place.directory);
    const auto opened = platform::OpenUnnamedDirectFile(place.directory);
    if (!opened) {
      return Error(std::format("the spill file in {}: {}", place.directory.string(),
                               std::generic_category().message(opened.error())));
    }
    spill_fd_ = *opened;
  }
  if (::ftruncate(spill_fd_, static_cast<off_t>(file_bytes)) != 0) {
    return Error("sizing the sparse conversation spill file");
  }
  // Fresh extents are zeroed on the device only when the file is holes
  // beneath them: never for a kept file, whose adopted extents (D-105) grow
  // from what the process before wrote there, though nothing is written
  // back yet.
  const bool zero = node.zero_state() && !place.keep;
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
          .write_back = region.used[i] != 0,
          .zero = zero};
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
  if (auto finished = FinishPreparation(); !finished) return finished;
  InvalidatePlaces();
  zeroed_ = false;
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
  if (retirement != nullptr) *retirement = CopyRetirement::kProven;
  if (auto finished = FinishPreparation(); !finished) return finished;
  if (!to_host) {
    InvalidatePlaces();
    zeroed_ = false;
  }
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
  (void)CheckPlacesImpl(scheduler, check);
}

bool LiveState::CheckPlacesImpl(const sc::Scheduler& scheduler, PlaceCheck& check) const {
  const auto stamp = scheduler.placement_stamp();
  if (stamp.cacheable() && checked_places_ == stamp) return true;
  InvalidatePlaces();
  const auto before = check.moved;
  for (const Region& region : regions_) {
    if (region.sources.size() != region.mapped.extents.size()) {
      check.Missing(std::format("{}'s write-back places", region.mapped.name));
    }
    for (std::size_t i = 0; i < region.mapped.extents.size() && i < region.sources.size(); ++i) {
      check.Check(scheduler, region.mapped.extents[i], region.sources[i]);
    }
  }
  if (stamp.cacheable() && check.moved == before) checked_places_ = stamp;
  return false;
}

LiveState::Status LiveState::DiscardGrowingState(PagedNode& node, bool keep_zeroed) {
  if (auto finished = FinishPreparation(); !finished) return finished;
  InvalidatePlaces();
  if (regions_.empty() ||
      !std::ranges::all_of(regions_, [](const Region& r) { return r.growing; })) {
    return Error("discarding requires nonempty growing state");
  }
  // Asked for, and zeroed for reuse just before with nothing changed since
  // (ZeroForReuse): resident extents keep their backing, invalidated.
  // Otherwise backing kept by an earlier clear is released with the rest.
  const bool keep = keep_zeroed && zeroed_;
  zeroed_ = false;
  quarantined_ = true;
  const auto used = extents();
  std::vector<ExtentId> all = used;
  const auto kept = kept_extents();
  all.insert(all.end(), kept.begin(), kept.end());
  std::vector<ExtentId> release = keep ? used : all;
  std::vector<std::uint8_t> resident(used.size(), 0);
  auto discarded = node.Call(
      [&]() -> Status {
        for (const ExtentId id : all) {
          auto view = node.catalog().Describe(id);
          if (!view || view->leases != 0 || view->registrations != 0 ||
              (view->state != catalog::ExtentState::kResident &&
               view->state != catalog::ExtentState::kNonresident)) {
            return Error("a state being cleared is held or has an operation in flight");
          }
        }
        for (std::size_t k = 0; k < all.size(); ++k) {
          const auto view = node.catalog().Describe(all[k]);
          if (view->state == catalog::ExtentState::kResident) {
            if (k < used.size()) resident[k] = 1;
            // A refused preparation invalidation remains owned but unusable.
            // After the hold is released, Clear can discard that unused page.
            if (!view->discarded && !node.catalog().InvalidateContents(all[k])) {
              return Error("invalidating a conversation state");
            }
          }
        }
        return {};
      },
      "discarding conversation state");
  if (!discarded) {
    return discarded;
  }
  if (keep) {
    std::erase_if(release, [&](ExtentId id) {
      const auto at = std::ranges::find(used, id);
      return at != used.end() && resident[static_cast<std::size_t>(at - used.begin())] != 0;
    });
  }
  if (auto evicted = node.Evict(release); !evicted) {
    return evicted;
  }
  auto forgotten = node.Call(
      [&]() -> Status {
        for (std::size_t k = 0; k < used.size(); ++k) {
          if ((!keep || resident[k] == 0) && !node.catalog().ForgetPreserved(used[k])) {
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
    for (std::size_t i = 0; i < r.used.size(); ++i) {
      if (!keep) {
        r.kept[i] = 0;
      } else if (r.used[i] != 0) {
        const auto at = std::ranges::find(used, r.mapped.extents[i]);
        r.kept[i] = resident[static_cast<std::size_t>(at - used.begin())];
      }
      r.used[i] = 0;
    }
  }
  restore_count_ = 0;
  commit_keep_ = 0;
  save_count_ = 0;
  saved_.clear();
  verify_rows_ = 0;
  quarantined_ = false;
  return {};
}

std::expected<bool, std::string> LiveState::ZeroForReuse(PagedNode& node,
                                                         const catalog::Closure& fence,
                                                         std::uint32_t stream) {
  if (auto finished = FinishPreparation(); !finished) return Error(finished.error());
  InvalidatePlaces();
  zeroed_ = false;
  if (regions_.empty() ||
      !std::ranges::all_of(regions_, [](const Region& r) { return r.growing; })) {
    return Error("zeroing for reuse requires nonempty growing state");
  }
  if (quarantined_) {
    return false;
  }
  const auto used = extents();
  bool covered = true;
  auto checked = node.Call(
      [&]() -> Status {
        for (const ExtentId id : used) {
          const auto view = node.catalog().Describe(id);
          const auto held = std::ranges::lower_bound(fence.extents, id, {},
                                                     [](const auto& e) { return e.first; });
          if (!view || view->state != catalog::ExtentState::kResident || view->discarded ||
              held == fence.extents.end() || held->first != id ||
              held->second != view->content_generation) {
            covered = false;
            return {};
          }
        }
        return {};
      },
      "checking conversation state for reuse");
  if (!checked) {
    return std::unexpected(checked.error());
  }
  if (!covered) {
    return false;
  }
  std::vector<std::pair<std::uint64_t, std::uint64_t>> zeroed;
  for (const Range& range : used_ranges()) {
    zeroed.emplace_back(base(range.region) + range.offset, range.bytes);
  }
  if (!zeroed.empty()) {
    // Unusable until zeroed; a failure keeps the quarantine.
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
        "zeroing conversation state for reuse", stream);
    if (!cleared) {
      return std::unexpected(cleared.error());
    }
    quarantined_ = false;
  }
  zeroed_ = true;
  return true;
}

LiveState::Status LiveState::Clear(PagedNode& node, const catalog::Closure& fence,
                                   std::uint32_t stream, std::string_view what) {
  if (auto finished = FinishPreparation(); !finished) return finished;
  InvalidatePlaces();
  zeroed_ = false;
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
  if (auto finished = FinishPreparation(); !finished) return finished;
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

bool LiveState::LiftIfPreserved(PagedNode& node) {
  if (!quarantined_) {
    return true;
  }
  const std::vector<ExtentId> used = extents();
  bool preserved = true;
  auto checked = node.Call(
      [&]() -> Status {
        for (const ExtentId id : used) {
          const auto view = node.catalog().Describe(id);
          if (!view || view->state != catalog::ExtentState::kNonresident || !view->preserved) {
            preserved = false;
          }
        }
        return {};
      },
      "checking a spilled state untouched");
  if (!checked || !preserved) {
    return false;
  }
  quarantined_ = false;
  return true;
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

LiveState::Status LiveState::AllocateSnapshot(RunnerResources& resources, std::uint32_t capacity,
                                              bool reserve_saves) {
  if (reserve_saves) {
    saved_.reserve(capacity);
    if (saved_.capacity() > capacity) return Error("verify saved metadata exceeds funded capacity");
  }
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
  if (owed()) InvalidatePlaces();
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
  if (preparation_) {
    problems.emplace_back("state preparation was not drained; complete owner retained");
    return;
  }
  InvalidatePlaces();
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
    // An unnamed file goes with it; a named one stays for the next process
    // to adopt or empty (D-105).
    (void)::close(spill_fd_);
    spill_fd_ = -1;
  }
}

}  // namespace jitllm::engine
