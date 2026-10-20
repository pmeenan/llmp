// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/slot_lifecycle.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <format>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "base/check.h"
#include "engine/checkpoint_file.h"
#include "engine/paged_weights.h"
#include "engine/plan_account.h"
#include "engine/support.h"
#include "model/state_contract.h"
#include "providers/device_runtime.h"
#include "scheduler/programs.h"
#include "scheduler/scheduler.h"

namespace llmp::engine {

namespace sc = llmp::scheduler;

namespace {
using catalog::ExtentId;
using support::Error;
using Range = LiveState::Range;

// A stage's ranges at the slot's interface (`region` the stage) as its own
// live state's (region 0), in their order.
std::vector<Range> OfStage(std::span<const Range> ranges, std::size_t stage) {
  std::vector<Range> out;
  for (const Range& r : ranges) {
    if (r.region == stage) out.push_back({.region = 0, .offset = r.offset, .bytes = r.bytes});
  }
  return out;
}

// A stage's spill file now holds it as it is: an incremental spill may
// skip what nothing recorded since. Not while a verify in any of the slot's
// stages awaits its Accept or owes a restore or commit (`settled` false):
// what runs those later may be a job of its own (LiveState::Rollback) that
// records no write, and a stage's commit is the model's code, which may
// write another stage's state (a stage's saves are its own: LiveState::Save),
// so the next spill writes every stage whole. Every place that marks a file
// current goes through here.
void FileHolds(bool settled, SpillTrack& track) {
  if (settled) {
    track.Saved();
  } else {
    track.Lost();
  }
}

std::vector<Range> AsRanges(std::span<const model::StateRange> ranges) {
  std::vector<Range> out;
  out.reserve(ranges.size());
  for (const auto& r : ranges) out.push_back({.region = 0, .offset = r.offset, .bytes = r.bytes});
  return out;
}

}  // namespace

SlotLifecycle::SlotLifecycle(PagedNode& node, SlotLifecycleOptions options, int owner,
                             std::uint32_t stream)
    : node_(node),
      options_(std::move(options)),
      owner_(owner),
      stream_(stream),
      cohort_(options_.model) {}

SlotLifecycle::~SlotLifecycle() = default;

std::unexpected<std::string> SlotLifecycle::Refuse(std::string_view what) const {
  return Error(std::format("{} {}", options_.model, what));
}

// ------------------------------------------------------------ the runner's life

Status SlotLifecycle::Setup() {
  if (setup_ || released_) return Refuse("slot setup is not repeatable");
  if (options_.slots == 0 || options_.slots > kMaxRequestSlots || options_.stages.empty() ||
      options_.read_align == 0 || options_.read_align % 256 != 0) {
    return Refuse("slots need 1 to 16 slots, a target stage and a read alignment of 256s");
  }
  for (std::size_t s = 0; s < options_.stages.size(); ++s) {
    const StageState& stage = options_.stages[s];
    if (stage.layout.bytes == 0) return Refuse(std::format("stage {} holds no state", stage.name));
    if (stage.read_align % 256 != 0) {
      return Refuse(std::format("stage {}'s read alignment is not of 256s", stage.name));
    }
    auto progress = model::DraftBounds(stage.layout);
    if (!progress) return std::unexpected(progress.error());
    if (progress->has_value() && s == 0)
      return Refuse("target stage cannot declare draft progress");
    if (progress->has_value() && stage.layout.context != options_.stages.front().layout.context)
      return Refuse("repeatable drafter and target need the same context bound");
    // A borrowed tensor is a drafter's view of the target's, which must
    // hold it.
    for (const auto& borrowed : stage.layout.borrowed) {
      if (s == 0 || model::BorrowedItem(options_.stages.front().layout, borrowed) == nullptr) {
        return Refuse(std::format("stage {} borrows {}, which the target does not hold", stage.name,
                                  borrowed.contract.name));
      }
    }
  }
  std::vector<model::StateLayoutDerived> layouts;
  for (const StageState& stage : options_.stages) layouts.push_back(stage.layout);
  kept_layout_id_ = model::KeptLayoutId(options_.spec, layouts);
  cohort_.set_slots(options_.slots);
  slots_.resize(options_.slots);
  for (std::uint32_t i = 0; i < options_.slots; ++i) {
    Slot& slot = slots_[i];
    slot.index = i;
    slot.stages.resize(options_.stages.size());
    for (std::size_t s = 0; s < options_.stages.size(); ++s) {
      slot.stages[s].live = std::make_unique<LiveState>(options_.model);
      if (auto added = slot.stages[s].live->AddGrowing(
              node_, std::format("{} {} state slot {}", options_.model, options_.stages[s].name, i),
              options_.stages[s].layout.bytes, owner_);
          !added) {
        return added;
      }
      auto bounds = model::DraftBounds(options_.stages[s].layout);
      if (!bounds) return std::unexpected(bounds.error());
      if (bounds->has_value()) {
        Stage& stage = slot.stages[s];
        stage.progress = std::make_unique<DraftProgressLedger>(**bounds);
        for (const auto& item : options_.stages[s].layout.items)
          if (item.contract.kind == model::StateKind::kDrafterProgress)
            stage.control_offset = item.offset;
        // Separate, cataloged allocations: KeepPinned can retain either
        // exact allocation if publication or hydration retirement is lost.
        auto publish = node_.Pinned(model::kDraftProgressBytes, owner_, progress_staging_);
        if (!publish) return std::unexpected(publish.error());
        stage.control_publish = static_cast<model::DraftProgressWords*>(*publish);
        auto hydrate = node_.Pinned(model::kDraftProgressBytes, owner_, progress_staging_);
        if (!hydrate) return std::unexpected(hydrate.error());
        stage.control_hydrate = static_cast<model::DraftProgressWords*>(*hydrate);
        // Tail and prefix can share one job; their immutable words cannot.
        auto tail = node_.Pinned(model::kDraftProgressBytes, owner_, progress_staging_);
        if (!tail) return std::unexpected(tail.error());
        stage.control_tail = static_cast<model::DraftProgressWords*>(*tail);
      }
    }
  }
  setup_ = true;
  return {};
}

Status SlotLifecycle::SealExecutionIdentity(const std::optional<base::Sha256Digest>& identity) {
  if (!setup_ || registered_ || released_ || execution_sealed_)
    return Refuse("execution identity must be sealed once before registration");
  if (identity) {
    constexpr std::string_view prefix = "llmp-state-v2:";
    if (!kept_layout_id_.starts_with(prefix) || kept_layout_id_.size() != prefix.size() + 64)
      return Refuse("execution identity requires the complete state layout");
    static_assert(sizeof(base::Sha256) <= 128);
    base::Sha256 hash;
    hash.Update("llmp-state-execution-v1");
    hash.Update(kept_layout_id_);
    hash.Update(std::as_bytes(std::span(*identity)));
    const auto digest = hash.Finish();
    // Reuse the existing 64-character digest storage. The only transient
    // ownership is fixed stack SHA state, with no second string allocation.
    constexpr std::string_view hex = "0123456789abcdef";
    for (std::size_t i = 0; i < digest.size(); ++i) {
      kept_layout_id_[prefix.size() + 2 * i] = hex[digest[i] / 16U];
      kept_layout_id_[prefix.size() + 2 * i + 1] = hex[digest[i] % 16U];
    }
  }
  execution_sealed_ = true;
  return {};
}

Status SlotLifecycle::Register() {
  if (!setup_ || registered_ || released_) return Refuse("slot registration order");
  for (Slot& slot : slots_) {
    for (std::uint32_t s = 0; s < slot.stages.size(); ++s) {
      const LiveState::SpillPlace place =
          options_.spill_place
              ? options_.spill_place(slot.index, s)
              : LiveState::SpillPlace{
                    .directory = options_.spill_directory, .dir = -1, .name = {}, .keep = false};
      if (auto registered = slot.stages[s].live->RegisterSpill(node_, place); !registered) {
        return registered;
      }
    }
  }
  registered_ = true;
  return {};
}

std::vector<ExtentId> SlotLifecycle::reserved_extents() const {
  std::vector<ExtentId> all;
  for (const Slot& slot : slots_) {
    for (const Stage& stage : slot.stages) {
      const auto reserved = stage.live->reserved_extents();
      all.insert(all.end(), reserved.begin(), reserved.end());
    }
  }
  return all;
}

Status SlotLifecycle::Bind(std::vector<ExtentId> shared, std::vector<const PagedWeights*> weights) {
  if (!registered_ || bound_ || released_) return Refuse("slot bind order");
  shared_ = std::move(shared);
  shared_.insert(shared_.end(), progress_staging_.begin(), progress_staging_.end());
  weights_ = std::move(weights);
  bound_ = true;
  if (auto refreshed = RefreshClosures(); !refreshed) {
    bound_ = false;
    return refreshed;
  }
  return {};
}

Status SlotLifecycle::TryExtendShared(std::span<const ExtentId> additional, bool* over_budget) {
  if (over_budget) *over_budget = false;
  if (!bound_ || released_ || cohort_.faulted()) return Refuse("shared growth needs usable slots");
  if (!node_.InRequest(stream_) && std::popcount(cohort_.active()) > 1) {
    cohort_.Fault(AllStates());
    return Refuse("multiple active owners need a held request before shared growth");
  }
  // Pay before copying either the old or candidate closure inventory. Bounds
  // include geometrically grown temporary vectors, closure extent/generation
  // pairs, domain-map nodes and allocation headers. State reservations bound
  // initialized extents without allocating a temporary extent list.
  std::uint64_t shared = 0, states = 0, bytes = 0, state_bytes = 0;
  if (__builtin_add_overflow(std::uint64_t{shared_.size()}, std::uint64_t{additional.size()},
                             &shared))
    return Refuse("shared growth inventory overflows");
  for (const Slot& slot : slots_)
    for (const Stage& stage : slot.stages)
      for (std::size_t region = 0; region < stage.live->regions(); ++region)
        if (__builtin_add_overflow(states, stage.live->mapped_bytes(region) / kPagedExtent,
                                   &states))
          return Refuse("shared growth state inventory overflows");
  // Eight shared and sixteen state extent/generation units cover simultaneous
  // candidate/output inventories plus the temporary lists in Cohort::Build.
  constexpr auto unit = sizeof(decltype(catalog::Closure::extents)::value_type);
  constexpr std::uint64_t metadata = (2 * kMaxRequestSlots + 24) * 256;
  if (__builtin_mul_overflow(shared, std::uint64_t{8 * unit}, &bytes) ||
      __builtin_mul_overflow(states, std::uint64_t{16 * unit}, &state_bytes) ||
      __builtin_add_overflow(bytes, state_bytes, &bytes) ||
      __builtin_add_overflow(bytes, metadata, &bytes))
    return Refuse("shared growth metadata overflows");
  auto paid = node_.PrepayHost(bytes);
  if (!paid) {
    if (over_budget) *over_budget = true;
    return Refuse(paid.error());
  }
  auto payment = std::make_shared<HostPayment>(std::move(*paid));
  std::vector<ExtentId> next;
  next.reserve(static_cast<std::size_t>(shared));
  next.insert(next.end(), shared_.begin(), shared_.end());
  next.insert(next.end(), additional.begin(), additional.end());
  std::ranges::sort(next);
  next.erase(std::ranges::unique(next).begin(), next.end());
  RequestCohort::Closures candidate;
  auto built = node_.Call(
      [&]() -> Status {
        std::array<std::vector<ExtentId>, kMaxRequestSlots> extents;
        std::array<const std::vector<ExtentId>*, kMaxRequestSlots> of{};
        for (const Slot& slot : slots_) {
          if (slot.spilled) continue;
          for (const Stage& stage : slot.stages) {
            const auto live = stage.live->extents();
            extents[slot.index].insert(extents[slot.index].end(), live.begin(), live.end());
          }
          of[slot.index] = &extents[slot.index];
        }
        auto made = cohort_.Build(node_.catalog(), next, of, cohort_.active());
        if (!made) return Refuse(made.error());
        candidate = std::move(*made);
        return {};
      },
      "preparing grown workspace closures");
  if (!built) return built;
  if (node_.InRequest(stream_)) {
    bool capacity = false;
    if (auto admitted = node_.TryRefreshRequest(stream_, candidate.execution, &capacity);
        !admitted) {
      if (over_budget) *over_budget = capacity;
      if (!capacity) cohort_.Fault(AllStates());
      return admitted;
    }
  } else {
    // A single active owner may execute without a persistent request lease.
    // Its new workspace still needs actual physical admission before any
    // closure or high-water publication. Acquire protects the stable shared
    // workspace and refuses while earlier device work is unretired.
    scheduler::AcquireReport report;
    bool capacity = false;
    if (auto admitted = node_.Acquire(candidate.execution, report,
                                      "admitting a single-owner workspace", &capacity);
        !admitted) {
      if (over_budget) *over_budget = capacity;
      if (!capacity) cohort_.Fault(AllStates());
      return admitted;
    }
  }
  // No failure follows replacement. Old vectors die before their old payment.
  {
    auto old_shared = std::move(shared_);
    auto old_everything = std::move(everything_);
    auto old_fence = std::move(fence_);
    auto old_execution = std::move(execution_);
    shared_ = std::move(next);
    everything_ = std::move(candidate.everything);
    fence_ = std::move(candidate.fence);
    execution_ = std::move(candidate.execution);
    for (Slot& slot : slots_) slot.fence = std::move(candidate.slot_fences[slot.index]);
  }
  shared_payment_ = std::move(payment);
  places_clean_.reset();
  return {};
}

void SlotLifecycle::Release(providers::VmmProvider& memory, std::vector<std::string>& problems) {
  if (released_) return;
  released_ = true;
  for (Slot& slot : slots_) {
    for (Stage& stage : slot.stages) {
      if (stage.live) stage.live->Release(memory, problems);
    }
  }
}

// ---------------------------------------------------------------------- views

std::expected<SlotLifecycle::Slot*, std::string> SlotLifecycle::Find(std::uint32_t slot) {
  if (released_ || slot >= slots_.size()) return Refuse(std::format("slot {} unavailable", slot));
  return &slots_[slot];
}

std::expected<const SlotLifecycle::Slot*, std::string> SlotLifecycle::Find(
    std::uint32_t slot) const {
  if (released_ || slot >= slots_.size()) return Refuse(std::format("slot {} unavailable", slot));
  return &slots_[slot];
}

const catalog::Closure& SlotLifecycle::slot_fence(std::uint32_t slot) const {
  return slots_.at(slot).fence;
}

LiveState& SlotLifecycle::live(std::uint32_t slot, std::size_t stage) {
  return *slots_.at(slot).stages.at(stage).live;
}
const LiveState& SlotLifecycle::live(std::uint32_t slot, std::size_t stage) const {
  return *slots_.at(slot).stages.at(stage).live;
}

DraftProgressLedger* SlotLifecycle::progress(std::uint32_t slot, std::size_t stage) {
  return slots_.at(slot).stages.at(stage).progress.get();
}

const DraftProgressLedger* SlotLifecycle::progress(std::uint32_t slot, std::size_t stage) const {
  return slots_.at(slot).stages.at(stage).progress.get();
}

std::expected<DraftProgressLedger::Unit, std::string> SlotLifecycle::BeginDraftUnit(
    std::uint32_t index, std::size_t s) {
  if (auto active = CheckActive(index); !active) return std::unexpected(active.error());
  Slot& slot = slots_.at(index);
  if (s >= slot.stages.size() || !slot.stages[s].progress)
    return Refuse("stage has no repeatable draft progress");
  Stage& stage = slot.stages[s];
  auto unit = stage.progress->BeginUnit();
  if (!unit) return std::unexpected(unit.error());
  stage.progress_unit = *unit;
  return unit;
}

std::expected<DraftProgressLedger::Successor, std::string> SlotLifecycle::BeginDraftSuccessor(
    std::uint32_t index, std::size_t s) {
  if (auto active = CheckActive(index); !active) return std::unexpected(active.error());
  Slot& slot = slots_.at(index);
  if (s >= slot.stages.size() || !slot.stages[s].progress)
    return Refuse("stage has no repeatable draft progress");
  Stage& stage = slot.stages[s];
  auto successor = stage.progress->BeginSuccessor();
  if (!successor) return std::unexpected(successor.error());
  stage.progress_unit = successor->unit;
  return successor;
}

Status SlotLifecycle::CancelDraftSuccessor(std::uint32_t index, std::size_t s,
                                           DraftProgressLedger::Unit unit) {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  if (s >= (*found)->stages.size() || !(*found)->stages[s].progress)
    return Refuse("stage has no repeatable draft progress");
  Stage& stage = (*found)->stages[s];
  if (auto canceled = stage.progress->CancelSuccessor(unit); !canceled) return canceled;
  stage.progress_unit.reset();
  return {};
}

Status SlotLifecycle::CheckProgressVerify(std::uint32_t index, std::uint32_t rows) const {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  for (const Stage& stage : (*found)->stages) {
    if (!stage.progress) continue;
    if (!stage.progress_unit) return Refuse("target verify has no drafter logical unit");
    if (auto checked = stage.progress->CanVerify(*stage.progress_unit, rows); !checked)
      return checked;
  }
  return {};
}

Status SlotLifecycle::BeginProgressVerify(std::uint32_t index, std::uint32_t rows) {
  if (auto checked = CheckProgressVerify(index, rows); !checked) return checked;
  for (Stage& stage : slots_.at(index).stages) {
    if (!stage.progress) continue;
    if (auto begun = stage.progress->BeginVerify(*stage.progress_unit, rows); !begun) {
      Quarantine(slots_.at(index));
      return begun;
    }
  }
  return {};
}

void SlotLifecycle::ProgressVerifyFailed(std::uint32_t index) {
  for (Stage& stage : slots_.at(index).stages) {
    if (!stage.progress || stage.progress->state() != DraftProgressLedger::State::kVerify) continue;
    if (stage.live->quarantined() || cohort_.faulted())
      stage.progress->Quarantine();
    else
      stage.progress_discard = true;
  }
}

std::vector<LiveState*> SlotLifecycle::AllStates() {
  std::vector<LiveState*> all;
  for (Slot& slot : slots_) {
    for (Stage& stage : slot.stages) all.push_back(stage.live.get());
  }
  return all;
}

std::vector<LiveState*> SlotLifecycle::StatesOf(Slot& slot) {
  std::vector<LiveState*> all;
  for (Stage& stage : slot.stages) all.push_back(stage.live.get());
  return all;
}

std::vector<ExtentId> SlotLifecycle::state() const {
  std::vector<ExtentId> all;
  for (const Slot& slot : slots_) {
    for (const Stage& stage : slot.stages) {
      const auto extents = stage.live->extents();
      all.insert(all.end(), extents.begin(), extents.end());
    }
  }
  return all;
}

std::vector<ExtentId> SlotLifecycle::kept_state() const {
  std::vector<ExtentId> all;
  for (const Slot& slot : slots_) {
    for (const Stage& stage : slot.stages) {
      const auto kept = stage.live->kept_extents();
      all.insert(all.end(), kept.begin(), kept.end());
    }
  }
  return all;
}

std::vector<ExtentId> SlotLifecycle::unchanged_state() const {
  std::vector<ExtentId> all;
  for (const Slot& slot : slots_) {
    if (slot.spilled) continue;
    for (std::size_t s = 0; s < slot.stages.size(); ++s) {
      std::vector<ExtentId> written;
      SplitForSpill(slot.index, s, written, all);
    }
  }
  return all;
}

std::uint32_t SlotLifecycle::positions(std::uint32_t slot) const {
  return slots_.at(slot).positions;
}
bool SlotLifecycle::spilled(std::uint32_t slot) const { return slots_.at(slot).spilled; }
bool SlotLifecycle::adopted(std::uint32_t slot) const {
  return std::ranges::any_of(slots_.at(slot).stages,
                             [](const Stage& s) { return !s.adopted.empty(); });
}
bool SlotLifecycle::restoring(std::uint32_t slot) const {
  return slots_.at(slot).restoring.has_value();
}
bool SlotLifecycle::Borrowed(const Slot& slot) const {
  return std::ranges::any_of(slot.stages, [](const Stage& s) { return s.borrows != 0; });
}
bool SlotLifecycle::borrowed(std::uint32_t slot) const { return Borrowed(slots_.at(slot)); }
bool SlotLifecycle::borrowed(std::uint32_t index, std::size_t stage) const {
  const auto& slot = slots_.at(index);
  auto readers = slot.stages.at(stage).borrows;
  if (stage == 0) {
    // Pending guards prevent writes but grant no ordinary readable snapshot.
    for (const auto& reader : slot.stages)
      if (reader.ordered_borrow && !reader.ordered_borrow->active && readers != 0) --readers;
  }
  return readers != 0;
}
bool SlotLifecycle::verify_pending(std::uint32_t slot) const {
  return std::ranges::any_of(slots_.at(slot).stages, [](const Stage& s) {
    return s.live->verify_rows() != 0 ||
           (s.progress && s.progress->state() == DraftProgressLedger::State::kVerify);
  });
}
std::uint32_t SlotLifecycle::verify_rows(std::uint32_t slot) const {
  std::uint32_t rows = 0;
  for (const Stage& s : slots_.at(slot).stages) rows = std::max(rows, s.live->verify_rows());
  return rows;
}
std::vector<std::size_t> SlotLifecycle::GrownBy(std::size_t stage) const {
  std::vector<std::size_t> out;
  for (std::size_t s = 0; s < options_.stages.size(); ++s) {
    if (stage == 0 || s == stage) out.push_back(s);
  }
  return out;
}
bool SlotLifecycle::owed(std::uint32_t slot) const {
  return std::ranges::any_of(slots_.at(slot).stages, [](const Stage& s) {
    return s.live->owed() || s.progress_discard || (s.progress && s.progress->owed());
  });
}
bool SlotLifecycle::Settled(const Slot& slot) const {
  return std::ranges::none_of(slot.stages, [](const Stage& s) {
    return s.live->owed() || s.live->verify_rows() != 0 || (s.progress && !s.progress->settled());
  });
}
bool SlotLifecycle::ProgressBusy(const Slot& slot) const {
  return std::ranges::any_of(slot.stages, [](const Stage& s) {
    if (!s.progress) return false;
    const auto state = s.progress->state();
    return s.progress->compound_owned() || state == DraftProgressLedger::State::kReplacing ||
           state == DraftProgressLedger::State::kClearing;
  });
}
bool SlotLifecycle::state_refused(std::uint32_t slot) const {
  return slots_.at(slot).state_refused;
}
void SlotLifecycle::ForgetRefusal(std::uint32_t slot) { slots_.at(slot).state_refused = false; }

bool SlotLifecycle::Usable(const Slot& slot) const {
  return !cohort_.faulted() && std::ranges::none_of(slot.stages, [](const Stage& s) {
    return s.live->quarantined() ||
           (s.progress && s.progress->state() == DraftProgressLedger::State::kQuarantined);
  });
}
bool SlotLifecycle::usable(std::uint32_t slot) const {
  const Slot& s = slots_.at(slot);
  return Usable(s) && !s.restoring && !ProgressBusy(s);
}
bool SlotLifecycle::restorable(std::uint32_t slot) const {
  const Slot& s = slots_.at(slot);
  if (!s.spilled || s.positions == 0 || !Usable(s) || s.restoring || Borrowed(s) ||
      verify_pending(slot) || owed(slot))
    return false;
  return std::ranges::all_of(s.stages, [](const Stage& stage) {
    if (stage.restore_ticket)
      return stage.progress && stage.progress->Replacing(*stage.restore_ticket);
    return !stage.progress || stage.progress->settled();
  });
}
bool SlotLifecycle::held(std::uint32_t slot) const {
  return cohort_.IsActive(slot) && node_.InRequest(stream_);
}

bool SlotLifecycle::kept_whole(std::uint32_t slot) const {
  const Slot& s = slots_.at(slot);
  if (!Usable(s) || s.restoring || !Settled(s)) return false;
  bool any = false;
  for (const Stage& stage : s.stages) {
    // Written since its file last held it: the file is behind the state.
    if (!stage.track.on_disk || !stage.adopted.empty() ||
        stage.track.written_from != SpillTrack::kUnwritten || !stage.track.written.empty()) {
      return false;
    }
    any = any || !stage.live->extents().empty();
  }
  return any;
}

std::uint64_t SlotLifecycle::used_bytes(std::uint32_t slot) const {
  std::uint64_t total = 0;
  for (const Stage& stage : slots_.at(slot).stages) total += stage.live->used_bytes();
  return total;
}
std::uint64_t SlotLifecycle::refused_bytes(std::uint32_t slot) const {
  std::uint64_t total = 0;
  for (const Stage& stage : slots_.at(slot).stages) total += stage.live->refused_bytes();
  return total;
}
std::uint64_t SlotLifecycle::spilled_bytes(std::uint32_t slot) const {
  const Slot& s = slots_.at(slot);
  if (!s.spilled) return 0;
  std::uint64_t total = 0;
  for (const Stage& stage : s.stages) {
    total += (stage.adopted.empty() ? stage.live->extents().size() : stage.adopted.size()) *
             kPagedExtent;
  }
  return total;
}
std::uint64_t SlotLifecycle::retained_spill_bytes(std::uint32_t slot) const {
  const Slot& s = slots_.at(slot);
  if (s.spilled) return 0;
  std::uint64_t total = 0;
  for (const Stage& stage : s.stages) total += stage.live->spill_file_bytes();
  return total;
}
std::uint64_t SlotLifecycle::spill_write_bytes(std::uint32_t slot) const {
  const Slot& s = slots_.at(slot);
  if (s.spilled) return 0;
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < s.stages.size(); ++i) {
    std::vector<ExtentId> written;
    std::vector<ExtentId> unchanged;
    SplitForSpill(slot, i, written, unchanged);
    total += written.size() * kPagedExtent;
  }
  return total;
}

// ------------------------------------------------------------------ closures

Status SlotLifecycle::RefreshClosures(SlotMask protect) {
  places_clean_.reset();
  auto refreshed = node_.Call(
      [&]() -> Status {
        std::array<std::vector<ExtentId>, kMaxRequestSlots> extents;
        std::array<const std::vector<ExtentId>*, kMaxRequestSlots> of{};
        for (const Slot& slot : slots_) {
          if (slot.spilled) continue;  // in no closure until Restore
          for (const Stage& stage : slot.stages) {
            const auto live = stage.live->extents();
            extents[slot.index].insert(extents[slot.index].end(), live.begin(), live.end());
          }
          of[slot.index] = &extents[slot.index];
        }
        auto closures = cohort_.Build(node_.catalog(), shared_, of, protect);
        if (!closures) return std::unexpected(closures.error());
        everything_ = std::move(closures->everything);
        fence_ = std::move(closures->fence);
        execution_ = std::move(closures->execution);
        for (Slot& slot : slots_) slot.fence = std::move(closures->slot_fences[slot.index]);
        return {};
      },
      "refreshing the slots' closures");
  const auto states = AllStates();
  if (!refreshed) {
    cohort_.Fault(states);
    return refreshed;
  }
  return cohort_.Hold(node_, stream_, execution_, states);
}

Status SlotLifecycle::KeepFor(const Slot& slot, catalog::Closure& keep) {
  std::vector<ExtentId> ids;
  for (const auto& [id, generation] : execution_.extents) {
    (void)generation;
    ids.push_back(id);
  }
  for (const Stage& stage : slot.stages) {
    const auto live = stage.live->extents();
    ids.insert(ids.end(), live.begin(), live.end());
  }
  return node_.Call(
      [&]() -> Status {
        auto of = node_.catalog().ClosureOfExtents(ids);
        if (!of) return Refuse("slot state is no longer cataloged");
        keep = std::move(*of);
        return {};
      },
      "describing a slot's protected state");
}

void SlotLifecycle::Quarantine(Slot& slot) {
  for (Stage& stage : slot.stages) {
    stage.live->Quarantine();
    if (stage.progress) stage.progress->Quarantine();
  }
}

void SlotLifecycle::QuarantineOwners(std::span<const std::uint32_t> indices) {
  for (const auto index : indices) {
    base::Check(index < slots_.size(), "quarantined owner is admitted");
    Quarantine(slots_[index]);
  }
}

void SlotLifecycle::Fault() { cohort_.Fault(AllStates()); }

void SlotLifecycle::CheckFailedJob() {
  const auto states = AllStates();
  cohort_.CheckFailedJob(node_, stream_, execution_, states);
}

Status SlotLifecycle::ResetSlot(Slot& slot) {
  slot.spilled = false;
  slot.positions = 0;
  slot.state_refused = false;
  slot.restoring.reset();
  slot.restore_needed.clear();
  slot.restored_bytes.clear();
  for (Stage& stage : slot.stages) stage.adopted.clear();
  for (Stage& stage : slot.stages) {
    if (!stage.progress) continue;
    // ResetSlot follows a proven zero/discard, with no work in flight.
    stage.progress->Quarantine();
    auto clear = stage.progress->BeginClear();
    if (!clear || !stage.progress->ClearCompleted(*clear)) {
      Quarantine(slot);
      return Refuse("cleared drafter progress could not reset");
    }
    stage.control_ticket.reset();
    stage.tail_ticket.reset();
    stage.tail_attempted = false;
    stage.tail_queued = false;
    stage.restore_ticket.reset();
    stage.control_queued = false;
    stage.progress_unit.reset();
    stage.progress_discard = false;
  }
  return {};
}

// ---------------------------------------------------------------------- verbs

Status SlotLifecycle::CheckActive(std::uint32_t index) const {
  return Ready(index, /*within_restore=*/false);
}

Status SlotLifecycle::Ready(std::uint32_t index, bool within_restore) const {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  const Slot& slot = **found;
  if (!bound_ || slot.spilled) return Refuse(std::format("slot {} is not ready", index));
  // Between a checkpoint restore's Prepare and Complete the state is part
  // old, part checkpoint: only the restore's own copies and completion.
  if (slot.restoring && !within_restore) {
    return Refuse(std::format("slot {} is mid-restore: complete it first", index));
  }
  if (ProgressBusy(slot) && !within_restore && !slot.compound_access)
    return Refuse("drafter progress has an owned compound or replacement transition");
  if (auto active = cohort_.Check(node_, stream_, index); !active) return active;
  for (const Stage& stage : slot.stages) {
    if (auto usable = stage.live->Usable(); !usable) return usable;
    if (stage.progress && stage.progress->state() == DraftProgressLedger::State::kQuarantined)
      return Refuse("drafter progress is quarantined");
  }
  return {};
}

Status SlotLifecycle::SelectSlots(std::span<const std::uint32_t> slots) {
  if (!bound_ || released_ || cohort_.faulted()) return Refuse("cohort requires retirement");
  auto mask = cohort_.MaskOf(slots);
  if (!mask) return std::unexpected(mask.error());
  for (const Slot& slot : slots_) {
    if ((*mask & (SlotMask{1} << slot.index)) != 0) continue;
    // A cold kept peer owns only its hydration ticket, not active work.
    // It may remain unselected until that peer's own request restores it.
    if (slot.restoring || Borrowed(slot) || verify_pending(slot.index) ||
        (ProgressBusy(slot) && !restorable(slot.index))) {
      return Refuse(std::format(
          "selection cannot drop slot {}: a restore, borrow or verify still needs it", slot.index));
    }
  }
  // An unchanged selection within an open request keeps its closure; state
  // growth, clears and restores still refresh it.
  if (*mask == cohort_.active() && node_.InRequest(stream_)) return {};
  cohort_.Select(*mask);
  return RefreshClosures();
}

Status SlotLifecycle::Clear(std::uint32_t index) {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  Slot& slot = **found;
  if (Borrowed(slot) || verify_pending(index) ||
      std::ranges::any_of(
          slot.stages, [](const Stage& s) { return s.progress && s.progress->compound_owned(); })) {
    return Refuse("state cannot clear while borrowed or while a verify awaits its Accept");
  }
  if (!bound_ || cohort_.faulted()) return Refuse("cohort requires retirement");
  if (auto active = cohort_.Check(node_, stream_, index); !active) return active;
  if (options_.invalidate) options_.invalidate(index);
  if (options_.content_changed) options_.content_changed(index);
  // Zeroed in place, the discard keeps the resident backing for the next
  // conversation's growth (LiveState::ZeroForReuse). From the first zeroing
  // on, whatever follows fails, no stage's file holds it.
  for (Stage& stage : slot.stages) stage.track.Lost();
  std::vector<bool> keep(slot.stages.size(), false);
  if (!slot.spilled) {
    for (std::size_t s = 0; s < slot.stages.size(); ++s) {
      auto zeroed = slot.stages[s].live->ZeroForReuse(node_, slot.fence, stream_);
      if (!zeroed) {
        Quarantine(slot);
        CheckFailedJob();
        return std::unexpected(zeroed.error());
      }
      keep[s] = *zeroed;
    }
  }
  // Only this slot's state leaves the held request; the shared extents and
  // every other active slot's state stay protected.
  if (auto others = RefreshClosures(cohort_.active() & ~(SlotMask{1} << index)); !others) {
    return others;
  }
  Status cleared;
  for (std::size_t s = 0; s < slot.stages.size(); ++s) {
    auto discarded = slot.stages[s].live->DiscardGrowingState(node_, keep[s]);
    if (!discarded && cleared) cleared = discarded;
  }
  if (cleared) cleared = ResetSlot(slot);
  if (auto refreshed = RefreshClosures(); !refreshed) return refreshed;
  return cleared;
}

Status SlotLifecycle::ClearIdle(std::uint32_t index) {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  Slot& slot = **found;
  if (!bound_ || cohort_.faulted()) return Refuse("cohort requires retirement");
  // Leased by the request open on the stream: cleared within it (Clear).
  if (held(index)) return Refuse("active slot is cleared within its own request");
  if (Borrowed(slot) || verify_pending(index) ||
      std::ranges::any_of(
          slot.stages, [](const Stage& s) { return s.progress && s.progress->compound_owned(); })) {
    return Refuse("state cannot clear while borrowed or while a verify awaits its Accept");
  }
  slot.state_refused = false;
  if (options_.invalidate) options_.invalidate(index);
  if (options_.content_changed) options_.content_changed(index);
  // Outside the stream's lease, which keeps protecting every active state;
  // the catalog refuses the discard if anything still holds or operates on
  // this state's extents (LiveState::DiscardGrowingState).
  Status cleared;
  for (Stage& stage : slot.stages) {
    auto discarded = stage.live->DiscardGrowingState(node_);
    stage.track.Lost();
    if (!discarded && cleared) cleared = discarded;
  }
  if (cleared) cleared = ResetSlot(slot);
  if (auto refreshed = RefreshClosures(); !refreshed) return refreshed;
  return cleared;
}

void SlotLifecycle::SplitForSpill(std::uint32_t index, std::size_t s,
                                  std::vector<ExtentId>& written,
                                  std::vector<ExtentId>& unchanged) const {
  const Slot& slot = slots_.at(index);
  const Stage& stage = slot.stages.at(s);
  std::vector<Range> changed = stage.track.written;
  bool known = stage.track.on_disk;
  if (known && stage.track.written_from != SpillTrack::kUnwritten) {
    // What writes from that position may change: the contract's mutable
    // ranges from it (a turn checkpoint's).
    auto writes = model::CheckpointWrites(
        options_.stages[s].layout,
        std::min(stage.track.written_from, options_.stages[s].layout.context));
    if (writes) {
      const auto ranges = AsRanges(*writes);
      changed.insert(changed.end(), ranges.begin(), ranges.end());
    } else {
      known = false;
    }
  }
  if (!known) {
    const auto all = stage.live->extents();
    written.insert(written.end(), all.begin(), all.end());
    return;
  }
  stage.live->SplitExtents(changed, written, unchanged);
}

Status SlotLifecycle::Spill(std::uint32_t index) {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  Slot& slot = **found;
  if (!bound_ || cohort_.faulted()) return Refuse("cohort requires retirement");
  if (slot.restoring) return Refuse("restore must complete before spill");
  if (Borrowed(slot)) return Refuse("borrowed state cannot spill");
  if (ProgressBusy(slot)) return Refuse("owned drafter progress cannot spill");
  if (slot.spilled) return {};
  for (const Stage& stage : slot.stages) {
    if (auto usable = stage.live->Usable(); !usable) return usable;
    if (auto settled = stage.live->AwaitingAccept(); !settled) return settled;
  }
  if (std::ranges::all_of(slot.stages, [](const Stage& s) { return s.live->extents().empty(); })) {
    return {};  // nothing initialized: nothing to spill
  }
  std::vector<std::vector<ExtentId>> written(slot.stages.size());
  std::vector<std::vector<ExtentId>> unchanged(slot.stages.size());
  for (std::size_t s = 0; s < slot.stages.size(); ++s) {
    SplitForSpill(index, s, written[s], unchanged[s]);
  }
  // Out of every closure first, so the held request no longer leases it
  // (its peers stay protected); then written back and released. A restore
  // or commit a verify still owes stays owed: its snapshot is runtime
  // memory, not spilled.
  slot.spilled = true;
  if (auto refreshed = RefreshClosures(); !refreshed) {
    slot.spilled = false;
    return refreshed;
  }
  // Spilled from here, written back whole or not. Its logical cells remain.
  if (options_.invalidate) options_.invalidate(index);
  for (std::size_t s = 0; s < slot.stages.size(); ++s) {
    // Only what changed since the spill file last held it is written; the
    // rest is released with the file's copy kept (the catalog checks the
    // place saved its current generation).
    for (const auto& [extents, same] :
         {std::pair{&written[s], false}, std::pair{&unchanged[s], true}}) {
      if (extents->empty()) continue;
      if (auto evicted = node_.Evict(*extents, {.unchanged = same}); !evicted) {
        // Some extents written back, others not: still spilled, so Restore
        // brings back whichever are not resident.
        slot.stages[s].track.Lost();
        return evicted;
      }
    }
    // An empty stage's file holds nothing either.
    FileHolds(Settled(slot), slot.stages[s].track);
  }
  return {};
}

Status SlotLifecycle::ReleaseSpillFiles(std::uint32_t index) {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  Slot& slot = **found;
  if (!bound_ || cohort_.faulted()) return Refuse("cohort requires retirement");
  if (slot.spilled || slot.restoring) return Refuse("a spilled state's files hold it");
  if (Borrowed(slot) || verify_pending(index)) {
    return Refuse("state cannot change files while borrowed or while a verify awaits its Accept");
  }
  for (Stage& stage : slot.stages) {
    // Lost first: whatever the release leaves, the next spill writes whole.
    stage.track.Lost();
    if (auto released = stage.live->ReleaseSpillFile(node_); !released) return released;
  }
  return {};
}

Status SlotLifecycle::Restore(std::uint32_t index) {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  Slot& slot = **found;
  slot.state_refused = false;
  if (!bound_ || cohort_.faulted()) return Refuse("cohort requires retirement");
  if (!Usable(slot)) return Refuse("quarantined state cannot restore");
  if (slot.restoring) return Refuse("restore must complete before a disk restore");
  if (!slot.spilled) return {};
  const bool kept =
      std::ranges::any_of(slot.stages, [](const Stage& s) { return s.restore_ticket.has_value(); });
  // Stages spilled here: read back at the generations they were written at
  // (invariant 4), under the budget. Their spill files still hold them as
  // they are, if the spill completed (SpillTrack).
  std::vector<ExtentId> spilled;
  for (const Stage& stage : slot.stages) {
    if (!stage.adopted.empty()) continue;
    const auto extents = stage.live->extents();
    spilled.insert(spilled.end(), extents.begin(), extents.end());
  }
  if (!spilled.empty()) {
    catalog::Closure closure;
    if (auto described = node_.Call(
            [&]() -> Status {
              auto of = node_.catalog().ClosureOfExtents(spilled);
              if (!of) return Refuse("spilled state is no longer cataloged");
              closure = std::move(*of);
              return {};
            },
            "describing a spilled state");
        !described) {
      return described;
    }
    scheduler::AcquireReport report;
    bool over_budget = false;
    if (auto acquired = node_.Acquire(closure, report, "restoring a spilled state", &over_budget);
        !acquired) {
      slot.state_refused = over_budget && Usable(slot);
      return acquired;
    }
  }
  // Stages kept from the process before (D-105): their first load reads
  // them from the named spill file, each beside the execution closure and
  // the slot's other stages; a clean refusal leaves the slot spilled.
  for (Stage& stage : slot.stages) {
    if (stage.adopted.empty()) continue;
    catalog::Closure keep;
    if (auto described = KeepFor(slot, keep); !described) return described;
    bool over_budget = false;
    if (auto used = stage.live->Use(node_, stage.adopted, &keep, &over_budget); !used) {
      if (auto refreshed = RefreshClosures(); !refreshed) return refreshed;
      slot.state_refused = over_budget && Usable(slot);
      return std::unexpected(used.error());
    }
    stage.adopted.clear();
    FileHolds(Settled(slot), stage.track);  // read from its file
  }
  slot.spilled = false;
  if (auto refreshed = RefreshClosures(); !refreshed) return refreshed;
  if (auto hydrated = HydrateProgress(slot, slot.positions); !hydrated) return hydrated;
  if (kept)
    for (Stage& stage : slot.stages) FileHolds(Settled(slot), stage.track);
  return {};
}

Status SlotLifecycle::BeginProgressReplacement(Slot& slot) {
  for (Stage& stage : slot.stages) {
    if (!stage.progress) continue;
    auto ticket = stage.progress->BeginReplacement();
    if (!ticket) {
      Quarantine(slot);
      return std::unexpected(ticket.error());
    }
    stage.restore_ticket = *ticket;
  }
  return {};
}

Status SlotLifecycle::HydrateProgress(Slot& slot, std::uint32_t positions) {
  // All records validate before any mirror is published. Required numeric
  // pages have already materialized, and the closure leases them through
  // each synchronous control read. Same-process ready/dirty mirrors are
  // authoritative across ordinary spill and have no restore ticket.
  for (std::size_t s = 0; s < slot.stages.size(); ++s) {
    Stage& stage = slot.stages[s];
    if (!stage.restore_ticket) continue;
    *stage.control_hydrate = {};
    if (positions != 0) {
      const Range range{
          .region = 0, .offset = stage.control_offset, .bytes = model::kDraftProgressBytes};
      CopyRetirement retirement = CopyRetirement::kProven;
      auto copied = stage.live->Copy(node_, slot.fence, stream_, stage.control_hydrate,
                                     std::span(&range, 1), true, &retirement);
      if (!copied) {
        Quarantine(slot);
        CheckFailedJob();
        if (retirement == CopyRetirement::kUnproven) Fault();
        return copied;
      }
    }
    auto bounds = model::DraftBounds(options_.stages[s].layout);
    if (!bounds || !bounds->has_value()) {
      Quarantine(slot);
      return Refuse("restored progress lost its declared bounds");
    }
    auto validated = model::DecodeDraftProgress(*stage.control_hydrate, positions, **bounds);
    if (!validated) {
      Quarantine(slot);
      return std::unexpected(validated.error());
    }
  }
  for (Stage& stage : slot.stages) {
    if (!stage.restore_ticket) continue;
    if (auto restored = stage.progress->RestoreCompleted(*stage.restore_ticket,
                                                         *stage.control_hydrate, positions);
        !restored) {
      Quarantine(slot);
      return restored;
    }
    stage.restore_ticket.reset();
  }
  return {};
}

Status SlotLifecycle::ReserveStateThrough(std::uint32_t index, std::uint32_t positions,
                                          std::uint32_t read_align, std::size_t stage) {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  Slot& slot = **found;
  if (stage >= options_.stages.size()) return Refuse(std::format("no stage {}", stage));
  if (slot.restoring) return Refuse("restore must complete before state growth");
  const std::vector<std::size_t> grown = GrownBy(stage);
  if (verify_pending(index) ||
      std::ranges::any_of(grown, [&](std::size_t s) { return slot.stages[s].borrows != 0; })) {
    return Refuse("state cannot grow while borrowed or while a verify awaits its Accept");
  }
  slot.state_refused = false;
  if (auto active = CheckActive(index); !active) return active;
  // The step's stage at its reads' alignment; any other stage it writes at
  // its own, through its own context at most.
  std::vector<std::vector<Range>> ranges(slot.stages.size());
  for (const std::size_t s : grown) {
    const std::uint32_t align = s == stage && read_align != 0 ? read_align : this->read_align(s);
    const model::StateLayoutDerived& layout = options_.stages[s].layout;
    auto used = model::UsedThrough(
        layout, s == stage ? positions : std::min(positions, layout.context), align);
    if (!used) return std::unexpected(used.error());
    ranges[s] = AsRanges(*used);
  }
  bool grew = false;
  bool over_budget = false;
  Status failed;
  for (const std::size_t s : grown) {
    if (!failed) break;
    // A later stage grows beside an earlier one's fresh extents, which the
    // execution closure does not hold yet.
    catalog::Closure keep;
    const catalog::Closure* protect = &execution_;
    if (grew) {
      if (auto described = KeepFor(slot, keep); !described) {
        failed = described;
        break;
      }
      protect = &keep;
    }
    auto used = slot.stages[s].live->Use(node_, ranges[s], protect, &over_budget);
    if (!used) {
      failed = std::unexpected(used.error());
    } else {
      grew = grew || *used;
    }
  }
  // Even a partial clean refusal may have initialized fresh extents.
  if (!failed || grew) {
    if (auto refreshed = RefreshClosures(); !refreshed) {
      Quarantine(slot);
      return Refuse(std::format("state growth{}; {}",
                                failed ? std::string() : ": " + failed.error(), refreshed.error()));
    }
  }
  if (!failed) {
    // Only a clean refusal that left the state usable and the cohort
    // healthy may be retried once leased state is freed.
    slot.state_refused = over_budget && Usable(slot);
    return failed;
  }
  return {};
}

std::expected<bool, std::string> SlotLifecycle::PrepareThrough(std::uint32_t index,
                                                               std::uint32_t positions,
                                                               std::uint32_t read_align,
                                                               std::size_t stage) {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  const Slot& slot = **found;
  if (stage >= options_.stages.size()) return Refuse(std::format("no stage {}", stage));
  const std::vector<std::size_t> grown = GrownBy(stage);
  // Preparation overlaps the current job only on a threaded node.
  if (!node_.threaded() || slot.restoring || verify_pending(index) || !Usable(slot) ||
      std::ranges::any_of(grown, [&](std::size_t s) { return slot.stages[s].borrows != 0; })) {
    return false;
  }
  if (auto active = CheckActive(index); !active) return false;
  bool submitted = false;
  for (const std::size_t s : grown) {
    const std::uint32_t align = s == stage && read_align != 0 ? read_align : this->read_align(s);
    const model::StateLayoutDerived& layout = options_.stages[s].layout;
    auto used = model::UsedThrough(
        layout, s == stage ? positions : std::min(positions, layout.context), align);
    if (!used) return std::unexpected(used.error());
    const auto ranges = AsRanges(*used);
    auto prepared = slot.stages[s].live->Prepare(node_, ranges);
    if (!prepared) return std::unexpected(prepared.error());
    submitted = submitted || *prepared;
  }
  return submitted;
}

Status SlotLifecycle::FinishPreparations(bool cancel) {
  Status first;
  for (Slot& slot : slots_) {
    for (Stage& stage : slot.stages) {
      if (auto finished = stage.live->FinishPreparation(cancel); !finished && first) {
        first = std::unexpected(finished.error());
      }
    }
  }
  return first;
}

std::expected<std::vector<Range>, std::string> SlotLifecycle::CheckpointRanges(
    std::uint32_t index, std::uint32_t positions) const {
  if (auto active = CheckActive(index); !active) return std::unexpected(active.error());
  const Slot& slot = slots_[index];
  if (slot.restoring || verify_pending(index) || owed(index)) {
    return Refuse("checkpoint needs a settled state: no restore or verify pending");
  }
  std::vector<Range> out;
  for (std::size_t s = 0; s < slot.stages.size(); ++s) {
    auto writes = model::CheckpointWrites(options_.stages[s].layout,
                                          std::min(positions, options_.stages[s].layout.context));
    if (!writes) return std::unexpected(writes.error());
    auto pages = CheckpointPages(slot.stages[s].live->used_ranges(), AsRanges(*writes));
    if (!pages) return std::unexpected(pages.error());
    for (Range& page : *pages) {
      page.region = s;
      out.push_back(page);
    }
  }
  return out;
}

Status SlotLifecycle::ValidateFootprint(std::uint32_t positions,
                                        std::span<const Range> footprint) const {
  std::size_t previous_stage = 0;
  std::uint64_t previous_end = 0;
  for (const Range& r : footprint) {
    if (r.region >= options_.stages.size()) return Refuse("restored footprint names no stage");
    const std::uint64_t bytes = options_.stages[r.region].layout.bytes;
    if (r.offset >= bytes || r.offset % kPagedExtent != 0 ||
        r.bytes != std::min(kPagedExtent, bytes - r.offset) || r.region < previous_stage ||
        (r.region == previous_stage && r.offset < previous_end)) {
      return Refuse("restored footprint is not an ordered initialized extent set");
    }
    previous_stage = r.region;
    previous_end = r.offset + r.bytes;
  }
  if (positions == 0) {
    if (!footprint.empty()) return Refuse("empty restore has backing");
    return {};
  }
  for (std::size_t s = 0; s < options_.stages.size(); ++s) {
    const auto& layout = options_.stages[s].layout;
    if (positions > layout.context) return Refuse("restored positions pass the context");
    auto needed = model::UsedThrough(layout, positions);
    if (!needed) return std::unexpected(needed.error());
    for (const auto& r : *needed) {
      if (r.bytes == 0) continue;
      for (std::uint64_t e = r.offset / kPagedExtent; e <= (r.offset + r.bytes - 1) / kPagedExtent;
           ++e) {
        if (std::ranges::none_of(footprint, [&](const Range& got) {
              return got.region == s && got.offset / kPagedExtent == e;
            })) {
          return Refuse("restored footprint does not fund its completed positions");
        }
      }
    }
  }
  return {};
}

Status SlotLifecycle::PrepareRestore(std::uint32_t index, std::uint32_t positions,
                                     std::span<const Range> footprint,
                                     std::string_view source_layout) {
  if (source_layout.empty() || source_layout != kept_layout_id_) {
    return Refuse("checkpoint source layout differs");
  }
  if (auto valid = ValidateFootprint(positions, footprint); !valid) return valid;
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  Slot& slot = **found;
  // A restore or commit a verify still owes would run after the restore,
  // over the checkpoint's bytes (or into its dropped tail): settled first.
  if (Borrowed(slot) || verify_pending(index) || owed(index)) {
    return Refuse(
        "state cannot restore while borrowed or while a verify awaits its Accept or owes a "
        "restore or commit");
  }
  if (auto active = CheckActive(index); !active) return active;
  if (slot.restoring) return Refuse("restore is already pending");
  // Retain needs this destination unleased; every selected peer stays held.
  if (auto others = RefreshClosures(cohort_.active() & ~(SlotMask{1} << index)); !others) {
    return others;
  }
  slot.state_refused = false;
  // Every stage grown to the footprint first, so a clean refusal leaves the
  // state as it was (with fresh zero pages), and what its runner derived
  // from it with it; only then the newer tail dropped, which replaces it.
  Status prepared;
  bool over_budget = false;
  for (std::size_t s = 0; s < slot.stages.size() && prepared; ++s) {
    catalog::Closure keep;
    if (auto described = KeepFor(slot, keep); !described) {
      prepared = described;
      Quarantine(slot);
      break;
    }
    if (auto used = slot.stages[s].live->Use(node_, OfStage(footprint, s), &keep, &over_budget);
        !used) {
      prepared = std::unexpected(used.error());
      slot.state_refused = over_budget && Usable(slot);
      if (!slot.state_refused) Quarantine(slot);
    }
  }
  if (!slot.state_refused && options_.invalidate) options_.invalidate(index);
  if (!slot.state_refused && options_.content_changed) options_.content_changed(index);
  if (prepared) prepared = BeginProgressReplacement(slot);
  if (prepared) {
    for (Stage& stage : slot.stages) stage.track.Lost();
    for (std::size_t s = 0; s < slot.stages.size() && prepared; ++s) {
      prepared = slot.stages[s].live->Retain(node_, OfStage(footprint, s));
      if (!prepared) Quarantine(slot);
    }
  }
  if (auto refreshed = RefreshClosures(); !refreshed) return refreshed;
  if (!prepared) return prepared;
  // What the checkpoint holds: the footprint's pages later positions could
  // have written (CheckpointRanges at its capture).
  std::vector<Range> needed;
  for (std::size_t s = 0; s < slot.stages.size(); ++s) {
    auto writes = model::CheckpointWrites(options_.stages[s].layout, positions);
    if (!writes) return std::unexpected(writes.error());
    auto pages = CheckpointPages(OfStage(footprint, s), AsRanges(*writes));
    if (!pages) return std::unexpected(pages.error());
    for (Range& page : *pages) {
      page.region = s;
      needed.push_back(page);
    }
  }
  slot.restore_needed = std::move(needed);
  slot.restored_bytes.assign(slot.restore_needed.size(), 0);
  slot.restoring = positions;
  return {};
}

Status SlotLifecycle::CompleteRestore(std::uint32_t index, std::uint32_t positions) {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  Slot& slot = **found;
  if (verify_pending(index) || owed(index)) {
    return Refuse("a verify awaits its Accept or owes a restore or commit");
  }
  if (auto active = Ready(index, /*within_restore=*/true); !active) return active;
  if (!slot.restoring || *slot.restoring != positions) {
    return Refuse("restore completion differs from its checked metadata");
  }
  for (std::size_t i = 0; i < slot.restore_needed.size(); ++i) {
    if (slot.restored_bytes[i] != slot.restore_needed[i].bytes) {
      return Refuse("restore has not completed every checkpoint page");
    }
  }
  if (options_.invalidate) options_.invalidate(index);
  if (options_.content_changed) options_.content_changed(index);
  if (auto hydrated = HydrateProgress(slot, positions); !hydrated) return hydrated;
  slot.positions = positions;
  slot.restoring.reset();
  slot.restore_needed.clear();
  slot.restored_bytes.clear();
  return {};
}

Status SlotLifecycle::CopyState(std::uint32_t index, void* pinned, std::span<const Range> ranges,
                                bool to_host, CopyRetirement* retirement) {
  if (retirement != nullptr) *retirement = CopyRetirement::kProven;
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  Slot& slot = **found;
  if (to_host == slot.restoring.has_value()) {
    return Refuse("copy direction differs from the restore state");
  }
  // Out, the bytes would not be the kept rows' state; in, a settle that
  // follows would overwrite the pages copied (and completion accept them).
  if (verify_pending(index) || owed(index)) {
    return Refuse("a verify awaits its Accept or owes a restore or commit");
  }
  if (!to_host && Borrowed(slot)) return Refuse("borrowed state cannot be overwritten");
  if (auto active = Ready(index, /*within_restore=*/!to_host); !active) return active;
  for (const Range& r : ranges) {
    if (r.region >= slot.stages.size()) return Refuse("a copied range names no stage");
  }
  if (!to_host) {
    if (auto recorded = RecordWrite(index, ranges); !recorded) return recorded;
    if (options_.invalidate) options_.invalidate(index);
    if (options_.content_changed) options_.content_changed(index);
  }
  // Packed in their order: each run of one stage's ranges is one copy.
  auto* host = static_cast<std::byte*>(pinned);
  Status copied;
  CopyRetirement completed = CopyRetirement::kProven;
  for (std::size_t at = 0; at < ranges.size() && copied;) {
    std::size_t end = at;
    std::uint64_t bytes = 0;
    std::vector<Range> run;
    while (end < ranges.size() && ranges[end].region == ranges[at].region) {
      run.push_back({.region = 0, .offset = ranges[end].offset, .bytes = ranges[end].bytes});
      bytes += ranges[end].bytes;
      ++end;
    }
    copied = slot.stages[ranges[at].region].live->Copy(node_, slot.fence, stream_, host, run,
                                                       to_host, &completed);
    host += bytes;
    at = end;
  }
  if (retirement != nullptr) *retirement = completed;
  if (!copied) {
    // A copy in may have written part of the state.
    if (!to_host) Quarantine(slot);
    CheckFailedJob();
    if (completed == CopyRetirement::kUnproven) Fault();
    return copied;
  }
  if (!to_host && slot.restoring) {
    // A checkpoint's pages are copied in order: each advances its page's
    // progress only from where it stands.
    for (std::size_t i = 0; i < slot.restore_needed.size(); ++i) {
      const Range& needed = slot.restore_needed[i];
      for (const Range& r : ranges) {
        const std::uint64_t at = needed.offset + slot.restored_bytes[i];
        if (r.region == needed.region && r.offset <= at && r.bytes > at - r.offset) {
          slot.restored_bytes[i] = std::min(needed.bytes, r.offset + r.bytes - needed.offset);
        }
      }
    }
  }
  return {};
}

Status SlotLifecycle::Adopt(std::uint32_t index, std::uint32_t positions,
                            std::span<const Range> footprint, std::string_view source_layout) {
  if (source_layout.empty() || source_layout != kept_layout_id_) {
    return Refuse("kept state's layout differs");
  }
  if (auto valid = ValidateFootprint(positions, footprint); !valid) return valid;
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  Slot& slot = **found;
  if (!bound_ || !Usable(slot) || positions == 0 || footprint.empty() || slot.positions != 0 ||
      slot.spilled || used_bytes(index) != 0 || slot.restoring || Borrowed(slot) || held(index)) {
    return Refuse("adopts kept state only into an empty, idle, healthy slot");
  }
  // Zeroed backing a clear kept would be taken back by the restore's
  // growth instead of the file's bytes being read.
  if (std::ranges::any_of(slot.stages,
                          [](const Stage& s) { return !s.live->kept_extents().empty(); })) {
    return Refuse("adopts kept state only into a slot holding no kept backing");
  }
  if (auto replacing = BeginProgressReplacement(slot); !replacing) return replacing;
  for (std::size_t s = 0; s < slot.stages.size(); ++s) {
    slot.stages[s].adopted = OfStage(footprint, s);
    slot.stages[s].track.Lost();
  }
  // Spilled, outside every closure, until Restore reads it from the files.
  slot.spilled = true;
  slot.positions = positions;
  if (options_.invalidate) options_.invalidate(index);
  if (options_.content_changed) options_.content_changed(index);
  return RefreshClosures();
}

Status SlotLifecycle::CheckPlaces() {
  if (!bound_ || released_) return Refuse("slots are not bound");
  const std::uint64_t changes = node_.scheduler().placement_changes();
  if (places_clean_ == changes) return {};
  PlaceCheck check;
  auto checked = node_.Call(
      [&]() -> Status {
        for (const PagedWeights* weights : weights_) {
          weights->CheckPlaces(node_.scheduler(), check);
        }
        for (const Slot& slot : slots_) {
          for (const Stage& stage : slot.stages) {
            stage.live->CheckPlaces(node_.scheduler(), check);
          }
        }
        return {};
      },
      "checking the slots' places");
  if (!checked) return checked;
  if (check.moved != 0) {
    if (options_.drop_plans) options_.drop_plans();
    return Refuse(std::format("{} places moved (first: {}); every graph was dropped", check.moved,
                              check.first));
  }
  places_clean_ = changes;
  return {};
}

void SlotLifecycle::StateWrittenBack(bool whole) {
  for (Slot& slot : slots_) {
    if (slot.spilled) continue;
    const bool saved = whole && Usable(slot) && !slot.restoring;
    for (Stage& stage : slot.stages) FileHolds(saved && Settled(slot), stage.track);
  }
}

std::expected<SlotMask, std::string> SlotLifecycle::RecoverInPlace(
    const std::function<void(std::uint32_t slot)>& before_discard) {
  if (released_ || !bound_) return Refuse("slots are not bound");
  if (node_.InRequest(stream_)) return Refuse("a request is still open on the stream");
  const bool faulted = cohort_.faulted();
  cohort_.Recover();
  SlotMask discarded = 0;
  // The stream fenced and no request open: no reader still holds a borrow,
  // whose Return the failure may have skipped; one returned later returns
  // nothing (its epoch is gone).
  ++borrow_epoch_;
  for (Slot& slot : slots_) {
    for (Stage& stage : slot.stages) {
      stage.borrows = 0;
      stage.ordered_borrow.reset();
    }
    // Likewise a verify's Accept or a checkpoint restore's last copies:
    // the slot goes whole (kept, outside the default selection, it would
    // refuse every selection that drops it).
    const bool interrupted = slot.restoring.has_value() || verify_pending(slot.index);
    // On disk since before the failure, and nothing ran on it (or holding
    // nothing at all): kept. Every stage must be, or the slot goes whole.
    if (!interrupted && (slot.spilled || std::ranges::all_of(slot.stages, [](const Stage& s) {
                           return s.live->extents().empty();
                         }))) {
      bool preserved = true;
      for (Stage& stage : slot.stages) preserved = stage.live->LiftIfPreserved(node_) && preserved;
      if (preserved) continue;
    }
    const bool touched =
        interrupted || faulted ||
        std::ranges::any_of(slot.stages, [](const Stage& s) { return s.live->quarantined(); });
    if (!touched) continue;  // untouched by the failed work
    if (before_discard) before_discard(slot.index);
    if (options_.invalidate) options_.invalidate(slot.index);
    if (options_.content_changed) options_.content_changed(slot.index);
    for (Stage& stage : slot.stages) {
      auto cleared = stage.live->DiscardGrowingState(node_);
      stage.track.Lost();
      if (!cleared) {
        Fault();
        return Refuse(std::format("discarding slot {}'s state: {}", slot.index, cleared.error()));
      }
    }
    if (auto reset = ResetSlot(slot); !reset) {
      Fault();
      return std::unexpected(reset.error());
    }
    discarded |= SlotMask{1} << slot.index;
  }
  if (options_.drop_plans) options_.drop_plans();
  if (auto refreshed = RefreshClosures(); !refreshed) return std::unexpected(refreshed.error());
  return discarded;
}

// ---------------------------------------------------------- what steps tell

// A spilled slot (an adopted one among them) is in no closure, so no job
// may write it: a record then means a step the runner should have refused
// (CheckActive), and an adopted slot's Restore would drop it unseen (its
// files hold the state then). Mid-restore a record is harmless: Prepare
// left every stage's file behind, and the restore's copies record theirs.
Status SlotLifecycle::RecordWrite(std::uint32_t index, std::uint32_t position) {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  if ((*found)->spilled) return Refuse(std::format("slot {} is spilled: nothing writes it", index));
  for (Stage& stage : (*found)->stages) stage.track.Wrote(position);
  return {};
}

Status SlotLifecycle::RecordWrite(std::uint32_t index, std::span<const Range> ranges) {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  Slot& s = **found;
  if (s.spilled) return Refuse(std::format("slot {} is spilled: nothing writes it", index));
  for (const Range& r : ranges) {
    if (r.region >= s.stages.size()) return Refuse("a written range names no stage");
  }
  for (std::size_t i = 0; i < s.stages.size(); ++i) {
    const auto of = OfStage(ranges, i);
    if (!of.empty()) s.stages[i].track.Wrote(of);
  }
  return {};
}

Status SlotLifecycle::CheckPositions(std::uint32_t index, std::uint32_t positions) const {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  if (positions > options_.stages.front().layout.context) {
    return Refuse("positions pass the context");
  }
  if (Borrowed(**found) && positions != (*found)->positions)
    return Refuse("borrowed target positions cannot change");
  if ((*found)->restoring) return Refuse("restore must complete first");
  // Spilled (or adopted, not yet read back): no step ran on it; its
  // positions are what its files hold.
  if ((*found)->spilled) return Refuse("a spilled slot's positions are its files'");
  for (const Stage& stage : (*found)->stages) {
    if (stage.progress &&
        (!stage.progress->settled() || stage.progress->completed().positions != positions))
      return Refuse("positions require completed drafter progress publication");
  }
  return {};
}

Status SlotLifecycle::SetPositions(std::uint32_t index, std::uint32_t positions) {
  const std::pair cursor{index, positions};
  return SetPositions(std::span(&cursor, 1));
}

Status SlotLifecycle::SetPositions(
    std::span<const std::pair<std::uint32_t, std::uint32_t>> positions) {
  for (std::size_t i = 0; i < positions.size(); ++i) {
    const auto [index, count] = positions[i];
    if (std::ranges::any_of(positions.first(i),
                            [&](const auto& prior) { return prior.first == index; }))
      return Refuse("cursor publication owners are duplicated");
    if (auto checked = CheckPositions(index, count); !checked) return checked;
  }
  for (const auto [index, count] : positions) slots_[index].positions = count;
  return {};
}

std::expected<StateBorrow, std::string> SlotLifecycle::Borrow(std::uint32_t index,
                                                              std::size_t stage) {
  if (auto active = CheckActive(index); !active) return std::unexpected(active.error());
  Slot& slot = slots_[index];
  if (stage >= slot.stages.size()) return Refuse(std::format("no stage {} to borrow", stage));
  if (slot.restoring) return Refuse("restore must complete before a borrow");
  // Until a verify is accepted and what it owes has run, the state holds
  // rows that may be rejected (a ring's overwritten cells, a recurrent
  // state not yet committed), and the owed work would write it while
  // frozen: settled first.
  if (verify_pending(index) || owed(index)) {
    return Refuse(
        "state cannot be borrowed while a verify awaits its Accept or owes a restore "
        "or commit");
  }
  if (slot.stages[stage].borrows == UINT32_MAX) return Refuse("state borrow count overflows");
  ++slot.stages[stage].borrows;
  StateBorrow borrow;
  borrow.owner_ = this;
  borrow.slot_ = index;
  borrow.stage_ = stage;
  borrow.epoch_ = borrow_epoch_;
  return borrow;
}

std::expected<StateBorrow, std::string> SlotLifecycle::BorrowAfterPrefix(
    std::uint32_t index, std::size_t reader, const DraftProgressLedger::Successor& successor) {
  return ReserveOrderedBorrow(index, reader, successor.replay.cache_first, successor.replay.count,
                              successor);
}

std::expected<StateBorrow, std::string> SlotLifecycle::BorrowAfterOwed(std::uint32_t index,
                                                                       std::size_t reader,
                                                                       std::uint32_t past,
                                                                       std::uint32_t rows) {
  return ReserveOrderedBorrow(index, reader, past, rows, std::nullopt);
}

std::expected<StateBorrow, std::string> SlotLifecycle::ReserveOrderedBorrow(
    std::uint32_t index, std::size_t reader, std::uint32_t past, std::uint32_t rows,
    std::optional<DraftProgressLedger::Successor> successor,
    std::optional<DraftProgressLedger::Unit> compound) {
  if (auto active = CheckActive(index); !active) return std::unexpected(active.error());
  Slot& slot = slots_.at(index);
  if (reader == 0 || reader >= slot.stages.size() || layout(reader).borrowed.empty())
    return Refuse("ordered borrow has no declared target reader");
  if (slot.restoring || verify_pending(index) || !owed(index))
    return Refuse("ordered borrow needs one accepted, unsettled prefix");
  Stage& stage = slot.stages[reader];
  if (stage.ordered_borrow) return Refuse("ordered borrow reader already has a reservation");
  if (successor) {
    if (!stage.progress) return Refuse("successor borrow has no progress ledger");
    if (auto exact = stage.progress->CheckSuccessor(successor->unit, successor->predecessor,
                                                    successor->replay);
        !exact)
      return std::unexpected(exact.error());
  } else if (compound) {
    if (!stage.progress) return Refuse("compound borrow has no owned consumer ledger");
    if (auto exact = stage.progress->CheckCompound(*compound); !exact)
      return std::unexpected(exact.error());
  } else if (stage.progress) {
    return Refuse("repeat-pending reader requires its exact successor reservation");
  }
  if (rows == 0 || past > context(reader) || rows > context(reader) - past)
    return Refuse("ordered borrow reader coordinates exceed its context");
  if (slot.stages[0].borrows == UINT32_MAX || borrow_identity_ == UINT64_MAX)
    return Refuse("state borrow identity or count overflows");
  stage.ordered_borrow = Stage::OrderedBorrow{.successor = successor,
                                              .compound = compound,
                                              .identity = ++borrow_identity_,
                                              .prefix_epoch = slot.accepted_epoch,
                                              .positions = slot.positions,
                                              .past = past,
                                              .rows = rows};
  ++slot.stages[0].borrows;
  StateBorrow borrow;
  borrow.owner_ = this;
  borrow.slot_ = index;
  borrow.stage_ = 0;
  borrow.epoch_ = borrow_epoch_;
  borrow.ordered_reader_ = reader;
  StateBorrowPermit permit;
  permit.owner_ = this;
  permit.slot_ = index;
  permit.reader_ = reader;
  permit.identity_ = stage.ordered_borrow->identity;
  borrow.permit_ = permit;
  return borrow;
}

std::expected<StateBorrow, std::string> SlotLifecycle::BorrowAfterCompoundPrefix(
    std::uint32_t index, std::size_t reader, DraftProgressLedger::Unit unit, std::uint32_t past,
    std::uint32_t rows) {
  return ReserveOrderedBorrow(index, reader, past, rows, std::nullopt, unit);
}

Status SlotLifecycle::CompanionPrefixQueued(std::uint32_t index, std::size_t reader,
                                            DraftProgressLedger::Unit unit) {
  auto* ledger = progress(index, reader);
  if (!ledger || !ledger->CheckCompound(unit))
    return Refuse("compound prefix has no exact owned consumer");
  if (!ledger->CompoundNeedsPrefix(unit)) return {};
  const auto& stage = slots_.at(index).stages.at(reader);
  if (!stage.control_queued || !stage.control_ticket ||
      !ledger->CheckPublication(*stage.control_ticket))
    return Refuse("compound execution has no fully queued control prefix");
  return ledger->CompoundPrefixQueued(unit);
}

Status SlotLifecycle::CheckBorrowPermit(const StateBorrowPermit& permit, std::uint32_t index,
                                        std::size_t reader, std::uint32_t past,
                                        std::uint32_t rows) const {
  if (permit.owner_ != this || permit.slot_ != index || permit.reader_ != reader)
    return Refuse("ordered borrow permit has a different lifecycle, owner or reader");
  if (auto active = CheckActive(index); !active) return active;
  const auto& slot = slots_.at(index);
  if (reader == 0 || reader >= slot.stages.size()) return Refuse("ordered reader has no stage");
  const auto& reservation = slot.stages[reader].ordered_borrow;
  if (!reservation || reservation->active || reservation->identity != permit.identity_ ||
      reservation->prefix_epoch != slot.accepted_epoch ||
      reservation->positions != slot.positions || reservation->past != past ||
      reservation->rows != rows)
    return Refuse("ordered borrow permit differs from its exact accepted prefix and first reader");
  if (reservation->successor)
    return CheckBorrowAfterPrefix(index, reader, reservation->successor->unit,
                                  reservation->successor->predecessor);
  if (reservation->compound) {
    const auto* ledger = slot.stages[reader].progress.get();
    if (!ledger) return Refuse("compound borrow lost its progress ledger");
    return ledger->CheckCompound(*reservation->compound);
  }
  return {};
}

Status SlotLifecycle::CheckBorrowAfterPrefix(std::uint32_t index, std::size_t reader,
                                             DraftProgressLedger::Unit unit,
                                             DraftProgressLedger::Transition predecessor) const {
  if (auto active = CheckActive(index); !active) return active;
  const Slot& slot = slots_.at(index);
  if (reader == 0 || reader >= slot.stages.size()) return Refuse("ordered reader has no stage");
  const Stage& stage = slot.stages[reader];
  if (!stage.ordered_borrow || stage.ordered_borrow->active || !stage.progress ||
      !stage.ordered_borrow->successor || layout(reader).borrowed.empty())
    return Refuse("reader has no exact pending target borrow");
  const auto& successor = *stage.ordered_borrow->successor;
  if (auto current =
          stage.progress->CheckSuccessor(successor.unit, successor.predecessor, successor.replay);
      !current)
    return current;
  return stage.progress->CheckSuccessor(unit, predecessor, successor.replay);
}

void SlotLifecycle::Return(std::uint32_t index, std::size_t stage, std::uint64_t epoch,
                           std::optional<std::size_t> ordered_reader) {
  // A recovery drops all old guards and changes the epoch before reuse.
  if (epoch != borrow_epoch_ || index >= slots_.size() || stage >= slots_[index].stages.size())
    return;
  auto& slot = slots_[index];
  if (ordered_reader) {
    if (*ordered_reader >= slot.stages.size()) return;
    auto& reservation = slot.stages[*ordered_reader].ordered_borrow;
    if (!reservation) return;
    // A returned stack handle cannot revoke an unproven queued reader/prefix.
    // Recovery owns that guard after the slot/ledger is quarantined.
    if (!Usable(slot)) return;
    reservation.reset();
  }
  std::uint32_t& borrows = slot.stages[stage].borrows;
  if (borrows != 0) --borrows;
}

StateBorrow& StateBorrow::operator=(StateBorrow&& other) noexcept {
  if (this != &other) {
    Return();
    owner_ = other.owner_;
    slot_ = other.slot_;
    stage_ = other.stage_;
    epoch_ = other.epoch_;
    ordered_reader_ = std::exchange(other.ordered_reader_, std::nullopt);
    permit_ = std::exchange(other.permit_, std::nullopt);
    other.owner_ = nullptr;
  }
  return *this;
}

void StateBorrow::Return() {
  if (owner_ == nullptr) return;
  owner_->Return(slot_, stage_, epoch_, ordered_reader_);
  owner_ = nullptr;
  ordered_reader_.reset();
  permit_.reset();
}

// ------------------------------------------------------------------ a verify

Status SlotLifecycle::Accept(std::uint32_t index, std::uint32_t keep) {
  auto found = Find(index);
  if (!found) return std::unexpected(found.error());
  Slot& slot = **found;
  const std::uint32_t rows = verify_rows(index);
  if (rows == 0) return Refuse(std::format("slot {} has no verify awaiting its Accept", index));
  if (keep == 0 || keep > rows) {
    return Refuse(std::format("accepting {} rows of a verify of {}", keep, rows));
  }
  // Every verifying stage first: none accepts unless all can.
  for (const Stage& stage : slot.stages) {
    if (auto usable = stage.live->Usable(); !usable) return usable;
    if (stage.live->verify_rows() != 0 && stage.live->verify_rows() != rows) {
      return Refuse(std::format("slot {}'s stages verified different rows", index));
    }
    if (stage.progress) {
      if (!stage.progress_unit) return Refuse("accepted progress has no logical unit");
      if (auto checked = stage.progress->AcceptCandidate(*stage.progress_unit, keep); !checked)
        return std::unexpected(checked.error());
    }
  }
  if (slot.spilled || slot.restoring || !Usable(slot)) {
    return Refuse(std::format("slot {} cannot accept now", index));
  }
  if (std::uint64_t{slot.positions} + keep > options_.stages.front().layout.context) {
    return Refuse("accepted positions pass the context");
  }
  if (Borrowed(slot) || slot.accepted_epoch == UINT64_MAX)
    return Refuse("accepted prefix is guarded or its identity overflows");
  ++slot.accepted_epoch;  // Checked before the first accepted data/control mutation.
  for (Stage& stage : slot.stages) {
    if (stage.live->verify_rows() == 0) continue;
    if (auto accepted = stage.live->Accept(keep); !accepted) {
      // Checked above; a stage that took its Accept and one that did not
      // would leave the slot's stages at different rows.
      Quarantine(slot);
      return accepted;
    }
  }
  for (Stage& stage : slot.stages) {
    if (!stage.progress) continue;
    if (auto accepted = stage.progress->Accept(*stage.progress_unit, keep); !accepted) {
      Quarantine(slot);
      return accepted;
    }
    stage.progress_unit.reset();
  }
  slot.positions += keep;
  return {};
}

Status SlotLifecycle::DiscardVerify(std::uint32_t index) {
  if (auto active = CheckActive(index); !active) return active;
  if (!verify_pending(index)) {
    return Refuse(std::format("slot {} has no verify awaiting its Accept", index));
  }
  for (Stage& stage : slots_[index].stages) {
    // Every saved range owed back, nothing committed.
    if (stage.live->verify_rows() != 0) (void)stage.live->Settle(true, false, false);
  }
  ProgressVerifyFailed(index);
  return Settle(index);
}

Status SlotLifecycle::Settle(std::uint32_t index) {
  return FlushOwed(std::span(&index, 1), "settling a verify");
}

Status SlotLifecycle::FlushOwed(std::span<const std::uint32_t> indices, std::string_view boundary) {
  constexpr auto kOwners = std::numeric_limits<SlotMask>::digits;
  if (indices.size() > kOwners) return Refuse("owed flush exceeds its cohort mask capacity");
  std::array<std::uint32_t, kOwners> pending{}, completed{}, partial{};
  std::size_t pending_count = 0, completed_count = 0, partial_count = 0;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const auto index = indices[i];
    if (std::ranges::find(indices.first(i), index) != indices.first(i).end())
      return Refuse("owed flush owners are duplicated");
    if (auto active = CheckActive(index); !active) return active;
    if (owed(index)) pending[pending_count++] = index;
  }
  if (pending_count == 0) return {};
  const PlanStep step;
  for (const auto index : std::span(pending).first(pending_count))
    if (auto prepared = PrepareOwed(index); !prepared) return prepared;
  std::string failure;
  bool wrote = false;
  JobReport report;
  const auto posted = node_.Job(
      execution_,
      [&](providers::NativeStream native) {
        for (const auto index : std::span(pending).first(pending_count)) {
          const auto queued = QueueOwed(index, native);
          wrote = wrote || queued.wrote;
          if (!queued.result) {
            if (queued.wrote || queued.result.error().unknown) partial[partial_count++] = index;
            failure = queued.result.error().detail;
            return queued.result.error().unknown ? sc::JobResult::kUnknown
                   : wrote                       ? sc::JobResult::kFailed
                                                 : sc::JobResult::kNotStarted;
          }
          completed[completed_count++] = index;
        }
        return sc::JobResult::kQueued;
      },
      boundary, stream_, {}, &report);
  const bool retired = report.retirement == JobReport::Retirement::kRetired;
  const auto retirement = retired ? CopyRetirement::kProven : CopyRetirement::kUnproven;
  const auto published =
      CompleteOwed(std::span(completed).first(completed_count), retired, retirement);
  if (partial_count != 0) {
    const auto owners = std::span(partial).first(partial_count);
    (void)CompleteOwed(owners, false, CopyRetirement::kUnproven);
    QuarantineOwners(owners);
  }
  if (!posted || !failure.empty() || !published) {
    CheckFailedJob();
    if (!failure.empty()) return Error(failure);
    if (!published) return published;
    return posted;
  }
  return {};
}

Status SlotLifecycle::PrepareOwed(std::uint32_t index) {
  if (auto active = CheckActive(index); !active) {
    CheckFailedJob();
    return active;
  }
  Slot& slot = slots_[index];
  for (Stage& stage : slot.stages) {
    if (stage.control_queued) return Refuse("drafter control copy has not retired");
    if (auto prepared = stage.live->PrepareOwed(); !prepared) return prepared;
  }
  for (std::size_t s = 0; s < slot.stages.size(); ++s) {
    Stage& stage = slot.stages[s];
    stage.control_ticket.reset();
    if (!stage.progress || !stage.progress->owed()) continue;
    auto copy = stage.progress->Publication();
    if (!copy) return std::unexpected(copy.error());
    *stage.control_publish = copy->words;
    stage.control_ticket = copy->ticket;
    const Range range{
        .region = s, .offset = stage.control_offset, .bytes = model::kDraftProgressBytes};
    if (auto written = RecordWrite(index, std::span(&range, 1)); !written) return written;
  }
  return {};
}

SlotLifecycle::OwedQueued SlotLifecycle::QueueOwed(std::uint32_t index,
                                                   providers::NativeStream stream) {
  Slot& slot = slots_.at(index);
  OwedQueued out;
  const auto refuse = [&](std::string what, bool unknown = false) {
    out.result =
        std::unexpected(LiveState::QueueFailure{.unknown = unknown, .detail = std::move(what)});
    return out;
  };
  // Every control checked before the first numeric write. Preparation
  // owns the exact publication generation and its distinct pinned words.
  for (const Stage& stage : slot.stages) {
    if (stage.ordered_borrow && !stage.ordered_borrow->active) {
      const auto& reservation = *stage.ordered_borrow;
      if (reservation.prefix_epoch != slot.accepted_epoch ||
          reservation.positions != slot.positions || !owed(index))
        return refuse("ordered target borrow lost its exact accepted prefix");
      if (reservation.successor &&
          (!stage.progress || !stage.progress->CheckSuccessor(reservation.successor->unit,
                                                              reservation.successor->predecessor,
                                                              reservation.successor->replay)))
        return refuse("ordered target borrow lost its exact publication prefix");
      if (reservation.compound &&
          (!stage.progress || !stage.progress->CheckCompound(*reservation.compound)))
        return refuse("ordered target borrow lost its exact compound predecessor");
    }
    if (stage.control_queued) return refuse("drafter control copy has not retired");
    if (stage.progress && stage.progress->owed() && !stage.control_ticket)
      return refuse("drafter control publication was not prepared");
    if (stage.control_ticket &&
        (!stage.progress || !stage.progress->CheckPublication(*stage.control_ticket)))
      return refuse("drafter control publication no longer names its prepared transition");
  }
  for (Stage& stage : slot.stages) {
    if (!stage.live->owed()) continue;
    if (auto queued = stage.live->QueueOwed(stream); !queued) {
      out.result = std::move(queued);
      return out;
    }
    out.wrote = true;
  }
  for (Stage& stage : slot.stages) {
    if (!stage.control_ticket) continue;
    stage.control_queued = true;
    const auto copied = providers::CopyAsync(
        stream, support::Pointer(stage.live->base(0) + stage.control_offset), stage.control_publish,
        model::kDraftProgressBytes, providers::CopyKind::kHostToDevice);
    if (!copied.ok())
      return refuse(std::format("drafter control publication: {}", copied.text()), true);
    out.wrote = true;
  }
  for (auto& stage : slot.stages)
    if (stage.ordered_borrow && !stage.ordered_borrow->active)
      stage.ordered_borrow->prefix_queued = true;
  return out;
}

Status SlotLifecycle::CompleteOwed(std::uint32_t index, bool success, CopyRetirement retirement) {
  return CompleteOwed(std::span(&index, 1), success, retirement);
}

Status SlotLifecycle::CompleteOwed(std::span<const std::uint32_t> indices, bool success,
                                   CopyRetirement retirement) {
  for (std::size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= slots_.size() ||
        std::ranges::find(indices.first(i), indices[i]) != indices.first(i).end())
      return Refuse("owed completion owners are invalid or duplicated");
  }
  bool failed = false;
  for (const auto index : indices) {
    Slot& slot = slots_[index];
    for (Stage& stage : slot.stages) {
      if (stage.ordered_borrow && !stage.ordered_borrow->active) {
        const auto& reservation = *stage.ordered_borrow;
        failed = failed || !success || retirement == CopyRetirement::kUnproven ||
                 !reservation.prefix_queued || reservation.prefix_epoch != slot.accepted_epoch ||
                 reservation.positions != slot.positions;
        if (reservation.successor)
          failed = failed || !stage.control_queued || !stage.progress ||
                   !stage.progress->CheckSuccessor(reservation.successor->unit,
                                                   reservation.successor->predecessor,
                                                   reservation.successor->replay);
        if (reservation.compound)
          failed =
              failed || !stage.progress || !stage.progress->CheckCompound(*reservation.compound) ||
              (stage.progress->CompoundNeedsPrefix(*reservation.compound) && !stage.control_queued);
      }
      if (stage.control_queued) {
        if (retirement == CopyRetirement::kUnproven) node_.KeepPinned(stage.control_publish);
        failed = failed || !success || retirement == CopyRetirement::kUnproven ||
                 !stage.control_ticket || !stage.progress ||
                 !stage.progress->CheckPublication(*stage.control_ticket);
      }
      if (stage.progress_discard)
        failed = failed || !success || retirement == CopyRetirement::kUnproven ||
                 !stage.progress_unit || !stage.progress ||
                 !stage.progress->CheckDiscard(*stage.progress_unit);
    }
  }
  if (failed) {
    // The whole publication set was preflighted without mutation. No earlier
    // owner/consumer may expose a newer mirror when a later ticket differs.
    for (const auto index : indices) Quarantine(slots_[index]);
    return Refuse("owed publication or discard has no matching completed transition");
  }
  for (const auto index : indices) {
    Slot& slot = slots_[index];
    for (Stage& stage : slot.stages) {
      if (!stage.control_queued) continue;
      // All tickets checked above; no callback or device wait occurs between
      // validation and these deterministic host transitions.
      if (!stage.progress->PublicationCompleted(*stage.control_ticket)) {
        for (const auto owner : indices) Quarantine(slots_[owner]);
        return Refuse("drafter publication changed after completion preflight");
      }
      stage.control_queued = false;
      stage.control_ticket.reset();
    }
    for (Stage& stage : slot.stages) {
      if (!stage.progress_discard) continue;
      if (!stage.progress->DiscardCompleted(*stage.progress_unit)) {
        for (const auto owner : indices) Quarantine(slots_[owner]);
        return Refuse("drafter discard changed after completion preflight");
      }
      stage.progress_unit.reset();
      stage.progress_discard = false;
    }
  }
  for (const auto index : indices)
    for (auto& stage : slots_[index].stages)
      if (stage.ordered_borrow) stage.ordered_borrow->active = true;
  return {};
}

namespace {
DraftProgressLedger::CompoundClose TailClose(const SlotLifecycle::CompoundTailOwner& owner,
                                             std::size_t consumer) {
  using Close = DraftProgressLedger::CompoundClose;
  return owner.closing_stage == 0          ? Close::kTarget
         : owner.closing_stage == consumer ? Close::kPost
                                           : Close::kEarlier;
}
}  // namespace

Status SlotLifecycle::PrepareCompoundTail(const CompoundTailOwner& owner) {
  if (auto active = CheckActive(owner.slot); !active) return active;
  auto& slot = slots_.at(owner.slot);
  if (owner.consumers.empty() || owner.closing_stage >= slot.stages.size())
    return Refuse("compound final publication has no consumer or closing stage");
  bool closing_found = owner.closing_stage == 0;
  for (std::size_t i = 0; i < owner.consumers.size(); ++i) {
    const auto& [stage, unit] = owner.consumers[i];
    if (stage == 0 || stage >= slot.stages.size() ||
        std::ranges::any_of(owner.consumers.first(i),
                            [&](const auto& prior) { return prior.first == stage; }))
      return Refuse("compound final publication consumer is invalid or duplicated");
    auto& consumer = slot.stages[stage];
    closing_found |= stage == owner.closing_stage;
    if (!consumer.progress || consumer.tail_attempted || consumer.tail_queued)
      return Refuse("compound final publication conflicts with a live tail");
    if (auto candidate = consumer.progress->CompoundTailCandidate(unit, TailClose(owner, stage));
        !candidate)
      return std::unexpected(candidate.error());
    else if (candidate->tap_first != slot.positions)
      return Refuse("compound final publication no longer continues its original cursor");
  }
  if (!closing_found) return Refuse("compound final publication omits its closing stage");
  for (std::size_t stage = 1; stage < slot.stages.size(); ++stage)
    if (slot.stages[stage].progress &&
        std::ranges::none_of(owner.consumers,
                             [&](const auto& consumer) { return consumer.first == stage; }))
      return Refuse("compound final publication omits a consumer");
  for (const auto& [stage, unit] : owner.consumers) {
    auto& consumer = slot.stages[stage];
    auto prepared = consumer.progress->PrepareCompoundTail(unit, TailClose(owner, stage));
    if (!prepared) return std::unexpected(prepared.error());
    *consumer.control_tail = prepared->words;
    consumer.tail_ticket = prepared->ticket;
    const Range range{
        .region = stage, .offset = consumer.control_offset, .bytes = model::kDraftProgressBytes};
    if (auto tracked = RecordWrite(owner.slot, std::span(&range, 1)); !tracked) return tracked;
  }
  return {};
}

Status SlotLifecycle::QueueCompoundTail(const CompoundTailOwner& owner,
                                        providers::NativeStream native) {
  auto& slot = slots_.at(owner.slot);
  for (const auto& [stage, unit] : owner.consumers) {
    auto& consumer = slot.stages.at(stage);
    if (!consumer.progress || !consumer.tail_ticket || consumer.tail_attempted ||
        !consumer.progress->CheckCompoundTailQueued(unit, *consumer.tail_ticket,
                                                    TailClose(owner, stage)))
      return Refuse("compound final control lost its exact numeric/publication owner");
  }
  for (const auto& [stage, unit] : owner.consumers) {
    (void)unit;
    auto& consumer = slot.stages[stage];
    consumer.tail_attempted = true;
    const auto copied = providers::CopyAsync(
        native, support::Pointer(consumer.live->base(0) + consumer.control_offset),
        consumer.control_tail, model::kDraftProgressBytes, providers::CopyKind::kHostToDevice);
    if (!copied.ok()) return Refuse(std::format("compound final control copy: {}", copied.text()));
    consumer.tail_queued = true;
  }
  return {};
}

Status SlotLifecycle::CompleteCompoundTails(std::span<const CompoundTailOwner> owners,
                                            JobReport::Retirement retirement) {
  if (owners.empty()) return {};
  constexpr auto kOwners = std::numeric_limits<SlotMask>::digits;
  if (owners.size() > kOwners) return Refuse("compound final publication exceeds cohort capacity");
  std::array<std::uint32_t, kOwners> positions{};
  bool attempted = false, queued = true;
  for (std::size_t i = 0; i < owners.size(); ++i) {
    const auto& owner = owners[i];
    if (owner.slot >= slots_.size() || std::ranges::any_of(owners.first(i), [&](const auto& prior) {
          return prior.slot == owner.slot;
        }))
      return Refuse("compound final publication owner is invalid or duplicated");
    for (const auto& [stage, unit] : owner.consumers) {
      (void)unit;
      if (stage == 0 || stage >= slots_[owner.slot].stages.size())
        return Refuse("compound final publication consumer is invalid");
      const auto& consumer = slots_[owner.slot].stages[stage];
      attempted |= consumer.tail_attempted;
      queued &= consumer.tail_queued;
    }
  }
  // A rejected useful program may still retire input/prefix copies. That
  // proof does not turn an entirely unattempted tail into submitted work.
  if (!attempted) return {};
  bool valid = queued && retirement == JobReport::Retirement::kRetired;
  for (std::size_t i = 0; i < owners.size(); ++i) {
    const auto& owner = owners[i];
    const auto& slot = slots_[owner.slot];
    valid = valid && !slot.spilled && !slot.restoring && Usable(slot) && !owner.consumers.empty();
    bool first = true;
    for (const auto& [stage, unit] : owner.consumers) {
      const auto& consumer = slot.stages[stage];
      const bool exact = consumer.progress && consumer.tail_ticket &&
                         consumer.progress->CheckCompoundTailRetired(unit, *consumer.tail_ticket,
                                                                     TailClose(owner, stage));
      valid &= exact;
      if (!exact) continue;
      auto final = consumer.progress->CompoundTailPositions(unit, *consumer.tail_ticket);
      if (!final || *final > options_.stages.front().layout.context ||
          !consumer.progress->CheckCompoundTailCursor(unit, *consumer.tail_ticket,
                                                      slot.positions)) {
        valid = false;
        continue;
      }
      if (first)
        positions[i] = *final;
      else
        valid &= positions[i] == *final;
      first = false;
    }
  }
  if (!valid) {
    for (const auto& owner : owners) {
      auto& slot = slots_[owner.slot];
      for (const auto& [stage, unit] : owner.consumers) {
        (void)unit;
        auto& consumer = slot.stages[stage];
        if (consumer.tail_attempted && (retirement != JobReport::Retirement::kRetired || !queued))
          node_.KeepPinned(consumer.control_tail);
      }
      Quarantine(slot);
    }
    return Refuse("compound final publication lacks complete matching retired work");
  }
  // All selected owners/consumers/cursors passed without mutation. The exact
  // final companion may still hold its target borrow, but its reads retired:
  // publish this authorized cursor with its mirrors before returning that guard.
  for (const auto& owner : owners)
    for (const auto& [stage, unit] : owner.consumers) {
      auto& consumer = slots_[owner.slot].stages[stage];
      if (!consumer.progress->CompleteCompoundTail(unit, *consumer.tail_ticket,
                                                   TailClose(owner, stage))) {
        for (const auto& selected : owners) Quarantine(slots_[selected.slot]);
        return Refuse("compound final publication changed after preflight");
      }
      consumer.tail_ticket.reset();
      consumer.tail_attempted = false;
      consumer.tail_queued = false;
    }
  for (std::size_t i = 0; i < owners.size(); ++i) slots_[owners[i].slot].positions = positions[i];
  return {};
}

}  // namespace llmp::engine
