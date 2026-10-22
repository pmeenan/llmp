// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "memory/reclaim.h"
#include "platform/memory_pressure.h"
#include "runtime/serving.h"
#include "runtime/serving_internal.h"
#include "runtime/serving_support.h"
#include "runtime/swap_room.h"

namespace llmp::runtime {
namespace {
using serving_detail::SwapPhaseTimer;
using serving_support::Error;
using serving_support::kExtent;
using serving_support::Seconds;
// The reclaim order's cost of idle conversation state (Server::Reclaim):
// its write-back and its restore, at the node's rates measured so far, and
// before any at the rates measured on a GB10 (spark-b, DeepSeek's four
// slots' state, 1.42 GB written back at 11.0 GB/s and read again at 14.5
// GB/s; docs/experiments/memory-pressure). Dropping it instead (no spill
// budget) costs its recomputation: its tokens at the model's
// recompute_ms_per_token.
constexpr double kSpillBytesPerSecond = 11.0e9;
constexpr double kRestoreBytesPerSecond = 14.5e9;
// Prepared weight page-in measured on the GB10 (D-055, memory-pressure).
constexpr double kWeightReadBytesPerSecond = 13.3e9;
// A graph's capture and instantiation a GiB counted, measured on a GB10
// (0.33–0.45 s in the services' logs): what a graph's capture weighs
// against before any graph is held (Server::Reclaim).
constexpr double kGraphSecondsPerGiB = 0.36;
std::uint64_t WeightHandle(catalog::ExtentId extent) {
  return (static_cast<std::uint64_t>(extent.generation()) << 32U) | extent.index();
}

catalog::ExtentId WeightExtent(std::uint64_t handle) {
  return {static_cast<std::uint32_t>(handle), static_cast<std::uint32_t>(handle >> 32U)};
}

}  // namespace
std::uint64_t Server::SpilledBytes(SpillView view) {
  std::uint64_t bytes = 0;
  for (const auto& m : models_) {
    if (!m->llm()) {
      continue;
    }
    auto& l = static_cast<Llm&>(*m);
    // A model a swap wrote back holds all its state on disk until it returns.
    const bool written_back =
        m.get() != view.in &&
        (m.get() == view.out || std::ranges::find(spilled_, m.get()) != spilled_.end());
    for (std::size_t i = 0; i < l.branches(); ++i) {
      if (auto b = l.branch(i); b) {
        // Otherwise its spilled conversations, and the files restored ones
        // keep for their next (incremental) spill: not the incoming model's
        // (its conversations stop counting once it is resident), nor those
        // of the one about to spill, whose whole state the caller counts.
        const bool retained = m.get() != view.in && *b != view.spilling;
        bytes += written_back
                     ? l.StateBytes(**b)
                     : l.SpilledStateBytes(**b) + (retained ? l.RetainedSpillBytes(**b) : 0);
      }
    }
  }
  return bytes;
}

void Server::ForEachDeletableSpill(
    SpillView view, std::optional<Clock::time_point> older_than,
    const std::function<void(Llm& model, Llm::Branch& branch, std::uint64_t counted,
                             bool retained)>& each) {
  for (const auto& m : models_) {
    if (!m->llm() || m.get() == view.in) {
      continue;  // the incoming model's stop counting once it is resident
    }
    auto& l = static_cast<Llm&>(*m);
    const bool written_back =
        m.get() == view.out || std::ranges::find(spilled_, m.get()) != spilled_.end();
    for (std::size_t i = 0; i < l.branches(); ++i) {
      auto b = l.branch(i);
      if (!b || !l.BranchIdle(**b) || (*b)->HeldByContinuation() ||
          (older_than && l.LastUsed(**b) >= *older_than)) {
        continue;
      }
      const std::uint64_t counted = written_back ? l.StateBytes(**b) : l.SpilledStateBytes(**b);
      if (counted != 0) {
        each(l, **b, counted, false);
      } else if (const std::uint64_t retained =
                     written_back || *b == view.spilling ? 0 : l.RetainedSpillBytes(**b);
                 retained != 0) {
        each(l, **b, retained, true);
      }
    }
  }
}

bool Server::SpillBudgetFits(std::uint64_t extra, SpillView view,
                             std::optional<Clock::time_point> older_than) {
  const std::uint64_t held = SpilledBytes(view);
  std::uint64_t deletable = 0;
  ForEachDeletableSpill(view, older_than, [&](Llm&, Llm::Branch&, std::uint64_t counted, bool) {
    deletable += counted;
  });
  const std::uint64_t left = held - std::min(held, deletable);
  // Eligible checkpoints may go before spilled conversations, but those
  // a request holds cannot: they consume disk at either admission point.
  return TurnCheckpointFits(spill_budget_, left, TurnCheckpointBytes(true), extra);
}

std::uint64_t Server::TurnCheckpointBytes(bool held_only) {
  std::uint64_t checkpoints = 0;
  for (const auto& m : models_) {
    if (m->llm()) {
      auto& l = static_cast<Llm&>(*m);
      for (std::size_t i = 0; i < l.branches(); ++i) {
        if (auto b = l.branch(i); b && (!held_only || !l.OldestTurnCheckpoint(**b))) {
          const auto bytes = (*b)->turn_checkpoint_bytes();
          if (bytes > std::numeric_limits<std::uint64_t>::max() - checkpoints)
            return std::numeric_limits<std::uint64_t>::max();
          checkpoints += bytes;
        }
      }
    }
  }
  return checkpoints;
}

bool Server::TrimTurnCheckpoints(std::uint64_t extra, SpillView view) {
  std::size_t dropped = 0;
  std::uint64_t dropped_bytes = 0;
  const auto drop = [&]() {
    // The oldest checkpoint of any idle conversation no request holds.
    Llm* model = nullptr;
    Llm::Branch* branch = nullptr;
    std::optional<Clock::time_point> oldest;
    for (const auto& m : models_) {
      if (!m->llm()) {
        continue;
      }
      auto& l = static_cast<Llm&>(*m);
      for (std::size_t i = 0; i < l.branches(); ++i) {
        auto b = l.branch(i);
        if (!b) {
          continue;
        }
        if (const auto at = l.OldestTurnCheckpoint(**b); at && (!oldest || *at < *oldest)) {
          oldest = at;
          model = &l;
          branch = *b;
        }
      }
    }
    if (branch == nullptr) {
      return false;  // what is left, requests hold
    }
    const std::uint64_t bytes = model->DropOldestTurnCheckpoint(*branch);
    if (bytes == 0) {
      return false;
    }
    ++dropped;
    dropped_bytes += bytes;
    return true;
  };
  const bool fits = TrimTurnCheckpointBudget(
      spill_budget_, extra, [this, view] { return SpilledBytes(view); },
      [this] { return TurnCheckpointBytes(); }, drop);
  if (dropped != 0) {
    Log(std::format(
        "deleted {} turn checkpoints ({:.1f} MiB), the oldest, past the spill budget of {} GiB "
        "beside the spilled conversations",
        dropped, static_cast<double>(dropped_bytes) / (1U << 20U), spill_budget_ >> 30U));
  }
  return fits;
}

bool Server::KeepWithinSpillBudget(std::uint64_t extra, SpillView view,
                                   std::optional<Clock::time_point> older_than) {
  // Nothing deleted for a conversation (or a swap) that would not fit even
  // with every deletable one gone.
  return SpillBudgetFits(extra, view, older_than) && DeleteSpilledPast(extra, view, older_than);
}

bool Server::DeleteSpilledPast(std::uint64_t extra, SpillView view,
                               std::optional<Clock::time_point> older_than) {
  // Optional caches yield first, before deleting any conversation; held
  // checkpoints remain charged and may leave too little room for a spill.
  (void)TrimTurnCheckpoints(extra, view);
  std::size_t deleted = 0;
  std::uint64_t deleted_bytes = 0;
  bool fits = false;
  std::vector<const Llm::Branch*> tried;  // retained files released, or refused
  for (;;) {
    const std::uint64_t held = SpilledBytes(view);
    if (TurnCheckpointFits(spill_budget_, held, TurnCheckpointBytes(), extra)) {
      fits = true;
      break;
    }
    // The least recently used spilled conversation no request holds; first
    // the files restored conversations kept, which cost only their next
    // spill's whole write.
    Llm* oldest = nullptr;
    Llm::Branch* branch = nullptr;
    bool file_only = false;
    ForEachDeletableSpill(
        view, older_than, [&](Llm& l, Llm::Branch& b, std::uint64_t, bool retained) {
          if (retained && std::ranges::find(tried, &b) != tried.end()) {
            return;  // released (or refused) once already
          }
          if (branch == nullptr || (retained && !file_only) ||
              (retained == file_only && l.LastUsed(b) < oldest->LastUsed(*branch))) {
            oldest = &l;
            branch = &b;
            file_only = retained;
          }
        });
    if (branch == nullptr) {
      break;  // nothing more to delete: still short
    }
    if (file_only) {
      // Once each: one refused (its state not wholly resident) stays counted.
      tried.push_back(branch);
      if (auto released = oldest->ReleaseRetainedSpill(*branch); !released) {
        Log(std::format("{}'s retained spill file could not be released: {}", oldest->name(),
                        released.error()));
      }
      continue;
    }
    const std::uint64_t bytes = oldest->StateBytes(*branch);
    if (auto released = branch->ReleaseIdleState(); !released) {
      Log(std::format("{}'s spilled conversation could not be deleted: {}", oldest->name(),
                      released.error()));
      break;
    }
    ++deleted;
    deleted_bytes += bytes;
  }
  if (deleted != 0) {
    Log(std::format(
        "deleted {} spilled conversations ({:.1f} MiB), the least recently used, "
        "past the spill budget of {} GiB",
        deleted, static_cast<double>(deleted_bytes) / (1U << 20U), spill_budget_ >> 30U));
  }
  return fits;
}

std::optional<double> Server::SpillRoomSeconds(std::uint64_t extra, SpillView view) {
  // DeleteSpilledPast's victims, without deleting; every eligible turn
  // checkpoint is trimmed before any of them (only held ones count).
  std::vector<SpillVictim> victims;
  ForEachDeletableSpill(
      view, std::nullopt, [&](Llm& l, Llm::Branch& b, std::uint64_t counted, bool retained) {
        victims.push_back({.conversation = !retained,
                           .used = l.LastUsed(b),
                           .counted = counted,
                           .seconds = retained ? 0.0
                                               : static_cast<double>(b.history().size()) *
                                                     l.settings().recompute_ms_per_token.value /
                                                     1000.0});
      });
  return SpillRoomCost(std::move(victims), SpilledBytes(view), TurnCheckpointBytes(true),
                       spill_budget_, extra);
}

bool Server::RoomFor(std::uint64_t needed, std::string_view why, const Llm::Branch* spare) {
  if (needed == 0 || !started_ || torn_down_) {
    return true;
  }
  auto free = node_.FreeBytes();
  return runtime::RoomFor(
      needed, free ? std::optional(*free) : std::nullopt, [&](std::uint64_t shortfall) {
        return Reclaim(shortfall, true, why, nullptr, std::nullopt, false, spare);
      });
}

// One driver-owned reclaim attempt. Gather refreshes eligibility on each round;
// taking a candidate waits outside node.Call and credits actual retirement.
// Measured transfer rates stay fixed while per-kind accounting accumulates
// across rounds. A pending weight selection is credited only by its swap.
struct Server::Reclamation {
  Server& server;
  std::uint64_t needed;
  bool states;
  std::string_view why;
  const Served* running;
  std::optional<memory::ReclaimKind> below_kind;
  bool partial;
  const Llm::Branch* spare;
  std::uint64_t token_incoming;
  PendingWeightEvictions* pending;
  bool allow_weights;
  bool diagnostic;
  std::set<catalog::ExtentId> protected_ids;
  std::uint64_t kept = 0;
  double spill_rate = kSpillBytesPerSecond;
  double restore_rate = kRestoreBytesPerSecond;
  double weight_rate = kWeightReadBytesPerSecond;
  std::array<std::uint64_t, memory::kReclaimKinds> count{};
  std::array<std::uint64_t, memory::kReclaimKinds> freed_by{};
  std::array<double, memory::kReclaimKinds> cost{};
  std::size_t dropped = 0;
  bool took = false;
  Llm* idle_owner = nullptr;
  struct HistoryVictim {
    Llm* model;
    Llm::Branch* branch;
    std::uint32_t owner;
    std::uint64_t slot;
  };
  std::vector<HistoryVictim> histories{};
  std::vector<TokenReclaimGroup> history_groups{};
  struct WeightView {
    std::uint32_t owner = 0;
    std::uint64_t generation = 0;
    std::uint64_t bytes = 0;
  };
  std::map<catalog::ExtentId, WeightView> weight_views{};

  static std::size_t KindIndex(memory::ReclaimKind kind) { return static_cast<std::size_t>(kind); }

  void ReleaseEmptyBacking(std::span<const catalog::ExtentId> protect) {
    // Empty kept state goes before restore-cost selection, with the same
    // activation protection as all later candidates.
    protected_ids.insert(server.activation_protect_.begin(), server.activation_protect_.end());
    protected_ids.insert(protect.begin(), protect.end());
    const std::vector<catalog::ExtentId> extents(protected_ids.begin(), protected_ids.end());
    kept = server.ReleaseKept(needed, running, extents);
  }

  void MeasureRates() {
    // Idle state's cost at the rates measured so far (once a GiB has moved).
    {
      Llm::SpillStats all;
      for (const auto& m : server.models_) {
        if (m->llm()) {
          const Llm::SpillStats& s = static_cast<Llm&>(*m).spill_stats();
          all.spilled_bytes += s.spilled_bytes;
          all.spill_seconds += s.spill_seconds;
          all.restored_bytes += s.restored_bytes;
          all.restore_seconds += s.restore_seconds;
        }
      }
      constexpr std::uint64_t kEnough = std::uint64_t{1} << 30U;
      if (all.spilled_bytes >= kEnough && all.spill_seconds > 0) {
        spill_rate = static_cast<double>(all.spilled_bytes) / all.spill_seconds;
      }
      if (all.restored_bytes >= kEnough && all.restore_seconds > 0) {
        restore_rate = static_cast<double>(all.restored_bytes) / all.restore_seconds;
      }
    }
    weight_rate =
        server.page_read_bytes_ >= (std::uint64_t{1} << 30U) && server.page_read_seconds_ > 0
            ? static_cast<double>(server.page_read_bytes_) / server.page_read_seconds_
            : kWeightReadBytesPerSecond;
  }

  void Gather(std::vector<memory::ReclaimCandidate>& candidates, double& below) {
    const SwapPhaseTimer timer(diagnostic ? &server.swap_host_phases_.gather_seconds : nullptr);
    idle_owner = nullptr;
    histories.clear();
    history_groups.clear();
    weight_views.clear();
    std::uint32_t resident_index = 0;
    for (std::uint32_t i = 0; i < server.models_.size(); ++i) {
      server.models_[i]->ReclaimCandidates(i, server.models_[i].get() == running, candidates);
      if (states && server.models_[i]->llm()) {
        auto& model = static_cast<Llm&>(*server.models_[i]);
        for (std::size_t slot = 0; slot < model.branches(); ++slot) {
          auto branch = model.branch(slot);
          if (!branch || *branch == spare || !model.HistoryReclaimable(**branch)) {
            continue;
          }
          const auto bytes = model.IdleHistoryBytes(**branch);
          if (bytes == 0) {
            continue;
          }
          histories.push_back({.model = &model, .branch = *branch, .owner = i, .slot = slot});
        }
      }
      if (server.models_[i].get() == server.resident_) {
        resident_index = i;
      }
    }
    if (server.options_.partial_weight_eviction && allow_weights) {
      std::set<catalog::ExtentId> pending_ids;
      if (pending != nullptr) {
        pending_ids = pending->rejected;
        for (const auto& selected : pending->selected) pending_ids.insert(selected.extent);
      }
      std::vector<std::vector<catalog::ExtentId>> weights;
      weights.reserve(server.models_.size());
      for (const auto& model : server.models_) {
        weights.push_back(model.get() == running ? std::vector<catalog::ExtentId>{}
                                                 : model->weights());
      }
      const auto listed = server.node_.Call(
          [&]() -> Status {
            for (std::uint32_t i = 0; i < weights.size(); ++i) {
              for (const auto extent : weights[i]) {
                if (protected_ids.contains(extent) || pending_ids.contains(extent)) continue;
                const auto view = server.node_.catalog().Describe(extent);
                const auto* source = server.node_.scheduler().SourceOf(extent);
                if (!view || !catalog::Catalog::Evictable(*view) || source == nullptr ||
                    view->descriptor.memory_class != catalog::MemoryClass::kWeights ||
                    view->descriptor.recovery != catalog::Recovery::kFromArtifact)
                  continue;
                if (!weight_views
                         .try_emplace(extent, WeightView{.owner = i,
                                                         .generation = view->content_generation,
                                                         .bytes = view->descriptor.size.value()})
                         .second) {
                  return Error("a prepared weight extent has more than one owner");
                }
                candidates.push_back(
                    {.kind = memory::ReclaimKind::kIdleWeights,
                     .owner = i,
                     .id = WeightHandle(extent),
                     .bytes = view->descriptor.size.value(),
                     .last_use = view->last_use,
                     .restore_seconds = static_cast<double>(source->read.length) / weight_rate});
                memory::SetUse(candidates.back(), server.models_[i]->weight_use());
              }
            }
            return {};
          },
          "gathering inactive weight reclaim candidates");
      if (!listed) {
        server.Log(std::format("listing inactive weights: {}", listed.error()));
        std::erase_if(candidates,
                      [](const auto& c) { return c.kind == memory::ReclaimKind::kIdleWeights; });
        weight_views.clear();
      }
    }
    std::ranges::sort(histories, [](const HistoryVictim& a, const HistoryVictim& b) {
      return a.model->LastUsed(*a.branch) < b.model->LastUsed(*b.branch);
    });
    std::vector<std::uint64_t> capacities;
    capacities.reserve(histories.size());
    for (const auto& history : histories) {
      capacities.push_back(history.model->IdleHistoryBytes(*history.branch));
    }
    history_groups = GroupTokenReclaim(capacities, server.token_memory_.used() + token_incoming,
                                       kExtent, needed);
    std::size_t first = 0;
    for (const auto& group : history_groups) {
      const auto& oldest = histories[first];
      double seconds = 0;
      bool group_running = false;
      for (std::size_t i = first; i < group.end; ++i) {
        const auto& h = histories[i];
        seconds += static_cast<double>(h.branch->history().size()) *
                   h.model->settings().recompute_ms_per_token.value / 1000.0;
        group_running = group_running || h.model == running;
      }
      candidates.push_back({.kind = memory::ReclaimKind::kTokenHistory,
                            .owner = oldest.owner,
                            .id = oldest.slot,
                            .bytes = group.catalog_bytes,
                            .last_use = static_cast<std::uint64_t>(
                                oldest.model->LastUsed(*oldest.branch).time_since_epoch().count()),
                            .restore_seconds = seconds,
                            .running = group_running});
      memory::SetUse(candidates.back(), oldest.model->LastStamp(*oldest.branch));
      first = group.end;
    }
    // The running model's next step finds what its last ones used.
    memory::ProtectFloor(candidates, running != nullptr ? running->plan_floor_bytes() : 0);
    if (states && server.resident_ != nullptr && server.resident_->llm()) {
      idle_owner = static_cast<Llm*>(server.resident_);
      AddIdleStateCandidates(*idle_owner, resident_index, server.resident_ == running, spare,
                             {.spill_rate = spill_rate,
                              .restore_rate = restore_rate,
                              .spill_budget = server.spill_budget_,
                              .spill_room =
                                  [this](const Llm::Branch& branch, std::uint64_t bytes) {
                                    return server.SpillRoomSeconds(bytes, {.spilling = &branch});
                                  }},
                             candidates);
    }
    cost = memory::KindCosts(candidates);
    if (below_kind) {
      // What the charge would cost to restore a GiB: its kind's measured
      // cost, or the GB10's when none is held.
      double charged = cost[KindIndex(*below_kind)];
      if (charged == 0) {
        charged = *below_kind == memory::ReclaimKind::kGraph ? kGraphSecondsPerGiB : 0;
      }
      below = memory::ReclaimInflation() + charged;
    }
  }

  std::optional<std::uint64_t> Occupancy() {
    std::optional<std::uint64_t> total;
    if (!server.node_.Call(
            [&]() -> Status {
              total = server.node_.catalog().OccupancyOf(server.node_.domain()).Total().value();
              return {};
            },
            "sizing a reclaim's catalog charge")) {
      return std::nullopt;
    }
    return total;
  }

  std::uint64_t Take(const memory::ReclaimCandidate& c) {
    const SwapPhaseTimer timer(diagnostic ? &server.swap_host_phases_.take_seconds : nullptr);
    std::uint64_t got = 0;
    if (c.kind == memory::ReclaimKind::kTokenHistory) {
      std::size_t first = 0;
      const auto prospective = [&]() {
        const auto target = server.token_memory_.used() + token_incoming;
        return (target / kExtent + (target % kExtent != 0 ? 1U : 0U)) * kExtent;
      };
      const auto before = prospective();
      const auto charged_before = server.node_.token_charged();
      const auto free_before = server.node_.FreeBytes();
      for (const auto& group : history_groups) {
        const auto& oldest = histories[first];
        if (oldest.owner == c.owner && oldest.slot == c.id) {
          for (std::size_t i = first; i < group.end; ++i) {
            auto& h = histories[i];
            if (h.branch != spare && h.model->HistoryReclaimable(*h.branch)) {
              std::ignore = h.branch->ReleaseIdleState();
            }
          }
          break;
        }
        first = group.end;
      }
      const auto after = prospective();
      const auto free_after = server.node_.FreeBytes();
      const auto actual_tokens =
          charged_before - std::min(charged_before, server.node_.token_charged());
      // Dropping a history can also release its resident native state. Count
      // the measured catalog gain once, including overlapping idle victims.
      got = free_before && free_after && *free_after >= *free_before ? *free_after - *free_before
                                                                     : actual_tokens;
      if (token_incoming != 0) {
        const auto target_drop = before - std::min(before, after);
        // The incoming allocation can change which rounded boundary matters.
        // Replace actual token release with the prospective target decrease.
        got = TokenReclaimCredit(got, actual_tokens, target_drop);
      }
    } else if (c.kind == memory::ReclaimKind::kIdleState) {
      if (idle_owner == nullptr) {
        return 0;  // only the resident model's idle state is a candidate
      }
      auto b = idle_owner->branch(c.id);
      if (!b) {
        return 0;
      }
      // Spill, not clear: within the spill budget (the least recently used
      // spilled state deleted first; once spilled its whole state counts);
      // past it, or with none, dropped. A state larger than the whole
      // budget deletes nothing else first.
      const std::uint64_t held = idle_owner->ResidentStateBytes(**b);
      if (held == 0) {
        return 0;  // gone meanwhile: nothing to give back
      }
      if (held <= server.spill_budget_ && server.KeepWithinSpillBudget(held, {.spilling = *b})) {
        const auto before = Occupancy();
        if (auto spilled = idle_owner->SpillIdle(**b); spilled) {
          got = c.bytes;
        } else {
          server.Log(std::format("{}'s idle conversation in slot {}: {}", idle_owner->name(), c.id,
                                 spilled.error()));
          // A spill that failed part way counts as spilled (its slot is out
          // of every closure) though some of it stays resident: what it
          // actually freed, never its whole state.
          const auto after = Occupancy();
          got = before && after ? *before - std::min(*before, *after) : 0;
        }
      } else if (auto released = (*b)->ReleaseIdleState(); released) {
        got = c.bytes;
        ++dropped;
      }
    } else {
      std::optional<std::uint64_t> before_charge;
      if (server.options_.partial_weight_eviction) {
        before_charge = Occupancy();
      }
      got = server.models_[c.owner]->Reclaim(c.kind, c.id);
      if (before_charge) {
        const auto after = Occupancy();
        got = after ? *before_charge - std::min(*before_charge, *after) : 0;
      }
    }
    if (got != 0) {
      took = true;
      ++count[KindIndex(c.kind)];
      freed_by[KindIndex(c.kind)] += got;
    }
    return got;
  }

  void TakeWeights(std::span<const memory::ReclaimCandidate> victims,
                   std::span<std::uint64_t> freed) {
    const SwapPhaseTimer timer(diagnostic ? &server.swap_host_phases_.take_seconds : nullptr);
    std::vector<catalog::ExtentId> eligible;
    const auto listed = server.node_.Call(
        [&]() -> Status {
          for (const auto& c : victims) {
            const auto extent = WeightExtent(c.id);
            const auto known = weight_views.find(extent);
            const auto view = server.node_.catalog().Describe(extent);
            if (known != weight_views.end() && known->second.owner == c.owner && view &&
                view->content_generation == known->second.generation &&
                view->descriptor.size.value() == c.bytes && catalog::Catalog::Evictable(*view) &&
                !protected_ids.contains(extent))
              eligible.push_back(extent);
          }
          return {};
        },
        "rechecking selected idle weights");
    if (!listed || eligible.empty()) return;
    const auto evicted = server.node_.Evict(eligible);
    if (!evicted) server.Log(std::format("reclaiming idle weights: {}", evicted.error()));
    const std::set<catalog::ExtentId> attempted(eligible.begin(), eligible.end());
    const auto reconciled = server.node_.Call(
        [&]() -> Status {
          for (std::size_t i = 0; i < victims.size(); ++i) {
            const auto extent = WeightExtent(victims[i].id);
            const auto view = server.node_.catalog().Describe(extent);
            if (attempted.contains(extent) && view &&
                view->state == catalog::ExtentState::kNonresident)
              freed[i] = victims[i].bytes;
          }
          return {};
        },
        "reconciling completed idle weight releases");
    if (!reconciled) return;
    for (const auto bytes : freed) {
      if (bytes != 0) {
        took = true;
        ++count[KindIndex(memory::ReclaimKind::kIdleWeights)];
        freed_by[KindIndex(memory::ReclaimKind::kIdleWeights)] += bytes;
      }
    }
  }

  memory::ReclaimRun SelectAndTake() {
    memory::ReclaimRun run;
    // All of it or nothing (memory::RunReclaim): a selection that cannot
    // cover what is needed reclaims nothing (the caller waits or refuses);
    // another round, without it, only when a victim gave back less than its
    // count (held, or gone meanwhile), and what the rounds before took then
    // stays taken even when the rest cannot be covered.
    if (pending == nullptr) {
      const auto begin = diagnostic ? Clock::now() : Clock::time_point{};
      const auto gather_before = server.swap_host_phases_.gather_seconds;
      const auto take_before = server.swap_host_phases_.take_seconds;
      run = memory::RunReclaim(
          needed - kept, partial,
          [this](auto& candidates, double& below) { Gather(candidates, below); },
          [this](const auto& candidate) { return Take(candidate); },
          [this](auto victims, auto freed) { TakeWeights(victims, freed); });
      if (diagnostic) {
        server.swap_host_phases_.selection_bookkeeping_seconds +=
            std::max(0.0, Seconds(Clock::now() - begin) -
                              (server.swap_host_phases_.gather_seconds - gather_before) -
                              (server.swap_host_phases_.take_seconds - take_before));
      }
    } else {
      std::vector<memory::ReclaimCandidate> candidates;
      double below = std::numeric_limits<double>::infinity();
      Gather(candidates, below);
      memory::ReclaimPlan plan;
      {
        const SwapPhaseTimer timer(
            diagnostic ? &server.swap_host_phases_.selection_bookkeeping_seconds : nullptr);
        plan = memory::SelectReclaim(candidates, needed - kept, below);
      }
      if (!plan.sufficient) {
        run.short_of_need = true;
      } else {
        std::uint64_t pending_credit = 0;
        for (std::size_t i = 0;
             i < plan.victims.size() && run.freed + pending_credit < needed - kept; ++i) {
          const auto& c = candidates[plan.victims[i]];
          if (c.kind == memory::ReclaimKind::kIdleWeights) {
            const auto extent = WeightExtent(c.id);
            const auto known = weight_views.find(extent);
            if (known != weight_views.end()) {
              pending->selected.push_back({.extent = extent,
                                           .generation = known->second.generation,
                                           .candidate = c,
                                           .priority = plan.priorities[i]});
              pending_credit += c.bytes;
            }
          } else {
            const auto got = Take(c);
            if (got != 0) {
              memory::RaiseReclaimInflation(plan.priorities[i]);
              run.freed += got;
            }
          }
        }
      }
    }
    return run;
  }

  std::uint64_t Finish(const memory::ReclaimRun& run) {
    const std::uint64_t freed = run.freed + kept;
    if (freed == 0 && run.short_of_need) {
      ++server.reclaims_short_;
    } else if (pending == nullptr && !partial && freed != 0 && freed < needed) {
      ++server.reclaims_cut_short_;
    }
    if (freed != 0) {
      const SwapPhaseTimer timer(diagnostic ? &server.swap_host_phases_.heap_release_seconds
                                            : nullptr);
      if (diagnostic) ++server.swap_host_phases_.heap_release_calls;
      platform::ReleaseFreeHeap();
    }
    if (took) {
      const auto mib = [](std::uint64_t b) { return static_cast<double>(b) / (1U << 20U); };
      using K = memory::ReclaimKind;
      // Counts, bytes and measured costs only (D-014).
      server.Log(std::format(
          "reclaimed {:.1f} MiB of {:.1f} MiB needed for {}: {} graphs ({:.1f} MiB), {} plans "
          "({:.1f} MiB), {} idle weights ({:.1f} MiB), {} idle conversations {} ({:.1f} MiB); "
          "measured s a GiB: graphs {:.2f}, "
          "plans {:.2f}, idle state {:.2f}; {} history groups discarded ({:.1f} MiB capacity)",
          mib(freed), mib(needed), why, count[KindIndex(K::kGraph)],
          mib(freed_by[KindIndex(K::kGraph)]), count[KindIndex(K::kPlan)],
          mib(freed_by[KindIndex(K::kPlan)]), count[KindIndex(K::kIdleWeights)],
          mib(freed_by[KindIndex(K::kIdleWeights)]), count[KindIndex(K::kIdleState)],
          dropped != 0 ? "dropped" : "spilled", mib(freed_by[KindIndex(K::kIdleState)]),
          cost[KindIndex(K::kGraph)], cost[KindIndex(K::kPlan)], cost[KindIndex(K::kIdleState)],
          count[KindIndex(K::kTokenHistory)], mib(freed_by[KindIndex(K::kTokenHistory)])));
    }
    return freed;
  }
};

std::uint64_t Server::Reclaim(std::uint64_t needed, bool states, std::string_view why,
                              const Served* running, std::optional<memory::ReclaimKind> below_kind,
                              bool partial, const Llm::Branch* spare, std::uint64_t token_incoming,
                              std::span<const catalog::ExtentId> protect,
                              PendingWeightEvictions* pending, bool allow_weights) {
  if (reclaiming_ || !started_ || torn_down_ || needed == 0) {
    return 0;
  }
  reclaiming_ = true;
  struct ReclaimGuard {
    bool& active;
    ~ReclaimGuard() { active = false; }
  } guard{reclaiming_};
  const bool diagnostic = options_.diagnostic_budget_cap_bytes.has_value();
  if (diagnostic) ++swap_host_phases_.reclaim_calls;
  Reclamation reclaim{.server = *this,
                      .needed = needed,
                      .states = states,
                      .why = why,
                      .running = running != nullptr ? running : resident_,
                      .below_kind = below_kind,
                      .partial = partial,
                      .spare = spare,
                      .token_incoming = token_incoming,
                      .pending = pending,
                      .allow_weights = allow_weights,
                      .diagnostic = diagnostic,
                      .protected_ids = {}};
  reclaim.ReleaseEmptyBacking(protect);
  if (reclaim.kept >= needed) return reclaim.kept;
  reclaim.MeasureRates();
  const auto run = reclaim.SelectAndTake();
  return reclaim.Finish(run);
}

Status Server::MakeRoomForSwap(Served& m, std::span<const catalog::ExtentId> out,
                               PendingWeightEvictions* pending) {
  const SwapPhaseTimer timer(options_.diagnostic_budget_cap_bytes ? &swap_host_phases_.room_seconds
                                                                  : nullptr);
  if (budget_ < configured_budget_) {
    const auto pressure = platform::ReadMemoryPressure();
    if (pressure.available.has_value() &&
        *pressure.available >= pressure_.limits().headroom_bytes) {
      node_.SetBudget(base::Bytes(configured_budget_));
      budget_ = configured_budget_;
    }
  }
  // Size physical occupancy afresh, including required charges above B.
  const auto add = [](std::uint64_t& value, std::uint64_t bytes) {
    if (bytes > UINT64_MAX - value) return false;
    value += bytes;
    return true;
  };
  // Round after round while each makes progress (RoomProgress): the bytes
  // still short shrink every round, so this ends.
  RoomProgress progress;
  for (;;) {
    std::uint64_t need = 0;
    std::uint64_t credit = 0;
    auto sized = node_.Call(
        [&]() -> Status {
          const auto& catalog = node_.catalog();
          std::uint64_t after = catalog.OccupancyOf(node_.domain()).Total().value();
          for (const auto& [extent, generation] : m.everything().extents) {
            const auto view = catalog.Describe(extent);
            if (!view || view->descriptor.domain != node_.domain() ||
                view->content_generation != generation || view->discarded ||
                view->state == catalog::ExtentState::kQuarantined)
              return Error("a swap's incoming closure is stale or invalid");
            if (view->state != catalog::ExtentState::kResident &&
                !add(after, view->descriptor.size.value()))
              return Error("a swap's incoming occupancy overflows");
          }
          std::set<catalog::ExtentId> counted;
          for (const auto extent : out) {
            const auto view = catalog.Describe(extent);
            if (counted.insert(extent).second && view &&
                view->state == catalog::ExtentState::kResident &&
                !add(credit, view->descriptor.size.value()))
              return Error("a swap's outgoing credit overflows");
          }
          need = after > node_.budget().value() ? after - node_.budget().value() : 0;
          if (pending != nullptr) {
            std::erase_if(pending->selected, [&](const auto& selected) {
              const auto view = catalog.Describe(selected.extent);
              const bool valid = view && view->content_generation == selected.generation &&
                                 view->descriptor.size.value() == selected.candidate.bytes &&
                                 catalog::Catalog::Evictable(*view) &&
                                 !counted.contains(selected.extent);
              if (!valid) pending->rejected.insert(selected.extent);
              return !valid;
            });
            // Keep only the policy-ordered prefix still needed after actual
            // cache-charge rounding. A single final extent may overshoot.
            std::size_t keep = 0;
            while (keep < pending->selected.size() && credit < need) {
              const auto& selected = pending->selected[keep++];
              if (!add(credit, selected.candidate.bytes))
                return Error("a swap's selected weight credit overflows");
            }
            pending->selected.resize(keep);
          }
          return {};
        },
        "sizing actual missing incoming and selected swap extents");
    if (!sized) return sized;
    if (credit >= need) return {};
    if (!progress.Shrank(need - credit)) break;
    const auto selected_before = pending == nullptr ? 0 : pending->selected.size();
    const auto freed = Reclaim(need - credit, false, "a swap's incoming model", &m, std::nullopt,
                               false, nullptr, 0, {}, pending);
    if (freed == 0 && (pending == nullptr || pending->selected.size() == selected_before)) break;
  }
  return Error(std::format("{} to {}: the incoming closure cannot fit without held extents",
                           resident_ != nullptr ? resident_->name() : "a swap", m.name()));
}

bool Server::AnyResidentState() {
  for (const auto& m : models_) {
    if (m->llm()) {
      auto& l = static_cast<Llm&>(*m);
      for (std::size_t i = 0; i < l.branches(); ++i) {
        if (auto b = l.branch(i); b && l.ResidentStateBytes(**b) > 0) {
          return true;
        }
      }
    }
  }
  return false;
}

bool Server::HasActiveTurnWork() const {
  if (reclaiming_) {
    return true;
  }
  for (const auto& m : models_) {
    if (m->has_paged() && node_.InRequest(m->paged().stream())) {
      return true;
    }
    if (m->llm()) {
      auto& l = static_cast<Llm&>(*m);
      for (std::size_t i = 0; i < l.branches(); ++i) {
        if (auto b = l.branch(i); b) {
          if (!l.BranchIdle(**b) || (*b)->HeldByContinuation()) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

std::expected<api::ReleaseReport, std::string> Server::ReleaseWorkingSet(std::string_view reason) {
  if (HasActiveTurnWork()) {
    return std::unexpected("busy");
  }
  api::ReleaseReport report;
  const std::uint64_t before_occupancy =
      node_.catalog().OccupancyOf(node_.domain()).Total().value();
  for (const auto& m : models_) {
    m->DropPlans();
  }
  for (const auto& m : models_) {
    if (!m->llm()) continue;
    auto& l = static_cast<Llm&>(*m);
    for (std::size_t i = 0; i < l.branches(); ++i) {
      auto b = l.branch(i);
      if (!b || !l.BranchIdle(**b) || (*b)->HeldByContinuation() || (*b)->history().empty()) {
        continue;
      }
      const std::uint64_t held = l.ResidentStateBytes(**b);
      if (held == 0) continue;
      if (spill_budget_ > 0 && held <= spill_budget_ &&
          KeepWithinSpillBudget(held, {.spilling = *b})) {
        if (auto spilled = l.SpillIdle(**b); spilled) {
          report.spilled_bytes += held;
        }
      } else if (auto released = (*b)->ReleaseIdleState(); released) {
        // State released on host/catalog
      }
    }
  }
  if (resident_ != nullptr) {
    report.evicted_model = resident_->name();
  }
  for (const auto& m : models_) {
    if (auto evicted = EvictPaged(*m); !evicted) {
      return std::unexpected(std::format("evicting {}: {}", m->name(), evicted.error()));
    }
  }
  resident_ = nullptr;
  DeactivateIdleModels();

  if (node_.workspace_pageable()) {
    std::vector<catalog::ExtentId> workspace_to_evict;
    for (const catalog::ExtentId extent : node_.workspace_extents()) {
      if (const auto view = node_.catalog().Describe(extent);
          view && view->state == catalog::ExtentState::kResident) {
        workspace_to_evict.push_back(extent);
      }
    }
    if (!workspace_to_evict.empty()) {
      (void)node_.Evict(workspace_to_evict);
    }
  }
  report.released_backings = node_.ReleaseUnusedBacking();
  const std::uint64_t after_occupancy = node_.catalog().OccupancyOf(node_.domain()).Total().value();
  report.reclaimed_bytes =
      before_occupancy > after_occupancy ? before_occupancy - after_occupancy : 0;
  platform::ReleaseFreeHeap();
  Log(std::format(
      "released working set ({}) [evicted={}, spilled={:.1f} MiB, reclaimed={:.1f} MiB, "
      "released_backings={}]",
      reason, report.evicted_model.empty() ? "none" : report.evicted_model,
      static_cast<double>(report.spilled_bytes) / (1ULL << 20U),
      static_cast<double>(report.reclaimed_bytes) / (1ULL << 20U), report.released_backings));
  return report;
}

}  // namespace llmp::runtime
