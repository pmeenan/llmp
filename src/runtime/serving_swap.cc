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
}  // namespace
Status Server::WaitReleased() {
  // Progress-based (D-102): only evictions that stop finishing for the
  // node's quiet time are given up on, however long the whole takes.
  auto give_up = Clock::now() + node_.quiet();
  std::size_t before = std::numeric_limits<std::size_t>::max();
  for (;;) {
    std::size_t left = 0;
    if (auto r = node_.Call(
            [&]() -> Status {
              left = node_.scheduler().evictions();
              return {};
            },
            "counting evictions");
        !r) {
      return r;
    }
    if (left == 0) {
      return {};
    }
    if (left < before) {
      before = left;
      give_up = Clock::now() + node_.quiet();
    } else if (Clock::now() > give_up) {
      return Error(
          "backing no load took was not released: no eviction finished for the node's "
          "quiet time");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

Status Server::FinishSwap(SwapParts& parts) {
  if (auto r = WaitReleased(); !r) {
    return r;
  }
  parts.release = Seconds(Clock::now() - parts.ready);
  auto after = node_.Stats();
  if (!after) {
    return std::unexpected(after.error());
  }
  parts.handed_off = after->handed_off - swap_before_.handed_off;
  parts.released_unused = after->released_unused - swap_before_.released_unused;
  parts.occupancy_released = after->catalog_occupancy_bytes;
  if (parts.diagnostic.enabled) {
    parts.diagnostic.finished = swap_host_phases_;
    parts.diagnostic.creates_finished = node_.backing_create_stats();
  }
  return {};
}

std::uint64_t Server::ReleaseKept(std::uint64_t needed, const Served* last,
                                  std::span<const catalog::ExtentId> protect) {
  const std::set<catalog::ExtentId> protected_ids(protect.begin(), protect.end());
  std::vector<catalog::ExtentId> kept;
  for (const bool running : {false, true}) {
    for (const auto& m : models_) {
      if ((m.get() == last) != running) {
        continue;
      }
      const std::vector<catalog::ExtentId> own = m->kept_state();
      kept.insert(kept.end(), own.begin(), own.end());
    }
  }
  if (kept.empty()) {
    return 0;
  }
  std::vector<catalog::ExtentId> resident;
  std::uint64_t bytes = 0;
  auto listed = node_.Call(
      [&]() -> Status {
        const catalog::Catalog& catalog = node_.catalog();
        for (const catalog::ExtentId extent : kept) {
          if (protected_ids.contains(extent)) continue;
          if (bytes >= needed) {
            break;
          }
          if (const auto view = catalog.Describe(extent);
              view && view->state == catalog::ExtentState::kResident && view->discarded &&
              view->leases == 0 && view->registrations == 0) {
            resident.push_back(extent);
            bytes += view->descriptor.size.value();
          }
        }
        return {};
      },
      "listing kept state backing");
  if (!listed || resident.empty() || !node_.Evict(resident)) {
    return 0;
  }
  return bytes;
}

Status Server::EvictPaged(Served& m, bool weights) {
  // Only what a swap moves: its weights and its conversation state (the
  // shared workspace and its own pinned runtime memory stay where they
  // are), with the zeroed backing its clears kept. State is written back
  // to its places.
  std::vector<catalog::ExtentId> own = weights ? m.weights() : std::vector<catalog::ExtentId>{};
  const std::vector<catalog::ExtentId> state = m.state();
  own.insert(own.end(), state.begin(), state.end());
  const std::vector<catalog::ExtentId> kept = m.kept_state();
  own.insert(own.end(), kept.begin(), kept.end());
  std::vector<catalog::ExtentId> resident;
  auto listed = node_.Call(
      [&]() -> Status {
        const catalog::Catalog& catalog = node_.catalog();
        for (const catalog::ExtentId extent : own) {
          if (const auto view = catalog.Describe(extent);
              view && view->state == catalog::ExtentState::kResident) {
            resident.push_back(extent);
          }
        }
        return {};
      },
      "listing a model's resident extents");
  if (!listed) {
    return listed;
  }
  return resident.empty() ? Status{} : node_.Evict(resident);
}

std::expected<std::vector<catalog::ExtentId>, std::string> Server::SpareWorkspace(
    const Served& in) {
  std::vector<catalog::ExtentId> spare;
  if (!node_.workspace_pageable()) return spare;
  std::set<catalog::ExtentId> named;
  for (const auto& [extent, generation] : in.everything().extents) {
    (void)generation;
    named.insert(extent);
  }
  const std::vector<catalog::ExtentId> workspace = node_.workspace_extents();
  auto listed = node_.Call(
      [&]() -> Status {
        for (const catalog::ExtentId extent : workspace) {
          if (named.contains(extent)) continue;
          if (const auto view = node_.catalog().Describe(extent);
              view && view->state == catalog::ExtentState::kResident)
            spare.push_back(extent);
        }
        return {};
      },
      "listing the workspace the incoming model does not use");
  if (!listed) return std::unexpected(listed.error());
  return spare;
}

Status Server::FenceModel(Served& m) {
  // A request still open on its stream may have ended with the
  // cancellation: ending it only frees what it held (PagedNode::FenceStream).
  return node_.FenceStream(m.paged().stream());
}

Status Server::RecoverModel(Served& m) {
  if (!NodeHealthy()) {
    return Error("the node's scheduler faulted");
  }
  // The fence, acknowledged by the model's runner: a step or route the hang
  // left unproven no longer refuses its next steps, and its plans go.
  if (auto fenced = node_.RecoverAfterFence(m.paged()); !fenced) {
    return Error(std::format("its stream could not be fenced: {}", fenced.error()));
  }
  if (auto recovered = m.RecoverInPlace(); !recovered) {
    return recovered;
  }
  if (m.llm()) {
    // The hung unit may have been timed: no calibration counts it (D-103).
    static_cast<Llm&>(m).calibration_samples().Forget();
  }
  // Evicted, so its next activation loads it whole: its weights, and the
  // state it kept (idle conversations) written back.
  if (auto evicted = EvictPaged(m); !evicted) {
    return Error(std::format("evicting it: {}", evicted.error()));
  }
  if (resident_ == &m) {
    resident_ = nullptr;
  }
  DeactivateIdleModels();  // its weight files close: it pages in again only once activated
  if (m.llm() && m.HasRetainedState() && std::ranges::find(spilled_, &m) == spilled_.end()) {
    // As a swap's write-back: the conversations it kept are wholly on disk,
    // and their records follow (D-105).
    m.StateWrittenBack(true);
    spilled_.push_back(&m);
    auto& l = static_cast<Llm&>(m);
    for (std::size_t i = 0; i < l.branches(); ++i) {
      if (auto b = l.branch(i); b && !(*b)->spilled()) {
        l.KeepBranch(**b);
      }
    }
  }
  return {};
}

Status Server::ActivateFiles(const Served& m) {
  for (std::size_t i = 0; i < models_.size(); ++i) {
    if (models_[i].get() == &m) {
      if (auto opened = node_.SetActive(static_cast<int>(i), true); !opened) {
        return Error(
            std::format("model {}: its weights cannot be read: {}", m.name(), opened.error()));
      }
    }
  }
  return {};
}

void Server::DeactivateIdleModels() {
  for (std::size_t i = 0; i < models_.size(); ++i) {
    if (models_[i].get() != resident_) {
      // Never refused: a file a read still uses closes at a later sync.
      std::ignore = node_.SetActive(static_cast<int>(i), false);
    }
  }
}

Status Server::UndoSwap(Served& out, Served& in, std::span<const catalog::ExtentId> preexisting) {
  const std::set<catalog::ExtentId> kept(preexisting.begin(), preexisting.end());
  std::vector<catalog::ExtentId> newly_resident;
  if (auto listed = node_.Call(
          [&]() -> Status {
            for (const auto& [extent, generation] : in.everything().extents) {
              (void)generation;
              const auto view = node_.catalog().Describe(extent);
              if (!kept.contains(extent) && view && view->state == catalog::ExtentState::kResident)
                newly_resident.push_back(extent);
            }
            return {};
          },
          "listing only newly loaded extents of a failed swap");
      !listed)
    return listed;
  if (!newly_resident.empty())
    if (auto evicted = node_.Evict(newly_resident); !evicted) return evicted;
  for (const auto& [extent, generation] : out.everything().extents) {
    (void)generation;
    activation_protect_.push_back(extent);
  }
  if (auto room = MakeRoomForSwap(out, {}); !room) return room;
  std::vector<catalog::ExtentId> again;
  auto listed = node_.Call(
      [&]() -> Status {
        const catalog::Catalog& catalog = node_.catalog();
        for (const auto& [extent, generation] : out.everything().extents) {
          (void)generation;
          if (const auto view = catalog.Describe(extent);
              view && view->state != catalog::ExtentState::kResident) {
            again.push_back(extent);
          }
        }
        return {};
      },
      "listing a failed swap's extents");
  if (!listed) {
    return listed;
  }
  if (!again.empty()) {
    std::vector<engine::LoadStats> log;
    if (auto r = node_.Load(again, std::format("{} back after a failed swap", out.name()), log);
        !r) {
      return r;
    }
  }
  return out.AfterLoad();  // its checks after a load, as after a swap
}

Status Server::Activate(Served& m, SwapParts& parts, std::optional<bool> spill_state) {
  // Records' hashing waits while a swap moves the models (StateKeeper::
  // Quiet): their reads and the swap's share the drive.
  const StateKeeper::Quiet quiet(keeper_.get());
  parts = SwapParts{};
  parts.to = m.name();
  last_activity_ = Clock::now();
  if (resident_ == &m) {
    return {};
  }
  parts.requested = Clock::now();
  // The activation is the incoming model's page-in need: its weight files
  // open now (inside the swap's timing), and whichever model is not
  // resident when this returns has its closed (the outgoing one's once its
  // weights are out).
  if (auto files = ActivateFiles(m); !files) {
    return files;
  }
  struct IdleFiles {
    Server& server;
    ~IdleFiles() { server.DeactivateIdleModels(); }
  } idle_files{*this};
  if (options_.diagnostic_budget_cap_bytes) {
    parts.diagnostic.enabled = true;
    parts.diagnostic.requested = swap_host_phases_;
    parts.diagnostic.creates_requested = node_.backing_create_stats();
    parts.diagnostic.requested_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(parts.requested.time_since_epoch())
            .count());
  }
  auto before = node_.Stats();
  if (!before) {
    return std::unexpected(before.error());
  }
  swap_before_ = *before;
  parts.occupancy_requested = before->catalog_occupancy_bytes;
  Served* const outgoing_model = resident_;
  const auto protected_before = activation_protect_.size();
  struct ProtectedActivation {
    std::vector<catalog::ExtentId>& ids;
    std::size_t before;
    ~ProtectedActivation() { ids.resize(before); }
  } protection{activation_protect_, protected_before};
  for (const auto& [extent, generation] : m.everything().extents) {
    (void)generation;
    activation_protect_.push_back(extent);
  }
  std::vector<catalog::ExtentId> preexisting;
  catalog::Closure saved_state;
  bool state_was_spilled = true;
  if (auto snapshot = node_.Call(
          [&]() -> Status {
            for (const auto& [extent, generation] : m.everything().extents) {
              const auto view = node_.catalog().Describe(extent);
              if (!view || view->content_generation != generation || view->discarded ||
                  view->state == catalog::ExtentState::kQuarantined)
                return Error("the incoming activation closure is stale or invalid");
              if (view->state == catalog::ExtentState::kResident) preexisting.push_back(extent);
            }
            for (const auto extent : m.state()) {
              const auto view = node_.catalog().Describe(extent);
              if (!view) return Error("the incoming state extent is absent");
              saved_state.extents.push_back({extent, view->content_generation});
              state_was_spilled =
                  state_was_spilled && view->state == catalog::ExtentState::kNonresident;
            }
            const std::set<catalog::ExtentId> cached(preexisting.begin(), preexisting.end());
            for (const auto extent : m.weights()) {
              if (cached.contains(extent))
                parts.retained_incoming_weight_bytes +=
                    node_.catalog().Describe(extent)->descriptor.size.value();
            }
            return {};
          },
          "snapshotting incoming cache and saved state before activation");
      !snapshot)
    return snapshot;
  PendingWeightEvictions pending;
  const auto completed_before = node_.counting().read_completed_bytes.load();
  const auto submitted_before = node_.counting().read_submitted_bytes.load();
  const bool restoring = std::ranges::find(spilled_, &m) != spilled_.end();
  // The incoming model's conversations a swap wrote back come back live:
  // their records go before anything of it loads (D-105), not before a
  // refusal that leaves them on disk. Those the reclaim order spilled stay
  // on disk, and kept, until a turn restores them.
  const auto unkeep_incoming = [&m]() {
    if (m.llm()) {
      auto& in = static_cast<Llm&>(m);
      for (std::size_t i = 0; i < in.branches(); ++i) {
        if (auto b = in.branch(i); b && !(*b)->spilled()) {
          in.Unkeep(**b);
        }
      }
    }
  };
  const auto rekeep_incoming = [&]() {
    if (!restoring || !state_was_spilled || saved_state.extents.empty() || !m.llm()) return;
    bool intact = true;
    const auto checked = node_.Call(
        [&]() -> Status {
          for (const auto& [extent, generation] : saved_state.extents) {
            const auto view = node_.catalog().Describe(extent);
            intact = intact && view && view->content_generation == generation &&
                     view->state == catalog::ExtentState::kNonresident && !view->discarded;
          }
          return {};
        },
        "proving a failed incoming load still has whole saved generations");
    if (!checked || !intact) return;
    m.StateWrittenBack(true);
    auto& in = static_cast<Llm&>(m);
    for (std::size_t i = 0; i < in.branches(); ++i)
      if (auto branch = in.branch(i); branch && !(*branch)->spilled()) in.KeepBranch(**branch);
  };
  const auto discard_new_incoming = [&]() -> Status {
    std::vector<catalog::ExtentId> newly_loaded;
    const std::set<catalog::ExtentId> cached(preexisting.begin(), preexisting.end());
    if (auto listed = node_.Call(
            [&]() -> Status {
              for (const auto& [extent, generation] : m.everything().extents) {
                (void)generation;
                const auto view = node_.catalog().Describe(extent);
                if (!cached.contains(extent) && view &&
                    view->state == catalog::ExtentState::kResident)
                  newly_loaded.push_back(extent);
              }
              return {};
            },
            "listing only additions to the incoming activation cache");
        !listed)
      return listed;
    return newly_loaded.empty() ? Status{} : node_.Evict(newly_loaded);
  };
  const auto account_pending = [&]() {
    std::vector<double> priorities;
    std::ignore = node_.Call(
        [&]() -> Status {
          for (const auto& selected : pending.selected) {
            const auto view = node_.catalog().Describe(selected.extent);
            if (view && view->content_generation == selected.generation &&
                (view->state == catalog::ExtentState::kNonresident ||
                 node_.scheduler().EvictionUnmapped(selected.extent))) {
              parts.evicted_weight_bytes += selected.candidate.bytes;
              priorities.push_back(selected.priority);
            }
          }
          return {};
        },
        "accounting only completed or parked selected weight evictions");
    for (const auto priority : priorities) memory::RaiseReclaimInflation(priority);
  };
  bool outgoing_published = false;
  bool outgoing_transfer_complete = false;
  catalog::Closure saved_outgoing;
  const auto publish_outgoing = [&]() {
    if (outgoing_published || outgoing_model == nullptr || !parts.with_state) return;
    outgoing_published = true;
    outgoing_model->StateWrittenBack(true);
    if (std::ranges::find(spilled_, outgoing_model) == spilled_.end())
      spilled_.push_back(outgoing_model);
    auto& out = static_cast<Llm&>(*outgoing_model);
    for (std::size_t i = 0; i < out.branches(); ++i)
      if (auto branch = out.branch(i); branch && !(*branch)->spilled()) out.KeepBranch(**branch);
  };
  const auto keep_outgoing_after_failed_undo = [&]() {
    if (!outgoing_transfer_complete || outgoing_model == nullptr || !parts.with_state ||
        saved_outgoing.extents.empty())
      return;
    std::vector<catalog::ExtentId> restored;
    auto proved = node_.Call(
        [&]() -> Status {
          for (const auto& [extent, generation] : saved_outgoing.extents) {
            const auto view = node_.catalog().Describe(extent);
            const auto* source = node_.scheduler().SourceOf(extent);
            if (!view || view->content_generation != generation ||
                view->saved_generation != generation || view->discarded || source == nullptr ||
                !source->write_back ||
                (view->state != catalog::ExtentState::kResident &&
                 view->state != catalog::ExtentState::kNonresident))
              return Error("outgoing saved generations cannot be proven after failed undo");
            if (view->state == catalog::ExtentState::kResident)
              restored.push_back(extent);
            else if (!view->preserved)
              return Error("outgoing saved state is not preserved");
          }
          return {};
        },
        "proving the completed outgoing spill after failed undo");
    // The successful original transfer saved every extent. The undo only
    // restored it; no outgoing request ran. Release that restored subset
    // unchanged before advertising the model as wholly spilled again.
    if (proved && !restored.empty()) proved = node_.Evict(restored, {.unchanged = true});
    if (proved)
      proved = node_.Call(
          [&]() -> Status {
            for (const auto& [extent, generation] : saved_outgoing.extents) {
              const auto view = node_.catalog().Describe(extent);
              if (!view || view->content_generation != generation ||
                  view->saved_generation != generation || !view->preserved ||
                  view->state != catalog::ExtentState::kNonresident)
                return Error("outgoing state did not retire wholly to its saved generations");
            }
            return {};
          },
          "checking whole outgoing spill retirement after failed undo");
    if (proved)
      publish_outgoing();
    else
      Log(std::format("failed undo's outgoing state was not kept: {}", proved.error()));
  };
  const auto release_subset = [&](std::span<const catalog::ExtentId> selected,
                                  std::vector<catalog::ExtentId>& release) -> Status {
    if (!options_.partial_weight_eviction || !handoff_) return {};
    const SwapPhaseTimer timer(
        options_.diagnostic_budget_cap_bytes ? &swap_host_phases_.partition_seconds : nullptr);
    std::vector<SwapBackingExtent> missing;
    std::vector<SwapBackingExtent> donors;
    std::uint64_t occupancy = 0;
    const auto listed = node_.Call(
        [&]() -> Status {
          const auto describe =
              [&](catalog::ExtentId extent) -> std::expected<SwapBackingExtent, std::string> {
            const auto view = node_.catalog().Describe(extent);
            const auto* source = node_.scheduler().SourceOf(extent);
            if (!view || source == nullptr)
              return Error("a swap extent has no descriptor or source");
            SwapBackingExtent result{
                .extent = extent, .bytes = view->descriptor.size.value(), .backing = std::nullopt};
            if (source->backing && source->backing->size.value() == result.bytes)
              result.backing = SwapBackingKey{.domain = view->descriptor.domain,
                                              .allocation_class = source->backing->allocation_class,
                                              .size = source->backing->size.value()};
            return result;
          };
          for (const auto& [extent, generation] : m.everything().extents) {
            (void)generation;
            const auto view = node_.catalog().Describe(extent);
            if (view && view->state != catalog::ExtentState::kResident) {
              auto entry = describe(extent);
              if (!entry) return std::unexpected(entry.error());
              missing.push_back(*entry);
            }
          }
          for (const auto extent : selected) {
            const auto view = node_.catalog().Describe(extent);
            if (view && view->state == catalog::ExtentState::kResident) {
              auto entry = describe(extent);
              if (!entry) return std::unexpected(entry.error());
              donors.push_back(*entry);
            }
          }
          occupancy = node_.catalog().OccupancyOf(node_.domain()).Total().value();
          return {};
        },
        "partitioning selected donor backing by actual class and size");
    if (!listed) return listed;
    const auto partition =
        ReleaseForHandoff(node_.domain(), occupancy, node_.budget().value(), missing, donors);
    if (!partition) return std::unexpected(partition.error());
    release = *partition;
    return {};
  };
  const auto selected_weights = [&]() {
    catalog::Closure closure;
    for (const auto& selected : pending.selected)
      closure.extents.push_back({selected.extent, selected.generation});
    return closure;
  };
  const auto record_eviction_split = [&](const scheduler::SwapReport& report) {
    if (!parts.diagnostic.enabled || report.started == Clock::time_point{}) return;
    parts.diagnostic.eviction_split_available = true;
    const auto ns = [](Clock::time_point at) {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(at.time_since_epoch()).count());
    };
    parts.diagnostic.scheduler_started_ns = ns(report.started);
    parts.diagnostic.scheduler_evicted_ns = ns(report.evicted);
    parts.diagnostic.scheduler_loaded_ns = ns(report.loaded);
    parts.diagnostic.eviction_prepare_seconds = Seconds(report.started - parts.requested);
    parts.diagnostic.eviction_retire_seconds = Seconds(report.evicted - report.started);
  };
  Clock::time_point evicted = parts.requested;
  Clock::time_point loaded;
  if (resident_ == nullptr) {
    // Nothing resident: the first load of its weights (its state is
    // resident since setup, or written back by a swap that failed and
    // could not be undone: restored with them).
    // Whatever another model still holds from a swap that failed and could
    // not be undone goes out first (state written back).
    for (const auto& other : models_) {
      if (other.get() != &m) {
        if (auto r = EvictPaged(*other, !options_.partial_weight_eviction); !r) {
          return r;
        }
        if (other->llm() && other->HasRetainedState() &&
            std::ranges::find(spilled_, other.get()) == spilled_.end()) {
          spilled_.push_back(other.get());
        }
      }
    }
    // And the workspace only they used: the incoming closure names its own.
    auto spare = SpareWorkspace(m);
    if (!spare) return std::unexpected(spare.error());
    if (!spare->empty()) {
      if (auto r = node_.Evict(std::move(*spare)); !r) return r;
    }
    // Every model's plans and graphs stay charged inside the budget: room
    // for the whole load is made through the reclaim order first, as a
    // swap's is, or a load after a failed swap (or a recovery) could be
    // refused for room again and again with nothing to free it.
    if (auto room = MakeRoomForSwap(m, {}, options_.partial_weight_eviction ? &pending : nullptr);
        !room) {
      return room;
    }
    std::vector<engine::LoadStats> log;
    std::vector<catalog::ExtentId> all;
    for (const auto& [extent, generation] : m.everything().extents) {
      (void)generation;
      all.push_back(extent);
    }
    Status loaded_result;
    if (options_.partial_weight_eviction) {
      std::vector<catalog::ExtentId> victims;
      for (const auto& selected : pending.selected) victims.push_back(selected.extent);
      std::vector<catalog::ExtentId> release;
      if (auto partition = release_subset(victims, release); !partition) return partition;
      scheduler::SwapReport report;
      unkeep_incoming();
      loaded_result = node_.Swap(std::move(victims), m.everything(), handoff_, report,
                                 std::move(release), selected_weights());
      account_pending();
      if (loaded_result) record_eviction_split(report);
      evicted = report.evicted == Clock::time_point{} ? evicted : report.evicted;
      parts.evicted = report.evictions;
      parts.loaded = report.loads;
    } else {
      unkeep_incoming();
      loaded_result = node_.Load(all, std::format("{}'s first load", m.name()), log);
      parts.loaded = log.empty() ? 0 : log.front().extents;
    }
    if (!loaded_result) {
      if (auto cleaned = discard_new_incoming(); !cleaned)
        Log(std::format("failed first-load additions could not be cleaned: {}", cleaned.error()));
      rekeep_incoming();
      return loaded_result;
    }
    loaded = Clock::now();
  } else {
    Served& out = *resident_;
    parts.from = out.name();
    // The caller yields only at a completed unit. Release its request even
    // when the incoming closure is cached and no weight victim is needed.
    if (node_.InRequest(out.paged().stream())) {
      if (auto ended = node_.EndRequest(out.paged().stream()); !ended) return ended;
    }
    if (auto used = RecordWeightUse(out); !used) return used;
    const bool conversation = out.llm() && out.HasRetainedState();
    parts.with_state = out.llm() && spill_state.value_or(conversation);
    std::vector<catalog::ExtentId> extents;
    std::vector<catalog::ExtentId> unchanged;
    const SpillView view{.out = &out, .in = &m};
    if (parts.with_state) {
      auto& l = static_cast<Llm&>(out);
      // The spill budget holds a swap's write-back too, counted as the swap
      // leaves it: all the outgoing model's state on disk, the incoming
      // model's resident (its conversations, which then stop counting, are
      // never deleted for it). Past it, the least recently used spilled
      // conversations are deleted first, the outgoing model's among them,
      // but only once the swap's room is made below: one a request holds
      // is not, and a swap that would not fit even with every other one
      // deleted is refused here, before anything moves or is deleted.
      if (!SpillBudgetFits(0, view)) {
        return Error("the swap's retained request state exceeds the spill budget");
      }
      // Kept across a restart (D-105): each conversation's last verify's
      // owed restore runs first (no request holds the stream during a
      // swap), so the write-back holds it whole. Before the room is sized:
      // whatever it charges is counted there.
      if (l.keeps()) {
        for (std::size_t i = 0; i < l.branches(); ++i) {
          if (auto b = l.branch(i); b && l.ResidentStateBytes(**b) != 0) {
            (void)l.SettleIdle(**b);
          }
        }
      }
    }
    if (options_.partial_weight_eviction && parts.with_state) {
      if (auto saved = node_.Call(
              [&]() -> Status {
                for (const auto extent : out.state()) {
                  const auto view = node_.catalog().Describe(extent);
                  if (!view) return Error("the settled outgoing state extent is absent");
                  saved_outgoing.extents.push_back({extent, view->content_generation});
                }
                return {};
              },
              "snapshotting settled outgoing state generations");
          !saved)
        return saved;
    }
    // Its plans and graphs stay (D-090 as amended 2026-10-02): charged
    // inside the budget, they go only when the reclaim order needs their
    // room, here first if the incoming model does not fit beside them. The
    // room is checked again after each reclaim, and a swap it cannot make
    // room for is refused before anything moves. What goes out is its
    // weights and, with its state, all of that (how much a write-back
    // writes, and whether the budget deletes some of it, does not change
    // what the swap frees; the reclaim order spills no idle state here).
    const std::vector<catalog::ExtentId> weights = out.weights();
    const std::vector<catalog::ExtentId> kept = out.kept_state();
    // What of the shared workspace only the outgoing model used goes out
    // with its weights (scratch: nothing is written back), so the incoming
    // model's closure, which names its own prefix, fits beside the rest.
    auto spare = SpareWorkspace(m);
    if (!spare) return std::unexpected(spare.error());
    const std::uint64_t graphs_before = out.graphs().kept + m.graphs().kept;
    {
      std::vector<catalog::ExtentId> going =
          options_.partial_weight_eviction ? std::vector<catalog::ExtentId>{} : weights;
      going.insert(going.end(), kept.begin(), kept.end());
      going.insert(going.end(), spare->begin(), spare->end());
      if (parts.with_state) {
        const std::vector<catalog::ExtentId> state = out.state();
        going.insert(going.end(), state.begin(), state.end());
        for (const catalog::ExtentId extent : out.unchanged_state()) {
          if (std::ranges::find(state, extent) == state.end()) {
            going.push_back(extent);
          }
        }
      }
      if (auto room =
              MakeRoomForSwap(m, going, options_.partial_weight_eviction ? &pending : nullptr);
          !room) {
        return room;
      }
    }
    if (parts.with_state) {
      auto& l = static_cast<Llm&>(out);
      // The room is made: the budget's deletions now, which the check above
      // found enough (nothing since spilled or restored; a deletion that
      // fails still refuses the swap before anything moves).
      if (!KeepWithinSpillBudget(0, view)) {
        return Error("the swap's retained request state exceeds the spill budget");
      }
      // Its state first, so its write-backs start first; what nothing wrote
      // since its slot's spill file last held it is released without
      // writing (an incremental spill), before the swap.
      unchanged = out.unchanged_state();
      for (const catalog::ExtentId extent : out.state()) {
        if (std::ranges::find(unchanged, extent) == unchanged.end()) {
          extents.push_back(extent);
        }
      }
      // What it writes back: its resident state (a conversation the reclaim
      // order spilled already is on disk, and passed over).
      for (std::size_t i = 0; i < l.branches(); ++i) {
        if (auto b = l.branch(i); b) {
          parts.spilled_bytes += l.ResidentStateBytes(**b);
        }
      }
    }
    extents.insert(extents.end(), kept.begin(), kept.end());
    if (options_.partial_weight_eviction) {
      for (const auto& selected : pending.selected) extents.push_back(selected.extent);
    } else {
      extents.insert(extents.end(), weights.begin(), weights.end());
    }
    extents.insert(extents.end(), spare->begin(), spare->end());
    const std::uint64_t graphs_after = out.graphs().kept + m.graphs().kept;
    parts.dropped_graphs = graphs_before > graphs_after ? graphs_before - graphs_after : 0;
    unkeep_incoming();
    if (!unchanged.empty()) {
      if (auto r = node_.Evict(unchanged, {.unchanged = true}); !r) {
        out.StateWrittenBack(false);
        if (auto undone = UndoSwap(out, m, preexisting); !undone) {
          resident_ = nullptr;  // the next activation loads one whole
        }
        rekeep_incoming();
        return r;
      }
    }
    scheduler::SwapReport report;
    std::vector<catalog::ExtentId> release;
    if (auto partition = release_subset(extents, release); !partition) {
      if (auto undone = UndoSwap(out, m, preexisting); !undone) resident_ = nullptr;
      rekeep_incoming();
      return partition;
    }
    const auto swapped = node_.Swap(std::move(extents), m.everything(), handoff_, report,
                                    std::move(release), selected_weights());
    account_pending();
    if (!swapped) {
      const auto& r = swapped;
      if (parts.with_state) {
        out.StateWrittenBack(false);
      }
      // Never the process: whatever of the incoming model came in goes out
      // again and the outgoing one comes back whole, so both stay usable and
      // only the request that needed the swap fails.
      if (auto undone = UndoSwap(out, m, preexisting); undone) {
        Log(std::format("swap {} -> {} failed and was undone: {}", out.name(), m.name(),
                        r.error()));
      } else {
        Log(
            std::format("swap {} -> {} failed ({}) and could not be undone ({}): no model is "
                        "resident until the next one loads whole",
                        out.name(), m.name(), r.error(), undone.error()));
        if (parts.with_state && std::ranges::find(spilled_, &out) == spilled_.end()) {
          spilled_.push_back(&out);
        }
        resident_ = nullptr;
      }
      rekeep_incoming();
      return r;
    }
    outgoing_transfer_complete = true;
    record_eviction_split(report);
    if (!options_.partial_weight_eviction) publish_outgoing();
    evicted = report.evicted;
    loaded = report.loaded;
    parts.evicted = report.evictions;
    parts.loaded = report.loads;
  }
  if (parts.diagnostic.enabled) parts.diagnostic.creates_loaded = node_.backing_create_stats();
  parts.evict = Seconds(evicted - parts.requested);
  Clock::time_point restored = evicted;
  if (restoring) {
    restored = std::max(restored, times_.Latest(m.state()));
    parts.restore = Seconds(restored - evicted);
    if (!options_.partial_weight_eviction) std::erase(spilled_, &m);
  }
  parts.page_in = Seconds(loaded - restored);
  parts.read_bytes = node_.counting().read_completed_bytes.load() - completed_before;
  parts.read_submitted_bytes = node_.counting().read_submitted_bytes.load() - submitted_before;
  if (outgoing_model != nullptr) {
    std::ignore = node_.Call(
        [&]() -> Status {
          for (const auto extent : outgoing_model->weights()) {
            const auto view = node_.catalog().Describe(extent);
            if (view && view->state == catalog::ExtentState::kResident)
              parts.retained_outgoing_weight_bytes += view->descriptor.size.value();
            else if (!options_.partial_weight_eviction && view &&
                     (view->state == catalog::ExtentState::kNonresident ||
                      node_.scheduler().EvictionUnmapped(extent)))
              parts.evicted_weight_bytes += view->descriptor.size.value();
          }
          return {};
        },
        "measuring actual retained outgoing weights");
  }
  resident_ = &m;
  // A model whose checks after its load fail is not left resident: its
  // kernels index what those checks cover unchecked (DeepSeek's hash
  // routing, Qwen3.8's n-gram hash) or replay graphs at its places (D-090).
  // Evicted (its state written back), its next activation loads it whole
  // and checks it again; only the requests that needed it fail (D-102).
  const auto unload = [&](std::string error) -> Status {
    if (options_.partial_weight_eviction && outgoing_model != nullptr) {
      if (auto undone = UndoSwap(*outgoing_model, m, preexisting); undone)
        resident_ = outgoing_model;
      else {
        error += std::format("; undoing it: {}", undone.error());
        resident_ = nullptr;
        keep_outgoing_after_failed_undo();
      }
      rekeep_incoming();
      return Error(std::move(error));
    }
    if (options_.partial_weight_eviction) {
      if (auto cleaned = discard_new_incoming(); !cleaned)
        error += std::format("; cleaning its new extents: {}", cleaned.error());
      resident_ = nullptr;
      rekeep_incoming();
      return Error(std::move(error));
    }
    if (auto evicted = EvictPaged(m); !evicted) {
      error += std::format("; evicting it: {}", evicted.error());
      m.StateWrittenBack(false);
    } else if (m.llm() && m.HasRetainedState() &&
               std::ranges::find(spilled_, &m) == spilled_.end()) {
      // As a swap's write-back: its conversations are wholly on disk, and
      // their records follow (D-105).
      m.StateWrittenBack(true);
      spilled_.push_back(&m);
      auto& l = static_cast<Llm&>(m);
      for (std::size_t i = 0; i < l.branches(); ++i) {
        if (auto b = l.branch(i); b && !(*b)->spilled()) {
          l.KeepBranch(**b);
        }
      }
    }
    resident_ = nullptr;
    return Error(std::move(error));
  };
  if (auto r = m.AfterLoad(); !r) {
    return unload(std::format("{} after its load: {}", m.name(), r.error()));
  }
  // Its places still pinned where it registered them (D-090).
  std::size_t unpinned = 0;
  const std::vector<catalog::ExtentId> managed = m.paged().managed_extents();
  if (auto r = node_.Call(
          [&]() -> Status {
            for (const catalog::ExtentId extent : managed) {
              unpinned += node_.scheduler().PlacePinned(extent) ? 0 : 1;
            }
            return {};
          },
          "checking places");
      !r) {
    return unload(std::format("checking incoming places failed: {}", r.error()));
  }
  if (unpinned != 0) {
    return unload(std::format("{} of {}'s {} extents are not pinned at their places", unpinned,
                              m.name(), managed.size()));
  }
  if (auto r = m.CheckPlaces(); !r) {
    return unload(std::format("{}'s places: {}", m.name(), r.error()));
  }
  publish_outgoing();
  if (restoring) std::erase(spilled_, &m);
  page_read_bytes_ += parts.read_bytes;
  page_read_seconds_ += parts.page_in + parts.restore;
  parts.ready = Clock::now();
  parts.setup = Seconds(parts.ready - loaded);
  parts.total = Seconds(parts.ready - parts.requested);
  if (parts.diagnostic.enabled) {
    parts.diagnostic.ready = swap_host_phases_;
    parts.diagnostic.creates_ready = node_.backing_create_stats();
  }
  return {};
}

}  // namespace llmp::runtime
