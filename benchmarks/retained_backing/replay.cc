// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "retained_backing/replay.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/check.h"
#include "base/sha256.h"
#include "providers/fake/fake_device_memory.h"
#include "retained_backing/designs.h"
#include "retained_backing/trace.h"

namespace llmp::rb {
namespace {

using base::Check;

class Engine final : public Env {
 public:
  Engine(const Trace& trace, Design& design, CountingMemory& memory,
         const providers::fake::FakeDeviceMemory& fake, Metrics& metrics,
         const ReplayOptions& options)
      : trace_(trace),
        design_(design),
        memory_(memory),
        fake_(fake),
        metrics_(metrics),
        options_(options),
        reference_(trace.groups, trace.budget),
        resident_(trace.groups.size(), false),
        leased_(trace.groups.size(), 0),
        wanted_(trace.groups.size(), 0),
        restored_(trace.groups.size(), 0),
        key_(trace.groups.size(), 0) {}

  std::uint64_t Limit() const override { return trace_.budget - shrunk_; }
  const Group& GroupOf(GroupId g) const override { return trace_.groups[g]; }
  bool Evictable(GroupId g) const override {
    return resident_[g] && leased_[g] == 0 && wanted_[g] != epoch_;
  }
  std::uint64_t Stamp(GroupId g) const override { return reference_.Stamp(g); }

  std::optional<GroupId> LruVictim() const override {
    for (const auto& [stamp, g] : lru_) {
      if (leased_[g] == 0 && wanted_[g] != epoch_) {
        return g;
      }
    }
    return std::nullopt;
  }

  void Evicted(GroupId g) override {
    Check(Evictable(g), "replay: a design evicted a protected group");
    Unresident(g);
    const std::uint64_t bytes = trace_.groups[g].stored;
    if (in_shrink_) {
      ++metrics_.shrink_extra_evictions;
      metrics_.shrink_extra_evicted_bytes += bytes;
    } else {
      ++metrics_.access_extra_evictions;
      metrics_.access_extra_evicted_bytes += bytes;
    }
    memory_.Digest(kEvictedTag, g, 0);
  }

  void Relocated(GroupId g) override {
    Check(resident_[g] && leased_[g] == 0 && !trace_.groups[g].expert,
          "replay: only unleased dense groups move at M2");
    ++metrics_.relocations;
    metrics_.relocated_bytes += trace_.groups[g].stored;
    delayed_ = true;
    memory_.Digest(kRelocatedTag, g, design_.PositionOf(g));
  }

  void Run() {
    for (std::size_t index = 0; index < trace_.events.size(); ++index) {
      if (!Step(index)) {
        metrics_.refusals = 1;
        metrics_.refused_at = static_cast<std::int64_t>(index);
        return;
      }
    }
    Check(!pending_evict_ && !pending_restore_ && leases_.empty() && shrunk_ == 0,
          "replay: the trace ends with open leases or pending records");
    Validate();
  }

 private:
  static constexpr std::uint64_t kEvictedTag = 1000;
  static constexpr std::uint64_t kPlacedTag = 1001;
  static constexpr std::uint64_t kRelocatedTag = 1002;

  void Resident(GroupId g) {
    resident_[g] = true;
    resident_bytes_ += trace_.groups[g].stored;
    key_[g] = reference_.Stamp(g);
    lru_.emplace(key_[g], g);
  }

  void Unresident(GroupId g) {
    Check(resident_[g], "replay: evicting a group that is not resident");
    resident_[g] = false;
    resident_bytes_ -= trace_.groups[g].stored;
    lru_.erase({key_[g], g});
  }

  // The reference touched these: move them in the design's recency order.
  void Retouch(std::span<const GroupId> ids) {
    for (const GroupId g : ids) {
      if (!resident_[g]) {
        continue;
      }
      const std::uint64_t now = reference_.Stamp(g);
      if (now != key_[g]) {
        lru_.erase({key_[g], g});
        design_.Touched(g, key_[g], now);
        key_[g] = now;
        lru_.emplace(now, g);
      }
    }
  }

  bool Step(std::size_t index) {
    const Event& event = trace_.events[index];
    const std::span<const GroupId> ids = trace_.Ids(event);
    if (event.ev == Ev::kEvict || event.ev == Ev::kRestore) {
      std::optional<std::span<const GroupId>>& pending =
          event.ev == Ev::kEvict ? pending_evict_ : pending_restore_;
      Check(!pending && !pending_restore_, "replay: evict or restore out of order");
      pending = ids;
      return true;
    }
    std::span<const GroupId> touched = ids;
    if (event.ev == Ev::kRelease) {
      const auto found = leases_.find(event.lease);
      Check(found != leases_.end(), "replay: release of an unknown lease");
      touched = found->second;
    }
    auto step = reference_.Apply(event, ids);
    Check(step.has_value(), "replay: the reference refuses the trace");
    auto same = [](const std::optional<std::span<const GroupId>>& recorded,
                   const std::vector<GroupId>& computed) {
      return recorded ? std::ranges::equal(*recorded, computed) : computed.empty();
    };
    Check(same(pending_evict_, step->evicted) && same(pending_restore_, step->restored),
          "replay: the trace's reference records are not the reference's choices");
    pending_evict_.reset();
    pending_restore_.reset();
    Retouch(touched);

    switch (event.ev) {
      case Ev::kLease:
      case Ev::kUse:
        return Access(event, ids, *step);
      case Ev::kShrink: {
        shrunk_ += event.bytes;
        ++epoch_;  // a shrink wants nothing
        DropVictims(step->evicted);
        in_shrink_ = true;
        const bool met = design_.Shrink(*this);
        in_shrink_ = false;
        if (!met) {
          return false;
        }
        Tick();
        Validate();
        return true;
      }
      case Ev::kGrow:
        shrunk_ -= event.bytes;
        return true;
      case Ev::kRelease: {
        const auto found = leases_.find(event.lease);
        for (const GroupId g : found->second) {
          --leased_[g];
        }
        leases_.erase(found);
        return true;
      }
      case Ev::kRequest:
      case Ev::kEvict:
      case Ev::kRestore:
        break;
    }
    return true;
  }

  // Every design evicts the reference's victims first.
  void DropVictims(const std::vector<GroupId>& victims) {
    for (const GroupId v : victims) {
      Check(leased_[v] == 0 && wanted_[v] != epoch_,
            "replay: the reference evicts a protected group");
      if (resident_[v]) {
        design_.Drop(v);
        Unresident(v);
        memory_.Digest(kEvictedTag, v, 1);
      }
    }
  }

  bool Access(const Event& event, std::span<const GroupId> ids, const Reference::Step& step) {
    ++epoch_;
    for (const GroupId g : ids) {
      wanted_[g] = epoch_;
    }
    DropVictims(step.evicted);
    for (const GroupId g : step.restored) {
      Check(!resident_[g], "replay: the reference restores what the design holds");
      restored_[g] = epoch_;
    }
    std::vector<GroupId> missing;
    for (const GroupId g : ids) {
      if (!resident_[g]) {
        missing.push_back(g);
        const std::uint64_t bytes = trace_.groups[g].stored;
        ++metrics_.restores;
        metrics_.restored_bytes += bytes;
        if (restored_[g] != epoch_) {
          ++metrics_.extra_restores;
          metrics_.extra_restored_bytes += bytes;
        }
      }
    }
    metrics_.reference_restores += step.restored.size();
    for (const GroupId g : step.restored) {
      metrics_.reference_restored_bytes += trace_.groups[g].stored;
    }
    delayed_ = false;
    if (!design_.Admit(missing, *this)) {
      return false;
    }
    for (const GroupId g : missing) {
      Resident(g);
      memory_.Digest(kPlacedTag, g, design_.PositionOf(g));
    }
    if (delayed_) {
      ++metrics_.delayed_admissions;
    }
    if (event.ev == Ev::kLease) {
      for (const GroupId g : ids) {
        ++leased_[g];
      }
      Check(leases_.emplace(event.lease, std::vector<GroupId>(ids.begin(), ids.end())).second,
            "replay: lease reused");
    }
    design_.EndAccess();
    Tick();
    return true;
  }

  void Tick() {
    const std::uint64_t held = design_.Held();
    Check(held <= Limit(), "replay: held backing exceeds the budget less the outstanding shrink");
    Check(held == fake_.in_use().value(),
          "replay: the design's held backing is not the provider's");
    Check(held >= resident_bytes_, "replay: resident bytes exceed held backing");
    const std::uint64_t waste = held - resident_bytes_;
    metrics_.waste_sum += waste;
    metrics_.waste_peak = std::max(metrics_.waste_peak, waste);
    metrics_.held_peak = std::max(metrics_.held_peak, held);
    ++metrics_.ticks;
    if (options_.check_every != 0 && metrics_.ticks % options_.check_every == 0) {
      Validate();
    }
  }

  void Validate() const {
    design_.CheckInvariants();
    std::uint64_t bytes = 0;
    for (GroupId g = 0; g < trace_.groups.size(); ++g) {
      if (resident_[g]) {
        Check(reference_.Resident(g), "replay: the design holds a group the reference does not");
        bytes += trace_.groups[g].stored;
      }
    }
    Check(bytes == resident_bytes_ && lru_.size() <= trace_.groups.size(),
          "replay: resident bytes out of step");
  }

  const Trace& trace_;
  Design& design_;
  CountingMemory& memory_;
  const providers::fake::FakeDeviceMemory& fake_;
  Metrics& metrics_;
  const ReplayOptions& options_;
  Reference reference_;
  std::vector<bool> resident_;  // in the design
  std::uint64_t resident_bytes_ = 0;
  std::vector<std::uint32_t> leased_;
  std::vector<std::uint64_t> wanted_;    // the epoch of the access that wants it
  std::vector<std::uint64_t> restored_;  // the epoch whose reference restores it
  std::vector<std::uint64_t> key_;       // its key in lru_
  std::set<std::pair<std::uint64_t, GroupId>> lru_;
  std::map<std::uint32_t, std::vector<GroupId>> leases_;
  std::optional<std::span<const GroupId>> pending_evict_;
  std::optional<std::span<const GroupId>> pending_restore_;
  std::uint64_t epoch_ = 0;
  std::uint64_t shrunk_ = 0;
  bool in_shrink_ = false;
  bool delayed_ = false;
};

}  // namespace

Metrics Replay(const Trace& trace, const DesignSpec& spec, const ReplayOptions& options) {
  providers::fake::FakeDeviceMemory fake(base::Bytes(kChunk), base::Bytes(trace.budget),
                                         providers::fake::Contents::kNone);
  CountingMemory memory(fake);
  const std::unique_ptr<Design> design =
      MakeDesign(spec, trace.groups, trace.models, trace.budget, memory);
  Metrics metrics;
  metrics.design = spec.name;
  metrics.file = trace.file;
  metrics.budget = trace.budget;
  Engine(trace, *design, memory, fake, metrics, options).Run();
  metrics.calls = memory.counts();
  metrics.digest = base::ToHex(memory.digest().Finish());
  memory.Freeze();
  design->Teardown();
  Check(fake.backings() == 0 && fake.reservations() == 0, "replay: the design leaked backing");
  return metrics;
}

std::string ToJson(const Metrics& m) {
  std::string calls;
  for (std::size_t i = 0; i < kCalls; ++i) {
    calls +=
        std::format("{}\"{}\":{}", i == 0 ? "" : ",", CallName(static_cast<Call>(i)), m.calls[i]);
  }
  return std::format(
      "{{\"design\":\"{}\",\"file\":\"{}\",\"budget_bytes\":{},\"ticks\":{},\"waste_sum\":{},"
      "\"waste_peak\":{},\"held_peak\":{},\"restores\":{},\"restored_bytes\":{},"
      "\"reference_restores\":{},\"reference_restored_bytes\":{},\"extra_restores\":{},"
      "\"extra_restored_bytes\":{},\"shrink_extra_evictions\":{},"
      "\"shrink_extra_evicted_bytes\":{},\"content_lost_bytes\":{},"
      "\"access_extra_evictions\":{},\"access_extra_evicted_bytes\":{},\"refusals\":{},"
      "\"refused_at\":{},\"delayed_admissions\":{},\"relocations\":{},\"relocated_bytes\":{},"
      "\"calls\":{{{}}},\"digest\":\"{}\"}}",
      m.design, m.file, m.budget, m.ticks, m.waste_sum, m.waste_peak, m.held_peak, m.restores,
      m.restored_bytes, m.reference_restores, m.reference_restored_bytes, m.extra_restores,
      m.extra_restored_bytes, m.shrink_extra_evictions, m.shrink_extra_evicted_bytes,
      m.content_lost(), m.access_extra_evictions, m.access_extra_evicted_bytes, m.refusals,
      m.refused_at, m.delayed_admissions, m.relocations, m.relocated_bytes, calls, m.digest);
}

}  // namespace llmp::rb
