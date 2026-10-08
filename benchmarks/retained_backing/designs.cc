// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "retained_backing/designs.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <format>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/check.h"
#include "providers/device_memory.h"
#include "retained_backing/trace.h"

namespace llmp::rb {
namespace {

using base::Bytes;
using base::Check;
using providers::Access;
using providers::BackingId;
using providers::Failure;
using providers::ReservationId;

// GPU-accessible host backing (D-034): the fake's second class.
constexpr std::size_t kHostClass = 1;
constexpr GroupId kNoGroup = UINT32_MAX;
constexpr std::int32_t kGeneral = -1;

template <typename T>
T Must(std::expected<T, Failure> result, const char* what) {
  Check(result.has_value(), what);
  return *std::move(result);
}

void Must(const std::expected<void, Failure>& result, const char* what) {
  Check(result.has_value(), what);
}

std::uint64_t CeilDiv(std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; }

// ------------------------------------------------------------------ D-033

class Handles final : public Design {
 public:
  Handles(std::span<const Group> groups, std::span<const Model> models, CountingMemory& memory)
      : groups_(groups),
        memory_(memory),
        region_(groups.size(), 0),
        mapped_(groups.size()),
        reservation_(models.size()),
        extent_(models.size(), 0) {
    for (std::size_t m = 0; m < models.size(); ++m) {
      std::uint64_t offset = 0;
      for (GroupId g = models[m].first; g < models[m].end; ++g) {
        region_[g] = offset;
        offset += Chunks(g) * kChunk;
      }
      extent_[m] = offset;
    }
  }

  void Drop(GroupId g) override { Unmap(g); }

  bool Admit(std::span<const GroupId> missing, Env& env) override {
    for (const GroupId g : missing) {
      Check(mapped_[g].empty(), "D-033: restoring a group that is mapped");
      const ReservationId reservation = ReservationFor(groups_[g].model);
      const std::uint64_t chunks = Chunks(g);
      for (std::uint64_t i = 0; i < chunks; ++i) {
        std::optional<BackingId> handle;
        while (!handle) {
          if (!handoff_.empty()) {
            handle = handoff_.back();  // handed over from an eviction for this access
            handoff_.pop_back();
          } else if ((handles_ + 1) * kChunk <= env.Limit()) {
            handle = Must(memory_.Create(kHostClass, Bytes(kChunk)), "D-033: create");
            ++handles_;
          } else {
            const std::optional<GroupId> victim = env.LruVictim();
            if (!victim) {
              return false;
            }
            Unmap(*victim);
            env.Evicted(*victim);
          }
        }
        Must(memory_.Map(reservation, Bytes(region_[g] + (i * kChunk)), *handle), "D-033: map");
        mapped_[g].push_back(*handle);
      }
      Must(memory_.SetAccess(reservation, Bytes(region_[g]), Bytes(chunks * kChunk),
                             Access::kReadWrite),
           "D-033: set access");
    }
    return true;
  }

  // No free pool: what the access did not take is released.
  void EndAccess() override { ReleaseHandoff(); }

  bool Shrink(Env& env) override {
    ReleaseHandoff();
    while (Held() > env.Limit()) {
      const std::optional<GroupId> victim = env.LruVictim();
      if (!victim) {
        return false;
      }
      Unmap(*victim);
      env.Evicted(*victim);
      ReleaseHandoff();
    }
    return true;
  }

  std::uint64_t Held() const override { return handles_ * kChunk; }

  // The model in the top bits, the offset in its range below.
  std::uint64_t PositionOf(GroupId g) const override {
    return (std::uint64_t{groups_[g].model} << 48U) | region_[g];
  }

  void CheckInvariants() const override {
    std::uint64_t mapped = 0;
    for (GroupId g = 0; g < groups_.size(); ++g) {
      Check(mapped_[g].empty() || mapped_[g].size() == Chunks(g),
            "D-033: a group is partly mapped");
      Check(region_[g] % kChunk == 0, "D-033: a region is not 2 MiB aligned");
      mapped += mapped_[g].size();
    }
    Check(mapped + handoff_.size() == handles_, "D-033: handle count");
  }

  void Teardown() override {
    for (GroupId g = 0; g < groups_.size(); ++g) {
      if (!mapped_[g].empty()) {
        Unmap(g);
      }
    }
    ReleaseHandoff();
    for (std::optional<ReservationId>& reservation : reservation_) {
      if (reservation) {
        Must(memory_.Free(*reservation), "D-033: free");
        reservation.reset();
      }
    }
  }

 private:
  std::uint64_t Chunks(GroupId g) const { return CeilDiv(groups_[g].stored, kChunk); }

  ReservationId ReservationFor(std::uint32_t model) {
    std::optional<ReservationId>& reservation = reservation_[model];
    if (!reservation) {
      reservation = Must(memory_.Reserve(Bytes(extent_[model])), "D-033: reserve");
    }
    return *reservation;
  }

  // One unmap over the group's chunks; its handles await a consumer.
  void Unmap(GroupId g) {
    std::vector<BackingId>& handles = mapped_[g];
    const std::optional<ReservationId>& reservation = reservation_[groups_[g].model];
    if (handles.empty() || !reservation) {
      base::Fatal("D-033: evicting a group that is not mapped");
    }
    Must(memory_.Unmap(*reservation, Bytes(region_[g]), Bytes(handles.size() * kChunk)),
         "D-033: unmap");
    handoff_.insert(handoff_.end(), handles.rbegin(), handles.rend());
    handles.clear();
  }

  void ReleaseHandoff() {
    for (const BackingId handle : handoff_) {
      Must(memory_.Release(handle), "D-033: release");
      --handles_;
    }
    handoff_.clear();
  }

  std::span<const Group> groups_;
  CountingMemory& memory_;
  std::vector<std::uint64_t> region_;  // offset in the model's range
  std::vector<std::vector<BackingId>> mapped_;
  std::vector<std::optional<ReservationId>> reservation_;
  std::vector<std::uint64_t> extent_;
  std::vector<BackingId> handoff_;
  std::uint64_t handles_ = 0;  // mapped and in handoff
};

// ------------------------------------------------------------------ slabs

class Slabs final : public Design {
 public:
  Slabs(std::span<const Group> groups, std::uint64_t budget, std::uint64_t slab, HolePolicy policy,
        CountingMemory& memory)
      : groups_(groups),
        memory_(memory),
        slab_(slab),
        policy_(policy),
        slots_(4 * budget / slab),
        where_(groups.size()),
        class_of_(groups.size(), kGeneral),
        stamp_of_(groups.size(), 0) {
    Check(slab % kChunk == 0 && budget % slab == 0 && !slots_.empty(), "slabs: geometry");
    std::map<std::uint64_t, std::int32_t> classes;
    for (GroupId g = 0; g < groups.size(); ++g) {
      const bool classed = (policy == HolePolicy::kSizeClasses && groups[g].stored <= slab) ||
                           (policy == HolePolicy::kHybrid && groups[g].expert);
      if (!classed) {
        continue;
      }
      Check(groups[g].stored <= slab, "slabs: a classed group exceeds a slab");
      const auto [it, added] =
          classes.emplace(groups[g].stored, static_cast<std::int32_t>(class_size_.size()));
      if (added) {
        class_size_.push_back(groups[g].stored);
      }
      class_of_[g] = it->second;
    }
    class_open_.resize(class_size_.size());
    class_lru_.resize(class_size_.size());
    arena_ = Must(memory_.Reserve(Bytes(slots_.size() * slab_)), "slabs: reserve");
  }

  void Drop(GroupId g) override { Remove(g); }

  bool Admit(std::span<const GroupId> missing, Env& env) override {
    for (const GroupId g : missing) {
      Check(!where_[g].placed, "slabs: restoring a placed group");
      const bool placed = class_of_[g] == kGeneral ? PlaceGeneral(g, env) : PlaceClass(g, env);
      if (!placed) {
        return false;
      }
    }
    return true;
  }

  bool Shrink(Env& env) override {
    while (held_ > env.Limit() / slab_) {
      if (const std::optional<std::uint64_t> empty = Empty(/*highest=*/true)) {
        ReleaseSlab(*empty);
        continue;
      }
      if (policy_ == HolePolicy::kCompaction && EmptyByRelocation(env)) {
        continue;
      }
      if (EvictSlab(env)) {
        continue;
      }
      const std::optional<GroupId> victim = env.LruVictim();
      if (!victim) {
        return false;
      }
      Evict(*victim, env);
    }
    return true;
  }

  void Touched(GroupId g, std::uint64_t /*before*/, std::uint64_t after) override {
    const std::int32_t c = class_of_[g];
    if (c != kGeneral && where_[g].placed) {
      auto& lru = class_lru_[static_cast<std::size_t>(c)];
      Check(lru.erase({stamp_of_[g], g}) == 1, "slabs: class recency out of step");
      stamp_of_[g] = after;
      lru.emplace(after, g);
    }
  }

  std::uint64_t Held() const override { return held_ * slab_; }

  std::uint64_t PositionOf(GroupId g) const override {
    Check(where_[g].placed, "slabs: position of an unplaced group");
    return where_[g].offset;
  }

  void CheckInvariants() const override;

  void Teardown() override {
    general_.clear();
    class_slabs_.clear();
    for (std::uint64_t k = 0; k < slots_.size(); ++k) {
      if (slots_[k].held) {
        slots_[k].users = 0;
        ReleaseSlab(k);
      }
    }
    Must(memory_.Free(arena_), "slabs: free");
  }

 private:
  struct Slot {
    bool held = false;
    BackingId backing;
    std::int32_t cls = kGeneral;  // a size class, or general space
    std::uint32_t users = 0;      // general groups over it, or occupied class slots
  };
  struct Place {
    std::uint64_t offset = 0;
    bool placed = false;
  };
  struct ClassSlab {
    std::vector<GroupId> slot;
    std::uint32_t free = 0;
  };
  struct Run {
    std::uint64_t lo = 0;  // bytes
    std::uint64_t hi = 0;
  };
  struct Hole {
    std::uint64_t start = 0;
    std::uint64_t len = 0;
  };

  std::uint64_t Size(GroupId g) const { return groups_[g].stored; }
  std::uint64_t First(std::uint64_t offset) const { return offset / slab_; }
  std::uint64_t Last(std::uint64_t offset, std::uint64_t len) const {
    return (offset + len - 1) / slab_;
  }
  bool HeldGeneral(std::uint64_t k) const { return slots_[k].held && slots_[k].cls == kGeneral; }
  bool IsEmpty(std::uint64_t k) const { return HeldGeneral(k) && slots_[k].users == 0; }
  std::uint64_t Capacity(const Env& env) const { return env.Limit() / slab_; }

  // ---- backing

  void CreateSlab(std::uint64_t k, const Env& env) {
    Check(!slots_[k].held && held_ + 1 <= Capacity(env), "slabs: creating beyond the budget");
    const BackingId backing = Must(memory_.Create(kHostClass, Bytes(slab_)), "slabs: create");
    Must(memory_.Map(arena_, Bytes(k * slab_), backing), "slabs: map");
    Must(memory_.SetAccess(arena_, Bytes(k * slab_), Bytes(slab_), Access::kReadWrite),
         "slabs: set access");
    memory_.Count(Call::kRegister, k, 0);  // registered for direct reads for its lifetime
    slots_[k] = Slot{.held = true, .backing = backing, .cls = kGeneral, .users = 0};
    ++held_;
  }

  void ReleaseSlab(std::uint64_t k) {
    Check(slots_[k].held && slots_[k].users == 0, "slabs: releasing a slab in use");
    memory_.Count(Call::kUnregister, k, 0);
    Must(memory_.Unmap(arena_, Bytes(k * slab_), Bytes(slab_)), "slabs: unmap");
    Must(memory_.Release(slots_[k].backing), "slabs: release");
    slots_[k] = Slot{};
    --held_;
  }

  // The lowest (or highest) held slab in general space with nothing on it.
  std::optional<std::uint64_t> Empty(bool highest) const {
    if (highest) {
      for (std::uint64_t k = slots_.size(); k-- > 0;) {
        if (IsEmpty(k)) {
          return k;
        }
      }
      return std::nullopt;
    }
    for (std::uint64_t k = 0; k < slots_.size(); ++k) {
      if (IsEmpty(k)) {
        return k;
      }
    }
    return std::nullopt;
  }

  // ---- placements

  void AddGeneral(GroupId g, std::uint64_t offset) {
    Check(offset % kFileAlign == 0, "slabs: a group is not 4 KiB aligned");
    Check(general_.emplace(offset, g).second, "slabs: two groups at one offset");
    where_[g] = Place{.offset = offset, .placed = true};
    for (std::uint64_t k = First(offset); k <= Last(offset, Size(g)); ++k) {
      Check(HeldGeneral(k), "slabs: a group on backing that is not held general space");
      ++slots_[k].users;
    }
  }

  void AddClass(GroupId g, std::uint64_t k, std::size_t index) {
    const auto c = static_cast<std::size_t>(class_of_[g]);
    ClassSlab& slab = class_slabs_.at(k);
    Check(slab.slot[index] == kNoGroup, "slabs: class slot taken");
    slab.slot[index] = g;
    --slab.free;
    ++slots_[k].users;
    if (slab.free == 0) {
      class_open_[c].erase(k);
    }
    where_[g] = Place{.offset = (k * slab_) + (index * class_size_[c]), .placed = true};
  }

  void Remove(GroupId g) {
    Place& place = where_[g];
    Check(place.placed, "slabs: removing an unplaced group");
    const std::int32_t cls = class_of_[g];
    if (cls == kGeneral) {
      Check(general_.erase(place.offset) == 1, "slabs: general map out of step");
      for (std::uint64_t k = First(place.offset); k <= Last(place.offset, Size(g)); ++k) {
        --slots_[k].users;
      }
    } else {
      const auto c = static_cast<std::size_t>(cls);
      const std::uint64_t k = First(place.offset);
      ClassSlab& slab = class_slabs_.at(k);
      const std::size_t index = (place.offset - (k * slab_)) / class_size_[c];
      Check(slab.slot[index] == g, "slabs: class slot out of step");
      slab.slot[index] = kNoGroup;
      ++slab.free;
      --slots_[k].users;
      class_open_[c].insert(k);
      class_lru_[c].erase({stamp_of_[g], g});
      if (slots_[k].users == 0) {  // the slab returns to general space, empty
        class_slabs_.erase(k);
        class_open_[c].erase(k);
        slots_[k].cls = kGeneral;
      }
    }
    place = Place{};
  }

  void Evict(GroupId g, Env& env) {
    Check(env.Evictable(g), "slabs: evicting a protected group");
    Remove(g);
    env.Evicted(g);
  }

  // ---- general space

  // Maximal runs of held general slabs.
  std::vector<Run> HeldRuns() const {
    std::vector<Run> runs;
    for (std::uint64_t k = 0; k < slots_.size();) {
      if (!HeldGeneral(k)) {
        ++k;
        continue;
      }
      std::uint64_t end = k;
      while (end < slots_.size() && HeldGeneral(end)) {
        ++end;
      }
      runs.push_back(Run{.lo = k * slab_, .hi = end * slab_});
      k = end;
    }
    return runs;
  }

  // The free gaps between general groups inside [lo, hi).
  template <typename F>
  void ForEachGap(std::uint64_t lo, std::uint64_t hi, F&& visit) const {
    std::uint64_t cursor = lo;
    for (auto it = general_.lower_bound(lo); it != general_.end() && it->first < hi; ++it) {
      if (it->first > cursor) {
        visit(Hole{.start = cursor, .len = it->first - cursor});
      }
      cursor = it->first + Size(it->second);
    }
    if (hi > cursor) {
      visit(Hole{.start = cursor, .len = hi - cursor});
    }
  }

  std::vector<Hole> HeldHoles() const {
    std::vector<Hole> holes;
    for (const Run& run : HeldRuns()) {
      ForEachGap(run.lo, run.hi, [&](const Hole& hole) { holes.push_back(hole); });
    }
    return holes;
  }

  // Best fit in held general space: the smallest hole that fits, lowest
  // address first.
  std::optional<std::uint64_t> FitHeld(std::uint64_t len) const {
    std::optional<Hole> best;
    for (const Hole& hole : HeldHoles()) {
      if (hole.len >= len && (!best || hole.len < best->len)) {
        best = hole;
      }
    }
    if (!best) {
      return std::nullopt;
    }
    return best->start;
  }

  // Placement that needs new slabs: the fewest new slabs, then the lowest
  // address. Empty slabs outside the window are released first if the
  // budget needs their room. Makes the slabs; returns the offset.
  std::optional<std::uint64_t> Grow(std::uint64_t len, const Env& env) {
    const std::uint64_t capacity = Capacity(env);
    Check(held_ <= capacity, "slabs: held beyond the budget");
    std::uint64_t empties = 0;
    for (std::uint64_t k = 0; k < slots_.size(); ++k) {
      empties += IsEmpty(k) ? 1 : 0;
    }
    if (held_ == capacity && empties == 0) {
      return std::nullopt;  // no room for a new slab
    }
    std::optional<std::pair<std::uint64_t, std::uint64_t>> best;  // (new slabs, offset)
    auto consider = [&](std::uint64_t p) {
      std::uint64_t fresh = 0;
      std::uint64_t empty_inside = 0;
      for (std::uint64_t k = First(p); k <= Last(p, len); ++k) {
        if (!slots_[k].held) {
          ++fresh;
        } else if (slots_[k].users == 0) {
          ++empty_inside;
        }
      }
      if (fresh == 0 || fresh > capacity - held_ + (empties - empty_inside)) {
        return;
      }
      if (!best || std::pair(fresh, p) < *best) {
        best = std::pair(fresh, p);
      }
    };
    auto held = [&](std::uint64_t k) { return slots_[k].held; };
    for (std::uint64_t k = 0; k < slots_.size();) {
      if (slots_[k].held && slots_[k].cls != kGeneral) {
        ++k;
        continue;
      }
      std::uint64_t end = k;
      while (end < slots_.size() && (!slots_[end].held || slots_[end].cls == kGeneral)) {
        ++end;
      }
      ForEachGap(k * slab_, end * slab_, [&](const Hole& gap) {
        if (gap.len < len) {
          return;
        }
        const std::uint64_t last = gap.start + gap.len - len;  // the highest start
        consider(gap.start);
        consider(last);
        const std::uint64_t aligned = CeilDiv(gap.start, slab_) * slab_;
        if (aligned <= last) {
          consider(aligned);
        }
        for (std::uint64_t q = aligned; q < gap.start + gap.len; q += slab_) {
          const std::uint64_t s = q / slab_;
          if (s == 0 || held(s - 1) != held(s)) {
            if (q <= last) {
              consider(q);
            }
            if (q >= gap.start + len) {
              consider(q - len);
            }
          }
        }
      });
      k = end;
    }
    if (!best) {
      return std::nullopt;
    }
    const auto [fresh, p] = *best;
    std::uint64_t release = fresh > capacity - held_ ? fresh - (capacity - held_) : 0;
    for (std::uint64_t k = slots_.size(); release > 0 && k-- > 0;) {
      if (IsEmpty(k) && (k < First(p) || k > Last(p, len))) {
        ReleaseSlab(k);
        --release;
      }
    }
    Check(release == 0, "slabs: growth could not release the empty slabs it counted");
    for (std::uint64_t k = First(p); k <= Last(p, len); ++k) {
      if (!slots_[k].held) {
        CreateSlab(k, env);
      }
    }
    return p;
  }

  // The window [p, p + len) over held general space with the best score
  // among those whose groups `allowed` accepts, groups in address order.
  // `score` ranks (max stamp, bytes, p) for eviction; compaction collects.
  struct Window {
    std::uint64_t p = 0;
    std::uint64_t newest = 0;
    std::uint64_t bytes = 0;
    std::vector<GroupId> groups;
  };
  template <typename Allowed, typename Visit>
  void ForEachWindow(std::uint64_t len, const Env& env, Allowed&& allowed, Visit&& visit) const {
    for (const Run& run : HeldRuns()) {
      if (run.hi - run.lo < len) {
        continue;
      }
      std::vector<GroupId> gs;
      for (auto it = general_.lower_bound(run.lo); it != general_.end() && it->first < run.hi;
           ++it) {
        gs.push_back(it->second);
      }
      std::vector<std::uint64_t> starts{run.lo, run.hi - len};
      for (const GroupId g : gs) {
        starts.push_back(where_[g].offset);
        starts.push_back(where_[g].offset + Size(g));
      }
      std::ranges::sort(starts);
      const auto [dup_begin, dup_end] = std::ranges::unique(starts);
      starts.erase(dup_begin, dup_end);
      std::size_t lo = 0;
      std::size_t hi = 0;
      std::size_t blocked = 0;
      std::uint64_t bytes = 0;
      std::deque<std::size_t> newest;  // indices, stamps decreasing
      for (const std::uint64_t p : starts) {
        if (p < run.lo || p + len > run.hi) {
          continue;
        }
        while (hi < gs.size() && where_[gs[hi]].offset < p + len) {
          blocked += allowed(gs[hi]) ? 0 : 1;
          bytes += Size(gs[hi]);
          while (!newest.empty() && env.Stamp(gs[newest.back()]) <= env.Stamp(gs[hi])) {
            newest.pop_back();
          }
          newest.push_back(hi);
          ++hi;
        }
        while (lo < hi && where_[gs[lo]].offset + Size(gs[lo]) <= p) {
          blocked -= allowed(gs[lo]) ? 0 : 1;
          bytes -= Size(gs[lo]);
          if (!newest.empty() && newest.front() == lo) {
            newest.pop_front();
          }
          ++lo;
        }
        if (blocked == 0 && lo < hi) {
          visit(p, env.Stamp(gs[newest.front()]), bytes, std::span(gs).subspan(lo, hi - lo));
        }
      }
    }
  }

  // Contiguous-run eviction: the window whose newest group is oldest, then
  // the fewest bytes, then the lowest address. Evicts it and places g.
  bool EvictRun(GroupId g, Env& env) {
    std::optional<Window> best;
    ForEachWindow(
        Size(g), env, [&](GroupId v) { return env.Evictable(v); },
        [&](std::uint64_t p, std::uint64_t newest, std::uint64_t bytes,
            std::span<const GroupId> victims) {
          if (!best ||
              std::tuple(newest, bytes, p) < std::tuple(best->newest, best->bytes, best->p)) {
            best = Window{.p = p,
                          .newest = newest,
                          .bytes = bytes,
                          .groups = std::vector(victims.begin(), victims.end())};
          }
        });
    if (!best) {
      return false;
    }
    for (const GroupId v : best->groups) {
      Evict(v, env);
    }
    AddGeneral(g, best->p);
    return true;
  }

  // Best fit of `moved` (largest first, then by id) into `holes`, as
  // (group, offset) pairs; empty if any does not fit.
  std::optional<std::vector<std::pair<GroupId, std::uint64_t>>> Fit(
      std::vector<GroupId> moved, const std::vector<Hole>& holes) const {
    std::ranges::sort(moved, [&](GroupId a, GroupId b) {
      return Size(a) != Size(b) ? Size(a) > Size(b) : a < b;
    });
    std::set<std::pair<std::uint64_t, std::uint64_t>> free;  // (len, start)
    for (const Hole& hole : holes) {
      free.emplace(hole.len, hole.start);
    }
    std::vector<std::pair<GroupId, std::uint64_t>> plan;
    for (const GroupId m : moved) {
      const auto it = free.lower_bound({Size(m), 0});
      if (it == free.end()) {
        return std::nullopt;
      }
      const auto [len, start] = *it;
      free.erase(it);
      plan.emplace_back(m, start);
      if (len > Size(m)) {
        free.emplace(len - Size(m), start + Size(m));
      }
    }
    return plan;
  }

  // Held holes with [lo, hi) cut out.
  std::vector<Hole> HolesOutside(std::uint64_t lo, std::uint64_t hi) const {
    std::vector<Hole> out;
    for (const Hole& hole : HeldHoles()) {
      const std::uint64_t end = hole.start + hole.len;
      if (end <= lo || hole.start >= hi) {
        out.push_back(hole);
        continue;
      }
      if (hole.start < lo) {
        out.push_back(Hole{.start = hole.start, .len = lo - hole.start});
      }
      if (end > hi) {
        out.push_back(Hole{.start = hi, .len = end - hi});
      }
    }
    return out;
  }

  void Relocate(const std::vector<std::pair<GroupId, std::uint64_t>>& plan, Env& env) {
    for (const auto& [m, to] : plan) {
      Remove(m);
    }
    for (const auto& [m, to] : plan) {
      AddGeneral(m, to);
      env.Relocated(m);
    }
  }

  bool Relocatable(GroupId g, const Env& env) const {
    return !groups_[g].expert && env.Evictable(g);  // dense groups only at M2
  }

  // Compaction: the window of free space and relocatable dense groups with
  // the fewest bytes to move (then the lowest address) whose groups fit
  // in holes outside it. Moves them and places g.
  bool Compact(GroupId g, Env& env) {
    std::vector<Window> candidates;
    ForEachWindow(
        Size(g), env, [&](GroupId v) { return Relocatable(v, env); },
        [&](std::uint64_t p, std::uint64_t /*newest*/, std::uint64_t bytes,
            std::span<const GroupId> moved) {
          candidates.push_back(
              Window{.p = p, .bytes = bytes, .groups = std::vector(moved.begin(), moved.end())});
        });
    if (candidates.empty()) {
      return false;
    }
    std::ranges::sort(candidates, [](const Window& a, const Window& b) {
      return std::pair(a.bytes, a.p) < std::pair(b.bytes, b.p);
    });
    std::uint64_t largest_hole = 0;
    for (const Hole& hole : HeldHoles()) {
      largest_hole = std::max(largest_hole, hole.len);
    }
    for (const Window& window : candidates) {
      const auto largest = std::ranges::max(window.groups, {}, [&](GroupId m) { return Size(m); });
      if (Size(largest) > largest_hole) {
        continue;
      }
      const auto plan = Fit(window.groups, HolesOutside(window.p, window.p + Size(g)));
      if (!plan) {
        continue;
      }
      Relocate(*plan, env);
      AddGeneral(g, window.p);
      return true;
    }
    return false;
  }

  // The groups on slab k.
  std::vector<GroupId> OnSlab(std::uint64_t k) const {
    std::vector<GroupId> out;
    if (slots_[k].cls != kGeneral) {
      for (const GroupId g : class_slabs_.at(k).slot) {
        if (g != kNoGroup) {
          out.push_back(g);
        }
      }
      return out;
    }
    const std::uint64_t lo = k * slab_;
    auto it = general_.lower_bound(lo);
    if (it != general_.begin()) {
      const auto before = std::prev(it);
      if (before->first + Size(before->second) > lo) {
        out.push_back(before->second);
      }
    }
    for (; it != general_.end() && it->first < lo + slab_; ++it) {
      out.push_back(it->second);
    }
    return out;
  }

  // The slab whose newest group is oldest (then fewest bytes, then lowest
  // index), among those whose groups are all evictable: (newest, bytes, k).
  std::optional<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>> OldestSlab(
      const Env& env) const {
    std::optional<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>> best;
    for (std::uint64_t k = 0; k < slots_.size(); ++k) {
      if (!slots_[k].held || slots_[k].users == 0) {
        continue;
      }
      std::uint64_t newest = 0;
      std::uint64_t bytes = 0;
      bool ok = true;
      for (const GroupId v : OnSlab(k)) {
        if (!env.Evictable(v)) {
          ok = false;
          break;
        }
        newest = std::max(newest, env.Stamp(v));
        bytes += Size(v);
      }
      if (ok && (!best || std::tuple(newest, bytes, k) < *best)) {
        best = std::tuple(newest, bytes, k);
      }
    }
    return best;
  }

  // Empties the oldest slab (OldestSlab).
  bool EvictSlab(Env& env) {
    const auto best = OldestSlab(env);
    if (!best) {
      return false;
    }
    for (const GroupId v : OnSlab(std::get<2>(*best))) {
      Evict(v, env);
    }
    return true;
  }

  // Compaction's shrink: empties the general slab whose dense groups are
  // cheapest to move into holes elsewhere.
  bool EmptyByRelocation(Env& env) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> candidates;  // (bytes, slab)
    for (std::uint64_t k = 0; k < slots_.size(); ++k) {
      if (!HeldGeneral(k) || slots_[k].users == 0) {
        continue;
      }
      std::uint64_t bytes = 0;
      bool ok = true;
      for (const GroupId v : OnSlab(k)) {
        ok = ok && Relocatable(v, env);
        bytes += Size(v);
      }
      if (ok) {
        candidates.emplace_back(bytes, k);
      }
    }
    std::ranges::sort(candidates);
    for (const auto& [bytes, k] : candidates) {
      const auto plan = Fit(OnSlab(k), HolesOutside(k * slab_, (k + 1) * slab_));
      if (plan) {
        Relocate(*plan, env);
        return true;
      }
    }
    return false;
  }

  bool PlaceGeneral(GroupId g, Env& env) {
    while (true) {
      if (const std::optional<std::uint64_t> p = FitHeld(Size(g))) {
        AddGeneral(g, *p);
        return true;
      }
      if (const std::optional<std::uint64_t> p = Grow(Size(g), env)) {
        AddGeneral(g, *p);
        return true;
      }
      if (policy_ == HolePolicy::kCompaction && Compact(g, env)) {
        return true;
      }
      if (EvictRun(g, env)) {
        return true;
      }
      // With class slabs, general space grows only by a slab emptied
      // whole: the oldest, rather than LRU victims scattered over slabs.
      if (!class_slabs_.empty() && EvictSlab(env)) {
        continue;
      }
      const std::optional<GroupId> victim = env.LruVictim();
      if (!victim) {
        return false;
      }
      Evict(*victim, env);
    }
  }

  bool PlaceClass(GroupId g, Env& env) {
    const auto c = static_cast<std::size_t>(class_of_[g]);
    stamp_of_[g] = env.Stamp(g);
    while (true) {
      if (!class_open_[c].empty()) {
        const std::uint64_t k = *class_open_[c].begin();
        const std::vector<GroupId>& slot = class_slabs_.at(k).slot;
        const auto index =
            static_cast<std::size_t>(std::ranges::find(slot, kNoGroup) - slot.begin());
        AddClass(g, k, index);
        class_lru_[c].emplace(stamp_of_[g], g);
        return true;
      }
      std::optional<std::uint64_t> k = Empty(/*highest=*/false);
      if (!k && held_ < Capacity(env)) {
        for (std::uint64_t s = 0; s < slots_.size(); ++s) {
          if (!slots_[s].held) {
            CreateSlab(s, env);
            k = s;
            break;
          }
        }
      }
      if (k) {
        slots_[*k].cls = static_cast<std::int32_t>(c);
        const auto count = static_cast<std::size_t>(slab_ / class_size_[c]);
        class_slabs_.emplace(*k, ClassSlab{.slot = std::vector<GroupId>(count, kNoGroup),
                                           .free = static_cast<std::uint32_t>(count)});
        class_open_[c].insert(*k);
        continue;
      }
      // Its own class's least recently used evictable group frees a slot,
      // unless a whole slab (of any class, or general space) holds only
      // older groups: then that slab is emptied for the class, so that
      // extra victims keep the reference's recency order across classes.
      std::optional<GroupId> victim;
      for (const auto& [stamp, v] : class_lru_[c]) {
        if (env.Evictable(v)) {
          victim = v;
          break;
        }
      }
      const auto oldest = OldestSlab(env);
      if (victim && (!oldest || env.Stamp(*victim) <= std::get<0>(*oldest))) {
        Evict(*victim, env);
        continue;
      }
      if (EvictSlab(env)) {
        continue;
      }
      victim = env.LruVictim();
      if (!victim) {
        return false;
      }
      Evict(*victim, env);
    }
  }

  std::span<const Group> groups_;
  CountingMemory& memory_;
  std::uint64_t slab_;
  HolePolicy policy_;
  ReservationId arena_;
  std::vector<Slot> slots_;
  std::uint64_t held_ = 0;
  std::map<std::uint64_t, GroupId> general_;  // offset -> group
  std::vector<Place> where_;
  std::vector<std::int32_t> class_of_;
  std::vector<std::uint64_t> class_size_;
  std::map<std::uint64_t, ClassSlab> class_slabs_;   // by slab index
  std::vector<std::set<std::uint64_t>> class_open_;  // slabs with a free slot, per class
  std::vector<std::set<std::pair<std::uint64_t, GroupId>>> class_lru_;
  std::vector<std::uint64_t> stamp_of_;  // class groups' stamps in class_lru_
};

void Slabs::CheckInvariants() const {
  std::vector<std::uint32_t> users(slots_.size(), 0);
  std::uint64_t end = 0;
  for (const auto& [offset, g] : general_) {
    Check(where_[g].placed && where_[g].offset == offset, "slabs: general placement record");
    Check(offset % kFileAlign == 0 && offset % 256 == 0, "slabs: alignment");
    Check(offset >= end, "slabs: overlapping groups");
    end = offset + Size(g);
    Check(Last(offset, Size(g)) < slots_.size(), "slabs: a group beyond the arena");
    for (std::uint64_t k = First(offset); k <= Last(offset, Size(g)); ++k) {
      Check(HeldGeneral(k), "slabs: a group off held general space");
      ++users[k];
    }
  }
  std::uint64_t held = 0;
  for (std::uint64_t k = 0; k < slots_.size(); ++k) {
    held += slots_[k].held ? 1 : 0;
    if (slots_[k].cls == kGeneral) {
      Check(users[k] == slots_[k].users, "slabs: general users out of step");
      continue;
    }
    Check(slots_[k].held, "slabs: a class on a slab that is not held");
    const auto c = static_cast<std::size_t>(slots_[k].cls);
    const ClassSlab& slab = class_slabs_.at(k);
    std::uint32_t occupied = 0;
    for (std::size_t i = 0; i < slab.slot.size(); ++i) {
      const GroupId g = slab.slot[i];
      if (g == kNoGroup) {
        continue;
      }
      ++occupied;
      Check(class_of_[g] == slots_[k].cls && where_[g].placed &&
                where_[g].offset == (k * slab_) + (i * class_size_[c]),
            "slabs: class placement record");
      Check(where_[g].offset % kFileAlign == 0, "slabs: class alignment");
      Check(class_lru_[c].contains({stamp_of_[g], g}), "slabs: class recency");
    }
    Check(occupied > 0 && occupied == slots_[k].users && slab.free + occupied == slab.slot.size() &&
              (slab.free > 0) == class_open_[c].contains(k) &&
              slab.slot.size() * class_size_[c] <= slab_,
          "slabs: class slab counts");
  }
  Check(held == held_, "slabs: held count");
  Check(class_slabs_.size() <= held_, "slabs: class slabs");
}

}  // namespace

// ------------------------------------------------------------------ counting

std::string_view CallName(Call call) {
  static constexpr std::array<std::string_view, kCalls> kNames = {
      "reserve",    "free",  "create",   "release",   "map",
      "set_access", "unmap", "register", "unregister"};
  return kNames[static_cast<std::size_t>(call)];
}

void CountingMemory::Count(Call call, std::uint64_t a, std::uint64_t b) {
  if (!frozen_) {
    ++counts_[static_cast<std::size_t>(call)];
    Digest(static_cast<std::uint64_t>(call), a, b);
  }
}

void CountingMemory::Digest(std::uint64_t a, std::uint64_t b, std::uint64_t c) {
  std::array<std::uint64_t, 3> words = {a, b, c};
  digest_.Update(std::as_bytes(std::span(words)));
}

std::expected<providers::ReservationId, providers::Failure> CountingMemory::Reserve(
    base::Bytes size) {
  auto result = inner_.Reserve(size);
  Check(result.has_value(), "replay: reserve failed");
  Count(Call::kReserve, size.value(), 0);
  return result;
}

std::expected<void, providers::Failure> CountingMemory::Free(providers::ReservationId reservation) {
  auto result = inner_.Free(reservation);
  Check(result.has_value(), "replay: free failed");
  Count(Call::kFree, reservation.index(), 0);
  return result;
}

std::expected<providers::BackingId, providers::Failure> CountingMemory::Create(
    std::size_t allocation_class, base::Bytes size) {
  auto result = inner_.Create(allocation_class, size);
  Check(result.has_value(), "replay: create failed");
  Count(Call::kCreate, size.value(), 0);
  return result;
}

std::expected<void, providers::Failure> CountingMemory::Release(providers::BackingId backing) {
  auto result = inner_.Release(backing);
  Check(result.has_value(), "replay: release failed");
  Count(Call::kRelease, backing.index(), 0);
  return result;
}

std::expected<void, providers::Failure> CountingMemory::Map(providers::ReservationId reservation,
                                                            base::Bytes offset,
                                                            providers::BackingId backing) {
  auto result = inner_.Map(reservation, offset, backing);
  Check(result.has_value(), "replay: map failed");
  Count(Call::kMap, offset.value(), backing.index());
  return result;
}

std::expected<void, providers::Failure> CountingMemory::SetAccess(
    providers::ReservationId reservation, base::Bytes offset, base::Bytes size,
    providers::Access access) {
  auto result = inner_.SetAccess(reservation, offset, size, access);
  Check(result.has_value(), "replay: set access failed");
  Count(Call::kSetAccess, offset.value(), size.value());
  return result;
}

std::expected<void, providers::Failure> CountingMemory::Unmap(providers::ReservationId reservation,
                                                              base::Bytes offset,
                                                              base::Bytes size) {
  auto result = inner_.Unmap(reservation, offset, size);
  Check(result.has_value(), "replay: unmap failed");
  Count(Call::kUnmap, offset.value(), size.value());
  return result;
}

// ------------------------------------------------------------------ designs

std::vector<DesignSpec> AllDesigns() {
  std::vector<DesignSpec> designs{DesignSpec{.name = "d033", .baseline = true}};
  constexpr std::array<std::pair<std::uint64_t, std::string_view>, 3> kSlabs = {
      std::pair{std::uint64_t{32} << 20U, "32m"}, std::pair{std::uint64_t{256} << 20U, "256m"},
      std::pair{std::uint64_t{1} << 30U, "1g"}};
  constexpr std::array<std::pair<HolePolicy, std::string_view>, 4> kPolicies = {
      std::pair{HolePolicy::kContiguousRun, "run"}, std::pair{HolePolicy::kSizeClasses, "class"},
      std::pair{HolePolicy::kHybrid, "hybrid"}, std::pair{HolePolicy::kCompaction, "compact"}};
  for (const auto& [slab, slab_name] : kSlabs) {
    for (const auto& [policy, policy_name] : kPolicies) {
      designs.push_back(DesignSpec{.name = std::format("slab{}-{}", slab_name, policy_name),
                                   .slab = slab,
                                   .policy = policy});
    }
  }
  return designs;
}

std::optional<DesignSpec> FindDesign(std::string_view name) {
  for (DesignSpec& spec : AllDesigns()) {
    if (spec.name == name) {
      return std::move(spec);
    }
  }
  return std::nullopt;
}

std::unique_ptr<Design> MakeDesign(const DesignSpec& spec, std::span<const Group> groups,
                                   std::span<const Model> models, std::uint64_t budget,
                                   CountingMemory& memory) {
  if (spec.baseline) {
    return std::make_unique<Handles>(groups, models, memory);
  }
  return std::make_unique<Slabs>(groups, budget, spec.slab, spec.policy, memory);
}

}  // namespace llmp::rb
