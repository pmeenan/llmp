// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A runner's plans by key, whatever its kernels (docs/engine.md): the
// cache every runner keeps per kind of plan, its charge to the node's
// account (plan_account.h) and its plans and graphs as the reclaim order's
// candidates. `Runs` is a plan's runs (StepPlanRuns for step programs,
// step_runs.h): it has_graph(), and holds that graph's `seconds` (what
// capturing it again costs), its `measured_bytes` and DropGraph().
// Part of the engine core.

#ifndef LLMP_ENGINE_PLAN_CACHE_H_
#define LLMP_ENGINE_PLAN_CACHE_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "engine/plan_account.h"
#include "memory/reclaim.h"

namespace llmp::engine {

// What a captured graph holds outside the catalog (the driver's host and
// device memory) for each node its plan launches: an allowance over the
// largest measured on GB10, a 4-slot DeepSeek decode wave graph's
// MemAvailable drop of up to 56 MiB at 4,912 launched nodes (11.7 KiB a
// node; the same report), with about 37% headroom.
inline constexpr std::uint64_t kGraphNodeHostBytes = 16384;

// A model's planned shapes, by key (a shape and what the model computes
// beside it), each with its runs: `Variants` of them where one plan runs
// with different outputs (a Qwen3.8 verify with and without its logits'
// copy). No fixed number is kept: plans and their graphs grow into the
// memory the node's budget leaves free and go only when the node reclaims
// them (PlanAccount, the reclaim order) or the runner forgets them all
// (Clear). Each is charged to `account` as it is added (PlannedHostBytes)
// and each graph before its capture (ChargeGraph; kGraphNodeHostBytes a
// node its plan launches), and uncharged as it goes. Entries never move:
// one stays put until it is dropped, and a step's own (used since its
// PlanStep began) are never reclaimed. Found by an index of the keys'
// hashes (`Hash`), each candidate's whole key compared only on a match.
template <typename Key, typename Planned, std::size_t Variants, typename Runs,
          typename Hash = std::hash<Key>>
class BasicPlanCache final : public PlanCacheBase {
 public:
  static constexpr std::uint64_t kNever = std::numeric_limits<std::uint64_t>::max();
  struct Entry {
    Key key{};
    std::unique_ptr<Planned> planned;
    std::array<Runs, Variants> runs;
    std::uint64_t host_bytes = 0;     // PlannedHostBytes, as the runner counted it
    std::uint64_t nodes = 0;          // PlannedNodes: its graphs' cost (kGraphNodeHostBytes each)
    std::uint64_t used = 0;           // NextPlanUse at its last Find or Add
    memory::ReclaimStamp stamp;       // the reclaim order's at its last Find or Add
    std::uint32_t graph_skip = 0;     // uses left before a refused graph charge is asked again
    std::uint32_t graph_backoff = 0;  // the last such wait (doubling)
    std::uint64_t serial = 0;         // its identity among the process's plans
    double seconds = 0;               // its planning, measured: what planning it again costs
    std::array<bool, Variants> charged{};  // each variant's graph charged

    bool has_graph() const {
      return std::ranges::any_of(runs, [](const Runs& r) { return r.has_graph(); });
    }
    std::uint64_t graph_count() const {
      return static_cast<std::uint64_t>(
          std::ranges::count_if(runs, [](const Runs& r) { return r.has_graph(); }));
    }
    std::uint64_t graph_bytes() const { return graph_count() * nodes * kGraphNodeHostBytes; }
    double graph_seconds() const {
      double s = 0;
      for (const Runs& r : runs) {
        s += r.has_graph() ? r.seconds : 0;
      }
      return s;
    }
  };

  explicit BasicPlanCache(PlanAccount* account = nullptr) : account_(account) {}
  ~BasicPlanCache() override = default;
  BasicPlanCache(const BasicPlanCache&) = delete;
  BasicPlanCache& operator=(const BasicPlanCache&) = delete;
  BasicPlanCache(BasicPlanCache&&) = delete;
  BasicPlanCache& operator=(BasicPlanCache&&) = delete;
  void set_account(PlanAccount* account) { account_ = account; }

  Entry* Find(const Key& key) {
    const auto [first, last] = index_.equal_range(Hash{}(key));
    for (auto it = first; it != last; ++it) {
      Entry* e = it->second;
      if (e->key == key) {
        Settle(*e);
        e->used = NextPlanUse();
        e->stamp = memory::StampUse();
        return e;
      }
    }
    return nullptr;
  }
  // Its plan's bytes charged first (required: the step under way needs
  // it), which may reclaim others not in use.
  Entry& Add(Key key, std::unique_ptr<Planned> planned, std::uint64_t host_bytes = 0,
             std::uint64_t nodes = 0, double seconds = 0) {
    Settle();
    if (account_ != nullptr) {
      (void)account_->Charge(host_bytes, true);
    }
    auto entry = std::make_unique<Entry>();
    entry->key = std::move(key);
    entry->planned = std::move(planned);
    entry->host_bytes = host_bytes;
    entry->nodes = nodes;
    entry->seconds = seconds;
    entry->serial = NextPlanUse();
    entry->used = NextPlanUse();
    entry->stamp = memory::StampUse();
    index_.emplace(Hash{}(entry->key), entry.get());
    return *entries_.emplace_back(std::move(entry));
  }
  // Before capturing variant `variant` of `entry`'s plan: its graph's
  // bytes charged; false (not charged) if they do not fit even after the
  // node reclaimed what it could: the plan then runs launch by launch. A
  // refused charge is not asked again at once: the next 1, 2, 4 … up to
  // kGraphRetryMost uses of the plan skip it, so a full budget is not
  // searched for room at every step.
  static constexpr std::uint32_t kGraphRetryMost = 256;
  bool ChargeGraph(Entry& entry, std::size_t variant = 0) {
    Settle(entry);
    if (variant >= Variants || entry.charged[variant]) {
      return variant < Variants;
    }
    if (entry.graph_skip > 0) {
      --entry.graph_skip;
      return false;
    }
    const std::uint64_t bytes = entry.nodes * kGraphNodeHostBytes;
    if (account_ != nullptr && !account_->Charge(bytes, false)) {
      entry.graph_backoff =
          std::min(std::max<std::uint32_t>(1, 2 * entry.graph_backoff), kGraphRetryMost);
      entry.graph_skip = entry.graph_backoff;
      return false;
    }
    entry.graph_backoff = 0;
    entry.charged[variant] = true;
    return true;
  }
  // Destroys every plan and graph (before the launch context), only
  // between jobs.
  void Clear() override {
    std::uint64_t bytes = 0;
    for (const auto& e : entries_) {
      bytes += e->host_bytes + Charged(*e);
    }
    index_.clear();
    entries_.clear();
    if (account_ != nullptr) {
      account_->Uncharge(bytes);
    }
  }
  template <typename F>
  void ForEach(F&& f) const {
    for (const auto& e : entries_) {
      f(*e);
    }
  }
  std::size_t size() const override { return entries_.size(); }
  // The host bytes its plans hold, as counted at Add.
  std::uint64_t host_bytes() const override {
    std::uint64_t n = 0;
    for (const auto& e : entries_) {
      n += e->host_bytes;
    }
    return n;
  }
  // What its graphs hold, as counted (each its plan's nodes at
  // kGraphNodeHostBytes).
  std::uint64_t graph_bytes() const override {
    std::uint64_t n = 0;
    for (const auto& e : entries_) {
      n += e->graph_bytes();
    }
    return n;
  }
  std::uint64_t graph_measured_bytes() const override {
    std::uint64_t n = 0;
    for (const auto& e : entries_) {
      for (const Runs& r : e->runs) {
        n += r.has_graph() ? r.measured_bytes : 0;
      }
    }
    return n;
  }
  // Graphs kept, every variant's.
  std::size_t graphs() const override {
    std::size_t n = 0;
    for (const auto& e : entries_) {
      n += e->graph_count();
    }
    return n;
  }

  void Collect(std::uint32_t owner, bool running,
               std::vector<memory::ReclaimCandidate>& out) override {
    Settle();
    const std::uint64_t since = PlanStepStart();
    for (const auto& e : entries_) {
      if (e->used >= since) {
        continue;  // the step under way holds it
      }
      const std::uint64_t graphs = Charged(*e);
      out.push_back({.kind = memory::ReclaimKind::kPlan,
                     .owner = owner,
                     .id = e->serial,
                     .bytes = e->host_bytes + graphs,
                     .last_use = e->used,
                     .restore_seconds = e->seconds + e->graph_seconds(),
                     .running = running});
      memory::SetUse(out.back(), e->stamp);
      if (graphs != 0) {
        out.push_back({.kind = memory::ReclaimKind::kGraph,
                       .owner = owner,
                       .id = e->serial,
                       .bytes = graphs,
                       .last_use = e->used,
                       .restore_seconds = e->graph_seconds(),
                       .running = running});
        memory::SetUse(out.back(), e->stamp);
      }
    }
  }
  std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t serial) override {
    Settle();
    const auto at = std::ranges::find_if(
        entries_, [serial](const std::unique_ptr<Entry>& e) { return e->serial == serial; });
    if (at == entries_.end() || (*at)->used >= PlanStepStart()) {
      return 0;
    }
    Entry& e = **at;
    std::uint64_t bytes = Charged(e);
    if (kind == memory::ReclaimKind::kGraph) {
      if (bytes == 0) {
        return 0;
      }
      for (std::size_t v = 0; v < Variants; ++v) {
        e.runs[v].DropGraph();
        e.charged[v] = false;
      }
      ++reclaimed_graphs_;
    } else if (kind == memory::ReclaimKind::kPlan) {
      bytes += e.host_bytes;
      Unindex(e);
      entries_.erase(at);
      ++reclaimed_plans_;
    } else {
      return 0;
    }
    if (account_ != nullptr) {
      account_->Uncharge(bytes);
    }
    return bytes;
  }

 private:
  // The bytes charged for an entry's graphs.
  static std::uint64_t Charged(const Entry& e) {
    return static_cast<std::uint64_t>(std::ranges::count(e.charged, true)) * e.nodes *
           kGraphNodeHostBytes;
  }
  // Reconciles each graph's charge with its graph: a capture that was
  // refused, or a graph dropped by its runner (a moved place), gives its
  // charge back once no step holds its plan (a step under way charged it
  // for the capture its job is about to make); a graph captured without
  // one is charged now.
  void Settle() {
    std::uint64_t release = 0;
    const std::uint64_t since = PlanStepStart();
    for (const auto& e : entries_) {
      release += Reconcile(*e, since);
    }
    if (release != 0 && account_ != nullptr) {
      account_->Uncharge(release);
    }
  }

  // Lookups and captures inspect only their own entry. Reclaim and cache
  // growth still reconcile every entry, so stale graph charges elsewhere
  // are released before memory pressure makes a decision about them.
  void Settle(Entry& e) {
    const std::uint64_t release = Reconcile(e, PlanStepStart());
    if (release != 0 && account_ != nullptr) {
      account_->Uncharge(release);
    }
  }

  std::uint64_t Reconcile(Entry& e, std::uint64_t since) {
    std::uint64_t release = 0;
    for (std::size_t v = 0; v < Variants; ++v) {
      const bool has = e.runs[v].has_graph();
      if (e.charged[v] && !has && e.used < since) {
        release += e.nodes * kGraphNodeHostBytes;
        e.charged[v] = false;
      } else if (!e.charged[v] && has) {
        if (account_ != nullptr) {
          (void)account_->Charge(e.nodes * kGraphNodeHostBytes, true);
        }
        e.charged[v] = true;
      }
    }
    return release;
  }

  void Unindex(const Entry& e) {
    const auto [first, last] = index_.equal_range(Hash{}(e.key));
    for (auto it = first; it != last; ++it) {
      if (it->second == &e) {
        index_.erase(it);
        return;
      }
    }
  }

  PlanAccount* account_ = nullptr;
  std::vector<std::unique_ptr<Entry>> entries_;
  std::unordered_multimap<std::size_t, Entry*> index_;  // by the key's hash
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_PLAN_CACHE_H_
