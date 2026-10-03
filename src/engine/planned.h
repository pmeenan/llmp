// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A model's planned shapes (docs/engine.md): the graph of one chunk (or
// drafter pass) shape built in its own tensor arena, planned, its computed
// tensors placed in the node's activations and planned again (which must
// give the same plan), its scratch checked against the GGML pool and its
// implementations bound against the registry (D-053); the cache of them a
// runner keeps per shape with each plan's runs (graph_runs.h); the caps on
// a model's plans and graphs (host and driver memory outside the catalog,
// D-090, which the memory guard counts beside it: Served::
// plan_host_bytes); and BP-A1's in-process check that every tensor a plan
// binds lies in cataloged, resident memory of the class it should.
//
// The model's graph builder, binding and inputs stay the model's own
// (dsv4_plan.h, qwen38_plan.h); only the mechanics are here.

#ifndef JITLLM_ENGINE_PLANNED_H_
#define JITLLM_ENGINE_PLANNED_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "engine/graph_runs.h"
#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::engine {

class PagedNode;

// What every planned shape holds besides its graph: its arena (which owns
// the graph's tensors, so it outlives them), plan, placement, bound
// implementations, and the pool scratch and staged input bytes it needs.
struct PlannedBase {
  std::optional<kernels::ggml::TensorArena> arena;
  kernels::ggml::GraphPlan plan;
  kernels::ggml::Placement placement;
  std::optional<kernels::ggml::BoundGraph> bound;
  std::uint64_t scratch = 0;
  std::uint64_t inputs_bytes = 0;  // as GraphRuns::Stage stages them
};

// A planned shape of a model's graph type (kernels/ggml/*_graph.h).
template <typename Graph>
struct PlannedGraph : PlannedBase {
  Graph graph;
};

// Places a graph's computed tensors: `inputs` bound at distinct placeless
// addresses and every computed node at its own, planned; the activations
// placed (`keep` live to the end); then, unless `activations` is 0 (measure
// only), bound in [activations, + activation_bytes), planned again, which
// must give the same plan.
std::expected<void, std::string> PlaceAndPlan(PlannedBase& out, std::span<ggml_tensor* const> nodes,
                                              std::span<ggml_tensor* const> inputs,
                                              std::span<ggml_tensor* const> keep,
                                              const kernels::ggml::DeviceChoices& choices,
                                              std::uint64_t activations,
                                              std::uint64_t activation_bytes);

// The scratch `planned`'s plan needs, checked against the launch context's
// pool, and its implementations bound against `registry` (D-053). `what`
// names the plan in a refusal ("the plan", "the draft").
std::expected<void, std::string> BindPlanned(PlannedBase& planned,
                                             kernels::ggml::LaunchContext& launch,
                                             const execution::Registry& registry,
                                             std::string_view what);

// What a planned shape holds on the host, as its runner's plan budget
// counts it: its arena's bytes (exact) and kPlanNodeHostBytes for each node
// its plan launches (the plan's steps, the activations' places and the
// bound launches; an upper bound measured on GB10,
// docs/experiments/deepseek-batching/README.md, "Plan memory").
inline constexpr std::uint64_t kPlanNodeHostBytes = 512;
std::uint64_t PlannedHostBytes(const PlannedBase& planned);
// The nodes a planned shape's plan launches (its steps' nodes).
std::uint64_t PlannedNodes(const PlannedBase& planned);
// What a captured graph holds outside the catalog (the driver's host and
// device memory) for each node its plan launches: an allowance over the
// largest measured on GB10, a 4-slot DeepSeek decode wave graph's
// MemAvailable drop of up to 56 MiB at 4,912 launched nodes (11.7 KiB a
// node; the same report), with about 37% headroom.
inline constexpr std::uint64_t kGraphNodeHostBytes = 16384;

// The order a runner's plans were last used in, across its caches
// (PlanCache's least recently used; one counter for the process).
std::uint64_t NextPlanUse();

// A model's planned shapes, by key (a shape and what the model computes
// beside it), each with its runs: `Variants` of them where one plan runs
// with different outputs (a Qwen3.8 verify with and without its logits'
// copy). At most `capacity` shapes are kept, the least recently used
// dropped with its graphs. Entries stay put until the next Add, Drop or
// Clear.
template <typename Key, typename Planned, std::size_t Variants = 1>
class PlanCache {
 public:
  static constexpr std::uint64_t kNever = std::numeric_limits<std::uint64_t>::max();
  struct Entry {
    Key key{};
    std::unique_ptr<Planned> planned;
    std::array<PlanRuns, Variants> runs;
    std::uint64_t host_bytes = 0;  // PlannedHostBytes, as the runner counted it
    std::uint64_t nodes = 0;       // PlannedNodes: its graphs' cost (kGraphNodeHostBytes each)
    std::uint64_t used = 0;        // NextPlanUse at its last Find or Add

    bool has_graph() const {
      return std::ranges::any_of(runs, [](const PlanRuns& r) { return r.graph.has_value(); });
    }
  };

  explicit PlanCache(std::size_t capacity) : capacity_(capacity) {}

  Entry* Find(const Key& key) {
    for (Entry& e : entries_) {
      if (e.key == key) {
        e.used = NextPlanUse();
        return &e;
      }
    }
    return nullptr;
  }
  Entry& Add(Key key, std::unique_ptr<Planned> planned, std::uint64_t host_bytes = 0,
             std::uint64_t nodes = 0) {
    if (entries_.size() >= capacity_) {
      DropOldest();  // its graphs with it
    }
    Entry& entry = entries_.emplace_back();
    entry.key = std::move(key);
    entry.planned = std::move(planned);
    entry.host_bytes = host_bytes;
    entry.nodes = nodes;
    entry.used = NextPlanUse();
    return entry;
  }
  // Destroys every plan and graph (before the launch context).
  void Clear() { entries_.clear(); }
  std::size_t size() const { return entries_.size(); }
  std::size_t capacity() const { return capacity_; }
  // The host bytes its plans hold, as counted at Add.
  std::uint64_t host_bytes() const {
    std::uint64_t n = 0;
    for (const Entry& e : entries_) {
      n += e.host_bytes;
    }
    return n;
  }
  // What its graphs hold, as counted (each its plan's nodes at
  // kGraphNodeHostBytes).
  std::uint64_t graph_bytes() const {
    std::uint64_t n = 0;
    for (const Entry& e : entries_) {
      for (const PlanRuns& r : e.runs) {
        n += r.graph.has_value() ? e.nodes * kGraphNodeHostBytes : 0;
      }
    }
    return n;
  }
  // Graphs kept, every variant's.
  std::size_t graphs() const {
    std::size_t n = 0;
    for (const Entry& e : entries_) {
      for (const PlanRuns& r : e.runs) {
        n += r.graph.has_value() ? 1 : 0;
      }
    }
    return n;
  }
  // The least recently used entry's use (kNever: none), and its drop with
  // its graphs, only between jobs (nothing in flight replays them).
  std::uint64_t OldestUse() const { return OldestOf(false); }
  void DropOldest() {
    if (Entry* e = Oldest(false); e != nullptr) {
      entries_.erase(entries_.begin() + (e - entries_.data()));
    }
  }
  // The same over the entries that hold a graph: its first variant's graph
  // destroyed (the plan stays). False if none holds one.
  std::uint64_t OldestGraphUse() const { return OldestOf(true); }
  bool DropOldestGraph() {
    Entry* e = Oldest(true);
    if (e == nullptr) {
      return false;
    }
    for (PlanRuns& r : e->runs) {
      if (r.graph.has_value()) {
        r.DropGraph();
        return true;
      }
    }
    return false;
  }

 private:
  Entry* Oldest(bool with_graph) {
    Entry* oldest = nullptr;
    for (Entry& e : entries_) {
      if ((!with_graph || e.has_graph()) && (oldest == nullptr || e.used < oldest->used)) {
        oldest = &e;
      }
    }
    return oldest;
  }
  std::uint64_t OldestOf(bool with_graph) const {
    std::uint64_t oldest = kNever;
    for (const Entry& e : entries_) {
      if (!with_graph || e.has_graph()) {
        oldest = std::min(oldest, e.used);
      }
    }
    return oldest;
  }

  std::size_t capacity_;
  std::vector<Entry> entries_;
};

// Before adding a plan to one of `caches` (one kind of a runner's plans,
// sharing one cap: each request slot's chunk plans, say): while they hold
// `most` plans or more, the least recently used of them goes, with its
// graphs. Only between jobs, and with no entry of any of them borrowed by
// the caller: an Add to one may drop another's.
template <typename Cache>
void RoomForPlan(std::size_t most, std::span<Cache* const> caches) {
  for (;;) {
    std::size_t held = 0;
    Cache* oldest = nullptr;
    for (Cache* cache : caches) {
      held += cache->size();
      if (cache->size() != 0 && (oldest == nullptr || cache->OldestUse() < oldest->OldestUse())) {
        oldest = cache;
      }
    }
    if (held < most || oldest == nullptr) {
      return;
    }
    oldest->DropOldest();
  }
}

// Before `adding` captures of `adding_bytes` in all (kGraphNodeHostBytes a
// node their plans launch): while `caches` keep more than `most` -
// `adding` graphs, or more than `budget` - `adding_bytes` bytes of them,
// the least recently used plan's graph goes (between jobs: nothing in
// flight replays it; a plan whose graph went runs launch by launch until
// captured again), each counted in `stats`. Every graph a model keeps
// counts: its target's, its drafter's and its waves'. False, dropping
// nothing, when the captures alone exceed the budget or the cap: they are
// not made (their plans run launch by launch), so the kept graphs never
// hold more than `budget`.
template <typename... Caches>
bool RoomForGraphs(std::size_t most, std::uint64_t budget, std::size_t adding,
                   std::uint64_t adding_bytes, GraphStats& stats, Caches&... caches) {
  if (adding > most || adding_bytes > budget) {
    return false;
  }
  for (;;) {
    const std::size_t kept = (caches.graphs() + ... + std::size_t{0});
    const std::uint64_t bytes = (caches.graph_bytes() + ... + std::uint64_t{0});
    if ((kept + adding <= most && bytes <= budget && adding_bytes <= budget - bytes) || kept == 0) {
      return true;
    }
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    ((oldest = std::min(oldest, caches.OldestGraphUse())), ...);
    bool dropped = false;
    ((dropped = dropped || (caches.OldestGraphUse() == oldest && caches.DropOldestGraph())), ...);
    if (!dropped) {
      return true;
    }
    ++stats.dropped;
  }
}

// Before one capture, under a cap on the number of graphs alone.
template <typename... Caches>
void RoomForGraph(std::size_t most, GraphStats& stats, Caches&... caches) {
  (void)RoomForGraphs(most, std::numeric_limits<std::uint64_t>::max(), 1, 0, stats, caches...);
}

// BP-A1's in-process check, once per planned shape: each tensor a plan
// binds (every node and its sources) should lie in cataloged, resident
// device memory of one class, the owner's or shared (PagedNode::Covered):
// live state for `state`, runtime for `runtime`, scratch for `scratch`,
// the inputs and every computed tensor; any other leaf is a weight.
// Initialized so callers may designate only the fields they use.
// NOLINTBEGIN(readability-redundant-member-init)
struct TensorClasses {
  std::span<const ggml_tensor* const> state = {};
  std::span<const ggml_tensor* const> runtime = {};
  std::span<const ggml_tensor* const> scratch = {};
  std::span<ggml_tensor* const> inputs = {};
  // A fill's sources only shape it: not checked (Qwen3.8's QSA zeros).
  bool fill_reads_nothing = false;
  std::string_view what = {};  // the plan, in the first violation ("", "the drafter's ")
};
// NOLINTEND(readability-redundant-member-init)
struct Coverage {
  std::uint64_t tensors = 0;
  std::uint64_t violations = 0;
  std::string first_violation;
};
void CheckCoverage(const PagedNode& node, int owner, std::span<ggml_tensor* const> nodes,
                   const TensorClasses& classes, Coverage& coverage);

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_PLANNED_H_
