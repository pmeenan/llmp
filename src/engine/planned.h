// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A model's planned shapes (docs/engine.md): the graph of one chunk (or
// drafter pass) shape built in its own tensor arena (sized to what its
// tensors use), planned, its computed tensors placed in the node's
// activations and planned again (which must give the same plan), its
// scratch checked against the GGML pool and its implementations bound
// against the registry (D-053); the cache of them a runner keeps per shape
// with each plan's runs (graph_runs.h); what they hold (host and driver
// memory outside the catalog's extents, D-090 as amended 2026-10-02),
// charged to the node beside the catalog (PlanAccount) and given back
// through the node's one reclaim order (memory/reclaim.h) as plans and
// graphs least recently used within their kind; and BP-A1's in-process
// check that every tensor a plan binds lies in cataloged, resident memory
// of the class it should.
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
#include <functional>
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
#include "memory/reclaim.h"

namespace jitllm::engine {

class PagedNode;

// Authenticate a plain token publication before queuing a copy: the exact
// packed I32 argmax output must be wholly inside current activation backing.
// The caller still validates every copied ID after completion before publishing
// any owner, and quarantines written state if that check fails.
std::expected<std::uint64_t, std::string> GreedyOutputBytes(const ggml_tensor* node,
                                                            std::uint32_t count,
                                                            std::uint64_t activations,
                                                            std::uint64_t activation_bytes);
// After completion, validate the whole copied batch before any owner receives
// its token. A caller with processed state quarantines it on refusal.
std::expected<void, std::string> CheckGreedyTokens(std::span<const std::int32_t> tokens,
                                                   std::uint32_t vocab);

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
  bool measurement_only = false;   // cannot bind, including an exact shortcut fallback
};

// A maximum established by earlier exact startup measurements, never a grant.
struct ActivationMeasurement {
  std::uint64_t exact_ceiling = 0;
};

// A planned shape of a model's graph type (kernels/ggml/*_graph.h).
template <typename Graph>
struct PlannedGraph : PlannedBase {
  Graph graph;
};

// An arena for a graph of at most `estimate` tensors that `build` (the
// model's graph builder over an arena; false if it could not build) makes,
// sized to the bytes its tensors use: `build` runs once over a scratch
// arena of `estimate` tensors (one for the process, reused under a lock,
// grown to the largest estimate seen), and the arena returned holds
// exactly what that build used, for the caller's own build of the same
// graph. Its metadata is trimmed; its separately counted graph traversal
// table retains the bounded estimate (docs/experiments/memory-pressure).
std::expected<kernels::ggml::TensorArena, std::string> SizedArena(
    std::size_t estimate, const std::function<bool(kernels::ggml::TensorArena&)>& build);
// That scratch arena plus the maximum measured graph reader-index allowance
// (one transient index, shared by sequential planning passes): host memory
// outside the catalog, which
// the start's guard counts beside a step's plans once the models measured
// their largest plans).
std::uint64_t ScratchArenaBytes();

// Places a graph's computed tensors: `inputs` bound at distinct placeless
// addresses and every computed node at its own, planned; the activations
// placed (`keep` live to the end); then, unless `activations` is 0 (measure
// only), bound in [activations, + activation_bytes), planned again, which
// must give the same plan. With `lanes`, both plans take its concurrent
// lanes (graph_plan.h AssignLanes) before the placement.
std::expected<void, std::string> PlaceAndPlan(PlannedBase& out, std::span<ggml_tensor* const> nodes,
                                              std::span<ggml_tensor* const> inputs,
                                              std::span<ggml_tensor* const> keep,
                                              const kernels::ggml::DeviceChoices& choices,
                                              std::uint64_t activations,
                                              std::uint64_t activation_bytes,
                                              const kernels::ggml::LaneTags* lanes = nullptr);

// Explicit startup-only overload. Nonzero activation addresses are refused;
// the returned plan cannot be bound. Every other measurement remains exact.
std::expected<void, std::string> PlaceAndPlan(PlannedBase& out, std::span<ggml_tensor* const> nodes,
                                              std::span<ggml_tensor* const> inputs,
                                              std::span<ggml_tensor* const> keep,
                                              const kernels::ggml::DeviceChoices& choices,
                                              std::uint64_t activations,
                                              std::uint64_t activation_bytes,
                                              std::optional<ActivationMeasurement> measurement,
                                              const kernels::ggml::LaneTags* lanes = nullptr);

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

// A runner's step (a chunk, a draft, a verify, a wave) borrows plans from
// its caches from its first Find to its job's end. While one is under way
// on this thread (from a PlanStep's construction to its destruction; they
// nest), a reclaim spares every plan used since it began: a charge that
// asks the node to reclaim (PlanAccount) never drops what the step holds.
// Between steps nothing is borrowed, and every plan is a candidate.
class PlanStep {
 public:
  PlanStep();
  PlanStep(const PlanStep&) = delete;
  PlanStep& operator=(const PlanStep&) = delete;
  PlanStep(PlanStep&&) = delete;
  PlanStep& operator=(PlanStep&&) = delete;
  ~PlanStep();
};
// The first use a reclaim must spare: the outermost step's start, or
// kNoStep between steps.
inline constexpr std::uint64_t kNoStep = std::numeric_limits<std::uint64_t>::max();
std::uint64_t PlanStepStart();

// What a model's plans and graphs hold, as the node charges it
// (PagedNode::ChargeHost; D-090 as amended 2026-10-02): counted inside the
// execution budget beside the catalog's extents, past a floor the start's
// guard sets apart (what one step of the model needs at once). A charge
// that does not fit has the node reclaim through its one order first. A
// required charge (a plan the step under way needs) is always taken; one
// that is not (a graph's capture) is refused when it still does not fit,
// and the plan then runs launch by launch. Unbound (a host test, or before
// the node binds it), it only counts.
class PlanAccount {
 public:
  using ChargeFn = std::function<bool(std::uint64_t bytes, bool required)>;
  using ReleaseFn = std::function<void(std::uint64_t bytes)>;
  void Bind(ChargeFn charge, ReleaseFn release) {
    charge_ = std::move(charge);
    release_ = std::move(release);
  }
  bool Charge(std::uint64_t bytes, bool required) {
    if (bytes != 0 && charge_ && !charge_(bytes, required)) {
      return false;
    }
    bytes_ += bytes;
    return true;
  }
  void Uncharge(std::uint64_t bytes) {
    bytes = std::min(bytes, bytes_);
    bytes_ -= bytes;
    if (bytes != 0 && release_) {
      release_(bytes);
    }
  }
  std::uint64_t bytes() const { return bytes_; }

 private:
  ChargeFn charge_;
  ReleaseFn release_;
  std::uint64_t bytes_ = 0;
};

// What the reclaimer sees of a plan cache, whatever its key and plan type:
// its plans and graphs as candidates (none a step under way holds), and a
// candidate's reclaim.
class PlanCacheBase {
 public:
  PlanCacheBase() = default;
  PlanCacheBase(const PlanCacheBase&) = delete;
  PlanCacheBase& operator=(const PlanCacheBase&) = delete;
  PlanCacheBase(PlanCacheBase&&) = delete;
  PlanCacheBase& operator=(PlanCacheBase&&) = delete;
  virtual ~PlanCacheBase() = default;
  // Each plan (with its graphs: kPlan) and each plan's graphs (kGraph)
  // not used since PlanStepStart(), `owner`'s, `running` if its model is.
  virtual void Collect(std::uint32_t owner, bool running,
                       std::vector<memory::ReclaimCandidate>& out) = 0;
  // Drops the plan `serial` with its graphs (kPlan) or only its graphs
  // (kGraph), between jobs: the bytes it freed, 0 if it is not this
  // cache's or a step under way holds it.
  virtual std::uint64_t Reclaim(memory::ReclaimKind kind, std::uint64_t serial) = 0;
  virtual std::size_t size() const = 0;
  virtual std::size_t graphs() const = 0;
  virtual std::uint64_t host_bytes() const = 0;
  virtual std::uint64_t graph_bytes() const = 0;
  // What its graphs took of the device's free memory at their captures
  // (PlanRuns::measured_bytes), to check the count against.
  virtual std::uint64_t graph_measured_bytes() const = 0;
  virtual void Clear() = 0;
  // Plans and graphs dropped by Reclaim since it was made.
  std::uint64_t reclaimed_plans() const { return reclaimed_plans_; }
  std::uint64_t reclaimed_graphs() const { return reclaimed_graphs_; }

 protected:
  std::uint64_t reclaimed_plans_ = 0;
  std::uint64_t reclaimed_graphs_ = 0;
};

// Over a runner's caches: every candidate, and one candidate's reclaim.
void CollectPlans(std::span<PlanCacheBase* const> caches, std::uint32_t owner, bool running,
                  std::vector<memory::ReclaimCandidate>& out);
std::uint64_t ReclaimPlan(std::span<PlanCacheBase* const> caches, memory::ReclaimKind kind,
                          std::uint64_t serial);

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
// PlanStep began) are never reclaimed.
template <typename Key, typename Planned, std::size_t Variants = 1>
class PlanCache final : public PlanCacheBase {
 public:
  static constexpr std::uint64_t kNever = std::numeric_limits<std::uint64_t>::max();
  struct Entry {
    Key key{};
    std::unique_ptr<Planned> planned;
    std::array<PlanRuns, Variants> runs;
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
      return std::ranges::any_of(runs, [](const PlanRuns& r) { return r.graph.has_value(); });
    }
    std::uint64_t graph_count() const {
      return static_cast<std::uint64_t>(
          std::ranges::count_if(runs, [](const PlanRuns& r) { return r.graph.has_value(); }));
    }
    std::uint64_t graph_bytes() const { return graph_count() * nodes * kGraphNodeHostBytes; }
    double graph_seconds() const {
      double s = 0;
      for (const PlanRuns& r : runs) {
        s += r.graph.has_value() ? r.seconds : 0;
      }
      return s;
    }
  };

  explicit PlanCache(PlanAccount* account = nullptr) : account_(account) {}
  ~PlanCache() override = default;
  PlanCache(const PlanCache&) = delete;
  PlanCache& operator=(const PlanCache&) = delete;
  PlanCache(PlanCache&&) = delete;
  PlanCache& operator=(PlanCache&&) = delete;
  void set_account(PlanAccount* account) { account_ = account; }

  Entry* Find(const Key& key) {
    Settle();
    for (const auto& e : entries_) {
      if (e->key == key) {
        e->used = NextPlanUse();
        e->stamp = memory::StampUse();
        return e.get();
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
    Settle();
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
      for (const PlanRuns& r : e->runs) {
        n += r.graph.has_value() ? r.measured_bytes : 0;
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
      for (std::size_t v = 0; v < Variants; ++v) {
        const bool has = e->runs[v].graph.has_value();
        if (e->charged[v] && !has && e->used < since) {
          release += e->nodes * kGraphNodeHostBytes;
          e->charged[v] = false;
        } else if (!e->charged[v] && has) {
          if (account_ != nullptr) {
            (void)account_->Charge(e->nodes * kGraphNodeHostBytes, true);
          }
          e->charged[v] = true;
        }
      }
    }
    if (release != 0 && account_ != nullptr) {
      account_->Uncharge(release);
    }
  }

  PlanAccount* account_ = nullptr;
  std::vector<std::unique_ptr<Entry>> entries_;
};

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
