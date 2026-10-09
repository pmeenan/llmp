// SPDX-FileCopyrightText: 2026 llmpalooza contributors
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

#ifndef LLMP_ENGINE_PLANNED_H_
#define LLMP_ENGINE_PLANNED_H_

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

#include "engine/coverage.h"
#include "engine/graph_runs.h"
#include "engine/plan_account.h"
#include "engine/plan_cache.h"
#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/tensors.h"
#include "memory/reclaim.h"

namespace llmp::engine {

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
// The graph reader-index allowance runtime plans get (PlaceAndPlan): the
// largest index the start measured and a quarter more. A family's startup
// probes are its widest shapes, but a graph's nodes need not grow with its
// rows: a row-count-gated fusion (Dsv4QHead from 16 rows, the per-group
// float products past 8) can give an intermediate shape more nodes than
// either probe, as a four-slot DeepSeek verify wave of 15 rows has 43 more
// than the 16-row one measured. Margin as the activations' and pool's.
std::uint64_t GraphIndexAllowance(std::uint64_t measured);
// That scratch arena plus the graph reader-index allowance (one transient
// index, shared by sequential planning passes): host memory outside the
// catalog, which the start's guard counts beside a step's plans once the
// models measured their largest plans.
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
// Startup descriptor work only; these count planned selections, not GPU launches.
// A bounded placement may replace an individual extent, but the runner's final
// maximum must equal the exact-only sequence's maximum.
struct StartupPlacementStats {
  std::uint64_t plans = 0;
  std::uint64_t nodes = 0;
  std::uint64_t bounded = 0;
  std::uint64_t exact = 0;
  std::uint64_t max_activations = 0;
  std::uint64_t max_inputs = 0;
  std::uint64_t max_host = 0;
  std::uint64_t max_nodes = 0;
  void Account(const PlannedBase& planned);
};

// A model's GGML planned shapes by key (plan_cache.h), each with its
// GraphRuns runs.
template <typename Key, typename Planned, std::size_t Variants = 1>
using PlanCache = BasicPlanCache<Key, Planned, Variants, PlanRuns>;

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
void CheckCoverage(const PagedNode& node, int owner, std::span<ggml_tensor* const> nodes,
                   const TensorClasses& classes, Coverage& coverage);

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_PLANNED_H_
