// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Implementation selection (D-053, D-108; docs/engine-components.md#the-graph-ir):
// for a node of a step's graph, the registry's candidates as Candidates()
// orders them (correctness, then measured rank, then the format ladder),
// asked with the device's facts and the policy's requirements, and the
// plan's step that runs it. The planner takes the first candidate. Where
// that candidate's capability names a group (GGML's "ggml.mul_mat", whose
// members upstream's routing chooses among by device), the step records
// the group: the lowering resolves it inside its module, and the bound
// plan still names one implementation (D-053).
//
// A capability query sees a node as the registry's query does: each
// source's format at its position, and a shape (QueryShape): a product's
// rows (tokens), K and N; attention's query rows and head sizes; a store's
// rows and values a row; anything else its rows and row length.

#ifndef LLMP_PLANNER_SELECT_H_
#define LLMP_PLANNER_SELECT_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "execution/registry.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "planner/policy.h"

namespace llmp::planner {

// What the planner knows of the device a plan runs on, as its provider
// and tuning record report it.
struct DeviceFacts {
  // The measured ranks' key (Policy::ranks): one class a measurement holds
  // for, such as "sm_121".
  std::string device_class{};
  std::uint32_t arch = 0;      // the provider's number (CUDA: major · 10 + minor)
  std::uint32_t features = 0;  // execution::DeviceFeature bits
  // Shapes implementations have tuning data for here (registry.h
  // TunedShape; their names point into the tuning record).
  std::vector<execution::TunedShape> tuned{};
};

struct QueryShape {
  std::uint64_t rows = 1;
  std::uint64_t k = 0;
  std::uint64_t n = 0;
};
// The shape a node of `key` with these sources and outputs (null: an
// absent source) presents to a capability.
QueryShape ShapeOfNode(ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
                       std::span<const ir::TensorDesc* const> outputs);

// Whether a node launches nothing: leaves and views (kView, kReshape,
// kPermute) are the lowering's bindings, not steps.
bool LaunchesNothing(ir::OpKey key);

// The registry, the device and the policy, as a pattern or selection asks
// them about a node that exists or would.
class SelectionView {
 public:
  SelectionView(const execution::Registry& registry, const DeviceFacts& device,
                const Policy& policy, bool row_invariant, bool capturable);

  const execution::Registry& registry() const { return registry_; }
  const Policy& policy() const { return policy_; }
  bool row_invariant() const { return row_invariant_; }

  // The candidates (registry indices, in Candidates()' order) for a node
  // of `key` over `sources` with `outputs` and parameters `params` (its
  // ParamBytes; empty: none given).
  std::vector<std::size_t> Candidates(ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
                                      std::span<const ir::TensorDesc* const> outputs,
                                      std::span<const std::byte> params = {}) const;
  // The same for a node of `graph` (with its parameters), as it is or
  // under another key (a batch's).
  std::vector<std::size_t> Candidates(const ir::Graph& graph, const ir::Node& node) const;
  std::vector<std::size_t> Candidates(const ir::Graph& graph, const ir::Node& node,
                                      ir::OpKey key) const;

 private:
  const execution::Registry& registry_;
  const DeviceFacts& device_;
  const Policy& policy_;
  bool row_invariant_ = false;
  bool capturable_ = false;
  std::vector<execution::MeasuredRank> ranks_;
};

// One step of a plan: an implementation over one node, or over a batch of
// consecutive nodes of one operation (a batch pattern's, pattern.h).
struct Selected {
  std::vector<ir::NodeId> nodes{};  // in the planned graph, in order
  std::size_t implementation = 0;   // a registry index
  // The group its module resolves inside (the capability's), or empty.
  std::string_view group{};
  // The batch pattern that joined `nodes`, or empty; and then each
  // member's own selection, which a member run alone takes
  // (AssignLanes splits a batch across lanes).
  std::string_view batch{};
  std::vector<std::size_t> singles{};
  std::vector<std::string_view> single_groups{};
  // AssignLanes': 0, the context's stream, or 1 to kMaxLanes.
  std::uint8_t lane = 0;
  bool operator==(const Selected&) const = default;
};

}  // namespace llmp::planner

#endif  // LLMP_PLANNER_SELECT_H_
