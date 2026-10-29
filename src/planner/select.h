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
// rows and values a row; Conv2d spatial positions, kernel reduction and output
// channels; anything else its rows and row length.

#ifndef LLMP_PLANNER_SELECT_H_
#define LLMP_PLANNER_SELECT_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "execution/registry.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "planner/policy.h"
#include "planner/state_span.h"

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
  std::string device_model{};
  std::uint32_t multiprocessors = 0;
  // Explicitly supplied source implementations retain compatibility preference
  // after measured ranks. Names are exact owned declarations, not prefixes.
  std::vector<std::string> preferred{};
};

// A source/backend's immutable copy of device facts, including the strings
// TunedShape normally borrows. Bound is allocation-free; Create requires the
// caller to have paid that bound before copying. The charge belongs to the
// backend owner and remains live while any view is borrowed.
class OwnedDeviceFacts {
 public:
  static std::expected<std::uint64_t, std::string> Bound(const DeviceFacts& facts);
  static std::expected<OwnedDeviceFacts, std::string> Create(const DeviceFacts& facts,
                                                             std::uint64_t paid_bytes);
  OwnedDeviceFacts(OwnedDeviceFacts&&) noexcept;
  OwnedDeviceFacts& operator=(OwnedDeviceFacts&&) noexcept;
  OwnedDeviceFacts(const OwnedDeviceFacts&) = delete;
  OwnedDeviceFacts& operator=(const OwnedDeviceFacts&) = delete;
  ~OwnedDeviceFacts();
  const DeviceFacts& facts() const;

 private:
  struct Storage;
  explicit OwnedDeviceFacts(std::unique_ptr<Storage> storage);
  std::unique_ptr<Storage> storage_;
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
// kPermute; the trait table's structural rows) are the lowering's
// bindings, not steps.
bool LaunchesNothing(ir::OpKey key);

// What an operand of `graph` is, for a capability (execution::
// OperandBinding): a whole bound weight (the weight leaf, or a reshape of
// all of it) or something else (invalid: something else).
execution::OperandBinding BindingOf(const ir::Graph& graph, ir::TensorId operand);

// Immutable root-allocation guarantees. The callback sees the actual graph
// at each rewrite/decomposition pass, never original IDs in a new graph.
// Zero means unknown; a physical binding must independently meet the promise.
using RootAlignment = std::function<std::uint64_t(const ir::Graph&, ir::TensorId)>;
struct RootAlignmentFact {
  ir::TensorId root{};
  std::uint64_t alignment = 0;
};
// Owns facts plus canonical digest AND exact root-ID-to-semantic-identity
// seal. Canonical digest alone ignores leaf declaration order. Returns an
// empty callback for malformed facts; new/reordered graphs return unknown.
// The caller must own/guarantee each promised root's base alignment. This
// constructor authenticates identity, not physical addresses or backing.
RootAlignment GraphRootAlignments(const ir::Graph& graph, std::span<const RootAlignmentFact> facts);

// Carries only authenticated external leaves through the exact tensor map.
// Unchanged graphs also retain their exact root facts. Ambiguous, altered or
// unmapped roots lose their facts; the old callback never sees borrowed IDs.
// The carried facts are sealed to `after`'s identity (ir::Graph::identity,
// which a move carries): no graph is hashed per pass or query, and another
// graph, even an equal clone, gets only what `facts` itself grants it.
RootAlignment ForwardRootAlignments(const ir::Graph& before, const ir::Graph& after,
                                    std::span<const ir::TensorId> tensors,
                                    const RootAlignment& facts);

// Actual arena-base guarantee and the placement contract that will fund it.
// Independent roots (including mapped staged leaves) get gcd(base, offset).
// Contained outputs remain unknown; no inner-offset contract is implied.
struct ActivationAlignment {
  std::uint64_t base = 0;
  std::uint64_t offset = 256;
  std::optional<std::vector<ir::TensorId>> staged{};
  std::function<ir::TensorId(const ir::Graph&, ir::TensorId)> within{};
};

// The registry, the device and the policy, as a pattern or selection asks
// them about a node that exists or would.
class SelectionView {
 public:
  SelectionView(const execution::Registry& registry, const DeviceFacts& device,
                const Policy& policy, bool row_invariant, bool capturable,
                StateSpans state_spans = {}, RootAlignment root_alignment = {},
                ActivationAlignment activation = {});

  // A rewrite's exact map, before the next round queries this view.
  void Forward(const ir::Graph& before, const ir::Graph& after,
               std::span<const ir::TensorId> tensors);
  std::uint64_t Alignment(const ir::Graph& graph, ir::TensorId root) const;
  std::vector<RootAlignmentFact> Alignments(const ir::Graph& graph) const;

  const execution::Registry& registry() const { return registry_; }
  const Policy& policy() const { return policy_; }
  bool row_invariant() const { return row_invariant_; }

  // The candidates (registry indices, in Candidates()' order) for a node
  // of `key` over `sources` with `outputs` and parameters `params` (its
  // ParamBytes; empty: none given). `bindings`, by source, are what the
  // asker knows of them (BindingOf); left out, unknown.
  std::vector<std::size_t> Candidates(
      ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
      std::span<const ir::TensorDesc* const> outputs, std::span<const std::byte> params = {},
      bool state_span = false, bool automatic_fusion = false,
      std::span<const execution::OperandBinding> bindings = {}) const;
  bool HasCandidate(ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
                    std::span<const ir::TensorDesc* const> outputs,
                    std::span<const std::byte> params = {}, bool state_span = false,
                    bool automatic_fusion = false,
                    std::span<const execution::OperandBinding> bindings = {}) const;
  std::optional<std::size_t> NextCandidate(
      ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
      std::span<const ir::TensorDesc* const> outputs, std::span<const std::byte> params,
      std::optional<std::size_t> after, bool state_span = false, bool automatic_fusion = false,
      std::span<const execution::OperandBinding> bindings = {}) const;
  // The same for a node of `graph` (with its parameters), as it is or
  // under another key (a batch's).
  std::vector<std::size_t> Candidates(const ir::Graph& graph, const ir::Node& node) const;
  std::vector<std::size_t> Candidates(const ir::Graph& graph, const ir::Node& node, ir::OpKey key,
                                      bool automatic_fusion = false) const;
  std::optional<std::size_t> NextCandidate(const ir::Graph& graph, const ir::Node& node,
                                           ir::OpKey key, std::optional<std::size_t> after,
                                           bool automatic_fusion = false) const;
  bool HasCandidate(const ir::Graph& graph, const ir::Node& node, ir::OpKey key,
                    bool automatic_fusion = false) const;
  bool IsCandidate(const ir::Graph& graph, const ir::Node& node, ir::OpKey key,
                   std::size_t implementation, bool automatic_fusion = false) const;

  // Borrow the same immutable typed query used for this node's selection.
  // The callback is synchronous and does not retain descriptors or reselect.
  using QueryInspector = std::function<void(const execution::CapabilityQuery&)>;
  void Inspect(const ir::Graph& graph, const ir::Node& node, const QueryInspector& inspect) const;
  void Inspect(const ir::Graph& graph, const ir::Node& node, ir::OpKey key,
               const QueryInspector& inspect, bool automatic_fusion = false) const;
  QueryShape Shape(const ir::Graph& graph, const ir::Node& node, ir::OpKey key) const;
  ir::ProductArithmetic IncumbentArithmetic(const ir::Graph& graph, const ir::Node& node) const;
  // A compound GLU changes two independent product contracts. Its operands
  // are priced/qualified as primitive products, each with its own weight role.
  static std::uint8_t NumericalOperands(const ir::Node& node) {
    return node.key == ir::Fused(ir::FusedKind::kMatMulGlu) ||
                   node.key == ir::Fused(ir::FusedKind::kMatMulIdGlu)
               ? 2
               : 1;
  }
  struct WeightOperand {
    ir::WeightRef reference{};
    ir::TensorId leaf{};
  };
  // The one canonical traversal used by qualification and report observation:
  // follow only this operand's source0 chain inside its component. Resolution
  // observes a WeightRef; matching stage/component/role and permission remain
  // Qualification's separate checks. Unknown/foreign facts grant nothing.
  static std::optional<WeightOperand> ResolveWeightOperand(const ir::Graph& graph,
                                                           const ir::Node& node,
                                                           std::uint8_t operand);
  std::optional<NumericalQualification> Qualification(NumericalChange change,
                                                      const ir::Graph& graph, const ir::Node& node,
                                                      std::uint8_t weight_operand = 0) const;
  bool NumericallyQualified(NumericalChange change, const ir::Graph& graph,
                            const ir::Node& node) const {
    for (std::uint8_t operand = 0; operand < NumericalOperands(node); ++operand)
      if (!Qualification(change, graph, node, operand)) return false;
    return true;
  }

 private:
  std::vector<std::size_t> CandidatesForNode(const ir::Graph& graph, const ir::Node& node,
                                             ir::OpKey key, bool automatic_fusion,
                                             const QueryInspector& inspect = {},
                                             std::optional<std::size_t>* next_after = nullptr,
                                             std::size_t* check_implementation = nullptr,
                                             bool* is_candidate = nullptr) const;
  std::vector<std::size_t> ForOperands(
      ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
      std::span<const ir::TensorDesc* const> outputs, std::span<const std::byte> params,
      bool state_span, std::span<const std::uint64_t> alignments,
      std::span<const execution::OperandBinding> bindings, bool automatic_fusion = false,
      const QueryInspector& inspect = {}, bool* has_candidate = nullptr,
      std::optional<std::size_t>* next_after = nullptr, std::size_t* check_implementation = nullptr,
      bool* is_candidate = nullptr) const;
  const execution::Registry& registry_;
  const DeviceFacts& device_;
  const Policy& policy_;
  bool row_invariant_ = false;
  bool capturable_ = false;
  StateSpans state_spans_{};
  RootAlignment root_alignment_{};
  ActivationAlignment activation_{};
  std::vector<execution::MeasuredRank> ranks_;
  std::vector<std::string_view> enabled_options_;
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
