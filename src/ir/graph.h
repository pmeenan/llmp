// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The graph IR (D-107; docs/engine-components.md#the-graph-ir): tensors,
// nodes and the graph a step runs, modelled on GGML's tensor and view
// model. A tensor has a format (D-108: encoding, layout, shape), byte
// strides, and is either a root holding its own bytes or a view of a
// root's bytes at an offset; weights and state are bound to their roles
// by reference (WeightRef, StateRef), never by address. A node is one
// operation of op set v1 (op.h) with up to kMaxSources sources and
// kMaxOutputs outputs; every tensor is the output of exactly one node, the
// leaves (inputs, weights, state) included, and nodes are in an order in
// which each node's sources come from earlier nodes: the order the step
// runs them in, which also orders writes into state (a cache store's
// output is a view of its cache).
//
// Graphs are made by a Builder (builder.h) and checked by Verify
// (verify.h); a Graph is immutable, move-only, and holds at most kMaxNodes
// nodes and kMaxTensors tensors. Its storage grows while it is built, so
// only a finished graph hands out references (a builder's are copies).
//
// Its digest (Digest()) is canonical: the plan-key and pattern input. Two
// graphs digest the same when they run the same operations in the same
// order on the same tensors: weights and state are named by what they bind
// (a WeightRef, a StateRef), so declaring them in another order digests
// the same; an input (or plan constant) by its position among the inputs,
// which is their declaration order, so inputs declared in another order
// are another graph; a plan constant also by its content's digest where
// Builder::Constant was given its content (a constant's values are bound
// outside the graph); a tensor by the node and output that make it, a view
// by its root; names (diagnostics only) are excluded. Nodes are not
// reordered: two orders of independent nodes are two graphs.

#ifndef LLMP_IR_GRAPH_H_
#define LLMP_IR_GRAPH_H_

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "base/sha256.h"
#include "format/tensor_format.h"
#include "ir/op.h"
#include "ir/verify.h"

namespace llmp::ir {

class Builder;

enum class Role : std::uint8_t {
  kWeight,      // a model weight, bound by its WeightRef
  kState,       // a request's state, bound by its StateRef
  kActivation,  // a node's result within the step
  kInput,       // bound each step (token ids, positions, masks' inputs)
  kOutput,      // a result the step returns (Builder::Output)
  kConstant,    // bound once for the plan (tables computed at planning)
};
std::string_view RoleName(Role role);

struct TensorId {
  static constexpr std::uint32_t kNone = ~std::uint32_t{0};
  std::uint32_t v = kNone;
  constexpr bool valid() const { return v != kNone; }
  auto operator<=>(const TensorId&) const = default;
};

struct NodeId {
  static constexpr std::uint32_t kNone = ~std::uint32_t{0};
  std::uint32_t v = kNone;
  constexpr bool valid() const { return v != kNone; }
  auto operator<=>(const NodeId&) const = default;
};

// A weight's role: the pipeline stage, the component's index in the spec,
// its weight role, the layer, and whether it is an expert array (all
// experts as one tensor along ne[2]).
struct WeightRef {
  std::uint16_t stage = 0;
  std::uint16_t component = 0;
  std::uint16_t role = 0;
  std::uint32_t layer = 0;
  bool expert_array = false;
  auto operator<=>(const WeightRef&) const = default;
};

// A state tensor: the stage, the tensor's index in its component's state
// contract (model/state_contract.h), the layer and the owner segment.
struct StateRef {
  std::uint16_t stage = 0;
  std::uint16_t tensor = 0;
  std::uint32_t layer = 0;
  std::uint8_t segment = 0;
  auto operator<=>(const StateRef&) const = default;
};

using Binding = std::variant<std::monostate, WeightRef, StateRef>;

// The bytes of a plan constant's parameters that may hold its content's
// SHA-256 (Builder::Constant); every other leaf's parameters are zero.
inline constexpr std::size_t kConstantDigestBytes = 32;

// The segment of joined rows: a tensor no single owner's.
inline constexpr std::uint8_t kJoinedSegment = 0xFF;
inline constexpr std::uint32_t kNoName = ~std::uint32_t{0};

struct TensorDesc {
  Role role = Role::kActivation;
  // Its encoding, layout and logical shape. An activation is a dense
  // encoding in kDense.
  format::TensorFormat format{};
  // Byte strides in GGML's order: nb[0] the bytes of one element (dense)
  // or one block (a block layout), nb[1] a row's, and so on. All zero for
  // a format stored in several parts, which has no strides (and no views).
  std::array<std::uint64_t, format::kMaxDims> nb{};
  // The root whose bytes this tensor views, and where; none for a root.
  TensorId view_of{};
  std::uint64_t view_offset = 0;
  Binding binding{};
  // The owner segment whose rows it holds, or kJoinedSegment.
  std::uint8_t segment = kJoinedSegment;
  std::uint32_t name = kNoName;  // Graph::name()

  const format::Shape& shape() const { return format.shape; }
  std::uint64_t ne(std::size_t d) const { return format.shape.ne.at(d); }
  bool is_view() const { return view_of.valid(); }
};

// The bytes one stride unit covers: an element of a dense encoding, a block
// of a block layout; 0 for a format with several parts or whose rows are
// not made of uniform units.
std::uint64_t UnitBytes(const format::TensorFormat& format);
// The values one unit holds along ne[0] (1 for dense encodings).
std::uint64_t UnitElements(const format::TensorFormat& format);
// Checked storage envelope for a valid tensor descriptor. Excludes backend
// readable-tail and write padding; those remain the selected recipe's contract.
std::expected<std::uint64_t, std::string> StorageBytes(const TensorDesc& tensor);
// A root's natural strides: units in order, rows packed (all zero where
// UnitBytes is 0).
std::array<std::uint64_t, format::kMaxDims> NaturalStrides(const format::TensorFormat& format);
// Whether its addressed units are packed in logical order. Strides of
// singleton axes (one block on axis zero) are unobservable and may differ.
bool IsContiguous(const TensorDesc& tensor);
// Whether a kView re-emitted over `now` in place of its original source
// `was` (a copy pass's substituted source: an expansion's, a rewrite's)
// reads the same elements. A view's description holds strides and an
// offset into its source's root, so the source must be laid out alike:
// the same format, strides and offset into its root. A pass refuses a view
// whose source changed layout rather than read other elements.
bool SameViewSource(const TensorDesc& was, const TensorDesc& now);

// MXFP8 rows: activations prepared for a block-scaled MXFP8 product (the
// codes, then the scales in CUTLASS's atoms, in one root; kPrepare's
// format). A fused kind's computed output may be recorded in them (its
// decomposition's result then prepared).
bool PreparedRows(const format::TensorFormat& format);
// The number of values.
std::uint64_t Elements(const format::Shape& shape);
// A dense format of `shape`.
format::TensorFormat Dense(format::Encoding encoding, const format::Shape& shape);
// A shape of ne (innermost first), rank = ne.size() (at most kMaxDims).
format::Shape ShapeOf(std::initializer_list<std::uint64_t> ne);

inline constexpr std::size_t kMaxSources = 10;
inline constexpr std::size_t kMaxOutputs = 4;

// Whether a node's rows are joined across requests (products, routed
// experts) or each owner's own (attention over one request's cache,
// recurrences); a per-owner node's outputs carry its owner segment.
enum class Batch : std::uint8_t { kJoined, kPerOwner };

struct Node {
  OpKey key{};
  std::array<std::byte, kMaxParamBytes> params{};
  std::array<TensorId, kMaxSources> src{};
  std::uint8_t nsrc = 0;  // positions, an absent optional one included
  std::array<TensorId, kMaxOutputs> out{};
  std::uint8_t nout = 0;
  std::uint16_t component = 0;  // the spec component that emitted it
  Batch batch = Batch::kJoined;

  std::span<const TensorId> sources() const { return std::span(src).first(nsrc); }
  std::span<const TensorId> outputs() const { return std::span(out).first(nout); }
  // The source at position i, or none past nsrc.
  TensorId source(std::size_t i) const { return i < nsrc ? src.at(i) : TensorId{}; }
  template <typename P>
  P Params() const {
    static_assert(kIsParams<P>);
    P p{};
    std::memcpy(&p, params.data(), sizeof(P));
    return p;
  }
  template <typename P>
  void SetParams(const P& p) {
    static_assert(kIsParams<P>);
    params = {};
    std::memcpy(params.data(), &p, sizeof(P));
  }
};

class Graph {
 public:
  static constexpr std::size_t kMaxNodes = std::size_t{1} << 16;
  static constexpr std::size_t kMaxTensors = std::size_t{1} << 17;
  using Builder = ir::Builder;

  // A move carries the identity; the moved-from graph has none.
  Graph(Graph&& other) noexcept;
  Graph& operator=(Graph&& other) noexcept;
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  ~Graph() = default;

  std::span<const Node> nodes() const { return nodes_; }
  const Node& node(NodeId id) const { return nodes_.at(id.v); }
  std::size_t tensor_count() const { return tensors_.size(); }
  const TensorDesc& tensor(TensorId id) const { return tensors_.at(id.v); }
  // The node that makes the tensor.
  NodeId producer(TensorId id) const { return producers_.at(id.v); }
  // The tensor itself for a root, else the root it views.
  TensorId root(TensorId id) const;
  // Leaves of role kInput or kConstant, in node order: the step's binding
  // interface.
  std::span<const TensorId> inputs() const { return inputs_; }
  // The step's results, and tensors kept past the step (features taps).
  std::span<const TensorId> outputs() const { return outputs_; }
  std::span<const TensorId> keep() const { return keep_; }
  std::string_view name(TensorId id) const;

  // The canonical digest (see the header comment).
  base::Sha256Digest Digest() const;

  // A copy (graphs are move-only so that a copy is always explicit): a
  // planner consumes one graph while its lowering reads the other. Refused
  // (kBounds) for a moved-from graph.
  std::expected<Graph, VerifyError> Clone() const;

  // This graph's identity in the process: unique to each built or cloned
  // graph (graphs never change once built), never reused, so a cache of
  // facts about one graph keyed on it never confuses another, even one
  // built where a freed graph was. Zero: a moved-from graph.
  std::uint64_t identity() const { return identity_; }

 private:
  friend class ir::Builder;
  Graph() = default;
  static std::uint64_t NextIdentity();
  base::Sha256Digest ComputeDigest() const;

  std::vector<Node> nodes_;
  std::vector<TensorDesc> tensors_;
  std::vector<NodeId> producers_;  // by tensor
  std::vector<TensorId> inputs_;
  std::vector<TensorId> outputs_;
  std::vector<TensorId> keep_;
  std::vector<std::string> names_;
  base::Sha256Digest digest_{};
  std::uint64_t identity_ = 0;
};

}  // namespace llmp::ir

#endif  // LLMP_IR_GRAPH_H_
