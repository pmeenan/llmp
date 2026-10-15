// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include "kernels/ggml/lower.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "components/step.h"
#include "execution/registry.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ggml.h"
#include "ir/decompose.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/ggml/attention_f32.h"
#include "kernels/ggml/conv2d_tiled.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/dsv4_outa.h"
#include "kernels/ggml/dsv4_qhead.h"
#include "kernels/ggml/dsv4_weighted_reduce.h"
#include "kernels/ggml/fattn_owner.h"
#include "kernels/ggml/gemma_moe_fusion.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/moe_layout.h"
#include "kernels/ggml/mxfp8_cutlass.h"
#include "kernels/ggml/set_rows_group.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"
#include "planner/placement.h"
#include "planner/plan.h"
#include "planner/select.h"

namespace llmp::kernels::ggml {
namespace {

namespace co = llmp::components;
using format::Encoding;
using ir::FusedKind;
using ir::NodeId;
using ir::Op;
using ir::TensorId;

std::unexpected<KernelFailure> Refused(std::string detail) {
  return std::unexpected(KernelFailure{.error = KernelError::kRejected,
                                       .detail = "lowering onto GGML: " + std::move(detail)});
}

ggml_tensor* TypedView(ggml_context* c, ggml_tensor* blob, ggml_type type, std::int64_t ne0,
                       std::int64_t ne1, std::size_t offset);

// A finite native output layout, checked even when callers ask allocation
// helpers about a directly emitted, unsupported Route. No unchecked product.
std::uint64_t SelectedRoutePlane(const ir::Graph& graph, const ir::Node& node) {
  if (node.key != ir::Primitive(Op::kRoute) || node.nout != 2 || !node.src[0].valid()) return 0;
  const auto p = node.Params<ir::RouteParams>();
  const bool biased = p.kind == ir::RouteKind::kSqrtSoftplusBias;
  if ((!biased && p.kind != ir::RouteKind::kHash) || p.experts != 256 || p.used < 1 ||
      p.used > 32 || !std::isfinite(p.scale) || (biased && p.scale <= 0.0F))
    return 0;
  if (biased ? (node.nsrc != 2 || !node.src[1].valid())
             : (node.nsrc != 3 || node.src[1].valid() || !node.src[2].valid()))
    return 0;
  const auto rows = graph.tensor(node.src[0]).ne(1);
  if (rows < 1 || rows > (biased ? static_cast<std::uint64_t>(kVecQMaxTokens) : 65535U)) return 0;
  const auto packed = [&](TensorId id, Encoding encoding, std::uint64_t width,
                          std::uint64_t count) {
    if (!id.valid()) return false;
    const auto& t = graph.tensor(id);
    return t.format.encoding == encoding && t.format.layout == format::Layout::kDense &&
           t.ne(0) == width && t.ne(1) == count && t.ne(2) == 1 && t.ne(3) == 1 &&
           t.view_offset == 0 && ir::IsContiguous(t);
  };
  if (!packed(node.src[0], Encoding::kF32, 256, rows) ||
      !packed(node.src[biased ? 1 : 2], biased ? Encoding::kF32 : Encoding::kI32,
              biased ? 256U : p.used, biased ? 1U : rows) ||
      !packed(node.out[0], Encoding::kI32, p.used, rows) ||
      !packed(node.out[1], Encoding::kF32, p.used, rows))
    return 0;
  return static_cast<std::uint64_t>(p.used) * rows * sizeof(float);
}

// A commit's input of a verify's save: its saver's layer.
std::uint32_t SavedLayer(const co::StepInput& input) { return input.layer; }

// Placeholder addresses, as PlaceAndPlan's first pass gives (planned.h):
// activations of a measurement from kDistinct, leaves without an address
// (a test's) from kLeaves, each its own range.
constexpr std::uint64_t kDistinct = std::uint64_t{1} << 46U;
constexpr std::uint64_t kLeaves = kDistinct - (std::uint64_t{1} << 40U);

std::uint64_t Round(std::uint64_t bytes, std::uint64_t alignment) {
  return (bytes + alignment - 1) / alignment * alignment;
}

// The columns a convolution's input is padded with (RE-032): past 32
// tokens, to whole 32-token windows.
std::int64_t SsmConvPad(const ir::Graph& graph, const ir::Node& n) {
  const auto nt = static_cast<std::int64_t>(graph.tensor(n.out[0]).ne(1));
  return nt > 32 && nt % 32 != 0 ? 32 - (nt % 32) : 0;
}

bool PackedProductPair(const ir::Graph& graph, const ir::Node& n) {
  if (n.key != ir::Fused(FusedKind::kPreparedProducts) || n.nsrc != 3 || n.nout != 2) return false;
  const auto& a = graph.tensor(n.out[0]);
  const auto& b = graph.tensor(n.out[1]);
  return graph.tensor(n.src[0]).format.encoding == Encoding::kExl3Trellis &&
         graph.tensor(n.src[1]).format.encoding == Encoding::kExl3Trellis &&
         a.format.encoding == Encoding::kF32 && a.format == b.format && a.ne(2) == 1 &&
         a.ne(3) == 1 && ir::IsContiguous(a) && ir::IsContiguous(b);
}

std::optional<ggml_type> TypeOf(Encoding e) {
  switch (e) {
    case Encoding::kF32:
      return GGML_TYPE_F32;
    case Encoding::kF16:
      return GGML_TYPE_F16;
    case Encoding::kBF16:
      return GGML_TYPE_BF16;
    case Encoding::kF64:
      return GGML_TYPE_F64;
    case Encoding::kI8:
      return GGML_TYPE_I8;
    case Encoding::kI16:
      return GGML_TYPE_I16;
    case Encoding::kI32:
      return GGML_TYPE_I32;
    case Encoding::kI64:
      return GGML_TYPE_I64;
    default:
      break;
  }
  const format::EncodingInfo& info = format::Info(e);
  if (!info.quantized || !info.ggml.has_value()) {
    return std::nullopt;
  }
  return static_cast<ggml_type>(info.ggml->type);
}

// The planned graph's tensors as GGML tensors, and what each step stands
// for (see the header).
class Lowering {
 public:
  Lowering(const LowerInput& in, const execution::Registry& registry, const DeviceChoices& device,
           const LowerOptions& options)
      : in_(in),
        plan_(in.plan),
        p_(in.plan.graph),
        g_(in.io.graph),
        registry_(registry),
        device_(device),
        options_(options) {}

  std::expected<LoweredStep, KernelFailure> Run();

 private:
  // A source of a node being lowered: a GGML tensor already made.
  using Resolve = std::function<ggml_tensor*(TensorId)>;

  std::expected<void, KernelFailure> Index();
  std::expected<void, KernelFailure> MakeNode(NodeId id);
  std::expected<void, KernelFailure> MakeSelected(NodeId id, const ir::Node& n);
  std::expected<void, KernelFailure> MakeFused(NodeId id, const ir::Node& n);
  std::expected<void, KernelFailure> MakeOwners(NodeId id, const ir::Node& n);
  // The composite ops an implementation of their own runs (pattern.h
  // route_reduce): GGML's Gemma routing and scaled reduce over the node
  // sequences their fusion gates take (gemma_moe_fusion.h).
  std::expected<void, KernelFailure> MakeRoute(NodeId id, const ir::Node& n);
  std::expected<void, KernelFailure> MakeReduce(NodeId id, const ir::Node& n);
  // The replaced primitive nodes of later fused nodes that come before
  // primitive node `limit`, made now where their sources are (see Run).
  std::expected<void, KernelFailure> Flush(NodeId limit);
  std::expected<void, KernelFailure> MakeReplaced(std::size_t entry);
  ggml_tensor* ResolveReplaced(TensorId primitive) const;
  std::expected<ggml_tensor*, KernelFailure> MakeLeaf(const ir::Node& n);
  // A weight of several parts (MXFP8's codes and scales, an NVFP4 table's
  // rows and scale, the CUTLASS experts' slab): a GGML tensor a part, the
  // first its stand-in.
  std::expected<ggml_tensor*, KernelFailure> MakeParts(TensorId out, const ir::TensorDesc& d,
                                                       const co::StepWeight& w);
  // GGML's builder call for primitive node `n` of `graph` over its
  // resolved sources (null for what it makes no tensor of: a mask's view).
  std::expected<ggml_tensor*, KernelFailure> Make(const ir::Graph& graph, const ir::Node& n,
                                                  const Resolve& resolve);
  std::expected<ggml_tensor*, KernelFailure> Mask(const ir::Graph& graph, const ir::Node& n,
                                                  const Resolve& resolve);
  std::expected<void, KernelFailure> Matches(const ir::TensorDesc& want, const ggml_tensor* got,
                                             std::string_view what) const;
  std::expected<void, KernelFailure> Bind();
  std::expected<void, KernelFailure> MakeRouting();
  std::expected<void, KernelFailure> PlanAndCompare();
  std::expected<void, KernelFailure> CheckSelectedHc();
  std::expected<void, KernelFailure> BindSelectedHc();
  std::expected<void, KernelFailure> CheckSelectedHcRecipe(const GraphPlan& graph) const;
  std::expected<void, KernelFailure> CheckOwnerProducts();
  std::expected<void, KernelFailure> BindOwnerProducts();
  std::expected<void, KernelFailure> CheckOwnerRecipes(const GraphPlan& plan) const;
  std::expected<void, KernelFailure> Slots();
  // `t` in the graph, standing for planned node `planned`; a computed node
  // among those it launches, and with `step` any node (a fused window's
  // views and reshapes, which PlanGraph's step lists too).
  void Node(ggml_tensor* t, NodeId planned, bool step = false);

  // Whether every reader of planned tensor `t` is a mask the lowering makes
  // without it (the producer reads the whole positions input; a host mask
  // reads nothing on the device).
  bool MaskOnly(TensorId t) const;

  // The fast forms (slice G3b): a fused node llmpalooza's kernels run over
  // GGML tensors (one or several custom nodes, not the primitive nodes its
  // rewrite replaced), the products, conversions and lookups of the
  // fast forms' formats, a verify's saves, and the nodes made at all.
  static constexpr std::size_t kNoImplementation = ~std::size_t{0};
  std::string_view Selected(NodeId id) const;
  bool Direct(NodeId id) const;
  std::expected<void, KernelFailure> MakeDirect(NodeId id, const ir::Node& n);
  std::expected<void, KernelFailure> MakeSpecial(NodeId id, const ir::Node& n, bool& made);
  std::expected<void, KernelFailure> MakeSaves();
  // The specialized QSA kernels derive position tables rather than
  // reading every IR operand. Prove those values from the typed staging
  // interface and graph structure before eliding their operands.
  std::expected<void, KernelFailure> CheckQsaInputs();
  void Needed();
  ggml_tensor* Part(TensorId planned, format::PartRole role) const;
  ggml_tensor* StoredInto(const ggml_tensor* leaf) const;
  void Inside(ggml_tensor* t, TensorId root, std::uint64_t offset);
  ggml_tensor* SegmentPositions(const ir::Node& n, std::size_t i, NodeId id);
  void OwnersAlone(ggml_tensor* vecq) const;

  const LowerInput& in_;
  const planner::StepPlan& plan_;
  const ir::Graph& p_;  // the planned graph
  const ir::Graph& g_;  // the primitive graph
  const execution::Registry& registry_;
  const DeviceChoices& device_;
  const LowerOptions& options_;
  ggml_context* c_ = nullptr;

  // Temporary build-time audit tables, bounded by the graph's products and
  // owners. They never enter the returned program or capture lifetime.
  struct OwnerProduct {
    const co::OwnerProduct* original = nullptr;
    const co::OwnerColumns* owner = nullptr;
    TensorId result{};
  };
  std::vector<OwnerProduct> owner_products_;
  std::map<const ggml_tensor*, std::string_view> owner_recipes_;

  struct SelectedHc {
    TensorId rows{}, norm{}, down{}, up{};
    NodeId low{}, high{};
  };
  static_assert(sizeof(SelectedHc) <= 24);
  static_assert(sizeof(std::optional<SelectedHc>) <= 32);
  std::optional<SelectedHc> selected_hc_{};

  LoweredStep out_;
  std::vector<ggml_tensor*> made_;            // by planned tensor
  std::vector<TensorId> primitive_;           // by planned tensor: the primitive's, where it is one
  std::vector<std::vector<NodeId>> readers_;  // by planned tensor (roots and views alike)
  std::vector<std::vector<ggml_tensor*>> launched_;       // by planned node: the nodes it launches
  std::map<std::uint32_t, const co::StepInput*> inputs_;  // by primitive tensor
  std::map<std::uint32_t, const co::StepWeight*> weights_;
  std::map<std::uint32_t, const co::StepState*> state_;
  std::map<std::uint32_t, std::size_t> applied_;  // planned fused node -> Applied
  // Every fused node's replaced primitive nodes, in primitive order, and
  // what each made (by primitive tensor).
  struct Replaced {
    NodeId primitive{};
    NodeId fused{};  // the planned node
    bool made = false;
  };
  std::vector<Replaced> replaced_;
  std::size_t cursor_ = 0;
  std::map<std::uint32_t, ggml_tensor*> replaced_out_;
  std::map<std::uint32_t, std::vector<std::pair<ggml_tensor*, TensorId>>> fused_made_;
  // Leaves with their binding: weights and state by reference.
  std::vector<std::pair<ggml_tensor*, const co::StepWeight*>> weight_leaves_;
  std::vector<std::pair<ggml_tensor*, const co::StepState*>> state_leaves_;
  struct StateBank {
    ggml_tensor* tensor = nullptr;
    std::array<ir::StateRef, 2> refs{};
    std::array<ggml_tensor*, 2> roots{};
    std::array<planner::StateCellSlice, 2> slices{};
    planner::StateSpanFact fact{};
    std::uint64_t raw_bytes = 0;
  };
  std::vector<StateBank> state_banks_;
  std::set<const ggml_tensor*> bank_attentions_;
  // Host masks and staged inputs: the slot each takes.
  std::vector<std::pair<ggml_tensor*, InputSlot>> input_leaves_;
  // A fused implementation's intermediates, which it never writes, and the
  // result whose place they take.
  struct Elided {
    ggml_tensor* tensor;
    ggml_tensor* result;
    // Only RopeStore: the fused launcher never accesses the RoPE output.
    bool read_only_operand = false;
  };
  static_assert(sizeof(Elided) <= 32);
  std::vector<Elided> elided_;
  // A planned tensor bound through another GGML tensor: a view's root (the
  // fused routing's ids, a view of its whole sort).
  std::map<std::uint32_t, ggml_tensor*> placed_root_;
  std::set<const ggml_tensor*> state_tensors_, weight_tensors_;

  std::vector<std::size_t> impl_of_;  // by planned node: its step's implementation
  std::vector<bool> needed_;          // by planned node: made
  std::vector<bool> quiet_;           // by planned node: launches nothing of its own
  bool dsv4_fast_ = false;
  // A multi-part weight's parts' tensors, by planned tensor.
  std::map<std::uint32_t, std::map<format::PartRole, ggml_tensor*>> parts_;
  std::vector<std::tuple<ggml_tensor*, const co::StepWeight*, format::PartRole>> part_leaves_;
  std::map<const ggml_tensor*, ggml_tensor*> stores_;  // a cache leaf: its latest store
  std::map<const ggml_tensor*, ggml_tensor*> pools_;   // block keys: llmp.qsa.pool's node
  struct InsideRoot {
    ggml_tensor* t = nullptr;
    TensorId root{};
    std::uint64_t offset = 0;
  };
  std::vector<InsideRoot> inside_;          // intermediates in a result's place
  std::map<std::uint32_t, NodeId> folded_;  // a preparation: the node whose kernel writes it
  std::set<const ggml_tensor*> extra_;      // the lowering's own nodes (a verify's saves)
  std::vector<std::pair<ggml_tensor*, const co::CommitSave*>> save_leaves_;
  std::vector<std::pair<ggml_tensor*, const co::StepInput*>> external_;  // a commit's saved rows
  // CUTLASS expert slabs by (stage, layer): one GGML tensor for a layer's
  // gate-and-up and down arrays.
  std::map<std::pair<std::uint16_t, std::uint32_t>, ggml_tensor*> slabs_;
  // Exact masks may reuse replicated positions only after proving that
  // their first section equals the mask's original plain position input.
  std::map<std::uint32_t, TensorId> exact_mask_positions_;
};

std::expected<void, KernelFailure> Lowering::CheckQsaInputs() {
  if (!in_.token) return {};
  // Inferred tables must retain their staged/MaskBuild values. A name or
  // declaration is not a proof once another node overwrites its bytes.
  std::set<TensorId> overwritten;
  for (const ir::Node& n : p_.nodes()) {
    if (n.key == ir::Primitive(Op::kCacheStore)) overwritten.insert(p_.root(n.src[0]));
    for (std::size_t i = 0; i < n.nout; ++i) {
      if (const auto alias = ir::AliasedSource(n.key, i); alias && n.src.at(*alias).valid())
        overwritten.insert(p_.root(n.src.at(*alias)));
    }
  }
  const auto input = [&](TensorId t) -> const co::StepInput* {
    if (!t.valid()) return nullptr;
    const TensorId root = p_.root(t);
    if (p_.node(p_.producer(root)).key != ir::Primitive(Op::kInput) || overwritten.contains(root)) {
      return nullptr;
    }
    const TensorId original = primitive_.at(root.v);
    const auto at = inputs_.find(original.v);
    return at == inputs_.end() ? nullptr : at->second;
  };
  const auto plain_cells = [](const co::CellKey& c, std::uint64_t capacity, std::uint32_t block) {
    return c == co::CellKey{.capacity = capacity, .block = block};
  };
  // InputKind, not a tensor's name, is what ModelRunner stages using
  // InputValues: four copies of every row's position, in section order.
  const auto positions = [&](TensorId t, std::uint8_t segment, bool whole) {
    const co::StepInput* in = input(t);
    if (in == nullptr || in->kind != co::InputKind::kPositions || in->sections != 4 ||
        in->segment != ir::kJoinedSegment)
      return false;
    const ir::TensorDesc& td = p_.tensor(t);
    const ir::TensorDesc& root = p_.tensor(p_.root(t));
    if (root.shape().ne != ir::ShapeOf({4ULL * in_.token->shape.rows()}).ne ||
        root.format.encoding != Encoding::kI32 || !ir::IsContiguous(root) || td.nb[0] != 4) {
      return false;
    }
    if (whole) return !td.is_view() && t == p_.root(t);
    if (segment >= in_.token->shape.segments.size()) return false;
    const std::uint64_t rows = in_.token->shape.segments[segment].rows;
    std::uint64_t first = 0;
    for (std::size_t s = 0; s < segment; ++s) first += in_.token->shape.segments[s].rows;
    return td.view_offset == 4 * first && td.ne(0) == (td.is_view() ? rows : 4 * rows) &&
           td.ne(1) == 1 && td.ne(2) == 1 && td.ne(3) == 1 &&
           (td.is_view() || in_.token->shape.segments.size() == 1);
  };
  const auto plain_positions = [&](TensorId t, std::uint8_t segment) {
    const co::StepInput* in = input(t);
    if (in == nullptr || in->kind != co::InputKind::kPositions || in->sections != 1 ||
        in->segment != ir::kJoinedSegment || segment >= in_.token->shape.segments.size())
      return false;
    const ir::TensorDesc& td = p_.tensor(t);
    const ir::TensorDesc& root = p_.tensor(p_.root(t));
    std::uint64_t first = 0;
    for (std::size_t s = 0; s < segment; ++s) first += in_.token->shape.segments[s].rows;
    return td.shape().ne == ir::ShapeOf({in_.token->shape.segments[segment].rows}).ne &&
           root.shape().ne == ir::ShapeOf({in_.token->shape.rows()}).ne &&
           td.view_offset == first * 4 && td.nb[0] == 4 && root.format.encoding == Encoding::kI32 &&
           ir::IsContiguous(root);
  };
  // The latest write to a root, including writes through aliases. This
  // prevents a different in-place operation from being mistaken for the
  // store or pool whose values the custom kernel consumes.
  const auto last_write = [&](TensorId root, NodeId before) -> const ir::Node* {
    const ir::Node* found = nullptr;
    for (std::uint32_t k = 0; k < before.v; ++k) {
      const ir::Node& n = p_.node(NodeId{k});
      if (n.key == ir::Primitive(Op::kCacheStore) && p_.root(n.src[0]) == root) found = &n;
      for (std::size_t i = 0; i < n.nout; ++i) {
        if (const auto alias = ir::AliasedSource(n.key, i);
            alias && n.src.at(*alias).valid() && p_.root(n.src.at(*alias)) == root)
          found = &n;
      }
    }
    return found;
  };
  const auto stored = [&](TensorId t, std::uint8_t segment, NodeId before) {
    const TensorId root = p_.root(t);
    const ir::TensorDesc& rd = p_.tensor(root);
    const TensorId original = primitive_.at(root.v);
    const auto state = state_.find(original.v);
    const ir::Node* write = last_write(root, before);
    if (state == state_.end() || state->second->borrowed || state->second->ref.segment != segment ||
        write == nullptr || write->key != ir::Primitive(Op::kCacheStore) || write->src[0] != root ||
        rd.ne(2) != 1 || rd.ne(3) != 1)
      return false;
    const co::StepInput* cells = input(write->src[2]);
    return cells != nullptr && cells->kind == co::InputKind::kStoreCells &&
           cells->segment == segment && plain_cells(cells->cells, rd.ne(1), 0) &&
           write->src[2] == p_.root(write->src[2]) &&
           p_.tensor(write->src[2]).format.encoding == Encoding::kI64 &&
           p_.tensor(write->src[2]).shape().ne ==
               ir::ShapeOf({in_.token->shape.segments[segment].rows}).ne &&
           p_.tensor(write->src[1]).ne(1) == in_.token->shape.segments[segment].rows;
  };
  const auto mask = [&](TensorId t, std::uint8_t segment, std::uint64_t count,
                        std::uint64_t capacity, std::uint32_t block, Encoding encoding) {
    const ir::Node& m = p_.node(p_.producer(t));
    if (m.key != ir::Primitive(Op::kMaskBuild) || overwritten.contains(p_.root(t))) return false;
    const auto mp = m.Params<ir::MaskParams>();
    const co::StepInput* held = input(m.src[1]);
    const ir::TensorDesc& td = p_.tensor(t);
    return mp.kind == ir::MaskKind::kCausal && mp.window == 0 && mp.block == 0 && held != nullptr &&
           held->kind == co::InputKind::kCellPositions && held->segment == segment &&
           held->read == count && plain_cells(held->cells, capacity, block) &&
           m.src[1] == p_.root(m.src[1]) &&
           p_.tensor(m.src[1]).shape().ne == ir::ShapeOf({count}).ne &&
           plain_positions(m.src[0], segment) && td.format.encoding == encoding &&
           td.ne(0) == count && td.ne(1) >= in_.token->shape.segments[segment].rows;
  };
  for (std::uint32_t k = 0; k < p_.nodes().size(); ++k) {
    const NodeId id{k};
    const ir::Node& n = p_.node(id);
    if (n.key == ir::Primitive(Op::kMaskBuild)) {
      const auto mp = n.Params<ir::MaskParams>();
      const co::StepInput* held = input(n.src[1]);
      if (mp.kind == ir::MaskKind::kAhead || (held != nullptr && held->cells.positions)) {
        const auto& out = p_.tensor(n.out[0]);
        const bool block = held != nullptr && held->cells.positions;
        if (held == nullptr || held->kind != co::InputKind::kCellPositions ||
            n.src[1] != p_.root(n.src[1]) || held->read != out.ne(0) ||
            held->read > held->cells.capacity || held->cells.capacity == 0 ||
            held->cells.capacity >
                static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
            held->cells.fixed || held->cells.per_row || held->cells.first_row != 0 ||
            held->cells.frozen || ((held->cells.block == 0) == block) ||
            (block && (held->cells.ring || held->cells.recent || mp.kind != ir::MaskKind::kCausal ||
                       mp.window != 0)) ||
            (!block && !held->cells.ring) || mp.block != 0 || mp.exact_rows ||
            mp.window > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
            p_.tensor(n.src[1]).format.encoding != Encoding::kI32 ||
            p_.tensor(n.src[1]).shape().ne != ir::ShapeOf({out.ne(0)}).ne ||
            !plain_positions(n.src[0], held->segment) || overwritten.contains(p_.root(n.out[0])) ||
            out.format.encoding != Encoding::kF16) {
          return Refused("a compressed or ahead mask requires its typed owner positions and cells");
        }
        const auto& segment = in_.token->shape.segments[held->segment];
        if (std::uint64_t{segment.past} + segment.rows >
                static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
            out.shape().ne != ir::ShapeOf({held->read, Round(segment.rows, 32)}).ne) {
          return Refused("a compressed or ahead mask exceeds its bounded rows");
        }
      }
    }
    if (options_.device_masks && n.key == ir::Primitive(Op::kMaskBuild) &&
        n.Params<ir::MaskParams>().exact_rows) {
      const auto mp = n.Params<ir::MaskParams>();
      const co::StepInput* held = input(n.src[1]);
      const ir::TensorDesc& out = p_.tensor(n.out[0]);
      if ((mp.kind != ir::MaskKind::kCausal && mp.kind != ir::MaskKind::kWindow) || mp.block != 0 ||
          mp.window > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
          overwritten.contains(p_.root(n.out[0])) || held == nullptr ||
          held->kind != co::InputKind::kCellPositions || n.src[1] != p_.root(n.src[1]) ||
          held->read != out.ne(0) || held->cells.capacity == 0 ||
          held->cells.capacity >
              static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
          held->read > held->cells.capacity ||
          held->cells !=
              co::CellKey{.capacity = held->cells.capacity, .frozen = held->cells.frozen} ||
          p_.tensor(n.src[1]).format.encoding != Encoding::kI32 ||
          p_.tensor(n.src[1]).shape().ne != ir::ShapeOf({out.ne(0)}).ne ||
          !ir::IsContiguous(p_.tensor(n.src[1])) || !plain_positions(n.src[0], held->segment) ||
          std::uint64_t{in_.token->shape.segments[held->segment].past} +
                  in_.token->shape.segments[held->segment].rows >
              held->cells.capacity ||
          out.shape().ne !=
              ir::ShapeOf({out.ne(0), in_.token->shape.segments[held->segment].rows}).ne ||
          out.format.encoding != Encoding::kF16) {
        return Refused("an exact device mask requires typed joined positions for its owner");
      }
      // Keep the original root if the staging interface contains several
      // multi-section inputs, or its sole one does not prove equivalent.
      // Unrelated declarations must never select a mask's positions.
      TensorId candidate{};
      std::size_t multi = 0;
      for (const co::StepInput& in : in_.token->step.inputs) {
        if (in.kind != co::InputKind::kPositions || in.sections <= 1) continue;
        ++multi;
        const TensorId t = plan_.tensors.at(in.tensor.v);
        if (t.valid() && positions(t, ir::kJoinedSegment, true)) candidate = t;
      }
      exact_mask_positions_[n.out[0].v] =
          multi == 1 && candidate.valid() ? candidate : p_.root(n.src[0]);
    }
    if (n.key == ir::Fused(FusedKind::kDsv4Compress)) {
      const auto p = n.Params<ir::CompressorParams>();
      const auto* reads = input(n.src[4]);
      const auto owner = p_.tensor(n.out[0]).segment;
      if (reads == nullptr || reads->kind != co::InputKind::kPoolReads || reads->segment != owner ||
          owner >= in_.token->shape.segments.size() || reads->ratio != p.ratio ||
          reads->overlap != p.overlap || n.src[4] != p_.root(n.src[4]) ||
          reads->ring != p_.tensor(n.src[0]).ne(1) ||
          p_.tensor(n.src[2]).ne(1) != in_.token->shape.segments[owner].rows) {
        return Refused("a compressor requires its owner's typed bounded pool reads");
      }
    }
    if (n.key == ir::Fused(FusedKind::kRmsNormMulRope) && Selected(id) == kQsaPrepName) {
      if (!positions(n.src[2], ir::kJoinedSegment, true) ||
          p_.tensor(n.src[0]).ne(2) != in_.token->shape.rows()) {
        return Refused("QSA prep requires typed replicated position sections for its rows");
      }
    }
    if (n.key != ir::Fused(FusedKind::kQsaPool) && n.key != ir::Fused(FusedKind::kQsaAttention))
      continue;
    const std::uint8_t segment = p_.tensor(n.out[0]).segment;
    if (segment >= in_.token->shape.segments.size())
      return Refused("QSA requires one declared owner");
    if (n.key == ir::Fused(FusedKind::kQsaPool)) {
      const auto pp = n.Params<ir::QsaPoolParams>();
      const ir::TensorDesc& raw = p_.tensor(n.src[0]);
      const ir::TensorDesc& blocks = p_.tensor(n.src[1]);
      const co::StepInput* starts = input(n.src[3]);
      const co::StepInput* cells = input(n.src[4]);
      const std::uint64_t count = p_.tensor(n.src[3]).ne(0) / 4;
      const auto block_input = [&](const co::StepInput* in, co::InputKind kind) {
        return in != nullptr && in->kind == kind && in->segment == segment && in->read == count &&
               plain_cells(in->cells, blocks.ne(1), pp.ratio);
      };
      const TensorId block_root = p_.root(n.src[1]);
      const auto block_state = state_.find(primitive_.at(block_root.v).v);
      if (!positions(n.source(5), segment, false) ||
          !block_input(starts, co::InputKind::kBlockPositions) || starts->sections != 4 ||
          !block_input(cells, co::InputKind::kStoreCells) || n.src[3] != p_.root(n.src[3]) ||
          n.src[4] != p_.root(n.src[4]) || p_.tensor(n.src[4]).format.encoding != Encoding::kI64 ||
          block_state == state_.end() || block_state->second->borrowed ||
          block_state->second->ref.segment != segment || n.src[1] != block_root ||
          !stored(n.src[0], segment, id) || raw.view_offset != 0 || raw.ne(1) != count * pp.ratio ||
          raw.nb[0] != 4 || raw.nb[1] != raw.ne(0) * 4 || raw.ne(2) != 1 || raw.ne(3) != 1) {
        return Refused(
            "QSA pool requires typed block positions/cells and canonical stored raw rows");
      }
      continue;
    }
    const auto ap = n.Params<ir::QsaAttnParams>();
    const ir::TensorDesc& keys = p_.tensor(n.src[1]);
    const ir::TensorDesc& values = p_.tensor(n.src[2]);
    const ir::TensorDesc& blocks = p_.tensor(n.src[4]);
    const std::uint64_t cells = keys.ne(1);
    const TensorId block_root = p_.root(n.src[4]);
    const ir::Node* pool = last_write(block_root, id);
    const auto canonical_cache = [&](TensorId t) {
      const ir::TensorDesc& td = p_.tensor(t);
      const ir::TensorDesc& root = p_.tensor(p_.root(t));
      return stored(t, segment, id) && td.view_offset == 0 && td.ne(3) == 1 &&
             root.ne(0) == td.ne(0) * td.ne(2) && td.nb[0] == 2 && td.nb[1] == root.nb[1] &&
             td.nb[2] == td.ne(0) * 2;
    };
    if (!positions(n.source(8), segment, false) || pool == nullptr ||
        pool->key != ir::Fused(FusedKind::kQsaPool) ||
        pool->Params<ir::QsaPoolParams>().ratio != ap.ratio ||
        p_.tensor(pool->out[0]).segment != segment || p_.root(pool->src[5]) != p_.root(n.src[8]) ||
        p_.tensor(pool->src[5]).view_offset != p_.tensor(n.src[8]).view_offset ||
        p_.tensor(pool->src[5]).shape().ne != p_.tensor(n.src[8]).shape().ne ||
        blocks.view_offset != 0 || blocks.nb[0] != 2 || blocks.nb[1] != blocks.ne(0) * 2 ||
        blocks.ne(2) != 1 || blocks.ne(3) != 1 || cells != blocks.ne(1) * ap.ratio ||
        p_.tensor(pool->src[3]).ne(0) != 4 * blocks.ne(1) || !canonical_cache(n.src[1]) ||
        !canonical_cache(n.src[2]) ||
        p_.tensor(p_.root(n.src[1])).ne(1) != p_.tensor(p_.root(n.src[2])).ne(1) ||
        values.shape().ne != keys.shape().ne ||
        !mask(n.src[5], segment, cells, p_.tensor(p_.root(n.src[1])).ne(1), 0, Encoding::kF32) ||
        !mask(n.src[6], segment, blocks.ne(1), p_.tensor(block_root).ne(1), ap.ratio,
              Encoding::kF32) ||
        !mask(n.src[7], segment, cells, p_.tensor(p_.root(n.src[1])).ne(1), 0, Encoding::kF16)) {
      return Refused(
          "QSA selection requires its typed pool, stored caches and canonical causal masks");
    }
  }
  return {};
}

// Several owners' rows joined in a llmp.vecq product (the plan's
// owner_row_invariant): each token's reduction and weight reads as it is
// alone (one a pass, never shared with another token's), as today's waves
// keep each slot's results its lone step's (qwen38_wave_plan.cc).
void Lowering::OwnersAlone(ggml_tensor* vecq) const {
  if (plan_.owner_row_invariant) SetVecQOneToken(vecq);
}

// The positions a QSA kernel reads (I32 [4 · t], the first t read): the
// step's whole positions, or of one owner of several (the IR's view of
// its rows of the joined positions) the 4 · t from its first row.
ggml_tensor* Lowering::SegmentPositions(const ir::Node& n, std::size_t i, NodeId id) {
  const ir::TensorDesc& d = p_.tensor(n.src.at(i));
  ggml_tensor* made = made_.at(n.src.at(i).v);
  if (!d.is_view()) return made;
  ggml_tensor* root = made_.at(p_.root(n.src.at(i)).v);
  const auto rows = static_cast<std::int64_t>(d.ne(0));
  ggml_tensor* t = ggml_view_1d(c_, root, 4 * rows, d.view_offset);
  Node(t, id);
  return t;
}

void Lowering::Node(ggml_tensor* t, NodeId planned, bool step) {
  out_.nodes.push_back(t);
  if (step || !(t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE ||
                t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE)) {
    launched_.at(planned.v).push_back(t);
  }
}

std::expected<void, KernelFailure> Lowering::Index() {
  made_.assign(p_.tensor_count(), nullptr);
  primitive_.assign(p_.tensor_count(), TensorId{});
  for (std::uint32_t t = 0; t < plan_.tensors.size(); ++t) {
    const TensorId planned = plan_.tensors[t];
    if (planned.valid()) {
      if (planned.v >= primitive_.size()) return Refused("the plan maps a tensor past its graph");
      // A decomposition can return an existing leaf unchanged (hash Route's
      // selected IDs). Its later output is another forward alias, not the
      // input/weight/state identity used by the step's binding interface.
      const auto planned_key = p_.node(p_.producer(planned)).key;
      const bool leaf = planned_key == ir::Primitive(Op::kInput) ||
                        planned_key == ir::Primitive(Op::kWeight) ||
                        planned_key == ir::Primitive(Op::kState);
      if (!leaf || g_.node(g_.producer(TensorId{t})).key == planned_key)
        primitive_[planned.v] = TensorId{t};
    }
  }
  readers_.assign(p_.tensor_count(), {});
  for (std::uint32_t i = 0; i < p_.nodes().size(); ++i) {
    for (const TensorId s : p_.nodes()[i].sources()) {
      if (s.valid()) readers_.at(s.v).push_back(NodeId{i});
    }
  }
  launched_.assign(p_.nodes().size(), {});
  quiet_.assign(p_.nodes().size(), false);
  impl_of_.assign(p_.nodes().size(), kNoImplementation);
  for (const planner::Selected& s : plan_.steps) {
    for (const NodeId id : s.nodes) impl_of_.at(id.v) = s.implementation;
    const std::string_view name = registry_.at(s.implementation).name;
    if (name == kDsv4RouteName || name == "llmp.dsv4.lid_mask" || name == kDsv4QHeadName ||
        name == kDsv4HcMixName || name == kDsv4CompressName || name == kDsv4OutAFastPackName) {
      dsv4_fast_ = true;
    }
  }
  if (in_.token)
    for (const co::StepInput& input : in_.token->step.inputs) {
      if (!input.tensor.valid() || input.tensor.v >= g_.tensor_count() ||
          g_.node(g_.producer(input.tensor)).key != ir::Primitive(Op::kInput) ||
          g_.tensor(input.tensor).is_view() || !inputs_.emplace(input.tensor.v, &input).second) {
        return Refused("the staging interface requires unique input leaves");
      }
    }
  for (const co::StepWeight& w : in_.io.weights) weights_[w.tensor.v] = &w;
  if (in_.token)
    for (const co::StepState& s : in_.token->step.state) state_[s.tensor.v] = &s;
  for (std::size_t a = 0; a < plan_.applied.size(); ++a) {
    for (const NodeId made : plan_.applied[a].made) applied_[made.v] = a;
  }
  // The GGML nodes a fused node runs over, in the primitive graph's order
  // (Run): a fused kind GGML implements over its primitive nodes, from a
  // rewrite that made only it.
  for (std::uint32_t i = 0; i < p_.nodes().size(); ++i) {
    const ir::Node& n = p_.nodes()[i];
    if (n.key.op != Op::kFused || n.key.fused == FusedKind::kOwnerAttention || Direct(NodeId{i}))
      continue;
    const auto at = applied_.find(i);
    if (at == applied_.end())
      return Refused(std::format("{} no rewrite made", ir::OpKeyName(n.key)));
    const planner::Applied& a = plan_.applied[at->second];
    if (a.made.size() != 1) return Refused("a fused node of a rewrite that made several");
    for (const NodeId r : a.replaced)
      replaced_.push_back({.primitive = r, .fused = NodeId{i}, .made = false});
  }
  std::ranges::sort(replaced_, {}, &Replaced::primitive);
  // A preparation into MXFP8 rows of a gated norm's result (through its
  // reshape): llmp.gdn.norm_gate writes the rows itself.
  for (std::uint32_t i = 0; i < p_.nodes().size(); ++i) {
    const ir::Node& n = p_.nodes()[i];
    if (n.key != ir::Primitive(Op::kPrepare) || !ir::PreparedRows(p_.tensor(n.out[0]).format))
      continue;
    const auto alone_read = [&](TensorId t) {
      return readers_.at(t.v).size() == 1 && !std::ranges::contains(p_.outputs(), t) &&
             !std::ranges::contains(p_.keep(), t);
    };
    TensorId t = n.src[0];
    bool alone = true;
    while (alone && p_.node(p_.producer(t)).key == ir::Primitive(Op::kReshape)) {
      alone = alone_read(t);
      t = p_.node(p_.producer(t)).src[0];
    }
    const NodeId from = p_.producer(t);
    if (alone && alone_read(t) && p_.node(from).key == ir::Fused(FusedKind::kNormGate) &&
        Direct(from)) {
      folded_[i] = from;
    }
  }
  return {};
}

bool Lowering::MaskOnly(TensorId t) const {
  const auto& readers = readers_.at(t.v);
  return !readers.empty() && std::ranges::all_of(readers, [&](NodeId r) {
    return p_.node(r).key == ir::Primitive(Op::kMaskBuild);
  });
}

std::expected<void, KernelFailure> Lowering::Matches(const ir::TensorDesc& want,
                                                     const ggml_tensor* got,
                                                     std::string_view what) const {
  const auto type = TypeOf(want.format.encoding);
  const bool f16_q =
      want.format.encoding == Encoding::kF32 && got->type == GGML_TYPE_F16 && got->ne[0] == 512;
  const std::uint64_t scale = f16_q ? 2 : 1;
  bool same = (type.has_value() && got->type == *type) || f16_q;
  for (std::size_t d = 0; d < 4 && same; ++d) {
    same = static_cast<std::uint64_t>(got->ne[d]) == want.ne(d);
    // Strides of views and of whatever has them (a root's are natural).
    if (same && want.nb.at(0) != 0)
      same = static_cast<std::uint64_t>(got->nb[d]) * scale == want.nb.at(d);
  }
  if (!same) {
    return Refused(std::format("{} made {} [{}, {}, {}, {}] nb [{}, {}, {}, {}], not the IR's {}",
                               what, ggml_type_name(got->type), got->ne[0], got->ne[1], got->ne[2],
                               got->ne[3], got->nb[0], got->nb[1], got->nb[2], got->nb[3],
                               format::Info(want.format.encoding).name));
  }
  return {};
}

std::expected<ggml_tensor*, KernelFailure> Lowering::MakeLeaf(const ir::Node& n) {
  const TensorId out = n.out[0];
  const ir::TensorDesc& d = p_.tensor(out);
  const TensorId prim = primitive_.at(out.v);
  if (n.key.op == Op::kWeight && prim.valid()) {
    // A weight of several parts (or one in an encoding GGML has no type
    // for): a leaf a part.
    const auto at = weights_.find(prim.v);
    if (at == weights_.end()) return Refused("a weight the step does not list");
    if (at->second->parts.size() != 1 || d.format.encoding == Encoding::kMxfp8 ||
        (d.format.encoding == Encoding::kNvfp4 &&
         d.format.layout != format::Layout::kGgmlNvfp4Blocks)) {
      return MakeParts(out, d, *at->second);
    }
  }
  const auto type = TypeOf(d.format.encoding);
  if (!prim.valid() || !type) return Refused("a leaf of no primitive tensor or GGML type");
  const bool blocks = d.format.layout == format::Layout::kGgmlBlocks ||
                      d.format.layout == format::Layout::kGgmlNvfp4Blocks;
  if (d.format.layout != format::Layout::kDense && !blocks) {
    return Refused(std::format("a leaf in layout {}", format::Info(d.format.layout).name));
  }
  std::array<std::int64_t, 4> ne{};
  for (std::size_t i = 0; i < 4; ++i) ne.at(i) = static_cast<std::int64_t>(d.ne(i));
  const int rank = std::max<int>(1, d.shape().rank);
  if (n.key.op == Op::kInput) {
    if (!in_.token) {
      if (!std::ranges::contains(in_.io.inputs, prim) || d.is_view() ||
          d.segment == ir::kJoinedSegment)
        return Refused("a spatial input is not an exact owned graph port");
      ggml_tensor* t = ggml_new_tensor(c_, *type, rank, ne.data());
      input_leaves_.emplace_back(
          t, InputSlot{.tensor = prim, .segment = d.segment, .kind = InputKind::kStepBytes});
      out_.leaves.push_back({.tensor = t, .primitive = prim});
      return t;
    }
    const auto at = inputs_.find(prim.v);
    if (at == inputs_.end()) return Refused("an input the step does not list");
    const co::StepInput& input = *at->second;
    InputSlot slot{.tensor = prim, .segment = input.segment};
    switch (input.kind) {
      case co::InputKind::kTokens:
        slot.kind = InputKind::kTokens;
        break;
      case co::InputKind::kPositions:
        slot.kind = InputKind::kPositions;
        slot.sections = std::max<std::uint32_t>(input.sections, 1);
        break;
      case co::InputKind::kOutputRows:
        slot.kind = InputKind::kOutputRows;
        break;
      case co::InputKind::kStoreCells:
        slot.kind = InputKind::kStoreCells;
        slot.cells = {.capacity = input.cells.capacity,
                      .ring = input.cells.ring,
                      .fixed = input.cells.fixed,
                      .block = input.cells.block,
                      .per_row = input.cells.per_row,
                      .first_row = input.cells.first_row};
        slot.read = input.read;
        break;
      case co::InputKind::kCellPositions:
        // Read only by masks, which derive (device) or fold in (host) them.
        if (!MaskOnly(out)) return Refused("cells' positions read by more than a mask");
        return nullptr;
      case co::InputKind::kStream:
      case co::InputKind::kResidual:
      case co::InputKind::kCarry:
      case co::InputKind::kSide:
        return Refused("a lone component's input");
      case co::InputKind::kSideRows:
        slot.kind = InputKind::kSideRows;
        slot.count = input.read;
        slot.component = input.component;
        break;
      case co::InputKind::kBlockPositions:
        slot.kind = InputKind::kBlockPositions;
        slot.cells = {.capacity = input.cells.capacity, .block = input.cells.block};
        slot.read = input.read;
        slot.sections = std::max<std::uint32_t>(input.sections, 1);
        break;
      case co::InputKind::kCommitRows: {
        // A verify's save, which the caller keeps: bound at its address,
        // never staged.
        ggml_tensor* t = ggml_new_tensor(c_, *type, rank, ne.data());
        external_.emplace_back(t, &input);
        out_.leaves.push_back({.tensor = t, .primitive = prim});
        return t;
      }
      case co::InputKind::kVisibleCells: {
        if (input.segment >= in_.token->shape.segments.size() || d.segment != input.segment ||
            d.format.encoding != Encoding::kI32 ||
            d.shape().ne != ir::ShapeOf({in_.token->shape.segments[input.segment].rows}).ne)
          return Refused("visible counts disagree with their owner rows");
        const auto values = co::InputValues(input, in_.token->shape, {});
        if (!values) return Refused(values.error());
        slot.kind = InputKind::kVisibleCells;
        slot.cells = {
            .capacity = input.cells.capacity, .block = input.cells.block, .positions = true};
        slot.read = input.read;
        slot.visible_capacity = input.visible_capacity;
        break;
      }
      case co::InputKind::kPositionPhases:
      case co::InputKind::kKeptRows:
      case co::InputKind::kKeptCells:
      case co::InputKind::kPoolReads:
      case co::InputKind::kBlockCells:
      case co::InputKind::kBlockStarts: {
        // These are computed inputs, never caller-supplied byte buffers.
        // Validate both their geometry and their complete value count before
        // exposing a slot the runner will stage.
        const bool phases = input.kind == co::InputKind::kPositionPhases;
        const bool kept =
            input.kind == co::InputKind::kKeptRows || input.kind == co::InputKind::kKeptCells;
        const bool pool = input.kind == co::InputKind::kPoolReads;
        const bool block_cells = input.kind == co::InputKind::kBlockCells;
        const auto& cells = input.cells;
        constexpr auto max = static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
        if (input.read != 0 || input.sections != 1 ||
            (phases ? input.segment != ir::kJoinedSegment
                    : input.segment >= in_.token->shape.segments.size()) ||
            cells.capacity > max || input.ring != cells.capacity || cells.frozen || cells.fixed ||
            cells.per_row || cells.first_row != 0 ||
            (kept ? input.ratio != 0 || input.overlap : input.ratio == 0 || input.ratio > max) ||
            (kept || pool
                 ? !cells.ring || !cells.recent || cells.positions || cells.block != 0 ||
                       cells.capacity == 0
                 : cells.ring || cells.recent ||
                       (block_cells
                            ? !cells.positions || cells.block != input.ratio || cells.capacity == 0
                            : cells.positions || cells.block != 0 || cells.capacity != 0)) ||
            (phases && input.overlap)) {
          return Refused("a compressor input has invalid typed metadata");
        }
        std::uint64_t count = in_.token->shape.rows();
        if (!phases) {
          const auto& segment = in_.token->shape.segments[input.segment];
          if (std::uint64_t{segment.past} + segment.rows > max ||
              (pool && cells.capacity + segment.rows > max))
            return Refused("a compressor input exceeds I32 positions or rows");
          count =
              kept ? std::min<std::uint64_t>(segment.rows, cells.capacity)
                   : co::CompressorBlocks(segment.rows, segment.past, input.ratio, input.overlap);
          if (pool) count *= std::uint64_t{input.ratio} * (input.overlap ? 2 : 1);
        }
        // Check the count before computing values, so malformed metadata cannot
        // request an allocation larger than the tensor the planner placed.
        if (d.shape().ne != ir::ShapeOf({count}).ne)
          return Refused("a compressor input disagrees with its typed geometry");
        const auto values = co::InputValues(input, in_.token->shape, {});
        const bool wide =
            input.kind == co::InputKind::kKeptCells || input.kind == co::InputKind::kBlockCells;
        if (!values || d.shape().ne != ir::ShapeOf({values ? values->size() : 0}).ne ||
            d.format.encoding != (wide ? Encoding::kI64 : Encoding::kI32)) {
          return Refused("a compressor input disagrees with its typed geometry");
        }
        switch (input.kind) {
          case co::InputKind::kPositionPhases:
            slot.kind = InputKind::kPositionPhases;
            break;
          case co::InputKind::kKeptRows:
            slot.kind = InputKind::kKeptRows;
            break;
          case co::InputKind::kKeptCells:
            slot.kind = InputKind::kKeptCells;
            break;
          case co::InputKind::kPoolReads:
            slot.kind = InputKind::kPoolReads;
            break;
          case co::InputKind::kBlockCells:
            slot.kind = InputKind::kBlockCells;
            break;
          case co::InputKind::kBlockStarts:
            slot.kind = InputKind::kBlockStarts;
            break;
          default:
            break;
        }
        slot.cells = {.capacity = input.cells.capacity,
                      .ring = input.cells.ring,
                      .frozen = input.cells.frozen,
                      .fixed = input.cells.fixed,
                      .block = input.cells.block,
                      .per_row = input.cells.per_row,
                      .first_row = input.cells.first_row,
                      .positions = input.cells.positions,
                      .recent = input.cells.recent};
        slot.ratio = input.ratio;
        slot.overlap = input.overlap;
        slot.ring = input.ring;
        break;
      }
      case co::InputKind::kHadamard:
        if (input.read == 0 || (input.read & (input.read - 1)) != 0 ||
            d.format.encoding != Encoding::kF32 ||
            d.shape().ne != ir::ShapeOf({input.read, input.read}).ne ||
            d.role != ir::Role::kConstant) {
          return Refused("a Hadamard constant disagrees with its order");
        }
        slot.kind = InputKind::kHadamard;
        slot.read = input.read;
        break;
      case co::InputKind::kFeatures:
        // A drafter's rows' features: copied in on the device by the
        // engine, from where the step's work names them.
        slot.kind = InputKind::kFeatures;
        break;
      case co::InputKind::kOutputCells: {
        // A target's tap rows' cells: its segment's last rows' store cells.
        const auto outputs = static_cast<std::uint32_t>(d.ne(0));
        if (input.cells.fixed || input.cells.block != 0) {
          return Refused("a tap's cells of fixed or block state");
        }
        if (input.segment >= in_.token->shape.segments.size() || outputs == 0 ||
            outputs != in_.token->shape.segments[input.segment].outputs) {
          return Refused("a tap's cells of no segment's outputs");
        }
        slot.kind = InputKind::kStoreCells;
        slot.cells = {.capacity = input.cells.capacity,
                      .ring = input.cells.ring,
                      .per_row = input.cells.per_row,
                      .first_row = input.cells.first_row};
        slot.outputs = outputs;
        break;
      }
    }
    ggml_tensor* t = ggml_new_tensor(c_, *type, rank, ne.data());
    input_leaves_.emplace_back(t, slot);
    out_.leaves.push_back({.tensor = t, .primitive = prim});
    return t;
  }
  if (n.key.op == Op::kWeight && in_.addresses.weight_rows) {
    const auto at = weights_.find(prim.v);
    const auto rows = in_.addresses.weight_rows(at->second->ref);
    if (rows != 0) {
      if (rank != 2 || rows > static_cast<std::uint64_t>(ne[1])) {
        return Refused("a row-paged weight is not a bounded table");
      }
      ne[1] = static_cast<std::int64_t>(rows);
    }
  }
  ggml_tensor* t = ggml_new_tensor(c_, *type, rank, ne.data());
  if (n.key.op == Op::kWeight) {
    const auto at = weights_.find(prim.v);
    // As the builders bind a block-quantized weight whose rows are not
    // whole padded steps: the artifact's readable range covers the read
    // past each row (artifact/representation.h kGgmlRowPadding).
    if (blocks && format::Info(d.format.encoding).block_elements > 1 &&
        d.ne(0) % format::kGgmlRowPadding != 0) {
      MarkRowPaddingReadable(t);
    }
    if (at->second->ref.expert_array) {
      // Its experts at the runner's pitch (a slab's), as the builders
      // stride a prepared expert array.
      const std::uint64_t stride =
          options_.expert_stride ? options_.expert_stride(at->second->ref) : 0;
      if (stride != 0) {
        if (stride < t->nb[2]) return Refused("an expert array's pitch below an expert's bytes");
        t->nb[2] = stride;
        t->nb[3] = stride * static_cast<std::uint64_t>(t->ne[2]);
      }
    }
    weight_leaves_.emplace_back(t, at->second);
    weight_tensors_.insert(t);
  } else if (n.key.op == Op::kState) {
    const auto at = state_.find(prim.v);
    if (at == state_.end()) return Refused("a state tensor the step does not list");
    state_leaves_.emplace_back(t, at->second);
    state_tensors_.insert(t);
  } else {
    return Refused(std::format("a leaf of role {}", ir::RoleName(d.role)));
  }
  out_.leaves.push_back({.tensor = t, .primitive = prim});
  return t;
}

std::expected<ggml_tensor*, KernelFailure> Lowering::MakeParts(TensorId out,
                                                               const ir::TensorDesc& d,
                                                               const co::StepWeight& w) {
  const auto bytes = [&](std::int64_t ne0, std::int64_t ne1, std::int64_t ne2) {
    std::array<std::int64_t, 4> ne{ne0, ne1, ne2, 1};
    return ggml_new_tensor(c_, GGML_TYPE_I8, ne2 > 1 ? 3 : 2, ne.data());
  };
  const auto k = static_cast<std::int64_t>(d.ne(0));
  const std::uint64_t slots = in_.addresses.weight_rows ? in_.addresses.weight_rows(w.ref) : 0;
  if (slots > d.ne(1)) return Refused("a row-paged weight exceeds its table");
  const auto rows = static_cast<std::int64_t>(slots != 0 ? slots : d.ne(1));
  const auto matrices = static_cast<std::int64_t>(d.ne(2));
  auto& parts = parts_[out.v];
  ggml_tensor* first = nullptr;
  const auto part = [&](format::PartRole role, ggml_tensor* t) {
    parts[role] = t;
    part_leaves_.emplace_back(t, &w, role);
    weight_tensors_.insert(t);
    out_.leaves.push_back({.tensor = t, .primitive = primitive_.at(out.v), .part = role});
    if (first == nullptr) first = t;
  };
  if (options_.selected_source != nullptr && d.format.encoding == Encoding::kExl3Trellis &&
      d.format.layout == format::Layout::kExl3Tiles && matrices == 1 && slots == 0) {
    const auto extents = format::PartsOf(d.format);
    if (!extents || extents->size() != w.parts.size())
      return Refused("invalid selected weight parts");
    for (const auto& extent : extents->parts()) {
      const auto count = std::ranges::count_if(
          w.parts, [&](const auto& entry) { return entry.second == extent.role; });
      if (count != 1 || extent.readable > static_cast<std::uint64_t>(INT64_MAX))
        return Refused("missing, duplicate or oversized selected weight part");
      part(extent.role, bytes(static_cast<std::int64_t>(extent.readable), 1, 1));
    }
  } else if (options_.selected_source != nullptr &&
             (d.format.encoding == Encoding::kNvfp4 || d.format.encoding == Encoding::kAffine) &&
             d.format.layout == format::Layout::kSplit && w.ref.expert_array) {
    if (!in_.addresses.expert_part) return Refused("split expert parts have no source binding");
    auto one = d.format;
    one.shape.rank = 2;
    one.shape.ne[2] = one.shape.ne[3] = 1;
    const auto extents = format::PartsOf(one);
    if (!extents || extents->size() != w.parts.size() || matrices <= 0)
      return Refused("split expert parts are incomplete");
    for (const auto& e : extents->parts()) {
      const auto named = std::ranges::find(w.parts, e.role, [](const auto& p) { return p.second; });
      if (named == w.parts.end() ||
          std::ranges::count_if(w.parts, [&](const auto& p) { return p.second == e.role; }) != 1)
        return Refused("split expert part is missing or duplicated");
      const auto binding = in_.addresses.expert_part(w.ref.stage, named->first);
      std::uint64_t part_bytes = 0;
      if (!binding || binding->format != one || binding->role != e.role ||
          binding->pitch < e.readable || binding->pitch > INT64_MAX ||
          __builtin_mul_overflow(static_cast<std::uint64_t>(matrices - 1), binding->pitch,
                                 &part_bytes) ||
          __builtin_add_overflow(part_bytes, e.readable, &part_bytes) ||
          part_bytes != binding->bytes || part_bytes > INT64_MAX || e.readable > INT64_MAX)
        return Refused("split expert source format, pitch or complete extent differs");
      auto* t = part_bytes == 0
                    ? nullptr
                    : ggml_new_tensor_2d(c_, GGML_TYPE_I8, static_cast<std::int64_t>(e.readable),
                                         matrices);
      if (!t) return Refused("empty split expert part");
      t->nb[1] = binding->pitch;
      t->nb[2] = t->nb[3] = part_bytes;
      part(e.role, t);
    }
  } else if (options_.selected_source != nullptr &&
             (d.format.encoding == Encoding::kNvfp4 || d.format.encoding == Encoding::kAffine) &&
             d.format.layout == format::Layout::kSplit && matrices == 1 && !w.ref.expert_array &&
             d.format.stacked == 1) {
    const auto extents = format::PartsOf(d.format);
    if (!extents) return Refused("invalid split weight format");
    for (const auto& extent : extents->parts()) {
      if (extent.role == format::PartRole::kTensorScale) {
        part(extent.role, ggml_new_tensor_1d(c_, GGML_TYPE_F32, 1));
      } else {
        const auto rb = format::RowBytes(d.format, extent.role);
        if (!rb) return Refused("invalid split weight part row bytes");
        part(extent.role, bytes(static_cast<std::int64_t>(*rb), rows, 1));
      }
    }
  } else if (d.format.encoding == Encoding::kMxfp8 && d.format.layout == format::Layout::kSplit &&
             matrices == 1) {
    // E4M3 codes [k, n] and E8M0 scales [k / 32, n], each a byte.
    part(format::PartRole::kCodes, bytes(k, rows, 1));
    part(format::PartRole::kBlockScales, bytes(k / 32, rows, 1));
  } else if (d.format.encoding == Encoding::kNvfp4 &&
             d.format.layout == format::Layout::kRowInterleaved && matrices == 1) {
    // Each row's codes then its scales, and the table's F32 scale.
    part(format::PartRole::kData, bytes((k / 2) + (k / 16), rows, 1));
    std::array<std::int64_t, 4> one{1, 1, 1, 1};
    part(format::PartRole::kTensorScale, ggml_new_tensor(c_, GGML_TYPE_F32, 1, one.data()));
  } else if (d.format.encoding == Encoding::kNvfp4 &&
             d.format.layout == format::Layout::kCutlassScaleAtoms && w.ref.expert_array) {
    // The CUTLASS layout's slab (moe_layout.h): a layer's gate-and-up and
    // down arrays at their offsets in each expert's slot, one tensor
    // [stride, experts] at the gate-and-up codes.
    const std::uint64_t stride = options_.expert_stride ? options_.expert_stride(w.ref) : 0;
    if (stride == 0) return Refused("a CUTLASS expert slab without its stride");
    const auto key = std::pair{w.ref.stage, w.ref.layer};
    auto [at, fresh] = slabs_.try_emplace(key, nullptr);
    if (fresh) {
      at->second = bytes(static_cast<std::int64_t>(stride), matrices, 1);
      part_leaves_.emplace_back(at->second, &w, format::PartRole::kCodes);
      weight_tensors_.insert(at->second);
      out_.leaves.push_back({.tensor = at->second,
                             .primitive = primitive_.at(out.v),
                             .part = format::PartRole::kCodes});
    }
    parts[format::PartRole::kCodes] = at->second;
    first = at->second;
  } else {
    return Refused(std::format("a weight of {} in {} parts", format::Info(d.format.encoding).name,
                               w.parts.size()));
  }
  return first;
}

std::expected<ggml_tensor*, KernelFailure> Lowering::Mask(const ir::Graph& graph, const ir::Node& n,
                                                          const Resolve& resolve) {
  const auto p = n.Params<ir::MaskParams>();
  if (p.kind != ir::MaskKind::kCausal && p.kind != ir::MaskKind::kWindow &&
      p.kind != ir::MaskKind::kAhead)
    return Refused("a block mask");
  const ir::TensorDesc& out = graph.tensor(n.out[0]);
  const ir::TensorDesc& rows = graph.tensor(n.src[0]);
  // The cells' positions input names the segment and its cells.
  const TensorId cells_tensor = &graph == &p_ ? primitive_.at(n.src[1].v) : n.src[1];
  const auto held = inputs_.find(cells_tensor.v);
  if (held == inputs_.end() || held->second->kind != co::InputKind::kCellPositions) {
    return Refused("a mask over no cells' positions input");
  }
  const co::StepInput& cells = *held->second;
  // A frozen read's cells hold the positions before the segment's (a
  // drafter over its target's borrowed state): the producer's frozen
  // policy, or the host mask's slot says so.
  const bool frozen = cells.cells.frozen;
  const bool ahead = p.kind == ir::MaskKind::kAhead;
  const std::uint32_t window = p.kind == ir::MaskKind::kWindow || ahead ? p.window : 0U;
  if (!options_.device_masks || cells.cells.block != 0) {
    if (out.format.encoding != Encoding::kF16) return Refused("a host mask not in F16");
    if (frozen && !options_.frozen_host_masks) {
      return Refused("a frozen read's host mask without a stager that builds it");
    }
    std::array<std::int64_t, 4> ne{static_cast<std::int64_t>(out.ne(0)),
                                   static_cast<std::int64_t>(out.ne(1)), 1, 1};
    ggml_tensor* t = ggml_new_tensor(c_, GGML_TYPE_F16, 2, ne.data());
    const InputSlot slot{.tensor = &graph == &p_ ? primitive_.at(n.out[0].v) : n.out[0],
                         .segment = cells.segment,
                         .kind = InputKind::kMask,
                         .cells = {.capacity = cells.cells.capacity,
                                   .ring = cells.cells.ring,
                                   .frozen = frozen,
                                   .block = cells.cells.block,
                                   .positions = cells.cells.positions,
                                   .recent = cells.cells.recent},
                         .read = out.ne(0),
                         .window = window,
                         .mask_rows = static_cast<std::uint32_t>(out.ne(1)),
                         .ahead = ahead};
    input_leaves_.emplace_back(t, slot);
    out_.leaves.push_back({.tensor = t, .primitive = slot.tensor});
    return t;
  }
  // The producer reads the whole positions input from the segment's first
  // row (the IR's mask reads a view of it).
  if (!rows.is_view() || rows.ne(1) != 1 || rows.view_offset % sizeof(std::int32_t) != 0) {
    return Refused("a mask's rows are not a view of the positions");
  }
  const auto to_int = [](std::uint64_t v) { return static_cast<std::int32_t>(v); };
  if (p.exact_rows && &graph == &p_) {
    // The proved position root is either the original plain input or a
    // replicated input's equivalent first section. The cells' capacity
    // bounds this exact mask's context.
    const auto proved = exact_mask_positions_.find(n.out[0].v);
    if (proved == exact_mask_positions_.end()) return Refused("an unproved exact device mask");
    const TensorId source = proved->second;
    if (made_.at(source.v) == nullptr) {
      if (auto made = MakeNode(p_.producer(source)); !made) {
        return std::unexpected(made.error());
      }
    }
    ggml_tensor* positions = made_.at(source.v);
    return CausalRingMask(c_, positions, static_cast<std::int64_t>(out.ne(0)),
                          to_int(rows.view_offset / sizeof(std::int32_t)), to_int(rows.ne(0)),
                          to_int(cells.cells.capacity), to_int(window),
                          to_int(cells.cells.capacity), CausalMaskRows::kExact, GGML_TYPE_F16,
                          frozen ? MaskPolicy::kFrozen : MaskPolicy::kCausal);
  }
  ggml_tensor* positions = resolve(rows.view_of);
  if (positions == nullptr) return Refused("a mask's positions were not made");
  if (frozen || ahead) {
    if (frozen && ahead) return Refused("an ahead mask over frozen state");
    ggml_tensor* mask = CausalRingMask(
        c_, positions, static_cast<std::int64_t>(out.ne(0)),
        to_int(rows.view_offset / sizeof(std::int32_t)), to_int(rows.ne(0)),
        to_int(cells.cells.capacity), to_int(window), to_int(in_.token->context),
        CausalMaskRows::kPad32, GGML_TYPE_F16, ahead ? MaskPolicy::kBlock : MaskPolicy::kFrozen);
    if (ahead && !Gemma4MaskFits(mask))
      return Refused("an ahead mask exceeds the device block mask's geometry");
    return mask;
  }
  return Gemma4Mask(c_, positions, static_cast<std::int64_t>(out.ne(0)),
                    to_int(rows.view_offset / sizeof(std::int32_t)), to_int(rows.ne(0)),
                    to_int(cells.cells.capacity), to_int(window), to_int(in_.token->context));
}

std::expected<ggml_tensor*, KernelFailure> Lowering::Make(const ir::Graph& graph, const ir::Node& n,
                                                          const Resolve& resolve) {
  const ir::TensorDesc& out = graph.tensor(n.out[0]);
  std::array<ggml_tensor*, ir::kMaxSources> s{};
  for (std::size_t i = 0; i < n.nsrc; ++i) {
    if (!n.src.at(i).valid()) continue;
    if (n.key == ir::Primitive(Op::kMaskBuild)) break;  // its own sources (Mask)
    s.at(i) = resolve(n.src.at(i));
    if (s.at(i) == nullptr) {
      return Refused(std::format("{}'s source {} was not made", ir::OpKeyName(n.key), i));
    }
  }
  const auto ne = [&](std::size_t d) { return static_cast<std::int64_t>(out.ne(d)); };
  switch (n.key.op) {
    case Op::kView: {
      // The IR's offset is from the root; GGML's from the viewed tensor.
      const ir::TensorDesc& a = graph.tensor(n.src[0]);
      const std::uint64_t base = a.is_view() ? a.view_offset : 0;
      if (out.view_offset < base) return Refused("a view before its source's bytes");
      const std::uint64_t scale =
          s[0]->type == GGML_TYPE_F16 && out.format.encoding == Encoding::kF32 ? 2 : 1;
      return ggml_view_4d(c_, s[0], ne(0), ne(1), ne(2), ne(3), out.nb.at(1) / scale,
                          out.nb.at(2) / scale, out.nb.at(3) / scale,
                          (out.view_offset - base) / scale);
    }
    case Op::kReshape:
      return ggml_reshape_4d(c_, s[0], ne(0), ne(1), ne(2), ne(3));
    case Op::kPermute: {
      const auto p = n.Params<ir::PermuteParams>();
      return ggml_permute(c_, s[0], p.axes[0], p.axes[1], p.axes[2], p.axes[3]);
    }
    case Op::kCont:
      return ggml_cont_4d(c_, s[0], ne(0), ne(1), ne(2), ne(3));
    case Op::kConcat:
      return ggml_concat(c_, s[0], s[1], n.Params<ir::ConcatParams>().dim);
    case Op::kHcPost:
      if (n.nsrc != 4 || s[3] == nullptr || n.Params<ir::HcParams>().streams != 4)
        return Refused("native HC post requires four streams and a combination matrix");
      return ggml_dsv4_hc_post(c_, s[0], s[1], s[2], s[3]);
    case Op::kSsmConv:
      return ggml_ssm_conv(c_, s[0], s[1]);
    case Op::kRepeat:
      return ggml_repeat_4d(c_, s[0], ne(0), ne(1), ne(2), ne(3));
    case Op::kGetRows:
      return ggml_get_rows(c_, s[0], s[1]);
    case Op::kMatMul: {
      const auto p = n.Params<ir::ProductParams>();
      const ir::TensorDesc& x = graph.tensor(n.src[1]);
      if (x.format.encoding == Encoding::kQ8_1) {
        // A prepared input (planner prepared_inputs): the prepared product
        // reads both forms (shared_q8.h).
        const ir::Node& prep = graph.node(graph.producer(graph.root(n.src[1])));
        if (prep.key != ir::Primitive(Op::kPrepare) || p.accumulate != ir::Accumulate::kDefault ||
            p.hint != ir::ProductHint::kNone) {
          return Refused("a Q8_1 product input that is no preparation");
        }
        ggml_tensor* raw = resolve(prep.src[0]);
        if (raw == nullptr) return Refused("a prepared input's source was not made");
        return MmvqPrepared(c_, s[0], s[1], raw);
      }
      ggml_tensor* t = ggml_mul_mat(c_, s[0], s[1]);
      if (p.accumulate == ir::Accumulate::kF32) ggml_prec_set_acc(t, GGML_PREC_F32);
      if (p.hint == ir::ProductHint::kHadamard)
        ggml_mul_mat_set_hint(t, GGML_HINT_SRC0_IS_HADAMARD);
      return t;
    }
    case Op::kMatMulId: {
      const auto p = n.Params<ir::ProductParams>();
      if (p.accumulate != ir::Accumulate::kDefault || p.hint != ir::ProductHint::kNone) {
        return Refused("an expert product with an accumulation or hint");
      }
      return ggml_mul_mat_id(c_, s[0], s[1], s[2]);
    }
    case Op::kPrepare:
      if (out.format.encoding != Encoding::kQ8_1) return Refused("a preparation not into Q8_1");
      return QuantizeQ8(c_, s[0]);
    case Op::kUpsample:
      if (n.Params<ir::UpsampleParams>().factor != 2 || s[0] == nullptr ||
          s[0]->type != GGML_TYPE_F32 || !ggml_is_contiguous(s[0]))
        return Refused("spatial upsampling needs contiguous F32 nearest2 input");
      return ggml_upscale(c_, s[0], 2, GGML_SCALE_MODE_NEAREST);
    case Op::kLayerNorm:
      return ggml_norm(c_, s[0], n.Params<ir::NormParams>().eps);
    case Op::kL2Norm:
      return ggml_l2_norm(c_, s[0], n.Params<ir::NormParams>().eps);
    case Op::kHcMix: {
      const auto p = n.Params<ir::HcParams>();
      if (p.mix != ir::HcMixKind::kSinkhorn || p.streams != 4 || p.iterations < 1 ||
          p.iterations > 100)
        return Refused("native HC combination requires four-stream Sinkhorn");
      return ggml_dsv4_hc_comb(c_, s[0], s[1], s[2], p.eps,
                               static_cast<std::int32_t>(p.iterations));
    }
    case Op::kHcPre: {
      const auto p = n.Params<ir::HcParams>();
      if (p.fold != ir::HcFold::kWeights || p.streams != 4 || p.scale != 1.0F)
        return Refused("native HC pre requires four weighted streams");
      return ggml_dsv4_hc_pre(c_, s[0], s[1]);
    }
    case Op::kRmsNorm:
      return ggml_rms_norm(c_, s[0], n.Params<ir::NormParams>().eps);
    case Op::kAdd:
      return ggml_add(c_, s[0], s[1]);
    case Op::kSub:
      return ggml_sub(c_, s[0], s[1]);
    case Op::kMul:
      return ggml_mul(c_, s[0], s[1]);
    case Op::kDiv:
      return ggml_div(c_, s[0], s[1]);
    case Op::kScale: {
      const auto p = n.Params<ir::ScaleParams>();
      return p.bias == 0.0F ? ggml_scale(c_, s[0], p.scale)
                            : ggml_scale_bias(c_, s[0], p.scale, p.bias);
    }
    case Op::kClamp: {
      const auto p = n.Params<ir::ClampParams>();
      return ggml_clamp(c_, s[0], p.min, p.max);
    }
    case Op::kUnary:
      switch (n.Params<ir::UnaryParams>().fn) {
        case ir::UnaryFn::kSilu:
          return ggml_silu(c_, s[0]);
        case ir::UnaryFn::kGeluErf:
          return ggml_gelu_erf(c_, s[0]);
        case ir::UnaryFn::kGeluTanh:
          return ggml_gelu(c_, s[0]);
        case ir::UnaryFn::kSigmoid:
          return ggml_sigmoid(c_, s[0]);
        case ir::UnaryFn::kTanh:
          return ggml_tanh(c_, s[0]);
        case ir::UnaryFn::kRelu:
          return ggml_relu(c_, s[0]);
        case ir::UnaryFn::kExp:
          return ggml_exp(c_, s[0]);
        case ir::UnaryFn::kSoftplus:
          return ggml_softplus(c_, s[0]);
        case ir::UnaryFn::kSqrt:
          return ggml_sqrt(c_, s[0]);
        case ir::UnaryFn::kSqr:
          return ggml_sqr(c_, s[0]);
        case ir::UnaryFn::kNeg:
          return ggml_neg(c_, s[0]);
        case ir::UnaryFn::kAbs:
          return ggml_abs(c_, s[0]);
        case ir::UnaryFn::kSgn:
          return ggml_sgn(c_, s[0]);
      }
      return Refused("an unknown unary function");
    case Op::kFill: {
      std::array<std::int64_t, 4> dims{ne(0), ne(1), ne(2), ne(3)};
      const auto type = TypeOf(out.format.encoding);
      if (!type) return Refused("a fill of no GGML type");
      ggml_tensor* t = ggml_fill(c_, ggml_new_tensor(c_, *type, 4, dims.data()),
                                 n.Params<ir::FillParams>().value);
      // As the M3 builders' fills were: no shape leaf.
      t->src[0] = nullptr;
      return t;
    }
    case Op::kGlu: {
      const auto p = n.Params<ir::GluParams>();
      if (s[1] == nullptr || p.swapped) return Refused("a GLU of one source");
      switch (p.kind) {
        case ir::GluKind::kGeGluTanh:
          return ggml_geglu_split(c_, s[0], s[1]);
        case ir::GluKind::kSwiGlu:
          return ggml_swiglu_split(c_, s[0], s[1]);
        case ir::GluKind::kSwiGluClamp:
          return ggml_swiglu_clamp(c_, s[0], s[1], p.limit);
        case ir::GluKind::kGeGluErf:
          return ggml_glu_split(c_, s[0], s[1], GGML_GLU_OP_GEGLU_ERF);
      }
      return Refused("an unknown GLU");
    }
    case Op::kRope: {
      const auto p = n.Params<ir::RopeParams>();
      if (p.table || (p.mode != ir::RopeMode::kNeox && p.mode != ir::RopeMode::kNormal)) {
        return Refused("a rotation other than NORMAL or NEOX over positions");
      }
      const auto rope = p.inverse ? ggml_rope_ext_back : ggml_rope_ext;
      return ggml_rope_set_offset(rope(c_, s[0], s[1], s[2], static_cast<int>(p.dims),
                                       p.mode == ir::RopeMode::kNeox ? GGML_ROPE_TYPE_NEOX : 0,
                                       static_cast<int>(p.original_context), p.base, p.scale,
                                       p.ext_factor, p.attn_factor, p.beta_fast, p.beta_slow),
                                  static_cast<int>(p.offset));
    }
    case Op::kCacheStore:
      // GGML lists the rows, the indices, then the cache.
      return ggml_set_rows(c_, s[0], s[1], s[2]);
    case Op::kAttention: {
      const auto p = n.Params<ir::AttnParams>();
      if (p.n_kv_max > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
        return Refused("an attention sparse bound beyond int32");
      // The owned F32 implementation retains the original mask descriptor and
      // privately converts it. GGML's public builder accepts only F16 masks.
      const bool packed_f32 = s[3] != nullptr && s[3]->type == GGML_TYPE_F32;
      ggml_tensor* t = ggml_flash_attn_ext(c_, s[0], s[1], s[2], packed_f32 ? nullptr : s[3],
                                           p.scale, p.max_bias, p.softcap);
      if (packed_f32) t->src[3] = s[3];
      if (p.accumulate == ir::Accumulate::kF32) ggml_prec_set_acc(t, GGML_PREC_F32);
      if (s[4] != nullptr) ggml_flash_attn_ext_add_sinks(t, s[4]);
      if (p.n_kv_max != 0)
        ggml_flash_attn_ext_set_n_kv_max(t, static_cast<std::int32_t>(p.n_kv_max));
      if (p.sparse_any) SetFlashAttnSparseAny(t);
      return t;
    }
    case Op::kMaskBuild:
      return Mask(graph, n, resolve);
    case Op::kSoftMax: {
      const auto p = n.Params<ir::SoftMaxParams>();
      if (s[2] != nullptr) return Refused("a softmax with sinks");
      return ggml_soft_max_ext(c_, s[0], s[1], p.scale, p.max_bias);
    }
    case Op::kSumRows:
      return ggml_sum_rows(c_, s[0]);
    case Op::kArgmax:
      return Argmax(c_, s[0], false,
                    n.Params<ir::ArgmaxParams>().nan == ir::ArgmaxNan::kFirstColumn
                        ? ArgmaxFlavor::kHostGreedy
                        : ArgmaxFlavor::kNative);
    case Op::kArgsort:
      return ggml_argsort(
          c_, s[0],
          n.Params<ir::ArgsortParams>().descending ? GGML_SORT_ORDER_DESC : GGML_SORT_ORDER_ASC);
    case Op::kIndexScore:
      return ggml_lightning_indexer(c_, s[0], s[1], s[2], s[3]);
    case Op::kVisibleTopK:
      return VisibleTopK(c_, s[0], s[1], n.Params<ir::TopKParams>().k);
    case Op::kRoundedIndexScore:
      return RoundedIndexScore(c_, s[0], s[1], s[2], s[3]);
    case Op::kPrefixOverwrite:
      return PrefixOverwrite(c_, s[0], s[1], n.Params<ir::PrefixOverwriteParams>().fill_bits);
    case Op::kSelectedMask: {
      const auto p = n.Params<ir::SelectedMaskParams>();
      return SelectedMask(c_, s[0], s[1], p.raw_cells, p.cells);
    }
    case Op::kTopK:
      return ggml_top_k(c_, s[0], static_cast<int>(n.Params<ir::TopKParams>().k));
    default:
      break;
  }
  return Refused(std::format("no GGML lowering of {}", ir::OpKeyName(n.key)));
}

ggml_tensor* Lowering::ResolveReplaced(TensorId t) const {
  if (const auto it = replaced_out_.find(t.v); it != replaced_out_.end()) return it->second;
  const TensorId planned = t.v < plan_.tensors.size() ? plan_.tensors[t.v] : TensorId{};
  return planned.valid() ? made_.at(planned.v) : nullptr;
}

std::expected<void, KernelFailure> Lowering::MakeReplaced(std::size_t entry) {
  Replaced& e = replaced_.at(entry);
  const ir::Node& prim = g_.node(e.primitive);
  if (prim.nout != 1) return Refused("a replaced node of several outputs");
  auto t = Make(g_, prim, [&](TensorId s) { return ResolveReplaced(s); });
  if (!t) return std::unexpected(t.error());
  if (*t == nullptr) return Refused("a replaced node made nothing");
  if (auto m = Matches(g_.tensor(prim.out[0]), *t, ir::OpKeyName(prim.key)); !m) {
    return std::unexpected(m.error());
  }
  replaced_out_[prim.out[0].v] = *t;
  fused_made_[e.fused.v].emplace_back(*t, prim.out[0]);
  Node(*t, e.fused);
  e.made = true;
  return {};
}

std::expected<void, KernelFailure> Lowering::Flush(NodeId limit) {
  while (cursor_ < replaced_.size() && replaced_[cursor_].primitive < limit) {
    if (!replaced_[cursor_].made) {
      const ir::Node& prim = g_.node(replaced_[cursor_].primitive);
      const ir::Node& fused = p_.node(replaced_[cursor_].fused);
      // A primitive's input can be ready before another input of its
      // selected fusion (for example a converted norm scale). Expanding it
      // then would put the norm before that conversion, splitting the
      // selected norm/mul sequence and changing PlanGraph's operation order.
      const bool ready =
          std::ranges::all_of(fused.sources(),
                              [&](TensorId s) { return !s.valid() || made_.at(s.v) != nullptr; }) &&
          std::ranges::all_of(prim.sources(), [&](TensorId s) {
            return !s.valid() || ResolveReplaced(s) != nullptr;
          });
      // Its fused node makes it, at its place, once its sources are made.
      if (!ready) return {};
      if (auto made = MakeReplaced(cursor_); !made) return made;
    }
    ++cursor_;
  }
  return {};
}

std::expected<void, KernelFailure> Lowering::MakeFused(NodeId id, const ir::Node& n) {
  if (n.key.fused == FusedKind::kOwnerAttention) return MakeOwners(id, n);
  // The primitive nodes it replaced not made yet (Flush), in their order:
  // the GGML nodes its implementation runs over.
  for (std::size_t e = 0; e < replaced_.size(); ++e) {
    if (replaced_[e].fused == id && !replaced_[e].made) {
      if (auto made = MakeReplaced(e); !made) return made;
    }
  }
  const auto& made = fused_made_[id.v];
  if (made.empty()) return Refused(std::format("{} replaced nothing", ir::OpKeyName(n.key)));
  // Its results are the replaced nodes' that the plan keeps; the others are
  // intermediates it never writes, at its first result's place.
  // A replaced node's output the planned graph keeps must be one of the
  // fused node's: anything else would be an intermediate another node
  // reads, which the fused implementation never writes.
  ggml_tensor* result = nullptr;
  for (const auto& [t, prim] : made) {
    const TensorId planned = plan_.tensors.at(prim.v);
    if (planned.valid()) {
      if (!std::ranges::contains(n.outputs(), planned)) {
        return Refused(
            std::format("{} keeps an intermediate it never writes", ir::OpKeyName(n.key)));
      }
      made_.at(planned.v) = t;
      if (result == nullptr) result = t;
    }
  }
  for (std::size_t k = 0; k < n.nout; ++k) {
    if (made_.at(n.out.at(k).v) == nullptr) return Refused("a fused node's result was not made");
  }
  if (result == nullptr) return Refused("a fused node with no result");
  if (n.key.fused == FusedKind::kRopeStore) {
    // The IR matcher permits reshape/view chains. The existing launcher names
    // exactly ROPE, VIEW, SET_ROWS; collapse only this closed, unwritten chain.
    auto* view = result->src[0];
    if (result->op != GGML_OP_SET_ROWS || view == nullptr || view->view_src == nullptr ||
        view->view_src->op != GGML_OP_ROPE || view->view_offs != 0 || !ggml_is_contiguous(view))
      return Refused("a fused rope-store lacks its whole flattened rotation view");
    auto* rotation = view->view_src;
    for (const auto& [tensor, prim] : made) {
      if (tensor == result || tensor == rotation || tensor == view) continue;
      if (plan_.tensors.at(prim.v).valid() || tensor->view_src != rotation ||
          tensor->view_offs != 0 || (tensor->op != GGML_OP_RESHAPE && tensor->op != GGML_OP_VIEW))
        return Refused("a fused rope-store has a retained or non-view intermediate");
    }
    view->op = GGML_OP_VIEW;
    view->src[0] = rotation;
    std::erase_if(out_.nodes, [&](const ggml_tensor* tensor) {
      return tensor != result && tensor != rotation && tensor != view &&
             std::ranges::any_of(made, [&](const auto& entry) { return entry.first == tensor; });
    });
  }
  for (const auto& [t, prim] : made) {
    if (!plan_.tensors.at(prim.v).valid() && t->view_src == nullptr) {
      if (n.key.fused == FusedKind::kRopeStore) {
        if (t->op != GGML_OP_ROPE || t->src[0] != made_.at(n.src[0].v))
          return Refused("a fused rope-store has an unexpected unwritten intermediate");
        // Both rope-store launchers read src[0] and write the cache directly.
        // The eliminated F32 output has no activation placement, and must not
        // masquerade as an F32 allocation over the retained F16/quantized cache.
        elided_.emplace_back(t, t->src[0], true);
      } else {
        elided_.emplace_back(t, result);
      }
    }
  }
  return {};
}

// fused.owner_attention as GGML's owner-root node (fattn_owner.h): its
// queries, mask and each owner's K and V views as the node's sources, the
// cohort's partition (an offset for a partial cohort's groups only, as
// the builders give it), the softcap the kernel takes as an integer, and
// bounded roots where a cache is narrower than the mask.
std::expected<void, KernelFailure> Lowering::MakeOwners(NodeId id, const ir::Node& n) {
  const auto p = n.Params<ir::OwnerAttnParams>();
  const std::uint32_t owners = p.owners;
  const std::uint32_t cohort = p.cohort == 0 ? owners : p.cohort;
  if (owners < 1 || owners > 4 || n.nsrc != 2 + (2 * owners) || n.nout != 1) {
    return Refused("an owner attention of other than one to four owners");
  }
  if (p.attn.scale != 1.0F || p.attn.max_bias != 0.0F || p.attn.sinks || p.attn.sparse_any ||
      p.attn.n_kv_max != 0 || (p.attn.softcap != 0.0F && p.attn.softcap != 50.0F)) {
    return Refused("an owner attention the owner-root kernel does not take");
  }
  const auto source = [&](std::size_t i) { return made_.at(n.src.at(i).v); };
  ggml_tensor* q = source(0);
  ggml_tensor* mask = source(1);
  std::array<ggml_tensor*, 4> k{}, v{};
  bool bounded = false;
  for (std::uint32_t o = 0; o < owners; ++o) {
    k.at(o) = source(2 + o);
    v.at(o) = source(2 + owners + o);
    if (k.at(o) == nullptr || v.at(o) == nullptr) return Refused("an owner's cache was not made");
  }
  if (q == nullptr || mask == nullptr) return Refused("an owner attention's sources were not made");
  for (std::uint32_t o = 0; o < owners; ++o) bounded = bounded || k.at(o)->ne[1] != mask->ne[0];
  ggml_tensor* t = FlashAttnOwnersNode(c_, q, mask, k, v, cohort, owners,
                                       detail::PartialOwnerCohort(cohort) ? p.offset : 0U,
                                       static_cast<std::uint32_t>(p.attn.softcap), bounded);
  if (auto m = Matches(p_.tensor(n.out[0]), t, "owner attention"); !m) {
    return std::unexpected(m.error());
  }
  made_.at(n.out[0].v) = t;
  Node(t, id);
  return {};
}

// kRoute as GGML's fused Gemma routing runs it: llama.cpp's build_moe_ffn
// softmax top-k, normalized (soft_max, the probabilities' rows, the sort,
// its first `used`, their rows, the sum, its clamp, the division and the
// weights' rows [1, used, T]), the ten nodes GemmaRoutingFusionAt takes.
// The kernel writes the ids (the first `used` of each row of the whole
// sort, which it backs: WrittenBytes) and the weights; the probabilities,
// the selected rows and the sums are never written and take the ids' and
// the weights' places.
std::expected<void, KernelFailure> Lowering::MakeRoute(NodeId id, const ir::Node& n) {
  const auto p = n.Params<ir::RouteParams>();
  if (p.kind == ir::RouteKind::kHash || p.kind == ir::RouteKind::kSqrtSoftplusBias) {
    const auto plane = SelectedRoutePlane(p_, n);
    if (plane == 0) return Refused("unsupported selected-ID route layout");
    auto* logits = made_.at(n.src[0].v);
    const bool biased = p.kind == ir::RouteKind::kSqrtSoftplusBias;
    auto* selected = made_.at(n.src[biased ? 1 : 2].v);
    if (!logits || !selected) return Refused("selected-ID route operands were not made");
    auto* route =
        biased ? RouteBiasedTopK(c_, logits, selected, p.used, p.normalize, p.scale)
               : RouteSelectedIds(c_, logits, selected, p.used, p.normalize, 0x1p-14F, p.scale);
    const auto rows = static_cast<std::int64_t>(p_.tensor(n.src[0]).ne(1));
    auto* ids = TypedView(c_, route, GGML_TYPE_I32, p.used, rows, 0);
    auto* weights =
        TypedView(c_, route, GGML_TYPE_F32, p.used, rows, static_cast<std::size_t>(plane));
    for (auto* node : {route, ids, weights}) Node(node, id);
    made_.at(n.out[0].v) = ids;
    made_.at(n.out[1].v) = weights;
    placed_root_[n.out[0].v] = route;
    return {};
  }
  if (p.kind != ir::RouteKind::kSoftmaxTopK || !p.normalize || p.scale != 1.0F || n.nsrc != 1 ||
      n.nout != 2) {
    return Refused("a routing other than a normalized softmax top-k");
  }
  ggml_tensor* logits = made_.at(n.src[0].v);
  if (logits == nullptr) return Refused("a routing's logits were not made");
  const auto e = static_cast<std::int64_t>(p.experts);
  const auto u = static_cast<std::int64_t>(p.used);
  const std::int64_t t = logits->ne[1];
  ggml_tensor* probs = ggml_soft_max(c_, logits);
  ggml_tensor* rows = ggml_reshape_3d(c_, probs, 1, e, t);
  ggml_tensor* order = ggml_argsort(c_, probs, GGML_SORT_ORDER_DESC);
  ggml_tensor* ids = ggml_view_4d(c_, order, u, order->ne[1], order->ne[2], order->ne[3],
                                  order->nb[1], order->nb[2], order->nb[3], 0);
  ggml_tensor* selected = ggml_get_rows(c_, rows, ids);
  ggml_tensor* flat = ggml_reshape_2d(c_, selected, u, t);
  ggml_tensor* sum = ggml_sum_rows(c_, flat);
  ggml_tensor* clamped = ggml_clamp(c_, sum, 0x1p-14F, std::numeric_limits<float>::infinity());
  ggml_tensor* weights = ggml_div(c_, flat, clamped);
  ggml_tensor* joined = ggml_reshape_3d(c_, weights, 1, u, t);
  for (ggml_tensor* node :
       {probs, rows, order, ids, selected, flat, sum, clamped, weights, joined}) {
    Node(node, id, true);
  }
  const ir::TensorDesc& want_ids = p_.tensor(n.out[0]);
  if (ids->type != GGML_TYPE_I32 || static_cast<std::uint64_t>(ids->ne[0]) != want_ids.ne(0) ||
      static_cast<std::uint64_t>(ids->ne[1]) != want_ids.ne(1)) {
    return Refused("a routing's ids are not the IR's");
  }
  if (auto m = Matches(p_.tensor(n.out[1]), weights, "a routing's weights"); !m) {
    return std::unexpected(m.error());
  }
  made_.at(n.out[0].v) = ids;
  made_.at(n.out[1].v) = weights;
  placed_root_[n.out[0].v] = order;
  elided_.emplace_back(probs, order);
  for (ggml_tensor* x : {selected, sum, clamped}) elided_.emplace_back(x, weights);
  return {};
}

// kExpertReduce (with each slot's scale gathered) as GGML's fused Gemma
// scaled reduce runs it: the experts' outputs times their scales, times
// their weights ([1, used, T], the weights' rows), each slot's view, then
// the slots added in order: the seventeen nodes GemmaReductionFusionAt
// takes. It writes only the sum; the products and partial sums take its
// place (WrittenBytes holds the products').
std::expected<void, KernelFailure> Lowering::MakeReduce(NodeId id, const ir::Node& n) {
  if (n.nsrc != 4 || n.nout != 1) return Refused("a reduce without its slots' scales");
  ggml_tensor* x = made_.at(n.src[0].v);
  ggml_tensor* w = made_.at(n.src[1].v);
  ggml_tensor* scale = made_.at(n.src[3].v);
  if (x == nullptr || w == nullptr || scale == nullptr) {
    return Refused("a reduce's sources were not made");
  }
  const std::int64_t width = x->ne[0];
  const std::int64_t u = x->ne[1];
  const std::int64_t t = x->ne[2];
  if (u < 2 || scale->ne[0] != 1 || scale->ne[1] != u || scale->ne[2] != t) {
    return Refused("a reduce of one slot, or of a scale per expert");
  }
  ggml_tensor* weights = ggml_reshape_3d(c_, w, 1, u, t);
  Node(weights, id);
  ggml_tensor* scaled = ggml_mul(c_, x, scale);
  ggml_tensor* weighted = ggml_mul(c_, scaled, weights);
  Node(scaled, id, true);
  Node(weighted, id, true);
  std::vector<ggml_tensor*> slots;
  for (std::int64_t i = 0; i < u; ++i) {
    slots.push_back(ggml_view_2d(c_, weighted, width, t, weighted->nb[2],
                                 static_cast<std::size_t>(i) * weighted->nb[1]));
    Node(slots.back(), id, true);
  }
  ggml_tensor* sum = slots.front();
  std::vector<ggml_tensor*> partial;
  for (std::int64_t i = 1; i < u; ++i) {
    sum = ggml_add(c_, sum, slots[static_cast<std::size_t>(i)]);
    Node(sum, id, true);
    partial.push_back(sum);
  }
  if (auto m = Matches(p_.tensor(n.out[0]), sum, "a reduce"); !m) return std::unexpected(m.error());
  made_.at(n.out[0].v) = sum;
  elided_.emplace_back(scaled, sum);
  elided_.emplace_back(weighted, sum);
  for (std::size_t i = 0; i + 1 < partial.size(); ++i) elided_.emplace_back(partial[i], sum);
  return {};
}

std::expected<void, KernelFailure> Lowering::MakeSelected(NodeId id, const ir::Node& n) {
  const bool paired = n.key == ir::Fused(FusedKind::kPreparedProducts) && n.nout == 2;
  if ((!paired && n.nout != 1) || n.nsrc > GGML_MAX_SRC)
    return Refused("unsupported selected-source output count");
  const auto& implementation = registry_.at(impl_of_.at(id.v));
  std::vector<SelectedOperand> inputs;
  std::vector<ggml_tensor*> dependencies;
  for (std::size_t i = 0; i < n.nsrc; ++i) {
    const auto tensor = n.src[i];
    SelectedOperand operand{
        .id = tensor, .desc = p_.tensor(tensor), .tensor = made_.at(tensor.v), .parts = {}};
    if (operand.tensor == nullptr) return Refused("a selected-source input was not made");
    if (n.key == ir::Primitive(Op::kGetRows)) {
      const auto& producer = p_.node(p_.producer(tensor));
      bool unchanged = p_.root(tensor) == tensor;
      for (const auto& node : p_.nodes()) {
        if (node.key == ir::Primitive(Op::kCacheStore) && p_.root(node.src[0]) == tensor)
          unchanged = false;
        for (std::size_t output = 0; output < node.nout; ++output)
          if (const auto alias = ir::AliasedSource(node.key, output);
              alias && node.src.at(*alias).valid() && p_.root(node.src.at(*alias)) == tensor)
            unchanged = false;
      }
      if (unchanged) {
        const auto original = primitive_.at(tensor.v);
        if (producer.key == ir::Primitive(Op::kInput)) {
          const auto at = inputs_.find(original.v);
          if (at != inputs_.end() && at->second->kind == co::InputKind::kSideRows &&
              at->second->segment == ir::kJoinedSegment)
            operand.side_rows_component = at->second->component;
        } else if (producer.key == ir::Primitive(Op::kWeight) && in_.token &&
                   in_.addresses.weight_rows) {
          const auto at = weights_.find(original.v);
          if (at != weights_.end() && at->second->ref.stage == in_.token->shape.stage &&
              in_.addresses.weight_rows(at->second->ref) != 0)
            operand.paged_weight = at->second->ref;
        }
      }
    }
    if (n.key == ir::Primitive(Op::kMatMulId) && i == 0 &&
        p_.node(p_.producer(tensor)).key == ir::Primitive(Op::kWeight) &&
        p_.root(tensor) == tensor) {
      const auto at = weights_.find(primitive_.at(tensor.v).v);
      if (at != weights_.end() && at->second->ref.expert_array)
        operand.expert_weight = at->second->ref;
    }
    const auto parts = parts_.find(tensor.v);
    if (parts != parts_.end()) {
      for (const auto& [role, part] : parts->second) {
        operand.parts.push_back({role, part});
        dependencies.push_back(part);
      }
    } else {
      dependencies.push_back(operand.tensor);
    }
    inputs.push_back(std::move(operand));
  }
  const auto& desc = p_.tensor(n.out[0]);
  const auto type = TypeOf(desc.format.encoding);
  if (!type || desc.format.layout != format::Layout::kDense || !ir::IsContiguous(desc) ||
      dependencies.size() > GGML_MAX_SRC)
    return Refused("unsupported selected-source output or too many physical dependencies");
  if (paired && !PackedProductPair(p_, n))
    return Refused("a paired selected product lacks its shared placement contract");
  std::array<std::int64_t, 4> ne{};
  for (std::size_t d = 0; d < 4; ++d) ne[d] = static_cast<std::int64_t>(desc.ne(d));
  ggml_tensor* marker =
      paired ? ggml_new_tensor_1d(c_, *type, ne[0] * ne[1] * 2)
             : ggml_new_tensor(c_, *type, std::max<int>(1, desc.shape().rank), ne.data());
  marker->op = GGML_OP_CUSTOM;
  for (std::size_t i = 0; i < dependencies.size(); ++i) marker->src[i] = dependencies[i];
  std::vector<SelectedOperand> outputs;
  for (std::size_t i = 0; i < n.nout; ++i) {
    ggml_tensor* tensor =
        paired ? ggml_view_2d(c_, marker, ne[0], ne[1], desc.nb[1], i * desc.nb[2]) : marker;
    outputs.push_back({.id = n.out[i], .desc = p_.tensor(n.out[i]), .tensor = tensor, .parts = {}});
  }
  auto call = options_.selected_source->Lower(implementation, n, inputs, outputs);
  if (!call) return std::unexpected(call.error());
  if (!*call || (*call)->owner() != options_.selected_source)
    return Refused("a selected-source operation returned another source's call");
  Node(marker, id);
  for (const auto& output : outputs) {
    made_.at(output.id.v) = output.tensor;
    if (output.tensor != marker) Node(output.tensor, id);
  }
  if (paired) placed_root_[n.out[0].v] = marker;
  out_.selected.push_back({marker, &implementation, std::move(*call)});
  return {};
}

std::expected<void, KernelFailure> Lowering::MakeNode(NodeId id) {
  const ir::Node& n = p_.node(id);
  // What nothing made reads (a fused selection's masks, a preparation
  // another kernel writes) launches nothing; made already (a leaf a mask
  // needed first), nothing more.
  if (!needed_.at(id.v)) {
    quiet_.at(id.v) = true;
    return {};
  }
  if (n.nout > 0 && made_.at(n.out[0].v) != nullptr && !folded_.contains(id.v)) return {};
  if (options_.selected_source && options_.selected_source->Owns(Selected(id)))
    return MakeSelected(id, n);
  switch (n.key.op) {
    case Op::kInput:
    case Op::kWeight:
    case Op::kState: {
      auto t = MakeLeaf(n);
      if (!t) return std::unexpected(t.error());
      made_.at(n.out[0].v) = *t;
      return {};
    }
    case Op::kFused:
      if (Direct(id)) return MakeDirect(id, n);
      return MakeFused(id, n);
    case Op::kMatMul:
    case Op::kMatMulId:
    case Op::kConvert:
    case Op::kPrepare:
    case Op::kGetRows:
    case Op::kSsmConv:
    case Op::kGatedDeltaNet: {
      bool special = false;
      if (auto made = MakeSpecial(id, n, special); !made || special) return made;
      break;
    }
    case Op::kConv2d: {
      if (n.nsrc != 3 || n.nout != 1 ||
          (Selected(id) != "ggml.conv2d.f32" && Selected(id) != kConv2dTiledF32))
        return Refused("spatial convolution needs its selected F32 implementation");
      auto* kernel = made_.at(n.src[0].v);
      auto* input = made_.at(n.src[1].v);
      auto* bias = made_.at(n.src[2].v);
      if (kernel == nullptr || input == nullptr || bias == nullptr)
        return Refused("spatial convolution operands were not made");
      const auto params = n.Params<ir::Conv2dParams>();
      if (kernel->type != GGML_TYPE_F32 || input->type != GGML_TYPE_F32 ||
          bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(kernel) ||
          !ggml_is_contiguous(input) || !ggml_is_contiguous(bias) ||
          kernel->ne[2] != input->ne[2] || ggml_nelements(bias) != kernel->ne[3] ||
          params.stride != 1 || params.dilation != 1 || params.padding > 1 ||
          (kernel->ne[0] != 1 && kernel->ne[0] != 3) || kernel->ne[1] != kernel->ne[0] ||
          input->ne[0] + 2 * params.padding < kernel->ne[0] ||
          input->ne[1] + 2 * params.padding < kernel->ne[1])
        return Refused("spatial convolution cannot build these physical operands");
      auto* conv = ggml_conv_2d_direct(c_, kernel, input, 1, 1, static_cast<int>(params.padding),
                                       static_cast<int>(params.padding), 1, 1);
      auto* broadcast = ggml_reshape_4d(c_, bias, 1, 1, kernel->ne[3], 1);
      auto* result = ggml_add(c_, conv, broadcast);
      if (auto valid = Matches(p_.tensor(n.out[0]), result, "spatial convolution"); !valid)
        return valid;
      Node(conv, id);
      Node(broadcast, id);
      Node(result, id);
      made_.at(n.out[0].v) = result;
      // The bias add reads/writes the same per-element activation. This is
      // ordinary checked Add in-place storage, not an elided convolution.
      elided_.emplace_back(conv, result);
      return {};
    }
    case Op::kRoute:
      return MakeRoute(id, n);
    case Op::kExpertReduce:
      return MakeReduce(id, n);
    case Op::kView:
      if (MaskOnly(n.out[0])) return {};  // the producer reads the positions themselves
      break;
    default:
      break;
  }
  if (n.nout != 1) return Refused(std::format("{} has several outputs", ir::OpKeyName(n.key)));
  const Resolve resolve = [&](TensorId t) { return made_.at(t.v); };
  auto t = Make(p_, n, resolve);
  if (!t) return std::unexpected(t.error());
  if (*t == nullptr) return Refused(std::format("{} made nothing", ir::OpKeyName(n.key)));
  // A prepared input is QuantizeQ8's padded bytes (WrittenBytes), which
  // its place holds (Bind); everything else the IR's tensor exactly.
  if (n.key != ir::Primitive(Op::kPrepare)) {
    if (auto m = Matches(p_.tensor(n.out[0]), *t, ir::OpKeyName(n.key)); !m) {
      return std::unexpected(m.error());
    }
  }
  made_.at(n.out[0].v) = *t;
  if ((*t)->op != GGML_OP_NONE) Node(*t, id);
  if (n.key == ir::Primitive(Op::kCacheStore)) {
    // The cache's latest store, which a kernel reading the stored cache
    // takes as its source (llmp.qsa.pool, llmp.qsa.attn).
    ggml_tensor* cache = made_.at(p_.root(n.src[0]).v);
    if (cache != nullptr) stores_[cache] = *t;
  }
  return {};
}

std::expected<void, KernelFailure> Lowering::Bind() {
  const Addresses& addresses = in_.addresses;
  const std::uint64_t base = addresses.measuring() ? kDistinct : addresses.activations;
  if (!addresses.measuring() && plan_.placement.extent > addresses.activation_bytes) {
    return Refused(std::format("the activations ({} bytes) exceed their region ({} bytes)",
                               plan_.placement.extent, addresses.activation_bytes));
  }
  for (const auto& [t, offset] : plan_.placement.offsets) {
    const auto root = placed_root_.find(t.v);
    ggml_tensor* g = root != placed_root_.end() ? root->second : made_.at(t.v);
    // What nothing made reads keeps its place unused (Needed).
    if (g == nullptr && !needed_.at(p_.producer(t).v)) continue;
    if (g == nullptr) {
      return Refused(std::format("a placed tensor the lowering did not make ({}, {})", p_.name(t),
                                 ir::OpKeyName(p_.node(p_.producer(t)).key)));
    }
    if (g->view_src != nullptr) return Refused("a placed tensor is a view");
    // What the placement made room for: the tensor's own bytes, or what the
    // module writes (WrittenBytes), which the plan was made with.
    const auto own = planner::RootBytes(p_.tensor(t));
    if (!own || ggml_nbytes(g) > std::max(*own, WrittenBytes(p_, t))) {
      return Refused(std::format(
          "{} ({}, {}) writes {} bytes, more than its place holds ({}: {} bytes, {} written)",
          ggml_op_desc(g), g->op == GGML_OP_CUSTOM ? static_cast<int>(LlmpOpOf(g)) : -1,
          ggml_type_name(g->type), ggml_nbytes(g), ir::OpKeyName(p_.node(p_.producer(t)).key),
          own ? *own : 0, WrittenBytes(p_, t)));
    }
    TensorArena::Bind(g, base + offset);
  }
  std::uint64_t leaf = kLeaves;
  const auto placeholder = [&](ggml_tensor* t) {
    const std::uint64_t at = leaf;
    leaf += Round(ggml_nbytes(t), 256) + 256;
    return at;
  };
  // Span-bearing owners use their actual layout offsets even in the
  // measurement pass. Independent per-leaf placeholders cannot prove a bank.
  using Owner = std::pair<std::uint16_t, std::uint8_t>;
  std::map<Owner, std::pair<std::uint64_t, std::uint64_t>> state_regions;
  for (const auto& bank : state_banks_) {
    const Owner owner{bank.refs[0].stage, bank.refs[0].segment};
    const auto found = state_regions.find(owner);
    if (found != state_regions.end()) {
      if (found->second.second != bank.fact.region_bytes)
        return Refused("state-bank facts disagree about their owner region");
      continue;
    }
    std::uint64_t next = 0;
    if (__builtin_add_overflow(leaf, bank.fact.region_bytes, &next) ||
        __builtin_add_overflow(next, std::uint64_t{511}, &next) || next >= kDistinct)
      return Refused("a state-bank measurement region overflows its placeholder range");
    state_regions.emplace(owner, std::pair{leaf, bank.fact.region_bytes});
    leaf = next / 256 * 256;
  }
  // A measurement binds placeholders where it has no address; a program
  // that runs is refused one (a weight or state the caller does not bind).
  for (const auto& [t, w] : weight_leaves_) {
    const std::uint64_t address =
        addresses.weight ? addresses.weight(w->ref, w->parts.front().second) : 0;
    if (address == 0 && !addresses.measuring()) return Refused("a weight has no address");
    TensorArena::Bind(t, address != 0 ? address : placeholder(t));
  }
  for (const auto& [t, s] : state_leaves_) {
    const std::uint64_t address = addresses.state ? addresses.state(s->ref) : 0;
    if (address == 0 && !addresses.measuring()) return Refused("a state tensor has no address");
    std::uint64_t bound = address;
    if (bound == 0) {
      const auto region = state_regions.find(Owner{s->ref.stage, s->ref.segment});
      if (region != state_regions.end() && !s->borrowed && !s->tap) {
        const std::array query{
            planner::StateCellSlice{.ref = s->ref, .cells = static_cast<std::uint64_t>(t->ne[1])}};
        const auto fact = in_.token->state_spans(query);
        if (!fact || fact->region_bytes != region->second.second ||
            fact->bank.range.bytes != ggml_nbytes(t) ||
            fact->bank.range.offset > fact->region_bytes ||
            fact->bank.range.bytes > fact->region_bytes - fact->bank.range.offset)
          return Refused("a state leaf lacks its measurement layout interval");
        bound = region->second.first + fact->bank.range.offset;
      } else {
        bound = placeholder(t);
      }
    }
    TensorArena::Bind(t, bound);
  }
  for (const auto& bank : state_banks_) {
    const auto again = in_.token->state_spans(bank.slices);
    if (!again || again->region_bytes != bank.fact.region_bytes ||
        again->bank.range.offset != bank.fact.bank.range.offset ||
        again->bank.range.bytes != bank.fact.bank.range.bytes ||
        again->bank.encoding != bank.fact.bank.encoding ||
        again->bank.cell_bytes != bank.fact.bank.cell_bytes)
      return Refused("a state-bank fact changed during binding");
    const auto raw =
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(bank.roots[0]->data));
    const auto compressed =
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(bank.roots[1]->data));
    std::uint64_t expected = 0, end = 0;
    if (raw == 0 || raw < bank.fact.bank.range.offset ||
        __builtin_add_overflow(raw, bank.raw_bytes, &expected) || expected != compressed ||
        __builtin_add_overflow(raw - bank.fact.bank.range.offset, bank.fact.region_bytes, &end))
      return Refused("bound state-bank pointers do not share the declared base and offsets");
    TensorArena::Bind(bank.tensor, raw);
  }
  for (const auto& [t, w, role] : part_leaves_) {
    std::uint64_t address = addresses.weight ? addresses.weight(w->ref, role) : 0;
    if (w->ref.expert_array && addresses.expert_part) {
      const auto named = std::ranges::find(w->parts, role, [](const auto& p) { return p.second; });
      if (named != w->parts.end()) {
        const auto binding = addresses.expert_part(w->ref.stage, named->first);
        if (binding &&
            (binding->format.encoding == Encoding::kNvfp4 ||
             binding->format.encoding == Encoding::kAffine) &&
            binding->format.layout == format::Layout::kSplit) {
          if (address != 0 && address != binding->address)
            return Refused("split source binding differs from the ordinary weight address");
          address = binding->address;
        }
      }
    }
    if (address == 0 && !addresses.measuring()) return Refused("a weight's part has no address");
    TensorArena::Bind(t, address != 0 ? address : placeholder(t));
  }
  // A verify's saves and a commit's rows of them: the caller's buffers.
  const auto save_at = [&](std::uint32_t layer, std::uint16_t component, std::string_view what,
                           std::uint8_t segment) -> std::uint64_t {
    return addresses.save ? addresses.save(layer, component, what, segment) : 0;
  };
  for (const auto& [t, s] : save_leaves_) {
    state_tensors_.insert(t);
    const std::uint64_t address = save_at(s->layer, s->component, s->what, s->segment);
    if (address == 0 && !addresses.measuring()) return Refused("a verify's save has no buffer");
    TensorArena::Bind(t, address != 0 ? address : placeholder(t));
  }
  for (const auto& [t, input] : external_) {
    state_tensors_.insert(t);
    // The layer is the saver's: a commit's input names its component, whose
    // instance's layer the input's name carries (components/step.h Saved).
    const std::uint64_t address =
        save_at(SavedLayer(*input), input->component, input->what, input->segment);
    if (address == 0 && !addresses.measuring()) return Refused("a commit's saved rows' buffer");
    TensorArena::Bind(t, address != 0 ? address : placeholder(t));
  }
  for (const auto& [t, slot] : input_leaves_) {
    if (t->data == nullptr) return Refused("an input the plan did not place");
  }
  for (const InsideRoot& in : inside_) {
    const auto root = placed_root_.find(in.root.v);
    const ggml_tensor* r = root != placed_root_.end() ? root->second : made_.at(in.root.v);
    if (r == nullptr || r->data == nullptr)
      return Refused("an intermediate's result is not placed");
    in.t->data = static_cast<char*>(r->data) + in.offset;
  }
  for (const auto& [t, result, read_only_operand] : elided_) {
    if (read_only_operand) {
      // Sources may be views, whose data is refreshed by BindViews only below.
      // Resolve their actual bound root without inventing a new allocation.
      const ggml_tensor* root = result;
      std::uint64_t offset = 0;
      while (root->view_src != nullptr) {
        if (__builtin_add_overflow(offset, static_cast<std::uint64_t>(root->view_offs), &offset))
          return Refused("a fused rope-store operand offset overflows");
        root = root->view_src;
      }
      const auto operand_base =
          static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(root->data));
      std::uint64_t address = 0;
      if (t->op != GGML_OP_ROPE || t->src[0] != result || operand_base == 0 ||
          offset > ggml_nbytes(root) || ggml_nbytes(result) > ggml_nbytes(root) - offset ||
          ggml_nbytes(t) > ggml_nbytes(result) ||
          __builtin_add_overflow(operand_base, offset, &address))
        return Refused("a fused rope-store operand has no bounded source address");
      TensorArena::Bind(t, address);
      continue;
    }
    const ggml_tensor* root = result;
    while (root->view_src != nullptr) root = root->view_src;
    if (state_tensors_.contains(root) || weight_tensors_.contains(root) || root->data == nullptr) {
      return Refused("a fused node's intermediates have no activation to stand in");
    }
    t->data = result->data;
  }
  BindViews(out_.nodes);
  if (!addresses.measuring()) {
    for (const auto& fact : plan_.root_alignments) {
      if (!fact.root.valid() || fact.root.v >= made_.size() || fact.alignment == 0 ||
          (fact.alignment & (fact.alignment - 1)) != 0)
        return Refused("a planned root alignment is malformed");
      const auto* tensor = made_.at(fact.root.v);
      if (tensor == nullptr && !needed_.at(p_.producer(fact.root).v)) continue;
      if (tensor == nullptr || tensor->data == nullptr ||
          reinterpret_cast<std::uintptr_t>(tensor->data) % fact.alignment != 0)
        return Refused("a bound root breaks its physical alignment promise");
    }
  }
  for (const ggml_tensor* t : out_.nodes) {
    if (t->data == nullptr) {
      return Refused(std::format("{} {} ({}) [{}, {}, {}, {}] is not bound", ggml_op_desc(t),
                                 ggml_get_name(t),
                                 t->op == GGML_OP_CUSTOM ? static_cast<int>(LlmpOpOf(t)) : -1,
                                 t->ne[0], t->ne[1], t->ne[2], t->ne[3]));
    }
  }
  return {};
}

// Scope validation happens before descriptors or plans are made. The proof
// covers the original graph; an arbitrary StepPlan is not itself a proof.
// The selected rows' basis is a computed input contract, not a dimension
// heuristic. Only a final, unprepared HC corridor with one selected row per
// owner may inherit scalar-column arithmetic when the input wave is multirow.
std::expected<void, KernelFailure> Lowering::CheckSelectedHc() {
  if (!in_.token) return {};
  if (!plan_.owner_rows_policy_enabled || plan_.owner_row_invariant ||
      in_.token->shape.segments.size() < 2)
    return {};
  struct GraphCheck {
    const ir::Graph& graph;
    std::vector<std::uint8_t> written;
    explicit GraphCheck(const ir::Graph& g) : graph(g), written(g.tensor_count(), 0) {
      for (const auto& n : g.nodes()) {
        if (n.key == ir::Primitive(Op::kCacheStore)) written[g.root(n.src[0]).v] = 1;
        for (std::size_t o = 0; o < n.nout; ++o)
          if (const auto a = ir::AliasedSource(n.key, o); a && n.src[*a].valid())
            written[g.root(n.src[*a]).v] = 1;
      }
    }
    bool Clean(TensorId t) const {
      return t.valid() && t.v < graph.tensor_count() && written[graph.root(t).v] == 0;
    }
    const ir::Node* At(TensorId t, ir::OpKey key, std::size_t sources) const {
      if (!Clean(t)) return nullptr;
      const auto& n = graph.node(graph.producer(t));
      if (n.key != key || n.nsrc != sources || n.nout != 1 || n.out[0] != t) return nullptr;
      for (std::size_t i = 0; i < sources; ++i)
        if (!Clean(n.src[i])) return nullptr;
      return &n;
    }
    const ir::Node* Prim(TensorId t, Op op, std::size_t sources) const {
      return At(t, ir::Primitive(op), sources);
    }
    bool Dense(TensorId t, Encoding encoding, format::Shape shape) const {
      return Clean(t) && graph.tensor(t).format.encoding == encoding &&
             graph.tensor(t).format.layout == format::Layout::kDense &&
             graph.tensor(t).shape().ne == shape.ne && ir::IsContiguous(graph.tensor(t));
    }
    const ir::Node* WholeReshape(TensorId t) const {
      const auto* n = Prim(t, Op::kReshape, 1);
      if (!n) return nullptr;
      const auto& x = graph.tensor(n->src[0]);
      const auto& y = graph.tensor(t);
      return x.format.encoding == y.format.encoding && ir::IsContiguous(x) && ir::IsContiguous(y) &&
                     graph.root(n->src[0]) == graph.root(t) && x.view_offset == y.view_offset
                 ? n
                 : nullptr;
    }
  };
  static_assert(sizeof(GraphCheck) <= 40);
  const GraphCheck original(g_), planned(p_);
  const auto mapped = [&](TensorId t) -> TensorId {
    return t.valid() && t.v < plan_.tensors.size() ? plan_.tensors[t.v] : TensorId{};
  };
  const auto same_leaf = [&](TensorId a, TensorId b, Op op) {
    if (mapped(a) != b || !original.Prim(a, op, 0) || !planned.Prim(b, op, 0)) return false;
    const auto& x = g_.tensor(a);
    const auto& y = p_.tensor(b);
    return !x.is_view() && !y.is_view() && x.format == y.format && x.nb == y.nb &&
           x.binding == y.binding && x.segment == y.segment;
  };
  const auto weight = [&](TensorId a, TensorId b) {
    if (!same_leaf(a, b, Op::kWeight)) return false;
    const auto* ref = std::get_if<ir::WeightRef>(&g_.tensor(a).binding);
    if (!ref) return false;
    const co::StepWeight* found = nullptr;
    for (const auto& w : in_.io.weights) {
      if (w.tensor != a) continue;
      if (found || w.ref != *ref || w.parts.size() != 1 ||
          w.parts[0].second != format::PartRole::kData)
        return false;
      found = &w;
    }
    return found != nullptr;
  };
  // At most one selected final corridor. This bound avoids unbounded walks
  // through overlapping fold chains in caller-rebuilt valid IR.
  const ir::Node* fold = nullptr;
  const ir::Node* prep = nullptr;
  const ir::Node* selected = nullptr;
  const ir::Node* gather = nullptr;
  for (const auto& n : p_.nodes()) {
    if (n.key != ir::Fused(FusedKind::kHcFold) || n.nsrc != 2 || n.nout != 1) continue;
    const auto& candidate_prep = p_.node(p_.producer(n.src[0]));
    if (candidate_prep.key != ir::Fused(FusedKind::kHcPrep) || candidate_prep.nsrc != 5 ||
        candidate_prep.nout != 1)
      continue;
    const auto& reshape = p_.node(p_.producer(candidate_prep.src[0]));
    if (reshape.key != ir::Primitive(Op::kReshape) || reshape.nsrc != 1) continue;
    const auto& rows = p_.node(p_.producer(reshape.src[0]));
    if (rows.key != ir::Primitive(Op::kGetRows) || rows.nsrc != 2) continue;
    if (fold) return Refused("selected HC: multiple selected corridors");
    fold = &n;
    prep = &candidate_prep;
    selected = &reshape;
    gather = &rows;
  }
  if (!fold) return {};
  const auto count = in_.token->shape.segments.size();
  std::uint64_t rows = 0, outputs = 0;
  bool one_per_owner = true;
  std::array<bool, 256> slots{};
  for (const auto& s : in_.token->shape.segments) {
    if (s.rows == 0 || s.outputs > s.rows || slots[s.slot])
      return Refused("selected HC: invalid or duplicate selected owner");
    slots[s.slot] = true;
    if (__builtin_add_overflow(rows, s.rows, &rows) ||
        __builtin_add_overflow(outputs, s.outputs, &outputs))
      return Refused("selected HC: selected row count overflow");
    one_per_owner = one_per_owner && s.outputs == 1;
  }
  const auto hp = prep->Params<ir::HcNormPostParams>();
  const auto fp = fold->Params<ir::HcFoldParams>();
  const auto& streams = p_.tensor(selected->out[0]);
  const auto width = streams.ne(0), hc = streams.ne(1);
  std::uint64_t hw = 0;
  if (__builtin_mul_overflow(width, hc, &hw)) return Refused("selected HC: stream width overflow");
  if (prep->src[2].valid() || prep->src[3].valid() || prep->src[4].valid() ||
      !planned.WholeReshape(selected->out[0]) || !planned.Prim(gather->out[0], Op::kGetRows, 2) ||
      !planned.Clean(prep->src[1]) || hc < 2 || hp.streams != hc || fp.streams != hc ||
      fp.prepared || fp.rounded ||
      fold->params != ir::ToBytes(ir::HcFoldParams{.streams = fp.streams}) ||
      !planned.Clean(fold->out[0]) ||
      !planned.Dense(selected->out[0], Encoding::kF32, ir::ShapeOf({width, hc, outputs})) ||
      !planned.Dense(gather->src[0], Encoding::kF32, ir::ShapeOf({hw, rows})) ||
      !planned.Dense(gather->out[0], Encoding::kF32, ir::ShapeOf({hw, outputs})) ||
      !planned.Dense(gather->src[1], Encoding::kI32, ir::ShapeOf({outputs})) ||
      !planned.Dense(prep->out[0], Encoding::kBF16, ir::ShapeOf({hw, outputs})))
    return Refused("selected HC: selected layout or preparation differs");
  const co::StepInput* indices = nullptr;
  for (const auto& input : in_.token->step.inputs) {
    if (input.kind != co::InputKind::kOutputRows) continue;
    if (indices || mapped(input.tensor) != gather->src[1] ||
        !same_leaf(input.tensor, gather->src[1], Op::kInput) ||
        input.segment != ir::kJoinedSegment || input.cells != co::CellKey{} || input.read != 0 ||
        input.sections != 1 || input.component != 0 || input.layer != 0 || input.ratio != 0 ||
        input.overlap || input.ring != 0 || input.visible_capacity != 0)
      return Refused("selected HC: noncanonical output-row staging");
    indices = &input;
  }
  if (!indices) return Refused("selected HC: missing typed output-row authority");
  if (std::ranges::count(in_.token->step.inputs, indices->tensor, &co::StepInput::tensor) != 1)
    return Refused("selected HC: ambiguous typed input binding");
  // Authenticate the original gather as well as the rewritten descriptors
  // before classifying a valid multi-output wave as an ordinary HC recipe.
  const ir::Node* original_gather = nullptr;
  const ir::Node* original_selected = nullptr;
  for (const auto& n : g_.nodes()) {
    if (n.key == ir::Primitive(Op::kGetRows) && n.nout == 1 && mapped(n.out[0]) == gather->out[0]) {
      if (original_gather) return Refused("selected HC: ambiguous original gather");
      original_gather = original.Prim(n.out[0], Op::kGetRows, 2);
      if (!original_gather) return Refused("selected HC: changed original gather");
    }
    if (n.key == ir::Primitive(Op::kReshape) && n.nout == 1 &&
        mapped(n.out[0]) == selected->out[0]) {
      if (original_selected) return Refused("selected HC: ambiguous original selection");
      original_selected = original.WholeReshape(n.out[0]);
      if (!original_selected) return Refused("selected HC: changed original selection");
    }
  }
  if (!original_gather || !original_selected ||
      original_selected->src[0] != original_gather->out[0] ||
      original_gather->src[1] != indices->tensor ||
      mapped(original_gather->src[0]) != gather->src[0] ||
      original_gather->params != gather->params ||
      !original.Dense(original_gather->src[0], Encoding::kF32, ir::ShapeOf({hw, rows})) ||
      !original.Dense(original_gather->out[0], Encoding::kF32, ir::ShapeOf({hw, outputs})) ||
      !original.Dense(original_selected->out[0], Encoding::kF32, ir::ShapeOf({width, hc, outputs})))
    return Refused("selected HC: original selected layout differs");
  // A valid verify wave may request several output rows per owner. It keeps
  // the existing ordinary recipe; it cannot acquire the protected pair proof.
  if (!one_per_owner) return {};
  if (count > static_cast<std::size_t>(kGemvBf16Columns))
    return Refused("selected HC: selected owners exceed vector columns");
  // Reconstruct the planned pair only through actual dependency edges.
  const auto* high = planned.Prim(fold->src[1], Op::kMatMul, 2);
  const auto* activation =
      high ? planned.At(high->src[1], ir::Fused(FusedKind::kScaledUnary), 1) : nullptr;
  const auto* low = activation ? planned.Prim(activation->src[0], Op::kMatMul, 2) : nullptr;
  if (!low || low->src[1] != prep->out[0] || low->params != ir::ToBytes(ir::ProductParams{}) ||
      high->params != low->params ||
      activation->params !=
          ir::ToBytes(ir::ScaledUnaryParams{.scale = 1.0F / static_cast<float>(hc)}))
    return Refused("selected HC: changed planned product corridor");
  const auto rank = p_.tensor(low->out[0]).ne(0);
  if (!planned.Dense(low->src[0], Encoding::kBF16, ir::ShapeOf({hw, rank})) ||
      !planned.Dense(low->out[0], Encoding::kF32, ir::ShapeOf({rank, count})) ||
      !planned.Dense(activation->out[0], Encoding::kBF16, ir::ShapeOf({rank, count})) ||
      !planned.Dense(high->src[0], Encoding::kBF16, ir::ShapeOf({rank, hw})) ||
      !planned.Dense(high->out[0], Encoding::kBF16, ir::ShapeOf({hw, count})) || hw % 8 != 0 ||
      rank % 8 != 0 || hw > static_cast<std::uint64_t>(kGemvBf16MaxK) ||
      rank > static_cast<std::uint64_t>(kGemvBf16MaxK))
    return Refused("selected HC: protected vector geometry is unsupported");
  // The rewrite elides original low/high results. Pair by the surviving
  // terminal output, never by Plan.from (all fused nodes may share an anchor).
  TensorId terminal{};
  for (std::uint32_t i = 0; i < g_.tensor_count(); ++i) {
    if (mapped(TensorId{i}) != fold->out[0]) continue;
    if (terminal.valid()) return Refused("selected HC: ambiguous original terminal");
    terminal = TensorId{i};
  }
  const auto* scale = original.Prim(terminal, Op::kScale, 1);
  if (!scale ||
      scale->params != ir::ToBytes(ir::ScaleParams{.scale = 1.0F / static_cast<float>(hc)}))
    return Refused("selected HC: original terminal differs");
  TensorId acc = scale->src[0], gated{};
  for (std::uint64_t h = hc; h-- > 0;) {
    const auto* combine = original.Prim(acc, h == 0 ? Op::kCont : Op::kAdd, h == 0 ? 1 : 2);
    const auto* view =
        combine ? original.Prim(combine->src[h == 0 ? 0 : 1], Op::kView, 1) : nullptr;
    if (!view || (gated.valid() && gated != view->src[0]))
      return Refused("selected HC: original ordered fold differs");
    gated = view->src[0];
    const auto& vd = g_.tensor(view->out[0]);
    const auto& gd = g_.tensor(gated);
    if (vd.shape().ne != ir::ShapeOf({width, count}).ne ||
        vd.view_offset != gd.view_offset + h * gd.nb[1] || vd.nb[0] != gd.nb[0] ||
        vd.nb[1] != gd.nb[2])
      return Refused("selected HC: original fold view differs");
    acc = combine->src[0];
  }
  const auto* shaped = original.WholeReshape(gated);
  const auto* mul = shaped ? original.Prim(shaped->src[0], Op::kMul, 2) : nullptr;
  const auto* sigmoid = mul ? original.Prim(mul->src[1], Op::kUnary, 1) : nullptr;
  const auto* old_high = sigmoid ? original.Prim(sigmoid->src[0], Op::kMatMul, 2) : nullptr;
  const auto* silu = old_high ? original.Prim(old_high->src[1], Op::kUnary, 1) : nullptr;
  const auto* downscale = silu ? original.Prim(silu->src[0], Op::kScale, 1) : nullptr;
  const auto* old_low = downscale ? original.Prim(downscale->src[0], Op::kMatMul, 2) : nullptr;
  const auto* normmul = old_low ? original.Prim(old_low->src[1], Op::kMul, 2) : nullptr;
  const auto* flat = normmul ? original.WholeReshape(normmul->src[0]) : nullptr;
  const auto* norm = flat ? original.Prim(flat->src[0], Op::kRmsNorm, 1) : nullptr;
  const auto* old_selected = norm ? original.WholeReshape(norm->src[0]) : nullptr;
  const auto* old_gather =
      old_selected ? original.Prim(old_selected->src[0], Op::kGetRows, 2) : nullptr;
  if (!old_gather || mapped(norm->src[0]) != selected->out[0] ||
      old_gather->src[1] != indices->tensor || mapped(old_gather->src[0]) != gather->src[0] ||
      old_gather->params != gather->params ||
      !original.Dense(norm->src[0], Encoding::kF32, ir::ShapeOf({width, hc, count})) ||
      !original.Dense(gated, Encoding::kF32, ir::ShapeOf({width, hc, count})) ||
      !original.Dense(normmul->out[0], Encoding::kF32, ir::ShapeOf({hw, count})) ||
      !original.Dense(flat->out[0], Encoding::kF32, ir::ShapeOf({hw, count})) ||
      mul->src[0] != normmul->out[0] || old_low->params != low->params ||
      old_high->params != high->params ||
      norm->params != ir::ToBytes(ir::NormParams{.eps = hp.eps}) ||
      downscale->params != scale->params ||
      silu->params != ir::ToBytes(ir::UnaryParams{.fn = ir::UnaryFn::kSilu}) ||
      sigmoid->params != ir::ToBytes(ir::UnaryParams{.fn = ir::UnaryFn::kSigmoid}) ||
      !weight(normmul->src[1], prep->src[1]) || !weight(old_low->src[0], low->src[0]) ||
      !weight(old_high->src[0], high->src[0]))
    return Refused("selected HC: original corridor or resource mapping differs");
  selected_hc_ = SelectedHc{.rows = indices->tensor,
                            .norm = normmul->src[1],
                            .down = old_low->src[0],
                            .up = old_high->src[0],
                            .low = p_.producer(low->out[0]),
                            .high = p_.producer(high->out[0])};
  return {};
}

std::expected<void, KernelFailure> Lowering::BindSelectedHc() {
  if (!selected_hc_) return {};
  const auto& proof = *selected_hc_;
  const auto mapped_tensor = [&](TensorId t) { return made_.at(plan_.tensors.at(t.v).v); };
  const auto* rows = mapped_tensor(proof.rows);
  std::size_t matched = 0;
  for (const auto& [tensor, slot] : input_leaves_) {
    if (tensor != rows) continue;
    ++matched;
    if (slot.tensor != proof.rows || slot.kind != InputKind::kOutputRows ||
        slot.segment != ir::kJoinedSegment)
      return Refused("selected HC: actual output-row input slot differs");
  }
  if (matched != 1 || rows->op != GGML_OP_NONE || rows->view_src != nullptr)
    return Refused("selected HC: actual computed input is missing");
  for (const TensorId t : {proof.norm, proof.down, proof.up}) {
    const auto* bound = mapped_tensor(t);
    std::size_t bindings = 0;
    for (const auto& [leaf, weight] : weight_leaves_) {
      if (leaf != bound) continue;
      ++bindings;
      if (weight->tensor != t || weight->ref != std::get<ir::WeightRef>(g_.tensor(t).binding))
        return Refused("selected HC: actual original weight binding differs");
    }
    if (bindings != 1 || bound->op != GGML_OP_NONE || bound->view_src != nullptr)
      return Refused("selected HC: original weight is not a bound leaf");
    if (auto matches = Matches(g_.tensor(t), bound, "selected HC original weight"); !matches)
      return matches;
  }
  for (const NodeId id : {proof.low, proof.high}) {
    const auto& n = p_.node(id);
    const auto* bound = made_.at(n.out[0].v);
    if (bound == nullptr || bound->src[0] != made_.at(n.src[0].v) ||
        bound->src[1] != made_.at(n.src[1].v) || !GemvBf16Wide(bound) || !UsesGemvBf16Vector(bound))
      return Refused("selected HC: bound protected product lost its vector recipe");
    if (auto checked = CheckGemmBf16(bound); !checked) return checked;
    for (const TensorId t : {n.src[0], n.src[1], n.out[0]})
      if (auto match = Matches(p_.tensor(t), made_.at(t.v), "selected HC operand"); !match)
        return match;
  }
  return {};
}

std::expected<void, KernelFailure> Lowering::CheckSelectedHcRecipe(const GraphPlan& graph) const {
  if (!selected_hc_) return {};
  for (const NodeId id : {selected_hc_->low, selected_hc_->high}) {
    const auto* node = made_.at(p_.node(id).out[0].v);
    std::size_t seen = 0;
    for (const auto& step : graph.steps) {
      if (!std::ranges::contains(step.nodes, node)) continue;
      if (++seen != 1 || step.nodes.size() != 1 || step.implementation != kGemmBf16Name ||
          step.operation != execution::Operation::kMatMul || !GemvBf16Wide(node) ||
          !UsesGemvBf16Vector(node))
        return Refused("selected HC: actual protected recipe differs");
    }
    if (seen != 1) return Refused("selected HC: actual plan omits a protected product");
  }
  return {};
}

std::expected<void, KernelFailure> Lowering::CheckOwnerProducts() {
  if (!in_.token) {
    if (!plan_.owner_products.empty() || plan_.owner_digest != base::Sha256Digest{})
      return Refused("owner-product proof requires token metadata");
    return {};
  }
  const auto& proof = in_.token->step.owner_products;
  const auto products = proof.products();
  if (products.empty()) {
    if (!plan_.owner_products.empty() || plan_.owner_digest != base::Sha256Digest{})
      return Refused("owner products: scopes without a proof");
    return {};
  }
  if (plan_.owner_digest != proof.digest() || proof.graph_digest() != co::OwnerGraphIdentity(g_) ||
      plan_.tensors.size() != g_.tensor_count() || in_.token->shape.segments.empty() ||
      in_.token->shape.segments.size() > co::kMaxSegments)
    return Refused("owner products: proof, shape or original graph differs");
  const auto mapped = [&](TensorId t) {
    return t.valid() && t.v < plan_.tensors.size() ? plan_.tensors[t.v] : TensorId{};
  };
  const auto valid = [&](TensorId t) { return t.valid() && t.v < p_.tensor_count(); };
  std::vector<co::SegmentRows> segments;
  std::uint64_t first = 0;
  for (const auto& s : in_.token->shape.segments) {
    if (first > std::numeric_limits<std::uint32_t>::max())
      return Refused("owner products: row partition overflow");
    segments.push_back({.slot = s.slot,
                        .first_row = static_cast<std::uint32_t>(first),
                        .rows = s.rows,
                        .outputs = s.outputs});
    first += s.rows;
  }
  std::vector<TensorId> requests;
  for (const auto& p : products) requests.push_back(p.output);
  auto repeated = co::ProveOwnerProducts(g_, in_.token->step.inputs, segments, requests);
  if (!repeated || repeated->digest() != proof.digest())
    return Refused("owner products: typed inputs or owner partition differ");

  using OwnerKey = std::pair<std::uint32_t, std::uint8_t>;
  std::map<OwnerKey, const planner::OwnerProductScope*> scopes;
  std::set<TensorId> results;
  for (const auto& scope : plan_.owner_products) {
    if (!valid(scope.result) || !results.insert(scope.result).second ||
        !scopes.emplace(OwnerKey{scope.primitive.v, scope.segment}, &scope).second)
      return Refused("owner products: duplicate or invalid scope");
  }
  std::vector<bool> dependency(g_.tensor_count(), false);
  std::vector<const co::StepWeight*> weights(g_.tensor_count());
  std::vector<bool> duplicate_weights(g_.tensor_count(), false);
  for (const auto& w : in_.io.weights) {
    if (!w.tensor.valid() || w.tensor.v >= weights.size()) continue;
    if (weights[w.tensor.v] != nullptr) duplicate_weights[w.tensor.v] = true;
    weights[w.tensor.v] = &w;
  }
  std::size_t covered = 0;
  for (const auto& product : products) {
    dependency[product.input.v] = true;
    dependency[product.weight.v] = true;
    if (!valid(mapped(product.input)) || !valid(mapped(product.weight)) ||
        !valid(mapped(product.output)))
      return Refused("owner products: an original operand is unmapped");
    const auto& original = g_.node(product.node);
    const auto pp = original.Params<ir::ProductParams>();
    if (pp.reserved != std::array<std::uint8_t, 2>{})
      return Refused("owner products: unsupported product parameters");
    std::vector<TensorId> ordered;
    for (const auto& owner : product.owners) {
      const auto at = scopes.find({product.node.v, owner.segment});
      if (at == scopes.end()) return Refused("owner products: missing owner scope");
      const auto& scope = *at->second;
      if (scope.slot != owner.slot || scope.columns != owner.columns || owner.columns > 16 ||
          owner.columns == 0)
        return Refused("owner products: scope differs from the proved owner");
      const auto& n = p_.node(p_.producer(scope.result));
      if (n.key != ir::Primitive(Op::kMatMul) || n.nout != 1 || n.nsrc != 2 ||
          n.params != original.params || n.src[0] != mapped(product.weight))
        return Refused("owner products: product or original weight changed");
      const auto& input = p_.tensor(n.src[1]);
      const auto& original_input = g_.tensor(product.input);
      auto shape = original_input.shape();
      shape.ne[1] = owner.columns;
      if (input.format.encoding != Encoding::kF32 || input.shape().ne != shape.ne ||
          input.nb != original_input.nb)
        return Refused("owner products: owner operand shape, encoding or strides changed");
      if (product.owners.size() == 1) {
        if (n.src[1] != mapped(product.input))
          return Refused("owner products: singleton input changed");
      } else {
        const auto& view = p_.node(p_.producer(n.src[1]));
        const auto& source = p_.tensor(mapped(product.input));
        if (view.key != ir::Primitive(Op::kView) || view.src[0] != mapped(product.input) ||
            input.view_of != p_.root(mapped(product.input)) ||
            input.view_offset != source.view_offset + owner.byte_offset)
          return Refused("owner products: owner view does not select original columns");
      }
      const auto& output = p_.tensor(scope.result);
      auto output_shape = g_.tensor(product.output).shape();
      output_shape.ne[1] = owner.columns;
      if (output.format.encoding != Encoding::kF32 || output.shape().ne != output_shape.ne ||
          output.is_view() || !ir::IsContiguous(output) ||
          output.segment !=
              (product.owners.size() == 1 ? g_.tensor(product.output).segment : owner.segment) ||
          n.batch != (product.owners.size() == 1 ? original.batch : ir::Batch::kPerOwner))
        return Refused("owner products: product output changed");
      owner_products_.push_back({.original = &product, .owner = &owner, .result = scope.result});
      ordered.push_back(scope.result);
      ++covered;
    }
    auto result = mapped(product.output);
    for (std::size_t i = ordered.size(); i > 1; --i) {
      const auto& concat = p_.node(p_.producer(result));
      if (concat.key != ir::Primitive(Op::kConcat) || concat.Params<ir::ConcatParams>().dim != 1 ||
          concat.src[1] != ordered[i - 1])
        return Refused("owner products: output owner order changed");
      results.insert(result);
      result = concat.src[0];
    }
    if (result != ordered.front()) return Refused("owner products: output owner coverage changed");
  }
  if (covered != scopes.size()) return Refused("owner products: an unrequested scope");

  // This first bounded audit accepts unchanged primitive preparation only.
  // It does not infer arithmetic equivalence for a rewritten/fused norm.
  // Mark dependencies backwards and check forwards: no recursive walk.
  for (std::size_t i = g_.nodes().size(); i > 0; --i) {
    const auto& n = g_.nodes()[i - 1];
    if (!std::ranges::any_of(n.outputs(), [&](TensorId t) { return dependency[t.v]; })) continue;
    for (auto t : n.sources())
      if (t.valid()) dependency[t.v] = true;
  }
  std::vector<bool> written(p_.tensor_count(), false);
  for (const auto& n : p_.nodes()) {
    if (n.key == ir::Primitive(Op::kView) || n.key == ir::Primitive(Op::kReshape) ||
        n.key == ir::Primitive(Op::kPermute))
      continue;
    for (auto t : n.outputs())
      if (p_.tensor(t).is_view()) written[p_.root(t).v] = true;
  }
  // The authenticated product must remain the value its consumer sees.
  // A store through any alias can otherwise alter a product (including a
  // singleton) or an intermediate/final concat without changing its edges.
  for (auto t : results)
    if (written[p_.root(t).v])
      return Refused("owner products: overwritten product or concat output");
  for (std::uint32_t i = 0; i < g_.tensor_count(); ++i) {
    if (!dependency[i]) continue;
    const TensorId t{i}, m = mapped(t);
    if (!valid(m) || written[p_.root(m).v])
      return Refused("owner products: unmapped or overwritten preparation");
    const auto& a = g_.tensor(t);
    const auto& b = p_.tensor(m);
    const auto& n = g_.node(g_.producer(t));
    const auto& next = p_.node(p_.producer(m));
    if (a.format != b.format || a.nb != b.nb || a.binding != b.binding || a.role != b.role ||
        a.segment != b.segment || a.view_offset != b.view_offset ||
        (a.is_view() ? mapped(a.view_of) != b.view_of : b.is_view()) || n.key != next.key ||
        n.params != next.params || n.nsrc != next.nsrc || n.nout != 1 || next.nout != 1)
      return Refused("owner products: original preparation or binding changed");
    for (std::size_t s = 0; s < n.nsrc; ++s)
      if (mapped(n.src[s]) != next.src[s])
        return Refused("owner products: original preparation edge changed");
    if (n.key == ir::Primitive(Op::kWeight)) {
      const auto* ref = std::get_if<ir::WeightRef>(&a.binding);
      const auto* w = weights[t.v];
      if (ref == nullptr || w == nullptr || duplicate_weights[t.v] || w->ref != *ref ||
          w->parts.size() != 1 || w->parts[0].second != format::PartRole::kData)
        return Refused("owner products: original weight resource binding changed");
    }
  }
  return {};
}

std::expected<void, KernelFailure> Lowering::BindOwnerProducts() {
  for (const auto& product : owner_products_) {
    const auto& original = *product.original;
    const auto& owner = *product.owner;
    auto* node = made_.at(product.result.v);
    auto* weight = made_.at(plan_.tensors[original.weight.v].v);
    auto* source = made_.at(plan_.tensors[original.input.v].v);
    const auto& n = p_.node(p_.producer(product.result));
    auto* input = made_.at(n.src[1].v);
    if (node == nullptr || weight == nullptr || source == nullptr || input == nullptr ||
        node->op != GGML_OP_MUL_MAT || node->src[0] != weight || node->src[1] != input ||
        weight->op != GGML_OP_NONE || weight->view_src != nullptr || node->view_src != nullptr ||
        input->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32 ||
        !std::ranges::all_of(node->op_params, [](auto p) { return p == 0; }))
      return Refused("owner products: bound operation/preparation differs");
    if (auto match = Matches(g_.tensor(original.weight), weight, "owner weight"); !match)
      return match;
    if (auto match = Matches(p_.tensor(n.src[1]), input, "owner input"); !match) return match;
    if (auto match = Matches(p_.tensor(product.result), node, "owner output"); !match) return match;
    std::uintptr_t address = 0;
    if (__builtin_add_overflow(reinterpret_cast<std::uintptr_t>(source->data),
                               static_cast<std::uintptr_t>(owner.byte_offset), &address) ||
        reinterpret_cast<std::uintptr_t>(input->data) != address ||
        (original.owners.size() > 1 && input->op != GGML_OP_VIEW))
      return Refused("owner products: bound owner address differs");
    if (auto valid = CheckMulMat(node); !valid) return valid;

    // Descriptor-only expected product, from the authenticated original
    // F32 owner operand, before any joined precision/preparation decision.
    ggml_tensor expected = *node;
    ggml_tensor expected_input = *input;
    ggml_tensor expected_weight = *weight;
    const auto& wd = g_.tensor(original.weight);
    const auto& xd = g_.tensor(original.input);
    for (std::size_t d = 0; d < 4; ++d) {
      expected_weight.ne[d] = static_cast<std::int64_t>(wd.ne(d));
      expected_weight.nb[d] = wd.nb[d];
      expected_input.ne[d] = static_cast<std::int64_t>(d == 1 ? owner.columns : xd.ne(d));
      expected_input.nb[d] = xd.nb[d];
    }
    expected_input.type = GGML_TYPE_F32;
    expected.src[0] = &expected_weight;
    expected.src[1] = &expected_input;
    std::string_view implementation;
    // Preserve explicit policy, including predicates keyed by the real
    // node identity. Only the derived owner override is absent here.
    const bool vector =
        device_.vector_floats || (device_.vector_float_node && device_.vector_float_node(node));
    if (device_.row_invariant || (vector && owner.columns <= kRowInvariantColumns)) {
      if (owner.columns > kRowInvariantColumns)
        return Refused("owner products: explicit row-invariant width unsupported");
      implementation = kMulMatVecFRows;
    } else {
      if (!device_.mul_mat) return Refused("owner products: missing actual device selector");
      const auto path = device_.mul_mat(&expected);
      if (!path) return std::unexpected(path.error());
      switch (*path) {
        case MulMatPath::kVector:
          implementation = kMulMatVector;
          break;
        case MulMatPath::kTensorCore:
          if (auto valid = CheckMulMatF(node); !valid) return valid;
          implementation = kMulMatTensorCore;
          break;
        default:
          return Refused("owner products: unresolved or unsupported concrete recipe");
      }
    }
    if (!owner_recipes_.emplace(node, implementation).second)
      return Refused("owner products: duplicate bound product");
  }
  return {};
}

std::expected<void, KernelFailure> Lowering::CheckOwnerRecipes(const GraphPlan& plan) const {
  std::map<const ggml_tensor*, std::size_t> seen;
  for (const auto& step : plan.steps) {
    for (const auto* node : step.nodes) {
      const auto found = owner_recipes_.find(node);
      if (found == owner_recipes_.end()) continue;
      if (++seen[node] != 1 || step.nodes.size() != 1 ||
          step.operation != execution::Operation::kMatMul || step.implementation != found->second)
        return Refused(
            "owner products: actual concrete recipe differs from "
            "original owner");
    }
  }
  if (seen.size() != owner_recipes_.size())
    return Refused("owner products: actual plan omits a protected product");
  return {};
}

std::expected<void, KernelFailure> Lowering::PlanAndCompare() {
  // The device's choices, with the policies the selection takes.
  DeviceChoices choices = device_;
  choices.conv2d_tiled = [this](const ggml_tensor* tensor) {
    for (const auto& step : plan_.steps) {
      if (registry_.at(step.implementation).name != kConv2dTiledF32) continue;
      if (step.nodes.size() != 1) continue;
      const auto& nodes = launched_.at(step.nodes.front().v);
      if (!nodes.empty() && nodes.front() == tensor) return true;
    }
    return false;
  };
  choices.selected = [this](const ggml_tensor* tensor) -> const execution::Implementation* {
    const auto at = std::ranges::find(out_.selected, tensor, &SelectedNode::tensor);
    return at == out_.selected.end() ? nullptr : at->implementation;
  };
  // Several owners' rows each as alone (the plan's owner_row_invariant):
  // float products of up to 8 columns on the vector kernel, as each
  // owner's one row alone takes (today's waves keep them per slot).
  if (owner_recipes_.empty()) {
    choices.vector_floats = choices.vector_floats || plan_.owner_row_invariant;
  } else {
    choices.vector_float_node = [this](const ggml_tensor* node) {
      return (device_.vector_float_node && device_.vector_float_node(node)) ||
             (plan_.owner_row_invariant && !owner_recipes_.contains(node));
    };
  }
  // An explicit selected vector product is independent of legacy owner-row
  // policy. Only its actual lowered node receives the request; neighbours,
  // expert products and wider physical inputs retain their own selection.
  const auto vector_count =
      static_cast<std::size_t>(std::ranges::count_if(plan_.steps, [&](const auto& s) {
        return registry_.at(s.implementation).name == kMulMatVecFRows;
      }));
  std::vector<const ggml_tensor*> selected_vectors;
  selected_vectors.reserve(vector_count);
  for (const auto& s : plan_.steps) {
    if (registry_.at(s.implementation).name != kMulMatVecFRows) continue;
    if (s.nodes.size() != 1) return Refused("a selected vector product has several IR nodes");
    const auto& ir_node = p_.node(s.nodes.front());
    const auto params = ir_node.Params<ir::ProductParams>();
    if (ir_node.key != ir::Primitive(ir::Op::kMatMul) ||
        params.accumulate != ir::Accumulate::kDefault || params.hint != ir::ProductHint::kNone ||
        params.reserved[0] || params.reserved[1])
      return Refused("a selected vector product left its primitive parameter domain");
    const auto& nodes = launched_.at(s.nodes.front().v);
    if (nodes.size() != 1) return Refused("a selected vector product has several physical nodes");
    const auto* node = nodes.front();
    if (auto checked = CheckMulMat(node); !checked) return checked;
    const auto* weight = node->src[0];
    const auto* input = node->src[1];
    if (weight->type != GGML_TYPE_BF16 || input->type != GGML_TYPE_F32 ||
        node->type != GGML_TYPE_F32 || input->ne[1] < 1 || input->ne[1] > 8 ||
        weight->ne[0] % 2 != 0 || node->ne[0] % 2 != 0 || weight->ne[2] != 1 ||
        weight->ne[3] != 1 || input->ne[2] != 1 || input->ne[3] != 1 ||
        !ggml_is_contiguous(weight) || !ggml_is_contiguous(input) || !ggml_is_contiguous(node))
      return Refused("a selected BF16 vector product left its packed physical domain");
    selected_vectors.push_back(node);
  }
  if (!selected_vectors.empty()) {
    auto prior = std::move(choices.vector_float_node);
    choices.vector_float_node = [&selected_vectors, prior = std::move(prior)](const auto* node) {
      return std::ranges::contains(selected_vectors, node) || (prior && prior(node));
    };
  }
  std::set<const ggml_tensor*> wide_attention;
  for (const planner::Selected& s : plan_.steps) {
    const std::string_view name = registry_.at(s.implementation).name;
    if (name == kFlashAttnMmaWideName ||
        (name == "ggml.state_bank_attention" && s.nodes.size() == 1 &&
         p_.node(s.nodes.front()).Params<ir::AttnParams>().sparse_any))
      for (const NodeId id : s.nodes)
        for (const ggml_tensor* node : launched_.at(id.v)) wide_attention.insert(node);
    choices.fuse_norms = choices.fuse_norms || name == kRmsNormMulFused;
    choices.fuse_norm_rope = choices.fuse_norm_rope || name == kGemmaNormRopeName;
    choices.fuse_norm_add = choices.fuse_norm_add || name == kGemmaNormAddName;
    choices.fuse_quant_glu =
        choices.fuse_quant_glu || name == kMulMatGeGluQFused || name == kMulMatGluQFused;
    choices.group_set_rows =
        choices.group_set_rows || name == kSetRowsGroupedName || name == kSetRowsQuantGroupName;
    choices.fuse_rope_store =
        choices.fuse_rope_store || name == kRopeSetRowsFused || name == kRopeSetRowsQuantName;
    choices.fuse_gemma_route = choices.fuse_gemma_route || name == kGemmaRouteName;
    choices.fuse_gemma_reduce = choices.fuse_gemma_reduce || name == kGemmaReduceName;
  }
  std::set<const ggml_tensor*> float_fusions;
  for (const auto& selected : plan_.steps) {
    const auto& name = registry_.at(selected.implementation).name;
    if (name != kMulMatGluFused && name != kMulMatAddFused) continue;
    for (const auto id : selected.nodes) {
      const auto& nodes = launched_.at(id.v);
      if (!nodes.empty()) float_fusions.insert(nodes.front());
    }
  }
  if (!float_fusions.empty()) {
    choices.float_product_fusion = [&float_fusions](const ggml_tensor* node) {
      return float_fusions.contains(node);
    };
  }
  if (!wide_attention.empty()) {
    const auto prior = choices.wide_sparse_attention_node;
    choices.wide_sparse_attention_node = [prior, &wide_attention](const ggml_tensor* node) {
      return wide_attention.contains(node) || (prior && prior(node));
    };
  }
  bool dsv4_fast = dsv4_fast_ || !wide_attention.empty();
  if (!dsv4_fast) {
    for (const planner::Selected& s : plan_.steps) {
      const std::string_view name = registry_.at(s.implementation).name;
      if (name == kDsv4RouteName || name == "llmp.dsv4.lid_mask" || name == kDsv4QHeadName ||
          name == kDsv4HcMixName || name == kDsv4CompressName || name == kDsv4OutAFastPackName) {
        dsv4_fast = true;
        break;
      }
    }
  }
  if (dsv4_fast) {
    choices.vector_floats = true;
    choices.d2r_experts = true;
    choices.pair_experts = true;
    choices.compact_experts = in_.token && in_.token->shape.rows() >= kDsv4CompactMinRows;
    choices.pair_glu = choices.compact_experts;
    choices.pair_glu_q8 = choices.compact_experts;
    choices.dense_pair = true;
    choices.hc_post_experts = true;
    choices.outa_fast_pack = true;
  }
  for (const TensorId t : p_.outputs()) out_.keep.push_back(made_.at(t.v));
  for (const TensorId t : p_.keep()) out_.keep.push_back(made_.at(t.v));
  if (std::ranges::contains(out_.keep, nullptr)) return Refused("an output was not made");
  auto planned =
      PlanGraph(out_.nodes, /*fusion=*/false, choices, out_.keep, options_.index_scratch_limit);
  if (!planned) return std::unexpected(planned.error());
  GraphPlan& gp = *planned;
  if (auto checked = CheckOwnerRecipes(gp); !checked) return checked;
  if (auto checked = CheckSelectedHcRecipe(gp); !checked) return checked;
  const auto describe = [this](std::span<ggml_tensor* const> nodes) {
    std::map<const ggml_tensor*, std::size_t> place;
    for (std::size_t i = 0; i < out_.nodes.size(); ++i) place.emplace(out_.nodes[i], i);
    std::string text;
    for (const ggml_tensor* t : nodes) {
      const auto at = place.find(t);
      text +=
          std::format(" #{}:{}[{},{},{},{}]",
                      at == place.end() ? std::int64_t{-1} : static_cast<std::int64_t>(at->second),
                      ggml_op_desc(t), t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    }
    return text;
  };
  std::size_t j = 0;
  std::vector<std::size_t> step_gp_first(plan_.steps.size(), 0);
  std::vector<std::size_t> step_gp_last(plan_.steps.size(), 0);
  std::vector<bool> paired_later(plan_.steps.size(), false);
  for (std::size_t i = 0; i < plan_.steps.size(); ++i) {
    if (paired_later[i]) continue;
    const std::size_t cur_first = j;
    step_gp_first[i] = cur_first;
    step_gp_last[i] = cur_first;
    const planner::Selected& s = plan_.steps[i];
    std::vector<ggml_tensor*> want;
    for (const NodeId id : s.nodes) {
      const auto& l = launched_.at(id.v);
      want.insert(want.end(), l.begin(), l.end());
    }
    if (want.empty()) {
      // A node made by another's kernel or not at all (quiet), or a
      // host-built mask, staged before the run: placed as the IR placed its
      // step, which holds only where nothing computed before it (the masks
      // come first, BuildStep).
      const bool quiet = std::ranges::all_of(s.nodes, [&](NodeId id) { return quiet_.at(id.v); });
      if (!quiet && std::ranges::any_of(std::span(gp.steps).first(j), [](const auto& prior) {
            return prior.implementation != kGemma4MaskName;
          }))
        return Refused("a host-built mask after the step's computed nodes");
      continue;
    }
    const std::string_view name = registry_.at(s.implementation).name;
    // An implementation whose node launches several of the module's steps,
    // or one named otherwise (Capability::launches): PlanGraph's steps over
    // its nodes, in order, each one of those.
    const auto& cap = registry_.at(s.implementation).capability;
    if (cap && !cap->launches.empty()) {
      std::vector<ggml_tensor*> covered;
      const std::size_t first = j;
      while (j < gp.steps.size() && covered.size() < want.size() &&
             std::ranges::contains(cap->launches, gp.steps[j].implementation) &&
             std::ranges::equal(
                 gp.steps[j].nodes,
                 std::span(want).subspan(covered.size(), std::min(want.size() - covered.size(),
                                                                  gp.steps[j].nodes.size())))) {
        covered.insert(covered.end(), gp.steps[j].nodes.begin(), gp.steps[j].nodes.end());
        gp.steps[j].lane = s.lane;
        ++j;
      }
      if (covered != want) {
        return Refused(std::format("PlanGraph's steps from {} are not {}'s launches over{}", first,
                                   name, describe(want)));
      }
      step_gp_last[i] = (j > cur_first) ? j - 1 : cur_first;
      continue;
    }
    if (j >= gp.steps.size()) {
      return Refused(
          std::format("PlanGraph has no step for the IR's {} over{}", name, describe(want)));
    }
    PlanStep& got = gp.steps[j];
    if (got.implementation == kRmsNormMulUnfused && name == "ggml.rms_norm" &&
        i + 1 < plan_.steps.size() &&
        registry_.at(plan_.steps[i + 1].implementation).name == "ggml.mul" &&
        plan_.steps[i + 1].lane == s.lane) {
      // Unfused, PlanGraph runs an RMS norm and the mul that follows it as
      // one step: rms_norm's launcher, then mul's, the IR's two steps.
      std::vector<ggml_tensor*> both = want;
      for (const NodeId id : plan_.steps[i + 1].nodes) {
        const auto& l = launched_.at(id.v);
        both.insert(both.end(), l.begin(), l.end());
      }
      if (got.nodes == both) {
        got.lane = s.lane;
        step_gp_first[i + 1] = cur_first;
        step_gp_last[i + 1] = cur_first;
        ++i;
        ++j;
        continue;
      }
    }
    if (got.implementation == kDsv4HcPostExpertsNormF16Name && i + 1 < plan_.steps.size() &&
        plan_.steps[i + 1].lane == s.lane) {
      std::vector<ggml_tensor*> combined = want;
      for (const NodeId id : plan_.steps[i + 1].nodes) {
        const auto& l = launched_.at(id.v);
        combined.insert(combined.end(), l.begin(), l.end());
      }
      if (got.nodes == combined) {
        got.lane = s.lane;
        step_gp_first[i + 1] = cur_first;
        step_gp_last[i + 1] = cur_first;
        ++i;
        ++j;
        continue;
      }
    }
    if ((got.implementation == kMulMatIdQPair || got.implementation == kMulMatIdQPairCompact) &&
        i + 1 < plan_.steps.size() && plan_.steps[i + 1].lane == s.lane) {
      std::vector<ggml_tensor*> pair_nodes = want;
      for (const NodeId id : plan_.steps[i + 1].nodes) {
        const auto& l = launched_.at(id.v);
        pair_nodes.insert(pair_nodes.end(), l.begin(), l.end());
      }
      if (got.nodes == pair_nodes) {
        got.lane = s.lane;
        step_gp_first[i + 1] = cur_first;
        step_gp_last[i + 1] = cur_first;
        ++i;
        ++j;
        continue;
      }
    }
    if ((got.implementation == kMulMatIdQPairGlu || got.implementation == kMulMatIdQPairGluQ8) &&
        i + 2 < plan_.steps.size() && plan_.steps[i + 1].lane == s.lane &&
        plan_.steps[i + 2].lane == s.lane) {
      std::vector<ggml_tensor*> glu_nodes;
      for (const NodeId id : plan_.steps[i + 1].nodes) {
        const auto& l = launched_.at(id.v);
        glu_nodes.insert(glu_nodes.end(), l.begin(), l.end());
      }
      glu_nodes.insert(glu_nodes.end(), want.begin(), want.end());
      for (const NodeId id : plan_.steps[i + 2].nodes) {
        const auto& l = launched_.at(id.v);
        glu_nodes.insert(glu_nodes.end(), l.begin(), l.end());
      }
      if (got.nodes == glu_nodes) {
        got.lane = s.lane;
        step_gp_first[i + 1] = cur_first;
        step_gp_last[i + 1] = cur_first;
        step_gp_first[i + 2] = cur_first;
        step_gp_last[i + 2] = cur_first;
        i += 2;
        ++j;
        continue;
      }
    }
    if (got.implementation == kMulMatQPairDense && got.nodes.size() == 2 && want.size() == 1 &&
        got.nodes[0] == want[0]) {
      std::size_t found_i2 = plan_.steps.size();
      for (std::size_t i2 = i + 1; i2 < plan_.steps.size(); ++i2) {
        if (paired_later[i2]) continue;
        const auto& s2 = plan_.steps[i2];
        std::vector<ggml_tensor*> want2;
        for (const NodeId id : s2.nodes) {
          const auto& l = launched_.at(id.v);
          want2.insert(want2.end(), l.begin(), l.end());
        }
        if (want2.size() == 1 && want2[0] == got.nodes[1]) {
          found_i2 = i2;
          break;
        }
      }
      if (found_i2 < plan_.steps.size()) {
        paired_later[found_i2] = true;
        step_gp_first[found_i2] = cur_first;
        step_gp_last[found_i2] = cur_first;
        got.lane = s.lane;
        ++j;
        continue;
      }
    }
    bool same = got.nodes == want && got.implementation == name;
    // The owners' row-invariant products (the plan's owner_row_invariant)
    // stand for their group's.
    same = same ||
           ((plan_.owner_row_invariant || dsv4_fast) && got.nodes == want &&
            (((got.implementation == kMulMatVecFRows || got.implementation == kMulMatVecQRows) &&
              s.group == "ggml.mul_mat") ||
             (got.implementation == kMulMatIdVecQRows && s.group == "ggml.mul_mat_id")));
    if (!same && got.nodes == want && !s.group.empty()) {
      // A member of the selected group, upstream's choice inside it.
      const auto member = registry_.Find(got.implementation);
      same = member && registry_.at(*member).capability &&
             registry_.at(*member).capability->group == s.group;
    }
    same = same || (got.nodes == want && got.implementation == kMulMatIdQCompactPrequant);
    if (!same) {
      return Refused(std::format(
          "PlanGraph's step {} is {} over{}, where the IR selected {}{} over{}", j,
          got.implementation, describe(got.nodes), name,
          s.group.empty() ? std::string{} : std::format(" (group {})", s.group), describe(want)));
    }
    got.lane = s.lane;
    ++j;
  }
  // The lowering's own steps (a verify's saves) after the IR's.
  while (j < gp.steps.size() &&
         std::ranges::all_of(
             gp.steps[j].nodes, [&](const ggml_tensor* t) { return extra_.contains(t); })) {
    ++j;
  }
  if (j != gp.steps.size()) {
    return Refused(std::format("PlanGraph has {} steps beyond the IR's: {} over{}",
                               gp.steps.size() - j, gp.steps[j].implementation,
                               describe(gp.steps[j].nodes)));
  }
  // The IR's lanes over the same steps (host masks, which launch nothing
  // here, are outside every region: the step lists only their own).
  std::vector<GraphPlan::Region> mapped_regions;
  mapped_regions.reserve(plan_.regions.size());
  for (const planner::Region& r : plan_.regions) {
    if (r.first > r.last || r.last >= plan_.steps.size()) {
      return Refused("invalid planner region bounds");
    }
    const std::size_t gp_first = step_gp_first[r.first];
    const std::size_t gp_last = step_gp_last[r.last];
    if (gp_first > gp_last || gp_last >= gp.steps.size()) {
      return Refused("mapped region bounds out of range of lowered steps");
    }
    mapped_regions.push_back({.first = static_cast<std::uint32_t>(gp_first),
                              .last = static_cast<std::uint32_t>(gp_last)});
  }
  std::ranges::sort(mapped_regions, {}, &GraphPlan::Region::first);
  for (const auto& span : mapped_regions) {
    if (!gp.regions.empty() && span.first <= gp.regions.back().last) {
      gp.regions.back().last = std::max(gp.regions.back().last, span.last);
    } else {
      gp.regions.push_back(span);
    }
  }
  out_.plan = std::move(gp);
  return {};
}

std::expected<void, KernelFailure> Lowering::Slots() {
  for (auto [t, slot] : input_leaves_) {
    slot.device_address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t->data));
    slot.bytes = ggml_nbytes(t);
    out_.inputs.push_back(slot);
  }
  if (!in_.token) {
    for (const TensorId result : in_.io.outputs) {
      const TensorId planned = plan_.tensors.at(result.v);
      const ggml_tensor* t = planned.valid() ? made_.at(planned.v) : nullptr;
      if (t == nullptr || t->data == nullptr || !ggml_is_contiguous(t))
        return Refused("a spatial result was not made as a complete contiguous port");
      const auto& desc = g_.tensor(result);
      if (desc.segment == ir::kJoinedSegment)
        return Refused("a spatial result has no individual owner");
      if (auto valid = Matches(desc, t, "spatial output"); !valid) return valid;
      out_.outputs.push_back(
          {.device_address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t->data)),
           .bytes = ggml_nbytes(t),
           .tensor = result,
           .segment = desc.segment});
    }
  } else {
    // Each owner's own outputs: its token, or its rows of the logits.
    const planner::StepShape& shape = in_.token->shape;
    const bool greedy = in_.token->step.greedy.valid();
    const TensorId result = greedy ? in_.token->step.greedy : in_.token->step.logits;
    if (result.valid() && shape.outputs() != 0) {
      const TensorId planned = plan_.tensors.at(result.v);
      const ggml_tensor* t = planned.valid() ? made_.at(planned.v) : nullptr;
      if (t == nullptr || t->data == nullptr) return Refused("the step's result was not made");
      const std::uint64_t row = greedy ? sizeof(std::int32_t) : t->nb[1];
      if (static_cast<std::uint64_t>(greedy ? t->ne[0] : t->ne[1]) != shape.outputs()) {
        return Refused("the step's result does not hold its outputs");
      }
      std::uint64_t first = 0;
      for (std::size_t i = 0; i < shape.segments.size(); ++i) {
        const std::uint32_t n = shape.segments[i].outputs;
        if (n == 0) continue;
        out_.outputs.push_back({.device_address = static_cast<std::uint64_t>(
                                                      reinterpret_cast<std::uintptr_t>(t->data)) +
                                                  (first * row),
                                .bytes = n * row,
                                .tensor = result,
                                .segment = static_cast<std::uint8_t>(i)});
        first += n;
      }
    }
    // A drafter head's next features, each owner's rows after its logits.
    if (in_.token->step.features.valid() && shape.outputs() != 0) {
      const TensorId planned = plan_.tensors.at(in_.token->step.features.v);
      const ggml_tensor* t = planned.valid() ? made_.at(planned.v) : nullptr;
      if (t == nullptr || t->data == nullptr || !ggml_is_contiguous(t) ||
          static_cast<std::uint64_t>(t->ne[1]) != shape.outputs()) {
        return Refused("the step's features were not made whole");
      }
      std::uint64_t first = 0;
      for (std::size_t i = 0; i < shape.segments.size(); ++i) {
        const std::uint32_t n = shape.segments[i].outputs;
        if (n == 0) continue;
        out_.outputs.push_back({.device_address = static_cast<std::uint64_t>(
                                                      reinterpret_cast<std::uintptr_t>(t->data)) +
                                                  (first * t->nb[1]),
                                .bytes = n * t->nb[1],
                                .tensor = in_.token->step.features,
                                .segment = static_cast<std::uint8_t>(i),
                                .features = true});
        first += n;
      }
    }
  }
  // Every tensor the steps touch, by the role of its storage.
  std::set<std::tuple<std::uint64_t, std::uint64_t, ir::Role>> seen;
  const auto use = [&](const ggml_tensor* t) {
    // A selected-only part descriptor names an entire pitched expert array,
    // but no operation reads that entire span. MakeRouting has authenticated
    // every reader and selected call before this exact leaf-identity check.
    // Actual part reservations and selected residency are checked by the
    // runner; ordinary aliases and all other tensor uses stay covered here.
    if (!out_.routing.empty() && out_.weight_reads) {
      for (const auto& [part_tensor, weight, role] : part_leaves_) {
        if (part_tensor != t) continue;
        const auto read = std::ranges::find(*out_.weight_reads, weight->ref, &RoutedWeight::ref);
        if (read != out_.weight_reads->end() && read->dynamic &&
            std::ranges::count_if(read->parts,
                                  [&](const auto& part) { return part.second == role; }) == 1)
          return;
      }
    }
    const ggml_tensor* root = t;
    while (root->view_src != nullptr) root = root->view_src;
    const ir::Role role = state_tensors_.contains(root)    ? ir::Role::kState
                          : weight_tensors_.contains(root) ? ir::Role::kWeight
                                                           : ir::Role::kActivation;
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t->data));
    if (seen.emplace(address, ggml_nbytes(t), role).second) {
      out_.tensors.push_back({.address = address, .bytes = ggml_nbytes(t), .role = role});
    }
  };
  for (const ggml_tensor* n : out_.nodes) {
    use(n);
    // A fill's source only shapes it, and the builders' fills have none.
    for (const ggml_tensor* src : n->src) {
      if (src != nullptr) use(src);
    }
  }
  return {};
}

// ------------------------------------------------- the fast forms (G3b)

// A packed `type` [ne0, ne1] view of a byte blob a llmpalooza operation
// wrote (several outputs in one tensor), from `offset`: qwen38_graph.cc's
// TypedView.
ggml_tensor* TypedView(ggml_context* c, ggml_tensor* blob, ggml_type type, std::int64_t ne0,
                       std::int64_t ne1, std::size_t offset) {
  const std::size_t row = ggml_row_size(type, ne0);
  ggml_tensor* v = ggml_view_1d(
      c, blob,
      static_cast<std::int64_t>(row * static_cast<std::size_t>(ne1) / ggml_type_size(blob->type)),
      offset);
  v->type = type;
  v->ne[0] = ne0;
  v->ne[1] = ne1;
  v->nb[0] = ggml_type_size(type);
  v->nb[1] = row;
  v->nb[2] = row * static_cast<std::size_t>(ne1);
  v->nb[3] = v->nb[2];
  return v;
}

// GGML's row norms take packed input: qwen38_graph.cc's Packed.
bool IsPacked(const ggml_tensor* x) {
  bool packed = x->nb[0] == ggml_type_size(x->type);
  for (int i = 1; i < GGML_MAX_DIMS; ++i) {
    packed = packed && x->nb[i] == x->nb[i - 1] * static_cast<std::size_t>(x->ne[i - 1]);
  }
  return packed;
}

std::string_view Lowering::Selected(NodeId id) const {
  const std::size_t impl = impl_of_.at(id.v);
  return impl == kNoImplementation ? std::string_view{} : registry_.at(impl).name;
}

bool Lowering::Direct(NodeId id) const {
  if (options_.selected_source && options_.selected_source->Owns(Selected(id))) return true;
  const ir::Node& n = p_.node(id);
  if (n.key.op != Op::kFused) return false;
  switch (n.key.fused) {
    case FusedKind::kGroupedMatMul:
    case FusedKind::kStateBankAttention:
    case FusedKind::kDsv4LidMask:
    case FusedKind::kDsv4QHead:
    case FusedKind::kDsv4Compress:
    case FusedKind::kDsv4HcPre:
    case FusedKind::kHcNormPost:
    case FusedKind::kHcPrep:
    case FusedKind::kScaledUnary:
    case FusedKind::kHcFold:
    case FusedKind::kHcCombine:
    case FusedKind::kHcMixNorm:
    case FusedKind::kHcNorm:
    case FusedKind::kGdnConv:
    case FusedKind::kGdnHistory:
    case FusedKind::kGdnStep:
    case FusedKind::kGdnGates:
    case FusedKind::kNormGate:
    case FusedKind::kHeadGate:
    case FusedKind::kQsaPool:
    case FusedKind::kQsaAttention:
    case FusedKind::kMoeRouter:
    case FusedKind::kMoeExperts:
    case FusedKind::kMoeGlu:
    case FusedKind::kMoeCombine:
    case FusedKind::kMatMulIdGlu:
    case FusedKind::kDsv4OutA:
      return true;
    case FusedKind::kRmsNormMulRope:
      return Selected(id) == kQsaPrepName;
    case FusedKind::kMatMulGlu:
      return Selected(id).starts_with("llmp.vecq");
    default:
      return false;
  }
}

ggml_tensor* Lowering::Part(TensorId planned, format::PartRole role) const {
  const auto at = parts_.find(planned.v);
  if (at == parts_.end()) return nullptr;
  const auto part = at->second.find(role);
  return part == at->second.end() ? nullptr : part->second;
}

ggml_tensor* Lowering::StoredInto(const ggml_tensor* leaf) const {
  const auto at = stores_.find(leaf);
  return at == stores_.end() ? nullptr : at->second;
}

void Lowering::Inside(ggml_tensor* t, TensorId root, std::uint64_t offset) {
  inside_.push_back({.t = t, .root = root, .offset = offset});
}

// The nodes the lowering makes: what the step returns, keeps or writes
// (state, a verify's saves), and what those read, through each node's
// sources but those its GGML form does not read (a fused selection's masks
// and block keys' view, a block-key pool's positions and cells, a
// preparation another kernel writes).
void Lowering::Needed() {
  needed_.assign(p_.nodes().size(), false);
  std::vector<NodeId> work;
  const auto need = [&](TensorId t) {
    if (!t.valid()) return;
    const NodeId id = p_.producer(t);
    if (!needed_.at(id.v)) {
      needed_.at(id.v) = true;
      work.push_back(id);
    }
  };
  for (const TensorId t : p_.outputs()) need(t);
  for (const TensorId t : p_.keep()) need(t);
  // A verify's saves and the rows they are stored at (MakeSaves).
  if (in_.token)
    for (const co::CommitSave& s : in_.token->step.saves) {
      need(plan_.tensors.at(s.tensor.v));
      if (s.rows.valid()) need(plan_.tensors.at(s.rows.v));
    }
  for (std::uint32_t i = 0; i < p_.nodes().size(); ++i) {
    const ir::Node& n = p_.nodes()[i];
    bool writes = false;
    if (n.key == ir::Primitive(Op::kCacheStore)) {
      writes = p_.tensor(p_.root(n.src[0])).role == ir::Role::kState;
    }
    for (std::size_t o = 0; o < n.nout; ++o) {
      writes = writes || (n.key.op == Op::kFused && ir::AliasedSource(n.key, o).has_value());
    }
    if (writes && !needed_.at(i)) {
      needed_.at(i) = true;
      work.push_back(NodeId{i});
    }
  }
  while (!work.empty()) {
    const NodeId id = work.back();
    work.pop_back();
    const ir::Node& n = p_.node(id);
    if (const auto f = folded_.find(id.v); f != folded_.end()) {
      need(p_.node(f->second).out[0]);
      continue;
    }
    for (std::size_t i = 0; i < n.nsrc; ++i) {
      const TensorId s = n.src.at(i);
      if (!s.valid()) continue;
      if (n.key == ir::Fused(FusedKind::kQsaAttention) && i >= 4 && i <= 7) {
        if (i == 4) need(p_.root(s));  // the block keys' leaf
        continue;
      }
      if (n.key == ir::Fused(FusedKind::kQsaPool) && (i == 3 || i == 4)) continue;
      // Stage the exact mask's proved equivalent positions, which may be
      // the original plain root rather than a replicated input.
      if (n.key == ir::Primitive(Op::kMaskBuild) && i == 0 &&
          n.Params<ir::MaskParams>().exact_rows && options_.device_masks) {
        need(exact_mask_positions_.at(n.out[0].v));
        continue;
      }
      need(s);
    }
    // A view reads its root.
    for (const TensorId o : n.outputs()) {
      if (p_.tensor(o).is_view()) need(p_.root(o));
    }
  }
}

// The fast forms' primitive products, conversions, preparations, lookups
// and the gated delta rule (`made` false: none of those, the generic path).
std::expected<void, KernelFailure> Lowering::MakeSpecial(NodeId id, const ir::Node& n, bool& made) {
  made = true;
  const std::string_view impl = Selected(id);
  const auto src = [&](std::size_t i) { return made_.at(n.src.at(i).v); };
  const ir::TensorDesc& out = p_.tensor(n.out[0]);
  const auto out_type = out.format.encoding == Encoding::kBF16 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
  switch (n.key.op) {
    case Op::kMatMul: {
      const ir::TensorDesc& w = p_.tensor(n.src[0]);
      const ir::TensorDesc& x = p_.tensor(n.src[1]);
      if (w.format.encoding == Encoding::kMxfp8) {
        ggml_tensor* codes = Part(n.src[0], format::PartRole::kCodes);
        ggml_tensor* scales = Part(n.src[0], format::PartRole::kBlockScales);
        if (codes == nullptr || scales == nullptr) return Refused("an MXFP8 product's parts");
        if (x.format.encoding == Encoding::kF32) {
          ggml_tensor* t = Mxfp8MulMatVec(c_, codes, scales, src(1));
          made_.at(n.out[0].v) = t;
          Node(t, id);
          return {};
        }
        if (!ir::PreparedRows(x.format)) return Refused("an MXFP8 product of an unprepared input");
        ggml_tensor* swizzled = Mxfp8Swizzle(c_, scales);
        ggml_tensor* t =
            Mxfp8Gemm(c_, src(1), codes, swizzled, out_type, static_cast<std::int64_t>(x.ne(1)));
        Node(swizzled, id);
        Node(t, id);
        made_.at(n.out[0].v) = t;
        Inside(swizzled, n.out[0], Round(ggml_nbytes(t), 256));
        return {};
      }
      if (w.format.encoding == Encoding::kBF16 && x.format.encoding == Encoding::kBF16) {
        // The hyper-connections' products (their normed streams' and
        // low-rank rows): GemvBf16, which takes its vector kernel at one
        // column; any other GemmBf16.
        const ir::Node& from = p_.node(p_.producer(p_.root(n.src[1])));
        const bool hc = from.key == ir::Fused(FusedKind::kHcPrep) ||
                        from.key == ir::Fused(FusedKind::kScaledUnary);
        ggml_tensor* t =
            hc ? GemvBf16(c_, src(0), src(1), out_type) : GemmBf16(c_, src(0), src(1), out_type);
        if (hc && (plan_.owner_row_invariant ||
                   (selected_hc_ && (selected_hc_->low == id || selected_hc_->high == id))))
          SetGemvBf16Wide(t);
        made_.at(n.out[0].v) = t;
        Node(t, id);
        return {};
      }
      if (x.format.encoding == Encoding::kQ8_1 && impl == "llmp.vecq.dense") {
        ggml_tensor* t =
            VecQ(c_, src(0), src(1), nullptr, static_cast<std::int64_t>(x.ne(1)), false);
        OwnersAlone(t);
        made_.at(n.out[0].v) = t;
        Node(t, id);
        return {};
      }
      break;
    }
    case Op::kMatMulId: {
      const ir::TensorDesc& x = p_.tensor(n.src[1]);
      if (x.format.encoding == Encoding::kQ8_1 && impl == "llmp.vecq.experts") {
        const ir::TensorDesc& ids = p_.tensor(n.src[2]);
        ggml_tensor* t =
            VecQ(c_, src(0), src(1), src(2), static_cast<std::int64_t>(ids.ne(1)), true);
        OwnersAlone(t);
        made_.at(n.out[0].v) = t;
        Node(t, id);
        return {};
      }
      break;
    }
    case Op::kConvert:
      if (impl == "ggml.convert") {
        if (!ir::IsContiguous(p_.tensor(n.src[0])))
          return Refused("a cast requires a packed source");
        const auto& dims = out.shape().ne;
        const auto cast_type =
            out.format.encoding == Encoding::kF16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        ggml_tensor* destination = ggml_new_tensor_4d(
            c_, cast_type, static_cast<std::int64_t>(dims[0]), static_cast<std::int64_t>(dims[1]),
            static_cast<std::int64_t>(dims[2]), static_cast<std::int64_t>(dims[3]));
        ggml_tensor* t = ggml_cpy(c_, src(0), destination);
        placed_root_[n.out[0].v] = destination;
        made_.at(n.out[0].v) = t;
        Node(destination, id);
        Node(t, id);
        return {};
      }
      if (impl == kBf16Name) {
        ggml_tensor* t = ToBf16(c_, src(0));
        made_.at(n.out[0].v) = t;
        Node(t, id);
        return {};
      }
      break;
    case Op::kPrepare:
      if (ir::PreparedRows(out.format)) {
        if (folded_.contains(id.v)) {
          // Written by the kernel of the node it is folded into.
          if (made_.at(n.out[0].v) == nullptr) return Refused("a folded preparation was not made");
          quiet_.at(id.v) = true;
          return {};
        }
        ggml_tensor* t = Mxfp8Quantize(c_, src(0));
        made_.at(n.out[0].v) = t;
        Node(t, id);
        return {};
      }
      break;
    case Op::kGetRows: {
      const ir::TensorDesc& table = p_.tensor(n.src[0]);
      if (table.format.encoding == Encoding::kNvfp4 &&
          table.format.layout == format::Layout::kRowInterleaved) {
        ggml_tensor* rows = Part(n.src[0], format::PartRole::kData);
        ggml_tensor* scale = Part(n.src[0], format::PartRole::kTensorScale);
        if (rows == nullptr || scale == nullptr) return Refused("an NVFP4 table's parts");
        ggml_tensor* t = Nvfp4Rows(c_, rows, src(1), scale, static_cast<std::int64_t>(table.ne(0)));
        made_.at(n.out[0].v) = t;
        Node(t, id);
        return {};
      }
      if (impl == kQRowsName) {
        ggml_tensor* t = QRows(c_, src(0), src(1));
        made_.at(n.out[0].v) = t;
        Node(t, id);
        return {};
      }
      break;
    }
    case Op::kSsmConv: {
      // Past 32 tokens upstream's convolution loads whole 32-token windows,
      // which would read past the input when the tokens are not a multiple
      // of 32 (RE-032; CheckSsmConv refuses that): the input padded to
      // whole windows with a copy of its first columns, whose outputs are
      // dropped, as Qwen3.8's builder pads it. The copy and the padded
      // input sit in the result's place after it (KernelBytes).
      ggml_tensor* x = src(0);
      const std::int64_t pad = SsmConvPad(p_, n);
      if (pad == 0) break;
      const std::int64_t nt = static_cast<std::int64_t>(out.ne(1));
      ggml_tensor* first = ggml_view_3d(c_, x, pad, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
      ggml_tensor* copy = ggml_cont(c_, first);
      ggml_tensor* wide = ggml_concat(c_, x, copy, 0);
      ggml_tensor* conv = ggml_ssm_conv(c_, wide, src(1));
      ggml_tensor* t = ggml_view_3d(c_, conv, conv->ne[0], nt, conv->ne[2], conv->nb[1],
                                    conv->nb[1] * static_cast<std::size_t>(nt), 0);
      for (ggml_tensor* g : {first, copy, wide, conv, t}) Node(g, id);
      made_.at(n.out[0].v) = t;
      placed_root_[n.out[0].v] = conv;
      Inside(copy, n.out[0], Round(ggml_nbytes(conv), 256));
      Inside(wide, n.out[0], Round(ggml_nbytes(conv), 256) + Round(ggml_nbytes(copy), 256));
      return {};
    }
    case Op::kGatedDeltaNet: {
      // GGML's one result: the attention, then the state; views of it.
      ggml_tensor* result =
          ggml_gated_delta_net(c_, src(0), src(1), src(2), src(3), src(4), src(5), 1);
      Node(result, id);
      const ir::TensorDesc& v = p_.tensor(n.src[2]);
      const std::int64_t d = static_cast<std::int64_t>(v.ne(0));
      const std::int64_t hv = static_cast<std::int64_t>(v.ne(1));
      const std::int64_t nt = static_cast<std::int64_t>(v.ne(2));
      const std::int64_t dk = static_cast<std::int64_t>(p_.tensor(n.src[0]).ne(0));
      ggml_tensor* attn = ggml_view_4d(c_, result, d, hv, nt, 1, ggml_row_size(result->type, d),
                                       ggml_row_size(result->type, d * hv),
                                       ggml_row_size(result->type, d * hv * nt), 0);
      ggml_tensor* state = ggml_view_4d(c_, result, dk, d, hv, 1, ggml_row_size(result->type, dk),
                                        ggml_row_size(result->type, dk * d),
                                        ggml_row_size(result->type, dk * d * hv),
                                        ggml_row_size(result->type, d * hv * nt));
      Node(attn, id);
      Node(state, id);
      made_.at(n.out[0].v) = attn;
      made_.at(n.out[1].v) = state;
      placed_root_[n.out[0].v] = result;
      return {};
    }
    default:
      break;
  }
  made = false;
  return {};
}

// llmpalooza's fast-form kernels over GGML tensors, as qwen38_graph.cc's
// builder makes them for the node's sources.
std::expected<void, KernelFailure> Lowering::MakeDirect(NodeId id, const ir::Node& n) {
  const auto src = [&](std::size_t i) -> ggml_tensor* {
    return i < n.nsrc && n.src.at(i).valid() ? made_.at(n.src.at(i).v) : nullptr;
  };
  const auto desc = [&](std::size_t i) -> const ir::TensorDesc& { return p_.tensor(n.src.at(i)); };
  const auto set = [&](std::size_t o, ggml_tensor* t) { made_.at(n.out.at(o).v) = t; };
  for (std::size_t i = 0; i < n.nsrc; ++i) {
    // (The sources its kernels do not read are made where needed.)
    if (n.src.at(i).valid() && made_.at(n.src.at(i).v) == nullptr &&
        !(n.key == ir::Fused(FusedKind::kQsaAttention) && i >= 4 && i <= 7) &&
        !(n.key == ir::Fused(FusedKind::kQsaPool) && (i == 3 || i == 4))) {
      return Refused(std::format("{}'s source {} was not made", ir::OpKeyName(n.key), i));
    }
  }
  switch (n.key.fused) {
    case FusedKind::kGroupedMatMul: {
      const auto& reshape = p_.node(p_.producer(n.src[0]));
      if (p_.tensor(n.out[0]).format.encoding != Encoding::kF32 ||
          Selected(id) != "llmp.vecq.grouped" || reshape.key != ir::Primitive(Op::kReshape) ||
          reshape.nsrc != 1 || p_.tensor(n.src[0]).view_offset != 0)
        return Refused("a grouped product requires its complete weight reshape");
      const auto& leaf = p_.node(p_.producer(reshape.src[0]));
      const auto& original = p_.tensor(reshape.src[0]);
      const auto& w = desc(0);
      const auto parts = format::PartsOf(original.format);
      const auto* reference = std::get_if<ir::WeightRef>(&original.binding);
      if (!reference || reference->expert_array || leaf.key != ir::Primitive(Op::kWeight) ||
          original.is_view() || original.format.shape.rank != 2 || original.format.stacked != 1 ||
          original.format.layout != format::Layout::kGgmlBlocks || original.ne(0) != w.ne(0) ||
          w.ne(2) < 2 || original.ne(1) % w.ne(2) != 0 || original.ne(1) / w.ne(2) != w.ne(1) ||
          !parts || parts->size() != 1 || (*parts)[0].role != format::PartRole::kData ||
          !ir::IsContiguous(desc(1)))
        return Refused("a grouped product lost its immutable complete source binding");
      ggml_tensor* prepared = QuantizeQ8(c_, src(1));
      ggml_tensor* result =
          VecQ(c_, src(0), prepared, nullptr, static_cast<std::int64_t>(desc(1).ne(2)), true);
      OwnersAlone(result);
      Node(prepared, id);
      Node(result, id);
      set(0, result);
      Inside(prepared, n.out[0], Round(ggml_nbytes(result), 256));
      return {};
    }
    case FusedKind::kStateBankAttention: {
      const auto p = n.Params<ir::AttnParams>();
      const auto slices = planner::StateBankSlices(p_, n.src[1], n.src[2]);
      if (!slices || !in_.token->state_spans ||
          (p.sparse_any && (src(0)->ne[0] != 512 || p.n_kv_max == 0)) ||
          p.n_kv_max > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
        return Refused("a state-bank attention without a supported span proof");
      const auto fact = in_.token->state_spans(*slices);
      const std::uint64_t raw_bytes = desc(1).ne(2) * desc(1).nb[2];
      const std::uint64_t bytes = raw_bytes + desc(2).ne(2) * desc(2).nb[2];
      if (!fact || fact->bank.encoding != Encoding::kF16 ||
          fact->bank.cell_bytes != desc(1).nb[2] || fact->bank.range.bytes != bytes ||
          fact->bank.range.offset > fact->region_bytes ||
          bytes > fact->region_bytes - fact->bank.range.offset)
        return Refused("a state-bank span differs from its exact operands");
      std::array<ggml_tensor*, 2> roots{};
      for (std::size_t i = 0; i < 2; ++i) {
        const auto root = p_.root(n.src[1 + i]);
        const auto prim = primitive_.at(root.v);
        const auto item = state_.find(prim.v);
        if (!prim.valid() || item == state_.end() || item->second->borrowed || item->second->tap ||
            item->second->ref != (*slices)[i].ref ||
            (*slices)[i].ref.stage != in_.token->shape.stage ||
            (*slices)[i].ref.segment >= in_.token->shape.segments.size())
          return Refused("a state-bank operand is not this owner's owned StepState");
        roots[i] = made_.at(root.v);
      }
      // Same descriptor and strides as Concat -> Permute, with the copy
      // replaced by a bound state span. Neither the mask nor any Attention
      // parameter or device choice changes.
      ggml_tensor* bank =
          ggml_new_tensor_4d(c_, GGML_TYPE_F16, src(1)->ne[0], 1, src(1)->ne[2] + src(2)->ne[2], 1);
      state_tensors_.insert(bank);
      state_banks_.push_back({.tensor = bank,
                              .refs = {(*slices)[0].ref, (*slices)[1].ref},
                              .roots = roots,
                              .slices = *slices,
                              .fact = *fact,
                              .raw_bytes = raw_bytes});
      ggml_tensor* keys = ggml_permute(c_, bank, 0, 2, 1, 3);
      ggml_tensor* values = ggml_permute(c_, bank, 0, 2, 1, 3);
      ggml_tensor* out =
          ggml_flash_attn_ext(c_, src(0), keys, values, src(3), p.scale, p.max_bias, p.softcap);
      if (p.accumulate == ir::Accumulate::kF32) ggml_prec_set_acc(out, GGML_PREC_F32);
      if (src(4) != nullptr) ggml_flash_attn_ext_add_sinks(out, src(4));
      if (p.n_kv_max != 0)
        ggml_flash_attn_ext_set_n_kv_max(out, static_cast<std::int32_t>(p.n_kv_max));
      if (p.sparse_any) SetFlashAttnSparseAny(out);
      if (auto match = Matches(p_.tensor(n.out[0]), out, "a state-bank attention"); !match)
        return std::unexpected(match.error());
      bank_attentions_.insert(out);
      for (ggml_tensor* t : {bank, keys, values, out}) Node(t, id);
      set(0, out);
      return {};
    }
    case FusedKind::kDsv4QHead: {
      const auto p = n.Params<ir::NormRopeParams>();
      const auto& r = p.rope;
      const bool f16_q =
          src(0)->ne[0] == 512 && src(0)->ne[2] >= 64 && !readers_.at(n.out[0].v).empty() &&
          std::ranges::all_of(readers_.at(n.out[0].v), [&](NodeId v_id) {
            const ir::Node& v = p_.node(v_id);
            return v.key == ir::Primitive(Op::kView) && !readers_.at(v.out[0].v).empty() &&
                   std::ranges::all_of(readers_.at(v.out[0].v), [&](NodeId perm_id) {
                     const ir::Node& perm = p_.node(perm_id);
                     return perm.key == ir::Primitive(Op::kPermute) &&
                            !readers_.at(perm.out[0].v).empty() &&
                            std::ranges::all_of(readers_.at(perm.out[0].v), [&](NodeId attn_id) {
                              const ir::Node& attn = p_.node(attn_id);
                              return (attn.key == ir::Fused(FusedKind::kStateBankAttention) ||
                                      attn.key == ir::Primitive(Op::kAttention)) &&
                                     attn.src[0] == perm.out[0];
                            });
                   });
          });
      ggml_tensor* out =
          Dsv4QHead(c_, src(0), src(1),
                    {.eps = p.eps,
                     .original_context = static_cast<std::int32_t>(r.original_context),
                     .base = r.base,
                     .scale = r.scale,
                     .extension = r.ext_factor,
                     .attention = r.attn_factor,
                     .beta_fast = r.beta_fast,
                     .beta_slow = r.beta_slow},
                    f16_q ? GGML_TYPE_F16 : GGML_TYPE_F32);
      Node(out, id);
      set(0, out);
      return {};
    }
    case FusedKind::kDsv4LidMask: {
      const auto p = n.Params<ir::LidMaskParams>();
      ggml_tensor* keys = ggml_view_2d(c_, src(1), src(1)->ne[0], src(1)->ne[2], src(1)->nb[2], 0);
      ggml_tensor* ids = Dsv4LidTopK(c_, src(0), keys, src(2), src(3), p.top);
      ggml_tensor* mask = Dsv4SparseMask(c_, src(4), ids, nullptr, p.raw_cells, p.cells);
      Node(keys, id);
      Node(ids, id);
      Node(mask, id);
      // IDs live after the visible mask in its funded root; score/radix
      // scratch is owned by the ordinary native GraphPlan workspace.
      Inside(ids, n.out[0], Round(ggml_nbytes(mask), 256));
      set(0, mask);
      return {};
    }
    case FusedKind::kDsv4Compress: {
      const auto p = n.Params<ir::CompressorParams>();
      ggml_tensor* out =
          Dsv4Compress(c_, src(0), src(1), src(2), src(3), src(4), p.ratio, p.overlap);
      Node(out, id);
      set(0, out);
      return {};
    }
    case FusedKind::kDsv4HcPre: {
      const auto p = n.Params<ir::SinkhornPreParams>();
      const auto width = src(0)->ne[0];
      const auto tokens = src(0)->ne[2];
      ggml_tensor* partials = Dsv4HcMix(c_, src(0), src(1));
      ggml_tensor* blob = Dsv4HcPre(c_, partials, src(0), src(2), src(3), src(4), p.eps, p.hc_eps,
                                    static_cast<std::int32_t>(p.iterations));
      Node(partials, id);
      Node(blob, id);
      placed_root_[n.out[0].v] = blob;
      Inside(partials, n.out[0], Round(ggml_nbytes(blob), 256));
      const auto tail = static_cast<std::size_t>(width * tokens) * sizeof(float);
      ggml_tensor* normed =
          ggml_view_2d(c_, blob, width, tokens, static_cast<std::size_t>(width) * sizeof(float), 0);
      ggml_tensor* post = ggml_view_2d(c_, blob, 4, tokens, kDsv4HcTail * sizeof(float), tail);
      ggml_tensor* comb = ggml_view_3d(c_, blob, 4, 4, tokens, 4 * sizeof(float),
                                       kDsv4HcTail * sizeof(float), tail + 4 * sizeof(float));
      Node(normed, id);
      Node(post, id);
      ggml_tensor* post_packed = ggml_cont(c_, post);
      Node(post_packed, id);
      Node(comb, id);
      ggml_tensor* comb_packed = ggml_cont(c_, comb);
      Node(comb_packed, id);
      set(0, normed);
      set(1, post_packed);
      set(2, comb_packed);
      return {};
    }
    case FusedKind::kHcNormPost: {
      const auto p = n.Params<ir::HcNormPostParams>();
      ggml_tensor* post = ggml_dsv4_hc_post(c_, src(0), src(1), src(2), src(3));
      ggml_tensor* flat = ggml_reshape_2d(c_, post, post->ne[0] * post->ne[1], post->ne[2]);
      ggml_tensor* norm = Dsv4HcNormF16(c_, flat, p.eps, GGML_TYPE_F32);
      Node(post, id);
      Node(flat, id);
      Node(norm, id);
      set(0, post);
      set(1, norm);
      return {};
    }
    case FusedKind::kHcPrep: {
      const auto p = n.Params<ir::HcNormPostParams>();
      const ir::TensorDesc& xd = desc(0);
      const auto width = static_cast<std::int64_t>(xd.ne(0));
      const auto hc = static_cast<std::int64_t>(xd.ne(1));
      const auto nt = static_cast<std::int64_t>(xd.ne(2));
      ggml_tensor* out = src(3) != nullptr ? ggml_reshape_2d(c_, src(3), width, nt) : nullptr;
      if (out != nullptr) Node(out, id);
      ggml_tensor* blob = HcPrep(c_, src(0), src(1), src(2), out, src(4), p.eps);
      Node(blob, id);
      const HcPrepLayout layout{.width = width,
                                .hc = hc,
                                .t = nt,
                                .combine = out != nullptr,
                                .inject = src(2) != nullptr};
      ggml_tensor* xn = TypedView(c_, blob, GGML_TYPE_BF16, width * hc, nt, layout.normed());
      Node(xn, id);
      set(0, xn);
      placed_root_[n.out[0].v] = blob;
      std::size_t k = 1;
      if (src(2) != nullptr) {
        ggml_tensor* logits = TypedView(c_, blob, GGML_TYPE_F32, hc, nt, layout.logits());
        Node(logits, id);
        set(k++, logits);
      }
      if (out != nullptr) {
        ggml_tensor* streams =
            TypedView(c_, blob, GGML_TYPE_F32, width * hc, nt, HcPrepLayout::streams());
        ggml_tensor* res = ggml_reshape_3d(c_, streams, width, hc, nt);
        Node(streams, id);
        Node(res, id);
        set(k++, res);
      }
      return {};
    }
    case FusedKind::kScaledUnary: {
      const auto p = n.Params<ir::ScaledUnaryParams>();
      const auto hc = static_cast<std::int64_t>(std::lround(1.0 / static_cast<double>(p.scale)));
      ggml_tensor* t = HcLo(c_, src(0), hc);
      Node(t, id);
      set(0, t);
      return {};
    }
    case FusedKind::kHcFold: {
      const auto p = n.Params<ir::HcFoldParams>();
      const auto hc = static_cast<std::int64_t>(p.streams);
      const ir::TensorDesc& od = p_.tensor(n.out[0]);
      const auto width = static_cast<std::int64_t>(od.ne(0));
      const auto nt = static_cast<std::int64_t>(od.ne(1));
      if (!p.prepared) {
        ggml_tensor* t = HcMixBf16(c_, src(0), src(1), hc);
        Node(t, id);
        set(0, t);
        return {};
      }
      ggml_tensor* both = HcMixBf16(c_, src(0), src(1), hc, true, p.rounded);
      Node(both, id);
      const HcMixLayout mix{.width = width, .t = nt, .bf16 = p.rounded};
      ggml_tensor* mixed = TypedView(c_, both, GGML_TYPE_F32, width, nt, HcMixLayout::mixed());
      ggml_tensor* rows =
          TypedView(c_, both, GGML_TYPE_I8, static_cast<std::int64_t>(mix.quantized_bytes()), 1,
                    mix.quantized());
      Node(mixed, id);
      Node(rows, id);
      set(0, mixed);
      set(1, rows);
      placed_root_[n.out[0].v] = both;
      if (p.rounded) {
        ggml_tensor* rounded = TypedView(c_, both, GGML_TYPE_BF16, width, nt, mix.rounded());
        Node(rounded, id);
        set(2, rounded);
      }
      return {};
    }
    case FusedKind::kHcCombine: {
      const ir::TensorDesc& xd = desc(0);
      ggml_tensor* out = ggml_reshape_2d(c_, src(1), static_cast<std::int64_t>(xd.ne(0)),
                                         static_cast<std::int64_t>(xd.ne(2)));
      Node(out, id);
      ggml_tensor* t = HcCombine(c_, src(0), out, src(2));
      Node(t, id);
      set(0, t);
      return {};
    }
    case FusedKind::kHcNorm: {
      const auto p = n.Params<ir::NormParams>();
      ggml_tensor* t = HcNorm(
          c_, src(0), src(1), p.eps,
          p_.tensor(n.out[0]).format.encoding == Encoding::kBF16 ? GGML_TYPE_BF16 : GGML_TYPE_F32);
      Node(t, id);
      set(0, t);
      return {};
    }
    case FusedKind::kHcMixNorm: {
      const auto p = n.Params<ir::HcNormPostParams>();
      ggml_tensor* t = HcMix(c_, src(0), src(1), src(2), p.eps);
      Node(t, id);
      set(0, t);
      return {};
    }
    case FusedKind::kGdnConv: {
      const auto p = n.Params<ir::GdnConvParams>();
      ggml_tensor* t = GdnConv(c_, src(0), src(1), src(2), p.qk_channels, p.head, p.eps, p.scale);
      Node(t, id);
      set(0, t);
      return {};
    }
    case FusedKind::kGdnHistory: {
      const auto p = n.Params<ir::GdnHistoryParams>();
      ggml_tensor* t = GdnHistory(c_, src(0), p.taps, src(1));
      Node(t, id);
      set(0, t);
      return {};
    }
    case FusedKind::kGdnStep: {
      const auto p = n.Params<ir::GdnStepParams>();
      const ir::TensorDesc& q = desc(0);
      const ir::TensorDesc& v = desc(2);
      ggml_tensor* state = ggml_reshape_4d(c_, src(5), static_cast<std::int64_t>(q.ne(0)),
                                           static_cast<std::int64_t>(v.ne(0)),
                                           static_cast<std::int64_t>(v.ne(1)), 1);
      Node(state, id);
      ggml_tensor* t = GdnStep(c_, src(0), src(1), src(2), src(3), src(4), state, p.write);
      Node(t, id);
      set(0, t);
      if (p.write) {
        set(1, src(5));
        stores_[src(5)] = t;
      }
      return {};
    }
    case FusedKind::kGdnGates: {
      ggml_tensor* gates = GdnGates(c_, src(0), src(1), src(2), src(3));
      Node(gates, id);
      const ir::TensorDesc& a = desc(0);
      const auto hv = static_cast<std::int64_t>(a.ne(0));
      const auto nt = static_cast<std::int64_t>(a.ne(1));
      const std::size_t row = ggml_row_size(GGML_TYPE_F32, hv);
      const std::size_t plane = row * static_cast<std::size_t>(nt);
      ggml_tensor* gate = ggml_view_4d(c_, gates, 1, hv, nt, 1, sizeof(float), row, plane, 0);
      ggml_tensor* beta = ggml_view_4d(c_, gates, 1, hv, nt, 1, sizeof(float), row, plane, plane);
      Node(gate, id);
      Node(beta, id);
      set(0, gate);
      set(1, beta);
      placed_root_[n.out[0].v] = gates;
      return {};
    }
    case FusedKind::kNormGate: {
      const auto p = n.Params<ir::NormParams>();
      const ir::TensorDesc& o = desc(0);
      const auto d = static_cast<std::int64_t>(o.ne(0));
      const auto heads = static_cast<std::int64_t>(o.ne(1));
      const auto nt = static_cast<std::int64_t>(o.ne(2));
      ggml_tensor* z = ggml_reshape_2d(c_, src(2), d * heads, nt);
      Node(z, id);
      // Folded into the preparation that reads it: MXFP8 rows.
      NodeId into{};
      for (const auto& [prepare, producer] : folded_) {
        if (producer == id) into = NodeId{prepare};
      }
      const ggml_type type = into.valid() ? GGML_TYPE_I8 : GGML_TYPE_F32;
      ggml_tensor* t = GdnNormGate(c_, src(0), src(1), z, p.eps, type);
      Node(t, id);
      set(0, t);
      if (into.valid()) made_.at(p_.node(into).out[0].v) = t;
      return {};
    }
    case FusedKind::kHeadGate: {
      const auto p = n.Params<ir::HeadGateParams>();
      ggml_tensor* t = QsaGateQuantize(c_, src(0), src(1), p.head);
      Node(t, id);
      set(0, t);
      return {};
    }
    case FusedKind::kRmsNormMulRope: {
      // llmp.qsa.prep: the heads of x's root at x's stride.
      const auto p = n.Params<ir::NormRopeParams>();
      ggml_tensor* x = src(0);
      ggml_tensor* root = x->view_src != nullptr ? x->view_src : x;
      const ir::TensorDesc& xd = desc(0);
      const auto d = static_cast<std::int64_t>(xd.ne(0));
      const auto heads = static_cast<std::int64_t>(xd.ne(1));
      const auto stride = static_cast<std::int64_t>(xd.nb.at(1) / sizeof(float));
      if (xd.view_offset != 0) return Refused("a head prep of a view past its root's start");
      const float theta = std::pow(p.rope.base, -2.0F / static_cast<float>(p.rope.dims));
      ggml_tensor* t = QsaPrep(c_, root, src(1), src(2), d, heads, stride, p.eps, theta);
      Node(t, id);
      set(0, t);
      return {};
    }
    case FusedKind::kQsaPool: {
      const auto p = n.Params<ir::QsaPoolParams>();
      ggml_tensor* blocks = src(1);
      ggml_tensor* positions = SegmentPositions(n, 5, id);
      ggml_tensor* raw_leaf = made_.at(p_.root(n.src[0]).v);
      ggml_tensor* raw = StoredInto(raw_leaf);
      if (raw == nullptr) return Refused("a block pool before its raw keys' store");
      const float theta = std::pow(p.rope.base, -2.0F / static_cast<float>(p.rope.dims));
      ggml_tensor* t = QsaPool(c_, raw, blocks, src(2), positions, p.ratio, p.eps, theta);
      Node(t, id);
      // Its I32 [1] result is an ordering token (llmp.qsa.topk reads it),
      // never written, apart from every operand: past the positions' bytes
      // in their place (WrittenBytes), which every later reader keeps.
      const ir::TensorId held = p_.root(n.src[5]);
      Inside(t, held, Round(planner::RootBytes(p_.tensor(held)).value_or(0), 256));
      set(0, blocks);
      pools_[blocks] = t;
      return {};
    }
    case FusedKind::kQsaAttention: {
      const auto p = n.Params<ir::QsaAttnParams>();
      ggml_tensor* blocks = made_.at(p_.root(n.src[4]).v);
      const auto pool = pools_.find(blocks);
      ggml_tensor* k = StoredInto(made_.at(p_.root(n.src[1]).v));
      ggml_tensor* v = StoredInto(made_.at(p_.root(n.src[2]).v));
      if (pool == pools_.end() || k == nullptr || v == nullptr) {
        return Refused("a selection before its block keys' pool or its caches' stores");
      }
      const auto n_blocks = static_cast<std::int64_t>(desc(4).ne(1));
      // (Over the positions its pool read: the same tensor, as the kernel
      // checks.)
      ggml_tensor* top = QsaTopK(c_, src(3), blocks, pool->second->src[3], pool->second, n_blocks,
                                 p.width, p.ratio);
      ggml_tensor* t = QsaAttn(c_, src(0), k, v, top, p.attn.scale);
      Node(top, id);
      Node(t, id);
      set(0, t);
      Inside(top, n.out[0], Round(ggml_nbytes(t), 256));
      return {};
    }
    case FusedKind::kMoeRouter: {
      const auto p = n.Params<ir::RouteParams>();
      const auto nt = static_cast<std::int64_t>(desc(0).ne(1));
      const auto used = static_cast<std::int64_t>(p.used);
      ggml_tensor* routed = MoeRouter(c_, src(0), src(1), src(2), used);
      Node(routed, id);
      const MoeRouterLayout layout{.used = used, .t = nt};
      ggml_tensor* ids = TypedView(c_, routed, GGML_TYPE_I32, used, nt, MoeRouterLayout::ids());
      ggml_tensor* w = TypedView(c_, routed, GGML_TYPE_F32, used, nt, layout.weights());
      ggml_tensor* gate = TypedView(c_, routed, GGML_TYPE_F32, 1, nt, layout.gate());
      for (ggml_tensor* t : {ids, w, gate}) Node(t, id);
      set(0, ids);
      set(1, w);
      set(2, gate);
      placed_root_[n.out[0].v] = routed;
      return {};
    }
    case FusedKind::kMoeGlu: {
      ggml_tensor* t = MoeGlu(c_, src(0), src(1), src(2), src(3), src(4));
      Node(t, id);
      set(0, t);
      return {};
    }
    case FusedKind::kMoeCombine: {
      if (Selected(id) == kDsv4CombineName) {
        ggml_tensor* w_down = src(2);
        if (w_down->ne[0] == 6 && w_down->ne[2] == 1) {
          w_down = ggml_reshape_3d(c_, w_down, 1, 6, w_down->ne[1]);
          Node(w_down, id);
        }
        const bool has_hc_norm_post =
            !readers_.at(n.out[0].v).empty() &&
            std::ranges::any_of(readers_.at(n.out[0].v), [&](NodeId r_id) {
              return p_.node(r_id).key == ir::Fused(FusedKind::kHcNormPost);
            });
        if (has_hc_norm_post && src(0)->ne[2] > kVecQTokens &&
            (device_.hc_post_experts || dsv4_fast_) && Dsv4WeightedReduceFits(src(0), w_down)) {
          ggml_tensor* reduce = Dsv4OrderedReduce(c_, src(0), w_down);
          Node(reduce, id);
          ggml_tensor* add = ggml_add(c_, reduce, src(3));
          Node(add, id);
          elided_.emplace_back(reduce, add);
          set(0, add);
          return {};
        }
        auto* t = CombineSelectedWeights(c_, src(0), src(2), src(3));
        Node(t, id);
        set(0, t);
        return {};
      }
      const ir::TensorDesc& w = desc(2);
      ggml_tensor* weights = ggml_reshape_3d(c_, src(2), 1, static_cast<std::int64_t>(w.ne(0)),
                                             static_cast<std::int64_t>(w.ne(1)));
      Node(weights, id);
      ggml_tensor* t = MoeCombine(c_, src(0), src(1), src(5), weights, src(3), src(4));
      Node(t, id);
      set(0, t);
      return {};
    }
    case FusedKind::kDsv4OutA: {
      const auto p = n.Params<ir::RopeParams>();
      const Dsv4OutAParams outa_params{
          .original_context = static_cast<std::int32_t>(p.original_context),
          .base = p.base,
          .scale = p.scale,
          .extension = p.ext_factor,
          .attention = p.attn_factor,
          .beta_fast = p.beta_fast,
          .beta_slow = p.beta_slow,
      };
      ggml_tensor* w_outa = src(0);
      if (w_outa->ne[1] == 8192 && w_outa->ne[2] == 1) {
        w_outa = ggml_reshape_3d(c_, w_outa, 4096, 1024, 8);
        Node(w_outa, id);
      }
      ggml_tensor* low = Dsv4OutA(c_, w_outa, src(1), src(2), outa_params);
      Node(low, id);
      const auto rows = src(1)->ne[2];
      if (rows % 16 != 0) {
        ggml_tensor* view = ggml_view_2d(c_, low, 8192, rows, low->nb[1], 0);
        Node(view, id);
        placed_root_[n.out[0].v] = low;
        set(0, view);
      } else {
        set(0, low);
      }
      return {};
    }
    case FusedKind::kMoeExperts: {
      const auto p = n.Params<ir::MoeExpertsParams>();
      ggml_tensor* experts = src(0);
      if (experts != src(1)) return Refused("the CUTLASS experts' arrays in two slabs");
      const ir::TensorDesc& xd = desc(2);
      const ir::TensorDesc& ids = desc(3);
      const auto width = static_cast<std::int64_t>(xd.ne(0));
      const auto nt = static_cast<std::int64_t>(xd.ne(1));
      const auto used = static_cast<std::int64_t>(ids.ne(0));
      const auto f = static_cast<std::int64_t>(p.hidden);
      const auto e = static_cast<std::int64_t>(p_.tensor(n.src[7]).ne(0));
      const moe::ExpertLayout layout{.ffn = static_cast<std::uint64_t>(f),
                                     .width = static_cast<std::uint64_t>(width)};
      ggml_tensor* weights = ggml_reshape_3d(c_, src(4), 1, used, nt);
      Node(weights, id);
      std::vector<ggml_tensor*> inner;
      ggml_tensor* t = nullptr;
      if (nt <= kMoeGemvWaveTokens) {
        ggml_tensor* x = ggml_reshape_3d(c_, src(2), width, 1, nt);
        Node(x, id);
        ggml_tensor* act =
            MoeGemvSwiglu(c_, experts, x, src(3), f, src(7), src(8),
                          moe::ExpertLayout::gate_up_codes(), layout.gate_up_scales());
        ggml_tensor* down = MoeGemv(c_, experts, act, src(3), width, 0, width, layout.down_codes(),
                                    layout.down_scales());
        t = MoeCombine(c_, down, src(3), src(9), weights, src(5), src(6));
        inner = {act, down};
      } else {
        ggml_tensor* route = MoeRoute(c_, src(3), e);
        ggml_tensor* a1 = MoeQuantize(c_, src(2), route);
        ggml_tensor* d1 = MoeGemm(c_, a1, route, experts, 2 * f, moe::ExpertLayout::gate_up_codes(),
                                  layout.gate_up_scales());
        ggml_tensor* a2 = MoeGluQuantize(c_, d1, a1, route, src(7), src(8));
        ggml_tensor* d2 =
            MoeGemm(c_, a2, route, experts, width, layout.down_codes(), layout.down_scales());
        t = MoeCombineSorted(c_, d2, a2, route, src(9), weights, src(5), src(6));
        inner = {route, a1, d1, a2, d2};
      }
      for (ggml_tensor* x : inner) Node(x, id);
      Node(t, id);
      set(0, t);
      std::uint64_t at = Round(ggml_nbytes(t), 256);
      for (ggml_tensor* x : inner) {
        Inside(x, n.out[0], at);
        at += Round(ggml_nbytes(x), 256);
      }
      return {};
    }
    case FusedKind::kMatMulGlu: {
      // llmp.vecq's GLU form: up's product with gate's (weights, then gate).
      const ir::TensorDesc& x = desc(2);
      const auto glu = n.Params<ir::GluParams>();
      const VecQGlu kind =
          glu.kind == ir::GluKind::kSwiGluClamp ? VecQGlu::kSwigluClamp : VecQGlu::kSwiglu;
      ggml_tensor* t = VecQ(c_, src(1), src(2), nullptr, static_cast<std::int64_t>(x.ne(1)), false,
                            src(0), kind, glu.limit);
      OwnersAlone(t);
      Node(t, id);
      set(0, t);
      return {};
    }
    case FusedKind::kMatMulIdGlu: {
      const ir::TensorDesc& ids = desc(3);
      const auto glu = n.Params<ir::MoeExpertsParams>().glu;
      const VecQGlu kind =
          glu.kind == ir::GluKind::kSwiGluClamp ? VecQGlu::kSwigluClamp : VecQGlu::kSwiglu;
      ggml_tensor* t = VecQ(c_, src(1), src(2), src(3), static_cast<std::int64_t>(ids.ne(1)), false,
                            src(0), kind, glu.limit);
      OwnersAlone(t);
      Node(t, id);
      set(0, t);
      return {};
    }
    default:
      break;
  }
  return Refused(std::format("no GGML lowering of {}", ir::OpKeyName(n.key)));
}

// A verify's saves (StepGraph::saves): each stored into its buffer at the
// caller's address, at its rows (qwen38_graph.cc's Save).
std::expected<void, KernelFailure> Lowering::MakeSaves() {
  if (!in_.token) return {};
  for (const co::CommitSave& s : in_.token->step.saves) {
    const TensorId planned = plan_.tensors.at(s.tensor.v);
    const TensorId rows_input = s.rows.valid() ? plan_.tensors.at(s.rows.v) : TensorId{};
    ggml_tensor* src = planned.valid() ? made_.at(planned.v) : nullptr;
    ggml_tensor* rows = rows_input.valid() ? made_.at(rows_input.v) : nullptr;
    if (src == nullptr || rows == nullptr) return Refused("a save or its rows were not made");
    const std::int64_t width = src->ne[0];
    const std::int64_t n = ggml_nelements(src) / width;
    const bool slotted = s.segment == ir::kJoinedSegment && in_.addresses.save_rows != 0;
    if (slotted && in_.addresses.save_rows < static_cast<std::uint64_t>(n)) {
      return Refused("a verify's joined save exceeds its buffer");
    }
    ggml_tensor* buffer = ggml_new_tensor_2d(
        c_, GGML_TYPE_F32, width, slotted ? static_cast<std::int64_t>(in_.addresses.save_rows) : n);
    if (slotted) {
      for (auto& [tensor, input] : input_leaves_) {
        if (tensor == rows) input.commit_save = true;
      }
    }
    save_leaves_.emplace_back(buffer, &s);
    out_.saves.push_back({.tensor = buffer,
                          .layer = s.layer,
                          .component = s.component,
                          .what = s.what,
                          .segment = s.segment});
    ggml_tensor* packed = IsPacked(src) ? src : ggml_cont(c_, src);
    if (packed != src) {
      out_.nodes.push_back(packed);
      extra_.insert(packed);
    }
    ggml_tensor* flat = ggml_reshape_2d(c_, packed, width, n);
    ggml_tensor* store = ggml_set_rows(c_, buffer, flat, rows);
    out_.nodes.push_back(flat);
    out_.nodes.push_back(store);
    extra_.insert(store);
    out_.keep.push_back(store);
  }
  return {};
}

std::expected<void, KernelFailure> Lowering::MakeRouting() {
  if (in_.addresses.expert_part) {
    out_.weight_reads.emplace();
    out_.weight_reads->reserve(in_.io.weights.size());
    for (const auto& w : in_.io.weights)
      out_.weight_reads->push_back({w.ref, g_.tensor(w.tensor).format, w.parts, false});
  }
  std::size_t products = 0;
  for (const auto& selected : out_.selected)
    if (selected.call->route()) ++products;
  if (products == 0) return {};
  if (products > out_.plan.steps.size() || products == SIZE_MAX || !options_.routed_funder)
    return Refused("routed products need funded ordered steps");
  struct Group {
    ggml_tensor* ids = nullptr;
    std::vector<std::size_t> calls;
    std::size_t cut = 0, last = 0;
    TensorId original{};
  };
  // Count before allocating. At most one group per selected product, total
  // call/member entries <= products. 4096 covers each group object, fixed
  // copy arrays, call/member vectors and temporary/retained overlap.
  static_assert(sizeof(LoweredRouting) + sizeof(Group) + 16 * sizeof(RoutedCopy) +
                    8 * sizeof(std::size_t) + 2 * sizeof(std::pair<std::size_t, std::uint64_t>) +
                    4 * sizeof(std::unique_ptr<RoutedRecording>) + 1024 <
                4096);
  std::uint64_t charge = 0;
  if (__builtin_mul_overflow(static_cast<std::uint64_t>(products), std::uint64_t{4096}, &charge) ||
      __builtin_add_overflow(charge, std::uint64_t{4096}, &charge) || charge > SIZE_MAX ||
      !options_.routed_funder->Charge(charge))
    return Refused("ordered routed metadata was not admitted");
  out_.routing_payment = std::shared_ptr<const void>(
      new std::byte{}, [owner = options_.routed_funder, charge](const void* token) {
        delete static_cast<const std::byte*>(token);
        owner->Release(charge);
      });
  std::vector<Group> groups;
  groups.reserve(products);
  out_.routing.reserve(products);
  for (std::size_t index = 0; index != out_.selected.size(); ++index) {
    const auto route = out_.selected[index].call->route();
    if (!route) continue;
    auto group = std::ranges::find(groups, route->ids, &Group::ids);
    if (group == groups.end()) {
      std::size_t count = 0;
      for (const auto& selected : out_.selected)
        if (const auto member = selected.call->route(); member && member->ids == route->ids)
          ++count;
      groups.push_back({.ids = route->ids, .calls = {}, .cut = out_.plan.steps.size()});
      group = groups.end() - 1;
      group->calls.reserve(count);
    }
    group->calls.push_back(index);
    const auto* marker = out_.selected[index].tensor;
    bool found = false;
    for (std::size_t step = 0; step != out_.plan.steps.size(); ++step) {
      const auto& operation = out_.plan.steps[step];
      if (!std::ranges::contains(operation.nodes, marker)) continue;
      if (found || operation.nodes.size() != 1 || operation.lane != 0)
        return Refused("routed product lacks one selected ordered operation");
      found = true;
      group->cut = std::min(group->cut, step);
      group->last = std::max(group->last, step);
    }
    if (!found) return Refused("routed product is absent from the selected plan");
  }
  std::ranges::sort(groups, {}, &Group::cut);
  for (std::size_t g = 1; g != groups.size(); ++g)
    if (groups[g - 1].last >= groups[g].cut)
      return Refused("routed group products interleave across continuation boundaries");
  for (const auto& r : out_.plan.regions) {
    if (r.first > r.last || r.last >= out_.plan.steps.size())
      return Refused("invalid region bounds in routed plan");
    for (const auto& g : groups) {
      if (r.first < g.cut && r.last >= g.cut)
        return Refused("lane region straddles routed continuation cut boundary");
      for (const auto call_idx : g.calls) {
        const auto* marker = out_.selected[call_idx].tensor;
        for (std::size_t s = r.first; s <= r.last; ++s) {
          if (std::ranges::contains(out_.plan.steps[s].nodes, marker))
            return Refused("lane region contains routed product operation");
        }
      }
    }
  }
  // Bind every primitive reader to its own real Route and effective lowered IDs.
  // A weight reused by several groups retains explicit membership in each.
  for (const auto& weight : in_.io.weights) {
    std::size_t readers = 0, selected_readers = 0;
    for (const auto& group : groups)
      for (const auto index : group.calls)
        if (out_.selected[index].call->route()->weight == weight.ref) ++selected_readers;
    if (selected_readers != 0) {
      for (const auto& node : g_.nodes())
        for (std::size_t i = 0; i != node.nsrc; ++i) {
          if (!node.src[i].valid() || g_.root(node.src[i]) != weight.tensor) continue;
          if (i != 0 || node.src[i] != weight.tensor || node.key != ir::Primitive(Op::kMatMulId) ||
              node.nsrc != 3)
            return Refused("dynamic expert weight has an undeclared ordinary or alias reader");
          const auto input = node.src[2];
          const auto& producer = g_.node(g_.producer(input));
          if (producer.key != ir::Primitive(Op::kRoute) || producer.out[0] != input)
            return Refused("dynamic experts lack their original router IDs output");
          const auto planned = plan_.tensors.at(input.v);
          if (!planned.valid()) return Refused("router IDs lost their planned mapping");
          auto group = std::ranges::find(groups, made_.at(planned.v), &Group::ids);
          if (group == groups.end() || (group->original.valid() && group->original != input))
            return Refused("router IDs changed their actual group identity");
          group->original = input;
          std::size_t matching = 0;
          for (const auto index : group->calls)
            if (out_.selected[index].call->route()->weight == weight.ref) ++matching;
          if (matching == 0) return Refused("routed reader omitted its group membership");
          ++readers;
        }
      if (readers != selected_readers)
        return Refused("dynamic expert reader inventory changed during planning");
      for (const auto& group : groups) {
        std::size_t actual_count = 0, selected_count = 0;
        for (const auto& node : g_.nodes()) {
          if (node.key != ir::Primitive(Op::kMatMulId) || node.nsrc != 3 ||
              node.src[0] != weight.tensor)
            continue;
          const auto planned = plan_.tensors.at(node.src[2].v);
          if (planned.valid() && made_.at(planned.v) == group.ids) ++actual_count;
        }
        for (const auto index : group.calls)
          if (out_.selected[index].call->route()->weight == weight.ref) ++selected_count;
        if (actual_count != selected_count)
          return Refused("one routed group's exact weight readers changed during planning");
      }
    }
    if (!out_.weight_reads) return Refused("routed weights lack actual source-part binding");
    const auto read = std::ranges::find(*out_.weight_reads, weight.ref, &RoutedWeight::ref);
    if (read == out_.weight_reads->end()) return Refused("routed weight inventory is incomplete");
    read->dynamic = selected_readers != 0;
  }
  for (const auto& group : groups) {
    if (!group.original.valid()) return Refused("routed group lacks primitive source authority");
    for (const auto& node : g_.nodes()) {
      if (node.key == ir::Primitive(Op::kCacheStore) && g_.root(node.src[0]) == group.original)
        return Refused("routed IDs are overwritten");
      for (std::size_t o = 0; o != node.nout; ++o)
        if (const auto alias = ir::AliasedSource(node.key, o);
            alias && node.src[*alias].valid() && g_.root(node.src[*alias]) == group.original)
          return Refused("routed IDs have an alias writer");
    }
  }
  std::uint64_t device = Round(out_.extent, 256);
  base::Sha256 identity;
  const auto graph = g_.Digest();
  identity.Update(std::as_bytes(std::span(graph)));
  identity.Update("ordered-routed-groups-v1");
  for (const auto& group : groups) {
    const auto first = out_.selected[group.calls.front()].call->route();
    auto* ids = group.ids;
    if (!ids || ids->type != GGML_TYPE_I32 || ids->ne[0] <= 0 || ids->ne[1] <= 0 ||
        ids->ne[2] != 1 || ids->ne[3] != 1 || ids->nb[0] != sizeof(std::int32_t) ||
        ids->nb[1] % sizeof(std::int32_t) != 0 ||
        ids->nb[1] / sizeof(std::int32_t) > static_cast<std::uint64_t>(INT32_MAX) ||
        static_cast<std::uint64_t>(ids->ne[0]) > ids->nb[1] / sizeof(std::int32_t))
      return Refused("routed IDs lack a complete pitched-row producer");
    const GroupedRouteShape shape{first->format.shape.ne[2], static_cast<std::uint64_t>(ids->ne[0]),
                                  static_cast<std::uint64_t>(ids->ne[1]), 1};
    const auto size = GroupedRouteBound(shape);
    if (!size) return Refused("routed worklist geometry exceeds its checked domain");
    LoweredRouting result;
    result.layout.shape = shape;
    result.graph_identity = graph;
    result.cut = group.cut;
    result.members.reserve(group.calls.size());
    result.descriptors.reserve(group.calls.size());
    result.uploads.reserve(3 + group.calls.size());
    result.layout.readback.reserve(5);
    identity.Update(std::format("{}:{}:{}:{}:{}:{};", group.original.v, group.cut, group.last,
                                shape.experts, shape.used, shape.tokens));
    for (const auto index : group.calls) {
      const auto route = out_.selected[index].call->route();
      if (route->format.shape.ne[2] != shape.experts)
        return Refused("one route group changes its expert count");
      const auto weight = std::ranges::find(*out_.weight_reads, route->weight, &RoutedWeight::ref);
      if (weight == out_.weight_reads->end()) return Refused("routed weight disappeared");
      const auto member = static_cast<std::size_t>(weight - out_.weight_reads->begin());
      if (!std::ranges::contains(result.members, member)) result.members.push_back(member);
      identity.Update(std::format("{}:{};", index, member));
    }
    const auto E = shape.experts, P = size->pairs;
    const moe::RouteLayout layout{static_cast<std::int64_t>(E), static_cast<std::int64_t>(P),
                                  static_cast<std::int64_t>((shape.tokens + 63) / 64)};
    std::uint64_t host = 0;
    // Every group retains the existing signed-index route bound. The aggregate
    // checked placement includes all groups and their actual selected calls.
    const auto reserve = [](std::uint64_t& cursor,
                            std::uint64_t bytes) -> std::optional<std::uint64_t> {
      if (bytes > (1ULL << 40) || cursor > (1ULL << 40) - Round(bytes, 256)) return {};
      const auto at = cursor;
      cursor += Round(bytes, 256);
      return at;
    };
    const auto base = in_.addresses.measuring() ? kDistinct : in_.addresses.activations;
    const auto route_at = reserve(device, static_cast<std::uint64_t>(layout.ints()) * 4);
    if (!route_at || base > UINT64_MAX - device) return Refused("routed workspace overflow");
    result.preparation = MoeRoute(c_, ids, static_cast<std::int64_t>(E));
    TensorArena::Bind(result.preparation, base + *route_at);
    if (auto checked = CheckMoeRoute(result.preparation); !checked) return checked;
    const auto read = [&](std::uint64_t from, std::uint64_t bytes, std::uint64_t& offset) -> bool {
      const auto at = reserve(host, bytes);
      if (!at) return false;
      offset = *at;
      result.layout.readback.push_back({from, *at, bytes});
      return true;
    };
    auto& r = result.layout;
    const auto prepared = base + *route_at;
    const auto id_address = reinterpret_cast<std::uintptr_t>(ids->data);
    const auto id_row = shape.used * sizeof(std::int32_t);
    std::uint64_t id_extent = 0;
    if (__builtin_mul_overflow(shape.tokens - 1, static_cast<std::uint64_t>(ids->nb[1]),
                               &id_extent) ||
        __builtin_add_overflow(id_extent, id_row, &id_extent) || id_extent > ggml_nbytes(ids) ||
        id_extent > UINT64_MAX - id_address)
      return Refused("routed IDs source extent overflow");
    if (id_address < base || id_address - base > out_.extent ||
        id_extent > out_.extent - (id_address - base))
      return Refused("actual routed IDs escaped the placed activation extent");
    if (!read(id_address, P * 4, r.ids)) return Refused("routed IDs readback overflow");
    if (shape.tokens > 1 && ids->nb[1] != id_row) {
      auto& copy = r.readback.back();
      copy.bytes = id_row;
      copy.rows = shape.tokens;
      copy.device_stride = ids->nb[1];
    }
    if (!read(prepared + static_cast<std::uint64_t>(layout.offsets()) * 4, (E + 1) * 4,
              r.offsets) ||
        !read(prepared + static_cast<std::uint64_t>(layout.position()) * 4, P * 4, r.position) ||
        !read(prepared + static_cast<std::uint64_t>(layout.token()) * 4, P * 4, r.token) ||
        !read(prepared + static_cast<std::uint64_t>(layout.expert()) * 4, P * 4, r.expert))
      return Refused("routed readback extent overflow");
    SelectedRouteBuffers buffers;
    const auto uploaded = [&](std::uint64_t bytes, std::uint64_t& h, std::uint64_t& d) -> bool {
      const auto at = reserve(host, bytes), to = reserve(device, bytes);
      if (!at || !to || base > UINT64_MAX - device) return false;
      h = *at;
      d = base + *to;
      result.uploads.push_back({d, h, bytes});
      return true;
    };
    if (!uploaded((E + 1) * 4, r.work_offsets, buffers.offsets) ||
        !uploaded(P * 4, r.work_tokens, buffers.tokens) ||
        !uploaded(P * 4, r.work_slots, buffers.slots))
      return Refused("routed worklist workspace overflow");
    const auto seen = reserve(host, P);
    if (!seen) return Refused("routed validation scratch overflow");
    r.seen = *seen;
    for (const auto index : group.calls) {
      auto& call = *out_.selected[index].call;
      std::uint64_t at = 0;
      if (!uploaded(call.route_descriptor_bytes(), at, buffers.descriptors))
        return Refused("routed descriptors overflow");
      if (auto bound = call.RouteBuffers(buffers); !bound) return bound;
      result.descriptors.emplace_back(index, at);
    }
    r.host_bytes = host;
    if (!in_.addresses.measuring() && device > in_.addresses.activation_bytes)
      return Refused("routed workspace exceeds the paid activation placement");
    result.layout.host_offset = out_.routed_pinned_bytes;
    if (__builtin_add_overflow(out_.routed_pinned_bytes, host, &out_.routed_pinned_bytes) ||
        out_.routed_pinned_bytes > (1ULL << 40))
      return Refused("aggregate routed pinned storage overflow");
    // Stable source-relative layout, not mutable process addresses. Measuring
    // and bound plans must preserve every group slice and physical copy pitch.
    identity.Update(std::format("{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{};", r.host_offset, r.host_bytes,
                                r.ids, r.offsets, r.position, r.token, r.expert, r.work_offsets,
                                r.work_tokens, r.work_slots, r.seen, ids->nb[1]));
    for (const auto& copy : r.readback) {
      identity.Update(std::format("r:{}:{}:{}:{}:{};", copy.device - base, copy.host_offset,
                                  copy.bytes, copy.rows, copy.device_stride));
    }
    for (const auto& copy : result.uploads)
      identity.Update(std::format("u:{}:{}:{};", copy.device - base, copy.host_offset, copy.bytes));
    for (const auto& [index, offset] : result.descriptors)
      identity.Update(std::format("d:{}:{};", index, offset));
    out_.routing.push_back(std::move(result));
  }
  const auto base = in_.addresses.measuring() ? kDistinct : in_.addresses.activations;
  out_.tensors.push_back({base + out_.extent, device - out_.extent, ir::Role::kActivation, false});
  out_.extent = device;
  out_.routing_identity = identity.Finish();
  return {};
}

std::expected<LoweredStep, KernelFailure> Lowering::Run() {
  if (!std::ranges::equal(in_.io.inputs, g_.inputs()) ||
      !std::ranges::equal(in_.io.outputs, g_.outputs()))
    return Refused("lowering needs the graph's exact typed ports");
  if (in_.token && &in_.token->step.graph != &g_)
    return Refused("token metadata belongs to another graph");
  if (!in_.token) {
    // First stateless physical domain. No token/state operation can enter a
    // branch that interprets a StepShape, synthesized inputs or state spans.
    const auto supported = [](const ir::Node& n) {
      // Closed explicit-operand forms only: the existing VAE shortcut and
      // learned RMS scaling. Both use the same validated primitive operands;
      // neither consults token metadata, state, side-input builders or saves.
      if (n.key == ir::Fused(FusedKind::kDupUpAdd) || n.key == ir::Fused(FusedKind::kRmsNormMul))
        return true;
      if (n.key != ir::Primitive(n.key.op)) return false;
      switch (n.key.op) {
        case Op::kInput:
        case Op::kWeight:
        case Op::kConv2d:
        case Op::kUpsample:
        case Op::kView:
        case Op::kReshape:
        case Op::kPermute:
        case Op::kCont:
        case Op::kRepeat:
        case Op::kConvert:
        case Op::kAdd:
        case Op::kMul:
        case Op::kScale:
        case Op::kUnary:
        case Op::kLayerNorm:
        case Op::kRmsNorm:
        case Op::kRope:
        case Op::kGlu:
        case Op::kConcat:
        case Op::kL2Norm:
        case Op::kMatMul:
        case Op::kGetRows:
        case Op::kSoftMax:
        case Op::kAttention:
          return true;
        default:
          return false;
      }
    };
    for (const ir::Graph* graph : {&g_, &p_}) {
      const auto bad = std::ranges::find_if_not(graph->nodes(), supported);
      if (bad != graph->nodes().end())
        return Refused(std::format("stateless GGML refuses {} in the {} graph",
                                   ir::OpKeyName(bad->key), graph == &g_ ? "source" : "planned"));
    }
  }
  if (auto checked = CheckSelectedHc(); !checked) return std::unexpected(checked.error());
  if (auto checked = CheckOwnerProducts(); !checked) return std::unexpected(checked.error());
  if (auto indexed = Index(); !indexed) return std::unexpected(indexed.error());
  if (auto checked = CheckQsaInputs(); !checked) return std::unexpected(checked.error());
  Needed();
  // Room for every planned tensor and the GGML nodes a fused node stands
  // for besides its results.
  std::size_t tensors = p_.tensor_count() + 16;
  for (const planner::Applied& a : plan_.applied) {
    for (const NodeId r : a.replaced) tensors += g_.node(r).nout;
  }
  // The composites' GGML sequences (MakeRoute's ten nodes, MakeReduce's
  // two products, a view and an add a slot, and the weights' rows); the
  // fast forms' kernels' views and intermediates (at most 8 a node), a
  // weight's parts and a verify's saves.
  for (const ir::Node& n : p_.nodes()) {
    if (n.key == ir::Primitive(Op::kRoute)) tensors += 10;
    // One completed-route descriptor per ordered group; the group count is
    // bounded by actual MatMulId products, before any arena allocation.
    if (n.key == ir::Primitive(Op::kMatMulId)) {
      if (tensors == SIZE_MAX) return Refused("ordered route descriptor count overflow");
      ++tensors;
    }
    // A fill's shape tensor and ggml_fill's duplicated result both need
    // descriptors, although the shape tensor is no execution dependency.
    if (n.key == ir::Primitive(Op::kFill)) ++tensors;
    // ggml.convert owns a destination root and returns its cpy view.
    if (n.key == ir::Primitive(Op::kConvert)) ++tensors;
    if (n.key == ir::Primitive(Op::kConv2d)) tensors += 2;
    if (n.key == ir::Primitive(Op::kExpertReduce) && n.nsrc >= 1) {
      tensors += 3 + (2 * p_.tensor(n.src[0]).ne(1));
    }
    tensors += n.key.op == Op::kFused || n.key.op == Op::kMatMul || n.key.op == Op::kWeight ||
                       n.key.op == Op::kGatedDeltaNet
                   ? 8
                   : 0;
  }
  if (in_.token) tensors += 4 * in_.token->step.saves.size();
  auto arena = TensorArena::Create(tensors);
  if (!arena) return std::unexpected(arena.error());
  out_.arena.emplace(std::move(*arena));
  if (auto room = out_.arena->Reserve(tensors); !room) return std::unexpected(room.error());
  c_ = out_.arena->context();
  // Planned nodes in order; a fused node's replaced primitive nodes each
  // at its own place in the primitive graph where its sources allow (the
  // builders' order: the narrowed final layer's residual gather between
  // the norm-add's norm and its add), else at the fused node's.
  for (std::uint32_t i = 0; i < p_.nodes().size(); ++i) {
    if (i < plan_.from.size()) {
      if (auto flushed = Flush(plan_.from[i]); !flushed) return std::unexpected(flushed.error());
    }
    if (auto made = MakeNode(NodeId{i}); !made) return std::unexpected(made.error());
  }
  if (auto saved = MakeSaves(); !saved) return std::unexpected(saved.error());
  if (auto bound = Bind(); !bound) return std::unexpected(bound.error());
  if (auto checked = BindSelectedHc(); !checked) return std::unexpected(checked.error());
  if (auto checked = BindOwnerProducts(); !checked) return std::unexpected(checked.error());
  for (const auto* node : bank_attentions_) {
    if (auto valid = CheckFlashAttnMma(node); !valid) return std::unexpected(valid.error());
  }
  if (auto compared = PlanAndCompare(); !compared) return std::unexpected(compared.error());
  // Capabilities describe shape/format metadata. The actual local fast
  // launches additionally prove their bound strides, addresses and aliasing.
  for (const auto& step : out_.plan.steps) {
    std::expected<void, KernelFailure> valid;
    if (step.implementation == kDsv4HcPostNormF16Name) {
      if (step.nodes.size() != 2) return Refused("an HC post/norm pair without both outputs");
      valid = CheckDsv4HcPostNormF16(step.nodes[0], step.nodes[1]);
    } else if (step.implementation == kDsv4HcPostExpertsNormF16Name) {
      if (step.nodes.size() != 4) return Refused("an HC post experts/norm without four nodes");
      valid =
          CheckDsv4HcPostExpertsNormF16(step.nodes[0], step.nodes[1], step.nodes[2], step.nodes[3]);
    } else if (step.nodes.size() == 1) {
      const auto* node = step.nodes[0];
      switch (LlmpOpOf(node)) {
        case LlmpOp::kDsv4LidTopK:
          valid = CheckDsv4LidTopK(node);
          break;
        case LlmpOp::kDsv4SparseMask:
          valid = CheckDsv4SparseMask(node);
          break;
        case LlmpOp::kRoundedIndexScore:
          valid = CheckRoundedIndexScore(node);
          break;
        case LlmpOp::kDsv4QHead:
          valid = CheckDsv4QHead(node);
          break;
        case LlmpOp::kDsv4Compress:
          valid = CheckDsv4Compress(node);
          break;
        case LlmpOp::kDsv4HcMix:
          valid = CheckDsv4HcMix(node);
          break;
        case LlmpOp::kDsv4HcPre:
          valid = CheckDsv4HcPre(node);
          break;
        case LlmpOp::kDsv4OutA:
          valid = CheckDsv4OutA(node);
          break;
        default:
          break;
      }
    }
    if (!valid) return std::unexpected(valid.error());
  }
  out_.extent = plan_.placement.extent;
  if (auto routed = MakeRouting(); !routed) return std::unexpected(routed.error());
  if (auto slots = Slots(); !slots) return std::unexpected(slots.error());
  out_.measuring = in_.addresses.measuring();
  return std::move(out_);
}

}  // namespace

namespace {

// The fast forms' kernels' result bytes and intermediates (laid in the
// result's place after it, each 256-byte aligned), built over stand-ins of
// the node's sources in a context that allocates nothing: what MakeDirect
// and MakeSpecial make.
std::uint64_t KernelBytes(const ir::Graph& graph, const ir::Node& n) {
  if (n.key == ir::Primitive(Op::kSsmConv)) {
    // The padded convolution's result, the copy and the padded input.
    const std::int64_t pad = SsmConvPad(graph, n);
    const ir::TensorDesc& x = graph.tensor(n.src[0]);
    const ir::TensorDesc& o = graph.tensor(n.out[0]);
    const std::uint64_t rows = x.ne(1) * x.ne(2) * sizeof(float);
    return Round((o.ne(1) + static_cast<std::uint64_t>(pad)) * o.ne(0) * o.ne(2) * sizeof(float),
                 256) +
           Round(static_cast<std::uint64_t>(pad) * rows, 256) +
           Round((x.ne(0) + static_cast<std::uint64_t>(pad)) * rows, 256);
  }
  const std::size_t room = ggml_tensor_overhead() * 32;
  ggml_init_params params{.mem_size = room, .mem_buffer = nullptr, .no_alloc = true};
  ggml_context* c = ggml_init(params);
  if (c == nullptr) return 0;
  const auto leaf = [&](ir::TensorId t, ggml_type type) {
    const ir::TensorDesc& d = graph.tensor(t);
    std::array<std::int64_t, 4> ne{};
    for (std::size_t i = 0; i < 4; ++i) ne.at(i) = static_cast<std::int64_t>(d.ne(i));
    return ggml_new_tensor(c, type, 4, ne.data());
  };
  std::uint64_t total = 0;
  const auto add = [&](const ggml_tensor* t) { total += Round(ggml_nbytes(t), 256); };
  if (n.key == ir::Fused(FusedKind::kMoeExperts)) {
    const auto p = n.Params<ir::MoeExpertsParams>();
    const ir::TensorDesc& xd = graph.tensor(n.src[2]);
    const ir::TensorDesc& idd = graph.tensor(n.src[3]);
    const auto width = static_cast<std::int64_t>(xd.ne(0));
    const auto nt = static_cast<std::int64_t>(xd.ne(1));
    const auto used = static_cast<std::int64_t>(idd.ne(0));
    const auto f = static_cast<std::int64_t>(p.hidden);
    const auto e = static_cast<std::int64_t>(graph.tensor(n.src[7]).ne(0));
    const moe::ExpertLayout layout{.ffn = static_cast<std::uint64_t>(f),
                                   .width = static_cast<std::uint64_t>(width)};
    ggml_tensor* experts = ggml_new_tensor_2d(c, GGML_TYPE_I8, 16, e);
    ggml_tensor* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, width, nt);
    ggml_tensor* ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, used, nt);
    ggml_tensor* scale = ggml_new_tensor_1d(c, GGML_TYPE_F32, e);
    ggml_tensor* weights = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, used, nt);
    ggml_tensor* shared = ggml_new_tensor_2d(c, GGML_TYPE_F32, width, nt);
    ggml_tensor* gate = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, nt);
    if (nt <= kMoeGemvWaveTokens) {
      ggml_tensor* x3 = ggml_reshape_3d(c, x, width, 1, nt);
      ggml_tensor* act = MoeGemvSwiglu(c, experts, x3, ids, f, scale, scale,
                                       moe::ExpertLayout::gate_up_codes(), layout.gate_up_scales());
      ggml_tensor* down =
          MoeGemv(c, experts, act, ids, width, 0, width, layout.down_codes(), layout.down_scales());
      add(MoeCombine(c, down, ids, scale, weights, shared, gate));
      add(act);
      add(down);
    } else {
      ggml_tensor* route = MoeRoute(c, ids, e);
      ggml_tensor* a1 = MoeQuantize(c, x, route);
      ggml_tensor* d1 = MoeGemm(c, a1, route, experts, 2 * f, moe::ExpertLayout::gate_up_codes(),
                                layout.gate_up_scales());
      ggml_tensor* a2 = MoeGluQuantize(c, d1, a1, route, scale, scale);
      ggml_tensor* d2 =
          MoeGemm(c, a2, route, experts, width, layout.down_codes(), layout.down_scales());
      add(MoeCombineSorted(c, d2, a2, route, scale, weights, shared, gate));
      for (ggml_tensor* t : {route, a1, d1, a2, d2}) add(t);
    }
  } else if (n.key == ir::Fused(FusedKind::kQsaAttention)) {
    const auto p = n.Params<ir::QsaAttnParams>();
    const ir::TensorDesc& q = graph.tensor(n.src[0]);
    add(leaf(n.out[0], GGML_TYPE_F32));
    total += Round(static_cast<std::uint64_t>(QsaTopKRow(p.width)) * q.ne(2) * sizeof(std::int32_t),
                   256);
  } else if (n.key == ir::Primitive(Op::kMatMul)) {
    const ir::TensorDesc& w = graph.tensor(n.src[0]);
    total = Round(planner::RootBytes(graph.tensor(n.out[0])).value_or(0), 256) +
            Round(mxfp8::SwizzledScaleBytes(w.ne(1), w.ne(0)), 256);
  } else if (n.key == ir::Primitive(Op::kGatedDeltaNet)) {
    ggml_tensor* r =
        ggml_gated_delta_net(c, leaf(n.src[0], GGML_TYPE_F32), leaf(n.src[1], GGML_TYPE_F32),
                             leaf(n.src[2], GGML_TYPE_F32), leaf(n.src[3], GGML_TYPE_F32),
                             leaf(n.src[4], GGML_TYPE_F32), leaf(n.src[5], GGML_TYPE_F32), 1);
    total = ggml_nbytes(r);
  }
  ggml_free(c);
  return total;
}

}  // namespace

ir::TensorId WithinRoot(const ir::Graph& graph, ir::TensorId root) {
  const ir::Node& n = graph.node(graph.producer(root));
  if (PackedProductPair(graph, n) && root == n.out[1]) return n.out[0];
  // A blob kernel's later outputs, in its first output's place.
  const bool blob =
      n.key == ir::Fused(FusedKind::kHcPrep) || n.key == ir::Fused(FusedKind::kHcFold) ||
      n.key == ir::Fused(FusedKind::kMoeRouter) || n.key == ir::Fused(FusedKind::kGdnGates) ||
      n.key == ir::Primitive(Op::kGatedDeltaNet) || SelectedRoutePlane(graph, n) != 0;
  if (blob && n.nout > 1 && root != n.out[0]) return n.out[0];
  // A preparation into MXFP8 rows of a gated norm's result alone (through
  // its reshape): llmp.gdn.norm_gate writes the rows in the norm's place.
  if (n.key == ir::Primitive(Op::kPrepare) && ir::PreparedRows(graph.tensor(root).format)) {
    const auto readers = [&](ir::TensorId t) {
      std::size_t count = 0;
      for (const ir::Node& r : graph.nodes()) {
        for (const ir::TensorId s : r.sources()) count += s == t ? 1 : 0;
      }
      return count + (std::ranges::contains(graph.outputs(), t) ? 1 : 0) +
             (std::ranges::contains(graph.keep(), t) ? 1 : 0);
    };
    ir::TensorId t = n.src[0];
    while (graph.node(graph.producer(t)).key == ir::Primitive(Op::kReshape)) {
      if (readers(t) != 1) return {};
      t = graph.node(graph.producer(t)).src[0];
    }
    const ir::Node& from = graph.node(graph.producer(t));
    if (from.key == ir::Fused(FusedKind::kNormGate) && readers(t) == 1) return from.out[0];
  }
  return {};
}

std::uint64_t WrittenBytes(const ir::Graph& graph, ir::TensorId root) {
  const ir::TensorDesc& t = graph.tensor(root);
  const ir::Node& made = graph.node(graph.producer(root));
  if (PackedProductPair(graph, made) && root == made.out[0]) {
    const auto own = planner::RootBytes(t);
    if (!own || *own > std::numeric_limits<std::uint64_t>::max() / 2)
      return std::numeric_limits<std::uint64_t>::max();
    return *own * 2;
  }
  // The fast forms' blobs and kernels with intermediates (lowered by
  // MakeDirect and MakeSpecial), at their first output.
  if (made.nout > 0 && made.out[0] == root) {
    // The positions a block pool reads hold its marker after them
    // (MakeDirect's llmp.qsa.pool).
    if (made.key.op == Op::kInput) {
      for (const ir::Node& r : graph.nodes()) {
        if (r.key == ir::Fused(FusedKind::kQsaPool) && r.nsrc > 5 && graph.root(r.src[5]) == root) {
          return Round(planner::RootBytes(t).value_or(0), 256) + 256;
        }
      }
    }
    const auto rows = [&](std::size_t i) { return graph.tensor(made.src.at(i)); };
    if (made.key == ir::Fused(FusedKind::kGroupedMatMul)) {
      const auto x = rows(1);
      const auto output = planner::RootBytes(t);
      // Validate before the padding/product arithmetic, including direct IR
      // callers whose native capability has not yet been selected.
      if (!output || *output > UINT64_MAX - 255 || x.ne(0) == 0 || x.ne(0) > INT32_MAX - 511 ||
          x.ne(0) % 32 != 0 || x.ne(1) < 2 || x.ne(1) > 256 || x.ne(2) == 0 ||
          x.ne(2) > static_cast<std::uint64_t>(kVecQMaxTokens) || x.ne(3) != 1)
        return UINT64_MAX;
      const auto prepared =
          Q8Bytes(static_cast<std::int64_t>(x.ne(0)), static_cast<std::int64_t>(x.ne(1) * x.ne(2)));
      const auto offset = Round(*output, 256);
      if (prepared <= 0 || static_cast<std::uint64_t>(prepared) > UINT64_MAX - offset)
        return UINT64_MAX;
      return offset + static_cast<std::uint64_t>(prepared);
    }
    if (made.key == ir::Fused(FusedKind::kDsv4LidMask)) {
      const auto p = made.Params<ir::LidMaskParams>();
      return Round(planner::RootBytes(t).value_or(0), 256) +
             Round(std::uint64_t{p.top} * t.ne(1) * sizeof(std::int32_t), 256);
    }
    if (made.key == ir::Fused(FusedKind::kDsv4HcPre)) {
      const auto x = rows(0);
      // First output's root: normalized values and pitched carry tails,
      // followed by the mix partials. Packed carry copies have own roots.
      return Round((x.ne(0) + kDsv4HcTail) * x.ne(2) * sizeof(float), 256) +
             Round(25 * kDsv4HcChunks * x.ne(2) * sizeof(float), 256);
    }
    if (made.key == ir::Fused(FusedKind::kHcPrep)) {
      const ir::TensorDesc& x = rows(0);
      return HcPrepLayout{.width = static_cast<std::int64_t>(x.ne(0)),
                          .hc = static_cast<std::int64_t>(x.ne(1)),
                          .t = static_cast<std::int64_t>(x.ne(2)),
                          .combine = made.nsrc > 3 && made.src[3].valid(),
                          .inject = made.nsrc > 2 && made.src[2].valid()}
          .bytes();
    }
    if (made.key == ir::Fused(FusedKind::kHcFold)) {
      const auto p = made.Params<ir::HcFoldParams>();
      if (!p.prepared) return 0;
      return HcMixLayout{.width = static_cast<std::int64_t>(t.ne(0)),
                         .t = static_cast<std::int64_t>(t.ne(1)),
                         .bf16 = p.rounded}
          .bytes();
    }
    if (made.key == ir::Fused(FusedKind::kMoeRouter)) {
      const auto p = made.Params<ir::RouteParams>();
      return static_cast<std::uint64_t>(
                 MoeRouterLayout{.used = p.used, .t = static_cast<std::int64_t>(t.ne(1))}.ints()) *
             sizeof(std::int32_t);
    }
    if (made.key == ir::Fused(FusedKind::kGdnGates)) {
      return rows(0).ne(0) * rows(0).ne(1) * 2 * sizeof(float);
    }
    if (made.key == ir::Fused(FusedKind::kDsv4OutA)) {
      return 8192 * static_cast<std::uint64_t>(Dsv4OutARows(static_cast<std::int64_t>(t.ne(1)))) *
             sizeof(float);
    }
    if ((made.key == ir::Primitive(Op::kSsmConv) && SsmConvPad(graph, made) != 0) ||
        made.key == ir::Fused(FusedKind::kMoeExperts) ||
        made.key == ir::Fused(FusedKind::kQsaAttention) ||
        made.key == ir::Primitive(Op::kGatedDeltaNet) ||
        (made.key == ir::Primitive(Op::kMatMul) &&
         graph.tensor(made.src[0]).format.encoding == Encoding::kMxfp8 &&
         ir::PreparedRows(graph.tensor(made.src[1]).format))) {
      return KernelBytes(graph, made);
    }
  }
  // The fused routing's ids are its whole sort's first slots (MakeRoute):
  // the sort's bytes, a row of every expert.
  if (made.key == ir::Primitive(Op::kRoute) && made.nout == 2 && made.out[0] == root) {
    if (made.Params<ir::RouteParams>().kind == ir::RouteKind::kHash ||
        made.Params<ir::RouteParams>().kind == ir::RouteKind::kSqrtSoftplusBias)
      return 2 * SelectedRoutePlane(graph, made);
    return made.Params<ir::RouteParams>().experts * t.ne(1) * sizeof(std::int32_t);
  }
  // The fused scaled reduce's products take its place (MakeReduce).
  if (made.key == ir::Primitive(Op::kExpertReduce) && made.nsrc >= 1) {
    const ir::TensorDesc& x = graph.tensor(made.src[0]);
    return x.ne(0) * x.ne(1) * x.ne(2) * sizeof(float);
  }
  // MXFP8 rows as llmp.mxfp8.quantize (or a fusion ending in it) writes
  // them: the scales at a 256-byte boundary past the codes.
  if (ir::PreparedRows(t.format)) {
    return mxfp8::RowsLayout{.k = t.ne(0), .rows = t.ne(1)}.bytes();
  }
  if (made.key != ir::Primitive(Op::kPrepare) || t.format.encoding != Encoding::kQ8_1 ||
      t.ne(3) != 1) {
    return 0;
  }
  const std::int64_t bytes =
      Q8Bytes(static_cast<std::int64_t>(t.ne(0)), static_cast<std::int64_t>(t.ne(1) * t.ne(2)));
  return bytes > 0 ? static_cast<std::uint64_t>(bytes) : 0;
}

std::expected<LoweredStep, KernelFailure> LowerStep(const LowerInput& input,
                                                    const execution::Registry& registry,
                                                    const DeviceChoices& device,
                                                    const LowerOptions& options) {
  Lowering lowering(input, registry, device, options);
  return lowering.Run();
}

}  // namespace llmp::kernels::ggml
