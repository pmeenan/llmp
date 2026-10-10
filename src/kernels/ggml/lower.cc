// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include "kernels/ggml/lower.h"

#include <algorithm>
#include <array>
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
#include <vector>

#include "components/step.h"
#include "execution/registry.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ggml.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/ggml/fattn_owner.h"
#include "kernels/ggml/gemma_moe_fusion.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/llmp_ops.h"
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

// Placeholder addresses, as PlaceAndPlan's first pass gives (planned.h):
// activations of a measurement from kDistinct, leaves without an address
// (a test's) from kLeaves, each its own range.
constexpr std::uint64_t kDistinct = std::uint64_t{1} << 46U;
constexpr std::uint64_t kLeaves = kDistinct - (std::uint64_t{1} << 40U);

std::uint64_t Round(std::uint64_t bytes, std::uint64_t alignment) {
  return (bytes + alignment - 1) / alignment * alignment;
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
        g_(in.step.graph),
        registry_(registry),
        device_(device),
        options_(options) {}

  std::expected<LoweredStep, KernelFailure> Run();

 private:
  // A source of a node being lowered: a GGML tensor already made.
  using Resolve = std::function<ggml_tensor*(TensorId)>;

  std::expected<void, KernelFailure> Index();
  std::expected<void, KernelFailure> MakeNode(NodeId id);
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
  // GGML's builder call for primitive node `n` of `graph` over its
  // resolved sources (null for what it makes no tensor of: a mask's view).
  std::expected<ggml_tensor*, KernelFailure> Make(const ir::Graph& graph, const ir::Node& n,
                                                  const Resolve& resolve);
  std::expected<ggml_tensor*, KernelFailure> Mask(const ir::Graph& graph, const ir::Node& n,
                                                  const Resolve& resolve);
  std::expected<void, KernelFailure> Matches(const ir::TensorDesc& want, const ggml_tensor* got,
                                             std::string_view what) const;
  std::expected<void, KernelFailure> Bind();
  std::expected<void, KernelFailure> PlanAndCompare();
  std::expected<void, KernelFailure> Slots();
  // `t` in the graph, standing for planned node `planned`; a computed node
  // among those it launches, and with `step` any node (a fused window's
  // views and reshapes, which PlanGraph's step lists too).
  void Node(ggml_tensor* t, NodeId planned, bool step = false);

  // Whether every reader of planned tensor `t` is a mask the lowering makes
  // without it (the producer reads the whole positions input; a host mask
  // reads nothing on the device).
  bool MaskOnly(TensorId t) const;

  const LowerInput& in_;
  const planner::StepPlan& plan_;
  const ir::Graph& p_;  // the planned graph
  const ir::Graph& g_;  // the primitive graph
  const execution::Registry& registry_;
  const DeviceChoices& device_;
  const LowerOptions& options_;
  ggml_context* c_ = nullptr;

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
  // Host masks and staged inputs: the slot each takes.
  std::vector<std::pair<ggml_tensor*, InputSlot>> input_leaves_;
  // A fused implementation's intermediates, which it never writes, and the
  // result whose place they take.
  std::vector<std::pair<ggml_tensor*, ggml_tensor*>> elided_;
  // A planned tensor bound through another GGML tensor: a view's root (the
  // fused routing's ids, a view of its whole sort).
  std::map<std::uint32_t, ggml_tensor*> placed_root_;
  std::set<const ggml_tensor*> state_tensors_, weight_tensors_;
};

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
  for (const co::StepInput& input : in_.step.inputs) inputs_[input.tensor.v] = &input;
  for (const co::StepWeight& w : in_.step.weights) weights_[w.tensor.v] = &w;
  for (const co::StepState& s : in_.step.state) state_[s.tensor.v] = &s;
  for (std::size_t a = 0; a < plan_.applied.size(); ++a) {
    for (const NodeId made : plan_.applied[a].made) applied_[made.v] = a;
  }
  // The GGML nodes a fused node runs over, in the primitive graph's order
  // (Run): a fused kind GGML implements over its primitive nodes, from a
  // rewrite that made only it.
  for (std::uint32_t i = 0; i < p_.nodes().size(); ++i) {
    const ir::Node& n = p_.nodes()[i];
    if (n.key.op != Op::kFused || n.key.fused == FusedKind::kOwnerAttention) continue;
    const auto at = applied_.find(i);
    if (at == applied_.end())
      return Refused(std::format("{} no rewrite made", ir::OpKeyName(n.key)));
    const planner::Applied& a = plan_.applied[at->second];
    if (a.made.size() != 1) return Refused("a fused node of a rewrite that made several");
    for (const NodeId r : a.replaced)
      replaced_.push_back({.primitive = r, .fused = NodeId{i}, .made = false});
  }
  std::ranges::sort(replaced_, {}, &Replaced::primitive);
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
  bool same = type.has_value() && got->type == *type;
  for (std::size_t d = 0; d < 4 && same; ++d) {
    same = static_cast<std::uint64_t>(got->ne[d]) == want.ne(d);
    // Strides of views and of whatever has them (a root's are natural).
    if (same && want.nb.at(0) != 0) same = static_cast<std::uint64_t>(got->nb[d]) == want.nb.at(d);
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
  const auto type = TypeOf(d.format.encoding);
  if (!prim.valid() || !type) return Refused("a leaf of no primitive tensor or GGML type");
  const bool blocks = d.format.layout == format::Layout::kGgmlBlocks;
  if (d.format.layout != format::Layout::kDense && !blocks) {
    return Refused(std::format("a leaf in layout {}", format::Info(d.format.layout).name));
  }
  std::array<std::int64_t, 4> ne{};
  for (std::size_t i = 0; i < 4; ++i) ne.at(i) = static_cast<std::int64_t>(d.ne(i));
  const int rank = std::max<int>(1, d.shape().rank);
  if (n.key.op == Op::kInput) {
    const auto at = inputs_.find(prim.v);
    if (at == inputs_.end()) return Refused("an input the step does not list");
    const co::StepInput& input = *at->second;
    InputSlot slot{.tensor = prim, .segment = input.segment};
    switch (input.kind) {
      case co::InputKind::kTokens:
        slot.kind = InputKind::kTokens;
        break;
      case co::InputKind::kPositions:
        if (input.sections > 1) return Refused("multi-section positions (slice G3b)");
        slot.kind = InputKind::kPositions;
        break;
      case co::InputKind::kOutputRows:
        slot.kind = InputKind::kOutputRows;
        break;
      case co::InputKind::kStoreCells:
        // Fixed, per-row and block cells (Qwen3.8's state, its drafter's
        // streams) are not lowered yet (slice G3b).
        if (input.cells.fixed || input.cells.per_row || input.cells.block != 0) {
          return Refused("store cells of fixed, per-row or block state");
        }
        slot.kind = InputKind::kStoreCells;
        slot.cells = {.capacity = input.cells.capacity, .ring = input.cells.ring};
        break;
      case co::InputKind::kCellPositions:
        // Read only by masks, which derive (device) or fold in (host) them.
        if (input.cells.block != 0) return Refused("a block tensor's positions");
        if (!MaskOnly(out)) return Refused("cells' positions read by more than a mask");
        return nullptr;
      case co::InputKind::kStream:
      case co::InputKind::kResidual:
      case co::InputKind::kCarry:
      case co::InputKind::kSide:
        return Refused("a lone component's input");
      case co::InputKind::kSideRows:
      case co::InputKind::kBlockPositions:
      case co::InputKind::kCommitRows:
        // Qwen3.8's inputs, lowered with its patterns (slice G3b).
        return Refused("a side table's rows, blocks' positions or a verify's saves");
      case co::InputKind::kPositionPhases:
      case co::InputKind::kKeptRows:
      case co::InputKind::kKeptCells:
      case co::InputKind::kPoolReads:
      case co::InputKind::kBlockCells:
      case co::InputKind::kBlockStarts:
      case co::InputKind::kHadamard:
        // DeepSeek V4's compressors' and indexer's inputs, lowered with its
        // patterns (slice G4b).
        return Refused(
            "a compressor's phases, kept rows, pooled reads or blocks, or the indexer's "
            "Hadamard matrix (slice G4b)");
      case co::InputKind::kFeatures:
        // A drafter's rows' features: copied in on the device by the
        // engine, from where the step's work names them.
        slot.kind = InputKind::kFeatures;
        break;
      case co::InputKind::kOutputCells: {
        // A target's tap rows' cells: its segment's last rows' store cells.
        const auto outputs = static_cast<std::uint32_t>(d.ne(0));
        if (input.cells.fixed || input.cells.per_row || input.cells.block != 0) {
          return Refused("a tap's cells of fixed, per-row or block state (slice G3b)");
        }
        if (input.segment >= in_.shape.segments.size() || outputs == 0 ||
            outputs != in_.shape.segments[input.segment].outputs) {
          return Refused("a tap's cells of no segment's outputs");
        }
        slot.kind = InputKind::kStoreCells;
        slot.cells = {.capacity = input.cells.capacity, .ring = input.cells.ring};
        slot.outputs = outputs;
        break;
      }
    }
    ggml_tensor* t = ggml_new_tensor(c_, *type, rank, ne.data());
    input_leaves_.emplace_back(t, slot);
    out_.leaves.push_back({.tensor = t, .primitive = prim});
    return t;
  }
  ggml_tensor* t = ggml_new_tensor(c_, *type, rank, ne.data());
  if (n.key.op == Op::kWeight) {
    const auto at = weights_.find(prim.v);
    if (at == weights_.end()) return Refused("a weight the step does not list");
    if (at->second->parts.size() != 1) return Refused("a weight stored in several parts");
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

std::expected<ggml_tensor*, KernelFailure> Lowering::Mask(const ir::Graph& graph, const ir::Node& n,
                                                          const Resolve& resolve) {
  const auto p = n.Params<ir::MaskParams>();
  if (p.kind != ir::MaskKind::kCausal && p.kind != ir::MaskKind::kWindow) {
    // A block drafter's mask (kAhead: DSpark's, GGML's MaskPolicy::kBlock
    // producer) is slice G4b's.
    return Refused(p.kind == ir::MaskKind::kAhead ? "a block drafter's mask (slice G4b)"
                                                  : "a block mask");
  }
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
  const std::uint32_t window = p.kind == ir::MaskKind::kWindow ? p.window : 0U;
  if (!options_.device_masks) {
    if (out.format.encoding != Encoding::kF16) return Refused("a host mask not in F16");
    if (frozen && !options_.frozen_host_masks) {
      return Refused("a frozen read's host mask without a stager that builds it");
    }
    std::array<std::int64_t, 4> ne{static_cast<std::int64_t>(out.ne(0)),
                                   static_cast<std::int64_t>(out.ne(1)), 1, 1};
    ggml_tensor* t = ggml_new_tensor(c_, GGML_TYPE_F16, 2, ne.data());
    const InputSlot slot{
        .tensor = &graph == &p_ ? primitive_.at(n.out[0].v) : n.out[0],
        .segment = cells.segment,
        .kind = InputKind::kMask,
        .cells = {.capacity = cells.cells.capacity, .ring = cells.cells.ring, .frozen = frozen},
        .read = out.ne(0),
        .window = window,
        .mask_rows = static_cast<std::uint32_t>(out.ne(1))};
    input_leaves_.emplace_back(t, slot);
    out_.leaves.push_back({.tensor = t, .primitive = slot.tensor});
    return t;
  }
  // The producer reads the whole positions input from the segment's first
  // row (the IR's mask reads a view of it).
  if (!rows.is_view() || rows.ne(1) != 1 || rows.view_offset % sizeof(std::int32_t) != 0) {
    return Refused("a mask's rows are not a view of the positions");
  }
  ggml_tensor* positions = resolve(rows.view_of);
  if (positions == nullptr) return Refused("a mask's positions were not made");
  const auto to_int = [](std::uint64_t v) { return static_cast<std::int32_t>(v); };
  if (frozen) {
    return CausalRingMask(c_, positions, static_cast<std::int64_t>(out.ne(0)),
                          to_int(rows.view_offset / sizeof(std::int32_t)), to_int(rows.ne(0)),
                          to_int(cells.cells.capacity), to_int(window), to_int(in_.context),
                          CausalMaskRows::kPad32, GGML_TYPE_F16, MaskPolicy::kFrozen);
  }
  return Gemma4Mask(c_, positions, static_cast<std::int64_t>(out.ne(0)),
                    to_int(rows.view_offset / sizeof(std::int32_t)), to_int(rows.ne(0)),
                    to_int(cells.cells.capacity), to_int(window), to_int(in_.context));
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
      return ggml_view_4d(c_, s[0], ne(0), ne(1), ne(2), ne(3), out.nb.at(1), out.nb.at(2),
                          out.nb.at(3), out.view_offset - base);
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
      if (p.table || p.inverse || p.offset != 0 ||
          (p.mode != ir::RopeMode::kNeox && p.mode != ir::RopeMode::kNormal)) {
        return Refused("a rotation other than NORMAL or NEOX over positions");
      }
      return ggml_rope_ext(c_, s[0], s[1], s[2], static_cast<int>(p.dims),
                           p.mode == ir::RopeMode::kNeox ? GGML_ROPE_TYPE_NEOX : 0,
                           static_cast<int>(p.original_context), p.base, p.scale, p.ext_factor,
                           p.attn_factor, p.beta_fast, p.beta_slow);
    }
    case Op::kCacheStore:
      // GGML lists the rows, the indices, then the cache.
      return ggml_set_rows(c_, s[0], s[1], s[2]);
    case Op::kAttention: {
      const auto p = n.Params<ir::AttnParams>();
      if (p.sparse_any) return Refused("a sparse gather at any length");
      ggml_tensor* t =
          ggml_flash_attn_ext(c_, s[0], s[1], s[2], s[3], p.scale, p.max_bias, p.softcap);
      if (p.accumulate == ir::Accumulate::kF32) ggml_prec_set_acc(t, GGML_PREC_F32);
      if (s[4] != nullptr) ggml_flash_attn_ext_add_sinks(t, s[4]);
      if (p.n_kv_max != 0)
        ggml_flash_attn_ext_set_n_kv_max(t, static_cast<std::int32_t>(p.n_kv_max));
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
      const bool ready = std::ranges::all_of(
          prim.sources(), [&](TensorId s) { return !s.valid() || ResolveReplaced(s) != nullptr; });
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
  for (const auto& [t, prim] : made) {
    if (!plan_.tensors.at(prim.v).valid() && t->view_src == nullptr) {
      elided_.emplace_back(t, result);
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

std::expected<void, KernelFailure> Lowering::MakeNode(NodeId id) {
  const ir::Node& n = p_.node(id);
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
      return MakeFused(id, n);
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
    if (g == nullptr) return Refused("a placed tensor the lowering did not make");
    if (g->view_src != nullptr) return Refused("a placed tensor is a view");
    // What the placement made room for: the tensor's own bytes, or what the
    // module writes (WrittenBytes), which the plan was made with.
    const auto own = planner::RootBytes(p_.tensor(t));
    if (!own || ggml_nbytes(g) > std::max(*own, WrittenBytes(p_, t))) {
      return Refused(std::format("{} writes {} bytes, more than its place holds", ggml_op_desc(g),
                                 ggml_nbytes(g)));
    }
    TensorArena::Bind(g, base + offset);
  }
  std::uint64_t leaf = kLeaves;
  const auto placeholder = [&](ggml_tensor* t) {
    const std::uint64_t at = leaf;
    leaf += Round(ggml_nbytes(t), 256) + 256;
    return at;
  };
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
    TensorArena::Bind(t, address != 0 ? address : placeholder(t));
  }
  for (const auto& [t, slot] : input_leaves_) {
    if (t->data == nullptr) return Refused("an input the plan did not place");
  }
  for (const auto& [t, result] : elided_) {
    const ggml_tensor* root = result;
    while (root->view_src != nullptr) root = root->view_src;
    if (state_tensors_.contains(root) || weight_tensors_.contains(root) || root->data == nullptr) {
      return Refused("a fused node's intermediates have no activation to stand in");
    }
    t->data = result->data;
  }
  BindViews(out_.nodes);
  for (const ggml_tensor* t : out_.nodes) {
    if (t->data == nullptr) return Refused(std::format("{} is not bound", ggml_op_desc(t)));
  }
  return {};
}

std::expected<void, KernelFailure> Lowering::PlanAndCompare() {
  // The device's choices, with the policies the selection takes.
  DeviceChoices choices = device_;
  for (const planner::Selected& s : plan_.steps) {
    const std::string_view name = registry_.at(s.implementation).name;
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
  for (const TensorId t : p_.outputs()) out_.keep.push_back(made_.at(t.v));
  for (const TensorId t : p_.keep()) out_.keep.push_back(made_.at(t.v));
  if (std::ranges::contains(out_.keep, nullptr)) return Refused("an output was not made");
  auto planned =
      PlanGraph(out_.nodes, /*fusion=*/false, choices, out_.keep, options_.index_scratch_limit);
  if (!planned) return std::unexpected(planned.error());
  GraphPlan& gp = *planned;
  // Each node by its place in the graph (a refusal's message).
  std::map<const ggml_tensor*, std::size_t> place;
  for (std::size_t i = 0; i < out_.nodes.size(); ++i) place.emplace(out_.nodes[i], i);
  const auto describe = [&place](std::span<ggml_tensor* const> nodes) {
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
  for (std::size_t i = 0; i < plan_.steps.size(); ++i) {
    const planner::Selected& s = plan_.steps[i];
    std::vector<ggml_tensor*> want;
    for (const NodeId id : s.nodes) {
      const auto& l = launched_.at(id.v);
      want.insert(want.end(), l.begin(), l.end());
    }
    if (want.empty()) {
      // A host-built mask, staged before the run: placed as the IR placed
      // its step, which holds only where nothing computed before it (the
      // masks come first, BuildStep).
      if (j != 0) return Refused("a host-built mask after the step's computed nodes");
      continue;
    }
    const std::string_view name = registry_.at(s.implementation).name;
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
        ++i;
        ++j;
        continue;
      }
    }
    bool same = got.nodes == want && got.implementation == name;
    if (!same && got.nodes == want && !s.group.empty()) {
      // A member of the selected group, upstream's choice inside it.
      const auto member = registry_.Find(got.implementation);
      same = member && registry_.at(*member).capability &&
             registry_.at(*member).capability->group == s.group;
    }
    if (!same) {
      return Refused(std::format(
          "PlanGraph's step {} is {} over{}, where the IR selected {}{} over{}", j,
          got.implementation, describe(got.nodes), name,
          s.group.empty() ? std::string{} : std::format(" (group {})", s.group), describe(want)));
    }
    got.lane = s.lane;
    ++j;
  }
  if (j != gp.steps.size()) {
    return Refused(std::format("PlanGraph has {} steps beyond the IR's: {} over{}",
                               gp.steps.size() - j, gp.steps[j].implementation,
                               describe(gp.steps[j].nodes)));
  }
  // The IR's lanes over the same steps (host masks, which launch nothing
  // here, are outside every region: the step lists only their own).
  for (const planner::Region& r : plan_.regions) {
    gp.regions.push_back({.first = r.first, .last = r.last});
  }
  if (!gp.regions.empty() && j != plan_.steps.size()) {
    return Refused("concurrent lanes with host-built masks or unfused norms");
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
  // Each owner's own outputs: its token, or its rows of the logits.
  const planner::StepShape& shape = in_.shape;
  const bool greedy = in_.step.greedy.valid();
  const TensorId result = greedy ? in_.step.greedy : in_.step.logits;
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
      out_.outputs.push_back(
          {.device_address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t->data)) +
                             (first * row),
           .bytes = n * row,
           .tensor = result,
           .segment = static_cast<std::uint8_t>(i)});
      first += n;
    }
  }
  // A drafter head's next features, each owner's rows after its logits.
  if (in_.step.features.valid() && shape.outputs() != 0) {
    const TensorId planned = plan_.tensors.at(in_.step.features.v);
    const ggml_tensor* t = planned.valid() ? made_.at(planned.v) : nullptr;
    if (t == nullptr || t->data == nullptr || !ggml_is_contiguous(t) ||
        static_cast<std::uint64_t>(t->ne[1]) != shape.outputs()) {
      return Refused("the step's features were not made whole");
    }
    std::uint64_t first = 0;
    for (std::size_t i = 0; i < shape.segments.size(); ++i) {
      const std::uint32_t n = shape.segments[i].outputs;
      if (n == 0) continue;
      out_.outputs.push_back(
          {.device_address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t->data)) +
                             (first * t->nb[1]),
           .bytes = n * t->nb[1],
           .tensor = in_.step.features,
           .segment = static_cast<std::uint8_t>(i),
           .features = true});
      first += n;
    }
  }
  // Every tensor the steps touch, by the role of its storage.
  std::set<std::tuple<std::uint64_t, std::uint64_t, ir::Role>> seen;
  const auto use = [&](const ggml_tensor* t) {
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

std::expected<LoweredStep, KernelFailure> Lowering::Run() {
  if (auto indexed = Index(); !indexed) return std::unexpected(indexed.error());
  // Room for every planned tensor and the GGML nodes a fused node stands
  // for besides its results.
  std::size_t tensors = p_.tensor_count() + 16;
  for (const planner::Applied& a : plan_.applied) {
    for (const NodeId r : a.replaced) tensors += g_.node(r).nout;
  }
  // The composites' GGML sequences (MakeRoute's ten nodes, MakeReduce's
  // two products, a view and an add a slot, and the weights' rows).
  for (const ir::Node& n : p_.nodes()) {
    if (n.key == ir::Primitive(Op::kRoute)) tensors += 10;
    if (n.key == ir::Primitive(Op::kExpertReduce) && n.nsrc >= 1) {
      tensors += 3 + (2 * p_.tensor(n.src[0]).ne(1));
    }
  }
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
  if (auto bound = Bind(); !bound) return std::unexpected(bound.error());
  if (auto compared = PlanAndCompare(); !compared) return std::unexpected(compared.error());
  if (auto slots = Slots(); !slots) return std::unexpected(slots.error());
  out_.extent = plan_.placement.extent;
  out_.measuring = in_.addresses.measuring();
  return std::move(out_);
}

}  // namespace

std::uint64_t WrittenBytes(const ir::Graph& graph, ir::TensorId root) {
  const ir::TensorDesc& t = graph.tensor(root);
  const ir::Node& made = graph.node(graph.producer(root));
  // The fused routing's ids are its whole sort's first slots (MakeRoute):
  // the sort's bytes, a row of every expert.
  if (made.key == ir::Primitive(Op::kRoute) && made.nout == 2 && made.out[0] == root) {
    return made.Params<ir::RouteParams>().experts * t.ne(1) * sizeof(std::int32_t);
  }
  // The fused scaled reduce's products take its place (MakeReduce).
  if (made.key == ir::Primitive(Op::kExpertReduce) && made.nsrc >= 1) {
    const ir::TensorDesc& x = graph.tensor(made.src[0]);
    return x.ne(0) * x.ne(1) * x.ne(2) * sizeof(float);
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
