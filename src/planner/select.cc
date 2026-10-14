// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/select.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "execution/registry.h"
#include "format/tensor_format.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "planner/pattern.h"
#include "planner/policy.h"

namespace llmp::planner {

struct OwnedDeviceFacts::Storage {
  // Heap-stable Storage and a fully populated name vector precede the views.
  // Moving OwnedDeviceFacts only moves this allocation's pointer.
  std::vector<std::string> names;
  DeviceFacts facts;
};

std::expected<std::uint64_t, std::string> OwnedDeviceFacts::Bound(const DeviceFacts& facts) {
  if (facts.device_class.empty()) return std::unexpected("device facts have no class");
  std::uint64_t bytes = sizeof(Storage) + sizeof(OwnedDeviceFacts);
  const auto add = [&](std::uint64_t n) { return !__builtin_add_overflow(bytes, n, &bytes); };
  // Account vector elements, all string payloads including terminators, and
  // conservative per-allocation bookkeeping. No temporary payload copy.
  constexpr std::uint64_t overhead = 64;
  std::uint64_t entries = 0;
  if (__builtin_mul_overflow(static_cast<std::uint64_t>(facts.tuned.size()),
                             sizeof(execution::TunedShape) + sizeof(std::string) + overhead,
                             &entries) ||
      !add(entries) || !add(facts.device_class.size()) || !add(1 + 4 * overhead))
    return std::unexpected("owned device facts byte bound overflows");
  for (std::size_t i = 0; i < facts.tuned.size(); ++i) {
    const auto& tune = facts.tuned[i];
    if (tune.implementation.empty() || tune.rows == 0 || tune.k == 0 || tune.n == 0 ||
        (tune.output_encoding &&
         std::ranges::find(format::Encodings(), *tune.output_encoding, &format::EncodingInfo::id) ==
             format::Encodings().end()))
      return std::unexpected("owned device facts have an invalid tuning key");
    for (std::size_t j = 0; j < i; ++j) {
      const auto& before = facts.tuned[j];
      if (before.implementation == tune.implementation && before.rows == tune.rows &&
          before.k == tune.k && before.n == tune.n &&
          before.output_encoding == tune.output_encoding)
        return std::unexpected("owned device facts repeat a tuning key");
    }
    if (!add(tune.implementation.size()) || !add(1))
      return std::unexpected("owned tuning name byte bound overflows");
  }
  return bytes;
}

std::expected<OwnedDeviceFacts, std::string> OwnedDeviceFacts::Create(const DeviceFacts& facts,
                                                                      std::uint64_t paid_bytes) {
  auto bound = Bound(facts);
  if (!bound) return std::unexpected(bound.error());
  if (*bound > paid_bytes) return std::unexpected("owned device facts were not funded");
  auto storage = std::make_unique<Storage>();
  storage->names.reserve(facts.tuned.size());
  for (const auto& tune : facts.tuned) storage->names.emplace_back(tune.implementation);
  storage->facts = {
      .device_class = facts.device_class, .arch = facts.arch, .features = facts.features};
  storage->facts.tuned.reserve(facts.tuned.size());
  for (std::size_t i = 0; i < facts.tuned.size(); ++i) {
    auto tune = facts.tuned[i];
    tune.implementation = storage->names[i];
    storage->facts.tuned.push_back(tune);
  }
  return OwnedDeviceFacts(std::move(storage));
}

OwnedDeviceFacts::OwnedDeviceFacts(std::unique_ptr<Storage> storage)
    : storage_(std::move(storage)) {}
OwnedDeviceFacts::OwnedDeviceFacts(OwnedDeviceFacts&&) noexcept = default;
OwnedDeviceFacts& OwnedDeviceFacts::operator=(OwnedDeviceFacts&&) noexcept = default;
OwnedDeviceFacts::~OwnedDeviceFacts() = default;
const DeviceFacts& OwnedDeviceFacts::facts() const { return storage_->facts; }

namespace {

using ir::FusedKind;
using ir::Op;

bool Leaf(ir::Op op) { return op == Op::kInput || op == Op::kWeight || op == Op::kState; }
struct RootIdentity {
  ir::TensorId id{};
  ir::Binding binding{};
  std::uint32_t input = ir::TensorId::kNone;
  std::uint32_t computation = ir::NodeId::kNone;
  std::uint8_t output = 0;
  ir::Role role{};
  format::TensorFormat format{};
  std::array<std::uint64_t, 4> nb{};
  std::uint8_t segment = ir::kJoinedSegment;
  bool operator==(const RootIdentity&) const = default;
};
std::vector<RootIdentity> RootSeal(const ir::Graph& graph) {
  std::vector<RootIdentity> seal;
  std::uint32_t computation = 0;
  for (const auto& node : graph.nodes()) {
    const bool leaf = Leaf(node.key.op);
    for (std::uint8_t slot = 0; slot < node.nout; ++slot) {
      const auto id = node.out[slot];
      const auto& desc = graph.tensor(id);
      if (desc.is_view()) continue;
      RootIdentity entry{.id = id,
                         .binding = desc.binding,
                         .output = slot,
                         .role = desc.role,
                         .format = desc.format,
                         .nb = desc.nb,
                         .segment = desc.segment};
      if (!leaf)
        entry.computation = computation;
      else if (std::holds_alternative<std::monostate>(desc.binding)) {
        const auto inputs = graph.inputs();
        const auto it = std::ranges::find(inputs, id);
        entry.input = static_cast<std::uint32_t>(it - inputs.begin());
      }
      seal.push_back(std::move(entry));
    }
    if (!leaf) ++computation;
  }
  return seal;
}

// Its rows: every dimension past the first.
std::uint64_t Rows(const ir::TensorDesc& t) { return t.ne(1) * t.ne(2) * t.ne(3); }

const ir::TensorDesc* At(std::span<const ir::TensorDesc* const> list, std::size_t i) {
  return i < list.size() ? list[i] : nullptr;
}

// A product of weights `w` [K, N, ...] and input `x` [K, M, ...].
QueryShape Product(const ir::TensorDesc* w, const ir::TensorDesc* x) {
  if (w == nullptr || x == nullptr) {
    return {};
  }
  return {.rows = Rows(*x), .k = x->ne(0), .n = w->ne(1)};
}

}  // namespace

RootAlignment GraphRootAlignments(const ir::Graph& graph,
                                  std::span<const RootAlignmentFact> facts) {
  std::vector<std::uint64_t> alignments(graph.tensor_count(), 0);
  for (const auto& fact : facts) {
    if (!fact.root.valid() || fact.root.v >= graph.tensor_count() ||
        graph.tensor(fact.root).is_view() || fact.alignment == 0 ||
        (fact.alignment & (fact.alignment - 1)) != 0 || alignments[fact.root.v] != 0)
      return {};
    alignments[fact.root.v] = fact.alignment;
  }
  struct CachedGraph {
    const ir::Node* nodes = nullptr;
    std::size_t node_count = 0;
    std::size_t tensor_count = 0;
    bool ok = false;
  };
  return [digest = graph.Digest(), seal = RootSeal(graph),
          alignments = std::move(alignments),
          cached = std::make_shared<CachedGraph>()](
             const ir::Graph& current, ir::TensorId root) -> std::uint64_t {
    if (!root.valid() || root.v >= alignments.size() || alignments[root.v] == 0 ||
        root.v >= current.tensor_count() || current.tensor(root).is_view())
      return std::uint64_t{0};
    if (cached->nodes != current.nodes().data() ||
        cached->node_count != current.nodes().size() ||
        cached->tensor_count != current.tensor_count()) {
      cached->nodes = current.nodes().data();
      cached->node_count = current.nodes().size();
      cached->tensor_count = current.tensor_count();
      cached->ok = (current.Digest() == digest && RootSeal(current) == seal);
    }
    return cached->ok ? alignments[root.v] : std::uint64_t{0};
  };
}

namespace {

bool PowerOfTwo(std::uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

bool SameLeaf(const ir::Graph& before, ir::TensorId a, const ir::Graph& after, ir::TensorId b) {
  if (!a.valid() || a.v >= before.tensor_count() || !b.valid() || b.v >= after.tensor_count())
    return false;
  const auto& x = before.tensor(a);
  const auto& y = after.tensor(b);
  const auto xo = before.node(before.producer(a)).key.op;
  const auto yo = after.node(after.producer(b)).key.op;
  if (!Leaf(xo) || xo != yo || x.is_view() || y.is_view() || x.binding != y.binding ||
      x.role != y.role || x.format != y.format || x.nb != y.nb || x.segment != y.segment)
    return false;
  if (xo == Op::kInput) {
    const auto ai = before.inputs(), bi = after.inputs();
    return std::ranges::find(ai, a) - ai.begin() == std::ranges::find(bi, b) - bi.begin();
  }
  return true;
}

}  // namespace

RootAlignment ForwardRootAlignments(const ir::Graph& before, const ir::Graph& after,
                                    std::span<const ir::TensorId> tensors,
                                    const RootAlignment& facts) {
  if (!facts || tensors.size() != before.tensor_count()) return {};
  const bool same = before.Digest() == after.Digest() && RootSeal(before) == RootSeal(after);
  std::vector<std::uint32_t> uses(after.tensor_count());
  for (const auto t : tensors)
    if (t.valid() && t.v < uses.size()) ++uses[t.v];
  std::vector<RootAlignmentFact> mapped;
  for (std::uint32_t i = 0; i < tensors.size(); ++i) {
    const ir::TensorId old{i}, next = tensors[i];
    if (!next.valid() || next.v >= after.tensor_count() || uses[next.v] != 1 ||
        before.tensor(old).is_view() || after.tensor(next).is_view() ||
        (!(same && old == next) && !SameLeaf(before, old, after, next)))
      continue;
    const auto alignment = facts(before, old);
    if (PowerOfTwo(alignment)) mapped.push_back({next, alignment});
  }
  const auto sealed = GraphRootAlignments(after, mapped);
  // A semantic binding callback may independently authenticate this current
  // graph (without hashing it on every query). Exact-ID callbacks return zero
  // after a rewrite, then use the newly sealed map. No old IDs escape.
  return [original = facts, sealed](const ir::Graph& graph, ir::TensorId root) {
    if (sealed) {
      const auto from_sealed = sealed(graph, root);
      if (PowerOfTwo(from_sealed)) return from_sealed;
    }
    const auto direct = original(graph, root);
    return PowerOfTwo(direct) ? direct : std::uint64_t{0};
  };
}

QueryShape ShapeOfNode(ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
                       std::span<const ir::TensorDesc* const> outputs) {
  const ir::TensorDesc* s0 = At(sources, 0);
  if (key.op == Op::kFused) {
    switch (key.fused) {
      case FusedKind::kMatMulGlu:
        return Product(s0, At(sources, 2));
      case FusedKind::kMatMulIdGlu:
      case FusedKind::kMoeExperts:
        return Product(s0, At(sources, 2));
      case FusedKind::kMatMulAdd:
        return Product(s0, At(sources, 1));
      case FusedKind::kPreparedProducts:
        return sources.empty() ? QueryShape{} : Product(s0, sources.back());
      case FusedKind::kStateBankAttention: {
        const ir::TensorDesc* raw = At(sources, 1);
        return s0 == nullptr || raw == nullptr
                   ? QueryShape{}
                   : QueryShape{.rows = s0->ne(1) * s0->ne(3), .k = s0->ne(0), .n = raw->ne(0)};
      }
      case FusedKind::kOwnerAttention: {
        const ir::TensorDesc* v = At(sources, 2 + ((sources.size() - 2) / 2));
        if (s0 == nullptr || v == nullptr) {
          return {};
        }
        return {.rows = s0->ne(1) * s0->ne(3), .k = s0->ne(0), .n = v->ne(0)};
      }
      case FusedKind::kRopeStore:
        return s0 == nullptr
                   ? QueryShape{}
                   : QueryShape{.rows = s0->ne(2) * s0->ne(3), .k = s0->ne(0) * s0->ne(1)};
      case FusedKind::kGroupedStores: {
        // A batch's member is one store (cache, src, idx).
        const ir::TensorDesc* src = At(sources, 1);
        return src == nullptr ? QueryShape{} : QueryShape{.rows = Rows(*src), .k = src->ne(0)};
      }
      case FusedKind::kDsv4OutA: {
        const ir::TensorDesc* x = At(sources, 1);
        return x == nullptr
                   ? QueryShape{}
                   : QueryShape{.rows = x->ne(2), .k = x->ne(0) * x->ne(1), .n = 8192};
      }
      default:
        break;
    }
  }
  switch (key.op) {
    case Op::kConv2d: {
      const auto* output = At(outputs, 0);
      if (s0 == nullptr || output == nullptr) return {};
      std::uint64_t rows = 0, k = 0;
      if (__builtin_mul_overflow(output->ne(0), output->ne(1), &rows) ||
          __builtin_mul_overflow(rows, output->ne(3), &rows) ||
          __builtin_mul_overflow(s0->ne(0), s0->ne(1), &k) ||
          __builtin_mul_overflow(k, s0->ne(2), &k))
        return {};
      // Spatial products use the same semantic tuning tuple as dense products:
      // output spatial positions, receptive-field reduction, output channels.
      return {.rows = rows, .k = k, .n = s0->ne(3)};
    }
    case Op::kMatMul:
      return Product(s0, At(sources, 1));
    case Op::kMatMulId: {
      const ir::TensorDesc* b = At(sources, 1);
      if (s0 == nullptr || b == nullptr) {
        return {};
      }
      return {.rows = b->ne(2), .k = b->ne(0), .n = s0->ne(1)};
    }
    case Op::kAttention: {
      const ir::TensorDesc* v = At(sources, 2);
      if (s0 == nullptr || v == nullptr) {
        return {};
      }
      return {.rows = s0->ne(1) * s0->ne(3), .k = s0->ne(0), .n = v->ne(0)};
    }
    case Op::kCacheStore: {
      const ir::TensorDesc* src = At(sources, 1);
      return src == nullptr ? QueryShape{} : QueryShape{.rows = Rows(*src), .k = src->ne(0)};
    }
    default:
      break;
  }
  const ir::TensorDesc* t = s0 != nullptr ? s0 : At(outputs, 0);
  return t == nullptr ? QueryShape{} : QueryShape{.rows = Rows(*t), .k = t->ne(0)};
}

bool LaunchesNothing(ir::OpKey key) {
  switch (key.op) {
    case Op::kInput:
    case Op::kWeight:
    case Op::kState:
    case Op::kView:
    case Op::kReshape:
    case Op::kPermute:
      return true;
    default:
      return false;
  }
}

SelectionView::SelectionView(const execution::Registry& registry, const DeviceFacts& device,
                             const Policy& policy, bool row_invariant, bool capturable,
                             StateSpans state_spans, RootAlignment root_alignment,
                             ActivationAlignment activation)
    : registry_(registry),
      device_(device),
      policy_(policy),
      row_invariant_(row_invariant || policy.row_invariant),
      capturable_(capturable),
      state_spans_(std::move(state_spans)),
      root_alignment_(std::move(root_alignment)),
      activation_(std::move(activation)),
      ranks_(RanksFor(policy, device.device_class)) {}

void SelectionView::Forward(const ir::Graph& before, const ir::Graph& after,
                            std::span<const ir::TensorId> tensors) {
  root_alignment_ = ForwardRootAlignments(before, after, tensors, root_alignment_);
  if (activation_.staged) {
    std::vector<ir::TensorId> mapped;
    for (const auto old : *activation_.staged) {
      if (!old.valid() || old.v >= tensors.size()) continue;
      const auto next = tensors[old.v];
      if (SameLeaf(before, old, after, next) && std::ranges::count(tensors, next) == 1)
        mapped.push_back(next);
    }
    activation_.staged = std::move(mapped);
  }
}

std::uint64_t SelectionView::Alignment(const ir::Graph& graph, ir::TensorId root) const {
  if (!root.valid() || root.v >= graph.tensor_count() || graph.tensor(root).is_view()) return 0;
  const auto& t = graph.tensor(root);
  const bool computed = t.role == ir::Role::kActivation || t.role == ir::Role::kOutput;
  const bool staged = activation_.staged ? std::ranges::contains(*activation_.staged, root)
                                         : std::ranges::contains(graph.inputs(), root);
  if (computed || staged) {
    if (activation_.within) {
      const auto containing = activation_.within(graph, root);
      if (containing.valid() && containing != root) return 0;
    }
    if (PowerOfTwo(activation_.base) && PowerOfTwo(activation_.offset))
      return std::gcd(activation_.base, activation_.offset);
  }
  const auto value = root_alignment_ ? root_alignment_(graph, root) : 0;
  return PowerOfTwo(value) ? value : 0;
}

std::vector<RootAlignmentFact> SelectionView::Alignments(const ir::Graph& graph) const {
  std::vector<RootAlignmentFact> out;
  for (std::uint32_t i = 0; i < graph.tensor_count(); ++i)
    if (const auto a = Alignment(graph, {i}); a != 0) out.push_back({{i}, a});
  return out;
}

std::vector<std::size_t> SelectionView::Candidates(ir::OpKey key,
                                                   std::span<const ir::TensorDesc* const> sources,
                                                   std::span<const ir::TensorDesc* const> outputs,
                                                   std::span<const std::byte> params,
                                                   bool state_span) const {
  // Hypothetical descriptors have no authenticated current graph/root.
  return ForOperands(key, sources, outputs, params, state_span, {});
}

std::vector<std::size_t> SelectionView::ForOperands(
    ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
    std::span<const ir::TensorDesc* const> outputs, std::span<const std::byte> params,
    bool state_span, std::span<const std::uint64_t> alignments) const {
  // A composite op whose own implementations the policy leaves off
  // (pattern.h route_reduce) has none: it decomposes.
  if (!CompositeImplemented(policy_, key)) {
    return {};
  }
  // Selected-ID hash routing is the opt-in small-row recipe (at most eight
  // rows). Larger products retain separately rounded primitive routing;
  // a kernel's larger launch limit is not authority to change that recipe.
  // Biased top-k has its own default-off small-row recipe. Other kinds
  // retain their existing native eligibility.
  if (key == ir::Primitive(Op::kRoute) && params.size() >= sizeof(ir::RouteParams)) {
    ir::RouteParams route{};
    std::memcpy(&route, params.data(), sizeof(route));
    if (route.kind == ir::RouteKind::kSqrtSoftplusBias &&
        (Disabled(policy_, "biased_route") || sources.empty() || sources[0] == nullptr ||
         sources[0]->ne(1) > 16))
      return {};
    if (route.kind == ir::RouteKind::kHash && (Disabled(policy_, "hash_route") || sources.empty() ||
                                               sources[0] == nullptr || sources[0]->ne(1) > 8))
      return {};
  }
  // The ungated four-input combination is a distinct ordered-FMA recipe.
  // Direct composites obey the same explicit opt-in as the primitive matcher.
  if (key == ir::Fused(ir::FusedKind::kMoeCombine) &&
      (sources.size() <= 4 || sources[4] == nullptr) && Disabled(policy_, "moe_combine"))
    return {};
  std::vector<execution::OperandFormat> operands;
  operands.reserve(sources.size());
  for (std::size_t i = 0; i < sources.size(); ++i) {
    if (sources[i] == nullptr) {
      continue;
    }
    const format::TensorFormat& f = sources[i]->format;
    operands.push_back(
        {.operand = static_cast<std::uint8_t>(i),
         .encoding = f.encoding,
         .layout = f.layout,
         .params = f.params,
         .ne = {f.shape.ne.at(0), f.shape.ne.at(1), f.shape.ne.at(2), f.shape.ne.at(3)},
         .memory = execution::OperandMemory{
             .nb = sources[i]->nb,
             .view_offset = sources[i]->view_offset,
             .root_alignment = i < alignments.size() ? alignments[i] : 0}});
  }
  // The outputs' formats, for rules and predicates on the result.
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i] == nullptr) {
      continue;
    }
    const format::TensorFormat& f = outputs[i]->format;
    operands.push_back(
        {.operand = static_cast<std::uint8_t>(execution::kOutputOperand + i),
         .encoding = f.encoding,
         .layout = f.layout,
         .params = f.params,
         .ne = {f.shape.ne.at(0), f.shape.ne.at(1), f.shape.ne.at(2), f.shape.ne.at(3)},
         .memory = execution::OperandMemory{.nb = outputs[i]->nb,
                                            .view_offset = outputs[i]->view_offset,
                                            .root_alignment = sources.size() + i < alignments.size()
                                                                  ? alignments[sources.size() + i]
                                                                  : 0}});
  }
  const QueryShape shape = ShapeOfNode(key, sources, outputs);
  const execution::CapabilityQuery query{
      .op = key,
      .operands = operands,
      .rows = shape.rows,
      .k = shape.k,
      .n = shape.n,
      .arch = device_.arch,
      .features = device_.features,
      .row_invariant = row_invariant_,
      .capturable = capturable_,
      .policy = {.max_decode_rows = policy_.max_decode_rows, .allow_lossy = policy_.allow_lossy},
      .measured = ranks_,
      .tuned = device_.tuned,
      .params = params,
      .state_span = state_span};
  return execution::Candidates(registry_, query);
}

std::vector<std::size_t> SelectionView::Candidates(const ir::Graph& graph,
                                                   const ir::Node& node) const {
  return Candidates(graph, node, node.key);
}

std::vector<std::size_t> SelectionView::Candidates(const ir::Graph& graph, const ir::Node& node,
                                                   ir::OpKey key) const {
  // Direct composites need the same immutable source authority as the
  // primitive matcher. Refuse native selection early so ordinary expansion
  // remains available for otherwise valid grouped products.
  const bool grouped = key == ir::Fused(FusedKind::kGroupedMatMul);
  if (grouped) {
    if (Disabled(policy_, "grouped_vecq") || node.nsrc != 2 || !node.src[0].valid()) return {};
    const auto& reshape = graph.node(graph.producer(node.src[0]));
    if (reshape.key != ir::Primitive(Op::kReshape) || reshape.nsrc != 1) return {};
    const auto& weight = graph.tensor(reshape.src[0]);
    const auto& physical = graph.tensor(node.src[0]);
    const auto* ref = std::get_if<ir::WeightRef>(&weight.binding);
    if (!ref || ref->expert_array ||
        graph.node(graph.producer(reshape.src[0])).key != ir::Primitive(Op::kWeight) ||
        weight.is_view() || weight.format.shape.rank != 2 || weight.format.stacked != 1 ||
        weight.format.layout != format::Layout::kGgmlBlocks || physical.view_offset != 0 ||
        physical.ne(0) != weight.ne(0) || physical.ne(2) == 0 ||
        weight.ne(1) / physical.ne(2) != physical.ne(1) || weight.ne(1) % physical.ne(2) != 0 ||
        !ir::IsContiguous(physical))
      return {};
    const auto parts = format::PartsOf(weight.format);
    if (!parts || parts->size() != 1 || (*parts)[0].role != format::PartRole::kData) return {};
  }
  std::vector<const ir::TensorDesc*> sources;
  for (const ir::TensorId s : node.sources()) {
    sources.push_back(s.valid() ? &graph.tensor(s) : nullptr);
  }
  std::vector<const ir::TensorDesc*> outputs;
  for (const ir::TensorId o : node.outputs()) {
    outputs.push_back(&graph.tensor(o));
  }
  bool state_span = false;
  if (state_spans_ && key == ir::Fused(FusedKind::kStateBankAttention)) {
    if (const auto slices = StateBankSlices(graph, node.source(1), node.source(2))) {
      const auto fact = state_spans_(*slices);
      const auto& raw = graph.tensor(node.src[1]);
      const auto& compressed = graph.tensor(node.src[2]);
      state_span = fact && fact->bank.encoding == format::Encoding::kF16 &&
                   fact->bank.cell_bytes == raw.nb[2] &&
                   fact->bank.range.bytes == (raw.ne(2) + compressed.ne(2)) * raw.nb[2] &&
                   fact->bank.range.offset <= fact->region_bytes &&
                   fact->bank.range.bytes <= fact->region_bytes - fact->bank.range.offset;
    }
  }
  std::vector<std::uint64_t> alignments;
  const auto alignment = [&](ir::TensorId tensor) {
    return tensor.valid() ? Alignment(graph, graph.root(tensor)) : 0;
  };
  for (const auto tensor : node.sources()) alignments.push_back(alignment(tensor));
  for (const auto tensor : node.outputs()) alignments.push_back(alignment(tensor));
  auto candidates = ForOperands(key, sources, outputs, node.params, state_span, alignments);
  if (grouped) {
    std::erase_if(candidates, [&](std::size_t index) {
      const auto& capability = registry_.at(index).capability;
      return !capability || !std::ranges::all_of(capability->launches, [&](std::string_view name) {
        return registry_.Find(name).has_value();
      });
    });
  }
  return candidates;
}

}  // namespace llmp::planner
