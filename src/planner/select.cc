// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/select.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
#include "planner/fusions.h"
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
      !add(entries) || !add(facts.device_model.size() + 1) || !add(facts.device_class.size()) ||
      !add(1 + 4 * overhead))
    return std::unexpected("owned device facts byte bound overflows");
  for (std::size_t i = 0; i < facts.tuned.size(); ++i) {
    const auto& tune = facts.tuned[i];
    if (tune.implementation.empty() || tune.rows == 0 || (tune.all_rows && tune.rows != 1) ||
        tune.k == 0 || tune.n == 0 || tune.weight_bits > 8 ||
        (tune.output_encoding &&
         std::ranges::find(format::Encodings(), *tune.output_encoding, &format::EncodingInfo::id) ==
             format::Encodings().end()))
      return std::unexpected("owned device facts have an invalid tuning key");
    for (std::size_t j = 0; j < i; ++j) {
      const auto& before = facts.tuned[j];
      if (before.implementation == tune.implementation && before.rows == tune.rows &&
          before.k == tune.k && before.n == tune.n &&
          before.output_encoding == tune.output_encoding &&
          before.weight_bits == tune.weight_bits && before.all_rows == tune.all_rows)
        return std::unexpected("owned device facts repeat a tuning key");
    }
    if (!add(tune.implementation.size()) || !add(1))
      return std::unexpected("owned tuning name byte bound overflows");
  }
  if (facts.preferred.size() > execution::Registry::kMaxImplementations)
    return std::unexpected("owned device facts have too many preferred implementations");
  for (const auto& name : facts.preferred)
    if (name.empty() || name.size() > execution::Registry::kMaxName ||
        !add(sizeof(std::string) + overhead) || !add(name.size() + 1))
      return std::unexpected("owned preferred implementation metadata is invalid or overflows");
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
  storage->facts = {.device_class = facts.device_class,
                    .arch = facts.arch,
                    .features = facts.features,
                    .device_model = facts.device_model,
                    .multiprocessors = facts.multiprocessors,
                    .preferred = facts.preferred};
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
class RootCursor {
 public:
  bool Next(const ir::Graph& graph, RootIdentity& entry) {
    const auto nodes = graph.nodes();
    while (node_ < nodes.size()) {
      const auto& node = nodes[node_];
      const bool leaf = Leaf(node.key.op);
      if (slot_ == node.nout) {
        if (!leaf) ++computation_;
        ++node_;
        slot_ = 0;
        continue;
      }
      const std::uint8_t slot = slot_++;
      const auto id = node.out[slot];
      const auto& desc = graph.tensor(id);
      if (desc.is_view()) continue;
      entry = {.id = id,
               .binding = desc.binding,
               .output = slot,
               .role = desc.role,
               .format = desc.format,
               .nb = desc.nb,
               .segment = desc.segment};
      if (!leaf)
        entry.computation = computation_;
      else if (std::holds_alternative<std::monostate>(desc.binding)) {
        const auto inputs = graph.inputs();
        const auto it = std::ranges::find(inputs, id);
        entry.input = static_cast<std::uint32_t>(it - inputs.begin());
      }
      return true;
    }
    return false;
  }

 private:
  std::size_t node_ = 0;
  std::uint8_t slot_ = 0;
  std::uint32_t computation_ = 0;
};

std::vector<RootIdentity> RootSeal(const ir::Graph& graph) {
  std::vector<RootIdentity> seal;
  RootCursor cursor;
  RootIdentity entry;
  while (cursor.Next(graph, entry)) seal.push_back(entry);
  return seal;
}

bool RootSealMatches(const ir::Graph& graph, std::span<const RootIdentity> seal) {
  RootCursor cursor;
  RootIdentity entry;
  std::size_t index = 0;
  while (cursor.Next(graph, entry))
    if (index == seal.size() || entry != seal[index++]) return false;
  return index == seal.size();
}

bool RootSealsEqual(const ir::Graph& a, const ir::Graph& b) {
  RootCursor left, right;
  RootIdentity x, y;
  for (;;) {
    const bool has_x = left.Next(a, x);
    const bool has_y = right.Next(b, y);
    if (has_x != has_y) return false;
    if (!has_x) return true;
    if (x != y) return false;
  }
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
  return {.rows = x->ne(1), .k = x->ne(0), .n = w->ne(1)};
}

}  // namespace

namespace {

// The facts' table for `graph`, or none when they are malformed.
std::optional<std::vector<std::uint64_t>> AlignmentTable(const ir::Graph& graph,
                                                         std::span<const RootAlignmentFact> facts) {
  std::vector<std::uint64_t> alignments(graph.tensor_count(), 0);
  for (const auto& fact : facts) {
    if (!fact.root.valid() || fact.root.v >= graph.tensor_count() ||
        graph.tensor(fact.root).is_view() || fact.alignment == 0 ||
        (fact.alignment & (fact.alignment - 1)) != 0 || alignments[fact.root.v] != 0)
      return std::nullopt;
    alignments[fact.root.v] = fact.alignment;
  }
  return alignments;
}

}  // namespace

RootAlignment GraphRootAlignments(const ir::Graph& graph,
                                  std::span<const RootAlignmentFact> facts) {
  auto table = AlignmentTable(graph, facts);
  if (!table || graph.identity() == 0) return {};
  // The graph sealed is the one queried while its identity (ir::Graph::
  // identity: unique to each built or cloned graph, carried by a move,
  // never reused; graphs never change once built) is the query's: no
  // hashing. Any other graph (a clone, an equal rebuild) is authenticated
  // by digest and root seal, once for each identity seen last.
  struct CachedGraph {
    std::uint64_t identity = 0;
    bool ok = false;
  };
  const std::uint64_t sealed = graph.identity();
  return [sealed, digest = graph.Digest(), seal = RootSeal(graph), alignments = std::move(*table),
          cached = std::make_shared<CachedGraph>()](const ir::Graph& current,
                                                    ir::TensorId root) -> std::uint64_t {
    if (!root.valid() || root.v >= alignments.size() || alignments[root.v] == 0 ||
        root.v >= current.tensor_count() || current.tensor(root).is_view())
      return std::uint64_t{0};
    const std::uint64_t identity = current.identity();
    if (identity != 0 && identity == sealed) return alignments[root.v];
    if (identity == 0) {
      // A graph without an identity is authenticated every time.
      return current.Digest() == digest && RootSealMatches(current, seal) ? alignments[root.v]
                                                                          : std::uint64_t{0};
    }
    if (cached->identity != identity) {
      cached->identity = identity;
      cached->ok = current.tensor_count() == alignments.size() && RootSealMatches(current, seal) &&
                   current.Digest() == digest;
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
  // Whether the pass changed nothing: needed only for a root that is not a
  // mapped leaf (a computed root keeps its facts only then), so the graphs
  // are compared, by digest and root seal, only for such a root.
  std::optional<bool> unchanged;
  const auto same = [&] {
    if (!unchanged)
      unchanged = (before.identity() != 0 && before.identity() == after.identity()) ||
                  (before.tensor_count() == after.tensor_count() && RootSealsEqual(before, after) &&
                   before.Digest() == after.Digest());
    return *unchanged;
  };
  std::vector<std::uint32_t> uses(after.tensor_count());
  for (const auto t : tensors)
    if (t.valid() && t.v < uses.size()) ++uses[t.v];
  std::vector<RootAlignmentFact> mapped;
  for (std::uint32_t i = 0; i < tensors.size(); ++i) {
    const ir::TensorId old{i}, next = tensors[i];
    if (!next.valid() || next.v >= after.tensor_count() || uses[next.v] != 1 ||
        before.tensor(old).is_view() || after.tensor(next).is_view())
      continue;
    const bool leaf = SameLeaf(before, old, after, next);
    if (!leaf && old != next) continue;
    const auto alignment = facts(before, old);
    if (!PowerOfTwo(alignment) || (!leaf && !same())) continue;
    mapped.push_back({next, alignment});
  }
  // (`mapped` is well formed: one fact a root, each a power of two.)
  auto table = AlignmentTable(after, mapped).value_or(std::vector<std::uint64_t>{});
  // The forwarded facts are `after`'s, the graph the planner moves on with
  // (a move carries its identity): sealed by that identity alone, never
  // hashed. Any other graph, a later pass's among them, gets none of them
  // and asks the earlier facts (`facts`), which authenticate it themselves.
  const std::uint64_t sealed = after.identity();
  if (sealed == 0) return {};
  const auto from_sealed = [sealed, alignments = std::move(table),
                            fallback =
                                sealed == 0 ? GraphRootAlignments(after, mapped) : RootAlignment{}](
                               const ir::Graph& graph, ir::TensorId root) -> std::uint64_t {
    if (fallback) return fallback(graph, root);
    if (!root.valid() || root.v >= alignments.size() || graph.identity() != sealed ||
        graph.tensor(root).is_view())
      return 0;
    return alignments[root.v];
  };
  // A semantic binding callback may independently authenticate this current
  // graph (without hashing it on every query). Exact-ID callbacks return zero
  // after a rewrite, then use the newly sealed map. No old IDs escape.
  return [original = facts, from_sealed](const ir::Graph& graph, ir::TensorId root) {
    const auto value = from_sealed(graph, root);
    if (PowerOfTwo(value)) return value;
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
      case FusedKind::kInverseRopeGroupedMatMul: {
        const ir::TensorDesc* x = At(sources, 1);
        return x == nullptr ? QueryShape{}
                            : QueryShape{.rows = x->ne(2),
                                         .k = x->ne(0) * x->ne(1),
                                         .n = s0 == nullptr ? 0 : s0->ne(1) * s0->ne(2)};
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
    case Op::kGetRows: {
      const auto* ids = At(sources, 1);
      return !s0 || !ids ? QueryShape{}
                         : QueryShape{.rows = ids->ne(0), .k = s0->ne(0), .n = s0->ne(1)};
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
      ranks_(RanksFor(policy, device.device_class)) {
  for (const auto& pattern : Patterns())
    if (!Disabled(policy, pattern.name)) enabled_options_.push_back(pattern.name);
}

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
                                                   bool state_span, bool automatic_fusion) const {
  // Hypothetical descriptors have no authenticated current graph/root.
  return ForOperands(key, sources, outputs, params, state_span, {}, automatic_fusion);
}

bool SelectionView::HasCandidate(ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
                                 std::span<const ir::TensorDesc* const> outputs,
                                 std::span<const std::byte> params, bool state_span,
                                 bool automatic_fusion) const {
  bool found = false;
  // Hypothetical descriptors have no authenticated current graph/root. The
  // existence-only path avoids constructing a ranked candidate vector.
  (void)ForOperands(key, sources, outputs, params, state_span, {}, automatic_fusion, {}, &found);
  return found;
}

std::optional<std::size_t> SelectionView::NextCandidate(
    ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
    std::span<const ir::TensorDesc* const> outputs, std::span<const std::byte> params,
    std::optional<std::size_t> after, bool state_span, bool automatic_fusion) const {
  (void)ForOperands(key, sources, outputs, params, state_span, {}, automatic_fusion, {}, nullptr,
                    &after);
  return after;
}

std::vector<std::size_t> SelectionView::ForOperands(
    ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
    std::span<const ir::TensorDesc* const> outputs, std::span<const std::byte> params,
    bool state_span, std::span<const std::uint64_t> alignments, bool automatic_fusion,
    const QueryInspector& inspect, bool* has_candidate, std::optional<std::size_t>* next_after,
    std::size_t* check_implementation, bool* is_candidate) const {
  const auto clear_cursor = [&] {
    if (next_after != nullptr) next_after->reset();
  };
  // A composite op whose own implementations the policy leaves off
  // (pattern.h route_reduce) has none: it decomposes.
  if (!CompositeImplemented(policy_, key)) {
    clear_cursor();
    return {};
  }
  if (std::ranges::any_of(alignments, [](auto alignment) { return alignment >= (1ULL << 63); })) {
    clear_cursor();
    return {};
  }
  if (sources.size() > ir::kMaxSources || outputs.size() > ir::kMaxOutputs) {
    clear_cursor();
    return {};
  }
  std::array<execution::OperandFormat, ir::kMaxSources + ir::kMaxOutputs> operand_storage{};
  std::size_t operand_count = 0;
  for (std::size_t i = 0; i < sources.size(); ++i) {
    if (sources[i] == nullptr) {
      continue;
    }
    const format::TensorFormat& f = sources[i]->format;
    operand_storage.at(operand_count++) = {
        .operand = static_cast<std::uint8_t>(i),
        .stacked = f.stacked,
        .encoding = f.encoding,
        .layout = f.layout,
        .rank = f.shape.rank,
        .params = f.params,
        .ne = {f.shape.ne.at(0), f.shape.ne.at(1), f.shape.ne.at(2), f.shape.ne.at(3)},
        .memory =
            execution::OperandMemory{.nb = sources[i]->nb,
                                     .view_offset = sources[i]->view_offset,
                                     .root_alignment = i < alignments.size() ? alignments[i] : 0,
                                     .is_view = sources[i]->is_view()},
    };
  }
  // The outputs' formats, for rules and predicates on the result.
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i] == nullptr) {
      continue;
    }
    const format::TensorFormat& f = outputs[i]->format;
    operand_storage.at(operand_count++) = {
        .operand = static_cast<std::uint8_t>(execution::kOutputOperand + i),
        .stacked = f.stacked,
        .encoding = f.encoding,
        .layout = f.layout,
        .rank = f.shape.rank,
        .params = f.params,
        .ne = {f.shape.ne.at(0), f.shape.ne.at(1), f.shape.ne.at(2), f.shape.ne.at(3)},
        .memory = execution::OperandMemory{.nb = outputs[i]->nb,
                                           .view_offset = outputs[i]->view_offset,
                                           .root_alignment = sources.size() + i < alignments.size()
                                                                 ? alignments[sources.size() + i]
                                                                 : 0,
                                           .is_view = outputs[i]->is_view()}};
  }
  const QueryShape shape = ShapeOfNode(key, sources, outputs);
  execution::CapabilityQuery query{
      .op = key,
      .operands = std::span(operand_storage).first(operand_count),
      .rows = shape.rows,
      .k = shape.k,
      .n = shape.n,
      .arch = device_.arch,
      .features = device_.features,
      .row_invariant = row_invariant_,
      .capturable = capturable_,
      .policy = {.max_decode_rows = policy_.max_decode_rows,
                 .allow_lossy = policy_.allow_lossy,
                 .vector_rows = policy_.vector_rows,
                 .options = std::span<const std::string_view>(enabled_options_)},
      .measured = ranks_,
      .tuned = device_.tuned,
      .params = params,
      .state_span = state_span,
      .device_class = device_.device_class,
      .device_model = device_.device_model,
      .multiprocessors = device_.multiprocessors,
      .automatic_fusion = automatic_fusion,
      .preferred = device_.preferred};
  if (inspect) {
    inspect(query);
    return {};
  }
  if (has_candidate != nullptr) {
    *has_candidate = execution::HasCandidate(registry_, query);
    return {};
  }
  if (next_after != nullptr) {
    *next_after = execution::NextCandidate(registry_, query, *next_after);
    return {};
  }
  if (check_implementation != nullptr) {
    if (is_candidate != nullptr)
      *is_candidate = execution::IsCandidate(registry_, query, *check_implementation);
    return {};
  }
  return execution::Candidates(registry_, query);
}

std::vector<std::size_t> SelectionView::Candidates(const ir::Graph& graph,
                                                   const ir::Node& node) const {
  return Candidates(graph, node, node.key);
}

std::vector<std::size_t> SelectionView::Candidates(const ir::Graph& graph, const ir::Node& node,
                                                   ir::OpKey key, bool automatic_fusion) const {
  return CandidatesForNode(graph, node, key, automatic_fusion);
}
std::optional<std::size_t> SelectionView::NextCandidate(const ir::Graph& graph,
                                                        const ir::Node& node, ir::OpKey key,
                                                        std::optional<std::size_t> after,
                                                        bool automatic_fusion) const {
  (void)CandidatesForNode(graph, node, key, automatic_fusion, {}, &after);
  return after;
}
bool SelectionView::HasCandidate(const ir::Graph& graph, const ir::Node& node, ir::OpKey key,
                                 bool automatic_fusion) const {
  return NextCandidate(graph, node, key, std::nullopt, automatic_fusion).has_value();
}
bool SelectionView::IsCandidate(const ir::Graph& graph, const ir::Node& node, ir::OpKey key,
                                std::size_t implementation, bool automatic_fusion) const {
  bool found = false;
  (void)CandidatesForNode(graph, node, key, automatic_fusion, {}, nullptr, &implementation, &found);
  return found;
}
void SelectionView::Inspect(const ir::Graph& graph, const ir::Node& node,
                            const QueryInspector& inspect) const {
  if (inspect) (void)CandidatesForNode(graph, node, node.key, false, inspect);
}
void SelectionView::Inspect(const ir::Graph& graph, const ir::Node& node, ir::OpKey key,
                            const QueryInspector& inspect, bool automatic_fusion) const {
  if (inspect) (void)CandidatesForNode(graph, node, key, automatic_fusion, inspect);
}
QueryShape SelectionView::Shape(const ir::Graph& graph, const ir::Node& node, ir::OpKey key) const {
  std::array<const ir::TensorDesc*, ir::kMaxSources> sources{};
  std::array<const ir::TensorDesc*, ir::kMaxOutputs> outputs{};
  for (std::size_t i = 0; i < node.nsrc; ++i) {
    const auto tensor = node.source(i);
    sources[i] = tensor.valid() ? &graph.tensor(tensor) : nullptr;
  }
  for (std::size_t i = 0; i < node.nout; ++i) outputs[i] = &graph.tensor(node.out[i]);
  return ShapeOfNode(key, std::span(sources).first(node.nsrc), std::span(outputs).first(node.nout));
}
ir::ProductArithmetic SelectionView::IncumbentArithmetic(const ir::Graph& graph,
                                                         const ir::Node& node) const {
  const auto candidates = Candidates(graph, node);
  if (candidates.empty()) return ir::ProductArithmetic::kImplementation;
  const auto& cap = registry_.at(candidates.front()).capability;
  if (!cap || !cap->product_arithmetic) return ir::ProductArithmetic::kImplementation;
  auto arithmetic = ir::ProductArithmetic::kImplementation;
  Inspect(graph, node,
          [&](const execution::CapabilityQuery& q) { arithmetic = cap->product_arithmetic(q); });
  return arithmetic;
}
std::optional<SelectionView::WeightOperand> SelectionView::ResolveWeightOperand(
    const ir::Graph& graph, const ir::Node& node, std::uint8_t operand) {
  auto weight = node.source(operand);
  for (std::size_t depth = 0; weight.valid() && depth < graph.nodes().size(); ++depth) {
    const auto& tensor = graph.tensor(weight);
    if (const auto* ref = std::get_if<ir::WeightRef>(&tensor.binding))
      return WeightOperand{*ref, weight};
    const auto& producer = graph.node(graph.producer(weight));
    if (producer.component != node.component || producer.nsrc == 0) break;
    weight = producer.source(0);
  }
  return std::nullopt;
}
std::optional<NumericalQualification> SelectionView::Qualification(
    NumericalChange change, const ir::Graph& graph, const ir::Node& node,
    std::uint8_t weight_operand) const {
  if (!policy_.numerical || policy_.active_stage == UINT16_MAX ||
      weight_operand >= NumericalOperands(node))
    return std::nullopt;
  QueryShape shape = Shape(graph, node, node.key);
  auto anchor = node.key;
  if (NumericalOperands(node) == 2) {
    // The fused output has no independent authority for its second product.
    // Project each product's actual K/N from that operand, with the shared
    // input's real row count; use the original primitive product anchor.
    anchor = ir::Primitive(node.key.fused == FusedKind::kMatMulGlu ? Op::kMatMul : Op::kMatMulId);
    const auto& weight = graph.tensor(node.source(weight_operand));
    shape.k = weight.ne(0);
    shape.n = weight.ne(1);
  }
  const bool product_node = node.key == ir::Primitive(Op::kMatMul) ||
                            node.key == ir::Primitive(Op::kMatMulId) ||
                            NumericalOperands(node) == 2;
  const bool product_contract = (change == NumericalChange::kQ8VectorReduction ||
                                 change == NumericalChange::kMxfp8ProductInput) &&
                                product_node;
  const auto weight = ResolveWeightOperand(graph, node, weight_operand);
  const bool viewed_weight = product_node && weight_operand < node.nsrc &&
                             node.source(weight_operand).valid() &&
                             graph.tensor(node.source(weight_operand)).is_view();
  std::optional<fusions::GroupedProduct> grouped;
  if (viewed_weight && node.key == ir::Primitive(Op::kMatMul) && weight_operand == 0 &&
      node.nout > 0 && node.out[0].valid()) {
    grouped = fusions::GroupedProductOf(graph, graph.producer(node.out[0]));
  }
  for (const auto& entry : policy_.numerical->entries()) {
    if (entry.change != change || entry.stage != policy_.active_stage ||
        entry.component != node.component || entry.anchor != anchor || entry.arch != device_.arch ||
        shape.rows < entry.min_rows || shape.rows > entry.max_rows)
      continue;
    if (entry.weight_role != UINT16_MAX || product_contract || viewed_weight) {
      // Follow this product's weight operand, never its sibling's binding or
      // another dependency branch. Stage, component and role all authenticate.
      if (!weight || weight->reference.stage != entry.stage ||
          weight->reference.component != node.component ||
          (entry.weight_role != UINT16_MAX && weight->reference.role != entry.weight_role))
        continue;
    }
    if (!viewed_weight) {
      if ((entry.exact_shape || entry.exact_dimensions)
              ? (shape.k != entry.max_k || shape.n != entry.max_n)
              : (shape.k > entry.max_k || shape.n > entry.max_n))
        continue;
      return entry;
    }
    if (entry.weight_role == UINT16_MAX || (!entry.exact_shape && !entry.exact_dimensions) ||
        !grouped || grouped->leaf != weight->leaf)
      continue;
    const auto& original = graph.tensor(grouped->leaf);
    if (original.ne(0) != entry.max_k || original.ne(1) != entry.max_n) continue;
    auto projected = entry;
    projected.max_k = graph.tensor(grouped->weight).ne(0);
    projected.max_n = graph.tensor(grouped->weight).ne(1);
    return projected;
  }
  return std::nullopt;
}

std::vector<std::size_t> SelectionView::CandidatesForNode(
    const ir::Graph& graph, const ir::Node& node, ir::OpKey key, bool automatic_fusion,
    const QueryInspector& inspect, std::optional<std::size_t>* next_after,
    std::size_t* check_implementation, bool* is_candidate) const {
  const auto cursor_start = next_after == nullptr ? std::optional<std::size_t>{} : *next_after;
  if (next_after != nullptr) next_after->reset();
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
  std::array<const ir::TensorDesc*, ir::kMaxSources> sources{};
  std::array<const ir::TensorDesc*, ir::kMaxOutputs> outputs{};
  std::array<std::uint64_t, ir::kMaxSources + ir::kMaxOutputs> alignments{};
  for (std::size_t i = 0; i < node.nsrc; ++i) {
    const ir::TensorId s = node.source(i);
    sources[i] = s.valid() ? &graph.tensor(s) : nullptr;
    alignments[i] = s.valid() ? Alignment(graph, graph.root(s)) : 0;
  }
  for (std::size_t i = 0; i < node.nout; ++i) {
    const ir::TensorId o = node.out[i];
    outputs[i] = &graph.tensor(o);
    alignments[node.nsrc + i] = Alignment(graph, graph.root(o));
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
  const auto srcs = std::span(sources).first(node.nsrc);
  const auto outs = std::span(outputs).first(node.nout);
  const auto aligned = std::span(alignments).first(node.nsrc + node.nout);
  if (next_after != nullptr) {
    auto after = cursor_start;
    for (;;) {
      (void)ForOperands(key, srcs, outs, node.params, state_span, aligned, automatic_fusion, {},
                        nullptr, &after);
      if (!after) {
        *next_after = {};
        return {};
      }
      bool available = true;
      if (grouped) {
        const auto& capability = registry_.at(*after).capability;
        available =
            capability && std::ranges::all_of(capability->launches, [&](std::string_view name) {
              return registry_.Find(name).has_value();
            });
      }
      if (available) {
        *next_after = after;
        return {};
      }
    }
  }
  if (check_implementation != nullptr) {
    (void)ForOperands(key, srcs, outs, node.params, state_span, aligned, automatic_fusion, {},
                      nullptr, nullptr, check_implementation, is_candidate);
    if (grouped && is_candidate != nullptr && *is_candidate) {
      const auto& capability = registry_.at(*check_implementation).capability;
      *is_candidate =
          capability && std::ranges::all_of(capability->launches, [&](std::string_view name) {
            return registry_.Find(name).has_value();
          });
    }
    return {};
  }
  auto candidates =
      ForOperands(key, srcs, outs, node.params, state_span, aligned, automatic_fusion, inspect);
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
