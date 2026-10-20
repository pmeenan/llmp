// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/owner_products.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "format/encoding.h"
#include "format/layout.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"

namespace llmp::planner {
namespace {
using components::InputKind;
using components::SegmentRows;
using components::StepInput;
using format::Encoding;
using ir::Op;
using ir::TensorId;

struct Rows {
  TensorId root{};
  std::uint64_t row_bytes = 0;
  bool ids = false;
  auto operator<=>(const Rows&) const = default;
};

bool Matrix(const ir::TensorDesc& d) { return d.ne(2) == 1 && d.ne(3) == 1 && ir::IsContiguous(d); }

class Proof {
 public:
  Proof(const ir::Graph& graph, std::span<const StepInput> inputs, std::uint64_t rows)
      : g_(graph),
        rows_(rows),
        memo_(graph.tensor_count()),
        inputs_(graph.tensor_count()),
        duplicate_inputs_(graph.tensor_count(), false),
        written_(graph.tensor_count(), false) {
    for (const auto& input : inputs) {
      if (!input.tensor.valid() || input.tensor.v >= inputs_.size()) continue;
      const auto i = input.tensor.v;
      if (inputs_[i]) duplicate_inputs_[i] = true;
      inputs_[i] = &input;
    }
    // Reject even same-value writes. Non-structural views are writes in
    // this IR (stores and fused stores), including writes after the product.
    for (const ir::Node& n : g_.nodes()) {
      if (n.key.fused == ir::FusedKind::kNone &&
          (n.key.op == Op::kView || n.key.op == Op::kReshape || n.key.op == Op::kPermute))
        continue;
      for (TensorId t : n.outputs())
        if (g_.tensor(t).is_view()) written_[g_.root(t).v] = true;
    }
    // Graphs are verified in producer order. Each dependency is already
    // memoized: proof work uses constant stack depth and one pass per graph.
    for (const ir::Node& n : g_.nodes())
      for (TensorId t : n.outputs()) memo_[t.v] = TraceOne(t);
  }

  std::optional<Rows> Trace(TensorId t) const {
    if (!t.valid() || t.v >= g_.tensor_count()) return std::nullopt;
    return memo_[t.v];
  }
  bool Written(TensorId t) const { return written_[g_.root(t).v]; }

 private:
  bool Independent(TensorId t, const ir::TensorDesc& out) const {
    const auto& n = g_.node(g_.producer(t));
    const auto& d = g_.tensor(t);
    return n.key == ir::Primitive(Op::kWeight) && !d.is_view() && !Written(t) &&
           d.ne(0) == out.ne(0) && d.ne(1) == 1 && d.ne(2) == 1 && d.ne(3) == 1;
  }

  std::optional<Rows> TraceOne(TensorId t) const {
    const auto& d = g_.tensor(t);
    const auto& n = g_.node(g_.producer(t));
    if (Written(t) || n.key.fused != ir::FusedKind::kNone) return std::nullopt;
    if (n.key.op == Op::kInput) {
      const auto* in = inputs_[t.v];
      if (!in || duplicate_inputs_[t.v] || in->segment != ir::kJoinedSegment || !Matrix(d))
        return std::nullopt;
      if (in->kind == InputKind::kTokens && d.format.encoding == Encoding::kI32 &&
          d.ne(0) == rows_ && d.ne(1) == 1)
        return Rows{.root = t, .row_bytes = 4, .ids = true};
      if (in->kind == InputKind::kFeatures && d.format.encoding == Encoding::kF32 &&
          d.ne(1) == rows_)
        return Rows{.root = t, .row_bytes = d.nb[1]};
      return std::nullopt;
    }
    if (n.key.op == Op::kGetRows) {
      auto ids = Trace(n.src[1]);
      const auto& w = g_.tensor(n.src[0]);
      if (!ids || !ids->ids || g_.node(g_.producer(n.src[0])).key != ir::Primitive(Op::kWeight) ||
          w.is_view() || Written(n.src[0]) || !Matrix(w) || !Matrix(d) || d.ne(1) != rows_ ||
          d.format.encoding != Encoding::kF32)
        return std::nullopt;
      return Rows{.root = ids->root, .row_bytes = d.nb[1]};
    }
    if (n.key.op == Op::kRmsNorm) {
      auto src = Trace(n.src[0]);
      const auto& a = g_.tensor(n.src[0]);
      // RMS reduces axis0. A row boundary must be outside that reduction.
      if (!src || src->ids || !Matrix(a) || !Matrix(d) || a.format != d.format ||
          d.format.encoding != Encoding::kF32 || src->row_bytes % d.nb[1] != 0)
        return std::nullopt;
      return src;
    }
    if (n.key.op == Op::kMul) {
      if (!Matrix(d) || d.format.encoding != Encoding::kF32) return std::nullopt;
      auto a = Trace(n.src[0]);
      auto b = Trace(n.src[1]);
      if (a && !a->ids && g_.tensor(n.src[0]).format == d.format &&
          ((b && *a == *b && g_.tensor(n.src[1]).format == d.format) || Independent(n.src[1], d)))
        return a;
      if (b && !b->ids && g_.tensor(n.src[1]).format == d.format && Independent(n.src[0], d))
        return b;
      return std::nullopt;
    }
    if (n.key.op == Op::kReshape || n.key.op == Op::kView) {
      auto src = Trace(n.src[0]);
      const auto& a = g_.tensor(n.src[0]);
      if (!src || src->ids || !Matrix(a) || !Matrix(d) || d.format.encoding != Encoding::kF32 ||
          a.format.encoding != d.format.encoding ||
          ir::Elements(a.shape()) != ir::Elements(d.shape()) || d.view_offset != a.view_offset ||
          g_.root(t) != g_.root(n.src[0]) || src->row_bytes % d.nb[1] != 0)
        return std::nullopt;
      return src;
    }
    return std::nullopt;
  }
  const ir::Graph& g_;
  std::uint64_t rows_;
  std::vector<std::optional<Rows>> memo_;
  std::vector<const StepInput*> inputs_;
  std::vector<bool> duplicate_inputs_;
  std::vector<bool> written_;
};
}  // namespace

base::Sha256Digest OwnerGraphIdentity(const ir::Graph& graph) {
  base::Sha256 hash;
  hash.Update("llmp.owner-graph.v1\n");
  const auto canonical = graph.Digest();
  hash.Update(std::as_bytes(std::span(canonical)));
  for (const auto& n : graph.nodes()) {
    hash.Update(
        std::format("{} {}:", static_cast<unsigned>(n.key.op), static_cast<unsigned>(n.key.fused)));
    for (auto s : n.sources()) hash.Update(std::format(" {}", s.v));
    hash.Update(" ->");
    for (auto o : n.outputs()) hash.Update(std::format(" {}", o.v));
    hash.Update("\n");
  }
  return hash.Finish();
}

std::expected<OwnerProducts, std::string> ProveOwnerProducts(const ir::Graph& graph,
                                                             std::span<const StepInput> inputs,
                                                             std::span<const SegmentRows> segments,
                                                             std::span<const TensorId> requests) {
  OwnerProducts out;
  out.graph_ = OwnerGraphIdentity(graph);
  if (requests.empty()) return out;
  if (segments.empty() || segments.size() > components::kMaxSegments)
    return std::unexpected("owner products: missing or excessive segments");
  std::array<bool, 256> slots{};
  std::uint64_t rows = 0;
  for (const auto& s : segments) {
    if (slots[s.slot] || s.rows == 0 || s.first_row != rows || s.outputs > s.rows)
      return std::unexpected("owner products: malformed row partition");
    slots[s.slot] = true;
    rows += s.rows;  // at most 16 uint32 counts
  }
  Proof proof(graph, inputs, rows);
  std::vector<bool> requested(graph.nodes().size(), false);
  for (TensorId result : requests) {
    if (!result.valid() || result.v >= graph.tensor_count())
      return std::unexpected("owner products: invalid result");
    const ir::NodeId id = graph.producer(result);
    const ir::Node& n = graph.node(id);
    if (n.key != ir::Primitive(Op::kMatMul) || n.nout != 1 ||
        n.Params<ir::ProductParams>().accumulate != ir::Accumulate::kDefault ||
        n.Params<ir::ProductParams>().hint != ir::ProductHint::kNone || requested[id.v])
      return std::unexpected("owner products: not a unique ordinary product");
    requested[id.v] = true;
    const auto& w = graph.tensor(n.src[0]);
    const auto& x = graph.tensor(n.src[1]);
    const auto& y = graph.tensor(result);
    if (graph.node(graph.producer(n.src[0])).key != ir::Primitive(Op::kWeight) || w.is_view() ||
        !Matrix(w) || proof.Written(n.src[0]) ||
        (w.format.encoding != Encoding::kF32 && w.format.encoding != Encoding::kF16 &&
         w.format.encoding != Encoding::kBF16) ||
        x.format.encoding != Encoding::kF32 || y.format.encoding != Encoding::kF32 || !Matrix(x) ||
        !Matrix(y))
      return std::unexpected("owner products: unsupported weight or product view");
    auto lineage = proof.Trace(n.src[1]);
    if (!lineage || lineage->ids || lineage->row_bytes % x.nb[1] != 0)
      return std::unexpected("owner products: unproved row lineage");
    const std::uint64_t factor = lineage->row_bytes / x.nb[1];
    std::uint64_t columns = 0;
    if (factor == 0 || __builtin_mul_overflow(rows, factor, &columns) || columns != x.ne(1))
      return std::unexpected("owner products: physical columns do not cover rows");
    OwnerProduct product{.node = id,
                         .root = lineage->root,
                         .weight = n.src[0],
                         .input = n.src[1],
                         .output = result,
                         .columns_per_row = factor,
                         .owners = {}};
    for (std::size_t i = 0; i < segments.size(); ++i) {
      const auto& s = segments[i];
      const std::uint64_t first = s.first_row * factor;
      const std::uint64_t count = s.rows * factor;
      std::uint64_t offset = 0, bytes = 0;
      if (__builtin_mul_overflow(first, x.nb[1], &offset) ||
          __builtin_mul_overflow(count, x.nb[1], &bytes))
        return std::unexpected("owner products: byte range overflow");
      product.owners.push_back({.slot = s.slot,
                                .segment = static_cast<std::uint8_t>(i),
                                .first_row = s.first_row,
                                .rows = s.rows,
                                .first_column = first,
                                .columns = count,
                                .byte_offset = offset,
                                .bytes = bytes});
    }
    out.products_.push_back(std::move(product));
  }
  base::Sha256 hash;
  hash.Update("llmp.owner-products.v1\n");
  hash.Update(std::as_bytes(std::span(out.graph_)));
  for (const auto& p : out.products_) {
    hash.Update(std::format("{} {} {} {} {} {}\n", p.node.v, p.root.v, p.weight.v, p.input.v,
                            p.output.v, p.columns_per_row));
    for (const auto& o : p.owners)
      hash.Update(std::format("{} {} {} {} {} {} {} {}\n", o.slot, o.segment, o.first_row, o.rows,
                              o.first_column, o.columns, o.byte_offset, o.bytes));
  }
  out.digest_ = hash.Finish();
  return out;
}

std::expected<OwnerRewritten, std::string> SplitOwnerProducts(const ir::Graph& graph,
                                                              const OwnerProducts& proof) {
  if (proof.graph_digest() != OwnerGraphIdentity(graph))
    return std::unexpected("owner products: proof belongs to another graph");
  ir::Builder b;
  std::vector<ir::TensorId> tensors(graph.tensor_count());
  std::vector<ir::NodeId> from;
  std::vector<OwnerProductScope> scopes;
  std::vector<Applied> applied;
  RewriteContext copy(graph, b, tensors);
  for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
    const ir::NodeId id{i};
    const ir::Node& n = graph.node(id);
    const auto p = std::ranges::find(proof.products(), id, &OwnerProduct::node);
    if (p != proof.products().end()) {
      for (const auto& owner : p->owners)
        if (owner.columns > 16)
          return std::unexpected("owner products: owner width exceeds the initial recipe domain");
    }
    if (p == proof.products().end() || p->owners.size() == 1) {
      copy.Copy(n);
      if (p != proof.products().end()) {
        const auto& owner = p->owners.front();
        scopes.push_back({.primitive = id,
                          .result = copy.Map(p->output),
                          .slot = owner.slot,
                          .segment = owner.segment,
                          .columns = owner.columns});
      }
    } else {
      const auto& x = graph.tensor(p->input);
      b.SetComponent(n.component);
      Applied a{.pattern = "owner_products", .replaced = {id}, .made = {}};
      const auto begin = b.node_count();
      ir::TensorId joined;
      for (const auto& owner : p->owners) {
        b.SetOwner(owner.segment);
        auto shape = x.shape();
        shape.ne[1] = owner.columns;
        const auto input = b.View(copy.Map(p->input), shape, x.nb, owner.byte_offset);
        const auto result = b.MatMul(copy.Map(p->weight), input, n.Params<ir::ProductParams>());
        scopes.push_back({.primitive = id,
                          .result = result,
                          .slot = owner.slot,
                          .segment = owner.segment,
                          .columns = owner.columns});
        b.SetJoined();
        joined = joined.valid() ? b.Concat(joined, result, 1) : result;
      }
      copy.Set(p->output, joined);
      for (std::size_t j = begin; j < b.node_count(); ++j)
        a.made.push_back(ir::NodeId{static_cast<std::uint32_t>(j)});
      applied.push_back(std::move(a));
    }
    if (!b.ok())
      return std::unexpected(std::format("owner rewrite at {}: {}", i, b.error()->detail));
    from.resize(b.node_count(), id);
  }
  for (auto t : graph.outputs()) b.Output(copy.Map(t));
  for (auto t : graph.keep()) b.Keep(copy.Map(t));
  auto made = std::move(b).Finish();
  if (!made) return std::unexpected(made.error().detail);
  return OwnerRewritten{.rewrite = {.graph = std::move(*made),
                                    .applied = std::move(applied),
                                    .from = std::move(from),
                                    .tensors = std::move(tensors)},
                        .scopes = std::move(scopes)};
}
}  // namespace llmp::planner
