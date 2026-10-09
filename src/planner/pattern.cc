// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/pattern.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "ir/verify.h"
#include "planner/policy.h"
#include "planner/select.h"

namespace llmp::planner {
namespace {

using ir::FusedKind;
using ir::NodeId;
using ir::Op;
using ir::TensorId;

constexpr ir::OpKey Prim(Op op) { return ir::Primitive(op); }

bool SameShape(const ir::TensorDesc& a, const ir::TensorDesc& b) {
  return a.shape().ne == b.shape().ne;
}

bool DefaultProduct(const ir::Node& n) {
  const auto p = n.Params<ir::ProductParams>();
  return p.accumulate == ir::Accumulate::kDefault && p.hint == ir::ProductHint::kNone;
}

// The candidates for a result of `key` over `sources` (absent ones
// included as null), shaped as `outputs`.
bool Implemented(const MatchContext& c, ir::OpKey key, std::span<const TensorId> sources,
                 std::span<const TensorId> outputs) {
  std::vector<const ir::TensorDesc*> s;
  for (const TensorId t : sources) {
    s.push_back(t.valid() ? &c.tensor(t) : nullptr);
  }
  std::vector<const ir::TensorDesc*> o;
  for (const TensorId t : outputs) {
    o.push_back(&c.tensor(t));
  }
  return !c.view().Candidates(key, s, o).empty();
}

// A fused kind's meaning is its decomposition, whose intermediates are
// F32: a match elides only intermediates the graph computed in F32 (or
// views of them), so fusing never drops a rounding the graph had.
bool ElidesF32(const MatchContext& c, const Match& m) {
  for (std::size_t i = 0; i + 1 < m.nodes.size(); ++i) {
    for (const TensorId o : c.node(m.nodes[i]).outputs()) {
      const ir::TensorDesc& t = c.tensor(o);
      if (!t.is_view() && t.format.encoding != format::Encoding::kF32) {
        return false;
      }
    }
  }
  return true;
}

// Calls `f` with each root `n` writes in place: a cache store's target, a
// fused kind's aliased outputs' (ir/decompose.h AliasedSource).
template <typename F>
void ForEachWrite(const ir::Graph& g, const ir::Node& n, F&& f) {
  if (n.key == Prim(Op::kCacheStore)) {
    f(g.root(n.src[0]));
    return;
  }
  if (n.key.op == Op::kFused) {
    for (std::size_t i = 0; i < n.nout; ++i) {
      if (const auto alias = ir::AliasedSource(n.key, i); alias && n.src.at(*alias).valid()) {
        f(g.root(n.src.at(*alias)));
      }
    }
  }
}

// Whether the rewrite keeps every read of state ordered against every
// write to it. Its replacement reads what a match node read at the match's
// last node (prepared_inputs: the input at its first product), so no other
// node between a match node and the last may write a root the match node
// reads, nor read or write a root it writes (a store a cache store or an
// activation it targets; ir/graph.h orders writes by node order).
bool KeepsOrder(const MatchContext& c, const Match& m) {
  const ir::Graph& g = c.graph();
  const NodeId last = m.nodes.back();
  for (std::size_t i = 0; i + 1 < m.nodes.size(); ++i) {
    const ir::Node& node = g.node(m.nodes[i]);
    std::vector<TensorId> reads;
    for (const TensorId s : node.sources()) {
      if (s.valid()) {
        reads.push_back(g.root(s));
      }
    }
    std::vector<TensorId> writes;
    ForEachWrite(g, node, [&](TensorId r) { writes.push_back(r); });
    for (std::uint32_t k = m.nodes[i].v + 1; k < last.v; ++k) {
      if (std::ranges::contains(m.nodes, NodeId{k})) {
        continue;
      }
      const ir::Node& other = g.node(NodeId{k});
      bool hazard = false;
      ForEachWrite(g, other, [&](TensorId r) {
        hazard = hazard || std::ranges::contains(reads, r) || std::ranges::contains(writes, r);
      });
      for (const TensorId s : other.sources()) {
        hazard = hazard || (s.valid() && std::ranges::contains(writes, g.root(s)));
      }
      if (hazard) {
        return false;
      }
    }
  }
  return true;
}

// Sources without trailing absent optional ones.
std::vector<TensorId> Trimmed(std::vector<TensorId> sources) {
  while (!sources.empty() && !sources.back().valid()) {
    sources.pop_back();
  }
  return sources;
}

// The next node's free reader of `t` when it is the only one.
std::optional<NodeId> Next(const MatchContext& c, TensorId t) {
  const auto r = c.OnlyReader(t);
  if (!r || c.taken(*r)) {
    return std::nullopt;
  }
  return r;
}

// An RMS norm and the mul by its weight.
struct NormMul {
  NodeId norm{};
  NodeId mul{};
  TensorId x{};
  TensorId w{};
  float eps = 0.0f;
};

std::optional<NormMul> NormMulAt(const MatchContext& c, NodeId anchor) {
  const ir::Node& n = c.node(anchor);
  if (n.key != Prim(Op::kRmsNorm) || c.taken(anchor)) {
    return std::nullopt;
  }
  const TensorId normed = n.out[0];
  const auto mul = Next(c, normed);
  if (!mul) {
    return std::nullopt;
  }
  const ir::Node& m = c.node(*mul);
  if (m.key != Prim(Op::kMul)) {
    return std::nullopt;
  }
  TensorId w{};
  if (m.src[0] == normed) {
    w = m.src[1];
  } else if (m.src[1] == normed && SameShape(c.tensor(m.src[0]), c.tensor(normed))) {
    w = m.src[0];
  } else {
    return std::nullopt;
  }
  if (!SameShape(c.tensor(m.out[0]), c.tensor(normed))) {
    return std::nullopt;
  }
  return NormMul{
      .norm = anchor, .mul = *mul, .x = n.src[0], .w = w, .eps = n.Params<ir::NormParams>().eps};
}

template <typename P>
Match Fusion(std::vector<NodeId> nodes, std::vector<TensorId> sources, const P& params) {
  return Match{.nodes = std::move(nodes),
               .sources = Trimmed(std::move(sources)),
               .params = ir::ToBytes(params)};
}

std::optional<Match> NormMulRopeMatch(const MatchContext& c, NodeId anchor) {
  const auto nm = NormMulAt(c, anchor);
  if (!nm) {
    return std::nullopt;
  }
  const TensorId product = c.node(nm->mul).out[0];
  const auto rope = Next(c, product);
  if (!rope) {
    return std::nullopt;
  }
  const ir::Node& r = c.node(*rope);
  if (r.key != Prim(Op::kRope) || r.src[0] != product ||
      !SameShape(c.tensor(r.out[0]), c.tensor(product))) {
    return std::nullopt;
  }
  Match m = Fusion({nm->norm, nm->mul, *rope}, {nm->x, nm->w, r.src[1], r.source(2)},
                   ir::NormRopeParams{.eps = nm->eps, .rope = r.Params<ir::RopeParams>()});
  if (!Implemented(c, ir::Fused(FusedKind::kRmsNormMulRope), m.sources, r.outputs())) {
    return std::nullopt;
  }
  return m;
}

std::optional<Match> NormMulAddMatch(const MatchContext& c, NodeId anchor) {
  const auto nm = NormMulAt(c, anchor);
  if (!nm) {
    return std::nullopt;
  }
  const TensorId product = c.node(nm->mul).out[0];
  const auto add = Next(c, product);
  if (!add) {
    return std::nullopt;
  }
  const ir::Node& a = c.node(*add);
  if (a.key != Prim(Op::kAdd)) {
    return std::nullopt;
  }
  const TensorId residual = a.src[0] == product   ? a.src[1]
                            : a.src[1] == product ? a.src[0]
                                                  : TensorId{};
  if (!residual.valid() || residual == product ||
      !SameShape(c.tensor(residual), c.tensor(product)) ||
      !SameShape(c.tensor(a.out[0]), c.tensor(product))) {
    return std::nullopt;
  }
  Match m =
      Fusion({nm->norm, nm->mul, *add}, {nm->x, nm->w, residual}, ir::NormParams{.eps = nm->eps});
  if (!Implemented(c, ir::Fused(FusedKind::kRmsNormMulAdd), m.sources, a.outputs())) {
    return std::nullopt;
  }
  return m;
}

std::optional<Match> NormMulMatch(const MatchContext& c, NodeId anchor) {
  const auto nm = NormMulAt(c, anchor);
  if (!nm) {
    return std::nullopt;
  }
  Match m = Fusion({nm->norm, nm->mul}, {nm->x, nm->w}, ir::NormParams{.eps = nm->eps});
  if (!Implemented(c, ir::Fused(FusedKind::kRmsNormMul), m.sources, c.node(nm->mul).outputs())) {
    return std::nullopt;
  }
  return m;
}

// Gate and up products of one input, read only by the GLU over them; the
// anchor is the earlier product.
std::optional<Match> MulMatGluMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& p = c.node(anchor);
  if (p.key != Prim(Op::kMatMul) || c.taken(anchor)) {
    return std::nullopt;
  }
  const auto glu = Next(c, p.out[0]);
  if (!glu) {
    return std::nullopt;
  }
  const ir::Node& g = c.node(*glu);
  if (g.key != Prim(Op::kGlu) || g.nsrc != 2 || !g.src[0].valid() || !g.src[1].valid() ||
      g.src[0] == g.src[1]) {
    return std::nullopt;
  }
  const NodeId gate = c.graph().producer(g.src[0]);
  const NodeId up = c.graph().producer(g.src[1]);
  if (std::min(gate, up) != anchor || c.taken(gate) || c.taken(up)) {
    return std::nullopt;
  }
  const ir::Node& gn = c.node(gate);
  const ir::Node& un = c.node(up);
  if (gn.key != Prim(Op::kMatMul) || un.key != Prim(Op::kMatMul) || gn.src[1] != un.src[1] ||
      !DefaultProduct(gn) || !DefaultProduct(un) || c.OnlyReader(gn.out[0]) != glu ||
      c.OnlyReader(un.out[0]) != glu) {
    return std::nullopt;
  }
  const ir::TensorDesc& wg = c.tensor(gn.src[0]);
  const ir::TensorDesc& wu = c.tensor(un.src[0]);
  if (wg.format != wu.format) {
    return std::nullopt;
  }
  std::vector<NodeId> nodes = {gate, up, *glu};
  std::ranges::sort(nodes);
  Match m = Fusion(std::move(nodes), {gn.src[0], un.src[0], gn.src[1]}, g.Params<ir::GluParams>());
  if (!Implemented(c, ir::Fused(FusedKind::kMatMulGlu), m.sources, g.outputs())) {
    return std::nullopt;
  }
  return m;
}

// A rotation [D, H, T] read only as its rows [D · H, T] (a reshape and a
// view of the whole, in either order or alone), which one cache store
// stores.
std::optional<Match> RopeStoreMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& r = c.node(anchor);
  if (r.key != Prim(Op::kRope) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::TensorDesc& x = c.tensor(r.src[0]);
  if (x.ne(3) != 1) {
    return std::nullopt;
  }
  const std::uint64_t width = x.ne(0) * x.ne(1);
  const std::uint64_t rows = x.ne(2);
  std::vector<NodeId> nodes = {anchor};
  TensorId t = r.out[0];
  std::optional<NodeId> next = Next(c, t);
  for (int link = 0; link < 2 && next && c.node(*next).key != Prim(Op::kCacheStore); ++link) {
    const ir::Node& v = c.node(*next);
    const ir::TensorDesc& out = c.tensor(v.out[0]);
    if ((v.key != Prim(Op::kReshape) && v.key != Prim(Op::kView)) || out.view_offset != 0 ||
        !ir::IsContiguous(out) || ir::Elements(out.shape()) != width * rows) {
      return std::nullopt;
    }
    nodes.push_back(*next);
    t = v.out[0];
    next = Next(c, t);
  }
  if (!next) {
    return std::nullopt;
  }
  const ir::Node& s = c.node(*next);
  const ir::TensorDesc& rows_desc = c.tensor(t);
  if (s.key != Prim(Op::kCacheStore) || s.src[1] != t || rows_desc.ne(0) != width ||
      rows_desc.ne(1) != rows) {
    return std::nullopt;
  }
  const ir::TensorDesc& cache = c.tensor(s.src[0]);
  const ir::TensorDesc& idx = c.tensor(s.src[2]);
  if (cache.ne(0) != width || cache.ne(2) != 1 || cache.ne(3) != 1 || idx.ne(0) != rows ||
      idx.ne(1) != 1 || idx.ne(2) != 1) {
    return std::nullopt;
  }
  nodes.push_back(*next);
  Match m = Fusion(std::move(nodes), {r.src[0], r.src[1], s.src[0], s.src[2], r.source(2)},
                   r.Params<ir::RopeParams>());
  if (!Implemented(c, ir::Fused(FusedKind::kRopeStore), m.sources, s.outputs())) {
    return std::nullopt;
  }
  return m;
}

// The format of an input prepared for the products: Q8_1 blocks of its
// shape (what GGML's MMVQ quantizes its activations into).
std::optional<format::TensorFormat> PreparedFormat(const ir::TensorDesc& x) {
  format::TensorFormat f{.encoding = format::Encoding::kQ8_1,
                         .layout = format::Layout::kGgmlBlocks,
                         .shape = x.shape()};
  if (!format::Validate(f)) {
    return std::nullopt;
  }
  return f;
}

// Two or more products reading one contiguous F32 input, each with an
// implementation that takes the input prepared; the anchor is the first.
std::optional<Match> PreparedInputsMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& p = c.node(anchor);
  if (p.key != Prim(Op::kMatMul) || c.taken(anchor)) {
    return std::nullopt;
  }
  const TensorId x = p.src[1];
  const ir::TensorDesc& xd = c.tensor(x);
  if (xd.format.encoding != format::Encoding::kF32 || !ir::IsContiguous(xd)) {
    return std::nullopt;
  }
  const auto prepared = PreparedFormat(xd);
  if (!prepared) {
    return std::nullopt;
  }
  ir::TensorDesc pd;
  pd.format = *prepared;
  pd.nb = ir::NaturalStrides(*prepared);
  const std::array<const ir::TensorDesc*, 1> prep_src = {&xd};
  const std::array<const ir::TensorDesc*, 1> prep_out = {&pd};
  if (c.view().Candidates(Prim(Op::kPrepare), prep_src, prep_out).empty()) {
    return std::nullopt;
  }
  std::vector<NodeId> products;
  for (const NodeId r : c.readers(x)) {
    const ir::Node& q = c.node(r);
    if (r < anchor || c.taken(r) || q.key != Prim(Op::kMatMul) || q.src[1] != x ||
        !DefaultProduct(q) || c.producer(q.src[0]).key != Prim(Op::kWeight) ||
        c.tensor(q.out[0]).format.encoding != format::Encoding::kF32 ||
        std::ranges::contains(products, r)) {
      continue;
    }
    const std::array<const ir::TensorDesc*, 2> s = {&c.tensor(q.src[0]), &pd};
    const std::array<const ir::TensorDesc*, 1> o = {&c.tensor(q.out[0])};
    if (!c.view().Candidates(Prim(Op::kMatMul), s, o).empty()) {
      products.push_back(r);
    }
  }
  if (products.size() < 2 || products.front() != anchor) {
    return std::nullopt;
  }
  return Match{.nodes = std::move(products), .sources = {x}};
}

// Consecutive cache stores (only views and leaves between them), none
// read after the step, into distinct caches, none reading another's
// result, which one implementation of the batch key takes.
std::optional<Match> GroupedStoresMatch(const MatchContext& c, NodeId anchor) {
  const ir::Graph& g = c.graph();
  const auto member = [&](NodeId id) {
    const ir::Node& n = g.node(id);
    return n.key == Prim(Op::kCacheStore) && !c.taken(id) && !c.external(n.out[0]);
  };
  if (!member(anchor)) {
    return std::nullopt;
  }
  std::vector<NodeId> run = {anchor};
  for (std::uint32_t i = anchor.v + 1; i < g.nodes().size(); ++i) {
    const NodeId id{i};
    const ir::Node& n = g.node(id);
    if (LaunchesNothing(n.key)) {
      continue;
    }
    if (!member(id)) {
      break;
    }
    const bool independent = std::ranges::none_of(run, [&](NodeId m) {
      const ir::Node& o = g.node(m);
      // Distinct caches, and no member reads what another writes.
      return g.root(o.src[0]) == g.root(n.src[0]) || g.root(n.src[1]) == g.root(o.out[0]) ||
             g.root(n.src[2]) == g.root(o.out[0]) || g.root(o.src[1]) == g.root(n.out[0]) ||
             g.root(o.src[2]) == g.root(n.out[0]);
    });
    if (!independent) {
      break;
    }
    run.push_back(id);
  }
  if (run.size() < 2) {
    return std::nullopt;
  }
  const ir::OpKey key = ir::Fused(FusedKind::kGroupedStores);
  std::vector<std::vector<std::size_t>> accepted;
  for (const NodeId id : run) {
    accepted.push_back(c.view().Candidates(g, g.node(id), key));
  }
  for (const std::size_t candidate : accepted.front()) {
    const auto& capability = c.view().registry().at(candidate).capability;
    const std::size_t most = std::min<std::size_t>(capability->batch, run.size());
    std::size_t count = 0;
    while (count < most && std::ranges::contains(accepted[count], candidate)) {
      ++count;
    }
    if (count >= 2) {
      run.resize(count);
      return Match{.nodes = std::move(run), .implementation = candidate};
    }
  }
  return std::nullopt;
}

// --------------------------------------------------------------- rewrites

void ContextOf(RewriteContext& c, const ir::Node& node) {
  ir::Builder& b = c.builder();
  b.SetComponent(node.component);
  if (node.batch == ir::Batch::kPerOwner && node.nout > 0) {
    b.SetOwner(c.original().tensor(node.out[0]).segment);
  } else {
    b.SetJoined();
  }
}

// A fused result at the place of the match's last node, standing for its
// outputs.
void FusedRewrite(RewriteContext& c, const Match& m, std::size_t index, FusedKind kind) {
  if (index + 1 != m.nodes.size()) {
    return;  // an elided intermediate
  }
  const ir::Node& last = c.original().node(m.nodes.back());
  ContextOf(c, last);
  std::vector<TensorId> sources;
  for (const TensorId s : m.sources) {
    sources.push_back(c.Map(s));
  }
  std::vector<format::Encoding> encodings;
  for (const TensorId o : last.outputs()) {
    encodings.push_back(c.original().tensor(o).format.encoding);
  }
  const ir::Outputs outs = c.builder().FusedBytes(kind, sources, m.params, encodings);
  for (std::size_t i = 0; i < last.nout && i < outs.n; ++i) {
    c.Set(last.out.at(i), outs[i]);
  }
}

template <FusedKind kKind>
void Fused(RewriteContext& c, const Match& m, std::size_t index) {
  FusedRewrite(c, m, index, kKind);
}

// The input prepared once, at the first product's place; every product
// reading it.
void PreparedRewrite(RewriteContext& c, const Match& m, std::size_t index) {
  const ir::Node& product = c.original().node(m.nodes.at(index));
  ContextOf(c, product);
  ir::Builder& b = c.builder();
  if (index == 0) {
    c.local()[0] =
        b.Prepare(c.Map(m.sources.front()), format::Encoding::kQ8_1, format::Layout::kGgmlBlocks);
  }
  c.Set(product.out[0],
        b.MatMul(c.Map(product.src[0]), c.local()[0], product.Params<ir::ProductParams>()));
}

constexpr std::array kNormAnchors = {Op::kRmsNorm};
constexpr std::array kProductAnchors = {Op::kMatMul};
constexpr std::array kRopeAnchors = {Op::kRope};
constexpr std::array kStoreAnchors = {Op::kCacheStore};

// The table (see the header): its order is priority; where two patterns
// meet at a node (the RMS norm), the order PlanGraph tries them today.
constexpr std::array kPatterns = {
    Pattern{.name = "rms_norm_mul_rope",
            .result = ir::Fused(FusedKind::kRmsNormMulRope),
            .anchors = kNormAnchors,
            .match = &NormMulRopeMatch,
            .rewrite = &Fused<FusedKind::kRmsNormMulRope>},
    Pattern{.name = "rms_norm_mul_add",
            .result = ir::Fused(FusedKind::kRmsNormMulAdd),
            .anchors = kNormAnchors,
            .match = &NormMulAddMatch,
            .rewrite = &Fused<FusedKind::kRmsNormMulAdd>},
    Pattern{.name = "mul_mat_glu",
            .result = ir::Fused(FusedKind::kMatMulGlu),
            .anchors = kProductAnchors,
            .match = &MulMatGluMatch,
            .rewrite = &Fused<FusedKind::kMatMulGlu>},
    Pattern{.name = "rope_store",
            .result = ir::Fused(FusedKind::kRopeStore),
            .anchors = kRopeAnchors,
            .match = &RopeStoreMatch,
            .rewrite = &Fused<FusedKind::kRopeStore>},
    Pattern{.name = "rms_norm_mul",
            .result = ir::Fused(FusedKind::kRmsNormMul),
            .anchors = kNormAnchors,
            .match = &NormMulMatch,
            .rewrite = &Fused<FusedKind::kRmsNormMul>},
    Pattern{.name = "prepared_inputs",
            .result = ir::Primitive(Op::kPrepare),
            .anchors = kProductAnchors,
            .match = &PreparedInputsMatch,
            .rewrite = &PreparedRewrite},
    Pattern{.name = "grouped_stores",
            .kind = PatternKind::kBatch,
            .result = ir::Fused(FusedKind::kGroupedStores),
            .anchors = kStoreAnchors,
            .match = &GroupedStoresMatch},
};

bool Anchored(const Pattern& p, Op op) { return std::ranges::contains(p.anchors, op); }

}  // namespace

// ------------------------------------------------------------ the context

MatchContext::MatchContext(const ir::Graph& graph, const SelectionView& view)
    : graph_(graph),
      view_(view),
      readers_(graph.tensor_count()),
      external_(graph.tensor_count(), false),
      taken_(graph.nodes().size(), false) {
  for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
    for (const TensorId s : graph.nodes()[i].sources()) {
      if (s.valid()) {
        readers_.at(s.v).push_back(NodeId{i});
      }
    }
  }
  for (const auto list : {graph.outputs(), graph.keep()}) {
    for (const TensorId t : list) {
      external_.at(t.v) = true;
      external_.at(graph.root(t).v) = true;
    }
  }
}

std::span<const NodeId> MatchContext::readers(TensorId t) const { return readers_.at(t.v); }

bool MatchContext::external(TensorId t) const {
  return external_.at(t.v) || external_.at(graph_.root(t).v);
}

void MatchContext::Take(std::span<const NodeId> nodes) {
  for (const NodeId n : nodes) {
    taken_.at(n.v) = true;
  }
}

std::optional<NodeId> MatchContext::OnlyReader(TensorId t) const {
  const auto& r = readers_.at(t.v);
  if (r.size() != 1 || external(t)) {
    return std::nullopt;
  }
  return r.front();
}

ir::Outputs RewriteContext::Copy(const ir::Node& node) {
  ContextOf(*this, node);
  std::vector<TensorId> sources;
  for (const TensorId s : node.sources()) {
    sources.push_back(Map(s));
  }
  std::vector<ir::TensorDesc> requested;
  for (const TensorId o : node.outputs()) {
    requested.push_back(original_.tensor(o));
  }
  const ir::Outputs outs =
      builder_.Emit(node.key, node.params, sources, requested,
                    node.nout > 0 ? original_.name(node.out[0]) : std::string_view());
  for (std::size_t i = 0; i < node.nout && i < outs.n; ++i) {
    Set(node.out.at(i), outs[i]);
  }
  return outs;
}

// ------------------------------------------------------------- the table

std::span<const Pattern> Patterns() { return kPatterns; }

const Pattern* FindPattern(std::string_view name) {
  const auto at = std::ranges::find(kPatterns, name, &Pattern::name);
  return at == kPatterns.end() ? nullptr : &*at;
}

base::Sha256Digest PatternsDigest(const Policy& policy) {
  base::Sha256 hash;
  hash.Update(std::string_view("llmp.planner.patterns.v1\n"));
  for (const Pattern& p : kPatterns) {
    if (!Disabled(policy, p.name)) {
      hash.Update(std::format("{} {}\n", p.name, p.version));
    }
  }
  return hash.Finish();
}

std::optional<Match> BatchAt(const Pattern& pattern, const MatchContext& context, NodeId anchor) {
  if (pattern.kind != PatternKind::kBatch || Disabled(context.view().policy(), pattern.name) ||
      !Anchored(pattern, context.node(anchor).key.op) ||
      context.node(anchor).key.fused != FusedKind::kNone) {
    return std::nullopt;
  }
  return pattern.match(context, anchor);
}

std::expected<Rewritten, ir::VerifyError> ApplyPatterns(const ir::Graph& graph,
                                                        const SelectionView& view) {
  MatchContext context(graph, view);
  struct Found {
    const Pattern* pattern = nullptr;
    Match match;
  };
  std::vector<Found> found;
  // The table's order is priority: each pattern over the whole graph, in
  // node order, before the next.
  for (const Pattern& p : kPatterns) {
    if (p.kind != PatternKind::kRewrite || Disabled(view.policy(), p.name)) {
      continue;
    }
    for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
      const NodeId id{i};
      const ir::Node& node = graph.node(id);
      if (context.taken(id) || node.key.fused != FusedKind::kNone || !Anchored(p, node.key.op)) {
        continue;
      }
      auto m = p.match(context, id);
      if (!m || (p.result.op == Op::kFused && !ElidesF32(context, *m)) ||
          !KeepsOrder(context, *m)) {
        continue;
      }
      context.Take(m->nodes);
      found.push_back({.pattern = &p, .match = std::move(*m)});
    }
  }
  // Recorded in node order (of each match's first node).
  std::ranges::sort(found, {}, [](const Found& f) { return f.match.nodes.front(); });
  std::vector<std::pair<std::size_t, std::size_t>> at(graph.nodes().size(),
                                                      {~std::size_t{0}, 0});  // match, index
  for (std::size_t f = 0; f < found.size(); ++f) {
    for (std::size_t k = 0; k < found[f].match.nodes.size(); ++k) {
      at.at(found[f].match.nodes[k].v) = {f, k};
    }
  }
  ir::Builder b;
  std::vector<TensorId> tensors(graph.tensor_count(), TensorId{});
  std::vector<Applied> applied(found.size());
  std::vector<NodeId> from;
  std::vector<std::array<TensorId, 4>> locals(found.size());
  RewriteContext rc(graph, b, tensors);
  for (std::size_t f = 0; f < found.size(); ++f) {
    applied[f].pattern = found[f].pattern->name;
    applied[f].replaced = found[f].match.nodes;
  }
  for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
    const std::size_t before = b.node_count();
    const auto [f, k] = at[i];
    if (f == ~std::size_t{0}) {
      rc.Copy(graph.nodes()[i]);
    } else {
      rc.local() = locals[f];
      found[f].pattern->rewrite(rc, found[f].match, k);
      locals[f] = rc.local();
      for (std::size_t n = before; n < b.node_count(); ++n) {
        applied[f].made.push_back(NodeId{static_cast<std::uint32_t>(n)});
      }
    }
    if (!b.ok()) {
      ir::VerifyError e = *b.error();
      e.node = i;
      return std::unexpected(std::move(e));
    }
    from.resize(b.node_count(), NodeId{i});
  }
  for (const TensorId o : graph.outputs()) {
    b.Output(rc.Map(o));
  }
  for (const TensorId k : graph.keep()) {
    b.Keep(rc.Map(k));
  }
  auto g = std::move(b).Finish();
  if (!g) {
    return std::unexpected(g.error());
  }
  return Rewritten{.graph = std::move(*g),
                   .applied = std::move(applied),
                   .from = std::move(from),
                   .tensors = std::move(tensors)};
}

}  // namespace llmp::planner
