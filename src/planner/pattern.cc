// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/pattern.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <initializer_list>
#include <limits>
#include <memory>
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
#include "planner/step_shape.h"

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
// included as null), shaped as `outputs`, with the match's parameters.
bool Implemented(const MatchContext& c, ir::OpKey key, std::span<const TensorId> sources,
                 std::span<const TensorId> outputs, const ir::ParamBytes& params) {
  std::vector<const ir::TensorDesc*> s;
  for (const TensorId t : sources) {
    s.push_back(t.valid() ? &c.tensor(t) : nullptr);
  }
  std::vector<const ir::TensorDesc*> o;
  for (const TensorId t : outputs) {
    o.push_back(&c.tensor(t));
  }
  return !c.view().Candidates(key, s, o, params).empty();
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
  if (!Implemented(c, ir::Fused(FusedKind::kRmsNormMulRope), m.sources, r.outputs(), m.params)) {
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
  if (!Implemented(c, ir::Fused(FusedKind::kRmsNormMulAdd), m.sources, a.outputs(), m.params)) {
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
  if (!Implemented(c, ir::Fused(FusedKind::kRmsNormMul), m.sources, c.node(nm->mul).outputs(),
                   m.params)) {
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
  if (!Implemented(c, ir::Fused(FusedKind::kMatMulGlu), m.sources, g.outputs(), m.params)) {
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
  if (!Implemented(c, ir::Fused(FusedKind::kRopeStore), m.sources, s.outputs(), m.params)) {
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
    if (!c.view().Candidates(Prim(Op::kMatMul), s, o, q.params).empty()) {
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

// ------------------------------------------------------- owner attention

// What owner_attention found: the step's joined owners in a form an
// owner-root implementation takes; every layer's per-owner attention (a
// group: each owner's query, cache views, attention and result rows, and
// the joins of their rows); the owners' masks by cell rule (a kind: each
// owner's mask over one kind of cache, joined once for every layer); and
// the zeros that pad a narrower owner's cache to its kind's width where
// the implementation needs equal widths.
struct OwnerPlan final : MatchDetail {
  struct Kind {
    std::vector<TensorId> masks;  // by owner
    std::uint64_t cells = 0;      // the joined mask's: the widest owner's
    std::uint64_t rows = 0;       // each mask's padded rows
    std::uint16_t component = 0;
  };
  struct Tail {
    std::uint32_t owner = 0;
    std::size_t kind = 0;
    std::array<std::uint64_t, 3> ne{};  // [head size, KV heads, cells]
  };
  struct Layer {
    TensorId q{};    // the joined queries [D, H, rows]
    TensorId out{};  // the joined rows its readers read [Dv · H, rows]
    std::size_t kind = 0;
    std::vector<TensorId> k, v;       // each owner's cache view [D, Hk, C_o]
    std::vector<int> k_tail, v_tail;  // into tails, or -1
    std::uint16_t component = 0;
  };
  std::vector<Kind> kinds;
  std::vector<std::size_t> kind_order;  // the kinds by their masks' node order
  std::vector<Tail> tails;
  std::vector<Layer> layers;
  // By match node: kElided, kMask or kLastMask (a mask, kept; after the
  // last one the joins and the tails), else the layer whose join it is.
  static constexpr std::int32_t kElided = -1;
  static constexpr std::int32_t kMask = -2;
  static constexpr std::int32_t kLastMask = -3;
  std::vector<std::int32_t> at;
  std::uint32_t owners = 0;
  std::uint64_t rows = 0;  // each owner's
  std::uint64_t kd = 0, vd = 0, heads = 0, kv_heads = 0;
  ir::AttnParams attn{};
};

constexpr std::uint32_t kOwnerGroup = 4;  // the owners of one node at most (op.h)

// One owner's attention of a layer as the components emit it (attention.
// dense): its queries a view of the layer's joined queries, K and V views
// of the owner's cache leaves, each permuted and read once, its mask a
// kMaskBuild's, no sinks, and its result's rows reshaped.
struct OwnerAttention {
  NodeId attention{};
  std::array<NodeId, 6> views{};  // q view, q permute, k view, k permute, v view, v permute
  NodeId rows{};                  // the reshape of its result
  TensorId q{};                   // the joined queries
  TensorId k{}, v{};              // the cache views [D, Hk, C_o]
  TensorId mask{};
  std::uint8_t owner = 0;
  std::uint64_t first_row = 0, query_rows = 0;
};

std::optional<OwnerAttention> OwnerAttentionAt(const MatchContext& c, NodeId id) {
  const ir::Node& a = c.node(id);
  if (a.key != Prim(Op::kAttention) || c.taken(id) || a.batch != ir::Batch::kPerOwner ||
      a.nsrc < 4 || a.source(4).valid() || !a.src[3].valid()) {
    return std::nullopt;
  }
  OwnerAttention o;
  o.attention = id;
  o.owner = c.tensor(a.out[0]).segment;
  // A permute (0, 2, 1, 3) of a view, each read once, by the next.
  const auto permuted_view = [&](TensorId t, NodeId reader, NodeId& view,
                                 NodeId& permute) -> std::optional<TensorId> {
    if (c.OnlyReader(t) != reader) {
      return std::nullopt;
    }
    permute = c.graph().producer(t);
    const ir::Node& p = c.node(permute);
    if (p.key != Prim(Op::kPermute) || c.taken(permute) ||
        p.Params<ir::PermuteParams>().axes != std::array<std::uint8_t, 4>{0, 2, 1, 3} ||
        c.OnlyReader(p.src[0]) != permute) {
      return std::nullopt;
    }
    view = c.graph().producer(p.src[0]);
    if (c.node(view).key != Prim(Op::kView) || c.taken(view)) {
      return std::nullopt;
    }
    return p.src[0];
  };
  const auto qv = permuted_view(a.src[0], id, o.views[0], o.views[1]);
  const auto kv = permuted_view(a.src[1], id, o.views[2], o.views[3]);
  const auto vv = permuted_view(a.src[2], id, o.views[4], o.views[5]);
  if (!qv || !kv || !vv) {
    return std::nullopt;
  }
  o.k = *kv;
  o.v = *vv;
  // The queries: whole rows of the joined [D, H, rows], from first_row.
  const ir::TensorDesc& view = c.tensor(*qv);
  o.q = c.node(o.views[0]).src[0];
  const ir::TensorDesc& q = c.tensor(o.q);
  if (q.is_view() || !ir::IsContiguous(q) || q.format.encoding != format::Encoding::kF32 ||
      q.ne(3) != 1 || view.ne(0) != q.ne(0) || view.ne(1) != q.ne(1) || view.ne(3) != 1 ||
      view.nb[0] != q.nb[0] || view.nb[1] != q.nb[1] || view.nb[2] != q.nb[2] || q.nb[2] == 0 ||
      view.view_offset % q.nb[2] != 0) {
    return std::nullopt;
  }
  o.first_row = view.view_offset / q.nb[2];
  o.query_rows = view.ne(2);
  // The caches: [D, Hk, C_o] from the start of a state leaf.
  for (const TensorId cache : {o.k, o.v}) {
    const ir::TensorDesc& d = c.tensor(cache);
    if (!d.is_view() || d.view_offset != 0 || d.ne(3) != 1 ||
        c.producer(d.view_of).key != Prim(Op::kState)) {
      return std::nullopt;
    }
  }
  o.mask = a.src[3];
  if (c.producer(o.mask).key != Prim(Op::kMaskBuild)) {
    return std::nullopt;
  }
  const auto rows = c.OnlyReader(a.out[0]);
  if (!rows || c.node(*rows).key != Prim(Op::kReshape) || c.taken(*rows)) {
    return std::nullopt;
  }
  o.rows = *rows;
  return o;
}

ir::TensorDesc DescOf(format::Encoding encoding, std::initializer_list<std::uint64_t> ne) {
  ir::TensorDesc d;
  d.format = ir::Dense(encoding, ir::ShapeOf(ne));
  d.nb = ir::NaturalStrides(d.format);
  return d;
}

// Whether every layer's owner-root nodes (groups of up to four owners)
// have an implementation, each owner's cache at its own width (`bounded`)
// or every cache at its kind's.
bool OwnerNodesImplemented(const MatchContext& c, const OwnerPlan& plan, bool bounded) {
  const std::uint32_t n = plan.owners;
  for (const OwnerPlan::Layer& layer : plan.layers) {
    const OwnerPlan::Kind& kind = plan.kinds.at(layer.kind);
    for (std::uint32_t first = 0; first < n; first += kOwnerGroup) {
      const std::uint32_t active = std::min(kOwnerGroup, n - first);
      std::vector<ir::TensorDesc> descs;
      descs.push_back(DescOf(format::Encoding::kF32, {plan.kd, plan.rows, plan.heads, active}));
      descs.push_back(DescOf(format::Encoding::kF16, {kind.cells, kind.rows, 1, active}));
      for (const auto* caches : {&layer.k, &layer.v}) {
        for (std::uint32_t o = first; o < first + active; ++o) {
          const ir::TensorDesc& view = c.tensor(caches->at(o));
          descs.push_back(DescOf(view.format.encoding,
                                 {view.ne(0), bounded ? view.ne(2) : kind.cells, view.ne(1), 1}));
        }
      }
      const ir::TensorDesc out =
          DescOf(format::Encoding::kF32, {plan.vd, plan.heads, plan.rows, active});
      std::vector<const ir::TensorDesc*> sources;
      sources.reserve(descs.size());
      for (const ir::TensorDesc& d : descs) {
        sources.push_back(&d);
      }
      const std::array<const ir::TensorDesc*, 1> outputs = {&out};
      const ir::ParamBytes params = ir::ToBytes(
          ir::OwnerAttnParams{.attn = plan.attn, .owners = active, .cohort = n, .offset = first});
      if (c.view()
              .Candidates(ir::Fused(FusedKind::kOwnerAttention), sources, outputs, params)
              .empty()) {
        return false;
      }
    }
  }
  return true;
}

// The whole step's owner attention, anchored at its first attention (see
// OwnerPlan): every attention of the step in a layer's group of all the
// step's owners, one row each with every owner returning it (the builders'
// owner decode) or two owners of equal rows reading equal cells (owner
// prefill); every layer's owner-root nodes implemented with each cache at
// its own width, else with every cache padded to its mask's.
std::optional<Match> OwnerAttentionMatch(const MatchContext& c, NodeId anchor) {
  const ir::Graph& g = c.graph();
  for (std::uint32_t i = 0; i < anchor.v; ++i) {
    if (g.nodes()[i].key == Prim(Op::kAttention)) {
      return std::nullopt;  // not the first
    }
  }
  auto plan = std::make_shared<OwnerPlan>();
  std::vector<std::vector<NodeId>> groups;  // each layer's nodes, in order
  const auto count = static_cast<std::uint32_t>(g.nodes().size());
  for (std::uint32_t at = anchor.v; at < count;) {
    if (g.nodes()[at].key != Prim(Op::kAttention)) {
      ++at;
      continue;
    }
    // A layer's first owner, and each next one's rows joined to those
    // before it.
    const auto first = OwnerAttentionAt(c, NodeId{at});
    if (!first || first->owner != 0 || first->first_row != 0) {
      return std::nullopt;
    }
    std::vector<OwnerAttention> owners = {*first};
    TensorId joined = c.node(first->rows).out[0];
    std::vector<NodeId> joins;
    for (;;) {
      const auto next = c.OnlyReader(joined);
      if (!next || c.taken(*next)) {
        break;
      }
      const ir::Node& cat = c.node(*next);
      if (cat.key != Prim(Op::kConcat) || cat.Params<ir::ConcatParams>().dim != 1 ||
          cat.src[0] != joined) {
        break;
      }
      const NodeId rows = g.producer(cat.src[1]);
      if (c.node(rows).key != Prim(Op::kReshape) || c.OnlyReader(cat.src[1]) != *next) {
        return std::nullopt;
      }
      const auto o = OwnerAttentionAt(c, g.producer(c.node(rows).src[0]));
      const OwnerAttention& prev = owners.back();
      if (!o || o->rows != rows || o->owner != prev.owner + 1 || o->q != first->q ||
          o->first_row != prev.first_row + prev.query_rows) {
        return std::nullopt;
      }
      owners.push_back(*o);
      joins.push_back(*next);
      joined = cat.out[0];
    }
    const auto n = static_cast<std::uint32_t>(owners.size());
    const ir::TensorDesc& q = c.tensor(first->q);
    if (n < 2 || n > ir::kMaxOwnerCohort ||
        owners.back().first_row + owners.back().query_rows != q.ne(2)) {
      return std::nullopt;
    }
    // One form for every layer: the owners, their rows, the heads, the
    // attention's parameters.
    const ir::Node& a0 = c.node(first->attention);
    const auto attn = a0.Params<ir::AttnParams>();
    if (plan->owners == 0) {
      plan->owners = n;
      plan->rows = first->query_rows;
      plan->kd = q.ne(0);
      plan->heads = q.ne(1);
      plan->vd = c.tensor(first->v).ne(0);
      plan->kv_heads = c.tensor(first->k).ne(1);
      plan->attn = attn;
    } else if (plan->owners != n || plan->kd != q.ne(0) || plan->heads != q.ne(1) ||
               std::memcmp(&plan->attn, a0.params.data(), sizeof(attn)) != 0) {
      return std::nullopt;
    }
    OwnerPlan::Layer layer;
    layer.q = first->q;
    layer.out = joined;
    layer.component = a0.component;
    std::vector<TensorId> masks;
    std::vector<NodeId> group;
    for (const OwnerAttention& o : owners) {
      const ir::TensorDesc& k = c.tensor(o.k);
      const ir::TensorDesc& v = c.tensor(o.v);
      const ir::TensorDesc& mask = c.tensor(o.mask);
      if (o.query_rows != plan->rows || k.ne(0) != plan->kd || v.ne(0) != plan->vd ||
          k.ne(1) != plan->kv_heads || v.ne(1) != plan->kv_heads || k.ne(2) != mask.ne(0) ||
          v.ne(2) != mask.ne(0) ||
          std::memcmp(&plan->attn, c.node(o.attention).params.data(), sizeof(attn)) != 0) {
        return std::nullopt;
      }
      layer.k.push_back(o.k);
      layer.v.push_back(o.v);
      masks.push_back(o.mask);
      group.insert(group.end(), o.views.begin(), o.views.end());
      group.push_back(o.attention);
      group.push_back(o.rows);
    }
    group.insert(group.end(), joins.begin(), joins.end());
    std::ranges::sort(group);
    // The layer's kind: its owners' masks.
    const auto kind = std::ranges::find(plan->kinds, masks, &OwnerPlan::Kind::masks);
    layer.kind = static_cast<std::size_t>(kind - plan->kinds.begin());
    if (kind == plan->kinds.end()) {
      OwnerPlan::Kind made;
      made.masks = masks;
      made.rows = c.tensor(masks.front()).ne(1);
      made.component = c.node(g.producer(masks.front())).component;
      for (const TensorId m : masks) {
        const ir::TensorDesc& d = c.tensor(m);
        if (d.ne(1) != made.rows || d.ne(2) != 1 || d.ne(3) != 1 ||
            d.format.encoding != format::Encoding::kF16 || c.taken(g.producer(m))) {
          return std::nullopt;
        }
        made.cells = std::max(made.cells, d.ne(0));
      }
      plan->kinds.push_back(std::move(made));
    }
    plan->layers.push_back(std::move(layer));
    groups.push_back(std::move(group));
    at = groups.back().back().v + 1;
  }
  if (plan->layers.empty()) {
    return std::nullopt;
  }
  // The form: one row an owner, each returned, or two owners of equal
  // rows and reads.
  const std::uint32_t n = plan->owners;
  const StepShape* shape = c.shape();
  const bool decode = plan->rows == 1 && shape != nullptr && shape->segments.size() == n &&
                      std::ranges::all_of(shape->segments, [](const Segment& s) {
                        return s.rows == 1 && s.outputs == 1;
                      });
  const auto equal_reads = [&](const OwnerPlan::Kind& kind) {
    return std::ranges::all_of(kind.masks,
                               [&](TensorId m) { return c.tensor(m).ne(0) == kind.cells; });
  };
  const bool prefill = n == 2 && plan->rows > 1 && std::ranges::all_of(plan->kinds, equal_reads);
  if (!decode && !prefill) {
    return std::nullopt;
  }
  plan->kind_order.resize(plan->kinds.size());
  for (std::size_t k = 0; k < plan->kinds.size(); ++k) {
    plan->kind_order[k] = k;
  }
  std::ranges::sort(plan->kind_order, {},
                    [&](std::size_t k) { return g.producer(plan->kinds[k].masks.front()); });
  // Each cache at its own width, else every cache at its kind's: the
  // zeros an owner's cache takes (K's, and V's where its head size
  // differs), owner by owner, kinds in mask order.
  const bool padded = !OwnerNodesImplemented(c, *plan, true);
  if (padded && !OwnerNodesImplemented(c, *plan, false)) {
    return std::nullopt;
  }
  std::vector<std::vector<std::pair<std::uint64_t, int>>> tails(plan->kinds.size() * n);
  if (padded) {
    for (std::uint32_t o = 0; o < n; ++o) {
      for (const std::size_t k : plan->kind_order) {
        const OwnerPlan::Kind& kind = plan->kinds[k];
        const std::uint64_t read = c.tensor(kind.masks[o]).ne(0);
        if (read == kind.cells) {
          continue;
        }
        auto& list = tails[(k * n) + o];
        for (const std::uint64_t d : {plan->kd, plan->vd}) {
          if (std::ranges::contains(list, d, &std::pair<std::uint64_t, int>::first)) {
            continue;
          }
          list.emplace_back(d, static_cast<int>(plan->tails.size()));
          plan->tails.push_back(
              {.owner = o, .kind = k, .ne = {d, plan->kv_heads, kind.cells - read}});
        }
      }
    }
  }
  const auto tail = [&](std::size_t k, std::uint32_t o, std::uint64_t d) {
    const auto& list = tails[(k * n) + o];
    const auto it = std::ranges::find(list, d, &std::pair<std::uint64_t, int>::first);
    return it == list.end() ? -1 : it->second;
  };
  for (OwnerPlan::Layer& layer : plan->layers) {
    for (std::uint32_t o = 0; o < n; ++o) {
      layer.k_tail.push_back(tail(layer.kind, o, plan->kd));
      layer.v_tail.push_back(tail(layer.kind, o, plan->vd));
    }
  }
  // Each layer's intermediates the step neither returns nor keeps, and its
  // reads ordered against the writes between them and its join's place.
  for (std::size_t l = 0; l < groups.size(); ++l) {
    for (const NodeId id : groups[l]) {
      for (const TensorId o : c.node(id).outputs()) {
        if (o != plan->layers[l].out && c.external(o)) {
          return std::nullopt;
        }
      }
    }
    Match group;
    group.nodes = groups[l];
    if (!KeepsOrder(c, group)) {
      return std::nullopt;
    }
  }
  // The match: the masks (kept, before every layer: BuildStep makes them
  // first), then every layer's group, in node order.
  std::vector<std::pair<NodeId, std::int32_t>> all;
  NodeId last_mask{0};
  for (const OwnerPlan::Kind& kind : plan->kinds) {
    for (const TensorId m : kind.masks) {
      all.emplace_back(g.producer(m), OwnerPlan::kMask);
      last_mask = std::max(last_mask, g.producer(m));
    }
  }
  if (last_mask > groups.front().front()) {
    return std::nullopt;
  }
  for (std::size_t l = 0; l < groups.size(); ++l) {
    for (const NodeId id : groups[l]) {
      all.emplace_back(id,
                       id == groups[l].back() ? static_cast<std::int32_t>(l) : OwnerPlan::kElided);
    }
  }
  std::ranges::sort(all);
  Match m;
  for (const auto& [id, role] : all) {
    m.nodes.push_back(id);
    plan->at.push_back(id == last_mask ? OwnerPlan::kLastMask : role);
  }
  m.detail = std::move(plan);
  return m;
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
    c.local().assign(1, TensorId{});
    c.local()[0] =
        b.Prepare(c.Map(m.sources.front()), format::Encoding::kQ8_1, format::Layout::kGgmlBlocks);
  }
  c.Set(product.out[0],
        b.MatMul(c.Map(product.src[0]), c.local()[0], product.Params<ir::ProductParams>()));
}

// owner_attention (OwnerPlan): the masks kept, and after the last one each
// kind's owners' masks joined on dim 3 (each padded with -inf cells to the
// widest, as the builders' join_mask) and the zero tails; each layer's
// owner-root nodes at its join's place: the joined queries [D, R, H,
// owners], each group of up to four owners a node over its views of the
// queries and the joined mask, each owner's K and V views (padded to the
// mask's width with its tail), and the groups' rows joined.
void OwnerRewrite(RewriteContext& c, const Match& m, std::size_t index) {
  const auto& plan = static_cast<const OwnerPlan&>(*m.detail);
  const std::int32_t role = plan.at.at(index);
  if (role == OwnerPlan::kElided) {
    return;
  }
  ir::Builder& b = c.builder();
  const std::size_t kinds = plan.kinds.size();
  const format::Encoding f16 = format::Encoding::kF16;
  if (role == OwnerPlan::kMask || role == OwnerPlan::kLastMask) {
    c.Copy(c.original().node(m.nodes.at(index)));
    if (role == OwnerPlan::kMask) {
      return;
    }
    c.local().assign(kinds + plan.tails.size(), TensorId{});
    for (const std::size_t k : plan.kind_order) {
      const OwnerPlan::Kind& kind = plan.kinds[k];
      b.SetComponent(kind.component);
      b.SetJoined();
      TensorId joined{};
      for (const TensorId original : kind.masks) {
        TensorId mask = c.Map(original);
        const std::uint64_t width = b.tensor(mask).ne(0);
        if (width != kind.cells) {
          const TensorId tail = b.Fill(ir::ShapeOf({kind.cells - width, kind.rows, 1, 1}),
                                       -std::numeric_limits<float>::infinity(), f16);
          mask = b.Concat(mask, tail, 0);
        }
        mask = b.Reshape(mask, ir::ShapeOf({kind.cells, kind.rows, 1, 1}));
        joined = joined.valid() ? b.Concat(joined, mask, 3) : mask;
      }
      c.local()[k] = joined;
    }
    for (std::size_t t = 0; t < plan.tails.size(); ++t) {
      const OwnerPlan::Tail& tail = plan.tails[t];
      b.SetComponent(plan.kinds[tail.kind].component);
      b.SetJoined();
      c.local()[kinds + t] =
          b.Fill(ir::ShapeOf({tail.ne[0], tail.ne[1], tail.ne[2], 1}), 0.0F, f16);
    }
    return;
  }
  const OwnerPlan::Layer& layer = plan.layers.at(static_cast<std::size_t>(role));
  const OwnerPlan::Kind& kind = plan.kinds.at(layer.kind);
  const std::uint32_t n = plan.owners;
  b.SetComponent(layer.component);
  b.SetJoined();
  const TensorId packed = b.Permute(
      b.Reshape(c.Map(layer.q), ir::ShapeOf({plan.kd, plan.heads, plan.rows, n})), {0, 2, 1, 3});
  const TensorId mask = c.local().at(layer.kind);
  const bool groups = n > kOwnerGroup;
  TensorId joined{};
  for (std::uint32_t first = 0; first < n; first += kOwnerGroup) {
    const std::uint32_t active = std::min(kOwnerGroup, n - first);
    TensorId q = packed;
    TensorId masks = mask;
    if (groups) {
      const auto qn = b.tensor(packed).nb;
      q = b.View(packed, ir::ShapeOf({plan.kd, plan.rows, plan.heads, active}), qn, first * qn[3]);
      const auto mn = b.tensor(mask).nb;
      masks = b.View(mask, ir::ShapeOf({kind.cells, kind.rows, 1, active}), mn, first * mn[3]);
    }
    std::vector<TensorId> sources = {q, masks};
    for (const bool values : {false, true}) {
      for (std::uint32_t o = first; o < first + active; ++o) {
        const ir::TensorDesc& view = c.original().tensor(values ? layer.v[o] : layer.k[o]);
        const int tail = values ? layer.v_tail[o] : layer.k_tail[o];
        b.SetOwner(static_cast<std::uint8_t>(o));
        TensorId cache = b.View(c.Map(view.view_of), view.shape(), view.nb, view.view_offset);
        if (tail >= 0) {
          cache = b.Concat(cache, c.local().at(kinds + static_cast<std::size_t>(tail)), 2);
        }
        sources.push_back(b.Permute(cache, {0, 2, 1, 3}));
        b.SetJoined();
      }
    }
    const TensorId attention = b.Fused(
        FusedKind::kOwnerAttention, sources,
        ir::OwnerAttnParams{.attn = plan.attn, .owners = active, .cohort = n, .offset = first})[0];
    const TensorId rows =
        b.Reshape(attention, ir::ShapeOf({plan.vd * plan.heads, active * plan.rows}));
    joined = joined.valid() ? b.Concat(joined, rows, 1) : rows;
  }
  c.Set(layer.out, joined);
}

constexpr std::array kNormAnchors = {Op::kRmsNorm};
constexpr std::array kAttentionAnchors = {Op::kAttention};
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
    Pattern{.name = "owner_attention",
            .result = ir::Fused(FusedKind::kOwnerAttention),
            .anchors = kAttentionAnchors,
            .match = &OwnerAttentionMatch,
            .own_checks = true,
            .rewrite = &OwnerRewrite},
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

MatchContext::MatchContext(const ir::Graph& graph, const SelectionView& view,
                           const StepShape* shape)
    : graph_(graph),
      view_(view),
      shape_(shape),
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
                                                        const SelectionView& view,
                                                        const StepShape* shape) {
  MatchContext context(graph, view, shape);
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
      if (!m || (!p.own_checks && ((p.result.op == Op::kFused && !ElidesF32(context, *m)) ||
                                   !KeepsOrder(context, *m)))) {
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
  std::vector<std::vector<TensorId>> locals(found.size());
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
