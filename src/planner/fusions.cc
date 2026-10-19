// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/fusions.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "planner/pattern.h"
#include "planner/select.h"
#include "planner/step_shape.h"

namespace llmp::planner::fusions {
namespace {

using format::Encoding;
using ir::FusedKind;
using ir::NodeId;
using ir::Op;
using ir::TensorId;

constexpr ir::OpKey Prim(Op op) { return ir::Primitive(op); }

bool Is(const MatchContext& c, NodeId n, Op op) { return n.valid() && c.node(n).key == Prim(op); }

// The node making `t`.
NodeId Made(const MatchContext& c, TensorId t) {
  return t.valid() ? c.graph().producer(t) : NodeId{};
}

// The one free reader of `t` (one source of one node, and the step does
// not return or keep `t` itself; a kept view of its root is the match's
// to stand for, Building::Closed).
std::optional<NodeId> Only(const MatchContext& c, TensorId t) {
  if (!t.valid()) {
    return std::nullopt;
  }
  const auto readers = c.readers(t);
  if (readers.size() != 1 || c.kept(t) || c.taken(readers.front())) {
    return std::nullopt;
  }
  return readers.front();
}

// The one free reader of `t` when it is a node of `op`.
std::optional<NodeId> OnlyOf(const MatchContext& c, TensorId t, Op op) {
  const auto r = Only(c, t);
  if (!r || !Is(c, *r, op)) {
    return std::nullopt;
  }
  return r;
}

bool IsLeaf(const MatchContext& c, TensorId t) {
  if (!t.valid()) {
    return false;
  }
  const Op op = c.producer(t).key.op;
  return op == Op::kWeight || op == Op::kState || op == Op::kInput;
}

bool Weight(const MatchContext& c, TensorId t) {
  return t.valid() && c.producer(t).key == Prim(Op::kWeight);
}

// A parameter: a weight, or what a component derives from one alone (a
// convert, scale or unary of it: components/defs.h F32Parameters' F32 norm
// weights, zero-centered norms' (1 + w), A's -exp(A_log)), which a fused
// kernel reads as it reads a weight. Source-encoded checkpoints (D-108)
// carry their norms and A unfolded, so their fast forms see these.
bool Parameter(const MatchContext& c, TensorId t, int depth = 4) {
  if (!t.valid()) return false;
  const ir::Node& p = c.producer(t);
  if (p.key == Prim(Op::kWeight)) return true;
  if (depth == 0 || p.nsrc != 1 ||
      (p.key != Prim(Op::kConvert) && p.key != Prim(Op::kScale) && p.key != Prim(Op::kUnary)))
    return false;
  return Parameter(c, p.src[0], depth - 1);
}

float ScaleOf(const MatchContext& c, NodeId n) { return c.node(n).Params<ir::ScaleParams>().scale; }

ir::UnaryFn FnOf(const MatchContext& c, NodeId n) { return c.node(n).Params<ir::UnaryParams>().fn; }

bool IsUnary(const MatchContext& c, NodeId n, ir::UnaryFn fn) {
  return Is(c, n, Op::kUnary) && FnOf(c, n) == fn;
}

// A tensor description as a candidate query takes it.
ir::TensorDesc Desc(const format::TensorFormat& f) {
  ir::TensorDesc d;
  d.format = f;
  d.nb = ir::NaturalStrides(f);
  return d;
}

ir::TensorDesc DenseDesc(Encoding e, std::initializer_list<std::uint64_t> ne) {
  return Desc(ir::Dense(e, ir::ShapeOf(ne)));
}

// Whether the selection has a candidate for a node of `key`.
bool Has(const MatchContext& c, ir::OpKey key, std::span<const ir::TensorDesc* const> sources,
         std::span<const ir::TensorDesc* const> outputs, const ir::ParamBytes& params,
         bool state_span = false) {
  return !c.view().Candidates(key, sources, outputs, params, state_span).empty();
}

// The sources' descriptions (absent ones null).
std::vector<const ir::TensorDesc*> Descs(const MatchContext& c, std::span<const TensorId> ids) {
  std::vector<const ir::TensorDesc*> out;
  for (const TensorId t : ids) {
    out.push_back(t.valid() ? &c.tensor(t) : nullptr);
  }
  return out;
}

// The builder's context for a node made in another's place: its component
// and owner.
void ContextOf(RewriteContext& c, const ir::Node& node) {
  ir::Builder& b = c.builder();
  b.SetComponent(node.component);
  if (node.batch == ir::Batch::kPerOwner && node.nout > 0) {
    b.SetOwner(c.original().tensor(node.out[0]).segment);
  } else {
    b.SetJoined();
  }
}

// What a match emits at its nodes' places: `at[i]` the emission of match
// node i (or -1), each `emit`'s.
struct Emissions final : MatchDetail {
  std::vector<int> at;
  std::function<void(RewriteContext&, int)> emit;
};

// A match under construction: its nodes, and where it emits.
class Building {
 public:
  explicit Building(const MatchContext& c) : c_(c) {}
  // Adds `n` (a node no match took); false if it cannot.
  bool Add(NodeId n) {
    if (!n.valid() || c_.taken(n)) {
      return false;
    }
    if (!std::ranges::contains(nodes_, n)) {
      nodes_.push_back(n);
    }
    return true;
  }
  bool Add(std::initializer_list<NodeId> ns) {
    return std::ranges::all_of(ns, [&](NodeId n) { return Add(n); });
  }
  bool Has(NodeId n) const { return std::ranges::contains(nodes_, n); }
  // Pulls the leaves among `sources` made after `place` into the match
  // (the replacement made at `place` reads them): each a leaf whose every
  // reader is a node of the match.
  bool Pull(NodeId place, std::span<const TensorId> sources) {
    for (const TensorId t : sources) {
      if (!t.valid() || !IsLeaf(c_, t)) {
        continue;
      }
      const NodeId leaf = Made(c_, t);
      if (leaf < place) {
        continue;
      }
      for (const NodeId r : c_.readers(t)) {
        if (!Has(r)) {
          return false;
        }
      }
      if (!Add(leaf)) {
        return false;
      }
      pulled_.push_back(leaf);
    }
    return true;
  }
  const std::vector<NodeId>& pulled() const { return pulled_; }
  // Whether every reader of every output of the match's nodes but `outs`
  // is a node of the match, and none is the step's (or the root of what the
  // step keeps, but through one of `outs`: a verify's save of a view of a
  // result the match gives).
  bool Closed(std::span<const TensorId> outs) const {
    for (const NodeId n : nodes_) {
      for (const TensorId o : c_.node(n).outputs()) {
        if (std::ranges::contains(outs, o)) {
          continue;
        }
        if (Escapes(o, outs)) {
          return false;
        }
        for (const NodeId r : c_.readers(o)) {
          if (!Has(r)) {
            return false;
          }
        }
      }
    }
    return true;
  }
  // Whether the step returns or keeps `o`, or a view of it (a root's) not
  // made from one of `outs`.
  bool Escapes(TensorId o, std::span<const TensorId> outs) const {
    if (c_.kept(o)) {
      return true;
    }
    if (c_.tensor(o).is_view() || !c_.external(o)) {
      return false;
    }
    const ir::Graph& g = c_.graph();
    for (const auto list : {g.outputs(), g.keep()}) {
      for (const TensorId k : list) {
        if (g.root(k) != o) continue;
        bool through = false;
        for (TensorId t = k; t.valid() && !through;) {
          through = std::ranges::contains(outs, t);
          const ir::Node& p = c_.producer(t);
          if (!c_.tensor(t).is_view() || p.nsrc == 0) break;
          t = p.src[0];
        }
        if (!through) return true;
      }
    }
    return false;
  }
  // The match: `emit` at the places `places` give (node, emission).
  std::optional<Match> Finish(std::vector<std::pair<NodeId, int>> places,
                              std::function<void(RewriteContext&, int)> emit,
                              std::vector<TensorId> sources = {}) {
    // These compound substitutions have no transfer proof for an already
    // constrained product recipe. Keep its explicit preparation/product chain;
    // neither another fusion nor a different decomposition may erase the seal.
    for (const auto id : nodes_) {
      const auto& node = c_.node(id);
      if ((node.key.op == Op::kMatMul || node.key.op == Op::kMatMulId ||
           node.key == ir::Fused(FusedKind::kGroupedMatMul)) &&
          node.Params<ir::ProductParams>().arithmetic != ir::ProductArithmetic::kImplementation)
        return std::nullopt;
    }
    std::ranges::sort(nodes_);
    // These patterns emit at several places and bypass the primitive
    // patterns' order check. Conservatively reject an intervening access
    // that could change a cached read or observe a moved write.
    const ir::Graph& g = c_.graph();
    std::set<TensorId> reads, writes;
    const auto written = [&](const ir::Node& n, auto&& f) {
      if (n.key == Prim(Op::kCacheStore)) f(g.root(n.src[0]));
      if (n.key == ir::Fused(FusedKind::kGdnStep) && n.src[5].valid()) f(g.root(n.src[5]));
      if (n.key == ir::Fused(FusedKind::kQsaPool) && n.src[3].valid()) f(g.root(n.src[3]));
      for (std::size_t i = 0; i < n.nout; ++i) {
        if (const auto alias = ir::AliasedSource(n.key, i); alias && n.src.at(*alias).valid())
          f(g.root(n.src.at(*alias)));
      }
    };
    for (const NodeId id : nodes_) {
      const ir::Node& n = g.node(id);
      for (const TensorId s : n.sources()) {
        if (s.valid()) reads.insert(g.root(s));
      }
      written(n, [&](TensorId t) { writes.insert(t); });
    }
    for (std::uint32_t k = nodes_.front().v + 1; k < nodes_.back().v; ++k) {
      if (Has(NodeId{k})) continue;
      const ir::Node& n = g.node(NodeId{k});
      bool hazard = false;
      written(n, [&](TensorId t) { hazard = hazard || reads.contains(t) || writes.contains(t); });
      for (const TensorId s : n.sources()) {
        hazard = hazard || (s.valid() && writes.contains(g.root(s)));
      }
      if (hazard) return std::nullopt;
    }
    auto detail = std::make_shared<Emissions>();
    detail->at.assign(nodes_.size(), -1);
    for (const auto& [node, id] : places) {
      const auto at = std::ranges::find(nodes_, node);
      detail->at.at(static_cast<std::size_t>(at - nodes_.begin())) = id;
    }
    detail->emit = std::move(emit);
    return Match{.nodes = nodes_, .sources = std::move(sources), .detail = std::move(detail)};
  }

 private:
  const MatchContext& c_;
  std::vector<NodeId> nodes_;
  std::vector<NodeId> pulled_;
};

// Copies the pulled leaves (Building::Pull) at the current place.
void PullLeaves(RewriteContext& c, std::span<const NodeId> pulled) {
  for (const NodeId n : pulled) {
    const ir::Node& leaf = c.original().node(n);
    if (!c.Map(leaf.out[0]).valid()) {
      c.Copy(leaf);
    }
  }
}

// Whether no node strictly between `from` and `to` (outside the match)
// writes or reads `root`'s bytes: a store moved from `from` to `to`.
bool Undisturbed(const MatchContext& c, const Building& m, NodeId from, NodeId to, TensorId root) {
  const ir::Graph& g = c.graph();
  for (std::uint32_t k = from.v + 1; k < to.v; ++k) {
    const NodeId id{k};
    if (m.Has(id)) {
      continue;
    }
    const ir::Node& n = g.node(id);
    for (const TensorId s : n.sources()) {
      if (s.valid() && g.root(s) == root) {
        return false;
      }
    }
    for (const TensorId o : n.outputs()) {
      if (g.root(o) == root && o != root) {
        return false;
      }
    }
  }
  return true;
}

// Whether caching a read at `from` is safe until `to`. Pure views do
// not write; stores and fused in-place outputs do.
bool Unwritten(const ir::Graph& g, NodeId from, NodeId to, TensorId tensor) {
  const TensorId root = g.root(tensor);
  for (std::uint32_t k = from.v + 1; k < to.v; ++k) {
    const ir::Node& n = g.node(NodeId{k});
    if (n.key == Prim(Op::kCacheStore) && g.root(n.src[0]) == root) return false;
    for (std::size_t i = 0; i < n.nout; ++i) {
      if (const auto alias = ir::AliasedSource(n.key, i);
          alias && n.src.at(*alias).valid() && g.root(n.src.at(*alias)) == root) {
        return false;
      }
    }
  }
  return true;
}

bool Unwritten(const MatchContext& c, NodeId from, NodeId to, TensorId tensor) {
  return Unwritten(c.graph(), from, to, tensor);
}

// ------------------------------------------------------- hyper-connections

// hc.combine's nodes (components/hc.cc CombineEmit): the block's rows
// repeated into every stream, its logits' weights 2 · sigmoid(logits /
// hc), the add onto the streams.
struct Combine {
  NodeId ro, rp, sc1, sg, sc2, rw, mu, add;
  TensorId streams;  // the streams before it
  TensorId out;      // the block's output [W, T]
  TensorId logits;   // [hc, T]
  std::uint64_t hc = 0;
};

std::optional<Combine> CombineAt(const MatchContext& c, NodeId add) {
  if (!Is(c, add, Op::kAdd) || c.taken(add)) {
    return std::nullopt;
  }
  const ir::Node& a = c.node(add);
  Combine m{};
  m.add = add;
  m.streams = a.src[0];
  m.mu = Made(c, a.src[1]);
  if (!Is(c, m.mu, Op::kMul) || Only(c, a.src[1]) != add) {
    return std::nullopt;
  }
  const ir::Node& mu = c.node(m.mu);
  m.rp = Made(c, mu.src[0]);
  m.rw = Made(c, mu.src[1]);
  if (!Is(c, m.rp, Op::kRepeat) || !Is(c, m.rw, Op::kReshape) || Only(c, mu.src[0]) != m.mu ||
      Only(c, mu.src[1]) != m.mu) {
    return std::nullopt;
  }
  m.ro = Made(c, c.node(m.rp).src[0]);
  if (!Is(c, m.ro, Op::kReshape) || Only(c, c.node(m.rp).src[0]) != m.rp) {
    return std::nullopt;
  }
  m.out = c.node(m.ro).src[0];
  m.sc2 = Made(c, c.node(m.rw).src[0]);
  if (!Is(c, m.sc2, Op::kScale) || ScaleOf(c, m.sc2) != 2.0F ||
      c.node(m.sc2).Params<ir::ScaleParams>().bias != 0.0F ||
      Only(c, c.node(m.rw).src[0]) != m.rw) {
    return std::nullopt;
  }
  m.sg = Made(c, c.node(m.sc2).src[0]);
  if (!IsUnary(c, m.sg, ir::UnaryFn::kSigmoid) || Only(c, c.node(m.sc2).src[0]) != m.sc2) {
    return std::nullopt;
  }
  m.sc1 = Made(c, c.node(m.sg).src[0]);
  if (!Is(c, m.sc1, Op::kScale) || Only(c, c.node(m.sg).src[0]) != m.sg) {
    return std::nullopt;
  }
  m.logits = c.node(m.sc1).src[0];
  const ir::TensorDesc& sd = c.tensor(m.streams);
  const ir::TensorDesc& od = c.tensor(m.out);
  const ir::TensorDesc& ld = c.tensor(m.logits);
  m.hc = sd.ne(1);
  if (Only(c, m.logits) != m.sc1 || sd.format.encoding != Encoding::kF32 || m.hc < 2 ||
      od.ne(0) != sd.ne(0) || od.ne(1) != sd.ne(2) || ld.ne(0) != m.hc || ld.ne(1) != sd.ne(2) ||
      ScaleOf(c, m.sc1) != 1.0F / static_cast<float>(m.hc) ||
      c.node(m.sc1).Params<ir::ScaleParams>().bias != 0.0F) {
    return std::nullopt;
  }
  for (const NodeId n : {m.ro, m.rp, m.sc1, m.sg, m.sc2, m.rw, m.mu}) {
    if (c.taken(n)) {
      return std::nullopt;
    }
  }
  return m;
}

// hc.mix's nodes (components/hc.cc MixEmit): each stream's norm and its
// weight (xn), the low-rank gate, the gated streams' fold.
struct Mix {
  NodeId norm, r1, m1;       // xn
  NodeId d, s1, u1, p;       // the products
  NodeId u2, g, r2, s2;      // the fold's sigmoid, product, reshape and scale
  std::vector<NodeId> fold;  // its views, cont and adds
  NodeId inject{};           // the next combine's product of xn, if any
  TensorId x, w_norm, w_down, w_up, w_inject;
  std::uint64_t width = 0, hc = 0, rows = 0;
  float eps = 0;
};

std::optional<Mix> MixAt(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kRmsNorm) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& n = c.node(anchor);
  Mix m{};
  m.norm = anchor;
  m.x = n.src[0];
  m.eps = n.Params<ir::NormParams>().eps;
  const ir::TensorDesc& xd = c.tensor(m.x);
  m.width = xd.ne(0);
  m.hc = xd.ne(1);
  m.rows = xd.ne(2);
  if (xd.format.encoding != Encoding::kF32 || m.hc < 2 || xd.ne(3) != 1) {
    return std::nullopt;
  }
  const std::uint64_t hw = m.width * m.hc;
  const auto r1 = OnlyOf(c, n.out[0], Op::kReshape);
  if (!r1 || c.tensor(c.node(*r1).out[0]).ne(0) != hw) {
    return std::nullopt;
  }
  m.r1 = *r1;
  const auto m1 = OnlyOf(c, c.node(*r1).out[0], Op::kMul);
  if (!m1 || c.node(*m1).src[0] != c.node(*r1).out[0] || !Parameter(c, c.node(*m1).src[1])) {
    return std::nullopt;
  }
  m.m1 = *m1;
  m.w_norm = c.node(*m1).src[1];
  const TensorId xn = c.node(*m1).out[0];
  if (c.external(xn)) {
    return std::nullopt;
  }
  // Its readers: the down product, the fold's product, and perhaps the
  // next combine's inject product.
  for (const NodeId r : c.readers(xn)) {
    if (c.taken(r)) {
      return std::nullopt;
    }
    const ir::Node& rn = c.node(r);
    if (rn.key == Prim(Op::kMul) && rn.src[0] == xn) {
      m.g = r;
      continue;
    }
    if (rn.key != Prim(Op::kMatMul) || rn.src[1] != xn || !Weight(c, rn.src[0])) {
      return std::nullopt;
    }
    const auto s = OnlyOf(c, rn.out[0], Op::kScale);
    if (!s) {
      return std::nullopt;
    }
    const auto u = Only(c, c.node(*s).out[0]);
    if (u && IsUnary(c, *u, ir::UnaryFn::kSilu) && !m.d.valid()) {
      m.d = r;
      m.s1 = *s;
      m.u1 = *u;
    } else if (!m.inject.valid() && c.tensor(rn.out[0]).ne(0) == m.hc) {
      m.inject = r;
    } else {
      return std::nullopt;
    }
  }
  if (!m.d.valid() || !m.g.valid()) {
    return std::nullopt;
  }
  m.w_down = c.node(m.d).src[0];
  const auto p = OnlyOf(c, c.node(m.u1).out[0], Op::kMatMul);
  if (!p || c.node(*p).src[1] != c.node(m.u1).out[0] || !Weight(c, c.node(*p).src[0])) {
    return std::nullopt;
  }
  m.p = *p;
  m.w_up = c.node(*p).src[0];
  const auto u2 = Only(c, c.node(*p).out[0]);
  if (!u2 || !IsUnary(c, *u2, ir::UnaryFn::kSigmoid) || Only(c, c.node(*u2).out[0]) != m.g ||
      c.node(m.g).src[1] != c.node(*u2).out[0]) {
    return std::nullopt;
  }
  m.u2 = *u2;
  const auto r2 = OnlyOf(c, c.node(m.g).out[0], Op::kReshape);
  if (!r2) {
    return std::nullopt;
  }
  m.r2 = *r2;
  const ir::TensorDesc& shaped = c.tensor(c.node(*r2).out[0]);
  if (shaped.shape().ne != ir::ShapeOf({m.width, m.hc, m.rows}).ne) {
    return std::nullopt;
  }
  // The fold: a view a stream, the first made packed, the others added in
  // order, then the scale.
  const TensorId gated = c.node(*r2).out[0];
  if (c.external(gated)) {
    return std::nullopt;
  }
  std::vector<NodeId> views;
  for (const NodeId r : c.readers(gated)) {
    if (!Is(c, r, Op::kView) || c.taken(r)) {
      return std::nullopt;
    }
    views.push_back(r);
  }
  if (views.size() != m.hc) {
    return std::nullopt;
  }
  std::ranges::sort(views);
  for (std::size_t h = 0; h < views.size(); ++h) {
    const ir::TensorDesc& vd = c.tensor(c.node(views[h]).out[0]);
    if (vd.shape().ne != ir::ShapeOf({m.width, m.rows}).ne ||
        vd.view_offset != shaped.view_offset + h * shaped.nb[1] || vd.nb[0] != shaped.nb[0] ||
        vd.nb[1] != shaped.nb[2]) {
      return std::nullopt;
    }
  }
  const auto c0 = OnlyOf(c, c.node(views[0]).out[0], Op::kCont);
  if (!c0) {
    return std::nullopt;
  }
  m.fold = views;
  m.fold.push_back(*c0);
  TensorId acc = c.node(*c0).out[0];
  for (std::size_t k = 1; k < views.size(); ++k) {
    const auto add = OnlyOf(c, acc, Op::kAdd);
    if (!add || c.node(*add).src[1] != c.node(views[k]).out[0] ||
        Only(c, c.node(views[k]).out[0]) != *add) {
      return std::nullopt;
    }
    m.fold.push_back(*add);
    acc = c.node(*add).out[0];
  }
  const auto s2 = OnlyOf(c, acc, Op::kScale);
  if (!s2) {
    return std::nullopt;
  }
  m.s2 = *s2;
  const ir::ScaleParams fold_scale = c.node(*s2).Params<ir::ScaleParams>();
  if (fold_scale.scale != 1.0F / static_cast<float>(m.hc) || fold_scale.bias != 0.0F) {
    return std::nullopt;
  }
  if (m.inject.valid()) {
    m.w_inject = c.node(m.inject).src[0];
  }
  return m;
}

// Whether the mixed stream feeds a router (a product whose only reader is
// routing): an FFN's mix.
bool FeedsRouter(const MatchContext& c, TensorId mixed) {
  for (const NodeId r : c.readers(mixed)) {
    if (!Is(c, r, Op::kMatMul)) {
      continue;
    }
    for (const NodeId q : c.readers(c.node(r).out[0])) {
      if (Is(c, q, Op::kRoute)) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

// hc_prep: a mix as the fast form's kernels run it (llmp.hc.prep, the BF16
// products, llmp.hc.lo, llmp.hc.mix_bf16): the streams normed in BF16, with
// the combine before it (when the mix reads its add) and the next combine's
// logits (its inject product); everything made at xn's place.
std::optional<Match> HcPrepMatch(const MatchContext& c, NodeId anchor) {
  const auto mix = MixAt(c, anchor);
  if (!mix) {
    return std::nullopt;
  }
  const Mix& m = *mix;
  if (c.node(m.s1).Params<ir::ScaleParams>().bias != 0.0F) return std::nullopt;
  Building b(c);
  std::optional<Combine> combine;
  if (const NodeId e = Made(c, m.x); Is(c, e, Op::kAdd)) {
    combine = CombineAt(c, e);
  }
  TensorId x = m.x;
  if (combine) {
    // Its streams read nowhere before the replacement's place.
    for (const NodeId r : c.readers(m.x)) {
      if (r != m.norm && r < m.m1) {
        return std::nullopt;
      }
    }
    if (!b.Add({combine->ro, combine->rp, combine->sc1, combine->sg, combine->sc2, combine->rw,
                combine->mu, combine->add})) {
      return std::nullopt;
    }
    x = combine->streams;
  }
  if (!b.Add({m.norm, m.r1, m.m1, m.d, m.s1, m.u1, m.p, m.u2, m.g, m.r2, m.s2})) {
    return std::nullopt;
  }
  for (const NodeId f : m.fold) {
    if (!b.Add(f)) {
      return std::nullopt;
    }
  }
  if (m.inject.valid() && !b.Add(m.inject)) {
    return std::nullopt;
  }
  const std::array<TensorId, 4> leaves = {m.w_norm, m.w_down, m.w_up, m.w_inject};
  if (!b.Pull(m.m1, leaves)) {
    return std::nullopt;
  }
  const TensorId mixed = c.node(m.s2).out[0];
  std::vector<TensorId> outs = {mixed};
  if (combine) outs.push_back(c.node(combine->add).out[0]);
  if (m.inject.valid()) outs.push_back(c.node(m.inject).out[0]);
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  // The candidates: the prep, the products over BF16, the low-rank
  // activation, the fold (prepared, and rounded for a router, past the
  // vector products' rows where its implementation takes them).
  const std::uint64_t hw = m.width * m.hc;
  const std::uint64_t rank = c.tensor(m.w_down).ne(1);
  const ir::HcNormPostParams prep_params{.eps = m.eps, .streams = static_cast<std::uint32_t>(m.hc)};
  const std::vector<TensorId> prep_src = {x, m.w_norm, m.w_inject,
                                          combine ? combine->out : TensorId{},
                                          combine ? combine->logits : TensorId{}};
  const ir::TensorDesc xn_d = DenseDesc(Encoding::kBF16, {hw, m.rows});
  const ir::TensorDesc logits_d = DenseDesc(Encoding::kF32, {m.hc, m.rows});
  const ir::TensorDesc streams_d = DenseDesc(Encoding::kF32, {m.width, m.hc, m.rows});
  std::vector<const ir::TensorDesc*> prep_out = {&xn_d};
  if (m.inject.valid()) prep_out.push_back(&logits_d);
  if (combine) prep_out.push_back(&streams_d);
  if (!Has(c, ir::Fused(FusedKind::kHcPrep), Descs(c, prep_src), prep_out,
           ir::ToBytes(prep_params))) {
    return std::nullopt;
  }
  const ir::TensorDesc lo_d = DenseDesc(Encoding::kF32, {rank, m.rows});
  const ir::TensorDesc lo_b = DenseDesc(Encoding::kBF16, {rank, m.rows});
  const ir::TensorDesc gate_d = DenseDesc(Encoding::kBF16, {hw, m.rows});
  {
    const std::array<const ir::TensorDesc*, 2> s = {&c.tensor(m.w_down), &xn_d};
    const std::array<const ir::TensorDesc*, 1> o = {&lo_d};
    if (!Has(c, Prim(Op::kMatMul), s, o, c.node(m.d).params)) return std::nullopt;
  }
  const ir::ScaledUnaryParams lo_params{.scale = ScaleOf(c, m.s1), .fn = ir::UnaryFn::kSilu};
  {
    const std::array<const ir::TensorDesc*, 1> s = {&lo_d};
    const std::array<const ir::TensorDesc*, 1> o = {&lo_b};
    if (!Has(c, ir::Fused(FusedKind::kScaledUnary), s, o, ir::ToBytes(lo_params))) {
      return std::nullopt;
    }
  }
  {
    const std::array<const ir::TensorDesc*, 2> s = {&c.tensor(m.w_up), &lo_b};
    const std::array<const ir::TensorDesc*, 1> o = {&gate_d};
    if (!Has(c, Prim(Op::kMatMul), s, o, c.node(m.p).params)) return std::nullopt;
  }
  const ir::TensorDesc mixed_d = DenseDesc(Encoding::kF32, {m.width, m.rows});
  const ir::TensorDesc rows_d = Desc(format::TensorFormat{.encoding = Encoding::kMxfp8,
                                                          .layout = format::Layout::kScaleAtoms,
                                                          .shape = ir::ShapeOf({m.width, m.rows})});
  const ir::TensorDesc rounded_d = DenseDesc(Encoding::kBF16, {m.width, m.rows});
  ir::HcFoldParams fold{.streams = static_cast<std::uint32_t>(m.hc)};
  const std::array<const ir::TensorDesc*, 2> fold_src = {&xn_d, &gate_d};
  const auto fold_ok = [&](const ir::HcFoldParams& p) {
    std::vector<const ir::TensorDesc*> o = {&mixed_d};
    if (p.prepared) o.push_back(&rows_d);
    if (p.rounded) o.push_back(&rounded_d);
    return Has(c, ir::Fused(FusedKind::kHcFold), fold_src, o, ir::ToBytes(p));
  };
  // A block's mix (not the head's, which reads the output rows' streams
  // gathered) gives its rows prepared where the fold's implementation takes
  // them, as the fast form's llmp.hc.mix_bf16 does whatever reads them, and
  // rounded for a router.
  const NodeId gathered =
      Is(c, Made(c, m.x), Op::kReshape) ? Made(c, c.node(Made(c, m.x)).src[0]) : NodeId{};
  if (!Is(c, gathered, Op::kGetRows)) {
    ir::HcFoldParams wide = fold;
    wide.prepared = true;
    wide.rounded = FeedsRouter(c, mixed);
    if (fold_ok(wide)) {
      fold = wide;
    }
  }
  if (!fold_ok(fold)) {
    return std::nullopt;
  }
  const Mix mm = m;
  const std::optional<Combine> cm = combine;
  const std::vector<NodeId> pulled = b.pulled();
  auto emit = [mm, cm, pulled, prep_params, lo_params, fold, x](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    PullLeaves(r, pulled);
    ContextOf(r, g.node(mm.m1));
    ir::Builder& b = r.builder();
    const std::array<TensorId, 5> src = {r.Map(x), r.Map(mm.w_norm), r.Map(mm.w_inject),
                                         cm ? r.Map(cm->out) : TensorId{},
                                         cm ? r.Map(cm->logits) : TensorId{}};
    const std::array<Encoding, 3> enc = {Encoding::kBF16, Encoding::kF32, Encoding::kF32};
    const ir::Outputs outs = b.Fused(FusedKind::kHcPrep, src, prep_params, enc);
    const TensorId xn = outs[0];
    std::size_t k = 1;
    if (mm.inject.valid()) r.Set(g.node(mm.inject).out[0], outs[k++]);
    if (cm) r.Set(g.node(cm->add).out[0], outs[k++]);
    ContextOf(r, g.node(mm.d));
    const TensorId lo = b.MatMul(r.Map(mm.w_down), xn, g.node(mm.d).Params<ir::ProductParams>());
    const std::array<Encoding, 1> bf16 = {Encoding::kBF16};
    const TensorId lob = b.Fused(FusedKind::kScaledUnary, std::array{lo}, lo_params, bf16)[0];
    ContextOf(r, g.node(mm.p));
    ir::TensorDesc gate_req;
    gate_req.format.encoding = Encoding::kBF16;
    gate_req.format.layout = format::Layout::kDense;
    const TensorId gate = b.Emit(Prim(Op::kMatMul), g.node(mm.p).params,
                                 std::array{r.Map(mm.w_up), lob}, std::span(&gate_req, 1))[0];
    ContextOf(r, g.node(mm.s2));
    const ir::Outputs folded = b.Fused(FusedKind::kHcFold, std::array{xn, gate}, fold);
    r.Set(g.node(mm.s2).out[0], folded[0]);
    // Its prepared and rounded rows stand for nothing of the original graph
    // yet; hc_fold_inputs (round 1) gives them their readers.
  };
  return b.Finish({{m.m1, 0}}, emit);
}

// hc_mix: a mix as GGML-form checkpoints' fused graph runs it (llmp.hc.norm
// at xn's place, llmp.hc.mix at the fold's: the norm again, its products
// GGML's).
std::optional<Match> HcMixMatch(const MatchContext& c, NodeId anchor) {
  const auto mix = MixAt(c, anchor);
  if (!mix) {
    return std::nullopt;
  }
  const Mix& m = *mix;
  Building b(c);
  if (!b.Add({m.norm, m.r1, m.m1, m.u2, m.g, m.r2, m.s2})) {
    return std::nullopt;
  }
  for (const NodeId f : m.fold) {
    if (!b.Add(f)) {
      return std::nullopt;
    }
  }
  const TensorId xn = c.node(m.m1).out[0];
  const TensorId mixed = c.node(m.s2).out[0];
  // xn's readers outside: the products (copied over the new xn).
  const std::array<TensorId, 2> outs = {xn, mixed};
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  const ir::NormParams norm{.eps = m.eps};
  const ir::HcNormPostParams fold{.eps = m.eps, .streams = static_cast<std::uint32_t>(m.hc)};
  {
    const std::array<TensorId, 2> s = {m.x, m.w_norm};
    const std::array<const ir::TensorDesc*, 1> o = {&c.tensor(xn)};
    if (!Has(c, ir::Fused(FusedKind::kHcNorm), Descs(c, s), o, ir::ToBytes(norm))) {
      return std::nullopt;
    }
    const std::array<TensorId, 3> s2 = {m.x, m.w_norm, c.node(m.p).out[0]};
    const std::array<const ir::TensorDesc*, 1> o2 = {&c.tensor(mixed)};
    if (!Has(c, ir::Fused(FusedKind::kHcMixNorm), Descs(c, s2), o2, ir::ToBytes(fold))) {
      return std::nullopt;
    }
  }
  const Mix mm = m;
  auto emit = [mm, norm, fold](RewriteContext& r, int id) {
    const ir::Graph& g = r.original();
    ir::Builder& b = r.builder();
    if (id == 0) {
      ContextOf(r, g.node(mm.m1));
      r.Set(g.node(mm.m1).out[0],
            b.Fused(FusedKind::kHcNorm, std::array{r.Map(mm.x), r.Map(mm.w_norm)}, norm)[0]);
      return;
    }
    ContextOf(r, g.node(mm.s2));
    const std::array<TensorId, 3> s = {r.Map(mm.x), r.Map(mm.w_norm), r.Map(g.node(mm.p).out[0])};
    r.Set(g.node(mm.s2).out[0], b.Fused(FusedKind::kHcMixNorm, s, fold)[0]);
  };
  return b.Finish({{m.m1, 0}, {m.s2, 1}}, emit);
}

// hc_combine: a combine no mix took (before a block that reads the streams
// itself, or the head's gather) as llmp.hc.combine.
std::optional<Match> HcCombineMatch(const MatchContext& c, NodeId anchor) {
  const auto cm = CombineAt(c, anchor);
  if (!cm) {
    return std::nullopt;
  }
  Building b(c);
  if (!b.Add({cm->ro, cm->rp, cm->sc1, cm->sg, cm->sc2, cm->rw, cm->mu, cm->add})) {
    return std::nullopt;
  }
  const std::array<TensorId, 1> outs = {c.node(cm->add).out[0]};
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  const ir::HcParams params{.streams = static_cast<std::uint32_t>(cm->hc)};
  const std::array<TensorId, 3> s = {cm->streams, cm->out, cm->logits};
  const std::array<const ir::TensorDesc*, 1> o = {&c.tensor(outs[0])};
  if (!Has(c, ir::Fused(FusedKind::kHcCombine), Descs(c, s), o, ir::ToBytes(params))) {
    return std::nullopt;
  }
  const Combine m = *cm;
  auto emit = [m, params](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    ContextOf(r, g.node(m.add));
    const std::array<TensorId, 3> s = {r.Map(m.streams), r.Map(m.out), r.Map(m.logits)};
    r.Set(g.node(m.add).out[0], r.builder().Fused(FusedKind::kHcCombine, s, params)[0]);
  };
  return b.Finish({{cm->add, 0}}, emit);
}

// ------------------------------------------------------------ Gated DeltaNet

// gdn_history: an owner's history update alone (components/gdn.cc Commit:
// the state and its segment's rows into the next history, stored back to
// the state) as llmp.gdn.history.
std::optional<Match> GdnHistoryMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kCacheStore) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& st = c.node(anchor);
  const TensorId state = st.src[0];
  if (!state.valid() || c.producer(state).key != Prim(Op::kState)) {
    return std::nullopt;
  }
  const TensorId kept = st.src[1];
  if (!kept.valid()) {
    return std::nullopt;
  }
  const NodeId r3 = Made(c, kept);
  if (!Is(c, r3, Op::kReshape) || c.taken(r3) || Only(c, kept) != anchor) {
    return std::nullopt;
  }
  const TensorId c2_out = c.node(r3).src[0];
  if (!c2_out.valid()) {
    return std::nullopt;
  }
  const NodeId c2 = Made(c, c2_out);
  if (!Is(c, c2, Op::kCont) || c.taken(c2) || Only(c, c2_out) != r3) {
    return std::nullopt;
  }
  const TensorId tv_out = c.node(c2).src[0];
  if (!tv_out.valid()) {
    return std::nullopt;
  }
  const NodeId tv = Made(c, tv_out);
  if (!Is(c, tv, Op::kView) || c.taken(tv) || Only(c, tv_out) != c2) {
    return std::nullopt;
  }
  const TensorId cat_out = c.node(tv).src[0];
  if (!cat_out.valid()) {
    return std::nullopt;
  }
  const NodeId cat = Made(c, cat_out);
  if (!Is(c, cat, Op::kConcat) || c.taken(cat) || Only(c, cat_out) != tv) {
    return std::nullopt;
  }
  const ir::Node& k = c.node(cat);
  if (k.Params<ir::ConcatParams>().dim != 0) {
    return std::nullopt;
  }
  const NodeId h = Made(c, k.src[0]);
  const NodeId cont = Made(c, k.src[1]);
  if (!Is(c, h, Op::kReshape) || !Is(c, cont, Op::kCont) || c.taken(h) || c.taken(cont) ||
      Only(c, k.src[0]) != cat || Only(c, k.src[1]) != cat) {
    return std::nullopt;
  }
  if (c.node(h).src[0] != state) {
    return std::nullopt;
  }
  const NodeId tr = Made(c, c.node(cont).src[0]);
  if (!Is(c, tr, Op::kPermute) || c.taken(tr) || c.readers(c.node(cont).src[0]).size() != 1) {
    return std::nullopt;
  }
  if (c.node(tr).Params<ir::PermuteParams>().axes != std::array<std::uint8_t, 4>{1, 0, 2, 3}) {
    return std::nullopt;
  }
  const TensorId rows3 = c.node(tr).src[0];
  const ir::TensorDesc& rd = c.tensor(rows3);
  const std::uint64_t ch = rd.ne(0);
  const std::uint64_t n = rd.ne(1);
  if (ch == 0 || rd.ne(2) != 1 || rd.ne(3) != 1 || c.tensor(state).ne(0) % ch != 0 ||
      c.tensor(state).ne(1) != 1 || c.tensor(state).ne(2) != 1 || c.tensor(state).ne(3) != 1) {
    return std::nullopt;
  }
  const std::uint64_t taps = c.tensor(state).ne(0) / ch;
  if (c.tensor(k.src[0]).shape().ne != ir::ShapeOf({taps, ch, 1}).ne) {
    return std::nullopt;
  }
  const ir::TensorDesc& tail = c.tensor(c.node(tv).out[0]);
  const ir::TensorDesc& input = c.tensor(k.out[0]);
  if (tail.shape().ne != ir::ShapeOf({taps, ch, 1}).ne || tail.view_offset != n * input.nb[0] ||
      tail.nb[0] != input.nb[0] || tail.nb[1] != input.nb[1]) {
    return std::nullopt;
  }
  if (c.tensor(c.node(r3).out[0]).shape().ne != ir::ShapeOf({taps * ch, 1}).ne) {
    return std::nullopt;
  }
  Building b(c);
  if (!b.Add({h, tr, cont, cat, tv, c2, r3, anchor})) {
    return std::nullopt;
  }
  if (!st.outputs().empty() && c.external(st.out[0])) {
    return std::nullopt;
  }
  std::vector<TensorId> outs;
  if (!st.outputs().empty()) {
    outs.push_back(st.out[0]);
  }
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  const ir::GdnHistoryParams hparams{.taps = static_cast<std::uint32_t>(taps)};
  const bool with_history = n < taps;
  const ir::TensorDesc hist_d = DenseDesc(Encoding::kF32, {taps * ch, 1});
  const ir::TensorDesc rows_d = DenseDesc(rd.format.encoding, {ch, n});
  const std::array<const ir::TensorDesc*, 2> s = {&rows_d,
                                                  with_history ? &c.tensor(state) : nullptr};
  const std::array<const ir::TensorDesc*, 1> o = {&hist_d};
  if (!Has(c, ir::Fused(FusedKind::kGdnHistory), s, o, ir::ToBytes(hparams))) {
    return std::nullopt;
  }
  auto emit = [=](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    ContextOf(r, g.node(anchor));
    ir::Builder& b = r.builder();
    const TensorId rows = b.Reshape(r.Map(rows3), ir::ShapeOf({ch, n}));
    const TensorId hist =
        b.Fused(FusedKind::kGdnHistory, std::array{rows, with_history ? r.Map(state) : TensorId{}},
                hparams)[0];
    const ir::Node& st_node = g.node(anchor);
    r.Set(st_node.out[0], b.CacheStore(r.Map(state), hist, r.Map(st_node.src[2])));
  };
  return b.Finish({{anchor, 0}}, emit);
}

// gdn_conv: an owner's convolution (components/gdn.cc Emit: its input the
// history then its rows transposed, the history's store, the convolution
// and SiLU, q and k L2-normed, v; a verify's save of the three) as
// llmp.gdn.conv over the rows and the history, and the next history
// (llmp.gdn.history) stored after it.
std::optional<Match> GdnConvMatch(const MatchContext& c, NodeId anchor) {
  if (Is(c, anchor, Op::kCacheStore)) {
    return GdnHistoryMatch(c, anchor);
  }
  if (!Is(c, anchor, Op::kSsmConv) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& sc = c.node(anchor);
  const NodeId cat = Made(c, sc.src[0]);
  if (!Is(c, cat, Op::kConcat) || c.taken(cat)) {
    return std::nullopt;
  }
  const ir::Node& k = c.node(cat);
  const NodeId h = Made(c, k.src[0]);
  const NodeId cont = Made(c, k.src[1]);
  if (!Is(c, h, Op::kReshape) || !Is(c, cont, Op::kCont) || Only(c, k.src[0]) != cat ||
      Only(c, k.src[1]) != cat) {
    return std::nullopt;
  }
  const TensorId state = c.node(h).src[0];
  if (!state.valid() || c.producer(state).key != Prim(Op::kState)) {
    return std::nullopt;
  }
  const NodeId tr = Made(c, c.node(cont).src[0]);
  // (The transposition is a view of the rows a verify keeps for its save:
  // its own reader alone.)
  if (!Is(c, tr, Op::kPermute) || c.readers(c.node(cont).src[0]).size() != 1) {
    return std::nullopt;
  }
  const TensorId rows3 = c.node(tr).src[0];  // [C, n, 1]: the qkv rows (a segment's)
  const ir::TensorDesc& rd = c.tensor(rows3);
  const std::uint64_t ch = rd.ne(0);
  const std::uint64_t n = rd.ne(1);
  if (ch == 0 || rd.ne(2) != 1 || rd.ne(3) != 1 || c.tensor(state).ne(0) % ch != 0 ||
      c.tensor(state).ne(1) != 1 || c.tensor(state).ne(2) != 1 || c.tensor(state).ne(3) != 1 ||
      c.node(tr).Params<ir::PermuteParams>().axes != std::array<std::uint8_t, 4>{1, 0, 2, 3} ||
      k.Params<ir::ConcatParams>().dim != 0)
    return std::nullopt;
  const std::uint64_t taps = c.tensor(state).ne(0) / ch;
  if (c.tensor(k.src[0]).shape().ne != ir::ShapeOf({taps, ch, 1}).ne) {
    return std::nullopt;
  }
  Building b(c);
  if (!b.Add({h, tr, cont, cat, anchor})) {
    return std::nullopt;
  }
  // The history's store (none in a verify).
  NodeId store{};
  for (const NodeId r : c.readers(k.out[0])) {
    if (r == anchor) {
      continue;
    }
    const auto tv = r;
    const auto c2 = Is(c, tv, Op::kView) ? OnlyOf(c, c.node(tv).out[0], Op::kCont) : std::nullopt;
    const auto r3 = c2 ? OnlyOf(c, c.node(*c2).out[0], Op::kReshape) : std::nullopt;
    const auto st = r3 ? OnlyOf(c, c.node(*r3).out[0], Op::kCacheStore) : std::nullopt;
    if (!st || store.valid() || c.node(*st).src[0] != state ||
        c.node(*st).src[1] != c.node(*r3).out[0]) {
      return std::nullopt;
    }
    const ir::TensorDesc& tail = c.tensor(c.node(tv).out[0]);
    const ir::TensorDesc& input = c.tensor(k.out[0]);
    if (tail.shape().ne != ir::ShapeOf({taps, ch, 1}).ne || tail.view_offset != n * input.nb[0] ||
        tail.nb[0] != input.nb[0] || tail.nb[1] != input.nb[1])
      return std::nullopt;
    store = *st;
    if (!b.Add({tv, *c2, *r3, store})) {
      return std::nullopt;
    }
  }
  if (store.valid() && (!c.node(store).outputs().empty() && c.external(c.node(store).out[0]))) {
    return std::nullopt;
  }
  const auto si = Only(c, sc.out[0]);
  if (!si || !IsUnary(c, *si, ir::UnaryFn::kSilu) || !b.Add(*si)) {
    return std::nullopt;
  }
  const TensorId conv = c.node(*si).out[0];
  // q and k L2-normed, v; a verify's concatenation of the three.
  struct L2 {
    NodeId view, cont, norm, scale;
  };
  std::vector<L2> qk;
  NodeId v{};
  NodeId v2{};
  for (const NodeId r : c.readers(conv)) {
    if (!Is(c, r, Op::kView) || c.taken(r)) {
      return std::nullopt;
    }
    const auto ct = OnlyOf(c, c.node(r).out[0], Op::kCont);
    const auto nm = ct ? OnlyOf(c, c.node(*ct).out[0], Op::kRmsNorm) : std::nullopt;
    if (nm) {
      const auto s = Only(c, c.node(*nm).out[0]);
      if (!s || !Is(c, *s, Op::kScale)) {
        return std::nullopt;
      }
      qk.push_back({r, *ct, *nm, *s});
    } else if (ct) {
      v2 = r;                 // a verify's v rows
    } else if (!v.valid()) {  // (a lone row's heads: rank 2)
      v = r;
    } else {
      return std::nullopt;
    }
  }
  if (qk.size() != 2 || !v.valid()) {
    return std::nullopt;
  }
  std::ranges::sort(qk, {}, &L2::view);
  for (const L2& l : qk) {
    if (!b.Add({l.view, l.cont, l.norm, l.scale})) {
      return std::nullopt;
    }
  }
  if (!b.Add(v)) {
    return std::nullopt;
  }
  const ir::TensorDesc& qd = c.tensor(c.node(qk[0].view).out[0]);
  const std::uint64_t head = qd.ne(0);
  const std::uint64_t hk = qd.ne(1);
  const float eps = c.node(qk[0].norm).Params<ir::NormParams>().eps;
  const float scale = ScaleOf(c, qk[0].scale);
  if (head == 0 || hk == 0 || 2 * head * hk >= ch || (ch - 2 * head * hk) % head != 0) {
    return std::nullopt;
  }
  for (std::size_t i = 0; i < qk.size(); ++i) {
    const L2& l = qk[i];
    const ir::TensorDesc& td = c.tensor(c.node(l.view).out[0]);
    const auto sp = c.node(l.scale).Params<ir::ScaleParams>();
    if (td.shape().ne != ir::ShapeOf({head, hk, n, 1}).ne || td.view_offset != i * head * hk * 4 ||
        td.nb[0] != 4 || td.nb[1] != head * 4 || td.nb[2] != ch * 4 ||
        c.node(l.norm).Params<ir::NormParams>().eps != eps || sp.scale != scale || sp.bias != 0.0F)
      return std::nullopt;
  }
  const ir::TensorDesc& vd = c.tensor(c.node(v).out[0]);
  if (vd.shape().ne != ir::ShapeOf({head, (ch - 2 * head * hk) / head, n, 1}).ne ||
      vd.view_offset != 2 * head * hk * 4 || vd.nb[0] != 4 || vd.nb[1] != head * 4 ||
      vd.nb[2] != ch * 4)
    return std::nullopt;
  // A verify's save: q and k rows concatenated, then v's.
  NodeId saved{};
  std::vector<NodeId> verify;
  if (v2.valid()) {
    const auto cv = OnlyOf(c, c.node(v2).out[0], Op::kCont);
    std::vector<NodeId> rq;
    for (const L2& l : qk) {
      std::optional<NodeId> rs;
      for (const NodeId r : c.readers(c.node(l.scale).out[0])) {
        if (Is(c, r, Op::kReshape)) rs = r;
      }
      if (!rs) return std::nullopt;
      rq.push_back(*rs);
    }
    // q's and k's rows joined, then v's (the step's save: no reader).
    const auto cc1 = OnlyOf(c, c.node(rq[0]).out[0], Op::kConcat);
    const auto cc2 = cc1 ? OnlyOf(c, c.node(*cc1).out[0], Op::kConcat) : std::nullopt;
    if (!cv || !cc1 || !cc2 || Only(c, c.node(rq[1]).out[0]) != *cc1 ||
        Only(c, c.node(*cv).out[0]) != *cc2 || !c.readers(c.node(*cc2).out[0]).empty()) {
      return std::nullopt;
    }
    const ir::TensorDesc& vrows = c.tensor(c.node(v2).out[0]);
    const ir::Node& first_cat = c.node(*cc1);
    const ir::Node& last_cat = c.node(*cc2);
    if (c.tensor(c.node(rq[0]).out[0]).shape().ne != ir::ShapeOf({head * hk, n}).ne ||
        c.tensor(c.node(rq[1]).out[0]).shape().ne != ir::ShapeOf({head * hk, n}).ne ||
        first_cat.Params<ir::ConcatParams>().dim != 0 || first_cat.src[0] != c.node(rq[0]).out[0] ||
        first_cat.src[1] != c.node(rq[1]).out[0] || last_cat.Params<ir::ConcatParams>().dim != 0 ||
        last_cat.src[0] != first_cat.out[0] || last_cat.src[1] != c.node(*cv).out[0] ||
        vrows.shape().ne != ir::ShapeOf({ch - 2 * head * hk, n}).ne ||
        vrows.view_offset != 2 * head * hk * 4 || vrows.nb[0] != 4 || vrows.nb[1] != ch * 4 ||
        c.tensor(last_cat.out[0]).shape().ne != ir::ShapeOf({ch, n}).ne)
      return std::nullopt;
    const NodeId c1 = *cc1;
    saved = *cc2;
    verify = {rq[0], rq[1], c1, v2, *cv, saved};
    for (const NodeId x : verify) {
      if (!b.Add(x)) return std::nullopt;
    }
  }
  // The rows as the products made them, [C, n].
  if (!store.valid() && !v2.valid()) {
    return std::nullopt;  // neither a step nor a verify
  }
  // Nothing between the history's store and the convolution touches the
  // history (the store now follows the convolution).
  if (store.valid() && !Undisturbed(c, b, store, anchor, c.graph().root(state))) {
    return std::nullopt;
  }
  std::vector<TensorId> outs = {c.node(qk[0].scale).out[0], c.node(qk[1].scale).out[0],
                                c.node(v).out[0]};
  if (saved.valid()) outs.push_back(c.node(saved).out[0]);
  if (store.valid()) outs.push_back(c.node(store).out[0]);
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  const ir::GdnConvParams params{.eps = eps,
                                 .scale = scale,
                                 .qk_channels = static_cast<std::uint32_t>(2 * hk * head),
                                 .head = static_cast<std::uint32_t>(head)};
  const TensorId w = sc.src[1];
  const ir::TensorDesc rows_d = DenseDesc(rd.format.encoding, {ch, n});
  const ir::TensorDesc conv_d = DenseDesc(Encoding::kF32, {ch, n});
  {
    const std::array<const ir::TensorDesc*, 3> s = {&rows_d, &c.tensor(state), &c.tensor(w)};
    const std::array<const ir::TensorDesc*, 1> o = {&conv_d};
    if (!Has(c, ir::Fused(FusedKind::kGdnConv), s, o, ir::ToBytes(params))) {
      return std::nullopt;
    }
  }
  const ir::GdnHistoryParams hparams{.taps = static_cast<std::uint32_t>(taps)};
  const bool with_history = n < taps;
  if (store.valid()) {
    const ir::TensorDesc hist_d = DenseDesc(Encoding::kF32, {taps * ch, 1});
    const std::array<const ir::TensorDesc*, 2> s = {&rows_d,
                                                    with_history ? &c.tensor(state) : nullptr};
    const std::array<const ir::TensorDesc*, 1> o = {&hist_d};
    if (!Has(c, ir::Fused(FusedKind::kGdnHistory), s, o, ir::ToBytes(hparams))) {
      return std::nullopt;
    }
  }
  const std::array<NodeId, 3> q_k_v = {qk[0].scale, qk[1].scale, v};
  auto emit = [=](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    ContextOf(r, g.node(anchor));
    ir::Builder& b = r.builder();
    const TensorId rows = b.Reshape(r.Map(rows3), ir::ShapeOf({ch, n}));
    const TensorId out =
        b.Fused(FusedKind::kGdnConv, std::array{rows, r.Map(state), r.Map(w)}, params)[0];
    if (store.valid()) {
      const ir::Node& st = g.node(store);
      const TensorId hist =
          b.Fused(FusedKind::kGdnHistory,
                  std::array{rows, with_history ? r.Map(state) : TensorId{}}, hparams)[0];
      r.Set(st.out[0], b.CacheStore(r.Map(state), hist, r.Map(st.src[2])));
    }
    const std::uint64_t e = 4;
    for (std::size_t i = 0; i < 3; ++i) {
      const ir::TensorDesc& od = g.tensor(g.node(q_k_v[i]).out[0]);
      const std::uint64_t first = i == 0 ? 0 : (i == 1 ? hk * head : 2 * hk * head);
      r.Set(g.node(q_k_v[i]).out[0],
            b.View(out, od.shape(), {e, head * e, ch * e, ch * n * e}, first * e));
    }
    if (saved.valid()) {
      r.Set(g.node(saved).out[0], out);
    }
  };
  return b.Finish({{anchor, 0}}, emit);
}

// gdn_gates: a verify's gates (the decay's log and beta) as llmp.gdn.gates
// over the alpha and beta products, at beta's place.
std::optional<Match> GdnGatesMatch(const MatchContext& c, NodeId anchor) {
  if (c.shape() == nullptr || c.shape()->phase != Phase::kVerify) {
    return std::nullopt;
  }
  // alpha's chain: reshape, add dt_bias, softplus, times A, reshape.
  if (!Is(c, anchor, Op::kAdd) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& ad = c.node(anchor);
  const NodeId ra = Made(c, ad.src[0]);
  if (!Is(c, ra, Op::kReshape) || !Parameter(c, ad.src[1]) || Only(c, ad.src[0]) != anchor) {
    return std::nullopt;
  }
  const NodeId alpha = Made(c, c.node(ra).src[0]);
  if (!Is(c, alpha, Op::kMatMul) || Only(c, c.node(ra).src[0]) != ra) {
    return std::nullopt;
  }
  const auto sp = Only(c, ad.out[0]);
  if (!sp || !IsUnary(c, *sp, ir::UnaryFn::kSoftplus)) {
    return std::nullopt;
  }
  const auto ma = OnlyOf(c, c.node(*sp).out[0], Op::kMul);
  if (!ma || !Parameter(c, c.node(*ma).src[1])) {
    return std::nullopt;
  }
  const auto rg = OnlyOf(c, c.node(*ma).out[0], Op::kReshape);
  if (!rg) {
    return std::nullopt;
  }
  // beta's: the next sigmoid of a reshaped product of the same input.
  const TensorId x = c.node(alpha).src[1];
  NodeId sb{};
  NodeId rb{};
  NodeId beta{};
  for (const NodeId r : c.readers(x)) {
    if (r == alpha || !Is(c, r, Op::kMatMul)) {
      continue;
    }
    const auto rr = OnlyOf(c, c.node(r).out[0], Op::kReshape);
    const auto s = rr ? Only(c, c.node(*rr).out[0]) : std::nullopt;
    if (s && IsUnary(c, *s, ir::UnaryFn::kSigmoid) &&
        c.tensor(c.node(*rr).out[0]).ne(1) == c.tensor(c.node(*rg).out[0]).ne(1) && *s > *rg) {
      beta = r;
      rb = *rr;
      sb = *s;
      break;
    }
  }
  if (!sb.valid()) {
    return std::nullopt;
  }
  Building b(c);
  if (!b.Add({ra, anchor, *sp, *ma, *rg, rb, sb})) {
    return std::nullopt;
  }
  const TensorId gate = c.node(*rg).out[0];
  const TensorId bout = c.node(sb).out[0];
  for (const NodeId r : c.readers(gate)) {
    if (r < sb) return std::nullopt;
  }
  const std::array<TensorId, 2> outs = {gate, bout};
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  const TensorId dt = ad.src[1];
  const TensorId a = c.node(*ma).src[1];
  const std::array<TensorId, 4> s = {c.node(alpha).out[0], c.node(beta).out[0], dt, a};
  const std::array<const ir::TensorDesc*, 2> o = {&c.tensor(gate), &c.tensor(bout)};
  if (!Has(c, ir::Fused(FusedKind::kGdnGates), Descs(c, s), o, {})) {
    return std::nullopt;
  }
  const NodeId alpha_n = alpha;
  const NodeId beta_n = beta;
  const NodeId rg_n = *rg;
  auto emit = [=](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    ContextOf(r, g.node(sb));
    const std::array<TensorId, 4> src = {r.Map(g.node(alpha_n).out[0]),
                                         r.Map(g.node(beta_n).out[0]), r.Map(dt), r.Map(a)};
    const ir::Outputs outs = r.builder().Fused(FusedKind::kGdnGates, src);
    r.Set(g.node(rg_n).out[0], outs[0]);
    r.Set(g.node(sb).out[0], outs[1]);
  };
  return b.Finish({{sb, 0}}, emit);
}

// gdn_step: an owner's recurrence and its state's store (or a verify's
// recurrence alone) as llmp.gdn.step, the state written in place.
std::optional<Match> GdnStepMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kGatedDeltaNet) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& gd = c.node(anchor);
  const NodeId sr = Made(c, gd.src[5]);
  if (!Is(c, sr, Op::kReshape) || Only(c, gd.src[5]) != anchor) {
    return std::nullopt;
  }
  const TensorId state = c.node(sr).src[0];
  if (c.producer(state).key != Prim(Op::kState)) {
    return std::nullopt;
  }
  Building b(c);
  if (!b.Add({sr, anchor})) {
    return std::nullopt;
  }
  NodeId store{};
  TensorId idx{};
  if (const auto rs = OnlyOf(c, gd.out[1], Op::kReshape)) {
    const auto st = OnlyOf(c, c.node(*rs).out[0], Op::kCacheStore);
    if (!st || c.node(*st).src[0] != state || !b.Add({*rs, *st})) {
      return std::nullopt;
    }
    store = *st;
    idx = c.node(*st).src[2];
  } else if (!c.readers(gd.out[1]).empty() || c.external(gd.out[1])) {
    return std::nullopt;
  }
  const bool write = store.valid();
  const NodeId place = write ? store : anchor;
  for (const NodeId r : c.readers(gd.out[0])) {
    if (r < place) return std::nullopt;
  }
  if (write && !Undisturbed(c, b, anchor, store, c.graph().root(state))) {
    return std::nullopt;
  }
  std::vector<TensorId> outs = {gd.out[0]};
  if (write) outs.push_back(c.node(store).out[0]);
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  const ir::GdnStepParams params{.write = write};
  std::vector<TensorId> src = {gd.src[0], gd.src[1], gd.src[2], gd.src[3], gd.src[4], state};
  if (write) src.push_back(idx);
  std::vector<const ir::TensorDesc*> o = {&c.tensor(gd.out[0])};
  if (write) o.push_back(&c.tensor(state));
  if (!Has(c, ir::Fused(FusedKind::kGdnStep), Descs(c, src), o, ir::ToBytes(params))) {
    return std::nullopt;
  }
  auto emit = [=](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    ContextOf(r, g.node(anchor));
    std::vector<TensorId> s;
    for (const TensorId t : src) s.push_back(r.Map(t));
    const ir::Outputs out = r.builder().Fused(FusedKind::kGdnStep, s, params);
    r.Set(g.node(anchor).out[0], out[0]);
    if (write) r.Set(g.node(store).out[0], out[1]);
  };
  return b.Finish({{place, 0}}, emit);
}

// gdn_norm_gate: the gated output norm (each head's RMS norm times the
// weight and sigmoid(z)) as llmp.gdn.norm_gate.
std::optional<Match> GdnNormGateMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kRmsNorm) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& nr = c.node(anchor);
  const auto nm = OnlyOf(c, nr.out[0], Op::kMul);
  if (!nm || c.node(*nm).src[0] != nr.out[0] || !Parameter(c, c.node(*nm).src[1])) {
    return std::nullopt;
  }
  const auto gm = OnlyOf(c, c.node(*nm).out[0], Op::kMul);
  if (!gm || c.node(*gm).src[0] != c.node(*nm).out[0]) {
    return std::nullopt;
  }
  const NodeId sz = Made(c, c.node(*gm).src[1]);
  if (!IsUnary(c, sz, ir::UnaryFn::kSigmoid) || Only(c, c.node(*gm).src[1]) != *gm || c.taken(sz)) {
    return std::nullopt;
  }
  const TensorId z = c.node(sz).src[0];
  if (!(c.tensor(z).shape().ne == c.tensor(nr.src[0]).shape().ne)) {
    return std::nullopt;
  }
  Building b(c);
  if (!b.Add({anchor, *nm, sz, *gm})) {
    return std::nullopt;
  }
  const std::array<TensorId, 1> outs = {c.node(*gm).out[0]};
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  const ir::NormParams params{.eps = nr.Params<ir::NormParams>().eps};
  const std::array<TensorId, 3> s = {nr.src[0], c.node(*nm).src[1], z};
  const std::array<const ir::TensorDesc*, 1> o = {&c.tensor(outs[0])};
  if (!Has(c, ir::Fused(FusedKind::kNormGate), Descs(c, s), o, ir::ToBytes(params))) {
    return std::nullopt;
  }
  const NodeId gmn = *gm;
  auto emit = [=](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    ContextOf(r, g.node(gmn));
    const std::array<TensorId, 3> src = {r.Map(s[0]), r.Map(s[1]), r.Map(s[2])};
    r.Set(g.node(gmn).out[0], r.builder().Fused(FusedKind::kNormGate, src, params)[0]);
  };
  return b.Finish({{gmn, 0}}, emit);
}

// --------------------------------------------------------------------- QSA

// qsa_prep: a head's norm, weight and rotation (llmp.qsa.prep), reading a
// strided view of its product's rows where the components copy them packed
// first (each head's query before its gate; the indexer's query heads
// before its key).
std::optional<Match> QsaPrepMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kRope) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& ro = c.node(anchor);
  const NodeId mm = Made(c, ro.src[0]);
  if (!Is(c, mm, Op::kMul) || Only(c, ro.src[0]) != anchor || !Parameter(c, c.node(mm).src[1])) {
    return std::nullopt;
  }
  const NodeId nn = Made(c, c.node(mm).src[0]);
  if (!Is(c, nn, Op::kRmsNorm) || Only(c, c.node(mm).src[0]) != mm) {
    return std::nullopt;
  }
  Building b(c);
  if (!b.Add({nn, mm, anchor})) {
    return std::nullopt;
  }
  TensorId x = c.node(nn).src[0];
  const NodeId cont = Made(c, x);
  if (Is(c, cont, Op::kCont) && Only(c, x) == nn &&
      Is(c, Made(c, c.node(cont).src[0]), Op::kView) && !c.taken(cont)) {
    const TensorId view = c.node(cont).src[0];
    if (c.tensor(view).shape().ne == c.tensor(x).shape().ne && b.Add(cont)) {
      x = view;
    }
  }
  const std::array<TensorId, 1> outs = {ro.out[0]};
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  const ir::NormRopeParams params{.eps = c.node(nn).Params<ir::NormParams>().eps,
                                  .rope = ro.Params<ir::RopeParams>()};
  const std::array<TensorId, 4> s = {x, c.node(mm).src[1], ro.src[1], ro.source(2)};
  const std::array<const ir::TensorDesc*, 1> o = {&c.tensor(ro.out[0])};
  if (!Has(c, ir::Fused(FusedKind::kRmsNormMulRope), Descs(c, s), o, ir::ToBytes(params))) {
    return std::nullopt;
  }
  auto emit = [=](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    ContextOf(r, g.node(anchor));
    std::vector<TensorId> src = {r.Map(s[0]), r.Map(s[1]), r.Map(s[2])};
    if (s[3].valid()) src.push_back(r.Map(s[3]));
    r.Set(g.node(anchor).out[0], r.builder().Fused(FusedKind::kRmsNormMulRope, src, params)[0]);
  };
  return b.Finish({{anchor, 0}}, emit);
}

namespace {

// The step's positions in four sections (QSA's rotations'), found by name.
TensorId Positions4(const MatchContext& c) {
  const ir::Graph& g = c.graph();
  for (const TensorId t : g.inputs()) {
    if (g.name(t) == "positions.4") {
      return t;
    }
  }
  return {};
}

// Segment s's view of the positions (the first of its rows' in section 0):
// the whole input for a step of one segment.
TensorId SegmentPositions(RewriteContext& r, const StepShape* shape, TensorId positions,
                          std::uint8_t segment) {
  if (shape == nullptr || shape->segments.size() <= 1 || segment == ir::kJoinedSegment) {
    return r.Map(positions);
  }
  std::uint64_t first = 0;
  for (std::size_t i = 0; i < segment && i < shape->segments.size(); ++i) {
    first += shape->segments[i].rows;
  }
  const std::uint64_t rows = shape->segments.at(segment).rows;
  return r.builder().View(r.Map(positions), ir::ShapeOf({rows}), {4, 4 * rows, 4 * rows, 4 * rows},
                          first * 4);
}

bool PrimitiveDefinition(const MatchContext& c, Building& match, const ir::Graph& definition,
                         TensorId expected, TensorId actual, std::span<const TensorId> sources,
                         const ir::Node& anchor, std::vector<TensorId>& mapped);

}  // namespace

// ------------------------------------------------ DeepSeek local fast forms

// The explicit rounded recipe only: generic IndexScore/TopK is a different
// numerical/visibility contract. Closed() authenticates all eliminated
// observers and the original read/write ordering, including aliases/stores.
std::optional<Match> Dsv4LidMaskMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kSelectedMask) || c.taken(anchor)) return std::nullopt;
  const auto& mask = c.node(anchor);
  const auto top = Made(c, mask.src[1]);
  if (!Is(c, top, Op::kVisibleTopK)) return std::nullopt;
  const auto& select = c.node(top);
  const auto score = Made(c, select.src[0]);
  if (!Is(c, score, Op::kRoundedIndexScore)) return std::nullopt;
  const auto& scored = c.node(score);
  const auto rounded = scored.Params<ir::RoundedIndexScoreParams>();
  const auto selected = mask.Params<ir::SelectedMaskParams>();
  if (rounded.query != ir::QueryRounding::kF16 || rounded.reserved[0] || rounded.reserved[1] ||
      rounded.reserved[2] || scored.src[3] != select.src[1] ||
      c.tensor(scored.src[1]).ne(2) != selected.cells || mask.component != scored.component ||
      select.component != scored.component || mask.batch != scored.batch ||
      select.batch != scored.batch)
    return std::nullopt;
  const ir::LidMaskParams params{.top = select.Params<ir::TopKParams>().k,
                                 .raw_cells = selected.raw_cells,
                                 .cells = selected.cells};
  const std::array sources{scored.src[0], scored.src[1], scored.src[2], scored.src[3], mask.src[0]};
  const std::array outputs{mask.out[0]};
  // These are existing boundary tensors, so preserve their authenticated
  // graph roots when querying the native recipe. Descriptor-only queries
  // deliberately have unknown alignment and cannot select this kernel.
  ir::Node candidate = mask;
  candidate.key = ir::Fused(FusedKind::kDsv4LidMask);
  candidate.SetParams(params);
  candidate.src = {};
  std::ranges::copy(sources, candidate.src.begin());
  candidate.nsrc = static_cast<std::uint8_t>(sources.size());
  Building b(c);
  if (!b.Add({score, top, anchor}) || !b.Closed(outputs) ||
      c.view().Candidates(c.graph(), candidate).empty())
    return std::nullopt;
  return b.Finish(
      {{anchor, 0}},
      [=](RewriteContext& r, int) {
        ContextOf(r, r.original().node(anchor));
        std::array<TensorId, 5> mapped{};
        for (std::size_t i = 0; i < sources.size(); ++i) mapped[i] = r.Map(sources[i]);
        r.Set(mask.out[0], r.builder().Fused(FusedKind::kDsv4LidMask, mapped, params)[0]);
      },
      std::vector<TensorId>(sources.begin(), sources.end()));
}

// Keep the entire score/TopK/mask path as operands. Only the exact K/V
// concatenation and its two permutations disappear, at Attention's place.
std::optional<Match> StateBankAttentionMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kAttention) || c.taken(anchor) || !c.state_spans()) return std::nullopt;
  const auto& attn = c.node(anchor);
  if (attn.batch != ir::Batch::kPerOwner || c.tensor(attn.out[0]).format.encoding != Encoding::kF32)
    return std::nullopt;
  const auto pk = Made(c, attn.src[1]);
  const auto pv = Made(c, attn.src[2]);
  if (!Is(c, pk, Op::kPermute) || !Is(c, pv, Op::kPermute) ||
      c.node(pk).Params<ir::PermuteParams>().axes != std::array<std::uint8_t, 4>{0, 2, 1, 3} ||
      c.node(pv).Params<ir::PermuteParams>().axes != std::array<std::uint8_t, 4>{0, 2, 1, 3} ||
      c.node(pk).src[0] != c.node(pv).src[0])
    return std::nullopt;
  const auto concat = Made(c, c.node(pk).src[0]);
  if (!Is(c, concat, Op::kConcat) || c.node(concat).Params<ir::ConcatParams>().dim != 2)
    return std::nullopt;
  const auto& join = c.node(concat);
  const auto slices = StateBankSlices(c.graph(), join.src[0], join.src[1]);
  if (!slices || c.tensor(attn.out[0]).segment != (*slices)[0].ref.segment) return std::nullopt;
  // A query sliced from joined products retains its root's joined segment
  // (the IR's alias identity). Preserve that exact operand and its strides;
  // only the same owner's K/V copy is removed.
  const auto query_owner = c.tensor(attn.src[0]).segment;
  if (query_owner != ir::kJoinedSegment && query_owner != (*slices)[0].ref.segment)
    return std::nullopt;
  const auto fact = c.state_spans()(*slices);
  const auto bytes =
      (c.tensor(join.src[0]).ne(2) + c.tensor(join.src[1]).ne(2)) * c.tensor(join.src[0]).nb[2];
  if (!fact || fact->bank.encoding != Encoding::kF16 ||
      fact->bank.cell_bytes != c.tensor(join.src[0]).nb[2] || fact->bank.range.bytes != bytes ||
      fact->bank.range.offset > fact->region_bytes ||
      bytes > fact->region_bytes - fact->bank.range.offset)
    return std::nullopt;
  Building b(c);
  if (!b.Add({concat, pk, pv, anchor}) || !b.Closed(std::array{attn.out[0]})) return std::nullopt;
  const std::array sources = {attn.src[0], join.src[0], join.src[1], attn.source(3),
                              attn.source(4)};
  const std::array outputs = {&c.tensor(attn.out[0])};
  if (!Has(c, ir::Fused(FusedKind::kStateBankAttention), Descs(c, sources), outputs, attn.params,
           true))
    return std::nullopt;
  const auto params = attn.Params<ir::AttnParams>();
  return b.Finish(
      {{anchor, 0}},
      [=](RewriteContext& r, int) {
        ContextOf(r, r.original().node(anchor));
        std::array<TensorId, 5> mapped{};
        for (std::size_t i = 0; i < sources.size(); ++i) mapped[i] = r.Map(sources[i]);
        r.Set(attn.out[0], r.builder().Fused(FusedKind::kStateBankAttention, mapped, params)[0]);
      },
      std::vector<TensorId>(sources.begin(), sources.end()));
}

std::optional<Match> Dsv4QHeadMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kRope) || c.taken(anchor)) return std::nullopt;
  const ir::Node& ro = c.node(anchor);
  const NodeId norm = Made(c, ro.src[0]);
  if (!Is(c, norm, Op::kRmsNorm) || Only(c, ro.src[0]) != anchor || ro.source(2).valid())
    return std::nullopt;
  const ir::Node& nn = c.node(norm);
  const ir::TensorDesc& q = c.tensor(nn.src[0]);
  const ir::NormRopeParams p{.eps = nn.Params<ir::NormParams>().eps,
                             .rope = ro.Params<ir::RopeParams>()};
  // This opt-in selects the existing packed normal-RoPE tail kernel. Other
  // rotary forms retain the primitive nodes, even with a permissive registry.
  if (q.format.encoding != Encoding::kF32 ||
      c.tensor(nn.out[0]).format.encoding != Encoding::kF32 ||
      c.tensor(ro.out[0]).format.encoding != Encoding::kF32 || !ir::IsContiguous(q) ||
      q.ne(0) == 0 || q.ne(3) != 1 || p.rope.mode != ir::RopeMode::kNormal || p.rope.dims == 0 ||
      p.rope.dims % 2 != 0 || p.rope.dims > q.ne(0) || p.rope.offset != q.ne(0) - p.rope.dims ||
      p.rope.inverse || p.rope.table || nn.component != ro.component || nn.batch != ro.batch ||
      c.tensor(nn.out[0]).segment != c.tensor(ro.out[0]).segment)
    return std::nullopt;
  Building b(c);
  const std::array<TensorId, 2> sources{nn.src[0], ro.src[1]};
  const std::array<TensorId, 1> outputs{ro.out[0]};
  const std::array<const ir::TensorDesc*, 1> descs{&c.tensor(ro.out[0])};
  if (!b.Add({norm, anchor}) || !b.Closed(outputs) ||
      !Has(c, ir::Fused(FusedKind::kDsv4QHead), Descs(c, sources), descs, ir::ToBytes(p)))
    return std::nullopt;
  const auto emit = [=](RewriteContext& r, int) {
    ContextOf(r, r.original().node(anchor));
    const std::array<TensorId, 2> mapped{r.Map(sources[0]), r.Map(sources[1])};
    r.Set(outputs[0], r.builder().Fused(FusedKind::kDsv4QHead, mapped, p)[0]);
  };
  return b.Finish({{anchor, 0}}, emit, std::vector<TensorId>(sources.begin(), sources.end()));
}

std::optional<Match> Dsv4OutAMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kCont) || c.taken(anchor)) return std::nullopt;
  const ir::Node& cont = c.node(anchor);
  const NodeId perm2 = Made(c, cont.src[0]);
  if (!Is(c, perm2, Op::kPermute) || Only(c, cont.src[0]) != anchor) return std::nullopt;
  const auto p2 = c.node(perm2).Params<ir::PermuteParams>();
  if (p2.axes != std::array<std::uint8_t, 4>{0, 2, 1, 3}) return std::nullopt;

  const NodeId matmul = Made(c, c.node(perm2).src[0]);
  if (!Is(c, matmul, Op::kMatMul) || Only(c, c.node(perm2).src[0]) != perm2) return std::nullopt;
  const ir::Node& mm = c.node(matmul);

  const NodeId perm1 = Made(c, mm.src[1]);
  if (!Is(c, perm1, Op::kPermute) || Only(c, mm.src[1]) != matmul) return std::nullopt;
  const auto p1 = c.node(perm1).Params<ir::PermuteParams>();
  if (p1.axes != std::array<std::uint8_t, 4>{0, 2, 1, 3}) return std::nullopt;

  const NodeId reshape1 = Made(c, c.node(perm1).src[0]);
  if (!Is(c, reshape1, Op::kReshape) || Only(c, c.node(perm1).src[0]) != perm1) return std::nullopt;

  const NodeId rope = Made(c, c.node(reshape1).src[0]);
  if (!Is(c, rope, Op::kRope) || Only(c, c.node(reshape1).src[0]) != reshape1) return std::nullopt;
  const ir::Node& ro = c.node(rope);
  const auto rp = ro.Params<ir::RopeParams>();
  // Match the operation, leaving specialized dimensions and row bounds to
  // the candidate kernel capability.
  if (!rp.inverse || rp.mode != ir::RopeMode::kNormal || rp.dims == 0 || rp.dims % 2 != 0 ||
      rp.table || ro.src[2].valid())
    return std::nullopt;

  const ir::TensorDesc& heads = c.tensor(ro.src[0]);
  if (rp.dims > heads.ne(0) || rp.offset != heads.ne(0) - rp.dims) return std::nullopt;

  NodeId groups_node = Made(c, mm.src[0]);
  TensorId weight_out_a = mm.src[0];
  if (Is(c, groups_node, Op::kReshape) && Only(c, mm.src[0]) == matmul) {
    weight_out_a = c.node(groups_node).src[0];
  }
  const ir::TensorDesc& w = c.tensor(weight_out_a);
  if (w.format.encoding != Encoding::kQ8_0 && w.format.encoding != Encoding::kF32 &&
      w.format.encoding != Encoding::kF16) {
    return std::nullopt;
  }

  const std::array<TensorId, 3> sources = {weight_out_a, ro.src[0], ro.src[1]};
  const std::array<TensorId, 1> outputs = {cont.out[0]};
  const std::array<const ir::TensorDesc*, 1> descs = {&c.tensor(cont.out[0])};

  if (!Has(c, ir::Fused(FusedKind::kDsv4OutA), Descs(c, sources), descs, ir::ToBytes(rp))) {
    return std::nullopt;
  }
  // Authenticate the complete assembly: valid overall input/output sizes
  // alone do not establish which rows and groups the product multiplies.
  ir::Builder definition_builder;
  std::array<TensorId, 3> inputs{};
  for (std::size_t i = 0; i < sources.size(); ++i)
    inputs[i] = definition_builder.Input(c.tensor(sources[i]).format);
  const auto expanded = ir::Decompose(definition_builder, ir::Fused(FusedKind::kDsv4OutA),
                                      ir::ToBytes(rp), inputs, {});
  if (!definition_builder.ok() || expanded.n != 1) return std::nullopt;
  definition_builder.Output(expanded[0]);
  auto definition = std::move(definition_builder).Finish();
  if (!definition) return std::nullopt;
  Building b(c);
  std::vector<TensorId> mapped(definition->tensor_count());
  if (!PrimitiveDefinition(c, b, *definition, definition->outputs()[0], outputs[0], sources, cont,
                           mapped) ||
      !b.Closed(outputs))
    return std::nullopt;

  const auto emit = [=](RewriteContext& r, int) {
    ContextOf(r, r.original().node(anchor));
    const std::array<TensorId, 3> mapped = {r.Map(sources[0]), r.Map(sources[1]),
                                            r.Map(sources[2])};
    r.Set(outputs[0], r.builder().Fused(FusedKind::kDsv4OutA, mapped, rp)[0]);
  };
  return b.Finish({{anchor, 0}}, emit, std::vector<TensorId>(sources.begin(), sources.end()));
}

namespace {

// The candidate's boundaries; the full assembly is authenticated below
// against its primitive definition, not only these producer op names.
struct CompressorSide {
  TensorId state, current, indices;
};

std::optional<CompressorSide> CompressorSideOf(const MatchContext& c, TensorId t, bool overlap) {
  NodeId n = Made(c, t);
  if (overlap) {
    if (!Is(c, n, Op::kConcat)) return std::nullopt;
    n = Made(c, c.node(n).src[0]);
  }
  if (!Is(c, n, Op::kReshape)) return std::nullopt;
  n = Made(c, c.node(n).src[0]);
  if (overlap) {
    if (!Is(c, n, Op::kCont)) return std::nullopt;
    n = Made(c, c.node(n).src[0]);
    if (!Is(c, n, Op::kView)) return std::nullopt;
    n = Made(c, c.node(n).src[0]);
  }
  if (!Is(c, n, Op::kGetRows)) return std::nullopt;
  const ir::Node& get = c.node(n);
  n = Made(c, get.src[0]);
  if (overlap) {
    if (!Is(c, n, Op::kConcat)) return std::nullopt;
    n = Made(c, c.node(n).src[0]);
  }
  if (!Is(c, n, Op::kConcat)) return std::nullopt;
  return CompressorSide{c.node(n).src[0], c.node(n).src[1], get.src[1]};
}

// Matches the definition's computed DAG against this graph, stopping at
// its explicit boundary inputs. Parameter bytes and every meaningful view
// stride/offset must agree. Boundary leaves retain their actual strides. Callers validate any views
// of those leaves; compressor views follow fresh packed concatenations.
bool PrimitiveDefinition(const MatchContext& c, Building& match, const ir::Graph& definition,
                         TensorId expected, TensorId actual, std::span<const TensorId> sources,
                         const ir::Node& anchor, std::vector<TensorId>& mapped) {
  if (!expected.valid() || !actual.valid()) return expected == actual;
  if (mapped.at(expected.v).valid()) return mapped.at(expected.v) == actual;
  const ir::Node& want = definition.node(definition.producer(expected));
  if (want.key == Prim(Op::kInput)) {
    const auto inputs = definition.inputs();
    const auto at = std::ranges::find(inputs, expected);
    if (at == inputs.end()) return false;
    return sources[static_cast<std::size_t>(at - inputs.begin())] == actual;
  }
  const NodeId id = Made(c, actual);
  const ir::Node& got = c.node(id);
  const ir::TensorDesc& wd = definition.tensor(expected);
  const ir::TensorDesc& gd = c.tensor(actual);
  // A shared weight's direct view stays joined even inside an owner's
  // fragment. It contains no owner data; all computed values still have
  // to belong to the anchor's owner.
  const bool weight_view = got.key == Prim(Op::kView) && got.nsrc == 1 && Weight(c, got.src[0]) &&
                           std::ranges::contains(sources, got.src[0]);
  if (want.key != got.key || want.params != got.params || want.nsrc != got.nsrc || want.nout != 1 ||
      got.nout != 1 || wd.shape().ne != gd.shape().ne || wd.format.encoding != gd.format.encoding ||
      wd.format.layout != gd.format.layout || wd.is_view() != gd.is_view() ||
      got.component != anchor.component ||
      (!weight_view &&
       (got.batch != anchor.batch || gd.segment != c.tensor(anchor.out[0]).segment)) ||
      c.taken(id))
    return false;
  for (std::size_t axis = 0; axis < format::kMaxDims; ++axis) {
    if (wd.ne(axis) > 1 && wd.nb[axis] != gd.nb[axis]) return false;
  }
  if (wd.is_view()) {
    // Offsets are part of the formula, including the compressor's current
    // channel half and the distinct scale/base runs of an HC mix.
    if (wd.view_offset != gd.view_offset) return false;
  }
  for (std::size_t i = 0; i < want.nsrc; ++i) {
    if (!PrimitiveDefinition(c, match, definition, want.src[i], got.src[i], sources, anchor,
                             mapped))
      return false;
  }
  mapped.at(expected.v) = actual;
  return match.Add(id);
}

}  // namespace

std::optional<Match> Dsv4CompressMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kPool) || c.taken(anchor)) return std::nullopt;
  const ir::Node& pool = c.node(anchor);
  const auto pp = pool.Params<ir::PoolParams>();
  if (pp.kind != ir::PoolKind::kSoftmax || pool.nsrc != 2) return std::nullopt;
  const bool overlap = Is(c, Made(c, pool.src[0]), Op::kConcat);
  if (overlap && pp.ratio % 2 != 0) return std::nullopt;
  const ir::CompressorParams params{.ratio = pp.ratio / (overlap ? 2U : 1U), .overlap = overlap};
  const auto values = CompressorSideOf(c, pool.src[0], overlap);
  const auto scores = CompressorSideOf(c, pool.src[1], overlap);
  if (!values || !scores || values->indices != scores->indices) return std::nullopt;
  const std::array<TensorId, 5> sources{values->state, scores->state, values->current,
                                        scores->current, values->indices};
  const std::array<TensorId, 1> outputs{pool.out[0]};
  const std::array<const ir::TensorDesc*, 1> descs{&c.tensor(outputs[0])};
  if (!Has(c, ir::Fused(FusedKind::kDsv4Compress), Descs(c, sources), descs, ir::ToBytes(params)))
    return std::nullopt;
  ir::Builder b;
  std::array<TensorId, 5> inputs{};
  for (std::size_t i = 0; i < sources.size(); ++i) inputs[i] = b.Input(c.tensor(sources[i]).format);
  const auto expanded =
      ir::Decompose(b, ir::Fused(FusedKind::kDsv4Compress), ir::ToBytes(params), inputs, {});
  if (!b.ok() || expanded.n != 1) return std::nullopt;
  b.Output(expanded[0]);
  auto definition = std::move(b).Finish();
  if (!definition) return std::nullopt;
  Building match(c);
  std::vector<TensorId> mapped(definition->tensor_count());
  if (!PrimitiveDefinition(c, match, *definition, definition->outputs()[0], outputs[0], sources,
                           pool, mapped) ||
      !match.Closed(outputs))
    return std::nullopt;
  const auto emit = [=](RewriteContext& r, int) {
    ContextOf(r, r.original().node(anchor));
    std::array<TensorId, 5> mapped_sources{};
    for (std::size_t i = 0; i < sources.size(); ++i) mapped_sources[i] = r.Map(sources[i]);
    r.Set(outputs[0], r.builder().Fused(FusedKind::kDsv4Compress, mapped_sources, params)[0]);
  };
  return match.Finish({{anchor, 0}}, emit, std::vector<TensorId>(sources.begin(), sources.end()));
}

std::optional<Match> Dsv4HcPreMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kMul) || c.taken(anchor)) return std::nullopt;
  const ir::Node& last = c.node(anchor);
  const NodeId norm = Made(c, last.src[0]);
  const NodeId fold = Is(c, norm, Op::kRmsNorm) ? Made(c, c.node(norm).src[0]) : NodeId{};
  if (!Is(c, fold, Op::kHcPre)) return std::nullopt;
  const TensorId x = c.node(fold).src[0];
  const auto& xd = c.tensor(x);
  if (c.max_decode_rows() > 1 && xd.ne(2) > c.max_decode_rows()) return std::nullopt;
  const NodeId pre = Made(c, c.node(fold).src[1]);
  const NodeId sigmoid = Is(c, pre, Op::kScale) ? Made(c, c.node(pre).src[0]) : NodeId{};
  const NodeId add =
      IsUnary(c, sigmoid, ir::UnaryFn::kSigmoid) ? Made(c, c.node(sigmoid).src[0]) : NodeId{};
  const NodeId mul = Is(c, add, Op::kAdd) ? Made(c, c.node(add).src[0]) : NodeId{};
  const NodeId run = Is(c, mul, Op::kMul) ? Made(c, c.node(mul).src[0]) : NodeId{};
  const NodeId product = Is(c, run, Op::kView) ? Made(c, c.node(run).src[0]) : NodeId{};
  if (!Is(c, product, Op::kMatMul)) return std::nullopt;
  const NodeId scale_view = Made(c, c.node(mul).src[1]);
  const NodeId base_view = Made(c, c.node(add).src[1]);
  if (!Is(c, scale_view, Op::kView) || !Is(c, base_view, Op::kView)) return std::nullopt;
  const std::array<TensorId, 5> sources{x, c.node(product).src[0], c.node(scale_view).src[0],
                                        c.node(base_view).src[0], last.src[1]};
  // The definition's input stand-ins are packed. Its views of scale/base
  // address these whole weights, never an offset within a containing root.
  if (!ir::IsContiguous(xd)) return std::nullopt;
  for (std::size_t i = 1; i < sources.size(); ++i)
    if (!Weight(c, sources[i]) || !ir::IsContiguous(c.tensor(sources[i]))) return std::nullopt;
  const TensorId mixes = c.node(product).out[0];
  for (const NodeId comb_id : c.readers(mixes)) {
    if (!Is(c, comb_id, Op::kHcMix)) continue;
    const ir::Node& comb = c.node(comb_id);
    const auto hp = comb.Params<ir::HcParams>();
    if (hp.mix != ir::HcMixKind::kSinkhorn || comb.src[1] != sources[2] ||
        comb.src[2] != sources[3])
      continue;
    const ir::SinkhornPreParams params{.streams = hp.streams,
                                       .eps = c.node(norm).Params<ir::NormParams>().eps,
                                       .hc_eps = hp.eps,
                                       .iterations = hp.iterations};
    ir::Builder b;
    std::array<TensorId, 5> inputs{};
    for (std::size_t i = 0; i < sources.size(); ++i)
      inputs[i] = b.Input(c.tensor(sources[i]).format);
    const auto expanded =
        ir::Decompose(b, ir::Fused(FusedKind::kDsv4HcPre), ir::ToBytes(params), inputs, {});
    if (!b.ok() || expanded.n != 3) continue;
    for (const TensorId out : expanded.span()) b.Output(out);
    auto definition = std::move(b).Finish();
    if (!definition) continue;
    for (std::uint32_t j = product.v + 1; j < anchor.v; ++j) {
      const NodeId post_id{j};
      if (!Is(c, post_id, Op::kScale)) continue;
      const std::array<TensorId, 3> outputs{last.out[0], c.node(post_id).out[0], comb.out[0]};
      Building match(c);
      std::vector<TensorId> mapped(definition->tensor_count());
      bool matches = true;
      for (std::size_t i = 0; i < outputs.size() && matches; ++i)
        matches = PrimitiveDefinition(c, match, *definition, definition->outputs()[i], outputs[i],
                                      sources, last, mapped);
      if (!matches || !match.Closed(outputs)) continue;
      // Both carries were computed earlier in the primitive graph. They
      // must not escape to a reader before their replacement is emitted.
      for (const TensorId out : outputs)
        for (const NodeId reader : c.readers(out))
          if (reader < anchor && !match.Has(reader)) matches = false;
      if (!matches || !Has(c, ir::Fused(FusedKind::kDsv4HcPre), Descs(c, sources),
                           Descs(c, outputs), ir::ToBytes(params)))
        continue;
      const auto emit = [=](RewriteContext& r, int) {
        ContextOf(r, r.original().node(anchor));
        std::array<TensorId, 5> mapped_sources{};
        for (std::size_t i = 0; i < sources.size(); ++i) mapped_sources[i] = r.Map(sources[i]);
        const auto out = r.builder().Fused(FusedKind::kDsv4HcPre, mapped_sources, params);
        for (std::size_t i = 0; i < outputs.size(); ++i) r.Set(outputs[i], out[i]);
      };
      return match.Finish({{anchor, 0}}, emit,
                          std::vector<TensorId>(sources.begin(), sources.end()));
    }
  }
  return std::nullopt;
}

std::optional<Match> Dsv4HcPostMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kRmsNorm) || c.taken(anchor)) return std::nullopt;
  const ir::Node& norm = c.node(anchor);
  const NodeId flat_id = Made(c, norm.src[0]);
  const NodeId post_id = Is(c, flat_id, Op::kReshape) ? Made(c, c.node(flat_id).src[0]) : NodeId{};
  if (!Is(c, post_id, Op::kHcPost) || Only(c, norm.src[0]) != anchor) return std::nullopt;
  const ir::Node& post = c.node(post_id);
  if (post.nsrc != 4 || !post.src[3].valid()) return std::nullopt;
  const auto hp = post.Params<ir::HcParams>();
  if (post.params != ir::ToBytes(ir::HcParams{.streams = hp.streams})) return std::nullopt;
  const auto& pd = c.tensor(post.out[0]);
  const auto& fd = c.tensor(norm.src[0]);
  const auto& nd = c.tensor(norm.out[0]);
  // A post belongs to the preceding combine, while the flat norm belongs
  // to the next mix. The direct data edge crosses components, but never
  // owners; the flat reshape and norm must share their component.
  if (c.node(flat_id).component != norm.component || post.batch != norm.batch ||
      c.node(flat_id).batch != norm.batch || pd.segment != nd.segment || fd.segment != nd.segment ||
      pd.format.encoding != Encoding::kF32 || nd.format.encoding != Encoding::kF32 ||
      fd.shape().ne != ir::ShapeOf({pd.ne(0) * pd.ne(1), pd.ne(2)}).ne || fd.view_offset != 0 ||
      !ir::IsContiguous(pd) || !ir::IsContiguous(fd))
    return std::nullopt;
  // The paired helper reads packed boundaries; strided consumers retain
  // the primitive sequence instead of selecting a pair that cannot launch.
  for (const TensorId source : post.sources())
    if (!ir::IsContiguous(c.tensor(source))) return std::nullopt;
  const ir::HcNormPostParams params{.eps = norm.Params<ir::NormParams>().eps,
                                    .streams = hp.streams};
  const std::array<TensorId, 2> outputs{post.out[0], norm.out[0]};
  if (!Has(c, ir::Fused(FusedKind::kHcNormPost), Descs(c, post.sources()), Descs(c, outputs),
           ir::ToBytes(params)))
    return std::nullopt;
  Building match(c);
  if (!match.Add({post_id, flat_id, anchor}) || !match.Closed(outputs)) return std::nullopt;
  const std::vector<TensorId> sources(post.sources().begin(), post.sources().end());
  const auto emit = [=](RewriteContext& r, int) {
    ContextOf(r, r.original().node(post_id));
    std::vector<TensorId> mapped;
    for (const TensorId input : sources) mapped.push_back(r.Map(input));
    const auto out = r.builder().Fused(FusedKind::kHcNormPost, mapped, params);
    for (std::size_t i = 0; i < outputs.size(); ++i) r.Set(outputs[i], out[i]);
  };
  return match.Finish({{post_id, 0}}, emit, sources);
}

// qsa_pool: an owner's block keys (their raw rows pooled, normed, rotated
// and rounded, stored at their cells) as llmp.qsa.pool, which writes only
// the blocks the step completes.
std::optional<Match> QsaPoolMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kCacheStore) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& cs = c.node(anchor);
  const TensorId blocks = cs.src[0];
  if (c.producer(blocks).key != Prim(Op::kState) ||
      c.tensor(blocks).format.encoding != Encoding::kBF16) {
    return std::nullopt;
  }
  const NodeId r2 = Made(c, cs.src[1]);
  const NodeId cv = Is(c, r2, Op::kReshape) ? Made(c, c.node(r2).src[0]) : NodeId{};
  const NodeId ro = Is(c, cv, Op::kConvert) ? Made(c, c.node(cv).src[0]) : NodeId{};
  const NodeId r1 = Is(c, ro, Op::kRope) ? Made(c, c.node(ro).src[0]) : NodeId{};
  const NodeId m = Is(c, r1, Op::kReshape) ? Made(c, c.node(r1).src[0]) : NodeId{};
  const NodeId n = Is(c, m, Op::kMul) ? Made(c, c.node(m).src[0]) : NodeId{};
  const NodeId po = Is(c, n, Op::kRmsNorm) ? Made(c, c.node(n).src[0]) : NodeId{};
  if (!Is(c, po, Op::kPool) || !Parameter(c, c.node(m).src[1]) || c.node(ro).source(2).valid()) {
    return std::nullopt;
  }
  const auto pool = c.node(po).Params<ir::PoolParams>();
  if (pool.kind != ir::PoolKind::kMean) {
    return std::nullopt;
  }
  Building b(c);
  if (!b.Add({po, n, m, r1, ro, cv, r2, anchor})) {
    return std::nullopt;
  }
  const std::array<TensorId, 1> outs = {cs.out[0]};
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  const TensorId positions = Positions4(c);
  if (!positions.valid()) {
    return std::nullopt;
  }
  const ir::QsaPoolParams params{.eps = c.node(n).Params<ir::NormParams>().eps,
                                 .ratio = pool.ratio,
                                 .rope = c.node(ro).Params<ir::RopeParams>()};
  const std::array<TensorId, 6> s = {c.node(po).src[0], blocks,    c.node(m).src[1],
                                     c.node(ro).src[1], cs.src[2], positions};
  const std::array<const ir::TensorDesc*, 1> o = {&c.tensor(blocks)};
  if (!Has(c, ir::Fused(FusedKind::kQsaPool), Descs(c, s), o, ir::ToBytes(params))) {
    return std::nullopt;
  }
  const StepShape* shape = c.shape();
  const std::uint8_t segment = c.tensor(blocks).segment;
  auto emit = [=](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    ContextOf(r, g.node(anchor));
    const TensorId pos = SegmentPositions(r, shape, positions, segment);
    ContextOf(r, g.node(anchor));
    const std::array<TensorId, 6> src = {r.Map(s[0]), r.Map(s[1]), r.Map(s[2]),
                                         r.Map(s[3]), r.Map(s[4]), pos};
    r.Set(g.node(anchor).out[0], r.builder().Fused(FusedKind::kQsaPool, src, params)[0]);
  };
  return b.Finish({{anchor, 0}}, emit);
}

// qsa_select: an owner's selection past the indexer's budget and its
// attention as llmp.qsa.topk and llmp.qsa.attn: the kept cells listed and
// attended alone (no mask of every cell by every row).
std::optional<Match> QsaSelectMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kAttention) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& at = c.node(anchor);
  const auto ap = at.Params<ir::AttnParams>();
  if (!ap.sparse_any || at.source(4).valid()) {
    return std::nullopt;
  }
  const NodeId pq = Made(c, at.src[0]);
  const NodeId ad2 = Made(c, at.src[3]);
  if (!Is(c, pq, Op::kPermute) || !Is(c, ad2, Op::kAdd) || Only(c, at.src[0]) != anchor ||
      Only(c, at.src[3]) != anchor) {
    return std::nullopt;
  }
  const NodeId rs3 = Made(c, c.node(ad2).src[0]);
  const NodeId vm = Made(c, c.node(ad2).src[1]);
  const NodeId cs = Is(c, rs3, Op::kReshape) ? Made(c, c.node(rs3).src[0]) : NodeId{};
  if (!Is(c, vm, Op::kView) || !Is(c, cs, Op::kCacheStore)) {
    return std::nullopt;
  }
  const NodeId f2 = Made(c, c.node(cs).src[0]);
  const NodeId f3 = Made(c, c.node(cs).src[1]);
  const NodeId tk = Made(c, c.node(cs).src[2]);
  const NodeId ad1 = Is(c, tk, Op::kTopK) ? Made(c, c.node(tk).src[0]) : NodeId{};
  if (!Is(c, f2, Op::kFill) || !Is(c, f3, Op::kFill) || !Is(c, ad1, Op::kAdd)) {
    return std::nullopt;
  }
  const NodeId rs2 = Made(c, c.node(ad1).src[0]);
  const TensorId causal = c.node(ad1).src[1];
  const NodeId rp = Is(c, rs2, Op::kReshape) ? Made(c, c.node(rs2).src[0]) : NodeId{};
  const NodeId rs1 = Is(c, rp, Op::kRepeat) ? Made(c, c.node(rp).src[0]) : NodeId{};
  const NodeId is = Is(c, rs1, Op::kReshape) ? Made(c, c.node(rs1).src[0]) : NodeId{};
  if (!Is(c, is, Op::kIndexScore)) {
    return std::nullopt;
  }
  const ir::Node& isn = c.node(is);
  const NodeId f1 = Made(c, isn.src[2]);
  const NodeId rb = Made(c, isn.src[3]);
  const NodeId scb = Is(c, rb, Op::kReshape) ? Made(c, c.node(rb).src[0]) : NodeId{};
  const NodeId cl = Is(c, scb, Op::kScale) ? Made(c, c.node(scb).src[0]) : NodeId{};
  if (!Is(c, f1, Op::kFill) || !Is(c, cl, Op::kClamp)) {
    return std::nullopt;
  }
  const auto clamp = c.node(cl).Params<ir::ClampParams>();
  const auto bias_scale = c.node(scb).Params<ir::ScaleParams>();
  if (c.node(f1).Params<ir::FillParams>().value != 1.0F ||
      c.node(f2).Params<ir::FillParams>().value != -std::numeric_limits<float>::infinity() ||
      c.node(f3).Params<ir::FillParams>().value != 0.0F || clamp.min != -1e9F ||
      clamp.max != 0.0F || bias_scale.scale != -1.0F || bias_scale.bias != 0.0F ||
      c.node(pq).Params<ir::PermuteParams>().axes != std::array<std::uint8_t, 4>{0, 2, 1, 3}) {
    return std::nullopt;
  }
  const TensorId seen = c.node(cl).src[0];
  const TensorId keys = isn.src[1];
  // The index queries: the rounded rows of the prep (one owner), or a view
  // of them (several).
  TensorId qi = isn.src[0];
  NodeId qi_view{};
  NodeId convert{};
  if (Is(c, Made(c, qi), Op::kView)) {
    qi_view = Made(c, qi);
    qi = c.node(qi_view).src[0];
  }
  if (Is(c, Made(c, qi), Op::kConvert)) {
    convert = Made(c, qi);
  } else {
    return std::nullopt;
  }
  if (c.tensor(c.node(convert).out[0]).format.encoding != Encoding::kBF16) return std::nullopt;
  const TensorId qi_f32 = c.node(convert).src[0];
  // The result's rows.
  const auto ra = OnlyOf(c, at.out[0], Op::kReshape);
  if (!ra) {
    return std::nullopt;
  }
  Building b(c);
  if (!b.Add({pq, anchor, *ra, f1, cl, scb, rb, is, rs1, rp, rs2, ad1, tk, f2, f3, cs, rs3, vm,
              ad2})) {
    return std::nullopt;
  }
  if (qi_view.valid() && !b.Add(qi_view)) {
    return std::nullopt;
  }
  if (!qi_view.valid() && Only(c, c.node(convert).out[0]) == is && !b.Add(convert)) {
    return std::nullopt;
  }
  const std::array<TensorId, 1> outs = {c.node(*ra).out[0]};
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  const TensorId positions = Positions4(c);
  if (!positions.valid()) {
    return std::nullopt;
  }
  const TensorId q = c.node(pq).src[0];
  const TensorId mask = c.node(vm).src[0];
  const auto tkp = c.node(tk).Params<ir::TopKParams>();
  const auto rpd = c.tensor(c.node(rp).out[0]);
  const ir::QsaAttnParams params{
      .attn = ap, .width = tkp.k, .ratio = static_cast<std::uint32_t>(rpd.ne(0))};
  // The index queries in F32 (the selection rounds them): the prep's rows,
  // or the owner's view of them.
  const ir::TensorDesc& qid = c.tensor(qi_f32);
  const std::uint64_t t = c.tensor(q).ne(2);
  std::uint64_t qi_offset = 0;
  if (qi_view.valid()) {
    // The view's offset in rows of the rounded queries, in the F32 ones'.
    const ir::TensorDesc& vd = c.tensor(isn.src[0]);
    const ir::TensorDesc& cd = c.tensor(c.node(convert).out[0]);
    qi_offset = vd.view_offset / cd.nb[2] * qid.nb[2];
  }
  const std::array<TensorId, 9> s = {q,      at.src[1], at.src[2], qi_f32,   keys,
                                     causal, seen,      mask,      positions};
  const std::array<const ir::TensorDesc*, 1> o = {&c.tensor(c.node(*ra).out[0])};
  // The block keys as rows [dim, blocks] (the scoring's view of them is
  // [dim, 1, blocks, 1]).
  const ir::TensorDesc& kd = c.tensor(keys);
  const std::uint64_t n_blocks = kd.ne(1) * kd.ne(2);
  ir::TensorDesc kv = kd;
  kv.format.shape = ir::ShapeOf({kd.ne(0), n_blocks});
  kv.nb = {kd.nb[0], kd.nb[2], kd.nb[2] * n_blocks, kd.nb[2] * n_blocks};
  if (kd.ne(1) != 1 && kd.ne(2) != 1) {
    return std::nullopt;
  }
  {
    std::vector<const ir::TensorDesc*> sd = Descs(c, s);
    ir::TensorDesc qv = qid;
    if (qi_view.valid()) {
      qv.format.shape = ir::ShapeOf({qid.ne(0), qid.ne(1), t});
    }
    sd[3] = &qv;
    sd[4] = &kv;
    if (!Has(c, ir::Fused(FusedKind::kQsaAttention), sd, o, ir::ToBytes(params))) {
      return std::nullopt;
    }
  }
  const StepShape* shape = c.shape();
  const NodeId ran = *ra;
  const std::uint8_t segment = c.tensor(at.out[0]).segment;
  const bool viewed = qi_view.valid();
  auto emit = [=](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    ContextOf(r, g.node(anchor));
    ir::Builder& b = r.builder();
    TensorId qi_in = r.Map(qi_f32);
    if (viewed) {
      const ir::TensorDesc& qd = g.tensor(qi_f32);
      qi_in = b.View(qi_in, ir::ShapeOf({qd.ne(0), qd.ne(1), t}),
                     {qd.nb[0], qd.nb[1], qd.nb[2], qd.nb[2] * t}, qi_offset);
    }
    const TensorId pos = SegmentPositions(r, shape, positions, segment);
    ContextOf(r, g.node(anchor));
    // The keys' rows: a view of what the scoring's view reads.
    // (Of the view's own source, the stored keys, so that the read follows
    // their store.)
    const ir::TensorDesc& kg = g.tensor(s[4]);
    const ir::Node& kn = g.node(g.producer(s[4]));
    const bool of_view = kn.key == Prim(Op::kView);
    const TensorId stored = of_view ? kn.src[0] : s[4];
    const std::uint64_t offset = of_view ? kg.view_offset - g.tensor(stored).view_offset : 0;
    const TensorId key_rows = b.View(r.Map(stored), ir::ShapeOf({kg.ne(0), n_blocks}),
                                     {kv.nb[0], kv.nb[1], kv.nb[2], kv.nb[3]}, offset);
    const std::array<TensorId, 9> src = {r.Map(s[0]), r.Map(s[1]), r.Map(s[2]), qi_in, key_rows,
                                         r.Map(s[5]), r.Map(s[6]), r.Map(s[7]), pos};
    r.Set(g.node(ran).out[0], b.Fused(FusedKind::kQsaAttention, src, params)[0]);
  };
  return b.Finish({{ran, 0}}, emit);
}

// qsa_gate: each head's output times sigmoid of its gate, for an output
// product past the vector product's rows (its input prepared from it:
// llmp.qsa.gate_quantize), as kHeadGate.
std::optional<Match> QsaGateMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kMul) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& g4 = c.node(anchor);
  const NodeId g3 = Made(c, g4.src[1]);
  if (!IsUnary(c, g3, ir::UnaryFn::kSigmoid) || Only(c, g4.src[1]) != anchor) {
    return std::nullopt;
  }
  const NodeId g2 = Made(c, c.node(g3).src[0]);
  const NodeId g1 = Is(c, g2, Op::kCont) ? Made(c, c.node(g2).src[0]) : NodeId{};
  if (!Is(c, g1, Op::kView) || Only(c, c.node(g3).src[0]) != g3 ||
      Only(c, c.node(g2).src[0]) != g2) {
    return std::nullopt;
  }
  const TensorId q_full = c.node(g1).src[0];
  const ir::TensorDesc& qd = c.tensor(q_full);
  const ir::TensorDesc& vd = c.tensor(c.node(g1).out[0]);
  const std::uint64_t head = vd.ne(0);
  if (vd.view_offset != head * 4 || qd.ne(0) != 2 * c.tensor(g4.src[0]).ne(0)) {
    return std::nullopt;
  }
  // Only before a product the vector product does not take.
  const auto o = OnlyOf(c, g4.out[0], Op::kMatMul);
  if (!o) {
    return std::nullopt;
  }
  const ir::TensorDesc& od = c.tensor(c.node(*o).src[0]);
  const ir::TensorDesc prepared = Desc(format::TensorFormat{.encoding = Encoding::kMxfp8,
                                                            .layout = format::Layout::kScaleAtoms,
                                                            .shape = c.tensor(g4.out[0]).shape()});
  {
    const std::array<const ir::TensorDesc*, 2> s = {&od, &prepared};
    const std::array<const ir::TensorDesc*, 1> out = {&c.tensor(c.node(*o).out[0])};
    if (!Has(c, Prim(Op::kMatMul), s, out, c.node(*o).params)) {
      return std::nullopt;
    }
  }
  // The gate and the product over its prepared rows.
  Building b(c);
  const NodeId product = *o;
  if (!b.Add({g1, g2, g3, anchor, product})) {
    return std::nullopt;
  }
  const std::array<TensorId, 1> outs = {c.node(product).out[0]};
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  const ir::HeadGateParams params{.head = static_cast<std::uint32_t>(head)};
  const std::array<TensorId, 2> s = {g4.src[0], q_full};
  const std::array<const ir::TensorDesc*, 1> out = {&prepared};
  if (!Has(c, ir::Fused(FusedKind::kHeadGate), Descs(c, s), out, ir::ToBytes(params))) {
    return std::nullopt;
  }
  auto emit = [=](RewriteContext& r, int id) {
    const ir::Graph& g = r.original();
    ir::Builder& bl = r.builder();
    if (id == 0) {
      ContextOf(r, g.node(anchor));
      const std::array<TensorId, 2> src = {r.Map(s[0]), r.Map(s[1])};
      const std::array<ir::TensorDesc, 1> req = {prepared};
      r.local().assign(1,
                       bl.Emit(ir::Fused(FusedKind::kHeadGate), ir::ToBytes(params), src, req)[0]);
      return;
    }
    const ir::Node& q = g.node(product);
    ContextOf(r, q);
    const std::array<TensorId, 2> src = {r.Map(q.src[0]), r.local()[0]};
    const std::array<ir::TensorDesc, 1> req = {g.tensor(q.out[0])};
    r.Set(q.out[0], bl.Emit(q.key, q.params, src, req)[0]);
  };
  auto match = b.Finish({{anchor, 0}, {product, 1}}, emit);
  if (match) {
    match->numerical_change = NumericalChange::kMxfp8ProductInput;
    match->numerical_sites = {product};
  }
  return match;
}

// --------------------------------------------------------------------- MoE

namespace {

// A per-expert scale gathered a slot (components/moe.cc EmitExperts'
// `gathered`): its reshape, repeat and rows.
struct Gathered {
  NodeId rs, rp, gr;
  TensorId scale;
};

std::optional<Gathered> GatheredAt(const MatchContext& c, TensorId t, TensorId ids) {
  const NodeId gr = Made(c, t);
  if (!Is(c, gr, Op::kGetRows) || c.node(gr).src[1] != ids) {
    return std::nullopt;
  }
  const NodeId rp = Made(c, c.node(gr).src[0]);
  const NodeId rs = Is(c, rp, Op::kRepeat) ? Made(c, c.node(rp).src[0]) : NodeId{};
  if (!Is(c, rs, Op::kReshape) || !Weight(c, c.node(rs).src[0]) ||
      Only(c, c.node(gr).src[0]) != gr || Only(c, c.node(rp).src[0]) != rp) {
    return std::nullopt;
  }
  return Gathered{.rs = rs, .rp = rp, .gr = gr, .scale = c.node(rs).src[0]};
}

// The shared expert's gated sum after the experts' reduce: sigmoid(logit),
// the shared output times it, the add.
struct SharedSum {
  NodeId sg, ms, add;
  TensorId shared, logit;
};

std::optional<SharedSum> SharedSumAt(const MatchContext& c, NodeId reduce) {
  const auto add = OnlyOf(c, c.node(reduce).out[0], Op::kAdd);
  if (!add || c.node(*add).src[0] != c.node(reduce).out[0]) {
    return std::nullopt;
  }
  const NodeId ms = Made(c, c.node(*add).src[1]);
  if (!Is(c, ms, Op::kMul) || Only(c, c.node(*add).src[1]) != *add || c.taken(ms)) {
    return std::nullopt;
  }
  const NodeId sg = Made(c, c.node(ms).src[1]);
  if (!IsUnary(c, sg, ir::UnaryFn::kSigmoid) || Only(c, c.node(ms).src[1]) != ms || c.taken(sg)) {
    return std::nullopt;
  }
  return SharedSum{
      .sg = sg, .ms = ms, .add = *add, .shared = c.node(ms).src[0], .logit = c.node(sg).src[0]};
}

}  // namespace

// moe_router: the routing and the shared expert's gate logit as
// llmp.moe.router (one kernel at the routing's place).
std::optional<Match> MoeRouterMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kRoute) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& rt = c.node(anchor);
  const auto rp = rt.Params<ir::RouteParams>();
  if (rp.kind != ir::RouteKind::kSoftmaxTopK || rt.source(1).valid() || rt.source(2).valid()) {
    return std::nullopt;
  }
  const NodeId product = Made(c, rt.src[0]);
  if (!Is(c, product, Op::kMatMul)) {
    return std::nullopt;
  }
  const TensorId x = c.node(product).src[1];
  // The gate logit: Σ x · gate, by a node of the same component.
  NodeId mx{};
  NodeId sr{};
  for (const NodeId r : c.readers(x)) {
    if (!Is(c, r, Op::kMul) || c.taken(r) || c.node(r).component != rt.component ||
        c.node(r).src[0] != x || !Parameter(c, c.node(r).src[1])) {
      continue;
    }
    const auto s = OnlyOf(c, c.node(r).out[0], Op::kSumRows);
    if (s) {
      mx = r;
      sr = *s;
    }
  }
  if (!mx.valid()) {
    return std::nullopt;
  }
  const TensorId gate_w = c.node(mx).src[1];
  Building b(c);
  if (!b.Add({anchor, mx, sr})) {
    return std::nullopt;
  }
  const std::array<TensorId, 1> leaves = {gate_w};
  if (!b.Pull(anchor, leaves)) {
    return std::nullopt;
  }
  const std::array<TensorId, 3> outs = {rt.out[0], rt.out[1], c.node(sr).out[0]};
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  const std::array<TensorId, 3> s = {rt.src[0], x, gate_w};
  const std::array<const ir::TensorDesc*, 3> o = {&c.tensor(outs[0]), &c.tensor(outs[1]),
                                                  &c.tensor(outs[2])};
  if (!Has(c, ir::Fused(FusedKind::kMoeRouter), Descs(c, s), o, ir::ToBytes(rp))) {
    return std::nullopt;
  }
  const std::vector<NodeId> pulled = b.pulled();
  const NodeId srn = sr;
  auto emit = [=](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    PullLeaves(r, pulled);
    ContextOf(r, g.node(anchor));
    const std::array<TensorId, 3> src = {r.Map(s[0]), r.Map(s[1]), r.Map(s[2])};
    const ir::Outputs out = r.builder().Fused(FusedKind::kMoeRouter, src, rp);
    r.Set(g.node(anchor).out[0], out[0]);
    r.Set(g.node(anchor).out[1], out[1]);
    r.Set(g.node(srn).out[0], out[2]);
  };
  return b.Finish({{anchor, 0}}, emit);
}

// moe_experts: the routed experts of a stacked gate and up (the CUTLASS
// layout's) with the shared expert's gated sum as one node (llmp.moe.gemv
// or the grouped GEMMs, and the combine), at the sum's place.
std::optional<Match> MoeExpertsMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kExpertReduce) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& er = c.node(anchor);
  const TensorId ids = er.src[2];
  const NodeId mi2 = Made(c, er.src[0]);
  if (!Is(c, mi2, Op::kMatMulId) || !er.source(3).valid()) {
    return std::nullopt;
  }
  const auto gd = GatheredAt(c, er.src[3], ids);
  const NodeId gl = Made(c, c.node(mi2).src[1]);
  if (!gd || !Is(c, gl, Op::kGlu)) {
    return std::nullopt;
  }
  const NodeId mg = Made(c, c.node(gl).src[0]);
  const NodeId mu = Made(c, c.node(gl).src[1]);
  if (!Is(c, mg, Op::kMul) || !Is(c, mu, Op::kMul)) {
    return std::nullopt;
  }
  const auto gg = GatheredAt(c, c.node(mg).src[1], ids);
  const auto gu = GatheredAt(c, c.node(mu).src[1], ids);
  const NodeId v1 = Made(c, c.node(mg).src[0]);
  const NodeId v2 = Made(c, c.node(mu).src[0]);
  if (!gg || !gu || !Is(c, v1, Op::kView) || !Is(c, v2, Op::kView) ||
      c.node(v1).src[0] != c.node(v2).src[0]) {
    return std::nullopt;
  }
  const NodeId mi1 = Made(c, c.node(v1).src[0]);
  if (!Is(c, mi1, Op::kMatMulId) || c.node(mi1).src[2] != ids) {
    return std::nullopt;
  }
  const NodeId ri = Made(c, c.node(mi1).src[1]);
  if (!Is(c, ri, Op::kReshape)) {
    return std::nullopt;
  }
  const auto sum = SharedSumAt(c, anchor);
  if (!sum) {
    return std::nullopt;
  }
  Building b(c);
  if (!b.Add({ri, mi1, v1,  v2,     gg->rs, gg->rp, gg->gr, mg,      gu->rs,  gu->rp,  gu->gr,
              mu, gl,  mi2, gd->rs, gd->rp, gd->gr, anchor, sum->sg, sum->ms, sum->add})) {
    return std::nullopt;
  }
  const std::array<TensorId, 1> outs = {c.node(sum->add).out[0]};
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  const TensorId x = c.node(ri).src[0];
  const ir::MoeExpertsParams params{
      .glu = c.node(gl).Params<ir::GluParams>(),
      .hidden = static_cast<std::uint32_t>(c.tensor(c.node(v1).out[0]).ne(0))};
  const std::array<TensorId, 10> s = {c.node(mi1).src[0], c.node(mi2).src[0], x,          ids,
                                      er.src[1],          sum->shared,        sum->logit, gg->scale,
                                      gu->scale,          gd->scale};
  const std::array<const ir::TensorDesc*, 1> o = {&c.tensor(outs[0])};
  if (!Has(c, ir::Fused(FusedKind::kMoeExperts), Descs(c, s), o, ir::ToBytes(params))) {
    return std::nullopt;
  }
  const NodeId add = sum->add;
  auto emit = [=](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    ContextOf(r, g.node(add));
    std::array<TensorId, 10> src{};
    for (std::size_t i = 0; i < s.size(); ++i) src.at(i) = r.Map(s.at(i));
    r.Set(g.node(add).out[0], r.builder().Fused(FusedKind::kMoeExperts, src, params)[0]);
  };
  return b.Finish({{add, 0}}, emit);
}

// moe_glu: the experts' products times their scales and the GLU (GGML's
// expert layout: llmp.moe.glu).
std::optional<Match> MoeGluMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kGlu) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& gl = c.node(anchor);
  if (gl.nsrc != 2 || !gl.src[1].valid()) {
    return std::nullopt;
  }
  const NodeId mg = Made(c, gl.src[0]);
  const NodeId mu = Made(c, gl.src[1]);
  if (!Is(c, mg, Op::kMul) || !Is(c, mu, Op::kMul)) {
    return std::nullopt;
  }
  const NodeId pg = Made(c, c.node(mg).src[0]);
  const NodeId pu = Made(c, c.node(mu).src[0]);
  if (!Is(c, pg, Op::kMatMulId) || !Is(c, pu, Op::kMatMulId)) {
    return std::nullopt;
  }
  const TensorId ids = c.node(pg).src[2];
  const auto gg = GatheredAt(c, c.node(mg).src[1], ids);
  const auto gu = GatheredAt(c, c.node(mu).src[1], ids);
  if (!gg || !gu || c.node(pu).src[2] != ids) {
    return std::nullopt;
  }
  Building b(c);
  if (!b.Add({gg->rs, gg->rp, gg->gr, mg, gu->rs, gu->rp, gu->gr, mu, anchor})) {
    return std::nullopt;
  }
  const std::array<TensorId, 1> outs = {gl.out[0]};
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  const auto params = gl.Params<ir::GluParams>();
  const std::array<TensorId, 5> s = {c.node(pg).out[0], c.node(pu).out[0], ids, gg->scale,
                                     gu->scale};
  const std::array<const ir::TensorDesc*, 1> o = {&c.tensor(gl.out[0])};
  if (!Has(c, ir::Fused(FusedKind::kMoeGlu), Descs(c, s), o, ir::ToBytes(params))) {
    return std::nullopt;
  }
  auto emit = [=](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    ContextOf(r, g.node(anchor));
    std::array<TensorId, 5> src{};
    for (std::size_t i = 0; i < s.size(); ++i) src.at(i) = r.Map(s.at(i));
    r.Set(g.node(anchor).out[0], r.builder().Fused(FusedKind::kMoeGlu, src, params)[0]);
  };
  return b.Finish({{anchor, 0}}, emit);
}

// moe_combine: the experts' reduce (with their down scales, where they have
// them) and the shared expert's gated sum (llmp.moe.combine).
std::optional<Match> MoeCombineMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kExpertReduce) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& er = c.node(anchor);
  const auto sum = SharedSumAt(c, anchor);
  // Prefer the existing gated recipe whenever its complete dataflow matches.
  // An ungated shared output plus an unscaled ordered expert reduction.
  // Keep the exact add order and closed intermediate; no ID sorting or weight
  // normalization is introduced by the replacement.
  if (!er.source(3).valid() && !sum) {
    const auto add = OnlyOf(c, er.out[0], Op::kAdd);
    if (add && c.node(*add).src[0] == er.out[0]) {
      const auto shared = c.node(*add).src[1];
      const std::array<TensorId, 4> sources = {er.src[0], er.src[2], er.src[1], shared};
      const std::array<TensorId, 1> outs = {c.node(*add).out[0]};
      const std::array<const ir::TensorDesc*, 1> outputs = {&c.tensor(outs[0])};
      Building plain(c);
      if (plain.Add({anchor, *add}) && plain.Closed(outs) &&
          Has(c, ir::Fused(FusedKind::kMoeCombine), Descs(c, sources), outputs, {})) {
        auto emit = [=](RewriteContext& r, int) {
          ContextOf(r, r.original().node(*add));
          std::array<TensorId, 4> mapped{};
          for (std::size_t i = 0; i < sources.size(); ++i) mapped[i] = r.Map(sources[i]);
          r.Set(outs[0], r.builder().Fused(FusedKind::kMoeCombine, mapped)[0]);
        };
        return plain.Finish({{*add, 0}}, emit);
      }
    }
  }
  const TensorId ids = er.src[2];
  std::optional<Gathered> gd;
  if (er.source(3).valid()) {
    gd = GatheredAt(c, er.src[3], ids);
    if (!gd) {
      return std::nullopt;
    }
  }
  if (!sum) {
    return std::nullopt;
  }
  Building b(c);
  if (!b.Add({anchor, sum->sg, sum->ms, sum->add})) {
    return std::nullopt;
  }
  if (gd && !b.Add({gd->rs, gd->rp, gd->gr})) {
    return std::nullopt;
  }
  const std::array<TensorId, 1> outs = {c.node(sum->add).out[0]};
  if (!b.Closed(outs)) {
    return std::nullopt;
  }
  std::vector<TensorId> s = {er.src[0], ids, er.src[1], sum->shared, sum->logit};
  if (gd) s.push_back(gd->scale);
  const std::array<const ir::TensorDesc*, 1> o = {&c.tensor(outs[0])};
  if (!Has(c, ir::Fused(FusedKind::kMoeCombine), Descs(c, s), o, {})) {
    return std::nullopt;
  }
  const NodeId add = sum->add;
  auto emit = [=](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    ContextOf(r, g.node(add));
    std::vector<TensorId> src;
    for (const TensorId t : s) src.push_back(r.Map(t));
    r.Set(g.node(add).out[0], r.builder().Fused(FusedKind::kMoeCombine, src)[0]);
  };
  return b.Finish({{add, 0}}, emit);
}

// --------------------------------------------------------- product inputs

namespace {

// The format of an input prepared for MXFP8 tensor-core products.
ir::TensorDesc MxfpRows(const ir::TensorDesc& x) {
  return Desc(format::TensorFormat{
      .encoding = Encoding::kMxfp8, .layout = format::Layout::kScaleAtoms, .shape = x.shape()});
}

ir::TensorDesc Q8Rows(const ir::TensorDesc& x) {
  return Desc(format::TensorFormat{
      .encoding = Encoding::kQ8_1, .layout = format::Layout::kGgmlBlocks, .shape = x.shape()});
}

// Products of input `x`, each with a candidate over it prepared as
// `prepared`: the anchor the first.
std::optional<Match> PreparedInputs(
    const MatchContext& c, NodeId anchor, TensorId x, const ir::TensorDesc& prepared,
    const std::function<bool(const ir::Node&)>& takes, Encoding encoding, format::Layout layout,
    ir::ProductArithmetic arithmetic = ir::ProductArithmetic::kImplementation) {
  std::vector<NodeId> products, unqualified;
  const auto change = encoding == Encoding::kMxfp8 ? NumericalChange::kMxfp8ProductInput
                                                   : NumericalChange::kQ8VectorReduction;
  for (const NodeId r : c.readers(x)) {
    if (r < anchor || c.taken(r) || std::ranges::contains(products, r)) {
      continue;
    }
    if (!Unwritten(c, anchor, r, x)) continue;
    const ir::Node& q = c.node(r);
    if (!takes(q)) {
      continue;
    }
    ir::ParamBytes params = q.params;
    if (arithmetic != ir::ProductArithmetic::kImplementation) {
      // Sharing a preparation requires the selected incumbent's complete
      // arithmetic fact, not a candidate for a different prepared algorithm.
      if (q.key != Prim(Op::kMatMul) || c.view().IncumbentArithmetic(c.graph(), q) != arithmetic)
        continue;
      auto product = q.Params<ir::ProductParams>();
      // A qualified numerical recipe must reach its own candidate selection,
      // rather than first being sealed to the ordinary incumbent's arithmetic.
      // Explicit product contracts still constrain every prepared competitor.
      if (product.arithmetic == ir::ProductArithmetic::kImplementation &&
          (c.view().Qualification(change, c.graph(), q) ||
           std::ranges::contains(c.view().policy().numerical_candidates, change)))
        continue;
      product.arithmetic = arithmetic;
      params = ir::ToBytes(product);
    }
    std::vector<const ir::TensorDesc*> s;
    for (std::size_t i = 0; i < q.nsrc; ++i) {
      const TensorId t = q.src.at(i);
      s.push_back(t == x ? &prepared : (t.valid() ? &c.tensor(t) : nullptr));
    }
    std::vector<const ir::TensorDesc*> o;
    for (const TensorId t : q.outputs()) o.push_back(&c.tensor(t));
    if (!c.view().Candidates(q.key, s, o, params).empty()) {
      if (arithmetic == ir::ProductArithmetic::kImplementation && r != anchor &&
          !c.view().NumericallyQualified(change, c.graph(), q) &&
          !std::ranges::contains(c.view().policy().numerical_candidates, change)) {
        unqualified.push_back(r);
        continue;
      }
      products.push_back(r);
    }
  }
  if (products.empty() || products.front() != anchor) {
    return std::nullopt;
  }
  {
    const std::array<const ir::TensorDesc*, 1> s = {&c.tensor(x)};
    const std::array<const ir::TensorDesc*, 1> o = {&prepared};
    if (c.view().Candidates(Prim(Op::kPrepare), s, o).empty()) {
      return std::nullopt;
    }
  }
  Building b(c);
  for (const NodeId n : products) b.Add(n);
  const std::vector<NodeId> nodes = products;
  auto emit = [x, nodes, encoding, layout, arithmetic](RewriteContext& r, int id) {
    const ir::Graph& g = r.original();
    const ir::Node& q = g.node(nodes.at(static_cast<std::size_t>(id)));
    ContextOf(r, q);
    ir::Builder& b = r.builder();
    if (id == 0) {
      r.local().assign(1, b.Prepare(r.Map(x), encoding, layout));
    }
    std::vector<TensorId> src;
    for (std::size_t i = 0; i < q.nsrc; ++i) {
      const TensorId t = q.src.at(i);
      src.push_back(t == x ? r.local()[0] : r.Map(t));
    }
    std::vector<ir::TensorDesc> req;
    for (const TensorId o : q.outputs()) req.push_back(g.tensor(o));
    ir::ParamBytes params = q.params;
    if (arithmetic != ir::ProductArithmetic::kImplementation) {
      auto product = q.Params<ir::ProductParams>();
      product.arithmetic = arithmetic;
      params = ir::ToBytes(product);
    }
    const ir::Outputs out = b.Emit(q.key, params, src, req);
    for (std::size_t i = 0; i < q.nout; ++i) r.Set(q.out.at(i), out[i]);
  };
  std::vector<std::pair<NodeId, int>> places;
  for (std::size_t i = 0; i < products.size(); ++i) {
    places.emplace_back(products[i], static_cast<int>(i));
  }
  auto match = b.Finish(std::move(places), emit, {x});
  if (match && arithmetic == ir::ProductArithmetic::kImplementation) {
    match->numerical_sites = products;
    match->unqualified_sites = std::move(unqualified);
  }
  return match;
}

}  // namespace

// mxfp8_inputs: the MXFP8 products past the vector product's rows over one
// input read its MXFP8 rows, prepared once (llmp.mxfp8.quantize; the fast
// form's tensor-core products).
std::optional<Match> Mxfp8InputsMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& p = c.node(anchor);
  if (p.key != Prim(Op::kMatMul) || c.taken(anchor)) {
    return std::nullopt;
  }
  const TensorId x = p.src[1];
  const ir::TensorDesc& xd = c.tensor(x);
  if (xd.format.encoding != Encoding::kF32 || xd.ne(2) != 1 || xd.ne(3) != 1) {
    return std::nullopt;
  }
  const auto mxfp8 = [&](const ir::Node& q) {
    return q.key == Prim(Op::kMatMul) && q.src[1] == x &&
           c.tensor(q.src[0]).format.encoding == Encoding::kMxfp8 &&
           c.producer(q.src[0]).key == Prim(Op::kWeight);
  };
  if (!mxfp8(p)) {
    return std::nullopt;
  }
  return PreparedInputs(c, anchor, x, MxfpRows(xd), mxfp8, Encoding::kMxfp8,
                        format::Layout::kScaleAtoms);
}

// vecq_glu: a quantized GLU of decode rows (llmp.vecq's GLU form: dense
// gate and up of one type and shape, or routed experts' without scales),
// for vecq_inputs (round 1) to prepare its input.
std::optional<Match> VecQGluMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kGlu) || c.taken(anchor)) {
    return std::nullopt;
  }
  const ir::Node& gl = c.node(anchor);
  if (gl.nsrc != 2 || !gl.src[1].valid()) {
    return std::nullopt;
  }
  const NodeId pg = Made(c, gl.src[0]);
  const NodeId pu = Made(c, gl.src[1]);
  const bool dense = Is(c, pg, Op::kMatMul) && Is(c, pu, Op::kMatMul);
  const bool routed = Is(c, pg, Op::kMatMulId) && Is(c, pu, Op::kMatMulId);
  if ((!dense && !routed) || c.taken(pg) || c.taken(pu) || Only(c, gl.src[0]) != anchor ||
      Only(c, gl.src[1]) != anchor) {
    return std::nullopt;
  }
  const ir::Node& gn = c.node(pg);
  const ir::Node& un = c.node(pu);
  const auto default_product = [](const ir::Node& node) {
    const auto params = node.Params<ir::ProductParams>();
    return params.accumulate == ir::Accumulate::kDefault && params.hint == ir::ProductHint::kNone &&
           params.arithmetic == ir::ProductArithmetic::kImplementation;
  };
  // The fused GLU encodes default products. Numerical authority for an input
  // preparation cannot erase a separate requested accumulation or recipe seal.
  if (!default_product(gn) || !default_product(un)) return std::nullopt;
  if (gn.src[1] != un.src[1] || (routed && gn.src[2] != un.src[2]) ||
      !(c.tensor(gn.src[0]).format == c.tensor(un.src[0]).format)) {
    return std::nullopt;
  }
  TensorId x = gn.src[1];
  NodeId reshape{};
  if (routed) {
    reshape = Made(c, x);
    if (!Is(c, reshape, Op::kReshape) || c.taken(reshape)) {
      return std::nullopt;
    }
    x = c.node(reshape).src[0];
  }
  const ir::TensorDesc prepared = Q8Rows(c.tensor(x));
  Building b(c);
  if (!b.Add({pg, pu, anchor})) {
    return std::nullopt;
  }
  const std::array<TensorId, 1> outs = {gl.out[0]};
  const auto glu = gl.Params<ir::GluParams>();
  if (dense) {
    const std::array<TensorId, 3> s = {gn.src[0], un.src[0], x};
    std::vector<const ir::TensorDesc*> sd = Descs(c, s);
    sd[2] = &prepared;
    const std::array<const ir::TensorDesc*, 1> o = {&c.tensor(outs[0])};
    if (!Has(c, ir::Fused(FusedKind::kMatMulGlu), sd, o, ir::ToBytes(glu))) {
      return std::nullopt;
    }
    if (!b.Closed(outs)) {
      return std::nullopt;
    }
    // A shared expert beside routed ones (their input's) takes the GLU form
    // only where theirs does (as the fast form runs its MoE: both, or
    // neither).
    for (const NodeId rs : c.readers(x)) {
      if (!Is(c, rs, Op::kReshape)) continue;
      for (const NodeId q : c.readers(c.node(rs).out[0])) {
        if (!Is(c, q, Op::kMatMulId)) continue;
        // (Its GLU may be a match's already.)
        const auto qr = c.readers(c.node(q).out[0]);
        if (qr.size() != 1 || !Is(c, qr.front(), Op::kGlu)) return std::nullopt;
        const ir::Node& rgl = c.node(qr.front());
        const NodeId rpg = Made(c, rgl.src[0]);
        const NodeId rpu = Made(c, rgl.src[1]);
        if (!Is(c, rpg, Op::kMatMulId) || !Is(c, rpu, Op::kMatMulId)) return std::nullopt;
        const ir::MoeExpertsParams rp{
            .glu = rgl.Params<ir::GluParams>(),
            .hidden = static_cast<std::uint32_t>(c.tensor(c.node(rpg).out[0]).ne(0))};
        const std::array<TensorId, 4> rsrc = {c.node(rpg).src[0], c.node(rpu).src[0], x,
                                              c.node(rpg).src[2]};
        std::vector<const ir::TensorDesc*> rd = Descs(c, rsrc);
        rd[2] = &prepared;
        const std::array<const ir::TensorDesc*, 1> ro = {&c.tensor(rgl.out[0])};
        if (!Has(c, ir::Fused(FusedKind::kMatMulIdGlu), rd, ro, ir::ToBytes(rp))) {
          return std::nullopt;
        }
      }
    }
    auto emit = [=](RewriteContext& r, int) {
      const ir::Graph& g = r.original();
      ContextOf(r, g.node(anchor));
      const std::array<TensorId, 3> src = {r.Map(s[0]), r.Map(s[1]), r.Map(s[2])};
      r.Set(g.node(anchor).out[0], r.builder().Fused(FusedKind::kMatMulGlu, src, glu)[0]);
    };
    auto match = b.Finish({{anchor, 0}}, emit);
    if (match) match->numerical_sites = {pg, pu};
    return match;
  }
  // Routed: the reshape of x read by both products only.
  for (const NodeId rd : c.readers(c.node(reshape).out[0])) {
    if (rd != pg && rd != pu) return std::nullopt;
  }
  if (!b.Add(reshape) || !b.Closed(outs)) {
    return std::nullopt;
  }
  const ir::MoeExpertsParams params{
      .glu = glu, .hidden = static_cast<std::uint32_t>(c.tensor(gn.out[0]).ne(0))};
  const std::array<TensorId, 4> s = {gn.src[0], un.src[0], x, gn.src[2]};
  std::vector<const ir::TensorDesc*> sd = Descs(c, s);
  sd[2] = &prepared;
  const std::array<const ir::TensorDesc*, 1> o = {&c.tensor(outs[0])};
  if (!Has(c, ir::Fused(FusedKind::kMatMulIdGlu), sd, o, ir::ToBytes(params))) {
    return std::nullopt;
  }
  auto emit = [=](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    ContextOf(r, g.node(anchor));
    const std::array<TensorId, 4> src = {r.Map(s[0]), r.Map(s[1]), r.Map(s[2]), r.Map(s[3])};
    r.Set(g.node(anchor).out[0], r.builder().Fused(FusedKind::kMatMulIdGlu, src, params)[0]);
  };
  auto match = b.Finish({{anchor, 0}}, emit);
  if (match) match->numerical_sites = {pg, pu};
  return match;
}

// ----------------------------------------------------------------- round 1

// hc_fold_inputs: a fold's prepared and rounded rows (kHcFold's outputs 1
// and 2) for its readers: a preparation of its result into MXFP8 rows or
// BF16 stands for them, and a BF16 product past the vector rows (a router's
// past 8 rows, any past 16) reads the rounded rows.
std::optional<Match> HcFoldInputsMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& n = c.node(anchor);
  if (c.taken(anchor) ||
      !(n.key == Prim(Op::kPrepare) || n.key == Prim(Op::kConvert) || n.key == Prim(Op::kMatMul))) {
    return std::nullopt;
  }
  const TensorId x = n.key == Prim(Op::kMatMul) ? n.src[1] : n.src[0];
  const NodeId f = Made(c, x);
  if (!f.valid() || c.node(f).key != ir::Fused(FusedKind::kHcFold) || c.node(f).out[0] != x) {
    return std::nullopt;
  }
  const auto fp = c.node(f).Params<ir::HcFoldParams>();
  const ir::Node& fn = c.node(f);
  const TensorId prepared = fp.prepared ? fn.out[1] : TensorId{};
  const TensorId rounded = fp.rounded ? fn.out[fp.prepared ? 2 : 1] : TensorId{};
  const ir::TensorDesc& od = c.tensor(n.out[0]);
  TensorId stands{};
  if (n.key == Prim(Op::kPrepare) && prepared.valid() && ir::PreparedRows(od.format)) {
    stands = prepared;
  } else if (n.key == Prim(Op::kConvert) && rounded.valid() &&
             od.format.encoding == Encoding::kBF16) {
    stands = rounded;
  }
  if (stands.valid() && c.tensor(stands).format == od.format) {
    Building b(c);
    b.Add(anchor);
    auto emit = [=](RewriteContext& r, int) {
      r.Set(r.original().node(anchor).out[0], r.Map(stands));
    };
    return b.Finish({{anchor, 0}}, emit);
  }
  if (n.key != Prim(Op::kMatMul) || !rounded.valid() ||
      c.tensor(n.src[0]).format.encoding != Encoding::kBF16) {
    return std::nullopt;
  }
  if (c.tensor(rounded).format.encoding != Encoding::kBF16 ||
      c.view().IncumbentArithmetic(c.graph(), n) != ir::ProductArithmetic::kBf16F32)
    return std::nullopt;
  auto params = n.Params<ir::ProductParams>();
  params.arithmetic = ir::ProductArithmetic::kBf16F32;
  const std::array<const ir::TensorDesc*, 2> s = {&c.tensor(n.src[0]), &c.tensor(rounded)};
  const std::array<const ir::TensorDesc*, 1> o = {&od};
  if (!Has(c, Prim(Op::kMatMul), s, o, ir::ToBytes(params))) {
    return std::nullopt;
  }
  Building b(c);
  b.Add(anchor);
  auto emit = [=](RewriteContext& r, int) {
    const ir::Graph& g = r.original();
    const ir::Node& q = g.node(anchor);
    ContextOf(r, q);
    const std::array<TensorId, 2> src = {r.Map(q.src[0]), r.Map(rounded)};
    const std::array<ir::TensorDesc, 1> req = {g.tensor(q.out[0])};
    r.Set(q.out[0], r.builder().Emit(q.key, ir::ToBytes(params), src, req)[0]);
  };
  return b.Finish({{anchor, 0}}, emit);
}

// Share an incumbent's authenticated BF16 RNE input conversion and F32
// accumulation. The selected source declares these facts for its actual path;
// neither dtype, row count nor a prepared candidate alone proves equivalence.
// A step's outputs (a head) and a fold's reader are separate contracts.
std::optional<Match> Bf16InputsMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& p = c.node(anchor);
  if (p.key != Prim(Op::kMatMul) || c.taken(anchor)) {
    return std::nullopt;
  }
  const TensorId x = p.src[1];
  const ir::TensorDesc& xd = c.tensor(x);
  if (xd.format.encoding != Encoding::kF32 || xd.ne(2) != 1 || xd.ne(3) != 1) {
    return std::nullopt;
  }
  // (A fold's rounded rows are hc_fold_inputs' to give.)
  if (const NodeId f = Made(c, x); f.valid() && c.node(f).key == ir::Fused(FusedKind::kHcFold) &&
                                   c.node(f).Params<ir::HcFoldParams>().rounded) {
    return std::nullopt;
  }
  const auto bf16 = [&](const ir::Node& q) {
    return q.key == Prim(Op::kMatMul) && q.src[1] == x &&
           c.tensor(q.src[0]).format.encoding == Encoding::kBF16 &&
           c.producer(q.src[0]).key == Prim(Op::kWeight) && !c.external(q.out[0]);
  };
  if (!bf16(p)) {
    return std::nullopt;
  }
  ir::TensorDesc rounded = xd;
  rounded.format.encoding = Encoding::kBF16;
  rounded.nb = ir::NaturalStrides(rounded.format);
  rounded.view_of = {};
  rounded.view_offset = 0;
  // A conversion, not a preparation: llmp.bf16.
  std::vector<NodeId> products;
  for (const NodeId r : c.readers(x)) {
    if (r < anchor || c.taken(r) || std::ranges::contains(products, r)) continue;
    if (!Unwritten(c, anchor, r, x)) continue;
    const ir::Node& q = c.node(r);
    if (!bf16(q) || c.view().IncumbentArithmetic(c.graph(), q) != ir::ProductArithmetic::kBf16F32)
      continue;
    auto params = q.Params<ir::ProductParams>();
    params.arithmetic = ir::ProductArithmetic::kBf16F32;
    const std::array<const ir::TensorDesc*, 2> s = {&c.tensor(q.src[0]), &rounded};
    const std::array<const ir::TensorDesc*, 1> o = {&c.tensor(q.out[0])};
    if (!c.view().Candidates(q.key, s, o, ir::ToBytes(params)).empty()) products.push_back(r);
  }
  if (products.empty() || products.front() != anchor) {
    return std::nullopt;
  }
  {
    const std::array<const ir::TensorDesc*, 1> s = {&xd};
    const std::array<const ir::TensorDesc*, 1> o = {&rounded};
    if (c.view().Candidates(Prim(Op::kConvert), s, o).empty()) return std::nullopt;
  }
  Building b(c);
  for (const NodeId q : products) b.Add(q);
  const std::vector<NodeId> nodes = products;
  auto emit = [x, nodes](RewriteContext& r, int id) {
    const ir::Graph& g = r.original();
    const ir::Node& q = g.node(nodes.at(static_cast<std::size_t>(id)));
    ContextOf(r, q);
    ir::Builder& b = r.builder();
    // Converted once, unless a match of the same round made it in BF16
    // (qk_attention's output).
    if (id == 0) {
      const TensorId mx = r.Map(x);
      r.local().assign(
          1, b.tensor(mx).format.encoding == Encoding::kBF16 ? mx : b.Convert(mx, Encoding::kBF16));
    }
    const std::array<TensorId, 2> src = {r.Map(q.src[0]), r.local()[0]};
    const std::array<ir::TensorDesc, 1> req = {g.tensor(q.out[0])};
    auto params = q.Params<ir::ProductParams>();
    params.arithmetic = ir::ProductArithmetic::kBf16F32;
    r.Set(q.out[0], b.Emit(q.key, ir::ToBytes(params), src, req)[0]);
  };
  std::vector<std::pair<NodeId, int>> places;
  for (std::size_t i = 0; i < products.size(); ++i)
    places.emplace_back(products[i], static_cast<int>(i));
  return b.Finish(std::move(places), emit, {x});
}

// bf16_rows: an MXFP8 tensor-core product whose rows only the convolution
// or a gated norm's gate read (through reshapes) records them in BF16, as
// the fast form's products give them.
std::optional<Match> Bf16RowsMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& p = c.node(anchor);
  if (p.key != Prim(Op::kMatMul) || c.taken(anchor) ||
      c.tensor(p.out[0]).format.encoding != Encoding::kF32 ||
      !ir::PreparedRows(c.tensor(p.src[1]).format)) {
    return std::nullopt;
  }
  // Its readers through reshapes and views (an owner's rows): a
  // convolution's rows or a gated norm's gate only. The reshapes and views
  // are the match's, made again over the BF16 rows.
  std::vector<TensorId> frontier = {p.out[0]};
  std::vector<NodeId> shaped;
  bool any = false;
  while (!frontier.empty()) {
    const TensorId t = frontier.back();
    frontier.pop_back();
    if (c.external(t)) return std::nullopt;
    for (const NodeId r : c.readers(t)) {
      const ir::Node& q = c.node(r);
      if (q.key == Prim(Op::kReshape) || q.key == Prim(Op::kView)) {
        if (c.taken(r)) return std::nullopt;
        shaped.push_back(r);
        frontier.push_back(q.out[0]);
      } else if (q.key == ir::Fused(FusedKind::kGdnConv) ||
                 q.key == ir::Fused(FusedKind::kGdnHistory) ||
                 (q.key == ir::Fused(FusedKind::kNormGate) && q.src[2] == t)) {
        any = true;
      } else {
        return std::nullopt;
      }
    }
  }
  if (!any) {
    return std::nullopt;
  }
  ir::TensorDesc out = c.tensor(p.out[0]);
  out.format.encoding = Encoding::kBF16;
  out.nb = ir::NaturalStrides(out.format);
  const std::array<const ir::TensorDesc*, 2> s = {&c.tensor(p.src[0]), &c.tensor(p.src[1])};
  const std::array<const ir::TensorDesc*, 1> o = {&out};
  if (!Has(c, Prim(Op::kMatMul), s, o, p.params)) {
    return std::nullopt;
  }
  Building b(c);
  b.Add(anchor);
  std::ranges::sort(shaped);
  for (const NodeId n : shaped) b.Add(n);
  auto emit = [=](RewriteContext& r, int id) {
    const ir::Graph& g = r.original();
    ir::Builder& bl = r.builder();
    if (id == 0) {
      const ir::Node& q = g.node(anchor);
      ContextOf(r, q);
      const std::array<TensorId, 2> src = {r.Map(q.src[0]), r.Map(q.src[1])};
      const std::array<ir::TensorDesc, 1> req = {out};
      r.Set(q.out[0], bl.Emit(q.key, q.params, src, req)[0]);
      return;
    }
    // A reshape or view of the rows, in BF16 units (half the F32 bytes).
    const ir::Node& q = g.node(shaped.at(static_cast<std::size_t>(id - 1)));
    ContextOf(r, q);
    const ir::TensorDesc& qd = g.tensor(q.out[0]);
    const TensorId src = r.Map(q.src[0]);
    if (q.key == Prim(Op::kReshape)) {
      r.Set(q.out[0], bl.Reshape(src, qd.shape()));
      return;
    }
    const ir::TensorDesc& sd = g.tensor(q.src[0]);
    const std::uint64_t base = sd.is_view() ? sd.view_offset : 0;
    std::array<std::uint64_t, 4> nb{};
    for (std::size_t d = 0; d < 4; ++d) nb.at(d) = qd.nb.at(d) / 2;
    const std::uint64_t offset = (qd.view_offset - base) / 2;  // from its source
    r.Set(q.out[0], bl.View(src, qd.shape(), nb, offset));
  };
  std::vector<std::pair<NodeId, int>> places = {{anchor, 0}};
  for (std::size_t i = 0; i < shaped.size(); ++i) {
    places.emplace_back(shaped[i], static_cast<int>(i + 1));
  }
  return b.Finish(std::move(places), emit);
}

// vecq_inputs: decode rows' quantized products over one input (dense,
// routed, their GLU forms) read its Q8_1 blocks, prepared once (llmp.q8_1
// then llmp.vecq: a GGML-form checkpoint's fast decode).
std::optional<Match> VecQInputsMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& p = c.node(anchor);
  if (c.taken(anchor)) {
    return std::nullopt;
  }
  const auto input = [&](const ir::Node& q) -> TensorId {
    if (q.key == Prim(Op::kMatMul) || q.key == Prim(Op::kMatMulId)) return q.src[1];
    if (q.key == ir::Fused(FusedKind::kMatMulGlu)) return q.src[2];
    if (q.key == ir::Fused(FusedKind::kMatMulIdGlu)) return q.src[2];
    return {};
  };
  const TensorId x = input(p);
  if (!x.valid()) {
    return std::nullopt;
  }
  const ir::TensorDesc& xd = c.tensor(x);
  if (xd.format.encoding != Encoding::kF32 || !ir::IsContiguous(xd)) {
    return std::nullopt;
  }
  const auto takes = [&](const ir::Node& q) {
    if (input(q) != x) return false;
    const TensorId w = q.src[0];
    return c.producer(w).key == Prim(Op::kWeight) &&
           c.tensor(w).format.layout == format::Layout::kGgmlBlocks;
  };
  if (!takes(p)) {
    return std::nullopt;
  }
  // The GLU forms' and the routed product's prepared input is the input
  // itself in their sources (kMatMulIdGlu's x [W, T]; kMatMulId's b, which
  // is the activations [F, U, T] a slot: prepared per slot).
  const ir::TensorDesc prepared = Q8Rows(xd);
  if (auto equivalent =
          PreparedInputs(c, anchor, x, prepared, takes, Encoding::kQ8_1,
                         format::Layout::kGgmlBlocks, ir::ProductArithmetic::kGgmlQ8_1F32))
    return equivalent;
  auto changed =
      PreparedInputs(c, anchor, x, prepared, takes, Encoding::kQ8_1, format::Layout::kGgmlBlocks);
  if (changed) changed->numerical_change = NumericalChange::kQ8VectorReduction;
  return changed;
}

std::optional<GroupedProduct> GroupedProductOf(const ir::Graph& g, NodeId product_id) {
  const auto is = [&](NodeId n, Op op) {
    return n.valid() && n.v < g.nodes().size() && g.node(n).key == Prim(op);
  };
  const auto made = [&](TensorId t) {
    return t.valid() && t.v < g.tensor_count() ? g.producer(t) : NodeId{};
  };
  const auto only = [&](TensorId t) -> std::optional<NodeId> {
    if (!t.valid() || std::ranges::contains(g.outputs(), t) || std::ranges::contains(g.keep(), t))
      return std::nullopt;
    std::optional<NodeId> found;
    for (std::uint32_t i = 0; i < g.nodes().size(); ++i) {
      for (const TensorId s : g.nodes()[i].sources()) {
        if (s != t) continue;
        if (found) return std::nullopt;
        found = NodeId{i};
      }
    }
    return found;
  };
  if (!is(product_id, Op::kMatMul)) return std::nullopt;
  constexpr std::array<std::uint8_t, 4> swap{0, 2, 1, 3};
  const auto& product = g.node(product_id);
  const auto params = product.Params<ir::ProductParams>();
  if (params.accumulate != ir::Accumulate::kDefault || params.hint != ir::ProductHint::kNone ||
      params.arithmetic != ir::ProductArithmetic::kImplementation)
    return std::nullopt;
  const auto output_id = only(product.out[0]);
  if (!output_id || !is(*output_id, Op::kPermute)) return std::nullopt;
  const auto& output = g.node(*output_id);
  if (output.Params<ir::PermuteParams>().axes != swap) return std::nullopt;
  const auto input_id = made(product.src[1]);
  const auto weight_id = made(product.src[0]);
  if (!is(input_id, Op::kPermute) || !is(weight_id, Op::kReshape) ||
      g.node(input_id).Params<ir::PermuteParams>().axes != swap ||
      only(product.src[1]) != product_id)
    return std::nullopt;
  const auto assembly = only(output.out[0]);
  if (!assembly || !is(*assembly, Op::kCont)) return std::nullopt;
  const auto x = g.node(input_id).src[0];
  const auto w = product.src[0];
  const auto leaf = g.node(weight_id).src[0];
  const auto& original = g.tensor(leaf);
  const auto& grouped = g.tensor(w);
  const auto& input = g.tensor(x);
  const auto& packed = g.tensor(g.node(*assembly).out[0]);
  const auto* reference = std::get_if<ir::WeightRef>(&original.binding);
  if (!reference || reference->expert_array ||
      g.tensor(product.out[0]).format.encoding != Encoding::kF32 ||
      g.tensor(output.out[0]).format.encoding != Encoding::kF32 ||
      packed.format.encoding != Encoding::kF32 || !is(made(leaf), Op::kWeight) ||
      original.is_view() || original.format.shape.rank != 2 || original.format.stacked != 1 ||
      original.format.layout != format::Layout::kGgmlBlocks || grouped.view_offset != 0 ||
      grouped.format.shape.rank != 3 || grouped.ne(3) != 1 || grouped.ne(0) != original.ne(0) ||
      grouped.ne(2) < 2 || original.ne(1) / grouped.ne(2) != grouped.ne(1) ||
      original.ne(1) % grouped.ne(2) != 0 || !ir::IsContiguous(grouped) ||
      input.format.encoding != Encoding::kF32 || !ir::IsContiguous(input) ||
      input.ne(0) != grouped.ne(0) || input.ne(1) != grouped.ne(2) || input.ne(3) != 1 ||
      packed.ne(3) != 1 ||
      !((packed.ne(0) == original.ne(1) && packed.ne(1) == input.ne(2) && packed.ne(2) == 1) ||
        (packed.ne(0) == grouped.ne(1) && packed.ne(1) == grouped.ne(2) &&
         packed.ne(2) == input.ne(2))) ||
      product.component != g.node(weight_id).component ||
      product.component != g.node(made(leaf)).component ||
      product.component != g.node(input_id).component || product.component != output.component ||
      product.component != g.node(*assembly).component || !Unwritten(g, input_id, *output_id, x))
    return std::nullopt;
  const auto parts = format::PartsOf(original.format);
  if (!parts || parts->size() != 1 || (*parts)[0].role != format::PartRole::kData)
    return std::nullopt;
  return GroupedProduct{.product = product_id,
                        .input_permute = input_id,
                        .output_permute = *output_id,
                        .assembly = *assembly,
                        .leaf = leaf,
                        .weight = w,
                        .input = x};
}

// A grouped product's actual layout chain, over an immutable 2D weight
// resource reshaped into its complete groups. No equal-shape authority.
std::optional<Match> GroupedVecQMatch(const MatchContext& c, NodeId anchor) {
  if (!Is(c, anchor, Op::kPermute) || c.taken(anchor)) return std::nullopt;
  const auto& output = c.node(anchor);
  const auto product_id = Made(c, output.src[0]);
  const auto grouped_product = GroupedProductOf(c.graph(), product_id);
  if (!grouped_product || grouped_product->output_permute != anchor ||
      Only(c, output.src[0]) != anchor || Only(c, c.node(product_id).src[1]) != product_id)
    return std::nullopt;
  const auto& product = c.node(product_id);
  const auto input_id = grouped_product->input_permute;
  const auto x = grouped_product->input;
  const auto w = grouped_product->weight;
  const auto& grouped = c.tensor(w);
  const auto& input = c.tensor(x);
  const auto params = product.Params<ir::ProductParams>();
  // The final packed assembly remains an ordinary Cont when it is a graph
  // output; otherwise its contiguous grouped result reshapes in place.
  const auto consumer = Only(c, output.out[0]);
  if (!consumer || *consumer != grouped_product->assembly) return std::nullopt;
  const std::array<TensorId, 2> sources{w, x};
  const auto result = DenseDesc(Encoding::kF32, {grouped.ne(1), grouped.ne(2), input.ne(2)});
  const std::array<const ir::TensorDesc*, 1> outputs{&result};
  const auto candidates = c.view().Candidates(ir::Fused(FusedKind::kGroupedMatMul),
                                              Descs(c, sources), outputs, product.params);
  const bool available = std::ranges::any_of(candidates, [&](std::size_t index) {
    const auto& capability = c.view().registry().at(index).capability;
    return capability && std::ranges::all_of(capability->launches, [&](std::string_view name) {
             return c.view().registry().Find(name).has_value();
           });
  });
  if (!available) return std::nullopt;
  const ir::Node& cont_node = c.node(*consumer);
  const TensorId cont_out = cont_node.out[0];
  const bool fold_cont = !c.external(cont_out);
  Building b(c);
  if (fold_cont) {
    if (!b.Add({input_id, product_id, anchor, *consumer}) || !b.Closed(std::array{cont_out}))
      return std::nullopt;
    const auto shape = c.tensor(cont_out).shape();
    auto emit = [=](RewriteContext& r, int) {
      ContextOf(r, r.original().node(anchor));
      const std::array src{r.Map(w), r.Map(x)};
      const TensorId grouped_out = r.builder().Fused(FusedKind::kGroupedMatMul, src, params)[0];
      r.Set(cont_out, r.builder().Reshape(grouped_out, shape));
    };
    auto match = b.Finish({{*consumer, 0}}, emit, {w, x});
    if (match) {
      match->numerical_change = NumericalChange::kQ8VectorReduction;
      match->numerical_sites = {product_id};
    }
    return match;
  }
  if (!b.Add({input_id, product_id, anchor}) || !b.Closed(std::array{output.out[0]}))
    return std::nullopt;
  auto emit = [=](RewriteContext& r, int) {
    ContextOf(r, r.original().node(anchor));
    const std::array src{r.Map(w), r.Map(x)};
    r.Set(output.out[0], r.builder().Fused(FusedKind::kGroupedMatMul, src, params)[0]);
  };
  auto match = b.Finish({{anchor, 0}}, emit, {w, x});
  if (match) {
    match->numerical_change = NumericalChange::kQ8VectorReduction;
    match->numerical_sites = {product_id};
  }
  return match;
}

// ---------------------------------------------------------------- rewrite

void Rewrite(RewriteContext& c, const Match& m, std::size_t index) {
  const auto& e = static_cast<const Emissions&>(*m.detail);
  const int id = e.at.at(index);
  if (id >= 0) {
    e.emit(c, id);
  }
}

}  // namespace llmp::planner::fusions
