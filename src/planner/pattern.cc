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
#include "planner/fusions.h"
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
  return p.accumulate == ir::Accumulate::kDefault && p.hint == ir::ProductHint::kNone &&
         p.arithmetic == ir::ProductArithmetic::kImplementation;
}

// Round-1 precision rewrites keep an earlier preparation's explicit BF16
// arithmetic seal. Only the same already-BF16 operands and authenticated F32
// accumulation contract may carry it through; emitters preserve ProductParams.
// The changed output storage still requires the pattern's numerical authority.
bool Bf16Round1Product(const MatchContext& c, const ir::Node& node) {
  if (DefaultProduct(node)) return true;
  const auto params = node.Params<ir::ProductParams>();
  return params.accumulate == ir::Accumulate::kDefault && params.hint == ir::ProductHint::kNone &&
         params.arithmetic == ir::ProductArithmetic::kBf16F32 &&
         c.tensor(node.src[0]).format.encoding == format::Encoding::kBF16 &&
         c.tensor(node.src[1]).format.encoding == format::Encoding::kBF16 &&
         c.view().IncumbentArithmetic(c.graph(), node) == ir::ProductArithmetic::kBf16F32;
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

// A head's RMS norm times its weight, then one to three plain rotations of
// consecutive sections covering the head, each by its own I32 positions:
// the multi-axis rotation (components/conditioning_blocks.cc RopeEmit).
// It also takes a widening Convert of the weight and a Cont of each
// section's packed positions where only the match reads them: neither
// rounds (own_checks: an I32 copy is not an F32 intermediate).
std::optional<Match> NormRopeSectionsMatch(const MatchContext& c, NodeId anchor) {
  const auto nm = NormMulAt(c, anchor);
  if (!nm) return std::nullopt;
  const ir::Graph& g = c.graph();
  const ir::TensorDesc& xd = c.tensor(nm->x);
  if (xd.format.encoding != format::Encoding::kF32) return std::nullopt;
  const std::uint64_t dim = xd.ne(0), rows = xd.ne(2);
  std::vector<NodeId> nodes{nm->norm, nm->mul};
  TensorId w = nm->w;
  if (c.tensor(w).shape().ne != ir::ShapeOf({dim}).ne) return std::nullopt;
  if (const NodeId convert = g.producer(w); convert.valid() &&
                                            c.node(convert).key == Prim(Op::kConvert) &&
                                            !c.taken(convert) && c.OnlyReader(w) == nm->mul) {
    const TensorId wide = c.node(convert).src[0];
    if (c.tensor(wide).format.encoding == format::Encoding::kBF16 &&
        c.tensor(w).format.encoding == format::Encoding::kF32 &&
        SameShape(c.tensor(wide), c.tensor(w))) {
      nodes.push_back(convert);
      w = wide;
    }
  }
  ir::RopeSectionsParams params{.eps = nm->eps};
  std::vector<TensorId> sources{nm->x, w};
  TensorId last = c.node(nm->mul).out[0];
  std::uint64_t covered = 0;
  while (covered < dim && params.sections < params.dims.size()) {
    const auto rope = Next(c, last);
    if (!rope) return std::nullopt;
    const ir::Node& r = c.node(*rope);
    if (r.key != Prim(Op::kRope) || r.src[0] != last || r.source(2).valid() ||
        !SameShape(c.tensor(r.out[0]), xd))
      return std::nullopt;
    const auto p = r.Params<ir::RopeParams>();
    if ((p.mode != ir::RopeMode::kNormal && p.mode != ir::RopeMode::kNeox) || p.inverse ||
        p.table || p.offset != covered || p.dims == 0 || p.scale != 1.0F || p.ext_factor != 0.0F ||
        p.attn_factor != 1.0F || p.sections != std::array<std::uint32_t, 4>{} ||
        (params.sections != 0 && (p.mode != params.mode || p.base != params.base)))
      return std::nullopt;
    params.mode = p.mode;
    params.base = p.base;
    params.dims.at(params.sections++) = p.dims;
    covered += p.dims;
    TensorId pos = r.src[1];
    if (c.tensor(pos).format.encoding != format::Encoding::kI32 ||
        c.tensor(pos).shape().ne != ir::ShapeOf({rows}).ne)
      return std::nullopt;
    if (const NodeId cont = g.producer(pos);
        cont.valid() && c.node(cont).key == Prim(Op::kCont) && !c.taken(cont) &&
        c.OnlyReader(pos) == *rope &&
        c.tensor(c.node(cont).src[0]).shape().ne == c.tensor(pos).shape().ne &&
        c.tensor(c.node(cont).src[0]).format.encoding == format::Encoding::kI32) {
      nodes.push_back(cont);
      pos = c.node(cont).src[0];
    }
    sources.push_back(pos);
    nodes.push_back(*rope);
    last = r.out[0];
  }
  if (covered != dim) return std::nullopt;
  std::ranges::sort(nodes);
  Match match = Fusion(std::move(nodes), std::move(sources), params);
  // Every elided intermediate but the positions' copies F32 (the
  // decomposition's), and no read moved past a write.
  for (std::size_t i = 0; i + 1 < match.nodes.size(); ++i) {
    const ir::Node& n = c.node(match.nodes[i]);
    if (n.key == Prim(Op::kCont) && c.tensor(n.out[0]).format.encoding == format::Encoding::kI32)
      continue;
    if (c.tensor(n.out[0]).format.encoding != format::Encoding::kF32) return std::nullopt;
  }
  if (!KeepsOrder(c, match) ||
      !Implemented(c, ir::Fused(FusedKind::kRmsNormMulRopeSections), match.sources,
                   c.node(match.nodes.back()).outputs(), match.params))
    return std::nullopt;
  return match;
}

// bf16_glu: a split SwiGLU of two products of BF16 weights (each read by
// it alone, through reshapes), whose result only products of BF16 weights
// read (through reshapes): each products' input converted to BF16 once,
// the gate and up products and the GLU recorded in BF16, its readers'
// products reading that (BF16 gate and up, SwiGluToBf16, a BF16 down
// product). The result rounds the gate, the up
// and the GLU to BF16 where the graph kept F32: a policy enables it (the
// image's).
void ContextOf(RewriteContext& c, const ir::Node& node);

struct Bf16GluPlan final : MatchDetail {
  std::vector<TensorId> inputs;  // the products' distinct F32 inputs
};

std::optional<Match> Bf16GluMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& glu = c.node(anchor);
  if (glu.key != Prim(Op::kGlu) || c.taken(anchor) || glu.nsrc != 2 || !glu.src[1].valid())
    return std::nullopt;
  const auto gp = glu.Params<ir::GluParams>();
  if (gp.kind != ir::GluKind::kSwiGlu || gp.swapped ||
      c.tensor(glu.out[0]).format.encoding != format::Encoding::kF32)
    return std::nullopt;
  const ir::Graph& g = c.graph();
  const auto bf16_weight = [&](TensorId w) {
    const ir::TensorDesc& d = c.tensor(w);
    return d.format.encoding == format::Encoding::kBF16 &&
           d.format.layout == format::Layout::kDense && !d.is_view() &&
           c.producer(w).key == Prim(Op::kWeight);
  };
  std::vector<NodeId> nodes{anchor};
  auto plan = std::make_shared<Bf16GluPlan>();
  // The gate and up: reshapes back to a product, each read once.
  for (std::size_t i = 0; i < 2; ++i) {
    TensorId t = glu.src[i];
    NodeId reader = anchor;
    for (;;) {
      const NodeId at = g.producer(t);
      const ir::Node& n = c.node(at);
      if (c.taken(at) || c.OnlyReader(t) != reader) return std::nullopt;
      nodes.push_back(at);
      if (n.key == Prim(Op::kReshape)) {
        reader = at;
        t = n.src[0];
        continue;
      }
      if (n.key != Prim(Op::kMatMul) || !DefaultProduct(n) || !bf16_weight(n.src[0]) ||
          c.tensor(n.out[0]).format.encoding != format::Encoding::kF32)
        return std::nullopt;
      const TensorId x = n.src[1];
      const ir::TensorDesc& xd = c.tensor(x);
      if (xd.format.encoding != format::Encoding::kF32 || !ir::IsContiguous(xd) || xd.ne(1) <= 16)
        return std::nullopt;
      if (!std::ranges::contains(plan->inputs, x)) plan->inputs.push_back(x);
      break;
    }
  }
  // Its readers: reshapes, then products of BF16 weights reading the
  // result as their input.
  std::vector<TensorId> frontier{glu.out[0]};
  std::vector<NodeId> readers;
  while (!frontier.empty()) {
    const TensorId t = frontier.back();
    frontier.pop_back();
    if (c.external(t) || c.readers(t).empty()) return std::nullopt;
    for (const NodeId r : c.readers(t)) {
      const ir::Node& n = c.node(r);
      if (c.taken(r)) return std::nullopt;
      if (n.key == Prim(Op::kReshape)) {
        nodes.push_back(r);
        frontier.push_back(n.out[0]);
      } else if (n.key == Prim(Op::kMatMul) && n.src[1] == t && n.src[0] != t &&
                 DefaultProduct(n) && bf16_weight(n.src[0])) {
        readers.push_back(r);
      } else {
        return std::nullopt;
      }
    }
  }
  // Implemented as rewritten: the products into BF16 from BF16 inputs, the
  // GLU, the readers from a BF16 input.
  const auto bf16 = [](const ir::TensorDesc& d) {
    ir::TensorDesc r = d;
    r.format.encoding = format::Encoding::kBF16;
    r.nb = ir::NaturalStrides(r.format);
    r.view_of = {};
    r.view_offset = 0;
    return r;
  };
  for (const NodeId n : nodes) {
    const ir::Node& q = c.node(n);
    if (q.key != Prim(Op::kMatMul)) continue;
    const ir::TensorDesc x = bf16(c.tensor(q.src[1])), out = bf16(c.tensor(q.out[0]));
    const std::array<const ir::TensorDesc*, 2> src{&c.tensor(q.src[0]), &x};
    const std::array<const ir::TensorDesc*, 1> o{&out};
    if (c.view().Candidates(q.key, src, o, q.params).empty()) return std::nullopt;
    const std::array<const ir::TensorDesc*, 1> xs{&c.tensor(q.src[1])};
    const std::array<const ir::TensorDesc*, 1> xo{&x};
    if (c.view().Candidates(Prim(Op::kConvert), xs, xo).empty()) return std::nullopt;
  }
  {
    const ir::TensorDesc gate = bf16(c.tensor(glu.src[0])), up = bf16(c.tensor(glu.src[1])),
                         out = bf16(c.tensor(glu.out[0]));
    const std::array<const ir::TensorDesc*, 2> src{&gate, &up};
    const std::array<const ir::TensorDesc*, 1> o{&out};
    if (c.view().Candidates(glu.key, src, o, glu.params).empty()) return std::nullopt;
  }
  for (const NodeId r : readers) {
    const ir::Node& q = c.node(r);
    const ir::TensorDesc x = bf16(c.tensor(q.src[1]));
    const std::array<const ir::TensorDesc*, 2> src{&c.tensor(q.src[0]), &x};
    const std::array<const ir::TensorDesc*, 1> o{&c.tensor(q.out[0])};
    if (c.view().Candidates(q.key, src, o, q.params).empty()) return std::nullopt;
  }
  std::ranges::sort(nodes);
  Match m{
      .nodes = std::move(nodes), .sources = {}, .params = {}, .implementation = 0, .detail = plan};
  if (!KeepsOrder(c, m)) return std::nullopt;
  return m;
}

// `x` in BF16: converted, unless an earlier match made it so
// (split_modulate).
TensorId InBf16(ir::Builder& b, TensorId x) {
  if (b.tensor(x).format.encoding == format::Encoding::kBF16) return x;
  return b.Convert(x, format::Encoding::kBF16);
}

// Each node of a bf16_glu match made again at its own place, in BF16.
void Bf16GluRewrite(RewriteContext& c, const Match& m, std::size_t index) {
  const auto& plan = static_cast<const Bf16GluPlan&>(*m.detail);
  const ir::Graph& g = c.original();
  const ir::Node& n = g.node(m.nodes.at(index));
  ContextOf(c, n);
  ir::Builder& b = c.builder();
  if (c.local().empty()) c.local().assign(plan.inputs.size(), TensorId{});
  const auto in_bf16 = [](const ir::TensorDesc& d) {
    ir::TensorDesc r = d;
    r.format.encoding = format::Encoding::kBF16;
    r.nb = ir::NaturalStrides(r.format);
    r.view_of = {};
    r.view_offset = 0;
    return r;
  };
  if (n.key == Prim(Op::kReshape)) {
    c.Set(n.out[0], b.Reshape(c.Map(n.src[0]), g.tensor(n.out[0]).shape()));
    return;
  }
  if (n.key == Prim(Op::kMatMul)) {
    const auto at =
        static_cast<std::size_t>(std::ranges::find(plan.inputs, n.src[1]) - plan.inputs.begin());
    TensorId& x = c.local().at(at);
    if (!x.valid()) x = InBf16(b, c.Map(n.src[1]));
    const std::array<TensorId, 2> src{c.Map(n.src[0]), x};
    const std::array requested{in_bf16(g.tensor(n.out[0]))};
    c.Set(n.out[0], b.Emit(n.key, n.params, src, requested)[0]);
    return;
  }
  // The GLU.
  const std::array<TensorId, 2> src{c.Map(n.src[0]), c.Map(n.src[1])};
  const std::array requested{in_bf16(g.tensor(n.out[0]))};
  c.Set(n.out[0], b.Emit(n.key, n.params, src, requested)[0]);
}

// qk_attention: the image's attention over its products (round 1): the
// query, key and value products of BF16 weights, the query's and key's
// sectioned norms and rotations (rms_norm_mul_rope_sections), their heads
// transposed for an attention of head 128 with a mask. Made: the products
// in BF16 from their input converted once, the key's norm and rotation in
// BF16, a packed mask and a rotation table shared by every attention of the
// step that reads them (RewriteContext::shared), and the attention with
// the query's norm and rotation in it (kQkAttention), recorded in BF16
// where only products of BF16 weights read it. The result rounds the
// heads to BF16: a policy enables it (the image's).
struct QkAttentionPlan final : MatchDetail {
  std::vector<TensorId> inputs;  // the products' distinct F32 inputs
  NodeId q_rot{}, k_rot{}, attention{};
  std::array<NodeId, 3> products{};  // q, k, v
  bool bf16_out = false;
};

// `t`, read by `reader` alone, through views that are reshapes of their
// whole source (a packed view of all of a tensor's bytes) and reshapes,
// each read by the next alone: the nodes and the tensor they reshape.
std::optional<std::pair<TensorId, std::vector<NodeId>>> Unreshaped(const MatchContext& c,
                                                                   TensorId t, NodeId reader) {
  std::vector<NodeId> nodes;
  for (;;) {
    if (c.OnlyReader(t) != reader) return std::nullopt;
    const NodeId at = c.graph().producer(t);
    const ir::Node& n = c.node(at);
    const bool whole_view = n.key == Prim(Op::kView) && ir::IsContiguous(c.tensor(t)) &&
                            c.tensor(t).view_offset == 0 &&
                            ir::Elements(c.tensor(t).shape()) ==
                                ir::Elements(c.tensor(c.graph().root(n.src[0])).shape()) &&
                            !c.tensor(n.src[0]).is_view();
    if ((n.key != Prim(Op::kReshape) && !whole_view) || c.taken(at)) return std::pair{t, nodes};
    nodes.push_back(at);
    reader = at;
    t = n.src[0];
  }
}

bool OnlyBf16Products(const MatchContext& c, TensorId t);

std::optional<Match> QkAttentionMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& a = c.node(anchor);
  if (a.key != Prim(Op::kAttention) || c.taken(anchor) || a.nsrc < 4 || !a.src[3].valid() ||
      a.source(4).valid())
    return std::nullopt;
  const auto ap = a.Params<ir::AttnParams>();
  if (ap.softcap != 0.0F || ap.max_bias != 0.0F || ap.window != 0 || ap.sinks || ap.sparse_any ||
      ap.n_kv_max != 0 || ap.accumulate != ir::Accumulate::kDefault)
    return std::nullopt;
  const ir::Graph& g = c.graph();
  auto plan = std::make_shared<QkAttentionPlan>();
  plan->attention = anchor;
  std::vector<NodeId> nodes{anchor};
  const auto bf16_weight = [&](TensorId w) {
    const ir::TensorDesc& d = c.tensor(w);
    return d.format.encoding == format::Encoding::kBF16 &&
           d.format.layout == format::Layout::kDense && !d.is_view() &&
           c.producer(w).key == Prim(Op::kWeight);
  };
  // A product of a BF16 weight whose result only `reader` reads.
  const auto product = [&](TensorId t, NodeId reader) -> std::optional<NodeId> {
    const NodeId at = g.producer(t);
    const ir::Node& n = c.node(at);
    if (c.taken(at) || c.OnlyReader(t) != reader || n.key != Prim(Op::kMatMul) ||
        !Bf16Round1Product(c, n) || !bf16_weight(n.src[0]) ||
        c.tensor(n.out[0]).format.encoding != format::Encoding::kF32)
      return std::nullopt;
    const ir::TensorDesc& x = c.tensor(n.src[1]);
    if ((x.format.encoding != format::Encoding::kF32 &&
         x.format.encoding != format::Encoding::kBF16) ||
        !ir::IsContiguous(x) || x.ne(1) <= 16)
      return std::nullopt;
    if (!std::ranges::contains(plan->inputs, n.src[1])) plan->inputs.push_back(n.src[1]);
    return at;
  };
  // A head operand: Cont (for q and k) of the heads [D, R, H] transposed
  // from [D, H, R].
  const auto heads = [&](TensorId t, bool cont) -> std::optional<TensorId> {
    NodeId reader = anchor;
    if (cont) {
      const NodeId at = g.producer(t);
      if (c.node(at).key != Prim(Op::kCont) || c.taken(at) || c.OnlyReader(t) != reader)
        return std::nullopt;
      nodes.push_back(at);
      reader = at;
      t = c.node(at).src[0];
    }
    const NodeId perm = g.producer(t);
    const ir::Node& pn = c.node(perm);
    if (pn.key != Prim(Op::kPermute) || c.taken(perm) || c.OnlyReader(t) != reader ||
        pn.Params<ir::PermuteParams>().axes != std::array<std::uint8_t, 4>{0, 2, 1, 3})
      return std::nullopt;
    nodes.push_back(perm);
    auto under = Unreshaped(c, pn.src[0], perm);
    if (!under) return std::nullopt;
    nodes.insert(nodes.end(), under->second.begin(), under->second.end());
    return under->first;
  };
  const std::uint64_t head_dim = c.tensor(a.src[0]).ne(0);
  if (head_dim == 0 || head_dim % 2 != 0 || c.tensor(a.src[1]).ne(0) != head_dim ||
      c.tensor(a.src[2]).ne(0) != head_dim) {
    return std::nullopt;
  }
  // q and k: their sectioned norms and rotations of products' heads.
  const auto rotated = [&](TensorId t, NodeId& rot, NodeId& prod) -> bool {
    const auto heads_of = heads(t, true);
    if (!heads_of) return false;
    rot = g.producer(*heads_of);
    const ir::Node& r = c.node(rot);
    if (r.key != ir::Fused(FusedKind::kRmsNormMulRopeSections) || c.taken(rot) ||
        c.tensor(*heads_of).ne(0) != head_dim)
      return false;
    const auto rp = r.Params<ir::RopeSectionsParams>();
    if (rp.mode != ir::RopeMode::kNormal) return false;
    nodes.push_back(rot);
    auto under = Unreshaped(c, r.src[0], rot);
    if (!under) return false;
    nodes.insert(nodes.end(), under->second.begin(), under->second.end());
    const auto p = product(under->first, under->second.empty() ? rot : under->second.back());
    if (!p) return false;
    prod = *p;
    nodes.push_back(prod);
    return true;
  };
  if (!rotated(a.src[0], plan->q_rot, plan->products[0]) ||
      !rotated(a.src[1], plan->k_rot, plan->products[1]))
    return std::nullopt;
  {
    const auto v = heads(a.src[2], false);
    if (!v) return std::nullopt;
    const NodeId last = nodes.back();
    const auto p = product(*v, last);
    if (!p) return std::nullopt;
    plan->products[2] = *p;
    nodes.push_back(*p);
  }
  // Its readers: products of BF16 weights through reshapes, each with a
  // candidate from a BF16 input (then BF16).
  plan->bf16_out = OnlyBf16Products(c, a.out[0]);
  std::ranges::sort(nodes);
  if (std::ranges::adjacent_find(nodes) != nodes.end()) return std::nullopt;
  // Implemented as made.
  const auto bf16 = [](ir::TensorDesc d) {
    d.format.encoding = format::Encoding::kBF16;
    d.nb = ir::NaturalStrides(d.format);
    d.view_of = {};
    d.view_offset = 0;
    return d;
  };
  for (const NodeId n : plan->products) {
    const ir::Node& q = c.node(n);
    const ir::TensorDesc x = bf16(c.tensor(q.src[1])), out = bf16(c.tensor(q.out[0]));
    const std::array<const ir::TensorDesc*, 2> src{&c.tensor(q.src[0]), &x};
    const std::array<const ir::TensorDesc*, 1> o{&out};
    if (c.view().Candidates(q.key, src, o, q.params).empty()) return std::nullopt;
  }
  const ir::Node& kr = c.node(plan->k_rot);
  {
    const ir::TensorDesc x = bf16(c.tensor(kr.src[0])), out = bf16(c.tensor(kr.out[0]));
    std::vector<const ir::TensorDesc*> src{&x};
    for (std::size_t i = 1; i < kr.nsrc; ++i) src.push_back(&c.tensor(kr.src[i]));
    const std::array<const ir::TensorDesc*, 1> o{&out};
    if (c.view().Candidates(kr.key, src, o, kr.params).empty()) return std::nullopt;
  }
  const ir::Node& qr = c.node(plan->q_rot);
  const auto rope = qr.Params<ir::RopeSectionsParams>();
  const ir::TensorDesc& mask = c.tensor(a.src[3]);
  const std::uint64_t rows = c.tensor(a.src[0]).ne(1);
  if (mask.format.encoding != format::Encoding::kF32 || mask.ne(0) != rows || mask.ne(1) != rows)
    return std::nullopt;
  const ir::MaskPackParams mp{.cells = static_cast<std::uint32_t>((rows + 255) / 256 * 256),
                              .rows = static_cast<std::uint32_t>(rows)};
  ir::TensorDesc packed;
  packed.format = ir::Dense(format::Encoding::kF16, ir::ShapeOf({mp.cells, mp.rows}));
  packed.nb = ir::NaturalStrides(packed.format);
  ir::TensorDesc table;
  table.format = ir::Dense(format::Encoding::kF32, ir::ShapeOf({2, head_dim / 2, rows}));
  table.nb = ir::NaturalStrides(table.format);
  {
    const std::array<const ir::TensorDesc*, 1> src{&mask};
    const std::array<const ir::TensorDesc*, 1> o{&packed};
    if (c.view()
            .Candidates(ir::Fused(FusedKind::kAttentionMaskPack), src, o, ir::ToBytes(mp))
            .empty())
      return std::nullopt;
  }
  {
    std::vector<const ir::TensorDesc*> src;
    for (std::size_t i = 2; i < qr.nsrc; ++i) src.push_back(&c.tensor(qr.src[i]));
    const std::array<const ir::TensorDesc*, 1> o{&table};
    if (c.view().Candidates(ir::Fused(FusedKind::kRopeTable), src, o, ir::ToBytes(rope)).empty())
      return std::nullopt;
  }
  {
    const ir::TensorDesc q = bf16(c.tensor(c.node(plan->products[0]).out[0]));
    const ir::TensorDesc k = bf16(c.tensor(c.node(plan->products[1]).out[0]));
    const ir::TensorDesc v = bf16(c.tensor(c.node(plan->products[2]).out[0]));
    ir::TensorDesc out = c.tensor(c.node(plan->products[0]).out[0]);
    if (plan->bf16_out) out = bf16(out);
    const std::array<const ir::TensorDesc*, 6> src{&q,    &c.tensor(qr.src[1]), &k, &v, &packed,
                                                   &table};
    const std::array<const ir::TensorDesc*, 1> o{&out};
    const ir::QkAttentionParams qp{.eps = rope.eps, .scale = ap.scale};
    if (c.view().Candidates(ir::Fused(FusedKind::kQkAttention), src, o, ir::ToBytes(qp)).empty())
      return std::nullopt;
  }
  Match m{
      .nodes = std::move(nodes), .sources = {}, .params = {}, .implementation = 0, .detail = plan};
  if (!KeepsOrder(c, m)) return std::nullopt;
  return m;
}

void QkAttentionRewrite(RewriteContext& c, const Match& m, std::size_t index) {
  const auto& plan = static_cast<const QkAttentionPlan&>(*m.detail);
  const ir::Graph& g = c.original();
  const NodeId id = m.nodes.at(index);
  const ir::Node& n = g.node(id);
  ir::Builder& b = c.builder();
  // local: the converted inputs, then the made q, k (rotated), v.
  const std::size_t inputs = plan.inputs.size();
  if (c.local().empty()) c.local().assign(inputs + 3, TensorId{});
  const auto in_bf16 = [](ir::TensorDesc d) {
    d.format.encoding = format::Encoding::kBF16;
    d.nb = ir::NaturalStrides(d.format);
    d.view_of = {};
    d.view_offset = 0;
    return d;
  };
  for (std::size_t i = 0; i < 3; ++i) {
    if (id != plan.products[i]) continue;
    ContextOf(c, n);
    const auto at =
        static_cast<std::size_t>(std::ranges::find(plan.inputs, n.src[1]) - plan.inputs.begin());
    TensorId& x = c.local().at(at);
    if (!x.valid()) x = InBf16(b, c.Map(n.src[1]));
    const std::array<TensorId, 2> src{c.Map(n.src[0]), x};
    const std::array requested{in_bf16(g.tensor(n.out[0]))};
    c.local().at(inputs + i) = b.Emit(n.key, n.params, src, requested)[0];
    return;
  }
  if (id == plan.k_rot) {
    // The key's norm and rotation in BF16, over its product's heads.
    ContextOf(c, n);
    const TensorId k = c.local().at(inputs + 1);
    std::vector<TensorId> src{b.Reshape(k, g.tensor(n.src[0]).shape())};
    for (std::size_t i = 1; i < n.nsrc; ++i) src.push_back(c.Map(n.src[i]));
    const std::array requested{in_bf16(g.tensor(n.out[0]))};
    c.local().at(inputs + 1) = b.Emit(n.key, n.params, src, requested)[0];
    return;
  }
  if (id != plan.attention) return;  // an elided intermediate
  ContextOf(c, n);
  const ir::Node& qr = g.node(plan.q_rot);
  const auto rope = qr.Params<ir::RopeSectionsParams>();
  const auto rows = static_cast<std::uint32_t>(g.tensor(n.src[0]).ne(1));
  // The packed mask and the table, once for every attention reading them.
  auto& packed = c.shared()[std::format("qk_attention.mask.{}", n.src[3].v)];
  if (!packed.valid()) {
    const ir::MaskPackParams mp{.cells = (rows + 255) / 256 * 256, .rows = rows};
    const std::array<TensorId, 1> src{c.Map(n.src[3])};
    packed = b.Fused(FusedKind::kAttentionMaskPack, src, mp)[0];
  }
  std::string key = std::format("qk_attention.table.{}.{}.{}", rope.base, rope.sections,
                                static_cast<int>(rope.mode));
  std::vector<TensorId> positions;
  for (std::size_t i = 2; i < qr.nsrc; ++i) {
    positions.push_back(c.Map(qr.src[i]));
    key += std::format(".{}:{}", qr.src[i].v, rope.dims.at(i - 2));
  }
  auto& table = c.shared()[key];
  if (!table.valid()) table = b.Fused(FusedKind::kRopeTable, positions, rope)[0];
  const TensorId q = c.local().at(inputs + 0);
  const TensorId k = c.local().at(inputs + 1);
  const TensorId v = c.local().at(inputs + 2);
  const auto& kd = b.tensor(k);
  const TensorId k_rows = b.Reshape(k, ir::ShapeOf({kd.ne(0) * kd.ne(1), kd.ne(2)}));
  const std::array<TensorId, 6> sources{q, c.Map(qr.src[1]), k_rows, v, packed, table};
  ir::TensorDesc out = g.tensor(g.node(plan.products[0]).out[0]);
  if (plan.bf16_out) out = in_bf16(out);
  const std::array<format::Encoding, 1> encodings{out.format.encoding};
  const ir::QkAttentionParams qp{.eps = rope.eps, .scale = n.Params<ir::AttnParams>().scale};
  const TensorId made = b.Fused(FusedKind::kQkAttention, sources, qp, encodings)[0];
  c.Set(n.out[0], b.Reshape(made, g.tensor(n.out[0]).shape()));
}

// A conditioning split between two row ranges, as the layout pass makes a
// modulation's conditioning rows (ir/layout.h): over rows [C, R], a [C]
// vector repeated over the first `split` rows, another over the rest (or
// one over all). The image's blocks share their conditioning rows: the
// match takes the rows' nodes only where `reader` alone reads them (else
// they stay for their other readers, and the fused node reads the vectors).
struct SplitSide {
  std::vector<NodeId> nodes;
  TensorId a{}, b{};
  std::uint32_t split = 0;
};
std::optional<SplitSide> SplitSideOf(const MatchContext& c, TensorId v, NodeId reader) {
  const ir::Graph& g = c.graph();
  const ir::TensorDesc& vd = c.tensor(v);
  if (vd.shape().rank > 2 || c.external(v)) return std::nullopt;
  const std::uint64_t cols = vd.ne(0), rows = vd.ne(1);
  // A [C] vector (or [C, 1]) repeated over n rows: its source.
  const auto repeated = [&](TensorId t, std::uint64_t n) -> std::optional<TensorId> {
    const ir::Node& r = c.producer(t);
    if (r.key != Prim(Op::kRepeat) || c.tensor(t).shape().ne != ir::ShapeOf({cols, n}).ne)
      return std::nullopt;
    const ir::TensorDesc& src = c.tensor(r.src[0]);
    if (src.format.encoding != format::Encoding::kF32 || src.shape().ne != ir::ShapeOf({cols}).ne)
      return std::nullopt;
    return r.src[0];
  };
  const NodeId at = g.producer(v);
  const ir::Node& n = c.node(at);
  SplitSide side;
  std::vector<std::pair<NodeId, TensorId>> chain;  // each node, and its result
  if (n.key == Prim(Op::kConcat) && n.Params<ir::ConcatParams>().dim == 1) {
    const std::uint64_t first = c.tensor(n.src[0]).ne(1);
    const auto a = repeated(n.src[0], first);
    const auto b = repeated(n.src[1], rows - first);
    if (!a || !b) return std::nullopt;
    side = {.nodes = {}, .a = *a, .b = *b, .split = static_cast<std::uint32_t>(first)};
    chain = {{at, v}, {g.producer(n.src[0]), n.src[0]}, {g.producer(n.src[1]), n.src[1]}};
  } else if (const auto a = repeated(v, rows)) {
    side = {.nodes = {}, .a = *a, .b = *a, .split = static_cast<std::uint32_t>(rows)};
    chain = {{at, v}};
  } else {
    return std::nullopt;
  }
  // Taken whole where nothing else reads them.
  bool owned = c.OnlyReader(v) == reader;
  for (std::size_t k = 1; k < chain.size() && owned; ++k)
    owned = c.OnlyReader(chain[k].second) == at;
  for (const auto& [node, result] : chain) owned = owned && !c.taken(node);
  if (owned)
    for (const auto& [node, result] : chain) side.nodes.push_back(node);
  return side;
}

// Every observer must already round this input to BF16 under its selected
// numerical recipe. A possible BF16 consumer alone is not authority.
// Copy() transfers the fact to each changed product's final selection.
bool OnlyBf16Products(const MatchContext& c, TensorId t) {
  std::vector<TensorId> frontier{t};
  while (!frontier.empty()) {
    const TensorId u = frontier.back();
    frontier.pop_back();
    if (c.external(u) || c.readers(u).empty()) return false;
    for (const NodeId r : c.readers(u)) {
      const ir::Node& n = c.node(r);
      if (n.key == Prim(Op::kReshape)) {
        frontier.push_back(n.out[0]);
        continue;
      }
      const ir::TensorDesc& w = c.tensor(n.src[0]);
      if (c.taken(r) || n.key != Prim(Op::kMatMul) || n.src[1] != u || n.src[0] == u ||
          !DefaultProduct(n) || w.format.encoding != format::Encoding::kBF16 || w.is_view() ||
          c.producer(n.src[0]).key != Prim(Op::kWeight))
        return false;
      ir::TensorDesc x = c.tensor(u);
      x.format.encoding = format::Encoding::kBF16;
      x.nb = ir::NaturalStrides(x.format);
      x.view_of = {};
      x.view_offset = 0;
      if (c.view().IncumbentArithmetic(c.graph(), n) != ir::ProductArithmetic::kBf16F32)
        return false;
      auto params = n.Params<ir::ProductParams>();
      params.arithmetic = ir::ProductArithmetic::kBf16F32;
      const std::array<const ir::TensorDesc*, 2> src{&w, &x};
      const std::array<const ir::TensorDesc*, 1> o{&c.tensor(n.out[0])};
      if (c.view().Candidates(n.key, src, o, ir::ToBytes(params)).empty()) return false;
    }
  }
  return true;
}

struct SplitModulatePlan final : MatchDetail {
  bool bf16 = false;  // recorded in BF16 (OnlyBf16Products)
};

// split_modulate: a layer norm times one plus a conditioning split between
// two row ranges (SplitSideOf): fused.layer_norm_modulate_split. BF16
// output is legal only when all selected consumers already perform that
// same conversion; otherwise the exact F32 result remains.
std::optional<Match> SplitModulateMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& m = c.node(anchor);
  if (m.key != Prim(Op::kMul) || c.taken(anchor)) return std::nullopt;
  for (std::size_t i = 0; i < 2; ++i) {
    const TensorId l = m.src[i], sv = m.src[1 - i];
    const NodeId ln = c.graph().producer(l);
    const NodeId sc = c.graph().producer(sv);
    const ir::Node& norm = c.node(ln);
    const ir::Node& scale = c.node(sc);
    if (norm.key != Prim(Op::kLayerNorm) || scale.key != Prim(Op::kScale) || c.taken(ln) ||
        c.taken(sc) || c.OnlyReader(l) != anchor || c.OnlyReader(sv) != anchor)
      continue;
    const auto sp = scale.Params<ir::ScaleParams>();
    if (sp.scale != 1.0F || sp.bias != 1.0F) continue;
    const ir::TensorDesc& x = c.tensor(norm.src[0]);
    if (x.format.encoding != format::Encoding::kF32 || !ir::IsContiguous(x) || x.shape().rank > 2 ||
        c.tensor(m.out[0]).shape().ne != x.shape().ne)
      continue;
    const auto side = SplitSideOf(c, scale.src[0], sc);
    if (!side) continue;
    std::vector<NodeId> nodes{anchor, ln, sc};
    nodes.insert(nodes.end(), side->nodes.begin(), side->nodes.end());
    std::ranges::sort(nodes);
    const ir::SplitModulateParams params{.eps = norm.Params<ir::NormParams>().eps,
                                         .split = side->split};
    Match match = Fusion(std::move(nodes), {norm.src[0], side->a, side->b}, params);
    auto plan = std::make_shared<SplitModulatePlan>();
    ir::TensorDesc out = c.tensor(m.out[0]);
    if (OnlyBf16Products(c, m.out[0])) {
      plan->bf16 = true;
      out.format.encoding = format::Encoding::kBF16;
      out.nb = ir::NaturalStrides(out.format);
    }
    std::array<const ir::TensorDesc*, 3> src{&x, &c.tensor(side->a), &c.tensor(side->b)};
    const std::array<const ir::TensorDesc*, 1> o{&out};
    if (c.view()
            .Candidates(ir::Fused(FusedKind::kLayerNormModulateSplit), src, o, match.params)
            .empty())
      continue;
    if (!ElidesF32(c, match) || !KeepsOrder(c, match)) continue;
    match.detail = std::move(plan);
    return match;
  }
  return std::nullopt;
}

// The modulation recorded in BF16 where only BF16 products read it.
void SplitModulateRewrite(RewriteContext& c, const Match& m, std::size_t index) {
  if (index + 1 != m.nodes.size()) return;
  const auto& plan = static_cast<const SplitModulatePlan&>(*m.detail);
  const ir::Node& last = c.original().node(m.nodes.back());
  ContextOf(c, last);
  std::vector<TensorId> sources;
  for (const TensorId s : m.sources) sources.push_back(c.Map(s));
  const std::array encodings{plan.bf16 ? format::Encoding::kBF16 : format::Encoding::kF32};
  const ir::Outputs outs =
      c.builder().FusedBytes(FusedKind::kLayerNormModulateSplit, sources, m.params, encodings);
  c.Set(last.out[0], outs[0]);
}

// split_gated_residual: carry + branch · tanh of a conditioning split
// between two row ranges: fused.gated_residual_split.
std::optional<Match> SplitGatedResidualMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& add = c.node(anchor);
  if (add.key != Prim(Op::kAdd) || c.taken(anchor)) return std::nullopt;
  const ir::Graph& g = c.graph();
  for (std::size_t i = 0; i < 2; ++i) {
    const TensorId carry = add.src[i], mt = add.src[1 - i];
    const NodeId mn = g.producer(mt);
    const ir::Node& mul = c.node(mn);
    if (mul.key != Prim(Op::kMul) || c.taken(mn) || c.OnlyReader(mt) != anchor) continue;
    for (std::size_t j = 0; j < 2; ++j) {
      const TensorId branch = mul.src[j], tt = mul.src[1 - j];
      const NodeId tn = g.producer(tt);
      const ir::Node& tanh = c.node(tn);
      if (tanh.key != Prim(Op::kUnary) || c.taken(tn) || c.OnlyReader(tt) != mn ||
          tanh.Params<ir::UnaryParams>().fn != ir::UnaryFn::kTanh)
        continue;
      const ir::TensorDesc& cd = c.tensor(carry);
      const ir::TensorDesc& bd = c.tensor(branch);
      if (cd.format.encoding != format::Encoding::kF32 || cd.shape().rank > 2 ||
          !ir::IsContiguous(cd) || !ir::IsContiguous(bd) || bd.shape().ne != cd.shape().ne ||
          c.tensor(add.out[0]).shape().ne != cd.shape().ne)
        continue;
      const auto side = SplitSideOf(c, tanh.src[0], tn);
      if (!side) continue;
      std::vector<NodeId> nodes{anchor, mn, tn};
      nodes.insert(nodes.end(), side->nodes.begin(), side->nodes.end());
      std::ranges::sort(nodes);
      Match match = Fusion(std::move(nodes), {carry, branch, side->a, side->b},
                           ir::SplitModulateParams{.eps = 0.0F, .split = side->split});
      if (!Implemented(c, ir::Fused(FusedKind::kGatedResidualSplit), match.sources, add.outputs(),
                       match.params) ||
          !ElidesF32(c, match) || !KeepsOrder(c, match))
        continue;
      return match;
    }
  }
  return std::nullopt;
}

// split_gated_residual_modulate (round 1): a split gated residual and the
// split modulation of its sum, one split: fused.gated_residual_modulate_split,
// the sum still an output (the next residual's carry reads it). With the
// option bf16_residual (a policy's, the image's: rounding there), the sum
// in BF16 where only later residuals read it, as their carry.
//
// Version 2: the modulation may read a range of the sum's rows (a view of
// whole rows, copied or not: the image's output norm over its latent rows,
// the text rows before them dropped). Norm and modulation are row-wise, so
// the fused node modulates every row and its readers read the range of
// them; its split is the residual's, where the modulation's (counted from
// the range's first row) selects the same vectors. The sum nothing else
// reads is then recorded in BF16 (never read, its rounding is none).
struct SplitResidualPlan final : MatchDetail {
  bool bf16_sum = false;
  // A range: its first row and row count (rows == 0: the whole sum).
  std::uint64_t first = 0, rows = 0;
};
// The descriptor of rows [first, first + rows) of the contiguous [C, R]
// `whole`, as a view.
ir::TensorDesc RowRange(const ir::TensorDesc& whole, std::uint64_t first, std::uint64_t rows) {
  ir::TensorDesc d = whole;
  d.format.shape.ne[1] = rows;
  d.nb = ir::NaturalStrides(d.format);
  d.view_offset = first * d.nb[1];
  return d;
}
std::optional<Match> SplitGatedResidualModulateMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& m = c.node(anchor);
  if (m.key != ir::Fused(FusedKind::kLayerNormModulateSplit) || c.taken(anchor))
    return std::nullopt;
  // The modulation's input: the sum, or a range of its rows (a view, then
  // perhaps a copy of it, each read by the next alone).
  std::vector<NodeId> between;
  TensorId x = m.src[0];
  for (int link = 0; link < 2; ++link) {
    const NodeId p = c.graph().producer(x);
    const ir::Node& n = c.node(p);
    if ((n.key != Prim(Op::kView) && n.key != Prim(Op::kCont)) || c.taken(p) ||
        c.external(n.out[0]) || c.OnlyReader(n.out[0]) != (between.empty() ? anchor : between[0]))
      break;
    between.insert(between.begin(), p);
    x = n.src[0];
  }
  const NodeId at = c.graph().producer(x);
  const ir::Node& g = c.node(at);
  if (g.key != ir::Fused(FusedKind::kGatedResidualSplit) || c.taken(at)) return std::nullopt;
  const auto gp = g.Params<ir::SplitModulateParams>();
  const auto mp = m.Params<ir::SplitModulateParams>();
  const ir::TensorDesc& sum = c.tensor(g.out[0]);
  std::uint64_t first = 0, rows = 0;
  if (!between.empty()) {
    // One view of whole rows of the packed sum; a copy only of all of it.
    // The modulation of the range alone is the step's (a view of every
    // row's then stands for it): never one the step returns or keeps.
    if (c.external(m.out[0])) return std::nullopt;
    const ir::Node& v = c.node(between.front());
    const ir::TensorDesc& vd = c.tensor(v.out[0]);
    if (v.key != Prim(Op::kView) || vd.view_of != c.graph().root(g.out[0]) ||
        !ir::IsContiguous(sum) || sum.is_view() || sum.ne(2) != 1 || sum.ne(3) != 1 ||
        vd.ne(0) != sum.ne(0) || vd.ne(2) != 1 || vd.ne(3) != 1 ||
        vd.format !=
            [&] {
              auto f = sum.format;
              f.shape = vd.format.shape;
              return f;
            }() ||
        vd.nb[1] != sum.nb[1] || vd.view_offset % sum.nb[1] != 0)
      return std::nullopt;
    first = vd.view_offset / sum.nb[1];
    rows = vd.ne(1);
    if (rows == 0 || first + rows > sum.ne(1) || !ir::IsContiguous(vd)) return std::nullopt;
    if (between.size() == 2) {
      const ir::TensorDesc& copy = c.tensor(c.node(between[1]).out[0]);
      if (copy.format != vd.format || copy.is_view()) return std::nullopt;
    }
    // The modulation's rows below its split take a: those below the
    // residual's split counted from the range's first row, or one vector.
    const auto clamp = [&](std::int64_t v) {
      return std::clamp<std::int64_t>(v, 0, static_cast<std::int64_t>(rows));
    };
    if (m.src[1] != m.src[2] && clamp(mp.split) != clamp(static_cast<std::int64_t>(gp.split) -
                                                         static_cast<std::int64_t>(first)))
      return std::nullopt;
  } else if (gp.split != mp.split) {
    return std::nullopt;
  }
  // Made at the modulation's place: nothing between reads the sum.
  for (const NodeId r : c.readers(g.out[0]))
    if (r.v < anchor.v && (between.empty() || r != between.front())) return std::nullopt;
  ir::SplitModulateParams params = mp;
  params.split = gp.split;
  Match match{.nodes = {at},
              .sources = {g.src[0], g.src[1], g.src[2], g.src[3], m.src[1], m.src[2]},
              .params = ir::ToBytes(params)};
  match.nodes.insert(match.nodes.end(), between.begin(), between.end());
  match.nodes.push_back(anchor);
  // The modulation of every row (the range's readers read a view of it).
  ir::TensorDesc modulated = c.tensor(m.out[0]);
  if (!between.empty()) {
    modulated.format.shape.ne[1] = sum.ne(1);
    modulated.nb = ir::NaturalStrides(modulated.format);
    for (const NodeId r : c.readers(m.out[0])) {
      const ir::Node& n = c.node(r);
      const ir::TensorDesc range = RowRange(modulated, first, rows);
      std::vector<const ir::TensorDesc*> src;
      for (const TensorId t : n.sources())
        src.push_back(!t.valid() ? nullptr : t == m.out[0] ? &range : &c.tensor(t));
      std::vector<const ir::TensorDesc*> out;
      for (const TensorId t : n.outputs()) out.push_back(&c.tensor(t));
      if (c.view().Candidates(n.key, src, out, n.params).empty()) return std::nullopt;
    }
  }
  std::vector<const ir::TensorDesc*> src;
  for (const TensorId t : match.sources) src.push_back(&c.tensor(t));
  if (c.view()
          .Candidates(ir::Fused(FusedKind::kGatedResidualModulateSplit), src,
                      std::array<const ir::TensorDesc*, 2>{&sum, &modulated}, match.params)
          .empty() ||
      !KeepsOrder(c, match))
    return std::nullopt;
  auto plan = std::make_shared<SplitResidualPlan>();
  plan->first = first;
  plan->rows = between.empty() ? 0 : rows;
  if (!Disabled(c.view().policy(), "bf16_residual") && !c.external(g.out[0])) {
    ir::TensorDesc bf16 = sum;
    bf16.format.encoding = format::Encoding::kBF16;
    bf16.nb = ir::NaturalStrides(bf16.format);
    bf16.view_of = {};
    bf16.view_offset = 0;
    bool readers = true;
    for (const NodeId r : c.readers(g.out[0])) {
      if (r == anchor || std::ranges::contains(between, r)) continue;
      const ir::Node& n = c.node(r);
      if (n.key != ir::Fused(FusedKind::kGatedResidualSplit) || n.src[0] != g.out[0] ||
          n.src[1] == g.out[0]) {
        readers = false;
        break;
      }
      std::vector<const ir::TensorDesc*> rs{&bf16};
      for (std::size_t i = 1; i < n.nsrc; ++i) rs.push_back(&c.tensor(n.src[i]));
      const std::array<const ir::TensorDesc*, 1> o{&c.tensor(n.out[0])};
      readers = !c.view().Candidates(n.key, rs, o, n.params).empty();
      if (!readers) break;
    }
    if (readers) {
      const std::array<const ir::TensorDesc*, 2> o{&bf16, &modulated};
      const bool supported =
          !c.view()
               .Candidates(ir::Fused(FusedKind::kGatedResidualModulateSplit), src, o, match.params)
               .empty();
      if (supported) {
        if (c.view().Qualification(NumericalChange::kBf16ResidualCarry, c.graph(), m) ||
            std::ranges::contains(c.view().policy().numerical_candidates,
                                  NumericalChange::kBf16ResidualCarry)) {
          plan->bf16_sum = true;
          match.numerical_change = NumericalChange::kBf16ResidualCarry;
        } else {
          match.unqualified_numerical = NumericalChange::kBf16ResidualCarry;
        }
      }
    }
  }
  match.detail = std::move(plan);
  return match;
}

// Both outputs at the modulation's place, the modulation in its encoding
// (and a range of its rows a view of them).
void SplitGatedResidualModulateRewrite(RewriteContext& c, const Match& m, std::size_t index) {
  if (index + 1 != m.nodes.size()) return;
  const ir::Graph& g = c.original();
  const ir::Node& gated = g.node(m.nodes.front());
  const ir::Node& last = g.node(m.nodes.back());
  ContextOf(c, last);
  std::vector<TensorId> sources;
  for (const TensorId s : m.sources) sources.push_back(c.Map(s));
  const auto& plan = static_cast<const SplitResidualPlan&>(*m.detail);
  const std::array encodings{plan.bf16_sum ? format::Encoding::kBF16 : format::Encoding::kF32,
                             g.tensor(last.out[0]).format.encoding};
  const ir::Outputs outs =
      c.builder().FusedBytes(FusedKind::kGatedResidualModulateSplit, sources, m.params, encodings);
  c.Set(gated.out[0], outs[0]);
  if (plan.rows == 0) {
    c.Set(last.out[0], outs[1]);
    return;
  }
  const ir::TensorDesc range = RowRange(c.builder().tensor(outs[1]), plan.first, plan.rows);
  c.Set(last.out[0], c.builder().View(outs[1], range.format.shape, range.nb, range.view_offset));
}

// bf16_branch (round 1): the product of a BF16 weight whose result only a
// split gated residual reads, as its branch, records it in BF16 from a BF16
// input (rounding there: a policy enables it, the image's).
std::optional<Match> Bf16BranchMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& p = c.node(anchor);
  if (p.key != Prim(Op::kMatMul) || c.taken(anchor)) return std::nullopt;
  const TensorId branch = p.out[0];
  const auto reader = c.OnlyReader(branch);
  if (!reader || c.node(*reader).key != ir::Fused(FusedKind::kGatedResidualSplit) ||
      c.node(*reader).src[1] != branch ||
      c.tensor(branch).format.encoding != format::Encoding::kF32)
    return std::nullopt;
  const ir::TensorDesc& w = c.tensor(p.src[0]);
  const ir::TensorDesc& x = c.tensor(p.src[1]);
  if (w.format.encoding != format::Encoding::kBF16 || w.format.layout != format::Layout::kDense ||
      w.is_view() || c.producer(p.src[0]).key != Prim(Op::kWeight) || !ir::IsContiguous(x) ||
      (x.format.encoding != format::Encoding::kF32 && x.format.encoding != format::Encoding::kBF16))
    return std::nullopt;
  if (!Bf16Round1Product(c, p)) return std::nullopt;
  const auto in_bf16 = [](ir::TensorDesc d) {
    d.format.encoding = format::Encoding::kBF16;
    d.nb = ir::NaturalStrides(d.format);
    d.view_of = {};
    d.view_offset = 0;
    return d;
  };
  const ir::TensorDesc xb = in_bf16(x), out = in_bf16(c.tensor(branch));
  const std::array<const ir::TensorDesc*, 2> src{&w, &xb};
  const std::array<const ir::TensorDesc*, 1> o{&out};
  if (c.view().Candidates(p.key, src, o, p.params).empty()) return std::nullopt;
  if (x.format.encoding != format::Encoding::kBF16) {
    const std::array<const ir::TensorDesc*, 1> xs{&x};
    const std::array<const ir::TensorDesc*, 1> xo{&xb};
    if (c.view().Candidates(Prim(Op::kConvert), xs, xo).empty()) return std::nullopt;
  }
  // The residual keeps its own match (round 1 may join it with its
  // modulation): only the product is taken.
  return Match{.nodes = {anchor}};
}

// The product again, from its input in BF16, into BF16.
void Bf16BranchRewrite(RewriteContext& c, const Match& m, std::size_t) {
  const ir::Node& p = c.original().node(m.nodes.front());
  ContextOf(c, p);
  ir::Builder& b = c.builder();
  const std::array<TensorId, 2> src{c.Map(p.src[0]), InBf16(b, c.Map(p.src[1]))};
  ir::TensorDesc out = c.original().tensor(p.out[0]);
  out.format.encoding = format::Encoding::kBF16;
  out.nb = ir::NaturalStrides(out.format);
  out.view_of = {};
  out.view_offset = 0;
  const std::array requested{out};
  c.Set(p.out[0], b.Emit(p.key, p.params, src, requested)[0]);
}

// The spatial channel norm as the spatial components emit it (components/
// image_blocks.cc SpatialNorm): x [W, H, C, N] to channel rows, L2-normed
// over C, scaled, times gamma [C], with or without SiLU, and back, each
// intermediate read by the next node only.
std::optional<Match> SpatialChannelNormMatch(const MatchContext& c, NodeId anchor) {
  const ir::Node& l2 = c.node(anchor);
  if (l2.key != Prim(Op::kL2Norm) || c.taken(anchor)) return std::nullopt;
  const auto permuted = [&](const ir::Node& n, std::array<std::uint8_t, 4> axes) {
    return n.key == Prim(Op::kPermute) && n.Params<ir::PermuteParams>().axes == axes;
  };
  // Back to the input: l2 <- cont <- permute(x, {1, 2, 0, 3}).
  const ir::Graph& g = c.graph();
  const NodeId cont_in = g.producer(l2.src[0]);
  const ir::Node& ci = c.node(cont_in);
  if (ci.key != Prim(Op::kCont) || c.taken(cont_in)) return std::nullopt;
  const NodeId permute_in = g.producer(ci.src[0]);
  const ir::Node& pi = c.node(permute_in);
  if (!permuted(pi, {1, 2, 0, 3}) || c.taken(permute_in) || c.OnlyReader(pi.out[0]) != cont_in ||
      c.OnlyReader(ci.out[0]) != anchor)
    return std::nullopt;
  const TensorId x = pi.src[0];
  const ir::TensorDesc& xd = c.tensor(x);
  if (xd.format.encoding != format::Encoding::kF32) return std::nullopt;
  // Forward: scale, mul by gamma, SiLU or not, permute(.., {2, 0, 1, 3}), cont.
  const auto scale = Next(c, l2.out[0]);
  if (!scale || c.node(*scale).key != Prim(Op::kScale) ||
      c.node(*scale).Params<ir::ScaleParams>().bias != 0.0f)
    return std::nullopt;
  const TensorId scaled = c.node(*scale).out[0];
  const auto mul = Next(c, scaled);
  if (!mul || c.node(*mul).key != Prim(Op::kMul)) return std::nullopt;
  const ir::Node& m = c.node(*mul);
  const TensorId gamma = m.src[0] == scaled ? m.src[1] : m.src[1] == scaled ? m.src[0] : TensorId{};
  // Ranks are canonical (trailing ones drop): gamma is [C] by its extents.
  if (!gamma.valid() || c.tensor(gamma).shape().ne != ir::ShapeOf({xd.ne(2)}).ne ||
      !SameShape(c.tensor(m.out[0]), c.tensor(scaled)))
    return std::nullopt;
  std::vector<NodeId> nodes{permute_in, cont_in, anchor, *scale, *mul};
  TensorId last = m.out[0];
  bool silu = false;
  auto next = Next(c, last);
  if (next && c.node(*next).key == Prim(Op::kUnary) &&
      c.node(*next).Params<ir::UnaryParams>().fn == ir::UnaryFn::kSilu) {
    silu = true;
    nodes.push_back(*next);
    last = c.node(*next).out[0];
    next = Next(c, last);
  }
  if (!next || !permuted(c.node(*next), {2, 0, 1, 3})) return std::nullopt;
  nodes.push_back(*next);
  const auto cont_out = Next(c, c.node(*next).out[0]);
  if (!cont_out || c.node(*cont_out).key != Prim(Op::kCont)) return std::nullopt;
  nodes.push_back(*cont_out);
  const ir::Node& out = c.node(*cont_out);
  if (!SameShape(c.tensor(out.out[0]), xd) ||
      c.tensor(out.out[0]).format.encoding != format::Encoding::kF32)
    return std::nullopt;
  std::ranges::sort(nodes);
  Match match =
      Fusion(std::move(nodes), {x, gamma},
             ir::SpatialNormParams{.eps = l2.Params<ir::NormParams>().eps,
                                   .scale = c.node(*scale).Params<ir::ScaleParams>().scale,
                                   .silu = silu});
  if (!Implemented(c, ir::Fused(FusedKind::kSpatialChannelNorm), match.sources, out.outputs(),
                   match.params))
    return std::nullopt;
  return match;
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
  // As today's plans take it (GGML's GemmaNormAddFusionAt and its gathered
  // form): the norm, its mul and the add one after another, or the
  // residual's gather between the mul and the add (the narrowed final
  // layer). Where two normed sums meet in one add (Gemma 4's routed and
  // dense branches), the later one's.
  // (Leaves come between nodes in the IR but in no GGML graph.)
  const auto leaves_between = [&](NodeId from, NodeId to, NodeId besides) {
    for (std::uint32_t k = from.v + 1; k < to.v; ++k) {
      const Op op = c.node(NodeId{k}).key.op;
      if (NodeId{k} != besides && op != Op::kInput && op != Op::kWeight && op != Op::kState) {
        return false;
      }
    }
    return true;
  };
  const NodeId gather = c.graph().producer(residual);
  const bool gathered = c.node(gather).key == Prim(Op::kGetRows) && gather > nm->mul;
  if (!leaves_between(nm->norm, nm->mul, NodeId{}) ||
      !leaves_between(nm->mul, *add, gathered ? gather : NodeId{})) {
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
std::optional<Match> ProductGluMatch(const MatchContext& c, NodeId anchor, bool floating) {
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
  if (wg.format != wu.format || (wg.format.encoding == format::Encoding::kF16) != floating) {
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

std::optional<Match> MulMatGluMatch(const MatchContext& c, NodeId anchor) {
  return ProductGluMatch(c, anchor, false);
}
std::optional<Match> FloatMulMatGluMatch(const MatchContext& c, NodeId anchor) {
  auto m = ProductGluMatch(c, anchor, true);
  if (!m) return std::nullopt;
  // A dense two-dimensional matrix and one contiguous input column. These
  // are a sufficient host form of upstream's MMVF stride/shape conditions;
  // lowering still checks its actual device predicate and fusion window.
  for (const auto t : m->sources)
    if (!ir::IsContiguous(c.tensor(t)) || c.tensor(t).ne(2) != 1 || c.tensor(t).ne(3) != 1)
      return std::nullopt;
  return m;
}
std::optional<Match> FloatMulMatAddMatch(const MatchContext& c, NodeId anchor) {
  const auto& p = c.node(anchor);
  if (c.taken(anchor) || p.key != Prim(Op::kMatMul) || !DefaultProduct(p) ||
      c.tensor(p.src[0]).format.encoding != format::Encoding::kF16)
    return std::nullopt;
  const auto next = Next(c, p.out[0]);
  if (!next) return std::nullopt;
  const auto& a = c.node(*next);
  if (a.key != Prim(Op::kAdd) || a.nsrc != 2 || a.src[0] != p.out[0] ||
      !SameShape(c.tensor(a.src[0]), c.tensor(a.src[1])) || c.OnlyReader(p.out[0]) != next)
    return std::nullopt;
  // Upstream's window has no computed operation between product and add.
  // Leaves may appear in IR order, but do not appear in a GGML graph.
  for (std::uint32_t i = anchor.v + 1; i < next->v; ++i) {
    const auto op = c.node(NodeId{i}).key.op;
    if (op != Op::kInput && op != Op::kWeight && op != Op::kState) return std::nullopt;
  }
  const std::array sources{p.src[0], p.src[1], a.src[1]};
  for (const auto t : sources)
    if (!ir::IsContiguous(c.tensor(t)) || c.tensor(t).ne(2) != 1 || c.tensor(t).ne(3) != 1)
      return std::nullopt;
  Match m = Fusion({anchor, *next}, {sources.begin(), sources.end()}, ir::ProductParams{});
  if (!Implemented(c, ir::Fused(FusedKind::kMatMulAdd), m.sources, a.outputs(), m.params))
    return std::nullopt;
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

// Two independent products consumed by one split GLU. The fused operation
// retains both F32 outputs; it does not fuse the GLU or change its rounding.
std::optional<Match> PairedGluProductsMatch(const MatchContext& c, NodeId anchor) {
  const auto& first = c.node(anchor);
  if (c.taken(anchor) || first.key != Prim(Op::kMatMul) || !DefaultProduct(first) ||
      c.tensor(first.out[0]).format.encoding != format::Encoding::kF32)
    return std::nullopt;
  const auto reader = c.OnlyReader(first.out[0]);
  if (!reader) return std::nullopt;
  const auto& glu = c.node(*reader);
  if (glu.key != Prim(Op::kGlu) || glu.nsrc != 2) return std::nullopt;
  const auto other = glu.src[0] == first.out[0] ? glu.src[1] : glu.src[0];
  const NodeId second_id = c.graph().producer(other);
  const auto& second = c.node(second_id);
  if (second_id <= anchor || c.taken(second_id) || second.key != Prim(Op::kMatMul) ||
      !DefaultProduct(second) || second.src[1] != first.src[1] ||
      c.OnlyReader(second.out[0]) != reader ||
      c.tensor(second.out[0]).format.encoding != format::Encoding::kF32 ||
      !SameShape(c.tensor(first.out[0]), c.tensor(second.out[0])))
    return std::nullopt;
  Match match{.nodes = {anchor, second_id}, .sources = {first.src[0], second.src[0], first.src[1]}};
  const std::array outputs{first.out[0], second.out[0]};
  if (!Implemented(c, ir::Fused(FusedKind::kPreparedProducts), match.sources, outputs,
                   match.params))
    return std::nullopt;
  return match;
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
    if (c.view().IncumbentArithmetic(c.graph(), q) != ir::ProductArithmetic::kGgmlQ8_1F32 ||
        c.view().Qualification(NumericalChange::kQ8VectorReduction, c.graph(), q) ||
        std::ranges::contains(c.view().policy().numerical_candidates,
                              NumericalChange::kQ8VectorReduction))
      continue;
    auto params = q.Params<ir::ProductParams>();
    params.arithmetic = ir::ProductArithmetic::kGgmlQ8_1F32;
    const std::array<const ir::TensorDesc*, 2> s = {&c.tensor(q.src[0]), &pd};
    const std::array<const ir::TensorDesc*, 1> o = {&c.tensor(q.out[0])};
    if (!c.view().Candidates(Prim(Op::kMatMul), s, o, ir::ToBytes(params)).empty()) {
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
    // Its head sizes and KV heads (layers differ: Gemma 4's local and
    // global ones).
    std::uint64_t kd = 0, vd = 0, kv_heads = 0;
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
  std::uint64_t heads = 0;
  ir::AttnParams attn{};
  // The masks joined one after another (owner_chained_masks), else in
  // pairs (at most four owners).
  bool chained = true;
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
    if (d.format.encoding != format::Encoding::kF16 || d.format.layout != format::Layout::kDense ||
        !d.is_view() || d.view_offset != 0 || d.ne(3) != 1 ||
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
      descs.push_back(DescOf(format::Encoding::kF32, {layer.kd, plan.rows, plan.heads, active}));
      descs.push_back(DescOf(format::Encoding::kF16, {kind.cells, kind.rows, 1, active}));
      for (const auto* caches : {&layer.k, &layer.v}) {
        for (std::uint32_t o = first; o < first + active; ++o) {
          const ir::TensorDesc& view = c.tensor(caches->at(o));
          descs.push_back(DescOf(view.format.encoding,
                                 {view.ne(0), bounded ? view.ne(2) : kind.cells, view.ne(1), 1}));
        }
      }
      const ir::TensorDesc out =
          DescOf(format::Encoding::kF32, {layer.vd, plan.heads, plan.rows, active});
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
    // attention's parameters (each layer its own head sizes).
    const ir::Node& a0 = c.node(first->attention);
    const auto attn = a0.Params<ir::AttnParams>();
    if (plan->owners == 0) {
      plan->owners = n;
      plan->rows = first->query_rows;
      plan->heads = q.ne(1);
      plan->attn = attn;
    } else if (plan->owners != n || plan->heads != q.ne(1) ||
               std::memcmp(&plan->attn, a0.params.data(), sizeof(attn)) != 0) {
      return std::nullopt;
    }
    OwnerPlan::Layer layer;
    layer.q = first->q;
    layer.out = joined;
    layer.component = a0.component;
    layer.kd = q.ne(0);
    layer.vd = c.tensor(first->v).ne(0);
    layer.kv_heads = c.tensor(first->k).ne(1);
    std::vector<TensorId> masks;
    std::vector<NodeId> group;
    for (const OwnerAttention& o : owners) {
      const ir::TensorDesc& k = c.tensor(o.k);
      const ir::TensorDesc& v = c.tensor(o.v);
      const ir::TensorDesc& mask = c.tensor(o.mask);
      if (o.query_rows != plan->rows || k.ne(0) != layer.kd || v.ne(0) != layer.vd ||
          k.ne(1) != layer.kv_heads || v.ne(1) != layer.kv_heads || k.ne(2) != mask.ne(0) ||
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
  // The policy's forms (the option patterns).
  const Policy& policy = c.view().policy();
  plan->chained = !Disabled(policy, "owner_chained_masks");
  if (!plan->chained && n > kOwnerGroup) {
    return std::nullopt;  // masks in pairs: one node's owners
  }
  plan->kind_order.resize(plan->kinds.size());
  for (std::size_t k = 0; k < plan->kinds.size(); ++k) {
    plan->kind_order[k] = k;
  }
  std::ranges::sort(plan->kind_order, {},
                    [&](std::size_t k) { return g.producer(plan->kinds[k].masks.front()); });
  // Owners of equal reads in every kind; otherwise each cache at its own
  // width (bounded roots), else every cache at its kind's: the zeros an
  // owner's cache takes (K's, and V's where its head size differs, for
  // each head size and KV heads of the kind's layers), owner by owner,
  // kinds in mask order.
  bool padded = false;
  if (std::ranges::all_of(plan->kinds, equal_reads)) {
    if (!OwnerNodesImplemented(c, *plan, true)) {
      return std::nullopt;
    }
  } else if (Disabled(policy, "owner_bounded_roots") || !OwnerNodesImplemented(c, *plan, true)) {
    if (Disabled(policy, "owner_padded_reads") || !OwnerNodesImplemented(c, *plan, false)) {
      return std::nullopt;
    }
    padded = true;
  }
  using Dims = std::pair<std::uint64_t, std::uint64_t>;  // head size, KV heads
  std::vector<std::vector<std::pair<Dims, int>>> tails(plan->kinds.size() * n);
  if (padded) {
    for (std::uint32_t o = 0; o < n; ++o) {
      for (const std::size_t k : plan->kind_order) {
        const OwnerPlan::Kind& kind = plan->kinds[k];
        const std::uint64_t read = c.tensor(kind.masks[o]).ne(0);
        if (read == kind.cells) {
          continue;
        }
        auto& list = tails[(k * n) + o];
        for (const OwnerPlan::Layer& layer : plan->layers) {
          if (layer.kind != k) {
            continue;
          }
          for (const Dims d : {Dims{layer.kd, layer.kv_heads}, Dims{layer.vd, layer.kv_heads}}) {
            if (std::ranges::contains(list, d, &std::pair<Dims, int>::first)) {
              continue;
            }
            list.emplace_back(d, static_cast<int>(plan->tails.size()));
            plan->tails.push_back(
                {.owner = o, .kind = k, .ne = {d.first, d.second, kind.cells - read}});
          }
        }
      }
    }
  }
  const auto tail = [&](std::size_t k, std::uint32_t o, Dims d) {
    const auto& list = tails[(k * n) + o];
    const auto it = std::ranges::find(list, d, &std::pair<Dims, int>::first);
    return it == list.end() ? -1 : it->second;
  };
  for (OwnerPlan::Layer& layer : plan->layers) {
    for (std::uint32_t o = 0; o < n; ++o) {
      layer.k_tail.push_back(tail(layer.kind, o, {layer.kd, layer.kv_heads}));
      layer.v_tail.push_back(tail(layer.kind, o, {layer.vd, layer.kv_heads}));
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

// Both product outputs are retained and mapped independently.
void PairedProductsRewrite(RewriteContext& c, const Match& m, std::size_t index) {
  if (index + 1 != m.nodes.size()) return;
  ContextOf(c, c.original().node(m.nodes.back()));
  std::vector<TensorId> sources;
  for (const auto source : m.sources) sources.push_back(c.Map(source));
  const std::array encodings{format::Encoding::kF32, format::Encoding::kF32};
  const auto outputs =
      c.builder().FusedBytes(FusedKind::kPreparedProducts, sources, m.params, encodings);
  for (std::size_t i = 0; i < 2; ++i) c.Set(c.original().node(m.nodes[i]).out[0], outputs[i]);
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
  auto params = product.Params<ir::ProductParams>();
  params.arithmetic = ir::ProductArithmetic::kGgmlQ8_1F32;
  c.Set(product.out[0], b.MatMul(c.Map(product.src[0]), c.local()[0], params));
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
      // Each mask padded with -inf cells to the widest; chained, each joined
      // as it is made.
      std::vector<TensorId> masks;
      TensorId joined{};
      for (const TensorId original : kind.masks) {
        TensorId mask = c.Map(original);
        const std::uint64_t width = b.tensor(mask).ne(0);
        if (width != kind.cells) {
          const TensorId tail = b.Fill(ir::ShapeOf({kind.cells - width, kind.rows, 1, 1}),
                                       -std::numeric_limits<float>::infinity(), f16);
          mask = b.Concat(mask, tail, 0);
        }
        if (plan.chained) {
          mask = b.Reshape(mask, ir::ShapeOf({kind.cells, kind.rows, 1, 1}));
          joined = joined.valid() ? b.Concat(joined, mask, 3) : mask;
        }
        masks.push_back(mask);
      }
      if (!plan.chained) {
        // In pairs, the pairs joined: (0 1) (2 3), or (0 1) 2.
        joined = b.Concat(masks[0], masks[1], 3);
        if (masks.size() == 3) joined = b.Concat(joined, masks[2], 3);
        if (masks.size() == 4) joined = b.Concat(joined, b.Concat(masks[2], masks[3], 3), 3);
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
      b.Reshape(c.Map(layer.q), ir::ShapeOf({layer.kd, plan.heads, plan.rows, n})), {0, 2, 1, 3});
  const TensorId mask = c.local().at(layer.kind);
  const bool groups = n > kOwnerGroup;
  TensorId joined{};
  for (std::uint32_t first = 0; first < n; first += kOwnerGroup) {
    const std::uint32_t active = std::min(kOwnerGroup, n - first);
    TensorId q = packed;
    TensorId masks = mask;
    if (groups) {
      const auto qn = b.tensor(packed).nb;
      q = b.View(packed, ir::ShapeOf({layer.kd, plan.rows, plan.heads, active}), qn, first * qn[3]);
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
        b.Reshape(attention, ir::ShapeOf({layer.vd * plan.heads, active * plan.rows}));
    joined = joined.valid() ? b.Concat(joined, rows, 1) : rows;
  }
  c.Set(layer.out, joined);
}

constexpr std::array kGroupedProductAnchors{Op::kPermute};
constexpr std::array kNormAnchors = {Op::kRmsNorm};
constexpr std::array kL2NormAnchors = {Op::kL2Norm};
constexpr std::array kAddAnchors = {Op::kAdd};
constexpr std::array kGdnConvAnchors = {Op::kSsmConv, Op::kCacheStore};
constexpr std::array kDeltaAnchors = {Op::kGatedDeltaNet};
constexpr std::array kMulAnchors = {Op::kMul};
constexpr std::array kGluAnchors = {Op::kGlu};
constexpr std::array kRouteAnchors = {Op::kRoute};
constexpr std::array kReduceAnchors = {Op::kExpertReduce};
constexpr std::array kFoldReaderAnchors = {Op::kPrepare, Op::kConvert, Op::kMatMul};
constexpr std::array kQuantProductAnchors = {Op::kMatMul, Op::kMatMulId, Op::kFused};
constexpr std::array kAttentionAnchors = {Op::kAttention};
constexpr std::array kFusedAnchors = {Op::kFused};
constexpr std::array kProductAnchors = {Op::kMatMul};
constexpr std::array kRopeAnchors = {Op::kRope};
constexpr std::array kPoolAnchors = {Op::kPool};
constexpr std::array kSelectedMaskAnchors = {Op::kSelectedMask};
constexpr std::array kStoreAnchors = {Op::kCacheStore};
constexpr std::array kContAnchors = {Op::kCont};
constexpr std::array kRouteReduce = {ir::Primitive(Op::kRoute), ir::Primitive(Op::kExpertReduce)};

// The table (see the header): its order is priority; where two patterns
// meet at a node (the RMS norm), the order PlanGraph tries them today.
constexpr std::array kPatterns = {
    // Closed structural fusions are available by default. Their own checks
    // and replacement capability queries preserve the numerical, state and
    // automatic-performance contracts before a graph is changed.
    Pattern{.name = "dsv4_lid_mask",
            .result = ir::Fused(FusedKind::kDsv4LidMask),
            .anchors = kSelectedMaskAnchors,
            .match = &fusions::Dsv4LidMaskMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "state_bank_attention",
            .result = ir::Fused(FusedKind::kStateBankAttention),
            .anchors = kAttentionAnchors,
            .match = &fusions::StateBankAttentionMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "dsv4_qhead",
            .result = ir::Fused(FusedKind::kDsv4QHead),
            .anchors = kRopeAnchors,
            .match = &fusions::Dsv4QHeadMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "dsv4_out_a",
            .result = ir::Fused(FusedKind::kDsv4OutA),
            .anchors = kContAnchors,
            .match = &fusions::Dsv4OutAMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "dsv4_compress",
            .result = ir::Fused(FusedKind::kDsv4Compress),
            .anchors = kPoolAnchors,
            .match = &fusions::Dsv4CompressMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "dsv4_hc_pre",
            .result = ir::Fused(FusedKind::kDsv4HcPre),
            .anchors = kMulAnchors,
            .match = &fusions::Dsv4HcPreMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "dsv4_hc_post",
            .result = ir::Fused(FusedKind::kHcNormPost),
            .anchors = kNormAnchors,
            .match = &fusions::Dsv4HcPostMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "hc_prep",
            .result = ir::Fused(FusedKind::kHcPrep),
            .anchors = kNormAnchors,
            .match = &fusions::HcPrepMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite,
            .numerical_change = NumericalChange::kBf16HyperConnectionIntermediates},
    Pattern{.name = "hc_mix",
            .result = ir::Fused(FusedKind::kHcMixNorm),
            .anchors = kNormAnchors,
            .match = &fusions::HcMixMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "hc_combine",
            .result = ir::Fused(FusedKind::kHcCombine),
            .anchors = kAddAnchors,
            .match = &fusions::HcCombineMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "gdn_conv",
            .result = ir::Fused(FusedKind::kGdnConv),
            .anchors = kGdnConvAnchors,
            .match = &fusions::GdnConvMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "gdn_gates",
            .result = ir::Fused(FusedKind::kGdnGates),
            .anchors = kAddAnchors,
            .match = &fusions::GdnGatesMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "gdn_step",
            .result = ir::Fused(FusedKind::kGdnStep),
            .anchors = kDeltaAnchors,
            .match = &fusions::GdnStepMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "gdn_norm_gate",
            .result = ir::Fused(FusedKind::kNormGate),
            .anchors = kNormAnchors,
            .match = &fusions::GdnNormGateMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "qsa_prep",
            .result = ir::Fused(FusedKind::kRmsNormMulRope),
            .anchors = kRopeAnchors,
            .match = &fusions::QsaPrepMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "qsa_pool",
            .result = ir::Fused(FusedKind::kQsaPool),
            .anchors = kStoreAnchors,
            .match = &fusions::QsaPoolMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "qsa_select",
            .result = ir::Fused(FusedKind::kQsaAttention),
            .anchors = kAttentionAnchors,
            .match = &fusions::QsaSelectMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "qsa_gate",
            .result = ir::Fused(FusedKind::kHeadGate),
            .anchors = kMulAnchors,
            .match = &fusions::QsaGateMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "moe_router",
            .result = ir::Fused(FusedKind::kMoeRouter),
            .anchors = kRouteAnchors,
            .match = &fusions::MoeRouterMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "moe_experts",
            .result = ir::Fused(FusedKind::kMoeExperts),
            .anchors = kReduceAnchors,
            .match = &fusions::MoeExpertsMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite,
            .numerical_change = NumericalChange::kNvfp4ExpertIntermediates},
    Pattern{.name = "moe_glu",
            .result = ir::Fused(FusedKind::kMoeGlu),
            .anchors = kGluAnchors,
            .match = &fusions::MoeGluMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "moe_combine",
            .result = ir::Fused(FusedKind::kMoeCombine),
            .anchors = kReduceAnchors,
            .match = &fusions::MoeCombineMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "vecq_glu",
            .version = 2,
            .result = ir::Fused(FusedKind::kMatMulGlu),
            .anchors = kGluAnchors,
            .match = &fusions::VecQGluMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite,
            .numerical_change = NumericalChange::kQ8VectorReduction},
    Pattern{.name = "mxfp8_inputs",
            .result = ir::Primitive(Op::kPrepare),
            .anchors = kProductAnchors,
            .match = &fusions::Mxfp8InputsMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite,
            .numerical_change = NumericalChange::kMxfp8ProductInput},
    // Round 1: over round 0's fused nodes.
    Pattern{.name = "hc_fold_inputs",
            .result = ir::Fused(FusedKind::kHcFold),
            .anchors = kFoldReaderAnchors,
            .match = &fusions::HcFoldInputsMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite,
            .round = 1},
    Pattern{.name = "qk_attention",
            .version = 2,
            .result = ir::Fused(FusedKind::kQkAttention),
            .anchors = kAttentionAnchors,
            .match = &QkAttentionMatch,
            .own_checks = true,
            .rewrite = &QkAttentionRewrite,
            .numerical_change = NumericalChange::kBf16AttentionIntermediates,
            .round = 1},
    Pattern{.name = "split_gated_residual_modulate",
            .version = 2,
            .result = ir::Fused(FusedKind::kGatedResidualModulateSplit),
            .anchors = kFusedAnchors,
            .match = &SplitGatedResidualModulateMatch,
            .own_checks = true,
            .rewrite = &SplitGatedResidualModulateRewrite,
            .round = 1},
    Pattern{.name = "bf16_branch",
            .version = 2,
            .result = ir::Primitive(Op::kMatMul),
            .anchors = kProductAnchors,
            .match = &Bf16BranchMatch,
            .own_checks = true,
            .rewrite = &Bf16BranchRewrite,
            .numerical_change = NumericalChange::kBf16ProductInputAndResult,
            .round = 1},
    Pattern{.name = "bf16_inputs",
            .result = ir::Primitive(Op::kConvert),
            .anchors = kProductAnchors,
            .match = &fusions::Bf16InputsMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite,
            .round = 1},
    Pattern{.name = "bf16_rows",
            .result = ir::Primitive(Op::kMatMul),
            .anchors = kProductAnchors,
            .match = &fusions::Bf16RowsMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite,
            .numerical_change = NumericalChange::kBf16ProductInputAndResult,
            .round = 1},
    Pattern{.name = "grouped_vecq",
            .result = ir::Fused(FusedKind::kGroupedMatMul),
            .anchors = kGroupedProductAnchors,
            .match = &fusions::GroupedVecQMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite},
    Pattern{.name = "vecq_inputs",
            .version = 2,
            .result = ir::Primitive(Op::kPrepare),
            .anchors = kQuantProductAnchors,
            .match = &fusions::VecQInputsMatch,
            .own_checks = true,
            .rewrite = &fusions::Rewrite,
            .round = 1},
    Pattern{.name = "spatial_channel_norm",
            .result = ir::Fused(FusedKind::kSpatialChannelNorm),
            .anchors = kL2NormAnchors,
            .match = &SpatialChannelNormMatch,
            .rewrite = &Fused<FusedKind::kSpatialChannelNorm>},
    Pattern{.name = "rms_norm_mul_rope",
            .result = ir::Fused(FusedKind::kRmsNormMulRope),
            .anchors = kNormAnchors,
            .match = &NormMulRopeMatch,
            .rewrite = &Fused<FusedKind::kRmsNormMulRope>},
    Pattern{.name = "rms_norm_mul_rope_sections",
            .result = ir::Fused(FusedKind::kRmsNormMulRopeSections),
            .anchors = kNormAnchors,
            .match = &NormRopeSectionsMatch,
            .own_checks = true,
            .rewrite = &Fused<FusedKind::kRmsNormMulRopeSections>},
    Pattern{.name = "rms_norm_mul_add",
            .result = ir::Fused(FusedKind::kRmsNormMulAdd),
            .anchors = kNormAnchors,
            .match = &NormMulAddMatch,
            .rewrite = &Fused<FusedKind::kRmsNormMulAdd>},
    // Before bf16_glu, which reads a BF16 modulation as it is.
    Pattern{.name = "split_modulate",
            .result = ir::Fused(FusedKind::kLayerNormModulateSplit),
            .anchors = kMulAnchors,
            .match = &SplitModulateMatch,
            .own_checks = true,
            .rewrite = &SplitModulateRewrite},
    Pattern{.name = "split_gated_residual",
            .result = ir::Fused(FusedKind::kGatedResidualSplit),
            .anchors = kAddAnchors,
            .match = &SplitGatedResidualMatch,
            .own_checks = true,
            .rewrite = &Fused<FusedKind::kGatedResidualSplit>},
    Pattern{.name = "bf16_glu",
            .result = ir::Primitive(Op::kGlu),
            .anchors = kGluAnchors,
            .match = &Bf16GluMatch,
            .own_checks = true,
            .rewrite = &Bf16GluRewrite,
            .numerical_change = NumericalChange::kBf16SwiGluIntermediates},
    Pattern{.name = "float_mul_mat_glu",
            .result = ir::Fused(FusedKind::kMatMulGlu),
            .anchors = kProductAnchors,
            .match = &FloatMulMatGluMatch,
            .rewrite = &Fused<FusedKind::kMatMulGlu>},
    Pattern{.name = "float_mul_mat_add",
            .result = ir::Fused(FusedKind::kMatMulAdd),
            .anchors = kProductAnchors,
            .match = &FloatMulMatAddMatch,
            .rewrite = &Fused<FusedKind::kMatMulAdd>},
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
    Pattern{.name = "paired_glu_products",
            .result = ir::Fused(FusedKind::kPreparedProducts),
            .anchors = kProductAnchors,
            .match = &PairedGluProductsMatch,
            .rewrite = &PairedProductsRewrite},
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
    Pattern{.name = "biased_route", .kind = PatternKind::kOption},
    Pattern{.name = "hash_route", .kind = PatternKind::kOption},
    Pattern{.name = "route_reduce", .kind = PatternKind::kComposite, .composites = kRouteReduce},
    Pattern{
        .name = "owner_bounded_roots", .kind = PatternKind::kOption, .parent = "owner_attention"},
    Pattern{
        .name = "owner_padded_reads", .kind = PatternKind::kOption, .parent = "owner_attention"},
    Pattern{
        .name = "owner_chained_masks", .kind = PatternKind::kOption, .parent = "owner_attention"},
    Pattern{.name = "narrow_final", .kind = PatternKind::kOption},
    Pattern{.name = "rounded_indexer", .kind = PatternKind::kOption, .opt_in = true},
    // A step of several one-row owners retains scalar product arithmetic
    // (each owner's row what it is alone: today's Qwen3.8 waves;
    // StepPlan::owner_row_invariant). Opt-in.
    Pattern{.name = "owner_row_invariant", .kind = PatternKind::kOption, .opt_in = true},
    // The step's builder removes its layout round trips (ir/layout.h,
    // components/spatial.h sink_layouts), subject to layout capability checks.
    Pattern{.name = "layout_sinking", .kind = PatternKind::kOption},
    // split_gated_residual_modulate records a residual's sum in BF16 where
    // only later residuals read it and its numerical change is qualified.
    Pattern{.name = "bf16_residual", .kind = PatternKind::kOption},
};

bool Anchored(const Pattern& p, Op op) { return std::ranges::contains(p.anchors, op); }

}  // namespace

// ------------------------------------------------------------ the context

MatchContext::MatchContext(const ir::Graph& graph, const SelectionView& view,
                           const StepShape* shape, StateSpans state_spans)
    : graph_(graph),
      view_(view),
      shape_(shape),
      state_spans_(std::move(state_spans)),
      readers_(graph.tensor_count()),
      external_(graph.tensor_count(), false),
      kept_(graph.tensor_count(), false),
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
      kept_.at(t.v) = true;
    }
  }
}

std::span<const NodeId> MatchContext::readers(TensorId t) const { return readers_.at(t.v); }

bool MatchContext::external(TensorId t) const {
  return external_.at(t.v) || external_.at(graph_.root(t).v);
}

bool MatchContext::kept(TensorId t) const { return kept_.at(t.v); }

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
  if (node.key == ir::Primitive(Op::kView) && !sources.empty() && sources[0].valid() &&
      !ir::SameViewSource(original_.tensor(node.src[0]), builder_.tensor(sources[0]))) {
    builder_.Refuse(ir::GraphError::kFormat,
                    "a view's source changed layout in a rewrite: the view would read other "
                    "elements");
    return {};
  }
  ir::ParamBytes params = node.params;
  if (view_ && node.key == Prim(Op::kMatMul) && node.nsrc == 2 && sources[1].valid() &&
      original_.tensor(node.src[1]).format.encoding == format::Encoding::kF32 &&
      builder_.tensor(sources[1]).format.encoding == format::Encoding::kBF16 &&
      view_->IncumbentArithmetic(original_, node) == ir::ProductArithmetic::kBf16F32) {
    auto product = node.Params<ir::ProductParams>();
    product.arithmetic = ir::ProductArithmetic::kBf16F32;
    params = ir::ToBytes(product);
  }
  const ir::Outputs outs =
      builder_.Emit(node.key, params, sources, requested,
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

bool CompositeImplemented(const Policy& policy, ir::OpKey key) {
  return std::ranges::none_of(kPatterns, [&](const Pattern& p) {
    return p.kind == PatternKind::kComposite && std::ranges::contains(p.composites, key) &&
           Disabled(policy, p.name);
  });
}

base::Sha256Digest PatternsDigest(const Policy& policy) {
  base::Sha256 hash;
  hash.Update(std::string_view("llmp.planner.patterns.v3;qualified-automatic-recipes\n"));
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

namespace {

// One round of the table's rewrite patterns over `graph`.
std::expected<Rewritten, ir::VerifyError> ApplyRound(
    const ir::Graph& graph, const SelectionView& view, const StepShape* shape,
    const StateSpans& state_spans, std::uint8_t round, std::span<const NodeId> preserve) {
  MatchContext context(graph, view, shape, state_spans);
  struct Found {
    const Pattern* pattern = nullptr;
    Match match;
    std::vector<NumericalProof> numerical{};
  };
  std::vector<Found> found;
  std::vector<MissingNumericalQualification> missing_numerical;
  const auto missing = [&](NumericalChange change, const ir::Node& node) {
    const MissingNumericalQualification record{
        .change = change, .component = node.component, .anchor = node.key};
    if (!std::ranges::contains(missing_numerical, record)) missing_numerical.push_back(record);
  };
  // The table's order is priority: each pattern over the whole graph, in
  // node order, before the next.
  for (const Pattern& p : kPatterns) {
    if (p.kind != PatternKind::kRewrite || p.round != round || Disabled(view.policy(), p.name)) {
      continue;
    }
    for (std::uint32_t i = 0; i < graph.nodes().size(); ++i) {
      const NodeId id{i};
      const ir::Node& node = graph.node(id);
      // Round 0 anchors at primitives; a later round at another's results
      // too.
      if (context.taken(id) || (round == 0 && node.key.fused != FusedKind::kNone) ||
          !Anchored(p, node.key.op)) {
        continue;
      }
      auto m = p.match(context, id);
      if (!m ||
          std::ranges::any_of(m->nodes,
                              [&](NodeId n) { return std::ranges::contains(preserve, n); }) ||
          (!p.own_checks &&
           ((p.result.op == Op::kFused && !ElidesF32(context, *m)) || !KeepsOrder(context, *m)))) {
        continue;
      }
      const auto change = m->numerical_change ? m->numerical_change : p.numerical_change;
      std::vector<NumericalProof> numerical;
      if (change) {
        for (const auto site : m->unqualified_sites) missing(*change, graph.node(site));
        const std::array anchor_site{id};
        const std::span<const NodeId> sites = m->numerical_sites.empty()
                                                  ? std::span<const NodeId>(anchor_site)
                                                  : std::span<const NodeId>(m->numerical_sites);
        bool authorized = true;
        for (const auto site : sites) {
          const auto& recipient = graph.node(site);
          for (std::uint8_t operand = 0; operand < SelectionView::NumericalOperands(recipient);
               ++operand) {
            const auto grant = view.Qualification(*change, graph, recipient, operand);
            if (!grant) {
              missing(*change, recipient);
              if (!std::ranges::contains(view.policy().numerical_candidates, *change)) {
                authorized = false;
                continue;
              }
            }
            std::array<const ir::TensorDesc*, ir::kMaxSources> sources{};
            std::array<const ir::TensorDesc*, ir::kMaxOutputs> outputs{};
            for (std::size_t j = 0; j < recipient.nsrc; ++j)
              if (recipient.src[j].valid()) sources[j] = &graph.tensor(recipient.src[j]);
            for (std::size_t j = 0; j < recipient.nout; ++j)
              outputs[j] = &graph.tensor(recipient.out[j]);
            auto actual = ShapeOfNode(recipient.key, std::span(sources).first(recipient.nsrc),
                                      std::span(outputs).first(recipient.nout));
            if (SelectionView::NumericalOperands(recipient) == 2) {
              const auto& weight = graph.tensor(recipient.source(operand));
              actual.k = weight.ne(0);
              actual.n = weight.ne(1);
            }
            NumericalProof proof{.site = site,
                                 .weight_operand = operand,
                                 .actual = actual,
                                 .candidate = !grant.has_value()};
            const bool product = recipient.key == ir::Primitive(Op::kMatMul) ||
                                 recipient.key == ir::Primitive(Op::kMatMulId) ||
                                 SelectionView::NumericalOperands(recipient) == 2;
            proof.observed.active_stage = view.policy().active_stage;
            if (product) {
              if (const auto weight =
                      SelectionView::ResolveWeightOperand(graph, recipient, operand)) {
                proof.observed.weight = weight->reference;
                const auto& leaf = graph.tensor(weight->leaf);
                const auto& actual_weight = graph.tensor(recipient.source(operand));
                const std::size_t input_at =
                    SelectionView::NumericalOperands(recipient) == 2 ? 2U : 1U;
                const auto& input = graph.tensor(recipient.source(input_at));
                proof.observed.operand_format = actual_weight.format;
                proof.observed.operand_strides = actual_weight.nb;
                proof.observed.leaf_format = leaf.format;
                proof.observed.leaf_strides = leaf.nb;
                proof.observed.input_format = input.format;
                proof.observed.input_strides = input.nb;
              }
            }
            if (grant) {
              proof.qualification = *grant;
              proof.grouped_projection = recipient.key == ir::Primitive(Op::kMatMul) &&
                                         operand == 0 &&
                                         fusions::GroupedProductOf(graph, site).has_value();
            } else {
              proof.qualification.change = *change;
              proof.qualification.stage = view.policy().active_stage;
              proof.qualification.component = recipient.component;
              proof.qualification.anchor =
                  SelectionView::NumericalOperands(recipient) == 2
                      ? ir::Primitive(recipient.key.fused == FusedKind::kMatMulGlu ? Op::kMatMul
                                                                                   : Op::kMatMulId)
                      : recipient.key;
            }
            numerical.push_back(std::move(proof));
          }
        }
        if (!authorized) continue;
      }
      if (m->unqualified_numerical) missing(*m->unqualified_numerical, node);
      context.Take(m->nodes);
      found.push_back({.pattern = &p, .match = std::move(*m), .numerical = std::move(numerical)});
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
  RewriteContext rc(graph, b, tensors, &view);
  for (std::size_t f = 0; f < found.size(); ++f) {
    applied[f].pattern = found[f].pattern->name;
    applied[f].replaced = found[f].match.nodes;
    applied[f].numerical_proofs = std::move(found[f].numerical);
    if (!applied[f].numerical_proofs.empty()) {
      const auto& first = applied[f].numerical_proofs.front().qualification;
      applied[f].numerical_change = first.change;
      applied[f].numerical_evidence = first.evidence;
      applied[f].numerical_source = first.source;
      applied[f].numerical_semantic = first.semantic;
      applied[f].numerical_candidate =
          std::ranges::any_of(applied[f].numerical_proofs, &NumericalProof::candidate);
      applied[f].numerical_incumbent =
          std::ranges::all_of(applied[f].numerical_proofs, [](const auto& proof) {
            return !proof.candidate && proof.qualification.incumbent_recipe;
          });
    }
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
                   .tensors = std::move(tensors),
                   .missing_numerical = std::move(missing_numerical)};
}

}  // namespace

std::expected<Rewritten, ir::VerifyError> ApplyPatterns(const ir::Graph& graph,
                                                        const SelectionView& view,
                                                        const StepShape* shape,
                                                        StateSpans state_spans,
                                                        std::span<const NodeId> preserve) {
  auto first = ApplyRound(graph, view, shape, state_spans, 0, preserve);
  if (!first) {
    return first;
  }
  const bool later = std::ranges::any_of(kPatterns, [&](const Pattern& p) {
    return p.kind == PatternKind::kRewrite && p.round == 1 && !Disabled(view.policy(), p.name);
  });
  if (!later) {
    return first;
  }
  SelectionView next_view = view;
  next_view.Forward(graph, first->graph, first->tensors);
  std::vector<NodeId> next_preserve;
  for (std::uint32_t i = 0; i < first->from.size(); ++i) {
    if (std::ranges::contains(preserve, first->from[i])) next_preserve.push_back(NodeId{i});
  }
  auto second = ApplyRound(first->graph, next_view, shape, state_spans, 1, next_preserve);
  if (!second) {
    return second;
  }
  // The maps composed: the primitive graph's tensors and nodes through
  // round 0's graph into round 1's; round 0's nodes it made, as round 1
  // copied them.
  Rewritten out{.graph = std::move(second->graph),
                .applied = {},
                .from = {},
                .tensors = {},
                .missing_numerical = std::move(first->missing_numerical)};
  for (const auto& missing : second->missing_numerical)
    if (!std::ranges::contains(out.missing_numerical, missing))
      out.missing_numerical.push_back(missing);
  out.tensors.assign(first->tensors.size(), TensorId{});
  for (std::size_t t = 0; t < first->tensors.size(); ++t) {
    const TensorId mid = first->tensors[t];
    out.tensors[t] = mid.valid() ? second->tensors.at(mid.v) : TensorId{};
  }
  out.from.resize(second->from.size());
  for (std::size_t i = 0; i < second->from.size(); ++i) {
    out.from[i] = first->from.at(second->from[i].v);
  }
  // A round-0 node round 1 did not rewrite is one node of round 1's graph.
  std::vector<bool> rewritten(first->graph.nodes().size(), false);
  for (const Applied& a : second->applied) {
    for (const NodeId r : a.replaced) rewritten.at(r.v) = true;
  }
  for (Applied& a : first->applied) {
    std::vector<NodeId> made;
    for (std::uint32_t j = 0; j < second->from.size(); ++j) {
      const NodeId mid = second->from[j];
      if (!rewritten.at(mid.v) && std::ranges::contains(a.made, mid)) made.push_back(NodeId{j});
    }
    a.made = std::move(made);
    out.applied.push_back(std::move(a));
  }
  // Round 1's: its replaced nodes as the primitive nodes they stood for.
  for (Applied& a : second->applied) {
    for (NodeId& r : a.replaced) r = first->from.at(r.v);
    for (auto& proof : a.numerical_proofs) proof.site = first->from.at(proof.site.v);
    out.applied.push_back(std::move(a));
  }
  return out;
}

}  // namespace llmp::planner
