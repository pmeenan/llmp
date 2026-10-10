// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include "ir/decompose.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "format/encoding.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/infer.h"
#include "ir/op.h"
#include "ir/verify.h"

namespace llmp::ir {
namespace {

using format::Shape;

Shape S(std::uint64_t a, std::uint64_t b = 1, std::uint64_t c = 1, std::uint64_t d = 1) {
  return ShapeOf({a, b, c, d});
}

using Nb = std::array<std::uint64_t, format::kMaxDims>;

template <typename P>
P Get(const ParamBytes& bytes) {
  P p{};
  std::memcpy(&p, bytes.data(), sizeof(P));
  return p;
}

// The context a decomposition runs in: the builder and the sources.
class D {
 public:
  D(Builder& b, std::span<const TensorId> src) : b_(b), src_(src) {}
  Builder& b() { return b_; }
  std::size_t count() const { return src_.size(); }
  TensorId s(std::size_t i) const { return i < src_.size() ? src_[i] : TensorId{}; }
  // A copy: the builder's tensors move as it grows.
  TensorDesc t(TensorId id) const { return b_.tensor(id); }
  bool need(std::size_t required, std::size_t total) {
    if (!b_.ok()) {
      return false;
    }
    if (src_.size() < required || src_.size() > total) {
      b_.Refuse(GraphError::kArity,
                std::format("takes {} to {} sources, not {}", required, total, src_.size()));
      return false;
    }
    for (std::size_t i = 0; i < required; ++i) {
      if (!src_[i].valid()) {
        b_.Refuse(GraphError::kArity, std::format("source {} is required", i));
        return false;
      }
    }
    return true;
  }
  bool check(bool ok, GraphError e, const std::string& what) {
    if (!ok && b_.ok()) {
      b_.Refuse(e, what);
    }
    return ok && b_.ok();
  }
  // A 2-D [n, m] view of `x` with strides nb0 and nbm at `offset` bytes.
  TensorId View2(TensorId x, std::uint64_t n, std::uint64_t m, std::uint64_t nb0, std::uint64_t nbm,
                 std::uint64_t offset) {
    return b_.View(x, S(n, m), Nb{nb0, nbm, nbm * m, nbm * m}, offset);
  }

 private:
  Builder& b_;
  std::span<const TensorId> src_;
};

// scale[ids] as [1, U, T]: each slot's value of a per-expert vector.
TensorId ScaleRows(D& d, TensorId scale, TensorId ids) {
  Builder& b = d.b();
  if (!b.ok()) {
    return {};
  }
  const TensorDesc& sc = d.t(scale);
  const TensorDesc& id = d.t(ids);
  const std::uint64_t e = sc.ne(0);
  const std::uint64_t u = id.ne(0);
  const std::uint64_t t = id.ne(1);
  const TensorId table =
      b.View(scale, S(1, e), Nb{sc.nb[0], sc.nb[0], sc.nb[0] * e, sc.nb[0] * e}, 0);
  const TensorId flat = b.Reshape(b.Cont(ids), S(u * t));
  const TensorId rows = b.GetRows(table, flat);  // [1, U · T]
  return b.Reshape(rows, S(1, u, t));
}

// One stream (or slot) of a [w, n, t] tensor as [w, t].
TensorId Slice(D& d, TensorId x, std::uint64_t i) {
  const TensorDesc& xt = d.t(x);
  return d.b().View(x, S(xt.ne(0), xt.ne(2)),
                    Nb{xt.nb[0], xt.nb[2], xt.nb[2] * xt.ne(2), xt.nb[2] * xt.ne(2)}, i * xt.nb[1]);
}

// Element `at` (a byte offset) of a vector, as [1].
TensorId Element(D& d, TensorId v, std::uint64_t index) {
  const TensorDesc& vt = d.t(v);
  const std::uint64_t e = vt.nb[0];
  return d.b().View(v, S(1), Nb{e, e, e, e}, index * e);
}

// Elements [first, first + n) of a vector, as [n].
TensorId Range(D& d, TensorId v, std::uint64_t first, std::uint64_t n) {
  const TensorDesc& vt = d.t(v);
  const std::uint64_t e = vt.nb[0];
  return d.b().View(v, S(n), Nb{e, e * n, e * n, e * n}, first * e);
}

// ---------------------------------------------------------------- composites

Outputs One(TensorId id) {
  Outputs o;
  o.ids[0] = id;
  o.n = id.valid() ? 1 : 0;
  return o;
}

Outputs HcPre(D& d, const HcParams& p) {
  if (!d.need(2, 2)) {
    return {};
  }
  Builder& b = d.b();
  TensorId x = d.s(0);
  TensorId acc{};
  if (p.fold == HcFold::kWeights) {
    const TensorDesc& w = d.t(d.s(1));
    for (std::uint64_t h = 0; h < p.streams && b.ok(); ++h) {
      const TensorId xh = Slice(d, x, h);
      const TensorId wh = d.View2(d.s(1), 1, w.ne(1), w.nb[0], w.nb[1], h * w.nb[0]);
      const TensorId term = b.Mul(xh, wh);
      acc = acc.valid() ? b.Add(acc, term) : term;
    }
    return One(acc);
  }
  const TensorId m = b.Mul(x, b.Unary(d.s(1), UnaryFn::kSigmoid));
  for (std::uint64_t h = 0; h < p.streams && b.ok(); ++h) {
    const TensorId term = Slice(d, m, h);
    acc = acc.valid() ? b.Add(acc, term) : b.Cont(term);
  }
  return One(b.Scale(acc, p.scale));
}

Outputs HcPost(D& d, const HcParams& p) {
  if (!d.need(3, 4)) {
    return {};
  }
  Builder& b = d.b();
  const TensorId y = d.s(0);
  const TensorId x = d.s(1);
  const TensorId post = d.s(2);
  const TensorId comb = d.s(3);
  const std::uint64_t hc = p.streams;
  const std::uint64_t w = d.t(y).ne(0);
  const std::uint64_t t = d.t(y).ne(1);
  TensorId out{};
  for (std::uint64_t dst = 0; dst < hc && b.ok(); ++dst) {
    const TensorDesc& pt = d.t(post);
    TensorId acc = b.Mul(y, d.View2(post, 1, t, pt.nb[0], pt.nb[1], dst * pt.nb[0]));
    if (comb.valid()) {
      const TensorDesc& ct = d.t(comb);
      for (std::uint64_t s = 0; s < hc && b.ok(); ++s) {
        const TensorId c =
            d.View2(comb, 1, t, ct.nb[0], ct.nb[2], (dst * ct.nb[0]) + (s * ct.nb[1]));
        acc = b.Add(acc, b.Mul(Slice(d, x, s), c));
      }
    } else {
      acc = b.Add(acc, Slice(d, x, dst));
    }
    const TensorId stream = b.Reshape(acc, S(w, 1, t));
    out = out.valid() ? b.Concat(out, stream, 1) : stream;
  }
  return One(out);
}

Outputs HcMix(D& d, const HcParams& p) {
  if (!d.need(3, 3)) {
    return {};
  }
  Builder& b = d.b();
  const TensorId mixes = d.s(0);
  const TensorId scale = d.s(1);
  const TensorId base = d.s(2);
  const std::uint64_t hc = p.streams;
  const TensorDesc& mt = d.t(mixes);
  const std::uint64_t t = mt.ne(1);
  if (p.mix == HcMixKind::kSigmoid) {
    const auto weights = [&](std::uint64_t at, std::uint64_t which) {
      TensorId v = d.View2(mixes, hc, t, mt.nb[0], mt.nb[1], at * mt.nb[0]);
      v = b.Add(b.Mul(v, Element(d, scale, which)), Range(d, base, at, hc));
      return b.Unary(v, UnaryFn::kSigmoid);
    };
    const TensorId pre = b.Scale(weights(0, 0), 1.0f, p.eps);
    const TensorId post = b.Scale(weights(hc, 1), 2.0f, 0.0f);
    return One(b.Concat(b.Reshape(pre, S(hc, 1, t)), b.Reshape(post, S(hc, 1, t)), 1));
  }
  // Sinkhorn: [dst, src, t] from the last hc² mixes.
  const std::uint64_t e = mt.nb[0];
  TensorId m = b.View(mixes, S(hc, hc, t), Nb{e, e * hc, mt.nb[1], mt.nb[1] * t}, 2 * hc * e);
  m = b.Mul(m, Element(d, scale, 2));
  const TensorDesc& bt = d.t(base);
  m = b.Add(m, b.View(base, S(hc, hc),
                      Nb{bt.nb[0], bt.nb[0] * hc, bt.nb[0] * hc * hc, bt.nb[0] * hc * hc},
                      2 * hc * bt.nb[0]));
  m = b.SoftMax(m, {}, {}, SoftMaxParams{});
  m = b.Scale(m, 1.0f, p.eps);
  const auto cols = [&] {
    TensorId s = b.SumRows(b.Cont(b.Permute(m, {1, 0, 2, 3})));  // [1, dst, t]
    s = b.Reshape(b.Scale(s, 1.0f, p.eps), S(hc, 1, t));
    m = b.Div(m, s);
  };
  const auto rows = [&] {
    m = b.Div(m, b.Scale(b.SumRows(m), 1.0f, p.eps));  // [1, src, t]
  };
  cols();
  for (std::uint32_t i = 1; i < p.iterations && b.ok(); ++i) {
    rows();
    cols();
  }
  return One(m);
}

Outputs IndexScore(D& d) {
  if (!d.need(3, 4)) {
    return {};
  }
  Builder& b = d.b();
  const TensorDesc& q = d.t(d.s(0));
  const TensorDesc& k = d.t(d.s(1));
  const TensorDesc& w = d.t(d.s(2));
  const std::uint64_t c = k.ne(2);
  const TensorId kv =
      b.View(d.s(1), S(k.ne(0), c, 1, k.ne(3)), Nb{k.nb[0], k.nb[2], k.nb[2] * c, k.nb[3]}, 0);
  TensorId r = b.Unary(b.MatMul(kv, d.s(0)), UnaryFn::kRelu);  // [C, H, T, S]
  const TensorId ww =
      b.View(d.s(2), S(1, w.ne(0), w.ne(1), w.ne(3)), Nb{w.nb[0], w.nb[0], w.nb[1], w.nb[3]}, 0);
  r = b.Mul(r, ww);
  TensorId sum = b.SumRows(b.Cont(b.Permute(r, {1, 0, 2, 3})));  // [1, C, T, S]
  sum = b.Reshape(sum, S(c, q.ne(2), 1, q.ne(3)));
  if (d.s(3).valid()) {
    sum = b.Add(sum, d.s(3));
  }
  return One(sum);
}

Outputs Pool(D& d, const PoolParams& p) {
  if (!d.need(1, 2)) {
    return {};
  }
  Builder& b = d.b();
  const TensorDesc& x = d.t(d.s(0));
  const std::uint64_t c = x.ne(0);
  const std::uint64_t r = p.ratio;
  const std::uint64_t n = x.ne(1) / r;
  const std::uint64_t a2 = x.ne(2);
  // Runs of matrix i3 as [R, C, N, A2]: a run's rows innermost.
  const auto runs = [&](TensorId v, std::uint64_t i3) {
    const TensorDesc& vt = d.t(v);
    const TensorId view = b.View(v, S(vt.ne(0), r, n, vt.ne(2)),
                                 Nb{vt.nb[0], vt.nb[1], vt.nb[1] * r, vt.nb[2]}, i3 * vt.nb[3]);
    return b.Cont(b.Permute(view, {1, 0, 2, 3}));
  };
  // Each of the A3 matrices pooled on its own (a view has four dims), then
  // joined along dim 3.
  TensorId out{};
  for (std::uint64_t i3 = 0; i3 < x.ne(3) && b.ok(); ++i3) {
    TensorId vals = runs(d.s(0), i3);
    if (p.kind == PoolKind::kSoftmax) {
      vals = b.Mul(vals, b.SoftMax(runs(d.s(1), i3), {}, {}, SoftMaxParams{}));
    }
    TensorId sum = b.Reshape(b.SumRows(vals), S(c, n, a2));
    if (p.kind == PoolKind::kMean) {
      sum = b.Scale(sum, 1.0f / static_cast<float>(r));
    }
    out = out.valid() ? b.Concat(out, sum, 3) : sum;
  }
  return One(out);
}

Outputs Route(D& d, const RouteParams& p) {
  if (!d.need(1, 4)) {
    return {};
  }
  Builder& b = d.b();
  const TensorId logits = d.s(0);
  const std::uint64_t e = p.experts;
  const std::uint64_t u = p.used;
  const std::uint64_t t = d.t(logits).ne(1);
  TensorId probs{};
  switch (p.kind) {
    case RouteKind::kSoftmaxTopK:
      probs = b.SoftMax(logits, {}, {}, SoftMaxParams{});
      break;
    case RouteKind::kSigmoidTopK:
      probs = b.Unary(logits, UnaryFn::kSigmoid);
      break;
    case RouteKind::kSqrtSoftplusBias:
    case RouteKind::kHash:
      probs = b.Unary(b.Unary(logits, UnaryFn::kSoftplus), UnaryFn::kSqrt);
      break;
  }
  // llama.cpp's build_moe_ffn order: the probabilities' rows, the
  // selection, the selected rows.
  const TensorId rows = b.Reshape(probs, S(1, e, t));
  TensorId ids{};
  if (p.kind == RouteKind::kHash) {
    // [U, T]: the table read by the tokens, or without them each row's ids
    // looked up already (a token-hash side input's).
    ids = d.s(3).valid() ? b.GetRows(d.s(2), d.s(3)) : d.s(2);
  } else {
    const TensorId selection = d.s(1).valid() ? b.Add(probs, d.s(1)) : probs;
    const TensorId order = b.Argsort(selection, true);
    const TensorDesc& ot = d.t(order);
    ids = b.View(order, S(u, t), Nb{ot.nb[0], ot.nb[1], ot.nb[1] * t, ot.nb[1] * t}, 0);
  }
  TensorId w = b.Reshape(b.GetRows(rows, ids), S(u, t));
  if (p.normalize) {
    const TensorId sum = b.Clamp(b.SumRows(w), 0x1p-14f, std::numeric_limits<float>::infinity());
    w = b.Div(w, sum);
  }
  if (p.scale != 1.0f) {
    w = b.Scale(w, p.scale);
  }
  Outputs o;
  o.ids[0] = ids;
  o.ids[1] = w;
  o.n = 2;
  return o;
}

// llama.cpp's build_moe_ffn reduce: x times each slot's scale (a per-expert
// scale's rows gathered, or one gathered already), times its weight, then
// the slots' views (all of them first) added in slot order onto the first.
TensorId Reduce(D& d, TensorId x, TensorId w, TensorId ids, TensorId scale) {
  Builder& b = d.b();
  const TensorDesc& xt = d.t(x);
  if (scale.valid()) {
    const TensorDesc& st = d.t(scale);
    const bool gathered =
        st.ne(0) == 1 && st.ne(1) == xt.ne(1) && st.ne(2) == xt.ne(2) && st.ne(3) == 1;
    x = b.Mul(x, gathered ? scale : ScaleRows(d, scale, ids));
  }
  const TensorDesc& wt = d.t(w);
  const TensorId w3 =
      b.View(w, S(1, wt.ne(0), wt.ne(1)), Nb{wt.nb[0], wt.nb[0], wt.nb[1], wt.nb[1] * wt.ne(1)}, 0);
  const TensorId xw = b.Mul(x, w3);  // [n, U, T]
  const std::uint64_t u = wt.ne(0);
  std::vector<TensorId> slots;
  for (std::uint64_t i = 0; i < u && b.ok(); ++i) slots.push_back(Slice(d, xw, i));
  if (!b.ok()) return {};
  // One slot: its view made a tensor of its own.
  TensorId acc = u == 1 ? b.Cont(slots.front()) : slots.front();
  for (std::uint64_t i = 1; i < u && b.ok(); ++i) {
    acc = b.Add(acc, slots[i]);
  }
  return acc;
}

Outputs ExpertReduce(D& d) {
  if (!d.need(2, 4)) {
    return {};
  }
  return One(Reduce(d, d.s(0), d.s(1), d.s(2), d.s(3)));
}

Outputs SsmConv(D& d) {
  if (!d.need(2, 2)) {
    return {};
  }
  Builder& b = d.b();
  const TensorDesc& sx = d.t(d.s(0));
  const TensorDesc& c = d.t(d.s(1));
  const std::uint64_t k = c.ne(0);
  const std::uint64_t t = sx.ne(0) - k + 1;
  TensorId acc{};
  for (std::uint64_t j = 0; j < k && b.ok(); ++j) {
    const TensorId xj = b.View(d.s(0), S(t, sx.ne(1), sx.ne(2)), sx.nb, j * sx.nb[0]);
    const TensorId cj = d.View2(d.s(1), 1, c.ne(1), c.nb[0], c.nb[1], j * c.nb[0]);
    const TensorId term = b.Mul(xj, cj);
    acc = acc.valid() ? b.Add(acc, term) : term;
  }
  return One(b.Cont(b.Permute(acc, {1, 0, 2, 3})));
}

Outputs GatedDeltaNet(D& d) {
  if (!d.need(6, 6)) {
    return {};
  }
  Builder& b = d.b();
  const TensorDesc& q = d.t(d.s(0));
  const TensorDesc& v = d.t(d.s(2));
  const std::uint64_t sk = q.ne(0);
  const std::uint64_t hk = q.ne(1);
  const std::uint64_t bq = q.ne(3);
  const std::uint64_t sv = v.ne(0);
  const std::uint64_t h = v.ne(1);
  const std::uint64_t t = v.ne(2);
  const std::uint64_t seqs = v.ne(3);
  // A token's [n, 1, heads, seqs] slice of a [n, heads, tokens, seqs] tensor.
  const auto token = [&](TensorId x, std::uint64_t i) {
    const TensorDesc& xt = d.t(x);
    return b.View(x, S(xt.ne(0), 1, xt.ne(1), xt.ne(3)), Nb{xt.nb[0], xt.nb[1], xt.nb[1], xt.nb[3]},
                  i * xt.nb[2]);
  };
  // q's or k's token i for every value head and sequence: head h reads h %
  // Hk and sequence s reads s / (B / Bq). Their [Sk, Hk, 1, Bq] view tiled
  // to [Sk, H, B / Bq, Bq] holds (h, s = j · (B / Bq) + r) at (h, r, j),
  // which is s's place in [Sk, H, B].
  const auto spread = [&](TensorId x, std::uint64_t i) {
    const TensorDesc& xt = d.t(x);
    const TensorId one =
        b.View(x, S(sk, hk, 1, bq), Nb{xt.nb[0], xt.nb[1], xt.nb[2], xt.nb[3]}, i * xt.nb[2]);
    return b.Reshape(b.Repeat(one, S(sk, h, seqs / bq, bq)), S(sk, 1, h, seqs));
  };
  // Each token's output, [Sv, 1, H, B] and so one row of Sv · H · B values,
  // is stored at its own row of one [Sv · H · B, T] buffer: a store copies
  // only that row, where joining the outputs would copy every earlier one
  // again. Token i goes to row rows[i] and is gathered back from it, so any
  // permutation serves as rows: an argsort's order of T equal values (0 to
  // T - 1 for a stable sort, but an unstable one's order is as good).
  const std::uint64_t width = sv * h * seqs;
  TensorId buffer = b.Fill(S(width, t), 0.0f);
  const TensorId rows = b.Argsort(b.Fill(S(t), 0.0f), false);
  TensorId state = d.s(5);
  const float scale = 1.0f / std::sqrt(static_cast<float>(sk));
  for (std::uint64_t i = 0; i < t && b.ok(); ++i) {
    const TensorId qt = spread(d.s(0), i);
    const TensorId kt = spread(d.s(1), i);
    const TensorId decay = b.Unary(token(d.s(3), i), UnaryFn::kExp);  // [1 or Sk, 1, H, B]
    state = b.Mul(state, decay);
    const TensorId kv = b.MatMul(state, kt);  // [Sv, 1, H, B]
    const TensorId delta = b.Mul(b.Sub(token(d.s(2), i), kv), token(d.s(4), i));
    const TensorId outer =
        b.MatMul(b.Reshape(kt, S(1, sk, h, seqs)), b.Reshape(delta, S(1, sv, h, seqs)));
    state = b.Add(state, outer);                               // [Sk, Sv, H, B]
    const TensorId out = b.Scale(b.MatMul(state, qt), scale);  // [Sv, 1, H, B]
    buffer = b.CacheStore(buffer, b.Reshape(out, S(width, 1)), Element(d, rows, i));
  }
  // [Sv · H · B, T] in token order, as [Sv, H, B, T], then [Sv, H, T, B].
  const TensorId tokens = b.Reshape(b.GetRows(buffer, rows), S(sv, h, seqs, t));
  Outputs o;
  o.ids[0] = b.Cont(b.Permute(tokens, {0, 1, 3, 2}));
  o.ids[1] = state;
  o.n = 2;
  return o;
}

// ---------------------------------------------------------------- fused kinds

Outputs Many(std::initializer_list<TensorId> ids) {
  Outputs o;
  for (const TensorId id : ids) {
    o.ids.at(o.n++) = id;
  }
  return o;
}

Outputs DupUpAdd(D& d, const DupUpParams& p) {
  if (!d.need(2, 2)) {
    return {};
  }
  Builder& b = d.b();
  const TensorDesc& x = d.t(d.s(0));
  const TensorDesc& sc = d.t(d.s(1));
  const std::uint64_t w = sc.ne(0);
  const std::uint64_t h = sc.ne(1);
  const std::uint64_t ci = sc.ne(2);
  const std::uint64_t co = x.ne(2);
  const std::uint64_t ft = p.factor_t;
  const std::uint64_t virt = co * ft * 4;
  if (!d.check(
          ft >= 1 && sc.ne(3) == 1 && virt % ci == 0 && x.ne(0) == 2 * w && x.ne(1) == 2 * h,
          GraphError::kShape,
          std::format("dup-up of [{}, {}, {}] into {} channels at {} frames", w, h, ci, co, ft))) {
    return {};
  }
  const std::uint64_t repeats = virt / ci;
  const std::uint64_t px = w * h;
  // Each input channel repeated: virtual channel j = c · repeats + r.
  const TensorId rep =
      b.Repeat(b.Reshape(b.Cont(d.s(1)), S(px, 1, ci)), S(px, repeats, ci));  // [P, rep, Ci]
  // j = (d + 2b) + 4 · (f + ft · o): the last frame's [P, 4, Co].
  const std::uint64_t e = d.t(rep).nb[0];
  const TensorId frame =
      b.Cont(b.View(rep, S(px, 4, co), Nb{e, e * px, e * px * 4 * ft, e * px * 4 * ft * co},
                    (ft - 1) * 4 * px * e));
  // [W, H, d, (b, o)] to [d, W, H, (b, o)]; then [(d, x), y, b, o] to
  // [(d, x), b, y, o], which is [2W, 2H, Co].
  const TensorId dx =
      b.Cont(b.Permute(b.Reshape(frame, S(w, h, 2, 2 * co)), {1, 2, 0, 3}));  // [2, W, H, 2Co]
  const TensorId out =
      b.Cont(b.Permute(b.Reshape(dx, S(2 * w, h, 2, co)), {0, 2, 1, 3}));  // [2W, 2, H, Co]
  return One(b.Add(d.s(0), b.Reshape(out, S(2 * w, 2 * h, co))));
}

Outputs DecomposeFused(D& d, FusedKind kind, const ParamBytes& params) {
  Builder& b = d.b();
  switch (kind) {
    case FusedKind::kRmsNormMul: {
      if (!d.need(2, 2)) {
        return {};
      }
      const auto p = Get<NormParams>(params);
      return One(b.Mul(b.RmsNorm(d.s(0), p.eps), d.s(1)));
    }
    case FusedKind::kRmsNormMulRope: {
      if (!d.need(3, 4)) {
        return {};
      }
      const auto p = Get<NormRopeParams>(params);
      return One(b.Rope(b.Mul(b.RmsNorm(d.s(0), p.eps), d.s(1)), d.s(2), d.s(3), p.rope));
    }
    case FusedKind::kRmsNormMulAdd: {
      if (!d.need(3, 3)) {
        return {};
      }
      const auto p = Get<NormParams>(params);
      return One(b.Add(b.Mul(b.RmsNorm(d.s(0), p.eps), d.s(1)), d.s(2)));
    }
    case FusedKind::kRopeStore: {
      if (!d.need(4, 5)) {
        return {};
      }
      const auto p = Get<RopeParams>(params);
      const TensorId r = b.Rope(d.s(0), d.s(1), d.s(4), p);
      if (!b.ok()) {
        return {};
      }
      const TensorDesc& rt = d.t(r);
      const TensorId rows = b.Reshape(r, S(rt.ne(0) * rt.ne(1), rt.ne(2), rt.ne(3)));
      return One(b.CacheStore(d.s(2), rows, d.s(3)));
    }
    case FusedKind::kMatMulGlu: {
      if (!d.need(3, 3)) {
        return {};
      }
      const auto p = Get<GluParams>(params);
      return One(b.Glu(b.MatMul(d.s(0), d.s(2)), b.MatMul(d.s(1), d.s(2)), p));
    }
    case FusedKind::kMatMulAdd: {
      if (!d.need(3, 3)) {
        return {};
      }
      return One(b.Add(b.MatMul(d.s(0), d.s(1)), d.s(2)));
    }
    case FusedKind::kOwnerAttention: {
      const auto p = Get<OwnerAttnParams>(params);
      const std::uint64_t n = p.owners;
      const std::uint64_t cohort = p.cohort == 0 ? n : p.cohort;
      if (!d.check(n >= 1 && n <= 4 && !p.attn.sinks, GraphError::kParams,
                   "one to four owners, no sinks") ||
          !d.check(cohort <= kMaxOwnerCohort && std::uint64_t{p.offset} + n <= cohort,
                   GraphError::kParams, "a group within its cohort") ||
          !d.need(2 + (2 * n), 2 + (2 * n))) {
        return {};
      }
      // Owners on dim 3 (op.h): each owner's queries and mask are its own
      // matrix, the mask's rows padded as its producer padded them.
      const TensorDesc& q = d.t(d.s(0));
      const TensorDesc& mask = d.t(d.s(1));
      if (!d.check(q.ne(3) == n && mask.ne(3) == n && mask.ne(2) == 1, GraphError::kShape,
                   "one query and mask matrix an owner, on dim 3")) {
        return {};
      }
      const std::uint64_t r = q.ne(1);
      TensorId out{};
      for (std::uint64_t o = 0; o < n && b.ok(); ++o) {
        const TensorId k = d.s(2 + o);
        const TensorId v = d.s(2 + n + o);
        const std::uint64_t cells = d.t(k).ne(1);
        if (!d.check(cells <= mask.ne(0) && r <= mask.ne(1), GraphError::kShape,
                     "each owner's mask covers its cells and rows")) {
          return {};
        }
        const TensorId qo = b.View(d.s(0), S(q.ne(0), r, q.ne(2), 1), q.nb, o * q.nb[3]);
        const TensorId mo = b.View(d.s(1), S(cells, r, 1, 1), mask.nb, o * mask.nb[3]);
        const TensorId a = b.Attention(qo, k, v, mo, {}, p.attn);  // [Dv, H, R, 1]
        out = out.valid() ? b.Concat(out, a, 3) : a;
      }
      return One(out);
    }
    case FusedKind::kMoeGlu: {
      if (!d.need(5, 5)) {
        return {};
      }
      const auto p = Get<GluParams>(params);
      const TensorId gate = b.Mul(d.s(0), ScaleRows(d, d.s(3), d.s(2)));
      const TensorId up = b.Mul(d.s(1), ScaleRows(d, d.s(4), d.s(2)));
      return One(b.Glu(gate, up, p));
    }
    case FusedKind::kMoeCombine: {
      if (!d.need(4, 6)) {
        return {};
      }
      // A gate is [1, T]: s_down in its place (a source short) would
      // broadcast over the shared expert's output unnoticed.
      if (d.s(4).valid() &&
          !d.check(d.t(d.s(4)).ne(0) == 1 && d.t(d.s(4)).ne(1) == d.t(d.s(3)).ne(1),
                   GraphError::kShape, "the shared expert's gate is [1, T]")) {
        return {};
      }
      const TensorId routed = b.ExpertReduce(d.s(0), d.s(2), d.s(1), d.s(5));
      const TensorId shared =
          d.s(4).valid() ? b.Mul(d.s(3), b.Unary(d.s(4), UnaryFn::kSigmoid)) : d.s(3);
      return One(b.Add(routed, shared));
    }
    case FusedKind::kHcNorm: {
      if (!d.need(2, 2)) {
        return {};
      }
      const auto p = Get<NormParams>(params);
      const TensorDesc& x = d.t(d.s(0));
      const TensorId n = b.Reshape(b.RmsNorm(d.s(0), p.eps), S(x.ne(0) * x.ne(1), x.ne(2)));
      return One(b.Mul(n, d.s(1)));
    }
    case FusedKind::kHcNormPost: {
      if (!d.need(3, 4)) {
        return {};
      }
      const auto p = Get<HcNormPostParams>(params);
      const TensorId post =
          b.HcPost(d.s(0), d.s(1), d.s(2), d.s(3), HcParams{.streams = p.streams});
      if (!b.ok()) {
        return {};
      }
      const TensorDesc& pt = d.t(post);
      return Many({post, b.RmsNorm(b.Reshape(post, S(pt.ne(0) * pt.ne(1), pt.ne(2))), p.eps)});
    }
    case FusedKind::kQkPrep: {
      if (!d.need(5, 6)) {
        return {};
      }
      const auto p = Get<NormRopeParams>(params);
      const TensorId q = b.Rope(b.Mul(b.RmsNorm(d.s(0), p.eps), d.s(1)), d.s(4), d.s(5), p.rope);
      const TensorId k = b.Rope(b.Mul(b.RmsNorm(d.s(2), p.eps), d.s(3)), d.s(4), d.s(5), p.rope);
      return Many({q, k});
    }
    case FusedKind::kPreparedProducts: {
      if (!d.need(2, 5)) {
        return {};
      }
      // The weights first, the input last.
      const std::size_t products = d.count() - 1;
      const TensorId x = d.s(products);
      Outputs o;
      for (std::size_t i = 0; i < products; ++i) {
        if (!d.check(d.s(i).valid() && x.valid(), GraphError::kArity,
                     "every product's weight, and the input")) {
          return {};
        }
        o.ids.at(i) = b.MatMul(d.s(i), x);
      }
      o.n = static_cast<std::uint8_t>(products);
      return o;
    }
    case FusedKind::kNormGate: {
      if (!d.need(3, 3)) {
        return {};
      }
      const auto p = Get<NormParams>(params);
      const TensorId n = b.Mul(b.RmsNorm(d.s(0), p.eps), d.s(1));
      return One(b.Mul(n, b.Unary(d.s(2), UnaryFn::kSigmoid)));
    }
    case FusedKind::kLayerNormModulate: {
      if (!d.need(2, 2)) {
        return {};
      }
      const auto p = Get<NormParams>(params);
      return One(b.Mul(b.LayerNorm(d.s(0), p.eps), b.Scale(d.s(1), 1.0f, 1.0f)));
    }
    case FusedKind::kGatedResidual: {
      if (!d.need(3, 3)) {
        return {};
      }
      return One(b.Add(d.s(0), b.Mul(d.s(2), b.Unary(d.s(1), UnaryFn::kTanh))));
    }
    case FusedKind::kGatedResidualNorm: {
      if (!d.need(4, 4)) {
        return {};
      }
      const auto p = Get<NormParams>(params);
      const TensorId x = b.Add(d.s(0), b.Mul(d.s(2), b.Unary(d.s(1), UnaryFn::kTanh)));
      return Many({x, b.Mul(b.LayerNorm(x, p.eps), b.Scale(d.s(3), 1.0f, 1.0f))});
    }
    case FusedKind::kEulerStep: {
      if (!d.need(3, 3)) {
        return {};
      }
      return One(b.Add(d.s(0), b.Mul(d.s(1), d.s(2))));
    }
    case FusedKind::kDupUpAdd:
      return DupUpAdd(d, Get<DupUpParams>(params));
    case FusedKind::kGroupedStores:  // a batch key (op.h), never a node
    case FusedKind::kNone:
      break;
  }
  b.Refuse(GraphError::kParams, "an unknown fused kind");
  return {};
}

bool Composite(Op op) {
  switch (op) {
    case Op::kHcPre:
    case Op::kHcPost:
    case Op::kHcMix:
    case Op::kIndexScore:
    case Op::kPool:
    case Op::kRoute:
    case Op::kExpertReduce:
    case Op::kSsmConv:
    case Op::kGatedDeltaNet:
      return true;
    default:
      return false;
  }
}

std::expected<Graph, VerifyError> ExpandOnce(const Graph& graph, bool composites,
                                             std::vector<TensorId>& map, bool& more) {
  Builder b;
  map.assign(graph.tensor_count(), TensorId{});
  more = false;
  std::vector<TensorId> sources;
  std::vector<TensorDesc> requested;
  std::vector<format::Encoding> encodings;
  for (std::size_t n = 0; n < graph.nodes().size(); ++n) {
    const Node& node = graph.nodes()[n];
    b.SetComponent(node.component);
    if (node.batch == Batch::kPerOwner && node.nout > 0) {
      b.SetOwner(graph.tensor(node.out[0]).segment);
    } else {
      b.SetJoined();
    }
    sources.clear();
    for (const TensorId s : node.sources()) {
      sources.push_back(s.valid() ? map.at(s.v) : TensorId{});
    }
    Outputs outs;
    const bool split = node.key.op == Op::kFused || (composites && Composite(node.key.op));
    if (split) {
      encodings.clear();
      for (const TensorId o : node.outputs()) {
        encodings.push_back(graph.tensor(o).format.encoding);
      }
      outs = Decompose(b, node.key, node.params, sources, encodings);
    } else {
      requested.clear();
      for (const TensorId o : node.outputs()) {
        requested.push_back(graph.tensor(o));
      }
      outs = b.Emit(node.key, node.params, sources, requested,
                    node.nout > 0 ? graph.name(node.out[0]) : std::string_view());
    }
    if (!b.ok()) {
      VerifyError e = *b.error();
      e.node = static_cast<std::uint32_t>(n);
      return std::unexpected(e);
    }
    for (std::size_t i = 0; i < node.nout; ++i) {
      map.at(node.out.at(i).v) = outs[i];
    }
  }
  for (const TensorId o : graph.outputs()) {
    b.Output(map.at(o.v));
  }
  for (const TensorId k : graph.keep()) {
    b.Keep(map.at(k.v));
  }
  auto g = std::move(b).Finish();
  if (g) {
    for (const Node& node : g->nodes()) {
      more = more || node.key.op == Op::kFused || (composites && Composite(node.key.op));
    }
  }
  return g;
}

}  // namespace

bool HasDecomposition(OpKey key) {
  if (key.op == Op::kFused) {
    return key.fused != FusedKind::kNone && key.fused != FusedKind::kGroupedStores &&
           static_cast<std::uint16_t>(key.fused) <=
               static_cast<std::uint16_t>(FusedKind::kDupUpAdd);
  }
  return key.fused == FusedKind::kNone && Composite(key.op);
}

std::optional<std::size_t> AliasedSource(OpKey key, std::size_t output) {
  if (key.op != Op::kFused) {
    return std::nullopt;
  }
  if (key.fused == FusedKind::kRopeStore && output == 0) {
    return 2;
  }
  return std::nullopt;
}

Outputs Decompose(Builder& b, OpKey key, const ParamBytes& params,
                  std::span<const TensorId> sources, std::span<const format::Encoding> encodings) {
  D d(b, sources);
  Outputs o;
  if (key.op == Op::kFused) {
    o = DecomposeFused(d, key.fused, params);
  } else {
    switch (key.op) {
      case Op::kHcPre:
        o = HcPre(d, Get<HcParams>(params));
        break;
      case Op::kHcPost:
        o = HcPost(d, Get<HcParams>(params));
        break;
      case Op::kHcMix:
        o = HcMix(d, Get<HcParams>(params));
        break;
      case Op::kIndexScore:
        o = IndexScore(d);
        break;
      case Op::kPool:
        o = Pool(d, Get<PoolParams>(params));
        break;
      case Op::kRoute:
        o = Route(d, Get<RouteParams>(params));
        break;
      case Op::kExpertReduce:
        o = ExpertReduce(d);
        break;
      case Op::kSsmConv:
        o = SsmConv(d);
        break;
      case Op::kGatedDeltaNet:
        o = GatedDeltaNet(d);
        break;
      default:
        b.Refuse(GraphError::kParams, std::format("{} has no decomposition", OpKeyName(key)));
        return {};
    }
  }
  if (!b.ok()) {
    return {};
  }
  for (std::size_t i = 0; i < o.n && i < encodings.size(); ++i) {
    if (AliasedSource(key, i)) {
      continue;
    }
    const format::Encoding have = b.tensor(o.ids.at(i)).format.encoding;
    if (encodings[i] != have && ResultEncoding(encodings[i]) && ResultEncoding(have)) {
      o.ids.at(i) = b.Convert(o.ids.at(i), encodings[i]);
    }
  }
  return o;
}

std::expected<Graph, VerifyError> Expand(const Graph& graph, bool composites,
                                         std::vector<TensorId>* map) {
  std::vector<TensorId> total;
  std::vector<TensorId> step;
  bool more = false;
  auto g = ExpandOnce(graph, composites, total, more);
  // A decomposition may emit composite ops (kMoeCombine's expert
  // reduction, kHcNormPost's spread): expand again until none is left.
  for (int pass = 0; g && more && pass < 4; ++pass) {
    auto next = ExpandOnce(*g, composites, step, more);
    if (!next) {
      return next;
    }
    for (TensorId& id : total) {
      id = id.valid() ? step.at(id.v) : id;
    }
    g = std::move(next);
  }
  if (g && map != nullptr) {
    *map = std::move(total);
  }
  return g;
}

}  // namespace llmp::ir
