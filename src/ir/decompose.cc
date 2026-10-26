// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include "ir/decompose.h"

#include <algorithm>
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
#include "ir/op_traits.h"
#include "ir/verify.h"

namespace llmp::ir {
namespace {

using format::Shape;

Shape S(std::uint64_t a, std::uint64_t b = 1, std::uint64_t c = 1, std::uint64_t d = 1) {
  return ShapeOf({a, b, c, d});
}

using Nb = std::array<std::uint64_t, format::kMaxDims>;

std::optional<std::uint64_t> VectorScratchBytes(std::uint64_t count, std::size_t element_size) {
  return VectorAdmissionBytes(count, element_size);
}

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
  // Reshape only packed values. Fused inference uses packed stand-ins,
  // while expansion receives the real sources and must retain their order.
  TensorId Packed(TensorId id) {
    if (!b_.ok()) return {};
    return IsContiguous(t(id)) ? id : b_.Cont(id);
  }
  bool need(std::size_t required, std::size_t total) {
    if (!b_.ok()) {
      return false;
    }
    if (src_.size() < required || src_.size() > total) {
      b_.Refuse(GraphError::kArity,
                DiagFormat("takes {} to {} sources, not {}", required, total, src_.size()));
      return false;
    }
    for (std::size_t i = 0; i < required; ++i) {
      if (!src_[i].valid()) {
        b_.Refuse(GraphError::kArity, DiagFormat("source {} is required", i));
        return false;
      }
    }
    return true;
  }
  bool check(bool ok, GraphError e, std::string_view what) {
    if (!ok && b_.ok()) {
      b_.Refuse(e, what);
    }
    return ok && b_.ok();
  }
  template <std::size_t N>
  bool check(bool ok, GraphError e, const DiagnosticMessage<N>& what) {
    if (!ok && b_.ok()) b_.Refuse(e, what);
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
    if (p.scale != 1.0f) {
      acc = b.Scale(acc, p.scale);
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
  return One(hc == 1 ? b.Cont(out) : out);
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
    const std::uint64_t off = p.mix_offset;
    const std::uint64_t sc_idx = p.scale_index;
    const auto weights = [&](std::uint64_t at, std::uint64_t which) {
      TensorId v = d.View2(mixes, hc, t, mt.nb[0], mt.nb[1], at * mt.nb[0]);
      v = b.Add(b.Mul(v, Element(d, scale, which)), Range(d, base, at, hc));
      return b.Unary(v, UnaryFn::kSigmoid);
    };
    const TensorId pre = b.Scale(weights(off, sc_idx), 1.0f, p.eps);
    const TensorId post = b.Scale(weights(off + hc, sc_idx + 1), 2.0f, 0.0f);
    const TensorId pre_rows = b.Reshape(pre, S(hc, 1, t));
    const TensorId post_rows = b.Reshape(post, S(hc, 1, t));
    return One(b.Concat(pre_rows, post_rows, 1));
  }
  // Sinkhorn: [dst, src, t] from the hc² mixes at mix_offset (or 2·hc when packed).
  const std::uint64_t e = mt.nb[0];
  const std::uint64_t off = p.mix_offset != 0 ? p.mix_offset : (mt.ne(0) == hc * hc ? 0 : 2 * hc);
  const std::uint64_t sc_idx = p.scale_index != 0 ? p.scale_index : (d.t(scale).ne(0) == 1 ? 0 : 2);
  TensorId m = b.View(mixes, S(hc, hc, t), Nb{e, e * hc, mt.nb[1], mt.nb[1] * t}, off * e);
  m = b.Mul(m, Element(d, scale, sc_idx));
  const TensorDesc& bt = d.t(base);
  m = b.Add(m, b.View(base, S(hc, hc),
                      Nb{bt.nb[0], bt.nb[0] * hc, bt.nb[0] * hc * hc, bt.nb[0] * hc * hc},
                      off * bt.nb[0]));
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
    // The score is F32. Widening a half mask is exact and makes the
    // primitive add's types explicit for backends with homogeneous adds.
    const TensorId mask = d.t(d.s(3)).format.encoding == format::Encoding::kF32
                              ? d.s(3)
                              : b.Convert(d.Packed(d.s(3)), format::Encoding::kF32);
    sum = b.Add(sum, mask);
  }
  return One(sum);
}

Outputs RoundedIndexScore(D& d, const RoundedIndexScoreParams& p) {
  if (!d.need(4, 4) ||
      !d.check(p.query == QueryRounding::kF16 && p.reserved == std::array<std::uint8_t, 3>{},
               GraphError::kParams, "F16 query rounding with zero reserved bytes"))
    return {};
  Builder& b = d.b();
  const TensorDesc q = d.t(d.s(0));
  const TensorId key_values = d.Packed(d.s(1));
  const TensorId head_weights = d.Packed(d.s(2));
  const TensorDesc k = d.t(key_values);
  const TensorDesc w = d.t(head_weights);
  // The primitive float product takes F32 activations. Widening the rounded
  // half query is exact; both conversion roots are ordinary funded values.
  const TensorId half = b.Convert(d.Packed(d.s(0)), format::Encoding::kF16);
  const TensorId query = b.Convert(half, format::Encoding::kF32);
  const std::uint64_t cells = k.ne(2);
  const TensorId keys =
      b.View(key_values, S(k.ne(0), cells), Nb{k.nb[0], k.nb[2], k.nb[2] * cells, k.nb[3]}, 0);
  TensorId dots = b.MatMul(keys, query, {.accumulate = Accumulate::kF32});
  dots = b.Unary(dots, UnaryFn::kRelu);
  const TensorId weights =
      b.View(head_weights, S(1, w.ne(0), w.ne(1)), Nb{w.nb[0], w.nb[0], w.nb[1], w.nb[3]}, 0);
  const TensorId weighted = b.Mul(dots, weights);
  const TensorId sum = b.SumRows(b.Cont(b.Permute(weighted, {1, 0, 2, 3})));
  return One(
      b.PrefixOverwrite(b.Reshape(sum, S(cells, q.ne(2))), d.s(3), {.fill_bits = 0xff800000U}));
}

Outputs VisibleTopK(D& d, const TopKParams& p) {
  if (!d.need(2, 2)) {
    return {};
  }
  Builder& b = d.b();
  const TensorDesc& x = d.t(d.s(0));
  const std::uint64_t c = x.ne(0);
  const std::uint64_t t = x.ne(1);
  const std::uint64_t keep = std::min<std::uint64_t>(p.k, c);
  // Overwrite invisible tail columns (>= visible_len) with quiet NaN so every
  // visible score (finite, -inf, or NaN at a lower column index) ranks ahead
  // of every invisible column under stable descending TopK.
  const TensorId prefixed = b.PrefixOverwrite(d.s(0), d.s(1), {.fill_bits = 0x7fc00000U});
  const TensorId top_ids = b.TopK(prefixed, static_cast<std::uint32_t>(keep));
  const TensorId top_f32 = b.Convert(top_ids, format::Encoding::kF32);
  const TensorId order = b.Argsort(top_f32, false);
  const TensorId sorted =
      b.Reshape(b.GetRows(b.Reshape(top_f32, S(1, keep, t)), order), S(keep, t));
  TensorId masked = b.PrefixOverwrite(sorted, d.s(1), {.fill_bits = 0xbf800000U});
  if (p.k > keep) {
    masked = b.Concat(masked, b.Fill(S(p.k - keep, t), -1.0f), 0);
  }
  return One(b.Convert(masked, format::Encoding::kI32));
}

Outputs SelectedMask(D& d, const SelectedMaskParams& p) {
  if (!d.need(2, 2)) {
    return {};
  }
  Builder& b = d.b();
  const TensorDesc& raw = d.t(d.s(0));
  const TensorDesc& ids = d.t(d.s(1));
  const std::uint64_t window = raw.ne(0);
  const std::uint64_t t = raw.ne(1);
  const std::uint64_t k = ids.ne(0);
  const std::uint64_t cells = p.cells;
  const float neg_inf = -std::numeric_limits<float>::infinity();
  TensorId raw_part = d.s(0);
  if (window < p.raw_cells) {
    const TensorId pad = b.Fill(S(p.raw_cells - window, t), neg_inf, format::Encoding::kF16);
    raw_part = b.Concat(raw_part, pad, 0);
  }
  const TensorId iota_i32 = b.Argsort(b.Fill(S(cells), 0.0f), false);
  const TensorId iota_f32 = b.Reshape(b.Convert(iota_i32, format::Encoding::kF32), S(cells, 1, 1));
  const TensorId grid = b.Repeat(iota_f32, S(cells, k, t));
  const TensorId ids_f32 = b.Reshape(b.Convert(d.s(1), format::Encoding::kF32), S(1, k, t));
  const TensorId diff = b.Sub(grid, ids_f32);
  const TensorId match = b.Scale(b.Unary(b.Unary(diff, UnaryFn::kAbs), UnaryFn::kSgn), -1.0f, 1.0f);
  const TensorId hits = b.Reshape(b.SumRows(b.Cont(b.Permute(match, {1, 0, 2, 3}))), S(cells, t));
  const TensorId cond = b.Convert(hits, format::Encoding::kI32);
  const TensorId zero_f16 = b.Fill(S(cells, t), 0.0f, format::Encoding::kF16);
  const TensorId ninf_f16 = b.Fill(S(cells, t), neg_inf, format::Encoding::kF16);
  const TensorId comp_part = b.Select(cond, zero_f16, ninf_f16);
  return One(b.Concat(raw_part, comp_part, 0));
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
      // IR softmax gives zero for an all-masked row. GGML CUDA's
      // softmax gives NaN there; its fminf/fmaxf clamp maps that to zero
      // while preserving every finite probability. The reference already
      // has zero probabilities and the clamp is its identity.
      const TensorId probabilities = b.SoftMax(runs(d.s(1), i3), {}, {}, SoftMaxParams{});
      vals = b.Mul(vals, b.Clamp(probabilities, 0.0F, 1.0F));
    }
    TensorId sum = b.Reshape(b.SumRows(vals), S(c, n, a2));
    if (p.kind == PoolKind::kMean) {
      sum = b.Scale(sum, 1.0f / static_cast<float>(r));
    }
    out = out.valid() ? b.Concat(out, sum, 3) : sum;
  }
  return One(b.tensor(out).is_view() ? b.Cont(out) : out);
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
    // looked up already (a token-hash side input's), aliased as they are
    // (a view of a strided side input is refused: SameViewSource).
    ids = d.s(3).valid() ? b.GetRows(d.s(2), d.s(3)) : d.s(2);
  } else {
    const TensorId selection = d.s(1).valid() ? b.Add(probs, d.s(1)) : probs;
    const TensorId order = b.Argsort(selection, true);
    const TensorDesc& ot = d.t(order);
    // The top U of each row's order, a view at the sort's row pitch (which
    // routed reads take back at that pitch). The fused node records packed
    // ids, so an expansion refuses a later view of them (SameViewSource)
    // rather than read the sort's other elements.
    ids = u == e ? order
                 : b.View(order, S(u, t), Nb{ot.nb[0], ot.nb[1], ot.nb[1] * t, ot.nb[1] * t}, 0);
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
  if (!b.ok()) return {};
  const auto scratch_bytes = VectorScratchBytes(u, sizeof(TensorId));
  if (!scratch_bytes) {
    b.Refuse(GraphError::kBounds, "expert reduction scratch byte count overflows");
    return {};
  }
  auto scratch_loan = b.FundScratch(*scratch_bytes);
  if (!b.ok()) return {};
  std::vector<TensorId> slots;
  slots.reserve(static_cast<std::size_t>(u));
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
  // Spread q or k across all tokens at once: head h reads h % Hk and sequence
  // s reads s / (B / Bq), producing [Sk, H, T, B].
  const auto spread_all = [&](TensorId x) {
    TensorId y = x;
    if (hk != h) {
      y = b.Repeat(y, S(sk, h, t, bq));
    }
    if (bq != seqs) {
      y = b.Reshape(d.Packed(y), S(sk * h * t, 1, bq));
      y = b.Reshape(b.Repeat(y, S(sk * h * t, seqs / bq, bq)), S(sk, h, t, seqs));
    }
    return y;
  };
  const TensorId q_all = spread_all(d.s(0));
  const TensorId k_all = spread_all(d.s(1));
  const TensorId kb_all = b.Mul(k_all, d.s(4));
  const TensorId decay_all = b.Unary(d.s(3), UnaryFn::kExp);
  if (!b.ok()) return {};
  const TensorDesc& kbt = d.t(kb_all);
  TensorId state = d.s(5);
  const float scale = 1.0f / std::sqrt(static_cast<float>(sk));
  const auto initial_scratch_bytes = VectorScratchBytes(t, sizeof(TensorId));
  if (!initial_scratch_bytes) {
    b.Refuse(GraphError::kBounds, "gated delta net scratch byte count overflows");
    return {};
  }
  auto outs_loan = b.FundScratch(*initial_scratch_bytes);
  if (!b.ok()) return {};
  std::vector<TensorId> outs;
  outs.reserve(static_cast<std::size_t>(t));
  for (std::uint64_t i = 0; i < t && b.ok(); ++i) {
    const TensorId qt = token(q_all, i);
    const TensorId kt = token(k_all, i);
    const TensorId decay = token(decay_all, i);  // [1 or Sk, 1, H, B]
    state = b.Mul(state, decay);
    const TensorId kv = b.MatMul(state, kt);  // [Sv, 1, H, B]
    const TensorId diff = b.Sub(token(d.s(2), i), kv);
    const TensorId kbt_rows = b.View(kb_all, S(1, sk, h, seqs),
                                     Nb{kbt.nb[0], kbt.nb[0], kbt.nb[1], kbt.nb[3]}, i * kbt.nb[2]);
    const TensorId outer = b.MatMul(kbt_rows, b.Reshape(diff, S(1, sv, h, seqs)));
    state = b.Add(state, outer);          // [Sk, Sv, H, B]
    outs.push_back(b.MatMul(state, qt));  // [Sv, 1, H, B]
  }
  while (outs.size() > 1 && b.ok()) {
    const std::size_t next_capacity = (outs.size() + 1) / 2;
    const auto next_scratch_bytes = VectorScratchBytes(next_capacity, sizeof(TensorId));
    if (!next_scratch_bytes) {
      b.Refuse(GraphError::kBounds, "gated delta net reduction scratch byte count overflows");
      return {};
    }
    auto next_loan = b.FundScratch(*next_scratch_bytes);
    if (!b.ok()) return {};
    std::vector<TensorId> next;
    next.reserve(next_capacity);
    for (std::size_t i = 0; i < outs.size() && b.ok(); i += 2) {
      next.push_back(i + 1 < outs.size() ? b.Concat(outs[i], outs[i + 1], 1) : outs[i]);
    }
    outs = std::move(next);
    outs_loan = std::move(next_loan);
  }
  if (!b.ok()) return {};
  const TensorId joined = outs.empty() ? TensorId{} : outs.front();  // [Sv, T, H, B]
  Outputs o;
  o.ids[0] = b.Scale(b.Cont(b.Permute(joined, {0, 2, 1, 3})), scale);
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
          DiagFormat("dup-up of [{}, {}, {}] into {} channels at {} frames", w, h, ci, co, ft))) {
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

// Qwen3.8's hyper-connection combine (hc.combine's order): x + out · 2 ·
// sigmoid(logits / hc) on each stream.
TensorId HcCombineOf(D& d, TensorId x, TensorId out, TensorId logits, std::uint64_t hc) {
  Builder& b = d.b();
  if (!b.ok()) return {};
  const TensorDesc xt = d.t(x);
  const std::uint64_t w = xt.ne(0);
  const std::uint64_t t = xt.ne(2);
  if (!d.check(hc != 0 && xt.shape().ne == S(w, hc, t).ne && d.t(out).shape().ne == S(w, t).ne &&
                   d.t(logits).shape().ne == S(hc, t).ne,
               GraphError::kShape, "one block output and logit per stream and row"))
    return {};
  const TensorId block = b.Repeat(b.Reshape(d.Packed(out), S(w, 1, t)), S(w, hc, t));
  TensorId weights = b.Unary(b.Scale(logits, 1.0f / static_cast<float>(hc)), UnaryFn::kSigmoid);
  weights = b.Reshape(b.Scale(weights, 2.0f), S(1, hc, t));
  return b.Add(x, b.Mul(block, weights));
}

// Each stream's RMS norm, the streams as one row times the weight: [W ·
// hc, T].
TensorId HcNormOf(D& d, TensorId x, TensorId norm, float eps) {
  Builder& b = d.b();
  if (!b.ok()) return {};
  const TensorDesc xt = d.t(x);
  return b.Mul(b.Reshape(b.RmsNorm(x, eps), S(xt.ne(0) * xt.ne(1), xt.ne(2))), norm);
}

// (1 / hc) Σ_k xn_k · sigmoid(gate_k), xn and gate [W · hc, T].
TensorId HcFoldOf(D& d, TensorId xn, TensorId gate, std::uint64_t hc) {
  Builder& b = d.b();
  if (!b.ok()) return {};
  const TensorDesc xt = d.t(xn);
  if (!d.check(hc != 0 && xt.ne(0) % hc == 0 && xt.ne(2) == 1 && xt.ne(3) == 1 &&
                   d.t(gate).shape().ne == xt.shape().ne,
               GraphError::kShape, "a gate per value of whole streams"))
    return {};
  const std::uint64_t w = xt.ne(0) / hc;
  const std::uint64_t t = xt.ne(1);
  const TensorId xn_streams = b.Reshape(d.Packed(xn), S(w, hc, t));
  const TensorId gate_streams = b.Reshape(d.Packed(gate), S(w, hc, t));
  return b.HcPre(xn_streams, gate_streams,
                 HcParams{.fold = HcFold::kGated,
                          .streams = static_cast<std::uint32_t>(hc),
                          .scale = 1.0f / static_cast<float>(hc)});
}

// A [C, T] tensor's rows transposed into packed columns [T, C, 1], in F32.
TensorId Columns(D& d, TensorId rows) {
  Builder& b = d.b();
  const TensorDesc rt = d.t(rows);
  TensorId r = rows;
  if (rt.format.encoding != format::Encoding::kF32) {
    r = b.Convert(r, format::Encoding::kF32);
  }
  return b.Cont(b.Transpose(b.Reshape(d.Packed(r), S(rt.ne(0), rt.ne(1), 1))));
}

Outputs DecomposeRmsNormMul(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(2, 2)) {
    return {};
  }
  const auto p = Get<NormParams>(params);
  return One(b.Mul(b.RmsNorm(d.s(0), p.eps), d.s(1)));
}

Outputs DecomposeRmsNormMulRope(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(2, 4)) {
    return {};
  }
  const auto p = Get<NormRopeParams>(params);
  if (d.count() == 2) {
    return One(b.Rope(b.RmsNorm(d.s(0), p.eps), d.s(1), {}, p.rope));
  }
  if (!d.need(3, 4)) {
    return {};
  }
  return One(b.Rope(b.Mul(b.RmsNorm(d.s(0), p.eps), d.s(1)), d.s(2), d.s(3), p.rope));
}

Outputs DecomposeRmsNormMulAdd(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(3, 3)) {
    return {};
  }
  const auto p = Get<NormParams>(params);
  return One(b.Add(b.Mul(b.RmsNorm(d.s(0), p.eps), d.s(1)), d.s(2)));
}

Outputs DecomposeRopeStore(D& d, const ParamBytes& params) {
  Builder& b = d.b();
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

Outputs DecomposeQkPrep(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(5, 6)) {
    return {};
  }
  const auto p = Get<NormRopeParams>(params);
  const TensorId q = b.Rope(b.Mul(b.RmsNorm(d.s(0), p.eps), d.s(1)), d.s(4), d.s(5), p.rope);
  const TensorId k = b.Rope(b.Mul(b.RmsNorm(d.s(2), p.eps), d.s(3)), d.s(4), d.s(5), p.rope);
  return Many({q, k});
}

Outputs DecomposeNormGate(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(3, 3)) {
    return {};
  }
  const auto p = Get<NormGateParams>(params);
  if (p.gate != NormGateFn::kSigmoid && p.gate != NormGateFn::kSilu) {
    b.Refuse(GraphError::kParams, "unknown norm output gate");
    return {};
  }
  const TensorId n = b.Mul(b.RmsNorm(d.s(0), p.eps), d.s(1));
  return One(
      b.Mul(n, b.Unary(d.s(2), p.gate == NormGateFn::kSilu ? UnaryFn::kSilu : UnaryFn::kSigmoid)));
}

Outputs DecomposeEulerStep(D& d, const ParamBytes&) {
  Builder& b = d.b();
  if (!d.need(3, 3)) {
    return {};
  }
  return One(b.Add(d.s(0), b.Mul(d.s(1), d.s(2))));
}

Outputs DecomposeDupUpAdd(D& d, const ParamBytes& params) {
  return DupUpAdd(d, Get<DupUpParams>(params));
}

Outputs DecomposeSpatialChannelNorm(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(2, 2)) {
    return {};
  }
  const auto p = Get<SpatialNormParams>(params);
  const TensorDesc& x = d.t(d.s(0));
  const TensorDesc& gamma = d.t(d.s(1));
  if (!d.check(gamma.shape().ne == ShapeOf({x.ne(2)}).ne, GraphError::kShape,
               "a [W, H, C, N] input and a [C] gamma")) {
    return {};
  }
  const TensorId rows = b.Cont(b.Permute(d.s(0), {1, 2, 0, 3}));
  TensorId out = b.Mul(b.Scale(b.L2Norm(rows, p.eps), p.scale), d.s(1));
  if (p.silu) {
    out = b.Unary(out, UnaryFn::kSilu);
  }
  return One(b.Cont(b.Permute(out, {2, 0, 1, 3})));
}

Outputs DecomposeRmsNormMulRopeSections(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  const auto p = Get<RopeSectionsParams>(params);
  if (!d.need(2 + std::size_t{p.sections}, 2 + std::size_t{p.sections})) {
    return {};
  }
  const TensorDesc& x = d.t(d.s(0));
  std::uint64_t dims = 0;
  bool tail_zero = true;
  for (std::size_t i = 0; i < p.dims.size(); ++i) {
    if (i < p.sections) {
      dims += p.dims.at(i);
    } else if (p.dims.at(i) != 0) {
      tail_zero = false;
    }
  }
  if (!d.check(p.sections >= 1 && p.sections <= 3 && tail_zero &&
                   (p.mode == RopeMode::kNormal || p.mode == RopeMode::kNeox) && dims == x.ne(0) &&
                   d.t(d.s(1)).shape().ne == ShapeOf({x.ne(0)}).ne,
               GraphError::kParams, "one to three sections covering the head and a [D] weight")) {
    return {};
  }
  TensorId w = d.s(1);
  if (d.t(w).format.encoding != format::Encoding::kF32) {
    w = b.Convert(w, format::Encoding::kF32);
  }
  TensorId out = b.Mul(b.RmsNorm(d.s(0), p.eps), w);
  std::uint32_t offset = 0;
  for (std::size_t i = 0; i < p.sections; ++i) {
    out = b.Rope(out, d.s(2 + i), {},
                 {.mode = p.mode, .dims = p.dims.at(i), .offset = offset, .base = p.base});
    offset += p.dims.at(i);
  }
  return One(out);
}

Outputs DecomposeGatedResidualModulateSplit(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(4, 6) || !d.check(d.count() == 4 || d.count() == 6, GraphError::kArity,
                                "takes 4 (unsplit) or 6 (split) sources")) {
    return {};
  }
  const std::array f32{format::Encoding::kF32};
  if (d.count() == 4) {
    const std::array gated{d.s(0), d.s(1), d.s(2)};
    const TensorId x = b.FusedBytes(FusedKind::kGatedResidualSplit, gated, params, f32)[0];
    const std::array modulated{x, d.s(3)};
    return Many({x, b.FusedBytes(FusedKind::kLayerNormModulateSplit, modulated, params, f32)[0]});
  }
  const std::array gated{d.s(0), d.s(1), d.s(2), d.s(3)};
  const TensorId x = b.FusedBytes(FusedKind::kGatedResidualSplit, gated, params, f32)[0];
  const std::array modulated{x, d.s(4), d.s(5)};
  return Many({x, b.FusedBytes(FusedKind::kLayerNormModulateSplit, modulated, params, f32)[0]});
}

template <bool gated>
Outputs DecomposeSplitModulate(D& d, const ParamBytes& params) {
  Builder& b = d.b();

  if (!d.need(gated ? 3 : 2, gated ? 4 : 3)) return {};
  const auto p = Get<SplitModulateParams>(params);
  if (!d.check(std::isfinite(p.eps) && p.eps >= 0.0f, GraphError::kParams,
               "eps must be finite and not negative")) {
    return {};
  }
  const auto f32 = [&](TensorId t) {
    return d.t(t).format.encoding == format::Encoding::kF32 ? t
                                                            : b.Convert(t, format::Encoding::kF32);
  };
  TensorId side{};
  if (d.count() == (gated ? 3u : 2u)) {
    if (!d.check(p.split == 0, GraphError::kParams, "unsplit form requires split == 0")) {
      return {};
    }
    side = f32(d.s(gated ? 2 : 1));
  } else {
    const TensorDesc& x = d.t(d.s(0));
    const std::uint64_t cols = x.ne(0), rows = x.ne(1);
    const TensorId a = d.s(gated ? 2 : 1), bv = d.s(gated ? 3 : 2);
    if (!d.check(x.shape().rank <= 2 && p.split <= rows &&
                     d.t(a).shape().ne == ShapeOf({cols}).ne &&
                     d.t(bv).shape().ne == ShapeOf({cols}).ne &&
                     (!gated || d.t(d.s(1)).shape().ne == x.shape().ne),
                 GraphError::kShape, "rows [C, R] and two [C] vectors split within R"))
      return {};
    const auto rows_of = [&](TensorId v, std::uint64_t n) {
      return b.Repeat(b.Reshape(d.Packed(v), ShapeOf({cols, 1})), ShapeOf({cols, n}));
    };
    side = p.split == 0      ? rows_of(bv, rows)
           : p.split == rows ? rows_of(a, rows)
                             : b.Concat(rows_of(a, p.split), rows_of(bv, rows - p.split), 1);
  }
  if (gated) {
    return One(b.Add(f32(d.s(0)), b.Mul(f32(d.s(1)), b.Unary(side, UnaryFn::kTanh))));
  }
  return One(b.Mul(b.LayerNorm(d.s(0), p.eps), b.Scale(side, 1.0F, 1.0F)));
}

Outputs DecomposeAttentionMaskPack(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(1, 1)) return {};
  const auto p = Get<MaskPackParams>(params);
  const TensorDesc& m = d.t(d.s(0));
  const std::uint64_t cols = m.ne(0), rows = m.ne(1);
  if (!d.check(m.shape().rank <= 2 && p.cells >= cols && p.rows >= rows, GraphError::kShape,
               "a mask [C, R] padded to at least its extents"))
    return {};
  TensorId out = b.Convert(d.s(0), format::Encoding::kF16);
  const float inf = -std::numeric_limits<float>::infinity();
  if (p.cells > cols)
    out = b.Concat(out, b.Fill(ShapeOf({p.cells - cols, rows}), inf, format::Encoding::kF16), 0);
  if (p.rows > rows)
    out = b.Concat(out, b.Fill(ShapeOf({p.cells, p.rows - rows}), inf, format::Encoding::kF16), 1);
  return One(out);
}

Outputs DecomposeRopeTable(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  const auto p = Get<RopeSectionsParams>(params);
  if (!d.need(std::size_t{p.sections}, std::size_t{p.sections})) return {};
  std::uint64_t dims = 0;
  bool tail_zero = true;
  for (std::size_t i = 0; i < p.dims.size(); ++i) {
    if (i < p.sections) {
      dims += p.dims.at(i);
    } else if (p.dims.at(i) != 0) {
      tail_zero = false;
    }
  }
  const std::uint64_t rows = d.t(d.s(0)).ne(0);
  if (!d.check(p.eps == 0.0f && p.sections >= 1 && p.sections <= 3 && tail_zero &&
                   p.mode == RopeMode::kNormal && dims != 0 && dims % 2 == 0,
               GraphError::kParams, "one to three normal sections of an even head"))
    return {};
  const TensorId pairs = b.Concat(b.Fill(ShapeOf({1, dims / 2, rows}), 1.0F),
                                  b.Fill(ShapeOf({1, dims / 2, rows}), 0.0F), 0);
  TensorId out = b.Reshape(b.Cont(pairs), ShapeOf({dims, 1, rows}));
  std::uint32_t offset = 0;
  for (std::size_t i = 0; i < p.sections; ++i) {
    out =
        b.Rope(out, d.s(i), {},
               {.mode = RopeMode::kNormal, .dims = p.dims.at(i), .offset = offset, .base = p.base});
    offset += p.dims.at(i);
  }
  return One(b.Reshape(out, ShapeOf({2, dims / 2, rows})));
}

Outputs DecomposeQkAttention(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(6, 6)) return {};
  const auto p = Get<QkAttentionParams>(params);
  const TensorDesc& q = d.t(d.s(0));
  const TensorDesc& k = d.t(d.s(2));
  const TensorDesc& table = d.t(d.s(5));
  const std::uint64_t dim = 2 * table.ne(1), rows = q.ne(1);
  if (!d.check(dim != 0 && q.ne(0) % dim == 0 && k.ne(0) % dim == 0 && k.ne(1) == rows &&
                   d.t(d.s(3)).shape().ne == k.shape().ne && table.ne(0) == 2 &&
                   table.ne(2) == rows && (q.ne(0) / dim) % (k.ne(0) / dim) == 0 &&
                   d.t(d.s(4)).ne(0) >= rows && d.t(d.s(4)).ne(1) >= rows,
               GraphError::kShape, "q, k, v heads of the table's width over one set of rows"))
    return {};
  const std::uint64_t heads = q.ne(0) / dim, kv_heads = k.ne(0) / dim;
  TensorId w = d.s(1);
  if (d.t(w).format.encoding != format::Encoding::kF32) w = b.Convert(w, format::Encoding::kF32);
  TensorId qh =
      b.Mul(b.RmsNorm(b.Reshape(d.Packed(d.s(0)), ShapeOf({dim, heads, rows})), p.eps), w);
  qh = b.Rope(qh, d.s(5), {},
              {.mode = RopeMode::kNormal, .table = true, .dims = static_cast<std::uint32_t>(dim)});
  const TensorId qa = b.Cont(b.Permute(qh, {0, 2, 1, 3}));
  const TensorId ka =
      b.Cont(b.Permute(b.Reshape(d.Packed(d.s(2)), ShapeOf({dim, kv_heads, rows})), {0, 2, 1, 3}));
  const TensorId va =
      b.Permute(b.Reshape(d.Packed(d.s(3)), ShapeOf({dim, kv_heads, rows})), {0, 2, 1, 3});
  const TensorId masks = d.Packed(d.s(4));
  const TensorDesc packed = d.t(masks);
  const std::uint64_t unit = UnitBytes(packed.format);
  const TensorId mask = b.View(
      masks, ShapeOf({rows, rows}),
      {unit, unit * packed.ne(0), unit * packed.ne(0) * rows, unit * packed.ne(0) * rows}, 0);
  const TensorId out = b.Attention(qa, ka, va, mask, {}, {.scale = p.scale});
  return One(b.Reshape(out, ShapeOf({dim * heads, rows})));
}

Outputs DecomposeMatMulGlu(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(3, 3)) {
    return {};
  }
  const auto p = Get<GluParams>(params);
  const TensorId gate = b.MatMul(d.s(0), d.s(2));
  const TensorId up = b.MatMul(d.s(1), d.s(2));
  return One(b.Glu(gate, up, p));
}

Outputs DecomposeMatMulAdd(D& d, const ParamBytes&) {
  Builder& b = d.b();
  if (!d.need(3, 3)) {
    return {};
  }
  return One(b.Add(b.MatMul(d.s(0), d.s(1)), d.s(2)));
}

Outputs DecomposeGroupedMatMul(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(2, 2)) return {};
  const auto& w = d.t(d.s(0));
  const auto& x = d.t(d.s(1));
  if (!d.check(w.ne(0) == x.ne(0) && w.ne(2) == x.ne(1) && w.ne(3) == 1 && x.ne(3) == 1,
               GraphError::kShape, "grouped product requires [K,N,G] and [K,G,T]"))
    return {};
  const auto product =
      b.MatMul(d.s(0), b.Permute(d.s(1), {0, 2, 1, 3}), Get<ProductParams>(params));
  return One(b.Cont(b.Permute(product, {0, 2, 1, 3})));
}

Outputs DecomposeInverseRopeGroupedMatMul(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(3, 3)) return {};
  const auto p = Get<RopeParams>(params);
  const TensorDesc w = d.t(d.s(0));
  const TensorDesc x = d.t(d.s(1));
  const std::uint64_t per = w.ne(0);
  const std::uint64_t total_features = x.ne(0) * x.ne(1);
  const bool stacked = w.shape().rank == 3;
  const std::uint64_t groups = per > 0 ? total_features / per : 0;
  if (!d.check(per > 0 && groups > 0 && total_features % per == 0 && x.ne(3) == 1 && w.ne(3) == 1 &&
                   (stacked ? w.ne(2) == groups : w.ne(2) == 1 && w.ne(1) % groups == 0),
               GraphError::kShape,
               "a weight of groups of the rows' features, each `per` wide, as many groups "
               "as the weight holds"))
    return {};
  const std::uint64_t o_rank = stacked ? w.ne(1) : w.ne(1) / groups;
  const std::uint64_t rows = x.ne(2);
  TensorId rotated = b.Rope(d.s(1), d.s(2), {}, p);
  TensorId out = b.Permute(b.Reshape(rotated, S(per, groups, rows)), {0, 2, 1, 3});
  TensorId g = stacked ? d.s(0) : b.Reshape(d.Packed(d.s(0)), S(per, o_rank, groups));
  TensorId low = b.Permute(b.MatMul(g, out), {0, 2, 1, 3});
  return One(b.Cont(low, S(o_rank * groups, rows)));
}

Outputs DecomposePreparedProducts(D& d, const ParamBytes&) {
  Builder& b = d.b();
  if (!d.need(2, 5)) {
    return {};
  }
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

Outputs DecomposeMoeGlu(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(5, 5)) {
    return {};
  }
  const auto p = Get<GluParams>(params);
  const TensorId gate = b.Mul(d.s(0), ScaleRows(d, d.s(3), d.s(2)));
  const TensorId up = b.Mul(d.s(1), ScaleRows(d, d.s(4), d.s(2)));
  return One(b.Glu(gate, up, p));
}

Outputs DecomposeMoeCombine(D& d, const ParamBytes&) {
  Builder& b = d.b();
  if (!d.need(4, 6)) {
    return {};
  }
  if (d.s(4).valid() && !d.check(d.t(d.s(4)).ne(0) == 1 && d.t(d.s(4)).ne(1) == d.t(d.s(3)).ne(1),
                                 GraphError::kShape, "the shared expert's gate is [1, T]")) {
    return {};
  }
  const TensorId routed = b.ExpertReduce(d.s(0), d.s(2), d.s(1), d.s(5));
  const TensorId shared =
      d.s(4).valid() ? b.Mul(d.s(3), b.Unary(d.s(4), UnaryFn::kSigmoid)) : d.s(3);
  return One(b.Add(routed, shared));
}

Outputs DecomposeMoeRouter(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(3, 3)) {
    return {};
  }
  const auto p = Get<RouteParams>(params);
  const Outputs routed = b.Route(d.s(0), {}, {}, {}, p);
  return Many({routed[0], routed[1], b.SumRows(b.Mul(d.s(1), d.s(2)))});
}

Outputs DecomposeMoeExperts(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(7, 10)) {
    return {};
  }
  const auto p = Get<MoeExpertsParams>(params);
  const TensorDesc xt = d.t(d.s(2));
  const TensorDesc it = d.t(d.s(3));
  const std::uint64_t f = p.hidden;
  const std::uint64_t u = it.ne(0);
  const std::uint64_t t = xt.ne(1);
  const TensorId pair = b.MatMulId(d.s(0), b.Reshape(d.Packed(d.s(2)), S(xt.ne(0), 1, t)), d.s(3));
  if (!b.ok() || !d.check(d.t(pair).ne(0) == 2 * f, GraphError::kShape,
                          "a stacked gate and up of twice the hidden width")) {
    return {};
  }
  const TensorDesc pt = d.t(pair);
  const Nb nb{pt.nb[0], pt.nb[1], pt.nb[2], pt.nb[2] * t};
  TensorId gate = b.View(pair, S(f, u, t), nb, 0);
  TensorId up = b.View(pair, S(f, u, t), nb, f * pt.nb[0]);
  if (d.s(7).valid()) {
    gate = b.Mul(gate, ScaleRows(d, d.s(7), d.s(3)));
  }
  if (d.s(8).valid()) {
    up = b.Mul(up, ScaleRows(d, d.s(8), d.s(3)));
  }
  const TensorId down = b.MatMulId(d.s(1), b.Glu(gate, up, p.glu), d.s(3));
  const TensorId routed = b.ExpertReduce(down, d.s(4), d.s(3), d.s(9));
  return One(b.Add(routed, b.Mul(d.s(5), b.Unary(d.s(6), UnaryFn::kSigmoid))));
}

Outputs DecomposeMatMulIdGlu(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(4, 4)) {
    return {};
  }
  const auto p = Get<MoeExpertsParams>(params);
  const TensorDesc xt = d.t(d.s(2));
  const TensorId x = b.Reshape(d.Packed(d.s(2)), S(xt.ne(0), 1, xt.ne(1)));
  const TensorId gate = b.MatMulId(d.s(0), x, d.s(3));
  const TensorId up = b.MatMulId(d.s(1), x, d.s(3));
  return One(b.Glu(gate, up, p.glu));
}

Outputs DecomposeOwnerAttention(D& d, const ParamBytes& params) {
  Builder& b = d.b();
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
    const TensorId a = b.Attention(qo, k, v, mo, {}, p.attn);
    out = out.valid() ? b.Concat(out, a, 3) : a;
  }
  return One(out);
}

Outputs DecomposeStateBankAttention(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(3, 5)) return {};
  const auto p = Get<AttnParams>(params);
  const TensorId joined = b.Concat(d.s(1), d.s(2), 2);
  const TensorId keys = b.Permute(joined, {0, 2, 1, 3});
  const TensorId values = b.Permute(joined, {0, 2, 1, 3});
  return One(b.Attention(d.s(0), keys, values, d.s(3), d.s(4), p));
}

Outputs DecomposeRoundedIndexMask(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(5, 5)) return {};
  const auto p = Get<RoundedIndexMaskParams>(params);
  if (!d.check(p.cells == d.t(d.s(1)).ne(2), GraphError::kShape,
               "the selected mask and rounded score have the same compressed cells"))
    return {};
  const auto scores = b.RoundedIndexScore(d.s(0), d.s(1), d.s(2), d.s(3));
  const auto selected = b.VisibleTopK(scores, d.s(3), p.top);
  return One(b.SelectedMask(d.s(4), selected, {.raw_cells = p.raw_cells, .cells = p.cells}));
}

Outputs DecomposeSoftmaxPoolCompress(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(5, 5)) return {};
  const auto p = Get<CompressorParams>(params);
  const TensorDesc sk = d.t(d.s(0));
  const TensorDesc ss = d.t(d.s(1));
  const TensorDesc kv = d.t(d.s(2));
  const TensorDesc sc = d.t(d.s(3));
  const TensorDesc ids = d.t(d.s(4));
  const auto matrix = [](const TensorDesc& t) {
    return t.format.encoding == format::Encoding::kF32 && t.ne(2) == 1 && t.ne(3) == 1;
  };
  if (!d.check(p.ratio > 0 && p.ratio <= std::numeric_limits<std::uint32_t>::max() / 2,
               GraphError::kParams, "a bounded positive compressor ratio") ||
      !d.check(matrix(sk) && matrix(ss) && matrix(kv) && matrix(sc), GraphError::kShape,
               "compressor values and scores are F32 matrices") ||
      !d.check(sk.shape().ne == ss.shape().ne && kv.shape().ne == sc.shape().ne &&
                   sk.ne(0) == kv.ne(0) && (!p.overlap || sk.ne(0) % 2 == 0),
               GraphError::kShape, "matching compressor channels, states and current rows") ||
      !d.check(ids.format.encoding == format::Encoding::kI32 && ids.ne(1) == 1 && ids.ne(2) == 1 &&
                   ids.ne(3) == 1,
               GraphError::kShape, "compressor read indices are an I32 vector")) {
    return {};
  }
  const std::uint64_t halves = p.overlap ? 2 : 1;
  const std::uint64_t per_block = halves * p.ratio;
  if (!d.check(ids.ne(0) % per_block == 0, GraphError::kShape,
               "one complete compressor index run per block"))
    return {};
  const std::uint64_t blocks = ids.ne(0) / per_block;
  const std::uint64_t dim = sk.ne(0) / halves;
  const std::uint64_t reads = std::uint64_t{p.ratio} * blocks;
  const auto rows = [&](std::size_t state, std::size_t current, bool scores) {
    TensorId source = b.Concat(d.s(state), d.s(current), 1);
    if (!b.ok()) return TensorId{};
    if (p.overlap) {
      const TensorDesc sd = d.t(source);
      const TensorId zero =
          b.Fill(ShapeOf({sd.ne(0), 1}), scores ? -std::numeric_limits<float>::infinity() : 0.0F,
                 format::Encoding::kF32);
      source = b.Concat(source, zero, 1);
    }
    const TensorId picked = b.GetRows(source, d.s(4));
    if (!b.ok()) return TensorId{};
    if (!p.overlap) return b.Reshape(picked, ShapeOf({dim, p.ratio, blocks}));
    const TensorDesc pd = d.t(picked);
    const auto half = [&](std::uint64_t offset) {
      const TensorId v = d.View2(picked, dim, reads, pd.nb[0], pd.nb[1], offset);
      return b.Reshape(b.Cont(v), ShapeOf({dim, p.ratio, blocks}));
    };
    const TensorId previous = half(0);
    const TensorId current_half = half(reads * pd.nb[1] + dim * pd.nb[0]);
    return b.Concat(previous, current_half, 1);
  };
  const TensorId values = rows(0, 2, false);
  const TensorId scores = rows(1, 3, true);
  return One(b.Pool(values, scores,
                    {.ratio = static_cast<std::uint32_t>(per_block), .kind = PoolKind::kSoftmax}));
}

Outputs DecomposeHeadGate(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(2, 2)) {
    return {};
  }
  const auto p = Get<HeadGateParams>(params);
  const TensorDesc at = d.t(d.s(0));
  const TensorDesc qt = d.t(d.s(1));
  const std::uint64_t dh = p.head;
  if (!d.check(dh != 0 && at.ne(0) % dh == 0 && qt.ne(0) == 2 * at.ne(0) && qt.ne(1) == at.ne(1),
               GraphError::kShape, "a gate of each head of the attention")) {
    return {};
  }
  const std::uint64_t heads = at.ne(0) / dh;
  const std::uint64_t t = at.ne(1);
  const std::uint64_t e = qt.nb[0];
  const TensorId gate =
      b.Cont(b.View(d.s(1), S(dh, heads, t), Nb{e, 2 * dh * e, qt.nb[1], qt.nb[1] * t}, dh * e),
             S(dh * heads, t));
  return One(b.Mul(d.s(0), b.Unary(gate, UnaryFn::kSigmoid)));
}

Outputs DecomposeMeanPoolNormRopeStore(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(5, 6)) {
    return {};
  }
  const auto p = Get<BlockPoolParams>(params);
  const TensorDesc raw = d.t(d.s(0));
  const TensorDesc bp = d.t(d.s(3));
  const std::uint64_t dim = raw.ne(0);
  const std::uint64_t blocks = bp.ne(0) / 4;
  if (!d.check(p.ratio != 0 && raw.ne(1) == blocks * p.ratio && d.t(d.s(4)).ne(0) == blocks,
               GraphError::kShape, "every block of the raw rows, a position and a cell each")) {
    return {};
  }
  TensorId keys = b.Pool(d.s(0), {}, PoolParams{.ratio = p.ratio, .kind = PoolKind::kMean});
  keys = b.Mul(b.RmsNorm(keys, p.eps), d.s(2));
  keys = b.Rope(b.Reshape(keys, S(dim, 1, blocks)), d.s(3), {}, p.rope);
  keys = b.Reshape(b.Convert(keys, format::Encoding::kBF16), S(dim, blocks));
  return One(b.CacheStore(d.s(1), keys, d.s(4)));
}

Outputs DecomposeBlockSparseAttention(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(8, 9)) {
    return {};
  }
  const auto p = Get<BlockSparseAttnParams>(params);
  const TensorDesc q = d.t(d.s(0));
  const TensorDesc k = d.t(d.s(1));
  const TensorDesc qi = d.t(d.s(3));
  const TensorDesc bl = d.t(d.s(4));
  const std::uint64_t t = q.ne(2);
  const std::uint64_t cells = k.ne(1);
  const std::uint64_t blocks = bl.ne(1);
  if (!d.check(p.ratio != 0 && cells == blocks * p.ratio && p.width != 0 && p.width <= cells &&
                   qi.ne(2) == t,
               GraphError::kShape, "a selection over every block of the attended cells")) {
    return {};
  }
  const TensorId keys = b.Reshape(d.Packed(d.s(4)), S(bl.ne(0), 1, blocks, 1));
  const TensorId bias = b.Scale(b.Clamp(d.s(6), -1e9f, 0.0f), -1.0f);
  const TensorId ones = b.Fill(S(qi.ne(1), t, 1, 1), 1.0f);
  const TensorId queries = b.Convert(d.s(3), format::Encoding::kBF16);
  TensorId score = b.IndexScore(queries, keys, ones, b.Reshape(bias, S(blocks, t, 1, 1)));
  score = b.Repeat(b.Reshape(score, S(1, blocks, t)), S(p.ratio, blocks, t));
  score = b.Add(b.Reshape(score, S(cells, t)), d.s(5));
  const TensorId kept = b.TopK(score, p.width);
  TensorId selected = b.Fill(S(1, cells, t, 1), -std::numeric_limits<float>::infinity());
  const TensorId zeros = b.Fill(S(1, p.width, t, 1), 0.0f);
  selected = b.CacheStore(selected, zeros, kept);
  const TensorDesc mt = d.t(d.s(7));
  selected = b.Reshape(selected, S(cells, t));
  const TensorId mask = b.Add(
      selected, b.View(d.s(7), S(cells, t), Nb{mt.nb[0], mt.nb[1], mt.nb[1] * t, mt.nb[1] * t}, 0));
  const TensorId a = b.Attention(b.Permute(d.s(0), {0, 2, 1, 3}), d.s(1), d.s(2), mask, {}, p.attn);
  return One(b.Reshape(a, S(q.ne(0) * q.ne(1), t)));
}

Outputs DecomposeHcNorm(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(2, 2)) {
    return {};
  }
  const auto p = Get<NormParams>(params);
  const TensorDesc& x = d.t(d.s(0));
  const TensorId n = b.Reshape(b.RmsNorm(d.s(0), p.eps), S(x.ne(0) * x.ne(1), x.ne(2)));
  return One(b.Mul(n, d.s(1)));
}

Outputs DecomposeHcNormPost(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(3, 4)) {
    return {};
  }
  const auto p = Get<HcNormPostParams>(params);
  const TensorId post = b.HcPost(d.s(0), d.s(1), d.s(2), d.s(3), HcParams{.streams = p.streams});
  if (!b.ok()) {
    return {};
  }
  const TensorDesc& pt = d.t(post);
  return Many({post, b.RmsNorm(b.Reshape(post, S(pt.ne(0) * pt.ne(1), pt.ne(2))), p.eps)});
}

Outputs DecomposeSinkhornHcPre(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(5, 5)) return {};
  const auto p = Get<SinkhornPreParams>(params);
  const TensorId x = d.s(0);
  const TensorDesc xd = d.t(x);
  const std::uint64_t width = xd.ne(0);
  const std::uint64_t hc = p.streams;
  const std::uint64_t tokens = xd.ne(2);
  const auto shape = [&](std::size_t at, const Shape& wanted) {
    const TensorDesc td = d.t(d.s(at));
    return td.shape().ne == wanted.ne && td.format.encoding == format::Encoding::kF32;
  };
  if (!d.check(hc >= 2 && hc <= 64 && p.iterations >= 1 && p.iterations <= 1024,
               GraphError::kParams, "bounded Sinkhorn streams and iterations") ||
      !d.check(shape(0, S(width, hc, tokens)) && shape(2, S(3)) && shape(3, S((2 + hc) * hc)) &&
                   shape(4, S(width)) && d.t(d.s(1)).shape().ne == S(width * hc, (2 + hc) * hc).ne,
               GraphError::kShape, "Sinkhorn streams, mixing weights, scales, bases and norm"))
    return {};
  const TensorId flat = b.Reshape(d.Packed(x), S(width * hc, tokens));
  const TensorId mixes = b.MatMul(d.s(1), b.RmsNorm(flat, p.eps));
  if (!b.ok()) return {};
  const TensorDesc md = d.t(mixes);
  const auto run = [&](std::uint64_t first) {
    return d.View2(mixes, hc, tokens, md.nb[0], md.nb[1], first * md.nb[0]);
  };
  TensorId post = b.Mul(run(hc), Element(d, d.s(2), 1));
  post = b.Scale(b.Unary(b.Add(post, Range(d, d.s(3), hc, hc)), UnaryFn::kSigmoid), 2.0F);
  const TensorId comb = b.HcMix(mixes, d.s(2), d.s(3),
                                {.mix = HcMixKind::kSinkhorn,
                                 .streams = p.streams,
                                 .iterations = p.iterations,
                                 .eps = p.hc_eps});
  TensorId pre = b.Mul(run(0), Element(d, d.s(2), 0));
  pre = b.Scale(b.Unary(b.Add(pre, Range(d, d.s(3), 0, hc)), UnaryFn::kSigmoid), 1.0F, p.hc_eps);
  const TensorId folded = b.HcPre(x, pre, {.fold = HcFold::kWeights, .streams = p.streams});
  return Many({b.Mul(b.RmsNorm(folded, p.eps), d.s(4)), post, comb});
}

Outputs DecomposeHcPrep(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(2, 5)) {
    return {};
  }
  const auto p = Get<HcNormPostParams>(params);
  if (!d.check(p.streams != 0 && d.t(d.s(0)).ne(1) == p.streams && d.t(d.s(0)).ne(3) == 1,
               GraphError::kShape, "a tensor of its declared streams"))
    return {};
  const bool combine = d.s(3).valid();
  if (!d.check(combine == d.s(4).valid(), GraphError::kArity,
               "the last block's output and its logits together")) {
    return {};
  }
  const TensorId streams = combine ? HcCombineOf(d, d.s(0), d.s(3), d.s(4), p.streams) : d.s(0);
  const TensorId xn = HcNormOf(d, streams, d.s(1), p.eps);
  Outputs o;
  o.ids.at(o.n++) = xn;
  if (d.s(2).valid()) {
    o.ids.at(o.n++) = b.MatMul(d.s(2), xn);
  }
  if (combine) {
    o.ids.at(o.n++) = streams;
  }
  return o;
}

Outputs DecomposeScaledUnary(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(1, 1)) {
    return {};
  }
  const auto p = Get<ScaledUnaryParams>(params);
  return One(b.Unary(b.Scale(d.s(0), p.scale), p.fn));
}

Outputs DecomposeHcFold(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(2, 3)) {
    return {};
  }
  const auto p = Get<HcFoldParams>(params);
  TensorId xn = d.s(0);
  TensorId gate = d.s(1);
  if (d.count() == 3) {
    if (!d.check(p.streams != 0 && d.t(d.s(0)).ne(1) == p.streams && d.t(d.s(0)).ne(3) == 1,
                 GraphError::kShape, "a tensor of its declared streams"))
      return {};
    xn = HcNormOf(d, d.s(0), d.s(1), p.eps);
    gate = d.s(2);
  }
  const TensorId mixed = HcFoldOf(d, xn, gate, p.streams);
  Outputs o = One(mixed);
  if (p.prepared && b.ok()) {
    o.ids.at(o.n++) = b.Prepare(mixed, format::Encoding::kMxfp8, format::Layout::kScaleAtoms);
  }
  if (p.rounded && b.ok()) {
    o.ids.at(o.n++) = b.Convert(mixed, format::Encoding::kBF16);
  }
  return o;
}

Outputs DecomposeHcCombine(D& d, const ParamBytes& params) {
  if (!d.need(3, 3)) {
    return {};
  }
  const auto p = Get<HcParams>(params);
  if (!d.check(static_cast<std::uint8_t>(p.mix) <= 1 && static_cast<std::uint8_t>(p.fold) <= 1 &&
                   std::isfinite(p.eps) && std::isfinite(p.scale),
               GraphError::kParams, "known kinds, finite eps and scale")) {
    return {};
  }
  return One(HcCombineOf(d, d.s(0), d.s(1), d.s(2), p.streams));
}

Outputs DecomposeGdnConv(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(3, 3)) {
    return {};
  }
  const auto p = Get<GdnConvParams>(params);
  if (!d.check(p.eps > 0.0f && std::isfinite(p.eps) && std::isfinite(p.scale), GraphError::kParams,
               "positive finite eps and finite scale")) {
    return {};
  }
  const TensorDesc rt = d.t(d.s(0));
  const TensorDesc wt = d.t(d.s(2));
  const std::uint64_t c = rt.ne(0);
  const std::uint64_t t = rt.ne(1);
  if (!d.check(p.head != 0 && p.qk_channels % p.head == 0 && p.qk_channels <= c && wt.ne(1) == c &&
                   wt.ne(0) >= 2,
               GraphError::kShape, "a convolution of whole heads over its channels")) {
    return {};
  }
  const std::uint64_t taps = wt.ne(0) - 1;
  const TensorId history = b.Reshape(d.Packed(d.s(1)), S(taps, c, 1));
  const TensorId input = b.Concat(history, Columns(d, d.s(0)), 0);
  const TensorId conv = b.Unary(b.SsmConv(input, d.s(2)), UnaryFn::kSilu);
  if (!b.ok()) {
    return {};
  }
  if (p.qk_channels == 0) {
    return One(b.Reshape(conv, S(c, t)));
  }
  const std::uint64_t e = 4;
  const std::uint64_t heads = p.qk_channels / p.head;
  TensorId qk = b.Cont(b.View(conv, S(p.head, heads, t), Nb{e, p.head * e, c * e, c * t * e}, 0));
  qk = b.Reshape(b.Scale(b.RmsNorm(qk, p.eps), p.scale), S(p.qk_channels, t));
  if (p.qk_channels == c) {
    return One(qk);
  }
  const std::uint64_t rest = c - p.qk_channels;
  const TensorId tail =
      b.Cont(b.View(conv, S(rest, t), Nb{e, c * e, c * t * e, c * t * e}, p.qk_channels * e));
  return One(b.Concat(qk, tail, 0));
}

Outputs DecomposeGdnHistory(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  if (!d.need(1, 2)) {
    return {};
  }
  const auto p = Get<GdnHistoryParams>(params);
  const TensorDesc rt = d.t(d.s(0));
  const std::uint64_t c = rt.ne(0);
  const std::uint64_t t = rt.ne(1);
  if (!d.check(p.taps != 0 && (d.s(1).valid() || t >= p.taps), GraphError::kShape,
               "a history of rows, or of the old history and the rows")) {
    return {};
  }
  TensorId input = Columns(d, d.s(0));
  if (d.s(1).valid()) {
    input = b.Concat(b.Reshape(d.Packed(d.s(1)), S(p.taps, c, 1)), input, 0);
  }
  if (!b.ok()) {
    return {};
  }
  const TensorDesc it = d.t(input);
  const TensorId tail = b.View(input, S(p.taps, c, 1), Nb{it.nb[0], it.nb[1], it.nb[2], it.nb[2]},
                               (it.ne(0) - p.taps) * it.nb[0]);
  return One(b.Reshape(b.Cont(tail), S(p.taps * c, 1)));
}

Outputs DecomposeGdnStep(D& d, const ParamBytes& params) {
  Builder& b = d.b();
  const auto p = Get<GdnStepParams>(params);
  const std::size_t n = p.write ? 7 : 6;
  if (!d.need(n, n)) {
    return {};
  }
  const TensorDesc q = d.t(d.s(0));
  const TensorDesc v = d.t(d.s(2));
  const TensorDesc st = d.t(d.s(5));
  if (!d.check(q.ne(0) * v.ne(0) * v.ne(1) == st.ne(0) * st.ne(1) * st.ne(2) * st.ne(3),
               GraphError::kShape, "a state of every value head's matrix")) {
    return {};
  }
  const TensorId state = b.Reshape(d.Packed(d.s(5)), S(q.ne(0), v.ne(0), v.ne(1), 1));
  const Outputs ran = b.GatedDeltaNet(d.s(0), d.s(1), d.s(2), d.s(3), d.s(4), state);
  if (!p.write || !b.ok()) {
    return One(ran[0]);
  }
  const TensorId rows = b.Reshape(ran[1], S(st.ne(0), st.ne(1), st.ne(2), st.ne(3)));
  return Many({ran[0], b.CacheStore(d.s(5), rows, d.s(6))});
}

Outputs DecomposeGdnGates(D& d, const ParamBytes&) {
  Builder& b = d.b();
  if (!d.need(4, 4)) {
    return {};
  }
  const TensorDesc at = d.t(d.s(0));
  const std::uint64_t h = at.ne(0);
  const std::uint64_t t = at.ne(1);
  TensorId gate = b.Reshape(d.Packed(d.s(0)), S(h, t, 1));
  gate = b.Mul(b.Unary(b.Add(gate, d.s(2)), UnaryFn::kSoftplus), d.s(3));
  gate = b.Reshape(gate, S(1, h, t, 1));
  const TensorId beta = b.Unary(b.Reshape(d.Packed(d.s(1)), S(1, h, t, 1)), UnaryFn::kSigmoid);
  return Many({gate, beta});
}

// Attention's ordinary fallback keeps scale/softcap before the original
// additive mask, including finite bias masks. Window/sparse fields only name
// cells already excluded by that mask; the dense formula visits all of them.
Outputs Attention(D& d, const AttnParams& p) {
  if (!d.need(3, 5)) return {};
  auto& b = d.b();
  const auto f32 = [&](TensorId id) {
    return d.t(id).format.encoding == format::Encoding::kF32
               ? id
               : b.Convert(id, format::Encoding::kF32);
  };
  const auto q = f32(d.s(0)), k = f32(d.s(1)), v = f32(d.s(2));
  auto scores = b.MatMul(k, q, ProductParams{.accumulate = p.accumulate});
  float scale = p.scale;
  if (p.softcap != 0.0f) {
    scores = b.Softcap(b.Scale(scores, scale), p.softcap);
    scale = 1.0f;
  }
  const auto mask = d.s(3).valid() ? f32(d.s(3)) : TensorId{};
  const auto probabilities =
      b.SoftMax(scores, mask, d.s(4), SoftMaxParams{.scale = scale, .max_bias = p.max_bias});
  const auto product =
      b.MatMul(b.Cont(b.Transpose(v)), probabilities, ProductParams{.accumulate = p.accumulate});
  return One(b.Cont(b.Permute(product, {0, 2, 1, 3})));
}

bool Composite(Op op) { return HasDecomposition(Primitive(op)); }

std::expected<Graph, VerifyError> ExpandOnce(const Graph& graph, bool composites,
                                             std::vector<TensorId>& map, StorageLoan& map_loan,
                                             bool& more) {
  Builder b;
  const auto funder = graph.storage_funder();
  if (funder && !b.SetStorageFunder(funder)) return std::unexpected(std::move(*b.TakeError()));
  const std::uint64_t map_capacity = std::max<std::uint64_t>(graph.tensor_count(), map.capacity());
  auto map_bytes = VectorScratchBytes(map_capacity, sizeof(TensorId));
  if (!map_bytes) {
    return std::unexpected(VerifyError{.error = GraphError::kAdmission, .detail = {}});
  }
  if (funder && !map_loan.GrowTo(funder, *map_bytes)) {
    return std::unexpected(VerifyError{.error = GraphError::kAdmission, .detail = {}});
  }
  map.assign(graph.tensor_count(), TensorId{});
  more = false;
  std::array<TensorId, kMaxSources> sources{};
  std::array<TensorDesc, kMaxOutputs> requested{};
  std::array<format::Encoding, kMaxOutputs> encodings{};
  for (std::size_t n = 0; n < graph.nodes().size(); ++n) {
    const Node& node = graph.nodes()[n];
    b.SetComponent(node.component);
    if (node.batch == Batch::kPerOwner && node.nout > 0) {
      b.SetOwner(graph.tensor(node.out[0]).segment);
    } else {
      b.SetJoined();
    }
    std::size_t source_count = 0;
    for (const TensorId s : node.sources()) {
      sources.at(source_count++) = s.valid() ? map.at(s.v) : TensorId{};
    }
    Outputs outs;
    const bool split = node.key.op == Op::kFused || (composites && Composite(node.key.op));
    if (split) {
      std::size_t encoding_count = 0;
      for (const TensorId o : node.outputs()) {
        encodings.at(encoding_count++) = graph.tensor(o).format.encoding;
      }
      outs = Decompose(b, node.key, node.params, std::span(sources).first(source_count),
                       std::span(encodings).first(encoding_count));
    } else {
      std::size_t requested_count = 0;
      for (const TensorId o : node.outputs()) {
        requested.at(requested_count++) = graph.tensor(o);
      }
      if (node.key == Primitive(Op::kView) && source_count > 0 && sources[0].valid() &&
          !SameViewSource(graph.tensor(node.src[0]), b.tensor(sources[0]))) {
        b.Refuse(GraphError::kFormat,
                 "a view's source changed layout in its expansion: the view would read "
                 "other elements");
      } else {
        outs = b.Emit(node.key, node.params, std::span(sources).first(source_count),
                      std::span(requested).first(requested_count),
                      node.nout > 0 ? graph.name(node.out[0]) : std::string_view());
      }
    }
    if (!b.ok()) {
      auto taken_error = b.TakeError();
      VerifyError e = std::move(*taken_error);
      e.node = static_cast<std::uint32_t>(n);
      return std::unexpected(std::move(e));
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

Outputs DecomposePrimitiveAttention(D& d, const ParamBytes& params) {
  return Attention(d, Get<AttnParams>(params));
}
Outputs DecomposePrimitiveHcPre(D& d, const ParamBytes& params) {
  return HcPre(d, Get<HcParams>(params));
}
Outputs DecomposePrimitiveHcPost(D& d, const ParamBytes& params) {
  return HcPost(d, Get<HcParams>(params));
}
Outputs DecomposePrimitiveHcMix(D& d, const ParamBytes& params) {
  return HcMix(d, Get<HcParams>(params));
}
Outputs DecomposePrimitiveIndexScore(D& d, const ParamBytes&) { return IndexScore(d); }
Outputs DecomposePrimitiveRoundedIndexScore(D& d, const ParamBytes& params) {
  return RoundedIndexScore(d, Get<RoundedIndexScoreParams>(params));
}
Outputs DecomposePrimitiveVisibleTopK(D& d, const ParamBytes& params) {
  return VisibleTopK(d, Get<TopKParams>(params));
}
Outputs DecomposePrimitiveSelectedMask(D& d, const ParamBytes& params) {
  return SelectedMask(d, Get<SelectedMaskParams>(params));
}
Outputs DecomposePrimitivePool(D& d, const ParamBytes& params) {
  return Pool(d, Get<PoolParams>(params));
}
Outputs DecomposePrimitiveRoute(D& d, const ParamBytes& params) {
  return Route(d, Get<RouteParams>(params));
}
Outputs DecomposePrimitiveExpertReduce(D& d, const ParamBytes&) { return ExpertReduce(d); }
Outputs DecomposePrimitiveSsmConv(D& d, const ParamBytes&) { return SsmConv(d); }
Outputs DecomposePrimitiveGatedDeltaNet(D& d, const ParamBytes&) { return GatedDeltaNet(d); }

using DecomposeHandler = Outputs (*)(D&, const ParamBytes&);
constexpr DecomposeHandler NoDecomposition = nullptr;
#define LLMP_DISPATCH(op, name, params, required, max, infer, validate, layout, partition, \
                      decomposes, handler, ...)                                            \
  static_assert((DecomposeHandler{handler} != nullptr) == decomposes);
#define LLMP_PRIMITIVE(...) LLMP_DISPATCH(__VA_ARGS__)
#define LLMP_FUSED(...) LLMP_DISPATCH(__VA_ARGS__)
#include "ir/op_traits.inc"
#undef LLMP_PRIMITIVE
#undef LLMP_FUSED
#undef LLMP_DISPATCH
#define LLMP_DISPATCH(op, name, params, required, max, infer, validate, layout, partition, \
                      decomposes, handler, ...)                                            \
  handler,
#define LLMP_PRIMITIVE(...) LLMP_DISPATCH(__VA_ARGS__)
#define LLMP_FUSED(...)
constexpr auto kPrimitiveDecompositions = std::to_array<DecomposeHandler>({
#include "ir/op_traits.inc"
});
#undef LLMP_PRIMITIVE
#undef LLMP_FUSED
#define LLMP_PRIMITIVE(...)
#define LLMP_FUSED(...) LLMP_DISPATCH(__VA_ARGS__)
constexpr auto kFusedDecompositions = std::to_array<DecomposeHandler>({
#include "ir/op_traits.inc"
});
#undef LLMP_PRIMITIVE
#undef LLMP_FUSED
#undef LLMP_DISPATCH
static_assert(kPrimitiveDecompositions.size() == kOpCount);
static_assert(kFusedDecompositions.size() == kFusedKindCount);
DecomposeHandler DecompositionOf(OpKey key) {
  if (!Traits(key)) return nullptr;
  return key.op == Op::kFused ? kFusedDecompositions[static_cast<std::size_t>(key.fused)]
                              : kPrimitiveDecompositions[static_cast<std::size_t>(key.op)];
}

}  // namespace

bool HasDecomposition(OpKey key) { return DecompositionOf(key) != nullptr; }

std::optional<std::size_t> AliasedSource(OpKey key, std::size_t output) {
  const OpTraits* traits = Traits(key);
  if (!traits || output >= traits->aliased_outputs.size() || traits->aliased_outputs[output] < 0)
    return std::nullopt;
  return static_cast<std::size_t>(traits->aliased_outputs[output]);
}

Outputs Decompose(Builder& b, OpKey key, const ParamBytes& params,
                  std::span<const TensorId> sources, std::span<const format::Encoding> encodings) {
  D d(b, sources);
  const auto handler = DecompositionOf(key);
  if (!handler) {
    b.Refuse(GraphError::kParams, DiagFormat("{} has no decomposition", key));
    return {};
  }
  Outputs o = handler(d, params);
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
    } else if (encodings[i] == format::Encoding::kMxfp8 && have == format::Encoding::kF32) {
      // An output recorded prepared (graph.h PreparedRows).
      o.ids.at(i) = b.Prepare(o.ids.at(i), format::Encoding::kMxfp8, format::Layout::kScaleAtoms);
    }
  }
  return o;
}

OwnedTensorMap::OwnedTensorMap(OwnedTensorMap&& other) noexcept
    : loan_(std::move(other.loan_)), values_(std::move(other.values_)) {}

void OwnedTensorMap::Swap(OwnedTensorMap& other) noexcept {
  std::swap(loan_, other.loan_);
  values_.swap(other.values_);
}

OwnedTensorMap& OwnedTensorMap::operator=(OwnedTensorMap&& other) noexcept {
  if (this != &other) {
    OwnedTensorMap incoming(std::move(other));
    Swap(incoming);
  }
  return *this;
}

namespace {

std::expected<Graph, VerifyError> ExpandImpl(const Graph& graph, bool composites,
                                             std::vector<TensorId>& result_map,
                                             StorageLoan& result_map_loan) {
  StorageLoan total_loan;
  std::vector<TensorId> total;
  StorageLoan step_loan;
  std::vector<TensorId> step;
  bool more = false;
  auto g = ExpandOnce(graph, composites, total, total_loan, more);
  // A decomposition may emit fused or composite ops (kMoeCombine's expert
  // reduction, kHcNormPost's spread): expand again until none is left, a
  // bounded number of times (the deepest nesting is a few levels).
  constexpr int kMaxPasses = 16;
  for (int pass = 1; g && more; ++pass) {
    if (pass == kMaxPasses) {
      return std::unexpected(VerifyError{
          .error = GraphError::kBounds,
          .node = 0,
          .detail = std::format("fused nodes remain after {} expansion passes", kMaxPasses)});
    }
    auto next = ExpandOnce(*g, composites, step, step_loan, more);
    if (!next) {
      return next;
    }
    for (TensorId& id : total) {
      id = id.valid() ? step.at(id.v) : id;
    }
    g = std::move(next);
  }
  if (g) {
    result_map = std::move(total);
    result_map_loan = std::move(total_loan);
  }
  return g;
}

}  // namespace

std::expected<Graph, VerifyError> Expand(const Graph& graph, bool composites) {
  StorageLoan map_loan;
  std::vector<TensorId> map;
  return ExpandImpl(graph, composites, map, map_loan);
}

std::expected<ExpansionResult, VerifyError> ExpandWithMap(const Graph& graph, bool composites) {
  OwnedTensorMap map;
  auto expanded = ExpandImpl(graph, composites, map.values_, map.loan_);
  if (!expanded) return std::unexpected(std::move(expanded.error()));
  return ExpansionResult{std::move(*expanded), std::move(map)};
}

}  // namespace llmp::ir
