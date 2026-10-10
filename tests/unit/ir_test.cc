// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The graph IR (D-107): the builder's shape rules, the verifier's
// refusals, the canonical digest and the decompositions' structure. The
// decompositions' values are checked in reference_exec_test.cc.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
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

namespace llmp::ir {
namespace {

using format::Encoding;
using format::Layout;

// The digest of DigestIsStableAcrossBuilds' graph.
constexpr std::string_view kGoldenDigest =
    "750dfb38849e3a682066a492272fd35ecbaef503f212a3d575e27a34b46d4ca7";

format::TensorFormat F32(std::initializer_list<std::uint64_t> ne) {
  return Dense(Encoding::kF32, ShapeOf(ne));
}
format::TensorFormat I32(std::initializer_list<std::uint64_t> ne) {
  return Dense(Encoding::kI32, ShapeOf(ne));
}

// The builder's error, which must be set.
GraphError ErrorOf(const Builder& b) {
  EXPECT_FALSE(b.ok());
  return b.ok() ? GraphError::kShape : b.error()->error;
}

// A small decoder layer in the shape of Gemma's: embedding rows scaled,
// RMS norm and weight, Q/K/V products, NEOX rotation, a cache store,
// masked attention with a softcap, the output product, a residual, a
// GeGLU feed-forward and a tied head with a final softcap.
struct Layer {
  TensorId tokens, positions, cells, store_at, embed, norm, wq, wk, wv, wo, wg, wu, wd, cache_k,
      cache_v, logits;
};

Layer BuildLayer(Builder& b, bool reorder_weights = false, std::string_view prefix = "") {
  constexpr std::uint64_t kWidth = 64;
  constexpr std::uint64_t kHead = 16;
  constexpr std::uint64_t kHeads = 4;
  constexpr std::uint64_t kKv = 2;
  constexpr std::uint64_t kRows = 3;
  constexpr std::uint64_t kCells = 8;
  constexpr std::uint64_t kVocab = 32;
  constexpr std::uint64_t kFfn = 96;
  Layer l;
  const auto w = [&](std::uint16_t role, std::initializer_list<std::uint64_t> ne) {
    return b.Weight(Dense(Encoding::kF16, ShapeOf(ne)), WeightRef{.component = 1, .role = role},
                    std::string(prefix) + std::to_string(role));
  };
  l.tokens = b.Input(I32({kRows}), "tokens");
  l.positions = b.Input(I32({kRows}), "positions");
  l.cells = b.Input(I32({kCells}), "cells");
  l.store_at = b.Input(Dense(Encoding::kI64, ShapeOf({kRows})), "store_at");
  if (reorder_weights) {
    l.wo = w(5, {kHeads * kHead, kWidth});
    l.wk = w(3, {kWidth, kKv * kHead});
    l.wq = w(2, {kWidth, kHeads * kHead});
  }
  l.embed = w(0, {kWidth, kVocab});
  l.norm = b.Weight(F32({kWidth}), WeightRef{.component = 1, .role = 1});
  if (!reorder_weights) {
    l.wq = w(2, {kWidth, kHeads * kHead});
    l.wk = w(3, {kWidth, kKv * kHead});
  }
  l.wv = w(4, {kWidth, kKv * kHead});
  if (!reorder_weights) {
    l.wo = w(5, {kHeads * kHead, kWidth});
  }
  l.wg = w(6, {kWidth, kFfn});
  l.wu = w(7, {kWidth, kFfn});
  l.wd = w(8, {kFfn, kWidth});
  l.cache_k = b.State(Dense(Encoding::kF16, ShapeOf({kKv * kHead, kCells})), StateRef{.tensor = 0});
  l.cache_v = b.State(Dense(Encoding::kF16, ShapeOf({kKv * kHead, kCells})), StateRef{.tensor = 1});

  TensorId x = b.Scale(b.GetRows(l.embed, l.tokens), 8.0f);
  const TensorId n = b.Mul(b.RmsNorm(x, 1e-6f), l.norm);
  const RopeParams rope{.mode = RopeMode::kNeox, .dims = kHead, .base = 10000.0f};
  const TensorId q =
      b.Rope(b.Reshape(b.MatMul(l.wq, n), ShapeOf({kHead, kHeads, kRows})), l.positions, {}, rope);
  const TensorId k =
      b.Rope(b.Reshape(b.MatMul(l.wk, n), ShapeOf({kHead, kKv, kRows})), l.positions, {}, rope);
  const TensorId kc =
      b.CacheStore(l.cache_k, b.Reshape(k, ShapeOf({kKv * kHead, kRows})), l.store_at);
  const TensorId vc = b.CacheStore(l.cache_v, b.MatMul(l.wv, n), l.store_at);
  const TensorId mask = b.MaskBuild(l.positions, l.cells, kRows, MaskParams{});
  const auto heads = [&](TensorId c) {
    return b.Permute(b.Reshape(c, ShapeOf({kHead, kKv, kCells})), {0, 2, 1, 3});
  };
  const TensorId attn = b.Attention(b.Permute(q, {0, 2, 1, 3}), heads(kc), heads(vc), mask, {},
                                    AttnParams{.scale = 0.25f, .softcap = 50.0f});
  x = b.Add(x, b.MatMul(l.wo, b.Reshape(b.Cont(attn), ShapeOf({kHeads * kHead, kRows}))));
  const TensorId ff =
      b.Glu(b.MatMul(l.wg, x), b.MatMul(l.wu, x), GluParams{.kind = GluKind::kGeGluTanh});
  x = b.Add(x, b.MatMul(l.wd, ff));
  l.logits = b.Softcap(b.MatMul(l.embed, x), 30.0f);
  b.Output(l.logits);
  return l;
}

TEST(IrTest, BuildsAndVerifiesALayer) {
  Builder b;
  const Layer l = BuildLayer(b);
  ASSERT_TRUE(b.ok()) << b.error()->detail;
  EXPECT_EQ(b.tensor(l.logits).shape().ne, (std::array<std::uint64_t, 4>{32, 3, 1, 1}));
  auto g = std::move(b).Finish();
  ASSERT_TRUE(g.has_value()) << g.error().detail;
  EXPECT_EQ(g->inputs().size(), 4U);
  EXPECT_EQ(g->outputs().size(), 1U);
  EXPECT_EQ(g->tensor(l.logits).role, Role::kOutput);
  EXPECT_EQ(g->name(l.tokens), "tokens");
  // The cache stores' results are views of the caches.
  std::size_t stores = 0;
  for (const Node& n : g->nodes()) {
    if (n.key.op == Op::kCacheStore) {
      EXPECT_EQ(g->root(n.out[0]), n.src[0]);
      ++stores;
    }
  }
  EXPECT_EQ(stores, 2U);
  EXPECT_TRUE(Verify(*g).has_value());
}

TEST(IrTest, InfersShapesAsGgmlDoes) {
  Builder b;
  const TensorId q = b.Input(F32({64, 5, 8, 1}));
  const TensorId k = b.Input(F32({64, 40, 2, 1}));
  const TensorId v = b.Input(F32({32, 40, 2, 1}));
  EXPECT_EQ(b.tensor(b.Attention(q, k, v, {}, {}, AttnParams{})).shape().ne,
            (std::array<std::uint64_t, 4>{32, 8, 5, 1}));
  const TensorId as = b.Weight(F32({16, 24, 6}), WeightRef{.role = 1});
  const TensorId x = b.Input(F32({16, 1, 7}));
  const TensorId ids = b.Input(I32({2, 7}));
  EXPECT_EQ(b.tensor(b.MatMulId(as, x, ids)).shape().ne,
            (std::array<std::uint64_t, 4>{24, 2, 7, 1}));
  const TensorId table = b.Weight(F32({10, 50}), WeightRef{.role = 2});
  const TensorId rows = b.Input(I32({4}));
  EXPECT_EQ(b.tensor(b.GetRows(table, rows)).shape().ne,
            (std::array<std::uint64_t, 4>{10, 4, 1, 1}));
  const TensorId p = b.Permute(b.Input(F32({2, 3, 4, 5})), {2, 0, 3, 1});
  EXPECT_EQ(b.tensor(p).shape().ne, (std::array<std::uint64_t, 4>{3, 5, 2, 4}));
  EXPECT_FALSE(IsContiguous(b.tensor(p)));
  const TensorId sx = b.Input(F32({3 + 6, 10, 1}));
  const TensorId taps = b.Weight(F32({4, 10}), WeightRef{.role = 3});
  EXPECT_EQ(b.tensor(b.SsmConv(sx, taps)).shape().ne, (std::array<std::uint64_t, 4>{10, 6, 1, 1}));
  const Outputs route = b.Route(b.Input(F32({8, 3})), {}, {}, {},
                                RouteParams{.normalize = true, .experts = 8, .used = 2});
  ASSERT_EQ(route.n, 2);
  EXPECT_EQ(b.tensor(route[0]).format.encoding, Encoding::kI32);
  EXPECT_EQ(b.tensor(route[1]).shape().ne, (std::array<std::uint64_t, 4>{2, 3, 1, 1}));
  // A result may be recorded as BF16; a cast changes the encoding.
  const TensorId half = b.Convert(b.Input(F32({4})), Encoding::kF16);
  EXPECT_EQ(b.tensor(half).format.encoding, Encoding::kF16);
  EXPECT_EQ(b.tensor(b.RmsNorm(half, 1e-6f)).format.encoding, Encoding::kF16);
  ASSERT_TRUE(b.ok()) << b.error()->detail;
  EXPECT_TRUE(std::move(b).Finish().has_value());
}

TEST(IrTest, ViewsOfBlockLayoutsAreRowsOfWholeBlocks) {
  Builder b;
  const format::TensorFormat q8{
      .encoding = Encoding::kQ8_0, .layout = Layout::kGgmlBlocks, .shape = ShapeOf({64, 6, 4})};
  const TensorId w = b.Weight(q8, WeightRef{.role = 1, .expert_array = true});
  const auto nb = b.tensor(w).nb;  // a copy: the builder's tensors move as it grows
  EXPECT_EQ(nb[0], 34U);           // block_q8_0
  EXPECT_EQ(nb[1], 68U);
  // Expert 2 as a matrix.
  const TensorId e2 = b.View(w, ShapeOf({64, 6}), {nb[0], nb[1], nb[2], nb[2]}, 2 * nb[2]);
  EXPECT_EQ(b.tensor(e2).view_offset, 2 * nb[2]);
  EXPECT_TRUE(b.ok());
  // Past the root.
  b.View(w, ShapeOf({64, 6}), {nb[0], nb[1], nb[2], nb[2]}, 4 * nb[2]);
  EXPECT_EQ(ErrorOf(b), GraphError::kBounds);
}

TEST(IrTest, RefusesWhatOpSetV1DoesNotDefine) {
  {
    Builder b;
    b.MatMul(b.Input(F32({8, 4})), b.Input(F32({6, 3})));
    EXPECT_EQ(ErrorOf(b), GraphError::kShape);
    EXPECT_NE(b.error()->detail.find("mul_mat"), std::string::npos) << b.error()->detail;
  }
  {
    Builder b;
    b.MatMul(b.Input(I32({8, 4})), b.Input(F32({8, 3})));
    EXPECT_EQ(ErrorOf(b), GraphError::kFormat);
  }
  {
    Builder b;
    const TensorId x = b.Input(F32({8, 2, 3}));
    b.Rope(x, b.Input(I32({3})), {}, RopeParams{.dims = 3});
    EXPECT_EQ(ErrorOf(b), GraphError::kParams);
  }
  {
    Builder b;
    const TensorId x = b.Input(F32({8, 2, 3}));
    b.Rope(x, b.Input(I32({4})), {}, RopeParams{.dims = 8});
    EXPECT_EQ(ErrorOf(b), GraphError::kShape);
  }
  {
    // A reserved parameter byte set.
    Builder b;
    const TensorId x = b.Input(F32({8}));
    GluParams p{};
    p.reserved[0] = 1;
    b.Glu(x, {}, p);
    EXPECT_EQ(ErrorOf(b), GraphError::kParams);
  }
  {
    // Sinks the parameters do not announce, and ALiBi without a mask.
    Builder b;
    const TensorId q = b.Input(F32({8, 2, 2}));
    const TensorId k = b.Input(F32({8, 4, 1}));
    b.Attention(q, k, k, {}, {}, AttnParams{.max_bias = 8.0f});
    EXPECT_EQ(ErrorOf(b), GraphError::kArity);
  }
  {
    // A sparse bound without the mask, and the sparse gather without one.
    Builder b;
    const TensorId q = b.Input(F32({8, 2, 2}));
    const TensorId k = b.Input(F32({8, 4, 1}));
    b.Attention(q, k, k, {}, {}, AttnParams{.n_kv_max = 2});
    EXPECT_EQ(ErrorOf(b), GraphError::kArity);
    Builder c;
    const TensorId q2 = c.Input(F32({8, 2, 2}));
    const TensorId k2 = c.Input(F32({8, 4, 1}));
    c.Attention(q2, k2, k2, c.Input(F32({4, 2})), {}, AttnParams{.sparse_any = true});
    EXPECT_EQ(ErrorOf(c), GraphError::kParams);
  }
  {
    // A Hadamard hint on a matrix that is not square, and an unknown
    // accumulation.
    Builder b;
    b.MatMul(b.Input(F32({8, 4})), b.Input(F32({8, 3})),
             ProductParams{.hint = ProductHint::kHadamard});
    EXPECT_EQ(ErrorOf(b), GraphError::kParams);
    Builder c;
    ParamBytes raw{};
    raw[0] = std::byte{2};
    c.Emit(Primitive(Op::kMatMul), raw, std::array{c.Input(F32({8, 4})), c.Input(F32({8, 3}))});
    EXPECT_EQ(ErrorOf(c), GraphError::kParams);
  }
  {
    // A routing scale of 0: an unset scale is written 1.
    Builder b;
    b.Route(b.Input(F32({8, 2})), {}, {}, {}, RouteParams{.experts = 8, .used = 2, .scale = 0.0f});
    EXPECT_EQ(ErrorOf(b), GraphError::kParams);
  }
  {
    // The MoE combine's expert scales [E] where its optional shared gate
    // [1, T] goes: E divides the width, so a product would broadcast.
    Builder b;
    const TensorId down = b.Input(F32({8, 2, 3}));
    const TensorId ids = b.Input(I32({2, 3}));
    const TensorId weights = b.Input(F32({2, 3}));
    const TensorId shared = b.Input(F32({8, 3}));
    b.Fused(FusedKind::kMoeCombine, std::array{down, ids, weights, shared, b.Input(F32({4}))});
    EXPECT_EQ(ErrorOf(b), GraphError::kShape);
  }
  {
    // A source another builder made.
    Builder other;
    const TensorId foreign = other.Input(F32({4}));
    Builder b;
    b.Unary(TensorId{foreign.v + 5}, UnaryFn::kExp);
    EXPECT_EQ(ErrorOf(b), GraphError::kCycle);
  }
  {
    // A route kind's sources.
    Builder b;
    b.Route(b.Input(F32({8, 2})), {}, {}, {},
            RouteParams{.kind = RouteKind::kSqrtSoftplusBias, .experts = 8, .used = 2});
    EXPECT_EQ(ErrorOf(b), GraphError::kArity);
  }
  {
    // Reshaping a strided view.
    Builder b;
    const TensorId t = b.Transpose(b.Input(F32({4, 3})));
    b.Reshape(t, ShapeOf({12}));
    EXPECT_EQ(ErrorOf(b), GraphError::kShape);
  }
  {
    // A view of a format stored in parts.
    Builder b;
    const format::TensorFormat split{
        .encoding = Encoding::kMxfp8, .layout = Layout::kSplit, .shape = ShapeOf({64, 4})};
    const TensorId w = b.Weight(split, WeightRef{.role = 1});
    b.View(w, ShapeOf({64, 2}), {1, 64, 128, 128}, 0);
    EXPECT_EQ(ErrorOf(b), GraphError::kFormat);
  }
}

TEST(IrTest, VerifyRefusesBindingsAndSegments) {
  {
    Builder b;
    b.Weight(F32({4}), WeightRef{.role = 3});
    b.Weight(F32({4}), WeightRef{.role = 3});
    const auto g = std::move(b).Finish();
    ASSERT_FALSE(g.has_value());
    EXPECT_EQ(g.error().error, GraphError::kBinding);
    EXPECT_EQ(g.error().node, 1U);
  }
  {
    // A per-owner node reads only its owner's state.
    Builder b;
    const TensorId s0 = b.State(F32({4, 8}), StateRef{.tensor = 0, .segment = 0});
    const TensorId s1 = b.State(F32({4, 8}), StateRef{.tensor = 0, .segment = 1});
    b.SetOwner(0);
    b.Add(s0, s0);
    b.Add(s0, s1);
    const auto g = std::move(b).Finish();
    ASSERT_FALSE(g.has_value());
    EXPECT_EQ(g.error().error, GraphError::kBinding);
    EXPECT_EQ(g.error().node, 3U);
  }
  {
    Builder b;
    const TensorId s0 = b.State(F32({4, 8}), StateRef{.tensor = 0, .segment = 2});
    const TensorId joined = b.Input(F32({4, 8}));
    b.SetOwner(2);
    const TensorId y = b.Scale(s0, 2.0f);
    EXPECT_EQ(b.tensor(y).segment, 2);
    // Its weights, and views of joined rows, stay joined.
    const TensorId w = b.Weight(F32({4}), WeightRef{.role = 1});
    const TensorId rows = b.View(joined, ShapeOf({4, 2}), {4, 16, 32, 32}, 0);
    EXPECT_EQ(b.tensor(rows).segment, kJoinedSegment);
    EXPECT_EQ(b.tensor(b.Mul(rows, w)).segment, 2);
    b.SetJoined();
    EXPECT_EQ(b.tensor(b.Add(y, y)).segment, kJoinedSegment);
    EXPECT_TRUE(std::move(b).Finish().has_value());
  }
}

TEST(IrTest, VerifyHoldsGgmlsInt32Limits) {
  // RE-037: [n_kv, rows] F32 permuted puts its plane stride (2^31 bytes
  // here) in a dimension GGML takes as int.
  Builder b;
  const TensorId scores = b.Input(F32({65536, 8192}));
  const TensorId p = b.Permute(scores, {0, 2, 1, 3});
  EXPECT_EQ(b.tensor(p).nb[1], 65536ULL * 8192 * 4);
  auto g = std::move(b).Finish();
  ASSERT_FALSE(g.has_value());
  EXPECT_EQ(g.error().error, GraphError::kLimit);
  EXPECT_EQ(g.error().node, 1U);

  // The same past the rank: a transpose's source's plane stride (GGML's
  // permute holds all four), and flash attention's mask plane at 262,144
  // cells and 4,096 rows (nb32 is int32).
  {
    Builder t;
    t.Transpose(t.Input(F32({65536, 8192})));
    auto r = std::move(t).Finish();
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().error, GraphError::kLimit);
  }
  {
    Builder a;
    const TensorId q = a.Input(F32({8, 4096}));
    const TensorId k = a.Input(F32({8, 262144}));
    const TensorId mask = a.Input(Dense(Encoding::kF16, ShapeOf({262144, 4096})));
    a.Attention(q, k, k, mask, {}, AttnParams{});
    auto r = std::move(a).Finish();
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().error, GraphError::kLimit);
    EXPECT_EQ(r.error().node, 3U);
  }

  Builder wide;
  wide.Input(F32({std::uint64_t{1} << 31}));
  auto w = std::move(wide).Finish();
  ASSERT_FALSE(w.has_value());
  EXPECT_EQ(w.error().error, GraphError::kLimit);

  // Within the limits a large weight's last stride may exceed it: it is past
  // the rank.
  Builder ok;
  ok.Weight(Dense(Encoding::kBF16, ShapeOf({4096, 262144})), WeightRef{.role = 1});
  EXPECT_TRUE(std::move(ok).Finish().has_value());
}

base::Sha256Digest DigestOf(bool reorder, std::string_view prefix) {
  Builder b;
  BuildLayer(b, reorder, prefix);
  auto g = std::move(b).Finish();
  EXPECT_TRUE(g.has_value());
  return g ? g->Digest() : base::Sha256Digest{};
}

TEST(IrTest, DigestIsCanonical) {
  const base::Sha256Digest d = DigestOf(false, "");
  // Built again, with weights declared in another order, with other names.
  EXPECT_EQ(DigestOf(false, ""), d);
  EXPECT_EQ(DigestOf(true, ""), d);
  EXPECT_EQ(DigestOf(false, "renamed."), d);

  // Anything the step computes changes it: a parameter, a format, an
  // operand, the outputs.
  const auto variant = [](int which) {
    Builder b;
    const TensorId x = b.Input(F32({8, 2}));
    const TensorId w =
        b.Weight(F32({8, 4}), WeightRef{.role = which == 2 ? std::uint16_t{9} : std::uint16_t{1}});
    const TensorId y = b.MatMul(w, b.RmsNorm(x, which == 1 ? 1e-5f : 1e-6f));
    const TensorId z = which == 3 ? b.Scale(y, 2.0f) : b.Scale(y, 2.0f, which == 4 ? 1.0f : 0.0f);
    if (which != 5) {
      b.Output(z);
    }
    auto g = std::move(b).Finish();
    EXPECT_TRUE(g.has_value());
    return g->Digest();
  };
  const base::Sha256Digest base = variant(0);
  EXPECT_EQ(variant(3), base);  // the same node, written another way
  EXPECT_NE(variant(1), base);
  EXPECT_NE(variant(2), base);
  EXPECT_NE(variant(4), base);
  EXPECT_NE(variant(5), base);
}

TEST(IrTest, DigestHashesAccumulationAndHints) {
  // Production's per-node choices are part of the graph a lowering
  // reproduces: two graphs that differ only in them digest apart.
  const auto digest = [](const ProductParams& product, const AttnParams& attn) {
    Builder b;
    const TensorId x = b.Input(F32({8, 4}));
    const TensorId w = b.Weight(Dense(Encoding::kF16, ShapeOf({8, 8})), WeightRef{.role = 1});
    const TensorId y = b.MatMul(w, x, product);
    const TensorId q = b.Reshape(y, ShapeOf({8, 4, 1}));
    b.Output(b.Attention(q, q, q, b.Input(F32({4, 4})), {}, attn));
    auto g = std::move(b).Finish();
    EXPECT_TRUE(g.has_value()) << (g ? "" : g.error().detail);
    return g ? base::ToHex(g->Digest()) : std::string();
  };
  const std::string plain = digest({}, {});
  EXPECT_NE(digest({.accumulate = Accumulate::kF32}, {}), plain);
  EXPECT_NE(digest({.hint = ProductHint::kHadamard}, {}), plain);
  EXPECT_NE(digest({}, {.accumulate = Accumulate::kF32}), plain);
  EXPECT_NE(digest({}, {.n_kv_max = 2}), plain);
  EXPECT_NE(digest({}, {.sparse_any = true, .n_kv_max = 2}), digest({}, {.n_kv_max = 2}));
}

TEST(IrTest, DigestIsStableAcrossBuilds) {
  // The canonical form's bytes are part of plan keys: a change to them is
  // a reviewed change (bump the graph digest's version string).
  Builder b;
  const TensorId x = b.Input(F32({8, 2}), "x");
  const TensorId w = b.Weight(Dense(Encoding::kF16, ShapeOf({8, 4})), WeightRef{.role = 1});
  b.Output(b.Unary(b.MatMul(w, x), UnaryFn::kSilu));
  auto g = std::move(b).Finish();
  ASSERT_TRUE(g.has_value());
  EXPECT_EQ(base::ToHex(g->Digest()), kGoldenDigest);
}

TEST(IrTest, ExpandLeavesOnlyPrimitives) {
  Builder b;
  const TensorId x = b.Input(F32({16, 4, 3}));  // [width, streams, tokens]
  const TensorId norm = b.Weight(F32({64}), WeightRef{.role = 1});
  const Outputs post =
      b.Fused(FusedKind::kHcNormPost, std::array{b.Input(F32({16, 3})), x, b.Input(F32({4, 3}))},
              HcNormPostParams{.eps = 1e-6f, .streams = 4});
  ASSERT_EQ(post.n, 2);
  const TensorId h = b.Fused(FusedKind::kHcNorm, std::array{post[0], norm}, NormParams{1e-6f},
                             std::array{Encoding::kBF16})[0];
  EXPECT_EQ(b.tensor(h).format.encoding, Encoding::kBF16);
  b.Output(h);
  b.Output(post[1]);
  auto g = std::move(b).Finish();
  ASSERT_TRUE(g.has_value()) << g.error().detail;

  std::vector<TensorId> map;
  auto fused_only = Expand(*g, false, &map);
  ASSERT_TRUE(fused_only.has_value()) << fused_only.error().detail;
  bool composite = false;
  for (const Node& n : fused_only->nodes()) {
    EXPECT_NE(n.key.op, Op::kFused);
    composite = composite || n.key.op == Op::kHcPost;
  }
  EXPECT_TRUE(composite);
  EXPECT_EQ(fused_only->tensor(map.at(h.v)).format.encoding, Encoding::kBF16);
  EXPECT_EQ(fused_only->outputs().size(), 2U);

  auto all = Expand(*g, true);
  ASSERT_TRUE(all.has_value()) << all.error().detail;
  for (const Node& n : all->nodes()) {
    EXPECT_FALSE(HasDecomposition(n.key)) << OpKeyName(n.key);
  }
}

TEST(IrTest, EveryFusedKindAndCompositeDecomposes) {
  // Every fused kind but grouped_stores, a batch key that is never a node.
  for (std::uint16_t k = 1; k <= static_cast<std::uint16_t>(kLastFusedKind); ++k) {
    EXPECT_EQ(HasDecomposition(Fused(static_cast<FusedKind>(k))),
              static_cast<FusedKind>(k) != FusedKind::kGroupedStores)
        << k;
  }
  EXPECT_FALSE(HasDecomposition(Fused(FusedKind::kNone)));
  for (const Op op : {Op::kHcPre, Op::kHcPost, Op::kHcMix, Op::kIndexScore, Op::kPool, Op::kRoute,
                      Op::kExpertReduce, Op::kSsmConv, Op::kGatedDeltaNet}) {
    EXPECT_TRUE(HasDecomposition(Primitive(op))) << OpName(op);
  }
  EXPECT_FALSE(HasDecomposition(Primitive(Op::kMatMul)));
  EXPECT_EQ(AliasedSource(Fused(FusedKind::kRopeStore), 0), 2U);
  EXPECT_FALSE(AliasedSource(Fused(FusedKind::kGroupedStores), 1).has_value());
  EXPECT_FALSE(AliasedSource(Fused(FusedKind::kRmsNormMul), 0).has_value());
}

TEST(IrTest, FusedStoresAreViewsOfTheirCaches) {
  Builder b;
  const TensorId x = b.Input(F32({8, 2, 3}));
  const TensorId pos = b.Input(I32({3}));
  const TensorId cache = b.State(Dense(Encoding::kF16, ShapeOf({16, 10})), StateRef{.tensor = 4});
  const TensorId idx = b.Input(I32({3}));
  const TensorId stored = b.Fused(FusedKind::kRopeStore, std::array{x, pos, cache, idx},
                                  RopeParams{.mode = RopeMode::kNeox, .dims = 8})[0];
  ASSERT_TRUE(b.ok()) << b.error()->detail;
  EXPECT_EQ(b.tensor(stored).view_of, cache);
  EXPECT_EQ(b.tensor(stored).role, Role::kState);
  EXPECT_TRUE(std::move(b).Finish().has_value());
}

TEST(IrTest, HyperConnectionFusionsRefuseInvalidStreamShapes) {
  for (const std::uint32_t streams : {0U, 3U}) {
    Builder b;
    const TensorId xn = b.Input(F32({8, 2}));
    const TensorId gate = b.Input(F32({8, 2}));
    b.Fused(FusedKind::kHcFold, std::array{xn, gate}, HcFoldParams{.streams = streams});
    EXPECT_EQ(ErrorOf(b), GraphError::kShape);
  }
  for (FusedKind kind : {FusedKind::kHcPrep, FusedKind::kHcMixNorm}) {
    Builder b;
    const TensorId x = b.Input(F32({4, 2, 3}));
    const TensorId norm = b.Input(F32({8}));
    const TensorId gate = b.Input(F32({8, 3}));
    const std::vector<TensorId> src = kind == FusedKind::kHcPrep
                                          ? std::vector<TensorId>{x, norm}
                                          : std::vector<TensorId>{x, norm, gate};
    b.Fused(kind, src, HcNormPostParams{.eps = 1e-6F, .streams = 1});
    EXPECT_EQ(ErrorOf(b), GraphError::kShape);
  }
  Builder b;
  const TensorId x = b.Input(F32({4, 2, 3}));
  const TensorId out = b.Input(F32({4, 3}));
  const TensorId logits = b.Input(F32({1, 3}));
  b.Fused(FusedKind::kHcCombine, std::array{x, out, logits}, HcParams{.streams = 1});
  EXPECT_EQ(ErrorOf(b), GraphError::kShape);
}

TEST(IrTest, NewHyperConnectionAndGdnFusionsExpandStridedSources) {
  for (FusedKind kind :
       {FusedKind::kHcFold, FusedKind::kHcCombine, FusedKind::kGdnGates, FusedKind::kGdnHistory}) {
    Builder b;
    Outputs out;
    if (kind == FusedKind::kHcFold) {
      const auto xn = b.Transpose(b.Input(F32({2, 8})));
      const auto gate = b.Transpose(b.Input(F32({2, 8})));
      out = b.Fused(kind, std::array{xn, gate}, HcFoldParams{.streams = 2});
    } else if (kind == FusedKind::kHcCombine) {
      const auto streams = b.Input(F32({4, 2, 3}));
      const auto rows = b.Transpose(b.Input(F32({3, 4})));
      const auto logits = b.Input(F32({2, 3}));
      out = b.Fused(kind, std::array{streams, rows, logits}, HcParams{.streams = 2});
    } else if (kind == FusedKind::kGdnHistory) {
      const auto rows = b.Transpose(b.Input(F32({3, 4})));
      out = b.Fused(kind, std::array{rows}, GdnHistoryParams{.taps = 2});
    } else {
      const auto alpha = b.Transpose(b.Input(F32({3, 2})));
      const auto beta = b.Transpose(b.Input(F32({3, 2})));
      const auto dt = b.Input(F32({2}));
      const auto a = b.Input(F32({2}));
      out = b.Fused(kind, std::array{alpha, beta, dt, a});
    }
    for (const auto id : out.span()) b.Output(id);
    auto g = std::move(b).Finish();
    ASSERT_TRUE(g.has_value()) << g.error().detail;
    auto expanded = Expand(*g, false);
    ASSERT_TRUE(expanded.has_value()) << expanded.error().detail;
    EXPECT_TRUE(std::ranges::none_of(expanded->nodes(),
                                     [](const Node& n) { return n.key.op == Op::kFused; }));
  }
  // A nested helper must leave an invalid parameter as a builder error,
  // not try to describe the invalid tensor returned by the inner helper.
  Builder b;
  const auto streams = b.Input(F32({4, 2, 3}));
  const auto norm = b.Input(F32({8}));
  const auto gate = b.Input(F32({8, 3}));
  b.Fused(FusedKind::kHcMixNorm, std::array{streams, norm, gate},
          HcNormPostParams{.eps = -1.0F, .streams = 2});
  EXPECT_FALSE(b.ok());
}

// grouped_stores names a batch of store nodes (planner/pattern.h); no node
// is one.
TEST(IrTest, GroupedStoresIsNoNode) {
  Builder b;
  const TensorId cache = b.State(Dense(Encoding::kF16, ShapeOf({16, 10})), StateRef{.tensor = 4});
  const TensorId other = b.State(Dense(Encoding::kF16, ShapeOf({16, 10})), StateRef{.tensor = 5});
  const TensorId src = b.Input(F32({16, 3}));
  const TensorId idx = b.Input(I32({3}));
  b.Fused(FusedKind::kGroupedStores, std::array{cache, src, idx, other, src, idx});
  EXPECT_EQ(ErrorOf(b), GraphError::kParams);
}

TEST(IrTest, NothingStoresIntoAWeightOrAConstant) {
  {
    Builder b;
    const TensorId w = b.Weight(Dense(Encoding::kF16, ShapeOf({8, 10})), WeightRef{.role = 1});
    // Through a view of the weight, too: the root is what is written.
    const TensorId rows = b.View(w, ShapeOf({8, 5}), {2, 16, 80, 80}, 0);
    b.CacheStore(rows, b.Input(F32({8, 2})), b.Input(I32({2})));
    EXPECT_EQ(ErrorOf(b), GraphError::kBinding);
  }
  {
    Builder b;
    const TensorId table = b.Constant(Dense(Encoding::kF16, ShapeOf({16, 10})));
    b.Fused(FusedKind::kRopeStore,
            std::array{b.Input(F32({8, 2, 3})), b.Input(I32({3})), table, b.Input(I32({3}))},
            RopeParams{.mode = RopeMode::kNeox, .dims = 8});
    EXPECT_EQ(ErrorOf(b), GraphError::kBinding);
  }
}

}  // namespace
}  // namespace llmp::ir
