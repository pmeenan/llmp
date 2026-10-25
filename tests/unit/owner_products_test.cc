// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "planner/owner_products.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "components/step.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "format/encoding.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "planner/alternative.h"
#include "planner/plan.h"
#include "planner/policy.h"

namespace llmp::planner {
namespace {
using format::Encoding;
using ir::Op;
using ir::TensorId;

struct Fixture {
  ir::Graph graph;
  std::vector<components::StepInput> inputs;
  std::vector<components::SegmentRows> segments;
  TensorId product{};
  TensorId feature{};
  TensorId weight{};
  TensorId output{};
};

class TestFunder final : public ir::StorageFunder {
 public:
  bool Charge(std::uint64_t bytes) const override {
    ++charge_count_;
    if (charge_count_ == fail_at_) return false;
    active_ += bytes;
    peak_ = std::max(peak_, active_);
    return true;
  }
  void Release(std::uint64_t bytes) const override { active_ -= bytes; }
  void FailAt(std::uint64_t charge) { fail_at_ = charge; }
  std::uint64_t charge_count() const { return charge_count_; }
  std::uint64_t active() const { return active_; }
  std::uint64_t peak() const { return peak_; }

 private:
  mutable std::uint64_t active_ = 0;
  mutable std::uint64_t peak_ = 0;
  mutable std::uint64_t charge_count_ = 0;
  std::uint64_t fail_at_ = 0;
};

Fixture MakeManyOwners(std::span<const std::uint32_t> widths) {
  ir::Builder b;
  constexpr std::uint64_t w = 8, hidden = 4;
  std::uint32_t rows = 0;
  std::vector<components::SegmentRows> segments;
  for (std::size_t i = 0; i < widths.size(); ++i) {
    segments.push_back({.slot = static_cast<std::uint8_t>(5 - i),
                        .first_row = rows,
                        .rows = widths[i],
                        .outputs = 1});
    rows += widths[i];
  }
  const auto feature = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({w * hidden, rows})));
  const auto norm = b.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({w * hidden})), {.role = 2});
  auto activation =
      b.Reshape(b.Mul(b.RmsNorm(feature, 1e-6F), norm), ir::ShapeOf({w, hidden * rows}));
  const auto weight = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({w, w})), {.role = 3});
  const auto product = b.MatMul(weight, activation);
  const auto output = b.Scale(product, 1.0F);
  b.Output(output);
  auto graph = std::move(b).Finish();
  EXPECT_TRUE(graph) << test_support::Failed(graph, &ir::VerifyError::detail).value_or("");
  return {.graph = std::move(*graph),
          .inputs = {{.tensor = feature, .kind = components::InputKind::kFeatures}},
          .segments = std::move(segments),
          .product = product,
          .feature = feature,
          .weight = weight,
          .output = output};
}

enum class Variant { kHidden, kEmbedding, kCrossNorm, kUnknownBroadcast, kWrite, kReordered };
Fixture Make(std::uint32_t a, std::uint32_t c, Variant variant = Variant::kHidden) {
  ir::Builder b;
  constexpr std::uint64_t w = 8, hc = 4;
  const std::uint64_t rows = a + c;
  std::vector<components::StepInput> inputs;
  TensorId feature;
  TensorId h;
  if (variant == Variant::kEmbedding || variant == Variant::kReordered) {
    feature = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({rows})));
    inputs.push_back({.tensor = feature, .kind = components::InputKind::kTokens});
    auto ids = feature;
    if (variant == Variant::kReordered) {
      const auto order = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({rows})));
      ids =
          b.Reshape(b.GetRows(b.Reshape(ids, ir::ShapeOf({1, rows})), order), ir::ShapeOf({rows}));
    }
    const auto table = b.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({w, 32})), {.role = 1});
    h = b.GetRows(table, ids);
  } else {
    feature = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({w * hc, rows})));
    inputs.push_back({.tensor = feature, .kind = components::InputKind::kFeatures});
    h = feature;
  }
  if (variant == Variant::kCrossNorm) h = b.Reshape(h, ir::ShapeOf({w * hc * rows, 1}));
  h = b.RmsNorm(h, 1e-6F);
  const auto norm =
      variant == Variant::kUnknownBroadcast
          ? b.Input(ir::Dense(Encoding::kF32, b.tensor(h).shape()))
          : b.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({b.tensor(h).ne(0)})), {.role = 2});
  h = b.Mul(h, norm);
  if (variant != Variant::kEmbedding && variant != Variant::kReordered)
    h = b.Reshape(h, ir::ShapeOf({w, hc * rows}));
  const auto weight = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({w, w})), {.role = 3});
  const auto product = b.MatMul(weight, h);
  const auto output = b.Scale(product, 1.0F);
  if (variant == Variant::kWrite) {
    const auto values = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({w * hc, 1})));
    const auto idx = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({1})));
    b.CacheStore(feature, values, idx);
  }
  b.Output(output);
  auto graph = std::move(b).Finish();
  EXPECT_TRUE(graph) << test_support::Failed(graph, &ir::VerifyError::detail).value_or("");
  return {.graph = std::move(*graph),
          .inputs = std::move(inputs),
          .segments = {{.slot = 5, .first_row = 0, .rows = a, .outputs = 1},
                       {.slot = 2, .first_row = a, .rows = c, .outputs = 1}},
          .product = product,
          .feature = feature,
          .weight = weight,
          .output = output};
}

auto Prove(const Fixture& f) {
  return ProveOwnerProducts(f.graph, f.inputs, f.segments, std::array{f.product});
}

TEST(OwnerProductsTest, MultiOwnerSplitPaysExpandedNodesAndRefundsOnDenial) {
  for (const auto& widths :
       {std::vector<std::uint32_t>{1U, 1U}, std::vector<std::uint32_t>{1U, 1U, 1U}}) {
    SCOPED_TRACE(widths.size());
    auto fixture = MakeManyOwners(widths);
    const auto proof = Prove(fixture);
    ASSERT_TRUE(proof) << test_support::Failed(proof).value_or("");

    auto funded = std::make_shared<TestFunder>();
    auto source = fixture.graph.Clone(funded);
    ASSERT_TRUE(source) << test_support::Failed(source, &ir::VerifyError::detail).value_or("");
    const auto source_bytes = funded->active();
    {
      PlannerMetadataAdmission admission(funded);
      auto rewritten = SplitOwnerProducts(*source, *proof, &admission);
      ASSERT_TRUE(rewritten)
          << test_support::Failed(rewritten, &ir::VerifyError::detail).value_or("");
      const std::size_t emitted = 3 * widths.size() - 1;
      ASSERT_EQ(rewritten->rewrite.applied.size(), 1U);
      EXPECT_EQ(rewritten->rewrite.applied[0].made.size(), emitted);
      EXPECT_EQ(rewritten->rewrite.graph.nodes().size(),
                fixture.graph.nodes().size() + emitted - 1);
      EXPECT_EQ(rewritten->rewrite.from.size(), rewritten->rewrite.graph.nodes().size());
      EXPECT_EQ(rewritten->scopes.size(), widths.size());
      EXPECT_GT(admission.storage_bytes(), 0U);
      EXPECT_GT(funded->peak(), source_bytes);
    }
    EXPECT_EQ(funded->active(), source_bytes);

    auto refusing = std::make_shared<TestFunder>();
    auto refusing_source = fixture.graph.Clone(refusing);
    ASSERT_TRUE(refusing_source)
        << test_support::Failed(refusing_source, &ir::VerifyError::detail).value_or("");
    const auto refusing_source_bytes = refusing->active();
    {
      PlannerMetadataAdmission admission(refusing);
      ASSERT_FALSE(admission.refused());
      const auto refusal_charge = refusing->charge_count() + 1;
      refusing->FailAt(refusal_charge);
      const auto denied = SplitOwnerProducts(*refusing_source, *proof, &admission);
      ASSERT_FALSE(denied);
      EXPECT_EQ(denied.error().error, ir::GraphError::kAdmission);
      EXPECT_TRUE(admission.refused());
      EXPECT_EQ(refusing->charge_count(), refusal_charge);
    }
    EXPECT_EQ(refusing->active(), refusing_source_bytes);
  }
}

execution::Registry Registry() {
  std::vector<execution::Implementation> all;
  for (const auto op : {Op::kRmsNorm, Op::kMul, Op::kGetRows, Op::kMatMul, Op::kConcat,
                        Op::kConvert, Op::kScale, Op::kCacheStore}) {
    all.push_back({.name = ir::OpKeyName(ir::Primitive(op)),
                   .source = "fake",
                   .revision = "r1",
                   .build = "cpu",
                   .variant = "owner-products",
                   .capability = execution::Capability{
                       .op = ir::Primitive(op),
                       .product_arithmetic = +[](const execution::CapabilityQuery& q) {
                         const auto w = std::ranges::find(q.operands, std::uint8_t{0},
                                                          &execution::OperandFormat::operand);
                         return w != q.operands.end() && w->encoding == Encoding::kBF16 &&
                                        q.rows > 16
                                    ? ir::ProductArithmetic::kBf16F32
                                    : ir::ProductArithmetic::kImplementation;
                       }}});
  }
  return std::move(execution::Registry::Create(std::move(all)).value());
}

TEST(OwnerProductsTest, RaggedRowsDerivePhysicalColumnsAndStableSlots) {
  for (auto widths : {std::array{1U, 3U}, std::array{3U, 1U}, std::array{4U, 4U}}) {
    auto f = Make(widths[0], widths[1]);
    const auto proved = Prove(f);
    ASSERT_TRUE(proved) << test_support::Failed(proved).value_or("");
    ASSERT_EQ(proved->products().size(), 1U);
    const auto& p = proved->products()[0];
    EXPECT_EQ(p.columns_per_row, 4U);
    EXPECT_EQ(p.basis, RowBasis::kInput);
    ASSERT_EQ(p.owners.size(), 2U);
    EXPECT_EQ(p.owners[0].slot, 5);
    EXPECT_EQ(p.owners[1].slot, 2);
    EXPECT_EQ(p.owners[1].first_column, widths[0] * 4U);
    EXPECT_EQ(p.owners[1].columns, widths[1] * 4U);
    EXPECT_EQ(p.owners[1].byte_offset, widths[0] * 4U * 8U * sizeof(float));
    // There are only two selected output rows. They do not describe input columns.
    EXPECT_EQ(p.owners[1].rows, widths[1]);
  }
}

TEST(OwnerProductsTest, TokenEmbeddingHasOnePhysicalColumnPerInputRow) {
  auto f = Make(1, 3, Variant::kEmbedding);
  auto p = Prove(f);
  ASSERT_TRUE(p) << test_support::Failed(p).value_or("");
  EXPECT_EQ(p->products()[0].columns_per_row, 1U);
  EXPECT_EQ(p->products()[0].root, f.feature);
}

TEST(OwnerProductsTest, LongTopologicalLineageHasNoRecursiveProofPath) {
  for (const bool unknown_tail : {false, true}) {
    ir::Builder b;
    const auto feature = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({32, 4})));
    const auto norm = b.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({32})), {.role = 1});
    auto h = feature;
    // More frames than a recursive proof can safely assume fit its stack.
    // Every supported node is row-local; an unsupported tail must still refuse.
    for (std::size_t i = 0; i < ir::Graph::kMaxNodes / 2; ++i) {
      if (i % 3 == 0)
        h = b.RmsNorm(h, 1e-6F);
      else if (i % 3 == 1)
        h = b.Mul(h, norm);
      else
        h = b.Reshape(h, ir::ShapeOf({32, 4}));
    }
    if (unknown_tail) h = b.Scale(h, 1.0F);
    const auto weight = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({8, 8})), {.role = 2});
    const auto product = b.MatMul(weight, b.Reshape(h, ir::ShapeOf({8, 16})));
    b.Output(product);
    const auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << test_support::Failed(graph, &ir::VerifyError::detail).value_or("");
    const auto proof =
        ProveOwnerProducts(*graph,
                           std::array{components::StepInput{
                               .tensor = feature, .kind = components::InputKind::kFeatures}},
                           std::array{components::SegmentRows{.slot = 5, .rows = 4, .outputs = 1}},
                           std::array{product});
    if (unknown_tail) {
      ASSERT_FALSE(proof);
      EXPECT_EQ(proof.error(), "owner products: unproved row lineage");
    } else {
      ASSERT_TRUE(proof) << test_support::Failed(proof).value_or("");
      EXPECT_EQ(proof->products()[0].columns_per_row, 4U);
    }
  }
}

TEST(OwnerProductsTest, RefusesUnknownOrMixedLineageAndAliasWrites) {
  for (const auto variant :
       {Variant::kCrossNorm, Variant::kUnknownBroadcast, Variant::kWrite, Variant::kReordered}) {
    auto f = Make(1, 3, variant);
    EXPECT_FALSE(Prove(f)) << static_cast<int>(variant);
  }
}

TEST(OwnerProductsTest, RefusesDuplicateMissingAndMalformedTypedRows) {
  auto f = Make(1, 3);
  auto inputs = f.inputs;
  f.inputs.clear();
  EXPECT_FALSE(Prove(f));
  f.inputs = inputs;
  f.inputs.push_back(inputs[0]);
  EXPECT_FALSE(Prove(f));
  f.inputs = inputs;
  f.inputs[0].kind = components::InputKind::kOutputRows;
  EXPECT_FALSE(Prove(f));
  f.inputs = inputs;
  f.segments[1].slot = f.segments[0].slot;
  EXPECT_FALSE(Prove(f));
  f.segments[1].slot = 2;
  f.segments[1].first_row = 0;
  EXPECT_FALSE(Prove(f));
}

TEST(OwnerProductsTest, ProofAndPartitionHaveDistinctIdentities) {
  auto f = Make(1, 3);
  auto p = Prove(f);
  ASSERT_TRUE(p);
  f.segments[0].slot = 9;
  auto q = Prove(f);
  ASSERT_TRUE(q);
  EXPECT_NE(p->digest(), q->digest());
  auto other = Make(3, 1);
  // Same total columns, but the proof must bind to the graph as well as partition.
  EXPECT_TRUE(SplitOwnerProducts(other.graph, *p));
  auto changed = Make(3, 3);
  EXPECT_FALSE(SplitOwnerProducts(changed.graph, *p));
}

TEST(OwnerProductsTest, SplitsBeforeBf16PreparationAndComposesEveryInterfaceMap) {
  const auto registry = Registry();
  for (auto widths :
       {std::array{1U, 3U}, std::array{3U, 1U}, std::array{3U, 3U}, std::array{4U, 4U}}) {
    auto f = Make(widths[0], widths[1]);
    auto proof = Prove(f);
    ASSERT_TRUE(proof) << test_support::Failed(proof).value_or("");
    Policy policy;

    const auto plain = Plan(f.graph.Clone(), registry, {}, policy);
    ASSERT_TRUE(plain) << plain.error().detail;
    const auto changed = Plan(f.graph.Clone(), registry, {}, policy, {.owner_products = &*proof});
    ASSERT_TRUE(changed) << changed.error().detail;
    EXPECT_EQ(changed->owner_digest, proof->digest());
    ASSERT_EQ(changed->owner_products.size(), 2U);
    for (std::size_t i = 0; i < 2; ++i) {
      const auto& scope = changed->owner_products[i];
      const auto& n = changed->graph.node(changed->graph.producer(scope.result));
      EXPECT_EQ(n.key, ir::Primitive(Op::kMatMul));
      EXPECT_EQ(changed->graph.tensor(n.src[1]).ne(1), widths[i] * 4U);
      EXPECT_EQ(changed->graph.tensor(n.src[1]).format.encoding, Encoding::kF32);
      EXPECT_EQ(changed->from[changed->graph.producer(scope.result).v],
                f.graph.producer(f.product));
    }
    EXPECT_EQ(changed->graph.node(changed->graph.producer(changed->tensors[f.product.v])).key,
              ir::Primitive(Op::kConcat));
    EXPECT_EQ(changed->graph.outputs()[0], changed->tensors[f.output.v]);
    EXPECT_EQ(changed->graph.inputs()[0], changed->tensors[f.feature.v]);
    EXPECT_EQ(changed->graph.tensor(changed->tensors[f.weight.v]).binding,
              f.graph.tensor(f.weight).binding);
    EXPECT_FALSE(std::ranges::any_of(changed->graph.nodes(), [](const auto& n) {
      return n.key == ir::Primitive(Op::kConvert);
    }));
    if (widths[0] + widths[1] > 4)
      EXPECT_TRUE(std::ranges::any_of(plain->graph.nodes(), [](const auto& n) {
        return n.key == ir::Primitive(Op::kConvert);
      }));
  }
}

TEST(OwnerProductsTest, SingletonKeepsThePrimitiveAndItsProtectedScope) {
  for (const auto rows : {1U, 3U}) {
    auto f = Make(rows, 0);
    f.segments.resize(1);
    const auto proof = Prove(f);
    ASSERT_TRUE(proof) << test_support::Failed(proof).value_or("");
    const auto split = SplitOwnerProducts(f.graph, *proof);
    ASSERT_TRUE(split) << test_support::Failed(split, &ir::VerifyError::detail).value_or("");
    EXPECT_EQ(split->rewrite.graph.Digest(), f.graph.Digest());
    EXPECT_TRUE(split->rewrite.applied.empty());
    ASSERT_EQ(split->scopes.size(), 1U);
    EXPECT_EQ(split->scopes[0].result, f.product);
    EXPECT_EQ(split->scopes[0].primitive, f.graph.producer(f.product));
    EXPECT_EQ(split->scopes[0].slot, 5);
    EXPECT_EQ(split->scopes[0].segment, 0);
    EXPECT_EQ(split->scopes[0].columns, 4U * rows);
    Policy policy;

    const auto planned = Plan(f.graph.Clone(), Registry(), {}, policy, {.owner_products = &*proof});
    ASSERT_TRUE(planned) << planned.error().detail;
    ASSERT_EQ(planned->owner_products.size(), 1U);
    const auto& scope = planned->owner_products[0];
    EXPECT_EQ(scope.primitive, f.graph.producer(f.product));
    EXPECT_EQ(scope.result, planned->tensors[f.product.v]);
    EXPECT_EQ(scope.columns, 4U * rows);
    const auto& n = planned->graph.node(planned->graph.producer(scope.result));
    EXPECT_EQ(n.key, ir::Primitive(Op::kMatMul));
    EXPECT_EQ(planned->graph.tensor(n.src[1]).format.encoding, Encoding::kF32);
  }
}

TEST(OwnerProductsTest, CanonicalEquivalentGraphCannotReuseDifferentTensorIds) {
  const auto make = [](bool weight_first) {
    ir::Builder b;
    TensorId weight;
    if (weight_first) weight = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({8, 8})), {});
    const auto feature = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({32, 4})));
    if (!weight_first) weight = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({8, 8})), {});
    const auto product = b.MatMul(weight, b.Reshape(feature, ir::ShapeOf({8, 16})));
    b.Output(product);
    return Fixture{.graph = std::move(b).Finish().value(),
                   .inputs = {{.tensor = feature, .kind = components::InputKind::kFeatures}},
                   .segments = {{.slot = 5, .first_row = 0, .rows = 1, .outputs = 1},
                                {.slot = 2, .first_row = 1, .rows = 3, .outputs = 1}},
                   .product = product,
                   .feature = feature,
                   .weight = weight,
                   .output = product};
  };
  auto a = make(false);
  auto b = make(true);
  ASSERT_EQ(a.graph.Digest(), b.graph.Digest());
  auto proof = Prove(a);
  ASSERT_TRUE(proof);
  EXPECT_FALSE(SplitOwnerProducts(b.graph, *proof));
}

TEST(OwnerProductsTest, StepShapeCannotReplaceTheProvedPartition) {
  auto f = Make(1, 3);
  auto proof = Prove(f);
  ASSERT_TRUE(proof);
  StepShape shape{.segments = {{.slot = 5, .rows = 3}, {.slot = 2, .rows = 1}}};
  EXPECT_FALSE(
      Plan(f.graph.Clone(), Registry(), {}, {}, {.shape = &shape, .owner_products = &*proof}));
}

TEST(OwnerProductsTest, ComponentOptInAddsProofWithoutChangingPrimitiveGraphOrDefault) {
  const spec::ComponentRef ref{
      .kind = "embedding.mtp",
      .version = 1,
      .params = {{.name = "eps", .value = 1e-6}, {.name = "tap", .value = std::string("streams")}},
      .weights = {}};
  auto in = components::MakeInstance(ref, {.width = 8, .context = 4096, .streams = 4},
                                     components::Site::kEmbedding, 0, 0);
  ASSERT_TRUE(in) << in.error();
  const std::array formats{
      ir::Dense(Encoding::kF32, ir::ShapeOf({8, 32})), ir::Dense(Encoding::kF32, ir::ShapeOf({8})),
      ir::Dense(Encoding::kBF16, ir::ShapeOf({8, 8})), ir::Dense(Encoding::kF32, ir::ShapeOf({32})),
      ir::Dense(Encoding::kBF16, ir::ShapeOf({8, 8}))};
  StepShape shape{.segments = {{.slot = 5, .rows = 1, .read_cells = 256, .outputs = 1},
                               {.slot = 2, .rows = 3, .read_cells = 256, .outputs = 1}}};
  const auto plain =
      components::BuildComponent(*in, shape, formats, std::array<std::uint64_t, 1>{16});
  ASSERT_TRUE(plain) << plain.error();
  const auto annotated = components::BuildComponent(
      *in, shape, formats, std::array<std::uint64_t, 1>{16}, {.owner_products = true});
  ASSERT_TRUE(annotated) << annotated.error();
  EXPECT_EQ(plain->step.graph.Digest(), annotated->step.graph.Digest());
  EXPECT_TRUE(plain->step.owner_products.empty());
  ASSERT_EQ(annotated->step.owner_products.size(), 1U);
  const std::array segments{
      components::SegmentRows{.slot = 5, .first_row = 0, .rows = 1, .outputs = 1},
      components::SegmentRows{.slot = 2, .first_row = 1, .rows = 3, .outputs = 1}};
  const auto proved = ProveOwnerProducts(annotated->step.graph, annotated->step.inputs, segments,
                                         annotated->step.owner_products);
  ASSERT_TRUE(proved) << proved.error();
  ASSERT_EQ(proved->products().size(), 1U);
  EXPECT_EQ(proved->products()[0].owners[1].columns, 12U);
}

TEST(OwnerProductsTest, WiderOwnerRefusesRatherThanUsingAggregateOrRoundedRecipe) {
  auto f = Make(1, 5);
  auto proof = Prove(f);
  ASSERT_TRUE(proof);
  EXPECT_FALSE(SplitOwnerProducts(f.graph, *proof));
  f = Make(5, 0);
  f.segments.resize(1);
  proof = Prove(f);
  ASSERT_TRUE(proof);
  EXPECT_FALSE(SplitOwnerProducts(f.graph, *proof));
  Policy policy;

  EXPECT_FALSE(Plan(f.graph.Clone(), Registry(), {}, policy, {.owner_products = &*proof}));
  const auto unopted = Plan(f.graph.Clone(), Registry(), {}, policy);
  ASSERT_TRUE(unopted) << unopted.error().detail;
  EXPECT_TRUE(std::ranges::any_of(
      unopted->graph.nodes(), [](const auto& n) { return n.key == ir::Primitive(Op::kConvert); }));
}
TEST(OwnerProductsTest, AlignmentSelectionSurvivesOwnerPatternAndFallbackMaps) {
  ir::Builder b;
  const auto feature = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({32, 4})));
  const auto norm = b.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({32})), {.role = 1});
  const auto normalized = b.RmsNorm(feature, 1e-6F);
  const auto weighted = b.Mul(normalized, norm);
  const auto matrix = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({8, 8})), {.role = 2});
  const auto rows = b.Reshape(weighted, ir::ShapeOf({8, 16}));
  const auto product = b.MatMul(matrix, rows);
  const auto result = b.Scale(product, 1.0F);
  b.Output(result);
  // A separate wide product forces an enabled preparation rewrite.
  const auto wide = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({8, 32})));
  const auto wide_product = b.MatMul(matrix, wide);
  b.Output(b.Scale(wide_product, 1.0F));
  const auto excluded = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({16})));
  const auto excluded_result = b.Scale(excluded, 1.0F);
  b.Output(excluded_result);
  const auto state = b.State(ir::Dense(Encoding::kF32, ir::ShapeOf({16})), {.tensor = 0});
  const auto aligned_result = b.Scale(state, 1.0F);
  b.Output(aligned_result);
  const auto state_weight = b.Weight(ir::Dense(Encoding::kF32, ir::ShapeOf({16})), {.role = 3});
  // No fused implementation exists, forcing an unrelated decomposition.
  const auto fallback = b.Fused(ir::FusedKind::kRmsNormMul, std::array{state, state_weight},
                                ir::NormParams{.eps = 1e-6F})[0];
  b.Output(fallback);
  // This staged leaf moves after owner splitting, preparation, and fallback.
  // Unlike feature (tensor 0), it detects lost staged-ID forwarding.
  const auto late = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({16})));
  const auto late_result = b.Scale(late, 1.0F);
  b.Output(late_result);
  const auto first_row = b.View(feature, ir::ShapeOf({32, 1}), b.tensor(feature).nb, 0);
  const auto aligned_view_result = b.Scale(first_row, 1.0F);
  b.Output(aligned_view_result);
  const auto second_row = b.View(feature, ir::ShapeOf({32, 1}), b.tensor(feature).nb, 128);
  const auto misaligned_result = b.Scale(second_row, 1.0F);
  b.Output(misaligned_result);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph) << test_support::Failed(graph, &ir::VerifyError::detail).value_or("");
  const auto proof = ProveOwnerProducts(
      *graph,
      std::array{
          components::StepInput{.tensor = feature, .kind = components::InputKind::kFeatures}},
      std::array{components::SegmentRows{.slot = 5, .first_row = 0, .rows = 1, .outputs = 1},
                 components::SegmentRows{.slot = 2, .first_row = 1, .rows = 3, .outputs = 1}},
      std::array{product});
  ASSERT_TRUE(proof) << test_support::Failed(proof).value_or("");
  const auto ordinary = Registry();
  std::vector<execution::Implementation> implementations;
  for (std::size_t i = 0; i < ordinary.size(); ++i) implementations.push_back(ordinary.at(i));
  implementations.push_back(
      {.name = "fake.aligned_scale",
       .source = "fake",
       .revision = "r1",
       .build = "cpu",
       .variant = "256-byte-root-and-view",
       .capability = execution::Capability{.op = ir::Primitive(Op::kScale),
                                           .priority = 10,
                                           .accepts = +[](const execution::CapabilityQuery& q) {
                                             return !q.operands.empty() && q.operands[0].memory &&
                                                    q.operands[0].memory->root_alignment >= 256 &&
                                                    q.operands[0].memory->view_offset % 256 == 0;
                                           }}});
  auto made_registry = execution::Registry::Create(std::move(implementations));
  ASSERT_TRUE(made_registry);
  const auto& registry = *made_registry;
  Policy policy;

  const auto facts = GraphRootAlignments(*graph, std::array{RootAlignmentFact{state, 256}});
  for (const bool split : {false, true}) {
    SCOPED_TRACE(split);
    auto planned = Plan(graph->Clone(), registry, {}, policy,
                        {.staged = std::vector{feature, wide, excluded, late},
                         .root_alignment = facts,
                         .activation_alignment = 256,
                         .alignment_staged = std::vector{feature, late},
                         .owner_products = split ? &*proof : nullptr});
    ASSERT_TRUE(planned) << test_support::Failed(planned, &PlanFailure::detail).value_or("");
    EXPECT_TRUE(std::ranges::any_of(planned->applied,
                                    [](const auto& a) { return a.pattern == "bf16_inputs"; }));
    EXPECT_EQ(planned->decomposed, std::vector<std::string>{"fused.rms_norm_mul"});
    const auto selected = [&](TensorId original) -> std::string_view {
      const auto node = planned->graph.producer(planned->tensors.at(original.v));
      const auto step = std::ranges::find_if(
          planned->steps, [&](const auto& s) { return std::ranges::contains(s.nodes, node); });
      EXPECT_NE(step, planned->steps.end());
      return step == planned->steps.end() ? std::string_view{}
                                          : registry.at(step->implementation).name;
    };
    EXPECT_EQ(selected(aligned_result), "fake.aligned_scale");
    EXPECT_EQ(selected(late_result), "fake.aligned_scale");
    EXPECT_EQ(selected(aligned_view_result), "fake.aligned_scale");
    EXPECT_NE(selected(excluded_result), "fake.aligned_scale");
    EXPECT_NE(selected(misaligned_result), "fake.aligned_scale");
    EXPECT_TRUE(std::ranges::any_of(planned->root_alignments, [&](const auto& f) {
      return f.root == planned->tensors.at(state.v) && f.alignment == 256;
    }));
    EXPECT_NE(planned->tensors.at(late.v), late);
    EXPECT_TRUE(std::ranges::any_of(planned->root_alignments, [&](const auto& f) {
      return f.root == planned->tensors.at(late.v) && f.alignment == 256;
    }));
    EXPECT_FALSE(std::ranges::any_of(planned->root_alignments, [&](const auto& f) {
      return f.root == planned->tensors.at(excluded.v);
    }));
    ASSERT_EQ(planned->graph.inputs().size(), graph->inputs().size());
    ASSERT_EQ(planned->graph.outputs().size(), graph->outputs().size());
    for (std::size_t i = 0; i < graph->inputs().size(); ++i)
      EXPECT_EQ(planned->graph.inputs()[i], planned->tensors.at(graph->inputs()[i].v));
    for (std::size_t i = 0; i < graph->outputs().size(); ++i)
      EXPECT_EQ(planned->graph.outputs()[i], planned->tensors.at(graph->outputs()[i].v));
    ASSERT_EQ(planned->owner_products.size(), split ? 2U : 0U);
    if (split) {
      EXPECT_EQ(planned->owner_digest, proof->digest());
      EXPECT_EQ(planned->owner_products[0].columns, 4U);
      EXPECT_EQ(planned->owner_products[1].columns, 12U);
      for (const auto& scope : planned->owner_products)
        EXPECT_EQ(planned->graph.node(planned->graph.producer(scope.result)).key,
                  ir::Primitive(Op::kMatMul));
    }
  }
}

}  // namespace
}  // namespace llmp::planner
