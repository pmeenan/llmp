// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Generic graph ordering, fusion, activation placement, lane ordering and
// quantized KV planning. Synthetic descriptors need no family graph builder.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/set_rows_group.h"
#include "kernels/ggml/tensors.h"

namespace {

namespace kg = llmp::kernels::ggml;
using kg::MulMatPath;

TEST(GraphOrder, BoundedMembershipPreservesDfsViewsParamsAndDuplicateOutputs) {
  std::array<ggml_tensor, 5> t{};
  t[1].flags = GGML_TENSOR_FLAG_PARAM;
  t[2].op = GGML_OP_ADD;
  t[2].src[0] = &t[0];
  t[2].src[1] = &t[1];
  t[2].src[2] = &t[1];
  t[3].op = GGML_OP_VIEW;
  t[3].view_src = &t[2];
  t[3].src[0] = &t[2];
  t[4].op = GGML_OP_MUL;
  t[4].src[0] = &t[2];
  t[4].src[1] = &t[3];
  const std::array<ggml_tensor*, 4> outputs{&t[3], &t[4], &t[3], &t[1]};
  auto arena = kg::TensorArena::Create(t.size());
  ASSERT_TRUE(arena);
  const std::vector<ggml_tensor*> expected{&t[1], &t[2], &t[3], &t[4]};
  EXPECT_EQ(kg::GraphOrder(outputs), expected);
  for (unsigned repeat = 0; repeat < 3; ++repeat) {
    auto ordered = kg::GraphOrder(outputs, *arena);
    ASSERT_TRUE(ordered);
    EXPECT_EQ(*ordered, expected);
  }
  // The old walk follows src[] only, not an unrelated view_src field.
  t[3].src[0] = nullptr;
  t[3].view_src = &t[4];
  const std::array<ggml_tensor*, 1> unrelated_view{&t[3]};
  auto viewed = kg::GraphOrder(unrelated_view, *arena);
  ASSERT_TRUE(viewed);
  EXPECT_EQ(*viewed, kg::GraphOrder(unrelated_view));
  EXPECT_EQ(*viewed, (std::vector<ggml_tensor*>{&t[3]}));
  auto empty = kg::GraphOrder({}, *arena);
  ASSERT_TRUE(empty);
  EXPECT_TRUE(empty->empty());
}

TEST(GraphOrder, BoundedMembershipPreservesSharedDagAndCyclicEdgeOrder) {
  std::array<ggml_tensor, 513> t{};
  for (std::size_t i = 0; i < t.size(); ++i) {
    t[i].op = i % 5 == 0 ? GGML_OP_NONE : GGML_OP_ADD;
    if (i % 13 == 0) t[i].flags = GGML_TENSOR_FLAG_PARAM;
    if (i > 0) t[i].src[0] = &t[i - 1];
    if (i > 1) t[i].src[1] = &t[i / 2];
    t[i].src[2] = &t[i];  // insertion precedes descent, as in the old walk
  }
  t[0].src[0] = &t.back();  // also an indirect cycle
  const std::array<ggml_tensor*, 3> outputs{&t.back(), &t[300], &t.back()};
  auto arena = kg::TensorArena::Create(t.size());
  ASSERT_TRUE(arena);
  const auto expected = kg::GraphOrder(outputs);
  auto ordered = kg::GraphOrder(outputs, *arena);
  ASSERT_TRUE(ordered);
  EXPECT_EQ(*ordered, expected);
}

TEST(GraphOrder, AllReachedLeavesCountAndForeignDescriptorsRefuseAtTheFundedBound) {
  std::array<ggml_tensor, 3> foreign{};
  foreign[2].op = GGML_OP_ADD;
  foreign[2].src[0] = &foreign[0];
  foreign[2].src[1] = &foreign[1];
  const std::array<ggml_tensor*, 1> output{&foreign[2]};
  EXPECT_EQ(kg::GraphOrder(output).size(), 1U);  // both NONE leaves are omitted
  auto short_arena = kg::TensorArena::Create(2);
  ASSERT_TRUE(short_arena);
  EXPECT_FALSE(kg::GraphOrder(output, *short_arena));
  const std::array<ggml_tensor*, 1> leaf{&foreign[0]};
  auto retry = kg::GraphOrder(leaf, *short_arena);
  ASSERT_TRUE(retry);
  EXPECT_TRUE(retry->empty());  // a failed walk leaves no membership behind
  const std::array<ggml_tensor*, 1> null_output{nullptr};
  EXPECT_FALSE(kg::GraphOrder(null_output, *short_arena));
  auto exact_arena = kg::TensorArena::Create(3);
  ASSERT_TRUE(exact_arena);
  auto exact = kg::GraphOrder(output, *exact_arena);
  ASSERT_TRUE(exact);
  EXPECT_EQ(*exact, kg::GraphOrder(output));
}

// Synthetic device choices exercise vector, tensor-core and cuBLAS plans.
kg::DeviceChoices ModelDevice() {
  return {.mul_mat = [](const ggml_tensor* node) -> std::expected<MulMatPath, kg::KernelFailure> {
            const std::int64_t columns = node->src[1]->ne[1];
            if (columns == 1) {
              return MulMatPath::kVector;
            }
            return columns <= 16 ? MulMatPath::kTensorCore : MulMatPath::kCublas;
          },
          .vector_fusible = [](const ggml_tensor* node) { return node->src[1]->ne[1] == 1; },
          .quant = nullptr,
          .dense_mmvq_shape = {}};
}

TEST(GemmaActivationPlanTest, PlansPrimitiveAndEligibleFusionWithSeparateOperationNames) {
  auto arena = kg::TensorArena::Create(96).value();
  auto* c = arena.context();
  auto device = ModelDevice();
  device.geglu_fusible = [](const ggml_tensor* up) { return up->ne[0] == 704; };
  for (const std::int64_t width : {2112, 704}) {
    for (const std::int64_t rows : {1, 4}) {
      auto* x = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2816, rows);
      auto* wg = ggml_new_tensor_2d(c, GGML_TYPE_F16, 2816, width);
      auto* wu = ggml_new_tensor_2d(c, GGML_TYPE_F16, 2816, width);
      auto* gate = ggml_mul_mat(c, wg, x);
      auto* up = ggml_mul_mat(c, wu, x);
      ggml_prec_set_acc(gate, GGML_PREC_F32);
      ggml_prec_set_acc(up, GGML_PREC_F32);
      auto* glu = ggml_geglu_split(c, gate, up);
      std::array<ggml_tensor*, 1> outputs{glu};
      auto nodes = kg::GraphOrder(outputs);
      kg::BindDistinct(nodes, std::uint64_t{1} << 40U);
      const auto ordinary = kg::PlanGraph(nodes, false, ModelDevice());
      ASSERT_TRUE(ordinary);
      ASSERT_EQ(ordinary->steps.size(), 3U);
      EXPECT_EQ(ordinary->steps.back().implementation, kg::kGeGluName);
      EXPECT_EQ(ordinary->steps.back().operation, llmp::execution::Operation::kGeGlu);
      const auto default_off = kg::PlanGraph(nodes, true, ModelDevice());
      ASSERT_TRUE(default_off);
      EXPECT_EQ(default_off->steps.size(), 3U);
      const auto fused = kg::PlanGraph(nodes, true, device);
      ASSERT_TRUE(fused);
      EXPECT_EQ(fused->steps.size(), rows == 1 && width == 704 ? 1U : 3U);
      EXPECT_EQ(fused->steps.back().implementation,
                rows == 1 && width == 704 ? kg::kMulMatGeGluFused : kg::kGeGluName);
      const std::array<ggml_tensor*, 1> keep{gate};
      const auto kept = kg::PlanGraph(nodes, true, device, keep);
      ASSERT_TRUE(kept);
      EXPECT_EQ(kept->steps.size(), 3U);  // retained intermediate must be produced
      ggml_prec_set_acc(up, GGML_PREC_DEFAULT);
      const auto precise = kg::PlanGraph(nodes, true, device);
      ASSERT_TRUE(precise);
      EXPECT_EQ(precise->steps.size(), 3U);  // GLU params must not replace product precision
      ggml_prec_set_acc(up, GGML_PREC_F32);
      glu->op_params[1] = 1;
      EXPECT_FALSE(kg::PlanGraph(nodes, false, ModelDevice()));
      glu->op_params[1] = 0;
      glu->op_params[0] = GGML_GLU_OP_GEGLU_ERF;
      EXPECT_FALSE(kg::PlanGraph(nodes, false, ModelDevice()));
    }
  }
}

TEST(GemmaActivationPlanTest, GeluTanhUsesTheUnaryPrimitive) {
  auto arena = kg::TensorArena::Create(8).value();
  auto* x = ggml_new_tensor_2d(arena.context(), GGML_TYPE_F32, 2112, 4);
  auto* gelu = ggml_gelu(arena.context(), x);
  std::array<ggml_tensor*, 1> outputs{gelu};
  const auto nodes = kg::GraphOrder(outputs);
  const auto plan = kg::PlanGraph(nodes, false, ModelDevice());
  ASSERT_TRUE(plan);
  ASSERT_EQ(plan->steps.size(), 1U);
  EXPECT_EQ(plan->steps[0].implementation, kg::kUnaryName);
}

TEST(GemmaActivationPlanTest, AdversarialRetainedViewsAndReadersKeepProductsLive) {
  auto arena = kg::TensorArena::Create(24).value();
  auto* c = arena.context();
  auto* x = ggml_new_tensor_1d(c, GGML_TYPE_F32, 32);
  auto* wg = ggml_new_tensor_2d(c, GGML_TYPE_F16, 32, 64);
  auto* wu = ggml_new_tensor_2d(c, GGML_TYPE_F16, 32, 64);
  auto* gate = ggml_mul_mat(c, wg, x);
  auto* up = ggml_mul_mat(c, wu, x);
  ggml_prec_set_acc(gate, GGML_PREC_F32);
  ggml_prec_set_acc(up, GGML_PREC_F32);
  auto* glu = ggml_geglu_split(c, gate, up);
  auto device = ModelDevice();
  device.geglu_fusible = [](const ggml_tensor*) { return true; };
  const std::array<ggml_tensor*, 1> outputs{glu};
  auto nodes = kg::GraphOrder(outputs);
  kg::BindDistinct(nodes, std::uint64_t{1} << 40U);
  const auto eligible = kg::PlanGraph(nodes, true, device);
  ASSERT_TRUE(eligible);
  ASSERT_EQ(eligible->steps.size(), 1U);
  for (auto* product : {gate, up}) {
    auto* retained = ggml_view_1d(c, product, 16, 16 * sizeof(float));
    const std::array<ggml_tensor*, 1> keep{retained};
    const auto kept = kg::PlanGraph(nodes, true, device, keep);
    ASSERT_TRUE(kept);
    ASSERT_EQ(kept->steps.size(), 3U);
    EXPECT_EQ(kept->steps[0].nodes.back(), gate);
    EXPECT_EQ(kept->steps[1].nodes.back(), up);
    auto* reader = ggml_cont(c, product);
    const std::array<ggml_tensor*, 2> extra_outputs{glu, reader};
    auto extra_nodes = kg::GraphOrder(extra_outputs);
    kg::BindDistinct(extra_nodes, std::uint64_t{1} << 40U);
    const auto read = kg::PlanGraph(extra_nodes, true, device);
    ASSERT_TRUE(read);
    ASSERT_EQ(read->steps.size(), 4U);
    EXPECT_EQ(read->steps[2].operation, llmp::execution::Operation::kGeGlu);
    EXPECT_EQ(read->steps[3].implementation, kg::kContName);
  }
}

TEST(GraphPlanTest, RefusesAPatternWithNoImplementation) {
  auto arena = kg::TensorArena::Create(64);
  ASSERT_TRUE(arena.has_value());
  ggml_context* c = arena->context();
  ggml_tensor* a = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, 4);
  ggml_tensor* b = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, 4);
  ggml_tensor* w = ggml_new_tensor_2d(c, GGML_TYPE_F16, 64, 64);
  const auto refused = [&](ggml_tensor* out, std::string_view pattern) {
    std::vector<ggml_tensor*> outputs = {out};
    const std::vector<ggml_tensor*> nodes = kg::GraphOrder(outputs);
    kg::BindDistinct(nodes, std::uint64_t{1} << 40U);
    EXPECT_EQ(kg::UnimplementedFusionAt(nodes, 0), pattern);
    EXPECT_FALSE(kg::PlanGraph(nodes, true, ModelDevice()).has_value());
    EXPECT_TRUE(kg::PlanGraph(nodes, false, ModelDevice()).has_value() || pattern == "SCALE");
  };
  refused(ggml_add(c, ggml_add(c, a, b), b), "add or mul chain");
  refused(ggml_mul(c, ggml_mul_mat(c, w, a), b), "mul_mat and scale");
  refused(ggml_add(c, ggml_mul(c, ggml_rms_norm(c, a, 1e-6f), b), b),
          "rms_norm, mul and rope or add");
  refused(ggml_scale(c, a, 2.0f), "SCALE");
}

// Where each placed tensor lives.
std::map<const ggml_tensor*, std::pair<std::uint64_t, std::uint64_t>> Ranges(
    const kg::Placement& placement) {
  std::map<const ggml_tensor*, std::pair<std::uint64_t, std::uint64_t>> ranges;
  for (const auto& [tensor, offset] : placement.offsets) {
    ranges[tensor] = {offset, offset + ggml_nbytes(tensor)};
  }
  return ranges;
}

TEST(ActivationMeasurement, ThresholdUsesValidatedUniqueRootsAndPreservesExactFallback) {
  auto arena = kg::TensorArena::Create(8);
  ASSERT_TRUE(arena);
  auto* x = ggml_new_tensor_1d(arena->context(), GGML_TYPE_F32, 1024);
  auto* a = ggml_add(arena->context(), x, x);
  auto* b = ggml_mul(arena->context(), a, a);
  auto* view = ggml_view_1d(arena->context(), b, 512, 0);
  const std::array<ggml_tensor*, 3> nodes{a, b, view};
  const std::array<ggml_tensor*, 2> inputs{x, x};
  const std::array<ggml_tensor*, 1> kept{a};
  auto plan = kg::PlanGraph(nodes, false, ModelDevice());
  ASSERT_TRUE(plan);
  for (const bool lanes : {false, true}) {
    if (lanes) plan->regions = {{0, 1}};
    const auto exact = kg::PlaceActivations(nodes, *plan, inputs, 256, kept);
    ASSERT_TRUE(exact);
    constexpr std::uint64_t bound = 3 * 4096;
    auto measured = kg::MeasureActivations(nodes, *plan, inputs, 256, kept, bound);
    ASSERT_TRUE(measured);
    EXPECT_TRUE(measured->measurement_bound);
    EXPECT_TRUE(measured->offsets.empty());
    EXPECT_EQ(measured->extent, bound);
    EXPECT_LE(exact->extent, measured->extent);
    measured = kg::MeasureActivations(nodes, *plan, inputs, 256, kept, bound - 1);
    ASSERT_TRUE(measured);
    EXPECT_FALSE(measured->measurement_bound);
    EXPECT_EQ(measured->extent, exact->extent);
    EXPECT_EQ(measured->offsets, exact->offsets);
    EXPECT_FALSE(exact->measurement_bound);
  }
  EXPECT_FALSE(kg::MeasureActivations(nodes, {}, inputs, 256, kept, UINT64_MAX));
  EXPECT_FALSE(kg::MeasureActivations(nodes, *plan, inputs, 3, kept, UINT64_MAX));
  const std::array<ggml_tensor*, 1> computed_input{a};
  EXPECT_FALSE(kg::MeasureActivations(nodes, *plan, computed_input, 256, kept, UINT64_MAX));
}

TEST(ActivationMeasurement, RefusesRoundingAndSummedStorageOverflowInBothPolicies) {
  auto arena = kg::TensorArena::Create(2);
  ASSERT_TRUE(arena);
  auto* a = ggml_new_tensor_1d(arena->context(), GGML_TYPE_F32, 1);
  auto* b = ggml_new_tensor_1d(arena->context(), GGML_TYPE_F32, 1);
  a->ne[1] = 3;
  a->nb[1] = (std::uint64_t{1} << 63U);  // raw ggml_nbytes wraps to four
  const std::array<ggml_tensor*, 1> one{a};
  EXPECT_FALSE(kg::MeasureActivations({}, {}, one, 256, {}, 256));
  EXPECT_FALSE(kg::PlaceActivations({}, {}, one, 256));
  a->nb[1] = (UINT64_MAX - 4) / 2;  // valid span UINT64_MAX-1; rounding overflows
  EXPECT_FALSE(kg::MeasureActivations({}, {}, one, 256, {}, UINT64_MAX));
  EXPECT_FALSE(kg::PlaceActivations({}, {}, one, 256));
  a->ne[1] = b->ne[1] = 2;
  a->nb[1] = b->nb[1] = (std::uint64_t{1} << 63U) - 4;
  const std::array<ggml_tensor*, 2> two{a, b};
  EXPECT_FALSE(kg::MeasureActivations({}, {}, two, 256, {}, UINT64_MAX));
  EXPECT_FALSE(kg::PlaceActivations({}, {}, two, 256));
}

TEST(ActivationMeasurement, BoundsActualNoncanonicalBlockedStorage) {
  auto arena = kg::TensorArena::Create(1);
  ASSERT_TRUE(arena);
  auto* input = ggml_new_tensor_1d(arena->context(), GGML_TYPE_Q4_1, 64);
  input->nb[0] = 40;  // two blocks, twice the canonical 20-byte block storage
  const std::array<ggml_tensor*, 1> inputs{input};
  ASSERT_EQ(ggml_nbytes(input), 80U);
  auto exact = kg::PlaceActivations({}, {}, inputs, 16);
  ASSERT_TRUE(exact);
  EXPECT_EQ(exact->extent, 80U);
  auto measured = kg::MeasureActivations({}, {}, inputs, 16, {}, 80);
  ASSERT_TRUE(measured);
  EXPECT_TRUE(measured->measurement_bound);
  EXPECT_EQ(measured->extent, exact->extent);
  measured = kg::MeasureActivations({}, {}, inputs, 16, {}, 79);
  ASSERT_TRUE(measured);
  EXPECT_FALSE(measured->measurement_bound);
  EXPECT_EQ(measured->extent, exact->extent);
}

// A graph whose last node is a view keeps the viewed tensor alive to the
// end: in `a, b, view(a)`, b must not take a's bytes, or it overwrites the
// output. The same through a view of the view.
TEST(GraphPlanTest, AFinalViewKeepsItsSourceAlive) {
  for (const bool chained : {false, true}) {
    auto arena = kg::TensorArena::Create(8);
    ASSERT_TRUE(arena.has_value());
    ggml_context* c = arena->context();
    ggml_tensor* x = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1024);
    ggml_tensor* a = ggml_add(c, x, x);
    ggml_tensor* b = ggml_mul(c, x, x);
    ggml_tensor* last = ggml_view_1d(c, a, 512, 0);
    std::vector<ggml_tensor*> nodes = {a, b, last};
    if (chained) {
      last = ggml_view_1d(c, last, 256, 0);
      nodes.push_back(last);
    }
    const auto plan = kg::PlanGraph(nodes, false, ModelDevice());
    ASSERT_TRUE(plan.has_value()) << plan.error().detail;
    ASSERT_EQ(plan->steps.size(), 2U);
    const std::array<ggml_tensor*, 1> inputs = {x};
    const auto placement = kg::PlaceActivations(nodes, *plan, inputs, 128);
    ASSERT_TRUE(placement.has_value()) << placement.error().detail;
    const auto ranges = Ranges(*placement);
    ASSERT_TRUE(ranges.contains(a) && ranges.contains(b));
    const auto [alo, ahi] = ranges.at(a);
    const auto [blo, bhi] = ranges.at(b);
    EXPECT_TRUE(ahi <= blo || bhi <= alo)
        << "chained " << chained << ": a at " << alo << ", b at " << blo;
  }
}

// Concurrent lanes (graph_plan.h AssignLanes): steps take their nodes'
// lanes, a region spans its tagged steps (overlapping ones joined), and
// what a region's steps compute or read lives for the whole span, so two
// lanes' tensors never share bytes though each dies early in step order.
TEST(GraphPlanTest, LanesSpanRegionsAndKeepTheirTensorsApart) {
  auto arena = kg::TensorArena::Create(32);
  ASSERT_TRUE(arena.has_value());
  ggml_context* c = arena->context();
  ggml_tensor* x = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1024);
  ggml_tensor* before = ggml_add(c, x, x);           // the stream's, before
  ggml_tensor* a1 = ggml_mul(c, before, before);     // lane 1
  ggml_tensor* a2 = ggml_add(c, a1, a1);             // lane 1, a1 dies
  ggml_tensor* b1 = ggml_mul(c, before, x);          // lane 2
  ggml_tensor* b2 = ggml_add(c, b1, b1);             // lane 2, b1 dies
  ggml_tensor* joined = ggml_add(c, a2, b2);         // the stream's, in the region
  ggml_tensor* after = ggml_mul(c, joined, joined);  // after the region
  const std::vector<ggml_tensor*> nodes = {before, a1, a2, b1, b2, joined, after};
  kg::LaneTags tags = {{a1, {1, 7}}, {a2, {1, 7}}, {b1, {2, 7}}, {b2, {2, 7}}, {joined, {0, 7}}};
  auto plan = kg::PlanGraph(nodes, false, ModelDevice());
  ASSERT_TRUE(plan.has_value()) << plan.error().detail;
  ASSERT_EQ(plan->steps.size(), nodes.size());
  kg::AssignLanes(*plan, tags);
  std::vector<std::uint8_t> lanes;
  for (const kg::PlanStep& step : plan->steps) {
    lanes.push_back(step.lane);
  }
  EXPECT_EQ(lanes, (std::vector<std::uint8_t>{0, 1, 1, 2, 2, 0, 0}));
  ASSERT_EQ(plan->regions.size(), 1U);
  EXPECT_EQ(plan->regions[0].first, 1U);
  EXPECT_EQ(plan->regions[0].last, 5U);
  const std::array<ggml_tensor*, 1> inputs = {x};
  const auto placement = kg::PlaceActivations(nodes, *plan, inputs, 128);
  ASSERT_TRUE(placement.has_value()) << placement.error().detail;
  const auto ranges = Ranges(*placement);
  // On one stream a1 could share bytes with b1 or b2; on lanes they run at
  // once, so the region's tensors all stay apart.
  const std::vector<ggml_tensor*> region = {before, a1, a2, b1, b2, joined};
  for (std::size_t i = 0; i < region.size(); ++i) {
    for (std::size_t j = i + 1; j < region.size(); ++j) {
      const auto [ilo, ihi] = ranges.at(region[i]);
      const auto [jlo, jhi] = ranges.at(region[j]);
      EXPECT_TRUE(ihi <= jlo || jhi <= ilo) << i << " and " << j;
    }
  }
  // The same plan without lanes: no regions, every step on the stream.
  auto plain = kg::PlanGraph(nodes, false, ModelDevice());
  ASSERT_TRUE(plain.has_value());
  EXPECT_TRUE(plain->regions.empty());
  EXPECT_FALSE(kg::SamePlan(*plan, *plain));
  // Overlapping regions are joined; a lane step outside any region (none
  // tagged with a region) stays on the stream; a cuBLAS product keeps
  // lane 0 within its region.
  kg::LaneTags overlapping = {{a1, {1, 1}}, {a2, {1, 2}}, {b1, {2, 2}}, {b2, {2, 1}}};
  kg::AssignLanes(*plain, overlapping);
  ASSERT_EQ(plain->regions.size(), 1U);
  EXPECT_EQ(plain->regions[0].first, 1U);
  EXPECT_EQ(plain->regions[0].last, 4U);
  kg::AssignLanes(*plain, {{a1, {1, 0}}});
  EXPECT_TRUE(plain->regions.empty());
  EXPECT_EQ(plain->steps[1].lane, 0U);
  // A step whose implementation must stay on the stream (one borrowing
  // cuBLAS) keeps lane 0 inside its region; the region is unchanged.
  kg::AssignLanes(*plan, tags,
                  [](std::string_view implementation) { return implementation == kg::kMulName; });
  lanes.clear();
  for (const kg::PlanStep& step : plan->steps) {
    lanes.push_back(step.lane);
  }
  EXPECT_EQ(lanes, (std::vector<std::uint8_t>{0, 0, 1, 0, 2, 0, 0}));
  ASSERT_EQ(plan->regions.size(), 1U);
  EXPECT_EQ(plan->regions[0].first, 1U);
  EXPECT_EQ(plan->regions[0].last, 5U);
}

// The order concurrent lanes keep inside a region (graph_plan.h LaneOrder):
// a read waits for the tensor's last writer on another lane, a write for its
// last writer and for every other lane that read it since, a lane never
// waits for itself, a wait covers everything the awaited lane queued, and a
// region's end clears it all.
TEST(GraphPlanTest, LanesOrderEveryReadAndWriteAcrossLanes) {
  auto arena = kg::TensorArena::Create(8);
  ASSERT_TRUE(arena.has_value());
  ggml_context* c = arena->context();
  const ggml_tensor* x = ggml_new_tensor_1d(c, GGML_TYPE_F32, 16);
  const ggml_tensor* y = ggml_new_tensor_1d(c, GGML_TYPE_F32, 16);
  const ggml_tensor* z = ggml_new_tensor_1d(c, GGML_TYPE_F32, 16);
  using Lanes = std::vector<std::uint8_t>;
  using Ts = std::vector<const ggml_tensor*>;
  kg::LaneOrder order;
  // Lane 1 writes x; lane 2 reads it (read after write).
  EXPECT_EQ(order.Before(1, Ts{}, Ts{x}), Lanes{});
  order.Queued(1, Ts{}, Ts{x});
  EXPECT_EQ(order.Before(1, Ts{x}, Ts{y}), Lanes{});  // its own lane, in order
  EXPECT_EQ(order.Before(2, Ts{x}, Ts{y}), Lanes{1});
  order.Waited(2, 1);
  EXPECT_EQ(order.Before(2, Ts{x}, Ts{y}), Lanes{});  // the wait covers it
  order.Queued(2, Ts{x}, Ts{y});
  // Lane 3 overwrites x: after its writer (lane 1) and its reader (lane 2).
  EXPECT_EQ(order.Before(3, Ts{}, Ts{x}), (Lanes{1, 2}));
  // The stream writes y, which lane 2 wrote (write after write).
  EXPECT_EQ(order.Before(0, Ts{}, Ts{y}), Lanes{2});
  order.Waited(3, 1);
  order.Waited(3, 2);
  order.Queued(3, Ts{}, Ts{x});
  // Readers before lane 3's write are settled; a later read of x follows
  // lane 3 alone; z, untouched, needs nothing.
  EXPECT_EQ(order.Before(1, Ts{x}, Ts{z}), Lanes{3});
  EXPECT_EQ(order.Before(4, Ts{z}, Ts{z}), Lanes{});
  // A wait covers only what was queued by then.
  order.Waited(4, 3);
  order.Queued(3, Ts{}, Ts{z});
  EXPECT_EQ(order.Before(4, Ts{z}, Ts{}), Lanes{3});
  order.Reset();
  EXPECT_EQ(order.Before(4, Ts{x, y, z}, Ts{x, y, z}), Lanes{});
}

// Block-quantized KV caches (kv_quant.h): a Q8_0 or Q4_0 cache store, a
// RoPE stored into one, and attention over them are planned by their
// destination's and K's type, with the F16 forms' fusion and grouping
// options; nothing changes for F16 caches.
TEST(KvQuantPlanTest, QuantizedCachesTakeTheQuantizingStoresAndAttention) {
  auto arena = kg::TensorArena::Create(64).value();
  auto* c = arena.context();
  constexpr std::int64_t kD = 256, kHeads = 8, kKv = 4, kRows = 3, kCells = 512;
  std::uint64_t next = std::uint64_t{1} << 36U;
  const auto leaf = [&](ggml_tensor* t) {
    kg::TensorArena::Bind(t, next);
    next += (ggml_nbytes(t) + 511) / 256 * 256;
    return t;
  };
  for (const ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_F16}) {
    SCOPED_TRACE(ggml_type_name(type));
    auto* x = leaf(ggml_new_tensor_3d(c, GGML_TYPE_F32, kD, kKv, kRows));
    auto* pos = leaf(ggml_new_tensor_1d(c, GGML_TYPE_I32, kRows));
    auto* ids = leaf(ggml_new_tensor_1d(c, GGML_TYPE_I64, kRows));
    auto* k_cache = leaf(ggml_new_tensor_2d(c, type, kD * kKv, kCells));
    auto* v_cache = leaf(ggml_new_tensor_2d(c, type, kD * kKv, kCells));
    auto* v_rows = leaf(ggml_new_tensor_2d(c, GGML_TYPE_F32, kD * kKv, kRows));
    auto* w_cache = leaf(ggml_new_tensor_2d(c, type, kD * kKv, kCells));
    auto* w_rows = leaf(ggml_new_tensor_2d(c, GGML_TYPE_F32, kD * kKv, kRows));
    auto* q = leaf(ggml_new_tensor_3d(c, GGML_TYPE_F32, kD, kRows, kHeads));
    auto* mask = leaf(ggml_new_tensor_2d(c, GGML_TYPE_F16, kCells, kRows));
    auto* rope = ggml_rope_ext(c, x, pos, nullptr, kD, GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f,
                               0.0f, 1.0f, 0.0f, 0.0f);
    auto* flat = ggml_view_2d(c, rope, kD * kKv, kRows, rope->nb[2], 0);
    auto* k_store = ggml_set_rows(c, k_cache, flat, ids);
    auto* v_store = ggml_set_rows(c, v_cache, v_rows, ids);
    auto* w_store = ggml_set_rows(c, w_cache, w_rows, ids);
    const std::size_t row = ggml_row_size(type, kD);
    auto* k = ggml_view_3d(c, k_store, kD, kCells, kKv, k_cache->nb[1], row, 0);
    auto* v = ggml_view_3d(c, v_store, kD, kCells, kKv, v_cache->nb[1], row, 0);
    auto* attention = ggml_flash_attn_ext(c, q, k, v, mask, 0.0625f, 0.0f, 0.0f);
    ggml_prec_set_acc(attention, GGML_PREC_F32);
    // V's and W's stores first, consecutive; then K's RoPE store and the
    // attention.
    std::array<ggml_tensor*, 3> outputs{v_store, w_store, attention};
    auto nodes = kg::GraphOrder(outputs);
    kg::BindDistinct(nodes, std::uint64_t{1} << 40U);
    const auto names = [&](const kg::DeviceChoices& device) {
      std::vector<std::string_view> out;
      const auto plan = kg::PlanGraph(nodes, false, device);
      EXPECT_TRUE(plan.has_value()) << (plan ? "" : plan.error().detail);
      if (plan) {
        for (const auto& step : plan->steps) out.push_back(step.implementation);
      }
      return out;
    };
    const auto has = [](const std::vector<std::string_view>& plan, std::string_view name) {
      return std::ranges::count(plan, name);
    };
    const bool quantized = type != GGML_TYPE_F16;
    const auto plain = names(ModelDevice());
    auto device = ModelDevice();
    device.fuse_rope_store = true;
    device.group_set_rows = true;
    const auto fused = names(device);
    if (quantized) {
      const std::string_view attend =
          type == GGML_TYPE_Q8_0 ? kg::kFlashAttnVecQ8Name : kg::kFlashAttnVecQ4Name;
      EXPECT_EQ(has(plain, attend), 1);
      EXPECT_EQ(has(plain, kg::kSetRowsQuantName), 3);
      EXPECT_EQ(has(plain, kg::kRopeName), 1);
      EXPECT_EQ(has(fused, attend), 1);
      EXPECT_EQ(has(fused, kg::kRopeSetRowsQuantName), 1);
      EXPECT_EQ(has(fused, kg::kRopeName), 0);
      EXPECT_EQ(has(fused, kg::kSetRowsQuantGroupName), 1);
      EXPECT_EQ(has(fused, kg::kSetRowsQuantName), 0);
      EXPECT_EQ(has(fused, kg::kSetRowsName) + has(fused, kg::kSetRowsGroupedName), 0);
    } else {
      EXPECT_EQ(has(plain, kg::kSetRowsQuantName) + has(plain, kg::kRopeSetRowsQuantName) +
                    has(fused, kg::kSetRowsQuantName) + has(fused, kg::kSetRowsQuantGroupName) +
                    has(fused, kg::kRopeSetRowsQuantName),
                0);
      EXPECT_EQ(has(fused, kg::kRopeSetRowsFused), 1);
    }
  }
  // Quantized V needs K of its encoding: no plan names an attention for it.
  auto* q = ggml_new_tensor_3d(c, GGML_TYPE_F32, 64, 1, 2);
  auto* k = ggml_new_tensor_3d(c, GGML_TYPE_F16, 64, 256, 1);
  auto* v = ggml_new_tensor_3d(c, GGML_TYPE_Q8_0, 64, 256, 1);
  auto* mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, 256, 1);
  for (auto* t : {q, k, v, mask}) leaf(t);
  std::array<ggml_tensor*, 1> outputs{ggml_flash_attn_ext(c, q, k, v, mask, 0.125f, 0.0f, 0.0f)};
  auto nodes = kg::GraphOrder(outputs);
  kg::BindDistinct(nodes, std::uint64_t{1} << 41U);
  EXPECT_FALSE(kg::PlanGraph(nodes, false, ModelDevice()).has_value());
}

TEST(KvQuantPlanTest, Head64PreservesEachCacheImplementation) {
  auto arena = kg::TensorArena::Create(32).value();
  auto* c = arena.context();
  std::uint64_t next = std::uint64_t{1} << 36U;
  const auto leaf = [&](ggml_tensor* t) {
    kg::TensorArena::Bind(t, next);
    next += (ggml_nbytes(t) + 511) / 256 * 256;
    return t;
  };
  for (const ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
    SCOPED_TRACE(ggml_type_name(type));
    auto* q = leaf(ggml_new_tensor_3d(c, GGML_TYPE_F32, 64, 1, 2));
    auto* k = leaf(ggml_new_tensor_3d(c, type, 64, 256, 1));
    auto* v = leaf(ggml_new_tensor_3d(c, type, 64, 256, 1));
    auto* mask = leaf(ggml_new_tensor_2d(c, GGML_TYPE_F16, 256, 1));
    auto* attention = ggml_flash_attn_ext(c, q, k, v, mask, 0.125f, 0.0f, 0.0f);
    ggml_prec_set_acc(attention, GGML_PREC_F32);
    const std::array<ggml_tensor*, 1> outputs{attention};
    auto nodes = kg::GraphOrder(outputs);
    kg::BindDistinct(nodes, std::uint64_t{1} << 41U);
    const auto plan = kg::PlanGraph(nodes, false, ModelDevice());
    ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
    ASSERT_EQ(plan->steps.size(), 1U);
    const std::string_view expected =
        type == GGML_TYPE_F16 ? "ggml.flash_attn_ext.vec" : kg::FlashAttnVecQuantName(type);
    EXPECT_EQ(plan->steps.front().implementation, expected);
  }
}

TEST(GraphPlanTest, AnUnchangedPlanIsTheSamePlan) {
  auto arena = kg::TensorArena::Create(8);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  auto* input = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, 1);
  auto* scale = ggml_new_tensor_1d(c, GGML_TYPE_F32, 64);
  auto* output = ggml_mul(c, ggml_rms_norm(c, input, 1e-6F), scale);
  const std::array<ggml_tensor*, 1> outputs{output};
  const auto nodes = kg::GraphOrder(outputs);
  kg::BindDistinct(nodes, std::uint64_t{1} << 40U);
  const auto a = kg::PlanGraph(nodes, true, ModelDevice());
  const auto b = kg::PlanGraph(nodes, true, ModelDevice());
  const auto u = kg::PlanGraph(nodes, false, ModelDevice());
  ASSERT_TRUE(a && b && u);
  EXPECT_TRUE(kg::SamePlan(*a, *b));
  EXPECT_FALSE(kg::SamePlan(*a, *u));
  const auto choices = a->Choices();
  ASSERT_EQ(choices.size(), a->steps.size());
  EXPECT_EQ(choices.front().implementation, "ggml.rms_norm_mul.fused");
  EXPECT_EQ(choices.front().operation, llmp::execution::Operation::kRmsNormMul);
}

}  // namespace
