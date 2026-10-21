// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Generic graph ordering and lane ordering. Synthetic descriptors need no family graph builder.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/plan_step.h"
#include "kernels/ggml/tensors.h"

namespace {

namespace kg = llmp::kernels::ggml;

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

// Concurrent lanes (plan_step.h AssignLanes): steps take their nodes'
// lanes, a region spans its tagged steps (overlapping ones joined), and
// what a region's steps compute or read lives for the whole span.
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
  kg::GraphPlan plan;
  for (ggml_tensor* node : nodes) {
    plan.steps.push_back({.operation = llmp::ir::Primitive(
                              node->op == GGML_OP_MUL ? llmp::ir::Op::kMul : llmp::ir::Op::kAdd),
                          .implementation = node->op == GGML_OP_MUL ? kg::kMulName : kg::kAddName,
                          .nodes = {node}});
  }
  ASSERT_EQ(plan.steps.size(), nodes.size());
  kg::AssignLanes(plan, tags);
  std::vector<std::uint8_t> lanes;
  for (const kg::PlanStep& step : plan.steps) {
    lanes.push_back(step.lane);
  }
  EXPECT_EQ(lanes, (std::vector<std::uint8_t>{0, 1, 1, 2, 2, 0, 0}));
  ASSERT_EQ(plan.regions.size(), 1U);
  EXPECT_EQ(plan.regions[0].first, 1U);
  EXPECT_EQ(plan.regions[0].last, 5U);
  // The same plan without lanes: no regions, every step on the stream.
  kg::GraphPlan plain;
  for (ggml_tensor* node : nodes) {
    plain.steps.push_back({.operation = llmp::ir::Primitive(
                               node->op == GGML_OP_MUL ? llmp::ir::Op::kMul : llmp::ir::Op::kAdd),
                           .implementation = node->op == GGML_OP_MUL ? kg::kMulName : kg::kAddName,
                           .nodes = {node}});
  }
  EXPECT_TRUE(plain.regions.empty());
  // Overlapping regions are joined; a lane step outside any region (none
  // tagged with a region) stays on the stream; a cuBLAS product keeps
  // lane 0 within its region.
  kg::LaneTags overlapping = {{a1, {1, 1}}, {a2, {1, 2}}, {b1, {2, 2}}, {b2, {2, 1}}};
  kg::AssignLanes(plain, overlapping);
  ASSERT_EQ(plain.regions.size(), 1U);
  EXPECT_EQ(plain.regions[0].first, 1U);
  EXPECT_EQ(plain.regions[0].last, 4U);
  kg::AssignLanes(plain, {{a1, {1, 0}}});
  EXPECT_TRUE(plain.regions.empty());
  EXPECT_EQ(plain.steps[1].lane, 0U);
  // A step whose implementation must stay on the stream keeps lane 0 inside
  // its region; the region is unchanged.
  kg::AssignLanes(plan, tags,
                  [](std::string_view implementation) { return implementation == kg::kMulName; });
  lanes.clear();
  for (const kg::PlanStep& step : plan.steps) {
    lanes.push_back(step.lane);
  }
  EXPECT_EQ(lanes, (std::vector<std::uint8_t>{0, 0, 1, 0, 2, 0, 0}));
  ASSERT_EQ(plan.regions.size(), 1U);
  EXPECT_EQ(plan.regions[0].first, 1U);
  EXPECT_EQ(plan.regions[0].last, 5U);
}

// The order concurrent lanes keep inside a region (plan_step.h LaneOrder):
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

}  // namespace
