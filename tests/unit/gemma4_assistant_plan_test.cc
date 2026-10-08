// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "engine/gemma4_assistant_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <ranges>

#include "engine/gemma4_assistant.h"
#include "expected_error.h"
#include "gemma4_assistant_fixture.h"

namespace {
namespace en = llmp::engine;
namespace kg = llmp::kernels::ggml;
namespace md = llmp::model;
namespace fixture = llmp::test_support::gemma4;
struct Case {
  const md::Gemma4Profile& target = md::Gemma4_26BA4B();
  const md::Gemma4AssistantProfile& profile = md::Gemma4Assistant26();
  md::Gemma4Binding tb = *md::BindGemma4(target, "gemma4", fixture::Resources(26));
  md::Gemma4AssistantBinding b =
      *md::BindGemma4Assistant(profile, "gemma4-assistant", fixture::AssistantResources(26));
  md::Gemma4StateLayout state = *md::Gemma4State(target, 4096, 128);
  kg::TensorArena arena = std::move(*kg::TensorArena::Create(kg::Gemma4AssistantGraphTensors(2)));
  kg::Gemma4AssistantGraph graph;
  explicit Case(std::uint32_t prefix = 1, bool peer = false) {
    kg::Gemma4AssistantShape shape;
    const auto add = [&](std::uint32_t slot, std::uint32_t p) {
      shape.segments.push_back(
          {slot, p, std::min(state.local_cells, (p + 255) / 256 * 256), (p + 255) / 256 * 256});
    };
    add(0, prefix);
    if (peer) add(1, 1025);
    graph = std::move(*kg::BuildGemma4AssistantGraph(arena, profile, b, target, tb, state, shape));
  }
};
TEST(Gemma4AssistantPlan, FrozenMasksSeparateQueryEndpointAndPhysicalRetention) {
  for (const auto prefix : {1U, 1023U, 1024U, 1025U, 2561U}) {
    Case c(prefix, true);
    const auto bytes = en::Gemma4AssistantSourceBytes(c.graph, true);
    ASSERT_TRUE(bytes) << *llmp::test_support::Failed(bytes);
    const std::array<en::Gemma4AssistantInput, 2> inputs{{{0, prefix, 2}, {1, 1025, 3}}};
    auto sources = en::Gemma4AssistantSources(c.graph, inputs, {}, true, *bytes);
    ASSERT_TRUE(sources) << *llmp::test_support::Failed(sources);
    EXPECT_EQ(sources->positions,
              (std::vector<std::int32_t>{static_cast<std::int32_t>(prefix), 1025}));
    for (std::size_t owner = 0; owner < inputs.size(); ++owner) {
      const auto p = inputs[owner].prefix;
      for (std::size_t kind = 0; kind < 2; ++kind) {
        const auto cells = kind == 0 ? c.graph.segments[owner].shape.local_n_kv
                                     : c.graph.segments[owner].shape.global_n_kv;
        std::vector<bool> visible(cells);
        // Independent unrolled absolute prefix; map each retained local
        // position to its physical ring cell and apply the query window.
        for (std::uint32_t position = 0; position < p; ++position) {
          if (kind == 0 && p - position >= c.target.window) continue;
          const auto cell = kind == 0 ? position % c.state.local_cells : position;
          visible[cell] = true;
        }
        const auto& mask = sources->masks[owner * 2 + kind];
        for (std::uint32_t cell = 0; cell < cells; ++cell)
          EXPECT_EQ(mask[cell], visible[cell] ? 0 : 0xfc00) << p << '/' << kind << '/' << cell;
        EXPECT_TRUE(std::ranges::all_of(mask | std::views::drop(cells),
                                        [](auto v) { return v == 0xfc00; }));
        if (p < cells) EXPECT_EQ(mask[p], 0xfc00);
      }
    }
    EXPECT_FALSE(en::Gemma4AssistantSources(c.graph, inputs, {}, true, *bytes - 1));
  }
}
TEST(Gemma4AssistantPlan, FreshEndpointWithinTheSameReadShapeRebuildsMasks) {
  Case c(1025);
  const auto bytes = *en::Gemma4AssistantSourceBytes(c.graph, true);
  const std::array<en::Gemma4AssistantInput, 1> good{{{0, 1279, 2}}};
  auto sources = en::Gemma4AssistantSources(c.graph, good, {}, true, bytes);
  ASSERT_TRUE(sources);
  EXPECT_EQ(sources->positions[0], 1279);
  EXPECT_EQ(sources->masks[0][255], 0xfc00);  // exact age1024
  EXPECT_EQ(sources->masks[0][256], 0);
  for (const auto prefix : {0U, 1281U, UINT32_MAX}) {
    const std::array<en::Gemma4AssistantInput, 1> bad{{{0, prefix, 2}}};
    EXPECT_FALSE(en::Gemma4AssistantSources(c.graph, bad, {}, true, bytes));
  }
}
TEST(Gemma4AssistantPlan, MutableSegmentCacheMaskAndWeightDomainsAreRevalidated) {
  Case c(1025, true);
  ASSERT_TRUE(en::Gemma4AssistantSourceBytes(c.graph, true));
  auto bad = c.graph;
  bad.segments[0].shape.prefix++;
  EXPECT_FALSE(en::Gemma4AssistantSourceBytes(bad, true));
  bad = c.graph;
  bad.local_capacity -= 256;
  EXPECT_FALSE(en::Gemma4AssistantSourceBytes(bad, true));
  bad = c.graph;
  bad.segments[1].shape.slot = 0;
  EXPECT_FALSE(en::Gemma4AssistantSourceBytes(bad, true));
  bad = c.graph;
  bad.segments[1].masks[0] = bad.segments[0].masks[0];
  bad.inputs[5] = bad.segments[0].masks[0];
  EXPECT_FALSE(en::Gemma4AssistantSourceBytes(bad, true));
  bad = c.graph;
  bad.segments[1].caches[0].first = bad.segments[0].caches[0].first;
  EXPECT_FALSE(en::Gemma4AssistantSourceBytes(bad, true));
  bad = c.graph;
  bad.weights[0].resource.ne.assign(64, 1);
  EXPECT_FALSE(en::Gemma4AssistantSourceBytes(bad, true));
  bad = c.graph;
  bad.weights[0].resource.index = UINT32_MAX;
  EXPECT_FALSE(en::Gemma4AssistantSourceBytes(bad, true));
  bad = c.graph;
  std::swap(bad.inputs[0], bad.inputs[2]);
  EXPECT_FALSE(en::Gemma4AssistantSourceBytes(bad, true));
}
}  // namespace

TEST(Gemma4AssistantPlan, CompleteOutputValidationRejectsEveryNonfiniteBatchBeforePublication) {
  std::vector<float> heads(2 * 262144, 0), features(2 * 2816, 0);
  EXPECT_TRUE(en::CheckGemma4AssistantOutputs(2, 2816, heads, features));
  EXPECT_FALSE(en::CheckGemma4AssistantOutputs(0, 2816, heads, features));
  EXPECT_FALSE(en::CheckGemma4AssistantOutputs(2, 1024, heads, features));
  EXPECT_FALSE(en::CheckGemma4AssistantOutputs(1, 2816, heads, features));
  for (const auto invalid :
       {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()}) {
    for (auto* output : {&heads, &features}) {
      output->back() = invalid;
      EXPECT_FALSE(en::CheckGemma4AssistantOutputs(2, 2816, heads, features));
      output->back() = 0;
    }
  }
  EXPECT_TRUE(en::CheckGemma4AssistantOutputs(2, 2816, heads, features));
}
