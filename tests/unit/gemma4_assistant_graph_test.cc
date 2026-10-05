// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernels/ggml/gemma4_assistant_graph.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>

#include "gemma4_assistant_fixture.h"

namespace {
namespace kg = jitllm::kernels::ggml;
namespace md = jitllm::model;
namespace fixture = jitllm::test_support::gemma4;
struct Case {
  const md::Gemma4Profile& target;
  const md::Gemma4AssistantProfile& profile;
  md::Gemma4Binding tb;
  md::Gemma4AssistantBinding b;
  md::Gemma4StateLayout state;
  kg::Gemma4AssistantShape shape;
  explicit Case(std::uint32_t size, std::uint32_t prefix = 1024)
      : target(size == 26 ? md::Gemma4_26BA4B() : md::Gemma4_31B()),
        profile(size == 26 ? md::Gemma4Assistant26() : md::Gemma4Assistant31()) {
    tb = std::move(*md::BindGemma4(target, "gemma4", fixture::Resources(size)));
    b = std::move(
        *md::BindGemma4Assistant(profile, "gemma4-assistant", fixture::AssistantResources(size)));
    state = std::move(*md::Gemma4State(target, 4096, 128));
    shape.segments.push_back({0, prefix, std::min(state.local_cells, (prefix + 255) / 256 * 256),
                              (prefix + 255) / 256 * 256});
  }
};
TEST(Gemma4AssistantGraph, FourQOnlyBlocksKeepCompleteHeadAndRecurrentFeature) {
  for (const auto size : {26U, 31U}) {
    Case c(size);
    c.shape.segments.push_back({1, 1025, 1280, 1280});
    auto arena = kg::TensorArena::Create(kg::Gemma4AssistantGraphTensors(2));
    ASSERT_TRUE(arena);
    auto g =
        kg::BuildGemma4AssistantGraph(*arena, c.profile, c.b, c.target, c.tb, c.state, c.shape);
    ASSERT_TRUE(g);
    EXPECT_EQ(g->logits->ne[0], 262144);
    EXPECT_EQ(g->logits->ne[1], 2);
    EXPECT_EQ(g->next_features->ne[0], c.target.width);
    EXPECT_EQ(g->next_features->ne[1], 2);
    EXPECT_EQ(std::ranges::count_if(g->nodes, [](auto* t) { return t->op == GGML_OP_SET_ROWS; }),
              0);
    EXPECT_EQ(
        std::ranges::count_if(g->nodes, [](auto* t) { return t->op == GGML_OP_FLASH_ATTN_EXT; }),
        8);
    EXPECT_EQ(g->weights.size(), 49);
    EXPECT_TRUE(std::ranges::all_of(g->segments, [](const auto& seg) {
      return std::ranges::all_of(seg.caches, [](const auto& cache) {
        return cache.first->op == GGML_OP_NONE && cache.second->op == GGML_OP_NONE;
      });
    }));
  }
}
TEST(Gemma4AssistantGraph, FrozenReadWidthsDoNotIncludeUnwrittenCurrentBlock) {
  for (const auto prefix : {1U, 1023U, 1024U, 1025U, 2561U}) {
    Case c(26, prefix);
    EXPECT_TRUE(kg::CheckGemma4AssistantGraph(c.profile, c.b, c.target, c.tb, c.state, c.shape));
    c.shape.segments[0].global_n_kv += 256;
    EXPECT_FALSE(kg::CheckGemma4AssistantGraph(c.profile, c.b, c.target, c.tb, c.state, c.shape));
  }
}
TEST(Gemma4AssistantGraph, RefusesEmptyPrefixDuplicateOwnerAndMutatedBindingBeforeArena) {
  Case c(31);
  for (const auto prefix : {0U, 4096U, UINT32_MAX}) {
    auto s = c.shape;
    s.segments[0].prefix = prefix;
    EXPECT_FALSE(kg::CheckGemma4AssistantGraph(c.profile, c.b, c.target, c.tb, c.state, s));
  }
  c.shape.segments.push_back(c.shape.segments[0]);
  EXPECT_FALSE(kg::CheckGemma4AssistantGraph(c.profile, c.b, c.target, c.tb, c.state, c.shape));
  c.shape.segments.pop_back();
  c.b.layers[0].q.ne[0] = UINT64_MAX;
  auto arena = kg::TensorArena::Create(1);
  ASSERT_TRUE(arena);
  const auto used = arena->used();
  EXPECT_FALSE(
      kg::BuildGemma4AssistantGraph(*arena, c.profile, c.b, c.target, c.tb, c.state, c.shape));
  EXPECT_EQ(arena->used(), used);
}
TEST(Gemma4AssistantGraph, PrefixIsFreshInputNotAPlanDimension) {
  Case c(26, 1025);
  auto shape = c.shape;
  shape.segments[0].prefix = 1279;
  EXPECT_EQ(shape, c.shape);
  EXPECT_TRUE(kg::CheckGemma4AssistantGraph(c.profile, c.b, c.target, c.tb, c.state, shape));
}
}  // namespace
