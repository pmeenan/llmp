// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/gemma4_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "base/check.h"
#include "expected_error.h"
#include "gemma4_fixture.h"
#include "kernels/ggml/jitllm_ops.h"

namespace {
namespace en = jitllm::engine;
namespace kg = jitllm::kernels::ggml;
namespace md = jitllm::model;
namespace fixture = jitllm::test_support::gemma4;
struct Case {
  const md::Gemma4Profile& p = md::Gemma4_26BA4B();
  md::Gemma4Binding binding;
  md::Gemma4StateLayout state;
  md::Gemma4ChunkInputs input;
  kg::Gemma4ChunkShape shape;
  std::vector<std::int32_t> frontier;
  explicit Case(std::uint32_t slots = 1, std::uint32_t past = 0) {
    binding = std::move(*md::BindGemma4(p, "gemma4", fixture::Resources(26)));
    state = std::move(*md::Gemma4State(p, 4096, 16));
    Set(slots, past);
  }
  void Set(std::uint32_t slots, std::uint32_t past, bool masks = true) {
    const std::array<std::int32_t, 2> tokens{1, 2};
    std::vector<md::Gemma4Segment> segments;
    for (std::uint32_t i = 0; i < slots; ++i) segments.push_back({i, past + i * 256, tokens});
    input = std::move(*md::Gemma4Chunk(p, state, segments, masks));
    shape = {};
    frontier.clear();
    for (const auto& s : input.segments) {
      shape.segments.push_back({s.slot, s.rows, s.n_past, s.global_n_kv, s.local_n_kv});
      frontier.push_back(static_cast<std::int32_t>(s.first_row + s.rows - 1));
    }
    shape.outputs = slots;
  }
};
en::Gemma4Model Places(const Case& c, const kg::Gemma4Graph& g) {
  en::Gemma4Model m;
  m.profile = &c.p;
  m.binding = &c.binding;
  m.state = &c.state;
  m.options = g.options;
  std::uint64_t address = std::uint64_t{1} << 36U;
  for (const auto& w : g.weights) {
    auto& domain = w.resource.expert_array ? m.arrays : m.resources;
    domain.resize(std::max(domain.size(), std::size_t{w.resource.index} + 1));
    const auto bytes =
        w.resource.readable +
        (w.resource.expert_array ? (std::uint64_t{c.p.experts} - 1) * w.tensor->nb[2] : 0);
    domain[w.resource.index] = {address, bytes};
    address += ((bytes + 255) / 256 + 1) * 256;
  }
  for (const auto& seg : g.segments) {
    m.slots.resize(std::max(m.slots.size(), std::size_t{seg.shape.slot} + 1));
    m.slots[seg.shape.slot] = {address, c.state.bytes};
    address += c.state.bytes + 256;
  }
  return m;
}

TEST(Gemma4Plan, StorePolicyKeepsAllRotatedStorageFundedAndWritesDiagnosticKeeps) {
  Case c(2, 1279);
  c.Set(2, 1279, false);
  kg::Gemma4GraphOptions options;
  options.device_masks = true;
  options.rope_store = true;
  auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
  ASSERT_TRUE(graph);
  auto model = Places(c, *graph);
  kg::DeviceChoices choices;
  choices.quant = [](const auto*) { return kg::QuantMulMatPath::kTile; };
  choices.mul_mat = [](const auto*) { return kg::MulMatPath::kCublas; };
  auto measured = en::PlanGemma4Chunk(model, c.shape, choices, 0, 0);
  ASSERT_TRUE(measured) << *jitllm::test_support::Failed(measured);
  constexpr auto address = std::uint64_t{1} << 53U;
  auto placed =
      en::PlanGemma4Chunk(model, c.shape, choices, address, (*measured)->placement.extent);
  ASSERT_TRUE(placed) << *jitllm::test_support::Failed(placed);
  const auto& p = **placed;
  // Independent graph arenas have different descriptor identities; compare
  // their dispatch. PlaceAndPlan itself checks identity before/after binding.
  EXPECT_TRUE(
      std::ranges::equal((*measured)->plan.steps, p.plan.steps, [](const auto& a, const auto& b) {
        return a.operation == b.operation && a.implementation == b.implementation &&
               a.nodes.size() == b.nodes.size();
      }));
  EXPECT_EQ(std::ranges::count_if(p.plan.steps,
                                  [](const auto& step) {
                                    return step.operation ==
                                           jitllm::execution::Operation::kRopeSetRows;
                                  }),
            c.p.layers * 2);
  // Existing CheckCoverage checks every original descriptor and its sources.
  // Even unwritten intermediates remain placed; every fused RoPE and flattening
  // view therefore still lies in the activation region it checks.
  for (const auto& step : p.plan.steps)
    if (step.operation == jitllm::execution::Operation::kRopeSetRows) {
      for (const auto* t : {step.nodes[0], step.nodes[1]->src[0]}) {
        const auto data = reinterpret_cast<std::uintptr_t>(t->data);
        EXPECT_GE(data, address);
        EXPECT_LE(data - address + ggml_nbytes(t), p.placement.extent);
      }
      EXPECT_TRUE(kg::CheckRopeSetRows(step.nodes[0], step.nodes[1]));
    }
  const std::array<std::string, 1> keep{"blk.0.slot.0.k_rope"};
  auto protected_plan = en::PlanGemma4Chunk(model, c.shape, choices, 0, 0, keep);
  ASSERT_TRUE(protected_plan) << *jitllm::test_support::Failed(protected_plan);
  auto* kept = (*protected_plan)->graph.Named(keep[0]);
  EXPECT_EQ(std::ranges::count_if((*protected_plan)->plan.steps,
                                  [kept](const auto& step) {
                                    return step.operation == jitllm::execution::Operation::kRope &&
                                           step.nodes[0] == kept;
                                  }),
            1);
  EXPECT_EQ(std::ranges::count_if((*protected_plan)->plan.steps,
                                  [](const auto& step) {
                                    return step.operation ==
                                           jitllm::execution::Operation::kRopeSetRows;
                                  }),
            c.p.layers * 2 - 1);
}

TEST(Gemma4Plan, DeviceMaskSourcesStayBoundedAndAuthenticateFreshPositionsAndProducers) {
  for (const auto slots : {1U, 2U, 4U, 16U}) {
    Case c;
    c.state = std::move(*md::Gemma4State(c.p, 4096, 32));
    c.Set(slots, 0, false);
    auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, slots));
    ASSERT_TRUE(arena);
    kg::Gemma4GraphOptions options;
    options.device_masks = true;
    auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
    ASSERT_TRUE(g);
    auto bytes = en::Gemma4SourceBytes(*g);
    ASSERT_TRUE(bytes);
    EXPECT_LT(*bytes, 1024);
    EXPECT_FALSE(en::Gemma4Sources(*g, c.input, c.frontier, {}, *bytes - 1));
    auto sources = en::Gemma4Sources(*g, c.input, c.frontier, {}, *bytes);
    ASSERT_TRUE(sources);
    EXPECT_TRUE(sources->masks.empty());
    EXPECT_EQ(sources->sources.size(), g->inputs.size());
    for (const auto& segment : g->segments)
      for (const auto* mask : {segment.global_mask, segment.local_mask}) {
        EXPECT_TRUE(kg::Gemma4MaskFits(mask));
        EXPECT_EQ(std::ranges::find(g->inputs, mask), g->inputs.end());
        EXPECT_EQ(std::count(g->nodes.begin(), g->nodes.end(), mask), 1);
      }
    auto bad = c.input;
    ++bad.positions[0];
    EXPECT_FALSE(en::Gemma4Sources(*g, bad, c.frontier, {}, *bytes));
    bad = c.input;
    ++bad.segments[0].local_cells[0];
    EXPECT_FALSE(en::Gemma4Sources(*g, bad, c.frontier, {}, *bytes));
    bad = c.input;
    bad.tokens[0] = -1;
    EXPECT_FALSE(en::Gemma4Sources(*g, bad, c.frontier, {}, *bytes));
    g->options.device_masks = false;
    EXPECT_FALSE(en::Gemma4SourceBytes(*g));
    g->options.device_masks = true;
    auto* producer = g->segments[0].local_mask;
    producer->src[0] = g->tokens;
    EXPECT_FALSE(en::Gemma4SourceBytes(*g));
    producer->src[0] = g->positions;
    EXPECT_TRUE(en::Gemma4SourceBytes(*g));
  }
}

TEST(Gemma4Plan, DeviceMasksReuseTheSameGraphAcrossARingWrap) {
  Case c(1, 2559);
  c.Set(1, 2559, false);
  auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 1));
  ASSERT_TRUE(arena);
  kg::Gemma4GraphOptions options;
  options.device_masks = true;
  auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
  ASSERT_TRUE(g);
  const auto bytes = en::Gemma4SourceBytes(*g);
  ASSERT_TRUE(bytes);
  ASSERT_TRUE(en::Gemma4Sources(*g, c.input, c.frontier, {}, *bytes));
  c.Set(1, 2561, false);
  EXPECT_EQ(c.shape, g->shape);
  const auto after = en::Gemma4Sources(*g, c.input, c.frontier, {}, *bytes);
  ASSERT_TRUE(after);
  EXPECT_TRUE(after->masks.empty());
  EXPECT_EQ(c.input.segments[0].local_cells, (std::vector<std::int64_t>{1, 2}));
  EXPECT_EQ(g->segments[0].shape.n_past, 2559U);
}

TEST(Gemma4Plan, BoundedBindingRefusesShortOverlappingAndOverflowingRegionsWithoutPartialBind) {
  Case c(2);
  auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(g);
  auto m = Places(c, *g);
  auto bad = m;
  bad.arrays[0].bytes = 1;
  EXPECT_FALSE(en::BindGemma4Weights(bad, *g));
  EXPECT_EQ(g->weights.front().tensor->data, nullptr);
  bad = m;
  bad.slots[1] = bad.slots[0];
  EXPECT_FALSE(en::BindGemma4Weights(bad, *g));
  EXPECT_EQ(g->weights.front().tensor->data, nullptr);
  bad = m;
  bad.slots[0].address = UINT64_MAX - 255;
  EXPECT_FALSE(en::BindGemma4Weights(bad, *g));
  bad = m;
  bad.slots[0].address = bad.resources[g->weights[0].resource.index].address;
  EXPECT_FALSE(en::BindGemma4Weights(bad, *g));
  auto reordered = c.binding;
  std::swap(reordered.layers[0].q.index, reordered.layers[1].q.index);
  ASSERT_TRUE(md::CheckGemma4Binding(c.p, reordered));
  bad = m;
  bad.binding = &reordered;
  EXPECT_FALSE(en::BindGemma4Weights(bad, *g));
  EXPECT_EQ(g->weights.front().tensor->data, nullptr);
  bad = m;
  bad.slots.push_back(bad.slots[0]);  // Occupied peer slot is not in this graph.
  EXPECT_FALSE(en::BindGemma4Weights(bad, *g));
  bad = m;
  bad.resources.push_back(bad.slots[0]);  // Unused immutable storage is live too.
  EXPECT_FALSE(en::BindGemma4Weights(bad, *g));
  EXPECT_TRUE(en::BindGemma4Weights(m, *g));
  EXPECT_NE(g->segments[0].caches[5].first->data, g->segments[0].caches[5].second->data);
  EXPECT_NE(g->segments[0].caches[0].first->data, g->segments[1].caches[0].first->data);
}
TEST(Gemma4Plan, ActivationRegionRefusesUnusedRetainedStorageBeforeArenaGrowth) {
  Case c;
  const kg::Gemma4GraphOptions options{
      .expert_stride = {}, .first_layer = 0, .layer_count = 1, .hidden_input = true, .head = false};
  c.shape.outputs = 0;
  auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 1));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape, options);
  ASSERT_TRUE(graph);
  auto model = Places(c, *graph);
  const auto scratch = en::ScratchArenaBytes();
  // Layer29 never appears in this diagnostic graph; its retained KV is live.
  const auto inactive = model.slots[0].address + c.state.tensors[58].offset;
  EXPECT_FALSE(en::PlanGemma4Chunk(model, c.shape, {}, inactive, 256));
  model.slots.push_back({std::uint64_t{1} << 55U, c.state.bytes});
  EXPECT_FALSE(en::PlanGemma4Chunk(model, c.shape, {}, model.slots[1].address, 256));
  model.resources.push_back({std::uint64_t{1} << 54U, 4096});
  EXPECT_FALSE(en::PlanGemma4Chunk(model, c.shape, {}, model.resources.back().address, 256));
  EXPECT_FALSE(en::PlanGemma4Chunk(model, c.shape, {}, UINT64_MAX - 255, 512));
  EXPECT_FALSE(en::PlanGemma4Chunk(model, c.shape, {}, (std::uint64_t{1} << 53U) + 1, 256));
  EXPECT_FALSE(en::PlanGemma4Chunk(model, c.shape, {}, std::uint64_t{1} << 53U, 0));
  EXPECT_EQ(en::ScratchArenaBytes(), scratch);
}
TEST(Gemma4Plan, ReferenceSourcesAreFundedPaddedAndIsolatedBeforeStaging) {
  for (const auto slots : {1U, 2U, 4U}) {
    Case c(slots);
    auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, slots));
    ASSERT_TRUE(arena);
    auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape);
    ASSERT_TRUE(g);
    auto bytes = en::Gemma4SourceBytes(*g);
    ASSERT_TRUE(bytes);
    EXPECT_FALSE(en::Gemma4Sources(*g, c.input, c.frontier, {}, *bytes - 1));
    auto sources = en::Gemma4Sources(*g, c.input, c.frontier, {}, *bytes);
    ASSERT_TRUE(sources) << *jitllm::test_support::Failed(sources);
    EXPECT_EQ(sources->sources.size(), g->inputs.size());
    EXPECT_EQ(sources->masks.size(), slots * 2);
    for (std::uint32_t i = 0; i < slots; ++i) {
      for (const auto local : {false, true}) {
        const auto& mask = sources->masks[i * 2 + (local ? 1U : 0U)];
        const auto& host = local ? c.input.segments[i].local_mask : c.input.segments[i].global_mask;
        EXPECT_TRUE(std::equal(host.begin(), host.end(), mask.begin()));
        EXPECT_TRUE(std::all_of(mask.begin() + static_cast<std::ptrdiff_t>(host.size()), mask.end(),
                                [](auto x) { return x == 0xFC00; }));
      }
    }
    auto bad = c.input;
    ++bad.segments[0].local_cells[0];
    EXPECT_FALSE(en::Gemma4Sources(*g, bad, c.frontier, {}, *bytes));
    bad = c.input;
    bad.segments[0].global_mask[bad.segments[0].n_past + 1] = 0;
    EXPECT_FALSE(en::Gemma4Sources(*g, bad, c.frontier, {}, *bytes));
    bad = c.input;
    bad.segments[0].local_mask[0] ^= 0xFC00;
    EXPECT_FALSE(en::Gemma4Sources(*g, bad, c.frontier, {}, *bytes));
    bad = c.input;
    bad.tokens[0] = -1;
    EXPECT_FALSE(en::Gemma4Sources(*g, bad, c.frontier, {}, *bytes));
    auto frontier = c.frontier;
    frontier[0] = 999;
    EXPECT_FALSE(en::Gemma4Sources(*g, c.input, frontier, {}, *bytes));
  }
}
TEST(Gemma4Plan, FreshPositionsReuseTheSameGraphAcrossARingWrap) {
  Case c(1, 2559);
  auto arena = kg::TensorArena::Create(kg::Gemma4GraphTensors(c.p, 1));
  ASSERT_TRUE(arena);
  auto g = kg::BuildGemma4Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(g);
  auto bytes = en::Gemma4SourceBytes(*g);
  ASSERT_TRUE(bytes);
  auto before = en::Gemma4Sources(*g, c.input, c.frontier, {}, *bytes);
  ASSERT_TRUE(before);
  EXPECT_EQ(c.input.segments[0].local_cells, (std::vector<std::int64_t>{1279, 0}));
  c.Set(1, 2561);
  EXPECT_EQ(c.shape, g->shape);
  auto after = en::Gemma4Sources(*g, c.input, c.frontier, {}, *bytes);
  ASSERT_TRUE(after) << *jitllm::test_support::Failed(after);
  EXPECT_EQ(c.input.segments[0].local_cells, (std::vector<std::int64_t>{1, 2}));
  auto stale = c.input;
  stale.segments[0].local_mask[(stale.segments[0].n_past - c.p.window) % c.state.local_cells] = 0;
  EXPECT_FALSE(en::Gemma4Sources(*g, stale, c.frontier, {}, *bytes));
  EXPECT_EQ(g->segments[0].shape.n_past, 2559U);  // no captured scalar position
}
TEST(Gemma4Plan, MalformedModelRefusesBeforeSizedArenaGrowth) {
  Case c;
  auto p = c.p;
  p.layers = UINT32_MAX;
  en::Gemma4Model m;
  m.profile = &p;
  m.binding = &c.binding;
  m.state = &c.state;
  const auto bytes = en::ScratchArenaBytes();
  EXPECT_FALSE(en::PlanGemma4Chunk(m, c.shape, {}, 0, 0));
  EXPECT_EQ(en::ScratchArenaBytes(), bytes);
}
}  // namespace
