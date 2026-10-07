// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/gemma2_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "expected_error.h"
#include "gemma2_fixture.h"
#include "kernels/ggml/fattn_owner.h"
#include "kernels/ggml/jitllm_ops.h"

namespace {
namespace md = jitllm::model;
namespace en = jitllm::engine;
namespace kg = jitllm::kernels::ggml;
static_assert(!std::is_copy_constructible_v<en::Gemma2HostInputs>);
static_assert(!std::is_copy_assignable_v<en::Gemma2HostInputs>);
static_assert(std::is_nothrow_move_constructible_v<en::Gemma2HostInputs>);
static_assert(std::is_nothrow_move_assignable_v<en::Gemma2HostInputs>);
struct Case {
  const md::Gemma2Profile& p = md::Gemma2_2B();
  md::Gemma2Binding binding =
      *md::BindGemma2(p, "gemma2", jitllm::test_support::gemma2::Resources());
  md::Gemma2StateLayout state = *md::Gemma2State(p, 8192, 16);
  std::array<std::int32_t, 3> a{1, 2, 3};
  std::array<std::int32_t, 1> b{4};
  std::array<md::Gemma2Segment, 2> segments{{{3, 4351, a}, {1, 7, b}}};
  md::Gemma2ChunkInputs input = *md::Gemma2Chunk(p, state, segments, true);
  kg::Gemma2ChunkShape shape{{{3, 3, 4351, 4608, 4352}, {1, 1, 7, 256, 256}}, 2};
  std::array<std::int32_t, 2> frontier{2, 3};
};
en::Gemma2Model Places(const Case& c, const kg::Gemma2Graph& g) {
  en::Gemma2Model m{.profile = &c.p,
                    .binding = &c.binding,
                    .state = &c.state,
                    .resources = {},
                    .slots = {},
                    .options = g.options};
  m.resources.resize(288);
  std::uint64_t address = std::uint64_t{1} << 36U;
  for (const auto& w : g.weights) {
    m.resources[w.resource.index] = {address, w.resource.readable};
    address += ((w.resource.readable + 255) / 256 + 1) * 256;
  }
  m.slots.resize(4);
  for (const auto& segment : g.segments) {
    m.slots[segment.shape.slot] = {address, c.state.bytes};
    address += c.state.bytes + 256;
  }
  return m;
}
kg::DeviceChoices Choices() {
  kg::DeviceChoices out;
  out.quant = [](const auto*) { return kg::QuantMulMatPath::kTile; };
  out.mul_mat = [](const auto*) { return kg::MulMatPath::kCublas; };
  return out;
}

TEST(Gemma2Plan, GreedyPlanRetainsArgmaxOutputAcrossPlacementAndRejectsDetachedOutput) {
  Case c;
  c.shape.greedy = true;
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(graph);
  const auto model = Places(c, *graph);
  const auto measured = en::PlanGemma2Chunk(model, c.shape, Choices(), 0, 0);
  ASSERT_TRUE(measured) << *jitllm::test_support::Failed(measured);
  const auto placed = en::PlanGemma2Chunk(model, c.shape, Choices(), std::uint64_t{1} << 53U,
                                          (*measured)->placement.extent);
  ASSERT_TRUE(placed) << *jitllm::test_support::Failed(placed);
  EXPECT_EQ((*placed)->plan.steps.back().implementation, kg::kArgmaxName);
  EXPECT_EQ((*placed)->plan.steps.back().nodes[0], (*placed)->graph.greedy);
  EXPECT_NE((*placed)->graph.greedy->data, nullptr);
  EXPECT_TRUE(kg::CheckArgmax((*placed)->graph.greedy));
  graph->greedy = ggml_new_tensor_1d(arena->context(), GGML_TYPE_I32, 2);
  EXPECT_FALSE(en::BindGemma2Weights(model, *graph));
}

TEST(Gemma2Plan, BothPlacementPassesRetainPrimitiveAttentionAndChargeEverySource) {
  Case c;
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(graph);
  const auto model = Places(c, *graph);
  const auto choices = Choices();
  auto measured = en::PlanGemma2Chunk(model, c.shape, choices, 0, 0);
  ASSERT_TRUE(measured) << *jitllm::test_support::Failed(measured);
  EXPECT_GT((*measured)->placement.extent, 0);
  const auto base = std::uint64_t{1} << 53U;
  EXPECT_FALSE(
      en::PlanGemma2Chunk(model, c.shape, choices, base, (*measured)->placement.extent - 1));
  auto placed = en::PlanGemma2Chunk(model, c.shape, choices, base, (*measured)->placement.extent);
  ASSERT_TRUE(placed) << *jitllm::test_support::Failed(placed);
  EXPECT_TRUE(std::ranges::equal(
      (*measured)->plan.steps, (*placed)->plan.steps, [](const auto& a, const auto& b) {
        return a.operation == b.operation && a.implementation == b.implementation &&
               a.nodes.size() == b.nodes.size() && a.lane == b.lane;
      }));
  EXPECT_GE((*placed)->arena->graph_capacity(), kg::Gemma2GraphTensors(c.p, 2));
  EXPECT_GE(en::PlannedHostBytes(**placed), (*placed)->arena->bytes());
  std::size_t attention = 0, writes = 0, norms = 0;
  for (const auto& step : (*placed)->plan.steps) {
    attention += step.implementation == kg::kFlashAttnMmaGqa2Name ? 1U : 0U;
    writes += step.implementation == kg::kSetRowsName ? 1U : 0U;
    norms += step.implementation == kg::kRmsNormMulUnfused ? 1U : 0U;
    EXPECT_NE(step.implementation, kg::kFlashAttnOwnersName);
    EXPECT_NE(step.implementation, kg::kRmsNormMulFused);
  }
  EXPECT_EQ(attention, 52);
  EXPECT_EQ(writes, 104);
  EXPECT_GT(norms, 0);
  std::uint64_t copied = 0;
  for (const auto* input : (*placed)->graph.inputs)
    copied += (ggml_nbytes(input) + 255) / 256 * 256;
  EXPECT_EQ((*placed)->inputs_bytes, copied);
}

TEST(Gemma2Plan, RegionRefusalIsAtomicAndProtectsUnusedPeerStorage) {
  Case c;
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(graph);
  const auto model = Places(c, *graph);
  auto bad = model;
  bad.slots[3].address = bad.slots[1].address;
  EXPECT_FALSE(en::BindGemma2Weights(bad, *graph));
  EXPECT_EQ(graph->weights[0].tensor->data, nullptr);
  EXPECT_EQ(graph->segments[0].caches[0].first->data, nullptr);
  bad = model;
  --bad.resources[graph->weights.back().resource.index].bytes;
  EXPECT_FALSE(en::BindGemma2Weights(bad, *graph));
  EXPECT_EQ(graph->weights[0].tensor->data, nullptr);
  bad = model;
  bad.slots[3].address = UINT64_MAX - 255;
  EXPECT_FALSE(en::BindGemma2Weights(bad, *graph));
  bad = model;
  bad.slots[0] = bad.slots[3];  // Occupied unselected peer is still live.
  EXPECT_FALSE(en::BindGemma2Weights(bad, *graph));
  const auto scratch = en::ScratchArenaBytes();
  EXPECT_FALSE(en::PlanGemma2Chunk(model, c.shape, Choices(),
                                   model.slots[3].address + c.state.tensors.back().offset, 256));
  EXPECT_EQ(en::ScratchArenaBytes(), scratch);
  auto* cache = graph->segments[1].caches[0].first;
  graph->segments[1].caches[0].first = graph->segments[0].caches[0].first;
  EXPECT_FALSE(en::BindGemma2Weights(model, *graph));
  EXPECT_EQ(graph->weights[0].tensor->data, nullptr);
  graph->segments[1].caches[0].first = cache;
  const auto weight_stride = graph->weights[0].tensor->nb[1];
  ++graph->weights[0].tensor->nb[1];
  EXPECT_FALSE(en::BindGemma2Weights(model, *graph));
  EXPECT_EQ(graph->weights[0].tensor->data, nullptr);
  graph->weights[0].tensor->nb[1] = weight_stride;
  EXPECT_TRUE(en::BindGemma2Weights(model, *graph));
  EXPECT_NE(graph->segments[0].caches[0].first->data, graph->segments[1].caches[0].first->data);
  // A successful initial binding cannot justify detached mutable rebind lists.
  std::vector<std::pair<ggml_tensor*, void*>> addresses;
  for (const auto& leaf : graph->weights) addresses.emplace_back(leaf.tensor, leaf.tensor->data);
  for (const auto& segment : graph->segments)
    for (const auto& [key, value] : segment.caches) {
      if (key != nullptr) addresses.emplace_back(key, key->data);
      if (value != nullptr) addresses.emplace_back(value, value->data);
    }
  for (auto* input : graph->inputs) addresses.emplace_back(input, input->data);
  auto moved = model;
  for (auto* regions : {&moved.resources, &moved.slots})
    for (auto& region : *regions)
      if (region.bytes != 0) region.address += 4096;
  const auto refuses_without_changes = [&] {
    EXPECT_FALSE(en::BindGemma2Weights(moved, *graph));
    for (const auto& [tensor, data] : addresses) EXPECT_EQ(tensor->data, data);
  };
  ASSERT_TRUE(arena->Reserve(3));
  auto leaves = graph->weights;
  graph->weights.erase(graph->weights.begin());
  refuses_without_changes();
  graph->weights = leaves;
  auto* original = graph->weights[0].tensor;
  auto* replacement =
      ggml_new_tensor_2d(arena->context(), original->type, original->ne[0], original->ne[1]);
  graph->weights[0].tensor = replacement;
  refuses_without_changes();
  EXPECT_EQ(replacement->data, nullptr);
  graph->weights = leaves;
  cache = graph->segments[0].caches[0].first;
  replacement = ggml_new_tensor_2d(arena->context(), cache->type, cache->ne[0], cache->ne[1]);
  graph->segments[0].caches[0].first = replacement;
  refuses_without_changes();
  EXPECT_EQ(replacement->data, nullptr);
  graph->segments[0].caches[0].first = cache;
  original = graph->tokens;
  replacement = ggml_new_tensor_1d(arena->context(), original->type, original->ne[0]);
  graph->tokens = replacement;
  graph->inputs[0] = replacement;
  refuses_without_changes();
  EXPECT_EQ(replacement->data, nullptr);
  graph->tokens = original;
  graph->inputs[0] = original;
  EXPECT_TRUE(en::BindGemma2Weights(moved, *graph));
  EXPECT_NE(graph->weights[0].tensor->data, addresses[0].second);
}

TEST(Gemma2Plan, FreshRaggedSourcesAreFundedPaddedAndRevalidatedForReuse) {
  Case c;
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(graph);
  auto bytes = en::Gemma2SourceBytes(*graph);
  ASSERT_TRUE(bytes) << *jitllm::test_support::Failed(bytes);
  EXPECT_FALSE(en::Gemma2Sources(*graph, c.input, c.frontier, {}, *bytes - 1));
  auto sources = en::Gemma2Sources(*graph, c.input, c.frontier, {}, *bytes);
  ASSERT_TRUE(sources) << *jitllm::test_support::Failed(sources);
  EXPECT_EQ(sources->sources.size(), graph->inputs.size());
  ASSERT_EQ(sources->masks.size(), 4);
  for (std::size_t i = 0; i < c.input.segments.size(); ++i) {
    const auto& input = c.input.segments[i];
    for (std::size_t local = 0; local < 2; ++local) {
      const auto& mask = sources->masks[i * 2 + local];
      const auto& logical = local == 0 ? input.global_mask : input.local_mask;
      ASSERT_GE(mask.size(), logical.size());
      EXPECT_TRUE(std::equal(logical.begin(), logical.end(), mask.begin()));
      EXPECT_TRUE(std::all_of(mask.begin() + static_cast<std::ptrdiff_t>(logical.size()),
                              mask.end(), [](const auto value) { return value == 0xFC00; }));
    }
  }
  auto moved = std::move(*sources);
  sources = std::unexpected(std::string("original owner retired"));
  const auto staged_address = [&](const ggml_tensor* tensor) {
    const auto found = std::ranges::find_if(
        moved.sources, [tensor](const auto& source) { return source.first == tensor; });
    return found == moved.sources.end() ? nullptr : found->second;
  };
  EXPECT_EQ(staged_address(graph->out_ids), moved.out_ids.data());
  for (std::size_t i = 0; i < graph->segments.size(); ++i) {
    EXPECT_EQ(staged_address(graph->segments[i].global_mask), moved.masks[i * 2].data());
    EXPECT_EQ(staged_address(graph->segments[i].local_mask), moved.masks[i * 2 + 1].data());
  }
  en::Gemma2HostInputs assigned;
  assigned = std::move(moved);
  EXPECT_EQ(assigned.sources[2].second, assigned.out_ids.data());
  EXPECT_EQ(assigned.sources[5].second, assigned.masks[0].data());
  const std::array<md::Gemma2Segment, 2> advanced{{{3, 4352, c.a}, {1, 8, c.b}}};
  auto fresh = md::Gemma2Chunk(c.p, c.state, advanced, true);
  ASSERT_TRUE(fresh);
  EXPECT_TRUE(en::Gemma2Sources(*graph, *fresh, c.frontier, {}, *bytes));
  // This family has host masks: unmasked descriptors alone cannot launch,
  // even though they suffice for placeless shape construction.
  auto unmasked = md::Gemma2Chunk(c.p, c.state, advanced, false);
  ASSERT_TRUE(unmasked);
  EXPECT_FALSE(en::Gemma2Sources(*graph, *unmasked, c.frontier, {}, *bytes));
  const auto with_masks = md::Gemma2HostInputBytes(c.p, c.state, advanced, true);
  const auto without_masks = md::Gemma2HostInputBytes(c.p, c.state, advanced, false);
  ASSERT_TRUE(with_masks);
  ASSERT_TRUE(without_masks);
  EXPECT_GT(*with_masks, *without_masks);
  auto malformed = *fresh;
  malformed.positions[0] = 4351;
  EXPECT_FALSE(en::Gemma2Sources(*graph, malformed, c.frontier, {}, *bytes));
  malformed = *fresh;
  malformed.segments[0].local_mask[0] = 0xFC00;  // Wrapped row4352 is visible.
  EXPECT_FALSE(en::Gemma2Sources(*graph, malformed, c.frontier, {}, *bytes));
  auto bad_frontier = c.frontier;
  bad_frontier[0] = 4;
  EXPECT_FALSE(en::Gemma2Sources(*graph, c.input, bad_frontier, {}, *bytes));
  auto* position = graph->positions;
  graph->positions = graph->tokens;
  EXPECT_FALSE(en::Gemma2SourceBytes(*graph));
  graph->positions = position;
  ++graph->segments[1].first_row;
  EXPECT_FALSE(en::Gemma2SourceBytes(*graph));
}

TEST(Gemma2Plan, StateOnlyAndHiddenDiagnosticSourcesRetainTheirOwnContracts) {
  Case c;
  c.shape.output_mode = kg::Gemma2OutputMode::kStateOnly;
  c.shape.outputs = 0;
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape);
  ASSERT_TRUE(graph);
  auto measured = en::PlanGemma2Chunk(Places(c, *graph), c.shape, Choices(), 0, 0);
  ASSERT_TRUE(measured) << *jitllm::test_support::Failed(measured);
  EXPECT_EQ((*measured)->graph.hidden, nullptr);
  EXPECT_EQ((*measured)->graph.logits, nullptr);
  auto bytes = en::Gemma2SourceBytes((*measured)->graph);
  ASSERT_TRUE(bytes);
  EXPECT_TRUE(en::Gemma2Sources((*measured)->graph, c.input, {}, {}, *bytes));
  EXPECT_FALSE(en::Gemma2Sources((*measured)->graph, c.input, c.frontier, {}, *bytes));
  arena->Reset();
  c.shape.output_mode = kg::Gemma2OutputMode::kHead;
  const kg::Gemma2GraphOptions options{
      .first_layer = 5, .layer_count = 1, .hidden_input = true, .head = false};
  graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape, options);
  ASSERT_TRUE(graph);
  EXPECT_EQ(graph->tokens, nullptr);
  EXPECT_EQ(graph->hidden->ne[1], 4);
  EXPECT_TRUE(en::PlanGemma2Chunk(Places(c, *graph), c.shape, Choices(), 0, 0));
  bytes = en::Gemma2SourceBytes(*graph);
  ASSERT_TRUE(bytes);
  const std::vector<float> hidden(std::size_t{c.p.width} * c.input.tokens.size());
  EXPECT_TRUE(en::Gemma2Sources(*graph, c.input, {}, hidden, *bytes));
  EXPECT_FALSE(en::Gemma2Sources(*graph, c.input, {}, {}, *bytes));
}
}  // namespace
