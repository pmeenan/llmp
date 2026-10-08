// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/gemma2_plan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
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
TEST(Gemma2Plan, PackedPrefillFundsRealCacheCopiesAndTwoSequenceMasks) {
  Case c;
  c.state = *md::Gemma2State(c.p, 8192, 128);
  c.shape = {{{0, 128, 0, 256, 256}, {1, 128, 0, 256, 256}}, 2};
  const kg::Gemma2GraphOptions options{
      .max_total_rows = 256, .narrow_final = true, .packed_prefill = true};
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape, options);
  ASSERT_TRUE(graph);
  const auto model = Places(c, *graph);
  auto measured = en::PlanGemma2Chunk(model, c.shape, Choices(), 0, 0);
  ASSERT_TRUE(measured) << *jitllm::test_support::Failed(measured);
  auto placed = en::PlanGemma2Chunk(model, c.shape, Choices(), std::uint64_t{1} << 53U,
                                    (*measured)->placement.extent);
  ASSERT_TRUE(placed) << *jitllm::test_support::Failed(placed);
  auto& g = (*placed)->graph;
  auto* attention = g.Named("blk.0.packed_prefill_attention");
  ASSERT_NE(attention, nullptr);
  EXPECT_FLOAT_EQ(std::bit_cast<float>(attention->op_params[2]), 50.0F);
  EXPECT_EQ(attention->src[0]->ne[1], 128);
  EXPECT_EQ(attention->src[0]->ne[3], 2);
  EXPECT_EQ(attention->src[3]->ne[1], 128);
  EXPECT_EQ(attention->src[3]->ne[3], 2);
  for (std::size_t i : {1U, 2U}) {
    const auto* packed = attention->src[i]->view_src;
    ASSERT_NE(packed, nullptr);
    EXPECT_EQ(packed->op, GGML_OP_CONCAT);
    EXPECT_EQ(packed->type, GGML_TYPE_F16);
    EXPECT_TRUE(ggml_is_contiguous(packed));
    EXPECT_EQ(ggml_nbytes(packed), 2U * 256U * 4U * 256U * sizeof(ggml_fp16_t));
    EXPECT_NE(packed->data, g.segments[0].caches[0].first->data);
    EXPECT_NE(packed->data, g.segments[1].caches[0].first->data);
    EXPECT_TRUE(kg::CheckConcat(packed));
  }
  std::size_t packed_steps = 0;
  for (const auto& step : (*placed)->plan.steps)
    if (step.implementation == kg::kFlashAttnMmaGqa2Name) ++packed_steps;
  EXPECT_EQ(packed_steps, 26U);
  const std::vector<std::int32_t> tokens(128, 2);
  const std::array<md::Gemma2Segment, 2> segments{{{0, 0, tokens}, {1, 0, tokens}}};
  auto input = md::Gemma2Chunk(c.p, c.state, segments, true, 256, 256);
  ASSERT_TRUE(input);
  const auto bytes = en::Gemma2SourceBytes(g);
  ASSERT_TRUE(bytes);
  const std::array<std::int32_t, 2> frontier{127, 255};
  EXPECT_TRUE(en::Gemma2Sources(g, *input, frontier, {}, *bytes));
  EXPECT_FALSE(en::Gemma2Sources(g, *input, frontier, {}, *bytes - 1));
}

TEST(Gemma2Plan, UnequalDecodePadsActivationsWithoutWideningStateOrHostSources) {
  Case c;
  c.state = *md::Gemma2State(c.p, 8192, 128);
  c.shape = {{{0, 1, 259, 512, 512}, {1, 1, 771, 1024, 1024}}, 2};
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape,
                                    {.narrow_final = true, .owner_decode = true});
  ASSERT_TRUE(graph);
  const auto model = Places(c, *graph);
  auto measured = en::PlanGemma2Chunk(model, c.shape, Choices(), 0, 0);
  ASSERT_TRUE(measured) << *jitllm::test_support::Failed(measured);
  auto placed = en::PlanGemma2Chunk(model, c.shape, Choices(), std::uint64_t{1} << 53U,
                                    (*measured)->placement.extent);
  ASSERT_TRUE(placed) << *jitllm::test_support::Failed(placed);
  auto& g = (*placed)->graph;
  EXPECT_EQ(g.segments[0].global_mask->ne[0], 512);
  EXPECT_EQ(g.segments[1].global_mask->ne[0], 1024);
  auto* attention = g.Named("blk.0.owner_attention");
  ASSERT_NE(attention, nullptr);
  auto owners = kg::FlashAttnOwnersFromNode(attention);
  ASSERT_TRUE(owners);
  EXPECT_EQ(owners->mask->ne[0], 1024);
  EXPECT_EQ(owners->mask->ne[1], 32);
  EXPECT_EQ(owners->k[0]->ne[1], 1024);
  const auto* short_copy = owners->k[0]->src[0];
  ASSERT_NE(short_copy, nullptr);
  EXPECT_EQ(short_copy->op, GGML_OP_CONCAT);
  EXPECT_EQ(short_copy->src[0]->ne[2], 512);  // actual cache prefix stays bounded
  EXPECT_EQ(short_copy->src[0]->view_src, g.segments[0].caches[0].first);
  auto* zero = short_copy->src[1];
  ASSERT_NE(zero, nullptr);
  EXPECT_EQ(zero->op, GGML_OP_FILL);
  EXPECT_EQ(zero->type, GGML_TYPE_F16);
  EXPECT_TRUE(kg::CheckFill(zero));
  for (const auto* source : zero->src) EXPECT_EQ(source, nullptr);
  zero->src[0] = ggml_new_tensor_1d(arena->context(), GGML_TYPE_F16, 1);
  EXPECT_FALSE(en::BindGemma2Weights(model, g));  // undeclared roots still refused
  zero->src[0] = nullptr;
  EXPECT_TRUE(en::BindGemma2Weights(model, g));
  std::size_t count = 0;
  for (const auto& step : (*placed)->plan.steps)
    count += step.implementation == kg::kFlashAttnOwnersName;
  EXPECT_EQ(count, 26U);
  const std::array<md::Gemma2Segment, 2> segments{{{0, 259, c.b}, {1, 771, c.b}}};
  const auto input = md::Gemma2Chunk(c.p, c.state, segments, true);
  ASSERT_TRUE(input);
  EXPECT_EQ(input->segments[0].global_n_kv, 512U);
  const auto bytes = en::Gemma2SourceBytes(g);
  ASSERT_TRUE(bytes);
  const std::array<std::int32_t, 2> frontier{0, 1};
  EXPECT_TRUE(en::Gemma2Sources(g, *input, frontier, {}, *bytes));
  EXPECT_FALSE(en::Gemma2Sources(g, *input, frontier, {}, *bytes - 1));
}

TEST(Gemma2Plan, BoundedOwnerRootsKeepActualViewsAndRecipeIdentity) {
  Case c;
  c.state = *md::Gemma2State(c.p, 8192, 128);
  c.shape = {{{0, 1, 259, 512, 512}, {1, 1, 771, 1024, 1024}}, 2};
  std::array<std::uint64_t, 2> extents{}, source_bytes{};
  for (const bool bounded : {false, true}) {
    auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
    ASSERT_TRUE(arena);
    auto graph = kg::BuildGemma2Graph(
        *arena, c.p, c.binding, c.state, c.shape,
        {.narrow_final = true, .owner_decode = true, .bounded_roots = bounded});
    ASSERT_TRUE(graph);
    auto model = Places(c, *graph);
    auto measured = en::PlanGemma2Chunk(model, c.shape, Choices(), 0, 0);
    ASSERT_TRUE(measured) << *jitllm::test_support::Failed(measured);
    extents[bounded] = (*measured)->placement.extent;
    auto placed = en::PlanGemma2Chunk(model, c.shape, Choices(), std::uint64_t{1} << 53U,
                                      (*measured)->placement.extent);
    ASSERT_TRUE(placed) << *jitllm::test_support::Failed(placed);
    auto& g = (*placed)->graph;
    std::size_t owners = 0, cache_copies = 0;
    for (const auto* node : g.nodes) {
      cache_copies += node->op == GGML_OP_CONCAT && node->ne[0] == 256 && node->ne[1] == 4;
      if (kg::JitllmOpOf(node) != kg::JitllmOp::kFlashAttnOwners) continue;
      ++owners;
      auto inputs = kg::FlashAttnOwnersFromNode(const_cast<ggml_tensor*>(node));
      ASSERT_TRUE(inputs);
      EXPECT_EQ(inputs->bounded_roots, bounded);
      EXPECT_EQ(inputs->mask->ne[0], 1024);
      EXPECT_EQ(inputs->k[0]->ne[1], bounded ? 512 : 1024);
      EXPECT_EQ(inputs->v[0]->ne[1], bounded ? 512 : 1024);
      EXPECT_EQ(inputs->k[1]->ne[1], 1024);
      if (bounded) EXPECT_NE(inputs->k[0]->view_src, nullptr);
    }
    EXPECT_EQ(owners, 26U);
    EXPECT_EQ(cache_copies, bounded ? 0U : 52U);
    EXPECT_EQ(g.segments[0].global_mask->ne[0], 512);
    EXPECT_EQ(g.segments[1].global_mask->ne[0], 1024);
    EXPECT_TRUE(en::BindGemma2Weights(model, g));
    g.options.bounded_roots = !bounded;
    EXPECT_FALSE(en::BindGemma2Weights(model, g));
    g.options.bounded_roots = bounded;
    EXPECT_TRUE(en::BindGemma2Weights(model, g));
    const std::array<md::Gemma2Segment, 2> segments{{{0, 259, c.b}, {1, 771, c.b}}};
    const auto input = md::Gemma2Chunk(c.p, c.state, segments, true);
    ASSERT_TRUE(input);
    const auto bytes = en::Gemma2SourceBytes(g);
    ASSERT_TRUE(bytes);
    source_bytes[bounded] = *bytes;
    const std::array<std::int32_t, 2> frontier{0, 1};
    EXPECT_TRUE(en::Gemma2Sources(g, *input, frontier, {}, *bytes));
    EXPECT_FALSE(en::Gemma2Sources(g, *input, frontier, {}, *bytes - 1));
  }
  EXPECT_LT(extents[1], extents[0]);
  EXPECT_EQ(source_bytes[0], source_bytes[1]);
}

TEST(Gemma2Plan, JoinedPrefillFundsTotalRowsWithoutChangingSlotLayout) {
  Case c;
  c.state = *md::Gemma2State(c.p, 8192, 128);
  const std::vector<std::int32_t> tokens(128, 2);
  const std::array<md::Gemma2Segment, 2> segments{{{3, 4351, tokens}, {1, 4608, tokens}}};
  auto input = md::Gemma2Chunk(c.p, c.state, segments, true, 256, 256);
  ASSERT_TRUE(input);
  c.shape.segments.clear();
  for (const auto& segment : input->segments)
    c.shape.segments.push_back(
        {segment.slot, segment.rows, segment.n_past, segment.global_n_kv, segment.local_n_kv});
  EXPECT_FALSE(kg::CheckGemma2Graph(c.p, c.binding, c.state, c.shape));
  const kg::Gemma2GraphOptions options{.max_total_rows = 256, .narrow_final = true};
  auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
  ASSERT_TRUE(arena);
  auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, c.shape, options);
  ASSERT_TRUE(graph);
  EXPECT_EQ(graph->max_rows, 128U);
  EXPECT_EQ(graph->local_capacity, 4352U);
  EXPECT_EQ(graph->options.max_total_rows, 256U);
  const auto model = Places(c, *graph);
  auto measured = en::PlanGemma2Chunk(model, c.shape, Choices(), 0, 0);
  ASSERT_TRUE(measured);
  auto placed = en::PlanGemma2Chunk(model, c.shape, Choices(), std::uint64_t{1} << 53U,
                                    (*measured)->placement.extent);
  ASSERT_TRUE(placed);
  const auto bytes = en::Gemma2SourceBytes((*placed)->graph);
  ASSERT_TRUE(bytes);
  const std::array<std::int32_t, 2> frontier{127, 255};
  EXPECT_FALSE(en::Gemma2Sources((*placed)->graph, *input, frontier, {}, *bytes - 1));
  EXPECT_TRUE(en::Gemma2Sources((*placed)->graph, *input, frontier, {}, *bytes));
  auto changed = (*placed)->graph.options.max_total_rows;
  (*placed)->graph.options.max_total_rows = 128;
  EXPECT_FALSE(en::Gemma2SourceBytes((*placed)->graph));
  (*placed)->graph.options.max_total_rows = changed;
  EXPECT_TRUE(en::Gemma2SourceBytes((*placed)->graph));
  auto invalid = c.shape;
  invalid.segments[0].rows = 129;
  EXPECT_FALSE(kg::CheckGemma2Graph(c.p, c.binding, c.state, invalid, options));
}

TEST(Gemma2Plan, DeviceMasksBindFreshSegmentPositionsAndExcludeHostMatrices) {
  Case c;
  c.state = *md::Gemma2State(c.p, 8192, 128);
  const std::array<std::int32_t, 128> tokens{};
  EXPECT_FALSE(kg::Gemma2GraphOptions{}.device_masks);
  for (const auto [past, rows] :
       {std::pair{4351U, 1U}, std::pair{4224U, 128U}, std::pair{7936U, 128U}}) {
    const std::array segments{md::Gemma2Segment{0, past, std::span(tokens).first(rows)},
                              md::Gemma2Segment{1, past + rows, std::span(tokens).first(rows)}};
    auto input = md::Gemma2Chunk(c.p, c.state, segments, false, 256, 256);
    ASSERT_TRUE(input);
    kg::Gemma2ChunkShape shape;
    shape.outputs = 2;
    for (const auto& seg : input->segments)
      shape.segments.push_back({seg.slot, seg.rows, seg.n_past, seg.global_n_kv, seg.local_n_kv});
    auto arena = kg::TensorArena::Create(kg::Gemma2GraphTensors(c.p, 2));
    ASSERT_TRUE(arena);
    auto graph = kg::BuildGemma2Graph(*arena, c.p, c.binding, c.state, shape,
                                      {.max_total_rows = 256, .device_masks = true});
    ASSERT_TRUE(graph);
    auto model = Places(c, *graph);
    EXPECT_TRUE(en::BindGemma2Weights(model, *graph));
    auto planned = en::PlanGemma2Chunk(model, shape, Choices(), std::uint64_t{1} << 30U,
                                       std::uint64_t{1} << 29U);
    ASSERT_TRUE(planned);
    EXPECT_EQ(std::ranges::count_if(
                  (*planned)->plan.steps,
                  [](const auto& step) { return step.implementation == kg::kGemma4MaskName; }),
              4);
    std::uint64_t mask_bytes = 0;
    for (const auto& seg : (*planned)->graph.segments)
      for (const auto* mask : {seg.global_mask, seg.local_mask}) {
        EXPECT_NE(mask->data, nullptr);
        mask_bytes += ggml_nbytes(mask);
      }
    EXPECT_GE((*planned)->placement.extent, mask_bytes);
    const auto bytes = en::Gemma2SourceBytes(*graph);
    ASSERT_TRUE(bytes);
    EXPECT_LT(*bytes, 1024);
    const std::array<std::int32_t, 2> frontier{static_cast<std::int32_t>(rows - 1),
                                               static_cast<std::int32_t>(2 * rows - 1)};
    auto source = en::Gemma2Sources(*graph, *input, frontier, {}, *bytes);
    ASSERT_TRUE(source);
    EXPECT_TRUE(source->masks.empty());
    EXPECT_EQ(source->sources.size(), graph->inputs.size());
    EXPECT_FALSE(en::Gemma2Sources(*graph, *input, frontier, {}, *bytes - 1));
    for (const auto& seg : graph->segments) {
      for (auto* mask : {seg.global_mask, seg.local_mask}) {
        EXPECT_TRUE(kg::Gemma4MaskFits(mask));
        EXPECT_EQ(std::ranges::count(graph->nodes, mask), 1);
        EXPECT_FALSE(std::ranges::contains(graph->inputs, mask));
        EXPECT_EQ(kg::JitllmOpInt(mask, 0), seg.first_row);
        EXPECT_EQ(kg::JitllmOpInt(mask, 1), rows);
        const std::vector<std::int32_t> saved(std::begin(mask->op_params),
                                              std::end(mask->op_params));
        for (std::uint32_t index = 0; index < 5; ++index) {
          std::array<std::int32_t, 5> params{};
          for (std::uint32_t i = 0; i < 5; ++i)
            params[i] = kg::JitllmOpInt(mask, static_cast<int>(i));
          ++params[index];
          auto* wrong = kg::Gemma4Mask(arena->context(), graph->positions, mask->ne[0], params[0],
                                       params[1], params[2], params[3], params[4]);
          std::ranges::copy(wrong->op_params, mask->op_params);
          EXPECT_FALSE(en::Gemma2SourceBytes(*graph));
          std::ranges::copy(saved, mask->op_params);
        }
        mask->src[0] = graph->tokens;
        EXPECT_FALSE(en::Gemma2SourceBytes(*graph));
        mask->src[0] = graph->positions;
        mask->src[2] = graph->tokens;
        EXPECT_FALSE(en::Gemma2SourceBytes(*graph));
        mask->src[2] = nullptr;
        graph->inputs.push_back(mask);
        EXPECT_FALSE(en::Gemma2SourceBytes(*graph));
        graph->inputs.pop_back();
      }
    }
    for (const auto kind : {0, 1, 2, 3, 4}) {
      auto bad = *input;
      if (kind == 0) ++bad.positions[rows];
      if (kind == 1) ++bad.segments[1].global_cells.back();
      if (kind == 2) ++bad.segments[1].local_cells.back();
      if (kind == 3) bad.tokens.back() = -1;
      if (kind == 4) bad.segments[0].global_mask.push_back(0);
      EXPECT_FALSE(en::Gemma2Sources(*graph, bad, frontier, {}, *bytes));
    }
    for (auto& seg : input->segments) {
      const std::int32_t delta = (seg.n_past + rows) % 256 == 0 ? -1 : 1;
      seg.n_past = static_cast<std::uint32_t>(static_cast<std::int64_t>(seg.n_past) + delta);
      for (std::uint32_t r = 0; r < rows; ++r) {
        input->positions[seg.first_row + r] += delta;
        seg.global_cells[r] += delta;
        seg.local_cells[r] = (seg.n_past + r) % c.state.local_cells;
      }
    }
    // Same read bucket and graph, fresh positions beside a ring boundary.
    EXPECT_TRUE(en::Gemma2Sources(*graph, *input, frontier, {}, *bytes));
    graph->options.device_masks = false;
    EXPECT_FALSE(en::Gemma2SourceBytes(*graph));
  }
}

}  // namespace
