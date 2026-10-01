// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Fake addresses are never dereferenced. These establish original shape,
// causal counts, complete ranges and refusal before submission.
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "kernels/ggml/dsv4_ds4_comp.h"

namespace {
namespace kg = jitllm::kernels::ggml;

kg::Ds4CompChunk Chunk(kg::Ds4CacheKind kind, std::uint32_t ratio, std::uint32_t first,
                       std::uint32_t tokens) {
  const auto head = kind == kg::Ds4CacheKind::kKv512 ? 512U : 128U;
  const auto width = head * (ratio == 4 ? 2U : 1U);
  const auto state = static_cast<std::uint64_t>(ratio == 4 ? 8U : ratio) * width * 4;
  const auto source = static_cast<std::uint64_t>(tokens) * width * 4;
  const auto after = (first + tokens) / ratio;
  const auto emitted = after - (first / ratio);
  return {
      .state = {.kv = {0x100000, state}, .score = {0x200000, state}, .kind = kind, .ratio = ratio},
      .kv = {0x10000000, source},
      .score = {0x20000000, source},
      .ape = {0x30000000, static_cast<std::uint64_t>(ratio) * width * 4},
      .norm = {0x40000000, static_cast<std::uint64_t>(head) * 4},
      .values = emitted != 0
                    ? kg::Ds4CacheBuffer{0x50000000, static_cast<std::uint64_t>(emitted) * head * 4}
                    : kg::Ds4CacheBuffer{},
      .rope = {.frequency_base = 10000, .frequency_scale = 1, .attention_factor = 1},
      .first = first,
      .tokens = tokens,
      .before = first / ratio,
      .capacity = after + 2,
      .rms_epsilon = 1};
}

TEST(Ds4CompValidate, ExactStateFootprintsKindRatioAlignmentAndAliases) {
  auto state = Chunk(kg::Ds4CacheKind::kKv512, 4, 0, 4).state;
  EXPECT_EQ(state.kv.bytes, 32768U);
  EXPECT_TRUE(kg::CheckDs4CompState(state));
  state.kv.bytes -= 4;
  EXPECT_FALSE(kg::CheckDs4CompState(state));
  state.kv.bytes += 4;
  state.score.address = state.kv.address;
  EXPECT_FALSE(kg::CheckDs4CompState(state));
  state.score.address = 0x200002;
  EXPECT_FALSE(kg::CheckDs4CompState(state));
  state.score = {std::numeric_limits<std::uint64_t>::max() - 3, 32768};
  EXPECT_FALSE(kg::CheckDs4CompState(state));
  state = Chunk(kg::Ds4CacheKind::kKv512, 128, 0, 128).state;
  EXPECT_EQ(state.kv.bytes, 262144U);
  EXPECT_TRUE(kg::CheckDs4CompState(state));
  state = Chunk(kg::Ds4CacheKind::kIndexer128, 4, 0, 4).state;
  EXPECT_EQ(state.kv.bytes, 8192U);
  EXPECT_TRUE(kg::CheckDs4CompState(state));
  state.ratio = 128;
  EXPECT_FALSE(kg::CheckDs4CompState(state));
  state.ratio = 4;
  // Unknown descriptor enum values must be refused before dispatch.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  state.kind = static_cast<kg::Ds4CacheKind>(255);
  EXPECT_FALSE(kg::CheckDs4CompState(state));
}

TEST(Ds4CompValidate, OriginalChunkChoicesRefreshAndCausalCounts) {
  auto desc = Chunk(kg::Ds4CacheKind::kKv512, 4, 0, 4096);
  auto plan = kg::PlanDs4Comp(desc);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->path, kg::Ds4CompPath::kZeroPrefix);
  EXPECT_EQ(plan->emitted, 1024U);
  EXPECT_TRUE(plan->refresh_required);
  EXPECT_TRUE(kg::CheckDs4CompChunk(desc));
  desc = Chunk(kg::Ds4CacheKind::kKv512, 4, 4096, 4096);
  plan = kg::PlanDs4Comp(desc);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->path, kg::Ds4CompPath::kAligned);
  EXPECT_EQ(plan->after, 2048U);
  EXPECT_TRUE(plan->refresh_required);
  desc = Chunk(kg::Ds4CacheKind::kKv512, 4, 4095, 2);
  plan = kg::PlanDs4Comp(desc);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->path, kg::Ds4CompPath::kRows);
  EXPECT_EQ(plan->emitted, 1U);
  EXPECT_FALSE(plan->refresh_required);
  desc = Chunk(kg::Ds4CacheKind::kKv512, 128, 4096, 4096);
  plan = kg::PlanDs4Comp(desc);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->path, kg::Ds4CompPath::kAligned);
  EXPECT_EQ(plan->emitted, 32U);
  EXPECT_FALSE(plan->refresh_required);
  EXPECT_EQ(kg::Ds4CompCausalCount(2, 4), 0U);
  EXPECT_EQ(kg::Ds4CompCausalCount(3, 4), 1U);
  EXPECT_EQ(kg::Ds4CompCausalCount(127, 128), 1U);
  EXPECT_FALSE(kg::Ds4CompCausalCount(std::numeric_limits<std::uint32_t>::max(), 4));
  EXPECT_FALSE(kg::Ds4CompCausalCount(5, 0));
  desc.before += 1;
  EXPECT_FALSE(kg::PlanDs4Comp(desc));
  desc.before -= 1;
  desc.capacity = desc.before;
  EXPECT_FALSE(kg::PlanDs4Comp(desc));
  desc.capacity = 100;
  desc.tokens = 4097;
  EXPECT_FALSE(kg::PlanDs4Comp(desc));
  desc.tokens = 0;
  EXPECT_FALSE(kg::PlanDs4Comp(desc));
  desc.tokens = 4096;
  desc.first = std::numeric_limits<std::uint32_t>::max() - 2;
  EXPECT_FALSE(kg::PlanDs4Comp(desc));
}

TEST(Ds4CompValidate, ChunkRefusesCrossStageAliasAndIncompleteMirrors) {
  auto desc = Chunk(kg::Ds4CacheKind::kKv512, 4, 0, 16);
  EXPECT_TRUE(kg::CheckDs4CompChunk(desc));
  desc.values.address = desc.state.kv.address;
  EXPECT_FALSE(kg::CheckDs4CompChunk(desc));
  desc.values.address = 0x50000000;
  desc.norm.address = desc.state.score.address;
  EXPECT_FALSE(kg::CheckDs4CompChunk(desc));
  desc.norm.address = 0x40000000;
  desc.codes = {0x60000000, 2816};
  EXPECT_FALSE(kg::CheckDs4CompChunk(desc));
  desc.scales = {0x70000000, 112};
  EXPECT_TRUE(kg::CheckDs4CompChunk(desc));
  desc.scales.bytes -= 1;
  EXPECT_FALSE(kg::CheckDs4CompChunk(desc));
  desc.scales.bytes += 1;
  desc.codes.address += 1;
  EXPECT_FALSE(kg::CheckDs4CompChunk(desc));
  desc.codes.address -= 1;
  desc.codes.address = desc.values.address;
  EXPECT_FALSE(kg::CheckDs4CompChunk(desc));
  desc = Chunk(kg::Ds4CacheKind::kIndexer128, 4, 0, 16);
  desc.codes = {0x60000001, 256};
  desc.scales = {0x70000000, 64};
  EXPECT_TRUE(kg::CheckDs4CompChunk(desc));  // packed nibbles need byte alignment
  desc.kv.bytes -= 4;
  EXPECT_FALSE(kg::CheckDs4CompChunk(desc));
}

TEST(Ds4CompValidate, EmptyEmissionAndTransformParametersAreExplicit) {
  auto desc = Chunk(kg::Ds4CacheKind::kKv512, 4, 1, 1);
  EXPECT_TRUE(kg::CheckDs4CompChunk(desc));
  desc.values = {0x50000000, 2048};
  EXPECT_FALSE(kg::CheckDs4CompChunk(desc));
  desc.values = {};
  desc.rope.extension = 1;
  EXPECT_FALSE(kg::CheckDs4CompChunk(desc));
  desc.rope.original_context = 16384;
  desc.rope.beta_fast = 32;
  desc.rope.beta_slow = 1;
  EXPECT_TRUE(kg::CheckDs4CompChunk(desc));
  desc.rope.frequency_base = 1;
  EXPECT_FALSE(kg::CheckDs4CompChunk(desc));
  desc.rope.frequency_base = 10000;
  desc.rope.frequency_scale = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(kg::CheckDs4CompChunk(desc));
  desc.rope.frequency_scale = 1;
  desc.rms_epsilon = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(kg::CheckDs4CompChunk(desc));
}

TEST(Ds4CompValidate, PoolAlignmentF16ApeAndAccessedPrefixes) {
  const auto chunk = Chunk(kg::Ds4CacheKind::kKv512, 4, 4, 8);
  kg::Ds4CompPool pool{.state = chunk.state,
                       .kv = chunk.kv,
                       .score = chunk.score,
                       .ape = chunk.ape,
                       .values = chunk.values,
                       .first = 4,
                       .tokens = 8};
  EXPECT_TRUE(kg::CheckDs4CompPool(pool));
  pool.tokens = 7;
  EXPECT_FALSE(kg::CheckDs4CompPool(pool));
  pool.tokens = 8;
  pool.first = 5;
  EXPECT_FALSE(kg::CheckDs4CompPool(pool));
  pool.first = 4;
  pool.values.bytes -= 4;
  EXPECT_FALSE(kg::CheckDs4CompPool(pool));
  pool.values.bytes += 4;
  pool.ape_format = kg::Ds4CompApe::kF16;
  pool.ape.bytes /= 2;
  pool.ape.address += 2;
  EXPECT_TRUE(kg::CheckDs4CompPool(pool));
  pool.ape.address += 1;
  EXPECT_FALSE(kg::CheckDs4CompPool(pool));
  pool.ape.address -= 1;
  pool.values.address = pool.kv.address;
  EXPECT_FALSE(kg::CheckDs4CompPool(pool));
}

TEST(Ds4CompValidate, RefreshRequiresFourTokenSmallInputsAndDistinctState) {
  const auto chunk = Chunk(kg::Ds4CacheKind::kKv512, 4, 0, 4);
  kg::Ds4CompRefresh refresh{
      .state = chunk.state, .kv = chunk.kv, .score = chunk.score, .ape = chunk.ape, .first = 5};
  EXPECT_TRUE(kg::CheckDs4CompRefresh(refresh));  // original tail phases can be ragged
  refresh.score.bytes -= 4;
  EXPECT_FALSE(kg::CheckDs4CompRefresh(refresh));
  refresh.score.bytes += 4;
  refresh.kv.address = refresh.state.kv.address;
  EXPECT_FALSE(kg::CheckDs4CompRefresh(refresh));
  refresh.kv.address = chunk.kv.address;
  refresh.state.ratio = 128;
  EXPECT_FALSE(kg::CheckDs4CompRefresh(refresh));
}

}  // namespace
