// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <string>
#include <vector>

#include "benchmarks/ds4_complete/executor.h"

namespace {
namespace complete = jitllm::benchmarks::ds4_complete;
namespace kg = jitllm::kernels::ggml;
namespace model = jitllm::model;
using Buffer = kg::Ds4CacheBuffer;

const void* ReadPointer(std::uint64_t address) {
  return std::bit_cast<const void*>(static_cast<std::uintptr_t>(address));
}
void* WritePointer(std::uint64_t address) {
  return std::bit_cast<void*>(static_cast<std::uintptr_t>(address));
}
// Only metadata for ledger controls. It intentionally contains no model
// operands; CheckChunk must still refuse it. These transitions are a CPU
// proof of counting/order, never a substitute for GPU Job completion.
complete::Chunk LedgerChunk(std::uint32_t first = 0) {
  complete::Chunk c;
  c.first = first;
  c.context = 8192;
  c.device_sms = 48;
  c.storage_generation = 7;
  c.model_generation = 11;
  c.artifact_id = complete::kCommunityArtifact;
  c.layers.resize(43);
  for (std::uint32_t i = 0; i < 43; ++i) {
    c.layers[i].index = i;
    c.layers[i].ratio = model::Dsv4Flash().compress_ratios[i];
  }
  if (first == 4096) c.frontier.emplace();
  return c;
}
complete::Progress LedgerFor(const complete::Chunk& c) {
  complete::Progress x;
  x.first = c.first;
  x.storage_generation = c.storage_generation;
  x.model_generation = c.model_generation;
  for (std::uint32_t i = 0; i < 43; ++i) {
    const auto ratio = c.layers[i].ratio;
    x.compressed[i] = ratio == 0 ? 0U : c.first / ratio;
    x.indexed[i] = ratio == 4 ? c.first / 4 : 0U;
  }
  return x;
}
complete::Chunk InventorySlice() {
  complete::Chunk c;
  c.embedding.tokens = {ReadPointer(0x100000), 4096ULL * 4};
  c.embedding.weights.storage = {ReadPointer(0x200000), 65536};
  c.embedding.output.storage = {WritePointer(0x300000), 131072};
  c.paid_ranges = {{0x100000, 4096ULL * 4}, {0x200000, 65536}, {0x300000, 131072}};
  return c;
}

TEST(Ds4CompleteExecutor, RequiresTheWholeQualifiedRecipeBeforeStarting) {
  auto c = LedgerChunk();
  EXPECT_FALSE(complete::CheckChunk(model::Dsv4Flash(), c));
  EXPECT_FALSE(complete::Begin(model::Dsv4Flash(), c));
  c.layers.pop_back();
  auto missing = complete::CheckChunk(model::Dsv4Flash(), c);
  ASSERT_FALSE(missing);
  EXPECT_NE(missing.error().find("all43"), std::string::npos);
  c = LedgerChunk();
  c.layers.push_back({});
  EXPECT_FALSE(complete::CheckChunk(model::Dsv4Flash(), c));
  c = LedgerChunk();
  c.first = 2048;
  EXPECT_FALSE(complete::CheckChunk(model::Dsv4Flash(), c));
  c = LedgerChunk();
  c.context = 16384;
  EXPECT_FALSE(complete::CheckChunk(model::Dsv4Flash(), c));
  c = LedgerChunk();
  c.artifact_id = "another-artifact";
  EXPECT_FALSE(complete::CheckChunk(model::Dsv4Flash(), c));
  EXPECT_FALSE(complete::ReadResolvedDispatch(c));
}

TEST(Ds4CompleteExecutor, PaidCoverageIncludesTheWholeDeclaredRangeAndSlack) {
  auto c = InventorySlice();
  ASSERT_TRUE(complete::CheckMappedOperands(c));
  c.paid_ranges.back().bytes -= 1;
  EXPECT_FALSE(complete::CheckMappedOperands(c));
  c = InventorySlice();
  c.embedding.output.storage.bytes += 256;  // A guarded tail must also be paid.
  EXPECT_FALSE(complete::CheckMappedOperands(c));
  c = InventorySlice();
  c.paid_ranges.clear();
  EXPECT_FALSE(complete::CheckMappedOperands(c));
  c = InventorySlice();
  c.paid_ranges.back() = {UINT64_MAX - 31, 64};
  EXPECT_FALSE(complete::CheckMappedOperands(c));
  c = InventorySlice();
  c.embedding.output.storage = {WritePointer(UINT64_MAX - 31), 64};
  EXPECT_FALSE(complete::CheckMappedOperands(c));
}

TEST(Ds4CompleteExecutor, ScratchCannotOverwriteTokensOrAnotherStagesWeights) {
  auto c = InventorySlice();
  c.embedding.output.storage = {WritePointer(0x100000), 16};
  EXPECT_FALSE(complete::CheckMappedOperands(c));
  c = InventorySlice();
  c.layers.resize(1);
  c.layers[0].query_a.weights.codes = {ReadPointer(0x300000), 131072};
  EXPECT_FALSE(complete::CheckMappedOperands(c));
  c = InventorySlice();
  c.layers.resize(1);
  c.layers[0].router.bias = {0x300000, 1024};
  EXPECT_FALSE(complete::CheckMappedOperands(c));
}

TEST(Ds4CompleteExecutor, NativeOutputBControlRequiresSeparatePaidOriginalInputStorage) {
  auto c = InventorySlice();
  c.output_b_control = {0x80000000, 4096ULL * 4096 * 4};
  EXPECT_FALSE(complete::CheckMappedOperands(c));
  c.paid_ranges.push_back(c.output_b_control);
  ASSERT_TRUE(complete::CheckMappedOperands(c));
  c.output_b_control = {0x200000, 1024};
  EXPECT_FALSE(complete::CheckMappedOperands(c));
  c = InventorySlice();
  c.layers.resize(1);
  const std::array live{
      Buffer{0x90000000, 4096ULL * 4096 * 4}, Buffer{0xa0000000, 4096ULL * 8192 * 4},
      Buffer{0xb0000000, 4096ULL * 64 * 144}, Buffer{0xc0000000, 4096ULL * 16384 * 4},
      Buffer{0xd0000000, 4096ULL * 24 * 4}};
  c.layers[0].output_b.output.storage = {WritePointer(live[0].address), live[0].bytes};
  c.layers[0].output_b.input.storage = {ReadPointer(live[1].address), live[1].bytes};
  c.layers[0].output_b.quantized.storage = {WritePointer(live[2].address), live[2].bytes};
  c.layers[0].hc_attention_pre.residual = live[3];
  c.layers[0].hc_attention_pre.coefficients.split = live[4];
  for (const auto operand : live) c.paid_ranges.push_back(operand);
  c.output_b_control = {0x80000000, 4096ULL * 4096 * 4};
  c.paid_ranges.push_back(c.output_b_control);
  ASSERT_TRUE(complete::CheckMappedOperands(c));
  for (const auto operand : live) {
    c.output_b_control.address = operand.address + 16;
    c.paid_ranges.back() = c.output_b_control;
    const auto alias = complete::CheckMappedOperands(c);
    ASSERT_FALSE(alias);
    EXPECT_NE(alias.error().find("output-B"), std::string::npos);
  }
  auto full = LedgerChunk();
  full.output_b_consumer = complete::OutputBConsumer::kNativeMmq;
  auto rejected = complete::CheckChunk(model::Dsv4Flash(), full);
  ASSERT_FALSE(rejected);
  EXPECT_NE(rejected.error().find("output-B"), std::string::npos);
  full.output_b_study = true;
  full.output_b_control = {0x80000000, 4096ULL * 4096 * 4};
  rejected = complete::CheckChunk(model::Dsv4Flash(), full);
  ASSERT_FALSE(rejected);
  EXPECT_NE(rejected.error().find("output-B"), std::string::npos);
}

TEST(Ds4CompleteExecutor, PersistentCachesSurviveScratchAndOtherLayerJobs) {
  auto c = InventorySlice();
  c.layers.resize(2);
  c.layers[0].raw_store.ring = {0x400000, 65536};
  c.layers[1].raw_store.ring = {0x500000, 65536};
  c.paid_ranges.push_back(c.layers[0].raw_store.ring);
  c.paid_ranges.push_back(c.layers[1].raw_store.ring);
  ASSERT_TRUE(complete::CheckMappedOperands(c));
  c.layers[1].raw_store.ring = {0x400100, 4096};
  EXPECT_FALSE(complete::CheckMappedOperands(c));
  c.layers[1].raw_store.ring = {0x500000, 65536};
  c.layers[1].attention.scratch = {0x400100, 4096};
  EXPECT_FALSE(complete::CheckMappedOperands(c));
}

TEST(Ds4CompleteExecutor, RoutedProbeCannotReplaceAnyLiveOriginalOperand) {
  auto c = InventorySlice();
  c.routed_ffn_study = true;
  c.layers.resize(1);
  auto& layer = c.layers[0];
  const std::array live{Buffer{0x90000000, 65536}, Buffer{0x91000000, 65536},
                        Buffer{0x92000000, 65536}, Buffer{0x93000000, 65536},
                        Buffer{0x94000000, 65536}, Buffer{0x95000000, 65536},
                        Buffer{0x96000000, 65536}, Buffer{0x97000000, 65536}};
  layer.routed.down = live[0];
  layer.routed.down_quant = live[1];
  layer.routed.input_quant = live[2];
  layer.routed.ids_source = live[3];
  layer.routed.ids_destination = live[4];
  layer.routed.work = live[5];
  layer.hc_ffn_pre.residual = live[6];
  layer.hc_ffn_pre.coefficients.split = live[7];
  for (auto buffer : live) c.paid_ranges.push_back(buffer);
  layer.routed_control.emplace();
  auto& probe = *layer.routed_control;
  probe.down = {0xa0000000, 65536};
  c.paid_ranges.push_back(probe.down);
  ASSERT_TRUE(complete::CheckMappedOperands(c));
  for (auto buffer : live) {
    probe.down.address = buffer.address + 16;
    c.paid_ranges.back() = probe.down;
    auto refused = complete::CheckMappedOperands(c);
    ASSERT_FALSE(refused);
    EXPECT_NE(refused.error().find("routed-FFN"), std::string::npos);
  }
  probe.down = {0xa0000000, 65536};
  c.paid_ranges.back() = probe.down;
  probe.ids_source = {live[3].address + 16, 4096};
  EXPECT_FALSE(complete::CheckMappedOperands(c));
  probe.ids_source = {};
  c.routed_intermediates[0] = {live[6].address + 16, 4096};
  EXPECT_FALSE(complete::CheckMappedOperands(c));
  c.routed_intermediates[0] = {};
  c.paid_ranges.back().bytes -= 1;
  EXPECT_FALSE(complete::CheckMappedOperands(c));
}

TEST(Ds4CompleteExecutor, RoutedTierRequiresPrivateStudyAndSymmetricStorage) {
  auto c = LedgerChunk();
  c.routed_ffn_tier = complete::RoutedFfnTier::kMaterialized;
  auto check = complete::CheckChunk(model::Dsv4Flash(), c);
  ASSERT_FALSE(check);
  EXPECT_NE(check.error().find("routed-FFN"), std::string::npos);
  c.routed_ffn_study = true;
  check = complete::CheckChunk(model::Dsv4Flash(), c);
  ASSERT_FALSE(check);
  EXPECT_NE(check.error().find("symmetrically paid"), std::string::npos);
  c.routed_ffn_tier = complete::RoutedFfnTier::kDirect;
  c.output_b_study = true;
  check = complete::CheckChunk(model::Dsv4Flash(), c);
  ASSERT_FALSE(check);
  EXPECT_NE(check.error().find("overlapping studies"), std::string::npos);
  EXPECT_TRUE(complete::CaptureRoutedFfnAt(4096, 21));
  EXPECT_FALSE(complete::CaptureRoutedFfnAt(0, 21));
  EXPECT_FALSE(complete::CaptureRoutedFfnAt(4096, 20));
}

TEST(Ds4CompleteExecutor, AdvancesOnlyTheCompletedLayerAndFinalHead) {
  auto c = LedgerChunk(4096);
  auto x = LedgerFor(c);
  ASSERT_TRUE(complete::CompleteEmbedding(c, x));
  for (std::uint32_t i = 0; i < 43; ++i) {
    ASSERT_TRUE(complete::CheckProgress(c, x, complete::Phase::kLayers, i));
    if (i < 42) {
      const auto ratio = c.layers[i + 1].ratio;
      EXPECT_EQ(x.compressed[i + 1], ratio == 0 ? 0U : 4096U / ratio);
    }
    ASSERT_TRUE(complete::CompleteLayer(c, i, x));
    const auto ratio = c.layers[i].ratio;
    EXPECT_EQ(x.compressed[i], ratio == 0 ? 0U : 8192U / ratio);
    EXPECT_EQ(x.indexed[i], ratio == 4 ? 2048U : 0U);
  }
  EXPECT_EQ(x.phase, complete::Phase::kFrontier);
  ASSERT_TRUE(complete::CompleteFrontier(c, x));
  EXPECT_EQ(x.phase, complete::Phase::kComplete);
  EXPECT_FALSE(complete::CompleteFrontier(c, x));
}

TEST(Ds4CompleteExecutor, FirstChunkHasNoUnrequestedHead) {
  auto c = LedgerChunk();
  auto x = LedgerFor(c);
  ASSERT_TRUE(complete::CompleteEmbedding(c, x));
  for (std::uint32_t i = 0; i < 43; ++i) ASSERT_TRUE(complete::CompleteLayer(c, i, x));
  EXPECT_EQ(x.phase, complete::Phase::kComplete);
  EXPECT_FALSE(complete::CompleteFrontier(c, x));
  EXPECT_EQ(x.compressed[2], 1024U);
  EXPECT_EQ(x.compressed[3], 32U);
}

TEST(Ds4CompleteExecutor, RefusalRetainsCountsAndFailurePoisonsContinuation) {
  auto c = LedgerChunk();
  auto x = LedgerFor(c);
  ASSERT_TRUE(complete::CompleteEmbedding(c, x));
  const auto before = x;
  EXPECT_FALSE(complete::CompleteLayer(c, 1, x));
  EXPECT_EQ(x.next_layer, before.next_layer);
  EXPECT_EQ(x.compressed, before.compressed);
  EXPECT_EQ(x.indexed, before.indexed);
  EXPECT_EQ(x.phase, before.phase);
  ASSERT_TRUE(complete::CompleteLayer(c, 0, x));
  EXPECT_FALSE(complete::CompleteLayer(c, 0, x));
  auto stale = x;
  stale.storage_generation += 1;
  EXPECT_FALSE(complete::CheckProgress(c, stale, complete::Phase::kLayers, 1));
  stale = x;
  stale.model_generation += 1;
  EXPECT_FALSE(complete::CheckProgress(c, stale, complete::Phase::kLayers, 1));
  stale = x;
  stale.compressed[2] += 1;
  EXPECT_FALSE(complete::CheckProgress(c, stale, complete::Phase::kLayers, 1));
  const auto counted = x.compressed;
  complete::Poison(x);
  EXPECT_EQ(x.phase, complete::Phase::kPoisoned);
  EXPECT_EQ(x.compressed, counted);
  EXPECT_FALSE(complete::CompleteLayer(c, 1, x));
}

TEST(Ds4CompleteExecutor, HashBinderChecksFinalRaggedBatchAndActualIdentity) {
  auto c = LedgerChunk();
  const auto& p = model::Dsv4Flash();
  std::vector<std::int32_t> ids(static_cast<std::size_t>(p.vocab) * 6);
  for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<std::int32_t>(i % 6);
  std::array<complete::HashTable, 3> tables{};
  for (std::uint32_t i = 0; i < 3; ++i) {
    c.layers[i].router.hash = {0x600000U + (static_cast<std::uint64_t>(i) * 0x400000U),
                               ids.size() * sizeof(std::int32_t)};
    tables[i] = {.layer = i, .device = c.layers[i].router.hash, .entries = ids};
  }
  ASSERT_TRUE(complete::CheckHashTables(p, c, tables));
  ids[ids.size() - 1] = 4;  // Duplicate in the last row of the final ragged batch.
  EXPECT_FALSE(complete::CheckHashTables(p, c, tables));
  ids[ids.size() - 1] = 256;
  EXPECT_FALSE(complete::CheckHashTables(p, c, tables));
  ids[ids.size() - 1] = 5;
  tables[2].device.address += 4;
  EXPECT_FALSE(complete::CheckHashTables(p, c, tables));
  tables[2].device = c.layers[2].router.hash;
  tables[2].entries = tables[2].entries.first(ids.size() - 1);
  EXPECT_FALSE(complete::CheckHashTables(p, c, tables));
}

TEST(Ds4CompleteExecutor, ConstructsOriginalCompressedAndRawYarnParameters) {
  const auto& p = model::Dsv4Flash();
  const auto raw = complete::OriginalRope(p, 0, 4096);
  const auto csa = complete::OriginalRope(p, 4, 4096);
  const auto hca = complete::OriginalRope(p, 128, 4096, true);
  ASSERT_TRUE(raw);
  ASSERT_TRUE(csa);
  ASSERT_TRUE(hca);
  EXPECT_EQ(raw->original_context, 0U);
  EXPECT_EQ(raw->base, 10000.0F);
  EXPECT_EQ(raw->scale, 1.0F);
  EXPECT_EQ(raw->extension, 0.0F);
  EXPECT_EQ(raw->attention, 1.0F);
  EXPECT_EQ(csa->base, 160000.0F);
  EXPECT_EQ(csa->scale, 0.0625F);
  EXPECT_EQ(csa->extension, 1.0F);
  EXPECT_EQ(csa->original_context, 65536U);
  EXPECT_EQ(hca->attention, csa->attention);
  EXPECT_TRUE(hca->inverse);
  EXPECT_FALSE(complete::OriginalRope(p, 8, 0));
  auto invalid = p;
  invalid.rope_scale = 0;
  EXPECT_FALSE(complete::OriginalRope(invalid, 4, 0));
}
}  // namespace
