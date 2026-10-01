// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/dsv4_ds4_weights.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace jitllm::engine {
namespace {
namespace kg = kernels::ggml;

Ds4PreparedWeight Paired() {
  Ds4PreparedWeight tensor;
  tensor.shape = {.kind = kg::Ds4AlignedKind::kQ2K, .input = 1024, .output = 6, .groups = 2};
  tensor.layout = *kg::Ds4AlignedLayoutOf(tensor.shape);
  tensor.sources = {{.shard = 0, .file_offset = 8448, .bytes = 2016, .file_end = 12288},
                    {.shard = 1, .file_offset = 41216, .bytes = 2016, .file_end = 45056}};
  return tensor;
}

TEST(Ds4WeightChunks, PairedRowsAcrossNoncontiguousExpertSources) {
  const auto tensor = Paired();
  const auto chunks = PlanDs4WeightChunks(tensor, 1000);
  ASSERT_TRUE(chunks.has_value()) << (chunks ? "" : chunks.error());
  ASSERT_EQ(chunks->size(), 6);
  for (std::uint64_t i = 0; i < 6; ++i) {
    const auto& chunk = (*chunks)[i];
    EXPECT_EQ(chunk.shard, i < 3 ? 0U : 1U);
    EXPECT_EQ(chunk.read_offset, i < 3 ? 8192U : 40960U);
    EXPECT_EQ(chunk.read_bytes, 4096U);
    EXPECT_EQ(chunk.payload_offset, 256U + ((i % 3) * 672U));
    EXPECT_EQ(chunk.payload_bytes, 672U);
    EXPECT_EQ(chunk.first_block, i * 8U);
    EXPECT_EQ(chunk.blocks, 8U);
  }
}

TEST(Ds4WeightChunks, NeverBlendsExpertSlicesAcrossShards) {
  auto tensor = Paired();
  tensor.shape = {.kind = kg::Ds4AlignedKind::kIq2Xxs, .input = 1024, .output = 9, .groups = 2};
  tensor.layout = *kg::Ds4AlignedLayoutOf(tensor.shape);
  for (auto& source : tensor.sources) source.bytes = 2376;
  const auto chunks = PlanDs4WeightChunks(tensor, 330);
  ASSERT_TRUE(chunks.has_value()) << (chunks ? "" : chunks.error());
  ASSERT_EQ(chunks->size(), 16);
  EXPECT_EQ((*chunks)[7].first_block, 35U);
  EXPECT_EQ((*chunks)[7].blocks, 1U);
  EXPECT_EQ((*chunks)[7].payload_bytes, 66U);
  EXPECT_EQ((*chunks)[8].first_block, 36U);
  EXPECT_EQ((*chunks)[8].shard, 1U);
  EXPECT_EQ(chunks->back().first_block + chunks->back().blocks, 72U);
}

TEST(Ds4WeightChunks, DirectReadSlackCrossesOnlyValidatedGroupBytes) {
  auto tensor = Paired();
  tensor.shape = {.kind = kg::Ds4AlignedKind::kQ8Dense, .input = 1024, .output = 128, .groups = 1};
  tensor.layout = *kg::Ds4AlignedLayoutOf(tensor.shape);
  tensor.sources = {{.shard = 3, .file_offset = 8448, .bytes = 139264, .file_end = 147456 + 4096}};
  const auto chunks = PlanDs4WeightChunks(tensor, 34000);
  ASSERT_TRUE(chunks.has_value()) << (chunks ? "" : chunks.error());
  std::uint64_t read_payload = 0;
  std::uint64_t blocks = 0;
  for (const auto& chunk : *chunks) {
    EXPECT_EQ(chunk.read_offset % 4096, 0U);
    EXPECT_EQ(chunk.read_bytes % 4096, 0U);
    EXPECT_LE(chunk.read_bytes, 34000U + 8192U);
    EXPECT_LE(chunk.read_offset + chunk.read_bytes, tensor.sources[0].file_end);
    EXPECT_EQ(chunk.read_offset + chunk.payload_offset, 8448U + read_payload);
    EXPECT_EQ(chunk.first_block, blocks);
    read_payload += chunk.payload_bytes;
    blocks += chunk.blocks;
  }
  EXPECT_EQ(read_payload, 139264U);
  EXPECT_EQ(blocks, 4096U);
}

TEST(Ds4WeightChunks, RefusesTooSmallMisboundAndUnboundedPlans) {
  auto tensor = Paired();
  EXPECT_FALSE(PlanDs4WeightChunks(tensor, 671));
  EXPECT_FALSE(PlanDs4WeightChunks(tensor, 0));
  EXPECT_FALSE(PlanDs4WeightChunks(tensor, (std::uint64_t{256} << 20U) + 1));
  tensor.sources.pop_back();
  EXPECT_FALSE(PlanDs4WeightChunks(tensor, 1000));
  tensor = Paired();
  ++tensor.sources[0].bytes;
  EXPECT_FALSE(PlanDs4WeightChunks(tensor, 1000));
  tensor = Paired();
  tensor.sources[0].file_end = tensor.sources[0].file_offset + tensor.sources[0].bytes;
  EXPECT_FALSE(PlanDs4WeightChunks(tensor, 1000));
  tensor = Paired();
  tensor.sources[0].file_offset = std::numeric_limits<std::uint64_t>::max() - 10;
  EXPECT_FALSE(PlanDs4WeightChunks(tensor, 1000));
}

TEST(Ds4WeightChunks, CapsJobMetadataBeforeAllocatingHugeVectors) {
  auto tensor = Paired();
  tensor.shape = {
      .kind = kg::Ds4AlignedKind::kIq2Xxs, .input = 4096, .output = 2048, .groups = 256};
  tensor.layout = *kg::Ds4AlignedLayoutOf(tensor.shape);
  tensor.sources.resize(256);
  EXPECT_FALSE(PlanDs4WeightChunks(tensor, 66));
}
}  // namespace
}  // namespace jitllm::engine
