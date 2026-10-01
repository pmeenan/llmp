// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Fake addresses are never dereferenced. Refusals protect bounded importer
// staging and the complete alternative physical weight set before submission.
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <limits>

#include "kernels/ggml/dsv4_ds4_repack.h"

namespace {
namespace kg = jitllm::kernels::ggml;

constexpr std::uint64_t kRaw = 1ULL << 40;
constexpr std::uint64_t kPacked = 1ULL << 60;

void Refused(const std::expected<void, kg::KernelFailure>& checked) {
  ASSERT_FALSE(checked.has_value());
  EXPECT_EQ(checked.error().error, kg::KernelError::kRejected);
}

kg::Ds4RepackChunk Chunk(kg::Ds4AlignedShape shape) {
  const auto layout = kg::Ds4AlignedLayoutOf(shape);
  if (!layout) return {};
  return {.shape = shape,
          .raw = {kRaw, layout->raw_bytes.value()},
          .packed = {kPacked, layout->packed_bytes.value()},
          .first_block = 0,
          .blocks = layout->blocks};
}

TEST(Ds4RepackValidate, AlignedSectionsHaveTheOriginalIndependentByteCounts) {
  struct Example {
    kg::Ds4AlignedShape shape;
    std::uint64_t blocks{};
    std::uint64_t raw{};
    std::uint64_t packed{};
    std::uint64_t scales{};
    std::uint64_t codes{};
  };
  constexpr std::array<Example, 4> kExamples = {{
      {{kg::Ds4AlignedKind::kIq2Xxs, 512, 3, 2}, 12, 792, 832, 0, 64},
      {{kg::Ds4AlignedKind::kQ8Dense, 96, 3, 1}, 9, 306, 352, 0, 64},
      {{kg::Ds4AlignedKind::kQ2K, 768, 6, 2}, 36, 3024, 3072, 192, 768},
      {{kg::Ds4AlignedKind::kQ2K, 768, 6, 1}, 18, 1512, 1600, 128, 448},
  }};
  for (const auto& example : kExamples) {
    const auto layout = kg::Ds4AlignedLayoutOf(example.shape);
    ASSERT_TRUE(layout);
    EXPECT_EQ(layout->blocks, example.blocks);
    EXPECT_EQ(layout->raw_bytes.value(), example.raw);
    EXPECT_EQ(layout->packed_bytes.value(), example.packed);
    EXPECT_EQ(layout->scales_offset, example.scales);
    EXPECT_EQ(layout->codes_offset, example.codes);
    EXPECT_TRUE(kg::CheckDs4RepackChunk(Chunk(example.shape)));
  }
}

TEST(Ds4RepackValidate, InvalidGeometryAndSectionOverflowAreRefused) {
  kg::Ds4AlignedShape shape{kg::Ds4AlignedKind::kIq2Xxs, 256, 3, 2};
  for (const auto invalid : {0U, 255U, 257U}) {
    auto bad = shape;
    bad.input = invalid;
    EXPECT_FALSE(kg::Ds4AlignedLayoutOf(bad));
  }
  auto bad = shape;
  bad.output = 0;
  EXPECT_FALSE(kg::Ds4AlignedLayoutOf(bad));
  bad = shape;
  bad.groups = 0;
  EXPECT_FALSE(kg::Ds4AlignedLayoutOf(bad));
  bad = {kg::Ds4AlignedKind::kQ2K, 256, 3, 2};
  EXPECT_FALSE(kg::Ds4AlignedLayoutOf(bad));
  bad = {kg::Ds4AlignedKind::kQ8Dense, 32, 3, 2};
  EXPECT_FALSE(kg::Ds4AlignedLayoutOf(bad));
  bad = shape;
  // Deliberately construct an unknown fixed-underlying enum value.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  bad.kind = static_cast<kg::Ds4AlignedKind>(255);
  EXPECT_FALSE(kg::Ds4AlignedLayoutOf(bad));
  bad = {kg::Ds4AlignedKind::kIq2Xxs, 4294967040U, std::numeric_limits<std::uint32_t>::max(),
         std::numeric_limits<std::uint32_t>::max()};
  EXPECT_FALSE(kg::Ds4AlignedLayoutOf(bad));
}

TEST(Ds4RepackValidate, CompleteOutputAndRawAccessesMustBeAlignedBoundedAndDisjoint) {
  constexpr std::array<kg::Ds4AlignedShape, 3> kShapes = {{
      {kg::Ds4AlignedKind::kIq2Xxs, 256, 3, 2},
      {kg::Ds4AlignedKind::kQ2K, 768, 6, 2},
      {kg::Ds4AlignedKind::kQ8Dense, 96, 3, 1},
  }};
  for (const auto& shape : kShapes) {
    const auto valid = Chunk(shape);
    ASSERT_TRUE(kg::CheckDs4RepackChunk(valid));
    auto bad = valid;
    --bad.raw.bytes;
    Refused(kg::CheckDs4RepackChunk(bad));
    bad = valid;
    ++bad.raw.address;
    Refused(kg::CheckDs4RepackChunk(bad));
    bad = valid;
    --bad.packed.bytes;
    Refused(kg::CheckDs4RepackChunk(bad));
    bad = valid;
    bad.packed.address += 2;
    Refused(kg::CheckDs4RepackChunk(bad));
    bad = valid;
    bad.raw.address = 0;
    Refused(kg::CheckDs4RepackChunk(bad));
    bad = valid;
    bad.packed.address = 0;
    Refused(kg::CheckDs4RepackChunk(bad));
    bad = valid;
    bad.raw.address = std::numeric_limits<std::uint64_t>::max() - 1;
    Refused(kg::CheckDs4RepackChunk(bad));
    bad = valid;
    bad.packed.address = std::numeric_limits<std::uint64_t>::max() - 63;
    Refused(kg::CheckDs4RepackChunk(bad));
    bad = valid;
    bad.raw.bytes = std::numeric_limits<std::uint64_t>::max();
    Refused(kg::CheckDs4RepackChunk(bad));
    bad = valid;
    bad.raw.address = bad.packed.address + bad.packed.bytes - 2;
    Refused(kg::CheckDs4RepackChunk(bad));
    bad.raw.address += 2;  // Exactly adjacent, with no accessed-byte overlap.
    EXPECT_TRUE(kg::CheckDs4RepackChunk(bad));
    bad = valid;
    bad.packed.address = bad.raw.address;
    Refused(kg::CheckDs4RepackChunk(bad));
    EXPECT_TRUE(kg::CheckDs4RepackInitialize(shape, valid.packed));
    auto output = valid.packed;
    --output.bytes;
    Refused(kg::CheckDs4RepackInitialize(shape, output));
    output = valid.packed;
    ++output.address;
    Refused(kg::CheckDs4RepackInitialize(shape, output));
  }
}

TEST(Ds4RepackValidate, PartialChunksUseAbsoluteOffsetsAndQ2WholeRowPairs) {
  auto chunk = Chunk({kg::Ds4AlignedKind::kIq2Xxs, 256, 37, 2});
  chunk.first_block = 11;
  chunk.blocks = 17;
  chunk.raw.bytes = std::uint64_t{17} * 66;
  EXPECT_TRUE(kg::CheckDs4RepackChunk(chunk));
  auto bad = chunk;
  bad.blocks = 0;
  Refused(kg::CheckDs4RepackChunk(bad));
  bad = chunk;
  bad.first_block = 74;
  Refused(kg::CheckDs4RepackChunk(bad));
  bad.first_block = std::numeric_limits<std::uint64_t>::max();
  Refused(kg::CheckDs4RepackChunk(bad));
  bad = chunk;
  bad.blocks = 64;
  Refused(kg::CheckDs4RepackChunk(bad));
  chunk = Chunk({kg::Ds4AlignedKind::kQ2K, 768, 6, 2});
  chunk.first_block = 12;
  chunk.blocks = 12;  // Last pair of expert0 and first pair of expert1.
  chunk.raw.bytes = std::uint64_t{12} * 84;
  EXPECT_TRUE(kg::CheckDs4RepackChunk(chunk));
  bad = chunk;
  bad.first_block = 3;  // One row is not a complete paired-row boundary.
  Refused(kg::CheckDs4RepackChunk(bad));
  bad = chunk;
  bad.blocks = 3;
  Refused(kg::CheckDs4RepackChunk(bad));
}

TEST(Ds4RepackValidate, GridEdgeIsCheckedWithoutMultiplicationOrRoundingOverflow) {
  auto chunk =
      Chunk({kg::Ds4AlignedKind::kIq2Xxs, 256, std::numeric_limits<std::uint32_t>::max(), 32});
  constexpr std::uint64_t kLastGridBlocks = std::uint64_t{2147483647} * 32;
  chunk.blocks = kLastGridBlocks;
  chunk.raw.bytes = chunk.blocks * 66;
  EXPECT_TRUE(kg::CheckDs4RepackChunk(chunk));
  ++chunk.blocks;
  chunk.raw.bytes += 66;
  Refused(kg::CheckDs4RepackChunk(chunk));
}

}  // namespace
