// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The device decoders' geometry (kernels/decode/operand.h), a local copy
// of the format layer's for device code, against the format layer itself:
// block sizes and bytes from format::Info(), the scale atoms' constants and
// every offset of a range of rows and blocks from format::ScaleAtomOffset.
// Every profile: the header is vendor-free.

#include <gtest/gtest.h>

#include <cstdint>

#include "format/encoding.h"
#include "format/layout.h"
#include "kernels/decode/operand.h"

namespace {

namespace format = llmp::format;
namespace decode = llmp::kernels::decode;

TEST(DecodeGeometryTest, BlocksMatchTheFormatLayer) {
  EXPECT_EQ(decode::kNvfp4Block, format::Info(format::Encoding::kNvfp4).block_elements);
  EXPECT_EQ(decode::kMxfp8Block, format::Info(format::Encoding::kMxfp8).block_elements);
  EXPECT_EQ(decode::kQ8_0Block, format::Info(format::Encoding::kQ8_0).block_elements);
  EXPECT_EQ(decode::kQ8_0BlockBytes, format::Info(format::Encoding::kQ8_0).ggml->block_bytes);
  EXPECT_EQ(decode::kGgmlNvfp4Block, format::Info(format::Encoding::kNvfp4).ggml->block_elements);
  EXPECT_EQ(decode::kGgmlNvfp4BlockBytes, format::Info(format::Encoding::kNvfp4).ggml->block_bytes);
  EXPECT_EQ(format::Info(format::Encoding::kNvfp4).element_bits, 4U);
  EXPECT_EQ(format::Info(format::Encoding::kMxfp8).element_bits, 8U);
  // Every block a whole number of segments.
  for (const std::uint32_t block :
       {decode::kNvfp4Block, decode::kMxfp8Block, decode::kQ8_0Block, decode::kGgmlNvfp4Block}) {
    EXPECT_EQ(block % decode::kSegment, 0U) << block;
  }
}

TEST(DecodeGeometryTest, ScaleAtomsMatchTheFormatLayer) {
  EXPECT_EQ(decode::kAtomRows, format::kScaleAtomRows);
  EXPECT_EQ(decode::kAtomBlocks, format::kScaleAtomBlocks);
  EXPECT_EQ(decode::kAtomBytes, format::kScaleAtomBytes);
  for (const std::uint64_t blocks : {4ULL, 8ULL, 40ULL, 160ULL}) {
    for (std::uint64_t row = 0; row < 300; ++row) {
      for (std::uint64_t block = 0; block < blocks; ++block) {
        ASSERT_EQ(decode::AtomOffset(row, block, blocks),
                  format::ScaleAtomOffset(row, block, blocks))
            << row << " " << block << " " << blocks;
      }
    }
  }
}

}  // namespace
