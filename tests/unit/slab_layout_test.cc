// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The resident expert slab's pages (engine/paged_weights.h LayOutSlab):
// each 2 MiB page of the slab's address range reads one 4 KiB-aligned
// range of one file, at most a slot long, and copies pieces of it into the
// page; together the pages put every stored byte of every group at
// slab + e·S + offset, exactly once, and nothing else; where the groups
// change shard, a page boundary falls in the gap between two groups.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "engine/paged_weights.h"
#include "expected_error.h"

namespace {

using llmp::engine::kPagedExtent;
using llmp::engine::kSlabSlotBytes;
using llmp::engine::LayOutSlab;
using llmp::engine::SlabLayout;
using llmp::test_support::Failed;

// Each page's read and pieces: aligned, within the slot and the page, each
// piece's first and last byte landing where its group's byte belongs and
// read from that byte's place in the file; and the pieces together as many
// bytes as the groups store.
void CheckCovers(const SlabLayout& slab, const std::vector<std::uint32_t>& shard,
                 const std::vector<std::uint64_t>& file, std::uint64_t stored) {
  std::uint64_t covered = 0;
  for (std::size_t p = 0; p < slab.pages.size(); ++p) {
    const auto& page = slab.pages[p];
    ASSERT_GT(page.pieces, 0U) << p;
    EXPECT_EQ(page.file_offset % 4096, 0U) << p;
    EXPECT_EQ(page.length % 4096, 0U) << p;
    EXPECT_LE(page.length, kSlabSlotBytes) << p;
    for (std::size_t i = 0; i < page.pieces; ++i) {
      EXPECT_LE(page.slot_offset.at(i) + page.bytes.at(i), page.length) << p;
      EXPECT_LE(page.page_offset.at(i) + page.bytes.at(i), kPagedExtent) << p;
      // The piece's first and last byte: in the slab at (p·2 MiB +
      // page offset - δ), which is group e's byte (that - e·S); in the
      // file at file_offset + slot offset, which must be file[e] + it.
      for (const std::uint64_t at : {std::uint64_t{0}, page.bytes.at(i) - 1}) {
        const std::uint64_t slab_at = (p * kPagedExtent) + page.page_offset.at(i) + at - slab.delta;
        const std::uint64_t e = slab_at / slab.stride;
        const std::uint64_t in_group = slab_at - (e * slab.stride);
        ASSERT_LT(e, file.size()) << p;
        ASSERT_LT(in_group, stored) << "page " << p << " copies a gap byte";
        EXPECT_EQ(page.shard, shard[e]) << p;
        EXPECT_EQ(page.file_offset + page.slot_offset.at(i) + at, file[e] + in_group) << p;
      }
      covered += page.bytes.at(i);
    }
  }
  EXPECT_EQ(covered, stored * file.size());  // every stored byte, once
}

TEST(SlabLayoutTest, PagesCoverEveryStoredByteOnceWithinOneFile) {
  const std::uint64_t stored =
      (std::uint64_t{3} << 20U) + (std::uint64_t{3} * 4096);  // 3 MiB + 12 KiB
  const std::uint64_t stride = stored + 3296;
  std::vector<std::uint32_t> shard(9, 4);
  std::vector<std::uint64_t> file(9);
  for (std::size_t e = 0; e < file.size(); ++e) {
    file[e] = 8192 + (e * stored);
  }
  const auto slab = LayOutSlab(shard, file, stored, stride);
  ASSERT_TRUE(slab.has_value()) << slab.error();
  EXPECT_EQ(slab->delta, 0U);
  EXPECT_EQ(slab->pages.size(), ((8 * stride) + stored + kPagedExtent - 1) / kPagedExtent);
  CheckCovers(*slab, shard, file, stored);
}

TEST(SlabLayoutTest, AShardChangeFallsOnAPageBoundaryInAGap) {
  const std::uint64_t stored = (std::uint64_t{5} << 20U) + (std::uint64_t{7} * 4096);
  const std::uint64_t stride = stored + 3296;
  std::vector<std::uint32_t> shard = {0, 0, 0, 1, 1, 1, 1};
  std::vector<std::uint64_t> file(shard.size());
  for (std::size_t e = 0; e < file.size(); ++e) {
    file[e] = (shard[e] == 0 ? 4096 : 12288) + ((e - (shard[e] == 0 ? 0 : 3)) * stored);
  }
  const auto slab = LayOutSlab(shard, file, stored, stride);
  ASSERT_TRUE(slab.has_value()) << slab.error();
  EXPECT_EQ(slab->delta % 256, 0U);
  // The boundary: group 3 starts within the gap's 3,296 bytes after a page.
  const std::uint64_t start = slab->delta + (3 * stride);
  EXPECT_LE(start % kPagedExtent, stride - stored);
  CheckCovers(*slab, shard, file, stored);
}

TEST(SlabLayoutTest, GroupsNotConsecutiveInTheFileOrChangingShardTwiceAreRefused) {
  const std::uint64_t stored = std::uint64_t{3} << 20U;
  const std::uint64_t stride = stored + 4096;
  const std::vector<std::uint32_t> one = {0, 0, 0};
  EXPECT_TRUE(Failed(LayOutSlab(one, std::vector<std::uint64_t>{0, stored + 4096, 2 * stored},
                                stored, stride))
                  .has_value());
  const std::vector<std::uint32_t> twice = {0, 1, 2};
  EXPECT_TRUE(
      Failed(LayOutSlab(twice, std::vector<std::uint64_t>{0, 0, 0}, stored, stride)).has_value());
  EXPECT_TRUE(Failed(LayOutSlab(one, std::vector<std::uint64_t>{0, stored, 2 * stored}, stored,
                                stored - 4096))
                  .has_value());  // a stride below the stored bytes
}

// A shard change that no 256-byte δ can put in the gap (δ − r is the
// change's position mod 256, 160 here, against a 16-byte gap): refused,
// never a page that reads two files.
TEST(SlabLayoutTest, AShardChangeTheGapCannotAlignIsRefused) {
  const std::uint64_t stored = std::uint64_t{3} << 20U;
  const std::uint64_t stride = stored + 16;
  std::vector<std::uint32_t> shard(12, 0);
  std::vector<std::uint64_t> file(shard.size());
  for (std::size_t e = 0; e < file.size(); ++e) {
    shard[e] = e < 10 ? 0 : 1;
    file[e] = (e < 10 ? e : e - 10) * stored;
  }
  const auto slab = LayOutSlab(shard, file, stored, stride);
  EXPECT_FALSE(slab.has_value());
  // The same groups in one shard lay out.
  const std::vector<std::uint32_t> one(shard.size(), 0);
  std::vector<std::uint64_t> consecutive(shard.size());
  for (std::size_t e = 0; e < consecutive.size(); ++e) {
    consecutive[e] = e * stored;
  }
  const auto whole = LayOutSlab(one, consecutive, stored, stride);
  ASSERT_TRUE(whole.has_value()) << whole.error();
  CheckCovers(*whole, one, consecutive, stored);
}

// Qwen3.8's slabs (engine/qwen38_runner.h): 512 groups of 2,768,896
// stored bytes at a stride of 2,768,976, an 80-byte gap. Where the change
// of shard needs δ ≡ r (mod 256) with r's remainder leaving more than 80
// bytes to the next 256, the default alignment is refused; at the stride's
// own 16 every change lays out.
TEST(SlabLayoutTest, AnAlignmentOfSixteenPutsAnyShardChangeInAnEightyByteGap) {
  const std::uint64_t stored = 2768896;
  const std::uint64_t stride = 2768976;
  std::size_t change = 0;
  for (std::size_t e = 1; e < 512 && change == 0; ++e) {
    const std::uint64_t r = (kPagedExtent - ((e * stride) % kPagedExtent)) % kPagedExtent;
    if (r % 256 != 0 && 256 - (r % 256) > stride - stored) {
      change = e;
    }
  }
  ASSERT_NE(change, 0U);
  std::vector<std::uint32_t> shard(512);
  std::vector<std::uint64_t> file(shard.size());
  for (std::size_t e = 0; e < file.size(); ++e) {
    shard[e] = e < change ? 3 : 4;
    file[e] = 4096 + ((e < change ? e : e - change) * stored);
  }
  EXPECT_FALSE(LayOutSlab(shard, file, stored, stride).has_value());
  const auto slab = LayOutSlab(shard, file, stored, stride, 16);
  ASSERT_TRUE(slab.has_value()) << slab.error();
  EXPECT_EQ(slab->delta % 16, 0U);
  EXPECT_EQ((slab->delta + (change * stride)) % kPagedExtent, 0U);
  CheckCovers(*slab, shard, file, stored);
  // Alignments that are not a power of two from 16 to 4,096 are refused.
  EXPECT_FALSE(LayOutSlab(shard, file, stored, stride, 8).has_value());
  EXPECT_FALSE(LayOutSlab(shard, file, stored, stride, 48).has_value());
  EXPECT_FALSE(LayOutSlab(shard, file, stored, stride, 8192).has_value());
}

}  // namespace
