// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// No addresses are dereferenced: these control what can be queued.
#include <gtest/gtest.h>

#include <cstdint>
#include <expected>
#include <initializer_list>
#include <limits>

#include "kernels/ggml/dsv4_ds4_cache.h"

namespace {
namespace kg = jitllm::kernels::ggml;

constexpr std::uint64_t kBase = 1ULL << 40;
constexpr std::uint64_t kSlot = 1ULL << 36;

void Refused(const std::expected<void, kg::KernelFailure>& checked) {
  ASSERT_FALSE(checked.has_value());
  EXPECT_EQ(checked.error().error, kg::KernelError::kRejected);
}

kg::Ds4CacheQat Qat(kg::Ds4CacheKind kind) {
  const bool kv = kind == kg::Ds4CacheKind::kKv512;
  return {.kind = kind,
          .values = {kBase, 3ULL * (kv ? 512 : 128) * 4},
          .codes = {kBase + kSlot, 3ULL * (kv ? 704 : 64)},
          .scales = {kBase + (2 * kSlot), 3ULL * (kv ? 28 : 16)},
          .rows = 3};
}

kg::Ds4CacheExpand Expand(kg::Ds4CacheKind kind) {
  const auto qat = Qat(kind);
  return {.kind = kind,
          .values = qat.values,
          .codes = qat.codes,
          .scales = qat.scales,
          .decode_table = kind == kg::Ds4CacheKind::kKv512
                              ? kg::Ds4CacheBuffer{kBase + (3 * kSlot), 512}
                              : kg::Ds4CacheBuffer{},
          .rows = qat.rows};
}

TEST(Ds4CacheValidateTest, QatRequiresBoundedDisjointViewsAndCompletePackedOutputs) {
  for (auto kind : {kg::Ds4CacheKind::kKv512, kg::Ds4CacheKind::kIndexer128}) {
    const auto valid = Qat(kind);
    ASSERT_TRUE(kg::CheckDs4CacheQat(valid));
    auto desc = valid;
    --desc.values.bytes;
    Refused(kg::CheckDs4CacheQat(desc));
    desc = valid;
    ++desc.values.address;
    Refused(kg::CheckDs4CacheQat(desc));
    desc = valid;
    --desc.codes.bytes;
    Refused(kg::CheckDs4CacheQat(desc));
    desc = valid;
    --desc.scales.bytes;
    Refused(kg::CheckDs4CacheQat(desc));
    desc = valid;
    desc.scales.address = desc.codes.address;
    Refused(kg::CheckDs4CacheQat(desc));
    desc = valid;
    desc.codes.address = desc.values.address + desc.values.bytes - 4;
    Refused(kg::CheckDs4CacheQat(desc));
    desc = valid;
    desc.scales = {};
    Refused(kg::CheckDs4CacheQat(desc));
    desc.codes = {};
    ASSERT_TRUE(kg::CheckDs4CacheQat(desc));
    desc = valid;
    desc.codes = {};
    if (kind == kg::Ds4CacheKind::kKv512)
      Refused(kg::CheckDs4CacheQat(desc));
    else
      ASSERT_TRUE(kg::CheckDs4CacheQat(desc));
    desc = valid;
    desc.scales = {0, desc.scales.bytes};
    Refused(kg::CheckDs4CacheQat(desc));
    desc = valid;
    desc.values.address = std::numeric_limits<std::uint64_t>::max() - 3;
    Refused(kg::CheckDs4CacheQat(desc));
    desc = valid;
    desc.rows = 0;
    Refused(kg::CheckDs4CacheQat(desc));
    desc.rows = std::numeric_limits<std::uint32_t>::max();
    Refused(kg::CheckDs4CacheQat(desc));
    desc = valid;
    // Deliberately construct an unknown fixed-underlying enum value.
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    desc.kind = static_cast<kg::Ds4CacheKind>(255);
    Refused(kg::CheckDs4CacheQat(desc));
  }
}

TEST(Ds4CacheValidateTest, ExpansionRequiresTheAccountedTableOnlyForKv) {
  auto desc = Expand(kg::Ds4CacheKind::kKv512);
  ASSERT_TRUE(kg::CheckDs4CacheExpand(desc));
  desc.decode_table.bytes = 511;
  Refused(kg::CheckDs4CacheExpand(desc));
  desc = Expand(kg::Ds4CacheKind::kKv512);
  desc.decode_table.address = desc.scales.address;
  Refused(kg::CheckDs4CacheExpand(desc));
  desc = Expand(kg::Ds4CacheKind::kKv512);
  desc.decode_table = {};
  Refused(kg::CheckDs4CacheExpand(desc));
  desc = Expand(kg::Ds4CacheKind::kIndexer128);
  ASSERT_TRUE(kg::CheckDs4CacheExpand(desc));
  desc.decode_table = {kBase + (3 * kSlot), 512};
  Refused(kg::CheckDs4CacheExpand(desc));
  desc = Expand(kg::Ds4CacheKind::kIndexer128);
  desc.values.address = desc.codes.address;
  Refused(kg::CheckDs4CacheExpand(desc));
}

TEST(Ds4CacheValidateTest, RawRingAllowsOneWrapWithoutRepeatedWriteSlotsOrPositionOverflow) {
  const kg::Ds4CacheRawStore valid{.source = {kBase, 3ULL * 512 * 4},
                                   .ring = {kBase + kSlot, 4ULL * 512 * 4},
                                   .first = 3,
                                   .rows = 3,
                                   .cells = 4};
  ASSERT_TRUE(kg::CheckDs4CacheRawStore(valid));
  auto desc = valid;
  desc.rows = 5;
  Refused(kg::CheckDs4CacheRawStore(desc));
  desc = valid;
  desc.first = std::numeric_limits<std::uint32_t>::max() - 1;
  Refused(kg::CheckDs4CacheRawStore(desc));
  desc.rows = 1;
  ASSERT_TRUE(kg::CheckDs4CacheRawStore(desc));
  desc = valid;
  desc.cells = 8193;
  Refused(kg::CheckDs4CacheRawStore(desc));
  desc = valid;
  desc.ring.address = desc.source.address + 4;
  Refused(kg::CheckDs4CacheRawStore(desc));
  desc = valid;
  --desc.ring.bytes;
  Refused(kg::CheckDs4CacheRawStore(desc));
}

TEST(Ds4CacheValidateTest, DecodeTableIncludesOriginalFiniteReservedEntry) {
  const auto table = kg::Ds4CacheDecodeTable();
  EXPECT_EQ(table[0], 0.0f);
  EXPECT_EQ(table[1], 1.0f / 512);
  EXPECT_EQ(table[8], 1.0f / 64);
  EXPECT_EQ(table[56], 1.0f);
  EXPECT_EQ(table[126], 448.0f);
  // Original table covers all128 magnitudes; the encoder caps at126.
  EXPECT_EQ(table[127], 480.0f);
}

}  // namespace
