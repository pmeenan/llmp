// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/decode/grouped_binding.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>

#include "ir/graph.h"

namespace {
namespace de = llmp::kernels::decode;
namespace fmt = llmp::format;
using Role = fmt::PartRole;
struct Fixture {
  fmt::TensorFormat format{.encoding = fmt::Encoding::kNvfp4,
                           .params = {.tensor_scale = true},
                           .layout = fmt::Layout::kSplit,
                           .shape = llmp::ir::ShapeOf({32, 16, 4}),
                           .stacked = 2};
  // Each pitch deliberately differs from the packed per-expert share.
  std::array<de::PitchedExpertPart, 3> parts{{{Role::kCodes, 0x10000, 4 * 512, 512},
                                              {Role::kBlockScales, 0x20000, 4 * 64, 64},
                                              {Role::kTensorScale, 0x30000, 4 * 16, 16}}};
  std::array<de::Operand, 4> out{};
  Fixture() {
    for (auto& op : out) op.pad = 0xFFFFFFFFU;
  }
  auto Run() { return de::BindGroupedInto(format, parts, out); }
  void Unchanged() {
    for (const auto& op : out) EXPECT_EQ(op.pad, 0xFFFFFFFFU);
  }
};
TEST(GroupedBindingTest, AllExpertsUseActualPitchesAndPreserveStackedScales) {
  Fixture f;
  const auto bound = de::GroupedBindingBound(f.format);
  ASSERT_TRUE(bound);
  EXPECT_EQ(bound->experts, 4U);
  EXPECT_EQ(bound->descriptors.value(), 4 * sizeof(de::Operand));
  ASSERT_TRUE(f.Run());
  for (std::size_t e = 0; e != 4; ++e) {
    const auto& op = f.out[e];
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(op.data), 0x10000 + e * 512);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(op.scales), 0x20000 + e * 64);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(op.tensor_scale), 0x30000 + e * 16);
    EXPECT_EQ(op.k, 32U);
    EXPECT_EQ(op.rows, 16U);
    EXPECT_EQ(op.rows_per_scale, 8U);
    EXPECT_EQ(op.row_bytes, 16U);
  }
}
TEST(GroupedBindingTest, InterleavedSlabChecksConsumedIntervalsAcrossExperts) {
  Fixture f;
  f.parts = {{{Role::kCodes, 0x10000, 4 * 512, 512},
              {Role::kBlockScales, 0x10100, 4 * 512 - 256, 512},
              {Role::kTensorScale, 0x10120, 4 * 512 - 288, 512}}};
  ASSERT_TRUE(f.Run());
  for (std::size_t e = 0; e != 4; ++e) {
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(f.out[e].scales), 0x10100 + e * 512);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(f.out[e].tensor_scale), 0x10120 + e * 512);
  }
  // Scale of one expert intersects the next expert's codes, though expert0
  // alone would be disjoint. Validate the complete sets before any writes.
  for (auto& op : f.out) op.pad = 0xFFFFFFFFU;
  f.parts[2].address = 0x101fc;
  EXPECT_FALSE(f.Run());
  f.Unchanged();
}
TEST(GroupedBindingTest, AllGeometryRefusesAtomicallyBeforeDescriptorWrites) {
  for (int bad = 0; bad != 8; ++bad) {
    SCOPED_TRACE(bad);
    Fixture f;
    switch (bad) {
      case 0:
        f.parts[1].role = Role::kCodes;
        break;
      case 1:
        f.parts[0].matrix_pitch = 128;
        break;
      case 2:
        f.parts[1].bytes = 64;
        break;
      case 3:
        f.parts[2].address += 1;
        break;
      case 4:
        f.parts[0].matrix_pitch += 1;
        break;
      case 5:
        f.parts[1].address = f.parts[0].address;
        break;
      case 6:
        f.parts[0].address = std::numeric_limits<std::uint64_t>::max() - 100;
        break;
      case 7:
        f.parts[2].matrix_pitch = 4;
        break;
    }
    EXPECT_FALSE(f.Run());
    f.Unchanged();
  }
}
TEST(GroupedBindingTest, ScalePresenceFollowsTheFullFormatAndExactOutputCapacity) {
  Fixture f;
  EXPECT_FALSE(de::BindGroupedInto(f.format, std::span(f.parts).first(2), f.out));
  f.Unchanged();
  f.format.params.tensor_scale = false;
  EXPECT_FALSE(f.Run());
  f.Unchanged();
  ASSERT_TRUE(de::BindGroupedInto(f.format, std::span(f.parts).first(2), f.out));
  for (const auto& op : f.out) EXPECT_EQ(op.tensor_scale, nullptr);
  EXPECT_FALSE(
      de::BindGroupedInto(f.format, std::span(f.parts).first(2), std::span(f.out).first(3)));
}
}  // namespace
