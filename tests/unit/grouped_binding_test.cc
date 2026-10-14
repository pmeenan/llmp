// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/decode/grouped_binding.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include "execution/registry.h"
#include "ir/graph.h"
#include "kernels/decode/implementations.h"
#include "kernels/decode/products.h"

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

TEST(GroupedBindingTest, AffineExpertArrayBindsDescriptorsWithScalesAndBiases) {
  fmt::TensorFormat format{.encoding = fmt::Encoding::kAffine,
                           .params = {.bits = 4, .group = 32, .scale = fmt::AffineScale::kBF16},
                           .layout = fmt::Layout::kSplit,
                           .shape = llmp::ir::ShapeOf({64, 16, 4})};
  std::array<de::PitchedExpertPart, 3> parts{{{Role::kCodes, 0x10000, 4 * 512, 512},
                                              {Role::kBlockScales, 0x20000, 4 * 64, 64},
                                              {Role::kBiases, 0x30000, 4 * 64, 64}}};
  std::array<de::Operand, 4> out{};
  const auto bound = de::GroupedBindingBound(format);
  ASSERT_TRUE(bound);
  EXPECT_EQ(bound->experts, 4U);
  EXPECT_EQ(bound->descriptors.value(), 4 * sizeof(de::Operand));
  ASSERT_TRUE(de::BindGroupedInto(format, parts, out));
  for (std::size_t e = 0; e != 4; ++e) {
    const auto& op = out[e];
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(op.data), 0x10000 + e * 512);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(op.scales), 0x20000 + e * 64);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(op.biases), 0x30000 + e * 64);
    EXPECT_EQ(op.k, 64U);
    EXPECT_EQ(op.rows, 16U);
    EXPECT_EQ(op.group, 32U);
    EXPECT_EQ(op.bits, 4U);
    EXPECT_EQ(op.scale_type, 2U);
    EXPECT_EQ(op.row_bytes, 32U);
    EXPECT_EQ(op.scale_row_bytes, 4U);
  }
}

TEST(GroupedBindingTest, AffineOperandOfChecksGeometryAndParts) {
  alignas(16) std::array<std::uint8_t, 512> codes{};
  alignas(16) std::array<std::uint8_t, 64> scales{};
  alignas(16) std::array<std::uint8_t, 64> biases{};

  fmt::TensorFormat f{.encoding = fmt::Encoding::kAffine,
                      .params = {.bits = 4, .group = 32, .scale = fmt::AffineScale::kBF16},
                      .layout = fmt::Layout::kSplit,
                      .shape = llmp::ir::ShapeOf({64, 16})};

  std::array<de::DevicePart, 3> parts = {
      de::DevicePart{.role = fmt::PartRole::kCodes, .address = codes.data(), .bytes = 512},
      de::DevicePart{.role = fmt::PartRole::kBlockScales, .address = scales.data(), .bytes = 64},
      de::DevicePart{.role = fmt::PartRole::kBiases, .address = biases.data(), .bytes = 64}};

  auto op = de::OperandOf({.format = f, .parts = parts});
  ASSERT_TRUE(op.has_value()) << op.error();
  EXPECT_EQ(op->k, 64U);
  EXPECT_EQ(op->rows, 16U);
  EXPECT_EQ(op->row_bytes, 32U);
  EXPECT_EQ(op->group, 32U);
  EXPECT_EQ(op->bits, 4U);
  EXPECT_EQ(op->scale_type, 2U);
  EXPECT_EQ(op->scale_row_bytes, 4U);
  EXPECT_EQ(op->data, codes.data());
  EXPECT_EQ(op->scales, scales.data());
  EXPECT_EQ(op->biases, biases.data());

  // 8-bit affine with group 64, F16 scale
  fmt::TensorFormat f8{.encoding = fmt::Encoding::kAffine,
                       .params = {.bits = 8, .group = 64, .scale = fmt::AffineScale::kF16},
                       .layout = fmt::Layout::kSplit,
                       .shape = llmp::ir::ShapeOf({64, 8})};
  std::array<de::DevicePart, 3> parts8 = {
      de::DevicePart{.role = fmt::PartRole::kCodes, .address = codes.data(), .bytes = 512},
      de::DevicePart{.role = fmt::PartRole::kBlockScales, .address = scales.data(), .bytes = 16},
      de::DevicePart{.role = fmt::PartRole::kBiases, .address = biases.data(), .bytes = 16}};
  auto op8 = de::OperandOf({.format = f8, .parts = parts8});
  ASSERT_TRUE(op8.has_value()) << op8.error();
  EXPECT_EQ(op8->bits, 8U);
  EXPECT_EQ(op8->group, 64U);
  EXPECT_EQ(op8->scale_type, 1U);
  EXPECT_EQ(op8->row_bytes, 64U);
  EXPECT_EQ(op8->scale_row_bytes, 2U);

  // Zero point mode
  fmt::TensorFormat f_zp = f;
  f_zp.params.zero_point = true;
  std::array<de::DevicePart, 3> parts_zp = {
      de::DevicePart{.role = fmt::PartRole::kCodes, .address = codes.data(), .bytes = 512},
      de::DevicePart{.role = fmt::PartRole::kBlockScales, .address = scales.data(), .bytes = 64},
      de::DevicePart{.role = fmt::PartRole::kZeros, .address = biases.data(), .bytes = 16}};
  auto op_zp = de::OperandOf({.format = f_zp, .parts = parts_zp});
  ASSERT_TRUE(op_zp.has_value()) << op_zp.error();
  EXPECT_TRUE((op_zp->scale_type & 0x80U) != 0);

  // Rejections
  auto f_bad_bits = f;
  f_bad_bits.params.bits = 3;
  EXPECT_FALSE(de::OperandOf({.format = f_bad_bits, .parts = parts}).has_value());

  auto f_bad_grp = f;
  f_bad_grp.params.group = 16;
  EXPECT_FALSE(de::OperandOf({.format = f_bad_grp, .parts = parts}).has_value());

  auto f_bad_scale = f;
  f_bad_scale.params.scale = fmt::AffineScale::kNone;
  EXPECT_FALSE(de::OperandOf({.format = f_bad_scale, .parts = parts}).has_value());
}

TEST(GroupedBindingTest, AffineImplementationsRegistered) {
  const auto implementations = de::Implementations();
  std::vector<std::string_view> names;
  for (const auto& impl : implementations) {
    names.push_back(impl.name);
  }
  EXPECT_TRUE(std::ranges::find(names, "llmp.decode.vector.affine.split") != names.end());
  EXPECT_TRUE(std::ranges::find(names, "llmp.decode.vector_glu.affine.split") != names.end());
  EXPECT_TRUE(std::ranges::find(names, "llmp.decode.matrix.affine.split") != names.end());
  EXPECT_TRUE(std::ranges::find(names, "llmp.decode.matrix_fma.affine.split") != names.end());
  EXPECT_TRUE(std::ranges::find(names, "llmp.decode.grouped.affine.split") != names.end());
  EXPECT_TRUE(std::ranges::find(names, "llmp.decode.rows.affine.split") != names.end());

  for (const auto& impl : implementations) {
    if (impl.name.find("affine.split") != std::string_view::npos) {
      ASSERT_TRUE(impl.capability.has_value());
      EXPECT_EQ(impl.capability->rung, llmp::execution::Rung::kOnChipDecode);
      EXPECT_TRUE(impl.capability->row_invariant);
      EXPECT_TRUE(impl.capability->capturable);
      EXPECT_EQ(impl.capability->shape.k_multiple, 16U);
    }
  }
}
}  // namespace
