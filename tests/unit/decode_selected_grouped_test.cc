// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include "kernels/decode/operand.h"
#include "kernels/decode/selected_grouped.h"
#include "kernels/decode/selected_rows.h"
#include "kernels/ggml/tensors.h"

namespace {
namespace de = llmp::kernels::decode;
namespace kg = llmp::kernels::ggml;
namespace ir = llmp::ir;
namespace fmt = llmp::format;
using Role = fmt::PartRole;
constexpr ir::WeightRef kWeight{.stage = 0, .component = 2, .role = 8, .expert_array = true};
struct Inputs {
  kg::TensorArena arena;
  std::array<kg::SelectedOperand, 3> input{};
  std::array<kg::SelectedOperand, 1> output{};
  ir::Node node;
  explicit Inputs(bool routed = false) : arena(std::move(*kg::TensorArena::Create(8))) {
    auto* c = arena.context();
    auto& w = input[0];
    w.desc.format = {.encoding = fmt::Encoding::kNvfp4,
                     .params = {.tensor_scale = true},
                     .layout = fmt::Layout::kSplit,
                     .shape = ir::ShapeOf({32, 64, 4}),
                     .stacked = 2};
    w.expert_weight = kWeight;
    w.parts = {{Role::kCodes, ggml_new_tensor_2d(c, GGML_TYPE_I8, 1024, 4)},
               {Role::kBlockScales, ggml_new_tensor_2d(c, GGML_TYPE_I8, 128, 4)},
               {Role::kTensorScale, ggml_new_tensor_2d(c, GGML_TYPE_I8, 8, 4)}};
    const std::array<std::uint64_t, 3> at{0, 1024, 1280};
    for (std::size_t i = 0; i != w.parts.size(); ++i) {
      auto* t = w.parts[i].tensor;
      t->nb[1] = 4096;
      t->nb[2] = t->nb[3] = 3 * 4096 + static_cast<std::uint64_t>(t->ne[0]);
      kg::TensorArena::Bind(t, 0x100000ULL + at[i]);
    }
    w.tensor = w.parts[0].tensor;
    input[1].desc.format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({32, routed ? 2U : 1U, 3}));
    input[1].desc.nb = ir::NaturalStrides(input[1].desc.format);
    input[1].tensor = ggml_new_tensor_3d(c, GGML_TYPE_F32, 32, routed ? 2 : 1, 3);
    kg::TensorArena::Bind(input[1].tensor, 0x200000);
    input[2].desc.format = ir::Dense(fmt::Encoding::kI32, ir::ShapeOf({2, 3}));
    input[2].desc.nb = ir::NaturalStrides(input[2].desc.format);
    input[2].tensor = ggml_new_tensor_2d(c, GGML_TYPE_I32, 2, 3);
    kg::TensorArena::Bind(input[2].tensor, 0x210000);
    output[0].desc.format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({64, 2, 3}));
    output[0].desc.nb = ir::NaturalStrides(output[0].desc.format);
    output[0].tensor = ggml_new_tensor_3d(c, GGML_TYPE_F32, 64, 2, 3);
    kg::TensorArena::Bind(output[0].tensor, 0x220000);
    node.key = ir::Primitive(ir::Op::kMatMulId);
    node.nsrc = 3;
    node.nout = 1;
  }
};
TEST(DecodeSelectedGrouped, StackedPaddedExpertsRetainEveryPhysicalPart) {
  auto source = de::GroupedSelectedSource();
  ASSERT_TRUE(source);
  const auto implementations = (*source)->Implementations();
  ASSERT_EQ(implementations.size(), 1U);
  for (const bool routed : {false, true}) {
    Inputs d(routed);
    auto call = (*source)->Lower(implementations[0], d.node, d.input, d.output);
    ASSERT_TRUE(call) << call.error().detail;
    ASSERT_TRUE((*call)->route());
    EXPECT_EQ((*call)->route()->weight, kWeight);
    EXPECT_EQ((*call)->route()->format.stacked, 2U);
    EXPECT_EQ((*call)->route()->format.shape.ne[2], 4U);
    alignas(256) std::array<de::Operand, 4> descriptors{};
    ASSERT_TRUE((*call)->StageDescriptors(std::as_writable_bytes(std::span(descriptors))));
    // Entire independent descriptor construction: actual padded pitch per part,
    // not packed expert size and not one expert's shape changing stacked.
    for (std::size_t e = 0; e != descriptors.size(); ++e) {
      const auto& x = descriptors[e];
      EXPECT_EQ(reinterpret_cast<std::uintptr_t>(x.data), 0x100000U + e * 4096);
      EXPECT_EQ(reinterpret_cast<std::uintptr_t>(x.scales), 0x100400U + e * 4096);
      EXPECT_EQ(reinterpret_cast<std::uintptr_t>(x.tensor_scale), 0x100500U + e * 4096);
      EXPECT_EQ(x.k, 32U);
      EXPECT_EQ(x.rows, 64U);
      EXPECT_EQ(x.rows_per_scale, 32U);
      EXPECT_EQ(x.row_bytes, 16U);
    }
  }
}
TEST(DecodeSelectedGrouped, PitchedRouterIdsRetainExactSourceGeometry) {
  auto source = de::GroupedSelectedSource();
  ASSERT_TRUE(source);
  const auto impl = (*source)->Implementations()[0];
  for (int bad = 0; bad != 5; ++bad) {
    SCOPED_TRACE(bad);
    Inputs d;
    auto& ids = d.input[2];
    // The actual E4/U2 Route view: each token retains the argsort row pitch.
    ids.desc.nb = {4, 16, 48, 48};
    if (bad == 1) ids.desc.nb[0] = 8;
    if (bad == 2) ids.desc.nb[1] = 4;
    if (bad == 3) ids.desc.nb[1] = 17;
    if (bad == 4) ids.desc.nb[1] = (std::uint64_t{INT32_MAX} + 1) * 4;
    for (std::size_t i = 0; i != 4; ++i) ids.tensor->nb[i] = ids.desc.nb[i];
    auto call = (*source)->Lower(impl, d.node, d.input, d.output);
    if (bad != 0) {
      EXPECT_FALSE(call);
      continue;
    }
    ASSERT_TRUE(call);
    ASSERT_TRUE((*call)->route());
    EXPECT_EQ((*call)->route()->ids, ids.tensor);
    alignas(256) std::array<de::Operand, 4> descriptors{};
    ASSERT_TRUE((*call)->StageDescriptors(std::as_writable_bytes(std::span(descriptors))));
    const auto before = std::as_bytes(std::span(descriptors));
    std::array<std::byte, sizeof(descriptors)> saved{};
    std::ranges::copy(before, saved.begin());
    ids.tensor->nb[1] = 24;
    EXPECT_FALSE((*call)->StageDescriptors(std::as_writable_bytes(std::span(descriptors))));
    EXPECT_TRUE(std::ranges::equal(std::as_bytes(std::span(descriptors)), saved));
  }
}

TEST(DecodeSelectedGrouped, ForeignSourceAndLogicalGeometryRefuse) {
  auto source = de::GroupedSelectedSource();
  ASSERT_TRUE(source);
  const auto impl = (*source)->Implementations()[0];
  for (int bad = 0; bad != 6; ++bad) {
    SCOPED_TRACE(bad);
    Inputs d;
    if (bad == 0) d.input[0].expert_weight.reset();
    if (bad == 1) d.input[0].expert_weight->expert_array = false;
    if (bad == 2) d.input[0].parts.pop_back();
    if (bad == 3) d.output[0].desc.format.encoding = fmt::Encoding::kBF16;
    if (bad == 4) d.input[1].desc.format.shape.ne[1] = 3;
    if (bad == 5) d.input[0].desc.format.stacked = 3;
    EXPECT_FALSE((*source)->Lower(impl, d.node, d.input, d.output));
  }
  Inputs d;
  auto foreign = impl;
  foreign.revision = "foreign-source";
  EXPECT_FALSE((*source)->Lower(foreign, d.node, d.input, d.output));
}
TEST(DecodeSelectedGrouped, ChangedBindingsRefuseBeforeAnyDescriptorWrite) {
  auto source = de::GroupedSelectedSource();
  ASSERT_TRUE(source);
  const auto impl = (*source)->Implementations()[0];
  for (int bad = 0; bad != 10; ++bad) {
    SCOPED_TRACE(bad);
    Inputs d;
    auto call = (*source)->Lower(impl, d.node, d.input, d.output);
    ASSERT_TRUE(call);
    auto* codes = d.input[0].parts[0].tensor;
    auto* scales = d.input[0].parts[1].tensor;
    auto* tensor = d.input[0].parts[2].tensor;
    if (bad == 0) kg::TensorArena::Bind(codes, 0x100001);
    if (bad == 1) kg::TensorArena::Bind(tensor, 0x100501);
    if (bad == 2) scales->nb[1] = 8192;
    if (bad == 3) kg::TensorArena::Bind(scales, 0x100200);
    if (bad == 4) codes->ne[1] = 3;
    if (bad == 5) kg::TensorArena::Bind(d.output[0].tensor, 0x200000);
    if (bad == 6) kg::TensorArena::Bind(d.input[2].tensor, 0x210001);
    if (bad == 7) {
      d.input[1].tensor->ne[0] = 16;
      d.input[1].tensor->ne[1] = 2;
    }
    if (bad == 8) tensor->nb[1] = UINT64_MAX;
    if (bad == 9) kg::TensorArena::Bind(d.output[0].tensor, 0x102000);
    alignas(256) std::array<std::byte, 4 * sizeof(de::Operand)> out;
    out.fill(std::byte{0x5a});
    const auto saved = out;
    EXPECT_FALSE((*call)->StageDescriptors(out));
    EXPECT_EQ(out, saved);
  }
}
TEST(DecodeSelectedGrouped, WorklistPlacementIsAlignedSingleAssignment) {
  auto source = de::GroupedSelectedSource();
  ASSERT_TRUE(source);
  Inputs d;
  auto call = (*source)->Lower((*source)->Implementations()[0], d.node, d.input, d.output);
  ASSERT_TRUE(call);
  EXPECT_FALSE((*call)->RouteBuffers({0x300001, 0x301000, 0x302000, 0x303000}));
  EXPECT_TRUE((*call)->RouteBuffers({0x300000, 0x301000, 0x302000, 0x303000}));
  EXPECT_FALSE((*call)->RouteBuffers({0x400000, 0x401000, 0x402000, 0x403000}));
}
TEST(DecodeSelectedGrouped, ExplicitIdentityIncludesRecipeAndDefaultSourcesRemainUnqualified) {
  const auto first = de::GroupedSelectedSource(), second = de::GroupedSelectedSource();
  const auto rows = de::RowsSelectedSource();
  ASSERT_TRUE(first && second && rows);
  ASSERT_TRUE((*first)->ExecutionIdentity());
  EXPECT_EQ((*first)->ExecutionIdentity(), (*second)->ExecutionIdentity());
  EXPECT_NE(*(*first)->ExecutionIdentity(), llmp::base::Sha256Digest{});
  EXPECT_FALSE((*rows)->ExecutionIdentity());
  const auto impl = (*first)->Implementations();
  EXPECT_NE(*(*first)->ExecutionIdentity(), llmp::execution::IdentityOf(impl[0]));
}
}  // namespace
