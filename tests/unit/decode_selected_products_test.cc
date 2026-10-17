// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <vector>

#include "execution/registry.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "kernels/decode/operand.h"
#include "kernels/decode/selected_products.h"
#include "kernels/decode/selected_source.h"
#include "kernels/ggml/tensors.h"
#include "providers/fake/fake_device_execution.h"

namespace {

namespace de = llmp::kernels::decode;
namespace kg = llmp::kernels::ggml;
namespace ir = llmp::ir;
namespace fmt = llmp::format;
namespace ex = llmp::execution;
using Role = fmt::PartRole;

constexpr std::uint32_t kK = 32;
constexpr std::uint32_t kN = 64;
constexpr std::uint32_t kRows = 4;

fmt::TensorFormat AffineWeightFormat() {
  return {.encoding = fmt::Encoding::kAffine,
          .params = {.bits = 4, .group = 32, .scale = fmt::AffineScale::kBF16},
          .layout = fmt::Layout::kSplit,
          .shape = ir::ShapeOf({kK, kN}),
          .stacked = 1};
}

struct SingleProductFixture {
  kg::TensorArena arena;
  std::array<kg::SelectedOperand, 2> inputs{};
  std::array<kg::SelectedOperand, 1> outputs{};
  ir::Node node;

  explicit SingleProductFixture(std::uint32_t rows = kRows)
      : arena(std::move(*kg::TensorArena::Create(8))) {
    auto* c = arena.context();
    auto& w = inputs[0];
    w.desc.format = AffineWeightFormat();
    w.parts = {{Role::kCodes, ggml_new_tensor_2d(c, GGML_TYPE_I8, 16, kN)},
               {Role::kBlockScales, ggml_new_tensor_2d(c, GGML_TYPE_I8, 2, kN)},
               {Role::kBiases, ggml_new_tensor_2d(c, GGML_TYPE_I8, 2, kN)}};
    const std::array<std::uint64_t, 3> addrs{0x100000ULL, 0x101000ULL, 0x102000ULL};
    for (std::size_t i = 0; i < w.parts.size(); ++i) {
      kg::TensorArena::Bind(w.parts[i].tensor, addrs[i]);
    }
    w.tensor = w.parts[0].tensor;

    auto& x = inputs[1];
    x.desc.format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({kK, rows}));
    x.desc.nb = ir::NaturalStrides(x.desc.format);
    x.tensor = ggml_new_tensor_2d(c, GGML_TYPE_F32, kK, rows);
    kg::TensorArena::Bind(x.tensor, 0x200000ULL);

    auto& y = outputs[0];
    y.desc.format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({kN, rows}));
    y.desc.nb = ir::NaturalStrides(y.desc.format);
    y.tensor = ggml_new_tensor_2d(c, GGML_TYPE_F32, kN, rows);
    kg::TensorArena::Bind(y.tensor, 0x300000ULL);

    node.key = ir::Primitive(ir::Op::kMatMul);
    node.nsrc = 2;
    node.nout = 1;
  }
};

struct GluProductFixture {
  kg::TensorArena arena;
  std::array<kg::SelectedOperand, 3> inputs{};
  std::array<kg::SelectedOperand, 1> outputs{};
  ir::Node node;

  explicit GluProductFixture(bool swapped = false) : arena(std::move(*kg::TensorArena::Create(8))) {
    auto* c = arena.context();

    auto& gate = inputs[0];
    gate.desc.format = AffineWeightFormat();
    gate.parts = {{Role::kCodes, ggml_new_tensor_2d(c, GGML_TYPE_I8, 16, kN)},
                  {Role::kBlockScales, ggml_new_tensor_2d(c, GGML_TYPE_I8, 2, kN)},
                  {Role::kBiases, ggml_new_tensor_2d(c, GGML_TYPE_I8, 2, kN)}};
    const std::array<std::uint64_t, 3> gate_addrs{0x100000ULL, 0x101000ULL, 0x102000ULL};
    for (std::size_t i = 0; i < gate.parts.size(); ++i) {
      kg::TensorArena::Bind(gate.parts[i].tensor, gate_addrs[i]);
    }
    gate.tensor = gate.parts[0].tensor;

    auto& up = inputs[1];
    up.desc.format = AffineWeightFormat();
    up.parts = {{Role::kCodes, ggml_new_tensor_2d(c, GGML_TYPE_I8, 16, kN)},
                {Role::kBlockScales, ggml_new_tensor_2d(c, GGML_TYPE_I8, 2, kN)},
                {Role::kBiases, ggml_new_tensor_2d(c, GGML_TYPE_I8, 2, kN)}};
    const std::array<std::uint64_t, 3> up_addrs{0x110000ULL, 0x111000ULL, 0x112000ULL};
    for (std::size_t i = 0; i < up.parts.size(); ++i) {
      kg::TensorArena::Bind(up.parts[i].tensor, up_addrs[i]);
    }
    up.tensor = up.parts[0].tensor;

    auto& x = inputs[2];
    x.desc.format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({kK, kRows}));
    x.desc.nb = ir::NaturalStrides(x.desc.format);
    x.tensor = ggml_new_tensor_2d(c, GGML_TYPE_F32, kK, kRows);
    kg::TensorArena::Bind(x.tensor, 0x200000ULL);

    auto& y = outputs[0];
    y.desc.format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({kN, kRows}));
    y.desc.nb = ir::NaturalStrides(y.desc.format);
    y.tensor = ggml_new_tensor_2d(c, GGML_TYPE_F32, kN, kRows);
    kg::TensorArena::Bind(y.tensor, 0x300000ULL);

    node.key = ir::Fused(ir::FusedKind::kMatMulGlu);
    node.nsrc = 3;
    node.nout = 1;
    node.SetParams(ir::GluParams{.kind = ir::GluKind::kSwiGlu, .swapped = swapped});
  }
};

// A host-only source that records the exact workspace lent to it.
class WorkspaceSource final : public kg::SelectedSource {
 public:
  explicit WorkspaceSource(std::uint64_t bytes) : bytes_(bytes) {}
  std::vector<ex::Implementation> Implementations() const override { return {}; }
  std::vector<ex::TunedShape> Tuning(std::uint32_t) const override { return {}; }
  bool Owns(std::string_view) const override { return false; }
  std::uint64_t persistent_bytes() const override { return bytes_; }
  std::expected<std::unique_ptr<kg::SelectedCall>, kg::KernelFailure> Lower(
      const ex::Implementation&, const ir::Node&, std::span<const kg::SelectedOperand>,
      std::span<const kg::SelectedOperand>) const override {
    return std::unexpected(kg::KernelFailure{.detail = "no calls"});
  }
  std::expected<std::unique_ptr<kg::SelectedLaunch>, kg::KernelFailure> Launch(
      int, llmp::providers::DeviceExecution& execution, llmp::providers::StreamId stream,
      kg::SelectedWorkspace workspace) const override {
    workspaces.push_back(workspace);
    class LaunchState final : public kg::SelectedLaunch {
     public:
      using SelectedLaunch::SelectedLaunch;
      bool faulted() const override { return false; }
    };
    return std::make_unique<LaunchState>(this, *execution.Submission(stream));
  }
  mutable std::vector<kg::SelectedWorkspace> workspaces;

 private:
  std::uint64_t bytes_;
};

TEST(DecodeSelectedProducts, CompositeWorkspacesAlignAndRefuseInvalidBoundsBeforeAnyChild) {
  llmp::providers::fake::FakeDeviceExecution execution;
  const auto stream = execution.CreateStream();
  ASSERT_TRUE(stream);
  const auto a = std::make_shared<WorkspaceSource>(1);
  const auto b = std::make_shared<WorkspaceSource>(257);
  const auto source = de::CompositeSelectedSource({a, b});
  EXPECT_EQ(source->persistent_bytes(), 768U);
  for (const kg::SelectedWorkspace workspace :
       {kg::SelectedWorkspace{.base = 0x100000, .bytes = 767},
        kg::SelectedWorkspace{.base = 0x100001, .bytes = 768},
        kg::SelectedWorkspace{.base = 0, .bytes = 768},
        kg::SelectedWorkspace{.base = UINT64_MAX - 255, .bytes = 768}}) {
    EXPECT_FALSE(source->Launch(0, execution, *stream, workspace));
    EXPECT_TRUE(a->workspaces.empty());
    EXPECT_TRUE(b->workspaces.empty());
  }
  auto launched = source->Launch(0, execution, *stream, {.base = 0x100000, .bytes = 768});
  ASSERT_TRUE(launched);
  ASSERT_EQ(a->workspaces.size(), 1U);
  ASSERT_EQ(b->workspaces.size(), 1U);
  EXPECT_EQ(a->workspaces[0].base, 0x100000U);
  EXPECT_EQ(a->workspaces[0].bytes, 1U);
  EXPECT_EQ(b->workspaces[0].base, 0x100100U);
  EXPECT_EQ(b->workspaces[0].bytes, 257U);
  // Overflow while rounding a child, and while summing individually valid children.
  for (const auto sizes : {std::array<std::uint64_t, 2>{UINT64_MAX, 1},
                           std::array<std::uint64_t, 2>{UINT64_MAX - 255, 256}}) {
    const auto huge = std::make_shared<WorkspaceSource>(sizes[0]);
    const auto tail = std::make_shared<WorkspaceSource>(sizes[1]);
    const auto overflow = de::CompositeSelectedSource({huge, tail});
    EXPECT_EQ(overflow->persistent_bytes(), UINT64_MAX);
    EXPECT_FALSE(overflow->Launch(0, execution, *stream, {.base = 0, .bytes = UINT64_MAX}));
    EXPECT_TRUE(huge->workspaces.empty());
    EXPECT_TRUE(tail->workspaces.empty());
  }
  const auto fence = execution.Record(*stream);
  ASSERT_TRUE(fence);
  execution.Drain();
  ASSERT_TRUE(execution.Query(*fence));
  EXPECT_TRUE(execution.Release(*fence));
  EXPECT_TRUE(execution.DestroyStream(*stream));
}

TEST(DecodeSelectedProducts, DeclaresAllAffineImplementations) {
  auto source = de::ProductsSelectedSource();
  ASSERT_TRUE(source);
  const auto implementations = (*source)->Implementations();
  ASSERT_EQ(implementations.size(), 4U);

  EXPECT_TRUE((*source)->Owns("llmp.decode.vector.affine.split"));
  EXPECT_TRUE((*source)->Owns("llmp.decode.vector_glu.affine.split"));
  EXPECT_TRUE((*source)->Owns("llmp.decode.matrix.affine.split"));
  EXPECT_TRUE((*source)->Owns("llmp.decode.matrix_fma.affine.split"));
  EXPECT_FALSE((*source)->Owns("llmp.decode.grouped.nvfp4.split"));

  EXPECT_TRUE((*source)->ExecutionIdentity().has_value());
  EXPECT_EQ((*source)->persistent_bytes(), 0U);

  auto affine_source = de::ProductsSelectedSource(fmt::Encoding::kAffine);
  ASSERT_TRUE(affine_source);
  EXPECT_EQ((*affine_source)->Implementations().size(), 4U);

  auto nvfp4_source = de::ProductsSelectedSource(fmt::Encoding::kNvfp4);
  EXPECT_FALSE(nvfp4_source.has_value());
}

TEST(DecodeSelectedProducts, VectorAffineLowersSuccessfully) {
  auto source = de::ProductsSelectedSource();
  ASSERT_TRUE(source);
  const auto impls = (*source)->Implementations();
  const auto at =
      std::ranges::find(impls, "llmp.decode.vector.affine.split", &ex::Implementation::name);
  ASSERT_NE(at, impls.end());

  SingleProductFixture f(kRows);
  auto call = (*source)->Lower(*at, f.node, f.inputs, f.outputs);
  ASSERT_TRUE(call) << call.error().detail;
  EXPECT_EQ((*call)->scratch_bytes(), 0U);
  EXPECT_EQ((*call)->launches(), 1U);
}

TEST(DecodeSelectedProducts, VectorGluAffineLowersSuccessfully) {
  auto source = de::ProductsSelectedSource();
  ASSERT_TRUE(source);
  const auto impls = (*source)->Implementations();
  const auto at =
      std::ranges::find(impls, "llmp.decode.vector_glu.affine.split", &ex::Implementation::name);
  ASSERT_NE(at, impls.end());

  for (const bool swapped : {false, true}) {
    GluProductFixture f(swapped);
    auto call = (*source)->Lower(*at, f.node, f.inputs, f.outputs);
    ASSERT_TRUE(call) << call.error().detail;
    EXPECT_EQ((*call)->scratch_bytes(), 0U);
    EXPECT_EQ((*call)->launches(), 1U);
  }
}

TEST(DecodeSelectedProducts, MatrixAffineAcceptsLargeRows) {
  auto source = de::ProductsSelectedSource();
  ASSERT_TRUE(source);
  const auto impls = (*source)->Implementations();

  const auto vec_at =
      std::ranges::find(impls, "llmp.decode.vector.affine.split", &ex::Implementation::name);
  const auto mat_at =
      std::ranges::find(impls, "llmp.decode.matrix.affine.split", &ex::Implementation::name);
  const auto fma_at =
      std::ranges::find(impls, "llmp.decode.matrix_fma.affine.split", &ex::Implementation::name);
  ASSERT_NE(vec_at, impls.end());
  ASSERT_NE(mat_at, impls.end());
  ASSERT_NE(fma_at, impls.end());

  SingleProductFixture f(32);  // 32 rows > 16 (max vector rows)

  auto vec_call = (*source)->Lower(*vec_at, f.node, f.inputs, f.outputs);
  EXPECT_FALSE(vec_call.has_value());

  auto mat_call = (*source)->Lower(*mat_at, f.node, f.inputs, f.outputs);
  ASSERT_TRUE(mat_call) << mat_call.error().detail;

  auto fma_call = (*source)->Lower(*fma_at, f.node, f.inputs, f.outputs);
  ASSERT_TRUE(fma_call) << fma_call.error().detail;
}

TEST(DecodeSelectedProducts, DecodeSelectedSourceMultiplexesAll) {
  auto decode_source = de::DecodeSelectedSource();
  ASSERT_TRUE(decode_source);

  EXPECT_TRUE((*decode_source)->Owns("llmp.decode.grouped.affine.split"));
  EXPECT_TRUE((*decode_source)->Owns("llmp.decode.grouped.nvfp4.split"));
  EXPECT_TRUE((*decode_source)->Owns("llmp.decode.rows.affine.split"));
  EXPECT_TRUE((*decode_source)->Owns("llmp.decode.rows.nvfp4.split"));
  EXPECT_TRUE((*decode_source)->Owns("llmp.decode.vector.affine.split"));
  EXPECT_TRUE((*decode_source)->Owns("llmp.decode.vector_glu.affine.split"));
  EXPECT_TRUE((*decode_source)->Owns("llmp.decode.matrix.affine.split"));
  EXPECT_TRUE((*decode_source)->Owns("llmp.decode.matrix_fma.affine.split"));

  EXPECT_TRUE((*decode_source)->ExecutionIdentity().has_value());
}

}  // namespace
