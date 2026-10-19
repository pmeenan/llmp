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
#include <string>
#include <vector>

#include "execution/registry.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "kernels/decode/implementations.h"
#include "kernels/decode/operand.h"
#include "kernels/decode/products.h"
#include "kernels/decode/selected_products.h"
#include "kernels/decode/selected_rows.h"
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
  explicit WorkspaceSource(std::uint64_t bytes, std::string name = {}, bool capturable = false)
      : bytes_(bytes), name_(std::move(name)), capturable_(capturable) {}
  std::vector<ex::Implementation> Implementations() const override {
    if (name_.empty()) return {};
    ex::Implementation implementation{};
    implementation.name = name_;
    implementation.operation = llmp::ir::Primitive(llmp::ir::Op::kAdd);
    implementation.capability =
        ex::Capability{.op = ir::Primitive(ir::Op::kAdd), .capturable = true};
    return {implementation};
  }
  std::vector<ex::TunedShape> Tuning(std::uint32_t) const override { return {}; }
  bool Owns(std::string_view name) const override { return !name_.empty() && name == name_; }
  std::uint64_t persistent_bytes() const override { return bytes_; }
  std::expected<std::unique_ptr<kg::SelectedCall>, kg::KernelFailure> Lower(
      const ex::Implementation&, const ir::Node&, std::span<const kg::SelectedOperand>,
      std::span<const kg::SelectedOperand>) const override {
    class Call final : public kg::SelectedCall {
     public:
      using SelectedCall::SelectedCall;
      std::uint64_t scratch_bytes() const override { return 0; }
      std::uint64_t host_bytes() const override { return sizeof(*this); }
      std::uint64_t launches() const override { return 0; }
      std::expected<std::unique_ptr<kg::BoundSelectedCall>, kg::KernelFailure> Bind(
          const ex::Implementation&, kg::SelectedLaunch&, kg::SelectedWorkspace) const override {
        class Bound final : public kg::BoundSelectedCall {
         public:
          std::expected<void, kg::KernelFailure> Run(llmp::providers::NativeStream) const override {
            return {};
          }
        };
        return std::make_unique<Bound>();
      }
    };
    if (name_.empty()) return std::unexpected(kg::KernelFailure{.detail = "no calls"});
    return std::make_unique<Call>(this);
  }
  std::expected<std::unique_ptr<kg::SelectedLaunch>, kg::KernelFailure> Launch(
      int, llmp::providers::DeviceExecution& execution, llmp::providers::StreamId stream,
      kg::SelectedWorkspace workspace) const override {
    workspaces.push_back(workspace);
    class LaunchState final : public kg::SelectedLaunch {
     public:
      LaunchState(const kg::SelectedSource* owner, llmp::providers::NativeStream native, bool safe)
          : SelectedLaunch(owner, native), safe_(safe) {}
      bool faulted() const override { return false; }
      bool capturable() const override { return safe_; }

     private:
      bool safe_;
    };
    return std::make_unique<LaunchState>(this, *execution.Submission(stream), capturable_);
  }
  mutable std::vector<kg::SelectedWorkspace> workspaces;

 private:
  std::uint64_t bytes_;
  std::string name_;
  bool capturable_;
};

TEST(DecodeSelectedProducts, CompositeCaptureRequiresOnlyTheExactCallsRuntimeOptIn) {
  llmp::providers::fake::FakeDeviceExecution execution;
  const auto stream = execution.CreateStream();
  ASSERT_TRUE(stream);
  const auto unsafe = std::make_shared<WorkspaceSource>(0, "fake.unsafe", false);
  const auto safe = std::make_shared<WorkspaceSource>(0, "fake.safe", true);
  const auto source = de::CompositeSelectedSource({de::CompositeSelectedSource({unsafe}), safe});
  const auto implementations = source->Implementations();
  auto launch = source->Launch(0, execution, *stream, {});
  ASSERT_TRUE(launch);
  EXPECT_FALSE((*launch)->capturable());  // Composite-wide opt-in is conservative.
  ir::Node node;
  auto unsafe_call =
      source->Lower(implementations[0], node, {}, std::span<const kg::SelectedOperand>{});
  auto safe_call =
      source->Lower(implementations[1], node, {}, std::span<const kg::SelectedOperand>{});
  ASSERT_TRUE(unsafe_call);
  ASSERT_TRUE(safe_call);
  EXPECT_FALSE((*unsafe_call)->capturable(**launch));
  EXPECT_TRUE((*safe_call)->capturable(**launch));
  ASSERT_TRUE((*unsafe_call)->Bind(implementations[0], **launch, {}));
  EXPECT_TRUE((*safe_call)->capturable(**launch));
  ASSERT_TRUE((*safe_call)->Bind(implementations[1], **launch, {}));
  EXPECT_FALSE((*unsafe_call)->capturable(**launch));
  auto foreign = safe->Launch(0, execution, *stream, {});
  ASSERT_TRUE(foreign);
  EXPECT_FALSE((*safe_call)->capturable(**foreign));
}

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
  ASSERT_EQ(implementations.size(), static_cast<std::size_t>(de::DecoderId::kCount) * 4 + 2);

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
  ASSERT_TRUE(nvfp4_source);
  EXPECT_EQ((*nvfp4_source)->Implementations().size(), 4U * 4U + 1U);
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

// Every declared decoder must cross the selected-source boundary, rather
// than being covered only by direct RunVector/RunMatrix kernel controls.
fmt::TensorFormat DecoderFormat(de::DecoderId decoder, std::uint64_t matrices = 1) {
  fmt::TensorFormat f{.encoding = de::DecoderEncoding(decoder),
                      .layout = de::DecoderLayout(decoder),
                      .shape = ir::ShapeOf({256, 130, matrices})};
  if (f.encoding == fmt::Encoding::kNvfp4) {
    f.params.tensor_scale = true;
    f.stacked = 2;
  }
  if (f.encoding == fmt::Encoding::kAffine) {
    f.params.bits = 4;
    f.params.group = 32;
    f.params.scale = fmt::AffineScale::kBF16;
  }
  return f;
}
kg::SelectedOperand RawWeight(ggml_context* c, const fmt::TensorFormat& format,
                              std::uint64_t base) {
  kg::SelectedOperand weight;
  weight.desc.format = format;
  weight.desc.nb = ir::NaturalStrides(format);
  const auto extents = fmt::PartsOf(format).value();
  for (const auto& extent : extents.parts()) {
    auto* t = ggml_new_tensor_1d(c, GGML_TYPE_I8, static_cast<std::int64_t>(extent.readable));
    kg::TensorArena::Bind(t, base);
    base += 0x100000;
    weight.parts.push_back({extent.role, t});
  }
  weight.tensor = weight.parts.front().tensor;
  return weight;
}
kg::SelectedOperand Activation(ggml_context* c, const fmt::Shape& shape, fmt::Encoding encoding,
                               ggml_type type, std::uint64_t base) {
  kg::SelectedOperand operand;
  operand.desc.format = ir::Dense(encoding, shape);
  operand.desc.nb = ir::NaturalStrides(operand.desc.format);
  std::array<std::int64_t, 4> ne{};
  for (std::size_t d = 0; d < 4; ++d) ne[d] = static_cast<std::int64_t>(shape.ne[d]);
  operand.tensor = ggml_new_tensor(c, type, std::max<int>(1, shape.rank), ne.data());
  kg::TensorArena::Bind(operand.tensor, base);
  return operand;
}
TEST(DecodeSelectedProducts, EveryDecoderLowersAllProductTemplatesWithBatchedBroadcast) {
  auto source = de::ProductsSelectedSource();
  ASSERT_TRUE(source);
  const auto implementations = (*source)->Implementations();
  for (std::size_t index = 0; index < static_cast<std::size_t>(de::DecoderId::kCount); ++index) {
    const auto decoder = static_cast<de::DecoderId>(index);
    SCOPED_TRACE(de::DecoderName(decoder));
    auto arena = kg::TensorArena::Create(16);
    ASSERT_TRUE(arena);
    auto* c = arena->context();
    const auto format = DecoderFormat(decoder, 2);
    std::array inputs{
        RawWeight(c, format, 0x100000),
        Activation(c, ir::ShapeOf({256, 3, 4, 2}), fmt::Encoding::kF32, GGML_TYPE_F32, 0x1000000)};
    std::array outputs{
        Activation(c, ir::ShapeOf({130, 3, 4, 2}), fmt::Encoding::kF32, GGML_TYPE_F32, 0x2000000)};
    ir::Node node;
    node.key = ir::Primitive(ir::Op::kMatMul);
    node.nsrc = 2;
    node.nout = 1;
    for (const auto tmpl :
         {de::Template::kVector, de::Template::kMatrix, de::Template::kMatrixFma}) {
      const auto name = de::ImplementationName({.tmpl = tmpl, .gate = decoder, .up = decoder});
      const auto at = std::ranges::find(implementations, name, &ex::Implementation::name);
      ASSERT_NE(at, implementations.end());
      auto call = (*source)->Lower(*at, node, inputs, outputs);
      ASSERT_TRUE(call) << call.error().detail;
      EXPECT_EQ((*call)->launches(), 8U);
      // A different matrix count cannot be hidden by equal total elements.
      auto wrong = outputs;
      wrong[0].desc.format.shape = ir::ShapeOf({130, 6, 2, 2});
      EXPECT_FALSE((*source)->Lower(*at, node, inputs, wrong));
    }
    // A row pitch supported by the product templates remains reachable.
    inputs[1].desc.nb = {4, 1040, 3120, 12480};
    std::ranges::copy(inputs[1].desc.nb, inputs[1].tensor->nb);
    const auto vector = std::ranges::find(
        implementations,
        de::ImplementationName({.tmpl = de::Template::kVector, .gate = decoder, .up = decoder}),
        &ex::Implementation::name);
    ASSERT_TRUE((*source)->Lower(*vector, node, inputs, outputs));
  }
}
TEST(DecodeSelectedProducts, EverySameAndMixedGluPairLowersInBothOperandOrders) {
  auto source = de::ProductsSelectedSource();
  ASSERT_TRUE(source);
  for (const auto& impl : (*source)->Implementations()) {
    const auto bound = de::Bind(impl);
    ASSERT_TRUE(bound);
    if (bound->tmpl != de::Template::kVectorGlu) continue;
    SCOPED_TRACE(impl.name);
    for (const bool swapped : {false, true}) {
      auto arena = kg::TensorArena::Create(16);
      ASSERT_TRUE(arena);
      auto* c = arena->context();
      auto gate = RawWeight(c, DecoderFormat(bound->gate), 0x100000);
      auto up = RawWeight(c, DecoderFormat(bound->up), 0x500000);
      std::array inputs{
          swapped ? up : gate, swapped ? gate : up,
          Activation(c, ir::ShapeOf({256, 3, 2}), fmt::Encoding::kF32, GGML_TYPE_F32, 0x1000000)};
      std::array outputs{
          Activation(c, ir::ShapeOf({130, 3, 2}), fmt::Encoding::kF32, GGML_TYPE_F32, 0x2000000)};
      ir::Node node;
      node.key = ir::Fused(ir::FusedKind::kMatMulGlu);
      node.nsrc = 3;
      node.nout = 1;
      const ir::GluParams params{.kind = ir::GluKind::kSwiGlu, .swapped = swapped};
      node.SetParams(params);
      auto call = (*source)->Lower(impl, node, inputs, outputs);
      ASSERT_TRUE(call) << call.error().detail;
      EXPECT_EQ((*call)->launches(), 2U);
    }
  }
}
TEST(DecodeSelectedProducts, CanonicalSingletonRowsAndExpertsUseLogicalExtents) {
  auto source = de::DecodeSelectedSource();
  ASSERT_TRUE(source);
  auto resident = de::DecodeResidentSelectedSource();
  ASSERT_TRUE(resident);
  const auto implementations = (*source)->Implementations();
  for (std::size_t index = 0; index < static_cast<std::size_t>(de::DecoderId::kCount); ++index) {
    const auto decoder = static_cast<de::DecoderId>(index);
    SCOPED_TRACE(de::DecoderName(decoder));
    auto arena = kg::TensorArena::Create(24).value();
    auto* c = arena.context();
    auto format = DecoderFormat(decoder);
    format.shape = ir::ShapeOf({256});
    format.stacked = 1;
    std::array inputs{
        RawWeight(c, format, 0x100000),
        Activation(c, ir::ShapeOf({256, 2}), fmt::Encoding::kF32, GGML_TYPE_F32, 0x1000000)};
    std::array outputs{
        Activation(c, ir::ShapeOf({1, 2}), fmt::Encoding::kF32, GGML_TYPE_F32, 0x2000000)};
    const auto implementation = [&](de::Template tmpl) -> const ex::Implementation& {
      return *std::ranges::find(
          implementations, de::ImplementationName({.tmpl = tmpl, .gate = decoder, .up = decoder}),
          &ex::Implementation::name);
    };
    ir::Node node;
    node.key = ir::Primitive(ir::Op::kMatMul);
    node.nsrc = 2;
    node.nout = 1;
    node.SetParams(ir::ProductParams{});
    for (const auto tmpl :
         {de::Template::kVector, de::Template::kMatrix, de::Template::kMatrixFma}) {
      auto call = (*source)->Lower(implementation(tmpl), node, inputs, outputs);
      ASSERT_TRUE(call) << call.error().detail;
    }
    std::array glu{inputs[0], RawWeight(c, format, 0x500000), inputs[1]};
    node.key = ir::Fused(ir::FusedKind::kMatMulGlu);
    node.nsrc = 3;
    node.SetParams(ir::GluParams{.kind = ir::GluKind::kSwiGlu});
    ASSERT_TRUE((*source)->Lower(implementation(de::Template::kVectorGlu), node, glu, outputs));
    std::array rows{inputs[0],
                    Activation(c, ir::ShapeOf({3}), fmt::Encoding::kI32, GGML_TYPE_I32, 0x3000000)};
    std::array gathered{
        Activation(c, ir::ShapeOf({256, 3}), fmt::Encoding::kF32, GGML_TYPE_F32, 0x4000000)};
    node.key = ir::Primitive(ir::Op::kGetRows);
    node.nsrc = 2;
    ASSERT_TRUE((*source)->Lower(implementation(de::Template::kRows), node, rows, gathered));
    std::array grouped{
        inputs[0],
        Activation(c, ir::ShapeOf({256, 1, 2}), fmt::Encoding::kF32, GGML_TYPE_F32, 0x5000000),
        Activation(c, ir::ShapeOf({1, 2}), fmt::Encoding::kI32, GGML_TYPE_I32, 0x6000000)};
    grouped[0].expert_weight =
        ir::WeightRef{.stage = 0, .component = 7, .role = 0, .expert_array = true};
    std::array products{
        Activation(c, ir::ShapeOf({1, 1, 2}), fmt::Encoding::kF32, GGML_TYPE_F32, 0x7000000)};
    node.key = ir::Primitive(ir::Op::kMatMulId);
    node.nsrc = 3;
    auto call = (*source)->Lower(implementation(de::Template::kGrouped), node, grouped, products);
    ASSERT_TRUE(call) << call.error().detail;
    alignas(256) std::array<de::Operand, 1> descriptor{};
    ASSERT_TRUE((*call)->StageDescriptors(std::as_writable_bytes(std::span(descriptor))));
    EXPECT_EQ(descriptor[0].k, 256U);
    EXPECT_EQ(descriptor[0].rows, 1U);
    ASSERT_TRUE(
        (*resident)->Lower(implementation(de::Template::kGrouped), node, grouped, products));
  }
}

TEST(DecodeSelectedProducts, EveryResidentRowDecoderLowersBatchedTables) {
  auto source = de::RowsSelectedSource();
  ASSERT_TRUE(source);
  const auto implementations = (*source)->Implementations();
  EXPECT_EQ(implementations.size(), static_cast<std::size_t>(de::DecoderId::kCount));
  for (std::size_t index = 0; index < static_cast<std::size_t>(de::DecoderId::kCount); ++index) {
    const auto decoder = static_cast<de::DecoderId>(index);
    SCOPED_TRACE(de::DecoderName(decoder));
    auto arena = kg::TensorArena::Create(12);
    ASSERT_TRUE(arena);
    auto* c = arena->context();
    std::array inputs{
        RawWeight(c, DecoderFormat(decoder, 2), 0x100000),
        Activation(c, ir::ShapeOf({5, 2}), fmt::Encoding::kI32, GGML_TYPE_I32, 0x1000000)};
    std::array outputs{
        Activation(c, ir::ShapeOf({256, 5, 2}), fmt::Encoding::kF32, GGML_TYPE_F32, 0x2000000)};
    ir::Node node;
    node.key = ir::Primitive(ir::Op::kGetRows);
    node.nsrc = 2;
    node.nout = 1;
    const auto at = std::ranges::find(
        implementations,
        de::ImplementationName({.tmpl = de::Template::kRows, .gate = decoder, .up = decoder}),
        &ex::Implementation::name);
    ASSERT_NE(at, implementations.end());
    auto call = (*source)->Lower(*at, node, inputs, outputs);
    ASSERT_TRUE(call) << call.error().detail;
    EXPECT_EQ((*call)->launches(), 2U);
  }
}

}  // namespace
