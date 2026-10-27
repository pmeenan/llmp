// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "components/step.h"
#include "ir/builder.h"
#include "kernels/common/step_program.h"
#include "kernels/decode/implementations.h"
#include "kernels/decode/operand.h"
#include "kernels/decode/products.h"
#include "kernels/decode/selected_grouped.h"
#include "kernels/decode/selected_rows.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/step_program.h"
#include "kernels/ggml/tensors.h"
#include "planner/plan.h"

namespace {
namespace co = llmp::components;
namespace de = llmp::kernels::decode;
namespace ex = llmp::execution;
namespace fmt = llmp::format;
namespace ir = llmp::ir;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace pl = llmp::planner;
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
                     .stacked = 2,
                     .shape = ir::ShapeOf({32, 64, 4})};
    w.expert_weight = kWeight;
    w.parts = {{Role::kCodes, kg::Ref(ggml_new_tensor_2d(c, GGML_TYPE_I8, 1024, 4))},
               {Role::kBlockScales, kg::Ref(ggml_new_tensor_2d(c, GGML_TYPE_I8, 128, 4))},
               {Role::kTensorScale, kg::Ref(ggml_new_tensor_2d(c, GGML_TYPE_I8, 8, 4))}};
    const std::array<std::uint64_t, 3> at{0, 1024, 1280};
    for (std::size_t i = 0; i != w.parts.size(); ++i) {
      auto* t = kg::TensorOf(w.parts[i].tensor);
      t->nb[1] = 4096;
      t->nb[2] = t->nb[3] = 3 * 4096 + static_cast<std::uint64_t>(t->ne[0]);
      kg::TensorArena::Bind(t, 0x100000ULL + at[i]);
    }
    w.tensor = w.parts[0].tensor;
    input[1].desc.format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({32, routed ? 2U : 1U, 3}));
    input[1].desc.nb = ir::NaturalStrides(input[1].desc.format);
    input[1].tensor = kg::Ref(ggml_new_tensor_3d(c, GGML_TYPE_F32, 32, routed ? 2 : 1, 3));
    kg::TensorArena::Bind(kg::TensorOf(input[1].tensor), 0x200000);
    input[2].desc.format = ir::Dense(fmt::Encoding::kI32, ir::ShapeOf({2, 3}));
    input[2].desc.nb = ir::NaturalStrides(input[2].desc.format);
    input[2].tensor = kg::Ref(ggml_new_tensor_2d(c, GGML_TYPE_I32, 2, 3));
    kg::TensorArena::Bind(kg::TensorOf(input[2].tensor), 0x210000);
    output[0].desc.format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({64, 2, 3}));
    output[0].desc.nb = ir::NaturalStrides(output[0].desc.format);
    output[0].tensor = kg::Ref(ggml_new_tensor_3d(c, GGML_TYPE_F32, 64, 2, 3));
    kg::TensorArena::Bind(kg::TensorOf(output[0].tensor), 0x220000);
    node.key = ir::Primitive(ir::Op::kMatMulId);
    node.nsrc = 3;
    node.nout = 1;
  }
};
TEST(DecodeSelectedGrouped, StackedPaddedExpertsRetainEveryPhysicalPart) {
  auto source = de::GroupedSelectedSource();
  ASSERT_TRUE(source);
  const auto implementations = (*source)->Implementations();
  ASSERT_EQ(implementations.size(), static_cast<std::size_t>(de::DecoderId::kCount));
  const auto at = std::ranges::find(implementations, "llmp.decode.grouped.nvfp4.split",
                                    &llmp::execution::Implementation::name);
  ASSERT_NE(at, implementations.end());
  const auto& impl = *at;
  for (const bool routed : {false, true}) {
    Inputs d(routed);
    auto call = (*source)->Lower(impl, d.node, d.input, d.output);
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
  const auto impls = (*source)->Implementations();
  const auto at = std::ranges::find(impls, "llmp.decode.grouped.nvfp4.split",
                                    &llmp::execution::Implementation::name);
  ASSERT_NE(at, impls.end());
  const auto& impl = *at;
  for (int bad = 0; bad != 6; ++bad) {
    SCOPED_TRACE(bad);
    Inputs d;
    auto& ids = d.input[2];
    // The actual E4/U2 Route view: each token retains the argsort row pitch.
    ids.desc.nb = {4, 16, 48, 48};
    if (bad == 1) ids.desc.nb[0] = 8;
    if (bad == 2) ids.desc.nb[1] = 4;
    if (bad == 3) ids.desc.nb[1] = 17;
    if (bad == 4) ids.desc.nb[1] = (std::uint64_t{INT32_MAX} + 1) * 4;
    for (std::size_t i = 0; i != 4; ++i) kg::TensorOf(ids.tensor)->nb[i] = ids.desc.nb[i];
    if (bad == 5) kg::TensorOf(ids.tensor)->nb[1] = 24;
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
    kg::TensorOf(ids.tensor)->nb[1] = 24;
    EXPECT_FALSE((*call)->StageDescriptors(std::as_writable_bytes(std::span(descriptors))));
    EXPECT_TRUE(std::ranges::equal(std::as_bytes(std::span(descriptors)), saved));
  }
}

TEST(DecodeSelectedGrouped, ForeignSourceAndLogicalGeometryRefuse) {
  auto source = de::GroupedSelectedSource();
  ASSERT_TRUE(source);
  const auto implementations = (*source)->Implementations();
  const auto impl = *std::ranges::find(implementations, "llmp.decode.grouped.nvfp4.split",
                                       &ex::Implementation::name);
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
  const auto implementations = (*source)->Implementations();
  const auto impl = *std::ranges::find(implementations, "llmp.decode.grouped.nvfp4.split",
                                       &ex::Implementation::name);
  for (int bad = 0; bad != 10; ++bad) {
    SCOPED_TRACE(bad);
    Inputs d;
    auto call = (*source)->Lower(impl, d.node, d.input, d.output);
    ASSERT_TRUE(call);
    auto* codes = kg::TensorOf(d.input[0].parts[0].tensor);
    auto* scales = kg::TensorOf(d.input[0].parts[1].tensor);
    auto* tensor = kg::TensorOf(d.input[0].parts[2].tensor);
    if (bad == 0) kg::TensorArena::Bind(codes, 0x100001);
    if (bad == 1) kg::TensorArena::Bind(tensor, 0x100501);
    if (bad == 2) scales->nb[1] = 8192;
    if (bad == 3) kg::TensorArena::Bind(scales, 0x100200);
    if (bad == 4) codes->ne[1] = 3;
    if (bad == 5) kg::TensorArena::Bind(kg::TensorOf(d.output[0].tensor), 0x200000);
    if (bad == 6) kg::TensorArena::Bind(kg::TensorOf(d.input[2].tensor), 0x210001);
    if (bad == 7) {
      kg::TensorOf(d.input[1].tensor)->ne[0] = 16;
      kg::TensorOf(d.input[1].tensor)->ne[1] = 2;
    }
    if (bad == 8) tensor->nb[1] = UINT64_MAX;
    if (bad == 9) kg::TensorArena::Bind(kg::TensorOf(d.output[0].tensor), 0x102000);
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
  const auto impls = (*source)->Implementations();
  const auto at = std::ranges::find(impls, "llmp.decode.grouped.nvfp4.split",
                                    &llmp::execution::Implementation::name);
  ASSERT_NE(at, impls.end());
  auto call = (*source)->Lower(*at, d.node, d.input, d.output);
  ASSERT_TRUE(call);
  EXPECT_FALSE((*call)->RouteBuffers({0x300001, 0x301000, 0x302000, 0x303000}));
  EXPECT_TRUE((*call)->RouteBuffers({0x300000, 0x301000, 0x302000, 0x303000}));
  EXPECT_FALSE((*call)->RouteBuffers({0x400000, 0x401000, 0x402000, 0x403000}));
}
TEST(DecodeSelectedGrouped, AffineGroupedExpertsLowerAndStageDescriptors) {
  auto source = de::GroupedSelectedSource();
  ASSERT_TRUE(source);
  const auto impls = (*source)->Implementations();
  const auto affine_at = std::ranges::find(impls, "llmp.decode.grouped.affine.split",
                                           &llmp::execution::Implementation::name);
  ASSERT_NE(affine_at, impls.end());
  const auto nvfp4_at = std::ranges::find(impls, "llmp.decode.grouped.nvfp4.split",
                                          &llmp::execution::Implementation::name);
  ASSERT_NE(nvfp4_at, impls.end());

  kg::TensorArena arena(std::move(*kg::TensorArena::Create(8)));
  auto* c = arena.context();
  std::array<kg::SelectedOperand, 3> input{};
  std::array<kg::SelectedOperand, 1> output{};
  ir::Node node;

  auto& w = input[0];
  w.desc.format = {.encoding = fmt::Encoding::kAffine,
                   .params = {.bits = 4, .group = 32, .scale = fmt::AffineScale::kBF16},
                   .layout = fmt::Layout::kSplit,
                   .stacked = 1,
                   .shape = ir::ShapeOf({32, 64, 4})};
  w.expert_weight = kWeight;
  // K=32, bits=4: codes = 16 bytes/row, N=64 -> 1024 bytes per expert
  // group=32, BF16: scales = 2 bytes/row, N=64 -> 128 bytes per expert
  // biases: 128 bytes per expert
  w.parts = {{Role::kCodes, kg::Ref(ggml_new_tensor_2d(c, GGML_TYPE_I8, 1024, 4))},
             {Role::kBlockScales, kg::Ref(ggml_new_tensor_2d(c, GGML_TYPE_I8, 128, 4))},
             {Role::kBiases, kg::Ref(ggml_new_tensor_2d(c, GGML_TYPE_I8, 128, 4))}};
  const std::array<std::uint64_t, 3> at{0, 1024, 1152};
  for (std::size_t i = 0; i != w.parts.size(); ++i) {
    auto* t = kg::TensorOf(w.parts[i].tensor);
    t->nb[1] = 4096;
    t->nb[2] = t->nb[3] = 3 * 4096 + static_cast<std::uint64_t>(t->ne[0]);
    kg::TensorArena::Bind(t, 0x100000ULL + at[i]);
  }
  w.tensor = w.parts[0].tensor;
  input[1].desc.format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({32, 1, 3}));
  input[1].desc.nb = ir::NaturalStrides(input[1].desc.format);
  input[1].tensor = kg::Ref(ggml_new_tensor_3d(c, GGML_TYPE_F32, 32, 1, 3));
  kg::TensorArena::Bind(kg::TensorOf(input[1].tensor), 0x200000);
  input[2].desc.format = ir::Dense(fmt::Encoding::kI32, ir::ShapeOf({2, 3}));
  input[2].desc.nb = ir::NaturalStrides(input[2].desc.format);
  input[2].tensor = kg::Ref(ggml_new_tensor_2d(c, GGML_TYPE_I32, 2, 3));
  kg::TensorArena::Bind(kg::TensorOf(input[2].tensor), 0x210000);
  output[0].desc.format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({64, 2, 3}));
  output[0].desc.nb = ir::NaturalStrides(output[0].desc.format);
  output[0].tensor = kg::Ref(ggml_new_tensor_3d(c, GGML_TYPE_F32, 64, 2, 3));
  kg::TensorArena::Bind(kg::TensorOf(output[0].tensor), 0x220000);
  node.key = ir::Primitive(ir::Op::kMatMulId);
  node.nsrc = 3;
  node.nout = 1;

  // NVFP4 implementation rejects affine weight
  EXPECT_FALSE((*source)->Lower(*nvfp4_at, node, input, output));

  // Affine implementation accepts affine weight
  auto call = (*source)->Lower(*affine_at, node, input, output);
  ASSERT_TRUE(call) << call.error().detail;
  alignas(256) std::array<de::Operand, 4> descriptors{};
  ASSERT_TRUE((*call)->StageDescriptors(std::as_writable_bytes(std::span(descriptors))));
  for (std::size_t e = 0; e != descriptors.size(); ++e) {
    const auto& x = descriptors[e];
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(x.data), 0x100000U + e * 4096);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(x.scales), 0x100400U + e * 4096);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(x.biases), 0x100480U + e * 4096);
    EXPECT_EQ(x.k, 32U);
    EXPECT_EQ(x.rows, 64U);
    EXPECT_EQ(x.row_bytes, 16U);
  }
}
TEST(DecodeSelectedGrouped, ExplicitIdentityIncludesRecipeAndDefaultSourcesRemainUnqualified) {
  const auto first = de::GroupedSelectedSource(), second = de::GroupedSelectedSource();
  const auto rows = de::RowsSelectedSource();
  ASSERT_TRUE(first && second && rows);
  ASSERT_TRUE((*first)->ExecutionIdentity());
  EXPECT_EQ((*first)->ExecutionIdentity(), (*second)->ExecutionIdentity());
  EXPECT_NE(*(*first)->ExecutionIdentity(), llmp::base::Sha256Digest{});
  EXPECT_TRUE((*rows)->ExecutionIdentity());
  const auto impl = (*first)->Implementations();
  EXPECT_NE(*(*first)->ExecutionIdentity(), llmp::execution::IdentityOf(impl[0]));
}

TEST(DecodeSelectedGrouped, AffineSplitExpertPartBindsLeavesInLowerStep) {
  auto source = de::GroupedSelectedSource();
  ASSERT_TRUE(source);
  auto implementations = kg::Implementations();
  const auto native = (*source)->Implementations();
  implementations.insert(implementations.end(), native.begin(), native.end());
  auto registry = ex::Registry::Create(implementations);
  ASSERT_TRUE(registry) << registry.error().detail;

  for (const bool alias : {false, true}) {
    SCOPED_TRACE(alias);
    ir::Builder builder;
    const fmt::TensorFormat format = {
        .encoding = fmt::Encoding::kAffine,
        .params = {.bits = 4, .group = 32, .scale = fmt::AffineScale::kBF16},
        .layout = fmt::Layout::kSplit,
        .stacked = 1,
        .shape = ir::ShapeOf({32, 64, 4})};
    const auto weight = builder.Weight(format, kWeight);
    const auto x = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({32, 1, 3})));
    const auto logits = builder.Input(ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({4, 3})));
    const auto route = builder.Route(logits, {}, {}, {},
                                     ir::RouteParams{.normalize = true, .experts = 4, .used = 2});
    const auto product_ids = alias ? builder.View(route[0], builder.tensor(route[0]).shape(),
                                                  builder.tensor(route[0]).nb, 0)
                                   : route[0];
    const auto out = builder.MatMulId(weight, x, product_ids);
    builder.Output(out);
    // Mixed consumers: direct product keeps the pitched source; byte readers
    // and the exposed IDs share a separately placed compact materialization.
    builder.Output(builder.View(route[0], ir::ShapeOf({2, 1}), builder.tensor(route[0]).nb, 8));
    builder.Output(builder.Reshape(route[0], ir::ShapeOf({6})));
    builder.Output(route[0]);
    auto graph = std::move(builder).Finish();
    ASSERT_TRUE(graph) << graph.error().detail;

    co::StepGraph step{
        .graph = std::move(*graph),
        .inputs = {{.tensor = x, .kind = co::InputKind::kFeatures, .component = kWeight.component},
                   {.tensor = logits,
                    .kind = co::InputKind::kFeatures,
                    .component = kWeight.component}},
        .weights = {{.tensor = weight,
                     .ref = kWeight,
                     .parts = {{"codes", Role::kCodes},
                               {"scales", Role::kBlockScales},
                               {"biases", Role::kBiases}}}},
        .state = {},
        .logits = {}};

    const pl::DeviceFacts device{.device_class = "sm_121", .arch = 121, .features = ex::kBf16Mma};
    auto plan = pl::Plan(step.graph.Clone(), *registry, device, {},
                         {.written = kg::WrittenBytes, .within = kg::WithinRoot});
    ASSERT_TRUE(plan) << plan.error().detail;
    EXPECT_TRUE(std::ranges::any_of(plan->steps, [&](const auto& s) {
      return registry->at(s.implementation).name == "llmp.decode.grouped.affine.split";
    }));

    auto one = format;
    one.shape.rank = 2;
    one.shape.ne[2] = one.shape.ne[3] = 1;
    const auto extents = fmt::PartsOf(one);
    ASSERT_TRUE(extents);
    constexpr std::uint64_t kPitch = 4096;
    constexpr std::uint64_t kMatrices = 4;

    const pl::StepShape shape{
        .stage = 0,
        .phase = pl::Phase::kDecode,
        .segments = {{.slot = 0,
                      .rows = 3,
                      .past = 0,
                      .read_cells = 2048,
                      .outputs = 3,
                      .want_head = false}},
        .read_align = 256,
    };

    class TestOwner final : public kn::BackendOwner {
     public:
      const kn::StepBackend& backend() const override {
        static const kn::StepBackend b{};
        return b;
      }
      bool Charge(std::uint64_t) const override { return true; }
      void Release(std::uint64_t) const override {}
      std::expected<ex::BackendDeclaration, std::string> Ready() const override {
        return ex::BackendDeclaration{};
      }
    };
    const auto funder = std::make_shared<TestOwner>();

    for (int mode = 0; mode != 3; ++mode) {
      SCOPED_TRACE(mode);
      kn::Addresses addresses{};
      addresses.activations = 0x800000;
      addresses.activation_bytes = 1048576;
      if (mode == 2) {
        addresses.weight = [](const ir::WeightRef&, Role) -> std::uint64_t { return 0x999999; };
      }
      addresses.expert_part = [mode, extents = *extents, one](
                                  std::uint16_t stage,
                                  std::string_view name) -> std::optional<kn::ExpertPartBinding> {
        if (mode == 1 || stage != kWeight.stage) return std::nullopt;
        for (const auto& e : extents.parts()) {
          if ((e.role == Role::kCodes && name == "codes") ||
              (e.role == Role::kBlockScales && name == "scales") ||
              (e.role == Role::kBiases && name == "biases")) {
            const std::uint64_t total_bytes = (kMatrices - 1) * kPitch + e.readable;
            std::uint64_t base_addr = 0x100000;
            if (e.role == Role::kBlockScales) base_addr = 0x200000;
            if (e.role == Role::kBiases) base_addr = 0x300000;
            return kn::ExpertPartBinding{
                .address = base_addr,
                .bytes = total_bytes,
                .pitch = kPitch,
                .format = one,
                .role = e.role,
            };
          }
        }
        return std::nullopt;
      };

      auto lowered = kg::LowerStep({*plan, step, shape, addresses, 2048}, *registry, {},
                                   {.routed_funder = funder, .selected_source = source->get()});
      if (mode == 0 && alias) {
        ASSERT_FALSE(lowered);
        EXPECT_EQ(lowered.error().detail,
                  "lowering onto GGML: dynamic experts lack their original router IDs output");
      } else if (mode == 0) {
        ASSERT_TRUE(lowered) << lowered.error().detail;
        ASSERT_EQ(lowered->selected.size(), 1U);
        const auto* ids = kg::TensorOf(lowered->selected.front().call->route()->ids);
        EXPECT_EQ(ids->nb[1], 4U * sizeof(std::int32_t));
        ASSERT_NE(ids->view_src, nullptr);
        EXPECT_EQ(ggml_nbytes(ids->view_src), 4U * 3U * sizeof(std::int32_t));
        const auto rt = std::ranges::find_if(plan->graph.nodes(), [](const auto& node) {
          return node.key == ir::Primitive(ir::Op::kRoute);
        });
        ASSERT_NE(rt, plan->graph.nodes().end());
        EXPECT_EQ(kg::WrittenBytes(plan->graph, rt->out[0]), 4U * 3U * sizeof(std::int32_t));
        const auto mapped = plan->tensors.at(route[0].v);
        EXPECT_EQ(plan->graph.node(plan->graph.producer(mapped)).key, ir::Primitive(ir::Op::kCont));
        EXPECT_EQ(ggml_nbytes(lowered->keep.back()), 2U * 3U * sizeof(std::int32_t));
        EXPECT_EQ(lowered->keep.back()->nb[1], 2U * sizeof(std::int32_t));
        EXPECT_NE(lowered->keep.back()->data, ids->data);
      } else {
        ASSERT_FALSE(lowered);
      }
    }
  }
}

TEST(DecodeSelectedGrouped, EveryDecoderRetainsPitchedCompletePhysicalParts) {
  auto routed = de::GroupedSelectedSource();
  ASSERT_TRUE(routed);
  auto resident = de::ResidentGroupedSelectedSource();
  ASSERT_TRUE(resident);
  const auto implementations = (*routed)->Implementations();
  for (std::size_t index = 0; index < static_cast<std::size_t>(de::DecoderId::kCount); ++index) {
    const auto decoder = static_cast<de::DecoderId>(index);
    SCOPED_TRACE(de::DecoderName(decoder));
    auto arena = kg::TensorArena::Create(12);
    ASSERT_TRUE(arena);
    auto* c = arena->context();
    std::array<kg::SelectedOperand, 3> inputs{};
    std::array<kg::SelectedOperand, 1> outputs{};
    auto& weight = inputs[0];
    weight.desc.format = {.encoding = de::DecoderEncoding(decoder),
                          .layout = de::DecoderLayout(decoder),
                          .shape = ir::ShapeOf({256, 130, 4})};
    if (weight.desc.format.encoding == fmt::Encoding::kAffine) {
      weight.desc.format.params.bits = 4;
      weight.desc.format.params.group = 32;
      weight.desc.format.params.scale = fmt::AffineScale::kBF16;
    }
    if (weight.desc.format.encoding == fmt::Encoding::kNvfp4) {
      weight.desc.format.params.tensor_scale = true;
      weight.desc.format.stacked = 2;
    }
    weight.expert_weight = kWeight;
    auto matrix = weight.desc.format;
    matrix.shape.rank = 2;
    matrix.shape.ne[2] = 1;
    const auto extents = fmt::PartsOf(matrix);
    ASSERT_TRUE(extents);
    std::uint64_t base = 0x100000;
    for (const auto& e : extents->parts()) {
      auto* t = ggml_new_tensor_2d(c, GGML_TYPE_I8, static_cast<std::int64_t>(e.readable), 4);
      t->nb[1] = (e.readable + 255) / 256 * 256;
      t->nb[2] = t->nb[3] = t->nb[1] * 3 + e.readable;
      kg::TensorArena::Bind(t, base);
      base += 0x100000;
      weight.parts.push_back({e.role, kg::Ref(t)});
    }
    weight.tensor = weight.parts[0].tensor;
    inputs[1].desc.format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({256, 1, 3}));
    inputs[1].desc.nb = ir::NaturalStrides(inputs[1].desc.format);
    inputs[1].tensor = kg::Ref(ggml_new_tensor_3d(c, GGML_TYPE_F32, 256, 1, 3));
    kg::TensorArena::Bind(kg::TensorOf(inputs[1].tensor), 0x1000000);
    inputs[2].desc.format = ir::Dense(fmt::Encoding::kI32, ir::ShapeOf({2, 3}));
    inputs[2].desc.nb = {4, 16, 48, 48};
    inputs[2].tensor = kg::Ref(ggml_new_tensor_2d(c, GGML_TYPE_I32, 2, 3));
    std::ranges::copy(inputs[2].desc.nb, kg::TensorOf(inputs[2].tensor)->nb);
    kg::TensorArena::Bind(kg::TensorOf(inputs[2].tensor), 0x1100000);
    outputs[0].desc.format = ir::Dense(fmt::Encoding::kF32, ir::ShapeOf({130, 2, 3}));
    outputs[0].desc.nb = ir::NaturalStrides(outputs[0].desc.format);
    outputs[0].tensor = kg::Ref(ggml_new_tensor_3d(c, GGML_TYPE_F32, 130, 2, 3));
    kg::TensorArena::Bind(kg::TensorOf(outputs[0].tensor), 0x1200000);
    ir::Node node;
    node.key = ir::Primitive(ir::Op::kMatMulId);
    node.nsrc = 3;
    node.nout = 1;
    const auto at = std::ranges::find(
        implementations,
        de::ImplementationName({.tmpl = de::Template::kGrouped, .gate = decoder, .up = decoder}),
        &ex::Implementation::name);
    ASSERT_NE(at, implementations.end());
    auto call = (*routed)->Lower(*at, node, inputs, outputs);
    ASSERT_TRUE(call) << call.error().detail;
    alignas(256) std::array<de::Operand, 4> descriptors{};
    auto staged = (*call)->StageDescriptors(std::as_writable_bytes(std::span(descriptors)));
    ASSERT_TRUE(staged) << staged.error().detail;
    EXPECT_EQ(descriptors[3].k, 256U);
    EXPECT_EQ(descriptors[3].rows, 130U);
    auto ordinary = (*resident)->Lower(*at, node, inputs, outputs);
    ASSERT_TRUE(ordinary) << ordinary.error().detail;
    // IR accepts an explicit degenerate fourth dimension as the same experts.
    weight.desc.format.shape.rank = 4;
    auto rank4 = (*routed)->Lower(*at, node, inputs, outputs);
    ASSERT_TRUE(rank4) << rank4.error().detail;
    auto rank4_resident = (*resident)->Lower(*at, node, inputs, outputs);
    ASSERT_TRUE(rank4_resident) << rank4_resident.error().detail;
    weight.desc.format.shape.ne[3] = 2;
    EXPECT_FALSE((*routed)->Lower(*at, node, inputs, outputs));
    EXPECT_FALSE((*resident)->Lower(*at, node, inputs, outputs));
    weight.desc.format.shape.ne[3] = 1;
    // Complete-read coverage is authoritative even when the pitch is larger.
    kg::TensorOf(weight.parts[0].tensor)->ne[0] -= 1;
    EXPECT_FALSE((*call)->StageDescriptors(std::as_writable_bytes(std::span(descriptors))));
  }
}

}  // namespace
