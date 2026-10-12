// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "components/spatial.h"
#include "engine/graph_plan.h"
#include "expected_error.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/validate_ext.h"
#include "spec/image_spatial.h"

namespace {
namespace co = llmp::components;
namespace en = llmp::engine;
namespace ex = llmp::execution;
namespace fm = llmp::format;
namespace ir = llmp::ir;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace mo = llmp::model;
namespace sp = llmp::spec;
using llmp::test_support::Failed;

std::expected<ex::Registry, ex::PlanRejection> Registry() {
  std::vector<ex::Implementation> entries;
  for (const auto& declaration : kg::CapabilityDeclarations()) {
    entries.push_back({.name = std::string(declaration.name),
                       .operation = declaration.operation,
                       .source = "ggml",
                       .revision = "host_descriptor_control",
                       .build = "host",
                       .variant = "unchanged_actual_capability",
                       .capability = *declaration.capability});
  }
  return ex::Registry::Create(std::move(entries));
}
sp::ModelSpec Spec() {
  sp::Stage stage{.name = "vae", .role = sp::StageRole::kVae, .artifact = 0};
  stage.stream.width = 2;
  sp::Layer layer;
  layer.components.push_back(
      {.kind = "upsample.nearest", .version = 1, .params = {{"factor", std::int64_t{2}}}});
  layer.components.push_back(
      {.kind = "conv.spatial",
       .version = 1,
       .params = {{"in_channels", std::int64_t{2}},
                  {"out_channels", std::int64_t{3}},
                  {"kernel", std::int64_t{3}},
                  {"padding", std::int64_t{1}}},
       .weights = {{.role = "kernel", .tensor = "kernel"}, {.role = "bias", .tensor = "bias"}}});
  stage.layers.push_back(std::move(layer));
  sp::ModelSpec model{.pipeline = sp::PipelineKind::kImage, .stages = {std::move(stage)}};
  sp::Canonicalize(model);
  return model;
}
struct Budget {
  std::uint64_t used = 0;
  co::SpatialPayment Payment() {
    return {[this](std::uint64_t bytes) {
              if (bytes > (32ULL << 20U) - used) return false;
              used += bytes;
              return true;
            },
            [this](std::uint64_t bytes) { used -= bytes; }};
  }
};

// Descriptor-only controls: these addresses are never dereferenced or allocated.
void CheckNearestCoordinateBounds() {
  const auto* capability = kg::CapabilityOf("ggml.upsample.nearest");
  ASSERT_NE(capability, nullptr);
  ASSERT_NE(capability->accepts, nullptr);
  const ir::UpsampleParams params{.factor = 2};
  for (std::size_t axis = 0; axis < 4; ++axis) {
    for (const bool beyond : {false, true}) {
      ggml_tensor input{};
      ggml_tensor output{};
      input.type = output.type = GGML_TYPE_F32;
      input.data = reinterpret_cast<void*>(0x100000000ULL);
      output.data = reinterpret_cast<void*>(0x200000000ULL);
      output.op = GGML_OP_UPSCALE;
      output.op_params[0] = GGML_SCALE_MODE_NEAREST;
      output.src[0] = &input;
      std::array<ex::OperandFormat, 2> operands{};
      operands[1].operand = ex::kOutputOperand;
      std::uint64_t input_stride = sizeof(float), output_stride = sizeof(float);
      for (std::size_t d = 0; d < 4; ++d) {
        const std::int64_t size = d == axis ? (1LL << 24U) + (beyond ? 4 : 0) : (d < 2 ? 2 : 1);
        output.ne[d] = size;
        input.ne[d] = d < 2 ? size / 2 : size;
        input.nb[d] = input_stride;
        output.nb[d] = output_stride;
        operands[0].ne[d] = static_cast<std::uint64_t>(input.ne[d]);
        operands[1].ne[d] = static_cast<std::uint64_t>(output.ne[d]);
        input_stride *= operands[0].ne[d];
        output_stride *= operands[1].ne[d];
      }
      operands[0].memory.emplace();
      operands[1].memory.emplace();
      std::ranges::copy(input.nb, operands[0].memory->nb.begin());
      std::ranges::copy(output.nb, operands[1].memory->nb.begin());
      const ex::CapabilityQuery query{.op = capability->op,
                                      .operands = operands,
                                      .arch = 121,
                                      .params = std::as_bytes(std::span(&params, 1))};
      EXPECT_EQ(capability->accepts(query), !beyond) << axis;
      EXPECT_EQ(kg::CheckUpsampleNearest(&output).has_value(), !beyond) << axis;
    }
  }
}

TEST(GgmlSpatialTest, RaggedTypedPortsLowerWithoutTokenMetadataInBothBindingModes) {
  CheckNearestCoordinateBounds();
  auto registry = Registry();
  ASSERT_TRUE(registry) << Failed(registry, &ex::PlanRejection::detail).value_or("");
  Budget budget;
  auto spec = Spec();
  const std::vector<co::BoundTensor> tensors{
      {"kernel", ir::Dense(fm::Encoding::kF32, ir::ShapeOf({3, 3, 2, 3}))},
      {"bias", ir::Dense(fm::Encoding::kF32, ir::ShapeOf({3}))}};
  const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
  const std::array owners{co::SpatialOwner{1, {3, 2, 2, 1}}, co::SpatialOwner{4, {2, 3, 2, 1}}};
  auto graph = co::BuildSpatialStep(spec, tables, owners, budget.Payment());
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  en::GraphBuildIO io{graph->graph(), graph->graph().inputs(), graph->graph().outputs(),
                      graph->weights()};
  kn::StepBackend backend{.registry = &*registry,
                          .device = {.arch = 121},
                          .written = kg::WrittenBytes,
                          .within = kg::WithinRoot};
  for (const bool measuring : {true, false}) {
    kn::Addresses addresses;
    addresses.weight = [](const ir::WeightRef& ref, fm::PartRole) {
      return 0x20000000ULL + (std::uint64_t{ref.role} << 16U);
    };
    addresses.activations = measuring ? 0ULL : 0x10000000ULL;
    addresses.activation_bytes = 1U << 20U;
    addresses.activation_alignment = 256;
    addresses.weight_alignment = [](const ir::WeightRef&, fm::PartRole) { return 256; };
    auto plan = en::PlanComponentGraph(
        io, backend, {}, addresses,
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = kg::WrittenBytes,
         .within = kg::WithinRoot,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
    ASSERT_TRUE(plan) << Failed(plan).value_or("");
    const kn::LowerInput input{*plan, io, addresses};
    EXPECT_FALSE(input.token);
    auto lowered = kg::LowerStep(input, *registry, {});
    ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
    ASSERT_EQ(lowered->inputs.size(), 2U);
    ASSERT_EQ(lowered->outputs.size(), 2U);
    ASSERT_EQ(lowered->plan.steps.size(), 6U);
    for (std::size_t owner = 0; owner < 2; ++owner) {
      EXPECT_EQ(lowered->inputs[owner].kind, kn::InputKind::kStepBytes);
      EXPECT_EQ(lowered->inputs[owner].tensor, io.inputs[owner]);
      EXPECT_EQ(lowered->inputs[owner].segment, owners[owner].owner);
      EXPECT_EQ(lowered->outputs[owner].tensor, io.outputs[owner]);
      EXPECT_EQ(lowered->outputs[owner].segment, owners[owner].owner);
      EXPECT_EQ(lowered->outputs[owner].bytes, 6U * 4U * 3U * sizeof(float));
    }
    for (auto& step : lowered->plan.steps) {
      if (step.implementation == "ggml.conv2d.f32") {
        ASSERT_TRUE(kg::CheckConv2dF32(step.nodes[0]));
        auto* node = step.nodes[0];
        const auto saved = node->src[0]->type;
        node->src[0]->type = GGML_TYPE_BF16;
        EXPECT_FALSE(kg::CheckConv2dF32(node));
        node->src[0]->type = saved;
        node->op_params[4] = 2;
        EXPECT_FALSE(kg::CheckConv2dF32(node));
        node->op_params[4] = 1;
      } else if (step.implementation == "ggml.upsample.nearest") {
        ASSERT_TRUE(kg::CheckUpsampleNearest(step.nodes[0]));
        auto* node = step.nodes[0];
        node->op_params[0] = GGML_SCALE_MODE_BILINEAR;
        EXPECT_FALSE(kg::CheckUpsampleNearest(node));
        node->op_params[0] = GGML_SCALE_MODE_NEAREST;
      } else {
        EXPECT_EQ(step.implementation, "ggml.add");
      }
    }
    auto bad = io;
    bad.outputs = bad.outputs.first(1);
    EXPECT_FALSE(kg::LowerStep(kn::LowerInput{*plan, bad, addresses}, *registry, {}));
  }
  auto unsupported = tensors;
  unsupported[0].format = ir::Dense(fm::Encoding::kBF16, ir::ShapeOf({3, 3, 2, 3}));
  const std::array<std::span<const co::BoundTensor>, 1> unsupported_tables{unsupported};
  auto bf16 = co::BuildSpatialStep(spec, unsupported_tables, owners, budget.Payment());
  ASSERT_TRUE(bf16) << Failed(bf16).value_or("");
  en::GraphBuildIO bf16_io{bf16->graph(), bf16->graph().inputs(), bf16->graph().outputs(),
                           bf16->weights()};
  EXPECT_FALSE(en::PlanComponentGraph(bf16_io, backend, {}, kn::Addresses{}));
}

TEST(GgmlSpatialTest, ActualVaeSourceSpansPreserveWeightsAndRefuseBranches) {
  const auto& profile = mo::QwenImage21().vae;
  const auto steps = mo::VaeDecoderPlan(profile, 64, 64);
  const auto tensors = mo::VaeDecoderTensors(profile);
  ASSERT_GT(steps.size(), 2U);
  auto prefix = sp::VaeSpatialSpec(std::span(steps).first(2), tensors);
  ASSERT_TRUE(prefix) << Failed(prefix).value_or("");
  const auto& components = prefix->stages[0].layers[0].components;
  ASSERT_EQ(components.size(), 2U);
  for (std::size_t i = 0; i < 2; ++i) {
    const auto kernel =
        std::ranges::find(components[i].weights, "kernel", &sp::WeightBinding::role);
    ASSERT_NE(kernel, components[i].weights.end());
    EXPECT_EQ(kernel->tensor, i == 0 ? "post_quant_conv.weight" : "decoder.conv_in.weight");
  }
  EXPECT_FALSE(sp::VaeSpatialSpec(std::span(steps).first(3), tensors));
  auto broken = std::vector(steps.begin(), steps.begin() + 2);
  broken[1].in = mo::kVaeQ;
  EXPECT_FALSE(sp::VaeSpatialSpec(broken, tensors));
  auto wrong = tensors;
  wrong[static_cast<std::size_t>(steps[0].weight)].dtype = "BF16";
  EXPECT_FALSE(sp::VaeSpatialSpec(std::span(steps).first(2), wrong));
  std::size_t spans = 0;
  for (std::size_t i = 0; i + 1 < steps.size(); ++i) {
    if (steps[i].kind != mo::VaeStep::Kind::kUpsample) continue;
    auto span = sp::VaeSpatialSpec(std::span(steps).subspan(i, 2), tensors);
    ASSERT_TRUE(span) << Failed(span).value_or("");
    EXPECT_EQ(span->stages[0].layers[0].components[0].kind, "upsample.nearest");
    ++spans;
  }
  EXPECT_EQ(spans, 4U);
}
}  // namespace
