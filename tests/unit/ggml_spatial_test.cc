// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "components/spatial.h"
#include "engine/graph_plan.h"
#include "expected_error.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/validate.h"
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

TEST(GgmlSpatialTest, CompleteActualDecoderGeometryPlansAndLowersWithoutTokenMetadata) {
  const auto& profile = mo::QwenImage21().vae;
  const auto steps = mo::VaeDecoderPlan(profile, 2, 2);
  const auto source = mo::VaeDecoderTensors(profile);
  ASSERT_EQ(steps.size(), 113U);
  ASSERT_EQ(source.size(), 128U);
  auto mapped = sp::VaeDecoderSpec(steps, source);
  ASSERT_TRUE(mapped) << Failed(mapped).value_or("");
  ASSERT_EQ(mapped->links.size(), 109U);  // saved Copy aliases need no primitive/component
  std::vector<co::BoundTensor> tensors;
  std::uint64_t source_bytes = 0;
  for (const auto& weight : source) {
    fm::Shape shape;
    shape.rank = static_cast<std::uint8_t>(weight.shape.size());
    std::uint64_t bytes = sizeof(float);
    for (std::size_t d = 0; d < weight.shape.size(); ++d) {
      shape.ne[d] = weight.shape[weight.shape.size() - d - 1];
      bytes *= shape.ne[d];
    }
    source_bytes += bytes;
    tensors.push_back({weight.name, ir::Dense(fm::Encoding::kF32, shape)});
  }
  EXPECT_EQ(source_bytes, 1012281936ULL);
  Budget budget;
  const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
  const std::array owners{co::SpatialOwner{2, {2, 2, 64, 1}}};
  auto graph = co::BuildSpatialStep(mapped->spec, tables, owners, budget.Payment(),
                                    {.links = mapped->links, .reference_buffers = false});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  EXPECT_EQ(graph->reference_bytes(), 0U);  // no billion-operation scalar payload oracle
  EXPECT_EQ(graph->outputs()[0].shape, (co::SpatialShape{32, 32, 4, 1}));
  ASSERT_EQ(graph->weights().size(), 128U);
  const auto count_op = [&](ir::Op op) {
    return std::ranges::count_if(graph->graph().nodes(),
                                 [&](const auto& node) { return node.key == ir::Primitive(op); });
  };
  EXPECT_EQ(count_op(ir::Op::kConv2d), 46);
  EXPECT_EQ(count_op(ir::Op::kL2Norm), 36);
  EXPECT_EQ(count_op(ir::Op::kSoftMax), 1);
  auto registry = Registry();
  ASSERT_TRUE(registry);
  en::GraphBuildIO io{graph->graph(), graph->graph().inputs(), graph->graph().outputs(),
                      graph->weights()};
  kn::StepBackend backend{.registry = &*registry,
                          .device = {.arch = 121},
                          .written = kg::WrittenBytes,
                          .within = kg::WithinRoot};
  for (const bool measuring : {true, false}) {
    kn::Addresses addresses;
    addresses.weight = [](const ir::WeightRef& ref, fm::PartRole) {
      return 0x100000000ULL + (std::uint64_t{ref.component} * 2 + ref.role) * (64ULL << 20U);
    };
    addresses.weight_alignment = [](const ir::WeightRef&, fm::PartRole) { return 256; };
    addresses.activations = measuring ? 0ULL : 0x10000000ULL;
    addresses.activation_bytes = 256ULL << 20U;
    addresses.activation_alignment = 256;
    auto plan = en::PlanComponentGraph(
        io, backend, {}, addresses,
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = kg::WrittenBytes,
         .within = kg::WithinRoot,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
    ASSERT_TRUE(plan) << Failed(plan).value_or("");
    const kn::LowerInput input{*plan, io, addresses};
    auto absent = kg::LowerStep(input, *registry, {});
    ASSERT_FALSE(absent);
    EXPECT_NE(absent.error().detail.find("no floating product selector"), std::string::npos);
    // This descriptor-only control models the existing cuBLAS product path;
    // actual GPU programs obtain their selector from the owned backend.
    kg::DeviceChoices device;
    device.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
      return kg::MulMatPath::kCublas;
    };
    auto lowered = kg::LowerStep(input, *registry, device);
    ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
    ASSERT_EQ(lowered->outputs.size(), 1U);
    EXPECT_EQ(lowered->outputs[0].segment, 2U);
    EXPECT_EQ(lowered->outputs[0].bytes, 32U * 32U * 4U * sizeof(float));
    std::size_t norms = 0, convolutions = 0;
    for (auto& step : lowered->plan.steps) {
      convolutions += step.implementation == "ggml.conv2d.f32";
      if (step.implementation != "ggml.l2_norm") continue;
      ++norms;
      auto* norm = step.nodes[0];
      ASSERT_TRUE(kg::CheckL2Norm(norm));
      float original = 0;
      std::memcpy(&original, norm->op_params, sizeof(original));
      const float underflow = 1e-30F;
      std::memcpy(norm->op_params, &underflow, sizeof(underflow));
      EXPECT_FALSE(kg::CheckL2Norm(norm));
      std::memcpy(norm->op_params, &original, sizeof(original));
    }
    EXPECT_EQ(norms, 36U);
    EXPECT_EQ(convolutions, 46U);
  }
  auto malformed = steps;
  malformed[2].in = mo::kVaeQ;  // uninitialized source at the first residual norm
  EXPECT_FALSE(sp::VaeDecoderSpec(malformed, source));
}

TEST(GgmlSpatialTest, FullSourceConditionedBlockUsesSharedProductsMaskAndThreeNormalRotations) {
  const auto& profile = mo::QwenImage21().denoiser;
  const auto source = mo::DenoiserTensors(profile);
  auto mapped = sp::ImageTransformerBlockSpec(profile, 0, 2, source);
  ASSERT_TRUE(mapped) << Failed(mapped).value_or("");
  std::vector<co::BoundTensor> tensors;
  for (const auto& component : mapped->spec.stages[0].layers[0].components)
    for (const auto& binding : component.weights) {
      const auto found = std::ranges::find(source, binding.tensor, &mo::QwenImageTensor::name);
      ASSERT_NE(found, source.end());
      fm::Shape shape;
      shape.rank = static_cast<std::uint8_t>(found->shape.size());
      for (std::size_t d = 0; d < found->shape.size(); ++d)
        shape.ne[d] = found->shape[found->shape.size() - d - 1];
      tensors.push_back({binding.tensor, ir::Dense(fm::Encoding::kBF16, shape)});
    }
  ASSERT_EQ(tensors.size(), 13U);
  const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
  const std::array owners{co::SpatialOwner{2, {1, 6, 4096, 1}}};
  const std::array ports{co::SpatialInput{2, 0, {1, 2, 256, 1}},
                         co::SpatialInput{2, 1, {6, 3, 1, 1}, fm::Encoding::kI32},
                         co::SpatialInput{2, 2, {6, 6, 1, 1}}};
  Budget budget;
  auto graph = co::BuildSpatialStep(mapped->spec, tables, owners, budget.Payment(),
                                    {.links = mapped->links,
                                     .reference_buffers = false,
                                     .exports = mapped->exports,
                                     .inputs = ports});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  auto registry = Registry();
  ASSERT_TRUE(registry);
  en::GraphBuildIO io{graph->graph(), graph->graph().inputs(), graph->graph().outputs(),
                      graph->weights()};
  kn::StepBackend backend{.registry = &*registry,
                          .device = {.arch = 121},
                          .written = kg::WrittenBytes,
                          .within = kg::WithinRoot};
  for (const bool measuring : {true, false}) {
    kn::Addresses addresses;
    addresses.weight = [](const ir::WeightRef& ref, fm::PartRole) {
      return 0x100000000ULL + std::uint64_t{ref.component} * (256ULL << 20U);
    };
    addresses.weight_alignment = [](const ir::WeightRef&, fm::PartRole) { return 256; };
    addresses.activations = measuring ? 0ULL : 0x10000000ULL;
    addresses.activation_bytes = 32ULL << 20U;
    addresses.activation_alignment = 256;
    auto plan = en::PlanComponentGraph(
        io, backend, {}, addresses,
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = kg::WrittenBytes,
         .within = kg::WithinRoot,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
    ASSERT_TRUE(plan) << Failed(plan).value_or("");
    std::vector<std::string> planned_keys;
    std::size_t fused_norms = 0;
    for (const auto& node : plan->graph.nodes()) {
      planned_keys.push_back(ir::OpKeyName(node.key));
      if (node.key != ir::Primitive(node.key.op)) {
        EXPECT_EQ(node.key, ir::Fused(ir::FusedKind::kRmsNormMul));
        ++fused_norms;
      }
    }
    EXPECT_EQ(fused_norms, 2U);
    std::ranges::sort(planned_keys);
    planned_keys.erase(std::unique(planned_keys.begin(), planned_keys.end()), planned_keys.end());
    std::string keys;
    for (const auto& key : planned_keys) keys += key + ",";
    ASSERT_LE(keys.size(), 2048U);
    SCOPED_TRACE(keys);
    RecordProperty(measuring ? "measured_planned_keys" : "bound_planned_keys", keys);
    const kn::LowerInput input{*plan, io, addresses};
    EXPECT_FALSE(input.token);
    kg::DeviceChoices choices;
    choices.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
      return kg::MulMatPath::kCublas;
    };
    auto lowered = kg::LowerStep(input, *registry, choices);
    ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
    ASSERT_EQ(lowered->inputs.size(), 4U);
    ASSERT_EQ(lowered->outputs.size(), 2U);
    EXPECT_EQ(lowered->outputs[0].bytes, 6U * 4096U * 4U);
    EXPECT_EQ(lowered->outputs[1].bytes, 2U * 4096U * 4U);
    std::size_t rotations = 0, products = 0, softmax = 0, norms = 0;
    for (const auto& step : lowered->plan.steps) {
      if (step.implementation == "ggml.norm") {
        ++norms;
        auto* norm = step.nodes[0];
        ASSERT_TRUE(kg::CheckLayerNorm(norm));
        float eps = 0;
        std::memcpy(&eps, norm->op_params, sizeof(eps));
        const float negative = -1.F;
        std::memcpy(norm->op_params, &negative, sizeof(negative));
        EXPECT_FALSE(kg::CheckLayerNorm(norm));
        std::memcpy(norm->op_params, &eps, sizeof(eps));
        const auto stride = norm->src[0]->nb[0];
        norm->src[0]->nb[0] = 8;
        EXPECT_FALSE(kg::CheckLayerNorm(norm));
        norm->src[0]->nb[0] = stride;
        const auto type = norm->src[0]->type;
        norm->src[0]->type = GGML_TYPE_F16;
        EXPECT_FALSE(kg::CheckLayerNorm(norm));
        norm->src[0]->type = type;
        ASSERT_TRUE(kg::CheckLayerNorm(norm));
      }
      rotations += step.implementation == "ggml.rope.ext";
      products += step.implementation == "ggml.mul_mat.cublas";
      softmax += step.implementation == "ggml.soft_max";
    }
    EXPECT_EQ(norms, 2U);
    EXPECT_EQ(rotations, 6U);
    EXPECT_EQ(products, 13U);  // eleven source projections and QK/AV
    EXPECT_EQ(softmax, 1U);
    for (const auto& slot : lowered->inputs) EXPECT_EQ(slot.kind, kn::InputKind::kStepBytes);
  }
}

}  // namespace
