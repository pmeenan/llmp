// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "components/spatial.h"
#include "engine/graph_plan.h"
#include "expected_error.h"
#include "image_family.h"
#include "kernels/ggml/attention_f32.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/conv2d_tiled.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/plan_step.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
#include "planner/policy.h"
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

// The image's planning policy, as the served pipeline plans it.
llmp::planner::Policy ImagePolicy() {
  auto policy = llmp::planner::ParsePolicy(::llmp::test_support::QwenImagePolicy());
  EXPECT_TRUE(policy) << (policy ? "" : policy.error());
  return policy.value_or(llmp::planner::Policy{});
}
std::expected<ex::Registry, ex::PlanRejection> Registry() {
  std::vector<ex::Implementation> entries;
  for (auto declaration : kg::PlanningImplementations()) {
    entries.push_back(std::move(declaration));
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
  std::uint64_t used = 0, limit = 32ULL << 20U;
  co::SpatialPayment Payment() {
    return {[this](std::uint64_t bytes) {
              if (bytes > limit - used) return false;
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

TEST(GgmlSpatialTest, MaskedAttentionPackingKeepsRealRowsAndRefusesInvalidDescriptors) {
  auto packed = [](std::array<std::int64_t, 4> dims, std::uintptr_t address) {
    ggml_tensor t{};
    t.type = GGML_TYPE_F32;
    t.data = reinterpret_cast<void*>(address);
    std::size_t stride = 4;
    for (std::size_t i = 0; i < dims.size(); ++i) {
      t.ne[i] = dims[i];
      t.nb[i] = stride;
      stride *= static_cast<std::size_t>(dims[i]);
    }
    return t;
  };
  for (const auto rows : {39LL, 1024LL, 4121LL}) {
    auto q = packed({128, rows, 32, 1}, 0x100000000ULL);
    auto k = packed({128, rows, 32, 1}, 0x200000000ULL);
    auto v = packed({rows, 128, 32, 1}, 0x300000000ULL);
    std::swap(v.ne[0], v.ne[1]);
    std::swap(v.nb[0], v.nb[1]);
    auto mask = packed({rows, rows, 1, 1}, 0x400000000ULL);
    auto out = packed({128, 32, rows, 1}, 0x500000000ULL);
    out.op = GGML_OP_FLASH_ATTN_EXT;
    out.src[0] = &q;
    out.src[1] = &k;
    out.src[2] = &v;
    out.src[3] = &mask;
    auto plan = kg::CheckAttentionF32(&out);
    ASSERT_TRUE(plan) << Failed(plan, &kg::KernelFailure::detail).value_or("");
    EXPECT_EQ(plan->cells, (static_cast<std::uint64_t>(rows) + 255) / 256 * 256);
    EXPECT_EQ(plan->rows, (static_cast<std::uint64_t>(rows) + 63) / 64 * 64);
    EXPECT_EQ(plan->packing_bytes, 2 * plan->kv_bytes + plan->mask_bytes);
    EXPECT_EQ(mask.ne[0], rows);
    EXPECT_EQ(mask.ne[1], rows);
    out.src[3] = nullptr;
    EXPECT_FALSE(kg::CheckAttentionF32(&out));
    out.src[3] = &mask;
    mask.ne[1]--;
    EXPECT_FALSE(kg::CheckAttentionF32(&out));
    mask.ne[1]++;
    mask.data = out.data;
    EXPECT_FALSE(kg::CheckAttentionF32(&out));
    mask.data = reinterpret_cast<void*>(0x400000000ULL);
    k.ne[2] = v.ne[2] = 16;
    EXPECT_FALSE(kg::CheckAttentionF32(&out));
  }
  const auto* cap = kg::CapabilityOf(kg::kAttentionF32);
  ASSERT_NE(cap, nullptr);
  EXPECT_TRUE(cap->needs_tuning);
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
        io, backend, ImagePolicy(), addresses,
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
  const auto& profile = ::llmp::test_support::QwenImageProfile().vae;
  const auto steps = mo::VaeDecoderPlan(profile, 64, 64);
  const auto tensors = mo::VaeDecoderTensors(profile, ::llmp::test_support::QwenImageNames().vae);
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
  for (const std::uint32_t latent : {2U, 64U}) {
    const auto& profile = ::llmp::test_support::QwenImageProfile().vae;
    const auto steps = mo::VaeDecoderPlan(profile, latent, latent);
    const auto source = mo::VaeDecoderTensors(profile, ::llmp::test_support::QwenImageNames().vae);
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
    const std::array owners{co::SpatialOwner{2, {latent, latent, 64, 1}}};
    auto graph = co::BuildSpatialStep(mapped->spec, tables, owners, budget.Payment(),
                                      {.links = mapped->links, .reference_buffers = false});
    ASSERT_TRUE(graph) << Failed(graph).value_or("");
    EXPECT_EQ(graph->reference_bytes(), 0U);  // no billion-operation scalar payload oracle
    EXPECT_EQ(graph->outputs()[0].shape, (co::SpatialShape{16 * latent, 16 * latent, 4, 1}));
    ASSERT_EQ(graph->weights().size(), 128U);
    const auto count_op = [&](ir::Op op) {
      return std::ranges::count_if(graph->graph().nodes(),
                                   [&](const auto& node) { return node.key == ir::Primitive(op); });
    };
    EXPECT_EQ(count_op(ir::Op::kConv2d), 46);  // the six 1x1 convolutions too
    EXPECT_EQ(count_op(ir::Op::kMatMul), 0);
    EXPECT_EQ(count_op(ir::Op::kL2Norm), 36);
    EXPECT_EQ(count_op(ir::Op::kAttention), 1);
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
      addresses.activations = measuring ? 0ULL : 0x10000000000ULL;
      addresses.activation_bytes = 8ULL << 30U;
      addresses.activation_alignment = 256;
      auto plan = en::PlanComponentGraph(
          io, backend, ImagePolicy(), addresses,
          {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
           .written = kg::WrittenBytes,
           .within = kg::WithinRoot,
           .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
      ASSERT_TRUE(plan) << Failed(plan).value_or("");
      const kn::LowerInput input{*plan, io, addresses};
      auto absent = kg::LowerStep(input, *registry, {});
      ASSERT_FALSE(absent);
      EXPECT_NE(absent.error().detail.find("selected float product group lacks device routing"),
                std::string::npos);
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
      EXPECT_EQ(lowered->outputs[0].bytes, 16ULL * latent * 16ULL * latent * 4ULL * sizeof(float));
      // Every channel norm is the planner's spatial_channel_norm pattern
      // (one launch a norm, no layout round trip); its check refuses an
      // underflowing or non-finite eps, as the primitive L2 norm's does.
      std::size_t norms = 0, primitive_norms = 0, convolutions = 0;
      for (auto& step : lowered->plan.steps) {
        convolutions +=
            step.implementation == "ggml.conv2d.f32" || step.implementation == kg::kConv2dTiledF32;
        primitive_norms += step.implementation == "ggml.l2_norm";
        if (step.implementation != kg::kSpatialChannelNormName) continue;
        ++norms;
        auto* norm = step.nodes[0];
        ASSERT_TRUE(kg::CheckSpatialChannelNorm(norm));
        // The eps sits after GGML's custom parameters (llmp_ops.cc kEpsOffset).
        auto* eps = reinterpret_cast<char*>(norm->op_params) + 32;
        float original = 0;
        std::memcpy(&original, eps, sizeof(original));
        EXPECT_EQ(original, kg::LlmpOpFloat(norm, 0));
        for (const float bad : {1e-30F, std::numeric_limits<float>::infinity()}) {
          std::memcpy(eps, &bad, sizeof(bad));
          EXPECT_FALSE(kg::CheckSpatialChannelNorm(norm)) << bad;
        }
        std::memcpy(eps, &original, sizeof(original));
      }
      EXPECT_EQ(norms, 36U);
      EXPECT_EQ(primitive_norms, 0U);
      EXPECT_EQ(convolutions, 40U);
    }
    auto malformed = steps;
    malformed[2].in = mo::kVaeQ;  // uninitialized source at the first residual norm
    EXPECT_FALSE(sp::VaeDecoderSpec(malformed, source));
  }
}

TEST(GgmlSpatialTest, FullSourceConditionedBlockUsesSharedProductsMaskAndThreeNormalRotations) {
  const auto& profile = ::llmp::test_support::QwenImageProfile().denoiser;
  const auto source = mo::DenoiserTensors(profile, ::llmp::test_support::QwenImageNames().denoiser);
  auto mapped = sp::ImageTransformerBlockSpec(profile, 0, 2, source);
  ASSERT_TRUE(mapped) << Failed(mapped).value_or("");
  std::vector<co::BoundTensor> tensors;
  for (const auto& component : mapped->spec.stages[0].layers[0].components)
    for (const auto& binding : component.weights) {
      const auto found = std::ranges::find(source, binding.tensor, &mo::ImageTensor::name);
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
        io, backend, ImagePolicy(), addresses,
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = kg::WrittenBytes,
         .within = kg::WithinRoot,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
    ASSERT_TRUE(plan) << Failed(plan).value_or("");
    std::vector<std::string> planned_keys;
    // The QK norms are the planner's rms_norm_mul_rope_sections (with
    // their three rotations); the other RMS norms fuse with their weight.
    std::size_t fused_norms = 0, sectioned = 0;
    for (const auto& node : plan->graph.nodes()) {
      planned_keys.push_back(ir::OpKeyName(node.key));
      if (node.key == ir::Fused(ir::FusedKind::kRmsNormMulRopeSections)) {
        EXPECT_EQ(node.Params<ir::RopeSectionsParams>().sections, 3U);
        ++sectioned;
      } else if (node.key != ir::Primitive(node.key.op)) {
        EXPECT_EQ(node.key, ir::Fused(ir::FusedKind::kRmsNormMul));
        ++fused_norms;
      }
    }
    EXPECT_EQ(sectioned, 2U);
    EXPECT_EQ(fused_norms, 0U);
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
      if (step.implementation == kg::kNormRopeSectionsName) {
        const auto sections = kg::CheckNormRopeSections(step.nodes.front());
        ASSERT_TRUE(sections) << Failed(sections, &kg::KernelFailure::detail).value_or("");
        rotations += sections->sections;
      }
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

// The same block built with layout sinking (ir/layout.h): it plans and
// lowers, with far fewer copies.
TEST(GgmlSpatialTest, LayoutSunkBlockPlansAndLowersWithFewerCopies) {
  const auto& profile = ::llmp::test_support::QwenImageProfile().denoiser;
  const auto source = mo::DenoiserTensors(profile, ::llmp::test_support::QwenImageNames().denoiser);
  auto mapped = sp::ImageTransformerBlockSpec(profile, 0, 2, source);
  ASSERT_TRUE(mapped) << Failed(mapped).value_or("");
  std::vector<co::BoundTensor> tensors;
  for (const auto& component : mapped->spec.stages[0].layers[0].components)
    for (const auto& binding : component.weights) {
      const auto found = std::ranges::find(source, binding.tensor, &mo::ImageTensor::name);
      ASSERT_NE(found, source.end());
      fm::Shape shape;
      shape.rank = static_cast<std::uint8_t>(found->shape.size());
      for (std::size_t d = 0; d < found->shape.size(); ++d)
        shape.ne[d] = found->shape[found->shape.size() - d - 1];
      tensors.push_back({binding.tensor, ir::Dense(fm::Encoding::kBF16, shape)});
    }
  const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
  const std::array owners{co::SpatialOwner{2, {1, 6, 4096, 1}}};
  const std::array ports{co::SpatialInput{2, 0, {1, 2, 256, 1}},
                         co::SpatialInput{2, 1, {6, 3, 1, 1}, fm::Encoding::kI32},
                         co::SpatialInput{2, 2, {6, 6, 1, 1}}};
  Budget budget;
  auto registry = Registry();
  ASSERT_TRUE(registry);
  kn::StepBackend backend{.registry = &*registry,
                          .device = {.arch = 121},
                          .written = kg::WrittenBytes,
                          .within = kg::WithinRoot};
  budget.limit = 256ULL << 20U;
  std::size_t copies[2] = {0, 0};
  for (const bool sunk : {false, true}) {
    auto graph = co::BuildSpatialStep(mapped->spec, tables, owners, budget.Payment(),
                                      {.links = mapped->links,
                                       .reference_buffers = false,
                                       .exports = mapped->exports,
                                       .inputs = ports,
                                       .sink_layouts = sunk});
    ASSERT_TRUE(graph) << Failed(graph).value_or("");
    en::GraphBuildIO io{graph->graph(), graph->graph().inputs(), graph->graph().outputs(),
                        graph->weights()};
    const auto policy = ImagePolicy();
    kn::Addresses addresses;
    addresses.weight = [](const ir::WeightRef& ref, fm::PartRole) {
      return 0x100000000ULL + std::uint64_t{ref.component} * (256ULL << 20U);
    };
    addresses.weight_alignment = [](const ir::WeightRef&, fm::PartRole) { return 256; };
    addresses.activations = 0x10000000ULL;
    addresses.activation_bytes = 32ULL << 20U;
    addresses.activation_alignment = 256;
    auto plan = en::PlanComponentGraph(
        io, backend, policy, addresses,
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = kg::WrittenBytes,
         .within = kg::WithinRoot,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
    ASSERT_TRUE(plan) << Failed(plan).value_or("");
    const kn::LowerInput input{*plan, io, addresses};
    kg::DeviceChoices choices;
    choices.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
      return kg::MulMatPath::kCublas;
    };
    auto lowered = kg::LowerStep(input, *registry, choices);
    ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
    std::string names;
    for (const auto& step : lowered->plan.steps) {
      copies[sunk] += step.implementation == "ggml.cont";
      names += std::string(step.implementation) + " ";
    }
    std::printf("%s: %zu steps, %zu copies\n%s\n", sunk ? "sunk" : "original",
                lowered->plan.steps.size(), copies[sunk], names.c_str());
  }
  EXPECT_LE(copies[1] * 3, copies[0] * 2);
}

// A whole small flow (one block of the source's head geometry) built with
// layout sinking: it plans and lowers.
TEST(GgmlSpatialTest, LayoutSunkFlowPlansAndLowers) {
  auto profile = ::llmp::test_support::QwenImageProfile().denoiser;
  profile.blocks = 1;
  profile.heads = 32;
  profile.width = profile.heads * profile.head_dim;
  profile.mlp = 1024;
  profile.context = profile.width;
  constexpr std::uint32_t kText = 8, kGrid = 16, kRows = kText + kGrid * kGrid;
  const auto source = mo::DenoiserTensors(profile, ::llmp::test_support::QwenImageNames().denoiser);
  auto mapped = sp::ImageFlowSpec(profile, kText, kGrid * kGrid, source);
  ASSERT_TRUE(mapped) << Failed(mapped).value_or("");
  std::vector<co::BoundTensor> tensors;
  for (const auto& tensor : source) {
    fm::Shape shape;
    shape.rank = static_cast<std::uint8_t>(tensor.shape.size());
    for (std::size_t d = 0; d < tensor.shape.size(); ++d)
      shape.ne[d] = tensor.shape[tensor.shape.size() - 1 - d];
    tensors.push_back({tensor.name, ir::Dense(fm::Encoding::kBF16, shape)});
  }
  const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
  const std::array owners{co::SpatialOwner{0, {1, kText, profile.context, 1}}};
  const std::array ports{co::SpatialInput{0, 0, {1, kGrid * kGrid, profile.in_channels, 1}},
                         co::SpatialInput{0, 1, {1, 2, profile.timestep_dim, 1}},
                         co::SpatialInput{0, 2, {kRows, 3, 1, 1}, fm::Encoding::kI32},
                         co::SpatialInput{0, 3, {kRows, kRows, 1, 1}},
                         co::SpatialInput{0, 4, {1, 1, 1, 1}}};
  Budget budget;
  budget.limit = 512ULL << 20U;
  auto graph = co::BuildSpatialStep(mapped->spec, tables, owners, budget.Payment(),
                                    {.links = mapped->links,
                                     .reference_buffers = false,
                                     .exports = mapped->exports,
                                     .inputs = ports,
                                     .sink_layouts = true});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  auto registry = Registry();
  ASSERT_TRUE(registry);
  en::GraphBuildIO io{graph->graph(), graph->graph().inputs(), graph->graph().outputs(),
                      graph->weights()};
  kn::StepBackend backend{.registry = &*registry,
                          .device = {.arch = 121},
                          .written = kg::WrittenBytes,
                          .within = kg::WithinRoot};
  const auto policy = ImagePolicy();
  kn::Addresses addresses;
  addresses.weight = [](const ir::WeightRef& ref, fm::PartRole) {
    return 0x100000000ULL + std::uint64_t{ref.component} * (256ULL << 20U);
  };
  addresses.weight_alignment = [](const ir::WeightRef&, fm::PartRole) { return 256; };
  addresses.activations = 0x10000000ULL;
  addresses.activation_bytes = 512ULL << 20U;
  addresses.activation_alignment = 256;
  auto plan =
      en::PlanComponentGraph(io, backend, policy, addresses,
                             {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
                              .written = kg::WrittenBytes,
                              .within = kg::WithinRoot,
                              .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
  ASSERT_TRUE(plan) << Failed(plan).value_or("");
  const kn::LowerInput input{*plan, io, addresses};
  kg::DeviceChoices choices;
  choices.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
    return kg::MulMatPath::kCublas;
  };
  auto lowered = kg::LowerStep(input, *registry, choices);
  if (!lowered) {
    for (std::uint32_t i = 0; i < plan->graph.nodes().size(); ++i) {
      const auto& n = plan->graph.nodes()[i];
      std::string line = std::format("{} {}", i, ir::OpKeyName(n.key));
      for (const auto src : n.sources())
        if (src.valid()) line += std::format(" t{}", src.v);
      for (const auto o : n.outputs()) line += std::format(" -> t{}", o.v);
      std::printf("%s\n", line.c_str());
    }
  }
  ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
}

// The image's planning policy (spec/image_spatial.h ::llmp::test_support::QwenImagePolicy()) over
// two flow blocks of the source's head geometry: every block's attention,
// MLP, modulations (two a block and the output's) and residuals fused by
// the plan, each residual joined with the modulation of its sum (the last
// with the output's, over its latent rows).
TEST(GgmlSpatialTest, PlannedFlowFusesEveryBlock) {
  auto profile = ::llmp::test_support::QwenImageProfile().denoiser;
  profile.blocks = 2;
  profile.heads = 32;
  profile.width = profile.heads * profile.head_dim;
  profile.mlp = 1024;
  profile.context = profile.width;
  constexpr std::uint32_t kText = 25, kGrid = 8, kRows = kText + kGrid * kGrid;
  const auto source = mo::DenoiserTensors(profile, ::llmp::test_support::QwenImageNames().denoiser);
  auto mapped = sp::ImageFlowSpec(profile, kText, kGrid * kGrid, source);
  ASSERT_TRUE(mapped) << Failed(mapped).value_or("");
  std::vector<co::BoundTensor> tensors;
  for (const auto& tensor : source) {
    fm::Shape shape;
    shape.rank = static_cast<std::uint8_t>(tensor.shape.size());
    for (std::size_t d = 0; d < tensor.shape.size(); ++d)
      shape.ne[d] = tensor.shape[tensor.shape.size() - 1 - d];
    tensors.push_back({tensor.name, ir::Dense(fm::Encoding::kBF16, shape)});
  }
  const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
  const std::array owners{co::SpatialOwner{0, {1, kText, profile.context, 1}}};
  const std::array ports{co::SpatialInput{0, 0, {1, kGrid * kGrid, profile.in_channels, 1}},
                         co::SpatialInput{0, 1, {1, 2, profile.timestep_dim, 1}},
                         co::SpatialInput{0, 2, {kRows, 3, 1, 1}, fm::Encoding::kI32},
                         co::SpatialInput{0, 3, {kRows, kRows, 1, 1}},
                         co::SpatialInput{0, 4, {1, 1, 1, 1}}};
  Budget budget;
  budget.limit = 512ULL << 20U;
  auto graph = co::BuildSpatialStep(mapped->spec, tables, owners, budget.Payment(),
                                    {.links = mapped->links,
                                     .reference_buffers = false,
                                     .exports = mapped->exports,
                                     .inputs = ports,
                                     .sink_layouts = true});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  auto registry = Registry();
  ASSERT_TRUE(registry);
  en::GraphBuildIO io{graph->graph(), graph->graph().inputs(), graph->graph().outputs(),
                      graph->weights()};
  kn::StepBackend backend{.registry = &*registry,
                          .device = {.arch = 121},
                          .written = kg::WrittenBytes,
                          .within = kg::WithinRoot};
  auto policy = llmp::planner::ParsePolicy(::llmp::test_support::QwenImagePolicy());
  ASSERT_TRUE(policy) << policy.error();
  // Synthetic fusion audit: propose only the three arithmetic boundaries
  // whose implementations are asserted below, without source-quality grants.
  policy->numerical_candidates = {llmp::planner::NumericalChange::kBf16SwiGluIntermediates,
                                  llmp::planner::NumericalChange::kBf16ProductInputAndResult,
                                  llmp::planner::NumericalChange::kBf16AttentionIntermediates};
  llmp::planner::Canonicalize(*policy);
  kn::Addresses addresses;
  addresses.weight = [](const ir::WeightRef& ref, fm::PartRole) {
    return 0x100000000ULL + std::uint64_t{ref.component} * (256ULL << 20U);
  };
  addresses.weight_alignment = [](const ir::WeightRef&, fm::PartRole) { return 256; };
  addresses.activations = 0x10000000ULL;
  addresses.activation_bytes = 512ULL << 20U;
  addresses.activation_alignment = 256;
  auto plan =
      en::PlanComponentGraph(io, backend, *policy, addresses,
                             {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
                              .written = kg::WrittenBytes,
                              .within = kg::WithinRoot,
                              .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
  ASSERT_TRUE(plan) << Failed(plan).value_or("");
  std::map<std::string, std::size_t> applied;
  for (const auto& a : plan->applied) ++applied[std::string(a.pattern)];
  std::string seen;
  for (const auto& [name, count] : applied) seen += std::format(" {}={}", name, count);
  std::printf("planned flow:%s\n", seen.c_str());
  for (const auto& [name, count] : std::initializer_list<std::pair<std::string, std::size_t>>{
           {"qk_attention", 2},
           {"bf16_branch", 4},
           {"bf16_glu", 2},
           {"split_modulate", 5},
           {"split_gated_residual", 4},
           {"split_gated_residual_modulate", 4}})
    EXPECT_EQ(applied[name], count) << name;
  if (::testing::Test::HasFailure() || std::getenv("LLMP_PLAN_DUMP") != nullptr) {
    const ir::Graph& g = graph->graph();
    for (std::uint32_t i = 0; i < g.nodes().size(); ++i) {
      const auto& n = g.nodes()[i];
      std::string line = std::format("{} {}", i, ir::OpKeyName(n.key));
      for (const auto src : n.sources())
        if (src.valid()) {
          const auto& t = g.tensor(src);
          line += std::format(" t{}[{} {} {} {}|{}]", src.v, t.ne(0), t.ne(1), t.ne(2), t.ne(3),
                              static_cast<int>(t.format.encoding));
        }
      for (const auto o : n.outputs()) line += std::format(" -> t{}", o.v);
      std::printf("%s\n", line.c_str());
    }
    // The planned steps: each implementation over its nodes.
    const ir::Graph& p = plan->graph;
    for (std::size_t s = 0; s < plan->steps.size(); ++s) {
      const auto& step = plan->steps[s];
      std::string line = std::format("step {} {}:", s, registry->at(step.implementation).name);
      for (const auto id : step.nodes) {
        const auto& n = p.node(id);
        line += std::format(" {}", ir::OpKeyName(n.key));
        for (const auto src : n.sources())
          if (src.valid()) {
            const auto& t = p.tensor(src);
            line += std::format(" t{}[{} {} {} {}|{}]", src.v, t.ne(0), t.ne(1), t.ne(2), t.ne(3),
                                static_cast<int>(t.format.encoding));
          }
        for (const auto o : n.outputs()) line += std::format(" -> t{}", o.v);
        line += ";";
      }
      std::printf("%s\n", line.c_str());
    }
  }
  auto plan_registry = Registry();
  ASSERT_TRUE(plan_registry);
  const kn::LowerInput input{*plan, io, addresses};
  kg::DeviceChoices choices;
  choices.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
    return kg::MulMatPath::kCublas;
  };
  auto lowered = kg::LowerStep(input, *plan_registry, choices);
  ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
}

TEST(GgmlSpatialTest, Whole32BlockFlowPreservesPhysicalPlansAndFinalEulerPorts) {
  const auto& profile = ::llmp::test_support::QwenImageProfile().denoiser;
  const auto source = mo::DenoiserTensors(profile, ::llmp::test_support::QwenImageNames().denoiser);
  auto mapped = sp::ImageFlowSpec(profile, 2, 4, source);
  ASSERT_TRUE(mapped) << Failed(mapped).value_or("");
  std::vector<co::BoundTensor> tensors;
  for (const auto& component : mapped->spec.stages[0].layers[0].components)
    for (const auto& binding : component.weights) {
      const auto found = std::ranges::find(source, binding.tensor, &mo::ImageTensor::name);
      ASSERT_NE(found, source.end());
      fm::Shape shape;
      shape.rank = static_cast<std::uint8_t>(found->shape.size());
      for (std::size_t d = 0; d < found->shape.size(); ++d)
        shape.ne[d] = found->shape[found->shape.size() - d - 1];
      tensors.push_back({binding.tensor, ir::Dense(fm::Encoding::kBF16, shape)});
    }
  ASSERT_EQ(tensors.size(), 297U);
  const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
  const std::array owners{co::SpatialOwner{2, {1, 2, 4096, 1}}};
  const std::array ports{
      co::SpatialInput{2, 0, {1, 4, 64, 1}}, co::SpatialInput{2, 1, {1, 2, 256, 1}},
      co::SpatialInput{2, 2, {6, 3, 1, 1}, fm::Encoding::kI32},
      co::SpatialInput{2, 3, {6, 6, 1, 1}}, co::SpatialInput{2, 4, {1, 1, 1, 1}}};
  Budget budget;
  budget.limit = 128ULL << 20U;
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
        io, backend, ImagePolicy(), addresses,
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = kg::WrittenBytes,
         .within = kg::WithinRoot,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
    ASSERT_TRUE(plan) << Failed(plan).value_or("");
    std::vector<std::string> planned_keys;
    // The QK norms are the planner's rms_norm_mul_rope_sections (with
    // their three rotations); the other RMS norms fuse with their weight.
    std::size_t fused_norms = 0, sectioned = 0;
    for (const auto& node : plan->graph.nodes()) {
      planned_keys.push_back(ir::OpKeyName(node.key));
      if (node.key == ir::Fused(ir::FusedKind::kRmsNormMulRopeSections)) {
        EXPECT_EQ(node.Params<ir::RopeSectionsParams>().sections, 3U);
        ++sectioned;
      } else if (node.key != ir::Primitive(node.key.op)) {
        EXPECT_EQ(node.key, ir::Fused(ir::FusedKind::kRmsNormMul));
        ++fused_norms;
      }
    }
    EXPECT_EQ(sectioned, 64U);
    EXPECT_EQ(fused_norms, 1U);
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
    ASSERT_EQ(lowered->inputs.size(), 6U);
    ASSERT_EQ(lowered->outputs.size(), 2U);
    EXPECT_EQ(lowered->outputs[0].bytes, 4U * 64U * 4U);
    EXPECT_EQ(lowered->outputs[1].bytes, 4U * 64U * 4U);
    std::size_t rotations = 0, products = 0, softmax = 0, norms = 0;
    for (const auto& step : lowered->plan.steps) {
      if (step.implementation == "ggml.norm") {
        ++norms;
        EXPECT_TRUE(kg::CheckLayerNorm(step.nodes[0]));
      }
      rotations += step.implementation == "ggml.rope.ext";
      if (step.implementation == kg::kNormRopeSectionsName) {
        const auto sections = kg::CheckNormRopeSections(step.nodes.front());
        ASSERT_TRUE(sections) << Failed(sections, &kg::KernelFailure::detail).value_or("");
        rotations += sections->sections;
      }
      products += step.implementation == "ggml.mul_mat.cublas";
      softmax += step.implementation == "ggml.soft_max";
    }
    EXPECT_EQ(norms, 65U);
    EXPECT_EQ(rotations, 192U);
    EXPECT_EQ(products, 296U);  // 232 source projections and 64 QK/AV
    EXPECT_EQ(softmax, 32U);
    for (const auto& slot : lowered->inputs) EXPECT_EQ(slot.kind, kn::InputKind::kStepBytes);
  }
}

TEST(GgmlSpatialTest, ActualTextEncoderUsesTypedIdsNeoxAndGqaWithoutTokenMetadata) {
  const auto& profile = ::llmp::test_support::QwenImageProfile().text;
  const auto source = mo::TextEncoderTensors(profile, ::llmp::test_support::QwenImageNames().text);
  auto mapping = sp::ImageTextSpec(profile, 39, 14, source);
  ASSERT_TRUE(mapping) << Failed(mapping).value_or("");
  std::vector<co::BoundTensor> tensors;
  for (const auto& tensor : source) {
    fm::Shape shape;
    shape.rank = static_cast<std::uint8_t>(tensor.shape.size());
    for (std::size_t d = 0; d < tensor.shape.size(); ++d)
      shape.ne[d] = tensor.shape[tensor.shape.size() - 1 - d];
    tensors.push_back({tensor.name, ir::Dense(fm::Encoding::kBF16, shape)});
  }
  const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
  const std::array owners{co::SpatialOwner{0, {1, 39, 1, 1}}};
  const std::array ports{co::SpatialInput{0, 0, {39, 1, 1, 1}, fm::Encoding::kI32},
                         co::SpatialInput{0, 1, {39, 39, 1, 1}}};
  Budget budget;
  budget.limit = 128ULL << 20;
  auto graph = co::BuildSpatialStep(mapping->spec, tables, owners, budget.Payment(),
                                    {.links = mapping->links,
                                     .reference_buffers = false,
                                     .input_encoding = fm::Encoding::kI32,
                                     .exports = mapping->exports,
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
      return 0x100000000ULL + std::uint64_t{ref.component} * (2ULL << 30);
    };
    addresses.weight_alignment = [](const ir::WeightRef&, fm::PartRole) { return 256; };
    addresses.activations = measuring ? 0ULL : 0x10000000ULL;
    addresses.activation_bytes = 128ULL << 20;
    addresses.activation_alignment = 256;
    auto plan = en::PlanComponentGraph(
        io, backend, ImagePolicy(), addresses,
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = kg::WrittenBytes,
         .within = kg::WithinRoot,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
    ASSERT_TRUE(plan) << Failed(plan).value_or("");
    const kn::LowerInput input{*plan, io, addresses};
    kg::DeviceChoices choices;
    choices.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
      return kg::MulMatPath::kCublas;
    };
    auto lowered = kg::LowerStep(input, *registry, choices);
    ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
    ASSERT_EQ(lowered->inputs.size(), 3U);
    ASSERT_EQ(lowered->outputs.size(), 1U);
    EXPECT_EQ(lowered->outputs[0].bytes, 25ULL * 4096 * 4);
    std::size_t rotations = 0, products = 0, bf16_products = 0, softmax = 0, rows = 0;
    for (const auto& step : lowered->plan.steps) {
      // The weights' products from BF16 inputs (bf16_inputs, bf16_glu).
      bf16_products +=
          step.implementation == "llmp.gemm.bf16" || step.implementation == kg::kGemmBf16LtName;
      rotations += step.implementation == "ggml.rope.neox";
      if (step.implementation == kg::kNormRopeSectionsName) {
        // The QK norms and their whole-head NEOX rotations, one section.
        const auto sections = kg::CheckNormRopeSections(step.nodes.front());
        ASSERT_TRUE(sections) << Failed(sections, &kg::KernelFailure::detail).value_or("");
        EXPECT_TRUE(sections->neox);
        rotations += sections->sections;
      }
      products += step.implementation == "ggml.mul_mat.cublas";
      if (step.implementation == "ggml.mul_mat.cublas") {
        // Lowering alone does not plan the physical cuBLAS scratch. Validate
        // every product after placement, including odd-token GQA score/value
        // matrices, using the same host planner as the actual launch path.
        ASSERT_EQ(step.nodes.size(), 1U);
        const auto* product = step.nodes.front();
        const auto compute = product->src[0]->type;
        auto physical = kg::CheckMulMatCublas(product, compute, compute == GGML_TYPE_BF16);
        ASSERT_TRUE(physical) << Failed(physical, &kg::KernelFailure::detail).value_or("");
      }
      softmax += step.implementation == "ggml.soft_max";
      rows += step.implementation == "ggml.get_rows";
    }
    EXPECT_EQ(rotations, 72U);
    EXPECT_EQ(products, 72U);        // the attention's scores and values
    EXPECT_EQ(bf16_products, 252U);  // seven a layer
    EXPECT_EQ(softmax, 36U);
    EXPECT_EQ(rows, 1U);
    for (const auto& slot : lowered->inputs) EXPECT_EQ(slot.kind, kn::InputKind::kStepBytes);
  }
}

TEST(GgmlSpatialTest, Actual1024FlowAndDecoderHaveMeasuredTypedStageEnvelopes) {
  const auto& profile = ::llmp::test_support::QwenImageProfile();
  std::vector<llmp::planner::Selected> tuned_steps;
  llmp::planner::Placement tuned_placement;
  std::vector<std::uint64_t> tuned_stripes;
  for (const int mode : {0, 1, 2, 3, 4}) {
    const bool decoder = mode != 0 && mode != 4;
    const bool tiled = mode == 2 || mode == 3;
    SCOPED_TRACE(decoder ? "decoder1024" : "flow4121");
    const auto source =
        decoder ? mo::VaeDecoderTensors(profile.vae, ::llmp::test_support::QwenImageNames().vae)
                : mo::DenoiserTensors(profile.denoiser,
                                      ::llmp::test_support::QwenImageNames().denoiser);
    auto mapping = decoder ? sp::ImageDecodeSpec(profile.vae, 64, source)
                           : sp::ImageFlowSpec(profile.denoiser, 25, 4096, source);
    ASSERT_TRUE(mapping) << Failed(mapping).value_or("");
    std::vector<co::BoundTensor> tensors;
    for (const auto& tensor : source) {
      fm::Shape shape;
      shape.rank = static_cast<std::uint8_t>(tensor.shape.size());
      for (std::size_t d = 0; d < tensor.shape.size(); ++d)
        shape.ne[d] = tensor.shape[tensor.shape.size() - 1 - d];
      tensors.push_back(
          {tensor.name, ir::Dense(decoder ? fm::Encoding::kF32 : fm::Encoding::kBF16, shape)});
    }
    const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
    const std::array owners{co::SpatialOwner{
        0, decoder ? co::SpatialShape{1, 4096, 64, 1} : co::SpatialShape{1, 25, 4096, 1}}};
    const std::vector ports =
        decoder
            ? std::vector{co::SpatialInput{0, 0, {1, 2, 64, 1}}}
            : std::vector{
                  co::SpatialInput{0, 0, {1, 4096, 64, 1}}, co::SpatialInput{0, 1, {1, 2, 256, 1}},
                  co::SpatialInput{0, 2, {4121, 3, 1, 1}, fm::Encoding::kI32},
                  co::SpatialInput{0, 3, {4121, 4121, 1, 1}}, co::SpatialInput{0, 4, {1, 1, 1, 1}}};
    Budget budget;
    budget.limit = 128ULL << 20;
    auto graph = co::BuildSpatialStep(mapping->spec, tables, owners, budget.Payment(),
                                      {.links = mapping->links,
                                       .reference_buffers = false,
                                       .exports = mapping->exports,
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
    llmp::planner::Policy policy;
    if (mode == 1) {
      backend.device.device_class = "sm_121";
      policy.ranks.push_back({"sm_121", "ggml.conv2d.f32", 0});
    }
    if (mode == 2) {
      backend.device.device_class = "sm_121";
      for (const auto& step : mo::VaeDecoderPlan(profile.vae, 64, 64)) {
        if (step.kind != mo::VaeStep::Kind::kConv3x3) continue;
        ex::TunedShape key{.implementation = kg::kConv2dTiledF32,
                           .rows = std::uint64_t{step.width} * step.height,
                           .k = 9ULL * step.in_channels,
                           .n = step.out_channels,
                           .output_encoding = fm::Encoding::kF32};
        if (std::ranges::none_of(backend.device.tuned, [&](const auto& old) {
              return old.rows == key.rows && old.k == key.k && old.n == key.n;
            }))
          backend.device.tuned.push_back(key);
      }
      policy.ranks.push_back({"sm_121", std::string(kg::kConv2dTiledF32), 0});
    }
    if (mode == 4) {
      backend.device.tuned.push_back({.implementation = kg::kAttentionF32,
                                      .rows = 4121,
                                      .k = 128,
                                      .n = 128,
                                      .output_encoding = fm::Encoding::kF32});
    }
    // Descriptor-only physical envelopes: no activation/weight payload allocation.
    kn::Addresses addresses;
    addresses.weight = [](const ir::WeightRef& ref, fm::PartRole) {
      return 0x10000000000ULL + std::uint64_t{ref.component} * (2ULL << 30);
    };
    addresses.weight_alignment = [](const ir::WeightRef&, fm::PartRole) { return 256; };
    addresses.activation_bytes = 32ULL << 30;
    addresses.activation_alignment = 256;
    auto plan = en::PlanComponentGraph(
        io, backend, policy, addresses,
        {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
         .written = kg::WrittenBytes,
         .within = kg::WithinRoot,
         .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
    ASSERT_TRUE(plan) << Failed(plan).value_or("");
    kg::DeviceChoices choices;
    choices.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
      return kg::MulMatPath::kCublas;
    };
    auto lowered = kg::LowerStep(kn::LowerInput{*plan, io, addresses}, *registry, choices);
    ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
    EXPECT_EQ(lowered->outputs.back().bytes, decoder ? 4ULL * 1024 * 1024 * 4 : 4096ULL * 64 * 4);
    EXPECT_EQ(lowered->inputs.size(), decoder ? 2U : 6U);
    EXPECT_GT(lowered->extent, 0U);
    for (const auto& step : lowered->plan.steps) {
      if (step.implementation != "ggml.mul_mat.cublas") continue;
      ASSERT_EQ(step.nodes.size(), 1U);
      const auto* product = step.nodes.front();
      const auto compute = product->src[0]->type;
      auto physical = kg::CheckMulMatCublas(product, compute, compute == GGML_TYPE_BF16);
      ASSERT_TRUE(physical) << Failed(physical, &kg::KernelFailure::detail).value_or("");
    }
    if (mode == 4) {
      std::uint64_t count = 0, packing = 0, bf16_products = 0;
      for (const auto& step : lowered->plan.steps) {
        if (step.implementation == kg::kAttentionF32) {
          ASSERT_EQ(step.nodes.size(), 1U);
          auto checked = kg::CheckAttentionF32(step.nodes.front());
          ASSERT_TRUE(checked) << Failed(checked, &kg::KernelFailure::detail).value_or("");
          packing = std::max(packing, checked->packing_bytes);
          ++count;
        }
        if (step.implementation == "ggml.mul_mat.cublas" &&
            step.nodes.front()->src[0]->type == GGML_TYPE_BF16) {
          // The existing projection already keeps BF16 operands/F32 output;
          // this experiment does not globally change product precision.
          EXPECT_NE(step.nodes.front()->op_params[0], GGML_PREC_F32);
          ++bf16_products;
        }
      }
      EXPECT_EQ(count, 32U);
      EXPECT_GT(bf16_products, 0U);
      EXPECT_EQ(packing, 107511808U);
      RecordProperty("flow_masked_attention_ops", std::to_string(count));
      RecordProperty("flow_masked_attention_packing_bytes", std::to_string(packing));
      RecordProperty("flow_masked_attention_activation_bytes", std::to_string(lowered->extent));
      RecordProperty("flow_existing_bf16_products", std::to_string(bf16_products));
      continue;
    }
    if (tiled) {
      std::vector<std::uint64_t> physical_stripes;
      std::uint64_t count = 0, launches = 0, peak = 0;
      for (const auto& step : lowered->plan.steps) {
        ASSERT_NE(step.implementation, "ggml.conv2d.f32");
        if (step.implementation != kg::kConv2dTiledF32) {
          launches += step.nodes.size();
          continue;
        }
        auto stripes = kg::CheckConv2dTiledF32(step.nodes.front());
        ASSERT_TRUE(stripes) << Failed(stripes, &kg::KernelFailure::detail).value_or("");
        ++count;
        for (const auto* tensor :
             {step.nodes.front()->src[0], step.nodes.front()->src[1], step.nodes.front()}) {
          for (std::size_t d = 0; d < 4; ++d) {
            physical_stripes.push_back(static_cast<std::uint64_t>(tensor->ne[d]));
            physical_stripes.push_back(tensor->nb[d]);
          }
        }
        for (const auto value : {stripes->rows, stripes->columns_bytes, stripes->product_bytes,
                                 stripes->scratch, stripes->stripes, stripes->launches})
          physical_stripes.push_back(value);
        launches += stripes->launches;
        peak = std::max(peak, stripes->scratch);
      }
      EXPECT_EQ(count, 40U);
      EXPECT_EQ(peak, 64880640U);
      // The 36 spatial channel norms fold their layout copies, L2 norm,
      // scale, gamma and SiLU into one launch each: 179 fewer than 7707.
      // The 40 tiled convolutions add their biases (40 fewer), the six 1x1
      // convolutions are one launch each, not four (18), and the four
      // duplicate-ups one, not six (20): 7450.
      EXPECT_EQ(launches, 7450U);
      EXPECT_EQ(lowered->extent, 3019898880ULL);
      if (mode == 2) {
        tuned_steps = plan->steps;
        tuned_placement = plan->placement;
        tuned_stripes = physical_stripes;
      } else {
        EXPECT_TRUE(backend.device.tuned.empty());
        EXPECT_TRUE(policy.ranks.empty());
        EXPECT_EQ(plan->steps, tuned_steps);
        EXPECT_EQ(plan->placement.offsets, tuned_placement.offsets);
        EXPECT_EQ(plan->placement.extent, tuned_placement.extent);
        EXPECT_EQ(physical_stripes, tuned_stripes);
        RecordProperty("decoder_default_matches_qualified_selection", "true");
        continue;
      }
      RecordProperty("decoder_tiled_convolutions", std::to_string(count));
      RecordProperty("decoder_tiled_launches", std::to_string(launches));
      RecordProperty("decoder_tiled_pool", std::to_string(peak));
      RecordProperty("decoder_tiled_capture_bytes", std::to_string((launches + 4) * 16384));
      RecordProperty("decoder_tiled_activation_bytes", std::to_string(lowered->extent));
      RecordProperty("decoder_tiled_tuning_keys", std::to_string(backend.device.tuned.size()));
      continue;
    }
    RecordProperty(decoder ? "decoder_scratch_bytes" : "flow_scratch_bytes",
                   std::to_string(lowered->extent));
    RecordProperty(decoder ? "decoder_selected_ops" : "flow_selected_ops",
                   std::to_string(lowered->plan.steps.size()));
  }
}

// No allocations at these addresses: exercise the actual physical contract.
struct ConvDescriptor {
  ggml_tensor weight{}, input{}, output{};
  static void Set(ggml_tensor& t, std::array<std::int64_t, 4> dims, std::uintptr_t address) {
    t = {};
    t.type = GGML_TYPE_F32;
    t.data = reinterpret_cast<void*>(address);
    std::size_t stride = sizeof(float);
    for (std::size_t i = 0; i < dims.size(); ++i) {
      t.ne[i] = dims[i];
      t.nb[i] = stride;
      stride *= static_cast<std::size_t>(dims[i]);
    }
  }
  ConvDescriptor(std::int64_t width, std::int64_t height, std::int64_t cin, std::int64_t cout,
                 std::int64_t batch = 1) {
    Set(weight, {3, 3, cin, cout}, 0x100000000ULL);
    Set(input, {width, height, cin, batch}, 0x200000000ULL);
    Set(output, {width, height, cout, batch}, 0x300000000ULL);
    output.op = GGML_OP_CONV_2D;
    output.src[0] = &weight;
    output.src[1] = &input;
    for (std::size_t i = 0; i < 6; ++i) output.op_params[i] = 1;
  }
};
TEST(GgmlSpatialTest, TiledConvolutionDerivesFullTailAndBatchLaunchesFromPaidWorkspace) {
  ConvDescriptor representative(128, 128, 64, 1152);
  auto plan = kg::CheckConv2dTiledF32(&representative.output);
  ASSERT_TRUE(plan) << Failed(plan, &kg::KernelFailure::detail).value_or("");
  EXPECT_EQ(plan->rows, 45U);
  EXPECT_EQ(plan->stripes, 3U);
  EXPECT_EQ(plan->launches, 12U);
  EXPECT_EQ(plan->scratch, 66355200U);
  EXPECT_EQ(128U % plan->rows, 38U);
  ConvDescriptor small(7, 5, 2, 3, 2);
  auto two = kg::CheckConv2dTiledF32(&small.output, 1792);
  ASSERT_TRUE(two) << Failed(two, &kg::KernelFailure::detail).value_or("");
  EXPECT_EQ(two->rows, 2U);
  EXPECT_EQ(two->stripes, 6U);
  EXPECT_EQ(two->launches, 24U);
  EXPECT_LE(two->scratch, 1792U);
  EXPECT_FALSE(kg::CheckConv2dTiledF32(&small.output, 1023));
  // The largest source decoder geometry still has a bounded row tile; no
  // fixed spatial policy cap or full 10.125 GiB columns allocation is needed.
  ConvDescriptor large(1024, 1024, 288, 288);
  auto five = kg::CheckConv2dTiledF32(&large.output);
  ASSERT_TRUE(five) << Failed(five, &kg::KernelFailure::detail).value_or("");
  EXPECT_EQ(five->rows, 5U);
  EXPECT_EQ(five->scratch, 64880640U);
  EXPECT_EQ(five->stripes, 205U);
}
TEST(GgmlSpatialTest, TiledConvolutionRefusesAliasesLayoutsAndUnpaidRowsBeforeExecution) {
  ConvDescriptor descriptor(128, 128, 64, 1152);
  const auto original = descriptor.output;
  descriptor.output.data = descriptor.input.data;
  EXPECT_FALSE(kg::CheckConv2dTiledF32(&descriptor.output));
  descriptor.output = original;
  descriptor.input.nb[2] += 4;
  EXPECT_FALSE(kg::CheckConv2dTiledF32(&descriptor.output));
  descriptor.input.nb[2] -= 4;
  descriptor.output.op_params[3] = 0;
  EXPECT_FALSE(kg::CheckConv2dTiledF32(&descriptor.output));
  descriptor.output = original;
  descriptor.weight.type = GGML_TYPE_BF16;
  EXPECT_FALSE(kg::CheckConv2dTiledF32(&descriptor.output));
  descriptor.weight.type = GGML_TYPE_F32;
  EXPECT_FALSE(kg::CheckConv2dTiledF32(&descriptor.output, 1474559));
  EXPECT_TRUE(kg::CheckConv2dTiledF32(&descriptor.output, 1474560));
  ConvDescriptor too_wide(1 << 20, 1, 1, 1);
  EXPECT_TRUE(kg::CheckConv2dTiledF32(&too_wide.output));
  ConvDescriptor unpaid(1 << 21, 1, 1, 1);
  EXPECT_FALSE(kg::CheckConv2dTiledF32(&unpaid.output));
}
TEST(GgmlSpatialTest, TiledConvolutionPhysicalSelectionFollowsTheChosenImplementation) {
  const auto* capability = kg::CapabilityOf(kg::kConv2dTiledF32);
  ASSERT_NE(capability, nullptr);
  EXPECT_FALSE(capability->needs_tuning);
  ConvDescriptor descriptor(128, 128, 64, 1152);
  EXPECT_TRUE(kg::CheckConv2dTiledF32(&descriptor.output));
}

TEST(GgmlSpatialTest, TiledDefaultQualificationKeepsPhysicalShapeDeviceAndFormatIdentity) {
  auto registry = Registry();
  ASSERT_TRUE(registry);
  const ir::Conv2dParams params{.stride = 1, .padding = 1, .dilation = 1};
  // The negative geometry cases have exactly the same collapsed rows/K/N as
  // qualified64x64. Their real W,H or batch differs and must remain direct.
  for (const auto dims : {std::array<std::int64_t, 3>{64, 64, 1}, {128, 32, 1}, {32, 64, 2}}) {
    ConvDescriptor descriptor(dims[0], dims[1], 64, 1152, dims[2]);
    ggml_tensor bias{};
    ConvDescriptor::Set(bias, {1152, 1, 1, 1}, 0x400000000ULL);
    std::array<ex::OperandFormat, 4> operands{};
    const std::array tensors{&descriptor.weight, &descriptor.input, &bias, &descriptor.output};
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      operands[i].operand = i == 3 ? ex::kOutputOperand : static_cast<std::uint8_t>(i);
      operands[i].encoding = fm::Encoding::kF32;
      operands[i].layout = fm::Layout::kDense;
      operands[i].memory.emplace();
      for (std::size_t d = 0; d < 4; ++d) {
        operands[i].ne[d] = static_cast<std::uint64_t>(tensors[i]->ne[d]);
        operands[i].memory->nb[d] = tensors[i]->nb[d];
      }
    }
    ex::CapabilityQuery query{.op = ir::Primitive(ir::Op::kConv2d),
                              .operands = operands,
                              .rows = 4096,
                              .k = 576,
                              .n = 1152,
                              .arch = 121,
                              .params = std::as_bytes(std::span(&params, 1))};
    for (const auto arch : {86U, 121U}) {
      query.arch = arch;
      if (arch != 121) {
        EXPECT_FALSE(kg::CapabilityOf(kg::kConv2dTiledF32)->accepts(query));
        continue;  // This build need not contain any implementation for arch86.
      }
      const auto candidates = ex::Candidates(*registry, query);
      ASSERT_FALSE(candidates.empty());
      const bool qualified = arch == 121 && dims == std::array<std::int64_t, 3>{64, 64, 1};
      EXPECT_EQ(registry->at(candidates.front()).name,
                qualified ? kg::kConv2dTiledF32 : std::string_view("ggml.conv2d.f32"));
    }
    query.arch = 121;
    operands[1].memory->nb[1] += 4;
    EXPECT_FALSE(kg::CapabilityOf(kg::kConv2dTiledF32)->accepts(query));
    operands[1].memory->nb[1] -= 4;
    operands[0].encoding = fm::Encoding::kBF16;
    EXPECT_FALSE(kg::CapabilityOf(kg::kConv2dTiledF32)->accepts(query));
  }
}

TEST(GgmlSpatialTest, TiledConvolutionUsesRealTuningAndPhysicalPlanInBothBindingModes) {
  auto registry = Registry();
  ASSERT_TRUE(registry) << Failed(registry, &ex::PlanRejection::detail).value_or("");
  Budget budget;
  const auto spec = Spec();
  const std::vector<co::BoundTensor> tensors{
      {"kernel", ir::Dense(fm::Encoding::kF32, ir::ShapeOf({3, 3, 2, 3}))},
      {"bias", ir::Dense(fm::Encoding::kF32, ir::ShapeOf({3}))}};
  const std::array<std::span<const co::BoundTensor>, 1> tables{tensors};
  const std::array owners{co::SpatialOwner{0, {3, 2, 2, 1}}};
  auto graph = co::BuildSpatialStep(spec, tables, owners, budget.Payment());
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  en::GraphBuildIO io{graph->graph(), graph->graph().inputs(), graph->graph().outputs(),
                      graph->weights()};
  kn::StepBackend backend{.registry = &*registry,
                          .device = {.device_class = "sm_121", .arch = 121},
                          .written = kg::WrittenBytes,
                          .within = kg::WithinRoot};
  llmp::planner::Policy policy;
  // Diagnostic forced choice, not a claimed measured rank.
  policy.ranks.push_back({"sm_121", std::string(kg::kConv2dTiledF32), 0});
  for (const bool tuning : {false, true}) {
    backend.device.tuned.clear();
    if (tuning)
      backend.device.tuned.push_back({.implementation = kg::kConv2dTiledF32,
                                      .rows = 24,
                                      .k = 18,
                                      .n = 3,
                                      .output_encoding = fm::Encoding::kF32});
    auto paid = llmp::planner::OwnedDeviceFacts::Bound(backend.device);
    ASSERT_TRUE(paid) << Failed(paid).value_or("");
    auto owned = llmp::planner::OwnedDeviceFacts::Create(backend.device, *paid);
    ASSERT_TRUE(owned) << Failed(owned).value_or("");
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
          io, backend, policy, addresses,
          {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
           .written = kg::WrittenBytes,
           .within = kg::WithinRoot,
           .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
      ASSERT_TRUE(plan) << Failed(plan).value_or("");
      auto lowered = kg::LowerStep(kn::LowerInput{*plan, io, addresses}, *registry, {});
      ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
      const auto count = std::ranges::count(
          lowered->plan.steps, tuning ? kg::kConv2dTiledF32 : std::string_view("ggml.conv2d.f32"),
          &kg::PlanStep::implementation);
      EXPECT_EQ(count, 1);
    }
  }
}

TEST(GgmlSpatialTest, RepresentativeTiledConvolutionFitsCompleteActivationPlacement) {
  auto registry = Registry();
  ASSERT_TRUE(registry);
  ir::Builder b(8);
  const auto weight =
      b.Weight(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({3, 3, 64, 1152})), {0, 0, 0});
  const auto bias = b.Weight(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({1152})), {0, 0, 1});
  b.SetOwner(0);
  const auto x = b.Input(ir::Dense(fm::Encoding::kF32, ir::ShapeOf({128, 128, 64, 1})));
  b.Output(b.Conv2d(weight, x, bias, {.stride = 1, .padding = 1, .dilation = 1}));
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph);
  const std::array<co::StepWeight, 2> weights{
      co::StepWeight{weight, {0, 0, 0}, {{"kernel", fm::PartRole::kData}}},
      co::StepWeight{bias, {0, 0, 1}, {{"bias", fm::PartRole::kData}}}};
  en::GraphBuildIO io{*graph, graph->inputs(), graph->outputs(), weights};
  kn::StepBackend backend{.registry = &*registry,
                          .device = {.device_class = "sm_121", .arch = 121},
                          .written = kg::WrittenBytes,
                          .within = kg::WithinRoot};
  backend.device.tuned.push_back({.implementation = kg::kConv2dTiledF32,
                                  .rows = 16384,
                                  .k = 576,
                                  .n = 1152,
                                  .output_encoding = fm::Encoding::kF32});
  llmp::planner::Policy policy;
  policy.ranks.push_back({"sm_121", std::string(kg::kConv2dTiledF32), 0});
  kn::Addresses addresses;
  addresses.weight = [](const ir::WeightRef& ref, fm::PartRole) {
    return 0x20000000ULL + (std::uint64_t{ref.role} << 24U);
  };
  addresses.weight_alignment = [](const ir::WeightRef&, fm::PartRole) { return 256; };
  addresses.activations = 0x100000000ULL;
  addresses.activation_bytes = 128ULL << 20U;
  addresses.activation_alignment = 256;
  auto plan =
      en::PlanComponentGraph(io, backend, policy, addresses,
                             {.staged = std::vector(io.inputs.begin(), io.inputs.end()),
                              .written = kg::WrittenBytes,
                              .within = kg::WithinRoot,
                              .alignment_staged = std::vector(io.inputs.begin(), io.inputs.end())});
  ASSERT_TRUE(plan) << Failed(plan).value_or("");
  auto lowered = kg::LowerStep(kn::LowerInput{*plan, io, addresses}, *registry, {});
  ASSERT_TRUE(lowered) << Failed(lowered, &kg::KernelFailure::detail).value_or("");
  EXPECT_LE(lowered->extent, addresses.activation_bytes);
  EXPECT_EQ(lowered->outputs[0].bytes, 75497472U);
  RecordProperty("representative_activation_bytes", std::to_string(lowered->extent));
}

}  // namespace
