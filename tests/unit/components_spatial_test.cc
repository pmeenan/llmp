// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "components/spatial.h"
#include "engine/graph_plan.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "kernels/reference/exec.h"
#include "spec/image_spatial.h"

namespace llmp::components {
namespace {
namespace ref = kernels::reference;
using test_support::Failed;

spec::ModelSpec SpatialSpec(std::int64_t kernel = 3) {
  spec::Stage stage{.name = "vae", .role = spec::StageRole::kVae, .artifact = 0};
  stage.stream.width = 2;
  spec::ComponentRef up{
      .kind = "upsample.nearest", .version = 1, .params = {{"factor", std::int64_t{2}}}};
  spec::ComponentRef conv{
      .kind = "conv.spatial",
      .version = 1,
      .params = {{"in_channels", std::int64_t{2}},
                 {"out_channels", std::int64_t{3}},
                 {"kernel", kernel},
                 {"padding", kernel == 3 ? std::int64_t{1} : std::int64_t{0}}},
      .weights = {{.role = "kernel", .tensor = "kernel"}, {.role = "bias", .tensor = "bias"}}};
  spec::Layer layer;
  layer.components.push_back(std::move(up));
  layer.components.push_back(std::move(conv));
  stage.layers.push_back(std::move(layer));
  spec::ModelSpec model{.pipeline = spec::PipelineKind::kImage, .stages = {std::move(stage)}};
  spec::Canonicalize(model);
  return model;
}
std::vector<BoundTensor> Tensors(std::uint64_t kernel, format::Encoding encoding) {
  return {{"kernel", ir::Dense(encoding, ir::ShapeOf({kernel, kernel, 2, 3}))},
          {"bias", ir::Dense(encoding, ir::ShapeOf({3}))}};
}
struct Budget {
  std::uint64_t used = 0, limit = 128ULL << 20U, calls = 0;
  SpatialPayment Payment() {
    return {[this](std::uint64_t n) {
              ++calls;
              if (n > limit - used) return false;
              used += n;
              return true;
            },
            [this](std::uint64_t n) {
              EXPECT_LE(n, used);
              used -= n;
            }};
  }
};
constexpr std::array kOwners{SpatialOwner{1, {3, 2, 2, 1}}, SpatialOwner{4, {2, 3, 2, 1}}};

TEST(SpatialComponentsTest, RaggedUpsampleAndChannelChangingConvolutionUseSharedReferenceAndPlan) {
  for (const auto encoding : {format::Encoding::kF32, format::Encoding::kBF16}) {
    Budget budget;
    auto model = SpatialSpec();
    auto tensors = Tensors(3, encoding);
    const std::array<std::span<const BoundTensor>, 1> tables{tensors};
    auto built = BuildSpatialStep(model, tables, kOwners, budget.Payment());
    ASSERT_TRUE(built) << Failed(built).value_or("");
    EXPECT_EQ(budget.used, built->metadata_bytes() + built->reference_bytes());
    ASSERT_EQ(built->weights().size(), 2U);
    ASSERT_EQ(built->graph().outputs().size(), 2U);
    EXPECT_EQ(built->outputs()[0].shape, (SpatialShape{6, 4, 3, 1}));
    EXPECT_EQ(built->outputs()[1].shape, (SpatialShape{4, 6, 3, 1}));
    for (const auto& weight : built->weights())
      EXPECT_EQ(built->graph().tensor(weight.tensor).format.encoding, encoding);
    auto executor = ref::Executor::Create(built->graph());
    ASSERT_TRUE(executor) << Failed(executor, &ref::ExecFailure::detail).value_or("");
    // Exactly representable source values keep the independent double oracle
    // focused on WHCN geometry, padding and owner isolation, not BF16 encoding.
    std::vector<float> kernel(54), bias{0.25F, -0.5F, 1.0F};
    for (std::size_t i = 0; i < kernel.size(); ++i)
      kernel[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 16.0F;
    for (const auto& weight : built->weights()) {
      const auto& values = weight.parts[0].first == "kernel" ? kernel : bias;
      ASSERT_TRUE(executor->BindValues(weight.tensor, values));
    }
    std::array<std::vector<float>, 2> inputs;
    for (std::size_t owner = 0; owner < inputs.size(); ++owner) {
      inputs[owner].resize(12);
      for (std::size_t i = 0; i < 12; ++i)
        inputs[owner][i] = static_cast<float>(i + owner * 16) / 8.0F;
      ASSERT_TRUE(executor->BindValues(built->graph().inputs()[owner], inputs[owner]));
    }
    ASSERT_TRUE(executor->Run());
    auto instances = InstancesOf(model, 0);
    ASSERT_TRUE(instances);
    const auto& up = instances->layers[0][0];
    const auto& conv = instances->layers[0][1];
    for (std::size_t owner = 0; owner < inputs.size(); ++owner) {
      std::vector<double> original(inputs[owner].begin(), inputs[owner].end());
      SpatialRefIO first{up, kOwners[owner].shape, original, {}, {}, {}};
      ASSERT_TRUE(up.def->spatial->reference(first));
      SpatialRefIO second{
          conv, *up.def->spatial->infer(up, first.shape), first.output, kernel, bias, {}};
      ASSERT_TRUE(conv.def->spatial->reference(second));
      auto actual = executor->Values(built->graph().outputs()[owner]);
      ASSERT_TRUE(actual);
      ASSERT_EQ(actual->size(), second.output.size());
      for (std::size_t i = 0; i < actual->size(); ++i)
        EXPECT_DOUBLE_EQ((*actual)[i], second.output[i]);
      EXPECT_EQ(built->graph().tensor(built->graph().outputs()[owner]).segment,
                kOwners[owner].owner);
    }
    std::vector implementations{
        execution::Implementation{
            .name = "test.conv",
            .operation = execution::Operation::kConv2d,
            .source = "reference_control",
            .revision = "v1",
            .build = "host",
            .variant = "planning_only",
            .capability = execution::Capability{.op = ir::Primitive(ir::Op::kConv2d)}},
        execution::Implementation{
            .name = "test.upsample",
            .operation = execution::Operation::kUpsample,
            .source = "reference_control",
            .revision = "v1",
            .build = "host",
            .variant = "planning_only",
            .capability = execution::Capability{.op = ir::Primitive(ir::Op::kUpsample)}}};
    auto registry = execution::Registry::Create(std::move(implementations));
    ASSERT_TRUE(registry);
    kernels::StepBackend backend{.registry = &*registry};
    engine::GraphBuildIO io{built->graph(), built->graph().inputs(), built->graph().outputs(),
                            built->weights()};
    auto plan = engine::PlanComponentGraph(io, backend, {}, {}, {});
    ASSERT_TRUE(plan) << Failed(plan).value_or("");
    EXPECT_EQ(plan->steps.size(), 4U);
    for (const auto id : io.outputs) {
      ASSERT_TRUE(plan->tensors[id.v].valid());
      EXPECT_EQ(plan->graph.tensor(plan->tensors[id.v]).format, built->graph().tensor(id).format);
      EXPECT_EQ(plan->graph.tensor(plan->tensors[id.v]).segment, built->graph().tensor(id).segment);
    }
    auto bad_io = io;
    bad_io.outputs = io.outputs.first(1);
    EXPECT_FALSE(engine::PlanComponentGraph(bad_io, backend, {}, {}));
    auto bad_weights = std::vector<StepWeight>(built->weights().begin(), built->weights().end());
    bad_io.outputs = io.outputs;
    bad_io.weights = bad_weights;
    bad_weights[0].tensor = {};
    EXPECT_FALSE(engine::PlanComponentGraph(bad_io, backend, {}, {}));
    bad_weights[0] = built->weights()[0];
    ++bad_weights[0].ref.role;
    EXPECT_FALSE(engine::PlanComponentGraph(bad_io, backend, {}, {}));
  }
}

TEST(SpatialComponentsTest, FundingAndMalformedGeometryRefuseWithoutPublishedGraph) {
  auto model = SpatialSpec();
  auto tensors = Tensors(3, format::Encoding::kF32);
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  Budget budget;
  budget.limit = 0;
  EXPECT_FALSE(BuildSpatialStep(model, tables, kOwners, budget.Payment()));
  EXPECT_EQ(budget.used, 0U);
  EXPECT_EQ(budget.calls, 1U);
  budget.limit = 128ULL << 20U;
  {
    auto built = BuildSpatialStep(model, tables, kOwners, budget.Payment());
    ASSERT_TRUE(built);
    budget.limit = built->metadata_bytes();
  }
  EXPECT_EQ(budget.used, 0U);
  EXPECT_FALSE(BuildSpatialStep(model, tables, kOwners, budget.Payment()));
  EXPECT_EQ(budget.used, 0U);  // second payment refused, first metadata charge refunded
  budget.limit = 128ULL << 20U;
  auto owners = kOwners;
  owners[1].owner = owners[0].owner;
  const auto calls = budget.calls;
  EXPECT_FALSE(BuildSpatialStep(model, tables, owners, budget.Payment()));
  owners = kOwners;
  owners[0].shape.width = UINT64_MAX;
  EXPECT_FALSE(BuildSpatialStep(model, tables, owners, budget.Payment()));
  EXPECT_EQ(budget.calls, calls);  // malformed borrowed geometry before any charge
  auto wrong = tensors;
  wrong[0].format = ir::Dense(format::Encoding::kF32, ir::ShapeOf({3, 3, 1, 3}));
  const std::array<std::span<const BoundTensor>, 1> bad_tables{wrong};
  EXPECT_FALSE(BuildSpatialStep(model, bad_tables, kOwners, budget.Payment()));
  EXPECT_EQ(budget.used, 0U);
  for (auto& p : model.stages[0].layers[0].components[1].params)
    if (p.name == "in_channels") p.value = std::int64_t{1};
  EXPECT_FALSE(BuildSpatialStep(model, tables, kOwners, budget.Payment()));
  EXPECT_EQ(budget.used, 0U);
}

TEST(SpatialComponentsTest, OneByOneAndSourceOwnedGeometryStayDistinctWithoutTokenPhase) {
  Budget budget;
  auto model = SpatialSpec(1);
  auto tensors = Tensors(1, format::Encoding::kBF16);
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  auto built = BuildSpatialStep(model, tables, kOwners, budget.Payment());
  ASSERT_TRUE(built) << Failed(built).value_or("");
  auto rotated = kOwners;
  std::swap(rotated[0].shape.width, rotated[0].shape.height);
  auto other = BuildSpatialStep(model, tables, rotated, budget.Payment());
  ASSERT_TRUE(other);
  EXPECT_FALSE(std::ranges::equal(built->parameters(), other->parameters()));
  EXPECT_NE(built->graph().Digest(), other->graph().Digest());
  const auto paid = budget.used;
  {
    auto moved = std::move(*built);
    EXPECT_EQ(budget.used, paid);
    EXPECT_EQ(moved.outputs()[0].shape.channels, 3U);
  }
  EXPECT_EQ(budget.used, other->metadata_bytes() + other->reference_bytes());
}

TEST(SpatialComponentsTest, BackwardResidualEdgesPreserveOwnersGammaEncodingAndRefusal) {
  auto model = SpatialSpec();
  auto& components = model.stages[0].layers[0].components;
  components.clear();
  components.push_back({.kind = "norm.channel",
                        .version = 1,
                        .params = {{"channels", std::int64_t{2}},
                                   {"weight_rank", std::int64_t{3}},
                                   {"weight_axis", std::int64_t{2}}},
                        .weights = {{.role = "weight", .tensor = "gamma"}}});
  components.push_back({.kind = "residual.spatial", .version = 1});
  spec::Canonicalize(model);
  const std::array tensors{
      BoundTensor{"gamma", ir::Dense(format::Encoding::kBF16, ir::ShapeOf({1, 1, 2}))}};
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  const std::array links{SpatialLink{0, {}}, SpatialLink{1, 0}};
  Budget budget;
  auto graph = BuildSpatialStep(model, tables, kOwners, budget.Payment(), {.links = links});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  ASSERT_EQ(graph->weights().size(), 1U);
  EXPECT_EQ(graph->graph().tensor(graph->weights()[0].tensor).format, tensors[0].format);
  auto exec = ref::Executor::Create(graph->graph());
  ASSERT_TRUE(exec);
  const std::array<float, 2> gamma{0.5F, 1.5F};
  ASSERT_TRUE(exec->BindValues(graph->weights()[0].tensor, gamma));
  std::array<std::vector<float>, 2> inputs;
  for (std::size_t owner = 0; owner < inputs.size(); ++owner) {
    inputs[owner].resize(12);
    for (std::size_t at = 0; at < 12; ++at)
      inputs[owner][at] = static_cast<float>(1 + at + owner * 11) / 16;
    ASSERT_TRUE(exec->BindValues(graph->graph().inputs()[owner], inputs[owner]));
  }
  ASSERT_TRUE(exec->Run());
  auto instances = InstancesOf(model, 0);
  ASSERT_TRUE(instances);
  for (std::size_t owner = 0; owner < inputs.size(); ++owner) {
    std::vector<double> input(inputs[owner].begin(), inputs[owner].end());
    SpatialRefIO norm{instances->layers[0][0], kOwners[owner].shape, input, gamma, {}, {}};
    ASSERT_TRUE(norm.instance.def->spatial->reference(norm));
    auto actual = exec->Values(graph->graph().outputs()[owner]);
    ASSERT_TRUE(actual);
    for (std::size_t at = 0; at < actual->size(); ++at)
      EXPECT_NEAR((*actual)[at], norm.output[at] + input[at], 2e-6);
  }
  auto wrong = links;
  wrong[1].auxiliary = 2;  // self/forward edge, refused before payment or graph work
  const auto before = budget.calls;
  EXPECT_FALSE(BuildSpatialStep(model, tables, kOwners, budget.Payment(), {.links = wrong}));
  EXPECT_EQ(before, budget.calls);
  wrong = links;
  wrong[0].auxiliary = 0;  // unary norm does not consume a residual
  const auto paid = budget.used;
  EXPECT_FALSE(BuildSpatialStep(model, tables, kOwners, budget.Payment(), {.links = wrong}));
  EXPECT_EQ(paid, budget.used);
  auto no_reference = BuildSpatialStep(model, tables, kOwners, budget.Payment(),
                                       {.links = links, .reference_buffers = false});
  ASSERT_TRUE(no_reference);
  EXPECT_GT(no_reference->metadata_bytes(), 0U);
  EXPECT_EQ(no_reference->reference_bytes(), 0U);
  EXPECT_EQ(no_reference->graph().Digest(), graph->graph().Digest());
}

TEST(SpatialComponentsTest, CompleteReducedDecoderSharesAllFragmentsAndIndependentReferences) {
  auto profile = model::QwenImage21().vae;
  profile.z_dim = 2;
  profile.base = 1;
  profile.dim_mult = {1, 2, 2, 2, 2};
  const auto steps = model::VaeDecoderPlan(profile, 2, 2);
  const auto source = model::VaeDecoderTensors(profile);
  auto mapped = spec::VaeDecoderSpec(steps, source);
  ASSERT_TRUE(mapped) << Failed(mapped).value_or("");
  std::vector<BoundTensor> tensors;
  for (const auto& weight : source) {
    format::Shape shape;
    shape.rank = static_cast<std::uint8_t>(weight.shape.size());
    for (std::size_t d = 0; d < weight.shape.size(); ++d)
      shape.ne[d] = weight.shape[weight.shape.size() - d - 1];
    tensors.push_back({weight.name, ir::Dense(format::Encoding::kF32, shape)});
  }
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  const std::array owners{SpatialOwner{3, {2, 2, 2, 1}}};
  Budget budget;
  auto graph =
      BuildSpatialStep(mapped->spec, tables, owners, budget.Payment(), {.links = mapped->links});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  EXPECT_EQ(graph->outputs()[0].shape, (SpatialShape{32, 32, 4, 1}));
  auto exec = ref::Executor::Create(graph->graph());
  ASSERT_TRUE(exec) << Failed(exec, &ref::ExecFailure::detail).value_or("");
  std::vector<std::vector<float>> weights(graph->weights().size());
  for (std::size_t i = 0; i < weights.size(); ++i) {
    const auto& weight = graph->weights()[i];
    std::uint64_t count = 1;
    for (auto n : graph->graph().tensor(weight.tensor).shape().ne) count *= n;
    weights[i].resize(static_cast<std::size_t>(count));
    for (std::size_t at = 0; at < count; ++at) weights[i][at] = static_cast<float>(1 + at % 5) / 64;
    ASSERT_TRUE(exec->BindValues(weight.tensor, weights[i]));
  }
  const std::array<float, 8> input{.1F, -.2F, .3F, .4F, -.1F, .5F, .2F, -.3F};
  ASSERT_TRUE(exec->BindValues(graph->graph().inputs()[0], input));
  ASSERT_TRUE(exec->Run()) << "all complete decoder primitive decompositions must execute";
  auto instances = InstancesOf(mapped->spec, 0);
  ASSERT_TRUE(instances);
  std::vector<SpatialShape> shapes{owners[0].shape};
  std::vector<std::vector<double>> values{{input.begin(), input.end()}};
  for (std::size_t i = 0; i < mapped->links.size(); ++i) {
    const auto& instance = instances->layers[0][i];
    const auto link = mapped->links[i];
    std::array<std::span<const float>, 2> roles{};
    for (std::size_t j = 0; j < graph->weights().size(); ++j) {
      const auto& weight = graph->weights()[j];
      if (weight.ref.component == instance.index) roles.at(weight.ref.role) = weights[j];
    }
    SpatialRefIO oracle{instance, shapes[link.input], values[link.input], roles[0], roles[1], {}};
    if (link.auxiliary) {
      oracle.auxiliary = values[*link.auxiliary];
      oracle.auxiliary_shape = shapes[*link.auxiliary];
    }
    ASSERT_TRUE(instance.def->spatial->reference(oracle)) << i;
    auto shape = instance.def->spatial->infer(instance, oracle.shape, oracle.auxiliary_shape);
    ASSERT_TRUE(shape);
    shapes.push_back(*shape);
    values.push_back(std::move(oracle.output));
  }
  auto actual = exec->Values(graph->graph().outputs()[0]);
  ASSERT_TRUE(actual);
  ASSERT_EQ(actual->size(), values.back().size());
  for (std::size_t i = 0; i < actual->size(); ++i)
    EXPECT_NEAR((*actual)[i], values.back()[i], 2e-5);
}

std::vector<BoundTensor> TimestepTensors(std::span<const model::QwenImageTensor> source) {
  std::vector<BoundTensor> out;
  for (const auto& tensor : source) {
    if (tensor.name != "time_text_embed.timestep_embedder.linear_1.weight" &&
        tensor.name != "time_text_embed.timestep_embedder.linear_2.weight" &&
        tensor.name != "modulation.1.weight" && tensor.name != "norm_out.linear.weight")
      continue;
    out.push_back({tensor.name, ir::Dense(format::Encoding::kBF16,
                                          ir::ShapeOf({tensor.shape[1], tensor.shape[0]}))});
  }
  return out;
}
// Deliberately scalar, independent WHCN matrix contraction; no builder or
// component emitter is used to compute the expected branch values.
std::vector<double> ProjectRows(std::span<const double> input, std::uint64_t pixels,
                                std::uint64_t ci, std::uint64_t co, std::span<const float> weight,
                                bool silu) {
  std::vector<double> out(static_cast<std::size_t>(pixels * co));
  for (std::uint64_t row = 0; row < pixels; ++row)
    for (std::uint64_t channel = 0; channel < co; ++channel) {
      double value = 0;
      for (std::uint64_t k = 0; k < ci; ++k)
        value += input[row + pixels * k] * weight[k + ci * channel];
      out[row + pixels * channel] = silu ? value / (1 + std::exp(-value)) : value;
    }
  return out;
}

TEST(SpatialComponentsTest, TimestepBranchesKeepBothOutputsAndChangedMultiownerInputs) {
  auto profile = model::QwenImage21().denoiser;
  profile.width = 4;
  profile.timestep_dim = 2;
  const auto source = model::DenoiserTensors(profile);
  auto mapping = spec::ImageTimestepSpec(profile, source);
  ASSERT_TRUE(mapping) << Failed(mapping).value_or("");
  auto tensors = TimestepTensors(source);
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  const std::array owners{SpatialOwner{1, {1, 2, 2, 1}}, SpatialOwner{4, {3, 1, 2, 1}}};
  Budget budget;
  {
    auto graph = BuildSpatialStep(mapping->spec, tables, owners, budget.Payment(),
                                  {.links = mapping->links, .exports = mapping->exports});
    ASSERT_TRUE(graph) << Failed(graph).value_or("");
    ASSERT_EQ(graph->graph().outputs().size(), 4U);
    ASSERT_EQ(graph->outputs().size(), 4U);
    ASSERT_EQ(graph->weights().size(), 4U);
    auto exec = ref::Executor::Create(graph->graph());
    ASSERT_TRUE(exec) << Failed(exec, &ref::ExecFailure::detail).value_or("");
    std::array<std::vector<float>, 4> weights;
    const std::array<std::uint64_t, 4> ci{2, 4, 4, 4}, co{4, 4, 16, 4};
    for (std::size_t i = 0; i < weights.size(); ++i) {
      weights[i].resize(static_cast<std::size_t>(ci[i] * co[i]));
      for (std::size_t j = 0; j < weights[i].size(); ++j)
        weights[i][j] = static_cast<float>(static_cast<int>((j * 3 + i * 7) % 17) - 8) / 32;
      const auto& weight = graph->weights()[i];
      EXPECT_EQ(graph->graph().tensor(weight.tensor).format.encoding, format::Encoding::kBF16);
      ASSERT_TRUE(exec->BindValues(weight.tensor, weights[i]));
    }
    std::array<std::vector<float>, 2> original_outputs;
    for (std::size_t revision = 0; revision < 2; ++revision) {
      std::array<std::vector<float>, 2> inputs;
      for (std::size_t owner = 0; owner < owners.size(); ++owner) {
        inputs[owner].resize(
            static_cast<std::size_t>(owners[owner].shape.width * owners[owner].shape.height * 2));
        for (std::size_t j = 0; j < inputs[owner].size(); ++j)
          inputs[owner][j] = static_cast<float>(j + owner * 3 + revision * 5) / 8;
        ASSERT_TRUE(exec->BindValues(graph->graph().inputs()[owner], inputs[owner]));
      }
      ASSERT_TRUE(exec->Run());
      for (std::size_t owner = 0; owner < owners.size(); ++owner) {
        const auto pixels = owners[owner].shape.width * owners[owner].shape.height;
        std::vector<double> input(inputs[owner].begin(), inputs[owner].end());
        auto first = ProjectRows(input, pixels, 2, 4, weights[0], true);
        auto second = ProjectRows(first, pixels, 4, 4, weights[1], true);
        const std::array expected{ProjectRows(second, pixels, 4, 16, weights[2], false),
                                  ProjectRows(second, pixels, 4, 4, weights[3], false)};
        for (std::size_t branch = 0; branch < 2; ++branch) {
          const auto port = owner * 2 + branch;
          EXPECT_EQ(graph->outputs()[port].owner, owners[owner].owner);
          EXPECT_EQ(graph->outputs()[port].shape.channels, branch == 0 ? 16U : 4U);
          const auto output = graph->graph().outputs()[port];
          EXPECT_EQ(graph->graph().tensor(output).segment, owners[owner].owner);
          auto actual = exec->Values(output);
          ASSERT_TRUE(actual);
          ASSERT_EQ(actual->size(), expected[branch].size());
          for (std::size_t j = 0; j < actual->size(); ++j)
            EXPECT_NEAR((*actual)[j], expected[branch][j], 1e-6);
          if (owner == 0) {
            if (revision == 0)
              original_outputs[branch] = *actual;
            else
              EXPECT_NE(*actual, original_outputs[branch]);
          }
        }
      }
    }
    const std::array<std::uint16_t, 2> reversed{4, 3};
    auto other = BuildSpatialStep(mapping->spec, tables, owners, budget.Payment(),
                                  {.links = mapping->links, .exports = reversed});
    ASSERT_TRUE(other);
    EXPECT_FALSE(std::ranges::equal(graph->parameters(), other->parameters()));
    EXPECT_EQ(other->outputs()[0].shape.channels, 4U);
    auto legacy = BuildSpatialStep(mapping->spec, tables, owners, budget.Payment(),
                                   {.links = mapping->links});
    ASSERT_TRUE(legacy);
    EXPECT_EQ(legacy->graph().outputs().size(), owners.size());
    EXPECT_EQ(legacy->outputs()[0].shape.channels, 4U);
  }
  EXPECT_EQ(budget.used, 0U);
}

TEST(SpatialComponentsTest, FullTimestepSourceMetadataDoesNotAllocateDecodedWeights) {
  const auto& profile = model::QwenImage21().denoiser;
  const auto source = model::DenoiserTensors(profile);
  auto mapping = spec::ImageTimestepSpec(profile, source);
  ASSERT_TRUE(mapping);
  auto tensors = TimestepTensors(source);
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  const std::array owners{SpatialOwner{0, {1, 2, 256, 1}}};
  Budget budget;
  {
    auto graph = BuildSpatialStep(
        mapping->spec, tables, owners, budget.Payment(),
        {.links = mapping->links, .reference_buffers = false, .exports = mapping->exports});
    ASSERT_TRUE(graph) << Failed(graph).value_or("");
    EXPECT_EQ(graph->reference_bytes(), 0U);
    EXPECT_EQ(budget.used, graph->metadata_bytes());
    EXPECT_LT(budget.used, 1U << 20U);
    std::uint64_t weight_bytes = 0;
    for (const auto& weight : graph->weights()) {
      const auto& desc = graph->graph().tensor(weight.tensor);
      EXPECT_EQ(desc.format.encoding, format::Encoding::kBF16);
      weight_bytes += 2 * desc.ne(0) * desc.ne(1);
    }
    EXPECT_EQ(weight_bytes, 203423744U);
    ASSERT_EQ(graph->outputs().size(), 2U);
    EXPECT_EQ(graph->outputs()[0].shape, (SpatialShape{1, 2, 16384, 1}));
    EXPECT_EQ(graph->outputs()[1].shape, (SpatialShape{1, 2, 4096, 1}));
  }
  EXPECT_EQ(budget.used, 0U);
  // A reference request must pay the full source-derived aggregate, not reuse
  // the metadata-only authorization. The 128MiB test allowance cannot pay it.
  auto refused = BuildSpatialStep(mapping->spec, tables, owners, budget.Payment(),
                                  {.links = mapping->links, .exports = mapping->exports});
  ASSERT_FALSE(refused);
  EXPECT_NE(refused.error().find("buffers were not funded"), std::string::npos);
  EXPECT_EQ(budget.used, 0U);
}

TEST(SpatialComponentsTest, TimestepRefusesMalformedSourcesExportsAndPaymentBeforeGraphCopies) {
  auto profile = model::QwenImage21().denoiser;
  profile.width = 4;
  profile.timestep_dim = 2;
  auto source = model::DenoiserTensors(profile);
  auto mapping = spec::ImageTimestepSpec(profile, source);
  ASSERT_TRUE(mapping);
  auto wrong = source;
  wrong[0].name = source[2].name;
  EXPECT_FALSE(spec::ImageTimestepSpec(profile, wrong));
  wrong = source;
  wrong[2].dtype = "F32";
  EXPECT_FALSE(spec::ImageTimestepSpec(profile, wrong));
  wrong = source;
  wrong[2].shape[1] = 3;
  EXPECT_FALSE(spec::ImageTimestepSpec(profile, wrong));
  wrong.erase(wrong.begin() + 2);
  EXPECT_FALSE(spec::ImageTimestepSpec(profile, wrong));
  auto tensors = TimestepTensors(source);
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  const std::array owners{SpatialOwner{0, {1, 2, 2, 1}}};
  for (const auto exports : {std::array<std::uint16_t, 2>{0, 3}, {3, 3}, {3, 5}}) {
    Budget budget;
    EXPECT_FALSE(BuildSpatialStep(mapping->spec, tables, owners, budget.Payment(),
                                  {.links = mapping->links, .exports = exports}));
    EXPECT_EQ(budget.calls, 0U);
    EXPECT_EQ(budget.used, 0U);
  }
  Budget denied;
  denied.limit = 0;
  EXPECT_FALSE(BuildSpatialStep(mapping->spec, tables, owners, denied.Payment(),
                                {.links = mapping->links, .exports = mapping->exports}));
  EXPECT_EQ(denied.calls, 1U);
  EXPECT_EQ(denied.used, 0U);
  Budget malformed;
  tensors[0].format.shape.ne[0] = UINT64_MAX;
  EXPECT_FALSE(BuildSpatialStep(
      mapping->spec, tables, owners, malformed.Payment(),
      {.links = mapping->links, .reference_buffers = false, .exports = mapping->exports}));
  EXPECT_EQ(malformed.used, 0U);
  const std::array huge{SpatialOwner{0, {UINT64_MAX, 2, 2, 1}}};
  EXPECT_FALSE(BuildSpatialStep(mapping->spec, tables, huge, malformed.Payment(),
                                {.links = mapping->links, .exports = mapping->exports}));
}

}  // namespace
}  // namespace llmp::components
