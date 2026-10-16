// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "components/spatial.h"
#include "engine/graph_plan.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "image_family.h"
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

TEST(SpatialComponentsTest, PointwiseProductsPreserveWHCNBiasBatchAndOwnerIsolation) {
  for (const auto encoding : {format::Encoding::kF32, format::Encoding::kBF16}) {
    auto model = SpatialSpec(1);
    model.stages[0].layers[0].components.erase(model.stages[0].layers[0].components.begin());
    auto tensors = Tensors(1, encoding);
    const std::array<std::span<const BoundTensor>, 1> tables{tensors};
    const std::array owners{SpatialOwner{1, {3, 2, 2, 2}}, SpatialOwner{4, {2, 3, 2, 1}}};
    Budget budget;
    auto graph = BuildSpatialStep(model, tables, owners, budget.Payment());
    ASSERT_TRUE(graph) << Failed(graph).value_or("");
    const auto count = [&](ir::Op op) {
      return std::ranges::count_if(graph->graph().nodes(),
                                   [&](const auto& n) { return n.key == ir::Primitive(op); });
    };
    // A 1x1 convolution in every encoding (its implementations read the
    // pixel-contiguous layout; no layout copies).
    EXPECT_EQ(count(ir::Op::kMatMul), 0);
    EXPECT_EQ(count(ir::Op::kConv2d), 2);
    EXPECT_EQ(count(ir::Op::kCont), 0);
    auto executor = ref::Executor::Create(graph->graph());
    ASSERT_TRUE(executor) << Failed(executor, &ref::ExecFailure::detail).value_or("");
    const std::vector<float> kernel{0.5F, -1.F, 0.25F, 2.F, -0.5F, 0.75F};
    const std::vector<float> bias{0.25F, -0.5F, 1.F};
    for (const auto& weight : graph->weights())
      ASSERT_TRUE(
          executor->BindValues(weight.tensor, weight.parts[0].first == "kernel" ? kernel : bias));
    std::array<std::vector<float>, 2> input;
    for (std::size_t o = 0; o < owners.size(); ++o) {
      input[o].resize(static_cast<std::size_t>(*SpatialElements(owners[o].shape)));
      for (std::size_t j = 0; j < input[o].size(); ++j)
        input[o][j] =
            static_cast<float>(static_cast<int>(j % 17) - 8 + 4 * static_cast<int>(o)) / 8.F;
      ASSERT_TRUE(executor->BindValues(graph->graph().inputs()[o], input[o]));
    }
    ASSERT_TRUE(executor->Run());
    auto instances = InstancesOf(model, 0);
    ASSERT_TRUE(instances);
    for (std::size_t o = 0; o < owners.size(); ++o) {
      const std::vector<double> original(input[o].begin(), input[o].end());
      SpatialRefIO oracle{instances->layers[0][0], owners[o].shape, original, kernel, bias, {}};
      ASSERT_TRUE(instances->layers[0][0].def->spatial->reference(oracle));
      auto actual = executor->Values(graph->graph().outputs()[o]);
      ASSERT_TRUE(actual);
      ASSERT_EQ(actual->size(), oracle.output.size());
      for (std::size_t j = 0; j < actual->size(); ++j)
        EXPECT_DOUBLE_EQ((*actual)[j], oracle.output[j]);
    }
    Budget short_payment;
    short_payment.limit = graph->metadata_bytes() - 1;
    EXPECT_FALSE(BuildSpatialStep(model, tables, owners, short_payment.Payment()));
    EXPECT_EQ(short_payment.used, 0U);
    // Padding changes the spatial footprint; retain the direct convolution.
    for (auto& param : model.stages[0].layers[0].components[0].params)
      if (param.name == "padding") param.value = std::int64_t{1};
    Budget padded_payment;
    auto padded = BuildSpatialStep(model, tables, owners, padded_payment.Payment());
    ASSERT_TRUE(padded) << Failed(padded).value_or("");
    EXPECT_EQ(std::ranges::count_if(
                  padded->graph().nodes(),
                  [](const auto& n) { return n.key == ir::Primitive(ir::Op::kConv2d); }),
              2);
  }
}

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
  auto profile = ::llmp::test_support::QwenImageProfile().vae;
  profile.z_dim = 2;
  profile.base = 1;
  profile.dim_mult = {1, 2, 2, 2, 2};
  const auto steps = model::VaeDecoderPlan(profile, 2, 2);
  const auto source = model::VaeDecoderTensors(profile, ::llmp::test_support::QwenImageNames().vae);
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
                                bool silu = false) {
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
  auto profile = ::llmp::test_support::QwenImageProfile().denoiser;
  profile.width = 4;
  profile.timestep_dim = 2;
  const auto source =
      model::DenoiserTensors(profile, ::llmp::test_support::QwenImageNames().denoiser);
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
  const auto& profile = ::llmp::test_support::QwenImageProfile().denoiser;
  const auto source =
      model::DenoiserTensors(profile, ::llmp::test_support::QwenImageNames().denoiser);
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
  auto profile = ::llmp::test_support::QwenImageProfile().denoiser;
  profile.width = 4;
  profile.timestep_dim = 2;
  auto source = model::DenoiserTensors(profile, ::llmp::test_support::QwenImageNames().denoiser);
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

std::vector<BoundTensor> BlockTensors(std::span<const model::QwenImageTensor> source) {
  auto out = TimestepTensors(source);
  for (const auto& tensor : source) {
    if (!tensor.name.starts_with("transformer_blocks.0.")) continue;
    format::Shape shape;
    shape.rank = static_cast<std::uint8_t>(tensor.shape.size());
    for (std::size_t i = 0; i < tensor.shape.size(); ++i)
      shape.ne[i] = tensor.shape[tensor.shape.size() - i - 1];
    out.push_back({tensor.name, ir::Dense(format::Encoding::kBF16, shape)});
  }
  return out;
}
std::vector<SpatialInput> BlockPorts(std::span<const SpatialOwner> owners, std::uint64_t timestep) {
  std::vector<SpatialInput> out;
  for (const auto& owner : owners) {
    const auto rows = owner.shape.height;
    out.push_back({owner.owner, 0, {1, 2, timestep, 1}});
    out.push_back({owner.owner, 1, {rows, 3, 1, 1}, format::Encoding::kI32});
    out.push_back({owner.owner, 2, {rows, rows, 1, 1}});
  }
  return out;
}
using BlockWeights = std::map<std::string, std::vector<float>>;
// Independent complete block equations over channel-major double vectors.
// Neither GraphLink nor component emit/reference code computes this expectation.
std::array<std::vector<double>, 2> BlockOracle(const model::QwenImageDenoiserProfile& p,
                                               std::uint64_t rows, std::span<const double> stream,
                                               std::span<const double> sinusoid,
                                               std::span<const double> positions,
                                               std::span<const double> mask, const BlockWeights& w,
                                               std::uint32_t block = 0) {
  const auto project = [&](std::span<const double> x, std::uint64_t n, std::uint64_t ci,
                           std::uint64_t co, const std::string& name, bool silu = false) {
    return ProjectRows(x, n, ci, co, w.at(name), silu);
  };
  const auto first = project(sinusoid, 2, p.timestep_dim, p.width,
                             "time_text_embed.timestep_embedder.linear_1.weight", true);
  const auto second = project(first, 2, p.width, p.width,
                              "time_text_embed.timestep_embedder.linear_2.weight", true);
  const auto modulation = project(second, 2, p.width, 4ULL * p.width, "modulation.1.weight");
  const auto final_scale = project(second, 2, p.width, p.width, "norm_out.linear.weight");
  const auto condition = [&](std::uint64_t row, std::uint64_t channel, std::uint64_t quarter) {
    return modulation[(row < 2 ? 1U : 0U) + 2 * (channel + quarter * p.width)];
  };
  const auto adaptive = [&](std::span<const double> input, std::uint64_t quarter) {
    std::vector<double> out(input.size());
    for (std::uint64_t row = 0; row < rows; ++row) {
      double mean = 0, variance = 0;
      for (std::uint64_t c = 0; c < p.width; ++c) mean += input[row + rows * c];
      mean /= p.width;
      for (std::uint64_t c = 0; c < p.width; ++c) {
        const double d = input[row + rows * c] - mean;
        variance += d * d;
      }
      const double scale = 1 / std::sqrt(variance / p.width + p.eps);
      for (std::uint64_t c = 0; c < p.width; ++c)
        out[row + rows * c] =
            (input[row + rows * c] - mean) * scale * (1 + condition(row, c, quarter));
    }
    return out;
  };
  const std::string prefix = "transformer_blocks." + std::to_string(block) + ".";
  const auto normalized = adaptive(stream, 0);
  auto q = project(normalized, rows, p.width, p.width, prefix + "attn.to_q.weight");
  auto k = project(normalized, rows, p.width, p.width, prefix + "attn.to_k.weight");
  const auto v = project(normalized, rows, p.width, p.width, prefix + "attn.to_v.weight");
  const auto rotate = [&](std::vector<double>& x, const std::string& name) {
    for (std::uint64_t row = 0; row < rows; ++row)
      for (std::uint64_t head = 0; head < p.heads; ++head) {
        double square = 0;
        for (std::uint64_t c = 0; c < p.head_dim; ++c) {
          const auto value = x[row + rows * (head * p.head_dim + c)];
          square += value * value;
        }
        const double scale = 1 / std::sqrt(square / p.head_dim + p.eps);
        std::uint64_t offset = 0;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          for (std::uint64_t pair = 0; pair < p.rope_axes[axis]; pair += 2) {
            const auto c0 = offset + pair, c1 = c0 + 1;
            const auto i0 = row + rows * (head * p.head_dim + c0), i1 = i0 + rows;
            const double a = x[i0] * scale * w.at(name)[c0];
            const double b = x[i1] * scale * w.at(name)[c1];
            const double angle =
                positions[row + rows * axis] *
                std::pow(p.rope_theta, -static_cast<double>(pair) / p.rope_axes[axis]);
            x[i0] = a * std::cos(angle) - b * std::sin(angle);
            x[i1] = a * std::sin(angle) + b * std::cos(angle);
          }
          offset += p.rope_axes[axis];
        }
      }
  };
  rotate(q, prefix + "attn.norm_q.weight");
  rotate(k, prefix + "attn.norm_k.weight");
  std::vector<double> attended(stream.size()), scores(rows);
  for (std::uint64_t head = 0; head < p.heads; ++head)
    for (std::uint64_t query = 0; query < rows; ++query) {
      for (std::uint64_t key = 0; key < rows; ++key) {
        double score = 0;
        for (std::uint64_t c = 0; c < p.head_dim; ++c)
          score +=
              q[query + rows * (head * p.head_dim + c)] * k[key + rows * (head * p.head_dim + c)];
        scores[key] = score / std::sqrt(p.head_dim) + mask[key + rows * query];
      }
      const double largest = *std::ranges::max_element(scores);
      double sum = 0;
      for (auto& score : scores) {
        score = std::exp(score - largest);
        sum += score;
      }
      for (std::uint64_t c = 0; c < p.head_dim; ++c) {
        double value = 0;
        for (std::uint64_t key = 0; key < rows; ++key)
          value += scores[key] / sum * v[key + rows * (head * p.head_dim + c)];
        attended[query + rows * (head * p.head_dim + c)] = value;
      }
    }
  auto branch = project(attended, rows, p.width, p.width, prefix + "attn.to_out.0.weight");
  std::vector<double> residual(stream.size());
  for (std::uint64_t c = 0; c < p.width; ++c)
    for (std::uint64_t row = 0; row < rows; ++row) {
      const auto i = row + rows * c;
      residual[i] = stream[i] + branch[i] * std::tanh(condition(row, c, 1));
    }
  const auto mlp = adaptive(residual, 2);
  auto gate = project(mlp, rows, p.width, p.mlp, prefix + "img_mlp.gate_layer.weight");
  const auto up = project(mlp, rows, p.width, p.mlp, prefix + "img_mlp.proj.weight");
  for (std::size_t i = 0; i < gate.size(); ++i)
    gate[i] = gate[i] / (1 + std::exp(-gate[i])) * up[i];
  branch = project(gate, rows, p.mlp, p.width, prefix + "img_mlp.out.weight");
  for (std::uint64_t c = 0; c < p.width; ++c)
    for (std::uint64_t row = 0; row < rows; ++row)
      residual[row + rows * c] += branch[row + rows * c] * std::tanh(condition(row, c, 3));
  return {std::move(residual), final_scale};
}

TEST(SpatialComponentsTest, ConditionedBlockConsumesBothCarriesAndOwnerLocalTypedInputs) {
  auto p = ::llmp::test_support::QwenImageProfile().denoiser;
  p.blocks = 1;
  p.width = 16;
  p.heads = 2;
  p.head_dim = 8;
  p.mlp = 48;
  p.timestep_dim = 4;
  p.rope_axes = {2, 2, 4};
  const auto source = model::DenoiserTensors(p, ::llmp::test_support::QwenImageNames().denoiser);
  auto mapped = spec::ImageTransformerBlockSpec(p, 0, 2, source);
  ASSERT_TRUE(mapped) << Failed(mapped).value_or("");
  const auto tensors = BlockTensors(source);
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  const std::array owners{SpatialOwner{1, {1, 6, 16, 1}}, SpatialOwner{4, {1, 7, 16, 1}}};
  const auto ports = BlockPorts(owners, p.timestep_dim);
  Budget budget;
  auto graph =
      BuildSpatialStep(mapped->spec, tables, owners, budget.Payment(),
                       {.links = mapped->links, .exports = mapped->exports, .inputs = ports});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  ASSERT_EQ(graph->weights().size(), 13U);
  ASSERT_EQ(graph->graph().inputs().size(), 8U);
  ASSERT_EQ(graph->graph().outputs().size(), 4U);
  auto exec = ref::Executor::Create(graph->graph());
  ASSERT_TRUE(exec) << Failed(exec, &ref::ExecFailure::detail).value_or("");
  BlockWeights weights;
  for (const auto& weight : graph->weights()) {
    const auto& format = graph->graph().tensor(weight.tensor).format;
    std::size_t n = 1;
    for (auto extent : format.shape.ne) n *= extent;
    auto& data = weights[weight.parts[0].first];
    data.resize(n);
    for (std::size_t i = 0; i < n; ++i)
      data[i] =
          format.shape.rank == 1
              ? 1.0F + static_cast<float>(i % 3) / 16
              : static_cast<float>(static_cast<int>((i * 13 + weights.size() * 7) % 23) - 11) / 64;
    ASSERT_TRUE(exec->BindValues(weight.tensor, data));
  }
  std::array<std::vector<float>, 2> original;
  for (std::size_t revision = 0; revision < 5; ++revision) {
    std::array<std::array<std::vector<double>, 2>, 2> expected;
    for (std::size_t owner = 0; owner < owners.size(); ++owner) {
      const auto rows = owners[owner].shape.height;
      std::vector<float> stream(rows * p.width), sinusoid(2 * p.timestep_dim), mask(rows * rows);
      std::vector<std::int64_t> positions(3 * rows);
      for (std::size_t i = 0; i < stream.size(); ++i)
        stream[i] = static_cast<float>(static_cast<int>((i * 7 + owner * 3) % 19) - 9) / 8;
      if (revision == 2)
        for (std::uint64_t c = 0; c < p.width; ++c)
          stream[rows - 1 + rows * c] += static_cast<float>(c % 3) / 4;
      for (std::size_t i = 0; i < sinusoid.size(); ++i)
        sinusoid[i] = static_cast<float>(i + owner + (revision == 1 && i % 2 == 0 ? 4U : 0U)) / 8;
      for (std::uint64_t row = 0; row < rows; ++row) {
        positions[row] = row < 2 ? static_cast<std::int64_t>(row) : 2;
        positions[rows + row] =
            row < 2 ? static_cast<std::int64_t>(row) : static_cast<std::int64_t>((row - 2) / 2) - 1;
        positions[2 * rows + row] =
            row < 2 ? static_cast<std::int64_t>(row) : static_cast<std::int64_t>((row - 2) % 2) - 1;
        for (std::uint64_t key = 0; key < rows; ++key)
          mask[key + rows * row] =
              row < 2 && key > row ? -std::numeric_limits<float>::infinity() : 0;
      }
      if (revision == 3) positions[rows + rows - 1] -= 2;
      ASSERT_TRUE(exec->BindValues(graph->graph().inputs()[owner * 4], stream));
      ASSERT_TRUE(exec->BindValues(graph->graph().inputs()[owner * 4 + 1], sinusoid));
      ASSERT_TRUE(exec->BindInts(graph->graph().inputs()[owner * 4 + 2], positions));
      ASSERT_TRUE(exec->BindValues(graph->graph().inputs()[owner * 4 + 3], mask));
      const std::vector<double> x(stream.begin(), stream.end()),
          t(sinusoid.begin(), sinusoid.end()), pos(positions.begin(), positions.end()),
          m(mask.begin(), mask.end());
      expected[owner] = BlockOracle(p, rows, x, t, pos, m, weights);
    }
    ASSERT_TRUE(exec->Run()) << "revision " << revision;
    for (std::size_t owner = 0; owner < owners.size(); ++owner) {
      for (std::size_t branch = 0; branch < 2; ++branch) {
        auto actual = exec->Values(graph->graph().outputs()[owner * 2 + branch]);
        ASSERT_TRUE(actual);
        ASSERT_EQ(actual->size(), expected[owner][branch].size());
        for (std::size_t i = 0; i < actual->size(); ++i)
          EXPECT_NEAR((*actual)[i], expected[owner][branch][i], 2e-5);
        EXPECT_EQ(graph->outputs()[owner * 2 + branch].owner, owners[owner].owner);
        if (branch == 0) {
          if (revision == 0)
            original[owner] = *actual;
          else if (revision == 4)
            EXPECT_EQ(*actual, original[owner]);
          else {
            EXPECT_NE(*actual, original[owner]);
            if (revision == 2 || revision == 3) {
              bool noncausal_key_used = false;
              for (std::uint64_t c = 0; c < p.width; ++c)
                noncausal_key_used |= (*actual)[2 + owners[owner].shape.height * c] !=
                                      original[owner][2 + owners[owner].shape.height * c];
              EXPECT_TRUE(noncausal_key_used);
            }
            for (std::uint64_t c = 0; c < p.width; ++c)
              for (std::uint64_t row = 0; row < 2; ++row)
                EXPECT_EQ((*actual)[row + owners[owner].shape.height * c],
                          original[owner][row + owners[owner].shape.height * c]);
          }
        }
      }
    }
  }
}

TEST(SpatialComponentsTest, SourceAxesUseNormalOffsetsAndIndependentNegativePositionFormula) {
  const auto& p = ::llmp::test_support::QwenImageProfile().denoiser;
  spec::Stage stage{.name = "rotation", .role = spec::StageRole::kDenoiser, .artifact = 0};
  stage.stream.width = 128;
  spec::ComponentRef rotate{.kind = "norm.rope_axes",
                            .version = 1,
                            .params = {{"heads", std::int64_t{1}},
                                       {"axis0", std::int64_t{16}},
                                       {"axis1", std::int64_t{56}},
                                       {"axis2", std::int64_t{56}},
                                       {"base", double{p.rope_theta}},
                                       {"eps", double{p.eps}}},
                            .weights = {{.role = "weight", .tensor = "gamma"}}};
  spec::Layer layer;
  layer.components.push_back(std::move(rotate));
  stage.layers.push_back(std::move(layer));
  spec::ModelSpec model{.pipeline = spec::PipelineKind::kImage, .stages = {std::move(stage)}};
  spec::Canonicalize(model);
  const std::vector<BoundTensor> tensors{
      {"gamma", ir::Dense(format::Encoding::kBF16, ir::ShapeOf({128}))}};
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  const std::array owners{SpatialOwner{0, {1, 6, 128, 1}}};
  const std::array ports{SpatialInput{0, 0, {6, 3, 1, 1}, format::Encoding::kI32}};
  const std::array links{SpatialLink{.auxiliary_port = 0}};
  Budget budget;
  auto graph =
      BuildSpatialStep(model, tables, owners, budget.Payment(), {.links = links, .inputs = ports});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  auto exec = ref::Executor::Create(graph->graph());
  ASSERT_TRUE(exec);
  const std::vector<float> gamma(128, 1.0F);
  std::vector<float> input(768);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<float>(static_cast<int>((i * 17) % 29) - 14) / 16;
  const std::array<std::int64_t, 18> positions{0,  1, 2, 2, 2, 2,  0, 1,  -1,
                                               -1, 0, 0, 0, 1, -1, 0, -1, 0};
  ASSERT_TRUE(exec->BindValues(graph->weights()[0].tensor, gamma));
  ASSERT_TRUE(exec->BindValues(graph->graph().inputs()[0], input));
  ASSERT_TRUE(exec->BindInts(graph->graph().inputs()[1], positions));
  ASSERT_TRUE(exec->Run());
  auto actual = exec->Values(graph->graph().outputs()[0]);
  ASSERT_TRUE(actual);
  const auto source_table = model::QwenImageDenoiserRotary(p, 2, 2);
  for (std::uint64_t row = 0; row < 6; ++row) {
    double square = 0;
    for (std::uint64_t c = 0; c < 128; ++c)
      square += double{input[row + 6 * c]} * input[row + 6 * c];
    const auto scale = 1 / std::sqrt(square / 128 + p.eps);
    std::uint64_t offset = 0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      for (std::uint64_t c = 0; c < p.rope_axes[axis]; c += 2) {
        const auto pair = (offset + c) / 2;
        const double angle = static_cast<double>(positions[row + 6 * axis]) *
                             std::pow(p.rope_theta, -static_cast<double>(c) / p.rope_axes[axis]);
        EXPECT_NEAR(source_table[(row * 64 + pair) * 2], std::cos(angle), 3e-7);
        EXPECT_NEAR(source_table[(row * 64 + pair) * 2 + 1], std::sin(angle), 3e-7);
        const double a = input[row + 6 * (offset + c)] * scale;
        const double b = input[row + 6 * (offset + c + 1)] * scale;
        EXPECT_NEAR((*actual)[row + 6 * (offset + c)], a * std::cos(angle) - b * std::sin(angle),
                    2e-6);
        EXPECT_NEAR((*actual)[row + 6 * (offset + c + 1)],
                    a * std::sin(angle) + b * std::cos(angle), 2e-6);
      }
      offset += p.rope_axes[axis];
    }
  }
  auto instances = InstancesOf(model, 0);
  ASSERT_TRUE(instances);
  std::vector<double> x(input.begin(), input.end()),
      invalid_positions(positions.begin(), positions.end());
  SpatialRefIO oracle{instances->layers[0][0], owners[0].shape, x, gamma, {}, {},
                      invalid_positions,       ports[0].shape};
  invalid_positions[2] = double{INT32_MAX} + 1;
  EXPECT_FALSE(oracle.instance.def->spatial->reference(oracle));
  invalid_positions[2] = 0.5;
  EXPECT_FALSE(oracle.instance.def->spatial->reference(oracle));
}

TEST(SpatialComponentsTest, ConditionedBlockPaysFullSourcesAndRefusesMalformedPortsAndCarries) {
  const auto& p = ::llmp::test_support::QwenImageProfile().denoiser;
  const auto source = model::DenoiserTensors(p, ::llmp::test_support::QwenImageNames().denoiser);
  auto mapped = spec::ImageTransformerBlockSpec(p, 0, 2, source);
  ASSERT_TRUE(mapped) << Failed(mapped).value_or("");
  const auto tensors = BlockTensors(source);
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  const std::array owners{SpatialOwner{0, {1, 6, p.width, 1}}};
  auto ports = BlockPorts(owners, p.timestep_dim);
  Budget budget;
  {
    auto graph = BuildSpatialStep(mapped->spec, tables, owners, budget.Payment(),
                                  {.links = mapped->links,
                                   .reference_buffers = false,
                                   .exports = mapped->exports,
                                   .inputs = ports});
    ASSERT_TRUE(graph) << Failed(graph).value_or("");
    EXPECT_EQ(graph->reference_bytes(), 0U);
    std::uint64_t bytes = 0;
    for (const auto& weight : graph->weights()) {
      auto parts = format::PartsOf(graph->graph().tensor(weight.tensor).format);
      ASSERT_TRUE(parts);
      bytes += (*parts)[0].bytes;
    }
    EXPECT_EQ(bytes, 639631872ULL);
    EXPECT_EQ(graph->outputs()[0].shape, (SpatialShape{1, 6, 4096, 1}));
    EXPECT_EQ(graph->outputs()[1].shape, (SpatialShape{1, 2, 4096, 1}));
  }
  EXPECT_EQ(budget.used, 0U);
  EXPECT_FALSE(
      BuildSpatialStep(mapped->spec, tables, owners, budget.Payment(),
                       {.links = mapped->links, .exports = mapped->exports, .inputs = ports}));
  EXPECT_EQ(budget.used, 0U);
  const auto refused = [&] {
    return !BuildSpatialStep(mapped->spec, tables, owners, budget.Payment(),
                             {.links = mapped->links,
                              .reference_buffers = false,
                              .exports = mapped->exports,
                              .inputs = ports});
  };
  ports[1].encoding = format::Encoding::kF32;
  EXPECT_TRUE(refused());
  ports[1].encoding = format::Encoding::kI32;
  ports[2].shape.width = 5;
  EXPECT_TRUE(refused());
  ports[2].shape.width = 6;
  ports[1].owner = 2;
  EXPECT_TRUE(refused());
  ports[1].owner = 0;
  ports[1].slot = 0;
  EXPECT_TRUE(refused());
  ports[1].slot = 1;
  const auto original_aux = mapped->links[8].auxiliary;
  mapped->links[8].auxiliary.reset();
  EXPECT_TRUE(refused());
  mapped->links[8].auxiliary = original_aux;
  const auto original_carry = mapped->links.back().carry;
  mapped->links.back().carry = static_cast<std::uint16_t>(mapped->links.size());
  EXPECT_TRUE(refused());
  mapped->links.back().carry.reset();
  EXPECT_TRUE(refused());
  mapped->links.back().carry = original_carry;
  budget.limit = 1;
  EXPECT_TRUE(refused());
  EXPECT_EQ(budget.used, 0U);
  auto wrong = source;
  auto role = std::ranges::find(wrong, "transformer_blocks.0.attn.norm_q.weight",
                                &model::QwenImageTensor::name);
  ASSERT_NE(role, wrong.end());
  role->shape[0] = 64;
  EXPECT_FALSE(spec::ImageTransformerBlockSpec(p, 0, 2, wrong));
  EXPECT_FALSE(spec::ImageTransformerBlockSpec(p, p.blocks, 2, source));
  auto invalid = p;
  invalid.rope_axes = {15, 57, 56};
  EXPECT_FALSE(spec::ImageTransformerBlockSpec(invalid, 0, 2, source));
}

std::vector<BoundTensor> FlowTensors(std::span<const model::QwenImageTensor> source) {
  std::vector<BoundTensor> out;
  for (const auto& tensor : source) {
    format::Shape shape;
    shape.rank = static_cast<std::uint8_t>(tensor.shape.size());
    for (std::size_t i = 0; i < tensor.shape.size(); ++i)
      shape.ne[i] = tensor.shape[tensor.shape.size() - i - 1];
    out.push_back({tensor.name, ir::Dense(format::Encoding::kBF16, shape)});
  }
  return out;
}
std::vector<SpatialInput> FlowPorts(std::span<const SpatialOwner> owners,
                                    const model::QwenImageDenoiserProfile& p) {
  std::vector<SpatialInput> out;
  for (const auto& owner : owners) {
    out.push_back({owner.owner, 0, {1, 4, p.in_channels, 1}});
    out.push_back({owner.owner, 1, {1, 2, p.timestep_dim, 1}});
    out.push_back({owner.owner, 2, {6, 3, 1, 1}, format::Encoding::kI32});
    out.push_back({owner.owner, 3, {6, 6, 1, 1}});
    out.push_back({owner.owner, 4, {1, 1, 1, 1}});
  }
  return out;
}
// Independent complete forward and Euler equations. The block oracle evaluates
// its own equations, never component references or graph links.
std::array<std::vector<double>, 2> FlowOracle(const model::QwenImageDenoiserProfile& p,
                                              std::span<const double> text,
                                              std::span<const double> latent,
                                              std::span<const double> sinusoid,
                                              std::span<const double> positions,
                                              std::span<const double> mask, double dt,
                                              const BlockWeights& weights) {
  std::vector<double> normalized(text.begin(), text.end());
  for (std::size_t row = 0; row < 2; ++row) {
    double square = 0;
    for (std::size_t c = 0; c < p.context; ++c) square += text[row + 2 * c] * text[row + 2 * c];
    const double scale = 1 / std::sqrt(square / p.context + p.eps);
    for (std::size_t c = 0; c < p.context; ++c)
      normalized[row + 2 * c] *= scale * (1 + weights.at("txt_in.text_norm.weight")[c]);
  }
  auto text_hidden =
      ProjectRows(normalized, 2, p.context, p.width, weights.at("txt_in.in_layer.weight"));
  for (auto& value : text_hidden)
    value = 0.5 * value *
            (1 + std::tanh(0.7978845608028654 * (value + 0.044715 * value * value * value)));
  const auto text_projected =
      ProjectRows(text_hidden, 2, p.width, p.width, weights.at("txt_in.out_layer.weight"));
  const auto image_projected =
      ProjectRows(latent, 4, p.in_channels, p.width, weights.at("img_in.weight"));
  std::vector<double> stream(6ULL * p.width), final_scale;
  for (std::size_t c = 0; c < p.width; ++c) {
    for (std::size_t row = 0; row < 2; ++row) stream[row + 6 * c] = text_projected[row + 2 * c];
    for (std::size_t row = 0; row < 4; ++row)
      stream[row + 2 + 6 * c] = image_projected[row + 4 * c];
  }
  for (std::uint32_t block = 0; block < p.blocks; ++block) {
    auto next = BlockOracle(p, 6, stream, sinusoid, positions, mask, weights, block);
    stream = std::move(next[0]);
    final_scale = std::move(next[1]);
  }
  normalized.resize(4ULL * p.width);
  for (std::size_t row = 0; row < 4; ++row) {
    double mean = 0, variance = 0;
    for (std::size_t c = 0; c < p.width; ++c) mean += stream[row + 2 + 6 * c];
    mean /= p.width;
    for (std::size_t c = 0; c < p.width; ++c) {
      const double delta = stream[row + 2 + 6 * c] - mean;
      variance += delta * delta;
    }
    const double scale = 1 / std::sqrt(variance / p.width + p.eps);
    for (std::size_t c = 0; c < p.width; ++c)
      normalized[row + 4 * c] = (stream[row + 2 + 6 * c] - mean) * scale * (1 + final_scale[2 * c]);
  }
  auto noise = ProjectRows(normalized, 4, p.width, p.out_channels, weights.at("proj_out.weight"));
  std::vector<double> next(latent.begin(), latent.end());
  for (std::size_t i = 0; i < next.size(); ++i) next[i] += dt * noise[i];
  return {std::move(noise), std::move(next)};
}

TEST(SpatialComponentsTest, CompleteFlowUsesAllBlocksAndIndependentBoundaryMathForTwoOwners) {
  auto p = ::llmp::test_support::QwenImageProfile().denoiser;
  p.blocks = 2;
  p.width = 16;
  p.context = 16;
  p.heads = 2;
  p.head_dim = 8;
  p.mlp = 48;
  p.timestep_dim = 4;
  p.rope_axes = {2, 2, 4};
  p.in_channels = 4;
  p.out_channels = 4;
  const auto source = model::DenoiserTensors(p, ::llmp::test_support::QwenImageNames().denoiser);
  auto mapping = spec::ImageFlowSpec(p, 2, 4, source);
  ASSERT_TRUE(mapping) << Failed(mapping).value_or("");
  ASSERT_EQ(mapping->links.size(), 53U);
  const auto tensors = FlowTensors(source);
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  const std::array owners{SpatialOwner{2, {1, 2, 16, 1}}, SpatialOwner{5, {1, 2, 16, 1}}};
  const auto ports = FlowPorts(owners, p);
  Budget budget;
  auto graph =
      BuildSpatialStep(mapping->spec, tables, owners, budget.Payment(),
                       {.links = mapping->links, .exports = mapping->exports, .inputs = ports});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  ASSERT_EQ(graph->weights().size(), 27U);
  ASSERT_EQ(graph->graph().inputs().size(), 12U);
  ASSERT_EQ(graph->graph().outputs().size(), 4U);
  auto exec = ref::Executor::Create(graph->graph());
  ASSERT_TRUE(exec) << Failed(exec, &ref::ExecFailure::detail).value_or("");
  BlockWeights weights;
  for (const auto& weight : graph->weights()) {
    const auto& format = graph->graph().tensor(weight.tensor).format;
    std::size_t n = 1;
    for (auto extent : format.shape.ne) n *= extent;
    const auto& name = weight.parts[0].first;
    auto& data = weights[name];
    data.resize(n);
    for (std::size_t i = 0; i < n; ++i)
      data[i] =
          format.shape.rank == 1
              ? (name == "txt_in.text_norm.weight" ? -0.25F : 1.0F) + static_cast<float>(i % 3) / 16
              : static_cast<float>(static_cast<int>((i * 13 + weights.size() * 7) % 23) - 11) / 64;
    ASSERT_TRUE(exec->BindValues(weight.tensor, data));
  }
  std::array<std::array<std::vector<float>, 2>, 2> original;
  for (std::size_t revision = 0; revision < 6; ++revision) {
    std::array<std::array<std::vector<double>, 2>, 2> expected;
    std::array<std::vector<float>, 2> latents;
    for (std::size_t owner = 0; owner < owners.size(); ++owner) {
      std::vector<float> text(2 * p.context), latent(4 * p.in_channels), time(2 * p.timestep_dim),
          mask(36);
      std::vector<std::int64_t> positions(18);
      for (std::size_t i = 0; i < text.size(); ++i)
        text[i] = static_cast<float>(static_cast<int>((i * 7 + owner * 3) % 19) - 9) / 8;
      for (std::size_t i = 0; i < latent.size(); ++i)
        latent[i] = static_cast<float>(static_cast<int>((i * 3 + owner * 5) % 17) - 8) / 8;
      if (revision == 2) latent[3] += 0.5F;
      if (revision == 3) text[1] += 0.5F;
      for (std::size_t i = 0; i < time.size(); ++i)
        time[i] = static_cast<float>(i + owner + (revision == 1 && i % 2 == 0 ? 4U : 0U)) / 8;
      for (std::size_t row = 0; row < 6; ++row) {
        positions[row] = row < 2 ? static_cast<std::int64_t>(row) : 2;
        positions[6 + row] =
            row < 2 ? static_cast<std::int64_t>(row) : static_cast<std::int64_t>((row - 2) / 2) - 1;
        positions[12 + row] =
            row < 2 ? static_cast<std::int64_t>(row) : static_cast<std::int64_t>((row - 2) % 2) - 1;
        for (std::size_t key = 0; key < 6; ++key)
          mask[key + 6 * row] = row < 2 && key > row ? -std::numeric_limits<float>::infinity() : 0;
      }
      const std::array dt{revision == 4 ? 0.F : -0.125F};
      const auto inputs = graph->graph().inputs();
      ASSERT_TRUE(exec->BindValues(inputs[owner * 6], text));
      ASSERT_TRUE(exec->BindValues(inputs[owner * 6 + 1], latent));
      ASSERT_TRUE(exec->BindValues(inputs[owner * 6 + 2], time));
      ASSERT_TRUE(exec->BindInts(inputs[owner * 6 + 3], positions));
      ASSERT_TRUE(exec->BindValues(inputs[owner * 6 + 4], mask));
      ASSERT_TRUE(exec->BindValues(inputs[owner * 6 + 5], dt));
      const std::vector<double> tx(text.begin(), text.end()), lx(latent.begin(), latent.end()),
          tm(time.begin(), time.end()), pos(positions.begin(), positions.end()),
          m(mask.begin(), mask.end());
      expected[owner] = FlowOracle(p, tx, lx, tm, pos, m, dt[0], weights);
      latents[owner] = latent;
    }
    ASSERT_TRUE(exec->Run()) << "revision " << revision;
    for (std::size_t owner = 0; owner < owners.size(); ++owner)
      for (std::size_t branch = 0; branch < 2; ++branch) {
        auto actual = exec->Values(graph->graph().outputs()[owner * 2 + branch]);
        ASSERT_TRUE(actual);
        ASSERT_EQ(actual->size(), expected[owner][branch].size());
        for (std::size_t i = 0; i < actual->size(); ++i)
          EXPECT_NEAR((*actual)[i], expected[owner][branch][i], 1e-4);
        if (revision == 0)
          original[owner][branch] = *actual;
        else if (revision == 5 || (revision == 4 && branch == 0))
          EXPECT_EQ(*actual, original[owner][branch]);
        else
          EXPECT_NE(*actual, original[owner][branch]);
        if (revision == 4 && branch == 1) EXPECT_EQ(*actual, latents[owner]);
        EXPECT_EQ(graph->outputs()[owner * 2 + branch].owner, owners[owner].owner);
      }
  }
}

TEST(SpatialComponentsTest, FullFlowPays563ComponentsAndRefusesDecodedReferenceBeforeCopy) {
  const auto& p = ::llmp::test_support::QwenImageProfile().denoiser;
  const auto source = model::DenoiserTensors(p, ::llmp::test_support::QwenImageNames().denoiser);
  ASSERT_EQ(source.size(), 297U);
  auto mapping = spec::ImageFlowSpec(p, 2, 4, source);
  ASSERT_TRUE(mapping) << Failed(mapping).value_or("");
  ASSERT_EQ(mapping->links.size(), 563U);
  const auto tensors = FlowTensors(source);
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  const std::array owners{SpatialOwner{0, {1, 2, 4096, 1}}};
  const auto ports = FlowPorts(owners, p);
  Budget budget;
  auto graph = BuildSpatialStep(mapping->spec, tables, owners, budget.Payment(),
                                {.links = mapping->links,
                                 .reference_buffers = false,
                                 .exports = mapping->exports,
                                 .inputs = ports});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  EXPECT_EQ(graph->weights().size(), 297U);
  EXPECT_EQ(graph->graph().outputs().size(), 2U);
  for (const auto& output : graph->outputs()) EXPECT_EQ(output.shape, (SpatialShape{1, 4, 64, 1}));
  std::uint64_t primitive_count = 0;
  for (const auto& layer : mapping->spec.stages[0].layers) {
    for (const auto& component : layer.components) {
      const auto* def = Find(component.kind, component.version);
      ASSERT_NE(def, nullptr);
      ASSERT_NE(def->spatial, nullptr);
      primitive_count += def->spatial->max_nodes;
    }
  }
  EXPECT_EQ(primitive_count, 32U * 169U + 137U);
  const auto metadata_nodes =
      owners.size() * (primitive_count + 1) + ports.size() + 2 * mapping->links.size();
  const auto metadata_bytes =
      4096ULL * (2 * metadata_nodes + source.size() + 4 * mapping->links.size() + 16 +
                 owners.size() * mapping->exports.size() + 2 * ports.size());
  EXPECT_EQ(metadata_bytes, 65253376U);
  EXPECT_EQ(graph->metadata_bytes(), metadata_bytes);
  RecordProperty("full_flow_primitive_ceiling", std::to_string(primitive_count));
  RecordProperty("full_flow_metadata_bytes", std::to_string(graph->metadata_bytes()));
  const auto paid = budget.used;
  EXPECT_EQ(paid, metadata_bytes);
  EXPECT_LT(paid, budget.limit);
  Budget denied;
  denied.limit = paid - 1;
  EXPECT_FALSE(BuildSpatialStep(mapping->spec, tables, owners, denied.Payment(),
                                {.links = mapping->links,
                                 .reference_buffers = false,
                                 .exports = mapping->exports,
                                 .inputs = ports}));
  EXPECT_EQ(denied.calls, 1U);
  EXPECT_EQ(denied.used, 0U);
  Budget reference;
  EXPECT_FALSE(BuildSpatialStep(mapping->spec, tables, owners, reference.Payment(),
                                {.links = mapping->links,
                                 .reference_buffers = true,
                                 .exports = mapping->exports,
                                 .inputs = ports}));
  EXPECT_EQ(reference.calls, 2U);
  EXPECT_EQ(reference.used, 0U);
  auto invalid = p;
  invalid.blocks = UINT32_MAX;
  EXPECT_FALSE(spec::ImageFlowSpecBound(invalid, source));
  invalid = p;
  invalid.context = 4095;
  EXPECT_FALSE(spec::ImageFlowSpec(invalid, 2, 4, source));
  auto bad_source = source;
  bad_source.push_back(source.front());
  EXPECT_FALSE(spec::ImageFlowSpec(p, 2, 4, bad_source));
  auto too_many = mapping->spec;
  auto& components = too_many.stages[0].layers[0].components;
  components.resize(spec::kMaxComponents + 1ULL);
  Budget invalid_count;
  EXPECT_FALSE(BuildSpatialStep(too_many, tables, owners, invalid_count.Payment()));
  EXPECT_EQ(invalid_count.calls, 0U);
}

TEST(SpatialComponentsTest, TypedTextEncoderUsesIndependentGatherGqaAndNeoxReference) {
  auto p = ::llmp::test_support::QwenImageProfile().text;
  p.layers = 2;
  p.width = 8;
  p.heads = 2;
  p.kv_heads = 1;
  p.head_dim = 4;
  p.ffn = 12;
  p.vocab = 9;
  const auto source = model::TextEncoderTensors(p, ::llmp::test_support::QwenImageNames().text);
  auto mapping = spec::ImageTextSpec(p, 3, 1, source);
  ASSERT_TRUE(mapping) << Failed(mapping).value_or("");
  const auto tensors = FlowTensors(source);
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  const std::array owners{SpatialOwner{2, {1, 3, 1, 1}}};
  const std::array ports{SpatialInput{2, 0, {3, 1, 1, 1}, format::Encoding::kI32},
                         SpatialInput{2, 1, {3, 3, 1, 1}}};
  Budget budget;
  auto graph = BuildSpatialStep(mapping->spec, tables, owners, budget.Payment(),
                                {.links = mapping->links,
                                 .input_encoding = format::Encoding::kI32,
                                 .exports = mapping->exports,
                                 .inputs = ports});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  auto exec = ref::Executor::Create(graph->graph());
  ASSERT_TRUE(exec) << Failed(exec, &ref::ExecFailure::detail).value_or("");
  BlockWeights weights;
  for (const auto& weight : graph->weights()) {
    const auto& desc = graph->graph().tensor(weight.tensor).format;
    std::size_t count = 1;
    for (const auto n : desc.shape.ne) count *= n;
    auto& data = weights[weight.parts[0].first];
    data.resize(count);
    for (std::size_t i = 0; i < count; ++i)
      data[i] = desc.shape.rank == 1 ? 1.F + static_cast<float>(i % 3) / 16
                                     : static_cast<float>(static_cast<int>((i * 7) % 17) - 8) / 64;
    ASSERT_TRUE(exec->BindValues(weight.tensor, data));
  }
  const std::array<std::int64_t, 3> ids{2, 1, 2}, positions{-2, 0, 3};
  std::array<float, 9> mask{};
  for (std::size_t query = 0; query < 3; ++query)
    for (std::size_t key = query + 1; key < 3; ++key)
      mask[key + 3 * query] = -std::numeric_limits<float>::infinity();
  ASSERT_TRUE(exec->BindInts(graph->graph().inputs()[0], ids));
  ASSERT_TRUE(exec->BindInts(graph->graph().inputs()[1], positions));
  ASSERT_TRUE(exec->BindValues(graph->graph().inputs()[2], mask));
  auto executed = exec->Run();
  ASSERT_TRUE(executed) << Failed(executed, &ref::ExecFailure::detail).value_or("");
  auto instances = InstancesOf(mapping->spec, 0);
  ASSERT_TRUE(instances);
  std::vector<std::vector<double>> values{std::vector<double>(ids.begin(), ids.end())};
  std::vector<SpatialShape> shapes{owners[0].shape};
  const std::array<std::vector<double>, 2> extra{
      std::vector<double>(positions.begin(), positions.end()),
      std::vector<double>(mask.begin(), mask.end())};
  const auto& components = mapping->spec.stages[0].layers[0].components;
  ASSERT_EQ(components.size(), 17U * p.layers + 2);
  for (std::size_t i = 0; i < components.size(); ++i) {
    const auto& in = instances->layers[0][i];
    const auto& link = mapping->links[i];
    const auto shape = shapes[link.input];
    std::span<const double> aux;
    std::optional<SpatialShape> aux_shape;
    if (link.auxiliary_port) {
      aux = extra[*link.auxiliary_port];
      aux_shape = ports[*link.auxiliary_port].shape;
    } else if (link.auxiliary) {
      aux = values[*link.auxiliary];
      aux_shape = shapes[*link.auxiliary];
    }
    std::span<const float> weight;
    if (!components[i].weights.empty()) weight = weights.at(components[i].weights[0].tensor);
    SpatialRefIO oracle{in, shape, values[link.input], weight, {}, {}, aux, aux_shape};
    ASSERT_TRUE(in.def->spatial->reference(oracle)) << i << " " << components[i].kind;
    auto output_shape = in.def->spatial->infer(in, shape, aux_shape);
    ASSERT_TRUE(output_shape);
    values.push_back(std::move(oracle.output));
    shapes.push_back(*output_shape);
  }
  auto actual = exec->Values(graph->graph().outputs()[0]);
  ASSERT_TRUE(actual);
  ASSERT_EQ(actual->size(), values.back().size());
  for (std::size_t i = 0; i < actual->size(); ++i)
    EXPECT_NEAR((*actual)[i], values.back()[i], 3e-6);
  EXPECT_EQ(graph->outputs()[0].shape, (SpatialShape{1, 2, 8, 1}));
  const auto& embedding = instances->layers[0][0];
  std::vector<double> bad_ids{-1, 1, 2};
  SpatialRefIO bad{
      embedding, owners[0].shape, bad_ids, weights.at(components[0].weights[0].tensor), {}, {}};
  EXPECT_FALSE(embedding.def->spatial->reference(bad));
  bad_ids[0] = p.vocab;
  EXPECT_FALSE(embedding.def->spatial->reference(bad));
  bad_ids[0] = 0.5;
  EXPECT_FALSE(embedding.def->spatial->reference(bad));
  Budget wrong_type;
  EXPECT_FALSE(
      BuildSpatialStep(mapping->spec, tables, owners, wrong_type.Payment(),
                       {.links = mapping->links, .exports = mapping->exports, .inputs = ports}));
  EXPECT_EQ(wrong_type.used, 0U);
}

TEST(SpatialComponentsTest, RealTextAnd1024FlowMetadataArePaidWithoutReferencePayloads) {
  const auto& profile = ::llmp::test_support::QwenImageProfile();
  const auto text_source =
      model::TextEncoderTensors(profile.text, ::llmp::test_support::QwenImageNames().text);
  auto text = spec::ImageTextSpec(profile.text, 39, 14, text_source);
  ASSERT_TRUE(text) << Failed(text).value_or("");
  const auto text_tensors = FlowTensors(text_source);
  const std::array<std::span<const BoundTensor>, 1> text_tables{text_tensors};
  const std::array text_owners{SpatialOwner{0, {1, 39, 1, 1}}};
  const std::array text_ports{SpatialInput{0, 0, {39, 1, 1, 1}, format::Encoding::kI32},
                              SpatialInput{0, 1, {39, 39, 1, 1}}};
  Budget text_budget;
  auto text_graph = BuildSpatialStep(text->spec, text_tables, text_owners, text_budget.Payment(),
                                     {.links = text->links,
                                      .reference_buffers = false,
                                      .input_encoding = format::Encoding::kI32,
                                      .exports = text->exports,
                                      .inputs = text_ports});
  ASSERT_TRUE(text_graph) << Failed(text_graph).value_or("");
  EXPECT_EQ(text_graph->weights().size(), 397U);
  EXPECT_EQ(text_graph->outputs()[0].shape, (SpatialShape{1, 25, 4096, 1}));
  EXPECT_EQ(text_graph->reference_bytes(), 0U);
  Budget deny_reference;
  EXPECT_FALSE(BuildSpatialStep(text->spec, text_tables, text_owners, deny_reference.Payment(),
                                {.links = text->links,
                                 .input_encoding = format::Encoding::kI32,
                                 .exports = text->exports,
                                 .inputs = text_ports}));
  EXPECT_EQ(deny_reference.used, 0U);
  const auto flow_source =
      model::DenoiserTensors(profile.denoiser, ::llmp::test_support::QwenImageNames().denoiser);
  auto flow = spec::ImageFlowSpec(profile.denoiser, 25, 4096, flow_source);
  ASSERT_TRUE(flow) << Failed(flow).value_or("");
  const auto flow_tensors = FlowTensors(flow_source);
  const std::array<std::span<const BoundTensor>, 1> flow_tables{flow_tensors};
  const std::array flow_owners{SpatialOwner{0, {1, 25, 4096, 1}}};
  const std::array flow_ports{
      SpatialInput{0, 0, {1, 4096, 64, 1}}, SpatialInput{0, 1, {1, 2, 256, 1}},
      SpatialInput{0, 2, {4121, 3, 1, 1}, format::Encoding::kI32},
      SpatialInput{0, 3, {4121, 4121, 1, 1}}, SpatialInput{0, 4, {1, 1, 1, 1}}};
  Budget flow_budget;
  auto flow_graph = BuildSpatialStep(flow->spec, flow_tables, flow_owners, flow_budget.Payment(),
                                     {.links = flow->links,
                                      .reference_buffers = false,
                                      .exports = flow->exports,
                                      .inputs = flow_ports});
  ASSERT_TRUE(flow_graph) << Failed(flow_graph).value_or("");
  EXPECT_EQ(flow_graph->weights().size(), 297U);
  EXPECT_EQ(flow_graph->outputs()[1].shape, (SpatialShape{1, 4096, 64, 1}));
  EXPECT_EQ(flow_graph->metadata_bytes(), 65253376U);
  EXPECT_EQ(flow_graph->reference_bytes(), 0U);
  RecordProperty("text_metadata_bytes", std::to_string(text_graph->metadata_bytes()));
  RecordProperty("real_flow_metadata_bytes", std::to_string(flow_graph->metadata_bytes()));
  EXPECT_FALSE(SpatialElements({UINT64_MAX, 2, 1, 1}));
  EXPECT_FALSE(SpatialElements({1, 1, 1, 0}));
  EXPECT_TRUE(SpatialElements({4121, 4121, 32, 1}));
  auto invalid = profile.text;
  invalid.kv_heads = 3;
  EXPECT_FALSE(spec::ImageTextSpec(invalid, 39, 14, text_source));
  invalid.layers = UINT32_MAX;
  EXPECT_FALSE(spec::ImageTextSpecBound(invalid, text_source));
}

TEST(SpatialComponentsTest, LatentReshapeAndChannelAffinePreserveOwnerPlanes) {
  spec::Stage stage{.name = "decode", .role = spec::StageRole::kVae, .artifact = 0};
  stage.stream.width = 2;
  spec::Layer layer;
  layer.components.push_back({.kind = "reshape.spatial",
                              .version = 1,
                              .params = {{"width", std::int64_t{2}}, {"height", std::int64_t{2}}}});
  layer.components.push_back({.kind = "affine.channel", .version = 1});
  stage.layers.push_back(std::move(layer));
  spec::ModelSpec model{.pipeline = spec::PipelineKind::kImage, .stages = {std::move(stage)}};
  spec::Canonicalize(model);
  const std::array<std::span<const BoundTensor>, 1> tables{};
  const std::array owners{SpatialOwner{2, {1, 4, 2, 1}}, SpatialOwner{5, {1, 4, 2, 1}}};
  const std::array ports{SpatialInput{2, 0, {1, 2, 2, 1}}, SpatialInput{5, 0, {1, 2, 2, 1}}};
  const std::array links{spec::GraphLink{0, {}}, spec::GraphLink{.input = 1, .auxiliary_port = 0}};
  Budget budget;
  auto graph =
      BuildSpatialStep(model, tables, owners, budget.Payment(), {.links = links, .inputs = ports});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  auto exec = ref::Executor::Create(graph->graph());
  ASSERT_TRUE(exec);
  std::array<std::vector<float>, 2> input, affine;
  for (std::size_t owner = 0; owner < 2; ++owner) {
    input[owner].resize(8);
    for (std::size_t i = 0; i < 8; ++i) input[owner][i] = static_cast<float>(i + 16 * owner) / 8.F;
    affine[owner] = {2.F, -1.F, -0.5F, static_cast<float>(owner + 2)};
    ASSERT_TRUE(exec->BindValues(graph->graph().inputs()[2 * owner], input[owner]));
    ASSERT_TRUE(exec->BindValues(graph->graph().inputs()[2 * owner + 1], affine[owner]));
  }
  ASSERT_TRUE(exec->Run());
  for (std::size_t owner = 0; owner < 2; ++owner) {
    auto output = exec->Values(graph->graph().outputs()[owner]);
    ASSERT_TRUE(output);
    ASSERT_EQ(output->size(), 8U);
    EXPECT_EQ(graph->outputs()[owner].shape, (SpatialShape{2, 2, 2, 1}));
    for (std::size_t i = 0; i < 8; ++i)
      EXPECT_DOUBLE_EQ((*output)[i], input[owner][i] * affine[owner][2 * (i / 4)] +
                                         affine[owner][2 * (i / 4) + 1]);
  }
  auto wrong_ports = ports;
  wrong_ports[1].shape = {2, 1, 2, 1};
  Budget refused;
  EXPECT_FALSE(BuildSpatialStep(model, tables, owners, refused.Payment(),
                                {.links = links, .inputs = wrong_ports}));
  EXPECT_EQ(refused.used, 0U);
  auto wrong_owners = owners;
  wrong_owners[0].shape.height = 5;
  EXPECT_FALSE(BuildSpatialStep(model, tables, wrong_owners, refused.Payment(),
                                {.links = links, .inputs = ports}));
  EXPECT_EQ(refused.used, 0U);
}

TEST(SpatialComponentsTest, Actual1024DecoderHandoffIsPaidBeforeReferenceAllocation) {
  const auto& profile = ::llmp::test_support::QwenImageProfile().vae;
  const auto source = model::VaeDecoderTensors(profile, ::llmp::test_support::QwenImageNames().vae);
  auto mapping = spec::ImageDecodeSpec(profile, 64, source);
  ASSERT_TRUE(mapping) << Failed(mapping).value_or("");
  auto tensors = FlowTensors(source);
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  const std::array owners{SpatialOwner{0, {1, 4096, 64, 1}}};
  const std::array ports{SpatialInput{0, 0, {1, 2, 64, 1}}};
  Budget budget;
  auto graph = BuildSpatialStep(mapping->spec, tables, owners, budget.Payment(),
                                {.links = mapping->links,
                                 .reference_buffers = false,
                                 .exports = mapping->exports,
                                 .inputs = ports});
  ASSERT_TRUE(graph) << Failed(graph).value_or("");
  EXPECT_EQ(graph->weights().size(), 128U);
  EXPECT_EQ(graph->outputs()[0].shape, (SpatialShape{1024, 1024, 4, 1}));
  EXPECT_EQ(graph->reference_bytes(), 0U);
  RecordProperty("decoder_metadata_bytes", std::to_string(graph->metadata_bytes()));
  Budget no_reference;
  EXPECT_FALSE(
      BuildSpatialStep(mapping->spec, tables, owners, no_reference.Payment(),
                       {.links = mapping->links, .exports = mapping->exports, .inputs = ports}));
  EXPECT_EQ(no_reference.used, 0U);
  auto invalid = profile;
  invalid.res_blocks = UINT32_MAX;
  EXPECT_FALSE(spec::ImageDecodeSpec(invalid, 64, source));
  invalid = profile;
  invalid.base = UINT32_MAX;
  EXPECT_FALSE(spec::ImageDecodeSpec(invalid, 64, source));
  invalid = profile;
  invalid.spatial_factor = 8;
  EXPECT_FALSE(spec::ImageDecodeSpec(invalid, 64, source));
  EXPECT_FALSE(spec::ImageDecodeSpec(profile, UINT32_MAX, source));
  EXPECT_FALSE(spec::ImageDecodeSpec(profile, 64, std::span(source).first(127)));
}

TEST(ComponentsSpatial, RectangularDecoderCarriesGeometryInItsGraphIdentity) {
  const auto profile = ::llmp::test_support::QwenImageProfile().vae;
  const auto source = model::VaeDecoderTensors(profile, ::llmp::test_support::QwenImageNames().vae);
  auto tensors = FlowTensors(source);
  const std::array<std::span<const BoundTensor>, 1> tables{tensors};
  const std::array ports{SpatialInput{0, 0, {1, 2, 64, 1}}};
  std::vector<base::Sha256Digest> identities;
  for (const auto dimensions : {std::array<std::uint32_t, 2>{32, 64}, {64, 32}, {31, 33}}) {
    const auto height = dimensions[0], width = dimensions[1];
    auto mapping = spec::ImageDecodeSpec(profile, height, width, source);
    ASSERT_TRUE(mapping) << Failed(mapping).value_or("");
    const std::array owners{SpatialOwner{0, {1, std::uint64_t{height} * width, 64, 1}}};
    Budget budget;
    auto graph = BuildSpatialStep(mapping->spec, tables, owners, budget.Payment(),
                                  {.links = mapping->links,
                                   .reference_buffers = false,
                                   .exports = mapping->exports,
                                   .inputs = ports});
    ASSERT_TRUE(graph) << Failed(graph).value_or("");
    EXPECT_EQ(graph->outputs()[0].shape, (SpatialShape{16ULL * width, 16ULL * height, 4, 1}));
    EXPECT_EQ(graph->weights().size(), 128U);
    EXPECT_EQ(graph->reference_bytes(), 0U);
    identities.push_back(graph->graph().Digest());
  }
  ASSERT_EQ(identities.size(), 3U);
  EXPECT_NE(identities[0], identities[1]);  // same area, different row/column layout
  EXPECT_NE(identities[1], identities[2]);
  // The source plan accepts a width beyond the old incidental4096 cap. Actual
  // activation/workspace admission and selected kernel contracts remain later checks.
  EXPECT_TRUE(spec::ImageDecodeSpec(profile, 1, 272, source));
  EXPECT_FALSE(spec::ImageDecodeSpec(profile, 0, 32, source));
  EXPECT_FALSE(spec::ImageDecodeSpec(profile, 32, 0, source));
  EXPECT_FALSE(spec::ImageDecodeSpec(profile, UINT32_MAX, 2, source));
}

}  // namespace
}  // namespace llmp::components
