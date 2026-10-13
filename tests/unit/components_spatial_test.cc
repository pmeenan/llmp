// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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
}  // namespace
}  // namespace llmp::components
