// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Position-dependent shape keys and the primitive lowering needed before
// DeepSeek V4 and DSpark can use shared component execution.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "components/step.h"
#include "dsv4_fixture.h"
#include "engine/component_source.h"
#include "execution/registry.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
#include "planner/pattern.h"
#include "planner/plan.h"
#include "spec/mappings/dsv4.h"

namespace {
namespace co = llmp::components;
namespace en = llmp::engine;
namespace ex = llmp::execution;
namespace fx = llmp::test_support::dsv4;
namespace ir = llmp::ir;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace md = llmp::model;
namespace pl = llmp::planner;
namespace sp = llmp::spec;

ex::Registry Registry(bool indexer = true) {
  std::vector<ex::Implementation> v;
  for (const auto& d : kg::CapabilityDeclarations()) {
    if (!indexer && d.name == kg::kLightningIndexerName) continue;
    v.push_back({.name = std::string(d.name),
                 .operation = d.operation,
                 .source = "ggml",
                 .revision = "planning",
                 .build = "cpu",
                 .variant = std::string(kg::AcceptsTextOf(d.name)),
                 .capability = *d.capability});
  }
  return std::move(ex::Registry::Create(std::move(v)).value());
}

struct Fixture {
  md::Dsv4Profile target = md::Dsv4Flash();
  md::DsparkProfile draft = md::DsparkDeepSeekV4Flash();
  sp::Source target_source =
      fx::SourceOf("deepseek4", fx::Resources(target), sp::mappings::Dsv4Mapping());
  sp::Source draft_source =
      fx::SourceOf("dflash", fx::DrafterResources(draft), sp::mappings::DsparkMapping());
  sp::ModelSpec spec;
  std::array<md::StateLayoutDerived, 2> layouts;
  ex::Registry registry = Registry();
  std::unique_ptr<en::StagedStepSource> source;

  bool Init() {
    const auto tv = sp::mappings::Dsv4V0Values(target);
    const auto dv = sp::mappings::DsparkV0Values(draft);
    auto ts = sp::Synthesize(sp::mappings::Dsv4Mapping(), target_source, *tv);
    if (!ts) {
      ADD_FAILURE() << ts.error();
      return false;
    }
    auto composed = sp::ComposeDrafter(*ts, sp::mappings::Dsv4Mapping(),
                                       sp::mappings::DsparkMapping(), draft_source, *dv);
    if (!composed) {
      ADD_FAILURE() << composed.error();
      return false;
    }
    spec = std::move(*composed);
    std::vector<std::unique_ptr<en::ComponentStepSource>> stages;
    for (std::uint16_t stage = 0; stage < 2; ++stage) {
      auto layout = co::StageLayout(spec, stage, 65536, 512);
      if (!layout) {
        ADD_FAILURE() << layout.error();
        return false;
      }
      layouts[stage] = std::move(*layout);
      auto made = en::ComponentStepSource::Create(
          {.spec = &spec,
           .stage = stage,
           .bindings = {.state = &layouts[stage]},
           .policy = pl::Policy{},
           .backend = {.registry = &registry,
                       .lower = [](const kn::LowerInput&)
                           -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
                         return std::unexpected("not used by key tests");
                       }}});
      if (!made) {
        ADD_FAILURE() << made.error();
        return false;
      }
      stages.push_back(std::move(*made));
    }
    auto joined = en::StagedStepSource::Create(std::move(stages));
    if (!joined) {
      ADD_FAILURE() << joined.error();
      return false;
    }
    source = std::move(*joined);
    return true;
  }

  pl::PlanKey Key(const pl::StepShape& shape) const {
    auto key = source->KeyBase();
    key.shape = shape;
    key.parameters = source->ShapeParameters(shape);
    return key;
  }
};

TEST(GgmlLowerDsv4Test, PlanKeysFollowCompressorDimensionsAndPreserveReuse) {
  Fixture f;
  ASSERT_TRUE(f.Init());
  pl::StepShape a{.segments = {{.rows = 130, .past = 0, .read_cells = 512, .outputs = 1}}};
  pl::StepShape b = a;
  b.segments[0].past = 127;
  ASSERT_EQ(a, b);  // The positional data alone remains outside shape equality.
  EXPECT_NE(f.Key(a), f.Key(b));
  b.segments[0].past = 1;
  EXPECT_EQ(f.Key(a), f.Key(b));
  // A compressed read's 256-cell boundary falls inside one raw read envelope.
  a.segments[0] = {.rows = 1, .past = 1026, .read_cells = 2048, .outputs = 1};
  b = a;
  b.segments[0].past = 1027;
  ASSERT_EQ(a, b);
  EXPECT_NE(f.Key(a), f.Key(b));
  a.read_align = b.read_align = 2048;
  EXPECT_NE(f.Key(a), f.Key(b));
  a.segments[0] = {.rows = 1, .past = 8194, .read_cells = 10240, .outputs = 1};
  b = a;
  b.segments[0].past = 8195;
  EXPECT_NE(f.Key(a), f.Key(b));
  // Every owner contributes, including one whose peers have not advanced.
  a.segments.push_back({.slot = 1, .rows = 130, .past = 0, .read_cells = 512, .outputs = 1});
  b = a;
  b.segments[1].past = 127;
  EXPECT_NE(f.Key(a), f.Key(b));
  // The window-only injected drafter has no position-dependent dimensions.
  a.stage = b.stage = 1;
  EXPECT_TRUE(f.Key(a).parameters.empty());
  EXPECT_EQ(f.Key(a), f.Key(b));
  EXPECT_TRUE(f.source->ShapeParameters({.stage = 2, .segments = {}}).empty());
}

TEST(GgmlLowerDsv4Test, CompressedReadsUseTheContractsPaddingAndHadamardMatchesProduction) {
  EXPECT_EQ(co::CompressedReadCells(1027, 4, 256), 256);
  EXPECT_EQ(co::CompressedReadCells(1028, 4, 256), 512);
  EXPECT_EQ(co::CompressedReadCells(1028, 4, 2048), 2048);
  EXPECT_EQ(co::CompressedReadCells(8196, 4, 2048), 4096);
  for (const auto n : {32, 64, 128, 256})
    EXPECT_EQ(co::HadamardValues(static_cast<std::uint64_t>(n)), kg::HadamardMatrix(n));
}

TEST(GgmlLowerDsv4Test, InvalidStageIsRejectedBeforeCompilingShapeCallbacks) {
  Fixture f;
  ASSERT_TRUE(f.Init());
  auto bad = en::ComponentStepSource::Create(
      {.spec = &f.spec,
       .stage = 2,
       .bindings = {.state = &f.layouts[0]},
       .policy = pl::Policy{},
       .backend = {.registry = &f.registry,
                   .lower = [](const kn::LowerInput&)
                       -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
                     return std::unexpected("unused");
                   }}});
  ASSERT_FALSE(bad);
  EXPECT_EQ(bad.error(), "no such stage");
}

kg::DeviceChoices Device() {
  kg::DeviceChoices d;
  d.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
    return kg::MulMatPath::kCublas;
  };
  d.quant = [](const ggml_tensor* n) -> std::expected<kg::QuantMulMatPath, kg::KernelFailure> {
    for (const auto& e : llmp::format::Encodings()) {
      if (!e.ggml || e.ggml->type != static_cast<std::uint32_t>(n->src[0]->type)) continue;
      const auto rows =
          static_cast<std::uint64_t>(n->op == GGML_OP_MUL_MAT_ID ? n->ne[2] : n->src[1]->ne[1]);
      return kg::MmvqRouted(e.id, 121, rows) ? kg::QuantMulMatPath::kVector
                                             : kg::QuantMulMatPath::kTile;
    }
    return std::unexpected(kg::KernelFailure{.detail = "no quantized encoding"});
  };
  return d;
}

TEST(GgmlLowerDsv4Test, PrimitiveTargetAndDraftStepsLowerWithTypedInputs) {
  Fixture f;
  ASSERT_TRUE(f.Init());
  const std::array<const md::StateLayoutDerived*, 2> layouts{&f.layouts[0], &f.layouts[1]};
  const std::array<std::span<const co::BoundTensor>, 2> tensors{f.target_source.tensors,
                                                                f.draft_source.tensors};
  const pl::DeviceFacts device{.device_class = "sm_121", .arch = 121};
  pl::Policy policy;
  for (const auto& pattern : pl::Patterns()) policy.disabled.emplace_back(pattern.name);
  pl::Canonicalize(policy);
  for (const auto& shape : std::vector<pl::StepShape>{
           {.segments = {{.rows = 7, .past = 0, .read_cells = 768, .outputs = 1}}},
           {.segments = {{.rows = 130, .past = 127, .read_cells = 768, .outputs = 1}}},
           {.segments = {{.rows = 64, .past = 300, .read_cells = 768, .outputs = 2}}},
           {.segments = {{.rows = 512, .read_cells = 768, .outputs = 0, .want_head = false}}},
           {.phase = pl::Phase::kDecode,
            .segments =
                {{.rows = 1, .past = 127, .read_cells = 768, .outputs = 1, .greedy = true}}},
           {.phase = pl::Phase::kDecode,
            .segments = {{.rows = 1, .past = 1027, .read_cells = 2048, .outputs = 1},
                         {.slot = 1, .rows = 1, .past = 40, .read_cells = 2048, .outputs = 1}},
            .read_align = 2048},
           {.stage = 1,
            .phase = pl::Phase::kDraft,
            .segments =
                {{.rows = 5, .past = 812, .read_cells = 256, .outputs = 5, .greedy = true}}},
           {.stage = 1,
            .phase = pl::Phase::kDraft,
            .segments = {{.rows = 5, .past = 812, .read_cells = 256, .outputs = 5, .greedy = true},
                         {.slot = 1,
                          .rows = 5,
                          .past = 945,
                          .read_cells = 256,
                          .outputs = 5,
                          .greedy = true}},
            .read_align = 2048}}) {
    SCOPED_TRACE(shape.stage);
    SCOPED_TRACE(shape.segments.front().past);
    auto spec = f.spec;
    if (shape.stage == 0 && shape.segments.front().past != 300) {
      spec.stages.resize(1);
      spec.stages[0].taps.clear();
    }
    const co::Bindings bindings{.tensors = tensors[shape.stage],
                                .state = layouts[shape.stage],
                                .stages = layouts,
                                .artifacts = tensors};
    auto step = co::BuildStep(spec, shape.stage, shape, bindings, {.narrow = false});
    ASSERT_TRUE(step) << step.error();
    std::vector<ir::TensorId> staged;
    for (const auto& input : step->inputs)
      if (input.kind != co::InputKind::kCellPositions) staged.push_back(input.tensor);
    for (const bool indexer : {true, false}) {
      SCOPED_TRACE(indexer);
      auto registry = Registry(indexer);
      auto plan = pl::Plan(step->graph.Clone(), registry, device, policy,
                           {.staged = staged, .written = kg::WrittenBytes, .shape = &shape});
      ASSERT_TRUE(plan) << plan.error().detail;
      for (const bool device_masks : {false, true}) {
        auto lowered = kg::LowerStep(
            {.plan = *plan, .step = *step, .shape = shape, .addresses = {}, .context = 65536},
            registry, Device(), {.device_masks = device_masks});
        ASSERT_TRUE(lowered) << lowered.error().detail;
        for (const auto* node : lowered->nodes) {
          std::expected<void, kg::KernelFailure> valid;
          switch (node->op) {
            case GGML_OP_CUSTOM:
              if (kg::LlmpOpOf(node) == kg::LlmpOp::kGemma4Mask) valid = kg::CheckGemma4Mask(node);
              break;
            case GGML_OP_ADD:
              valid = kg::CheckBinary(node, GGML_OP_ADD);
              break;
            case GGML_OP_CPY:
              valid = kg::CheckConvert(node);
              break;
            case GGML_OP_TOP_K:
              valid = kg::CheckTopK(node);
              break;
            case GGML_OP_LIGHTNING_INDEXER:
              valid = kg::CheckLightningIndexer(node);
              break;
            case GGML_OP_ROPE:
            case GGML_OP_ROPE_BACK:
              valid = kg::CheckRopeExt(node);
              break;
            default:
              break;
          }
          EXPECT_TRUE(valid) << valid.error().detail << " op " << ggml_op_name(node->op);
        }
        EXPECT_EQ(lowered->outputs.size(), shape.outputs() == 0 ? 0 : shape.segments.size());
        EXPECT_TRUE(std::ranges::any_of(lowered->inputs,
                                        [&](const kn::InputSlot& in) {
                                          return shape.stage == 0
                                                     ? in.kind == kn::InputKind::kPoolReads
                                                     : in.kind == kn::InputKind::kMask && in.ahead;
                                        }) ||
                    (shape.stage == 1 && device_masks));
      }
    }
  }
}

TEST(GgmlLowerDsv4Test, OptInLocalFormsLowerActualTargetAndDraftFragments) {
  Fixture f;
  ASSERT_TRUE(f.Init());
  const std::array<const md::StateLayoutDerived*, 2> layouts{&f.layouts[0], &f.layouts[1]};
  const std::array<std::span<const co::BoundTensor>, 2> tensors{f.target_source.tensors,
                                                                f.draft_source.tensors};
  const std::array<std::string, 4> forms{"dsv4_qhead", "dsv4_compress", "dsv4_hc_pre",
                                         "dsv4_hc_post"};
  for (const auto& shape : std::vector<pl::StepShape>{
           {.segments = {{.rows = 16, .read_cells = 768, .outputs = 1}}},
           {.segments = {{.rows = 130, .past = 127, .read_cells = 768, .outputs = 1}}},
           {.segments = {{.rows = 7, .past = 1027, .read_cells = 2048, .outputs = 1},
                         {.slot = 1, .rows = 9, .past = 127, .read_cells = 2048, .outputs = 1}},
            .read_align = 2048},
           {.stage = 1,
            .phase = pl::Phase::kDraft,
            .segments = {{.rows = 5, .past = 812, .read_cells = 256, .outputs = 5},
                         {.slot = 1, .rows = 1, .past = 945, .read_cells = 256, .outputs = 1}}}}) {
    SCOPED_TRACE(shape.stage);
    SCOPED_TRACE(shape.rows());
    auto spec = f.spec;
    if (shape.stage == 0) {
      spec.stages.resize(1);
      spec.stages[0].taps.clear();
    }
    const co::Bindings bindings{.tensors = tensors[shape.stage],
                                .state = layouts[shape.stage],
                                .stages = layouts,
                                .artifacts = tensors};
    auto step = co::BuildStep(spec, shape.stage, shape, bindings, {.narrow = false});
    ASSERT_TRUE(step) << step.error();
    std::vector<ir::TensorId> staged;
    for (const auto& input : step->inputs)
      if (input.kind != co::InputKind::kCellPositions) staged.push_back(input.tensor);
    // Test each form independently: HC pre includes the flat norm that
    // HC post can pair with, so enabling both gives the earlier pattern priority.
    for (const auto& form : forms) {
      if (shape.stage == 1 && form == "dsv4_compress") continue;
      if (shape.rows() < 16 && form == "dsv4_qhead") continue;
      SCOPED_TRACE(form);
      pl::Policy policy;
      policy.enabled = {form};
      for (const auto& pattern : pl::Patterns())
        if (pattern.name != form) policy.disabled.emplace_back(pattern.name);
      pl::Canonicalize(policy);
      auto plan =
          pl::Plan(step->graph.Clone(), f.registry, {.device_class = "sm_121", .arch = 121}, policy,
                   {.staged = staged, .written = kg::WrittenBytes, .shape = &shape});
      ASSERT_TRUE(plan) << plan.error().detail;
      const auto count = std::ranges::count_if(
          plan->applied, [&](const auto& applied) { return applied.pattern == form; });
      EXPECT_GT(count, 0) << "the actual component fragment must select its local form";
      auto lowered = kg::LowerStep(
          {.plan = *plan, .step = *step, .shape = shape, .addresses = {}, .context = 65536},
          f.registry, Device());
      ASSERT_TRUE(lowered) << lowered.error().detail;
      // LowerStep calls the actual CheckDsv4* launch validators, including
      // packed carry copies and the disjoint HC post/norm output pair.
      EXPECT_EQ(lowered->outputs.size(), shape.segments.size());
    }
  }
}

TEST(GgmlLowerDsv4Test, ConvertChainsBudgetEveryDestinationAndCopyView) {
  using llmp::format::Encoding;
  ir::Builder b;
  const auto input = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({16, 1})));
  auto output = input;
  for (int i = 0; i < 18; ++i)
    output = b.Convert(output, i % 2 == 0 ? Encoding::kF16 : Encoding::kF32);
  b.Output(output);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph) << graph.error().detail;
  co::StepGraph step{.graph = std::move(*graph),
                     .inputs = {{.tensor = input, .kind = co::InputKind::kFeatures}},
                     .weights = {},
                     .state = {},
                     .logits = output};
  const pl::StepShape shape{.segments = {{.rows = 1, .read_cells = 256, .outputs = 1}}};
  auto registry = Registry();
  auto plan =
      pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, {},
               {.staged = std::vector{input}, .written = kg::WrittenBytes, .shape = &shape});
  ASSERT_TRUE(plan) << plan.error().detail;
  auto lowered =
      kg::LowerStep({.plan = *plan, .step = step, .shape = shape, .addresses = {}, .context = 256},
                    registry, Device());
  ASSERT_TRUE(lowered) << lowered.error().detail;
  EXPECT_EQ(lowered->plan.steps.size(), 18U);
  for (const auto& operation : lowered->plan.steps) {
    ASSERT_EQ(operation.nodes.size(), 1U);
    const auto valid = kg::CheckConvert(operation.nodes.front());
    EXPECT_TRUE(valid) << valid.error().detail;
  }
}

TEST(GgmlLowerDsv4Test, FillGraphsBudgetEveryShapeAndResultDescriptor) {
  using llmp::format::Encoding;
  ir::Builder b;
  ir::TensorId output;
  for (int i = 0; i < 18; ++i) {
    output = b.Fill(ir::ShapeOf({16, 1}),
                    i % 2 == 0 ? 0.0F : -std::numeric_limits<float>::infinity(), Encoding::kF32);
    b.Output(output);
  }
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph) << graph.error().detail;
  co::StepGraph step{
      .graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}, .logits = output};
  const pl::StepShape shape{.segments = {{.rows = 1, .read_cells = 256, .outputs = 1}}};
  auto registry = Registry();
  auto plan = pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, {},
                       {.written = kg::WrittenBytes, .shape = &shape});
  ASSERT_TRUE(plan) << plan.error().detail;
  auto lowered =
      kg::LowerStep({.plan = *plan, .step = step, .shape = shape, .addresses = {}, .context = 256},
                    registry, Device());
  ASSERT_TRUE(lowered) << lowered.error().detail;
  ASSERT_EQ(lowered->plan.steps.size(), 18U);
  for (const auto& operation : lowered->plan.steps) {
    ASSERT_EQ(operation.nodes.size(), 1U);
    const auto* node = operation.nodes.front();
    EXPECT_EQ(node->op, GGML_OP_FILL);
    EXPECT_EQ(node->type, GGML_TYPE_F32);
    EXPECT_EQ(node->src[0], nullptr);
    EXPECT_NE(node->data, nullptr);
  }
}

TEST(GgmlLowerDsv4Test, DeviceAheadMasksRefuseUnsupportedBlockGeometry) {
  using llmp::format::Encoding;
  for (const auto [read, window] :
       {std::pair{256U, 0U}, std::pair{128U, 128U}, std::pair{256U, 256U}, std::pair{256U, 128U}}) {
    SCOPED_TRACE(read);
    SCOPED_TRACE(window);
    ir::Builder b;
    const auto positions = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({1})));
    const auto held = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({read})));
    b.SetOwner(0);
    const auto query = b.View(positions, ir::ShapeOf({1}), {4, 4, 4, 4}, 0);
    b.Output(b.MaskBuild(query, held, 32, {.kind = ir::MaskKind::kAhead, .window = window},
                         Encoding::kF16));
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << graph.error().detail;
    co::StepGraph step{.graph = std::move(*graph),
                       .inputs = {{.tensor = positions, .kind = co::InputKind::kPositions},
                                  {.tensor = held,
                                   .kind = co::InputKind::kCellPositions,
                                   .segment = 0,
                                   .cells = {.capacity = 256, .ring = true},
                                   .read = read}},
                       .weights = {},
                       .state = {}};
    const pl::StepShape shape{
        .segments = {{.rows = 1, .read_cells = 256, .outputs = 0, .want_head = false}}};
    auto registry = Registry();
    auto plan =
        pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, {},
                 {.staged = std::vector{positions}, .written = kg::WrittenBytes, .shape = &shape});
    ASSERT_TRUE(plan) << plan.error().detail;
    auto device = kg::LowerStep(
        {.plan = *plan, .step = step, .shape = shape, .addresses = {}, .context = 65536}, registry,
        Device(), {.device_masks = true});
    const bool supported = read == 256 && window == 128;
    EXPECT_EQ(device.has_value(), supported);
    if (!device)
      EXPECT_NE(device.error().detail.find("device block mask's geometry"), std::string::npos);
    auto host = kg::LowerStep(
        {.plan = *plan, .step = step, .shape = shape, .addresses = {}, .context = 65536}, registry,
        Device(), {.device_masks = false});
    EXPECT_TRUE(host) << host.error().detail;
  }
}

TEST(GgmlLowerDsv4Test, FusedCompressorRequiresGeneratedReadsForItsExactOwnerGeometry) {
  using llmp::format::Encoding;
  for (int mutation = 0; mutation <= 6; ++mutation) {
    SCOPED_TRACE(mutation);
    ir::Builder b;
    b.SetOwner(0);
    std::vector<co::StepInput> inputs;
    const auto f32 = [&](std::uint64_t rows) {
      const auto t = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({2, rows})));
      inputs.push_back({.tensor = t, .kind = co::InputKind::kFeatures});
      return t;
    };
    const auto state = f32(2), score = f32(2), current = f32(1), current_score = f32(1);
    const auto ids = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({2})));
    inputs.push_back({.tensor = ids,
                      .kind = mutation == 1 ? co::InputKind::kFeatures : co::InputKind::kPoolReads,
                      .segment = static_cast<std::uint8_t>(mutation == 5 ? 1 : 0),
                      .cells = {.capacity = 2, .ring = true, .recent = true},
                      .ratio = mutation == 2 ? 2U : 1U,
                      .overlap = mutation != 3,
                      .ring = mutation == 4 ? 4U : 2U});
    const auto reads = mutation == 6 ? b.View(ids, ir::ShapeOf({2}), {4, 8, 8, 8}, 0) : ids;
    b.Output(b.Fused(ir::FusedKind::kDsv4Compress,
                     std::array{state, score, current, current_score, reads},
                     ir::CompressorParams{.ratio = 1, .overlap = true})[0]);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << graph.error().detail;
    co::StepGraph step{.graph = std::move(*graph), .inputs = inputs, .weights = {}, .state = {}};
    const pl::StepShape shape{.segments = {{.rows = 1, .outputs = 0, .want_head = false}}};
    std::vector<ir::TensorId> staged;
    for (const auto& in : inputs) staged.push_back(in.tensor);
    auto registry = Registry();
    auto plan = pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, {},
                         {.staged = staged, .written = kg::WrittenBytes, .shape = &shape});
    ASSERT_TRUE(plan) << plan.error().detail;
    auto lowered = kg::LowerStep(
        {.plan = *plan, .step = step, .shape = shape, .addresses = {}, .context = 65536}, registry,
        Device());
    EXPECT_EQ(lowered.has_value(), mutation == 0);
    if (!lowered)
      EXPECT_NE(lowered.error().detail.find("typed bounded pool reads"), std::string::npos);
  }
}

TEST(GgmlLowerDsv4Test, StridedHcPostBoundariesRetainThePrimitiveFallback) {
  for (int strided = 0; strided < 4; ++strided) {
    SCOPED_TRACE(strided);
    ir::Builder b;
    std::vector<co::StepInput> inputs;
    const auto input = [&](int index, llmp::format::Shape shape) {
      auto source_shape = shape;
      if (index == strided) source_shape.ne[0] *= 2;
      const auto root = b.Input(ir::Dense(llmp::format::Encoding::kF32, source_shape));
      inputs.push_back({.tensor = root, .kind = co::InputKind::kFeatures});
      return index == strided ? b.View(root, shape, b.tensor(root).nb, 0) : root;
    };
    const auto y = input(0, ir::ShapeOf({4096, 3}));
    const auto x = input(1, ir::ShapeOf({4096, 4, 3}));
    const auto post = input(2, ir::ShapeOf({4, 3}));
    const auto comb = input(3, ir::ShapeOf({4, 4, 3}));
    const auto out = b.HcPost(y, x, post, comb, {.streams = 4});
    b.Output(out);
    b.Output(b.RmsNorm(b.Reshape(out, ir::ShapeOf({16384, 3})), 1e-6F));
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << graph.error().detail;
    co::StepGraph step{.graph = std::move(*graph), .inputs = inputs, .weights = {}, .state = {}};
    const pl::StepShape shape{.segments = {{.rows = 3, .outputs = 0, .want_head = false}}};
    auto registry = Registry();
    pl::Policy policy;
    policy.enabled = {"dsv4_hc_post"};
    for (const auto& pattern : pl::Patterns())
      if (pattern.name != "dsv4_hc_post") policy.disabled.emplace_back(pattern.name);
    pl::Canonicalize(policy);
    std::vector<ir::TensorId> staged;
    for (const auto& in : inputs) staged.push_back(in.tensor);
    auto plan = pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121},
                         policy, {.staged = staged, .written = kg::WrittenBytes, .shape = &shape});
    ASSERT_TRUE(plan) << plan.error().detail;
    EXPECT_TRUE(plan->applied.empty());
    auto lowered = kg::LowerStep(
        {.plan = *plan, .step = step, .shape = shape, .addresses = {}, .context = 65536}, registry,
        Device());
    ASSERT_TRUE(lowered) << lowered.error().detail;
  }
}

TEST(GgmlLowerDsv4Test, CompressorCapabilitiesAndLaunchValidationBoundBothGridAxes) {
  using llmp::format::Encoding;
  const auto* capability = kg::CapabilityOf(kg::kDsv4CompressName);
  ASSERT_NE(capability, nullptr);
  auto arena = kg::TensorArena::Create(64);
  ASSERT_TRUE(arena);
  std::uint64_t address = 1ULL << 40;
  const auto bind = [&](ggml_tensor* t) {
    kg::TensorArena::Bind(t, address);
    address += 1ULL << 36;
    return t;
  };
  for (const bool overlap : {false, true}) {
    for (const std::uint64_t head : {65535ULL * 64, 65535ULL * 64 + 2}) {
      const std::uint64_t channels = head * (overlap ? 2 : 1);
      const std::uint64_t count = overlap ? 2 : 1;
      const ir::CompressorParams p{.ratio = 1, .overlap = overlap};
      const auto raw = ir::ToBytes(p);
      const std::array<ex::OperandFormat, 6> operands{
          {{.operand = 0, .encoding = Encoding::kF32, .ne = {channels, 1, 1, 1}},
           {.operand = 1, .encoding = Encoding::kF32, .ne = {channels, 1, 1, 1}},
           {.operand = 2, .encoding = Encoding::kF32, .ne = {channels, 1, 1, 1}},
           {.operand = 3, .encoding = Encoding::kF32, .ne = {channels, 1, 1, 1}},
           {.operand = 4, .encoding = Encoding::kI32, .ne = {count, 1, 1, 1}},
           {.operand = ex::kOutputOperand, .encoding = Encoding::kF32, .ne = {head, 1, 1, 1}}}};
      const ex::CapabilityQuery query{
          .op = capability->op, .operands = operands, .arch = 121, .params = raw};
      const bool supported = head == 65535ULL * 64;
      EXPECT_EQ(capability->accepts(query), supported);
      const auto matrix = [&] {
        return bind(ggml_new_tensor_2d(arena->context(), GGML_TYPE_F32,
                                       static_cast<std::int64_t>(channels), 1));
      };
      auto* sk = matrix();
      auto* ss = matrix();
      auto* kv = matrix();
      auto* sc = matrix();
      auto* ids = bind(
          ggml_new_tensor_1d(arena->context(), GGML_TYPE_I32, static_cast<std::int64_t>(count)));
      auto* output = bind(kg::Dsv4Compress(arena->context(), sk, ss, kv, sc, ids, 1, overlap));
      EXPECT_EQ(kg::CheckDsv4Compress(output).has_value(), supported);
    }
  }
}

TEST(GgmlLowerDsv4Test, PrimitivePredicatesRejectUnsupportedTypesAndBounds) {
  using llmp::format::Encoding;
  std::array<ex::OperandFormat, 2> operands{
      {{.operand = 0, .encoding = Encoding::kF32, .ne = {256, 7, 1, 1}},
       {.operand = 1, .encoding = Encoding::kF16, .ne = {256, 7, 1, 1}}}};
  const auto* add = kg::CapabilityOf("ggml.add");
  ASSERT_NE(add, nullptr);
  ex::CapabilityQuery q{.op = add->op, .operands = operands, .arch = 121};
  EXPECT_FALSE(add->accepts(q));
  operands[0].encoding = Encoding::kF16;
  EXPECT_TRUE(add->accepts(q));
  const auto* top = kg::CapabilityOf(kg::kTopKName);
  ASSERT_NE(top, nullptr);
  ir::TopKParams p{.k = 256};
  q.op = top->op;
  q.params = std::as_bytes(std::span(&p, 1));
  operands[0].encoding = Encoding::kF32;
  EXPECT_TRUE(top->accepts(q));
  p.k = 257;
  EXPECT_FALSE(top->accepts(q));
  p.k = 256;
  operands[0].ne[1] = std::numeric_limits<std::uint32_t>::max();
  EXPECT_FALSE(top->accepts(q));
  operands[0].ne = {128, 32, 7, 1};
  operands[1] = {.operand = 3, .encoding = Encoding::kF16, .ne = {256, 7, 1, 1}};
  const auto* indexer = kg::CapabilityOf(kg::kLightningIndexerName);
  ASSERT_NE(indexer, nullptr);
  q.op = indexer->op;
  // Missing keys must refuse even if the query and mask are shaped.
  EXPECT_FALSE(indexer->accepts(q));
}

TEST(GgmlLowerDsv4Test, RejectsMalformedCompressorInputsAndMaskMetadata) {
  Fixture f;
  ASSERT_TRUE(f.Init());
  f.spec.stages.resize(1);
  f.spec.stages[0].taps.clear();
  const pl::StepShape shape{.segments = {{.rows = 7, .read_cells = 768, .outputs = 1}}};
  auto step =
      co::BuildStep(f.spec, 0, shape, {.tensors = f.target_source.tensors, .state = &f.layouts[0]});
  ASSERT_TRUE(step) << step.error();
  pl::Policy policy;
  for (const auto& pattern : pl::Patterns()) policy.disabled.emplace_back(pattern.name);
  pl::Canonicalize(policy);
  std::vector<ir::TensorId> staged;
  for (const auto& input : step->inputs)
    if (input.kind != co::InputKind::kCellPositions) staged.push_back(input.tensor);
  auto plan = pl::Plan(step->graph.Clone(), f.registry, {.device_class = "sm_121", .arch = 121},
                       policy, {.staged = staged, .written = kg::WrittenBytes, .shape = &shape});
  ASSERT_TRUE(plan) << plan.error().detail;
  for (const auto kind :
       {co::InputKind::kPoolReads, co::InputKind::kBlockCells, co::InputKind::kPositionPhases,
        co::InputKind::kHadamard, co::InputKind::kCellPositions}) {
    SCOPED_TRACE(static_cast<int>(kind));
    auto at = std::ranges::find_if(step->inputs, [&](const auto& in) {
      return in.kind == kind && (kind != co::InputKind::kCellPositions || in.cells.positions);
    });
    ASSERT_NE(at, step->inputs.end());
    const auto original = *at;
    switch (kind) {
      case co::InputKind::kPoolReads:
        at->ratio = std::numeric_limits<std::uint32_t>::max();
        break;
      case co::InputKind::kBlockCells:
        at->cells.capacity = 0;
        break;
      case co::InputKind::kPositionPhases:
        at->segment = 0;
        break;
      case co::InputKind::kHadamard:
        at->read = 3;
        break;
      default:
        at->cells.block = 0;
        break;
    }
    auto lowered = kg::LowerStep(
        {.plan = *plan, .step = *step, .shape = shape, .addresses = {}, .context = 65536},
        f.registry, Device());
    EXPECT_FALSE(lowered);
    *at = original;
  }
}

TEST(GgmlLowerDsv4Test, ContractBackingCoversCompressedViewsAtWideRawAlignment) {
  Fixture f;
  ASSERT_TRUE(f.Init());
  f.spec.stages.resize(1);
  f.spec.stages[0].taps.clear();
  for (const auto past : {0U, 1027U, 8195U}) {
    const auto read = ((past + 1 + 2047) / 2048) * 2048;
    const pl::StepShape shape{
        .phase = pl::Phase::kDecode,
        .segments = {{.rows = 1, .past = past, .read_cells = read, .outputs = 1}},
        .read_align = 2048};
    auto step = co::BuildStep(f.spec, 0, shape,
                              {.tensors = f.target_source.tensors, .state = &f.layouts[0]});
    ASSERT_TRUE(step) << step.error();
    auto used = md::UsedThrough(f.layouts[0], read, shape.read_align);
    ASSERT_TRUE(used) << used.error();
    std::size_t checked = 0;
    bool unaligned = false;
    for (const auto& state : step->state) {
      const auto& item = f.layouts[0].items[state.item];
      if (item.contract.cells != md::CellRule::kPerBlock) continue;
      unaligned = unaligned || item.offset % (2ULL << 20U) != 0;
      for (std::uint32_t t = 0; t < step->graph.tensor_count(); ++t) {
        const auto& tensor = step->graph.tensor(ir::TensorId{t});
        if (!tensor.is_view() || step->graph.root(tensor.view_of) != state.tensor) continue;
        // CacheStore aliases its entire destination, but reads/writes only
        // the explicitly indexed cells; the read views below bound attention.
        if (tensor.shape().ne == step->graph.tensor(state.tensor).shape().ne) continue;
        std::uint64_t bytes = ir::UnitBytes(tensor.format);
        for (std::size_t d = 0; d < 4; ++d) bytes += (tensor.ne(d) - 1) * tensor.nb[d];
        const std::uint64_t begin = item.offset + tensor.view_offset;
        EXPECT_TRUE(std::ranges::any_of(*used,
                                        [&](const md::StateRange& r) {
                                          return begin >= r.offset &&
                                                 begin + bytes <= r.offset + r.bytes;
                                        }))
            << "uncovered compressed view at " << begin << " bytes " << bytes;
        ++checked;
      }
    }
    EXPECT_GT(checked, 0U);
    EXPECT_TRUE(unaligned);  // the production cache begins inside a backing extent
  }
}

}  // namespace
