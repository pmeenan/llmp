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
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "components/step.h"
#include "dsv4_fixture.h"
#include "engine/component_source.h"
#include "execution/registry.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/dsv4_outa.h"
#include "kernels/ggml/dsv4_qhead.h"
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
    if (!indexer && (d.name == kg::kLightningIndexerName || d.name == kg::kRoundedIndexScoreName))
      continue;
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

TEST(GgmlLowerDsv4Test, StateBankLayoutIdentityChangesOnlyOptedInSourceKeys) {
  Fixture f;
  ASSERT_TRUE(f.Init());
  auto separated = co::StageLayout(f.spec, 0, 65536, 512, 2 * 1024 * 1024);
  ASSERT_TRUE(separated);
  const pl::StepShape shape{
      .segments = {{.rows = 7, .past = 127, .read_cells = 768, .outputs = 1}}};
  for (const bool enabled : {false, true}) {
    const auto source = [&](const md::StateLayoutDerived& layout) {
      pl::Policy policy;
      if (enabled) policy.enabled = {"state_bank_attention"};
      return en::ComponentStepSource::Create(
          {.spec = &f.spec,
           .bindings = {.state = &layout},
           .policy = policy,
           .backend = {.registry = &f.registry,
                       .lower = [](const kn::LowerInput&)
                           -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
                         return std::unexpected("unused by key proof");
                       }}});
    };
    auto contiguous = source(f.layouts[0]);
    auto gapped = source(*separated);
    ASSERT_TRUE(contiguous);
    ASSERT_TRUE(gapped);
    EXPECT_EQ((*contiguous)->KeyBase(), (*gapped)->KeyBase());
    EXPECT_EQ((*contiguous)->ShapeParameters(shape) == (*gapped)->ShapeParameters(shape), !enabled);
    auto next = shape;
    next.segments[0].past = 128;  // same compressor blocks and read widths
    EXPECT_EQ((*contiguous)->ShapeParameters(shape), (*contiguous)->ShapeParameters(next));
  }
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
  d.pair_glu_fits = [](const ggml_tensor*, const ggml_tensor*) { return true; };
  return d;
}

TEST(GgmlLowerDsv4Test, DirectBankFormulaRequiresProofAndF32OutputForNativeSelection) {
  using llmp::format::Encoding;
  auto registry = Registry();
  const pl::StepShape shape{.segments = {{.rows = 2, .read_cells = 512, .outputs = 2}}};
  const std::vector<std::pair<std::uint32_t, md::StateContract>> contracts{
      {0,
       {.tensors = {{.name = "raw", .cell_bytes = 512},
                    {.name = "compressed", .cell_bytes = 512}}}}};
  const auto layout = md::Layout(contracts, 512, 128);
  ASSERT_TRUE(layout);
  const pl::StateSpans proof =
      [&](std::span<const pl::StateCellSlice> slices) -> std::optional<pl::StateSpanFact> {
    std::vector<md::StateSlice> requested;
    for (const auto& slice : slices) {
      if (slice.ref.stage != 0 || slice.ref.segment != 0) return std::nullopt;
      requested.push_back({.layer = slice.ref.layer,
                           .tensor = slice.ref.tensor,
                           .first = slice.first,
                           .cells = slice.cells});
    }
    auto bank = md::ContiguousStateSpan(*layout, requested);
    return bank ? std::optional<pl::StateSpanFact>{{.bank = *bank, .region_bytes = layout->bytes}}
                : std::nullopt;
  };
  for (const auto encoding : {Encoding::kF32, Encoding::kF16, Encoding::kBF16}) {
    for (const bool has_proof : {false, true}) {
      SCOPED_TRACE(static_cast<int>(encoding));
      SCOPED_TRACE(has_proof);
      ir::Builder b;
      b.SetOwner(0);
      const auto q = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({256, 2, 8})));
      const auto raw = b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({256, 512})), {.tensor = 0});
      const auto compressed =
          b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({256, 512})), {.tensor = 1});
      const auto av = b.View(raw, ir::ShapeOf({256, 1, 512}), {2, 512, 512, 262144}, 0);
      const auto zv = b.View(compressed, ir::ShapeOf({256, 1, 256}), {2, 512, 512, 262144}, 0);
      const auto mask = b.Fill(ir::ShapeOf({768, 32}), 0, Encoding::kF16);
      const auto attended = b.Fused(ir::FusedKind::kStateBankAttention, std::array{q, av, zv, mask},
                                    ir::AttnParams{.scale = 0.1F}, std::array{encoding})[0];
      const auto output = b.Reshape(attended, ir::ShapeOf({2048, 2}));
      b.Output(output);
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
      co::StepGraph step{
          .graph = std::move(*graph),
          .inputs = {{.tensor = q, .kind = co::InputKind::kFeatures, .segment = 0}},
          .weights = {},
          .state = {{.tensor = raw, .ref = {.tensor = 0}, .item = 0, .read = true},
                    {.tensor = compressed, .ref = {.tensor = 1}, .item = 1, .read = true}},
          .logits = output};
      const auto facts = has_proof ? proof : pl::StateSpans{};
      auto plan =
          pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, {},
                   {.staged = std::vector{q},
                    .written = kg::WrittenBytes,
                    .shape = &shape,
                    .state_spans = facts});
      ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
      const bool native = has_proof && encoding == Encoding::kF32;
      EXPECT_EQ(plan->decomposed.empty(), native);
      if (!native)
        EXPECT_EQ(plan->decomposed, std::vector<std::string>{"fused.state_bank_attention"});
      EXPECT_EQ(plan->graph.tensor(plan->graph.outputs()[0]).format.encoding, encoding);
      EXPECT_EQ(std::ranges::count_if(plan->steps,
                                      [&](const auto& selected) {
                                        return registry.at(selected.implementation).name ==
                                               "ggml.state_bank_attention";
                                      }),
                native ? 1 : 0);
      auto lowered = kg::LowerStep({*plan, step, shape, {}, 512, facts}, registry, Device());
      ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
      EXPECT_EQ(std::ranges::any_of(lowered->nodes,
                                    [](const auto* node) { return node->op == GGML_OP_CONCAT; }),
                !native);
      EXPECT_EQ(std::ranges::any_of(
                    plan->graph.nodes(),
                    [](const auto& node) { return node.key == ir::Primitive(ir::Op::kConvert); }),
                encoding != Encoding::kF32);
    }
  }
}

TEST(GgmlLowerDsv4Test, StateBanksBindLayoutSpansAndRefusePointerOrFactDrift) {
  using llmp::format::Encoding;
  ir::Builder b;
  b.SetOwner(0);
  const auto half = [](std::initializer_list<std::uint64_t> ne) {
    return ir::Dense(Encoding::kF16, ir::ShapeOf(ne));
  };
  const auto q = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({512, 2, 8})));
  const auto raw = b.State(half({512, 512}), {.tensor = 0});
  const auto compressed = b.State(half({512, 512}), {.tensor = 1});
  const auto av = b.View(raw, ir::ShapeOf({512, 1, 512}), {2, 1024, 1024, 524288}, 0);
  const auto zv = b.View(compressed, ir::ShapeOf({512, 1, 256}), {2, 1024, 1024, 524288}, 0);
  const auto joined = b.Concat(av, zv, 2);
  const auto mask = b.Fill(ir::ShapeOf({768, 32}), 0, Encoding::kF16);
  const auto attn = b.Attention(q, b.Permute(joined, {0, 2, 1, 3}), b.Permute(joined, {0, 2, 1, 3}),
                                mask, {}, {.scale = 0.1F, .accumulate = ir::Accumulate::kF32});
  const auto output = b.Reshape(attn, ir::ShapeOf({4096, 2}));
  b.Output(output);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  co::StepGraph step{
      .graph = std::move(*graph),
      .inputs = {{.tensor = q, .kind = co::InputKind::kFeatures, .segment = 0}},
      .weights = {},
      .state = {{.tensor = raw, .ref = {.tensor = 0}, .item = 0, .read = true},
                {.tensor = compressed, .ref = {.tensor = 1}, .item = 1, .read = true}},
      .logits = output};
  const pl::StepShape shape{.segments = {{.rows = 2, .read_cells = 512, .outputs = 2}}};
  const std::vector<std::pair<std::uint32_t, md::StateContract>> contracts{
      {0,
       {.tensors = {{.name = "raw", .cell_bytes = 1024},
                    {.name = "compressed", .cell_bytes = 1024}}}}};
  const auto layout = md::Layout(contracts, 512, 128);
  ASSERT_TRUE(layout);
  const pl::StateSpans proof =
      [&](std::span<const pl::StateCellSlice> slices) -> std::optional<pl::StateSpanFact> {
    std::vector<md::StateSlice> requested;
    for (const auto& slice : slices) {
      if (slice.ref.stage != 0 || slice.ref.segment != 0) return std::nullopt;
      requested.push_back({.layer = slice.ref.layer,
                           .tensor = slice.ref.tensor,
                           .first = slice.first,
                           .cells = slice.cells});
    }
    auto bank = md::ContiguousStateSpan(*layout, requested);
    return bank ? std::optional<pl::StateSpanFact>{{.bank = *bank, .region_bytes = layout->bytes}}
                : std::nullopt;
  };
  auto registry = Registry();
  pl::Policy policy;
  policy.enabled = {"state_bank_attention"};
  const auto plan =
      pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, policy,
               {.staged = std::vector{q},
                .written = kg::WrittenBytes,
                .shape = &shape,
                .state_spans = proof});
  ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
  ASSERT_EQ(std::ranges::count_if(
                plan->applied,
                [](const auto& applied) { return applied.pattern == "state_bank_attention"; }),
            1);
  for (const int mode : {0, 1, 2, 3}) {
    SCOPED_TRACE(mode);  // measurement, bound, bad pointer, changing fact
    constexpr std::uint64_t base = std::uint64_t{1} << 40;
    kn::Addresses addresses{};
    if (mode != 0) {
      addresses.state = [&](const ir::StateRef& ref) {
        return base + layout->items[ref.tensor].offset + (mode == 2 && ref.tensor == 1 ? 16 : 0);
      };
      addresses.activations = std::uint64_t{1} << 41;
      addresses.activation_bytes = plan->placement.extent;
    }
    std::uint32_t queries = 0;
    const pl::StateSpans current = [&](std::span<const pl::StateCellSlice> slices) {
      auto fact = proof(slices);
      if (fact && mode == 3 && slices.size() == 2 && queries++ != 0) ++fact->bank.range.offset;
      return fact;
    };
    auto lower = kg::LowerStep({*plan, step, shape, addresses, 512, current}, registry, Device());
    if (mode >= 2) {
      ASSERT_FALSE(lower);
      EXPECT_NE(lower.error().detail.find(mode == 2 ? "pointers" : "changed"), std::string::npos);
      continue;
    }
    ASSERT_TRUE(lower) << (lower ? "" : lower.error().detail);
    const auto attention = std::ranges::find_if(
        lower->nodes, [](const auto* node) { return node->op == GGML_OP_FLASH_ATTN_EXT; });
    ASSERT_NE(attention, lower->nodes.end());
    EXPECT_TRUE(kg::CheckFlashAttnMma(*attention));
    const auto address =
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>((*attention)->src[1]->data));
    if (mode == 1) EXPECT_EQ(address, base);
    EXPECT_TRUE(std::ranges::any_of(lower->tensors, [&](const auto& use) {
      return use.role == ir::Role::kState && use.address == address && use.bytes == 768 * 1024;
    }));
    EXPECT_FALSE(std::ranges::any_of(lower->nodes,
                                     [](const auto* node) { return node->op == GGML_OP_CONCAT; }));
  }
  auto primitive =
      pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, policy,
               {.staged = std::vector{q}, .written = kg::WrittenBytes, .shape = &shape});
  ASSERT_TRUE(primitive);
  EXPECT_TRUE(primitive->applied.empty());
  auto fallback = kg::LowerStep({*primitive, step, shape, {}, 512}, registry, Device());
  ASSERT_TRUE(fallback) << (fallback ? "" : fallback.error().detail);
  EXPECT_TRUE(std::ranges::any_of(fallback->nodes,
                                  [](const auto* node) { return node->op == GGML_OP_CONCAT; }));
}

TEST(GgmlLowerDsv4Test, ComponentSourcesForwardBankLayoutFactsAndRejectMissingBindings) {
  Fixture f;
  ASSERT_TRUE(f.Init());
  auto spec = f.spec;
  spec.stages.resize(1);
  spec.stages[0].taps.clear();
  const auto separated = co::StageLayout(spec, 0, 65536, 512, 2 * 1024 * 1024);
  ASSERT_TRUE(separated);
  const pl::StepShape shape{
      .segments = {{.slot = 3, .rows = 2, .past = 1027, .read_cells = 2048, .outputs = 2},
                   {.slot = 8, .rows = 3, .past = 127, .read_cells = 2048, .outputs = 3}},
      .read_align = 2048};
  pl::Policy policy;
  for (const auto& pattern : pl::Patterns())
    if (pattern.name != "state_bank_attention") policy.disabled.emplace_back(pattern.name);
  policy.enabled = {"state_bank_attention"};
  pl::Canonicalize(policy);
  // Measurement, valid binding, one missing item, one displaced item,
  // all runtime bindings missing, and a layout containing alignment gaps.
  for (const int mode : {0, 1, 2, 3, 4, 5}) {
    SCOPED_TRACE(mode);
    const auto& layout = mode == 5 ? *separated : f.layouts[0];
    kn::Addresses addresses{};
    if (mode != 0 && mode != 5) addresses.activations = std::uint64_t{1} << 42;
    if (mode >= 1 && mode <= 3) {
      addresses.state = [&](const ir::StateRef& ref) -> std::uint64_t {
        if (ref.stage != 0 || ref.segment >= shape.segments.size()) return 0;
        const auto item = std::ranges::find_if(layout.items, [&](const auto& item) {
          return item.layer == ref.layer && item.tensor == ref.tensor;
        });
        if (item == layout.items.end()) return 0;
        if (mode == 2 && item->contract.kind == md::StateKind::kCompressed) return 0;
        const auto base =
            (std::uint64_t{1} << 40) + (std::uint64_t{shape.segments[ref.segment].slot} << 32);
        return base + item->offset +
               (mode == 3 && item->contract.kind == md::StateKind::kCompressed ? 16 : 0);
      };
    }
    bool reached = false;
    auto source = en::ComponentStepSource::Create(
        {.spec = &spec,
         .bindings = {.tensors = f.target_source.tensors, .state = &layout},
         .policy = policy,
         .backend = {
             .registry = &f.registry,
             .device = {.device_class = "sm_121", .arch = 121},
             .written = kg::WrittenBytes,
             .lower = [&](const kn::LowerInput& in)
                 -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
               reached = true;
               if (!in.token->state_spans) return std::unexpected("missing state-span callback");
               std::optional<kg::LoweredStep> measured;
               if (mode == 0) {
                 auto lowered = kg::LowerStep(in, f.registry, Device());
                 EXPECT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
                 if (!lowered) return std::unexpected(lowered.error().detail);
                 measured.emplace(std::move(*lowered));
               }
               std::array<bool, 2> owners{};
               std::size_t pairs = 0;
               for (const auto& node : in.token->step.graph.nodes()) {
                 if (node.key != ir::Primitive(ir::Op::kConcat)) continue;
                 const auto slices =
                     pl::StateBankSlices(in.token->step.graph, node.src[0], node.src[1]);
                 if (!slices) continue;
                 ++pairs;
                 owners.at(slices->front().ref.segment) = true;
                 const auto fact = in.token->state_spans(*slices);
                 EXPECT_EQ(fact.has_value(), mode <= 1);
                 if (fact) {
                   const auto& raw = in.token->step.graph.tensor(node.src[0]);
                   const auto& compressed = in.token->step.graph.tensor(node.src[1]);
                   EXPECT_EQ(fact->region_bytes, layout.bytes);
                   EXPECT_EQ(fact->bank.range.bytes, (raw.ne(2) + compressed.ne(2)) * raw.nb[2]);
                   const std::array requested{md::StateSlice{.layer = slices->front().ref.layer,
                                                             .tensor = slices->front().ref.tensor,
                                                             .cells = slices->front().cells},
                                              md::StateSlice{.layer = slices->back().ref.layer,
                                                             .tensor = slices->back().ref.tensor,
                                                             .cells = slices->back().cells}};
                   const auto independently = md::ContiguousStateSpan(layout, requested);
                   EXPECT_TRUE(independently);
                   if (independently) {
                     EXPECT_EQ(fact->bank.range.offset, independently->range.offset);
                     EXPECT_EQ(fact->bank.range.bytes, independently->range.bytes);
                   }
                   const auto owner = slices->front().ref.segment;
                   const auto used =
                       md::UsedThrough(layout, shape.segments[owner].read_cells, shape.read_align);
                   EXPECT_TRUE(used);
                   if (used) {
                     std::uint64_t covered = fact->bank.range.offset;
                     for (const auto& range : *used) {
                       if (range.offset <= covered && covered < range.offset + range.bytes)
                         covered = range.offset + range.bytes;
                       if (covered >= fact->bank.range.offset + fact->bank.range.bytes) break;
                     }
                     EXPECT_GE(covered, fact->bank.range.offset + fact->bank.range.bytes);
                   }
                   if (measured) {
                     const auto root = in.token->step.graph.root(node.src[0]);
                     const auto leaf = std::ranges::find_if(
                         measured->leaves,
                         [&](const auto& leaf) { return leaf.primitive == root; });
                     EXPECT_NE(leaf, measured->leaves.end());
                     if (leaf != measured->leaves.end()) {
                       const auto address = static_cast<std::uint64_t>(
                           reinterpret_cast<std::uintptr_t>(leaf->tensor->data));
                       EXPECT_TRUE(std::ranges::any_of(measured->tensors, [&](const auto& use) {
                         return use.role == ir::Role::kState && use.address == address &&
                                use.bytes == fact->bank.range.bytes;
                       }));
                     }
                   }
                 }
                 auto bad = *slices;
                 bad[1].ref.segment ^= 1;
                 EXPECT_FALSE(in.token->state_spans(bad));
                 bad = *slices;
                 ++bad[1].ref.stage;
                 EXPECT_FALSE(in.token->state_spans(bad));
                 bad = *slices;
                 bad[0].ref.segment = bad[1].ref.segment = 2;
                 EXPECT_FALSE(in.token->state_spans(bad));
               }
               EXPECT_GT(pairs, 0U);
               EXPECT_TRUE(owners[0]);
               EXPECT_TRUE(owners[1]);
               EXPECT_EQ(std::ranges::count_if(in.plan.applied,
                                               [](const auto& applied) {
                                                 return applied.pattern == "state_bank_attention";
                                               }),
                         mode <= 1 ? pairs : 0U);
               return std::unexpected("state-span callback checked");
             }}});
    ASSERT_TRUE(source) << (source ? "" : source.error());
    const auto built = (*source)->Build(shape, addresses);
    ASSERT_FALSE(built);
    EXPECT_TRUE(reached) << built.error();
    EXPECT_EQ(built.error(), "state-span callback checked");
  }
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
        auto lowered = kg::LowerStep({*plan, *step, shape, {}, 65536}, registry, Device(),
                                     {.device_masks = device_masks});
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
      if (shape.rows() > 16 && form == "dsv4_hc_pre") continue;
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
      auto lowered = kg::LowerStep({*plan, *step, shape, {}, 65536}, f.registry, Device());
      ASSERT_TRUE(lowered) << lowered.error().detail;
      // LowerStep calls the actual CheckDsv4* launch validators, including
      // packed carry copies and the disjoint HC post/norm output pair.
      EXPECT_EQ(lowered->outputs.size(), shape.segments.size());
    }
  }
}

TEST(GgmlLowerDsv4Test, RoundedIndexScoreUsesFundedPrimitiveRoundTripWithoutNativeScore) {
  using llmp::format::Encoding;
  ir::Builder b;
  const auto q = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({128, 64, 3})));
  const auto k = b.Input(ir::Dense(Encoding::kF16, ir::ShapeOf({128, 1, 4})));
  const auto w = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({64, 3})));
  const auto counts = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({3})));
  b.Output(b.RoundedIndexScore(q, k, w, counts));
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph) << graph.error().detail;
  co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
  const std::vector<ir::TensorId> staged{q, k, w, counts};
  for (const auto input : staged)
    step.inputs.push_back({.tensor = input, .kind = co::InputKind::kFeatures});
  const pl::StepShape shape{.segments = {{.rows = 3, .outputs = 3}}};
  auto registry = Registry(false);
  pl::Policy policy;
  for (const auto& pattern : pl::Patterns()) policy.disabled.emplace_back(pattern.name);
  pl::Canonicalize(policy);
  auto plan = pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121},
                       policy, {.staged = staged, .written = kg::WrittenBytes, .shape = &shape});
  ASSERT_TRUE(plan) << plan.error().detail;
  EXPECT_EQ(plan->decomposed, std::vector<std::string>{"rounded_index_score"});
  std::vector<ir::TensorId> cast_outputs;
  std::vector<Encoding> cast_encodings;
  int products = 0, reductions = 0, prefixes = 0;
  for (const auto& node : plan->graph.nodes()) {
    EXPECT_NE(node.key.op, ir::Op::kRoundedIndexScore);
    EXPECT_NE(node.key.op, ir::Op::kIndexScore);
    if (node.key.op == ir::Op::kConvert) {
      cast_outputs.push_back(node.out[0]);
      cast_encodings.push_back(plan->graph.tensor(node.out[0]).format.encoding);
    }
    products += node.key.op == ir::Op::kMatMul;
    reductions += node.key.op == ir::Op::kSumRows;
    prefixes += node.key.op == ir::Op::kPrefixOverwrite;
  }
  EXPECT_EQ(cast_encodings, (std::vector<Encoding>{Encoding::kF16, Encoding::kF32}));
  EXPECT_EQ(products, 1);
  EXPECT_EQ(reductions, 1);
  EXPECT_EQ(prefixes, 1);
  ASSERT_EQ(cast_outputs.size(), 2U);
  std::array<std::uint64_t, 2> offsets{};
  for (std::size_t i = 0; i < 2; ++i) {
    const auto at = std::ranges::find_if(plan->placement.offsets, [&](const auto& placed) {
      return placed.first == cast_outputs[i];
    });
    ASSERT_NE(at, plan->placement.offsets.end());
    offsets[i] = at->second;
    const auto bytes = pl::RootBytes(plan->graph.tensor(cast_outputs[i]));
    ASSERT_TRUE(bytes);
    EXPECT_LE(offsets[i] + *bytes, plan->placement.extent);
  }
  EXPECT_NE(offsets[0], offsets[1]) << "widening reads the half root while writing its F32 root";
  auto lowered = kg::LowerStep({*plan, step, shape, {}, 256}, registry, Device());
  ASSERT_TRUE(lowered) << lowered.error().detail;
  int native_prefixes = 0, copies = 0;
  for (const auto* node : lowered->nodes) {
    if (kg::LlmpOpOf(node) == kg::LlmpOp::kPrefixOverwrite) {
      ++native_prefixes;
      EXPECT_TRUE(kg::CheckPrefixOverwrite(node));
    }
    if (node->op == GGML_OP_CPY) {
      ++copies;
      EXPECT_TRUE(kg::CheckConvert(node));
    }
  }
  EXPECT_EQ(native_prefixes, 1);
  EXPECT_EQ(copies, 2);
}

TEST(GgmlLowerDsv4Test, ContCapabilityMatchesElementTypesAndGridBound) {
  using llmp::format::Encoding;
  const auto* cap = kg::CapabilityOf(kg::kContName);
  ASSERT_NE(cap, nullptr);
  constexpr auto max = static_cast<std::uint64_t>(INT32_MAX) * 64 - 1;
  for (const auto encoding : {Encoding::kF32, Encoding::kF16, Encoding::kI32, Encoding::kBF16})
    for (const auto cells : {std::uint64_t{0}, std::uint64_t{1}, max, max + 1}) {
      std::array<ex::OperandFormat, 2> operands{
          {{.operand = 0, .encoding = encoding, .ne = {cells, 1, 1, 1}},
           {.operand = ex::kOutputOperand, .encoding = encoding, .ne = {cells, 1, 1, 1}}}};
      const ex::CapabilityQuery query{.op = cap->op, .operands = operands, .arch = 121};
      const bool supported = encoding != Encoding::kBF16 && cells > 0 && cells <= max;
      EXPECT_EQ(cap->accepts(query), supported);
      auto arena = kg::TensorArena::Create(4);
      ASSERT_TRUE(arena);
      const auto type = encoding == Encoding::kF32   ? GGML_TYPE_F32
                        : encoding == Encoding::kF16 ? GGML_TYPE_F16
                        : encoding == Encoding::kI32 ? GGML_TYPE_I32
                                                     : GGML_TYPE_BF16;
      auto* input = ggml_new_tensor_1d(arena->context(), type, static_cast<std::int64_t>(cells));
      kg::TensorArena::Bind(input, 1ULL << 40);
      auto* output = ggml_cont(arena->context(), input);
      kg::TensorArena::Bind(output, 1ULL << 43);
      EXPECT_EQ(kg::CheckCont(output).has_value(), supported);
      operands[1].encoding = encoding == Encoding::kF32 ? Encoding::kF16 : Encoding::kF32;
      EXPECT_FALSE(cap->accepts(query));
    }
}

TEST(GgmlLowerDsv4Test, PrefixOverwriteDescriptorPreservesEveryLiteralParameterBit) {
  auto arena = kg::TensorArena::Create(16);
  ASSERT_TRUE(arena);
  auto* values = ggml_new_tensor_2d(arena->context(), GGML_TYPE_F32, 5, 3);
  auto* counts = ggml_new_tensor_1d(arena->context(), GGML_TYPE_I32, 3);
  for (const std::uint32_t fill : {0U, 0x80000000U, 0xff800000U, 0x7f800000U, 0x7fa00001U,
                                   0xffa12345U, 0xffffffffU, 0x807fffffU}) {
    const auto* output = kg::PrefixOverwrite(arena->context(), values, counts, fill);
    EXPECT_EQ(kg::LlmpOpOf(output), kg::LlmpOp::kPrefixOverwrite);
    EXPECT_EQ(static_cast<std::uint32_t>(kg::LlmpOpInt(output, 0)), fill);
  }
}

TEST(GgmlLowerDsv4Test, PrefixOverwriteCapabilityAndLaunchAgreeOnBoundsAndAliasing) {
  using llmp::format::Encoding;
  for (const auto cells :
       {std::uint64_t{1}, std::uint64_t{INT32_MAX / 2}, std::uint64_t{INT32_MAX / 2 + 1}}) {
    auto arena = kg::TensorArena::Create(8);
    ASSERT_TRUE(arena);
    std::uint64_t address = 1ULL << 40;
    const auto bind = [&](ggml_tensor* t) {
      kg::TensorArena::Bind(t, address);
      address += 1ULL << 36;
      return t;
    };
    auto* values = bind(
        ggml_new_tensor_2d(arena->context(), GGML_TYPE_F32, static_cast<std::int64_t>(cells), 2));
    auto* counts = bind(ggml_new_tensor_1d(arena->context(), GGML_TYPE_I32, 2));
    auto* output = bind(kg::PrefixOverwrite(arena->context(), values, counts, 0x7fa00001));
    const auto* cap = kg::CapabilityOf(kg::kPrefixOverwriteName);
    ASSERT_NE(cap, nullptr);
    const auto params = ir::ToBytes(ir::PrefixOverwriteParams{.fill_bits = 0x7fa00001});
    const std::array<ex::OperandFormat, 3> operands{
        {{.operand = 0, .encoding = Encoding::kF32, .ne = {cells, 2, 1, 1}},
         {.operand = 1, .encoding = Encoding::kI32, .ne = {2, 1, 1, 1}},
         {.operand = ex::kOutputOperand, .encoding = Encoding::kF32, .ne = {cells, 2, 1, 1}}}};
    const ex::CapabilityQuery query{
        .op = cap->op, .operands = operands, .arch = 121, .params = params};
    const bool supported = cells <= INT32_MAX / 2;
    EXPECT_EQ(cap->accepts(query), supported);
    EXPECT_EQ(kg::CheckPrefixOverwrite(output).has_value(), supported);
    if (!supported) continue;
    values->nb[0] = counts->nb[0] = 0;
    EXPECT_TRUE(kg::CheckPrefixOverwrite(output));
    values->nb[0] = 3;
    EXPECT_FALSE(kg::CheckPrefixOverwrite(output));
    values->nb[0] = 4;
    counts->nb[0] = 3;
    EXPECT_FALSE(kg::CheckPrefixOverwrite(output));
    counts->nb[0] = 4;
    output->data = values->data;
    EXPECT_FALSE(kg::CheckPrefixOverwrite(output));
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
  auto lowered = kg::LowerStep({*plan, step, shape, {}, 256}, registry, Device());
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
  auto lowered = kg::LowerStep({*plan, step, shape, {}, 256}, registry, Device());
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
    auto device =
        kg::LowerStep({*plan, step, shape, {}, 65536}, registry, Device(), {.device_masks = true});
    const bool supported = read == 256 && window == 128;
    EXPECT_EQ(device.has_value(), supported);
    if (!device)
      EXPECT_NE(device.error().detail.find("device block mask's geometry"), std::string::npos);
    auto host =
        kg::LowerStep({*plan, step, shape, {}, 65536}, registry, Device(), {.device_masks = false});
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
    auto lowered = kg::LowerStep({*plan, step, shape, {}, 65536}, registry, Device());
    EXPECT_EQ(lowered.has_value(), mutation == 0);
    if (!lowered)
      EXPECT_NE(lowered.error().detail.find("typed bounded pool reads"), std::string::npos);
  }
}

TEST(GgmlLowerDsv4Test, SinkhornHcMixKeepsUnsupportedFormsOnTheirPrimitiveFallback) {
  using llmp::format::Encoding;
  for (int control = 0; control < 9; ++control) {
    SCOPED_TRACE(control);
    ir::Builder b;
    b.SetOwner(0);
    const std::uint64_t hc = control == 5 ? 2 : 4, count = (hc + 2) * hc;
    std::vector<co::StepInput> inputs;
    const auto input = [&](llmp::format::Shape shape, bool strided, bool offset) {
      auto root_shape = shape;
      if (strided) root_shape.ne[0] *= 2;
      if (offset) root_shape = ir::ShapeOf({count * 3 + 1});
      const auto root = b.Input(ir::Dense(Encoding::kF32, root_shape));
      inputs.push_back({.tensor = root, .kind = co::InputKind::kFeatures});
      if (offset)
        return b.View(root, shape, ir::NaturalStrides(ir::Dense(Encoding::kF32, shape)), 4);
      // IR views require unit nb[0]. The larger root retains a valid
      // noncanonical row/matrix pitch even for a vector; do not invent
      // an element-strided vector outside the IR view contract.
      return strided ? b.View(root, shape, b.tensor(root).nb, 0) : root;
    };
    const auto mixes = input(ir::ShapeOf({count, 3}), control == 1, control == 8);
    ASSERT_TRUE(mixes.valid());
    const auto scale = input(ir::ShapeOf({control == 7 ? 4U : 3U}), control == 2, false);
    ASSERT_TRUE(scale.valid());
    const auto base = input(ir::ShapeOf({count}), control == 3, false);
    ASSERT_TRUE(base.valid());
    b.Output(b.HcMix(mixes, scale, base,
                     {.mix = control == 4 ? ir::HcMixKind::kSigmoid : ir::HcMixKind::kSinkhorn,
                      .streams = static_cast<std::uint32_t>(hc),
                      .iterations = 5,
                      .eps = 1e-6F}));
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
    co::StepGraph step{.graph = std::move(*graph), .inputs = inputs, .weights = {}, .state = {}};
    std::vector<ex::Implementation> declarations;
    for (const auto& d : kg::CapabilityDeclarations()) {
      if (control == 6 && d.name == kg::kHcCombName) continue;
      declarations.push_back({.name = std::string(d.name),
                              .operation = d.operation,
                              .source = "ggml",
                              .revision = "test",
                              .build = "test",
                              .variant = "test",
                              .capability = *d.capability});
    }
    auto registry = ex::Registry::Create(std::move(declarations));
    ASSERT_TRUE(registry);
    const pl::StepShape shape{.segments = {{.rows = 3, .outputs = 0, .want_head = false}}};
    std::vector<ir::TensorId> staged;
    for (const auto& in : inputs) staged.push_back(in.tensor);
    auto plan = pl::Plan(step.graph.Clone(), *registry, {.device_class = "sm_121", .arch = 121}, {},
                         {.staged = staged, .written = kg::WrittenBytes, .shape = &shape});
    ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
    const bool native = control == 0 || control == 8;
    EXPECT_EQ(std::ranges::count_if(plan->steps,
                                    [&](const auto& s) {
                                      return registry->at(s.implementation).name == kg::kHcCombName;
                                    }),
              native ? 1 : 0);
    auto lowered = kg::LowerStep({*plan, step, shape, {}, 256}, *registry, Device());
    ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
    EXPECT_EQ(
        std::ranges::count_if(lowered->plan.steps,
                              [](const auto& s) { return s.implementation == kg::kHcCombName; }),
        native ? 1 : 0);
    if (native) {
      ASSERT_EQ(lowered->plan.steps.size(), 1U);
      const auto* node = lowered->plan.steps[0].nodes[0];
      EXPECT_EQ(node->op, GGML_OP_DSV4_HC_COMB);
      EXPECT_EQ(node->op_params[1], 5);
      EXPECT_TRUE(kg::CheckHcComb(node));
    }
  }
}

TEST(GgmlLowerDsv4Test, SinkhornHcMixAdmissionAndBoundLaunchKeepTheirOwnGuards) {
  using llmp::format::Encoding;
  const auto* cap = kg::CapabilityOf(kg::kHcCombName);
  ASSERT_NE(cap, nullptr);
  constexpr std::uint64_t limit = static_cast<std::uint64_t>(INT32_MAX) * 256 - 1;
  for (const auto rows : std::array<std::uint64_t, 3>{1, limit, limit + 1}) {
    auto arena = kg::TensorArena::Create(8);
    ASSERT_TRUE(arena);
    std::uint64_t address = 1ULL << 48;
    const auto bind = [&](ggml_tensor* t) {
      kg::TensorArena::Bind(t, address);
      address += 1ULL << 48;
      return t;
    };
    auto* mixes = bind(
        ggml_new_tensor_2d(arena->context(), GGML_TYPE_F32, 24, static_cast<std::int64_t>(rows)));
    auto* scale = bind(ggml_new_tensor_1d(arena->context(), GGML_TYPE_F32, 3));
    auto* base = bind(ggml_new_tensor_1d(arena->context(), GGML_TYPE_F32, 24));
    auto* out = bind(ggml_dsv4_hc_comb(arena->context(), mixes, scale, base, 1e-6F, 5));
    std::array<ex::OperandFormat, 4> operands{};
    const std::array tensors{mixes, scale, base, out};
    for (std::size_t i = 0; i < operands.size(); ++i) {
      auto& o = operands[i];
      o.operand = i == 3 ? ex::kOutputOperand : static_cast<std::uint8_t>(i);
      o.encoding = Encoding::kF32;
      o.memory = ex::OperandMemory{};
      for (std::size_t d = 0; d < 4; ++d) {
        o.ne[d] = static_cast<std::uint64_t>(tensors[i]->ne[d]);
        o.memory->nb[d] = tensors[i]->nb[d];
      }
    }
    const ir::HcParams p{
        .mix = ir::HcMixKind::kSinkhorn, .streams = 4, .iterations = 5, .eps = 1e-6F};
    auto params = ir::ToBytes(p);
    const ex::CapabilityQuery query{
        .op = cap->op, .operands = operands, .arch = 121, .params = params};
    EXPECT_EQ(cap->accepts(query), rows <= limit);
    EXPECT_EQ(kg::CheckHcComb(out).has_value(), rows <= limit);
    if (rows > limit) continue;
    const auto packed = operands;
    for (std::size_t i = 0; i < operands.size(); ++i) {
      operands[i].memory.reset();
      EXPECT_FALSE(cap->accepts(query));
      operands = packed;
      operands[i].memory->view_offset = 2;
      EXPECT_FALSE(cap->accepts(query));
      operands = packed;
      operands[i].encoding = Encoding::kBF16;
      EXPECT_FALSE(cap->accepts(query));
      operands = packed;
      operands[i].memory->nb[0] *= 2;
      EXPECT_FALSE(cap->accepts(query));
      operands = packed;
    }
    for (const std::uint32_t iterations : {0U, 1U, 100U, 101U, UINT32_MAX}) {
      auto changed = p;
      changed.iterations = iterations;
      params = ir::ToBytes(changed);
      EXPECT_EQ(cap->accepts(query), iterations >= 1 && iterations <= 100);
    }
    for (const float eps : {-1.0F, 0.0F, std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
      auto changed = p;
      changed.eps = eps;
      params = ir::ToBytes(changed);
      EXPECT_EQ(cap->accepts(query), eps == 0.0F);
    }
    params = ir::ToBytes(p);
    const auto output_address = out->data;
    for (const auto* source : {mixes, scale, base}) {
      out->data = source->data;
      EXPECT_FALSE(kg::CheckHcComb(out));
    }
    out->data = static_cast<std::byte*>(output_address) + 2;
    EXPECT_FALSE(kg::CheckHcComb(out));
    out->data = output_address;
    EXPECT_TRUE(kg::CheckHcComb(out));
  }
}

TEST(GgmlLowerDsv4Test, WeightedHcPreKeepsUnsupportedFormsOnTheirPrimitiveFallback) {
  using llmp::format::Encoding;
  for (int control = 0; control < 7; ++control) {
    SCOPED_TRACE(control);
    ir::Builder b;
    b.SetOwner(0);
    const std::uint64_t hc = control == 4 ? 2 : 4;
    std::vector<co::StepInput> inputs;
    const auto input = [&](llmp::format::Shape shape, bool strided, bool offset) {
      const auto strides = ir::NaturalStrides(ir::Dense(Encoding::kF32, shape));
      auto root_shape = shape;
      if (strided) root_shape.ne[0] *= 2;
      if (offset) root_shape = ir::ShapeOf({32 * 4 * 3 + 1});
      const auto root = b.Input(ir::Dense(Encoding::kF32, root_shape));
      inputs.push_back({.tensor = root, .kind = co::InputKind::kFeatures});
      if (offset) return b.View(root, shape, strides, 4);
      return strided ? b.View(root, shape, b.tensor(root).nb, 0) : root;
    };
    const auto x = input(ir::ShapeOf({32, hc, 3}), control == 1, control == 6);
    const bool gated = control == 3;
    const auto weights =
        input(gated ? ir::ShapeOf({32, hc, 3}) : ir::ShapeOf({hc, 3}), control == 2, false);
    b.Output(b.HcPre(x, weights,
                     {.fold = gated ? ir::HcFold::kGated : ir::HcFold::kWeights,
                      .streams = static_cast<std::uint32_t>(hc)}));
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
    co::StepGraph step{.graph = std::move(*graph), .inputs = inputs, .weights = {}, .state = {}};
    std::vector<ex::Implementation> declarations;
    for (const auto& d : kg::CapabilityDeclarations()) {
      if (control == 5 && d.name == kg::kHcPreName) continue;
      declarations.push_back({.name = std::string(d.name),
                              .operation = d.operation,
                              .source = "ggml",
                              .revision = "test",
                              .build = "test",
                              .variant = "test",
                              .capability = *d.capability});
    }
    auto made_registry = ex::Registry::Create(std::move(declarations));
    ASSERT_TRUE(made_registry);
    auto& registry = *made_registry;
    const pl::StepShape shape{.segments = {{.rows = 3, .outputs = 0, .want_head = false}}};
    std::vector<ir::TensorId> staged;
    for (const auto& in : inputs) staged.push_back(in.tensor);
    auto plan = pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, {},
                         {.staged = staged, .written = kg::WrittenBytes, .shape = &shape});
    ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
    const bool native = control == 0 || control == 6;
    EXPECT_EQ(std::ranges::count_if(plan->steps,
                                    [&](const auto& s) {
                                      return registry.at(s.implementation).name == kg::kHcPreName;
                                    }),
              native ? 1 : 0);
    auto lowered = kg::LowerStep({*plan, step, shape, {}, 256}, registry, Device());
    ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
    EXPECT_EQ(
        std::ranges::count_if(lowered->plan.steps,
                              [](const auto& s) { return s.implementation == kg::kHcPreName; }),
        native ? 1 : 0);
    if (native) {
      ASSERT_EQ(lowered->plan.steps.size(), 1U);
      const auto* node = lowered->plan.steps[0].nodes[0];
      EXPECT_EQ(node->op, GGML_OP_DSV4_HC_PRE);
      EXPECT_TRUE(kg::CheckHcPre(node));
    }
  }
}

TEST(GgmlLowerDsv4Test, SinkhornComponentLayerUsesNativeFoldAndFinalHeadKeepsSeparateProducts) {
  using llmp::format::Encoding;
  const auto dense = [](std::initializer_list<std::uint64_t> dims) {
    // Artifact weight roles retain their declared rank, including the
    // head's one-element scale vector; IR ShapeOf canonicalizes it to rank 0.
    auto shape = ir::ShapeOf(dims);
    shape.rank = static_cast<std::uint8_t>(dims.size());
    return ir::Dense(Encoding::kF32, shape);
  };
  for (int control = 0; control < 3; ++control) {
    SCOPED_TRACE(control);
    const bool head = control != 0, gather = control == 2;
    sp::ComponentRef ref{
        .kind = "hc.mix",
        .version = 1,
        .params = {{.name = "eps", .value = sp::ParamValue(1e-6)},
                   {.name = "hc_eps", .value = sp::ParamValue(1e-6)},
                   {.name = "mix", .value = sp::ParamValue(std::string("sinkhorn"))}},
        .weights = {}};
    if (head) {
      ref.params.push_back({.name = "inject", .value = sp::ParamValue(false)});
      ref.params.push_back({.name = "gather", .value = sp::ParamValue(gather)});
    } else
      ref.params.push_back({.name = "iterations", .value = sp::ParamValue(std::int64_t{5})});
    std::ranges::sort(ref.params, {}, &sp::Param::name);
    auto instance = co::MakeInstance(ref, {.width = 32, .context = 256, .streams = 4},
                                     head ? co::Site::kFinalNorm : co::Site::kLayer, 0, 2);
    ASSERT_TRUE(instance) << (instance ? "" : instance.error());
    const std::uint64_t mixes = head ? 4 : 24;
    const std::array formats{dense({128}),        dense({1}),     dense({1}),
                             dense({128, mixes}), dense({mixes}), dense({head ? 1U : 3U}),
                             dense({32})};
    const pl::StepShape shape{
        .segments = {{.rows = 3, .read_cells = 256, .outputs = gather ? 1U : 3U}}};
    auto component = co::BuildComponent(*instance, shape, formats, {}, {.residual = true});
    ASSERT_TRUE(component) << (component ? "" : component.error());
    auto& step = component->step;
    // BuildComponent exposes caller-owned leaves. This fixture stages
    // those operands without changing the actual fragment or output-row
    // metadata; LowerStep deliberately refuses unbound lone inputs.
    for (auto& in : step.inputs)
      if (in.kind == co::InputKind::kStream || in.kind == co::InputKind::kResidual ||
          in.kind == co::InputKind::kCarry || in.kind == co::InputKind::kSide)
        in.kind = co::InputKind::kFeatures;
    EXPECT_EQ(
        std::ranges::count_if(step.graph.nodes(),
                              [](const auto& n) { return n.key == ir::Primitive(ir::Op::kHcPre); }),
        head ? 0 : 1);
    EXPECT_EQ(
        std::ranges::count_if(step.graph.nodes(),
                              [](const auto& n) { return n.key == ir::Primitive(ir::Op::kHcMix); }),
        head ? 0 : 1);
    if (head) {
      // The actual final-norm fragment's three width-wide accumulation
      // Adds stay explicit; this is not a same-shaped synthetic stand-in.
      EXPECT_EQ(std::ranges::count_if(step.graph.nodes(),
                                      [&](const auto& n) {
                                        return n.key == ir::Primitive(ir::Op::kAdd) &&
                                               step.graph.tensor(n.out[0]).ne(0) == 32;
                                      }),
                3);
    }
    auto registry = Registry();
    pl::Policy policy;
    for (const auto& pattern : pl::Patterns()) policy.disabled.emplace_back(pattern.name);
    pl::Canonicalize(policy);
    std::vector<ir::TensorId> staged;
    for (const auto& in : step.inputs) staged.push_back(in.tensor);
    auto plan = pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121},
                         policy, {.staged = staged, .written = kg::WrittenBytes, .shape = &shape});
    ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
    EXPECT_EQ(std::ranges::count_if(plan->steps,
                                    [&](const auto& s) {
                                      return registry.at(s.implementation).name == kg::kHcPreName;
                                    }),
              head ? 0 : 1);
    auto lowered = kg::LowerStep({*plan, step, shape, {}, 256}, registry, Device());
    ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
    EXPECT_EQ(
        std::ranges::count_if(lowered->plan.steps,
                              [](const auto& s) { return s.implementation == kg::kHcPreName; }),
        head ? 0 : 1);
    EXPECT_EQ(std::ranges::count_if(plan->steps,
                                    [&](const auto& s) {
                                      return registry.at(s.implementation).name == kg::kHcCombName;
                                    }),
              head ? 0 : 1);
    EXPECT_EQ(
        std::ranges::count_if(lowered->plan.steps,
                              [](const auto& s) { return s.implementation == kg::kHcCombName; }),
        head ? 0 : 1);
    for (const auto& s : lowered->plan.steps) {
      if (s.implementation == kg::kHcPreName) EXPECT_TRUE(kg::CheckHcPre(s.nodes[0]));
      if (s.implementation == kg::kHcCombName) EXPECT_TRUE(kg::CheckHcComb(s.nodes[0]));
    }
  }
}

TEST(GgmlLowerDsv4Test, WeightedHcPreAdmissionAndBoundLaunchKeepTheirOwnGuards) {
  using llmp::format::Encoding;
  const auto* cap = kg::CapabilityOf(kg::kHcPreName);
  ASSERT_NE(cap, nullptr);
  const auto params = ir::ToBytes(ir::HcParams{.streams = 4});
  constexpr std::uint64_t limit = static_cast<std::uint64_t>(INT32_MAX) * 256 - 1;
  for (const std::uint64_t width : std::array<std::uint64_t, 3>{32, limit, limit + 1}) {
    auto arena = kg::TensorArena::Create(8);
    ASSERT_TRUE(arena);
    std::uint64_t address = 1ULL << 44;
    const auto bind = [&](ggml_tensor* t) {
      kg::TensorArena::Bind(t, address);
      address += 1ULL << 44;
      return t;
    };
    auto* x = bind(ggml_new_tensor_3d(arena->context(), GGML_TYPE_F32,
                                      static_cast<std::int64_t>(width), 4, 1));
    auto* weights = bind(ggml_new_tensor_2d(arena->context(), GGML_TYPE_F32, 4, 1));
    auto* out = bind(ggml_dsv4_hc_pre(arena->context(), x, weights));
    std::array<ex::OperandFormat, 3> operands{};
    const std::array tensors{x, weights, out};
    for (std::size_t i = 0; i < operands.size(); ++i) {
      auto& o = operands[i];
      o.operand = i == 2 ? ex::kOutputOperand : static_cast<std::uint8_t>(i);
      o.encoding = Encoding::kF32;
      o.memory = ex::OperandMemory{};
      for (std::size_t d = 0; d < 4; ++d) {
        o.ne[d] = static_cast<std::uint64_t>(tensors[i]->ne[d]);
        o.memory->nb[d] = tensors[i]->nb[d];
      }
    }
    const ex::CapabilityQuery query{
        .op = cap->op, .operands = operands, .arch = 121, .params = params};
    EXPECT_EQ(cap->accepts(query), width <= limit);
    EXPECT_EQ(kg::CheckHcPre(out).has_value(), width <= limit);
    if (width > limit) continue;
    const auto packed = operands;
    operands[0].memory.reset();
    EXPECT_FALSE(cap->accepts(query));
    operands = packed;
    operands[1].memory->view_offset = 2;
    EXPECT_FALSE(cap->accepts(query));
    operands = packed;
    operands[2].encoding = Encoding::kBF16;
    EXPECT_FALSE(cap->accepts(query));
    operands = packed;
    operands[0].memory->nb[1] += 4;
    EXPECT_FALSE(cap->accepts(query));
    const auto output_address = out->data;
    out->data = x->data;
    EXPECT_FALSE(kg::CheckHcPre(out));
    out->data = static_cast<std::byte*>(output_address) + 2;
    EXPECT_FALSE(kg::CheckHcPre(out));
    out->data = output_address;
    EXPECT_TRUE(kg::CheckHcPre(out));
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
    auto lowered = kg::LowerStep({*plan, step, shape, {}, 65536}, registry, Device());
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

TEST(GgmlLowerDsv4Test, ExactSelectionPredicatesAgreeWithActualLaunchBounds) {
  using llmp::format::Encoding;
  for (const std::uint64_t cells : {17ULL, 536870911ULL, 536870912ULL}) {
    for (const std::uint32_t top : {1U, 65536U, 65537U}) {
      auto arena = kg::TensorArena::Create(8);
      ASSERT_TRUE(arena);
      std::uint64_t address = 1ULL << 40;
      const auto bind = [&](ggml_tensor* t) {
        kg::TensorArena::Bind(t, address);
        address += 1ULL << 36;
        return t;
      };
      auto* scores = bind(
          ggml_new_tensor_2d(arena->context(), GGML_TYPE_F32, static_cast<std::int64_t>(cells), 2));
      auto* counts = bind(ggml_new_tensor_1d(arena->context(), GGML_TYPE_I32, 2));
      auto* selected = bind(kg::VisibleTopK(arena->context(), scores, counts, top));
      const auto* cap = kg::CapabilityOf(kg::kVisibleTopKName);
      ASSERT_NE(cap, nullptr);
      const auto params = ir::ToBytes(ir::TopKParams{.k = top});
      const std::array<ex::OperandFormat, 3> operands{
          {{.operand = 0, .encoding = Encoding::kF32, .ne = {cells, 2, 1, 1}},
           {.operand = 1, .encoding = Encoding::kI32, .ne = {2, 1, 1, 1}},
           {.operand = ex::kOutputOperand, .encoding = Encoding::kI32, .ne = {top, 2, 1, 1}}}};
      const ex::CapabilityQuery q{
          .op = cap->op, .operands = operands, .arch = 121, .params = params};
      const bool supported = cells <= 536870911ULL && top <= 65536;
      EXPECT_EQ(cap->accepts(q), supported);
      EXPECT_EQ(kg::CheckVisibleTopK(selected).has_value(), supported);
      if (!supported) continue;
      scores->nb[0] = 0;
      counts->nb[0] = 0;
      EXPECT_TRUE(kg::CheckVisibleTopK(selected));
      scores->nb[0] = 3;
      EXPECT_FALSE(kg::CheckVisibleTopK(selected));
      scores->nb[0] = 4;
      const auto saved = selected->data;
      selected->data = scores->data;
      EXPECT_FALSE(kg::CheckVisibleTopK(selected));
      selected->data = saved;
      selected->ne[0] = 0;
      EXPECT_FALSE(kg::CheckVisibleTopK(selected));
    }
  }
  for (const std::uint32_t cells : {268431352U, 268431353U}) {
    auto arena = kg::TensorArena::Create(8);
    ASSERT_TRUE(arena);
    auto* raw = ggml_new_tensor_2d(arena->context(), GGML_TYPE_F16, 5, 1);
    auto* ids = ggml_new_tensor_2d(arena->context(), GGML_TYPE_I32, 17, 1);
    auto* mask = kg::SelectedMask(arena->context(), raw, ids, 8, cells);
    kg::TensorArena::Bind(raw, 1ULL << 40);
    kg::TensorArena::Bind(ids, 2ULL << 40);
    kg::TensorArena::Bind(mask, 3ULL << 40);
    const auto* cap = kg::CapabilityOf(kg::kSelectedMaskName);
    ASSERT_NE(cap, nullptr);
    const auto params = ir::ToBytes(ir::SelectedMaskParams{.raw_cells = 8, .cells = cells});
    const std::array<ex::OperandFormat, 3> operands{
        {{.operand = 0, .encoding = Encoding::kF16, .ne = {5, 1, 1, 1}},
         {.operand = 1, .encoding = Encoding::kI32, .ne = {17, 1, 1, 1}},
         {.operand = ex::kOutputOperand,
          .encoding = Encoding::kF16,
          .ne = {8ULL + cells, 1, 1, 1}}}};
    const ex::CapabilityQuery q{.op = cap->op, .operands = operands, .arch = 121, .params = params};
    const bool supported = cells == 268431352;
    EXPECT_EQ(cap->accepts(q), supported);
    EXPECT_EQ(kg::CheckSelectedMask(mask).has_value(), supported);
    if (!supported) continue;
    ids->nb[0] = 0;
    EXPECT_TRUE(kg::CheckSelectedMask(mask));
    ids->nb[0] = 3;
    EXPECT_FALSE(kg::CheckSelectedMask(mask));
    ids->nb[0] = 4;
    mask->data = raw->data;
    EXPECT_FALSE(kg::CheckSelectedMask(mask));
  }
}

TEST(GgmlLowerDsv4Test, ExactPrimitivesLowerWithTypedOwnerCountsAndRefuseMalformedCounts) {
  using llmp::format::Encoding;
  ir::Builder b;
  co::StepContext context(
      b, {{.slot = 1, .rows = 2, .past = 2}, {.slot = 5, .first_row = 2, .rows = 3, .past = 7}},
      false);
  for (std::size_t owner = 0; owner < 2; ++owner) {
    const auto rows = context.segments()[owner].rows;
    auto counts =
        context.VisibleCells(owner, {.capacity = 256, .block = 4, .positions = true}, 17, 3);
    ASSERT_TRUE(counts);
    b.SetOwner(static_cast<std::uint8_t>(owner));
    auto scores = b.Fill(ir::ShapeOf({17, rows}), 1);
    auto raw = b.Fill(ir::ShapeOf({5, rows}), 0, Encoding::kF16);
    const auto ids = b.VisibleTopK(scores, *counts, 19);
    b.Output(ids);
    b.Output(b.SelectedMask(raw, ids, {.raw_cells = 8, .cells = 17}));
  }
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  co::StepGraph step{
      .graph = std::move(*graph), .inputs = context.TakeInputs(), .weights = {}, .state = {}};
  const pl::StepShape shape{
      .segments = {{.slot = 1, .rows = 2, .past = 2}, {.slot = 5, .rows = 3, .past = 7}}};
  auto registry = Registry();
  std::vector<ir::TensorId> staged;
  for (const auto& in : step.inputs) staged.push_back(in.tensor);
  auto plan = pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, {},
                       {.staged = staged, .written = kg::WrittenBytes, .shape = &shape});
  ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
  const auto lower = [&] {
    return kg::LowerStep({*plan, step, shape, {}, 1024}, registry, Device());
  };
  auto lowered = lower();
  ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
  ASSERT_EQ(lowered->inputs.size(), 2U);
  for (const auto& input : lowered->inputs) {
    EXPECT_EQ(input.kind, kn::InputKind::kVisibleCells);
    EXPECT_EQ(input.read, 17U);
    EXPECT_EQ(input.visible_capacity, 3U);
    EXPECT_EQ(input.bytes, shape.segments[input.segment].rows * 4);
  }
  int selected = 0, masked = 0;
  for (const auto* node : lowered->nodes) {
    if (kg::LlmpOpOf(node) == kg::LlmpOp::kVisibleTopK) {
      ++selected;
      EXPECT_TRUE(kg::CheckVisibleTopK(node));
    }
    if (kg::LlmpOpOf(node) == kg::LlmpOp::kSelectedMask) {
      ++masked;
      EXPECT_TRUE(kg::CheckSelectedMask(node));
    }
  }
  EXPECT_EQ(selected, 2);
  EXPECT_EQ(masked, 2);
  const auto saved = step.inputs[0];
  for (int bad = 0; bad < 4; ++bad) {
    step.inputs[0] = saved;
    switch (bad) {
      case 0:
        step.inputs[0].segment = 2;
        break;
      case 1:
        step.inputs[0].read = 0;
        break;
      case 2:
        step.inputs[0].visible_capacity = 257;
        break;
      case 3:
        step.inputs[0].cells.ring = true;
        break;
    }
    EXPECT_FALSE(lower()) << bad;
  }
}

TEST(GgmlLowerDsv4Test, ExactPrimitivesLowerRefusesCountsFromAnotherEqualRowOwner) {
  ir::Builder b;
  co::StepContext context(
      b, {{.slot = 1, .rows = 2, .past = 2}, {.slot = 5, .first_row = 2, .rows = 2, .past = 7}},
      false);
  auto counts = context.VisibleCells(0, {.capacity = 256, .block = 4, .positions = true}, 17, 3);
  ASSERT_TRUE(counts);
  b.SetOwner(0);
  const auto scores = b.Fill(ir::ShapeOf({17, 2}), 1);
  b.Output(b.VisibleTopK(scores, *counts, 2));
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  co::StepGraph step{
      .graph = std::move(*graph), .inputs = context.TakeInputs(), .weights = {}, .state = {}};
  const pl::StepShape shape{
      .segments = {{.slot = 1, .rows = 2, .past = 2}, {.slot = 5, .rows = 2, .past = 7}}};
  auto registry = Registry();
  const std::vector<ir::TensorId> staged{*counts};
  auto plan = pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, {},
                       {.staged = staged, .written = kg::WrittenBytes, .shape = &shape});
  ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
  const auto lower = [&] {
    return kg::LowerStep({*plan, step, shape, {}, 1024}, registry, Device());
  };
  ASSERT_TRUE(lower());
  ASSERT_EQ(step.inputs.size(), 1U);
  ASSERT_EQ(step.graph.tensor(*counts).segment, 0);
  const auto own_values = co::InputValues(step.inputs[0], shape, {});
  ASSERT_TRUE(own_values);
  step.inputs[0].segment = 1;
  const auto other_values = co::InputValues(step.inputs[0], shape, {});
  ASSERT_TRUE(other_values);
  EXPECT_NE(*own_values, *other_values) << "equal row counts must not authenticate another owner";
  auto refused = lower();
  ASSERT_FALSE(refused);
  EXPECT_EQ(refused.error().detail,
            "lowering onto GGML: visible counts disagree with their owner rows");
  step.inputs[0].segment = ir::kJoinedSegment;
  EXPECT_FALSE(lower());
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
    auto lowered = kg::LowerStep({*plan, *step, shape, {}, 65536}, f.registry, Device());
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
    std::size_t checked = 0, index_keys = 0;
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
        index_keys += item.contract.kind == md::StateKind::kIndex;
      }
    }
    EXPECT_GT(checked, 0U);
    EXPECT_GT(index_keys, 0U);
    EXPECT_TRUE(unaligned);  // the production cache begins inside a backing extent
  }
}

TEST(GgmlLowerDsv4Test, ExplicitWindowSparseRecipeSelectsWideWithoutChangingNeighbor) {
  using llmp::format::Encoding;
  auto registry = Registry();
  for (const std::uint32_t rows : {1U, 512U}) {
    ir::Builder b;
    b.SetOwner(0);
    const auto input = [&](Encoding encoding, std::initializer_list<std::uint64_t> shape) {
      return b.Input(ir::Dense(encoding, ir::ShapeOf(shape)));
    };
    const auto q = input(Encoding::kF32, {512, rows, 64});
    const auto kv = input(Encoding::kF16, {512, 768, 1});
    const auto mask = input(Encoding::kF16, {768, std::max(rows, 32U)});
    const auto sinks = input(Encoding::kF32, {64});
    for (const bool sparse : {false, true})
      b.Output(b.Attention(q, kv, kv, mask, sinks,
                           {.scale = 0.044194173F,
                            .sinks = true,
                            .accumulate = ir::Accumulate::kF32,
                            .sparse_any = sparse,
                            .n_kv_max = sparse ? 128U : 0U}));
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
    co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
    const std::vector staged{q, kv, mask, sinks};
    for (const auto tensor : staged)
      step.inputs.push_back({.tensor = tensor, .kind = co::InputKind::kFeatures, .segment = 0});
    const pl::StepShape shape{.segments = {{.rows = rows, .read_cells = 768, .outputs = 1}}};
    auto plan = pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, {},
                         {.staged = staged,
                          .written = kg::WrittenBytes,
                          .shape = &shape,
                          .activation_alignment = 256});
    ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
    ASSERT_EQ(plan->steps.size(), 2U);
    EXPECT_EQ(registry.at(plan->steps[0].implementation).name, kg::kFlashAttnMmaName);
    EXPECT_EQ(registry.at(plan->steps[1].implementation).name, kg::kFlashAttnMmaWideName);
    auto lowered = kg::LowerStep({*plan, step, shape, {}, 3072}, registry, Device());
    ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
    ASSERT_EQ(lowered->plan.steps.size(), 2U);
    for (std::size_t i = 0; i < 2; ++i) {
      EXPECT_EQ(lowered->plan.steps[i].implementation,
                i == 0 ? kg::kFlashAttnMmaName : kg::kFlashAttnMmaWideName);
      ASSERT_EQ(lowered->plan.steps[i].nodes.size(), 1U);
      const auto* node = lowered->plan.steps[i].nodes[0];
      EXPECT_EQ(node->op, GGML_OP_FLASH_ATTN_EXT);
      EXPECT_EQ(node->op_params[4], i == 0 ? 0 : 128);
      EXPECT_EQ(node->op_params[kg::kFlashAttnSparseParam], i == 0 ? 0 : 1);
      EXPECT_EQ(node->src[1]->ne[1], 768);
      EXPECT_TRUE(kg::CheckFlashAttnMma(node));
    }
  }
}

TEST(GgmlLowerDsv4Test, ClosedLidMaskLowersExactlyTwoNativeHelpersAndFundedIds) {
  using llmp::format::Encoding;
  auto registry = Registry();
  for (const std::uint32_t rows : {1U, 512U}) {
    for (const std::uint32_t cells : {256U, 768U}) {
      for (const bool direct : {false, true}) {
        SCOPED_TRACE(rows);
        SCOPED_TRACE(cells);
        SCOPED_TRACE(direct);
        ir::Builder b;
        b.SetOwner(0);
        const auto input = [&](Encoding encoding, std::initializer_list<std::uint64_t> shape) {
          return b.Input(ir::Dense(encoding, ir::ShapeOf(shape)));
        };
        const auto q = input(Encoding::kF32, {128, 64, rows});
        const auto k = input(Encoding::kF16, {128, 1, cells});
        const auto w = input(Encoding::kF32, {64, rows});
        const auto counts = input(Encoding::kI32, {rows});
        const auto raw = input(Encoding::kF16, {640, rows});
        const std::uint32_t top = std::min(512U, cells);
        ir::TensorId output;
        if (direct) {
          const std::array sources{q, k, w, counts, raw};
          output = b.Fused(ir::FusedKind::kDsv4LidMask, sources,
                           ir::LidMaskParams{.top = top, .raw_cells = 640, .cells = cells})[0];
        } else {
          const auto scores = b.RoundedIndexScore(q, k, w, counts);
          const auto ids = b.VisibleTopK(scores, counts, top);
          output = b.SelectedMask(raw, ids, {.raw_cells = 640, .cells = cells});
        }
        b.Output(output);
        auto graph = std::move(b).Finish();
        ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
        co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
        const std::vector staged{q, k, w, counts, raw};
        for (const auto tensor : staged)
          step.inputs.push_back({.tensor = tensor, .kind = co::InputKind::kFeatures, .segment = 0});
        const pl::StepShape shape{.segments = {{.rows = rows, .read_cells = 3072, .outputs = 1}}};
        pl::Policy policy;
        policy.enabled = {"dsv4_lid_mask"};
        auto plan =
            pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, policy,
                     {.staged = staged,
                      .written = kg::WrittenBytes,
                      .shape = &shape,
                      .activation_alignment = 256});
        ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
        ASSERT_EQ(plan->steps.size(), 1U);
        EXPECT_EQ(registry.at(plan->steps[0].implementation).name, "llmp.dsv4.lid_mask");
        ASSERT_EQ(plan->graph.node(plan->steps[0].nodes[0]).key,
                  ir::Fused(ir::FusedKind::kDsv4LidMask));
        auto lowered = kg::LowerStep({*plan, step, shape, {}, 3072}, registry, Device());
        ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
        ASSERT_EQ(lowered->plan.steps.size(), 2U);
        EXPECT_EQ(lowered->plan.steps[0].implementation, kg::kDsv4LidTopKName);
        EXPECT_EQ(lowered->plan.steps[1].implementation, kg::kDsv4SparseMaskName);
        ASSERT_EQ(lowered->plan.steps[0].nodes.size(), 1U);
        ASSERT_EQ(lowered->plan.steps[1].nodes.size(), 1U);
        const auto* ids = lowered->plan.steps[0].nodes[0];
        const auto* mask = lowered->plan.steps[1].nodes[0];
        EXPECT_EQ(kg::LlmpOpOf(ids), kg::LlmpOp::kDsv4LidTopK);
        EXPECT_EQ(kg::LlmpOpOf(mask), kg::LlmpOp::kDsv4SparseMask);
        EXPECT_EQ(mask->src[1], ids);
        EXPECT_TRUE(kg::CheckDsv4LidTopK(ids));
        EXPECT_TRUE(kg::CheckDsv4SparseMask(mask));
        const auto round = [](std::uint64_t bytes) { return (bytes + 255) / 256 * 256; };
        const auto mask_bytes = ggml_nbytes(mask);
        const auto ids_bytes = ggml_nbytes(ids);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ids->data),
                  reinterpret_cast<std::uintptr_t>(mask->data) + round(mask_bytes));
        EXPECT_EQ(kg::WrittenBytes(plan->graph, plan->graph.outputs()[0]),
                  round(mask_bytes) + round(ids_bytes));
        for (const auto* node : lowered->nodes) {
          EXPECT_NE(kg::LlmpOpOf(node), kg::LlmpOp::kRoundedIndexScore);
          EXPECT_NE(kg::LlmpOpOf(node), kg::LlmpOp::kVisibleTopK);
          EXPECT_NE(kg::LlmpOpOf(node), kg::LlmpOp::kSelectedMask);
        }
      }
    }
  }
}

TEST(GgmlLowerDsv4Test, RoundedIndexScoreRequiresAuthenticatedRootFactsAndExactNativeViews) {
  using llmp::format::Encoding;
  auto registry = Registry();
  for (int form = 0; form < 5; ++form) {
    SCOPED_TRACE(form);
    ir::Builder b;
    const auto q_root = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({130, 64, 3})));
    const auto k_root = b.Input(ir::Dense(Encoding::kF16, ir::ShapeOf({136, 1, 256})));
    const auto w = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({64, 3})));
    const auto counts = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({3})));
    const auto q = b.View(q_root, ir::ShapeOf({128, 64, 3}), {4, 520, 520 * 64, 520 * 64 * 3},
                          form == 2 ? 4 : 8);
    const auto k =
        b.View(k_root, ir::ShapeOf({128, 1, 256}), {2, 272, 272, 272 * 256}, form == 3 ? 2 : 16);
    const auto output = b.RoundedIndexScore(q, k, w, counts);
    b.Output(output);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph) << graph.error().detail;
    std::vector<pl::RootAlignmentFact> facts;
    for (std::uint32_t t = 0; t < graph->tensor_count(); ++t)
      if (!graph->tensor({t}).is_view()) facts.push_back({.root = {t}, .alignment = 256});
    if (form == 4) facts.back().alignment = 2;
    co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
    const std::vector staged{q_root, k_root, w, counts};
    for (const auto input : staged)
      step.inputs.push_back({.tensor = input, .kind = co::InputKind::kFeatures});
    const pl::StepShape shape{.segments = {{.rows = 3, .outputs = 3}}};
    const auto guarantee =
        form == 1 ? pl::RootAlignment{} : pl::GraphRootAlignments(step.graph, facts);
    auto plan = pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, {},
                         {.staged = staged,
                          .written = kg::WrittenBytes,
                          .shape = &shape,
                          .root_alignment = guarantee});
    ASSERT_TRUE(plan) << plan.error().detail;
    const auto native = std::ranges::count_if(plan->graph.nodes(), [](const auto& node) {
      return node.key == ir::Primitive(ir::Op::kRoundedIndexScore);
    });
    EXPECT_EQ(native, form == 0 ? 1 : 0);
    auto lowered = kg::LowerStep({*plan, step, shape, {}, 256}, registry, Device());
    ASSERT_TRUE(lowered) << lowered.error().detail;
    const auto custom = std::ranges::count_if(lowered->nodes, [](const auto* node) {
      return kg::LlmpOpOf(node) == kg::LlmpOp::kRoundedIndexScore;
    });
    EXPECT_EQ(custom, native);
    if (form != 0) continue;
    // The same logical plan cannot bind an unaligned physical arena.
    const kn::Addresses bad{.weight = {},
                            .weight_rows = {},
                            .state = {},
                            .save = {},
                            .activations = (std::uint64_t{1} << 44) + 4,
                            .activation_bytes = plan->placement.extent};
    auto rebound = kg::LowerStep({*plan, step, shape, bad, 256}, registry, Device());
    EXPECT_FALSE(rebound);
  }
}

TEST(GgmlLowerDsv4Test, RoundedIndexScoreCapabilityAndActualLaunchShareEveryGeometryBound) {
  using llmp::format::Encoding;
  auto arena = kg::TensorArena::Create(32);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  auto* q = ggml_new_tensor_3d(c, GGML_TYPE_F32, 128, 64, 3);
  auto* k = ggml_new_tensor_3d(c, GGML_TYPE_F16, 128, 1, 256);
  auto* w = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, 3);
  auto* counts = ggml_new_tensor_1d(c, GGML_TYPE_I32, 3);
  auto* output = kg::RoundedIndexScore(c, q, k, w, counts);
  const std::array tensors{q, k, w, counts, output};
  for (std::size_t i = 0; i < tensors.size(); ++i)
    kg::TensorArena::Bind(tensors[i], (std::uint64_t{i} + 1) << 44);
  const std::array<ggml_tensor, 5> original{*q, *k, *w, *counts, *output};
  const auto* cap = kg::CapabilityOf(kg::kRoundedIndexScoreName);
  ASSERT_NE(cap, nullptr);
  const auto params = ir::ToBytes(ir::RoundedIndexScoreParams{});
  for (int mutation = 0; mutation < 16; ++mutation) {
    SCOPED_TRACE(mutation);
    for (std::size_t i = 0; i < tensors.size(); ++i) *tensors[i] = original[i];
    switch (mutation) {
      case 1:
        q->nb[1] += 4;
        break;
      case 2:
        k->nb[2] += 2;
        break;
      case 3:
        w->nb[1] += 2;
        break;
      case 4:
        counts->nb[0] = 8;
        break;
      case 5:
        q->data = reinterpret_cast<void*>((std::uint64_t{1} << 44) + 4);
        break;
      case 6:
        k->data = reinterpret_cast<void*>((std::uint64_t{2} << 44) + 8);
        break;
      case 7:
        q->nb[1] = (std::uint64_t{INT32_MAX} / 63 + 2) * 4;
        break;
      case 8:
        q->nb[2] = (std::uint64_t{INT32_MAX} / 2 + 1) * 4;
        break;
      case 9:
        k->nb[2] = (std::uint64_t{INT32_MAX} / 255 + 8) * 2;
        break;
      case 10:
        w->nb[1] = (std::uint64_t{INT32_MAX} / 2 + 1) * 4;
        break;
      case 11:
        output->nb[1] += 4;
        break;
      case 12:
        q->nb[1] = q->nb[2] = k->nb[2] = w->nb[1] = 0;
        break;
      case 13:
      case 14:
        q->ne[2] = w->ne[1] = counts->ne[0] = output->ne[1] =
            std::int64_t{65535} * 4 + (mutation == 14 ? 1 : 0);
        q->nb[1] = q->nb[2] = w->nb[1] = 0;
        break;
      case 15:
        output->ne[0] += 1;
        break;
      default:
        break;
    }
    std::vector<ex::OperandFormat> operands;
    constexpr std::array encodings{Encoding::kF32, Encoding::kF16, Encoding::kF32, Encoding::kI32,
                                   Encoding::kF32};
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      const auto* t = tensors[i];
      ex::OperandFormat f{.operand = static_cast<std::uint8_t>(i == 4 ? ex::kOutputOperand : i),
                          .encoding = encodings[i]};
      f.memory = ex::OperandMemory{.view_offset = reinterpret_cast<std::uintptr_t>(t->data),
                                   .root_alignment = 256};
      for (std::size_t d = 0; d < 4; ++d) {
        f.ne[d] = static_cast<std::uint64_t>(t->ne[d]);
        f.memory->nb[d] = t->nb[d];
      }
      operands.push_back(f);
    }
    const ex::CapabilityQuery query{
        .op = cap->op, .operands = operands, .arch = 121, .params = params};
    const bool supported = mutation == 0 || mutation == 12 || mutation == 13;
    EXPECT_EQ(cap->accepts(query), supported);
    EXPECT_EQ(kg::CheckRoundedIndexScore(output).has_value(), supported);
    if (mutation == 0) {
      operands[1].memory.reset();
      EXPECT_FALSE(cap->accepts(query));
      output->data = k->data;
      EXPECT_FALSE(kg::CheckRoundedIndexScore(output));
    }
  }
}

TEST(GgmlLowerDsv4Test, ComponentPhysicalFactsAreSemanticAndBrokenBindingsRefuseEarly) {
  Fixture f;
  ASSERT_TRUE(f.Init());
  auto spec = f.spec;
  spec.stages.resize(1);
  spec.stages[0].layers.resize(1);
  spec.stages[0].taps.clear();
  auto layout = co::StageLayout(spec, 0, 65536, 512);
  ASSERT_TRUE(layout);
  const pl::StepShape shape{
      .segments = {{.slot = 3, .rows = 2, .past = 127, .read_cells = 256, .outputs = 2}}};
  for (int mode = 0; mode < 5; ++mode) {
    SCOPED_TRACE(mode);
    bool lowered = false;
    kn::Addresses addresses;
    addresses.activation_alignment = mode == 4 ? 3 : 2U << 20;
    addresses.state_alignment = [&](const ir::StateRef& ref) -> std::uint64_t {
      const auto at = std::ranges::find_if(layout->items, [&](const auto& item) {
        return item.layer == ref.layer && item.tensor == ref.tensor;
      });
      return at == layout->items.end() ? 0 : std::gcd(std::uint64_t{2} << 20, at->offset);
    };
    addresses.state = [&](const ir::StateRef& ref) -> std::uint64_t {
      const auto at = std::ranges::find_if(layout->items, [&](const auto& item) {
        return item.layer == ref.layer && item.tensor == ref.tensor;
      });
      return at == layout->items.end()
                 ? 0
                 : (std::uint64_t{1} << 40) + at->offset + (mode == 2 ? 4 : 0);
    };
    if (mode == 1 || mode == 3) {
      addresses.activations = (std::uint64_t{1} << 42) + (mode == 3 ? 4 : 0);
      addresses.activation_bytes = std::uint64_t{1} << 32;
    }
    auto source = en::ComponentStepSource::Create(
        {.spec = &spec,
         .bindings = {.tensors = f.target_source.tensors, .state = &*layout},
         .policy = pl::Policy{},
         .backend = {.registry = &f.registry,
                     .device = {.device_class = "sm_121", .arch = 121},
                     .written = kg::WrittenBytes,
                     .within = kg::WithinRoot,
                     .lower = [&](const kn::LowerInput& input)
                         -> std::expected<std::unique_ptr<kn::StepProgram>, std::string> {
                       lowered = true;
                       std::size_t states = 0, activations = 0;
                       for (const auto& fact : input.plan.root_alignments) {
                         const auto& desc = input.plan.graph.tensor(fact.root);
                         if (const auto* ref = std::get_if<ir::StateRef>(&desc.binding)) {
                           EXPECT_EQ(fact.alignment, addresses.state_alignment(*ref));
                           ++states;
                         } else {
                           EXPECT_EQ(fact.alignment, 256U);
                           ++activations;
                         }
                       }
                       EXPECT_GT(states, 0U);
                       EXPECT_GT(activations, 0U);
                       return std::unexpected("observed physical facts");
                     }}});
    ASSERT_TRUE(source) << source.error();
    const auto key = (*source)->KeyBase();
    auto built = (*source)->Build(shape, addresses);
    ASSERT_FALSE(built);
    EXPECT_EQ(lowered, mode < 2);
    EXPECT_EQ((*source)->KeyBase(), key);
    if (mode < 2)
      EXPECT_EQ(built.error(), "observed physical facts");
    else
      EXPECT_NE(built.error().find("alignment"), std::string::npos);
  }
}

TEST(GgmlLowerDsv4Test, ArenaFactsSelectRoundedScoreAndRejectBrokenPhysicalBase) {
  using llmp::format::Encoding;
  auto registry = Registry();
  ir::Builder b;
  const auto q = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({128, 64, 3})));
  const auto k = b.Input(ir::Dense(Encoding::kF16, ir::ShapeOf({128, 1, 256})));
  const auto w = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({64, 3})));
  const auto counts = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({3})));
  b.Output(b.RoundedIndexScore(q, k, w, counts));
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph);
  co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
  for (const auto t : {q, k, w, counts})
    step.inputs.push_back({.tensor = t, .kind = co::InputKind::kFeatures});
  const pl::StepShape shape{.segments = {{.rows = 3, .outputs = 3}}};
  for (const auto alignment : {0U, 256U}) {
    auto plan =
        pl::Plan(step.graph.Clone(), registry, {.device_class = "sm_121", .arch = 121}, {},
                 {.written = kg::WrittenBytes, .shape = &shape, .activation_alignment = alignment});
    ASSERT_TRUE(plan) << plan.error().detail;
    const auto native = std::ranges::count_if(
        plan->graph.nodes(), [](const auto& n) { return n.key.op == ir::Op::kRoundedIndexScore; });
    EXPECT_EQ(native, alignment == 0 ? 0 : 1);
    auto measured = kg::LowerStep({*plan, step, shape, {}, 256}, registry, Device());
    ASSERT_TRUE(measured) << measured.error().detail;
    if (alignment == 0) continue;
    for (const std::uint64_t offset : {0U, 4U}) {
      kn::Addresses addresses;
      addresses.activations = (std::uint64_t{1} << 44) + offset;
      addresses.activation_bytes = plan->placement.extent;
      auto bound = kg::LowerStep({*plan, step, shape, addresses, 256}, registry, Device());
      EXPECT_EQ(bound.has_value(), offset == 0);
      if (offset != 0 && !bound)
        EXPECT_NE(bound.error().detail.find("alignment promise"), std::string::npos);
    }
  }
}

TEST(GgmlLowerDsv4Test, DeepSeekServingPolicySelectsFastFormsAndDecodeThreshold) {
  Fixture f;
  f.target.swiglu_limit = 7.0f;
  f.target.swiglu_limit_shared = 7.0f;
  ASSERT_TRUE(f.Init());

  const auto policy = pl::ParsePolicy(sp::mappings::kDeepSeekServingPolicy);
  ASSERT_TRUE(policy) << policy.error();

  auto spec = f.spec;
  spec.stages.resize(1);
  spec.stages[0].taps.clear();

  const std::array<const md::StateLayoutDerived*, 2> layouts{&f.layouts[0], &f.layouts[1]};
  const std::array<std::span<const co::BoundTensor>, 2> tensors{f.target_source.tensors,
                                                                f.draft_source.tensors};
  const co::Bindings bindings{
      .tensors = tensors[0], .state = layouts[0], .stages = layouts, .artifacts = tensors};

  // A step with rows <= 16 (e.g. 8) lowers dsv4_hc_pre and vecq_glu with
  // VecQGlu::kSwigluClamp (limit = 7.0f).
  {
    const pl::StepShape shape{.segments = {{.rows = 8, .read_cells = 768, .outputs = 1}}};
    auto step = co::BuildStep(spec, 0, shape, bindings, {.narrow = false});
    ASSERT_TRUE(step) << step.error();

    std::vector<ir::TensorId> staged;
    for (const auto& input : step->inputs) {
      if (input.kind != co::InputKind::kCellPositions) staged.push_back(input.tensor);
    }

    auto plan = pl::Plan(step->graph.Clone(), f.registry, {.device_class = "sm_121", .arch = 121},
                         *policy, {.staged = staged, .written = kg::WrittenBytes, .shape = &shape});
    ASSERT_TRUE(plan) << plan.error().detail;

    const auto hc_pre_count = std::ranges::count_if(
        plan->applied, [](const auto& a) { return a.pattern == "dsv4_hc_pre"; });
    const auto vecq_glu_count =
        std::ranges::count_if(plan->applied, [](const auto& a) { return a.pattern == "vecq_glu"; });
    EXPECT_GT(hc_pre_count, 0);
    EXPECT_GT(vecq_glu_count, 0);

    auto lowered = kg::LowerStep({*plan, *step, shape, {}, 65536}, f.registry, Device());
    ASSERT_TRUE(lowered) << lowered.error().detail;

    bool found_hc_pre = false;
    bool found_vecq_clamp = false;
    for (const auto* node : lowered->nodes) {
      if (kg::LlmpOpOf(node) == kg::LlmpOp::kDsv4HcPre) {
        found_hc_pre = true;
      }
      if (kg::LlmpOpOf(node) == kg::LlmpOp::kVecQ) {
        const auto glu = static_cast<kg::VecQGlu>(kg::LlmpOpInt(node, 2));
        const float limit = kg::LlmpOpFloat(node, 3);
        if (glu == kg::VecQGlu::kSwigluClamp && limit == 7.0f) {
          found_vecq_clamp = true;
        }
      }
    }
    EXPECT_TRUE(found_hc_pre);
    EXPECT_TRUE(found_vecq_clamp);
  }

  // A step with rows = 130 lowers dsv4_hc_post instead of dsv4_hc_pre.
  {
    const pl::StepShape shape{
        .segments = {{.rows = 130, .past = 127, .read_cells = 768, .outputs = 1}}};
    auto step = co::BuildStep(spec, 0, shape, bindings, {.narrow = false});
    ASSERT_TRUE(step) << step.error();

    std::vector<ir::TensorId> staged;
    for (const auto& input : step->inputs) {
      if (input.kind != co::InputKind::kCellPositions) staged.push_back(input.tensor);
    }

    auto plan = pl::Plan(step->graph.Clone(), f.registry, {.device_class = "sm_121", .arch = 121},
                         *policy, {.staged = staged, .written = kg::WrittenBytes, .shape = &shape});
    ASSERT_TRUE(plan) << plan.error().detail;

    const auto hc_pre_count = std::ranges::count_if(
        plan->applied, [](const auto& a) { return a.pattern == "dsv4_hc_pre"; });
    const auto hc_post_count = std::ranges::count_if(
        plan->applied, [](const auto& a) { return a.pattern == "dsv4_hc_post"; });
    EXPECT_EQ(hc_pre_count, 0);
    EXPECT_GT(hc_post_count, 0);

    auto lowered = kg::LowerStep({*plan, *step, shape, {}, 65536}, f.registry, Device());
    ASSERT_TRUE(lowered) << lowered.error().detail;

    bool found_hc_pre = false;
    bool found_hc_post = false;
    for (const auto* node : lowered->nodes) {
      if (kg::LlmpOpOf(node) == kg::LlmpOp::kDsv4HcPre) {
        found_hc_pre = true;
      }
      if (node->op == GGML_OP_DSV4_HC_POST) {
        found_hc_post = true;
      }
    }
    EXPECT_FALSE(found_hc_pre);
    EXPECT_TRUE(found_hc_post);
  }
}

TEST(GgmlLowerDsv4Test, MultiSlotDecodeLowersSuccessfully) {
  Fixture f;
  f.target.swiglu_limit = 7.0f;
  f.target.swiglu_limit_shared = 7.0f;
  ASSERT_TRUE(f.Init());

  const auto policy = pl::ParsePolicy(sp::mappings::kDeepSeekServingPolicy);
  ASSERT_TRUE(policy) << policy.error();

  auto spec = f.spec;
  spec.stages.resize(1);
  spec.stages[0].taps.clear();

  const std::array<const md::StateLayoutDerived*, 2> layouts{&f.layouts[0], &f.layouts[1]};
  const std::array<std::span<const co::BoundTensor>, 2> tensors{f.target_source.tensors,
                                                                f.draft_source.tensors};
  const co::Bindings bindings{
      .tensors = tensors[0], .state = layouts[0], .stages = layouts, .artifacts = tensors};

  const pl::StepShape shape{.phase = pl::Phase::kDecode,
                            .segments =
                                {
                                    {.slot = 0,
                                     .rows = 1,
                                     .past = 255,
                                     .read_cells = 256,
                                     .outputs = 1,
                                     .want_head = true},
                                    {.slot = 1,
                                     .rows = 1,
                                     .past = 255,
                                     .read_cells = 256,
                                     .outputs = 1,
                                     .want_head = true},
                                    {.slot = 2,
                                     .rows = 1,
                                     .past = 255,
                                     .read_cells = 256,
                                     .outputs = 1,
                                     .want_head = true},
                                    {.slot = 3,
                                     .rows = 1,
                                     .past = 255,
                                     .read_cells = 256,
                                     .outputs = 1,
                                     .want_head = true},
                                },
                            .read_align = 256};
  auto step = co::BuildStep(spec, 0, shape, bindings, {.narrow = true, .dsv4_lid_mask = true});
  ASSERT_TRUE(step) << step.error();

  std::vector<ir::TensorId> staged;
  for (const auto& input : step->inputs) {
    if (input.kind != co::InputKind::kCellPositions) staged.push_back(input.tensor);
  }

  pl::LaneTags lane_tags;
  for (std::uint32_t i = 0; i < step->graph.nodes().size(); ++i) {
    const auto& node = step->graph.nodes()[i];
    if (node.batch == ir::Batch::kPerOwner && node.nout > 0) {
      const auto segment = step->graph.tensor(node.out[0]).segment;
      if (segment != ir::kJoinedSegment) {
        lane_tags.push_back(
            {ir::NodeId{i}, pl::LaneTag{
                                .lane = static_cast<std::uint8_t>((segment % pl::kMaxLanes) + 1),
                                .region = static_cast<std::uint32_t>(node.component + 1),
                            }});
      }
    }
  }

  auto plan =
      pl::Plan(step->graph.Clone(), f.registry, {.device_class = "sm_121", .arch = 121}, *policy,
               {.staged = staged,
                .lanes = std::move(lane_tags),
                .written = kg::WrittenBytes,
                .shape = &shape});
  ASSERT_TRUE(plan) << plan.error().detail;
  EXPECT_FALSE(plan->regions.empty());

  auto lowered = kg::LowerStep({*plan, *step, shape, {}, 65536}, f.registry, Device());
  ASSERT_TRUE(lowered) << lowered.error().detail;
  EXPECT_FALSE(lowered->plan.regions.empty());
}

TEST(GgmlLowerDsv4Test, Prefill4096LowersWithExpertFusions) {
  Fixture f;
  f.target.swiglu_limit = 7.0f;
  f.target.swiglu_limit_shared = 7.0f;
  ASSERT_TRUE(f.Init());

  const auto policy = pl::ParsePolicy(sp::mappings::kDeepSeekServingPolicy);
  ASSERT_TRUE(policy) << policy.error();

  auto spec = f.spec;
  spec.stages.resize(1);
  spec.stages[0].taps.clear();

  auto layout4096 = co::StageLayout(spec, 0, 65536, 4096);
  ASSERT_TRUE(layout4096) << layout4096.error();

  const std::array<const md::StateLayoutDerived*, 2> layouts{&*layout4096, &f.layouts[1]};
  const std::array<std::span<const co::BoundTensor>, 2> tensors{f.target_source.tensors,
                                                                f.draft_source.tensors};
  const co::Bindings bindings{
      .tensors = tensors[0], .state = layouts[0], .stages = layouts, .artifacts = tensors};

  const pl::StepShape shape{.phase = pl::Phase::kPrefill,
                            .segments =
                                {
                                    {.slot = 0,
                                     .rows = 4096,
                                     .past = 0,
                                     .read_cells = 4096,
                                     .outputs = 0,
                                     .want_head = false},
                                },
                            .read_align = 256};
  auto step = co::BuildStep(spec, 0, shape, bindings, {.narrow = true, .dsv4_lid_mask = true});
  ASSERT_TRUE(step) << step.error();

  std::vector<ir::TensorId> staged;
  for (const auto& input : step->inputs) {
    if (input.kind != co::InputKind::kCellPositions) staged.push_back(input.tensor);
  }

  auto plan = pl::Plan(step->graph.Clone(), f.registry, {.device_class = "sm_121", .arch = 121},
                       *policy, {.staged = staged, .written = kg::WrittenBytes, .shape = &shape});
  ASSERT_TRUE(plan) << plan.error().detail;

  auto lowered = kg::LowerStep({*plan, *step, shape, {}, 65536}, f.registry, Device());
  ASSERT_TRUE(lowered) << lowered.error().detail;

  bool found_pair_glu = false;
  bool found_combine = false;
  bool found_f16_qhead = false;
  bool found_outa = false;
  bool found_hc_post_norm_f16 = false;
  for (const auto& s : lowered->plan.steps) {
    if (s.implementation == kg::kMulMatIdQPairGluQ8 || s.implementation == kg::kMulMatIdQPairGlu) {
      found_pair_glu = true;
    }
    if (s.implementation == kg::kDsv4CombineName ||
        s.implementation == kg::kDsv4HcPostExpertsNormF16Name) {
      found_combine = true;
    }
    if (s.implementation == kg::kDsv4QHeadName) {
      if (!s.nodes.empty() && s.nodes[0]->type == GGML_TYPE_F16) {
        found_f16_qhead = true;
      }
    }
    if (s.implementation == kg::kDsv4OutAFastPackName || s.implementation == kg::kDsv4OutAName) {
      found_outa = true;
    }
    if (s.implementation == kg::kDsv4HcPostExpertsNormF16Name) {
      found_hc_post_norm_f16 = true;
    }
  }
  EXPECT_TRUE(found_pair_glu);
  EXPECT_TRUE(found_combine);
  EXPECT_TRUE(found_f16_qhead);
  EXPECT_TRUE(found_outa);
  EXPECT_TRUE(found_hc_post_norm_f16);
}

}  // namespace
