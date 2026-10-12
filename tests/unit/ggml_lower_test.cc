// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML lowering (kernels/ggml/lower.h) of the Gemma 2 and Gemma 3
// components' steps, structurally, on the CPU (every profile builds the
// planner and the lowering). For every shape the engine measures at setup
// at the serving defaults (Gemma 2 at 8,192 with two owners; Gemma 3 at
// 4,096 with two and 8,448 with one), and Gemma 3's joined decode of 2 to
// 12 owners with every count of long owners, with device masks and with
// host masks, the step planned under the architecture mapping's policy
// lowers: the lowered graph's leaves each bind a weight resource, a
// layer's, owner's and tensor's cache or a step input; the plan's steps
// cover its nodes; its owners' outputs and staged inputs are the shape's;
// and the policy's patterns (the owner-root attention among them) apply.
// (These graphs were once compared node for node with the M3 graph
// builders, retired with their runners after the migration's gate.) The
// planner selects over the GGML module's own capabilities
// (capabilities.h), on a model of the GB10's product routing that is
// upstream's (MmvqRouted).

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <map>
#include <optional>
#include <print>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "components/step.h"
#include "engine/support.h"
#include "execution/registry.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ggml.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/common/lower.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "model/gemma2.h"
#include "model/gemma3.h"
#include "planner/plan.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec_fixtures.h"

namespace {

namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace md = llmp::model;
namespace ex = llmp::execution;
namespace pl = llmp::planner;
namespace co = llmp::components;
namespace ir = llmp::ir;

constexpr std::uint64_t kExtent = 2ULL << 20U;
constexpr std::uint32_t kGb10 = 121;

// The GGML module's capabilities as a registry (placeholder identities: a
// CPU build has no kernels to bind).
ex::Registry PlanningRegistry() {
  std::vector<ex::Implementation> v;
  for (const kg::CapabilityDeclaration& d : kg::CapabilityDeclarations()) {
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

const pl::DeviceFacts& Gb10() {
  static const pl::DeviceFacts facts{.device_class = "sm_121", .arch = kGb10};
  return facts;
}

std::optional<llmp::format::Encoding> EncodingOf(ggml_type type) {
  for (const auto& info : llmp::format::Encodings()) {
    if (info.ggml && info.ggml->type == static_cast<std::uint32_t>(type)) return info.id;
  }
  return std::nullopt;
}

// The device's own choices on the GB10, as both sides take them: upstream's
// product routing (MMVQ where ggml_cuda_should_use_mmvq takes it, MMQ
// otherwise; MulMatVecQGluFusible's one column), no cuBLAS product.
kg::DeviceChoices Device() {
  kg::DeviceChoices d;
  d.mul_mat = [](const ggml_tensor*) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
    return kg::MulMatPath::kCublas;
  };
  d.quant = [](const ggml_tensor* n) -> std::expected<kg::QuantMulMatPath, kg::KernelFailure> {
    const auto e = EncodingOf(n->src[0]->type);
    return e && kg::MmvqRouted(*e, kGb10, static_cast<std::uint64_t>(n->src[1]->ne[1]))
               ? kg::QuantMulMatPath::kVector
               : kg::QuantMulMatPath::kTile;
  };
  d.quant_glu_fusible = [](const ggml_tensor*, const ggml_tensor* up, const ggml_tensor*) {
    const auto e = EncodingOf(up->src[0]->type);
    return up->ne[1] == 1 && e && kg::MmvqRouted(*e, kGb10, 1);
  };
  d.dense_mmvq_shape = [](ggml_type type, std::int64_t columns) {
    const auto e = EncodingOf(type);
    return columns >= 1 && columns <= 8 && e &&
           kg::MmvqRouted(*e, kGb10, static_cast<std::uint64_t>(columns));
  };
  return d;
}

// Each family: its mapping's policy, spec and source, and the cells its
// state holds (the K/V caches' global and ring cells, in closed form).
struct Cells {
  std::uint32_t global_cells = 0, local_cells = 0;
};
Cells CellsOf(std::uint32_t window, std::uint32_t context, std::uint32_t rows) {
  const auto pad = [](std::uint64_t v) {
    return static_cast<std::uint32_t>((v + 255) / 256 * 256);
  };
  return {pad(context), pad(std::min<std::uint64_t>(context, window + rows))};
}

struct Gemma2Family {
  static std::string_view Policy() {
    return llmp::spec::mappings::Gemma2Mapping().policies.front().text;
  }
  static const md::Gemma2Profile& P() { return md::Gemma2_2B(); }
  static auto Resources() { return llmp::test_support::gemma2::Resources(); }
  static Cells State(std::uint32_t context, std::uint32_t rows) {
    return CellsOf(P().window, context, rows);
  }
  static llmp::spec::ModelSpec Spec() {
    auto values = llmp::spec::mappings::Gemma2V0Values(P());
    return *llmp::spec::Synthesize(llmp::spec::mappings::Gemma2Mapping(),
                                   llmp::test_support::Gemma2Source(), *values);
  }
  static llmp::spec::Source Source() { return llmp::test_support::Gemma2Source(); }
  static std::vector<std::pair<std::uint32_t, std::uint32_t>> Contexts() { return {{8192, 2}}; }
  static std::vector<std::pair<std::uint32_t, std::uint32_t>> Cohorts() { return {}; }
};

struct Gemma3Family {
  static std::string_view Policy() {
    return llmp::spec::mappings::Gemma3Mapping().policies.front().text;
  }
  static const md::Gemma3Profile& P() { return md::Gemma3_4BQat(); }
  static auto Resources() { return llmp::test_support::gemma3::Resources(); }
  static Cells State(std::uint32_t context, std::uint32_t rows) {
    return CellsOf(P().window, context, rows);
  }
  static llmp::spec::ModelSpec Spec() {
    auto values = llmp::spec::mappings::Gemma3V0Values(P());
    return *llmp::spec::Synthesize(llmp::spec::mappings::Gemma3Mapping(),
                                   llmp::test_support::Gemma3Source(), *values);
  }
  static llmp::spec::Source Source() { return llmp::test_support::Gemma3Source(); }
  static std::vector<std::pair<std::uint32_t, std::uint32_t>> Contexts() {
    return {{4096, 2}, {8448, 1}};
  }
  // The joined decode beyond the serving envelope (the four-root
  // carriers): at 4,096 with twelve owners.
  static std::vector<std::pair<std::uint32_t, std::uint32_t>> Cohorts() { return {{4096, 12}}; }
};

// A step: per owner (slot, rows, past), and its outputs.
enum class Out : std::uint8_t { kHead, kAll, kGreedy, kStateOnly };
struct Case {
  std::vector<std::array<std::uint32_t, 3>> segments;
  Out out = Out::kHead;
  std::string Name() const {
    std::string s;
    for (const auto& [slot, rows, past] : segments)
      s += std::format("[{}:{}@{}]", slot, rows, past);
    return s + std::format(" out {}", static_cast<int>(out));
  }
};

// The shapes the engine measures at setup (with shared Q8's supplemental
// rows) with `slots` owners of 128 rows.
std::vector<Case> SetupCases(std::uint32_t context, std::uint32_t slots) {
  constexpr std::uint32_t kMaxRows = 128;
  std::vector<Case> out;
  const std::uint32_t wave_rows = kMaxRows * slots;
  const std::uint32_t head_rows = slots;
  for (const std::uint32_t budget : {wave_rows, head_rows}) {
    for (std::uint32_t count = 1; count <= slots; ++count) {
      for (const auto& rows :
           llmp::engine::support::ChunkMeasurementRows(kMaxRows, budget, count, true)) {
        const std::uint32_t most = *std::ranges::max_element(rows);
        for (const std::uint32_t past : {0U, context - most}) {
          for (const bool mixed : {false, true}) {
            if (mixed && (count != 2 || past != 0)) continue;
            Case c;
            for (std::uint32_t i = 0; i < count; ++i) {
              const std::uint32_t n = rows[i];
              c.segments.push_back({i, n, (past == 0 && (!mixed || i == 0)) ? 0U : context - n});
            }
            for (const Out o : {Out::kHead, Out::kAll, Out::kStateOnly, Out::kGreedy}) {
              if ((o == Out::kAll) != (budget == head_rows && budget != wave_rows)) continue;
              c.out = o;
              out.push_back(c);
            }
          }
        }
      }
    }
    if (head_rows == wave_rows) break;
  }
  return out;
}

// The joined one-row steps of 2 to `slots` owners measured at setup
// (every count of long owners, the last ones near the context's end),
// returning every owner's row, its token, or nothing.
std::vector<Case> CohortCases(std::uint32_t context, std::uint32_t slots) {
  std::vector<Case> out;
  for (std::uint32_t count = 2; count <= slots; ++count) {
    for (std::uint32_t long_count = 0; long_count < count; ++long_count) {
      Case c;
      for (std::uint32_t i = 0; i < count; ++i) {
        c.segments.push_back({i, 1, i < count - long_count ? 0U : context - 1});
      }
      for (const Out o : {Out::kHead, Out::kGreedy, Out::kStateOnly}) {
        c.out = o;
        out.push_back(c);
      }
    }
  }
  return out;
}

std::uint32_t ReadOf(const std::array<std::uint32_t, 3>& s, const Cells& state, bool local) {
  const std::uint32_t read = (s[1] + s[2] + 255) / 256 * 256;
  return std::min(read, local ? state.local_cells : state.global_cells);
}

pl::StepShape IrShape(const Case& c, const Cells& state) {
  pl::StepShape shape;
  for (const auto& s : c.segments) {
    shape.segments.push_back({.slot = static_cast<std::uint8_t>(s[0]),
                              .rows = s[1],
                              .past = s[2],
                              .read_cells = ReadOf(s, state, false),
                              .outputs = c.out == Out::kStateOnly ? 0U
                                         : c.out == Out::kAll     ? s[1]
                                                                  : 1U,
                              .want_head = c.out != Out::kStateOnly,
                              .greedy = c.out == Out::kGreedy});
  }
  return shape;
}

// ------------------------------------------------------------ the lowering

struct Ir {
  ex::Registry registry = PlanningRegistry();
  llmp::spec::ModelSpec spec;
  llmp::spec::Source source;
  std::optional<md::StateLayoutDerived> layout{};
};

struct Lowered {
  std::size_t owners = 0;  // owner_attention applied
  std::optional<co::StepGraph> step;
  std::optional<pl::StepPlan> plan;
  std::optional<kg::LoweredStep> lowered;
  std::map<const ggml_tensor*, std::string> leaves;
};

// The step's leaves by what they bind, as the builders' are named.
template <typename Resources>
std::map<std::uint32_t, std::string> StepLeaves(const co::StepGraph& step,
                                                const Resources& resources) {
  std::map<std::uint32_t, std::string> out;
  const auto where = [](const co::StepInput& in) {
    return std::format("{}.{}", in.segment, in.cells.ring ? "local" : "global");
  };
  std::map<std::uint32_t, const co::StepInput*> inputs;
  for (const co::StepInput& in : step.inputs) {
    inputs[in.tensor.v] = &in;
    switch (in.kind) {
      case co::InputKind::kTokens:
        out[in.tensor.v] = "tokens";
        break;
      case co::InputKind::kPositions:
        out[in.tensor.v] = "positions";
        break;
      case co::InputKind::kOutputRows:
        out[in.tensor.v] = "out_ids";
        break;
      case co::InputKind::kStoreCells:
        out[in.tensor.v] = "cells." + where(in);
        break;
      default:
        break;
    }
  }
  for (const co::StepState& st : step.state) {
    out[st.tensor.v] =
        std::format("{}.{}.{}", st.ref.tensor == 0 ? "k" : "v", st.ref.layer, st.ref.segment);
  }
  for (const co::StepWeight& w : step.weights) {
    const std::string& name = w.parts.front().first;
    const auto at = std::ranges::find_if(
        resources, [&](const auto& r) { return std::ranges::contains(r.roles, name); });
    out[w.tensor.v] = std::format("w.{}", static_cast<std::size_t>(at - resources.begin()));
  }
  for (const ir::Node& n : step.graph.nodes()) {
    if (n.key.op == ir::Op::kMaskBuild) {
      const auto held = inputs.find(n.src[1].v);
      if (held != inputs.end()) out[n.out[0].v] = "mask." + where(*held->second);
    }
  }
  return out;
}

template <typename F>
Lowered Lower(Ir& ir_, const Case& c, const Cells& state, bool device_masks,
              std::string_view policy_text) {
  Lowered out;
  const pl::StepShape shape = IrShape(c, state);
  auto step =
      co::BuildStep(ir_.spec, 0, shape, {.tensors = ir_.source.tensors, .state = &*ir_.layout});
  EXPECT_TRUE(step.has_value()) << step.error();
  if (!step) return out;
  std::vector<ir::TensorId> staged;
  for (const co::StepInput& in : step->inputs) {
    if (in.kind != co::InputKind::kCellPositions) staged.push_back(in.tensor);
  }
  const pl::Policy policy = pl::ParsePolicy(policy_text).value();
  auto plan = pl::Plan(
      step->graph.Clone(), ir_.registry, Gb10(), policy,
      {.capturable = true, .staged = staged, .written = &kg::WrittenBytes, .shape = &shape});
  EXPECT_TRUE(plan.has_value()) << (plan ? "" : plan.error().detail);
  if (!plan) return out;
  out.owners = static_cast<std::size_t>(
      std::ranges::count(plan->applied, "owner_attention", &pl::Applied::pattern));
  out.step.emplace(std::move(*step));
  out.plan.emplace(std::move(*plan));
  const kn::Addresses addresses{};  // a measurement: placeholder addresses
  const kn::LowerInput input{.plan = *out.plan,
                             .step = *out.step,
                             .shape = shape,
                             .addresses = addresses,
                             .context = ir_.layout->context};
  auto lowered = kg::LowerStep(input, ir_.registry, Device(), {.device_masks = device_masks});
  EXPECT_TRUE(lowered.has_value()) << (lowered ? "" : lowered.error().detail);
  if (!lowered) return out;
  const auto names = StepLeaves(*out.step, F::Resources());
  for (const kg::LoweredLeaf& leaf : lowered->leaves) {
    const auto at = names.find(leaf.primitive.v);
    out.leaves[leaf.tensor] = at == names.end() ? "?" : at->second;
  }
  out.lowered.emplace(std::move(*lowered));
  return out;
}

// ------------------------------------------------------------ checks

// Every leaf binds something the step names (a weight resource, a cache, an
// input), every plan step's nodes are the graph's own, and the plan covers
// every node that launches something.
void ExpectWellFormed(const Lowered& w) {
  const auto& nodes = w.lowered->nodes;
  std::set<const ggml_tensor*> mine(nodes.begin(), nodes.end());
  EXPECT_EQ(mine.size(), nodes.size());
  for (const auto& [tensor, name] : w.leaves) {
    EXPECT_NE(name, "?") << ggml_get_name(tensor);
    EXPECT_FALSE(mine.contains(tensor)) << name;
  }
  std::set<const ggml_tensor*> planned;
  for (const kg::PlanStep& step : w.lowered->plan.steps) {
    EXPECT_FALSE(step.nodes.empty());
    for (const ggml_tensor* node : step.nodes) {
      EXPECT_TRUE(mine.contains(node));
      planned.insert(node);
    }
  }
  // Every node that launches something is some step's.
  for (const ggml_tensor* node : nodes) {
    if (!kg::LaunchesNothing(node)) EXPECT_TRUE(planned.contains(node)) << ggml_op_desc(node);
  }
}

// The inputs the step stages: a mask only in the host-mask build, of its
// rows and cells; every other one a named input.
void ExpectInputs(const Lowered& w, bool device_masks) {
  std::map<std::uint64_t, const ggml_tensor*> by_address;
  for (const auto& [t, name] : w.leaves) {
    by_address[static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(t->data))] = t;
  }
  for (const kn::InputSlot& slot : w.lowered->inputs) {
    const auto found = by_address.find(slot.device_address);
    ASSERT_NE(found, by_address.end());
    EXPECT_NE(w.leaves.at(found->second), "?");
    if (slot.kind == kn::InputKind::kMask) {
      EXPECT_FALSE(device_masks);
      EXPECT_EQ(slot.bytes, slot.read * slot.mask_rows * 2);
    }
  }
}

template <typename F>
void ExpectFamilyLowers() {
  std::size_t shapes = 0, nodes = 0, steps = 0, owner_forms = 0;
  std::uint64_t biggest = 0;
  std::vector<std::tuple<std::uint32_t, std::uint32_t, bool>> runs;
  for (const auto& [context, slots] : F::Contexts()) runs.emplace_back(context, slots, false);
  for (const auto& [context, slots] : F::Cohorts()) runs.emplace_back(context, slots, true);
  for (const auto& [context, slots, cohorts] : runs) {
    const Cells state = F::State(context, 128);
    Ir ir_{.spec = F::Spec(), .source = F::Source()};
    auto layout = co::StageLayout(ir_.spec, 0, context, 128, kExtent);
    ASSERT_TRUE(layout.has_value()) << layout.error();
    ir_.layout.emplace(std::move(*layout));
    for (const Case& c : cohorts ? CohortCases(context, slots) : SetupCases(context, slots)) {
      for (const bool device_masks : {true, false}) {
        SCOPED_TRACE(
            std::format("context {} {}{}", context, c.Name(), device_masks ? "" : ", host masks"));
        const Lowered w = Lower<F>(ir_, c, state, device_masks, F::Policy());
        ASSERT_TRUE(w.lowered.has_value());
        ExpectWellFormed(w);
        if (::testing::Test::HasFatalFailure()) return;
        ExpectInputs(w, device_masks);
        // Each owner's own outputs, and nothing placed past the extent.
        std::uint32_t outputs = 0;
        for (const kn::OutputSlot& o : w.lowered->outputs) {
          EXPECT_LT(o.segment, c.segments.size());
          outputs += 1;
        }
        EXPECT_EQ(outputs, c.out == Out::kStateOnly ? 0U : c.segments.size());
        biggest = std::max(biggest, w.lowered->extent);
        owner_forms += w.owners;
        nodes += w.lowered->nodes.size();
        steps += w.lowered->plan.steps.size();
        ++shapes;
      }
    }
  }
  std::println(
      "{} lowered steps ({} with owner attention), {} nodes, {} plan steps, the largest "
      "activation extent {} bytes",
      shapes, owner_forms, nodes, steps, biggest);
  EXPECT_GT(shapes, 0U);
  EXPECT_GT(owner_forms, 0U);
}

TEST(GgmlLowerTest, Gemma2LowersEverySetupShape) { ExpectFamilyLowers<Gemma2Family>(); }

TEST(GgmlLowerTest, Gemma3LowersEverySetupShape) { ExpectFamilyLowers<Gemma3Family>(); }

// A late-built norm scale is a dependency of the selected fusion, although
// the original RMS primitive only needs the already-built activation.
TEST(GgmlLowerTest, FusedExpansionWaitsForComputedNormScale) {
  using llmp::format::Encoding;
  const auto registry = PlanningRegistry();
  for (const auto encoding : {Encoding::kF16, Encoding::kBF16}) {
    for (const std::uint32_t rows : {1U, 32U}) {
      SCOPED_TRACE(static_cast<int>(encoding));
      SCOPED_TRACE(rows);
      ir::Builder b;
      const auto x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({896, rows})));
      const auto weight = b.Input(ir::Dense(encoding, ir::ShapeOf({896})));
      const auto norm = b.RmsNorm(x, 1.0e-6F);
      const auto scale = b.Convert(weight, Encoding::kF32);
      b.Output(b.Mul(norm, scale));
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph.has_value()) << graph.error().detail;
      co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
      for (const auto tensor : {x, weight})
        step.inputs.push_back({.tensor = tensor, .kind = co::InputKind::kFeatures});
      const std::vector<ir::TensorId> staged{x, weight};
      auto plan = pl::Plan(step.graph.Clone(), registry, Gb10(), {},
                           {.staged = staged, .written = &kg::WrittenBytes});
      ASSERT_TRUE(plan.has_value()) << plan.error().detail;
      ASSERT_EQ(plan->steps.size(), 2U);
      EXPECT_EQ(registry.at(plan->steps[0].implementation).name, "ggml.convert");
      EXPECT_EQ(registry.at(plan->steps[1].implementation).name, kg::kRmsNormMulFused);
      const pl::StepShape shape{};
      const kn::Addresses addresses{};
      const kn::LowerInput input{
          .plan = *plan, .step = step, .shape = shape, .addresses = addresses};
      auto lowered = kg::LowerStep(input, registry, Device());
      ASSERT_TRUE(lowered.has_value()) << lowered.error().detail;
      ASSERT_EQ(lowered->plan.steps.size(), 2U);
      const auto& convert = lowered->plan.steps[0];
      const auto& fused = lowered->plan.steps[1];
      EXPECT_EQ(convert.implementation, "ggml.convert");
      EXPECT_EQ(fused.implementation, kg::kRmsNormMulFused);
      ASSERT_EQ(convert.nodes.size(), 1U);
      ASSERT_EQ(fused.nodes.size(), 2U);
      EXPECT_EQ(convert.nodes[0]->op, GGML_OP_CPY);
      EXPECT_EQ(fused.nodes[0]->op, GGML_OP_RMS_NORM);
      EXPECT_EQ(fused.nodes[1]->op, GGML_OP_MUL);
      EXPECT_EQ(fused.nodes[1]->src[0], fused.nodes[0]);
      EXPECT_EQ(fused.nodes[1]->src[1], convert.nodes[0]);
    }
  }
}

// Source logarithmic BF16 parameters stay external weight leaves. Their
// F32 cast and scalar arithmetic have ordinary paid activation placements.
TEST(GgmlLowerTest, Bf16LogParameterHasTransientPlacementAndStrictLowering) {
  using llmp::format::Encoding;
  const auto registry = PlanningRegistry();
  ir::Builder b;
  const ir::WeightRef ref{.stage = 0, .component = 0, .role = 0};
  // Structural CPU lowering only: own the source bytes through every bound
  // descriptor. Dense BF16 has one data part, not a multipart representation.
  alignas(256) std::array<std::uint16_t, 48> source{};
  static_assert(sizeof(source) == 96);
  const auto weight = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({48})), ref);
  const auto cast = b.Convert(weight, Encoding::kF32);
  const auto exponential = b.Unary(cast, ir::UnaryFn::kExp);
  const auto negative = b.Unary(exponential, ir::UnaryFn::kNeg);
  b.Output(negative);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph) << graph.error().detail;
  co::StepGraph step{.graph = std::move(*graph),
                     .inputs = {},
                     .weights = {{.tensor = weight,
                                  .ref = ref,
                                  .parts = {{"test.a_log", llmp::format::PartRole::kData}}}},
                     .state = {}};
  auto plan = pl::Plan(step.graph.Clone(), registry, Gb10(), {}, {.written = &kg::WrittenBytes});
  ASSERT_TRUE(plan) << plan.error().detail;
  ASSERT_EQ(plan->steps.size(), 3U);
  EXPECT_EQ(step.graph.tensor(weight).format.encoding, Encoding::kBF16);
  EXPECT_EQ(step.graph.tensor(weight).format.layout, llmp::format::Layout::kDense);
  const auto source_bytes = pl::RootBytes(step.graph.tensor(weight));
  ASSERT_TRUE(source_bytes);
  EXPECT_EQ(*source_bytes, 96U);
  const auto placed = [&](ir::TensorId original) {
    const auto mapped = plan->tensors.at(original.v);
    return std::ranges::find(plan->placement.offsets, mapped,
                             &std::pair<ir::TensorId, std::uint64_t>::first);
  };
  EXPECT_EQ(placed(weight), plan->placement.offsets.end());
  for (const auto tensor : {cast, exponential, negative}) {
    const auto at = placed(tensor);
    ASSERT_NE(at, plan->placement.offsets.end());
    EXPECT_LE(at->second + 48U * sizeof(float), plan->placement.extent);
  }
  const pl::StepShape shape{};
  const auto source_address =
      static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(source.data()));
  const kn::Addresses addresses{
      .weight =
          [&](const ir::WeightRef& actual, llmp::format::PartRole part) {
            EXPECT_EQ(actual, ref);
            EXPECT_EQ(part, llmp::format::PartRole::kData);
            return actual == ref && part == llmp::format::PartRole::kData ? source_address : 0;
          },
      .weight_rows = {},
      .state = {},
      .save = {}};
  const kn::LowerInput input{.plan = *plan, .step = step, .shape = shape, .addresses = addresses};
  const auto lowered = kg::LowerStep(input, registry, Device());
  ASSERT_TRUE(lowered) << lowered.error().detail;
  ASSERT_EQ(lowered->plan.steps.size(), 3U);
  EXPECT_EQ(lowered->plan.steps[0].implementation, "ggml.convert");
  EXPECT_EQ(lowered->plan.steps[1].implementation, "ggml.unary");
  EXPECT_EQ(lowered->plan.steps[2].implementation, "ggml.unary");
  ASSERT_EQ(lowered->plan.steps[0].nodes.size(), 1U);
  const auto* bound = lowered->plan.steps[0].nodes[0]->src[0];
  ASSERT_NE(bound, nullptr);
  EXPECT_EQ(bound->type, GGML_TYPE_BF16);
  EXPECT_EQ(bound->data, source.data());
  EXPECT_EQ(bound->op, GGML_OP_NONE);
  EXPECT_EQ(bound->view_src, nullptr);
  EXPECT_EQ(bound->ne[0], 48);
  for (std::size_t d = 1; d < 4; ++d) {
    EXPECT_EQ(bound->ne[d], 1);
    EXPECT_EQ(bound->nb[d], sizeof(source));
  }
  EXPECT_EQ(bound->nb[0], sizeof(source[0]));
  EXPECT_TRUE(ggml_is_contiguous(bound));
  EXPECT_EQ(ggml_nbytes(bound), sizeof(source));
  EXPECT_TRUE(lowered->measuring);
  EXPECT_EQ(lowered->plan.steps[0].nodes[0]->type, GGML_TYPE_F32);
  EXPECT_EQ(lowered->extent, plan->placement.extent);
}

TEST(GgmlLowerTest, Bf16NormOffsetHasTransientPlacementAndStrictLowering) {
  using llmp::format::Encoding;
  const auto registry = PlanningRegistry();
  ir::Builder b;
  const ir::WeightRef ref{.stage = 0, .component = 0, .role = 0};
  // Structural CPU lowering only: own the source bytes through every bound
  // descriptor. Dense BF16 has one data part, not a multipart representation.
  alignas(256) std::array<std::uint16_t, 48> source{};
  static_assert(sizeof(source) == 96);
  const auto weight = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({48})), ref);
  const auto cast = b.Convert(weight, Encoding::kF32);
  const auto offset = b.Scale(cast, 1.0F, 1.0F);
  b.Output(offset);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph) << graph.error().detail;
  co::StepGraph step{.graph = std::move(*graph),
                     .inputs = {},
                     .weights = {{.tensor = weight,
                                  .ref = ref,
                                  .parts = {{"test.raw_norm", llmp::format::PartRole::kData}}}},
                     .state = {}};
  auto plan = pl::Plan(step.graph.Clone(), registry, Gb10(), {}, {.written = &kg::WrittenBytes});
  ASSERT_TRUE(plan) << plan.error().detail;
  ASSERT_EQ(plan->steps.size(), 2U);
  EXPECT_EQ(step.graph.tensor(weight).format.encoding, Encoding::kBF16);
  EXPECT_EQ(step.graph.tensor(weight).format.layout, llmp::format::Layout::kDense);
  const auto source_bytes = pl::RootBytes(step.graph.tensor(weight));
  ASSERT_TRUE(source_bytes);
  EXPECT_EQ(*source_bytes, 96U);
  const auto placed = [&](ir::TensorId original) {
    const auto mapped = plan->tensors.at(original.v);
    return std::ranges::find(plan->placement.offsets, mapped,
                             &std::pair<ir::TensorId, std::uint64_t>::first);
  };
  EXPECT_EQ(placed(weight), plan->placement.offsets.end());
  for (const auto tensor : {cast, offset}) {
    const auto at = placed(tensor);
    ASSERT_NE(at, plan->placement.offsets.end());
    EXPECT_LE(at->second + 48U * sizeof(float), plan->placement.extent);
  }
  const pl::StepShape shape{};
  const auto source_address =
      static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(source.data()));
  const kn::Addresses addresses{
      .weight =
          [&](const ir::WeightRef& actual, llmp::format::PartRole part) {
            EXPECT_EQ(actual, ref);
            EXPECT_EQ(part, llmp::format::PartRole::kData);
            return actual == ref && part == llmp::format::PartRole::kData ? source_address : 0;
          },
      .weight_rows = {},
      .state = {},
      .save = {}};
  const kn::LowerInput input{.plan = *plan, .step = step, .shape = shape, .addresses = addresses};
  const auto lowered = kg::LowerStep(input, registry, Device());
  ASSERT_TRUE(lowered) << lowered.error().detail;
  ASSERT_EQ(lowered->plan.steps.size(), 2U);
  EXPECT_EQ(lowered->plan.steps[0].implementation, "ggml.convert");
  EXPECT_EQ(lowered->plan.steps[1].implementation, "ggml.scale");
  ASSERT_EQ(lowered->plan.steps[1].nodes.size(), 1U);
  const auto* shifted = lowered->plan.steps[1].nodes[0];
  EXPECT_EQ(shifted->type, GGML_TYPE_F32);
  EXPECT_EQ(shifted->op, GGML_OP_SCALE);
  EXPECT_EQ(shifted->src[0], lowered->plan.steps[0].nodes[0]);
  std::array<float, 2> scale_bias{};
  std::memcpy(scale_bias.data(), shifted->op_params, sizeof(scale_bias));
  EXPECT_EQ(scale_bias, (std::array<float, 2>{1.0F, 1.0F}));
  ASSERT_EQ(lowered->plan.steps[0].nodes.size(), 1U);
  const auto* bound = lowered->plan.steps[0].nodes[0]->src[0];
  ASSERT_NE(bound, nullptr);
  EXPECT_EQ(bound->type, GGML_TYPE_BF16);
  EXPECT_EQ(bound->data, source.data());
  EXPECT_EQ(bound->op, GGML_OP_NONE);
  EXPECT_EQ(bound->view_src, nullptr);
  EXPECT_EQ(bound->ne[0], 48);
  for (std::size_t d = 1; d < 4; ++d) {
    EXPECT_EQ(bound->ne[d], 1);
    EXPECT_EQ(bound->nb[d], sizeof(source));
  }
  EXPECT_EQ(bound->nb[0], sizeof(source[0]));
  EXPECT_TRUE(ggml_is_contiguous(bound));
  EXPECT_EQ(ggml_nbytes(bound), sizeof(source));
  EXPECT_TRUE(lowered->measuring);
  EXPECT_EQ(lowered->plan.steps[0].nodes[0]->type, GGML_TYPE_F32);
  EXPECT_EQ(lowered->extent, plan->placement.extent);
}

// RopeStore writes the cache, not its eliminated F32 rotation. Its descriptor
// must borrow the read operand, while a retained rotation still owns storage.
TEST(GgmlLowerTest, RopeStoreUsesReadOperandStandInWithoutAliasingRetainedRotation) {
  using llmp::format::Encoding;
  const auto registry = PlanningRegistry();
  for (const bool retained : {false, true}) {
    for (const bool measuring : {true, false}) {
      SCOPED_TRACE(std::format("retained={} measuring={}", retained, measuring));
      ir::Builder b;
      const auto root = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({64, 1, 4})));
      const auto x = b.View(root, ir::ShapeOf({64, 1, 3}), b.tensor(root).nb, 256);
      const auto pos = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({3})));
      const auto idx = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({3})));
      const ir::StateRef ref{.stage = 0, .tensor = 0, .layer = 0, .segment = 0};
      const auto cache = b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({64, 32})), ref);
      const auto target = b.View(cache, ir::ShapeOf({64, 16}), b.tensor(cache).nb, 128);
      const auto rope =
          b.Rope(x, pos, {}, {.mode = ir::RopeMode::kNeox, .dims = 64, .base = 10000.0F});
      const auto flat = b.Reshape(rope, ir::ShapeOf({64, 3}));
      const auto rows = b.View(flat, ir::ShapeOf({64, 3}), b.tensor(flat).nb, 0);
      b.Output(b.CacheStore(target, rows, idx));
      if (retained) b.Output(rope);
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
      co::StepGraph step{
          .graph = std::move(*graph),
          .inputs = {{.tensor = root, .kind = co::InputKind::kFeatures},
                     {.tensor = pos, .kind = co::InputKind::kPositions},
                     {.tensor = idx, .kind = co::InputKind::kStoreCells, .segment = 0}},
          .weights = {},
          .state = {{.tensor = cache, .ref = ref, .written = true}}};
      const std::vector<ir::TensorId> staged{root, pos, idx};
      auto plan = pl::Plan(step.graph.Clone(), registry, Gb10(), {},
                           {.staged = staged, .written = &kg::WrittenBytes});
      ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
      EXPECT_EQ(std::ranges::count(plan->applied, "rope_store", &pl::Applied::pattern),
                retained ? 0 : 1);
      alignas(256) std::array<std::byte, 16384> activation{};
      alignas(256) std::array<std::byte, 4096> state{};
      ASSERT_LE(plan->placement.extent, activation.size());
      const kn::Addresses addresses{
          .weight = {},
          .weight_rows = {},
          .state =
              [&](const ir::StateRef& actual) {
                EXPECT_EQ(actual, ref);
                return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(state.data()));
              },
          .save = {},
          .activations = measuring ? 0 : reinterpret_cast<std::uintptr_t>(activation.data()),
          .activation_bytes = activation.size()};
      const pl::StepShape shape{};
      const kn::LowerInput input{
          .plan = *plan, .step = step, .shape = shape, .addresses = addresses};
      auto lowered = kg::LowerStep(input, registry, Device());
      ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
      const auto rotation = std::ranges::find_if(
          lowered->nodes, [](const auto* node) { return node->op == GGML_OP_ROPE; });
      const auto store = std::ranges::find_if(
          lowered->nodes, [](const auto* node) { return node->op == GGML_OP_SET_ROWS; });
      ASSERT_NE(rotation, lowered->nodes.end());
      ASSERT_NE(store, lowered->nodes.end());
      EXPECT_EQ((*store)->data, state.data() + 128);
      EXPECT_EQ(reinterpret_cast<std::uintptr_t>((*rotation)->src[0]->data),
                reinterpret_cast<std::uintptr_t>((*rotation)->src[0]->view_src->data) + 256);
      EXPECT_NE((*rotation)->data, (*store)->data);
      if (retained) {
        EXPECT_NE((*rotation)->data, (*rotation)->src[0]->data);
        EXPECT_EQ(lowered->plan.steps.size(), 2U);
        EXPECT_TRUE(kg::CheckRope(*rotation));
        EXPECT_TRUE(kg::CheckSetRows(*store));
      } else {
        EXPECT_EQ((*rotation)->data, (*rotation)->src[0]->data);
        ASSERT_EQ(lowered->plan.steps.size(), 1U);
        EXPECT_EQ(lowered->plan.steps[0].implementation, kg::kRopeSetRowsFused);
        EXPECT_TRUE(kg::CheckRopeSetRows(*rotation, *store));
      }
      EXPECT_EQ(lowered->extent, plan->placement.extent);
    }
  }
}

// A selection PlanGraph would not make is refused, never run otherwise:
// without the norm-mul pattern the IR selects the norm and its mul alone
// (which unfused PlanGraph runs as one step of both launchers, the IR's two
// steps), but on a device whose own choices fuse norms PlanGraph runs the
// two as the fused RMSNorm-mul.
TEST(GgmlLowerTest, RefusesASelectionPlanGraphDoesNotMake) {
  Ir ir_{.spec = Gemma2Family::Spec(), .source = Gemma2Family::Source()};
  const Cells state = Gemma2Family::State(8192, 128);
  auto layout = co::StageLayout(ir_.spec, 0, 8192, 128, kExtent);
  ASSERT_TRUE(layout.has_value());
  ir_.layout.emplace(std::move(*layout));
  const Case c{.segments = {{0, 1, 7}}, .out = Out::kHead};
  const pl::StepShape shape = IrShape(c, state);
  auto step =
      co::BuildStep(ir_.spec, 0, shape, {.tensors = ir_.source.tensors, .state = &*ir_.layout});
  ASSERT_TRUE(step.has_value());
  std::vector<ir::TensorId> staged;
  for (const co::StepInput& in : step->inputs) {
    if (in.kind != co::InputKind::kCellPositions) staged.push_back(in.tensor);
  }
  const pl::Policy policy = pl::ParsePolicy(
                                "max_decode_rows 16\ndisable grouped_stores\ndisable "
                                "rms_norm_mul\ndisable rms_norm_mul_rope\ndisable rope_store\n")
                                .value();
  auto plan = pl::Plan(step->graph.Clone(), ir_.registry, Gb10(), policy,
                       {.staged = staged, .written = &kg::WrittenBytes});
  ASSERT_TRUE(plan.has_value());
  const kn::Addresses addresses{};
  const kn::LowerInput input{
      .plan = *plan, .step = *step, .shape = shape, .addresses = addresses, .context = 8192};
  // Unfused, the two run as PlanGraph's one step of both launchers.
  ASSERT_TRUE(kg::LowerStep(input, ir_.registry, Device()).has_value());
  kg::DeviceChoices fusing = Device();
  fusing.fuse_norms = true;
  auto lowered = kg::LowerStep(input, ir_.registry, fusing);
  ASSERT_FALSE(lowered.has_value());
  EXPECT_NE(lowered.error().detail.find("PlanGraph's step"), std::string::npos)
      << lowered.error().detail;
}

// Upstream's routing as the capabilities model it (mmvq.cu
// ggml_cuda_should_use_mmvq at the pin): the GB10's bound for its types,
// the RTX 3080 Ti's (sm_86) default, Ada's k-quant bounds, and nothing for
// an unquantized type.
TEST(GgmlLowerTest, CapabilitiesRouteAsUpstream) {
  using llmp::format::Encoding;
  EXPECT_TRUE(kg::MmvqRouted(Encoding::kQ4_0, 121, 8));
  EXPECT_FALSE(kg::MmvqRouted(Encoding::kQ4_0, 121, 9));
  EXPECT_TRUE(kg::MmvqRouted(Encoding::kQ2_K, 121, 6));
  EXPECT_FALSE(kg::MmvqRouted(Encoding::kQ2_K, 121, 7));
  EXPECT_TRUE(kg::MmvqRouted(Encoding::kQ8_0, 86, 8));
  EXPECT_FALSE(kg::MmvqRouted(Encoding::kQ2_K, 89, 5));
  EXPECT_TRUE(kg::MmvqRouted(Encoding::kQ3_K, 89, 6));
  EXPECT_FALSE(kg::MmvqRouted(Encoding::kF16, 121, 1));
  // The fused quantized GLU takes one column of gate and up weights of one
  // type (MulMatVecQGluFusible), so a mixed pair stays two products.
  const ex::Capability* glu = kg::CapabilityOf("ggml.mul_mat_geglu.mmvq_fused");
  ASSERT_NE(glu, nullptr);
  ASSERT_NE(glu->accepts, nullptr);
  const ir::GluParams geglu{.kind = ir::GluKind::kGeGluTanh};
  using llmp::format::Layout;
  std::array<ex::OperandFormat, 3> operands{
      {{.operand = 0, .encoding = Encoding::kQ4_0, .layout = Layout::kGgmlBlocks},
       {.operand = 1, .encoding = Encoding::kQ4_0, .layout = Layout::kGgmlBlocks},
       {.operand = 2, .encoding = Encoding::kF32, .layout = Layout::kDense}}};
  ex::CapabilityQuery query{.op = glu->op,
                            .operands = operands,
                            .rows = 1,
                            .k = 2560,
                            .n = 10240,
                            .arch = 121,
                            .params = std::as_bytes(std::span(&geglu, 1))};
  EXPECT_TRUE(glu->accepts(query));
  query.rows = 2;
  EXPECT_FALSE(glu->accepts(query));
  query.rows = 1;
  operands[1].encoding = Encoding::kQ6_K;
  EXPECT_FALSE(glu->accepts(query));
  // Every declared capability is a capability of its implementation's IR
  // key and, but for the module's own families, stands alone.
  for (const kg::CapabilityDeclaration& d : kg::CapabilityDeclarations()) {
    ASSERT_NE(d.capability, nullptr) << d.name;
    EXPECT_EQ(kg::CapabilityOf(d.name), d.capability) << d.name;
    EXPECT_FALSE(kg::AcceptsTextOf(d.name).empty()) << d.name;
    EXPECT_TRUE(d.capability->group.empty() || d.capability->group == "ggml.mul_mat" ||
                d.capability->group == "ggml.mul_mat_id" ||
                d.capability->group == "ggml.flash_attn_ext" ||
                d.capability->group == "ggml.gated_delta_net")
        << d.name;
  }
}

}  // namespace
