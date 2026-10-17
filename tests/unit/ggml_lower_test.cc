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
#include <limits>
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

#include "components/state_encoding.h"
#include "components/step.h"
#include "engine/support.h"
#include "execution/registry.h"
#include "families/gemma.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ggml.h"
#include "ir/builder.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/common/lower.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/lower.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "model/state_contract.h"
#include "planner/plan.h"
#include "planner/policy.h"
#include "planner/select.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec_fixtures.h"

namespace {

namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace md = llmp::model;
namespace fam = llmp::test_support::families;
namespace ex = llmp::execution;
namespace pl = llmp::planner;
namespace co = llmp::components;
namespace ir = llmp::ir;

constexpr std::uint64_t kExtent = 2ULL << 20U;
constexpr std::uint32_t kGb10 = 121;

// The GGML module's capabilities as a registry (placeholder identities: a
// CPU build has no kernels to bind).
ex::Registry PlanningRegistry(bool hc_post = true, bool grouped = true,
                              std::string_view missing = {}) {
  std::vector<ex::Implementation> v;
  for (const kg::CapabilityDeclaration& d : kg::CapabilityDeclarations()) {
    if (!hc_post && d.name == kg::kHcPostName) continue;
    if (!grouped && d.name == "llmp.vecq.grouped") continue;
    if (d.name == missing) continue;
    v.push_back({.name = std::string(d.name),
                 .operation = d.operation,
                 .source = "ggml",
                 .revision = "planning",
                 .build = "cpu",
                 .variant = std::string(kg::AcceptsTextOf(d.name)),
                 .capability = *d.capability});
  }
  // VecQ's executable primitive has no standalone IR capability. The
  // planning registry still needs its availability for composite recipes.
  if (missing != kg::kVecQName)
    v.push_back({.name = std::string(kg::kVecQName),
                 .operation = ex::Operation::kMatMul,
                 .source = "ggml",
                 .revision = "planning",
                 .build = "cpu",
                 .variant = "native VecQ launch"});
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
  static const fam::Gemma2Profile& P() { return fam::Gemma2_2B(); }
  static auto Resources() { return llmp::test_support::gemma2::Resources(); }
  static Cells State(std::uint32_t context, std::uint32_t rows) {
    return CellsOf(P().window, context, rows);
  }
  static llmp::spec::ModelSpec Spec() {
    auto values = fam::Gemma2Values(P());
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
  static const fam::Gemma3Profile& P() { return fam::Gemma3_4BQat(); }
  static auto Resources() { return llmp::test_support::gemma3::Resources(); }
  static Cells State(std::uint32_t context, std::uint32_t rows) {
    return CellsOf(P().window, context, rows);
  }
  static llmp::spec::ModelSpec Spec() {
    auto values = fam::Gemma3Values(P());
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
  const kn::LowerInput input{*out.plan, *out.step, shape, addresses, ir_.layout->context};
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
    auto layout = co::StageLayoutUniform(ir_.spec, 0, context, 128, kExtent);
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

TEST(GgmlLowerTest, GroupedVecQRejectsUnknownPhysicalAndArithmeticContracts) {
  const auto registry = PlanningRegistry();
  const pl::Policy policy{};
  const pl::SelectionView selection(registry, Gb10(), policy, false, false);
  const auto dense = [](std::initializer_list<std::uint64_t> shape) {
    ir::TensorDesc t{.format = ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf(shape))};
    t.nb = ir::NaturalStrides(t.format);
    return t;
  };
  ir::TensorDesc weight{.format = {.encoding = llmp::format::Encoding::kQ8_0,
                                   .layout = llmp::format::Layout::kGgmlBlocks,
                                   .shape = ir::ShapeOf({64, 16, 3})}};
  weight.nb = ir::NaturalStrides(weight.format);
  auto input = dense({64, 3, 3}), output = dense({16, 3, 3});
  ir::ProductParams params{};
  const auto accepts = [&] {
    const std::array<const ir::TensorDesc*, 2> sources{&weight, &input};
    const std::array<const ir::TensorDesc*, 1> outputs{&output};
    return !selection
                .Candidates(ir::Fused(ir::FusedKind::kGroupedMatMul), sources, outputs,
                            ir::ToBytes(params))
                .empty();
  };
  ASSERT_TRUE(accepts());
  output.format.encoding = llmp::format::Encoding::kBF16;
  EXPECT_FALSE(accepts());
  output = dense({16, 3, 3});
  params.accumulate = ir::Accumulate::kF32;
  EXPECT_FALSE(accepts());
  params = {};
  weight.nb[2] += 34;
  EXPECT_FALSE(accepts());
  weight.nb = ir::NaturalStrides(weight.format);
  input.view_offset = 4;
  EXPECT_FALSE(accepts());
  input.view_offset = 0;
  input = dense({64, 3, 17});
  output = dense({16, 3, 17});
  EXPECT_FALSE(accepts());
  weight.format.shape = ir::ShapeOf({64, 16, 257});
  weight.nb = ir::NaturalStrides(weight.format);
  input = dense({64, 257, 1});
  output = dense({16, 257, 1});
  EXPECT_FALSE(accepts());
  weight.format.encoding = llmp::format::Encoding::kF32;
  weight.format.layout = llmp::format::Layout::kDense;
  weight.format.shape = ir::ShapeOf({64, 16, 3});
  weight.nb = ir::NaturalStrides(weight.format);
  input = dense({64, 3, 3});
  output = dense({16, 3, 3});
  EXPECT_FALSE(accepts());
}

TEST(GgmlLowerTest, GroupedVecQWorkspaceRejectsOutOfDomainBeforeByteArithmetic) {
  ir::Builder b;
  const auto weight = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({64, 16, 257})));
  const auto input = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({64, 257, 1})));
  const auto output =
      b.Fused(ir::FusedKind::kGroupedMatMul, std::array{weight, input}, ir::ProductParams{})[0];
  b.Output(output);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph);
  EXPECT_EQ(kg::WrittenBytes(*graph, output), UINT64_MAX);
}

// The vector forms' row bound is the policy's own vector_rows (a mapping's
// step rule sets it), never read from the decode-shaped bound's value: a
// VecQ product of 12 rows under max_decode_rows 8 alone remains one, under
// vector_rows 8 is not; past 16 rows, the kernel's own bound.
TEST(GgmlLowerTest, VectorRowsBoundTheVectorForms) {
  if (!kg::CompiledFor(kGb10)) GTEST_SKIP() << "no GB10 kernels in this build";
  const auto registry = PlanningRegistry();
  const auto vecq = [&](std::uint64_t rows, ex::SelectionPolicy policy) {
    const std::array operands{ex::OperandFormat{.operand = 0,
                                                .encoding = llmp::format::Encoding::kQ8_0,
                                                .layout = llmp::format::Layout::kGgmlBlocks,
                                                .ne = {64, 32, 1, 1}},
                              ex::OperandFormat{.operand = 1,
                                                .encoding = llmp::format::Encoding::kQ8_1,
                                                .layout = llmp::format::Layout::kGgmlBlocks,
                                                .ne = {64, rows, 1, 1}}};
    const ex::CapabilityQuery query{.op = ir::Primitive(ir::Op::kMatMul),
                                    .operands = operands,
                                    .rows = rows,
                                    .k = 64,
                                    .n = 32,
                                    .arch = kGb10,
                                    .policy = policy};
    return std::ranges::any_of(ex::Candidates(registry, query), [&](std::size_t i) {
      return registry.at(i).name == "llmp.vecq.dense";
    });
  };
  EXPECT_TRUE(vecq(12, {}));
  EXPECT_TRUE(vecq(12, {.max_decode_rows = 8}));
  EXPECT_FALSE(vecq(12, {.max_decode_rows = 8, .vector_rows = 8}));
  EXPECT_FALSE(vecq(12, {.vector_rows = 8}));
  EXPECT_TRUE(vecq(8, {.max_decode_rows = 8, .vector_rows = 8}));
  EXPECT_TRUE(vecq(16, {.vector_rows = 32}));
  EXPECT_FALSE(vecq(17, {}));
}

TEST(GgmlLowerTest, GroupedVecQRequiresClosedSourceRecipeAndPaysBothLaunches) {
  // 0: native, 1: disabled, 2: missing implementation, 3: input weight,
  // 4: a weight view, 5: excess tokens, 6: wrong permutation despite equal
  // dimensions, 7: another reader of the product, 8: one group.
  for (unsigned scenario = 0; scenario < 12; ++scenario) {
    SCOPED_TRACE(scenario);
    const std::uint64_t k = 64, n = 16;
    const std::uint64_t groups = scenario == 8 ? 1 : 3;
    const std::uint32_t tokens = scenario == 5 ? 17U : 3U;
    const auto registry = PlanningRegistry(true, scenario != 2,
                                           scenario == 9    ? kg::kQuantizeQ8Name
                                           : scenario == 10 ? kg::kVecQName
                                                            : std::string_view{});
    ir::Builder b;
    const llmp::format::TensorFormat format{.encoding = llmp::format::Encoding::kQ8_0,
                                            .layout = llmp::format::Layout::kGgmlBlocks,
                                            .shape = ir::ShapeOf({k, n * groups})};
    const auto leaf =
        scenario == 3 ? b.Input(format) : b.Weight(format, {.expert_array = scenario == 11});
    auto weight = leaf;
    if (scenario == 4) weight = b.View(leaf, format.shape, ir::NaturalStrides(format), 0);
    weight = b.Reshape(weight, ir::ShapeOf({k, n, groups}));
    const auto input =
        b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({k, groups, tokens})));
    const auto shuffled = b.Permute(input, scenario == 6 ? std::array<std::uint8_t, 4>{0, 1, 2, 3}
                                                         : std::array<std::uint8_t, 4>{0, 2, 1, 3});
    const auto product = b.MatMul(weight, shuffled);
    if (scenario == 7) b.Output(product);
    b.Output(b.Cont(b.Permute(product, {0, 2, 1, 3}), ir::ShapeOf({n * groups, tokens})));
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph);
    co::StepGraph step{.graph = std::move(*graph),
                       .inputs = {{.tensor = input, .kind = co::InputKind::kFeatures}},
                       .weights = {},
                       .state = {}};
    if (scenario != 3)
      step.weights.push_back(
          {.tensor = leaf, .ref = {}, .parts = {{"weight", llmp::format::PartRole::kData}}});
    pl::Policy policy;
    if (scenario != 1) policy.enabled = {"grouped_vecq"};
    pl::Canonicalize(policy);
    std::vector<ir::TensorId> staged{input};
    if (scenario == 3) staged.push_back(leaf);
    const pl::StepShape shape{.segments = {{.rows = tokens, .outputs = 0, .want_head = false}}};
    auto plan = pl::Plan(step.graph.Clone(), registry, Gb10(), policy,
                         {.staged = staged, .written = kg::WrittenBytes});
    ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
    const bool native = scenario == 0;
    EXPECT_EQ(std::ranges::count_if(plan->graph.nodes(),
                                    [](const auto& node) {
                                      return node.key == ir::Fused(ir::FusedKind::kGroupedMatMul);
                                    }),
              native ? 1 : 0);
    if (!native) continue;
    for (const auto& node : plan->graph.nodes())
      if (node.key == ir::Fused(ir::FusedKind::kGroupedMatMul)) {
        const auto bytes = kg::WrittenBytes(plan->graph, node.out[0]);
        EXPECT_GT(bytes, n * groups * tokens * sizeof(float));
        EXPECT_GE(plan->placement.extent, bytes);
      }
    const kn::Addresses addresses{};
    auto lowered = kg::LowerStep({*plan, step, shape, addresses}, registry, Device());
    ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
    ASSERT_EQ(lowered->plan.steps.size(), 3U);
    EXPECT_EQ(lowered->plan.steps[0].implementation, kg::kQuantizeQ8Name);
    EXPECT_EQ(lowered->plan.steps[1].implementation, kg::kVecQName);
    EXPECT_EQ(lowered->plan.steps[2].implementation, kg::kContName);
  }
}

TEST(GgmlLowerTest, DirectGroupedVecQRequiresPolicySourceAuthorityAndBothLaunches) {
  // Native direct2D reshape, default-off, quantized Input, direct3D Weight,
  // missing quantizer, missing VecQ, and expert-array WeightRef.
  for (unsigned scenario = 0; scenario < 7; ++scenario) {
    SCOPED_TRACE(scenario);
    const auto registry = PlanningRegistry(true, true,
                                           scenario == 4   ? kg::kQuantizeQ8Name
                                           : scenario == 5 ? kg::kVecQName
                                                           : std::string_view{});
    ir::Builder b;
    llmp::format::TensorFormat format{
        .encoding = llmp::format::Encoding::kQ8_0,
        .layout = llmp::format::Layout::kGgmlBlocks,
        .shape = scenario == 3 ? ir::ShapeOf({64, 16, 3}) : ir::ShapeOf({64, 48})};
    const auto leaf =
        scenario == 2 ? b.Input(format) : b.Weight(format, {.expert_array = scenario == 6});
    const auto weight = scenario == 3 ? leaf : b.Reshape(leaf, ir::ShapeOf({64, 16, 3}));
    const auto input = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({64, 3, 3})));
    b.Output(
        b.Fused(ir::FusedKind::kGroupedMatMul, std::array{weight, input}, ir::ProductParams{})[0]);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph);
    pl::Policy policy;
    if (scenario != 1) policy.enabled = {"grouped_vecq"};
    pl::Canonicalize(policy);
    std::vector<ir::TensorId> staged{input};
    if (scenario == 2) staged.push_back(leaf);
    auto plan = pl::Plan(std::move(*graph), registry, Gb10(), policy,
                         {.staged = staged, .written = kg::WrittenBytes});
    ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
    EXPECT_EQ(std::ranges::count_if(plan->graph.nodes(),
                                    [](const auto& node) {
                                      return node.key == ir::Fused(ir::FusedKind::kGroupedMatMul);
                                    }),
              scenario == 0 ? 1 : 0);
    if (scenario != 0) {
      EXPECT_GT(std::ranges::count_if(
                    plan->graph.nodes(),
                    [](const auto& node) { return node.key == ir::Primitive(ir::Op::kMatMul); }),
                0);
    }
  }
}

TEST(GgmlLowerTest, UngatedExpertCombineRequiresOptInAndFallsBackCompletely) {
  for (const std::uint32_t rows : {1U, 3U, 8U, 9U, 65535U, 65536U}) {
    for (const bool padded_ids : {false, true})
      for (const bool direct : {false, true}) {
        for (const unsigned mode : {0U, 1U, 2U}) {
          SCOPED_TRACE(rows);
          SCOPED_TRACE(direct);
          SCOPED_TRACE(padded_ids);
          SCOPED_TRACE(mode);
          const auto registry = PlanningRegistry(true, true, mode == 2 ? kg::kDsv4CombineName : "");
          pl::Policy policy;
          if (mode != 0) policy.enabled = {"moe_combine"};
          pl::Canonicalize(policy);
          ir::Builder b;
          const auto down =
              b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 6, rows})));
          const auto ids_root = b.Input(
              ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({padded_ids ? 256U : 6U, rows})));
          const auto ids = padded_ids ? b.View(ids_root, ir::ShapeOf({6, rows}),
                                               {4, 1024, 1024ULL * rows, 1024ULL * rows}, 0)
                                      : ids_root;
          const auto weights =
              b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({6, rows})));
          const auto shared =
              b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, rows})));
          const std::array inputs = {down, ids_root, weights, shared};
          const std::array sources = {down, ids, weights, shared};
          b.Output(direct ? b.Fused(ir::FusedKind::kMoeCombine, sources)[0]
                          : b.Add(b.ExpertReduce(down, weights, ids, {}), shared));
          b.Output(ids);
          auto graph = std::move(b).Finish();
          ASSERT_TRUE(graph);
          co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
          for (auto input : inputs)
            step.inputs.push_back({.tensor = input, .kind = co::InputKind::kFeatures});
          const pl::StepShape shape{.segments = {{.rows = rows, .outputs = 0, .want_head = false}}};
          auto plan = pl::Plan(step.graph.Clone(), registry, Gb10(), policy,
                               {.staged = std::vector<ir::TensorId>(inputs.begin(), inputs.end()),
                                .written = kg::WrittenBytes,
                                .within = kg::WithinRoot});
          ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
          const kn::Addresses addresses{};
          auto lowered = kg::LowerStep({*plan, step, shape, addresses}, registry, Device());
          ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
          const bool native = mode == 1 && rows <= 65535;
          EXPECT_EQ(std::ranges::count_if(lowered->plan.steps,
                                          [](const auto& item) {
                                            return item.implementation == kg::kDsv4CombineName;
                                          }),
                    native ? 1 : 0);
          ASSERT_EQ(lowered->keep.size(), 2U);
          EXPECT_EQ(ggml_nbytes(lowered->keep[0]), 16U * rows * 4U);
          EXPECT_EQ(lowered->keep[1]->nb[0], 4U);
          EXPECT_EQ(lowered->keep[1]->nb[1], padded_ids ? 1024U : 24U);
          EXPECT_EQ(ggml_nbytes(lowered->keep[1]), (padded_ids ? 1024U : 24U) * (rows - 1U) + 24U);
          if (native) {
            ASSERT_EQ(lowered->plan.steps.size(), 1U);
            EXPECT_EQ(kg::LlmpOpInt(lowered->keep[0], 0), 1);
            EXPECT_EQ(lowered->keep[0]->src[1]->ne[0], 6);
          }
        }
      }
  }
}

TEST(GgmlLowerTest, UngatedExpertCombineOnlyIgnoresUnusedSingleColumnIdPitches) {
  const auto registry = PlanningRegistry();
  pl::Policy policy;
  policy.enabled = {"moe_combine"};
  pl::Canonicalize(policy);
  const pl::SelectionView selection(registry, Gb10(), policy, false, false);
  const auto tensor = [](llmp::format::Encoding encoding,
                         std::initializer_list<std::uint64_t> shape) {
    ir::TensorDesc t{.format = ir::Dense(encoding, ir::ShapeOf(shape))};
    t.nb = ir::NaturalStrides(t.format);
    return t;
  };
  for (const std::uint64_t used : {1U, 6U, 32U}) {
    SCOPED_TRACE(used);
    auto down = tensor(llmp::format::Encoding::kF32, {4096, used, 1});
    auto ids = tensor(llmp::format::Encoding::kI32, {used, 1});
    auto weights = tensor(llmp::format::Encoding::kF32, {used, 1});
    auto shared = tensor(llmp::format::Encoding::kF32, {4096, 1});
    auto output = shared;
    const std::array<const ir::TensorDesc*, 4> inputs{&down, &ids, &weights, &shared};
    const std::array<const ir::TensorDesc*, 1> outputs{&output};
    const auto accepts = [&] {
      const auto candidates =
          selection.Candidates(ir::Fused(ir::FusedKind::kMoeCombine), inputs, outputs, {});
      return std::ranges::any_of(candidates, [&](const auto candidate) {
        return registry.at(candidate).name == kg::kDsv4CombineName;
      });
    };
    ASSERT_TRUE(accepts());
    ids.nb = {4, 1024, 1024, 1024};  // Exact first-six-of256 learned-sort view.
    ASSERT_TRUE(accepts());
    ids.nb[0] = 8;
    EXPECT_FALSE(accepts());
    ids.nb[0] = 4;
    ids.view_offset = 2;
    EXPECT_FALSE(accepts());
    ids.view_offset = std::numeric_limits<std::uint64_t>::max() - 3;
    EXPECT_FALSE(accepts());  // Aligned offset, overflowing accessed I32 footprint.
    ids.view_offset = 4;
    EXPECT_TRUE(accepts());
    ids.view_offset = 0;
    weights.nb[1] += 4;
    EXPECT_FALSE(accepts());  // No relaxation for any other operand.
    weights.nb[1] -= 4;
    ids.format.encoding = llmp::format::Encoding::kF32;
    EXPECT_FALSE(accepts());
    ids.format.encoding = llmp::format::Encoding::kI32;
    down = tensor(llmp::format::Encoding::kF32, {4096, used, 2});
    ids = tensor(llmp::format::Encoding::kI32, {used, 2});
    weights = tensor(llmp::format::Encoding::kF32, {used, 2});
    shared = tensor(llmp::format::Encoding::kF32, {4096, 2});
    output = shared;
    ASSERT_TRUE(accepts());
    ids.nb = {4, 1024, 2048, 2048};
    EXPECT_TRUE(accepts());  // Strided column view across tokens is valid.
    ids.nb = {4, 1024, 1024, 1024};
    EXPECT_FALSE(accepts());  // Overlapping token rows are rejected.
  }
}

TEST(GgmlLowerTest, UngatedExpertCombineKeepsExistingGatedRecipe) {
  const auto registry = PlanningRegistry();
  pl::Policy policy;
  policy.enabled = {"moe_combine"};
  pl::Canonicalize(policy);
  for (const std::uint32_t rows : {3U, 9U, 512U})
    for (const bool direct : {false, true}) {
      SCOPED_TRACE(rows);
      SCOPED_TRACE(direct);
      ir::Builder b;
      const auto down =
          b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 6, rows})));
      const auto ids = b.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({6, rows})));
      const auto weights = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({6, rows})));
      const auto shared = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, rows})));
      const auto gate = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({1, rows})));
      const std::array inputs = {down, ids, weights, shared, gate};
      b.Output(direct ? b.Fused(ir::FusedKind::kMoeCombine, inputs)[0]
                      : b.Add(b.ExpertReduce(down, weights, ids, {}),
                              b.Mul(shared, b.Unary(gate, ir::UnaryFn::kSigmoid))));
      b.Output(ids);
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph);
      co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
      for (auto input : inputs)
        step.inputs.push_back({.tensor = input, .kind = co::InputKind::kFeatures});
      const pl::StepShape shape{.segments = {{.rows = rows, .outputs = 0, .want_head = false}}};
      auto plan = pl::Plan(step.graph.Clone(), registry, Gb10(), policy,
                           {.staged = std::vector<ir::TensorId>(inputs.begin(), inputs.end()),
                            .written = kg::WrittenBytes,
                            .within = kg::WithinRoot});
      ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
      const kn::Addresses addresses{};
      auto lowered = kg::LowerStep({*plan, step, shape, addresses}, registry, Device());
      ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
      ASSERT_EQ(lowered->plan.steps.size(), 1U);
      EXPECT_EQ(lowered->plan.steps[0].implementation, kg::kMoeCombineName);
      ASSERT_EQ(lowered->keep.size(), 2U);
      EXPECT_EQ(kg::LlmpOpOf(lowered->keep[0]), kg::LlmpOp::kMoeCombine);
      EXPECT_EQ(lowered->keep[0]->src[4]->ne[0], 1);
    }
}

TEST(GgmlLowerTest, UngatedExpertCombineChecksPackedBackingAlignmentAndAliases) {
  auto arena = kg::TensorArena::Create(16);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  auto* down = ggml_new_tensor_3d(c, GGML_TYPE_F32, 16, 6, 3);
  auto* weights = ggml_new_tensor_2d(c, GGML_TYPE_F32, 6, 3);
  auto* shared = ggml_new_tensor_2d(c, GGML_TYPE_F32, 16, 3);
  auto* result = kg::CombineSelectedWeights(c, down, weights, shared);
  kg::TensorArena::Bind(down, 0x100000);
  kg::TensorArena::Bind(weights, 0x200000);
  kg::TensorArena::Bind(shared, 0x300000);
  kg::TensorArena::Bind(result, 0x400000);
  ASSERT_TRUE(kg::CheckDsv4Combine(result));
  for (auto* operand : {down, weights, shared, result}) {
    const auto address = reinterpret_cast<std::uintptr_t>(operand->data);
    kg::TensorArena::Bind(operand, address + 1);
    EXPECT_FALSE(kg::CheckDsv4Combine(result));
    kg::TensorArena::Bind(operand, address);
    ASSERT_TRUE(kg::CheckDsv4Combine(result));
    operand->nb[1] += 4;
    EXPECT_FALSE(kg::CheckDsv4Combine(result));
    operand->nb[1] -= 4;
    ASSERT_TRUE(kg::CheckDsv4Combine(result));
  }
  for (auto* input : {down, weights, shared}) {
    kg::TensorArena::Bind(result, reinterpret_cast<std::uintptr_t>(input->data));
    EXPECT_FALSE(kg::CheckDsv4Combine(result));
  }
  kg::TensorArena::Bind(result, 0x400000);
  ASSERT_TRUE(kg::CheckDsv4Combine(result));
  down->ne[3] = 2;
  EXPECT_FALSE(kg::CheckDsv4Combine(result));
  down->ne[3] = 1;
  weights->type = GGML_TYPE_I32;
  EXPECT_FALSE(kg::CheckDsv4Combine(result));
}

TEST(GgmlLowerTest, BiasedRouteNativeChecksRejectInvalidPhysicalAndParameterContracts) {
  auto arena = kg::TensorArena::Create(32);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  auto* logits = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, 3);
  auto* ids = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256);
  auto* route = kg::RouteBiasedTopK(c, logits, ids, 6, true, 1.5F);
  kg::TensorArena::Bind(logits, 0x100000);
  kg::TensorArena::Bind(ids, 0x200000);
  kg::TensorArena::Bind(route, 0x300000);
  ASSERT_TRUE(kg::CheckDsv4Route(route));
  for (auto* operand : {logits, ids, route}) {
    const auto address = reinterpret_cast<std::uintptr_t>(operand->data);
    kg::TensorArena::Bind(operand, address + 1);
    EXPECT_FALSE(kg::CheckDsv4Route(route));
    kg::TensorArena::Bind(operand, address);
    ASSERT_TRUE(kg::CheckDsv4Route(route));
  }
  kg::TensorArena::Bind(logits, UINT64_MAX - 3);
  EXPECT_FALSE(kg::CheckDsv4Route(route));
  kg::TensorArena::Bind(logits, 0x100000);
  kg::TensorArena::Bind(route, 0x200000);
  EXPECT_FALSE(kg::CheckDsv4Route(route));
  kg::TensorArena::Bind(route, 0x300000);
  ids->type = GGML_TYPE_I32;
  EXPECT_FALSE(kg::CheckDsv4Route(route));
  ids->type = GGML_TYPE_F32;
  route->type = GGML_TYPE_F32;
  EXPECT_FALSE(kg::CheckDsv4Route(route));
  route->type = GGML_TYPE_I32;
  ids->nb[1] += 4;
  EXPECT_FALSE(kg::CheckDsv4Route(route));
  ids->nb[1] -= 4;
  kg::TensorArena::Bind(route, 0x100000);
  EXPECT_FALSE(kg::CheckDsv4Route(route));
  kg::TensorArena::Bind(route, 0x300000);
  route->src[2] = ids;
  EXPECT_FALSE(kg::CheckDsv4Route(route));
  route->src[2] = nullptr;
  ASSERT_TRUE(kg::CheckDsv4Route(route));
  for (const auto [index, value] :
       std::array<std::pair<std::size_t, std::int32_t>, 4>{{{1, 2}, {2, 0}, {3, 0}, {4, 3}}}) {
    const auto original = route->op_params[8 + index];
    route->op_params[8 + index] = value;
    EXPECT_FALSE(kg::CheckDsv4Route(route));
    route->op_params[8 + index] = original;
  }
  const auto scale = route->op_params[11];
  for (const float value :
       {-1.0F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    std::memcpy(&route->op_params[11], &value, sizeof(value));
    EXPECT_FALSE(kg::CheckDsv4Route(route));
  }
  route->op_params[11] = scale;
  ASSERT_TRUE(kg::CheckDsv4Route(route));
  auto* down = ggml_new_tensor_3d(c, GGML_TYPE_F32, 16, 6, 3);
  auto* shared = ggml_new_tensor_2d(c, GGML_TYPE_F32, 16, 3);
  auto* combine = kg::Dsv4Combine(c, down, route, shared);
  kg::TensorArena::Bind(down, 0x400000);
  kg::TensorArena::Bind(shared, 0x500000);
  kg::TensorArena::Bind(combine, 0x600000);
  EXPECT_FALSE(kg::CheckDsv4Combine(combine));  // legacy consumer expects interleaved rows
}

TEST(GgmlLowerTest, BiasedRouteRequiresOptInAndPaysBothPackedOutputPlanes) {
  for (const std::uint32_t rows : {1U, 3U, 16U, 17U, 512U}) {
    for (const unsigned mode : {0U, 1U, 2U, 3U}) {
      SCOPED_TRACE(rows);
      SCOPED_TRACE(mode);
      const auto registry = PlanningRegistry(true, true, mode == 2 ? kg::kDsv4RouteName : "");
      pl::Policy policy;
      if (mode != 0) policy.enabled = {"biased_route"};
      if (mode == 3) policy.disabled = {"route_reduce"};
      pl::Canonicalize(policy);
      ir::Builder b;
      const auto logits =
          b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({256, rows})));
      const auto ids = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({256})));
      const auto route = b.Route(logits, ids, {}, {},
                                 {.kind = ir::RouteKind::kSqrtSoftplusBias,
                                  .normalize = true,
                                  .experts = 256,
                                  .used = 6,
                                  .scale = 1.5F});
      b.Output(route[0]);
      b.Output(route[1]);
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph);
      EXPECT_EQ(kg::WrittenBytes(*graph, route[0]), rows <= 16 ? 2U * 6U * rows * 4U : 0U);
      EXPECT_EQ(kg::WithinRoot(*graph, route[1]), rows <= 16 ? route[0] : ir::TensorId{});
      co::StepGraph step{.graph = std::move(*graph),
                         .inputs = {{.tensor = logits, .kind = co::InputKind::kFeatures},
                                    {.tensor = ids, .kind = co::InputKind::kFeatures}},
                         .weights = {},
                         .state = {}};
      const pl::StepShape shape{.segments = {{.rows = rows, .outputs = 0, .want_head = false}}};
      auto plan = pl::Plan(step.graph.Clone(), registry, Gb10(), policy,
                           {.staged = std::vector<ir::TensorId>{logits, ids},
                            .written = kg::WrittenBytes,
                            .within = kg::WithinRoot});
      ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
      const kn::Addresses addresses{};
      auto lowered = kg::LowerStep({*plan, step, shape, addresses}, registry, Device());
      ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
      const bool native = mode == 1 && rows <= 16;
      EXPECT_EQ(std::ranges::count_if(
                    lowered->plan.steps,
                    [](const auto& entry) { return entry.implementation == kg::kDsv4RouteName; }),
                native ? 1 : 0);
      ASSERT_EQ(lowered->keep.size(), 2U);
      const auto ids_leaf = std::ranges::find(lowered->leaves, ids, &kg::LoweredLeaf::primitive);
      ASSERT_NE(ids_leaf, lowered->leaves.end());
      EXPECT_EQ(std::ranges::count(lowered->leaves, ids, &kg::LoweredLeaf::primitive), 1);
      if (native) {
        ASSERT_EQ(lowered->plan.steps.size(), 1U);
        const auto* out_ids = lowered->keep[0];
        const auto* weights = lowered->keep[1];
        EXPECT_EQ(out_ids->type, GGML_TYPE_I32);
        EXPECT_EQ(weights->type, GGML_TYPE_F32);
        EXPECT_EQ(out_ids->view_src, weights->view_src);
        EXPECT_EQ(out_ids->view_offs, 0U);
        EXPECT_EQ(weights->view_offs, 6U * rows * 4U);
        EXPECT_EQ(ggml_nbytes(out_ids->view_src), 2U * 6U * rows * 4U);
        EXPECT_EQ(out_ids->nb[1], 24U);
        EXPECT_EQ(weights->nb[1], 24U);
      }
    }
  }
}

TEST(GgmlLowerTest, BiasedRouteRejectsUnsupportedSourcesAndPhysicalContracts) {
  const auto registry = PlanningRegistry();
  pl::Policy policy;
  policy.enabled = {"biased_route"};
  pl::Canonicalize(policy);
  const pl::SelectionView selection(registry, Gb10(), policy, false, false);
  const auto tensor = [](llmp::format::Encoding encoding, std::uint64_t width, std::uint64_t rows) {
    ir::TensorDesc t{.format = ir::Dense(encoding, ir::ShapeOf({width, rows}))};
    t.nb = ir::NaturalStrides(t.format);
    return t;
  };
  auto logits = tensor(llmp::format::Encoding::kF32, 256, 3);
  auto ids = tensor(llmp::format::Encoding::kF32, 256, 1);
  auto out = tensor(llmp::format::Encoding::kI32, 6, 3);
  auto weights = tensor(llmp::format::Encoding::kF32, 6, 3);
  std::array<const ir::TensorDesc*, 4> inputs{&logits, &ids, nullptr, nullptr};
  const std::array<const ir::TensorDesc*, 2> outputs{&out, &weights};
  ir::RouteParams params{.kind = ir::RouteKind::kSqrtSoftplusBias,
                         .normalize = true,
                         .experts = 256,
                         .used = 6,
                         .scale = 1.5F};
  const auto accepts = [&] {
    return !selection
                .Candidates(ir::Primitive(ir::Op::kRoute), inputs, outputs,
                            std::as_bytes(std::span(&params, 1)))
                .empty();
  };
  ASSERT_TRUE(accepts());
  inputs[2] = &logits;
  EXPECT_FALSE(accepts());
  inputs[2] = nullptr;
  inputs[3] = &ids;
  EXPECT_FALSE(accepts());
  inputs[3] = nullptr;
  ids.nb[1] += 4;
  EXPECT_FALSE(accepts());
  ids.nb[1] -= 4;
  weights.view_offset = 4;
  EXPECT_FALSE(accepts());
  weights.view_offset = 0;
  weights.format.encoding = llmp::format::Encoding::kI32;
  EXPECT_FALSE(accepts());
  weights.format.encoding = llmp::format::Encoding::kF32;
  params.experts = 128;
  EXPECT_FALSE(accepts());
  params.experts = 256;
  params.used = 33;
  EXPECT_FALSE(accepts());
  params.used = 6;
  params.scale = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(accepts());
  params.scale = 1.5F;
  logits = tensor(llmp::format::Encoding::kF32, 256, 17);
  out = tensor(llmp::format::Encoding::kI32, 6, 17);
  weights = tensor(llmp::format::Encoding::kF32, 6, 17);
  EXPECT_FALSE(accepts());
}

TEST(GgmlLowerTest, BiasedRouteBindsOwnedBiasAndRejectsChangedSource) {
  const auto registry = PlanningRegistry();
  ir::Builder b;
  const auto logits = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({256, 3})));
  const auto bias = b.Weight(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({256})), {});
  const auto out = b.Route(logits, bias, {}, {},
                           {.kind = ir::RouteKind::kSqrtSoftplusBias,
                            .normalize = true,
                            .experts = 256,
                            .used = 6,
                            .scale = 1.5F});
  b.Output(out[0]);
  b.Output(out[1]);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph);
  co::StepGraph step{
      .graph = std::move(*graph),
      .inputs = {{.tensor = logits, .kind = co::InputKind::kFeatures}},
      .weights = {{.tensor = bias, .ref = {}, .parts = {{"bias", llmp::format::PartRole::kData}}}},
      .state = {}};
  pl::Policy policy;
  policy.enabled = {"biased_route"};
  pl::Canonicalize(policy);
  const pl::StepShape shape{.segments = {{.rows = 3, .outputs = 0, .want_head = false}}};
  auto plan = pl::Plan(step.graph.Clone(), registry, Gb10(), policy,
                       {.staged = std::vector<ir::TensorId>{logits},
                        .written = kg::WrittenBytes,
                        .within = kg::WithinRoot});
  ASSERT_TRUE(plan);
  const kn::Addresses addresses{
      .weight = [](const ir::WeightRef& ref, llmp::format::PartRole part) -> std::uint64_t {
        return ref == ir::WeightRef{} && part == llmp::format::PartRole::kData ? 0x100000U : 0;
      },
      .weight_rows = {},
      .state = {},
      .save = {},
      .activations = 0x200000U,
      .activation_bytes = plan->placement.extent};
  auto lowered = kg::LowerStep({*plan, step, shape, addresses}, registry, Device());
  ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
  ASSERT_EQ(lowered->plan.steps.size(), 1U);
  const auto* route = lowered->keep[0]->view_src;
  ASSERT_NE(route, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(route->src[1]->data), 0x100000U);
  EXPECT_EQ(kg::LlmpOpInt(route, 4), 2);
  step.weights[0].parts[0].second = llmp::format::PartRole::kBlockScales;
  EXPECT_FALSE(kg::LowerStep({*plan, step, shape, addresses}, registry, Device()));
}

TEST(GgmlLowerTest, SelectedHashRouteNativeChecksRejectAliasesAndLegacyCombine) {
  auto arena = kg::TensorArena::Create(32);
  ASSERT_TRUE(arena);
  auto* c = arena->context();
  auto* logits = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, 3);
  auto* ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, 6, 3);
  auto* route = kg::RouteSelectedIds(c, logits, ids, 6, true, 0x1p-14F, 1.5F);
  kg::TensorArena::Bind(logits, 0x100000);
  kg::TensorArena::Bind(ids, 0x200000);
  kg::TensorArena::Bind(route, 0x300000);
  ASSERT_TRUE(kg::CheckDsv4Route(route));
  for (auto* operand : {logits, ids, route}) {
    const auto address = reinterpret_cast<std::uintptr_t>(operand->data);
    kg::TensorArena::Bind(operand, address + 1);
    EXPECT_FALSE(kg::CheckDsv4Route(route));
    kg::TensorArena::Bind(operand, address);
    ASSERT_TRUE(kg::CheckDsv4Route(route));
  }
  ids->nb[1] += 4;
  EXPECT_FALSE(kg::CheckDsv4Route(route));
  ids->nb[1] -= 4;
  kg::TensorArena::Bind(route, 0x100000);
  EXPECT_FALSE(kg::CheckDsv4Route(route));
  kg::TensorArena::Bind(route, 0x300000);
  route->src[2] = ids;
  EXPECT_FALSE(kg::CheckDsv4Route(route));
  route->src[2] = nullptr;
  ASSERT_TRUE(kg::CheckDsv4Route(route));
  auto* down = ggml_new_tensor_3d(c, GGML_TYPE_F32, 16, 6, 3);
  auto* shared = ggml_new_tensor_2d(c, GGML_TYPE_F32, 16, 3);
  auto* combine = kg::Dsv4Combine(c, down, route, shared);
  kg::TensorArena::Bind(down, 0x400000);
  kg::TensorArena::Bind(shared, 0x500000);
  kg::TensorArena::Bind(combine, 0x600000);
  EXPECT_FALSE(kg::CheckDsv4Combine(combine));  // legacy consumer expects interleaved rows
}

TEST(GgmlLowerTest, SelectedHashRouteRequiresOptInAndPaysBothPackedOutputPlanes) {
  for (const std::uint32_t rows : {1U, 3U, 8U, 9U, 512U}) {
    for (const unsigned mode : {0U, 1U, 2U, 3U}) {
      SCOPED_TRACE(rows);
      SCOPED_TRACE(mode);
      const auto registry = PlanningRegistry(true, true, mode == 2 ? kg::kDsv4RouteName : "");
      pl::Policy policy;
      if (mode != 0) policy.enabled = {"hash_route"};
      if (mode == 3) policy.disabled = {"route_reduce"};
      pl::Canonicalize(policy);
      ir::Builder b;
      const auto logits =
          b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({256, rows})));
      const auto ids = b.Input(ir::Dense(llmp::format::Encoding::kI32, ir::ShapeOf({6, rows})));
      const auto route = b.Route(logits, {}, ids, {},
                                 {.kind = ir::RouteKind::kHash,
                                  .normalize = true,
                                  .experts = 256,
                                  .used = 6,
                                  .scale = 1.5F});
      b.Output(route[0]);
      b.Output(route[1]);
      auto graph = std::move(b).Finish();
      ASSERT_TRUE(graph);
      EXPECT_EQ(kg::WrittenBytes(*graph, route[0]), 2U * 6U * rows * 4U);
      EXPECT_EQ(kg::WithinRoot(*graph, route[1]), route[0]);
      co::StepGraph step{.graph = std::move(*graph),
                         .inputs = {{.tensor = logits, .kind = co::InputKind::kFeatures},
                                    {.tensor = ids, .kind = co::InputKind::kFeatures}},
                         .weights = {},
                         .state = {}};
      const pl::StepShape shape{.segments = {{.rows = rows, .outputs = 0, .want_head = false}}};
      auto plan = pl::Plan(step.graph.Clone(), registry, Gb10(), policy,
                           {.staged = std::vector<ir::TensorId>{logits, ids},
                            .written = kg::WrittenBytes,
                            .within = kg::WithinRoot});
      ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
      const kn::Addresses addresses{};
      auto lowered = kg::LowerStep({*plan, step, shape, addresses}, registry, Device());
      ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
      const bool native = mode == 1 && rows <= 8;
      EXPECT_EQ(std::ranges::count_if(
                    lowered->plan.steps,
                    [](const auto& entry) { return entry.implementation == kg::kDsv4RouteName; }),
                native ? 1 : 0);
      ASSERT_EQ(lowered->keep.size(), 2U);
      const auto ids_leaf = std::ranges::find(lowered->leaves, ids, &kg::LoweredLeaf::primitive);
      ASSERT_NE(ids_leaf, lowered->leaves.end());
      EXPECT_EQ(std::ranges::count(lowered->leaves, ids, &kg::LoweredLeaf::primitive), 1);
      if (!native) {
        EXPECT_EQ(plan->tensors.at(route[0].v), plan->tensors.at(ids.v));
        EXPECT_EQ(lowered->keep[0], ids_leaf->tensor);
      }
      if (native) {
        ASSERT_EQ(lowered->plan.steps.size(), 1U);
        const auto* out_ids = lowered->keep[0];
        const auto* weights = lowered->keep[1];
        EXPECT_EQ(out_ids->type, GGML_TYPE_I32);
        EXPECT_EQ(weights->type, GGML_TYPE_F32);
        EXPECT_EQ(out_ids->view_src, weights->view_src);
        EXPECT_EQ(out_ids->view_offs, 0U);
        EXPECT_EQ(weights->view_offs, 6U * rows * 4U);
        EXPECT_EQ(ggml_nbytes(out_ids->view_src), 2U * 6U * rows * 4U);
        EXPECT_EQ(out_ids->nb[1], 24U);
        EXPECT_EQ(weights->nb[1], 24U);
      }
    }
  }
}

TEST(GgmlLowerTest, SelectedHashRouteRejectsUnsupportedSourcesAndPhysicalContracts) {
  const auto registry = PlanningRegistry();
  pl::Policy policy;
  policy.enabled = {"hash_route"};
  pl::Canonicalize(policy);
  const pl::SelectionView selection(registry, Gb10(), policy, false, false);
  const auto tensor = [](llmp::format::Encoding encoding, std::uint64_t width, std::uint64_t rows) {
    ir::TensorDesc t{.format = ir::Dense(encoding, ir::ShapeOf({width, rows}))};
    t.nb = ir::NaturalStrides(t.format);
    return t;
  };
  auto logits = tensor(llmp::format::Encoding::kF32, 256, 3);
  auto ids = tensor(llmp::format::Encoding::kI32, 6, 3);
  auto out = ids;
  auto weights = tensor(llmp::format::Encoding::kF32, 6, 3);
  std::array<const ir::TensorDesc*, 4> inputs{&logits, nullptr, &ids, nullptr};
  const std::array<const ir::TensorDesc*, 2> outputs{&out, &weights};
  ir::RouteParams params{
      .kind = ir::RouteKind::kHash, .normalize = true, .experts = 256, .used = 6, .scale = 1.5F};
  const auto accepts = [&] {
    return !selection
                .Candidates(ir::Primitive(ir::Op::kRoute), inputs, outputs,
                            std::as_bytes(std::span(&params, 1)))
                .empty();
  };
  ASSERT_TRUE(accepts());
  inputs[1] = &logits;
  EXPECT_FALSE(accepts());
  inputs[1] = nullptr;
  inputs[3] = &ids;
  EXPECT_FALSE(accepts());
  inputs[3] = nullptr;
  ids.nb[1] += 4;
  EXPECT_FALSE(accepts());
  ids.nb[1] -= 4;
  weights.view_offset = 4;
  EXPECT_FALSE(accepts());
  weights.view_offset = 0;
  weights.format.encoding = llmp::format::Encoding::kI32;
  EXPECT_FALSE(accepts());
  weights.format.encoding = llmp::format::Encoding::kF32;
  params.experts = 128;
  EXPECT_FALSE(accepts());
  params.experts = 256;
  params.used = 33;
  EXPECT_FALSE(accepts());
  params.used = 6;
  params.scale = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(accepts());
  params.scale = 1.5F;
  logits = tensor(llmp::format::Encoding::kF32, 256, 65536);
  ids = tensor(llmp::format::Encoding::kI32, 6, 65536);
  out = ids;
  weights = tensor(llmp::format::Encoding::kF32, 6, 65536);
  EXPECT_FALSE(accepts());
}

TEST(GgmlLowerTest, StandaloneHcPostKeepsNativeAndMissingImplementationDecomposes) {
  for (bool native : {true, false}) {
    SCOPED_TRACE(native);
    auto registry = PlanningRegistry(native);
    ir::Builder b;
    const auto y = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({17, 3})));
    const auto x = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({17, 4, 3})));
    const auto post = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({4, 3})));
    const auto comb = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({4, 4, 3})));
    b.Output(b.HcPost(y, x, post, comb, {.streams = 4}));
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph);
    co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
    const std::vector<ir::TensorId> staged{y, x, post, comb};
    for (auto tensor : staged)
      step.inputs.push_back({.tensor = tensor, .kind = co::InputKind::kFeatures});
    const pl::StepShape shape{.segments = {{.rows = 3, .outputs = 0, .want_head = false}}};
    auto plan = pl::Plan(step.graph.Clone(), registry, Gb10(), {},
                         {.staged = staged, .written = kg::WrittenBytes});
    ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
    const kn::Addresses addresses{};
    auto lowered = kg::LowerStep({*plan, step, shape, addresses}, registry, Device());
    ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
    EXPECT_EQ(lowered->plan.steps.size(), native ? 1U : 39U);
    EXPECT_EQ(
        std::ranges::count_if(lowered->plan.steps,
                              [](const auto& s) { return s.implementation == kg::kHcPostName; }),
        native ? 1 : 0);
    if (native) EXPECT_EQ(lowered->nodes.back()->op, GGML_OP_DSV4_HC_POST);
  }
}

TEST(GgmlLowerTest, UnsupportedHcPostFormsStillLowerThroughPrimitives) {
  const auto registry = PlanningRegistry();
  for (const std::uint32_t streams : {3U, 4U}) {
    // Three streams with a combination matrix, or four without one.
    const bool has_comb = streams == 3;
    ir::Builder b;
    const auto y = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({17, 1})));
    const auto x = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({17, streams, 1})));
    const auto post = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({streams, 1})));
    std::vector<ir::TensorId> staged{y, x, post};
    ir::TensorId comb{};
    if (has_comb) {
      comb = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({streams, streams, 1})));
      staged.push_back(comb);
    }
    b.Output(b.HcPost(y, x, post, comb, {.streams = streams}));
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph);
    co::StepGraph step{.graph = std::move(*graph), .inputs = {}, .weights = {}, .state = {}};
    for (auto tensor : staged)
      step.inputs.push_back({.tensor = tensor, .kind = co::InputKind::kFeatures});
    const pl::StepShape shape{.segments = {{.rows = 1, .outputs = 0, .want_head = false}}};
    auto plan = pl::Plan(step.graph.Clone(), registry, Gb10(), {},
                         {.staged = staged, .written = kg::WrittenBytes});
    ASSERT_TRUE(plan) << (plan ? "" : plan.error().detail);
    const kn::Addresses addresses{};
    auto lowered = kg::LowerStep({*plan, step, shape, addresses}, registry, Device());
    ASSERT_TRUE(lowered) << (lowered ? "" : lowered.error().detail);
    EXPECT_EQ(lowered->plan.steps.size(), has_comb ? 23U : 11U);
    EXPECT_TRUE(std::ranges::none_of(
        lowered->plan.steps, [](const auto& s) { return s.implementation == kg::kHcPostName; }));
  }
}

TEST(GgmlLowerTest, StandaloneHcPostCapabilityRejectsOutsideNativeContract) {
  const auto registry = PlanningRegistry();
  const pl::Policy policy{};
  const pl::SelectionView selection(registry, Gb10(), policy, false, false);
  const auto tensor = [](std::initializer_list<std::uint64_t> shape) {
    ir::TensorDesc t{.format = ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf(shape))};
    t.nb = ir::NaturalStrides(t.format);
    return t;
  };
  auto y = tensor({17, 3}), x = tensor({17, 4, 3}), post = tensor({4, 3});
  auto comb = tensor({4, 4, 3}), out = tensor({17, 4, 3});
  std::array<const ir::TensorDesc*, 4> inputs{&y, &x, &post, &comb};
  const std::array<const ir::TensorDesc*, 1> outputs{&out};
  ir::HcParams params{.streams = 4};
  const auto accepts = [&] {
    return !selection
                .Candidates(ir::Primitive(ir::Op::kHcPost), inputs, outputs,
                            std::as_bytes(std::span(&params, 1)))
                .empty();
  };
  ASSERT_TRUE(accepts());
  params.streams = 3;
  EXPECT_FALSE(accepts());
  params.streams = 4;
  inputs[3] = nullptr;
  EXPECT_FALSE(accepts());
  inputs[3] = &comb;
  x.nb[0] = 2;
  EXPECT_FALSE(accepts());
  x = tensor({17, 4, 3});
  x.view_offset = 2;
  EXPECT_FALSE(accepts());
  x.view_offset = 0;
  post.format = ir::Dense(llmp::format::Encoding::kF16, ir::ShapeOf({4, 3}));
  EXPECT_FALSE(accepts());
  post = tensor({4, 3});
  out = tensor({17, 4, 2});
  EXPECT_FALSE(accepts());
  // No allocation: the logical output exceeds CheckHcPost's grid contract.
  constexpr std::uint64_t width = 1ULL << 40;
  y = tensor({width, 1});
  x = tensor({width, 4, 1});
  post = tensor({4, 1});
  comb = tensor({4, 4, 1});
  out = tensor({width, 4, 1});
  EXPECT_FALSE(accepts());
}

TEST(GgmlLowerTest, Gemma2LowersEverySetupShape) { ExpectFamilyLowers<Gemma2Family>(); }

TEST(GgmlLowerTest, Gemma3LowersEverySetupShape) { ExpectFamilyLowers<Gemma3Family>(); }

TEST(GgmlLowerTest, Q8DenseCacheUsesBlockedPrimitiveAttentionAndStores) {
  Ir inputs{.spec = Gemma2Family::Spec(), .source = Gemma2Family::Source()};
  const auto bytes = co::KvCacheEncodingBytes(inputs.spec, llmp::format::Encoding::kQ8_0);
  ASSERT_TRUE(bytes) << bytes.error();
  ASSERT_TRUE(co::ApplyKvCacheEncoding(inputs.spec, llmp::format::Encoding::kQ8_0, *bytes));
  auto layout = co::StageLayoutUniform(inputs.spec, 0, 512, 8, kExtent);
  ASSERT_TRUE(layout) << layout.error();
  inputs.layout.emplace(std::move(*layout));
  const Cells cells = Gemma2Family::State(512, 8);
  for (const Case& c :
       {Case{.segments = {{0, 3, 0}, {1, 1, 5}}}, Case{.segments = {{0, 1, 256}, {1, 1, 300}}}}) {
    const auto lowered = Lower<Gemma2Family>(inputs, c, cells, true, Gemma2Family::Policy());
    ASSERT_TRUE(lowered.lowered);
    ExpectWellFormed(lowered);
    EXPECT_EQ(lowered.owners,
              0U);  // F16-only owner fusion must fall back, not forge a dense Q8 view
    std::size_t attentions = 0, stores = 0;
    for (const auto& step : lowered.lowered->plan.steps) {
      if (step.implementation == kg::kFlashAttnVecQ8Name) ++attentions;
      if (step.implementation == kg::kSetRowsQuantName ||
          step.implementation == kg::kSetRowsQuantGroupName ||
          step.implementation == kg::kRopeSetRowsQuantName)
        ++stores;
    }
    EXPECT_EQ(attentions, Gemma2Family::P().layers * c.segments.size());
    EXPECT_GT(stores, 0U);
    for (const auto& state : lowered.step->state) {
      const auto& tensor = lowered.step->graph.tensor(state.tensor);
      EXPECT_EQ(tensor.format.encoding, llmp::format::Encoding::kQ8_0);
    }
  }
}

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
      const kn::LowerInput input{*plan, step, shape, addresses};
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
  const kn::LowerInput input{*plan, step, shape, addresses};
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
  const kn::LowerInput input{*plan, step, shape, addresses};
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
      const kn::LowerInput input{*plan, step, shape, addresses};
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
  auto layout = co::StageLayoutUniform(ir_.spec, 0, 8192, 128, kExtent);
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
  const kn::LowerInput input{*plan, *step, shape, addresses, 8192};
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
