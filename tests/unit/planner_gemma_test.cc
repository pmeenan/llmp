// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The planner (planner/plan.h) on the Gemma 2 and Gemma 3 specs' steps
// with each family's production recipe (the mappings' policies: fused
// norm-mul, quantized GeGLU and norm-add for both, norm-rope and grouped KV
// stores for Gemma 3 only, device masks, shared Q8 preparation), on a fake
// registry that declares what that recipe selects as capabilities, and a
// model of the GB10's product routing (MMVQ through 8 columns, the fused
// GeGLU at one). For every shape the engine measures at setup at the
// serving defaults (its ChunkMeasurementRows loop):
// - the recipe's fusions apply where the policy enables them (the fused
//   norm-mul, norm-add and quantized GeGLU in both; norm-rope and grouped
//   stores in Gemma 3's alone, none in Gemma 2's), and shared Q8 prepares
//   the activations each group of products shares;
// - placement: the placement holds: no two tensors live at once share
//   bytes, by lifetimes recomputed here from the planned steps, and the
//   tensors fit the extent. (The planned extents were once compared with
//   PlaceActivations' for the M3 graph builders, retired with their
//   runners after the migration's gate; the owner-attention forms are the
//   lowering's, ggml_lower_test.)

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <map>
#include <optional>
#include <print>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "components/step.h"
#include "engine/support.h"
#include "execution/registry.h"
#include "families/gemma.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ggml.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/set_rows_group.h"
#include "kernels/ggml/tensors.h"
#include "model/state_contract.h"
#include "planner/placement.h"
#include "planner/plan.h"
#include "planner/policy.h"
#include "planner/step_shape.h"
#include "spec/mapping.h"
#include "spec/mappings/gemma2.h"
#include "spec/mappings/gemma3.h"
#include "spec_fixtures.h"

namespace llmp::planner {
namespace {

namespace kg = llmp::kernels::ggml;
namespace fam = llmp::test_support::families;
namespace ex = llmp::execution;
using format::Encoding;
using format::Layout;

constexpr std::uint64_t kExtent = 2ULL << 20U;

// ------------------------------------------------- the fake registry

constexpr std::array kQuantized = {Encoding::kQ4_0, Encoding::kQ8_0};
constexpr std::array kF32 = {Encoding::kF32};
constexpr std::array kQ8_1 = {Encoding::kQ8_1};
constexpr std::array kF16 = {Encoding::kF16};
// MMVQ and MMQ: quantized weights and F32 activations; the prepared
// product: the activations already Q8_1.
constexpr std::array kProductRules = {ex::OperandRule{.operand = 0, .encodings = kQuantized},
                                      ex::OperandRule{.operand = 1, .encodings = kF32}};
constexpr std::array kPreparedRules = {ex::OperandRule{.operand = 0, .encodings = kQuantized},
                                       ex::OperandRule{.operand = 1, .encodings = kQ8_1}};
constexpr std::array kGluRules = {ex::OperandRule{.operand = 0, .encodings = kQuantized},
                                  ex::OperandRule{.operand = 1, .encodings = kQuantized},
                                  ex::OperandRule{.operand = 2, .encodings = kF32}};
constexpr std::array kStoreRules = {ex::OperandRule{.operand = 0, .encodings = kF16},
                                    ex::OperandRule{.operand = 1, .encodings = kF32}};

// CheckGemmaNormRope's full D256/D512 rotation; CheckGemmaNormAdd's widths.
bool NormRopeWidths(const ex::CapabilityQuery& q) { return q.k == 256 || q.k == 512; }
bool NormAddWidths(const ex::CapabilityQuery& q) {
  return q.k == 2304 || q.k == 2560 || q.k == 2816 || q.k == 5376;
}

ex::Implementation Fake(std::string name, ex::Operation operation, ex::Capability capability) {
  return {.name = std::move(name),
          .operation = operation,
          .source = "fake",
          .revision = "r1",
          .build = "cpu",
          .variant = "model of the GB10's GGML selection",
          .capability = capability};
}

ex::Capability Any(ir::OpKey key) { return ex::Capability{.op = key}; }

ex::Registry GemmaRegistry() {
  using ex::Operation;
  using ir::FusedKind;
  using ir::Op;
  const auto prim = [](Op op) { return Any(ir::Primitive(op)); };
  std::vector<ex::Implementation> v = {
      Fake("fake.get_rows", Operation::kGetRows, prim(Op::kGetRows)),
      Fake("fake.scale", Operation::kScale, prim(Op::kScale)),
      Fake("fake.rms_norm", Operation::kRmsNorm, prim(Op::kRmsNorm)),
      Fake("fake.mul", Operation::kMul, prim(Op::kMul)),
      Fake("fake.add", Operation::kAdd, prim(Op::kAdd)),
      Fake("fake.rope", Operation::kRope, prim(Op::kRope)),
      Fake("fake.set_rows", Operation::kSetRows, prim(Op::kCacheStore)),
      Fake("fake.flash_attn", Operation::kFlashAttn, prim(Op::kAttention)),
      Fake("fake.concat", Operation::kConcat, prim(Op::kConcat)),
      Fake("fake.cont", Operation::kCont, prim(Op::kCont)),
      Fake("fake.geglu", Operation::kGeGlu, prim(Op::kGlu)),
      Fake("fake.unary", Operation::kUnary, prim(Op::kUnary)),
      Fake("fake.argmax", Operation::kTopK, prim(Op::kArgmax)),
      Fake("fake.mask", Operation::kFill, prim(Op::kMaskBuild)),
      Fake("fake.q8_1", Operation::kQuantize,
           ex::Capability{.op = ir::Primitive(Op::kPrepare), .shape = {.max_rows = 8}}),
      Fake("fake.mul_mat.mmvq", Operation::kMatMul,
           ex::Capability{.op = ir::Primitive(Op::kMatMul),
                          .operands = kProductRules,
                          .shape = {.max_rows = 8},
                          .group = "fake.mul_mat"}),
      Fake("fake.mul_mat.mmq", Operation::kMatMul,
           ex::Capability{.op = ir::Primitive(Op::kMatMul),
                          .operands = kProductRules,
                          .shape = {.min_rows = 9},
                          .group = "fake.mul_mat"}),
      Fake("fake.mmvq.prepared", Operation::kMatMul,
           ex::Capability{.op = ir::Primitive(Op::kMatMul),
                          .operands = kPreparedRules,
                          .shape = {.max_rows = 8}}),
      Fake("fake.rms_norm_mul.fused", Operation::kRmsNormMul,
           Any(ir::Fused(FusedKind::kRmsNormMul))),
      Fake("fake.rms_norm_mul_rope.fused", Operation::kRmsNormMulRope,
           ex::Capability{.op = ir::Fused(FusedKind::kRmsNormMulRope), .accepts = &NormRopeWidths}),
      Fake("fake.rms_norm_mul_add.fused", Operation::kRmsNormMulAdd,
           ex::Capability{.op = ir::Fused(FusedKind::kRmsNormMulAdd), .accepts = &NormAddWidths}),
      Fake("fake.mul_mat_geglu.mmvq_fused", Operation::kMulMatGeGlu,
           ex::Capability{.op = ir::Fused(FusedKind::kMatMulGlu),
                          .operands = kGluRules,
                          .shape = {.max_rows = 1}}),
      Fake("fake.set_rows.grouped", Operation::kSetRows,
           ex::Capability{
               .op = ir::Fused(FusedKind::kGroupedStores), .operands = kStoreRules, .batch = 16}),
      Fake("fake.rope_set_rows.fused", Operation::kRopeSetRows,
           Any(ir::Fused(FusedKind::kRopeStore))),
  };
  return std::move(ex::Registry::Create(std::move(v)).value());
}

const DeviceFacts& Gb10() {
  static const DeviceFacts facts{.device_class = "sm_121", .arch = 121};
  return facts;
}

// The recipes as policies: the architecture mappings' text (one ungated
// rule each).
const std::string_view kGemma2Policy = spec::mappings::Gemma2Mapping().policies.front().text;
const std::string_view kGemma3Policy = spec::mappings::Gemma3Mapping().policies.front().text;

Policy PolicyOf(std::string_view text, bool shared_q8) {
  Policy p = ParsePolicy(text).value();
  if (!shared_q8) {
    p.disabled.emplace_back("prepared_inputs");
    Canonicalize(p);
  }
  return p;
}

// ------------------------------------------------- the families

// The cells a family's state holds (the K/V caches' global and ring cells,
// in closed form).
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
  static constexpr bool kGemma3 = false;
  static const fam::Gemma2Profile& P() { return fam::Gemma2_2B(); }
  static auto Resources() { return test_support::gemma2::Resources(); }
  static Cells State(std::uint32_t context, std::uint32_t rows) {
    return CellsOf(P().window, context, rows);
  }
  static spec::ModelSpec Spec() {
    auto values = fam::Gemma2Values(P());
    return *spec::Synthesize(spec::mappings::Gemma2Mapping(), test_support::Gemma2Source(),
                             *values);
  }
  static spec::Source Source() { return test_support::Gemma2Source(); }
};

struct Gemma3Family {
  static constexpr bool kGemma3 = true;
  static const fam::Gemma3Profile& P() { return fam::Gemma3_4BQat(); }
  static auto Resources() { return test_support::gemma3::Resources(); }
  static Cells State(std::uint32_t context, std::uint32_t rows) {
    return CellsOf(P().window, context, rows);
  }
  static spec::ModelSpec Spec() {
    auto values = fam::Gemma3Values(P());
    return *spec::Synthesize(spec::mappings::Gemma3Mapping(), test_support::Gemma3Source(),
                             *values);
  }
  static spec::Source Source() { return test_support::Gemma3Source(); }
};

// A step: per owner (slot, rows, past), and its outputs: the head (one row
// an owner), every row, greedy, or none.
enum class Out : std::uint8_t { kHead, kAll, kGreedy, kStateOnly };
struct Case {
  std::vector<std::array<std::uint32_t, 3>> segments;  // slot, rows, past
  Out out = Out::kHead;
  std::string Name() const {
    std::string s;
    for (const auto& [slot, rows, past] : segments) {
      s += std::format("[{}:{}@{}]", slot, rows, past);
    }
    return s + std::format(" out {}", static_cast<int>(out));
  }
};

// The shapes the engine measures at setup with `slots` owners of up to
// `max_rows` rows.
std::vector<Case> SetupCases(std::uint32_t context, std::uint32_t max_rows, std::uint32_t slots) {
  std::vector<Case> out;
  const std::uint32_t wave_rows = max_rows * slots;
  const std::uint32_t head_rows = slots;
  for (const std::uint32_t budget : {wave_rows, head_rows}) {
    for (std::uint32_t count = 1; count <= slots; ++count) {
      for (const auto& rows :
           engine::support::ChunkMeasurementRows(max_rows, budget, count, true)) {
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

std::uint32_t ReadOf(const std::array<std::uint32_t, 3>& s, const Cells& state, bool local) {
  const std::uint32_t read = (s[1] + s[2] + 255) / 256 * 256;
  return std::min(read, local ? state.local_cells : state.global_cells);
}

StepShape IrShape(const Case& c, const Cells& state) {
  StepShape shape;
  for (const auto& s : c.segments) {
    shape.segments.push_back({.slot = static_cast<std::uint8_t>(s[0]),
                              .rows = s[1],
                              .past = s[2],
                              .read_cells = ReadOf(s, state, false),
                              .outputs = c.out == Out::kStateOnly ? 0U
                                         : c.out == Out::kAll     ? s[1]
                                                                  : 1U,
                              .greedy = c.out == Out::kGreedy});
  }
  return shape;
}

// ------------------------------------------------- the planner side

struct Planned {
  std::optional<components::StepGraph> step;
  std::optional<StepPlan> plan;
};

template <typename F>
struct Ir {
  spec::ModelSpec spec = F::Spec();
  spec::Source source = F::Source();
  std::optional<model::StateLayoutDerived> layout;
  ex::Registry registry = GemmaRegistry();
};

template <typename F>
Planned PlanIr(Ir<F>& ir, const Case& c, const Cells& state, const Policy& policy) {
  Planned out;
  auto step = components::BuildStep(ir.spec, 0, IrShape(c, state),
                                    {.tensors = ir.source.tensors, .state = &*ir.layout});
  EXPECT_TRUE(step.has_value()) << step.error();
  if (!step) return out;
  // The device masks derive their cells' positions (the lowering's
  // producer): those inputs are not staged.
  std::vector<ir::TensorId> staged;
  for (const components::StepInput& in : step->inputs) {
    if (in.kind != components::InputKind::kCellPositions) staged.push_back(in.tensor);
  }
  // A copy of the primitive graph for the plan (graphs are move-only):
  // rebuilt from the same step.
  auto again = components::BuildStep(ir.spec, 0, IrShape(c, state),
                                     {.tensors = ir.source.tensors, .state = &*ir.layout});
  EXPECT_TRUE(again.has_value());
  auto plan = Plan(std::move(again->graph), ir.registry, Gb10(), policy, {.staged = staged});
  EXPECT_TRUE(plan.has_value()) << (plan ? "" : plan.error().detail);
  out.step.emplace(std::move(*step));
  if (plan) out.plan.emplace(std::move(*plan));
  return out;
}

// Weight leaves by what they bind, to their resource's index.
template <typename Resources>
std::map<ir::WeightRef, std::size_t> WeightIndex(const components::StepGraph& step,
                                                 const Resources& resources) {
  std::map<ir::WeightRef, std::size_t> weight;
  for (const components::StepWeight& w : step.weights) {
    const std::string& name = w.parts.front().first;
    const auto at = std::ranges::find_if(
        resources, [&](const auto& r) { return std::ranges::contains(r.roles, name); });
    weight[w.ref] = static_cast<std::size_t>(at - resources.begin());
  }
  return weight;
}

// Every two placed tensors whose lifetimes meet hold disjoint bytes, the
// lifetimes recomputed from the planned steps: from the step computing a
// root (a staged input from the start) to the last step reading it or a
// view of it, or reading its prepared form (kPrepare's source); the
// outputs, kept tensors and the last launching node's results to the end.
void ExpectPlacedDisjoint(const StepPlan& p) {
  const ir::Graph& g = p.graph;
  const auto end = static_cast<std::int64_t>(p.steps.size());
  struct Life {
    std::int64_t first = 0;
    std::int64_t last = -1;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
  };
  std::map<std::uint32_t, Life> life;
  for (const auto& [t, offset] : p.placement.offsets) {
    const ir::Role role = g.tensor(t).role;
    const auto bytes = RootBytes(g.tensor(t));
    ASSERT_TRUE(bytes.has_value());
    life[t.v] = {.first = role == ir::Role::kInput || role == ir::Role::kConstant ? -1 : end + 1,
                 .offset = offset,
                 .bytes = (*bytes + 255) / 256 * 256};
    EXPECT_LE(offset + life[t.v].bytes, p.placement.extent);
  }
  const auto read = [&](ir::TensorId t, std::int64_t s) {
    if (const auto it = life.find(g.root(t).v); it != life.end()) {
      it->second.last = std::max(it->second.last, s);
    }
  };
  for (std::size_t s = 0; s < p.steps.size(); ++s) {
    const auto step = static_cast<std::int64_t>(s);
    for (const ir::NodeId id : p.steps[s].nodes) {
      const ir::Node& n = g.node(id);
      for (const ir::TensorId o : n.outputs()) {
        if (const auto it = life.find(o.v); it != life.end()) {
          it->second.first = std::min(it->second.first, step);
        }
      }
      for (const ir::TensorId src : n.sources()) {
        if (!src.valid()) continue;
        read(src, step);
        const ir::Node& made = g.node(g.producer(g.root(src)));
        if (made.key == ir::Primitive(ir::Op::kPrepare)) read(made.src[0], step);
      }
    }
  }
  for (const auto list : {g.outputs(), g.keep()}) {
    for (const ir::TensorId t : list) read(t, end);
  }
  for (std::size_t i = g.nodes().size(); i-- > 0;) {
    if (LaunchesNothing(g.nodes()[i].key)) continue;
    for (const ir::TensorId o : g.nodes()[i].outputs()) read(o, end);
    break;
  }
  std::vector<std::pair<std::uint32_t, Life>> order(life.begin(), life.end());
  for (auto& [t, l] : order) {
    ASSERT_LE(l.first, end) << "tensor " << t << " is computed by no step";
    l.last = std::max(l.last, l.first);
  }
  std::ranges::sort(order, {}, [](const auto& e) { return e.second.first; });
  std::vector<std::pair<std::uint32_t, Life>> live;
  for (const auto& [t, l] : order) {
    std::erase_if(live, [&](const auto& e) { return e.second.last < l.first; });
    for (const auto& [u, m] : live) {
      ASSERT_TRUE(l.offset + l.bytes <= m.offset || m.offset + m.bytes <= l.offset)
          << "tensors " << t << " and " << u << " share bytes while both live";
    }
    live.emplace_back(t, l);
  }
}

template <typename Resources>
std::vector<std::vector<std::size_t>> IrShared(const components::StepGraph& step,
                                               const StepPlan& plan, const Resources& resources) {
  const auto weight = WeightIndex(step, resources);
  const ir::Graph& g = plan.graph;
  std::vector<std::vector<std::size_t>> out;
  std::map<std::uint32_t, std::size_t> prep;
  for (const ir::Node& n : g.nodes()) {
    if (n.key == ir::Primitive(ir::Op::kPrepare)) {
      prep[n.out[0].v] = out.size();
      out.emplace_back();
    }
  }
  for (const ir::Node& n : g.nodes()) {
    if (n.key == ir::Primitive(ir::Op::kMatMul) && prep.contains(n.src[1].v)) {
      const auto& ref = std::get<ir::WeightRef>(g.tensor(n.src[0]).binding);
      out.at(prep.at(n.src[1].v)).push_back(weight.at(ref));
    }
  }
  for (auto& v : out) std::ranges::sort(v);
  return out;
}

// The shapes the checks run: the setup shapes at the serving defaults.
template <typename F>
std::vector<std::pair<std::uint32_t, std::vector<Case>>> AllCases() {
  std::vector<std::pair<std::uint32_t, std::vector<Case>>> out;
  if constexpr (F::kGemma3) {
    out.emplace_back(4096, SetupCases(4096, 128, 2));
    out.emplace_back(8448, SetupCases(8448, 128, 1));
  } else {
    out.emplace_back(8192, SetupCases(8192, 128, 2));
  }
  return out;
}

std::size_t CountApplied(const StepPlan& plan, std::string_view pattern) {
  return static_cast<std::size_t>(std::ranges::count(plan.applied, pattern, &Applied::pattern)) +
         static_cast<std::size_t>(std::ranges::count_if(
             plan.steps, [&](const Selected& s) { return s.batch == pattern; }));
}

template <typename F>
void ExpectFamilyPlans(std::string_view policy_text) {
  std::size_t shapes = 0, fused = 0, shared = 0, glu = 0;
  for (auto& [context, cases] : AllCases<F>()) {
    const Cells state = F::State(context, 128);
    Ir<F> ir;
    auto layout = components::StageLayoutUniform(ir.spec, 0, context, 128, kExtent);
    ASSERT_TRUE(layout.has_value()) << layout.error();
    ir.layout.emplace(std::move(*layout));
    for (const Case& c : cases) {
      SCOPED_TRACE(std::format("context {} {}", context, c.Name()));
      // The fusions, without shared Q8 (its custom nodes would change the
      // node correspondence).
      {
        const Planned p = PlanIr(ir, c, state, PolicyOf(policy_text, false));
        ASSERT_TRUE(p.plan.has_value());
        EXPECT_GT(CountApplied(*p.plan, "rms_norm_mul"), 0U);
        EXPECT_GT(CountApplied(*p.plan, "rms_norm_mul_add"), 0U);
        // Gemma 3's recipe alone fuses the norm and the rotation and
        // groups the KV stores.
        EXPECT_EQ(CountApplied(*p.plan, "rms_norm_mul_rope") > 0, F::kGemma3);
        EXPECT_EQ(CountApplied(*p.plan, "grouped_stores") > 0, F::kGemma3);
        // A decode row's quantized products fuse as the GeGLU; a prompt
        // chunk's do not.
        glu += CountApplied(*p.plan, "mul_mat_glu");
        fused += p.plan->applied.size();
      }
      // Shared Q8 as served: products share each preparation, and the
      // placement holds.
      {
        const Planned p = PlanIr(ir, c, state, PolicyOf(policy_text, true));
        ASSERT_TRUE(p.plan.has_value());
        const auto groups = IrShared(*p.step, *p.plan, F::Resources());
        for (const auto& g : groups) EXPECT_GT(g.size(), 1U);
        shared += groups.size();
        ASSERT_GT(p.plan->placement.extent, 0U);
        ExpectPlacedDisjoint(*p.plan);
        if (::testing::Test::HasFatalFailure()) return;
      }
      ++shapes;
    }
  }
  std::println("{} shapes, {} applied patterns, {} quantized GeGLUs, {} shared preparations",
               shapes, fused, glu, shared);
  EXPECT_GT(fused, 0U);
  EXPECT_GT(glu, 0U);
  EXPECT_GT(shared, 0U);
}

TEST(PlannerGemmaTest, Gemma2PlansItsRecipe) { ExpectFamilyPlans<Gemma2Family>(kGemma2Policy); }

TEST(PlannerGemmaTest, Gemma3PlansItsRecipe) { ExpectFamilyPlans<Gemma3Family>(kGemma3Policy); }

// The two recipes differ as data: one policy text each, in its mapping,
// canonical.
TEST(PlannerGemmaTest, TheRecipesArePolicies) {
  const Policy g2 = ParsePolicy(kGemma2Policy).value();
  const Policy g3 = ParsePolicy(kGemma3Policy).value();
  EXPECT_EQ(ToText(g2), kGemma2Policy);
  EXPECT_EQ(ToText(g3), kGemma3Policy);
  EXPECT_FALSE(Disabled(g2, "owner_attention"));
  EXPECT_FALSE(Disabled(g3, "owner_attention"));
  EXPECT_TRUE(Disabled(g2, "rms_norm_mul_rope"));
  EXPECT_FALSE(Disabled(g3, "rms_norm_mul_rope"));
  EXPECT_TRUE(Disabled(g2, "grouped_stores"));
  EXPECT_FALSE(Disabled(g3, "grouped_stores"));
  EXPECT_NE(PatternsDigest(g2), PatternsDigest(g3));
  EXPECT_NE(Digest(g2), Digest(g3));
}

}  // namespace
}  // namespace llmp::planner
