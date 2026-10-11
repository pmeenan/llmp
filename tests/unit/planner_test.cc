// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The planner (planner/plan.h) on small graphs and a fake registry:
// - the policy's canonical text, its refusals and digests;
// - each rewrite pattern where it matches, and where it must not (an
//   intermediate read elsewhere or returned, no implementation for the
//   result, the policy's switch, another shape, a read it would move past
//   a store into the same bytes), the table's order where
//   patterns overlap, and the batch of cache stores;
// - selection: measured ranks per device class before the ladder, groups,
//   tuning data an implementation needs, row-invariant and capturable
//   requirements, the primitive fallback for a fused node and the refusal
//   of a node nothing computes; the same plan every time;
// - placement: lifetimes, the prepared input kept with its products, and
//   no two tensors live together in the same bytes;
// - lanes, and plan keys that follow what decides a plan and not a
//   segment's past.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "execution/registry.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "planner/fusions.h"
#include "planner/lanes.h"
#include "planner/pattern.h"
#include "planner/placement.h"
#include "planner/plan.h"
#include "planner/policy.h"
#include "planner/select.h"
#include "planner/step_shape.h"

namespace llmp::planner {
namespace {

namespace ex = llmp::execution;
using format::Encoding;
using format::Layout;
using ir::FusedKind;
using ir::Op;
using ir::TensorId;

// ------------------------------------------------------------ fixtures

constexpr std::array kQ8 = {Encoding::kQ8_0};
constexpr std::array kQ8_1 = {Encoding::kQ8_1};
constexpr std::array kF32 = {Encoding::kF32};
constexpr std::array kProductRules = {ex::OperandRule{.operand = 0, .encodings = kQ8},
                                      ex::OperandRule{.operand = 1, .encodings = kF32}};
constexpr std::array kPreparedRules = {ex::OperandRule{.operand = 0, .encodings = kQ8},
                                       ex::OperandRule{.operand = 1, .encodings = kQ8_1}};

bool Width64(const ex::CapabilityQuery& q) { return q.k == 64; }

ex::Implementation Fake(std::string name, ex::Capability capability,
                        ex::Operation operation = ex::Operation::kAdd) {
  return {.name = std::move(name),
          .operation = operation,
          .source = "fake",
          .revision = "r1",
          .build = "cpu",
          .variant = "test",
          .capability = capability};
}

ex::Capability Any(ir::OpKey key) { return ex::Capability{.op = key}; }

// Everything the tests' graphs use, primitive and fused.
std::vector<ex::Implementation> Everything() {
  const auto prim = [](Op op) { return Any(ir::Primitive(op)); };
  return {
      Fake("t.scale", prim(Op::kScale)),
      Fake("t.rms_norm", prim(Op::kRmsNorm)),
      Fake("t.mul", prim(Op::kMul)),
      Fake("t.add", prim(Op::kAdd)),
      Fake("t.rope", prim(Op::kRope)),
      Fake("t.glu", prim(Op::kGlu)),
      Fake("t.set_rows", prim(Op::kCacheStore)),
      Fake("t.q8_1", ex::Capability{.op = ir::Primitive(Op::kPrepare), .shape = {.max_rows = 8}}),
      Fake("t.mul_mat.vector", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                              .operands = kProductRules,
                                              .shape = {.max_rows = 8},
                                              .group = "t.mul_mat"}),
      Fake("t.mul_mat.tile", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                            .operands = kProductRules,
                                            .shape = {.min_rows = 9},
                                            .group = "t.mul_mat"}),
      Fake("t.mul_mat.prepared", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                                .operands = kPreparedRules,
                                                .shape = {.max_rows = 8}}),
      Fake("t.rms_norm_mul", Any(ir::Fused(FusedKind::kRmsNormMul))),
      Fake("t.rms_norm_mul_rope",
           ex::Capability{.op = ir::Fused(FusedKind::kRmsNormMulRope), .accepts = &Width64}),
      Fake("t.rms_norm_mul_add", Any(ir::Fused(FusedKind::kRmsNormMulAdd))),
      Fake("t.mul_mat_glu",
           ex::Capability{.op = ir::Fused(FusedKind::kMatMulGlu), .shape = {.max_rows = 1}}),
      Fake("t.rope_store", Any(ir::Fused(FusedKind::kRopeStore))),
      Fake("t.set_rows.grouped",
           ex::Capability{.op = ir::Fused(FusedKind::kGroupedStores), .batch = 16}),
  };
}

ex::Registry Registry(std::vector<ex::Implementation> implementations = Everything()) {
  return std::move(ex::Registry::Create(std::move(implementations)).value());
}

// Everything but the named implementations.
ex::Registry Without(std::initializer_list<std::string_view> names) {
  std::vector<ex::Implementation> v;
  for (ex::Implementation& i : Everything()) {
    if (std::ranges::find(names, std::string_view(i.name)) == names.end()) {
      v.push_back(std::move(i));
    }
  }
  return Registry(std::move(v));
}

const DeviceFacts kDevice{.device_class = "test", .arch = 121};

format::TensorFormat F32(std::initializer_list<std::uint64_t> ne) {
  return ir::Dense(Encoding::kF32, ir::ShapeOf(ne));
}
format::TensorFormat Q8(std::uint64_t k, std::uint64_t n) {
  return {.encoding = Encoding::kQ8_0, .layout = Layout::kGgmlBlocks, .shape = ir::ShapeOf({k, n})};
}

ir::WeightRef W(std::uint16_t role, std::uint32_t layer = 0) {
  return {.stage = 0, .component = 1, .role = role, .layer = layer};
}

ir::Graph Finish(ir::Builder& b) {
  auto g = std::move(b).Finish();
  EXPECT_TRUE(g.has_value()) << (g ? "" : g.error().detail);
  return std::move(*g);
}

StepPlan PlanOf(ir::Graph g, const ex::Registry& registry, const Policy& policy = {},
                const PlanOptions& options = {}) {
  auto p = Plan(std::move(g), registry, kDevice, policy, options);
  EXPECT_TRUE(p.has_value()) << (p ? "" : p.error().detail);
  return std::move(*p);
}

std::vector<std::string_view> Patterns(const StepPlan& p) {
  std::vector<std::string_view> out;
  for (const Applied& a : p.applied) out.push_back(a.pattern);
  return out;
}

std::vector<std::string> Names(const StepPlan& p, const ex::Registry& registry) {
  std::vector<std::string> out;
  for (const Selected& s : p.steps) out.push_back(registry.at(s.implementation).name);
  return out;
}

std::vector<ir::OpKey> Keys(const ir::Graph& g) {
  std::vector<ir::OpKey> out;
  for (const ir::Node& n : g.nodes()) {
    if (!LaunchesNothing(n.key)) out.push_back(n.key);
  }
  return out;
}

Policy Disabling(std::initializer_list<std::string> names) {
  Policy p{.disabled = names};
  Canonicalize(p);
  return p;
}

// x [64, rows] normed, times w.
struct NormGraph {
  ir::Builder b;
  TensorId x, w, n, m;
  explicit NormGraph(std::uint64_t width = 64, std::uint64_t rows = 3) {
    x = b.Input(F32({width, rows}));
    w = b.Weight(F32({width}), W(0));
    n = b.RmsNorm(x, 1e-6f);
    m = b.Mul(n, w);
  }
};

// ---------------------------------------------------------------- policy

TEST(PlannerPolicyTest, TextIsCanonicalAndChecked) {
  Policy p{.disabled = {"rope_store", "grouped_stores"},
           .max_decode_rows = 8,
           .row_invariant = true,
           .ranks = {{.device_class = "sm_86", .implementation = "a.b", .rank = 1},
                     {.device_class = "sm_121", .implementation = "c", .rank = 0}}};
  EXPECT_FALSE(Validate(p).has_value());  // not in order
  Canonicalize(p);
  ASSERT_TRUE(Validate(p).has_value());
  const std::string text = ToText(p);
  EXPECT_EQ(text,
            "max_decode_rows 8\nrow_invariant\ndisable grouped_stores\ndisable rope_store\n"
            "rank sm_121 c 0\nrank sm_86 a.b 1\n");
  EXPECT_EQ(ParsePolicy(text).value(), p);
  EXPECT_EQ(ToText(Policy{}), "max_decode_rows 16\n");
  for (const std::string_view bad : {
           "max_decode_rows 16",                             // no newline
           "disable rope_store\nmax_decode_rows 16\n",       // out of order
           "max_decode_rows 16\ndisable fuse_everything\n",  // no such pattern
           "max_decode_rows 16\ndisable rope_store\ndisable rope_store\n",
           "max_decode_rows 0\n",                                     // out of range
           "max_decode_rows 016\n",                                   // not canonical
           "max_decode_rows 16\nfuse harder\n",                       // no such directive
           "max_decode_rows 16\nrank sm.121 x 0\n",                   // a class with a dot
           "max_decode_rows 16\nrank sm_121 x 0\nrank sm_121 x 1\n",  // ranked twice
       }) {
    EXPECT_FALSE(ParsePolicy(bad).has_value()) << bad;
  }
}

TEST(PlannerPolicyTest, DigestsFollowWhatChanges) {
  const Policy base{};
  const Policy off = Disabling({"rms_norm_mul_rope"});
  EXPECT_NE(Digest(base), Digest(off));
  EXPECT_NE(PatternsDigest(base), PatternsDigest(off));
  Policy rows = base;
  rows.max_decode_rows = 8;
  EXPECT_NE(Digest(base), Digest(rows));
  EXPECT_EQ(PatternsDigest(base), PatternsDigest(rows));  // same patterns
  EXPECT_EQ(Digest(base), Digest(Policy{}));
}

TEST(PlannerPolicyTest, RanksAreTheDeviceClasses) {
  Policy p{.ranks = {{.device_class = "a", .implementation = "x", .rank = 0},
                     {.device_class = "b", .implementation = "y", .rank = 0}}};
  const auto ranks = RanksFor(p, "b");
  ASSERT_EQ(ranks.size(), 1U);
  EXPECT_EQ(ranks[0].implementation, "y");
  EXPECT_TRUE(RanksFor(p, "c").empty());
}

// ------------------------------------------------------- rewrite patterns

TEST(PlannerPatternTest, DirectBankFormulaDecomposesWithoutProofAndAllowsGeneralRecipes) {
  std::vector<ex::Implementation> implementations = {
      Fake("t.concat", Any(ir::Primitive(Op::kConcat))),
      Fake("t.attention", Any(ir::Primitive(Op::kAttention))),
      Fake("t.bank", {.op = ir::Fused(FusedKind::kStateBankAttention),
                      .priority = 1,
                      .needs_state_span = true})};
  const auto registry = Registry(implementations);
  implementations.push_back(Fake("t.general", Any(ir::Fused(FusedKind::kStateBankAttention))));
  const auto general = Registry(implementations);
  for (const bool state : {false, true}) {
    ir::Builder b;
    b.SetOwner(0);
    const auto half = [](std::initializer_list<std::uint64_t> ne) {
      return ir::Dense(Encoding::kF16, ir::ShapeOf(ne));
    };
    const auto q = b.Input(F32({4, 1, 8}));
    const auto a = state ? b.State(half({4, 256}), {.tensor = 0}) : b.Input(half({4, 256}));
    const auto z = state ? b.State(half({4, 512}), {.tensor = 1}) : b.Input(half({4, 512}));
    const auto av = b.View(a, ir::ShapeOf({4, 1, 256}), {2, 8, 8, 2048}, 0);
    const auto zv = b.View(z, ir::ShapeOf({4, 1, 256}), {2, 8, 8, 4096}, 0);
    const auto mask = b.Input(half({512, 32}));
    b.Output(
        b.Fused(FusedKind::kStateBankAttention, std::array{q, av, zv, mask}, ir::AttnParams{})[0]);
    auto graph = std::move(b).Finish();
    ASSERT_TRUE(graph);
    auto fallback = Plan(graph->Clone(), registry, kDevice, {});
    ASSERT_TRUE(fallback) << (fallback ? "" : fallback.error().detail);
    EXPECT_EQ(fallback->decomposed, std::vector<std::string>{"fused.state_bank_attention"});
    const StateSpans malformed = [](std::span<const StateCellSlice>) {
      return std::optional<StateSpanFact>{{.bank = {.range = {.offset = 0, .bytes = 1},
                                                    .encoding = Encoding::kF16,
                                                    .cell_bytes = 8},
                                           .region_bytes = 6144}};
    };
    auto bad_fact = Plan(graph->Clone(), registry, kDevice, {}, {.state_spans = malformed});
    ASSERT_TRUE(bad_fact);
    EXPECT_EQ(bad_fact->decomposed, fallback->decomposed);
    auto ordinary = Plan(graph->Clone(), general, kDevice, {});
    ASSERT_TRUE(ordinary);
    EXPECT_TRUE(ordinary->decomposed.empty());
    ASSERT_EQ(ordinary->steps.size(), 1U);
    EXPECT_EQ(general.at(ordinary->steps.front().implementation).name, "t.general");
  }
}

TEST(PlannerPatternTest, StateBanksKeepAttentionParametersAndRefuseObservableCopies) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.concat", Any(ir::Primitive(Op::kConcat))));
  implementations.push_back(Fake("t.attention", Any(ir::Primitive(Op::kAttention))));
  implementations.push_back(
      Fake("t.bank", {.op = ir::Fused(FusedKind::kStateBankAttention), .needs_state_span = true}));
  const auto registry = Registry(std::move(implementations));
  Policy on;
  on.enabled = {"state_bank_attention"};
  const SelectionView enabled(registry, kDevice, on, false, false);
  const Policy default_policy{};
  const SelectionView off(registry, kDevice, default_policy, false, false);
  // 0: canonical; others change exactly one proof premise.
  for (int mutation = 0; mutation < 14; ++mutation) {
    SCOPED_TRACE(mutation);
    ir::Builder b;
    b.SetOwner(0);
    const auto half = [](std::initializer_list<std::uint64_t> ne) {
      return ir::Dense(Encoding::kF16, ir::ShapeOf(ne));
    };
    ir::TensorId q;
    if (mutation == 12) {
      b.SetJoined();
      const auto joined_q = b.Input(F32({4, 5, 2}));
      b.SetOwner(0);
      q = b.View(joined_q, ir::ShapeOf({4, 2, 2}), {4, 16, 80, 160}, 48);
    } else {
      if (mutation == 13) b.SetOwner(1);
      q = b.Input(F32({4, 2, 2}));
      b.SetOwner(0);
    }
    const auto raw = b.State(half({mutation == 2 ? 8U : 4U, 256}), {.tensor = 0});
    const auto compressed = b.State(half({4, 256}), {.tensor = 1});
    const std::uint64_t raw_cells = mutation == 1 || mutation == 3 ? 255 : 256;
    const auto& rd = b.tensor(raw);
    const auto a = b.View(raw, ir::ShapeOf({4, 1, raw_cells}), {2, rd.nb[1], rd.nb[1], rd.nb[2]},
                          mutation == 3 ? 8 : 0);
    const auto z = b.View(compressed, ir::ShapeOf({4, 1, 256}), {2, 8, 8, 2048}, 0);
    const auto joined = b.Concat(a, z, 2);
    if (mutation == 4) b.Keep(joined);
    const auto keys = b.Permute(joined, {0, 2, 1, 3});
    const auto values = b.Permute(mutation == 5 ? b.Concat(a, z, 2) : joined, {0, 2, 1, 3});
    if (mutation == 6) b.Keep(values);
    if (mutation == 7 || mutation == 8) {
      // Mutate either state root through an alias between copy and Attention.
      const auto target = mutation == 8 ? compressed : raw;
      const auto alias = b.View(target, ir::ShapeOf({4, 256}), {2, 8, 2048, 2048}, 0);
      const auto replacement = b.Input(F32({4, 1}));
      const auto row = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({1})));
      b.CacheStore(alias, replacement, row);
    }
    const auto mask = b.Input(half({raw_cells + 256, 32}));
    const auto sinks = b.Input(F32({2}));
    const ir::AttnParams params{.scale = 0.25F,
                                .softcap = 2,
                                .max_bias = 4,
                                .window = 128,
                                .sinks = true,
                                .accumulate = ir::Accumulate::kF32,
                                .sparse_any = true,
                                .n_kv_max = 128};
    ir::TensorDesc rounded;
    rounded.format = half({4, 2, 2});
    const auto output =
        mutation == 9 ? b.Emit(ir::Primitive(Op::kAttention), ir::ToBytes(params),
                               std::array{q, keys, values, mask, sinks}, std::array{rounded})[0]
                      : b.Attention(q, keys, values, mask, sinks, params);
    b.Output(output);
    auto graph = std::move(b).Finish();
    if (mutation == 13) {
      ASSERT_FALSE(graph);  // the IR already rejects a foreign owner's query
      EXPECT_NE(graph.error().detail.find("reads segment 1 for owner 0"), std::string::npos);
      continue;
    }
    ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
    const StateSpans proof =
        [=](std::span<const StateCellSlice> slices) -> std::optional<StateSpanFact> {
      if (mutation == 10 || slices.size() != 2) return std::nullopt;
      return StateSpanFact{.bank = {.range = {.offset = 4096, .bytes = 4096},
                                    .encoding = Encoding::kF16,
                                    .cell_bytes = mutation == 11 ? 10U : 8U},
                           .region_bytes = 16384};
    };
    const MatchContext context(*graph, enabled, nullptr, proof);
    EXPECT_EQ(fusions::StateBankAttentionMatch(context, graph->producer(output)).has_value(),
              mutation == 0 || mutation == 12);
    if (mutation != 0 && mutation != 12) continue;
    auto changed = ApplyPatterns(*graph, enabled, nullptr, proof);
    ASSERT_TRUE(changed);
    ASSERT_EQ(changed->applied.size(), 1U);
    const auto& result = changed->graph.node(changed->graph.producer(changed->graph.outputs()[0]));
    EXPECT_EQ(result.key, ir::Fused(FusedKind::kStateBankAttention));
    EXPECT_EQ(result.src[0], changed->tensors.at(q.v));  // exact joined owner view retained
    EXPECT_EQ(result.params, ir::ToBytes(params));       // every field, including sparse flags
    EXPECT_EQ(changed->applied[0].replaced.size(), 4U);  // concat, two views, attention
    auto disabled = ApplyPatterns(*graph, off, nullptr, proof);
    ASSERT_TRUE(disabled);
    EXPECT_TRUE(disabled->applied.empty());
    auto absent = ApplyPatterns(*graph, enabled);
    ASSERT_TRUE(absent);
    EXPECT_TRUE(absent->applied.empty());
    auto no_bank = Everything();
    no_bank.push_back(Fake("t.concat", Any(ir::Primitive(Op::kConcat))));
    no_bank.push_back(Fake("t.attention", Any(ir::Primitive(Op::kAttention))));
    const auto fallback_registry = Registry(std::move(no_bank));
    auto fallback = Plan(changed->graph.Clone(), fallback_registry, kDevice, on);
    ASSERT_TRUE(fallback) << (fallback ? "" : fallback.error().detail);
    EXPECT_EQ(fallback->decomposed, std::vector<std::string>{"fused.state_bank_attention"});
    const auto& old_attn =
        fallback->graph.node(fallback->graph.producer(fallback->graph.outputs()[0]));
    EXPECT_EQ(old_attn.key, ir::Primitive(Op::kAttention));
    EXPECT_EQ(old_attn.params, ir::ToBytes(params));
  }
}

TEST(PlannerPatternTest, Dsv4QHeadChecksTailOwnershipSideReadersAndInterveningWrites) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.dsv4_qhead", Any(ir::Fused(FusedKind::kDsv4QHead))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{};
  const SelectionView selection(registry, kDevice, policy, false, false);
  for (int mutation = 0; mutation <= 10; ++mutation) {
    SCOPED_TRACE(mutation);
    ir::Builder b;
    b.SetComponent(1);
    const TensorId root = mutation == 5
                              ? b.State(F32({512, 6}), ir::StateRef{})
                              : b.Input(ir::Dense(mutation == 8 ? Encoding::kF16 : Encoding::kF32,
                                                  ir::ShapeOf({512, 6})));
    const TensorId q = b.Reshape(root, ir::ShapeOf({512, 2, 3}));
    const TensorId positions = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({3})));
    ir::TensorDesc rounded;
    rounded.format = ir::Dense(Encoding::kF16, ir::ShapeOf({512, 2, 3}));
    const TensorId normalized = mutation == 9 ? b.Emit(ir::Primitive(Op::kRmsNorm),
                                                       ir::ToBytes(ir::NormParams{.eps = 1e-6F}),
                                                       std::array{q}, std::array{rounded})[0]
                                              : b.RmsNorm(q, 1e-6F);
    if (mutation == 4) b.Keep(normalized);
    if (mutation == 5) {
      const TensorId replacement = b.Input(F32({512, 1}));
      const TensorId cell = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({1})));
      b.CacheStore(root, replacement, cell);
    }
    const TensorId frequency = mutation == 7 ? b.Input(F32({32})) : TensorId{};
    if (mutation == 6) b.SetComponent(2);
    const ir::RopeParams rope{.mode = mutation == 2 ? ir::RopeMode::kNeox : ir::RopeMode::kNormal,
                              .inverse = mutation == 3,
                              .dims = 64,
                              .offset = mutation == 1 ? 446U : 448U,
                              .original_context = 65536,
                              .base = 1000000.0F,
                              .scale = 0.125F,
                              .ext_factor = 1.0F,
                              .attn_factor = 1.25F};
    ir::TensorDesc result_desc;
    result_desc.format =
        ir::Dense(mutation == 10 ? Encoding::kF16 : Encoding::kF32, ir::ShapeOf({512, 2, 3}));
    const TensorId out =
        b.Emit(ir::Primitive(Op::kRope), ir::ToBytes(rope),
               std::array{normalized, positions, frequency}, std::array{result_desc})[0];
    b.Output(out);
    auto g = std::move(b).Finish();
    ASSERT_TRUE(g) << (g ? "" : g.error().detail);
    const MatchContext c(*g, selection, nullptr);
    EXPECT_EQ(fusions::Dsv4QHeadMatch(c, g->producer(out)).has_value(), mutation == 0);
    if (mutation != 0) continue;
    Policy on;
    on.enabled = {"dsv4_qhead"};
    const SelectionView enabled(registry, kDevice, on, false, false);
    auto rewritten = ApplyPatterns(*g, enabled, nullptr);
    ASSERT_TRUE(rewritten) << (rewritten ? "" : rewritten.error().detail);
    ASSERT_EQ(rewritten->applied.size(), 1U);
    EXPECT_EQ(rewritten->applied[0].pattern, "dsv4_qhead");
    const auto& result =
        rewritten->graph.node(rewritten->graph.producer(rewritten->graph.outputs()[0]));
    EXPECT_EQ(result.key, ir::Fused(FusedKind::kDsv4QHead));
    const auto params = result.Params<ir::NormRopeParams>();
    EXPECT_EQ(ir::ToBytes(params.rope), g->node(g->producer(out)).params);
    EXPECT_EQ(params.eps, 1e-6F);
    auto disabled = ApplyPatterns(*g, selection, nullptr);
    ASSERT_TRUE(disabled);
    EXPECT_TRUE(disabled->applied.empty());
    const auto primitive_registry = Registry();
    const SelectionView unavailable(primitive_registry, kDevice, on, false, false);
    EXPECT_FALSE(fusions::Dsv4QHeadMatch(MatchContext(*g, unavailable, nullptr), g->producer(out)));
  }
}

TEST(PlannerPatternTest, Dsv4HcPreChecksTheFullCarryAndPrimitiveArithmetic) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.dsv4_hc_pre", Any(ir::Fused(FusedKind::kDsv4HcPre))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{};
  const SelectionView selection(registry, kDevice, policy, false, false);
  for (int mutation = 0; mutation <= 10; ++mutation) {
    SCOPED_TRACE(mutation);
    ir::Builder b;
    b.SetComponent(1);
    b.SetOwner(0);
    const TensorId x =
        mutation == 7 ? b.State(F32({8, 4, 3}), ir::StateRef{}) : b.Input(F32({8, 4, 3}));
    const TensorId fn = b.Weight(F32({32, 24}), ir::WeightRef{});
    const TensorId scale = b.Weight(F32({3}), ir::WeightRef{.role = 1});
    const TensorId base = b.Weight(F32({24}), ir::WeightRef{.role = 2});
    const TensorId norm_weight =
        mutation == 10 ? b.Input(F32({8})) : b.Weight(F32({8}), ir::WeightRef{.role = 3});
    const TensorId mixes =
        b.MatMul(fn, b.RmsNorm(b.Reshape(x, ir::ShapeOf({32, 3})), mutation == 1 ? 1e-3F : 1e-6F));
    const auto run = [&](std::uint64_t first) {
      return b.View(mixes, ir::ShapeOf({4, 3}), {4, 96, 288, 288}, first * 4);
    };
    const auto sub = [&](TensorId t, std::uint64_t n, std::uint64_t first) {
      return b.View(t, ir::ShapeOf({n}), {4, n * 4, n * 4, n * 4}, first * 4);
    };
    TensorId post = b.Mul(run(4), sub(scale, 1, 1));
    post = b.Scale(b.Unary(b.Add(post, sub(base, 4, 4)), ir::UnaryFn::kSigmoid),
                   mutation == 3 ? 3.0F : 2.0F);
    const TensorId comb =
        b.HcMix(mixes, mutation == 5 ? b.Input(F32({3})) : scale, base,
                {.mix = ir::HcMixKind::kSinkhorn, .streams = 4, .iterations = 5, .eps = 1e-5F});
    if (mutation == 6) b.Keep(mixes);
    if (mutation == 7) {
      const TensorId replacement = b.Input(F32({8, 1, 3}));
      const TensorId idx = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({1, 3})));
      b.CacheStore(x, replacement, idx);
    }
    if (mutation == 9) b.Output(b.Add(post, post));
    TensorId pre = b.Mul(run(0), sub(scale, 1, 0));
    pre = b.Scale(b.Unary(b.Add(pre, sub(base, 4, 0)), ir::UnaryFn::kSigmoid), 1.0F,
                  mutation == 2 ? 1e-4F : 1e-5F);
    if (mutation == 8) b.SetComponent(2);
    const TensorId folded = b.HcPre(mutation == 4 ? b.Input(F32({8, 4, 3})) : x, pre,
                                    {.fold = ir::HcFold::kWeights, .streams = 4});
    const TensorId out = b.Mul(b.RmsNorm(folded, 1e-6F), norm_weight);
    for (const TensorId t : {out, post, comb}) b.Output(t);
    auto g = std::move(b).Finish();
    ASSERT_TRUE(g) << (g ? "" : g.error().detail);
    const MatchContext c(*g, selection, nullptr);
    EXPECT_EQ(fusions::Dsv4HcPreMatch(c, g->producer(out)).has_value(), mutation == 0);
    if (mutation != 0) continue;
    Policy on;
    on.enabled = {"dsv4_hc_pre"};
    const SelectionView enabled(registry, kDevice, on, false, false);
    auto rewritten = ApplyPatterns(*g, enabled, nullptr);
    ASSERT_TRUE(rewritten) << (rewritten ? "" : rewritten.error().detail);
    ASSERT_EQ(rewritten->applied.size(), 1U);
    EXPECT_EQ(rewritten->applied[0].pattern, "dsv4_hc_pre");
    const auto& node =
        rewritten->graph.node(rewritten->graph.producer(rewritten->graph.outputs()[0]));
    EXPECT_EQ(node.key, ir::Fused(FusedKind::kDsv4HcPre));
    EXPECT_EQ(node.nout, 3);
    for (const TensorId t : node.outputs())
      EXPECT_TRUE(ir::IsContiguous(rewritten->graph.tensor(t)));
  }
}

TEST(PlannerPatternTest, Dsv4HcPostPreservesBothOutputsAcrossComponentBoundaries) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.dsv4_hc_post", Any(ir::Fused(FusedKind::kHcNormPost))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{};
  const SelectionView selection(registry, kDevice, policy, false, false);
  for (int mutation = 0; mutation <= 9; ++mutation) {
    SCOPED_TRACE(mutation);
    ir::Builder b;
    b.SetComponent(1);
    b.SetOwner(0);
    const auto input = [&](int strided, std::uint64_t width, std::uint64_t inner) {
      const auto root = b.Input(F32({width * (mutation == strided ? 2U : 1U), inner, 3}));
      return mutation == strided
                 ? b.View(root, ir::ShapeOf({width, inner, 3}), b.tensor(root).nb, 0)
                 : root;
    };
    const TensorId x = input(6, 8, 4);
    const TensorId y = b.Reshape(b.Input(F32({mutation == 7 ? 16U : 8U, 3})),
                                 ir::ShapeOf({mutation == 7 ? 16U : 8U, 3}));
    const TensorId yy = mutation == 7 ? b.View(y, ir::ShapeOf({8, 3}), b.tensor(y).nb, 0) : y;
    const TensorId p_root = b.Input(F32({mutation == 8 ? 8U : 4U, 3}));
    const TensorId p =
        mutation == 8 ? b.View(p_root, ir::ShapeOf({4, 3}), b.tensor(p_root).nb, 0) : p_root;
    const TensorId c = mutation == 1 ? TensorId{} : input(9, 4, 4);
    const TensorId post = b.HcPost(yy, x, p, c, {.streams = 4});
    b.SetComponent(2);
    if (mutation == 2) b.SetJoined();
    const TensorId flat = mutation == 3 ? post : b.Reshape(post, ir::ShapeOf({32, 3}));
    if (mutation == 4) b.Keep(flat);
    if (mutation == 5) b.SetComponent(3);
    const TensorId norm = b.RmsNorm(flat, 1e-6F);
    b.Output(post);
    b.Output(norm);
    auto g = std::move(b).Finish();
    ASSERT_TRUE(g) << (g ? "" : g.error().detail);
    EXPECT_EQ(fusions::Dsv4HcPostMatch(MatchContext(*g, selection, nullptr), g->producer(norm))
                  .has_value(),
              mutation == 0);
    if (mutation != 0) continue;
    Policy on;
    on.enabled = {"dsv4_hc_post"};
    const SelectionView enabled(registry, kDevice, on, false, false);
    auto rewritten = ApplyPatterns(*g, enabled, nullptr);
    ASSERT_TRUE(rewritten) << (rewritten ? "" : rewritten.error().detail);
    ASSERT_EQ(rewritten->applied.size(), 1U);
    EXPECT_EQ(rewritten->applied[0].pattern, "dsv4_hc_post");
    EXPECT_EQ(rewritten->graph.producer(rewritten->graph.outputs()[0]),
              rewritten->graph.producer(rewritten->graph.outputs()[1]));
  }
}

TEST(PlannerPatternTest, Dsv4CompressorDefinitionRejectsMalformedTypedOperands) {
  for (int mutation = 0; mutation <= 7; ++mutation) {
    SCOPED_TRACE(mutation);
    ir::Builder b;
    const std::uint64_t channels = mutation == 3 ? 7 : 8;
    const TensorId state = b.Input(F32({channels, 8}));
    const TensorId state_score = b.Input(F32({channels, mutation == 2 ? 7U : 8U}));
    const TensorId current = b.Input(F32({channels, 3}));
    const TensorId current_score = b.Input(
        ir::Dense(mutation == 1 ? Encoding::kF16 : Encoding::kF32, ir::ShapeOf({channels, 3})));
    const TensorId indices = b.Input(ir::Dense(mutation == 4 ? Encoding::kI64 : Encoding::kI32,
                                               ir::ShapeOf({mutation == 5 ? 15U : 16U})));
    const ir::CompressorParams params{.ratio = mutation == 0 ? 0U : 4U, .overlap = true};
    ir::ParamBytes raw = ir::ToBytes(params);
    if (mutation == 6) raw[4] = std::byte{2};  // noncanonical bool
    if (mutation == 7) raw[5] = std::byte{1};  // reserved byte
    b.FusedBytes(FusedKind::kDsv4Compress,
                 std::array{state, state_score, current, current_score, indices}, raw, {});
    EXPECT_FALSE(b.ok());
  }
}

TEST(PlannerPatternTest, Dsv4CompressorChecksTheEntireCanonicalAssemblyAndReadOrder) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.dsv4_compress", Any(ir::Fused(FusedKind::kDsv4Compress))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{};
  const SelectionView selection(registry, kDevice, policy, false, false);
  for (bool overlap : {false, true}) {
    for (int mutation = 0; mutation <= 9; ++mutation) {
      if (!overlap && mutation >= 1 && mutation <= 3) continue;
      SCOPED_TRACE(overlap);
      SCOPED_TRACE(mutation);
      constexpr std::uint64_t kDim = 8, kRatio = 4, kBlocks = 2, kRows = 3;
      const std::uint64_t channels = kDim * (overlap ? 2 : 1);
      const std::uint64_t reads = kRatio * kBlocks;
      ir::Builder b;
      b.SetComponent(1);
      b.SetOwner(0);
      const TensorId state = b.State(F32({channels, 8}), ir::StateRef{});
      const TensorId state_score = b.State(F32({channels, 8}), ir::StateRef{.tensor = 1});
      const TensorId current = b.Input(F32({channels, kRows}));
      const TensorId current_score = b.Input(F32({channels, kRows}));
      const auto index = [&]() {
        return b.Input(ir::Dense(mutation == 8 ? Encoding::kI64 : Encoding::kI32,
                                 ir::ShapeOf({reads * (overlap ? 2 : 1)})));
      };
      const TensorId indices = index();
      const TensorId other_indices = mutation == 4 ? index() : indices;
      const auto side = [&](TensorId saved, TensorId fresh, bool scores) {
        TensorId source = b.Concat(saved, fresh, 1);
        if (overlap) {
          const TensorId zero = b.Fill(
              ir::ShapeOf({channels, 1}),
              scores ? (mutation == 1 ? -1000.0F : -std::numeric_limits<float>::infinity()) : 0.0F,
              Encoding::kF32);
          source = b.Concat(source, zero, 1);
        }
        const TensorId picked = b.GetRows(source, scores ? other_indices : indices);
        if (!scores && mutation == 5) b.Keep(picked);
        if (!scores && mutation == 9)
          b.Keep(b.View(picked, ir::ShapeOf({channels}),
                        {4, channels * 4, channels * 4, channels * 4}, 0));
        if (!overlap) return b.Reshape(picked, ir::ShapeOf({kDim, kRatio, kBlocks}));
        const std::array<std::uint64_t, 4> strides{4, channels * 4, channels * 4 * reads,
                                                   channels * 4 * reads};
        const auto half = [&](std::uint64_t offset) {
          return b.Reshape(b.Cont(b.View(picked, ir::ShapeOf({kDim, reads}), strides, offset)),
                           ir::ShapeOf({kDim, kRatio, kBlocks}));
        };
        const TensorId previous = half(0);
        const TensorId next = half(mutation == 2 ? 0 : reads * channels * 4 + kDim * 4);
        return mutation == 3 ? b.Concat(next, previous, 1) : b.Concat(previous, next, 1);
      };
      const TensorId values = side(state, current, false);
      if (mutation == 6) {
        const TensorId replacement = b.Input(F32({channels, 1}));
        const TensorId cell = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({1})));
        b.CacheStore(state, replacement, cell);
      }
      const TensorId scores = side(state_score, current_score, true);
      if (mutation == 7) b.SetJoined();
      const TensorId out = b.Pool(values, scores,
                                  {.ratio = static_cast<std::uint32_t>(kRatio * (overlap ? 2 : 1)),
                                   .kind = ir::PoolKind::kSoftmax});
      b.Output(out);
      auto g = std::move(b).Finish();
      if (mutation == 8) {
        // The IR itself refuses I64 gathers, before pattern selection.
        EXPECT_FALSE(g);
        continue;
      }
      ASSERT_TRUE(g) << (g ? "" : g.error().detail);
      const MatchContext c(*g, selection, nullptr);
      EXPECT_EQ(fusions::Dsv4CompressMatch(c, g->producer(out)).has_value(), mutation == 0);
      if (mutation != 0) continue;
      Policy on;
      on.enabled = {"dsv4_compress"};
      const SelectionView enabled(registry, kDevice, on, false, false);
      auto rewritten = ApplyPatterns(*g, enabled, nullptr);
      ASSERT_TRUE(rewritten) << (rewritten ? "" : rewritten.error().detail);
      ASSERT_EQ(rewritten->applied.size(), 1U);
      EXPECT_EQ(rewritten->applied[0].pattern, "dsv4_compress");
      const auto& result =
          rewritten->graph.node(rewritten->graph.producer(rewritten->graph.outputs()[0]));
      EXPECT_EQ(result.key, ir::Fused(FusedKind::kDsv4Compress));
      EXPECT_EQ(result.Params<ir::CompressorParams>().ratio, kRatio);
      EXPECT_EQ(result.Params<ir::CompressorParams>().overlap, overlap);
      EXPECT_EQ(result.nsrc, 5U);
      auto disabled = ApplyPatterns(*g, selection, nullptr);
      ASSERT_TRUE(disabled);
      EXPECT_TRUE(disabled->applied.empty());
    }
  }
}

TEST(PlannerPatternTest, HyperConnectionMixChecksItsStreamViewsAndFinalScale) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.hc_norm", Any(ir::Fused(FusedKind::kHcNorm))));
  implementations.push_back(Fake("t.hc_mix_norm", Any(ir::Fused(FusedKind::kHcMixNorm))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{};
  const SelectionView selection(registry, kDevice, policy, false, false);
  for (int mutation : {0, 1, 2, 3, 4}) {
    SCOPED_TRACE(mutation);
    ir::Builder b;
    const TensorId x =
        mutation == 4 ? b.State(F32({4, 2, 3}), ir::StateRef{}) : b.Input(F32({4, 2, 3}));
    const TensorId norm = b.Weight(F32({8}), W(0));
    const TensorId down = b.Weight(F32({8, 2}), W(1));
    const TensorId up = b.Weight(F32({2, 8}), W(2));
    const TensorId rms = b.RmsNorm(x, 1e-6F);
    const TensorId xn = b.Mul(b.Reshape(rms, ir::ShapeOf({8, 3})), norm);
    if (mutation == 4) {
      const TensorId replacement = b.Input(F32({4, 2, 3}));
      const TensorId ids = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({2})));
      b.CacheStore(x, replacement, ids);
    }
    const TensorId low = b.Unary(b.Scale(b.MatMul(down, xn), 0.5F), ir::UnaryFn::kSilu);
    const TensorId gate = b.Unary(b.MatMul(up, low), ir::UnaryFn::kSigmoid);
    const TensorId gated = b.Reshape(b.Mul(xn, gate), ir::ShapeOf({4, 2, 3}));
    const auto first = b.View(gated, ir::ShapeOf({4, 3}), {4, 32, 96, 96}, 0);
    const TensorId packed = b.Cont(first);
    const TensorId second =
        b.View(gated, ir::ShapeOf({4, 3}), {4, 32, 96, 96}, mutation == 1 ? 0 : 16);
    const TensorId mixed =
        b.Scale(b.Add(packed, second), mutation == 2 ? 1.0F : 0.5F, mutation == 3 ? 1.0F : 0.0F);
    b.Output(mixed);
    const auto g = Finish(b);
    const MatchContext c(g, selection, nullptr);
    EXPECT_EQ(fusions::HcMixMatch(c, g.producer(rms)).has_value(), mutation == 0);
  }
}

TEST(PlannerPatternTest, HyperConnectionCombineRejectsUnrepresentedScaleParameters) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.hc_combine", Any(ir::Fused(FusedKind::kHcCombine))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{};
  const SelectionView selection(registry, kDevice, policy, false, false);
  for (int mutation : {0, 1, 2, 3}) {
    SCOPED_TRACE(mutation);
    ir::Builder b;
    const TensorId streams = b.Input(F32({4, 2, 3}));
    const TensorId out = b.Input(F32({4, 3}));
    const TensorId logits = b.Input(F32({2, 3}));
    const TensorId repeated =
        b.Repeat(b.Reshape(out, ir::ShapeOf({4, 1, 3})), ir::ShapeOf({4, 2, 3}));
    const TensorId scaled =
        b.Scale(logits, mutation == 3 ? 0.5000001F : 0.5F, mutation == 1 ? 1.0F : 0.0F);
    const TensorId weights =
        b.Scale(b.Unary(scaled, ir::UnaryFn::kSigmoid), 2.0F, mutation == 2 ? 1.0F : 0.0F);
    const TensorId mixed =
        b.Add(streams, b.Mul(repeated, b.Reshape(weights, ir::ShapeOf({1, 2, 3}))));
    b.Output(mixed);
    const auto g = Finish(b);
    const MatchContext c(g, selection, nullptr);
    EXPECT_EQ(fusions::HcCombineMatch(c, g.producer(mixed)).has_value(), mutation == 0);
  }
}

TEST(PlannerPatternTest, HyperConnectionPrepRejectsLowRankScaleBias) {
  auto implementations = Everything();
  for (const FusedKind kind : {FusedKind::kHcPrep, FusedKind::kScaledUnary, FusedKind::kHcFold}) {
    implementations.push_back(
        Fake("t.hc_" + std::to_string(static_cast<int>(kind)), Any(ir::Fused(kind))));
  }
  implementations.push_back(Fake("t.any_product", Any(ir::Primitive(Op::kMatMul))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{};
  const SelectionView selection(registry, kDevice, policy, false, false);
  for (bool bias : {false, true}) {
    SCOPED_TRACE(bias);
    ir::Builder b;
    const TensorId x = b.Input(F32({4, 2, 3}));
    const TensorId norm = b.Weight(F32({8}), W(0));
    const TensorId down = b.Weight(F32({8, 2}), W(1));
    const TensorId up = b.Weight(F32({2, 8}), W(2));
    const TensorId rms = b.RmsNorm(x, 1e-6F);
    const TensorId xn = b.Mul(b.Reshape(rms, ir::ShapeOf({8, 3})), norm);
    const TensorId low =
        b.Unary(b.Scale(b.MatMul(down, xn), 0.5F, bias ? 1.0F : 0.0F), ir::UnaryFn::kSilu);
    const TensorId gated = b.Reshape(b.Mul(xn, b.Unary(b.MatMul(up, low), ir::UnaryFn::kSigmoid)),
                                     ir::ShapeOf({4, 2, 3}));
    const TensorId first = b.Cont(b.View(gated, ir::ShapeOf({4, 3}), {4, 32, 96, 96}, 0));
    const TensorId second = b.View(gated, ir::ShapeOf({4, 3}), {4, 32, 96, 96}, 16);
    b.Output(b.Scale(b.Add(first, second), 0.5F));
    const auto g = Finish(b);
    const MatchContext c(g, selection, nullptr);
    EXPECT_EQ(fusions::HcPrepMatch(c, g.producer(rms)).has_value(), !bias);
  }
}

TEST(PlannerPatternTest, GdnConvolutionChecksBothNormalizedSlicesAndHistoryTail) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.gdn_conv", Any(ir::Fused(FusedKind::kGdnConv))));
  implementations.push_back(Fake("t.gdn_history", Any(ir::Fused(FusedKind::kGdnHistory))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{};
  const SelectionView selection(registry, kDevice, policy, false, false);
  for (int mutation : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) {
    SCOPED_TRACE(mutation);
    ir::Builder b;
    const TensorId state = b.State(F32({18, 1}), ir::StateRef{});
    const TensorId rows = b.Input(F32({6, 2, 1}));
    const TensorId weights = b.Weight(F32({4, 6}), W(0));
    const TensorId input =
        b.Concat(b.Reshape(state, ir::ShapeOf({3, 6, 1})), b.Cont(b.Transpose(rows)), 0);
    if (mutation < 6) {
      const TensorId tail =
          b.View(input, ir::ShapeOf({3, 6, 1}), {4, 20, 120, 120}, mutation == 5 ? 0 : 8);
      const TensorId ids = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({1})));
      b.CacheStore(state, b.Reshape(b.Cont(tail), ir::ShapeOf({18, 1})), ids);
    }
    const TensorId convolution = b.SsmConv(input, weights);
    const TensorId conv = b.Unary(convolution, ir::UnaryFn::kSilu);
    std::vector<TensorId> qk;
    for (std::uint64_t i = 0; i < 2; ++i) {
      const TensorId view = b.View(conv, ir::ShapeOf({2, 1, 2, 1}), {4, 8, 24, 48},
                                   i == 0 && mutation == 4 ? 4 : i * 8);
      const TensorId norm = b.RmsNorm(b.Cont(view), i == 1 && mutation == 1 ? 0.25F : 1e-6F);
      qk.push_back(b.Scale(norm, i == 1 && mutation == 2 ? 2.0F : 0.5F,
                           i == 1 && mutation == 3 ? 1.0F : 0.0F));
      b.Output(qk.back());
    }
    b.Output(b.View(conv, ir::ShapeOf({2, 1, 2, 1}), {4, 8, 24, 48}, 16));
    if (mutation >= 6) {
      const TensorId q = b.Reshape(qk[0], ir::ShapeOf({2, 2}));
      const TensorId k = b.Reshape(qk[1], ir::ShapeOf({2, 2}));
      const std::uint8_t dim = mutation == 10 ? 1 : 0;
      const TensorId both = mutation == 7 ? b.Concat(k, q, dim) : b.Concat(q, k, dim);
      const TensorId v =
          b.Cont(b.View(conv, ir::ShapeOf({2, 2}), {4, 24, 48, 48}, mutation == 8 ? 0 : 16));
      b.Keep(mutation == 9 ? b.Concat(v, both, dim) : b.Concat(both, v, dim));
    }
    const auto g = Finish(b);
    const MatchContext c(g, selection, nullptr);
    EXPECT_EQ(fusions::GdnConvMatch(c, g.producer(convolution)).has_value(),
              mutation == 0 || mutation == 6);
  }
}

TEST(PlannerPatternTest, QsaSelectionChecksItsFixedWeightsMasksAndBias) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.qsa_attention", Any(ir::Fused(FusedKind::kQsaAttention))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{};
  const SelectionView selection(registry, kDevice, policy, false, false);
  for (int mutation : {0, 1, 2, 3, 4, 5, 6}) {
    SCOPED_TRACE(mutation);
    ir::Builder b;
    b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({4})), "positions.4");
    const TensorId q = b.Input(F32({2, 2, 1}));
    const TensorId k = b.Input(F32({2, 4, 1}));
    const TensorId v = b.Input(F32({2, 4, 1}));
    const TensorId queries = b.Convert(b.Input(F32({2, 2, 1})), Encoding::kBF16);
    const TensorId keys = b.Input(F32({2, 1, 2}));
    const TensorId seen = b.Input(F32({2, 1}));
    const TensorId causal = b.Input(F32({4, 1}));
    const TensorId mask = b.Input(F32({4, 1}));
    const TensorId bias = b.Scale(b.Clamp(seen, mutation == 4 ? -1.0F : -1e9F, 0.0F),
                                  mutation == 5 ? -2.0F : -1.0F, mutation == 6 ? 1.0F : 0.0F);
    const TensorId ones = b.Fill(ir::ShapeOf({2, 1, 1, 1}), mutation == 1 ? 2.0F : 1.0F);
    const TensorId score =
        b.IndexScore(queries, keys, ones, b.Reshape(bias, ir::ShapeOf({2, 1, 1, 1})));
    const TensorId spread =
        b.Repeat(b.Reshape(score, ir::ShapeOf({1, 2, 1})), ir::ShapeOf({2, 2, 1}));
    const TensorId top = b.TopK(b.Add(b.Reshape(spread, ir::ShapeOf({4, 1})), causal), 2);
    const TensorId selected = b.Fill(
        ir::ShapeOf({1, 4, 1, 1}), mutation == 2 ? 0.0F : -std::numeric_limits<float>::infinity());
    const TensorId zeros = b.Fill(ir::ShapeOf({1, 2, 1, 1}), mutation == 3 ? 1.0F : 0.0F);
    const TensorId stored = b.CacheStore(selected, zeros, top);
    const TensorId viewed_mask = b.View(mask, ir::ShapeOf({4, 1}), {4, 16, 16, 16}, 0);
    const TensorId attn = b.Attention(b.Permute(q, {0, 2, 1, 3}), k, v,
                                      b.Add(b.Reshape(stored, ir::ShapeOf({4, 1})), viewed_mask),
                                      {}, {.scale = 0.5F, .sparse_any = true, .n_kv_max = 2});
    b.Output(b.Reshape(attn, ir::ShapeOf({4, 1})));
    const auto g = Finish(b);
    const MatchContext c(g, selection, nullptr);
    EXPECT_EQ(fusions::QsaSelectMatch(c, g.producer(attn)).has_value(), mutation == 0);
  }
}

TEST(PlannerPatternTest, QsaPoolingDoesNotDropRotationFrequencyFactors) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.qsa_pool", Any(ir::Fused(FusedKind::kQsaPool))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{};
  const SelectionView selection(registry, kDevice, policy, false, false);
  for (bool frequencies : {false, true}) {
    SCOPED_TRACE(frequencies);
    ir::Builder b;
    b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({4})), "positions.4");
    const TensorId raw = b.Input(F32({2, 4}));
    const TensorId blocks =
        b.State(ir::Dense(Encoding::kBF16, ir::ShapeOf({2, 2})), ir::StateRef{});
    const TensorId norm = b.Weight(F32({2}), W(0));
    const TensorId positions = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({2})));
    const TensorId cells = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({2})));
    const TensorId freq = frequencies ? b.Input(F32({1})) : TensorId{};
    const TensorId pooled = b.Pool(raw, {}, {.ratio = 2, .kind = ir::PoolKind::kMean});
    const TensorId weighted = b.Mul(b.RmsNorm(pooled, 1e-6F), norm);
    const TensorId rotated =
        b.Rope(b.Reshape(weighted, ir::ShapeOf({2, 1, 2})), positions, freq, {.dims = 2});
    const TensorId rounded = b.Convert(rotated, Encoding::kBF16);
    const TensorId stored = b.CacheStore(blocks, b.Reshape(rounded, ir::ShapeOf({2, 2})), cells);
    b.Keep(stored);
    const auto g = Finish(b);
    const MatchContext c(g, selection, nullptr);
    EXPECT_EQ(fusions::QsaPoolMatch(c, g.producer(stored)).has_value(), !frequencies);
  }
}

TEST(PlannerPatternTest, SharedInputPreparationStopsAtAnInPlaceWrite) {
  for (bool bf16 : {false, true}) {
    SCOPED_TRACE(bf16);
    auto implementations = Everything();
    implementations.push_back(Fake("t.any_product", Any(ir::Primitive(Op::kMatMul))));
    implementations.push_back(Fake("t.convert", Any(ir::Primitive(Op::kConvert))));
    const auto registry = Registry(std::move(implementations));
    const Policy policy{};
    const SelectionView selection(registry, kDevice, policy, false, false);
    ir::Builder b;
    const std::uint64_t rows = bf16 ? 17 : 2;
    const TensorId x = b.State(F32({64, rows}), ir::StateRef{});
    const TensorId w =
        b.Weight(bf16 ? ir::Dense(Encoding::kBF16, ir::ShapeOf({64, 4})) : Q8(64, 4), W(0));
    const TensorId first = b.MatMul(w, x);
    const TensorId replacement = b.Input(F32({64, rows}));
    const TensorId indices = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({rows})));
    b.CacheStore(x, replacement, indices);
    const TensorId second = b.MatMul(w, x);
    b.Output(b.Scale(first, 1.0F));
    b.Output(b.Scale(second, 1.0F));
    const auto g = Finish(b);
    const MatchContext c(g, selection, nullptr);
    const auto match = bf16 ? fusions::Bf16InputsMatch(c, g.producer(first))
                            : fusions::VecQInputsMatch(c, g.producer(first));
    ASSERT_TRUE(match.has_value());
    EXPECT_FALSE(std::ranges::contains(match->nodes, g.producer(second)));
  }
}

TEST(PlannerPatternTest, NormAndItsWeightFuse) {
  NormGraph t;
  t.b.Output(t.m);
  const ex::Registry r = Registry();
  const StepPlan p = PlanOf(Finish(t.b), r);
  EXPECT_EQ(Patterns(p), std::vector<std::string_view>{"rms_norm_mul"});
  EXPECT_EQ(Keys(p.graph), std::vector{ir::Fused(FusedKind::kRmsNormMul)});
  EXPECT_EQ(Names(p, r), std::vector<std::string>{"t.rms_norm_mul"});
  ASSERT_EQ(p.applied.size(), 1U);
  EXPECT_EQ(p.applied[0].replaced.size(), 2U);
  EXPECT_FALSE(p.tensors[t.n.v].valid());  // elided
  EXPECT_TRUE(p.tensors[t.m.v].valid());
  EXPECT_EQ(p.graph.outputs()[0], p.tensors[t.m.v]);
  EXPECT_EQ(p.graph.tensor(p.tensors[t.m.v]).shape(), ir::ShapeOf({64, 3}));
}

TEST(PlannerPatternTest, NothingFusesAnIntermediateReadElsewhere) {
  const ex::Registry r = Registry();
  const std::vector primitive = {ir::Primitive(Op::kRmsNorm), ir::Primitive(Op::kMul)};
  {
    NormGraph t;  // the norm also returned
    t.b.Output(t.m);
    t.b.Output(t.n);
    EXPECT_EQ(Keys(PlanOf(Finish(t.b), r).graph), primitive);
  }
  {
    NormGraph t;  // the norm kept past the step
    t.b.Output(t.m);
    t.b.Keep(t.n);
    EXPECT_EQ(Keys(PlanOf(Finish(t.b), r).graph), primitive);
  }
  {
    NormGraph t;  // the norm read by another node too
    t.b.Output(t.m);
    t.b.Output(t.b.Scale(t.n, 2.0f));
    EXPECT_EQ(Keys(PlanOf(Finish(t.b), r).graph).size(), 3U);
  }
  {
    NormGraph t;  // a view of the norm
    t.b.Output(t.m);
    t.b.Output(t.b.Scale(t.b.Reshape(t.n, ir::ShapeOf({192})), 2.0f));
    EXPECT_EQ(Patterns(PlanOf(Finish(t.b), r)).size(), 0U);
  }
  {
    // A norm the graph rounds to F16: the fused kind's would not be.
    ir::Builder b;
    const TensorId x = b.Input(F32({64, 3}));
    ir::TensorDesc f16;
    f16.format = ir::Dense(Encoding::kF16, ir::ShapeOf({64, 3}));
    f16.nb = ir::NaturalStrides(f16.format);
    const TensorId n =
        b.Emit(ir::Primitive(Op::kRmsNorm), ir::ToBytes(ir::NormParams{.eps = 1e-6f}),
               std::array{x}, std::array{f16})[0];
    ASSERT_EQ(b.tensor(n).format.encoding, Encoding::kF16);
    b.Output(b.Mul(n, b.Weight(F32({64}), W(0))));
    EXPECT_TRUE(Patterns(PlanOf(Finish(b), r)).empty());
  }
}

// A rewrite moves its reads to the match's last node: never past a store
// into what they read (here the norm's input, an activation a store
// targets between the norm and its mul), nor a prepared input's snapshot
// past a store into its source.
TEST(PlannerPatternTest, NothingMovesAReadPastAWrite) {
  const ex::Registry r = Registry();
  const auto rows = [](ir::Builder& b) {
    return std::pair{b.Input(F32({64, 1})), b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({1})))};
  };
  for (const bool store : {false, true}) {
    ir::Builder b;
    const TensorId x = b.Scale(b.Input(F32({64, 3})), 2.0f);
    const TensorId n = b.RmsNorm(x, 1e-6f);
    const auto [src, idx] = rows(b);
    if (store) {
      b.CacheStore(x, src, idx);
    } else {
      b.Output(b.Scale(src, 1.0f));
    }
    b.Output(b.Mul(n, b.Weight(F32({64}), W(0))));
    EXPECT_EQ(Patterns(PlanOf(Finish(b), r)).size(), store ? 0U : 1U) << store;
  }
  for (const bool store : {false, true}) {
    ir::Builder b;
    const TensorId x = b.Scale(b.Input(F32({64, 2})), 2.0f);
    const TensorId first = b.MatMul(b.Weight(Q8(64, 32), W(0)), x);
    const auto [src, idx] = rows(b);
    if (store) {
      b.CacheStore(x, src, idx);
    } else {
      b.Output(b.Scale(src, 1.0f));
    }
    b.Output(b.Add(first, b.MatMul(b.Weight(Q8(64, 32), W(1)), x)));
    EXPECT_EQ(Patterns(PlanOf(Finish(b), r)).size(), store ? 0U : 1U) << store;
  }
}

TEST(PlannerPatternTest, NothingFusesWithoutAnImplementationOrWhenDisabled) {
  const std::vector primitive = {ir::Primitive(Op::kRmsNorm), ir::Primitive(Op::kMul)};
  {
    NormGraph t;
    t.b.Output(t.m);
    const ex::Registry r = Without({"t.rms_norm_mul"});
    const StepPlan p = PlanOf(Finish(t.b), r);
    EXPECT_EQ(Keys(p.graph), primitive);
    EXPECT_EQ(Names(p, r), (std::vector<std::string>{"t.rms_norm", "t.mul"}));
  }
  {
    NormGraph t;
    t.b.Output(t.m);
    EXPECT_EQ(Keys(PlanOf(Finish(t.b), Registry(), Disabling({"rms_norm_mul"})).graph), primitive);
  }
  {
    // A factor of the norm's own shape may be the mul's first operand.
    ir::Builder b;
    const TensorId x = b.Input(F32({64, 3}));
    const TensorId full = b.Input(F32({64, 3}));
    b.Output(b.Mul(full, b.RmsNorm(x, 1e-6f)));
    const StepPlan p = PlanOf(Finish(b), Registry());
    EXPECT_EQ(Patterns(p), std::vector<std::string_view>{"rms_norm_mul"});
  }
}

TEST(PlannerPatternTest, TheLongerChainComesFirst) {
  const auto graph = [](std::uint64_t width) {
    ir::Builder b;
    const TensorId x = b.Input(F32({width, 2, 3}));
    const TensorId pos = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({3})));
    const TensorId w = b.Weight(F32({width}), W(0));
    const ir::RopeParams rope{
        .mode = ir::RopeMode::kNeox, .dims = static_cast<std::uint32_t>(width), .base = 10000.0f};
    b.Output(b.Rope(b.Mul(b.RmsNorm(x, 1e-6f), w), pos, {}, rope));
    return Finish(b);
  };
  const ex::Registry r = Registry();
  EXPECT_EQ(Patterns(PlanOf(graph(64), r)), std::vector<std::string_view>{"rms_norm_mul_rope"});
  EXPECT_EQ(Keys(PlanOf(graph(64), r).graph), std::vector{ir::Fused(FusedKind::kRmsNormMulRope)});
  // Disabled, or at a width its implementation does not take: the norm and
  // its weight still fuse, the rotation stays.
  EXPECT_EQ(Patterns(PlanOf(graph(64), r, Disabling({"rms_norm_mul_rope"}))),
            std::vector<std::string_view>{"rms_norm_mul"});
  const StepPlan wide = PlanOf(graph(128), r);
  EXPECT_EQ(Patterns(wide), std::vector<std::string_view>{"rms_norm_mul"});
  EXPECT_EQ(Keys(wide.graph),
            (std::vector{ir::Fused(FusedKind::kRmsNormMul), ir::Primitive(Op::kRope)}));
}

TEST(PlannerPatternTest, NormThenResidualAdd) {
  const auto graph = [](bool broadcast) {
    NormGraph t;
    const TensorId residual = t.b.Input(broadcast ? F32({64}) : F32({64, 3}));
    t.b.Output(t.b.Add(t.m, residual));
    return Finish(t.b);
  };
  const ex::Registry r = Registry();
  EXPECT_EQ(Patterns(PlanOf(graph(false), r)), std::vector<std::string_view>{"rms_norm_mul_add"});
  // A residual broadcast over the rows is not the fused kind's.
  EXPECT_EQ(Patterns(PlanOf(graph(true), r)), std::vector<std::string_view>{"rms_norm_mul"});
}

// Gate and up products of x, and the GLU over them.
ir::Graph GluGraph(std::uint64_t rows, bool read_gate, bool other_input, bool accumulate) {
  ir::Builder b;
  const TensorId x = b.Input(F32({64, rows}));
  const TensorId x2 = b.Input(F32({64, rows}));
  const TensorId wg = b.Weight(Q8(64, 32), W(0));
  const TensorId wu = b.Weight(Q8(64, 32), W(1));
  const TensorId gate =
      b.MatMul(wg, x, {.accumulate = accumulate ? ir::Accumulate::kF32 : ir::Accumulate::kDefault});
  const TensorId up = b.MatMul(wu, other_input ? x2 : x);
  b.Output(b.Glu(gate, up, {.kind = ir::GluKind::kGeGluTanh}));
  if (read_gate) b.Output(gate);
  return Finish(b);
}

TEST(PlannerPatternTest, GateAndUpProductsJoinTheirGlu) {
  const ex::Registry r = Registry();
  const StepPlan p = PlanOf(GluGraph(1, false, false, false), r);
  EXPECT_EQ(Patterns(p), std::vector<std::string_view>{"mul_mat_glu"});
  EXPECT_EQ(Names(p, r), std::vector<std::string>{"t.mul_mat_glu"});
  const ir::Node& fused = p.graph.nodes().back();
  EXPECT_EQ(fused.Params<ir::GluParams>().kind, ir::GluKind::kGeGluTanh);
  // Not past the implementation's rows (the two products then share their
  // prepared input instead), not with the gate read elsewhere, not over
  // two inputs, not where a product asks for F32 sums.
  EXPECT_EQ(Patterns(PlanOf(GluGraph(2, false, false, false), r)),
            std::vector<std::string_view>{"prepared_inputs"});
  EXPECT_TRUE(
      Patterns(PlanOf(GluGraph(1, true, false, false), r, Disabling({"prepared_inputs"}))).empty());
  EXPECT_TRUE(Patterns(PlanOf(GluGraph(1, false, true, false), r)).empty());
  EXPECT_TRUE(
      Patterns(PlanOf(GluGraph(1, false, false, true), r, Disabling({"prepared_inputs"}))).empty());
}

// x [8, 2, rows] rotated, its rows stored at idx; `owners` stores of
// row slices.
ir::Graph RopeStoreGraph(int owners) {
  ir::Builder b;
  const std::uint64_t rows = 4;
  const TensorId x = b.Input(F32({8, 2, rows}));
  const TensorId pos = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({rows})));
  const ir::RopeParams rope{.mode = ir::RopeMode::kNeox, .dims = 8, .base = 10000.0f};
  const TensorId r = b.Reshape(b.Rope(x, pos, {}, rope), ir::ShapeOf({16, rows}));
  const std::uint64_t each = rows / static_cast<std::uint64_t>(owners);
  for (int o = 0; o < owners; ++o) {
    const TensorId cache =
        b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({16, 32})),
                {.stage = 0, .tensor = 0, .layer = 0, .segment = static_cast<std::uint8_t>(o)});
    const TensorId idx = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({each})));
    const TensorId slice = owners == 1
                               ? r
                               : b.View(r, ir::ShapeOf({16, each}), b.tensor(r).nb,
                                        static_cast<std::uint64_t>(o) * each * b.tensor(r).nb[1]);
    b.Output(b.CacheStore(cache, slice, idx));
  }
  return Finish(b);
}

TEST(PlannerPatternTest, ARotationStoredWhole) {
  const ex::Registry r = Registry();
  const StepPlan one = PlanOf(RopeStoreGraph(1), r);
  EXPECT_EQ(Patterns(one), std::vector<std::string_view>{"rope_store"});
  EXPECT_EQ(Keys(one.graph), std::vector{ir::Fused(FusedKind::kRopeStore)});
  // The store's result is a view of its cache, as the fused one's is.
  EXPECT_TRUE(one.graph.tensor(one.graph.outputs()[0]).is_view());
  // Two owners' slices: the rotation has two readers.
  const StepPlan two = PlanOf(RopeStoreGraph(2), r, Disabling({"grouped_stores"}));
  EXPECT_TRUE(Patterns(two).empty());
  EXPECT_TRUE(Patterns(PlanOf(RopeStoreGraph(1), r, Disabling({"rope_store"}))).empty());
}

// Products of one input: `weights` of them, and one of another input.
ir::Graph SharedGraph(int weights, std::uint64_t rows) {
  ir::Builder b;
  const TensorId x = b.Input(F32({64, rows}));
  for (int i = 0; i < weights; ++i) {
    b.Output(b.MatMul(b.Weight(Q8(64, 32), W(static_cast<std::uint16_t>(i))), x));
  }
  b.Output(b.MatMul(b.Weight(Q8(64, 32), W(9)), b.Input(F32({64, rows}))));
  return Finish(b);
}

TEST(PlannerPatternTest, ProductsOfOneInputShareItsPreparation) {
  const ex::Registry r = Registry();
  const StepPlan p = PlanOf(SharedGraph(3, 2), r);
  EXPECT_EQ(Patterns(p), std::vector<std::string_view>{"prepared_inputs"});
  EXPECT_EQ(Names(p, r),
            (std::vector<std::string>{"t.q8_1", "t.mul_mat.prepared", "t.mul_mat.prepared",
                                      "t.mul_mat.prepared", "t.mul_mat.vector"}));
  ASSERT_EQ(p.applied.size(), 1U);
  EXPECT_EQ(p.applied[0].replaced.size(), 3U);
  EXPECT_EQ(p.applied[0].made.size(), 4U);
  const ir::TensorDesc& prepared = p.graph.tensor(p.graph.nodes()[p.applied[0].made[0].v].out[0]);
  EXPECT_EQ(prepared.format.encoding, Encoding::kQ8_1);
  // One product alone, or past the prepared products' rows: none.
  EXPECT_TRUE(Patterns(PlanOf(SharedGraph(1, 2), r)).empty());
  const StepPlan wide = PlanOf(SharedGraph(3, 16), r);
  EXPECT_TRUE(Patterns(wide).empty());
  EXPECT_EQ(Names(wide, r)[0], "t.mul_mat.tile");
  EXPECT_EQ(wide.steps[0].group, "t.mul_mat");
}

// ------------------------------------------------------------ batches

// Stores of rows into `caches` caches, each kept past the step.
ir::Graph KeptStoresGraph(int caches) {
  ir::Builder b;
  const TensorId src = b.Input(F32({16, 2}));
  const TensorId idx = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({2})));
  for (int i = 0; i < caches; ++i) {
    const TensorId cache =
        b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({16, 8})),
                {.stage = 0, .tensor = static_cast<std::uint16_t>(i), .layer = 0, .segment = 0});
    b.Keep(b.CacheStore(cache, src, idx));
  }
  return Finish(b);
}

std::vector<std::size_t> StepSizes(const StepPlan& p) {
  std::vector<std::size_t> out;
  for (const Selected& s : p.steps) out.push_back(s.nodes.size());
  return out;
}

TEST(PlannerBatchTest, ConsecutiveStoresRunAsOne) {
  // Stores of rows into `caches` caches, a scale before the third when
  // `split`.
  const auto graph = [](int caches, bool split) {
    ir::Builder b;
    const TensorId src = b.Input(F32({16, 2}));
    const TensorId idx = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({2})));
    for (int i = 0; i < caches; ++i) {
      const TensorId cache =
          b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({16, 8})),
                  {.stage = 0, .tensor = static_cast<std::uint16_t>(i), .layer = 0, .segment = 0});
      if (split && i == 2) b.Output(b.Scale(src, 2.0f));
      b.CacheStore(cache, src, idx);
    }
    return Finish(b);
  };
  const ex::Registry r = Registry();
  const StepPlan three = PlanOf(graph(3, false), r);
  EXPECT_EQ(StepSizes(three), std::vector<std::size_t>{3});
  EXPECT_EQ(three.steps[0].batch, "grouped_stores");
  EXPECT_EQ(Names(three, r), std::vector<std::string>{"t.set_rows.grouped"});
  EXPECT_EQ(three.steps[0].singles.size(), 3U);
  EXPECT_TRUE(Patterns(three).empty());  // a batch rewrites nothing
  EXPECT_EQ(StepSizes(PlanOf(graph(3, false), r, Disabling({"grouped_stores"}))),
            (std::vector<std::size_t>{1, 1, 1}));
  EXPECT_EQ(StepSizes(PlanOf(graph(4, true), r)), (std::vector<std::size_t>{2, 1, 2}));
  // The implementation's batch bound.
  std::vector<ex::Implementation> two = Everything();
  for (auto& i : two) {
    if (i.name == "t.set_rows.grouped") i.capability->batch = 2;
  }
  EXPECT_EQ(StepSizes(PlanOf(graph(3, false), Registry(std::move(two)))),
            (std::vector<std::size_t>{2, 1}));
  // Stores read after the step stay apart.
  EXPECT_EQ(StepSizes(PlanOf(KeptStoresGraph(3), r)), (std::vector<std::size_t>{1, 1, 1}));
  // So does a store of rows read from a cache an earlier one writes.
  ir::Builder b;
  const TensorId src = b.Input(F32({16, 2}));
  const TensorId idx = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({2})));
  const TensorId first = b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({16, 8})),
                                 {.stage = 0, .tensor = 0, .layer = 0, .segment = 0});
  const TensorId second = b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({16, 8})),
                                  {.stage = 0, .tensor = 1, .layer = 0, .segment = 0});
  b.CacheStore(first, src, idx);
  b.CacheStore(second, b.View(first, ir::ShapeOf({16, 2}), b.tensor(first).nb, 0), idx);
  EXPECT_EQ(StepSizes(PlanOf(Finish(b), r)), (std::vector<std::size_t>{1, 1}));
}

// ------------------------------------------------------------ selection

TEST(PlannerSelectionTest, MeasuredRanksOfTheDeviceClassComeFirst) {
  std::vector<ex::Implementation> v = Everything();
  v.push_back(Fake("t.scale.slow", Any(ir::Primitive(Op::kScale))));
  const ex::Registry r = Registry(std::move(v));
  const auto graph = [] {
    ir::Builder b;
    b.Output(b.Scale(b.Input(F32({4})), 2.0f));
    return Finish(b);
  };
  // Unmeasured: the ladder, then the name.
  EXPECT_EQ(Names(PlanOf(graph(), r), r), std::vector<std::string>{"t.scale"});
  Policy measured{.ranks = {{.device_class = "test", .implementation = "t.scale.slow", .rank = 0}}};
  EXPECT_EQ(Names(PlanOf(graph(), r, measured), r), std::vector<std::string>{"t.scale.slow"});
  Policy elsewhere{
      .ranks = {{.device_class = "other", .implementation = "t.scale.slow", .rank = 0}}};
  EXPECT_EQ(Names(PlanOf(graph(), r, elsewhere), r), std::vector<std::string>{"t.scale"});
}

TEST(PlannerSelectionTest, AnImplementationNeedingTuningNeedsItsShape) {
  std::vector<ex::Implementation> v = Everything();
  v.push_back(Fake("t.mul_mat.pinned", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                                      .operands = kProductRules,
                                                      .priority = 9,
                                                      .needs_tuning = true}));
  const ex::Registry r = Registry(std::move(v));
  const auto graph = [] {
    ir::Builder b;
    b.Output(b.MatMul(b.Weight(Q8(64, 32), W(0)), b.Input(F32({64, 145}))));
    return Finish(b);
  };
  EXPECT_EQ(Names(PlanOf(graph(), r), r), std::vector<std::string>{"t.mul_mat.tile"});
  DeviceFacts tuned = kDevice;
  tuned.tuned = {{.implementation = "t.mul_mat.pinned", .rows = 145, .k = 64, .n = 32}};
  auto p = Plan(graph(), r, tuned, {});
  ASSERT_TRUE(p.has_value());
  EXPECT_EQ(Names(*p, r), std::vector<std::string>{"t.mul_mat.pinned"});
  tuned.tuned[0].rows = 144;
  EXPECT_EQ(Names(*Plan(graph(), r, tuned, {}), r), std::vector<std::string>{"t.mul_mat.tile"});
}

TEST(PlannerSelectionTest, RowInvariantPlansTakeRowInvariantImplementations) {
  std::vector<ex::Implementation> v;
  for (ex::Implementation& i : Everything()) {
    if (i.name != "t.mul_mat_glu" && i.name != "t.q8_1") i.capability->row_invariant = true;
    if (i.name == "t.mul_mat.vector") i.capability->row_invariant = false;
    v.push_back(std::move(i));
  }
  v.push_back(Fake("t.mul_mat.rows", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                                    .operands = kProductRules,
                                                    .shape = {.max_rows = 8},
                                                    .row_invariant = true}));
  const ex::Registry r = Registry(std::move(v));
  // Fusion of the GLU and the shared preparation need implementations that
  // are not row-invariant: a verify's plan takes neither.
  const StepPlan plain = PlanOf(GluGraph(1, false, false, false), r);
  EXPECT_EQ(Patterns(plain), std::vector<std::string_view>{"mul_mat_glu"});
  const StepPlan verify = PlanOf(GluGraph(1, false, false, false), r, {}, {.row_invariant = true});
  EXPECT_TRUE(Patterns(verify).empty());
  EXPECT_EQ(Names(verify, r),
            (std::vector<std::string>{"t.mul_mat.rows", "t.mul_mat.rows", "t.glu"}));
  // The policy may ask for it too.
  EXPECT_EQ(Names(PlanOf(GluGraph(1, false, false, false), r, Policy{.row_invariant = true}), r),
            Names(verify, r));
  // The option preserving several one-row owners' scalar arithmetic must
  // not turn ordinary multirow verifications into scalar products. Exact
  // verifications still select row-invariant kernels independently.
  const Policy owners{.enabled = {"owner_row_invariant"}};
  for (const auto rows : {std::array{1U, 1U}, std::array{1U, 3U}, std::array{3U, 3U}}) {
    StepShape shape{.phase = Phase::kVerify,
                    .segments = {{.slot = 0, .rows = rows[0]}, {.slot = 1, .rows = rows[1]}}};
    const auto ordinary =
        PlanOf(GluGraph(rows[0] + rows[1], false, false, false), r, owners, {.shape = &shape});
    EXPECT_EQ(ordinary.owner_row_invariant, rows[0] == 1 && rows[1] == 1);
    const auto exact = PlanOf(GluGraph(rows[0] + rows[1], false, false, false), r, owners,
                              {.row_invariant = true, .shape = &shape});
    EXPECT_EQ(Names(exact, r), Names(verify, r));
    EXPECT_EQ(exact.owner_row_invariant, ordinary.owner_row_invariant);
  }
  // And capture: an implementation that cannot be recorded is left out.
  std::vector<ex::Implementation> w = Everything();
  for (auto& i : w) {
    if (i.name == "t.rms_norm_mul") i.capability->capturable = false;
  }
  const ex::Registry rw = Registry(std::move(w));
  NormGraph t;
  t.b.Output(t.m);
  EXPECT_TRUE(Patterns(PlanOf(Finish(t.b), rw, {}, {.capturable = true})).empty());
}

TEST(PlannerSelectionTest, AFusedNodeWithoutImplementationDecomposes) {
  ir::Builder b;
  const TensorId x = b.Input(F32({64, 3}));
  const TensorId w = b.Weight(F32({64}), W(0));
  b.Output(b.Fused(FusedKind::kRmsNormMul, std::array{x, w}, ir::NormParams{.eps = 1e-6f})[0]);
  const ex::Registry r = Without({"t.rms_norm_mul"});
  const StepPlan p = PlanOf(Finish(b), r);
  EXPECT_EQ(p.decomposed, std::vector<std::string>{"fused.rms_norm_mul"});
  EXPECT_EQ(Names(p, r), (std::vector<std::string>{"t.rms_norm", "t.mul"}));
  EXPECT_EQ(p.graph.outputs().size(), 1U);
  EXPECT_TRUE(p.tensors[x.v].valid());
}

TEST(PlannerSelectionTest, ANodeNothingComputesRefusesThePlan) {
  NormGraph t;
  t.b.Output(t.m);
  const auto p = Plan(Finish(t.b), Without({"t.rms_norm", "t.rms_norm_mul"}), kDevice, {});
  ASSERT_FALSE(p.has_value());
  EXPECT_EQ(p.error().error, PlanError::kNoImplementation);
  EXPECT_NE(p.error().detail.find("rms_norm"), std::string::npos);
  // A policy that breaks its rules refuses too.
  NormGraph u;
  u.b.Output(u.m);
  const auto bad = Plan(Finish(u.b), Registry(), kDevice, Policy{.disabled = {"nope"}});
  ASSERT_FALSE(bad.has_value());
  EXPECT_EQ(bad.error().error, PlanError::kPolicy);
}

// A small model step: norms, products, a GLU, stores and a residual.
ir::Graph LayerGraph(std::uint64_t rows) {
  ir::Builder b;
  TensorId x = b.Input(F32({64, rows}));
  const TensorId idx = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({rows})));
  for (std::uint32_t layer = 0; layer < 2; ++layer) {
    const TensorId h = b.Mul(b.RmsNorm(x, 1e-6f), b.Weight(F32({64}), W(0, layer)));
    const TensorId k = b.MatMul(b.Weight(Q8(64, 64), W(1, layer)), h);
    const TensorId v = b.MatMul(b.Weight(Q8(64, 64), W(2, layer)), h);
    for (std::uint16_t s = 0; s < 2; ++s) {
      const TensorId cache = b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({64, 16})),
                                     {.stage = 0, .tensor = s, .layer = layer, .segment = 0});
      b.CacheStore(cache, s == 0 ? k : v, idx);
    }
    const TensorId gate = b.MatMul(b.Weight(Q8(64, 64), W(3, layer)), h);
    const TensorId up = b.MatMul(b.Weight(Q8(64, 64), W(4, layer)), h);
    const TensorId glu = b.Glu(gate, up, {.kind = ir::GluKind::kGeGluTanh});
    const TensorId d = b.MatMul(b.Weight(Q8(64, 64), W(5, layer)), glu);
    x = b.Add(b.Mul(b.RmsNorm(d, 1e-6f), b.Weight(F32({64}), W(6, layer))), x);
  }
  b.Output(x);
  return Finish(b);
}

TEST(PlannerSelectionTest, PlansAreDeterministic) {
  const ex::Registry r = Registry();
  for (const std::uint64_t rows : {1U, 4U, 32U}) {
    const StepPlan a = PlanOf(LayerGraph(rows), r);
    const StepPlan b = PlanOf(LayerGraph(rows), r);
    EXPECT_EQ(a.graph.Digest(), b.graph.Digest());
    EXPECT_EQ(a.steps, b.steps);
    EXPECT_EQ(a.placement.offsets, b.placement.offsets);
    EXPECT_EQ(a.placement.extent, b.placement.extent);
    EXPECT_EQ(a.from, b.from);
    EXPECT_EQ(a.tensors, b.tensors);
  }
  // At one row the GLU fuses and K and V share a preparation; at 32 rows
  // neither.
  EXPECT_EQ(Patterns(PlanOf(LayerGraph(1), r)),
            (std::vector<std::string_view>{"rms_norm_mul", "prepared_inputs", "mul_mat_glu",
                                           "rms_norm_mul_add", "rms_norm_mul", "prepared_inputs",
                                           "mul_mat_glu", "rms_norm_mul_add"}));
  EXPECT_EQ(Patterns(PlanOf(LayerGraph(32), r)),
            (std::vector<std::string_view>{"rms_norm_mul", "rms_norm_mul_add", "rms_norm_mul",
                                           "rms_norm_mul_add"}));
}

// ------------------------------------------------------------ placement

// Every pair of placed tensors whose lifetimes meet holds disjoint bytes,
// lifetimes recomputed here from the steps.
void ExpectDisjoint(const StepPlan& p) {
  const ir::Graph& g = p.graph;
  std::map<std::uint32_t, std::pair<std::int64_t, std::int64_t>> life;  // root: first, last
  for (const auto& [t, offset] : p.placement.offsets) {
    life[t.v] = {g.tensor(t).role == ir::Role::kInput ? -1 : 1 << 30, -1};
  }
  const auto end = static_cast<std::int64_t>(p.steps.size());
  for (std::size_t s = 0; s < p.steps.size(); ++s) {
    const auto step = static_cast<std::int64_t>(s);
    for (const ir::NodeId id : p.steps[s].nodes) {
      const ir::Node& n = g.node(id);
      for (const TensorId o : n.outputs()) {
        if (life.contains(o.v)) life[o.v].first = std::min(life[o.v].first, step);
      }
      for (const TensorId src : n.sources()) {
        if (src.valid() && life.contains(g.root(src).v)) {
          auto& l = life[g.root(src).v];
          l.second = std::max(l.second, step);
        }
      }
    }
  }
  for (const TensorId o : g.outputs()) life[g.root(o).v].second = end;
  std::map<std::uint32_t, std::uint64_t> bytes;
  for (const auto& [t, offset] : p.placement.offsets) {
    bytes[t.v] = (RootBytes(g.tensor(t)).value() + 255) / 256 * 256;
  }
  for (const auto& [a, oa] : p.placement.offsets) {
    EXPECT_LE(oa + bytes[a.v], p.placement.extent);
    for (const auto& [b, ob] : p.placement.offsets) {
      if (a.v >= b.v) continue;
      auto la = life[a.v];
      auto lb = life[b.v];
      la.second = std::max(la.second, la.first);
      lb.second = std::max(lb.second, lb.first);
      if (la.first <= lb.second && lb.first <= la.second) {
        EXPECT_TRUE(oa + bytes[a.v] <= ob || ob + bytes[b.v] <= oa) << a.v << " and " << b.v;
      }
    }
  }
}

TEST(PlannerPlacementTest, AChainReusesBytesAndInputsLiveFromTheStart) {
  ir::Builder b;
  const TensorId x = b.Input(F32({64}));  // 256 bytes
  b.Output(b.Scale(b.Scale(b.Scale(x, 2.0f), 2.0f), 2.0f));
  const StepPlan p = PlanOf(Finish(b), Registry());
  EXPECT_EQ(p.placement.offsets.size(), 4U);
  EXPECT_EQ(p.placement.extent, 512U);  // two of them at a time
  EXPECT_EQ(p.placement.offsets[0], std::make_pair(p.tensors[x.v], std::uint64_t{0}));
  ExpectDisjoint(p);
  // Without staging the input, only the activations are placed.
  const StepPlan bare = PlanOf(
      [] {
        ir::Builder c;
        c.Output(c.Scale(c.Scale(c.Scale(c.Input(F32({64})), 2.0f), 2.0f), 2.0f));
        return Finish(c);
      }(),
      Registry(), {}, {.staged = std::vector<TensorId>{}});
  EXPECT_EQ(bare.placement.offsets.size(), 3U);
  EXPECT_EQ(bare.placement.extent, 512U);
}

TEST(PlannerPlacementTest, StepsNeverShareLiveBytes) {
  const ex::Registry r = Registry();
  for (const std::uint64_t rows : {1U, 4U, 32U}) {
    ExpectDisjoint(PlanOf(LayerGraph(rows), r));
  }
}

TEST(PlannerPlacementTest, APreparedInputKeepsItsSourceLive) {
  // x (512 bytes, the largest, placed first at 0) is read by the
  // preparation alone, but lives as long as the products of its prepared
  // form: none of them (256 bytes each, then summed) takes its bytes,
  // which they could if it died with the preparation.
  ir::Builder b;
  const TensorId x = b.Input(F32({64, 2}));
  TensorId sum{};
  for (std::uint16_t i = 0; i < 3; ++i) {
    const TensorId p = b.MatMul(b.Weight(Q8(64, 32), W(i)), x);
    sum = sum.valid() ? b.Add(sum, p) : p;
  }
  b.Output(sum);
  const StepPlan p = PlanOf(Finish(b), Registry());
  ASSERT_EQ(Patterns(p), std::vector<std::string_view>{"prepared_inputs"});
  std::map<std::uint32_t, std::uint64_t> offset;
  for (const auto& [t, o] : p.placement.offsets) offset[t.v] = o;
  EXPECT_EQ(offset.at(p.tensors[x.v].v), 0U);
  for (std::size_t s = 1; s <= 3; ++s) {  // the products
    const TensorId out = p.graph.node(p.steps[s].nodes[0]).out[0];
    EXPECT_GE(offset.at(out.v), 512U) << s;
  }
  ExpectDisjoint(p);
}

// ------------------------------------------------------------ lanes

TEST(PlannerLanesTest, TaggedOwnersRunOnTheirLanes) {
  ir::Builder b;
  const TensorId src = b.Input(F32({16, 2}));
  const TensorId idx = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({2})));
  b.Output(b.Scale(src, 2.0f));  // node 2: untagged
  std::vector<ir::NodeId> stores;
  for (std::uint8_t o = 0; o < 2; ++o) {
    b.SetOwner(o);
    const TensorId cache = b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({16, 8})),
                                   {.stage = 0, .tensor = 0, .layer = 0, .segment = o});
    stores.push_back(ir::NodeId{static_cast<std::uint32_t>(b.node_count())});
    b.CacheStore(cache, src, idx);
  }
  b.SetJoined();
  ir::Graph g = Finish(b);
  ASSERT_EQ(g.node(stores[0]).key, ir::Primitive(Op::kCacheStore));
  const ex::Registry r = Registry();
  PlanOptions options;
  options.lanes = {{stores[0], {.lane = 1, .region = 1}}, {stores[1], {.lane = 2, .region = 1}}};
  const StepPlan p = PlanOf(std::move(g), r, {}, options);
  // The batch splits across the lanes; each store runs alone on its lane,
  // inside one region.
  EXPECT_EQ(StepSizes(p), (std::vector<std::size_t>{1, 1, 1}));
  EXPECT_EQ(Names(p, r), (std::vector<std::string>{"t.scale", "t.set_rows", "t.set_rows"}));
  EXPECT_EQ(p.steps[0].lane, 0);
  EXPECT_EQ(p.steps[1].lane, 1);
  EXPECT_EQ(p.steps[2].lane, 2);
  EXPECT_EQ(p.regions, (std::vector<Region>{{.first = 1, .last = 2}}));
  // An implementation that stays on the stream keeps lane 0.
  options.on_stream = [](std::string_view name) { return name == "t.set_rows"; };
  ir::Builder c;
  const TensorId s2 = c.Input(F32({16, 2}));
  const TensorId i2 = c.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({2})));
  c.Output(c.Scale(s2, 2.0f));
  for (std::uint8_t o = 0; o < 2; ++o) {
    c.SetOwner(o);
    c.CacheStore(c.State(ir::Dense(Encoding::kF16, ir::ShapeOf({16, 8})),
                         {.stage = 0, .tensor = 0, .layer = 0, .segment = o}),
                 s2, i2);
  }
  const StepPlan q = PlanOf(Finish(c), r, {}, options);
  EXPECT_EQ(q.steps[1].lane, 0);
  EXPECT_EQ(q.steps[2].lane, 0);
}

// ------------------------------------------------------------ plan keys

TEST(PlannerKeyTest, KeysFollowWhatDecidesAPlan) {
  base::Sha256Digest spec{};
  spec[0] = 1;
  StepShape shape{
      .phase = Phase::kDecode,
      .segments = {{.slot = 0, .rows = 1, .past = 10, .read_cells = 256, .outputs = 1}}};
  const Policy policy{};
  const PlanKey key = KeyOf(spec, shape, policy);
  EXPECT_EQ(key, KeyOf(spec, shape, policy));
  EXPECT_EQ(key.patterns, PatternsDigest(policy));
  EXPECT_EQ(key.policy, Digest(policy));
  // A segment's past is the step's data, not its plan.
  StepShape later = shape;
  later.segments[0].past = 200;
  EXPECT_EQ(KeyOf(spec, later, policy), key);
  // Everything else that decides the plan does change it.
  base::Sha256Digest other = spec;
  other[0] = 2;
  EXPECT_NE(KeyOf(other, shape, policy), key);
  StepShape rows = shape;
  rows.segments[0].rows = 2;
  EXPECT_NE(KeyOf(spec, rows, policy), key);
  StepShape reads = shape;
  reads.segments[0].read_cells = 512;
  EXPECT_NE(KeyOf(spec, reads, policy), key);
  StepShape verify = shape;
  verify.phase = Phase::kVerify;
  verify.row_invariant = true;
  EXPECT_NE(KeyOf(spec, verify, policy), key);
  EXPECT_NE(KeyOf(spec, shape, Disabling({"rms_norm_mul"})), key);
  EXPECT_NE(KeyOf(spec, shape, policy, 1), key);
}

TEST(PlannerSelectionTest, RootAlignmentFactsOwnAnExactRootIdentitySeal) {
  const auto external = [](bool reverse) {
    ir::Builder b;
    const auto f = ir::Dense(Encoding::kF32, ir::ShapeOf({16}));
    ir::TensorId a, z;
    if (reverse) {
      z = b.State(f, {.tensor = 1});
      a = b.State(f, {.tensor = 0});
    } else {
      a = b.State(f, {.tensor = 0});
      z = b.State(f, {.tensor = 1});
    }
    b.Output(b.Add(a, z));
    return std::move(b).Finish();
  };
  auto original = external(false);
  auto reversed = external(true);
  ASSERT_TRUE(original);
  ASSERT_TRUE(reversed);
  ASSERT_EQ(original->Digest(), reversed->Digest());
  const std::array facts{RootAlignmentFact{.root = {0}, .alignment = 256},
                         RootAlignmentFact{.root = {1}, .alignment = 4}};
  auto aligned = GraphRootAlignments(*original, facts);
  ASSERT_TRUE(aligned);
  EXPECT_EQ(aligned(*original, {0}), 256);
  EXPECT_EQ(aligned(*original, {1}), 4);
  EXPECT_EQ(aligned(*reversed, {0}), 0);
  EXPECT_EQ(aligned(*reversed, {1}), 0);
  auto clone = original->Clone();
  EXPECT_EQ(aligned(clone, {0}), 256);
  clone = std::move(*reversed);  // same Graph address, different root meanings
  EXPECT_EQ(aligned(clone, {0}), 0);
  EXPECT_EQ(aligned(*original, original->outputs()[0]), 0);  // no arena guarantee
  EXPECT_EQ(aligned(*original, {999}), 0);
  for (const auto& bad : std::vector<std::vector<RootAlignmentFact>>{
           {{{0}, 0}}, {{{0}, 3}}, {{{999}, 16}}, {{{0}, 16}, {{0}, 16}}})
    EXPECT_FALSE(GraphRootAlignments(*original, bad));

  ir::Builder views;
  const auto root = views.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({32})));
  const auto view = views.View(root, ir::ShapeOf({16}), {4, 64, 64, 64}, 4);
  views.Output(view);
  auto graph = std::move(views).Finish();
  ASSERT_TRUE(graph);
  EXPECT_FALSE(GraphRootAlignments(*graph, std::array{RootAlignmentFact{view, 256}}));
  auto owner = GraphRootAlignments(*graph, std::array{RootAlignmentFact{root, 256}});
  auto moved = std::move(*graph);
  EXPECT_EQ(owner(moved, root), 256);
  EXPECT_EQ(owner(moved, view), 0);
}

TEST(PlannerSelectionTest, RootAlignmentQueriesDoNotLendFactsToHypotheticalDescriptors) {
  ir::Builder b;
  const auto x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({16})));
  const auto y = b.Unary(x, ir::UnaryFn::kRelu);
  b.Output(y);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph);
  auto cap = Any(ir::Primitive(Op::kUnary));
  cap.accepts = [](const ex::CapabilityQuery& q) {
    return q.operands[0].memory && q.operands[0].memory->root_alignment >= 16;
  };
  auto registry = ex::Registry::Create({Fake("aligned", cap)});
  ASSERT_TRUE(registry);
  const DeviceFacts device;
  const Policy policy;
  const SelectionView view(*registry, device, policy, false, false, {},
                           GraphRootAlignments(*graph, std::array{RootAlignmentFact{x, 256}}));
  const auto& node = graph->node(graph->producer(y));
  EXPECT_EQ(view.Candidates(*graph, node).size(), 1);
  const std::array sources{&graph->tensor(x)}, outputs{&graph->tensor(y)};
  EXPECT_TRUE(view.Candidates(node.key, sources, outputs, node.params).empty());

  ir::Builder inserted;
  const auto before = inserted.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({16})));
  const auto original_input = inserted.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({16})));
  inserted.Keep(before);
  const auto later = inserted.Unary(original_input, ir::UnaryFn::kRelu);
  inserted.Output(later);
  auto changed = std::move(inserted).Finish();
  ASSERT_TRUE(changed);
  EXPECT_TRUE(view.Candidates(*changed, changed->node(changed->producer(later))).empty());
}

TEST(PlannerSelectionTest, RootAlignmentFactsFailClosedAcrossPatternAndExpansionGraphs) {
  ir::Builder b;
  const auto x = b.Input(F32({16, 2}));
  const auto weight = b.Input(F32({16}));
  const auto y = b.Mul(b.RmsNorm(x, 1e-5F), weight);
  b.Output(y);
  auto graph = Finish(b);
  const auto facts = GraphRootAlignments(graph, std::array{RootAlignmentFact{x, 256}});
  const auto registry = Registry();
  const Policy policy;
  const SelectionView selection(registry, kDevice, policy, false, false, {}, facts);
  auto rewritten = ApplyPatterns(graph, selection, nullptr);
  ASSERT_TRUE(rewritten);
  ASSERT_EQ(rewritten->applied.size(), 1U);
  EXPECT_EQ(facts(rewritten->graph, rewritten->tensors.at(x.v)), 0U);
  const auto rewritten_x = rewritten->tensors.at(x.v);
  const auto rewritten_facts =
      GraphRootAlignments(rewritten->graph, std::array{RootAlignmentFact{rewritten_x, 256}});
  std::vector<TensorId> map;
  auto expanded = ir::Expand(rewritten->graph, true, &map);
  ASSERT_TRUE(expanded);
  EXPECT_EQ(rewritten_facts(*expanded, map.at(rewritten_x.v)), 0U);
  EXPECT_EQ(facts(graph.Clone(), x), 256U);
}

}  // namespace
}  // namespace llmp::planner
