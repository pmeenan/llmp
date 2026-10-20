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
#include <cmath>
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

#include "base/json.h"
#include "base/sha256.h"
#include "execution/registry.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/decompose.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "planner/construction.h"
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
ir::ProductArithmetic FixtureVectorArithmetic(const ex::CapabilityQuery& q) {
  return q.rows <= 8 ? ir::ProductArithmetic::kGgmlQ8_1F32 : ir::ProductArithmetic::kImplementation;
}

ex::Implementation Fake(std::string name, ex::Capability capability,
                        llmp::ir::OpKey operation = llmp::ir::Primitive(llmp::ir::Op::kAdd)) {
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
                                              .group = "t.mul_mat",
                                              .product_arithmetic = &FixtureVectorArithmetic}),
      Fake("t.mul_mat.tile", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                            .operands = kProductRules,
                                            .shape = {.min_rows = 9},
                                            .group = "t.mul_mat"}),
      Fake("t.mul_mat.prepared", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                                .operands = kPreparedRules,
                                                .shape = {.max_rows = 8},
                                                .product_arithmetic = &FixtureVectorArithmetic}),
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

// These synthetic numerical controls qualify their own exact graph sites;
// they are not historical checkpoint or production quality attestations.
Policy FixtureNumericalPolicy(const ir::Graph& graph, Policy policy,
                              std::span<const NumericalChange> changes) {
  const auto source = base::Sha256{}.Update("planner numerical control source").Finish();
  const auto semantic = graph.Digest();
  const auto evidence = base::Sha256{}.Update("independent synthetic numerical control").Finish();
  std::vector<NumericalQualification> entries;
  for (const auto& node : graph.nodes()) {
    if (node.key.op == ir::Op::kInput || node.key.op == ir::Op::kWeight ||
        node.key.op == ir::Op::kState)
      continue;
    std::vector<const ir::TensorDesc*> sources, outputs;
    for (const auto id : node.sources())
      sources.push_back(id.valid() ? &graph.tensor(id) : nullptr);
    for (const auto id : node.outputs()) outputs.push_back(&graph.tensor(id));
    const auto shape = ShapeOfNode(node.key, sources, outputs);
    for (const auto change : changes) {
      const auto add = [&](ir::OpKey anchor) {
        entries.push_back({.change = change,
                           .evidence = evidence,
                           .source = source,
                           .semantic = semantic,
                           .stage = 0,
                           .component = node.component,
                           .anchor = anchor,
                           .arch = kDevice.arch,
                           .min_rows = shape.rows,
                           .max_rows = shape.rows,
                           .max_k = shape.k,
                           .max_n = shape.n,
                           .exact_shape = true});
      };
      add(node.key);
      if (change == NumericalChange::kBf16ResidualCarry)
        add(ir::Fused(FusedKind::kLayerNormModulateSplit));
    }
  }
  auto bound = NumericalQualifications::Bind(entries, source, semantic);
  EXPECT_TRUE(bound) << (bound ? "" : bound.error());
  if (bound) policy.numerical = std::move(*bound);
  policy.active_stage = 0;
  return policy;
}
StepPlan NumericalPlanOf(ir::Graph graph, const ex::Registry& registry, const Policy& policy,
                         std::initializer_list<NumericalChange> changes) {
  const auto qualified = FixtureNumericalPolicy(graph, policy, changes);
  return PlanOf(std::move(graph), registry, qualified);
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

TEST(PlannerReportTest, StreamsActualOrderedSelectionsWithoutValuesOrTensorNames) {
  auto registry = ex::Registry::Create(Everything()).value();
  ir::Builder b;
  auto x = b.Input(F32({64, 3}), "private_prompt_contents");
  auto w = b.Weight(F32({64}), W(0));
  b.Output(b.Mul(b.RmsNorm(x, 1e-6F), w));
  auto plan = PlanOf(Finish(b), registry);
  std::vector<std::string> records;
  ReportPlan(plan, registry, [&](std::string_view record) { records.emplace_back(record); });
  ASSERT_FALSE(records.empty());
  std::size_t selected = 0;
  for (const auto& record : records) {
    EXPECT_EQ(record.find("private_prompt_contents"), std::string::npos);
    auto parsed = base::json::Parse(record);
    ASSERT_TRUE(parsed) << record;
    const auto kind = parsed->root().find("kind");
    ASSERT_TRUE(kind);
    if (kind->string() != "selection") continue;
    ASSERT_LT(selected, plan.steps.size());
    const auto& step = plan.steps[selected];
    EXPECT_EQ(parsed->root().find("implementation")->string(),
              registry.at(step.implementation).name);
    EXPECT_EQ(parsed->root().find("identity")->string(),
              base::ToHex(registry.identity(step.implementation)));
    EXPECT_EQ(parsed->root().find("step")->int64(), selected);
    ++selected;
  }
  EXPECT_EQ(selected, plan.steps.size());
  std::vector<std::string> repeated;
  ReportPlan(plan, registry, [&](std::string_view record) { repeated.emplace_back(record); });
  EXPECT_EQ(records, repeated);
  ReportPlan(plan, registry, {});
}

TEST(PlannerReportTest, ExactFragmentCountParsesEscapesAndMaximumUnsignedFields) {
  auto registry = ex::Registry::Create(Everything()).value();
  ir::Builder b;
  auto x = b.Input(F32({64, 3}));
  auto w = b.Weight(F32({64}), W(0));
  b.Output(b.Mul(b.RmsNorm(x, 1e-6F), w));
  auto plan = PlanOf(Finish(b), registry);
  EXPECT_EQ(plan.graph_digest, plan.graph.Digest());
  ASSERT_FALSE(plan.steps.empty());
  ASSERT_FALSE(plan.placement.offsets.empty());
  const std::string escaped = std::string("quoted\" slash\\ tab\t line\n") + char{0} + char{1};
  plan.steps[0].group = escaped;
  plan.applied.push_back({.pattern = escaped});
  plan.decomposed.push_back(escaped);
  plan.placement.extent = UINT64_MAX;
  plan.placement.offsets[0].second = UINT64_MAX;
  std::size_t counted = 0, fragments = 0;
  StreamPlanReport(plan, registry, [&](std::string_view part) {
    counted += part.size();
    ++fragments;
  });
  EXPECT_GT(fragments, plan.steps.size());
  std::string filled(counted, '\0');
  std::size_t written = 0;
  StreamPlanReport(plan, registry, [&](std::string_view part) {
    ASSERT_LE(part.size(), filled.size() - written);
    std::ranges::copy(part, filled.begin() + static_cast<std::ptrdiff_t>(written));
    written += part.size();
  });
  ASSERT_EQ(written, counted);
  std::string wrapped;
  ReportPlan(plan, registry, [&](std::string_view record) {
    wrapped.append(record);
    wrapped += '\n';
  });
  EXPECT_EQ(wrapped, filled);
  bool header = false, group = false, pattern = false, fallback = false, offset = false;
  std::string_view records = filled;
  while (!records.empty()) {
    const auto end = records.find('\n');
    ASSERT_NE(end, std::string_view::npos);
    const auto record = records.substr(0, end);
    auto parsed = base::json::Parse(record);
    ASSERT_TRUE(parsed) << record;
    const auto root = parsed->root();
    const auto kind = root.find("kind")->string();
    if (kind == "plan") {
      EXPECT_EQ(root.find("activation_bytes")->number(), "18446744073709551615");
      EXPECT_EQ(root.find("graph")->string(), base::ToHex(plan.graph_digest));
      header = true;
    } else if (kind == "selection" && root.find("step")->int64() == 0) {
      EXPECT_EQ(root.find("group")->string(), escaped);
      group = true;
      EXPECT_EQ(root.find("nodes")->size(), plan.steps[0].nodes.size());
      EXPECT_EQ(root.find("implementation")->string(),
                registry.at(plan.steps[0].implementation).name);
    } else if (kind == "pattern" && root.find("name")->string() == escaped) {
      pattern = true;
    } else if (kind == "fallback" && root.find("op")->string() == escaped) {
      EXPECT_EQ(root.find("reason")->string(), "no_eligible_implementation");
      fallback = true;
    } else if (kind == "activation" && root.find("offset")->number() == "18446744073709551615") {
      offset = true;
    }
    records.remove_prefix(end + 1);
  }
  EXPECT_TRUE(header && group && pattern && fallback && offset);
}

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

TEST(PlannerPolicyTest, ProductChoicesAreIndependentCanonicalAndKeyed) {
  const Policy base{};
  Policy p;
  p.products.dense_pair = true;
  const auto text = ToText(p);
  EXPECT_EQ(text, "max_decode_rows 16\nproduct dense_pair\n");
  EXPECT_EQ(ParsePolicy(text).value(), p);
  EXPECT_NE(Digest(base), Digest(p));
  EXPECT_FALSE(p.products.d2r_experts);
  EXPECT_FALSE(p.products.vector_floats);
  p.products.d2r_experts = true;
  EXPECT_NE(Digest(ParsePolicy(text).value()), Digest(p));
  EXPECT_EQ(ParsePolicy(ToText(p)).value(), p);
  EXPECT_FALSE(ParsePolicy("max_decode_rows 16\nproduct unknown\n"));
  EXPECT_FALSE(ParsePolicy("max_decode_rows 16\nproduct dense_pair\nproduct dense_pair\n"));
}

TEST(PlannerPolicyTest, CompactExpertCrossoverPreservesDefaultTextAndKeysOverrides) {
  Policy policy;
  EXPECT_EQ(ToText(policy), "max_decode_rows 16\n");
  policy.products.compact_experts = true;
  const auto original = Digest(policy);
  policy.products.compact_expert_min_rows = 512;
  EXPECT_EQ(ToText(policy),
            "max_decode_rows 16\ncompact_expert_min_rows 512\nproduct compact_experts\n");
  EXPECT_EQ(ParsePolicy(ToText(policy)).value(), policy);
  EXPECT_NE(Digest(policy), original);
  for (const auto bound : {0U, 65537U}) {
    policy.products.compact_expert_min_rows = bound;
    EXPECT_FALSE(Validate(policy));
  }
  EXPECT_FALSE(ParsePolicy("max_decode_rows 16\ncompact_expert_min_rows 2048\n"));
  EXPECT_FALSE(ParsePolicy(
      "max_decode_rows 16\ncompact_expert_min_rows 512\ncompact_expert_min_rows 512\n"));
}

TEST(PlannerPolicyTest, CompactExpertCrossoverResolvesBeforeLowering) {
  const auto registry = Registry();
  Policy policy;
  policy.products.compact_experts = true;
  policy.products.compact_expert_min_rows = 512;
  for (const auto rows : {511U, 512U, 513U}) {
    NormGraph graph(64, rows);
    graph.b.Output(graph.m);
    const StepShape shape{.segments = {{.rows = rows, .outputs = rows}}};
    const auto plan = PlanOf(Finish(graph.b), registry, policy, {.shape = &shape});
    EXPECT_EQ(plan.products.compact_experts, rows >= 512);
    EXPECT_EQ(plan.products.compact_expert_min_rows, 512U);
  }
  NormGraph graph;
  graph.b.Output(graph.m);
  EXPECT_FALSE(PlanOf(Finish(graph.b), registry, policy).products.compact_experts);
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

TEST(PlannerPolicyTest, NumericalCandidatesAreTypedCanonicalAndNeverEvidence) {
  Policy candidate;
  candidate.numerical_candidates = {NumericalChange::kQ8VectorReduction,
                                    NumericalChange::kBf16ResidualCarry};
  Canonicalize(candidate);
  const auto text = ToText(candidate);
  EXPECT_EQ(text,
            "max_decode_rows 16\ncandidate_numerics bf16_residual_carry\n"
            "candidate_numerics q8_vector_reduction\n");
  ASSERT_TRUE(ParsePolicy(text));
  EXPECT_EQ(ParsePolicy(text).value(), candidate);
  EXPECT_NE(Digest(candidate), Digest(Policy{}));
  EXPECT_FALSE(candidate.numerical);
  EXPECT_FALSE(ParsePolicy("max_decode_rows 16\ncandidate_numerics all\n"));
  EXPECT_FALSE(
      ParsePolicy("max_decode_rows 16\ncandidate_numerics q8_vector_reduction\n"
                  "candidate_numerics q8_vector_reduction\n"));
  EXPECT_FALSE(
      ParsePolicy("max_decode_rows 16\ncandidate_numerics q8_vector_reduction\n"
                  "candidate_numerics bf16_residual_carry\n"));
}

TEST(PlannerPolicyTest, NumericalQualificationsBindOnlyTheActiveNonzeroIdentityAndExactSite) {
  const auto source = base::Sha256{}.Update("authenticated source receipt set").Finish();
  const auto semantic = base::Sha256{}.Update("admitted semantic spec").Finish();
  NumericalQualification grant{.change = NumericalChange::kBf16ProductInputAndResult,
                               .evidence = base::Sha256{}.Update("same quality control").Finish(),
                               .source = source,
                               .semantic = semantic,
                               .stage = 0,
                               .component = 3,
                               .anchor = ir::Primitive(Op::kMatMul),
                               .arch = 121,
                               .min_rows = 17,
                               .max_rows = 17,
                               .max_k = 64,
                               .max_n = 32,
                               .exact_shape = true,
                               .weight_role = 0};
  const std::array entries{grant};
  EXPECT_FALSE(NumericalQualifications::Bind(entries, {}, semantic));
  EXPECT_FALSE(NumericalQualifications::Bind(entries, source, {}));
  EXPECT_FALSE(NumericalQualifications::Bind(entries, semantic, semantic));
  EXPECT_FALSE(NumericalQualifications::Bind(entries, source, source));
  auto unproven = grant;
  unproven.evidence = {};
  EXPECT_FALSE(NumericalQualifications::Bind(std::array{unproven}, source, semantic));
  unproven = grant;
  unproven.max_rows = 18;  // Exact evidence is not a general smaller-shape rule.
  EXPECT_FALSE(NumericalQualifications::Bind(std::array{unproven}, source, semantic));
  auto facts = NumericalQualifications::Bind(entries, source, semantic);
  ASSERT_TRUE(facts);
  EXPECT_EQ((*facts)->host_bytes(), NumericalQualificationBytes(entries.size()).value());
  Policy qualified{.numerical = *facts, .active_stage = 0};
  const auto key = Digest(qualified);
  EXPECT_NE(key, Digest(Policy{}));
  EXPECT_FALSE(ParsePolicy("max_decode_rows 16\nnumerical same_quality\n"));
  const auto registry = Registry();
  for (int mutation = 0; mutation < 8; ++mutation) {
    ir::Builder builder;
    builder.SetComponent(mutation == 1 ? 4 : 3);
    auto ref = W(mutation == 6 ? 1 : 0);
    ref.component = mutation == 1 ? 4 : 3;
    if (mutation == 7) ref.stage = 1;  // same role/K/N from another admitted stage
    const auto weight = builder.Weight(Q8(mutation == 3 ? 96 : 64, mutation == 4 ? 16 : 32), ref);
    const auto input = builder.Input(F32({mutation == 3 ? 96U : 64U, mutation == 2 ? 16U : 17U}));
    const auto output = builder.MatMul(weight, input);
    builder.Output(output);
    const auto graph = Finish(builder);
    Policy policy = qualified;
    if (mutation == 5) policy.active_stage = 1;
    const SelectionView view(registry, kDevice, policy, false, false);
    const auto matched =
        view.Qualification(grant.change, graph, graph.node(graph.producer(output)));
    EXPECT_EQ(matched.has_value(), mutation == 0);
    if (matched) EXPECT_EQ(matched->evidence, grant.evidence);
  }
  qualified.active_stage = 1;
  EXPECT_NE(Digest(qualified), key);
  auto other = grant;
  other.evidence = base::Sha256{}.Update("different measured evidence").Finish();
  qualified.numerical = NumericalQualifications::Bind(std::array{other}, source, semantic).value();
  qualified.active_stage = 0;
  EXPECT_NE(Digest(qualified), key);
}

TEST(PlannerPolicyTest, ScalarIncumbentCannotAuthorizeMoreRowsOtherRolesOrGeometry) {
  const auto source = base::Sha256{}.Update("exact scalar checkpoint receipts").Finish();
  const auto semantic = base::Sha256{}.Update("exact scalar admitted spec").Finish();
  const NumericalQualification grant{
      .change = NumericalChange::kQ8VectorReduction,
      .evidence = base::Sha256{}.Update("retained scalar incumbent reference evidence").Finish(),
      .source = source,
      .semantic = semantic,
      .stage = 0,
      .component = 3,
      .anchor = ir::Primitive(Op::kMatMul),
      .arch = 121,
      .min_rows = 1,
      .max_rows = 1,
      .max_k = 64,
      .max_n = 32,
      .exact_dimensions = true,
      .incumbent_recipe = true,
      .weight_role = 0};
  auto facts = NumericalQualifications::Bind(std::array{grant}, source, semantic);
  ASSERT_TRUE(facts);
  EXPECT_FALSE(NumericalQualifications::Bind(std::array{grant}, semantic, semantic));
  const auto registry = Registry();
  for (const auto rows : {1U, 2U, 3U, 4U, 8U}) {
    for (unsigned mutation = 0; mutation != 5; ++mutation) {
      SCOPED_TRACE(rows);
      SCOPED_TRACE(mutation);
      ir::Builder builder;
      builder.SetComponent(3);
      auto ref = W(mutation == 1 ? 1 : 0);
      ref.component = 3;
      if (mutation == 4) ref.stage = 1;
      const auto k = mutation == 2 ? 96U : 64U;
      const auto n = mutation == 3 ? 16U : 32U;
      const auto weight = builder.Weight(Q8(k, n), ref);
      const auto input = builder.Input(F32({k, rows}));
      const auto output = builder.MatMul(weight, input);
      builder.Output(output);
      const auto graph = Finish(builder);
      const Policy policy{.numerical = *facts, .active_stage = 0};
      const SelectionView view(registry, kDevice, policy, false, false);
      EXPECT_EQ(
          view.Qualification(grant.change, graph, graph.node(graph.producer(output))).has_value(),
          rows == 1 && mutation == 0);
    }
  }
}

TEST(PlannerPolicyTest, DefaultAvailabilityDoesNotTurnEnableIntoNumericalAuthority) {
  EXPECT_FALSE(Disabled({}, "rms_norm_mul_rope"));
  EXPECT_FALSE(Disabled({}, "bf16_glu"));
  EXPECT_TRUE(Disabled({}, "rounded_indexer"));
  EXPECT_TRUE(Disabled({}, "owner_row_invariant"));
  const auto explicit_enable = ParsePolicy("max_decode_rows 16\nenable bf16_glu\n");
  ASSERT_TRUE(explicit_enable);
  EXPECT_FALSE(explicit_enable->numerical);
  EXPECT_TRUE(Validate(Policy{.enabled = {"rms_norm_mul"}}));
}

TEST(PlannerPolicyTest, StepRulesAreCanonicalAndChecked) {
  Policy p{
      .enabled = {"vecq_glu"},
      .vector_rows = 12,
      .rules = {
          {.scope = StepScope::kJoined, .test = RowTest::kPast, .rows = 16, .pattern = "vecq_glu"},
          {.scope = StepScope::kSolo,
           .test = RowTest::kPast,
           .rows = 8,
           .action = StepAction::kVectorRows,
           .value = 8},
          {.scope = StepScope::kAny,
           .test = RowTest::kUnder,
           .rows = 64,
           .pattern = "rope_store"}}};
  EXPECT_FALSE(Validate(p).has_value());  // not in order
  Canonicalize(p);
  ASSERT_TRUE(Validate(p).has_value());
  const std::string text = ToText(p);
  EXPECT_EQ(text,
            "max_decode_rows 16\nvector_rows 12\nenable vecq_glu\n"
            "when any under 64 disable rope_store\nwhen solo past 8 vector_rows 8\n"
            "when joined past 16 disable vecq_glu\n");
  EXPECT_EQ(ParsePolicy(text).value(), p);
  EXPECT_NE(Digest(p), Digest(Policy{.enabled = {"vecq_glu"}, .vector_rows = 12}));
  for (const std::string_view bad : {
           "max_decode_rows 16\nvector_rows 0\n",                          // its absence
           "max_decode_rows 16\nvector_rows 65537\n",                      // out of range
           "max_decode_rows 16\nwhen solo past 8 disable nothing_such\n",  // no such pattern
           "max_decode_rows 16\nwhen some past 8 disable rope_store\n",    // no such scope
           "max_decode_rows 16\nwhen solo over 8 disable rope_store\n",    // no such test
           "max_decode_rows 16\nwhen solo past 8 enable rope_store\n",     // no such action
           "max_decode_rows 16\nwhen solo past 8 vector_rows 0\n",         // no bound
           "max_decode_rows 16\nwhen solo past 8 max_decode_rows x\n",     // not a number
           "max_decode_rows 16\nwhen any under 0 disable rope_store\n",    // holds for nothing
           "max_decode_rows 16\nwhen solo past 8 disable rope_store\n"
           "when any past 8 disable rope_store\n",  // out of order
           "max_decode_rows 16\nwhen solo past 8 disable rope_store\n"
           "when solo past 8 disable rope_store\n",  // twice
           "max_decode_rows 16\nrank sm_121 x 0\nwhen solo past 8 disable rope_store\n",
       }) {
    EXPECT_FALSE(ParsePolicy(bad).has_value()) << bad;
  }
}

// A step's policy: the rules that hold for its rows and owners applied,
// none left; others' patterns and bounds unchanged.
TEST(PlannerPolicyTest, ForStepAppliesTheRulesThatHold) {
  const Policy p =
      ParsePolicy(
          "max_decode_rows 16\nenable grouped_vecq\nenable vecq_glu\nenable vecq_inputs\n"
          "when any under 4 disable rope_store\nwhen solo past 8 disable vecq_glu\n"
          "when solo past 8 max_decode_rows 8\nwhen solo past 8 vector_rows 8\n"
          "when solo past 12 vector_rows 4\nwhen joined past 16 disable vecq_inputs\n")
          .value();
  const auto at = [&](std::uint64_t rows, std::size_t segments) {
    const Policy step = ForStep(p, rows, segments).value();
    EXPECT_TRUE(step.rules.empty());
    EXPECT_TRUE(Validate(step).has_value());
    EXPECT_FALSE(Disabled(step, "grouped_vecq"));
    return step;
  };
  {
    const Policy step = at(2, 1);  // a solo step of 2 rows: only rope_store off
    EXPECT_TRUE(Disabled(step, "rope_store"));
    EXPECT_FALSE(Disabled(step, "vecq_glu"));
    EXPECT_EQ(step.max_decode_rows, 16U);
    EXPECT_EQ(step.vector_rows, 0U);
  }
  {
    const Policy step = at(8, 1);  // the gate is past 8, not at it
    EXPECT_FALSE(Disabled(step, "rope_store"));
    EXPECT_FALSE(Disabled(step, "vecq_glu"));
    EXPECT_EQ(ToText(step), ToText(ForStep(Policy{.enabled = p.enabled}, 8, 1).value()));
  }
  {
    const Policy step = at(9, 1);
    EXPECT_TRUE(Disabled(step, "vecq_glu"));
    EXPECT_FALSE(Disabled(step, "vecq_inputs"));
    EXPECT_EQ(step.max_decode_rows, 8U);
    EXPECT_EQ(step.vector_rows, 8U);
  }
  EXPECT_EQ(at(13, 1).vector_rows, 4U);  // the least bound
  {
    const Policy step = at(13, 2);  // joined: the solo rules do not hold
    EXPECT_FALSE(Disabled(step, "vecq_glu"));
    EXPECT_FALSE(Disabled(step, "vecq_inputs"));
    EXPECT_EQ(step.max_decode_rows, 16U);
    EXPECT_EQ(step.vector_rows, 0U);
  }
  EXPECT_TRUE(Disabled(at(17, 3), "vecq_inputs"));
  EXPECT_FALSE(Disabled(at(17, 3), "vecq_glu"));
  // Without rules, the policy itself.
  const Policy plain = ParsePolicy("max_decode_rows 16\nenable vecq_glu\n").value();
  EXPECT_EQ(ForStep(plain, 4096, 1).value(), plain);
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
  const Policy default_policy{.disabled = {"state_bank_attention"}};
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

TEST(PlannerPatternTest, GroupedOutputShapeComesFromItsWeightRatherThanACheckpoint) {
  ir::TensorDesc weight, heads;
  weight.format = F32({128, 384, 2});
  heads.format = F32({64, 4, 5});
  const std::array<const ir::TensorDesc*, 2> sources{&weight, &heads};
  const auto shape = ShapeOfNode(ir::Fused(FusedKind::kInverseRopeGroupedMatMul), sources, {});
  EXPECT_EQ(shape.rows, 5U);
  EXPECT_EQ(shape.k, 256U);
  EXPECT_EQ(shape.n, 768U);
}

TEST(PlannerPatternTest, GroupedOutputMatcherDelegatesSpecializedDimensionsToTheCandidate) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.outa", Any(ir::Fused(FusedKind::kInverseRopeGroupedMatMul))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{};
  const SelectionView selection(registry, kDevice, policy, false, false);
  ir::Builder b;
  const auto weight = b.Input(F32({128, 384, 2}));
  const auto heads = b.Input(F32({64, 4, 5}));
  const auto positions = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({5})));
  const auto rotated =
      b.Rope(heads, positions, {},
             {.mode = ir::RopeMode::kNormal, .inverse = true, .dims = 32, .offset = 32});
  const auto reshaped = b.Reshape(rotated, ir::ShapeOf({128, 2, 5}));
  const auto low = b.MatMul(weight, b.Permute(reshaped, {0, 2, 1, 3}));
  const auto out = b.Cont(b.Permute(low, {0, 2, 1, 3}), ir::ShapeOf({768, 5}));
  b.Output(out);
  const auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph) << (graph ? "" : graph.error().detail);
  const auto match = fusions::InverseRopeGroupedMatMulMatch(
      MatchContext(*graph, selection, nullptr), graph->producer(out));
  EXPECT_TRUE(match);
}

TEST(PlannerPatternTest, Dsv4QHeadChecksTailOwnershipSideReadersAndInterveningWrites) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.dsv4_qhead", Any(ir::Fused(FusedKind::kRmsNormMulRope))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{.disabled = {"rms_norm_rope_tail"}};
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
    EXPECT_EQ(fusions::RmsNormRopeTailMatch(c, g->producer(out)).has_value(), mutation == 0);
    if (mutation != 0) continue;
    Policy on;
    on.enabled = {"rms_norm_rope_tail"};
    const SelectionView enabled(registry, kDevice, on, false, false);
    auto rewritten = ApplyPatterns(*g, enabled, nullptr);
    ASSERT_TRUE(rewritten) << (rewritten ? "" : rewritten.error().detail);
    ASSERT_EQ(rewritten->applied.size(), 1U);
    EXPECT_EQ(rewritten->applied[0].pattern, "rms_norm_rope_tail");
    const auto& result =
        rewritten->graph.node(rewritten->graph.producer(rewritten->graph.outputs()[0]));
    EXPECT_EQ(result.key, ir::Fused(FusedKind::kRmsNormMulRope));
    const auto params = result.Params<ir::NormRopeParams>();
    EXPECT_EQ(ir::ToBytes(params.rope), g->node(g->producer(out)).params);
    EXPECT_EQ(params.eps, 1e-6F);
    auto disabled = ApplyPatterns(*g, selection, nullptr);
    ASSERT_TRUE(disabled);
    EXPECT_TRUE(disabled->applied.empty());
    const auto primitive_registry = Registry();
    const SelectionView unavailable(primitive_registry, kDevice, on, false, false);
    EXPECT_FALSE(
        fusions::RmsNormRopeTailMatch(MatchContext(*g, unavailable, nullptr), g->producer(out)));
  }
}

TEST(PlannerPatternTest, Dsv4HcPreChecksTheFullCarryAndPrimitiveArithmetic) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.dsv4_hc_pre", Any(ir::Fused(FusedKind::kSinkhornHcPre))));
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
    EXPECT_EQ(fusions::SinkhornHcPreMatch(c, g->producer(out)).has_value(), mutation == 0);
    if (mutation != 0) continue;
    Policy on;
    on.enabled = {"sinkhorn_hc_pre"};
    const SelectionView enabled(registry, kDevice, on, false, false);
    auto rewritten = ApplyPatterns(*g, enabled, nullptr);
    ASSERT_TRUE(rewritten) << (rewritten ? "" : rewritten.error().detail);
    ASSERT_EQ(rewritten->applied.size(), 1U);
    EXPECT_EQ(rewritten->applied[0].pattern, "sinkhorn_hc_pre");
    const auto& node =
        rewritten->graph.node(rewritten->graph.producer(rewritten->graph.outputs()[0]));
    EXPECT_EQ(node.key, ir::Fused(FusedKind::kSinkhornHcPre));
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
    EXPECT_EQ(fusions::HcPostFlatNormMatch(MatchContext(*g, selection, nullptr), g->producer(norm))
                  .has_value(),
              mutation == 0);
    if (mutation != 0) continue;
    Policy on;
    on.enabled = {"hc_post_flat_norm"};
    const SelectionView enabled(registry, kDevice, on, false, false);
    auto rewritten = ApplyPatterns(*g, enabled, nullptr);
    ASSERT_TRUE(rewritten) << (rewritten ? "" : rewritten.error().detail);
    ASSERT_EQ(rewritten->applied.size(), 1U);
    EXPECT_EQ(rewritten->applied[0].pattern, "hc_post_flat_norm");
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
    b.FusedBytes(FusedKind::kSoftmaxPoolCompress,
                 std::array{state, state_score, current, current_score, indices}, raw, {});
    EXPECT_FALSE(b.ok());
  }
}

TEST(PlannerPatternTest, Dsv4CompressorChecksTheEntireCanonicalAssemblyAndReadOrder) {
  auto implementations = Everything();
  implementations.push_back(
      Fake("t.dsv4_compress", Any(ir::Fused(FusedKind::kSoftmaxPoolCompress))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{.disabled = {"softmax_pool_compress"}};
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
      EXPECT_EQ(fusions::SoftmaxPoolCompressMatch(c, g->producer(out)).has_value(), mutation == 0);
      if (mutation != 0) continue;
      Policy on;
      on.enabled = {"softmax_pool_compress"};
      const SelectionView enabled(registry, kDevice, on, false, false);
      auto rewritten = ApplyPatterns(*g, enabled, nullptr);
      ASSERT_TRUE(rewritten) << (rewritten ? "" : rewritten.error().detail);
      ASSERT_EQ(rewritten->applied.size(), 1U);
      EXPECT_EQ(rewritten->applied[0].pattern, "softmax_pool_compress");
      const auto& result =
          rewritten->graph.node(rewritten->graph.producer(rewritten->graph.outputs()[0]));
      EXPECT_EQ(result.key, ir::Fused(FusedKind::kSoftmaxPoolCompress));
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
  implementations.push_back(Fake("t.hc_fold", Any(ir::Fused(FusedKind::kHcFold))));
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

TEST(PlannerPatternTest, GdnHistoryChecksStandaloneCommitTail) {
  for (int mutation : {0, 1, 2, 3, 4, 5, 6, 7, 8}) {
    SCOPED_TRACE(mutation);
    auto implementations = Everything();
    if (mutation != 6) {
      implementations.push_back(Fake("t.gdn_history", Any(ir::Fused(FusedKind::kGdnHistory))));
    }
    const auto registry = Registry(std::move(implementations));
    const Policy policy{.disabled = {"gdn_conv"}};
    const SelectionView selection(registry, kDevice, policy, false, false);

    ir::Builder b;
    const TensorId state = b.State(F32({18, 1}), ir::StateRef{.tensor = 0});
    const TensorId other_state = b.State(F32({18, 1}), ir::StateRef{.tensor = 1});
    const TensorId rows = b.Input(F32({6, 3, 1}));
    const TensorId history = b.Reshape(mutation == 3 ? other_state : state, ir::ShapeOf({3, 6, 1}));
    const TensorId transposed = b.Cont(b.Transpose(rows));
    const std::uint8_t cat_dim = mutation == 4 ? 1 : 0;
    const TensorId input =
        mutation == 2 ? b.Concat(transposed, history, 0) : b.Concat(history, transposed, cat_dim);
    const std::uint64_t offset = mutation == 1 ? 0 : 12;
    const std::array<std::uint64_t, 4> tail_strides =
        mutation == 4 ? std::array<std::uint64_t, 4>{4, 12, 144, 144}
                      : std::array<std::uint64_t, 4>{4, 24, 144, 144};
    const TensorId tail =
        b.View(input, ir::ShapeOf({3, 6, 1}), tail_strides, mutation == 4 ? 0 : offset);
    if (mutation == 5) {
      b.Keep(tail);
    }
    const TensorId kept = b.Reshape(b.Cont(tail), ir::ShapeOf({18, 1}));
    if (mutation == 7) {
      b.Keep(kept);
    }
    const TensorId ids = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({1})));
    const TensorId store = b.CacheStore(state, kept, ids);
    if (mutation == 8) {
      b.Keep(store);
    }
    const auto g = Finish(b);
    const MatchContext c(g, selection, nullptr);
    const bool valid = (mutation == 0);
    EXPECT_EQ(fusions::GdnHistoryMatch(c, g.producer(store)).has_value(), valid);
    EXPECT_EQ(fusions::GdnConvMatch(c, g.producer(store)).has_value(), valid);
    if (!valid) continue;

    Policy on;
    on.enabled = {"gdn_conv"};
    const SelectionView enabled(registry, kDevice, on, false, false);
    auto rewritten = ApplyPatterns(g, enabled, nullptr);
    ASSERT_TRUE(rewritten) << (rewritten ? "" : rewritten.error().detail);
    ASSERT_EQ(rewritten->applied.size(), 1U);
    EXPECT_EQ(rewritten->applied[0].pattern, "gdn_conv");
    EXPECT_EQ(rewritten->applied[0].replaced.size(), 8U);
    bool found_history = false;
    for (const ir::Node& n : rewritten->graph.nodes()) {
      if (n.key == ir::Fused(FusedKind::kGdnHistory)) {
        found_history = true;
        EXPECT_EQ(n.Params<ir::GdnHistoryParams>().taps, 3U);
      }
    }
    EXPECT_TRUE(found_history);

    auto disabled = ApplyPatterns(g, selection, nullptr);
    ASSERT_TRUE(disabled);
    EXPECT_TRUE(disabled->applied.empty());
  }
}

constexpr ir::InputParams kPositions4{.fact = ir::InputFact::kPositions, .sections = 4};

TEST(PlannerPatternTest, QsaSelectionChecksItsFixedWeightsMasksAndBias) {
  auto implementations = Everything();
  implementations.push_back(
      Fake("t.qsa_attention", Any(ir::Fused(FusedKind::kBlockSparseAttention))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{};
  const SelectionView selection(registry, kDevice, policy, false, false);
  for (int mutation : {0, 1, 2, 3, 4, 5, 6}) {
    SCOPED_TRACE(mutation);
    ir::Builder b;
    b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({4})), {}, kPositions4);
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
    EXPECT_EQ(fusions::BlockSparseAttentionMatch(c, g.producer(attn)).has_value(), mutation == 0);
  }
}

TEST(PlannerPatternTest, QsaPoolingDoesNotDropRotationFrequencyFactors) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.qsa_pool", Any(ir::Fused(FusedKind::kMeanPoolNormRopeStore))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{};
  const SelectionView selection(registry, kDevice, policy, false, false);
  for (bool frequencies : {false, true}) {
    SCOPED_TRACE(frequencies);
    ir::Builder b;
    b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({4})), {}, kPositions4);
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
    EXPECT_EQ(fusions::MeanPoolNormRopeStoreMatch(c, g.producer(stored)).has_value(), !frequencies);
  }
}

// The four-section positions a QSA pool reads are the input whose typed
// fact says so (ir::InputParams, in the digest), never the one named so: two
// graphs that digest alike plan alike.
TEST(PlannerPatternTest, QsaPositionsAreTheTypedInputNotANamedOne) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.qsa_pool", Any(ir::Fused(FusedKind::kMeanPoolNormRopeStore))));
  const auto registry = Registry(std::move(implementations));
  const Policy policy{};
  const SelectionView selection(registry, kDevice, policy, false, false);
  struct Built {
    ir::Graph graph;
    TensorId stored;
  };
  const auto build = [](std::string_view name, ir::InputParams fact) {
    ir::Builder b;
    b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({4})), name, fact);
    const TensorId raw = b.Input(F32({2, 4}));
    const TensorId blocks =
        b.State(ir::Dense(Encoding::kBF16, ir::ShapeOf({2, 2})), ir::StateRef{});
    const TensorId norm = b.Weight(F32({2}), W(0));
    const TensorId positions = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({2})));
    const TensorId cells = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({2})));
    const TensorId pooled = b.Pool(raw, {}, {.ratio = 2, .kind = ir::PoolKind::kMean});
    const TensorId weighted = b.Mul(b.RmsNorm(pooled, 1e-6F), norm);
    const TensorId rotated =
        b.Rope(b.Reshape(weighted, ir::ShapeOf({2, 1, 2})), positions, {}, {.dims = 2});
    const TensorId rounded = b.Convert(rotated, Encoding::kBF16);
    const TensorId stored = b.CacheStore(blocks, b.Reshape(rounded, ir::ShapeOf({2, 2})), cells);
    b.Keep(stored);
    return Built{Finish(b), stored};
  };
  const auto matches = [&](const Built& built) {
    const MatchContext c(built.graph, selection, nullptr);
    return fusions::MeanPoolNormRopeStoreMatch(c, built.graph.producer(built.stored)).has_value();
  };
  const Built typed = build("", kPositions4);
  const Built named = build("positions.4", {});
  const Built other = build("anything", {});
  const Built two = build("positions.4", {.fact = ir::InputFact::kPositions, .sections = 2});
  EXPECT_TRUE(matches(typed));
  EXPECT_TRUE(matches(build("renamed", kPositions4)));
  EXPECT_FALSE(matches(named));
  EXPECT_FALSE(matches(two));
  // Names stay out of the digest; the fact is in it.
  EXPECT_EQ(named.graph.Digest(), other.graph.Digest());
  EXPECT_EQ(typed.graph.Digest(), build("renamed", kPositions4).graph.Digest());
  EXPECT_NE(typed.graph.Digest(), named.graph.Digest());
  EXPECT_NE(typed.graph.Digest(), two.graph.Digest());
}

TEST(PlannerIrTest, InputFactsAreChecked) {
  for (const auto fact :
       {ir::InputParams{.fact = ir::InputFact::kNone, .sections = 1},
        ir::InputParams{.fact = ir::InputFact::kPositions, .sections = 0},
        ir::InputParams{.fact = static_cast<ir::InputFact>(9), .sections = 1},
        ir::InputParams{.fact = ir::InputFact::kPositions, .sections = 4, .reserved = {1, 0}}}) {
    ir::Builder b;
    b.Output(b.Scale(b.Input(F32({4}), {}, fact), 1.0F, 0.0F));
    EXPECT_FALSE(std::move(b).Finish().has_value());
  }
}

// A rewrite whose builder refuses it drops that match alone: the round runs
// again without it, its node stays primitive, and the other matches apply.
constexpr std::array kScaleAnchors = {Op::kScale};
constexpr std::array kRefusingTable = {Pattern{
    .name = "t.refusing",
    .kind = PatternKind::kRewrite,
    .result = ir::Primitive(Op::kScale),
    .anchors = kScaleAnchors,
    .match = [](const MatchContext&, ir::NodeId anchor) -> std::optional<Match> {
      return Match{.nodes = {anchor}};
    },
    .rewrite =
        [](RewriteContext& r, const Match& m, std::size_t) {
          const ir::Node& node = r.original().node(m.nodes[0]);
          if (node.Params<ir::ScaleParams>().scale == 2.0F) {
            r.builder().Refuse(ir::GraphError::kFormat, "a test's refused emission");
            return;
          }
          r.Copy(node);
        },
}};

TEST(PlannerPatternTest, ARefusedRewriteFallsBackToItsPrimitives) {
  const auto registry = Registry();
  const Policy policy{};
  const SelectionView view(registry, kDevice, policy, false, false);
  ir::Builder b;
  const TensorId x = b.Input(F32({8}));
  const TensorId first = b.Scale(x, 3.0F);
  const TensorId refused = b.Scale(first, 2.0F);
  b.Output(b.Scale(refused, 5.0F));
  const ir::Graph g = Finish(b);
  const auto out = ApplyPatternTable(kRefusingTable, g, view);
  ASSERT_TRUE(out.has_value()) << out.error().detail;
  ASSERT_EQ(out->dropped.size(), 1U);
  EXPECT_EQ(out->dropped[0].pattern, "t.refusing");
  EXPECT_EQ(out->dropped[0].anchor, g.producer(refused));
  EXPECT_NE(out->dropped[0].detail.find("refused emission"), std::string::npos);
  EXPECT_EQ(out->applied.size(), 2U);
  for (const Applied& a : out->applied) EXPECT_NE(a.replaced.front(), g.producer(refused));
  EXPECT_EQ(out->graph.Digest(), g.Digest());
  // The built-in table refuses none of it (StepPlan::dropped_rewrites).
  const auto plan = PlanOf(g.Clone(), registry);
  EXPECT_TRUE(plan.dropped_rewrites.empty());
}

// One write set (ir::ForEachWrittenSource): a cache store writes its cache,
// a fused node each aliased output's source (a mean-pool store its blocks,
// source 1, not its rotation's source 3).
TEST(PlannerIrTest, WriteSetsAreTheStoresAndAliasedSources) {
  const auto written = [](ir::OpKey key, std::uint8_t nsrc, std::uint8_t nout) {
    ir::Node n{.key = key, .nsrc = nsrc, .nout = nout};
    for (std::uint32_t i = 0; i < nsrc; ++i) n.src.at(i) = TensorId{i};
    for (std::uint32_t i = 0; i < nout; ++i) n.out.at(i) = TensorId{100 + i};
    std::vector<std::size_t> out;
    ir::ForEachWrittenSource(n, [&](std::size_t s) { out.push_back(s); });
    EXPECT_EQ(ir::WritesInPlace(n), !out.empty());
    return out;
  };
  using V = std::vector<std::size_t>;
  EXPECT_EQ(written(ir::Primitive(Op::kCacheStore), 3, 1), V{0});
  EXPECT_EQ(written(ir::Fused(FusedKind::kMeanPoolNormRopeStore), 6, 1), V{1});
  EXPECT_EQ(written(ir::Fused(FusedKind::kGdnStep), 7, 2), V{5});
  EXPECT_EQ(written(ir::Fused(FusedKind::kRopeStore), 4, 1), V{2});
  EXPECT_EQ(written(ir::Primitive(Op::kAdd), 2, 1), V{});
  EXPECT_EQ(written(ir::Fused(FusedKind::kRmsNormMul), 2, 1), V{});
  // An absent aliased source writes nothing.
  ir::Node absent{.key = ir::Fused(FusedKind::kGdnStep), .nsrc = 7, .nout = 2};
  EXPECT_FALSE(ir::WritesInPlace(absent));
}

// A routing implementation's pitched IDs (Capability::pitched_output) get
// the compact copy their byte-layout readers need; the planner reads the
// declared fact, never an implementation's name.
TEST(PlannerPlanTest, PitchedRouteIdsGetACompactCopyByTheirCapability) {
  for (const bool pitched : {false, true}) {
    SCOPED_TRACE(pitched);
    auto implementations = Everything();
    implementations.push_back(
        Fake("ggml.gemma.route.fused",
             ex::Capability{.op = ir::Primitive(Op::kRoute), .pitched_output = pitched},
             ir::Primitive(Op::kRoute)));
    implementations.push_back(
        Fake("t.cont", Any(ir::Primitive(Op::kCont)), ir::Primitive(Op::kCont)));
    const auto registry = Registry(std::move(implementations));
    ir::Builder b;
    const auto logits = b.Input(F32({4, 3}));
    const auto route = b.Route(logits, {}, {}, {}, {.normalize = true, .experts = 4, .used = 2});
    b.Output(b.Reshape(route[0], ir::ShapeOf({6})));
    b.Output(route[1]);
    const auto plan = PlanOf(Finish(b), registry);
    const auto names = Names(plan, registry);
    EXPECT_EQ(std::ranges::count(names, "ggml.gemma.route.fused"), 1);
    EXPECT_EQ(std::ranges::count(names, "t.cont"), pitched ? 1 : 0);
  }
}

TEST(PlannerPolicyTest, NumericalConstructionRejectsUnrepresentableResourceInventory) {
  EXPECT_EQ(NumericalConstructionBytes(0), 0U);
  EXPECT_FALSE(NumericalConstructionBytes(UINT64_MAX));
  ASSERT_TRUE(NumericalConstructionBytes(ir::Graph::kMaxNodes));
}

// Exact role-bound synthetic evidence for numerical admission controls only.
Policy RoleNumericalPolicy(const ir::Graph& graph, Policy policy, NumericalChange change,
                           std::span<const std::uint16_t> roles,
                           bool wrong_second_geometry = false) {
  const auto source = base::Sha256{}.Update("role-bound numerical control source").Finish();
  const auto semantic = graph.Digest();
  std::vector<NumericalQualification> entries;
  for (const auto& node : graph.nodes()) {
    if (node.key != ir::Primitive(Op::kMatMul) && node.key != ir::Primitive(Op::kMatMulId))
      continue;
    const auto& weight = graph.tensor(graph.root(node.src[0]));
    const auto* ref = std::get_if<ir::WeightRef>(&weight.binding);
    if (!ref || !std::ranges::contains(roles, ref->role)) continue;
    std::array<const ir::TensorDesc*, ir::kMaxSources> sources{};
    for (std::size_t i = 0; i < node.nsrc; ++i) sources[i] = &graph.tensor(node.src[i]);
    const std::array outputs{&graph.tensor(node.out[0])};
    auto shape = ShapeOfNode(node.key, std::span(sources).first(node.nsrc), outputs);
    auto max_n = weight.ne(1);
    if (wrong_second_geometry && ref->role == 1) ++max_n;
    entries.push_back({.change = change,
                       .evidence = base::Sha256{}.Update(std::to_string(ref->role)).Finish(),
                       .source = source,
                       .semantic = semantic,
                       .stage = 0,
                       .component = node.component,
                       .anchor = node.key,
                       .arch = kDevice.arch,
                       .min_rows = shape.rows,
                       .max_rows = shape.rows,
                       .max_k = weight.ne(0),
                       .max_n = max_n,
                       .exact_shape = true,
                       .weight_role = ref->role});
  }
  auto bound = NumericalQualifications::Bind(entries, source, semantic);
  EXPECT_TRUE(bound) << (bound ? "" : bound.error());
  if (bound) policy.numerical = std::move(*bound);
  policy.active_stage = 0;
  return policy;
}

TEST(PlannerPatternTest, QuantizedGluAuthenticatesBothProductsAndPreparedCompoundOperands) {
  for (const bool routed : {false, true}) {
    for (const bool already_fused : {false, true}) {
      for (unsigned authority = 0; authority < 8; ++authority) {
        SCOPED_TRACE(routed);
        SCOPED_TRACE(already_fused);
        SCOPED_TRACE(authority);
        auto implementations = Everything();
        // These controls have no incumbent conversion-equivalence fact.
        auto product = Any(ir::Primitive(routed ? Op::kMatMulId : Op::kMatMul));
        product.priority = 100;
        implementations.push_back(Fake("test.changed_product", product));
        auto fused = Any(ir::Fused(routed ? FusedKind::kMatMulIdGlu : FusedKind::kMatMulGlu));
        fused.priority = 100;
        implementations.push_back(Fake("test.changed_glu", fused));
        const auto registry = Registry(std::move(implementations));
        ir::Builder b;
        b.SetComponent(1);
        const auto x = b.Input(F32({256, 4}));
        std::array<TensorId, 2> weights{};
        for (std::uint16_t role = 0; role < 2; ++role) {
          auto ref = W(role);
          if ((authority == 3 || authority == 7) && role == 1) ref.stage = 1;
          auto format = Q8(256, 128);
          if (routed) format.shape = ir::ShapeOf({256, 128, 4});
          weights[role] = b.Weight(format, ref);
        }
        const auto ids =
            routed ? b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({2, 4}))) : TensorId{};
        const auto input = routed ? b.Reshape(x, ir::ShapeOf({256, 1, 4})) : x;
        const auto gate = routed ? b.MatMulId(weights[0], input, ids) : b.MatMul(weights[0], input);
        const auto up = routed ? b.MatMulId(weights[1], input, ids) : b.MatMul(weights[1], input);
        b.Output(b.Glu(gate, up, {}));
        auto primitive = Finish(b);
        std::array<ir::NodeId, 2> original_sites{};
        std::size_t found = 0;
        for (std::uint32_t i = 0; i < primitive.nodes().size(); ++i)
          if (primitive.node({i}).key == product.op) original_sites.at(found++) = {i};
        ASSERT_EQ(found, 2U);
        Policy policy{.disabled = {"float_mul_mat_glu", "mul_mat_glu", "paired_glu_products"}};
        const std::array<std::uint16_t, 1> first{0};
        const std::array<std::uint16_t, 2> both{0, 1};
        if (authority != 0 && authority != 4)
          policy = RoleNumericalPolicy(primitive, policy, NumericalChange::kQ8VectorReduction,
                                       authority == 1 ? std::span<const std::uint16_t>(first)
                                                      : std::span<const std::uint16_t>(both),
                                       authority == 5);
        if (authority == 6 || authority == 7) {
          std::vector<NumericalQualification> entries(policy.numerical->entries().begin(),
                                                      policy.numerical->entries().end());
          if (authority == 6) {
            // The review's counterexample supplied gate-side GLU/compound
            // grants rather than either original product contract.
            entries.resize(2);
            entries[0].anchor = ir::Primitive(Op::kGlu);
            entries[0].max_k = 128;
            entries[0].max_n = 0;
            entries[0].weight_role = 0;
            entries[1] = entries[0];
            entries[1].anchor = fused.op;
            entries[1].max_k = 256;
            entries[1].max_n = 128;
          } else {
            // Even a whole-component role wildcard cannot borrow authority
            // for a weight belonging to a different authenticated stage.
            for (auto& entry : entries) entry.weight_role = UINT16_MAX;
          }
          auto qualified = NumericalQualifications::Bind(entries, entries.front().source,
                                                         entries.front().semantic);
          ASSERT_TRUE(qualified) << (qualified ? "" : qualified.error());
          policy.numerical = std::move(*qualified);
        }
        if (authority == 4) policy.numerical_candidates = {NumericalChange::kQ8VectorReduction};
        auto graph = primitive.Clone();
        if (already_fused) {
          // Supply the fused input directly rather than relying on a round-0
          // proof transfer. Grants still describe each original product.
          ir::Builder direct;
          direct.SetComponent(1);
          const auto dx = direct.Input(F32({256, 4}));
          std::array<TensorId, 2> dw{};
          for (std::size_t i = 0; i < dw.size(); ++i)
            dw[i] = direct.Weight(primitive.tensor(weights[i]).format,
                                  std::get<ir::WeightRef>(primitive.tensor(weights[i]).binding));
          if (routed) {
            const auto di = direct.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({2, 4})));
            const std::array src{dw[0], dw[1], dx, di};
            direct.Output(direct.Fused(FusedKind::kMatMulIdGlu, src,
                                       ir::MoeExpertsParams{.glu = {}, .hidden = 128})[0]);
          } else {
            const std::array src{dw[0], dw[1], dx};
            direct.Output(direct.Fused(FusedKind::kMatMulGlu, src, ir::GluParams{})[0]);
          }
          graph = Finish(direct);
        }
        const auto plan = PlanOf(std::move(graph), registry, policy);
        const bool permitted = authority == 2 || authority == 4;
        const auto compound = std::ranges::find(plan.graph.nodes(), fused.op, &ir::Node::key);
        if (already_fused || permitted) {
          ASSERT_NE(compound, plan.graph.nodes().end());
          EXPECT_EQ(plan.graph.tensor(compound->src[2]).format.encoding == Encoding::kQ8_1,
                    permitted);
        }
        std::size_t preparations = 0, paired = 0;
        for (const auto& applied : plan.applied) {
          if (applied.pattern != "vecq_glu" && applied.pattern != "vecq_inputs") continue;
          if (applied.pattern == "vecq_glu") {
            ++paired;
            ASSERT_EQ(applied.numerical_proofs.size(), 2U);
            EXPECT_EQ(applied.numerical_proofs[0].site, original_sites[0]);
            EXPECT_EQ(applied.numerical_proofs[1].site, original_sites[1]);
          }
          if (applied.pattern == "vecq_inputs" && permitted) {
            ++preparations;
            ASSERT_EQ(applied.numerical_proofs.size(), 2U);
            EXPECT_EQ(applied.numerical_proofs[0].weight_operand, 0U);
            EXPECT_EQ(applied.numerical_proofs[1].weight_operand, 1U);
          }
          for (const auto& proof : applied.numerical_proofs) {
            EXPECT_EQ(proof.candidate, authority == 4);
            EXPECT_EQ(proof.actual.rows, 4U);
            EXPECT_EQ(proof.actual.k, 256U);
            EXPECT_EQ(proof.actual.n, 128U);
            ASSERT_TRUE(proof.observed.weight);
            EXPECT_EQ(proof.observed.weight->component, 1U);
            EXPECT_LT(proof.observed.weight->role, 2U);
            EXPECT_EQ(
                proof.observed.weight->stage,
                ((authority == 3 || authority == 7) && proof.observed.weight->role == 1) ? 1U : 0U);
            EXPECT_EQ(proof.observed.operand_format.encoding, Encoding::kQ8_0);
            EXPECT_EQ(proof.observed.leaf_format.encoding, Encoding::kQ8_0);
            if (!proof.candidate) {
              EXPECT_EQ(proof.qualification.stage, 0U);
              EXPECT_EQ(proof.qualification.anchor, product.op);
              EXPECT_NE(proof.qualification.evidence, base::Sha256Digest{});
            }
          }
        }
        EXPECT_EQ(paired, !already_fused && permitted ? 1U : 0U);
        EXPECT_EQ(preparations, permitted ? 1U : 0U);
        if (authority == 1 || authority == 3 || authority == 5 || authority == 6 || authority == 7)
          EXPECT_FALSE(plan.missing_numerical.empty());
      }
    }
  }
}

TEST(PlannerPatternTest, QuantizedGluCannotEraseAnExplicitProductConstraint) {
  auto implementations = Everything();
  auto product = Any(ir::Primitive(Op::kMatMul));
  product.priority = 100;
  implementations.push_back(Fake("test.changed_product", product));
  implementations.push_back(Fake("test.changed_glu", Any(ir::Fused(FusedKind::kMatMulGlu))));
  const auto registry = Registry(std::move(implementations));
  ir::Builder builder;
  builder.SetComponent(1);
  const auto x = builder.Input(F32({256, 4}));
  const auto gate = builder.MatMul(builder.Weight(Q8(256, 128), W(0)), x);
  const auto up =
      builder.MatMul(builder.Weight(Q8(256, 128), W(1)), x, {.accumulate = ir::Accumulate::kF32});
  builder.Output(builder.Glu(gate, up, {}));
  auto graph = Finish(builder);
  const std::array<std::uint16_t, 2> roles{0, 1};
  auto policy = RoleNumericalPolicy(graph, {}, NumericalChange::kQ8VectorReduction, roles);
  const auto plan = PlanOf(std::move(graph), registry, policy);
  EXPECT_EQ(std::ranges::count(plan.applied, "vecq_glu", &Applied::pattern), 0);
  EXPECT_EQ(std::ranges::count(plan.graph.nodes(), ir::Primitive(Op::kGlu), &ir::Node::key), 1);
  const auto constrained = std::ranges::find_if(plan.graph.nodes(), [](const auto& node) {
    return node.key == ir::Primitive(Op::kMatMul) &&
           node.template Params<ir::ProductParams>().accumulate == ir::Accumulate::kF32;
  });
  ASSERT_NE(constrained, plan.graph.nodes().end());
  EXPECT_EQ(plan.graph.tensor(constrained->src[1]).format.encoding, Encoding::kQ8_1);
}

TEST(PlannerPatternTest, SharedChangedPreparationQualifiesEveryRecipientAndRetainsEachProof) {
  for (const auto encoding : {Encoding::kMxfp8, Encoding::kQ8_0}) {
    for (unsigned authority = 0; authority < 6; ++authority) {
      SCOPED_TRACE(static_cast<unsigned>(encoding));
      SCOPED_TRACE(authority);
      auto implementations = Everything();
      auto product = Any(ir::Primitive(Op::kMatMul));
      product.priority = 100;  // No incumbent-equivalence fact, for either input encoding.
      implementations.push_back(Fake("t.changed_product", product));
      implementations.push_back(Fake("t.prepare", Any(ir::Primitive(Op::kPrepare))));
      const auto registry = Registry(std::move(implementations));
      ir::Builder b;
      b.SetComponent(1);
      const auto x = b.Input(F32({128, 2}));
      std::array<ir::NodeId, 2> sites{};
      for (std::uint16_t role = 0; role < 2; ++role) {
        auto ref = W(role);
        if (authority == 3 && role == 1)
          ref.stage = 1;  // A foreign-stage weight stays unqualified.
        const auto format = encoding == Encoding::kMxfp8
                                ? format::TensorFormat{.encoding = encoding,
                                                       .layout = Layout::kScaleAtoms,
                                                       .shape = ir::ShapeOf({128, 8 + 8ULL * role})}
                                : Q8(128, 8 + 8ULL * role);
        const auto w = b.Weight(format, ref);
        const auto y = b.MatMul(w, x);
        b.Output(b.Scale(y, 2));
      }
      const auto graph = Finish(b);
      // Resolve primitive IDs before planning/remapping.
      std::size_t at = 0;
      for (std::uint32_t i = 0; i < graph.nodes().size(); ++i)
        if (graph.node({i}).key == ir::Primitive(Op::kMatMul)) sites.at(at++) = {i};
      const auto change = encoding == Encoding::kMxfp8 ? NumericalChange::kMxfp8ProductInput
                                                       : NumericalChange::kQ8VectorReduction;
      const std::array<std::uint16_t, 2> both{0, 1};
      const std::array<std::uint16_t, 1> first{0};
      Policy policy;
      if (authority != 0 && authority != 4)
        policy = RoleNumericalPolicy(graph, policy, change,
                                     authority == 1 ? std::span<const std::uint16_t>(first)
                                                    : std::span<const std::uint16_t>(both),
                                     authority == 5);
      if (authority == 4) policy.numerical_candidates = {change};
      const auto plan = PlanOf(graph.Clone(), registry, policy);
      std::size_t changed = 0, proof_count = 0;
      for (const auto& node : plan.graph.nodes())
        if (node.key == ir::Primitive(Op::kMatMul) &&
            plan.graph.tensor(node.src[1]).format.encoding != Encoding::kF32)
          ++changed;
      EXPECT_EQ(changed, authority == 0
                             ? 0U
                             : (authority == 1 || authority == 3 || authority == 5 ? 1U : 2U));
      for (const auto& applied : plan.applied) {
        for (const auto& proof : applied.numerical_proofs) {
          EXPECT_TRUE(std::ranges::contains(sites, proof.site));
          if (!proof.candidate)
            EXPECT_EQ(proof.site, sites.at(proof.qualification.weight_role == 1 ? 1 : 0));
          EXPECT_EQ(proof.actual.rows, 2U);
          EXPECT_EQ(proof.actual.k, 128U);
          EXPECT_EQ(proof.candidate, authority == 4);
          if (!proof.candidate) {
            EXPECT_EQ(proof.qualification.stage, 0U);
            EXPECT_EQ(proof.actual.n, proof.qualification.max_n);
            EXPECT_NE(proof.qualification.evidence, base::Sha256Digest{});
          }
          ++proof_count;
        }
      }
      EXPECT_EQ(proof_count, changed);
      EXPECT_EQ(std::ranges::count(plan.graph.nodes(), ir::Primitive(Op::kPrepare), &ir::Node::key),
                authority == 0 ? 0 : 1);
      if (authority == 1 || authority == 3 || authority == 5)
        EXPECT_FALSE(plan.missing_numerical.empty());
    }
  }
}

TEST(PlannerPatternTest, GroupedVecQRequiresTheActualProductQualification) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.grouped", Any(ir::Fused(FusedKind::kGroupedMatMul))));
  implementations.push_back(Fake("t.cont", Any(ir::Primitive(Op::kCont))));
  const auto registry = Registry(std::move(implementations));
  ir::Builder b;
  b.SetComponent(1);
  const auto leaf = b.Weight(Q8(64, 48), W(7));
  const auto w = b.Reshape(leaf, ir::ShapeOf({64, 16, 3}));
  const auto x = b.Input(F32({64, 3, 2}));
  const auto y = b.MatMul(w, b.Permute(x, {0, 2, 1, 3}));
  b.Output(b.Cont(b.Permute(y, {0, 2, 1, 3}), ir::ShapeOf({48, 2})));
  const auto graph = Finish(b);
  const auto product = graph.producer(y);
  const auto grouped_product = fusions::GroupedProductOf(graph, product);
  ASSERT_TRUE(grouped_product.has_value());
  EXPECT_EQ(grouped_product->product, product);
  EXPECT_EQ(grouped_product->leaf, leaf);
  EXPECT_EQ(grouped_product->weight, w);
  EXPECT_EQ(grouped_product->input, x);
  const std::array<std::uint16_t, 1> role{7};
  // 0: unqualified, 1: concrete full-leaf role qualification [64, 48],
  // 2: explicit candidate proposal, 3: wildcard role [64, 48],
  // 4: unprojected per-group [64, 16] without full-leaf authority,
  // 5: non-exact full-leaf [64, 48].
  for (unsigned authority = 0; authority < 6; ++authority) {
    SCOPED_TRACE(authority);
    Policy policy;
    if (authority == 1 || authority >= 3) {
      policy = RoleNumericalPolicy(graph, policy, NumericalChange::kQ8VectorReduction, role);
      if (authority >= 3) {
        std::vector<NumericalQualification> entries(policy.numerical->entries().begin(),
                                                    policy.numerical->entries().end());
        ASSERT_EQ(entries.size(), 1U);
        if (authority == 3) entries[0].weight_role = UINT16_MAX;
        if (authority == 4) entries[0].max_n = 16;
        if (authority == 5) {
          entries[0].exact_shape = false;
          entries[0].exact_dimensions = false;
        }
        auto bound = NumericalQualifications::Bind(entries, entries[0].source, entries[0].semantic);
        ASSERT_TRUE(bound) << (bound ? "" : bound.error());
        policy.numerical = std::move(*bound);
      }
    }
    if (authority == 2) policy.numerical_candidates = {NumericalChange::kQ8VectorReduction};
    const SelectionView view(registry, kDevice, policy, false, false);
    const auto matched =
        view.Qualification(NumericalChange::kQ8VectorReduction, graph, graph.node(product));
    EXPECT_EQ(matched.has_value(), authority == 1);
    if (matched) {
      EXPECT_EQ(matched->weight_role, 7U);
      EXPECT_EQ(matched->max_k, 64U);
      EXPECT_EQ(matched->max_n, 16U);
    }
    const auto plan = PlanOf(graph.Clone(), registry, policy);
    const auto found = std::ranges::find(plan.applied, "grouped_vecq", &Applied::pattern);
    const bool permitted = authority == 1 || authority == 2;
    EXPECT_EQ(found != plan.applied.end(), permitted);
    if (found != plan.applied.end()) {
      ASSERT_EQ(found->numerical_proofs.size(), 1U);
      const auto& proof = found->numerical_proofs[0];
      EXPECT_EQ(proof.site, product);
      EXPECT_EQ(proof.candidate, authority == 2);
      EXPECT_EQ(proof.grouped_projection, authority == 1);
      EXPECT_EQ(proof.actual.rows, 2U);
      EXPECT_EQ(proof.actual.k, 64U);
      EXPECT_EQ(proof.actual.n, 16U);
      ASSERT_TRUE(proof.observed.weight.has_value());
      EXPECT_EQ(*proof.observed.weight, W(7));
      EXPECT_EQ(proof.observed.operand_format.shape.ne,
                (std::array<std::uint64_t, 4>{64, 16, 3, 1}));
      EXPECT_EQ(proof.observed.leaf_format.shape.ne, (std::array<std::uint64_t, 4>{64, 48, 1, 1}));
      if (authority == 1) {
        EXPECT_EQ(proof.qualification.max_k, 64U);
        EXPECT_EQ(proof.qualification.max_n, 16U);
        const auto bytes = PlanReportInventory::StorageBytes(plan, registry);
        ASSERT_TRUE(bytes.has_value());
        const auto inventory = PlanReportInventory::Capture(plan, registry, *bytes);
        ASSERT_TRUE(inventory.has_value());
        std::string streamed, captured;
        StreamPlanReport(plan, registry, [&](std::string_view part) { streamed.append(part); });
        inventory->Stream([&](std::string_view part) { captured.append(part); });
        EXPECT_EQ(streamed, captured);
        bool saw_pattern = false;
        std::string_view remaining = streamed;
        while (!remaining.empty()) {
          const auto end = remaining.find('\n');
          ASSERT_NE(end, std::string_view::npos);
          auto json = base::json::Parse(remaining.substr(0, end));
          ASSERT_TRUE(json.has_value());
          if (json->root().find("kind")->string() == "pattern" &&
              json->root().find("name")->string() == "grouped_vecq") {
            const auto recipient = json->root().find("numerical_recipients")->at(0);
            EXPECT_EQ(recipient.find("rows")->int64(), 2);
            EXPECT_EQ(recipient.find("k")->int64(), 64);
            EXPECT_EQ(recipient.find("n")->int64(), 16);
            EXPECT_TRUE(recipient.find("grouped_projection")->boolean());
            const auto envelope = recipient.find("qualified_envelope");
            ASSERT_TRUE(envelope.has_value());
            EXPECT_EQ(envelope->at(2).int64(), 64);
            EXPECT_EQ(envelope->at(3).int64(), 16);
            const auto leaf_envelope = recipient.find("qualified_leaf_envelope");
            ASSERT_TRUE(leaf_envelope.has_value());
            EXPECT_EQ(leaf_envelope->at(2).int64(), 64);
            EXPECT_EQ(leaf_envelope->at(3).int64(), 48);
            const auto observed = recipient.find("observed");
            ASSERT_TRUE(observed.has_value());
            EXPECT_EQ(observed->find("weight_operand")->find("shape")->at(1).int64(), 16);
            EXPECT_EQ(observed->find("weight_operand")->find("shape")->at(2).int64(), 3);
            EXPECT_EQ(observed->find("weight_leaf")->find("shape")->at(1).int64(), 48);
            saw_pattern = true;
          }
          remaining.remove_prefix(end + 1);
        }
        EXPECT_TRUE(saw_pattern);
      }
    } else {
      EXPECT_FALSE(plan.missing_numerical.empty());
    }
  }
}

TEST(PlannerPatternTest,
     GroupedQualificationProjectionRejectsMalformedUnclosedAndSharedLeafMisuse) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.grouped", Any(ir::Fused(FusedKind::kGroupedMatMul))));
  implementations.push_back(Fake("t.cont", Any(ir::Primitive(Op::kCont))));
  implementations.push_back(Fake("t.prepare", Any(ir::Primitive(Op::kPrepare))));
  auto changed = Any(ir::Primitive(Op::kMatMul));
  changed.priority = 100;
  implementations.push_back(Fake("t.changed_product", changed));
  const auto registry = Registry(std::move(implementations));
  const std::array<std::uint16_t, 1> role{7};

  // 0: valid closed grouped product, 1: non-zero view offset on weight,
  // 2: non-contiguous weight view, 3: wrong input permute axes,
  // 4: missing input permute, 5: wrong output permute axes,
  // 6: missing Cont assembly, 7: mismatched Cont shape,
  // 8: groups < 2, 9: unclosed product output escaping to graph outputs.
  for (unsigned mutation = 0; mutation < 10; ++mutation) {
    SCOPED_TRACE(mutation);
    ir::Builder b;
    b.SetComponent(1);
    const std::uint64_t groups = mutation == 8 ? 1U : 3U;
    const std::uint64_t n_per = mutation == 8 ? 48U : 16U;
    const auto leaf = b.Weight(Q8(64, 48), W(7));
    TensorId w{};
    if (mutation == 1) {
      const auto sub = b.View(leaf, ir::ShapeOf({64, 32}), ir::NaturalStrides(Q8(64, 32)), 68);
      w = b.Reshape(sub, ir::ShapeOf({64, 16, 2}));
    } else if (mutation == 2) {
      w = b.View(leaf, ir::ShapeOf({64, 8, 3}), {34, 136, 1088, 3264}, 0);
    } else {
      w = b.Reshape(leaf, ir::ShapeOf({64, n_per, groups}));
    }
    ASSERT_TRUE(b.ok()) << (b.error() ? b.error()->detail : "");
    const std::uint64_t w_groups = b.tensor(w).ne(2);
    const std::uint64_t w_n = b.tensor(w).ne(1);
    const bool swapped_x = mutation == 3 || mutation == 4;
    const auto x = b.Input(F32({64, swapped_x ? 2U : w_groups, swapped_x ? w_groups : 2U}));
    const auto x_in = mutation == 4
                          ? x
                          : b.Permute(x, mutation == 3 ? std::array<std::uint8_t, 4>{0, 1, 2, 3}
                                                       : std::array<std::uint8_t, 4>{0, 2, 1, 3});
    const auto y = b.MatMul(w, x_in);
    if (mutation == 9) b.Output(y);
    const auto y_out = b.Permute(y, mutation == 5 ? std::array<std::uint8_t, 4>{0, 1, 2, 3}
                                                  : std::array<std::uint8_t, 4>{0, 2, 1, 3});
    if (mutation == 6) {
      b.Output(y_out);
    } else {
      const auto cont_shape =
          mutation == 7 ? ir::ShapeOf({w_n * w_groups * 2, 1}) : ir::ShapeOf({w_n * w_groups, 2});
      b.Output(b.Cont(y_out, cont_shape));
    }
    const auto graph = Finish(b);
    const auto product = graph.producer(y);
    EXPECT_EQ(fusions::GroupedProductOf(graph, product).has_value(), mutation == 0);
    const auto policy = RoleNumericalPolicy(graph, {}, NumericalChange::kQ8VectorReduction, role);
    const SelectionView view(registry, kDevice, policy, false, false);
    EXPECT_EQ(view.Qualification(NumericalChange::kQ8VectorReduction, graph, graph.node(product))
                  .has_value(),
              mutation == 0);
    const auto plan = PlanOf(graph.Clone(), registry, policy);
    const auto found = std::ranges::find(plan.applied, "grouped_vecq", &Applied::pattern);
    EXPECT_EQ(found != plan.applied.end(), mutation == 0);
  }

  // Shared-leaf isolation: a single weight leaf [64, 48] consumed in the same
  // plan by a valid dense [64, 48] MatMul, a valid closed grouped [64, 16, 3]
  // product, and malformed/unclosed consumers qualifies the dense consumer with
  // full-leaf [1, 1, 64, 48] (grouped_projection == false), the closed grouped
  // consumer with projected [1, 1, 64, 16] (grouped_projection == true and
  // qualified_leaf_envelope == [1, 1, 64, 48]), and rejects only the malformed
  // consumer sites.
  ir::Builder shared;
  shared.SetComponent(1);
  const auto leaf = shared.Weight(Q8(64, 48), W(7));
  const auto x_dense = shared.Input(F32({64, 1}));
  const auto y_dense = shared.MatMul(leaf, x_dense);
  shared.Output(y_dense);
  const auto w_grouped = shared.Reshape(leaf, ir::ShapeOf({64, 16, 3}));
  const auto x_grouped = shared.Input(F32({64, 3, 1}));
  const auto y_grouped = shared.MatMul(w_grouped, shared.Permute(x_grouped, {0, 2, 1, 3}));
  shared.Output(shared.Cont(shared.Permute(y_grouped, {0, 2, 1, 3}), ir::ShapeOf({48, 1})));
  const auto w_slice = shared.View(leaf, ir::ShapeOf({64, 16}), ir::NaturalStrides(Q8(64, 48)), 0);
  const auto x_slice = shared.Input(F32({64, 1}));
  const auto y_slice = shared.MatMul(w_slice, x_slice);
  shared.Output(y_slice);
  const auto w_unclosed = shared.Reshape(leaf, ir::ShapeOf({64, 16, 3}));
  const auto x_unclosed = shared.Input(F32({64, 3, 1}));
  const auto y_unclosed = shared.MatMul(w_unclosed, shared.Permute(x_unclosed, {0, 2, 1, 3}));
  shared.Output(y_unclosed);
  shared.Output(shared.Cont(shared.Permute(y_unclosed, {0, 2, 1, 3}), ir::ShapeOf({48, 1})));
  const auto shared_graph = Finish(shared);
  const auto dense_site = shared_graph.producer(y_dense);
  const auto grouped_site = shared_graph.producer(y_grouped);
  const auto slice_site = shared_graph.producer(y_slice);
  const auto unclosed_site = shared_graph.producer(y_unclosed);
  EXPECT_TRUE(fusions::GroupedProductOf(shared_graph, grouped_site).has_value());
  EXPECT_FALSE(fusions::GroupedProductOf(shared_graph, slice_site).has_value());
  EXPECT_FALSE(fusions::GroupedProductOf(shared_graph, unclosed_site).has_value());

  // Bind a single concrete full-leaf [1, 1, 64, 48] qualification for role 7.
  const auto source = base::Sha256{}.Update("shared-leaf control source").Finish();
  const auto semantic = shared_graph.Digest();
  const NumericalQualification full_leaf_grant{
      .change = NumericalChange::kQ8VectorReduction,
      .evidence = base::Sha256{}.Update("role-7-full-leaf").Finish(),
      .source = source,
      .semantic = semantic,
      .stage = 0,
      .component = 1,
      .anchor = ir::Primitive(Op::kMatMul),
      .arch = kDevice.arch,
      .min_rows = 1,
      .max_rows = 1,
      .max_k = 64,
      .max_n = 48,
      .exact_dimensions = true,
      .weight_role = 7};
  auto bound = NumericalQualifications::Bind(std::array{full_leaf_grant}, source, semantic);
  ASSERT_TRUE(bound.has_value());
  const Policy policy{.numerical = std::move(*bound), .active_stage = 0};
  const SelectionView view(registry, kDevice, policy, false, false);
  const auto q_dense = view.Qualification(NumericalChange::kQ8VectorReduction, shared_graph,
                                          shared_graph.node(dense_site));
  ASSERT_TRUE(q_dense.has_value());
  EXPECT_EQ(q_dense->max_k, 64U);
  EXPECT_EQ(q_dense->max_n, 48U);
  const auto q_grouped = view.Qualification(NumericalChange::kQ8VectorReduction, shared_graph,
                                            shared_graph.node(grouped_site));
  ASSERT_TRUE(q_grouped.has_value());
  EXPECT_EQ(q_grouped->max_k, 64U);
  EXPECT_EQ(q_grouped->max_n, 16U);
  EXPECT_FALSE(view.Qualification(NumericalChange::kQ8VectorReduction, shared_graph,
                                  shared_graph.node(slice_site))
                   .has_value());
  EXPECT_FALSE(view.Qualification(NumericalChange::kQ8VectorReduction, shared_graph,
                                  shared_graph.node(unclosed_site))
                   .has_value());

  const auto plan = PlanOf(shared_graph.Clone(), registry, policy);
  const auto found_grouped = std::ranges::find(plan.applied, "grouped_vecq", &Applied::pattern);
  ASSERT_NE(found_grouped, plan.applied.end());
  ASSERT_EQ(found_grouped->numerical_proofs.size(), 1U);
  EXPECT_EQ(found_grouped->numerical_proofs[0].site, grouped_site);
  EXPECT_TRUE(found_grouped->numerical_proofs[0].grouped_projection);
  EXPECT_EQ(found_grouped->numerical_proofs[0].qualification.max_k, 64U);
  EXPECT_EQ(found_grouped->numerical_proofs[0].qualification.max_n, 16U);

  const auto found_dense = std::ranges::find(plan.applied, "vecq_inputs", &Applied::pattern);
  ASSERT_NE(found_dense, plan.applied.end());
  ASSERT_EQ(found_dense->numerical_proofs.size(), 1U);
  EXPECT_EQ(found_dense->numerical_proofs[0].site, dense_site);
  EXPECT_FALSE(found_dense->numerical_proofs[0].grouped_projection);
  EXPECT_EQ(found_dense->numerical_proofs[0].qualification.max_k, 64U);
  EXPECT_EQ(found_dense->numerical_proofs[0].qualification.max_n, 48U);

  std::string streamed;
  StreamPlanReport(plan, registry, [&](std::string_view part) { streamed.append(part); });
  bool saw_grouped_report = false, saw_dense_report = false;
  std::string_view remaining = streamed;
  while (!remaining.empty()) {
    const auto end = remaining.find('\n');
    ASSERT_NE(end, std::string_view::npos);
    auto json = base::json::Parse(remaining.substr(0, end));
    ASSERT_TRUE(json.has_value());
    if (json->root().find("kind")->string() == "pattern") {
      const auto name = json->root().find("name")->string();
      if (name == "grouped_vecq") {
        const auto recipient = json->root().find("numerical_recipients")->at(0);
        EXPECT_TRUE(recipient.find("grouped_projection")->boolean());
        const auto env = recipient.find("qualified_envelope");
        ASSERT_TRUE(env.has_value());
        EXPECT_EQ(env->at(0).int64(), 1);
        EXPECT_EQ(env->at(1).int64(), 1);
        EXPECT_EQ(env->at(2).int64(), 64);
        EXPECT_EQ(env->at(3).int64(), 16);
        const auto leaf_env = recipient.find("qualified_leaf_envelope");
        ASSERT_TRUE(leaf_env.has_value());
        EXPECT_EQ(leaf_env->at(0).int64(), 1);
        EXPECT_EQ(leaf_env->at(1).int64(), 1);
        EXPECT_EQ(leaf_env->at(2).int64(), 64);
        EXPECT_EQ(leaf_env->at(3).int64(), 48);
        saw_grouped_report = true;
      } else if (name == "vecq_inputs") {
        const auto recipient = json->root().find("numerical_recipients")->at(0);
        EXPECT_FALSE(recipient.find("grouped_projection").has_value());
        EXPECT_FALSE(recipient.find("qualified_leaf_envelope").has_value());
        const auto env = recipient.find("qualified_envelope");
        ASSERT_TRUE(env.has_value());
        EXPECT_EQ(env->at(0).int64(), 1);
        EXPECT_EQ(env->at(1).int64(), 1);
        EXPECT_EQ(env->at(2).int64(), 64);
        EXPECT_EQ(env->at(3).int64(), 48);
        saw_dense_report = true;
      }
    }
    remaining.remove_prefix(end + 1);
  }
  EXPECT_TRUE(saw_grouped_report);
  EXPECT_TRUE(saw_dense_report);

  // Only the malformed sites fail qualification; the dense MatMul takes Q8_1,
  // the closed grouped product becomes kGroupedMatMul, and the two malformed
  // MatMul sites remain F32.
  EXPECT_EQ(
      std::ranges::count(plan.graph.nodes(), ir::Fused(FusedKind::kGroupedMatMul), &ir::Node::key),
      1);
  EXPECT_EQ(std::ranges::count(plan.graph.nodes(), ir::Primitive(Op::kPrepare), &ir::Node::key), 1);
  std::size_t q8_products = 0, f32_products = 0;
  for (const auto& node : plan.graph.nodes()) {
    if (node.key != ir::Primitive(Op::kMatMul)) continue;
    if (plan.graph.tensor(node.src[1]).format.encoding == Encoding::kQ8_1)
      ++q8_products;
    else if (plan.graph.tensor(node.src[1]).format.encoding == Encoding::kF32)
      ++f32_products;
  }
  EXPECT_EQ(q8_products, 1U);
  EXPECT_EQ(f32_products, 2U);
  EXPECT_TRUE(plan.missing_numerical.empty());
}

TEST(PlannerPatternTest, MqaOutAGroupedAndDenseVecQQualifyAtOneRowAndRefuseMultiRow) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.grouped", Any(ir::Fused(FusedKind::kGroupedMatMul))));
  implementations.push_back(Fake("t.cont", Any(ir::Primitive(Op::kCont))));
  implementations.push_back(Fake("t.prepare", Any(ir::Primitive(Op::kPrepare))));
  auto changed = Any(ir::Primitive(Op::kMatMul));
  changed.priority = 100;
  implementations.push_back(Fake("t.changed_product", changed));
  const auto registry = Registry(std::move(implementations));

  // Real MQA output projection geometry: out_a (role 6) full leaf [1536, 32768]
  // reshaped to [1536, 2048, 16] for grouped_vecq, feeding out_b (role 7)
  // [32768, 7168] for vecq_inputs, under a single-row (rows = {1}) incumbent
  // qualification.
  for (const std::uint64_t rows : {1ULL, 2ULL, 4ULL}) {
    SCOPED_TRACE(rows);
    ir::Builder b;
    b.SetComponent(3);
    const auto out_a =
        b.Weight(Q8(1536, 32768), ir::WeightRef{.stage = 0, .component = 3, .role = 6});
    const auto out_b =
        b.Weight(Q8(32768, 7168), ir::WeightRef{.stage = 0, .component = 3, .role = 7});
    const auto x = b.Input(F32({1536, 16, rows}));
    const auto groups = b.Reshape(out_a, ir::ShapeOf({1536, 2048, 16}));
    const auto low_grouped = b.MatMul(groups, b.Permute(x, {0, 2, 1, 3}));
    const auto low = b.Cont(b.Permute(low_grouped, {0, 2, 1, 3}), ir::ShapeOf({32768, rows}));
    const auto y = b.MatMul(out_b, low);
    b.Output(y);
    const auto graph = Finish(b);
    const auto out_a_site = graph.producer(low_grouped);
    const auto out_b_site = graph.producer(y);

    const auto source = base::Sha256{}.Update("mqa-out-a-incumbent-source").Finish();
    const auto semantic = graph.Digest();
    const std::array<NumericalQualification, 2> grants{
        NumericalQualification{.change = NumericalChange::kQ8VectorReduction,
                               .evidence = base::Sha256{}.Update("mqa-out-a").Finish(),
                               .source = source,
                               .semantic = semantic,
                               .stage = 0,
                               .component = 3,
                               .anchor = ir::Primitive(Op::kMatMul),
                               .arch = kDevice.arch,
                               .min_rows = 1,
                               .max_rows = 1,
                               .max_k = 1536,
                               .max_n = 32768,
                               .exact_dimensions = true,
                               .incumbent_recipe = true,
                               .weight_role = 6},
        NumericalQualification{.change = NumericalChange::kQ8VectorReduction,
                               .evidence = base::Sha256{}.Update("mqa-out-b").Finish(),
                               .source = source,
                               .semantic = semantic,
                               .stage = 0,
                               .component = 3,
                               .anchor = ir::Primitive(Op::kMatMul),
                               .arch = kDevice.arch,
                               .min_rows = 1,
                               .max_rows = 1,
                               .max_k = 32768,
                               .max_n = 7168,
                               .exact_dimensions = true,
                               .incumbent_recipe = true,
                               .weight_role = 7},
    };
    auto bound = NumericalQualifications::Bind(grants, source, semantic);
    ASSERT_TRUE(bound.has_value());
    const Policy policy{.numerical = std::move(*bound), .active_stage = 0};
    const auto plan = PlanOf(graph.Clone(), registry, policy);
    const auto grouped = std::ranges::find(plan.applied, "grouped_vecq", &Applied::pattern);
    const auto vecq = std::ranges::find(plan.applied, "vecq_inputs", &Applied::pattern);
    if (rows == 1) {
      ASSERT_NE(grouped, plan.applied.end());
      ASSERT_NE(vecq, plan.applied.end());
      EXPECT_TRUE(plan.missing_numerical.empty());
      ASSERT_EQ(grouped->numerical_proofs.size(), 1U);
      EXPECT_EQ(grouped->numerical_proofs[0].site, out_a_site);
      EXPECT_TRUE(grouped->numerical_proofs[0].grouped_projection);
      EXPECT_EQ(grouped->numerical_proofs[0].actual.rows, 1U);
      EXPECT_EQ(grouped->numerical_proofs[0].actual.k, 1536U);
      EXPECT_EQ(grouped->numerical_proofs[0].actual.n, 2048U);
      EXPECT_EQ(grouped->numerical_proofs[0].qualification.max_k, 1536U);
      EXPECT_EQ(grouped->numerical_proofs[0].qualification.max_n, 2048U);
      ASSERT_EQ(vecq->numerical_proofs.size(), 1U);
      EXPECT_EQ(vecq->numerical_proofs[0].site, out_b_site);
      EXPECT_FALSE(vecq->numerical_proofs[0].grouped_projection);
      EXPECT_EQ(vecq->numerical_proofs[0].actual.rows, 1U);
      EXPECT_EQ(vecq->numerical_proofs[0].actual.k, 32768U);
      EXPECT_EQ(vecq->numerical_proofs[0].actual.n, 7168U);
      EXPECT_EQ(std::ranges::count(plan.graph.nodes(), ir::Fused(FusedKind::kGroupedMatMul),
                                   &ir::Node::key),
                1);
      EXPECT_EQ(std::ranges::count(plan.graph.nodes(), ir::Primitive(Op::kPrepare), &ir::Node::key),
                1);
      const auto final_matmul =
          std::ranges::find(plan.graph.nodes(), ir::Primitive(Op::kMatMul), &ir::Node::key);
      ASSERT_NE(final_matmul, plan.graph.nodes().end());
      EXPECT_EQ(plan.graph.tensor(final_matmul->src[1]).format.encoding, Encoding::kQ8_1);
    } else {
      EXPECT_EQ(grouped, plan.applied.end());
      EXPECT_EQ(vecq, plan.applied.end());
      EXPECT_FALSE(plan.missing_numerical.empty());
      EXPECT_EQ(std::ranges::count(plan.graph.nodes(), ir::Fused(FusedKind::kGroupedMatMul),
                                   &ir::Node::key),
                0);
      EXPECT_EQ(std::ranges::count(plan.graph.nodes(), ir::Primitive(Op::kPrepare), &ir::Node::key),
                0);
      EXPECT_FALSE(
          SelectionView(registry, kDevice, policy, false, false)
              .Qualification(NumericalChange::kQ8VectorReduction, graph, graph.node(out_a_site))
              .has_value());
      EXPECT_FALSE(
          SelectionView(registry, kDevice, policy, false, false)
              .Qualification(NumericalChange::kQ8VectorReduction, graph, graph.node(out_b_site))
              .has_value());
    }
  }
}

TEST(PlannerPatternTest, QsaGateQualifiesItsDownstreamProductRatherThanItsMulAnchor) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.product", Any(ir::Primitive(Op::kMatMul))));
  implementations.push_back(Fake("t.cont", Any(ir::Primitive(Op::kCont))));
  implementations.push_back(Fake("t.unary", Any(ir::Primitive(Op::kUnary))));
  implementations.push_back(Fake("t.gate", Any(ir::Fused(FusedKind::kHeadGate))));
  const auto registry = Registry(std::move(implementations));
  ir::Builder b;
  b.SetComponent(1);
  const auto x = b.Input(F32({128, 2}));
  const auto full = b.Input(F32({256, 2}));
  const auto view = b.View(full, ir::ShapeOf({64, 2, 2}), {4, 512, 1024, 2048}, 256);
  const auto gate = b.Cont(view, ir::ShapeOf({128, 2}));
  const auto gated = b.Mul(x, b.Unary(gate, ir::UnaryFn::kSigmoid));
  const auto w = b.Weight(
      {.encoding = Encoding::kMxfp8, .layout = Layout::kScaleAtoms, .shape = ir::ShapeOf({128, 8})},
      W(3));
  const auto product = b.MatMul(w, gated);
  b.Output(product);
  const auto graph = Finish(b);
  const std::array<std::uint16_t, 1> role{3};
  for (unsigned authority = 0; authority < 3; ++authority) {
    Policy policy;
    if (authority == 1)
      policy = RoleNumericalPolicy(graph, policy, NumericalChange::kMxfp8ProductInput, role);
    if (authority == 2) policy.numerical_candidates = {NumericalChange::kMxfp8ProductInput};
    const auto plan = PlanOf(graph.Clone(), registry, policy);
    const auto found = std::ranges::find(plan.applied, "prepared_head_gate", &Applied::pattern);
    EXPECT_EQ(found != plan.applied.end(), authority != 0);
    if (found != plan.applied.end()) {
      ASSERT_EQ(found->numerical_proofs.size(), 1U);
      EXPECT_EQ(found->numerical_proofs[0].site, graph.producer(product));
      EXPECT_EQ(found->numerical_proofs[0].candidate, authority == 2);
    }
  }
}

TEST(PlannerPatternTest, DefaultPreparationRequiresIncumbentFactsAndConstrainsFinalCompetitors) {
  constexpr std::array bf16{Encoding::kBF16};
  const std::array raw{ex::OperandRule{.operand = 0, .encodings = bf16},
                       ex::OperandRule{.operand = 1, .encodings = kF32}};
  const std::array rounded{ex::OperandRule{.operand = 0, .encodings = bf16},
                           ex::OperandRule{.operand = 1, .encodings = bf16}};
  for (const bool facts : {false, true}) {
    auto implementations = Everything();
    auto incumbent = ex::Capability{.op = ir::Primitive(Op::kMatMul), .operands = raw};
    if (facts)
      incumbent.product_arithmetic =
          +[](const ex::CapabilityQuery&) { return ir::ProductArithmetic::kBf16F32; };
    implementations.push_back(Fake("t.raw", incumbent));
    implementations.push_back(
        Fake("t.rounded", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                         .operands = rounded,
                                         .product_arithmetic = +[](const ex::CapabilityQuery&) {
                                           return ir::ProductArithmetic::kBf16F32;
                                         }}));
    // Correct for its typed operands, but it offers no incumbent-equivalence
    // contract. Priority cannot override the required numerical recipe.
    implementations.push_back(Fake(
        "t.unqualified",
        ex::Capability{.op = ir::Primitive(Op::kMatMul), .operands = rounded, .priority = 100}));
    implementations.push_back(Fake("t.convert", Any(ir::Primitive(Op::kConvert))));
    const auto registry = Registry(std::move(implementations));
    ir::Builder b;
    const auto x = b.Input(F32({64, 17}));
    for (std::uint16_t role = 0; role < 2; ++role) {
      const auto w = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({64, 8})), W(role));
      b.Output(b.Scale(b.MatMul(w, x), 2));
    }
    const auto plan = PlanOf(Finish(b), registry);
    EXPECT_EQ(
        std::ranges::any_of(plan.applied,
                            [](const auto& applied) { return applied.pattern == "bf16_inputs"; }),
        facts);
    std::size_t products = 0;
    for (const auto& step : plan.steps) {
      for (const auto id : step.nodes) {
        const auto& node = plan.graph.node(id);
        if (node.key.op != Op::kMatMul) continue;
        ++products;
        EXPECT_EQ(registry.at(step.implementation).name, facts ? "t.rounded" : "t.raw");
        EXPECT_EQ(node.Params<ir::ProductParams>().arithmetic,
                  facts ? ir::ProductArithmetic::kBf16F32 : ir::ProductArithmetic::kImplementation);
      }
    }
    EXPECT_EQ(products, 2U);
  }
}

TEST(PlannerPatternTest, QualifiedVectorRecipientsReachTheirOwnArithmeticSelection) {
  // Exercise both sibling orders: an equivalent sharing match must neither
  // capture a qualified sibling nor use its authority for an unqualified one.
  for (unsigned control = 0; control < 9; ++control) {
    SCOPED_TRACE(control);
    auto implementations = Everything();
    if (control != 8) {
      auto changed = ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                    .operands = kPreparedRules,
                                    .shape = {.max_rows = 8},
                                    .priority = 100};
      implementations.push_back(Fake("t.changed_vector", changed));
    }
    const auto registry = Registry(std::move(implementations));
    ir::Builder builder;
    builder.SetComponent(1);
    const auto x = builder.Input(F32({128, 2}));
    for (std::uint16_t role = 0; role < 2; ++role) {
      auto ref = W(role);
      if (control == 4 && role == 0) ref.stage = 1;
      const auto w = builder.Weight(Q8(128, 8 + 8ULL * role), ref);
      ir::ProductParams params;
      if (control == 7 && role == 0) params.arithmetic = ir::ProductArithmetic::kGgmlQ8_1F32;
      builder.Output(builder.Scale(builder.MatMul(w, x, params), 2));
    }
    const auto graph = Finish(builder);
    const std::array<std::uint16_t, 1> first{0}, second{1};
    const std::array<std::uint16_t, 2> both{0, 1};
    Policy policy;
    if (control != 0 && control != 6)
      policy = RoleNumericalPolicy(
          graph, policy, NumericalChange::kQ8VectorReduction,
          control == 2 ? std::span<const std::uint16_t>(second)
                       : (control == 3 || control == 7 ? std::span<const std::uint16_t>(both)
                                                       : std::span<const std::uint16_t>(first)));
    if (control == 5) {
      // Same source/role/geometry, but this evidence covers one actual row.
      auto entries = std::vector<NumericalQualification>(policy.numerical->entries().begin(),
                                                         policy.numerical->entries().end());
      for (auto& entry : entries) entry.min_rows = entry.max_rows = 1;
      auto bound =
          NumericalQualifications::Bind(entries, entries.front().source, entries.front().semantic);
      ASSERT_TRUE(bound) << (bound ? "" : bound.error());
      policy.numerical = std::move(*bound);
    }
    if (control == 6) policy.numerical_candidates = {NumericalChange::kQ8VectorReduction};
    const auto plan = PlanOf(graph.Clone(), registry, policy);
    std::array<bool, 2> seen{};
    for (const auto& step : plan.steps) {
      for (const auto id : step.nodes) {
        const auto& node = plan.graph.node(id);
        if (node.key != ir::Primitive(Op::kMatMul)) continue;
        const auto& ref = std::get<ir::WeightRef>(plan.graph.tensor(node.src[0]).binding);
        ASSERT_LT(ref.role, seen.size());
        EXPECT_FALSE(seen[ref.role]);
        seen[ref.role] = true;
        const bool changed = control == 3 || control == 6 || (control == 1 && ref.role == 0) ||
                             ((control == 2 || control == 7) && ref.role == 1);
        const bool explicit_raw = control == 7 && ref.role == 0;
        EXPECT_EQ(registry.at(step.implementation).name,
                  changed ? "t.changed_vector"
                          : (explicit_raw ? "t.mul_mat.vector" : "t.mul_mat.prepared"));
        EXPECT_EQ(node.Params<ir::ProductParams>().arithmetic,
                  changed || (control == 8 && ref.role == 0)
                      ? ir::ProductArithmetic::kImplementation
                      : ir::ProductArithmetic::kGgmlQ8_1F32);
        EXPECT_EQ(plan.graph.tensor(node.src[1]).format.encoding,
                  explicit_raw ? Encoding::kF32 : Encoding::kQ8_1);
      }
    }
    EXPECT_TRUE(std::ranges::all_of(seen, [](bool found) { return found; }));
    std::array<unsigned, 2> proofs{};
    for (const auto& applied : plan.applied) {
      for (const auto& proof : applied.numerical_proofs) {
        ASSERT_TRUE(proof.observed.weight.has_value());
        const auto role = proof.observed.weight->role;
        ASSERT_LT(role, proofs.size());
        ++proofs[role];
        EXPECT_EQ(proof.actual.rows, 2U);
        EXPECT_EQ(proof.observed.weight->stage, 0U);
        EXPECT_EQ(proof.candidate, control == 6);
        if (!proof.candidate) EXPECT_EQ(proof.qualification.weight_role, role);
      }
    }
    EXPECT_EQ(proofs[0], control == 1 || control == 3 || control == 6 || control == 8 ? 1U : 0U);
    EXPECT_EQ(proofs[1], control == 2 || control == 3 || control == 6 || control == 7 ? 1U : 0U);
  }
}

TEST(PlannerPatternTest, SharedInputPreparationStopsAtAnInPlaceWrite) {
  for (bool bf16 : {false, true}) {
    SCOPED_TRACE(bf16);
    auto implementations = Everything();
    auto product = Any(ir::Primitive(Op::kMatMul));
    if (bf16)
      product.product_arithmetic =
          +[](const ex::CapabilityQuery&) { return ir::ProductArithmetic::kBf16F32; };
    implementations.push_back(Fake("t.any_product", product));
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

TEST(PlannerPatternTest, EveryRewriteLookupRequiresQualificationWhileDirectNodesRemainAvailable) {
  auto implementations = Everything();
  for (auto& implementation : implementations)
    if (implementation.name == "t.rms_norm_mul")
      implementation.capability->fusion_needs_measurement = true;
  const auto registry = Registry(std::move(implementations));
  NormGraph fixture;
  fixture.b.Output(fixture.m);
  auto graph = Finish(fixture.b);
  EXPECT_EQ(Names(PlanOf(graph.Clone(), registry), registry),
            (std::vector<std::string>{"t.rms_norm", "t.mul"}));
  Policy measured;
  measured.ranks = {{.device_class = "test", .implementation = "t.rms_norm_mul", .rank = 0}};
  EXPECT_EQ(Names(PlanOf(graph.Clone(), registry, measured), registry),
            std::vector<std::string>{"t.rms_norm_mul"});
  measured.ranks[0].device_class = "different_device";
  EXPECT_TRUE(Patterns(PlanOf(graph.Clone(), registry, measured)).empty());
  measured.ranks[0].device_class = "test";
  measured.ranks[0].implementation = "different_implementation";
  EXPECT_TRUE(Patterns(PlanOf(graph.Clone(), registry, measured)).empty());
  ir::Builder direct;
  const auto x = direct.Input(F32({64, 3}));
  const auto w = direct.Weight(F32({64}), W(0));
  direct.Output(
      direct.Fused(FusedKind::kRmsNormMul, std::array{x, w}, ir::NormParams{.eps = 1e-6f})[0]);
  EXPECT_EQ(Names(PlanOf(Finish(direct), registry), registry),
            std::vector<std::string>{"t.rms_norm_mul"});
}

TEST(PlannerPatternTest, FinalRewriteSelectionCannotBorrowAnotherImplementationsQualification) {
  auto implementations = Everything();
  std::erase_if(implementations, [](const auto& implementation) {
    return implementation.name == "t.rms_norm_mul_add";
  });
  auto competitor = Any(ir::Fused(FusedKind::kRmsNormMul));
  competitor.priority = 100;
  competitor.fusion_needs_measurement = true;
  implementations.push_back(Fake("t.unmeasured_norm", competitor));
  const auto registry = Registry(std::move(implementations));
  Policy policy;
  policy.enabled = {"bf16_branch"};  // Exercise both rewrite rounds, even without a branch match.
  for (const bool fallback : {false, true}) {
    SCOPED_TRACE(fallback);
    ir::Builder b;
    const auto x = b.Input(F32({64, 3}));
    const auto w = b.Weight(F32({64}), W(0));
    (void)b.Scale(x, 7.0F);  // Pruning renumbers the rewrite's nodes.
    b.Output(b.Mul(b.RmsNorm(x, 1e-6F), w));
    if (fallback) {
      const auto residual = b.Input(F32({64, 3}));
      b.Output(b.Fused(ir::FusedKind::kRmsNormMulAdd, std::array{x, w, residual},
                       ir::NormParams{.eps = 1e-6F})[0]);
    }
    const auto graph = Finish(b);
    const auto ordinary = PlanOf(graph.Clone(), registry, policy);
    const auto ordinary_names = Names(ordinary, registry);
    EXPECT_EQ(std::ranges::count(ordinary_names, "t.rms_norm_mul"), 1);
    EXPECT_FALSE(std::ranges::contains(ordinary_names, "t.unmeasured_norm"));
    EXPECT_EQ(ordinary.decomposed.empty(), !fallback);
    policy.ranks = {{.device_class = "test", .implementation = "t.unmeasured_norm", .rank = 0}};
    EXPECT_EQ(std::ranges::count(Names(PlanOf(graph.Clone(), registry, policy), registry),
                                 "t.unmeasured_norm"),
              1);
    policy.ranks.clear();
  }
  ir::Builder direct;
  const auto x = direct.Input(F32({64, 3}));
  const auto w = direct.Weight(F32({64}), W(0));
  direct.Output(
      direct.Fused(FusedKind::kRmsNormMul, std::array{x, w}, ir::NormParams{.eps = 1e-6F})[0]);
  EXPECT_EQ(Names(PlanOf(Finish(direct), registry), registry),
            std::vector<std::string>{"t.unmeasured_norm"});
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
    const auto plan = PlanOf(Finish(b), r);
    EXPECT_EQ(std::ranges::count(plan.applied, "prepared_inputs", &Applied::pattern),
              store ? 0 : 1);
    if (store) {
      std::array<std::uint32_t, 2> preparations{};
      std::size_t count = 0;
      std::uint32_t store_at = UINT32_MAX;
      for (std::uint32_t i = 0; i < plan.graph.nodes().size(); ++i) {
        const auto key = plan.graph.node({i}).key;
        if (key == ir::Primitive(Op::kPrepare)) {
          ASSERT_LT(count, preparations.size());
          preparations[count++] = i;
        }
        if (key == ir::Primitive(Op::kCacheStore)) store_at = i;
      }
      ASSERT_EQ(count, 2U);
      EXPECT_LT(preparations[0], store_at);
      EXPECT_LT(store_at, preparations[1]);
    }
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

// The spatial channel norm as components/image_blocks.cc emits it: the
// pixels' channel rows, L2-normed, scaled, times gamma, SiLU or not, and
// back. Packed [W, H, C] pixels (rank 3: an image extent of one drops).
TEST(PlannerPatternTest, SpatialChannelNormTakesTheWholeLayoutRoundTrip) {
  std::vector<ex::Implementation> everything = Everything();
  for (const Op op : {Op::kL2Norm, Op::kCont, Op::kUnary})
    everything.push_back(
        Fake(std::string("t.") + std::string(ir::OpName(op)), Any(ir::Primitive(op))));
  const ex::Registry r = Registry(everything);
  everything.push_back(
      Fake("t.spatial_channel_norm", Any(ir::Fused(FusedKind::kSpatialChannelNorm))));
  const ex::Registry fused = Registry(std::move(everything));
  struct Variant {
    bool silu = false;
    bool rows_read = false;  // the channel rows also a step output
    float bias = 0.0F;       // a scale with a bias is not the norm's
    std::array<std::uint8_t, 4> back{2, 0, 1, 3};
  };
  const auto graph = [](const Variant& v) {
    ir::Builder b;
    const TensorId x = b.Scale(b.Input(F32({4, 3, 8})), 2.0F);
    const TensorId gamma = b.Weight(F32({8}), W(0));
    const TensorId rows = b.Cont(b.Permute(x, {1, 2, 0, 3}));
    TensorId out = b.Mul(b.Scale(b.L2Norm(rows, 1e-6F), std::sqrt(8.0F), v.bias), gamma);
    if (v.silu) out = b.Unary(out, ir::UnaryFn::kSilu);
    b.Output(b.Cont(b.Permute(out, v.back)));
    if (v.rows_read) b.Output(rows);
    return Finish(b);
  };
  for (const bool silu : {false, true}) {
    const StepPlan p = PlanOf(graph({.silu = silu}), fused);
    EXPECT_EQ(Patterns(p), std::vector<std::string_view>{"spatial_channel_norm"}) << silu;
    ASSERT_EQ(p.applied.size(), 1U);
    EXPECT_EQ(p.applied[0].replaced.size(), silu ? 8U : 7U);
    EXPECT_EQ(Keys(p.graph),
              (std::vector{ir::Primitive(Op::kScale), ir::Fused(FusedKind::kSpatialChannelNorm)}));
    const ir::Node& norm = p.graph.node(p.graph.producer(p.graph.outputs()[0]));
    const auto params = norm.Params<ir::SpatialNormParams>();
    EXPECT_EQ(params.eps, 1e-6F);
    EXPECT_EQ(params.scale, std::sqrt(8.0F));
    EXPECT_EQ(params.silu, silu);
    EXPECT_EQ(p.graph.tensor(p.graph.outputs()[0]).shape(), ir::ShapeOf({4, 3, 8}));
    // Without an implementation, or disabled: the primitives stay.
    EXPECT_TRUE(Patterns(PlanOf(graph({.silu = silu}), r)).empty());
    EXPECT_TRUE(Patterns(PlanOf(graph({.silu = silu}), fused, Disabling({"spatial_channel_norm"})))
                    .empty());
  }
  EXPECT_TRUE(Patterns(PlanOf(graph({.rows_read = true}), fused)).empty());
  EXPECT_TRUE(Patterns(PlanOf(graph({.bias = 1.0F}), fused)).empty());
  EXPECT_TRUE(Patterns(PlanOf(graph({.back = {1, 2, 0, 3}}), fused)).empty());
}

// The multi-axis rotation as components/conditioning_blocks.cc RopeEmit
// emits it: heads normed, times a BF16 weight widened to F32, then three
// sections rotated by packed copies of their positions' views.
TEST(PlannerPatternTest, RotatedSectionsTakeTheWeightWideningAndPositionCopies) {
  std::vector<ex::Implementation> everything = Everything();
  for (const Op op : {Op::kCont, Op::kConvert})
    everything.push_back(
        Fake(std::string("t.") + std::string(ir::OpName(op)), Any(ir::Primitive(op))));
  const ex::Registry r = Registry(everything);
  everything.push_back(
      Fake("t.rms_norm_mul_rope_sections", Any(ir::Fused(FusedKind::kRmsNormMulRopeSections))));
  const ex::Registry fused = Registry(std::move(everything));
  Policy on{.enabled = {"rms_norm_mul_rope_sections"}};
  Canonicalize(on);
  struct Variant {
    std::array<std::uint32_t, 3> dims{4, 2, 2};
    float last_base = 50.0F;  // the third section's
    bool copy_read = false;   // the second positions' copy also an output
  };
  const auto graph = [](const Variant& v) {
    ir::Builder b;
    const TensorId x = b.Scale(b.Input(F32({8, 2, 3})), 2.0F);
    const TensorId aux = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({3, 3})));
    const TensorId w =
        b.Convert(b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({8})), W(0)), Encoding::kF32);
    TensorId out = b.Mul(b.RmsNorm(x, 1e-6F), w);
    std::uint32_t offset = 0;
    for (std::uint64_t i = 0; i < 3; ++i) {
      const TensorId pos = b.Cont(b.View(aux, ir::ShapeOf({3}), {4, 12, 12, 12}, i * 12));
      if (v.copy_read && i == 1) b.Output(pos);
      out = b.Rope(out, pos, {},
                   {.mode = ir::RopeMode::kNeox,
                    .dims = v.dims.at(i),
                    .offset = offset,
                    .base = i == 2 ? v.last_base : 50.0F});
      offset += v.dims.at(i);
    }
    b.Output(out);
    return Finish(b);
  };
  {
    const StepPlan p = PlanOf(graph({}), fused, on);
    EXPECT_EQ(Patterns(p), std::vector<std::string_view>{"rms_norm_mul_rope_sections"});
    ASSERT_EQ(p.applied.size(), 1U);
    EXPECT_EQ(p.applied[0].replaced.size(), 9U);  // norm, mul, widening, 3 copies, 3 rotations
    EXPECT_EQ(Keys(p.graph), (std::vector{ir::Primitive(Op::kScale),
                                          ir::Fused(FusedKind::kRmsNormMulRopeSections)}));
    const ir::Node& n = p.graph.node(p.graph.producer(p.graph.outputs()[0]));
    const auto params = n.Params<ir::RopeSectionsParams>();
    EXPECT_EQ(params.sections, 3U);
    EXPECT_EQ(params.dims, (std::array<std::uint32_t, 3>{4, 2, 2}));
    EXPECT_EQ(params.mode, ir::RopeMode::kNeox);
    EXPECT_EQ(params.base, 50.0F);
    EXPECT_EQ(p.graph.tensor(n.src[1]).format.encoding, Encoding::kBF16);
    for (std::size_t i = 2; i < 5; ++i) EXPECT_TRUE(p.graph.tensor(n.src[i]).is_view()) << i;
  }
  {
    // A copy read elsewhere stays; the match reads it.
    const StepPlan p = PlanOf(graph({.copy_read = true}), fused, on);
    EXPECT_EQ(Patterns(p), std::vector<std::string_view>{"rms_norm_mul_rope_sections"});
    ASSERT_EQ(p.applied.size(), 1U);
    EXPECT_EQ(p.applied[0].replaced.size(), 8U);
  }
  // Explicitly disabled, without an implementation, with sections
  // that leave part of the head, or of different bases: no sections (the
  // norm and weight still fuse).
  const std::vector<std::string_view> norm_only{"rms_norm_mul"};
  EXPECT_EQ(Patterns(PlanOf(graph({}), fused, Disabling({"rms_norm_mul_rope_sections"}))),
            norm_only);
  EXPECT_EQ(Patterns(PlanOf(graph({}), r, on)), norm_only);
  EXPECT_EQ(Patterns(PlanOf(graph({.dims = {2, 2, 2}}), fused, on)), norm_only);
  EXPECT_EQ(Patterns(PlanOf(graph({.last_base = 60.0F}), fused, on)), norm_only);
}

// Each exact matcher can be disabled independently. Disable both matching
// norm/rotation patterns when the control requires primitive rotation.
TEST(PlannerPatternTest, DisablingNormRopeIsNotUndoneByTheSectionedForm) {
  std::vector<ex::Implementation> everything = Everything();
  everything.push_back(
      Fake("t.rms_norm_mul_rope_sections", Any(ir::Fused(FusedKind::kRmsNormMulRopeSections))));
  const ex::Registry r = Registry(std::move(everything));
  const auto graph = [] {
    ir::Builder b;
    const TensorId x = b.Input(F32({64, 2, 3}));
    const TensorId pos = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({3})));
    const ir::RopeParams rope{.mode = ir::RopeMode::kNeox, .dims = 64, .base = 10000.0f};
    b.Output(b.Rope(b.Mul(b.RmsNorm(x, 1e-6f), b.Weight(F32({64}), W(0))), pos, {}, rope));
    return Finish(b);
  };
  // Both fused fakes take this graph.
  EXPECT_EQ(Patterns(PlanOf(graph(), r, Disabling({"interleaved_norm_rope"}))),
            std::vector<std::string_view>{"rms_norm_mul_rope"});
  const StepPlan off = PlanOf(
      graph(), r,
      Disabling({"interleaved_norm_rope", "rms_norm_mul_rope", "rms_norm_mul_rope_sections"}));
  EXPECT_EQ(Patterns(off), std::vector<std::string_view>{"rms_norm_mul"});
  EXPECT_EQ(Keys(off.graph),
            (std::vector{ir::Fused(FusedKind::kRmsNormMul), ir::Primitive(Op::kRope)}));
  EXPECT_EQ(Names(off, r), (std::vector<std::string>{"t.rms_norm_mul", "t.rope"}));
  // Leaving the sectioned matcher available selects that exact alternative.
  Policy image{.enabled = {"rms_norm_mul_rope_sections"},
               .disabled = {"interleaved_norm_rope", "rms_norm_mul_rope"}};
  Canonicalize(image);
  EXPECT_EQ(Patterns(PlanOf(graph(), r, image)),
            std::vector<std::string_view>{"rms_norm_mul_rope_sections"});
}

// bf16_glu: the gate and up products of BF16 weights, their SwiGLU and its
// readers' products in BF16, the products' input converted once.
TEST(PlannerPatternTest, Bf16GluRecordsTheMlpInBf16) {
  static constexpr std::array kBf16 = {Encoding::kBF16};
  static constexpr std::array kBf16Products = {ex::OperandRule{.operand = 0, .encodings = kBf16},
                                               ex::OperandRule{.operand = 1, .encodings = kBf16}};
  static constexpr std::array kFloatProducts = {ex::OperandRule{.operand = 0, .encodings = kBf16},
                                                ex::OperandRule{.operand = 1, .encodings = kF32}};
  static constexpr std::array kF32Products = {ex::OperandRule{.operand = 0, .encodings = kF32},
                                              ex::OperandRule{.operand = 1, .encodings = kF32}};
  std::vector<ex::Implementation> everything = Everything();
  everything.push_back(Fake("t.convert", Any(ir::Primitive(Op::kConvert))));
  everything.push_back(Fake("t.mul_mat.float", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                                              .operands = kFloatProducts}));
  everything.push_back(Fake(
      "t.mul_mat.f32", ex::Capability{.op = ir::Primitive(Op::kMatMul), .operands = kF32Products}));
  const ex::Registry plain = Registry(everything);
  everything.push_back(Fake("t.mul_mat.bf16", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                                             .operands = kBf16Products}));
  const ex::Registry r = Registry(std::move(everything));
  Policy on{.enabled = {"bf16_glu"}};
  Canonicalize(on);
  struct Variant {
    bool gate_read = false;   // the gate product also an output
    bool glu_scaled = false;  // the GLU also read by a scale
    bool f32_down = false;    // the down product's weight F32
  };
  const auto graph = [](const Variant& v) {
    ir::Builder b;
    const TensorId x = b.Scale(b.Input(F32({64, 32})), 2.0F);
    const auto bf16 = [&](std::uint64_t k, std::uint64_t n, std::uint16_t role) {
      return b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({k, n})), W(role));
    };
    const TensorId gate = b.MatMul(bf16(64, 128, 0), x);
    const TensorId up = b.MatMul(bf16(64, 128, 1), x);
    const TensorId glu = b.Glu(gate, up, {.kind = ir::GluKind::kSwiGlu});
    const TensorId wd = v.f32_down ? b.Weight(F32({128, 64}), W(2)) : bf16(128, 64, 2);
    b.Output(b.MatMul(wd, glu));
    if (v.gate_read) b.Output(gate);
    if (v.glu_scaled) b.Output(b.Scale(glu, 2.0F));
    return Finish(b);
  };
  const StepPlan p = NumericalPlanOf(graph({}), r, on, {NumericalChange::kBf16SwiGluIntermediates});
  EXPECT_EQ(Patterns(p), std::vector<std::string_view>{"bf16_glu"});
  std::size_t converts = 0, bf16_products = 0;
  for (const ir::Node& n : p.graph.nodes()) {
    converts += n.key == ir::Primitive(Op::kConvert);
    if (n.key == ir::Primitive(Op::kMatMul)) {
      EXPECT_EQ(p.graph.tensor(n.src[1]).format.encoding, Encoding::kBF16);
      bf16_products += p.graph.tensor(n.out[0]).format.encoding == Encoding::kBF16;
    }
    if (n.key == ir::Primitive(Op::kGlu))
      EXPECT_EQ(p.graph.tensor(n.out[0]).format.encoding, Encoding::kBF16);
  }
  EXPECT_EQ(converts, 1U);       // the products' shared input, once
  EXPECT_EQ(bf16_products, 2U);  // the gate and the up; the down writes F32
  EXPECT_EQ(p.graph.tensor(p.graph.outputs()[0]).format.encoding, Encoding::kF32);
  // Without numerical authority, without BF16 products, a gate read elsewhere, or a GLU read by
  // other than a product; an F32 down product: nothing.
  const auto preserved = PlanOf(graph({}), r);
  EXPECT_TRUE(Patterns(preserved).empty());
  EXPECT_FALSE(preserved.missing_numerical.empty());
  Policy candidate;
  candidate.numerical_candidates = {NumericalChange::kBf16SwiGluIntermediates};
  const auto proposed = PlanOf(graph({}), r, candidate);
  ASSERT_EQ(proposed.applied.size(), 1U);
  EXPECT_EQ(proposed.applied[0].pattern, "bf16_glu");
  EXPECT_TRUE(proposed.applied[0].numerical_candidate);
  EXPECT_FALSE(proposed.applied[0].numerical_incumbent);
  EXPECT_EQ(proposed.applied[0].numerical_evidence, base::Sha256Digest{});
  EXPECT_FALSE(proposed.missing_numerical.empty());
  // The diagnostic cannot relax observer closure or implementation support.
  EXPECT_TRUE(Patterns(PlanOf(graph({.gate_read = true}), r, candidate)).empty());
  EXPECT_TRUE(Patterns(PlanOf(graph({}), plain, candidate)).empty());
  EXPECT_TRUE(
      Patterns(NumericalPlanOf(graph({}), plain, on, {NumericalChange::kBf16SwiGluIntermediates}))
          .empty());
  EXPECT_TRUE(Patterns(NumericalPlanOf(graph({.gate_read = true}), r, on,
                                       {NumericalChange::kBf16SwiGluIntermediates}))
                  .empty());
  EXPECT_TRUE(Patterns(NumericalPlanOf(graph({.glu_scaled = true}), r, on,
                                       {NumericalChange::kBf16SwiGluIntermediates}))
                  .empty());
  EXPECT_TRUE(Patterns(NumericalPlanOf(graph({.f32_down = true}), r, on,
                                       {NumericalChange::kBf16SwiGluIntermediates}))
                  .empty());
}

TEST(PlannerPatternTest, SplitModulationAndGatedResidualFuseTheirConditioning) {
  static constexpr std::array kBf16 = {Encoding::kBF16};
  static constexpr std::array kBf16Products = {ex::OperandRule{.operand = 0, .encodings = kBf16},
                                               ex::OperandRule{.operand = 1, .encodings = kBf16}};
  static constexpr std::array kFloatProducts = {ex::OperandRule{.operand = 0, .encodings = kBf16},
                                                ex::OperandRule{.operand = 1, .encodings = kF32}};
  const auto prim = [](Op op) { return Any(ir::Primitive(op)); };
  std::vector<ex::Implementation> everything = Everything();
  for (const Op op : {Op::kLayerNorm, Op::kRepeat, Op::kConcat, Op::kUnary, Op::kConvert})
    everything.push_back(Fake(std::format("t.{}", ir::OpKeyName(ir::Primitive(op))), prim(op)));
  everything.push_back(
      Fake("t.mul_mat.float", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                             .operands = kFloatProducts,
                                             .product_arithmetic = +[](const ex::CapabilityQuery&) {
                                               return ir::ProductArithmetic::kBf16F32;
                                             }}));
  everything.push_back(
      Fake("t.modulate_split", Any(ir::Fused(FusedKind::kLayerNormModulateSplit))));
  everything.push_back(Fake("t.gated_split", Any(ir::Fused(FusedKind::kGatedResidualSplit))));
  const ex::Registry f32_inputs = Registry(everything);
  everything.push_back(
      Fake("t.mul_mat.bf16", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                            .operands = kBf16Products,
                                            .product_arithmetic = +[](const ex::CapabilityQuery&) {
                                              return ir::ProductArithmetic::kBf16F32;
                                            }}));
  const ex::Registry r = Registry(std::move(everything));
  Policy on{.enabled = {"split_gated_residual", "split_modulate"}};
  Canonicalize(on);
  struct Variant {
    bool one_side = false;     // one conditioning vector over every row
    bool norm_scaled = false;  // the modulation also read by a scale
    float bias = 1.0F;         // the conditioning's added one
  };
  const auto graph = [](const Variant& v) {
    ir::Builder b;
    const TensorId x = b.Scale(b.Input(F32({64, 32})), 2.0F);
    const auto side = [&] {
      const TensorId a = b.Input(F32({64}));
      if (v.one_side) return b.Repeat(a, ir::ShapeOf({64, 32}));
      const TensorId c = b.Input(F32({64}));
      return b.Concat(b.Repeat(a, ir::ShapeOf({64, 12})), b.Repeat(c, ir::ShapeOf({64, 20})), 1);
    };
    const TensorId m = b.Mul(b.LayerNorm(x, 1e-6F), b.Scale(side(), 1.0F, v.bias));
    const TensorId w = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({64, 64})), W(0));
    const TensorId y = b.MatMul(w, m);
    b.Output(b.Add(x, b.Mul(y, b.Unary(side(), ir::UnaryFn::kTanh))));
    if (v.norm_scaled) b.Output(b.Scale(m, 2.0F));
    return Finish(b);
  };
  const auto split_of = [](const StepPlan& p, FusedKind kind) -> std::optional<std::uint32_t> {
    for (const ir::Node& n : p.graph.nodes())
      if (n.key == ir::Fused(kind)) return n.Params<ir::SplitModulateParams>().split;
    return std::nullopt;
  };
  const auto modulation = [](const StepPlan& p) {
    for (const ir::Node& n : p.graph.nodes())
      if (n.key == ir::Fused(FusedKind::kLayerNormModulateSplit))
        return p.graph.tensor(n.out[0]).format.encoding;
    return Encoding::kF32;
  };
  const auto converts = [](const StepPlan& p) {
    return std::ranges::count(p.graph.nodes(), ir::Primitive(Op::kConvert), &ir::Node::key);
  };
  {
    const StepPlan p = PlanOf(graph({}), r, on);
    EXPECT_EQ(Patterns(p),
              (std::vector<std::string_view>{"split_modulate", "split_gated_residual"}));
    EXPECT_EQ(split_of(p, FusedKind::kLayerNormModulateSplit), 12U);
    EXPECT_EQ(split_of(p, FusedKind::kGatedResidualSplit), 12U);
    // Read only by a product of a BF16 weight that takes BF16 input: made
    // in BF16, no conversion.
    EXPECT_EQ(modulation(p), Encoding::kBF16);
    EXPECT_EQ(converts(p), 0);
  }
  {
    const StepPlan p = PlanOf(graph({.one_side = true}), r, on);
    EXPECT_EQ(split_of(p, FusedKind::kLayerNormModulateSplit), 32U);
    EXPECT_EQ(split_of(p, FusedKind::kGatedResidualSplit), 32U);
  }
  // F32 where its product takes only F32 input, or another reads it.
  EXPECT_EQ(modulation(PlanOf(graph({}), f32_inputs, on)), Encoding::kF32);
  {
    const StepPlan p = PlanOf(graph({.norm_scaled = true}), r, on);
    EXPECT_EQ(split_of(p, FusedKind::kLayerNormModulateSplit), 12U);
    EXPECT_EQ(modulation(p), Encoding::kF32);
  }
  // An explicit control disables both exact matchers. A conditioning
  // without its one still refuses the modulation.
  EXPECT_EQ(Patterns(PlanOf(graph({}), r, Disabling({"split_gated_residual", "split_modulate"}))),
            std::vector<std::string_view>{"bf16_inputs"});
  EXPECT_EQ(Patterns(PlanOf(graph({.bias = 2.0F}), r, on)),
            (std::vector<std::string_view>{"split_gated_residual", "bf16_inputs"}));
}

TEST(PlannerPatternTest, SplitResidualJoinsTheModulationOfItsSum) {
  const auto prim = [](Op op) { return Any(ir::Primitive(op)); };
  std::vector<ex::Implementation> everything = Everything();
  for (const Op op : {Op::kLayerNorm, Op::kRepeat, Op::kConcat, Op::kUnary, Op::kConvert})
    everything.push_back(Fake(std::format("t.{}", ir::OpKeyName(ir::Primitive(op))), prim(op)));
  everything.push_back(
      Fake("t.modulate_split", Any(ir::Fused(FusedKind::kLayerNormModulateSplit))));
  everything.push_back(Fake("t.gated_split", Any(ir::Fused(FusedKind::kGatedResidualSplit))));
  const ex::Registry apart = Registry(everything);
  everything.push_back(
      Fake("t.gated_modulate_split", Any(ir::Fused(FusedKind::kGatedResidualModulateSplit))));
  const ex::Registry r = Registry(std::move(everything));
  Policy on{.enabled = {"split_gated_residual", "split_gated_residual_modulate", "split_modulate"}};
  Canonicalize(on);
  const auto graph = [](bool read_between) {
    ir::Builder b;
    const TensorId x = b.Scale(b.Input(F32({64, 32})), 2.0F);
    const TensorId y = b.Scale(b.Input(F32({64, 32})), 3.0F);
    const auto side = [&] {
      return b.Concat(b.Repeat(b.Input(F32({64})), ir::ShapeOf({64, 12})),
                      b.Repeat(b.Input(F32({64})), ir::ShapeOf({64, 20})), 1);
    };
    const TensorId sum = b.Add(x, b.Mul(y, b.Unary(side(), ir::UnaryFn::kTanh)));
    // A reader of the sum before its modulation.
    if (read_between) b.Output(b.Scale(sum, 2.0F));
    b.Output(b.Mul(b.LayerNorm(sum, 1e-6F), b.Scale(side(), 1.0F, 1.0F)));
    b.Output(b.Scale(sum, 4.0F));
    return Finish(b);
  };
  const auto count = [](const StepPlan& p, FusedKind kind) {
    return std::ranges::count(p.graph.nodes(), ir::Fused(kind), &ir::Node::key);
  };
  const StepPlan p = PlanOf(graph(false), r, on);
  EXPECT_EQ(Patterns(p), (std::vector<std::string_view>{"split_gated_residual", "split_modulate",
                                                        "split_gated_residual_modulate"}));
  EXPECT_EQ(count(p, FusedKind::kGatedResidualModulateSplit), 1);
  EXPECT_EQ(count(p, FusedKind::kGatedResidualSplit), 0);
  EXPECT_EQ(count(p, FusedKind::kLayerNormModulateSplit), 0);
  // Explicitly disabled, not implemented, or read before modulation: apart.
  Policy without = on;
  std::erase(without.enabled, "split_gated_residual_modulate");
  without.disabled.push_back("split_gated_residual_modulate");
  Canonicalize(without);
  EXPECT_EQ(count(PlanOf(graph(false), r, without), FusedKind::kGatedResidualModulateSplit), 0);
  EXPECT_EQ(count(PlanOf(graph(false), apart, on), FusedKind::kGatedResidualModulateSplit), 0);
  const StepPlan between = PlanOf(graph(true), r, on);
  EXPECT_EQ(count(between, FusedKind::kGatedResidualModulateSplit), 0);
  EXPECT_EQ(count(between, FusedKind::kGatedResidualSplit), 1);
}

// The modulation of a range of the sum's rows (the image's output norm over
// its latent rows): joined, its readers reading the range of every row's.
TEST(PlannerPatternTest, SplitResidualJoinsTheModulationOfARangeOfItsRows) {
  const auto prim = [](Op op) { return Any(ir::Primitive(op)); };
  std::vector<ex::Implementation> everything = Everything();
  for (const Op op :
       {Op::kLayerNorm, Op::kRepeat, Op::kConcat, Op::kUnary, Op::kConvert, Op::kCont})
    everything.push_back(Fake(std::format("t.{}", ir::OpKeyName(ir::Primitive(op))), prim(op)));
  everything.push_back(
      Fake("t.modulate_split", Any(ir::Fused(FusedKind::kLayerNormModulateSplit))));
  everything.push_back(Fake("t.gated_split", Any(ir::Fused(FusedKind::kGatedResidualSplit))));
  everything.push_back(
      Fake("t.gated_modulate_split", Any(ir::Fused(FusedKind::kGatedResidualModulateSplit))));
  const ex::Registry r = Registry(std::move(everything));
  Policy on{.enabled = {"bf16_residual", "split_gated_residual", "split_gated_residual_modulate",
                        "split_modulate"}};
  Canonicalize(on);
  // Rows [first, 32) of the sum, its split at 12; `two`: the range's own
  // modulation split at `split` between two vectors.
  const auto graph = [](std::uint64_t first, bool copy, bool two, std::uint64_t split) {
    ir::Builder b;
    const TensorId x = b.Scale(b.Input(F32({64, 32})), 2.0F);
    const TensorId y = b.Scale(b.Input(F32({64, 32})), 3.0F);
    const TensorId side = b.Concat(b.Repeat(b.Input(F32({64})), ir::ShapeOf({64, 12})),
                                   b.Repeat(b.Input(F32({64})), ir::ShapeOf({64, 20})), 1);
    const TensorId sum = b.Add(x, b.Mul(y, b.Unary(side, ir::UnaryFn::kTanh)));
    const std::uint64_t rows = 32 - first;
    TensorId range =
        b.View(sum, ir::ShapeOf({64, rows}), {4, 256, 256 * rows, 256 * rows}, first * 256);
    if (copy) range = b.Cont(range);
    const TensorId scale =
        two ? b.Concat(b.Repeat(b.Input(F32({64})), ir::ShapeOf({64, split})),
                       b.Repeat(b.Input(F32({64})), ir::ShapeOf({64, rows - split})), 1)
            : b.Repeat(b.Input(F32({64})), ir::ShapeOf({64, rows}));
    b.Output(b.Scale(b.Mul(b.LayerNorm(range, 1e-6F), b.Scale(scale, 1.0F, 1.0F)), 2.0F));
    return Finish(b);
  };
  const auto count = [](const StepPlan& p, FusedKind kind) {
    return std::ranges::count(p.graph.nodes(), ir::Fused(kind), &ir::Node::key);
  };
  for (const bool copy : {false, true}) {
    const StepPlan p =
        NumericalPlanOf(graph(12, copy, false, 0), r, on, {NumericalChange::kBf16ResidualCarry});
    EXPECT_EQ(count(p, FusedKind::kGatedResidualModulateSplit), 1) << copy;
    EXPECT_EQ(count(p, FusedKind::kGatedResidualSplit), 0) << copy;
    EXPECT_EQ(count(p, FusedKind::kLayerNormModulateSplit), 0) << copy;
    EXPECT_EQ(std::ranges::count(p.graph.nodes(), ir::Primitive(Op::kCont), &ir::Node::key), 0);
    for (const ir::Node& n : p.graph.nodes())
      if (n.key == ir::Fused(FusedKind::kGatedResidualModulateSplit)) {
        // The sum nothing else reads, in BF16; every row modulated.
        EXPECT_EQ(p.graph.tensor(n.out[0]).format.encoding, Encoding::kBF16);
        EXPECT_EQ(p.graph.tensor(n.out[1]).ne(1), 32U);
        EXPECT_EQ(n.Params<ir::SplitModulateParams>().split, 12U);
      }
  }
  // A range's split at the residual's (counted from its first row): joined.
  EXPECT_EQ(
      count(NumericalPlanOf(graph(8, false, true, 4), r, on, {NumericalChange::kBf16ResidualCarry}),
            FusedKind::kGatedResidualModulateSplit),
      1);
  // Elsewhere: apart.
  const StepPlan apart =
      NumericalPlanOf(graph(8, false, true, 6), r, on, {NumericalChange::kBf16ResidualCarry});
  EXPECT_EQ(count(apart, FusedKind::kGatedResidualModulateSplit), 0);
  EXPECT_EQ(count(apart, FusedKind::kGatedResidualSplit), 1);
}

TEST(PlannerPatternTest, Bf16BranchRecordsTheResidualsBranchInBf16) {
  static constexpr std::array kBf16 = {Encoding::kBF16};
  static constexpr std::array kBf16Products = {ex::OperandRule{.operand = 0, .encodings = kBf16},
                                               ex::OperandRule{.operand = 1, .encodings = kBf16}};
  static constexpr std::array kFloatProducts = {ex::OperandRule{.operand = 0, .encodings = kBf16},
                                                ex::OperandRule{.operand = 1, .encodings = kF32}};
  const auto prim = [](Op op) { return Any(ir::Primitive(op)); };
  std::vector<ex::Implementation> everything = Everything();
  for (const Op op : {Op::kRepeat, Op::kConcat, Op::kUnary, Op::kConvert})
    everything.push_back(Fake(std::format("t.{}", ir::OpKeyName(ir::Primitive(op))), prim(op)));
  everything.push_back(Fake("t.gated_split", Any(ir::Fused(FusedKind::kGatedResidualSplit))));
  everything.push_back(Fake("t.mul_mat.float", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                                              .operands = kFloatProducts}));
  everything.push_back(Fake("t.mul_mat.bf16", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                                             .operands = kBf16Products}));
  const ex::Registry r = Registry(std::move(everything));
  Policy on{.enabled = {"bf16_branch", "split_gated_residual"}};
  Canonicalize(on);
  const auto graph = [](bool read_twice) {
    ir::Builder b;
    const TensorId x = b.Scale(b.Input(F32({64, 32})), 2.0F);
    const TensorId carry = b.Scale(b.Input(F32({64, 32})), 3.0F);
    const TensorId w = b.Weight(ir::Dense(Encoding::kBF16, ir::ShapeOf({64, 64})), W(0));
    const TensorId y = b.MatMul(w, x);
    const TensorId side = b.Repeat(b.Input(F32({64})), ir::ShapeOf({64, 32}));
    b.Output(b.Add(carry, b.Mul(y, b.Unary(side, ir::UnaryFn::kTanh))));
    if (read_twice) b.Output(b.Scale(y, 2.0F));
    return Finish(b);
  };
  const auto branch = [](const StepPlan& p) {
    for (const ir::Node& n : p.graph.nodes())
      if (n.key == ir::Fused(FusedKind::kGatedResidualSplit))
        return p.graph.tensor(n.src[1]).format.encoding;
    return Encoding::kF16;  // none
  };
  const StepPlan p =
      NumericalPlanOf(graph(false), r, on, {NumericalChange::kBf16ProductInputAndResult});
  EXPECT_EQ(Patterns(p), (std::vector<std::string_view>{"split_gated_residual", "bf16_branch"}));
  EXPECT_EQ(branch(p), Encoding::kBF16);
  // The input converted once, for the product.
  EXPECT_EQ(std::ranges::count(p.graph.nodes(), ir::Primitive(Op::kConvert), &ir::Node::key), 1);
  // Missing numerical authority or another product reader keeps F32.
  Policy without{.enabled = {"split_gated_residual"}};
  Canonicalize(without);
  EXPECT_EQ(branch(PlanOf(graph(false), r, without)), Encoding::kF32);
  EXPECT_EQ(
      branch(NumericalPlanOf(graph(true), r, on, {NumericalChange::kBf16ProductInputAndResult})),
      Encoding::kF32);
}

TEST(PlannerPatternTest, Round1Bf16BranchPreservesOnlyItsMatchingPreparedArithmeticSeal) {
  static constexpr std::array bf16{Encoding::kBF16};
  static constexpr std::array rules{ex::OperandRule{.operand = 0, .encodings = bf16},
                                    ex::OperandRule{.operand = 1, .encodings = bf16}};
  for (unsigned control = 0; control < 8; ++control) {
    SCOPED_TRACE(control);
    auto implementations = Everything();
    for (const auto op : {Op::kRepeat, Op::kUnary, Op::kConvert})
      implementations.push_back(
          Fake(std::format("test.{}", ir::OpName(op)), Any(ir::Primitive(op))));
    implementations.push_back(
        Fake("test.residual", Any(ir::Fused(FusedKind::kGatedResidualSplit))));
    auto arithmetic = +[](const ex::CapabilityQuery&) { return ir::ProductArithmetic::kBf16F32; };
    if (control == 7)
      arithmetic = +[](const ex::CapabilityQuery& query) {
        const auto output =
            std::ranges::find(query.operands, ex::kOutputOperand, &ex::OperandFormat::operand);
        return output != query.operands.end() && output->encoding == Encoding::kF32
                   ? ir::ProductArithmetic::kBf16F32
                   : ir::ProductArithmetic::kImplementation;
      };
    implementations.push_back(
        Fake("test.bf16_product", ex::Capability{.op = ir::Primitive(Op::kMatMul),
                                                 .operands = rules,
                                                 .product_arithmetic = arithmetic}));
    const auto registry = Registry(std::move(implementations));
    ir::Builder builder;
    builder.SetComponent(1);
    auto input = ir::Dense(control == 5 ? Encoding::kF32 : Encoding::kBF16, ir::ShapeOf({64, 32}));
    const auto x = builder.Input(input);
    const auto w = builder.Weight(
        ir::Dense(control == 6 ? Encoding::kF32 : Encoding::kBF16, ir::ShapeOf({64, 64})), W(0));
    ir::ProductParams params{.arithmetic = control == 2 ? ir::ProductArithmetic::kGgmlQ8_1F32
                                                        : ir::ProductArithmetic::kBf16F32};
    if (control == 3) params.accumulate = ir::Accumulate::kF32;
    if (control == 4) params.hint = ir::ProductHint::kHadamard;
    const auto product = builder.MatMul(w, x, params);
    const auto carry = builder.Input(F32({64, 32}));
    const auto side = builder.Repeat(builder.Input(F32({64})), ir::ShapeOf({64, 32}));
    builder.Output(
        builder.Add(carry, builder.Mul(product, builder.Unary(side, ir::UnaryFn::kTanh))));
    auto graph = Finish(builder);
    Policy policy;
    if (control != 1) {
      const std::array changes{NumericalChange::kBf16ProductInputAndResult};
      policy = FixtureNumericalPolicy(graph, policy, changes);
    }
    const SelectionView view(registry, kDevice, policy, false, false);
    // ApplyPatterns isolates matcher refusal even for an intentionally
    // unsupported foreign seal. It never invents a runnable implementation.
    auto rewritten = ApplyPatterns(graph, view);
    ASSERT_TRUE(rewritten) << (rewritten ? "" : rewritten.error().detail);
    EXPECT_EQ(std::ranges::count(rewritten->applied, "bf16_branch", &Applied::pattern),
              control == 0 ? 1 : 0);
    const auto kept =
        std::ranges::find(rewritten->graph.nodes(), ir::Primitive(Op::kMatMul), &ir::Node::key);
    ASSERT_NE(kept, rewritten->graph.nodes().end());
    EXPECT_EQ(kept->params, ir::ToBytes(params));
    EXPECT_EQ(rewritten->graph.tensor(kept->out[0]).format.encoding,
              control == 0 ? Encoding::kBF16 : Encoding::kF32);
    if (control == 1) EXPECT_FALSE(rewritten->missing_numerical.empty());
    if (control == 0) {
      // Final selection must honor the retained seal as well as the matcher.
      const auto plan = PlanOf(graph.Clone(), registry, policy);
      EXPECT_EQ(std::ranges::count(plan.applied, "bf16_branch", &Applied::pattern), 1);
      const auto selected =
          std::ranges::find(plan.graph.nodes(), ir::Primitive(Op::kMatMul), &ir::Node::key);
      ASSERT_NE(selected, plan.graph.nodes().end());
      EXPECT_EQ(selected->params, ir::ToBytes(params));
      EXPECT_EQ(plan.graph.tensor(selected->out[0]).format.encoding, Encoding::kBF16);
    }
  }
}

TEST(PlannerPatternTest, Bf16ResidualKeepsTheSumANextResidualCarriesInBf16) {
  const auto prim = [](Op op) { return Any(ir::Primitive(op)); };
  std::vector<ex::Implementation> everything = Everything();
  for (const Op op : {Op::kLayerNorm, Op::kRepeat, Op::kConcat, Op::kUnary, Op::kConvert})
    everything.push_back(Fake(std::format("t.{}", ir::OpKeyName(ir::Primitive(op))), prim(op)));
  everything.push_back(
      Fake("t.modulate_split", Any(ir::Fused(FusedKind::kLayerNormModulateSplit))));
  everything.push_back(Fake("t.gated_split", Any(ir::Fused(FusedKind::kGatedResidualSplit))));
  everything.push_back(
      Fake("t.gated_modulate_split", Any(ir::Fused(FusedKind::kGatedResidualModulateSplit))));
  const ex::Registry r = Registry(std::move(everything));
  Policy on{.enabled = {"bf16_residual", "split_gated_residual", "split_gated_residual_modulate",
                        "split_modulate"}};
  Canonicalize(on);
  const auto graph = [](bool read_elsewhere) {
    ir::Builder b;
    const TensorId x = b.Scale(b.Input(F32({64, 32})), 2.0F);
    const auto side = [&] { return b.Repeat(b.Input(F32({64})), ir::ShapeOf({64, 32})); };
    const auto gated = [&](TensorId carry) {
      const TensorId y = b.Scale(b.Input(F32({64, 32})), 3.0F);
      return b.Add(carry, b.Mul(y, b.Unary(side(), ir::UnaryFn::kTanh)));
    };
    const TensorId sum = gated(x);
    b.Output(b.Mul(b.LayerNorm(sum, 1e-6F), b.Scale(side(), 1.0F, 1.0F)));
    b.Output(gated(sum));
    if (read_elsewhere) b.Output(b.Scale(sum, 2.0F));
    return Finish(b);
  };
  const auto sum = [](const StepPlan& p) {
    for (const ir::Node& n : p.graph.nodes())
      if (n.key == ir::Fused(FusedKind::kGatedResidualModulateSplit))
        return p.graph.tensor(n.out[0]).format.encoding;
    return Encoding::kF16;  // none
  };
  EXPECT_EQ(sum(NumericalPlanOf(graph(false), r, on, {NumericalChange::kBf16ResidualCarry})),
            Encoding::kBF16);
  // Missing numerical authority or another sum reader keeps F32.
  Policy without = on;
  std::erase(without.enabled, "bf16_residual");
  without.disabled.push_back("bf16_residual");
  Canonicalize(without);
  EXPECT_EQ(sum(NumericalPlanOf(graph(false), r, without, {NumericalChange::kBf16ResidualCarry})),
            Encoding::kF32);
  EXPECT_EQ(sum(NumericalPlanOf(graph(true), r, on, {NumericalChange::kBf16ResidualCarry})),
            Encoding::kF32);
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
  EXPECT_EQ(Patterns(PlanOf(graph(64), r, Disabling({"interleaved_norm_rope"}))),
            std::vector<std::string_view>{"rms_norm_mul_rope"});
  EXPECT_EQ(Keys(PlanOf(graph(64), r, Disabling({"interleaved_norm_rope"})).graph),
            std::vector{ir::Fused(FusedKind::kRmsNormMulRope)});
  // Disabled, or at a width its implementation does not take: the norm and
  // its weight still fuse, the rotation stays.
  EXPECT_EQ(
      Patterns(PlanOf(graph(64), r, Disabling({"interleaved_norm_rope", "rms_norm_mul_rope"}))),
      std::vector<std::string_view>{"rms_norm_mul"});
  const StepPlan wide = PlanOf(graph(128), r, Disabling({"interleaved_norm_rope"}));
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
  const auto isolated = [&](ir::Graph graph, Policy policy = {}) {
    policy.disabled.push_back("vecq_inputs");
    Canonicalize(policy);
    return PlanOf(std::move(graph), r, policy);
  };
  const StepPlan p = isolated(GluGraph(1, false, false, false));
  EXPECT_EQ(Patterns(p), std::vector<std::string_view>{"mul_mat_glu"});
  EXPECT_EQ(Names(p, r), std::vector<std::string>{"t.mul_mat_glu"});
  const ir::Node& fused = p.graph.nodes().back();
  EXPECT_EQ(fused.Params<ir::GluParams>().kind, ir::GluKind::kGeGluTanh);
  // Not past the implementation's rows (the two products then share their
  // prepared input instead), not with the gate read elsewhere, not over
  // two inputs, not where a product asks for F32 sums.
  EXPECT_EQ(Patterns(isolated(GluGraph(2, false, false, false))),
            std::vector<std::string_view>{"prepared_inputs"});
  EXPECT_TRUE(
      Patterns(isolated(GluGraph(1, true, false, false), Disabling({"prepared_inputs"}))).empty());
  EXPECT_TRUE(Patterns(isolated(GluGraph(1, false, true, false))).empty());
  EXPECT_TRUE(
      Patterns(isolated(GluGraph(1, false, false, true), Disabling({"prepared_inputs"}))).empty());
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
  const auto isolated = [&](ir::Graph graph, Policy policy = {}) {
    policy.disabled.push_back("vecq_inputs");
    Canonicalize(policy);
    return PlanOf(std::move(graph), r, policy);
  };
  const StepPlan p = isolated(SharedGraph(3, 2));
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
  EXPECT_TRUE(Patterns(isolated(SharedGraph(1, 2))).empty());
  const StepPlan wide = isolated(SharedGraph(3, 16));
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

TEST(PlannerBatchTest, GraphCandidateQueriesRequireMeasuredQualificationToo) {
  auto implementations = Everything();
  for (auto& implementation : implementations)
    if (implementation.name == "t.set_rows.grouped")
      implementation.capability->fusion_needs_measurement = true;
  const auto registry = Registry(std::move(implementations));
  // Grouping requires independent stores whose results are not externally
  // exposed. The separate kept-output control below must remain ungrouped.
  const auto graph = [] {
    ir::Builder b;
    const auto source = b.Input(F32({16, 2}));
    const auto ids = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({2})));
    for (std::uint16_t tensor = 0; tensor < 2; ++tensor) {
      const auto cache = b.State(ir::Dense(Encoding::kF16, ir::ShapeOf({16, 8})),
                                 {.stage = 0, .tensor = tensor, .layer = 0, .segment = 0});
      b.CacheStore(cache, source, ids);
    }
    return Finish(b);
  };
  EXPECT_EQ(StepSizes(PlanOf(graph(), registry)), (std::vector<std::size_t>{1, 1}));
  Policy measured;
  measured.ranks = {{.device_class = "test", .implementation = "t.set_rows.grouped", .rank = 0}};
  EXPECT_EQ(StepSizes(PlanOf(graph(), registry, measured)), std::vector<std::size_t>{2});
  EXPECT_EQ(StepSizes(PlanOf(KeptStoresGraph(2), registry, measured)),
            (std::vector<std::size_t>{1, 1}));
  measured.ranks[0].device_class = "different_device";
  EXPECT_EQ(StepSizes(PlanOf(graph(), registry, measured)), (std::vector<std::size_t>{1, 1}));
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
  // At one row the GLU fuses and K and V share a preparation; each
  // down-product also prepares its singleton input. At 32 rows none apply.
  EXPECT_EQ(
      Patterns(PlanOf(LayerGraph(1), r)),
      (std::vector<std::string_view>{
          "rms_norm_mul", "prepared_inputs", "mul_mat_glu", "rms_norm_mul_add", "rms_norm_mul",
          "prepared_inputs", "mul_mat_glu", "rms_norm_mul_add", "vecq_inputs", "vecq_inputs"}));
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

TEST(PlannerPlanTest, RepeatedExplicitAndAutomaticNodesRetainTheirSelectionContracts) {
  auto implementations = Everything();
  auto competitor = Any(ir::Fused(FusedKind::kRmsNormMul));
  competitor.priority = 100;
  competitor.fusion_needs_measurement = true;
  implementations.push_back(Fake("t.unmeasured_norm", competitor));
  const auto registry = Registry(std::move(implementations));
  for (const bool direct_first : {false, true}) {
    SCOPED_TRACE(direct_first);
    ir::Builder b;
    const auto x = b.Input(F32({64, 3}));
    const auto w = b.Weight(F32({64}), W(0));
    const auto direct = [&] {
      return b.Fused(FusedKind::kRmsNormMul, std::array{x, w}, ir::NormParams{.eps = 1e-6F})[0];
    };
    const auto automatic = [&] { return b.Mul(b.RmsNorm(x, 1e-6F), w); };
    const auto first = direct_first ? direct() : automatic();
    const auto second = direct_first ? automatic() : direct();
    b.Output(b.Add(b.Scale(first, 2.0F), b.Scale(second, 3.0F)));
    const auto plan = PlanOf(Finish(b), registry);
    const auto names = Names(plan, registry);
    EXPECT_EQ(std::ranges::count(names, "t.rms_norm_mul"), 1);
    EXPECT_EQ(std::ranges::count(names, "t.unmeasured_norm"), 1);
    ExpectDisjoint(plan);
  }
}

// Invariant inputs (PlanOptions::invariant): the steps that read only them
// and weights are the plan's prologue; what the rest reads from it stays
// in place the whole step, sharing its bytes with nothing.
TEST(PlannerPlanTest, InvariantInputsMakeAPrologueWhoseResultsStay) {
  const auto graph = [] {
    ir::Builder b;
    const TensorId fixed = b.Input(F32({64, 8}));
    const TensorId varying = b.Input(F32({64, 8}));
    const TensorId w = b.Weight(F32({64}), W(0));
    // The prologue: a scale of the invariant input, then its product with a
    // weight; a scale of it the step reads twice.
    const TensorId prepared = b.Mul(b.Scale(fixed, 2.0F), w);
    // The rest: what varies, a long chain the prepared value must survive.
    TensorId x = b.Add(b.Scale(varying, 3.0F), prepared);
    for (int i = 0; i < 6; ++i) x = b.Scale(b.Add(x, x), 0.5F);
    b.Output(b.Add(x, prepared));
    return Finish(b);
  };
  const ex::Registry r = Registry(Everything());
  const ir::Graph g = graph();
  const TensorId fixed = g.inputs()[0];
  const StepPlan plain = PlanOf(graph(), r);
  EXPECT_TRUE(plain.prologue.empty());
  const StepPlan p = PlanOf(graph(), r, {}, {.invariant = {fixed}});
  EXPECT_EQ(p.invariant, std::vector<TensorId>{fixed});
  ASSERT_EQ(p.prologue.size(), 2U);  // the scale and the product
  EXPECT_EQ(p.prologue, (std::vector<std::uint32_t>{0, 1}));
  ExpectDisjoint(p);
  // The product the rest reads shares its bytes with no other tensor.
  const ir::Node& made = p.graph.node(p.steps[1].nodes.front());
  const TensorId kept = made.out[0];
  std::uint64_t at = 0;
  for (const auto& [t, offset] : p.placement.offsets)
    if (t == kept) at = offset;
  const std::uint64_t bytes = (RootBytes(p.graph.tensor(kept)).value() + 255) / 256 * 256;
  for (const auto& [t, offset] : p.placement.offsets) {
    if (t == kept) continue;
    const std::uint64_t other = (RootBytes(p.graph.tensor(t)).value() + 255) / 256 * 256;
    EXPECT_TRUE(at + bytes <= offset || offset + other <= at) << t.v;
  }
  // The other input declared instead: its own scale alone.
  const StepPlan other = PlanOf(graph(), r, {}, {.invariant = {g.inputs()[1]}});
  EXPECT_EQ(other.prologue.size(), 1U);
  ExpectDisjoint(other);
  // Both: every step.
  const StepPlan both = PlanOf(graph(), r, {}, {.invariant = {fixed, g.inputs()[1]}});
  EXPECT_EQ(both.prologue.size(), both.steps.size());
}

// Never the prologue's: what reads a state, and a store or a collective
// reading an invariant input; such an input is then copied every run (not
// kept in StepPlan::invariant), and nothing reuses a kept one's bytes.
TEST(PlannerPlanTest, AStateReadAStoreOrACollectiveIsNoPrologue) {
  auto implementations = Everything();
  implementations.push_back(Fake("t.all_reduce", Any(ir::Primitive(Op::kAllReduce))));
  const ex::Registry r = Registry(std::move(implementations));
  {  // A state read beside the invariant input: only the scale is fixed.
    const auto graph = [] {
      ir::Builder b;
      const TensorId x = b.Input(F32({64, 8}));
      const TensorId state = b.State(F32({64, 8}), ir::StateRef{});
      const TensorId fixed = b.Scale(x, 2.0F);
      b.Output(b.Add(b.Mul(fixed, state), b.Scale(state, 3.0F)));
      return Finish(b);
    };
    const TensorId in = graph().inputs()[0];
    const StepPlan p = PlanOf(graph(), r, {}, {.invariant = {in}});
    EXPECT_EQ(p.prologue.size(), 1U);
    EXPECT_EQ(p.invariant, std::vector<TensorId>{in});
    ExpectDisjoint(p);
  }
  const auto stored = [](bool collective) {
    ir::Builder b;
    const TensorId x = b.Input(F32({128, 3}));
    const TensorId cache = b.State(F32({128, 8}), ir::StateRef{});
    const TensorId idx = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({3})));
    if (collective) {
      b.Output(b.AllReduce(b.Scale(x, 2.0F), {}));
      b.Output(b.AllReduce(x, {}));
    } else {
      b.CacheStore(cache, x, idx);
      b.Output(b.Scale(x, 2.0F));
    }
    return Finish(b);
  };
  for (const bool collective : {false, true}) {
    SCOPED_TRACE(collective);
    const ir::Graph g = stored(collective);
    const TensorId x = g.inputs()[0];
    const StepPlan p = PlanOf(stored(collective), r, {}, {.invariant = {x}});
    // The input a store or a collective reads is written in place or sent:
    // not kept, nothing is the prologue's, and it is copied every run.
    EXPECT_TRUE(p.prologue.empty());
    EXPECT_TRUE(p.invariant.empty());
    ExpectDisjoint(p);
  }
}

TEST(PlannerPlanTest, NodesNothingReadsAreNotPlanned) {
  const auto graph = [](bool dead) {
    ir::Builder b;
    const TensorId x = b.Input(F32({64, 8}));
    const TensorId y = b.Input(F32({64, 8}));  // read only by the unread chain
    const TensorId w = b.Weight(F32({64}), W(0));
    const TensorId v = b.Weight(F32({64}), W(1));
    const TensorId kept = b.Mul(b.Scale(x, 2.0F), w);
    if (dead) {
      // Unread: a chain over the kept value, a weight and the other input,
      // and a view of it.
      const TensorId chain = b.Add(b.Mul(kept, v), b.Scale(y, 3.0F));
      (void)b.Reshape(b.Scale(chain, 0.5F), ir::ShapeOf({512}));
    }
    b.Output(b.Add(kept, x));
    return Finish(b);
  };
  const ex::Registry r = Registry(Everything());
  const StepPlan live = PlanOf(graph(false), r);
  const StepPlan pruned = PlanOf(graph(true), r);
  EXPECT_EQ(pruned.steps.size(), live.steps.size());
  // The leaves stay: the step's inputs and weights are its interface.
  EXPECT_EQ(pruned.graph.inputs().size(), 2U);
  std::size_t computed = 0, computed_live = 0;
  for (const auto& n : pruned.graph.nodes())
    computed += n.key.op != ir::Op::kInput && n.key.op != ir::Op::kWeight ? 1 : 0;
  for (const auto& n : live.graph.nodes())
    computed_live += n.key.op != ir::Op::kInput && n.key.op != ir::Op::kWeight ? 1 : 0;
  EXPECT_EQ(computed, computed_live);
  ExpectDisjoint(pruned);
}

TEST(PlannerPlanTest, ARepeatedNodeRunsOnce) {
  const auto graph = [](bool repeat, bool other) {
    ir::Builder b;
    const TensorId x = b.Input(F32({64, 8}));
    const TensorId y = b.Input(F32({64, 8}));
    const TensorId first = b.Concat(x, y, 1);
    // Again over the same sources, or over others.
    const TensorId again = repeat ? b.Concat(x, other ? x : y, 1) : first;
    b.Output(b.Add(b.Scale(first, 2.0F), b.Scale(again, 3.0F)));
    return Finish(b);
  };
  auto implementations = Everything();
  implementations.push_back(Fake("t.concat", Any(ir::Primitive(Op::kConcat))));
  const ex::Registry r = Registry(std::move(implementations));
  const StepPlan once = PlanOf(graph(false, false), r);
  const StepPlan joined = PlanOf(graph(true, false), r);
  EXPECT_EQ(joined.steps.size(), once.steps.size());
  ExpectDisjoint(joined);
  const StepPlan distinct = PlanOf(graph(true, true), r);
  EXPECT_EQ(distinct.steps.size(), once.steps.size() + 1);
}

// Never joined: the step's outputs and kept tensors, stores, and reads of
// a root a store or a fused node's aliased output writes in place (the
// same read before and after a write sees different values).
TEST(PlannerPlanTest, NoRepeatJoinsAWriteInPlaceOrWhatTheStepReturns) {
  const ex::Registry r = Registry(Everything());
  const auto scales = [](const StepPlan& p) {
    return std::ranges::count(p.graph.nodes(), ir::Primitive(Op::kScale), &ir::Node::key);
  };
  const auto stores = [](const StepPlan& p) {
    return std::ranges::count(p.graph.nodes(), ir::Primitive(Op::kCacheStore), &ir::Node::key);
  };
  {  // Two outputs.
    ir::Builder b;
    const TensorId x = b.Input(F32({64, 8}));
    b.Output(b.Scale(x, 2.0F));
    b.Output(b.Scale(x, 2.0F));
    EXPECT_EQ(scales(PlanOf(Finish(b), r)), 2);
  }
  {  // One kept.
    ir::Builder b;
    const TensorId x = b.Input(F32({64, 8}));
    b.Keep(b.Scale(x, 2.0F));
    b.Output(b.Add(b.Scale(x, 2.0F), x));
    EXPECT_EQ(scales(PlanOf(Finish(b), r)), 2);
  }
  const auto stored = [](bool fused) {
    ir::Builder b;
    const TensorId cache = b.State(F32({128, 8}), ir::StateRef{});
    const TensorId idx = b.Input(ir::Dense(Encoding::kI64, ir::ShapeOf({3})));
    const TensorId before = b.Scale(cache, 2.0F);
    if (fused) {
      const TensorId x = b.Input(F32({64, 2, 3}));
      const TensorId pos = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({3})));
      const ir::RopeParams rope{.mode = ir::RopeMode::kNeox, .dims = 64, .base = 10000.0F};
      (void)b.Fused(FusedKind::kRopeStore, std::array{x, pos, cache, idx, TensorId{}}, rope);
    } else {
      // The same store twice: each a write, neither the other's repeat.
      const TensorId rows = b.Input(F32({128, 3}));
      b.CacheStore(cache, rows, idx);
      b.CacheStore(cache, rows, idx);
    }
    b.Output(b.Add(before, b.Scale(cache, 2.0F)));
    return Finish(b);
  };
  const StepPlan store = PlanOf(stored(false), r);
  EXPECT_EQ(scales(store), 2);
  EXPECT_EQ(stores(store), 2);
  EXPECT_EQ(scales(PlanOf(stored(true), r)), 2);
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

// A graph's identity is its own: a clone or another build has another, a
// move carries it (the moved-from graph has none), and the alignment
// cache keys on it, so a graph built where a freed one was (equal counts,
// often the same node storage) is authenticated afresh, never served the
// freed graph's facts.
TEST(PlannerSelectionTest, RootAlignmentCacheKeysOnGraphIdentity) {
  const auto build = [](bool reverse) {
    ir::Builder b;
    const ir::StateRef first{.tensor = 0}, second{.tensor = 1};
    const auto f = ir::Dense(Encoding::kF32, ir::ShapeOf({16}));
    const auto a = b.State(f, reverse ? second : first);
    const auto z = b.State(f, reverse ? first : second);
    b.Output(b.Add(a, z));
    return std::move(b).Finish();
  };
  auto first = build(false);
  ASSERT_TRUE(first);
  const std::uint64_t identity = first->identity();
  EXPECT_NE(identity, 0U);
  auto clone = first->Clone();
  EXPECT_NE(clone.identity(), 0U);
  EXPECT_NE(clone.identity(), identity);
  ir::Graph moved = std::move(*first);
  EXPECT_EQ(moved.identity(), identity);
  EXPECT_EQ(first->identity(), 0U);  // NOLINT(bugprone-use-after-move): its identity is gone
  const auto aligned =
      GraphRootAlignments(moved, std::array{RootAlignmentFact{.root = {0}, .alignment = 256}});
  ASSERT_TRUE(aligned);
  for (int attempt = 0; attempt < 8; ++attempt) {
    std::optional<ir::Graph> held = clone.Clone();
    EXPECT_EQ(aligned(*held, {0}), 256U);
    const ir::Node* storage = held->nodes().data();
    held.reset();
    auto other = build(true);  // the same counts, its roots' meanings swapped
    ASSERT_TRUE(other);
    EXPECT_NE(other->identity(), identity);
    SCOPED_TRACE(other->nodes().data() == storage ? "reused storage" : "new storage");
    EXPECT_EQ(aligned(*other, {0}), 0U);
  }
  EXPECT_EQ(aligned(moved, {0}), 256U);
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

TEST(PlannerSelectionTest, RootFactsForwardOnlyThroughAuthenticatedLeafMaps) {
  const auto make = [](bool reverse, bool extra) {
    ir::Builder b;
    TensorId a, z;
    if (reverse) {
      z = b.State(F32({16}), {.tensor = 1});
      a = b.State(F32({16}), {.tensor = 0});
    } else {
      a = b.State(F32({16}), {.tensor = 0});
      z = b.State(F32({16}), {.tensor = 1});
    }
    const auto sum = b.Add(a, z);
    b.Output(extra ? b.Unary(sum, ir::UnaryFn::kRelu) : sum);
    return Finish(b);
  };
  const auto original = make(false, false), changed = make(true, true);
  const auto facts = GraphRootAlignments(
      original, std::array{RootAlignmentFact{{0}, 256}, RootAlignmentFact{{1}, 4},
                           RootAlignmentFact{{2}, 256}});
  const std::array map{TensorId{1}, TensorId{0}, TensorId{2}};
  auto moved = ForwardRootAlignments(original, changed, map, facts);
  ASSERT_TRUE(moved);
  EXPECT_EQ(moved(changed, {0}), 4U);
  EXPECT_EQ(moved(changed, {1}), 256U);
  // The carried facts are sealed to `changed`'s identity, which a move keeps
  // and a clone does not (a planner pass's graph is never hashed for them).
  ir::Graph kept = changed.Clone();
  const auto kept_facts = ForwardRootAlignments(original, kept, map, facts);
  ir::Graph moved_on = std::move(kept);
  EXPECT_EQ(kept_facts(moved_on, {0}), 4U);
  EXPECT_EQ(kept_facts(moved_on.Clone(), {0}), 0U);
  EXPECT_EQ(moved(changed, {2}), 0U);  // computed root cannot inherit old placement
  // An unchanged graph (an equal clone: another identity, the same digest
  // and roots) keeps its computed root's fact; a changed one does not.
  const std::array identity_map{TensorId{0}, TensorId{1}, TensorId{2}};
  const ir::Graph same = original.Clone();
  const auto unchanged = ForwardRootAlignments(original, same, identity_map, facts);
  ASSERT_TRUE(unchanged);
  EXPECT_EQ(unchanged(same, {2}), 256U);
  EXPECT_EQ(unchanged(same, {0}), 256U);
  const ir::Graph extended = make(false, true);
  const auto altered = ForwardRootAlignments(original, extended, identity_map, facts);
  ASSERT_TRUE(altered);
  EXPECT_EQ(altered(extended, {2}), 0U);
  EXPECT_EQ(altered(extended, {0}), 256U);  // the leaf itself is unchanged
  EXPECT_EQ(moved(changed, {3}), 0U);
  const std::array wrong{TensorId{0}, TensorId{1}, TensorId{2}};
  auto rejected = ForwardRootAlignments(original, changed, wrong, facts);
  EXPECT_EQ(rejected(changed, {0}), 0U);
  EXPECT_EQ(rejected(changed, {1}), 0U);
  const std::array duplicate{TensorId{1}, TensorId{1}, TensorId{2}};
  rejected = ForwardRootAlignments(original, changed, duplicate, facts);
  EXPECT_EQ(rejected(changed, {1}), 0U);
  EXPECT_FALSE(ForwardRootAlignments(original, changed, std::span(map).first(2), facts));
  EXPECT_EQ(moved(make(false, true), {0}), 0U);  // equal-shape root IDs differ again
}

TEST(PlannerSelectionTest, ActivationFactsFollowMappedStagingAndExcludeContainedRoots) {
  ir::Builder b;
  const auto x = b.Input(F32({16, 2}));
  const auto not_staged = b.Input(F32({16}));
  const auto n = b.RmsNorm(x, 1e-5F);
  const auto output = b.Mul(n, not_staged);
  b.Output(output);
  auto graph = Finish(b);
  const auto registry = Registry();
  const Policy policy;
  SelectionView view(registry, kDevice, policy, false, false, {}, {},
                     {.base = 65536, .offset = 256, .staged = std::vector{x}});
  EXPECT_EQ(view.Alignment(graph, x), 256U);
  EXPECT_EQ(view.Alignment(graph, not_staged), 0U);
  EXPECT_EQ(view.Alignment(graph, n), 256U);
  const auto view_tensor = graph.root(output);
  EXPECT_EQ(view.Alignment(graph, view_tensor), 256U);
  auto rewritten = ApplyPatterns(graph, view);
  ASSERT_TRUE(rewritten);
  ASSERT_EQ(rewritten->applied.size(), 1U);
  view.Forward(graph, rewritten->graph, rewritten->tensors);
  EXPECT_EQ(view.Alignment(rewritten->graph, rewritten->tensors[x.v]), 256U);
  EXPECT_EQ(view.Alignment(rewritten->graph, rewritten->tensors[not_staged.v]), 0U);
  std::vector<TensorId> map;
  auto expanded = ir::Expand(rewritten->graph, true, &map);
  ASSERT_TRUE(expanded);
  view.Forward(rewritten->graph, *expanded, map);
  EXPECT_EQ(view.Alignment(*expanded, map[rewritten->tensors[x.v].v]), 256U);
  for (const auto id : expanded->outputs())
    EXPECT_EQ(view.Alignment(*expanded, expanded->root(id)), 256U);
  SelectionView contained(registry, kDevice, policy, false, false, {}, {},
                          {.base = 65536,
                           .offset = 256,
                           .staged = std::vector{x},
                           .within = [n, output](const ir::Graph&, TensorId id) {
                             return id == output ? n : TensorId{};
                           }});
  EXPECT_EQ(contained.Alignment(graph, output), 0U);
  EXPECT_EQ(contained.Alignment(graph, n), 256U);
  SelectionView unknown(registry, kDevice, policy, false, false);
  EXPECT_EQ(unknown.Alignment(graph, x), 0U);
  EXPECT_EQ(unknown.Alignment(graph, output), 0U);
}

TEST(PlannerTest, AlignmentFactsSurviveFallbackAndAreAuditedAgainstFinalPlacement) {
  ir::Builder b;
  const auto x = b.Input(F32({16, 2}));
  const auto weight = b.Input(F32({16}));
  b.Output(b.Fused(FusedKind::kRmsNormMul, std::array{x, weight}, ir::NormParams{.eps = 1e-5F})[0]);
  auto graph = Finish(b);
  auto registry = Without({"t.rms_norm_mul"});
  auto plan = Plan(graph.Clone(), registry, kDevice, {}, {.activation_alignment = 65536});
  ASSERT_TRUE(plan) << plan.error().detail;
  EXPECT_EQ(plan->decomposed, std::vector<std::string>{"fused.rms_norm_mul"});
  EXPECT_FALSE(plan->root_alignments.empty());
  for (const auto& fact : plan->root_alignments) EXPECT_EQ(fact.alignment, 256U);
  auto unsupported = Plan(graph.Clone(), registry, kDevice, {}, {.activation_alignment = 3});
  ASSERT_FALSE(unsupported);
  EXPECT_EQ(unsupported.error().error, PlanError::kPlacement);
  // A caller cannot claim a stronger guarantee than its actual offsets.
  const auto too_strong = GraphRootAlignments(
      graph, std::array{RootAlignmentFact{x, 65536}, RootAlignmentFact{weight, 65536}});
  auto invalid = Plan(graph.Clone(), registry, kDevice, {}, {.root_alignment = too_strong});
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error().error, PlanError::kPlacement);
}

TEST(PlannerTest, ExternalSaveInputsDoNotInheritTheArenaPromise) {
  ir::Builder b;
  const auto staged = b.Input(F32({16}));
  const auto saved = b.Input(F32({16}));
  b.Output(b.Add(staged, saved));
  auto graph = Finish(b);
  const auto registry = Registry();
  auto plan = Plan(graph.Clone(), registry, kDevice, {},
                   {.staged = std::vector{staged, saved},
                    .activation_alignment = 65536,
                    .alignment_staged = std::vector{staged}});
  ASSERT_TRUE(plan) << plan.error().detail;
  EXPECT_TRUE(std::ranges::any_of(plan->root_alignments, [&](const auto& fact) {
    return fact.root == plan->tensors[staged.v] && fact.alignment == 256;
  }));
  EXPECT_FALSE(std::ranges::any_of(plan->root_alignments, [&](const auto& fact) {
    return fact.root == plan->tensors[saved.v];
  }));
  EXPECT_TRUE(std::ranges::any_of(plan->placement.offsets, [&](const auto& place) {
    return place.first == plan->tensors[saved.v];  // legacy placement unchanged
  }));
  auto invalid = Plan(graph.Clone(), registry, kDevice, {},
                      {.staged = std::vector{staged},
                       .activation_alignment = 65536,
                       .alignment_staged = std::vector{saved}});
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error().error, PlanError::kPlacement);
}

TEST(PlannerPlacementTest, PhysicalContractOmitsProofStorageAndRejectsUnfundedReads) {
  ir::Builder b;
  const auto x = b.Input(F32({64}));
  const auto proof = b.Fill(ir::ShapeOf({4096}), 0.0F);
  const auto y = b.Add(x, x);
  b.Output(y);
  auto graph = Finish(b);
  std::vector<Selected> steps;
  for (std::uint32_t i = 0; i < graph.nodes().size(); ++i)
    if (!LaunchesNothing(graph.nodes()[i].key)) steps.push_back({.nodes = {ir::NodeId{i}}});
  const std::array staged{x};
  auto logical = PlaceActivations(graph, steps, {}, staged);
  ASSERT_TRUE(logical);
  PhysicalStorage physical;
  physical.materialized.resize(graph.tensor_count(), true);
  physical.materialized[proof.v] = false;
  for (const auto& n : graph.nodes())
    physical.reads.emplace_back(n.sources().begin(), n.sources().end());
  auto real = PlaceActivations(graph, steps, {}, staged, 256, {}, {}, {}, &physical);
  ASSERT_TRUE(real) << real.error();
  EXPECT_LT(real->extent, logical->extent);
  EXPECT_FALSE(std::ranges::any_of(real->offsets, [&](const auto& p) { return p.first == proof; }));
  physical.reads[graph.producer(y).v].push_back(proof);
  auto invalid = PlaceActivations(graph, steps, {}, staged, 256, {}, {}, {}, &physical);
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error(), "a physical read names an absent materialization");
}

}  // namespace
}  // namespace llmp::planner
