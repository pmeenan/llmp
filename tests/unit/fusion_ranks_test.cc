// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "execution/fusion_ranks.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <vector>

#include "ir/builder.h"
#include "ir/decompose.h"
#include "planner/plan.h"
#include "planner/select.h"

namespace {
namespace ex = llmp::execution;
namespace ir = llmp::ir;
namespace pl = llmp::planner;
using llmp::format::Encoding;

ex::Implementation Declaration(std::string name, ir::OpKey op) {
  return {.name = std::move(name),
          .operation = op,
          .source = "synthetic",
          .revision = "test-r1",
          .build = "test",
          .variant = "test",
          .capability = ex::Capability{.op = op, .standalone_comparator = true}};
}
std::vector<ex::Implementation> Declarations() {
  return {Declaration("test.fusion", ir::Fused(ir::FusedKind::kRmsNormMul)),
          Declaration("test.norm", ir::Primitive(ir::Op::kRmsNorm)),
          Declaration("test.mul", ir::Primitive(ir::Op::kMul))};
}
ir::Graph Graph() {
  ir::Builder b;
  const auto x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({256, 4})));
  const auto w = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({256})));
  const std::array sources{x, w};
  b.Output(b.Fused(ir::FusedKind::kRmsNormMul, sources, ir::NormParams{.eps = 1e-6F})[0]);
  return std::move(b).Finish().value();
}
const pl::DeviceFacts kDevice{.device_class = "test.device",
                              .arch = 10,
                              .device_model = "test processor",
                              .multiprocessors = 8};

// All timings here are synthetic unit-test values, never installed backend data.
struct Fixture {
  std::vector<ex::Implementation> declarations = Declarations();
  ex::Registry registry = ex::Registry::Create(declarations).value();
  ir::Graph graph = Graph();
  std::array<ex::OperandFormat, 3> operands{};
  ir::ParamBytes params{};
  ex::CapabilityQuery query{};
  std::vector<std::size_t> terminal;
  ex::FusionComparison record{};
  Fixture() {
    const pl::Policy policy;
    pl::SelectionView view(registry, kDevice, policy, false, false);
    view.Inspect(graph, graph.nodes().back(), [&](const ex::CapabilityQuery& q) {
      std::ranges::copy(q.operands, operands.begin());
      std::ranges::copy(q.params, params.begin());
      terminal.assign(q.terminal_decomposition.begin(), q.terminal_decomposition.end());
      query = q;
      query.terminal_decomposition = terminal;
      query.operands = operands;
      query.params = params;
      // SelectionView owns the policy-option views. Copying the key while it
      // lives below is enough for plan tests; mutation tests use no options.
      query.policy.options = {};
    });
    record = {
        .shape_class = "synthetic short wave",
        .query = *ex::FusionQueryKey(query),
        .selection_environment = registry.SelectionDigest(),
        .fusion = {"test.fusion", registry.identity(0)},
        .decomposition = {{"test.norm", registry.identity(1)}, {"test.mul", registry.identity(2)}},
        .fused_ns = 200,
        .decomposition_ns = 100,
        .samples = 9,
        .provenance = "synthetic test only"};
  }
  ex::Registry Ranked() {
    return ex::Registry::Create(declarations, {.fusions = {record}}).value();
  }
};

TEST(FusionRanksTest, IndexedRecordsAreSeparatedByOperationQueryAndImplementationAfterMove) {
  Fixture f;
  f.declarations.push_back(Declaration("test.other", ir::Fused(ir::FusedKind::kRmsNormMulAdd)));
  f.declarations.push_back(Declaration("test.alternative", f.query.op));
  auto declarations = ex::Registry::Create(f.declarations).value();
  f.record.selection_environment = declarations.SelectionDigest();
  auto other = f.record;
  other.fusion = {"test.other", declarations.identity(3)};
  auto alternative = f.record;
  alternative.fusion = {"test.alternative", declarations.identity(4)};
  alternative.fused_ns = 50;
  auto different = f.record;
  different.query[0] ^= 1;
  auto original =
      ex::Registry::Create(f.declarations, {.fusions = {different, other, alternative, f.record}})
          .value();
  auto registry = std::move(original);
  EXPECT_TRUE(registry.HasFusionComparisons(f.query.op));
  EXPECT_FALSE(registry.HasFusionComparisons(ir::Fused(ir::FusedKind::kRopeStore)));
  EXPECT_FALSE(registry.HasFusionComparisons(ir::Primitive(ir::Op::kRmsNorm)));
  const auto matches = registry.FusionComparisons(f.query.op, f.record.query);
  ASSERT_EQ(matches.size(), 2U);
  EXPECT_EQ(matches[0].comparison->fusion.name, "test.alternative");
  EXPECT_EQ(matches[1].comparison->fusion.name, "test.fusion");
  EXPECT_EQ(registry.FusionComparisons(f.query.op, different.query).size(), 1U);
  auto missing = different.query;
  missing[1] ^= 1;
  EXPECT_TRUE(registry.FusionComparisons(f.query.op, missing).empty());
  EXPECT_EQ(ex::FusionPreferenceFor(registry, f.query, "test.fusion"),
            ex::FusionPreference::kDecomposition);
  EXPECT_EQ(ex::FusionPreferenceFor(registry, f.query, "test.alternative"),
            ex::FusionPreference::kFusion);
  EXPECT_EQ(ex::FusionPreferenceFor(registry, f.query, "test.other"),
            ex::FusionPreference::kUnmeasured);
  auto unrelated = f.query;
  unrelated.op = ir::Fused(ir::FusedKind::kRopeStore);
  EXPECT_TRUE(ex::FusionPreferences(registry, unrelated).empty());
}

TEST(FusionRanksTest, UnmeasuredNeutralWinningAndLosingAreDistinct) {
  Fixture f;
  EXPECT_EQ(ex::FusionPreferenceFor(f.registry, f.query, "test.fusion"),
            ex::FusionPreference::kUnmeasured);
  auto ranked = f.Ranked();
  EXPECT_EQ(ex::FusionPreferenceFor(ranked, f.query, "test.fusion"),
            ex::FusionPreference::kDecomposition);
  EXPECT_TRUE(ex::Candidates(ranked, f.query).empty());
  f.record.fused_ns = 105;
  const auto neutral = f.Ranked();
  EXPECT_EQ(ex::FusionPreferenceFor(neutral, f.query, "test.fusion"),
            ex::FusionPreference::kNeutral);
  EXPECT_EQ(ex::Candidates(neutral, f.query).size(), 1U);
  f.record.fused_ns = 50;
  const auto winning = f.Ranked();
  EXPECT_EQ(ex::FusionPreferenceFor(winning, f.query, "test.fusion"),
            ex::FusionPreference::kFusion);
  EXPECT_EQ(ex::Candidates(winning, f.query).size(), 1U);
}

TEST(FusionRanksTest, LatencyBandUsesTheDecompositionReferenceInBothDirections) {
  Fixture f;
  f.record.decomposition_ns = 1000;
  struct Case {
    std::uint64_t fused;
    ex::FusionPreference expected;
  };
  for (const auto& c : std::array{
           Case{899, ex::FusionPreference::kFusion}, Case{900, ex::FusionPreference::kNeutral},
           Case{905, ex::FusionPreference::kNeutral}, Case{1100, ex::FusionPreference::kNeutral},
           Case{1101, ex::FusionPreference::kDecomposition}}) {
    SCOPED_TRACE(c.fused);
    EXPECT_EQ(ex::ClassifyFusionLatency(c.fused, 1000), c.expected);
    f.record.fused_ns = c.fused;
    const auto ranked = f.Ranked();
    EXPECT_EQ(ex::FusionPreferenceFor(ranked, f.query, "test.fusion"), c.expected);
  }
  constexpr auto max = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(ex::ClassifyFusionLatency(max - max / 10, max), ex::FusionPreference::kNeutral);
  EXPECT_EQ(ex::ClassifyFusionLatency(max - max / 10 - 1, max), ex::FusionPreference::kFusion);
  EXPECT_EQ(ex::ClassifyFusionLatency(max, 1), ex::FusionPreference::kDecomposition);
  EXPECT_EQ(ex::ClassifyFusionLatency(0, 1000), ex::FusionPreference::kUnmeasured);
  EXPECT_EQ(ex::ClassifyFusionLatency(1000, 0), ex::FusionPreference::kUnmeasured);
}

TEST(FusionRanksTest, KeyCoversEveryOperandParameterDeviceAndSelectionFact) {
  Fixture f;
  const auto baseline = ex::FusionQueryKey(f.query);
  const auto misses = [&] { EXPECT_NE(ex::FusionQueryKey(f.query), baseline); };
  for (auto& operand : f.operands) {
    const auto original = operand;
    operand.encoding = Encoding::kBF16;
    misses();
    operand = original;
    operand.params.tensor_scale = true;
    misses();
    operand = original;
    ++*operand.rank;
    misses();
    operand = original;
    ++operand.stacked;
    misses();
    operand = original;
    for (std::size_t i = 0; i < 4; ++i) {
      ++operand.ne[i];
      misses();
      operand = original;
      ++operand.memory->nb[i];
      misses();
      operand = original;
    }
    ++operand.memory->view_offset;
    misses();
    operand = original;
    operand.memory->root_alignment = 256;
    misses();
    operand = original;
    operand.memory->is_view = true;
    misses();
    operand = original;
  }
  const auto q = f.query;
  const std::array<std::string_view, 0> no_options{};
  f.query.policy.options = no_options;
  misses();  // Absent policy options and an explicitly empty set differ.
  f.query = q;
  std::array tunes{ex::TunedShape{.implementation = "test.norm", .rows = 1, .k = 256, .n = 4}};
  f.query.tuned = tunes;
  const auto exact_tune = ex::FusionQueryKey(f.query);
  tunes[0].all_rows = true;
  EXPECT_NE(ex::FusionQueryKey(f.query), exact_tune);
  f.query = q;
  const auto memory = f.operands[0].memory;
  f.operands[0].memory.reset();
  EXPECT_FALSE(ex::FusionQueryKey(f.query));
  f.operands[0].memory = memory;
  f.query.device_class = "other";
  misses();
  f.query = q;
  f.query.device_model = "other";
  misses();
  f.query = q;
  ++f.query.arch;
  misses();
  f.query = q;
  ++f.query.multiprocessors;
  misses();
  f.query = q;
  ++f.query.features;
  misses();
  f.query = q;
  ++f.query.rows;
  misses();
  f.query = q;
  ++f.query.k;
  misses();
  f.query = q;
  ++f.query.n;
  misses();
  f.query = q;
  f.query.capturable = !q.capturable;
  misses();
  f.query = q;
  f.query.row_invariant = !q.row_invariant;
  misses();
  f.query = q;
  f.query.state_span = !q.state_span;
  misses();
  f.query = q;
  f.params[0] ^= std::byte{1};
  misses();
  f.params[0] ^= std::byte{1};
  f.query.policy.allow_lossy = !q.policy.allow_lossy;
  misses();
  f.query = q;
  f.query.automatic_fusion = !q.automatic_fusion;
  EXPECT_EQ(ex::FusionQueryKey(f.query), baseline);
  std::swap(f.operands[0], f.operands[2]);
  EXPECT_EQ(ex::FusionQueryKey(f.query), baseline);
  f.operands[0].operand = f.operands[1].operand;
  EXPECT_FALSE(ex::FusionQueryKey(f.query));
}

TEST(FusionRanksTest, SourceChangeMakesEvidenceStaleAndDataChangesOnlyRegistryIdentity) {
  Fixture f;
  const auto before = f.Ranked();
  ++f.record.fused_ns;
  const auto changed = f.Ranked();
  EXPECT_NE(before.Digest(), changed.Digest());
  EXPECT_NE(before.RankDigest(), changed.RankDigest());
  EXPECT_EQ(before.identity(0), changed.identity(0));
  for (std::size_t i = 0; i < f.declarations.size(); ++i) {
    f.declarations[i].revision = "changed";
    const auto stale = f.Ranked();
    EXPECT_EQ(ex::FusionPreferenceFor(stale, f.query, "test.fusion"), ex::FusionPreference::kStale);
    EXPECT_EQ(ex::Candidates(stale, f.query).size(), 1U);
    f.declarations[i].revision = "test-r1";
  }
}

TEST(FusionRanksTest, CurrentAlternativeRankAndEligibilityChangesInvalidateComparators) {
  Fixture f;
  const auto original_norm = f.registry.identity(1);
  auto fast = Declaration("test.norm.fast", ir::Primitive(ir::Op::kRmsNorm));
  fast.capability->priority = 1;
  f.declarations.push_back(fast);
  const auto changed = f.Ranked();
  EXPECT_EQ(changed.identity(1), original_norm);
  EXPECT_EQ(ex::FusionPreferenceFor(changed, f.query, "test.fusion"), ex::FusionPreference::kStale);
  const pl::Policy policy;
  const pl::SelectionView fast_view(changed, kDevice, policy, false, false);
  fast_view.Inspect(f.graph, f.graph.nodes().back(), [&](const ex::CapabilityQuery& q) {
    ASSERT_EQ(q.terminal_decomposition.size(), 2U);
    EXPECT_EQ(changed.at(q.terminal_decomposition[0]).name, "test.norm.fast");
  });
  // Ordinary device-rank data participates even when this particular row is
  // unrelated: the environment is conservatively complete, not a partial cache.
  const ex::DeviceMeasuredRank rank{.device_class = "test.device",
                                    .device_model = {},
                                    .op = ir::Primitive(ir::Op::kMul),
                                    .implementation = "test.mul",
                                    .rank = 1};
  const auto ranked =
      ex::Registry::Create(Declarations(), {.records = {rank}, .fusions = {f.record}});
  ASSERT_TRUE(ranked);
  EXPECT_EQ(ranked->identity(1), original_norm);
  EXPECT_EQ(ex::FusionPreferenceFor(*ranked, f.query, "test.fusion"), ex::FusionPreference::kStale);
  // New alternative still exists, but its changed eligibility removes it.
  f.declarations.back().capability->shape.min_rows = 5;
  const auto ineligible = f.Ranked();
  const pl::SelectionView eligible_view(ineligible, kDevice, policy, false, false);
  eligible_view.Inspect(f.graph, f.graph.nodes().back(), [&](const ex::CapabilityQuery& q) {
    ASSERT_EQ(q.terminal_decomposition.size(), 2U);
    EXPECT_EQ(ineligible.at(q.terminal_decomposition[0]).name, "test.norm");
  });
  EXPECT_EQ(ex::FusionPreferenceFor(ineligible, f.query, "test.fusion"),
            ex::FusionPreference::kStale);
  // Explicit primitive ranking changes the actual ordered chain and query.
  auto ranked_policy = policy;
  ranked_policy.ranks.push_back(
      {.device_class = "test.device", .implementation = "test.norm", .rank = 0});
  const pl::SelectionView ranked_view(changed, kDevice, ranked_policy, false, false);
  ranked_view.Inspect(f.graph, f.graph.nodes().back(), [&](const ex::CapabilityQuery& q) {
    ASSERT_EQ(q.terminal_decomposition.size(), 2U);
    EXPECT_EQ(changed.at(q.terminal_decomposition[0]).name, "test.norm");
    EXPECT_NE(ex::FusionQueryKey(q), f.record.query);
  });
}

TEST(FusionRanksTest, SemanticOptionEligibilityAndRankOrderArePartOfTheQuery) {
  Fixture f;
  auto fast = Declaration("test.norm.fast", ir::Primitive(ir::Op::kRmsNorm));
  fast.variant = "requires rms_norm_mul option";
  fast.capability->priority = 1;
  fast.capability->accepts = [](const ex::CapabilityQuery& q) {
    return q.policy.options && std::ranges::contains(*q.policy.options, "rms_norm_mul");
  };
  f.declarations.push_back(fast);
  const auto registry = ex::Registry::Create(f.declarations).value();
  const pl::Policy enabled;
  std::optional<llmp::base::Sha256Digest> baseline;
  const pl::SelectionView view(registry, kDevice, enabled, false, false);
  view.Inspect(f.graph, f.graph.nodes().back(), [&](const ex::CapabilityQuery& q) {
    baseline = ex::FusionQueryKey(q);
    ASSERT_EQ(q.terminal_decomposition.size(), 2U);
    EXPECT_EQ(registry.at(q.terminal_decomposition[0]).name, "test.norm.fast");
  });
  auto disabled = enabled;
  disabled.disabled = {"rms_norm_mul"};
  const pl::SelectionView changed(registry, kDevice, disabled, false, false);
  changed.Inspect(f.graph, f.graph.nodes().back(), [&](const ex::CapabilityQuery& q) {
    EXPECT_NE(ex::FusionQueryKey(q), baseline);
    ASSERT_EQ(q.terminal_decomposition.size(), 2U);
    EXPECT_EQ(registry.at(q.terminal_decomposition[0]).name, "test.norm");
  });
  std::array ranks{ex::MeasuredRank{.implementation = "test.norm", .rank = 1},
                   ex::MeasuredRank{.implementation = "test.norm", .rank = 2}};
  f.query.measured = ranks;
  const auto ordered = ex::FusionQueryKey(f.query);
  std::swap(ranks[0], ranks[1]);
  EXPECT_NE(ex::FusionQueryKey(f.query), ordered);
}

TEST(FusionRanksTest, ComparisonBytesCannotAffectTerminalSelectionAndProofIsRequired) {
  Fixture f;
  auto other = f.record;
  other.query[0] ^= 1;
  other.fused_ns = 1;
  const auto extra = ex::Registry::Create(f.declarations, {.fusions = {f.record, other}}).value();
  EXPECT_EQ(extra.SelectionDigest(), f.registry.SelectionDigest());
  EXPECT_EQ(ex::FusionPreferenceFor(extra, f.query, "test.fusion"),
            ex::FusionPreference::kDecomposition);
  f.query.terminal_decomposition = {};
  EXPECT_EQ(ex::FusionPreferenceFor(extra, f.query, "test.fusion"),
            ex::FusionPreference::kUnmeasured);
  f.query.terminal_decomposition = f.terminal;
  std::swap(f.terminal[0], f.terminal[1]);
  EXPECT_EQ(ex::FusionPreferenceFor(extra, f.query, "test.fusion"), ex::FusionPreference::kStale);
}

TEST(FusionRanksTest, GenericTerminalContractRejectsUncertifiedGroupedAndExpandingChildren) {
  Fixture f;
  const pl::Policy policy;
  for (unsigned mutation = 0; mutation != 5; ++mutation) {
    auto declarations = Declarations();
    auto& cap = *declarations[1].capability;
    if (mutation == 0) cap.standalone_comparator = false;
    if (mutation == 1) cap.group = "test.group";
    if (mutation == 2) cap.batch = 2;
    if (mutation == 3) cap.expansion = ex::Expansion::kDenseF32Accumulation;
    if (mutation == 4) cap.pitched_output = true;
    const auto registry = ex::Registry::Create(std::move(declarations)).value();
    const pl::SelectionView view(registry, kDevice, policy, false, false);
    view.Inspect(f.graph, f.graph.nodes().back(), [&](const ex::CapabilityQuery& q) {
      EXPECT_TRUE(q.terminal_decomposition.empty());
    });
  }
  // A different fusion also qualifies by structure, without an op whitelist.
  ir::Builder b;
  const auto x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({256, 4})));
  const auto w = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({256})));
  const auto r = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({256, 4})));
  b.Output(
      b.Fused(ir::FusedKind::kRmsNormMulAdd, std::array{x, w, r}, ir::NormParams{.eps = 1e-6F})[0]);
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph);
  auto declarations = Declarations();
  declarations.push_back(Declaration("test.add", ir::Primitive(ir::Op::kAdd)));
  const auto registry = ex::Registry::Create(declarations).value();
  const pl::SelectionView view(registry, kDevice, policy, false, false);
  view.Inspect(*graph, graph->nodes().back(), [&](const ex::CapabilityQuery& q) {
    ASSERT_EQ(q.terminal_decomposition.size(), 3U);
    EXPECT_EQ(registry.at(q.terminal_decomposition[2]).name, "test.add");
  });
}

TEST(FusionRanksTest, ImmediateCompositeChildrenRemainUnmeasuredWithoutRecursivePreference) {
  ir::Builder b;
  const auto down = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({8, 2, 3})));
  const auto ids = b.Input(ir::Dense(Encoding::kI32, ir::ShapeOf({2, 3})));
  const auto weights = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({2, 3})));
  const auto shared = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({8, 3})));
  b.Output(b.Fused(ir::FusedKind::kMoeCombine, std::array{down, ids, weights, shared})[0]);
  const auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph);
  auto declarations = Declarations();
  declarations.push_back(Declaration("test.combine", ir::Fused(ir::FusedKind::kMoeCombine)));
  declarations.push_back(Declaration("test.reduce", ir::Primitive(ir::Op::kExpertReduce)));
  declarations.push_back(Declaration("test.add", ir::Primitive(ir::Op::kAdd)));
  const auto registry = ex::Registry::Create(declarations).value();
  const pl::Policy policy;
  const pl::SelectionView view(registry, kDevice, policy, false, false);
  view.Inspect(*graph, graph->nodes().back(), [&](const ex::CapabilityQuery& q) {
    EXPECT_TRUE(q.terminal_decomposition.empty());
  });
}

TEST(FusionRanksTest, MalformedOrAmbiguousComparisonsAreRejected) {
  Fixture f;
  const auto baseline = f.record;
  const auto rejects = [&] {
    EXPECT_FALSE(ex::Registry::Create(f.declarations, {.fusions = {f.record}}));
    f.record = baseline;
  };
  f.record.fused_ns = 0;
  rejects();
  f.record.decomposition_ns = 0;
  rejects();
  f.record.samples = 0;
  rejects();
  f.record.query = {};
  rejects();
  f.record.selection_environment = {};
  rejects();
  f.record.fusion.identity = {};
  rejects();
  f.record.decomposition.clear();
  rejects();
  f.record.decomposition[0].name = "unknown";
  rejects();
  f.record.decomposition[0] = f.record.fusion;
  rejects();
  f.record.provenance.clear();
  rejects();
  EXPECT_FALSE(ex::Registry::Create(f.declarations, {.fusions = {f.record, f.record}}));
}

TEST(FusionRanksTest, RepeatedSourceRootCannotReuseDistinctRootEvidence) {
  Fixture f;
  const auto graph_of = [](bool repeated) {
    ir::Builder b;
    const auto format = ir::Dense(Encoding::kF32, ir::ShapeOf({256, 4}));
    const auto x = b.Input(format);
    const auto w = repeated ? x : b.Input(format);
    b.Output(
        b.Fused(ir::FusedKind::kRmsNormMul, std::array{x, w}, ir::NormParams{.eps = 1e-6F})[0]);
    return std::move(b).Finish().value();
  };
  const auto distinct = graph_of(false), repeated = graph_of(true);
  const pl::Policy policy;
  const pl::SelectionView view(f.registry, kDevice, policy, false, false);
  view.Inspect(distinct, distinct.nodes().back(), [&](const ex::CapabilityQuery& q) {
    ASSERT_EQ(q.terminal_decomposition.size(), 2U);
    f.record.query = *ex::FusionQueryKey(q);
  });
  const auto ranked = f.Ranked();
  const pl::SelectionView ranked_view(ranked, kDevice, policy, false, false);
  ranked_view.Inspect(repeated, repeated.nodes().back(), [&](const ex::CapabilityQuery& q) {
    EXPECT_EQ(ex::FusionQueryKey(q), f.record.query);  // Same formats, different alias partition.
    EXPECT_TRUE(q.terminal_decomposition.empty());
    EXPECT_EQ(ex::FusionPreferenceFor(ranked, q, "test.fusion"), ex::FusionPreference::kUnmeasured);
  });
  const auto distinct_plan = pl::Plan(distinct.Clone(), ranked, kDevice, policy);
  ASSERT_TRUE(distinct_plan);
  ASSERT_EQ(distinct_plan->steps.size(), 2U);
  EXPECT_EQ(ranked.at(distinct_plan->steps[0].implementation).name, "test.norm");
  EXPECT_EQ(ranked.at(distinct_plan->steps[1].implementation).name, "test.mul");
  const auto repeated_plan = pl::Plan(repeated.Clone(), ranked, kDevice, policy);
  ASSERT_TRUE(repeated_plan);
  ASSERT_EQ(repeated_plan->steps.size(), 1U);
  EXPECT_EQ(ranked.at(repeated_plan->steps[0].implementation).name, "test.fusion");
}

TEST(FusionRanksTest, MeasuredLoserUsesItsOwnDecompositionAndOtherShapesStayUnmeasured) {
  Fixture f;
  // Query from the actual planner, including its enabled kernel options.
  const pl::Policy policy;
  pl::SelectionView view(f.registry, kDevice, policy, false, false);
  view.Inspect(f.graph, f.graph.nodes().back(),
               [&](const ex::CapabilityQuery& q) { f.record.query = *ex::FusionQueryKey(q); });
  const auto ranked = f.Ranked();
  const pl::SelectionView ranked_view(ranked, kDevice, policy, false, false);
  EXPECT_EQ(ranked_view.Candidates(f.graph, f.graph.nodes().back()).size(), 1U);
  EXPECT_TRUE(
      ranked_view.FinalCandidates(f.graph, f.graph.nodes().back(), f.graph.nodes().back().key)
          .empty());
  const auto planned = pl::Plan(f.graph.Clone(), ranked, kDevice, policy);
  ASSERT_TRUE(planned) << (planned ? "" : planned.error().detail);
  ASSERT_EQ(planned->steps.size(), 2U);
  EXPECT_EQ(ranked.at(planned->steps[0].implementation).name, "test.norm");
  EXPECT_EQ(ranked.at(planned->steps[1].implementation).name, "test.mul");
  const auto ordinary = pl::Plan(f.graph.Clone(), f.registry, kDevice, policy);
  ASSERT_TRUE(ordinary);
  ASSERT_EQ(ordinary->steps.size(), 1U);
  EXPECT_EQ(f.registry.at(ordinary->steps[0].implementation).name, "test.fusion");
}
}  // namespace
