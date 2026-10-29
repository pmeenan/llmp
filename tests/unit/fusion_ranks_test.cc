// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "execution/fusion_ranks.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ir/builder.h"
#include "kernels/ggml/capabilities.h"
#include "kernels/ggml/measured_ranks.h"
#include "planner/plan.h"
#include "planner/select.h"

namespace {
namespace ex = llmp::execution;
namespace ir = llmp::ir;
namespace pl = llmp::planner;
using llmp::format::Encoding;
using llmp::format::Layout;

ex::Implementation Declaration(std::string name, ir::OpKey op, std::uint32_t version = 1) {
  return {.name = std::move(name),
          .operation = op,
          .source = "synthetic",
          .revision = "test-r1",
          .build = "test",
          .variant = "test",
          .version = version,
          .capability = ex::Capability{.op = op}};
}
std::vector<ex::Implementation> Declarations(std::uint32_t fusion_version = 1) {
  return {Declaration("test.fusion", ir::Fused(ir::FusedKind::kRmsNormMul), fusion_version),
          Declaration("test.norm", ir::Primitive(ir::Op::kRmsNorm)),
          Declaration("test.mul", ir::Primitive(ir::Op::kMul))};
}
ir::Graph Graph(std::uint64_t width = 256, std::uint64_t rows = 4) {
  ir::Builder b;
  const auto x = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width, rows})));
  const auto w = b.Input(ir::Dense(Encoding::kF32, ir::ShapeOf({width})));
  const std::array sources{x, w};
  b.Output(b.Fused(ir::FusedKind::kRmsNormMul, sources, ir::NormParams{.eps = 1e-6F})[0]);
  return std::move(b).Finish().value();
}
const pl::DeviceFacts kDevice{.device_class = "test.device",
                              .arch = 10,
                              .device_model = "test processor",
                              .multiprocessors = 8};

// All timings here are synthetic unit-test values, never installed backend data.
ex::FusionRank Row(std::uint64_t fused_ns, std::uint64_t decomposition_ns) {
  return {.device_class = "test.device",
          .op = ir::Fused(ir::FusedKind::kRmsNormMul),
          .implementation = "test.fusion",
          .version = 1,
          .encoding = Encoding::kF32,
          .layout = Layout::kDense,
          .rows_min = 1,
          .rows_max = 8,
          .width_min = 128,
          .width_max = 512,
          .fused_ns = fused_ns,
          .decomposition_ns = decomposition_ns,
          .samples = 9,
          .provenance = "synthetic test only"};
}

// The query the planner asks for the graph's fused node.
struct Fixture {
  ir::Graph graph;
  ex::Registry registry;
  std::array<ex::OperandFormat, 3> operands{};
  ir::ParamBytes params{};
  ex::CapabilityQuery query{};
  explicit Fixture(ir::Graph g = Graph(), std::vector<ex::FusionRank> rows = {},
                   std::uint32_t fusion_version = 1)
      : graph(std::move(g)),
        registry(ex::Registry::Create(Declarations(fusion_version), {.fusions = std::move(rows)})
                     .value()) {
    const pl::Policy policy;
    const pl::SelectionView view(registry, kDevice, policy, false, false);
    view.Inspect(graph, graph.nodes().back(), [&](const ex::CapabilityQuery& q) {
      std::ranges::copy(q.operands, operands.begin());
      std::ranges::copy(q.params, params.begin());
      query = q;
      query.operands = std::span(operands).first(q.operands.size());
      query.params = params;
      query.policy.options = {};
    });
  }
};

TEST(FusionRanksTest, LatencyBandUsesTheDecompositionReferenceInBothDirections) {
  struct Case {
    std::uint64_t fused;
    ex::FusionPreference expected;
  };
  for (const auto& c : std::array{
           Case{949, ex::FusionPreference::kFusion}, Case{950, ex::FusionPreference::kNeutral},
           Case{1005, ex::FusionPreference::kNeutral}, Case{1050, ex::FusionPreference::kNeutral},
           Case{1051, ex::FusionPreference::kDecomposition}}) {
    SCOPED_TRACE(c.fused);
    EXPECT_EQ(ex::ClassifyFusionLatency(c.fused, 1000), c.expected);
    const Fixture f(Graph(), {Row(c.fused, 1000)});
    EXPECT_EQ(ex::FusionPreferenceFor(f.registry, f.query, "test.fusion"), c.expected);
  }
  constexpr auto max = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(ex::ClassifyFusionLatency(max - max / 20, max), ex::FusionPreference::kNeutral);
  EXPECT_EQ(ex::ClassifyFusionLatency(max - max / 20 - 1, max), ex::FusionPreference::kFusion);
  EXPECT_EQ(ex::ClassifyFusionLatency(max, 1), ex::FusionPreference::kDecomposition);
  EXPECT_EQ(ex::ClassifyFusionLatency(0, 1000), ex::FusionPreference::kUnmeasured);
  EXPECT_EQ(ex::ClassifyFusionLatency(1000, 0), ex::FusionPreference::kUnmeasured);
}

TEST(FusionRanksTest, ARowDecidesOnlyInsideItsClass) {
  const Fixture unranked;
  EXPECT_EQ(ex::FusionPreferenceFor(unranked.registry, unranked.query, "test.fusion"),
            ex::FusionPreference::kUnmeasured);
  const Fixture f(Graph(), {Row(200, 100)});
  EXPECT_EQ(f.query.rows, 4U);
  EXPECT_EQ(f.query.k, 256U);
  EXPECT_EQ(ex::FusionPreferenceFor(f.registry, f.query, "test.fusion"),
            ex::FusionPreference::kDecomposition);
  // Another implementation, device class, encoding, layout or operation:
  // unmeasured.
  EXPECT_EQ(ex::FusionPreferenceFor(f.registry, f.query, "test.other"),
            ex::FusionPreference::kUnmeasured);
  const auto misses = [&](auto mutate) {
    auto query = f.query;
    auto operands = f.operands;
    query.operands = std::span(operands).first(f.query.operands.size());
    mutate(query, operands);
    EXPECT_EQ(ex::FusionPreferenceFor(f.registry, query, "test.fusion"),
              ex::FusionPreference::kUnmeasured);
  };
  misses([](auto& q, auto&) { q.device_class = "other.device"; });
  misses([](auto& q, auto&) { q.op = ir::Fused(ir::FusedKind::kRmsNormMulAdd); });
  misses([](auto&, auto& o) { o[0].encoding = Encoding::kBF16; });
  misses([](auto&, auto& o) { o[0].layout = Layout::kSplit; });
  // Outside the row and width bands: never extrapolated.
  misses([](auto& q, auto&) { q.rows = 9; });
  misses([](auto& q, auto&) { q.k = 127; });
  misses([](auto& q, auto&) { q.k = 513; });
  // A row without an output band holds every n, a norm's none (0) included.
  for (const std::uint64_t n : {0U, 1U, 1U << 20U}) {
    auto query = f.query;
    query.n = n;
    EXPECT_EQ(ex::FusionPreferenceFor(f.registry, query, "test.fusion"),
              ex::FusionPreference::kDecomposition);
  }
  // The bands' ends are inclusive.
  for (const auto& [rows, k] : std::array{std::pair{1U, 128U}, std::pair{8U, 512U}}) {
    auto query = f.query;
    query.rows = rows;
    query.k = k;
    EXPECT_EQ(ex::FusionPreferenceFor(f.registry, query, "test.fusion"),
              ex::FusionPreference::kDecomposition);
  }
}

// Rows of one class split by the query's outputs (a product's n): the first
// row whose bands hold it decides, past every band nothing does.
TEST(FusionRanksTest, AnOutputBandSplitsOneClassByTheQuerysN) {
  auto wide = Row(200, 100);
  wide.outputs_min = 8192;
  auto narrow = Row(100, 200);
  narrow.outputs_max = 4096;
  const Fixture f(Graph(), {wide, narrow});
  const auto at = [&](std::uint64_t n) {
    auto query = f.query;
    query.n = n;
    return ex::FusionPreferenceFor(f.registry, query, "test.fusion");
  };
  EXPECT_EQ(at(8192), ex::FusionPreference::kDecomposition);
  EXPECT_EQ(at(21504), ex::FusionPreference::kDecomposition);
  EXPECT_EQ(at(4096), ex::FusionPreference::kFusion);
  EXPECT_EQ(at(0), ex::FusionPreference::kFusion);
  EXPECT_EQ(at(4097), ex::FusionPreference::kUnmeasured);
  EXPECT_EQ(at(8191), ex::FusionPreference::kUnmeasured);
}

TEST(FusionRanksTest, ARowOfAnotherKernelVersionIsStaleAndDecidesNothing) {
  const Fixture f(Graph(), {Row(200, 100)}, /*fusion_version=*/2);
  EXPECT_EQ(ex::FusionPreferenceFor(f.registry, f.query, "test.fusion"),
            ex::FusionPreference::kStale);
  const auto candidates = ex::Candidates(f.registry, f.query);
  ASSERT_EQ(candidates.size(), 1U);
  EXPECT_EQ(f.registry.at(candidates.front()).name, "test.fusion");
  // The version is part of the implementation's identity past the first.
  EXPECT_NE(ex::IdentityOf(Declarations(2).front()), ex::IdentityOf(Declarations(1).front()));
  EXPECT_EQ(ex::IdentityOf(Declarations(1).front()),
            ex::IdentityOf(Declaration("test.fusion", ir::Fused(ir::FusedKind::kRmsNormMul))));
}

TEST(FusionRanksTest, NeutralAndWinningKeepTheFusionALosingFusionIsExcluded) {
  for (const auto& [fused, kept] :
       std::array{std::pair{105U, true}, std::pair{50U, true}, std::pair{200U, false}}) {
    SCOPED_TRACE(fused);
    const Fixture f(Graph(), {Row(fused, 100)});
    EXPECT_EQ(ex::IsCandidate(f.registry, f.query, 0), kept);
    EXPECT_EQ(ex::HasCandidate(f.registry, f.query), kept);
    EXPECT_EQ(ex::BestCandidate(f.registry, f.query).has_value(), kept);
    EXPECT_EQ(ex::Candidates(f.registry, f.query).size(), kept ? 1U : 0U);
  }
}

TEST(FusionRanksTest, AMeasuredLoserPlansItsOwnDecompositionOtherShapesKeepTheFusion) {
  const pl::Policy policy;
  {
    const Fixture f(Graph(256, 4), {Row(200, 100)});
    const auto planned = pl::Plan(f.graph.Clone(), f.registry, kDevice, policy);
    ASSERT_TRUE(planned) << planned.error().detail;
    ASSERT_EQ(planned->steps.size(), 2U);
    EXPECT_EQ(f.registry.at(planned->steps[0].implementation).name, "test.norm");
    EXPECT_EQ(f.registry.at(planned->steps[1].implementation).name, "test.mul");
  }
  {
    // A width past the row's band keeps the fusion.
    const Fixture f(Graph(1024, 4), {Row(200, 100)});
    const auto planned = pl::Plan(f.graph.Clone(), f.registry, kDevice, policy);
    ASSERT_TRUE(planned) << planned.error().detail;
    ASSERT_EQ(planned->steps.size(), 1U);
    EXPECT_EQ(f.registry.at(planned->steps[0].implementation).name, "test.fusion");
  }
}

TEST(FusionRanksTest, MalformedRowsAreRefusedAndRowsChangeTheRankDigest) {
  const auto create = [](ex::FusionRank row) {
    return ex::Registry::Create(Declarations(), {.fusions = {std::move(row)}});
  };
  ASSERT_TRUE(create(Row(200, 100)));
  const auto refused = [&](auto mutate) {
    auto row = Row(200, 100);
    mutate(row);
    EXPECT_FALSE(create(std::move(row)));
  };
  refused([](auto& r) { r.device_class.clear(); });
  refused([](auto& r) { r.implementation.clear(); });
  refused([](auto& r) { r.op = ir::Primitive(ir::Op::kRmsNorm); });
  refused([](auto& r) { r.version = 0; });
  refused([](auto& r) { r.rows_min = 0; });
  refused([](auto& r) { r.rows_min = 9; });
  refused([](auto& r) { r.width_min = 600; });
  refused([](auto& r) {
    r.outputs_min = 9;
    r.outputs_max = 8;
  });
  refused([](auto& r) { r.fused_ns = 0; });
  refused([](auto& r) { r.decomposition_ns = 0; });
  refused([](auto& r) { r.samples = 0; });
  refused([](auto& r) { r.provenance.clear(); });
  EXPECT_FALSE(ex::Registry::Create({Declaration("test.zero", ir::Primitive(ir::Op::kMul), 0)}));
  const auto plain = ex::Registry::Create(Declarations()).value();
  const auto ranked = create(Row(200, 100)).value();
  auto other = Row(200, 100);
  other.width_max = 1024;
  const auto widened = create(other).value();
  auto banded = Row(200, 100);
  banded.outputs_min = 8192;
  const auto outputs = create(banded).value();
  EXPECT_NE(plain.RankDigest(), ranked.RankDigest());
  EXPECT_NE(ranked.RankDigest(), widened.RankDigest());
  EXPECT_NE(ranked.RankDigest(), outputs.RankDigest());
  EXPECT_NE(ranked.Digest(), widened.Digest());
}

// The rows of `data` that name no declared implementation, or one of
// another operation or kernel version.
std::vector<std::string> StaleRows(std::span<const ex::Implementation> declared,
                                   const ex::RankData& data) {
  std::vector<std::string> stale;
  const auto check = [&](std::string_view name, ir::OpKey op, std::uint32_t version) {
    const auto at = std::ranges::find(declared, name, &ex::Implementation::name);
    if (at == declared.end() || at->operation != op || at->version != version)
      stale.emplace_back(name);
  };
  for (const auto& r : data.records) check(r.implementation, r.op, r.version);
  for (const auto& r : data.fusions) check(r.implementation, r.op, r.version);
  return stale;
}

// The check itself: a row naming a missing, renamed, other-operation or
// other-version implementation is reported; a current row is not.
TEST(FusionRanksTest, TheStaleRowCheckReportsEveryKindOfStaleRow) {
  const auto declared = Declarations();
  EXPECT_TRUE(StaleRows(declared, {.fusions = {Row(200, 100)}}).empty());
  const auto stale = [&](auto mutate) {
    auto row = Row(200, 100);
    mutate(row);
    EXPECT_EQ(StaleRows(declared, {.fusions = {row}}).size(), 1U);
  };
  stale([](auto& r) { r.implementation = "test.removed"; });
  stale([](auto& r) { r.op = ir::Fused(ir::FusedKind::kRmsNormMulAdd); });
  stale([](auto& r) { r.version = 2; });
  EXPECT_EQ(StaleRows(Declarations(/*fusion_version=*/2), {.fusions = {Row(200, 100)}}).size(), 1U);
  ex::DeviceMeasuredRank record;
  record.device_class = "test.device";
  record.op = ir::Primitive(ir::Op::kMul);
  record.implementation = "test.mul";
  EXPECT_TRUE(StaleRows(declared, {.records = {record}}).empty());
  record.version = 3;
  EXPECT_EQ(StaleRows(declared, {.records = {record}}).size(), 1U);
}

// The shipped table: every GGML rank and fusion row names an implementation
// the backend declares, of the row's operation, at the row's kernel version.
// A kernel author who bumps Implementation::version, renames or removes an
// implementation fails here until the row is re-measured or retired
// (docs/experiments/fusion-ranks/README.md).
TEST(FusionRanksTest, NoShippedRowNamesAMissingOrStaleImplementation) {
  const auto declared = llmp::kernels::ggml::PlanningImplementations();
  const auto data = llmp::kernels::ggml::MeasuredRanks();
  EXPECT_EQ(StaleRows(declared, data), std::vector<std::string>{});
  for (const auto& r : data.fusions) {
    SCOPED_TRACE(r.implementation);
    EXPECT_TRUE(ex::ValidFusionRank(r));
  }
  EXPECT_TRUE(ex::Registry::Create(declared, data));
}

}  // namespace
