// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "execution/fusion_ranks.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "ir/builder.h"
#include "ir/decompose.h"
#include "planner/plan.h"
#include "planner/select.h"

namespace {
namespace ex = llmp::execution;
namespace ir = llmp::ir;
namespace pl = llmp::planner;
namespace base = llmp::base;
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
base::Sha256Digest TestDigest(std::uint8_t value) {
  base::Sha256Digest digest{};
  digest.fill(value);
  return digest;
}
ex::RecipeComparison RecipeRecord(const base::Sha256Digest& selection) {
  return {.key = {.version = 1,
                  .region_canonicalizer = 2,
                  .discovery_operation = ir::Primitive(ir::Op::kRmsNorm),
                  .mode = ex::RecipeRunMode::kWarm,
                  .subject = TestDigest(1),
                  .boundary = TestDigest(2),
                  .policy = TestDigest(3),
                  .device = TestDigest(4),
                  .selection_environment = selection,
                  .preparation_domain = TestDigest(5),
                  .incumbent = {.planned_region = TestDigest(6),
                                .physical_recipe = TestDigest(7),
                                .backend_recipe = TestDigest(8),
                                .backend_build = TestDigest(9),
                                .backend_selection = TestDigest(10)},
                  .alternative = {.planned_region = TestDigest(11),
                                  .physical_recipe = TestDigest(12),
                                  .backend_recipe = TestDigest(13),
                                  .backend_build = TestDigest(14),
                                  .backend_selection = TestDigest(15)}},
          .incumbent_ns = 100,
          .alternative_ns = 80,
          .samples = 7,
          .protocol = TestDigest(16),
          .receipt = TestDigest(17),
          .provenance = "synthetic recipe evidence",
          .shape_class = "synthetic exact region"};
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

void LegacyField(llmp::base::Sha256& hash, std::string_view value) {
  hash.Update(std::format("{}:", value.size()));
  hash.Update(value);
}

std::optional<llmp::base::Sha256Digest> LegacyFusionQueryKey(const ex::CapabilityQuery& q) {
  if (q.op.op != ir::Op::kFused || q.op.fused == ir::FusedKind::kNone || q.device_class.empty() ||
      q.device_model.empty() || q.arch == 0 || q.multiprocessors == 0 || q.operands.empty())
    return {};
  std::vector<const ex::OperandFormat*> operands;
  for (const auto& o : q.operands) {
    if (o.stacked == 0 || !o.memory || !o.rank || *o.rank > llmp::format::kMaxDims ||
        std::ranges::contains(o.ne, 0U) ||
        static_cast<std::size_t>(o.encoding) >= llmp::format::Encodings().size() ||
        static_cast<std::size_t>(o.layout) >= llmp::format::Layouts().size())
      return {};
    operands.push_back(&o);
  }
  std::ranges::sort(operands, {}, [](const auto* o) { return o->operand; });
  if (std::ranges::adjacent_find(
          operands, [](const auto* a, const auto* b) { return a->operand == b->operand; }) !=
          operands.end() ||
      operands.front()->operand != 0 || operands.back()->operand < ex::kOutputOperand)
    return {};
  llmp::base::Sha256 hash;
  LegacyField(hash, "llmp-fusion-query-v2");
  LegacyField(hash, ir::OpKeyName(q.op));
  LegacyField(hash, q.device_class);
  LegacyField(hash, q.device_model);
  LegacyField(hash, std::format("{}:{}:{}:{}:{}:{}:{}:{}:{}", q.arch, q.features, q.multiprocessors,
                                q.rows, q.k, q.n, q.row_invariant, q.capturable, q.state_span));
  LegacyField(hash, std::format("{}:{}:{}", q.policy.max_decode_rows, q.policy.allow_lossy,
                                q.policy.vector_rows));
  LegacyField(hash, q.policy.options ? "policy-options-present" : "policy-options-absent");
  if (q.policy.options) {
    LegacyField(hash, std::format("{}", q.policy.options->size()));
    for (const auto value : *q.policy.options) LegacyField(hash, value);
  }
  LegacyField(hash, std::format("{}", q.preferred.size()));
  for (const auto& value : q.preferred) LegacyField(hash, value);
  LegacyField(hash, std::format("{}", q.measured.size()));
  for (const auto& r : q.measured) {
    LegacyField(hash, r.implementation);
    LegacyField(hash, std::format("{}", r.rank));
  }
  LegacyField(hash, std::format("{}", q.tuned.size()));
  for (const auto& t : q.tuned) {
    LegacyField(hash, t.implementation);
    LegacyField(hash, std::format("{}:{}:{}:{}:{}:{}", t.rows, t.k, t.n, t.weight_bits,
                                  t.output_encoding ? static_cast<int>(*t.output_encoding) : -1,
                                  t.all_rows));
  }
  LegacyField(hash, std::format("{}", q.params.size()));
  hash.Update(q.params);
  LegacyField(hash, std::format("{}", operands.size()));
  for (const auto* operand : operands) {
    const auto& o = *operand;
    LegacyField(hash,
                std::format("{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}", o.operand, o.stacked,
                            static_cast<unsigned>(o.encoding), static_cast<unsigned>(o.layout),
                            o.params.k_bits, static_cast<unsigned>(o.params.codebook),
                            o.params.tensor_scale, o.params.bits, o.params.group,
                            static_cast<unsigned>(o.params.scale), o.params.zero_point,
                            o.params.block_rows, o.params.block_cols));
    LegacyField(hash, std::format("{}", *o.rank));
    for (auto n : o.ne) LegacyField(hash, std::format("{}", n));
    for (auto n : o.memory->nb) LegacyField(hash, std::format("{}", n));
    LegacyField(hash, std::format("{}:{}:{}", o.memory->view_offset,
                                  static_cast<std::uint64_t>(o.memory->root_alignment),
                                  static_cast<bool>(o.memory->is_view)));
  }
  return hash.Finish();
}

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
  const auto ranked_candidates = ex::Candidates(ranked, f.query);
  EXPECT_TRUE(ranked_candidates.empty());
  EXPECT_FALSE(ex::BestCandidate(ranked, f.query));
  EXPECT_FALSE(ex::NextCandidate(ranked, f.query, std::nullopt));
  EXPECT_FALSE(ex::IsCandidate(ranked, f.query, 0));
  EXPECT_FALSE(ex::HasCandidate(ranked, f.query));
  f.record.fused_ns = 105;
  const auto neutral = f.Ranked();
  EXPECT_EQ(ex::FusionPreferenceFor(neutral, f.query, "test.fusion"),
            ex::FusionPreference::kNeutral);
  const auto neutral_candidates = ex::Candidates(neutral, f.query);
  ASSERT_EQ(neutral_candidates.size(), 1U);
  EXPECT_EQ(ex::BestCandidate(neutral, f.query), neutral_candidates.front());
  EXPECT_TRUE(ex::HasCandidate(neutral, f.query));
  f.record.fused_ns = 50;
  const auto winning = f.Ranked();
  EXPECT_EQ(ex::FusionPreferenceFor(winning, f.query, "test.fusion"),
            ex::FusionPreference::kFusion);
  const auto winning_candidates = ex::Candidates(winning, f.query);
  ASSERT_EQ(winning_candidates.size(), 1U);
  EXPECT_EQ(ex::BestCandidate(winning, f.query), winning_candidates.front());
  std::optional<std::size_t> cursor;
  for (const auto expected : winning_candidates) {
    cursor = ex::NextCandidate(winning, f.query, cursor);
    ASSERT_TRUE(cursor);
    EXPECT_EQ(*cursor, expected);
    EXPECT_TRUE(ex::IsCandidate(winning, f.query, *cursor));
  }
  EXPECT_FALSE(ex::NextCandidate(winning, f.query, cursor));
  EXPECT_TRUE(ex::HasCandidate(winning, f.query));
}

TEST(FusionRanksTest, AllocationFreeCursorMatchesFullRankedCandidateOrder) {
  Fixture f;
  auto declarations = f.declarations;
  declarations.push_back(Declaration("test.fusion.alt", f.query.op));
  const auto registry = ex::Registry::Create(declarations).value();
  const auto candidates = ex::Candidates(registry, f.query);
  ASSERT_EQ(candidates.size(), 2U);
  std::optional<std::size_t> cursor;
  for (const auto expected : candidates) {
    cursor = ex::NextCandidate(registry, f.query, cursor);
    ASSERT_TRUE(cursor);
    EXPECT_EQ(*cursor, expected);
    EXPECT_TRUE(ex::IsCandidate(registry, f.query, expected));
  }
  EXPECT_FALSE(ex::NextCandidate(registry, f.query, cursor));
}

TEST(FusionRanksTest, ComparisonModeBypassesOnlyFusionPreferences) {
  Fixture f;
  auto ranked = f.Ranked();
  EXPECT_TRUE(ex::Candidates(ranked, f.query).empty());
  EXPECT_EQ(ex::FusionPreferenceFor(ranked, f.query, "test.fusion"),
            ex::FusionPreference::kDecomposition);

  std::array<ex::MeasuredRank, 1> measured{{{"test.fusion", 0}}};
  auto comparison = f.query;
  comparison.fusion_comparisons = false;
  comparison.measured = measured;
  EXPECT_TRUE(ex::FusionPreferences(ranked, comparison).empty());
  EXPECT_EQ(ex::FusionPreferenceFor(ranked, comparison, "test.fusion"),
            ex::FusionPreference::kUnmeasured);
  const auto candidates = ex::Candidates(ranked, comparison);
  ASSERT_EQ(candidates.size(), 1U);
  EXPECT_EQ(ex::BestCandidate(ranked, comparison), candidates.front());
  EXPECT_EQ(ranked.at(candidates.front()).name, "test.fusion");
  EXPECT_TRUE(ex::HasCandidate(ranked, comparison));

  // Default queries still honor the same fusion row. Disabling that row does
  // not erase normal measured rank data or alter the default product policy.
  EXPECT_TRUE(ex::Candidates(ranked, f.query).empty());
  EXPECT_FALSE(ex::HasCandidate(ranked, f.query));
  EXPECT_EQ(ex::Candidates(f.registry, f.query).size(), 1U);
  EXPECT_TRUE(ex::HasCandidate(f.registry, f.query));
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

TEST(FusionRanksTest, AllocationFreeKeyIsByteIdenticalToLegacyFormatter) {
  Fixture f;
  auto query = f.query;
  std::string long_device(4096, 'd');
  std::string long_model(3072, 'm');
  std::string long_option(2048, 'o');
  std::string long_preferred(1536, 'p');
  std::string long_name(1024, 'n');
  const std::array<std::string_view, 2> options{long_option, "unknown.option"};
  const std::vector<std::string> preferred{long_preferred, "unknown.preference"};
  const std::array measured{ex::MeasuredRank{long_name, std::numeric_limits<std::uint32_t>::max()}};
  const std::array tuned{ex::TunedShape{.implementation = long_name,
                                        .rows = std::numeric_limits<std::uint64_t>::max(),
                                        .k = std::numeric_limits<std::uint64_t>::max(),
                                        .n = std::numeric_limits<std::uint64_t>::max(),
                                        .output_encoding = {},
                                        .weight_bits = std::numeric_limits<std::uint8_t>::max(),
                                        .all_rows = false},
                         ex::TunedShape{.implementation = "unknown.tuned",
                                        .rows = std::numeric_limits<std::uint64_t>::max(),
                                        .k = 1,
                                        .n = std::numeric_limits<std::uint64_t>::max(),
                                        .output_encoding = Encoding::kBF16,
                                        .weight_bits = 1,
                                        .all_rows = true}};
  query.device_class = long_device;
  query.device_model = long_model;
  query.arch = std::numeric_limits<std::uint32_t>::max();
  query.features = std::numeric_limits<std::uint32_t>::max();
  query.multiprocessors = std::numeric_limits<std::uint32_t>::max();
  query.rows = std::numeric_limits<std::uint64_t>::max();
  query.k = std::numeric_limits<std::uint64_t>::max();
  query.n = std::numeric_limits<std::uint64_t>::max();
  query.row_invariant = true;
  query.capturable = false;
  query.state_span = true;
  query.policy.max_decode_rows = std::numeric_limits<std::uint64_t>::max();
  query.policy.allow_lossy = true;
  query.policy.vector_rows = std::numeric_limits<std::uint64_t>::max();
  query.policy.options = options;
  query.preferred = preferred;
  query.measured = measured;
  query.tuned = tuned;
  auto operands = f.operands;
  operands[0].stacked = std::numeric_limits<std::uint8_t>::max();
  operands[0].params.k_bits = std::numeric_limits<std::uint8_t>::max();
  operands[0].params.group = std::numeric_limits<std::uint32_t>::max();
  operands[0].params.tensor_scale = true;
  operands[0].params.zero_point = true;
  operands[0].params.block_rows = std::numeric_limits<std::uint32_t>::max();
  operands[0].params.block_cols = std::numeric_limits<std::uint32_t>::max();
  operands[0].ne.fill(std::numeric_limits<std::uint64_t>::max());
  operands[0].memory->nb.fill(std::numeric_limits<std::uint64_t>::max());
  operands[0].memory->view_offset = std::numeric_limits<std::uint64_t>::max();
  operands[0].memory->root_alignment = (std::uint64_t{1} << 63) - 1;
  operands[0].memory->is_view = true;
  query.operands = operands;

  const auto expect_same = [&] {
    EXPECT_EQ(ex::FusionQueryKey(query), LegacyFusionQueryKey(query));
  };
  expect_same();  // Includes max unsigned values, both boolean spellings, signed -1 and long
                  // fields.
  std::ranges::reverse(operands);
  query.operands = operands;
  expect_same();  // The historical key sorts operands by their byte-sized namespace.

  auto duplicate = operands;
  duplicate[2].operand = duplicate[1].operand;
  query.operands = duplicate;
  EXPECT_EQ(ex::FusionQueryKey(query), LegacyFusionQueryKey(query));
  EXPECT_FALSE(ex::FusionQueryKey(query));
  duplicate = operands;
  duplicate[2].operand = 3;  // remove source zero while retaining the output slot
  query.operands = duplicate;
  EXPECT_EQ(ex::FusionQueryKey(query), LegacyFusionQueryKey(query));
  EXPECT_FALSE(ex::FusionQueryKey(query));
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
    ++operand.params.block_rows;
    misses();
    operand = original;
    ++operand.params.block_cols;
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
  f.operands[2].operand = 3;  // Leave source 0 absent while retaining the output slot.
  EXPECT_FALSE(ex::FusionQueryKey(f.query));
  f.operands[2].operand = f.operands[1].operand;
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
    EXPECT_EQ(ex::HasCandidate(stale, f.query), !ex::Candidates(stale, f.query).empty());
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
  const auto before = f.Ranked();
  auto changed_record = f.record;
  ++changed_record.fused_ns;
  const auto changed = ex::Registry::Create(f.declarations, {.fusions = {changed_record}}).value();
  EXPECT_NE(before.Digest(), changed.Digest());
  EXPECT_NE(before.RankDigest(), changed.RankDigest());
  EXPECT_EQ(before.SelectionDigest(), changed.SelectionDigest());

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

TEST(FusionRanksTest, RecipeKeyHashesEveryExactFieldAndPreservesOrderedArms) {
  Fixture f;
  auto comparison = RecipeRecord(f.registry.SelectionDigest());
  const auto original = comparison.key;
  const auto digest = ex::RecipeComparisonKeyDigest(original);
  const auto flip = [](base::Sha256Digest& value) { value[0] ^= 1; };
  const auto changed = [&](auto mutation) {
    auto key = original;
    mutation(key);
    EXPECT_NE(ex::RecipeComparisonKeyDigest(key), digest);
  };
  changed([](auto& key) { ++key.version; });
  changed([](auto& key) { ++key.region_canonicalizer; });
  changed([](auto& key) { key.discovery_operation = ir::Primitive(ir::Op::kMul); });
  changed([](auto& key) { key.mode = ex::RecipeRunMode::kFull; });
  changed([&](auto& key) { flip(key.subject); });
  changed([&](auto& key) { flip(key.boundary); });
  changed([&](auto& key) { flip(key.policy); });
  changed([&](auto& key) { flip(key.device); });
  changed([&](auto& key) { flip(key.selection_environment); });
  changed([&](auto& key) { flip(key.preparation_domain); });
  changed([&](auto& key) { flip(key.incumbent.planned_region); });
  changed([&](auto& key) { flip(key.incumbent.physical_recipe); });
  changed([&](auto& key) { flip(key.incumbent.backend_recipe); });
  changed([&](auto& key) { flip(key.incumbent.backend_build); });
  changed([&](auto& key) { flip(key.incumbent.backend_selection); });
  changed([&](auto& key) { flip(key.alternative.planned_region); });
  changed([&](auto& key) { flip(key.alternative.physical_recipe); });
  changed([&](auto& key) { flip(key.alternative.backend_recipe); });
  changed([&](auto& key) { flip(key.alternative.backend_build); });
  changed([&](auto& key) { flip(key.alternative.backend_selection); });
  auto swapped = original;
  std::swap(swapped.incumbent, swapped.alternative);
  EXPECT_NE(ex::RecipeComparisonKeyDigest(swapped), digest);
}

TEST(FusionRanksTest, ExactQualityRejectionHasNoLatencyAndRetainsEvidenceAuthority) {
  Fixture f;
  auto row = RecipeRecord(f.registry.SelectionDigest());
  const auto latency = row;
  row.kind = ex::RecipeComparisonKind::kRejectIncumbentQuality;
  row.incumbent_ns = row.alternative_ns = row.samples = 0;
  ASSERT_TRUE(ex::ValidateRecipeComparison(row));
  EXPECT_EQ(ex::RecipeComparisonPreference(row), ex::RecipePreference::kAlternative);
  EXPECT_EQ(ex::RecipeComparisonKeyDigest(row.key), ex::RecipeComparisonKeyDigest(latency.key));
  EXPECT_NE(ex::RecipeComparisonsDigest(std::span(&row, 1)),
            ex::RecipeComparisonsDigest(std::span(&latency, 1)));
  auto created = ex::Registry::Create(f.declarations, {.recipe_comparisons = {row}});
  ASSERT_TRUE(created);
  ASSERT_NE(created->FindRecipeComparison(row.key), nullptr);
  auto miss = row.key;
  miss.preparation_domain[0] ^= 1;
  EXPECT_EQ(created->FindRecipeComparison(miss), nullptr);
  const auto reject = [&](auto mutation) {
    auto bad = row;
    mutation(bad);
    EXPECT_FALSE(ex::ValidateRecipeComparison(bad));
  };
  reject([](auto& r) { r.kind = static_cast<ex::RecipeComparisonKind>(255); });
  reject([](auto& r) { r.incumbent_ns = 1; });
  reject([](auto& r) { r.alternative_ns = 1; });
  reject([](auto& r) { r.samples = 1; });
  reject([](auto& r) { r.protocol = {}; });
  reject([](auto& r) { r.receipt = {}; });
  reject([](auto& r) { r.provenance = "UNREVIEWED"; });
  reject([](auto& r) { r.shape_class.clear(); });
  reject([](auto& r) { r.kind = ex::RecipeComparisonKind::kLatency; });
}

TEST(FusionRanksTest, RecipeLookupIsExactMoveStableAndSelectionBound) {
  Fixture f;
  const auto selection = f.registry.SelectionDigest();
  auto comparison = RecipeRecord(selection);
  auto old_canonicalizer = comparison;
  old_canonicalizer.key.region_canonicalizer = 1;
  EXPECT_FALSE(ex::Registry::Create(f.declarations, {.recipe_comparisons = {old_canonicalizer}}));
  auto created = ex::Registry::Create(f.declarations, {.recipe_comparisons = {comparison}});
  ASSERT_TRUE(created) << (created ? "" : created.error().detail);
  EXPECT_TRUE(created->HasRecipeComparisons(comparison.key.discovery_operation));
  EXPECT_FALSE(created->HasRecipeComparisons(ir::Primitive(ir::Op::kMul)));
  auto moved = std::move(*created);
  const auto* found = moved.FindRecipeComparison(comparison.key);
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found, &moved.ranks().recipe_comparisons.front());

  auto near = comparison.key;
  near.alternative.physical_recipe[0] ^= 1;
  EXPECT_EQ(moved.FindRecipeComparison(near), nullptr);
  near = comparison.key;
  near.selection_environment[0] ^= 1;
  EXPECT_EQ(moved.FindRecipeComparison(near), nullptr);
  near = comparison.key;
  near.version = 2;
  EXPECT_EQ(moved.FindRecipeComparison(near), nullptr);
}

TEST(FusionRanksTest, RecipeRowsValidateAndDataDigestExcludesSelection) {
  Fixture f;
  const auto selection = f.registry.SelectionDigest();
  auto record = RecipeRecord(selection);
  auto invalid = record;
  invalid.key.incumbent = invalid.key.alternative;
  EXPECT_FALSE(ex::Registry::Create(f.declarations, {.recipe_comparisons = {invalid}}));
  invalid = record;
  invalid.key.selection_environment[0] ^= 1;
  auto stale_registry = ex::Registry::Create(f.declarations, {.recipe_comparisons = {invalid}});
  ASSERT_TRUE(stale_registry) << (stale_registry ? "" : stale_registry.error().detail);
  EXPECT_TRUE(stale_registry->HasRecipeComparisons(record.key.discovery_operation));
  EXPECT_EQ(stale_registry->FindRecipeComparison(invalid.key), nullptr);
  auto current_key = invalid.key;
  current_key.selection_environment = stale_registry->SelectionDigest();
  EXPECT_EQ(stale_registry->FindRecipeComparison(current_key), nullptr);
  invalid = record;
  invalid.provenance = "UNREVIEWED";
  EXPECT_FALSE(ex::Registry::Create(f.declarations, {.recipe_comparisons = {invalid}}));
  invalid = record;
  invalid.shape_class = "invalid\nlabel";
  EXPECT_FALSE(ex::Registry::Create(f.declarations, {.recipe_comparisons = {invalid}}));
  invalid = record;
  invalid.key.mode = static_cast<ex::RecipeRunMode>(255);
  EXPECT_FALSE(ex::Registry::Create(f.declarations, {.recipe_comparisons = {invalid}}));
  invalid = record;
  invalid.key.discovery_operation = {static_cast<ir::Op>(std::numeric_limits<std::uint16_t>::max()),
                                     ir::FusedKind::kNone};
  EXPECT_FALSE(ex::Registry::Create(f.declarations, {.recipe_comparisons = {invalid}}));
  auto duplicate = record;
  duplicate.alternative_ns += 1;
  EXPECT_FALSE(ex::Registry::Create(f.declarations, {.recipe_comparisons = {record, duplicate}}));
  EXPECT_EQ(ex::ClassifyRecipeLatency(100, 89), ex::RecipePreference::kAlternative);
  EXPECT_EQ(ex::ClassifyRecipeLatency(100, 91), ex::RecipePreference::kNeutral);
  EXPECT_EQ(ex::ClassifyRecipeLatency(89, 100), ex::RecipePreference::kIncumbent);

  auto second = record;
  second.key.subject[0] ^= 1;
  std::array records{record, second};
  auto reverse = records;
  std::ranges::reverse(reverse);
  EXPECT_EQ(ex::RecipeComparisonsDigest(records), ex::RecipeComparisonsDigest(reverse));
  const auto key_digest = ex::RecipeComparisonKeyDigest(record.key);
  const auto data_digest = ex::RecipeComparisonsDigest(std::span(&record, 1));
  const auto expect_evidence_change = [&](auto mutation) {
    auto changed = record;
    mutation(changed);
    EXPECT_EQ(ex::RecipeComparisonKeyDigest(changed.key), key_digest);
    EXPECT_NE(ex::RecipeComparisonsDigest(std::span(&changed, 1)), data_digest);
  };
  expect_evidence_change([](auto& row) { ++row.incumbent_ns; });
  expect_evidence_change([](auto& row) { ++row.alternative_ns; });
  expect_evidence_change([](auto& row) { ++row.samples; });
  expect_evidence_change([](auto& row) { row.protocol[0] ^= 1; });
  expect_evidence_change([](auto& row) { row.receipt[0] ^= 1; });
  expect_evidence_change([](auto& row) { row.provenance += " changed"; });
  expect_evidence_change([](auto& row) { row.shape_class += " changed"; });

  const auto baseline = f.Ranked();
  auto recipe = RecipeRecord(baseline.SelectionDigest());
  auto recipe_data = baseline.ranks();
  recipe_data.recipe_comparisons.push_back(recipe);
  const auto with_recipe = ex::Registry::Create(f.declarations, std::move(recipe_data));
  ASSERT_TRUE(with_recipe) << (with_recipe ? "" : with_recipe.error().detail);
  EXPECT_NE(baseline.Digest(), with_recipe->Digest());
  EXPECT_NE(baseline.RankDigest(), with_recipe->RankDigest());
  EXPECT_EQ(baseline.SelectionDigest(), with_recipe->SelectionDigest());
  EXPECT_EQ(ex::Candidates(baseline, f.query), ex::Candidates(*with_recipe, f.query));
  EXPECT_EQ(ex::FusionPreferenceFor(baseline, f.query, "test.fusion"),
            ex::FusionPreferenceFor(*with_recipe, f.query, "test.fusion"));

  auto measured = f.record;
  ++measured.fused_ns;
  const auto with_fusion = ex::Registry::Create(f.declarations, {.fusions = {measured}});
  ASSERT_TRUE(with_fusion) << (with_fusion ? "" : with_fusion.error().detail);
  EXPECT_NE(baseline.Digest(), with_fusion->Digest());
  EXPECT_NE(baseline.RankDigest(), with_fusion->RankDigest());
  EXPECT_EQ(baseline.SelectionDigest(), with_fusion->SelectionDigest());
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
  const auto all_candidates = ranked_view.Candidates(f.graph, f.graph.nodes().back());
  ASSERT_EQ(all_candidates.size(), 1U);
  std::optional<std::size_t> candidate_cursor;
  for (const auto expected : all_candidates) {
    candidate_cursor = ranked_view.NextCandidate(f.graph, f.graph.nodes().back(),
                                                 f.graph.nodes().back().key, candidate_cursor);
    ASSERT_TRUE(candidate_cursor);
    EXPECT_EQ(*candidate_cursor, expected);
    EXPECT_TRUE(ranked_view.IsCandidate(f.graph, f.graph.nodes().back(), f.graph.nodes().back().key,
                                        *candidate_cursor));
  }
  EXPECT_FALSE(ranked_view.NextCandidate(f.graph, f.graph.nodes().back(),
                                         f.graph.nodes().back().key, candidate_cursor));
  pl::Policy disabled_policy{.disabled = {"rms_norm_mul"}};
  const pl::SelectionView disabled_view(ranked, kDevice, disabled_policy, false, false);
  EXPECT_FALSE(disabled_view.NextCandidate(f.graph, f.graph.nodes().back(),
                                           f.graph.nodes().back().key, candidate_cursor));
  const pl::RootAlignment invalid_alignment = [](const ir::Graph&, ir::TensorId) {
    return std::uint64_t{1} << 63;
  };
  const pl::SelectionView invalid_view(ranked, kDevice, policy, false, false, {},
                                       invalid_alignment);
  EXPECT_TRUE(invalid_view.Candidates(f.graph, f.graph.nodes().back()).empty());
  EXPECT_FALSE(invalid_view.NextCandidate(f.graph, f.graph.nodes().back(),
                                          f.graph.nodes().back().key, candidate_cursor));
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
