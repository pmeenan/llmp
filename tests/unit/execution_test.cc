// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The implementation registry and plans (execution/registry.h, D-053), on
// fake declarations, in every build profile (backend-proof P1):
// - a plan's identity follows each operation's implementation identity
//   (BP-S2);
// - a plan made for another identity is rejected before anything binds
//   (BP-S2);
// - a build lacking an implementation reports the plan unsupported, and
//   nothing is substituted (BP-S4);
// - capabilities (D-108): every legacy operation's key in the op set, and
//   Candidates() by measured rank, then the ladder's order, refusing a
//   transient expansion for decode-shaped products, anything lossy
//   without the opt-in, and an implementation that needs tuning data for
//   a shape the device's data does not list.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "execution/measured_ranks.h"
#include "execution/registry.h"
#include "expected_error.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "ir/op.h"

namespace {

using llmp::execution::Choice;
using llmp::execution::Implementation;
using llmp::execution::Operation;
using llmp::execution::OperationName;
using llmp::execution::Plan;
using llmp::execution::PlanError;
using llmp::execution::PlanRejection;
using llmp::execution::Registry;
using llmp::execution::Resolve;
using llmp::test_support::Failed;
using llmp::test_support::FailedCode;

Implementation Fake(std::string name, Operation operation, std::string revision = "r1") {
  return {.name = std::move(name),
          .operation = operation,
          .source = "fake",
          .revision = std::move(revision),
          .build = "cpu",
          .variant = "reference"};
}

// A core build: fused and unfused RMSNorm-mul, an add and one matrix
// multiplication. A build with an optional module adds a second one.
std::vector<Implementation> Core(std::string fused_revision = "r1") {
  return {Fake("fake.rms_norm_mul.fused", Operation::kRmsNormMul, std::move(fused_revision)),
          Fake("fake.rms_norm_mul.unfused", Operation::kRmsNormMul),
          Fake("fake.add", Operation::kAdd), Fake("fake.mul_mat.gemm", Operation::kMatMul)};
}

std::vector<Implementation> WithModule() {
  std::vector<Implementation> implementations = Core();
  implementations.push_back(Fake("module.mul_mat.gemv", Operation::kMatMul));
  return implementations;
}

std::vector<Choice> Layer(std::string norm, std::string mul_mat = "fake.mul_mat.gemm") {
  return {{.operation = Operation::kRmsNormMul, .implementation = std::move(norm)},
          {.operation = Operation::kMatMul, .implementation = std::move(mul_mat)},
          {.operation = Operation::kAdd, .implementation = "fake.add"}};
}

// An implementation's and a plan's identities hash the operation's name, so
// no two operations share one.
TEST(PlanRegistryTest, EveryOperationHasItsOwnName) {
  std::set<std::string_view> names;
  for (int i = 0; i <= static_cast<int>(Operation::kGatedResidualModulateSplit); ++i) {
    const std::string_view name = OperationName(static_cast<Operation>(i));
    EXPECT_NE(name, "unknown") << i;
    EXPECT_TRUE(names.insert(name).second) << name;
  }
}

TEST(PlanRegistryTest, ResolvesEveryOperationToTheImplementationItNames) {
  const auto registry = Registry::Create(Core()).value();
  for (const char* norm : {"fake.rms_norm_mul.fused", "fake.rms_norm_mul.unfused"}) {
    const auto plan = Plan::Build(registry, Layer(norm)).value();
    const auto bound = Resolve(registry, plan);
    ASSERT_TRUE(bound.has_value()) << bound.error().detail;
    ASSERT_EQ(bound->size(), 3U);
    EXPECT_EQ(bound->at(0).name, norm);
    EXPECT_EQ(bound->at(1).name, "fake.mul_mat.gemm");
    EXPECT_EQ(bound->at(2).name, "fake.add");
    EXPECT_EQ(bound->identity(), plan.identity());
    // Bound to the registry's own declarations.
    const std::size_t add = registry.Find("fake.add").value_or(registry.size());
    ASSERT_LT(add, registry.size());
    EXPECT_EQ(&bound->at(2), &registry.at(add));
  }
}

TEST(PlanRegistryTest, PlanIdentityFollowsEachImplementationIdentity) {
  const auto registry = Registry::Create(Core()).value();
  const auto fused = Plan::Build(registry, Layer("fake.rms_norm_mul.fused")).value();
  const auto again = Plan::Build(registry, Layer("fake.rms_norm_mul.fused")).value();
  const auto unfused = Plan::Build(registry, Layer("fake.rms_norm_mul.unfused")).value();
  EXPECT_EQ(fused.identity(), again.identity());
  EXPECT_NE(fused.identity(), unfused.identity());

  // The same choices against a build whose fused implementation changed.
  const auto rebuilt = Registry::Create(Core("r2")).value();
  const auto moved = Plan::Build(rebuilt, Layer("fake.rms_norm_mul.fused")).value();
  EXPECT_NE(fused.identity(), moved.identity());
  // A plan not using the changed implementation keeps its identity.
  EXPECT_EQ(Plan::Build(rebuilt, Layer("fake.rms_norm_mul.unfused")).value().identity(),
            unfused.identity());

  // Order is part of a plan.
  std::vector<Choice> reordered = Layer("fake.rms_norm_mul.fused");
  std::swap(reordered[1], reordered[2]);
  EXPECT_NE(Plan::Build(registry, reordered).value().identity(), fused.identity());
}

TEST(PlanRegistryTest, IdentityFieldsCannotTradeBytes) {
  Implementation a = Fake("fake.a", Operation::kAdd);
  Implementation b = a;
  a.revision = "ab";
  a.build = "c";
  b.revision = "a";
  b.build = "bc";
  EXPECT_NE(llmp::execution::IdentityOf(a), llmp::execution::IdentityOf(b));
  Implementation c = Fake("fake.a", Operation::kMul);
  EXPECT_NE(llmp::execution::IdentityOf(Fake("fake.a", Operation::kAdd)),
            llmp::execution::IdentityOf(c));
}

TEST(PlanRegistryTest, APlanMadeForAnotherIdentityIsStale) {
  const auto old_build = Registry::Create(Core("r1")).value();
  const auto new_build = Registry::Create(Core("r2")).value();
  const auto plan = Plan::Build(old_build, Layer("fake.rms_norm_mul.fused")).value();
  const auto refused = Resolve(new_build, plan);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().error, PlanError::kStale);
  EXPECT_EQ(refused.error().operation, 0U);
  // The unfused implementation is unchanged, and still not used instead.
  EXPECT_TRUE(Resolve(new_build, Plan::Build(old_build, Layer("fake.rms_norm_mul.unfused")).value())
                  .has_value());
}

TEST(PlanRegistryTest, AnImplementationThisBuildLacksIsUnsupported) {
  // Made in a build with the optional module, resolved in the core build:
  // the core build's other matrix multiplication is never substituted.
  const auto module_build = Registry::Create(WithModule()).value();
  const auto core_build = Registry::Create(Core()).value();
  const auto plan =
      Plan::Build(module_build, Layer("fake.rms_norm_mul.fused", "module.mul_mat.gemv")).value();
  ASSERT_TRUE(Resolve(module_build, plan).has_value());
  const auto refused = Resolve(core_build, plan);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().error, PlanError::kUnsupported);
  EXPECT_EQ(refused.error().operation, 1U);
  EXPECT_NE(refused.error().detail.find("module.mul_mat.gemv"), std::string::npos);

  // A planner in the core build cannot choose it either.
  const auto unplanned =
      Plan::Build(core_build, Layer("fake.rms_norm_mul.fused", "module.mul_mat.gemv"));
  ASSERT_FALSE(unplanned.has_value());
  EXPECT_EQ(unplanned.error().error, PlanError::kUnsupported);
  EXPECT_EQ(unplanned.error().operation, 1U);

  // A build with no implementation of an operation at all.
  const auto empty = Registry::Create({}).value();
  EXPECT_EQ(FailedCode(Resolve(empty, plan)), PlanError::kUnsupported);
  EXPECT_EQ(Failed(Resolve(empty, plan), &PlanRejection::operation), 0U);
}

TEST(PlanRegistryTest, AnImplementationOfAnotherOperationDoesNotBind) {
  const auto registry = Registry::Create(Core()).value();
  const std::vector<Choice> choices = {
      {.operation = Operation::kMul, .implementation = "fake.add"}};
  const auto refused = Plan::Build(registry, choices);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().error, PlanError::kWrongOperation);

  // Nor when a later build reuses a name for another operation.
  const auto plan = Plan::Build(registry, Layer("fake.rms_norm_mul.fused")).value();
  std::vector<Implementation> renamed = Core();
  renamed[2].operation = Operation::kMul;  // fake.add
  const auto later = Registry::Create(std::move(renamed)).value();
  const auto rejected = Resolve(later, plan);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().error, PlanError::kWrongOperation);
  EXPECT_EQ(rejected.error().operation, 2U);
}

TEST(PlanRegistryTest, RefusesMalformedRegistriesAndPlans) {
  for (const char* name : {"", "Fake.upper", "fake add", "fake/add", "fake-add"}) {
    const auto refused = Registry::Create({Fake(name, Operation::kAdd)});
    ASSERT_FALSE(refused.has_value()) << name;
    EXPECT_EQ(refused.error().error, PlanError::kInvalid);
  }
  EXPECT_FALSE(Registry::Create({Fake(std::string(Registry::kMaxName + 1, 'a'), Operation::kAdd)})
                   .has_value());
  EXPECT_TRUE(
      Registry::Create({Fake(std::string(Registry::kMaxName, 'a'), Operation::kAdd)}).has_value());
  // Two declarations of one name, even of different operations.
  const auto duplicate =
      Registry::Create({Fake("fake.x", Operation::kAdd), Fake("fake.y", Operation::kAdd),
                        Fake("fake.x", Operation::kMul)});
  ASSERT_FALSE(duplicate.has_value());
  EXPECT_EQ(duplicate.error().error, PlanError::kInvalid);
  std::vector<Implementation> many;
  for (std::size_t i = 0; i <= Registry::kMaxImplementations; ++i) {
    many.push_back(Fake("fake." + std::to_string(i), Operation::kAdd));
  }
  EXPECT_FALSE(Registry::Create(many).has_value());
  many.pop_back();
  EXPECT_TRUE(Registry::Create(std::move(many)).has_value());

  const auto registry = Registry::Create(Core()).value();
  EXPECT_EQ(FailedCode(Plan::Build(registry, {})), PlanError::kInvalid);
  const std::vector<Choice> too_many(Plan::kMaxOperations + 1,
                                     {.operation = Operation::kAdd, .implementation = "fake.add"});
  EXPECT_EQ(FailedCode(Plan::Build(registry, too_many)), PlanError::kInvalid);
  const std::vector<Choice> junk = {{.operation = Operation::kAdd, .implementation = "fake.add\n"}};
  EXPECT_EQ(FailedCode(Plan::Build(registry, junk)), PlanError::kInvalid);
}

// ---------------------------------------------------------------- capabilities

namespace ex = llmp::execution;
namespace ir = llmp::ir;
using llmp::format::Encoding;
using llmp::format::Layout;

TEST(CapabilityTest, EveryOperationHasAKeyInTheOpSet) {
  std::set<std::string_view> ops;
  for (int i = 0; i <= static_cast<int>(ir::Op::kPrefixOverwrite); ++i) {
    const std::string_view name = ir::OpName(static_cast<ir::Op>(i));
    EXPECT_NE(name, "unknown") << i;
    EXPECT_TRUE(ops.insert(name).second) << name;
  }
  std::set<std::string_view> fused;
  for (int i = 0; i <= static_cast<int>(ir::kLastFusedKind); ++i) {
    const std::string_view name = ir::FusedKindName(static_cast<ir::FusedKind>(i));
    EXPECT_NE(name, "unknown") << i;
    EXPECT_TRUE(fused.insert(name).second) << name;
  }
  for (int i = 0; i <= static_cast<int>(Operation::kGatedResidualModulateSplit); ++i) {
    const auto operation = static_cast<Operation>(i);
    const ir::OpKey key = ex::OpKeyOf(operation);
    EXPECT_EQ(key.op == ir::Op::kFused, key.fused != ir::FusedKind::kNone)
        << OperationName(operation);
    EXPECT_NE(ir::OpKeyName(key), "unknown") << OperationName(operation);
  }
  EXPECT_EQ(ex::OpKeyOf(Operation::kRmsNormMul), ir::Fused(ir::FusedKind::kRmsNormMul));
  EXPECT_EQ(ex::OpKeyOf(Operation::kMatMul), ir::Primitive(ir::Op::kMatMul));
  EXPECT_EQ(ex::OpKeyOf(Operation::kQuantLinear), ir::Primitive(ir::Op::kMatMul));
  EXPECT_EQ(ir::OpKeyName(ex::OpKeyOf(Operation::kRopeSetRows)), "fused.rope_store");
}

constexpr std::array kFp4 = {Encoding::kNvfp4};
constexpr std::array kDecoded = {Encoding::kNvfp4, Encoding::kMxfp8};
constexpr std::array kSplitOnly = {Layout::kSplit};
constexpr std::array kNativeOperands = {
    ex::OperandRule{.operand = 1, .encodings = kFp4, .layouts = kSplitOnly}};
constexpr std::array kDecodedOperands = {
    ex::OperandRule{.operand = 1, .encodings = kDecoded, .layouts = {}}};

bool RefusesOddK(const ex::CapabilityQuery& q) { return q.k % 2 == 0; }

Implementation Capable(std::string name, const ex::Capability& capability) {
  Implementation implementation = Fake(std::move(name), Operation::kMatMul);
  implementation.capability = capability;
  return implementation;
}

std::vector<Implementation> Ladder() {
  const ir::OpKey mm = ir::Primitive(ir::Op::kMatMul);
  return {
      Capable("fake.mm.native", {.op = mm,
                                 .operands = kNativeOperands,
                                 .device = {.min_arch = 120, .max_arch = 121, .needs = ex::kFp4Mma},
                                 .rung = ex::Rung::kNative}),
      Capable("fake.mm.decode_vector", {.op = mm,
                                        .operands = kDecodedOperands,
                                        .shape = {.min_rows = 1, .max_rows = 16},
                                        .rung = ex::Rung::kOnChipDecode,
                                        .row_invariant = true}),
      Capable("fake.mm.decode_tile", {.op = mm,
                                      .operands = kDecodedOperands,
                                      .shape = {.k_multiple = 64},
                                      .rung = ex::Rung::kOnChipDecode,
                                      .priority = 1}),
      Capable("fake.mm.expand", {.op = mm,
                                 .operands = kDecodedOperands,
                                 .rung = ex::Rung::kOnChipDecode,
                                 .expansion = ex::Expansion::kTransientWorkspace,
                                 .priority = 5}),
      Capable("fake.mm.bf16", {.op = mm, .rung = ex::Rung::kBf16Decode, .accepts = RefusesOddK}),
      Capable("fake.mm.lossy", {.op = mm, .rung = ex::Rung::kNative, .lossy = true}),
      Capable("fake.mm.primitive", {.op = mm, .rung = ex::Rung::kPrimitive, .capturable = false}),
      Capable("fake.add.any", {.op = ir::Primitive(ir::Op::kAdd)}),
      Fake("fake.mm.undeclared", Operation::kMatMul),
  };
}

std::vector<std::string> Names(const Registry& registry, const std::vector<std::size_t>& found) {
  std::vector<std::string> names;
  for (const std::size_t i : found) {
    names.push_back(registry.at(i).name);
  }
  return names;
}

TEST(CapabilityTest, AStateSpanRequirementIsExplicitInSelectionAndIdentity) {
  const auto key = ir::Fused(ir::FusedKind::kStateBankAttention);
  auto implementation = Capable("fake.bank", {.op = key});
  const auto before = ex::IdentityOf(implementation);
  implementation.capability->needs_state_span = true;
  EXPECT_NE(before, ex::IdentityOf(implementation));
  auto made = Registry::Create({implementation});
  ASSERT_TRUE(made);
  ex::CapabilityQuery query{.op = key};
  EXPECT_TRUE(ex::Candidates(*made, query).empty());
  query.state_span = true;
  EXPECT_EQ(Names(*made, ex::Candidates(*made, query)), std::vector<std::string>{"fake.bank"});
}

TEST(CapabilityTest, CandidatesFollowTheLadder) {
  const auto registry = Registry::Create(Ladder()).value();
  const std::array weights = {
      ex::OperandFormat{.operand = 1, .encoding = Encoding::kNvfp4, .layout = Layout::kSplit}};
  ex::CapabilityQuery q{.op = ir::Primitive(ir::Op::kMatMul),
                        .operands = weights,
                        .rows = 1,
                        .k = 2560,
                        .n = 512,
                        .arch = 121,
                        .features = ex::kFp4Mma | ex::kBf16Mma};
  // Decode-shaped on a GB10: no expansion, nothing lossy.
  EXPECT_EQ(
      Names(registry, ex::Candidates(registry, q)),
      (std::vector<std::string>{"fake.mm.native", "fake.mm.decode_tile", "fake.mm.decode_vector",
                                "fake.mm.bf16", "fake.mm.primitive"}));
  // At 16 rows still decode-shaped; at 17 the expansion is a candidate and
  // the vector kernel is not.
  q.rows = 16;
  EXPECT_EQ(Names(registry, ex::Candidates(registry, q))[1], "fake.mm.decode_tile");
  q.rows = 17;
  EXPECT_EQ(Names(registry, ex::Candidates(registry, q)),
            (std::vector<std::string>{"fake.mm.native", "fake.mm.expand", "fake.mm.decode_tile",
                                      "fake.mm.bf16", "fake.mm.primitive"}));
  // A measured rank orders before the static priority, and across rungs:
  // what it ranks comes first, the rest in the ladder's order after.
  const std::array measured = {ex::MeasuredRank{.implementation = "fake.mm.decode_tile", .rank = 0},
                               ex::MeasuredRank{.implementation = "fake.mm.expand", .rank = 1}};
  q.measured = measured;
  EXPECT_EQ(Names(registry, ex::Candidates(registry, q)),
            (std::vector<std::string>{"fake.mm.decode_tile", "fake.mm.expand", "fake.mm.native",
                                      "fake.mm.bf16", "fake.mm.primitive"}));
  const std::array bf16_fastest = {ex::MeasuredRank{.implementation = "fake.mm.bf16", .rank = 0},
                                   ex::MeasuredRank{.implementation = "fake.mm.native", .rank = 1}};
  q.measured = bf16_fastest;
  EXPECT_EQ(Names(registry, ex::Candidates(registry, q)),
            (std::vector<std::string>{"fake.mm.bf16", "fake.mm.native", "fake.mm.expand",
                                      "fake.mm.decode_tile", "fake.mm.primitive"}));
  q.measured = {};
  // A policy with another decode bound.
  q.rows = 8;
  q.policy.max_decode_rows = 4;
  EXPECT_EQ(Names(registry, ex::Candidates(registry, q))[1], "fake.mm.expand");
  q.policy.max_decode_rows = 16;
  // sm_86 has no FP4 matrix unit: the native kernel drops out.
  q.rows = 1;
  q.arch = 86;
  q.features = ex::kBf16Mma | ex::kInt8Mma;
  EXPECT_EQ(Names(registry, ex::Candidates(registry, q))[0], "fake.mm.decode_tile");
  // The lossy kernel only by the alias's opt-in.
  q.arch = 121;
  q.features = ex::kFp4Mma;
  q.policy.allow_lossy = true;
  EXPECT_EQ(
      Names(registry, ex::Candidates(registry, q)),
      (std::vector<std::string>{"fake.mm.lossy", "fake.mm.native", "fake.mm.decode_tile",
                                "fake.mm.decode_vector", "fake.mm.bf16", "fake.mm.primitive"}));
  q.policy.allow_lossy = false;
  // Requirements: row invariance, capture.
  q.row_invariant = true;
  EXPECT_EQ(Names(registry, ex::Candidates(registry, q)),
            (std::vector<std::string>{"fake.mm.decode_vector"}));
  q.row_invariant = false;
  q.capturable = true;
  EXPECT_EQ(Names(registry, ex::Candidates(registry, q)).back(), "fake.mm.bf16");
  q.capturable = false;
  // Operand formats the rules do not take; K the shape rule or the
  // predicate refuses.
  const std::array q4k = {
      ex::OperandFormat{.operand = 1, .encoding = Encoding::kQ4_K, .layout = Layout::kGgmlBlocks}};
  q.operands = q4k;
  EXPECT_EQ(Names(registry, ex::Candidates(registry, q)),
            (std::vector<std::string>{"fake.mm.bf16", "fake.mm.primitive"}));
  // A query that leaves out an operand a rule constrains matches only the
  // implementations that leave it unconstrained.
  q.operands = {};
  EXPECT_EQ(Names(registry, ex::Candidates(registry, q)),
            (std::vector<std::string>{"fake.mm.bf16", "fake.mm.primitive"}));
  q.operands = weights;
  q.k = 2561;
  EXPECT_EQ(
      Names(registry, ex::Candidates(registry, q)),
      (std::vector<std::string>{"fake.mm.native", "fake.mm.decode_vector", "fake.mm.primitive"}));
  // Another operation.
  q.op = ir::Primitive(ir::Op::kAdd);
  EXPECT_EQ(Names(registry, ex::Candidates(registry, q)),
            (std::vector<std::string>{"fake.add.any"}));
  q.op = ir::Fused(ir::FusedKind::kMatMulGlu);
  EXPECT_TRUE(ex::Candidates(registry, q).empty());
}

// An implementation that needs tuning data (EXL3's pinned reconstruction)
// is a candidate only for a shape the query's tuning data lists for it.
TEST(CapabilityTest, TuningDataGatesWhatNeedsIt) {
  const ir::OpKey mm = ir::Primitive(ir::Op::kMatMul);
  const auto registry =
      Registry::Create({Capable("fake.mm.pinned", {.op = mm, .priority = 9, .needs_tuning = true}),
                        Capable("fake.mm.plain", {.op = mm})})
          .value();
  ex::CapabilityQuery q{.op = mm, .rows = 145, .k = 896, .n = 896};
  EXPECT_EQ(Names(registry, ex::Candidates(registry, q)),
            std::vector<std::string>{"fake.mm.plain"});
  const std::array tuned = {
      ex::TunedShape{.implementation = "fake.mm.pinned", .rows = 145, .k = 896, .n = 896}};
  q.tuned = tuned;
  EXPECT_EQ(Names(registry, ex::Candidates(registry, q)),
            (std::vector<std::string>{"fake.mm.pinned", "fake.mm.plain"}));
  for (const auto& [rows, k, n] : std::array<std::array<std::uint64_t, 3>, 3>{
           {{144, 896, 896}, {145, 4864, 896}, {145, 896, 4864}}}) {
    q.rows = rows;
    q.k = k;
    q.n = n;
    EXPECT_EQ(Names(registry, ex::Candidates(registry, q)),
              std::vector<std::string>{"fake.mm.plain"});
  }
  // Another implementation's tuning data is not this one's.
  const std::array other = {
      ex::TunedShape{.implementation = "fake.mm.plain", .rows = 145, .k = 896, .n = 896}};
  q = {.op = mm, .rows = 145, .k = 896, .n = 896, .tuned = other};
  EXPECT_EQ(Names(registry, ex::Candidates(registry, q)),
            std::vector<std::string>{"fake.mm.plain"});
}

TEST(CapabilityTest, TuningFactsPreserveThePinnedOutputEncoding) {
  const ir::OpKey mm = ir::Primitive(ir::Op::kMatMul);
  const auto registry =
      Registry::Create({Capable("fake.mm.pinned", {.op = mm, .needs_tuning = true})}).value();
  const std::array tuned{ex::TunedShape{.implementation = "fake.mm.pinned",
                                        .rows = 145,
                                        .k = 896,
                                        .n = 128,
                                        .output_encoding = Encoding::kF16}};
  ex::CapabilityQuery query{.op = mm, .rows = 145, .k = 896, .n = 128, .tuned = tuned};
  EXPECT_TRUE(ex::Candidates(registry, query).empty());
  const std::array wrong{
      ex::OperandFormat{.operand = ex::kOutputOperand, .encoding = Encoding::kF32}};
  query.operands = wrong;
  EXPECT_TRUE(ex::Candidates(registry, query).empty());
  const std::array right{
      ex::OperandFormat{.operand = ex::kOutputOperand, .encoding = Encoding::kF16}};
  query.operands = right;
  EXPECT_EQ(ex::Candidates(registry, query).size(), 1U);
}

TEST(CapabilityTest, ADeclaredCapabilityIsPartOfTheIdentity) {
  const Implementation plain = Fake("fake.mm", Operation::kMatMul);
  Implementation declared = plain;
  declared.capability = ex::Capability{.op = ir::Primitive(ir::Op::kMatMul)};
  EXPECT_NE(ex::IdentityOf(plain), ex::IdentityOf(declared));
  Implementation other = declared;
  other.capability->rung = ex::Rung::kOnChipDecode;
  EXPECT_NE(ex::IdentityOf(other), ex::IdentityOf(declared));
  other = declared;
  other.capability->operands = kDecodedOperands;
  EXPECT_NE(ex::IdentityOf(other), ex::IdentityOf(declared));
  other = declared;
  other.capability->needs_tuning = true;
  EXPECT_NE(ex::IdentityOf(other), ex::IdentityOf(declared));
  other = declared;
  other.capability->batch = 16;
  EXPECT_NE(ex::IdentityOf(other), ex::IdentityOf(declared));
}

TEST(CapabilityTest, RefusesMalformedCapabilities) {
  const ir::OpKey mm = ir::Primitive(ir::Op::kMatMul);
  constexpr std::array kTwice = {ex::OperandRule{.operand = 1}, ex::OperandRule{.operand = 1}};
  const std::vector<ex::Capability> bad = {
      {.op = mm, .operands = kTwice},
      {.op = mm, .shape = {.min_rows = 17, .max_rows = 16}},
      {.op = mm, .shape = {.k_multiple = 0}},
      {.op = mm, .device = {.min_arch = 121, .max_arch = 86}},
      {.op = mm, .rung = static_cast<ex::Rung>(3)},
      {.op = mm, .group = "Not A Name"},
      {.op = ir::OpKey{.op = ir::Op::kFused, .fused = ir::FusedKind::kNone}},
      {.op = ir::OpKey{.op = ir::Op::kMatMul, .fused = ir::FusedKind::kMatMulGlu}},
      {.op = ir::OpKey{.op = static_cast<ir::Op>(999)}},
  };
  for (std::size_t i = 0; i < bad.size(); ++i) {
    const auto refused = Registry::Create({Capable("fake.mm", bad[i])});
    ASSERT_FALSE(refused.has_value()) << i;
    EXPECT_EQ(refused.error().error, PlanError::kInvalid) << i;
  }
  EXPECT_TRUE(
      Registry::Create({Capable("fake.mm", {.op = mm, .group = "ggml.mul_mat"})}).has_value());
}

TEST(PlanRegistryTest, SetDigestIsCanonicalAndCoversImplementationIdentity) {
  auto a = Registry::Create({Fake("fake.a", Operation::kAdd), Fake("fake.b", Operation::kMatMul)});
  auto b = Registry::Create({Fake("fake.b", Operation::kMatMul), Fake("fake.a", Operation::kAdd)});
  auto changed =
      Registry::Create({Fake("fake.a", Operation::kAdd), Fake("fake.b", Operation::kMatMul, "r2")});
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  ASSERT_TRUE(changed);
  EXPECT_EQ(a->Digest(), b->Digest());
  EXPECT_NE(a->Digest(), changed->Digest());
}
TEST(CapabilityTest, DenseExpansionFollowsRequestedPrecisionAndWeightEncoding) {
  const auto implementation = Capable(
      "fake.dense",
      {.op = ir::Primitive(ir::Op::kMatMul), .expansion = ex::Expansion::kDenseF32Accumulation});
  auto registry = Registry::Create({implementation});
  ASSERT_TRUE(registry);
  std::array operands{ex::OperandFormat{.operand = 0, .encoding = Encoding::kBF16}};
  ir::ProductParams params{};
  ex::CapabilityQuery query{.op = ir::Primitive(ir::Op::kMatMul),
                            .operands = operands,
                            .rows = 1,
                            .params = std::as_bytes(std::span(&params, 1))};
  EXPECT_EQ(ex::Candidates(*registry, query).size(), 1U);
  params.accumulate = ir::Accumulate::kF32;
  EXPECT_TRUE(ex::Candidates(*registry, query).empty());
  operands[0].encoding = Encoding::kF32;
  EXPECT_EQ(ex::Candidates(*registry, query).size(), 1U);
  operands[0].encoding = Encoding::kBF16;
  query.rows = 17;
  EXPECT_EQ(ex::Candidates(*registry, query).size(), 1U);
}

TEST(CapabilityTest, AutomaticFusionRequiresEvidenceWithoutHidingDirectOperations) {
  const auto op = ir::Fused(ir::FusedKind::kMatMulGlu);
  auto implementation = Capable("fake.fusion", {.op = op});
  const auto before = ex::IdentityOf(implementation);
  implementation.capability->fusion_needs_measurement = true;
  EXPECT_NE(before, ex::IdentityOf(implementation));
  auto registry = Registry::Create({implementation}).value();
  ex::CapabilityQuery query{.op = op};
  EXPECT_EQ(ex::Candidates(registry, query).size(), 1U);
  query.automatic_fusion = true;
  EXPECT_TRUE(ex::Candidates(registry, query).empty());
  const std::array measured{ex::MeasuredRank{.implementation = "fake.fusion", .rank = 0}};
  query.measured = measured;
  EXPECT_EQ(ex::Candidates(registry, query).size(), 1U);
}

TEST(CapabilityTest, ExplicitSourcePreferencePreservesPinsAfterMeasuredRanks) {
  const auto op = ir::Primitive(ir::Op::kMatMul);
  auto registry = Registry::Create({Capable("fake.new", {.op = op, .priority = 100}),
                                    Capable("fake.pin", {.op = op})})
                      .value();
  const std::vector<std::string> preferred{"fake.pin"};
  ex::CapabilityQuery query{.op = op, .preferred = preferred};
  EXPECT_EQ(Names(registry, ex::Candidates(registry, query)).front(), "fake.pin");
  const std::array measured{ex::MeasuredRank{.implementation = "fake.new", .rank = 0}};
  query.measured = measured;
  EXPECT_EQ(Names(registry, ex::Candidates(registry, query)).front(), "fake.new");
}

TEST(CapabilityTest, DeviceRanksRequireEveryMeasuredFactAndUnbatchedOperands) {
  ASSERT_EQ(ex::kDeviceMeasuredRanks.size(), 2U);
  for (const auto& record : ex::kDeviceMeasuredRanks) {
    SCOPED_TRACE(record.implementation);
    const bool product = record.op == ir::Primitive(ir::Op::kMatMul);
    std::array operands{
        ex::OperandFormat{.operand = 0,
                          .stacked = record.stacked,
                          .encoding = record.encoding,
                          .layout = record.layout,
                          .params = record.params,
                          .ne = {record.k, record.n, 1, 1}},
        ex::OperandFormat{
            .operand = 1,
            .encoding = product ? Encoding::kF32 : Encoding::kI32,
            .layout = Layout::kDense,
            .ne = {product ? record.k : record.rows_min, product ? record.rows_min : 1, 1, 1}},
        ex::OperandFormat{.operand = ex::kOutputOperand,
                          .encoding = Encoding::kF32,
                          .layout = Layout::kDense,
                          .ne = {product ? record.n : record.k, record.rows_min, 1, 1}},
    };
    ex::CapabilityQuery query{.op = record.op,
                              .operands = operands,
                              .rows = record.rows_min,
                              .k = record.k,
                              .n = record.n,
                              .arch = record.arch,
                              .device_class = record.device_class,
                              .device_model = record.device_model,
                              .multiprocessors = record.multiprocessors};
    ASSERT_EQ(ex::DeviceRank(query, record.implementation), 0U);
    EXPECT_FALSE(ex::DeviceRank(query, "fake.unmeasured"));
    auto registry =
        Registry::Create({Capable(std::string(record.implementation), {.op = record.op}),
                          Capable("fake.alternative", {.op = record.op, .priority = 100})})
            .value();
    EXPECT_EQ(Names(registry, ex::Candidates(registry, query)).front(), record.implementation);
    const auto original = query;
    const auto original_operands = operands;
    const auto misses = [&] {
      EXPECT_FALSE(ex::DeviceRank(query, record.implementation));
      query = original;
      operands = original_operands;
    };
    query.device_class = "sm_86";
    misses();
    query.device_model = "another device";
    misses();
    --query.arch;
    misses();
    --query.multiprocessors;
    misses();
    query.op = ir::Primitive(ir::Op::kAdd);
    misses();
    operands[0].encoding = Encoding::kBF16;
    misses();
    operands[0].layout = Layout::kDense;
    misses();
    operands[0].params.tensor_scale = !operands[0].params.tensor_scale;
    misses();
    ++operands[0].stacked;
    misses();
    --query.rows;
    misses();
    ++query.rows;
    misses();
    ++query.k;
    misses();
    ++query.n;
    misses();
    ++operands[0].ne[0];
    misses();
    ++operands[0].ne[1];
    misses();
    ++operands[1].ne[product ? 1 : 0];
    misses();
    for (auto& operand : operands) {
      for (const std::size_t dimension : {2U, 3U}) {
        operand.ne[dimension] = 2;
        misses();
      }
    }
    query.operands = {};
    misses();
    query.operands = std::span(operands).first(1);
    misses();
    const std::array override{ex::MeasuredRank{.implementation = "fake.alternative", .rank = 0}};
    query.measured = override;
    EXPECT_EQ(Names(registry, ex::Candidates(registry, query)).front(), "fake.alternative");
  }
}

TEST(CapabilityTest, TuningPinsCannotBeBorrowedAcrossExl3Rates) {
  const auto op = ir::Primitive(ir::Op::kMatMul);
  auto registry = Registry::Create({Capable("fake.pin", {.op = op, .needs_tuning = true})}).value();
  const std::array tuned{ex::TunedShape{
      .implementation = "fake.pin", .rows = 1, .k = 896, .n = 896, .weight_bits = 4}};
  std::array operands{
      ex::OperandFormat{.operand = 0, .encoding = Encoding::kExl3Trellis, .params = {.k_bits = 4}}};
  ex::CapabilityQuery query{
      .op = op, .operands = operands, .rows = 1, .k = 896, .n = 896, .tuned = tuned};
  EXPECT_EQ(ex::Candidates(registry, query).size(), 1U);
  operands[0].params.k_bits = 5;
  EXPECT_TRUE(ex::Candidates(registry, query).empty());
}

}  // namespace
