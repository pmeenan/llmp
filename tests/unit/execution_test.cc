// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The implementation registry and plans (execution/registry.h, D-053), on
// fake declarations, in every build profile (backend-proof P1):
// - a plan's identity follows each operation's implementation identity
//   (BP-S2);
// - a plan made for another identity is rejected before anything binds
//   (BP-S2);
// - a build lacking an implementation reports the plan unsupported, and
//   nothing is substituted (BP-S4).

#include <gtest/gtest.h>

#include <cstddef>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "expected_error.h"

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
  for (int i = 0; i <= static_cast<int>(Operation::kDupUpAdd); ++i) {
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

}  // namespace
