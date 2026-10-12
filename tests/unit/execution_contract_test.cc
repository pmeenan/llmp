// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/execution_contract.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "engine/component_source.h"
#include "expected_error.h"

namespace {
namespace en = llmp::engine;
namespace ex = llmp::execution;
namespace pl = llmp::planner;
using llmp::test_support::Failed;

llmp::base::Sha256Digest Seal(std::string_view text) {
  llmp::base::Sha256 hash;
  hash.Update(text);
  return hash.Finish();
}

en::NumericalStage Stage(std::uint16_t stage, std::string name) {
  return {
      .stage = stage,
      .name = std::move(name),
      .uses = {en::NumericalUse::kPrompt, en::NumericalUse::kDecode, en::NumericalUse::kScore,
               en::NumericalUse::kPromptCompanions},
      .phases = {
          {.phase = pl::Phase::kPrefill,
           .max_owners = 4,
           .max_owner_rows = 512,
           .max_total_rows = 1024},
          {.phase = pl::Phase::kDecode, .max_owners = 4, .max_owner_rows = 1, .max_total_rows = 4},
          {.phase = pl::Phase::kScore,
           .max_owners = 4,
           .max_owner_rows = 512,
           .max_total_rows = 1024}}};
}

pl::StepShape Shape() {
  return {.stage = 0,
          .phase = pl::Phase::kPrefill,
          .segments = {{.slot = 0, .rows = 3, .past = 10, .read_cells = 256},
                       {.slot = 2, .rows = 7, .past = 90, .read_cells = 256}}};
}

ex::Implementation Implementation(std::string name = "fake.score", std::string variant = "v1") {
  return {.name = std::move(name),
          .operation = ex::Operation::kLightningIndexer,
          .source = "fake",
          .revision = "test source",
          .build = "test build",
          .variant = std::move(variant)};
}

en::StageExecutionInput Input(const ex::Registry& registry, std::uint16_t stage, std::string name) {
  return {.stage = stage,
          .name = std::move(name),
          .policy = {},
          .patterns = Seal("test pattern versions"),
          .registry = &registry,
          .device = {.device_class = "test", .arch = 121, .features = ex::kBf16Mma},
          .backend = {.recipe = Seal("test recipe"),
                      .build = Seal("test compiler flags and images"),
                      .selection = Seal("test selector options and device facts")},
          .reservation_alignment = 2097152,
          .placement_alignment = 256};
}

TEST(ExecutionContractTest, DomainOwnsDeclarationsAndRefusesUnknownAndUnqualifiedUses) {
  auto declarations = std::vector{Stage(0, "target"), Stage(1, "companion")};
  auto domain = en::NumericalDomain::Create(declarations);
  ASSERT_TRUE(domain) << Failed(domain).value_or("");
  declarations[0].phases[0].row_invariant = true;
  declarations[0].phases[0].max_total_rows = 1;
  const auto original = Shape();
  EXPECT_TRUE(domain->CheckShape(original));
  auto shape = original;
  shape.row_invariant = true;
  EXPECT_FALSE(domain->CheckShape(shape));
  shape = original;
  shape.phase = pl::Phase::kVerify;
  EXPECT_FALSE(domain->CheckShape(shape));
  shape.phase = static_cast<pl::Phase>(255);
  EXPECT_FALSE(domain->CheckShape(shape));
  shape = original;
  shape.stage = 2;
  EXPECT_FALSE(domain->CheckShape(shape));
  shape = original;
  shape.segments[0].rows = 0;
  EXPECT_FALSE(domain->CheckShape(shape));
  shape = original;
  shape.segments[1].slot = 0;
  EXPECT_FALSE(domain->CheckShape(shape));
  shape = original;
  shape.segments[0].rows = 513;
  EXPECT_FALSE(domain->CheckShape(shape));
  shape = original;
  shape.phase = pl::Phase::kDecode;
  EXPECT_FALSE(domain->CheckShape(shape));
  std::array shapes{original, original};
  shapes[1].stage = 1;
  EXPECT_TRUE(domain->CheckUse(en::NumericalUse::kPromptCompanions, shapes));
  // Adaptive policy choosing plain Single today cannot authorize this use.
  EXPECT_FALSE(domain->CheckUse(en::NumericalUse::kSpeculative, shapes));
  EXPECT_FALSE(domain->CheckUse(static_cast<en::NumericalUse>(255), shapes));
  EXPECT_FALSE(domain->CheckUse(en::NumericalUse::kPrompt, {}));
  shapes[1].row_invariant = true;
  EXPECT_FALSE(domain->CheckUse(en::NumericalUse::kPrompt, shapes));
}

TEST(ExecutionContractTest, MalformedDeclarationsRefuseBeforeThereIsAnIdentity) {
  EXPECT_FALSE(en::NumericalDomain::Create({}));
  auto stage = Stage(0, "target");
  stage.uses.push_back(stage.uses.front());
  EXPECT_FALSE(en::NumericalDomain::Create({stage}));
  stage = Stage(0, "target");
  stage.phases[1].phase = stage.phases[0].phase;
  EXPECT_FALSE(en::NumericalDomain::Create({stage}));
  stage = Stage(0, "target");
  stage.phases[0].max_owners = 0;
  EXPECT_FALSE(en::NumericalDomain::Create({stage}));
  stage = Stage(0, "target");
  stage.stage = 1;
  EXPECT_FALSE(en::NumericalDomain::Create({stage}));
}

TEST(ExecutionContractTest, IdentitySealsAllOrderedStagesAndSelectionFacts) {
  auto registry = ex::Registry::Create({Implementation()});
  ASSERT_TRUE(registry) << Failed(registry, &ex::PlanRejection::detail).value_or("");
  auto domain = en::NumericalDomain::Create({Stage(0, "target"), Stage(1, "companion")});
  ASSERT_TRUE(domain) << Failed(domain).value_or("");
  std::array inputs{Input(*registry, 0, "target"), Input(*registry, 1, "companion")};
  auto contract = en::ExecutionContract::Create(*domain, inputs);
  ASSERT_TRUE(contract) << Failed(contract).value_or("");
  const auto identity = contract->identity();
  const auto different = [&](auto change) {
    auto next = inputs;
    change(next[1]);
    auto changed = en::ExecutionContract::Create(*domain, next);
    ASSERT_TRUE(changed) << Failed(changed).value_or("");
    EXPECT_NE(changed->identity(), identity);
  };
  different([](auto& in) { ++in.policy.max_decode_rows; });
  different([](auto& in) { in.policy.row_invariant = true; });
  different([](auto& in) { in.policy.allow_lossy = true; });
  different([](auto& in) { in.patterns = Seal("another pattern version"); });
  different([](auto& in) { in.backend.recipe = Seal("another recipe"); });
  different([](auto& in) { in.backend.build = Seal("another actual compiler flag"); });
  different([](auto& in) { in.backend.selection = Seal("another backend option"); });
  different([](auto& in) { ++in.device.arch; });
  different([](auto& in) { in.device.features |= ex::kFp8Mma; });
  different([](auto& in) { in.device.device_class = "another_class"; });
  different([](auto& in) { in.device.tuned = {{"fake.score", 1, 128, 256}}; });
  different([](auto& in) { in.reservation_alignment = 0; });
  different([](auto& in) { in.placement_alignment = 512; });
  auto revised = ex::Registry::Create({Implementation("fake.score", "v2")});
  ASSERT_TRUE(revised) << Failed(revised, &ex::PlanRejection::detail).value_or("");
  different([&](auto& in) { in.registry = &*revised; });
  auto reordered = inputs;
  std::swap(reordered[0], reordered[1]);
  // Numeric stage/name order is authenticated, not sorted away.
  EXPECT_FALSE(en::ExecutionContract::Create(*domain, reordered));
  auto changed_stages = std::vector{Stage(0, "companion"), Stage(1, "target")};
  auto reordered_domain = en::NumericalDomain::Create(std::move(changed_stages));
  ASSERT_TRUE(reordered_domain) << Failed(reordered_domain).value_or("");
  reordered[0].stage = 0;
  reordered[1].stage = 1;
  auto ordered_again = en::ExecutionContract::Create(*reordered_domain, reordered);
  ASSERT_TRUE(ordered_again) << Failed(ordered_again).value_or("");
  EXPECT_NE(ordered_again->identity(), identity);
  // The factory owns the result; input edits cannot rewrite a live contract.
  inputs[1].backend.build = {};
  EXPECT_EQ(contract->identity(), identity);
  EXPECT_FALSE(en::ExecutionContract::Create(*domain, inputs));
}

TEST(ExecutionContractTest, TuningOutputPrecisionAndPresenceHaveDistinctIdentities) {
  auto registry = ex::Registry::Create({Implementation()});
  ASSERT_TRUE(registry) << Failed(registry, &ex::PlanRejection::detail).value_or("");
  auto domain = en::NumericalDomain::Create({Stage(0, "target")});
  ASSERT_TRUE(domain) << Failed(domain).value_or("");
  auto in = Input(*registry, 0, "target");
  in.device.tuned = {{"fake.score", 1, 128, 256}};
  const auto seal = [&]() { return en::ExecutionContract::Create(*domain, std::span(&in, 1)); };
  auto omitted = seal();
  ASSERT_TRUE(omitted) << Failed(omitted).value_or("");
  in.device.tuned[0].output_encoding = llmp::format::Encoding::kF16;
  auto half = seal();
  ASSERT_TRUE(half) << Failed(half).value_or("");
  in.device.tuned[0].output_encoding = llmp::format::Encoding::kF32;
  auto full = seal();
  ASSERT_TRUE(full) << Failed(full).value_or("");
  EXPECT_NE(omitted->identity(), half->identity());
  EXPECT_NE(omitted->identity(), full->identity());
  EXPECT_NE(half->identity(), full->identity());
  auto repeated = seal();
  ASSERT_TRUE(repeated) << Failed(repeated).value_or("");
  EXPECT_EQ(full->identity(), repeated->identity());
  in.device.tuned[0].output_encoding = static_cast<llmp::format::Encoding>(65535);
  EXPECT_FALSE(seal());
  EXPECT_NE(full->identity(), half->identity());
  in.device.tuned[0].output_encoding.reset();
  auto absent_again = seal();
  ASSERT_TRUE(absent_again) << Failed(absent_again).value_or("");
  EXPECT_EQ(omitted->identity(), absent_again->identity());
}

TEST(ExecutionContractTest, MissingAndMalformedBackendFactsNeverBecomeLegacyPermission) {
  auto registry = ex::Registry::Create({Implementation()});
  ASSERT_TRUE(registry) << Failed(registry, &ex::PlanRejection::detail).value_or("");
  auto domain = en::NumericalDomain::Create({Stage(0, "target")});
  ASSERT_TRUE(domain) << Failed(domain).value_or("");
  const auto refused = [&](auto change) {
    auto in = Input(*registry, 0, "target");
    change(in);
    EXPECT_FALSE(en::ExecutionContract::Create(*domain, std::span(&in, 1)));
  };
  refused([](auto& in) { in.backend.recipe = {}; });
  refused([](auto& in) { in.backend.build = {}; });
  refused([](auto& in) { in.backend.selection = {}; });
  refused([](auto& in) { in.patterns = {}; });
  refused([](auto& in) { in.registry = nullptr; });
  refused([](auto& in) { in.device.device_class.clear(); });
  refused([](auto& in) { in.reservation_alignment = 3; });
  refused([](auto& in) { in.placement_alignment = 0; });
  refused([](auto& in) { in.device.tuned = {{"unknown", 1, 128, 256}}; });
}

TEST(OwnedDeviceFactsTest, CallerMutationAndMovesCannotChangeRetainedTuning) {
  std::string name = "small";
  pl::DeviceFacts input{.device_class = "sm_121",
                        .arch = 121,
                        .features = ex::kBf16Mma,
                        .tuned = {{name, 8, 2560, 512, llmp::format::Encoding::kF32}}};
  auto bytes = pl::OwnedDeviceFacts::Bound(input);
  ASSERT_TRUE(bytes) << Failed(bytes).value_or("");
  EXPECT_FALSE(pl::OwnedDeviceFacts::Create(input, *bytes - 1));
  auto owned = pl::OwnedDeviceFacts::Create(input, *bytes);
  ASSERT_TRUE(owned) << Failed(owned).value_or("");
  name.assign(200, 'x');
  input.tuned.clear();
  input.device_class = "other";
  const auto* retained = owned->facts().tuned.front().implementation.data();
  auto moved = std::move(*owned);
  EXPECT_EQ(moved.facts().device_class, "sm_121");
  ASSERT_EQ(moved.facts().tuned.size(), 1);
  EXPECT_EQ(moved.facts().tuned.front().implementation, "small");
  EXPECT_EQ(moved.facts().tuned.front().implementation.data(), retained);
  EXPECT_EQ(moved.facts().tuned.front().output_encoding, llmp::format::Encoding::kF32);
}

TEST(OwnedDeviceFactsTest, DuplicateAndUnknownOutputKeysRefuseBeforeCopy) {
  pl::DeviceFacts input{.device_class = "sm_121",
                        .arch = 121,
                        .tuned = {{"native.product", 8, 2560, 512, llmp::format::Encoding::kF32}}};
  input.tuned.push_back(input.tuned.front());
  EXPECT_FALSE(pl::OwnedDeviceFacts::Bound(input));
  input.tuned.pop_back();
  input.tuned.front().output_encoding = static_cast<llmp::format::Encoding>(65535);
  EXPECT_FALSE(pl::OwnedDeviceFacts::Bound(input));
  input.tuned.front().output_encoding.reset();
  EXPECT_TRUE(pl::OwnedDeviceFacts::Bound(input));
  input.tuned.front().rows = 0;
  EXPECT_FALSE(pl::OwnedDeviceFacts::Bound(input));
}

// Test backend authority: production sources receive the real backend's owned
// declaration. Its mutable controls let failures occur at the setup boundary.
struct DeclaredFakeBackend final : llmp::kernels::BackendOwner {
  explicit DeclaredFakeBackend(const ex::Registry& registry) {
    table.registry = &registry;
    table.device = {.device_class = "fake", .arch = 121};
    table.lower = [](const llmp::kernels::LowerInput&)
        -> std::expected<std::unique_ptr<llmp::kernels::StepProgram>, std::string> {
      return std::unexpected("fake lowering not used by identity controls");
    };
  }
  const llmp::kernels::StepBackend& backend() const override { return table; }
  bool Charge(std::uint64_t bytes) const override {
    if (allowance) return allowance->Charge(bytes);
    if (bytes > ceiling - paid) return false;
    paid += bytes;
    return true;
  }
  void Release(std::uint64_t bytes) const override {
    if (allowance) {
      allowance->Release(bytes);
      return;
    }
    EXPECT_LE(bytes, paid);
    paid -= bytes;
  }
  std::expected<ex::BackendDeclaration, std::string> Ready() const override {
    ++queries;
    if (!ready) return std::unexpected("fake backend is unprepared");
    return declaration;
  }
  std::shared_ptr<en::ExecutionMetadataAllowance> allowance;
  llmp::kernels::StepBackend table;
  ex::BackendDeclaration declaration{Seal("recipe"), Seal("build"), Seal("config")};
  mutable std::uint64_t paid = 0, queries = 0;
  std::uint64_t ceiling = 1U << 20;
  bool ready = true;
};

struct SourceIdentityTest : ::testing::Test {
  void SetUp() override {
    auto made = ex::Registry::Create({Implementation()});
    ASSERT_TRUE(made);
    registry = std::make_unique<ex::Registry>(std::move(*made));
    spec.stages = {{.name = "target", .stream = {.width = 8}},
                   {.name = "companion", .stream = {.width = 8}}};
    auto declared = en::NumericalDomain::Create({Stage(0, "target"), Stage(1, "companion")});
    ASSERT_TRUE(declared);
    domain = std::make_shared<const en::NumericalDomain>(std::move(*declared));
    for (auto& owner : owners) owner = std::make_shared<DeclaredFakeBackend>(*registry);
  }
  std::expected<std::unique_ptr<en::ComponentStepSource>, std::string> Source(std::uint16_t stage) {
    auto table = owners[stage]->backend();
    table.owner = owners[stage];
    // A copied table cannot override the authority returned by its owner.
    table.device.device_class = "caller forged this copy";
    table.registry = nullptr;
    return en::ComponentStepSource::Create({.spec = &spec,
                                            .stage = stage,
                                            .bindings = {.state = &layout},
                                            .bound_digest = spec.Digest(),
                                            .slots = 2,
                                            .policy = pl::Policy{},
                                            .backend = std::move(table),
                                            .execution_domain = domain});
  }
  std::expected<std::unique_ptr<en::StagedStepSource>, std::string> Composition() {
    std::vector<std::unique_ptr<en::ComponentStepSource>> sources;
    for (std::uint16_t i = 0; i < 2; ++i) {
      auto source = Source(i);
      if (!source) return std::unexpected(source.error());
      sources.push_back(std::move(*source));
    }
    return en::StagedStepSource::Create(std::move(sources));
  }
  std::unique_ptr<ex::Registry> registry;
  llmp::spec::ModelSpec spec;
  llmp::model::StateLayoutDerived layout{};
  std::shared_ptr<const en::NumericalDomain> domain;
  std::array<std::shared_ptr<DeclaredFakeBackend>, 2> owners;
};

TEST(ExecutionMetadataAllowanceTest, RefusesBeforeCatalogOrAllocationWithoutHeadroom) {
  llmp::catalog::Catalog catalog;
  const auto domain = catalog.AddDomain("test");
  constexpr std::uint64_t bytes = 1U << 20, margin = 6ULL << 30;
  EXPECT_FALSE(en::ExecutionMetadataAllowance::Admit(catalog, domain, 0, bytes + margin, margin));
  EXPECT_FALSE(en::ExecutionMetadataAllowance::Admit(catalog, domain, bytes, 0, margin));
  EXPECT_FALSE(
      en::ExecutionMetadataAllowance::Admit(catalog, domain, bytes, bytes + margin - 1, margin));
  EXPECT_EQ(catalog.OccupancyOf(domain).Total().value(), 0U);
  auto admitted =
      en::ExecutionMetadataAllowance::Admit(catalog, domain, bytes, bytes + margin, margin);
  ASSERT_TRUE(admitted);
  EXPECT_EQ(catalog.OccupancyOf(domain).Total().value(), bytes);
  EXPECT_LT((*admitted)->capacity(), bytes);
  EXPECT_FALSE((*admitted)->Charge(bytes));
  EXPECT_TRUE((*admitted)->Charge((*admitted)->capacity()));
  EXPECT_FALSE((*admitted)->Charge(1));
  (*admitted)->Release((*admitted)->capacity() + 1);
  EXPECT_FALSE((*admitted)->Charge(1));
  (*admitted)->Release((*admitted)->capacity());
  EXPECT_EQ((*admitted)->used(), 0U);
  admitted->reset();
  // Node-lifetime payment, not a callback into a potentially live scheduler.
  EXPECT_EQ(catalog.OccupancyOf(domain).Total().value(), bytes);
}

TEST_F(SourceIdentityTest, ProductionAllowanceRejectsConstructionAndFundsRetryOnlyOnce) {
  llmp::catalog::Catalog catalog;
  const auto node = catalog.AddDomain("test");
  constexpr std::uint64_t bytes = 1U << 20, margin = 6ULL << 30;
  auto admitted =
      en::ExecutionMetadataAllowance::Admit(catalog, node, bytes, bytes + margin, margin);
  ASSERT_TRUE(admitted);
  owners[0]->allowance = *admitted;
  auto source = Composition();
  ASSERT_TRUE(source);
  const auto capacity = (*admitted)->capacity();
  ASSERT_TRUE((*admitted)->Charge(capacity));
  EXPECT_FALSE((*source)->PrepareExecution());
  EXPECT_EQ(owners[0]->queries, 0U);
  EXPECT_FALSE((*source)->stage(0).ExecutionIdentity());
  EXPECT_FALSE((*source)->stage(1).ExecutionIdentity());
  (*admitted)->Release(capacity);
  ASSERT_TRUE((*source)->PrepareExecution());
  EXPECT_GT((*admitted)->used(), 0U);
  EXPECT_EQ(catalog.OccupancyOf(node).Total().value(), bytes);
  source->reset();
  EXPECT_EQ((*admitted)->used(), 0U);
  source = Composition();
  ASSERT_TRUE(source && (*source)->PrepareExecution());
  EXPECT_EQ(catalog.OccupancyOf(node).Total().value(), bytes);
  source->reset();
  EXPECT_EQ((*admitted)->used(), 0U);
}

TEST_F(SourceIdentityTest, FailedCompanionOrPaymentCannotPublishPartialIdentity) {
  auto source = Composition();
  ASSERT_TRUE(source) << Failed(source).value_or("");
  EXPECT_FALSE((*source)->ExecutionIdentity());
  const auto key = (*source)->KeyBase();
  owners[0]->ceiling = 0;
  EXPECT_FALSE((*source)->PrepareExecution());
  EXPECT_EQ(owners[0]->queries, 0U);
  EXPECT_EQ(owners[0]->paid, 0U);
  owners[0]->ceiling = 1U << 20;
  owners[1]->ready = false;
  EXPECT_FALSE((*source)->PrepareExecution());
  EXPECT_EQ(owners[0]->paid, 0U);
  EXPECT_FALSE((*source)->stage(0).ExecutionIdentity());
  EXPECT_FALSE((*source)->stage(1).ExecutionIdentity());
  EXPECT_EQ((*source)->KeyBase(), key);
  owners[1]->ready = true;
  ASSERT_TRUE((*source)->PrepareExecution());
  EXPECT_GT(owners[0]->paid, 0U);
  auto identity = (*source)->ExecutionIdentity();
  ASSERT_TRUE(identity && identity->has_value());
  EXPECT_NE((*source)->KeyBase().patterns, key.patterns);
  EXPECT_FALSE((*source)->PrepareExecution());
  EXPECT_EQ((*source)->ExecutionIdentity(), identity);
  source->reset();
  EXPECT_EQ(owners[0]->paid, 0U);
}

TEST_F(SourceIdentityTest, CompleteSourceScopesShapeModeAndCompanionConfiguration) {
  auto source = Composition();
  ASSERT_TRUE(source);
  ASSERT_TRUE((*source)->PrepareExecution());
  auto identity = (*source)->ExecutionIdentity();
  ASSERT_TRUE(identity && identity->has_value());
  EXPECT_TRUE((*source)->CheckExecutionShape(Shape()));
  EXPECT_TRUE((*source)->CheckExecutionUse(en::NumericalUse::kPrompt));
  EXPECT_FALSE((*source)->CheckExecutionUse(en::NumericalUse::kSpeculative));
  auto wide = Shape();
  wide.segments.front().rows = 513;
  EXPECT_FALSE((*source)->CheckExecutionShape(wide));
  const auto saved = owners[1]->declaration;
  owners[1]->declaration.selection = Seal("changed companion config");
  EXPECT_FALSE((*source)->ExecutionIdentity());
  EXPECT_FALSE((*source)->CheckExecutionShape(Shape()));
  EXPECT_FALSE((*source)->Build(Shape(), {}));
  owners[1]->declaration = saved;
  EXPECT_EQ((*source)->ExecutionIdentity(), identity);
  owners[1]->ready = false;
  EXPECT_FALSE((*source)->CheckExecutionUse(en::NumericalUse::kPrompt));
}

TEST_F(SourceIdentityTest, OmittingACompanionOrMisnamingItsDomainRefuses) {
  auto target = Source(0);
  ASSERT_TRUE(target);
  EXPECT_FALSE((*target)->PrepareExecution());
  EXPECT_EQ(owners[0]->paid, 0U);
  spec.stages[1].name = "another companion";
  auto both = Composition();
  ASSERT_TRUE(both);
  EXPECT_FALSE((*both)->PrepareExecution());
  EXPECT_EQ(owners[0]->paid, 0U);
}

}  // namespace
