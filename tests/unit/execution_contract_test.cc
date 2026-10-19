// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/execution_contract.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <thread>
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
          .operation = llmp::ir::Primitive(llmp::ir::Op::kIndexScore),
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
  different([](auto& in) { in.device.device_model = "actual GPU model"; });
  different([](auto& in) { ++in.device.multiprocessors; });
  different([](auto& in) { in.device.preferred = {"fake.score"}; });
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
  in.device.tuned[0].weight_bits = 4;
  auto rate = seal();
  ASSERT_TRUE(rate);
  EXPECT_NE(full->identity(), rate->identity());
  in.device.tuned[0].weight_bits = 0;
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

TEST(SharedCompilerAllowanceTest, OneMaximumExtentPaysOverlapsAndRejectsForeignThreads) {
  llmp::catalog::Catalog catalog, other;
  const auto domain = catalog.AddDomain("node"), foreign = other.AddDomain("node");
  constexpr std::uint64_t bytes = 1U << 20, cap = 4 * bytes, margin = bytes;
  en::SharedCompilerAllowance compiler;
  EXPECT_FALSE(compiler.Acquire(1));
  EXPECT_FALSE(compiler.Reserve(catalog, domain, bytes, bytes - 1, cap, margin));
  EXPECT_EQ(catalog.OccupancyOf(domain).Total().value(), 0U);
  ASSERT_TRUE(compiler.Reserve(catalog, domain, bytes, cap, cap, margin));
  ASSERT_TRUE(compiler.Reserve(catalog, domain, bytes, cap, cap, margin));
  EXPECT_EQ(catalog.ExtentsOf(domain).size(), 1U);
  EXPECT_EQ(compiler.paid_bytes(), bytes);
  EXPECT_FALSE(compiler.Reserve(other, foreign, bytes, cap, cap, margin));
  const auto usable = bytes - en::ExecutionMetadataAllowance::OverheadBytes();
  {
    auto first = compiler.Acquire(usable / 2);
    ASSERT_TRUE(first);
    auto second = compiler.Acquire(usable - usable / 2);
    ASSERT_TRUE(second);
    EXPECT_FALSE(compiler.Acquire(1));
    EXPECT_FALSE(compiler.Reserve(catalog, domain, 2 * bytes, cap, cap, margin));
  }
  // Refused nested/foreign use does not spend capacity. Growth keeps one owner.
  bool acquired = true, reserved = true;
  std::thread worker([&] {
    acquired = compiler.Acquire(1).has_value();
    reserved = compiler.Reserve(catalog, domain, 2 * bytes, cap, cap, margin).has_value();
  });
  worker.join();
  EXPECT_FALSE(acquired);
  EXPECT_FALSE(reserved);
  ASSERT_TRUE(compiler.Reserve(catalog, domain, 2 * bytes, cap, cap, margin));
  EXPECT_EQ(catalog.ExtentsOf(domain).size(), 1U);
  EXPECT_EQ(catalog.OccupancyOf(domain).Total().value(), 2 * bytes);
  EXPECT_EQ(compiler.paid_bytes(), 2 * bytes);
  EXPECT_TRUE(compiler.Acquire(2 * bytes - en::ExecutionMetadataAllowance::OverheadBytes()));
}

TEST_F(SourceIdentityTest, SharedCompilerChargePrecedesGraphAndReturnsOnBuildRefusal) {
  llmp::catalog::Catalog catalog;
  const auto node = catalog.AddDomain("node");
  constexpr std::uint64_t bytes = 1U << 20;
  auto compiler = std::make_shared<en::SharedCompilerAllowance>();
  ASSERT_TRUE(compiler->Reserve(catalog, node, bytes, 2 * bytes, 2 * bytes, bytes));
  const auto usable = bytes - en::ExecutionMetadataAllowance::OverheadBytes();
  spec.stages.resize(1);
  auto declared = en::NumericalDomain::Create({Stage(0, "target")});
  ASSERT_TRUE(declared);
  auto table = owners[0]->backend();
  table.owner = owners[0];
  auto source = en::ComponentStepSource::Create(
      {.spec = &spec,
       .stage = 0,
       .bindings = {.state = &layout},
       .bound_digest = spec.Digest(),
       .slots = 2,
       .policy = pl::Policy{},
       .backend = std::move(table),
       .execution_domain = std::make_shared<const en::NumericalDomain>(std::move(*declared)),
       .compiler = compiler,
       .compiler_bytes = usable});
  ASSERT_TRUE(source) << Failed(source).value_or("");
  ASSERT_TRUE((*source)->PrepareExecution());
  {
    auto held = compiler->Acquire(usable);
    ASSERT_TRUE(held);
    auto program = (*source)->Build(Shape(), {});
    ASSERT_FALSE(program);
    EXPECT_EQ(program.error(), "compiler construction exceeds its shared payment");
  }
  // This deliberately incomplete spec fails BuildStep. Its paid temporary
  // compiler use must return even on that early graph-construction failure.
  auto program = (*source)->Build(Shape(), {});
  ASSERT_FALSE(program);
  EXPECT_NE(program.error(), "compiler construction exceeds its shared payment");
  EXPECT_TRUE(compiler->Acquire(usable));
}

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

TEST(ExecutionMetadataAllowanceTest, PreparedOwnerRequiresItsStillHeldCatalogPayment) {
  llmp::catalog::Catalog catalog, other;
  const auto domain = catalog.AddDomain("node"), foreign = other.AddDomain("node");
  constexpr std::uint64_t bytes = 1U << 20;
  auto owner = en::ExecutionMetadataAllowance::Admit(catalog, domain, bytes, 2 * bytes, bytes);
  ASSERT_TRUE(owner);
  ASSERT_TRUE((*owner)->HeldBy(catalog, domain));
  EXPECT_FALSE((*owner)->HeldBy(other, foreign));  // even matching numeric IDs are foreign
  EXPECT_FALSE((*owner)->HeldBy(catalog, catalog.AddDomain("different domain")));
  const auto extents = catalog.ExtentsOf(domain);
  ASSERT_EQ(extents.size(), 1U);
  ASSERT_TRUE(catalog.ReleasePinned(extents[0].first));
  ASSERT_TRUE(catalog.RemoveExtent(extents[0].first));
  EXPECT_FALSE((*owner)->HeldBy(catalog, domain));
}

TEST(ExecutionMetadataAllowanceTest, RetainedGrowthPreservesIdentityAndRefusalPayment) {
  llmp::catalog::Catalog catalog;
  const auto domain = catalog.AddDomain("node");
  constexpr std::uint64_t unit = 1U << 20;
  auto admitted = en::ExecutionMetadataAllowance::Admit(catalog, domain, unit, 8 * unit, unit);
  ASSERT_TRUE(admitted);
  auto& owner = **admitted;
  ASSERT_TRUE(owner.Charge(unit / 2));
  const auto extent = catalog.ExtentsOf(domain).front().first;
  const auto before = catalog.OccupancyOf(domain).Total();
  EXPECT_FALSE(owner.Grow(catalog, domain, 2 * unit, 2 * unit, 8 * unit, unit));
  EXPECT_FALSE(owner.Grow(catalog, domain, 2 * unit, 8 * unit, unit, unit));
  EXPECT_EQ(catalog.OccupancyOf(domain).Total(), before);
  EXPECT_EQ(owner.used(), unit / 2);
  auto closure = catalog.ClosureOfExtents(std::array{extent});
  ASSERT_TRUE(closure);
  auto lease = catalog.AcquireLease(*closure);
  ASSERT_TRUE(lease);
  EXPECT_FALSE(owner.Grow(catalog, domain, 2 * unit, 8 * unit, 8 * unit, unit));
  ASSERT_TRUE(catalog.ReleaseLease(*lease));
  ASSERT_TRUE(owner.Grow(catalog, domain, 2 * unit, 8 * unit, 8 * unit, unit));
  EXPECT_FALSE(catalog.AcquireLease(*closure));
  EXPECT_EQ(catalog.ExtentsOf(domain).size(), 1U);
  EXPECT_EQ(catalog.ExtentsOf(domain).front().first, extent);
  EXPECT_EQ(catalog.OccupancyOf(domain).Total().value(), 2 * unit + owner.OverheadBytes());
  for (unsigned i = 0; i < 33; ++i) {
    ASSERT_TRUE(owner.Grow(catalog, domain, unit, 8 * unit, 8 * unit, unit));
    ASSERT_TRUE(owner.Charge(unit));
    owner.Release(unit);
  }
  EXPECT_EQ(catalog.ExtentsOf(domain).size(), 1U);
  EXPECT_EQ(owner.used(), unit / 2);
  EXPECT_TRUE(owner.HeldBy(catalog, domain));
}

TEST(ExecutionMetadataAllowanceTest, InitialBootstrapHonorsBudgetBeforeAnyCatalogMutation) {
  llmp::catalog::Catalog catalog;
  const auto domain = catalog.AddDomain("node");
  constexpr std::uint64_t unit = 1U << 20;
  EXPECT_FALSE(en::ExecutionMetadataAllowance::AdmitWithin(catalog, domain, unit, unit - 1,
                                                           16 * unit, unit));
  EXPECT_EQ(catalog.OccupancyOf(domain).Total().value(), 0U);
  auto first =
      en::ExecutionMetadataAllowance::AdmitWithin(catalog, domain, unit, 2 * unit, 16 * unit, unit);
  ASSERT_TRUE(first);
  EXPECT_FALSE(en::ExecutionMetadataAllowance::AdmitWithin(catalog, domain, unit, 2 * unit - 1,
                                                           16 * unit, unit));
  EXPECT_EQ(catalog.ExtentsOf(domain).size(), 1U);
  EXPECT_EQ(catalog.OccupancyOf(domain).Total().value(), unit);
}

TEST(ExecutionMetadataAllowanceTest, ActualChargeGrowthFundsPeakAndRetriesWithoutNewExtents) {
  llmp::catalog::Catalog catalog;
  const auto domain = catalog.AddDomain("node");
  constexpr std::uint64_t unit = 1U << 20;
  auto admitted = en::ExecutionMetadataAllowance::Admit(catalog, domain, unit, 16 * unit, unit);
  ASSERT_TRUE(admitted);
  auto& owner = **admitted;
  const auto extent = catalog.ExtentsOf(domain).front().first;
  ASSERT_TRUE(owner.Charge(unit / 2));
  const auto initial = owner.capacity();
  // A declaration's temporary copy is live when its retained facts are paid.
  ASSERT_TRUE(owner.ChargeOrGrow(catalog, domain, unit, 16 * unit, 16 * unit, unit));
  EXPECT_EQ(owner.used(), 3 * unit / 2);
  EXPECT_EQ(owner.capacity(), owner.used());
  EXPECT_EQ(owner.peak(), owner.used());
  const auto occupied = catalog.OccupancyOf(domain).Total();
  EXPECT_FALSE(owner.ChargeOrGrow(catalog, domain, 2 * unit, 3 * unit, 16 * unit, unit));
  EXPECT_FALSE(owner.ChargeOrGrow(catalog, domain, 2 * unit, 16 * unit, 2 * unit, unit));
  EXPECT_FALSE(owner.ChargeOrGrow(catalog, domain, UINT64_MAX, 16 * unit, 16 * unit, unit));
  EXPECT_EQ(catalog.OccupancyOf(domain).Total(), occupied);
  EXPECT_EQ(owner.capacity(), 3 * unit / 2);
  EXPECT_EQ(owner.used(), 3 * unit / 2);
  EXPECT_EQ(owner.peak(), 3 * unit / 2);
  ASSERT_TRUE(owner.ChargeOrGrow(catalog, domain, 2 * unit, 16 * unit, 16 * unit, unit));
  owner.Release(unit);  // Destroy the temporary copy, retaining the declaration.
  EXPECT_EQ(owner.used(), 5 * unit / 2);
  EXPECT_EQ(owner.peak(), 7 * unit / 2);
  EXPECT_EQ(owner.capacity(), 7 * unit / 2);
  EXPECT_GT(owner.capacity(), initial);
  // Retrying construction reuses the paid high-water extent.
  for (unsigned retry = 0; retry < 3; ++retry) {
    ASSERT_TRUE(owner.ChargeOrGrow(catalog, domain, unit, 16 * unit, 16 * unit, unit));
    owner.Release(unit);
  }
  EXPECT_EQ(catalog.ExtentsOf(domain).size(), 1U);
  EXPECT_EQ(catalog.ExtentsOf(domain).front().first, extent);
  EXPECT_EQ(catalog.OccupancyOf(domain).Total().value(), owner.capacity() + owner.OverheadBytes());
  EXPECT_EQ(owner.peak(), 7 * unit / 2);
  owner.Release(owner.used());
  EXPECT_EQ(owner.used(), 0U);
  EXPECT_EQ(owner.peak(), 7 * unit / 2);
  ASSERT_TRUE(catalog.ReleasePinned(extent));
  ASSERT_TRUE(catalog.RemoveExtent(extent));
  EXPECT_FALSE(owner.ChargeOrGrow(catalog, domain, 1, 16 * unit, 16 * unit, unit));
  EXPECT_EQ(owner.used(), 0U);
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

TEST_F(SourceIdentityTest, RoundedIndexerRequiresOwnedPreparedBoundedOrdinaryDomain) {
  spec.stages.resize(1);
  layout.context = 256;
  layout.max_rows = 8;
  auto stage = Stage(0, "target");
  stage.uses.pop_back();  // No companion operation is being admitted.
  for (auto& phase : stage.phases) {
    phase.max_owners = 1;
    phase.max_owner_rows = phase.phase == pl::Phase::kDecode ? 1 : 8;
    phase.max_total_rows = phase.max_owner_rows;
  }
  const auto make = [&](const en::NumericalStage& declared, bool owner, bool rounded) {
    auto domain_value = en::NumericalDomain::Create({declared});
    EXPECT_TRUE(domain_value);
    auto table = owners[0]->backend();
    if (owner) table.owner = owners[0];
    return en::ComponentStepSource::Create(
        {.spec = &spec,
         .bindings = {.state = &layout},
         .bound_digest = spec.Digest(),
         .slots = 1,
         .policy = rounded ? pl::Policy{.enabled = {"rounded_indexer"}} : pl::Policy{},
         .backend = std::move(table),
         .execution_domain = owner && domain_value ? std::make_shared<const en::NumericalDomain>(
                                                         std::move(*domain_value))
                                                   : nullptr});
  };
  EXPECT_FALSE(make(stage, false, true));
  for (int bad = 0; bad != 5; ++bad) {
    auto changed = stage;
    if (bad == 0) changed.uses.push_back(en::NumericalUse::kPromptCompanions);
    if (bad == 1) changed.phases[0].max_owners = 2;
    if (bad == 2) changed.phases[0].max_owner_rows = 9;
    if (bad == 3) changed.phases[0].row_invariant = true;
    if (bad == 4) changed.name = "another-target";
    EXPECT_FALSE(make(changed, true, true));
  }
  auto opted = make(stage, true, true);
  ASSERT_TRUE(opted) << Failed(opted).value_or("");
  pl::StepShape shape{.segments = {{.rows = 4, .past = 2, .read_cells = 256}}};
  EXPECT_FALSE((*opted)->CheckExecutionShape(shape));
  owners[0]->ready = false;
  EXPECT_FALSE((*opted)->PrepareExecution());
  owners[0]->ready = true;
  ASSERT_TRUE((*opted)->PrepareExecution());
  EXPECT_TRUE((*opted)->CheckExecutionShape(shape));
  for (auto use : {en::NumericalUse::kPrompt, en::NumericalUse::kDecode, en::NumericalUse::kScore})
    EXPECT_TRUE((*opted)->CheckExecutionUse(use));
  EXPECT_FALSE((*opted)->CheckExecutionUse(en::NumericalUse::kSpeculative));
  EXPECT_FALSE((*opted)->CommitBuffers(0, 1));
  for (int bad = 0; bad != 5; ++bad) {
    auto changed = shape;
    if (bad == 0) changed.phase = pl::Phase::kVerify;
    if (bad == 1) changed.segments.front().slot = 1;
    if (bad == 2) changed.segments.front().past = 255;
    if (bad == 3) changed.segments.front().read_cells = 512;
    if (bad == 4) changed.segments.front().rows = 9;
    EXPECT_FALSE((*opted)->CheckExecutionShape(changed));
    EXPECT_FALSE((*opted)->Build(changed, {}));
  }
  auto ordinary = make(stage, true, false);
  ASSERT_TRUE(ordinary);
  ASSERT_TRUE((*ordinary)->PrepareExecution());
  EXPECT_NE((*ordinary)->KeyBase(), (*opted)->KeyBase());
  EXPECT_NE((*ordinary)->ExecutionIdentity(), (*opted)->ExecutionIdentity());
  // Existing snapshot/kept controls reject unequal execution identities before copying.
  owners[0]->ready = false;
  EXPECT_FALSE((*opted)->CheckExecutionShape(shape));
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

struct JoinedSourceIdentityTest : SourceIdentityTest {
  void Configure() {
    namespace md = llmp::model;
    spec.stages[0].role = llmp::spec::StageRole::kTarget;
    spec.stages[1].role = llmp::spec::StageRole::kDrafter;
    for (auto& stage : spec.stages) stage.context = 512;
    md::StateContract target{.tensors = {{.name = "kv",
                                          .kind = md::StateKind::kKv,
                                          .encoding = md::Encoding::kF32,
                                          .cell_bytes = 32,
                                          .cells = md::CellRule::kPerPosition}}};
    auto draft = target;
    draft.tensors[0].replayable = true;
    draft.tensors.push_back({.name = "features",
                             .kind = md::StateKind::kFeatures,
                             .encoding = md::Encoding::kF32,
                             .cell_bytes = 32,
                             .cells = md::CellRule::kPerRow,
                             .window_or_ratio = 1,
                             .rollback = md::Rollback::kSnapshot,
                             .target_writes = true});
    draft.tensors.push_back(
        md::DraftProgressControl("progress", md::DraftRelation::kTokenAfterCache));
    for (std::size_t i = 0; i < 2; ++i) {
      const std::array contracts{std::pair{0U, i == 0 ? target : draft}};
      auto made = md::Layout(contracts, 512, 16, 256);
      ASSERT_TRUE(made) << Failed(made).value_or("");
      layouts[i] = std::move(*made);
      layout_views[i] = &layouts[i];
    }
    declarations = {Stage(0, "target"), Stage(1, "companion")};
    for (auto& declared : declarations) {
      declared.uses = {en::NumericalUse::kPrompt, en::NumericalUse::kDecode,
                       en::NumericalUse::kPromptCompanions, en::NumericalUse::kSpeculative,
                       en::NumericalUse::kJoinedRepeatPending};
      declared.phases = declared.stage == 0
                            ? std::vector<en::NumericalPhase>{{.phase = pl::Phase::kPrefill,
                                                               .max_owners = 2,
                                                               .max_owner_rows = 16,
                                                               .max_total_rows = 32},
                                                              {.phase = pl::Phase::kDecode,
                                                               .max_owners = 2,
                                                               .max_owner_rows = 1,
                                                               .max_total_rows = 2},
                                                              {.phase = pl::Phase::kVerify,
                                                               .max_owners = 2,
                                                               .max_owner_rows = 3,
                                                               .max_total_rows = 6,
                                                               .row_invariant = true},
                                                              {.phase = pl::Phase::kCommit,
                                                               .max_owners = 2,
                                                               .max_owner_rows = 3,
                                                               .max_total_rows = 6}}
                            : std::vector<en::NumericalPhase>{{.phase = pl::Phase::kDraft,
                                                               .max_owners = 2,
                                                               .max_owner_rows = 16,
                                                               .max_total_rows = 32}};
    }
  }
  std::expected<std::unique_ptr<en::StagedStepSource>, std::string> Joined(int fault = 0) {
    auto made = en::NumericalDomain::Create(declarations);
    if (!made) return std::unexpected(made.error());
    domain = std::make_shared<const en::NumericalDomain>(std::move(*made));
    divergent = spec;
    divergent.stages[0].context += 1;
    std::vector<std::unique_ptr<en::ComponentStepSource>> sources;
    for (std::uint16_t i = 0; i < 2; ++i) {
      auto source = en::ComponentStepSource::Create(
          {.spec = fault == 1 && i == 1 ? &divergent : &spec,
           .stage = i,
           .bindings = {.tensors = tensors,
                        .state = &layouts[i],
                        .stages = layout_views,
                        .artifacts =
                            fault == 2
                                ? std::span<const std::span<const llmp::components::BoundTensor>>{}
                                : std::span<const std::span<const llmp::components::BoundTensor>>(
                                      artifacts)},
           .bound_digest = spec.Digest(),
           .slots = 2,
           .policy = pl::Policy{},
           .backend = {.owner = owners[i]},
           .execution_domain = domain});
      if (!source) return std::unexpected(source.error());
      sources.push_back(std::move(*source));
    }
    return en::StagedStepSource::Create(std::move(sources));
  }
  std::array<llmp::model::StateLayoutDerived, 2> layouts;
  std::array<const llmp::model::StateLayoutDerived*, 2> layout_views;
  std::vector<llmp::components::BoundTensor> tensors;
  std::array<std::span<const llmp::components::BoundTensor>, 1> artifacts{tensors};
  std::vector<en::NumericalStage> declarations;
  llmp::spec::ModelSpec divergent;
};

TEST_F(JoinedSourceIdentityTest, JoinedReplayRequiresWholePreparedOwnerAndActualShapePermission) {
  Configure();
  auto both = Joined();
  ASSERT_TRUE(both) << Failed(both).value_or("");
  EXPECT_FALSE((*both)->JoinedRepeatPending(0));
  EXPECT_FALSE((*both)->JoinedRepeatPending(1));
  ASSERT_TRUE((*both)->PrepareExecution());
  EXPECT_TRUE((*both)->JoinedRepeatPending(0));
  EXPECT_TRUE((*both)->JoinedRepeatPending(1));
  EXPECT_FALSE((*both)->JoinedRepeatPending(2));
  pl::StepShape shape{.stage = 1,
                      .phase = pl::Phase::kDraft,
                      .segments = {{.slot = 0, .rows = 4}, {.slot = 1, .rows = 2}},
                      .read_align = 256};
  EXPECT_TRUE((*both)->CheckExecutionShape(shape));
  shape.segments[0].rows = 17;
  EXPECT_FALSE((*both)->CheckExecutionShape(shape));
  shape.segments[0].rows = 4;
  shape.phase = pl::Phase::kVerify;
  EXPECT_FALSE((*both)->CheckExecutionShape(shape));
  owners[1]->declaration.selection = Seal("changed actual backend");
  EXPECT_FALSE((*both)->JoinedRepeatPending(0));
  EXPECT_FALSE((*both)->JoinedRepeatPending(1));
}

TEST_F(JoinedSourceIdentityTest, PartialCompositionAndLayoutPromisesRefuseBeforePublication) {
  for (int mode = 0; mode != 10; ++mode) {
    Configure();
    if (mode == 0) declarations[1].uses.pop_back();
    if (mode == 1) declarations[0].uses.erase(declarations[0].uses.begin() + 3);
    if (mode == 2) declarations[1].phases[0].phase = pl::Phase::kDecode;
    if (mode == 3) declarations[0].phases.pop_back();
    if (mode == 4) spec.stages[1].role = llmp::spec::StageRole::kEncoder;
    if (mode == 5) layout_views[1] = &layouts[0];
    if (mode == 6) layouts[1].items[0].contract.replayable = false;
    if (mode == 7) declarations[1].phases[0].max_owner_rows = 17;
    auto both = Joined(mode == 8 ? 1 : mode == 9 ? 2 : 0);
    ASSERT_TRUE(both) << mode << ": " << Failed(both).value_or("");
    const auto refused = (*both)->PrepareExecution();
    EXPECT_FALSE(refused) << mode;
    EXPECT_FALSE((*both)->JoinedRepeatPending(0));
    EXPECT_FALSE((*both)->JoinedRepeatPending(1));
    EXPECT_EQ(owners[0]->paid, 0U) << mode;
    EXPECT_EQ(owners[1]->paid, 0U) << mode;
    EXPECT_EQ(owners[0]->queries, 0U) << mode;
    EXPECT_EQ(owners[1]->queries, 0U) << mode;
  }
}

TEST_F(JoinedSourceIdentityTest, ExistingDomainDoesNotAcquireJoinedPermissionImplicitly) {
  Configure();
  for (auto& stage : declarations) stage.uses.pop_back();
  auto ordinary = Joined();
  ASSERT_TRUE(ordinary);
  ASSERT_TRUE((*ordinary)->PrepareExecution());
  EXPECT_FALSE((*ordinary)->JoinedRepeatPending(0));
  EXPECT_FALSE((*ordinary)->JoinedRepeatPending(1));
}

}  // namespace
