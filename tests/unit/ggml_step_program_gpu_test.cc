// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Native construction admission through the real backend. No program runs:
// arena allocations are observed, and the production catalog allowance pays
// both backend controls and native loans independently of numerical domains.
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "components/step.h"
#include "engine/execution_contract.h"
#include "engine/graph_plan.h"
#include "engine/paged_node.h"
#include "engine/runner_resources.h"
#include "format/encoding.h"
#include "ggml.h"
#include "ir/builder.h"
#include "kernels/ggml/step_program.h"
#include "platform/host_probe.h"

namespace {
// Watch only the descriptor arena's exact byte-array request, not diagnostic
// strings or the test framework's allocations. Fixed slots allocate nothing.
struct ArenaWatch {
  std::size_t bytes = 0, allocations = 0, live = 0;
  std::size_t report_bytes = 0, report_allocations = 0;
  std::array<void*, 8> blocks{};
};
thread_local ArenaWatch* arena_watch = nullptr;
struct WatchScope {
  explicit WatchScope(ArenaWatch& watch) { arena_watch = &watch; }
  ~WatchScope() { arena_watch = nullptr; }
};
}  // namespace

void* operator new[](std::size_t bytes) {
  void* pointer = std::malloc(bytes == 0 ? 1 : bytes);
  if (!pointer) std::abort();
  if (arena_watch && bytes == arena_watch->report_bytes) ++arena_watch->report_allocations;
  if (arena_watch && bytes == arena_watch->bytes) {
    ++arena_watch->allocations;
    ++arena_watch->live;
    bool recorded = false;
    for (auto& block : arena_watch->blocks) {
      if (!block) {
        block = pointer;
        recorded = true;
        break;
      }
    }
    if (!recorded) std::abort();
  }
  return pointer;
}
void operator delete[](void* pointer) noexcept {
  if (arena_watch) {
    for (auto& block : arena_watch->blocks) {
      if (block != pointer || !pointer) continue;
      block = nullptr;
      --arena_watch->live;
      break;
    }
  }
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete[](pointer); }

namespace {
namespace co = llmp::components;
namespace en = llmp::engine;
namespace ir = llmp::ir;
namespace kg = llmp::kernels::ggml;
namespace kn = llmp::kernels;
namespace pl = llmp::planner;
constexpr std::uint64_t kMiB = 1ULL << 20U;

struct Probe {
  std::shared_ptr<en::ExecutionMetadataAllowance> paid;
  std::shared_ptr<ArenaWatch> watch = std::make_shared<ArenaWatch>();
  std::uint64_t native_bytes = 0, requests = 0, native_refunds = 0;
  bool refuse = false, transferring = false, refuse_report = false;
  std::size_t retained_arenas = 0;
  bool Charge(std::uint64_t bytes) {
    ++requests;
    return !refuse && !(refuse_report && bytes != native_bytes) && paid->Charge(bytes);
  }
  void Release(std::uint64_t bytes) {
    EXPECT_LE(bytes, paid->used());
    if (native_bytes != 0 && bytes == native_bytes) {
      ++native_refunds;
      // Destruction without transfer must free the actual arena first.
      if (!transferring) EXPECT_EQ(watch->live, retained_arenas);
    }
    paid->Release(bytes);
  }
};
struct CompilerPayment {
  std::shared_ptr<en::ExecutionMetadataAllowance> paid;
  std::uint64_t bytes = 0;
  ~CompilerPayment() { paid->Release(bytes); }
};

// Positive teardown uses the normal node fence protocol even though tests
// submit no program work. An unproven teardown retains the complete owner.
struct Life final : en::PagedModel {
  en::PagedNode node{{.compute_streams = 1, .slots = 1}};
  en::RunnerResources resources{node, 0, 0};
  std::shared_ptr<Probe> probe;
  std::unique_ptr<kg::GgmlBackend> backend;
  bool opened = false;
  llmp::catalog::Closure empty;
  std::uint32_t stream() const override { return 0; }
  const llmp::catalog::Closure& fence_closure() const override { return empty; }
  std::vector<llmp::catalog::ExtentId> managed_extents() const override { return {}; }
  en::Status Release() override {
    backend.reset();
    std::vector<std::string> errors;
    resources.Release(errors);
    if (!errors.empty()) return std::unexpected(errors.front());
    return {};
  }
};
struct Retire {
  void operator()(Life* life) const {
    if (!life) return;
    if (life->opened) {
      const std::array<en::PagedModel*, 1> models{life};
      auto retired = life->node.TearDown(models);
      EXPECT_TRUE(retired) << (retired ? "" : retired.error());
      if (!retired) return;
    }
    delete life;
  }
};

class GgmlConstructionGpu : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    life.reset(new Life);
    auto opened = life->node.Open();
    ASSERT_TRUE(opened) << (opened ? "" : opened.error());
    life->opened = true;
    const auto available = llmp::platform::AvailableMemoryBytes();
    ASSERT_TRUE(available);
    auto paid = en::ExecutionMetadataAllowance::Admit(life->node.catalog(), life->node.domain(),
                                                      16 * kMiB, *available, 6ULL << 30U);
    ASSERT_TRUE(paid) << (paid ? "" : paid.error());
    life->probe = std::make_shared<Probe>();
    life->probe->paid = *paid;
    auto made = kg::GgmlBackend::Create(0, life->node.execution(), life->node.stream(0), nullptr);
    ASSERT_TRUE(made) << (made ? "" : made.error());
    life->backend = std::move(*made);
  }
  void Fund(bool declare, bool declare_first = false) {
    const auto charge = [probe = life->probe](std::uint64_t bytes) { return probe->Charge(bytes); };
    const auto release = [probe = life->probe](std::uint64_t bytes) { probe->Release(bytes); };
    if (declare) {
      auto opened = life->resources.OpenLibrary("construction test declaration workspace");
      ASSERT_TRUE(opened) << (opened ? "" : opened.error());
      if (declare_first)
        ASSERT_TRUE(life->backend->Declare({}, life->resources.library_bytes(), charge, release));
    }
    ASSERT_TRUE(life->backend->FundConstruction(charge, release));
    EXPECT_FALSE(life->backend->FundConstruction(charge, release));
    if (declare) {
      if (!declare_first)
        ASSERT_TRUE(life->backend->Declare({}, life->resources.library_bytes(), charge, release));
      ASSERT_TRUE(life->backend->LendLibrary(&life->resources.library()));
    }
  }
  std::unique_ptr<Life, Retire> life;
};

TEST_P(GgmlConstructionGpu, MissingPartialAndRefusedFunderCannotAllocateNativeStorage) {
  const auto probe = life->probe;
  const auto charge = [probe](std::uint64_t bytes) { return probe->Charge(bytes); };
  const auto release = [probe](std::uint64_t bytes) { probe->Release(bytes); };
  EXPECT_FALSE(life->backend->FundConstruction(charge, {}));
  EXPECT_FALSE(life->backend->FundConstruction({}, release));
  EXPECT_EQ(probe->requests, 0U);
  probe->refuse = true;
  EXPECT_FALSE(life->backend->FundConstruction(charge, release));
  EXPECT_EQ(probe->paid->used(), 0U);
  probe->refuse = false;
  // A backend without a hook cannot lower even a valid input. Using registry()
  // for inspection does not lend a producer table or install an unpaid path.
  ir::Builder b;
  b.SetOwner(0);
  const auto x = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 1})));
  b.Output(b.Scale(x, 2));
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph);
  auto clone = graph->Clone();
  ASSERT_TRUE(clone) << clone.error().detail;
  pl::StepPlan plan{.graph = std::move(*clone),
                    .steps = {},
                    .regions = {},
                    .placement = {},
                    .applied = {},
                    .from = {},
                    .tensors = {},
                    .decomposed = {}};
  const co::GraphBuildIO io{*graph, graph->inputs(), graph->outputs(), {}};
  const kn::Addresses addresses{};
  const auto inventory = kg::LowerConstructionOf({plan, io, addresses});
  ASSERT_TRUE(inventory);
  probe->watch->bytes = inventory->descriptors * ggml_tensor_overhead();
  WatchScope watch(*probe->watch);
  const auto requests = probe->requests;
  auto missing = life->backend->Lower({plan, io, addresses});
  ASSERT_FALSE(missing);
  EXPECT_NE(missing.error().find("paid backend funder"), std::string::npos);
  EXPECT_EQ(probe->requests, requests);
  EXPECT_EQ(probe->watch->allocations, 0U);
  ASSERT_NO_FATAL_FAILURE(Fund(GetParam()));
}

TEST_P(GgmlConstructionGpu, LibraryLoanReturnsBeforeOwnerDestructionAndCanBeRelent) {
  ASSERT_NO_FATAL_FAILURE(Fund(GetParam()));
  for (unsigned cycle = 0; cycle < 2; ++cycle) {
    if (!GetParam() || cycle != 0)
      ASSERT_TRUE(life->resources.OpenLibrary("descriptor library loan"));
    ASSERT_TRUE(life->resources.OpenSelectedWorkspace("descriptor library source state",
                                                      life->backend->persistent_bytes()));
    ASSERT_TRUE(life->backend->LendLibrary(&life->resources.library()));
    {
      const auto& state = life->resources.selected_workspace();
      auto launch =
          life->backend->Launch({.base = 0, .size = llmp::base::Bytes(0)},
                                {.base = state.base, .bytes = life->backend->persistent_bytes()});
      ASSERT_TRUE(launch) << (launch ? "" : launch.error());
      launch->reset();
    }
    life->backend->LaunchReleased();
    std::vector<std::string> problems;
    life->resources.Release(problems);
    ASSERT_TRUE(problems.empty());
    // The same backend and descriptor context survive destruction of each owner.
  }
}

TEST_P(GgmlConstructionGpu, ReportsUseTheSamePayerWithoutRequiringADeclaration) {
  // Image construction declares first, then installs its independent payer.
  ASSERT_NO_FATAL_FAILURE(Fund(GetParam(), true));
  const auto probe = life->probe;
  const auto table = life->backend->Backend();
  const auto& backend = table.owner ? table.owner->backend() : table;
  ASSERT_TRUE(backend.construction_funder);
  ir::Builder b;
  b.SetOwner(0);
  const auto x = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 2})));
  b.Output(b.Scale(x, 2));
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph);
  const co::GraphBuildIO io{*graph, graph->inputs(), graph->outputs(), {}};
  const kn::Addresses addresses{};
  auto plan = en::PlanComponentGraph(io, backend, {}, addresses);
  ASSERT_TRUE(plan) << (plan ? "" : plan.error());
  const auto inventory = kg::LowerConstructionOf({*plan, io, addresses});
  ASSERT_TRUE(inventory);
  probe->native_bytes = inventory->bytes;
  auto report_bytes = pl::PlanReportInventory::StorageBytes(*plan, *backend.registry);
  ASSERT_TRUE(report_bytes);
  probe->watch->bytes = inventory->descriptors * ggml_tensor_overhead();
  probe->watch->report_bytes = *report_bytes;
  WatchScope watch(*probe->watch);
  const auto baseline = probe->paid->used();
  probe->refuse_report = true;
  // Force the independent table token for both declared and undeclared cases.
  auto independent = backend;
  independent.owner.reset();
  auto refused = en::BuildComponentProgram(io, independent, {}, addresses, {});
  ASSERT_FALSE(refused);
  EXPECT_NE(refused.error().find("report inventory could not be funded"), std::string::npos);
  EXPECT_GT(probe->watch->allocations, 0U);  // The separately funded native loan succeeded.
  EXPECT_EQ(probe->watch->report_allocations, 0U);
  EXPECT_EQ(probe->watch->live, 0U);
  EXPECT_EQ(probe->paid->used(), baseline);
  probe->refuse_report = false;
  auto result = en::BuildComponentProgram(io, independent, {}, addresses, {});
  ASSERT_TRUE(result) << (result ? "" : result.error());
  EXPECT_EQ(probe->watch->report_allocations, 1U);
  EXPECT_GT(probe->paid->used(), baseline + inventory->bytes);
  const auto retained = (*result)->host_bytes();
  ASSERT_TRUE(probe->paid->Charge(retained));
  probe->transferring = true;
  (*result)->ConstructionPaid();
  EXPECT_EQ(probe->paid->used(), baseline + retained);
  (*result)->ConstructionPaid();
  EXPECT_EQ(probe->paid->used(), baseline + retained);
  result->reset();
  EXPECT_EQ(probe->watch->live, 0U);
  probe->paid->Release(retained);
  EXPECT_EQ(probe->paid->used(), baseline);
}

TEST_P(GgmlConstructionGpu, ActualFundingTokenSurvivesDeclaredWrapperAndLateFunding) {
  // Image declarations precede funding; ordinary LLMs can fund first.
  ASSERT_NO_FATAL_FAILURE(Fund(GetParam(), true));
  const auto exported = life->backend->Backend();
  auto payer = exported.ConstructionFunding();
  ASSERT_TRUE(payer);
  ASSERT_TRUE(payer->charge);
  ASSERT_TRUE(payer->release);
  const auto before = life->probe->paid->used();
  ASSERT_TRUE(payer->charge(256));
  EXPECT_EQ(life->probe->paid->used(), before + 256);
  payer->release(256);
  EXPECT_EQ(life->probe->paid->used(), before);
  EXPECT_EQ(payer, exported.owner ? exported.owner->backend().construction_funder
                                  : exported.construction_funder);
  if (exported.owner) {
    auto changed = exported;
    changed.construction_funder = std::make_shared<const kn::ConstructionFunder>();
    EXPECT_EQ(changed.ConstructionFunding(), payer);
  }
}

TEST_P(GgmlConstructionGpu, LoansSurviveReturnAndTransferOnlyAfterRetainedPayment) {
  ASSERT_NO_FATAL_FAILURE(Fund(GetParam()));
  const auto probe = life->probe;
  const auto table = life->backend->Backend();
  const auto& backend = table.owner ? table.owner->backend() : table;
  ir::Builder b;
  b.SetOwner(0);
  const auto x = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({16, 2})));
  b.Output(b.Scale(x, 2));
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph);
  const co::GraphBuildIO io{*graph, graph->inputs(), graph->outputs(), {}};
  const kn::Addresses addresses{};
  auto plan = en::PlanComponentGraph(io, backend, {}, addresses);
  ASSERT_TRUE(plan) << (plan ? "" : plan.error());
  const auto inventory = kg::LowerConstructionOf({*plan, io, addresses});
  ASSERT_TRUE(inventory);
  probe->native_bytes = inventory->bytes;
  probe->watch->bytes = inventory->descriptors * ggml_tensor_overhead();
  WatchScope watch(*probe->watch);
  const auto baseline = probe->paid->used();
  probe->refuse = true;
  auto denied = life->backend->Lower({*plan, io, addresses});
  ASSERT_FALSE(denied);
  EXPECT_NE(denied.error().find("not funded"), std::string::npos);
  EXPECT_EQ(probe->watch->allocations, 0U);  // Admission precedes TensorArena allocation.
  EXPECT_EQ(probe->paid->used(), baseline);
  probe->refuse = false;
  const co::GraphBuildIO wrong_ports{*graph, {}, graph->outputs(), {}};
  EXPECT_FALSE(life->backend->Lower({*plan, wrong_ports, addresses}));
  EXPECT_EQ(probe->paid->used(), baseline);
  EXPECT_EQ(probe->watch->live, 0U);
  auto invalid = en::BuildComponentProgram(
      io, backend, {}, addresses, {}, std::nullopt,
      [](const kn::LowerInput&, const kn::StepProgram&) -> en::Status {
        return std::unexpected("deliberate validator refusal");
      },
      table.owner);
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error(), "deliberate validator refusal");
  EXPECT_GT(probe->watch->allocations, 0U);
  EXPECT_EQ(probe->watch->live, 0U);
  EXPECT_EQ(probe->paid->used(), baseline);
  {
    auto abandoned = life->backend->Lower({*plan, io, addresses});
    ASSERT_TRUE(abandoned) << (abandoned ? "" : abandoned.error());
    EXPECT_EQ(probe->watch->live, 1U);
    EXPECT_EQ(probe->paid->used(), baseline + inventory->bytes);
  }
  EXPECT_EQ(probe->paid->used(), baseline);
  EXPECT_EQ(probe->watch->live, 0U);
  // The outer ReportedProgram has its own loan slot; the inner native payment
  // must not consume it. This models the actual ComponentStepSource handoff.
  constexpr std::uint64_t compiler_bytes = kMiB;
  ASSERT_TRUE(probe->paid->Charge(compiler_bytes));
  auto compiler = std::make_shared<CompilerPayment>();
  compiler->paid = probe->paid;
  compiler->bytes = compiler_bytes;
  auto result =
      en::BuildComponentProgram(io, backend, {}, addresses, {}, std::nullopt, {}, table.owner);
  ASSERT_TRUE(result) << (result ? "" : result.error());
  ASSERT_TRUE((*result)->HoldConstruction(std::move(compiler)));
  EXPECT_GE(probe->paid->used(), baseline + compiler_bytes + inventory->bytes);
  const auto retained = (*result)->host_bytes();
  ASSERT_TRUE(probe->paid->Charge(retained));  // Paid before transferring either loan.
  probe->transferring = true;
  (*result)->ConstructionPaid();
  EXPECT_EQ(probe->paid->used(), baseline + retained);
  EXPECT_EQ(probe->watch->live, 1U);  // Native storage is still retained.
  const auto refunds = probe->native_refunds;
  (*result)->ConstructionPaid();
  EXPECT_EQ(probe->native_refunds, refunds);
  EXPECT_EQ(probe->paid->used(), baseline + retained);
  result->reset();
  EXPECT_EQ(probe->watch->live, 0U);
  probe->paid->Release(retained);
  EXPECT_EQ(probe->paid->used(), baseline);
}

TEST_P(GgmlConstructionGpu, PostLowerScratchRefusalDestroysArenaBeforeRefund) {
  ASSERT_NO_FATAL_FAILURE(Fund(GetParam()));
  const auto table = life->backend->Backend();
  const auto& backend = table.owner ? table.owner->backend() : table;
  // Descriptor-only geometry: every row and column is individually valid,
  // while their product exceeds the real bitonic scratch planner's int index.
  // No payload or device backing for these dimensions is ever allocated.
  constexpr std::uint32_t rows = 1U << 24U;
  ir::Builder b;
  b.SetOwner(0);
  const auto x = b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({128, rows})));
  b.Output(b.TopK(x, 8));
  auto graph = std::move(b).Finish();
  ASSERT_TRUE(graph);
  co::StepGraph step{.graph = std::move(*graph),
                     .inputs = {{.tensor = x, .kind = co::InputKind::kFeatures}},
                     .weights = {},
                     .state = {}};
  const pl::StepShape shape{.segments = {{.rows = rows, .outputs = 0, .want_head = false}}};
  const auto io = co::GraphIO(step);
  const kn::Addresses addresses{};
  auto plan = en::PlanComponentGraph(io, backend, {}, addresses, {.shape = &shape});
  ASSERT_TRUE(plan) << (plan ? "" : plan.error());
  const kn::LowerInput input{*plan, step, shape, addresses};
  const auto inventory = kg::LowerConstructionOf(input);
  ASSERT_TRUE(inventory);
  const auto probe = life->probe;
  probe->native_bytes = inventory->bytes;
  probe->watch->bytes = inventory->descriptors * ggml_tensor_overhead();
  WatchScope watch(*probe->watch);
  const auto baseline = probe->paid->used();
  auto failed = life->backend->Lower(input);
  ASSERT_FALSE(failed);
  EXPECT_NE(failed.error().find("bitonic kernel's 32-bit indexing"), std::string::npos)
      << failed.error();
  EXPECT_GT(probe->watch->allocations, 0U);
  EXPECT_EQ(probe->watch->live, 0U);
  EXPECT_EQ(probe->paid->used(), baseline);
  EXPECT_EQ(probe->native_refunds, 1U);
}
TEST_P(GgmlConstructionGpu, FunderRetainsItsAllowanceWithoutAnInstallingBorrower) {
  ASSERT_NO_FATAL_FAILURE(Fund(GetParam()));
  const auto paid = life->probe->paid;
  std::weak_ptr<Probe> observer = life->probe;
  life->probe.reset();
  EXPECT_FALSE(observer.expired());
  EXPECT_GT(paid->used(), 0U);
  // No program borrows this backend. Destruction releases both declaration
  // and native-funder controls through the retained callbacks, then drops them.
  life->backend.reset();
  EXPECT_EQ(paid->used(), 0U);
  EXPECT_TRUE(observer.expired());
}

TEST_P(GgmlConstructionGpu, ImageGeometryReplacementFailureKeepsThePaidCachedProgram) {
  ASSERT_NO_FATAL_FAILURE(Fund(GetParam()));
  const auto table = life->backend->Backend();
  const auto& backend = table.owner ? table.owner->backend() : table;
  const auto graph = [](std::uint64_t width) {
    ir::Builder b;
    b.SetOwner(0);
    const auto x =
        b.Input(ir::Dense(llmp::format::Encoding::kF32, ir::ShapeOf({width, width, 4, 1})));
    b.Output(b.Scale(x, 2));
    return std::move(b).Finish();
  };
  auto current = graph(2), replacement = graph(4);
  ASSERT_TRUE(current);
  ASSERT_TRUE(replacement);
  const co::GraphBuildIO io{*current, current->inputs(), current->outputs(), {}};
  const co::GraphBuildIO candidate{*replacement, replacement->inputs(), replacement->outputs(), {}};
  const kn::Addresses addresses{};
  auto plan = en::PlanComponentGraph(io, backend, {}, addresses);
  ASSERT_TRUE(plan) << (plan ? "" : plan.error());
  const auto inventory = kg::LowerConstructionOf({*plan, io, addresses});
  ASSERT_TRUE(inventory);
  const auto probe = life->probe;
  probe->native_bytes = inventory->bytes;
  probe->watch->bytes = inventory->descriptors * ggml_tensor_overhead();
  WatchScope watch(*probe->watch);
  const auto baseline = probe->paid->used();
  auto cached =
      en::BuildComponentProgram(io, backend, {}, addresses, {}, std::nullopt, {}, table.owner);
  ASSERT_TRUE(cached) << (cached ? "" : cached.error());
  const auto retained = (*cached)->host_bytes();
  ASSERT_TRUE(probe->paid->Charge(retained));
  probe->transferring = true;
  (*cached)->ConstructionPaid();
  probe->transferring = false;
  probe->retained_arenas = 1;
  EXPECT_EQ(probe->paid->used(), baseline + retained);
  const auto original_bytes = (*cached)->outputs().front().bytes;
  auto invalid = en::BuildComponentProgram(
      candidate, backend, {}, addresses, {}, std::nullopt,
      [](const kn::LowerInput&, const kn::StepProgram&) -> en::Status {
        return std::unexpected("replacement cache admission refused");
      },
      table.owner);
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error(), "replacement cache admission refused");
  EXPECT_EQ(probe->watch->live, 1U);
  EXPECT_EQ(probe->paid->used(), baseline + retained);
  // A second candidate can also be denied before an arena exists. The old
  // geometry's output ports and retained ownership survive both refusals.
  const auto allocated = probe->watch->allocations;
  probe->refuse = true;
  EXPECT_FALSE(en::BuildComponentProgram(candidate, backend, {}, addresses, {}, std::nullopt, {},
                                         table.owner));
  EXPECT_EQ(probe->watch->allocations, allocated);
  EXPECT_EQ((*cached)->outputs().front().bytes, original_bytes);
  EXPECT_EQ(probe->paid->used(), baseline + retained);
  probe->refuse = false;
  cached->reset();
  probe->retained_arenas = 0;
  EXPECT_EQ(probe->watch->live, 0U);
  probe->paid->Release(retained);
  EXPECT_EQ(probe->paid->used(), baseline);
}

INSTANTIATE_TEST_SUITE_P(DeclaredAndUndeclared, GgmlConstructionGpu, testing::Bool());
}  // namespace
