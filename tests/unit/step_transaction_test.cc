// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/step_transaction.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <thread>

#include "base/bytes.h"
#include "engine/execution_contract.h"
#include "engine/host_buffer.h"
#include "expected_error.h"
#include "providers/device_runtime.h"
#include "providers/fake/fake_device_execution.h"
#include "scheduler/commands.h"
#include "scheduler/scheduler.h"

namespace {
namespace en = llmp::engine;
namespace sc = llmp::scheduler;
namespace pr = llmp::providers;

TEST(StepTransactionTest, JobRetirementIsIndependentOfCallbackSuccessOnEveryDispatchPath) {
  for (const bool direct : {false, true}) {
    en::PagedNode node(en::NodeSettings{
        .compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent, .direct_steps = direct});
    ASSERT_TRUE(node.Open());
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    ASSERT_TRUE(node.Start(llmp::base::Bytes(fixed + 16 * en::kPagedExtent)));
    node.Run();
    for (const bool request : {false, true}) {
      if (request) ASSERT_TRUE(node.BeginRequest(0, {}, "job report test"));
      for (const auto outcome : {sc::JobResult::kNotStarted, sc::JobResult::kQueued,
                                 sc::JobResult::kFailed, sc::JobResult::kUnknown}) {
        std::array<std::uint8_t, 256> landing{};
        en::JobReport report;
        const auto posted = node.Job(
            {},
            [&](pr::NativeStream stream) {
              if (outcome != sc::JobResult::kNotStarted &&
                  !pr::FillAsync(stream, landing.data(), 37, landing.size()).ok())
                return sc::JobResult::kUnknown;
              return outcome;
            },
            "job report test", 0, {}, &report);
        EXPECT_EQ(posted.has_value(), outcome == sc::JobResult::kQueued);
        ASSERT_TRUE(report.callback_result);
        EXPECT_EQ(*report.callback_result, outcome);
        EXPECT_EQ(report.retirement, outcome == sc::JobResult::kNotStarted
                                         ? en::JobReport::Retirement::kNotSubmitted
                                         : en::JobReport::Retirement::kRetired);
        EXPECT_EQ(landing.front(), outcome == sc::JobResult::kNotStarted ? 0 : 37);
        EXPECT_EQ(landing.back(), landing.front());
      }
      if (request) ASSERT_TRUE(node.EndRequest(0));
    }
    ASSERT_TRUE(node.TearDown({}));
  }
}

TEST(StepTransactionTest,
     RequestWorkspaceGrowthAdmitsAtomicallyAndCapacityRefusalKeepsTheOldLease) {
  for (const bool direct : {false, true}) {
    SCOPED_TRACE(direct);
    en::PagedNode node({.compute_streams = 1,
                        .slots = 1,
                        .inline_lanes = true,
                        .slot_bytes = en::kPagedExtent,
                        .direct_steps = direct});
    ASSERT_TRUE(node.Open());
    ASSERT_TRUE(node.MapWorkspace(en::kPagedExtent, en::kPagedExtent, 8 * en::kPagedExtent, true));
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    ASSERT_TRUE(node.Start(llmp::base::Bytes(fixed + 4 * en::kPagedExtent)));
    node.Run();
    auto initial = node.catalog().ClosureOfExtents(node.activations().extents);
    auto wider = node.catalog().ClosureOfExtents(node.workspace_extents());
    ASSERT_TRUE(initial);
    ASSERT_TRUE(wider);
    ASSERT_TRUE(node.BeginRequest(0, *initial, "workspace replacement"));
    unsigned calls = 0;
    const auto job = [&](pr::NativeStream) {
      ++calls;
      return sc::JobResult::kQueued;
    };
    ASSERT_TRUE(node.Job(*initial, job, "initial protected work", 0));
    const auto leases = node.scheduler().held();
    ASSERT_TRUE(node.TryRefreshRequest(0, *wider));
    EXPECT_EQ(node.scheduler().held(), leases);
    ASSERT_TRUE(node.Job(*wider, job, "enlarged protected work", 0));
    EXPECT_EQ(calls, 2U);
    ASSERT_TRUE(node.GrowWorkspace(4 * en::kPagedExtent, en::kPagedExtent));
    auto cannot_fit = node.catalog().ClosureOfExtents(node.workspace_extents());
    ASSERT_TRUE(cannot_fit);
    bool over_budget = false;
    EXPECT_FALSE(node.TryRefreshRequest(0, *cannot_fit, &over_budget));
    EXPECT_TRUE(over_budget);
    EXPECT_TRUE(node.InRequest(0));
    EXPECT_EQ(node.scheduler().held(), leases);
    const auto refused = node.Job(*cannot_fit, job, "unadmitted workspace", 0);
    EXPECT_FALSE(refused);
    EXPECT_EQ(calls, 2U);
    ASSERT_TRUE(node.Job(*wider, job, "old authority survives refusal", 0));
    EXPECT_EQ(calls, 3U);
    ASSERT_TRUE(node.TryRefreshRequest(0, *initial));
    EXPECT_EQ(node.scheduler().held(), leases);
    ASSERT_TRUE(node.EndRequest(0));
    ASSERT_TRUE(node.TearDown({}));
  }
}

TEST(StepTransactionTest,
     RequestReplacementRefusesForeignAndForgedOccupancyBeforeEitherAcquisitionPolicy) {
  for (const bool reclaim : {false, true}) {
    SCOPED_TRACE(reclaim);
    en::PagedNode node(
        {.compute_streams = 1, .slots = 1, .inline_lanes = true, .slot_bytes = en::kPagedExtent});
    ASSERT_TRUE(node.Open());
    ASSERT_TRUE(node.MapWorkspace(en::kPagedExtent, en::kPagedExtent));
    auto local = node.catalog().ClosureOfExtents(node.activations().extents);
    ASSERT_TRUE(local);
    const auto other_domain = node.catalog().AddDomain("foreign physical budget");
    auto foreign = node.catalog().AddExtent({.domain = other_domain,
                                             .memory_class = llmp::catalog::MemoryClass::kRuntime,
                                             .recovery = llmp::catalog::Recovery::kPinned,
                                             .size = llmp::base::Bytes(en::kPagedExtent),
                                             .content = {}},
                                            true);
    ASSERT_TRUE(foreign);
    auto foreign_closure = node.catalog().ClosureOfExtents(std::array{*foreign});
    ASSERT_TRUE(foreign_closure);
    const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
    ASSERT_TRUE(node.Start(llmp::base::Bytes(fixed + 8 * en::kPagedExtent)));
    node.Run();
    ASSERT_TRUE(node.BeginRequest(0, *local, "local replacement owner"));
    auto paid = node.PrepayHost(1);
    ASSERT_TRUE(paid);
    const auto before_payment = paid->accounted_bytes();
    const auto before_held = node.scheduler().held();
    const auto before_occupancy = node.catalog().OccupancyOf(node.domain()).Total();
    unsigned reclaim_calls = 0, job_calls = 0;
    if (reclaim)
      node.SetReclaimer(
          [&](std::uint64_t, en::PagedNode::ReclaimFor, std::span<const llmp::catalog::ExtentId>) {
            ++reclaim_calls;
            return std::uint64_t{0};
          });
    bool over_budget = true;
    EXPECT_FALSE(node.TryRefreshRequest(0, *foreign_closure, &over_budget));
    EXPECT_FALSE(over_budget);
    auto forged = *local;
    forged.bytes_by_domain.clear();
    EXPECT_FALSE(node.TryRefreshRequest(0, forged, &over_budget));
    EXPECT_FALSE(over_budget);
    EXPECT_EQ(paid->accounted_bytes(), before_payment);
    EXPECT_EQ(node.scheduler().held(), before_held);
    EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).Total(), before_occupancy);
    EXPECT_EQ(node.catalog().Describe(*foreign)->leases, 0U);
    EXPECT_EQ(reclaim_calls, 0U);
    const auto job = [&](pr::NativeStream) {
      ++job_calls;
      return sc::JobResult::kQueued;
    };
    EXPECT_FALSE(node.Job(*foreign_closure, job, "foreign work remains inadmissible", 0));
    EXPECT_EQ(job_calls, 0U);
    ASSERT_TRUE(node.Job(*local, job, "original local authority remains usable", 0));
    EXPECT_EQ(job_calls, 1U);
    ASSERT_TRUE(node.EndRequest(0));
    node.SetReclaimer({});
    paid->Reset();
    ASSERT_TRUE(node.Call(
        [&]() -> en::Status {
          if (!node.catalog().ReleasePinned(*foreign))
            return std::unexpected("retiring the foreign test extent");
          return {};
        },
        "retiring foreign domain control"));
    ASSERT_TRUE(node.TearDown({}));
  }
}

class RequestExtensionTest : public ::testing::Test {
 protected:
  std::unique_ptr<en::PagedNode> node_;
  llmp::catalog::Closure permanent_, private_;
  void Start(bool direct, std::uint64_t budget_extents = 16) {
    node_ = std::make_unique<en::PagedNode>(en::NodeSettings{.compute_streams = 1,
                                                             .slots = 1,
                                                             .inline_lanes = true,
                                                             .slot_bytes = en::kPagedExtent,
                                                             .direct_steps = direct});
    ASSERT_TRUE(node_->Open());
    ASSERT_TRUE(
        node_->MapWorkspace(en::kPagedExtent, en::kPagedExtent, 8 * en::kPagedExtent, true));
    const auto fixed = node_->catalog().OccupancyOf(node_->domain()).Total().value();
    ASSERT_TRUE(node_->Start(llmp::base::Bytes(fixed + budget_extents * en::kPagedExtent)));
    node_->Run();
    auto permanent = node_->catalog().ClosureOfExtents(node_->activations().extents);
    auto extra = node_->catalog().ClosureOfExtents(node_->pool().extents);
    ASSERT_TRUE(permanent);
    ASSERT_TRUE(extra);
    permanent_ = std::move(*permanent);
    private_ = std::move(*extra);
    ASSERT_TRUE(node_->BeginRequest(0, permanent_, "prepared private resources"));
  }
  void Stop() {
    if (!node_) return;
    if (node_->InRequest(0)) EXPECT_TRUE(node_->EndRequest(0));
    EXPECT_TRUE(node_->TearDown({}));
    node_.reset();
  }
  void TearDown() override { Stop(); }
};

TEST_F(RequestExtensionTest, ReverseRemovesOnlyPrivateAdditionsAndKeepsPermanentGrowth) {
  for (const bool direct : {false, true}) {
    SCOPED_TRACE(direct);
    Start(direct);
    ASSERT_FALSE(HasFatalFailure());
    auto prepared = node_->PrepareRequestExtension(0, private_);
    ASSERT_TRUE(prepared) << llmp::test_support::Failed(prepared).value_or("");
    auto copied = *prepared;
    const auto held = node_->scheduler().held();
    unsigned callbacks = 0;
    const auto job = [&](pr::NativeStream) {
      ++callbacks;
      return sc::JobResult::kQueued;
    };
    EXPECT_FALSE(node_->Job(private_, job, "preview grants no private execution", 0));
    EXPECT_EQ(callbacks, 0U);
    ASSERT_TRUE(node_->Job(permanent_, job, "original authority after prepare", 0));
    ASSERT_TRUE(node_->ApplyRequestExtension(*prepared));
    ASSERT_TRUE(node_->ApplyRequestExtension(copied));  // same holder, idempotent
    EXPECT_EQ(node_->scheduler().held(), held);
    ASSERT_TRUE(node_->Job(private_, job, "actual private extension", 0));
    ASSERT_TRUE(node_->GrowWorkspace(2 * en::kPagedExtent, en::kPagedExtent));
    auto grown = node_->catalog().ClosureOfExtents(node_->activations().extents);
    ASSERT_TRUE(grown);
    // A permanent refresh does not accidentally drop the private scope.
    ASSERT_TRUE(node_->TryRefreshRequest(0, *grown));
    ASSERT_TRUE(node_->Job(private_, job, "private rights survive permanent growth", 0));
    auto payment = node_->PrepayHost(1);
    ASSERT_TRUE(payment);
    const auto counted = payment->accounted_bytes();
    std::uint32_t reclaims = 0;
    node_->SetReclaimer(
        [&](std::uint64_t, en::PagedNode::ReclaimFor, std::span<const llmp::catalog::ExtentId>) {
          ++reclaims;
          return std::uint64_t{0};
        });
    ASSERT_TRUE(node_->CloseRequestExtension(copied));
    EXPECT_EQ(payment->accounted_bytes(), counted);
    EXPECT_EQ(reclaims, 0U);
    EXPECT_EQ(node_->scheduler().held(), held);
    for (const auto& [extent, generation] : private_.extents) {
      (void)generation;
      EXPECT_EQ(node_->catalog().Describe(extent)->leases, 0U);
    }
    for (const auto& [extent, generation] : grown->extents) {
      (void)generation;
      EXPECT_GT(node_->catalog().Describe(extent)->leases, 0U);
    }
    ASSERT_TRUE(node_->Job(*grown, job, "permanent additions survive reverse exchange", 0));
    const auto completed = callbacks;
    EXPECT_FALSE(node_->Job(private_, job, "closed copies grant no private execution", 0));
    EXPECT_EQ(callbacks, completed);
    EXPECT_FALSE(node_->ApplyRequestExtension(copied));
    EXPECT_TRUE(node_->CloseRequestExtension(*prepared));
    node_->SetReclaimer({});
    Stop();
  }
}

TEST_F(RequestExtensionTest, CapacityRefusalOfPreparedForwardKeepsOriginalAuthorityAndPayment) {
  for (const bool direct : {false, true}) {
    SCOPED_TRACE(direct);
    Start(direct, 4);
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_TRUE(node_->GrowWorkspace(en::kPagedExtent, 4 * en::kPagedExtent));
    auto wider = node_->catalog().ClosureOfExtents(node_->pool().extents);
    ASSERT_TRUE(wider);
    auto prepared = node_->PrepareRequestExtension(0, *wider);
    ASSERT_TRUE(prepared);
    const auto held = node_->scheduler().held();
    bool over_budget = false;
    EXPECT_FALSE(node_->ApplyRequestExtension(*prepared, &over_budget));
    EXPECT_TRUE(over_budget);
    EXPECT_TRUE(node_->InRequest(0));
    EXPECT_EQ(node_->scheduler().held(), held);
    unsigned calls = 0;
    const auto job = [&](pr::NativeStream) {
      ++calls;
      return sc::JobResult::kQueued;
    };
    ASSERT_TRUE(node_->Job(permanent_, job, "original closure after prepared capacity refusal", 0));
    EXPECT_EQ(calls, 1U);
    EXPECT_FALSE(node_->Job(*wider, job, "unpublished candidate remains unauthorized", 0));
    EXPECT_EQ(calls, 1U);
    ASSERT_TRUE(node_->CloseRequestExtension(*prepared));
    Stop();
  }
}

TEST_F(RequestExtensionTest, StalePreparationAndForeignTokensCannotReplaceAnotherHolder) {
  Start(false);
  ASSERT_FALSE(HasFatalFailure());
  auto prepared = node_->PrepareRequestExtension(0, private_);
  ASSERT_TRUE(prepared);
  ASSERT_TRUE(node_->GrowWorkspace(2 * en::kPagedExtent, en::kPagedExtent));
  auto grown = node_->catalog().ClosureOfExtents(node_->activations().extents);
  ASSERT_TRUE(grown);
  ASSERT_TRUE(node_->TryRefreshRequest(0, *grown));
  EXPECT_FALSE(node_->ApplyRequestExtension(*prepared));
  ASSERT_TRUE(node_->CloseRequestExtension(*prepared));
  auto current = node_->PrepareRequestExtension(0, private_);
  ASSERT_TRUE(current);
  en::PagedNode foreign(en::NodeSettings{});
  EXPECT_FALSE(foreign.ApplyRequestExtension(*current));
  EXPECT_FALSE(foreign.CloseRequestExtension(*current));
  ASSERT_TRUE(node_->EndRequest(0));
  ASSERT_TRUE(node_->BeginRequest(0, *grown, "fresh request under the same node"));
  EXPECT_FALSE(node_->ApplyRequestExtension(*current));
  ASSERT_TRUE(node_->CloseRequestExtension(*current));
  unsigned calls = 0;
  ASSERT_TRUE(node_->Job(
      *grown,
      [&](pr::NativeStream) {
        ++calls;
        return sc::JobResult::kQueued;
      },
      "stale prepared owner never poisons the new holder", 0));
  EXPECT_EQ(calls, 1U);
}

TEST(RequestExtensionLifetimeTest, AReusedNodeAddressCannotAuthenticateOldPreparedMetadata) {
  std::optional<en::PagedNode> node;
  en::RequestExtension retained;
  en::HostPayment probe;
  std::uint64_t baseline = 0;
  const en::NodeSettings settings{
      .compute_streams = 1, .slots = 1, .inline_lanes = true, .slot_bytes = en::kPagedExtent};
  const auto begin = [&] {
    node.emplace(settings);
    ASSERT_TRUE(node->Open());
    ASSERT_TRUE(node->MapWorkspace(en::kPagedExtent, en::kPagedExtent));
    const auto fixed = node->catalog().OccupancyOf(node->domain()).Total().value();
    ASSERT_TRUE(node->Start(llmp::base::Bytes(fixed + 16 * en::kPagedExtent)));
    node->Run();
    const auto base = node->catalog().ClosureOfExtents(node->activations().extents);
    ASSERT_TRUE(base);
    ASSERT_TRUE(node->BeginRequest(0, *base, "node generation control"));
  };
  begin();
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  auto paid = node->PrepayHost(1);
  ASSERT_TRUE(paid);
  probe = std::move(*paid);
  baseline = probe.accounted_bytes();
  {
    const auto extra = node->catalog().ClosureOfExtents(node->pool().extents);
    ASSERT_TRUE(extra);
    auto prepared = node->PrepareRequestExtension(0, *extra);
    ASSERT_TRUE(prepared);
    retained = std::move(*prepared);
  }
  EXPECT_GT(probe.accounted_bytes(), baseline);
  ASSERT_TRUE(node->EndRequest(0));
  ASSERT_TRUE(node->TearDown({}));
  node.reset();
  begin();  // optional storage reuses its address, request/lease IDs can repeat
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  EXPECT_FALSE(node->ApplyRequestExtension(retained));
  EXPECT_FALSE(node->CloseRequestExtension(retained));
  ASSERT_TRUE(node->EndRequest(0));
  ASSERT_TRUE(node->TearDown({}));
  node.reset();
  std::thread release([owned = std::move(retained)]() mutable { owned = {}; });
  release.join();
  EXPECT_EQ(probe.accounted_bytes(), baseline);
}

struct UnprovenNodeOwner {
  explicit UnprovenNodeOwner(bool direct)
      : node(en::NodeSettings{.compute_streams = 1,
                              .slots = 1,
                              .inline_lanes = true,
                              .slot_bytes = en::kPagedExtent,
                              .direct_steps = direct}) {}
  en::PagedNode node;
  en::RequestExtension extension;
  std::atomic<unsigned> callbacks{0};
};

void CheckUnprovenNodeAdmission(bool direct) {
  // Construct all execution owners in the child and retain them until exit:
  // neither a failed job nor a destructor may certify the missing fence.
  EXPECT_EXIT(
      {
        auto* retained = new UnprovenNodeOwner(direct);
        // Both retained and direct are by value; a capture-list comma would
        // split the enclosing EXPECT_EXIT macro before C++ parsing.
        const auto exercise = [=] {
          auto& node = retained->node;
          ASSERT_TRUE(node.Open());
          ASSERT_TRUE(node.MapWorkspace(en::kPagedExtent, en::kPagedExtent));
          const auto closure = node.catalog().ClosureOfExtents(node.workspace_extents());
          ASSERT_TRUE(closure);
          const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
          ASSERT_TRUE(node.Start(llmp::base::Bytes(fixed + 16 * en::kPagedExtent)));
          node.Run();
          ASSERT_TRUE(node.BeginRequest(0, {}, "unproven request admission"));
          auto prepared = node.PrepareRequestExtension(0, *closure);
          ASSERT_TRUE(prepared);
          retained->extension = std::move(*prepared);
          ASSERT_TRUE(node.ApplyRequestExtension(retained->extension));
          auto& execution = static_cast<pr::fake::FakeDeviceExecution&>(node.execution());
          en::JobReport first;
          const auto submitted = node.Job(
              *closure,
              [&](pr::NativeStream stream) {
                ++retained->callbacks;
                if (!pr::FillAsync(stream, reinterpret_cast<void*>(node.activations().base), 37,
                                   256)
                         .ok())
                  return sc::JobResult::kUnknown;
                // Arm after this callback really queued work. No earlier
                // materialization/preparation fence can consume the fault.
                execution.FailNextQuery(pr::ProviderError::kUnknown);
                return sc::JobResult::kQueued;
              },
              "first unproven job", 0, {}, &first);
          ASSERT_FALSE(submitted);
          ASSERT_EQ(first.callback_result, sc::JobResult::kQueued);
          ASSERT_EQ(first.retirement, en::JobReport::Retirement::kUnproven);
          ASSERT_EQ(retained->callbacks.load(), 1U);
          ASSERT_TRUE(node.InRequest(0));
          const auto held = node.scheduler().held();
          const auto occupied = node.catalog().OccupancyOf(node.domain()).held;
          ASSERT_GT(held, 0U);
          ASSERT_GT(occupied.value(), 0U);
          // Neither an unchanged lease nor a smaller closure can launder a
          // missing fence into a completed-boundary replacement.
          EXPECT_FALSE(node.TryRefreshRequest(0, *closure));
          EXPECT_FALSE(node.TryRefreshRequest(0, {}));
          EXPECT_FALSE(node.CloseRequestExtension(retained->extension));
          EXPECT_FALSE(node.ApplyRequestExtension(retained->extension));
          EXPECT_EQ(node.scheduler().held(), held);
          EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).held, occupied);
          en::JobReport second;
          const auto refused = node.Job(
              *closure,
              [retained](pr::NativeStream) {
                ++retained->callbacks;
                return sc::JobResult::kQueued;
              },
              "must not enter second callback", 0, {}, &second);
          EXPECT_FALSE(refused);
          EXPECT_EQ(retained->callbacks.load(), 1U);
          EXPECT_FALSE(second.callback_result);
          EXPECT_EQ(second.retirement, en::JobReport::Retirement::kNotSubmitted);
          EXPECT_EQ(first.retirement, en::JobReport::Retirement::kUnproven);
          EXPECT_TRUE(node.InRequest(0));
          if (direct) {
            auto observed = node.PrepayHost(1);
            ASSERT_TRUE(observed);
            const auto still_paid = observed->accounted_bytes();
            EXPECT_FALSE(node.EndRequest(0));
            EXPECT_TRUE(node.InRequest(0));
            EXPECT_EQ(observed->accounted_bytes(), still_paid);
          }
          EXPECT_EQ(node.scheduler().held(), held);
          EXPECT_EQ(node.catalog().OccupancyOf(node.domain()).held, occupied);
          for (const auto id : node.workspace_extents()) {
            const auto view = node.catalog().Describe(id);
            ASSERT_TRUE(view);
            EXPECT_GT(view->leases, 0U);
          }
        };
        exercise();
        const auto* result = ::testing::UnitTest::GetInstance()->current_test_info()->result();
        for (int i = 0; i < result->total_part_count(); ++i) {
          const auto& part = result->GetTestPartResult(i);
          if (part.failed())
            std::fprintf(stderr, "direct=%d %s:%d: %s\n", direct ? 1 : 0,
                         part.file_name() ? part.file_name() : "child", part.line_number(),
                         part.message());
        }
        std::fflush(stderr);
        std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(PagedNodeUnprovenDeathTest, DirectRequestRejectsSecondCallbackAndRetainsHeldOwner) {
  CheckUnprovenNodeAdmission(true);
}

TEST(PagedNodeUnprovenDeathTest, ScheduledRequestRejectsSecondCallbackAndRetainsHeldOwner) {
  CheckUnprovenNodeAdmission(false);
}

TEST(StepTransactionTest, PrepaidSupplementalNodeControlsReusePaidCapacityAndRejectReopenedHolder) {
  en::PagedNode node(
      en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
  ASSERT_TRUE(node.Open());
  ASSERT_TRUE(node.MapWorkspace(en::kPagedExtent, en::kPagedExtent));
  auto allowance = en::ExecutionMetadataAllowance::Admit(node.catalog(), node.domain(), 65536,
                                                         64 * en::kPagedExtent, en::kPagedExtent);
  ASSERT_TRUE(allowance) << llmp::test_support::Failed(allowance).value_or("");
  auto closure = node.catalog().ClosureOfExtents(node.workspace_extents());
  ASSERT_TRUE(closure);
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(llmp::base::Bytes(fixed + 16 * en::kPagedExtent)));
  node.Run();
  ASSERT_TRUE(node.BeginRequest(0, {}, "prepaid supplemental base"));
  auto capacity = node.PrepareSupplementalCapacity(0, closure->extents.size(), *allowance);
  ASSERT_TRUE(capacity) << llmp::test_support::Failed(capacity).value_or("");
  const auto filler = (*allowance)->capacity() - (*allowance)->used();
  ASSERT_TRUE((*allowance)->Charge(filler));
  auto payload = std::make_shared<int>(0);
  int calls = 0;
  for (int iteration = 0; iteration != 2; ++iteration) {
    en::JobReport report;
    auto ran = node.SupplementalJob(
        *closure,
        [&](pr::NativeStream) {
          ++calls;
          return sc::JobResult::kQueued;
        },
        payload, *allowance, "paid supplemental continuation", 0, &report, *capacity);
    ASSERT_TRUE(ran) << llmp::test_support::Failed(ran).value_or("");
    EXPECT_EQ(report.retirement, en::JobReport::Retirement::kRetired);
    EXPECT_EQ((*allowance)->used(), (*allowance)->capacity());
  }
  ASSERT_TRUE(node.EndRequest(0));
  ASSERT_TRUE(node.BeginRequest(0, {}, "different supplemental holder"));
  en::JobReport refused;
  EXPECT_FALSE(node.SupplementalJob(
      *closure,
      [&](pr::NativeStream) {
        ++calls;
        return sc::JobResult::kQueued;
      },
      payload, *allowance, "stale supplemental receipt", 0, &refused, *capacity));
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(refused.retirement, en::JobReport::Retirement::kNotSubmitted);
  ASSERT_TRUE(node.EndRequest(0));
  (*allowance)->Release(filler);
  *capacity = {};
  EXPECT_EQ((*allowance)->used(), 0U);
  ASSERT_TRUE(node.TearDown({}));
}

TEST(StepTransactionTest, SupplementalReportsActualRetirementEvenWhenItsQueuedWorkFails) {
  en::PagedNode node(
      en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
  ASSERT_TRUE(node.Open());
  ASSERT_TRUE(node.MapWorkspace(en::kPagedExtent, en::kPagedExtent));
  constexpr std::uint64_t allowance_bytes = 32 * 1024;
  constexpr std::uint64_t available_bytes = 64 * en::kPagedExtent;
  constexpr std::uint64_t physical_margin = en::kPagedExtent;
  static_assert(physical_margin > 0 && allowance_bytes + physical_margin <= available_bytes);
  auto paid = en::ExecutionMetadataAllowance::Admit(node.catalog(), node.domain(), allowance_bytes,
                                                    available_bytes, physical_margin);
  ASSERT_TRUE(paid) << llmp::test_support::Failed(paid).value_or("");
  using Payload = std::array<std::uint8_t, 256>;
  constexpr auto payload_charge = sizeof(Payload) + 128;  // malloc + shared control
  ASSERT_TRUE((*paid)->Charge(payload_charge));
  auto payload = std::shared_ptr<Payload>(new Payload{}, [allowance = *paid](Payload* block) {
    delete block;
    allowance->Release(payload_charge);
  });
  const auto selected = node.catalog().ClosureOfExtents(node.workspace_extents());
  ASSERT_TRUE(selected);
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(llmp::base::Bytes(fixed + 16 * en::kPagedExtent)));
  node.Run();
  for (const auto outcome :
       {sc::JobResult::kQueued, sc::JobResult::kFailed, sc::JobResult::kUnknown}) {
    ASSERT_TRUE(node.BeginRequest(0, {}, "supplemental proof base"));
    en::JobReport report;
    const auto posted = node.SupplementalJob(
        *selected,
        [payload, outcome](pr::NativeStream native) {
          if (!pr::FillAsync(native, payload->data(), 71, payload->size()).ok())
            return sc::JobResult::kUnknown;
          return outcome;
        },
        payload, *paid, "supplemental proof continuation", 0, &report);
    EXPECT_EQ(posted.has_value(), outcome == sc::JobResult::kQueued);
    EXPECT_EQ(report.retirement, en::JobReport::Retirement::kRetired);
    EXPECT_EQ(payload->front(), 71);
    EXPECT_EQ(payload->back(), 71);
    // SupplementalJob failure cancels the base request even after positive
    // retirement. Ending it reports that cancellation, then removes it.
    EXPECT_EQ(node.EndRequest(0).has_value(), outcome == sc::JobResult::kQueued);
    EXPECT_FALSE(node.InRequest(0));
  }
  en::JobReport unsubmitted;
  bool called = false;
  EXPECT_FALSE(node.SupplementalJob(
      *selected,
      [&](pr::NativeStream) {
        called = true;
        return sc::JobResult::kQueued;
      },
      payload, *paid, "no base request", 0, &unsubmitted));
  EXPECT_FALSE(called);
  EXPECT_EQ(unsubmitted.retirement, en::JobReport::Retirement::kNotSubmitted);
  payload.reset();
  EXPECT_EQ((*paid)->used(), 0U);
  ASSERT_TRUE(node.TearDown({}));
}

TEST(StepTransactionTest, APreCallbackRefusalReportsNoSubmissionAndDoesNotInvokeItsPayload) {
  en::PagedNode node(
      en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
  ASSERT_TRUE(node.Open());
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(llmp::base::Bytes(fixed + 16 * en::kPagedExtent)));
  node.Run();
  en::JobReport refused;
  bool called = false;
  const auto posted = node.Job(
      {},
      [&](pr::NativeStream) {
        const auto nested = node.Job(
            {},
            [&](pr::NativeStream) {
              called = true;
              return sc::JobResult::kQueued;
            },
            "forbidden recursive job", 0, {}, &refused);
        EXPECT_FALSE(nested);
        return sc::JobResult::kQueued;
      },
      "outer job report test", 0);
  ASSERT_TRUE(posted);
  EXPECT_FALSE(called);
  EXPECT_FALSE(refused.callback_result);
  EXPECT_EQ(refused.retirement, en::JobReport::Retirement::kNotSubmitted);
  ASSERT_TRUE(node.TearDown({}));
}
TEST(StepTransactionTest, ResultDescriptorsAndPayloadKeepOnePaymentThroughMovesAndLastThread) {
  en::PagedNode node(
      en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
  ASSERT_TRUE(node.Open());
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  node.SetHostFloor(0);
  ASSERT_TRUE(node.Start(llmp::base::Bytes(fixed + 16 * en::kPagedExtent)));
  node.Run();
  const std::array<std::uint64_t, 3> segments{12, 0, 20};
  auto made = en::HostOutputs::Zeroed(node, segments, 16);
  ASSERT_TRUE(made) << llmp::test_support::Failed(made).value_or("");
  auto first = made->front();
  auto last = made->at(2);
  auto joined = made->joined();
  const auto payment = first.paid_bytes();
  EXPECT_GT(payment, 12 + 20 + 16);
  EXPECT_EQ(last.paid_bytes(), payment);
  EXPECT_EQ(joined.paid_bytes(), payment);
  const auto paid = node.host_counted();
  auto copy = *made;
  made = en::HostOutputs{};
  auto moved = std::move(copy);
  EXPECT_TRUE(copy.empty());
  EXPECT_EQ(moved.size(), 3U);
  auto& self = moved;
  moved = std::move(self);
  EXPECT_EQ(moved.size(), 3U);
  EXPECT_EQ(moved.front().size(), 12U);
  EXPECT_EQ(moved.at(0).size(), 12U);
  EXPECT_TRUE(moved.at(1).empty());
  EXPECT_EQ(moved.at(2).size(), 20U);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(last.data()) % alignof(std::max_align_t), 0U);
  moved = en::HostOutputs{};
  first = {};
  joined = {};
  EXPECT_EQ(node.host_counted(), paid);  // The last slice still owns descriptors too.
  std::thread consumer([buffer = std::move(last)]() mutable { buffer = {}; });
  consumer.join();
  EXPECT_TRUE(last.empty());
  EXPECT_EQ(node.host_counted(), paid - payment);
  ASSERT_TRUE(node.TryChargeHost(0));  // Driver collects the deferred refund once.
  EXPECT_EQ(node.host_counted(), paid - payment);
  ASSERT_TRUE(node.TryChargeHost(0));
  EXPECT_EQ(node.host_counted(), paid - payment);
  ASSERT_TRUE(node.TearDown({}));
}

TEST(StepTransactionTest, EmptyResultsAndOverflowOrCapacityRefusalsAllocateNothing) {
  en::PagedNode node(
      en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
  ASSERT_TRUE(node.Open());
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  node.SetHostFloor(0);
  ASSERT_TRUE(node.Start(llmp::base::Bytes(fixed + 2 * en::kPagedExtent)));
  node.Run();
  const auto before = node.host_counted();
  auto empty = en::HostOutputs::Zeroed(node, std::array<std::uint64_t, 3>{}, 0);
  ASSERT_TRUE(empty);
  EXPECT_EQ(empty->size(), 3U);
  EXPECT_TRUE(empty->front().empty());
  EXPECT_TRUE(empty->joined().empty());
  EXPECT_EQ(empty->front().paid_bytes(), 0U);
  EXPECT_EQ(node.host_counted(), before);
  auto overflow = en::HostOutputs::Zeroed(node, std::array{UINT64_MAX, std::uint64_t{1}}, 0);
  EXPECT_FALSE(overflow);
  EXPECT_EQ(node.host_counted(), before);
  auto refused =
      en::HostOutputs::Zeroed(node, std::array<std::uint64_t, 1>{32 * en::kPagedExtent}, 0);
  EXPECT_FALSE(refused);
  EXPECT_EQ(node.host_counted(), before);  // Even first-use counter funding was refused.
  ASSERT_TRUE(node.TearDown({}));
}

TEST(StepTransactionTest, AResultAndItsPrepaidReleaseAccountCanOutliveTheNode) {
  en::HostBuffer retained;
  en::HostPayment receipt;
  std::uint64_t counted = 0;
  {
    auto node = std::make_unique<en::PagedNode>(
        en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
    ASSERT_TRUE(node->Open());
    const auto fixed = node->catalog().OccupancyOf(node->domain()).Total().value();
    ASSERT_TRUE(node->Start(llmp::base::Bytes(fixed + 16 * en::kPagedExtent)));
    node->Run();
    auto result = en::HostOutputs::Zeroed(*node, std::array<std::uint64_t, 1>{32}, 8);
    ASSERT_TRUE(result);
    retained = result->front();
    auto paid = node->PrepayHost(128);
    ASSERT_TRUE(paid);
    receipt = std::move(*paid);
    counted = receipt.accounted_bytes();
    EXPECT_GT(counted, retained.paid_bytes() + receipt.bytes());  // Counter metadata stays paid.
    ASSERT_TRUE(node->TearDown({}));
  }
  EXPECT_EQ(retained.size(), 32U);
  EXPECT_NE(retained.data(), nullptr);
  EXPECT_EQ(receipt.accounted_bytes(), counted);
  const auto result_paid = retained.paid_bytes();
  std::thread consumer([buffer = std::move(retained)]() mutable { buffer = {}; });
  consumer.join();
  EXPECT_EQ(receipt.accounted_bytes(), counted - result_paid);
  receipt.Reset();  // Last owner deletes its account without accessing the dead node.
  EXPECT_EQ(receipt.bytes(), 0U);
}

TEST(StepTransactionTest, UnprovenStepFailureRetainsWitnessesUntilFenceRecovery) {
  en::PagedNode node(en::NodeSettings{
      .compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent, .direct_steps = true});
  ASSERT_TRUE(node.Open());
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  ASSERT_TRUE(node.Start(llmp::base::Bytes(fixed + 16 * en::kPagedExtent)));
  node.Run();

  class TestModel final : public en::PagedModel {
   public:
    explicit TestModel(std::uint32_t stream) : stream_(stream) {}
    std::uint32_t stream() const override { return stream_; }
    const llmp::catalog::Closure& fence_closure() const override { return closure_; }
    std::vector<llmp::catalog::ExtentId> managed_extents() const override { return {}; }
    en::Status Release() override { return {}; }
    en::Status AcknowledgeFence() override {
      unproven.ClearAfterFence();
      acknowledged = true;
      return {};
    }
    en::UnprovenWork unproven;
    bool acknowledged = false;

   private:
    std::uint32_t stream_ = 0;
    llmp::catalog::Closure closure_;
  };

  TestModel model(0);
  ASSERT_TRUE(node.BeginRequest(0, {}, "request with lost fence"));
  static_cast<llmp::providers::fake::FakeDeviceExecution&>(node.execution())
      .FailNextQuery(llmp::providers::ProviderError::kUnknown);
  model.unproven.Mark();
  en::JobReport failed_report;
  auto step = node.Job(
      {}, [](pr::NativeStream) { return sc::JobResult::kQueued; }, "unprovable step", 0, {},
      &failed_report);
  EXPECT_FALSE(step);
  EXPECT_EQ(failed_report.retirement, en::JobReport::Retirement::kUnproven);
  EXPECT_TRUE(model.unproven.active());
  EXPECT_TRUE(node.unproven());
  EXPECT_TRUE(node.InRequest(0));
  EXPECT_FALSE(node.EndRequest(0));

  // Stream fence recovery proves past work ended, clears tainted witness, and acknowledges model
  // fence
  auto recovered = node.RecoverAfterFence(model);
  ASSERT_TRUE(recovered) << recovered.error();
  EXPECT_TRUE(model.acknowledged);
  EXPECT_FALSE(model.unproven.active());
  EXPECT_FALSE(node.unproven());
  EXPECT_FALSE(node.InRequest(0));

  // Verify clean subsequent service
  ASSERT_TRUE(node.BeginRequest(0, {}, "subsequent clean request"));
  EXPECT_TRUE(node.InRequest(0));
  en::JobReport clean_report;
  ASSERT_TRUE(node.Job(
      {}, [](pr::NativeStream) { return sc::JobResult::kQueued; }, "clean step", 0, {},
      &clean_report));
  EXPECT_EQ(clean_report.retirement, en::JobReport::Retirement::kRetired);
  ASSERT_TRUE(node.EndRequest(0));
  EXPECT_FALSE(node.InRequest(0));

  auto& execution = static_cast<llmp::providers::fake::FakeDeviceExecution&>(node.execution());
  const pr::FenceId leaked(0, 1);
  ASSERT_EQ(execution.Query(leaked).value_or(pr::FenceState::kPending), pr::FenceState::kComplete);
  ASSERT_TRUE(execution.Release(leaked));
  ASSERT_TRUE(node.TearDown({}));
}

TEST(StepTransactionTest, UnprovenWorkLifecycle) {
  en::UnprovenWork work;
  EXPECT_FALSE(work.active());
  EXPECT_FALSE(static_cast<bool>(work));

  work.Mark();
  EXPECT_TRUE(work.active());
  EXPECT_TRUE(static_cast<bool>(work));

  work.ClearAfterFence();
  EXPECT_FALSE(work.active());

  int dummy = 42;
  work.KeepPinned(&dummy);
  EXPECT_TRUE(work.active());
  EXPECT_EQ(work.kept_pinned(), 1u);

  // Cleared after fence retains kept pinned allocations
  work.ClearAfterFence();
  EXPECT_TRUE(work.active());
  EXPECT_EQ(work.kept_pinned(), 1u);

  // Full clear drops kept pinned allocations
  work.Clear();
  EXPECT_FALSE(work.active());
  EXPECT_EQ(work.kept_pinned(), 0u);
}

TEST(StepTransactionTest, SupplementalJobReclaimsBeforeMaterializingSelectedExtents) {
  en::PagedNode node(
      en::NodeSettings{.compute_streams = 1, .slots = 1, .slot_bytes = en::kPagedExtent});
  ASSERT_TRUE(node.Open());
  ASSERT_TRUE(
      node.MapWorkspace(2 * en::kPagedExtent, en::kPagedExtent, 8 * en::kPagedExtent, true));
  auto allowance = en::ExecutionMetadataAllowance::Admit(node.catalog(), node.domain(), 65536,
                                                         64 * en::kPagedExtent, en::kPagedExtent);
  ASSERT_TRUE(allowance) << llmp::test_support::Failed(allowance).value_or("");
  const auto fixed = node.catalog().OccupancyOf(node.domain()).Total().value();
  // Start with budget equal to fixed + 2 extents (so 3 workspace extents exceed budget by 1
  // extent).
  ASSERT_TRUE(node.Start(llmp::base::Bytes(fixed + 2 * en::kPagedExtent)));
  node.Run();
  auto base_closure =
      node.catalog().ClosureOfExtents(std::span{node.activations().extents}.first(1));
  auto selected_closure = node.catalog().ClosureOfExtents(node.workspace_extents());
  ASSERT_TRUE(base_closure);
  ASSERT_TRUE(selected_closure);
  ASSERT_TRUE(node.BeginRequest(0, *base_closure, "supplemental reclaim base"));

  auto payload = std::make_shared<int>(0);
  int reclaim_calls = 0;
  bool allow_reclaim = false;
  node.SetReclaimer([&](std::uint64_t needed, en::PagedNode::ReclaimFor what,
                        std::span<const llmp::catalog::ExtentId> protect) -> std::uint64_t {
    ++reclaim_calls;
    EXPECT_GE(needed, en::kPagedExtent);
    EXPECT_EQ(what, en::PagedNode::ReclaimFor::kMaterialize);
    EXPECT_NE(std::ranges::find(protect, node.activations().extents.front()), protect.end());
    if (!allow_reclaim) return 0;
    node.SetBudget(llmp::base::Bytes(fixed + 4 * en::kPagedExtent));
    return needed;
  });

  // When reclaimer returns 0, SupplementalJob refuses cleanly before offering the program.
  auto refused = node.SupplementalJob(
      *selected_closure, [](pr::NativeStream) { return sc::JobResult::kQueued; }, payload,
      *allowance, "over-budget supplemental", 0);
  EXPECT_FALSE(refused);
  EXPECT_EQ(reclaim_calls, 1);

  // When reclaimer frees budget, SupplementalJob succeeds and retires cleanly.
  allow_reclaim = true;
  en::JobReport report;
  auto succeeded = node.SupplementalJob(
      *selected_closure, [](pr::NativeStream) { return sc::JobResult::kQueued; }, payload,
      *allowance, "reclaimed supplemental", 0, &report);
  ASSERT_TRUE(succeeded) << llmp::test_support::Failed(succeeded).value_or("");
  EXPECT_EQ(reclaim_calls, 2);
  EXPECT_EQ(report.retirement, en::JobReport::Retirement::kRetired);

  ASSERT_TRUE(node.EndRequest(0));
  ASSERT_TRUE(node.TearDown({}));
}

}  // namespace
