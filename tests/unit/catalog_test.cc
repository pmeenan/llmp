// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The base identities and byte counts, and the catalog's state machine,
// closures, leases, registrations and occupancy, against the pager
// invariants they carry (docs/architecture.md#mandatory-pager-invariants-18).

#include "catalog/catalog.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "base/bytes.h"
#include "base/check.h"
#include "base/ids.h"
#include "expected_error.h"

namespace {

using jitllm::base::Bytes;
using jitllm::test_support::Failed;
using jitllm::base::operator""_KiB;
using jitllm::base::operator""_MiB;
using jitllm::catalog::Catalog;
using jitllm::catalog::CatalogError;
using jitllm::catalog::Closure;
using jitllm::catalog::DomainId;
using jitllm::catalog::ExtentDescriptor;
using jitllm::catalog::ExtentId;
using jitllm::catalog::ExtentState;
using jitllm::catalog::MemoryClass;
using jitllm::catalog::Operation;
using jitllm::catalog::Range;
using jitllm::catalog::Recovery;
using jitllm::catalog::ResourceId;
using ::testing::ElementsAre;
using ::testing::Pair;

// Large enough that no test here meets it.
constexpr Bytes kBudget = 1024_MiB;

struct ThingTag {
  static constexpr const char* kName = "thing";
};

TEST(SlotTable, StaleIdentitiesNameNothing) {
  jitllm::base::SlotTable<ThingTag, int> table;
  const auto first = table.Insert(1);
  ASSERT_TRUE(first.valid());
  EXPECT_EQ(*table.Find(first), 1);
  EXPECT_TRUE(table.Erase(first));
  EXPECT_EQ(table.Find(first), nullptr);
  EXPECT_FALSE(table.Erase(first));
  const auto second = table.Insert(2);  // reuses the slot, a new generation
  EXPECT_EQ(second.index(), first.index());
  EXPECT_NE(second.generation(), first.generation());
  EXPECT_EQ(table.Find(first), nullptr);
  EXPECT_EQ(*table.Find(second), 2);
  EXPECT_EQ(table.Find({}), nullptr);
  EXPECT_EQ(second.ToString(), std::string("thing#0.2"));
}

TEST(Bytes, ArithmeticIsChecked) {
  static_assert(0_KiB == Bytes());
  static_assert(0_MiB == Bytes());
  static_assert(2_KiB == Bytes(2048));
  static_assert(18014398509481983_KiB == Bytes(UINT64_MAX - 1023));
  static_assert(17592186044415_MiB == Bytes(UINT64_MAX - 1048575));
  EXPECT_EQ(Bytes(2).Plus(Bytes(3)), Bytes(5));
  EXPECT_FALSE(Bytes(UINT64_MAX).Plus(Bytes(1)).has_value());
  EXPECT_EQ(Bytes(5).Minus(Bytes(3)), Bytes(2));
  EXPECT_FALSE(Bytes(3).Minus(Bytes(5)).has_value());
  EXPECT_EQ(2_MiB, Bytes(std::uint64_t{2} * 1024 * 1024));
}

TEST(CheckDeathTest, AViolatedInvariantIsFatal) {
  EXPECT_DEATH(jitllm::base::Check(false, "the test's invariant"),
               "internal invariant violated: the test's invariant");
}

class CatalogTest : public ::testing::Test {
 protected:
  void SetUp() override { domain_ = catalog_.AddDomain("spark"); }

  ExtentId Weights(std::uint32_t chunk, Bytes size = 2_MiB) {
    ExtentDescriptor descriptor{.domain = domain_,
                                .memory_class = MemoryClass::kWeights,
                                .recovery = Recovery::kFromArtifact,
                                .size = size,
                                .content = {}};
    descriptor.content.chunk = chunk;
    auto extent = catalog_.AddExtent(descriptor);
    EXPECT_TRUE(extent.has_value());
    return extent.value_or(ExtentId{});
  }

  void Load(ExtentId extent) {
    auto ticket = catalog_.BeginLoad(extent, kBudget);
    ASSERT_TRUE(ticket.has_value());
    ASSERT_TRUE(catalog_.CompleteLoad(*ticket).has_value());
  }

  ExtentState State(ExtentId extent) { return catalog_.Describe(extent).value().state; }

  Closure Of(std::initializer_list<ExtentId> extents) {
    const std::vector<ExtentId> list(extents);
    return catalog_.ClosureOfExtents(list).value();
  }

  Catalog catalog_;
  DomainId domain_;
};

TEST_F(CatalogTest, LoadsEvictsAndAdvancesGenerations) {
  const ExtentId extent = Weights(0);
  EXPECT_EQ(State(extent), ExtentState::kNonresident);
  EXPECT_EQ(catalog_.OccupancyOf(domain_).Total(), Bytes());
  auto ticket = catalog_.BeginLoad(extent, kBudget);
  ASSERT_TRUE(ticket.has_value());
  EXPECT_EQ(catalog_.OccupancyOf(domain_).loading, 2_MiB);
  ASSERT_TRUE(catalog_.CompleteLoad(*ticket).has_value());
  EXPECT_EQ(catalog_.OccupancyOf(domain_).idle, 2_MiB);
  auto evict = catalog_.BeginEvict(extent);
  ASSERT_TRUE(evict.has_value());
  EXPECT_EQ(catalog_.OccupancyOf(domain_).evicting, 2_MiB);
  ASSERT_TRUE(catalog_.CompleteEvict(*evict).has_value());
  const auto view = catalog_.Describe(extent).value();
  EXPECT_EQ(view.state, ExtentState::kNonresident);
  EXPECT_EQ(view.backing_generation, 2U);
  EXPECT_EQ(view.content_generation, 1U);  // artifact contents are unchanged
  EXPECT_EQ(catalog_.OccupancyOf(domain_).Total(), Bytes());
}

// Invariants 2 and 3: a ticket names one operation. A late completion of
// an earlier load or eviction cannot publish or release anything, even
// when a newer operation of the same kind is in progress.
TEST_F(CatalogTest, StaleTicketsCannotTouchANewerOperation) {
  const ExtentId extent = Weights(0);
  auto first = catalog_.BeginLoad(extent, kBudget).value();
  ASSERT_TRUE(catalog_.FailLoad(first, /*completion_known=*/true).has_value());
  auto second = catalog_.BeginLoad(extent, kBudget).value();
  EXPECT_EQ(Failed(catalog_.CompleteLoad(first)), CatalogError::kStaleTicket);
  EXPECT_EQ(Failed(catalog_.FailLoad(first, false)), CatalogError::kStaleTicket);
  EXPECT_EQ(State(extent), ExtentState::kLoading);
  ASSERT_TRUE(catalog_.CompleteLoad(second).has_value());
  // An eviction cancelled, then begun again: the first ticket is dead.
  auto cancelled = catalog_.BeginEvict(extent).value();
  ASSERT_TRUE(catalog_.CancelEvict(cancelled).has_value());
  auto evict = catalog_.BeginEvict(extent).value();
  EXPECT_EQ(Failed(catalog_.CompleteEvict(cancelled)), CatalogError::kStaleTicket);
  EXPECT_EQ(Failed(catalog_.QuarantineEviction(cancelled)), CatalogError::kStaleTicket);
  EXPECT_EQ(State(extent), ExtentState::kEvicting);
  // A load ticket cannot finish an eviction, nor the reverse.
  const jitllm::catalog::Ticket as_load{
      .extent = extent, .operation = Operation::kLoad, .serial = evict.serial};
  EXPECT_EQ(Failed(catalog_.CompleteLoad(as_load)), CatalogError::kStaleTicket);
  ASSERT_TRUE(catalog_.CompleteEvict(evict).has_value());
  EXPECT_EQ(Failed(catalog_.CompleteEvict(evict)), CatalogError::kWrongState);
  auto reload = catalog_.BeginLoad(extent, kBudget).value();
  EXPECT_EQ(Failed(catalog_.CancelEvict(evict)), CatalogError::kStaleTicket);
  EXPECT_TRUE(catalog_.CompleteLoad(reload).has_value());
}

// Joining a load in progress is the resource service's job (D-048): the
// catalog starts one operation and refuses a second.
TEST_F(CatalogTest, ALoadInProgressIsNotStartedTwice) {
  const ExtentId extent = Weights(0);
  auto ticket = catalog_.BeginLoad(extent, kBudget).value();
  EXPECT_EQ(Failed(catalog_.BeginLoad(extent, kBudget)), CatalogError::kWrongState);
  EXPECT_EQ(catalog_.OccupancyOf(domain_).loading, 2_MiB);  // charged once
  ASSERT_TRUE(catalog_.CompleteLoad(ticket).has_value());
  EXPECT_EQ(Failed(catalog_.CompleteLoad(ticket)), CatalogError::kWrongState);
}

// Every materialization checks actual occupancy against B (D-050), and
// capacity an eviction will free is not available until it completes.
TEST_F(CatalogTest, LoadsNeverExceedTheBudget) {
  const ExtentId first = Weights(0);
  const ExtentId second = Weights(1);
  auto ticket = catalog_.BeginLoad(first, 3_MiB).value();
  EXPECT_EQ(Failed(catalog_.BeginLoad(second, 3_MiB)), CatalogError::kOverBudget);
  EXPECT_EQ(State(second), ExtentState::kNonresident);
  ASSERT_TRUE(catalog_.CompleteLoad(ticket).has_value());
  auto evict = catalog_.BeginEvict(first).value();
  EXPECT_EQ(Failed(catalog_.BeginLoad(second, 3_MiB)), CatalogError::kOverBudget);
  ASSERT_TRUE(catalog_.CompleteEvict(evict).has_value());
  EXPECT_TRUE(catalog_.BeginLoad(second, 4_MiB).has_value());  // exactly B
}

// Invariant 8: an unknown completion is not reclaimed memory.
TEST_F(CatalogTest, UnknownCompletionQuarantinesAndStaysCharged) {
  const ExtentId extent = Weights(0);
  auto ticket = catalog_.BeginLoad(extent, kBudget).value();
  ASSERT_TRUE(catalog_.FailLoad(ticket, /*completion_known=*/false).has_value());
  EXPECT_EQ(State(extent), ExtentState::kQuarantined);
  EXPECT_EQ(catalog_.OccupancyOf(domain_).quarantined, 2_MiB);
  EXPECT_EQ(Failed(catalog_.BeginLoad(extent, kBudget)), CatalogError::kWrongState);
  EXPECT_EQ(Failed(catalog_.BeginEvict(extent)), CatalogError::kWrongState);
  EXPECT_EQ(Failed(catalog_.RemoveExtent(extent)), CatalogError::kWrongState);

  const ExtentId other = Weights(1);
  Load(other);
  auto evict = catalog_.BeginEvict(other).value();
  ASSERT_TRUE(catalog_.QuarantineEviction(evict).has_value());
  EXPECT_EQ(catalog_.OccupancyOf(domain_).quarantined, 4_MiB);
}

// Invariant 5: shared backing is counted once.
TEST_F(CatalogTest, ClosuresCountSharedExtentsOnce) {
  const ExtentId shared = Weights(0);
  const ExtentId own = Weights(1);
  const std::vector<Range> tied = {{.extent = shared, .offset = Bytes(0), .length = 1_MiB}};
  const std::vector<Range> embedding = {{.extent = shared, .offset = 1_MiB, .length = 1_MiB},
                                        {.extent = own, .offset = Bytes(0), .length = 2_MiB}};
  const ResourceId a = catalog_.AddResource(tied).value();
  const ResourceId b = catalog_.AddResource(embedding).value();
  const std::vector<ResourceId> both = {a, b, a};
  const Closure closure = catalog_.ClosureOf(both).value();
  EXPECT_THAT(closure.extents, ElementsAre(Pair(shared, 1U), Pair(own, 1U)));
  EXPECT_EQ(closure.bytes_by_domain.at(domain_), 4_MiB);
  // A range must lie inside its extent.
  const std::vector<Range> outside = {{.extent = own, .offset = 1_MiB, .length = 2_MiB}};
  EXPECT_EQ(Failed(catalog_.AddResource(outside)), CatalogError::kBadRange);
  // An extent a resource uses cannot be forgotten.
  EXPECT_EQ(Failed(catalog_.RemoveExtent(own)), CatalogError::kWrongState);
  ASSERT_TRUE(catalog_.RemoveResource(b).has_value());
  EXPECT_TRUE(catalog_.RemoveExtent(own).has_value());
}

// Invariant 6: leasing and starting an eviction exclude each other, and a
// lease is all or none.
TEST_F(CatalogTest, LeasesAndEvictionExcludeEachOther) {
  const ExtentId a = Weights(0);
  const ExtentId b = Weights(1);
  Load(a);
  const Closure only_a = Of({a});
  const Closure both = Of({a, b});
  EXPECT_EQ(Failed(catalog_.AcquireLease(both)), CatalogError::kNotResident);
  EXPECT_EQ(catalog_.Describe(a).value().leases, 0U);  // nothing half-leased
  const auto lease = catalog_.AcquireLease(only_a).value();
  EXPECT_EQ(catalog_.OccupancyOf(domain_).held, 2_MiB);
  EXPECT_EQ(Failed(catalog_.BeginEvict(a)), CatalogError::kHeld);
  ASSERT_TRUE(catalog_.ReleaseLease(lease).has_value());
  // Releasing is not evicting: the contents stay resident and eligible.
  EXPECT_EQ(State(a), ExtentState::kResident);
  EXPECT_EQ(catalog_.OccupancyOf(domain_).idle, 2_MiB);
  EXPECT_EQ(Failed(catalog_.ReleaseLease(lease)), CatalogError::kUnknownId);
  auto evict = catalog_.BeginEvict(a).value();
  EXPECT_EQ(Failed(catalog_.AcquireLease(only_a)), CatalogError::kNotResident);
  ASSERT_TRUE(catalog_.CancelEvict(evict).has_value());
  EXPECT_TRUE(catalog_.AcquireLease(only_a).has_value());
}

TEST_F(CatalogTest, RegistrationsHoldLikeLeases) {
  const ExtentId extent = Weights(0);
  EXPECT_EQ(Failed(catalog_.AddRegistration(extent)), CatalogError::kNotResident);
  Load(extent);
  const auto registration = catalog_.AddRegistration(extent).value();
  EXPECT_EQ(Failed(catalog_.BeginEvict(extent)), CatalogError::kHeld);
  ASSERT_TRUE(catalog_.RetireRegistration(registration).has_value());
  EXPECT_TRUE(catalog_.BeginEvict(extent).has_value());
}

// Invariant 4: mutable contents are preserved or deliberately invalidated,
// and a closure taken before contents changed cannot lease the new ones.
TEST_F(CatalogTest, MutableStateMustBeInvalidatedBeforeEviction) {
  const auto state = catalog_
                         .AddExtent({.domain = domain_,
                                     .memory_class = MemoryClass::kLiveState,
                                     .recovery = Recovery::kPreserve,
                                     .size = 2_MiB,
                                     .content = {}})
                         .value();
  Load(state);
  EXPECT_EQ(Failed(catalog_.BeginEvict(state)), CatalogError::kNotEvictable);
  const Closure old = Of({state});
  const auto before = catalog_.Describe(state).value().content_generation;
  ASSERT_TRUE(catalog_.InvalidateContents(state).has_value());
  EXPECT_EQ(catalog_.Describe(state).value().content_generation, before + 1);
  EXPECT_TRUE(catalog_.Describe(state).value().discarded);
  // Recovery is intrinsic: invalidating does not relabel the extent.
  EXPECT_EQ(catalog_.Describe(state).value().descriptor.recovery, Recovery::kPreserve);
  EXPECT_EQ(Failed(catalog_.AcquireLease(old)), CatalogError::kStaleContent);
  EXPECT_EQ(Failed(catalog_.AcquireLease(Of({state}))), CatalogError::kStaleContent);  // discarded
  auto evict = catalog_.BeginEvict(state).value();
  ASSERT_TRUE(catalog_.CompleteEvict(evict).has_value());
  EXPECT_EQ(catalog_.Describe(state).value().content_generation, before + 2);  // gone
  // New contents are preserved again: not evictable until invalidated.
  Load(state);
  EXPECT_FALSE(catalog_.Describe(state).value().discarded);
  EXPECT_EQ(Failed(catalog_.BeginEvict(state)), CatalogError::kNotEvictable);
  const auto lease = catalog_.AcquireLease(Of({state})).value();
  const Closure taken = Of({state});
  ASSERT_TRUE(catalog_.ReplaceContents(state).has_value());  // the writer, under its lease
  EXPECT_EQ(Failed(catalog_.AcquireLease(taken)), CatalogError::kStaleContent);
  EXPECT_EQ(Failed(catalog_.BeginEvict(state)), CatalogError::kHeld);
  // Leased contents cannot be invalidated under their readers.
  EXPECT_EQ(Failed(catalog_.InvalidateContents(state)), CatalogError::kHeld);
  ASSERT_TRUE(catalog_.ReleaseLease(lease).has_value());
  EXPECT_EQ(Failed(catalog_.BeginEvict(state)), CatalogError::kNotEvictable);
}

// An owner may take back resident state it zeroed and then invalidated
// (a cleared conversation's backing kept for reuse): only unheld, only
// discarded, and at a new generation that no earlier closure can lease.
TEST_F(CatalogTest, DiscardedStateIsRevivedOnlyByItsUnheldOwnerAtANewGeneration) {
  const auto state = catalog_
                         .AddExtent({.domain = domain_,
                                     .memory_class = MemoryClass::kLiveState,
                                     .recovery = Recovery::kPreserve,
                                     .size = 2_MiB,
                                     .content = {}})
                         .value();
  Load(state);
  EXPECT_EQ(Failed(catalog_.ReviveDiscarded(state)), CatalogError::kWrongState);  // live
  ASSERT_TRUE(catalog_.InvalidateContents(state).has_value());
  const Closure discarded = Of({state});
  const auto before = catalog_.Describe(state).value().content_generation;
  ASSERT_TRUE(catalog_.ReviveDiscarded(state).has_value());
  EXPECT_FALSE(catalog_.Describe(state).value().discarded);
  EXPECT_EQ(catalog_.Describe(state).value().content_generation, before + 1);
  EXPECT_EQ(Failed(catalog_.AcquireLease(discarded)), CatalogError::kStaleContent);
  // Live again: preserved, so not evictable until invalidated.
  EXPECT_EQ(Failed(catalog_.BeginEvict(state)), CatalogError::kNotEvictable);
  EXPECT_EQ(Failed(catalog_.ReviveDiscarded(state)), CatalogError::kWrongState);
  const auto lease = catalog_.AcquireLease(Of({state})).value();
  ASSERT_TRUE(catalog_.ReleaseLease(lease).has_value());
  // Registered: not revivable. Discarded: not registrable.
  const auto registration = catalog_.AddRegistration(state).value();
  EXPECT_EQ(Failed(catalog_.InvalidateContents(state)), CatalogError::kHeld);
  ASSERT_TRUE(catalog_.RetireRegistration(registration).has_value());
  ASSERT_TRUE(catalog_.InvalidateContents(state).has_value());
  EXPECT_EQ(Failed(catalog_.AddRegistration(state)), CatalogError::kStaleContent);
  // Reclaimed meanwhile: nothing resident to take back.
  auto evict = catalog_.BeginEvict(state).value();
  ASSERT_TRUE(catalog_.CompleteEvict(evict).has_value());
  EXPECT_EQ(Failed(catalog_.ReviveDiscarded(state)), CatalogError::kWrongState);
  const auto weights = catalog_
                           .AddExtent({.domain = domain_,
                                       .memory_class = MemoryClass::kWeights,
                                       .recovery = Recovery::kFromArtifact,
                                       .size = 2_MiB,
                                       .content = {}})
                           .value();
  Load(weights);
  EXPECT_EQ(Failed(catalog_.ReviveDiscarded(weights)), CatalogError::kWrongState);
}

// Write-back (invariant 4): live mutable contents may be evicted only by
// an eviction that writes them back. A completed one keeps the content
// generation and marks the contents preserved, so a closure taken before
// is still current and a load restores it; an abandoned one leaves them
// resident and unpreserved. Invalidated contents and other classes are
// never written back.
TEST_F(CatalogTest, AWriteBackEvictionPreservesTheContentGeneration) {
  const auto state = catalog_
                         .AddExtent({.domain = domain_,
                                     .memory_class = MemoryClass::kLiveState,
                                     .recovery = Recovery::kPreserve,
                                     .size = 2_MiB,
                                     .content = {}})
                         .value();
  const auto weights = catalog_
                           .AddExtent({.domain = domain_,
                                       .memory_class = MemoryClass::kWeights,
                                       .recovery = Recovery::kFromArtifact,
                                       .size = 2_MiB,
                                       .content = {}})
                           .value();
  Load(state);
  Load(weights);
  EXPECT_EQ(Failed(catalog_.BeginEvict(weights, true)), CatalogError::kNotEvictable);
  const Closure taken = Of({state});
  const auto contents = catalog_.Describe(state).value().content_generation;

  // Abandoned: resident again, nothing preserved.
  auto abandoned = catalog_.BeginEvict(state, true).value();
  EXPECT_EQ(Failed(catalog_.AcquireLease(taken)), CatalogError::kNotResident);
  ASSERT_TRUE(catalog_.CancelEvict(abandoned).has_value());
  EXPECT_EQ(catalog_.Describe(state).value().state, ExtentState::kResident);
  EXPECT_FALSE(catalog_.Describe(state).value().preserved);
  // A plain eviction of the same contents is still refused.
  EXPECT_EQ(Failed(catalog_.BeginEvict(state)), CatalogError::kNotEvictable);

  auto written = catalog_.BeginEvict(state, true).value();
  ASSERT_TRUE(catalog_.CompleteEvict(written).has_value());
  EXPECT_EQ(catalog_.Describe(state).value().state, ExtentState::kNonresident);
  EXPECT_TRUE(catalog_.Describe(state).value().preserved);
  EXPECT_EQ(catalog_.Describe(state).value().content_generation, contents);
  // A failed restore keeps them preserved; a completed one hands the live
  // copy back, under the closure taken before.
  auto failed = catalog_.BeginLoad(state, kBudget).value();
  ASSERT_TRUE(catalog_.FailLoad(failed, true).has_value());
  EXPECT_TRUE(catalog_.Describe(state).value().preserved);
  Load(state);
  EXPECT_FALSE(catalog_.Describe(state).value().preserved);
  const auto lease = catalog_.AcquireLease(taken).value();
  ASSERT_TRUE(catalog_.ReleaseLease(lease).has_value());

  // Invalidated contents are evicted plainly, never written back.
  ASSERT_TRUE(catalog_.InvalidateContents(state).has_value());
  EXPECT_EQ(Failed(catalog_.BeginEvict(state, true)), CatalogError::kNotEvictable);
  auto plain = catalog_.BeginEvict(state).value();
  ASSERT_TRUE(catalog_.CompleteEvict(plain).has_value());
  EXPECT_FALSE(catalog_.Describe(state).value().preserved);
  EXPECT_EQ(catalog_.Describe(state).value().content_generation, contents + 2);
}

TEST_F(CatalogTest, ReplacingContentsRequiresExclusiveAccess) {
  const auto state = catalog_
                         .AddExtent({.domain = domain_,
                                     .memory_class = MemoryClass::kLiveState,
                                     .recovery = Recovery::kPreserve,
                                     .size = 2_MiB,
                                     .content = {}},
                                    /*resident=*/true)
                         .value();
  const Closure contents = Of({state});
  const auto writer = catalog_.AcquireLease(contents).value();
  const auto reader = catalog_.AcquireLease(contents).value();
  EXPECT_EQ(Failed(catalog_.ReplaceContents(state)), CatalogError::kHeld);
  EXPECT_EQ(catalog_.Describe(state).value().content_generation, 1U);
  ASSERT_TRUE(catalog_.ReleaseLease(reader).has_value());
  const auto registration = catalog_.AddRegistration(state).value();
  EXPECT_EQ(Failed(catalog_.ReplaceContents(state)), CatalogError::kHeld);
  EXPECT_EQ(catalog_.Describe(state).value().content_generation, 1U);
  ASSERT_TRUE(catalog_.RetireRegistration(registration).has_value());
  ASSERT_TRUE(catalog_.ReplaceContents(state).has_value());
  EXPECT_EQ(Failed(catalog_.AcquireLease(contents)), CatalogError::kStaleContent);
  EXPECT_TRUE(catalog_.ReleaseLease(writer).has_value());
}

// Artifact contents are never replaced in place; a reload restores them
// at the same content generation.
TEST_F(CatalogTest, ForgettingSavedStateInvalidatesOldClosuresAndCannotForgetImmutableContents) {
  const auto state = catalog_
                         .AddExtent({.domain = domain_,
                                     .memory_class = MemoryClass::kLiveState,
                                     .recovery = Recovery::kPreserve,
                                     .size = 2_MiB,
                                     .content = {}})
                         .value();
  Load(state);
  EXPECT_EQ(Failed(catalog_.ForgetPreserved(state)), CatalogError::kWrongState);
  const auto before = Of({state});
  const auto ticket = catalog_.BeginEvict(state, true).value();
  EXPECT_EQ(Failed(catalog_.ForgetPreserved(state)), CatalogError::kWrongState);
  ASSERT_TRUE(catalog_.CompleteEvict(ticket));
  ASSERT_TRUE(catalog_.Describe(state)->preserved);
  const auto generation = catalog_.Describe(state)->content_generation;
  ASSERT_TRUE(catalog_.ForgetPreserved(state));
  EXPECT_FALSE(catalog_.Describe(state)->preserved);
  EXPECT_GT(catalog_.Describe(state)->content_generation, generation);
  EXPECT_EQ(catalog_.OccupancyOf(domain_).Total(), Bytes());
  Load(state);
  EXPECT_EQ(Failed(catalog_.AcquireLease(before)), CatalogError::kStaleContent);
  EXPECT_EQ(Failed(catalog_.ForgetPreserved(Weights(7))), CatalogError::kWrongState);
  EXPECT_EQ(Failed(catalog_.ForgetPreserved({})), CatalogError::kUnknownId);
}

TEST_F(CatalogTest, ArtifactContentsAreNotReplaced) {
  const ExtentId extent = Weights(0);
  Load(extent);
  EXPECT_EQ(Failed(catalog_.ReplaceContents(extent)), CatalogError::kWrongState);
  const Closure closure = Of({extent});
  auto evict = catalog_.BeginEvict(extent).value();
  ASSERT_TRUE(catalog_.CompleteEvict(evict).has_value());
  Load(extent);
  EXPECT_TRUE(catalog_.AcquireLease(closure).has_value());
}

// Unknown allocations are charged and never evictable (D-006).
TEST_F(CatalogTest, UnknownAllocationsArePinned) {
  const auto unknown = catalog_
                           .AddExtent({.domain = domain_,
                                       .memory_class = MemoryClass::kUnknown,
                                       .recovery = Recovery::kPinned,
                                       .size = 3_MiB,
                                       .content = {}},
                                      /*resident=*/true)
                           .value();
  const auto occupancy = catalog_.OccupancyOf(domain_);
  EXPECT_EQ(occupancy.pinned, 3_MiB);
  EXPECT_EQ(occupancy.by_class.at(static_cast<std::size_t>(MemoryClass::kUnknown)), 3_MiB);
  EXPECT_EQ(Failed(catalog_.BeginEvict(unknown)), CatalogError::kNotEvictable);
  EXPECT_EQ(Failed(catalog_.InvalidateContents(unknown)), CatalogError::kWrongState);
  EXPECT_EQ(Failed(catalog_.ReplaceContents(unknown)), CatalogError::kWrongState);
  EXPECT_FALSE(Catalog::Evictable(catalog_.Describe(unknown).value()));
  // An unknown allocation that is not pinned is refused.
  EXPECT_EQ(Failed(catalog_.AddExtent({.domain = domain_,
                                       .memory_class = MemoryClass::kUnknown,
                                       .recovery = Recovery::kFromArtifact,
                                       .size = 2_MiB,
                                       .content = {}},
                                      /*resident=*/true)),
            CatalogError::kBadRange);
  // Its owner releases it once nothing holds it.
  const auto registration = catalog_.AddRegistration(unknown).value();
  EXPECT_EQ(Failed(catalog_.ReleasePinned(unknown)), CatalogError::kHeld);
  ASSERT_TRUE(catalog_.RetireRegistration(registration).has_value());
  ASSERT_TRUE(catalog_.ReleasePinned(unknown).has_value());
  EXPECT_EQ(State(unknown), ExtentState::kNonresident);
  EXPECT_EQ(catalog_.OccupancyOf(domain_).Total(), Bytes());
  EXPECT_EQ(Failed(catalog_.ReleasePinned(Weights(1))), CatalogError::kWrongState);
}

// Descriptors come from untrusted manifests: out-of-range fields and sizes
// that would overflow the domain's total are refused, never fatal.
TEST_F(CatalogTest, DescriptorsAreChecked) {
  const auto add = [&](MemoryClass memory_class, Recovery recovery, Bytes size) {
    return catalog_.AddExtent({.domain = domain_,
                               .memory_class = memory_class,
                               .recovery = recovery,
                               .size = size,
                               .content = {}},
                              /*resident=*/true);
  };
  // Deliberately out of range, as a corrupt manifest could be.
  const auto bad_class =
      static_cast<MemoryClass>(9);  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
  const auto bad_recovery =
      static_cast<Recovery>(4);  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_EQ(Failed(add(bad_class, Recovery::kPinned, 2_MiB)), CatalogError::kBadRange);
  EXPECT_EQ(Failed(add(MemoryClass::kScratch, bad_recovery, 2_MiB)), CatalogError::kBadRange);
  EXPECT_EQ(Failed(add(MemoryClass::kScratch, Recovery::kDiscardable, Bytes())),
            CatalogError::kBadRange);
  const Bytes half(std::uint64_t{1} << 63);
  ASSERT_TRUE(add(MemoryClass::kScratch, Recovery::kDiscardable, half).has_value());
  EXPECT_EQ(Failed(add(MemoryClass::kScratch, Recovery::kDiscardable, half)),
            CatalogError::kBadRange);
  EXPECT_EQ(catalog_.OccupancyOf(domain_).Total(), half);
}

// Only the writer holding a lease replaces preserved contents; artifact
// contents cannot be invalidated; discarded scratch is usable once reloaded.
TEST_F(CatalogTest, ContentChangesNeedTheRightHolder) {
  const auto scratch = catalog_
                           .AddExtent({.domain = domain_,
                                       .memory_class = MemoryClass::kScratch,
                                       .recovery = Recovery::kDiscardable,
                                       .size = 2_MiB,
                                       .content = {}})
                           .value();
  const auto state = catalog_
                         .AddExtent({.domain = domain_,
                                     .memory_class = MemoryClass::kLiveState,
                                     .recovery = Recovery::kPreserve,
                                     .size = 2_MiB,
                                     .content = {}})
                         .value();
  const ExtentId weights = Weights(0);
  Load(scratch);
  Load(state);
  Load(weights);
  EXPECT_EQ(Failed(catalog_.ReplaceContents(state)), CatalogError::kWrongState);  // no lease
  EXPECT_EQ(Failed(catalog_.InvalidateContents(weights)), CatalogError::kWrongState);
  ASSERT_TRUE(catalog_.InvalidateContents(scratch).has_value());
  auto evict = catalog_.BeginEvict(scratch).value();
  ASSERT_TRUE(catalog_.CompleteEvict(evict).has_value());
  EXPECT_FALSE(catalog_.Describe(scratch).value().discarded);
  Load(scratch);
  EXPECT_TRUE(catalog_.AcquireLease(Of({scratch})).has_value());
}

TEST_F(CatalogTest, DomainsAreSeparateAndIdentitiesChecked) {
  const DomainId other = catalog_.AddDomain("spark-b");
  const ExtentId here = Weights(0);
  const auto there = catalog_
                         .AddExtent({.domain = other,
                                     .memory_class = MemoryClass::kWeights,
                                     .recovery = Recovery::kFromArtifact,
                                     .size = 2_MiB,
                                     .content = {}})
                         .value();
  Load(there);
  EXPECT_EQ(catalog_.OccupancyOf(domain_).Total(), Bytes());
  EXPECT_EQ(catalog_.OccupancyOf(other).Total(), 2_MiB);
  // A resource cannot span domains.
  const std::vector<Range> spanning = {{.extent = here, .offset = Bytes(0), .length = 1_MiB},
                                       {.extent = there, .offset = Bytes(0), .length = 1_MiB}};
  EXPECT_EQ(Failed(catalog_.AddResource(spanning)), CatalogError::kBadRange);
  EXPECT_EQ(Failed(catalog_.AddExtent({.domain = DomainId{},
                                       .memory_class = MemoryClass::kWeights,
                                       .recovery = Recovery::kFromArtifact,
                                       .size = 2_MiB,
                                       .content = {}})),
            CatalogError::kUnknownDomain);
  EXPECT_EQ(Failed(catalog_.Describe(ExtentId{})), CatalogError::kUnknownId);
}

TEST_F(CatalogTest, UseIsRecordedThroughLeases) {
  const ExtentId extent = Weights(0);
  Load(extent);
  const auto lease = catalog_.AcquireLease(Of({extent})).value();
  ASSERT_TRUE(catalog_.RecordUse(lease, 7).has_value());
  ASSERT_TRUE(catalog_.RecordUse(lease, 5).has_value());  // never backwards
  EXPECT_EQ(catalog_.Describe(extent).value().last_use, 7U);
}

}  // namespace
