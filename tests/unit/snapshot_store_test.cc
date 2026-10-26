// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/snapshot_store.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "engine/paged_node.h"
#include "expected_error.h"
#include "llm_fixture.h"
#include "providers/fake/fake_device_execution.h"
#include "providers/fake/fake_device_runtime.h"
#include "runtime/serving.h"

namespace llmp::runtime {
// Impossible producer metadata is tested at the store boundary. Actual state
// copies always use the concrete Llm and fake device. The address access below
// is confined to lifetime/free fault injection, never a production snapshot API.
struct SnapshotStoreTestAccess {
  static Status Prepare(SnapshotStore& store, const std::shared_ptr<const std::byte>& issuer,
                        std::optional<base::Sha256Digest> identity = {},
                        std::string_view layout = {},
                        std::span<const engine::LiveState::Range> ranges = {}) {
    return store.Prepare(issuer, identity, layout, 0, ranges, {}, execution::AdaptiveDepth{});
  }
  static StateSnapshot Publish(SnapshotStore& store) { return store.Publish(); }
  static Status Read(const SnapshotStore& store, const StateSnapshot& saved,
                     const std::shared_ptr<const std::byte>& issuer,
                     std::optional<base::Sha256Digest> identity, std::string_view layout) {
    auto read = store.Read(saved, issuer, identity, layout);
    if (!read) return std::unexpected(read.error());
    return {};
  }
  static void* BorrowedAddress(const SnapshotStore& store) { return store.WriteAddress(); }
};
}  // namespace llmp::runtime

namespace {
namespace en = llmp::engine;
namespace rt = llmp::runtime;
namespace cat = llmp::catalog;
namespace fake = llmp::providers::fake;
namespace fixture = llmp::test_support::llm_fixture;
using llmp::base::Bytes;
using llmp::test_support::Failed;
using Access = rt::SnapshotStoreTestAccess;
using Range = en::LiveState::Range;
constexpr std::uint64_t kRoom = 2 * en::kPagedExtent;

llmp::base::Sha256Digest Seal(std::string_view value) {
  return llmp::base::Sha256().Update(value).Finish();
}

class SnapshotStoreTest : public fixture::LlmFixture {
 protected:
  void TearDown() override {
    fake::ClearCopyFailure();
    fake::ClearPinnedFreeHook();
    fixture::LlmFixture::TearDown();
  }
  // Pure store arithmetic needs no execution producer or model registration.
  void StartBoundaryNode() {
    node_ = std::make_unique<en::PagedNode>(en::NodeSettings{});
    ASSERT_TRUE(node_->Open());
    const auto fixed = node_->catalog().OccupancyOf(node_->domain()).Total().value();
    ASSERT_TRUE(node_->Start(Bytes(fixed + kRoom)));
    node_->Run();
  }
  void Warm() {
    Open({&BranchAt(0)});
    std::vector<float> logits;
    ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 3>{4, 5, 6}, logits));
  }
  std::uint64_t Occupancy() {
    std::uint64_t result = 0;
    EXPECT_TRUE(node_->Call(
        [&]() -> en::Status {
          result = node_->catalog().OccupancyOf(node_->domain()).Total().value();
          return {};
        },
        "snapshot occupancy"));
    return result;
  }
  std::uint64_t PinnedSpanBytes(const void* pointer) {
    std::uint64_t bytes = 0;
    EXPECT_TRUE(node_->Call(
        [&]() -> en::Status {
          node_->EraseSpans([&](const en::Span& span) {
            if (span.base == reinterpret_cast<std::uintptr_t>(pointer)) bytes = span.size;
            return false;
          });
          return {};
        },
        "snapshot retained span"));
    return bytes;
  }
  std::uint64_t Operations() {
    auto stats = node_->Stats();
    EXPECT_TRUE(stats) << Failed(stats).value_or("");
    return stats ? stats->held_operations : 0;
  }
  std::size_t CopyBytes() const {
    return llm_->runner().slots().live(0, 0).used_ranges().front().bytes;
  }
  llmp::base::Sha256Digest PayloadDigest() const {
    llmp::base::Sha256 digest;
    const auto& slots = llm_->runner().slots();
    for (std::size_t stage = 0; stage < slots.stages(); ++stage) {
      const auto& live = slots.live(0, stage);
      for (const auto& range : live.used_ranges()) {
        digest.Update({reinterpret_cast<const std::byte*>(live.base(range.region) + range.offset),
                       static_cast<std::size_t>(range.bytes)});
      }
    }
    return digest.Finish();
  }
};

TEST_F(SnapshotStoreTest, SingleAllocationSeparatesPayloadFromMetadataAndReleasesTheExactCharge) {
  execution_identity_ = Seal("fake recipe A");
  Start(false, false);
  Warm();
  ObserveDepth(0, Depth(0).maximum(), 1);
  const auto original_decoding = Depth(0);
  const auto expected = PayloadDigest();
  const auto image = NativeImage(0);
  const auto before = Occupancy();
  rt::StateSnapshot capability;
  std::uint64_t before_release = 0, charge = 0;
  {
    rt::SnapshotStore store(*node_, kRoom);
    auto saved = BranchAt(0).CaptureState(store);
    ASSERT_TRUE(saved) << Failed(saved).value_or("");
    capability = std::move(*saved);
    ASSERT_TRUE(store.Digest(capability));
    EXPECT_EQ(*store.Digest(capability), expected);
    EXPECT_GT(store.capacity(), CopyBytes() + 3 * sizeof(std::int32_t));
    EXPECT_EQ(Occupancy() - before, store.capacity());
    ObserveDepth(0, Depth(0).maximum(), 1);
    ASSERT_NE(Depth(0), original_decoding);
    std::vector<float> logits;
    ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 2>{7, 8}, logits));
    ASSERT_TRUE(BranchAt(0).RestoreState(store, capability));
    EXPECT_EQ(BranchAt(0).history(), (std::vector<std::int32_t>{4, 5, 6}));
    EXPECT_EQ(llm_->runner().slots().positions(0), 3U);
    EXPECT_EQ(Depth(0), original_decoding);
    EXPECT_EQ(NativeImage(0), image);
    EXPECT_EQ(PayloadDigest(), expected);
    EXPECT_EQ(*store.Digest(capability), expected);  // excludes metadata and aligned padding
    before_release = Occupancy();
    charge = store.capacity();
  }
  // The intervening prefill may add a plan; only the store's exact allocation
  // is returned here, independently of the runner's retained plan cache.
  EXPECT_EQ(Occupancy(), before_release - charge);
  rt::SnapshotStore foreign(*node_, kRoom);
  EXPECT_FALSE(foreign.Digest(capability));
}

TEST_F(SnapshotStoreTest, EmptyPayloadChargesThePinnedMinimumAndBoundsItBeforeAllocation) {
  StartBoundaryNode();
  const auto before = Occupancy();
  const auto issuer = std::make_shared<const std::byte>();
  rt::SnapshotStore too_small(*node_, 255);
  EXPECT_FALSE(Access::Prepare(too_small, issuer));
  EXPECT_EQ(too_small.capacity(), 0U);
  EXPECT_EQ(Occupancy(), before);
  rt::SnapshotStore enough(*node_, 256);
  ASSERT_TRUE(Access::Prepare(enough, issuer));
  auto saved = Access::Publish(enough);
  EXPECT_EQ(enough.capacity(), 256U);
  EXPECT_EQ(Occupancy() - before, 256U);
  ASSERT_TRUE(enough.Digest(saved));
  EXPECT_EQ(*enough.Digest(saved), llmp::base::Sha256().Finish());
}

TEST_F(SnapshotStoreTest,
       ForeignExecutionLayoutUnknownIdentityAndStaleGenerationRefuseBeforeEffects) {
  execution_identity_ = Seal("fake recipe A");
  Start(false, false);
  Warm();
  rt::SnapshotStore store(*node_, kRoom);
  auto saved = BranchAt(0).CaptureState(store);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  const auto operations = Operations();
  const auto queued = source_->queued();
  const auto image = NativeImage(0);
  const auto capacity = store.capacity();
  const auto refused = [&] {
    EXPECT_FALSE(BranchAt(0).RestoreState(store, *saved));
    EXPECT_EQ(Operations(), operations);
    EXPECT_EQ(source_->queued(), queued);
    EXPECT_EQ(NativeImage(0), image);
    EXPECT_EQ(store.capacity(), capacity);
  };
  execution_identity_ = Seal("fake recipe B");
  refused();
  execution_identity_ = std::unexpected("unknown test execution identity");
  refused();
  execution_identity_ = Seal("fake recipe A");
  EXPECT_FALSE(BranchAt(1).RestoreState(store, *saved));
  EXPECT_EQ(Operations(), operations);
  // Layout and identity are immutable after real setup. Test the store's
  // independent comparisons directly, without forging a production source.
  const auto issuer = std::make_shared<const std::byte>();
  rt::SnapshotStore boundary(*node_, kRoom);
  ASSERT_TRUE(Access::Prepare(boundary, issuer, Seal("A"), "layout A"));
  auto control = Access::Publish(boundary);
  EXPECT_FALSE(Access::Read(boundary, control, issuer, Seal("B"), "layout A"));
  EXPECT_FALSE(Access::Read(boundary, control, issuer, Seal("A"), "layout B"));
  EXPECT_TRUE(Access::Read(boundary, control, issuer, Seal("A"), "layout A"));
  auto next = BranchAt(0).CaptureState(store);
  ASSERT_TRUE(next) << Failed(next).value_or("");
  EXPECT_FALSE(store.Digest(*saved));
  EXPECT_TRUE(store.Digest(*next));
  const auto after_capture = Operations();
  EXPECT_FALSE(BranchAt(0).RestoreState(store, *saved));
  EXPECT_EQ(Operations(), after_capture);
}

TEST_F(SnapshotStoreTest, ExpiredIssuerRejectsDirectReadsWhileNodeLivesAndAllowsNewCapture) {
  compute_streams_ = 2;
  std::unique_ptr<rt::Llm> next;
  Start(false, false, [&](rt::Llm&) {
    auto entry = entry_;
    entry.name = "next";
    next = std::make_unique<rt::Llm>(*node_, entry, Settings(false), roles_, 1, Build(false));
    ASSERT_TRUE(next->Setup());
    ASSERT_TRUE(next->Register());
  });
  ASSERT_NE(next, nullptr);
  ASSERT_TRUE(next->Bind());
  Warm();
  rt::SnapshotStore store(*node_, kRoom);
  auto saved = BranchAt(0).CaptureState(store);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  const auto capacity = store.capacity();
  Close();
  llm_.reset();
  EXPECT_TRUE(store.usable());
  EXPECT_FALSE(store.Digest(*saved));
  llm_ = std::move(next);
  EXPECT_FALSE(BranchAt(0).RestoreState(store, *saved));
  const std::array<rt::Llm::Branch*, 1> selected{&BranchAt(0)};
  ASSERT_TRUE(llm_->SelectBranches(selected));
  ASSERT_TRUE(node_->BeginRequest(llm_->paged().stream(), llm_->request_closure(), "next issuer"));
  std::vector<float> logits;
  ASSERT_TRUE(BranchAt(0).Prefill(std::array<std::int32_t, 3>{4, 5, 6}, logits));
  auto current = BranchAt(0).CaptureState(store);
  ASSERT_TRUE(current) << Failed(current).value_or("");
  EXPECT_EQ(store.capacity(), capacity);
  EXPECT_TRUE(store.Digest(*current));
  EXPECT_FALSE(store.Digest(*saved));
}

// A capture whose copy failed with its fence seen ended: the store keeps its
// allocation (unpublished) and serves the next capture.
TEST_F(SnapshotStoreTest, AFailedCaptureWhoseEndWasProvenKeepsTheStoreAndItsAllocation) {
  Start(false, false);
  Warm();
  const auto before = Occupancy();
  rt::SnapshotStore store(*node_, kRoom);
  auto previous = BranchAt(0).CaptureState(store);
  ASSERT_TRUE(previous) << Failed(previous).value_or("");
  const auto charge = store.capacity();
  fake::FailNextCopy(llmp::providers::CopyKind::kDeviceToHost, CopyBytes(), true);
  EXPECT_FALSE(BranchAt(0).CaptureState(store));
  EXPECT_TRUE(store.usable());
  EXPECT_FALSE(store.Digest(*previous));  // its generation was replaced
  EXPECT_EQ(node_->kept_pinned(), 0U);
  EXPECT_EQ(Occupancy() - before, charge);
}

// A capture whose copy's fence was lost: its allocation goes to the node's
// quarantine on the model's stream, whole and charged, even after its issuer
// and store retire, and the store refuses until a fence of that stream frees
// it; then it serves again.
TEST_F(SnapshotStoreTest, AnUnprovenCaptureQuarantinesItsAllocationUntilItsStreamsFence) {
  Start(false, false);
  Warm();
  const auto before = Occupancy();
  auto store = std::make_unique<rt::SnapshotStore>(*node_, kRoom);
  auto previous = BranchAt(0).CaptureState(*store);
  ASSERT_TRUE(previous) << Failed(previous).value_or("");
  const auto charge = store->capacity();
  static_cast<llmp::providers::fake::FakeDeviceExecution&>(node_->execution())
      .FailNextQuery(llmp::providers::ProviderError::kUnknown);
  EXPECT_FALSE(BranchAt(0).CaptureState(*store));
  EXPECT_FALSE(store->usable());
  EXPECT_FALSE(store->Digest(*previous));
  EXPECT_EQ(node_->kept_pinned(), 1U);
  EXPECT_EQ(Occupancy() - before, charge);
  const auto operations = Operations();
  EXPECT_FALSE(BranchAt(0).CaptureState(*store));
  EXPECT_EQ(Operations(), operations);
  const auto stream = llm_->paged().stream();
  ASSERT_TRUE(node_->FenceStream(stream));
  EXPECT_EQ(node_->kept_pinned(), 0U);
  EXPECT_TRUE(store->usable());
  EXPECT_EQ(Occupancy(), before);
}

TEST_F(SnapshotStoreTest, AnUnprovenRestoreInvalidatesTheCapabilityUntilItsStreamsFence) {
  Start(false, false);
  Warm();
  const auto before = Occupancy();
  rt::SnapshotStore store(*node_, kRoom);
  auto saved = BranchAt(0).CaptureState(store);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  static_cast<llmp::providers::fake::FakeDeviceExecution&>(node_->execution())
      .FailNextQuery(llmp::providers::ProviderError::kUnknown);
  EXPECT_FALSE(BranchAt(0).RestoreState(store, *saved));
  EXPECT_FALSE(store.usable());
  EXPECT_FALSE(store.Digest(*saved));
  EXPECT_EQ(node_->kept_pinned(), 1U);
  const auto operations = Operations();
  EXPECT_FALSE(BranchAt(0).RestoreState(store, *saved));
  EXPECT_EQ(Operations(), operations);  // no second restore submission
  ASSERT_TRUE(node_->FenceStream(llm_->paged().stream()));
  EXPECT_EQ(node_->kept_pinned(), 0U);
  EXPECT_TRUE(store.usable());
  EXPECT_FALSE(store.Digest(*saved));  // the capability stays invalid
  EXPECT_EQ(Occupancy(), before);
}

TEST_F(SnapshotStoreTest, MalformedRangeAndTotalSizesNeverReachCopyOrAllocation) {
  StartBoundaryNode();
  const auto before = Occupancy();
  const auto issuer = std::make_shared<const std::byte>();
  rt::SnapshotStore store(*node_, kRoom);
  const std::vector<std::vector<Range>> bad = {
      {{.region = 0, .offset = UINT64_MAX, .bytes = 2}},
      {{.region = 0, .offset = 0, .bytes = UINT64_MAX}},
      {{.region = 0, .offset = 0, .bytes = UINT64_MAX - 8},
       {.region = 1, .offset = 0, .bytes = 10}},
      {{.region = 0, .offset = 0, .bytes = 0}},
      {{.region = 1, .offset = 0, .bytes = 1}, {.region = 0, .offset = 0, .bytes = 1}},
      {{.region = 0, .offset = 0, .bytes = 9}, {.region = 0, .offset = 8, .bytes = 1}}};
  for (const auto& ranges : bad) {
    EXPECT_FALSE(Access::Prepare(store, issuer, {}, {}, ranges));
    EXPECT_EQ(store.capacity(), 0U);
    EXPECT_EQ(Occupancy(), before);
  }
}

TEST_F(SnapshotStoreTest, GrowthRefusalInvalidatesOldCaptureAndRetriesOnlyOneAccountedAllocation) {
  StartBoundaryNode();
  const auto before = Occupancy();
  const auto issuer = std::make_shared<const std::byte>();
  std::uint32_t reclaims = 0;
  rt::SnapshotStore store(*node_, 2 * kRoom, [&](std::uint64_t bytes) {
    ++reclaims;
    EXPECT_GT(bytes, kRoom);
    // The old allocation is released before any replacement is attempted.
    EXPECT_EQ(Occupancy(), before);
    return true;  // a retry still refuses: this callback released nothing
  });
  ASSERT_TRUE(Access::Prepare(store, issuer));
  auto saved = Access::Publish(store);
  EXPECT_FALSE(Access::Prepare(store, issuer, {}, std::string(kRoom, 'x')));
  EXPECT_EQ(reclaims, 1U);
  EXPECT_EQ(store.capacity(), 0U);
  EXPECT_EQ(Occupancy(), before);
  EXPECT_FALSE(store.Digest(saved));
}

TEST_F(SnapshotStoreTest, PhysicallyReleasedAllocationWithFailedBookkeepingNeverBecomesReusable) {
  // FreePinned takes a retirement ticket before the provider free. Once that
  // free happens, failed catalog retirement must fail-stop, never return a
  // released allocation to its store.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH(
      {
        Start(false, false);
        Warm();
        auto store = std::make_unique<rt::SnapshotStore>(*node_, kRoom);
        auto saved = BranchAt(0).CaptureState(*store);
        if (!saved) std::abort();  // not the fail-stop the matcher expects
        void* pointer = Access::BorrowedAddress(*store);
        if (pointer == nullptr) std::abort();
        const auto original_capacity = store->capacity();
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        fake::AfterNextPinnedFree(pointer, [&] {
          // Only the retired integer address identifies its span record.
          (void)node_->Call(
              [&]() -> en::Status {
                std::optional<cat::ExtentId> extent;
                node_->EraseSpans([&](const en::Span& span) {
                  if (span.base != address) return false;
                  extent = span.extent;
                  return true;
                });
                if (!extent) return std::unexpected("no span at the freed address");
                bool completed = false;
                for (std::uint64_t serial = 1; serial <= 64 && !completed; ++serial) {
                  completed = node_->catalog()
                                  .CompleteEvict({.extent = *extent,
                                                  .operation = cat::Operation::kEvict,
                                                  .serial = serial})
                                  .has_value();
                }
                if (!completed || !node_->catalog().RemoveExtent(*extent)) {
                  return std::unexpected("the post-free fixture could not retire its record");
                }
                return {};
              },
              "injecting a post-free snapshot bookkeeping error");
        });
        const auto issuer = std::make_shared<const std::byte>();
        // Grow metadata without inventing a mutable execution layout.
        (void)Access::Prepare(*store, issuer, {}, std::string(original_capacity, 'x'));
      },
      "freed pinned staging must leave the catalog");
}

TEST_F(SnapshotStoreTest,
       RawRequiredAndUnknownIdentityRefuseBeforeSettlementOrCopyWhileLegacyWorks) {
  Start(false, false);
  Warm();
  std::vector<std::byte> raw(BranchAt(0).state_snapshot_bytes());
  ASSERT_TRUE(BranchAt(0).SaveState(raw.data()));
  ASSERT_TRUE(BranchAt(0).RestoreState(raw.data()));
  const auto operations = Operations();
  const auto queued = source_->queued();
  const auto refuse = [&] {
    EXPECT_FALSE(BranchAt(0).SaveState(raw.data()));
    EXPECT_FALSE(BranchAt(0).RestoreState(raw.data()));
    EXPECT_EQ(Operations(), operations);
    EXPECT_EQ(source_->queued(), queued);
  };
  execution_identity_ = std::unexpected("unknown test execution identity");
  refuse();
  execution_identity_ = std::nullopt;
  EXPECT_TRUE(BranchAt(0).SaveState(raw.data()));
  EXPECT_TRUE(BranchAt(0).RestoreState(raw.data()));
  Stop();
  // A required identity is authenticated during Setup, before this model is
  // admitted. Changing a live source's recipe is independently rejected.
  execution_identity_ = Seal("required fake identity");
  Start(false, false);
  Warm();
  const auto required_operations = Operations();
  const auto required_queued = source_->queued();
  EXPECT_FALSE(BranchAt(0).SaveState(raw.data()));
  EXPECT_FALSE(BranchAt(0).RestoreState(raw.data()));
  EXPECT_EQ(Operations(), required_operations);
  EXPECT_EQ(source_->queued(), required_queued);
}

}  // namespace
