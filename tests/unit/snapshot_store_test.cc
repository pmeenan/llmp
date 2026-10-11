// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/snapshot_store.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "base/bytes.h"
#include "base/sha256.h"
#include "engine/paged_node.h"
#include "expected_error.h"
#include "providers/fake/fake_device_runtime.h"
#include "runtime/serving.h"

namespace {
namespace en = llmp::engine;
namespace rt = llmp::runtime;
namespace cat = llmp::catalog;
using llmp::base::Bytes;
using llmp::test_support::Failed;
using Range = en::LiveState::Range;
constexpr std::uint64_t kRoom = 2 * en::kPagedExtent;

llmp::base::Sha256Digest Seal(std::string_view value) {
  return llmp::base::Sha256().Update(value).Finish();
}

class FakePaged final : public en::PagedModel {
 public:
  std::uint32_t stream() const override { return 0; }
  const cat::Closure& fence_closure() const override { return closure_; }
  std::vector<cat::ExtentId> managed_extents() const override { return {}; }
  en::Status Release() override { return {}; }

 private:
  cat::Closure closure_;
};

// This test producer owns its synthetic bytes and identity. It exercises the
// snapshot protocol only; it is not a qualified production execution source.
class SnapshotLlm final : public rt::Llm {
 public:
  explicit SnapshotLlm(en::PagedNode& node) {
    node_ = &node;
    name_ = "snapshot fake";
    context_ = 64;
    max_rows_ = 8;
  }
  en::PagedModel& paged() override { return paged_; }
  rt::Status Setup() override { return {}; }
  std::uint64_t activations_needed() const override { return 0; }
  std::uint64_t pool_needed() const override { return 0; }
  rt::Status Register() override { return {}; }
  rt::Status Bind() override { return {}; }
  std::vector<cat::ExtentId> weights() const override { return {}; }
  std::vector<cat::ExtentId> state() const override { return {}; }
  const cat::Closure& everything() const override { return paged_.fence_closure(); }
  std::uint64_t weight_read_bytes() const override { return 0; }
  std::uint32_t logit_width() const override { return 2; }
  void Defaults(llmp::chat::Conversation&) const override {}
  std::string KeptLayout() const override { return layout; }
  std::expected<std::optional<llmp::base::Sha256Digest>, std::string> ExecutionIdentity()
      const override {
    if (unknown_identity) return std::unexpected("unknown test execution identity");
    return identity;
  }
  std::optional<llmp::base::Sha256Digest> identity;
  bool unknown_identity = false;
  std::string layout = "test layout";
  std::vector<Range> ranges = {{.region = 0, .offset = 3, .bytes = 13},
                               {.region = 1, .offset = 9, .bytes = 8}};
  std::array<std::byte, 21> state_bytes{};
  std::uint32_t settlements = 0, saves = 0, restores = 0;
  bool fail_save = false, fail_restore = false;
  // Only the fake copy producer can observe its borrowed destination. A test
  // can complete a previously failed copy later, as an asynchronous device can.
  void* pending_destination = nullptr;
  void* last_destination = nullptr;
  std::uint32_t position = 0;
  llmp::execution::AdaptiveDepth decoding{3};

 protected:
  rt::Status RunChunk(std::span<const std::int32_t> all, std::uint32_t, bool,
                      std::vector<float>& logits) override {
    position = static_cast<std::uint32_t>(all.size());
    logits = {1.0F, 0.0F};
    return {};
  }
  rt::Status SpecStep(std::span<const std::int32_t>, std::uint32_t, std::uint32_t,
                      std::vector<std::int32_t>&, std::vector<std::vector<float>>*,
                      std::uint64_t&) override {
    return std::unexpected("no fake speculation");
  }
  rt::Status Settle() override {
    ++settlements;
    return {};
  }
  rt::Status ClearState() override {
    position = 0;
    return {};
  }
  bool StateUsable() const override { return true; }
  rt::Status PrepareDecodeState(std::uint32_t, std::uint32_t) override { return {}; }
  std::uint64_t target_state_base() const override { return 0; }
  std::uint64_t target_state_bytes() const override { return state_bytes.size(); }
  std::uint64_t drafter_state_base() const override { return 0; }
  std::uint64_t drafter_state_bytes() const override { return 0; }
  std::uint64_t used_state_bytes() const override { return state_bytes.size(); }
  std::vector<Range> used_state_ranges() const override { return ranges; }
  rt::Status SaveUsedState(void* host, std::span<const Range> copied) override {
    ++saves;
    last_destination = host;
    if (fail_save) {
      pending_destination = host;
      return std::unexpected("synthetic submitted copy has unknown completion");
    }
    std::size_t count = 0;
    for (const auto& range : copied) count += range.bytes;
    EXPECT_EQ(count, ranges.empty() ? 0U : state_bytes.size());
    if (count != 0) std::memcpy(host, state_bytes.data(), count);
    return {};
  }
  rt::Status RestoreUsedState(void* host, std::span<const Range> copied) override {
    ++restores;
    if (fail_restore) return std::unexpected("synthetic restore has unknown completion");
    std::size_t count = 0;
    for (const auto& range : copied) count += range.bytes;
    EXPECT_EQ(count, state_bytes.size());
    if (count != 0) std::memcpy(state_bytes.data(), host, count);
    return {};
  }
  std::expected<std::vector<Range>, std::string> CheckpointRanges(std::uint32_t) const override {
    return ranges;
  }
  rt::Status PrepareRestoreState(std::span<const Range>) override { return {}; }
  rt::Status CopyCheckpointState(void*, std::span<const Range>, bool) override { return {}; }
  std::uint32_t cursor() const override { return position; }
  void set_cursor(std::uint32_t value) override { position = value; }
  llmp::execution::AdaptiveDepth TurnDecodingState() const override { return decoding; }
  void RestoreTurnDecodingState(const llmp::execution::AdaptiveDepth& value) override {
    decoding = value;
  }

 private:
  FakePaged paged_;
};

class SnapshotStoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_TRUE(node_.Open());
    fixed_ = node_.catalog().OccupancyOf(node_.domain()).Total().value();
    ASSERT_TRUE(node_.Start(Bytes(fixed_ + kRoom)));
    node_.Run();
  }
  void TearDown() override {
    llmp::providers::fake::ClearPinnedFreeHook();
    EXPECT_TRUE(node_.TearDown({}));
  }
  std::uint64_t Occupancy() {
    std::uint64_t result = 0;
    EXPECT_TRUE(node_.Call(
        [&]() -> en::Status {
          result = node_.catalog().OccupancyOf(node_.domain()).Total().value();
          return {};
        },
        "snapshot occupancy"));
    return result;
  }
  en::PagedNode node_{en::NodeSettings{}};
  std::uint64_t fixed_ = 0;
};

TEST_F(SnapshotStoreTest, SingleAllocationSeparatesPayloadFromMetadataAndReleasesTheExactCharge) {
  SnapshotLlm model(node_);
  model.identity = Seal("fake recipe A");
  for (std::size_t i = 0; i < model.state_bytes.size(); ++i) model.state_bytes[i] = std::byte(i);
  const auto expected = llmp::base::Sha256().Update(model.state_bytes).Finish();
  std::vector<float> logits;
  ASSERT_TRUE(model.Prefill(std::array<std::int32_t, 3>{4, 5, 6}, logits));
  for (unsigned i = 0; i < 4; ++i) model.decoding.Observe(3, 1);
  const auto original_decoding = model.decoding;
  const auto before = Occupancy();
  rt::StateSnapshot capability;
  {
    rt::SnapshotStore store(node_, kRoom);
    auto saved = model.default_branch().CaptureState(store);
    ASSERT_TRUE(saved) << Failed(saved).value_or("");
    capability = std::move(*saved);
    ASSERT_TRUE(store.Digest(capability));
    EXPECT_EQ(*store.Digest(capability), expected);
    EXPECT_GT(store.capacity(), model.state_bytes.size() + 3 * sizeof(std::int32_t));
    EXPECT_EQ(Occupancy() - before, store.capacity());
    model.state_bytes.fill(std::byte{0x77});
    model.decoding.Observe(2, 3);
    ASSERT_NE(model.decoding, original_decoding);
    ASSERT_TRUE(model.Prefill(std::array<std::int32_t, 2>{7, 8}, logits));
    ASSERT_TRUE(model.default_branch().RestoreState(store, capability));
    EXPECT_EQ(model.history(), (std::vector<std::int32_t>{4, 5, 6}));
    EXPECT_EQ(model.position, 3U);
    EXPECT_EQ(model.decoding, original_decoding);
    EXPECT_EQ(llmp::base::Sha256().Update(model.state_bytes).Finish(), expected);
    EXPECT_EQ(*store.Digest(capability), expected);  // excludes metadata and aligned padding
  }
  EXPECT_EQ(Occupancy(), before);
  rt::SnapshotStore foreign(node_, kRoom);
  EXPECT_FALSE(foreign.Digest(capability));
}

TEST_F(SnapshotStoreTest, EmptyPayloadChargesThePinnedMinimumAndBoundsItBeforeAllocation) {
  SnapshotLlm model(node_);
  model.ranges.clear();
  model.layout.clear();
  rt::SnapshotStore too_small(node_, 255);
  EXPECT_FALSE(model.default_branch().CaptureState(too_small));
  EXPECT_EQ(too_small.capacity(), 0U);
  EXPECT_EQ(model.saves, 0U);
  EXPECT_EQ(Occupancy(), fixed_);
  rt::SnapshotStore enough(node_, 256);
  auto saved = model.default_branch().CaptureState(enough);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  EXPECT_EQ(enough.capacity(), 256U);
  EXPECT_EQ(Occupancy() - fixed_, 256U);
  ASSERT_TRUE(enough.Digest(*saved));
  EXPECT_EQ(*enough.Digest(*saved), llmp::base::Sha256().Finish());
}

TEST_F(SnapshotStoreTest,
       ForeignExecutionLayoutUnknownIdentityAndStaleGenerationRefuseBeforeEffects) {
  SnapshotLlm model(node_);
  model.identity = Seal("fake recipe A");
  rt::SnapshotStore store(node_, kRoom);
  auto saved = model.default_branch().CaptureState(store);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  const auto settled = model.settlements;
  const auto capacity = store.capacity();
  const auto refused = [&] {
    EXPECT_FALSE(model.default_branch().RestoreState(store, *saved));
    EXPECT_EQ(model.settlements, settled);
    EXPECT_EQ(model.restores, 0U);
    EXPECT_EQ(store.capacity(), capacity);
  };
  model.identity = Seal("fake recipe B");
  refused();
  model.identity = Seal("fake recipe A");
  model.layout = "different same-sized layout";
  refused();
  model.layout = "test layout";
  model.unknown_identity = true;
  refused();
  model.unknown_identity = false;
  auto next = model.default_branch().CaptureState(store);
  ASSERT_TRUE(next) << Failed(next).value_or("");
  EXPECT_FALSE(store.Digest(*saved));
  EXPECT_TRUE(store.Digest(*next));
  EXPECT_FALSE(model.default_branch().RestoreState(store, *saved));
  EXPECT_EQ(model.restores, 0U);
}

TEST_F(SnapshotStoreTest, ExpiredIssuerRejectsDirectReadsWhileNodeLivesAndAllowsNewCapture) {
  auto model = std::make_unique<SnapshotLlm>(node_);
  rt::SnapshotStore store(node_, kRoom);
  auto saved = model->default_branch().CaptureState(store);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  const auto capacity = store.capacity();
  model.reset();
  EXPECT_TRUE(store.usable());
  EXPECT_FALSE(store.Digest(*saved));
  SnapshotLlm next(node_);
  EXPECT_FALSE(next.default_branch().RestoreState(store, *saved));
  auto current = next.default_branch().CaptureState(store);
  ASSERT_TRUE(current) << Failed(current).value_or("");
  EXPECT_EQ(store.capacity(), capacity);
  EXPECT_TRUE(store.Digest(*current));
  EXPECT_FALSE(store.Digest(*saved));
}

TEST_F(SnapshotStoreTest, UnknownCaptureRetainsTheWholeChargeAndBorrowAfterIssuerAndStoreRetire) {
  auto model = std::make_unique<SnapshotLlm>(node_);
  std::vector<float> logits;
  ASSERT_TRUE(model->Prefill(std::array<std::int32_t, 4>{1, 2, 3, 4}, logits));
  auto store = std::make_unique<rt::SnapshotStore>(node_, kRoom);
  auto previous = model->default_branch().CaptureState(*store);
  ASSERT_TRUE(previous) << Failed(previous).value_or("");
  const auto charge = store->capacity();
  model->fail_save = true;
  EXPECT_FALSE(model->default_branch().CaptureState(*store));
  EXPECT_FALSE(store->usable());
  EXPECT_FALSE(store->Digest(*previous));
  EXPECT_EQ(node_.kept_pinned(), 1U);
  EXPECT_EQ(Occupancy() - fixed_, charge);
  EXPECT_FALSE(model->default_branch().CaptureState(*store));
  auto* pending = model->pending_destination;
  ASSERT_NE(pending, nullptr);
  model.reset();
  store.reset();
  // A late producer still owns a valid borrow; both payload and private tail
  // remain charged after their metadata/issuer objects have been destroyed.
  std::memset(pending, 0x42, 21);
  EXPECT_EQ(node_.kept_pinned(), 1U);
  EXPECT_EQ(Occupancy() - fixed_, charge);
}

TEST_F(SnapshotStoreTest, UnknownRestoreInvalidatesCapabilityAndRetainsItsWholeAllocation) {
  SnapshotLlm model(node_);
  rt::SnapshotStore store(node_, kRoom);
  auto saved = model.default_branch().CaptureState(store);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  model.fail_restore = true;
  EXPECT_FALSE(model.default_branch().RestoreState(store, *saved));
  EXPECT_FALSE(store.usable());
  EXPECT_FALSE(store.Digest(*saved));
  EXPECT_EQ(node_.kept_pinned(), 1U);
  EXPECT_EQ(Occupancy() - fixed_, store.capacity());
  EXPECT_FALSE(model.default_branch().RestoreState(store, *saved));
  EXPECT_EQ(model.restores, 1U);
}

TEST_F(SnapshotStoreTest, MalformedRangeAndTotalSizesNeverReachCopyOrAllocation) {
  SnapshotLlm model(node_);
  rt::SnapshotStore store(node_, kRoom);
  const std::vector<std::vector<Range>> bad = {
      {{.region = 0, .offset = UINT64_MAX, .bytes = 2}},
      {{.region = 0, .offset = 0, .bytes = UINT64_MAX}},
      {{.region = 0, .offset = 0, .bytes = UINT64_MAX - 8},
       {.region = 1, .offset = 0, .bytes = 10}},
      {{.region = 0, .offset = 0, .bytes = 0}},
      {{.region = 1, .offset = 0, .bytes = 1}, {.region = 0, .offset = 0, .bytes = 1}},
      {{.region = 0, .offset = 0, .bytes = 9}, {.region = 0, .offset = 8, .bytes = 1}}};
  for (const auto& ranges : bad) {
    model.ranges = ranges;
    EXPECT_FALSE(model.default_branch().CaptureState(store));
    EXPECT_EQ(model.saves, 0U);
    EXPECT_EQ(store.capacity(), 0U);
    EXPECT_EQ(Occupancy(), fixed_);
  }
}

TEST_F(SnapshotStoreTest, GrowthRefusalInvalidatesOldCaptureAndRetriesOnlyOneAccountedAllocation) {
  SnapshotLlm model(node_);
  std::uint32_t reclaims = 0;
  rt::SnapshotStore store(node_, 2 * kRoom, [&](std::uint64_t bytes) {
    ++reclaims;
    EXPECT_GT(bytes, kRoom);
    // The old allocation is released before any replacement is attempted.
    EXPECT_EQ(Occupancy(), fixed_);
    return true;  // a retry still refuses, because this fake released nothing
  });
  auto saved = model.default_branch().CaptureState(store);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  model.layout.assign(kRoom, 'x');
  EXPECT_FALSE(model.default_branch().CaptureState(store));
  EXPECT_EQ(reclaims, 1U);
  EXPECT_EQ(model.saves, 1U);
  EXPECT_EQ(store.capacity(), 0U);
  EXPECT_EQ(Occupancy(), fixed_);
  EXPECT_FALSE(store.Digest(*saved));
}

TEST_F(SnapshotStoreTest, PhysicallyReleasedAllocationWithFailedBookkeepingNeverBecomesReusable) {
  SnapshotLlm model(node_);
  auto store = std::make_unique<rt::SnapshotStore>(node_, kRoom);
  auto saved = model.default_branch().CaptureState(*store);
  ASSERT_TRUE(saved) << Failed(saved).value_or("");
  const auto original_layout = model.layout;
  const auto original_capacity = store->capacity();
  ASSERT_NE(model.last_destination, nullptr);
  const auto address = reinterpret_cast<std::uintptr_t>(model.last_destination);
  unsigned physical_frees = 0;
  struct ClearHook {
    ~ClearHook() { llmp::providers::fake::ClearPinnedFreeHook(); }
  } clear_hook;
  llmp::providers::fake::AfterNextPinnedFree(model.last_destination, [&] {
    ++physical_frees;
    // This hook runs only after the actual provider free. Remove its catalog
    // record now, so PagedNode's subsequent bookkeeping Call fails. Never read
    // the retired address; the saved integer identifies only its span record.
    EXPECT_TRUE(node_.Call(
        [&]() -> en::Status {
          std::optional<cat::ExtentId> extent;
          node_.EraseSpans([&](const en::Span& span) {
            if (span.base != address) return false;
            extent = span.extent;
            return true;
          });
          if (!extent || !node_.catalog().ReleasePinned(*extent) ||
              !node_.catalog().RemoveExtent(*extent)) {
            return std::unexpected("the post-free fixture could not retire its catalog record");
          }
          return {};
        },
        "injecting a post-free snapshot bookkeeping error"));
  });
  model.layout.assign(original_capacity, 'x');  // force growth of the allocation
  auto growth = model.default_branch().CaptureState(*store);
  EXPECT_FALSE(growth);
  ASSERT_EQ(physical_frees, 1U);
  ASSERT_FALSE(growth) << "the injected bookkeeping failure must reach the caller";
  EXPECT_NE(growth.error().find("forgetting retired pinned staging"), std::string::npos);
  // This assertion discriminates the old implementation without executing
  // its unsafe recapture. A failed release must permanently close the store.
  ASSERT_FALSE(store->usable());
  EXPECT_FALSE(store->Digest(*saved));
  EXPECT_EQ(Occupancy(), fixed_);
  EXPECT_EQ(node_.kept_pinned(), 0U);  // it was freed, so there is nothing to retain
  const auto settled = model.settlements;
  for (const auto& layout : {original_layout, std::string{}}) {
    model.layout = layout;
    EXPECT_FALSE(model.default_branch().CaptureState(*store));
    EXPECT_FALSE(model.default_branch().RestoreState(*store, *saved));
    EXPECT_EQ(model.settlements, settled);
    EXPECT_EQ(model.saves, 1U);
    EXPECT_EQ(model.restores, 0U);
  }
  store.reset();
  EXPECT_EQ(physical_frees, 1U);
  EXPECT_EQ(Occupancy(), fixed_);
  EXPECT_EQ(node_.kept_pinned(), 0U);
}

TEST_F(SnapshotStoreTest,
       RawRequiredAndUnknownIdentityRefuseBeforeSettlementOrCopyWhileLegacyWorks) {
  SnapshotLlm model(node_);
  std::array<std::byte, 21> raw{};
  ASSERT_TRUE(model.SaveState(raw.data()));
  ASSERT_TRUE(model.RestoreState(raw.data()));
  const auto settled = model.settlements;
  const auto saves = model.saves;
  const auto restores = model.restores;
  const auto refuse = [&] {
    EXPECT_FALSE(model.SaveState(raw.data()));
    EXPECT_FALSE(model.RestoreState(raw.data()));
    EXPECT_EQ(model.settlements, settled);
    EXPECT_EQ(model.saves, saves);
    EXPECT_EQ(model.restores, restores);
  };
  model.identity = Seal("required fake identity");
  refuse();
  model.identity.reset();
  model.unknown_identity = true;
  refuse();
  model.unknown_identity = false;
  EXPECT_TRUE(model.SaveState(raw.data()));
  EXPECT_TRUE(model.RestoreState(raw.data()));
}

}  // namespace
