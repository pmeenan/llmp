// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The device-execution provider on the fakes: copies run in stream order
// only when stepped, fences name exactly the work before them, a stream
// can wait for another's fence, and a fence is released only once seen
// complete (docs/async-model.md).

#include "providers/device_execution.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stop_token>
#include <thread>

#include "base/bounded_queue.h"
#include "base/bytes.h"
#include "expected_error.h"
#include "providers/device_memory.h"
#include "providers/fake/fake_device_execution.h"
#include "providers/fake/fake_device_memory.h"

namespace {

using llmp::base::Bytes;
using llmp::test_support::FailedCode;
using llmp::base::operator""_MiB;
using llmp::providers::Access;
using llmp::providers::FenceState;
using llmp::providers::ProviderError;
using llmp::providers::fake::FakeDeviceExecution;
using llmp::providers::fake::FakeDeviceMemory;

class DeviceExecutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    reservation_ = memory_.Reserve(6_MiB).value();
    for (std::uint64_t i = 0; i < 3; ++i) {
      const auto backing = memory_.Create(1, 2_MiB).value();
      ASSERT_TRUE(
          memory_.Map(reservation_, llmp::base::Bytes(i * (2_MiB).value()), backing).has_value());
    }
    ASSERT_TRUE(memory_.SetAccess(reservation_, llmp::base::Bytes(0), 6_MiB, Access::kReadWrite)
                    .has_value());
    base_ = memory_.RangeOf(reservation_).value().base;
  }
  std::uint64_t Slot(int i) const {
    return base_ + (static_cast<std::uint64_t>(i) * (2_MiB).value());
  }
  std::byte* Data(int i) const {
    return reinterpret_cast<std::byte*>(Slot(i));  // NOLINT(performance-no-int-to-ptr)
  }

  FakeDeviceMemory memory_{2_MiB, 16_MiB};
  FakeDeviceExecution execution_;
  llmp::providers::ReservationId reservation_;
  std::uint64_t base_ = 0;
};

TEST_F(DeviceExecutionTest, AFenceNamesTheWorkBeforeIt) {
  std::memset(Data(0), 1, 16);
  const auto stream = execution_.CreateStream().value();
  ASSERT_TRUE(execution_.Copy(stream, Slot(1), Slot(0), llmp::base::Bytes(16)).has_value());
  const auto fence = execution_.Record(stream).value();
  EXPECT_EQ(execution_.Query(fence).value(), FenceState::kPending);
  EXPECT_NE(Data(1)[0], std::byte{1});  // not run yet: the consumer must wait
  EXPECT_EQ(FailedCode(execution_.Release(fence)),
            ProviderError::kInvalid);  // never seen complete
  EXPECT_EQ(FailedCode(execution_.DestroyStream(stream)), ProviderError::kInvalid);
  ASSERT_TRUE(execution_.Step(stream));  // the copy
  EXPECT_EQ(Data(1)[0], std::byte{1});
  EXPECT_EQ(execution_.Query(fence).value(), FenceState::kPending);
  ASSERT_TRUE(execution_.Step(stream));  // the fence
  EXPECT_EQ(execution_.Query(fence).value(), FenceState::kComplete);
  ASSERT_TRUE(execution_.Release(fence).has_value());
  EXPECT_EQ(FailedCode(execution_.Query(fence)), ProviderError::kInvalid);  // stale
  ASSERT_TRUE(execution_.DestroyStream(stream).has_value());
}

TEST_F(DeviceExecutionTest, AStreamWaitsForAnothersFence) {
  std::memset(Data(0), 2, 16);
  const auto producer = execution_.CreateStream().value();
  const auto consumer = execution_.CreateStream().value();
  ASSERT_TRUE(execution_.Copy(producer, Slot(1), Slot(0), llmp::base::Bytes(16)).has_value());
  const auto produced = execution_.Record(producer).value();
  ASSERT_TRUE(execution_.Wait(consumer, produced).has_value());
  ASSERT_TRUE(execution_.Copy(consumer, Slot(2), Slot(1), llmp::base::Bytes(16)).has_value());
  const auto consumed = execution_.Record(consumer).value();
  EXPECT_FALSE(execution_.Step(consumer));  // blocked on the producer
  execution_.Drain();
  EXPECT_EQ(Data(2)[0], std::byte{2});
  EXPECT_EQ(execution_.Query(consumed).value(), FenceState::kComplete);
  EXPECT_EQ(execution_.Query(produced).value(), FenceState::kComplete);
  ASSERT_TRUE(execution_.Release(produced).has_value());
  ASSERT_TRUE(execution_.Release(consumed).has_value());
}

// A device fault leaves the fenced work's outcome unknown: the fence stays
// unreleased (and its memory quarantined) until a later query settles it.
TEST_F(DeviceExecutionTest, AFaultIsNotACompletion) {
  const auto stream = execution_.CreateStream().value();
  const auto fence = execution_.Record(stream).value();
  execution_.Drain();
  execution_.FailNextQuery(fence, ProviderError::kUnknown);
  EXPECT_EQ(FailedCode(execution_.Query(fence)), ProviderError::kUnknown);
  EXPECT_EQ(FailedCode(execution_.Release(fence)), ProviderError::kInvalid);
  EXPECT_EQ(execution_.Query(fence).value(), FenceState::kComplete);
  ASSERT_TRUE(execution_.Release(fence).has_value());
}

// Destroying a stream does not wait for its work, so it is refused until
// everything queued lies behind a released fence.
TEST_F(DeviceExecutionTest, AStreamWithUnfencedWorkIsNotDestroyed) {
  const auto stream = execution_.CreateStream().value();
  ASSERT_TRUE(execution_.Copy(stream, Slot(1), Slot(0), llmp::base::Bytes(16)).has_value());
  execution_.Drain();  // it ran, but nothing proved it did
  EXPECT_EQ(FailedCode(execution_.DestroyStream(stream)), ProviderError::kInvalid);
  const auto fence = execution_.Record(stream).value();
  execution_.Drain();
  ASSERT_EQ(execution_.Query(fence).value(), FenceState::kComplete);
  EXPECT_EQ(FailedCode(execution_.DestroyStream(stream)), ProviderError::kInvalid);  // unreleased
  ASSERT_TRUE(execution_.Release(fence).has_value());
  EXPECT_TRUE(execution_.DestroyStream(stream).has_value());
}

// A kernel implementation takes the stream's native handle to launch on
// (D-053): distinct per stream, stable, and itself queued work that a
// fence must cover before the stream can go.
TEST_F(DeviceExecutionTest, ASubmissionHandleIsQueuedWork) {
  const auto first = execution_.CreateStream().value();
  const auto second = execution_.CreateStream().value();
  const auto handle = execution_.Submission(first).value();
  EXPECT_NE(handle.handle, nullptr);
  EXPECT_EQ(execution_.Submission(first).value().handle, handle.handle);
  EXPECT_NE(execution_.Submission(second).value().handle, handle.handle);
  EXPECT_EQ(FailedCode(execution_.DestroyStream(first)), ProviderError::kInvalid);
  for (const auto stream : {first, second}) {
    const auto fence = execution_.Record(stream).value();
    execution_.Drain();
    ASSERT_EQ(execution_.Query(fence).value(), FenceState::kComplete);
    ASSERT_TRUE(execution_.Release(fence).has_value());
    EXPECT_TRUE(execution_.DestroyStream(stream).has_value());
  }
  EXPECT_EQ(FailedCode(execution_.Submission(first)), ProviderError::kInvalid);
}

// The submission lane records and releases while the completion lane
// queries (D-048): no fence is lost, and each is released exactly once.
TEST_F(DeviceExecutionTest, SubmissionAndCompletionLanesShareTheProvider) {
  constexpr int kFences = 2000;
  const auto stream = execution_.CreateStream().value();
  llmp::base::BoundedQueue<llmp::providers::FenceId> recorded(kFences, 0);
  llmp::base::BoundedQueue<llmp::providers::FenceId> completed(kFences, 0);
  std::jthread completion([&] {
    while (std::optional<llmp::providers::FenceId> fence = recorded.Pop(std::stop_token{})) {
      while (execution_.Query(*fence).value_or(FenceState::kPending) != FenceState::kComplete) {
        std::this_thread::yield();
      }
      (void)completed.TryPush(llmp::providers::FenceId{*fence});
    }
    completed.Close();
  });
  for (int i = 0; i < kFences; ++i) {
    ASSERT_TRUE(execution_.Copy(stream, Slot(1), Slot(0), llmp::base::Bytes(64)).has_value());
    ASSERT_EQ(recorded.TryPush(execution_.Record(stream).value()),
              llmp::base::PushResult::kAccepted);
    execution_.Drain();
    while (std::optional<llmp::providers::FenceId> done = completed.TryPop()) {
      ASSERT_TRUE(execution_.Release(*done).has_value());
    }
  }
  recorded.Close();
  completion.join();
  while (std::optional<llmp::providers::FenceId> done = completed.TryPop()) {
    ASSERT_TRUE(execution_.Release(*done).has_value());
  }
  EXPECT_EQ(execution_.fences(), 0U);
  EXPECT_TRUE(execution_.DestroyStream(stream).has_value());
}

}  // namespace
