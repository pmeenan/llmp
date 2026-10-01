// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// CPU-only setup-failure control; no CUDA context or model is opened.
#include <gtest/gtest.h>

#include "base/bytes.h"
#include "engine/paged_node.h"
#include "expected_error.h"
#include "providers/fake/fake_device_memory.h"

namespace {
namespace engine = jitllm::engine;
namespace providers = jitllm::providers;
namespace fake = providers::fake;
using jitllm::base::operator""_MiB;
using jitllm::test_support::FailedCode;

TEST(Ds4CompleteRetirement, PartialCreateFailureReleasesCompletedBackingAndReservation) {
  fake::FakeDeviceMemory memory(2_MiB, 4_MiB);
  engine::Mapped mapped;
  mapped.name = "partial original reference setup";
  mapped.bytes = (4_MiB).value();
  const auto reservation = memory.Reserve(4_MiB);
  ASSERT_TRUE(reservation.has_value());
  mapped.reservation = *reservation;
  const auto range = memory.RangeOf(mapped.reservation);
  ASSERT_TRUE(range.has_value());
  mapped.base = range->base;
  const auto first = memory.Create(0, 2_MiB);
  ASSERT_TRUE(first.has_value());
  mapped.backings.push_back(*first);
  ASSERT_TRUE(memory.Map(mapped.reservation, 0_MiB, *first).has_value());
  ASSERT_TRUE(memory.SetAccess(mapped.reservation, 0_MiB, 2_MiB, providers::Access::kReadWrite)
                  .has_value());
  memory.FailNext(fake::Operation::kCreate, providers::ProviderError::kOutOfMemory);
  EXPECT_EQ(FailedCode(memory.Create(0, 2_MiB)), providers::ProviderError::kOutOfMemory);
  EXPECT_EQ(memory.in_use(), 2_MiB);
  EXPECT_EQ(memory.backings(), 1U);
  EXPECT_EQ(memory.reservations(), 1U);
  ASSERT_TRUE(engine::ReleaseMapped(memory, mapped));
  EXPECT_FALSE(mapped.reservation.valid());
  EXPECT_TRUE(mapped.backings.empty());
  EXPECT_EQ(memory.in_use(), 0_MiB);
  EXPECT_EQ(memory.backings(), 0U);
  EXPECT_EQ(memory.reservations(), 0U);
  EXPECT_EQ(FailedCode(memory.RangeOf(*reservation)), providers::ProviderError::kInvalid);
  EXPECT_TRUE(engine::ReleaseMapped(memory, mapped));
  EXPECT_EQ(memory.in_use(), 0_MiB);
  EXPECT_EQ(memory.backings(), 0U);
  EXPECT_EQ(memory.reservations(), 0U);
}

TEST(Ds4CompleteRetirement, FailedReservationLeavesAnEmptyIdempotentOwner) {
  fake::FakeDeviceMemory memory(2_MiB, 4_MiB);
  engine::Mapped mapped;
  mapped.name = "failed original reference reservation";
  mapped.bytes = (4_MiB).value();
  memory.FailNext(fake::Operation::kReserve, providers::ProviderError::kOutOfMemory);
  EXPECT_EQ(FailedCode(memory.Reserve(4_MiB)), providers::ProviderError::kOutOfMemory);
  EXPECT_FALSE(mapped.reservation.valid());
  EXPECT_TRUE(engine::ReleaseMapped(memory, mapped));
  EXPECT_TRUE(engine::ReleaseMapped(memory, mapped));
  EXPECT_EQ(memory.in_use(), 0_MiB);
  EXPECT_EQ(memory.backings(), 0U);
  EXPECT_EQ(memory.reservations(), 0U);
}
}  // namespace
