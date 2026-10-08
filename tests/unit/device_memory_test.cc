// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The device-memory provider's shared rules (VmmProvider), on the fake
// (docs/architecture.md#providers): explicit reservations, backing mapped
// whole into holes, access granted explicitly, and absent backing that
// faults when touched.

#include "providers/device_memory.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <span>
#include <thread>
#include <tuple>

#include "base/bytes.h"
#include "expected_error.h"
#include "providers/fake/fake_device_memory.h"

namespace {

using llmp::base::Bytes;
using llmp::test_support::FailedCode;
using llmp::base::operator""_MiB;
using llmp::providers::Access;
using llmp::providers::BackingId;
using llmp::providers::BackingKind;
using llmp::providers::ProviderError;
using llmp::providers::ReservationId;
using llmp::providers::fake::Contents;
using llmp::providers::fake::FakeDeviceMemory;
using llmp::providers::fake::kPoison;
using llmp::providers::fake::Operation;

constexpr std::size_t kDevice = 0;
constexpr std::size_t kHost = 1;

class DeviceMemoryTest : public ::testing::Test {
 protected:
  std::byte* At(ReservationId reservation, Bytes offset) {
    const std::uint64_t address = memory_.RangeOf(reservation).value().base + offset.value();
    return reinterpret_cast<std::byte*>(address);  // NOLINT(performance-no-int-to-ptr)
  }

  FakeDeviceMemory memory_{2_MiB, 16_MiB};
};

TEST_F(DeviceMemoryTest, BackingIsMappedWholeAndAccessIsExplicit) {
  ASSERT_EQ(memory_.Classes().size(), 2U);
  EXPECT_EQ(memory_.Classes()[kHost].kind, BackingKind::kHost);
  const ReservationId reservation = memory_.Reserve(8_MiB).value();
  EXPECT_EQ(memory_.RangeOf(reservation).value().base % (2_MiB).value(), 0U);
  const BackingId backing = memory_.Create(kHost, 4_MiB).value();
  ASSERT_TRUE(memory_.Map(reservation, 2_MiB, backing).has_value());
  ASSERT_TRUE(memory_.SetAccess(reservation, 2_MiB, 4_MiB, Access::kReadWrite).has_value());
  std::byte* data = At(reservation, 2_MiB);
  EXPECT_EQ(data[0], kPoison);  // fresh backing holds nothing to rely on
  std::memset(data, 7, (4_MiB).value());
  // The same backing, unmapped and mapped elsewhere, keeps its contents.
  ASSERT_TRUE(memory_.Unmap(reservation, 2_MiB, 4_MiB).has_value());
  ASSERT_TRUE(memory_.Map(reservation, 4_MiB, backing).has_value());
  ASSERT_TRUE(memory_.SetAccess(reservation, 4_MiB, 4_MiB, Access::kRead).has_value());
  EXPECT_EQ(At(reservation, 4_MiB)[(4_MiB).value() - 1], std::byte{7});
  ASSERT_TRUE(memory_.Unmap(reservation, 4_MiB, 4_MiB).has_value());
  ASSERT_TRUE(memory_.Release(backing).has_value());
  ASSERT_TRUE(memory_.Free(reservation).has_value());
  EXPECT_EQ(memory_.reservations(), 0U);
  EXPECT_EQ(memory_.backings(), 0U);
  EXPECT_EQ(memory_.in_use(), Bytes());
}

TEST_F(DeviceMemoryTest, MisuseIsRefused) {
  EXPECT_EQ(FailedCode(memory_.Reserve(Bytes(4096))), ProviderError::kInvalid);  // not a granule
  const ReservationId reservation = memory_.Reserve(6_MiB).value();
  const BackingId a = memory_.Create(kDevice, 4_MiB).value();
  const BackingId b = memory_.Create(kDevice, 2_MiB).value();
  EXPECT_EQ(FailedCode(memory_.Create(kDevice, Bytes(4096))), ProviderError::kInvalid);
  EXPECT_EQ(FailedCode(memory_.Create(7, 2_MiB)), ProviderError::kInvalid);
  EXPECT_EQ(FailedCode(memory_.Map(reservation, 4_MiB, a)),
            ProviderError::kInvalid);  // runs outside
  EXPECT_EQ(FailedCode(memory_.Map(reservation, Bytes(4096), a)),
            ProviderError::kInvalid);  // unaligned
  ASSERT_TRUE(memory_.Map(reservation, 0_MiB, a).has_value());
  EXPECT_EQ(FailedCode(memory_.Map(reservation, 2_MiB, b)), ProviderError::kInvalid);  // overlaps
  EXPECT_EQ(FailedCode(memory_.Map(reservation, 4_MiB, a)),
            ProviderError::kInvalid);  // mapped twice
  ASSERT_TRUE(memory_.Map(reservation, 4_MiB, b).has_value());
  // Unmap and access take whole mappings only.
  EXPECT_EQ(FailedCode(memory_.Unmap(reservation, 0_MiB, 2_MiB)), ProviderError::kInvalid);
  EXPECT_EQ(FailedCode(memory_.SetAccess(reservation, 2_MiB, 4_MiB, Access::kRead)),
            ProviderError::kInvalid);
  EXPECT_TRUE(memory_.SetAccess(reservation, 0_MiB, 6_MiB, Access::kRead).has_value());
  // Nothing mapped is released or freed.
  EXPECT_EQ(FailedCode(memory_.Release(a)), ProviderError::kInvalid);
  EXPECT_EQ(FailedCode(memory_.Free(reservation)), ProviderError::kInvalid);
  ASSERT_TRUE(memory_.Unmap(reservation, 0_MiB, 6_MiB).has_value());  // both at once
  ASSERT_TRUE(memory_.Release(a).has_value());
  EXPECT_EQ(FailedCode(memory_.Release(a)), ProviderError::kInvalid);  // stale
  ASSERT_TRUE(memory_.Release(b).has_value());
  ASSERT_TRUE(memory_.Free(reservation).has_value());
  EXPECT_EQ(FailedCode(memory_.RangeOf(reservation)), ProviderError::kInvalid);
}

TEST_F(DeviceMemoryTest, CapacityAndScriptedFailuresChangeNothing) {
  const BackingId all = memory_.Create(kHost, 16_MiB).value();
  EXPECT_EQ(FailedCode(memory_.Create(kHost, 2_MiB)), ProviderError::kOutOfMemory);
  ASSERT_TRUE(memory_.Release(all).has_value());
  const ReservationId reservation = memory_.Reserve(2_MiB).value();
  const BackingId backing = memory_.Create(kHost, 2_MiB).value();
  memory_.FailNext(Operation::kMap, ProviderError::kFailed);
  EXPECT_EQ(FailedCode(memory_.Map(reservation, 0_MiB, backing)), ProviderError::kFailed);
  ASSERT_TRUE(
      memory_.Map(reservation, 0_MiB, backing).has_value());  // nothing changed: it maps now
  ASSERT_TRUE(memory_.Unmap(reservation, 0_MiB, 2_MiB).has_value());
  ASSERT_TRUE(memory_.Release(backing).has_value());
  ASSERT_TRUE(memory_.Free(reservation).has_value());
}

TEST(DeviceMemoryRangeTest, AlignmentPaddingCannotWrapAReservation) {
  constexpr std::uint64_t kGranularity = std::uint64_t{3} * 4096;
  FakeDeviceMemory memory(Bytes(kGranularity), Bytes(0));
  const Bytes largest((std::numeric_limits<std::uint64_t>::max() / kGranularity) * kGranularity);
  EXPECT_EQ(FailedCode(memory.Reserve(largest)), ProviderError::kOutOfMemory);
  EXPECT_EQ(memory.reservations(), 0U);
}

// An unknown outcome is a fault, not a retry: what it touched is
// undetermined, and every later call on it is refused, even when the
// driver did what it was asked.
TEST_F(DeviceMemoryTest, UnknownOutcomesAreNeverRetried) {
  const ReservationId reservation = memory_.Reserve(4_MiB).value();
  const BackingId first = memory_.Create(kHost, 2_MiB).value();
  const BackingId second = memory_.Create(kHost, 2_MiB).value();
  memory_.FailNext(Operation::kMap, ProviderError::kUnknown, /*applied=*/true);
  EXPECT_EQ(FailedCode(memory_.Map(reservation, 0_MiB, first)), ProviderError::kUnknown);
  EXPECT_TRUE(memory_.Undetermined(reservation));
  EXPECT_TRUE(memory_.Undetermined(first));
  EXPECT_EQ(FailedCode(memory_.RangeOf(reservation)), ProviderError::kUndetermined);
  EXPECT_EQ(FailedCode(memory_.Map(reservation, 2_MiB, second)), ProviderError::kUndetermined);
  EXPECT_EQ(FailedCode(memory_.Release(first)), ProviderError::kUndetermined);
  EXPECT_EQ(FailedCode(memory_.Free(reservation)), ProviderError::kUndetermined);
  EXPECT_FALSE(memory_.Undetermined(second));
  // A release whose outcome is unknown is never repeated.
  memory_.FailNext(Operation::kRelease, ProviderError::kUnknown, /*applied=*/true);
  EXPECT_EQ(FailedCode(memory_.Release(second)), ProviderError::kUnknown);
  EXPECT_EQ(FailedCode(memory_.Release(second)), ProviderError::kUndetermined);
  // A create whose outcome is unknown may have made backing: it stays
  // charged, with no handle anyone can use.
  memory_.FailNext(Operation::kCreate, ProviderError::kUnknown, /*applied=*/true);
  EXPECT_EQ(FailedCode(memory_.Create(kHost, 2_MiB)), ProviderError::kUnknown);
  // What is left is the owner's to quarantine; the fake cleans up at the end.
  EXPECT_EQ(memory_.backings(), 3U);
  EXPECT_EQ(memory_.UndeterminedBytes(), 6_MiB);
}

TEST_F(DeviceMemoryTest, UnknownAccessKeepsItsBackingChargedAsUndetermined) {
  const ReservationId reservation = memory_.Reserve(4_MiB).value();
  const BackingId first = memory_.Create(kHost, 2_MiB).value();
  const BackingId second = memory_.Create(kHost, 2_MiB).value();
  ASSERT_TRUE(memory_.Map(reservation, 0_MiB, first).has_value());
  ASSERT_TRUE(memory_.Map(reservation, 2_MiB, second).has_value());
  memory_.FailNext(Operation::kSetAccess, ProviderError::kUnknown, /*applied=*/true);
  EXPECT_EQ(FailedCode(memory_.SetAccess(reservation, 0_MiB, 4_MiB, Access::kReadWrite)),
            ProviderError::kUnknown);
  EXPECT_TRUE(memory_.Undetermined(first));
  EXPECT_TRUE(memory_.Undetermined(second));
  EXPECT_EQ(memory_.UndeterminedBytes(), 4_MiB);
  EXPECT_EQ(FailedCode(memory_.RangeOf(reservation)), ProviderError::kUndetermined);
  EXPECT_EQ(FailedCode(memory_.Unmap(reservation, 0_MiB, 4_MiB)), ProviderError::kUndetermined);
}

// Touching absent backing is a bug, and the fake makes it fault: reserved
// address space, a mapping with no access yet, and an unmapped range.
using DeviceMemoryDeathTest = DeviceMemoryTest;

TEST_F(DeviceMemoryDeathTest, AbsentBackingFaults) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const ReservationId reservation = memory_.Reserve(4_MiB).value();
  volatile std::byte* hole = At(reservation, 0_MiB);
  EXPECT_DEATH((void)hole[0], "");
  const BackingId backing = memory_.Create(kDevice, 2_MiB).value();
  ASSERT_TRUE(memory_.Map(reservation, 2_MiB, backing).has_value());
  volatile std::byte* no_access = At(reservation, 2_MiB);
  EXPECT_DEATH((void)no_access[0], "");
  ASSERT_TRUE(memory_.SetAccess(reservation, 2_MiB, 2_MiB, Access::kRead).has_value());
  const std::byte seen = no_access[0];
  EXPECT_EQ(seen, kPoison);
  EXPECT_DEATH(no_access[0] = std::byte{1}, "");  // read-only
  ASSERT_TRUE(memory_.Unmap(reservation, 2_MiB, 2_MiB).has_value());
  EXPECT_DEATH((void)no_access[0], "");
  ASSERT_TRUE(memory_.Release(backing).has_value());
  ASSERT_TRUE(memory_.Free(reservation).has_value());
}

// The address-only fake holds no bytes, so a replay can count calls at real
// scale (here 256 GiB of backing) under the same rules and capacity, and
// every byte of it still faults when touched.
TEST_F(DeviceMemoryDeathTest, AddressOnlyBackingKeepsTheRulesAndFaults) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  constexpr std::uint64_t kHuge = std::uint64_t{256} << 30U;  // 256 GiB
  FakeDeviceMemory memory(2_MiB, Bytes(kHuge), Contents::kNone);
  const ReservationId reservation = memory.Reserve(Bytes(kHuge)).value();
  const BackingId big = memory.Create(kHost, Bytes(kHuge - (2_MiB).value())).value();
  EXPECT_EQ(FailedCode(memory.Create(kHost, 4_MiB)), ProviderError::kOutOfMemory);
  const BackingId small = memory.Create(kHost, 2_MiB).value();
  EXPECT_EQ(memory.in_use(), Bytes(kHuge));
  ASSERT_TRUE(memory.Map(reservation, 0_MiB, small).has_value());
  EXPECT_EQ(FailedCode(memory.Map(reservation, 0_MiB, big)), ProviderError::kInvalid);
  ASSERT_TRUE(memory.Map(reservation, 2_MiB, big).has_value());
  ASSERT_TRUE(memory.SetAccess(reservation, 0_MiB, 2_MiB, Access::kReadWrite).has_value());
  const std::uint64_t base = memory.RangeOf(reservation).value().base;
  volatile auto* granted =
      reinterpret_cast<volatile std::byte*>(base);  // NOLINT(performance-no-int-to-ptr)
  EXPECT_DEATH((void)granted[0], "");               // no bytes behind it
  EXPECT_EQ(FailedCode(memory.Release(small)), ProviderError::kInvalid);  // still mapped
  ASSERT_TRUE(memory.Unmap(reservation, 0_MiB, Bytes(kHuge)).has_value());
  ASSERT_TRUE(memory.Release(small).has_value());
  ASSERT_TRUE(memory.Release(big).has_value());
  ASSERT_TRUE(memory.Free(reservation).has_value());
  EXPECT_EQ(memory.in_use(), Bytes());
  EXPECT_EQ(memory.backings(), 0U);
  EXPECT_EQ(memory.reservations(), 0U);
}

// A provider whose Create waits inside the driver until let go, so that a
// second caller can arrive while it is under way.
class BlockingProvider final : public llmp::providers::VmmProvider {
 public:
  BlockingProvider() : VmmProvider(2_MiB) {}
  std::span<const llmp::providers::AllocationClass> Classes() const override { return classes_; }

  bool Inside() const { return inside_.load(); }  // Create is under way
  void Go() { go_.store(true); }                  // lets it return

 protected:
  using Failure = llmp::providers::Failure;
  std::expected<std::uint64_t, Failure> DoReserve(Bytes size) override {
    const std::uint64_t base = next_;
    next_ += size.value();
    return base;
  }
  std::expected<void, Failure> DoFree(std::uint64_t /*base*/, Bytes /*size*/) override {
    return {};
  }
  std::expected<Handle, Failure> DoCreate(
      const llmp::providers::AllocationClass& /*allocation_class*/, Bytes /*size*/) override {
    inside_.store(true);
    while (!go_.load()) {
      std::this_thread::yield();
    }
    return Handle{1};
  }
  std::expected<void, Failure> DoRelease(Handle /*handle*/, Bytes /*size*/) override { return {}; }
  std::expected<void, Failure> DoMap(std::uint64_t /*address*/, Bytes /*size*/,
                                     Handle /*handle*/) override {
    return {};
  }
  std::expected<void, Failure> DoSetAccess(std::uint64_t /*address*/, Bytes /*size*/,
                                           Access /*access*/, bool /*host*/) override {
    return {};
  }
  std::expected<void, Failure> DoUnmap(std::uint64_t /*address*/, Bytes /*size*/) override {
    return {};
  }

 private:
  std::array<llmp::providers::AllocationClass, 1> classes_{llmp::providers::AllocationClass{
      .kind = BackingKind::kDevice, .location = 0, .granularity = 2_MiB}};
  std::uint64_t next_ = std::uint64_t{1} << 40U;
  std::atomic<bool> inside_{false};
  std::atomic<bool> go_{false};
};

// One thread calls a provider at a time (device_memory.h): a change made
// while another is under way (the VMM lane's Create, say) is fatal, not a
// silent race on the provider's tables. One after the other from two
// threads is fine.
TEST(DeviceMemoryCallerDeathTest, ChangesFromTwoThreadsAtOnceAreFatal) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto overlap = [] {
    BlockingProvider provider;
    std::jthread lane([&] { std::ignore = provider.Create(0, 2_MiB); });
    while (!provider.Inside()) {
      std::this_thread::yield();
    }
    std::ignore = provider.Reserve(2_MiB);  // while the lane's Create is under way
    provider.Go();
  };
  EXPECT_DEATH(overlap(), "one thread calls it at a time");
  BlockingProvider provider;
  provider.Go();
  std::jthread([&] { EXPECT_TRUE(provider.Create(0, 2_MiB).has_value()); }).join();
  EXPECT_TRUE(provider.Reserve(2_MiB).has_value());
}

}  // namespace
