// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The CUDA device-memory provider on a real device (label `gpu`, and
// `gpu-discrete` on a discrete GPU the build targets, D-082): the shared
// rules hold under the driver, device and host classes exist on every GPU
// `llmp doctor` accepts, host backing given access is the CPU's at the
// same address, and direct file reads land in it with no copy (D-034).

#include "providers/cuda/cuda_device_memory.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

#include "base/bytes.h"
#include "expected_error.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_memory.h"
#include "providers/direct_reader.h"
#include "providers/uring_storage.h"

namespace {

using llmp::base::Bytes;
using llmp::providers::Access;
using llmp::providers::BackingKind;
using llmp::providers::ProviderError;
using llmp::providers::VmmProvider;
using llmp::test_support::FailedCode;

std::unique_ptr<VmmProvider> Open() {
  auto memory = llmp::providers::cuda::OpenDeviceMemory(0);
  EXPECT_TRUE(memory.has_value()) << (memory ? "" : memory.error().detail);
  return memory ? std::move(*memory) : nullptr;
}

std::size_t ClassOf(const VmmProvider& memory, BackingKind kind) {
  for (std::size_t i = 0; i < memory.Classes().size(); ++i) {
    if (memory.Classes()[i].kind == kind) {
      return i;
    }
  }
  ADD_FAILURE() << "no allocation class of that kind";
  return 0;
}

TEST(CudaDeviceMemory, HostBackingIsTheCpusAtTheSameAddress) {
  const std::unique_ptr<VmmProvider> memory = Open();
  ASSERT_NE(memory, nullptr);
  ASSERT_EQ(memory->Classes().size(), 2U);  // host-NUMA VMM, which doctor requires
  const Bytes granule = memory->Granularity();
  ASSERT_GT(granule.value(), 0U);
  const auto reservation = memory->Reserve(Bytes(granule.value() * 4)).value();
  const auto host =
      memory->Create(ClassOf(*memory, BackingKind::kHost), Bytes(granule.value() * 2)).value();
  ASSERT_TRUE(memory->Map(reservation, granule, host).has_value());
  ASSERT_TRUE(
      memory->SetAccess(reservation, granule, Bytes(granule.value() * 2), Access::kReadWrite)
          .has_value());
  auto* data =
      reinterpret_cast<std::uint8_t*>(memory->RangeOf(reservation).value().base +  // NOLINT
                                      granule.value());
  std::memset(data, 0x5a, granule.value() * 2);
  EXPECT_EQ(data[(granule.value() * 2) - 1], 0x5a);
  // Unmapped and mapped again elsewhere, the backing keeps its contents.
  ASSERT_TRUE(memory->Unmap(reservation, granule, Bytes(granule.value() * 2)).has_value());
  ASSERT_TRUE(memory->Map(reservation, Bytes(granule.value() * 2), host).has_value());
  ASSERT_TRUE(memory
                  ->SetAccess(reservation, Bytes(granule.value() * 2), Bytes(granule.value() * 2),
                              Access::kRead)
                  .has_value());
  const auto* moved =
      reinterpret_cast<const std::uint8_t*>(memory->RangeOf(reservation).value().base +  // NOLINT
                                            (granule.value() * 2));
  EXPECT_EQ(moved[0], 0x5a);
  ASSERT_TRUE(memory->Unmap(reservation, Bytes(granule.value() * 2), Bytes(granule.value() * 2))
                  .has_value());
  ASSERT_TRUE(memory->Release(host).has_value());
  ASSERT_TRUE(memory->Free(reservation).has_value());
  EXPECT_EQ(memory->reservations(), 0U);
  EXPECT_EQ(memory->backings(), 0U);
}

TEST(CudaDeviceMemory, DeviceBackingMapsAndTheRulesHold) {
  const std::unique_ptr<VmmProvider> memory = Open();
  ASSERT_NE(memory, nullptr);
  const Bytes granule = memory->Granularity();
  const auto reservation = memory->Reserve(Bytes(granule.value() * 2)).value();
  const auto device = memory->Create(ClassOf(*memory, BackingKind::kDevice), granule).value();
  const auto other = memory->Create(ClassOf(*memory, BackingKind::kDevice), granule).value();
  ASSERT_TRUE(memory->Map(reservation, Bytes(0), device).has_value());
  EXPECT_EQ(FailedCode(memory->Map(reservation, Bytes(0), other)), ProviderError::kInvalid);
  ASSERT_TRUE(memory->Map(reservation, granule, other).has_value());
  ASSERT_TRUE(
      memory->SetAccess(reservation, Bytes(0), Bytes(granule.value() * 2), Access::kReadWrite)
          .has_value());
  EXPECT_EQ(FailedCode(memory->Release(device)), ProviderError::kInvalid);  // still mapped
  EXPECT_EQ(FailedCode(memory->Free(reservation)), ProviderError::kInvalid);
  ASSERT_TRUE(memory->Unmap(reservation, Bytes(0), Bytes(granule.value() * 2)).has_value());
  ASSERT_TRUE(memory->Release(device).has_value());
  ASSERT_TRUE(memory->Release(other).has_value());
  ASSERT_TRUE(memory->Free(reservation).has_value());
}

// D-034's path: io_uring reads a direct-I/O file straight into host VMM
// backing, the landing zone the device copies from (D-081).
TEST(CudaDeviceMemory, DirectReadsLandInHostBacking) {
  const std::unique_ptr<VmmProvider> memory = Open();
  ASSERT_NE(memory, nullptr);
  const Bytes granule = memory->Granularity();
  const std::uint64_t length = granule.value() * 2;
  const char* base = std::getenv("LLMP_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  const std::filesystem::path directory =
      base != nullptr ? std::filesystem::path(base) : std::filesystem::path(::testing::TempDir());
  std::filesystem::create_directories(directory);
  const int fd = ::open(directory.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
  ASSERT_GE(fd, 0);
  // The file's contents, written through aligned memory.
  auto* staging = static_cast<std::uint8_t*>(
      std::aligned_alloc(4096, length));  // NOLINT(cppcoreguidelines-no-malloc)
  for (std::uint64_t i = 0; i < length; ++i) {
    staging[i] = static_cast<std::uint8_t>((i * 13) + (i >> 20));
  }
  ASSERT_EQ(::pwrite(fd, staging, length, 0), static_cast<ssize_t>(length));

  const auto reservation = memory->Reserve(Bytes(length)).value();
  const auto host = memory->Create(ClassOf(*memory, BackingKind::kHost), Bytes(length)).value();
  ASSERT_TRUE(memory->Map(reservation, Bytes(0), host).has_value());
  ASSERT_TRUE(
      memory->SetAccess(reservation, Bytes(0), Bytes(length), Access::kReadWrite).has_value());
  auto* destination =
      reinterpret_cast<std::byte*>(memory->RangeOf(reservation).value().base);  // NOLINT

  auto storage = llmp::providers::UringStorage::Create(4);
  ASSERT_TRUE(storage.has_value()) << storage.error().message();
  llmp::providers::DirectReader reader(**storage, llmp::providers::ReaderSettings{});
  ASSERT_TRUE(reader.Read(1, {.fd = fd, .offset = 0, .memory = destination, .length = length}, 1)
                  .has_value());
  std::vector<llmp::providers::FinishedRead> finished;
  for (int i = 0; i < 1000 && finished.empty(); ++i) {
    finished = reader.Poll(true);
  }
  ASSERT_EQ(finished.size(), 1U);
  EXPECT_EQ(finished[0].outcome, llmp::providers::ReadOutcome::kComplete);
  EXPECT_EQ(finished[0].bytes, length);
  EXPECT_EQ(std::memcmp(destination, staging, length), 0);

  ASSERT_TRUE(memory->Unmap(reservation, Bytes(0), Bytes(length)).has_value());
  ASSERT_TRUE(memory->Release(host).has_value());
  ASSERT_TRUE(memory->Free(reservation).has_value());
  std::free(staging);  // NOLINT(cppcoreguidelines-no-malloc)
  (void)::close(fd);
}

// Host to device and back on llmpalooza's streams, with fences observed by
// query alone.
TEST(CudaDeviceExecution, CopiesFollowTheirFences) {
  const std::unique_ptr<VmmProvider> memory = Open();
  ASSERT_NE(memory, nullptr);
  auto opened = llmp::providers::cuda::OpenDeviceExecution(0);
  ASSERT_TRUE(opened.has_value()) << (opened ? "" : opened.error().detail);
  llmp::providers::DeviceExecution& execution = **opened;
  const Bytes granule = memory->Granularity();
  const auto reservation = memory->Reserve(Bytes(granule.value() * 3)).value();
  const auto source = memory->Create(ClassOf(*memory, BackingKind::kHost), granule).value();
  const auto device = memory->Create(ClassOf(*memory, BackingKind::kDevice), granule).value();
  const auto back = memory->Create(ClassOf(*memory, BackingKind::kHost), granule).value();
  ASSERT_TRUE(memory->Map(reservation, Bytes(0), source).has_value());
  ASSERT_TRUE(memory->Map(reservation, granule, device).has_value());
  ASSERT_TRUE(memory->Map(reservation, Bytes(granule.value() * 2), back).has_value());
  ASSERT_TRUE(
      memory->SetAccess(reservation, Bytes(0), Bytes(granule.value() * 3), Access::kReadWrite)
          .has_value());
  const std::uint64_t base = memory->RangeOf(reservation).value().base;
  auto* host_in = reinterpret_cast<std::uint8_t*>(base);                           // NOLINT
  auto* host_out = reinterpret_cast<std::uint8_t*>(base + (granule.value() * 2));  // NOLINT
  std::memset(host_in, 0x3c, granule.value());
  std::memset(host_out, 0, granule.value());

  const auto upload = execution.CreateStream().value();
  const auto download = execution.CreateStream().value();
  ASSERT_TRUE(execution.Copy(upload, base + granule.value(), base, granule).has_value());
  const auto uploaded = execution.Record(upload).value();
  ASSERT_TRUE(execution.Wait(download, uploaded).has_value());
  ASSERT_TRUE(
      execution.Copy(download, base + (granule.value() * 2), base + granule.value(), granule)
          .has_value());
  const auto downloaded = execution.Record(download).value();
  // The completion lane queries while the submission lane goes on.
  std::expected<llmp::providers::FenceState, llmp::providers::Failure> state =
      llmp::providers::FenceState::kPending;
  std::jthread completion([&] {
    for (int i = 0; i < 1000000 && state && *state == llmp::providers::FenceState::kPending; ++i) {
      state = execution.Query(downloaded);
    }
  });
  EXPECT_EQ(FailedCode(execution.DestroyStream(download)),
            ProviderError::kInvalid);  // unreleased fence
  completion.join();
  ASSERT_TRUE(state.has_value()) << state.error().detail;
  ASSERT_EQ(*state, llmp::providers::FenceState::kComplete);
  EXPECT_EQ(host_out[granule.value() - 1], 0x3c);
  EXPECT_EQ(execution.Query(uploaded).value(), llmp::providers::FenceState::kComplete);
  ASSERT_TRUE(execution.Release(uploaded).has_value());
  ASSERT_TRUE(execution.Release(downloaded).has_value());
  ASSERT_TRUE(execution.DestroyStream(upload).has_value());
  ASSERT_TRUE(execution.DestroyStream(download).has_value());
  ASSERT_TRUE(memory->Unmap(reservation, Bytes(0), Bytes(granule.value() * 3)).has_value());
  ASSERT_TRUE(memory->Release(source).has_value());
  ASSERT_TRUE(memory->Release(device).has_value());
  ASSERT_TRUE(memory->Release(back).has_value());
  ASSERT_TRUE(memory->Free(reservation).has_value());
}

}  // namespace
