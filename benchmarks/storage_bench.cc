// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Storage queue depth and request (run) size through llmpalooza's own providers
// (docs/m2-record.md, "Providers"; D-034): direct reads by UringStorage into
// CUDA host-VMM backing from VmmProvider, the path the runtime pages
// through. Writes an unnamed direct-I/O file (O_TMPFILE, gone when the
// process exits) of the given size in the given directory, then reads it
// sequentially at each depth and request size, keeping that many requests
// in flight. Prints Markdown tables; the report is under
// docs/experiments/storage-queue/.
//
//   llmp_storage_bench DIRECTORY GIB

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <print>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_memory.h"
#include "providers/storage.h"
#include "providers/uring_storage.h"

namespace {

using Clock = std::chrono::steady_clock;
using llmp::base::Bytes;
using llmp::providers::Access;
using llmp::providers::BackingKind;
using llmp::providers::IoCompletion;
using llmp::providers::IoKind;
using llmp::providers::IoRequest;
using llmp::providers::Submission;
using llmp::providers::UringStorage;
using llmp::providers::VmmProvider;

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;

double Percentile(std::vector<double> samples, double p) {
  if (samples.empty()) {
    return 0;
  }
  std::ranges::sort(samples);
  return samples[static_cast<std::size_t>(p * static_cast<double>(samples.size() - 1))];
}

// Host backing for `bytes`, mapped with read-write access; its address.
std::byte* HostBuffer(VmmProvider& memory, std::uint64_t bytes) {
  std::size_t host = memory.Classes().size();
  for (std::size_t i = 0; i < memory.Classes().size(); ++i) {
    if (memory.Classes()[i].kind == BackingKind::kHost) {
      host = i;
    }
  }
  if (host == memory.Classes().size()) {
    return nullptr;
  }
  const auto reservation = memory.Reserve(Bytes(bytes));
  const auto backing = memory.Create(host, Bytes(bytes));
  if (!reservation || !backing || !memory.Map(*reservation, Bytes(0), *backing) ||
      !memory.SetAccess(*reservation, Bytes(0), Bytes(bytes), Access::kReadWrite)) {
    return nullptr;
  }
  return reinterpret_cast<std::byte*>(memory.RangeOf(*reservation).value_or({}).base);  // NOLINT
}

// Reads the whole file once, keeping `depth` requests of `request` bytes in
// flight, each into its own slot of `buffer`.
bool Measure(UringStorage& storage, std::byte* buffer, int fd, std::uint64_t file_bytes,
             std::size_t depth, std::uint64_t request) {
  std::vector<double> latencies_us;
  std::vector<Clock::time_point> started(depth);
  std::deque<std::size_t> free_slots;
  for (std::size_t slot = 0; slot < depth; ++slot) {
    free_slots.push_back(slot);
  }
  std::uint64_t next_offset = 0;
  std::uint64_t done = 0;
  bool failed = false;
  std::array<IoCompletion, 64> completions{};
  const Clock::time_point start = Clock::now();
  while (done < file_bytes && !failed) {
    while (!free_slots.empty() && next_offset < file_bytes) {
      const std::size_t slot = free_slots.front();
      const IoRequest io{
          .token = slot,
          .kind = IoKind::kRead,
          .fd = fd,
          .offset = next_offset,
          .memory =
              buffer + (slot * request),  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
          .length = static_cast<std::uint32_t>(request),
          .segments = {}};
      started[slot] = Clock::now();
      if (storage.Submit(io) == Submission::kNotStarted) {
        break;
      }
      free_slots.pop_front();
      next_offset += request;
    }
    const std::size_t count = storage.Harvest(completions, true);
    const Clock::time_point now = Clock::now();
    for (std::size_t c = 0; c < count; ++c) {
      const auto slot = static_cast<std::size_t>(completions[c].token);
      if (std::cmp_not_equal(completions[c].result, request)) {
        std::println(stderr, "a read returned {}", completions[c].result);
        failed = true;
        continue;
      }
      latencies_us.push_back(
          std::chrono::duration<double, std::micro>(now - started[slot]).count());
      done += request;
      free_slots.push_back(slot);
    }
  }
  while (storage.in_flight() > 0) {
    (void)storage.Harvest(completions, true);
  }
  if (failed) {
    return false;
  }
  const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
  std::println("| {} | {} | {} | {:.3f} | {:.0f} | {:.0f} |", depth, request / kMiB,
               (depth * request) / kMiB, static_cast<double>(done) / seconds / 1e9,
               Percentile(latencies_us, 0.5), Percentile(latencies_us, 0.99));
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(
      argv, argv + argc);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  if (args.size() != 3) {
    std::println(stderr, "usage: llmp_storage_bench DIRECTORY GIB");
    return 2;
  }
  std::uint64_t gib = 0;
  const char* end = args[2].data() + args[2].size();
  const auto parsed = std::from_chars(args[2].data(), end, gib);
  if (parsed.ec != std::errc{} || parsed.ptr != end || gib == 0 ||
      gib > (static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) >> 30U)) {
    std::println(stderr, "GIB must be a positive integer that fits the file-offset range");
    return 2;
  }
  const std::uint64_t file_bytes = gib << 30U;
  constexpr std::size_t kMaxDepth = 8;
  constexpr std::uint64_t kMaxRequest = 8 * kMiB;

  auto memory = llmp::providers::cuda::OpenDeviceMemory(0);
  if (!memory) {
    std::println(stderr, "no device memory: {}", memory.error().detail);
    return 1;
  }
  std::byte* buffer = HostBuffer(**memory, kMaxDepth * kMaxRequest);
  if (buffer == nullptr) {
    std::println(stderr, "cannot map host backing");
    return 1;
  }
  const int fd = ::open(args[1].c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
  if (fd < 0) {
    std::println(stderr, "cannot create a direct-I/O file in {}", args[1]);
    return 1;
  }
  // Fill the file with non-zero data through the same aligned memory.
  for (std::uint64_t i = 0; i < kMaxRequest; ++i) {
    buffer[i] =
        static_cast<std::byte>(i * 31);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  }
  for (std::uint64_t offset = 0; offset < file_bytes; offset += kMaxRequest) {
    if (::pwrite(fd, buffer, kMaxRequest, static_cast<off_t>(offset)) !=
        static_cast<ssize_t>(kMaxRequest)) {
      std::println(stderr, "cannot write the file");
      return 1;
    }
  }
  (void)::fsync(fd);

  auto storage = UringStorage::Create(kMaxDepth);
  if (!storage) {
    std::println(stderr, "no io_uring: {}", storage.error().message());
    return 1;
  }
  std::println("file: {} GiB, direct I/O into host VMM (granularity {} KiB)\n", file_bytes >> 30U,
               (*memory)->Granularity().value() >> 10U);
  std::println("| Depth | Request MiB | In flight MiB | GB/s | p50 us | p99 us |");
  std::println("| ---: | ---: | ---: | ---: | ---: | ---: |");
  for (const std::uint64_t request : {2 * kMiB, 4 * kMiB, 8 * kMiB}) {
    for (const std::size_t depth : {1U, 2U, 4U, 8U}) {
      if (!Measure(**storage, buffer, fd, file_bytes, depth, request)) {
        (void)::close(fd);
        return 1;
      }
    }
  }
  (void)::close(fd);
  return 0;
}
