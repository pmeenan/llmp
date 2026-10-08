// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "providers/fake/fake_device_memory.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <optional>
#include <string>

#include "base/bytes.h"
#include "base/check.h"
#include "platform/files.h"
#include "providers/device_memory.h"

namespace llmp::providers::fake {
namespace {

std::unexpected<Failure> SystemFailure(const char* what) {
  const int error = errno;
  return std::unexpected(Failure{
      .error = error == ENOMEM ? ProviderError::kOutOfMemory : ProviderError::kFailed,
      .detail =
          std::format("{}: {}", what, std::strerror(error))});  // NOLINT(concurrency-mt-unsafe)
}

void* At(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

// Fills a backing file with the poison pattern through a temporary mapping.
bool Poison(int fd, Bytes size) {
  void* view = ::mmap(nullptr, size.value(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (view == MAP_FAILED) {
    return false;
  }
  std::memset(view, std::to_integer<int>(kPoison), size.value());
  return ::munmap(view, size.value()) == 0;
}

}  // namespace

FakeDeviceMemory::FakeDeviceMemory(Bytes granularity, Bytes capacity, Contents contents)
    : VmmProvider(granularity),
      classes_{
          AllocationClass{.kind = BackingKind::kDevice, .location = 0, .granularity = granularity},
          AllocationClass{.kind = BackingKind::kHost, .location = 0, .granularity = granularity}},
      capacity_(capacity),
      contents_(contents) {
  const long page = ::sysconf(_SC_PAGESIZE);
  base::Check(page > 0 && granularity.value() > 0 &&
                  granularity.value() % static_cast<std::uint64_t>(page) == 0,
              "the fake's granularity must be a multiple of the page size");
}

FakeDeviceMemory::~FakeDeviceMemory() {
  // Whatever the caller leaked goes with the fake; tests check reservations()
  // and backings() to catch the leak itself.
  for (const auto& [base, size] : reserved_) {
    (void)::munmap(At(base), size.value());
  }
  if (contents_ == Contents::kPoisoned) {
    for (const auto& [handle, size] : files_) {
      (void)::close(static_cast<int>(handle));
    }
  }
}

std::optional<FakeDeviceMemory::Scripted> FakeDeviceMemory::Take(Operation operation) {
  const auto found = failures_.find(operation);
  if (found == failures_.end()) {
    return std::nullopt;
  }
  const Scripted scripted = found->second;
  failures_.erase(found);
  return scripted;
}

std::expected<std::uint64_t, Failure> FakeDeviceMemory::DoReserve(Bytes size) {
  const std::optional<Scripted> script = Take(Operation::kReserve);
  if (script && !script->applied) {
    return std::unexpected(Failure{.error = script->error, .detail = "scripted by the test"});
  }
  // Over-reserve and trim, so the base is aligned to the granularity.
  const std::uint64_t alignment = Granularity().value();
  const auto padded = size.Plus(Granularity());
  if (!padded) {
    return std::unexpected(Failure{.error = ProviderError::kOutOfMemory,
                                   .detail = "the aligned reservation size overflows"});
  }
  const std::uint64_t span = padded->value();
  void* raw = ::mmap(nullptr, span, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (raw == MAP_FAILED) {
    return SystemFailure("mmap");
  }
  const auto start = reinterpret_cast<std::uint64_t>(raw);
  const std::uint64_t base = (start + alignment - 1) / alignment * alignment;
  if (base > start) {
    (void)::munmap(raw, base - start);
  }
  const std::uint64_t tail = start + span - (base + size.value());
  if (tail > 0) {
    (void)::munmap(At(base + size.value()), tail);
  }
  reserved_.emplace(base, size);
  if (script) {
    return std::unexpected(
        Failure{.error = script->error, .detail = "scripted by the test, and applied"});
  }
  return base;
}

std::expected<void, Failure> FakeDeviceMemory::DoFree(std::uint64_t base, Bytes size) {
  const std::optional<Scripted> script = Take(Operation::kFree);
  if (script && !script->applied) {
    return std::unexpected(Failure{.error = script->error, .detail = "scripted by the test"});
  }
  if (::munmap(At(base), size.value()) != 0) {
    return SystemFailure("munmap");
  }
  reserved_.erase(base);
  if (script) {
    return std::unexpected(
        Failure{.error = script->error, .detail = "scripted by the test, and applied"});
  }
  return {};
}

std::expected<FakeDeviceMemory::Handle, Failure> FakeDeviceMemory::DoCreate(
    const AllocationClass& /*allocation_class*/, Bytes size) {
  const std::optional<Scripted> script = Take(Operation::kCreate);
  if (script && !script->applied) {
    return std::unexpected(Failure{.error = script->error, .detail = "scripted by the test"});
  }
  const std::optional<Bytes> after = in_use_.Plus(size);
  if (!after || *after > capacity_) {
    return std::unexpected(
        Failure{.error = ProviderError::kOutOfMemory, .detail = "the fake's capacity is used up"});
  }
  if (contents_ == Contents::kNone) {
    const Handle handle = next_unbacked_++;
    in_use_ = *after;
    files_.emplace(handle, size);
    if (script) {
      return std::unexpected(
          Failure{.error = script->error, .detail = "scripted by the test, and applied"});
    }
    return handle;
  }
  const auto opened = platform::OpenAnonymousMemoryFile("llmp-fake-backing");
  if (!opened) {
    errno = opened.error();
    return SystemFailure("memfd_create");
  }
  const int fd = *opened;
  if (::ftruncate(fd, static_cast<off_t>(size.value())) != 0 || !Poison(fd, size)) {
    auto failure = SystemFailure("ftruncate");
    (void)::close(fd);
    return failure;
  }
  in_use_ = *after;
  files_.emplace(static_cast<Handle>(fd), size);
  if (script) {
    return std::unexpected(
        Failure{.error = script->error, .detail = "scripted by the test, and applied"});
  }
  return static_cast<Handle>(fd);
}

std::expected<void, Failure> FakeDeviceMemory::DoRelease(Handle handle, Bytes size) {
  const std::optional<Scripted> script = Take(Operation::kRelease);
  if (script && !script->applied) {
    return std::unexpected(Failure{.error = script->error, .detail = "scripted by the test"});
  }
  if (contents_ == Contents::kPoisoned) {
    const int fd = static_cast<int>(handle);
    (void)Poison(fd, size);  // what is freed reads as poison to any stale view
    (void)::close(fd);
  }
  files_.erase(handle);
  in_use_ = in_use_.Minus(size).value_or(Bytes());
  if (script) {
    return std::unexpected(
        Failure{.error = script->error, .detail = "scripted by the test, and applied"});
  }
  return {};
}

std::expected<void, Failure> FakeDeviceMemory::DoMap(std::uint64_t address, Bytes size,
                                                     Handle handle) {
  const std::optional<Scripted> script = Take(Operation::kMap);
  if (script && !script->applied) {
    return std::unexpected(Failure{.error = script->error, .detail = "scripted by the test"});
  }
  if (contents_ == Contents::kPoisoned) {
    void* mapped = ::mmap(At(address), size.value(), PROT_NONE, MAP_SHARED | MAP_FIXED,
                          static_cast<int>(handle), 0);
    if (mapped == MAP_FAILED) {
      return SystemFailure("mmap");
    }
  }
  if (script) {
    return std::unexpected(
        Failure{.error = script->error, .detail = "scripted by the test, and applied"});
  }
  return {};
}

std::expected<void, Failure> FakeDeviceMemory::DoSetAccess(std::uint64_t address, Bytes size,
                                                           Access access, bool /*host*/) {
  const std::optional<Scripted> script = Take(Operation::kSetAccess);
  if (script && !script->applied) {
    return std::unexpected(Failure{.error = script->error, .detail = "scripted by the test"});
  }
  int protection = PROT_NONE;
  if (access == Access::kRead) {
    protection = PROT_READ;
  } else if (access == Access::kReadWrite) {
    protection = PROT_READ | PROT_WRITE;
  }
  if (contents_ == Contents::kPoisoned && ::mprotect(At(address), size.value(), protection) != 0) {
    return SystemFailure("mprotect");
  }
  if (script) {
    return std::unexpected(
        Failure{.error = script->error, .detail = "scripted by the test, and applied"});
  }
  return {};
}

std::expected<void, Failure> FakeDeviceMemory::DoUnmap(std::uint64_t address, Bytes size) {
  const std::optional<Scripted> script = Take(Operation::kUnmap);
  if (script && !script->applied) {
    return std::unexpected(Failure{.error = script->error, .detail = "scripted by the test"});
  }
  // Back to a no-access hole in the reservation.
  if (contents_ == Contents::kPoisoned) {
    void* hole = ::mmap(At(address), size.value(), PROT_NONE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0);
    if (hole == MAP_FAILED) {
      return SystemFailure("mmap");
    }
  }
  if (script) {
    return std::unexpected(
        Failure{.error = script->error, .detail = "scripted by the test, and applied"});
  }
  return {};
}

}  // namespace llmp::providers::fake
