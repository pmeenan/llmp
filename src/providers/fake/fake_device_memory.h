// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The deterministic device-memory fake (docs/architecture.md#providers):
// VmmProvider over real host memory, so misuse is caught, not assumed.
//   - A reservation is address space with no access at all: touching it
//     faults unless backing is mapped there with access granted.
//   - Backing is an anonymous memory file, filled with a poison pattern
//     when it is created and again before it is released, so code that
//     reads what nothing wrote sees the poison.
//   - Mapping maps that file at the address with no access; SetAccess
//     grants it; Unmap puts the reservation's no-access hole back.
// Every allocation class is served from host memory: the fake proves
// llmpalooza's logic, not GPU placement or performance. Tests can make the
// next call of an operation fail with a chosen error.
//
// An address-only fake (Contents::kNone) keeps the shared rules, the
// capacity and the no-access reservations, but its backing holds no bytes:
// Map, SetAccess and Unmap leave the reservation a no-access hole, so
// touching any of it faults. It is for replays that count provider calls
// at real scale and never touch contents (the retained-backing replay).

#ifndef LLMP_PROVIDERS_FAKE_FAKE_DEVICE_MEMORY_H_
#define LLMP_PROVIDERS_FAKE_FAKE_DEVICE_MEMORY_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <span>
#include <vector>

#include "base/bytes.h"
#include "providers/device_memory.h"

namespace llmp::providers::fake {

// The byte every fresh or released backing is filled with.
inline constexpr std::byte kPoison{0xa5};

enum class Operation : std::uint8_t {
  kReserve,
  kFree,
  kCreate,
  kRelease,
  kMap,
  kSetAccess,
  kUnmap
};

// Whether backing holds host memory (kPoisoned) or nothing (kNone).
enum class Contents : std::uint8_t { kPoisoned, kNone };

class FakeDeviceMemory final : public VmmProvider {
 public:
  // One device class and one host class at `granularity`, which must be a
  // multiple of the host page size. `capacity` bounds the backing that may
  // exist at once, across classes.
  FakeDeviceMemory(Bytes granularity, Bytes capacity, Contents contents = Contents::kPoisoned);
  ~FakeDeviceMemory() override;

  FakeDeviceMemory(const FakeDeviceMemory&) = delete;
  FakeDeviceMemory& operator=(const FakeDeviceMemory&) = delete;
  FakeDeviceMemory(FakeDeviceMemory&&) = delete;
  FakeDeviceMemory& operator=(FakeDeviceMemory&&) = delete;

  std::span<const AllocationClass> Classes() const override { return classes_; }

  // The next call of `operation` fails with `error`. By default it changes
  // nothing; with `applied`, it does what it was asked and still reports
  // the failure, as a driver whose outcome is unknown may have.
  void FailNext(Operation operation, ProviderError error, bool applied = false) {
    failures_[operation] = Scripted{.error = error, .applied = applied};
  }

  Bytes in_use() const { return in_use_; }

 protected:
  std::expected<std::uint64_t, Failure> DoReserve(Bytes size) override;
  std::expected<void, Failure> DoFree(std::uint64_t base, Bytes size) override;
  std::expected<Handle, Failure> DoCreate(const AllocationClass& allocation_class,
                                          Bytes size) override;
  std::expected<void, Failure> DoRelease(Handle handle, Bytes size) override;
  std::expected<void, Failure> DoMap(std::uint64_t address, Bytes size, Handle handle) override;
  std::expected<void, Failure> DoSetAccess(std::uint64_t address, Bytes size, Access access,
                                           bool host) override;
  std::expected<void, Failure> DoUnmap(std::uint64_t address, Bytes size) override;

 private:
  struct Scripted {
    ProviderError error = ProviderError::kFailed;
    bool applied = false;
  };
  // A scripted failure for this operation, consumed.
  std::optional<Scripted> Take(Operation operation);

  std::vector<AllocationClass> classes_;
  Bytes capacity_;
  Contents contents_;
  Handle next_unbacked_ = 0;  // handles of address-only backing
  Bytes in_use_;
  std::map<Operation, Scripted> failures_;
  // Reservations this fake made, to unmap any left at destruction.
  std::map<std::uint64_t, Bytes> reserved_;
  // Backing by handle: the file descriptor, or with Contents::kNone a
  // counter that names no file.
  std::map<Handle, Bytes> files_;
};

}  // namespace llmp::providers::fake

#endif  // LLMP_PROVIDERS_FAKE_FAKE_DEVICE_MEMORY_H_
