// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The device-memory provider (D-006, D-026, D-033, D-034;
// docs/architecture.md#providers): explicit virtual reservations, physical
// backing in an allocation class, and mapping with access control. Nothing
// here pages on demand. Touching address space whose backing is absent is a
// bug, and the fake makes it fault.
//
// This header holds no vendor types. A CUDA build implements it through the
// driver's VMM API (providers/cuda/), and every build has the
// deterministic fake (providers/fake/). Calls can block: they run on the
// VMM lane, or the device submission lane without one, never on the
// scheduler thread (D-048).
//
// One thread calls a provider at a time; nothing inside is synchronized.
// While the lanes run that is the one lane given the provider (a program
// with a VMM lane gives its device lane none), and the program's own
// thread calls it only when that lane has nothing in flight: at setup, at
// teardown, or after the work it waited for. A VMM lane with a handle
// reserve (scheduler::ReserveSettings) also calls it while it has no
// command, so beside one the program's thread calls it only before the
// lane runs or after it has stopped. A call that changes the
// provider (Reserve, Free, Create, Release, Map, SetAccess, Unmap) while
// another is under way is fatal; a read (RangeOf, MappedAt, Undetermined,
// the counts) must not overlap one either, which nothing checks. The
// driver's own calls may run alongside the device lanes' copies and fences.
//
// VmmProvider holds the rules every implementation shares, so the fake and
// CUDA reject the same misuse the same way:
//   - a reservation is address space only, a multiple of the granularity;
//   - backing is created in one allocation class, a multiple of that
//     class's granularity, and filled with nothing a caller may rely on;
//   - a backing is mapped whole, at a granularity-aligned offset inside one
//     reservation, into a hole, at most once at a time;
//   - a mapping starts with no access, until SetAccess grants it;
//   - Unmap removes whole mappings only;
//   - backing is released only when unmapped, and a reservation freed only
//     when nothing is mapped in it.
// Identities are generation-checked: a stale one names nothing. A primitive
// whose outcome is unknown (kUnknown) leaves the reservation and backing it
// touched undetermined: every later call on them is refused
// (kUndetermined), never retried, since a retried release or free could hit
// a handle or range the driver has since handed to someone else. Their
// owner quarantines them (D-048), and anything a refusal names.
//
// Addresses are the device's virtual addresses. On validated Spark
// configurations host-kind backing is GPU-accessible host memory, and with
// read or write access the same address is a CPU pointer too: direct file
// I/O lands there (D-034), and the GPU copies it into device backing (D-081).

#ifndef LLMP_PROVIDERS_DEVICE_MEMORY_H_
#define LLMP_PROVIDERS_DEVICE_MEMORY_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "base/bytes.h"
#include "base/ids.h"

namespace llmp::providers {

using base::Bytes;

enum class BackingKind : std::uint8_t {
  kDevice,  // device-local memory
  kHost,    // GPU-accessible host memory on a NUMA node (D-034)
};

struct AllocationClass {
  BackingKind kind = BackingKind::kDevice;
  std::uint32_t location = 0;  // the device ordinal, or the host NUMA node
  Bytes granularity;           // backing and offsets are multiples of it
};

enum class Access : std::uint8_t { kNone, kRead, kReadWrite };

enum class ProviderError : std::uint8_t {
  kInvalid,      // misuse: bad size or offset, overlap, still mapped, stale identity
  kOutOfMemory,  // the class has no room for the backing
  kUnsupported,  // the provider cannot do this here
  kFailed,       // known failure; a primitive changes nothing, a composite may be partial
  kUnknown,      // the outcome is unknown: a fault (D-048), not a retry
  // Refused, changing nothing: an earlier unknown outcome left the
  // reservation or backing undetermined. Not a misuse and not proof of
  // anything: the owner quarantines what it names, never retries.
  kUndetermined,
};

struct Failure {
  ProviderError error = ProviderError::kFailed;
  std::string detail;
};

std::string ToString(ProviderError error);

struct ReservationTag {
  static constexpr const char* kName = "reservation";
};
struct BackingTag {
  static constexpr const char* kName = "backing";
};
using ReservationId = base::Id<ReservationTag>;
using BackingId = base::Id<BackingTag>;

struct AddressRange {
  std::uint64_t base = 0;
  Bytes size;
};

class DeviceMemory {
 public:
  DeviceMemory() = default;
  DeviceMemory(const DeviceMemory&) = delete;
  DeviceMemory& operator=(const DeviceMemory&) = delete;
  DeviceMemory(DeviceMemory&&) = delete;
  DeviceMemory& operator=(DeviceMemory&&) = delete;
  virtual ~DeviceMemory() = default;

  virtual std::span<const AllocationClass> Classes() const = 0;
  // The granularity of reservations and offsets: a multiple of every
  // class's granularity.
  virtual Bytes Granularity() const = 0;

  virtual std::expected<ReservationId, Failure> Reserve(Bytes size) = 0;
  virtual std::expected<AddressRange, Failure> RangeOf(ReservationId reservation) const = 0;
  virtual std::expected<void, Failure> Free(ReservationId reservation) = 0;

  virtual std::expected<BackingId, Failure> Create(std::size_t allocation_class, Bytes size) = 0;
  virtual std::expected<void, Failure> Release(BackingId backing) = 0;

  virtual std::expected<void, Failure> Map(ReservationId reservation, Bytes offset,
                                           BackingId backing) = 0;
  // May change earlier runs before a later run fails in a mixed-kind
  // range. After a known failure, retry the whole range before using it;
  // an unknown outcome quarantines the reservation and affected backing.
  virtual std::expected<void, Failure> SetAccess(ReservationId reservation, Bytes offset,
                                                 Bytes size, Access access) = 0;
  virtual std::expected<void, Failure> Unmap(ReservationId reservation, Bytes offset,
                                             Bytes size) = 0;
  // The backing mapped at exactly `offset` of the reservation, if any: how
  // an owner that tracks places, not handles, finds what to release after
  // unmapping it.
  virtual std::optional<BackingId> MappedAt(ReservationId reservation, Bytes offset) const = 0;
};

// The shared rules over an implementation's primitive operations.
class VmmProvider : public DeviceMemory {
 public:
  Bytes Granularity() const override { return granularity_; }

  std::expected<ReservationId, Failure> Reserve(Bytes size) override;
  std::expected<AddressRange, Failure> RangeOf(ReservationId reservation) const override;
  std::expected<void, Failure> Free(ReservationId reservation) override;
  std::expected<BackingId, Failure> Create(std::size_t allocation_class, Bytes size) override;
  std::expected<void, Failure> Release(BackingId backing) override;
  std::expected<void, Failure> Map(ReservationId reservation, Bytes offset,
                                   BackingId backing) override;
  std::expected<void, Failure> SetAccess(ReservationId reservation, Bytes offset, Bytes size,
                                         Access access) override;
  std::expected<void, Failure> Unmap(ReservationId reservation, Bytes offset, Bytes size) override;
  std::optional<BackingId> MappedAt(ReservationId reservation, Bytes offset) const override;

  // Whether an unknown outcome left it undetermined.
  bool Undetermined(ReservationId reservation) const;
  bool Undetermined(BackingId backing) const;
  // Backing an unknown outcome left undetermined, including backing a
  // Create with an unknown outcome may have made: the owner keeps it
  // charged as quarantined (invariant 8).
  Bytes UndeterminedBytes() const;

  // What is live, for tests and leak checks at shutdown.
  std::size_t reservations() const { return reservations_.size(); }
  std::size_t backings() const { return backings_.size(); }

 protected:
  // The implementation's handle for a backing: an opaque number.
  using Handle = std::uint64_t;

  // `granularity` must be a non-zero multiple of every class's granularity.
  explicit VmmProvider(Bytes granularity) : granularity_(granularity) {}

  // The primitives, called only once the rules above pass.
  virtual std::expected<std::uint64_t, Failure> DoReserve(Bytes size) = 0;
  virtual std::expected<void, Failure> DoFree(std::uint64_t base, Bytes size) = 0;
  virtual std::expected<Handle, Failure> DoCreate(const AllocationClass& allocation_class,
                                                  Bytes size) = 0;
  virtual std::expected<void, Failure> DoRelease(Handle handle, Bytes size) = 0;
  virtual std::expected<void, Failure> DoMap(std::uint64_t address, Bytes size, Handle handle) = 0;
  // Applies to every mapping inside the range, all of one kind: with
  // `host`, host backing, which the CPU is given the same access to.
  virtual std::expected<void, Failure> DoSetAccess(std::uint64_t address, Bytes size, Access access,
                                                   bool host) = 0;
  virtual std::expected<void, Failure> DoUnmap(std::uint64_t address, Bytes size) = 0;

 private:
  struct Mapping {
    BackingId backing;
    Bytes size;
  };
  struct Reservation {
    std::uint64_t base = 0;
    Bytes size;
    std::map<std::uint64_t, Mapping> mappings;  // by offset
    bool undetermined = false;
  };
  struct Backing {
    std::size_t allocation_class = 0;
    Bytes size;
    Handle handle = 0;
    bool mapped = false;
    bool undetermined = false;
  };

  // Held by each call that changes the provider: a second at once is fatal
  // (one thread at a time, the header's contract).
  class Exclusive {
   public:
    explicit Exclusive(std::atomic<bool>& changing);
    ~Exclusive() { changing_.store(false, std::memory_order_release); }
    Exclusive(const Exclusive&) = delete;
    Exclusive& operator=(const Exclusive&) = delete;
    Exclusive(Exclusive&&) = delete;
    Exclusive& operator=(Exclusive&&) = delete;

   private:
    std::atomic<bool>& changing_;
  };

  // The mappings exactly covering [offset, offset + size), or nothing if
  // the range splits a mapping, leaves a hole, or runs outside.
  static std::expected<std::vector<std::uint64_t>, Failure> Covered(const Reservation& reservation,
                                                                    Bytes offset, Bytes size);
  bool Aligned(Bytes value) const { return value.value() % granularity_.value() == 0; }

  Bytes granularity_;
  std::atomic<bool> changing_{false};
  base::SlotTable<ReservationTag, Reservation> reservations_;
  base::SlotTable<BackingTag, Backing> backings_;
};

}  // namespace llmp::providers

#endif  // LLMP_PROVIDERS_DEVICE_MEMORY_H_
