// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "providers/device_memory.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/check.h"

namespace llmp::providers {
namespace {

std::unexpected<Failure> Invalid(std::string detail) {
  return std::unexpected(Failure{.error = ProviderError::kInvalid, .detail = std::move(detail)});
}

std::unexpected<Failure> RefuseUndetermined() {
  return std::unexpected(
      Failure{.error = ProviderError::kUndetermined,
              .detail = "undetermined after an unknown outcome: the owner must quarantine it"});
}

bool Unknown(const Failure& failure) { return failure.error == ProviderError::kUnknown; }

}  // namespace

std::string ToString(ProviderError error) {
  switch (error) {
    case ProviderError::kInvalid:
      return "invalid request";
    case ProviderError::kOutOfMemory:
      return "out of memory";
    case ProviderError::kUnsupported:
      return "unsupported";
    case ProviderError::kFailed:
      return "failed";
    case ProviderError::kUnknown:
      return "outcome unknown";
    case ProviderError::kUndetermined:
      return "undetermined after an unknown outcome";
  }
  return "unknown provider error";
}

VmmProvider::Exclusive::Exclusive(std::atomic<bool>& changing) : changing_(changing) {
  base::Check(!changing_.exchange(true, std::memory_order_acquire),
              "two threads changed a VMM provider at once: one thread calls it at a time");
}

std::expected<ReservationId, Failure> VmmProvider::Reserve(Bytes size) {
  const Exclusive exclusive(changing_);
  if (size == Bytes() || !Aligned(size)) {
    return Invalid(std::format("a reservation of {} bytes is not a multiple of {}", size.value(),
                               granularity_.value()));
  }
  auto base = DoReserve(size);
  if (!base) {
    if (Unknown(base.error())) {
      // The range may exist: keep it charged, with no address to use.
      (void)reservations_.Insert(
          Reservation{.base = 0, .size = size, .mappings = {}, .undetermined = true});
    }
    return std::unexpected(base.error());
  }
  const ReservationId id = reservations_.Insert(
      Reservation{.base = *base, .size = size, .mappings = {}, .undetermined = false});
  if (!id.valid()) {
    const auto undone = DoFree(*base, size);
    return std::unexpected(
        Failure{.error = undone ? ProviderError::kFailed : undone.error().error,
                .detail = undone ? "the reservation table is full"
                                 : "the reservation table is full, and freeing the range failed: " +
                                       undone.error().detail});
  }
  return id;
}

bool VmmProvider::Undetermined(ReservationId reservation) const {
  const Reservation* found = reservations_.Find(reservation);
  return found != nullptr && found->undetermined;
}

Bytes VmmProvider::UndeterminedBytes() const {
  std::uint64_t total = 0;
  backings_.ForEach([&](BackingId, const Backing& backing) {
    if (backing.undetermined) {
      total += backing.size.value();
    }
  });
  return Bytes(total);
}

bool VmmProvider::Undetermined(BackingId backing) const {
  const Backing* found = backings_.Find(backing);
  return found != nullptr && found->undetermined;
}

std::expected<AddressRange, Failure> VmmProvider::RangeOf(ReservationId reservation) const {
  const Reservation* found = reservations_.Find(reservation);
  if (found == nullptr) {
    return Invalid("stale or unknown reservation");
  }
  if (found->undetermined) {
    return RefuseUndetermined();
  }
  return AddressRange{.base = found->base, .size = found->size};
}

std::optional<BackingId> VmmProvider::MappedAt(ReservationId reservation, Bytes offset) const {
  const Reservation* found = reservations_.Find(reservation);
  if (found == nullptr) {
    return std::nullopt;
  }
  const auto mapping = found->mappings.find(offset.value());
  return mapping != found->mappings.end() ? std::optional(mapping->second.backing) : std::nullopt;
}

std::expected<void, Failure> VmmProvider::Free(ReservationId reservation) {
  const Exclusive exclusive(changing_);
  Reservation* found = reservations_.Find(reservation);
  if (found == nullptr) {
    return Invalid("stale or unknown reservation");
  }
  if (found->undetermined) {
    return RefuseUndetermined();
  }
  if (!found->mappings.empty()) {
    return Invalid("the reservation still has mappings");
  }
  if (auto freed = DoFree(found->base, found->size); !freed) {
    reservations_.Find(reservation)->undetermined = Unknown(freed.error());
    return freed;
  }
  (void)reservations_.Erase(reservation);
  return {};
}

std::expected<BackingId, Failure> VmmProvider::Create(std::size_t allocation_class, Bytes size) {
  const Exclusive exclusive(changing_);
  const std::span<const AllocationClass> classes = Classes();
  if (allocation_class >= classes.size()) {
    return Invalid("no such allocation class");
  }
  const AllocationClass& chosen = classes[allocation_class];
  if (size == Bytes() || size.value() % chosen.granularity.value() != 0 || !Aligned(size)) {
    return Invalid(
        std::format("backing of {} bytes is not a multiple of the granularity", size.value()));
  }
  auto handle = DoCreate(chosen, size);
  if (!handle) {
    if (Unknown(handle.error())) {
      // The backing may exist: keep it charged, with no handle to use.
      (void)backings_.Insert(Backing{.allocation_class = allocation_class,
                                     .size = size,
                                     .handle = 0,
                                     .mapped = false,
                                     .undetermined = true});
    }
    return std::unexpected(handle.error());
  }
  const BackingId id = backings_.Insert(Backing{.allocation_class = allocation_class,
                                                .size = size,
                                                .handle = *handle,
                                                .mapped = false,
                                                .undetermined = false});
  if (!id.valid()) {
    const auto undone = DoRelease(*handle, size);
    return std::unexpected(
        Failure{.error = undone ? ProviderError::kFailed : undone.error().error,
                .detail = undone ? "the backing table is full"
                                 : "the backing table is full, and releasing the backing failed: " +
                                       undone.error().detail});
  }
  return id;
}

std::expected<void, Failure> VmmProvider::Release(BackingId backing) {
  const Exclusive exclusive(changing_);
  Backing* found = backings_.Find(backing);
  if (found == nullptr) {
    return Invalid("stale or unknown backing");
  }
  if (found->undetermined) {
    return RefuseUndetermined();
  }
  if (found->mapped) {
    return Invalid("the backing is still mapped");
  }
  if (auto released = DoRelease(found->handle, found->size); !released) {
    // Retrying an unknown release could free a handle the driver has
    // since given to someone else.
    found->undetermined = Unknown(released.error());
    return released;
  }
  (void)backings_.Erase(backing);
  return {};
}

std::expected<void, Failure> VmmProvider::Map(ReservationId reservation, Bytes offset,
                                              BackingId backing) {
  const Exclusive exclusive(changing_);
  Reservation* place = reservations_.Find(reservation);
  Backing* found = backings_.Find(backing);
  if (place == nullptr || found == nullptr) {
    return Invalid("stale or unknown reservation or backing");
  }
  if (place->undetermined || found->undetermined) {
    return RefuseUndetermined();
  }
  if (found->mapped) {
    return Invalid("the backing is already mapped");
  }
  const std::optional<Bytes> end = offset.Plus(found->size);
  if (!Aligned(offset) || !end || *end > place->size) {
    return Invalid("the mapping does not fit the reservation at that offset");
  }
  // Into a hole: no mapping may overlap [offset, end).
  auto after = place->mappings.lower_bound(offset.value());
  if (after != place->mappings.end() && after->first < end->value()) {
    return Invalid("the mapping overlaps another");
  }
  if (after != place->mappings.begin()) {
    const auto before = std::prev(after);
    if (before->first + before->second.size.value() > offset.value()) {
      return Invalid("the mapping overlaps another");
    }
  }
  if (auto mapped = DoMap(place->base + offset.value(), found->size, found->handle); !mapped) {
    if (Unknown(mapped.error())) {
      place->undetermined = true;  // the hole may hold the mapping now
      found->undetermined = true;
    }
    return mapped;
  }
  place->mappings.emplace(offset.value(), Mapping{.backing = backing, .size = found->size});
  found->mapped = true;
  return {};
}

std::expected<std::vector<std::uint64_t>, Failure> VmmProvider::Covered(
    const Reservation& reservation, Bytes offset, Bytes size) {
  const std::optional<Bytes> end = offset.Plus(size);
  if (size == Bytes() || !end || *end > reservation.size) {
    return Invalid("the range runs outside the reservation");
  }
  std::vector<std::uint64_t> covered;
  std::uint64_t at = offset.value();
  auto mapping = reservation.mappings.find(at);
  while (at < end->value()) {
    if (mapping == reservation.mappings.end() || mapping->first != at) {
      return Invalid("the range does not cover whole mappings");
    }
    covered.push_back(at);
    at += mapping->second.size.value();
    ++mapping;
  }
  if (at != end->value()) {
    return Invalid("the range splits a mapping");
  }
  return covered;
}

std::expected<void, Failure> VmmProvider::SetAccess(ReservationId reservation, Bytes offset,
                                                    Bytes size, Access access) {
  const Exclusive exclusive(changing_);
  Reservation* place = reservations_.Find(reservation);
  if (place == nullptr) {
    return Invalid("stale or unknown reservation");
  }
  if (place->undetermined) {
    return RefuseUndetermined();
  }
  auto covered = Covered(*place, offset, size);
  if (!covered) {
    return std::unexpected(covered.error());
  }
  const auto quarantine = [&] {
    place->undetermined = true;
    for (const std::uint64_t at : *covered) {
      backings_.Find(place->mappings.at(at).backing)->undetermined = true;
    }
  };
  // One call per run of mappings of the same kind: the CPU's access applies
  // to host backing only, and a driver refuses it over device backing.
  // A failure part-way leaves earlier runs changed; access is idempotent,
  // so the caller retries the whole range.
  std::uint64_t run_start = offset.value();
  std::uint64_t run_end = run_start;
  bool run_host = false;
  for (const std::uint64_t at : *covered) {
    const Mapping& mapping = place->mappings.at(at);
    const Backing* backing = backings_.Find(mapping.backing);
    base::Check(backing != nullptr, "a mapping of a released backing");
    const bool host = Classes()[backing->allocation_class].kind == BackingKind::kHost;
    if (run_end > run_start && host != run_host) {
      if (auto set =
              DoSetAccess(place->base + run_start, Bytes(run_end - run_start), access, run_host);
          !set) {
        if (Unknown(set.error())) {
          quarantine();
        }
        return set;
      }
      run_start = run_end;
    }
    run_host = host;
    run_end = at + mapping.size.value();
  }
  auto set = DoSetAccess(place->base + run_start, Bytes(run_end - run_start), access, run_host);
  if (!set && Unknown(set.error())) {
    quarantine();
  }
  return set;
}

std::expected<void, Failure> VmmProvider::Unmap(ReservationId reservation, Bytes offset,
                                                Bytes size) {
  const Exclusive exclusive(changing_);
  Reservation* place = reservations_.Find(reservation);
  if (place == nullptr) {
    return Invalid("stale or unknown reservation");
  }
  if (place->undetermined) {
    return RefuseUndetermined();
  }
  auto covered = Covered(*place, offset, size);
  if (!covered) {
    return std::unexpected(covered.error());
  }
  if (auto unmapped = DoUnmap(place->base + offset.value(), size); !unmapped) {
    if (Unknown(unmapped.error())) {
      place->undetermined = true;
      for (const std::uint64_t at : *covered) {
        backings_.Find(place->mappings.at(at).backing)->undetermined = true;
      }
    }
    return unmapped;
  }
  for (const std::uint64_t at : *covered) {
    Backing* backing = backings_.Find(place->mappings.at(at).backing);
    base::Check(backing != nullptr, "a mapping of a released backing");
    backing->mapped = false;
    place->mappings.erase(at);
  }
  return {};
}

}  // namespace llmp::providers
