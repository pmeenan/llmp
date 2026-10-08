// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "catalog/catalog.h"

#include <algorithm>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/check.h"

namespace llmp::catalog {
namespace {

Bytes Add(Bytes a, Bytes b) {
  const std::optional<Bytes> sum = a.Plus(b);
  base::Check(sum.has_value(), "occupancy overflow");
  return sum.value_or(Bytes());
}

Bytes Subtract(Bytes a, Bytes b) {
  const std::optional<Bytes> difference = a.Minus(b);
  base::Check(difference.has_value(), "occupancy underflow");
  return difference.value_or(Bytes());
}

// 64-bit generations: at one change per nanosecond they last centuries.
void Advance(std::uint64_t& generation) {
  base::Check(generation < UINT64_MAX, "generation exhausted");
  ++generation;
}

}  // namespace

Bytes Occupancy::Total() const {
  Bytes total;
  for (const Bytes part : {held, idle, pinned, loading, evicting, quarantined}) {
    total = Add(total, part);
  }
  return total;
}

std::string ToString(CatalogError error) {
  switch (error) {
    case CatalogError::kUnknownDomain:
      return "unknown memory domain";
    case CatalogError::kUnknownId:
      return "stale or unknown identity";
    case CatalogError::kWrongState:
      return "the extent is not in the state this transition needs";
    case CatalogError::kStaleTicket:
      return "the ticket names an earlier or different operation";
    case CatalogError::kStaleContent:
      return "the closure recorded contents that have since changed";
    case CatalogError::kHeld:
      return "the extent is leased or registered";
    case CatalogError::kNotEvictable:
      return "the extent is not evictable";
    case CatalogError::kNotResident:
      return "an extent of the closure is not resident";
    case CatalogError::kBadRange:
      return "a range lies outside its extent or spans domains, or a descriptor is invalid";
    case CatalogError::kOverBudget:
      return "the domain's occupancy would exceed its budget";
    case CatalogError::kExhausted:
      return "an identity table is exhausted";
  }
  return "unknown catalog error";
}

bool Catalog::Evictable(const ExtentView& extent) {
  if (extent.state != ExtentState::kResident || extent.leases != 0 || extent.registrations != 0 ||
      extent.descriptor.memory_class == MemoryClass::kUnknown) {
    return false;
  }
  switch (extent.descriptor.recovery) {
    case Recovery::kFromArtifact:
    case Recovery::kDiscardable:
      return true;
    case Recovery::kPreserve:
      return extent.discarded;  // only once deliberately invalidated
    case Recovery::kPinned:
      return false;
  }
  return false;
}

Ticket Catalog::Begin(ExtentId id, ExtentRecord& record, Operation operation) {
  base::Check(record.operations < UINT64_MAX, "operation counter exhausted");
  ++record.operations;
  record.operation = operation;
  return Ticket{.extent = id, .operation = operation, .serial = record.operations};
}

std::expected<Catalog::ExtentRecord*, CatalogError> Catalog::Current(const Ticket& ticket,
                                                                     Operation operation,
                                                                     ExtentState state) {
  ExtentRecord* record = Extent(ticket.extent);
  if (record == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  if (ticket.operation != operation || ticket.operation != record->operation ||
      ticket.serial != record->operations) {
    return std::unexpected(CatalogError::kStaleTicket);
  }
  if (record->view.state != state) {
    return std::unexpected(CatalogError::kWrongState);
  }
  return record;
}

Catalog::Bucket Catalog::BucketOf(const ExtentView& view) {
  switch (view.state) {
    case ExtentState::kNonresident:
      return Bucket::kNone;
    case ExtentState::kLoading:
      return Bucket::kLoading;
    case ExtentState::kEvicting:
      return Bucket::kEvicting;
    case ExtentState::kQuarantined:
      return Bucket::kQuarantined;
    case ExtentState::kResident:
      if (view.descriptor.recovery == Recovery::kPinned) {
        return Bucket::kPinned;
      }
      return view.leases > 0 || view.registrations > 0 ? Bucket::kHeld : Bucket::kIdle;
  }
  return Bucket::kNone;
}

void Catalog::Recount(ExtentRecord& record, Bucket before) {
  const Bucket after = BucketOf(record.view);
  if (before == after) {
    return;
  }
  Domain* domain = domains_.Find(record.view.descriptor.domain);
  base::Check(domain != nullptr, "extent in an unknown domain");
  Occupancy& occupancy = domain->occupancy;
  const Bytes size = record.view.descriptor.size;
  const auto bucket = [&occupancy](Bucket which) -> Bytes* {
    switch (which) {
      case Bucket::kNone:
        return nullptr;
      case Bucket::kHeld:
        return &occupancy.held;
      case Bucket::kIdle:
        return &occupancy.idle;
      case Bucket::kPinned:
        return &occupancy.pinned;
      case Bucket::kLoading:
        return &occupancy.loading;
      case Bucket::kEvicting:
        return &occupancy.evicting;
      case Bucket::kQuarantined:
        return &occupancy.quarantined;
    }
    return nullptr;
  };
  Bytes& by_class =
      occupancy.by_class.at(static_cast<std::size_t>(record.view.descriptor.memory_class));
  if (Bytes* from = bucket(before)) {
    *from = Subtract(*from, size);
  } else {
    by_class = Add(by_class, size);  // backing starts to exist
  }
  if (Bytes* to = bucket(after)) {
    *to = Add(*to, size);
  } else {
    by_class = Subtract(by_class, size);  // backing is gone
  }
}

DomainId Catalog::AddDomain(std::string name) {
  return domains_.Insert(Domain{.name = std::move(name), .occupancy = {}, .registered = {}});
}

Occupancy Catalog::OccupancyOf(DomainId domain) const {
  const Domain* found = domains_.Find(domain);
  return found != nullptr ? found->occupancy : Occupancy{};
}

std::expected<ExtentId, CatalogError> Catalog::AddExtent(const ExtentDescriptor& descriptor,
                                                         bool resident) {
  if (domains_.Find(descriptor.domain) == nullptr) {
    return std::unexpected(CatalogError::kUnknownDomain);
  }
  if (descriptor.size == Bytes() ||
      static_cast<std::size_t>(descriptor.memory_class) >= kMemoryClassCount ||
      descriptor.recovery > Recovery::kPinned ||
      (descriptor.memory_class == MemoryClass::kUnknown &&
       descriptor.recovery != Recovery::kPinned)) {
    return std::unexpected(CatalogError::kBadRange);  // unknown allocations are pinned (D-006)
  }
  const std::optional<Bytes> registered =
      domains_.Find(descriptor.domain)->registered.Plus(descriptor.size);
  if (!registered) {
    return std::unexpected(CatalogError::kBadRange);
  }
  ExtentRecord record;
  record.view.descriptor = descriptor;
  const ExtentId id = extents_.Insert(record);
  if (!id.valid()) {
    return std::unexpected(CatalogError::kExhausted);
  }
  domains_.Find(descriptor.domain)->registered = *registered;
  if (resident) {
    ExtentRecord& added = *Extent(id);
    const Bucket before = BucketOf(added.view);
    added.view.state = ExtentState::kResident;
    Recount(added, before);
  }
  return id;
}

std::expected<void, CatalogError> Catalog::RemoveExtent(ExtentId extent) {
  const ExtentRecord* record = Extent(extent);
  if (record == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  if (record->view.state != ExtentState::kNonresident || record->resources != 0) {
    return std::unexpected(CatalogError::kWrongState);
  }
  Domain* domain = domains_.Find(record->view.descriptor.domain);
  domain->registered = Subtract(domain->registered, record->view.descriptor.size);
  (void)extents_.Erase(extent);
  return {};
}

std::expected<void, CatalogError> Catalog::ReleasePinned(ExtentId extent) {
  ExtentRecord* record = Extent(extent);
  if (record == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  ExtentView& view = record->view;
  if (view.state != ExtentState::kResident || view.descriptor.recovery != Recovery::kPinned) {
    return std::unexpected(CatalogError::kWrongState);
  }
  if (view.leases > 0 || view.registrations > 0) {
    return std::unexpected(CatalogError::kHeld);
  }
  const Bucket before = BucketOf(view);
  view.state = ExtentState::kNonresident;
  Advance(view.backing_generation);
  Advance(view.content_generation);
  Recount(*record, before);
  return {};
}

std::expected<ResourceId, CatalogError> Catalog::AddResource(std::span<const Range> ranges) {
  if (ranges.empty()) {
    return std::unexpected(CatalogError::kBadRange);
  }
  std::optional<DomainId> domain;
  for (const Range& range : ranges) {
    const ExtentRecord* record = Extent(range.extent);
    if (record == nullptr) {
      return std::unexpected(CatalogError::kUnknownId);
    }
    const std::optional<Bytes> end = range.offset.Plus(range.length);
    if (range.length == Bytes() || !end || *end > record->view.descriptor.size) {
      return std::unexpected(CatalogError::kBadRange);
    }
    if (domain && *domain != record->view.descriptor.domain) {
      return std::unexpected(CatalogError::kBadRange);
    }
    domain = record->view.descriptor.domain;
  }
  const ResourceId id = resources_.Insert(Resource{.ranges = {ranges.begin(), ranges.end()}});
  if (!id.valid()) {
    return std::unexpected(CatalogError::kExhausted);
  }
  for (const Range& range : ranges) {
    ++Extent(range.extent)->resources;
  }
  return id;
}

std::expected<void, CatalogError> Catalog::CanRetirePinned(ExtentId extent) const {
  const ExtentRecord* record = Extent(extent);
  if (record == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  const auto& view = record->view;
  if (view.state != ExtentState::kResident || view.descriptor.recovery != Recovery::kPinned ||
      record->resources != 0) {
    return std::unexpected(CatalogError::kWrongState);
  }
  if (view.leases != 0 || view.registrations != 0) {
    return std::unexpected(CatalogError::kHeld);
  }
  return {};
}

std::expected<void, CatalogError> Catalog::RemoveResource(ResourceId resource) {
  const Resource* found = resources_.Find(resource);
  if (found == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  for (const Range& range : found->ranges) {
    ExtentRecord* record = Extent(range.extent);
    base::Check(record != nullptr && record->resources > 0, "resource placed in a missing extent");
    --record->resources;
  }
  (void)resources_.Erase(resource);
  return {};
}

std::expected<Closure, CatalogError> Catalog::ClosureOf(
    std::span<const ResourceId> resources) const {
  std::vector<ExtentId> extents;
  for (const ResourceId resource : resources) {
    const Resource* found = resources_.Find(resource);
    if (found == nullptr) {
      return std::unexpected(CatalogError::kUnknownId);
    }
    for (const Range& range : found->ranges) {
      extents.push_back(range.extent);
    }
  }
  return ClosureOfExtents(extents);
}

std::expected<Closure, CatalogError> Catalog::ClosureOfExtents(
    std::span<const ExtentId> extents) const {
  std::vector<ExtentId> unique(extents.begin(), extents.end());
  std::ranges::sort(unique);
  const auto [first, last] = std::ranges::unique(unique);
  unique.erase(first, last);
  Closure closure;
  closure.extents.reserve(unique.size());
  // Physical backing, counted once per extent however many resources share it.
  for (const ExtentId extent : unique) {
    const ExtentRecord* record = Extent(extent);
    if (record == nullptr) {
      return std::unexpected(CatalogError::kUnknownId);
    }
    closure.extents.emplace_back(extent, record->view.content_generation);
    Bytes& bytes = closure.bytes_by_domain[record->view.descriptor.domain];
    bytes = Add(bytes, record->view.descriptor.size);
  }
  return closure;
}

std::expected<Ticket, CatalogError> Catalog::BeginLoad(ExtentId extent, Bytes budget) {
  ExtentRecord* record = Extent(extent);
  if (record == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  ExtentView& view = record->view;
  if (view.state != ExtentState::kNonresident) {
    return std::unexpected(CatalogError::kWrongState);
  }
  const std::optional<Bytes> after =
      OccupancyOf(view.descriptor.domain).Total().Plus(view.descriptor.size);
  if (!after || *after > budget) {
    return std::unexpected(CatalogError::kOverBudget);
  }
  const Bucket before = BucketOf(view);
  view.state = ExtentState::kLoading;
  Recount(*record, before);
  return Begin(extent, *record, Operation::kLoad);
}

std::expected<void, CatalogError> Catalog::CompleteLoad(const Ticket& ticket) {
  auto record = Current(ticket, Operation::kLoad, ExtentState::kLoading);
  if (!record) {
    return std::unexpected(record.error());
  }
  ExtentView& view = (*record)->view;
  const Bucket before = BucketOf(view);
  view.state = ExtentState::kResident;
  view.discarded = false;  // fresh contents
  view.preserved = false;  // the resident copy is the live one now
  Recount(**record, before);
  return {};
}

std::expected<void, CatalogError> Catalog::FailLoad(const Ticket& ticket, bool completion_known) {
  auto record = Current(ticket, Operation::kLoad, ExtentState::kLoading);
  if (!record) {
    return std::unexpected(record.error());
  }
  ExtentView& view = (*record)->view;
  const Bucket before = BucketOf(view);
  if (completion_known) {
    view.state = ExtentState::kNonresident;
    Advance(view.backing_generation);
  } else {
    view.state = ExtentState::kQuarantined;  // stays charged (D-048)
  }
  Recount(**record, before);
  return {};
}

std::expected<LeaseId, CatalogError> Catalog::AcquireLease(const Closure& closure) {
  // All or none: check every extent before changing any.
  std::vector<ExtentId> extents;
  extents.reserve(closure.extents.size());
  for (const auto& [extent, content_generation] : closure.extents) {
    const ExtentRecord* record = Extent(extent);
    if (record == nullptr) {
      return std::unexpected(CatalogError::kUnknownId);
    }
    if (record->view.state != ExtentState::kResident) {
      return std::unexpected(CatalogError::kNotResident);
    }
    if (record->view.content_generation != content_generation || record->view.discarded) {
      return std::unexpected(CatalogError::kStaleContent);
    }
    extents.push_back(extent);
  }
  std::ranges::sort(extents);
  const auto [first, last] = std::ranges::unique(extents);
  extents.erase(first, last);
  const LeaseId lease = leases_.Insert(Lease{.extents = extents});
  if (!lease.valid()) {
    return std::unexpected(CatalogError::kExhausted);
  }
  for (const ExtentId extent : extents) {
    ExtentRecord& record = *Extent(extent);
    const Bucket before = BucketOf(record.view);
    base::Check(record.view.leases < UINT32_MAX, "lease count overflow");
    ++record.view.leases;
    Recount(record, before);
  }
  return lease;
}

std::expected<void, CatalogError> Catalog::ReleaseLease(LeaseId lease) {
  const Lease* found = leases_.Find(lease);
  if (found == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  for (const ExtentId extent : found->extents) {
    ExtentRecord* record = Extent(extent);
    base::Check(record != nullptr && record->view.leases > 0,
                "a lease names an extent it does not hold");
    const Bucket before = BucketOf(record->view);
    --record->view.leases;
    Recount(*record, before);
  }
  (void)leases_.Erase(lease);
  return {};
}

std::expected<void, CatalogError> Catalog::RecordUse(LeaseId lease, std::uint64_t tick) {
  const Lease* found = leases_.Find(lease);
  if (found == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  for (const ExtentId extent : found->extents) {
    ExtentView& view = Extent(extent)->view;
    view.last_use = std::max(view.last_use, tick);
  }
  return {};
}

std::expected<RegistrationId, CatalogError> Catalog::AddRegistration(ExtentId extent) {
  ExtentRecord* record = Extent(extent);
  if (record == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  if (record->view.state != ExtentState::kResident) {
    return std::unexpected(CatalogError::kNotResident);
  }
  if (record->view.discarded) {
    return std::unexpected(CatalogError::kStaleContent);  // nothing may write discarded contents
  }
  const RegistrationId registration = registrations_.Insert(Registration{.extent = extent});
  if (!registration.valid()) {
    return std::unexpected(CatalogError::kExhausted);
  }
  const Bucket before = BucketOf(record->view);
  base::Check(record->view.registrations < UINT32_MAX, "registration count overflow");
  ++record->view.registrations;
  Recount(*record, before);
  return registration;
}

std::expected<void, CatalogError> Catalog::RetireRegistration(RegistrationId registration) {
  const Registration* found = registrations_.Find(registration);
  if (found == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  ExtentRecord* record = Extent(found->extent);
  base::Check(record != nullptr && record->view.registrations > 0,
              "a registration names an extent it does not hold");
  const Bucket before = BucketOf(record->view);
  --record->view.registrations;
  Recount(*record, before);
  (void)registrations_.Erase(registration);
  return {};
}

std::expected<void, CatalogError> Catalog::InvalidateContents(ExtentId extent) {
  ExtentRecord* record = Extent(extent);
  if (record == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  ExtentView& view = record->view;
  if (view.state != ExtentState::kResident || view.descriptor.recovery == Recovery::kPinned ||
      view.descriptor.recovery == Recovery::kFromArtifact) {
    return std::unexpected(CatalogError::kWrongState);
  }
  if (view.leases > 0 || view.registrations > 0) {
    return std::unexpected(CatalogError::kHeld);  // consumers may still read it
  }
  Advance(view.content_generation);
  view.discarded = true;
  return {};
}

std::expected<void, CatalogError> Catalog::ForgetPreserved(ExtentId extent) {
  ExtentRecord* record = Extent(extent);
  if (record == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  ExtentView& view = record->view;
  if (view.state != ExtentState::kNonresident || view.descriptor.recovery != Recovery::kPreserve) {
    return std::unexpected(CatalogError::kWrongState);
  }
  if (view.leases != 0 || view.registrations != 0) {
    return std::unexpected(CatalogError::kHeld);
  }
  Advance(view.content_generation);
  view.preserved = false;
  view.discarded = false;
  return {};
}

std::expected<void, CatalogError> Catalog::ReviveDiscarded(ExtentId extent) {
  ExtentRecord* record = Extent(extent);
  if (record == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  ExtentView& view = record->view;
  if (view.state != ExtentState::kResident || view.descriptor.recovery != Recovery::kPreserve ||
      !view.discarded) {
    return std::unexpected(CatalogError::kWrongState);
  }
  if (view.leases != 0 || view.registrations != 0) {
    return std::unexpected(CatalogError::kHeld);
  }
  Advance(view.content_generation);
  view.discarded = false;
  return {};
}

std::expected<void, CatalogError> Catalog::ReplaceContents(ExtentId extent) {
  ExtentRecord* record = Extent(extent);
  if (record == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  ExtentView& view = record->view;
  if (view.state != ExtentState::kResident || view.descriptor.recovery != Recovery::kPreserve ||
      view.leases == 0) {
    return std::unexpected(CatalogError::kWrongState);  // only its lease-holding writer
  }
  if (view.leases != 1 || view.registrations != 0) {
    return std::unexpected(CatalogError::kHeld);  // old readers must retire before replacement
  }
  Advance(view.content_generation);
  view.discarded = false;
  return {};
}

std::expected<Ticket, CatalogError> Catalog::BeginEvict(ExtentId extent, bool write_back) {
  ExtentRecord* record = Extent(extent);
  if (record == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  ExtentView& view = record->view;
  if (view.state != ExtentState::kResident) {
    return std::unexpected(CatalogError::kWrongState);
  }
  if (view.leases > 0 || view.registrations > 0) {
    return std::unexpected(CatalogError::kHeld);
  }
  if (write_back) {
    // Only live mutable contents are written back: invalidated contents
    // need no preserving, and other classes are restorable or pinned.
    if (view.descriptor.recovery != Recovery::kPreserve || view.discarded ||
        view.descriptor.memory_class == MemoryClass::kUnknown) {
      return std::unexpected(CatalogError::kNotEvictable);
    }
  } else if (!Evictable(view)) {
    return std::unexpected(CatalogError::kNotEvictable);
  }
  const Bucket before = BucketOf(view);
  view.state = ExtentState::kEvicting;  // no new lease from here on
  Recount(*record, before);
  record->writing_back = write_back;
  return Begin(extent, *record, Operation::kEvict);
}

std::expected<void, CatalogError> Catalog::CompleteEvict(const Ticket& ticket) {
  auto record = Current(ticket, Operation::kEvict, ExtentState::kEvicting);
  if (!record) {
    return std::unexpected(record.error());
  }
  ExtentView& view = (*record)->view;
  const Bucket before = BucketOf(view);
  view.state = ExtentState::kNonresident;
  Advance(view.backing_generation);
  const bool preserved = (*record)->writing_back;
  if (view.descriptor.recovery != Recovery::kFromArtifact && !preserved) {
    Advance(view.content_generation);  // the contents are gone
  }
  view.discarded = false;
  view.preserved = preserved;  // written back whole: a load restores them
  if (preserved) {
    view.saved_generation = view.content_generation;
  }
  (*record)->writing_back = false;
  Recount(**record, before);
  return {};
}

std::expected<void, CatalogError> Catalog::CancelEvict(const Ticket& ticket) {
  auto record = Current(ticket, Operation::kEvict, ExtentState::kEvicting);
  if (!record) {
    return std::unexpected(record.error());
  }
  const Bucket before = BucketOf((*record)->view);
  (*record)->view.state = ExtentState::kResident;
  (*record)->writing_back = false;
  Recount(**record, before);
  return {};
}

std::expected<void, CatalogError> Catalog::QuarantineEviction(const Ticket& ticket) {
  auto record = Current(ticket, Operation::kEvict, ExtentState::kEvicting);
  if (!record) {
    return std::unexpected(record.error());
  }
  const Bucket before = BucketOf((*record)->view);
  (*record)->view.state = ExtentState::kQuarantined;
  (*record)->writing_back = false;
  Recount(**record, before);
  return {};
}

std::expected<ExtentView, CatalogError> Catalog::Describe(ExtentId extent) const {
  const ExtentRecord* record = Extent(extent);
  if (record == nullptr) {
    return std::unexpected(CatalogError::kUnknownId);
  }
  return record->view;
}

std::vector<std::pair<ExtentId, ExtentView>> Catalog::ExtentsOf(DomainId domain) const {
  std::vector<std::pair<ExtentId, ExtentView>> out;
  extents_.ForEach([&](ExtentId id, const ExtentRecord& record) {
    if (record.view.descriptor.domain == domain) {
      out.emplace_back(id, record.view);
    }
  });
  return out;
}

}  // namespace llmp::catalog
