// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/paged_node_internal.h"

namespace llmp::engine {

void PlaceCheck::Check(const sc::Scheduler& scheduler, ExtentId extent,
                       const sc::PageSource& registered) {
  const sc::PageSource* now = scheduler.SourceOf(extent);
  if (now == nullptr || !sc::SamePlace(*now, registered) || !scheduler.PlacePinned(extent)) {
    Missing(std::format("extent {}", extent.index()));
  }
}

void PlaceCheck::Missing(std::string what) {
  if (moved++ == 0) {
    first = std::move(what);
  }
}

bool HavePinned(void*& pointer, std::uint64_t bytes) {
  if (pointer != nullptr) {
    return true;
  }
  auto allocated = providers::AllocatePinned(bytes);
  if (!allocated) {
    return false;
  }
  pointer = *allocated;
  return true;
}

Status PagedNode::MapWorkspace(std::uint64_t activations, std::uint64_t pool,
                               std::uint64_t virtual_capacity, bool pageable) {
  if (scheduler_) return Error("workspace reservation must precede node start");
  if (activations_.reservation.valid() || pool_.reservation.valid())
    return Error("the workspace is mapped once");
  workspace_pageable_ = pageable;
  const std::uint64_t effective_virtual_capacity =
      virtual_capacity == 0 ? (std::uint64_t{16} << 30U) : virtual_capacity;
  if (auto r =
          MapRegion(activations_, "the activations", activations, providers::BackingKind::kDevice,
                    MemoryClass::kScratch, catalog::Recovery::kDiscardable, kShared, !pageable,
                    effective_virtual_capacity);
      !r) {
    if (effective_virtual_capacity != virtual_capacity && !activations_.reservation.valid()) {
      if (auto fallback =
              MapRegion(activations_, "the activations", activations,
                        providers::BackingKind::kDevice, MemoryClass::kScratch,
                        catalog::Recovery::kDiscardable, kShared, !pageable, virtual_capacity);
          !fallback)
        return fallback;
    } else {
      return r;
    }
  }
  if (auto r = MapRegion(pool_, "the GGML pool", pool, providers::BackingKind::kDevice,
                         MemoryClass::kScratch, catalog::Recovery::kDiscardable, kShared, !pageable,
                         effective_virtual_capacity);
      !r) {
    if (effective_virtual_capacity != virtual_capacity && !pool_.reservation.valid()) {
      if (auto fallback = MapRegion(pool_, "the GGML pool", pool, providers::BackingKind::kDevice,
                                    MemoryClass::kScratch, catalog::Recovery::kDiscardable, kShared,
                                    !pageable, virtual_capacity);
          !fallback)
        return fallback;
    } else {
      return r;
    }
  }
  return {};
}

std::vector<ExtentId> PagedNode::workspace_extents() const {
  std::vector<ExtentId> all = activations_.extents;
  all.insert(all.end(), pool_.extents.begin(), pool_.extents.end());
  return all;
}

sc::PageSource PagedNode::WorkspaceSource(const Mapped& mapped, std::size_t index) const {
  // Scratch has no contents to restore: a load maps backing at the
  // extent's own place and is done, with no read, no landing slot and no
  // copy (scheduler.h, PageSource::fresh). The place never moves.
  const std::uint64_t offset = index * kPagedExtent;
  return sc::PageSource{.read = {.fd = -1, .offset = 0, .memory = nullptr, .length = kPagedExtent},
                        .landed = true,
                        .destination = mapped.base + offset,
                        .backing = sc::BackingPlace{.reservation = mapped.reservation,
                                                    .offset = Bytes(offset),
                                                    .size = Bytes(kPagedExtent),
                                                    .allocation_class = device_class_},
                        .fresh = true};
}

Status PagedNode::RegisterWorkspaceSources(const Mapped& mapped) {
  for (std::size_t i = 0; i < mapped.extents.size(); ++i) {
    if (auto set = scheduler_->SetSource(mapped.extents[i], WorkspaceSource(mapped, i)); !set) {
      return Error(std::format("{}'s place: {}", mapped.name, sc::ToString(set.error())));
    }
  }
  if (auto pinned = scheduler_->PinPlaces(mapped.extents); !pinned) {
    return Error(std::format("pinning {}'s places: {}", mapped.name, sc::ToString(pinned.error())));
  }
  return {};
}

Status PagedNode::GrowPageable(Mapped& mapped, std::uint64_t bytes) {
  if (!scheduler_ || job_active_ || supplemental_ || torn_down_ ||
      bytes > UINT64_MAX - (kPagedExtent - 1))
    return Error("workspace growth has invalid size or an unretired operation");
  const auto rounded =
      (std::max<std::uint64_t>(bytes, 1) + kPagedExtent - 1) / kPagedExtent * kPagedExtent;
  if (rounded <= mapped.bytes) return {};
  if (!mapped.reservation.valid() || rounded > mapped.capacity_bytes)
    return Error("workspace growth exceeds its virtual reservation");
  const auto count = rounded / kPagedExtent;
  // The host metadata a growth adds (catalog, span and source entries,
  // with the vectors' transient copies), charged as GrowMapped charges it.
  if (count > (UINT64_MAX - 4096) / 1024) return Error("workspace metadata overflows");
  const std::uint64_t metadata = count * 1024 + 4096;
  if (!TryChargeHost(metadata)) return Error("workspace metadata does not fit");
  std::size_t added = 0;
  auto cataloged = RegionCatalog(
      [&]() -> Status {
        // No physical effect: each new extent is NONRESIDENT until a closure
        // that names it is materialized. An extent joins the region only
        // with its source set and its place pinned; one refused either is
        // forgotten, so no extent of the region lacks a place.
        Status failed;
        std::vector<ExtentId> placed;
        spans_.reserve(spans_.size() + static_cast<std::size_t>(count - mapped.extents.size()));
        for (std::uint64_t at = mapped.bytes; at < rounded; at += kPagedExtent) {
          const auto extent = catalog_.AddExtent({.domain = domain_,
                                                  .memory_class = MemoryClass::kScratch,
                                                  .recovery = catalog::Recovery::kDiscardable,
                                                  .size = Bytes(kPagedExtent),
                                                  .content = {}},
                                                 false);
          if (!extent) {
            failed = Error(std::format("cataloging {}", mapped.name));
            break;
          }
          const std::array one = {*extent};
          if (auto set = scheduler_->SetSource(*extent, WorkspaceSource(mapped, at / kPagedExtent));
              !set || !scheduler_->PinPlaces(one)) {
            base::Check(catalog_.RemoveExtent(*extent).has_value(), "an unplaced workspace extent");
            failed = Error(std::format("{}'s place: {}", mapped.name,
                                       set ? "not pinned" : sc::ToString(set.error())));
            break;
          }
          mapped.extents.push_back(*extent);
          AddSpan({.base = mapped.base + at,
                   .size = kPagedExtent,
                   .extent = *extent,
                   .memory_class = MemoryClass::kScratch,
                   .device = true,
                   .owner = kShared});
          mapped.bytes = at + kPagedExtent;
          ++added;
        }
        SortSpans();
        return failed;
      },
      "cataloging workspace growth");
  if (added == 0) {
    UnchargeHost(metadata);  // nothing grew: the region's charge is as it was
  } else {
    UnchargeHost(mapped.metadata_bytes);
    mapped.metadata_bytes = metadata;
  }
  return cataloged;
}

Status PagedNode::GrowWorkspace(std::uint64_t activations, std::uint64_t pool) {
  if (workspace_pageable_) {
    if (auto r = GrowPageable(activations_, activations); !r) return r;
    return GrowPageable(pool_, pool);
  }
  if (auto r = GrowMapped(activations_, activations, providers::BackingKind::kDevice,
                          MemoryClass::kScratch, catalog::Recovery::kDiscardable, kShared);
      !r)
    return r;
  // A refused second growth leaves a usable larger first prefix. It is retained
  // node-wide capacity, not an allocation owned by the failed image request.
  return GrowMapped(pool_, pool, providers::BackingKind::kDevice, MemoryClass::kScratch,
                    catalog::Recovery::kDiscardable, kShared);
}

Status PagedNode::MapResident(Mapped& mapped, std::string name, std::uint64_t bytes,
                              providers::BackingKind kind, MemoryClass memory_class,
                              catalog::Recovery recovery, int owner, bool* over_budget) {
  if (over_budget) *over_budget = false;
  if (scheduler_) {
    if (mapped.reservation.valid()) return Error("an owned region is already mapped");
    mapped.name = std::move(name);
    return GrowMapped(mapped, bytes, kind, memory_class, recovery, owner, over_budget);
  }
  return MapRegion(mapped, std::move(name), bytes, kind, memory_class, recovery, owner, true);
}

Status PagedNode::ReserveState(Mapped& mapped, std::string name, std::uint64_t bytes, int owner) {
  if (threads_.empty()) {
    // Before Run (or lanes driven on this thread): nothing else calls the
    // provider or touches the catalog meanwhile.
    return MapRegion(mapped, std::move(name), bytes, providers::BackingKind::kDevice,
                     MemoryClass::kLiveState, catalog::Recovery::kPreserve, owner, false);
  }
  // After Run the VMM provider is the backing lane's alone (its handle
  // reserve refills in the background) and the catalog the scheduler's: the
  // reservation goes to the lane, the extents to the scheduler's thread.
  // Never from inside a Call: both wait for the scheduler.
  if (mapped.reservation.valid()) return Error("a mapped region cannot be overwritten");
  if (job_active_ || supplemental_ || torn_down_ ||
      bytes > std::numeric_limits<std::uint64_t>::max() - kPagedExtent + 1)
    return Error("state reservation has invalid size or an unretired operation");
  const std::uint64_t rounded =
      (std::max<std::uint64_t>(bytes, 1) + kPagedExtent - 1) / kPagedExtent * kPagedExtent;
  const std::uint64_t count = rounded / kPagedExtent;
  // The host metadata it adds (catalog and span entries), as GrowPageable
  // charges it.
  if (count > (UINT64_MAX - 4096) / 1024) return Error("state reservation metadata overflows");
  const std::uint64_t metadata = count * 1024 + 4096;
  if (!TryChargeHost(metadata)) return Error("state reservation metadata does not fit");
  auto result = std::make_shared<sc::RegionResult>();
  if (auto reserved = Region({.kind = sc::RegionWork::Kind::kReserve,
                              .size = Bytes(rounded),
                              .extent_size = Bytes(kPagedExtent),
                              .offset = {},
                              .allocation_class = 0,
                              .reservation = {},
                              .result = result},
                             "reserving a state region");
      !reserved) {
    UnchargeHost(metadata);
    return reserved;
  }
  const std::uint64_t at_base = result->base;
  std::vector<ExtentId> extents;
  extents.reserve(static_cast<std::size_t>(count));
  auto cataloged = RegionCatalog(
      [&]() -> Status {
        // No physical effect: each extent is NONRESIDENT until its page
        // source materializes it, as MapRegion catalogs one before Run.
        spans_.reserve(spans_.size() + static_cast<std::size_t>(count));
        for (std::uint64_t at = 0; at < rounded; at += kPagedExtent) {
          const auto extent = catalog_.AddExtent({.domain = domain_,
                                                  .memory_class = MemoryClass::kLiveState,
                                                  .recovery = catalog::Recovery::kPreserve,
                                                  .size = Bytes(kPagedExtent),
                                                  .content = {}},
                                                 false);
          if (!extent) {
            for (const ExtentId added : extents)
              base::Check(catalog_.RemoveExtent(added).has_value(), "an unpublished state extent");
            extents.clear();
            return Error(std::format("cataloging {}", name));
          }
          extents.push_back(*extent);
        }
        for (std::size_t i = 0; i < extents.size(); ++i) {
          AddSpan({.base = at_base + (i * kPagedExtent),
                   .size = kPagedExtent,
                   .extent = extents[i],
                   .memory_class = MemoryClass::kLiveState,
                   .device = true,
                   .owner = owner});
        }
        SortSpans();
        return {};
      },
      "cataloging a state region");
  if (!cataloged) {
    // Nothing of it is cataloged: its empty reservation goes back.
    if (auto freed = Region({.kind = sc::RegionWork::Kind::kFree,
                             .size = {},
                             .extent_size = Bytes(kPagedExtent),
                             .offset = {},
                             .allocation_class = 0,
                             .reservation = result->reservation,
                             .result = result},
                            "freeing an uncataloged state reservation");
        !freed) {
      // Kept owned by the region, released with it at teardown.
      mapped.name = std::move(name);
      mapped.reservation = result->reservation;
      mapped.base = at_base;
      mapped.metadata_bytes = metadata;
      return freed;
    }
    UnchargeHost(metadata);
    return cataloged;
  }
  mapped.name = std::move(name);
  mapped.reservation = result->reservation;
  mapped.base = at_base;
  mapped.bytes = mapped.capacity_bytes = rounded;
  mapped.extents = std::move(extents);
  mapped.metadata_bytes = metadata;
  return {};
}

Status PagedNode::MapRegion(Mapped& mapped, std::string name, std::uint64_t bytes,
                            providers::BackingKind kind, MemoryClass memory_class,
                            catalog::Recovery recovery, int owner, bool resident,
                            std::uint64_t virtual_capacity) {
  if (mapped.reservation.valid()) return Error("a mapped region cannot be overwritten");
  // It calls the provider and the catalog from this thread: after Run they
  // are the backing lane's and the scheduler's (ReserveState, GrowMapped).
  if (!threads_.empty()) return Error("a region is mapped directly only before Run");
  if (bytes > std::numeric_limits<std::uint64_t>::max() - kPagedExtent + 1) {
    return Error("a region's size overflows its extent alignment");
  }
  mapped.name = std::move(name);
  mapped.bytes =
      (std::max<std::uint64_t>(bytes, 1) + kPagedExtent - 1) / kPagedExtent * kPagedExtent;
  if (virtual_capacity > UINT64_MAX - (kPagedExtent - 1))
    return Error("virtual reservation alignment overflows");
  mapped.capacity_bytes =
      std::max(mapped.bytes, (virtual_capacity + kPagedExtent - 1) / kPagedExtent * kPagedExtent);
  auto reservation = memory_->Reserve(Bytes(mapped.capacity_bytes));
  if (!reservation) {
    return Error(std::format("reserving {}: {}", mapped.name, reservation.error().detail));
  }
  mapped.reservation = *reservation;
  mapped.base = memory_->RangeOf(*reservation).value().base;
  const std::size_t allocation_class =
      kind == providers::BackingKind::kDevice ? device_class_ : host_class_;
  mapped.allocation_class = allocation_class;
  for (std::uint64_t at = 0; at < mapped.bytes; at += kPagedExtent) {
    if (resident) {
      auto backing = memory_->Create(allocation_class, Bytes(kPagedExtent));
      if (!backing) {
        return Error(std::format("backing {}: {}", mapped.name, backing.error().detail));
      }
      mapped.backings.push_back(*backing);
      if (auto map = memory_->Map(*reservation, Bytes(at), *backing); !map) {
        return Error(std::format("mapping {}: {}", mapped.name, map.error().detail));
      }
    }
    auto extent = catalog_.AddExtent({.domain = domain_,
                                      .memory_class = memory_class,
                                      .recovery = recovery,
                                      .size = Bytes(kPagedExtent),
                                      .content = {}},
                                     resident);
    if (!extent) {
      return Error(std::format("cataloging {}", mapped.name));
    }
    mapped.extents.push_back(*extent);
    AddSpan({.base = mapped.base + at,
             .size = kPagedExtent,
             .extent = *extent,
             .memory_class = memory_class,
             .device = kind == providers::BackingKind::kDevice,
             .owner = owner});
  }
  if (resident) {
    if (auto access = memory_->SetAccess(*reservation, Bytes(0), Bytes(mapped.bytes),
                                         providers::Access::kReadWrite);
        !access) {
      return Error(std::format("access to {}: {}", mapped.name, access.error().detail));
    }
  }
  return {};
}

Status PagedNode::Region(sc::RegionWork work, std::string_view what) {
  class RegionProgram final : public sc::ReportingProgram {
   public:
    RegionProgram(Done& done, sc::RegionWork work)
        : ReportingProgram(done), work_(std::move(work)) {}
    sc::Step Advance(sc::TaskContext& context) override {
      if (context.TakeFailure()) return sc::Step::Finish(sc::TaskOutcome::kFailed);
      if (submitted_) return sc::Step::Finish(sc::TaskOutcome::kSucceeded);
      const auto submitted = context.SubmitRegion({}, work_);
      if (!submitted) {
        if (submitted.error() == sc::WorkError::kBusy) return sc::Step::Yield();
        return Fail(submitted.error());
      }
      submitted_ = true;
      return sc::Step::Wait();
    }

   private:
    sc::RegionWork work_;
    bool submitted_ = false;
  };
  if (!scheduler_ || job_active_ || supplemental_ || torn_down_)
    return Error("a region transition requires a live idle node");
  Done done;
  auto result = work.result;
  // Reset before submission, including cancellation that never reaches a lane.
  result->outcome = sc::RegionResult::Outcome::kUntouched;
  result->failure.reset();
  auto ran = Post(std::make_unique<RegionProgram>(done, std::move(work)), done, what);
  // Task cancellation can follow positive provider completion. Its effects
  // still belong to this caller and must be published/retired, never refunded.
  if (result->outcome == sc::RegionResult::Outcome::kComplete) return {};
  base::Check(result->outcome != sc::RegionResult::Outcome::kUnknown,
              "unknown region completion cannot return to its owner");
  if (result->failure) return Error(std::format("{}: {}", what, result->failure->detail));
  return ran ? Error("region task retired without completed provider work") : ran;
}

Status PagedNode::RegionCatalog(std::function<Status()> mutation, std::string_view what) {
  std::optional<Status> completed;
  auto task = Call(
      [&]() -> Status {
        completed = mutation();
        return *completed;
      },
      what);
  // The completed transaction is authoritative even if its task is cancelled
  // afterwards. If it never ran, no catalog effects were made by this call.
  return completed ? std::move(*completed) : task;
}

Status PagedNode::GrowMapped(Mapped& mapped, std::uint64_t bytes, providers::BackingKind kind,
                             MemoryClass memory_class, catalog::Recovery recovery, int owner,
                             bool* over_budget) {
  if (over_budget) *over_budget = false;
  if (!scheduler_ || job_active_ || supplemental_ || torn_down_ ||
      bytes > UINT64_MAX - (kPagedExtent - 1))
    return Error("owned-region growth has invalid size or an unretired operation");
  const auto rounded =
      (std::max<std::uint64_t>(bytes, 1) + kPagedExtent - 1) / kPagedExtent * kPagedExtent;
  if (rounded <= mapped.bytes) return {};
  if (mapped.reservation.valid() && rounded > mapped.capacity_bytes)
    return Error("owned-region growth exceeds its virtual reservation");
  const auto added = rounded - mapped.bytes;
  const auto count = rounded / kPagedExtent;
  // Per extent: both old/new vector storage, provider and catalog descriptors,
  // span and transition handles. This charge precedes vector growth and is
  // retained with the mapping. The old metadata charge remains during copying.
  static_assert(sizeof(catalog::ExtentView) + sizeof(Span) + 4 * sizeof(providers::BackingId) +
                    4 * sizeof(ExtentId) <=
                512);
  if (count > (UINT64_MAX - 4096) / 1024) return Error("owned-region metadata overflows");
  std::uint64_t catalog_storage = 0;
  if (auto read = RegionCatalog(
          [&]() -> Status {
            catalog_storage = catalog_.ExtentStorageBytes();
            return {};
          },
          "measuring owned-region metadata");
      !read)
    return read;
  // Appending one extent may reallocate the whole existing catalog/span
  // vectors. Their capacity survives geometry retirement, so retain a separate
  // node-wide high-water charge, including transient old/new vector storage.
  const std::uint64_t metadata = count * 1024 + 4096;
  std::uint64_t retained = 0, charge = 0;
  if (__builtin_add_overflow(catalog_storage, std::uint64_t{spans_.capacity()} * sizeof(Span),
                             &retained) ||
      __builtin_mul_overflow(retained, std::uint64_t{2}, &retained) ||
      __builtin_add_overflow(retained, metadata, &retained))
    return Error("owned-region retained metadata copy overflows");
  const auto growth = retained > region_metadata_bytes_ ? retained - region_metadata_bytes_ : 0;
  if (__builtin_add_overflow(metadata, growth, &charge))
    return Error("owned-region metadata charge overflows");
  if (!TryChargeHost(charge)) {
    if (over_budget) *over_budget = true;
    return Error("owned-region metadata does not fit");
  }
  region_metadata_bytes_ += growth;
  auto result = std::make_shared<sc::RegionResult>();
  const auto new_count = static_cast<std::size_t>(added / kPagedExtent);
  result->backings.reserve(new_count);
  mapped.backings.reserve(static_cast<std::size_t>(count));
  mapped.extents.reserve(static_cast<std::size_t>(count));
  std::vector<catalog::Ticket> tickets;
  tickets.reserve(new_count);
  // Each new physical extent is cataloged LOADING before any provider effect.
  // Existing extents and their generations remain unchanged. The scheduler is
  // the only publisher; the backing lane receives only already-paid work.
  const auto cancel_tickets = [&] {
    for (auto ticket : tickets) {
      base::Check(catalog_.FailLoad(ticket, true).has_value(), "owned growth rollback ticket");
      base::Check(catalog_.RemoveExtent(ticket.extent).has_value(), "owned growth rollback extent");
    }
    tickets.clear();
  };
  auto admitted = RegionCatalog(
      [&]() -> Status {
        if (Shortfall(added) != 0) {
          if (over_budget) *over_budget = true;
          return Error("owned-region backing does not fit");
        }
        spans_.reserve(spans_.size() + new_count);
        for (std::size_t i = 0; i < new_count; ++i) {
          const auto extent = catalog_.AddExtent({.domain = domain_,
                                                  .memory_class = memory_class,
                                                  .recovery = recovery,
                                                  .size = Bytes(kPagedExtent),
                                                  .content = {}},
                                                 false);
          if (!extent) {
            cancel_tickets();
            return Error("owned-region catalog capacity");
          }
          const auto ticket = catalog_.BeginLoad(*extent, budget());
          if (!ticket) {
            base::Check(catalog_.RemoveExtent(*extent).has_value(), "unadmitted owned extent");
            cancel_tickets();
            return Error("owned-region backing admission");
          }
          tickets.push_back(*ticket);
        }
        return {};
      },
      "admitting owned-region growth");
  if (!admitted) {
    region_metadata_bytes_ -= growth;
    UnchargeHost(charge);
    return admitted;
  }
  const auto repay = [&] {
    auto canceled = RegionCatalog(
        [&]() -> Status {
          cancel_tickets();
          return {};
        },
        "undoing untouched owned-region growth");
    base::Check(canceled.has_value(), "paid owned growth rollback");
    region_metadata_bytes_ -= growth;
    UnchargeHost(charge);
  };
  const bool fresh = !mapped.reservation.valid();
  if (fresh) {
    auto reserved = Region({.kind = sc::RegionWork::Kind::kReserve,
                            .size = Bytes(rounded),
                            .extent_size = Bytes(kPagedExtent),
                            .offset = {},
                            .allocation_class = 0,
                            .reservation = {},
                            .result = result},
                           "reserving owned region");
    if (!reserved) {
      repay();
      return reserved;
    }
    mapped.reservation = result->reservation;
    mapped.base = result->base;
    mapped.capacity_bytes = rounded;
  }
  const auto allocation_class =
      kind == providers::BackingKind::kDevice ? device_class_ : host_class_;
  mapped.allocation_class = allocation_class;
  auto grown = Region({.kind = sc::RegionWork::Kind::kGrow,
                       .size = Bytes(added),
                       .extent_size = Bytes(kPagedExtent),
                       .offset = Bytes(mapped.bytes),
                       .allocation_class = allocation_class,
                       .reservation = mapped.reservation,
                       .result = result},
                      "growing owned region");
  if (!grown) {
    // A known refusal has rolled back every new handle. Unknown operations
    // cannot retire their RegionProgram and are stopped by Await's failstop.
    if (fresh) {
      auto canceled = RegionCatalog(
          [&]() -> Status {
            cancel_tickets();
            return {};
          },
          "retiring rolled-back owned growth tickets");
      base::Check(canceled.has_value(), "known rolled-back growth tickets");
      if (auto freed = Region({.kind = sc::RegionWork::Kind::kFree,
                               .size = {},
                               .extent_size = Bytes(kPagedExtent),
                               .offset = {},
                               .allocation_class = 0,
                               .reservation = mapped.reservation,
                               .result = result},
                              "unwinding owned reservation");
          !freed) {
        // No physical suffix remains; retain the reservation and local metadata
        // so candidate cleanup or terminal owner teardown can release them.
        mapped.metadata_bytes = metadata;
        return freed;
      }
      mapped.reservation = {};
      mapped.base = mapped.capacity_bytes = 0;
    }
    repay();
    return grown;
  }
  mapped.backings.insert(mapped.backings.end(), result->backings.begin(), result->backings.end());
  auto published = RegionCatalog(
      [&]() -> Status {
        std::uint64_t at = mapped.bytes;
        for (auto ticket : tickets) {
          base::Check(catalog_.CompleteLoad(ticket).has_value(), "owned growth publication");
          mapped.extents.push_back(ticket.extent);
          AddSpan({.base = mapped.base + at,
                   .size = kPagedExtent,
                   .extent = ticket.extent,
                   .memory_class = memory_class,
                   .device = kind == providers::BackingKind::kDevice,
                   .owner = owner});
          at += kPagedExtent;
        }
        SortSpans();
        return {};
      },
      "publishing owned-region growth");
  base::Check(published.has_value(), "completed owned growth must publish its paid tickets");
  mapped.bytes = rounded;
  UnchargeHost(mapped.metadata_bytes);
  mapped.metadata_bytes = metadata;
  return {};
}

Status PagedNode::RetireMapped(Mapped& mapped) {
  if (!mapped.reservation.valid()) return {};
  if (!scheduler_ || job_active_ || supplemental_ || torn_down_ ||
      mapped.backings.size() != mapped.extents.size())
    return Error("owned mapping cannot retire with pending work or managed backing");
  // Metadata for this temporary copy is held in the mapping's growth charge,
  // or its original setup construction allowance. No payload copy is made.
  auto result = std::make_shared<sc::RegionResult>();
  result->backings = mapped.backings;
  result->mapped = result->backings.size();
  std::vector<catalog::Ticket> tickets;
  tickets.reserve(mapped.extents.size());
  auto checked = RegionCatalog(
      [&]() -> Status {
        for (auto extent : mapped.extents)
          if (!catalog_.CanRetirePinned(extent)) return Error("owned mapping remains borrowed");
        for (auto extent : mapped.extents) {
          const auto ticket = catalog_.BeginRetirePinned(extent);
          if (!ticket) {
            for (auto prior : tickets) (void)catalog_.CancelEvict(prior);
            tickets.clear();
            return Error("owned mapping retirement ticket");
          }
          tickets.push_back(*ticket);
        }
        return {};
      },
      "excluding new owned-region borrowers");
  if (!checked) return checked;
  auto retired = Region({.kind = sc::RegionWork::Kind::kRetire,
                         .size = Bytes(mapped.bytes),
                         .extent_size = Bytes(kPagedExtent),
                         .offset = {},
                         .allocation_class = mapped.allocation_class,
                         .reservation = mapped.reservation,
                         .result = result},
                        "retiring owned region");
  if (!retired) {
    auto restored = RegionCatalog(
        [&]() -> Status {
          for (auto ticket : tickets) (void)catalog_.CancelEvict(ticket);
          return {};
        },
        "undoing untouched owned-region retirement");
    base::Check(restored.has_value(), "untouched retirement must restore its tickets");
    return retired;
  }
  auto forgotten = RegionCatalog(
      [&]() -> Status {
        for (auto ticket : tickets) {
          if (!catalog_.CompleteEvict(ticket) || !catalog_.RemoveExtent(ticket.extent))
            return Error("forgetting retired owned-region extent");
        }
        EraseSpans([&](const Span& span) {
          return std::ranges::find(mapped.extents, span.extent) != mapped.extents.end();
        });
        return {};
      },
      "forgetting retired owned region");
  base::Check(forgotten.has_value(), "retired backing must leave the catalog");
  mapped.backings.clear();
  mapped.extents.clear();
  mapped.bytes = 0;
  if (auto freed = Region({.kind = sc::RegionWork::Kind::kFree,
                           .size = {},
                           .extent_size = Bytes(kPagedExtent),
                           .offset = {},
                           .allocation_class = 0,
                           .reservation = mapped.reservation,
                           .result = result},
                          "freeing retired owned reservation");
      !freed)
    return freed;  // empty reservation/local charge remain owned for retry
  mapped.reservation = {};
  UnchargeHost(mapped.metadata_bytes);
  mapped = {};
  return {};
}

std::expected<void*, std::string> PagedNode::Pinned(std::uint64_t bytes, int owner,
                                                    std::vector<ExtentId>& staging,
                                                    bool* over_budget) {
  if (over_budget) *over_budget = false;
  bytes = std::max<std::uint64_t>(bytes, 256);
  if (scheduler_ != nullptr) {
    // What does not fit has the reclaim order free it first, then is
    // checked again (and again when it is registered).
    for (int attempt = 0;; ++attempt) {
      std::uint64_t shortfall = 0;
      auto checked = Call(
          [&]() -> Status {
            shortfall = Shortfall(bytes);
            return {};
          },
          "checking pinned staging capacity");
      if (!checked) {
        return std::unexpected(checked.error());
      }
      if (shortfall == 0) {
        break;
      }
      // In whole extents (what the plans' charge gives back moves in them),
      // checked again after each reclaim, a few times at most.
      const std::uint64_t ask = (shortfall + kPagedExtent - 1) / kPagedExtent * kPagedExtent;
      if (attempt >= 3 || AskReclaim(ask, ReclaimFor::kStaging) == 0) {
        if (over_budget) *over_budget = true;
        return Error("pinned staging exceeds the execution budget");
      }
    }
  }
  auto allocated = providers::AllocatePinned(bytes);
  if (!allocated) {
    return Error("pinned memory");
  }
  void* const pointer = *allocated;
  auto register_memory = [&]() -> Status {
    if (scheduler_ != nullptr) {
      const auto occupancy = catalog_.OccupancyOf(domain_).Total().value();
      if (occupancy > budget().value() || bytes > budget().value() - occupancy) {
        if (over_budget) *over_budget = true;
        return Error("pinned staging exceeds the execution budget");
      }
    }
    auto extent = catalog_.AddExtent({.domain = domain_,
                                      .memory_class = MemoryClass::kStaging,
                                      .recovery = catalog::Recovery::kPinned,
                                      .size = Bytes(bytes),
                                      .content = {}},
                                     true);
    if (!extent) {
      return Error("cataloging the staging");
    }
    staging.push_back(*extent);
    AddSpan({.base = Address(pointer),
             .size = bytes,
             .extent = *extent,
             .memory_class = MemoryClass::kStaging,
             .device = false,
             .owner = owner});
    if (scheduler_ != nullptr) {
      SortSpans();
    }
    return {};
  };
  auto registered =
      scheduler_ == nullptr ? register_memory() : RegionCatalog(register_memory, "pinned staging");
  if (!registered) {
    providers::FreePinned(pointer);
    return std::unexpected(registered.error());
  }
  pinned_.push_back(pointer);
  return pointer;
}

Status PagedNode::ReserveStaging(std::uint64_t bytes) {
  if (staging_ != nullptr) {
    return {};
  }
  std::vector<ExtentId> extents;
  auto pinned = Pinned(bytes, kShared, extents);
  if (!pinned) {
    return std::unexpected(pinned.error());
  }
  staging_ = *pinned;
  staging_bytes_ = bytes;
  return {};
}

void* PagedNode::TakeStaging(std::uint64_t bytes) {
  if (staging_ == nullptr || staging_taken_ || bytes > staging_bytes_) {
    return nullptr;
  }
  staging_taken_ = true;
  return staging_;
}

void PagedNode::ReturnStaging(void* pointer, bool proven) {
  if (pointer == nullptr || pointer != staging_) {
    return;
  }
  if (proven) {
    staging_taken_ = false;
    return;
  }
  // A copy may still reach it: out of use, in the quarantine (tagged with
  // the copy's stream when the copier put it there, else to teardown). Once
  // a fence has freed it, the reservation is made again (FenceStream).
  KeepPinned(staging_);
  staging_lost_ = staging_;
  staging_owed_ = staging_bytes_;
  staging_ = nullptr;
  staging_bytes_ = 0;
  staging_taken_ = false;
}

void PagedNode::RestoreStaging() {
  if (staging_owed_ == 0 || staging_ != nullptr) return;
  if (staging_lost_ != nullptr && std::ranges::any_of(quarantine_, [&](const Quarantined& q) {
        return q.pointer == staging_lost_;
      }))
    return;  // not freed yet: its room is not the budget's again
  staging_lost_ = nullptr;
  if (ReserveStaging(staging_owed_)) staging_owed_ = 0;  // else retried at the next fence
}

void PagedNode::SortSpans() { std::ranges::sort(spans_, {}, &Span::base); }

std::uint64_t PagedNode::StateCapacity() const {
  std::uint64_t bytes = 0;
  for (const Span& span : spans_) {
    if (span.device && span.memory_class == MemoryClass::kLiveState) {
      bytes += span.size;
    }
  }
  return bytes;
}

std::optional<MemoryClass> PagedNode::Covered(std::uint64_t address, std::uint64_t bytes, int owner,
                                              bool resident) const {
  if (bytes == 0 || bytes > std::numeric_limits<std::uint64_t>::max() - address) {
    return std::nullopt;
  }
  std::optional<MemoryClass> found;
  std::uint64_t at = address;
  const std::uint64_t end = address + bytes;
  auto span = std::ranges::upper_bound(spans_, at, {}, &Span::base);
  if (span == spans_.begin()) {
    return std::nullopt;
  }
  --span;
  while (at < end) {
    if (span == spans_.end() || at < span->base || at >= span->base + span->size || !span->device ||
        (span->owner != owner && span->owner != kShared) ||
        (found && *found != span->memory_class)) {
      return std::nullopt;
    }
    // Residency is the catalog's; the reservation alone is the spans' (a
    // span leaves with its extent), which the driver may read at any time.
    if (resident) {
      const auto view = catalog_.Describe(span->extent);
      if (!view || view->state != catalog::ExtentState::kResident) {
        return std::nullopt;
      }
    }
    found = span->memory_class;
    at = span->base + span->size;
    ++span;
  }
  return found;
}

}  // namespace llmp::engine
