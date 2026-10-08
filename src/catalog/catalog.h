// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The node's resource catalog (D-006, D-007; docs/architecture.md#data-model-4):
// every managed backing extent, the logical resources that live in byte
// ranges of them, residency leases, registrations and generations, and what
// each memory domain holds by class and state (the occupancy ledger's
// facts). The scheduler thread is its only writer (D-048); it is not
// thread-safe and never blocks.
//
// An extent is the unit of backing: independently reclaimable, charged
// once however many resources or leases name it (invariant 5). Its state:
//
//   NONRESIDENT -> LOADING -> RESIDENT (held or not) -> EVICTING -> NONRESIDENT
//                  LOADING -> NONRESIDENT (a load that failed with known completion)
//   LOADING, EVICTING -> QUARANTINED (completion unknown; stays charged)
//   EVICTING -> RESIDENT (a cancelled eviction)
//   RESIDENT (pinned, unheld) -> NONRESIDENT (its owner released it)
//   RESIDENT (discarded, unheld) -> RESIDENT (its owner's initial contents
//     taken back with no load: ReviveDiscarded)
//
// A lease protects a closure of resident extents at the content
// generations the closure recorded, while consumers run; releasing it makes
// them eligible for reclaim without touching their contents (D-007).
// Leasing and starting an eviction exclude each other (invariant 6).
// Registrations (a buffer registered with a device or NIC) hold an extent
// the same way until they are retired.
//
// Each load or eviction is one operation with a ticket naming it; only that
// ticket can complete, fail, cancel or quarantine it, so a late completion
// of an older operation never touches a newer one (invariants 2 and 3).
// Several requests for one load are coalesced by the resource service that
// owns the operation and its bounded waiter list (D-048); the catalog hands
// out one ticket per operation. The backing generation advances whenever an
// extent's backing is released or reassigned; the content generation
// advances when its contents are replaced or deliberately invalidated
// (invariant 4).
//
// The catalog names no model architecture and no vendor type (D-026).

#ifndef LLMP_CATALOG_CATALOG_H_
#define LLMP_CATALOG_CATALOG_H_

#include <array>
#include <cstdint>
#include <expected>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/ids.h"

namespace llmp::catalog {

using base::Bytes;

struct DomainTag {
  static constexpr const char* kName = "domain";
};
struct ExtentTag {
  static constexpr const char* kName = "extent";
};
struct ResourceTag {
  static constexpr const char* kName = "resource";
};
struct LeaseTag {
  static constexpr const char* kName = "lease";
};
struct RegistrationTag {
  static constexpr const char* kName = "registration";
};

// A memory domain (D-004, D-026): one physical budget. A Spark is one;
// two Sparks are two; a discrete GPU would be another.
using DomainId = base::Id<DomainTag>;
using ExtentId = base::Id<ExtentTag>;
using ResourceId = base::Id<ResourceTag>;
using LeaseId = base::Id<LeaseTag>;
using RegistrationId = base::Id<RegistrationTag>;

// What an extent holds (docs/architecture.md#memory-classes-5).
enum class MemoryClass : std::uint8_t {
  kWeights,        // immutable weights, restorable from their artifact
  kRoutedExperts,  // routed expert weights, acquired per selected closure
  kLiveState,      // KV, recurrent or compressed state of admitted requests
  kRetainedState,  // reusable completed-prefix or continuation state (D-055)
  kScratch,        // workspace and activations
  kRuntime,        // graph and runtime objects, kernel code
  kCommunication,  // registered communication buffers
  kStaging,        // bounded transfer staging
  kUnknown,        // an allocation the catalog cannot classify: always pinned
};
inline constexpr std::size_t kMemoryClassCount = 9;

// How an extent's contents can be recovered once its backing is released:
// a property of the extent, fixed when it is added.
enum class Recovery : std::uint8_t {
  kFromArtifact,  // clean: reload the same bytes from the artifact
  kDiscardable,   // contents may be dropped (scratch)
  kPreserve,      // mutable contents that must be written back, or
                  // explicitly invalidated, before release
  kPinned,        // never evictable: unknown allocations and runtime
                  // objects the catalog cannot recreate; only their owner
                  // releases them
};

enum class ExtentState : std::uint8_t {
  kNonresident,
  kLoading,
  kResident,
  kEvicting,
  kQuarantined,
};

// Where artifact content comes from: the artifact (SHA-256 of its manifest,
// D-056), a dependency group and a group-relative 2 MiB chunk. Non-artifact
// extents leave it zero. It orders ties in victim selection, so replays
// are deterministic (docs/architecture.md#victim-selection-initial-baseline).
struct ContentKey {
  std::array<std::uint8_t, 32> artifact{};
  std::uint32_t group = 0;
  std::uint32_t chunk = 0;
  auto operator<=>(const ContentKey&) const = default;
};

struct ExtentDescriptor {
  DomainId domain;
  MemoryClass memory_class = MemoryClass::kUnknown;
  Recovery recovery = Recovery::kPinned;
  // The physical bytes the backing occupies: the provider's allocation,
  // not a tensor's logical size.
  Bytes size;
  ContentKey content;
};

// Occupancy of one domain, by bucket and by class: backing that exists,
// counted once per extent. Nonresident extents occupy nothing.
struct Occupancy {
  Bytes held;    // resident, with a lease or a registration
  Bytes idle;    // resident and unheld (reclaimable if evictable)
  Bytes pinned;  // resident and never evictable (unknown, runtime)
  Bytes loading;
  Bytes evicting;
  Bytes quarantined;
  std::array<Bytes, kMemoryClassCount> by_class{};

  // Everything above: what the domain's backing occupies.
  Bytes Total() const;
};

enum class CatalogError : std::uint8_t {
  kUnknownDomain,
  kUnknownId,     // stale or invalid identity
  kWrongState,    // the extent is not in the state the transition needs
  kStaleTicket,   // a ticket of an earlier or different operation
  kStaleContent,  // a closure recorded contents that have since changed
  kHeld,          // leased or registered
  kNotEvictable,  // pinned, or mutable contents neither preserved nor invalidated
  kNotResident,   // a lease needs every extent resident
  kBadRange,      // a range outside its extent or domain, or a bad descriptor
  kOverBudget,    // the domain's occupancy would exceed its budget
  kExhausted,     // an identity table is full
};

std::string ToString(CatalogError error);

enum class Operation : std::uint8_t { kLoad, kEvict };

// A started load or eviction: the only handle that can finish it.
struct Ticket {
  ExtentId extent;
  Operation operation = Operation::kLoad;
  std::uint64_t serial = 0;  // the extent's operation counter when it began
};

struct ExtentView {
  ExtentDescriptor descriptor;
  ExtentState state = ExtentState::kNonresident;
  std::uint64_t backing_generation = 1;
  std::uint64_t content_generation = 1;
  // The current contents were deliberately invalidated: the extent may be
  // released without preserving them. Cleared when new contents arrive.
  bool discarded = false;
  // A nonresident kPreserve extent whose current contents a write-back
  // eviction preserved whole (BeginEvict with write-back): a load restores
  // them at the same content generation. Cleared when a load completes,
  // since the resident copy is then the live one.
  bool preserved = false;
  // The content generation a write-back eviction last saved at its place
  // (0: none). While it equals content_generation, the place holds these
  // contents unless a job wrote them in place since they were restored
  // (which only their owner knows): an eviction its owner declares
  // unchanged then writes nothing (scheduler::EvictOptions::unchanged).
  std::uint64_t saved_generation = 0;
  std::uint32_t leases = 0;
  std::uint32_t registrations = 0;
  // The last *actual* use (a consumer ran against it), as a logical tick;
  // 0 when never used. Prefetch does not count (D-055).
  std::uint64_t last_use = 0;
};

// A logical resource's placement: a byte range of one extent.
struct Range {
  ExtentId extent;
  Bytes offset;
  Bytes length;
};

// The unique extents a set of resources depends on, the content generation
// each held when the closure was taken, and their physical bytes counted
// once per domain.
struct Closure {
  std::vector<std::pair<ExtentId, std::uint64_t>> extents;  // sorted by extent, unique
  std::map<DomainId, Bytes> bytes_by_domain;
};

class Catalog {
 public:
  Catalog() = default;

  DomainId AddDomain(std::string name);
  // The domain's occupancy; empty for an unknown domain.
  Occupancy OccupancyOf(DomainId domain) const;

  // Registers backing that does not exist yet (NONRESIDENT), or, with
  // `resident`, backing that already does, such as an allocation the
  // catalog must count but cannot classify. An unknown allocation must be
  // pinned. Sizes come from untrusted manifests: a descriptor out of range,
  // or one that would overflow the domain's total, is refused.
  std::expected<ExtentId, CatalogError> AddExtent(const ExtentDescriptor& descriptor,
                                                  bool resident = false);
  // Forgets a NONRESIDENT extent that no resource uses.
  std::expected<void, CatalogError> RemoveExtent(ExtentId extent);
  // RESIDENT -> NONRESIDENT for a pinned extent whose owner has freed it,
  // once nothing holds it. The owner proves its retirement.
  std::expected<void, CatalogError> ReleasePinned(ExtentId extent);
  // Preflight for removing an owner's pinned allocation: no logical
  // resource, lease or registration may still name it. Does not mutate.
  std::expected<void, CatalogError> CanRetirePinned(ExtentId extent) const;

  std::expected<ResourceId, CatalogError> AddResource(std::span<const Range> ranges);
  std::expected<void, CatalogError> RemoveResource(ResourceId resource);

  // The closure of some resources, or of extents directly.
  std::expected<Closure, CatalogError> ClosureOf(std::span<const ResourceId> resources) const;
  std::expected<Closure, CatalogError> ClosureOfExtents(std::span<const ExtentId> extents) const;

  // NONRESIDENT -> LOADING: a new load operation, refused if the domain's
  // occupancy would exceed `budget`, its execution budget B: every
  // materialization checks actual occupancy against B (D-050). Extents
  // still evicting count until their eviction completes. A second request
  // for a load in progress joins it through the resource service, not here.
  std::expected<Ticket, CatalogError> BeginLoad(ExtentId extent, Bytes budget);
  // LOADING -> RESIDENT: the contents of the current content generation
  // are in place and verified.
  std::expected<void, CatalogError> CompleteLoad(const Ticket& ticket);
  // A load that ended without contents. If the provider proved no further
  // access, the backing is released (NONRESIDENT); otherwise the extent is
  // QUARANTINED and stays charged (D-048).
  std::expected<void, CatalogError> FailLoad(const Ticket& ticket, bool completion_known);

  // Leases every extent of a closure, all or none: each must be RESIDENT
  // with the contents the closure recorded.
  std::expected<LeaseId, CatalogError> AcquireLease(const Closure& closure);
  // Ends a lease; its extents stay resident (release is not eviction).
  std::expected<void, CatalogError> ReleaseLease(LeaseId lease);
  // Records an actual use of every extent a lease covers, at tick.
  std::expected<void, CatalogError> RecordUse(LeaseId lease, std::uint64_t tick);

  // Refused for discarded contents (kStaleContent): nothing may write them.
  std::expected<RegistrationId, CatalogError> AddRegistration(ExtentId extent);
  // Called only once the registration's retirement is confirmed.
  std::expected<void, CatalogError> RetireRegistration(RegistrationId registration);

  // Deliberately invalidates an unheld extent's mutable contents: the
  // content generation advances and the extent may be released without
  // preserving them. Artifact contents are immutable (evict them instead),
  // and pinned ones belong to their owner.
  std::expected<void, CatalogError> InvalidateContents(ExtentId extent);
  // Discards the saved contents of an unheld NONRESIDENT kPreserve extent.
  // Its next materialization starts a new content generation. An owner
  // replaces its external source before making it readable again.
  std::expected<void, CatalogError> ForgetPreserved(ExtentId extent);
  // Its owner takes back an unheld RESIDENT kPreserve extent it invalidated
  // after filling it with its initial contents (a cleared conversation's
  // zeroed state, kept for reuse): since a discarded extent can be neither
  // leased nor registered, nothing has written it since. The contents become current
  // again at a new content generation, so no closure taken before can
  // lease them.
  std::expected<void, CatalogError> ReviveDiscarded(ExtentId extent);
  // A writer, holding the sole lease and with no registrations live,
  // replaced a kPreserve extent's contents in place: the content generation
  // advances, so closures taken before cannot lease the new contents. The
  // caller establishes exclusive access before writing. Artifact and pinned contents are never
  // replaced this way. A reload of a kPreserve extent restores its contents
  // only if a write-back eviction preserved them (ExtentView::preserved);
  // otherwise it is fresh backing for new contents at a new generation.
  std::expected<void, CatalogError> ReplaceContents(ExtentId extent);

  // RESIDENT -> EVICTING: a new eviction operation, which excludes new
  // leases. The extent must be unheld and evictable. With `write_back`,
  // a kPreserve extent whose contents were not invalidated may be evicted
  // too: its owner writes the contents back before releasing the backing,
  // and a completed eviction then keeps its content generation and marks
  // it preserved (invariant 4). An abandoned write-back is CancelEvict.
  std::expected<Ticket, CatalogError> BeginEvict(ExtentId extent, bool write_back = false);
  // EVICTING -> NONRESIDENT once every consumer and registration is gone
  // and the backing is released or handed off; the backing generation
  // advances. Mutable contents are gone (the content generation advances)
  // unless the eviction wrote them back.
  std::expected<void, CatalogError> CompleteEvict(const Ticket& ticket);
  // EVICTING -> RESIDENT: the eviction was abandoned before the backing
  // was touched.
  std::expected<void, CatalogError> CancelEvict(const Ticket& ticket);
  // An eviction whose release did not complete knowably.
  std::expected<void, CatalogError> QuarantineEviction(const Ticket& ticket);

  std::expected<ExtentView, CatalogError> Describe(ExtentId extent) const;
  // Every extent of a domain, in identity order.
  std::vector<std::pair<ExtentId, ExtentView>> ExtentsOf(DomainId domain) const;

  // Whether an extent could start eviction now.
  static bool Evictable(const ExtentView& extent);

 private:
  struct Domain {
    std::string name;
    Occupancy occupancy;
    // Every extent's size, resident or not: it bounds every occupancy sum
    // and closure total in the domain, so none of them can overflow.
    Bytes registered;
  };
  struct ExtentRecord {
    ExtentView view;
    std::uint32_t resources = 0;             // resources placed in it
    std::uint64_t operations = 0;            // loads and evictions begun
    Operation operation = Operation::kLoad;  // the kind of the latest
    bool writing_back = false;               // the eviction in progress preserves contents
  };
  struct Resource {
    std::vector<Range> ranges;
  };
  struct Lease {
    std::vector<ExtentId> extents;
  };
  struct Registration {
    ExtentId extent;
  };

  // Moves an extent's bytes between occupancy buckets as its state or
  // holds change; the only place occupancy changes.
  enum class Bucket : std::uint8_t {
    kNone,
    kHeld,
    kIdle,
    kPinned,
    kLoading,
    kEvicting,
    kQuarantined
  };
  static Bucket BucketOf(const ExtentView& view);
  void Recount(ExtentRecord& record, Bucket before);
  // The extent a ticket's operation is still in progress on, or the error.
  std::expected<ExtentRecord*, CatalogError> Current(const Ticket& ticket, Operation operation,
                                                     ExtentState state);
  static Ticket Begin(ExtentId id, ExtentRecord& record, Operation operation);

  ExtentRecord* Extent(ExtentId id) { return extents_.Find(id); }
  const ExtentRecord* Extent(ExtentId id) const { return extents_.Find(id); }

  base::SlotTable<DomainTag, Domain> domains_;
  base::SlotTable<ExtentTag, ExtentRecord> extents_;
  base::SlotTable<ResourceTag, Resource> resources_;
  base::SlotTable<LeaseTag, Lease> leases_;
  base::SlotTable<RegistrationTag, Registration> registrations_;
};

}  // namespace llmp::catalog

#endif  // LLMP_CATALOG_CATALOG_H_
