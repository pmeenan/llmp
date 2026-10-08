// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The backing designs the retained-backing comparison replays
// (docs/backend-proof.md#retained-backing-comparison, D-035, D-079). These
// are CANDIDATES for a measurement, not the memory manager: they place the
// trace's groups, choose extra victims and relocations, and drive a
// device-memory provider, so that their provider calls are real calls
// counted at the provider. Nothing here reads or writes group contents.
//
//   - D-033, the baseline: one 2 MiB handle per chunk of a group, mapped
//     at the group's fixed 2 MiB-aligned region of its model's virtual
//     range; no free pool; handles freed by evictions for an access are
//     handed to that access's restores, and the rest are released when it
//     ends.
//   - Slab designs: persistently mapped slabs of 32 MiB, 256 MiB or 1 GiB
//     in one virtual arena, with software suballocation at 4 KiB, under one
//     of four hole policies (HolePolicy). A group may span adjacent slabs,
//     which the arena keeps contiguous in virtual address space.
//
// Every design evicts the reference's victims first (the replay does, with
// Drop). A design evicts more, or relocates, only when it cannot otherwise
// place a restore or meet a shrink, and then chooses its extra victims by
// its hole policy, or else the least recently used evictable group in the
// reference's recency order (Env::LruVictim).

#ifndef LLMP_BENCHMARKS_RETAINED_BACKING_DESIGNS_H_
#define LLMP_BENCHMARKS_RETAINED_BACKING_DESIGNS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "providers/device_memory.h"
#include "retained_backing/trace.h"

namespace llmp::rb {

// What the replay offers a design while it places an access's restores or
// meets a shrink.
class Env {
 public:
  Env() = default;
  Env(const Env&) = delete;
  Env& operator=(const Env&) = delete;
  Env(Env&&) = delete;
  Env& operator=(Env&&) = delete;
  virtual ~Env() = default;

  // The budget less any outstanding shrink: held backing never exceeds it.
  virtual std::uint64_t Limit() const = 0;
  virtual const Group& GroupOf(GroupId g) const = 0;
  // Resident in the design, unleased and not wanted by the current access.
  virtual bool Evictable(GroupId g) const = 0;
  // The reference's recency stamp; larger is more recent.
  virtual std::uint64_t Stamp(GroupId g) const = 0;
  // The least recently used evictable group, in the reference's order.
  virtual std::optional<GroupId> LruVictim() const = 0;
  // The design has evicted g beyond the reference's victims.
  virtual void Evicted(GroupId g) = 0;
  // The design has moved g's contents to a new place.
  virtual void Relocated(GroupId g) = 0;
};

// Provider calls by kind, and io_uring buffer (un)registrations.
enum class Call : std::uint8_t {
  kReserve,
  kFree,
  kCreate,
  kRelease,
  kMap,
  kSetAccess,
  kUnmap,
  kRegister,
  kUnregister,
  kCount
};
inline constexpr std::size_t kCalls = static_cast<std::size_t>(Call::kCount);
std::string_view CallName(Call call);
using CallCounts = std::array<std::uint64_t, kCalls>;

// A DeviceMemory that counts every call, folds each into a decision digest
// and treats any failure as fatal: a replay never expects one.
class CountingMemory final : public providers::DeviceMemory {
 public:
  explicit CountingMemory(providers::DeviceMemory& inner) : inner_(inner) {}

  std::span<const providers::AllocationClass> Classes() const override { return inner_.Classes(); }
  base::Bytes Granularity() const override { return inner_.Granularity(); }
  std::expected<providers::ReservationId, providers::Failure> Reserve(base::Bytes size) override;
  std::expected<providers::AddressRange, providers::Failure> RangeOf(
      providers::ReservationId reservation) const override {
    return inner_.RangeOf(reservation);
  }
  std::expected<void, providers::Failure> Free(providers::ReservationId reservation) override;
  std::expected<providers::BackingId, providers::Failure> Create(std::size_t allocation_class,
                                                                 base::Bytes size) override;
  std::expected<void, providers::Failure> Release(providers::BackingId backing) override;
  std::expected<void, providers::Failure> Map(providers::ReservationId reservation,
                                              base::Bytes offset,
                                              providers::BackingId backing) override;
  std::expected<void, providers::Failure> SetAccess(providers::ReservationId reservation,
                                                    base::Bytes offset, base::Bytes size,
                                                    providers::Access access) override;
  std::expected<void, providers::Failure> Unmap(providers::ReservationId reservation,
                                                base::Bytes offset, base::Bytes size) override;
  // A query, not a call the replay counts.
  std::optional<providers::BackingId> MappedAt(providers::ReservationId reservation,
                                               base::Bytes offset) const override {
    return inner_.MappedAt(reservation, offset);
  }

  // Registration is not a provider call: the design counts what it would
  // register with io_uring.
  void Count(Call call, std::uint64_t a, std::uint64_t b);

  const CallCounts& counts() const { return counts_; }
  // Stops counting (teardown after the trace is not the design's cost).
  void Freeze() { frozen_ = true; }
  void Digest(std::uint64_t a, std::uint64_t b, std::uint64_t c);
  base::Sha256& digest() { return digest_; }

 private:
  providers::DeviceMemory& inner_;
  CallCounts counts_{};
  bool frozen_ = false;
  base::Sha256 digest_;
};

// One design's placement and backing, driven by the replay.
class Design {
 public:
  Design() = default;
  Design(const Design&) = delete;
  Design& operator=(const Design&) = delete;
  Design(Design&&) = delete;
  Design& operator=(Design&&) = delete;
  virtual ~Design() = default;

  // Evicts one of the reference's victims.
  virtual void Drop(GroupId g) = 0;
  // Places an access's missing groups, evicting or relocating only if it
  // must. False is a refusal.
  virtual bool Admit(std::span<const GroupId> missing, Env& env) = 0;
  // The access is done: release whatever it no longer needs.
  virtual void EndAccess() {}
  // Brings held backing within Limit(). False if it cannot.
  virtual bool Shrink(Env& env) = 0;
  // The reference touched a resident group.
  virtual void Touched(GroupId /*g*/, std::uint64_t /*before*/, std::uint64_t /*after*/) {}
  // Backing held now, handoff included.
  virtual std::uint64_t Held() const = 0;
  // Where g's first byte is now, relative to the design's reservations
  // (stable across runs, unlike their addresses): for the digest and tests.
  virtual std::uint64_t PositionOf(GroupId g) const = 0;
  // Fatal unless the design's structures are consistent: placements
  // disjoint, 4 KiB aligned and on held backing, counts agreeing.
  virtual void CheckInvariants() const = 0;
  // Releases everything and frees the reservations (after the trace).
  virtual void Teardown() = 0;
};

enum class HolePolicy : std::uint8_t {
  kContiguousRun,  // evict the coldest contiguous run that makes a hole
  kSizeClasses,    // groups up to a slab in slabs of their exact size's class
  kHybrid,         // expert closures in size classes, dense groups by runs
  kCompaction,     // runs, after relocating unleased dense groups to open a hole
};

struct DesignSpec {
  std::string name;
  bool baseline = false;  // D-033
  std::uint64_t slab = 0;
  HolePolicy policy = HolePolicy::kContiguousRun;
};

// The 13 designs, in the criteria's order: D-033, then each slab size with
// the four policies.
std::vector<DesignSpec> AllDesigns();
std::optional<DesignSpec> FindDesign(std::string_view name);

std::unique_ptr<Design> MakeDesign(const DesignSpec& spec, std::span<const Group> groups,
                                   std::span<const Model> models, std::uint64_t budget,
                                   CountingMemory& memory);

}  // namespace llmp::rb

#endif  // LLMP_BENCHMARKS_RETAINED_BACKING_DESIGNS_H_
