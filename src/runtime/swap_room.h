// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Room for a swap's incoming model (Server::MakeRoomForSwap): what the
// incoming model pages in beyond what the outgoing one gives back and the
// budget's free room is reclaimed first, and the room checked again after
// each reclaim, because what a reclaim reports freed need not be what the
// budget's occupancy dropped by (the plans' and graphs' charge moves in
// whole extents past its floor); it stops once a round makes no progress
// (RoomProgress), never after a count of rounds; and before reclaiming it
// waits for an earlier swap's releases still in flight (WaitForReleaseTail).
// Vendor-free and
// host-only, so the CPU tests drive it with a ledger of their own.

#ifndef LLMP_RUNTIME_SWAP_ROOM_H_
#define LLMP_RUNTIME_SWAP_ROOM_H_

#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "catalog/catalog.h"
#include "scheduler/eviction_receipt.h"

namespace llmp::runtime {

// The rounds of a swap's room-making go on while each makes progress: the
// bytes still short must shrink from one round's sizing to the next, so
// the rounds end however many it takes, and stop at the first that does
// not (a reclaim that reports bytes freed but moves no occupancy).
class RoomProgress {
 public:
  // After a round's sizing, `short_by` bytes short: whether another round
  // may follow (the first always may).
  bool Shrank(std::uint64_t short_by) {
    const bool shrank = !last_ || short_by < *last_;
    last_ = short_by;
    return shrank;
  }

 private:
  std::optional<std::uint64_t> last_;
};

// What an incoming extent in `state` adds to the domain's occupancy once
// it is resident: its bytes only if nothing charges it now (nonresident).
// A loading extent is charged already, and so is an evicting one until its
// release lands; a page-in of a parked one takes its own backing back with
// its charge, and one still releasing is charged again only after its
// charge has left (D-033). Counting those twice would have a swap wait for,
// or reclaim room for, bytes the occupancy already holds.
inline std::uint64_t IncomingCharge(catalog::ExtentState state, std::uint64_t bytes) {
  return state == catalog::ExtentState::kNonresident ? bytes : 0;
}

// Whether a swap short of room waits for an earlier activation's backing
// releases (its eviction receipt's tail, D-033's lazy handoff) before it
// reclaims anything: their extents count as occupancy until each release
// lands, and evicting cache for room they are about to give back would be
// eager (D-007). It waits while the tail is incomplete, the domain still has
// `evicting` bytes for it to give back (outside the incoming closure: its
// own extents' releases give it no room), and it moves; it reclaims once the
// tail is complete, nothing is left evicting (what keeps it incomplete holds
// no room the swap could use, such as a failed load's transferred backing),
// or it has stalled for the node's quiet time
// (EvictionReceipt::Snapshot::Stalled), so the wait is bounded by
// progress, not by a clock of its own (D-102).
enum class RoomWait : std::uint8_t { kReclaim, kWait };
RoomWait WaitForReleaseTail(const scheduler::EvictionReceipt::Snapshot& tail,
                            std::uint64_t evicting,
                            scheduler::EvictionReceipt::Clock::time_point now,
                            std::chrono::milliseconds quiet);

// A scheduler donor matches all three fields and charges exactly `size`.
// Missing/selected entries without a key need/give actual free bytes instead.
struct SwapBackingKey {
  catalog::DomainId domain;
  std::size_t allocation_class = 0;
  std::uint64_t size = 0;
  auto operator<=>(const SwapBackingKey&) const = default;
};
struct SwapBackingExtent {
  catalog::ExtentId extent;
  std::uint64_t bytes = 0;
  std::optional<SwapBackingKey> backing;
};

// Partitions an already selected eviction set without changing its policy.
// These extents must release, rather than park, before any incoming load.
// Accounts for initial occupancy above budget and refuses insufficient or
// malformed selections. All entries belong to this one physical domain.
std::expected<std::vector<catalog::ExtentId>, std::string> ReleaseForHandoff(
    catalog::DomainId domain, std::uint64_t occupancy, std::uint64_t budget,
    std::span<const SwapBackingExtent> missing, std::span<const SwapBackingExtent> selected);

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_SWAP_ROOM_H_
