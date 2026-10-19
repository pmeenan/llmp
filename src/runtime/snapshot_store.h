// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_RUNTIME_SNAPSHOT_STORE_H_
#define LLMP_RUNTIME_SNAPSHOT_STORE_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "base/sha256.h"
#include "engine/live_state.h"
#include "execution/adaptive_depth.h"
#include "runtime/score_rows.h"

namespace llmp::engine {
class PagedNode;
class StagingLifetime;
}  // namespace llmp::engine

namespace llmp::runtime {
class Llm;
class SnapshotStore;
struct SnapshotControl;

// An unforgeable reference to one completed capture. It owns neither the
// issuer nor the store. No writable view of captured bytes is public.
class StateSnapshot {
 public:
  StateSnapshot() = default;
  StateSnapshot(StateSnapshot&&) noexcept = default;
  StateSnapshot& operator=(StateSnapshot&&) noexcept = default;
  StateSnapshot(const StateSnapshot&) = delete;
  StateSnapshot& operator=(const StateSnapshot&) = delete;

 private:
  friend class SnapshotStore;
  friend class Llm;
  ScoreRows score_rows_;  // request publication ownership, never a native payload copy
  std::weak_ptr<SnapshotControl> control_;
  std::uint64_t generation_ = 0;
};

// One bounded, catalog-charged pinned allocation, driver-thread only. Its
// payload, history, ranges, layout and decoding metadata share that allocation.
// There is no token-pool borrow or second state payload. A failed copy retains
// the complete allocation and its charge through PagedNode::KeepPinned and
// permanently disables this store. A failed release also disables it, retaining
// only any allocation the node still owns. Normal retirement precedes node teardown;
// a late store destructor observes the expired node token and never calls it.
class SnapshotStore {
 public:
  // Optional driver-owned reclaim runs once only after an allocation refusal.
  // It may release eligible storage; it must not reenter this store or retire
  // the node, store or active issuer.
  using Reclaim = std::function<bool(std::uint64_t)>;
  SnapshotStore(engine::PagedNode& node, std::uint64_t ceiling, Reclaim reclaim = {});
  ~SnapshotStore();
  SnapshotStore(const SnapshotStore&) = delete;
  SnapshotStore& operator=(const SnapshotStore&) = delete;
  SnapshotStore(SnapshotStore&&) = delete;
  SnapshotStore& operator=(SnapshotStore&&) = delete;

  // Hash only captured initialized state, after authenticating store,
  // generation, live issuer and completed copy. Never exposes its address.
  std::expected<base::Sha256Digest, std::string> Digest(const StateSnapshot& snapshot) const;
  std::uint64_t capacity() const;
  bool usable() const;

 private:
  friend class Llm;
  struct CaptureView {
    void* payload = nullptr;
    std::span<const engine::LiveState::Range> ranges;
    std::span<const std::int32_t> history;
    std::string_view layout;
    std::uint32_t cursor = 0;
    execution::AdaptiveDepth decoding;
  };
  bool NodeAlive() const;
  std::expected<void, std::string> CheckNode(const engine::PagedNode* node) const;
  std::expected<void, std::string> Prepare(std::weak_ptr<const std::byte> issuer,
                                           std::optional<base::Sha256Digest> execution_identity,
                                           std::string_view layout, std::uint32_t cursor,
                                           std::span<const engine::LiveState::Range> ranges,
                                           std::span<const std::int32_t> history,
                                           const execution::AdaptiveDepth& decoding);
  void* WriteAddress() const;
  StateSnapshot Publish();
  void Unproven();
  std::expected<CaptureView, std::string> Read(const StateSnapshot& snapshot,
                                               const std::shared_ptr<const std::byte>& issuer,
                                               std::optional<base::Sha256Digest> execution_identity,
                                               std::string_view layout) const;
  std::expected<void, std::string> Authenticate(const StateSnapshot& snapshot) const;

  engine::PagedNode* node_;
  const std::weak_ptr<const engine::StagingLifetime> node_lifetime_;
  const std::uint64_t ceiling_;
  const Reclaim reclaim_;
  std::shared_ptr<SnapshotControl> control_;
};

}  // namespace llmp::runtime

#endif  // LLMP_RUNTIME_SNAPSHOT_STORE_H_
