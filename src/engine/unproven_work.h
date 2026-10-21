// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_UNPROVEN_WORK_H_
#define LLMP_ENGINE_UNPROVEN_WORK_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "providers/storage.h"

namespace llmp::engine {

class StepTransaction;

struct PinnedKeepAlive {
  void* pointer = nullptr;
};

struct StorageRingKeepAlive {
  providers::Storage* ring = nullptr;
};

// Unified tracking for unproven effects cleared only by a positive stream fence.
class UnprovenWork {
 public:
  UnprovenWork() = default;
  ~UnprovenWork() = default;

  bool active() const noexcept {
    return tainted_ || step_ != nullptr || routed_ != nullptr || !pinned_.empty() ||
           !storage_rings_.empty();
  }

  explicit operator bool() const noexcept { return active(); }

  void Mark() noexcept { tainted_ = true; }
  void MarkTainted() noexcept { tainted_ = true; }
  void ClearTainted() noexcept { tainted_ = false; }

  // Cleared after a proven stream fence: unproven flights and transaction witnesses
  // are dropped. Any kept pinned allocations and storage rings are preserved.
  void ClearAfterFence() noexcept {
    tainted_ = false;
    step_.reset();
    routed_.reset();
  }

  // Full reset (e.g. at teardown).
  void Clear() noexcept {
    tainted_ = false;
    step_.reset();
    routed_.reset();
    pinned_.clear();
    storage_rings_.clear();
  }

  void KeepPinned(void* ptr) {
    if (ptr != nullptr) {
      pinned_.push_back(PinnedKeepAlive{.pointer = ptr});
    }
  }

  void RetainStorageRing(providers::Storage* ring) {
    if (ring != nullptr) {
      storage_rings_.push_back(StorageRingKeepAlive{.ring = ring});
    }
  }

  std::size_t kept_pinned() const noexcept { return pinned_.size(); }
  std::size_t unproven_storage_rings() const noexcept { return storage_rings_.size(); }

  void RetainStep(std::shared_ptr<StepTransaction> step) noexcept { step_ = std::move(step); }
  void ReleaseStep() noexcept { step_.reset(); }
  const std::shared_ptr<StepTransaction>& step() const noexcept { return step_; }
  bool has_step() const noexcept { return step_ != nullptr; }

  void RetainRouted(std::shared_ptr<void> routed) noexcept { routed_ = std::move(routed); }
  bool has_routed() const noexcept { return routed_ != nullptr; }
  template <typename T>
  std::shared_ptr<T> routed_as() const noexcept {
    return std::static_pointer_cast<T>(routed_);
  }
  void ReleaseProvenRouted() noexcept { routed_.reset(); }

  const std::vector<PinnedKeepAlive>& pinned() const noexcept { return pinned_; }
  const std::vector<StorageRingKeepAlive>& storage_rings() const noexcept { return storage_rings_; }

 private:
  bool tainted_ = false;
  std::shared_ptr<StepTransaction> step_;
  std::shared_ptr<void> routed_;
  std::vector<PinnedKeepAlive> pinned_;
  std::vector<StorageRingKeepAlive> storage_rings_;
};

using NodeUnproven = UnprovenWork;
using RequestUnproven = UnprovenWork;

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_UNPROVEN_WORK_H_
