// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A fixed-capacity FIFO shared by threads (D-048,
// docs/async-model.md#bounded-queues-and-progress-under-saturation). Its
// storage is allocated once, when it is built; a push never blocks and never
// allocates, and a full queue refuses the push, which the caller sees before
// any work starts. Part of the capacity is held back for cleanup commands,
// so deregistration and unwind can still be queued when ordinary submission
// is full. Closing refuses new pushes; what is already queued still drains.
//
// Guarded by a mutex: one short critical section per push or pop. Whether
// a lock-free ring is worth it is a measurement, not an assumption.

#ifndef LLMP_BASE_BOUNDED_QUEUE_H_
#define LLMP_BASE_BOUNDED_QUEUE_H_

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stop_token>
#include <utility>
#include <vector>

#include "base/check.h"

namespace llmp::base {

enum class PushResult : std::uint8_t {
  kAccepted,
  kFull,    // no room in the capacity this push may use
  kClosed,  // the queue takes nothing more
};

// Which capacity a push may use.
enum class PushKind : std::uint8_t {
  kOrdinary,  // the capacity less the cleanup reserve
  kCleanup,   // all of it
};

template <typename T>
class BoundedQueue {
 public:
  // `capacity` entries in all, `reserved` of them only for cleanup.
  BoundedQueue(std::size_t capacity, std::size_t reserved)
      : ring_(capacity), ordinary_(capacity - reserved) {
    Check(capacity > 0 && reserved < capacity, "a bounded queue needs ordinary capacity");
  }

  BoundedQueue(const BoundedQueue&) = delete;
  BoundedQueue& operator=(const BoundedQueue&) = delete;
  BoundedQueue(BoundedQueue&&) = delete;
  BoundedQueue& operator=(BoundedQueue&&) = delete;
  ~BoundedQueue() = default;

  // Moves from `value` only if it is accepted: a refused command stays
  // with the caller, who still owns whatever it holds.
  PushResult TryPush(T&& value, PushKind kind = PushKind::kOrdinary) {
    {
      const std::scoped_lock lock(mutex_);
      if (closed_) {
        return PushResult::kClosed;
      }
      const std::size_t limit = kind == PushKind::kCleanup ? ring_.size() : ordinary_;
      if (count_ >= limit) {
        return PushResult::kFull;
      }
      ring_[(head_ + count_) % ring_.size()].emplace(std::move(value));
      ++count_;
    }
    ready_.notify_one();
    return PushResult::kAccepted;
  }

  // The oldest entry, if any; never blocks.
  std::optional<T> TryPop() {
    const std::scoped_lock lock(mutex_);
    return PopLocked();
  }

  // Waits for the oldest entry. Returns nothing if stop is requested while
  // the queue is empty, or once it is closed and empty.
  std::optional<T> Pop(const std::stop_token& stop) {
    std::unique_lock lock(mutex_);
    ready_.wait(lock, stop, [this] { return count_ > 0 || closed_; });
    return PopLocked();
  }

  // Refuses every later push and wakes waiting consumers; entries already
  // queued can still be popped.
  void Close() {
    {
      const std::scoped_lock lock(mutex_);
      closed_ = true;
    }
    ready_.notify_all();
  }

  std::size_t size() const {
    const std::scoped_lock lock(mutex_);
    return count_;
  }
  // Closed, with nothing left to pop: its consumer can stop.
  bool drained() const {
    const std::scoped_lock lock(mutex_);
    return closed_ && count_ == 0;
  }
  std::size_t capacity() const { return ring_.size(); }

 private:
  std::optional<T> PopLocked() {
    if (count_ == 0) {
      return std::nullopt;
    }
    std::optional<T> value = std::move(ring_[head_]);
    ring_[head_].reset();
    head_ = (head_ + 1) % ring_.size();
    --count_;
    return value;
  }

  mutable std::mutex mutex_;
  std::condition_variable_any ready_;
  std::vector<std::optional<T>> ring_;
  std::size_t ordinary_;
  std::size_t head_ = 0;
  std::size_t count_ = 0;
  bool closed_ = false;
};

}  // namespace llmp::base

#endif  // LLMP_BASE_BOUNDED_QUEUE_H_
