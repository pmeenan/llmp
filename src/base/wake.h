// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A coalesced wake flag for one owner thread (D-048,
// docs/async-model.md#bounded-queues-and-progress-under-saturation): any
// number of producers publish their results first, then signal; the owner
// consumes the signal, rechecks everything that could have been published,
// and sleeps only while no signal is pending. Many signals before the owner
// looks cost one wakeup.
//
// Why no wakeup is lost: a producer's publication happens before its
// signal (release on the flag, acquire where the owner consumes it). A
// producer that sets the flag while the owner is checking it either is seen
// by that check, or takes the handshake mutex after the owner reaches its
// wait. The producer releases that mutex before notifying. Producers must
// finish Signal before the WakeFlag is destroyed; observing a publication
// alone does not prove that its producer has returned.
//
// An anticipation (the runtime wake, docs/experiments/runtime-wake/): a
// thread that expects to publish soon, such as a completion lane about to
// see a decode step's fence, asks the owner to stay awake until then. A
// thread asleep on the Spark takes hundreds of microseconds to run once
// signalled (RE-017), so the owner is woken ahead of the publication and
// polls instead of sleeping until the anticipation's deadline passes. It
// is a hint about when to poll, never a publication: the owner still
// learns what happened only from what producers published and signalled.

#ifndef LLMP_BASE_WAKE_H_
#define LLMP_BASE_WAKE_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>

namespace llmp::base {

class WakeFlag {
 public:
  using Clock = std::chrono::steady_clock;

  // Any thread, after publishing: wake the owner.
  void Signal() {
    if (!pending_.exchange(true, std::memory_order_acq_rel)) {
      {
        // Keep the waiter's predicate-to-sleep handshake, then let the
        // awakened owner take the mutex without contending with this notifier.
        const std::scoped_lock lock(mutex_);
      }
      wake_.notify_one();
    }
  }

  // The owner: takes a pending signal, if there is one, without waiting.
  bool Consume() { return pending_.exchange(false, std::memory_order_acquire); }

  // The owner: waits until a signal is pending or `timeout` passes, and
  // takes the signal. True if there was one. Spurious returns are possible
  // only as `false`, which the owner treats as a timer tick.
  template <typename Rep, typename Period>
  bool WaitFor(std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock lock(mutex_);
    return wake_.wait_for(lock, timeout, [this] { return Consume(); });
  }

  // The owner: as WaitFor, until `deadline`.
  bool WaitUntil(Clock::time_point deadline) {
    std::unique_lock lock(mutex_);
    return wake_.wait_until(lock, deadline, [this] { return Consume(); });
  }

  // Any thread: the owner should stay awake, polling, until `until`; wakes
  // it if this raises the deadline. Deadlines only rise: an earlier one
  // than the owner already has changes nothing.
  void Anticipate(Clock::time_point until) {
    const std::int64_t wanted = until.time_since_epoch().count();
    std::int64_t current = anticipated_.load(std::memory_order_relaxed);
    while (current < wanted &&
           !anticipated_.compare_exchange_weak(current, wanted, std::memory_order_relaxed)) {
    }
    if (current < wanted) {
      Signal();
    }
  }

  // The owner: an anticipation runs past `now`.
  bool Anticipating(Clock::time_point now) const {
    return anticipated_.load(std::memory_order_relaxed) > now.time_since_epoch().count();
  }

 private:
  std::atomic<bool> pending_{false};
  std::atomic<std::int64_t> anticipated_{std::numeric_limits<std::int64_t>::min()};
  std::mutex mutex_;
  std::condition_variable wake_;
};

// How long recurring work takes (a stream's fences, a request's steps),
// for a thread that sleeps through most of it and spins only around its
// likely ends: each of the last kSamples lengths is a likely end, so work
// whose length alternates (a decode step and a prefill chunk, verifies of
// different widths) is caught at whichever it takes. One thread's; not
// synchronized.
class Expectation {
 public:
  using Duration = WakeFlag::Clock::duration;
  static constexpr std::size_t kSamples = 8;

  void Add(Duration sample) {
    samples_[next_ % kSamples] = sample;
    ++next_;
  }
  bool known() const { return next_ > 0; }
  // Only once known(): the shortest of the last kSamples.
  Duration value() const {
    Duration shortest = Duration::max();
    for (std::size_t i = 0; i < kSamples && i < next_; ++i) {
      shortest = std::min(shortest, samples_[i]);
    }
    return shortest;
  }
  // The next likely end of work that has run for `elapsed`: the shortest
  // of the last kSamples not yet more than `past` behind it. None once the
  // work has outlasted them all (or nothing is known).
  std::optional<Duration> Next(Duration elapsed, Duration past) const {
    std::optional<Duration> next;
    for (std::size_t i = 0; i < kSamples && i < next_; ++i) {
      if (samples_[i] + past > elapsed && (!next || samples_[i] < *next)) {
        next = samples_[i];
      }
    }
    return next;
  }

 private:
  std::array<Duration, kSamples> samples_{};
  std::size_t next_ = 0;
};

}  // namespace llmp::base

#endif  // LLMP_BASE_WAKE_H_
