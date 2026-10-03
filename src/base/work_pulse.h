// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The pulse of long work on the CPU (D-102's hang recovery,
// docs/runtime-serving.md#hang-recovery): work that moves is not a hang.
// A thread that serves (the runtime's driver) is given a pulse its watcher
// reads; long CPU work on it (a template's rendering, a tokenization)
// beats it every so often, and at its cancellation
// checkpoints stops when the watcher asked it to (a confirmed hang's
// rung 1 on work that waits on nothing it could cancel). A thread given no
// pulse beats nothing and is never asked to stop. Header-only and
// lock-free: a beat is one relaxed increment.

#ifndef JITLLM_BASE_WORK_PULSE_H_
#define JITLLM_BASE_WORK_PULSE_H_

#include <atomic>
#include <cstdint>

namespace jitllm::base {

class WorkPulse {
 public:
  // Any thread.
  std::uint64_t beats() const { return beats_.load(std::memory_order_relaxed); }
  bool cancelled() const { return cancel_.load(std::memory_order_acquire); }
  // Cancellations asked so far: work that stopped compares it with the
  // count it read before it began, to tell the watcher's from its own.
  std::uint64_t cancels() const { return cancels_.load(std::memory_order_acquire); }
  // The watcher's: the work under way asked to stop at its next
  // checkpoint, until Clear.
  void Cancel() {
    cancels_.fetch_add(1, std::memory_order_acq_rel);
    cancel_.store(true, std::memory_order_release);
  }
  void Clear() { cancel_.store(false, std::memory_order_release); }

  // The working thread's.
  void Beat() { beats_.fetch_add(1, std::memory_order_relaxed); }

 private:
  std::atomic<std::uint64_t> beats_{0};
  std::atomic<std::uint64_t> cancels_{0};
  std::atomic<bool> cancel_{false};
};

namespace work_pulse_internal {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
inline thread_local WorkPulse* current = nullptr;
}  // namespace work_pulse_internal

// The calling thread's pulse from now on (none: nullptr). Not owned.
inline void SetThreadPulse(WorkPulse* pulse) { work_pulse_internal::current = pulse; }
inline WorkPulse* ThreadPulse() { return work_pulse_internal::current; }

// From long CPU work, every so often: a beat of the calling thread's
// pulse; false when its watcher asked the work to stop (it then fails as
// cancelled). Without a pulse, true.
inline bool Pulse() {
  WorkPulse* pulse = work_pulse_internal::current;
  if (pulse == nullptr) {
    return true;
  }
  pulse->Beat();
  return !pulse->cancelled();
}

}  // namespace jitllm::base

#endif  // JITLLM_BASE_WORK_PULSE_H_
