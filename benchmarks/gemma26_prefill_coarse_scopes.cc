// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#include <nvtx3/nvToolsExt.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <ostream>
#include <thread>

#include "gemma26_prefill_coarse.h"

namespace jitllm::benchmark::coarse {
namespace {
constexpr std::uint32_t kThreads = 4;
constexpr auto kPhases = static_cast<std::uint32_t>(Phase::kCount);
constexpr std::array<const char*, kPhases> kNames{"wave",
                                                  "places.call",
                                                  "places.body",
                                                  "inputs",
                                                  "state.grow",
                                                  "state.describe",
                                                  "state.describe.body",
                                                  "state.acquire",
                                                  "state.register",
                                                  "state.register.body",
                                                  "plan.hit",
                                                  "plan.miss",
                                                  "graph.build_bind",
                                                  "plan.first",
                                                  "placement",
                                                  "plan.second",
                                                  "plan.bind",
                                                  "coverage",
                                                  "stage",
                                                  "job.completed",
                                                  "job.submit",
                                                  "publish",
                                                  "host.release",
                                                  "refresh.call",
                                                  "refresh.body",
                                                  "cohort.hold",
                                                  "probe.busy",
                                                  "probe.wait"};
struct Counter {
  std::uint64_t count = 0, wall = 0, cpu = 0;
};
struct Owner {
  std::atomic<std::uint64_t> tid{0};
  std::array<Counter, kPhases> counters{};
};
// Exactly one definition shared by all overlaid translation units. Each
// counter is written only by its recorded owning thread, read after all
// existing node.Call/node.Job completion boundaries and End().
std::array<Owner, kThreads> owners;
std::atomic<std::uint32_t> context{0};
std::atomic<bool> invalid{false};
std::atomic<bool> began{false};
static_assert(sizeof(owners) + sizeof(context) + sizeof(invalid) + sizeof(began) <= 4096);
std::int64_t Clock(clockid_t id) {
  timespec t{};
  if (clock_gettime(id, &t) != 0 || t.tv_sec < 0 || t.tv_nsec < 0 || t.tv_nsec >= 1000000000 ||
      t.tv_sec > (std::numeric_limits<std::int64_t>::max() - t.tv_nsec) / 1000000000) {
    invalid.store(true, std::memory_order_relaxed);
    return -1;
  }
  return t.tv_sec * 1000000000 + t.tv_nsec;
}
std::uint32_t OwnerIndex() {
  const auto raw_tid = syscall(SYS_gettid);
  if (raw_tid <= 0) {
    invalid.store(true);
    return kThreads;
  }
  const auto tid = static_cast<std::uint64_t>(raw_tid);
  for (std::uint32_t i = 0; i < kThreads; ++i) {
    auto current = owners[i].tid.load(std::memory_order_acquire);
    if (current == tid) return i;
    if (current == 0 &&
        owners[i].tid.compare_exchange_strong(current, tid, std::memory_order_acq_rel))
      return i;
  }
  invalid.store(true, std::memory_order_relaxed);
  return kThreads;
}
}  // namespace
Context Current() {
  const auto value = context.load(std::memory_order_acquire);
  return {.enabled = (value & 0x100) != 0, .ordinal = static_cast<std::uint8_t>(value & 0xff)};
}
bool Begin() {
  if (began.exchange(true) || Clock(CLOCK_MONOTONIC) < 0 || Clock(CLOCK_THREAD_CPUTIME_ID) < 0)
    return false;
  context.store(0x100, std::memory_order_release);
  return true;
}
bool Ordinal(std::uint32_t ordinal) {
  if (ordinal >= 8 || !Current().enabled) {
    invalid.store(true);
    return false;
  }
  context.store(0x100 | ordinal, std::memory_order_release);
  return true;
}
void End() { context.store(0, std::memory_order_release); }
Scope::Scope(Phase phase, Context captured) : phase_(phase), context_(captured) {
  if (!context_.enabled) return;
  owner_ = OwnerIndex();
  wall_ = Clock(CLOCK_MONOTONIC);
  cpu_ = Clock(CLOCK_THREAD_CPUTIME_ID);
  nvtxEventAttributes_t event{};
  event.version = NVTX_VERSION;
  event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  event.messageType = NVTX_MESSAGE_TYPE_ASCII;
  event.message.ascii = kNames[static_cast<std::uint32_t>(phase_)];
  event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
  event.payload.ullValue = context_.ordinal;
  nvtxRangePushEx(&event);
}
Scope::~Scope() {
  if (!context_.enabled) return;
  nvtxRangePop();
  const auto cpu = Clock(CLOCK_THREAD_CPUTIME_ID);
  const auto wall = Clock(CLOCK_MONOTONIC);
  if (owner_ >= kThreads ||
      owners[owner_].tid.load() != static_cast<std::uint64_t>(syscall(SYS_gettid)) || wall_ < 0 ||
      cpu_ < 0 || wall < wall_ || cpu < cpu_) {
    invalid.store(true, std::memory_order_relaxed);
    return;
  }
  auto& counter = owners[owner_].counters[static_cast<std::uint32_t>(phase_)];
  const auto elapsed = static_cast<std::uint64_t>(wall - wall_);
  const auto charged = static_cast<std::uint64_t>(cpu - cpu_);
  if (counter.count == UINT64_MAX || elapsed > UINT64_MAX - counter.wall ||
      charged > UINT64_MAX - counter.cpu) {
    invalid.store(true);
    return;
  }
  ++counter.count;
  counter.wall += elapsed;
  counter.cpu += charged;
}
bool Report(std::ostream& output) {
  if (Current().enabled || invalid.load() || !began.load()) return false;
  output << "PREFILL_COARSE known_static_bytes="
         << sizeof(owners) + sizeof(context) + sizeof(invalid) + sizeof(began)
         << " clock=thread_cpu_ns elapsed=monotonic_ns inclusive=1\n";
  for (std::uint32_t i = 0; i < kThreads; ++i) {
    if (owners[i].tid.load() == 0) continue;
    for (std::uint32_t p = 0; p < kPhases; ++p) {
      const auto& c = owners[i].counters[p];
      if (c.count != 0)
        output << "PREFILL_SCOPE owner=" << i << " tid=" << owners[i].tid.load()
               << " phase=" << kNames[p] << " count=" << c.count << " wall_ns=" << c.wall
               << " thread_cpu_ns=" << c.cpu << '\n';
    }
  }
  return output.good();
}
bool ClockProbe(std::ostream& output) {
  // No model/provider/device allocation. Busy then timed condition wait on
  // the same worker; no perf permissions or system configuration changes.
  if (!Begin()) return false;
  std::int64_t bw = -1, bc = -1, ww = -1, wc = -1;
  std::uint64_t witness = 1;
  std::thread worker([&] {
    {
      Scope busy_scope(Phase::kProbeBusy);
      const auto a = Clock(CLOCK_MONOTONIC), b = Clock(CLOCK_THREAD_CPUTIME_ID);
      if (a < 0 || b < 0 || a > INT64_MAX - 150000000) return;
      const auto stop = a + 150000000;
      while (true) {
        const auto now = Clock(CLOCK_MONOTONIC);
        if (now < 0) return;
        if (now >= stop) break;
        for (int j = 0; j < 256; ++j) witness = witness * 6364136223846793005ULL + 1;
        std::atomic_signal_fence(std::memory_order_seq_cst);
      }
      bc = Clock(CLOCK_THREAD_CPUTIME_ID) - b;
      bw = Clock(CLOCK_MONOTONIC) - a;
    }
    Scope wait_scope(Phase::kProbeWait);
    std::mutex mutex;
    std::condition_variable cv;
    std::unique_lock lock(mutex);
    const auto c = Clock(CLOCK_MONOTONIC), d = Clock(CLOCK_THREAD_CPUTIME_ID);
    if (c < 0 || d < 0) return;
    cv.wait_for(lock, std::chrono::milliseconds(150), [] { return false; });
    wc = Clock(CLOCK_THREAD_CPUTIME_ID) - d;
    ww = Clock(CLOCK_MONOTONIC) - c;
  });
  worker.join();
  End();
  const bool reported = Report(output);
  output << "CLOCK_PROBE busy_wall_ns=" << bw << " busy_cpu_ns=" << bc << " wait_wall_ns=" << ww
         << " wait_cpu_ns=" << wc << " witness=" << witness << '\n';
  return reported && !invalid.load() && bw > 0 && bc > 0 && ww >= 150000000 && wc >= 0 &&
         bc > bw / 2 && wc < ww / 10 && output.good();
}
}  // namespace jitllm::benchmark::coarse
