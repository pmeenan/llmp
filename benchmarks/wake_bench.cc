// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The runtime wake (docs/experiments/runtime-wake/): how quickly the chain
// from a step's fence to the next step's launch wakes on the GB10, and at
// what CPU cost, for each candidate way of waiting. Two modes:
//
//   llmp_wake_bench chain [--steps N] [--step-us US] [--host-us US]
//                           [--repeats N] [--mix] [--only NAME,...]
//                           [--spin-ahead-us US]
//   llmp_wake_bench node [--steps N] [--step-us US] [--host-us US]
//                          [--repeats N] [--poll-us US] [--spin-ahead-us US]
//
// chain: four threads in the roles of the runtime's, over raw CUDA: a
// submission lane (S) launches a step (a kernel that keeps the GPU busy
// --step-us and stamps its start and end with the GPU's global timer) and
// hands the fence to a completion lane (C); C waits for the fence in the
// configuration's way and publishes it to a scheduler (K), which tells a
// client (D); D does --host-us of host work (sampling) and asks K for the
// next step, which K posts to S. K and S wait on llmpalooza's WakeFlag, as the
// scheduler and the lanes do. Each configuration (Configs below) sets how C
// detects the completion and how K, S and D wait:
//   (a) a blocking-sync event, (b) a host function signalling a futex,
//   (c) a stream memory operation writing a mapped host flag (polled with
//   yield, with a timed futex wait, or with WFE), (d) an adaptive spin sized
//   to the expected step, (e) combinations: C predicts the step's end and
//   wakes K and S ahead of it (the relay), with or without a host
//   function's futex as the backstop.
// A fence is always proven by cuEventQuery, whatever woke C (D-048).
// --mix alternates 1x and 1/9x --step-us steps, pseudo-randomly, so a
// prediction is often wrong.
//
// node: the real runtime path on tests/support's paged node: a request
// holding its lease, N steps of the same kernel as device jobs under it,
// each step's round trip (the step's wall less its device span) and the
// process's CPU. --poll-us overrides the scheduler's and the device lane's
// poll windows (the harness's old 100 ms, a diagnostic); without it the
// runtime's own defaults apply.
//
// --spin-ahead-us (a diagnostic, both modes): how long before a likely end
// the completion lane and the client start to spin, instead of 1 ms
// (docs/experiments/runtime-wake/#a-latency-hold measured shorter ones).
//
// Output: Markdown rows. Round trip = the GPU's idle gap between a step's
// end and the next step's start, less the client's host work. Hops are
// host times, the GPU's related to the host clock by a calibration
// (Calibrate). CPU is the process's CPU time over wall time, in cores,
// while stepping and while idle (the second after the last step, from
// 0.3 s on).

#include <cuda.h>
#include <cuda_runtime.h>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <format>
#include <limits>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "base/bytes.h"
#include "base/wake.h"
#include "catalog/catalog.h"
#include "paged_node.h"
#include "scheduler/commands.h"
#include "wake_bench_kernels.h"

namespace {

using Clock = std::chrono::steady_clock;
using llmp::base::WakeFlag;
using std::chrono::microseconds;
using std::chrono::milliseconds;

std::int64_t Ns(Clock::time_point t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

double ProcessCpu() {
  timespec ts{};
  (void)::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return static_cast<double>(ts.tv_sec) + (static_cast<double>(ts.tv_nsec) / 1e9);
}

double Percentile(std::vector<double> samples, double p) {
  if (samples.empty()) {
    return 0;
  }
  std::ranges::sort(samples);
  const auto at = static_cast<std::size_t>(p * static_cast<double>(samples.size() - 1));
  return samples[at];
}

void Spin(std::chrono::nanoseconds work) {
  const Clock::time_point until = Clock::now() + work;
  while (Clock::now() < until) {
  }
}

// A 32-bit word that host threads and the GPU write, read atomically.
std::uint32_t Load(const std::uint32_t* word) { return __atomic_load_n(word, __ATOMIC_ACQUIRE); }

long FutexWait(std::uint32_t* word, std::uint32_t expected, std::int64_t timeout_ns) {
  timespec ts{.tv_sec = timeout_ns / 1000000000, .tv_nsec = timeout_ns % 1000000000};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg): the futex system call
  return ::syscall(SYS_futex, word, FUTEX_WAIT_PRIVATE, expected, &ts, nullptr, 0);
}

void FutexWake(std::uint32_t* word) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg): the futex system call
  (void)::syscall(SYS_futex, word, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
}

// Waits for a GPU write to reach `target` with WFE: the load-exclusive arms
// the monitor, which a write to the line (or the event stream) clears.
void WaitWfe(const std::uint32_t* word, std::uint32_t target) {
#ifdef __aarch64__
  while (true) {
    std::uint32_t value = 0;
    asm volatile("ldaxr %w0, [%1]" : "=r"(value) : "r"(word) : "memory");
    if (value >= target) {
      asm volatile("clrex" ::: "memory");
      return;
    }
    asm volatile("wfe" ::: "memory");
  }
#else
  while (Load(word) < target) {
    std::this_thread::yield();
  }
#endif
}

// How C learns that a step's fence completed.
enum class Detect : std::uint8_t {
  kQuery,          // cuEventQuery, yielding between (the runtime until now)
  kEvent,          // (a) cuEventSynchronize on a CU_EVENT_BLOCKING_SYNC event
  kHostFn,         // (b) cuLaunchHostFunc bumps a futex word C sleeps on
  kMemopSpin,      // (c) cuStreamWriteValue32 to a mapped host flag, polled with yield
  kMemopFutex,     // (c) ... polled with a 50 us timed futex wait
  kMemopWfe,       // (c) ... polled with WFE
  kPredict,        // (d) sleep until the expected end less a margin, then spin
  kPredictHostFn,  // (e) as kPredict, woken early by a host function's futex
};

// How K and S wait for their next input.
enum class Owner : std::uint8_t {
  kWindow,  // poll `window` after their last progress, then sleep on the wake flag
  kRelay,   // (e) as kWindow, and poll while C's anticipation runs; K polls
            // after a completion for as long as the client's next step
            // usually takes, and has S do the same
};

// How D waits for its step's result.
enum class Client : std::uint8_t {
  kSpin,     // yield until it comes (the paged harness's driver)
  kPredict,  // sleep until the expected end less a margin, spin, then sleep
};

struct Config {
  std::string_view name;
  microseconds window;  // K's and S's poll window
  Detect detect;
  Owner owner;
  Client client;
};

constexpr microseconds kDefaultWindow(200);  // the runtime's
constexpr std::array<Config, 11> kConfigs{{
    {"harness-polled (100 ms windows)", microseconds(100000), Detect::kQuery, Owner::kWindow,
     Client::kSpin},
    {"runtime until now (200 us windows)", kDefaultWindow, Detect::kQuery, Owner::kWindow,
     Client::kSpin},
    {"query-spin, client predicts", kDefaultWindow, Detect::kQuery, Owner::kWindow,
     Client::kPredict},
    {"(a) blocking-sync event", kDefaultWindow, Detect::kEvent, Owner::kWindow, Client::kPredict},
    {"(b) host function + futex", kDefaultWindow, Detect::kHostFn, Owner::kWindow,
     Client::kPredict},
    {"(c) memop flag, yield", kDefaultWindow, Detect::kMemopSpin, Owner::kWindow, Client::kPredict},
    {"(c) memop flag, futex 50 us", kDefaultWindow, Detect::kMemopFutex, Owner::kWindow,
     Client::kPredict},
    {"(c) memop flag, WFE", kDefaultWindow, Detect::kMemopWfe, Owner::kWindow, Client::kPredict},
    {"(d) adaptive spin (C only)", kDefaultWindow, Detect::kPredict, Owner::kWindow,
     Client::kPredict},
    {"(e) adaptive spin + relay", kDefaultWindow, Detect::kPredict, Owner::kRelay,
     Client::kPredict},
    {"(e) adaptive spin + relay, host fn backstop", kDefaultWindow, Detect::kPredictHostFn,
     Owner::kRelay, Client::kPredict},
}};

// The prediction's settings (the ones the runtime took, DeviceSettings),
// over the runtime's expectation: the shortest of the last eight.
constexpr auto kMargin = microseconds(1000);    // spin from this long before the expected end
                                                // (--spin-ahead-us overrides it)
constexpr auto kSlack = microseconds(1000);     // and until this long after it
constexpr auto kBackstop = microseconds(1000);  // sleeping, query at least this often
constexpr auto kBackoffMax =
    microseconds(1000);  // past the slack, sleep up to this between queries
using llmp::base::Expectation;

struct StepRecord {
  std::int64_t request = 0;    // D asked for the step
  std::int64_t k_request = 0;  // K took the request
  std::int64_t s_command = 0;  // S took the command
  std::int64_t handed = 0;     // S launched it and handed its fence to C
  std::int64_t detected = 0;   // C saw the fence complete
  std::int64_t k_done = 0;     // K took the completion
  std::int64_t d_done = 0;     // D saw the result
  std::int64_t d_posted = 0;   // D finished its host work
};

struct ChainOptions {
  int steps = 100;
  std::uint64_t step_us = 45000;
  std::uint64_t host_us = 200;
  int repeats = 3;
  bool mix = false;
  std::vector<std::string> only;
  microseconds spin_ahead = kMargin;
};

// The GPU's global timer against the host's steady clock: host minus GPU,
// the smallest seen (a write seen later only makes the difference larger).
std::int64_t Calibrate(cudaStream_t stream, std::uint64_t* tick_host, std::uint64_t* tick_dev) {
  *reinterpret_cast<volatile std::uint64_t*>(tick_host) = 0;
  if (llmp::wake::Tick(stream, 20000000, tick_dev) != cudaSuccess) {
    return 0;
  }
  std::int64_t best = std::numeric_limits<std::int64_t>::max();
  std::uint64_t last = 0;
  const auto until = Clock::now() + milliseconds(30);
  while (Clock::now() < until) {
    const std::uint64_t seen = *reinterpret_cast<volatile std::uint64_t*>(tick_host);
    const std::int64_t host = Ns(Clock::now());
    if (seen != 0 && seen != last) {
      best = std::min(best, host - static_cast<std::int64_t>(seen));
      last = seen;
    }
  }
  (void)cudaStreamSynchronize(stream);
  return best;
}

void PrintHeader() {
  std::println(
      "| Configuration | Steps | Round trip p50 / p99 / max (us) | Detect | C to K | K to D | D to "
      "K | K to S | S to GPU (p50 / p99, us) | Cores stepping | Cores idle |");
  std::println("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
}

std::string Pair(const std::vector<double>& v) {
  return std::format("{:.0f} / {:.0f}", Percentile(v, 0.5), Percentile(v, 0.99));
}

bool RunChain(const Config& config, const ChainOptions& o) {
  CUstream stream = nullptr;
  if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
    return false;
  }
  CUevent plain = nullptr;
  CUevent blocking = nullptr;
  (void)cuEventCreate(&plain, CU_EVENT_DISABLE_TIMING);
  (void)cuEventCreate(&blocking, CU_EVENT_DISABLE_TIMING | CU_EVENT_BLOCKING_SYNC);
  const auto steps = static_cast<std::size_t>(o.steps);
  // Mapped host memory: the steps' stamps, the memop flag and the ticker.
  std::uint64_t* stamps = nullptr;
  std::uint32_t* flag = nullptr;
  std::uint64_t* tick = nullptr;
  if (cudaHostAlloc(reinterpret_cast<void**>(&stamps), steps * 2 * sizeof(std::uint64_t),
                    cudaHostAllocMapped) != cudaSuccess ||
      cudaHostAlloc(reinterpret_cast<void**>(&flag), 64, cudaHostAllocMapped) != cudaSuccess ||
      cudaHostAlloc(reinterpret_cast<void**>(&tick), 64, cudaHostAllocMapped) != cudaSuccess) {
    return false;
  }
  std::uint64_t* stamps_dev = nullptr;
  std::uint32_t* flag_dev = nullptr;
  std::uint64_t* tick_dev = nullptr;
  (void)cudaHostGetDevicePointer(reinterpret_cast<void**>(&stamps_dev), stamps, 0);
  (void)cudaHostGetDevicePointer(reinterpret_cast<void**>(&flag_dev), flag, 0);
  (void)cudaHostGetDevicePointer(reinterpret_cast<void**>(&tick_dev), tick, 0);
  *reinterpret_cast<volatile std::uint32_t*>(flag) = 0;
  auto* const cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const std::int64_t offset = Calibrate(cuda_stream, tick, tick_dev);

  // Each step's GPU time: --step-us, or with --mix 1/9 of it about a third
  // of the time.
  std::vector<std::uint64_t> step_ns(steps);
  std::uint32_t lcg = 12345;
  for (std::size_t i = 0; i < steps; ++i) {
    lcg = (lcg * 1103515245U) + 12345U;
    const bool shorter = o.mix && ((lcg >> 16U) % 3U) == 0;
    step_ns[i] = (shorter ? o.step_us / 9 : o.step_us) * 1000;
  }

  std::vector<StepRecord> records(steps);
  WakeFlag k_wake;
  WakeFlag s_wake;
  WakeFlag c_wake;
  WakeFlag d_wake;
  std::atomic<std::int64_t> requested{-1};    // D to K: the step asked for
  std::atomic<std::int64_t> commanded{-1};    // K to S
  std::atomic<std::int64_t> handed{-1};       // S to C
  std::atomic<std::int64_t> completed{-1};    // C to K
  std::atomic<std::int64_t> reported{-1};     // K to D
  alignas(64) std::uint32_t hostfn_word = 0;  // host functions bump it, C waits on it
  std::atomic<bool> stop{false};
  std::atomic<bool> failed{false};
  const microseconds margin = o.spin_ahead;
  const bool relay = config.owner == Owner::kRelay;
  const bool hostfn = config.detect == Detect::kHostFn || config.detect == Detect::kPredictHostFn;
  const bool memop = config.detect == Detect::kMemopSpin || config.detect == Detect::kMemopFutex ||
                     config.detect == Detect::kMemopWfe;

  // An owner's wait: poll (yield) while `polling`, else sleep on its flag.
  const auto idle = [](WakeFlag& wake, bool polling) {
    if (polling) {
      std::this_thread::yield();
    } else {
      (void)wake.WaitFor(milliseconds(100));
    }
  };

  std::jthread k_thread([&] {
    std::int64_t seen_request = -1;
    std::int64_t seen_done = -1;
    Clock::time_point last = Clock::now();
    Clock::duration window = config.window;
    Clock::time_point done_at{};
    std::int64_t gap_estimate = 0;  // the client's usual think time, ns
    while (!stop.load(std::memory_order_acquire)) {
      (void)k_wake.Consume();
      bool progress = false;
      if (const std::int64_t done = completed.load(std::memory_order_acquire); done > seen_done) {
        seen_done = done;
        const auto now = Clock::now();
        records[static_cast<std::size_t>(done)].k_done = Ns(now);
        reported.store(done, std::memory_order_release);
        d_wake.Signal();
        progress = true;
        window = config.window;
        if (relay) {
          // Poll for as long as the client usually takes to ask again, and
          // have S do the same.
          const auto follow = std::clamp<std::int64_t>((gap_estimate * 3 / 2) + 200000,
                                                       config.window.count() * 1000, 10000000);
          window = std::chrono::nanoseconds(follow);
          s_wake.Anticipate(now + std::chrono::nanoseconds(follow));
          done_at = now;
        }
      }
      if (const std::int64_t request = requested.load(std::memory_order_acquire);
          request > seen_request) {
        seen_request = request;
        const auto now = Clock::now();
        records[static_cast<std::size_t>(request)].k_request = Ns(now);
        if (relay && done_at != Clock::time_point{}) {
          const std::int64_t gap = Ns(now) - Ns(done_at);
          if (gap < 10000000) {
            gap_estimate = gap_estimate == 0 ? gap : ((gap_estimate * 3) + gap) / 4;
          }
        }
        commanded.store(request, std::memory_order_release);
        s_wake.Signal();
        progress = true;
        window = config.window;
      }
      const auto now = Clock::now();
      if (progress) {
        last = now;
        continue;
      }
      idle(k_wake, now - last < window || (relay && k_wake.Anticipating(now)));
    }
  });

  std::jthread s_thread([&] {
    (void)cudaSetDevice(0);
    std::int64_t seen = -1;
    Clock::time_point last = Clock::now();
    while (!stop.load(std::memory_order_acquire)) {
      (void)s_wake.Consume();
      const std::int64_t command = commanded.load(std::memory_order_acquire);
      const auto now = Clock::now();
      if (command <= seen) {
        idle(s_wake, now - last < config.window || (relay && s_wake.Anticipating(now)));
        continue;
      }
      seen = command;
      const auto i = static_cast<std::size_t>(command);
      records[i].s_command = Ns(now);
      bool ok = llmp::wake::Step(cuda_stream, step_ns[i], stamps_dev + (2 * i)) == cudaSuccess;
      ok = ok && cuEventRecord(config.detect == Detect::kEvent ? blocking : plain, stream) ==
                     CUDA_SUCCESS;
      if (hostfn) {
        ok = ok && cuLaunchHostFunc(
                       stream,
                       [](void* data) {
                         auto* word = static_cast<std::uint32_t*>(data);
                         (void)__atomic_fetch_add(word, 1, __ATOMIC_RELEASE);
                         FutexWake(word);
                       },
                       &hostfn_word) == CUDA_SUCCESS;
      }
      if (memop) {
        ok = ok && cuStreamWriteValue32(stream, reinterpret_cast<CUdeviceptr>(flag_dev),
                                        static_cast<std::uint32_t>(command + 1),
                                        CU_STREAM_WRITE_VALUE_DEFAULT) == CUDA_SUCCESS;
      }
      if (!ok) {
        failed.store(true);
      }
      records[i].handed = Ns(Clock::now());
      handed.store(command, std::memory_order_release);
      c_wake.Signal();
      last = Clock::now();
    }
  });

  std::jthread c_thread([&] {
    (void)cudaSetDevice(0);
    std::int64_t seen = -1;
    Expectation expected;
    while (!stop.load(std::memory_order_acquire)) {
      (void)c_wake.Consume();
      const std::int64_t fence = handed.load(std::memory_order_acquire);
      if (fence <= seen) {
        (void)c_wake.WaitFor(milliseconds(100));  // no fence out: sleep
        continue;
      }
      seen = fence;
      const auto i = static_cast<std::size_t>(fence);
      const auto target = static_cast<std::uint32_t>(fence + 1);
      CUevent event = config.detect == Detect::kEvent ? blocking : plain;
      const Clock::time_point from = Clock::time_point(Clock::duration(records[i].handed));
      Clock::time_point relayed{};  // until when K and S were last told to poll
      Clock::duration backoff = microseconds(50);
      while (true) {
        switch (config.detect) {
          case Detect::kEvent:
            (void)cuEventSynchronize(event);
            break;
          case Detect::kHostFn:
            for (std::uint32_t word = Load(&hostfn_word); word < target;
                 word = Load(&hostfn_word)) {
              (void)FutexWait(&hostfn_word, word, 100000000);
            }
            break;
          case Detect::kMemopSpin:
            while (Load(flag) < target) {
              std::this_thread::yield();
            }
            break;
          case Detect::kMemopFutex:
            // The GPU's write wakes no futex: this is a 50 us timed poll.
            for (std::uint32_t word = Load(flag); word < target; word = Load(flag)) {
              (void)FutexWait(flag, word, 50000);
            }
            break;
          case Detect::kMemopWfe:
            WaitWfe(flag, target);
            break;
          default:
            break;
        }
        if (cuEventQuery(event) == CUDA_SUCCESS) {
          break;  // the proof, whatever woke C
        }
        const auto now = Clock::now();
        if (config.detect != Detect::kPredict && config.detect != Detect::kPredictHostFn) {
          std::this_thread::yield();  // kQuery; a woken wait whose fence is not yet seen
          continue;
        }
        // The next likely end (one of the stream's last eight lengths) not
        // yet outlasted, as the runtime's completion lane does.
        const auto next = expected.Next(now - from, kSlack);
        if (!expected.known() || (next && now >= from + *next - margin)) {
          if (relay && next && relayed < from + *next + kSlack) {
            relayed = from + *next + kSlack;
            k_wake.Anticipate(relayed);
            s_wake.Anticipate(relayed);
          }
          if (!expected.known() && now >= from + kSlack) {
            (void)c_wake.WaitFor(backoff);  // no history and past its first spin: back off
            backoff = std::min<Clock::duration>(backoff * 2, kBackoffMax);
          } else {
            std::this_thread::yield();  // spinning through a likely end
          }
          continue;
        }
        if (!next) {
          (void)c_wake.WaitFor(backoff);  // longer than every recent one: back off
          backoff = std::min<Clock::duration>(backoff * 2, kBackoffMax);
          continue;
        }
        // Asleep until the spin, querying at least every kBackstop.
        const auto until = std::min(from + *next - margin, now + kBackstop);
        if (config.detect == Detect::kPredictHostFn) {
          const std::uint32_t word = Load(&hostfn_word);
          if (word < target) {
            (void)FutexWait(&hostfn_word, word, Ns(until) - Ns(now));
          }
        } else {
          (void)c_wake.WaitUntil(until);
        }
      }
      const auto now = Clock::now();
      records[i].detected = Ns(now);
      expected.Add(now - from);
      completed.store(fence, std::memory_order_release);
      k_wake.Signal();
    }
  });

  // D: the client, on this thread.
  Expectation step_wall;
  const double cpu_start = ProcessCpu();
  const auto wall_start = Clock::now();
  for (std::size_t i = 0; i < steps && !failed.load(); ++i) {
    const auto asked = Clock::now();
    records[i].request = Ns(asked);
    requested.store(static_cast<std::int64_t>(i), std::memory_order_release);
    k_wake.Signal();
    const auto has = [&] {
      return reported.load(std::memory_order_acquire) >= static_cast<std::int64_t>(i);
    };
    while (!has()) {
      if (config.client == Client::kSpin) {
        std::this_thread::yield();
        continue;
      }
      const auto now = Clock::now();
      const auto next = step_wall.Next(now - asked, kSlack);
      if (step_wall.known() && !next) {
        (void)d_wake.WaitFor(kSlack);  // longer than every recent one
      } else if (next && now < asked + *next - margin) {
        (void)d_wake.WaitUntil(asked + *next - margin);
      } else {
        std::this_thread::yield();
      }
    }
    (void)d_wake.Consume();
    const auto seen = Clock::now();
    records[i].d_done = Ns(seen);
    step_wall.Add(seen - asked);
    Spin(microseconds(o.host_us));
    records[i].d_posted = Ns(Clock::now());
  }
  const double step_cpu = ProcessCpu() - cpu_start;
  const double step_wall_s = std::chrono::duration<double>(Clock::now() - wall_start).count();
  std::this_thread::sleep_for(milliseconds(300));
  const double idle_cpu_start = ProcessCpu();
  const auto idle_start = Clock::now();
  std::this_thread::sleep_for(milliseconds(1000));
  const double idle_cpu = ProcessCpu() - idle_cpu_start;
  const double idle_wall = std::chrono::duration<double>(Clock::now() - idle_start).count();
  stop.store(true, std::memory_order_release);
  k_wake.Signal();
  s_wake.Signal();
  c_wake.Signal();
  k_thread.join();
  s_thread.join();
  c_thread.join();
  (void)cuStreamSynchronize(stream);

  std::vector<double> round_trip;
  std::vector<double> detect;
  std::vector<double> c_k;
  std::vector<double> k_d;
  std::vector<double> d_k;
  std::vector<double> k_s;
  std::vector<double> s_gpu;
  const auto us = [](std::int64_t ns) { return static_cast<double>(ns) / 1e3; };
  for (std::size_t i = 1; i < steps; ++i) {
    const StepRecord& p = records[i - 1];
    const StepRecord& r = records[i];
    const auto end = static_cast<std::int64_t>(stamps[(2 * (i - 1)) + 1]);
    const auto start = static_cast<std::int64_t>(stamps[2 * i]);
    round_trip.push_back(us(start - end - (p.d_posted - p.d_done)));
    detect.push_back(us(p.detected - (end + offset)));
    c_k.push_back(us(p.k_done - p.detected));
    k_d.push_back(us(p.d_done - p.k_done));
    d_k.push_back(us(r.k_request - r.request));
    k_s.push_back(us(r.s_command - r.k_request));
    s_gpu.push_back(us(start + offset - r.s_command));
  }
  std::println(
      "| {} | {} | {:.0f} / {:.0f} / {:.0f} | {} | {} | {} | {} | {} | {} | {:.2f} | {:.2f} |",
      config.name, steps, Percentile(round_trip, 0.5), Percentile(round_trip, 0.99),
      Percentile(round_trip, 1.0), Pair(detect), Pair(c_k), Pair(k_d), Pair(d_k), Pair(k_s),
      Pair(s_gpu), step_cpu / step_wall_s, idle_cpu / idle_wall);
  (void)cudaFreeHost(stamps);
  (void)cudaFreeHost(flag);
  (void)cudaFreeHost(tick);
  (void)cuEventDestroy(plain);
  (void)cuEventDestroy(blocking);
  (void)cuStreamDestroy(stream);
  return !failed.load();
}

struct NodeOptions {
  int steps = 100;
  std::uint64_t step_us = 45000;
  std::uint64_t host_us = 200;
  int repeats = 3;
  std::int64_t poll_us = -1;     // the runtime's defaults
  std::int64_t spin_ahead = -1;  // the runtime's default (DeviceSettings::spin_ahead)
};

namespace ts = llmp::test_support;
namespace sc = llmp::scheduler;

bool RunNode(const NodeOptions& o) {
  ts::NodeSettings settings;
  settings.compute_streams = 1;
  if (o.poll_us >= 0) {
    settings.poll_window = microseconds(o.poll_us);
  }
  if (o.spin_ahead >= 0) {
    settings.spin_ahead = microseconds(o.spin_ahead);
  }
  ts::PagedNode node(settings);
  auto fail = [](std::string_view what, const std::string& why) {
    std::println(stderr, "{}: {}", what, why);
    return false;
  };
  if (auto r = node.Open(); !r) {
    return fail("open", r.error());
  }
  if (auto r = node.MapWorkspace(ts::kPagedExtent, ts::kPagedExtent); !r) {
    return fail("workspace", r.error());
  }
  if (auto r = node.Start(llmp::base::Bytes(std::uint64_t{1} << 30)); !r) {
    return fail("start", r.error());
  }
  node.Run();
  const auto steps = static_cast<std::size_t>(o.steps);
  std::uint64_t* stamps = nullptr;
  if (cudaHostAlloc(reinterpret_cast<void**>(&stamps), 2 * sizeof(std::uint64_t),
                    cudaHostAllocMapped) != cudaSuccess) {
    return false;
  }
  std::uint64_t* stamps_dev = nullptr;
  (void)cudaHostGetDevicePointer(reinterpret_cast<void**>(&stamps_dev), stamps, 0);
  const auto closure = node.catalog().ClosureOfExtents(node.activations().extents);
  if (!closure) {
    return false;
  }
  // The GPU's clock against the host's, on a stream of the bench's own,
  // before each repeat: the two drift apart by microseconds a second.
  cudaStream_t own = nullptr;
  std::uint64_t* tick = nullptr;
  std::uint64_t* tick_dev = nullptr;
  if (cudaStreamCreateWithFlags(&own, cudaStreamNonBlocking) != cudaSuccess ||
      cudaHostAlloc(reinterpret_cast<void**>(&tick), 64, cudaHostAllocMapped) != cudaSuccess) {
    return false;
  }
  (void)cudaHostGetDevicePointer(reinterpret_cast<void**>(&tick_dev), tick, 0);
  bool ok = true;
  for (int repeat = 0; repeat < o.repeats && ok; ++repeat) {
    const std::int64_t offset = Calibrate(own, tick, tick_dev);
    if (auto r = node.BeginRequest(0, *closure, "the wake bench's request"); !r) {
      return fail("request", r.error());
    }
    std::vector<double> round_trip;
    std::vector<double> to_gpu;    // the call to the kernel's start
    std::vector<double> from_gpu;  // the kernel's end to the return
    std::vector<double> to_job;    // the call to the job's start on the device lane
    double device = 0;
    const double cpu_start = ProcessCpu();
    const auto wall_start = Clock::now();
    for (std::size_t i = 0; i < steps && ok; ++i) {
      auto job = [&o, stamps_dev](llmp::providers::NativeStream native) {
        return llmp::wake::Step(static_cast<cudaStream_t>(native.handle), o.step_us * 1000,
                                stamps_dev) == cudaSuccess
                   ? sc::JobResult::kQueued
                   : sc::JobResult::kUnknown;
      };
      const std::int64_t called = Ns(Clock::now());
      if (auto r = node.Job(*closure, std::move(job), "a step", 0); !r) {
        ok = fail("step", r.error());
        break;
      }
      const std::int64_t returned = Ns(Clock::now());
      const ts::StepTimes t = node.TakeTimes(0);
      if (i > 0) {  // the first includes the scheduler's first turn after the lease
        round_trip.push_back((t.wall - t.device) * 1e6);
        device += t.device;
        const auto start = static_cast<std::int64_t>(stamps[0]) + offset;
        const auto end = static_cast<std::int64_t>(stamps[1]) + offset;
        to_gpu.push_back(static_cast<double>(start - called) / 1e3);
        from_gpu.push_back(static_cast<double>(returned - end) / 1e3);
        to_job.push_back(t.dispatch * 1e6);
      }
      Spin(microseconds(o.host_us));
    }
    const double step_cpu = ProcessCpu() - cpu_start;
    const double step_wall = std::chrono::duration<double>(Clock::now() - wall_start).count();
    if (auto r = node.EndRequest(0); !r) {
      return fail("end", r.error());
    }
    std::this_thread::sleep_for(milliseconds(300));
    const double idle_cpu_start = ProcessCpu();
    const auto idle_start = Clock::now();
    std::this_thread::sleep_for(milliseconds(1000));
    const double idle_cpu = ProcessCpu() - idle_cpu_start;
    const double idle_wall = std::chrono::duration<double>(Clock::now() - idle_start).count();
    std::println(
        "| node, {} | {} | {:.1f} / {:.1f} / {:.1f} | {} | {} | {} | {:.2f} | {:.2f} | {:.2f} |",
        std::format(
            "{}{}",
            o.poll_us < 0 ? std::string("runtime defaults")
                          : std::format("poll windows {} us", o.poll_us),
            o.spin_ahead < 0 ? std::string() : std::format(", spin ahead {} us", o.spin_ahead)),
        round_trip.size(), Percentile(round_trip, 0.5), Percentile(round_trip, 0.99),
        Percentile(round_trip, 1.0), Pair(to_job), Pair(to_gpu), Pair(from_gpu),
        device * 1e3 / static_cast<double>(std::max<std::size_t>(round_trip.size(), 1)),
        step_cpu / step_wall, idle_cpu / idle_wall);
  }
  std::vector<ts::PagedModel*> none;
  if (auto r = node.TearDown(none); !r) {
    return fail("teardown", r.error());
  }
  (void)cudaStreamDestroy(own);
  (void)cudaFreeHost(tick);
  (void)cudaFreeHost(stamps);
  return ok;
}

template <typename T>
bool Number(std::string_view text, T& out) {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), out);
  return error == std::errc() && end == text.data() + text.size();
}

}  // namespace

int main(int argc, char** argv) {
  const std::span<char*> args(argv, static_cast<std::size_t>(argc));
  if (args.size() < 2) {
    std::println(stderr, "usage: llmp_wake_bench chain|node [options] (see the header)");
    return 2;
  }
  const std::string_view mode = args[1];
  ChainOptions chain;
  NodeOptions node;
  for (std::size_t i = 2; i < args.size(); ++i) {
    const std::string_view a = args[i];
    const auto value = [&]() -> std::string_view {
      return i + 1 < args.size() ? std::string_view(args[++i]) : std::string_view();
    };
    bool ok = true;
    if (a == "--steps") {
      ok = Number(value(), chain.steps);
      node.steps = chain.steps;
    } else if (a == "--step-us") {
      ok = Number(value(), chain.step_us);
      node.step_us = chain.step_us;
    } else if (a == "--host-us") {
      ok = Number(value(), chain.host_us);
      node.host_us = chain.host_us;
    } else if (a == "--repeats") {
      ok = Number(value(), chain.repeats);
      node.repeats = chain.repeats;
    } else if (a == "--poll-us") {
      ok = Number(value(), node.poll_us);
    } else if (a == "--spin-ahead-us") {
      std::uint32_t us = 0;
      ok = Number(value(), us) && us <= 10000;
      chain.spin_ahead = microseconds(us);
      node.spin_ahead = us;
    } else if (a == "--mix") {
      chain.mix = true;
    } else if (a == "--only") {
      std::string_view list = value();
      while (!list.empty()) {
        const std::size_t comma = list.find(',');
        chain.only.emplace_back(list.substr(0, comma));
        list = comma == std::string_view::npos ? std::string_view() : list.substr(comma + 1);
      }
    } else {
      ok = false;
    }
    if (!ok || chain.steps < 2) {
      std::println(stderr, "bad argument: {}", a);
      return 2;
    }
  }
  if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    std::println(stderr, "no CUDA device");
    return 1;
  }
  if (mode == "node") {
    std::println(
        "| Runtime path | Steps | Round trip p50 / p99 / max (us) | Call to job | Call to GPU | "
        "GPU "
        "to return (p50 / p99, us) | Device per step (ms) | Cores stepping | Cores idle |");
    std::println("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    return RunNode(node) ? 0 : 1;
  }
  if (mode != "chain") {
    std::println(stderr, "unknown mode {}", mode);
    return 2;
  }
  std::println("steps {} x {} us{}, host work {} us, spin ahead {} us", chain.steps, chain.step_us,
               chain.mix ? " (mixed with 1/9)" : "", chain.host_us, chain.spin_ahead.count());
  PrintHeader();
  for (int repeat = 0; repeat < chain.repeats; ++repeat) {
    for (const Config& config : kConfigs) {
      if (!chain.only.empty() && std::ranges::none_of(chain.only, [&](const std::string& prefix) {
            return config.name.starts_with(prefix);
          })) {
        continue;
      }
      if (!RunChain(config, chain)) {
        std::println(stderr, "{} failed", config.name);
        return 1;
      }
    }
  }
  return 0;
}
