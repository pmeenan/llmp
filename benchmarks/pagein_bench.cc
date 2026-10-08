// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Page-in throughput through the scheduler and its lanes (D-081, BP-P6;
// docs/experiments/pagein-perf/): a file of 2 MiB extents loaded by one
// task that materializes them all, through the landing zone into device
// VMM (the runtime's path) or read in place into host VMM (D-034's path),
// with backing mapped once or made on each load (D-033). Every lane runs
// on its own thread, as a program wires them.
//
//   llmp_pagein_bench --file PATH | --dir DIR  [--gib N | --mib N] [--offset BYTES]
//                       [--mode zone|inplace] [--backing premapped|managed]
//                       [--depth D] [--slots S] [--loads K] [--verify]
//                       [--vmm-lane on|off] [--submit-poll US] [--storage-poll US]
//   llmp_pagein_bench ... --decode STEPS [--gpu-mib M] [--host-us H] [--pagein-ms P]
//
// --file reads an existing file of dmabuf_probe's pattern
// (docs/experiments/dmabuf-direct/), so the standalone probe and this
// harness read the same bytes; --dir writes an unnamed one (O_TMPFILE)
// there first. Each process loads --gib GiB (or --mib MiB) from --offset
// (4 KiB-aligned; an artifact's chunks start at 4 KiB multiples) K times,
// evicting everything between loads (untimed); a load runs from posting
// its task to the task's finish. --verify checks every extent against the
// pattern after each load (untimed); a premapped destination is cleared
// before each load, so a missing copy shows. The zone has 2 x depth slots
// unless --slots says otherwise. --vmm-lane off gives VMM work to the
// device lane, and --submit-poll and --storage-poll set the submission and
// storage lanes' poll windows (for ablation). Prints one line per load:
// GB/s; the per-extent latency from its read's publication to its
// publication as resident (p50, p99, max); and, with managed backing, how
// long before its read each extent's backing was mapped (p10, p50), from
// the scheduler's page-in observer.
//
// --decode STEPS runs a decode-like loop instead, to see what the lanes'
// polling costs between sparse page-ins: one task runs STEPS steps, each a
// device job on the first extent (leased) that spins H us on the host (the
// step's host-side work) and then queues a memset of M MiB, the next step
// starting once its fence completes. Every P ms (none by default) this
// thread posts a load of the next extent of the file, paged as above, while
// the steps run. Prints the steps' latency (submission to the task's wake:
// p50, p99, max) and the CPU time of each lane's thread over the loop, as
// a share of its wall time.

#include <cuda.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <expected>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/wake.h"
#include "catalog/catalog.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/direct_reader.h"
#include "providers/uring_storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/scheduler.h"
#include "scheduler/services.h"

namespace {

namespace sc = llmp::scheduler;
using llmp::base::Bytes;
using llmp::catalog::ExtentId;
using llmp::providers::BackingKind;
using llmp::providers::ReservationId;
using Clock = std::chrono::steady_clock;
using Status = std::expected<void, std::string>;

constexpr std::uint64_t kExtent = std::uint64_t{2} << 20U;
constexpr std::uint64_t kGiB = std::uint64_t{1} << 30U;
constexpr auto kPatience = std::chrono::seconds(120);

std::unexpected<std::string> Error(std::string message) {
  return std::unexpected(std::move(message));
}

// dmabuf_probe's pattern: the file's 32-bit word i.
std::uint32_t Pattern(std::uint64_t index) {
  auto x =
      static_cast<std::uint32_t>(index) ^ static_cast<std::uint32_t>(index >> 32U) ^ 0x9e3779b9U;
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  x *= 0x846ca68bU;
  return x ^ (x >> 16U);
}

struct Options {
  std::string file;
  std::string dir;
  std::uint64_t gib = 8;
  std::uint64_t mib = 0;     // overrides --gib
  std::uint64_t offset = 0;  // where in the file the first extent starts (4 KiB-aligned)
  bool zone = true;
  bool premapped = true;
  std::size_t depth = 4;
  std::size_t slots = 0;  // 2 x depth unless given
  int loads = 5;
  bool verify = false;
  bool vmm_lane = true;                         // a BackingService for VMM work
  std::chrono::microseconds submit_poll{200};   // DeviceSettings::poll_window
  std::chrono::microseconds storage_poll{200};  // StorageService's poll window
  int decode = 0;                               // steps of the decode loop; 0 loads instead
  std::uint64_t gpu_mib = 64;                   // each step's memset
  std::int64_t host_us = 100;                   // each step's host work, in its job
  std::int64_t pagein_ms = 0;                   // a page-in every P ms during the loop
};

std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options o;
  const auto number = [](std::string_view v, auto& out) {
    return std::from_chars(v.data(), v.data() + v.size(), out).ec == std::errc{};
  };
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view a = args[i];
    if (a == "--verify") {
      o.verify = true;
      continue;
    }
    if (i + 1 >= args.size()) {
      return Error(std::format("{} needs a value", a));
    }
    const std::string_view v = args[++i];
    bool ok = true;
    if (a == "--file") {
      o.file = v;
    } else if (a == "--dir") {
      o.dir = v;
    } else if (a == "--gib") {
      ok = number(v, o.gib) && o.gib > 0;
    } else if (a == "--mib") {
      ok = number(v, o.mib) && o.mib >= 2 && o.mib % 2 == 0;
    } else if (a == "--offset") {
      ok = number(v, o.offset) && o.offset % 4096 == 0;
    } else if (a == "--mode") {
      ok = v == "zone" || v == "inplace";
      o.zone = v == "zone";
    } else if (a == "--backing") {
      ok = v == "premapped" || v == "managed";
      o.premapped = v == "premapped";
    } else if (a == "--depth") {
      ok = number(v, o.depth) && o.depth > 0 && o.depth <= 64;
    } else if (a == "--slots") {
      ok = number(v, o.slots) && o.slots > 0 && o.slots <= 256;
    } else if (a == "--loads") {
      ok = number(v, o.loads) && o.loads > 0;
    } else if (a == "--vmm-lane") {
      ok = v == "on" || v == "off";
      o.vmm_lane = v == "on";
    } else if (a == "--submit-poll") {
      std::int64_t us = 0;
      ok = number(v, us) && us >= 0 && us <= 1000000;
      o.submit_poll = std::chrono::microseconds(us);
    } else if (a == "--storage-poll") {
      std::int64_t us = 0;
      ok = number(v, us) && us >= 0 && us <= 1000000;
      o.storage_poll = std::chrono::microseconds(us);
    } else if (a == "--decode") {
      ok = number(v, o.decode) && o.decode > 0;
    } else if (a == "--gpu-mib") {
      ok = number(v, o.gpu_mib) && o.gpu_mib > 0 && o.gpu_mib <= 4096;
    } else if (a == "--host-us") {
      ok = number(v, o.host_us) && o.host_us >= 0 && o.host_us <= 1000000;
    } else if (a == "--pagein-ms") {
      ok = number(v, o.pagein_ms) && o.pagein_ms >= 0 && o.pagein_ms <= 60000;
    } else {
      ok = false;
    }
    if (!ok) {
      return Error(std::format("bad argument {} {}", a, v));
    }
  }
  if (o.file.empty() == o.dir.empty()) {
    return Error(
        "usage: llmp_pagein_bench --file PATH | --dir DIR [--gib N] [--mode zone|inplace] "
        "[--backing premapped|managed] [--depth D] [--slots S] [--loads K] [--verify] "
        "[--vmm-lane on|off] [--submit-poll US] [--storage-poll US] "
        "[--decode STEPS [--gpu-mib M] [--host-us H] [--pagein-ms P]]");
  }
  if (o.slots == 0) {
    o.slots = 2 * o.depth;
  }
  return o;
}

// Writes `bytes` of the pattern to an unnamed direct-I/O file in `dir`.
std::expected<int, std::string> CreateFile(const std::string& dir, std::uint64_t bytes) {
  const int fd = ::open(dir.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
  if (fd < 0) {
    return Error(std::format("O_TMPFILE in {}: {}", dir, std::generic_category().message(errno)));
  }
  constexpr std::uint64_t kChunk = std::uint64_t{64} << 20U;
  void* buffer = std::aligned_alloc(4096, kChunk);
  auto* words = static_cast<std::uint32_t*>(buffer);
  for (std::uint64_t at = 0; at < bytes; at += kChunk) {
    const std::uint64_t first = at / 4;
    std::vector<std::jthread> workers;
    workers.reserve(8);
    for (std::uint64_t k = 0; k < 8; ++k) {
      workers.emplace_back([&, k] {
        for (std::uint64_t i = k * (kChunk / 32); i < (k + 1) * (kChunk / 32); ++i) {
          words[i] = Pattern(first + i);
        }
      });
    }
    workers.clear();
    if (::pwrite(fd, buffer, kChunk, static_cast<off_t>(at)) != static_cast<ssize_t>(kChunk)) {
      std::free(buffer);
      return Error("writing the file");
    }
  }
  std::free(buffer);
  if (::fsync(fd) != 0) {
    return Error("fsync");
  }
  return fd;
}

// Counts words of `span` (starting at file word `first`) off the pattern.
std::uint64_t Check(const std::uint32_t* words, std::uint64_t count, std::uint64_t first) {
  std::atomic<std::uint64_t> bad{0};
  std::vector<std::jthread> workers;
  constexpr std::uint64_t kWorkers = 8;
  workers.reserve(kWorkers);
  for (std::uint64_t k = 0; k < kWorkers; ++k) {
    workers.emplace_back([&, k] {
      std::uint64_t mine = 0;
      const std::uint64_t begin = count * k / kWorkers;
      const std::uint64_t end = count * (k + 1) / kWorkers;
      for (std::uint64_t i = begin; i < end; ++i) {
        mine += words[i] != Pattern(first + i) ? 1 : 0;
      }
      bad.fetch_add(mine);
    });
  }
  workers.clear();
  return bad.load();
}

double Percentile(std::vector<double> samples, double p) {
  if (samples.empty()) {
    return 0;
  }
  std::ranges::sort(samples);
  return samples[static_cast<std::size_t>(p * static_cast<double>(samples.size() - 1))];
}

struct Done {
  std::atomic<int> outcome{-1};
  std::atomic<bool> gone{false};
  std::atomic<std::int64_t> finished_ns{0};
};

std::int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
      .count();
}

class Program : public sc::TaskProgram {
 public:
  explicit Program(Done& done) : done_(done) {}
  Program(const Program&) = delete;
  Program& operator=(const Program&) = delete;
  Program(Program&&) = delete;
  Program& operator=(Program&&) = delete;
  ~Program() override { done_.gone.store(true); }
  void Finished(sc::TaskOutcome outcome) override {
    done_.finished_ns.store(NowNs());
    done_.outcome.store(static_cast<int>(outcome));
  }

 protected:
  Done& done_;
};

// Materializes every extent at once and waits for them.
class LoadProgram final : public Program {
 public:
  LoadProgram(Done& done, llmp::catalog::Closure closure)
      : Program(done), closure_(std::move(closure)) {}
  sc::Step Advance(sc::TaskContext& context) override {
    if (context.TakeFailure()) {
      return sc::Step::Finish(sc::TaskOutcome::kFailed);
    }
    const auto ready = context.Materialize(closure_);
    if (!ready) {
      return ready.error() == sc::WorkError::kBusy ? sc::Step::Yield()
                                                   : sc::Step::Finish(sc::TaskOutcome::kFailed);
    }
    return *ready == sc::Readiness::kWaiting ? sc::Step::Wait()
                                             : sc::Step::Finish(sc::TaskOutcome::kSucceeded);
  }

 private:
  llmp::catalog::Closure closure_;
};

// Evicts every resident extent, waiting once for all the unmaps.
class EvictProgram final : public Program {
 public:
  EvictProgram(Done& done, std::vector<ExtentId> extents)
      : Program(done), extents_(std::move(extents)) {}
  sc::Step Advance(sc::TaskContext& context) override {
    if (context.TakeFailure()) {
      return sc::Step::Finish(sc::TaskOutcome::kFailed);
    }
    bool waiting = false;
    while (next_ < extents_.size()) {
      const ExtentId extent = extents_[next_];
      if (context.catalog().Describe(extent).value().state !=
          llmp::catalog::ExtentState::kResident) {
        ++next_;
        continue;
      }
      const auto evicted = context.Evict(extent);
      if (!evicted) {
        if (evicted.error() == sc::WorkError::kBusy) {
          return waiting ? sc::Step::Wait() : sc::Step::Yield();
        }
        return sc::Step::Finish(sc::TaskOutcome::kFailed);
      }
      waiting = waiting || *evicted == sc::Readiness::kWaiting;
      ++next_;
    }
    if (waiting) {
      return sc::Step::Wait();
    }
    return sc::Step::Finish(sc::TaskOutcome::kSucceeded);
  }

 private:
  std::vector<ExtentId> extents_;
  std::size_t next_ = 0;
};

// The decode-like loop: a step is one device job over `weights` (leased,
// materialized first); the next is submitted once its fence completes.
// Records each step's submission and the task's wake after it.
class DecodeProgram final : public Program {
 public:
  DecodeProgram(Done& done, llmp::catalog::Closure weights, int steps,
                std::function<sc::DeviceJob()> make, std::vector<std::int64_t>& starts,
                std::vector<std::int64_t>& ends)
      : Program(done),
        weights_(std::move(weights)),
        steps_(static_cast<std::size_t>(steps)),
        make_(std::move(make)),
        starts_(starts),
        ends_(ends) {}
  sc::Step Advance(sc::TaskContext& context) override {
    if (context.TakeFailure()) {
      return sc::Step::Finish(sc::TaskOutcome::kFailed);
    }
    if (!ready_) {
      const auto ready = context.Materialize(weights_);
      if (!ready) {
        return ready.error() == sc::WorkError::kBusy ? sc::Step::Yield()
                                                     : sc::Step::Finish(sc::TaskOutcome::kFailed);
      }
      if (*ready == sc::Readiness::kWaiting) {
        return sc::Step::Wait();
      }
      ready_ = true;
    }
    if (submitted_) {
      ends_.push_back(NowNs());
      submitted_ = false;
    }
    if (ends_.size() >= steps_) {
      return sc::Step::Finish(sc::TaskOutcome::kSucceeded);
    }
    const auto submitted =
        context.SubmitLaunch(weights_, sc::LaunchWork{.stream = 0, .job = make_()});
    if (!submitted) {
      return submitted.error() == sc::WorkError::kBusy ? sc::Step::Yield()
                                                       : sc::Step::Finish(sc::TaskOutcome::kFailed);
    }
    starts_.push_back(NowNs());
    submitted_ = true;
    return sc::Step::Wait();
  }

 private:
  llmp::catalog::Closure weights_;
  std::size_t steps_;
  std::function<sc::DeviceJob()> make_;
  std::vector<std::int64_t>& starts_;
  std::vector<std::int64_t>& ends_;
  bool ready_ = false;
  bool submitted_ = false;
};

// Records, on the scheduler thread, when each extent's read was published
// and when it was published resident.
class Observer final : public sc::PageInObserver {
 public:
  explicit Observer(std::size_t extents) : mapped_(extents), read_(extents), resident_(extents) {}
  void Staged(ExtentId extent, sc::PageInEvent event) override {
    const std::int64_t now = NowNs();
    if (extent.index() >= read_.size()) {
      return;
    }
    if (event == sc::PageInEvent::kMapped) {
      mapped_[extent.index()] = now;
    } else if (event == sc::PageInEvent::kReading) {
      read_[extent.index()] = now;
    } else if (event == sc::PageInEvent::kResident) {
      resident_[extent.index()] = now;
    }
  }
  // Microseconds from mapped backing to the read's publication: how far
  // ahead of its slot each load's backing was ready.
  std::vector<double> MapLeads() const {
    std::vector<double> out;
    for (std::size_t i = 0; i < read_.size(); ++i) {
      if (mapped_[i] != 0 && read_[i] >= mapped_[i]) {
        out.push_back(static_cast<double>(read_[i] - mapped_[i]) / 1e3);
      }
    }
    return out;
  }
  // Microseconds from read to resident.
  std::vector<double> Latencies() const {
    std::vector<double> out;
    for (std::size_t i = 0; i < read_.size(); ++i) {
      if (read_[i] != 0 && resident_[i] >= read_[i]) {
        out.push_back(static_cast<double>(resident_[i] - read_[i]) / 1e3);
      }
    }
    return out;
  }
  void Clear() {
    std::ranges::fill(mapped_, 0);
    std::ranges::fill(read_, 0);
    std::ranges::fill(resident_, 0);
  }

 private:
  std::vector<std::int64_t> mapped_;
  std::vector<std::int64_t> read_;
  std::vector<std::int64_t> resident_;
};

class Bench {
 public:
  explicit Bench(const Options& o) : o_(o) {}
  Bench(const Bench&) = delete;
  Bench& operator=(const Bench&) = delete;
  Bench(Bench&&) = delete;
  Bench& operator=(Bench&&) = delete;
  ~Bench() { std::ignore = Teardown(); }

  Status Run();
  Status Teardown();

 private:
  Status Setup();
  void Start(std::unique_ptr<sc::TaskProgram> program);
  Status Post(std::unique_ptr<sc::TaskProgram> program, Done& done);
  Status Verify(int load);
  Status Decode();
  // Each lane thread's CPU time so far, in seconds (threads_'s order).
  std::vector<double> ThreadSeconds();

  const Options& o_;
  CUcontext context_ = nullptr;
  CUdevice device_ = 0;
  int fd_ = -1;
  std::uint64_t extents_ = 0;
  std::unique_ptr<llmp::providers::VmmProvider> memory_;
  std::unique_ptr<llmp::providers::DeviceExecution> execution_;
  std::unique_ptr<llmp::providers::UringStorage> storage_;
  llmp::providers::StreamId stream_;
  std::size_t device_class_ = 0;
  std::size_t host_class_ = 0;
  ReservationId zone_;
  std::uint64_t zone_base_ = 0;
  std::vector<llmp::providers::BackingId> zone_backings_;
  ReservationId destination_;
  std::uint64_t destination_base_ = 0;
  std::vector<llmp::providers::BackingId> premapped_;

  llmp::catalog::Catalog catalog_;
  std::vector<ExtentId> extent_ids_;
  llmp::base::WakeFlag wake_;
  std::unique_ptr<Observer> observer_;
  std::unique_ptr<sc::CompletionBoard> board_;
  std::unique_ptr<sc::StorageService> storage_lane_;
  std::unique_ptr<sc::DeviceService> device_lane_;
  std::unique_ptr<sc::BackingService> backing_lane_;
  std::unique_ptr<sc::Scheduler> scheduler_;
  std::vector<std::jthread> threads_;
  std::optional<std::expected<void, sc::Fault>> stopped_;
  std::uint64_t request_ = 0;
  void* check_buffer_ = nullptr;
  CUdeviceptr decode_buffer_ = 0;  // what each decode step's memset writes
  std::uint32_t decode_value_ = 0;
  bool torn_down_ = false;
};

Status Bench::Setup() {
  const std::uint64_t bytes = o_.mib != 0 ? o_.mib << 20U : o_.gib * kGiB;
  extents_ = bytes / kExtent;
  if (!o_.dir.empty()) {
    std::println("writing {} GiB of the pattern to an unnamed file in {}", o_.gib, o_.dir);
    auto fd = CreateFile(o_.dir, o_.offset + bytes + kExtent);
    if (!fd) {
      return std::unexpected(fd.error());
    }
    fd_ = *fd;
  } else {
    fd_ = ::open(o_.file.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd_ < 0) {
      return Error(std::format("{}: {}", o_.file, std::generic_category().message(errno)));
    }
    if (::lseek(fd_, 0, SEEK_END) < static_cast<off_t>(o_.offset + bytes)) {
      return Error("the file is shorter than --gib");
    }
  }
  CUdevice device = 0;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&context_, device) != CUDA_SUCCESS ||
      cuCtxSetCurrent(context_) != CUDA_SUCCESS) {
    return Error("no CUDA device");
  }
  device_ = device;
  auto memory = llmp::providers::cuda::OpenDeviceMemory(0);
  auto execution = llmp::providers::cuda::OpenDeviceExecution(0);
  if (!memory || !execution) {
    return Error("the CUDA providers");
  }
  memory_ = std::move(*memory);
  execution_ = std::move(*execution);
  auto stream = execution_->CreateStream();
  if (!stream) {
    return Error("CreateStream");
  }
  stream_ = *stream;
  for (std::size_t i = 0; i < memory_->Classes().size(); ++i) {
    (memory_->Classes()[i].kind == BackingKind::kDevice ? device_class_ : host_class_) = i;
  }
  auto storage = llmp::providers::UringStorage::Create(o_.depth);
  if (!storage) {
    return Error(std::format("io_uring: {}", storage.error().message()));
  }
  storage_ = std::move(*storage);

  // The zone: host VMM the CPU and the device map.
  const auto map_all = [&](ReservationId& reservation, std::uint64_t& base, std::uint64_t bytes,
                           std::size_t allocation_class,
                           std::vector<llmp::providers::BackingId>* backings) -> Status {
    auto reserved = memory_->Reserve(Bytes(bytes));
    if (!reserved) {
      return Error("Reserve");
    }
    reservation = *reserved;
    base = memory_->RangeOf(reservation).value().base;
    if (backings == nullptr) {
      return {};
    }
    for (std::uint64_t at = 0; at < bytes; at += kExtent) {
      auto backing = memory_->Create(allocation_class, Bytes(kExtent));
      if (!backing || !memory_->Map(reservation, Bytes(at), *backing)) {
        return Error("Create or Map");
      }
      backings->push_back(*backing);
    }
    if (!memory_->SetAccess(reservation, Bytes(0), Bytes(bytes),
                            llmp::providers::Access::kReadWrite)) {
      return Error("SetAccess");
    }
    return {};
  };
  if (auto r = map_all(zone_, zone_base_, o_.slots * kExtent, host_class_, &zone_backings_); !r) {
    return r;
  }
  const std::size_t destination_class = o_.zone ? device_class_ : host_class_;
  if (auto r = map_all(destination_, destination_base_, extents_ * kExtent, destination_class,
                       o_.premapped ? &premapped_ : nullptr);
      !r) {
    return r;
  }

  const auto domain = catalog_.AddDomain("gb10");
  observer_ = std::make_unique<Observer>(extents_ + 16);
  const std::size_t operations = extents_ + 64;
  board_ = std::make_unique<sc::CompletionBoard>(operations, wake_);
  storage_lane_ = std::make_unique<sc::StorageService>(
      *storage_,
      llmp::providers::ReaderSettings{.alignment = 4096,
                                      .request_bytes = 2U << 20U,
                                      .retries = 3,
                                      .reads = operations,
                                      .waiters = 8,
                                      .span_bytes = llmp::providers::kNoCoalescing,
                                      .span_segments = llmp::providers::kMaxSegments},
      *board_, sc::QueueSettings{.capacity = 256, .reserved = 16, .batch = 32}, o_.storage_poll);
  const std::array<llmp::providers::StreamId, 1> streams = {stream_};
  device_lane_ = std::make_unique<sc::DeviceService>(
      *execution_, streams, *board_,
      sc::DeviceSettings{.queue = {.capacity = 256, .reserved = 16, .batch = 32},
                         .handoff = 256,
                         .poll_window = o_.submit_poll},
      o_.vmm_lane ? nullptr : memory_.get());  // one lane calls the provider
  if (o_.vmm_lane) {
    backing_lane_ = std::make_unique<sc::BackingService>(
        memory_.get(), *board_, sc::QueueSettings{.capacity = 256, .reserved = 16, .batch = 32});
  }
  sc::LandingZone landing{.slots = {}, .slot_bytes = Bytes(kExtent), .stream = 0};
  for (std::size_t i = 0; i < o_.slots; ++i) {
    landing.slots.push_back(zone_base_ + (i * kExtent));
  }
  sc::SchedulerSettings settings{.tasks = 16, .budget = Bytes(std::uint64_t{120} << 30U)};
  settings.landing = landing;
  settings.observer = observer_.get();
  scheduler_ = std::make_unique<sc::Scheduler>(catalog_, *board_, wake_,
                                               sc::Lanes{.storage = storage_lane_.get(),
                                                         .device = device_lane_.get(),
                                                         .cpu = nullptr,
                                                         .backing = backing_lane_.get()},
                                               settings);
  for (std::uint64_t i = 0; i < extents_; ++i) {
    auto extent = catalog_.AddExtent(
        {.domain = domain,
         .memory_class = llmp::catalog::MemoryClass::kWeights,
         .recovery = llmp::catalog::Recovery::kFromArtifact,
         .size = Bytes(kExtent),
         .content = {.artifact = {}, .group = 0, .chunk = static_cast<std::uint32_t>(i)}});
    if (!extent) {
      return Error("AddExtent");
    }
    extent_ids_.push_back(*extent);
    const std::uint64_t address = destination_base_ + (i * kExtent);
    std::optional<sc::BackingPlace> backing;
    if (!o_.premapped) {
      backing = sc::BackingPlace{.reservation = destination_,
                                 .offset = Bytes(i * kExtent),
                                 .size = Bytes(kExtent),
                                 .allocation_class = destination_class};
    }
    auto set = scheduler_->SetSource(
        *extent,
        sc::PageSource{.read = {.fd = fd_,
                                .offset = o_.offset + (i * kExtent),
                                // NOLINTNEXTLINE(performance-no-int-to-ptr): host VMM, CPU-mapped
                                .memory = o_.zone ? nullptr : reinterpret_cast<std::byte*>(address),
                                .length = kExtent},
                       .landed = o_.zone,
                       .destination = o_.zone ? address : 0,
                       .backing = backing});
    if (!set) {
      return Error(std::format("SetSource: {}", sc::ToString(set.error())));
    }
  }
  if (o_.verify && o_.zone && cuMemAllocHost(&check_buffer_, 64 * kExtent) != CUDA_SUCCESS) {
    return Error("the check buffer");
  }
  if (o_.decode > 0 && cuMemAlloc(&decode_buffer_, o_.gpu_mib << 20U) != CUDA_SUCCESS) {
    return Error("the decode buffer");
  }
  threads_.emplace_back([this] { stopped_ = scheduler_->Run(); });
  threads_.emplace_back([this] { storage_lane_->Run(); });
  threads_.emplace_back([this] { device_lane_->RunSubmission(); });
  threads_.emplace_back([this] { device_lane_->RunCompletion(); });
  if (backing_lane_) {
    threads_.emplace_back([this] { backing_lane_->Run(); });
  }
  return {};
}

void Bench::Start(std::unique_ptr<sc::TaskProgram> program) {
  sc::Control start =
      sc::StartRequest{.request = ++request_, .priority = 1, .program = std::move(program)};
  // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
  while (scheduler_->Post(std::move(start)) == llmp::base::PushResult::kFull) {
    std::this_thread::yield();
  }
}

Status Bench::Post(std::unique_ptr<sc::TaskProgram> program, Done& done) {
  Start(std::move(program));
  const auto give_up = Clock::now() + kPatience;
  while (!done.gone.load()) {
    if (Clock::now() > give_up) {
      std::println(stderr, "a task did not finish: aborting");
      std::abort();
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  if (done.outcome.load() != static_cast<int>(sc::TaskOutcome::kSucceeded)) {
    return Error("a task failed");
  }
  return {};
}

Status Bench::Verify(int load) {
  std::uint64_t bad = 0;
  if (!o_.zone) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr): host VMM, CPU-mapped
    bad = Check(reinterpret_cast<const std::uint32_t*>(destination_base_), extents_ * kExtent / 4,
                o_.offset / 4);
  } else {
    for (std::uint64_t at = 0; at < extents_ * kExtent; at += 64 * kExtent) {
      const std::uint64_t bytes = std::min(64 * kExtent, (extents_ * kExtent) - at);
      if (cuMemcpyDtoH(check_buffer_, destination_base_ + at, bytes) != CUDA_SUCCESS) {
        return Error("copying back");
      }
      bad +=
          Check(static_cast<const std::uint32_t*>(check_buffer_), bytes / 4, (o_.offset + at) / 4);
    }
  }
  if (bad != 0) {
    return Error(std::format("load {}: {} words differ from the file", load, bad));
  }
  return {};
}

std::vector<double> Bench::ThreadSeconds() {
  std::vector<double> out;
  for (std::jthread& thread : threads_) {
    clockid_t clock = 0;
    timespec now{};
    if (pthread_getcpuclockid(thread.native_handle(), &clock) != 0 ||
        clock_gettime(clock, &now) != 0) {
      out.push_back(0);
      continue;
    }
    out.push_back(static_cast<double>(now.tv_sec) + (static_cast<double>(now.tv_nsec) / 1e9));
  }
  return out;
}

Status Bench::Decode() {
  const auto weights = catalog_.ClosureOfExtents(std::vector{extent_ids_.front()}).value();
  const auto make = [this]() -> sc::DeviceJob {
    return [this](llmp::providers::NativeStream stream) {
      const auto until = Clock::now() + std::chrono::microseconds(o_.host_us);
      while (Clock::now() < until) {
        // the step's host-side work
      }
      if (cuCtxSetCurrent(context_) != CUDA_SUCCESS) {
        return sc::JobResult::kNotStarted;  // nothing was queued
      }
      if (cuMemsetD32Async(decode_buffer_, ++decode_value_, (o_.gpu_mib << 20U) / 4,
                           static_cast<CUstream>(stream.handle)) != CUDA_SUCCESS) {
        return sc::JobResult::kUnknown;  // a driver error: its effect is unknown
      }
      return sc::JobResult::kQueued;
    };
  };
  std::vector<std::int64_t> starts;
  std::vector<std::int64_t> ends;
  starts.reserve(static_cast<std::size_t>(o_.decode));
  ends.reserve(static_cast<std::size_t>(o_.decode));
  // The weights first, so the loop's first step does not wait for a load.
  Done loaded;
  if (auto r = Post(std::make_unique<LoadProgram>(loaded, weights), loaded); !r) {
    return r;
  }
  observer_->Clear();
  const std::vector<double> before = ThreadSeconds();
  rusage process_before{};
  (void)getrusage(RUSAGE_SELF, &process_before);
  const std::int64_t start = NowNs();
  Done done;
  Start(std::make_unique<DecodeProgram>(done, weights, o_.decode, make, starts, ends));
  std::deque<Done> pageins;
  std::size_t next = 1;
  const auto every = std::chrono::milliseconds(o_.pagein_ms);
  auto due = Clock::now() + every;
  const auto give_up = Clock::now() + kPatience;
  while (!done.gone.load()) {
    if (Clock::now() > give_up) {
      std::println(stderr, "the decode loop did not finish: aborting");
      std::abort();
    }
    if (o_.pagein_ms > 0 && Clock::now() >= due && next < extent_ids_.size()) {
      Done& pagein = pageins.emplace_back();
      Start(std::make_unique<LoadProgram>(
          pagein, catalog_.ClosureOfExtents(std::vector{extent_ids_[next++]}).value()));
      due += every;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  const double wall = static_cast<double>(done.finished_ns.load() - start) / 1e9;
  const std::vector<double> after = ThreadSeconds();
  rusage process_after{};
  (void)getrusage(RUSAGE_SELF, &process_after);
  for (const Done& pagein : pageins) {
    while (!pagein.gone.load()) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    if (pagein.outcome.load() != static_cast<int>(sc::TaskOutcome::kSucceeded)) {
      return Error("a page-in failed");
    }
  }
  if (done.outcome.load() != static_cast<int>(sc::TaskOutcome::kSucceeded) ||
      ends.size() != starts.size()) {
    return Error("the decode loop failed");
  }
  std::vector<double> steps;
  steps.reserve(ends.size());
  for (std::size_t i = 0; i < ends.size(); ++i) {
    steps.push_back(static_cast<double>(ends[i] - starts[i]) / 1e3);
  }
  const std::vector<double> pagein_latencies = observer_->Latencies();
  std::println(
      "decode,steps,{},wall_s,{:.3f},step_p50_us,{:.0f},step_p99_us,{:.0f},step_max_us,{:.0f},"
      "pageins,{},pagein_p50_us,{:.0f},pagein_p99_us,{:.0f}",
      ends.size(), wall, Percentile(steps, 0.5), Percentile(steps, 0.99), Percentile(steps, 1.0),
      pageins.size(), Percentile(pagein_latencies, 0.5), Percentile(pagein_latencies, 0.99));
  constexpr std::array<std::string_view, 5> kNames = {"scheduler", "storage", "submission",
                                                      "completion", "vmm"};
  for (std::size_t i = 0; i < after.size() && i < kNames.size(); ++i) {
    std::println("cpu,{},{:.3f},{:.3f}", kNames.at(i), after[i] - before[i],
                 (after[i] - before[i]) / wall);
  }
  const auto seconds = [](const timeval& t) {
    return static_cast<double>(t.tv_sec) + (static_cast<double>(t.tv_usec) / 1e6);
  };
  const double process = seconds(process_after.ru_utime) + seconds(process_after.ru_stime) -
                         seconds(process_before.ru_utime) - seconds(process_before.ru_stime);
  std::println("cpu,process,{:.3f},{:.3f}", process, process / wall);
  return {};
}

Status Bench::Run() {
  if (auto r = Setup(); !r) {
    return r;
  }
  if (o_.decode > 0) {
    std::println(
        "# decode mode {} backing {} steps {} gpu_mib {} host_us {} pagein_ms {} "
        "submit_poll_us {} storage_poll_us {}",
        o_.zone ? "zone" : "inplace", o_.premapped ? "premapped" : "managed", o_.decode, o_.gpu_mib,
        o_.host_us, o_.pagein_ms, o_.submit_poll.count(), o_.storage_poll.count());
    return Decode();
  }
  const auto closure = catalog_.ClosureOfExtents(extent_ids_).value();
  std::println("# mode {} backing {} depth {} slots {} extents {} bytes {}",
               o_.zone ? "zone" : "inplace", o_.premapped ? "premapped" : "managed", o_.depth,
               o_.slots, extents_, extents_ * kExtent);
  std::println("# load,gbps,seconds,p50_us,p99_us,max_us,verified,map_lead_p10_us,map_lead_p50_us");
  for (int load = 1; load <= o_.loads; ++load) {
    if (o_.premapped && o_.zone) {
      if (cuMemsetD8(destination_base_, 0, extents_ * kExtent) != CUDA_SUCCESS ||
          cuCtxSynchronize() != CUDA_SUCCESS) {
        return Error("clearing the destination");
      }
    } else if (o_.premapped) {
      // NOLINTNEXTLINE(performance-no-int-to-ptr): host VMM, CPU-mapped
      std::memset(reinterpret_cast<void*>(destination_base_), 0, extents_ * kExtent);
    }
    observer_->Clear();
    Done done;
    const std::int64_t start = NowNs();
    if (auto r = Post(std::make_unique<LoadProgram>(done, closure), done); !r) {
      return r;
    }
    const double seconds = static_cast<double>(done.finished_ns.load() - start) / 1e9;
    const std::vector<double> latencies = observer_->Latencies();
    const std::vector<double> leads = observer_->MapLeads();
    bool verified = false;
    if (o_.verify) {
      if (auto r = Verify(load); !r) {
        return r;
      }
      verified = true;
    }
    std::println("load,{},{:.3f},{:.4f},{:.0f},{:.0f},{:.0f},{},{:.0f},{:.0f}", load,
                 static_cast<double>(extents_ * kExtent) / seconds / 1e9, seconds,
                 Percentile(latencies, 0.5), Percentile(latencies, 0.99),
                 Percentile(latencies, 1.0), verified ? "yes" : "no", Percentile(leads, 0.1),
                 Percentile(leads, 0.5));
    (void)std::fflush(stdout);
    Done evicted;
    if (auto r = Post(std::make_unique<EvictProgram>(evicted, extent_ids_), evicted); !r) {
      return r;
    }
  }
  return {};
}

Status Bench::Teardown() {
  if (torn_down_) {
    return {};
  }
  torn_down_ = true;
  std::vector<std::string> problems;
  if (scheduler_ != nullptr) {
    Done evicted;
    if (!Post(std::make_unique<EvictProgram>(evicted, extent_ids_), evicted)) {
      problems.emplace_back("the final eviction");
    }
    scheduler_->RequestShutdown();
    threads_.front().join();
    storage_lane_->Close();
    device_lane_->Close();
    if (backing_lane_) {
      backing_lane_->Close();
    }
    threads_.clear();
    if (!stopped_ || !stopped_->has_value()) {
      problems.emplace_back("the scheduler stopped with a fault");
    }
  }
  if (execution_ != nullptr && stream_.valid()) {
    std::ignore = execution_->DestroyStream(stream_);
  }
  if (memory_ != nullptr) {
    const auto release = [&](ReservationId reservation,
                             std::vector<llmp::providers::BackingId>& backings) {
      if (!reservation.valid()) {
        return;
      }
      if (!backings.empty()) {
        std::ignore = memory_->Unmap(reservation, Bytes(0), Bytes(backings.size() * kExtent));
        for (const auto backing : backings) {
          std::ignore = memory_->Release(backing);
        }
      }
      if (!memory_->Free(reservation)) {
        problems.emplace_back("a reservation still has mappings");
      }
    };
    release(zone_, zone_backings_);
    release(destination_, premapped_);
  }
  if (check_buffer_ != nullptr) {
    (void)cuMemFreeHost(check_buffer_);
  }
  if (decode_buffer_ != 0) {
    (void)cuMemFree(decode_buffer_);
  }
  if (fd_ >= 0) {
    (void)::close(fd_);
  }
  if (context_ != nullptr) {
    (void)cuDevicePrimaryCtxRelease(device_);
  }
  if (!problems.empty()) {
    std::string joined;
    for (const auto& p : problems) {
      joined += (joined.empty() ? "" : "; ") + p;
    }
    return Error(joined);
  }
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  const auto options = Parse(std::span<char*>(argv, static_cast<std::size_t>(argc)));
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  Status ran;
  {
    Bench bench(*options);
    ran = bench.Run();
    if (auto finished = bench.Teardown(); !finished && ran) {
      ran = finished;
    }
  }
  if (!ran) {
    std::println(stderr, "{}", ran.error());
    return 1;
  }
  return 0;
}
