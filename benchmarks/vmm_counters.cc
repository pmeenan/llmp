// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Which OS counters include VMM backing (docs/architecture.md#memory-breakdown,
// M2): creates GIB of backing of each kind (device-local, then host-NUMA)
// in 2 MiB extents through the CUDA provider, and at each step (reserve,
// create, map with access, touch, unmap, release) prints how far each
// counter moved from its value before the reservation, in MiB:
//   - every /proc/meminfo field that moves by at least 64 MiB at some step;
//   - the process's VmRSS, RssAnon, RssFile, RssShmem, VmLck and VmPin;
//   - its cgroup's memory.current and memory.stat anon, file, shmem, kernel;
//   - the driver's free memory (cuMemGetInfo).
// Touching writes every byte: host backing from the CPU, device backing
// with copies from a touched host extent. The report is under
// docs/experiments/vmm-counters/.
//
//   llmp_vmm_counters GIB

#include <cuda.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <fstream>
#include <map>
#include <memory>
#include <print>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"

namespace {

using llmp::base::Bytes;
using llmp::providers::Access;
using llmp::providers::BackingId;
using llmp::providers::BackingKind;
using llmp::providers::VmmProvider;

constexpr std::int64_t kMiB = std::int64_t{1} << 20;
using Counters = std::map<std::string, std::int64_t>;  // bytes

// "Key:   123 kB" lines.
void ReadKeyed(const std::string& path, const std::string& prefix, Counters& into) {
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    std::istringstream rest(line.substr(colon + 1));
    std::int64_t value = 0;
    std::string unit;
    if (rest >> value) {
      rest >> unit;
      into[prefix + line.substr(0, colon)] = unit == "kB" ? value * 1024 : value;
    }
  }
}

// "key value" lines.
void ReadPairs(const std::string& path, const std::string& prefix, Counters& into) {
  std::ifstream file(path);
  std::string key;
  std::int64_t value = 0;
  while (file >> key >> value) {
    into[prefix + key] = value;
  }
}

std::string CgroupPath() {
  std::ifstream file("/proc/self/cgroup");
  std::string line;
  while (std::getline(file, line)) {
    if (line.starts_with("0::")) {
      return "/sys/fs/cgroup" + line.substr(3);
    }
  }
  return {};
}

Counters Snapshot(const std::string& cgroup) {
  // Let the kernel's per-CPU counters settle.
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  Counters counters;
  ReadKeyed("/proc/meminfo", "meminfo.", counters);
  Counters status;
  ReadKeyed("/proc/self/status", "", status);
  for (const char* key : {"VmRSS", "RssAnon", "RssFile", "RssShmem", "VmLck", "VmPin"}) {
    counters[std::string("status.") + key] = status[key];
  }
  if (!cgroup.empty()) {
    std::ifstream current(cgroup + "/memory.current");
    std::int64_t value = 0;
    if (current >> value) {
      counters["cgroup.memory.current"] = value;
    }
    Counters stat;
    ReadPairs(cgroup + "/memory.stat", "", stat);
    for (const char* key : {"anon", "file", "shmem", "kernel"}) {
      counters[std::string("cgroup.stat.") + key] = stat[key];
    }
  }
  std::size_t free = 0;
  std::size_t total = 0;
  if (cuMemGetInfo(&free, &total) == CUDA_SUCCESS) {
    counters["cuda.free"] = static_cast<std::int64_t>(free);
  }
  return counters;
}

void* Pointer(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

// Reports a failed call.
bool Ok(const std::expected<void, llmp::providers::Failure>& result) {
  if (!result) {
    std::println(stderr, "{}", result.error().detail);
  }
  return result.has_value();
}

struct Phase {
  std::string name;
  Counters counters;
};

void Report(const char* kind, const std::vector<Phase>& phases) {
  const Counters& base = phases.front().counters;
  // meminfo fields that moved; everything else always.
  std::vector<std::string> keys;
  for (const auto& [key, value] : base) {
    bool moved = !key.starts_with("meminfo.");
    for (const Phase& phase : phases) {
      const auto found = phase.counters.find(key);
      if (found != phase.counters.end() && std::abs(found->second - value) >= 64 * kMiB) {
        moved = true;
      }
    }
    if (moved) {
      keys.push_back(key);
    }
  }
  std::print("\n## {} backing\n\n| Counter |", kind);
  for (std::size_t i = 1; i < phases.size(); ++i) {
    std::print(" {} |", phases[i].name);
  }
  std::print("\n| --- |");
  for (std::size_t i = 1; i < phases.size(); ++i) {
    std::print(" ---: |");
  }
  std::println();
  for (const std::string& key : keys) {
    std::print("| {} |", key);
    for (std::size_t i = 1; i < phases.size(); ++i) {
      const auto found = phases[i].counters.find(key);
      const std::int64_t delta =
          found != phases[i].counters.end() ? found->second - base.at(key) : 0;
      std::print(" {} |", delta / kMiB);
    }
    std::println();
  }
}

bool Run(VmmProvider& memory, llmp::providers::DeviceExecution& execution, BackingKind kind,
         std::uint64_t bytes, const std::string& cgroup) {
  std::size_t allocation_class = 0;
  for (std::size_t i = 0; i < memory.Classes().size(); ++i) {
    if (memory.Classes()[i].kind == kind) {
      allocation_class = i;
    }
  }
  const Bytes extent = memory.Granularity();
  const std::uint64_t count = bytes / extent.value();
  std::vector<Phase> phases;
  phases.push_back({"baseline", Snapshot(cgroup)});
  const auto reservation = memory.Reserve(Bytes(count * extent.value()));
  if (!reservation) {
    return false;
  }
  phases.push_back({"reserve", Snapshot(cgroup)});
  std::vector<BackingId> backings;
  for (std::uint64_t i = 0; i < count; ++i) {
    auto backing = memory.Create(allocation_class, extent);
    if (!backing) {
      std::println(stderr, "create: {}", backing.error().detail);
      return false;
    }
    backings.push_back(*backing);
  }
  phases.push_back({"create", Snapshot(cgroup)});
  for (std::uint64_t i = 0; i < count; ++i) {
    if (!memory.Map(*reservation, Bytes(i * extent.value()), backings[i])) {
      return false;
    }
  }
  if (!memory.SetAccess(*reservation, Bytes(0), Bytes(count * extent.value()),
                        Access::kReadWrite)) {
    return false;
  }
  phases.push_back({"map+access", Snapshot(cgroup)});
  const std::uint64_t base = memory.RangeOf(*reservation).value_or({}).base;
  if (kind == BackingKind::kHost) {
    std::memset(Pointer(base), 0x5a, count * extent.value());
  } else {
    // A touched host extent, copied over every device extent.
    const auto staging_reservation = memory.Reserve(extent);
    std::size_t host = 0;
    for (std::size_t i = 0; i < memory.Classes().size(); ++i) {
      if (memory.Classes()[i].kind == BackingKind::kHost) {
        host = i;
      }
    }
    const auto staging = memory.Create(host, extent);
    if (!staging_reservation || !staging || !memory.Map(*staging_reservation, Bytes(0), *staging) ||
        !memory.SetAccess(*staging_reservation, Bytes(0), extent, Access::kReadWrite)) {
      return false;
    }
    const std::uint64_t source = memory.RangeOf(*staging_reservation).value_or({}).base;
    std::memset(Pointer(source), 0x5a, extent.value());
    const auto stream = execution.CreateStream().value_or({});
    for (std::uint64_t i = 0; i < count; ++i) {
      if (!Ok(execution.Copy(stream, base + (i * extent.value()), source, extent))) {
        return false;
      }
    }
    const auto fence = execution.Record(stream).value_or({});
    while (execution.Query(fence).value_or(llmp::providers::FenceState::kComplete) ==
           llmp::providers::FenceState::kPending) {
    }
    if (!Ok(execution.Release(fence)) || !Ok(execution.DestroyStream(stream)) ||
        !Ok(memory.Unmap(*staging_reservation, Bytes(0), extent)) ||
        !Ok(memory.Release(*staging)) || !Ok(memory.Free(*staging_reservation))) {
      return false;
    }
  }
  phases.push_back({"touch", Snapshot(cgroup)});
  if (!memory.Unmap(*reservation, Bytes(0), Bytes(count * extent.value()))) {
    return false;
  }
  phases.push_back({"unmap", Snapshot(cgroup)});
  for (const BackingId backing : backings) {
    if (!Ok(memory.Release(backing))) {
      return false;
    }
  }
  if (!Ok(memory.Free(*reservation))) {
    return false;
  }
  phases.push_back({"release", Snapshot(cgroup)});
  Report(kind == BackingKind::kHost ? "Host (NUMA)" : "Device-local", phases);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::println(stderr, "usage: llmp_vmm_counters GIB");
    return 2;
  }
  const std::uint64_t bytes = std::strtoull(argv[1], nullptr, 10)
                              << 30U;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  auto memory = llmp::providers::cuda::OpenDeviceMemory(0);
  auto execution = llmp::providers::cuda::OpenDeviceExecution(0);
  if (!memory || !execution) {
    std::println(stderr, "no CUDA device");
    return 1;
  }
  // The providers make the device's primary context current per call; make
  // it current here too, so the driver's free memory reads from the start.
  CUdevice device = 0;
  CUcontext context = nullptr;
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS ||
      cuCtxSetCurrent(context) != CUDA_SUCCESS) {
    std::println(stderr, "no CUDA context");
    return 1;
  }
  const std::string cgroup = CgroupPath();
  std::println(
      "{} GiB of each kind in {} KiB extents; deltas in MiB from before the reservation; cgroup {}",
      bytes >> 30U, (*memory)->Granularity().value() >> 10U, cgroup);
  const bool ok = Run(**memory, **execution, BackingKind::kDevice, bytes, cgroup) &&
                  Run(**memory, **execution, BackingKind::kHost, bytes, cgroup);
  (void)cuDevicePrimaryCtxRelease(device);
  return ok ? 0 : 1;
}
