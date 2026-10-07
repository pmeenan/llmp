// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Standalone GB10 probe of CUDA VMM call costs across ranges of adjacent,
// independently created 2 MiB mappings (D-033's extents): per-extent versus
// one-call cuMemSetAccess and cuMemUnmap, create/map/release costs, parallel
// cuMemCreate scaling, and a 2 MiB zero fill by device memset versus a
// host-to-device copy. Prints one JSON object. Failure exits the process;
// context teardown reclaims its resources.
#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t kExtent = 2U << 20U;

void Check(CUresult result, const char* expression) {
  if (result == CUDA_SUCCESS) return;
  const char* name = nullptr;
  cuGetErrorName(result, &name);
  std::fprintf(stderr, "%s: %s\n", expression, name ? name : "unknown CUDA error");
  std::exit(1);
}
#define CUDA(expr) Check((expr), #expr)

double Since(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
double Median(std::vector<double> v) {
  std::ranges::sort(v);
  return v.empty() ? 0 : v[v.size() / 2];
}

CUmemAllocationProp Prop(int device) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  return prop;
}
CUmemAccessDesc Access(int device) {
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return access;
}

struct RangeResult {
  std::size_t extents = 0;
  double create = 0, map = 0, release = 0;            // per extent, us
  double access_each = 0, access_range = 0;           // per extent, us
  double unmap_each = 0, unmap_range = 0;             // per extent, us
  bool range_access_ok = false, range_unmap_ok = false;
};

// One range of `n` adjacent extents, `rounds` times, alternating per-extent
// and one-call forms on freshly mapped ranges.
RangeResult Range(int device, std::size_t n, int rounds) {
  const auto prop = Prop(device);
  const auto access = Access(device);
  CUdeviceptr base = 0;
  CUDA(cuMemAddressReserve(&base, n * kExtent, kExtent, 0, 0));
  std::vector<CUmemGenericAllocationHandle> handles(n);
  std::vector<double> create, map, release, access_each, access_range, unmap_each, unmap_range;
  RangeResult r{.extents = n, .range_access_ok = true, .range_unmap_ok = true};
  for (int round = 0; round < rounds; ++round) {
    auto t = Clock::now();
    for (auto& h : handles) CUDA(cuMemCreate(&h, kExtent, &prop, 0));
    create.push_back(Since(t) / n);
    for (int form = 0; form < 2; ++form) {
      t = Clock::now();
      for (std::size_t i = 0; i < n; ++i)
        CUDA(cuMemMap(base + i * kExtent, kExtent, 0, handles[i], 0));
      map.push_back(Since(t) / n);
      if (form == 0) {
        t = Clock::now();
        for (std::size_t i = 0; i < n; ++i)
          CUDA(cuMemSetAccess(base + i * kExtent, kExtent, &access, 1));
        access_each.push_back(Since(t) / n);
      } else {
        t = Clock::now();
        const CUresult result = cuMemSetAccess(base, n * kExtent, &access, 1);
        const double us = Since(t) / n;
        if (result != CUDA_SUCCESS) {
          r.range_access_ok = false;
          for (std::size_t i = 0; i < n; ++i)
            CUDA(cuMemSetAccess(base + i * kExtent, kExtent, &access, 1));
        } else {
          access_range.push_back(us);
        }
      }
      // Touch every extent so the mappings are live, not just recorded.
      CUDA(cuMemsetD8(base, 0, n * kExtent));
      CUDA(cuCtxSynchronize());
      if (form == 0) {
        t = Clock::now();
        for (std::size_t i = 0; i < n; ++i) CUDA(cuMemUnmap(base + i * kExtent, kExtent));
        unmap_each.push_back(Since(t) / n);
      } else {
        t = Clock::now();
        const CUresult result = cuMemUnmap(base, n * kExtent);
        const double us = Since(t) / n;
        if (result != CUDA_SUCCESS) {
          r.range_unmap_ok = false;
          for (std::size_t i = 0; i < n; ++i) CUDA(cuMemUnmap(base + i * kExtent, kExtent));
        } else {
          unmap_range.push_back(us);
        }
      }
    }
    t = Clock::now();
    for (auto h : handles) CUDA(cuMemRelease(h));
    release.push_back(Since(t) / n);
  }
  CUDA(cuMemAddressFree(base, n * kExtent));
  r.create = Median(create);
  r.map = Median(map);
  r.release = Median(release);
  r.access_each = Median(access_each);
  r.access_range = Median(access_range);
  r.unmap_each = Median(unmap_each);
  r.unmap_range = Median(unmap_range);
  return r;
}

// One handle of `n` extents' size: per-2 MiB create, access and unmap cost,
// to tell per-handle costs from per-byte ones.
void Large(int device, std::size_t n, int rounds, double& create, double& access_us,
           double& unmap) {
  const auto prop = Prop(device);
  const auto access = Access(device);
  CUdeviceptr base = 0;
  CUDA(cuMemAddressReserve(&base, n * kExtent, kExtent, 0, 0));
  std::vector<double> c, a, u;
  for (int round = 0; round < rounds; ++round) {
    CUmemGenericAllocationHandle h{};
    auto t = Clock::now();
    CUDA(cuMemCreate(&h, n * kExtent, &prop, 0));
    c.push_back(Since(t) / n);
    CUDA(cuMemMap(base, n * kExtent, 0, h, 0));
    t = Clock::now();
    CUDA(cuMemSetAccess(base, n * kExtent, &access, 1));
    a.push_back(Since(t) / n);
    CUDA(cuMemsetD8(base, 0, n * kExtent));
    CUDA(cuCtxSynchronize());
    t = Clock::now();
    CUDA(cuMemUnmap(base, n * kExtent));
    u.push_back(Since(t) / n);
    CUDA(cuMemRelease(h));
  }
  CUDA(cuMemAddressFree(base, n * kExtent));
  create = Median(c);
  access_us = Median(a);
  unmap = Median(u);
}

// Wall time per created extent with `threads` threads creating in parallel.
double ParallelCreate(CUcontext context, int device, int threads, std::size_t per_thread) {
  const auto prop = Prop(device);
  std::vector<std::vector<CUmemGenericAllocationHandle>> made(threads);
  const auto t = Clock::now();
  std::vector<std::thread> pool;
  for (int k = 0; k < threads; ++k)
    pool.emplace_back([&, k] {
      CUDA(cuCtxSetCurrent(context));
      made[k].resize(per_thread);
      for (auto& h : made[k]) CUDA(cuMemCreate(&h, kExtent, &prop, 0));
    });
  for (auto& thread : pool) thread.join();
  const double us = Since(t) / (threads * per_thread);
  for (auto& v : made)
    for (auto h : v) CUDA(cuMemRelease(h));
  return us;
}

// Zero fill of one mapped 2 MiB extent: device memset versus a host-to-device
// copy from pinned zeros, each queued and completed on a stream.
void ZeroFill(int device, int rounds, double& memset_us, double& copy_us) {
  const auto prop = Prop(device);
  const auto access = Access(device);
  CUdeviceptr base = 0;
  CUmemGenericAllocationHandle h{};
  CUDA(cuMemAddressReserve(&base, kExtent, kExtent, 0, 0));
  CUDA(cuMemCreate(&h, kExtent, &prop, 0));
  CUDA(cuMemMap(base, kExtent, 0, h, 0));
  CUDA(cuMemSetAccess(base, kExtent, &access, 1));
  void* zeros = nullptr;
  CUDA(cuMemHostAlloc(&zeros, kExtent, 0));
  std::fill_n(static_cast<char*>(zeros), kExtent, 0);
  CUstream stream{};
  CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  std::vector<double> a, b;
  for (int i = 0; i < rounds; ++i) {
    auto t = Clock::now();
    CUDA(cuMemsetD8Async(base, 0, kExtent, stream));
    CUDA(cuStreamSynchronize(stream));
    a.push_back(Since(t));
    t = Clock::now();
    CUDA(cuMemcpyHtoDAsync(base, zeros, kExtent, stream));
    CUDA(cuStreamSynchronize(stream));
    b.push_back(Since(t));
  }
  memset_us = Median(a);
  copy_us = Median(b);
  CUDA(cuStreamDestroy(stream));
  CUDA(cuMemFreeHost(zeros));
  CUDA(cuMemUnmap(base, kExtent));
  CUDA(cuMemRelease(h));
  CUDA(cuMemAddressFree(base, kExtent));
}
// Host-to-device copy bandwidth (2 MiB pinned copies, synchronized each)
// over `seconds`, alone or while another thread loops VMM calls over 256
// other extents: `mode` 0 none, 1 unmap+map, 2 unmap+map+access,
// 3 create+map+access+unmap+release.
double CopyBandwidth(CUcontext context, int device, int mode, double seconds) {
  const auto prop = Prop(device);
  const auto access = Access(device);
  constexpr std::size_t kSide = 256;
  CUdeviceptr side = 0;
  CUDA(cuMemAddressReserve(&side, kSide * kExtent, kExtent, 0, 0));
  std::vector<CUmemGenericAllocationHandle> handles(kSide);
  for (std::size_t i = 0; i < kSide; ++i) {
    CUDA(cuMemCreate(&handles[i], kExtent, &prop, 0));
    CUDA(cuMemMap(side + i * kExtent, kExtent, 0, handles[i], 0));
    CUDA(cuMemSetAccess(side + i * kExtent, kExtent, &access, 1));
  }
  CUdeviceptr target = 0;
  CUDA(cuMemAlloc(&target, kExtent));
  void* host = nullptr;
  CUDA(cuMemHostAlloc(&host, kExtent, 0));
  CUstream stream{};
  CUDA(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  std::atomic<bool> stop{false};
  std::atomic<std::uint64_t> side_ops{0};
  std::thread churn;
  if (mode != 0) {
    churn = std::thread([&] {
      CUDA(cuCtxSetCurrent(context));
      std::size_t i = 0;
      while (!stop.load()) {
        const CUdeviceptr at = side + (i % kSide) * kExtent;
        CUDA(cuMemUnmap(at, kExtent));
        if (mode == 3) {
          CUDA(cuMemRelease(handles[i % kSide]));
          CUDA(cuMemCreate(&handles[i % kSide], kExtent, &prop, 0));
        }
        CUDA(cuMemMap(at, kExtent, 0, handles[i % kSide], 0));
        if (mode >= 2) CUDA(cuMemSetAccess(at, kExtent, &access, 1));
        side_ops.fetch_add(1);
        ++i;
      }
    });
  }
  std::uint64_t copies = 0;
  const auto t = Clock::now();
  while (Since(t) < seconds * 1e6) {
    CUDA(cuMemcpyHtoDAsync(target, host, kExtent, stream));
    CUDA(cuStreamSynchronize(stream));
    ++copies;
  }
  const double us = Since(t);
  stop.store(true);
  if (churn.joinable()) churn.join();
  (void)std::fprintf(stderr, "copy mode %d: %llu side ops\n", mode,
                     static_cast<unsigned long long>(side_ops.load()));
  CUDA(cuStreamDestroy(stream));
  CUDA(cuMemFreeHost(host));
  CUDA(cuMemFree(target));
  for (std::size_t i = 0; i < kSide; ++i) {
    CUDA(cuMemUnmap(side + i * kExtent, kExtent));
    CUDA(cuMemRelease(handles[i]));
  }
  CUDA(cuMemAddressFree(side, kSide * kExtent));
  return static_cast<double>(copies) * kExtent / us / 1e3;  // GB/s
}
}  // namespace

int main(int argc, char** argv) {
  const int rounds = argc > 1 ? std::atoi(argv[1]) : 7;
  CUDA(cuInit(0));
  CUdevice device{};
  CUDA(cuDeviceGet(&device, 0));
  CUcontext context{};
  CUDA(cuDevicePrimaryCtxRetain(&context, device));
  CUDA(cuCtxSetCurrent(context));
  int driver = 0;
  CUDA(cuDriverGetVersion(&driver));
  char name[128] = {};
  CUDA(cuDeviceGetName(name, sizeof(name), device));
  (void)Range(device, 16, 2);  // warm the driver's paths
  std::printf("{\"device\":\"%s\",\"driver\":%d,\"rounds\":%d,\"ranges\":[", name, driver, rounds);
  bool first = true;
  for (const std::size_t n : {1, 2, 4, 8, 16, 64, 256, 1024}) {
    const auto r = Range(device, n, rounds);
    std::printf(
        "%s{\"extents\":%zu,\"create_us\":%.2f,\"map_us\":%.2f,\"release_us\":%.2f,"
        "\"access_each_us\":%.2f,\"access_range_us\":%.2f,\"range_access_ok\":%s,"
        "\"unmap_each_us\":%.2f,\"unmap_range_us\":%.2f,\"range_unmap_ok\":%s}",
        first ? "" : ",", r.extents, r.create, r.map, r.release, r.access_each, r.access_range,
        r.range_access_ok ? "true" : "false", r.unmap_each, r.unmap_range,
        r.range_unmap_ok ? "true" : "false");
    first = false;
  }
  std::printf("],\"one_handle\":[");
  first = true;
  for (const std::size_t n : {1, 8, 64, 256}) {
    double c = 0, a = 0, u = 0;
    Large(device, n, rounds, c, a, u);
    std::printf("%s{\"extents\":%zu,\"create_us\":%.2f,\"access_us\":%.2f,\"unmap_us\":%.2f}",
                first ? "" : ",", n, c, a, u);
    first = false;
  }
  std::printf("],\"parallel_create_us_per_extent\":{");
  first = true;
  for (const int threads : {1, 2, 4, 8}) {
    std::printf("%s\"%d\":%.2f", first ? "" : ",", threads,
                ParallelCreate(context, device, threads, 256 / threads));
    first = false;
  }
  double memset_us = 0, copy_us = 0;
  ZeroFill(device, rounds * 4, memset_us, copy_us);
  std::printf("},\"zero_fill_2mib\":{\"memset_us\":%.2f,\"htod_copy_us\":%.2f}", memset_us,
              copy_us);
  std::printf(",\"htod_gbps_beside\":[");
  first = true;
  for (const int mode : {0, 1, 2, 3, 0}) {
    std::printf("%s[%d,%.2f]", first ? "" : ",", mode, CopyBandwidth(context, device, mode, 2.0));
    first = false;
  }
  std::printf("]}\n");
  CUDA(cuDevicePrimaryCtxRelease(device));
  return 0;
}
