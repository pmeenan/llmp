// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <cuda.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
constexpr std::size_t MiB = 1024 * 1024;
void check(CUresult result, const char* expression) {
  if (result == CUDA_SUCCESS) return;
  const char* name = nullptr;
  cuGetErrorName(result, &name);
  std::fprintf(stderr, "%s: %s\n", expression, name ? name : "unknown CUDA error");
  // Standalone probe: failure exits the process; context teardown reclaims resources.
  std::exit(1);
}
#define CUDA(expr) check((expr), #expr)
void require(bool condition, const char* message) {
  if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
struct Load {
  CUfunction function{};
  CUstream stream{};
  CUevent begin{}, end{};
  int* started{};
  CUdeviceptr device_started{}, output{};
  int blocks{};
  void start(int count) {
    blocks = count;
    if (!blocks) return;
    std::atomic_ref<int>(*started).store(0, std::memory_order_release);
    unsigned long long duration = 10000000; // 10 ms, no unbounded GPU spin.
    void* args[] = {&device_started, &output, &duration};
    CUDA(cuEventRecord(begin, stream));
    CUDA(cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0, stream, args, nullptr));
    CUDA(cuEventRecord(end, stream));
    const auto deadline = Clock::now() + std::chrono::seconds(2);
    while (std::atomic_ref<int>(*started).load(std::memory_order_acquire) != blocks) {
      require(Clock::now() < deadline, "kernel start handshake timed out");
      std::this_thread::yield();
    }
    require(cuEventQuery(end) == CUDA_ERROR_NOT_READY, "load completed before timed operation");
  }
  bool pending() {
    if (!blocks) return false;
    const auto result = cuEventQuery(end);
    if (result == CUDA_ERROR_NOT_READY) return true;
    CUDA(result);
    return false;
  }
  float finish() {
    if (!blocks) return 0;
    CUDA(cuEventSynchronize(end));
    float elapsed = 0;
    CUDA(cuEventElapsedTime(&elapsed, begin, end));
    return elapsed;
  }
};
void snapshot(const char* phase) {
  CUDA(cuCtxSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  std::size_t free = 0, total = 0;
  CUDA(cuMemGetInfo(&free, &total));
  std::ifstream input("/proc/meminfo");
  std::string line;
  unsigned long long available = 0;
  while (std::getline(input, line)) {
    if (line.starts_with("MemAvailable:")) {
      require(std::sscanf(line.c_str(), "MemAvailable: %llu kB", &available) == 1,
              "could not parse MemAvailable");
    }
  }
  require(available != 0, "missing MemAvailable");
  std::printf("memory,%s,%zu,%zu,%llu\n", phase, free, total, available * 1024);
}
int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "Usage: %s KERNEL_CUBIN\n", argv[0]); return 2; }
  CUDA(cuInit(0));
  CUdevice device{};
  CUDA(cuDeviceGet(&device, 0));
  int supported = 0, sms = 0, driver = 0;
  CUDA(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));
  require(supported, "VMM is unsupported");
  CUDA(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  CUDA(cuDriverGetVersion(&driver));
  char name[128]{};
  CUDA(cuDeviceGetName(name, sizeof(name), device));
  CUcontext context{};
  CUDA(cuDevicePrimaryCtxRetain(&context, device));
  CUDA(cuCtxSetCurrent(context));
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  std::size_t minimum = 0, recommended = 0;
  CUDA(cuMemGetAllocationGranularity(&minimum, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  CUDA(cuMemGetAllocationGranularity(&recommended, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  std::printf("device,%s,driver_api,%d,header,%d,sms,%d,minimum,%zu,recommended,%zu\n",
              name, driver, CUDA_VERSION, sms, minimum, recommended);
  CUmemAccessDesc access{};
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUmodule module{};
  CUDA(cuModuleLoad(&module, argv[1]));
  Load load;
  CUDA(cuModuleGetFunction(&load.function, module, "busy"));
  CUfunction verify{};
  CUDA(cuModuleGetFunction(&verify, module, "verify"));
  CUDA(cuStreamCreate(&load.stream, CU_STREAM_NON_BLOCKING));
  CUDA(cuEventCreate(&load.begin, CU_EVENT_DEFAULT));
  CUDA(cuEventCreate(&load.end, CU_EVENT_DEFAULT));
  CUDA(cuMemHostAlloc(reinterpret_cast<void**>(&load.started), sizeof(int), CU_MEMHOSTALLOC_DEVICEMAP));
  CUDA(cuMemHostGetDevicePointer(&load.device_started, load.started, 0));
  CUDA(cuMemAlloc(&load.output, static_cast<std::size_t>(sms) * 128 * sizeof(float)));
  CUdeviceptr errors{};
  CUDA(cuMemAlloc(&errors, sizeof(unsigned)));
  // Warm the kernel and driver path before any measurements.
  load.start(sms); load.finish();
  std::vector<std::size_t> sizes{minimum, 2 * MiB, 8 * MiB, 32 * MiB, 128 * MiB};
  for (auto& size : sizes) size = (size + minimum - 1) / minimum * minimum;
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  std::puts("sample_header,mode,bytes,iteration,operation,host_us,kernel_ms,pending_after");
  for (const auto size : sizes) {
    CUdeviceptr address{};
    CUDA(cuMemAddressReserve(&address, size, 0, 0, 0));
    for (int mode = 0; mode < 3; ++mode) {
      const char* label = mode == 0 ? "idle" : mode == 1 ? "one_block" : "sm_count_blocks";
      const int blocks = mode == 0 ? 0 : mode == 1 ? 1 : sms;
      const int iterations = mode == 0 ? 100 : 30;
      for (int iteration = -5; iteration < iterations; ++iteration) {
        auto measure = [&](const char* operation, auto action) {
          load.start(blocks);
          const auto start = Clock::now();
          action();
          const auto end = Clock::now();
          const bool pending = load.pending();
          const auto kernel_ms = load.finish();
          if (iteration >= 0) std::printf("sample,%s,%zu,%d,%s,%.3f,%.3f,%d\n", label,
            size, iteration, operation, std::chrono::duration<double, std::micro>(end - start).count(),
            kernel_ms, pending);
        };
        CUmemGenericAllocationHandle handle{};
        measure("create", [&] { CUDA(cuMemCreate(&handle, size, &prop, 0)); });
        measure("map", [&] { CUDA(cuMemMap(address, size, 0, handle, 0)); });
        measure("access", [&] { CUDA(cuMemSetAccess(address, size, &access, 1)); });
        // Access every byte outside timed intervals; complete use before unmapping.
        CUDA(cuMemsetD32(address, 0x12345678U, size / sizeof(unsigned)));
        CUDA(cuCtxSynchronize());
        measure("unmap", [&] { CUDA(cuMemUnmap(address, size)); });
        measure("release", [&] { CUDA(cuMemRelease(handle)); });
      }
    }
    CUDA(cuMemAddressFree(address, size));
  }
  std::puts("memory_header,phase,cuda_free_bytes,cuda_total_bytes,mem_available_bytes");
  snapshot("baseline");
  constexpr int count = 16;
  const auto chunk = (64 * MiB + minimum - 1) / minimum * minimum;
  const auto pool_bytes = chunk * count;
  CUdeviceptr base{};
  CUDA(cuMemAddressReserve(&base, pool_bytes, 0, 0, 0));
  snapshot("address_reserved");
  std::vector<CUmemGenericAllocationHandle> handles(count);
  for (auto& handle : handles) CUDA(cuMemCreate(&handle, chunk, &prop, 0));
  snapshot("created");
  auto map = [&] {
    for (int i = 0; i < count; ++i) CUDA(cuMemMap(base + i * chunk, chunk, 0, handles[i], 0));
    CUDA(cuMemSetAccess(base, pool_bytes, &access, 1));
  };
  auto unmap = [&] {
    for (int i = 0; i < count; ++i) CUDA(cuMemUnmap(base + i * chunk, chunk));
  };
  auto verify_all = [&](unsigned expected_mismatches = 0) {
    CUDA(cuMemsetD32Async(errors, 0, 1, load.stream));
    for (int i = 0; i < count; ++i) {
      auto pointer = base + i * chunk;
      unsigned long long words = chunk / sizeof(unsigned);
      unsigned expected = 0xabc00000U + i;
      void* args[] = {&pointer, &words, &expected, &errors};
      CUDA(cuLaunchKernel(verify, 128, 1, 1, 256, 1, 1, 0, load.stream, args, nullptr));
    }
    CUDA(cuStreamSynchronize(load.stream));
    unsigned mismatches = 0;
    CUDA(cuMemcpyDtoH(&mismatches, errors, sizeof(mismatches)));
    require(mismatches == expected_mismatches, "unexpected pool mismatch count");
  };
  map();
  for (int i = 0; i < count; ++i) CUDA(cuMemsetD32(base + i * chunk, 0xabc00000U + i, chunk / sizeof(unsigned)));
  CUDA(cuCtxSynchronize());
  verify_all();
  // Negative control: the verifier must detect exactly one damaged word.
  CUDA(cuMemsetD32(base, 0, 1));
  CUDA(cuCtxSynchronize());
  verify_all(1);
  CUDA(cuMemsetD32(base, 0xabc00000U, 1));
  CUDA(cuCtxSynchronize());
  verify_all();
  std::puts("verification_negative_control,PASS");
  snapshot("mapped_touched");
  unmap();
  snapshot("unmapped_handles_retained");
  map();
  verify_all();
  snapshot("remapped_verified");
  unmap();
  for (auto handle : handles) CUDA(cuMemRelease(handle));
  snapshot("handles_released");
  CUDA(cuMemAddressFree(base, pool_bytes));
  snapshot("address_freed");
  std::printf("pool_verified_bytes,%zu\n", pool_bytes);
  CUDA(cuMemFree(errors));
  CUDA(cuMemFree(load.output));
  CUDA(cuMemFreeHost(load.started));
  CUDA(cuEventDestroy(load.begin));
  CUDA(cuEventDestroy(load.end));
  CUDA(cuStreamDestroy(load.stream));
  CUDA(cuModuleUnload(module));
  CUDA(cuDevicePrimaryCtxRelease(device));
  std::puts("PASS");
}
