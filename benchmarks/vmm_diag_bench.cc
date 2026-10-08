// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The host-VMM diagnosis (docs/experiments/host-vmm-diagnosis/): why BP-F1's
// GGML matrix products ran slower with their memory in host VMM than in
// cudaMalloc memory, and what each alternative costs. Diagnostic only: it
// changes nothing about BP-F1's rule, harness (ggml_vmm_bench.cc) or verdict.
//
// Memory arms (each allocation is one of these):
//   malloc      cudaMalloc
//   dvmm        cuMemCreate at the device, mapped for the device
//   dvmm-cpu    the same, also mapped for the CPU on the host NUMA node (if the
//               driver allows it)
//   hvmm        cuMemCreate at the host NUMA node, one handle, mapped for the
//               device and the CPU: what llmpalooza's CUDA provider does
//   hvmm-jit    the same through llmpalooza's provider itself (VmmProvider)
//   hvmm-gpu    host NUMA backing mapped for the device only
//   hvmm-2m     host NUMA backing in 2 MiB handles (the pager's extents)
//   hvmm-host   CU_MEM_LOCATION_TYPE_HOST backing (no NUMA node)
//   pinned      cudaMallocHost
//   registered  aligned_alloc plus cudaHostRegister
//   pageable    aligned_alloc, read by the GPU through the host page tables
//   pageable-thp  the same, madvise(MADV_HUGEPAGE) before the first touch
//   registered-thp  that, plus cudaHostRegister
//   managed     cudaMallocManaged
//   managed-prefetch  that, touched by the CPU, then prefetched to the device
//
// Modes, each printing CSV rows (# lines are comments):
//   info [--arm A]... [--file F]      device facts; each arm's pointer facts
//                                     and, with a file, whether a direct
//                                     (O_DIRECT) read into it works
//   micro --arm A... [--rounds R] [--bytes B] [--test T]...
//                                     microkernels, arms interleaved per round
//   copy --from A --to B [--rounds R] copy bandwidth and 2 MiB copy latency
//   restore --dir D --gib G --landing A --to A --method M [--depth N] [--rounds R]
//                                     a restore through llmpalooza's io_uring
//                                     provider: direct reads of 2 MiB extents
//                                     of an unnamed G GiB file in D, into the
//                                     destination itself (M none; A must then
//                                     be CPU-mapped) or into a landing slot
//                                     that a copy (M memcpy or kernel) moves
//                                     into the destination
//   ggml --weights A --acts A --scratch A --workspace A [--outputs A]
//        [--case NAME@ROWS]...
//                                     GGML products with each buffer group
//                                     placed separately

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <expected>
#include <format>
#include <fstream>
#include <memory>
#include <numeric>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "ggml.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/tensors.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/storage.h"
#include "providers/uring_storage.h"
#include "vmm_diag_kernels.h"

namespace {

using llmp::base::Bytes;
using llmp::kernels::ggml::CublasHandle;
using llmp::kernels::ggml::KernelFailure;
using llmp::kernels::ggml::LaunchContext;
using llmp::kernels::ggml::TensorArena;
using llmp::providers::Access;
using llmp::providers::BackingKind;
using llmp::providers::DeviceExecution;
using llmp::providers::StreamId;
using llmp::providers::VmmProvider;

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
constexpr std::uint64_t kAlign = 256;

template <typename... Args>
std::unexpected<std::string> Fail(std::format_string<Args...> format, Args&&... args) {
  return std::unexpected(std::format(format, std::forward<Args>(args)...));
}

std::uint64_t RoundUp(std::uint64_t value, std::uint64_t to) { return (value + to - 1) / to * to; }

// Cleanup whose failure changes no measurement.
template <typename T>
void Discard(const T& /*result*/) {}

std::expected<void, std::string> Cuda(cudaError_t error, std::string_view what) {
  if (error == cudaSuccess) {
    return {};
  }
  return Fail("{}: {}", what, cudaGetErrorString(error));
}

std::expected<void, std::string> Driver(CUresult result, std::string_view what) {
  if (result == CUDA_SUCCESS) {
    return {};
  }
  const char* name = nullptr;
  (void)cuGetErrorName(result, &name);
  return Fail("{}: {}", what, name == nullptr ? "unknown error" : name);
}

// ---- Memory arms ----

enum class Arm : std::uint8_t {
  kMalloc,
  kDeviceVmm,
  kDeviceVmmCpu,
  kHostVmm,
  kHostVmmJit,
  kHostVmmGpu,
  kHostVmm2M,
  kHostVmmHost,
  kPinned,
  kRegistered,
  kPageable,
  kPageableThp,
  kRegisteredThp,
  kManaged,
  kManagedPrefetch,
};

constexpr std::array<std::pair<std::string_view, Arm>, 15> kArms = {{
    {"malloc", Arm::kMalloc},
    {"dvmm", Arm::kDeviceVmm},
    {"dvmm-cpu", Arm::kDeviceVmmCpu},
    {"hvmm", Arm::kHostVmm},
    {"hvmm-jit", Arm::kHostVmmJit},
    {"hvmm-gpu", Arm::kHostVmmGpu},
    {"hvmm-2m", Arm::kHostVmm2M},
    {"hvmm-host", Arm::kHostVmmHost},
    {"pinned", Arm::kPinned},
    {"registered", Arm::kRegistered},
    {"pageable", Arm::kPageable},
    {"pageable-thp", Arm::kPageableThp},
    {"registered-thp", Arm::kRegisteredThp},
    {"managed", Arm::kManaged},
    {"managed-prefetch", Arm::kManagedPrefetch},
}};

std::expected<Arm, std::string> ParseArm(std::string_view name) {
  for (const auto& [arm_name, arm] : kArms) {
    if (arm_name == name) {
      return arm;
    }
  }
  return Fail("unknown memory arm {}", name);
}

std::string_view ArmName(Arm arm) {
  for (const auto& [arm_name, value] : kArms) {
    if (value == arm) {
      return arm_name;
    }
  }
  return "?";
}

struct Buffer {
  Arm arm = Arm::kMalloc;
  std::uint64_t base = 0;
  std::uint64_t size = 0;      // as asked
  std::uint64_t reserved = 0;  // VMM: the reservation and backing
  std::vector<CUmemGenericAllocationHandle> handles;
  void* pointer = nullptr;  // runtime and system allocations
  llmp::providers::ReservationId reservation;
  llmp::providers::BackingId backing;
};

class Memory {
 public:
  static std::expected<std::unique_ptr<Memory>, std::string> Open() {
    auto provider = llmp::providers::cuda::OpenDeviceMemory(0);
    if (!provider) {
      return Fail("no device memory: {}", provider.error().detail);
    }
    auto memory = std::unique_ptr<Memory>(new Memory(std::move(*provider)));  // NOLINT
    if (auto got = Driver(cuDeviceGet(&memory->device_, 0), "cuDeviceGet"); !got) {
      return std::unexpected(got.error());
    }
    if (auto got = Driver(
            cuDeviceGetAttribute(&memory->numa_, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, memory->device_),
            "host NUMA id");
        !got) {
      return std::unexpected(got.error());
    }
    // cudart and the provider share the primary context.
    if (auto got = Cuda(cudaSetDevice(0), "cudaSetDevice"); !got) {
      return std::unexpected(got.error());
    }
    for (const Arm arm : {Arm::kDeviceVmm, Arm::kHostVmm, Arm::kHostVmmHost}) {
      auto granularity = memory->Granularity(arm, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
      if (granularity) {
        memory->granularity_ = std::lcm(memory->granularity_, *granularity);
      }
    }
    return memory;
  }

  Memory(const Memory&) = delete;
  Memory& operator=(const Memory&) = delete;
  Memory(Memory&&) = delete;
  Memory& operator=(Memory&&) = delete;
  ~Memory() = default;

  CUdevice device() const { return device_; }
  int numa() const { return numa_; }
  std::uint64_t granularity() const { return granularity_; }

  CUmemAllocationProp Properties(Arm arm) const {
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    if (arm == Arm::kDeviceVmm || arm == Arm::kDeviceVmmCpu) {
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = device_;
    } else if (arm == Arm::kHostVmmHost) {
      prop.location.type = CU_MEM_LOCATION_TYPE_HOST;
      prop.location.id = 0;
    } else {
      prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
      prop.location.id = numa_;
    }
    return prop;
  }

  std::expected<std::uint64_t, std::string> Granularity(
      Arm arm, CUmemAllocationGranularity_flags which) const {
    const CUmemAllocationProp prop = Properties(arm);
    std::size_t granularity = 0;
    if (auto got = Driver(cuMemGetAllocationGranularity(&granularity, &prop, which),
                          "cuMemGetAllocationGranularity");
        !got) {
      return std::unexpected(got.error());
    }
    return granularity;
  }

  std::expected<Buffer, std::string> Allocate(Arm arm, std::uint64_t size) {
    Buffer buffer;
    buffer.arm = arm;
    buffer.size = size;
    switch (arm) {
      case Arm::kMalloc:
        if (auto got = Cuda(cudaMalloc(&buffer.pointer, size), "cudaMalloc"); !got) {
          return std::unexpected(got.error());
        }
        break;
      case Arm::kPinned:
        if (auto got = Cuda(cudaMallocHost(&buffer.pointer, size), "cudaMallocHost"); !got) {
          return std::unexpected(got.error());
        }
        break;
      case Arm::kManaged:
      case Arm::kManagedPrefetch:
        if (auto got = Cuda(cudaMallocManaged(&buffer.pointer, size), "cudaMallocManaged"); !got) {
          return std::unexpected(got.error());
        }
        if (arm == Arm::kManagedPrefetch) {
          std::fill_n(static_cast<unsigned char*>(buffer.pointer), size, 0);
          cudaMemLocation device{};
          device.type = cudaMemLocationTypeDevice;
          device.id = 0;
          if (auto got = Cuda(cudaMemPrefetchAsync(buffer.pointer, size, device, 0, nullptr),
                              "cudaMemPrefetchAsync");
              !got) {
            return std::unexpected(got.error());
          }
          (void)cudaDeviceSynchronize();
        }
        break;
      case Arm::kRegistered:
      case Arm::kPageable:
      case Arm::kPageableThp:
      case Arm::kRegisteredThp: {
        buffer.reserved = RoundUp(size, 2 * kMiB);
        buffer.pointer = std::aligned_alloc(2 * kMiB, buffer.reserved);
        if (buffer.pointer == nullptr) {
          return Fail("aligned_alloc of {} bytes", buffer.reserved);
        }
        if (arm == Arm::kPageableThp || arm == Arm::kRegisteredThp) {
          if (madvise(buffer.pointer, buffer.reserved, MADV_HUGEPAGE) != 0) {
            std::free(buffer.pointer);  // NOLINT(cppcoreguidelines-no-malloc)
            return Fail("madvise(MADV_HUGEPAGE) failed");
          }
        }
        // Touch every page on the CPU first, as a direct read would.
        std::fill_n(static_cast<unsigned char*>(buffer.pointer), buffer.reserved, 0);
        if (arm == Arm::kRegistered || arm == Arm::kRegisteredThp) {
          if (auto got =
                  Cuda(cudaHostRegister(buffer.pointer, buffer.reserved, cudaHostRegisterDefault),
                       "cudaHostRegister");
              !got) {
            std::free(buffer.pointer);  // NOLINT(cppcoreguidelines-no-malloc)
            return std::unexpected(got.error());
          }
        }
        break;
      }
      case Arm::kHostVmmJit:
        return Jit(size);
      case Arm::kDeviceVmm:
      case Arm::kDeviceVmmCpu:
      case Arm::kHostVmm:
      case Arm::kHostVmmGpu:
      case Arm::kHostVmm2M:
      case Arm::kHostVmmHost:
        return Vmm(arm, size);
    }
    buffer.base = reinterpret_cast<std::uintptr_t>(buffer.pointer);
    return buffer;
  }

  void Free(Buffer& buffer) {
    switch (buffer.arm) {
      case Arm::kMalloc:
      case Arm::kManaged:
      case Arm::kManagedPrefetch:
        (void)cudaFree(buffer.pointer);
        break;
      case Arm::kPinned:
        (void)cudaFreeHost(buffer.pointer);
        break;
      case Arm::kRegistered:
      case Arm::kRegisteredThp:
        (void)cudaHostUnregister(buffer.pointer);
        std::free(buffer.pointer);  // NOLINT(cppcoreguidelines-no-malloc)
        break;
      case Arm::kPageable:
      case Arm::kPageableThp:
        std::free(buffer.pointer);  // NOLINT(cppcoreguidelines-no-malloc)
        break;
      case Arm::kHostVmmJit:
        Discard(provider_->Unmap(buffer.reservation, Bytes(0), Bytes(buffer.reserved)));
        Discard(provider_->Release(buffer.backing));
        Discard(provider_->Free(buffer.reservation));
        break;
      default: {
        (void)cuMemUnmap(static_cast<CUdeviceptr>(buffer.base), buffer.reserved);
        for (const CUmemGenericAllocationHandle handle : buffer.handles) {
          (void)cuMemRelease(handle);
        }
        (void)cuMemAddressFree(static_cast<CUdeviceptr>(buffer.base), buffer.reserved);
        break;
      }
    }
    buffer = Buffer{};
  }

 private:
  explicit Memory(std::unique_ptr<VmmProvider> provider) : provider_(std::move(provider)) {}

  std::expected<Buffer, std::string> Vmm(Arm arm, std::uint64_t size) {
    Buffer buffer;
    buffer.arm = arm;
    buffer.size = size;
    buffer.reserved = RoundUp(size, granularity_);
    CUdeviceptr base = 0;
    if (auto got = Driver(cuMemAddressReserve(&base, buffer.reserved, granularity_, 0, 0),
                          "cuMemAddressReserve");
        !got) {
      return std::unexpected(got.error());
    }
    buffer.base = base;
    const CUmemAllocationProp prop = Properties(arm);
    const std::uint64_t piece = arm == Arm::kHostVmm2M ? 2 * kMiB : buffer.reserved;
    for (std::uint64_t offset = 0; offset < buffer.reserved; offset += piece) {
      CUmemGenericAllocationHandle handle = 0;
      if (auto got = Driver(cuMemCreate(&handle, piece, &prop, 0), "cuMemCreate"); !got) {
        Free(buffer);
        return std::unexpected(got.error());
      }
      buffer.handles.push_back(handle);
      if (auto got = Driver(cuMemMap(base + offset, piece, 0, handle, 0), "cuMemMap"); !got) {
        Free(buffer);
        return std::unexpected(got.error());
      }
    }
    std::vector<CUmemAccessDesc> access;
    CUmemAccessDesc gpu{};
    gpu.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    gpu.location.id = device_;
    gpu.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access.push_back(gpu);
    if (arm != Arm::kDeviceVmm && arm != Arm::kHostVmmGpu) {
      CUmemAccessDesc cpu{};
      cpu.location = prop.location;
      if (arm == Arm::kDeviceVmmCpu) {
        cpu.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
        cpu.location.id = numa_;
      }
      cpu.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      access.push_back(cpu);
    }
    if (auto got = Driver(cuMemSetAccess(base, buffer.reserved, access.data(), access.size()),
                          "cuMemSetAccess");
        !got) {
      Free(buffer);
      return std::unexpected(got.error());
    }
    return buffer;
  }

  std::expected<Buffer, std::string> Jit(std::uint64_t size) {
    Buffer buffer;
    buffer.arm = Arm::kHostVmmJit;
    buffer.size = size;
    buffer.reserved = RoundUp(size, provider_->Granularity().value());
    std::size_t host_class = provider_->Classes().size();
    for (std::size_t i = 0; i < provider_->Classes().size(); ++i) {
      if (provider_->Classes()[i].kind == BackingKind::kHost) {
        host_class = i;
      }
    }
    if (host_class == provider_->Classes().size()) {
      return Fail("the provider has no host class");
    }
    auto reservation = provider_->Reserve(Bytes(buffer.reserved));
    if (!reservation) {
      return Fail("reserve: {}", reservation.error().detail);
    }
    auto backing = provider_->Create(host_class, Bytes(buffer.reserved));
    if (!backing) {
      Discard(provider_->Free(*reservation));
      return Fail("create: {}", backing.error().detail);
    }
    buffer.reservation = *reservation;
    buffer.backing = *backing;
    if (auto mapped = provider_->Map(*reservation, Bytes(0), *backing); !mapped) {
      return Fail("map: {}", mapped.error().detail);
    }
    if (auto access = provider_->SetAccess(*reservation, Bytes(0), Bytes(buffer.reserved),
                                           Access::kReadWrite);
        !access) {
      return Fail("access: {}", access.error().detail);
    }
    buffer.base = provider_->RangeOf(*reservation).value().base;
    return buffer;
  }

  std::unique_ptr<VmmProvider> provider_;
  CUdevice device_ = 0;
  int numa_ = -1;
  std::uint64_t granularity_ = 1;
};

// Frees its buffers when it goes out of scope.
class Held {
 public:
  explicit Held(Memory& memory) : memory_(memory) {}
  Held(const Held&) = delete;
  Held& operator=(const Held&) = delete;
  Held(Held&&) = delete;
  Held& operator=(Held&&) = delete;
  ~Held() {
    (void)cudaDeviceSynchronize();
    for (Buffer& buffer : buffers_) {
      memory_.Free(buffer);
    }
  }
  std::expected<std::uint64_t, std::string> Allocate(Arm arm, std::uint64_t size) {
    auto buffer = memory_.Allocate(arm, size);
    if (!buffer) {
      return std::unexpected(buffer.error());
    }
    buffers_.push_back(std::move(*buffer));
    return buffers_.back().base;
  }

 private:
  Memory& memory_;
  std::vector<Buffer> buffers_;
};

class Events {
 public:
  Events() {
    (void)cudaEventCreate(&start_);
    (void)cudaEventCreate(&stop_);
  }
  Events(const Events&) = delete;
  Events& operator=(const Events&) = delete;
  Events(Events&&) = delete;
  Events& operator=(Events&&) = delete;
  ~Events() {
    (void)cudaEventDestroy(start_);
    (void)cudaEventDestroy(stop_);
  }

  // Microseconds that `work` takes on `stream`.
  template <typename Work>
  std::expected<double, std::string> Time(cudaStream_t stream, Work&& work) {
    if (auto got = Cuda(cudaEventRecord(start_, stream), "event"); !got) {
      return std::unexpected(got.error());
    }
    if (auto done = std::forward<Work>(work)(); !done) {
      return std::unexpected(done.error());
    }
    if (auto got = Cuda(cudaEventRecord(stop_, stream), "event"); !got) {
      return std::unexpected(got.error());
    }
    if (auto got = Cuda(cudaEventSynchronize(stop_), "event sync"); !got) {
      return std::unexpected(got.error());
    }
    float ms = 0;
    if (auto got = Cuda(cudaEventElapsedTime(&ms, start_, stop_), "elapsed"); !got) {
      return std::unexpected(got.error());
    }
    return static_cast<double>(ms) * 1000.0;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

// A field of the smaps entry holding `address` in this process, in KiB, or
// 0 when the CPU has no mapping there.
std::uint64_t Smaps(std::uint64_t address, std::string_view field) {
  std::ifstream smaps("/proc/self/smaps");
  std::string line;
  bool inside = false;
  while (std::getline(smaps, line)) {
    const std::size_t dash = line.find('-');
    const std::size_t space = line.find(' ');
    if (dash != std::string::npos && space != std::string::npos && dash < space &&
        line.find(':') > space) {
      std::uint64_t low = 0;
      std::uint64_t high = 0;
      std::from_chars(line.data(), line.data() + dash, low, 16);
      std::from_chars(line.data() + dash + 1, line.data() + space, high, 16);
      inside = address >= low && address < high;
      continue;
    }
    if (inside && line.starts_with(field)) {
      std::uint64_t kib = 0;
      const std::size_t digits = line.find_first_of("0123456789");
      if (digits != std::string::npos) {
        std::from_chars(line.data() + digits, line.data() + line.size(), kib);
      }
      return kib;
    }
  }
  return 0;
}

// ---- info ----

// Whether the CPU can address `arm`'s memory at its device address.
bool CpuMapped(Arm arm) {
  return arm != Arm::kMalloc && arm != Arm::kDeviceVmm && arm != Arm::kHostVmmGpu;
}

// A direct read of `file`'s first `size` bytes into `base`, checked against
// a buffered read of the same bytes copied back by the GPU.
std::string DirectRead(const std::string& file, std::uint64_t base, std::uint64_t size) {
  const int fd = open(file.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);  // NOLINT
  if (fd < 0) {
    return std::format("open failed (errno {})", errno);
  }
  const ssize_t got = pread(fd, reinterpret_cast<void*>(base),  // NOLINT(performance-no-int-to-ptr)
                            size, 0);
  const int error = errno;
  (void)close(fd);
  if (got < 0) {
    return std::format("direct read failed (errno {})", error);
  }
  std::vector<std::byte> expected(static_cast<std::size_t>(got));
  std::vector<std::byte> seen(static_cast<std::size_t>(got));
  std::ifstream buffered(file, std::ios::binary);
  buffered.read(reinterpret_cast<char*>(expected.data()),  // NOLINT
                static_cast<std::streamsize>(expected.size()));
  if (cudaMemcpy(seen.data(),
                 reinterpret_cast<const void*>(base),  // NOLINT(performance-no-int-to-ptr)
                 seen.size(), cudaMemcpyDefault) != cudaSuccess) {
    return "the GPU copy back failed";
  }
  return std::format("read {} bytes, contents {}", got, seen == expected ? "match" : "DIFFER");
}

int Info(Memory& memory, std::span<const Arm> arms, const std::string& file) {
  const CUdevice device = memory.device();
  const std::array<std::pair<const char*, CUdevice_attribute>, 14> attributes = {{
      {"integrated", CU_DEVICE_ATTRIBUTE_INTEGRATED},
      {"l2_cache_bytes", CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE},
      {"persisting_l2_max_bytes", CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE},
      {"sms", CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT},
      {"pageable_memory_access", CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS},
      {"pageable_uses_host_page_tables",
       CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES},
      {"concurrent_managed_access", CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS},
      {"direct_managed_mem_access_from_host",
       CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST},
      {"host_native_atomics", CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED},
      {"can_use_host_pointer_for_registered_mem",
       CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM},
      {"host_numa_vmm", CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED},
      {"host_numa_id", CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID},
      {"global_l1_cache", CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED},
      {"gpu_direct_rdma_with_vmm", CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED},
  }};
  for (const auto& [name, attribute] : attributes) {
    int value = 0;
    const CUresult result = cuDeviceGetAttribute(&value, attribute, device);
    if (result == CUDA_SUCCESS) {
      std::println("info,attribute,{},{}", name, value);
    } else {
      std::println("info,attribute,{},error {}", name, static_cast<int>(result));
    }
  }
  int driver = 0;
  (void)cuDriverGetVersion(&driver);
  std::println("info,driver_api,{}", driver);
  for (const Arm arm : {Arm::kDeviceVmm, Arm::kHostVmm, Arm::kHostVmmHost}) {
    for (const auto& [which, flag] :
         {std::pair{"minimum", CU_MEM_ALLOC_GRANULARITY_MINIMUM},
          std::pair{"recommended", CU_MEM_ALLOC_GRANULARITY_RECOMMENDED}}) {
      auto granularity = memory.Granularity(arm, flag);
      if (granularity) {
        std::println("info,granularity,{},{},{}", ArmName(arm), which, *granularity);
      } else {
        std::println("info,granularity,{},{},{}", ArmName(arm), which, granularity.error());
      }
    }
  }
  for (const Arm arm : arms) {
    constexpr std::uint64_t kSize = 64 * kMiB;
    auto buffer = memory.Allocate(arm, kSize);
    if (!buffer) {
      std::println("info,arm,{},allocation,{}", ArmName(arm), buffer.error());
      continue;
    }
    unsigned memory_type = 0;
    const CUresult typed = cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                                 static_cast<CUdeviceptr>(buffer->base));
    int mapped = 0;
    const CUresult is_mapped = cuPointerGetAttribute(&mapped, CU_POINTER_ATTRIBUTE_MAPPED,
                                                     static_cast<CUdeviceptr>(buffer->base));
    std::println("info,arm,{},memory_type,{},mapped,{},cpu_page_kib,{},anon_huge_kib,{}",
                 ArmName(arm), typed == CUDA_SUCCESS ? static_cast<int>(memory_type) : -1,
                 is_mapped == CUDA_SUCCESS ? mapped : -1, Smaps(buffer->base, "KernelPageSize:"),
                 Smaps(buffer->base, "AnonHugePages:"));
    if (!file.empty() && CpuMapped(arm)) {
      std::println("info,arm,{},direct_read,{}", ArmName(arm),
                   DirectRead(file, buffer->base, kSize));
    }
    memory.Free(*buffer);
  }
  return 0;
}

// ---- micro ----

struct Test {
  std::string_view name;
  std::uint64_t moved = 0;  // bytes the kernel reads or writes
};

int Micro(Memory& memory, std::span<const Arm> arms, int rounds, std::uint64_t bytes,
          std::span<const std::string> only) {
  int sms = 0;
  (void)cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int blocks = sms * 16;
  cudaStream_t stream = nullptr;
  if (auto got = Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream"); !got) {
    std::println(stderr, "{}", got.error());
    return 1;
  }
  Held held(memory);
  std::vector<std::uint64_t> buffers;
  for (const Arm arm : arms) {
    auto base = held.Allocate(arm, bytes);
    if (!base) {
      std::println(stderr, "{}: {}", ArmName(arm), base.error());
      return 1;
    }
    if (auto got = Cuda(llmp::diag::Fill(stream, *base, bytes / 4, 1), "fill"); !got) {
      std::println(stderr, "{}", got.error());
      return 1;
    }
    buffers.push_back(*base);
  }
  void* out_pointer = nullptr;
  constexpr std::uint64_t kOut = std::uint64_t{64} << 20U;
  if (auto got = Cuda(cudaMalloc(&out_pointer, kOut), "cudaMalloc"); !got) {
    std::println(stderr, "{}", got.error());
    return 1;
  }
  const auto out = reinterpret_cast<std::uintptr_t>(out_pointer);
  if (auto got = Cuda(cudaStreamSynchronize(stream), "sync"); !got) {
    std::println(stderr, "{}", got.error());
    return 1;
  }
  constexpr std::uint64_t kScan = 256 * kMiB;
  constexpr int kLines = 64;
  const std::uint64_t warps = static_cast<std::uint64_t>(blocks) * 256 / 32;
  const std::uint64_t random_bytes = warps * kLines * 128;
  // L2 set-aside for the access-policy variant of the reread.
  (void)cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 8 * kMiB);
  constexpr std::uint64_t kSparse = std::uint64_t{1} << 20U;  // writes, one per 128 B
  constexpr std::uint64_t kBlocks = 151936;                   // the output head's rows
  const std::array<Test, 14> tests = {{
      {"scan4", kScan},
      {"scan16", kScan},
      {"write16", kScan},
      {"write-sparse", kSparse * 4},
      {"write-per-block", kBlocks * 4},
      {"reread-4m", 4 * kMiB * 64},
      {"reread-4m-persist", 4 * kMiB * 64},
      {"reread-16m", 16 * kMiB * 16},
      {"reread-64m", 64 * kMiB * 4},
      {"rand-all", random_bytes},
      {"rand-2m", random_bytes},
      {"rand-64k", random_bytes},
      {"rand-8m-span", random_bytes},
      {"copy16-self", 2 * kScan},
  }};
  const auto launch = [&](std::string_view test, std::uint64_t base) -> cudaError_t {
    if (test == "scan4") {
      return llmp::diag::ScanWords(stream, base, kScan, out);
    }
    if (test == "scan16") {
      return llmp::diag::ScanVector(stream, base, kScan, out, blocks);
    }
    if (test == "write16") {
      return llmp::diag::WriteVector(stream, base + kScan, kScan, blocks);
    }
    if (test == "write-sparse") {
      return llmp::diag::WriteSparse(stream, base + kScan, kSparse, 128, blocks);
    }
    if (test == "write-per-block") {
      return llmp::diag::WritePerBlock(stream, base + kScan, kBlocks);
    }
    if (test == "reread-4m") {
      return llmp::diag::Reread(stream, base, 4 * kMiB, 64, out, blocks);
    }
    if (test == "reread-4m-persist") {
      // The same, inside a persisting access-policy window over the buffer.
      cudaStreamAttrValue window{};
      window.accessPolicyWindow.base_ptr =
          reinterpret_cast<void*>(base);  // NOLINT(performance-no-int-to-ptr)
      window.accessPolicyWindow.num_bytes = 4 * kMiB;
      window.accessPolicyWindow.hitRatio = 1.0f;
      window.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
      window.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
      if (const cudaError_t error =
              cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &window);
          error != cudaSuccess) {
        return error;
      }
      const cudaError_t launched = llmp::diag::Reread(stream, base, 4 * kMiB, 64, out, blocks);
      window.accessPolicyWindow.num_bytes = 0;
      (void)cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &window);
      (void)cudaCtxResetPersistingL2Cache();
      return launched;
    }
    if (test == "reread-16m") {
      return llmp::diag::Reread(stream, base, 16 * kMiB, 16, out, blocks);
    }
    if (test == "reread-64m") {
      return llmp::diag::Reread(stream, base, 64 * kMiB, 4, out, blocks);
    }
    if (test == "rand-all") {
      return llmp::diag::RandomLines(stream, base, bytes, 0, 1, kLines, out, blocks);
    }
    if (test == "rand-2m") {
      return llmp::diag::RandomLines(stream, base, bytes, 2 * kMiB, kLines, kLines, out, blocks);
    }
    if (test == "rand-64k") {
      return llmp::diag::RandomLines(stream, base, bytes, std::uint64_t{64} << 10U, kLines, kLines,
                                     out, blocks);
    }
    if (test == "rand-8m-span") {
      return llmp::diag::RandomLines(stream, base, 8 * kMiB, 0, 1, kLines, out, blocks);
    }
    return llmp::diag::CopyVector(stream, base + kScan, base, kScan, blocks);
  };
  std::println("# micro,arm,test,bytes_moved,round,sample,us");
  Events events;
  constexpr int kWarm = 2;
  constexpr int kSamples = 10;
  for (int round = 0; round < rounds; ++round) {
    for (std::size_t a = 0; a < arms.size(); ++a) {
      for (const Test& test : tests) {
        if (!only.empty() && std::ranges::find(only, test.name) == only.end()) {
          continue;
        }
        for (int i = 0; i < kWarm + kSamples; ++i) {
          auto us = events.Time(stream, [&]() -> std::expected<void, std::string> {
            return Cuda(launch(test.name, buffers[a]), test.name);
          });
          if (!us) {
            std::println(stderr, "{}", us.error());
            return 1;
          }
          if (i >= kWarm) {
            std::println("micro,{},{},{},{},{},{:.3f}", ArmName(arms[a]), test.name, test.moved,
                         round, i - kWarm, *us);
          }
        }
      }
    }
  }
  (void)cudaFree(out_pointer);
  (void)cudaStreamDestroy(stream);
  return 0;
}

// ---- copy ----

int Copy(Memory& memory, Arm from, Arm to, int rounds) {
  int sms = 0;
  (void)cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  cudaStream_t stream = nullptr;
  if (auto got = Cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream"); !got) {
    std::println(stderr, "{}", got.error());
    return 1;
  }
  constexpr std::uint64_t kBytes = 1024 * kMiB;
  Held held(memory);
  auto source = held.Allocate(from, kBytes);
  auto destination = held.Allocate(to, kBytes);
  if (!source || !destination) {
    std::println(stderr, "allocation: {}", !source ? source.error() : destination.error());
    return 1;
  }
  if (auto got = Cuda(llmp::diag::Fill(stream, *source, kBytes / 4, 3), "fill"); !got) {
    std::println(stderr, "{}", got.error());
    return 1;
  }
  if (auto got = Cuda(llmp::diag::Fill(stream, *destination, kBytes / 4, 4), "fill"); !got) {
    std::println(stderr, "{}", got.error());
    return 1;
  }
  std::println("# copy,from,to,method,bytes,round,sample,us");
  Events events;
  constexpr int kWarm = 2;
  constexpr int kSamples = 10;
  for (int round = 0; round < rounds; ++round) {
    for (const std::uint64_t size : {2 * kMiB, 64 * kMiB, kBytes}) {
      for (const std::string_view method : {"memcpy", "kernel"}) {
        for (int i = 0; i < kWarm + kSamples; ++i) {
          // Successive 2 MiB copies walk through the buffer, as a restore's
          // extents would.
          const std::uint64_t offset =
              size == kBytes ? 0 : (static_cast<std::uint64_t>(i) * size) % kBytes;
          auto us = events.Time(stream, [&]() -> std::expected<void, std::string> {
            if (method == "memcpy") {
              return Cuda(
                  cudaMemcpyAsync(
                      reinterpret_cast<void*>(*destination +  // NOLINT(performance-no-int-to-ptr)
                                              offset),
                      reinterpret_cast<const void*>(*source +  // NOLINT(performance-no-int-to-ptr)
                                                    offset),
                      size, cudaMemcpyDefault, stream),
                  "cudaMemcpyAsync");
            }
            return Cuda(llmp::diag::CopyVector(stream, *destination + offset, *source + offset,
                                               size, sms * 16),
                        "copy kernel");
          });
          if (!us) {
            std::println(stderr, "{}", us.error());
            return 1;
          }
          if (i >= kWarm) {
            std::println("copy,{},{},{},{},{},{},{:.3f}", ArmName(from), ArmName(to), method, size,
                         round, i - kWarm, *us);
          }
        }
      }
    }
  }
  (void)cudaStreamDestroy(stream);
  return 0;
}

// ---- restore ----

struct Restore {
  std::string dir;
  std::uint64_t gib = 8;
  Arm landing = Arm::kHostVmm;
  Arm to = Arm::kDeviceVmm;
  std::string method = "memcpy";
  std::size_t depth = 4;
  int rounds = 3;
};

double Percentile(std::vector<double> samples, double p) {
  std::ranges::sort(samples);
  return samples[static_cast<std::size_t>(p * static_cast<double>(samples.size() - 1))];
}

int RunRestore(Memory& memory, const Restore& r) {
  using Clock = std::chrono::steady_clock;
  using llmp::providers::IoCompletion;
  using llmp::providers::IoKind;
  using llmp::providers::IoRequest;
  using llmp::providers::Submission;
  using llmp::providers::UringStorage;
  constexpr std::uint64_t kExtent = 2 * kMiB;
  const bool in_place = r.method == "none";
  if ((in_place && !CpuMapped(r.to)) || !CpuMapped(r.landing)) {
    std::println(stderr, "direct reads need a CPU-mapped destination or landing zone");
    return 2;
  }
  int sms = 0;
  (void)cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const std::uint64_t file_bytes = r.gib << 30U;
  Held held(memory);
  auto destination = held.Allocate(r.to, file_bytes);
  // Twice the read depth: a slot is busy from its read's submission until its
  // copy has completed.
  const std::size_t slots = in_place ? 0 : 2 * r.depth;
  auto landing = in_place ? destination : held.Allocate(r.landing, slots * kExtent);
  if (!destination || !landing) {
    std::println(stderr, "allocation: {}", !destination ? destination.error() : landing.error());
    return 1;
  }
  const int fd = open(r.dir.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);  // NOLINT
  if (fd < 0) {
    std::println(stderr, "cannot create a direct-I/O file in {}", r.dir);
    return 1;
  }
  // Non-zero contents, written through the landing (or destination) memory.
  std::fill_n(reinterpret_cast<unsigned char*>(*landing),  // NOLINT(performance-no-int-to-ptr)
              kExtent, static_cast<unsigned char>(0x5a));
  for (std::uint64_t offset = 0; offset < file_bytes; offset += kExtent) {
    if (pwrite(fd, reinterpret_cast<const void*>(*landing),  // NOLINT(performance-no-int-to-ptr)
               kExtent, static_cast<off_t>(offset)) != static_cast<ssize_t>(kExtent)) {
      std::println(stderr, "cannot write the file");
      (void)close(fd);
      return 1;
    }
  }
  (void)fsync(fd);
  auto storage = UringStorage::Create(r.depth);
  cudaStream_t stream = nullptr;
  if (!storage || cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    std::println(stderr, "no io_uring or stream");
    (void)close(fd);
    return 1;
  }
  std::vector<cudaEvent_t> copied(std::max<std::size_t>(slots, 1));
  for (cudaEvent_t& event : copied) {
    (void)cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  }
  std::println("# restore,landing,to,method,depth,round,gbps,p50_us,p99_us,extents");
  int status = 0;
  for (int round = 0; round < r.rounds && status == 0; ++round) {
    const std::uint64_t extents = file_bytes / kExtent;
    std::vector<Clock::time_point> started(extents);
    std::vector<double> latency_us;
    std::deque<std::size_t> free_slots;
    for (std::size_t slot = 0; slot < slots; ++slot) {
      free_slots.push_back(slot);
    }
    // Copies queued and not yet seen complete: (slot, extent).
    std::deque<std::pair<std::size_t, std::uint64_t>> copying;
    std::uint64_t next = 0;
    std::uint64_t done = 0;
    std::array<IoCompletion, 64> completions{};
    const Clock::time_point start = Clock::now();
    while (done < extents && status == 0) {
      while (next < extents && (*storage)->in_flight() < r.depth &&
             (in_place || !free_slots.empty())) {
        const std::size_t slot = in_place ? 0 : free_slots.front();
        const std::uint64_t target =
            in_place ? *destination + (next * kExtent) : *landing + (slot * kExtent);
        const IoRequest io{
            .token = in_place ? next : (next << 8U) | slot,
            .kind = IoKind::kRead,
            .fd = fd,
            .offset = next * kExtent,
            .memory = reinterpret_cast<std::byte*>(target),  // NOLINT(performance-no-int-to-ptr)
            .length = static_cast<std::uint32_t>(kExtent),
            .segments = {}};
        started[next] = Clock::now();
        if ((*storage)->Submit(io) != Submission::kAccepted) {
          break;
        }
        if (!in_place) {
          free_slots.pop_front();
        }
        ++next;
      }
      const std::size_t count = (*storage)->Harvest(completions, copying.empty());
      for (std::size_t c = 0; c < count; ++c) {
        if (std::cmp_not_equal(completions[c].result, kExtent)) {
          std::println(stderr, "a read returned {}", completions[c].result);
          status = 1;
          continue;
        }
        if (in_place) {
          latency_us.push_back(std::chrono::duration<double, std::micro>(
                                   Clock::now() - started[completions[c].token])
                                   .count());
          ++done;
          continue;
        }
        const std::uint64_t extent = completions[c].token >> 8U;
        const std::size_t slot = completions[c].token & 0xffU;
        const std::uint64_t from = *landing + (slot * kExtent);
        const std::uint64_t to = *destination + (extent * kExtent);
        const cudaError_t queued =
            r.method == "memcpy"
                ? cudaMemcpyAsync(reinterpret_cast<void*>(to),  // NOLINT(performance-no-int-to-ptr)
                                  reinterpret_cast<const void*>(from),  // NOLINT
                                  kExtent, cudaMemcpyDefault, stream)
                : llmp::diag::CopyVector(stream, to, from, kExtent, sms * 4);
        if (queued != cudaSuccess || cudaEventRecord(copied[slot], stream) != cudaSuccess) {
          std::println(stderr, "cannot queue a copy");
          status = 1;
          continue;
        }
        copying.emplace_back(slot, extent);
      }
      // Copies complete in stream order.
      while (!copying.empty() && cudaEventQuery(copied[copying.front().first]) == cudaSuccess) {
        const auto [slot, extent] = copying.front();
        copying.pop_front();
        latency_us.push_back(
            std::chrono::duration<double, std::micro>(Clock::now() - started[extent]).count());
        free_slots.push_back(slot);
        ++done;
      }
    }
    while ((*storage)->in_flight() > 0) {
      (void)(*storage)->Harvest(completions, true);
    }
    (void)cudaStreamSynchronize(stream);
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    if (status == 0) {
      std::println("restore,{},{},{},{},{},{:.3f},{:.0f},{:.0f},{}",
                   in_place ? "none" : ArmName(r.landing), ArmName(r.to), r.method, r.depth, round,
                   static_cast<double>(file_bytes) / seconds / 1e9, Percentile(latency_us, 0.5),
                   Percentile(latency_us, 0.99), extents);
    }
  }
  for (cudaEvent_t event : copied) {
    (void)cudaEventDestroy(event);
  }
  (void)cudaStreamDestroy(stream);
  (void)close(fd);
  return status;
}

// ---- ggml ----

// BP-F1's products (docs/experiments/backend-proof-p1/bpf1-cases.txt) with
// the implementation its recorded plan chose. The "weights" operand of an
// attention product is the KV cache (1,024 cells; `kv` in use), of 64-wide
// heads, 14 query heads over 2 KV heads.
enum class Kind : std::uint8_t { kLinear, kKq, kKqv };
enum class Impl : std::uint8_t { kMmvf, kMmf, kCublas };

struct Case {
  std::string_view name;
  Kind kind = Kind::kLinear;
  std::int64_t rows = 1;
  Impl impl = Impl::kMmvf;
  std::int64_t k = 0;  // linear: in and out features; attention: cells in use
  std::int64_t n = 0;
};

constexpr std::size_t kTensorsPerNode = 8;  // the most Node makes
constexpr std::int64_t kCacheCells = 1024;
constexpr std::int64_t kHead = 64;
constexpr std::int64_t kHeads = 14;
constexpr std::int64_t kKvHeads = 2;

constexpr std::array<Case, 21> kCases = {{
    {"linear.q_o", Kind::kLinear, 1, Impl::kMmvf, 896, 896},
    {"linear.k_v", Kind::kLinear, 1, Impl::kMmvf, 896, 128},
    {"linear.gate_up", Kind::kLinear, 1, Impl::kMmvf, 896, 4864},
    {"linear.down", Kind::kLinear, 1, Impl::kMmvf, 4864, 896},
    {"linear.lm_head", Kind::kLinear, 1, Impl::kMmvf, 896, 151936},
    {"attn.kq.kv768", Kind::kKq, 1, Impl::kMmf, 768, 0},
    {"attn.kqv.kv768", Kind::kKqv, 1, Impl::kMmvf, 768, 0},
    {"linear.q_o", Kind::kLinear, 16, Impl::kMmf, 896, 896},
    {"linear.k_v", Kind::kLinear, 16, Impl::kMmf, 896, 128},
    {"linear.gate_up", Kind::kLinear, 16, Impl::kMmf, 896, 4864},
    {"linear.down", Kind::kLinear, 16, Impl::kMmf, 4864, 896},
    {"linear.lm_head", Kind::kLinear, 16, Impl::kMmf, 896, 151936},
    {"attn.kq.kv256", Kind::kKq, 16, Impl::kMmf, 256, 0},
    {"attn.kqv.kv256", Kind::kKqv, 16, Impl::kMmf, 256, 0},
    {"linear.q_o", Kind::kLinear, 512, Impl::kCublas, 896, 896},
    {"linear.k_v", Kind::kLinear, 512, Impl::kCublas, 896, 128},
    {"linear.gate_up", Kind::kLinear, 512, Impl::kCublas, 896, 4864},
    {"linear.down", Kind::kLinear, 512, Impl::kCublas, 4864, 896},
    {"linear.lm_head", Kind::kLinear, 512, Impl::kCublas, 896, 151936},
    {"attn.kq.kv768", Kind::kKq, 512, Impl::kCublas, 768, 0},
    {"attn.kqv.kv768", Kind::kKqv, 512, Impl::kCublas, 768, 0},
}};

// Element counts of the weight (or cache) operand, the input and the output.
struct Shape {
  std::uint64_t weights = 0;
  std::uint64_t input = 0;
  std::uint64_t output = 0;
};

Shape ShapeOf(const Case& c) {
  const auto r = static_cast<std::uint64_t>(c.rows);
  const auto k = static_cast<std::uint64_t>(c.k);
  const auto n = static_cast<std::uint64_t>(c.n);
  constexpr auto kCache = static_cast<std::uint64_t>(kCacheCells * kHead * kKvHeads);
  switch (c.kind) {
    case Kind::kLinear:
      return {.weights = k * n, .input = k * r, .output = n * r};
    case Kind::kKq:
      return {.weights = kCache, .input = kHead * kHeads * r, .output = k * r * kHeads};
    case Kind::kKqv:
      return {.weights = kCache, .input = k * r * kHeads, .output = kHead * r * kHeads};
  }
  return {};
}

// llama.cpp's attention products without flash attention, as BP-F1 builds
// them (ggml_vmm_bench.cc).
ggml_tensor* Node(ggml_context* context, const Case& c, std::uint64_t weights, std::uint64_t input,
                  std::uint64_t output) {
  ggml_tensor* node = nullptr;
  switch (c.kind) {
    case Kind::kLinear: {
      ggml_tensor* w = ggml_new_tensor_2d(context, GGML_TYPE_F16, c.k, c.n);
      ggml_tensor* x = ggml_new_tensor_2d(context, GGML_TYPE_F32, c.k, c.rows);
      TensorArena::Bind(w, weights);
      TensorArena::Bind(x, input);
      node = ggml_mul_mat(context, w, x);
      break;
    }
    case Kind::kKq: {
      ggml_tensor* cache =
          ggml_new_tensor_2d(context, GGML_TYPE_F16, kHead * kKvHeads, kCacheCells);
      ggml_tensor* q = ggml_new_tensor_3d(context, GGML_TYPE_F32, kHead, kHeads, c.rows);
      TensorArena::Bind(cache, weights);
      TensorArena::Bind(q, input);
      ggml_tensor* k =
          ggml_permute(context,
                       ggml_view_3d(context, cache, kHead, kKvHeads, c.k,
                                    ggml_row_size(cache->type, kHead), cache->nb[1], 0),
                       0, 2, 1, 3);
      node = ggml_mul_mat(context, k, ggml_permute(context, q, 0, 2, 1, 3));
      (void)ggml_prec_set_acc(node, GGML_PREC_F32);
      break;
    }
    case Kind::kKqv: {
      ggml_tensor* cache =
          ggml_new_tensor_2d(context, GGML_TYPE_F16, kCacheCells, kHead * kKvHeads);
      ggml_tensor* p = ggml_new_tensor_3d(context, GGML_TYPE_F32, c.k, c.rows, kHeads);
      TensorArena::Bind(cache, weights);
      TensorArena::Bind(p, input);
      ggml_tensor* v = ggml_view_3d(context, cache, c.k, kHead, kKvHeads, cache->nb[1],
                                    cache->nb[1] * static_cast<std::size_t>(kHead), 0);
      node = ggml_mul_mat(context, v, p);
      break;
    }
  }
  TensorArena::Bind(node, output);
  return node;
}

struct Placement {
  Arm weights = Arm::kMalloc;
  Arm acts = Arm::kMalloc;  // inputs, and outputs unless split
  Arm scratch = Arm::kMalloc;
  Arm workspace = Arm::kMalloc;
  bool split = false;  // outputs in their own allocation, of `outputs`
  Arm outputs = Arm::kMalloc;
};

std::string PlacementName(const Placement& p) {
  std::string name = std::format("W={}/A={}/S={}/K={}", ArmName(p.weights), ArmName(p.acts),
                                 ArmName(p.scratch), ArmName(p.workspace));
  if (p.split) {
    name += std::format("/O={}", ArmName(p.outputs));
  }
  return name;
}

constexpr int kInvocations = 10;
constexpr int kWarmReplays = 5;
constexpr int kSampleReplays = 31;
constexpr int kReplays = kWarmReplays + kSampleReplays;

std::uint64_t Fnv(std::span<const std::byte> bytes) {
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  for (const std::byte b : bytes) {
    hash = (hash ^ std::to_integer<std::uint64_t>(b)) * 0x100000001b3ULL;
  }
  return hash;
}

class Graphs {
 public:
  Graphs() = default;
  Graphs(const Graphs&) = delete;
  Graphs& operator=(const Graphs&) = delete;
  Graphs(Graphs&&) = delete;
  Graphs& operator=(Graphs&&) = delete;
  ~Graphs() {
    for (cudaGraphExec_t exec : execs) {
      (void)cudaGraphExecDestroy(exec);
    }
    for (cudaGraph_t graph : graphs) {
      (void)cudaGraphDestroy(graph);
    }
  }
  std::vector<cudaGraph_t> graphs;
  std::vector<cudaGraphExec_t> execs;
};

struct Rig {
  Memory& memory;
  DeviceExecution& execution;
  StreamId stream;
  cudaStream_t native = nullptr;
  CublasHandle& cublas;
  std::uint64_t l2 = 0;
};

std::expected<void, std::string> RunCase(Rig& rig, const Case& c, const Placement& placement) {
  namespace ops = llmp::kernels::ggml;
  const Shape shape = ShapeOf(c);
  const std::uint64_t weight_bytes = RoundUp(shape.weights * 2, kAlign);
  const std::uint64_t input_bytes = RoundUp(shape.input * 4, kAlign);
  const std::uint64_t output_bytes = RoundUp(shape.output * 4, kAlign);
  const std::uint64_t act_bytes = input_bytes + (placement.split ? 0 : output_bytes);
  constexpr std::uint64_t kUsed = static_cast<std::uint64_t>(kReplays) * kInvocations;
  const auto ring = [&](std::uint64_t set) {
    return std::min(kUsed, std::max<std::uint64_t>(2, (4 * rig.l2 / set) + 1));
  };
  const std::uint64_t weight_sets = ring(weight_bytes);
  const std::uint64_t act_sets = ring(act_bytes);

  Held held(rig.memory);
  auto weights = held.Allocate(placement.weights, weight_sets * weight_bytes);
  if (!weights) {
    return std::unexpected(weights.error());
  }
  auto acts = held.Allocate(placement.acts, act_sets * act_bytes);
  if (!acts) {
    return std::unexpected(acts.error());
  }
  // Where set s's output is.
  std::uint64_t outputs_base = 0;
  if (placement.split) {
    auto allocated = held.Allocate(placement.outputs, act_sets * output_bytes);
    if (!allocated) {
      return std::unexpected(allocated.error());
    }
    outputs_base = *allocated;
  }
  const auto output_of = [&](std::uint64_t set) {
    return placement.split ? outputs_base + (set * output_bytes)
                           : *acts + (set * act_bytes) + input_bytes;
  };
  for (std::uint64_t s = 0; s < weight_sets; ++s) {
    if (auto got =
            Cuda(llmp::diag::FillHalf(rig.native, *weights + (s * weight_bytes), shape.weights, 1),
                 "fill");
        !got) {
      return got;
    }
  }
  for (std::uint64_t s = 0; s < act_sets; ++s) {
    if (auto got = Cuda(llmp::diag::FillFloat(rig.native, *acts + (s * act_bytes), shape.input, 2),
                        "fill");
        !got) {
      return got;
    }
  }
  if (auto got = Cuda(cudaStreamSynchronize(rig.native), "sync"); !got) {
    return got;
  }

  auto arena = TensorArena::Create(kUsed * kTensorsPerNode);
  if (!arena) {
    return Fail("tensor arena: {}", arena.error().detail);
  }
  std::vector<ggml_tensor*> nodes;
  for (std::uint64_t i = 0; i < kUsed; ++i) {
    if (!arena->Reserve(kTensorsPerNode)) {
      return Fail("the tensor arena is full");
    }
    const std::uint64_t set = i % act_sets;
    ggml_tensor* node = Node(arena->context(), c, *weights + ((i % weight_sets) * weight_bytes),
                             *acts + (set * act_bytes), output_of(set));
    if (ggml_nbytes(node) != shape.output * 4 || !ggml_is_contiguous(node)) {
      return Fail("{}: the output is not the {} bytes laid out", c.name, shape.output * 4);
    }
    nodes.push_back(node);
  }

  std::uint64_t scratch = 0;
  if (c.impl == Impl::kCublas) {
    auto planner = LaunchContext::Create(0, rig.execution, rig.stream,
                                         {.base = 0, .size = Bytes(0)}, &rig.cublas);
    if (!planner) {
      return Fail("launch context: {}", planner.error().detail);
    }
    auto plan = ops::PlanMulMatCublas(**planner, nodes[0]);
    if (!plan) {
      return Fail("plan: {}", plan.error().detail);
    }
    scratch = plan->scratch;
  }
  std::uint64_t scratch_base = 0;
  if (scratch > 0) {
    auto allocated = held.Allocate(placement.scratch, scratch);
    if (!allocated) {
      return std::unexpected(allocated.error());
    }
    scratch_base = *allocated;
  }
  auto launch = LaunchContext::Create(0, rig.execution, rig.stream,
                                      {.base = scratch_base, .size = Bytes(scratch)}, &rig.cublas);
  if (!launch) {
    return Fail("launch context: {}", launch.error().detail);
  }
  const auto invoke = [&](std::uint64_t i) -> std::expected<void, std::string> {
    std::expected<void, KernelFailure> ran;
    switch (c.impl) {
      case Impl::kMmvf:
        ran = ops::MulMatVecF(**launch, nodes[i]);
        break;
      case Impl::kMmf:
        ran = ops::MulMatF(**launch, nodes[i]);
        break;
      case Impl::kCublas:
        ran = ops::MulMatCublas(**launch, nodes[i]);
        break;
    }
    if (!ran) {
      return Fail("{} at {} rows: {}", c.name, c.rows, ran.error().detail);
    }
    return {};
  };

  // Output hash of set 0 after one eager invocation.
  if (auto ran = invoke(0); !ran) {
    return ran;
  }
  std::vector<std::byte> output(shape.output * 4);
  if (auto got =
          Cuda(cudaMemcpyAsync(output.data(),
                               reinterpret_cast<const void*>(  // NOLINT(performance-no-int-to-ptr)
                                   output_of(0)),
                               output.size(), cudaMemcpyDefault, rig.native),
               "download");
      !got) {
    return got;
  }
  if (auto got = Cuda(cudaStreamSynchronize(rig.native), "sync"); !got) {
    return got;
  }
  const std::uint64_t hash = Fnv(output);

  Graphs graphs;
  for (int replay = 0; replay < kReplays; ++replay) {
    if (auto got = Cuda(cudaStreamBeginCapture(rig.native, cudaStreamCaptureModeThreadLocal),
                        "begin capture");
        !got) {
      return got;
    }
    std::expected<void, std::string> captured;
    for (int i = 0; i < kInvocations && captured; ++i) {
      captured = invoke((static_cast<std::uint64_t>(replay) * kInvocations) +
                        static_cast<std::uint64_t>(i));
    }
    cudaGraph_t graph = nullptr;
    const cudaError_t ended = cudaStreamEndCapture(rig.native, &graph);
    if (graph != nullptr) {
      graphs.graphs.push_back(graph);
    }
    if (!captured) {
      return captured;
    }
    if (auto got = Cuda(ended, "end capture"); !got) {
      return got;
    }
    cudaGraphExec_t exec = nullptr;
    if (auto got = Cuda(cudaGraphInstantiate(&exec, graph, 0), "instantiate"); !got) {
      return got;
    }
    graphs.execs.push_back(exec);
  }
  Events events;
  std::vector<double> samples;
  for (int replay = 0; replay < kReplays; ++replay) {
    auto us = events.Time(rig.native, [&]() -> std::expected<void, std::string> {
      return Cuda(cudaGraphLaunch(graphs.execs[static_cast<std::size_t>(replay)], rig.native),
                  "graph launch");
    });
    if (!us) {
      return std::unexpected(us.error());
    }
    if (replay >= kWarmReplays) {
      samples.push_back(*us / kInvocations);
    }
  }
  for (std::size_t i = 0; i < samples.size(); ++i) {
    std::println("ggml,{},{},{},{},{},{},{:016x},{:.4f}", PlacementName(placement), c.name, c.rows,
                 weight_bytes, weight_sets, i, hash, samples[i]);
  }
  std::vector<double> sorted = samples;
  std::ranges::sort(sorted);
  std::println(stderr, "{} {}@{}: median {:.3f} us, {} weight sets, {} act sets",
               PlacementName(placement), c.name, c.rows, sorted[sorted.size() / 2], weight_sets,
               act_sets);
  return {};
}

int Ggml(Memory& memory, const Placement& placement, std::span<const std::string> only) {
  auto execution = llmp::providers::cuda::OpenDeviceExecution(0);
  if (!execution) {
    std::println(stderr, "no device execution: {}", execution.error().detail);
    return 1;
  }
  auto stream = (*execution)->CreateStream();
  if (!stream) {
    std::println(stderr, "no stream: {}", stream.error().detail);
    return 1;
  }
  auto native = (*execution)->Submission(*stream);
  if (!native) {
    std::println(stderr, "no native stream: {}", native.error().detail);
    return 1;
  }
  int l2 = 0;
  int major = 0;
  int minor = 0;
  (void)cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);
  (void)cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  (void)cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
  const Bytes workspace_size = CublasHandle::UpstreamWorkspace((100 * major) + (10 * minor));
  int status = 0;
  {
    Held held(memory);
    auto workspace = held.Allocate(placement.workspace, workspace_size.value());
    if (!workspace) {
      std::println(stderr, "{}", workspace.error());
      return 1;
    }
    auto cublas =
        CublasHandle::Create(0, **execution, *stream, {.base = *workspace, .size = workspace_size});
    if (!cublas) {
      std::println(stderr, "cuBLAS handle: {}", cublas.error().detail);
      return 1;
    }
    Rig rig{.memory = memory,
            .execution = **execution,
            .stream = *stream,
            .native = static_cast<cudaStream_t>(native->handle),
            .cublas = **cublas,
            .l2 = static_cast<std::uint64_t>(l2)};
    std::println("# ggml,placement,case,rows,weight_set_bytes,weight_sets,sample,output_fnv,us");
    for (const Case& c : kCases) {
      const std::string key = std::format("{}@{}", c.name, c.rows);
      if (!only.empty() && std::ranges::find(only, key) == only.end()) {
        continue;
      }
      if (auto ran = RunCase(rig, c, placement); !ran) {
        std::println(stderr, "{}", ran.error());
        status = 1;
        break;
      }
    }
    (void)cudaStreamSynchronize(rig.native);
  }
  Discard((*execution)->DestroyStream(*stream));
  return status;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(
      argv, argv + argc);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  if (args.size() < 2) {
    std::println(stderr, "usage: llmp_vmm_diag_bench info|micro|copy|ggml [options]");
    return 2;
  }
  const std::string& mode = args[1];
  std::vector<Arm> arms;
  std::vector<std::string> only;
  std::string file;
  Restore restore;
  Placement placement;
  Arm from = Arm::kHostVmm;
  Arm to = Arm::kDeviceVmm;
  int rounds = 3;
  std::uint64_t bytes = 1024 * kMiB;
  for (std::size_t i = 2; i + 1 < args.size(); i += 2) {
    const std::string& flag = args[i];
    const std::string& value = args[i + 1];
    if (flag == "--rounds") {
      std::from_chars(value.data(), value.data() + value.size(), rounds);
      continue;
    }
    if (flag == "--bytes") {
      std::from_chars(value.data(), value.data() + value.size(), bytes);
      continue;
    }
    if (flag == "--file") {
      file = value;
      continue;
    }
    if (flag == "--dir") {
      restore.dir = value;
      continue;
    }
    if (flag == "--method") {
      restore.method = value;
      continue;
    }
    if (flag == "--gib") {
      std::from_chars(value.data(), value.data() + value.size(), restore.gib);
      continue;
    }
    if (flag == "--depth") {
      std::from_chars(value.data(), value.data() + value.size(), restore.depth);
      continue;
    }
    if (flag == "--case" || flag == "--test") {
      only.push_back(value);
      continue;
    }
    auto arm = ParseArm(value);
    if (!arm) {
      std::println(stderr, "{}", arm.error());
      return 2;
    }
    if (flag == "--arm") {
      arms.push_back(*arm);
    } else if (flag == "--from") {
      from = *arm;
    } else if (flag == "--to") {
      to = *arm;
    } else if (flag == "--landing") {
      restore.landing = *arm;
    } else if (flag == "--weights") {
      placement.weights = *arm;
    } else if (flag == "--acts") {
      placement.acts = *arm;
    } else if (flag == "--outputs") {
      placement.split = true;
      placement.outputs = *arm;
    } else if (flag == "--scratch") {
      placement.scratch = *arm;
    } else if (flag == "--workspace") {
      placement.workspace = *arm;
    } else {
      std::println(stderr, "unknown flag {}", flag);
      return 2;
    }
  }
  auto memory = Memory::Open();
  if (!memory) {
    std::println(stderr, "{}", memory.error());
    return 1;
  }
  if (mode == "info") {
    return Info(**memory, arms, file);
  }
  if (mode == "micro") {
    if (arms.empty()) {
      std::println(stderr, "micro needs --arm");
      return 2;
    }
    return Micro(**memory, arms, rounds, bytes, only);
  }
  if (mode == "copy") {
    return Copy(**memory, from, to, rounds);
  }
  if (mode == "restore") {
    restore.to = to;
    restore.rounds = rounds;
    const bool method_known =
        restore.method == "none" || restore.method == "memcpy" || restore.method == "kernel";
    if (restore.dir.empty() || !method_known || restore.depth == 0 || restore.depth > 64) {
      std::println(stderr, "restore needs --dir, --method none|memcpy|kernel and a depth of 1-64");
      return 2;
    }
    return RunRestore(**memory, restore);
  }
  if (mode == "ggml") {
    return Ggml(**memory, placement, only);
  }
  std::println(stderr, "unknown mode {}", mode);
  return 2;
}
