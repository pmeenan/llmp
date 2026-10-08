// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "providers/cuda/cuda_device_memory.h"

#include <cuda.h>

#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "providers/cuda/cuda_errors.h"
#include "providers/device_memory.h"

namespace llmp::providers::cuda {
namespace {

CUmemLocation Location(const AllocationClass& allocation_class) {
  CUmemLocation location{};
  location.type = allocation_class.kind == BackingKind::kHost ? CU_MEM_LOCATION_TYPE_HOST_NUMA
                                                              : CU_MEM_LOCATION_TYPE_DEVICE;
  location.id = static_cast<int>(allocation_class.location);
  return location;
}

CUmemAllocationProp Properties(const AllocationClass& allocation_class) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location = Location(allocation_class);
  return prop;
}

CUdeviceptr Pointer(std::uint64_t address) { return static_cast<CUdeviceptr>(address); }

class CudaDeviceMemory final : public VmmProvider {
 public:
  CudaDeviceMemory(CUdevice device, CUcontext context, std::vector<AllocationClass> classes,
                   Bytes granularity)
      : VmmProvider(granularity),
        device_(device),
        context_(context),
        classes_(std::move(classes)) {}

  ~CudaDeviceMemory() override { (void)cuDevicePrimaryCtxRelease(device_); }

  CudaDeviceMemory(const CudaDeviceMemory&) = delete;
  CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;
  CudaDeviceMemory(CudaDeviceMemory&&) = delete;
  CudaDeviceMemory& operator=(CudaDeviceMemory&&) = delete;

  std::span<const AllocationClass> Classes() const override { return classes_; }

 protected:
  std::expected<std::uint64_t, Failure> DoReserve(Bytes size) override {
    if (auto current = Current(); !current) {
      return std::unexpected(current.error());
    }
    CUdeviceptr base = 0;
    if (const CUresult result =
            cuMemAddressReserve(&base, size.value(), Granularity().value(), 0, 0);
        result != CUDA_SUCCESS) {
      return Error(result, "cuMemAddressReserve");
    }
    return static_cast<std::uint64_t>(base);
  }

  std::expected<void, Failure> DoFree(std::uint64_t base, Bytes size) override {
    return Call(cuMemAddressFree(Pointer(base), size.value()), "cuMemAddressFree");
  }

  std::expected<Handle, Failure> DoCreate(const AllocationClass& allocation_class,
                                          Bytes size) override {
    if (auto current = Current(); !current) {
      return std::unexpected(current.error());
    }
    const CUmemAllocationProp prop = Properties(allocation_class);
    CUmemGenericAllocationHandle handle = 0;
    if (const CUresult result = cuMemCreate(&handle, size.value(), &prop, 0);
        result != CUDA_SUCCESS) {
      return Error(result, "cuMemCreate");
    }
    return static_cast<Handle>(handle);
  }

  std::expected<void, Failure> DoRelease(Handle handle, Bytes /*size*/) override {
    return Call(cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handle)), "cuMemRelease");
  }

  std::expected<void, Failure> DoMap(std::uint64_t address, Bytes size, Handle handle) override {
    return Call(cuMemMap(Pointer(address), size.value(), 0,
                         static_cast<CUmemGenericAllocationHandle>(handle), 0),
                "cuMemMap");
  }

  std::expected<void, Failure> DoSetAccess(std::uint64_t address, Bytes size, Access access,
                                           bool host) override {
    CUmemAccess_flags flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    if (access == Access::kRead) {
      flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
    } else if (access == Access::kReadWrite) {
      flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    std::vector<CUmemAccessDesc> descriptors;
    for (const AllocationClass& allocation_class : classes_) {
      // The device always; the CPU on the host node when the range holds
      // host backing, so direct I/O and the device share one address.
      if (allocation_class.kind == BackingKind::kDevice || host) {
        CUmemAccessDesc descriptor{};
        descriptor.location = Location(allocation_class);
        descriptor.flags = flags;
        descriptors.push_back(descriptor);
      }
    }
    return Call(
        cuMemSetAccess(Pointer(address), size.value(), descriptors.data(), descriptors.size()),
        "cuMemSetAccess");
  }

  std::expected<void, Failure> DoUnmap(std::uint64_t address, Bytes size) override {
    return Call(cuMemUnmap(Pointer(address), size.value()), "cuMemUnmap");
  }

 private:
  std::expected<void, Failure> Current() {
    if (const CUresult result = cuCtxSetCurrent(context_); result != CUDA_SUCCESS) {
      return Error(result, "cuCtxSetCurrent");
    }
    return {};
  }

  static std::expected<void, Failure> Call(CUresult result, const char* call) {
    if (result != CUDA_SUCCESS) {
      return Error(result, call);
    }
    return {};
  }

  CUdevice device_;
  CUcontext context_;
  std::vector<AllocationClass> classes_;
};

std::expected<Bytes, Failure> MinimumGranularity(const AllocationClass& allocation_class) {
  const CUmemAllocationProp prop = Properties(allocation_class);
  std::size_t granularity = 0;
  if (const CUresult result =
          cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
      result != CUDA_SUCCESS) {
    return Error(result, "cuMemGetAllocationGranularity");
  }
  if (granularity == 0) {
    return std::unexpected(
        Failure{.error = ProviderError::kUnsupported, .detail = "a zero allocation granularity"});
  }
  return Bytes(granularity);
}

}  // namespace

std::expected<std::unique_ptr<VmmProvider>, Failure> OpenDeviceMemory(int ordinal) {
  if (const CUresult result = cuInit(0); result != CUDA_SUCCESS) {
    return Error(result, "cuInit");
  }
  CUdevice device = 0;
  if (const CUresult result = cuDeviceGet(&device, ordinal); result != CUDA_SUCCESS) {
    return Error(result, "cuDeviceGet");
  }
  int vmm = 0;
  if (const CUresult result = cuDeviceGetAttribute(
          &vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
      result != CUDA_SUCCESS || vmm == 0) {
    return std::unexpected(
        Failure{.error = ProviderError::kUnsupported, .detail = "the device has no VMM"});
  }
  std::vector<AllocationClass> classes = {
      AllocationClass{.kind = BackingKind::kDevice,
                      .location = static_cast<std::uint32_t>(ordinal),
                      .granularity = {}}};
  int host_vmm = 0;
  int host_node = -1;
  if (cuDeviceGetAttribute(&host_vmm,
                           CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                           device) == CUDA_SUCCESS &&
      host_vmm != 0 &&
      cuDeviceGetAttribute(&host_node, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device) == CUDA_SUCCESS &&
      host_node >= 0) {
    classes.push_back(AllocationClass{.kind = BackingKind::kHost,
                                      .location = static_cast<std::uint32_t>(host_node),
                                      .granularity = {}});
  }
  std::uint64_t common = 1;
  for (AllocationClass& allocation_class : classes) {
    auto granularity = MinimumGranularity(allocation_class);
    if (!granularity) {
      return std::unexpected(granularity.error());
    }
    allocation_class.granularity = *granularity;
    common = std::lcm(common, granularity->value());
  }
  CUcontext context = nullptr;
  if (const CUresult result = cuDevicePrimaryCtxRetain(&context, device); result != CUDA_SUCCESS) {
    return Error(result, "cuDevicePrimaryCtxRetain");
  }
  return std::make_unique<CudaDeviceMemory>(device, context, std::move(classes), Bytes(common));
}

}  // namespace llmp::providers::cuda
