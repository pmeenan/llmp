// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "providers/cuda/cuda_probe.h"

#include <cuda.h>
#include <dlfcn.h>
#include <link.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <optional>
#include <string>

#include "base/report.h"
#include "platform/host_probe.h"
#include "providers/cuda/cuda_facts.h"
#include "providers/device_probe.h"

#ifndef JITLLM_CUDA_ARCHITECTURES
#error "the build defines JITLLM_CUDA_ARCHITECTURES (src/providers/cuda/CMakeLists.txt)"
#endif

namespace jitllm::providers::cuda {
namespace {

std::string ErrorText(CUresult result, const char* call) {
  const char* name = nullptr;
  if (result == CUDA_ERROR_STUB_LIBRARY) {
    // The stub cannot name its own error.
    name = "CUDA_ERROR_STUB_LIBRARY: NVIDIA's link stub was loaded, not the driver";
  } else if (cuGetErrorName(result, &name) != CUDA_SUCCESS || name == nullptr) {
    name = "unknown error";
  }
  return std::format("{} returned {} ({})", call, name, static_cast<int>(result));
}

std::optional<int> Attribute(CUdevice device, CUdevice_attribute attribute) {
  int value = 0;
  if (cuDeviceGetAttribute(&value, attribute, device) != CUDA_SUCCESS) {
    return std::nullopt;
  }
  return value;
}

std::optional<bool> Flag(CUdevice device, CUdevice_attribute attribute) {
  const auto value = Attribute(device, attribute);
  return value ? std::optional<bool>(*value != 0) : std::nullopt;
}

CudaGranularity Granularity(CUmemLocationType type, int id) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = type;
  prop.location.id = id;
  std::size_t minimum = 0;
  std::size_t recommended = 0;
  CUresult result =
      cuMemGetAllocationGranularity(&minimum, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (result == CUDA_SUCCESS) {
    result =
        cuMemGetAllocationGranularity(&recommended, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  }
  CudaGranularity granularity;
  if (result != CUDA_SUCCESS) {
    granularity.error = ErrorText(result, "cuMemGetAllocationGranularity");
  } else {
    granularity.minimum = minimum;
    granularity.recommended = recommended;
  }
  return granularity;
}

CudaDeviceFacts ProbeDevice(int ordinal) {
  CudaDeviceFacts facts;
  facts.ordinal = ordinal;
  CUdevice device = 0;
  if (const CUresult result = cuDeviceGet(&device, ordinal); result != CUDA_SUCCESS) {
    facts.error = ErrorText(result, "cuDeviceGet");
    return facts;
  }
  std::array<char, 256> name{};
  if (const CUresult result =
          cuDeviceGetName(name.data(), static_cast<int>(name.size() - 1), device);
      result != CUDA_SUCCESS) {
    facts.error = ErrorText(result, "cuDeviceGetName");
    return facts;
  }
  facts.name = name.data();
  const auto major = Attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
  const auto minor = Attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
  if (!major || !minor) {
    facts.error = "cannot query the compute capability";
    return facts;
  }
  facts.major = *major;
  facts.minor = *minor;
  facts.integrated = Flag(device, CU_DEVICE_ATTRIBUTE_INTEGRATED);
  facts.vmm = Flag(device, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED);
  facts.host_numa_vmm =
      Flag(device, CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED);
  facts.gpu_direct_rdma = Flag(device, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED);
  facts.host_numa_id = Attribute(device, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID);
  if (const auto mode = Attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE)) {
    switch (*mode) {
      case CU_COMPUTEMODE_DEFAULT:
        facts.compute_mode = CudaComputeMode::kDefault;
        break;
      case CU_COMPUTEMODE_EXCLUSIVE_PROCESS:
        facts.compute_mode = CudaComputeMode::kExclusiveProcess;
        break;
      case CU_COMPUTEMODE_PROHIBITED:
        facts.compute_mode = CudaComputeMode::kProhibited;
        break;
      default:
        facts.compute_mode = CudaComputeMode::kOther;
        break;
    }
  }
  std::size_t bytes = 0;
  if (cuDeviceTotalMem(&bytes, device) == CUDA_SUCCESS) {
    facts.memory_bytes = bytes;
  }
  if (facts.vmm == true) {
    facts.device_local = Granularity(CU_MEM_LOCATION_TYPE_DEVICE, device);
  }
  if (facts.host_numa_vmm == true && facts.host_numa_id && *facts.host_numa_id >= 0) {
    facts.host_numa = Granularity(CU_MEM_LOCATION_TYPE_HOST_NUMA, *facts.host_numa_id);
  }
  return facts;
}

// Where the loader found the driver library this binary links.
std::string LibraryPath() {
  std::string path = "unknown";
  void* handle = ::dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
  if (handle == nullptr) {
    return path;
  }
  link_map* map = nullptr;
  if (::dlinfo(handle, RTLD_DI_LINKMAP, static_cast<void*>(&map)) == 0 && map != nullptr &&
      map->l_name != nullptr && map->l_name[0] != '\0') {
    path = map->l_name;
  }
  (void)::dlclose(handle);  // drops only the reference RTLD_NOLOAD added
  return path;
}

}  // namespace

CudaFacts ProbeCuda(const std::filesystem::path& root) {
  CudaFacts facts;
  facts.built_version = CUDA_VERSION;
  facts.built_architectures = ParseCudaArchitectures(JITLLM_CUDA_ARCHITECTURES);
  facts.nvidia = platform::FindKernelModule(root, "nvidia");
  facts.nvidia_fs = platform::FindKernelModule(root, "nvidia_fs");
  facts.library = LibraryPath();
  int version = 0;
  if (cuDriverGetVersion(&version) == CUDA_SUCCESS && version > 0) {
    facts.driver_version = version;
  }
  if (const CUresult result = cuInit(0); result != CUDA_SUCCESS) {
    facts.init_error = ErrorText(result, "cuInit");
    return facts;
  }
  int count = 0;
  if (const CUresult result = cuDeviceGetCount(&count); result != CUDA_SUCCESS) {
    facts.init_error = ErrorText(result, "cuDeviceGetCount");
    return facts;
  }
  facts.device_count = count;
  for (int ordinal = 0; ordinal < std::min(count, kMaxProbedDevices); ++ordinal) {
    facts.devices.push_back(ProbeDevice(ordinal));
  }
  return facts;
}

}  // namespace jitllm::providers::cuda

namespace jitllm::providers {

void DescribeDevices(const std::filesystem::path& root, base::Report& report) {
  cuda::DescribeCuda(cuda::ProbeCuda(root), report);
}

std::string DeviceIdentity(const std::filesystem::path& root) {
  const cuda::CudaFacts facts = cuda::ProbeCuda(root);
  std::string device = "no device";
  for (const cuda::CudaDeviceFacts& d : facts.devices) {
    if (d.ordinal == cuda::kUsedDevice && d.error.empty()) {
      device = std::format("{} sm_{}{}", d.name, d.major, d.minor);
    }
  }
  return std::format("{}, driver {}, CUDA {}", device,
                     facts.nvidia.version.empty() ? "unknown" : facts.nvidia.version,
                     facts.driver_version > 0 ? cuda::CudaVersionText(facts.driver_version)
                                              : std::string("unknown"));
}

}  // namespace jitllm::providers
