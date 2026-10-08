// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// What the CUDA device probe found, in plain types, and the judgment of it
// (D-006, D-026, D-033, D-034, D-081, D-082). ProbeCuda() (cuda_probe.h)
// fills CudaFacts from the driver; DescribeCuda() needs no driver, so tests
// give it any facts.

#ifndef LLMP_PROVIDERS_CUDA_CUDA_FACTS_H_
#define LLMP_PROVIDERS_CUDA_CUDA_FACTS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "base/report.h"
#include "platform/host_probe.h"

namespace llmp::providers::cuda {

// The allocation granularity of one backing class. `error` says why it
// could not be queried; it is empty when the sizes are valid.
struct CudaGranularity {
  std::uint64_t minimum = 0;
  std::uint64_t recommended = 0;
  std::string error;
};

// How a GPU's memory relates to the host's (D-004, D-082).
enum class CudaDeviceClass : std::uint8_t {
  kUnified,   // integrated: one physical budget with the host, as the GB10's
  kDiscrete,  // its own device memory, a separate budget from the host's
};

// Whether processes may create contexts on the device (nvidia-smi -c).
enum class CudaComputeMode { kDefault, kExclusiveProcess, kProhibited, kOther };

struct CudaDeviceFacts {
  int ordinal = 0;
  std::string name;
  // Why the device could not be queried; empty if it was.
  std::string error;
  int major = 0;
  int minor = 0;
  // Attributes, absent where the driver could not answer.
  std::optional<bool> integrated;
  std::optional<bool> vmm;
  std::optional<bool> host_numa_vmm;
  std::optional<bool> gpu_direct_rdma;
  std::optional<int> host_numa_id;
  std::optional<CudaComputeMode> compute_mode;
  std::optional<std::uint64_t> memory_bytes;
  // Device-local backing, queried when the device has VMM.
  std::optional<CudaGranularity> device_local;
  // Backing on the device's host NUMA node (D-034), queried when the device
  // supports it and has a NUMA node.
  std::optional<CudaGranularity> host_numa;
};

struct CudaFacts {
  // CUDA_VERSION of the headers this build used, such as 13040 for 13.4.
  int built_version = 0;
  // The SASS architectures this build compiled, such as {121} for sm_121.
  std::vector<int> built_architectures;
  // Where the loader found libcuda.so.1.
  std::string library;
  // cuDriverGetVersion(), such as 13000 for CUDA 13.0; 0 if unknown.
  int driver_version = 0;
  std::string init_error;
  // What cuDeviceGetCount() reported; the probe queries at most
  // kMaxProbedDevices of them.
  int device_count = 0;
  std::vector<CudaDeviceFacts> devices;
  platform::KernelModule nvidia;
  platform::KernelModule nvidia_fs;
};

inline constexpr int kMaxProbedDevices = 64;

// The one GPU llmpalooza uses: the driver's device 0, after CUDA_VISIBLE_DEVICES
// (D-082). Multi-GPU hosts are out of scope; other GPUs are reported only.
inline constexpr int kUsedDevice = 0;

// The largest backing granularity a targeted GPU may have: D-056's
// artifacts page in 2 MiB chunks.
inline constexpr std::uint64_t kPagingChunkBytes = std::uint64_t{2} << 20;

// CMAKE_CUDA_ARCHITECTURES, joined with commas ("121-real,90"), as the
// architecture numbers the build has SASS for ({121, 90}). A `-virtual`
// (PTX-only) entry, or one without a number, is skipped.
std::vector<int> ParseCudaArchitectures(std::string_view architectures);

// "13.4" for 13040.
std::string CudaVersionText(int version);

// The device's class from its integrated attribute; nullopt if unknown.
std::optional<CudaDeviceClass> DeviceClassOf(const CudaDeviceFacts& device);

// The class of GPU this build's code for `architecture` is for: sm_121 is
// the GB10's, unified; every other architecture a build targets is a
// discrete GPU's (D-082).
CudaDeviceClass TargetClass(int architecture);

// Adds the `NVIDIA driver` section and one section per GPU, with the
// problems and warnings they show. Only GPU 0 (kUsedDevice) is judged.
void DescribeCuda(const CudaFacts& facts, base::Report& report);

}  // namespace llmp::providers::cuda

#endif  // LLMP_PROVIDERS_CUDA_CUDA_FACTS_H_
