// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// jitLLM's definitions of the ggml-cuda.cu symbols that GGML's CUDA
// operation launchers use (D-053; docs/backend-proof.md#ggml-llamacpp-b29c606e2).
// jitLLM does not compile ggml-cuda.cu, GGML's CUDA backend runtime; these
// are adapted from it at llama.cpp b29c606e2:
//
// - ggml_cuda_error records the first failure on this thread for the launch
//   context (launch.h) and returns, where upstream aborts. The build's
//   GGML_JITLLM drops its [[noreturn]] (third_party/patches/ggml/).
// - ggml_cuda_set_device and ggml_cuda_get_device are upstream's, with no
//   virtual devices, and get_device returns device 0 after a failure
//   instead of an uninitialized value.
// - ggml_cuda_info reports upstream's device properties without upstream's
//   side effects: no process-wide scheduling flag (upstream sets
//   cudaDeviceScheduleSpin on compute capability 12.1), no peer access, no
//   environment and no logging.
// - The launch context's destructor destroys nothing, since jitLLM lends
//   its stream and pool and takes them back first; new_pool_for_device is
//   fatal, since jitLLM always lends a pool.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "base/check.h"
#include "kernels/ggml/ggml_support.h"
// GGML's CUDA definitions, with the driver and runtime headers.
#include "common.cuh"

namespace {

// The first CUDA failure a launcher recorded on this thread and not yet
// taken. Launches run on the thread that runs their job: the device
// submission lane (D-048), or a driver's direct step (engine/paged_node.h).
thread_local std::optional<std::string> recorded_error;

ggml_cuda_device_info DeviceInfo() {
  // Only the read's own failures void the table. An error already pending
  // on this thread is put back first, and a failure of the read is
  // appended to it.
  std::optional<std::string> pending = std::exchange(recorded_error, std::nullopt);
  ggml_cuda_device_info info = {};
  CUDA_CHECK(cudaGetDeviceCount(&info.physical_device_count));
  info.physical_device_count = std::clamp(info.physical_device_count, 0, GGML_CUDA_MAX_DEVICES);
  info.device_count = info.physical_device_count;
  for (int id = 0; id < info.device_count; ++id) {
    ggml_cuda_device_info::cuda_device_info& device = info.devices[id];
    device.physical_device = id;
    device.physical_share_count = 1;
    device.virtual_index = 0;

    int vmm = 0;
    CUdevice handle = 0;
    CU_CHECK(cuDeviceGet(&handle, id));
    CU_CHECK(cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                  handle));
    if (vmm != 0) {
      CUmemAllocationProp prop = {};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = id;
      CU_CHECK(cuMemGetAllocationGranularity(&device.vmm_granularity, &prop,
                                             CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    }
    device.vmm = vmm != 0;

    cudaDeviceProp prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
    device.total_vram = prop.totalGlobalMem;
    info.default_tensor_split[static_cast<std::size_t>(id)] = 0.0f;
    device.integrated = false;  // as upstream, which disables it (llama.cpp #15034)
    device.nsm = prop.multiProcessorCount;
    device.smpb = prop.sharedMemPerBlock;
    device.warp_size = prop.warpSize;
    int cooperative = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, id));
    device.supports_cooperative_launch = cooperative != 0;
    device.smpbo = prop.sharedMemPerBlockOptin;
    device.cc = (100 * prop.major) + (10 * prop.minor);
  }
  if (recorded_error) {
    // A partly read table is no table: every Create refuses, and the first
    // reports the recorded failure.
    info.physical_device_count = 0;
    info.device_count = 0;
  }
  if (pending) {
    // Report both: the earlier error, and what voided the table.
    recorded_error = recorded_error
                         ? *pending + "; then reading the device table: " + *recorded_error
                         : *pending;
  }
  return info;
}

}  // namespace

void ggml_cuda_error(const char* stmt, const char* func, const char* file, int line,
                     const char* msg) {
  if (!recorded_error) {
    recorded_error = std::string(msg) + " (" + stmt + " in " + func + " at " + file + ":" +
                     std::to_string(line) + ")";
  }
}

void ggml_cuda_set_device(int device) {
  int current = -1;
  CUDA_CHECK(cudaGetDevice(&current));
  if (device == current) {
    return;
  }
  CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
  int id = 0;
  CUDA_CHECK(cudaGetDevice(&id));
  return id;
}

// Read once per process, as upstream does: a failure here (reported to the
// first LaunchContext::Create) leaves no device until the process restarts.
const ggml_cuda_device_info& ggml_cuda_info() {
  static const ggml_cuda_device_info info = DeviceInfo();
  return info;
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
  // Whatever GGML created itself would be a hidden allocation (D-053).
  jitllm::base::Check(copy_event == nullptr, "GGML created no copy event");
  for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
    for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
      jitllm::base::Check(streams[i][j] == nullptr && cublas_handles[i][j] == nullptr &&
                              cublas_workspaces[i][j] == nullptr && pools[i][j] == nullptr,
                          "the launch context took back what it lent GGML");
    }
  }
}

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) {
  jitllm::base::Fatal("a GGML launcher asked for a pool the launch context did not lend");
}

namespace jitllm::kernels::ggml::internal {

std::optional<std::string> TakeCudaError() { return std::exchange(recorded_error, std::nullopt); }

bool CudaErrorPending() { return recorded_error.has_value(); }

cublasContext* CublasHandleOf(ggml_backend_cuda_context& context) {
  if (context.cublas_handles[context.device][context.curr_stream_no] == nullptr) {
    return nullptr;
  }
  return context.cublas_handle();
}

bool HoldsCublasWorkspace(const ggml_backend_cuda_context& context) {
  for (const auto& per_device : context.cublas_workspaces) {
    for (const void* workspace : per_device) {
      if (workspace != nullptr) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace jitllm::kernels::ggml::internal
