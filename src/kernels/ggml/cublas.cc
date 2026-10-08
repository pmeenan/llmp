// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/cublas.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "base/bytes.h"
#include "base/check.h"

namespace llmp::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

// GGML_CUDA_CC_HOPPER (common.cuh).
constexpr int kHopper = 900;

// The cuBLAS this build was compiled against: the SDK's pinned 13.8.0.4
// (D-076). The loaded libraries report major.minor.patch (not the build
// number) as major * 10000 + minor * 100 + patch.
constexpr std::size_t kPinned = CUBLAS_VERSION;

// cuBLAS's own environment switches, apart from logging and profiling
// ranges, change which kernels run and how they round (NVIDIA_TF32_OVERRIDE,
// the emulation and heuristics switches, CUBLAS_WORKSPACE_CONFIG), so none
// may be set. Logging, which records executed plans (P0), and NVTX ranges
// change neither.
bool NumericsSwitch(std::string_view name) {
  for (const std::string_view logging :
       {"CUBLAS_LOGINFO_DBG", "CUBLAS_LOGDEST_DBG", "CUBLASLT_LOG_LEVEL", "CUBLASLT_LOG_FILE",
        "CUBLASLT_LOG_MASK", "CUBLAS_NVTX_LEVEL", "CUBLASLT_NVTX_LEVEL"}) {
    if (name == logging) {
      return false;
    }
  }
  return name.starts_with("CUBLAS") || name == "NVIDIA_TF32_OVERRIDE";
}

// The loaded cuBLAS is the pinned one, running as built: a system cuBLAS
// found first (LD_LIBRARY_PATH precedes the tests' RUNPATH) or an
// environment switch would change the executed plan.
std::expected<void, KernelFailure> CheckLibrary() {
  int major = 0;
  int minor = 0;
  int patch = 0;
  if (cublasGetProperty(MAJOR_VERSION, &major) != CUBLAS_STATUS_SUCCESS ||
      cublasGetProperty(MINOR_VERSION, &minor) != CUBLAS_STATUS_SUCCESS ||
      cublasGetProperty(PATCH_LEVEL, &patch) != CUBLAS_STATUS_SUCCESS) {
    return Rejected("cuBLAS does not report its version");
  }
  const std::size_t loaded = (static_cast<std::size_t>(major) * 10000) +
                             (static_cast<std::size_t>(minor) * 100) +
                             static_cast<std::size_t>(patch);
  const std::size_t lt = cublasLtGetVersion();
  if (loaded != kPinned || lt != kPinned) {
    return Rejected(std::format("cuBLAS {} and cuBLASLt {} are loaded; this build needs {}", loaded,
                                lt, kPinned));
  }
  for (char** entry = environ; *entry != nullptr; ++entry) {
    const std::string_view variable(*entry);
    const std::string_view name = variable.substr(0, variable.find('='));
    if (NumericsSwitch(name)) {
      return Rejected(std::format("{} is set, which changes how cuBLAS computes", name));
    }
  }
  return {};
}

}  // namespace

base::Bytes CublasHandle::UpstreamWorkspace(int cc) {
  return base::Bytes(cc >= kHopper ? 32ULL << 20 : 4ULL << 20);
}

std::expected<std::unique_ptr<CublasHandle>, KernelFailure> CublasHandle::Create(
    int device, providers::DeviceExecution& execution, providers::StreamId stream,
    LaunchContext::Workspace workspace) {
  if (auto library = CheckLibrary(); !library) {
    return std::unexpected(library.error());
  }
  // cublasSetWorkspace refuses a workspace aligned to less than 256 bytes.
  if (workspace.size.value() > 0 && (workspace.base == 0 || workspace.base % 256 != 0 ||
                                     workspace.base + workspace.size.value() < workspace.base)) {
    return Rejected("the cuBLAS workspace is not a 256-byte aligned device range");
  }
  // Also makes the provider's context current, which cuBLAS binds to.
  auto native = execution.Submission(stream);
  if (!native || native->handle == nullptr) {
    return Rejected("the provider refused the stream" +
                    (native ? std::string() : ": " + native.error().detail));
  }
  int current = -1;
  if (cudaGetDevice(&current) != cudaSuccess || current != device) {
    return Rejected(std::format("device {} is not current on this thread", device));
  }
  // The provider's context, which cuBLAS binds the handle to.
  CUcontext owner = nullptr;
  if (cuCtxGetCurrent(&owner) != CUDA_SUCCESS || owner == nullptr) {
    return Rejected("no CUDA context is current after taking the stream");
  }
  cublasHandle_t handle = nullptr;
  if (const cublasStatus_t status = cublasCreate(&handle); status != CUBLAS_STATUS_SUCCESS) {
    return Rejected(std::string("cublasCreate failed: ") + cublasGetStatusString(status));
  }
  // Upstream's order: setting the stream resets the workspace to cuBLAS's
  // own, so the workspace comes last.
  cublasStatus_t status = cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasSetStream(handle, static_cast<cudaStream_t>(native->handle));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr): a device address.
    void* base = reinterpret_cast<void*>(workspace.base);
    status = cublasSetWorkspace(handle, base, workspace.size.value());
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    (void)cublasDestroy(handle);
    return Rejected(std::string("setting up the cuBLAS handle failed: ") +
                    cublasGetStatusString(status));
  }
  return std::unique_ptr<CublasHandle>(
      new CublasHandle(handle, owner, device, stream, *native, workspace));
}

CublasHandle::CublasHandle(cublasContext* handle, CUctx_st* owner, int device,
                           providers::StreamId stream, providers::NativeStream native_stream,
                           LaunchContext::Workspace workspace)
    : handle_(handle),
      owner_(owner),
      device_(device),
      stream_(stream),
      native_stream_(native_stream),
      workspace_(workspace) {}

CublasHandle::~CublasHandle() {
  base::Check(borrowers_ == 0, "no launch context still borrows a cuBLAS handle being destroyed");
  // cuBLAS destroys a handle in the context it was created in. If that
  // context cannot be made current, the handle is left, not destroyed
  // elsewhere. A failure leaves nothing to do either way: the handle's
  // resources are the library's, and aborting on a device fault would
  // break invariant 7.
  if (cuCtxPushCurrent(owner_) != CUDA_SUCCESS) {
    return;
  }
  (void)cublasDestroy(handle_);
  CUcontext popped = nullptr;
  (void)cuCtxPopCurrent(&popped);
}

}  // namespace llmp::kernels::ggml
