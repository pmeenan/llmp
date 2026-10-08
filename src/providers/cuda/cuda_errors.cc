// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "providers/cuda/cuda_errors.h"

#include <cuda.h>

#include <expected>
#include <format>

#include "providers/device_memory.h"

namespace llmp::providers::cuda {
namespace {

// Errors the driver reports before doing anything: the call changed
// nothing, and its outcome is known. Every other error, including those the
// driver documents as requiring a process restart (device asserts, launch
// timeouts, contained or uncorrectable faults), leaves the outcome unknown.
bool Refused(CUresult result) {
  switch (result) {
    case CUDA_ERROR_INVALID_VALUE:
    case CUDA_ERROR_INVALID_HANDLE:
    case CUDA_ERROR_INVALID_DEVICE:
    case CUDA_ERROR_INVALID_CONTEXT:
    case CUDA_ERROR_NOT_INITIALIZED:
    case CUDA_ERROR_DEINITIALIZED:
    case CUDA_ERROR_NO_DEVICE:
    case CUDA_ERROR_NOT_PERMITTED:
    case CUDA_ERROR_NOT_FOUND:
    case CUDA_ERROR_NOT_MAPPED:
    case CUDA_ERROR_ALREADY_MAPPED:
      return true;
    default:
      return false;
  }
}

}  // namespace

std::unexpected<Failure> Error(CUresult result, const char* call) {
  const char* name = nullptr;
  if (cuGetErrorName(result, &name) != CUDA_SUCCESS || name == nullptr) {
    name = "unrecognized error";
  }
  ProviderError error = ProviderError::kUnknown;
  if (result == CUDA_ERROR_OUT_OF_MEMORY) {
    error = ProviderError::kOutOfMemory;
  } else if (result == CUDA_ERROR_NOT_SUPPORTED) {
    error = ProviderError::kUnsupported;
  } else if (Refused(result)) {
    error = ProviderError::kFailed;
  }
  return std::unexpected(Failure{
      .error = error, .detail = std::format("{}: {} ({})", call, name, static_cast<int>(result))});
}

}  // namespace llmp::providers::cuda
