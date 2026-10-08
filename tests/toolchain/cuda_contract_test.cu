// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The CUDA toolchain contract (D-032, D-060, D-066): NVCC with the SDK's
// Clang as host compiler, C++23 in host and device code, no exceptions in
// the host pass, SASS only (sm_121, and the build's discrete targets such as
// sm_86, D-082) and a static CUDA runtime. Needs a GPU the build has SASS
// for: a GB10 (label `gpu`), or a discrete target (`gpu-discrete` too).

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <string>
#include <string_view>

#ifndef LLMP_CUDA_SASS
#error "the build defines LLMP_CUDA_SASS (tests/toolchain/CMakeLists.txt)"
#endif

static_assert(__cplusplus >= 202302L, "C++23 in CUDA code too (D-032)");
#if !defined(__CUDA_ARCH__) && (defined(__cpp_exceptions) || defined(__EXCEPTIONS))
#error "the host pass builds with -fno-exceptions (D-066)"
#endif

namespace {

constexpr int kCount = 257;
// The compute capabilities this build has SASS for, such as "/121/86/".
constexpr std::string_view kSass = LLMP_CUDA_SASS;

// `if consteval` needs C++23 in both passes.
__host__ __device__ constexpr int Offset() {
  if consteval {
    return 5;
  } else {
    return 7;
  }
}
static_assert(Offset() == 5);

__global__ void Transform(int* data, int count) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) {
    data[i] = 3 * i + Offset();
  }
}

bool Ok(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(status));
    return false;
  }
  return true;
}

}  // namespace

int main() {
  cudaDeviceProp prop{};
  if (!Ok(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties")) {
    return 1;
  }
  const std::string sass = "/" + std::to_string((prop.major * 10) + prop.minor) + "/";
  const bool targeted = prop.minor >= 0 && prop.minor <= 9 && kSass.contains(sass);
  if (!targeted) {
    std::fprintf(stderr, "FAIL: needs a GPU this build has SASS for, found %s %d.%d\n", prop.name,
                 prop.major, prop.minor);
    return 1;
  }
  std::array<int, kCount> host{};
  int* device = nullptr;
  if (!Ok(cudaMalloc(&device, sizeof(host)), "cudaMalloc")) {
    return 1;
  }
  Transform<<<2, 256>>>(device, kCount);
  bool ok = Ok(cudaGetLastError(), "launch") && Ok(cudaDeviceSynchronize(), "synchronize") &&
            Ok(cudaMemcpy(host.data(), device, sizeof(host), cudaMemcpyDeviceToHost), "copy");
  ok = Ok(cudaFree(device), "cudaFree") && ok;
  if (!ok) {
    return 1;
  }
  for (int i = 0; i < kCount; ++i) {
    if (host[static_cast<std::size_t>(i)] != 3 * i + 7) {
      std::fprintf(stderr, "FAIL: value %d is %d\n", i, host[static_cast<std::size_t>(i)]);
      return 1;
    }
  }
  std::printf("cuda contract: %s %d.%d, %d values, CUDA runtime %d\n", prop.name, prop.major,
              prop.minor, kCount, CUDART_VERSION);
  return 0;
}
