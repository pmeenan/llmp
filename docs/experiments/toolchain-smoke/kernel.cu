// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cstdio>
__host__ __device__ constexpr int offset() {
#ifdef REQUIRE_CUDA_CXX23
  static_assert(__cplusplus >= 202302L);
  if consteval { return 5; } else { return 7; }
#else
  return 7;
#endif
}
#ifdef REQUIRE_CUDA_CXX23
static_assert(offset() == 5);
#endif
__global__ void transform(int* data, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) data[i] = 3 * i + offset();
}
extern "C" int cuda_smoke() {
  constexpr int count = 257;
  int host[count]{};
  int* device = nullptr;
  cudaDeviceProp prop{};
  cudaError_t status = cudaGetDeviceProperties(&prop, 0);
  if (status != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); return 1; }
  if (prop.major != 12 || prop.minor != 1) { std::fprintf(stderr, "Expected GB10 capability 12.1\n"); return 1; }
  status = cudaMalloc(&device, sizeof(host));
  if (status != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); return 1; }
  transform<<<2, 256>>>(device, count);
  status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status == cudaSuccess) status = cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost);
  cudaError_t released = cudaFree(device);
  if (status != cudaSuccess || released != cudaSuccess) {
    std::fprintf(stderr, "CUDA failure: %s / free: %s\n", cudaGetErrorString(status), cudaGetErrorString(released)); return 1;
  }
  for (int i = 0; i < count; ++i) if (host[i] != 3 * i + 7) return 1;
  std::printf("CUDA PASS: %s, capability=%d.%d, checked=%d\n", prop.name, prop.major, prop.minor, count);
  return 0;
}
