// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// Read-only capability probe for the interconnect experiment.
#include <cuda.h>
#include <cstdio>
int main() {
  if (cuInit(0) != CUDA_SUCCESS) return 1;
  CUdevice dev;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 1;
  const struct { const char* name; CUdevice_attribute attr; } attrs[] = {
    {"gpu_direct_rdma", CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED},
    {"dma_buf", CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED},
    {"pageable_memory_access", CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS},
    {"host_native_atomics", CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED},
    {"can_map_host_memory", CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY},
  };
  for (const auto& a : attrs) {
    int value = -1;
    const auto rc = cuDeviceGetAttribute(&value, a.attr, dev);
    std::printf("%s value=%d result=%d\n", a.name, value, static_cast<int>(rc));
    if (rc != CUDA_SUCCESS) return 2;
  }
}
