// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// NVCC probe: GCC 16.2 library headers in host code, static cudart.
#include <expected>
#include <flat_map>
#include <format>
#include <mdspan>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <vector>

__global__ void twice(int* values, int n) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) {
    values[i] *= 2;
  }
}

namespace {

std::expected<int, std::string> sum(std::span<const int> values) {
  int total = 0;
  for (const int v : values | std::views::take(4)) {
    total += v;
  }
  return total;
}

}  // namespace

int main() {
  std::vector<int> host{1, 2, 3, 4};
  int* device = nullptr;
  if (cudaMalloc(&device, host.size() * sizeof(int)) != cudaSuccess) {
    std::println(stderr, "FAIL: cudaMalloc");
    return 2;
  }
  cudaMemcpy(device, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice);
  twice<<<1, 4>>>(device, 4);
  cudaMemcpy(host.data(), device, host.size() * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(device);

  const std::mdspan view(host.data(), 2, 2);
  std::flat_map<std::string, int> names{{"sum", sum(host).value()}};
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::println("{} sum={} corner={} glibcxx={}", prop.name, names["sum"], view[1, 1],
               __GLIBCXX__);
  return names["sum"] == 20 && view[1, 1] == 8 ? 0 : 1;
}
