// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <cstdint>
#include <expected>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/dsv4_weighted_reduce.h"
#include "kernels/ggml/launch.h"

namespace jitllm::kernels::ggml {
namespace {

constexpr std::uint32_t kThreads = 256;

// Each thread owns one column of one token. Intended to preserve the
// separate MUL results before their ascending ADD chain: do not start
// from +0, contract a product/add, reassociate, skip a slot or guard NaNs.
__global__ void WeightedReduceKernel(const float* __restrict__ down,
                                     const float* __restrict__ weights,
                                     float* __restrict__ values) {
  const std::uint64_t token = blockIdx.y;
  const std::uint64_t column = (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (column >= static_cast<std::uint64_t>(kDsv4WeightedReduceWidth)) {
    return;
  }
  const std::uint64_t first = token * kDsv4WeightedReduceSlots;
  float sum = __fmul_rn(down[(first * kDsv4WeightedReduceWidth) + column], weights[first]);
#pragma unroll
  for (int slot = 1; slot < kDsv4WeightedReduceSlots; ++slot) {
    const std::uint64_t row = first + static_cast<std::uint64_t>(slot);
    const float product = __fmul_rn(down[(row * kDsv4WeightedReduceWidth) + column], weights[row]);
    sum = __fadd_rn(sum, product);
  }
  values[(token * kDsv4WeightedReduceWidth) + column] = sum;
}

}  // namespace

std::expected<void, KernelFailure> RunDsv4WeightedReduce(LaunchContext& launch,
                                                         const Dsv4WeightedReduce& desc) {
  if (auto checked = CheckDsv4WeightedReduce(desc); !checked) {
    return checked;
  }
  // LaunchContext invokes the callable synchronously. Neither the kernel
  // nor a captured graph retains these tensor descriptors, only addresses.
  const auto* down = static_cast<const float*>(desc.down.tensor->data);
  const auto* weights = static_cast<const float*>(desc.weights.tensor->data);
  auto* values = static_cast<float*>(desc.values.tensor->data);
  const auto rows = static_cast<std::uint32_t>(desc.down.tensor->ne[2]);
  return launch.Run(base::Bytes(0), [down, weights, values, rows](auto& context) {
    const dim3 blocks(static_cast<std::uint32_t>(kDsv4WeightedReduceWidth) / kThreads, rows, 1);
    WeightedReduceKernel<<<blocks, kThreads, 0, context.stream()>>>(down, weights, values);
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunDsv4OrderedReduce(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckDsv4OrderedReduce(node); !checked) {
    return checked;
  }
  return RunDsv4WeightedReduce(launch,
                               {.down = {node->src[0], base::Bytes(ggml_nbytes(node->src[0]))},
                                .weights = {node->src[1], base::Bytes(ggml_nbytes(node->src[1]))},
                                .values = {node, base::Bytes(ggml_nbytes(node))}});
}

}  // namespace jitllm::kernels::ggml
