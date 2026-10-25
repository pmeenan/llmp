// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// The quantizing cache stores (kv_quant.h): set_rows into Q8_0 or Q4_0, one
// store or a group in one launch. The index decomposition follows GGML's
// k_set_rows_quant (set-rows.cu at the pin), a thread per block; the
// quantization is kv_quant.cuh's. This unit builds without fast math
// (CMakeLists.txt), though kv_quant.cuh's arithmetic would not change with it.

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/kv_quant.cuh"
#include "kernels/ggml/kv_quant.h"
#include "kernels/ggml/launch.h"

namespace llmp::kernels::ggml {
namespace {

struct Store {
  const float* values;
  const void* ids;
  char* destination;
  QuantStoreMemberRecipe recipe;
  uint3 blocks;  // blocks per row
  uint3 rows, planes, ids_planes, ids_batches;
  bool ids64;
};
struct Stores {
  Store entries[kSetRowsQuantGroupMax];
};
// CUDA launch arguments own the descriptors; no pointer refers to host
// stack memory after launch or capture. Below the portable 4 KiB limit.
static_assert(sizeof(Stores) <= 4096);
__global__ void QuantizedStores(const __grid_constant__ Stores stores) {
  const Store& s = stores.entries[blockIdx.y];
  const std::uint64_t flat = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (flat >= s.recipe.total_blocks) {
    return;
  }
  auto split = fast_div_modulo(static_cast<std::uint32_t>(flat), s.blocks);
  const auto block = split.y;
  split = fast_div_modulo(split.x, s.rows);
  const auto row = split.y;
  split = fast_div_modulo(split.x, s.planes);
  const auto plane = split.y;
  const auto batch = split.x;
  const auto ids_plane = fastmodulo(plane, s.ids_planes);
  const auto ids_batch = fastmodulo(batch, s.ids_batches);
  const std::uint64_t at = row * s.recipe.ids_stride[0] + ids_plane * s.recipe.ids_stride[1] +
                           ids_batch * s.recipe.ids_stride[2];
  const std::int64_t destination_row = s.ids64 ? static_cast<const std::int64_t*>(s.ids)[at]
                                               : static_cast<const std::int32_t*>(s.ids)[at];
  const float* values = s.values + row * s.recipe.values_stride[0] +
                        plane * s.recipe.values_stride[1] + batch * s.recipe.values_stride[2] +
                        std::uint64_t{block} * kv::kBlockValues;
  float x[kv::kBlockValues];
#pragma unroll
  for (int j = 0; j < kv::kBlockValues; ++j) {
    x[j] = values[j];
  }
  const std::uint64_t block_bytes =
      s.recipe.type == GGML_TYPE_Q8_0 ? sizeof(block_q8_0) : sizeof(block_q4_0);
  char* destination = s.destination + destination_row * s.recipe.destination_stride[0] +
                      plane * s.recipe.destination_stride[1] +
                      batch * s.recipe.destination_stride[2] + block * block_bytes;
  kv::StoreBlock(s.recipe.type, x, destination);
}

std::expected<void, KernelFailure> Launch(LaunchContext& launch,
                                          std::span<const ggml_tensor* const> nodes) {
  auto described = DescribeSetRowsQuantLaunch(nodes);
  if (!described) return std::unexpected(described.error());
  Stores stores{};
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const ggml_tensor* node = nodes[i];
    const ggml_tensor* values = node->src[0];
    const ggml_tensor* ids = node->src[1];
    auto& store = stores.entries[i];
    store.values = static_cast<const float*>(values->data);
    store.ids = ids->data;
    store.destination = static_cast<char*>(node->data);
    store.recipe = described->members[i];
    store.blocks = init_fastdiv_values(store.recipe.blocks_per_row);
    store.rows = init_fastdiv_values(store.recipe.rows);
    store.planes = init_fastdiv_values(store.recipe.planes);
    store.ids_planes = init_fastdiv_values(store.recipe.ids_planes);
    store.ids_batches = init_fastdiv_values(store.recipe.ids_batches);
    store.ids64 = store.recipe.index_bytes == sizeof(std::int64_t);
  }
  const auto groups = described->grid_y;
  const auto blocks = described->grid_x;
  const auto threads = described->block_x;
  const auto shared = described->dynamic_shared_bytes;
  return launch.Run(base::Bytes(described->pool_bytes), [stores, groups, blocks, threads, shared](
                                                            ggml_backend_cuda_context& context) {
    QuantizedStores<<<dim3(blocks, groups, 1), threads, shared, context.stream()>>>(stores);
  });
}

}  // namespace

std::expected<void, KernelFailure> SetRowsQuant(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckSetRowsQuant(node); !checked) {
    return checked;
  }
  const std::array<const ggml_tensor*, 1> nodes = {node};
  return Launch(launch, nodes);
}

std::expected<void, KernelFailure> SetRowsQuantGroup(LaunchContext& launch,
                                                     std::span<ggml_tensor* const> nodes) {
  std::array<const ggml_tensor*, kSetRowsQuantGroupMax> checked{};
  if (nodes.size() > checked.size()) {
    return CheckSetRowsQuantGroup({});
  }
  std::copy(nodes.begin(), nodes.end(), checked.begin());
  const std::span<const ggml_tensor* const> group(checked.data(), nodes.size());
  if (auto result = CheckSetRowsQuantGroup(group); !result) {
    return result;
  }
  return Launch(launch, group);
}

}  // namespace llmp::kernels::ggml
