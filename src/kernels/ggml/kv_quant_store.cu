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
  // Strides of dimensions 1 to 3: the values' in floats, the ids' (of
  // dimensions 0 to 2) in elements, the destination's in bytes.
  std::uint64_t values_stride[3];
  std::uint64_t ids_stride[3];
  std::uint64_t destination_stride[3];
  uint3 blocks;  // blocks per row
  uint3 rows, planes, ids_planes, ids_batches;
  std::uint32_t total;  // blocks
  ggml_type type;
  bool ids64;
};
struct Stores {
  Store entries[kSetRowsQuantGroupMax];
};
// CUDA launch arguments own the descriptors; no pointer refers to host
// stack memory after launch or capture. Below the portable 4 KiB limit.
static_assert(sizeof(Stores) <= 4096);
constexpr unsigned kThreads = 256;

__global__ void QuantizedStores(const __grid_constant__ Stores stores) {
  const Store& s = stores.entries[blockIdx.y];
  const std::uint64_t flat = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (flat >= s.total) {
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
  const std::uint64_t at =
      row * s.ids_stride[0] + ids_plane * s.ids_stride[1] + ids_batch * s.ids_stride[2];
  const std::int64_t destination_row = s.ids64 ? static_cast<const std::int64_t*>(s.ids)[at]
                                               : static_cast<const std::int32_t*>(s.ids)[at];
  const float* values = s.values + row * s.values_stride[0] + plane * s.values_stride[1] +
                        batch * s.values_stride[2] + std::uint64_t{block} * kv::kBlockValues;
  float x[kv::kBlockValues];
#pragma unroll
  for (int j = 0; j < kv::kBlockValues; ++j) {
    x[j] = values[j];
  }
  const std::uint64_t block_bytes =
      s.type == GGML_TYPE_Q8_0 ? sizeof(block_q8_0) : sizeof(block_q4_0);
  char* destination = s.destination + destination_row * s.destination_stride[0] +
                      plane * s.destination_stride[1] + batch * s.destination_stride[2] +
                      block * block_bytes;
  kv::StoreBlock(s.type, x, destination);
}

Store Describe(const ggml_tensor* node) {
  const ggml_tensor* values = node->src[0];
  const ggml_tensor* ids = node->src[1];
  Store s{};
  s.values = static_cast<const float*>(values->data);
  s.ids = ids->data;
  s.destination = static_cast<char*>(node->data);
  const std::uint64_t id_size = ggml_type_size(ids->type);
  for (unsigned d = 0; d < 3; ++d) {
    s.values_stride[d] = values->nb[d + 1] / sizeof(float);
    s.ids_stride[d] = ids->nb[d] / id_size;
    s.destination_stride[d] = node->nb[d + 1];
  }
  s.blocks = init_fastdiv_values(static_cast<std::uint32_t>(values->ne[0] / kv::kBlockValues));
  s.rows = init_fastdiv_values(static_cast<std::uint32_t>(values->ne[1]));
  s.planes = init_fastdiv_values(static_cast<std::uint32_t>(values->ne[2]));
  s.ids_planes = init_fastdiv_values(static_cast<std::uint32_t>(ids->ne[1]));
  s.ids_batches = init_fastdiv_values(static_cast<std::uint32_t>(ids->ne[2]));
  s.total = static_cast<std::uint32_t>(ggml_nelements(values) / kv::kBlockValues);
  s.type = node->type;
  s.ids64 = ids->type == GGML_TYPE_I64;
  return s;
}

std::expected<void, KernelFailure> Launch(LaunchContext& launch,
                                          std::span<const ggml_tensor* const> nodes) {
  Stores stores{};
  std::uint32_t maximum = 0;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    stores.entries[i] = Describe(nodes[i]);
    maximum = std::max(maximum, stores.entries[i].total);
  }
  const auto groups = static_cast<unsigned>(nodes.size());
  const auto blocks = static_cast<unsigned>((std::uint64_t{maximum} + kThreads - 1) / kThreads);
  return launch.Run(base::Bytes(0), [stores, groups, blocks](ggml_backend_cuda_context& context) {
    QuantizedStores<<<dim3(blocks, groups), kThreads, 0, context.stream()>>>(stores);
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
