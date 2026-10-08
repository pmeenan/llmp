// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include <algorithm>
#include <array>
#include <cstdint>

#include "base/bytes.h"
#include "common.cuh"
#include "convert.cuh"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/set_rows_group.h"

namespace llmp::kernels::ggml {
namespace {
// Index decomposition and conversion follow locked GGML set-rows.cu
// k_set_rows<float, int64_t, half>; only independent store dispatch is grouped.
struct Store {
  const float* values;
  const std::int64_t* ids;
  half* destination;
  std::uint64_t values_stride[3], ids_stride[3], destination_stride[3];
  uint3 width, rows, planes, ids_planes, ids_batches;
  std::uint32_t total;
};
struct Stores {
  Store entries[kSetRowsGroupMax];
};
// CUDA launch arguments own the descriptors; no pointer refers to host stack
// memory after launch or capture. Keep below the portable 4 KiB argument limit.
static_assert(sizeof(Stores) <= 4096);
constexpr unsigned kThreads = 256;
__global__ void GroupedStores(const __grid_constant__ Stores stores) {
  const Store& s = stores.entries[blockIdx.y];
  const std::uint64_t flat = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (flat >= s.total) return;
  auto split = fast_div_modulo(static_cast<std::uint32_t>(flat), s.width);
  const auto column = split.y;
  split = fast_div_modulo(split.x, s.rows);
  const auto row = split.y;
  split = fast_div_modulo(split.x, s.planes);
  const auto plane = split.y;
  const auto batch = split.x;
  const auto ids_plane = fastmodulo(plane, s.ids_planes);
  const auto ids_batch = fastmodulo(batch, s.ids_batches);
  const auto destination_row =
      s.ids[row * s.ids_stride[0] + ids_plane * s.ids_stride[1] + ids_batch * s.ids_stride[2]];
  const auto* values =
      s.values + row * s.values_stride[0] + plane * s.values_stride[1] + batch * s.values_stride[2];
  auto* destination = s.destination + destination_row * s.destination_stride[0] +
                      plane * s.destination_stride[1] + batch * s.destination_stride[2];
  // The primitive's exact conversion, including rounding/NaN/signed zero.
  destination[column] = ggml_cuda_cast<half>(values[column]);
}
}  // namespace
std::expected<void, KernelFailure> RunSetRowsGroup(LaunchContext& launch,
                                                   std::span<ggml_tensor* const> nodes) {
  std::array<const ggml_tensor*, kSetRowsGroupMax> checked{};
  if (nodes.size() > checked.size()) return CheckSetRowsGroup({});
  std::copy(nodes.begin(), nodes.end(), checked.begin());
  if (auto r = CheckSetRowsGroup({checked.data(), nodes.size()}); !r) return r;
  Stores stores{};
  std::uint32_t maximum = 0;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto* node = nodes[i];
    const auto* values = node->src[0];
    const auto* ids = node->src[1];
    Store& s = stores.entries[i];
    s.values = static_cast<const float*>(values->data);
    s.ids = static_cast<const std::int64_t*>(ids->data);
    s.destination = static_cast<half*>(node->data);
    for (unsigned d = 0; d < 3; ++d) {
      s.values_stride[d] = values->nb[d + 1] / sizeof(float);
      s.ids_stride[d] = ids->nb[d] / sizeof(std::int64_t);
      s.destination_stride[d] = node->nb[d + 1] / sizeof(half);
    }
    s.width = init_fastdiv_values(static_cast<std::uint32_t>(values->ne[0]));
    s.rows = init_fastdiv_values(static_cast<std::uint32_t>(values->ne[1]));
    s.planes = init_fastdiv_values(static_cast<std::uint32_t>(values->ne[2]));
    s.ids_planes = init_fastdiv_values(static_cast<std::uint32_t>(ids->ne[1]));
    s.ids_batches = init_fastdiv_values(static_cast<std::uint32_t>(ids->ne[2]));
    s.total = static_cast<std::uint32_t>(ggml_nelements(values));
    maximum = std::max(maximum, s.total);
  }
  const unsigned groups = static_cast<unsigned>(nodes.size());
  const unsigned blocks = static_cast<unsigned>((std::uint64_t{maximum} + kThreads - 1) / kThreads);
  return launch.Run(base::Bytes(0), [stores, groups, blocks](ggml_backend_cuda_context& context) {
    GroupedStores<<<dim3(blocks, groups), kThreads, 0, context.stream()>>>(stores);
  });
}
}  // namespace llmp::kernels::ggml
