// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Native preparation for the locked MIT ds4 token-tile HCA core. A Run
// owns the chronological F16 mirror and original record/count arrays in
// one planned workspace scope. There is no new cache representation.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>

#include "base/bytes.h"
#include "common.cuh"
#include "ds4_attn_tokentile.cuh"
#include "kernels/ggml/dsv4_ds4_attention.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {
namespace {

constexpr std::uint64_t Round(std::uint64_t bytes) { return (bytes + 255) & ~std::uint64_t{255}; }

struct Layout {
  std::uint32_t first = 0;
  std::uint32_t tokens = 0;
  std::uint32_t raw_cells = 0;
  std::uint32_t compressed = 0;
  std::uint32_t tiles = 0;
  std::uint64_t counts = 0;
  std::uint64_t raw = 0;
  std::uint64_t bytes = 0;
};

std::expected<Layout, KernelFailure> LayoutOf(const ggml_tensor* node) {
  if (auto checked = CheckDsv4HcaTokentile(node); !checked) return std::unexpected(checked.error());
  Layout out;
  std::memcpy(&out.first, &node->op_params[kDsv4HcaFirstParam], sizeof(out.first));
  out.tokens = static_cast<std::uint32_t>(node->src[0]->ne[1]);
  out.raw_cells = static_cast<std::uint32_t>(LlmpOpInt(node->src[3], 0));
  out.compressed = static_cast<std::uint32_t>(node->src[1]->ne[1]) - out.raw_cells;
  out.tiles = (out.tokens + 3) / 4;
  out.counts = Round(static_cast<std::uint64_t>(out.tiles) * out.compressed * 8);
  out.raw = Round(out.counts + static_cast<std::uint64_t>(out.tiles) * 4);
  out.bytes = Round(out.raw + (static_cast<std::uint64_t>(out.tokens) + 127) * 1024);
  return out;
}

// Stored F16 bytes, including their signs, are copied without conversion.
// Negative prefix positions are absent; the original core's raw_row_min
// excludes the zero mirror rows from every softmax.
__global__ void RawMirror(std::uint16_t* dst, const std::uint16_t* kv, std::uint32_t first,
                          std::uint32_t cells) {
  const std::uint32_t row = blockIdx.x;
  const std::uint32_t channel = threadIdx.x * 2;
  const std::int64_t position = static_cast<std::int64_t>(first) - 127 + row;
  const std::uint64_t to = static_cast<std::uint64_t>(row) * 512 + channel;
  if (position < 0) {
    dst[to] = 0;
    dst[to + 1] = 0;
  } else {
    const auto slot = static_cast<std::uint64_t>(position) % cells;
    const std::uint64_t from = slot * 512 + channel;
    dst[to] = kv[from];
    dst[to + 1] = kv[from + 1];
  }
}

bool Supported(const LaunchContext& launch) {
  const auto& device = ggml_cuda_info().devices[launch.device()];
  return device.cc == 1210 && device.smpbo >= 88576;
}

bool WorkspaceOverlaps(const LaunchContext& launch, const ggml_tensor* node, std::uint64_t bytes) {
  const auto base = launch.workspace().base;
  const ggml_tensor* const operands[] = {node, node->src[0], node->src[1], node->src[3],
                                         node->src[4]};
  for (const auto* tensor : operands) {
    const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
    const auto extent = detail::Extent(tensor);
    if (!extent || (base >= address ? base - address < *extent : address - base < bytes))
      return true;
  }
  return false;
}

void Recorded(int result) {
  if (result != 0)
    ggml_cuda_error(cudaGetErrorString(static_cast<cudaError_t>(result)), __func__, __FILE__,
                    __LINE__, "ds4 HCA launcher failed after native preparation");
}

}  // namespace

bool Dsv4HcaTokentileFits(const LaunchContext& launch, const ggml_tensor* node) {
  if (!CheckDsv4HcaTokentile(node) || !Supported(launch)) return false;
  // Any prefill chunk of the ring's (raw cells: its most rows and 256; the
  // mirror copies the chunk's rows and the 127 before them, wherever they
  // sit in the ring), at the compressed widths measured: 256 cells, and
  // 1,024 for chunks of up to 2,048 rows.
  const auto tokens = node->src[0]->ne[1];
  const auto raw = LlmpOpInt(node->src[3], 0);
  const auto compressed = node->src[1]->ne[1] - raw;
  return tokens >= kDsv4HcaMinRows && tokens <= kDsv4HcaMaxRows && raw >= tokens + 256 &&
         (compressed == 256 || (compressed == 1024 && tokens <= 2048));
}

std::expected<std::uint64_t, KernelFailure> PlanDsv4HcaTokentile(const LaunchContext& launch,
                                                                 const ggml_tensor* node) {
  auto layout = LayoutOf(node);
  if (!layout) return std::unexpected(layout.error());
  if (!Supported(launch)) return detail::Rejected("ds4 HCA is measured only on GB10");
  // Attribute setup is deliberately outside Run and graph capture.
  const int result =
      node->src[0]->type == GGML_TYPE_F16 ? Ds4HcaCoreQ16Prepare() : llmp_ds4_hca_prepare();
  if (result != 0) {
    (void)cudaGetLastError();
    return detail::Rejected("the ds4 HCA shared-memory opt-in failed");
  }
  return layout->bytes;
}

std::expected<void, KernelFailure> Dsv4HcaTokentile(LaunchContext& launch, ggml_tensor* node) {
  auto layout = LayoutOf(node);
  if (!layout) return std::unexpected(layout.error());
  if (!Supported(launch)) return detail::Rejected("ds4 HCA is measured only on GB10");
  // The chunk's first position is read here, on the host, and becomes a
  // launch parameter, which a replay would keep (D-090).
  if (launch.capturing())
    return detail::Rejected("ds4 HCA's first position is not graph-replayable");
  if (WorkspaceOverlaps(launch, node, layout->bytes))
    return detail::Rejected("ds4 HCA scratch overlaps an operand");
  return launch.Run(base::Bytes(layout->bytes), [node, layout = *layout](auto& context) {
    ggml_cuda_pool_alloc<std::byte> scratch(context.pool(), static_cast<std::size_t>(layout.bytes));
    auto* bytes = scratch.get();
    auto* records = bytes;
    auto* counts = bytes + layout.counts;
    auto* raw = reinterpret_cast<std::uint16_t*>(bytes + layout.raw);
    const auto* kv = static_cast<const std::uint16_t*>(node->src[1]->data);
    const auto stream = context.stream();
    RawMirror<<<layout.tokens + 127, 256, 0, stream>>>(raw, kv, layout.first, layout.raw_cells);
    CUDA_CHECK(cudaGetLastError());
    if (internal::CudaErrorPending()) return;
    Recorded(llmp_ds4_hca_records_launch(records, counts, layout.first, layout.tokens,
                                         layout.compressed, layout.compressed, stream));
    if (internal::CudaErrorPending()) return;
    const auto* compressed = kv + static_cast<std::uint64_t>(layout.raw_cells) * 512;
    if (node->src[0]->type == GGML_TYPE_F16) {
      Recorded(Ds4HcaCoreQ16Launch(
          static_cast<float*>(node->data), static_cast<const float*>(node->src[4]->data),
          node->src[0]->data, raw, compressed, records, counts, layout.compressed, layout.tokens,
          64, layout.first < 127 ? 127 - layout.first : 0, stream));
    } else {
      Recorded(llmp_ds4_hca_core_launch(
          static_cast<float*>(node->data), static_cast<const float*>(node->src[4]->data),
          static_cast<const float*>(node->src[0]->data), raw, compressed, records, counts,
          layout.compressed, layout.tokens, 64, layout.first < 127 ? 127 - layout.first : 0,
          stream));
    }
    // The graph pins this same cataloged workspace. Foreign launchers
    // consume cudaGetLastError, so their result is recorded before returning.
  });
}

}  // namespace llmp::kernels::ggml
