// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/dsv4_ds4_product_raw.h"
#include "kernels/ggml/dsv4_outa.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {
namespace {

constexpr std::uint64_t kBlocks = 8192ULL * (4096 / 32);
constexpr std::uint64_t kScales = kBlocks * sizeof(std::uint16_t);
constexpr std::uint64_t kCodes = kBlocks * 32;
constexpr std::uint64_t kTable = 4096ULL * 32 * 8;
constexpr std::uint64_t kScratch = kScales + kCodes + kTable;
constexpr std::uint64_t kShared = ((2ULL * 128) + (2ULL * 128)) * (32 + 8) * 2;

// Only a byte-layout conversion: raw Q8_0 scales and signed codes retain
// every bit. The staged F16/HMMA arithmetic is the qualified product core.
__global__ void PackOutA(const std::uint16_t* raw, std::uint16_t* scales, std::int8_t* codes) {
  const auto block = (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (block >= kBlocks) return;
  scales[block] = raw[block * 17];
  const auto* source = reinterpret_cast<const std::int8_t*>(raw + (block * 17) + 1);
  for (std::uint32_t j = 0; j < 32; ++j) codes[(block * 32) + j] = source[j];
}

// The same bytes, coalesced: each thread moves one of a block's seventeen
// 16-bit words (the F16 scale, then sixteen code pairs).
__global__ void PackOutAFast(const std::uint16_t* __restrict__ raw,
                             std::uint16_t* __restrict__ scales,
                             std::uint16_t* __restrict__ codes) {
  const auto word = (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (word >= kBlocks * 17) return;
  const std::uint64_t block = word / 17;
  const std::uint64_t k = word - (block * 17);
  const std::uint16_t value = raw[word];
  if (k == 0) {
    scales[block] = value;
  } else {
    codes[(block * 16) + k - 1] = value;
  }
}

bool WorkspaceOverlaps(const LaunchContext& launch, const ggml_tensor* node) {
  const auto base = launch.workspace().base;
  const std::array<const ggml_tensor*, 4> operands = {node, node->src[0], node->src[1],
                                                      node->src[2]};
  // NOLINTNEXTLINE(readability-use-anyofallof): explicit bounded overlap checks.
  for (const auto* tensor : operands) {
    const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
    const auto extent = detail::Extent(tensor);
    if (!extent || (base >= address ? base - address < *extent : address - base < kScratch)) {
      return true;
    }
  }
  return false;
}

}  // namespace

bool Dsv4OutASupported(const LaunchContext& launch) {
  const auto& info = ggml_cuda_info();
  if (launch.device() < 0 || launch.device() >= info.device_count) return false;
  const auto& d = info.devices[launch.device()];
  return d.cc == 1210 && d.warp_size == 32 && d.smpbo >= kShared;
}

std::expected<std::uint64_t, KernelFailure> PlanDsv4OutA(const LaunchContext& launch,
                                                         const ggml_tensor* node) {
  if (auto checked = CheckDsv4OutA(node); !checked) return std::unexpected(checked.error());
  if (!Dsv4OutASupported(launch))
    return detail::Rejected("DeepSeek staged output-A is measured only on GB10");
  return kScratch;
}

std::expected<void, KernelFailure> RunDsv4OutA(LaunchContext& launch, ggml_tensor* node,
                                               bool fast_pack) {
  auto planned = PlanDsv4OutA(launch, node);
  if (!planned) return std::unexpected(planned.error());
  if (WorkspaceOverlaps(launch, node))
    return detail::Rejected("DeepSeek output-A scratch overlaps an operand");
  const auto p = Dsv4OutAParamsOf(node);
  const ds4_product::Rope rope = {
      .positions = static_cast<const std::int32_t*>(node->src[2]->data),
      .original_context = static_cast<std::uint32_t>(p.original_context),
      .rotary = 64,
      .base = p.base,
      .scale = p.scale,
      .extension = p.extension,
      .attention = p.attention,
      .beta_fast = p.beta_fast,
      .beta_slow = p.beta_slow,
      .inverse = true};
  return launch.Run(base::Bytes(*planned), [node, rope, fast_pack](auto& context) {
    ggml_cuda_pool_alloc<std::byte> scratch(context.pool(), static_cast<std::size_t>(kScratch));
    auto* bytes = scratch.get();
    auto* scales = reinterpret_cast<std::uint16_t*>(bytes);
    auto* codes = reinterpret_cast<std::int8_t*>(bytes + kScales);
    auto* table = bytes + kScales + kCodes;
    if (fast_pack) {
      PackOutAFast<<<static_cast<unsigned>(((kBlocks * 17) + 255) / 256), 256, 0,
                     context.stream()>>>(static_cast<const std::uint16_t*>(node->src[0]->data),
                                         scales, reinterpret_cast<std::uint16_t*>(codes));
    } else {
      PackOutA<<<static_cast<unsigned>((kBlocks + 255) / 256), 256, 0, context.stream()>>>(
          static_cast<const std::uint16_t*>(node->src[0]->data), scales, codes);
    }
    CUDA_CHECK(cudaGetLastError());
    if (internal::CudaErrorPending()) return;
    // The borrowed core queues table zero/preparation and the full canonical
    // product on this provider-owned stream. No output-B or D4 emission. Its
    // table holds the rows rounded up to 128 (kTable: 4,096, the most), and
    // it stores whole 16-row tiles, which the output's rows hold
    // (Dsv4OutARows).
    const auto rows = static_cast<std::uint32_t>(node->src[1]->ne[2]);
    CUDA_CHECK(ds4_product::OutA(scales, codes, static_cast<const float*>(node->src[1]->data),
                                 static_cast<float*>(node->data), table, nullptr, rows, rope,
                                 context.stream()));
  });
}

}  // namespace llmp::kernels::ggml
