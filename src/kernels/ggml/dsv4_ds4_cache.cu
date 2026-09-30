// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Original cache/QAT numerical cores, borrowed by jitLLM's launch scope.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <expected>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/dsv4_ds4_cache.h"
#include "kernels/ggml/launch.h"

namespace jitllm::kernels::ggml {
namespace {

// Keep the original source inside this unit: no public CUDA/runtime types.
#include "kernels/ggml/dsv4_ds4_cache_core.cuh"

template <typename T>
T* Pointer(const Ds4CacheBuffer& buffer) {
  return reinterpret_cast<T*>(static_cast<std::uintptr_t>(buffer.address));
}

}  // namespace

std::expected<void, KernelFailure> RunDs4CacheQat(LaunchContext& launch, const Ds4CacheQat& desc) {
  if (auto checked = CheckDs4CacheQat(desc); !checked) return checked;
  return launch.Run(base::Bytes(0), [desc](auto& context) {
    auto* values = Pointer<float>(desc.values);
    auto* codes = Pointer<unsigned char>(desc.codes);
    auto* scales = Pointer<float>(desc.scales);
    if (desc.kind == Ds4CacheKind::kKv512) {
      fp8_kv_quantize_kernel<<<desc.rows, 64, 0, context.stream()>>>(values, desc.rows, kDs4KvWidth,
                                                                     64, codes, scales);
    } else {
      indexer_hadamard_fp4_kernel<<<desc.rows, 128, 0, context.stream()>>>(
          values, desc.rows, kDs4IndexerWidth, codes, scales);
    }
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunDs4CacheExpand(LaunchContext& launch,
                                                     const Ds4CacheExpand& desc) {
  if (auto checked = CheckDs4CacheExpand(desc); !checked) return checked;
  return launch.Run(base::Bytes(0), [desc](auto& context) {
    auto* values = Pointer<float>(desc.values);
    const auto* codes = Pointer<const unsigned char>(desc.codes);
    const auto* scales = Pointer<const float>(desc.scales);
    if (desc.kind == Ds4CacheKind::kKv512) {
      fp8_kv_dequant_rows_kernel<<<desc.rows, 128, 0, context.stream()>>>(
          values, codes, scales, desc.rows, kDs4KvWidth, Pointer<const float>(desc.decode_table));
    } else {
      indexer_fp4_dequant_rows_kernel<<<desc.rows, 128, 0, context.stream()>>>(
          values, codes, scales, desc.rows, kDs4IndexerWidth);
    }
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunDs4CacheRawStore(LaunchContext& launch,
                                                       const Ds4CacheRawStore& desc) {
  if (auto checked = CheckDs4CacheRawStore(desc); !checked) return checked;
  if (launch.capturing()) {
    return std::unexpected(
        KernelFailure{.error = KernelError::kRejected,
                      .detail = "ds4 raw batch position is not graph-replayable"});
  }
  return launch.Run(base::Bytes(0), [desc](auto& context) {
    const auto values = static_cast<std::uint64_t>(desc.rows) * kDs4KvWidth;
    const auto blocks = static_cast<std::uint32_t>((values + 255) / 256);
    store_raw_kv_batch_kernel<<<blocks, 256, 0, context.stream()>>>(
        Pointer<float>(desc.ring), Pointer<const float>(desc.source), desc.cells, desc.first,
        desc.rows, kDs4KvWidth, nullptr, nullptr, nullptr);
    CUDA_CHECK(cudaGetLastError());
  });
}

}  // namespace jitllm::kernels::ggml
