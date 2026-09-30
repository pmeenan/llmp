// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <expected>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/dsv4_ds4_hc.h"
#include "kernels/ggml/launch.h"

namespace jitllm::kernels::ggml {
namespace {

#include "kernels/ggml/dsv4_ds4_hc_core.cuh"

template <typename T>
T* Pointer(const Ds4CacheBuffer& buffer) {
  return reinterpret_cast<T*>(static_cast<std::uintptr_t>(buffer.address));
}

bool Absent(const Ds4CacheBuffer& buffer) { return buffer.address == 0 && buffer.bytes == 0; }

std::uint32_t Blocks(std::uint64_t elements) {
  return static_cast<std::uint32_t>((elements + 255) / 256);
}

}  // namespace

std::expected<void, KernelFailure> RunDs4Rms(LaunchContext& launch, const Ds4Rms& desc) {
  if (auto checked = CheckDs4Rms(desc); !checked) return checked;
  return launch.Run(base::Bytes(0), [desc](auto& context) {
    const auto* source = Pointer<const float>(desc.source);
    const auto* weights = Pointer<const float>(desc.weights);
    auto* values = Pointer<float>(desc.values);
    auto* half = Pointer<__half>(desc.values_f16);
    if (Absent(desc.weights)) {
      if (Absent(desc.values_f16)) {
        rms_norm_plain_kernel<<<desc.rows, 256, 0, context.stream()>>>(values, source, desc.width,
                                                                       desc.rows, desc.epsilon);
      } else {
        rms_norm_plain_f16_rows_kernel<<<desc.rows, 256, 0, context.stream()>>>(
            half, source, desc.width, desc.rows, desc.epsilon);
      }
    } else if (!Absent(desc.q8_d4)) {
      rms_norm_weight_f16q8_kernel<<<desc.rows, 256, 0, context.stream()>>>(
          values, half, Pointer<char>(desc.q8_d4), source, weights, desc.width, desc.rows,
          desc.epsilon);
    } else if (!Absent(desc.values_f16)) {
      rms_norm_weight_f16pair_kernel<<<desc.rows, 256, 0, context.stream()>>>(
          values, half, source, weights, desc.width, desc.rows, desc.epsilon);
    } else {
      rms_norm_weight_kernel<<<desc.rows, 256, 0, context.stream()>>>(
          values, source, weights, desc.width, desc.rows, desc.epsilon);
    }
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunDs4HcSplit(LaunchContext& launch, const Ds4HcSplit& desc) {
  if (auto checked = CheckDs4HcSplit(desc); !checked) return checked;
  return launch.Run(base::Bytes(0), [desc](auto& context) {
    hc_split_sinkhorn_kernel<<<Blocks(desc.rows), 256, 0, context.stream()>>>(
        Pointer<float>(desc.split), Pointer<const float>(desc.mix),
        Pointer<const float>(desc.scale), Pointer<const float>(desc.base), desc.rows,
        desc.iterations, desc.epsilon);
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunDs4HcWeighted(LaunchContext& launch,
                                                    const Ds4HcWeighted& desc) {
  if (auto checked = CheckDs4HcWeighted(desc); !checked) return checked;
  return launch.Run(base::Bytes(0), [desc](auto& context) {
    hc_weighted_sum_kernel<<<Blocks(static_cast<std::uint64_t>(desc.rows) * desc.width), 256, 0,
                             context.stream()>>>(
        Pointer<float>(desc.values), Pointer<const float>(desc.residual),
        Pointer<const float>(desc.weights), desc.width, kDs4HcLanes, desc.rows, desc.weight_stride);
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunDs4HcPre(LaunchContext& launch, const Ds4HcPre& desc) {
  if (auto checked = CheckDs4HcPre(desc); !checked) return checked;
  return launch.Run(base::Bytes(0), [desc](auto& context) {
    const auto& coefficients = desc.coefficients;
    hc_split_weighted_sum_fused_kernel<<<coefficients.rows, 256, 0, context.stream()>>>(
        Pointer<float>(desc.values), Pointer<float>(coefficients.split),
        Pointer<const float>(coefficients.mix), Pointer<const float>(desc.residual),
        Pointer<const float>(coefficients.scale), Pointer<const float>(coefficients.base),
        desc.width, kDs4HcLanes, coefficients.rows, coefficients.iterations, coefficients.epsilon);
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunDs4HcExpand(LaunchContext& launch, const Ds4HcExpand& desc) {
  if (auto checked = CheckDs4HcExpand(desc); !checked) return checked;
  return launch.Run(base::Bytes(0), [desc](auto& context) {
    const bool moe = !Absent(desc.moe_unsummed);
    const bool add = !Absent(desc.add);
    const auto* split = Pointer<const float>(desc.split);
    // Original folded core forms both row pointers unconditionally. Supply
    // a real, bounded view for each unused pointer, like the original shim.
    const auto* block = Pointer<const float>(moe ? desc.add : desc.block);
    const auto* extra = Pointer<const float>(add ? desc.add : desc.block);
    if (Absent(desc.values_f16)) {
      hc_expand_kernel<<<Blocks(static_cast<std::uint64_t>(desc.rows) * kDs4HcLanes * desc.width),
                         256, 0, context.stream()>>>(
          Pointer<float>(desc.values), block, extra, Pointer<const float>(desc.residual), split + 4,
          split + 8, desc.width, kDs4HcLanes, desc.rows, kDs4HcSplitWidth, kDs4HcSplitWidth,
          static_cast<int>(add));
    } else {
      hc_expand_split_rmsf16_kernel<4096U, 4U><<<desc.rows, 256, 0, context.stream()>>>(
          Pointer<float>(desc.values), Pointer<__half>(desc.values_f16), block, extra,
          Pointer<const float>(desc.residual), split + 4, split + 8, desc.rows, kDs4HcSplitWidth,
          kDs4HcSplitWidth, static_cast<int>(add), desc.epsilon,
          Pointer<const float>(desc.moe_unsummed), moe ? 6U : 0U);
    }
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunDs4HcHeadWeights(LaunchContext& launch,
                                                       const Ds4HcHeadWeights& desc) {
  if (auto checked = CheckDs4HcHeadWeights(desc); !checked) return checked;
  return launch.Run(base::Bytes(0), [desc](auto& context) {
    output_hc_weights_kernel<<<Blocks(static_cast<std::uint64_t>(desc.rows) * kDs4HcLanes), 256, 0,
                               context.stream()>>>(
        Pointer<float>(desc.values), Pointer<const float>(desc.pre),
        Pointer<const float>(desc.scale), Pointer<const float>(desc.base), kDs4HcLanes, desc.rows,
        desc.epsilon);
    CUDA_CHECK(cudaGetLastError());
  });
}

}  // namespace jitllm::kernels::ggml
