// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 The ds4.c authors
// SPDX-FileCopyrightText: 2026 Entrpi <entrpi@proton.me> (batched-serving fork modifications)
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// The original ds4 fused own output-A core (dsv4_ds4_product_original.cuh)
// behind a narrow CUDA ABI for jitLLM's output-A adapter (dsv4_outa.cu).
// Only its device kernels are used; no original context, registry, pool,
// allocator or host dispatcher is instantiated. The adapter supplies the
// operands and scratch.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>

#include "kernels/ggml/dsv4_ds4_product_raw.h"

namespace jitllm::kernels::ggml::ds4_product::original {
// Pre-included standard/vendor dependencies stay outside this namespace.
#include "kernels/ggml/dsv4_ds4_product_original.cuh"
}  // namespace jitllm::kernels::ggml::ds4_product::original

namespace jitllm::kernels::ggml::ds4_product {

using namespace original;

cudaError_t OutA(const void* scales, const void* codes, const float* heads, float* low, void* table,
                 void* quantized, std::uint32_t rows, Rope r, cudaStream_t stream) {
  const auto table_rows = ((static_cast<std::uint64_t>(rows) + 127) / 128) * 128;
  // The original core reads table entries for its dummy CTA rows and stores
  // whole WMMA rows. Explicit physical padding closes those source bounds.
  auto status = cudaMemsetAsync(table, 0, table_rows * 32 * sizeof(float2), stream);
  if (status != cudaSuccess) return status;
  attention_outa_rope_cs_table_kernel<<<(rows * 32 + 255) / 256, 256, 0, stream>>>(
      static_cast<float2*>(table), rows, r.first, r.positions, r.original_context, r.base, r.scale,
      r.extension, r.attention, r.beta_fast, r.beta_slow);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  const dim3 grid(kOARank / kOATileN, (rows + kOATileM - 1) / kOATileM, kOAGroups);
  constexpr auto shared = (2 * kOATileM + 2 * kOATileN) * (kOATileK + kOAPad) * sizeof(__half);
  attention_outa_fused_own_kernel<<<grid, kOAWarps * 32, shared, stream>>>(
      low, heads, static_cast<const float2*>(table), static_cast<const __half*>(scales),
      static_cast<const std::int8_t*>(codes), static_cast<char*>(quantized), rows);
  return cudaGetLastError();
}
}  // namespace jitllm::kernels::ggml::ds4_product
