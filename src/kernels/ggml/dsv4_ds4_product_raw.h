// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Private CUDA ABI between llmpalooza's output-A adapter (dsv4_outa.cu) and the
// original ds4 fused own output-A core (dsv4_ds4_product_raw.cu). No
// original host runtime, device catalog, pool, handle or context is
// constructed.
#ifndef LLMP_KERNELS_GGML_DSV4_DS4_PRODUCT_RAW_H_
#define LLMP_KERNELS_GGML_DSV4_DS4_PRODUCT_RAW_H_

#include <cuda_runtime_api.h>

#include <cstdint>

namespace llmp::kernels::ggml::ds4_product {

// Copy of the neutral parameters rather than original ds4 tensor objects.
struct Rope {
  const std::int32_t* positions = nullptr;
  std::uint32_t first = 0;
  std::uint32_t step = 1;
  std::uint32_t original_context = 0;
  std::uint32_t rotary = 0;
  float base = 0;
  float scale = 0;
  float extension = 0;
  float attention = 0;
  float beta_fast = 0;
  float beta_slow = 0;
  bool inverse = false;
};
cudaError_t OutA(const void* scales, const void* codes, const float* heads, float* low, void* table,
                 void* quantized, std::uint32_t rows, Rope rope, cudaStream_t stream);

}  // namespace llmp::kernels::ggml::ds4_product
#endif  // LLMP_KERNELS_GGML_DSV4_DS4_PRODUCT_RAW_H_
