// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Private CUDA ABI between jitLLM's completion-aware adapter and the
// separately compiled original ds4 numerical header closure. No original
// host runtime, device catalog, pool, handle or context is constructed.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_PRODUCT_RAW_H_
#define JITLLM_KERNELS_GGML_DSV4_DS4_PRODUCT_RAW_H_

#include <cuda_runtime_api.h>

#include <cstdint>

namespace jitllm::kernels::ggml::ds4_product {

struct Device {
  int cc = 0;
  int multiprocessors = 0;
  int warp = 0;
  std::uint64_t shared_optin = 0;
};
cudaError_t Embedding(const std::int32_t* tokens, const void* weights, float* output,
                      std::uint32_t vocab, std::uint32_t rows, std::uint32_t width,
                      std::uint32_t hc, cudaStream_t stream);
cudaError_t F16Conversion(const float* input, void* output, std::uint64_t count,
                          cudaStream_t stream);
cudaError_t QkvNorm(const float* q, const float* qw, float* qo, std::uint32_t qn, const float* kv,
                    const float* kvw, float* kvo, std::uint32_t kvn, std::uint32_t rows,
                    float epsilon, cudaStream_t stream);
struct MmqPlan {
  int columns = 0;
  int rows = 0;
  int warps = 0;
  int blocks = 0;
  int column_tiles = 0;
  int shared = 0;
  std::uint64_t fixup = 0;
};
bool PlanMmq(Device device, int m, int n, int k, MmqPlan& plan);
cudaError_t F16Vector(const void* weights, const float* input, float* output, int m, int n, int k,
                      int split, void* partial, std::uint64_t partial_bytes, cudaStream_t stream);
cudaError_t Mmq(const void* raw, const void* quantized, float* output, int m, int n, int k,
                const MmqPlan& plan, void* fixup, std::uint64_t fixup_bytes, cudaStream_t stream);
cudaError_t D4(const float* input, void* output, int rows, int columns, cudaStream_t stream);
// Original post-MMQ kernel, also paid by the native consumer control.
cudaError_t Sanitize(float* output, std::uint64_t count, cudaStream_t stream);
cudaError_t Q81(const float* input, void* output, int rows, int columns, cudaStream_t stream);
cudaError_t Q8Vector(const void* raw, const void* scales, const void* codes, const void* quantized,
                     float* output, int m, int n, int k, bool aligned, cudaStream_t stream);
std::uint64_t DenseD2rSharedBytes();
cudaError_t DenseD2r(const void* scales, const void* codes, const void* quantized, float* output,
                     int m, int n, int k, cudaStream_t stream);

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
cudaError_t HeadRope(float* input, std::uint32_t rows, std::uint32_t heads, std::uint32_t width,
                     Rope rope, bool normalize, float epsilon, cudaStream_t stream);
cudaError_t OutA(const void* scales, const void* codes, const float* heads, float* low, void* table,
                 void* quantized, std::uint32_t rows, Rope rope, cudaStream_t stream);

}  // namespace jitllm::kernels::ggml::ds4_product
#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_PRODUCT_RAW_H_
