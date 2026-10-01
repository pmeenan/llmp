// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Private CUDA ABI. Plain pointers/device facts replace original host runtime.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_MOE_RAW_H_
#define JITLLM_KERNELS_GGML_DSV4_DS4_MOE_RAW_H_
#include <cuda_runtime_api.h>

#include <cstdint>
namespace jitllm::kernels::ggml::ds4_moe {
struct Device {
  int cc = 0, sm = 0, warp = 0;
  std::uint64_t shared = 0;
};
struct MmqPlan {
  int x = 0, y = 0, warps = 0, blocks = 0, column_tiles = 0, shared = 0;
  std::uint64_t fixup = 0;
};
struct Plan {
  MmqPlan gate{}, down{};
  std::uint64_t fixup = 0;
};
struct Router {
  float* logits;
  const float* bias;
  const std::int32_t* hash;
  const std::int32_t* tokens;
  std::int32_t* selected;
  float* weights;
  float* probabilities;
  int rows;
  std::uint32_t hash_rows;
  int select;
};
struct Call {
  const float* input;
  const void* gate_weights;
  const void* up_weights;
  const void* down_weights;
  const std::int32_t* selected;
  const float* weights;
  std::int32_t* compact_ids;
  float* compact_weights;
  std::int32_t* ids_source;
  std::int32_t* ids_destination;
  std::int32_t* bounds;
  int* work;
  void* input_quant;
  void* down_quant;
  float* gate;
  float* up;
  float* middle;
  float* down;
  float* sum;
  int rows, input_width, middle_width, output_width, selected_stride, weight_stride, tier;
  bool produced;
};
struct PostPair {
  const float* gate;
  const float* up;
  const float* weights;
  const std::int32_t* ids_destination;
  const std::int32_t* bounds;
  const void* down_weights;
  float* middle;
  void* down_quant;
  int* work;
  float* down;
  float* sum;
};
struct Maps {
  const std::int32_t* selected;
  std::int32_t* ids_source;
  std::int32_t* ids_destination;
  std::int32_t* bounds;
};
bool PlanClassic(Device device, int rows, int input, int middle, int output, Plan& plan);
bool CooperativeFits(int sm, int rows);
cudaError_t Select(const Router& router, cudaStream_t stream);
cudaError_t Cooperative(const Router& router, const float* input, const void* weights,
                        float* partials, cudaStream_t stream);
cudaError_t Shared(const float* gate, const float* up, float* out, int rows, int width,
                   cudaStream_t stream);
cudaError_t Sum(const float* down, float* out, int rows, int width, cudaStream_t stream);
cudaError_t Moe(Device device, const Call& call, const Plan& plan, void* fixup,
                cudaStream_t stream);
cudaError_t MoePostPair(const PostPair& call, cudaStream_t stream);
cudaError_t MoeMaps(Device device, const Maps& call, cudaStream_t stream);
}  // namespace jitllm::kernels::ggml::ds4_moe
#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_MOE_RAW_H_
