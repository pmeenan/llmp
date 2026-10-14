// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// The elementwise, row, routing, recurrent and DeepSeek V4 operations of
// ops_ext.h over GGML's launchers. PlanTopK is a recorded copy of what
// ggml_cuda_op_top_k draws from the pool in llmpalooza's build, which has no
// CUB (third_party/patches/ggml/0001): top_k_radix_cuda's state and
// histograms for rows over 1,024, else the bitonic argsort's full indices
// (top-k.cu:175-181 and 262-268 at llama.cpp b29c606e2).

#include <algorithm>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>

#include "argsort.cuh"
#include "base/bytes.h"
#include "binbcast.cuh"
#include "clamp.cuh"
#include "common.cuh"
#include "concat.cuh"
#include "conv2d.cuh"
#include "dsv4-hc.cuh"
#include "fill.cuh"
#include "fwht.cuh"
#include "gated_delta_net.cuh"
#include "getrows.cuh"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
#include "lightning-indexer.cuh"
#include "rope.cuh"
#include "scale.cuh"
#include "set-rows.cuh"
#include "ssm-conv.cuh"
#include "sumrows.cuh"
#include "top-k.cuh"
#include "unary.cuh"
#include "upscale.cuh"

// PlanTopK sizes the paths of a build without CUB (patches/ggml/0001).
#if defined(GGML_CUDA_USE_CUB)
#error "llmpalooza's GGML build takes no CUB; PlanTopK sizes the radix and bitonic paths"
#endif

namespace llmp::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

const ggml_cuda_device_info::cuda_device_info& Device(const LaunchContext& launch) {
  return ggml_cuda_info().devices[launch.device()];
}

using Launcher = void (*)(ggml_backend_cuda_context&, ggml_tensor*);

// The upstream launcher for a node CheckUnary accepts.
Launcher UnaryLauncher(const ggml_tensor* node) {
  if (node->op == GGML_OP_SQRT) {
    return &ggml_cuda_op_sqrt;
  }
  switch (ggml_get_unary_op(node)) {
    case GGML_UNARY_OP_ABS:
      return &ggml_cuda_op_abs;
    case GGML_UNARY_OP_SGN:
      return &ggml_cuda_op_sgn;
    case GGML_UNARY_OP_NEG:
      return &ggml_cuda_op_neg;
    case GGML_UNARY_OP_SILU:
      return &ggml_cuda_op_silu;
    case GGML_UNARY_OP_GELU:
      return &ggml_cuda_op_gelu;
    case GGML_UNARY_OP_TANH:
      return &ggml_cuda_op_tanh;
    case GGML_UNARY_OP_RELU:
      return &ggml_cuda_op_relu;
    case GGML_UNARY_OP_SIGMOID:
      return &ggml_cuda_op_sigmoid;
    case GGML_UNARY_OP_EXP:
      return &ggml_cuda_op_exp;
    case GGML_UNARY_OP_SOFTPLUS:
      return &ggml_cuda_op_softplus;
    default:
      return nullptr;
  }
}

// Checks, then runs one scratch-free launcher over the node.
std::expected<void, KernelFailure> RunChecked(LaunchContext& launch, ggml_tensor* node,
                                              std::expected<void, KernelFailure> checked,
                                              Launcher launcher) {
  if (!checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node, launcher](ggml_backend_cuda_context& context) {
    launcher(context, node);
  });
}

bool CanShiftLeft(const ggml_tensor& t) {
  const std::size_t elem = ggml_type_size(t.type);
  return t.ne[0] == 1 && (t.ne[1] == 1 || t.nb[1] == elem);
}

void ShiftLeftOne(ggml_tensor& t) {
  const std::size_t elem = ggml_type_size(t.type);
  for (int d = 0; d < 3; ++d) {
    t.ne[d] = t.ne[d + 1];
    t.nb[d] = t.nb[d + 1];
  }
  t.ne[3] = 1;
  t.nb[3] = t.nb[2] * static_cast<std::size_t>(t.ne[2]);
  if (t.ne[0] == 1) t.nb[0] = elem;
}

void ShiftBinaryLeft(ggml_tensor& dst, ggml_tensor& s0, ggml_tensor& s1) {
  dst.src[0] = &s0;
  dst.src[1] = &s1;
  while ((dst.ne[1] > 1 || dst.ne[2] > 1 || dst.ne[3] > 1) && CanShiftLeft(dst) &&
         CanShiftLeft(s0) && CanShiftLeft(s1)) {
    ShiftLeftOne(dst);
    ShiftLeftOne(s0);
    ShiftLeftOne(s1);
  }
}

__device__ __forceinline__ float ReduceBlockSum256(float val, float* shared) {
  val = warp_reduce_sum(val);
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  val = (threadIdx.x < 8) ? shared[threadIdx.x] : 0.0f;
  if (wid == 0) {
    val = warp_reduce_sum<8>(val);
    if (lane == 0) shared[8] = val;
  }
  __syncthreads();
  return shared[8];
}

template <bool kXIsBf16 = false>
__global__ void __launch_bounds__(256)
    LayerNormModulateRows4096Kernel(const void* __restrict__ x_in,
                                    const float* __restrict__ side_whc,
                                    const float* __restrict__ side_vec0,
                                    const float* __restrict__ side_vec1, int side_split_row,
                                    float* __restrict__ dst_rows,
                                    nv_bfloat16* __restrict__ dst_bf16, int rows, float eps) {
  __shared__ float shared[9];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const std::size_t row_off = static_cast<std::size_t>(row) * 4096;
  const float* s_vec =
      (side_vec0 != nullptr) ? (row < side_split_row ? side_vec0 : side_vec1) : nullptr;

  float reg_x[4][4];
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int c = tid * 4 + k * 1024;
    float4 v4;
    if constexpr (kXIsBf16) {
      const uint2 xp =
          *reinterpret_cast<const uint2*>(static_cast<const nv_bfloat16*>(x_in) + row_off + c);
      const float2 x01 = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&xp.x));
      const float2 x23 = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&xp.y));
      v4 = make_float4(x01.x, x01.y, x23.x, x23.y);
    } else {
      v4 = *reinterpret_cast<const float4*>(static_cast<const float*>(x_in) + row_off + c);
    }
    reg_x[k][0] = v4.x;
    reg_x[k][1] = v4.y;
    reg_x[k][2] = v4.z;
    reg_x[k][3] = v4.w;
    sum += (v4.x + v4.y) + (v4.z + v4.w);
  }
  const float mean = ReduceBlockSum256(sum, shared) * (1.0f / 4096.0f);

  float var = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float d = reg_x[k][j] - mean;
      var += d * d;
    }
  }
  const float inv_std = rsqrtf(ReduceBlockSum256(var, shared) * (1.0f / 4096.0f) + eps);

  float* dst = dst_rows != nullptr ? (dst_rows + row_off) : nullptr;
  nv_bfloat16* dst_b = dst_bf16 != nullptr ? (dst_bf16 + row_off) : nullptr;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int c = tid * 4 + k * 1024;
    float4 s4;
    if (s_vec != nullptr) {
      s4 = *reinterpret_cast<const float4*>(s_vec + c);
    } else {
      const std::size_t r_sz = static_cast<std::size_t>(row);
      const std::size_t rows_sz = static_cast<std::size_t>(rows);
      s4.x = side_whc[r_sz + rows_sz * (c + 0)];
      s4.y = side_whc[r_sz + rows_sz * (c + 1)];
      s4.z = side_whc[r_sz + rows_sz * (c + 2)];
      s4.w = side_whc[r_sz + rows_sz * (c + 3)];
    }
    const float o0 = (reg_x[k][0] - mean) * inv_std * (s4.x + 1.0f);
    const float o1 = (reg_x[k][1] - mean) * inv_std * (s4.y + 1.0f);
    const float o2 = (reg_x[k][2] - mean) * inv_std * (s4.z + 1.0f);
    const float o3 = (reg_x[k][3] - mean) * inv_std * (s4.w + 1.0f);
    if (dst != nullptr) {
      *reinterpret_cast<float4*>(dst + c) = make_float4(o0, o1, o2, o3);
    }
    if (dst_b != nullptr) {
      const nv_bfloat162 b01 = __floats2bfloat162_rn(o0, o1);
      const nv_bfloat162 b23 = __floats2bfloat162_rn(o2, o3);
      uint2 packed;
      packed.x = *reinterpret_cast<const unsigned int*>(&b01);
      packed.y = *reinterpret_cast<const unsigned int*>(&b23);
      *reinterpret_cast<uint2*>(dst_b + c) = packed;
    }
  }
}

__global__ void LayerNormModulateRowsKernel(const float* __restrict__ x_in,
                                            const float* __restrict__ side_whc,
                                            const float* __restrict__ side_vec0,
                                            const float* __restrict__ side_vec1, int side_split_row,
                                            float* __restrict__ dst_rows,
                                            nv_bfloat16* __restrict__ dst_bf16, int cols, int rows,
                                            float eps) {
  extern __shared__ float s_x[];
  __shared__ float shared[9];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float* dst = dst_rows != nullptr ? (dst_rows + static_cast<std::size_t>(row) * cols) : nullptr;
  nv_bfloat16* dst_b =
      dst_bf16 != nullptr ? (dst_bf16 + static_cast<std::size_t>(row) * cols) : nullptr;
  const float* s_vec =
      (side_vec0 != nullptr) ? (row < side_split_row ? side_vec0 : side_vec1) : nullptr;
  const bool cache_x = (cols <= 8192);
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += 256) {
    const float v = x_in[static_cast<std::size_t>(row) * cols + c];
    if (cache_x) s_x[c] = v;
    sum += v;
  }
  const float mean = ReduceBlockSum256(sum, shared) / static_cast<float>(cols);

  float var = 0.0f;
  for (int c = threadIdx.x; c < cols; c += 256) {
    const float v = cache_x ? s_x[c] : x_in[static_cast<std::size_t>(row) * cols + c];
    const float d = v - mean;
    var += d * d;
  }
  const float inv_std = rsqrtf(ReduceBlockSum256(var, shared) / static_cast<float>(cols) + eps);

  for (int c = threadIdx.x; c < cols; c += 256) {
    const float v = cache_x ? s_x[c] : x_in[static_cast<std::size_t>(row) * cols + c];
    const float s =
        (s_vec != nullptr)
            ? __ldg(&s_vec[c])
            : side_whc[static_cast<std::size_t>(row) + static_cast<std::size_t>(rows) * c];
    const float out_v = (v - mean) * inv_std * (s + 1.0f);
    if (dst != nullptr) dst[c] = out_v;
    if (dst_b != nullptr) dst_b[c] = __float2bfloat16(out_v);
  }
}

__global__ void LayerNormModulateWhcTiledKernel(const float* __restrict__ x_in,
                                                const float* __restrict__ side_whc,
                                                const float* __restrict__ side_vec0,
                                                const float* __restrict__ side_vec1,
                                                int side_split_row, float* __restrict__ dst_rows,
                                                nv_bfloat16* __restrict__ dst_bf16, int cols,
                                                int rows, float eps) {
  __shared__ float s_sum[8][33];
  __shared__ float s_sum_sq[8][33];
  __shared__ float s_mean[32];
  __shared__ float s_inv_std[32];
  __shared__ float tile[32][33];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = static_cast<int>(blockIdx.x) * 32 + tx;
  const bool valid_row = (row < rows);

  const float shift = valid_row ? x_in[row] : 0.0f;
  float sum_d = 0.0f;
  float sum_d2 = 0.0f;
  if (valid_row) {
    for (int c = ty; c < cols; c += 8) {
      const float v = x_in[static_cast<std::size_t>(row) + static_cast<std::size_t>(rows) * c];
      const float d = v - shift;
      sum_d += d;
      sum_d2 += d * d;
    }
  }
  s_sum[ty][tx] = sum_d;
  s_sum_sq[ty][tx] = sum_d2;
  __syncthreads();

  if (ty == 0) {
    const float total_d = s_sum[0][tx] + s_sum[1][tx] + s_sum[2][tx] + s_sum[3][tx] + s_sum[4][tx] +
                          s_sum[5][tx] + s_sum[6][tx] + s_sum[7][tx];
    const float total_d2 = s_sum_sq[0][tx] + s_sum_sq[1][tx] + s_sum_sq[2][tx] + s_sum_sq[3][tx] +
                           s_sum_sq[4][tx] + s_sum_sq[5][tx] + s_sum_sq[6][tx] + s_sum_sq[7][tx];
    const float mean_d = total_d / static_cast<float>(cols);
    const float var = fmaxf(0.0f, total_d2 / static_cast<float>(cols) - mean_d * mean_d);
    s_mean[tx] = shift + mean_d;
    s_inv_std[tx] = rsqrtf(var + eps);
  }
  __syncthreads();

  const float mean = s_mean[tx];
  const float inv_std = s_inv_std[tx];
  const float* s_vec =
      (side_vec0 != nullptr) ? (row < side_split_row ? side_vec0 : side_vec1) : nullptr;

  for (int base_c = 0; base_c < cols; base_c += 32) {
    for (int j = 0; j < 32; j += 8) {
      const int c = base_c + ty + j;
      if (valid_row && c < cols) {
        const std::size_t idx = static_cast<std::size_t>(row) + static_cast<std::size_t>(rows) * c;
        const float v = x_in[idx];
        const float s = (s_vec != nullptr) ? __ldg(&s_vec[c]) : side_whc[idx];
        tile[tx][ty + j] = (v - mean) * inv_std * (s + 1.0f);
      }
    }
    __syncthreads();
    for (int j = 0; j < 32; j += 8) {
      const int r = static_cast<int>(blockIdx.x) * 32 + ty + j;
      const int c = base_c + tx;
      if (r < rows && c < cols) {
        const float out_v = tile[ty + j][tx];
        const std::size_t out_idx =
            static_cast<std::size_t>(c) + static_cast<std::size_t>(cols) * r;
        if (dst_rows != nullptr) {
          dst_rows[out_idx] = out_v;
        }
        if (dst_bf16 != nullptr) {
          dst_bf16[out_idx] = __float2bfloat16(out_v);
        }
      }
    }
    __syncthreads();
  }
}

__global__ void GatedResidualF32Kernel(const float* carry, const float* branch, const float* gate,
                                       float* dst, std::int64_t n) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = carry[i] + branch[i] * tanhf(gate[i]);
  }
}

template <bool kBranchIsBf16 = false>
__global__ void GatedResidualRowsBranchF32Kernel(const float* __restrict__ carry,
                                                 const void* __restrict__ branch_rows,
                                                 const float* __restrict__ gate,
                                                 const float* __restrict__ gate_vec0,
                                                 const float* __restrict__ gate_vec1,
                                                 int gate_split_row, float* __restrict__ dst,
                                                 std::int64_t rows, std::int64_t cols) {
  __shared__ float tile[32][33];
  const std::int64_t base_col = static_cast<std::int64_t>(blockIdx.x) * 32;
  const std::int64_t base_row = static_cast<std::int64_t>(blockIdx.y) * 32;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  for (int j = 0; j < 32; j += 8) {
    const std::int64_t r = base_row + ty + j;
    const std::int64_t c = base_col + tx;
    if (r < rows && c < cols) {
      const std::int64_t r_idx = c + cols * r;
      if constexpr (kBranchIsBf16) {
        tile[ty + j][tx] = __bfloat162float(static_cast<const nv_bfloat16*>(branch_rows)[r_idx]);
      } else {
        tile[ty + j][tx] = static_cast<const float*>(branch_rows)[r_idx];
      }
    }
  }
  __syncthreads();
  const std::int64_t r = base_row + tx;
  const float* g_vec =
      (gate_vec0 != nullptr) ? (r < gate_split_row ? gate_vec0 : gate_vec1) : nullptr;
  for (int j = 0; j < 32; j += 8) {
    const std::int64_t c = base_col + ty + j;
    if (r < rows && c < cols) {
      const std::int64_t idx = r + rows * c;
      const float g = (g_vec != nullptr) ? __ldg(&g_vec[c]) : gate[idx];
      dst[idx] = carry[idx] + tile[tx][ty + j] * tanhf(g);
    }
  }
}

template <bool kBranchIsBf16 = false, bool kCarryIsBf16 = false>
__global__ void GatedResidualBothRowsToWhcF32Kernel(const void* __restrict__ carry_rows,
                                                    const void* __restrict__ branch_rows,
                                                    const float* __restrict__ gate,
                                                    const float* __restrict__ gate_vec0,
                                                    const float* __restrict__ gate_vec1,
                                                    int gate_split_row, float* __restrict__ dst_whc,
                                                    std::int64_t rows, std::int64_t cols) {
  __shared__ float tile[32][33];
  const std::int64_t base_col = static_cast<std::int64_t>(blockIdx.x) * 32;
  const std::int64_t base_row = static_cast<std::int64_t>(blockIdx.y) * 32;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const std::int64_t c_load = base_col + tx;
  for (int j = 0; j < 32; j += 8) {
    const std::int64_t r_load = base_row + ty + j;
    if (r_load < rows && c_load < cols) {
      const std::int64_t r_idx = c_load + cols * r_load;
      const float g = (gate_vec0 != nullptr)
                          ? __ldg(&(r_load < gate_split_row ? gate_vec0 : gate_vec1)[c_load])
                          : gate[r_load + rows * c_load];
      const float b_val =
          kBranchIsBf16 ? __bfloat162float(static_cast<const nv_bfloat16*>(branch_rows)[r_idx])
                        : static_cast<const float*>(branch_rows)[r_idx];
      const float c_val = kCarryIsBf16
                              ? __bfloat162float(static_cast<const nv_bfloat16*>(carry_rows)[r_idx])
                              : static_cast<const float*>(carry_rows)[r_idx];
      tile[ty + j][tx] = c_val + b_val * tanhf(g);
    }
  }
  __syncthreads();
  const std::int64_t r_store = base_row + tx;
  for (int j = 0; j < 32; j += 8) {
    const std::int64_t c_store = base_col + ty + j;
    if (r_store < rows && c_store < cols) {
      dst_whc[r_store + rows * c_store] = tile[tx][ty + j];
    }
  }
}

template <bool kBranchIsBf16 = false, bool kOutIsBf16 = false>
__global__ void GatedResidualWhcCarryToRowsKernel(const float* __restrict__ carry_whc,
                                                  const void* __restrict__ branch_rows,
                                                  const float* __restrict__ gate,
                                                  const float* __restrict__ gate_vec0,
                                                  const float* __restrict__ gate_vec1,
                                                  int gate_split_row, void* __restrict__ dst_rows,
                                                  std::int64_t rows, std::int64_t cols) {
  __shared__ float tile[32][33];
  const std::int64_t base_col = static_cast<std::int64_t>(blockIdx.x) * 32;
  const std::int64_t base_row = static_cast<std::int64_t>(blockIdx.y) * 32;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const std::int64_t r_load = base_row + tx;
  for (int j = 0; j < 32; j += 8) {
    const std::int64_t c_load = base_col + ty + j;
    if (r_load < rows && c_load < cols) {
      tile[tx][ty + j] = carry_whc[r_load + rows * c_load];
    }
  }
  __syncthreads();
  const std::int64_t c_store = base_col + tx;
  for (int j = 0; j < 32; j += 8) {
    const std::int64_t r_store = base_row + ty + j;
    if (r_store < rows && c_store < cols) {
      const std::int64_t out_idx = c_store + cols * r_store;
      const float g = (gate_vec0 != nullptr)
                          ? __ldg(&(r_store < gate_split_row ? gate_vec0 : gate_vec1)[c_store])
                          : gate[r_store + rows * c_store];
      const float b_val =
          kBranchIsBf16 ? __bfloat162float(static_cast<const nv_bfloat16*>(branch_rows)[out_idx])
                        : static_cast<const float*>(branch_rows)[out_idx];
      const float u = tile[ty + j][tx] + b_val * tanhf(g);
      if constexpr (kOutIsBf16) {
        static_cast<nv_bfloat16*>(dst_rows)[out_idx] = __float2bfloat16_rn(u);
      } else {
        static_cast<float*>(dst_rows)[out_idx] = u;
      }
    }
  }
}

template <bool kBranchIsBf16 = false, bool kCarryIsBf16 = false, bool kOutRowsIsBf16 = false>
__global__ void __launch_bounds__(256) GatedResidualLayerNormModulateRows4096Kernel(
    const void* __restrict__ carry_rows, const void* __restrict__ branch_rows,
    const float* __restrict__ gate_whc, const float* __restrict__ gate_vec0,
    const float* __restrict__ gate_vec1, int gate_split_row, void* __restrict__ residual_out,
    const float* __restrict__ side_whc, const float* __restrict__ side_vec0,
    const float* __restrict__ side_vec1, int side_split_row, float* __restrict__ norm_dst_rows,
    nv_bfloat16* __restrict__ dst_bf16, int rows, float eps) {
  __shared__ float shared[9];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const std::size_t row_off = static_cast<std::size_t>(row) * 4096;
  const float* g_vec =
      (gate_vec0 != nullptr) ? (row < gate_split_row ? gate_vec0 : gate_vec1) : nullptr;
  const float* s_vec =
      (side_vec0 != nullptr) ? (row < side_split_row ? side_vec0 : side_vec1) : nullptr;

  float reg_u[4][4];
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int c = tid * 4 + k * 1024;
    float4 c4;
    if constexpr (kCarryIsBf16) {
      const uint2 cp = *reinterpret_cast<const uint2*>(static_cast<const nv_bfloat16*>(carry_rows) +
                                                       row_off + c);
      const float2 c01 = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&cp.x));
      const float2 c23 = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&cp.y));
      c4 = make_float4(c01.x, c01.y, c23.x, c23.y);
    } else {
      c4 = *reinterpret_cast<const float4*>(static_cast<const float*>(carry_rows) + row_off + c);
    }
    float4 b4;
    if constexpr (kBranchIsBf16) {
      const uint2 bp = *reinterpret_cast<const uint2*>(
          static_cast<const nv_bfloat16*>(branch_rows) + row_off + c);
      const float2 b01 = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&bp.x));
      const float2 b23 = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&bp.y));
      b4 = make_float4(b01.x, b01.y, b23.x, b23.y);
    } else {
      b4 = *reinterpret_cast<const float4*>(static_cast<const float*>(branch_rows) + row_off + c);
    }
    float4 g4;
    if (g_vec != nullptr) {
      g4 = *reinterpret_cast<const float4*>(g_vec + c);
    } else {
      const std::size_t r_sz = static_cast<std::size_t>(row);
      const std::size_t rows_sz = static_cast<std::size_t>(rows);
      g4.x = gate_whc[r_sz + rows_sz * (c + 0)];
      g4.y = gate_whc[r_sz + rows_sz * (c + 1)];
      g4.z = gate_whc[r_sz + rows_sz * (c + 2)];
      g4.w = gate_whc[r_sz + rows_sz * (c + 3)];
    }
    const float u0 = c4.x + b4.x * tanhf(g4.x);
    const float u1 = c4.y + b4.y * tanhf(g4.y);
    const float u2 = c4.z + b4.z * tanhf(g4.z);
    const float u3 = c4.w + b4.w * tanhf(g4.w);
    if constexpr (kOutRowsIsBf16) {
      const nv_bfloat162 u01 = __floats2bfloat162_rn(u0, u1);
      const nv_bfloat162 u23 = __floats2bfloat162_rn(u2, u3);
      uint2 up;
      up.x = *reinterpret_cast<const unsigned int*>(&u01);
      up.y = *reinterpret_cast<const unsigned int*>(&u23);
      *reinterpret_cast<uint2*>(static_cast<nv_bfloat16*>(residual_out) + row_off + c) = up;
    } else {
      *reinterpret_cast<float4*>(static_cast<float*>(residual_out) + row_off + c) =
          make_float4(u0, u1, u2, u3);
    }
    reg_u[k][0] = u0;
    reg_u[k][1] = u1;
    reg_u[k][2] = u2;
    reg_u[k][3] = u3;
    sum += (u0 + u1) + (u2 + u3);
  }
  const float mean = ReduceBlockSum256(sum, shared) * (1.0f / 4096.0f);

  float var = 0.0f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float d = reg_u[k][j] - mean;
      var += d * d;
    }
  }
  const float inv_std = rsqrtf(ReduceBlockSum256(var, shared) * (1.0f / 4096.0f) + eps);

  float* dst = norm_dst_rows != nullptr ? (norm_dst_rows + row_off) : nullptr;
  nv_bfloat16* dst_b = dst_bf16 != nullptr ? (dst_bf16 + row_off) : nullptr;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int c = tid * 4 + k * 1024;
    float4 s4;
    if (s_vec != nullptr) {
      s4 = *reinterpret_cast<const float4*>(s_vec + c);
    } else {
      const std::size_t r_sz = static_cast<std::size_t>(row);
      const std::size_t rows_sz = static_cast<std::size_t>(rows);
      s4.x = side_whc[r_sz + rows_sz * (c + 0)];
      s4.y = side_whc[r_sz + rows_sz * (c + 1)];
      s4.z = side_whc[r_sz + rows_sz * (c + 2)];
      s4.w = side_whc[r_sz + rows_sz * (c + 3)];
    }
    const float o0 = (reg_u[k][0] - mean) * inv_std * (s4.x + 1.0f);
    const float o1 = (reg_u[k][1] - mean) * inv_std * (s4.y + 1.0f);
    const float o2 = (reg_u[k][2] - mean) * inv_std * (s4.z + 1.0f);
    const float o3 = (reg_u[k][3] - mean) * inv_std * (s4.w + 1.0f);
    if (dst != nullptr) {
      *reinterpret_cast<float4*>(dst + c) = make_float4(o0, o1, o2, o3);
    }
    if (dst_b != nullptr) {
      const nv_bfloat162 b01 = __floats2bfloat162_rn(o0, o1);
      const nv_bfloat162 b23 = __floats2bfloat162_rn(o2, o3);
      uint2 packed;
      packed.x = *reinterpret_cast<const unsigned int*>(&b01);
      packed.y = *reinterpret_cast<const unsigned int*>(&b23);
      *reinterpret_cast<uint2*>(dst_b + c) = packed;
    }
  }
}

template <bool kBranchIsBf16 = false>
__global__ void GatedResidualLayerNormModulateRowsKernel(
    const float* __restrict__ carry_rows, const void* __restrict__ branch_rows,
    const float* __restrict__ gate_whc, const float* __restrict__ gate_vec0,
    const float* __restrict__ gate_vec1, int gate_split_row, float* __restrict__ residual_out,
    const float* __restrict__ side_whc, const float* __restrict__ side_vec0,
    const float* __restrict__ side_vec1, int side_split_row, float* __restrict__ norm_dst_rows,
    nv_bfloat16* __restrict__ dst_bf16, int cols, int rows, float eps) {
  extern __shared__ float s_x[];
  __shared__ float shared[9];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const std::size_t row_off = static_cast<std::size_t>(row) * cols;
  float* u_row = residual_out + row_off;
  float* dst = norm_dst_rows != nullptr ? (norm_dst_rows + row_off) : nullptr;
  nv_bfloat16* dst_b = dst_bf16 != nullptr ? (dst_bf16 + row_off) : nullptr;
  const float* g_vec =
      (gate_vec0 != nullptr) ? (row < gate_split_row ? gate_vec0 : gate_vec1) : nullptr;
  const float* s_vec =
      (side_vec0 != nullptr) ? (row < side_split_row ? side_vec0 : side_vec1) : nullptr;
  const bool cache_x = (cols <= 8192);

  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += 256) {
    const float g =
        (g_vec != nullptr)
            ? __ldg(&g_vec[c])
            : gate_whc[static_cast<std::size_t>(row) + static_cast<std::size_t>(rows) * c];
    const float b_val =
        kBranchIsBf16 ? __bfloat162float(static_cast<const nv_bfloat16*>(branch_rows)[row_off + c])
                      : static_cast<const float*>(branch_rows)[row_off + c];
    const float u = carry_rows[row_off + c] + b_val * tanhf(g);
    u_row[c] = u;
    if (cache_x) s_x[c] = u;
    sum += u;
  }
  const float mean = ReduceBlockSum256(sum, shared) / static_cast<float>(cols);

  float var = 0.0f;
  for (int c = threadIdx.x; c < cols; c += 256) {
    const float u = cache_x ? s_x[c] : u_row[c];
    const float d = u - mean;
    var += d * d;
  }
  const float inv_std = rsqrtf(ReduceBlockSum256(var, shared) / static_cast<float>(cols) + eps);

  for (int c = threadIdx.x; c < cols; c += 256) {
    const float u = cache_x ? s_x[c] : u_row[c];
    const float s =
        (s_vec != nullptr)
            ? __ldg(&s_vec[c])
            : side_whc[static_cast<std::size_t>(row) + static_cast<std::size_t>(rows) * c];
    const float out_v = (u - mean) * inv_std * (s + 1.0f);
    if (dst != nullptr) dst[c] = out_v;
    if (dst_b != nullptr) dst_b[c] = __float2bfloat16(out_v);
  }
}

__global__ void RmsNormMulRope3dKernel(const void* __restrict__ x_in, bool x_is_bf16,
                                       const void* __restrict__ weight, bool weight_is_bf16,
                                       float eps, const std::int32_t* __restrict__ pos0,
                                       const std::int32_t* __restrict__ pos1,
                                       const std::int32_t* __restrict__ pos2, int d0, int d1,
                                       int d2, float freq_base, bool neox, void* __restrict__ dst,
                                       bool dst_is_f16, int dim, int heads, int rows, int out_rows,
                                       bool transpose_out) {
  __shared__ float s_sin[256];
  __shared__ float s_cos[256];
  __shared__ float s_w0[256];
  __shared__ float s_w1[256];
  __shared__ int s_c0[256];
  __shared__ int s_c1[256];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * 32 + tx;
  const int row = blockIdx.x;
  if (row >= out_rows) return;
  if (row >= rows) {
    if (dst_is_f16) {
      auto* y_h = static_cast<half*>(dst);
      for (int head = ty; head < heads; head += 8) {
        const std::size_t out_base =
            transpose_out
                ? static_cast<std::size_t>(dim) *
                      (static_cast<std::size_t>(row) + static_cast<std::size_t>(out_rows) * head)
                : static_cast<std::size_t>(dim) *
                      (static_cast<std::size_t>(head) + static_cast<std::size_t>(heads) * row);
        for (int c = tx; c < dim; c += 32) {
          y_h[out_base + c] = __ushort_as_half(0x0000U);
        }
      }
    }
    return;
  }

  const int half_dim = dim / 2;
  const int half0 = d0 / 2;
  const int half1 = d1 / 2;
  const std::int32_t p0_val = pos0[row];
  const std::int32_t p1_val = pos1[row];
  const std::int32_t p2_val = pos2[row];

  for (int pair = tid; pair < half_dim; pair += 256) {
    int axis_dim = d0;
    int local_pair = pair;
    int offset = 0;
    std::int32_t p = p0_val;
    if (pair >= half0 + half1) {
      axis_dim = d2;
      local_pair = pair - (half0 + half1);
      offset = d0 + d1;
      p = p2_val;
    } else if (pair >= half0) {
      axis_dim = d1;
      local_pair = pair - half0;
      offset = d0;
      p = p1_val;
    }
    const float angle =
        static_cast<float>(p) *
        powf(freq_base, -static_cast<float>(2 * local_pair) / static_cast<float>(axis_dim));
    float sin_a = 0.0f;
    float cos_a = 1.0f;
    sincosf(angle, &sin_a, &cos_a);
    const int c0 = offset + (neox ? local_pair : 2 * local_pair);
    const int c1 = c0 + (neox ? axis_dim / 2 : 1);
    const float w0 = weight_is_bf16 ? __bfloat162float(static_cast<const nv_bfloat16*>(weight)[c0])
                                    : static_cast<const float*>(weight)[c0];
    const float w1 = weight_is_bf16 ? __bfloat162float(static_cast<const nv_bfloat16*>(weight)[c1])
                                    : static_cast<const float*>(weight)[c1];
    s_sin[pair] = sin_a;
    s_cos[pair] = cos_a;
    s_w0[pair] = w0;
    s_w1[pair] = w1;
    s_c0[pair] = c0;
    s_c1[pair] = c1;
  }
  __syncthreads();

  for (int head = ty; head < heads; head += 8) {
    const std::size_t in_base =
        static_cast<std::size_t>(dim) *
        (static_cast<std::size_t>(head) + static_cast<std::size_t>(heads) * row);
    const std::size_t out_base =
        transpose_out ? static_cast<std::size_t>(dim) * (static_cast<std::size_t>(row) +
                                                         static_cast<std::size_t>(out_rows) * head)
                      : in_base;

    float sum_sq = 0.0f;
    if (x_is_bf16) {
      const auto* x_b = static_cast<const nv_bfloat16*>(x_in) + in_base;
      for (int c = tx; c < dim; c += 32) {
        const float v = __bfloat162float(x_b[c]);
        sum_sq += v * v;
      }
    } else {
      const auto* x_f = static_cast<const float*>(x_in) + in_base;
      for (int c = tx; c < dim; c += 32) {
        const float v = x_f[c];
        sum_sq += v * v;
      }
    }
    sum_sq = warp_reduce_sum(sum_sq);
    const float inv_rms = rsqrtf(sum_sq / static_cast<float>(dim) + eps);

    for (int pair = tx; pair < half_dim; pair += 32) {
      const int c0 = s_c0[pair];
      const int c1 = s_c1[pair];
      const float sin_a = s_sin[pair];
      const float cos_a = s_cos[pair];
      const float x0 = x_is_bf16
                           ? __bfloat162float(static_cast<const nv_bfloat16*>(x_in)[in_base + c0])
                           : static_cast<const float*>(x_in)[in_base + c0];
      const float x1 = x_is_bf16
                           ? __bfloat162float(static_cast<const nv_bfloat16*>(x_in)[in_base + c1])
                           : static_cast<const float*>(x_in)[in_base + c1];
      const float a = x0 * inv_rms * s_w0[pair];
      const float b = x1 * inv_rms * s_w1[pair];
      const float o0 = a * cos_a - b * sin_a;
      const float o1 = a * sin_a + b * cos_a;
      if (dst_is_f16) {
        auto* y_h = static_cast<half*>(dst) + out_base;
        y_h[c0] = __float2half_rn(o0);
        y_h[c1] = __float2half_rn(o1);
      } else {
        auto* y_f = static_cast<float*>(dst) + out_base;
        y_f[c0] = o0;
        y_f[c1] = o1;
      }
    }
  }
}

__global__ void RmsNormMulRope3dQKInplaceBf16Kernel(
    nv_bfloat16* __restrict__ q_inout, const void* __restrict__ q_weight, bool q_weight_is_bf16,
    float q_eps, nv_bfloat16* __restrict__ k_inout, const void* __restrict__ k_weight,
    bool k_weight_is_bf16, float k_eps, const std::int32_t* __restrict__ pos0,
    const std::int32_t* __restrict__ pos1, const std::int32_t* __restrict__ pos2, int d0, int d1,
    int d2, float freq_base, bool neox, int dim, int heads_q, int heads_k, int rows,
    float* __restrict__ freqs_ws, bool freqs_precomputed) {
  __shared__ float s_sin[256];
  __shared__ float s_cos[256];
  __shared__ float s_wq0[256];
  __shared__ float s_wq1[256];
  __shared__ float s_wk0[256];
  __shared__ float s_wk1[256];
  __shared__ int s_c0[256];
  __shared__ int s_c1[256];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * 32 + tx;
  const int block_threads = static_cast<int>(blockDim.y) * 32;
  const int row = blockIdx.x;
  if (row >= rows) return;

  const int half_dim = dim / 2;
  const int half0 = d0 / 2;
  const int half1 = d1 / 2;
  const std::int32_t p0_val = (freqs_ws != nullptr && freqs_precomputed) ? 0 : pos0[row];
  const std::int32_t p1_val = (freqs_ws != nullptr && freqs_precomputed) ? 0 : pos1[row];
  const std::int32_t p2_val = (freqs_ws != nullptr && freqs_precomputed) ? 0 : pos2[row];

  for (int pair = tid; pair < half_dim; pair += block_threads) {
    int axis_dim = d0;
    int local_pair = pair;
    int offset = 0;
    std::int32_t p = p0_val;
    if (pair >= half0 + half1) {
      axis_dim = d2;
      local_pair = pair - (half0 + half1);
      offset = d0 + d1;
      p = p2_val;
    } else if (pair >= half0) {
      axis_dim = d1;
      local_pair = pair - half0;
      offset = d0;
      p = p1_val;
    }
    float sin_a = 0.0f;
    float cos_a = 1.0f;
    if (freqs_ws != nullptr && freqs_precomputed) {
      const float2 cs = *reinterpret_cast<const float2*>(
          freqs_ws + static_cast<std::size_t>(row) * dim + 2 * pair);
      cos_a = cs.x;
      sin_a = cs.y;
    } else {
      const float angle =
          static_cast<float>(p) *
          powf(freq_base, -static_cast<float>(2 * local_pair) / static_cast<float>(axis_dim));
      sincosf(angle, &sin_a, &cos_a);
      if (freqs_ws != nullptr) {
        *reinterpret_cast<float2*>(freqs_ws + static_cast<std::size_t>(row) * dim + 2 * pair) =
            make_float2(cos_a, sin_a);
      }
    }
    const int c0 = offset + (neox ? local_pair : 2 * local_pair);
    const int c1 = c0 + (neox ? axis_dim / 2 : 1);
    s_sin[pair] = sin_a;
    s_cos[pair] = cos_a;
    if (q_inout != nullptr) {
      s_wq0[pair] = q_weight_is_bf16
                        ? __bfloat162float(static_cast<const nv_bfloat16*>(q_weight)[c0])
                        : static_cast<const float*>(q_weight)[c0];
      s_wq1[pair] = q_weight_is_bf16
                        ? __bfloat162float(static_cast<const nv_bfloat16*>(q_weight)[c1])
                        : static_cast<const float*>(q_weight)[c1];
    }
    s_wk0[pair] = k_weight_is_bf16 ? __bfloat162float(static_cast<const nv_bfloat16*>(k_weight)[c0])
                                   : static_cast<const float*>(k_weight)[c0];
    s_wk1[pair] = k_weight_is_bf16 ? __bfloat162float(static_cast<const nv_bfloat16*>(k_weight)[c1])
                                   : static_cast<const float*>(k_weight)[c1];
    s_c0[pair] = c0;
    s_c1[pair] = c1;
  }
  __syncthreads();

  auto apply_heads = [&](nv_bfloat16* __restrict__ x_inout, int heads, float eps, const float* s_w0,
                         const float* s_w1) {
    for (int head = ty; head < heads; head += static_cast<int>(blockDim.y)) {
      nv_bfloat16* x_b =
          x_inout + static_cast<std::size_t>(dim) *
                        (static_cast<std::size_t>(head) + static_cast<std::size_t>(heads) * row);
      float sum_sq = 0.0f;
      for (int c = tx; c < dim; c += 32) {
        const float v = __bfloat162float(x_b[c]);
        sum_sq += v * v;
      }
      sum_sq = warp_reduce_sum(sum_sq);
      const float inv_rms = rsqrtf(sum_sq / static_cast<float>(dim) + eps);

      for (int pair = tx; pair < half_dim; pair += 32) {
        const int c0 = s_c0[pair];
        const int c1 = s_c1[pair];
        const float sin_a = s_sin[pair];
        const float cos_a = s_cos[pair];
        const float x0 = __bfloat162float(x_b[c0]);
        const float x1 = __bfloat162float(x_b[c1]);
        const float a = x0 * inv_rms * s_w0[pair];
        const float b = x1 * inv_rms * s_w1[pair];
        const float o0 = a * cos_a - b * sin_a;
        const float o1 = a * sin_a + b * cos_a;
        x_b[c0] = __float2bfloat16_rn(o0);
        x_b[c1] = __float2bfloat16_rn(o1);
      }
    }
  };
  if (q_inout != nullptr) {
    apply_heads(q_inout, heads_q, q_eps, s_wq0, s_wq1);
  }
  apply_heads(k_inout, heads_k, k_eps, s_wk0, s_wk1);
}

__global__ void SwiGluToBf16Vec8Kernel(const uint4* __restrict__ gate, const uint4* __restrict__ up,
                                       uint4* __restrict__ dst_bf16, std::int64_t chunks) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= chunks) return;
  const uint4 g_vec = gate[idx];
  const uint4 u_vec = up[idx];
  const auto* g_pairs = reinterpret_cast<const nv_bfloat162*>(&g_vec);
  const auto* u_pairs = reinterpret_cast<const nv_bfloat162*>(&u_vec);
  uint4 out_vec;
  auto* out_pairs = reinterpret_cast<nv_bfloat162*>(&out_vec);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 g2 = __bfloat1622float2(g_pairs[i]);
    const float2 u2 = __bfloat1622float2(u_pairs[i]);
    const float v0 = (g2.x / (1.0f + expf(-g2.x))) * u2.x;
    const float v1 = (g2.y / (1.0f + expf(-g2.y))) * u2.y;
    out_pairs[i] = __floats2bfloat162_rn(v0, v1);
  }
  dst_bf16[idx] = out_vec;
}

template <typename InT>
__global__ void SwiGluToBf16Kernel(const InT* __restrict__ gate, const InT* __restrict__ up,
                                   nv_bfloat16* __restrict__ dst_bf16, std::int64_t n) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t base = idx * 2;
  if (base + 1 < n) {
    float2 g2;
    float2 u2;
    if constexpr (std::is_same_v<InT, nv_bfloat16>) {
      g2 = __bfloat1622float2(reinterpret_cast<const nv_bfloat162*>(gate)[idx]);
      u2 = __bfloat1622float2(reinterpret_cast<const nv_bfloat162*>(up)[idx]);
    } else {
      g2 = reinterpret_cast<const float2*>(gate)[idx];
      u2 = reinterpret_cast<const float2*>(up)[idx];
    }
    const float v0 = (g2.x / (1.0f + expf(-g2.x))) * u2.x;
    const float v1 = (g2.y / (1.0f + expf(-g2.y))) * u2.y;
    reinterpret_cast<nv_bfloat162*>(dst_bf16)[idx] = __floats2bfloat162_rn(v0, v1);
  } else if (base < n) {
    float g0;
    float u0;
    if constexpr (std::is_same_v<InT, nv_bfloat16>) {
      g0 = __bfloat162float(gate[base]);
      u0 = __bfloat162float(up[base]);
    } else {
      g0 = gate[base];
      u0 = up[base];
    }
    dst_bf16[base] = __float2bfloat16((g0 / (1.0f + expf(-g0))) * u0);
  }
}

template <bool kSilu>
__global__ void SpatialChannelNormF32Kernel(const float* __restrict__ x_whc,
                                            const float* __restrict__ gamma,
                                            float* __restrict__ dst_whc, int channels,
                                            std::int64_t pixels, float eps_sq, float scale) {
  const std::int64_t p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= pixels) return;
  float sum_sq = 0.0f;
  for (int c = 0; c < channels; ++c) {
    const float v = x_whc[p + pixels * c];
    sum_sq += v * v;
  }
  const float inv_norm = rsqrtf(fmaxf(sum_sq, eps_sq));
  for (int c = 0; c < channels; ++c) {
    const std::int64_t idx = p + pixels * c;
    float v = ((x_whc[idx] * inv_norm) * scale) * __ldg(&gamma[c]);
    if constexpr (kSilu) {
      v = v / (1.0f + expf(-v));
    }
    dst_whc[idx] = v;
  }
}

__global__ void DupUpAddF32Kernel(const float* __restrict__ x, const float* __restrict__ sc,
                                  float* __restrict__ dst, int out_w, int out_h, int out_c, int ft,
                                  int repeats) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t total = static_cast<std::int64_t>(out_w) * out_h * out_c;
  if (idx >= total) return;
  const int ox = static_cast<int>(idx % out_w);
  const std::int64_t rem = idx / out_w;
  const int oy = static_cast<int>(rem % out_h);
  const int oc = static_cast<int>(rem / out_h);
  const int sc_w = out_w >> 1;
  const int sc_h = out_h >> 1;
  const int src_c = (((oc * ft + ft - 1) * 2 + (oy & 1)) * 2 + (ox & 1)) / repeats;
  const std::int64_t sc_idx = (ox >> 1) + static_cast<std::int64_t>(sc_w) *
                                              ((oy >> 1) + static_cast<std::int64_t>(sc_h) * src_c);
  dst[idx] = x[idx] + sc[sc_idx];
}

__global__ void AddBiasWhcF32Kernel(float* __restrict__ dst_whc, const float* __restrict__ bias,
                                    std::int64_t pixels, int channels) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t total = pixels * channels;
  if (idx >= total) return;
  const int c = static_cast<int>(idx / pixels);
  dst_whc[idx] += __ldg(&bias[c]);
}

}  // namespace

std::expected<void, KernelFailure> MulMatHadamard(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMulMatHadamard(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    // CheckMulMatHadamard admits only what the transform takes, so it
    // always reports that it ran (fwht.cu:61-101).
    const bool ran = ggml_cuda_op_fwht(context, node->src[1], node);
    if (!ran) {
      ggml_cuda_error("ggml_cuda_op_fwht", __func__, __FILE__, __LINE__,
                      "the Hadamard transform refused a node its check admitted");
    }
  });
}

std::expected<void, KernelFailure> Unary(LaunchContext& launch, ggml_tensor* node) {
  auto checked = CheckUnary(node);
  return RunChecked(launch, node, checked, checked ? UnaryLauncher(node) : nullptr);
}

std::expected<void, KernelFailure> Conv2dF32(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckConv2dF32(node), &ggml_cuda_op_conv2d);
}
std::expected<void, KernelFailure> UpsampleNearest(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckUpsampleNearest(node), &ggml_cuda_op_upscale);
}

std::expected<void, KernelFailure> Scale(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckScale(node), &ggml_cuda_op_scale);
}

std::expected<void, KernelFailure> Clamp(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckClamp(node), &ggml_cuda_op_clamp);
}

std::expected<void, KernelFailure> Fill(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckFill(node), &ggml_cuda_op_fill);
}

std::expected<void, KernelFailure> Repeat(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckRepeat(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_tensor dst = *node;
    ggml_tensor s0 = *node->src[0];
    dst.src[0] = &s0;
    while ((dst.ne[1] > 1 || dst.ne[2] > 1 || dst.ne[3] > 1) && CanShiftLeft(dst) &&
           CanShiftLeft(s0)) {
      ShiftLeftOne(dst);
      ShiftLeftOne(s0);
    }
    ggml_cuda_op_repeat(context, &dst);
  });
}

std::expected<void, KernelFailure> Sub(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckBinary(node, GGML_OP_SUB); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_tensor dst = *node;
    ggml_tensor s0 = *node->src[0];
    ggml_tensor s1 = *node->src[1];
    ShiftBinaryLeft(dst, s0, s1);
    ggml_cuda_op_sub(context, &dst);
  });
}

std::expected<void, KernelFailure> Div(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckBinary(node, GGML_OP_DIV); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_tensor dst = *node;
    ggml_tensor s0 = *node->src[0];
    ggml_tensor s1 = *node->src[1];
    ShiftBinaryLeft(dst, s0, s1);
    ggml_cuda_op_div(context, &dst);
  });
}

std::expected<void, KernelFailure> Concat(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckConcat(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_tensor dst = *node;
    ggml_tensor s0 = *node->src[0];
    ggml_tensor s1 = *node->src[1];
    dst.src[0] = &s0;
    dst.src[1] = &s1;
    while (dst.op_params[0] > 0 && (dst.ne[1] > 1 || dst.ne[2] > 1 || dst.ne[3] > 1) &&
           CanShiftLeft(dst) && CanShiftLeft(s0) && CanShiftLeft(s1)) {
      ShiftLeftOne(dst);
      ShiftLeftOne(s0);
      ShiftLeftOne(s1);
      --dst.op_params[0];
    }
    ggml_cuda_op_concat(context, &dst);
  });
}

std::expected<void, KernelFailure> SumRows(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckSumRows(node), &ggml_cuda_op_sum_rows);
}

std::expected<void, KernelFailure> SwiGluClamp(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckSwiGluClamp(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    if (node->src[1] != nullptr) {
      ggml_tensor dst = *node;
      ggml_tensor s0 = *node->src[0];
      ggml_tensor s1 = *node->src[1];
      ShiftBinaryLeft(dst, s0, s1);
      ggml_cuda_op_swiglu_clamp(context, &dst);
      return;
    }
    ggml_cuda_op_swiglu_clamp(context, node);
  });
}

std::expected<void, KernelFailure> RopeExt(LaunchContext& launch, ggml_tensor* node) {
  auto checked = CheckRopeExt(node);
  return RunChecked(
      launch, node, checked,
      checked && node->op == GGML_OP_ROPE_BACK ? &ggml_cuda_op_rope_back : &ggml_cuda_op_rope);
}

std::expected<void, KernelFailure> GetRowsExt(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckGetRowsExt(node), &ggml_cuda_op_get_rows);
}

std::expected<void, KernelFailure> SetRowsExt(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckSetRowsExt(node), &ggml_cuda_op_set_rows);
}

std::expected<void, KernelFailure> LayerNormModulateF32(
    LaunchContext& launch, const ggml_tensor* x_in, const ggml_tensor* side_whc,
    ggml_tensor* dst_rows, float eps, bool x_is_whc, bool write_bf16_workspace, bool skip_f32_dst,
    SplitBroadcastWhc side_bcast) {
  const int cols = static_cast<int>(dst_rows->ne[0]);
  const int rows = static_cast<int>(ggml_nrows(dst_rows));
  const std::int64_t ne = ggml_nelements(dst_rows);
  const std::uint64_t bf16_bytes =
      write_bf16_workspace ? static_cast<std::uint64_t>(ne) * sizeof(nv_bfloat16) : 0U;
  if (bf16_bytes > launch.workspace().size.value()) {
    return Rejected("LayerNormModulateF32 BF16 workspace exceeds pool");
  }
  const std::size_t smem = (!x_is_whc && cols != 4096 && cols <= 8192)
                               ? static_cast<std::size_t>(cols) * sizeof(float)
                               : 0U;
  return launch.Run(
      base::Bytes(bf16_bytes),
      [x_in, side_whc, dst_rows, cols, rows, ne, eps, x_is_whc, write_bf16_workspace, skip_f32_dst,
       side_bcast, smem](ggml_backend_cuda_context& context) {
        ggml_cuda_pool_alloc<nv_bfloat16> bf16_alloc(context.pool());
        nv_bfloat16* dst_bf16 =
            write_bf16_workspace ? bf16_alloc.alloc(static_cast<std::size_t>(ne)) : nullptr;
        float* dst_f32 = skip_f32_dst ? nullptr : static_cast<float*>(dst_rows->data);
        const float* side_ptr =
            side_whc != nullptr ? static_cast<const float*>(side_whc->data) : nullptr;
        if (x_is_whc) {
          const dim3 block(32, 8, 1);
          const dim3 grid(static_cast<unsigned>((rows + 31) / 32), 1, 1);
          LayerNormModulateWhcTiledKernel<<<grid, block, 0, context.stream()>>>(
              static_cast<const float*>(x_in->data), side_ptr, side_bcast.vec0, side_bcast.vec1,
              side_bcast.split_row, dst_f32, dst_bf16, cols, rows, eps);
        } else if (cols == 4096) {
          LayerNormModulateRows4096Kernel<false>
              <<<static_cast<unsigned>(rows), 256, 0, context.stream()>>>(
                  x_in->data, side_ptr, side_bcast.vec0, side_bcast.vec1, side_bcast.split_row,
                  dst_f32, dst_bf16, rows, eps);
        } else {
          LayerNormModulateRowsKernel<<<static_cast<unsigned>(rows), 256, smem, context.stream()>>>(
              static_cast<const float*>(x_in->data), side_ptr, side_bcast.vec0, side_bcast.vec1,
              side_bcast.split_row, dst_f32, dst_bf16, cols, rows, eps);
        }
        CUDA_CHECK(cudaGetLastError());
      });
}

std::expected<void, KernelFailure> GatedResidualF32(LaunchContext& launch, const ggml_tensor* carry,
                                                    const ggml_tensor* branch,
                                                    const ggml_tensor* gate, ggml_tensor* dst,
                                                    bool branch_is_rows, bool carry_is_rows,
                                                    SplitBroadcastWhc gate_bcast,
                                                    bool branch_is_bf16, bool carry_is_bf16) {
  const std::int64_t n = ggml_nelements(dst);
  return launch.Run(
      base::Bytes(0), [carry, branch, gate, dst, n, branch_is_rows, carry_is_rows, gate_bcast,
                       branch_is_bf16, carry_is_bf16](ggml_backend_cuda_context& ctx) {
        const float* gate_ptr = gate != nullptr ? static_cast<const float*>(gate->data) : nullptr;
        if (branch_is_rows && carry_is_rows) {
          const std::int64_t rows = dst->ne[0] * dst->ne[1];
          const std::int64_t cols = dst->ne[2];
          const dim3 block(32, 8, 1);
          const dim3 grid(static_cast<unsigned>((cols + 31) / 32),
                          static_cast<unsigned>((rows + 31) / 32), 1);
          if (branch_is_bf16 && carry_is_bf16) {
            GatedResidualBothRowsToWhcF32Kernel<true, true><<<grid, block, 0, ctx.stream()>>>(
                carry->data, branch->data, gate_ptr, gate_bcast.vec0, gate_bcast.vec1,
                gate_bcast.split_row, static_cast<float*>(dst->data), rows, cols);
          } else if (branch_is_bf16) {
            GatedResidualBothRowsToWhcF32Kernel<true, false><<<grid, block, 0, ctx.stream()>>>(
                carry->data, branch->data, gate_ptr, gate_bcast.vec0, gate_bcast.vec1,
                gate_bcast.split_row, static_cast<float*>(dst->data), rows, cols);
          } else if (carry_is_bf16) {
            GatedResidualBothRowsToWhcF32Kernel<false, true><<<grid, block, 0, ctx.stream()>>>(
                carry->data, branch->data, gate_ptr, gate_bcast.vec0, gate_bcast.vec1,
                gate_bcast.split_row, static_cast<float*>(dst->data), rows, cols);
          } else {
            GatedResidualBothRowsToWhcF32Kernel<false, false><<<grid, block, 0, ctx.stream()>>>(
                carry->data, branch->data, gate_ptr, gate_bcast.vec0, gate_bcast.vec1,
                gate_bcast.split_row, static_cast<float*>(dst->data), rows, cols);
          }
        } else if (branch_is_rows) {
          const std::int64_t rows = dst->ne[0] * dst->ne[1];
          const std::int64_t cols = dst->ne[2];
          const dim3 block(32, 8, 1);
          const dim3 grid(static_cast<unsigned>((cols + 31) / 32),
                          static_cast<unsigned>((rows + 31) / 32), 1);
          if (branch_is_bf16) {
            GatedResidualRowsBranchF32Kernel<true><<<grid, block, 0, ctx.stream()>>>(
                static_cast<const float*>(carry->data), branch->data, gate_ptr, gate_bcast.vec0,
                gate_bcast.vec1, gate_bcast.split_row, static_cast<float*>(dst->data), rows, cols);
          } else {
            GatedResidualRowsBranchF32Kernel<false><<<grid, block, 0, ctx.stream()>>>(
                static_cast<const float*>(carry->data), branch->data, gate_ptr, gate_bcast.vec0,
                gate_bcast.vec1, gate_bcast.split_row, static_cast<float*>(dst->data), rows, cols);
          }
        } else {
          const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
          GatedResidualF32Kernel<<<blocks, 256, 0, ctx.stream()>>>(
              static_cast<const float*>(carry->data), static_cast<const float*>(branch->data),
              gate_ptr, static_cast<float*>(dst->data), n);
        }
        CUDA_CHECK(cudaGetLastError());
      });
}

std::expected<void, KernelFailure> GatedResidualLayerNormModulateF32(
    LaunchContext& launch, const ggml_tensor* carry, const ggml_tensor* branch_rows,
    const ggml_tensor* gate_whc, ggml_tensor* residual_out, const ggml_tensor* side_whc,
    ggml_tensor* norm_dst_rows, float eps, bool carry_is_rows, bool write_bf16_workspace,
    bool skip_f32_dst, SplitBroadcastWhc gate_bcast, SplitBroadcastWhc side_bcast,
    bool branch_is_bf16, bool carry_is_bf16, bool residual_out_is_bf16) {
  const int cols = static_cast<int>(norm_dst_rows->ne[0]);
  const int rows = static_cast<int>(ggml_nrows(norm_dst_rows));
  const std::int64_t ne = ggml_nelements(norm_dst_rows);
  const std::uint64_t bf16_bytes =
      write_bf16_workspace ? static_cast<std::uint64_t>(ne) * sizeof(nv_bfloat16) : 0U;
  if (bf16_bytes > launch.workspace().size.value()) {
    return Rejected("GatedResidualLayerNormModulateF32 BF16 workspace exceeds pool");
  }
  const std::size_t smem =
      (cols != 4096 && cols <= 8192) ? static_cast<std::size_t>(cols) * sizeof(float) : 0U;
  return launch.Run(base::Bytes(bf16_bytes), [=](ggml_backend_cuda_context& ctx) {
    ggml_cuda_pool_alloc<nv_bfloat16> bf16_alloc(ctx.pool());
    nv_bfloat16* dst_bf16 =
        write_bf16_workspace ? bf16_alloc.alloc(static_cast<std::size_t>(ne)) : nullptr;
    float* dst_f32 = skip_f32_dst ? nullptr : static_cast<float*>(norm_dst_rows->data);
    const float* gate_ptr =
        gate_whc != nullptr ? static_cast<const float*>(gate_whc->data) : nullptr;
    const float* side_ptr =
        side_whc != nullptr ? static_cast<const float*>(side_whc->data) : nullptr;
    if (!carry_is_rows) {
      const dim3 block(32, 8, 1);
      const dim3 grid(static_cast<unsigned>((cols + 31) / 32),
                      static_cast<unsigned>((rows + 31) / 32), 1);
      if (branch_is_bf16 && residual_out_is_bf16 && cols == 4096) {
        GatedResidualWhcCarryToRowsKernel<true, true><<<grid, block, 0, ctx.stream()>>>(
            static_cast<const float*>(carry->data), branch_rows->data, gate_ptr, gate_bcast.vec0,
            gate_bcast.vec1, gate_bcast.split_row, residual_out->data, rows, cols);
        LayerNormModulateRows4096Kernel<true>
            <<<static_cast<unsigned>(rows), 256, 0, ctx.stream()>>>(
                residual_out->data, side_ptr, side_bcast.vec0, side_bcast.vec1,
                side_bcast.split_row, dst_f32, dst_bf16, rows, eps);
      } else {
        if (branch_is_bf16) {
          GatedResidualWhcCarryToRowsKernel<true, false><<<grid, block, 0, ctx.stream()>>>(
              static_cast<const float*>(carry->data), branch_rows->data, gate_ptr, gate_bcast.vec0,
              gate_bcast.vec1, gate_bcast.split_row, residual_out->data, rows, cols);
        } else {
          GatedResidualWhcCarryToRowsKernel<false, false><<<grid, block, 0, ctx.stream()>>>(
              static_cast<const float*>(carry->data), branch_rows->data, gate_ptr, gate_bcast.vec0,
              gate_bcast.vec1, gate_bcast.split_row, residual_out->data, rows, cols);
        }
        if (cols == 4096) {
          LayerNormModulateRows4096Kernel<false>
              <<<static_cast<unsigned>(rows), 256, 0, ctx.stream()>>>(
                  residual_out->data, side_ptr, side_bcast.vec0, side_bcast.vec1,
                  side_bcast.split_row, dst_f32, dst_bf16, rows, eps);
        } else {
          LayerNormModulateRowsKernel<<<static_cast<unsigned>(rows), 256, smem, ctx.stream()>>>(
              static_cast<const float*>(residual_out->data), side_ptr, side_bcast.vec0,
              side_bcast.vec1, side_bcast.split_row, dst_f32, dst_bf16, cols, rows, eps);
        }
      }
    } else if (cols == 4096) {
      if (branch_is_bf16 && carry_is_bf16 && residual_out_is_bf16) {
        GatedResidualLayerNormModulateRows4096Kernel<true, true, true>
            <<<static_cast<unsigned>(rows), 256, 0, ctx.stream()>>>(
                carry->data, branch_rows->data, gate_ptr, gate_bcast.vec0, gate_bcast.vec1,
                gate_bcast.split_row, residual_out->data, side_ptr, side_bcast.vec0,
                side_bcast.vec1, side_bcast.split_row, dst_f32, dst_bf16, rows, eps);
      } else if (branch_is_bf16) {
        GatedResidualLayerNormModulateRows4096Kernel<true, false, false>
            <<<static_cast<unsigned>(rows), 256, 0, ctx.stream()>>>(
                carry->data, branch_rows->data, gate_ptr, gate_bcast.vec0, gate_bcast.vec1,
                gate_bcast.split_row, residual_out->data, side_ptr, side_bcast.vec0,
                side_bcast.vec1, side_bcast.split_row, dst_f32, dst_bf16, rows, eps);
      } else {
        GatedResidualLayerNormModulateRows4096Kernel<false, false, false>
            <<<static_cast<unsigned>(rows), 256, 0, ctx.stream()>>>(
                carry->data, branch_rows->data, gate_ptr, gate_bcast.vec0, gate_bcast.vec1,
                gate_bcast.split_row, residual_out->data, side_ptr, side_bcast.vec0,
                side_bcast.vec1, side_bcast.split_row, dst_f32, dst_bf16, rows, eps);
      }
    } else {
      if (branch_is_bf16) {
        GatedResidualLayerNormModulateRowsKernel<true>
            <<<static_cast<unsigned>(rows), 256, smem, ctx.stream()>>>(
                static_cast<const float*>(carry->data), branch_rows->data, gate_ptr,
                gate_bcast.vec0, gate_bcast.vec1, gate_bcast.split_row,
                static_cast<float*>(residual_out->data), side_ptr, side_bcast.vec0, side_bcast.vec1,
                side_bcast.split_row, dst_f32, dst_bf16, cols, rows, eps);
      } else {
        GatedResidualLayerNormModulateRowsKernel<false>
            <<<static_cast<unsigned>(rows), 256, smem, ctx.stream()>>>(
                static_cast<const float*>(carry->data), branch_rows->data, gate_ptr,
                gate_bcast.vec0, gate_bcast.vec1, gate_bcast.split_row,
                static_cast<float*>(residual_out->data), side_ptr, side_bcast.vec0, side_bcast.vec1,
                side_bcast.split_row, dst_f32, dst_bf16, cols, rows, eps);
      }
    }
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RmsNormMulRope3d(LaunchContext& launch, const ggml_tensor* x_in,
                                                    const ggml_tensor* weight, float eps,
                                                    std::span<const ggml_tensor* const> pos_views,
                                                    std::span<const int> axis_dims, float freq_base,
                                                    bool neox, ggml_tensor* dst, bool transpose_out,
                                                    bool x_is_bf16, bool dst_is_f16,
                                                    int padded_rows, bool dst_in_workspace) {
  const int dim = static_cast<int>(x_in->ne[0]);
  const int heads = static_cast<int>(x_in->ne[1]);
  const int rows = static_cast<int>(x_in->ne[2]);
  const int out_rows = padded_rows > rows ? padded_rows : rows;
  const bool weight_is_bf16 = (weight->type == GGML_TYPE_BF16);
  const auto* p0 = static_cast<const std::int32_t*>(pos_views[0]->data);
  const auto* p1 = pos_views.size() > 1 ? static_cast<const std::int32_t*>(pos_views[1]->data) : p0;
  const auto* p2 = pos_views.size() > 2 ? static_cast<const std::int32_t*>(pos_views[2]->data) : p0;
  const int d0 = axis_dims[0];
  const int d1 = axis_dims.size() > 1 ? axis_dims[1] : 0;
  const int d2 = axis_dims.size() > 2 ? axis_dims[2] : 0;
  const std::uint64_t ws_bytes =
      dst_in_workspace
          ? static_cast<std::uint64_t>(dim) * static_cast<std::uint64_t>(heads) *
                static_cast<std::uint64_t>(out_rows) * (dst_is_f16 ? sizeof(half) : sizeof(float))
          : 0U;
  if (ws_bytes > launch.workspace().size.value()) {
    return Rejected("RmsNormMulRope3d workspace output exceeds pool");
  }
  return launch.Run(base::Bytes(ws_bytes), [=](ggml_backend_cuda_context& ctx) {
    ggml_cuda_pool_alloc<char> ws_alloc(ctx.pool());
    void* out_ptr =
        dst_in_workspace ? ws_alloc.alloc(static_cast<std::size_t>(ws_bytes)) : dst->data;
    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned>(out_rows), 1, 1);
    RmsNormMulRope3dKernel<<<grid, block, 0, ctx.stream()>>>(
        x_in->data, x_is_bf16, weight->data, weight_is_bf16, eps, p0, p1, p2, d0, d1, d2, freq_base,
        neox, out_ptr, dst_is_f16, dim, heads, rows, out_rows, transpose_out);
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RmsNormMulRope3dQKInplaceBf16(
    LaunchContext& launch, ggml_tensor* q_inout, const ggml_tensor* q_weight, float q_eps,
    ggml_tensor* k_inout, const ggml_tensor* k_weight, float k_eps,
    std::span<const ggml_tensor* const> pos_views, std::span<const int> axis_dims, float freq_base,
    bool neox, std::uint64_t freqs_ws_offset, bool freqs_precomputed) {
  const int dim = static_cast<int>(k_inout->ne[0]);
  const int heads_q = q_inout != nullptr ? static_cast<int>(q_inout->ne[1]) : 0;
  const int heads_k = static_cast<int>(k_inout->ne[1]);
  const int rows = static_cast<int>(k_inout->ne[2]);
  const bool q_weight_is_bf16 = (q_weight != nullptr && q_weight->type == GGML_TYPE_BF16);
  const bool k_weight_is_bf16 = (k_weight->type == GGML_TYPE_BF16);
  const auto* p0 = static_cast<const std::int32_t*>(pos_views[0]->data);
  const auto* p1 = pos_views.size() > 1 ? static_cast<const std::int32_t*>(pos_views[1]->data) : p0;
  const auto* p2 = pos_views.size() > 2 ? static_cast<const std::int32_t*>(pos_views[2]->data) : p0;
  const int d0 = axis_dims[0];
  const int d1 = axis_dims.size() > 1 ? axis_dims[1] : 0;
  const int d2 = axis_dims.size() > 2 ? axis_dims[2] : 0;
  const std::uint64_t freqs_bytes =
      freqs_ws_offset != UINT64_MAX
          ? ((static_cast<std::uint64_t>(dim) * static_cast<std::uint64_t>(rows) * sizeof(float)) +
             255) &
                ~std::uint64_t{255}
          : 0U;
  if (freqs_ws_offset != UINT64_MAX &&
      freqs_ws_offset + freqs_bytes > launch.workspace().size.value()) {
    return Rejected("RmsNormMulRope3dQKInplaceBf16 freqs workspace exceeds pool");
  }
  const std::uint64_t ws_base = launch.workspace().base;
  return launch.Run(base::Bytes(0), [=](ggml_backend_cuda_context& ctx) {
    float* freqs_ptr = freqs_ws_offset != UINT64_MAX
                           ? reinterpret_cast<float*>(ws_base + freqs_ws_offset)
                           : nullptr;
    const unsigned warps_y = (q_inout == nullptr && heads_k <= 4) ? 4U : 8U;
    const dim3 block(32, warps_y, 1);
    const dim3 grid(static_cast<unsigned>(rows), 1, 1);
    RmsNormMulRope3dQKInplaceBf16Kernel<<<grid, block, 0, ctx.stream()>>>(
        q_inout != nullptr ? static_cast<nv_bfloat16*>(q_inout->data) : nullptr,
        q_weight != nullptr ? q_weight->data : nullptr, q_weight_is_bf16, q_eps,
        static_cast<nv_bfloat16*>(k_inout->data), k_weight->data, k_weight_is_bf16, k_eps, p0, p1,
        p2, d0, d1, d2, freq_base, neox, dim, heads_q, heads_k, rows, freqs_ptr, freqs_precomputed);
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> SpatialChannelNormF32(LaunchContext& launch,
                                                         const ggml_tensor* x_whc,
                                                         const ggml_tensor* gamma,
                                                         ggml_tensor* dst_whc, float eps,
                                                         float scale, bool silu) {
  const std::int64_t pixels = x_whc->ne[0] * x_whc->ne[1];
  const int channels = static_cast<int>(x_whc->ne[2]);
  const float eps_sq = eps * eps;
  return launch.Run(base::Bytes(0), [=](ggml_backend_cuda_context& ctx) {
    const unsigned blocks = static_cast<unsigned>((pixels + 255) / 256);
    if (silu) {
      SpatialChannelNormF32Kernel<true><<<blocks, 256, 0, ctx.stream()>>>(
          static_cast<const float*>(x_whc->data), static_cast<const float*>(gamma->data),
          static_cast<float*>(dst_whc->data), channels, pixels, eps_sq, scale);
    } else {
      SpatialChannelNormF32Kernel<false><<<blocks, 256, 0, ctx.stream()>>>(
          static_cast<const float*>(x_whc->data), static_cast<const float*>(gamma->data),
          static_cast<float*>(dst_whc->data), channels, pixels, eps_sq, scale);
    }
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> DupUpAddF32(LaunchContext& launch, const ggml_tensor* x,
                                               const ggml_tensor* sc, ggml_tensor* dst, int ft,
                                               int repeats) {
  const int out_w = static_cast<int>(dst->ne[0]);
  const int out_h = static_cast<int>(dst->ne[1]);
  const int out_c = static_cast<int>(dst->ne[2]);
  const std::int64_t total = static_cast<std::int64_t>(out_w) * out_h * out_c;
  return launch.Run(base::Bytes(0), [=](ggml_backend_cuda_context& ctx) {
    const unsigned blocks = static_cast<unsigned>((total + 255) / 256);
    DupUpAddF32Kernel<<<blocks, 256, 0, ctx.stream()>>>(
        static_cast<const float*>(x->data), static_cast<const float*>(sc->data),
        static_cast<float*>(dst->data), out_w, out_h, out_c, ft, repeats);
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> SpatialConv1x1F32(LaunchContext& launch,
                                                     const ggml_tensor* x_whc,
                                                     const ggml_tensor* weight,
                                                     const ggml_tensor* bias,
                                                     ggml_tensor* dst_whc) {
  const int pixels = static_cast<int>(x_whc->ne[0] * x_whc->ne[1]);
  const int in_c = static_cast<int>(x_whc->ne[2]);
  const int out_c = static_cast<int>(dst_whc->ne[2]);
  const std::int64_t total = static_cast<std::int64_t>(pixels) * out_c;
  return launch.Run(base::Bytes(0), [=](ggml_backend_cuda_context& ctx) {
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), ctx.stream()));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, pixels, out_c, in_c,
                              &alpha, x_whc->data, CUDA_R_32F, pixels, weight->data, CUDA_R_32F,
                              in_c, &beta, dst_whc->data, CUDA_R_32F, pixels, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
    const unsigned blocks = static_cast<unsigned>((total + 255) / 256);
    AddBiasWhcF32Kernel<<<blocks, 256, 0, ctx.stream()>>>(
        static_cast<float*>(dst_whc->data), static_cast<const float*>(bias->data), pixels, out_c);
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> SwiGluToBf16(LaunchContext& launch, const ggml_tensor* gate,
                                                const ggml_tensor* up, bool swapped,
                                                bool bf16_inputs, void* dst_override) {
  const std::int64_t n = ggml_nelements(gate);
  const std::uint64_t bytes =
      dst_override != nullptr ? 0U : static_cast<std::uint64_t>(n) * sizeof(nv_bfloat16);
  if (bytes > launch.workspace().size.value()) {
    return Rejected("SwiGluToBf16 exceeds workspace");
  }
  const void* g_raw = swapped ? up->data : gate->data;
  const void* u_raw = swapped ? gate->data : up->data;
  return launch.Run(base::Bytes(bytes), [=](ggml_backend_cuda_context& ctx) {
    ggml_cuda_pool_alloc<nv_bfloat16> dst_alloc(ctx.pool());
    nv_bfloat16* out_ptr = dst_override != nullptr ? static_cast<nv_bfloat16*>(dst_override)
                                                   : dst_alloc.alloc(static_cast<std::size_t>(n));
    if (bf16_inputs && (n % 8) == 0 && (reinterpret_cast<std::uintptr_t>(g_raw) % 16) == 0 &&
        (reinterpret_cast<std::uintptr_t>(u_raw) % 16) == 0 &&
        (reinterpret_cast<std::uintptr_t>(out_ptr) % 16) == 0) {
      const std::int64_t chunks = n / 8;
      const unsigned blocks = static_cast<unsigned>((chunks + 255) / 256);
      SwiGluToBf16Vec8Kernel<<<blocks, 256, 0, ctx.stream()>>>(
          static_cast<const uint4*>(g_raw), static_cast<const uint4*>(u_raw),
          reinterpret_cast<uint4*>(out_ptr), chunks);
    } else {
      const std::int64_t pairs = (n + 1) / 2;
      const unsigned blocks = static_cast<unsigned>((pairs + 255) / 256);
      if (bf16_inputs) {
        SwiGluToBf16Kernel<<<blocks, 256, 0, ctx.stream()>>>(static_cast<const nv_bfloat16*>(g_raw),
                                                             static_cast<const nv_bfloat16*>(u_raw),
                                                             out_ptr, n);
      } else {
        SwiGluToBf16Kernel<<<blocks, 256, 0, ctx.stream()>>>(
            static_cast<const float*>(g_raw), static_cast<const float*>(u_raw), out_ptr, n);
      }
    }
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> Argsort(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckArgsort(node); !checked) {
    return checked;
  }
  // Upstream sorts in CUB beyond the shared memory the bitonic kernel may
  // use (argsort.cu:272-278); llmpalooza's build has no CUB, and the bitonic
  // launcher would assert.
  if (ArgsortSharedBytes(node) > Device(launch).smpb) {
    return Rejected("an argsort row beyond the device's shared memory");
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_argsort(context, node);
  });
}

std::expected<std::uint64_t, KernelFailure> PlanTopK(const LaunchContext& launch,
                                                     const ggml_tensor* node) {
  if (auto checked = CheckTopK(node); !checked) {
    return std::unexpected(checked.error());
  }
  const auto columns = static_cast<std::uint64_t>(node->src[0]->ne[0]);
  const auto rows = static_cast<std::uint64_t>(ggml_nrows(node->src[0]));
  if (columns <= 1024) {
    // The bitonic argsort of every row into scratch, then the first k of
    // each copied out; its padded row must fit shared memory
    // (argsort.cu:233-239).
    std::uint64_t padded = 1;
    while (padded < columns) {
      padded *= 2;
    }
    if (padded * sizeof(int) > Device(launch).smpb) {
      return Rejected("a top_k row beyond the device's shared memory");
    }
    // The bitonic kernel offsets each row in an int (argsort.cu:166).
    if (columns * rows > static_cast<std::uint64_t>(INT32_MAX)) {
      return Rejected("top_k rows beyond the bitonic kernel's 32-bit indexing");
    }
    return columns * rows * sizeof(int);
  }
  // top_k_radix_cuda: a state per row (top_k_radix_state: two uint32 and
  // three int), then 256 bins per row block, both held at once; one block
  // per row block, at most 64 per row.
  constexpr std::uint64_t kState = 5 * sizeof(std::int32_t);
  constexpr std::uint64_t kBins = 256;
  const std::uint64_t blocks_per_row = std::min<std::uint64_t>((columns + 1023) / 1024, 64);
  if (rows * blocks_per_row > static_cast<std::uint64_t>(INT32_MAX)) {
    return Rejected("top_k beyond the radix select's grid");
  }
  const std::uint64_t states = rows * kState;
  return ((states + 255) / 256 * 256) + (rows * blocks_per_row * kBins * sizeof(int));
}

std::expected<void, KernelFailure> TopK(LaunchContext& launch, ggml_tensor* node) {
  auto scratch = PlanTopK(launch, node);
  if (!scratch) {
    return std::unexpected(scratch.error());
  }
  return launch.Run(base::Bytes(*scratch), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_top_k(context, node);
  });
}

std::expected<void, KernelFailure> SsmConv(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckSsmConv(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_ssm_conv(context, node);
  });
}

std::expected<void, KernelFailure> GatedDeltaNet(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckGatedDeltaNet(node), &ggml_cuda_op_gated_delta_net);
}

std::expected<void, KernelFailure> LightningIndexer(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckLightningIndexer(node); !checked) {
    return checked;
  }
  // The launcher's tensor-core kernel (lightning-indexer.cu:447-465), which
  // needs Turing's matrix instructions; elsewhere it would take the vector
  // kernel, which is the same computation but not this implementation.
  if (!GGML_CUDA_CC_IS_NVIDIA(Device(launch).cc) || !turing_mma_available(Device(launch).cc)) {
    return Rejected("the lightning indexer's tensor-core kernel needs Turing or later");
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_lightning_indexer(context, node);
  });
}

std::expected<void, KernelFailure> HcComb(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckHcComb(node), &ggml_cuda_op_dsv4_hc_comb);
}

std::expected<void, KernelFailure> HcPre(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckHcPre(node), &ggml_cuda_op_dsv4_hc_pre);
}

std::expected<void, KernelFailure> HcPost(LaunchContext& launch, ggml_tensor* node) {
  return RunChecked(launch, node, CheckHcPost(node), &ggml_cuda_op_dsv4_hc_post);
}

}  // namespace llmp::kernels::ggml
