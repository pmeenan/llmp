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
#include <initializer_list>
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
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/llmp_ops.h"
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

// A BF16 residual (kOutIsBf16) only with the row cached (cols <= 8192):
// the norm reads the unrounded sum.
template <bool kBranchIsBf16 = false, bool kCarryIsBf16 = false, bool kOutIsBf16 = false>
__global__ void GatedResidualLayerNormModulateRowsKernel(
    const void* __restrict__ carry_rows, const void* __restrict__ branch_rows,
    const float* __restrict__ gate_whc, const float* __restrict__ gate_vec0,
    const float* __restrict__ gate_vec1, int gate_split_row, void* __restrict__ residual_out,
    const float* __restrict__ side_whc, const float* __restrict__ side_vec0,
    const float* __restrict__ side_vec1, int side_split_row, float* __restrict__ norm_dst_rows,
    nv_bfloat16* __restrict__ dst_bf16, int cols, int rows, float eps) {
  extern __shared__ float s_x[];
  __shared__ float shared[9];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const std::size_t row_off = static_cast<std::size_t>(row) * cols;
  float* u_row = kOutIsBf16 ? nullptr : static_cast<float*>(residual_out) + row_off;
  nv_bfloat16* u_row_bf16 =
      kOutIsBf16 ? static_cast<nv_bfloat16*>(residual_out) + row_off : nullptr;
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
    const float carry =
        kCarryIsBf16 ? __bfloat162float(static_cast<const nv_bfloat16*>(carry_rows)[row_off + c])
                     : static_cast<const float*>(carry_rows)[row_off + c];
    const float u = carry + b_val * tanhf(g);
    if constexpr (kOutIsBf16) {
      u_row_bf16[c] = __float2bfloat16(u);
    } else {
      u_row[c] = u;
    }
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
                                       bool transpose_out, bool dst_is_bf16 = false) {
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
      } else if (dst_is_bf16) {
        auto* y_b = static_cast<nv_bfloat16*>(dst) + out_base;
        y_b[c0] = __float2bfloat16_rn(o0);
        y_b[c1] = __float2bfloat16_rn(o1);
      } else {
        auto* y_f = static_cast<float*>(dst) + out_base;
        y_f[c0] = o0;
        y_f[c1] = o1;
      }
    }
  }
}

// llmp.image.gated_residual_split: an element a thread over rows [C, R]:
// carry + branch · tanh(gate of its row's range).
template <bool kBranchBf16, bool kCarryBf16>
__global__ void GatedResidualSplitKernel(const void* __restrict__ carry,
                                         const void* __restrict__ branch,
                                         const float* __restrict__ a, const float* __restrict__ b,
                                         int split, float* __restrict__ out, int cols,
                                         std::int64_t n) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int c = static_cast<int>(i % cols);
  const std::int64_t row = i / cols;
  const float g = row < split ? a[c] : b[c];
  const float x = kBranchBf16 ? __bfloat162float(static_cast<const nv_bfloat16*>(branch)[i])
                              : static_cast<const float*>(branch)[i];
  const float y = kCarryBf16 ? __bfloat162float(static_cast<const nv_bfloat16*>(carry)[i])
                             : static_cast<const float*>(carry)[i];
  out[i] = y + x * tanhf(g);
}

// llmp.image.rope_table: a row a block, a pair a thread: the pair's cos and
// sin of its section's angle, at [row][pair] (the layout
// FlashForwardBf16RowMajorKernel reads its rotation from).
__global__ void RopeTableKernel(const std::int32_t* __restrict__ pos0,
                                const std::int32_t* __restrict__ pos1,
                                const std::int32_t* __restrict__ pos2, int d0, int d1, int d2,
                                float freq_base, int dim, float* __restrict__ table) {
  const int row = blockIdx.x;
  const int half0 = d0 / 2;
  const int half1 = d1 / 2;
  for (int pair = static_cast<int>(threadIdx.x); pair < dim / 2;
       pair += static_cast<int>(blockDim.x)) {
    int axis_dim = d0;
    int local_pair = pair;
    std::int32_t p = pos0[row];
    if (pair >= half0 + half1) {
      axis_dim = d2;
      local_pair = pair - (half0 + half1);
      p = pos2[row];
    } else if (pair >= half0) {
      axis_dim = d1;
      local_pair = pair - half0;
      p = pos1[row];
    }
    const float angle =
        static_cast<float>(p) *
        powf(freq_base, -static_cast<float>(2 * local_pair) / static_cast<float>(axis_dim));
    float sin_a = 0.0f;
    float cos_a = 1.0f;
    sincosf(angle, &sin_a, &cos_a);
    *reinterpret_cast<float2*>(table + static_cast<std::size_t>(row) * dim + 2 * pair) =
        make_float2(cos_a, sin_a);
  }
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

bool Aligned16(std::initializer_list<const void*> pointers) {
  return std::ranges::all_of(
      pointers, [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; });
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

std::expected<void, KernelFailure> RmsNormMulRope3d(
    LaunchContext& launch, const ggml_tensor* x_in, const ggml_tensor* weight, float eps,
    std::span<const ggml_tensor* const> pos_views, std::span<const int> axis_dims, float freq_base,
    bool neox, ggml_tensor* dst, bool transpose_out, bool x_is_bf16, bool dst_is_f16,
    int padded_rows, bool dst_in_workspace, bool dst_is_bf16) {
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
        neox, out_ptr, dst_is_f16, dim, heads, rows, out_rows, transpose_out, dst_is_bf16);
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

std::expected<void, KernelFailure> RunSpatialChannelNorm(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckSpatialChannelNorm(node); !checked) return checked;
  return SpatialChannelNormF32(launch, node->src[0], node->src[1], node, LlmpOpFloat(node, 0),
                               LlmpOpFloat(node, 1), LlmpOpInt(node, 2) != 0);
}

std::expected<void, KernelFailure> RunLayerNormModulateSplit(LaunchContext& launch,
                                                             ggml_tensor* node) {
  if (auto checked = CheckLayerNormModulateSplit(node); !checked) return checked;
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* x = node->src[0];
    const auto* a = static_cast<const float*>(node->src[1]->data);
    const auto* b = static_cast<const float*>(node->src[2]->data);
    const int cols = static_cast<int>(x->ne[0]);
    const int rows = static_cast<int>(x->ne[1]);
    const int split = LlmpOpInt(node, 1);
    const float eps = LlmpOpFloat(node, 0);
    const bool bf16 = node->type == GGML_TYPE_BF16;
    float* dst_f32 = bf16 ? nullptr : static_cast<float*>(node->data);
    auto* dst_bf16 = bf16 ? static_cast<nv_bfloat16*>(node->data) : nullptr;
    // The 4096-wide kernel's vector loads and stores where aligned.
    if (cols == 4096 && Aligned16({x->data, a, b}) &&
        reinterpret_cast<std::uintptr_t>(node->data) % (bf16 ? 8 : 16) == 0) {
      LayerNormModulateRows4096Kernel<false>
          <<<static_cast<unsigned>(rows), 256, 0, context.stream()>>>(x->data, nullptr, a, b, split,
                                                                      dst_f32, dst_bf16, rows, eps);
    } else {
      const std::size_t smem = cols <= 8192 ? static_cast<std::size_t>(cols) * sizeof(float) : 0U;
      LayerNormModulateRowsKernel<<<static_cast<unsigned>(rows), 256, smem, context.stream()>>>(
          static_cast<const float*>(x->data), nullptr, a, b, split, dst_f32, dst_bf16, cols, rows,
          eps);
    }
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunGatedResidualSplit(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGatedResidualSplit(node); !checked) return checked;
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* carry = node->src[0];
    const ggml_tensor* branch = node->src[1];
    const auto* a = static_cast<const float*>(node->src[2]->data);
    const auto* b = static_cast<const float*>(node->src[3]->data);
    const std::int64_t n = ggml_nelements(node);
    const int cols = static_cast<int>(carry->ne[0]);
    const int split = LlmpOpInt(node, 1);
    const auto blocks = static_cast<unsigned>((n + 255) / 256);
    const auto launch_with = [&]<bool kBranch, bool kCarry>() {
      GatedResidualSplitKernel<kBranch, kCarry><<<blocks, 256, 0, context.stream()>>>(
          carry->data, branch->data, a, b, split, static_cast<float*>(node->data), cols, n);
    };
    const bool branch_bf16 = branch->type == GGML_TYPE_BF16;
    const bool carry_bf16 = carry->type == GGML_TYPE_BF16;
    if (branch_bf16 && carry_bf16) {
      launch_with.template operator()<true, true>();
    } else if (branch_bf16) {
      launch_with.template operator()<true, false>();
    } else if (carry_bf16) {
      launch_with.template operator()<false, true>();
    } else {
      launch_with.template operator()<false, false>();
    }
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunGatedResidualModulateSplit(LaunchContext& launch,
                                                                 ggml_tensor* residual,
                                                                 ggml_tensor* modulated) {
  if (auto checked = CheckGatedResidualModulateSplit(residual, modulated); !checked) return checked;
  return launch.Run(base::Bytes(0), [residual, modulated](ggml_backend_cuda_context& context) {
    const ggml_tensor* carry = residual->src[0];
    const ggml_tensor* branch = residual->src[1];
    const auto* ga = static_cast<const float*>(residual->src[2]->data);
    const auto* gb = static_cast<const float*>(residual->src[3]->data);
    const auto* ma = static_cast<const float*>(modulated->src[1]->data);
    const auto* mb = static_cast<const float*>(modulated->src[2]->data);
    const int cols = static_cast<int>(carry->ne[0]);
    const int rows = static_cast<int>(carry->ne[1]);
    const int split = LlmpOpInt(residual, 1);
    const float eps = LlmpOpFloat(modulated, 0);
    const bool bf16 = modulated->type == GGML_TYPE_BF16;
    float* dst_f32 = bf16 ? nullptr : static_cast<float*>(modulated->data);
    auto* dst_bf16 = bf16 ? static_cast<nv_bfloat16*>(modulated->data) : nullptr;
    const bool branch_bf16 = branch->type == GGML_TYPE_BF16;
    const bool carry_bf16 = carry->type == GGML_TYPE_BF16;
    const bool out_bf16 = residual->type == GGML_TYPE_BF16;
    const auto grid = static_cast<unsigned>(rows);
    // The 4096-wide kernel's vector loads and stores where aligned.
    const auto aligned = [](const void* p, bool half) {
      return reinterpret_cast<std::uintptr_t>(p) % (half ? 8 : 16) == 0;
    };
    const bool wide = cols == 4096 && Aligned16({ga, gb, ma, mb}) &&
                      aligned(carry->data, carry_bf16) && aligned(branch->data, branch_bf16) &&
                      aligned(residual->data, out_bf16) && aligned(modulated->data, bf16);
    const std::size_t smem = cols <= 8192 ? static_cast<std::size_t>(cols) * sizeof(float) : 0U;
    const auto run = [&]<bool kBranch, bool kCarry, bool kOut>() {
      if (wide) {
        GatedResidualLayerNormModulateRows4096Kernel<kBranch, kCarry, kOut>
            <<<grid, 256, 0, context.stream()>>>(carry->data, branch->data, nullptr, ga, gb, split,
                                                 residual->data, nullptr, ma, mb, split, dst_f32,
                                                 dst_bf16, rows, eps);
      } else {
        GatedResidualLayerNormModulateRowsKernel<kBranch, kCarry, kOut>
            <<<grid, 256, smem, context.stream()>>>(carry->data, branch->data, nullptr, ga, gb,
                                                    split, residual->data, nullptr, ma, mb, split,
                                                    dst_f32, dst_bf16, cols, rows, eps);
      }
    };
    const auto with_out = [&]<bool kBranch, bool kCarry>() {
      if (out_bf16) {
        run.template operator()<kBranch, kCarry, true>();
      } else {
        run.template operator()<kBranch, kCarry, false>();
      }
    };
    const auto with_carry = [&]<bool kBranch>() {
      if (carry_bf16) {
        with_out.template operator()<kBranch, true>();
      } else {
        with_out.template operator()<kBranch, false>();
      }
    };
    if (branch_bf16) {
      with_carry.template operator()<true>();
    } else {
      with_carry.template operator()<false>();
    }
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunDupUpAdd(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckDupUpAdd(node); !checked) return checked;
  const ggml_tensor* x = node->src[0];
  const ggml_tensor* sc = node->src[1];
  const int ft = LlmpOpInt(node, 0);
  const auto repeats = static_cast<int>(x->ne[2] * ft * 4 / sc->ne[2]);
  return DupUpAddF32(launch, x, sc, node, ft, repeats);
}

std::expected<void, KernelFailure> RunConv1x1(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckConv1x1(node); !checked) return checked;
  if (launch.cublas() == nullptr) return Rejected("the launch context lends no cuBLAS handle");
  // cuBLAS writes its workspace while it reads the operands.
  const auto lt = launch.cublas()->workspace();
  for (const ggml_tensor* t : {node, node->src[0], node->src[1], node->src[2]})
    if (auto clear = CheckClearOf(t, lt.base, lt.size.value()); !clear) return clear;
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* w = node->src[0];
    const ggml_tensor* x = node->src[1];
    const int pixels = static_cast<int>(x->ne[0] * x->ne[1]);
    const int in_c = static_cast<int>(x->ne[2]);
    const int out_c = static_cast<int>(node->ne[2]);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // [P, Ci] (pixels contiguous) times [Ci, Co] into [P, Co], column-major.
    CUBLAS_CHECK(cublasGemmEx(context.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N, pixels, out_c,
                              in_c, &alpha, x->data, CUDA_R_32F, pixels, w->data, CUDA_R_32F, in_c,
                              &beta, node->data, CUDA_R_32F, pixels, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
    const std::int64_t total = static_cast<std::int64_t>(pixels) * out_c;
    AddBiasWhcF32Kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, context.stream()>>>(
        static_cast<float*>(node->data), static_cast<const float*>(node->src[2]->data), pixels,
        out_c);
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunRopeTable(LaunchContext& launch, ggml_tensor* node) {
  const auto args = CheckRopeTable(node);
  if (!args) return std::unexpected(args.error());
  return launch.Run(base::Bytes(0), [node, a = *args](ggml_backend_cuda_context& ctx) {
    const auto* p0 = static_cast<const std::int32_t*>(a.positions[0]->data);
    const auto* p1 = a.sections > 1 ? static_cast<const std::int32_t*>(a.positions[1]->data) : p0;
    const auto* p2 = a.sections > 2 ? static_cast<const std::int32_t*>(a.positions[2]->data) : p0;
    const int dim = static_cast<int>(node->ne[1] * 2);
    RopeTableKernel<<<static_cast<unsigned>(node->ne[2]), 128, 0, ctx.stream()>>>(
        p0, p1, p2, a.dims[0], a.sections > 1 ? a.dims[1] : 0, a.sections > 2 ? a.dims[2] : 0,
        a.base, dim, static_cast<float*>(node->data));
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunSwiGluBf16(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckSwiGluBf16(node); !checked) return checked;
  return SwiGluToBf16(launch, node->src[0], node->src[1], /*swapped=*/false,
                      node->src[0]->type == GGML_TYPE_BF16, node->data);
}

std::expected<void, KernelFailure> RunNormRopeSections(LaunchContext& launch, ggml_tensor* node) {
  const auto args = CheckNormRopeSections(node);
  if (!args) return std::unexpected(args.error());
  return RmsNormMulRope3d(launch, node->src[0], node->src[1], args->eps,
                          std::span(args->positions).first(args->sections),
                          std::span(args->dims).first(args->sections), args->base, args->neox, node,
                          /*transpose_out=*/false, node->src[0]->type == GGML_TYPE_BF16,
                          /*dst_is_f16=*/false, /*padded_rows=*/0, /*dst_in_workspace=*/false,
                          node->type == GGML_TYPE_BF16);
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
