// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <cstring>
#include <expected>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "binbcast.cuh"
#include "common.cuh"
#include "cpy.cuh"
#include "getrows.cuh"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate.h"
#include "mmf.cuh"
#include "mmvf.cuh"
#include "norm.cuh"
#include "rope.cuh"
#include "set-rows.cuh"
#include "softmax.cuh"
#include "unary.cuh"

namespace llmp::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

// The launch context's device, read without a CUDA call: its table was
// read when the context was created.
const ggml_cuda_device_info::cuda_device_info& Device(const LaunchContext& launch) {
  return ggml_cuda_info().devices[launch.device()];
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

struct CollapsedCont {
  std::size_t dims = 0;
  std::array<std::int64_t, 4> ne{1, 1, 1, 1};
  std::array<std::size_t, 4> nb{sizeof(float), sizeof(float), sizeof(float), sizeof(float)};
};

CollapsedCont CollapseF32Cont(const ggml_tensor* src) {
  CollapsedCont c{};
  for (int d = 0; d < 4; ++d) {
    if (src->ne[d] > 1) {
      if (c.dims > 0 &&
          src->nb[d] == static_cast<std::size_t>(c.ne[c.dims - 1]) * c.nb[c.dims - 1]) {
        c.ne[c.dims - 1] *= src->ne[d];
      } else {
        c.ne[c.dims] = src->ne[d];
        c.nb[c.dims] = src->nb[d];
        ++c.dims;
      }
    }
  }
  if (c.dims == 0) c.dims = 1;
  return c;
}

__global__ void Transpose2dKernel(const char* __restrict__ src, float* __restrict__ dst,
                                  std::int64_t ne0, std::int64_t ne1, std::size_t nb0,
                                  std::size_t nb2) {
  __shared__ float tile[32][33];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const std::int64_t base1 = static_cast<std::int64_t>(blockIdx.x) * 32;
  const std::int64_t base0 = static_cast<std::int64_t>(blockIdx.y) * 32;
  const std::int64_t i2 = blockIdx.z;
  const char* batch_src = src + static_cast<std::size_t>(i2) * nb2;
  float* batch_dst = dst + i2 * ne0 * ne1;
  for (int j = 0; j < 32; j += 8) {
    const std::int64_t i0 = base0 + ty + j;
    const std::int64_t i1 = base1 + tx;
    if (i0 < ne0 && i1 < ne1) {
      const auto* row =
          reinterpret_cast<const float*>(batch_src + static_cast<std::size_t>(i0) * nb0);
      tile[ty + j][tx] = row[i1];
    }
  }
  __syncthreads();
  for (int j = 0; j < 32; j += 8) {
    const std::int64_t i1 = base1 + ty + j;
    const std::int64_t i0 = base0 + tx;
    if (i1 < ne1 && i0 < ne0) {
      batch_dst[i0 + ne0 * i1] = tile[tx][ty + j];
    }
  }
}

__global__ void Copy3dContigDim0Kernel(const char* __restrict__ src, float* __restrict__ dst,
                                       std::int64_t ne0, std::int64_t ne1, std::size_t nb1,
                                       std::size_t nb2) {
  const std::int64_t i0 = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t i1 = blockIdx.y;
  const std::int64_t i2 = blockIdx.z;
  if (i0 < ne0) {
    const auto* row = reinterpret_cast<const float*>(src + static_cast<std::size_t>(i1) * nb1 +
                                                     static_cast<std::size_t>(i2) * nb2);
    dst[i0 + ne0 * (i1 + ne1 * i2)] = row[i0];
  }
}

// I32 copy never passes through floating point, including NaN bit patterns.
// The validated source strides address logical elements; the result is packed.
__global__ void CopyI32ContKernel(const char* __restrict__ src, std::int32_t* __restrict__ dst,
                                  std::int64_t count, std::int64_t ne0, std::int64_t ne1,
                                  std::int64_t ne2, std::size_t nb0, std::size_t nb1,
                                  std::size_t nb2, std::size_t nb3) {
  const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  auto rest = index;
  const auto i0 = rest % ne0;
  rest /= ne0;
  const auto i1 = rest % ne1;
  rest /= ne1;
  const auto i2 = rest % ne2;
  const auto i3 = rest / ne2;
  const auto offset = static_cast<std::size_t>(i0) * nb0 + static_cast<std::size_t>(i1) * nb1 +
                      static_cast<std::size_t>(i2) * nb2 + static_cast<std::size_t>(i3) * nb3;
  dst[index] = *reinterpret_cast<const std::int32_t*>(src + offset);
}

__device__ float ToF32(float v) { return v; }
__device__ float ToF32(half v) { return __half2float(v); }
template <typename T>
__device__ T FromF32(float v);
template <>
__device__ float FromF32<float>(float v) {
  return v;
}
template <>
__device__ half FromF32<half>(float v) {
  return __float2half(v);
}

template <typename SrcT, typename DstT, bool kSetRows>
__global__ void Llama3RopeKernel(const char* x_bytes, const std::int32_t* pos_data,
                                 const float* freq_factors, const std::int64_t* row_ids,
                                 char* dst_bytes, std::int64_t ne0, std::int64_t ne1,
                                 std::int64_t ne2, std::size_t x_nb1, std::size_t x_nb2,
                                 std::size_t x_nb3, std::size_t dst_nb1, std::size_t dst_nb2,
                                 std::size_t dst_nb3, int n_dims, int n_offs, int n_ctx_orig,
                                 float freq_base, float freq_scale, float attn_factor,
                                 float beta_fast, float beta_slow, bool neox, bool inverse) {
  const std::int64_t flat = blockIdx.x;
  const std::int64_t i1 = flat % ne1;
  const std::int64_t rem = flat / ne1;
  const std::int64_t i2 = rem % ne2;
  const std::int64_t i3 = rem / ne2;
  const auto* x_row = reinterpret_cast<const SrcT*>(x_bytes + static_cast<std::size_t>(i1) * x_nb1 +
                                                    static_cast<std::size_t>(i2) * x_nb2 +
                                                    static_cast<std::size_t>(i3) * x_nb3);
  DstT* dst_row = nullptr;
  if constexpr (kSetRows) {
    dst_row = reinterpret_cast<DstT*>(dst_bytes + static_cast<std::size_t>(row_ids[i2]) * dst_nb1) +
              i1 * ne0;
  } else {
    dst_row = reinterpret_cast<DstT*>(dst_bytes + static_cast<std::size_t>(i1) * dst_nb1 +
                                      static_cast<std::size_t>(i2) * dst_nb2 +
                                      static_cast<std::size_t>(i3) * dst_nb3);
  }
  for (int c = static_cast<int>(threadIdx.x); c < static_cast<int>(ne0);
       c += static_cast<int>(blockDim.x)) {
    if (c < n_offs || c >= n_offs + n_dims) {
      dst_row[c] = FromF32<DstT>(ToF32(x_row[c]));
    }
  }
  const int pairs = n_dims / 2;
  const float pos = static_cast<float>(pos_data[i2]);
  const float old_ctx = static_cast<float>(n_ctx_orig);
  const float low_freq_wavelen = old_ctx / beta_slow;
  const float high_freq_wavelen = old_ctx / beta_fast;
  const float mscale = attn_factor != 0.0f ? attn_factor : 1.0f;
  for (int pr = static_cast<int>(threadIdx.x); pr < pairs; pr += static_cast<int>(blockDim.x)) {
    const int idx0 = neox ? (n_offs + pr) : (n_offs + 2 * pr);
    const int idx1 = neox ? (idx0 + pairs) : (idx0 + 1);
    const float base_theta =
        powf(freq_base, (2.0f * static_cast<float>(pr)) / static_cast<float>(n_dims));
    const float freq = 1.0f / base_theta;
    const float wavelen = 6.28318530717958647692f / freq;
    float scaled_freq = freq;
    if (wavelen > low_freq_wavelen) {
      scaled_freq = freq * freq_scale;
    } else if (wavelen > high_freq_wavelen && beta_fast > beta_slow) {
      const float smooth = (old_ctx / wavelen - beta_slow) / (beta_fast - beta_slow);
      scaled_freq = (1.0f - smooth) * freq * freq_scale + smooth * freq;
    }
    const float ff = freq_factors != nullptr ? freq_factors[pr] : 1.0f;
    const float theta = (pos * scaled_freq) / ff;
    const float c_val = cosf(theta) * mscale;
    const float s_val = (inverse ? -sinf(theta) : sinf(theta)) * mscale;
    const float x0 = ToF32(x_row[idx0]);
    const float x1 = ToF32(x_row[idx1]);
    dst_row[idx0] = FromF32<DstT>(x0 * c_val - x1 * s_val);
    dst_row[idx1] = FromF32<DstT>(x0 * s_val + x1 * c_val);
  }
}

bool IsLlama3Rope(const ggml_tensor* rope) {
  float ext_factor = 0.0f;
  std::memcpy(&ext_factor, rope->op_params + 7, sizeof(float));
  return ext_factor < 0.0f && rope->op_params[4] > 0;
}

void LaunchLlama3Rope(ggml_backend_cuda_context& context, const ggml_tensor* rope,
                      ggml_tensor* set_rows) {
  const ggml_tensor* x = rope->src[0];
  const auto* pos = static_cast<const std::int32_t*>(rope->src[1]->data);
  const auto* ff =
      rope->src[2] != nullptr ? static_cast<const float*>(rope->src[2]->data) : nullptr;
  const int n_dims = rope->op_params[1];
  const int mode = rope->op_params[2];
  const int n_offs = rope->op_params[15];
  const int n_ctx_orig = rope->op_params[4];
  float freq_base = 0.0f, freq_scale = 0.0f, attn_factor = 0.0f, beta_fast = 0.0f, beta_slow = 0.0f;
  std::memcpy(&freq_base, rope->op_params + 5, sizeof(float));
  std::memcpy(&freq_scale, rope->op_params + 6, sizeof(float));
  std::memcpy(&attn_factor, rope->op_params + 8, sizeof(float));
  std::memcpy(&beta_fast, rope->op_params + 9, sizeof(float));
  std::memcpy(&beta_slow, rope->op_params + 10, sizeof(float));
  const bool neox = (mode & GGML_ROPE_TYPE_NEOX) != 0;
  const bool inverse = rope->op == GGML_OP_ROPE_BACK;
  const auto blocks = static_cast<unsigned>(x->ne[1] * x->ne[2] * x->ne[3]);
  if (blocks == 0) return;
  cudaStream_t stream = context.stream();
  if (set_rows != nullptr) {
    const auto* ids = static_cast<const std::int64_t*>(set_rows->src[1]->data);
    Llama3RopeKernel<float, half, true><<<blocks, 128, 0, stream>>>(
        static_cast<const char*>(x->data), pos, ff, ids, static_cast<char*>(set_rows->data),
        x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], x->nb[3], set_rows->nb[1], 0, 0, n_dims,
        n_offs, n_ctx_orig, freq_base, freq_scale, attn_factor, beta_fast, beta_slow, neox,
        inverse);
  } else if (x->type == GGML_TYPE_F16) {
    Llama3RopeKernel<half, half, false><<<blocks, 128, 0, stream>>>(
        static_cast<const char*>(x->data), pos, ff, nullptr, static_cast<char*>(rope->data),
        x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], x->nb[3], rope->nb[1], rope->nb[2],
        rope->nb[3], n_dims, n_offs, n_ctx_orig, freq_base, freq_scale, attn_factor, beta_fast,
        beta_slow, neox, inverse);
  } else {
    Llama3RopeKernel<float, float, false><<<blocks, 128, 0, stream>>>(
        static_cast<const char*>(x->data), pos, ff, nullptr, static_cast<char*>(rope->data),
        x->ne[0], x->ne[1], x->ne[2], x->nb[1], x->nb[2], x->nb[3], rope->nb[1], rope->nb[2],
        rope->nb[3], n_dims, n_offs, n_ctx_orig, freq_base, freq_scale, attn_factor, beta_fast,
        beta_slow, neox, inverse);
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

std::expected<void, KernelFailure> LayerNorm(LaunchContext& launch, ggml_tensor* norm) {
  if (auto checked = CheckLayerNorm(norm); !checked) return checked;
  return launch.Run(base::Bytes(0), [norm](ggml_backend_cuda_context& context) {
    ggml_cuda_op_norm(context, norm);
  });
}

std::expected<void, KernelFailure> L2Norm(LaunchContext& launch, ggml_tensor* norm) {
  if (auto checked = CheckL2Norm(norm); !checked) return checked;
  return launch.Run(base::Bytes(0), [norm](ggml_backend_cuda_context& context) {
    ggml_cuda_op_l2_norm(context, norm);
  });
}

std::expected<void, KernelFailure> RmsNorm(LaunchContext& launch, ggml_tensor* norm) {
  if (auto checked = CheckRmsNorm(norm); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [norm](ggml_backend_cuda_context& context) {
    ggml_cuda_op_rms_norm(context, norm);
  });
}

std::expected<void, KernelFailure> RmsNormMul(LaunchContext& launch, ggml_tensor* norm,
                                              ggml_tensor* mul) {
  if (auto checked = CheckRmsNormMul(norm, mul); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [norm, mul](ggml_backend_cuda_context& context) {
    ggml_cuda_op_rms_norm_fused(context, norm, mul);
  });
}

std::expected<void, KernelFailure> RmsNormThenMul(LaunchContext& launch, ggml_tensor* norm,
                                                  ggml_tensor* mul) {
  if (auto checked = CheckRmsNormThenMul(norm, mul); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [norm, mul](ggml_backend_cuda_context& context) {
    ggml_cuda_op_rms_norm(context, norm);
    ggml_cuda_op_mul(context, mul);
  });
}

std::expected<void, KernelFailure> Add(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckBinary(node, GGML_OP_ADD); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_tensor dst = *node, s0 = *node->src[0], s1 = *node->src[1];
    ShiftBinaryLeft(dst, s0, s1);
    ggml_cuda_op_add(context, &dst);
  });
}

std::expected<void, KernelFailure> Mul(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckBinary(node, GGML_OP_MUL); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_tensor dst = *node, s0 = *node->src[0], s1 = *node->src[1];
    ShiftBinaryLeft(dst, s0, s1);
    ggml_cuda_op_mul(context, &dst);
  });
}

std::expected<void, KernelFailure> MulMatVecF(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMulMat(node); !checked) {
    return checked;
  }
  const ggml_tensor* weights = node->src[0];
  if (!ggml_cuda_should_use_mmvf(weights->type, Device(launch).cc, Device(launch).warp_size,
                                 weights->ne, weights->nb, node->src[1]->ne[1])) {
    return Rejected("upstream does not select MMVF for these operands");
  }
  // The launcher's even column-stride assertion holds: CheckMulMat requires
  // every activation stride to be a multiple of 8 bytes.
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_mul_mat_vec_f(context, node->src[0], node->src[1], nullptr, node);
  });
}

std::expected<void, KernelFailure> MulMatF(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMulMatF(node); !checked) {
    return checked;
  }
  const ggml_tensor* weights = node->src[0];
  const auto& device = Device(launch);
  if (!ggml_cuda_should_use_mmf(weights->type, device.cc, device.warp_size, weights->ne,
                                weights->nb, static_cast<int>(node->src[1]->ne[1]),
                                /*mul_mat_id=*/false)) {
    return Rejected("upstream does not select MMF for these operands");
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_mul_mat_f(context, node->src[0], node->src[1], nullptr, node);
  });
}

std::expected<MulMatPath, KernelFailure> SelectMulMat(const LaunchContext& launch,
                                                      const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_MUL_MAT || node->src[0] == nullptr ||
      node->src[1] == nullptr) {
    return Rejected("not a ggml_mul_mat node");
  }
  const ggml_tensor* src0 = node->src[0];
  const ggml_tensor* src1 = node->src[1];
  // Only a quantized tensor is padded in a CUDA buffer, so upstream's
  // bad_padding_clear never holds for the others; quantized weights take
  // MMVQ or MMQ, which ops_ext.h's SelectMulMatQ chooses between.
  if (ggml_is_quantized(src0->type)) {
    return Rejected("quantized weights take MMVQ or MMQ: ops_ext.h SelectMulMatQ");
  }
  if (src1->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) {
    return MulMatPath::kCublas;
  }
  const auto& device = Device(launch);
  const std::int64_t ne11 = src1->ne[1];
  if (ggml_cuda_should_use_mmvf(src0->type, device.cc, device.warp_size, src0->ne, src0->nb,
                                ne11)) {
    return MulMatPath::kVector;
  }
  if (src0->ne[1] == 1 && ne11 > MMVF_MAX_BATCH_SIZE && node->ne[2] == 1 && node->ne[3] == 1 &&
      src0->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
      ggml_is_contiguous(node) &&
      ggml_cuda_should_use_mmvf(src1->type, device.cc, device.warp_size, src1->ne, src1->nb,
                                /*ne11=*/1)) {
    return Rejected("upstream takes the transposed vector product, which is not implemented");
  }
  if (ggml_cuda_should_use_mmf(src0->type, device.cc, device.warp_size, src0->ne, src0->nb,
                               static_cast<int>(ne11), /*mul_mat_id=*/false)) {
    return MulMatPath::kTensorCore;
  }
  return MulMatPath::kCublas;
}

bool MulMatVecFusible(const LaunchContext& launch, const ggml_tensor* mul_mat) {
  if (mul_mat == nullptr || mul_mat->op != GGML_OP_MUL_MAT || mul_mat->src[0] == nullptr ||
      mul_mat->src[1] == nullptr) {
    return false;
  }
  const ggml_tensor* weights = mul_mat->src[0];
  const bool types = (weights->type == GGML_TYPE_F32 || weights->type == GGML_TYPE_F16 ||
                      weights->type == GGML_TYPE_BF16) &&
                     mul_mat->src[1]->type == GGML_TYPE_F32 && mul_mat->type == GGML_TYPE_F32;
  return types && mul_mat->ne[1] == 1 &&
         ggml_cuda_should_use_mmvf(weights->type, Device(launch).cc, Device(launch).warp_size,
                                   weights->ne, weights->nb, mul_mat->src[1]->ne[1]);
}

std::expected<void, KernelFailure> GetRows(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGetRows(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_get_rows(context, node);
  });
}

std::expected<void, KernelFailure> SetRows(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckSetRows(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_set_rows(context, node);
  });
}

std::expected<void, KernelFailure> Rope(LaunchContext& launch, ggml_tensor* rope) {
  if (auto checked = CheckRope(rope); !checked) {
    return checked;
  }
  if (rope->op_params[11] != 0) return RopeExt(launch, rope);
  return launch.Run(base::Bytes(0), [rope](ggml_backend_cuda_context& context) {
    if (IsLlama3Rope(rope)) {
      LaunchLlama3Rope(context, rope, nullptr);
      return;
    }
    ggml_cuda_op_rope(context, rope);
  });
}

std::expected<void, KernelFailure> RopeSetRows(LaunchContext& launch, ggml_tensor* rope,
                                               ggml_tensor* set_rows) {
  if (auto checked = CheckRopeSetRows(rope, set_rows); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [rope, set_rows](ggml_backend_cuda_context& context) {
    if (IsLlama3Rope(rope)) {
      LaunchLlama3Rope(context, rope, set_rows);
      return;
    }
    ggml_cuda_op_rope_fused(context, rope, set_rows);
  });
}

std::expected<void, KernelFailure> SoftMax(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckSoftMax(node); !checked) {
    return checked;
  }
  // Beyond the opt-in limit upstream takes other kernels (softmax.cu:349-375),
  // one of them drawing scratch; none is implemented here.
  if (SoftMaxSharedBytes(node) > Device(launch).smpbo) {
    return Rejected("a soft_max row larger than the device's shared memory");
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_soft_max(context, node);
  });
}

std::expected<void, KernelFailure> Cont(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckCont(node); !checked) {
    return std::unexpected(checked.error());
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    const ggml_tensor* src = node->src[0];
    if (src->type == GGML_TYPE_I32) {
      if (ggml_is_contiguous(src)) {
        CUDA_CHECK(cudaMemcpyAsync(node->data, src->data, ggml_nbytes(node),
                                   cudaMemcpyDeviceToDevice, context.stream()));
      } else {
        const auto count = ggml_nelements(src);
        const auto blocks = static_cast<unsigned>((count + 63) / 64);
        CopyI32ContKernel<<<blocks, 64, 0, context.stream()>>>(
            static_cast<const char*>(src->data), static_cast<std::int32_t*>(node->data), count,
            src->ne[0], src->ne[1], src->ne[2], src->nb[0], src->nb[1], src->nb[2], src->nb[3]);
        CUDA_CHECK(cudaGetLastError());
      }
      return;
    }
    if (src->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 && ggml_is_contiguous(node)) {
      const CollapsedCont c = CollapseF32Cont(src);
      cudaStream_t stream = context.stream();
      if (c.dims == 1 && c.nb[0] == sizeof(float)) {
        CUDA_CHECK(cudaMemcpyAsync(node->data, src->data, ggml_nbytes(node),
                                   cudaMemcpyDeviceToDevice, stream));
        return;
      }
      // cudaMemcpy2DAsync refuses a source pitch below the row's width, so a
      // view of overlapping rows takes a kernel below.
      if (c.dims == 2 && c.nb[0] == sizeof(float) &&
          c.nb[1] >= static_cast<std::size_t>(c.ne[0]) * sizeof(float)) {
        const std::size_t width = static_cast<std::size_t>(c.ne[0]) * sizeof(float);
        CUDA_CHECK(cudaMemcpy2DAsync(node->data, width, src->data, c.nb[1], width,
                                     static_cast<std::size_t>(c.ne[1]), cudaMemcpyDeviceToDevice,
                                     stream));
        return;
      }
      if ((c.dims == 2 || (c.dims == 3 && c.ne[2] <= 65535)) && c.nb[1] == sizeof(float) &&
          ((c.ne[0] + 31) / 32) <= 65535 && ((c.ne[1] + 31) / 32) <= 65535) {
        const dim3 grid(static_cast<unsigned>((c.ne[1] + 31) / 32),
                        static_cast<unsigned>((c.ne[0] + 31) / 32),
                        static_cast<unsigned>(c.dims == 3 ? c.ne[2] : 1));
        Transpose2dKernel<<<grid, dim3(32, 8), 0, stream>>>(
            static_cast<const char*>(src->data), static_cast<float*>(node->data), c.ne[0], c.ne[1],
            c.nb[0], c.dims == 3 ? c.nb[2] : 0);
        CUDA_CHECK(cudaGetLastError());
        return;
      }
      if (c.dims == 3 && c.nb[0] == sizeof(float) && c.ne[1] <= 65535 && c.ne[2] <= 65535) {
        const dim3 grid(static_cast<unsigned>((c.ne[0] + 255) / 256),
                        static_cast<unsigned>(c.ne[1]), static_cast<unsigned>(c.ne[2]));
        Copy3dContigDim0Kernel<<<grid, 256, 0, stream>>>(static_cast<const char*>(src->data),
                                                         static_cast<float*>(node->data), c.ne[0],
                                                         c.ne[1], c.nb[1], c.nb[2]);
        CUDA_CHECK(cudaGetLastError());
        return;
      }
    }
    ggml_cuda_dup(context, node);
  });
}

std::expected<void, KernelFailure> Convert(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckConvert(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_cpy(context, node->src[0], node->src[1]);
  });
}

std::expected<void, KernelFailure> SwiGlu(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckSwiGlu(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_tensor dst = *node;
    ggml_tensor s0 = *node->src[0];
    ggml_tensor s1 = *node->src[1];
    ShiftBinaryLeft(dst, s0, s1);
    ggml_cuda_op_swiglu(context, &dst);
  });
}

std::expected<void, KernelFailure> GeGlu(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckGeGlu(node); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_tensor dst = *node;
    ggml_tensor s0 = *node->src[0];
    ggml_tensor s1 = *node->src[1];
    ShiftBinaryLeft(dst, s0, s1);
    ggml_cuda_op_geglu(context, &dst);
  });
}

std::expected<void, KernelFailure> MulMatVecBias(LaunchContext& launch, ggml_tensor* mul_mat,
                                                 ggml_tensor* add) {
  if (auto checked = CheckMulMatVecBias(mul_mat, add); !checked) {
    return checked;
  }
  if (!MulMatVecFusible(launch, mul_mat)) {
    return Rejected("upstream does not select MMVF for this product");
  }
  const ggml_tensor* bias = add->src[0] == mul_mat ? add->src[1] : add->src[0];
  // As ggml_cuda_try_fuse calls it (ggml-cuda.cu:4113-4118): the add is the
  // node written, and its parameters give the accumulation precision.
  return launch.Run(base::Bytes(0), [mul_mat, add, bias](ggml_backend_cuda_context& context) {
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.x_bias = bias;
    ggml_cuda_mul_mat_vec_f(context, mul_mat->src[0], mul_mat->src[1], nullptr, add, &fusion);
  });
}

std::expected<void, KernelFailure> MulMatVecGlu(LaunchContext& launch, ggml_tensor* gate,
                                                ggml_tensor* up, ggml_tensor* glu) {
  if (auto checked = CheckMulMatVecGlu(gate, up, glu); !checked) {
    return checked;
  }
  if (!MulMatVecFusible(launch, up)) {
    return Rejected("upstream does not select MMVF for the up product");
  }
  // As ggml_cuda_try_fuse calls it (ggml-cuda.cu:3950-3957): the up
  // product's operands, the gate's weights, and the GLU as the node written,
  // whose first parameter (its GLU operation) the launcher reads as the
  // precision.
  return launch.Run(base::Bytes(0), [gate, up, glu](ggml_backend_cuda_context& context) {
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.gate = gate->src[0];
    fusion.glu_op = ggml_get_glu_op(glu);
    float limit = 0.0f;
    std::memcpy(&limit, &glu->op_params[3], sizeof(limit));
    fusion.glu_limit = limit;
    ggml_cuda_mul_mat_vec_f(context, up->src[0], up->src[1], nullptr, glu, &fusion);
  });
}

std::expected<void, KernelFailure> MulMatVecGeGlu(LaunchContext& launch, ggml_tensor* gate,
                                                  ggml_tensor* up, ggml_tensor* glu) {
  if (auto checked = CheckMulMatVecGeGlu(gate, up, glu); !checked) {
    return checked;
  }
  if (!MulMatVecFusible(launch, up)) {
    return Rejected("upstream does not select MMVF for the up product");
  }
  // As ggml_cuda_try_fuse calls it (ggml-cuda.cu:3950-3957): the up
  // product's operands, the gate's weights, and the GLU as the node written,
  // whose first parameter (its GLU operation) the launcher reads as the
  // precision.
  return launch.Run(base::Bytes(0), [gate, up, glu](ggml_backend_cuda_context& context) {
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.gate = gate->src[0];
    fusion.glu_op = ggml_get_glu_op(glu);
    float limit = 0.0f;
    std::memcpy(&limit, &glu->op_params[3], sizeof(limit));
    fusion.glu_limit = limit;
    ggml_cuda_mul_mat_vec_f(context, up->src[0], up->src[1], nullptr, glu, &fusion);
  });
}

}  // namespace llmp::kernels::ggml
