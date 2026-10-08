// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// GGML's cuBLAS matrix multiplication as a llmpalooza implementation (D-053,
// D-077; docs/backend-proof.md#dispatch-and-implementations-d-053): a
// recorded copy of ggml_cuda_mul_mat_cublas, ggml_cuda_mul_mat_cublas_impl
// and k_compute_batched_ptrs, which ggml-cuda.cu at llama.cpp b29c606e2
// keeps static. What differs from upstream:
//
// - The compute type is upstream's automatic choice; the
//   GGML_CUDA_CUBLAS_COMPUTE_TYPE environment switch is not read.
// - Every decision the launcher makes (conversions, output precision, the
//   cuBLAS entry point and its leading dimensions) comes from one host plan
//   (validate.h's CublasMulMat), made before launch, so the scratch bound
//   and the executed calls cannot disagree, and what upstream would assert
//   on or cuBLAS would refuse is refused before anything is queued.
// - Only F32 activations (or F16 activations of F16 weights, read
//   directly) and outputs and F16, BF16 or F32 weights, and only operands
//   upstream's ggml_cuda_mul_mat would route to cuBLAS.
// - Where upstream aborts on a failed pointer-array launch, the copy skips
//   the GEMM that would read the arrays; the run faults (launch.h).
//
// k_compute_batched_ptrs is upstream's text at global scope, so that its
// mangled name and SASS compare with the bridge's (docs/backend-proof.md,
// Tier E); this file builds with the bridge's -use_fast_math.

#include <cublas_v2.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "base/check.h"
#include "common.cuh"
#include "convert.cuh"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/validate.h"
#include "mmf.cuh"
#include "mmvf.cuh"

// Upstream's kernel, unchanged.
// clang-format off
static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}
// clang-format on

namespace llmp::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

// Upstream's batched_mul_mat_traits: the CUDA and cuBLAS types of each
// compute type, its unit scalars and its conversions.
template <ggml_type T>
struct Traits;

template <>
struct Traits<GGML_TYPE_F32> {
  using cuda_type = float;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  static constexpr cudaDataType_t data_type = CUDA_R_32F;
  static const void* alpha() {
    static const float value = 1.0f;
    return &value;
  }
  static const void* beta() {
    static const float value = 0.0f;
    return &value;
  }
  static auto convert(ggml_type type) { return ggml_get_to_fp32_cuda(type); }
  static auto convert_nc(ggml_type type) { return ggml_get_to_fp32_nc_cuda(type); }
};

template <>
struct Traits<GGML_TYPE_BF16> {
  using cuda_type = nv_bfloat16;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  static constexpr cudaDataType_t data_type = CUDA_R_16BF;
  static const void* alpha() {
    static const float value = 1.0f;
    return &value;
  }
  static const void* beta() {
    static const float value = 0.0f;
    return &value;
  }
  static auto convert(ggml_type type) { return ggml_get_to_bf16_cuda(type); }
  static auto convert_nc(ggml_type type) { return ggml_get_to_bf16_nc_cuda(type); }
};

template <>
struct Traits<GGML_TYPE_F16> {
  using cuda_type = half;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  static constexpr cudaDataType_t data_type = CUDA_R_16F;
  static const void* alpha() {
    static const half value = 1.0;
    return &value;
  }
  static const void* beta() {
    static const half value = 0.0;
    return &value;
  }
  static auto convert(ggml_type type) { return ggml_get_to_fp16_cuda(type); }
  static auto convert_nc(ggml_type type) { return ggml_get_to_fp16_nc_cuda(type); }
};

// ggml_cuda_mul_mat_cublas's choice of compute type for non-quantized
// weights, without the environment switch.
ggml_type ComputeType(const ggml_tensor* node, int cc) {
  const ggml_tensor* src0 = node->src[0];
  ggml_type compute = src0->type;
  if (compute == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
    compute = GGML_TYPE_F32;
  } else if (compute == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
    if (GGML_CUDA_CC_IS_AMD(cc) && node->src[1]->ne[1] > 32) {
      compute = GGML_TYPE_F32;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && node->src[1]->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
      compute = GGML_TYPE_F32;
    }
  }
  if (node->op_params[0] == GGML_PREC_F32) {
    compute = GGML_TYPE_F32;
  }
  return compute;
}

// The impl's prefer_f32_output: whether cuBLAS writes F32 directly.
bool F32Output(ggml_type compute, int cc) {
  if (compute == GGML_TYPE_F16) {
    return cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
  }
  if (compute == GGML_TYPE_BF16) {
    return !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
  }
  return false;
}

// Whether ggml_cuda_mul_mat routes the node to cuBLAS: none of MMVF (either
// way round) or MMF takes it. MMVQ and MMQ take quantized weights only.
bool UpstreamSelectsCublas(const ggml_tensor* node, int cc, int warp_size) {
  const ggml_tensor* src0 = node->src[0];
  const ggml_tensor* src1 = node->src[1];
  const std::int64_t ne11 = src1->ne[1];
  if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11)) {
    return false;
  }
  if (src0->ne[1] == 1 && ne11 > MMVF_MAX_BATCH_SIZE && node->ne[2] == 1 && node->ne[3] == 1 &&
      src0->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
      ggml_is_contiguous(node) &&
      ggml_cuda_should_use_mmvf(src1->type, cc, warp_size, src1->ne, src1->nb, /*ne11=*/1)) {
    return false;
  }
  return !ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb,
                                   static_cast<int>(ne11), /*mul_mat_id=*/false);
}

// ggml_cuda_mul_mat_cublas_impl, following `plan`.
template <ggml_type kCompute>
void Launch(ggml_backend_cuda_context& ctx, const CublasMulMat& plan, const ggml_tensor* src0,
            const ggml_tensor* src1, ggml_tensor* dst) {
  using traits = Traits<kCompute>;
  using cuda_t = typename traits::cuda_type;

  const std::int64_t ne_dst = ggml_nelements(dst);
  cudaStream_t main_stream = ctx.stream();
  cublasHandle_t cublas_h = ctx.cublas_handle();

  // Pool blocks in upstream's order, returned as the launcher returns.
  ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
  ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());
  const auto operand = [main_stream](const ggml_tensor* src, CublasOperand how,
                                     ggml_cuda_pool_alloc<cuda_t>& alloc) -> const cuda_t* {
    if (how == CublasOperand::kDirect) {
      return static_cast<const cuda_t*>(src->data);
    }
    const std::int64_t elements = ggml_nelements(src);
    alloc.alloc(static_cast<std::size_t>(elements));
    if (how == CublasOperand::kConverted) {
      const auto convert = traits::convert(src->type);
      base::Check(convert != nullptr, "GGML converts every planned type");
      convert(src->data, alloc.get(), elements, main_stream);
    } else {
      const auto convert = traits::convert_nc(src->type);
      base::Check(convert != nullptr, "GGML converts every planned type");
      const auto size = static_cast<std::int64_t>(ggml_type_size(src->type));
      convert(src->data, alloc.get(), src->ne[0], src->ne[1], src->ne[2], src->ne[3],
              static_cast<std::int64_t>(src->nb[1]) / size,
              static_cast<std::int64_t>(src->nb[2]) / size,
              static_cast<std::int64_t>(src->nb[3]) / size, main_stream);
    }
    return alloc.get();
  };
  const cuda_t* src0_ptr = operand(src0, plan.weights, src0_alloc);
  const cuda_t* src1_ptr = operand(src1, plan.input, src1_alloc);

  ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
  char* dst_ptr = static_cast<char*>(dst->data);
  std::size_t nbd2 = dst->nb[2];
  std::size_t nbd3 = dst->nb[3];
  cublasComputeType_t cu_compute_type = traits::compute_type;
  cudaDataType_t cu_data_type = traits::data_type;
  const cudaDataType_t cu_data_type_a = traits::data_type;
  const cudaDataType_t cu_data_type_b = traits::data_type;
  const void* alpha = traits::alpha();
  const void* beta = traits::beta();
  if (plan.f32_output) {
    cu_compute_type = Traits<GGML_TYPE_F32>::compute_type;
    cu_data_type = Traits<GGML_TYPE_F32>::data_type;
    alpha = Traits<GGML_TYPE_F32>::alpha();
    beta = Traits<GGML_TYPE_F32>::beta();
  } else if constexpr (kCompute != GGML_TYPE_F32) {
    dst_ptr = reinterpret_cast<char*>(dst_temp.alloc(static_cast<std::size_t>(ne_dst)));
    nbd2 /= sizeof(float) / sizeof(cuda_t);
    nbd3 /= sizeof(float) / sizeof(cuda_t);
  }

  // cuBLAS takes its sizes as int: the plan checked that they fit.
  const int m = static_cast<int>(src0->ne[1]);
  const int n = static_cast<int>(src1->ne[1]);
  const int k = static_cast<int>(src0->ne[0]);
  const int lda = static_cast<int>(plan.s01);
  const int ldb = static_cast<int>(plan.s11);
  const int ldc = static_cast<int>(dst->ne[0]);
  const std::int64_t ne12 = src1->ne[2];
  const std::int64_t ne13 = src1->ne[3];
  switch (plan.gemm) {
    case CublasGemm::kSgemm:
      CUBLAS_CHECK(cublasSgemm(
          cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, static_cast<const float*>(alpha),
          reinterpret_cast<const float*>(src0_ptr), lda, reinterpret_cast<const float*>(src1_ptr),
          ldb, static_cast<const float*>(beta), reinterpret_cast<float*>(dst_ptr), ldc));
      break;
    case CublasGemm::kGemmEx:
      CUBLAS_CHECK(cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, src0_ptr,
                                cu_data_type_a, lda, src1_ptr, cu_data_type_b, ldb, beta, dst_ptr,
                                cu_data_type, ldc, cu_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      break;
    case CublasGemm::kGemmStridedBatchedEx: {
      // With a [0, 2, 1, 3] permutation and one channel, the matrix
      // strides come from dimension 3.
      const long long sma = src0->ne[2] == 1 ? plan.s03 : plan.s02;
      const long long smb = ne12 == 1 ? plan.s13 : plan.s12;
      CUBLAS_CHECK(cublasGemmStridedBatchedEx(
          cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, src0_ptr, cu_data_type_a, lda, sma,
          src1_ptr, cu_data_type_b, ldb, smb, beta, dst_ptr, cu_data_type, ldc,
          static_cast<long long>(dst->ne[1]) * dst->ne[0], static_cast<int>(ne12 * ne13),
          cu_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      break;
    }
    case CublasGemm::kGemmBatchedEx: {
      const std::int64_t ne23 = ne12 * ne13;
      ggml_cuda_pool_alloc<const void*> ptrs_src(ctx.pool(), static_cast<std::size_t>(2 * ne23));
      ggml_cuda_pool_alloc<void*> ptrs_dst(ctx.pool(), static_cast<std::size_t>(ne23));
      constexpr std::size_t src_type_size = sizeof(cuda_t);
      const dim3 block_dims(16, 16);
      const dim3 grid_dims(static_cast<unsigned>((ne13 + 15) / 16),
                           static_cast<unsigned>((ne12 + 15) / 16));
      k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
          src0_ptr, src1_ptr, dst_ptr, ptrs_src.get(), ptrs_dst.get(), ne12, ne13, ne23,
          static_cast<std::size_t>(plan.s02) * src_type_size,
          static_cast<std::size_t>(plan.s03) * src_type_size,
          static_cast<std::size_t>(plan.s12) * src_type_size,
          static_cast<std::size_t>(plan.s13) * src_type_size, nbd2, nbd3, ne12 / src0->ne[2],
          ne13 / src0->ne[3]);
      CUDA_CHECK(cudaGetLastError());
      if (internal::CudaErrorPending()) {
        // Upstream aborts here. The arrays may hold an earlier operation's
        // pointers, so the GEMM must not run; the run then faults.
        return;
      }
      CUBLAS_CHECK(cublasGemmBatchedEx(
          cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, ptrs_src.get(), cu_data_type_a, lda,
          ptrs_src.get() + ne23, cu_data_type_b, ldb, beta, ptrs_dst.get(), cu_data_type, ldc,
          static_cast<int>(ne23), cu_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      break;
    }
  }

  // The compute-type output, converted to the node's F32.
  if (cu_data_type != CUDA_R_32F) {
    const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(kCompute);
    to_fp32_cuda(dst_temp.get(), static_cast<float*>(dst->data), ne_dst, main_stream);
  }
}

}  // namespace

std::expected<CublasMulMat, KernelFailure> PlanMulMatCublas(const LaunchContext& launch,
                                                            const ggml_tensor* node) {
  if (auto checked = CheckMulMatCublasOperands(node); !checked) {
    return std::unexpected(checked.error());
  }
  // Read without a CUDA call: the table was read when the context was
  // created.
  const auto& device = ggml_cuda_info().devices[launch.device()];
  if (!UpstreamSelectsCublas(node, device.cc, device.warp_size)) {
    return Rejected("upstream does not select cuBLAS for these operands");
  }
  const ggml_type compute = ComputeType(node, device.cc);
  return CheckMulMatCublas(node, compute, F32Output(compute, device.cc));
}

std::expected<void, KernelFailure> MulMatCublas(LaunchContext& launch, ggml_tensor* node) {
  auto plan = PlanMulMatCublas(launch, node);
  if (!plan) {
    return std::unexpected(plan.error());
  }
  if (launch.cublas() == nullptr) {
    // GGML would create a handle and a workspace of its own.
    return Rejected("the launch context lends no cuBLAS handle");
  }
  // The launch writes both workspaces while it reads the operands.
  for (const LaunchContext::Workspace workspace :
       {launch.workspace(), launch.cublas()->workspace()}) {
    if (auto clear = CheckClearOf(node, workspace.base, workspace.size.value()); !clear) {
      return clear;
    }
  }
  return launch.Run(base::Bytes(plan->scratch),
                    [&plan = *plan, node](ggml_backend_cuda_context& context) {
                      const ggml_tensor* src0 = node->src[0];
                      const ggml_tensor* src1 = node->src[1];
                      switch (plan.compute) {
                        case GGML_TYPE_F16:
                          Launch<GGML_TYPE_F16>(context, plan, src0, src1, node);
                          break;
                        case GGML_TYPE_BF16:
                          Launch<GGML_TYPE_BF16>(context, plan, src0, src1, node);
                          break;
                        default:
                          Launch<GGML_TYPE_F32>(context, plan, src0, src1, node);
                          break;
                      }
                    });
}

}  // namespace llmp::kernels::ggml
