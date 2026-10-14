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

#include <cublasLt.h>
#include <cublas_v2.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/check.h"
#include "common.cuh"
#include "convert.cuh"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/mul_mat_cublas_internal.h"
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

struct LtPinShape {
  int m_rows = 0;  // n in column-major cuBLAS (src1->ne[1])
  int n_cols = 0;  // m in column-major cuBLAS (src0->ne[1])
  int k = 0;       // k in column-major cuBLAS (src0->ne[0])
  std::array<std::uint64_t, 9> config{};
};

constexpr std::array<LtPinShape, 12> kGb10LtPins{{
    {4096, 4096, 4096, {67, 463, 1, 0, 0, 30, 35, 0, 0}},
    {4096, 12288, 4096, {67, 463, 1, 0, 0, 30, 35, 0, 0}},
    {4096, 4096, 12288, {67, 412, 1, 0, 0, 31, 35, 0, 0}},
    {4121, 4096, 4096, {67, 463, 1, 0, 0, 30, 35, 0, 0}},
    {4121, 12288, 4096, {67, 463, 1, 0, 0, 30, 35, 0, 0}},
    {4121, 4096, 12288, {67, 412, 1, 0, 0, 31, 35, 0, 0}},
    {4096, 4096, 64, {67, 409, 1, 0, 0, 68, 35, 0, 0}},
    {4096, 64, 4096, {21, 11, 3, 1, 0, 0, 19, 0, 0}},
    {25, 4096, 4096, {21, 11, 6, 1, 0, 0, 19, 0, 0}},
    {39, 4096, 4096, {21, 11, 3, 1, 0, 0, 19, 0, 0}},
    {39, 1024, 4096, {67, 314, 1, 0, 0, 29, 36, 0, 0}},
    {39, 12288, 4096, {67, 30, 1, 0, 0, 23, 35, 0, 0}},
}};

constexpr std::array<cublasLtMatmulAlgoConfigAttributes_t, 9> kLtAttributes{
    CUBLASLT_ALGO_CONFIG_ID,
    CUBLASLT_ALGO_CONFIG_TILE_ID,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
    CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
    CUBLASLT_ALGO_CONFIG_STAGES_ID,
    CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
    CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};

struct LtCachedPin {
  int m = 0, n = 0, k = 0;
  cudaDataType_t d_type = CUDA_R_16BF;
  bool valid = false;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t d = nullptr;
  cublasLtMatmulAlgo_t algo{};
  std::size_t workspace = 0;

  LtCachedPin() = default;
  LtCachedPin(const LtCachedPin&) = delete;
  LtCachedPin& operator=(const LtCachedPin&) = delete;
  ~LtCachedPin() {
    if (d != nullptr) (void)cublasLtMatrixLayoutDestroy(d);
    if (b != nullptr) (void)cublasLtMatrixLayoutDestroy(b);
    if (a != nullptr) (void)cublasLtMatrixLayoutDestroy(a);
    if (op != nullptr) (void)cublasLtMatmulDescDestroy(op);
  }
};

bool TryLaunchPinnedLtGemm(cublasHandle_t cublas_h, cudaStream_t stream, int cc, int sm_count,
                           std::uintptr_t ws_base, std::size_t ws_bytes, int m, int n, int k,
                           int lda, int ldb, int ldc, const void* a_ptr, const void* b_ptr,
                           void* d_ptr, cudaDataType_t d_type) {
  if (cc != 1210 || sm_count != 48 || lda != k || ldb != k || ldc != m) return false;
  if (reinterpret_cast<std::uintptr_t>(a_ptr) % 256 != 0 ||
      reinterpret_cast<std::uintptr_t>(b_ptr) % 256 != 0 ||
      reinterpret_cast<std::uintptr_t>(d_ptr) % 256 != 0) {
    return false;
  }
  const LtPinShape* pin = nullptr;
  for (const LtPinShape& cand : kGb10LtPins) {
    if (cand.n_cols == m && cand.m_rows == n && cand.k == k) {
      pin = &cand;
      break;
    }
  }
  if (pin == nullptr) return false;

  thread_local std::vector<std::unique_ptr<LtCachedPin>> cache;
  auto* lt_h = reinterpret_cast<cublasLtHandle_t>(cublas_h);
  LtCachedPin* entry = nullptr;
  for (const auto& item : cache) {
    if (item->m == m && item->n == n && item->k == k && item->d_type == d_type) {
      entry = item.get();
      break;
    }
  }
  if (entry == nullptr) {
    if (cublasLtGetVersion() != 130800) return false;
    auto created = std::make_unique<LtCachedPin>();
    created->m = m;
    created->n = n;
    created->k = k;
    created->d_type = d_type;
    const cublasOperation_t transa = CUBLAS_OP_T;
    cublasStatus_t s = cublasLtMatmulDescCreate(&created->op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (s == CUBLAS_STATUS_SUCCESS) {
      s = cublasLtMatmulDescSetAttribute(created->op, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                                         sizeof(transa));
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
      s = cublasLtMatrixLayoutCreate(&created->a, CUDA_R_16BF, static_cast<std::uint64_t>(k),
                                     static_cast<std::uint64_t>(m), k);
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
      s = cublasLtMatrixLayoutCreate(&created->b, CUDA_R_16BF, static_cast<std::uint64_t>(k),
                                     static_cast<std::uint64_t>(n), k);
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
      s = cublasLtMatrixLayoutCreate(&created->d, d_type, static_cast<std::uint64_t>(m),
                                     static_cast<std::uint64_t>(n), m);
    }
    if (s == CUBLAS_STATUS_SUCCESS) {
      s = cublasLtMatmulAlgoInit(lt_h, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                                 d_type, d_type, static_cast<int>(pin->config[0]), &created->algo);
      for (std::size_t i = 1; s == CUBLAS_STATUS_SUCCESS && i < kLtAttributes.size(); ++i) {
        std::size_t sz = 0;
        s = cublasLtMatmulAlgoConfigGetAttribute(&created->algo, kLtAttributes[i], nullptr, 0, &sz);
        if (s != CUBLAS_STATUS_SUCCESS || sz == 0 || sz > sizeof(std::uint64_t)) {
          s = CUBLAS_STATUS_NOT_SUPPORTED;
          break;
        }
        const std::uint64_t val = pin->config[i];
        s = cublasLtMatmulAlgoConfigSetAttribute(&created->algo, kLtAttributes[i], &val, sz);
      }
      cublasLtMatmulHeuristicResult_t checked{};
      if (s == CUBLAS_STATUS_SUCCESS &&
          cublasLtMatmulAlgoCheck(lt_h, created->op, created->a, created->b, created->d, created->d,
                                  &created->algo, &checked) == CUBLAS_STATUS_SUCCESS &&
          checked.workspaceSize <= ws_bytes) {
        created->workspace = checked.workspaceSize;
        created->valid = true;
      }
    }
    cache.push_back(std::move(created));
    entry = cache.back().get();
  }
  if (!entry->valid || entry->workspace > ws_bytes) return false;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  void* ws_ptr = entry->workspace > 0 ? reinterpret_cast<void*>(ws_base) : nullptr;
  return cublasLtMatmul(lt_h, entry->op, &alpha, a_ptr, entry->a, b_ptr, entry->b, &beta, d_ptr,
                        entry->d, d_ptr, entry->d, &entry->algo, ws_ptr, entry->workspace,
                        stream) == CUBLAS_STATUS_SUCCESS;
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
            const ggml_tensor* src1, ggml_tensor* dst, bool reuse_converted_src1 = false,
            bool bf16_dst = false, bool src1_is_bf16 = false, bool allow_lt_pin = false,
            std::uintptr_t lt_ws_base = 0, std::size_t lt_ws_bytes = 0) {
  using traits = Traits<kCompute>;
  using cuda_t = typename traits::cuda_type;

  const std::int64_t ne_dst = ggml_nelements(dst);
  cudaStream_t main_stream = ctx.stream();
  cublasHandle_t cublas_h = ctx.cublas_handle();

  // Pool blocks in upstream's order, returned as the launcher returns.
  ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
  ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());
  const auto operand = [main_stream](const ggml_tensor* src, CublasOperand how,
                                     ggml_cuda_pool_alloc<cuda_t>& alloc,
                                     bool skip_convert) -> const cuda_t* {
    if (how == CublasOperand::kDirect) {
      return static_cast<const cuda_t*>(src->data);
    }
    const std::int64_t elements = ggml_nelements(src);
    alloc.alloc(static_cast<std::size_t>(elements));
    if (skip_convert) {
      return alloc.get();
    }
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
  const cuda_t* src0_ptr = operand(src0, plan.weights, src0_alloc, false);
  const cuda_t* src1_ptr =
      (kCompute == GGML_TYPE_BF16 && src1_is_bf16 && plan.weights == CublasOperand::kDirect &&
       plan.input == CublasOperand::kConverted)
          ? static_cast<const cuda_t*>(src1->data)
          : operand(src1, plan.input, src1_alloc,
                    reuse_converted_src1 && plan.weights == CublasOperand::kDirect &&
                        plan.input == CublasOperand::kConverted);

  const bool direct_bf16_out =
      (kCompute == GGML_TYPE_BF16 && bf16_dst && plan.gemm == CublasGemm::kGemmEx);
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
  if (!direct_bf16_out) {
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
    case CublasGemm::kGemmEx: {
      bool launched_lt = false;
      if constexpr (kCompute == GGML_TYPE_BF16) {
        if (allow_lt_pin) {
          const auto& dev = ggml_cuda_info().devices[ctx.device];
          launched_lt = TryLaunchPinnedLtGemm(cublas_h, main_stream, dev.cc, dev.nsm, lt_ws_base,
                                              lt_ws_bytes, m, n, k, lda, ldb, ldc, src0_ptr,
                                              src1_ptr, dst_ptr, cu_data_type);
        }
      }
      if (!launched_lt) {
        CUBLAS_CHECK(cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, src0_ptr,
                                  cu_data_type_a, lda, src1_ptr, cu_data_type_b, ldb, beta, dst_ptr,
                                  cu_data_type, ldc, cu_compute_type,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      break;
    }
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
  if (cu_data_type != CUDA_R_32F && !direct_bf16_out) {
    const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(kCompute);
    to_fp32_cuda(dst_temp.get(), static_cast<float*>(dst->data), ne_dst, main_stream);
  }
}

void LaunchPlannedWithReuse(ggml_backend_cuda_context& context, const CublasMulMat& plan,
                            ggml_tensor* node, bool reuse_converted_src1, bool bf16_dst = false,
                            bool src1_is_bf16 = false, bool allow_lt_pin = false,
                            std::uintptr_t lt_ws_base = 0, std::size_t lt_ws_bytes = 0) {
  switch (plan.compute) {
    case GGML_TYPE_F16:
      Launch<GGML_TYPE_F16>(context, plan, node->src[0], node->src[1], node, reuse_converted_src1);
      break;
    case GGML_TYPE_BF16:
      Launch<GGML_TYPE_BF16>(context, plan, node->src[0], node->src[1], node, reuse_converted_src1,
                             bf16_dst, src1_is_bf16, allow_lt_pin, lt_ws_base, lt_ws_bytes);
      break;
    default:
      Launch<GGML_TYPE_F32>(context, plan, node->src[0], node->src[1], node, false);
      break;
  }
}

}  // namespace

void internal::LaunchPlannedMulMatCublas(ggml_backend_cuda_context& context,
                                         const CublasMulMat& plan, ggml_tensor* node) {
  LaunchPlannedWithReuse(context, plan, node, false);
}

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
  return MulMatCublasWithReuse(launch, node, nullptr, false);
}

std::expected<void, KernelFailure> MulMatCublasWithReuse(LaunchContext& launch, ggml_tensor* node,
                                                         const ggml_tensor* src1_override,
                                                         bool reuse_converted_src1, bool bf16_dst,
                                                         bool src1_is_bf16, bool allow_lt_pin) {
  ggml_tensor adjusted = *node;
  if (src1_override != nullptr) {
    adjusted.src[1] = const_cast<ggml_tensor*>(src1_override);
  }
  ggml_tensor* target = src1_override != nullptr ? &adjusted : node;
  auto plan = PlanMulMatCublas(launch, target);
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
    if (auto clear = CheckClearOf(target, workspace.base, workspace.size.value()); !clear) {
      return clear;
    }
  }
  const auto lt_ws = launch.cublas()->workspace();
  return launch.Run(base::Bytes(plan->scratch), [&plan = *plan, target, reuse_converted_src1,
                                                 bf16_dst, src1_is_bf16, allow_lt_pin,
                                                 lt_ws](ggml_backend_cuda_context& context) {
    LaunchPlannedWithReuse(context, plan, target, reuse_converted_src1, bf16_dst, src1_is_bf16,
                           allow_lt_pin, lt_ws.base, lt_ws.size.value());
  });
}

}  // namespace llmp::kernels::ggml
