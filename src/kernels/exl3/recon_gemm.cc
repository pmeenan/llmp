// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/exl3/recon_gemm.h"

#include <cublasLt.h>
#include <cublas_api.h>
#include <cuda_runtime_api.h>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "kernels/exl3/launch.h"
#include "kernels/exl3/validate.h"

namespace llmp::kernels::exl3 {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

// The cuBLASLt this build was compiled against: the SDK's pinned 13.8.0.4
// (D-076), which reports major * 10000 + minor * 100 + patch.
constexpr std::size_t kPinned = CUBLAS_VERSION;

// cuBLAS's switches that change which kernels run or how they round;
// logging and NVTX ranges change neither (kernels/ggml/cublas.cc).
bool NumericsSwitch(std::string_view name) {
  for (const std::string_view logging :
       {"CUBLAS_LOGINFO_DBG", "CUBLAS_LOGDEST_DBG", "CUBLASLT_LOG_LEVEL", "CUBLASLT_LOG_FILE",
        "CUBLASLT_LOG_MASK", "CUBLAS_NVTX_LEVEL", "CUBLASLT_NVTX_LEVEL"}) {
    if (name == logging) {
      return false;
    }
  }
  return name.starts_with("CUBLAS") || name == "NVIDIA_TF32_OVERRIDE";
}

constexpr std::array<cublasLtMatmulAlgoConfigAttributes_t, 9> kAttributes{
    CUBLASLT_ALGO_CONFIG_ID,
    CUBLASLT_ALGO_CONFIG_TILE_ID,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
    CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
    CUBLASLT_ALGO_CONFIG_STAGES_ID,
    CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
    CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};

// Descriptors of one GEMM, destroyed with it.
struct Descriptors {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t d = nullptr;

  Descriptors() = default;
  Descriptors(const Descriptors&) = delete;
  Descriptors& operator=(const Descriptors&) = delete;
  Descriptors(Descriptors&&) = delete;
  Descriptors& operator=(Descriptors&&) = delete;
  ~Descriptors() {
    if (d != nullptr) {
      (void)cublasLtMatrixLayoutDestroy(d);
    }
    if (b != nullptr) {
      (void)cublasLtMatrixLayoutDestroy(b);
    }
    if (a != nullptr) {
      (void)cublasLtMatrixLayoutDestroy(a);
    }
    if (operation != nullptr) {
      (void)cublasLtMatmulDescDestroy(operation);
    }
  }
};

std::string Status(cublasStatus_t status) { return cublasLtGetStatusString(status); }

}  // namespace

std::expected<std::unique_ptr<ReconGemm>, KernelFailure> ReconGemm::Create() {
  if (const std::size_t loaded = cublasLtGetVersion(); loaded != kPinned) {
    return Rejected(std::format("cuBLASLt {} is loaded; this build needs {}", loaded, kPinned));
  }
  for (char** entry = environ; *entry != nullptr; ++entry) {
    const std::string_view variable(*entry);
    const std::string_view name = variable.substr(0, variable.find('='));
    if (NumericsSwitch(name)) {
      return Rejected(std::format("{} is set, which changes how cuBLAS computes", name));
    }
  }
  cublasLtHandle_t handle = nullptr;
  if (const cublasStatus_t status = cublasLtCreate(&handle); status != CUBLAS_STATUS_SUCCESS) {
    return Rejected("cublasLtCreate failed: " + Status(status));
  }
  return std::unique_ptr<ReconGemm>(new ReconGemm(handle));
}

ReconGemm::~ReconGemm() { (void)cublasLtDestroy(handle_); }

// The GEMM's descriptors and its pinned algorithm, checked by cuBLASLt.
struct ReconGemm::Prepared {
  Descriptors descriptors;
  cublasLtMatmulAlgo_t algo{};
};

std::expected<std::unique_ptr<ReconGemm::Prepared>, KernelFailure> ReconGemm::Prepare(
    const ReconGemmOperands& o, const LtAlgorithm& algorithm, int sm_count) const {
  if (auto checked = CheckReconGemm(o); !checked) {
    return std::unexpected(checked.error());
  }
  if (sm_count < 1) {
    return Rejected(std::format("{} SMs targeted", sm_count));
  }
  // A pin runs only the GEMM it was pinned for (validate.h, LtAlgorithm).
  if (algorithm.m != o.m || algorithm.k != o.k || algorithm.n != o.n || algorithm.ldc != o.ldc ||
      algorithm.output != o.output) {
    return Rejected(std::format(
        "the algorithm was pinned for m = {}, k = {}, n = {}, ldc = {} ({}), not m = {}, k = {}, "
        "n = {}, ldc = {} ({}): each GEMM runs its own pin",
        algorithm.m, algorithm.k, algorithm.n, algorithm.ldc,
        algorithm.output == Output::kF32 ? "HSS" : "HSH", o.m, o.k, o.n, o.ldc,
        o.output == Output::kF32 ? "HSS" : "HSH"));
  }
  auto prepared = std::make_unique<Prepared>();
  Descriptors& descriptors = prepared->descriptors;
  const cudaDataType_t out_type = o.output == Output::kF32 ? CUDA_R_32F : CUDA_R_16F;
  cublasStatus_t status =
      cublasLtMatmulDescCreate(&descriptors.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  const std::int32_t sms = sm_count;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(descriptors.operation,
                                            CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &sms, sizeof sms);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(&descriptors.a, CUDA_R_16F, static_cast<std::uint64_t>(o.n),
                                        static_cast<std::uint64_t>(o.k), o.n);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(&descriptors.b, CUDA_R_16F, static_cast<std::uint64_t>(o.k),
                                        static_cast<std::uint64_t>(o.m), o.k);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(&descriptors.d, out_type, static_cast<std::uint64_t>(o.n),
                                        static_cast<std::uint64_t>(o.m), o.ldc);
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    return Rejected("describing the GEMM failed: " + Status(status));
  }
  // The algorithm from its configuration alone (cublaslt_pin_probe.cc):
  // initialized by its ID, then every other attribute set at the size
  // cuBLASLt reports for it.
  const std::uint64_t algo_id = algorithm.config[0];
  if (algo_id > INT32_MAX) {
    return Rejected(std::format("algorithm {}", algo_id));
  }
  cublasLtMatmulAlgo_t& algo = prepared->algo;
  status = cublasLtMatmulAlgoInit(handle_, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F,
                                  out_type, out_type, static_cast<int>(algo_id), &algo);
  for (std::size_t i = 1; status == CUBLAS_STATUS_SUCCESS && i < kAttributes.size(); ++i) {
    std::size_t size = 0;
    status = cublasLtMatmulAlgoConfigGetAttribute(&algo, kAttributes.at(i), nullptr, 0, &size);
    if (status != CUBLAS_STATUS_SUCCESS || size == 0 || size > sizeof(std::uint64_t)) {
      return Rejected(std::format("cuBLASLt reports no size for algorithm attribute {}", i));
    }
    const std::uint64_t value = algorithm.config.at(i);
    if (size < sizeof(std::uint64_t) && (value >> (8 * size)) != 0) {
      return Rejected(
          std::format("algorithm attribute {} = {} does not fit {} bytes", i, value, size));
    }
    // Little-endian: the value's low bytes.
    status = cublasLtMatmulAlgoConfigSetAttribute(&algo, kAttributes.at(i), &value, size);
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    return Rejected("building the pinned algorithm failed: " + Status(status));
  }
  cublasLtMatmulHeuristicResult_t checked{};
  status = cublasLtMatmulAlgoCheck(handle_, descriptors.operation, descriptors.a, descriptors.b,
                                   descriptors.d, descriptors.d, &algo, &checked);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return Rejected("cuBLASLt refuses the pinned algorithm for this GEMM: " + Status(status));
  }
  if (checked.workspaceSize != 0) {
    return Rejected(std::format("the pinned algorithm needs {} bytes of workspace; none is given",
                                checked.workspaceSize));
  }
  return prepared;
}

std::expected<void, KernelFailure> ReconGemm::Check(const ReconGemmOperands& operands,
                                                    const LtAlgorithm& algorithm,
                                                    int sm_count) const {
  auto prepared = Prepare(operands, algorithm, sm_count);
  if (!prepared) {
    return std::unexpected(prepared.error());
  }
  return {};
}

std::expected<void, KernelFailure> ReconGemm::Run(LaunchContext& launch, const ReconGemmOperands& o,
                                                  const LtAlgorithm& algorithm, int sm_count) {
  auto prepared = Prepare(o, algorithm, sm_count);
  if (!prepared) {
    return std::unexpected(prepared.error());
  }
  const Descriptors& descriptors = (*prepared)->descriptors;
  const cublasLtMatmulAlgo_t& algo = (*prepared)->algo;
  const float alpha = 1.0F;
  const float beta = 0.0F;
  return launch.Run([&](void* stream) -> std::expected<void, KernelFailure> {
    // NOLINTBEGIN(performance-no-int-to-ptr): device addresses.
    const cublasStatus_t launched =
        cublasLtMatmul(handle_, descriptors.operation, &alpha, reinterpret_cast<const void*>(o.w),
                       descriptors.a, reinterpret_cast<const void*>(o.x), descriptors.b, &beta,
                       reinterpret_cast<void*>(o.y), descriptors.d, reinterpret_cast<void*>(o.y),
                       descriptors.d, &algo, nullptr, 0, static_cast<cudaStream_t>(stream));
    // NOLINTEND(performance-no-int-to-ptr)
    if (launched != CUBLAS_STATUS_SUCCESS) {
      return std::unexpected(KernelFailure{.error = KernelError::kUnknown,
                                           .detail = "cublasLtMatmul failed: " + Status(launched)});
    }
    return {};
  });
}

}  // namespace llmp::kernels::exl3
