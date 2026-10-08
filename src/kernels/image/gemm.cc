// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/image/gemm.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace llmp::kernels::image {
namespace {

constexpr std::int64_t kIntMax = std::numeric_limits<int>::max();

bool Fits(std::int64_t v) { return v > 0 && v <= kIntMax; }

// Column-major C (m x n) = op(A) op(B), as cuBLAS sees row-major operands.
Status Gemm(cublasContext* handle, cublasOperation_t op_a, cublasOperation_t op_b, std::int64_t m,
            std::int64_t n, std::int64_t k, const void* a, std::int64_t lda, const void* b,
            std::int64_t ldb, void* c, cudaDataType_t c_type, std::int64_t ldc, float beta,
            const char* what) {
  if (handle == nullptr || !Fits(m) || !Fits(n) || !Fits(k) || !Fits(lda) || !Fits(ldb) ||
      !Fits(ldc)) {
    return std::unexpected(std::format("{}: sizes beyond cuBLAS's int", what));
  }
  const float alpha = 1.0f;
  const cublasStatus_t status = cublasGemmEx(
      handle, op_a, op_b, static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), &alpha, a,
      CUDA_R_16BF, static_cast<int>(lda), b, CUDA_R_16BF, static_cast<int>(ldb), &beta, c, c_type,
      static_cast<int>(ldc), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return std::unexpected(
        std::format("{}: cublasGemmEx failed ({})", what, static_cast<int>(status)));
  }
  return {};
}

}  // namespace

Status Linear(cublasContext* handle, const Bf16* x, std::int64_t ldx, const Bf16* w,
              std::int64_t ldw, Bf16* out, std::int64_t ldo, std::int64_t m, std::int64_t n,
              std::int64_t k, bool accumulate) {
  if (ldx < k || ldw < k || ldo < n) {
    return std::unexpected(std::string("Linear: strides"));
  }
  return Gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, w, ldw, x, ldx, out, CUDA_R_16BF, ldo,
              accumulate ? 1.0f : 0.0f, "Linear");
}

Status LinearF32(cublasContext* handle, const Bf16* x, std::int64_t ldx, const Bf16* w,
                 std::int64_t ldw, float* out, std::int64_t ldo, std::int64_t m, std::int64_t n,
                 std::int64_t k) {
  if (ldx < k || ldw < k || ldo < n) {
    return std::unexpected(std::string("LinearF32: strides"));
  }
  return Gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, w, ldw, x, ldx, out, CUDA_R_32F, ldo, 0.0f,
              "LinearF32");
}

Status ConvProduct(cublasContext* handle, const Bf16* w, std::int64_t out_channels,
                   std::int64_t inner, const Bf16* col, std::int64_t ldc, Bf16* out,
                   std::int64_t ldo, std::int64_t pixels, bool accumulate) {
  if (ldc < pixels || ldo < pixels) {
    return std::unexpected(std::string("ConvProduct: strides"));
  }
  // Column-major, col is A (lda = ldc) and out is C (ldc = ldo).
  // NOLINTNEXTLINE(readability-suspicious-call-argument)
  return Gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, pixels, out_channels, inner, col, ldc, w, inner,
              out, CUDA_R_16BF, ldo, accumulate ? 1.0f : 0.0f, "ConvProduct");
}

Status ScoresQtK(cublasContext* handle, const Bf16* q, const Bf16* k, float* scores,
                 std::int64_t channels, std::int64_t tokens, std::int64_t ld) {
  if (ld < tokens) {
    return std::unexpected(std::string("ScoresQtK: strides"));
  }
  return Gemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, tokens, tokens, channels, k, ld, q, ld, scores,
              CUDA_R_32F, tokens, 0.0f, "ScoresQtK");
}

Status ValuesTimesProbs(cublasContext* handle, const Bf16* v, const Bf16* probs, Bf16* out,
                        std::int64_t channels, std::int64_t tokens, std::int64_t ld) {
  if (ld < tokens) {
    return std::unexpected(std::string("ValuesTimesProbs: strides"));
  }
  return Gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, tokens, channels, tokens, probs, tokens, v, ld, out,
              CUDA_R_16BF, ld, 0.0f, "ValuesTimesProbs");
}

// ---------------------------------------------------------------- cuBLASLt

namespace {

// The pinned algorithms, tuned on `spark` (GB10, 48 SMs, driver 580.178.04)
// with cuBLASLt 13.8.0.4 by llmp_qwen_image_gemm_tune (sustained timing:
// each candidate's median over five batches of back-to-back launches after
// half a second of warm-up) at the pipeline's 1,024² shapes: for each, the
// fastest of cuBLASLt's heuristic candidates (32 MiB workspace) whose
// output equals cublasGemmEx's bit for bit
// (docs/experiments/qwen-image-native/README.md, "Speed"). Times in the
// comments: the pin's, then cublasGemmEx's, sustained, spark, 2026-09-28.
constexpr std::array<GemmPin, 16> kGb10Pins{{
    // The denoiser at image rows (1,024^2: 4,096) and at the first step's
    // joint rows (4,096 + 25 text rows).
    {4096, 4096, 4096, {67, 463, 1, 0, 0, 30, 35, 0, 0}},   // 1.480, 1.641 ms
    {4096, 12288, 4096, {67, 463, 1, 0, 0, 30, 35, 0, 0}},  // 4.205, 4.347 ms
    {4096, 4096, 12288, {67, 412, 1, 0, 0, 31, 35, 0, 0}},  // 4.527, 4.516 ms
    {4121, 4096, 4096, {67, 463, 1, 0, 0, 30, 35, 0, 0}},   // 1.499, 1.706 ms
    {4121, 12288, 4096, {67, 463, 1, 0, 0, 30, 35, 0, 0}},  // 4.250, 4.391 ms
    {4121, 4096, 12288, {67, 412, 1, 0, 0, 31, 35, 0, 0}},  // 4.516, 4.523 ms
    {4096, 4096, 64, {67, 409, 1, 0, 0, 68, 35, 0, 0}},     // img_in: 0.173, 0.196 ms
    {4096, 64, 4096, {21, 11, 3, 1, 0, 0, 19, 0, 0}},       // proj_out: 0.122, 0.122 ms
    {25, 4096, 4096, {21, 11, 6, 1, 0, 0, 19, 0, 0}},       // txt_in: 0.137, 0.137 ms
    {2, 4096, 256, {21, 5, 1, 0, 0, 0, 19, 0, 0}},          // the timestep's
    {2, 4096, 4096, {21, 5, 1, 0, 0, 0, 19, 0, 0}},         // 0.133, 0.133 ms
    {2, 16384, 4096, {21, 5, 1, 0, 0, 0, 20, 0, 0}},        // modulation: 0.574, 0.575 ms
    // The text encoder over the teapot prompt's 39 tokens.
    {39, 4096, 4096, {21, 11, 3, 1, 0, 0, 19, 0, 0}},     // 0.136, 0.136 ms
    {39, 1024, 4096, {67, 314, 1, 0, 0, 29, 36, 0, 0}},   // 0.015, 0.015 ms
    {39, 12288, 4096, {67, 30, 1, 0, 0, 23, 35, 0, 0}},   // 0.469, 0.472 ms
    {39, 4096, 12288, {67, 463, 4, 2, 0, 31, 35, 0, 0}},  // 0.487, 0.492 ms
}};

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

std::string LtStatus(cublasStatus_t status) { return cublasLtGetStatusString(status); }

bool Aligned256(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 256 == 0; }

}  // namespace

GemmPins PinnedGemms() {
  return {.compute_capability = 1210, .sm_count = 48, .cublaslt = 130800, .pins = kGb10Pins};
}

// One shape's descriptors and algorithm.
struct LtGemm::Prepared {
  std::int64_t ldx = 0, ldw = 0, ldo = 0, m = 0, n = 0, k = 0;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t d = nullptr;
  cublasLtMatmulAlgo_t algo{};
  std::uint64_t workspace = 0;
  bool pinned = false;
  std::array<std::uint64_t, 9> config{};

  Prepared() = default;
  Prepared(const Prepared&) = delete;
  Prepared& operator=(const Prepared&) = delete;
  Prepared(Prepared&&) = delete;
  Prepared& operator=(Prepared&&) = delete;
  ~Prepared() {
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

LtGemm::LtGemm(cublasLtContext* handle, std::uint64_t workspace, std::uint64_t workspace_bytes,
               bool pins, const GemmPins& table, int cc, int sms)
    : handle_(handle),
      workspace_(workspace),
      workspace_bytes_(workspace_bytes),
      pins_(pins),
      table_(table),
      cc_(cc),
      sms_(sms) {}

LtGemm::~LtGemm() {
  prepared_.clear();
  (void)cublasLtDestroy(handle_);
}

std::expected<std::unique_ptr<LtGemm>, std::string> LtGemm::Create(std::uint64_t workspace,
                                                                   std::uint64_t workspace_bytes,
                                                                   bool pins,
                                                                   const GemmPins* table) {
  if (const std::size_t loaded = cublasLtGetVersion(); loaded != CUBLAS_VERSION) {
    return std::unexpected(
        std::format("cuBLASLt {} is loaded; this build needs {}", loaded, CUBLAS_VERSION));
  }
  for (char** entry = environ; *entry != nullptr; ++entry) {
    const std::string_view variable(*entry);
    const std::string_view name = variable.substr(0, variable.find('='));
    if (NumericsSwitch(name)) {
      return std::unexpected(std::format("{} is set, which changes how cuBLAS computes", name));
    }
  }
  if (workspace_bytes > 0 && (workspace == 0 || workspace % 256 != 0)) {
    return std::unexpected(std::string("the cuBLASLt workspace is not 256-byte aligned"));
  }
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    return std::unexpected(std::string("the device's properties"));
  }
  cublasLtHandle_t handle = nullptr;
  if (const cublasStatus_t s = cublasLtCreate(&handle); s != CUBLAS_STATUS_SUCCESS) {
    return std::unexpected("cublasLtCreate failed: " + LtStatus(s));
  }
  return std::unique_ptr<LtGemm>(new LtGemm(
      handle, workspace, workspace_bytes, pins, table != nullptr ? *table : PinnedGemms(),
      (prop.major * 100) + (prop.minor * 10), prop.multiProcessorCount));
}

Status LtGemm::Prepare(std::int64_t ldx, std::int64_t ldw, std::int64_t ldo, std::int64_t m,
                       std::int64_t n, std::int64_t k) {
  auto found = Find(ldx, ldw, ldo, m, n, k);
  if (!found) {
    return std::unexpected(found.error());
  }
  return {};
}

std::expected<LtGemm::Prepared*, std::string> LtGemm::Find(std::int64_t ldx, std::int64_t ldw,
                                                           std::int64_t ldo, std::int64_t m,
                                                           std::int64_t n, std::int64_t k) {
  for (const auto& p : prepared_) {
    if (p->ldx == ldx && p->ldw == ldw && p->ldo == ldo && p->m == m && p->n == n && p->k == k) {
      return p.get();
    }
  }
  if (!Fits(m) || !Fits(n) || !Fits(k) || ldx < k || ldw < k || ldo < n || !Fits(ldx) ||
      !Fits(ldw) || !Fits(ldo)) {
    return std::unexpected(
        std::format("LtGemm: {} x {} x {} with strides {}, {}, {}", m, n, k, ldx, ldw, ldo));
  }
  auto p = std::make_unique<Prepared>();
  p->ldx = ldx;
  p->ldw = ldw;
  p->ldo = ldo;
  p->m = m;
  p->n = n;
  p->k = k;
  // Column-major, as Linear calls cublasGemmEx: D (n x m) = op(A) B with A
  // the weights (k x n, transposed), B the input (k x m).
  const cublasOperation_t transa = CUBLAS_OP_T;
  cublasStatus_t s = cublasLtMatmulDescCreate(&p->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (s == CUBLAS_STATUS_SUCCESS) {
    s = cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                                       sizeof transa);
  }
  if (s == CUBLAS_STATUS_SUCCESS) {
    s = cublasLtMatrixLayoutCreate(&p->a, CUDA_R_16BF, static_cast<std::uint64_t>(k),
                                   static_cast<std::uint64_t>(n), ldw);
  }
  if (s == CUBLAS_STATUS_SUCCESS) {
    s = cublasLtMatrixLayoutCreate(&p->b, CUDA_R_16BF, static_cast<std::uint64_t>(k),
                                   static_cast<std::uint64_t>(m), ldx);
  }
  if (s == CUBLAS_STATUS_SUCCESS) {
    s = cublasLtMatrixLayoutCreate(&p->d, CUDA_R_16BF, static_cast<std::uint64_t>(n),
                                   static_cast<std::uint64_t>(m), ldo);
  }
  if (s != CUBLAS_STATUS_SUCCESS) {
    return std::unexpected("LtGemm: describing the product failed: " + LtStatus(s));
  }
  // A pin, for this device and cuBLASLt, packed operands only.
  const GemmPins& table = table_;
  const GemmPin* pin = nullptr;
  if (pins_ && cc_ == table.compute_capability && sms_ == table.sm_count &&
      cublasLtGetVersion() == table.cublaslt && ldx == k && ldw == k && ldo == n) {
    for (const GemmPin& candidate : table.pins) {
      if (candidate.m == m && candidate.n == n && candidate.k == k) {
        pin = &candidate;
      }
    }
  }
  // A pinned algorithm that cuBLASLt no longer builds or accepts for the
  // shape (or that needs more workspace than there is) is not used: the
  // shape falls back to the heuristic's choice, recorded as unpinned
  // (Describe), rather than refusing the product.
  const auto build_pin = [&](const GemmPin& chosen) -> bool {
    const std::uint64_t id = chosen.config[0];
    if (id > static_cast<std::uint64_t>(kIntMax)) {
      return false;
    }
    cublasStatus_t b =
        cublasLtMatmulAlgoInit(handle_, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                               CUDA_R_16BF, CUDA_R_16BF, static_cast<int>(id), &p->algo);
    for (std::size_t i = 1; b == CUBLAS_STATUS_SUCCESS && i < kAttributes.size(); ++i) {
      std::size_t size = 0;
      b = cublasLtMatmulAlgoConfigGetAttribute(&p->algo, kAttributes.at(i), nullptr, 0, &size);
      if (b != CUBLAS_STATUS_SUCCESS || size == 0 || size > sizeof(std::uint64_t)) {
        return false;
      }
      const std::uint64_t value = chosen.config.at(i);
      if (size < sizeof(std::uint64_t) && (value >> (8 * size)) != 0) {
        return false;
      }
      // Little-endian: the value's low bytes.
      b = cublasLtMatmulAlgoConfigSetAttribute(&p->algo, kAttributes.at(i), &value, size);
    }
    cublasLtMatmulHeuristicResult_t checked{};
    if (b != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulAlgoCheck(handle_, p->operation, p->a, p->b, p->d, p->d, &p->algo,
                                &checked) != CUBLAS_STATUS_SUCCESS ||
        checked.workspaceSize > workspace_bytes_) {
      return false;
    }
    p->workspace = checked.workspaceSize;
    p->pinned = true;
    p->config = chosen.config;
    return true;
  };
  if (pin == nullptr || !build_pin(*pin)) {
    p->algo = {};
    cublasLtMatmulPreference_t preference = nullptr;
    s = cublasLtMatmulPreferenceCreate(&preference);
    if (s == CUBLAS_STATUS_SUCCESS) {
      s = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_bytes_, sizeof workspace_bytes_);
    }
    cublasLtMatmulHeuristicResult_t result{};
    int returned = 0;
    if (s == CUBLAS_STATUS_SUCCESS) {
      s = cublasLtMatmulAlgoGetHeuristic(handle_, p->operation, p->a, p->b, p->d, p->d, preference,
                                         1, &result, &returned);
    }
    if (preference != nullptr) {
      (void)cublasLtMatmulPreferenceDestroy(preference);
    }
    if (s != CUBLAS_STATUS_SUCCESS || returned < 1) {
      return std::unexpected("LtGemm: cuBLASLt has no algorithm for a product: " + LtStatus(s));
    }
    p->algo = result.algo;
    p->workspace = result.workspaceSize;
    for (std::size_t i = 0; i < kAttributes.size(); ++i) {
      std::size_t size = 0;
      (void)cublasLtMatmulAlgoConfigGetAttribute(&p->algo, kAttributes.at(i), nullptr, 0, &size);
      std::uint64_t value = 0;
      if (size > 0 && size <= sizeof value) {
        (void)cublasLtMatmulAlgoConfigGetAttribute(&p->algo, kAttributes.at(i), &value, size,
                                                   &size);
      }
      p->config.at(i) = value;
    }
  }
  if (p->workspace > workspace_bytes_) {
    return std::unexpected(std::format("LtGemm: an algorithm needs {} bytes of workspace, not {}",
                                       p->workspace, workspace_bytes_));
  }
  prepared_.push_back(std::move(p));
  return prepared_.back().get();
}

Status LtGemm::Linear(const Bf16* x, std::int64_t ldx, const Bf16* w, std::int64_t ldw, Bf16* out,
                      std::int64_t ldo, std::int64_t m, std::int64_t n, std::int64_t k,
                      Stream stream) {
  if (!Aligned256(x) || !Aligned256(w) || !Aligned256(out)) {
    return std::unexpected(std::string("LtGemm: operands not 256-byte aligned"));
  }
  auto found = Find(ldx, ldw, ldo, m, n, k);
  if (!found) {
    return std::unexpected(found.error());
  }
  const Prepared& p = **found;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // NOLINTBEGIN(performance-no-int-to-ptr): the workspace is a device address.
  const cublasStatus_t s =
      cublasLtMatmul(handle_, p.operation, &alpha, w, p.a, x, p.b, &beta, out, p.d, out, p.d,
                     &p.algo, p.workspace > 0 ? reinterpret_cast<void*>(workspace_) : nullptr,
                     p.workspace, static_cast<cudaStream_t>(stream));
  // NOLINTEND(performance-no-int-to-ptr)
  if (s != CUBLAS_STATUS_SUCCESS) {
    return std::unexpected("LtGemm: cublasLtMatmul failed: " + LtStatus(s));
  }
  return {};
}

std::string LtGemm::Describe() const {
  std::string out = "[";
  for (std::size_t i = 0; i < prepared_.size(); ++i) {
    const Prepared& p = *prepared_[i];
    out += std::format(
        R"({}{{"m": {}, "n": {}, "k": {}, "pinned": {}, "workspace": {}, "config": [{}, {}, {}, {}, {}, {}, {}, {}, {}]}})",
        i == 0 ? "" : ", ", p.m, p.n, p.k, p.pinned, p.workspace, p.config[0], p.config[1],
        p.config[2], p.config[3], p.config[4], p.config[5], p.config[6], p.config[7], p.config[8]);
  }
  return out + "]";
}

}  // namespace llmp::kernels::image
