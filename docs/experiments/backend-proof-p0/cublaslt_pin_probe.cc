// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
// Reference probe for the backend proof's P0 (docs/backend-proof.md, EXL3
// reconstruction-path Tier E); it does not implement llmpalooza inference. For
// each reconstruction GEMM recorded in exl3-recon-plan.json (cuBLAS
// 13.8.0.4), on seeded random FP16 inputs:
//
//   legacy:    cublasGemmEx exactly as ExLlamaV3's hgemm does it (a handle
//              with default math, host pointer mode and a 16 MiB workspace);
//   heuristic: cublasLtMatmul with the heuristic's first algorithm for the
//              same layouts, COMPUTE_32F, 48 SMs targeted and a 16 MiB,
//              16-byte-aligned preference; its complete configuration (all
//              nine CUBLASLT_ALGO_CONFIG attributes) is printed;
//   pinned:    cublasLtMatmul with an algorithm rebuilt by
//              cublasLtMatmulAlgoInit from that configuration alone.
//
// It prints one JSON object per GEMM: the configuration, and whether the
// three outputs are bit-identical.
//
//   cublaslt_pin_probe < gemms.txt
//
// Each input line: HSH|HSS n k m lda ldb ldc (the Lt layouts: A n x k, B k x
// m, D n x m, column-major), as cublas_plan.py records them.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "probe failed: %s\n", what);
    std::exit(1);
  }
}

#define CUDA(x) check((x) == cudaSuccess, #x)
#define BLAS(x) check((x) == CUBLAS_STATUS_SUCCESS, #x)

constexpr size_t kWorkspace = 16u << 20;

struct Config {
  uint64_t values[9];
  size_t sizes[9];
};

constexpr cublasLtMatmulAlgoConfigAttributes_t kAttributes[9] = {
    CUBLASLT_ALGO_CONFIG_ID,          CUBLASLT_ALGO_CONFIG_TILE_ID,      CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
    CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
    CUBLASLT_ALGO_CONFIG_STAGES_ID,   CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};
constexpr const char* kNames[9] = {"algo_id", "tile", "splitk", "reduction", "swizzle", "custom", "stages",
                                   "inner_shape", "cluster_shape"};

Config read(const cublasLtMatmulAlgo_t& algo) {
  // Attributes are 16- or 32-bit; query each one's size, then read it into
  // the low bytes of a zeroed 64-bit value (little-endian).
  Config c{};
  for (int i = 0; i < 9; ++i) {
    size_t size = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&algo, kAttributes[i], nullptr, 0, &size);
    check(size > 0 && size <= sizeof(uint64_t), "attribute size");
    size_t written = 0;
    BLAS(cublasLtMatmulAlgoConfigGetAttribute(&algo, kAttributes[i], &c.values[i], size, &written));
    c.sizes[i] = size;
  }
  return c;
}

bool same(const void* a, const void* b, size_t bytes) {
  std::vector<unsigned char> x(bytes), y(bytes);
  CUDA(cudaMemcpy(x.data(), a, bytes, cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(y.data(), b, bytes, cudaMemcpyDeviceToHost));
  return std::memcmp(x.data(), y.data(), bytes) == 0;
}

}  // namespace

int main() {
  cublasHandle_t legacy;
  cublasLtHandle_t lt;
  BLAS(cublasCreate(&legacy));
  BLAS(cublasLtCreate(&lt));
  void* workspace = nullptr;
  CUDA(cudaMalloc(&workspace, kWorkspace));
  BLAS(cublasSetPointerMode(legacy, CUBLAS_POINTER_MODE_HOST));
  BLAS(cublasSetMathMode(legacy, CUBLAS_DEFAULT_MATH));
  BLAS(cublasSetWorkspace(legacy, workspace, kWorkspace));
  std::mt19937 rng(20260926);
  std::normal_distribution<float> normal(0.0f, 0.1f);
  char kind[8];
  long n, k, m, lda, ldb, ldc;
  while (std::scanf("%7s %ld %ld %ld %ld %ld %ld", kind, &n, &k, &m, &lda, &ldb, &ldc) == 7) {
    const bool fp32_out = std::strcmp(kind, "HSS") == 0;
    const cudaDataType_t out_type = fp32_out ? CUDA_R_32F : CUDA_R_16F;
    const size_t out_elem = fp32_out ? 4 : 2;
    std::vector<__half> ha(static_cast<size_t>(lda * k)), hb(static_cast<size_t>(ldb * m));
    for (auto& v : ha) v = __float2half(normal(rng));
    for (auto& v : hb) v = __float2half(normal(rng));
    void *a, *b, *d[3];
    const size_t out_bytes = static_cast<size_t>(ldc * m) * out_elem;
    CUDA(cudaMalloc(&a, ha.size() * 2));
    CUDA(cudaMalloc(&b, hb.size() * 2));
    CUDA(cudaMemcpy(a, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(b, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));
    for (auto& p : d) {
      CUDA(cudaMalloc(&p, out_bytes));
      CUDA(cudaMemset(p, 0, out_bytes));
    }
    const float alpha = 1.0f, beta = 0.0f;
    BLAS(cublasGemmEx(legacy, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m),
                      static_cast<int>(k), &alpha, a, CUDA_R_16F, static_cast<int>(lda), b, CUDA_R_16F,
                      static_cast<int>(ldb), &beta, d[0], out_type, static_cast<int>(ldc), CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    cublasLtMatmulDesc_t op;
    BLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const int32_t sms = 48;
    BLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &sms, sizeof sms));
    cublasLtMatrixLayout_t la, lb, ld;
    BLAS(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, n, k, lda));
    BLAS(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, k, m, ldb));
    BLAS(cublasLtMatrixLayoutCreate(&ld, out_type, n, m, ldc));
    cublasLtMatmulPreference_t pref;
    BLAS(cublasLtMatmulPreferenceCreate(&pref));
    const uint64_t ws = kWorkspace;
    const uint32_t align = 16;
    BLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
    for (auto attr : {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES}) {
      BLAS(cublasLtMatmulPreferenceSetAttribute(pref, attr, &align, sizeof align));
    }
    cublasLtMatmulHeuristicResult_t result{};
    int found = 0;
    BLAS(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 1, &result, &found));
    check(found == 1, "no heuristic result");
    const Config config = read(result.algo);
    BLAS(cublasLtMatmul(lt, op, &alpha, a, la, b, lb, &beta, d[1], ld, d[1], ld, &result.algo, workspace,
                        kWorkspace, nullptr));

    cublasLtMatmulAlgo_t pinned;
    BLAS(cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, out_type, out_type,
                                static_cast<int>(config.values[0]), &pinned));
    for (int i = 1; i < 9; ++i) {
      BLAS(cublasLtMatmulAlgoConfigSetAttribute(&pinned, kAttributes[i], &config.values[i], config.sizes[i]));
    }
    cublasLtMatmulHeuristicResult_t checked{};
    BLAS(cublasLtMatmulAlgoCheck(lt, op, la, lb, ld, ld, &pinned, &checked));
    BLAS(cublasLtMatmul(lt, op, &alpha, a, la, b, lb, &beta, d[2], ld, d[2], ld, &pinned, workspace, kWorkspace,
                        nullptr));
    CUDA(cudaDeviceSynchronize());

    std::printf("{\"kind\": \"%s\", \"n\": %ld, \"k\": %ld, \"m\": %ld, \"lda\": %ld, \"ldb\": %ld, \"ldc\": %ld, "
                "\"workspace_needed\": %zu, \"config\": {",
                kind, n, k, m, lda, ldb, ldc, result.workspaceSize);
    for (int i = 0; i < 9; ++i) {
      std::printf("%s\"%s\": %llu", i ? ", " : "", kNames[i], static_cast<unsigned long long>(config.values[i]));
    }
    std::printf("}, \"heuristic_equals_legacy\": %s, \"pinned_equals_legacy\": %s}\n",
                same(d[0], d[1], out_bytes) ? "true" : "false", same(d[0], d[2], out_bytes) ? "true" : "false");
    std::fflush(stdout);
    BLAS(cublasLtMatmulPreferenceDestroy(pref));
    BLAS(cublasLtMatrixLayoutDestroy(la));
    BLAS(cublasLtMatrixLayoutDestroy(lb));
    BLAS(cublasLtMatrixLayoutDestroy(ld));
    BLAS(cublasLtMatmulDescDestroy(op));
    for (auto p : {a, b, d[0], d[1], d[2]}) CUDA(cudaFree(p));
  }
  CUDA(cudaFree(workspace));
  BLAS(cublasLtDestroy(lt));
  BLAS(cublasDestroy(legacy));
  return 0;
}
