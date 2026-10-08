// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen-Image-2.1 pipeline's BF16 products on cuBLASLt, for choosing the
// algorithm each shape pins (docs/experiments/qwen-image-native/README.md,
// "Speed"; kernels/image/gemm.h). For every product the pipeline makes at
// the given image size, prompt length and batch, it asks cuBLASLt's
// heuristic for up to --candidates algorithms (BF16 operands and output,
// COMPUTE_32F, the given workspace), times each (the median of --reps
// launches after a warm-up, CUDA events, operands over 256 MiB so L2 holds
// none of them between launches), checks that two launches write the same
// bits, and compares each candidate's output with cublasGemmEx's (the call
// the image kernels made before, CUBLAS_GEMM_DEFAULT_TENSOR_OP). Output:
// JSON lines, one per candidate, with its heuristic rank and its nine
// CUBLASLT_ALGO_CONFIG attributes (recon_gemm.h's), and one per shape with
// cublasGemmEx's time.
//
//   llmp_qwen_image_gemm_tune [--rows N] [--candidates N] [--reps N]
//                               [--workspace MiB] [--shape m,n,k[,batch]]...
//
// Operands are filled with a fixed pseudo-random BF16 pattern (|x| < 1).

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <format>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

using Status = std::expected<void, std::string>;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

Status Cuda(cudaError_t r, std::string_view what) {
  if (r != cudaSuccess) {
    return Error(std::format("{}: {}", what, cudaGetErrorString(r)));
  }
  return {};
}

Status Lt(cublasStatus_t r, std::string_view what) {
  if (r != CUBLAS_STATUS_SUCCESS) {
    return Error(std::format("{}: {}", what, cublasLtGetStatusString(r)));
  }
  return {};
}

// One product: out[m, n] = x[m, k] . w[n, k]^T, `batch` of them with
// separate weights and outputs and a shared x (strided, weights `n * k`
// apart).
struct Shape {
  std::int64_t m = 0, n = 0, k = 0, batch = 1;
};

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

std::array<std::uint64_t, 9> Config(const cublasLtMatmulAlgo_t& algo) {
  std::array<std::uint64_t, 9> out{};
  for (std::size_t i = 0; i < kAttributes.size(); ++i) {
    // Read at the size cuBLASLt reports for the attribute (recon_gemm.cc).
    std::size_t size = 0;
    (void)cublasLtMatmulAlgoConfigGetAttribute(&algo, kAttributes.at(i), nullptr, 0, &size);
    std::uint64_t v = 0;
    if (size > 0 && size <= sizeof v) {
      (void)cublasLtMatmulAlgoConfigGetAttribute(&algo, kAttributes.at(i), &v, size, &size);
    }
    out.at(i) = v;
  }
  return out;
}

Status Fill(void* device, std::int64_t n, std::uint32_t seed) {
  std::vector<std::uint16_t> host(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    std::uint32_t h = (static_cast<std::uint32_t>(i) * 2654435761U) ^ seed;
    h ^= h >> 15U;
    h *= 2246822519U;
    h ^= h >> 13U;
    const float f = (static_cast<float>(h & 0xffffU) / 32768.0f) - 1.0f;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof bits);
    host[static_cast<std::size_t>(i)] = static_cast<std::uint16_t>(bits >> 16U);
  }
  return Cuda(cudaMemcpy(device, host.data(), host.size() * 2, cudaMemcpyHostToDevice), "fill");
}

float F(std::uint16_t b) {
  const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16U;
  float f = 0;
  std::memcpy(&f, &bits, sizeof f);
  return f;
}

// How many of n BF16 values differ, and the largest relative difference.
struct Differ {
  std::uint64_t count = 0;
  double max_rel = 0;
};
std::expected<Differ, std::string> Compare(const void* a, const void* b, std::int64_t n) {
  std::vector<std::uint16_t> x(static_cast<std::size_t>(n));
  std::vector<std::uint16_t> y(static_cast<std::size_t>(n));
  if (auto r = Cuda(cudaMemcpy(x.data(), a, x.size() * 2, cudaMemcpyDeviceToHost), "d2h"); !r) {
    return std::unexpected(r.error());
  }
  if (auto r = Cuda(cudaMemcpy(y.data(), b, y.size() * 2, cudaMemcpyDeviceToHost), "d2h"); !r) {
    return std::unexpected(r.error());
  }
  Differ d;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (x[i] != y[i]) {
      ++d.count;
      const double rel = std::abs(static_cast<double>(F(x[i])) - F(y[i])) /
                         std::max(std::abs(static_cast<double>(F(y[i]))), 1e-3);
      d.max_rel = std::max(d.max_rel, rel);
    }
  }
  return d;
}

struct Options {
  std::int64_t rows = 4096;
  int candidates = 16;
  int reps = 20;
  std::int64_t workspace_mib = 32;
  std::vector<Shape> shapes;
};

std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options o;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view a = args[i];
    if (i + 1 >= args.size()) {
      return Error(std::format("{} needs a value", a));
    }
    const std::string v = args[++i];
    if (a == "--rows") {
      o.rows = std::stoll(v);
    } else if (a == "--candidates") {
      o.candidates = std::stoi(v);
    } else if (a == "--reps") {
      o.reps = std::stoi(v);
    } else if (a == "--workspace") {
      o.workspace_mib = std::stoll(v);
    } else if (a == "--shape") {
      // m,n,k[,batch], each a positive decimal.
      std::array<std::int64_t, 4> dims = {0, 0, 0, 1};
      std::size_t at = 0;
      std::size_t count = 0;
      while (at <= v.size() && count < dims.size()) {
        const std::size_t end = std::min(v.find(',', at), v.size());
        const std::string_view field = std::string_view(v).substr(at, end - at);
        std::int64_t value = 0;
        const auto [ptr, ec] = std::from_chars(field.data(), field.data() + field.size(), value);
        if (ec != std::errc() || ptr != field.data() + field.size() || value <= 0) {
          return Error("--shape m,n,k[,batch]");
        }
        dims.at(count++) = value;
        at = end + 1;
      }
      if (count < 3 || at <= v.size()) {
        return Error("--shape m,n,k[,batch]");
      }
      o.shapes.push_back({.m = dims[0], .n = dims[1], .k = dims[2], .batch = dims[3]});
    } else {
      return Error(std::format("unknown argument {}", a));
    }
  }
  if (o.shapes.empty()) {
    // The denoiser's block products at `rows` image rows, the batched gate
    // and up, and its input and output projections.
    const std::int64_t r = o.rows;
    o.shapes = {{r, 4096, 4096, 1},  {r, 12288, 4096, 1}, {r, 4096, 12288, 1}, {r, 4096, 4096, 3},
                {r, 12288, 4096, 2}, {r, 8192, 4096, 1},  {r, 4096, 64, 1},    {r, 64, 4096, 1}};
  }
  return o;
}

struct Buffers {
  void* x = nullptr;
  void* w = nullptr;
  void* out = nullptr;
  void* ref = nullptr;
  void* again = nullptr;
  void* workspace = nullptr;
};

Status Tune(cublasLtHandle_t lt, cublasHandle_t blas, const Options& o, const Shape& s,
            const Buffers& b, cudaStream_t stream) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const std::int64_t out_elems = s.m * s.n;
  const auto w_stride = s.n * s.k;
  cudaEvent_t e0 = nullptr;
  cudaEvent_t e1 = nullptr;
  (void)cudaEventCreate(&e0);
  (void)cudaEventCreate(&e1);
  // Sustained, as the pipeline runs them: a warm-up of about half a second
  // of back-to-back launches, then five batches of --reps back-to-back
  // launches each; the median batch's time per launch.
  auto time = [&](auto&& launch) -> std::expected<double, std::string> {
    (void)cudaEventRecord(e0, stream);
    if (auto r = launch(b.out); !r) return std::unexpected(r.error());
    (void)cudaEventRecord(e1, stream);
    if (auto r = Cuda(cudaEventSynchronize(e1), "timing"); !r) return std::unexpected(r.error());
    float once = 0;
    (void)cudaEventElapsedTime(&once, e0, e1);
    const int warm = std::clamp(static_cast<int>(500.0f / std::max(once, 0.01f)), 3, 2000);
    for (int i = 0; i < warm; ++i) {
      if (auto r = launch(b.out); !r) return std::unexpected(r.error());
    }
    std::vector<float> ms;
    for (int batch = 0; batch < 5; ++batch) {
      (void)cudaEventRecord(e0, stream);
      for (int i = 0; i < o.reps; ++i) {
        if (auto r = launch(b.out); !r) return std::unexpected(r.error());
      }
      (void)cudaEventRecord(e1, stream);
      if (auto r = Cuda(cudaEventSynchronize(e1), "timing"); !r) return std::unexpected(r.error());
      float t = 0;
      (void)cudaEventElapsedTime(&t, e0, e1);
      ms.push_back(t / static_cast<float>(o.reps));
    }
    std::ranges::sort(ms);
    return static_cast<double>(ms[ms.size() / 2]);
  };
  // cublasGemmEx, the reference and the old path (batch: one call each).
  auto gemm_ex = [&](void* out) -> Status {
    for (std::int64_t i = 0; i < s.batch; ++i) {
      if (auto r = cublasGemmEx(
              blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(s.n), static_cast<int>(s.m),
              static_cast<int>(s.k), &alpha, static_cast<std::uint16_t*>(b.w) + (i * w_stride),
              CUDA_R_16BF, static_cast<int>(s.k), b.x, CUDA_R_16BF, static_cast<int>(s.k), &beta,
              static_cast<std::uint16_t*>(out) + (i * out_elems), CUDA_R_16BF,
              static_cast<int>(s.n), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
          r != CUBLAS_STATUS_SUCCESS) {
        return Error("cublasGemmEx failed");
      }
    }
    return {};
  };
  if (auto r = gemm_ex(b.ref); !r) return r;
  auto ex_ms = time(gemm_ex);
  if (!ex_ms) return std::unexpected(ex_ms.error());
  const double flop = 2.0 * static_cast<double>(s.m) * static_cast<double>(s.n) *
                      static_cast<double>(s.k) * static_cast<double>(s.batch);
  std::println(
      R"({{"shape": [{}, {}, {}, {}], "path": "cublasGemmEx", "ms": {:.4f}, "tflops": {:.1f}}})",
      s.m, s.n, s.k, s.batch, *ex_ms, flop / (*ex_ms * 1e9));
  (void)std::fflush(stdout);

  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t la = nullptr;
  cublasLtMatrixLayout_t lb = nullptr;
  cublasLtMatrixLayout_t ld = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  const cublasOperation_t transa = CUBLAS_OP_T;
  const std::uint64_t ws = static_cast<std::uint64_t>(o.workspace_mib) << 20U;
  Status st = Lt(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");
  st = st ? Lt(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                                              sizeof transa),
               "transa")
          : st;
  st = st ? Lt(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, static_cast<std::uint64_t>(s.k),
                                          static_cast<std::uint64_t>(s.n), s.k),
               "a")
          : st;
  st = st ? Lt(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, static_cast<std::uint64_t>(s.k),
                                          static_cast<std::uint64_t>(s.m), s.k),
               "b")
          : st;
  st = st ? Lt(cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, static_cast<std::uint64_t>(s.n),
                                          static_cast<std::uint64_t>(s.m), s.n),
               "d")
          : st;
  if (st && s.batch > 1) {
    const auto count = static_cast<std::int32_t>(s.batch);
    const std::int64_t zero = 0;
    const std::int64_t d_stride = out_elems;
    for (auto* l : {la, lb, ld}) {
      st = st ? Lt(cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count,
                                                    sizeof count),
                   "batch")
              : st;
    }
    st = st ? Lt(cublasLtMatrixLayoutSetAttribute(la, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &w_stride, sizeof w_stride),
                 "a stride")
            : st;
    st = st ? Lt(cublasLtMatrixLayoutSetAttribute(lb, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &zero, sizeof zero),
                 "b stride")
            : st;
    st = st ? Lt(cublasLtMatrixLayoutSetAttribute(ld, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &d_stride, sizeof d_stride),
                 "d stride")
            : st;
  }
  st = st ? Lt(cublasLtMatmulPreferenceCreate(&pref), "pref") : st;
  st = st ? Lt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &ws, sizeof ws),
               "pref ws")
          : st;
  if (!st) return st;
  std::vector<cublasLtMatmulHeuristicResult_t> results(static_cast<std::size_t>(o.candidates));
  int returned = 0;
  st = Lt(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, o.candidates, results.data(),
                                         &returned),
          "heuristic");
  for (int rank = 0; st && rank < returned; ++rank) {
    const auto& h = results[static_cast<std::size_t>(rank)];
    auto launch = [&](void* out) -> Status {
      return Lt(cublasLtMatmul(lt, op, &alpha, b.w, la, b.x, lb, &beta, out, ld, out, ld, &h.algo,
                               b.workspace, ws, stream),
                "matmul");
    };
    auto ms = time(launch);
    if (!ms) {
      std::println(R"({{"shape": [{}, {}, {}, {}], "rank": {}, "error": "{}"}})", s.m, s.n, s.k,
                   s.batch, rank, ms.error());
      continue;
    }
    // Repeatable, and how far from cublasGemmEx's.
    if (auto r = launch(b.again); !r) return r;
    if (auto r = Cuda(cudaStreamSynchronize(stream), "sync"); !r) return r;
    auto repeat = Compare(b.out, b.again, out_elems * s.batch);
    auto vs_ex = Compare(b.out, b.ref, out_elems * s.batch);
    if (!repeat || !vs_ex) return Error("compare");
    const auto c = Config(h.algo);
    std::println(
        R"({{"shape": [{}, {}, {}, {}], "rank": {}, "ms": {:.4f}, "tflops": {:.1f}, )"
        R"("workspace": {}, "waves": {:.3f}, "repeat_differ": {}, "vs_gemmex_differ": {}, )"
        R"("vs_gemmex_max_rel": {:.3g}, "config": [{}, {}, {}, {}, {}, {}, {}, {}, {}]}})",
        s.m, s.n, s.k, s.batch, rank, *ms, flop / (*ms * 1e9), h.workspaceSize, h.wavesCount,
        repeat->count, vs_ex->count, vs_ex->max_rel, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
        c[8]);
    (void)std::fflush(stdout);
  }
  (void)cublasLtMatmulPreferenceDestroy(pref);
  (void)cublasLtMatrixLayoutDestroy(ld);
  (void)cublasLtMatrixLayoutDestroy(lb);
  (void)cublasLtMatrixLayoutDestroy(la);
  (void)cublasLtMatmulDescDestroy(op);
  (void)cudaEventDestroy(e0);
  (void)cudaEventDestroy(e1);
  return st;
}

Status Run(const Options& o) {
  std::int64_t x_elems = 0;
  std::int64_t w_elems = 0;
  std::int64_t out_elems = 0;
  for (const Shape& s : o.shapes) {
    x_elems = std::max(x_elems, s.m * s.k);
    w_elems = std::max(w_elems, s.n * s.k * s.batch);
    out_elems = std::max(out_elems, s.m * s.n * s.batch);
  }
  Buffers b;
  const std::uint64_t ws = static_cast<std::uint64_t>(o.workspace_mib) << 20U;
  Status r = Cuda(cudaMalloc(&b.x, static_cast<std::size_t>(x_elems) * 2), "x");
  r = r ? Cuda(cudaMalloc(&b.w, static_cast<std::size_t>(w_elems) * 2), "w") : r;
  for (void** p : {&b.out, &b.ref, &b.again}) {
    r = r ? Cuda(cudaMalloc(p, static_cast<std::size_t>(out_elems) * 2), "out") : r;
  }
  r = r ? Cuda(cudaMalloc(&b.workspace, std::max<std::uint64_t>(ws, 256)), "workspace") : r;
  r = r ? Fill(b.x, x_elems, 1U) : r;
  r = r ? Fill(b.w, w_elems, 2U) : r;
  if (!r) return r;
  cudaStream_t stream = nullptr;
  (void)cudaStreamCreate(&stream);
  cublasLtHandle_t lt = nullptr;
  cublasHandle_t blas = nullptr;
  if (auto e = Lt(cublasLtCreate(&lt), "cublasLtCreate"); !e) return e;
  if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) return Error("cublasCreate");
  (void)cublasSetStream(blas, stream);
  (void)cublasSetWorkspace(blas, b.workspace, ws);
  cudaDeviceProp prop{};
  (void)cudaGetDeviceProperties(&prop, 0);
  std::println(R"({{"device": "{}", "sms": {}, "cublaslt": {}, "workspace_mib": {}}})", prop.name,
               prop.multiProcessorCount, cublasLtGetVersion(), o.workspace_mib);
  for (const Shape& s : o.shapes) {
    if (auto t = Tune(lt, blas, o, s, b, stream); !t) {
      std::println(stderr, "shape {}x{}x{}: {}", s.m, s.n, s.k, t.error());
    }
  }
  (void)cublasDestroy(blas);
  (void)cublasLtDestroy(lt);
  (void)cudaStreamDestroy(stream);
  for (void* p : {b.x, b.w, b.out, b.ref, b.again, b.workspace}) {
    (void)cudaFree(p);
  }
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  auto o = Parse(std::span(argv, static_cast<std::size_t>(argc)));
  if (!o) {
    std::println(stderr, "{}", o.error());
    return 2;
  }
  if (auto r = Run(*o); !r) {
    std::println(stderr, "error: {}", r.error());
    return 1;
  }
  return 0;
}
