// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// BF16 matrix products for the Qwen-Image-2.1 pipeline (M3): BF16 operands,
// F32 accumulation (CUBLAS_COMPUTE_32F), BF16 or F32 out.
//
// - The cuBLAS calls (Linear, ConvProduct and the VAE attention's two
//   products): cublasGemmEx with CUBLAS_GEMM_DEFAULT_TENSOR_OP, the call
//   PyTorch makes for a BF16 nn.Linear (at::cuda::blas::gemm<at::BFloat16>).
//   The handle is llmpalooza's (kernels/ggml/cublas.h): created on the stream the
//   products queue on, with its declared workspace.
// - LtGemm: the same product through cuBLASLt with the algorithm chosen per
//   shape, deterministically: pinned by its nine CUBLASLT_ALGO_CONFIG
//   attributes (recon_gemm.h's) for the shapes gemm.cc's table records on
//   the device it was tuned on, and otherwise cuBLASLt's first heuristic
//   choice for the given workspace (also for a pin cuBLASLt no longer
//   accepts, which is then recorded as unpinned). Every pinned algorithm
//   wrote the same bits as cublasGemmEx's choice when tuned (five of the
//   sixteen split K, reduced in a fixed order), and none is slower
//   beyond the timing's noise (benchmarks/qwen_image_gemm_tune.cc;
//   docs/experiments/qwen-image-native).

#ifndef LLMP_KERNELS_IMAGE_GEMM_H_
#define LLMP_KERNELS_IMAGE_GEMM_H_

#include <array>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "kernels/image/ops.h"

struct cublasContext;
struct cublasLtContext;

namespace llmp::kernels::image {

// A linear layer over row-major matrices: out[m, n] = x[m, k] . w[n, k]^T
// (+ out when accumulate), x rows `ldx` apart, w rows `ldw` apart, out rows
// `ldo` apart.
Status Linear(cublasContext* handle, const Bf16* x, std::int64_t ldx, const Bf16* w,
              std::int64_t ldw, Bf16* out, std::int64_t ldo, std::int64_t m, std::int64_t n,
              std::int64_t k, bool accumulate = false);
Status LinearF32(cublasContext* handle, const Bf16* x, std::int64_t ldx, const Bf16* w,
                 std::int64_t ldw, float* out, std::int64_t ldo, std::int64_t m, std::int64_t n,
                 std::int64_t k);

// A channels-first convolution as a product: out[co, p] = w[co, kk] .
// col[kk, p] (+ out when accumulate), for `pixels` output pixels whose
// column block starts at col (rows of `ldc` elements) and output block at
// out (rows of `ldo` elements).
Status ConvProduct(cublasContext* handle, const Bf16* w, std::int64_t out_channels,
                   std::int64_t inner, const Bf16* col, std::int64_t ldc, Bf16* out,
                   std::int64_t ldo, std::int64_t pixels, bool accumulate);

// Single-head attention's two products, channels-first q, k, v [c, n]:
// scores[i, j] = sum_c q[c, i] k[c, j] (F32), and out[c, i] = sum_j
// probs[i, j] v[c, j].
Status ScoresQtK(cublasContext* handle, const Bf16* q, const Bf16* k, float* scores,
                 std::int64_t channels, std::int64_t tokens, std::int64_t ld);
Status ValuesTimesProbs(cublasContext* handle, const Bf16* v, const Bf16* probs, Bf16* out,
                        std::int64_t channels, std::int64_t tokens, std::int64_t ld);

// One pinned algorithm: the product's m, n and k (packed operands, BF16
// out) and the nine CUBLASLT_ALGO_CONFIG attributes, in recon_gemm.h's
// order (ID, tile, split-K, reduction, swizzle, custom option, stages,
// inner shape, cluster shape).
struct GemmPin {
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  std::array<std::uint64_t, 9> config{};
};

// The pins gemm.cc records, and the device and cuBLASLt they hold for.
struct GemmPins {
  int compute_capability = 0;  // 100 * major + 10 * minor: 1,210 for sm_121
  int sm_count = 0;
  std::size_t cublaslt = 0;  // cublasLtGetVersion()
  std::span<const GemmPin> pins;
};
GemmPins PinnedGemms();

class LtGemm {
 public:
  // On the current device, with `workspace` (a device range, 256-byte
  // aligned, that the caller keeps and charges, used by one stream at a
  // time). Refused unless the loaded cuBLASLt is the one this build pins
  // (D-076) and none of cuBLAS's numerics switches is set in the
  // environment (kernels/ggml/cublas.h). `pins` false: cuBLASLt's first
  // heuristic choice for every shape (the tuning baseline). `table`: pins
  // in place of PinnedGemms()'s (a test's, kept by the caller for the
  // LtGemm's life).
  static std::expected<std::unique_ptr<LtGemm>, std::string> Create(
      std::uint64_t workspace, std::uint64_t workspace_bytes, bool pins = true,
      const GemmPins* table = nullptr);
  LtGemm(const LtGemm&) = delete;
  LtGemm& operator=(const LtGemm&) = delete;
  LtGemm(LtGemm&&) = delete;
  LtGemm& operator=(LtGemm&&) = delete;
  ~LtGemm();

  // Linear's product (no accumulation), BF16 out. The shape's descriptors
  // and algorithm are made on its first use (host work, nothing queued) and
  // kept; a refusal queues nothing. Every operand 256-byte aligned (the
  // heuristic's assumption), leading dimensions at least the rows. The
  // weights are read in place from their artifact's extents, so this rests
  // on the artifact's member_alignment of 256 (docs/artifact-format.md),
  // which the importer must keep guaranteeing (owner, 2026-09-29).
  Status Linear(const Bf16* x, std::int64_t ldx, const Bf16* w, std::int64_t ldw, Bf16* out,
                std::int64_t ldo, std::int64_t m, std::int64_t n, std::int64_t k, Stream stream);
  // Makes the shape's descriptors and algorithm now (Linear's first use).
  Status Prepare(std::int64_t ldx, std::int64_t ldw, std::int64_t ldo, std::int64_t m,
                 std::int64_t n, std::int64_t k);

  // Each prepared shape and how its algorithm was chosen, JSON.
  std::string Describe() const;

  struct Prepared;

 private:
  LtGemm(cublasLtContext* handle, std::uint64_t workspace, std::uint64_t workspace_bytes, bool pins,
         const GemmPins& table, int cc, int sms);
  std::expected<Prepared*, std::string> Find(std::int64_t ldx, std::int64_t ldw, std::int64_t ldo,
                                             std::int64_t m, std::int64_t n, std::int64_t k);

  cublasLtContext* handle_;
  std::uint64_t workspace_;
  std::uint64_t workspace_bytes_;
  bool pins_;
  GemmPins table_;
  int cc_;
  int sms_;
  std::vector<std::unique_ptr<Prepared>> prepared_;
};

}  // namespace llmp::kernels::image

#endif  // LLMP_KERNELS_IMAGE_GEMM_H_
