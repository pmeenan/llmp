// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The reconstruction path's GEMM (docs/backend-proof.md, "EXL3
// reconstruction-path linears"), CUDA builds only: y = x · w for
// reconstructed F16 weights w, through cuBLASLt with an algorithm pinned by
// its complete configuration. Upstream's hgemm_recon makes a legacy
// cublasGemmEx call (FP32 compute, host pointer mode), whose heuristic
// resolves the algorithm per shape and cuBLAS version; the approved rule
// replaces that choice with the nine attributes recorded for each of the
// 21 GEMMs in exl3-recon-pin.json, which a probe showed bit-identical to
// the legacy call under the pinned cuBLAS 13.8.0.4.
//
// The call is the probe's: operands in cuBLAS's column-major terms, A the
// weights (n × k, leading dimension n), B the input (k × m, leading
// dimension k), D the output (n × m, leading dimension ldc, so that a slice
// of a wider output is written in place); COMPUTE_32F with FP32 scaling,
// alpha 1 and beta 0, the device's SM count targeted, no workspace (none of
// the pinned algorithms uses one; cublasLtMatmulAlgoCheck must agree).
//
// Create refuses unless the loaded cuBLASLt is the pinned one and none of
// cuBLAS's own numerics switches is set in the environment, as
// kernels/ggml/cublas.h does. One instance per device, used on the device
// submission lane only; the GEMM queues on the launch context's stream.

#ifndef LLMP_KERNELS_EXL3_RECON_GEMM_H_
#define LLMP_KERNELS_EXL3_RECON_GEMM_H_

#include <cstdint>
#include <expected>
#include <memory>

#include "kernels/exl3/launch.h"
#include "kernels/exl3/validate.h"

struct cublasLtContext;

namespace llmp::kernels::exl3 {

class ReconGemm {
 public:
  static std::expected<std::unique_ptr<ReconGemm>, KernelFailure> Create();

  ReconGemm(const ReconGemm&) = delete;
  ReconGemm& operator=(const ReconGemm&) = delete;
  ReconGemm(ReconGemm&&) = delete;
  ReconGemm& operator=(ReconGemm&&) = delete;
  ~ReconGemm();

  // Queues the GEMM on the context's stream with `algorithm`, targeting
  // `sm_count` SMs. Which algorithm is the caller's plan, taken from
  // exl3-recon-pin.json with the GEMM it was pinned for (LtAlgorithm): a
  // pin recorded for another GEMM (other m, k, n, ldc or output) is
  // refused, so no GEMM runs at a size the table does not pin. That its
  // nine attributes are the table's is the plan's to guarantee: cuBLASLt
  // checks only that it can run this GEMM, and another valid algorithm
  // need not give upstream's bits. Refused, with nothing queued, if the
  // operands do not check, the pin is another GEMM's, cuBLASLt rejects the
  // algorithm for them or it needs a workspace. Any error cublasLtMatmul
  // returns faults the context: its documentation does not promise that
  // an error leaves nothing queued, and Check has already run the same
  // descriptors through cublasLtMatmulAlgoCheck, so only a device or
  // launch failure is expected there.
  std::expected<void, KernelFailure> Run(LaunchContext& launch, const ReconGemmOperands& operands,
                                         const LtAlgorithm& algorithm, int sm_count);
  // Run's checks, cuBLASLt's included, without queueing anything.
  std::expected<void, KernelFailure> Check(const ReconGemmOperands& operands,
                                           const LtAlgorithm& algorithm, int sm_count) const;

 private:
  struct Prepared;
  explicit ReconGemm(cublasLtContext* handle) : handle_(handle) {}
  std::expected<std::unique_ptr<Prepared>, KernelFailure> Prepare(const ReconGemmOperands& operands,
                                                                  const LtAlgorithm& algorithm,
                                                                  int sm_count) const;

  cublasLtContext* handle_;
};

}  // namespace llmp::kernels::exl3

#endif  // LLMP_KERNELS_EXL3_RECON_GEMM_H_
