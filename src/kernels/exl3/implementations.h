// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The EXL3 module's entries in the implementation registry (D-053;
// execution/registry.h), CUDA builds only: a build without the device code
// declares none, so a plan naming one is unsupported there (BP-S4).
//
//   exl3.linear.gemm               kQuantLinear       packed: exl3_gemm_kernel
//                                                     at the plan's tile shape
//                                                     and grid (EXL3-G's
//                                                     packed linears), then
//                                                     the bias
//   exl3.linear.gemv               kQuantLinear       packed: exl3_gemv_kernel
//                                                     at the plan's
//                                                     configuration and grid
//                                                     (EXL3-O's up to eight
//                                                     rows), then the bias
//   exl3.linear.reconstruct        kQuantLinear       reconstruction: input
//                                                     transform, rotated
//                                                     weights and the pinned
//                                                     cuBLASLt GEMM per slice,
//                                                     output transform, bias
//   exl3.linear.reconstruct_fused  kQuantLinear       fused reconstruction:
//                                                     transformed weights and
//                                                     the pinned GEMM per
//                                                     slice, bias
//   exl3.multi_linear.mgemm        kQuantMultiLinear  gate and up through
//                                                     exl3_mgemm_kernel
//   exl3.bias_add                  kBiasAdd           add_kernel_hhh: the
//                                                     q/k/v bias, as the
//                                                     native EXL3 plan's own
//                                                     operation (its record
//                                                     lists it apart from the
//                                                     linear); the linears
//                                                     then run without one
//
// linear.h has each path's kernels. The GEMM and the GEMV are the natural
// pair of implementations of one operation at up to eight rows
// (docs/backend-proof.md, "Coexistence and swapping"); a plan names one.
//
// Each identity covers everything that decides what an implementation
// computes and launches: the prepared ExLlamaV3 tree's digest (upstream's
// bytes, llmpalooza's patches and the build of the kernels, D-057); a digest of
// every file of llmpalooza's code in src/kernels/exl3, written at build time
// (module_digest.cmake); the SDK, target, device architecture, build type
// and sanitizers; for the reconstruction paths the pinned cuBLASLt; the
// name and a variant naming the kernels. The launch plan (tile shape, grid,
// pinned algorithms) is the plan's data, passed with each call, as the
// tuning record gives it.

#ifndef LLMP_KERNELS_EXL3_IMPLEMENTATIONS_H_
#define LLMP_KERNELS_EXL3_IMPLEMENTATIONS_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

#include "execution/registry.h"
#include "kernels/exl3/launch.h"
#include "kernels/exl3/linear.h"
#include "kernels/exl3/recon_gemm.h"
#include "kernels/exl3/validate.h"

namespace llmp::kernels::exl3 {

// What this module declares to the registry.
std::vector<execution::Implementation> Implementations();

// The digest of the module's own files, generated at build time.
std::string_view ModuleSourcesDigest();

// One EXL3 implementation, bound from its declaration. Each takes the call
// of its own path and refuses any other.
class Kernel {
 public:
  // Refused unless `implementation` is one this module declares, identity
  // and all: a stale or foreign declaration never selects a kernel.
  static std::expected<Kernel, KernelFailure> Bind(const execution::Implementation& implementation);

  std::expected<void, KernelFailure> Run(LaunchContext& launch, const LinearOperands& operands,
                                         const GemmPlan& plan, std::uint64_t bias) const;
  std::expected<void, KernelFailure> Run(LaunchContext& launch, const LinearOperands& operands,
                                         const GemvPlan& plan, std::uint64_t bias) const;
  std::expected<void, KernelFailure> Run(LaunchContext& launch, ReconGemm& gemm,
                                         const ReconstructedOperands& operands,
                                         std::span<const LtAlgorithm> algorithms) const;
  std::expected<void, KernelFailure> Run(LaunchContext& launch, const MultiLinearOperands& operands,
                                         const MultiGemmPlan& plan) const;
  std::expected<void, KernelFailure> Run(LaunchContext& launch, const BiasOperands& operands) const;

  std::string_view name() const;
  execution::Operation operation() const;

  struct Entry;

 private:
  explicit Kernel(const Entry& entry) : entry_(&entry) {}

  const Entry* entry_;
};

}  // namespace llmp::kernels::exl3

#endif  // LLMP_KERNELS_EXL3_IMPLEMENTATIONS_H_
