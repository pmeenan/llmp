// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// One EXL3 linear, through each of the paths upstream's LinearEXL3 takes
// (exllamav3/modules/quant/exl3.py and libtorch/linear.cpp at 6b84a21b;
// docs/backend-proof.md, "Native EXL3 operation plan"), CUDA builds only.
// Each path queues the kernels upstream queues, in its order, on one launch
// context (launch.h), with the plan's fixed launch configuration:
//
//   packed, GEMM   exl3_gemm_kernel, then the bias (add_kernel_hhh) if any;
//   packed, GEMV   exl3_gemv_kernel, then the bias;
//   multi          exl3_mgemm_kernel for gate and up (no bias: upstream
//                  fuses them only without one);
//   reconstructed  the input transform (had_hf_r_128_kernel<true, false>
//                  into xh), then per slice of at most 32,768 columns the
//                  rotated weights (reconstruct_kernel into w) and the
//                  pinned GEMM of xh and w into the output's columns, then
//                  the output transform in place, then the bias;
//   fused          per slice the weights with both transforms applied
//                  (reconstruct_had_kernel into w) and the pinned GEMM of x
//                  and w, then the bias.
//
// Upstream adds the bias on the reconstruction paths with PyTorch's
// elementwise add; the plan uses add_kernel_hhh on every path, which gives
// the same bits (docs/backend-proof.md, "Bias add"). A bias needs an F16
// output. Upstream's thresholds decide between the paths
// (validate.h's UpstreamPath); the scratch (a_had, xh, w) is the caller's, declared and
// charged like every operand, and stays so until a fence after the last
// launch has completed.

#ifndef LLMP_KERNELS_EXL3_LINEAR_H_
#define LLMP_KERNELS_EXL3_LINEAR_H_

#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "kernels/exl3/launch.h"
#include "kernels/exl3/recon_gemm.h"
#include "kernels/exl3/validate.h"

namespace llmp::kernels::exl3 {

std::expected<void, KernelFailure> PackedGemmLinear(LaunchContext& launch,
                                                    const LinearOperands& operands,
                                                    const GemmPlan& plan, std::uint64_t bias);
std::expected<void, KernelFailure> PackedGemvLinear(LaunchContext& launch,
                                                    const LinearOperands& operands,
                                                    const GemvPlan& plan, std::uint64_t bias);
std::expected<void, KernelFailure> MultiLinear(LaunchContext& launch,
                                               const MultiLinearOperands& operands,
                                               const MultiGemmPlan& plan);

// `algorithms` holds one pinned algorithm per slice (ReconstructSlices);
// the operands are validate.h's ReconstructedOperands.
std::expected<void, KernelFailure> ReconstructedLinear(LaunchContext& launch, ReconGemm& gemm,
                                                       const ReconstructedOperands& operands,
                                                       bool fused,
                                                       std::span<const LtAlgorithm> algorithms);

}  // namespace llmp::kernels::exl3

#endif  // LLMP_KERNELS_EXL3_LINEAR_H_
