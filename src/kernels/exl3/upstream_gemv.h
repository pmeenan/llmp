// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Upstream's choice of the GEMV kernel, as a llmpalooza function (D-053, D-080;
// docs/backend-proof.md, "EXL3-G vs EXL3-O"): a recorded copy of
// exl3_gemv_cfg and the eligibility and grid rules of exl3_gemv_try_launch
// in ExLlamaV3's quant/exl3_gemv.cu at 6b84a21b, which llmpalooza does not
// compile (its ATen entry point and device context are upstream's). A plan
// asks it where upstream's EXL3-O profile would launch the GEMV, and with
// which configuration and grid; the launcher itself takes any valid plan
// (validate.h). What differs from upstream:
//
// - Only its default heuristic (EXL3_GEMV unset or 1) and shuffle
//   extraction (EXL3_GEMV_SMEM unset); no environment is read.
// - Only integer rates and full-length side vectors, as validate.h admits;
//   half-integer rates, which need the mul1 codebook, are not compiled.
// - The co-resident block counts come from the caller (the device's
//   occupancy times its SMs), as upstream computes and caches them.

#ifndef LLMP_KERNELS_EXL3_UPSTREAM_GEMV_H_
#define LLMP_KERNELS_EXL3_UPSTREAM_GEMV_H_

#include <optional>

#include "kernels/exl3/validate.h"

namespace llmp::kernels::exl3 {

// Upstream's compute-capability classes (quant/exl3_devctx.cuh) and how
// DevCtx::get_cc assigns them from a device's major and minor version.
enum class CcClass : int { kOld = 1, kAmpere = 2, kAda = 3, kHopper = 4, kBlackwell = 5 };
CcClass UpstreamCcClass(int major, int minor);

// The codebooks, as upstream numbers them: 0 3INST, 1 mcg, 2 mul1.
inline constexpr int kCodebookMcg = 1;

// exl3_gemv_cfg in mode 1: -1 where upstream keeps the GEMM kernel, else
// the configuration (0 narrow, 1 wide). `narrow_coresident` is how many
// blocks of the narrow kernel fit at once.
int UpstreamGemvConfig(CcClass cc, int m, int k, int n, int bits, int codebook,
                       int narrow_coresident);

// The GEMV launch upstream's exl3_gemm makes for an unforced call
// (force_shape_idx and force_num_sms both unset) with suh, a_had and svh,
// or nothing where it launches the GEMM kernel instead: the configuration,
// and the grid min(n / columns, co-resident blocks of that kernel).
std::optional<GemvPlan> UpstreamGemvPlan(CcClass cc, int m, int k, int n, int bits, int codebook,
                                         int narrow_coresident, int wide_coresident);

}  // namespace llmp::kernels::exl3

#endif  // LLMP_KERNELS_EXL3_UPSTREAM_GEMV_H_
