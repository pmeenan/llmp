// SPDX-FileCopyrightText: 2025 Turboderp
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include "kernels/exl3/upstream_gemv.h"

#include <algorithm>
#include <optional>

#include "kernels/exl3/validate.h"

namespace llmp::kernels::exl3 {

CcClass UpstreamCcClass(int major, int minor) {
  if (major >= 10) {
    return CcClass::kBlackwell;
  }
  if (major >= 9) {
    return CcClass::kHopper;
  }
  if (major >= 8 && minor >= 9) {
    return CcClass::kAda;
  }
  if (major >= 8) {
    return CcClass::kAmpere;
  }
  return CcClass::kOld;
}

// exl3_gemv_cfg(cc, size_m, size_k, size_n, K, cb, mode = 1,
// narrow_coresident), line for line.
int UpstreamGemvConfig(CcClass cc, int m, int k, int n, int bits, int codebook,
                       int narrow_coresident) {
  if (bits < 2 || bits > 4) {
    return -1;
  }
  if (bits != 4 && codebook == 0) {
    return -1;
  }
  if (m > kGemvMaxRows) {
    return -1;
  }
  if (k % 128 != 0 || n % 128 != 0) {
    return -1;
  }
  if (bits == 2) {
    return n <= 8192 ? 0 : 1;
  }
  if (bits == 3 && cc == CcClass::kAda) {
    return n <= 8192 ? 0 : 1;
  }
  if (n / 32 <= narrow_coresident) {
    return 0;
  }
  if (k <= 2048 && n <= 8192) {
    return 0;
  }
  if (bits == 3) {
    return -1;
  }
  if (n >= 8192 && k <= 4096) {
    return 1;
  }
  if (n >= 8192 && n <= 10240 && k <= 5120 && cc == CcClass::kAmpere) {
    return 1;
  }
  return -1;
}

std::optional<GemvPlan> UpstreamGemvPlan(CcClass cc, int m, int k, int n, int bits, int codebook,
                                         int narrow_coresident, int wide_coresident) {
  // exl3_gemv_try_launch's integer-rate checks, before the heuristic.
  if (bits < 2 || bits > 4 || (bits != 4 && codebook == 0) || m > kGemvMaxRows || k % 128 != 0 ||
      n % 128 != 0) {
    return std::nullopt;
  }
  const int config = UpstreamGemvConfig(cc, m, k, n, bits, codebook, narrow_coresident);
  if (config < 0) {
    return std::nullopt;
  }
  const int blocks =
      std::min(n / GemvColumns(config), config == 0 ? narrow_coresident : wide_coresident);
  if (blocks < 1) {
    return std::nullopt;
  }
  return GemvPlan{.config = config, .blocks = blocks};
}

}  // namespace llmp::kernels::exl3
