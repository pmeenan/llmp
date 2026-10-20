// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include "components/rotation_reference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>

namespace llmp::components::defs {
namespace {

std::array<float, 2> YarnCorrDims(const ir::RopeParams& p) {
  const auto corr = [&](float beta) {
    return static_cast<float>(p.dims) *
           std::log(static_cast<float>(p.original_context) /
                    (beta * 2.0F * std::numbers::pi_v<float>)) /
           (2.0F * std::log(p.base));
  };
  const float lo = std::max(0.0F, std::floor(corr(p.beta_fast)));
  const float hi = std::min(static_cast<float>(p.dims - 1), std::ceil(corr(p.beta_slow)));
  return {lo, hi};
}

std::size_t SectionForPair(const ir::RopeParams& p, std::uint64_t pair, double& exponent) {
  exponent = static_cast<double>(pair);
  const auto& sec = p.sections;
  const std::uint64_t sect_dims = std::uint64_t{sec[0]} + sec[1] + sec[2] + sec[3];
  if (p.mode == ir::RopeMode::kInterleavedMulti && sect_dims > 0) {
    const std::uint64_t sector = (2 * pair) % sect_dims;
    if (sector < 4ULL * sec[1] && sector % 4 == 1) return 1;
    if (sector < 4ULL * sec[2] && sector % 4 == 2) return 2;
    if (sector < 4ULL * sec[3] && sector % 4 == 3) return 3;
    return 0;
  }
  if (p.mode == ir::RopeMode::kMulti && sect_dims > 0) {
    const std::uint64_t sector = pair % sect_dims;
    if (sector < sec[0]) return 0;
    if (sector < std::uint64_t{sec[0]} + sec[1]) return 1;
    if (sector < std::uint64_t{sec[0]} + sec[1] + sec[2]) return 2;
    return 3;
  }
  if (p.mode == ir::RopeMode::kVision && (sec[0] + sec[1]) > 0) {
    const std::uint64_t sector = pair % (std::uint64_t{sec[0]} + sec[1]);
    exponent = static_cast<double>(sector < sec[0] ? sector : sector - sec[0]);
    return sector < sec[0] ? 0 : 1;
  }
  return 0;
}

}  // namespace

void RotateHeads(std::span<double> row, std::uint64_t heads, std::uint64_t head_dim,
                 const ir::RopeParams& params, std::array<std::int64_t, 4> positions,
                 std::span<const float> freqs) {
  const std::uint64_t pairs =
      params.mode == ir::RopeMode::kVision ? params.dims : (params.dims / 2);
  if (pairs == 0) return;
  const float theta_scale =
      static_cast<float>(std::pow(static_cast<double>(params.base),
                                  static_cast<double>(-2.0F / static_cast<float>(params.dims))));
  const std::array<float, 2> corr =
      params.ext_factor != 0.0F ? YarnCorrDims(params) : std::array<float, 2>{0.0F, 0.0F};
  for (std::uint64_t h = 0; h < heads; ++h) {
    double* x = &row[h * head_dim];
    for (std::uint64_t i = 0; i < pairs; ++i) {
      double exponent = 0.0;
      const std::size_t section = SectionForPair(params, i, exponent);
      const auto pos = static_cast<float>(positions.at(section));
      const float power = static_cast<float>(std::pow(static_cast<double>(theta_scale), exponent));
      float extrap = pos * power;
      if (!freqs.empty()) {
        extrap /= freqs[static_cast<std::size_t>(i)];
      }
      const float interp = params.scale * extrap;
      float theta = interp;
      float mscale = params.attn_factor;
      if (params.ext_factor != 0.0F) {
        const float y = (static_cast<float>(i) - corr[0]) / std::max(0.001F, corr[1] - corr[0]);
        const float ramp = (1.0F - std::min(1.0F, std::max(0.0F, y))) * params.ext_factor;
        const float kept = interp * (1.0F - ramp);
        const float mixed = extrap * ramp;
        theta = kept + mixed;
        const float magnitude = 0.1F * std::log(1.0F / params.scale);
        mscale *= 1.0F + magnitude;
      }
      const double c = std::cos(static_cast<double>(theta)) * static_cast<double>(mscale);
      const double s = (params.inverse ? -std::sin(static_cast<double>(theta))
                                       : std::sin(static_cast<double>(theta))) *
                       static_cast<double>(mscale);
      std::uint64_t idx0 = 0;
      std::uint64_t idx1 = 0;
      if (params.mode == ir::RopeMode::kNormal) {
        idx0 = params.offset + (2 * i);
        idx1 = idx0 + 1;
      } else if (params.mode == ir::RopeMode::kVision) {
        idx0 = i;
        idx1 = i + params.dims;
      } else {
        idx0 = params.offset + i;
        idx1 = idx0 + (params.dims / 2);
      }
      const double x0 = x[idx0];
      const double x1 = x[idx1];
      x[idx0] = (x0 * c) - (x1 * s);
      x[idx1] = (x0 * s) + (x1 * c);
    }
  }
}

}  // namespace llmp::components::defs
