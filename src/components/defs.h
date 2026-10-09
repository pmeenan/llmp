// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Internal to src/components: the catalog's definitions, and helpers their
// fragments and scalar references share. The references' arithmetic is
// written here from the op set's definitions (ir/op.h), not taken from the
// reference executor, so a component's reference is an oracle of its own.

#ifndef LLMP_COMPONENTS_DEFS_H_
#define LLMP_COMPONENTS_DEFS_H_

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <initializer_list>
#include <numbers>
#include <span>
#include <string>
#include <vector>

#include "components/component.h"
#include "format/encoding.h"
#include "format/tensor_format.h"
#include "ir/graph.h"

namespace llmp::components::defs {

const ComponentDef& EmbeddingToken();
const ComponentDef& NormRms();
const ComponentDef& AttentionDense();
const ComponentDef& FfnGlu();
const ComponentDef& HeadLinear();

// The encodings a norm's weight takes: the converters keep norms F32.
inline constexpr std::array<format::Encoding, 1> kF32Only{format::Encoding::kF32};

// A role's shape.
inline RoleShape Role(std::uint16_t role, std::initializer_list<std::uint64_t> ne) {
  RoleShape r{.role = role};
  std::size_t i = 0;
  for (const std::uint64_t n : ne) r.ne.at(i++) = n;
  r.rank = static_cast<std::uint8_t>(ne.size());
  return r;
}

// GGML's view strides: nb[0] the element, then the given row and matrix
// strides, each later one the previous times its extent (ggml_view_*).
inline std::array<std::uint64_t, 4> ViewStrides(std::uint64_t element, std::uint64_t nb1,
                                                std::uint64_t nb2, const format::Shape& shape) {
  std::array<std::uint64_t, 4> nb{element, nb1, nb2, 0};
  if (shape.rank < 3) nb[2] = nb[1] * shape.ne[1];
  nb[3] = nb[2] * shape.ne[2];
  return nb;
}

inline format::Shape Shape(std::initializer_list<std::uint64_t> ne) { return ir::ShapeOf(ne); }

// ------------------------------------------------------------ references

inline float F32(double x) { return static_cast<float>(x); }

// An F16's value nearest to `x` (ties to even; overflow to infinity).
inline double RoundF16(double x) {
  const float f = static_cast<float>(x);
  std::uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  const std::uint32_t sign = (u >> 16U) & 0x8000U;
  std::uint32_t mag = u & 0x7FFFFFFFU;
  std::uint16_t h = 0;
  if (mag >= 0x7F800000U) {
    h = static_cast<std::uint16_t>(sign | 0x7C00U | (mag > 0x7F800000U ? 0x200U : 0U));
  } else if (mag >= 0x477FF000U) {
    h = static_cast<std::uint16_t>(sign | 0x7C00U);  // rounds past 65504
  } else if (mag < 0x38800000U) {
    // A subnormal F16: value / 2^-24, rounded to nearest even.
    const double v = std::fabs(static_cast<double>(f)) * 16777216.0;
    const double r = std::nearbyint(v);
    h = static_cast<std::uint16_t>(sign | static_cast<std::uint32_t>(r));
  } else {
    const std::uint32_t e = ((mag >> 23U) - 112U) << 10U;
    const std::uint32_t m = (mag >> 13U) & 0x3FFU;
    std::uint32_t half = e | m;
    const std::uint32_t rest = mag & 0x1FFFU;
    if (rest > 0x1000U || (rest == 0x1000U && (half & 1U) != 0U)) ++half;
    h = static_cast<std::uint16_t>(sign | half);
  }
  // Back to a value.
  const std::uint32_t s = (h & 0x8000U) != 0U ? 1U : 0U;
  const std::uint32_t exp = (h >> 10U) & 0x1FU;
  const std::uint32_t man = h & 0x3FFU;
  double v = 0.0;
  if (exp == 0) {
    v = std::ldexp(static_cast<double>(man), -24);
  } else if (exp == 31) {
    v = man == 0 ? INFINITY : NAN;
  } else {
    v = std::ldexp(static_cast<double>(man | 0x400U), static_cast<int>(exp) - 25);
  }
  return s != 0 ? -v : v;
}

// A value stored in a state encoding and read back.
inline double Stored(format::Encoding e, double x) {
  if (e == format::Encoding::kF16) return RoundF16(x);
  if (e == format::Encoding::kF32) return static_cast<double>(static_cast<float>(x));
  return x;
}

// rms_norm(x) · w over each row of `n` values (w empty: unweighted).
inline void RmsRows(std::span<double> x, std::size_t n, double eps, std::span<const float> w) {
  for (std::size_t r = 0; r + n <= x.size(); r += n) {
    double ss = 0.0;
    for (std::size_t i = 0; i < n; ++i) ss += x[r + i] * x[r + i];
    const double inv = 1.0 / std::sqrt((ss / static_cast<double>(n)) + eps);
    for (std::size_t i = 0; i < n; ++i) {
      x[r + i] *= inv;
      if (!w.empty()) x[r + i] *= static_cast<double>(w[i % w.size()]);
    }
  }
}

// out[rows][n] = Σ_k w[n][k] · x[rows][k]: a [k, n] weight times [k, rows].
inline std::vector<double> Product(std::span<const float> w, std::size_t k, std::size_t n,
                                   std::span<const double> x) {
  const std::size_t rows = x.size() / k;
  std::vector<double> out(rows * n);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t i = 0; i < k; ++i) s += static_cast<double>(w[(j * k) + i]) * x[(r * k) + i];
      out[(r * n) + j] = s;
    }
  }
  return out;
}

inline double GeluTanh(double x) {
  const double c = std::sqrt(2.0 / std::numbers::pi);
  return 0.5 * x * (1.0 + std::tanh(c * (x + (0.044715 * x * x * x))));
}
inline double Silu(double x) { return x / (1.0 + std::exp(-x)); }

// The rotation angle of pair `pair` at `position` as op.h defines it (each
// step rounded to F32), for an unscaled-extrapolation rotation.
inline float RopeAngle(float base, std::uint32_t dims, float scale, std::uint64_t pair,
                       std::int64_t position) {
  const float theta_scale = static_cast<float>(
      std::pow(static_cast<double>(base), static_cast<double>(-2.0F / static_cast<float>(dims))));
  const float power =
      static_cast<float>(std::pow(static_cast<double>(theta_scale), static_cast<double>(pair)));
  const float extrap = static_cast<float>(position) * power;
  return scale * extrap;
}

}  // namespace llmp::components::defs

#endif  // LLMP_COMPONENTS_DEFS_H_
