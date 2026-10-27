// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Internal to src/components: the catalog's definitions, and helpers their
// fragments and scalar references share. The references' arithmetic is
// written here from the op set's definitions (ir/op.h), not taken from the
// reference executor, so a component's reference is an oracle of its own.

#ifndef LLMP_COMPONENTS_DEFS_H_
#define LLMP_COMPONENTS_DEFS_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <initializer_list>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "components/component.h"
#include "format/encoding.h"
#include "format/tensor_format.h"
#include "ir/builder.h"
#include "ir/graph.h"

namespace llmp::components::defs {

const ComponentDef& SliceSpatial();
const ComponentDef& ActivationElementwise();
const ComponentDef& ReshapeSpatial();
const ComponentDef& AffineChannel();
const ComponentDef& AffineScalar();
const ComponentDef& ResidualScaled();
const ComponentDef& ConditionRows();
const ComponentDef& ConcatSpatial();
const ComponentDef& Conv2d();
const ComponentDef& ProjectionSpatial();
const ComponentDef& UpsampleNearest();
const ComponentDef& ResidualSpatial();
const ComponentDef& ResidualUpsample();
const ComponentDef& AttentionSpatial();
const ComponentDef& EmbeddingToken();
const ComponentDef& EmbeddingProjected();
const ComponentDef& NormAdaptive();
const ComponentDef& NormChannel();
const ComponentDef& ModulateScale();
const ComponentDef& NormRms();
const ComponentDef& ScaleLearned();
const ComponentDef& AttentionDense();
const ComponentDef& AttentionBorrowed();
const ComponentDef& FfnGlu();
const ComponentDef& FfnMoe();
const ComponentDef& HeadLinear();
const ComponentDef& ResidualGated();
const ComponentDef& ResidualHc();
const ComponentDef& HcMix();
const ComponentDef& HcCombine();
const ComponentDef& AttentionGdn();
const ComponentDef& AttentionQsa();
const ComponentDef& SideNgram();
const ComponentDef& ResidualPle();
const ComponentDef& EmbeddingMtp();
const ComponentDef& AttentionMqa();
const ComponentDef& SideHash();
const ComponentDef& EmbeddingInject();

// Source encodings admitted by the individual parameter-role contracts.
inline constexpr std::array<format::Encoding, 1> kF32Only{format::Encoding::kF32};
inline constexpr std::array kNormEncodings{format::Encoding::kF32, format::Encoding::kBF16,
                                           format::Encoding::kF16};
inline constexpr std::array kSpatialEncodings{format::Encoding::kF32, format::Encoding::kBF16};
inline constexpr std::array kBiasEncodings{format::Encoding::kF32, format::Encoding::kF16};

// Norm scales keep their source encoding resident. Widen a low-precision
// scale in transient graph storage before the F32 normalization product.
inline ir::TensorId NormScale(ir::Builder& b, ir::TensorId weight) {
  return b.tensor(weight).format.encoding == format::Encoding::kF32
             ? weight
             : b.Convert(weight, format::Encoding::kF32);
}

// Explicit source convention for the norm roles a component declares. A
// recorded fold already changed those bytes; applying the offset again is invalid.
inline constexpr ParamSpec kNormZeroCentered{.name = "norm_zero_centered",
                                             .type = ParamType::kBool,
                                             .required = false,
                                             .least = 0,
                                             .most = 1,
                                             .fallback = 0};
inline bool NormOffsetContradictsFold(const Instance& in) {
  return in.params.Flag("norm_zero_centered") &&
         std::ranges::find(in.stage.folds, "norm.plus_one") != in.stage.folds.end();
}
// The roles NormTransform and a logarithmic A derive (WeightRole::derived).
inline constexpr DerivedValue kZeroCenteredNorm{.flag = "norm_zero_centered",
                                                .fold = "norm.plus_one"};
inline constexpr DerivedValue kLogDecay{.flag = "a_log", .fold = "ssm.a_neg_exp"};
enum class ParameterTransform : std::uint8_t { kNone, kNegativeExp, kPlusOne };
inline ParameterTransform NormTransform(const Instance& in) {
  return in.params.Flag("norm_zero_centered") ? ParameterTransform::kPlusOne
                                              : ParameterTransform::kNone;
}

// Tiny parameter roles remain immutable source-encoded weight leaves. Cache
// only their transient graph results, lazily at the original Weight call site:
// eager leaf creation would change existing F32 graph identities. Joined casts
// are shared across owners, then the caller's emission context is restored.
// Callers validate F32/BF16 role encodings before emitting the fragment.
// The widened operand is shared, but each transform has a distinct cache
// entry: a raw lookup cannot return a previously offset or exponentiated value.
template <std::size_t Roles>
class F32Parameters {
 public:
  explicit F32Parameters(Fragment& fragment) : f_(fragment) {}
  ir::TensorId Get(std::uint16_t role, bool negative_exp = false) {
    return Transformed(role,
                       negative_exp ? ParameterTransform::kNegativeExp : ParameterTransform::kNone);
  }
  ir::TensorId Norm(std::uint16_t role) { return Transformed(role, NormTransform(f_.instance)); }

 private:
  ir::TensorId Transformed(std::uint16_t role, ParameterTransform transform) {
    const ir::TensorId weight = f_.Weight(role);
    if (f_.b.tensor(weight).format.encoding == format::Encoding::kF32 &&
        transform == ParameterTransform::kNone)
      return weight;
    const auto index = static_cast<std::size_t>(transform);
    if (results_.at(role).at(index).valid()) return results_[role][index];
    const auto segment = f_.b.segment();
    f_.b.SetJoined();
    auto& plain = results_[role][static_cast<std::size_t>(ParameterTransform::kNone)];
    if (!plain.valid()) plain = NormScale(f_.b, weight);
    ir::TensorId value = plain;
    if (transform == ParameterTransform::kNegativeExp)
      value = f_.b.Unary(f_.b.Unary(value, ir::UnaryFn::kExp), ir::UnaryFn::kNeg);
    if (transform == ParameterTransform::kPlusOne) value = f_.b.Scale(value, 1.0F, 1.0F);
    if (segment == ir::kJoinedSegment)
      f_.b.SetJoined();
    else
      f_.b.SetOwner(segment);
    results_[role][index] = value;
    return value;
  }

  Fragment& f_;
  std::array<std::array<ir::TensorId, 3>, Roles> results_{};
};

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

// A tensor's description (a copy, as Builder::tensor gives), or an empty
// one where the builder refused an earlier node (its ids are then
// invalid): the step fails with the builder's error at Finish rather than
// here.
inline ir::TensorDesc Desc(const ir::Builder& b, ir::TensorId t) {
  return t.valid() && b.ok() ? b.tensor(t) : ir::TensorDesc{};
}

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

}  // namespace llmp::components::defs

#endif  // LLMP_COMPONENTS_DEFS_H_
