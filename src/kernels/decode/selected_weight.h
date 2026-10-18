// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_KERNELS_DECODE_SELECTED_WEIGHT_H_
#define LLMP_KERNELS_DECODE_SELECTED_WEIGHT_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <span>

#include "kernels/decode/products.h"
#include "kernels/ggml/selected_operation.h"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::decode::detail {
using Failure = ggml::KernelFailure;
inline std::unexpected<Failure> RefuseWeight(const char* text) {
  return std::unexpected(Failure{.detail = text});
}
inline bool NativeType(const ggml_tensor* t, const format::TensorFormat& f) {
  if (!t) return false;
  switch (f.encoding) {
    case format::Encoding::kBF16:
      return t->type == GGML_TYPE_BF16;
    case format::Encoding::kF16:
      return t->type == GGML_TYPE_F16;
    case format::Encoding::kQ8_0:
      return t->type == GGML_TYPE_Q8_0;
    case format::Encoding::kNvfp4:
      return t->type == GGML_TYPE_NVFP4;
    default:
      return false;
  }
}
struct WeightGeometry {
  format::TensorFormat format{};
  std::array<ggml_tensor*, format::PartList::kMax> parts{};
  std::array<format::PartRole, format::PartList::kMax> roles{};
  std::array<ggml_type, format::PartList::kMax> types{};
  std::array<std::array<std::int64_t, 4>, format::PartList::kMax> ne{};
  std::array<std::array<std::size_t, 4>, format::PartList::kMax> nb{};
  std::size_t count = 0;
};
inline std::expected<WeightGeometry, Failure> WeightOf(const ggml::SelectedOperand& w) {
  const auto& f = w.desc.format;
  const auto decoder = DecoderOf(f);
  const auto extents = format::PartsOf(f);
  if (!decoder || !extents || f.shape.ne[0] >= (1ULL << 30) || f.shape.ne[1] >= (1ULL << 31) ||
      f.shape.ne[0] % DecoderKMultiple(*decoder))
    return RefuseWeight("selected decoder weight format or index range is unsupported");
  const bool native = w.parts.empty();
  if ((native && (extents->size() != 1 || (*extents)[0].role != format::PartRole::kData ||
                  !ir::IsContiguous(w.desc))) ||
      (!native && w.parts.size() != extents->size()))
    return RefuseWeight("selected decoder needs every physical weight part exactly once");
  WeightGeometry g;
  g.format = f;
  g.count = extents->size();
  for (std::size_t i = 0; i < g.count; ++i) {
    const auto& e = (*extents)[i];
    ggml_tensor* t = w.tensor;
    if (!native) {
      const auto at = std::ranges::find(w.parts, e.role, &ggml::SelectedPart::role);
      if (at == w.parts.end() ||
          std::ranges::count(w.parts, e.role, &ggml::SelectedPart::role) != 1)
        return RefuseWeight("selected decoder weight part missing or duplicated");
      t = at->tensor;
    }
    if (!t || !ggml_is_contiguous(t) || ggml_nbytes(t) < e.bytes || ggml_nbytes(t) > e.readable)
      return RefuseWeight("selected decoder weight part physical extent differs");
    if (native) {
      if (!NativeType(t, f)) return RefuseWeight("selected decoder native scalar type differs");
      for (std::size_t d = 0; d < 4; ++d)
        if (t->ne[d] != static_cast<std::int64_t>(f.shape.ne[d]))
          return RefuseWeight("selected decoder native weight dimensions differ");
    } else if (t->type != GGML_TYPE_I8 &&
               !(e.role == format::PartRole::kTensorScale && t->type == GGML_TYPE_F32)) {
      return RefuseWeight("selected decoder raw weight part has the wrong scalar type");
    }
    g.parts[i] = t;
    g.roles[i] = e.role;
    g.types[i] = t->type;
    std::ranges::copy(t->ne, g.ne[i].begin());
    std::ranges::copy(t->nb, g.nb[i].begin());
  }
  return g;
}
inline std::expected<std::array<DevicePart, format::PartList::kMax>, Failure> BindWeightParts(
    const WeightGeometry& g) {
  std::array<DevicePart, format::PartList::kMax> parts{};
  for (std::size_t i = 0; i < g.count; ++i) {
    const auto* t = g.parts[i];
    if (!t || t->type != g.types[i] || !std::ranges::equal(t->ne, g.ne[i]) ||
        !std::ranges::equal(t->nb, g.nb[i]) || !ggml::detail::Current(t) ||
        !ggml::detail::Extent(t) || t->data == nullptr)
      return RefuseWeight("selected decoder weight binding or physical geometry changed");
    const auto address = reinterpret_cast<std::uintptr_t>(t->data);
    if (ggml_nbytes(t) > UINT64_MAX - address)
      return RefuseWeight("selected decoder weight address overflows");
    parts[i] = {g.roles[i], t->data, ggml_nbytes(t)};
  }
  // Every matrix is checked before any kernel can be queued.
  for (std::uint64_t m = 0; m < format::Matrices(g.format); ++m) {
    if (auto operand = OperandOf({g.format, std::span(parts).first(g.count), m}); !operand)
      return std::unexpected(Failure{.detail = operand.error()});
  }
  return parts;
}
}  // namespace llmp::kernels::decode::detail
#endif
