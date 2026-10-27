// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_KERNELS_DECODE_SELECTED_WEIGHT_H_
#define LLMP_KERNELS_DECODE_SELECTED_WEIGHT_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <expected>
#include <span>

#include "kernels/common/selected_operation.h"
#include "kernels/common/tensor_ref.h"
#include "kernels/decode/products.h"

namespace llmp::kernels::decode::detail {
using Failure = kernels::KernelFailure;
inline std::unexpected<Failure> RefuseWeight(const char* text) {
  return std::unexpected(Failure{.detail = text});
}
inline bool NativeType(StorageType type, const format::TensorFormat& f) {
  switch (f.encoding) {
    case format::Encoding::kBF16:
      return type == StorageType::kBF16;
    case format::Encoding::kF16:
      return type == StorageType::kF16;
    case format::Encoding::kQ8_0:
      return type == StorageType::kQ8_0;
    case format::Encoding::kNvfp4:
      return type == StorageType::kNvfp4;
    default:
      return false;
  }
}
inline bool NativeType(const TensorRef& t, const format::TensorFormat& f) {
  return t && NativeType(t.facts().type, f);
}
struct WeightGeometry {
  format::TensorFormat format{};
  std::array<TensorRef, format::PartList::kMax> parts{};
  std::array<format::PartRole, format::PartList::kMax> roles{};
  std::array<StorageType, format::PartList::kMax> types{};
  std::array<std::array<std::int64_t, 4>, format::PartList::kMax> ne{};
  std::array<std::array<std::uint64_t, 4>, format::PartList::kMax> nb{};
  std::size_t count = 0;
};
inline std::expected<WeightGeometry, Failure> WeightOf(const SelectedOperand& w) {
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
    TensorRef t = w.tensor;
    if (!native) {
      const auto at = std::ranges::find(w.parts, e.role, &SelectedPart::role);
      if (at == w.parts.end() || std::ranges::count(w.parts, e.role, &SelectedPart::role) != 1)
        return RefuseWeight("selected decoder weight part missing or duplicated");
      t = at->tensor;
    }
    if (!t) return RefuseWeight("selected decoder weight part physical extent differs");
    const auto facts = t.facts();
    if (!facts.contiguous || facts.bytes < e.bytes || facts.bytes > e.readable)
      return RefuseWeight("selected decoder weight part physical extent differs");
    if (native) {
      if (!NativeType(facts.type, f))
        return RefuseWeight("selected decoder native scalar type differs");
      for (std::size_t d = 0; d < 4; ++d)
        if (facts.ne[d] != static_cast<std::int64_t>(f.shape.ne[d]))
          return RefuseWeight("selected decoder native weight dimensions differ");
    } else if (facts.type != StorageType::kI8 &&
               !(e.role == format::PartRole::kTensorScale && facts.type == StorageType::kF32)) {
      return RefuseWeight("selected decoder raw weight part has the wrong scalar type");
    }
    g.parts[i] = t;
    g.roles[i] = e.role;
    g.types[i] = facts.type;
    g.ne[i] = facts.ne;
    g.nb[i] = facts.nb;
  }
  return g;
}
inline std::expected<std::array<DevicePart, format::PartList::kMax>, Failure> BindWeightParts(
    const WeightGeometry& g) {
  std::array<DevicePart, format::PartList::kMax> parts{};
  for (std::size_t i = 0; i < g.count; ++i) {
    if (!g.parts[i])
      return RefuseWeight("selected decoder weight binding or physical geometry changed");
    const auto t = g.parts[i].facts();
    if (t.type != g.types[i] || t.ne != g.ne[i] || t.nb != g.nb[i] || !t.current || !t.extent ||
        t.data == nullptr)
      return RefuseWeight("selected decoder weight binding or physical geometry changed");
    const auto address = AddressOf(t);
    if (t.bytes > UINT64_MAX - address)
      return RefuseWeight("selected decoder weight address overflows");
    parts[i] = {g.roles[i], t.data, t.bytes};
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
