// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/decode/grouped_binding.h"

#include <array>
#include <cstddef>
#include <limits>

#include "kernels/decode/products.h"

namespace llmp::kernels::decode {
namespace {
bool Overlap(std::uint64_t a, std::uint64_t n, std::uint64_t b, std::uint64_t m) {
  return a < b + m && b < a + n;
}
format::TensorFormat MatrixFormat(format::TensorFormat f) {
  f.shape.rank = 2;
  f.shape.ne[2] = f.shape.ne[3] = 1;
  return f;
}
}  // namespace

std::expected<GroupedBindingSize, GroupedBindingError> GroupedBindingBound(
    const format::TensorFormat& f) {
  if (!format::Validate(f) || !DecoderOf(f)) return std::unexpected(GroupedBindingError::kFormat);
  const auto count = format::Matrices(f);
  if (count == 0 || count > 4096) return std::unexpected(GroupedBindingError::kExtent);
  return GroupedBindingSize{.experts = count, .descriptors = base::Bytes(count * sizeof(Operand))};
}

std::expected<void, GroupedBindingError> BindGroupedInto(const format::TensorFormat& f,
                                                         std::span<const PitchedExpertPart> parts,
                                                         std::span<Operand> into) {
  const auto bound = GroupedBindingBound(f);
  if (!bound) return std::unexpected(bound.error());
  const auto matrix = MatrixFormat(f);
  const auto extents = format::PartsOf(matrix);
  if (!extents) return std::unexpected(GroupedBindingError::kFormat);
  if (parts.size() != extents->size() || into.size() != bound->experts)
    return std::unexpected(GroupedBindingError::kExtent);
  constexpr auto max = std::numeric_limits<std::uintptr_t>::max();
  const auto dest = reinterpret_cast<std::uintptr_t>(into.data());
  const auto source = reinterpret_cast<std::uintptr_t>(parts.data());
  if (dest == 0 || source == 0 || dest % alignof(Operand) != 0 ||
      source % alignof(PitchedExpertPart) != 0 || into.size_bytes() > max - dest ||
      parts.size_bytes() > max - source)
    return std::unexpected(GroupedBindingError::kAddress);
  if (Overlap(dest, into.size_bytes(), source, parts.size_bytes()))
    return std::unexpected(GroupedBindingError::kAlias);
  std::array<const PitchedExpertPart*, format::PartList::kMax> ordered{};
  for (std::size_t i = 0; i != extents->size(); ++i) {
    const auto& extent = (*extents)[i];
    for (const auto& part : parts) {
      if (part.role != extent.role) continue;
      if (ordered[i] != nullptr) return std::unexpected(GroupedBindingError::kParts);
      ordered[i] = &part;
    }
    const auto* part = ordered[i];
    if (part == nullptr) return std::unexpected(GroupedBindingError::kParts);
    if (part->address == 0 || part->address > max || part->bytes > max - part->address ||
        part->matrix_pitch < extent.readable || part->bytes < extent.readable ||
        (bound->experts - 1) > (part->bytes - extent.readable) / part->matrix_pitch)
      return std::unexpected(GroupedBindingError::kAddress);
    if (Overlap(dest, into.size_bytes(), part->address, part->bytes))
      return std::unexpected(GroupedBindingError::kAlias);
    for (std::size_t j = 0; j != i; ++j) {
      // Interleaved part views can share a bounding span without sharing any
      // consumed bytes. Each list is sorted and internally disjoint by pitch.
      std::uint64_t a = 0, b = 0;
      const auto& other = *ordered[j];
      while (a != bound->experts && b != bound->experts) {
        const auto x = part->address + a * part->matrix_pitch;
        const auto y = other.address + b * other.matrix_pitch;
        if (Overlap(x, extent.readable, y, (*extents)[j].readable))
          return std::unexpected(GroupedBindingError::kAlias);
        if (x + extent.readable <= y)
          ++a;
        else
          ++b;
      }
    }
  }
  const auto operand = [&](std::uint64_t e) {
    std::array<DevicePart, format::PartList::kMax> local{};
    for (std::size_t i = 0; i != extents->size(); ++i) {
      const auto& part = *ordered[i];
      local[i] = {.role = part.role,
                  .address = reinterpret_cast<const void*>(part.address + e * part.matrix_pitch),
                  .bytes = (*extents)[i].readable};
    }
    return OperandOf({matrix, std::span(local).first(extents->size()), 0});
  };
  // Validation pass before writes: failures leave every old descriptor intact.
  for (std::uint64_t e = 0; e != bound->experts; ++e)
    if (!operand(e)) return std::unexpected(GroupedBindingError::kAddress);
  for (std::size_t e = 0; e != into.size(); ++e) into[e] = *operand(e);
  return {};
}
}  // namespace llmp::kernels::decode
